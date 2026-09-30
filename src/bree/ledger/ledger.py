"""The BREE ledger: per-person baskets, payment matching, and theft reconciliation.

How it works, in one screen:

1. Every tracked person gets a `PersonRecord`. PICK adds an item to their basket,
   PUT_BACK removes one, CONCEAL marks an item as hidden (or adds one if the pick
   was missed), PAY (register dwell) records a register visit, EXIT starts
   reconciliation.

2. External payments (POS receipts, card/RFID taps at a cooler) arrive with a
   terminal name and a timestamp but no person. We attribute each one to the
   person who was standing at that terminal's zone at that time (register), or
   who picked from that cooler around that time (cooler tap).

3. After a person exits we wait a short grace period (late POS messages), and
   longer if their decision depends on someone still in the store (a group
   that came in together, or a crowded pick that might have been someone else's).

4. Reconciliation = basket minus paid items, matched by SKU when we have it and
   by category otherwise. Every unpaid item gets an evidence score from the
   table in `LedgerConfig`. Scores are combined with noisy-OR into one
   confidence. Above `alert_threshold` -> "alert"; above `review_threshold` ->
   "review" (a manager looks later, nobody gets confronted); else dropped.

Design rule: one uncorroborated pick must never be enough for an "alert".
A missed put-back looks exactly like that, and false accusations kill a pilot.
"""
from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from bree.alerts.types import Alert, UnpaidItem
from bree.events.types import Catalog, Event, EventType, LineItem, Payment


@dataclass
class LedgerConfig:
    # "pos": payments carry item lists (POS / tap integration). Partial pay is detectable.
    # "dwell": no POS feed. A register visit counts as paying for everything picked before it.
    payment_mode: str = "pos"

    # Evidence weights for one unpaid item (summed, clipped to [0, 1]).
    w_pick: float = 0.5              # x pick confidence. Alone this can never reach alert_threshold.
    w_conceal: float = 0.35          # item was concealed (pocket / bag / waistband)
    w_held_at_exit: float = 0.2      # item visibly in hand when the person walked out
    w_no_register: float = 0.15      # person never went to the register
    ambiguous_factor: float = 0.5    # crowded pick: someone else may have taken it

    alert_threshold: float = 0.7
    review_threshold: float = 0.4

    exit_grace_s: float = 5.0        # wait for late POS messages after exit
    max_hold_s: float = 120.0        # longest we hold a decision for a group mate / crowd candidate
    group_window_s: float = 4.0      # people entering within this window may be one party
    cooler_tap_window_s: float = 20.0  # a tap is matched to a pick at that cooler within this window
    register_slack_s: float = 3.0    # POS timestamp may fall just outside the dwell interval
    payment_expiry_s: float = 300.0  # unmatched payments are dropped (and logged) after this


@dataclass
class BasketItem:
    category: str
    sku: str | None
    t_pick: float
    zone: str | None
    confidence: float
    ambiguous_with: list[int] = field(default_factory=list)
    concealed: bool = False
    held_at_exit: bool = False
    source: str = "pick"             # "pick" or "conceal" (pick missed, inferred from concealment)
    dwell_paid: bool = False         # dwell mode only: covered by a register visit


@dataclass
class RegisterVisit:
    t_start: float
    t_end: float | None              # None while the person is still standing there
    zone: str
    n_payments: int = 0

    def contains(self, t: float, slack: float = 0.0) -> bool:
        end = float("inf") if self.t_end is None else self.t_end
        return self.t_start - slack <= t <= end + slack


@dataclass
class PersonRecord:
    person_id: int
    t_enter: float
    basket: list[BasketItem] = field(default_factory=list)
    paid: list[LineItem] = field(default_factory=list)
    register_visits: list[RegisterVisit] = field(default_factory=list)
    t_exit: float | None = None
    reconciled: bool = False
    unpaid: list[BasketItem] = field(default_factory=list)    # filled at reconciliation
    surplus: Counter = field(default_factory=Counter)         # paid-but-not-seen, by category
    group: list[int] = field(default_factory=list)
    log: list[str] = field(default_factory=list)              # human-readable audit trail


class Ledger:
    def __init__(self, catalog: Catalog, config: LedgerConfig | None = None,
                 terminal_zones: dict[str, str] | None = None,
                 zone_kinds: dict[str, str] | None = None):
        self.catalog = catalog
        self.cfg = config or LedgerConfig()
        self.terminal_zones = terminal_zones or {}   # terminal name -> zone name
        self.zone_kinds = zone_kinds or {}           # zone name -> shelf/cooler/register/exit
        self.people: dict[int, PersonRecord] = {}
        self.unassigned_payments: list[Payment] = []
        self.orphan_payments: list[Payment] = []
        self.alerts: list[Alert] = []
        self.dropped: list[dict] = []                # below review threshold (kept for eval/debug)
        self.now: float = 0.0
        self.pending: set[int] = set()               # exited, not yet reconciled
        self._alert_ids = itertools.count(1)

    # ------------------------------------------------------------------ input

    def on_event(self, ev: Event) -> list[Alert]:
        """Feed one event. Returns any alerts that became final at this time."""
        emitted = self.tick(ev.t)
        p = self._person(ev.person_id, ev.t)

        if ev.type == EventType.ENTER:
            p.t_enter = min(p.t_enter, ev.t)
        elif ev.type == EventType.PICK:
            self._on_pick(p, ev)
        elif ev.type == EventType.PUT_BACK:
            self._on_put_back(p, ev)
        elif ev.type == EventType.CONCEAL:
            self._on_conceal(p, ev)
        elif ev.type == EventType.PAY:
            self._on_register_visit(p, ev)
        elif ev.type == EventType.EXIT:
            p.t_exit = ev.t
            self.pending.add(p.person_id)
            held = Counter(ev.meta.get("held_items", []))
            for item in reversed(p.basket):
                if held[item.category] > 0:
                    item.held_at_exit = True
                    held[item.category] -= 1
            p.log.append(f"{ev.t:7.1f}s exit (held: {ev.meta.get('held_items', [])})")

        self._match_payments()
        return emitted + self.tick(ev.t)

    def on_payment(self, pay: Payment) -> list[Alert]:
        emitted = self.tick(pay.t)
        self.unassigned_payments.append(pay)
        self._match_payments()
        return emitted

    def tick(self, t: float) -> list[Alert]:
        """Advance the clock; finalize any exits whose waiting period is over."""
        self.now = max(self.now, t)
        self._expire_payments()
        emitted: list[Alert] = []
        progress = True
        while progress:  # reconciling one person can unblock another
            progress = False
            for p in sorted((self.people[i] for i in self.pending), key=lambda r: r.t_exit):
                if p.reconciled:
                    continue
                if self._ready(p):
                    emitted += self._reconcile(p)
                    progress = True
        return emitted

    def next_deadline(self) -> float | None:
        """Earliest camera time at which a pending exit could become decidable.
        Always > now: once the grace period has passed, the next deadline is max_hold."""
        times = []
        for i in self.pending:
            t_exit = self.people[i].t_exit
            grace_end = t_exit + self.cfg.exit_grace_s
            times.append(grace_end if self.now < grace_end else t_exit + self.cfg.max_hold_s)
        return min(times) if times else None

    def finalize(self) -> list[Alert]:
        """End of stream: advance the clock through every pending deadline."""
        out = []
        while self.pending:
            out += self.tick(self.next_deadline())
        return out

    # ------------------------------------------------------------ basket ops

    def _person(self, pid: int, t: float) -> PersonRecord:
        if pid not in self.people:
            self.people[pid] = PersonRecord(person_id=pid, t_enter=t)
        return self.people[pid]

    def _category(self, ev: Event) -> str:
        return ev.item or self.catalog.category_of(ev.sku) or "unknown"

    def _on_pick(self, p: PersonRecord, ev: Event) -> None:
        others = [c for c in ev.candidates if c != p.person_id]
        p.basket.append(BasketItem(self._category(ev), ev.sku, ev.t, ev.zone, ev.confidence,
                                   ambiguous_with=others))
        amb = f" (ambiguous with {others})" if others else ""
        p.log.append(f"{ev.t:7.1f}s pick {self._category(ev)} from {ev.zone} conf={ev.confidence:.2f}{amb}")

    def _on_put_back(self, p: PersonRecord, ev: Event) -> None:
        cat = self._category(ev)
        # Remove the most recent matching item. Unknown category -> most recent item.
        for i in range(len(p.basket) - 1, -1, -1):
            if cat == "unknown" or p.basket[i].category == cat:
                removed = p.basket.pop(i)
                p.log.append(f"{ev.t:7.1f}s put_back {removed.category} to {ev.zone}")
                return
        p.log.append(f"{ev.t:7.1f}s put_back {cat} ignored (not in basket)")

    def _on_conceal(self, p: PersonRecord, ev: Event) -> None:
        cat = self._category(ev)
        for item in reversed(p.basket):
            if not item.concealed and (cat == "unknown" or item.category == cat):
                item.concealed = True
                p.log.append(f"{ev.t:7.1f}s conceal {item.category} conf={ev.confidence:.2f}")
                return
        # We saw it go into a pocket but never saw the pick: add it, at the conceal confidence.
        p.basket.append(BasketItem(cat, ev.sku, ev.t, ev.zone, ev.confidence,
                                   concealed=True, source="conceal"))
        p.log.append(f"{ev.t:7.1f}s conceal {cat} (pick not seen; added to basket)")

    def _on_register_visit(self, p: PersonRecord, ev: Event) -> None:
        """PAY from vision = a register visit. meta.phase is "start" (dwell threshold
        reached, still standing there), "end" (left the counter), or absent (a
        complete visit reported at once, with meta.t_start / meta.t_end)."""
        zone = ev.zone or "register"
        phase = ev.meta.get("phase")
        open_visit = next((v for v in reversed(p.register_visits)
                           if v.zone == zone and v.t_end is None), None)
        if phase == "start":
            if open_visit is None:
                p.register_visits.append(RegisterVisit(ev.meta.get("t_start", ev.t), None, zone))
            p.log.append(f"{ev.t:7.1f}s at register")
            return
        if phase == "end" and open_visit is not None:
            open_visit.t_end = ev.t
        else:
            p.register_visits.append(RegisterVisit(ev.meta.get("t_start", ev.t),
                                                   ev.meta.get("t_end", ev.t), zone))
        v = p.register_visits[-1] if open_visit is None else open_visit
        p.log.append(f"{ev.t:7.1f}s register visit {v.t_start:.1f}-{v.t_end:.1f}s")
        if self.cfg.payment_mode == "dwell":
            # No POS feed: assume they paid for everything they had picked by then.
            for item in p.basket:
                if item.t_pick <= v.t_end and not item.dwell_paid:
                    p.paid.append(LineItem(sku=item.sku, category=item.category))
                    item.dwell_paid = True

    # ------------------------------------------------------- payment matching

    def _match_payments(self) -> None:
        still: list[Payment] = []
        for pay in self.unassigned_payments:
            pid = self._payer_for(pay)
            if pid is None:
                still.append(pay)
                continue
            p = self.people[pid]
            for li in pay.items:
                for _ in range(li.qty):
                    p.paid.append(LineItem(sku=li.sku, category=self.catalog.category_of(li.sku, li.category)))
            p.log.append(f"{pay.t:7.1f}s payment {pay.txn_id or ''} at {pay.terminal}: "
                         f"{[(li.sku or li.category, li.qty) for li in pay.items]}")
        self.unassigned_payments = still

    def _payer_for(self, pay: Payment) -> int | None:
        if pay.person_id is not None:
            self._person(pay.person_id, pay.t)
            return pay.person_id
        zone = self.terminal_zones.get(pay.terminal, pay.terminal)
        kind = self.zone_kinds.get(zone, "register")
        if kind == "cooler":
            # Tap-to-open cooler: whoever picked from this cooler closest in time to the tap.
            best, best_dt = None, self.cfg.cooler_tap_window_s
            for p in self._recent(pay.t - self.cfg.cooler_tap_window_s):
                for item in p.basket:
                    dt = abs(item.t_pick - pay.t)
                    if item.zone == zone and dt <= best_dt:
                        best, best_dt = p.person_id, dt
            return best
        # Register: whoever was at the register when the POS closed the sale.
        # Ranking, in order: standing there at that exact time (vs. only within the
        # slack), hasn't already paid during this visit, got to the counter first
        # (the head of the queue is the one being served).
        s = self.cfg.register_slack_s
        ranked = []
        for p in self._recent(pay.t - s):
            for v in p.register_visits:
                if v.zone == zone and v.contains(pay.t, s):
                    ranked.append((not v.contains(pay.t), v.n_payments > 0, v.t_start, p.person_id, v))
        if not ranked:
            return None
        best = min(ranked, key=lambda r: r[:4])
        best[4].n_payments += 1
        return best[3]

    def _expire_payments(self) -> None:
        keep = []
        for pay in self.unassigned_payments:
            if self.now - pay.t > self.cfg.payment_expiry_s:
                self.orphan_payments.append(pay)
            else:
                keep.append(pay)
        self.unassigned_payments = keep

    # --------------------------------------------------------- reconciliation

    def _group_mates(self, p: PersonRecord) -> list[PersonRecord]:
        """People who came in with p (entered within group_window_s)."""
        w = self.cfg.group_window_s
        return [q for q in self._recent(p.t_enter - w)
                if q.person_id != p.person_id and abs(q.t_enter - p.t_enter) <= w]

    def _recent(self, since: float) -> list[PersonRecord]:
        """People who were in the store at or after `since` (keeps lookups O(active))."""
        return [q for q in self.people.values() if q.t_exit is None or q.t_exit >= since]

    def _ready(self, p: PersonRecord) -> bool:
        waited = self.now - p.t_exit
        if waited < self.cfg.exit_grace_s:
            return False
        if waited >= self.cfg.max_hold_s:
            return True
        # Hold while a group mate is still inside (one person often pays for the party).
        if any(q.t_exit is None for q in self._group_mates(p)):
            return False
        # Hold until everyone who might own one of our ambiguous picks has been
        # reconciled (their leftover payments may cover the item). If they are in
        # turn waiting on us, don't deadlock: go first.
        for item in p.basket:
            for other in item.ambiguous_with:
                q = self.people.get(other)
                if q is None or q.reconciled:
                    continue
                waits_on_us = any(p.person_id in it.ambiguous_with for it in q.basket)
                if q.t_exit is None or not waits_on_us:
                    return False
        return True

    def _reconcile(self, p: PersonRecord) -> list[Alert]:
        """Decide for p (and any group mates who have also left)."""
        party = [p] + [q for q in self._group_mates(p)
                       if q.t_exit is not None and not q.reconciled]
        for q in party:
            q.reconciled = True
            self.pending.discard(q.person_id)
            q.group = sorted(r.person_id for r in party)

        # Pool baskets and payments across the party.
        basket = [(q, it) for q in party for it in q.basket]
        paid = [li for q in party for li in q.paid]
        unpaid, surplus = self._subtract(basket, paid)

        # Ambiguous crowded picks: hand the item to another candidate who paid for one extra.
        still_unpaid = []
        for q, item in unpaid:
            moved = False
            for other in item.ambiguous_with:
                r = self.people.get(other)
                if r is not None and r.surplus[item.category] > 0:
                    r.surplus[item.category] -= 1
                    q.log.append(f"ambiguous {item.category} attributed to {other} (they paid for an extra)")
                    moved = True
                    break
            if not moved:
                still_unpaid.append((q, item))
        # Leftover surplus stays on the payer so a later-exiting crowd candidate can claim it.
        p.surplus.update(surplus)
        for q, item in still_unpaid:
            q.unpaid.append(item)

        if not still_unpaid:
            for q in party:
                q.log.append(f"{self.now:7.1f}s reconciled: all paid")
            return []
        return self._score_and_emit(p, party, [it for _, it in still_unpaid], paid)

    def _subtract(self, basket: list[tuple[PersonRecord, BasketItem]], paid: list[LineItem]):
        """Multiset basket - paid. SKU matches first, then category."""
        remaining = list(paid)
        unpaid = []
        # Pass 1: exact SKU.
        leftovers = []
        for q, item in basket:
            idx = next((i for i, li in enumerate(remaining)
                        if item.sku is not None and li.sku == item.sku), None)
            if idx is None:
                leftovers.append((q, item))
            else:
                remaining.pop(idx)
        # Pass 2: category. Payments clear the least suspicious items first: if someone
        # pocketed one soda and paid for the one in their hand, the pocketed one is unpaid.
        leftovers.sort(key=lambda qi: (qi[1].concealed, not qi[1].held_at_exit))
        for q, item in leftovers:
            idx = next((i for i, li in enumerate(remaining) if li.category == item.category), None)
            if idx is None:
                unpaid.append((q, item))
            else:
                remaining.pop(idx)
        surplus = Counter(li.category for li in remaining)
        return unpaid, surplus

    def _score_item(self, item: BasketItem, visited_register: bool) -> tuple[float, list[str]]:
        c = self.cfg
        why = [f"{item.category}: taken from {item.zone} at {item.t_pick:.1f}s "
               f"(pick conf {item.confidence:.2f}) and not paid for"]
        score = c.w_pick * item.confidence
        if item.concealed:
            score += c.w_conceal
            why.append(f"{item.category}: concealed" + (" (pick itself not seen)" if item.source == "conceal" else ""))
        if item.held_at_exit:
            score += c.w_held_at_exit
            why.append(f"{item.category}: visible in hand at exit")
        if not visited_register:
            score += c.w_no_register
        if item.ambiguous_with:
            score *= c.ambiguous_factor
            why.append(f"{item.category}: crowded pick, may belong to person(s) {item.ambiguous_with}")
        return min(score, 1.0), why

    def _score_and_emit(self, p: PersonRecord, party: list[PersonRecord],
                        unpaid: list[BasketItem], paid: list[LineItem]) -> list[Alert]:
        visited = any(q.register_visits for q in party) or bool(paid)
        items, reasons = [], []
        miss = 1.0
        for item in unpaid:
            s, why = self._score_item(item, visited)
            miss *= (1.0 - s)
            reasons += why
            items.append(UnpaidItem(item.category, item.sku, item.t_pick, item.zone, item.confidence,
                                    item.concealed, item.held_at_exit, item.ambiguous_with, round(s, 3)))
        if not visited:
            reasons.append("never went to the register; no payment matched")
        elif paid:
            reasons.append(f"paid for {len(paid)} item(s), but not these")
        conf = round(1.0 - miss, 3)

        tier = ("alert" if conf >= self.cfg.alert_threshold
                else "review" if conf >= self.cfg.review_threshold else None)
        who = min(party, key=lambda q: q.person_id) if len(party) > 1 else p
        for q in party:
            q.log.append(f"{self.now:7.1f}s reconciled: {len(unpaid)} unpaid, confidence {conf:.2f} -> {tier}")
        if tier is None:
            self.dropped.append({"person_id": who.person_id, "confidence": conf,
                                 "group": [q.person_id for q in party]})
            return []
        t_exit = max(q.t_exit for q in party)
        alert = Alert(
            alert_id=f"A{next(self._alert_ids):05d}",
            person_id=who.person_id, tier=tier, confidence=conf,
            t_exit=t_exit, t_emitted=self.now,
            unpaid_items=items, reasons=reasons,
            paid_items=[{"sku": li.sku, "category": li.category} for li in paid],
            visited_register=visited, group=[q.person_id for q in party] if len(party) > 1 else [],
        )
        self.alerts.append(alert)
        return [alert]

    # ---------------------------------------------------------------- queries

    def basket_of(self, pid: int) -> Counter:
        p = self.people.get(pid)
        return Counter(it.category for it in p.basket) if p else Counter()

    def replay(self, events: Iterable[Event], payments: Iterable[Payment] = ()) -> list[Alert]:
        """Run a full recorded stream (events + payments merged by time)."""
        stream = [(e.t, 0, e) for e in events] + [(pay.t, 1, pay) for pay in payments]
        stream.sort(key=lambda x: (x[0], x[1]))
        out = []
        for t, _, x in stream:
            # Advance the clock through any decision deadlines before this item,
            # exactly as a live system ticking every frame would.
            while (d := self.next_deadline()) is not None and d <= t:
                out += self.tick(d)
            out += self.on_payment(x) if isinstance(x, Payment) else self.on_event(x)
        out += self.finalize()
        return out


def build_ledger(store, **overrides) -> Ledger:
    """Ledger wired to a StoreConfig (terminals, zone kinds, catalog, YAML ledger settings)."""
    known = LedgerConfig.__dataclass_fields__
    settings = {k: v for k, v in {**store.ledger, **overrides}.items() if k in known}
    return Ledger(store.catalog, LedgerConfig(**settings),
                  terminal_zones=store.terminals, zone_kinds=store.zone_kinds)
