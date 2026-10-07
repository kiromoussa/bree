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
   A receipt that arrives even later (batched POS export) can still retract or
   downgrade the decision, never raise it (`_late_receipts`).

4. Reconciliation = basket minus paid items, matched by SKU when we have it and
   by category otherwise. Every unpaid item gets an evidence score from the
   table in `LedgerConfig`. Scores are combined with noisy-OR into one
   confidence. Above `alert_threshold` -> "alert"; above `review_threshold` ->
   "review" (a manager looks later, nobody gets confronted); else dropped.

Design rule: one uncorroborated pick must never be enough for an "alert".
A missed put-back looks exactly like that, and false accusations kill a pilot.
Same rule for identity: when the engine had to pick between two people it could not tell
apart (closed-world identity, `meta.identity_uncertain` on their events), a decision that
rests on that identity is capped at "review" and the reason goes into the audit log.
"""
from __future__ import annotations

import itertools
import math
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
    # The discount ends when every other candidate has been reconciled without paying for an extra one: then the
    # item is unpaid whoever of them took it (review at most while it is not concealed). False: discount for ever.
    ambiguous_settles: bool = True
    # An unpaid pick in doubt (crowded, or on an uncertain identity) is covered by a paid item nobody saw its payer
    # take, whoever that payer is (receipts given out by visit time land on the wrong one of two people at the counter).
    doubt_takes_any_extra: bool = True
    no_receipt_factor: float = 0.6   # went to the register but no receipt matched: likely a POS gap
    # A paid item nobody saw them take stands for one unpaid pick that was seen: most likely one product
    # under two names (vision misread it). That pick's score is multiplied by this. Concealed items never.
    misread_factor: float = 0.5
    min_conceal_conf: float = 0.3    # conceal detections below this are ignored
    conceal_conf_ref: float = 0.6    # w_conceal is scaled by min(1, conceal_conf / this)
    # "alert" needs corroboration: at least one unpaid item that was concealed or seen in hand
    # at the exit, and more such items than unmatched paid items (a receipt we couldn't match to
    # the basket is most likely a vision/POS category mismatch). Otherwise at most "review".
    require_corroboration: bool = True
    # Combining items into one confidence. "max_plus": strongest item + extra_item_bonus per
    # additional unpaid item (capped). "noisy_or": 1 - prod(1 - s_i). Items in one basket are NOT
    # independent evidence (one payment-matching failure leaves all of them unpaid), so max_plus.
    combine: str = "max_plus"
    extra_item_bonus: float = 0.05
    extra_item_cap: float = 0.10

    alert_threshold: float = 0.7
    review_threshold: float = 0.4

    exit_grace_s: float = 5.0        # wait for late POS messages after exit
    max_hold_s: float = 120.0        # longest we hold a decision for a group mate / crowd candidate
    group_window_s: float = 4.0      # people entering within this window may be one party
    cooler_tap_window_s: float = 20.0  # a tap is matched to a pick at that cooler within this window
    register_slack_s: float = 3.0    # POS timestamp may fall just outside the dwell interval
    # The receipt reaches us this long (shortest, longest) after its payer was served: the sale closes and the
    # receipt prints after the customer has turned away, and the next one in line is often already at the counter.
    # A property of the store's POS feed, measured once per store. (0, 0): the receipt is stamped while they stand there.
    pos_lag_s: tuple[float, float] = (0.0, 0.0)
    # Register receipts are given out together, when somebody who could be the payer is reconciled, not one at a time
    # as they arrive: among everyone at the counter within `register_slack_s` of when a receipt's payer was served,
    # the receipts go where most of their items are found in the baskets, each receipt once (`_settle_receipts`).
    # Two people at the counter at once, or a track swap there, make "who stood there at that second" a guess.
    # Off here: a receipt is then credited as it arrives (the per-camera engine and the dashboard read `paid` live).
    # The store pipeline (bree.shelf.store.JOINT_RECEIPTS) switches it on.
    joint_receipts: bool = False
    payment_expiry_s: float = 300.0  # unmatched payments are dropped (and logged) after this
    # A receipt that reaches us after the decision (batched POS export, network lag) can still
    # lower it if it arrives within this long after the exit. 0 = never retract.
    late_receipt_window_s: float = 300.0


@dataclass
class BasketItem:
    category: str
    sku: str | None
    t_pick: float
    zone: str | None
    confidence: float
    ambiguous_with: list[int] = field(default_factory=list)
    concealed: bool = False
    conceal_conf: float = 0.0
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
    party: list[int] | None = None                            # who they walk with, when the tracker says (ENTER meta "party"); None: go by entry time
    t_last: float = 0.0                                       # last event time (stale-track pruning)
    identity_uncertain: tuple[str, float] | None = None       # (why, until): engine could not tell this person from another
    log: list[str] = field(default_factory=list)              # human-readable audit trail


@dataclass
class _Flagged:
    """An alert/review decision that a late receipt can still retract (see `_late_receipts`)."""
    alert: Alert
    decision: dict
    payer: PersonRecord
    party: list[PersonRecord]
    surplus: Counter
    n_paid: dict[int, int]           # person -> paid lines already counted in the decision
    tier: str | None


TIER_RANK = {None: 0, "review": 1, "alert": 2}


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
        self.active: dict[int, PersonRecord] = {}    # people who can still matter for a decision
        self.decisions: list[dict] = []              # every reconciliation, clean ones included
        self.archived: list[PersonRecord] = []       # earlier visits of re-used track ids
        self.flagged: list[_Flagged] = []            # flagged decisions still open to a late receipt
        self._last_prune = 0.0
        self._alert_ids = itertools.count(1)

    # ------------------------------------------------------------------ input

    def on_event(self, ev: Event) -> list[Alert]:
        """Feed one event. Returns any alerts that became final at this time."""
        emitted = self.tick(ev.t)
        old = self.people.get(ev.person_id)
        if ev.type == EventType.ENTER and old is not None and old.t_exit is not None:
            # Track id reused for a new visit: archive the old record, start a fresh one.
            self.archived.append(old)
            del self.people[ev.person_id]
            self.active.pop(ev.person_id, None)
            self.pending.discard(ev.person_id)
        p = self._person(ev.person_id, ev.t)
        p.t_last = max(p.t_last, ev.t)

        why = ev.meta.get("identity_uncertain")
        if why and (p.identity_uncertain is None or p.identity_uncertain[0] != why):
            p.log.append(f"{ev.t:7.1f}s identity uncertain: {why}")
        if why:
            p.identity_uncertain = (why, ev.meta.get("identity_uncertain_until", float("inf")))

        if ev.type == EventType.ENTER:
            p.t_enter = min(p.t_enter, ev.t)
            if "party" in ev.meta:
                p.party = list(ev.meta["party"])
            if ev.meta.get("missed_entry"):
                p.log.append(f"{ev.t:7.1f}s first seen inside the store (entry not seen)")
        elif ev.type == EventType.PICK:
            self._on_pick(p, ev)
        elif ev.type == EventType.PUT_BACK:
            self._on_put_back(p, ev)
        elif ev.type == EventType.CONCEAL:
            self._on_conceal(p, ev)
        elif ev.type == EventType.PAY:
            self._on_register_visit(p, ev)
        elif ev.type == EventType.EXIT:
            if p.reconciled:
                p.log.append(f"{ev.t:7.1f}s second exit for an already reconciled track: ignored")
                return emitted
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
        emitted = self.tick(pay.t if pay.t_received is None else pay.t_received)
        self.unassigned_payments.append(pay)
        self._match_payments()
        return emitted + self._late_receipts()

    def tick(self, t: float) -> list[Alert]:
        """Advance the clock; finalize any exits whose waiting period is over."""
        self.now = max(self.now, t)
        self._expire_payments()
        if self.now - self._last_prune > 60:
            self._prune()
        emitted: list[Alert] = []
        progress = True
        while progress:  # reconciling one person can unblock another
            progress = False
            for p in sorted((self.people[i] for i in self.pending), key=lambda r: r.t_exit):
                if p.reconciled:
                    self.pending.discard(p.person_id)
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
            if self.people[i].reconciled:
                continue
            t_exit = self.people[i].t_exit
            grace_end = t_exit + self.cfg.exit_grace_s
            times.append(grace_end if self.now < grace_end else t_exit + self.cfg.max_hold_s)
        return min(times) if times else None

    def finalize(self) -> list[Alert]:
        """End of stream: advance the clock through every pending deadline."""
        out = []
        while self.pending:
            before = (self.now, len(self.pending))
            out += self.tick(self.next_deadline())
            if (self.now, len(self.pending)) == before:
                out += self.tick(self.now + self.cfg.max_hold_s + 1.0)   # force everything out
                break
        return out

    # ------------------------------------------------------------ basket ops

    def _person(self, pid: int, t: float) -> PersonRecord:
        if pid not in self.people:
            self.people[pid] = PersonRecord(person_id=pid, t_enter=t, t_last=t)
            self.active[pid] = self.people[pid]
        return self.people[pid]

    def _prune(self) -> None:
        """Drop people from the working set once they can no longer affect anyone's decision:
        reconciled long enough ago, or a track that went silent inside the store (lost)."""
        self._last_prune = self.now
        keep_after_exit = max(self.cfg.max_hold_s + self.cfg.cooler_tap_window_s + self.cfg.group_window_s,
                              self.cfg.late_receipt_window_s) + 60
        for pid, p in list(self.active.items()):
            if p.reconciled and p.t_exit is not None and self.now - p.t_exit > keep_after_exit:
                del self.active[pid]
            elif p.t_exit is None and self.now - p.t_last > 3600:
                p.log.append(f"{self.now:7.1f}s track silent for 1h without exit: dropped (not reconciled)")
                del self.active[pid]

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
        # Remove the most recent matching item that is NOT concealed (someone who pockets one
        # candy bar and puts a second one back still has the pocketed one). Unknown category ->
        # any item. A concealed item is only removed if nothing else matches (logged).
        matches = [i for i in range(len(p.basket)) if cat == "unknown" or p.basket[i].category == cat]
        if not matches:
            p.log.append(f"{ev.t:7.1f}s put_back {cat} ignored (not in basket)")
            return
        open_ = [i for i in matches if not p.basket[i].concealed]
        removed = p.basket.pop((open_ or matches)[-1])
        note = "" if open_ else " (was marked concealed: suspicious)"
        p.log.append(f"{ev.t:7.1f}s put_back {removed.category} to {ev.zone}{note}")

    def _on_conceal(self, p: PersonRecord, ev: Event) -> None:
        cat = self._category(ev)
        if ev.confidence < self.cfg.min_conceal_conf:
            p.log.append(f"{ev.t:7.1f}s conceal {cat} ignored (conf {ev.confidence:.2f} too low)")
            return
        for item in reversed(p.basket):
            if not item.concealed and (cat == "unknown" or item.category == cat):
                item.concealed, item.conceal_conf = True, ev.confidence
                # This person hid it: a crowded pick is no longer ambiguous.
                item.ambiguous_with = []
                p.log.append(f"{ev.t:7.1f}s conceal {item.category} conf={ev.confidence:.2f}")
                return
        # We saw it go into a pocket but never saw the pick: add it, at the conceal confidence.
        p.basket.append(BasketItem(cat, ev.sku, ev.t, ev.zone, ev.confidence,
                                   concealed=True, conceal_conf=ev.confidence, source="conceal"))
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
                p.register_visits.append(RegisterVisit(ev.meta.get("t_start") or ev.t, None, zone))
            p.log.append(f"{ev.t:7.1f}s at register")
            return
        if phase == "end" and open_visit is not None:
            open_visit.t_end = ev.t
        else:
            p.register_visits.append(RegisterVisit(ev.meta.get("t_start") or ev.t,
                                                   ev.meta.get("t_end") or ev.t, zone))
        v = p.register_visits[-1] if open_visit is None else open_visit
        p.log.append(f"{ev.t:7.1f}s register visit {v.t_start:.1f}-{v.t_end:.1f}s")
        if self.cfg.payment_mode == "dwell":
            # No POS feed: assume they paid for everything they had picked by then.
            for item in p.basket:
                if item.t_pick <= v.t_end and not item.dwell_paid and not item.concealed:
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
        receipt = Counter(self.catalog.category_of(li.sku, li.category) for li in pay.items for _ in range(li.qty))
        if kind == "cooler":
            # Tap at a cooler: someone who picked from this cooler around the tap. Ranking:
            # receipt matches their (party's) unpaid picks best, then closest in time.
            ranked = []
            for p in self._recent(pay.t - self.cfg.cooler_tap_window_s):
                dts = [abs(it.t_pick - pay.t) for it in p.basket if it.zone == zone]
                if dts and min(dts) <= self.cfg.cooler_tap_window_s:
                    ranked.append((-self._receipt_overlap(p, receipt), min(dts), p.person_id))
            return min(ranked)[2] if ranked else None
        # Register: one of the people at the register when the payer was served (the receipt
        # time less the POS lag). Ranking, in order:
        #   1. standing there at that time (vs. only within the slack)
        #   2. receipt matches their basket: most receipt items found among their unpaid
        #      picks (the receipt says "soda + candy"; the person holding soda + candy paid)
        #   3. hasn't already paid during this visit
        #   4. got to the counter first
        # Time before content: baskets come from vision and have wrong and missing items, and a
        # thief's unpaid item must not pull in the receipt of the next customer who bought the same.
        if self.cfg.joint_receipts:
            return None          # held until one of the people who could be the payer is reconciled (`_settle_receipts`)
        s = self.cfg.register_slack_s
        lo, hi = pay.t - max(self.cfg.pos_lag_s), pay.t - min(self.cfg.pos_lag_s)     # when the payer was being served

        def served(v: RegisterVisit, slack: float = 0.0) -> bool:
            return v.t_start - slack <= hi and lo <= (float("inf") if v.t_end is None else v.t_end) + slack
        ranked = []
        for p in self._recent(lo - s):
            for v in p.register_visits:
                if v.zone == zone and served(v, s):
                    overlap = self._receipt_overlap(p, receipt, pay.t)
                    ranked.append((not served(v), -overlap, v.n_payments > 0, v.t_start, p.person_id, v))
        if not ranked:
            return None
        best = min(ranked, key=lambda r: r[:5])
        if best[1] == 0 and sum(receipt.values()) > 0:
            # The receipt matches none of the best candidate's picks. Credit it only if they were
            # the one person at the counter then (vision missed or misnamed their picks; the
            # unmatched paid items are weighed at reconciliation). Otherwise don't guess: it stays
            # unassigned and is claimed at reconciliation (basket match, then visit time).
            there = {r[4] for r in ranked if r[0] == best[0]}
            if len(there) != 1 or (best[0] and self._receipt_party_unpaid(self.people[best[4]], pay.t)):
                return None
        best[5].n_payments += 1
        return best[4]

    def _settle_receipts(self, party: list[PersonRecord]) -> list[Payment]:
        """Give out the open register receipts together. A receipt can go to anyone not yet reconciled (or in `party`)
        who was at that counter within the slack of when its payer was served. Of all ways to hand them out, take the
        one with the most receipt items found among the unpaid picks of the people they go to; then fewest receipts
        given on the slack alone, fewest second receipts for one person, closest in time. A receipt that adds no
        matched item goes by time only when one person could be its payer; otherwise it stays open. Receipts that
        fall to `party` are credited now. -> the receipts kept for somebody else (not to be claimed by basket match)."""
        s = self.cfg.register_slack_s
        opts: list[tuple[Payment, Counter, list[tuple[PersonRecord, float]]]] = []
        for pay in self.unassigned_payments:
            zone = self.terminal_zones.get(pay.terminal, pay.terminal)
            if pay.person_id is not None or self.zone_kinds.get(zone, "register") != "register":
                continue
            lo, hi = pay.t - max(self.cfg.pos_lag_s), pay.t - min(self.cfg.pos_lag_s)
            who = []
            for q in self._recent(lo - s):
                if q.reconciled and q not in party:
                    continue
                off = min((max(0.0, v.t_start - hi, lo - (float("inf") if v.t_end is None else v.t_end)) for v in q.register_visits if v.zone == zone), default=float("inf"))
                if off <= s:
                    who.append((q, off))
            if who:
                opts.append((pay, Counter(self.catalog.category_of(li.sku, li.category) for li in pay.items for _ in range(li.qty)), who))
        if not opts:
            return []
        unpaid: dict[tuple[int, float], Counter] = {}

        def matched(give: tuple) -> int:
            got: dict[int, list[int]] = {}
            for i, q in enumerate(give):
                if q is not None:
                    got.setdefault(q.person_id, []).append(i)
            n = 0
            for pid, rs in got.items():
                t = max(opts[i][0].t for i in rs)
                if (pid, t) not in unpaid:
                    unpaid[(pid, t)] = self._receipt_party_unpaid(self.people[pid], t)
                n += sum((sum((opts[i][1] for i in rs), Counter()) & unpaid[(pid, t)]).values())
            return n

        def rank(give: tuple):
            offs = [dict((q.person_id, off) for q, off in opts[i][2])[g.person_id] for i, g in enumerate(give) if g is not None]
            per = Counter(g.person_id for g in give if g is not None)
            return (-matched(give), sum(o > 0 for o in offs), sum(n - 1 for n in per.values()), round(sum(offs), 3), [g.person_id if g else -1 for g in give])
        ways = math.prod(len(w) for _, _, w in opts)
        if ways <= 20000:
            best = min(itertools.product(*[[q for q, _ in w] for _, _, w in opts]), key=rank)
        else:
            # ponytail: a counter this crowded is handed out one receipt at a time, in time order; split by people in common if a store needs it
            best = (None,) * len(opts)
            for i in range(len(opts)):
                best = min((best[:i] + (q,) + best[i + 1:] for q, _ in opts[i][2]), key=rank)
        best, total = list(best), matched(tuple(best))
        for i, (pay, _, who) in enumerate(opts):          # no item of it found on them, and others stood there too: don't guess
            if len(who) > 1 and matched(tuple(None if k == i else g for k, g in enumerate(best))) == total:
                best[i] = None
                total = matched(tuple(best))
        kept = []
        for (pay, _, _), q in zip(opts, best):
            if q is None:
                continue
            if q not in party:
                kept.append(pay)
                continue
            self.unassigned_payments.remove(pay)
            for li in pay.items:
                for _ in range(li.qty):
                    q.paid.append(LineItem(sku=li.sku, category=self.catalog.category_of(li.sku, li.category)))
            q.log.append(f"{pay.t:7.1f}s payment {pay.txn_id or ''} at {pay.terminal}: {[(li.sku or li.category, li.qty) for li in pay.items]}")
        return kept

    def _receipt_party_unpaid(self, p: PersonRecord, before: float = float("inf")) -> Counter:
        """Unpaid picks (by category) of p's party: p + people who came in with p. `before`: only
        picks made by then (a receipt cannot list an item taken after it was printed)."""
        party = [p] + self._group_mates(p)
        return Counter(it.category for q in party for it in q.basket if it.t_pick <= before) - \
            Counter(li.category for q in party for li in q.paid)

    def _receipt_overlap(self, p: PersonRecord, receipt: Counter, before: float = float("inf")) -> int:
        """How many receipt items are among the unpaid picks of p's party
        (one person often pays for the group)."""
        return sum((receipt & self._receipt_party_unpaid(p, before)).values())

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
        """People who came in with p: the ones the tracker saw walking with them, else (no such
        information) whoever entered within group_window_s."""
        if p.party is not None:
            return [self.people[i] for i in p.party if i != p.person_id and i in self.people]
        w = self.cfg.group_window_s
        return [q for q in self._recent(p.t_enter - w)
                if q.person_id != p.person_id and abs(q.t_enter - p.t_enter) <= w]

    def _recent(self, since: float) -> list[PersonRecord]:
        """People who were in the store at or after `since` (keeps lookups O(active))."""
        return [q for q in self.active.values() if q.t_exit is None or q.t_exit >= since]

    def _ready(self, p: PersonRecord) -> bool:
        waited = self.now - p.t_exit + 1e-6          # epsilon: t_exit + hold - t_exit can round below hold
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

        self._claim_unassigned_receipts(p, party, self._settle_receipts(party) if self.cfg.joint_receipts else [])

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
            if not moved and self.cfg.doubt_takes_any_extra and (item.ambiguous_with or q.identity_uncertain):
                # Who took this pick, or who this person is, was in doubt, and somebody else in the store since then
                # paid for one of these that nobody saw them take: the store was paid for it.
                r = next((r for r in self.people.values() if r.reconciled and r not in party and r.surplus[item.category] > 0
                          and r.t_exit is not None and r.t_exit >= item.t_pick), None)
                if r is not None:
                    r.surplus[item.category] -= 1
                    q.log.append(f"{item.category} in doubt (who took it, or who this is): person {r.person_id} paid for one nobody saw them take")
                    moved = True
            if not moved:
                still_unpaid.append((q, item))
        # Leftover surplus stays on the payer so a later-exiting crowd candidate can claim it.
        p.surplus.update(surplus)
        for q, item in still_unpaid:
            q.unpaid.append(item)

        if not still_unpaid:
            for q in party:
                q.log.append(f"{self.now:7.1f}s reconciled: all paid")
            self.decisions.append({"person_id": p.person_id, "group": [q.person_id for q in party],
                                   "confidence": 0.0, "tier": None, "t": self.now,
                                   "t_exit": max(q.t_exit for q in party), "unpaid": []})
            return []
        return self._score_and_emit(p, party, still_unpaid, paid, surplus)

    def _claim_unassigned_receipts(self, p: PersonRecord, party: list[PersonRecord], kept: list[Payment] = ()) -> None:
        """Fallback when time/place attribution failed (register visit not seen, POS clock
        off): a still-unassigned receipt from while the party was in the store, whose items
        are mostly among the party's unpaid picks, is theirs."""
        t0 = min(q.t_enter for q in party)
        t1 = max(q.t_exit for q in party) + self.cfg.exit_grace_s
        while True:
            # Only open (non-concealed) items count: a claimed receipt can't clear a concealed item.
            unpaid = Counter(it.category for q in party for it in q.basket if not it.concealed) - \
                Counter(li.category for q in party for li in q.paid)
            best, best_overlap = None, 0
            for pay in self.unassigned_payments:
                if any(pay is k for k in kept):          # the joint hand-out gave it to somebody still to be reconciled
                    continue
                if not (t0 <= pay.t <= t1) or self.zone_kinds.get(self.terminal_zones.get(pay.terminal, ""), "") == "cooler":
                    continue
                receipt = Counter(self.catalog.category_of(li.sku, li.category) for li in pay.items for _ in range(li.qty))
                overlap = sum((receipt & unpaid).values())
                if overlap > best_overlap and overlap >= 0.5 * sum(receipt.values()):
                    best, best_overlap = pay, overlap
            if best is None:
                best = self._receipt_during_visit(party, t0, t1, kept)
                if best is None:
                    return
            self.unassigned_payments.remove(best)
            for li in best.items:
                for _ in range(li.qty):
                    p.paid.append(LineItem(sku=li.sku, category=self.catalog.category_of(li.sku, li.category),
                                           via="basket_match"))
            p.log.append(f"{best.t:7.1f}s receipt {best.txn_id} claimed by basket match "
                         f"(register visit not seen / POS clock off)")

    def _receipt_during_visit(self, party: list[PersonRecord], t0: float, t1: float, kept: list[Payment] = ()) -> Payment | None:
        """Last resort for a party that stood at the register but has no receipt at all:
        an unassigned register receipt printed while they were standing there."""
        if any(q.paid for q in party):
            return None
        s = self.cfg.register_slack_s
        for pay in self.unassigned_payments:
            zone = self.terminal_zones.get(pay.terminal, pay.terminal)
            if self.zone_kinds.get(zone, "register") != "register" or any(pay is k for k in kept):
                continue
            if any(v.zone == zone and v.contains(pay.t, s) for q in party for v in q.register_visits):
                return pay
        return None

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
            # A receipt we only claimed by basket match (not seen at the register) can't clear a
            # concealed item: otherwise a walkout thief could "claim" a stranger's receipt.
            idx = next((i for i, li in enumerate(remaining) if li.category == item.category
                        and not (item.concealed and li.via == "basket_match")), None)
            if idx is None and not item.concealed:
                # Receipt line whose SKU isn't in the catalog: could be anything, let it cover
                # an open (non-concealed) item.
                idx = next((i for i, li in enumerate(remaining) if li.category is None), None)
            if idx is None:
                unpaid.append((q, item))
            else:
                remaining.pop(idx)
        surplus = Counter(li.category for li in remaining)
        return unpaid, surplus

    def _score_item(self, item: BasketItem, visited_register: bool, paid_any: bool = True) -> tuple[float, list[str]]:
        c = self.cfg
        why = [f"{item.category}: taken from {item.zone} at {item.t_pick:.1f}s "
               f"(pick conf {item.confidence:.2f}) and not paid for"]
        score = c.w_pick * item.confidence
        if item.concealed:
            score += c.w_conceal * min(1.0, item.conceal_conf / c.conceal_conf_ref)
            why.append(f"{item.category}: concealed" + (" (pick itself not seen)" if item.source == "conceal" else ""))
        if item.held_at_exit:
            score += c.w_held_at_exit
            why.append(f"{item.category}: visible in hand at exit")
        if not visited_register:
            score += c.w_no_register
        elif not paid_any:
            score *= c.no_receipt_factor
        if item.ambiguous_with:
            # The doubt is who took it. While another candidate's receipts are still to come, the item is discounted.
            # Once every candidate has been reconciled and none paid for an extra one, it left unpaid with one of them.
            waiting = [o for o in item.ambiguous_with if (r := self.people.get(o)) is not None and not r.reconciled]
            if waiting or not c.ambiguous_settles:
                score *= c.ambiguous_factor
            why.append(f"{item.category}: crowded pick, may belong to person(s) {item.ambiguous_with}"
                       + ("" if waiting or not c.ambiguous_settles else " (all have left, none paid for it)"))
        return min(score, 1.0), why

    def _assess(self, party: list[PersonRecord], unpaid_owned: list[tuple[PersonRecord, BasketItem]],
                paid: list[LineItem], surplus: Counter):
        """Score a party's unpaid items -> (confidence, tier, alert_eligible, items, reasons, who)."""
        visited = any(q.register_visits for q in party) or bool(paid)
        items, reasons = [], []
        miss, scores = 1.0, []
        best_owner, best_score = party[0], -1.0
        for owner, item in unpaid_owned:
            s, why = self._score_item(item, visited, bool(paid))
            if s > best_score:
                best_owner, best_score = owner, s
            miss *= (1.0 - s)
            scores.append(s)
            reasons += why
            items.append(UnpaidItem(item.category, item.sku, item.t_pick, item.zone, item.confidence,
                                    item.concealed, item.held_at_exit, item.ambiguous_with, round(s, 3)))
        # each paid-but-not-seen item explains one open unpaid pick, the weakest ones first
        open_ = sorted((i for i, (_, it) in enumerate(unpaid_owned) if not it.concealed and not it.held_at_exit), key=lambda i: scores[i])
        for i in open_[:sum(surplus.values())]:
            scores[i] *= self.cfg.misread_factor
            items[i].score = round(scores[i], 3)
            reasons.append(f"{items[i].category}: a paid item did not match the basket, this pick may be that item under another name")
        best_owner = unpaid_owned[max(range(len(scores)), key=scores.__getitem__)][0]
        miss = math.prod(1.0 - x for x in scores)
        if not visited:
            reasons.append("never went to the register; no payment matched")
        elif not paid:
            reasons.append("went to the register but no receipt matched (possible POS feed gap: downweighted)")
        elif paid:
            reasons.append(f"paid for {len(paid)} item(s), but not these")
        if self.cfg.combine == "noisy_or":
            conf = round(1.0 - miss, 3)
        else:
            bonus = min(self.cfg.extra_item_cap, self.cfg.extra_item_bonus * (len(scores) - 1))
            conf = round(min(1.0, max(scores) + bonus), 3)

        tier = ("alert" if conf >= self.cfg.alert_threshold
                else "review" if conf >= self.cfg.review_threshold else None)
        corroborated = sum(1 for _, it in unpaid_owned if it.concealed or it.held_at_exit)
        unmatched_paid = sum(surplus.values())
        eligible = not (self.cfg.require_corroboration and corroborated <= unmatched_paid)
        if tier == "alert" and not eligible:
            tier = "review"
            reasons.append("capped at review: " + ("no concealment or item-in-hand at exit" if not corroborated
                           else f"{unmatched_paid} paid item(s) didn't match the basket (possible misrecognition)"))
        if tier == "alert" and any(it.ambiguous_with and not it.concealed for _, it in unpaid_owned):
            tier, eligible = "review", False
            reasons.append("capped at review: a pick that also fits another person")
        # Never accuse on a guessed identity: the basket may be someone else's.
        unsure = [q for q in party if q.identity_uncertain and (q.t_exit or 0.0) <= q.identity_uncertain[1]]
        if tier == "alert" and unsure:
            tier, eligible = "review", False
            reasons.append("capped at review: identity uncertain (" + "; ".join(q.identity_uncertain[0] for q in unsure) + ")")
            for q in party:
                q.log.append(f"{self.now:7.1f}s alert downgraded to review: identity uncertain "
                             f"({'; '.join(r.identity_uncertain[0] for r in unsure)})")
        # in a group: the person holding the strongest unpaid item
        return conf, tier, eligible, items, reasons, best_owner

    def _score_and_emit(self, p: PersonRecord, party: list[PersonRecord],
                        unpaid_owned: list[tuple[PersonRecord, BasketItem]], paid: list[LineItem],
                        surplus: Counter | None = None) -> list[Alert]:
        surplus = surplus or Counter()
        conf, tier, eligible, items, reasons, who = self._assess(party, unpaid_owned, paid, surplus)
        visited = any(q.register_visits for q in party) or bool(paid)
        for q in party:
            q.log.append(f"{self.now:7.1f}s reconciled: {len(unpaid_owned)} unpaid, confidence {conf:.2f} -> {tier}")
        decision = {"person_id": who.person_id, "group": [q.person_id for q in party], "confidence": conf,
                    "alert_eligible": eligible,
                    "tier": tier, "t": self.now, "t_exit": max(q.t_exit for q in party),
                    "unpaid": [i.category for i in items]}
        self.decisions.append(decision)
        if tier is None:
            self.dropped.append(decision)
            return []
        t_exit = max(q.t_exit for q in party)
        alert = Alert(
            alert_id=f"A{next(self._alert_ids):05d}",
            person_id=who.person_id, tier=tier, confidence=conf,
            t_exit=t_exit, t_emitted=self.now,
            unpaid_items=items, reasons=reasons,
            paid_items=[{"sku": li.sku, "category": li.category} for li in paid],
            visited_register=visited, group=[q.person_id for q in party] if len(party) > 1 else [],
            basket=[it.category for q in party for it in q.basket],
            audit_log=self._audit(party),
        )
        self.alerts.append(alert)
        if self.cfg.late_receipt_window_s > 0:
            self.flagged.append(_Flagged(alert, decision, p, party, Counter(surplus),
                                         {q.person_id: len(q.paid) for q in party}, tier))
        return [alert]

    @staticmethod
    def _audit(party: list[PersonRecord]) -> list[str]:
        return [(f"P{q.person_id} " if len(party) > 1 else "") + line for q in party for line in q.log]

    def _late_receipts(self) -> list[Alert]:
        """Receipts that reached us after an alert/review decision. Same attribution rules as on
        time: place/time + content in `_match_payments` first, then the basket-match claim (which
        can't clear a concealed item). If the re-scored tier is lower, emit a retraction: an Alert
        with `retracts` = the original id and tier "review" (downgrade) or "retracted". A late
        receipt never raises a tier. Open for `late_receipt_window_s` after the exit."""
        out = []
        for f in self.flagged:
            if self.now - f.alert.t_exit > self.cfg.late_receipt_window_s:
                continue
            self._claim_unassigned_receipts(f.payer, f.party, self._settle_receipts(f.party) if self.cfg.joint_receipts else [])
            new = [li for q in f.party for li in q.paid[f.n_paid[q.person_id]:]]
            if not new:
                continue
            f.n_paid = {q.person_id: len(q.paid) for q in f.party}
            before = [(q, it) for q in f.party for it in q.unpaid]
            unpaid, extra = self._subtract(before, new)
            for q in f.party:
                q.unpaid = [it for owner, it in unpaid if owner is q]
            f.surplus += extra
            paid = [li for q in f.party for li in q.paid]
            if unpaid:
                conf, tier, eligible, items, reasons, _ = self._assess(f.party, unpaid, paid, f.surplus)
            else:
                conf, tier, eligible, items, reasons = 0.0, None, False, [], []
            what = (f"late receipt ({len(new)} paid item(s) arrived after the decision) covered "
                    f"{len(before) - len(unpaid)} of {len(before)} unpaid item(s)")
            if TIER_RANK[tier] >= TIER_RANK[f.tier]:
                f.payer.log.append(f"{self.now:7.1f}s {what}; decision stays {f.tier}")
                continue
            for q in f.party:
                q.log.append(f"{self.now:7.1f}s {what}: {f.tier} {f.alert.alert_id} -> {tier or 'retracted'}")
            f.tier = tier
            f.decision.update(confidence=conf, tier=tier, alert_eligible=eligible, retracted_t=self.now)
            r = Alert(
                alert_id=f"A{next(self._alert_ids):05d}", person_id=f.alert.person_id,
                tier=tier or "retracted", confidence=conf, t_exit=f.alert.t_exit, t_emitted=self.now,
                unpaid_items=items, reasons=[what] + reasons,
                paid_items=[{"sku": li.sku, "category": li.category} for li in paid],
                visited_register=f.alert.visited_register, group=f.alert.group, basket=f.alert.basket,
                audit_log=self._audit(f.party), retracts=f.alert.alert_id)
            self.alerts.append(r)
            out.append(r)
        self.flagged = [f for f in self.flagged
                        if f.tier is not None and self.now - f.alert.t_exit <= self.cfg.late_receipt_window_s]
        return out

    # ---------------------------------------------------------------- queries

    def basket_of(self, pid: int) -> Counter:
        p = self.people.get(pid)
        return Counter(it.category for it in p.basket) if p else Counter()

    def replay(self, events: Iterable[Event], payments: Iterable[Payment] = ()) -> list[Alert]:
        """Run a full recorded stream (events + payments merged by time)."""
        stream = [(e.t, 0, e) for e in events] + \
            [(pay.t if pay.t_received is None else pay.t_received, 1, pay) for pay in payments]
        stream.sort(key=lambda x: (x[0], x[1]))
        out = []
        for t, _, x in stream:
            # Advance the clock through any decision deadlines before this item,
            # exactly as a live system ticking every frame would.
            while (d := self.next_deadline()) is not None and d <= t:
                before = (self.now, len(self.pending))
                out += self.tick(d)
                if (self.now, len(self.pending)) == before:   # no progress possible: never spin
                    break
            out += self.on_payment(x) if isinstance(x, Payment) else self.on_event(x)
        out += self.finalize()
        return out


def build_ledger(store, **overrides) -> Ledger:
    """Ledger wired to a StoreConfig (terminals, zone kinds, catalog, YAML ledger settings)."""
    known = LedgerConfig.__dataclass_fields__
    settings = {k: v for k, v in {**store.ledger, **overrides}.items() if k in known}
    return Ledger(store.catalog, LedgerConfig(**settings),
                  terminal_zones=store.terminals, zone_kinds=store.zone_kinds)
