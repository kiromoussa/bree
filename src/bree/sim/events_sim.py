"""Event-level shopper simulator with ground truth.

Two layers, kept strictly apart:

1. WORLD (ground truth). Shopper sessions arrive over simulated store hours:
   normal buyers, browsers who put things back, lingerers, groups where one
   person pays, pocket-then-pay shoppers, and thieves (walkout, conceal +
   partial pay, open carry past the register). One register with a real FIFO
   queue; some shoppers pay by card tap at the cooler. Crowded moments: two
   strangers reaching into the cooler at the same instant.

2. VISION NOISE (what the camera pipeline would report). Every true action
   passes through `VisionNoise`: missed picks, false picks on browse touches,
   missed put-backs, missed/false concealment, category confusions, crowded-
   pick misattribution, track ID switches and swaps, missed exits, POS
   timestamp jitter and dropped POS messages. These rates are ASSUMPTIONS
   (see `VisionNoise` docstrings), not measurements: we have no store footage
   yet. The benchmark reports results under several noise levels for that reason.

The ledger only ever sees layer 2. Scoring compares its alerts to layer 1.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field, replace

import numpy as np

from bree.events.types import Event, EventType as E, LineItem, Payment

# Which zone each category lives in (example store layout).
CATEGORY_ZONE = {
    "soda_bottle": "cooler_bank", "water_bottle": "cooler_bank", "energy_drink": "cooler_bank",
    "beer_pack": "cooler_bank", "candy_bar": "shelf_A", "chips_bag": "shelf_B", "jerky": "shelf_B",
    "phone_accessory": "shelf_C", "lighter": "shelf_C",
}
# How often a normal shopper buys from each category (relative weights).
BUY_WEIGHTS = {"soda_bottle": 5, "water_bottle": 3, "energy_drink": 4, "beer_pack": 2, "candy_bar": 4,
               "chips_bag": 3, "jerky": 1, "phone_accessory": 0.5, "lighter": 1}
# What thieves take (relative weights): small, high-value, easy to pocket.
THEFT_WEIGHTS = {"soda_bottle": 1, "water_bottle": 0.5, "energy_drink": 4, "beer_pack": 2, "candy_bar": 3,
                 "chips_bag": 1, "jerky": 3, "phone_accessory": 2, "lighter": 2}
POCKETABLE = {"energy_drink", "candy_bar", "jerky", "phone_accessory", "lighter"}


# --------------------------------------------------------------------- config

@dataclass
class SimConfig:
    hours: float = 200.0                 # simulated store-open hours
    arrivals_per_hour: float = 40.0      # mean visits/hour (gas-station c-store, daytime)
    rush_fraction: float = 0.2           # share of hours at 2x traffic (crowding, queues)
    rush_multiplier: float = 2.0
    # Session mix (probabilities; groups count as one session of 2-3 people).
    p_normal: float = 0.56
    p_browse_putback: float = 0.12
    p_linger: float = 0.10
    p_group: float = 0.07
    p_pocket_then_pay: float = 0.03
    p_cooler_tap: float = 0.07           # buys only cooler items, pays by tap at the cooler
    p_thief_walkout: float = 0.02        # takes items, never goes to the register
    p_thief_conceal_partial: float = 0.02  # pays for some items, pockets others
    p_thief_open_partial: float = 0.01   # pays for some, walks past with another in hand
    p_crowd_pair: float = 0.15           # a cooler pick is made at the same instant as another shopper's
    payment_mode: str = "pos"


@dataclass
class VisionNoise:
    """Per-event error rates of the perception layer. ASSUMED values, not measured.

    Baseline guesses are for a decent ceiling camera with an unobstructed view
    of the shelves and cooler doors. Real numbers must come from pilot footage.
    """
    p_pick_detected: float = 0.92        # a real pick produces a PICK event
    pick_conf_mean: float = 0.85
    p_false_pick_on_touch: float = 0.05  # a browse touch (no item taken) produces a PICK
    p_putback_detected: float = 0.85     # missed put-back -> phantom item in basket (false-alert risk)
    p_conceal_detected: float = 0.60     # concealment is hard to see
    p_false_conceal: float = 0.03        # per carried item (phone to pocket, hand in pocket...)
    p_category_confusion: float = 0.04   # soda seen as water, etc. (within the same zone)
    p_crowd_candidates: float = 0.90     # crowded pick: engine lists both people as candidates
    p_crowd_swap: float = 0.25           # crowded pick credited to the wrong one of the two
    p_visible_at_exit: float = 0.65      # item carried openly is seen in hand at the door
    p_false_held_at_exit: float = 0.03
    p_register_visit_detected: float = 0.97
    p_exit_detected: float = 0.98        # else track lost at the door -> nobody reconciles
    p_id_switch: float = 0.03            # track splits into two IDs mid-visit
    p_id_swap: float = 0.01              # two overlapping shoppers' IDs get exchanged
    p_pos_dropped: float = 0.01          # POS message never arrives
    pos_jitter_s: float = 2.0            # POS clock vs camera clock

    def scaled(self, k: float) -> "VisionNoise":
        """k=0: perfect vision. k=1: baseline. k=2: every error rate doubled."""
        def err(p):  # error probability scaled, capped
            return min(1.0, p * k)
        return replace(
            self,
            p_pick_detected=1 - err(1 - self.p_pick_detected),
            p_false_pick_on_touch=err(self.p_false_pick_on_touch),
            p_putback_detected=1 - err(1 - self.p_putback_detected),
            p_conceal_detected=1 - err(1 - self.p_conceal_detected),
            p_false_conceal=err(self.p_false_conceal),
            p_category_confusion=err(self.p_category_confusion),
            p_crowd_candidates=1 - err(1 - self.p_crowd_candidates),
            p_crowd_swap=err(self.p_crowd_swap),
            p_visible_at_exit=1 - err(1 - self.p_visible_at_exit),
            p_false_held_at_exit=err(self.p_false_held_at_exit),
            p_register_visit_detected=1 - err(1 - self.p_register_visit_detected),
            p_exit_detected=1 - err(1 - self.p_exit_detected),
            p_id_switch=err(self.p_id_switch),
            p_id_swap=err(self.p_id_swap),
            p_pos_dropped=err(self.p_pos_dropped),
            pos_jitter_s=self.pos_jitter_s * k,
            pick_conf_mean=self.pick_conf_mean if k <= 1 else max(0.6, self.pick_conf_mean - 0.05 * (k - 1)),
        )


# ---------------------------------------------------------------- ground truth

@dataclass
class TrueItem:
    uid: int
    sku: str
    category: str
    zone: str
    t_pick: float
    returned_t: float | None = None
    concealed_t: float | None = None
    paid: bool = False


@dataclass
class TruePerson:
    pid: int                                      # ground-truth person id
    kind: str
    t_enter: float
    t_exit: float = 0.0
    group: int | None = None
    items: list[TrueItem] = field(default_factory=list)
    touches: list[tuple[float, str]] = field(default_factory=list)   # browse touches (nothing taken)
    register: tuple[float, float] | None = None   # (t_arrive_zone, t_leave_zone)
    payments: list[Payment] = field(default_factory=list)

    @property
    def basket(self) -> list[TrueItem]:
        return [i for i in self.items if i.returned_t is None]

    @property
    def stolen(self) -> list[TrueItem]:
        return [i for i in self.basket if not i.paid]


@dataclass
class World:
    people: list[TruePerson]
    payments: list[Payment]
    hours: float
    groups: dict[int, list[int]]


def _choice(rng, weights: dict[str, float], k: int = 1) -> list[str]:
    keys = list(weights)
    p = np.array([weights[c] for c in keys], float)
    return list(rng.choice(keys, size=k, p=p / p.sum()))


def _sku_for(rng, category: str, skus_by_cat: dict[str, list[str]]) -> str:
    return str(rng.choice(skus_by_cat[category]))


class WorldBuilder:
    def __init__(self, cfg: SimConfig, catalog: dict[str, str], seed: int):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.skus_by_cat: dict[str, list[str]] = {}
        for sku, cat in catalog.items():
            self.skus_by_cat.setdefault(cat, []).append(sku)
        self.buy_w = {c: w for c, w in BUY_WEIGHTS.items() if c in self.skus_by_cat}
        self.theft_w = {c: w for c, w in THEFT_WEIGHTS.items() if c in self.skus_by_cat}
        self.people: list[TruePerson] = []
        self.payments: list[Payment] = []
        self.groups: dict[int, list[int]] = {}
        self.queue: list = []
        self.uid = 0
        self.txn = 0

    # -- helpers ------------------------------------------------------------
    def _new_person(self, kind: str, t: float, group: int | None = None) -> TruePerson:
        p = TruePerson(len(self.people) + 1, kind, t, group=group)
        self.people.append(p)
        return p

    def _take(self, p: TruePerson, t: float, category: str) -> TrueItem:
        self.uid += 1
        it = TrueItem(self.uid, _sku_for(self.rng, category, self.skus_by_cat), category,
                      CATEGORY_ZONE[category], t)
        p.items.append(it)
        return it

    def _shop(self, p: TruePerson, t: float, cats: list[str], browse_touches: int = 0) -> float:
        """Walk the store picking `cats` in random order. Returns time done."""
        order = list(cats) + [None] * browse_touches
        self.rng.shuffle(order)
        for c in order:
            t += float(self.rng.uniform(8, 35))                  # walk + look
            if c is None:
                p.touches.append((t, str(self.rng.choice(list(set(CATEGORY_ZONE.values()))))))
            else:
                self._take(p, t, c)
        return t

    def _queue_register(self, payer: TruePerson, t_arrive: float, items: list[TrueItem], then=None) -> None:
        """Join the register queue. Service order is decided later, strictly by arrival time
        (sessions are generated one after another, not in time order). `then(leave_time)`
        runs once the payer has been served; by default the payer walks out."""
        self.queue.append((t_arrive, len(self.queue), payer, items, then or (lambda leave: self._exit(payer, leave))))

    def _serve_queue(self) -> None:
        free_at = 0.0
        for t_arrive, _, payer, items, then in sorted(self.queue, key=lambda q: q[:2]):
            start = max(t_arrive, free_at)
            end = start + float(self.rng.uniform(12, 40)) + 4 * len(items)
            free_at = end
            if items:
                self.txn += 1
                pay = Payment(end - float(self.rng.uniform(1, 4)), "pos_1",
                              [LineItem(sku=i.sku) for i in items], txn_id=f"T{self.txn}")
                payer.payments.append(pay)
                self.payments.append(pay)
                for i in items:
                    i.paid = True
            leave = end + float(self.rng.uniform(1, 3))
            payer.register = (t_arrive, leave)
            then(leave)

    def _exit(self, p: TruePerson, t: float) -> None:
        p.t_exit = t + float(self.rng.uniform(5, 15))

    # -- session types ---------------------------------------------------------
    def session(self, t: float) -> None:
        c, r = self.cfg, self.rng.random()
        kinds = [("normal", c.p_normal), ("browse_putback", c.p_browse_putback), ("linger", c.p_linger),
                 ("group", c.p_group), ("pocket_then_pay", c.p_pocket_then_pay), ("cooler_tap", c.p_cooler_tap),
                 ("thief_walkout", c.p_thief_walkout), ("thief_conceal_partial", c.p_thief_conceal_partial),
                 ("thief_open_partial", c.p_thief_open_partial)]
        total = sum(w for _, w in kinds)
        acc = 0.0
        for kind, w in kinds:
            acc += w / total
            if r <= acc:
                getattr(self, "_s_" + kind)(t)
                return
        self._s_normal(t)

    def _n_items(self) -> int:
        return int(self.rng.choice([1, 1, 1, 2, 2, 3, 4]))

    def _s_normal(self, t):
        p = self._new_person("normal", t)
        done = self._shop(p, t, _choice(self.rng, self.buy_w, self._n_items()), int(self.rng.integers(0, 2)))
        self._queue_register(p, done + 5, p.basket)

    def _s_browse_putback(self, t):
        p = self._new_person("browse_putback", t)
        done = self._shop(p, t, _choice(self.rng, self.buy_w, self._n_items() + 1), int(self.rng.integers(0, 3)))
        # put back 1..n-? items (keep at least zero)
        n_back = int(self.rng.integers(1, len(p.items) + 1))
        for it in list(self.rng.permutation(p.items))[:n_back]:
            it.returned_t = min(done, it.t_pick + float(self.rng.uniform(3, 25)))
        if p.basket:
            self._queue_register(p, done + 5, p.basket)
        else:
            self._exit(p, done)

    def _s_linger(self, t):
        p = self._new_person("linger", t)
        done = self._shop(p, t, [], int(self.rng.integers(1, 5)))
        done += float(self.rng.uniform(30, 300))
        if self.rng.random() < 0.3:          # stands near the counter a while (asks for directions...)
            p.register = (done, done + float(self.rng.uniform(5, 30)))
            done = p.register[1]
        self._exit(p, done)

    def _s_group(self, t):
        gid = len(self.groups) + 1
        n = int(self.rng.choice([2, 2, 3]))
        members = [self._new_person("group", t + float(self.rng.uniform(0, 3)), group=gid) for _ in range(n)]
        self.groups[gid] = [m.pid for m in members]
        ends = [self._shop(m, m.t_enter, _choice(self.rng, self.buy_w, int(self.rng.integers(1, 3)))) for m in members]
        payer = members[0]
        all_items = [i for m in members for i in m.basket]
        t_arrive = max(ends) + 5

        def after(leave):
            for m in members[1:]:
                if self.rng.random() < 0.5:      # companion waits in the register zone too
                    m.register = (t_arrive + float(self.rng.uniform(0, 5)), leave - float(self.rng.uniform(0, 3)))
                self._exit(m, leave + float(self.rng.uniform(-2, 2)))
            self._exit(payer, leave)
        self._queue_register(payer, t_arrive, all_items, after)

    def _s_pocket_then_pay(self, t):
        p = self._new_person("pocket_then_pay", t)
        cats = _choice(self.rng, {c: w for c, w in self.buy_w.items() if c in POCKETABLE}, 1) + \
            _choice(self.rng, self.buy_w, int(self.rng.integers(0, 2)))
        done = self._shop(p, t, cats)
        p.items[0].concealed_t = p.items[0].t_pick + float(self.rng.uniform(2, 10))
        self._queue_register(p, done + 5, p.basket)

    def _s_cooler_tap(self, t):
        p = self._new_person("cooler_tap", t)
        cooler = {c: w for c, w in self.buy_w.items() if CATEGORY_ZONE[c] == "cooler_bank"}
        t_tap = t + float(self.rng.uniform(8, 30))
        cats = _choice(self.rng, cooler, int(self.rng.choice([1, 1, 2])))
        items = [self._take(p, t_tap + float(self.rng.uniform(1, 6)) + 2 * k, c) for k, c in enumerate(cats)]
        self.txn += 1
        # The smart cooler reports what was charged a few seconds after the door closes.
        pay = Payment(max(i.t_pick for i in items) + float(self.rng.uniform(1, 5)), "cooler_tap_1",
                      [LineItem(sku=i.sku) for i in items], txn_id=f"C{self.txn}")
        p.payments.append(pay)
        self.payments.append(pay)
        for i in items:
            i.paid = True
        self._exit(p, max(i.t_pick for i in items) + 5)

    def _s_thief_walkout(self, t):
        p = self._new_person("thief_walkout", t)
        done = self._shop(p, t, _choice(self.rng, self.theft_w, int(self.rng.choice([1, 1, 2, 3]))))
        if self.rng.random() < 0.5:          # half hide it on the way out, half just walk
            for it in p.items:
                if it.category in POCKETABLE:
                    it.concealed_t = it.t_pick + float(self.rng.uniform(1, 8))
        self._exit(p, done)

    def _s_thief_conceal_partial(self, t):
        p = self._new_person("thief_conceal_partial", t)
        pocket = _choice(self.rng, {c: w for c, w in self.theft_w.items() if c in POCKETABLE},
                         int(self.rng.choice([1, 1, 2])))
        buy = _choice(self.rng, self.buy_w, int(self.rng.choice([1, 1, 2])))
        done = self._shop(p, t, buy + pocket)
        hidden = []
        for it in p.items:
            if it.category in pocket and len(hidden) < len(pocket):
                it.concealed_t = it.t_pick + float(self.rng.uniform(1, 10))
                hidden.append(it)
        self._queue_register(p, done + 5, [i for i in p.basket if i not in hidden])

    def _s_thief_open_partial(self, t):
        p = self._new_person("thief_open_partial", t)
        done = self._shop(p, t, _choice(self.rng, self.buy_w, 1) + _choice(self.rng, self.theft_w, 1))
        paid = [p.items[0]]
        self._queue_register(p, done + 5, paid)

    # -- crowding --------------------------------------------------------------
    def make_crowds(self) -> int:
        """Move some cooler picks to coincide with another shopper's cooler pick."""
        cooler_picks = sorted(((it.t_pick, p, it) for p in self.people for it in p.items
                               if it.zone == "cooler_bank" and it.returned_t is None), key=lambda x: x[0])
        n = 0
        for i, (t, p, it) in enumerate(cooler_picks):
            if self.rng.random() >= self.cfg.p_crowd_pair:
                continue
            # another shopper in the store at time t who picks from the cooler within +-60 s
            for t2, q, it2 in cooler_picks[max(0, i - 20): i + 20]:
                if q.pid != p.pid and q.group is None and p.group is None and abs(t2 - t) < 60 \
                        and q.t_enter < t < (q.register[0] if q.register else q.t_exit) and not it2.paid \
                        and q.kind != "cooler_tap" and p.kind != "cooler_tap":
                    new_t = t + float(self.rng.uniform(-0.4, 0.4))
                    if it2.concealed_t is not None:
                        it2.concealed_t = min(it2.concealed_t + new_t - it2.t_pick, q.t_exit - 1)
                    it2.t_pick = new_t
                    n += 1
                    break
        return n

    def build(self) -> World:
        t, hours = 0.0, self.cfg.hours
        end = hours * 3600
        while t < end:
            hour = int(t // 3600)
            rush = (hour * 2654435761 % 1000) / 1000 < self.cfg.rush_fraction   # deterministic rush hours
            rate = self.cfg.arrivals_per_hour * (self.cfg.rush_multiplier if rush else 1) / 3600
            t += float(self.rng.exponential(1 / rate))
            if t < end:
                self.session(t)
        self._serve_queue()
        crowds = self.make_crowds()
        w = World(self.people, sorted(self.payments, key=lambda p: p.t), hours, self.groups)
        w.crowd_pairs = crowds  # type: ignore[attr-defined]
        return w


# ---------------------------------------------------------------- observation

@dataclass
class Observed:
    events: list[Event]
    payments: list[Payment]
    id_map: dict[int, list[int]]          # true person -> observed track ids
    exit_owner: dict[int, int]            # observed id that exited -> true person
    notes: Counter


def observe(world: World, noise: VisionNoise, seed: int, sku_level: bool = False) -> Observed:
    """Turn ground truth into what the vision pipeline + POS feed would report."""
    rng = np.random.default_rng(seed)
    notes = Counter()
    next_id = [1000]

    def new_id():
        next_id[0] += 1
        return next_id[0]

    events: list[Event] = []
    id_map: dict[int, list[int]] = {}
    exit_owner: dict[int, int] = {}
    categories = sorted(CATEGORY_ZONE)
    same_zone = {c: [d for d in categories if CATEGORY_ZONE[d] == CATEGORY_ZONE[c] and d != c] for c in categories}

    def seen_cat(c: str) -> str:
        if same_zone[c] and rng.random() < noise.p_category_confusion:
            notes["category_confusions"] += 1
            return str(rng.choice(same_zone[c]))
        return c

    # Track ID per person over time: [(t_from, obs_id)]
    tracks: dict[int, list[tuple[float, int]]] = {}
    for p in world.people:
        segs = [(p.t_enter, new_id())]
        if rng.random() < noise.p_id_switch:
            ts = float(rng.uniform(p.t_enter, p.t_exit))
            segs.append((ts, new_id()))
            notes["id_switches"] += 1
        tracks[p.pid] = segs
    # ID swaps between shoppers who overlap in time (they cross paths).
    by_time = sorted(world.people, key=lambda p: p.t_enter)
    for i, p in enumerate(by_time):
        for q in by_time[i + 1:i + 6]:
            lo, hi = max(p.t_enter, q.t_enter), min(p.t_exit, q.t_exit)
            if hi - lo > 10 and rng.random() < noise.p_id_swap:
                ts = float(rng.uniform(lo, hi))
                a, b = tracks[p.pid], tracks[q.pid]
                id_a, id_b = _id_at(a, ts), _id_at(b, ts)
                a.append((ts, id_b)); b.append((ts, id_a))
                a.sort(); b.sort()
                notes["id_swaps"] += 1
                break

    def oid(p: TruePerson, t: float) -> int:
        return _id_at(tracks[p.pid], t)

    for p in world.people:
        id_map[p.pid] = sorted({i for _, i in tracks[p.pid]})
        for ts, i in tracks[p.pid]:
            events.append(Event(E.ENTER, ts, i))
        # picks / put-backs / conceals
        for it in p.items:
            if rng.random() < noise.p_pick_detected:
                conf = float(np.clip(rng.normal(noise.pick_conf_mean, 0.07), 0.3, 0.99))
                events.append(Event(E.PICK, it.t_pick, oid(p, it.t_pick), item=seen_cat(it.category),
                                    sku=it.sku if sku_level else None, zone=it.zone, confidence=round(conf, 3),
                                    meta={"true_uid": it.uid}))
            else:
                notes["missed_picks"] += 1
            if it.returned_t is not None:
                if rng.random() < noise.p_putback_detected:
                    events.append(Event(E.PUT_BACK, it.returned_t, oid(p, it.returned_t), item=it.category,
                                        zone=it.zone))
                else:
                    notes["missed_putbacks"] += 1
            if it.concealed_t is not None:
                if rng.random() < noise.p_conceal_detected:
                    events.append(Event(E.CONCEAL, it.concealed_t, oid(p, it.concealed_t), item=it.category,
                                        confidence=round(float(np.clip(rng.normal(0.7, 0.1), 0.3, 0.95)), 3)))
                else:
                    notes["missed_conceals"] += 1
            elif it.returned_t is None and rng.random() < noise.p_false_conceal:
                tc = float(rng.uniform(it.t_pick, p.t_exit))
                events.append(Event(E.CONCEAL, tc, oid(p, tc), item=it.category, confidence=0.6))
                notes["false_conceals"] += 1
        for t, zone in p.touches:
            if rng.random() < noise.p_false_pick_on_touch:
                cats = [c for c in categories if CATEGORY_ZONE[c] == zone] or categories
                events.append(Event(E.PICK, t, oid(p, t), item=str(rng.choice(cats)), zone=zone,
                                    confidence=round(float(np.clip(rng.normal(0.6, 0.1), 0.3, 0.9)), 3)))
                notes["false_picks"] += 1
        # register visit
        if p.register is not None and rng.random() < noise.p_register_visit_detected:
            t0, t1 = p.register
            if t1 - t0 >= 2.5:
                events.append(Event(E.PAY, t0 + 2.5, oid(p, t0 + 2.5), zone="register",
                                    meta={"phase": "start", "t_start": t0}))
                events.append(Event(E.PAY, t1, oid(p, t1), zone="register",
                                    meta={"phase": "end", "t_start": t0, "t_end": t1}))
        # exit
        if rng.random() < noise.p_exit_detected:
            held = []
            for it in p.basket:
                openly = it.concealed_t is None
                if openly and rng.random() < noise.p_visible_at_exit:
                    held.append(it.category)
            if rng.random() < noise.p_false_held_at_exit:
                held.append(str(rng.choice(categories)))
            events.append(Event(E.EXIT, p.t_exit, oid(p, p.t_exit), zone="door", meta={"held_items": held}))
            exit_owner[oid(p, p.t_exit)] = p.pid
        else:
            notes["missed_exits"] += 1

    # Crowded picks: same zone, different people, within 1 s.
    picks = sorted((e for e in events if e.type == E.PICK), key=lambda e: e.t)
    for i, e in enumerate(picks):
        for f in picks[i + 1:]:
            if f.t - e.t > 1.0:
                break
            if f.zone == e.zone and f.person_id != e.person_id:
                notes["crowded_picks"] += 1
                if rng.random() < noise.p_crowd_candidates:
                    e.candidates = [e.person_id, f.person_id]
                    f.candidates = [f.person_id, e.person_id]
                if rng.random() < noise.p_crowd_swap:
                    e.person_id, f.person_id = f.person_id, e.person_id
                    if e.candidates:
                        e.candidates, f.candidates = f.candidates, e.candidates
                    notes["crowd_swaps"] += 1

    # POS feed: jitter, drops.
    pays = []
    for pay in world.payments:
        if rng.random() < noise.p_pos_dropped:
            notes["pos_dropped"] += 1
            continue
        pays.append(Payment(pay.t + float(rng.normal(0, noise.pos_jitter_s / 2)) if noise.pos_jitter_s else pay.t,
                            pay.terminal, pay.items, pay.txn_id, pay.method))
    events.sort(key=lambda e: e.t)
    return Observed(events, pays, id_map, exit_owner, notes)


def _id_at(segs: list[tuple[float, int]], t: float) -> int:
    cur = segs[0][1]
    for ts, i in segs:
        if ts <= t:
            cur = i
    return cur


def build_world(catalog: dict[str, str], cfg: SimConfig | None = None, seed: int = 0) -> World:
    return WorldBuilder(cfg or SimConfig(), catalog, seed).build()
