"""Core event types shared by the vision engine, the simulator, and the ledger.

Everything downstream of perception speaks in these events. That is the point:
the ledger does not care whether an event came from a camera or from the
simulator, so we can test the theft logic without any video at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class EventType(str, Enum):
    ENTER = "enter"          # person first seen inside the store
    PICK = "pick"            # item taken off a shelf / out of a cooler
    PUT_BACK = "put_back"    # item returned to a shelf / cooler
    CONCEAL = "conceal"      # held item disappeared near torso/pocket/bag away from register
    PAY = "pay"              # person dwelled at register (visit), or a matched POS/tap payment
    EXIT = "exit"            # person left through the door


@dataclass
class Event:
    type: EventType
    t: float                         # seconds since stream start (camera clock)
    person_id: int
    item: str | None = None          # item category (what vision can see), e.g. "soda_can"
    sku: str | None = None           # exact SKU, when known (POS / RFID / fine-grained model)
    zone: str | None = None          # zone name where it happened
    confidence: float = 1.0          # detector/rule confidence in [0, 1]
    # For crowded picks: every person who could plausibly have taken the item.
    # The event is attributed to person_id (best guess); candidates lists all of them.
    candidates: list[int] = field(default_factory=list)
    frame: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ambiguous(self) -> bool:
        return len(self.candidates) > 1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = self.type.value
        return d

    @staticmethod
    def from_dict(d: dict) -> "Event":
        d = dict(d)
        d["type"] = EventType(d["type"])
        return Event(**d)


@dataclass
class LineItem:
    sku: str | None = None
    category: str | None = None
    qty: int = 1


@dataclass
class Payment:
    """A payment from an external source (POS register, card/RFID tap at a cooler).

    `terminal` names where it happened; the store config maps terminals to zones
    so the ledger can work out *who* was standing there. `person_id` is only set
    when the source already knows it (e.g. the simulator, or a future app-based
    checkout).
    """
    t: float
    terminal: str
    items: list[LineItem]
    txn_id: str = ""
    method: str = "card"
    person_id: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Payment":
        d = dict(d)
        d["items"] = [LineItem(**li) for li in d.get("items", [])]
        return Payment(**d)


class Catalog:
    """Maps SKUs to the coarse categories that the vision system can recognise.

    Vision sees "a soda can", the POS sees "COKE-12OZ". Reconciliation has to
    meet in the middle, so every SKU belongs to exactly one category.
    """

    def __init__(self, sku_to_category: dict[str, str]):
        self.sku_to_category = dict(sku_to_category)

    def category_of(self, sku: str | None, fallback: str | None = None) -> str | None:
        if sku is None:
            return fallback
        return self.sku_to_category.get(sku, fallback)

    @property
    def categories(self) -> list[str]:
        return sorted(set(self.sku_to_category.values()))
