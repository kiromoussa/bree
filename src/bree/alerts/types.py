"""Alert records produced by the ledger."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class UnpaidItem:
    category: str
    sku: str | None
    t_pick: float
    zone: str | None
    pick_confidence: float
    concealed: bool
    held_at_exit: bool
    ambiguous_with: list[int]
    score: float                   # this item's evidence score in [0, 1]


@dataclass
class Alert:
    alert_id: str
    person_id: int
    tier: str                      # "alert" (notify staff now) or "review" (manager reviews later)
    confidence: float
    t_exit: float                  # camera time the person left
    t_emitted: float               # camera time the ledger emitted this alert
    unpaid_items: list[UnpaidItem]
    reasons: list[str]
    paid_items: list[dict[str, Any]] = field(default_factory=list)
    visited_register: bool = False
    group: list[int] = field(default_factory=list)
    clip_path: str | None = None
    basket: list[str] = field(default_factory=list)       # everything the party picked (categories)
    audit_log: list[str] = field(default_factory=list)    # the ledger's reasoning, line by line

    @property
    def latency_s(self) -> float:
        return self.t_emitted - self.t_exit

    def to_dict(self) -> dict:
        d = asdict(self)
        d["latency_s"] = round(self.latency_s, 3)
        return d
