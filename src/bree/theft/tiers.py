"""POS-less mode: no receipt feed, so every flag is for a human reviewer, never for staff.

Without receipts the ledger cannot know what was paid for. The `dwell` payment mode assumes a register visit paid for
everything picked before it, and `max_tier: review` stops the ledger from ever raising an alert. What is left are
review items, split into bands by the evidence behind them (the band is the answer to "why am I looking at this"):

  R1  concealment seen: something that was picked went into a pocket, bag or waistband, and was not covered by a visit
      to the register. Strongest evidence we can produce from vision alone.
  R2  carried out: no concealment, but a picked item was visible in hand at the exit and the person never stood at the
      register. Walking out holding an unbought item.
  R3  unverified: a pick (or several) and no register visit, nothing else seen. This is also what a missed put-back, an
      item left on the counter, or a paid-by-someone-else item looks like. Lowest band, kept so that a reviewer can
      see the whole picture; it is the band that carries most of the false items.

Nothing here says "theft". A band means "a person should look at this clip". The numbers per band come from the
simulator (`bree.theft.eval_sim`, SIMULATED) until a store's reviewers have labelled real items.
"""
from __future__ import annotations

from bree.alerts.types import Alert

BANDS = ("R1", "R2", "R3")
BAND_TEXT = {
    "R1": "concealment seen and not covered by a register visit",
    "R2": "item in hand at the exit, never at the register",
    "R3": "picked, never at the register, nothing else seen (may be a missed put-back)",
}
POSLESS_LEDGER = {"payment_mode": "dwell", "max_tier": "review"}   # LedgerConfig overrides


def band_of(alert: Alert) -> str | None:
    """Evidence band of a review-tier alert (None for a retraction that withdrew it completely)."""
    if alert.tier == "retracted":
        return None
    items = alert.unpaid_items
    if any(i.concealed for i in items):
        return "R1"
    if any(i.held_at_exit for i in items) and not alert.visited_register:
        return "R2"
    return "R3"


def describe(alert: Alert) -> str:
    b = band_of(alert)
    if b is None:
        return "withdrawn"
    cats = ", ".join(sorted({i.category for i in alert.unpaid_items}))
    return f"{b}: {BAND_TEXT[b]} ({cats})"
