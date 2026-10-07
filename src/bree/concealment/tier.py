"""The tier rule of the concealment cue, applied to the ledger's finished records (off unless called).

ALERT only when all three hold, otherwise the record stays where the ledger put it (review):
  1. count reconciliation says unpaid: the record lists an unpaid item (the ledger made it, nothing is added here);
  2. the concealment score is high: an unpaid item of the record is marked concealed (a cue of bree.concealment.cue
     landed on it) and, when the per-shopper scores are given, the score of a person of the record is at or above `bar`;
  3. identity is not uncertain: the ledger's log of the record has no "identity uncertain" line.

The ledger has this rule already and two more caps: "a paid item did not match the basket" and "a pick that also fits
another person". This function leaves those two out, which is the rule as the brief states it. It never lowers a tier
and never makes a record. Whether to use it instead of the ledger's caps is a decision for whoever owns the ledger:
docs/fragments/concealment has the measured difference on thieves and on honest shoppers (SIMULATED clips).
"""
from __future__ import annotations


def identity_uncertain(alert: dict) -> bool:
    return any("identity uncertain" in line for line in alert.get("audit_log") or [])


def retier(alerts: list[dict], scores: dict | None = None, bar: float = 0.6, ignore_identity: bool = False) -> list[dict]:
    """alerts: Alert.to_dict() rows as written to pipeline/alerts.jsonl. scores: {person_id: {"score": ...}} from
    bree.concealment.cue.conceal_cues, or None to go by the concealed mark alone. -> new rows (the input is not changed).
    ignore_identity is for measuring a bound only."""
    scores = {int(k): v for k, v in (scores or {}).items()} if scores is not None else None
    out = []
    for a in alerts:
        a = dict(a)
        if a.get("tier") == "review" and not a.get("retracts") and any(i.get("concealed") for i in a.get("unpaid_items") or []):
            s = None if scores is None else max((scores.get(int(p), {}).get("score", 0.0) for p in [a["person_id"], *(a.get("group") or [])]), default=0.0)
            if (s is None or s >= bar) and (ignore_identity or not identity_uncertain(a)):
                a["tier"] = "alert"
                a["reasons"] = [*a.get("reasons", []), "raised to alert (bree.concealment.tier): unpaid, an unpaid item was concealed"
                                + (f" (shopper concealment score {s:.2f})" if s is not None else "") + ", identity not uncertain"]
        out.append(a)
    return out
