"""POS CSV exports -> payments, through a small declarative mapping (YAML).

Real POS systems export CSV with their own column names, one row per line item. The mapping says
which column is what, how to read the time, and how POS item names map to our catalog. See
configs/pos_mapping_example.yaml for every key. Rows are grouped into one payment per
(terminal, receipt id); its time is the receipt's latest row. Output is the payment wire format
of bree.ledger.payments (`ts` = unix seconds), so it goes through the same `parse_payment`.

Clocks: a POS prints local wall time. `time.timezone` (IANA name, DST handled) turns it into unix
time, then `time.offset_s` is added: edge box clock minus POS clock, in seconds. The ledger runs
on the edge box's clock (stream time = ts - the box's time.time() at stream start), so after the
offset a receipt lands at the moment the camera saw the sale. Ambiguous local times (the repeated
hour when DST ends) resolve to the first occurrence.
"""
from __future__ import annotations

import csv
import fnmatch
import io
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

UNMAPPED = ("keep", "skip", "error")


def load_mapping(path: str | Path) -> dict:
    m = yaml.safe_load(Path(path).read_text())
    cols = m.get("columns") or {}
    for need in ("time", "receipt", "sku"):
        if not cols.get(need):
            raise ValueError(f"POS mapping {path}: columns.{need} is required")
    if not cols.get("terminal") and not m.get("terminal"):
        raise ValueError(f"POS mapping {path}: give columns.terminal or a constant `terminal`")
    if m.get("unmapped", "keep") not in UNMAPPED:
        raise ValueError(f"POS mapping {path}: unmapped must be one of {UNMAPPED}")
    ZoneInfo(m.get("time", {}).get("timezone", "UTC"))     # fail now on a misspelt zone
    return m


def _epoch(value: str, tcfg: dict) -> float:
    fmt = tcfg.get("format", "iso")
    if fmt == "epoch":
        dt_ts = float(value)
    else:
        dt = datetime.fromisoformat(value) if fmt == "iso" else datetime.strptime(value, fmt)
        if dt.tzinfo is None:                              # local POS wall time
            dt = dt.replace(tzinfo=ZoneInfo(tcfg.get("timezone", "UTC")))
        dt_ts = dt.timestamp()
    return dt_ts + float(tcfg.get("offset_s", 0))


def _lookup(name: str, items: dict):
    """Exact entry first, then glob entries (`UNLEADED*`) in file order."""
    if name in items:
        return items[name]
    return next((v for k, v in items.items() if "*" in k and fnmatch.fnmatchcase(name, k)), None)


def _item(name: str, m: dict, unmapped: set | None) -> dict | None:
    """POS item value -> {"sku": ...} or {"category": ...}; None = not merchandise (skip)."""
    target = _lookup(name, m.get("items") or {})
    if target is None:
        if unmapped is not None:
            unmapped.add(name)
        how = m.get("unmapped", "keep")
        if how == "error":
            raise ValueError(f"POS item {name!r} is not in the mapping")
        return None if how == "skip" else {"sku": name}
    if target == "skip":
        return None
    return dict(target) if isinstance(target, dict) else {"sku": str(target)}


def _num(s: str) -> float:
    return float(s.replace("$", "").replace(",", "").strip())


def csv_payments(text: str, m: dict, unmapped: set | None = None, keep_empty: bool = False) -> list[dict]:
    """CSV text -> payment dicts in file order (one per receipt). `unmapped` collects POS item
    names that had no mapping entry (for a warning). Receipts left with no items (fully voided,
    fuel only) are dropped unless `keep_empty`."""
    fmt, cols, tcfg = m.get("csv") or {}, m["columns"], m.get("time") or {}
    lines = text.splitlines(keepends=True)[int(fmt.get("skip_rows", 0)):]
    reader = csv.DictReader(io.StringIO("".join(lines)), delimiter=fmt.get("delimiter", ","))
    wanted = [c for v in cols.values() for c in (v if isinstance(v, list) else [v]) if c]
    missing = [c for c in wanted if c not in (reader.fieldnames or [])]
    if missing:
        raise ValueError(f"CSV has no column(s) {missing}; header is {reader.fieldnames}")
    receipts: dict[tuple, dict] = {}
    for row in reader:
        if not any(isinstance(v, str) and v.strip() for v in row.values()):
            continue
        get = lambda key: (row.get(cols[key]) or "").strip() if cols.get(key) else ""   # noqa: E731
        tcol = cols["time"]
        stamp = " ".join((row.get(c) or "").strip() for c in tcol) if isinstance(tcol, list) else get("time")
        term = get("terminal")
        term = (m.get("terminals") or {}).get(term, term) or m.get("terminal")
        key = (term, get("receipt"))
        r = receipts.setdefault(key, {"terminal": term, "ts": float("-inf"), "txn_id": key[1], "items": {}})
        r["ts"] = max(r["ts"], _epoch(stamp, tcfg))
        item = _item(get("sku"), m, unmapped)
        if item is None:
            continue
        ikey = item.get("sku") or "cat:" + item["category"]
        line = r["items"].setdefault(ikey, {**item, "qty": 0})
        line["qty"] += int(round(_num(get("qty")))) if cols.get("qty") else 1    # voids are negative
        if cols.get("price") and get("price") and "price" not in line:
            line["price"] = _num(get("price"))
    out = []
    for r in receipts.values():
        r["items"] = [i for i in r["items"].values() if i["qty"] > 0]
        if r["items"] or keep_empty:                       # fully voided / fuel-only: not a payment here
            out.append(r)
    return out


def read_csv_payments(path: str | Path, m: dict, unmapped: set | None = None) -> list[dict]:
    enc = (m.get("csv") or {}).get("encoding", "utf-8-sig")
    return csv_payments(Path(path).read_text(encoding=enc), m, unmapped)
