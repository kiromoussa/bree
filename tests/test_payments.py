"""Payment inputs: parsing, JSONL replay, local HTTP endpoint."""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import pytest

from bree.ledger.payments import HttpPayments, JsonlPayments, MockPayments, open_payments, parse_payment
from bree.events.types import LineItem, Payment


def test_parse_stream_time_and_epoch():
    p = parse_payment({"terminal": "pos_1", "t": 12.5, "items": [{"sku": "COKE", "qty": 2}]})
    assert p.t == 12.5 and p.items[0].qty == 2
    q = parse_payment({"terminal": "pos_1", "ts": 1000.0, "items": []}, stream_start_wall=990.0)
    assert q.t == pytest.approx(10.0)
    with pytest.raises(ValueError):
        parse_payment({"terminal": "pos_1", "ts": 1000.0})          # epoch without a stream start
    with pytest.raises(ValueError):
        parse_payment({"t": 1.0})                                   # no terminal


def test_jsonl_replay_releases_by_time(tmp_path):
    f = tmp_path / "pay.jsonl"
    f.write_text("\n".join(json.dumps({"terminal": "pos_1", "t": t, "items": [{"sku": "X"}]}) for t in (5, 1, 9))
                 + "\nnot json\n")
    src = JsonlPayments(f)
    assert [p.t for p in src.poll(4)] == [1]
    assert sorted(p.t for p in src.poll(10)) == [5, 9]
    assert src.poll(100) == [] and len(src.errors) == 1


def test_mock_payments():
    src = MockPayments([Payment(3, "pos_1", [LineItem(sku="A")]), Payment(1, "pos_1", [])])
    assert [p.t for p in src.poll(2)] == [1] and [p.t for p in src.poll(3)] == [3]


def test_http_endpoint():
    src = HttpPayments(port=0)
    try:
        body = json.dumps([{"terminal": "pos_1", "t": 2.0, "items": [{"sku": "COKE"}]},
                           {"terminal": "cooler_tap_1", "t": 3.0, "items": [{"sku": "REDBULL"}]}]).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{src.port}/payments", data=body, method="POST")
        assert urllib.request.urlopen(req, timeout=5).status == 202
        time.sleep(0.05)
        got = src.poll(10)
        assert [p.terminal for p in got] == ["pos_1", "cooler_tap_1"]
    finally:
        src.close()


def test_open_payments_none_is_empty():
    assert open_payments(None).poll(1e9) == []


# ------------------------------------------------------------------ POS CSV via a column mapping

ROOT = Path(__file__).resolve().parents[1]
MAPPING = ROOT / "configs" / "pos_mapping_example.yaml"
SAMPLE = ROOT / "tests" / "fixtures" / "pos_sample.csv"
# What the sample CSV says, written by hand in the JSONL wire format. 2:15:05 PM EDT = 18:15:05 UTC.
SAMPLE_JSONL = [
    {"terminal": "pos_1", "ts": 1790878505.0, "txn_id": "10421",
     "items": [{"sku": "COKE-20OZ", "qty": 1}, {"sku": "SNICKERS", "qty": 2}]},
    {"terminal": "pos_1", "ts": 1790878664.0, "txn_id": "10422",
     "items": [{"sku": "REDBULL-12OZ", "qty": 1}, {"category": "soda_bottle", "qty": 1}]},   # fuel line skipped
    {"terminal": "pos_1", "ts": 1790878812.0, "txn_id": "10423", "items": [{"sku": "BIC-LIGHTER", "qty": 1}]},
]                                                          # 10423's chips voided; 10424 is fuel only


def test_csv_gives_the_same_payments_as_jsonl(tmp_path):
    from bree.ledger.pos_csv import load_mapping, read_csv_payments
    start = 1790878500.0
    got = [parse_payment(d, start) for d in read_csv_payments(SAMPLE, load_mapping(MAPPING))]
    f = tmp_path / "pay.jsonl"
    f.write_text("".join(json.dumps(d) + "\n" for d in SAMPLE_JSONL))
    assert got == JsonlPayments(f, stream_start_wall=start).poll(1e9)
    assert [p.t for p in got] == [5.0, 164.0, 312.0]


def test_csv_offset_constant_terminal_and_unmapped():
    from bree.ledger.pos_csv import csv_payments
    m = {"columns": {"time": "When", "receipt": "Rcpt", "sku": "Item"}, "terminal": "pos_1",
         "time": {"format": "iso", "timezone": "America/Los_Angeles", "offset_s": -2.5},
         "items": {"COKE": "COKE-20OZ"}, "unmapped": "skip"}
    text = "When,Rcpt,Item\n2026-01-15 09:00:00,7,COKE\n2026-01-15 09:00:01,7,MYSTERY\n"
    unmapped: set = set()
    (d,) = csv_payments(text, m, unmapped)
    assert d["terminal"] == "pos_1" and d["items"] == [{"sku": "COKE-20OZ", "qty": 1}]
    assert d["ts"] == 1768496401.0 - 2.5                    # 09:00:01 PST = 17:00:01 UTC, plus the offset
    assert unmapped == {"MYSTERY"}


def test_pos_convert_cli(capsys):
    from bree.cli import main
    main(["pos-convert", "--mapping", str(MAPPING), str(SAMPLE)])
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [r["txn_id"] for r in rows] == ["10421", "10422", "10423"] and rows[0]["items"][0]["price"] == 2.49
