"""Payment inputs: parsing, JSONL replay, local HTTP endpoint."""
from __future__ import annotations

import json
import time
import urllib.request

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
