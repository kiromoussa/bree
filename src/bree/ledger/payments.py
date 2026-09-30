"""Payment inputs: how POS receipts and cooler card/RFID taps reach the ledger.

Wire format: one JSON object per payment, e.g.

    {"terminal": "pos_1", "ts": 1767000000.25, "txn_id": "T1042", "method": "card",
     "items": [{"sku": "COKE-20OZ", "qty": 1}, {"sku": "SNICKERS", "qty": 2}]}

Time is either `ts` (unix epoch seconds, what a real POS sends) or `t` (seconds
since the video started, for replaying recorded footage). The store config maps
`terminal` to a zone. Items can give `sku` (preferred) or just `category`.

Sources:
  JsonlPayments  - a .jsonl file; replay by timestamp, or tail it live (POS export drop)
  FolderPayments - a folder the POS drops export files into (shadow mode); polled for new files
  StdinPayments  - JSON lines on stdin (`pos_bridge | bree run ... --payments stdin`)
  HttpPayments   - POST /payments on a local port (webhook from a POS / tap reader)
  MockPayments   - an in-memory list (tests, simulator)
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Protocol

from bree.events.types import LineItem, Payment


def parse_payment(d: dict, stream_start_wall: float | None = None) -> Payment:
    if "t" in d:
        t = float(d["t"])
    elif "ts" in d and stream_start_wall is not None:
        t = float(d["ts"]) - stream_start_wall
    else:
        raise ValueError("payment needs 't' (stream seconds) or 'ts' (epoch) with a known stream start")
    items = [LineItem(sku=i.get("sku"), category=i.get("category"), qty=int(i.get("qty", 1)))
             for i in d.get("items", [])]
    if not d.get("terminal"):
        raise ValueError("payment needs a 'terminal'")
    return Payment(t=t, terminal=d["terminal"], items=items, txn_id=str(d.get("txn_id", "")),
                   method=d.get("method", "card"), person_id=d.get("person_id"))


class PaymentSource(Protocol):
    def poll(self, t: float) -> list[Payment]:
        """Payments that are known by stream time t (not returned before)."""
        ...


class MockPayments:
    def __init__(self, payments: list[Payment]):
        self._pending = sorted(payments, key=lambda p: p.t)

    def poll(self, t: float) -> list[Payment]:
        out = [p for p in self._pending if p.t <= t]
        self._pending = self._pending[len(out):]
        return out


class _QueueSource:
    """Shared base for live sources: a background thread fills a queue."""

    def __init__(self, stream_start_wall: float | None):
        self.stream_start_wall = stream_start_wall
        self.q: queue.Queue = queue.Queue()
        self.errors: list[str] = []
        self._held: list[Payment] = []

    def _ingest_line(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        try:
            data = json.loads(line)
            for d in data if isinstance(data, list) else [data]:
                self.q.put(parse_payment(d, self.stream_start_wall))
        except (ValueError, json.JSONDecodeError) as e:
            self.errors.append(f"bad payment {line[:80]!r}: {e}")

    def poll(self, t: float) -> list[Payment]:
        while True:
            try:
                self._held.append(self.q.get_nowait())
            except queue.Empty:
                break
        out = [p for p in self._held if p.t <= t]
        self._held = [p for p in self._held if p.t > t]
        return out


class JsonlPayments(_QueueSource):
    def __init__(self, path: str | Path, stream_start_wall: float | None = None, follow: bool = False):
        super().__init__(stream_start_wall)
        self.path = Path(path)
        if follow:
            threading.Thread(target=self._tail, daemon=True).start()
        else:
            for line in self.path.read_text().splitlines():
                self._ingest_line(line)

    def _tail(self) -> None:
        with self.path.open() as f:
            while True:
                line = f.readline()
                if line:
                    self._ingest_line(line)
                else:
                    time.sleep(0.2)


class FolderPayments(_QueueSource):
    """Every *.jsonl / *.json file in `folder`, same line format as above (a line may also be a list).

    Scanned at most every `every_s` seconds from `poll()`, so no thread. Picks up new files and
    lines appended to files it has seen. A line is read once it ends in a newline, or once the
    file stopped growing between two scans (last line written without one). Files already in the
    folder at start are skipped: they are receipts from before the camera was running.
    Only payments at `terminals` are returned (the ones this camera's store config knows).
    """

    def __init__(self, folder: str | Path, stream_start_wall: float | None = None, every_s: float = 2.0,
                 terminals=None, skip_existing: bool = True):
        super().__init__(stream_start_wall)
        self.folder = Path(folder)
        self.every_s = every_s
        self.terminals = set(terminals) if terminals else None
        self._read: dict[str, int] = {}     # file name -> bytes consumed
        self._size: dict[str, int] = {}     # file name -> size at the previous scan
        self._next_scan = 0.0
        if skip_existing:
            for f in self._files():
                self._read[f.name] = self._size[f.name] = f.stat().st_size

    def _files(self) -> list[Path]:
        return sorted(f for f in self.folder.glob("*") if f.suffix in (".jsonl", ".json") and f.is_file())

    def scan(self) -> None:
        for f in self._files():
            size, done = f.stat().st_size, self._read.get(f.name, 0)
            if size < done:                  # truncated or replaced: read it again
                done = 0
            stable = self._size.get(f.name) == size
            self._size[f.name] = size
            if size == done:
                continue
            with f.open("rb") as fh:
                fh.seek(done)
                chunk = fh.read(size - done)
            end = len(chunk) if stable else chunk.rfind(b"\n") + 1
            for line in chunk[:end].decode("utf-8", "replace").splitlines():
                self._ingest_line(line)
            self._read[f.name] = done + end

    def poll(self, t: float) -> list[Payment]:
        if time.monotonic() >= self._next_scan:
            self.scan()
            self._next_scan = time.monotonic() + self.every_s
        return [p for p in super().poll(t) if self.terminals is None or p.terminal in self.terminals]


class StdinPayments(_QueueSource):
    def __init__(self, stream_start_wall: float | None = None):
        super().__init__(stream_start_wall)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in sys.stdin:
            self._ingest_line(line)


class HttpPayments(_QueueSource):
    """POST JSON (one payment or a list) to http://127.0.0.1:<port>/payments."""

    def __init__(self, port: int = 8765, stream_start_wall: float | None = None, host: str = "127.0.0.1"):
        super().__init__(stream_start_wall)
        src = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 (http.server API)
                if self.path.rstrip("/") != "/payments":
                    self.send_response(404); self.end_headers(); return
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                try:
                    data = json.loads(body)
                    for d in data if isinstance(data, list) else [data]:
                        src.q.put(parse_payment(d, src.stream_start_wall))
                    self.send_response(202)
                except (ValueError, json.JSONDecodeError) as e:
                    src.errors.append(str(e))
                    self.send_response(400)
                self.end_headers()

            def log_message(self, *args):  # keep stdout clean
                pass

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()


def open_payments(spec: str | None, stream_start_wall: float | None = None) -> PaymentSource:
    """`spec`: None | "stdin" | "http" | "http:PORT" | path/to/file.jsonl"""
    if not spec:
        return MockPayments([])
    if spec == "stdin":
        return StdinPayments(stream_start_wall)
    if spec.startswith("http"):
        port = int(spec.split(":", 1)[1]) if ":" in spec else 8765
        return HttpPayments(port, stream_start_wall)
    return JsonlPayments(spec, stream_start_wall)
