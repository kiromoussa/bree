"""Watchdog wrapper: run a command, restart-by-exit when it hangs. For systemd units with WatchdogSec=.

    python -m bree.trust.supervise --heartbeat /path/hb --stall-s 120 -- bree shadow --config ...

The child proves it is making progress by touching the heartbeat file (bree.trust.supervise.Heartbeat.beat, called from
the pipeline's frame loop at most every few seconds). This wrapper sends READY=1 to systemd once the child has
beaten (or after `--ready-s`), and WATCHDOG=1 at half the watchdog interval ONLY while the heartbeat is fresh.
A child that is alive but stuck (a hung RTSP read, a deadlock, a GPU that stopped answering) stops the pings, and
systemd kills and restarts the unit; the wrapper also terminates the child itself after `--stall-s` and exits 75.
A child that exits is mirrored (its exit code), so Restart=on-failure does the rest.
Without NOTIFY_SOCKET (run by hand) the pings are skipped and the stall rule still applies.
"""
from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


class Heartbeat:
    """`beat()` is cheap: it touches the file at most once per `every_s`. `Heartbeat.from_env()` reads
    BREE_HEARTBEAT_FILE (set by the wrapper) and is a no-op when unset."""

    def __init__(self, path: str | Path | None, every_s: float = 5.0):
        self.path, self.every_s, self._t = (Path(path) if path else None), every_s, 0.0

    @classmethod
    def from_env(cls) -> "Heartbeat":
        return cls(os.environ.get("BREE_HEARTBEAT_FILE"))

    def beat(self) -> None:
        if self.path is None:
            return
        now = time.monotonic()
        if now - self._t >= self.every_s:
            self._t = now
            try:
                self.path.touch()
            except OSError:
                pass

    def age(self) -> float:
        try:
            return time.time() - self.path.stat().st_mtime
        except (OSError, AttributeError):
            return float("inf")


def sd_notify(msg: str) -> bool:
    """Send a message to systemd's notify socket (stdlib only). False when not under systemd."""
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return False
    if addr.startswith("@"):
        addr = "\0" + addr[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.connect(addr)
            s.sendall(msg.encode())
        return True
    except OSError:
        return False


def run(cmd: list[str], heartbeat: Path, stall_s: float = 120, ready_s: float = 30, poll_s: float = 1.0) -> int:
    hb = Heartbeat(heartbeat)
    heartbeat.parent.mkdir(parents=True, exist_ok=True)
    heartbeat.touch()
    wd_us = int(os.environ.get("WATCHDOG_USEC", "0") or 0)
    ping_every = (wd_us / 1e6) / 2 if wd_us else 0
    child = subprocess.Popen(cmd, env={**os.environ, "BREE_HEARTBEAT_FILE": str(heartbeat)})

    def _term(signum, frame):
        child.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, _term)
    t0, last_ping, ready = time.time(), 0.0, False
    while True:
        rc = child.poll()
        if rc is not None:
            return rc
        age = hb.age()
        fresh = age < stall_s
        if not ready and (time.time() - t0 > ready_s or age < poll_s + 1):
            ready = sd_notify("READY=1") or True
        if fresh and ping_every and time.time() - last_ping >= ping_every:
            sd_notify("WATCHDOG=1")
            last_ping = time.time()
        if not fresh and time.time() - t0 > stall_s:
            sd_notify("STATUS=stalled: no heartbeat, restarting")
            child.terminate()
            try:
                child.wait(15)
            except subprocess.TimeoutExpired:
                child.kill()
            return 75
        time.sleep(poll_s)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--heartbeat", required=True)
    ap.add_argument("--stall-s", type=float, default=120)
    ap.add_argument("--ready-s", type=float, default=30)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("give the command after --")
    sys.exit(run(cmd, Path(a.heartbeat), a.stall_s, a.ready_s))


if __name__ == "__main__":
    main()
