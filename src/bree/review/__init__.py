"""Human review feedback loop: review store (SQLite), local reviewer page, label export, metrics.

  store.py     every alert sent to review + every reviewer decision; retention and automatic deletion
  server.py    local reviewer page (stdlib http.server, 127.0.0.1 only)
  labels.py    decisions -> training examples (detector, pick, conceal) + manifest + retrain hook
  metrics.py   precision over time, review rate, time to decision, agreement, weekly owner report
  synthetic.py SYNTHETIC alerts and decisions for tests and the end to end demo

CLI: python -m bree.review --help
"""
from bree.review.store import DECISIONS, ReviewStore  # noqa: F401
