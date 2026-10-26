"""Operator product: store signals, metrics, digests, notifications and the owner dashboard.

Everything here reads one small contract, the signal log (see signals.py and docs/ultra/product.md),
so any track (perception, shelf, theft, simulator) can feed it without importing this package.
"""
