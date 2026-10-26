#!/usr/bin/env python3
"""Record rolling segments from a station DVR. Thin wrapper: the implementation is bree.ingest.record.

  PYTHONPATH=src python scripts/imou_record.py --station trumanrd --channels 7,29,12 --hours 12

Same as `bree record`. Read-only Imou APIs, credentials from ~/.config/bree/imou.env, stops by itself when the
platform serves stub video (flow allowance used up). See docs/ultra/platform.md.
"""
import argparse
import sys

from bree.stationcli import register


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    register(sub)
    sys.argv.insert(1, "record")
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
