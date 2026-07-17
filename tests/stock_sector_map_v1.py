#!/usr/bin/env python3
"""Regression tests for static stock-sector capture."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scanner  # noqa: E402


def main() -> int:
    assert scanner._sector_for_ticker("AAPL") == "Technology"
    assert scanner._sector_for_ticker("spy") == "ETF"
    assert scanner._sector_for_ticker("UNKNOWN") is None

    missing = [
        symbol
        for symbol in scanner.WATCHLIST
        if not scanner.STOCK_SECTOR_MAP.get(str(symbol or "").strip().upper())
    ]
    if missing:
        raise AssertionError(f"WATCHLIST symbols missing sector map entries: {', '.join(missing)}")

    print("Stock sector map v1 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
