#!/usr/bin/env python3
"""Regression tests for default scanner universe selection."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scanner  # noqa: E402


def run_with_stubs(discover: bool = False):
    calls = {"finviz": 0, "prefilter_watchlist": None}
    original_finviz = scanner.get_finviz_watchlist
    original_batch = scanner._batch_download
    original_prefilter = scanner._prefilter_stock_universe
    original_background = scanner._ensure_background_refresh_started

    def fake_finviz():
        calls["finviz"] += 1
        return ["DYN1", "DYN2"]

    def fake_batch(tickers, period, interval):
        return {}

    def fake_prefilter(watchlist, daily_data):
        calls["prefilter_watchlist"] = list(watchlist)
        return [], [{"ticker": t, "reason": "no price data"} for t in watchlist]

    scanner.get_finviz_watchlist = fake_finviz
    scanner._batch_download = fake_batch
    scanner._prefilter_stock_universe = fake_prefilter
    scanner._ensure_background_refresh_started = lambda: None
    try:
        rows, near_miss, meta = scanner.scan_all(discover=discover)
        return calls, rows, near_miss, meta
    finally:
        scanner.get_finviz_watchlist = original_finviz
        scanner._batch_download = original_batch
        scanner._prefilter_stock_universe = original_prefilter
        scanner._ensure_background_refresh_started = original_background


def main() -> int:
    calls, rows, near_miss, meta = run_with_stubs(discover=False)
    assert calls["finviz"] == 0
    assert calls["prefilter_watchlist"] == scanner.WATCHLIST
    assert rows == []
    assert near_miss == []
    assert meta["configured_universe_count"] == len(scanner.WATCHLIST)

    calls, rows, near_miss, meta = run_with_stubs(discover=True)
    assert calls["finviz"] == 1
    assert calls["prefilter_watchlist"] == ["DYN1", "DYN2"]
    assert rows == []
    assert near_miss == []
    assert meta["configured_universe_count"] == 2

    assert scanner._analysis_cache_key(None) == ("default",)
    assert scanner._analysis_cache_key(None, discover=True) == ("discover",)

    print("Default scan universe v1 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
