#!/usr/bin/env python3
"""Regression tests for default scanner universe selection."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scanner  # noqa: E402


def run_with_stubs(discover: bool = False, watchlist=None, max_symbols=200):
    calls = {"finviz": 0, "prefilter_watchlist": None, "batch_downloads": []}
    original_finviz = scanner.get_finviz_watchlist
    original_batch = scanner._batch_download
    original_prefilter = scanner._prefilter_stock_universe
    original_background = scanner._ensure_background_refresh_started

    def fake_finviz():
        calls["finviz"] += 1
        return ["DYN1", "DYN2"]

    def fake_batch(tickers, period, interval):
        calls["batch_downloads"].append((tuple(tickers), period, interval))
        return {}

    def fake_prefilter(watchlist, daily_data, trusted_options_symbols=None):
        calls["prefilter_watchlist"] = list(watchlist)
        return [], [{"ticker": t, "reason": "no price data"} for t in watchlist]

    scanner.get_finviz_watchlist = fake_finviz
    scanner._batch_download = fake_batch
    scanner._prefilter_stock_universe = fake_prefilter
    scanner._ensure_background_refresh_started = lambda: None
    try:
        rows, near_miss, meta = scanner.scan_all(watchlist=watchlist, discover=discover, max_symbols=max_symbols)
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
    assert meta["partial_result"] is False
    assert meta["partial_result_reasons"] == []
    assert "performance" in meta
    assert meta["performance"]["peak_worker_count"] == 12
    assert meta["performance"]["market_data_engine"]["requests"] == 0
    assert meta["performance"]["market_data_engine"]["incremental_updates_used"] == 0
    assert [item[1:] for item in calls["batch_downloads"]] == [("1y", "1d")]

    calls, rows, near_miss, meta = run_with_stubs(discover=True)
    assert calls["finviz"] == 1
    assert calls["prefilter_watchlist"] == ["DYN1", "DYN2"]
    assert rows == []
    assert near_miss == []
    assert meta["configured_universe_count"] == 2

    custom_symbols = [f"T{i}" for i in range(250)]
    calls, rows, near_miss, meta = run_with_stubs(watchlist=custom_symbols)
    assert len(calls["prefilter_watchlist"]) == 200
    assert calls["prefilter_watchlist"] == custom_symbols[:200]
    assert meta["configured_universe_count"] == 200

    calls, rows, near_miss, meta = run_with_stubs(watchlist=custom_symbols, max_symbols=None)
    assert len(calls["prefilter_watchlist"]) == 250
    assert calls["prefilter_watchlist"] == custom_symbols
    assert meta["configured_universe_count"] == 250

    assert scanner._analysis_cache_key(None) == ("default",)
    assert scanner._analysis_cache_key(None, discover=True) == ("discover",)
    discovered_key = scanner._analysis_cache_key(["AAPL", "MSFT"], universe="discovered")
    assert discovered_key == ("universe", "discovered", ("AAPL", "MSFT"))
    assert discovered_key != scanner._analysis_cache_key(["AAPL", "MSFT"])

    original_raw = scanner._download_price_batch_raw
    try:
        raw_calls = []

        def fake_raw(tickers, period, interval, provider=None):
            raw_calls.append((list(tickers), period, interval, getattr(provider, "name", None)))
            return {}

        scanner._download_price_batch_raw = fake_raw
        scanner._cache_snapshot(reset=True)
        result = scanner._batch_download(["AAPL", "AAPL", "MSFT", "MSFT"], period="1y", interval="1d")
        stats = scanner._cache_snapshot(reset=True)
        assert result == {}
        assert raw_calls and raw_calls[0][0] == ["AAPL", "MSFT"]
        assert stats["prices_duplicate_symbols_eliminated"] == 2
        assert stats["prices_request_1d_count"] == 1
        assert stats["prices_request_1d_symbols"] == 2
        assert stats["prices_miss"] == 2
    finally:
        scanner._download_price_batch_raw = original_raw

    print("Default scan universe v1 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
