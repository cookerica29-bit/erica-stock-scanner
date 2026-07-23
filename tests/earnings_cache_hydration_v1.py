#!/usr/bin/env python3
"""Regression tests for earnings cache preservation and scan-row hydration."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scanner  # noqa: E402


def main() -> int:
    original_ticker = scanner.yf.Ticker
    original_submit = scanner._submit_background_job
    try:
        with scanner._cache_lock:
            scanner._earnings_cache.clear()
            scanner._earnings_deferred_until.clear()

        dated = {
            "loaded": True,
            "date": "2026-07-30",
            "days_until": 13,
            "source": "yfinance",
        }
        with scanner._cache_lock:
            scanner._earnings_cache["AAPL"] = {
                "fetched_at": datetime.utcnow(),
                "data": dated,
            }

        class FailingTicker:
            def __init__(self, ticker):
                self.ticker = ticker

            def get_earnings_dates(self, limit=1):
                raise RuntimeError("provider timeout")

            @property
            def calendar(self):
                raise RuntimeError("provider timeout")

        scanner.yf.Ticker = FailingTicker
        preserved = scanner._fetch_earnings_for_ticker("AAPL")
        assert preserved == dated

        with scanner._cache_lock:
            scanner._earnings_cache["AAPL"] = {
                "fetched_at": datetime.utcnow() - scanner.EARNINGS_FAILURE_PRESERVE_TTL - timedelta(seconds=1),
                "data": dated,
            }

        expired = scanner._fetch_earnings_for_ticker("AAPL")
        assert expired.get("loaded") is False
        assert expired.get("date") is None
        assert expired.get("source") == "unavailable"

        with scanner._cache_lock:
            scanner._earnings_cache["AAPL"] = {
                "fetched_at": datetime.utcnow() - scanner.EARNINGS_UNAVAILABLE_CACHE_TTL + timedelta(seconds=1),
                "data": expired,
            }
        cached = scanner._earnings_for_ticker("AAPL", allow_fetch=False)
        assert cached.get("source") == "unavailable"

        submit_calls = []

        def fake_submit(key, callback, *args, **kwargs):
            submit_calls.append((key, args))
            return True

        scanner._submit_background_job = fake_submit

        with scanner._cache_lock:
            scanner._earnings_cache["AAPL"] = {
                "fetched_at": datetime.utcnow() - scanner.EARNINGS_UNAVAILABLE_CACHE_TTL - timedelta(seconds=1),
                "data": expired,
            }
        stale = scanner._earnings_for_ticker("AAPL", allow_fetch=False)
        assert stale.get("source") == "unavailable"
        assert submit_calls and submit_calls[-1][0] == ("earnings", "AAPL")
        submit_calls.clear()

        with scanner._cache_lock:
            scanner._earnings_cache.clear()
            scanner._earnings_deferred_until.clear()
            scanner._cache_stats.clear()
        scanner._scan_activity_started()
        try:
            deferred = scanner._earnings_for_ticker("NVDA", allow_fetch=False)
        finally:
            scanner._scan_activity_finished()
        assert deferred.get("loading") is True
        assert submit_calls == []
        with scanner._cache_lock:
            assert scanner._earnings_deferred_until.get("NVDA") is not None
            assert scanner._cache_stats.get("earnings_deferred_scan") == 1

        submitted = scanner._earnings_for_ticker("NVDA", allow_fetch=False)
        assert submitted.get("loading") is True
        assert submit_calls and submit_calls[-1][0] == ("earnings", "NVDA")

        with scanner._cache_lock:
            scanner._earnings_cache.clear()
            scanner._earnings_cache["MSFT"] = {
                "fetched_at": datetime.utcnow(),
                "data": {
                    "loaded": True,
                    "date": "2026-07-29",
                    "days_until": 12,
                    "source": "yfinance",
                },
            }

        rows = [{
            "ticker": "MSFT",
            "earnings": {
                "loaded": False,
                "loading": True,
                "status": "loading",
                "date": None,
                "days_until": None,
                "source": "background_refresh",
            },
        }]
        hydrated = scanner._hydrate_scan_rows_from_cache(rows)
        assert hydrated[0]["earnings"]["loaded"] is True
        assert hydrated[0]["earnings"]["date"] == "2026-07-29"
        assert hydrated[0]["earnings"]["source"] == "yfinance"

        already_dated = [{
            "ticker": "MSFT",
            "earnings": {
                "loaded": True,
                "date": "2026-08-01",
                "days_until": 15,
                "source": "scan",
            },
        }]
        hydrated = scanner._hydrate_scan_rows_from_cache(already_dated)
        assert hydrated[0]["earnings"]["date"] == "2026-08-01"
        assert hydrated[0]["earnings"]["source"] == "scan"

        print("Earnings cache hydration v1 tests passed")
        return 0
    finally:
        scanner.yf.Ticker = original_ticker
        scanner._submit_background_job = original_submit
        with scanner._cache_lock:
            scanner._earnings_cache.clear()
            scanner._earnings_deferred_until.clear()


if __name__ == "__main__":
    raise SystemExit(main())
