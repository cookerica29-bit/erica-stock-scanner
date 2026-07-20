#!/usr/bin/env python3
"""Regression tests for Yahoo option-expiration rate-limit handling."""

from __future__ import annotations

from datetime import datetime, timedelta
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scanner  # noqa: E402


class RateLimitedTicker:
    @property
    def options(self):
        raise RuntimeError("Too Many Requests. Rate limited. Try after a while.")


class TransientFailTicker:
    @property
    def options(self):
        raise RuntimeError("temporary provider failure")


class EmptyOptionsTicker:
    @property
    def options(self):
        return []


class GoodOptionsTicker:
    @property
    def options(self):
        return ["2026-08-28", "2026-09-18"]


class FakeYF:
    def __init__(self, ticker_class):
        self.ticker_class = ticker_class

    def Ticker(self, _ticker):
        return self.ticker_class()


def reset_options_state():
    with scanner._cache_lock:
        scanner._option_chain_cache.clear()
        scanner._cache_stats.clear()
        scanner._option_yahoo_backoff_until = None


def daily_df():
    return pd.DataFrame({"Close": [100.0], "Volume": [1_000_000]})


def test_rate_limit_preserves_existing_good_expirations():
    reset_options_state()
    original_yf = scanner.yf
    scanner.yf = FakeYF(RateLimitedTicker)
    fetched_at = datetime.utcnow() - timedelta(minutes=30)
    with scanner._cache_lock:
        scanner._option_chain_cache["AAPL"] = {
            "fetched_at": fetched_at,
            "expirations_fetched_at": fetched_at,
            "expirations_status": "ready",
            "expirations": ["2026-08-28"],
            "chains": {},
        }
    try:
        expirations = scanner._fetch_option_expirations("AAPL")
        assert expirations == ["2026-08-28"]
        known, cached = scanner._cached_option_expirations_for_ticker("AAPL")
        assert known is True
        assert cached == ["2026-08-28"]
    finally:
        scanner.yf = original_yf


def test_rate_limit_without_prior_data_is_unknown_not_no_options():
    reset_options_state()
    original_yf = scanner.yf
    original_submit = scanner._submit_background_job
    submitted = []
    scanner.yf = FakeYF(RateLimitedTicker)
    scanner._submit_background_job = lambda *args, **kwargs: submitted.append(args) or True
    try:
        expirations = scanner._fetch_option_expirations("ZZZ")
        assert expirations == []
        with scanner._cache_lock:
            cached = scanner._option_chain_cache["ZZZ"]
            assert cached["expirations_status"] == "unknown"
            assert cached["expirations"] == []
            assert cached["expirations_failure_reason"] == "rate_limited"
        known, cached_expirations = scanner._cached_option_expirations_for_ticker("ZZZ")
        assert known is False
        assert cached_expirations == []
        reason = scanner._stock_universe_skip_reason("ZZZ", daily_df())
        assert reason is None
    finally:
        scanner.yf = original_yf
        scanner._submit_background_job = original_submit


def test_confirmed_empty_expirations_still_means_no_options():
    reset_options_state()
    original_yf = scanner.yf
    scanner.yf = FakeYF(EmptyOptionsTicker)
    try:
        expirations = scanner._fetch_option_expirations("NOOPT")
        assert expirations == []
        with scanner._cache_lock:
            assert scanner._option_chain_cache["NOOPT"]["expirations_status"] == "empty"
        known, cached = scanner._cached_option_expirations_for_ticker("NOOPT")
        assert known is True
        assert cached == []
        assert scanner._stock_universe_skip_reason("NOOPT", daily_df()) == "no options"
    finally:
        scanner.yf = original_yf


def test_successful_expiration_fetch_uses_long_availability_ttl():
    reset_options_state()
    original_yf = scanner.yf
    scanner.yf = FakeYF(GoodOptionsTicker)
    try:
        expirations = scanner._fetch_option_expirations("MSFT")
        assert expirations == ["2026-08-28", "2026-09-18"]
        with scanner._cache_lock:
            cached = scanner._option_chain_cache["MSFT"]
            assert cached["expirations_status"] == "ready"
            cached["expirations_fetched_at"] = datetime.utcnow() - timedelta(hours=12)
        known, cached_expirations = scanner._cached_option_expirations_for_ticker("MSFT")
        assert known is True
        assert cached_expirations == ["2026-08-28", "2026-09-18"]
        assert scanner._option_expirations_for_ticker("MSFT") == ["2026-08-28", "2026-09-18"]
    finally:
        scanner.yf = original_yf


def test_backoff_prevents_prefilter_from_submitting_more_yahoo_jobs():
    reset_options_state()
    original_submit = scanner._submit_background_job
    submitted = []
    scanner._submit_background_job = lambda *args, **kwargs: submitted.append(args) or True
    try:
        scanner._mark_option_yahoo_rate_limited()
        reason = scanner._stock_universe_skip_reason("PENDING", daily_df())
        assert reason is None
        assert submitted == []
    finally:
        scanner._submit_background_job = original_submit


def test_unknown_expiration_failure_retries_after_short_ttl():
    reset_options_state()
    original_yf = scanner.yf
    original_submit = scanner._submit_background_job
    submitted = []
    scanner.yf = FakeYF(TransientFailTicker)
    scanner._submit_background_job = lambda *args, **kwargs: submitted.append(args) or True
    try:
        expirations = scanner._fetch_option_expirations("RETRY")
        assert expirations == []
        with scanner._cache_lock:
            cached = scanner._option_chain_cache["RETRY"]
            assert cached["expirations_status"] == "unknown"
            assert cached["expirations_failure_reason"] == "fetch_error"
        assert scanner._stock_universe_skip_reason("RETRY", daily_df()) is None
        assert submitted == []
        with scanner._cache_lock:
            scanner._option_chain_cache["RETRY"]["expirations_failure_at"] = (
                datetime.utcnow() - scanner.OPTION_EXPIRATION_FAILURE_RETRY_TTL - timedelta(seconds=1)
            )
        assert scanner._stock_universe_skip_reason("RETRY", daily_df()) is None
        assert len(submitted) == 1
    finally:
        scanner.yf = original_yf
        scanner._submit_background_job = original_submit


def main() -> int:
    test_rate_limit_preserves_existing_good_expirations()
    test_rate_limit_without_prior_data_is_unknown_not_no_options()
    test_confirmed_empty_expirations_still_means_no_options()
    test_successful_expiration_fetch_uses_long_availability_ttl()
    test_backoff_prevents_prefilter_from_submitting_more_yahoo_jobs()
    test_unknown_expiration_failure_retries_after_short_ttl()
    print("Options expiration rate-limit v1 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
