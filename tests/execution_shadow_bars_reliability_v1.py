"""_recent_4h_bars_for_execution_shadow reliability: logging, single retry,
short TTL cache -- candidates_router.py.

Real motivation (confirmed live, 2026-08-31): one production request for
ticker S returned <15 4H bars ("Need 15 recent 4H bars for execution
check"); a second request 6 minutes later for the same ticker/window
returned the full 15 and passed. The function had zero logging anywhere in
its path before this change -- no way to tell transient flakiness from a
systematic issue after the fact, and no historical evidence either way (see
the constants' own comments in candidates_router.py). This adds:

  - logger.warning on the empty-download branch (exception, or an empty/None
    response) and the under-15-bars branch, both with ticker + bar count --
    ships regardless of the other two, since it's what makes this failure
    mode observable at all going forward.
  - a single retry, after EXECUTION_SHADOW_BARS_RETRY_DELAY_SECONDS, when the
    first attempt comes back under 15 bars -- mirrors the existing
    OPTION_EXPIRATION_EMPTY_RETRY_DELAY_SECONDS precedent in scanner.py
    (empty/thin single-entity response -> log -> sleep -> retry once).
  - a short (EXECUTION_SHADOW_BARS_CACHE_TTL) process-wide cache on the raw
    bars, keyed by ticker, so a burst of requests for the same ticker within
    that window isn't each independently exposed to single-call flakiness.

Tests below control time.sleep (no real waiting) and the cache's stored
timestamp directly (no datetime mocking) -- exercised against the real
module state and functions, not reimplemented copies of the logic.
"""

import logging
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import candidates_router as router  # noqa: E402


def _bars_df(n: int, start: str = "2026-08-30T00:00:00Z") -> pd.DataFrame:
    index = pd.date_range(start=start, periods=n, freq="4h", tz="UTC")
    return pd.DataFrame(
        {
            "Open": [100.0 + i for i in range(n)],
            "High": [101.0 + i for i in range(n)],
            "Low": [99.0 + i for i in range(n)],
            "Close": [100.5 + i for i in range(n)],
            "Volume": [1000 + i for i in range(n)],
        },
        index=index,
    )


def _install_provider(monkeypatch, results):
    """Fake AlpacaMarketDataProvider -- returns/raises the next queued
    result on each .download() call, in order. A NEW instance is
    constructed on every real call site (AlpacaMarketDataProvider()...), so
    the queue/call-log live in the closure, not on the instance."""
    queue = list(results)
    calls: list = []

    class _FakeProvider:
        def download(self, tickers, period=None, interval=None, auto_adjust=None):
            calls.append(tickers)
            result = queue.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

    monkeypatch.setattr(router, "AlpacaMarketDataProvider", lambda: _FakeProvider())
    return calls


def _no_sleep(monkeypatch):
    sleeps: list = []
    monkeypatch.setattr(router.time, "sleep", lambda seconds: sleeps.append(seconds))
    return sleeps


@pytest.fixture(autouse=True)
def _clear_execution_shadow_bars_cache():
    router._execution_shadow_bars_cache.clear()
    yield
    router._execution_shadow_bars_cache.clear()


def test_fetch_once_download_exception_logs_ticker_and_zero_bars(monkeypatch, caplog):
    _install_provider(monkeypatch, [RuntimeError("boom")])
    with caplog.at_level(logging.WARNING, logger="candidates_router"):
        result = router._fetch_recent_4h_bars_once("XYZ")
    assert result == []
    assert "XYZ" in caplog.text
    assert "bars=0" in caplog.text


def test_fetch_once_empty_dataframe_logs_ticker_and_zero_bars(monkeypatch, caplog):
    _install_provider(monkeypatch, [pd.DataFrame()])
    with caplog.at_level(logging.WARNING, logger="candidates_router"):
        result = router._fetch_recent_4h_bars_once("ABC")
    assert result == []
    assert "ABC" in caplog.text
    assert "bars=0" in caplog.text


def test_fetch_once_none_result_logs_ticker_and_zero_bars(monkeypatch, caplog):
    _install_provider(monkeypatch, [None])
    with caplog.at_level(logging.WARNING, logger="candidates_router"):
        result = router._fetch_recent_4h_bars_once("NONE")
    assert result == []
    assert "NONE" in caplog.text
    assert "bars=0" in caplog.text


def test_recent_bars_retries_once_when_first_attempt_is_short(monkeypatch, caplog):
    calls = _install_provider(monkeypatch, [_bars_df(8), _bars_df(15)])
    sleeps = _no_sleep(monkeypatch)
    with caplog.at_level(logging.WARNING, logger="candidates_router"):
        result = router._recent_4h_bars_for_execution_shadow("RETRY")
    assert len(calls) == 2
    assert len(result) == 15
    assert sleeps == [router.EXECUTION_SHADOW_BARS_RETRY_DELAY_SECONDS]
    assert "RETRY" in caplog.text
    assert "bars=8" in caplog.text
    assert "need=15" in caplog.text


def test_recent_bars_does_not_retry_a_second_time(monkeypatch):
    # Both attempts come back short -- exactly one retry, never a loop.
    calls = _install_provider(monkeypatch, [_bars_df(3), _bars_df(5)])
    _no_sleep(monkeypatch)
    result = router._recent_4h_bars_for_execution_shadow("STILLSHORT")
    assert len(calls) == 2
    assert len(result) == 5


def test_recent_bars_first_attempt_at_threshold_skips_retry(monkeypatch):
    calls = _install_provider(monkeypatch, [_bars_df(15)])
    _no_sleep(monkeypatch)
    result = router._recent_4h_bars_for_execution_shadow("CLEAN")
    assert len(calls) == 1
    assert len(result) == 15


def test_recent_bars_cache_hit_skips_second_live_call(monkeypatch):
    calls = _install_provider(monkeypatch, [_bars_df(15)])
    _no_sleep(monkeypatch)
    first = router._recent_4h_bars_for_execution_shadow("CACHED")
    second = router._recent_4h_bars_for_execution_shadow("CACHED")
    assert len(calls) == 1
    assert first == second


def test_recent_bars_caches_a_short_result_too(monkeypatch):
    # Even a still-short result (after its one retry) is cached briefly --
    # a second request for the same ticker within the TTL shouldn't pile on
    # more live calls on top of the one retry it already spent.
    calls = _install_provider(monkeypatch, [_bars_df(3), _bars_df(3)])
    _no_sleep(monkeypatch)
    router._recent_4h_bars_for_execution_shadow("BOTHSHORT")
    assert len(calls) == 2
    router._recent_4h_bars_for_execution_shadow("BOTHSHORT")
    assert len(calls) == 2


def test_recent_bars_cache_expires_after_ttl(monkeypatch):
    calls = _install_provider(monkeypatch, [_bars_df(15), _bars_df(15)])
    _no_sleep(monkeypatch)
    router._recent_4h_bars_for_execution_shadow("EXPIRING")
    assert len(calls) == 1
    fetched_at, bars = router._execution_shadow_bars_cache["EXPIRING"]
    router._execution_shadow_bars_cache["EXPIRING"] = (
        fetched_at - router.EXECUTION_SHADOW_BARS_CACHE_TTL - timedelta(seconds=1),
        bars,
    )
    router._recent_4h_bars_for_execution_shadow("EXPIRING")
    assert len(calls) == 2


def test_recent_bars_cache_is_per_ticker(monkeypatch):
    calls = _install_provider(monkeypatch, [_bars_df(15), _bars_df(15)])
    _no_sleep(monkeypatch)
    router._recent_4h_bars_for_execution_shadow("AAA")
    router._recent_4h_bars_for_execution_shadow("BBB")
    assert len(calls) == 2


