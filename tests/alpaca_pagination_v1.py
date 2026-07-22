#!/usr/bin/env python3
"""Regression tests for Alpaca historical-bar pagination."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from urllib.error import URLError

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import market_data  # noqa: E402


def bar(day: int, close: float) -> dict:
    return {
        "t": f"2026-07-{day:02d}T13:30:00Z",
        "o": close - 1,
        "h": close + 1,
        "l": close - 2,
        "c": close,
        "v": 1000 + day,
    }


def provider_with_pages(pages):
    provider = market_data.AlpacaMarketDataProvider(api_key="key", secret_key="secret")
    calls = []

    def fake_page(params):
        calls.append(dict(params))
        index = len(calls) - 1
        page = pages[index]
        if isinstance(page, Exception):
            raise page
        return page

    provider._request_bars_page = fake_page
    return provider, calls


def assert_multi_has_symbol(df: pd.DataFrame, symbol: str, count: int) -> None:
    assert isinstance(df.columns, pd.MultiIndex)
    assert symbol in df.columns.get_level_values(0)
    assert len(df[symbol].dropna(how="all")) == count


def test_one_page_response():
    provider, calls = provider_with_pages([
        {"bars": {"AAPL": [bar(1, 200), bar(2, 201)]}},
    ])
    df = provider.download(["AAPL"], period="1y", interval="1d")
    assert len(calls) == 1
    assert "page_token" not in calls[0]
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(df) == 2


def test_two_page_response():
    provider, calls = provider_with_pages([
        {"bars": {"AAPL": [bar(1, 200)]}, "next_page_token": "next-1"},
        {"bars": {"AAPL": [bar(2, 201)]}},
    ])
    df = provider.download(["AAPL"], period="1y", interval="1d")
    assert len(calls) == 2
    assert calls[1]["page_token"] == "next-1"
    assert len(df) == 2


def test_symbols_split_across_pages():
    provider, calls = provider_with_pages([
        {"bars": {"AAPL": [bar(1, 200)]}, "next_page_token": "next-1"},
        {"bars": {"MSFT": [bar(1, 500), bar(2, 501)]}},
    ])
    df = provider.download(["AAPL", "MSFT"], period="1y", interval="1d")
    assert len(calls) == 2
    assert_multi_has_symbol(df, "AAPL", 1)
    assert_multi_has_symbol(df, "MSFT", 2)


def test_repeated_token_guard_fails_closed():
    provider, calls = provider_with_pages([
        {"bars": {"AAPL": [bar(1, 200)]}, "next_page_token": "repeat"},
        {"bars": {"AAPL": [bar(2, 201)]}, "next_page_token": "repeat"},
    ])
    df = provider.download(["AAPL", "MSFT"], period="1y", interval="1d")
    assert len(calls) == 2
    assert df.empty
    assert isinstance(df.columns, pd.MultiIndex)


def test_max_page_guard_fails_closed():
    previous = os.environ.get("ALPACA_MAX_PAGES")
    os.environ["ALPACA_MAX_PAGES"] = "1"
    try:
        provider, calls = provider_with_pages([
            {"bars": {"AAPL": [bar(1, 200)]}, "next_page_token": "next-1"},
        ])
        df = provider.download(["AAPL", "MSFT"], period="1y", interval="1d")
        assert len(calls) == 1
        assert df.empty
        assert isinstance(df.columns, pd.MultiIndex)
    finally:
        if previous is None:
            os.environ.pop("ALPACA_MAX_PAGES", None)
        else:
            os.environ["ALPACA_MAX_PAGES"] = previous


def test_provider_error_mid_pagination_fails_closed():
    provider, calls = provider_with_pages([
        {"bars": {"AAPL": [bar(1, 200)]}, "next_page_token": "next-1"},
        URLError("timeout"),
    ])
    df = provider.download(["AAPL", "MSFT"], period="1y", interval="1d")
    assert len(calls) == 2
    assert df.empty
    assert isinstance(df.columns, pd.MultiIndex)


def test_large_multi_symbol_download_chunks_requests_and_retains_successes():
    previous = os.environ.get("ALPACA_BAR_SYMBOL_CHUNK_SIZE")
    os.environ["ALPACA_BAR_SYMBOL_CHUNK_SIZE"] = "1"
    try:
        provider, calls = provider_with_pages([
            {"bars": {"AAPL": [bar(1, 200), bar(2, 201)]}},
            URLError("timeout"),
        ])
        market_data.reset_provider_metrics()
        df = provider.download(["AAPL", "MSFT"], period="2y", interval="1wk")
        assert len(calls) == 2
        assert calls[0]["symbols"] == "AAPL"
        assert calls[1]["symbols"] == "MSFT"
        assert_multi_has_symbol(df, "AAPL", 2)
        assert "MSFT" not in df.columns.get_level_values(0)
        metrics = market_data.provider_metrics_snapshot()
        assert metrics["alpaca_bar_requests"] == 2
        assert metrics["alpaca_bar_symbols_requested"] == 2
        assert metrics["alpaca_bar_symbols_succeeded"] == 1
        assert metrics["alpaca_bar_symbols_failed"] == 1
    finally:
        if previous is None:
            os.environ.pop("ALPACA_BAR_SYMBOL_CHUNK_SIZE", None)
        else:
            os.environ["ALPACA_BAR_SYMBOL_CHUNK_SIZE"] = previous


def test_page_ceiling_env_is_bounded():
    previous = os.environ.get("ALPACA_BARS_MAX_PAGES")
    os.environ["ALPACA_BARS_MAX_PAGES"] = "9999"
    try:
        assert market_data._parse_bounded_positive_int_env(
            "ALPACA_BARS_MAX_PAGES",
            market_data.DEFAULT_ALPACA_MAX_PAGES,
            market_data.MAX_ALPACA_MAX_PAGES,
        ) == market_data.MAX_ALPACA_MAX_PAGES
    finally:
        if previous is None:
            os.environ.pop("ALPACA_BARS_MAX_PAGES", None)
        else:
            os.environ["ALPACA_BARS_MAX_PAGES"] = previous


def main() -> int:
    test_one_page_response()
    test_two_page_response()
    test_symbols_split_across_pages()
    test_repeated_token_guard_fails_closed()
    test_max_page_guard_fails_closed()
    test_provider_error_mid_pagination_fails_closed()
    test_large_multi_symbol_download_chunks_requests_and_retains_successes()
    test_page_ceiling_env_is_bounded()
    print("Alpaca pagination v1 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
