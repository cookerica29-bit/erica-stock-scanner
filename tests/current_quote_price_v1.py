#!/usr/bin/env python3
"""Regression tests for display-only current quote enrichment."""

from __future__ import annotations

import sys
import os
from pathlib import Path
from urllib.error import URLError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import market_data  # noqa: E402
import scanner  # noqa: E402


def provider_with_quote_pages(pages):
    provider = market_data.AlpacaMarketDataProvider(api_key="key", secret_key="secret")
    calls = []

    def fake_latest_quotes(params):
        calls.append(dict(params))
        page = pages[len(calls) - 1]
        if isinstance(page, Exception):
            raise page
        return page

    provider._request_latest_quotes = fake_latest_quotes
    return provider, calls


def test_latest_quotes_uses_midpoint_then_ask_then_bid():
    provider, calls = provider_with_quote_pages([
        {
            "quotes": {
                "AAPL": {"bp": 100.0, "ap": 101.0, "t": "2026-07-20T15:00:00Z"},
                "MSFT": {"bp": 0, "ap": 402.25, "t": "2026-07-20T15:00:01Z"},
                "NVDA": {"bp": 203.5, "ap": 0, "t": "2026-07-20T15:00:02Z"},
                "TSLA": {"bp": 0, "ap": 0, "t": "2026-07-20T15:00:03Z"},
            }
        }
    ])

    quotes = provider.latest_quotes(["AAPL", "MSFT", "NVDA", "TSLA"])

    assert len(calls) == 1
    assert calls[0]["symbols"] == "AAPL,MSFT,NVDA,TSLA"
    assert quotes["AAPL"]["price"] == 100.5
    assert quotes["MSFT"]["price"] == 402.25
    assert quotes["NVDA"]["price"] == 203.5
    assert "TSLA" not in quotes
    assert quotes["AAPL"]["source"] == "alpaca_latest_quote"
    assert quotes["AAPL"]["timestamp"] == "2026-07-20T15:00:00Z"


def test_latest_quotes_chunks_requests():
    provider, calls = provider_with_quote_pages([
        {"quotes": {"AAPL": {"bp": 100.0, "ap": 101.0}}},
        {"quotes": {"MSFT": {"bp": 400.0, "ap": 402.0}}},
    ])

    old_env = os.environ.get("ALPACA_QUOTE_CHUNK_SIZE")
    os.environ["ALPACA_QUOTE_CHUNK_SIZE"] = "1"
    try:
        quotes = provider.latest_quotes(["AAPL", "MSFT"])
    finally:
        if old_env is None:
            os.environ.pop("ALPACA_QUOTE_CHUNK_SIZE", None)
        else:
            os.environ["ALPACA_QUOTE_CHUNK_SIZE"] = old_env

    assert len(calls) == 2
    assert calls[0]["symbols"] == "AAPL"
    assert calls[1]["symbols"] == "MSFT"
    assert quotes["AAPL"]["price"] == 100.5
    assert quotes["MSFT"]["price"] == 401.0


def test_latest_quotes_failure_returns_available_quotes_only():
    provider, calls = provider_with_quote_pages([
        {"quotes": {"AAPL": {"bp": 100.0, "ap": 101.0}}},
        URLError("timeout"),
    ])

    old_env = os.environ.get("ALPACA_QUOTE_CHUNK_SIZE")
    os.environ["ALPACA_QUOTE_CHUNK_SIZE"] = "1"
    try:
        quotes = provider.latest_quotes(["AAPL", "MSFT"])
    finally:
        if old_env is None:
            os.environ.pop("ALPACA_QUOTE_CHUNK_SIZE", None)
        else:
            os.environ["ALPACA_QUOTE_CHUNK_SIZE"] = old_env

    assert len(calls) == 2
    assert quotes == {
        "AAPL": {
            "price": 100.5,
            "bid": 100.0,
            "ask": 101.0,
            "timestamp": None,
            "source": "alpaca_latest_quote",
        }
    }


def test_attach_current_quotes_is_display_only():
    rows = [
        {"ticker": "AAPL", "price": 100.0},
        {"ticker": "MSFT", "price": 400.0},
    ]

    class FakeProvider:
        def latest_quotes(self, symbols):
            assert symbols == ["AAPL", "MSFT"]
            return {
                "AAPL": {"price": 100.52, "timestamp": "2026-07-20T15:00:00Z", "source": "alpaca_latest_quote"},
                "MSFT": {"price": None},
            }

    original = scanner.build_market_data_provider
    scanner.build_market_data_provider = lambda name=None: FakeProvider()
    try:
        scanner._attach_current_quotes(rows)
    finally:
        scanner.build_market_data_provider = original

    assert rows[0]["price"] == 100.0
    assert rows[0]["current_quote_price"] == 100.52
    assert rows[0]["current_quote_source"] == "alpaca_latest_quote"
    assert rows[0]["current_quote_timestamp"] == "2026-07-20T15:00:00Z"
    assert rows[1]["price"] == 400.0
    assert "current_quote_price" not in rows[1]


def test_attach_current_quotes_failure_leaves_rows_unchanged():
    rows = [{"ticker": "AAPL", "price": 100.0}]

    class FailingProvider:
        def latest_quotes(self, symbols):
            raise RuntimeError("provider unavailable")

    original = scanner.build_market_data_provider
    scanner.build_market_data_provider = lambda name=None: FailingProvider()
    try:
        scanner._attach_current_quotes(rows)
    finally:
        scanner.build_market_data_provider = original

    assert rows == [{"ticker": "AAPL", "price": 100.0}]


def main() -> int:
    test_latest_quotes_uses_midpoint_then_ask_then_bid()
    test_latest_quotes_chunks_requests()
    test_latest_quotes_failure_returns_available_quotes_only()
    test_attach_current_quotes_is_display_only()
    test_attach_current_quotes_failure_leaves_rows_unchanged()
    print("Current quote price v1 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
