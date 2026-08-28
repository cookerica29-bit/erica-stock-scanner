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
    assert quotes["AAPL"]["price_branch"] == "midpoint"
    assert quotes["MSFT"]["price"] == 402.25
    assert quotes["MSFT"]["price_branch"] == "ask_only"
    assert quotes["NVDA"]["price"] == 203.5
    assert quotes["NVDA"]["price_branch"] == "bid_only"
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
            "price_branch": "midpoint",
            "timestamp": None,
            "source": "alpaca_latest_quote",
        }
    }


def test_latest_quotes_batch_failure_retries_by_halving_not_full_fallback():
    """Verified against a real reproduced failure (2026-08-28): a batch
    doesn't fail because of rate-limiting -- it fails because Alpaca's
    quotes endpoint rejects the WHOLE request the moment it contains even
    one symbol it doesn't recognize (confirmed directly: the same real
    157-symbol chunk succeeded once two known-bad symbols were removed, and
    those two symbols alone reproduced the failure on their own). Falling
    back to one request per symbol in the whole failed batch turned a
    single bad symbol into up to chunk_size sequential requests -- this is
    what made the endpoint slow. Fixed by halving instead: isolates a bad
    symbol in O(log n) requests, not O(n), and only ever falls back to a
    genuine single-symbol request once a batch is already down to size 1.
    """
    provider, calls = provider_with_quote_pages([
        URLError("bad symbol poisoned batch"),          # AAPL,BAD,MSFT -> fail
        {"quotes": {"AAPL": {"bp": 100.0, "ap": 101.0}}},  # AAPL -> ok
        URLError("bad symbol"),                          # BAD,MSFT -> fail
        URLError("bad symbol"),                          # BAD -> fail (size 1, gives up)
        {"quotes": {"MSFT": {"bp": 400.0, "ap": 402.0}}},  # MSFT -> ok
    ])

    old_env = os.environ.get("ALPACA_QUOTE_CHUNK_SIZE")
    os.environ["ALPACA_QUOTE_CHUNK_SIZE"] = "3"
    try:
        quotes = provider.latest_quotes(["AAPL", "BAD", "MSFT"])
    finally:
        if old_env is None:
            os.environ.pop("ALPACA_QUOTE_CHUNK_SIZE", None)
        else:
            os.environ["ALPACA_QUOTE_CHUNK_SIZE"] = old_env

    # 5 requests to fully resolve a 3-symbol batch with 1 bad symbol -- more
    # than the "ideal" 3 individual requests would have been for THIS TINY
    # case, but the win is in how this scales: for a real 157-symbol batch
    # with 1-2 bad symbols, this is ~17 requests instead of ~157 (see the
    # commit message for the real measured numbers this was verified
    # against, not assumed).
    assert [call["symbols"] for call in calls] == ["AAPL,BAD,MSFT", "AAPL", "BAD,MSFT", "BAD", "MSFT"]
    assert quotes["AAPL"]["price"] == 100.5
    assert quotes["MSFT"]["price"] == 401.0
    assert "BAD" not in quotes


def test_latest_quotes_isolates_bad_symbols_in_a_large_batch_with_far_fewer_than_n_requests():
    """The real scaling claim: for a realistically large batch (64 symbols,
    2 bad), request count should be far below 64 (the old full-fallback
    count), not just "less than 64 in this one tiny 3-symbol test above."""
    good = [f"GOOD{i}" for i in range(64)]
    symbols = good[:30] + ["BAD1"] + good[30:62] + ["BAD2"] + good[62:]
    assert len(symbols) == 66

    provider = market_data.AlpacaMarketDataProvider(api_key="key", secret_key="secret")
    calls = []

    def fake_latest_quotes(params):
        calls.append(params["symbols"])
        requested = params["symbols"].split(",")
        if "BAD1" in requested or "BAD2" in requested:
            raise URLError("bad symbol poisoned batch")
        return {"quotes": {s: {"bp": 10.0, "ap": 10.2} for s in requested}}

    provider._request_latest_quotes = fake_latest_quotes

    old_env = os.environ.get("ALPACA_QUOTE_CHUNK_SIZE")
    os.environ["ALPACA_QUOTE_CHUNK_SIZE"] = "66"
    try:
        quotes = provider.latest_quotes(symbols)
    finally:
        if old_env is None:
            os.environ.pop("ALPACA_QUOTE_CHUNK_SIZE", None)
        else:
            os.environ["ALPACA_QUOTE_CHUNK_SIZE"] = old_env

    assert len(quotes) == 64
    assert "BAD1" not in quotes
    assert "BAD2" not in quotes
    assert all(f"GOOD{i}" in quotes for i in range(64))
    # Old behavior: 1 (whole batch) + 66 (every symbol individually) = 67.
    # This isolates 2 bad symbols via halving in well under half that.
    assert len(calls) < 30, f"expected roughly O(log n) requests, got {len(calls)}"


def test_attach_current_quotes_is_display_only():
    rows = [
        {"ticker": "AAPL", "price": 100.0},
        {"ticker": "MSFT", "price": 400.0},
    ]

    class FakeProvider:
        def latest_quotes(self, symbols):
            assert symbols == ["AAPL", "MSFT"]
            return {
                "AAPL": {
                    "price": 100.52,
                    "bid": 100.5,
                    "ask": 100.54,
                    "price_branch": "midpoint",
                    "timestamp": "2026-07-20T15:00:00Z",
                    "source": "alpaca_latest_quote",
                },
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
    assert rows[0]["current_quote_bid"] == 100.5
    assert rows[0]["current_quote_ask"] == 100.54
    assert rows[0]["current_quote_price_branch"] == "midpoint"
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
    test_latest_quotes_batch_failure_retries_by_halving_not_full_fallback()
    test_latest_quotes_isolates_bad_symbols_in_a_large_batch_with_far_fewer_than_n_requests()
    test_attach_current_quotes_is_display_only()
    test_attach_current_quotes_failure_leaves_rows_unchanged()
    print("Current quote price v1 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
