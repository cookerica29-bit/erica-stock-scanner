#!/usr/bin/env python3
"""Regression tests for discovered-universe optionability provenance."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scanner  # noqa: E402


def daily_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "Open": [100.0] * 30,
        "High": [101.0] * 30,
        "Low": [99.0] * 30,
        "Close": [100.0] * 30,
        "Volume": [5_000_000] * 30,
    })


def test_discovered_optionability_bypasses_yahoo_expiration_negative():
    previous_cached = scanner._cached_option_expirations_for_ticker
    previous_submit = scanner._submit_background_job
    calls = []
    try:
        scanner._cached_option_expirations_for_ticker = lambda ticker: calls.append(ticker) or (True, [])
        scanner._submit_background_job = lambda *args, **kwargs: False

        reason = scanner._stock_universe_skip_reason("AAPL", daily_frame(), trusted_options_eligible=True)
        assert reason is None
        assert calls == []
    finally:
        scanner._cached_option_expirations_for_ticker = previous_cached
        scanner._submit_background_job = previous_submit


def test_default_watchlist_still_rejects_confirmed_no_options():
    previous_cached = scanner._cached_option_expirations_for_ticker
    previous_submit = scanner._submit_background_job
    try:
        scanner._cached_option_expirations_for_ticker = lambda ticker: (True, [])
        scanner._submit_background_job = lambda *args, **kwargs: False

        reason = scanner._stock_universe_skip_reason("AAPL", daily_frame(), trusted_options_eligible=False)
        assert reason == "no options"
    finally:
        scanner._cached_option_expirations_for_ticker = previous_cached
        scanner._submit_background_job = previous_submit


def test_discovered_prefilter_records_only_untrusted_no_options():
    previous_cached = scanner._cached_option_expirations_for_ticker
    previous_submit = scanner._submit_background_job
    try:
        scanner._cached_option_expirations_for_ticker = lambda ticker: (True, [])
        scanner._submit_background_job = lambda *args, **kwargs: False

        accepted, skipped = scanner._prefilter_stock_universe(
            ["AAPL", "MSFT"],
            {"AAPL": daily_frame(), "MSFT": daily_frame()},
            trusted_options_symbols={"AAPL"},
        )
        assert accepted == ["AAPL"]
        assert skipped == [{"ticker": "MSFT", "reason": "no options"}]
    finally:
        scanner._cached_option_expirations_for_ticker = previous_cached
        scanner._submit_background_job = previous_submit


def test_coverage_exposes_provider_diagnostics_and_discovery_optionability():
    snapshot = scanner.build_discovered_scan_coverage_snapshot(
        [],
        [
            {
                "ticker": "AAPL",
                "setupGrade": "B",
                "entryStatus": "Waiting",
                "setupStatus": "Pullback Active",
                "trade_eval": {"trade_stage": "BUILDING / WATCHLIST", "no_trade_reasons": []},
                "best_contract": {
                    "available": False,
                    "source": "loading",
                    "loading": True,
                    "reason": "Options data is loading in the background",
                },
            }
        ],
        {
            "configured_universe_count": 1,
            "symbols_successfully_processed": 1,
            "tradeability_skipped": 0,
            "tradeability_skip_reasons": {},
            "no_setup_or_failed_count": 0,
            "symbols_omitted_or_rejected": 0,
            "provider_metrics": {
                "alpaca_bar_requests": 3,
                "alpaca_bar_pages": 7,
                "alpaca_max_pages_exceeded_count": 0,
            },
            "cache_stats": {
                "prices_hit": 4,
                "prices_miss": 2,
                "api_option_expirations_call": 0,
                "api_option_chain_call": 2,
            },
            "option_eligibility_from_discovery": 1,
        },
        {
            "universe_source": "discovered",
            "universe_generated_at": "2026-07-22T12:00:00Z",
            "universe_symbol_count": 1,
            "discovery": {"effective_cap": 750},
        },
    )
    diagnostics = snapshot["provider_diagnostics"]
    assert diagnostics["alpaca_bar_requests"] == 3
    assert diagnostics["alpaca_bar_pages"] == 7
    assert diagnostics["alpaca_max_pages_exceeded_count"] == 0
    assert diagnostics["bar_cache_hits"] == 4
    assert diagnostics["bar_cache_misses"] == 2
    assert diagnostics["option_eligibility_from_discovery"] == 1
    assert diagnostics["yahoo_expiration_requests"] == 0
    assert diagnostics["live_chain_requests"] == 2
    assert diagnostics["live_chain_failures"] == 1


def main() -> int:
    test_discovered_optionability_bypasses_yahoo_expiration_negative()
    test_default_watchlist_still_rejects_confirmed_no_options()
    test_discovered_prefilter_records_only_untrusted_no_options()
    test_coverage_exposes_provider_diagnostics_and_discovery_optionability()
    print("Discovered options provenance v1 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
