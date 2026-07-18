#!/usr/bin/env python3
"""Regression tests for per-timeframe market-data provider routing."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import market_data  # noqa: E402
import scanner  # noqa: E402


def with_profile(value, callback):
    previous = os.environ.get("STOCK_DATA_PROVIDER_PROFILE")
    try:
        if value is None:
            os.environ.pop("STOCK_DATA_PROVIDER_PROFILE", None)
        else:
            os.environ["STOCK_DATA_PROVIDER_PROFILE"] = value
        callback()
    finally:
        if previous is None:
            os.environ.pop("STOCK_DATA_PROVIDER_PROFILE", None)
        else:
            os.environ["STOCK_DATA_PROVIDER_PROFILE"] = previous


def test_default_profile_is_yahoo_only():
    def run():
        assert market_data.configured_provider_profile_name() == market_data.PROVIDER_PROFILE_PRODUCTION_YAHOO
        assert market_data.provider_name_for_timeframe("1D") == market_data.YAHOO_PROVIDER_NAME
        assert market_data.provider_name_for_timeframe("1W") == market_data.YAHOO_PROVIDER_NAME
        assert market_data.provider_name_for_timeframe("4H") == market_data.YAHOO_PROVIDER_NAME
        assert scanner._price_provider_for_interval("1d").name == market_data.YAHOO_PROVIDER_NAME

    with_profile(None, run)


def test_hybrid_profile_routes_1d_1w_to_alpaca_and_4h_to_yahoo():
    def run():
        assert market_data.configured_provider_profile_name() == market_data.PROVIDER_PROFILE_PROPOSED_HYBRID
        assert market_data.provider_name_for_timeframe("1D") == market_data.ALPACA_PROVIDER_NAME
        assert market_data.provider_name_for_timeframe("1W") == market_data.ALPACA_PROVIDER_NAME
        assert market_data.provider_name_for_timeframe("4H") == market_data.YAHOO_PROVIDER_NAME
        assert scanner._price_provider_for_interval("1d").name == market_data.ALPACA_PROVIDER_NAME
        assert scanner._price_provider_for_interval("1wk").name == market_data.ALPACA_PROVIDER_NAME
        assert scanner._price_provider_for_interval("4h").name == market_data.YAHOO_PROVIDER_NAME

    with_profile(market_data.PROVIDER_PROFILE_PROPOSED_HYBRID, run)


def test_invalid_profile_falls_back_to_yahoo_only():
    def run():
        assert market_data.configured_provider_profile_name() == market_data.PROVIDER_PROFILE_PRODUCTION_YAHOO
        assert market_data.provider_name_for_timeframe("1D") == market_data.YAHOO_PROVIDER_NAME
        assert scanner._price_provider_for_interval("1d").name == market_data.YAHOO_PROVIDER_NAME

    with_profile("not-a-real-profile", run)


def test_price_cache_keys_are_provider_scoped():
    yahoo_key = scanner._price_cache_key("AAPL", "1y", "1d", market_data.YAHOO_PROVIDER_NAME)
    alpaca_key = scanner._price_cache_key("AAPL", "1y", "1d", market_data.ALPACA_PROVIDER_NAME)
    assert yahoo_key != alpaca_key
    assert yahoo_key == ("AAPL", "1y", "1d", market_data.YAHOO_PROVIDER_NAME)
    assert alpaca_key == ("AAPL", "1y", "1d", market_data.ALPACA_PROVIDER_NAME)


def main() -> int:
    test_default_profile_is_yahoo_only()
    test_hybrid_profile_routes_1d_1w_to_alpaca_and_4h_to_yahoo()
    test_invalid_profile_falls_back_to_yahoo_only()
    test_price_cache_keys_are_provider_scoped()
    print("Timeframe provider routing v1 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
