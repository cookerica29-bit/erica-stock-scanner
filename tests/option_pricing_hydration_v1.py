#!/usr/bin/env python3
"""Async option quote hydration tests."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scanner


def _chain(row: dict, option_type: str = "CALL"):
    frame = pd.DataFrame([row])
    if option_type == "CALL":
        return SimpleNamespace(calls=frame, puts=pd.DataFrame())
    return SimpleNamespace(calls=pd.DataFrame(), puts=frame)


def test_live_ask_contract_cost_and_cache_update():
    original = scanner._option_chain_for_ticker
    scanner._option_quote_cache.clear()
    try:
        scanner._option_chain_for_ticker = lambda ticker, expiry: _chain({
            "contractSymbol": "ABC260918C00100000",
            "strike": 100.0,
            "bid": 1.75,
            "ask": 1.85,
            "lastPrice": 1.8,
            "volume": 80,
            "openInterest": 250,
        })
        descriptor = {
            "ticker": "ABC",
            "type": "CALL",
            "expiry": "2026-09-18",
            "strike": 100.0,
            "key": scanner._option_pricing_key("ABC", "2026-09-18", 100.0, "CALL"),
        }
        pricing = scanner._fetch_option_pricing(descriptor)
        assert pricing["status"] == "ready"
        assert pricing["quality"] == "live_ask"
        assert pricing["ask"] == 1.85
        assert pricing["estimated_contract_cost"] == 185.0
        hydrated = scanner._apply_option_pricing_to_row({
            "ticker": "ABC",
            "direction": "LONG",
            "option": {"type": "CALL", "strike": 100.0, "expiry": "2026-09-18"},
            "best_contract": {"available": False, "source": "option_plan"},
        }, pricing)
        assert hydrated["pricing_status"] == "ready"
        assert hydrated["best_contract"]["source"] == "option_quote"
        assert hydrated["best_contract"]["estimated_contract_cost"] == 185.0
    finally:
        scanner._option_chain_for_ticker = original
        scanner._option_quote_cache.clear()


def test_last_price_fallback_and_missing_contract_reason():
    original = scanner._option_chain_for_ticker
    scanner._option_quote_cache.clear()
    try:
        scanner._option_chain_for_ticker = lambda ticker, expiry: _chain({
            "contractSymbol": "XYZ260918P00050000",
            "strike": 50.0,
            "bid": 0.0,
            "ask": 0.0,
            "lastPrice": 0.72,
            "volume": 0,
            "openInterest": 0,
        }, option_type="PUT")
        descriptor = {
            "ticker": "XYZ",
            "type": "PUT",
            "expiry": "2026-09-18",
            "strike": 50.0,
            "key": scanner._option_pricing_key("XYZ", "2026-09-18", 50.0, "PUT"),
        }
        pricing = scanner._fetch_option_pricing(descriptor)
        assert pricing["status"] == "ready"
        assert pricing["quality"] == "last_price_fallback"
        assert pricing["estimated_contract_cost"] == 72.0

        missing = {
            "ticker": "XYZ",
            "type": "PUT",
            "expiry": "2026-09-18",
            "strike": 55.0,
            "key": scanner._option_pricing_key("XYZ", "2026-09-18", 55.0, "PUT"),
        }
        unavailable = scanner._fetch_option_pricing(missing)
        assert unavailable["status"] == "unavailable"
        assert unavailable["reason"] == "contract_not_found"
        assert unavailable["estimated_contract_cost"] is None
    finally:
        scanner._option_chain_for_ticker = original
        scanner._option_quote_cache.clear()


def test_analysis_cache_rows_update_without_changing_stock_plan():
    original = scanner._option_chain_for_ticker
    original_submit = scanner._submit_background_job
    scanner._option_quote_cache.clear()
    scanner._analysis_cache.clear()
    try:
        scanner._submit_background_job = lambda *args, **kwargs: False
        scanner._option_chain_for_ticker = lambda ticker, expiry: _chain({
            "contractSymbol": "MO260918C00068000",
            "strike": 68.0,
            "bid": 1.7,
            "ask": 1.85,
            "lastPrice": 1.76,
            "volume": 120,
            "openInterest": 600,
        })
        key = ("default",)
        row = {
            "ticker": "MO",
            "direction": "LONG",
            "entry": 68.39,
            "sl": 66.90,
            "tp1": 71.37,
            "ranking": {"rank": 1, "status_bucket": "EARLY_ENTRY"},
            "option": {"type": "CALL", "strike": 68.0, "expiry": "2026-09-18"},
            "best_contract": {"available": False, "source": "option_plan"},
        }
        cached = scanner._store_analysis_cache(key, [row], [], {"strategy_version": "v1.0"})
        generation = cached["generated_at"]
        pending = scanner.analysis_cache_snapshot(universe="default")["rows"][0]
        assert pending["pricing_status"] == "pending"
        descriptor, reason = scanner._option_pricing_descriptor(row)
        assert reason is None
        scanner._run_option_pricing_for_cache(key, generation, [descriptor])
        hydrated = scanner.analysis_cache_snapshot(universe="default")["rows"][0]
        assert hydrated["pricing_status"] == "ready"
        assert hydrated["best_contract"]["estimated_contract_cost"] == 185.0
        assert hydrated["entry"] == 68.39
        assert hydrated["sl"] == 66.90
        assert hydrated["tp1"] == 71.37
        diagnostics = scanner.option_pricing_diagnostics(universe="default")
        assert diagnostics["ready"] == 1
        assert diagnostics["live_asks_found"] == 1
    finally:
        scanner._option_chain_for_ticker = original
        scanner._submit_background_job = original_submit
        scanner._option_quote_cache.clear()
        scanner._analysis_cache.clear()


def test_auto_pricing_cap_marks_unqueued_rows_truthfully():
    original_limit = scanner.OPTION_PRICING_AUTO_LIMIT
    original_submit = scanner._submit_background_job
    scanner._option_quote_cache.clear()
    scanner._analysis_cache.clear()
    try:
        scanner.OPTION_PRICING_AUTO_LIMIT = 1
        scanner._submit_background_job = lambda *args, **kwargs: False
        first = {
            "ticker": "AAA",
            "direction": "LONG",
            "ranking": {"rank": 1, "status_bucket": "ENTER_NOW"},
            "option": {"type": "CALL", "strike": 20.0, "expiry": "2026-09-18"},
        }
        second = {
            "ticker": "BBB",
            "direction": "LONG",
            "ranking": {"rank": 2, "status_bucket": "WAITING"},
            "option": {"type": "CALL", "strike": 30.0, "expiry": "2026-09-18"},
        }
        scanner._store_analysis_cache(("default",), [first, second], [], {})
        rows = scanner.analysis_cache_snapshot(universe="default")["rows"]
        assert rows[0]["pricing_status"] == "pending"
        assert rows[1]["pricing_status"] == "unavailable"
        assert rows[1]["option_pricing"]["reason"] == "not_queued"
        diagnostics = scanner.option_pricing_diagnostics(universe="default")
        assert diagnostics["queue_depth"] == 1
        assert diagnostics["unavailable_reasons"]["not_queued"] == 1
    finally:
        scanner.OPTION_PRICING_AUTO_LIMIT = original_limit
        scanner._submit_background_job = original_submit
        scanner._option_quote_cache.clear()
        scanner._analysis_cache.clear()


if __name__ == "__main__":
    test_live_ask_contract_cost_and_cache_update()
    test_last_price_fallback_and_missing_contract_reason()
    test_analysis_cache_rows_update_without_changing_stock_plan()
    test_auto_pricing_cap_marks_unqueued_rows_truthfully()
    print("Option pricing hydration v1 tests passed")
