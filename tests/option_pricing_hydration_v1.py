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
        assert unavailable["nearest_available_strikes"] == [50.0]
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
        assert rows[1]["pricing_status"] == "not_requested"
        assert rows[1]["option_pricing"]["reason"] == "outside_auto_hydration_cap"
        scan_payload = scanner.scan_cached(universe="default")
        assert scan_payload["rows"][0]["pricing_status"] == "pending"
        assert scan_payload["rows"][1]["pricing_status"] == "not_requested"
        assert scan_payload["rows"][1]["option_pricing"]["reason"] == "outside_auto_hydration_cap"
        diagnostics = scanner.option_pricing_diagnostics(universe="default")
        assert diagnostics["queue_depth"] == 1
        assert diagnostics["not_requested"] == 1
        assert diagnostics["unavailable_reasons"]["outside_auto_hydration_cap"] == 1
    finally:
        scanner.OPTION_PRICING_AUTO_LIMIT = original_limit
        scanner._submit_background_job = original_submit
        scanner._option_quote_cache.clear()
        scanner._analysis_cache.clear()


def test_lazy_queue_promotes_not_requested_contract_to_pending():
    original_limit = scanner.OPTION_PRICING_AUTO_LIMIT
    original_submit = scanner._submit_background_job
    scanner._option_quote_cache.clear()
    scanner._analysis_cache.clear()
    submitted = []
    try:
        scanner.OPTION_PRICING_AUTO_LIMIT = 0
        scanner._submit_background_job = lambda *args, **kwargs: submitted.append(args) or False
        row = {
            "ticker": "LZY",
            "direction": "SHORT",
            "ranking": {"rank": 10, "status_bucket": "WAITING"},
            "option": {"type": "PUT", "strike": 40.0, "expiry": "2026-09-18"},
        }
        scanner._store_analysis_cache(("default",), [row], [], {})
        assert scanner.analysis_cache_snapshot(universe="default")["rows"][0]["pricing_status"] == "not_requested"
        result = scanner.queue_option_pricing_for_contracts([{
            "ticker": "LZY",
            "type": "PUT",
            "strike": 40.0,
            "expiry": "2026-09-18",
        }], universe="default")
        assert result["queued"] == 0
        assert result["reason"] == "duplicate_or_executor_unavailable"
        assert submitted
        queued = scanner.analysis_cache_snapshot(universe="default")["rows"][0]
        assert queued["pricing_status"] == "pending"
        assert queued["option_pricing"]["reason"] == "queued"
    finally:
        scanner.OPTION_PRICING_AUTO_LIMIT = original_limit
        scanner._submit_background_job = original_submit
        scanner._option_quote_cache.clear()
        scanner._analysis_cache.clear()


def test_missing_option_type_is_not_used_for_missing_plan_inputs():
    descriptor, reason = scanner._option_pricing_descriptor({
        "ticker": "MISS",
        "direction": "LONG",
        "entry": None,
        "tp1": None,
        "option_plan": {"available": False, "reason": "missing TP1", "source": "kairos_trade_plan"},
    })
    assert descriptor is None
    assert reason == "option_plan_unavailable:missing_tp1"

    descriptor, reason = scanner._option_pricing_descriptor({
        "ticker": "PROP",
        "direction": "SHORT",
        "option": {"strike": 45.0, "expiry": "2026-09-18"},
    })
    assert reason is None
    assert descriptor["type"] == "PUT"


def test_batch_pricing_fetches_one_chain_for_multiple_contracts():
    original = scanner._option_chain_for_ticker
    original_sleep = scanner.time.sleep
    original_random = scanner.random.uniform
    scanner._option_quote_cache.clear()
    calls = []
    try:
        scanner.time.sleep = lambda *_args, **_kwargs: None
        scanner.random.uniform = lambda *_args, **_kwargs: 0

        def fake_chain(ticker, expiry):
            calls.append((ticker, expiry))
            return SimpleNamespace(
                calls=pd.DataFrame([{
                    "contractSymbol": "BAT260918C00100000",
                    "strike": 100.0,
                    "bid": 1.0,
                    "ask": 1.1,
                    "lastPrice": 1.05,
                    "volume": 100,
                    "openInterest": 500,
                }, {
                    "contractSymbol": "BAT260918C00105000",
                    "strike": 105.0,
                    "bid": 0.7,
                    "ask": 0.8,
                    "lastPrice": 0.75,
                    "volume": 80,
                    "openInterest": 400,
                }]),
                puts=pd.DataFrame(),
            )

        scanner._option_chain_for_ticker = fake_chain
        descriptors = [{
            "ticker": "BAT",
            "type": "CALL",
            "expiry": "2026-09-18",
            "strike": 100.0,
            "key": scanner._option_pricing_key("BAT", "2026-09-18", 100.0, "CALL"),
        }, {
            "ticker": "BAT",
            "type": "CALL",
            "expiry": "2026-09-18",
            "strike": 105.0,
            "key": scanner._option_pricing_key("BAT", "2026-09-18", 105.0, "CALL"),
        }]
        results, diagnostics = scanner._option_pricing_batch_for_descriptors(descriptors)
        assert calls == [("BAT", "2026-09-18")]
        assert diagnostics["chain_groups"] == 1
        assert diagnostics["duplicate_chain_requests_eliminated"] == 1
        assert diagnostics["chain_requests_attempted"] == 1
        assert results[descriptors[0]["key"]]["estimated_contract_cost"] == 110.0
        assert results[descriptors[1]["key"]]["estimated_contract_cost"] == 80.0
    finally:
        scanner._option_chain_for_ticker = original
        scanner.time.sleep = original_sleep
        scanner.random.uniform = original_random
        scanner._option_quote_cache.clear()


def test_provider_timeout_is_retried_by_chain_and_not_cached():
    original = scanner._option_chain_for_ticker
    original_sleep = scanner.time.sleep
    original_random = scanner.random.uniform
    scanner._option_quote_cache.clear()
    calls = []
    try:
        scanner.time.sleep = lambda *_args, **_kwargs: None
        scanner.random.uniform = lambda *_args, **_kwargs: 0

        def fake_chain(ticker, expiry):
            calls.append((ticker, expiry))
            return None

        scanner._option_chain_for_ticker = fake_chain
        descriptor = {
            "ticker": "TOUT",
            "type": "PUT",
            "expiry": "2026-09-18",
            "strike": 30.0,
            "key": scanner._option_pricing_key("TOUT", "2026-09-18", 30.0, "PUT"),
        }
        results, diagnostics = scanner._option_pricing_batch_for_descriptors([descriptor])
        assert len(calls) == 2
        assert diagnostics["provider_retries"] == 1
        assert diagnostics["chain_requests_attempted"] == 2
        assert results[descriptor["key"]]["status"] == "unavailable"
        assert results[descriptor["key"]]["reason"] == "provider_timeout"
        assert descriptor["key"] not in scanner._option_quote_cache
    finally:
        scanner._option_chain_for_ticker = original
        scanner.time.sleep = original_sleep
        scanner.random.uniform = original_random
        scanner._option_quote_cache.clear()


def test_worker_records_latency_distribution_diagnostics():
    original = scanner._option_chain_for_ticker
    original_submit = scanner._submit_background_job
    original_sleep = scanner.time.sleep
    original_random = scanner.random.uniform
    scanner._option_quote_cache.clear()
    scanner._analysis_cache.clear()
    try:
        scanner._submit_background_job = lambda *args, **kwargs: False
        scanner.time.sleep = lambda *_args, **_kwargs: None
        scanner.random.uniform = lambda *_args, **_kwargs: 0
        scanner._option_chain_for_ticker = lambda ticker, expiry: _chain({
            "contractSymbol": "LAT260918C00020000",
            "strike": 20.0,
            "bid": 0.9,
            "ask": 1.0,
            "lastPrice": 0.95,
            "volume": 100,
            "openInterest": 500,
        })
        key = ("default",)
        row = {
            "ticker": "LAT",
            "direction": "LONG",
            "ranking": {"rank": 1, "status_bucket": "ENTER_NOW"},
            "option": {"type": "CALL", "strike": 20.0, "expiry": "2026-09-18"},
        }
        cached = scanner._store_analysis_cache(key, [row], [], {})
        descriptor, reason = scanner._option_pricing_descriptor(row)
        assert reason is None
        scanner._run_option_pricing_for_cache(key, cached["generated_at"], [descriptor])
        diagnostics = scanner.option_pricing_diagnostics(universe="default")
        assert diagnostics["chain_groups"] == 1
        assert diagnostics["chain_requests_attempted"] == 1
        assert diagnostics["latency_p50_ms"] is not None
        assert diagnostics["latency_p95_ms"] is not None
        assert diagnostics["slowest_requests"][0]["symbol"] == "LAT"
    finally:
        scanner._option_chain_for_ticker = original
        scanner._submit_background_job = original_submit
        scanner.time.sleep = original_sleep
        scanner.random.uniform = original_random
        scanner._option_quote_cache.clear()
        scanner._analysis_cache.clear()


def test_invalid_expiration_is_reported_without_retrying():
    original_yf = scanner.yf
    original_sleep = scanner.time.sleep
    original_random = scanner.random.uniform
    scanner._option_chain_cache.clear()
    scanner._option_quote_cache.clear()
    try:
        scanner.time.sleep = lambda *_args, **_kwargs: None
        scanner.random.uniform = lambda *_args, **_kwargs: 0

        class FakeTicker:
            def __init__(self, ticker):
                self.ticker = ticker

            def option_chain(self, expiry):
                raise ValueError(
                    "Expiration `2026-09-11` cannot be found. "
                    "Available expirations are: [2026-08-21, 2026-09-18]"
                )

        scanner.yf = SimpleNamespace(Ticker=FakeTicker)
        descriptor = {
            "ticker": "BADX",
            "type": "CALL",
            "expiry": "2026-09-11",
            "strike": 100.0,
            "key": scanner._option_pricing_key("BADX", "2026-09-11", 100.0, "CALL"),
        }
        results, diagnostics = scanner._option_pricing_batch_for_descriptors([descriptor])
        pricing = results[descriptor["key"]]
        assert diagnostics["chain_requests_attempted"] == 1
        assert diagnostics["provider_retries"] == 0
        assert pricing["status"] == "unavailable"
        assert pricing["reason"] == "invalid_expiration"
        assert pricing["available_expirations"] == ["2026-08-21", "2026-09-18"]
        assert diagnostics["slowest_requests"][0]["result"] == "invalid_expiration"
    finally:
        scanner.yf = original_yf
        scanner.time.sleep = original_sleep
        scanner.random.uniform = original_random
        scanner._option_chain_cache.clear()
        scanner._option_quote_cache.clear()


if __name__ == "__main__":
    test_live_ask_contract_cost_and_cache_update()
    test_last_price_fallback_and_missing_contract_reason()
    test_analysis_cache_rows_update_without_changing_stock_plan()
    test_auto_pricing_cap_marks_unqueued_rows_truthfully()
    test_lazy_queue_promotes_not_requested_contract_to_pending()
    test_missing_option_type_is_not_used_for_missing_plan_inputs()
    test_batch_pricing_fetches_one_chain_for_multiple_contracts()
    test_provider_timeout_is_retried_by_chain_and_not_cached()
    test_worker_records_latency_distribution_diagnostics()
    test_invalid_expiration_is_reported_without_retrying()
    print("Option pricing hydration v1 tests passed")
