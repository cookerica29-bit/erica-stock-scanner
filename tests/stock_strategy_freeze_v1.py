#!/usr/bin/env python3
"""Regression guard for Stock Scanner Strategy v1.0.

This file intentionally avoids network calls. It verifies frozen strategy helper
outputs with synthetic fixtures and source hashes for the core strategy surface.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scanner  # noqa: E402


EXPECTED_VERSION = "v1.0"
EXPECTED_BASELINE = "7441aac88d5cdf2bb479b85f0e73e4cec629ed57"

CORE_FUNCTION_HASHES = {
    "_build_trade_stage_eval": "8754add5cb2820fafbde5dd213a3c1e8776ccb35cd28857a7a3227430530cd96",
    "analyze_ticker": "a27ef40c18524a296a9298e36334c079638532190c0804b01e3f7ce2b5f9241b",
    "_build_chart_coach": "7642863ce83136bacf46e441b8a5965e7552d067435b1dec2800ee79e39561f0",
    "_stock_setup_grade": "9cab2e9632eb1a179a6cb27910dfa3fbec1a4ae8249b23261fe95ba8bb0761de",
    "_stock_entry_status": "9dd1abeeac4f42f27dbfbbf906b0fb6728f4148146b9e10f649d9044d6b09f6f",
    "_stock_setup_status": "30daf790f61a2dad531c5890aae3a4be2d85f2b5b16c3f9146edac9bd8ba0983",
    "_nearest_target": "c5ecc64b2333725ee75ace358a3930d976bc5effd69a48948df505d27c0544e4",
    "_room_to_target": "a6377c2e0cd1d8ea7192090d5902e81d2b2f31a01723fd479102a29ede31c3c4",
    "_best_timeframe_result": "e2d86006018118abf256b435216a43e8692e778d55ca14192811e604984ad308",
}


def source_hash(fn) -> str:
    return hashlib.sha256(inspect.getsource(fn).encode("utf-8")).hexdigest()


def assert_equal(label: str, actual, expected) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def base_result(**overrides) -> dict:
    result = {
        "ticker": "FIXTURE",
        "timeframe": "1D",
        "direction": "LONG",
        "trend": "LONG",
        "price": 100.0,
        "atr": 2.0,
        "ema20": 105.0,
        "bos_confirmed": True,
        "in_ob": True,
        "near_ob": False,
        "ob_low": 98.0,
        "ob_high": 102.0,
        "entry": 100.0,
        "sl": 96.0,
        "tp1": 108.0,
        "setup_status": "QUALIFIED",
        "quality": {"score": 78, "grade": "B", "cleanliness": "Clean impulse"},
        "trade_eval": {
            "trade_stage": "B+ TRADEABLE",
            "trigger_confirmed": False,
            "a_plus_ready": False,
            "b_plus_tradeable": True,
            "displacement": "WEAK",
            "rejection_confirmed": False,
            "sweep_taken": False,
            "location_percentile": 20.0,
            "no_trade_reasons": [],
        },
    }
    for key, value in overrides.items():
        if key == "trade_eval":
            result["trade_eval"].update(value)
        elif key == "quality":
            result["quality"].update(value)
        else:
            result[key] = value
    return result


def enrich_snapshot(result: dict, daily_trend: str, h4_trend: str) -> dict:
    enriched = scanner._enrich_stock_scout_fields(
        copy.deepcopy(result),
        {"trend": daily_trend},
        {"trend": h4_trend},
        None,
        None,
    )
    return {
        "direction": enriched.get("direction"),
        "setup_grade": enriched.get("setupGrade"),
        "trade_stage": (enriched.get("trade_eval") or {}).get("trade_stage"),
        "entry_status": enriched.get("entryStatus"),
        "entry": enriched.get("entry"),
        "stop": enriched.get("sl"),
        "target": enriched.get("tp1"),
        "qualification": enriched.get("setup_status"),
    }


CASES = [
    {
        "name": "valid bullish setup",
        "result": base_result(trade_eval={
            "trade_stage": "A+ READY",
            "trigger_confirmed": True,
            "a_plus_ready": True,
            "displacement": "STRONG",
            "rejection_confirmed": True,
            "sweep_taken": True,
            "location_percentile": 20.0,
        }),
        "daily": "LONG",
        "h4": "LONG",
        "expected": {
            "direction": "LONG",
            "setup_grade": "A",
            "trade_stage": "A+ READY",
            "entry_status": "Tradeable",
            "entry": 100.0,
            "stop": 96.0,
            "target": 108.0,
            "qualification": "QUALIFIED",
        },
    },
    {
        "name": "valid bearish setup",
        "result": base_result(
            direction="SHORT",
            trend="SHORT",
            price=100.0,
            entry=100.0,
            sl=104.0,
            tp1=92.0,
            trade_eval={
                "trade_stage": "A+ READY",
                "trigger_confirmed": True,
                "a_plus_ready": True,
                "displacement": "STRONG",
                "rejection_confirmed": True,
                "sweep_taken": True,
                "location_percentile": 80.0,
            },
        ),
        "daily": "SHORT",
        "h4": "SHORT",
        "expected": {
            "direction": "SHORT",
            "setup_grade": "A",
            "trade_stage": "A+ READY",
            "entry_status": "Tradeable",
            "entry": 100.0,
            "stop": 104.0,
            "target": 92.0,
            "qualification": "QUALIFIED",
        },
    },
    {
        "name": "near-entry setup",
        "result": base_result(price=100.9, entry=100.0, sl=96.0, tp1=108.0),
        "daily": "LONG",
        "h4": "LONG",
        "expected": {
            "direction": "LONG",
            "setup_grade": "A",
            "trade_stage": "B+ TRADEABLE",
            "entry_status": "Near Entry",
            "entry": 100.0,
            "stop": 96.0,
            "target": 108.0,
            "qualification": "QUALIFIED",
        },
    },
    {
        "name": "waiting setup",
        "result": base_result(
            price=101.5,
            entry=100.0,
            bos_confirmed=False,
            trade_eval={
                "trade_stage": "BUILDING / WATCHLIST",
                "b_plus_tradeable": False,
                "location_percentile": 20.0,
            },
            setup_status="DEVELOPING",
        ),
        "daily": "LONG",
        "h4": "LONG",
        "expected": {
            "direction": "LONG",
            "setup_grade": "B",
            "trade_stage": "BUILDING / WATCHLIST",
            "entry_status": "Waiting",
            "entry": 100.0,
            "stop": 96.0,
            "target": 108.0,
            "qualification": "DEVELOPING",
        },
    },
    {
        "name": "range/no-trade setup",
        "result": base_result(
            direction=None,
            trend="NEUTRAL",
            entry=None,
            sl=None,
            tp1=None,
            setup_status="SKIPPED",
            trade_eval={
                "trade_stage": "RANGE / NO TRADE",
                "b_plus_tradeable": False,
                "location_percentile": 50.0,
                "no_trade_reasons": ["No clear HTF bias"],
            },
        ),
        "daily": "NEUTRAL",
        "h4": "NEUTRAL",
        "expected": {
            "direction": None,
            "setup_grade": "C",
            "trade_stage": "RANGE / NO TRADE",
            "entry_status": "Waiting",
            "entry": None,
            "stop": None,
            "target": None,
            "qualification": "SKIPPED",
        },
    },
    {
        "name": "poor-location rejection",
        "result": base_result(trade_eval={
            "trade_stage": "RANGE / NO TRADE",
            "b_plus_tradeable": False,
            "location_percentile": 80.0,
            "no_trade_reasons": ["Poor premium/discount location"],
        }),
        "daily": "LONG",
        "h4": "LONG",
        "expected": {
            "direction": "LONG",
            "setup_grade": "C",
            "trade_stage": "RANGE / NO TRADE",
            "entry_status": "Tradeable",
            "entry": 100.0,
            "stop": 96.0,
            "target": 108.0,
            "qualification": "QUALIFIED",
        },
    },
    {
        "name": "choppy-structure rejection",
        "result": base_result(
            setup_status="DEVELOPING",
            bos_confirmed=False,
            trade_eval={
                "trade_stage": "RANGE / NO TRADE",
                "b_plus_tradeable": False,
                "location_percentile": 20.0,
                "no_trade_reasons": ["Choppy/internal structure"],
            },
        ),
        "daily": "LONG",
        "h4": "LONG",
        "expected": {
            "direction": "LONG",
            "setup_grade": "B",
            "trade_stage": "RANGE / NO TRADE",
            "entry_status": "Tradeable",
            "entry": 100.0,
            "stop": 96.0,
            "target": 108.0,
            "qualification": "DEVELOPING",
        },
    },
    {
        "name": "R:R below 1:2 rejection",
        "result": base_result(
            setup_status="DEVELOPING",
            bos_confirmed=False,
            trade_eval={
                "trade_stage": "RANGE / NO TRADE",
                "b_plus_tradeable": False,
                "location_percentile": 20.0,
                "no_trade_reasons": ["RR < 1:2"],
            },
        ),
        "daily": "LONG",
        "h4": "LONG",
        "expected": {
            "direction": "LONG",
            "setup_grade": "B",
            "trade_stage": "RANGE / NO TRADE",
            "entry_status": "Tradeable",
            "entry": 100.0,
            "stop": 96.0,
            "target": 108.0,
            "qualification": "DEVELOPING",
        },
    },
]


def main() -> int:
    assert_equal("strategy version", scanner.STOCK_SCANNER_STRATEGY_VERSION, EXPECTED_VERSION)
    assert_equal("strategy baseline", scanner.STOCK_SCANNER_STRATEGY_BASELINE_COMMIT, EXPECTED_BASELINE)

    original_earnings = scanner._earnings_for_ticker
    original_best_contract = scanner._best_contract
    scanner._earnings_for_ticker = lambda *args, **kwargs: {"status": "unavailable"}
    scanner._best_contract = lambda *args, **kwargs: {"available": False, "source": "loading", "loading": True}
    try:
        for case in CASES:
            actual = enrich_snapshot(case["result"], case["daily"], case["h4"])
            assert_equal(case["name"], actual, case["expected"])
    finally:
        scanner._earnings_for_ticker = original_earnings
        scanner._best_contract = original_best_contract

    for name, expected_hash in CORE_FUNCTION_HASHES.items():
        actual_hash = source_hash(getattr(scanner, name))
        if expected_hash == "__FILL__":
            raise AssertionError(f"{name}: fill expected hash with {actual_hash}")
        assert_equal(f"{name} source hash", actual_hash, expected_hash)

    print(f"Stock Scanner Strategy {EXPECTED_VERSION} regression passed ({len(CASES)} cases).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
