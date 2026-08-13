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

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scanner  # noqa: E402


EXPECTED_VERSION = "v1.0"
EXPECTED_BASELINE = "7441aac88d5cdf2bb479b85f0e73e4cec629ed57"

CORE_FUNCTION_HASHES = {
    # Updated 2026-07-19: corrected RR disqualifier wording to match the existing 1.5R threshold.
    "_build_trade_stage_eval": "6d4e58dde9e3c2b4e6cf43e16593753f17592d3ef3326b02bf46dd345867c8ef",
    # Updated 2026-07-18: added daily-only HTF swing tolerance for provider precision stability (non-strategy).
    "analyze_ticker": "6b526f843e5dfc1a78101d2eb5424b6f3d1d2bf9a638d71d038104f696dd7284",
    "_build_chart_coach": "7642863ce83136bacf46e441b8a5965e7552d067435b1dec2800ee79e39561f0",
    # Updated 2026-07-19: split A/B grading so A requires strength plus no no-trade reasons.
    "_stock_setup_grade": "895ae1de0d2d910dfe6bc2473939aa736c502af337fbde4a1843c27f495b3c9c",
    "_stock_entry_status": "9dd1abeeac4f42f27dbfbbf906b0fb6728f4148146b9e10f649d9044d6b09f6f",
    "_stock_setup_status": "30daf790f61a2dad531c5890aae3a4be2d85f2b5b16c3f9146edac9bd8ba0983",
    "_nearest_target": "c5ecc64b2333725ee75ace358a3930d976bc5effd69a48948df505d27c0544e4",
    # Updated 2026-07-19: corrected RR room label to match the existing 1.5R threshold.
    "_room_to_target": "2129bf616075452c8b0ec6b6b0c8160a8d3114e221b51694a69d444a82c9cab1",
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


FREEZE_BOS_SWINGS = [
    {"type": "high", "index": 1, "price": 96.0},
    {"type": "low", "index": 2, "price": 94.0},
    {"type": "high", "index": 3, "price": 100.0},
    {"type": "low", "index": 5, "price": 98.0},
    {"type": "high", "index": 7, "price": 104.0},
]


def displacement_df(candles: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1000000}
            for open_, high, low, close in candles
        ],
        index=pd.date_range("2026-01-01", periods=len(candles), freq="D", tz="UTC"),
    )


def displacement_snapshot(candles: list[tuple[float, float, float, float]], *, atr: float = 2.0) -> dict:
    df = displacement_df(candles)
    bos_index = scanner._first_bos_close_index(df, FREEZE_BOS_SWINGS, "LONG", 100.0)
    components = scanner._displacement_measurement_components(df, atr, "LONG", bos_index=bos_index)
    classification, score = scanner.detect_displacement(df, atr, "LONG", True, bos_index=bos_index)
    selected_index = components["displacement_index"]
    return {
        "selected_bar_timestamp": scanner._format_utc_timestamp(df.index[selected_index]),
        "avg_body_atr": round(components["avg_body_atr"], 4) if components["avg_body_atr"] is not None else None,
        "last_range_atr": round(components["last_range_atr"], 4) if components["last_range_atr"] is not None else None,
        "directional_count": len(components["directional"]),
        "classification": classification,
        "score": score,
    }


def test_displacement_anchors_to_bos_close_bar() -> None:
    snapshot = displacement_snapshot([
        (95.0, 96.0, 94.0, 95.2),
        (95.0, 96.2, 94.8, 95.5),
        (97.0, 98.4, 96.8, 98.0),
        (98.4, 99.8, 98.2, 99.2),
        (100.2, 101.8, 98.8, 100.8),
        (101.0, 101.4, 100.4, 100.7),
        (100.7, 101.0, 100.0, 100.4),
        (100.4, 100.8, 99.7, 100.1),
    ])
    assert_equal("anchored selected bar", snapshot["selected_bar_timestamp"], "2026-01-05T00:00:00Z")
    assert_equal("anchored avg body", snapshot["avg_body_atr"], 0.4)
    assert_equal("anchored last range", snapshot["last_range_atr"], 1.5)
    assert_equal("anchored directional count", snapshot["directional_count"], 3)
    assert_equal("anchored classification", snapshot["classification"], "STRONG")


def test_displacement_last_range_non_zero_on_bos_bar() -> None:
    snapshot = displacement_snapshot([
        (95.0, 96.0, 94.0, 95.2),
        (95.0, 96.2, 94.8, 95.5),
        (97.0, 98.4, 96.8, 98.0),
        (98.4, 99.8, 98.2, 99.2),
        (100.2, 101.8, 98.8, 100.8),
        (101.0, 101.4, 100.4, 100.7),
        (100.7, 101.0, 100.0, 100.4),
        (100.4, 100.8, 99.7, 100.1),
    ])
    if not snapshot["last_range_atr"] or snapshot["last_range_atr"] <= 0:
        raise AssertionError(f"last_range_atr should be non-zero on BOS candle, got {snapshot['last_range_atr']!r}")


def test_displacement_body_only_strong() -> None:
    snapshot = displacement_snapshot([
        (95.0, 96.0, 94.0, 95.2),
        (95.0, 96.2, 94.8, 95.5),
        (97.0, 99.0, 96.8, 98.7),
        (98.8, 100.4, 98.6, 100.3),
        (100.2, 101.5, 99.5, 101.2),
        (101.0, 101.4, 100.4, 100.7),
        (100.7, 101.0, 100.0, 100.4),
        (100.4, 100.8, 99.7, 100.1),
    ])
    assert_equal("body-only selected bar", snapshot["selected_bar_timestamp"], "2026-01-05T00:00:00Z")
    assert_equal("body-only avg body", snapshot["avg_body_atr"], 0.7)
    assert_equal("body-only last range", snapshot["last_range_atr"], 1.0)
    assert_equal("body-only directional count", snapshot["directional_count"], 3)
    assert_equal("body-only classification", snapshot["classification"], "STRONG")


def test_displacement_range_only_strong() -> None:
    snapshot = displacement_snapshot([
        (95.0, 96.0, 94.0, 95.2),
        (95.0, 96.2, 94.8, 95.5),
        (97.0, 97.8, 96.8, 97.5),
        (98.0, 98.9, 97.8, 98.6),
        (100.2, 101.8, 98.8, 100.8),
        (101.0, 101.4, 100.4, 100.7),
        (100.7, 101.0, 100.0, 100.4),
        (100.4, 100.8, 99.7, 100.1),
    ])
    assert_equal("range-only selected bar", snapshot["selected_bar_timestamp"], "2026-01-05T00:00:00Z")
    assert_equal("range-only avg body", snapshot["avg_body_atr"], 0.2833)
    assert_equal("range-only last range", snapshot["last_range_atr"], 1.5)
    assert_equal("range-only directional count", snapshot["directional_count"], 3)
    assert_equal("range-only classification", snapshot["classification"], "STRONG")


def test_displacement_failing_strong_size_arms_is_weak() -> None:
    snapshot = displacement_snapshot([
        (95.0, 96.0, 94.0, 95.2),
        (95.0, 96.2, 94.8, 95.5),
        (97.0, 98.1, 96.8, 97.9),
        (98.0, 98.9, 97.8, 98.8),
        (100.2, 101.4, 99.8, 101.0),
        (101.0, 101.4, 100.4, 100.7),
        (100.7, 101.0, 100.0, 100.4),
        (100.4, 100.8, 99.7, 100.1),
    ])
    assert_equal("weak selected bar", snapshot["selected_bar_timestamp"], "2026-01-05T00:00:00Z")
    assert_equal("weak avg body", snapshot["avg_body_atr"], 0.4167)
    assert_equal("weak last range", snapshot["last_range_atr"], 0.8)
    assert_equal("weak directional count", snapshot["directional_count"], 3)
    assert_equal("weak classification", snapshot["classification"], "WEAK")


def test_displacement_directional_majority_blocks_strong() -> None:
    snapshot = displacement_snapshot([
        (95.0, 96.0, 94.0, 95.2),
        (95.0, 96.2, 94.8, 95.5),
        (98.0, 98.4, 96.8, 97.0),
        (99.2, 99.8, 98.2, 98.4),
        (100.2, 101.8, 98.8, 100.8),
        (101.0, 101.4, 100.4, 100.7),
        (100.7, 101.0, 100.0, 100.4),
        (100.4, 100.8, 99.7, 100.1),
    ])
    assert_equal("majority selected bar", snapshot["selected_bar_timestamp"], "2026-01-05T00:00:00Z")
    assert_equal("majority avg body", snapshot["avg_body_atr"], 0.3)
    assert_equal("majority last range", snapshot["last_range_atr"], 1.5)
    assert_equal("majority directional count", snapshot["directional_count"], 1)
    assert_equal("majority classification", snapshot["classification"], "WEAK")


DISPLACEMENT_CASES = [
    test_displacement_anchors_to_bos_close_bar,
    test_displacement_last_range_non_zero_on_bos_bar,
    test_displacement_body_only_strong,
    test_displacement_range_only_strong,
    test_displacement_failing_strong_size_arms_is_weak,
    test_displacement_directional_majority_blocks_strong,
]


def executable_fixture(**overrides) -> dict:
    row = {
        "ticker": "EXECFIX",
        "timeframe": "1D",
        "direction": "LONG",
        "entryStatus": "Tradeable",
        "entry": 100.0,
        "price": 100.0,
        "current_quote_price": 100.0,
        "atr": 4.0,
        "distanceFromEntryAtr": 0.0,
    }
    row.update(overrides)
    return row


def test_executable_short_live_quote_above_entry_blocks() -> None:
    row = executable_fixture(
        direction="SHORT",
        entry=1219.60,
        price=1217.17,
        current_quote_price=1243.99,
        atr=25.80,
        distanceFromEntryAtr=0.09,
    )
    assert_equal("short wrong-side live quote", scanner._ranking_entry_executable(row), False)


def test_executable_long_live_quote_below_entry_blocks() -> None:
    row = executable_fixture(
        direction="LONG",
        entry=100.0,
        price=100.2,
        current_quote_price=99.9,
        atr=2.0,
        distanceFromEntryAtr=0.1,
    )
    assert_equal("long wrong-side live quote", scanner._ranking_entry_executable(row), False)


def test_executable_short_both_prices_below_entry_passes() -> None:
    row = executable_fixture(
        direction="SHORT",
        entry=100.0,
        price=99.8,
        current_quote_price=99.7,
        atr=2.0,
        distanceFromEntryAtr=0.1,
    )
    assert_equal("short correct-side live quote", scanner._ranking_entry_executable(row), True)


def test_executable_quote_within_plausibility_limit_passes() -> None:
    row = executable_fixture(
        direction="LONG",
        entry=100.0,
        price=100.0,
        current_quote_price=109.99,
        atr=2.0,
        distanceFromEntryAtr=0.0,
    )
    assert_equal("plausible quote correct-side", scanner._ranking_entry_executable(row), True)
    assert_equal("plausible quote blocker absent", row.get("execution_quote_blocker"), None)


def test_executable_implausible_quote_blocks_with_reason() -> None:
    row = executable_fixture(
        direction="LONG",
        entry=100.0,
        price=100.0,
        current_quote_price=111.0,
        atr=2.0,
        distanceFromEntryAtr=0.0,
    )
    assert_equal("implausible quote blocks", scanner._ranking_entry_executable(row), False)
    assert_equal("implausible quote reason", row.get("execution_quote_blocker"), "quote_implausible_vs_candle_close")
    assert_equal("implausible quote divergence", row.get("execution_quote_divergence"), 0.11)


def test_executable_implausible_correct_side_quote_still_blocks() -> None:
    row = executable_fixture(
        direction="SHORT",
        entry=100.0,
        price=100.0,
        current_quote_price=80.0,
        atr=2.0,
        distanceFromEntryAtr=0.0,
    )
    assert_equal("implausible correct-side quote blocks", scanner._ranking_entry_executable(row), False)
    assert_equal("implausible correct-side reason", row.get("execution_quote_blocker"), "quote_implausible_vs_candle_close")


def test_executable_missing_live_quote_blocks() -> None:
    row = executable_fixture(
        direction="LONG",
        entry=100.0,
        price=100.1,
        current_quote_price=None,
        atr=2.0,
        distanceFromEntryAtr=0.05,
    )
    assert_equal("missing live quote fails closed", scanner._ranking_entry_executable(row), False)


def test_executable_quote_divergence_boundary_passes() -> None:
    row = executable_fixture(
        direction="LONG",
        entry=100.0,
        price=100.0,
        current_quote_price=110.0,
        atr=2.0,
        distanceFromEntryAtr=0.0,
    )
    assert_equal("ten percent quote boundary passes", scanner._ranking_entry_executable(row), True)


EXECUTABLE_GATE_CASES = [
    test_executable_short_live_quote_above_entry_blocks,
    test_executable_long_live_quote_below_entry_blocks,
    test_executable_short_both_prices_below_entry_passes,
    test_executable_quote_within_plausibility_limit_passes,
    test_executable_implausible_quote_blocks_with_reason,
    test_executable_implausible_correct_side_quote_still_blocks,
    test_executable_missing_live_quote_blocks,
    test_executable_quote_divergence_boundary_passes,
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

    for displacement_case in DISPLACEMENT_CASES:
        displacement_case()

    for executable_case in EXECUTABLE_GATE_CASES:
        executable_case()

    print(
        f"Stock Scanner Strategy {EXPECTED_VERSION} regression passed "
        f"({len(CASES)} snapshot cases + {len(DISPLACEMENT_CASES)} displacement cases "
        f"+ {len(EXECUTABLE_GATE_CASES)} executable gate cases)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
