#!/usr/bin/env python3
"""Regression tests for hybrid provider strategy-output diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import market_data  # noqa: E402


def snapshot(**overrides):
    base = {
        "ticker": "AAPL",
        "selected_timeframe": "1D",
        "setup_status": "QUALIFIED",
        "trend": "LONG",
        "direction": "LONG",
        "setup_grade": "A",
        "quality_grade": "A",
        "quality_score": 82,
        "confirmation_started": True,
        "entry_status": "Tradeable",
        "entry": 200.00,
        "stop": 195.00,
        "target_1": 210.00,
        "target_2": 215.00,
        "target_3": 220.00,
        "risk": 5.00,
        "trade_stage": "A+ READY",
        "a_plus_ready": True,
        "b_plus_tradeable": True,
        "trigger_confirmed": True,
    }
    base.update(overrides)
    return base


def test_values_equivalent_exact_and_none():
    assert market_data._values_equivalent("A", "A") is True
    assert market_data._values_equivalent(None, None) is True
    assert market_data._values_equivalent(None, 0) is False
    assert market_data._values_equivalent("1", 1) is False


def test_values_equivalent_numeric_tolerance():
    assert market_data._values_equivalent(100.00, 100.00) is True
    assert market_data._values_equivalent(100.00, 100.009) is True
    assert market_data._values_equivalent(100.00, 100.01) is True
    assert market_data._values_equivalent(100.00, 100.011) is False


def test_compare_strategy_outputs_exact_match():
    production = snapshot()
    hybrid = snapshot()
    comparison = market_data._compare_strategy_outputs(production, hybrid)
    assert comparison["differences"] == []
    assert comparison["material_differences"] == []


def test_compare_strategy_outputs_ignores_sub_cent_entry_drift():
    production = snapshot(entry=200.00)
    hybrid = snapshot(entry=200.009)
    comparison = market_data._compare_strategy_outputs(production, hybrid)
    assert comparison["differences"] == []
    assert comparison["material_differences"] == []


def test_compare_strategy_outputs_minor_non_material_difference():
    production = snapshot(quality_score=82)
    hybrid = snapshot(quality_score=84)
    comparison = market_data._compare_strategy_outputs(production, hybrid)
    assert comparison["differences"] == [
        {"field": "quality_score", "production": 82, "hybrid": 84}
    ]
    assert comparison["material_differences"] == []


def test_compare_strategy_outputs_material_grade_difference():
    production = snapshot(setup_grade="A")
    hybrid = snapshot(setup_grade="B")
    comparison = market_data._compare_strategy_outputs(production, hybrid)
    assert any(diff["field"] == "setup_grade" for diff in comparison["differences"])
    assert comparison["material_differences"] == ["setup_grade"]


def test_compare_strategy_outputs_material_direction_difference():
    production = snapshot(direction="LONG", trend="LONG")
    hybrid = snapshot(direction="SHORT", trend="SHORT")
    comparison = market_data._compare_strategy_outputs(production, hybrid)
    assert sorted(comparison["material_differences"]) == ["direction", "trend"]


def test_compare_strategy_outputs_material_entry_difference():
    production = snapshot(entry=200.00)
    hybrid = snapshot(entry=200.02)
    comparison = market_data._compare_strategy_outputs(production, hybrid)
    assert any(diff["field"] == "entry" for diff in comparison["differences"])
    assert comparison["material_differences"] == ["entry"]


def test_compare_strategy_outputs_missing_output_is_material():
    comparison = market_data._compare_strategy_outputs(snapshot(), None)
    assert comparison["differences"] == ["missing scanner output"]
    assert comparison["material_differences"] == ["missing scanner output"]


def test_strategy_output_snapshot_extracts_expected_fields():
    result = {
        "ticker": "AAPL",
        "timeframe": "4H",
        "setup_status": "QUALIFIED",
        "trend": "SHORT",
        "direction": "SHORT",
        "setupGrade": "B",
        "quality": {"grade": "B", "score": 74},
        "confirmationStarted": True,
        "confirmationReason": "Rejection confirmed",
        "entryStatus": "Tradeable",
        "entry": 198.5,
        "sl": 202.0,
        "tp1": 192.0,
        "tp2": 188.0,
        "tp3": 184.0,
        "risk": 3.5,
        "trade_eval": {
            "trade_stage": "B+ TRADEABLE",
            "a_plus_ready": False,
            "b_plus_tradeable": True,
            "trigger_confirmed": False,
            "no_trade_reasons": [],
        },
        "setupStatusReason": "Tradeable pullback",
        "setupGradeReason": "Good structure",
    }
    extracted = market_data._strategy_output_snapshot(result)
    assert extracted == {
        "ticker": "AAPL",
        "selected_timeframe": "4H",
        "setup_status": "QUALIFIED",
        "trend": "SHORT",
        "direction": "SHORT",
        "setup_grade": "B",
        "quality_grade": "B",
        "quality_score": 74,
        "confirmation_started": True,
        "confirmation_reason": "Rejection confirmed",
        "entry_status": "Tradeable",
        "entry": 198.5,
        "stop": 202.0,
        "target_1": 192.0,
        "target_2": 188.0,
        "target_3": 184.0,
        "risk": 3.5,
        "trade_stage": "B+ TRADEABLE",
        "a_plus_ready": False,
        "b_plus_tradeable": True,
        "trigger_confirmed": False,
        "no_trade_reasons": [],
        "setup_status_reason": "Tradeable pullback",
        "grade_reason": "Good structure",
    }


def main() -> int:
    test_values_equivalent_exact_and_none()
    test_values_equivalent_numeric_tolerance()
    test_compare_strategy_outputs_exact_match()
    test_compare_strategy_outputs_ignores_sub_cent_entry_drift()
    test_compare_strategy_outputs_minor_non_material_difference()
    test_compare_strategy_outputs_material_grade_difference()
    test_compare_strategy_outputs_material_direction_difference()
    test_compare_strategy_outputs_material_entry_difference()
    test_compare_strategy_outputs_missing_output_is_material()
    test_strategy_output_snapshot_extracts_expected_fields()
    print("Hybrid provider diagnostics v1 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
