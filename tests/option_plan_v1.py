#!/usr/bin/env python3
"""Regression tests for deterministic Kairos Option Plan generation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scanner  # noqa: E402


def setup(**overrides):
    base = {
        "ticker": "AAPL",
        "direction": "LONG",
        "entry": 188.50,
        "sl": 182.00,
        "tp1": 198.75,
        "setupGrade": "A",
        "entryStatus": "Tradeable",
        "trade_eval": {
            "trade_stage": "A+ READY",
            "trigger_confirmed": False,
            "a_plus_ready": False,
            "b_plus_tradeable": True,
        },
    }
    base.update(overrides)
    return base


def test_call_plan_generation_and_formatting():
    plan = scanner.build_option_plan(setup())
    assert plan["available"] is True
    assert plan["type"] == "CALL"
    assert plan["preferred_strike"] == 195.0
    assert plan["expected_move"]["dollars"] == 10.25
    assert plan["expected_move"]["percent"] == 5.4
    assert plan["expected_move"]["label"] == "+$10.25 (+5.4%)"
    assert plan["expected_hold"]["label"] == "7–12 Trading Days"
    assert plan["suggested_expiration"]["label"] == "21–35 DTE"
    assert plan["confidence"]["label"] == "★★★★☆"


def test_put_plan_generation_and_formatting():
    plan = scanner.build_option_plan(setup(direction="SHORT", entry=172.0, tp1=160.0, setupGrade="B"))
    assert plan["available"] is True
    assert plan["type"] == "PUT"
    assert plan["preferred_strike"] == 165.0
    assert plan["expected_move"]["dollars"] == -12.0
    assert plan["expected_move"]["percent"] == -7.0
    assert plan["expected_move"]["label"] == "-$12.00 (-7.0%)"
    assert plan["confidence"]["label"] == "★★★☆☆"


def test_strike_rounding_bands():
    assert scanner.build_option_plan(setup(entry=20.1, tp1=23.2))["preferred_strike"] == 22.0
    assert scanner.build_option_plan(setup(entry=57.4, tp1=62.2))["preferred_strike"] == 60.0
    assert scanner.build_option_plan(setup(entry=188.5, tp1=198.5))["preferred_strike"] == 195.0
    assert scanner.build_option_plan(setup(entry=314.5, tp1=330.0))["preferred_strike"] == 325.0
    assert scanner.build_option_plan(setup(direction="SHORT", entry=172, tp1=160))["preferred_strike"] == 165.0


def test_call_strike_rounding_clamps_below_tp1():
    plan = scanner.build_option_plan(setup(direction="LONG", entry=17.86, tp1=18.36))
    assert plan["available"] is True
    assert plan["type"] == "CALL"
    assert plan["preferred_strike"] == 18.0
    assert plan["preferred_strike"] < plan["tp1"]


def test_put_strike_rounding_clamps_above_tp1():
    plan = scanner.build_option_plan(setup(direction="SHORT", entry=146.53, tp1=142.53))
    assert plan["available"] is True
    assert plan["type"] == "PUT"
    assert plan["preferred_strike"] == 145.0
    assert plan["preferred_strike"] > plan["tp1"]


def test_strike_rounding_already_itm_at_tp1_is_unchanged():
    plan = scanner.build_option_plan(setup(direction="LONG", entry=18.04, tp1=19.36))
    assert plan["available"] is True
    assert plan["preferred_strike"] == 19.0
    assert plan["preferred_strike"] < plan["tp1"]


def test_option_plan_suppressed_when_no_strike_exists_between_entry_and_tp1():
    plan = scanner.build_option_plan(setup(direction="SHORT", entry=17.78, tp1=17.40))
    assert plan["available"] is False
    assert plan["reason"] == "no strike between entry and TP1"


def test_expected_hold_and_expiration_mapping():
    fast = scanner.build_option_plan(setup(trade_eval={"trigger_confirmed": True}))
    assert fast["expected_hold"]["label"] == "3–7 Trading Days"
    assert fast["suggested_expiration"]["label"] == "21–30 DTE"

    extended = scanner.build_option_plan(setup(entryStatus="Waiting", confirmationStarted=False, trade_eval={}))
    assert extended["expected_hold"]["label"] == "10–18 Trading Days"
    assert extended["suggested_expiration"]["label"] == "30–45 DTE"

    explicit = scanner.build_option_plan(setup(expected_trading_days_low=19, expected_trading_days_high=22))
    assert explicit["expected_hold"]["label"] == "19–22 Trading Days"
    assert explicit["expected_hold"]["fallback_used"] is False
    assert explicit["suggested_expiration"]["label"] == "45–60 DTE"


def test_confidence_star_mapping():
    assert scanner.build_option_plan(setup(setupGrade="A", trade_eval={"trigger_confirmed": True}))["confidence"]["label"] == "★★★★★"
    assert scanner.build_option_plan(setup(setupGrade="A"))["confidence"]["label"] == "★★★★☆"
    assert scanner.build_option_plan(setup(setupGrade="B"))["confidence"]["label"] == "★★★☆☆"
    assert scanner.build_option_plan(setup(setupGrade="C"))["confidence"]["label"] == "★★☆☆☆"


def test_planned_entry_fallback_and_missing_tp1_failure():
    fallback = scanner.build_option_plan(setup(entry=None, entry_price=None, price=100.0, tp1=110.0))
    assert fallback["available"] is True
    assert fallback["entry_source"] == "fallback_price"
    assert fallback["planned_entry"] == 100.0

    missing_tp1 = scanner.build_option_plan(setup(tp1=None))
    assert missing_tp1["available"] is False
    assert missing_tp1["reason"] == "missing TP1"


def test_invalid_projected_move_fails_safely():
    call = scanner.build_option_plan(setup(direction="LONG", entry=100, tp1=95))
    put = scanner.build_option_plan(setup(direction="SHORT", entry=100, tp1=105))
    assert call["reason"] == "invalid projected move"
    assert put["reason"] == "invalid projected move"


def test_option_plan_does_not_call_yahoo_contract_selection():
    original_best_contract = scanner._best_contract
    called = {"value": False}

    def fail_if_called(*args, **kwargs):
        called["value"] = True
        raise AssertionError("_best_contract should not be required for Option Plan generation")

    scanner._best_contract = fail_if_called
    try:
        plan = scanner.build_option_plan(setup())
    finally:
        scanner._best_contract = original_best_contract
    assert plan["available"] is True
    assert called["value"] is False


def test_option_pricing_descriptor_prefers_option_plan_contract():
    descriptor, reason = scanner._option_pricing_descriptor({
        "ticker": "NXT",
        "direction": "SHORT",
        "option_plan": {
            "available": True,
            "type": "PUT",
            "preferred_strike": 100.0,
            "expiration": "2026-09-18",
            "expiration_source": "option_plan",
        },
        "option": {
            "type": "PUT",
            "strike": 105.0,
            "expiry": "2026-09-25",
            "expiration_source": "fallback",
        },
    })
    assert reason is None
    assert descriptor["type"] == "PUT"
    assert descriptor["strike"] == 100.0
    assert descriptor["expiry"] == "2026-09-18"
    assert descriptor["expiration_source"] == "option_plan"
    assert descriptor["key"] == ("NXT", "2026-09-18", 100.0, "PUT")


def test_option_pricing_descriptor_prefers_plan_type_and_expiration():
    descriptor, reason = scanner._option_pricing_descriptor({
        "ticker": "KTOS",
        "direction": "SHORT",
        "option_plan": {
            "available": True,
            "type": "PUT",
            "preferred_strike": 59.0,
            "expiration": "2026-09-18",
            "expiration_source": "option_plan",
        },
        "option": {
            "type": "CALL",
            "strike": 63.0,
            "expiry": "2026-09-25",
            "expiration_source": "fallback",
        },
    })
    assert reason is None
    assert descriptor["type"] == "PUT"
    assert descriptor["strike"] == 59.0
    assert descriptor["expiry"] == "2026-09-18"
    assert descriptor["expiration_source"] == "option_plan"
    assert descriptor["key"] == ("KTOS", "2026-09-18", 59.0, "PUT")


def test_option_pricing_descriptor_uses_option_plan_without_row_option():
    descriptor, reason = scanner._option_pricing_descriptor({
        "ticker": "ILMN",
        "direction": "LONG",
        "option_plan": {
            "available": True,
            "type": "CALL",
            "preferred_strike": 200.0,
            "suggested_expiration": {
                "expiration": "2026-09-18",
                "min_dte": 21,
                "max_dte": 30,
            },
        },
    })
    assert reason is None
    assert descriptor["type"] == "CALL"
    assert descriptor["strike"] == 200.0
    assert descriptor["expiry"] == "2026-09-18"
    assert descriptor["expiration_min_dte"] == 21
    assert descriptor["expiration_max_dte"] == 30
    assert descriptor["key"] == ("ILMN", "2026-09-18", 200.0, "CALL")


def test_option_pricing_descriptor_falls_back_to_row_option_without_plan():
    descriptor, reason = scanner._option_pricing_descriptor({
        "ticker": "CHWY",
        "direction": "LONG",
        "option": {
            "type": "CALL",
            "strike": 23.0,
            "expiry": "2026-09-25",
            "expiration_source": "fallback",
        },
    })
    assert reason is None
    assert descriptor["type"] == "CALL"
    assert descriptor["strike"] == 23.0
    assert descriptor["expiry"] == "2026-09-25"
    assert descriptor["expiration_source"] == "fallback"
    assert descriptor["key"] == ("CHWY", "2026-09-25", 23.0, "CALL")


def main() -> int:
    test_call_plan_generation_and_formatting()
    test_put_plan_generation_and_formatting()
    test_strike_rounding_bands()
    test_call_strike_rounding_clamps_below_tp1()
    test_put_strike_rounding_clamps_above_tp1()
    test_strike_rounding_already_itm_at_tp1_is_unchanged()
    test_option_plan_suppressed_when_no_strike_exists_between_entry_and_tp1()
    test_expected_hold_and_expiration_mapping()
    test_confidence_star_mapping()
    test_planned_entry_fallback_and_missing_tp1_failure()
    test_invalid_projected_move_fails_safely()
    test_option_plan_does_not_call_yahoo_contract_selection()
    test_option_pricing_descriptor_prefers_option_plan_contract()
    test_option_pricing_descriptor_prefers_plan_type_and_expiration()
    test_option_pricing_descriptor_uses_option_plan_without_row_option()
    test_option_pricing_descriptor_falls_back_to_row_option_without_plan()
    print("Option Plan v1 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
