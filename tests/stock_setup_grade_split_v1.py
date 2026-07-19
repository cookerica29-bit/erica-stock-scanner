#!/usr/bin/env python3
"""Regression tests for stock setup A/B/C grade split."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scanner  # noqa: E402


def assert_equal(label: str, actual, expected) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def setup(**overrides) -> dict:
    base = {
        "direction": "LONG",
        "structureLabel": "Bullish Structure",
        "bos_confirmed": True,
        "in_ob": True,
        "near_ob": False,
        "price": 101.0,
        "ema20": 100.0,
        "trade_eval": {
            "trigger_confirmed": False,
            "a_plus_ready": False,
            "b_plus_tradeable": False,
            "rejection_confirmed": False,
            "no_trade_reasons": [],
        },
    }
    base.update(overrides)
    return base


def with_trade_eval(row: dict, **overrides) -> dict:
    trade_eval = dict(row.get("trade_eval") or {})
    trade_eval.update(overrides)
    row["trade_eval"] = trade_eval
    return row


def grade(row: dict, daily_direction: str = "Bullish", setup_direction: str = "Bullish", location: str = "discount") -> tuple:
    return scanner._stock_setup_grade(row, daily_direction, setup_direction, location, "Pullback Active")


def a_requires_strength_and_clean_no_trade_reasons():
    row = with_trade_eval(setup(), b_plus_tradeable=True)
    actual = grade(row)
    assert_equal("clean B+ setup grades A", actual[0], "A")
    assert_equal("clean B+ setup remains confirmed", actual[2], True)


def rejection_at_zone_can_qualify_as_a_when_clean():
    row = with_trade_eval(setup(), rejection_confirmed=True)
    actual = grade(row)
    assert_equal("clean rejection at zone grades A", actual[0], "A")
    assert_equal("clean rejection reason", actual[3], "strong bullish reaction from demand/support")


def no_trade_reasons_demote_strength_to_b():
    row = with_trade_eval(setup(), b_plus_tradeable=True, no_trade_reasons=["RR < 1.5:1"])
    actual = grade(row)
    assert_equal("B+ with no-trade reason grades B", actual[0], "B")
    assert_equal(
        "B+ with no-trade reason explains review",
        actual[1],
        "B Setup — Promising but needs review: bullish structure shift after pullback",
    )


def rejection_at_zone_with_no_trade_reasons_demotes_to_b():
    row = with_trade_eval(setup(), rejection_confirmed=True, no_trade_reasons=["Choppy/internal structure"])
    actual = grade(row)
    assert_equal("rejection at zone with no-trade reason grades B", actual[0], "B")
    assert_equal(
        "rejection at zone with no-trade reason keeps strongest reason",
        actual[3],
        "strong bullish reaction from demand/support",
    )


def confirmation_started_without_strength_demotes_to_b():
    row = setup()
    actual = grade(row)
    assert_equal("BOS-only confirmation now grades B", actual[0], "B")
    assert_equal("BOS-only confirmation still started", actual[2], True)


def aligned_without_confirmation_remains_b():
    row = setup(bos_confirmed=False, price=99.0)
    actual = grade(row)
    assert_equal("aligned unconfirmed setup grades B", actual[0], "B")
    assert_equal(
        "aligned unconfirmed setup reason unchanged",
        actual[1],
        "B Setup — Wait: trend and location aligned, confirmation not started",
    )


def c_gates_still_run_before_ab_split():
    assert_equal(
        "mixed trend remains C",
        grade(with_trade_eval(setup(), b_plus_tradeable=True), daily_direction="Mixed")[0],
        "C",
    )
    assert_equal(
        "counter-trend remains C",
        grade(with_trade_eval(setup(), b_plus_tradeable=True), daily_direction="Bearish")[0],
        "C",
    )
    assert_equal(
        "location conflict remains C",
        grade(with_trade_eval(setup(), b_plus_tradeable=True), location="premium")[0],
        "C",
    )
    assert_equal(
        "weak location remains C",
        grade(with_trade_eval(setup(in_ob=False, near_ob=False), b_plus_tradeable=True), location="midrange")[0],
        "C",
    )


def rr_label_matches_actual_block_threshold():
    room = scanner._room_to_target(
        price=100.0,
        direction="LONG",
        swings=[{"type": "high", "price": 105.0}],
        entry=100.0,
        stop=96.0,
        fallback_target=105.0,
    )
    assert_equal("RR below 1.5 is blocked", room["blocked"], True)
    assert_equal("RR label matches 1.5 threshold", room["label"], "Blocked: RR < 1.5:1")


def main() -> int:
    a_requires_strength_and_clean_no_trade_reasons()
    rejection_at_zone_can_qualify_as_a_when_clean()
    no_trade_reasons_demote_strength_to_b()
    rejection_at_zone_with_no_trade_reasons_demotes_to_b()
    confirmation_started_without_strength_demotes_to_b()
    aligned_without_confirmation_remains_b()
    c_gates_still_run_before_ab_split()
    rr_label_matches_actual_block_threshold()
    print("Stock setup grade split v1 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
