#!/usr/bin/env python3
"""Regression tests for stock confirmation reason priority."""

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


def long_setup(**overrides):
    setup = {
        "direction": "LONG",
        "structureLabel": "Bullish Structure",
        "bos_confirmed": False,
        "in_ob": False,
        "near_ob": False,
        "price": 101.0,
        "ema20": 100.0,
        "trade_eval": {
            "trigger_confirmed": False,
            "rejection_confirmed": False,
            "displacement": "NONE",
        },
    }
    setup.update(overrides)
    return setup


def short_setup(**overrides):
    setup = {
        "direction": "SHORT",
        "structureLabel": "Bearish Structure",
        "bos_confirmed": False,
        "in_ob": False,
        "near_ob": False,
        "price": 99.0,
        "ema20": 100.0,
        "trade_eval": {
            "trigger_confirmed": False,
            "rejection_confirmed": False,
            "displacement": "NONE",
        },
    }
    setup.update(overrides)
    return setup


def with_trade_eval(setup, **overrides):
    trade_eval = dict(setup.get("trade_eval") or {})
    trade_eval.update(overrides)
    setup["trade_eval"] = trade_eval
    return setup


def confirmation(setup, setup_direction, setup_status=""):
    return scanner._stock_confirmation(setup, setup_direction, setup_status)


def trigger_confirmed_beats_all_other_signals():
    for label, setup, setup_direction, expected_reason in [
        (
            "long",
            with_trade_eval(
                long_setup(bos_confirmed=True, in_ob=True),
                trigger_confirmed=True,
                rejection_confirmed=True,
                displacement="STRONG",
            ),
            "Bullish",
            "trigger confirmed after pullback",
        ),
        (
            "short",
            with_trade_eval(
                short_setup(bos_confirmed=True, near_ob=True),
                trigger_confirmed=True,
                rejection_confirmed=True,
                displacement="STRONG",
            ),
            "Bearish",
            "trigger confirmed after pullback",
        ),
    ]:
        assert_equal(f"{label} trigger beats all", confirmation(setup, setup_direction), (True, expected_reason))


def rejection_in_zone_beats_bos_for_long_and_short():
    long = with_trade_eval(long_setup(bos_confirmed=True, in_ob=True), rejection_confirmed=True)
    short = with_trade_eval(short_setup(bos_confirmed=True, near_ob=True), rejection_confirmed=True)

    assert_equal(
        "long rejection beats BOS",
        confirmation(long, "Bullish"),
        (True, "strong bullish reaction from demand/support"),
    )
    assert_equal(
        "short rejection beats BOS",
        confirmation(short, "Bearish"),
        (True, "strong bearish reaction from supply/resistance"),
    )


def displacement_beats_bos_when_rejection_absent_for_long_and_short():
    long = with_trade_eval(long_setup(bos_confirmed=True), displacement="STRONG")
    short = with_trade_eval(short_setup(bos_confirmed=True), displacement="STRONG")

    assert_equal(
        "long displacement beats BOS",
        confirmation(long, "Bullish"),
        (True, "strong bullish reaction candle from support"),
    )
    assert_equal(
        "short displacement beats BOS",
        confirmation(short, "Bearish"),
        (True, "strong bearish reaction candle from resistance"),
    )


def bos_beats_ema_reclaim_for_long_and_short():
    assert_equal(
        "long BOS beats EMA reclaim",
        confirmation(long_setup(bos_confirmed=True), "Bullish"),
        (True, "bullish structure shift after pullback"),
    )
    assert_equal(
        "short BOS beats EMA rejection",
        confirmation(short_setup(bos_confirmed=True), "Bearish"),
        (True, "bearish structure shift after pullback"),
    )


def ema_reclaim_beats_status_text_fallback_for_long_and_short():
    assert_equal(
        "long EMA beats status fallback",
        confirmation(long_setup(), "Bullish", "Early Confirmation"),
        (True, "reclaimed EMA20 with bullish short-term flow"),
    )
    assert_equal(
        "short EMA beats status fallback",
        confirmation(short_setup(), "Bearish", "Strong Confirmation"),
        (True, "rejected EMA20 with bearish short-term flow"),
    )


def status_text_fallback_still_works_alone():
    long = long_setup(price=99.0, ema20=100.0)
    short = short_setup(price=101.0, ema20=100.0)

    assert_equal(
        "long status fallback alone",
        confirmation(long, "Bullish", "Trend Resumption"),
        (True, "bullish confirmation started"),
    )
    assert_equal(
        "short status fallback alone",
        confirmation(short, "Bearish", "Early Confirmation"),
        (True, "bearish confirmation started"),
    )


def waiting_result_unchanged_when_no_signal_exists():
    assert_equal(
        "long waiting unchanged",
        confirmation(long_setup(price=99.0, ema20=100.0), "Bullish", "Pullback Active"),
        (False, "waiting for bullish structure shift or support reaction"),
    )
    assert_equal(
        "short waiting unchanged",
        confirmation(short_setup(price=101.0, ema20=100.0), "Bearish", "Pullback Active"),
        (False, "waiting for bearish structure shift or resistance rejection"),
    )


def main() -> int:
    trigger_confirmed_beats_all_other_signals()
    rejection_in_zone_beats_bos_for_long_and_short()
    displacement_beats_bos_when_rejection_absent_for_long_and_short()
    bos_beats_ema_reclaim_for_long_and_short()
    ema_reclaim_beats_status_text_fallback_for_long_and_short()
    status_text_fallback_still_works_alone()
    waiting_result_unchanged_when_no_signal_exists()
    print("Stock confirmation priority v1 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
