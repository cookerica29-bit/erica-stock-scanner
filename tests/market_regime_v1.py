#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scanner  # noqa: E402


def classify(**overrides):
    base = {
        "ema_direction": "Bullish",
        "structure_trend": "LONG",
        "daily_aligned": False,
        "relative_volume": 1.0,
        "atr_expansion": 1.0,
        "range_context": {
            "inside_tolerance": True,
            "range_low": 100,
            "range_high": 110,
            "price": 105,
            "atr": 2,
            "atr_tolerance": scanner.REGIME_RANGE_ATR_TOLERANCE,
        },
    }
    base.update(overrides)
    return scanner._classify_market_regime_from_components(**base)


def assert_equal(label, actual, expected):
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


# ATO 4H known-good: EMA and swings agree, score clears trending threshold.
ato = classify(
    ema_direction="Bullish",
    structure_trend="LONG",
    daily_aligned=False,
    relative_volume=1.09,
    atr_expansion=1.09,
)
assert_equal("ATO-style regime", ato["regime"], "TRENDING")
assert_equal("ATO-style score", ato["score"], 82)


# BMY 1D: EMA and swings conflict inside broad range context.
bmy_daily = classify(
    ema_direction="Bullish",
    structure_trend="SHORT",
    daily_aligned=True,
    relative_volume=0.91,
    atr_expansion=1.14,
)
assert_equal("BMY daily regime", bmy_daily["regime"], "RANGING")
assert_equal("BMY daily range override", bmy_daily["range_override"], True)


# BMY 4H: EMA directional, swings neutral, still inside range context.
bmy_h4 = classify(
    ema_direction="Bullish",
    structure_trend="NEUTRAL",
    daily_aligned=True,
    relative_volume=0.85,
    atr_expansion=1.11,
)
assert_equal("BMY 4H regime", bmy_h4["regime"], "RANGING")
assert_equal("BMY 4H range override", bmy_h4["range_override"], True)


# CRM 4H known-good: EMA and swings agree, adjusted weights lift it to trending.
crm = classify(
    ema_direction="Bullish",
    structure_trend="LONG",
    daily_aligned=False,
    relative_volume=0.74,
    atr_expansion=0.95,
)
assert_equal("CRM-style regime", crm["regime"], "TRENDING")
assert_equal("CRM-style score", crm["score"], 76)


# AAPL-style transition: conflict, but outside range context remains MIXED.
aapl = classify(
    ema_direction="Bullish",
    structure_trend="SHORT",
    daily_aligned=True,
    relative_volume=0.95,
    atr_expansion=0.93,
    range_context={"inside_tolerance": False},
)
assert_equal("outside-range conflict remains mixed", aapl["regime"], "MIXED")


assert_equal("range ATR tolerance constant", scanner.REGIME_RANGE_ATR_TOLERANCE, 1.25)

print("Market regime classification tests passed")
