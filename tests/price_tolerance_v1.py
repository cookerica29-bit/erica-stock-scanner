#!/usr/bin/env python3
"""Regression tests for daily swing/trend tolerance against provider precision noise."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scanner  # noqa: E402


def assert_equal(label: str, actual, expected) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def assert_true(label: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(label)


def trend_from_points(high1, high2, low1, low2, tolerance=None):
    swing_tolerance = scanner._price_level_tolerance(tolerance)
    swings = [
        {"index": 1, "price": high1, "type": "high"},
        {"index": 2, "price": low1, "type": "low"},
        {"index": 3, "price": high2, "type": "high"},
        {"index": 4, "price": low2, "type": "low"},
    ]
    if swing_tolerance:
        for swing in swings:
            swing["tolerance"] = swing_tolerance
    return scanner._get_trend(swings)


def aal_style_half_cent_daily_lows_are_equivalent():
    tolerance = scanner.SWING_DAILY_PRICE_TOLERANCE
    assert_equal("daily tolerance is fixed, not price/ATR scaled", tolerance, 0.006)
    assert_equal(
        "Yahoo exact-tie lows should not infer a new trend",
        trend_from_points(15.40, 18.79, 13.180000305175781, 13.180000305175781, tolerance),
        "NEUTRAL",
    )
    assert_equal(
        "Alpaca half-cent lower low should be treated as equivalent provider precision noise",
        trend_from_points(15.40, 18.79, 13.18, 13.175, tolerance),
        "NEUTRAL",
    )


def genuine_trend_changes_beyond_fixed_tolerance_still_register():
    tolerance = scanner.SWING_DAILY_PRICE_TOLERANCE
    assert_equal(
        "higher high and higher low beyond tolerance should still be LONG",
        trend_from_points(15.40, 18.79, 13.18, 13.18 + tolerance + 0.001, tolerance),
        "LONG",
    )
    assert_equal(
        "lower high and lower low beyond tolerance should still be SHORT",
        trend_from_points(110.0, 110.0 - tolerance - 0.001, 100.0, 100.0 - tolerance - 0.001, tolerance),
        "SHORT",
    )


def boundary_stress_just_beyond_tolerance_registers_trend_change():
    tolerance = scanner.SWING_DAILY_PRICE_TOLERANCE
    assert_equal(
        "difference just inside tolerance should be neutral",
        trend_from_points(21.0, 21.0 + tolerance * 0.5, 19.0, 19.0 + tolerance * 0.5, tolerance),
        "NEUTRAL",
    )
    assert_equal(
        "difference just beyond tolerance should register as LONG",
        trend_from_points(21.0, 21.0 + tolerance + 0.0001, 19.0, 19.0 + tolerance + 0.0001, tolerance),
        "LONG",
    )


def bac_gs_mcd_boundaries_are_not_swallowed_by_daily_tolerance():
    tolerance = scanner.SWING_DAILY_PRICE_TOLERANCE
    assert_true(
        "BAC one-cent daily low difference must remain meaningful",
        abs(56.849998474121094 - 56.84000015258789) > tolerance,
    )
    assert_true(
        "GS forty-one-cent daily low difference must remain meaningful",
        abs(1000.4500122070312 - 1000.0399780273438) > tolerance,
    )
    assert_true(
        "MCD daily high difference just under one cent must remain meaningful",
        abs(310.43458618702203 - 310.4246540236003) > tolerance,
    )
    assert_true(
        "MCD two-cent daily low difference must remain meaningful",
        abs(264.54998779296875 - 264.5299987792969) > tolerance,
    )


def daily_find_swings_marks_provider_noise_consistently():
    rows = [
        (15.00, 14.50), (15.10, 14.40), (15.20, 14.30), (15.30, 14.20),
        (15.40, 14.10), (15.20, 13.90), (15.00, 13.70), (14.80, 13.50),
        (14.60, 13.18), (14.80, 13.175), (15.00, 13.40), (15.40, 13.80),
        (18.79, 14.50), (18.40, 14.80), (18.20, 15.00), (18.00, 15.20),
        (17.80, 15.40),
    ]
    df = pd.DataFrame(rows, columns=["High", "Low"])
    df["Open"] = df["Low"] + 0.10
    df["Close"] = df["High"] - 0.10
    df["Volume"] = 1000000
    swings = scanner._find_swings(df, margin=2, tolerance=scanner.SWING_DAILY_PRICE_TOLERANCE)
    lows = [s for s in swings if s["type"] == "low"]
    assert_true("daily fixture should detect both near-equal low candidates", len(lows) >= 2)
    assert_equal(
        "near-equal low candidates should not create a false lower-low trend",
        scanner._get_trend(swings),
        "NEUTRAL",
    )


def default_find_swings_remains_exact_for_non_daily_callers():
    rows = [
        (12.20, 11.90), (12.10, 11.80), (12.00, 11.70), (11.90, 11.60),
        (11.80, 11.50), (11.70, 11.40), (11.60, 11.30), (11.50, 11.20),
        (11.40, 11.100000381469727), (11.50, 11.105000267028809),
        (11.60, 11.30), (11.70, 11.40), (11.80, 11.50),
    ]
    df = pd.DataFrame(rows, columns=["High", "Low"])
    df["Open"] = df["Low"] + 0.10
    df["Close"] = df["High"] - 0.10
    df["Volume"] = 1000000
    exact_swings = scanner._find_swings(df, margin=2)
    tolerant_swings = scanner._find_swings(df, margin=2, tolerance=scanner.SWING_DAILY_PRICE_TOLERANCE)
    exact_lows = [(s["index"], s["price"]) for s in exact_swings if s["type"] == "low"]
    tolerant_lows = [(s["index"], s["price"]) for s in tolerant_swings if s["type"] == "low"]
    assert_true("default non-daily swing detection should remain exact", (9, 11.105000267028809) not in exact_lows)
    assert_true("explicit daily tolerance may include a near-equal low", (9, 11.105000267028809) in tolerant_lows)


if __name__ == "__main__":
    aal_style_half_cent_daily_lows_are_equivalent()
    genuine_trend_changes_beyond_fixed_tolerance_still_register()
    boundary_stress_just_beyond_tolerance_registers_trend_change()
    bac_gs_mcd_boundaries_are_not_swallowed_by_daily_tolerance()
    daily_find_swings_marks_provider_noise_consistently()
    default_find_swings_remains_exact_for_non_daily_callers()
    print("Swing price tolerance v1 tests passed")
