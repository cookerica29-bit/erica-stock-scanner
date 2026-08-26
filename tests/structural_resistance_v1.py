#!/usr/bin/env python3
"""Tests for structural_resistance.py -- the gap-day / swing-pivot target
validation used to clamp candidate targets in candidates_router.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scanner import _find_swings  # noqa: E402
from structural_resistance import clamp_target, levels_near_target  # noqa: E402


def _ohlc(rows: list[tuple]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"])
    return df


def test_tigo_gap_reversal_flagged_as_weak():
    # Real TIGO daily bars -- Aug 6 spiked to 107.13 and gave most of it back
    # by Aug 7 (a genuine gap/spike, not organic structure).
    rows = [
        (95.080002, 95.72, 89.78, 94.30),
        (95.18, 95.84, 91.59, 92.86),
        (92.28, 94.15, 91.5, 91.79),
        (99.5, 107.13, 99.5, 104.31),
        (103.68, 104.70, 96, 97.099998),
    ]
    df = _ohlc(rows)
    swings = _find_swings(df, margin=1)
    findings = levels_near_target(df, swings, target_price=107.13, atr=3.33,
                                   direction="long", proximity_atr=0.2)
    assert len(findings) == 1
    assert findings[0]["kind"] == "gap_extreme"
    assert findings[0]["strength"] == "weak"


def test_direction_gating_ignores_the_wrong_side_of_the_bar():
    # Bar 2 (0-indexed): High=108 is NOT a local max (flanked by 114, 116) --
    # not real swing-high structure. Low=50 on that same bar IS a genuine
    # local min. A LONG target sitting on 108 must not get credited with
    # swing-pivot structure just because the bar's low happens to be a pivot.
    rows = [
        (112, 114, 62, 112),
        (112, 114, 58, 112),
        (112, 108, 50, 107),
        (107, 116, 70, 114),
        (114, 116, 72, 114),
    ]
    df = _ohlc(rows)
    swings = _find_swings(df, margin=2)

    long_findings = levels_near_target(df, swings, target_price=108, atr=2,
                                        direction="long", proximity_atr=0.3)
    assert long_findings == [], "108 was never a real swing high -- must not be flagged"

    short_findings = levels_near_target(df, swings, target_price=50, atr=2,
                                         direction="short", proximity_atr=0.3)
    assert len(short_findings) == 1
    assert short_findings[0]["kind"] == "swing_pivot"
    assert short_findings[0]["strength"] == "strong"


def test_swing_pivot_with_hard_move_away_is_strong():
    # A real swing high that price then sold off >= 1 ATR below.
    rows = [
        (100, 102, 99, 101),
        (101, 104, 100, 103),
        (103, 110, 102, 108),  # swing high at 110
        (108, 109, 103, 104),
        (104, 105, 95, 96),    # sold off hard after the high
        (96, 97, 90, 92),
    ]
    df = _ohlc(rows)
    swings = _find_swings(df, margin=2)
    findings = levels_near_target(df, swings, target_price=110, atr=3,
                                   direction="long", proximity_atr=0.2)
    assert len(findings) == 1
    assert findings[0]["kind"] == "swing_pivot"
    assert findings[0]["strength"] == "strong"


def test_clamp_target_pulls_target_back_and_lowers_rr():
    findings = [{"index": 0, "price": 112.0, "kind": "swing_pivot", "strength": "strong",
                 "distance_atr": 0.0, "note": "test"}]
    result = clamp_target(entry=100.0, stop=95.0, target=115.0, atr=5.0,
                           findings=findings, direction="long")
    assert result["clamped"] is True
    assert result["adjusted_target"] < 115.0
    assert result["adjusted_rr"] < result["original_rr"]
    assert result["badge"] == "NEAR REJECTED HIGH"


def test_clamp_target_no_findings_leaves_target_unchanged():
    result = clamp_target(entry=100.0, stop=95.0, target=115.0, atr=5.0,
                           findings=[], direction="long")
    assert result["clamped"] is False
    assert result["adjusted_target"] == 115.0
    assert result["badge"] is None


def test_clamp_target_picks_nearest_not_strongest():
    # A "strong" level sits farther from entry (closer to the raw target)
    # than a "moderate" level that's actually the first obstacle in price's
    # path. Clamping to the "strongest" one would sail right past the
    # nearer, weaker one without acknowledging it.
    findings = [
        {"index": 0, "price": 114.5, "kind": "swing_pivot", "strength": "strong",
         "distance_atr": 0.05, "note": "farther, strong"},
        {"index": 1, "price": 111.0, "kind": "swing_pivot", "strength": "moderate",
         "distance_atr": 0.28, "note": "nearer, moderate -- hit first"},
    ]
    result = clamp_target(entry=100.0, stop=95.0, target=115.0, atr=5.0,
                           findings=findings, direction="long")
    assert result["nearest_finding"]["price"] == 111.0
    assert result["adjusted_target"] <= 111.0, (
        "clamp must respect the nearer (moderate) obstacle, not skip past it "
        "toward the farther (strong) one"
    )


def test_clamp_target_refuses_when_it_would_cross_entry():
    # A large ATR relative to a tight entry-to-target gap can push the
    # buffered clamp below entry. The clamp must refuse rather than emit a
    # long trade whose target sits below its own entry.
    findings = [{"index": 0, "price": 102.0, "kind": "swing_pivot", "strength": "strong",
                 "distance_atr": 0.0, "note": "test"}]
    result = clamp_target(entry=100.0, stop=97.0, target=102.0, atr=30.0,
                           findings=findings, direction="long")
    assert result["clamped"] is False
    assert result["adjusted_target"] == 102.0, "must keep the raw target, not an inverted one"
    assert result["clamp_refused_reason"] is not None
    assert result["badge"] is not None, "finding should still be surfaced even when refused"


def test_clamp_target_refuses_below_min_viable_rr():
    findings = [{"index": 0, "price": 103.0, "kind": "swing_pivot", "strength": "moderate",
                 "distance_atr": 0.0, "note": "test"}]
    result = clamp_target(entry=100.0, stop=95.0, target=110.0, atr=5.0,
                           findings=findings, direction="long",
                           min_viable_rr=1.5)
    # clamp target ~ 103 - buf(0.5) = 102.5 -> reward 2.5 / risk 5 = 0.5 R:R, below 1.5
    assert result["clamped"] is False
    assert result["clamp_refused_reason"] is not None
    assert "R:R" in result["clamp_refused_reason"]


def test_short_direction_symmetry():
    # Mirror of test_swing_pivot_with_hard_move_away_is_strong for shorts:
    # a real swing low that price then rallied >= 1 ATR above.
    rows = [
        (100, 101, 98, 99),
        (99, 100, 96, 97),
        (97, 98, 90, 92),   # swing low at 90
        (92, 97, 91, 96),
        (96, 105, 95, 104),  # rallied hard after the low
        (104, 108, 103, 107),
    ]
    df = _ohlc(rows)
    swings = _find_swings(df, margin=2)
    findings = levels_near_target(df, swings, target_price=90, atr=3,
                                   direction="short", proximity_atr=0.2)
    assert len(findings) == 1
    assert findings[0]["kind"] == "swing_pivot"
    assert findings[0]["strength"] == "strong"

    clamp = clamp_target(entry=95.0, stop=100.0, target=90.0, atr=3.0,
                          findings=findings, direction="short")
    assert clamp["clamped"] is True
    assert clamp["adjusted_target"] > 90.0
    assert clamp["badge"] == "NEAR REJECTED LOW"
