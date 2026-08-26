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

from scanner import _find_order_block, _find_swings  # noqa: E402
from structural_resistance import clamp_target, levels_near_target, resolve_stop  # noqa: E402


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


# --- resolve_stop -----------------------------------------------------------

def test_resolve_stop_uses_order_block_when_clean():
    order_block = {"high": 51.5, "low": 50.3, "index": 11}
    result = resolve_stop(entry=58.8, direction="long", atr=1.0, atr_multiplier=1.5,
                           order_block=order_block)
    assert result["stop_source"] == "order_block"
    assert result["stop"] == 50.3 - 0.1 * 1.0  # low - default buffer_atr(0.1)*atr
    assert result["raw_stop"] == 58.8 - 1.5 * 1.0  # flat ATR stop, always populated


def test_resolve_stop_falls_back_when_no_order_block():
    result = resolve_stop(entry=58.8, direction="long", atr=1.0, atr_multiplier=1.5,
                           order_block=None)
    assert result["stop_source"] == "atr_multiple"
    assert result["stop"] == result["raw_stop"]
    assert result["raw_stop"] == 58.8 - 1.5 * 1.0


def test_resolve_stop_falls_back_when_order_block_would_cross_entry():
    # order block's low sits ABOVE entry -- using it would put a long's stop
    # on the wrong side of entry (or coincide with it). Must never emit an
    # un-computable/inverted stop -- fall back to the flat ATR stop instead.
    order_block = {"high": 60.0, "low": 59.0, "index": 5}
    result = resolve_stop(entry=58.8, direction="long", atr=1.0, atr_multiplier=1.5,
                           order_block=order_block)
    assert result["stop_source"] == "atr_multiple"
    assert result["stop"] == result["raw_stop"]


def test_resolve_stop_short_direction_symmetry():
    order_block = {"high": 65.0, "low": 63.0, "index": 11}
    result = resolve_stop(entry=58.8, direction="short", atr=1.0, atr_multiplier=1.5,
                           order_block=order_block)
    assert result["stop_source"] == "order_block"
    assert result["stop"] == 65.0 + 0.1 * 1.0
    assert result["raw_stop"] == 58.8 + 1.5 * 1.0

    # and the short-side crossing-entry fallback
    order_block_bad = {"high": 57.0, "low": 55.0, "index": 5}
    result_bad = resolve_stop(entry=58.8, direction="short", atr=1.0, atr_multiplier=1.5,
                              order_block=order_block_bad)
    assert result_bad["stop_source"] == "atr_multiple"
    assert result_bad["stop"] == result_bad["raw_stop"]


def test_resolve_stop_raw_stop_always_populated_regardless_of_branch():
    flat = 58.8 - 1.5 * 1.0
    for order_block in (None, {"high": 51.5, "low": 50.3, "index": 11}, {"high": 60.0, "low": 59.0, "index": 5}):
        result = resolve_stop(entry=58.8, direction="long", atr=1.0, atr_multiplier=1.5,
                               order_block=order_block)
        assert result["raw_stop"] == flat


def test_find_order_block_integration_with_real_swings_feeds_resolve_stop():
    # Real _find_swings/_find_order_block from scanner.py, not a hand-built
    # dict -- confirms the actual integration, not just resolve_stop in
    # isolation. Bar 10 is a genuine swing low (lowest Low in its window);
    # bar 11 is a bearish candle (Close < Open) between that swing low and
    # the current bar -- the bullish order block _find_order_block should
    # return for a LONG, before the subsequent bullish impulse to bar 19.
    rows = [
        (60.0, 61.0, 59.0, 59.5),
        (59.5, 60.0, 58.0, 58.5),
        (58.5, 59.0, 57.0, 57.5),
        (57.5, 58.0, 56.0, 56.5),
        (56.5, 57.0, 55.0, 55.5),
        (55.5, 56.0, 54.0, 54.5),
        (54.5, 55.0, 53.0, 53.5),
        (53.5, 54.0, 52.0, 52.5),
        (52.5, 53.0, 51.0, 51.5),
        (51.5, 52.0, 50.2, 50.7),
        (50.7, 51.0, 50.0, 50.5),   # bar 10: swing low
        (51.0, 51.5, 50.3, 50.4),   # bar 11: bearish candle -- the order block
        (50.4, 52.0, 50.3, 51.8),
        (51.8, 53.0, 51.5, 52.8),
        (52.8, 54.0, 52.5, 53.8),
        (53.8, 55.0, 53.5, 54.8),
        (54.8, 56.0, 54.5, 55.8),
        (55.8, 57.0, 55.5, 56.8),
        (56.8, 58.0, 56.5, 57.8),
        (57.8, 59.0, 57.5, 58.8),   # bar 19: entry sits near here
    ]
    df = _ohlc(rows)
    swings = _find_swings(df, margin=4)
    order_block = _find_order_block(df, "LONG", swings)
    assert order_block is not None
    assert order_block["index"] == 11
    assert order_block["low"] == 50.3
    assert order_block["high"] == 51.5

    result = resolve_stop(entry=58.8, direction="long", atr=1.0, atr_multiplier=1.5,
                           order_block=order_block)
    assert result["stop_source"] == "order_block"
    assert result["stop"] == 50.3 - 0.1 * 1.0


def test_resolve_stop_falls_back_when_too_tight():
    # Order block's invalidation level is on the correct side of entry, but
    # the resulting stop distance (0.15 ATR) is below MIN_STOP_DISTANCE_ATR
    # (0.25) -- real case that motivated this: AMZN landed at 0.177 ATR
    # before this floor existed. Too tight to be a real invalidation level,
    # not just noise -- must fall back rather than emit it.
    entry = 100.0
    atr = 1.0
    order_block = {"high": 101.0, "low": entry - 0.05, "index": 5}  # low = 99.95
    # candidate_stop = 99.95 - 0.1*1.0 = 99.85 -> distance = 0.15 ATR from entry
    result = resolve_stop(entry=entry, direction="long", atr=atr, atr_multiplier=1.5,
                           order_block=order_block)
    assert result["stop_source"] == "atr_multiple"
    assert result["stop"] == result["raw_stop"]
    assert result["raw_stop"] == entry - 1.5 * atr


def test_resolve_stop_boundary_exactly_at_min_distance_passes():
    # Exactly MIN_STOP_DISTANCE_ATR (0.25) away is deliberately inclusive --
    # "closer than the floor" is what's rejected, not "at or closer than".
    entry = 100.0
    atr = 1.0
    order_block = {"high": 101.0, "low": entry - 0.15, "index": 5}  # low = 99.85
    # candidate_stop = 99.85 - 0.1*1.0 = 99.75 -> distance = exactly 0.25 ATR
    result = resolve_stop(entry=entry, direction="long", atr=atr, atr_multiplier=1.5,
                           order_block=order_block)
    assert result["stop_source"] == "order_block"
    assert result["stop"] == 99.75
