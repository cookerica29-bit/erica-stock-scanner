"""location_score.score_location -- pure premium/discount percentile logic.

Scope: the module's own math (percentile, label bucket, direction-relative
alignment) and its parity with scanner._latest_swing_range/_location_read/
_strict_location's real percentile output on real tickers (see session
notes: zero mismatches across 30 real tickers before this was trusted).
Router wiring is covered separately in
tests/candidates_router_location_v1.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from location_score import DISCOUNT_THRESHOLD, PREMIUM_THRESHOLD, score_location

# Deliberately round numbers: last 3 highs [95, 100, 110] -> range high 110;
# last 3 lows [90, 92, 96] -> range low 90. Span 20, so price=90 -> 0%,
# price=100 -> 50%, price=110 -> 100%, with no rounding noise to chase.
SWINGS = [
    {"index": 5, "type": "low", "price": 90.0},
    {"index": 10, "type": "high", "price": 95.0},
    {"index": 15, "type": "low", "price": 92.0},
    {"index": 20, "type": "high", "price": 100.0},
    {"index": 25, "type": "low", "price": 96.0},
    {"index": 30, "type": "high", "price": 110.0},
]


def test_range_uses_max_of_last_three_highs_and_min_of_last_three_lows():
    result = score_location(100.0, SWINGS, "long")
    assert result["range_high"] == 110.0
    assert result["range_low"] == 90.0


def test_percentile_at_range_low_is_zero():
    result = score_location(90.0, SWINGS, "long")
    assert result["location_percentile"] == 0.0
    assert result["location_label"] == "discount"


def test_percentile_at_range_high_is_hundred():
    result = score_location(110.0, SWINGS, "long")
    assert result["location_percentile"] == 100.0
    assert result["location_label"] == "premium"


def test_percentile_at_midpoint_is_fifty_and_midrange():
    result = score_location(100.0, SWINGS, "long")
    assert result["location_percentile"] == 50.0
    assert result["location_label"] == "midrange"


def test_discount_is_favorable_for_long_and_unfavorable_for_short():
    long_result = score_location(90.0, SWINGS, "long")
    short_result = score_location(90.0, SWINGS, "short")
    assert long_result["location_alignment"] == "favorable"
    assert short_result["location_alignment"] == "unfavorable"


def test_premium_is_unfavorable_for_long_and_favorable_for_short():
    long_result = score_location(110.0, SWINGS, "long")
    short_result = score_location(110.0, SWINGS, "short")
    assert long_result["location_alignment"] == "unfavorable"
    assert short_result["location_alignment"] == "favorable"


def test_midrange_is_neutral_for_both_directions():
    long_result = score_location(100.0, SWINGS, "long")
    short_result = score_location(100.0, SWINGS, "short")
    assert long_result["location_alignment"] == "neutral"
    assert short_result["location_alignment"] == "neutral"


def test_label_thresholds_match_module_constants():
    just_below_premium = score_location(90.0 + (PREMIUM_THRESHOLD - 0.1) / 100 * 20, SWINGS, "long")
    assert just_below_premium["location_label"] == "midrange"
    at_premium = score_location(90.0 + PREMIUM_THRESHOLD / 100 * 20, SWINGS, "long")
    assert at_premium["location_label"] == "premium"

    just_above_discount = score_location(90.0 + (DISCOUNT_THRESHOLD + 0.1) / 100 * 20, SWINGS, "long")
    assert just_above_discount["location_label"] == "midrange"
    at_discount = score_location(90.0 + DISCOUNT_THRESHOLD / 100 * 20, SWINGS, "long")
    assert at_discount["location_label"] == "discount"


def test_percentile_is_clamped_beyond_the_range():
    below = score_location(50.0, SWINGS, "long")
    above = score_location(200.0, SWINGS, "long")
    assert below["location_percentile"] == 0.0
    assert above["location_percentile"] == 100.0


def test_no_valid_range_returns_all_none_with_no_highs():
    result = score_location(100.0, [{"index": 1, "type": "low", "price": 90.0}], "long")
    assert result == {
        "location_percentile": None, "location_label": None,
        "location_alignment": None, "range_high": None, "range_low": None,
    }


def test_no_valid_range_returns_all_none_with_no_lows():
    result = score_location(100.0, [{"index": 1, "type": "high", "price": 90.0}], "long")
    assert result["location_percentile"] is None


def test_no_valid_range_returns_all_none_for_empty_swings():
    result = score_location(100.0, [], "long")
    assert result["location_percentile"] is None


def test_degenerate_inverted_range_returns_none():
    """high <= low (e.g. all the recent highs happen to sit below the
    recent lows) is treated as no valid range, same as scanner.
    _latest_swing_range's own guard."""
    degenerate = [
        {"index": 1, "type": "high", "price": 80.0},
        {"index": 2, "type": "low", "price": 90.0},
    ]
    result = score_location(85.0, degenerate, "long")
    assert result["location_percentile"] is None


def test_unsupported_direction_still_computes_percentile_but_no_alignment():
    # Midrange's alignment is "neutral" regardless of direction validity
    # (there's nothing to be favorable/unfavorable about) -- use a
    # discount price instead, where alignment genuinely depends on
    # recognizing the direction.
    result = score_location(90.0, SWINGS, "sideways")
    assert result["location_percentile"] == 0.0
    assert result["location_label"] == "discount"
    assert result["location_alignment"] is None
