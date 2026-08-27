#!/usr/bin/env python3
"""Tests for displacement_score.py -- the continuous 0-100 displacement/
conviction score, replacing the legacy scanner.detect_displacement hard gate
as a grading input (not a gate) for candidate previews/promotions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from displacement_score import score_displacement  # noqa: E402


def _quiet_then_displacement_df() -> pd.DataFrame:
    # 25 quiet, small-range/small-volume bars, then one big bullish
    # displacement bar -- same synthetic construction as the prototype.
    np.random.seed(0)
    n = 26
    base = 100.0
    quiet_opens = base + np.random.normal(0, 0.1, n - 1)
    quiet_closes = quiet_opens + np.random.normal(0, 0.15, n - 1)
    quiet_highs = np.maximum(quiet_opens, quiet_closes) + np.abs(np.random.normal(0, 0.1, n - 1))
    quiet_lows = np.minimum(quiet_opens, quiet_closes) - np.abs(np.random.normal(0, 0.1, n - 1))
    quiet_volumes = np.random.normal(1_000_000, 50_000, n - 1)

    return pd.DataFrame({
        "Open": list(quiet_opens) + [101.0],
        "Close": list(quiet_closes) + [104.5],   # big bullish body
        "High": list(quiet_highs) + [104.7],      # closes near the high -> strong CLV
        "Low": list(quiet_lows) + [100.8],
        "Volume": list(quiet_volumes) + [4_500_000],  # ~4.5x average volume
    })


def test_big_displacement_after_quiet_action_scores_strong():
    df = _quiet_then_displacement_df()
    result = score_displacement(df, direction="long", index=len(df) - 1)
    assert result["label"] == "STRONG"


def test_volume_percentile_reflects_the_expansion():
    df = _quiet_then_displacement_df()
    result = score_displacement(df, direction="long", index=len(df) - 1)
    assert result["components"]["volume_percentile"] >= 95


def test_clv_reflects_close_near_the_high():
    df = _quiet_then_displacement_df()
    result = score_displacement(df, direction="long", index=len(df) - 1)
    assert result["components"]["close_location_value"] > 80


def test_same_candle_on_low_volume_scores_lower():
    df = _quiet_then_displacement_df()
    baseline = score_displacement(df, direction="long", index=len(df) - 1)
    df_low_vol = df.copy()
    df_low_vol.loc[len(df_low_vol) - 1, "Volume"] = 400_000  # below average
    result = score_displacement(df_low_vol, direction="long", index=len(df_low_vol) - 1)
    assert result["score"] < baseline["score"]


def test_indecisive_close_scores_lower_than_close_near_high():
    df = _quiet_then_displacement_df()
    baseline = score_displacement(df, direction="long", index=len(df) - 1)
    df_indecisive = df.copy()
    df_indecisive.loc[len(df_indecisive) - 1, "Close"] = 102.0  # closes mid-range
    result = score_displacement(df_indecisive, direction="long", index=len(df_indecisive) - 1)
    assert result["score"] < baseline["score"]


def test_wrong_direction_candle_does_not_score_strong():
    df = _quiet_then_displacement_df()
    df_wrong_dir = df.copy()
    df_wrong_dir.loc[len(df_wrong_dir) - 1, "Open"] = 104.5
    df_wrong_dir.loc[len(df_wrong_dir) - 1, "Close"] = 101.0
    result = score_displacement(df_wrong_dir, direction="long", index=len(df_wrong_dir) - 1)
    assert result["label"] != "STRONG"


def test_ordinary_quiet_bar_does_not_score_strong():
    df = _quiet_then_displacement_df()
    result = score_displacement(df, direction="long", index=15)
    assert result["label"] != "STRONG"


# --- adaptation-specific coverage, beyond the ported prototype checks -------

def test_index_defaults_to_last_bar():
    df = _quiet_then_displacement_df()
    explicit = score_displacement(df, direction="long", index=len(df) - 1)
    default = score_displacement(df, direction="long")
    assert explicit == default


def test_short_direction_clv_rewards_close_near_the_low():
    # Mirror of the long CLV check: for a short, conviction means closing
    # near the bar's LOW, not its high.
    df = _quiet_then_displacement_df()
    df_short = df.copy()
    # Rebuild the last bar as a big bearish displacement candle closing near its low.
    df_short.loc[len(df_short) - 1, ["Open", "Close", "High", "Low", "Volume"]] = [
        104.0, 100.5, 104.2, 100.3, 4_500_000,
    ]
    result = score_displacement(df_short, direction="short", index=len(df_short) - 1)
    assert result["components"]["close_location_value"] > 80
    assert result["label"] == "STRONG"


def test_insufficient_history_returns_na_not_a_crash():
    df = _quiet_then_displacement_df().iloc[:4]
    result = score_displacement(df, direction="long", index=len(df) - 1)
    assert result["label"] == "N/A"
    assert result["score"] == 0.0


def test_zero_range_bar_handled_without_crash():
    df = _quiet_then_displacement_df()
    flat_idx = len(df) - 1
    df.loc[flat_idx, ["Open", "Close", "High", "Low"]] = [100.0, 100.0, 100.0, 100.0]
    result = score_displacement(df, direction="long", index=flat_idx)
    assert result["label"] == "NONE"
    assert result["score"] == 0.0


# --- displacement_read: the three-way favorable/adverse/quiet split --------
#
# score/label alone can't distinguish "nothing happened" from "something real
# happened against the trade" -- both land in WEAK, and the adverse case can
# even score *lower* than the quiet one, since WRONG_DIRECTION_PENALTY_MULTIPLIER
# discounts an already-computed composite without regard to how large it was
# pre-discount. displacement_read exists specifically so a card can tell these
# apart. These are regression tests for that exact bug, not just new-feature
# coverage -- each one locks in real, previously-confirmed behavior so the
# conflation can't silently reappear.

def test_favorable_read_for_strong_right_direction_candle():
    df = _quiet_then_displacement_df()
    result = score_displacement(df, direction="long", index=len(df) - 1)
    assert result["label"] == "STRONG"
    assert result["displacement_read"] == "favorable"


def test_quiet_read_for_ordinary_bar():
    # A deliberately minimal, controlled bar -- NOT an arbitrary index into
    # the noisy quiet series (checked directly: index 15 in that series
    # happens to be a small red candle that itself clears the "adverse"
    # threshold by chance, which would make this test flaky/wrong). This
    # bar's body/range/volume are all pinned below the quiet history's own
    # range, so it is unambiguously "nothing happened" regardless of the
    # underlying random seed.
    df = _quiet_then_displacement_df()
    df.loc[len(df) - 1, ["Open", "Close", "High", "Low", "Volume"]] = [
        100.0, 100.05, 100.08, 99.98, 1_000_000,
    ]
    result = score_displacement(df, direction="long", index=len(df) - 1)
    assert result["displacement_read"] == "quiet"


def test_adverse_read_for_big_wrong_direction_candle():
    # Same construction as test_wrong_direction_candle_does_not_score_strong
    # (a candle the same magnitude as the STRONG displacement bar, just
    # flipped) -- confirms it reads "adverse", not silently "quiet".
    df = _quiet_then_displacement_df()
    df_wrong_dir = df.copy()
    df_wrong_dir.loc[len(df_wrong_dir) - 1, "Open"] = 104.5
    df_wrong_dir.loc[len(df_wrong_dir) - 1, "Close"] = 101.0
    result = score_displacement(df_wrong_dir, direction="long", index=len(df_wrong_dir) - 1)
    assert result["displacement_read"] == "adverse"


def test_case_a_quiet_vs_case_b_adverse_are_distinguished():
    # The exact regression case from the bug report: a genuinely quiet candle
    # (case A) and a massive candle moving hard the wrong way (case B) used to
    # both land in WEAK, with case B sometimes scoring *lower* than case A --
    # backwards from what a reader would want. displacement_read must
    # distinguish them even though score/label alone still don't.
    np.random.seed(0)
    n = 26
    base = 100.0
    quiet_opens = base + np.random.normal(0, 0.1, n - 1)
    quiet_closes = quiet_opens + np.random.normal(0, 0.15, n - 1)
    quiet_highs = np.maximum(quiet_opens, quiet_closes) + np.abs(np.random.normal(0, 0.1, n - 1))
    quiet_lows = np.minimum(quiet_opens, quiet_closes) - np.abs(np.random.normal(0, 0.1, n - 1))
    quiet_volumes = np.random.normal(1_000_000, 50_000, n - 1)

    def make_df(last_open, last_close, last_high, last_low, last_vol):
        return pd.DataFrame({
            "Open": list(quiet_opens) + [last_open],
            "Close": list(quiet_closes) + [last_close],
            "High": list(quiet_highs) + [last_high],
            "Low": list(quiet_lows) + [last_low],
            "Volume": list(quiet_volumes) + [last_vol],
        })

    case_a_quiet = make_df(100.0, 100.05, 100.15, 99.95, 1_000_000)
    case_b_adverse = make_df(104.5, 101.0, 104.7, 100.8, 4_500_000)

    result_a = score_displacement(case_a_quiet, direction="long", index=len(case_a_quiet) - 1)
    result_b = score_displacement(case_b_adverse, direction="long", index=len(case_b_adverse) - 1)

    assert result_a["label"] == "WEAK"
    assert result_b["label"] == "WEAK"
    assert result_b["score"] < result_a["score"]  # confirms the label-only ambiguity still exists
    assert result_a["displacement_read"] == "quiet"
    assert result_b["displacement_read"] == "adverse"  # ...but displacement_read no longer conflates them


def test_amzn_like_modest_wrong_direction_candle_reads_quiet_not_adverse():
    # Real finding, not assumed: AMZN's actual live components on 2026-08-27
    # (body ~64th percentile, range ~20th, poor CLV ~8-16, near-zero volume
    # percentile, directional=False) produced a pre-penalty raw_magnitude_score
    # in the high-20s to low-30s -- below MODERATE_LABEL_MIN_SCORE (40). This
    # was the original motivating question ("does AMZN's 8.8/WEAK mean
    # 'nothing happened' or 'momentum against the trade'?"). The honest
    # answer, confirmed by re-running the fix against fresh live data rather
    # than assuming: AMZN's real move was genuinely modest, not large --
    # it correctly reads "quiet", not "adverse". The fix distinguishes
    # magnitude from direction; it doesn't relabel every red candle on a
    # long as a warning. "Adverse" is reserved for candles whose magnitude
    # would have been real regardless of direction (see
    # test_adverse_read_for_big_wrong_direction_candle for that case).
    np.random.seed(1)
    n = 26
    base = 100.0
    quiet_opens = base + np.random.normal(0, 0.1, n - 1)
    quiet_closes = quiet_opens + np.random.normal(0, 0.15, n - 1)
    quiet_highs = np.maximum(quiet_opens, quiet_closes) + np.abs(np.random.normal(0, 0.1, n - 1))
    quiet_lows = np.minimum(quiet_opens, quiet_closes) - np.abs(np.random.normal(0, 0.1, n - 1))
    quiet_volumes = np.random.normal(1_000_000, 50_000, n - 1)

    df = pd.DataFrame({
        "Open": list(quiet_opens) + [101.0],
        "Close": list(quiet_closes) + [100.87],   # modest red candle
        "High": list(quiet_highs) + [100.96],
        "Low": list(quiet_lows) + [100.86],
        "Volume": list(quiet_volumes) + [850_000],  # below-average volume, not a real expansion
    })
    result = score_displacement(df, direction="long", index=len(df) - 1)
    assert result["components"]["directional"] is False
    assert result["raw_magnitude_score"] < 40.0
    assert result["displacement_read"] == "quiet"


def test_raw_magnitude_score_not_reduced_by_wrong_direction_penalty():
    # raw_magnitude_score must reflect the candle's real size regardless of
    # direction -- it's meant to be the input displacement_read uses to tell
    # "big candle, wrong way" apart from "nothing happened". If this were
    # accidentally penalized the same way `score` is, adverse candles would
    # collapse right back into looking quiet.
    df = _quiet_then_displacement_df()
    df_wrong_dir = df.copy()
    df_wrong_dir.loc[len(df_wrong_dir) - 1, "Open"] = 104.5
    df_wrong_dir.loc[len(df_wrong_dir) - 1, "Close"] = 101.0
    result = score_displacement(df_wrong_dir, direction="long", index=len(df_wrong_dir) - 1)
    assert result["raw_magnitude_score"] > result["score"]
    assert result["raw_magnitude_score"] >= 70.0  # this candle is objectively large
