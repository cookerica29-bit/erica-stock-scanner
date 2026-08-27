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
