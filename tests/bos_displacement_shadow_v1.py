#!/usr/bin/env python3
"""Regression tests for BOS-anchored displacement shadow diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scanner  # noqa: E402


def candle_frame(rows):
    index = pd.date_range("2026-01-01", periods=len(rows), freq="D", tz="UTC")
    return pd.DataFrame(rows, index=index)


def base_rows(count=28, price=100.0):
    rows = []
    for i in range(count):
        open_ = price + (i * 0.05)
        close = open_ + (0.1 if i % 2 == 0 else -0.1)
        rows.append({
            "Open": open_,
            "High": max(open_, close) + 0.6,
            "Low": min(open_, close) - 0.6,
            "Close": close,
            "Volume": 1000000,
        })
    return rows


def bullish_bos_df(*, older_than_three=True):
    rows = base_rows()
    rows[13].update({"Open": 103.0, "High": 103.5, "Low": 102.4, "Close": 102.7})
    rows[14].update({"Open": 102.7, "High": 103.2, "Low": 102.0, "Close": 102.2})
    rows[15].update({"Open": 103.0, "High": 108.8, "Low": 102.8, "Close": 108.0})
    rows[16].update({"Open": 108.0, "High": 109.0, "Low": 107.5, "Close": 108.6})
    for i in range(17, len(rows)):
        open_ = 108.6 - ((i - 16) * 0.12)
        close = open_ - 0.08
        rows[i].update({"Open": open_, "High": open_ + 0.5, "Low": close - 0.5, "Close": close})
    if older_than_three:
        rows[25].update({"Open": 108.0, "High": 108.4, "Low": 107.4, "Close": 107.8})
        rows[26].update({"Open": 107.8, "High": 108.1, "Low": 107.1, "Close": 107.5})
        rows[27].update({"Open": 107.5, "High": 107.9, "Low": 106.9, "Close": 107.2})
    return candle_frame(rows)


def bearish_bos_df():
    rows = base_rows(price=120.0)
    rows[13].update({"Open": 112.0, "High": 112.8, "Low": 111.7, "Close": 112.4})
    rows[14].update({"Open": 112.4, "High": 113.0, "Low": 111.9, "Close": 112.8})
    rows[15].update({"Open": 112.0, "High": 112.2, "Low": 105.0, "Close": 106.0})
    rows[16].update({"Open": 106.0, "High": 106.5, "Low": 104.8, "Close": 105.4})
    return candle_frame(rows)


def with_swings(swings):
    original = scanner._find_swings
    scanner._find_swings = lambda df, margin=2, tolerance=None: list(swings)
    return original


def restore_swings(original):
    scanner._find_swings = original


def bullish_swings():
    return [
        {"type": "high", "index": 5, "price": 105.0},
        {"type": "low", "index": 8, "price": 99.0},
        {"type": "high", "index": 10, "price": 106.0},
        {"type": "low", "index": 14, "price": 102.0},
    ]


def bearish_swings():
    return [
        {"type": "low", "index": 5, "price": 111.0},
        {"type": "high", "index": 8, "price": 121.0},
        {"type": "low", "index": 10, "price": 110.0},
        {"type": "high", "index": 14, "price": 113.0},
    ]


def test_bullish_bos_candle_strong():
    df = bullish_bos_df(older_than_three=False)
    original = with_swings(bullish_swings())
    try:
        events = scanner.detect_bos_events_with_displacement(df, direction="LONG", evaluation_index=16)
    finally:
        restore_swings(original)
    assert len(events) == 1
    assert events[0]["direction"] == "LONG"
    assert events[0]["bos_candle_only_strength"]["displacement"] == "STRONG"
    assert events[0]["bos_impulse_window_strength"]["displacement"] == "STRONG"
    assert events[0]["atr_source"] == "bos_candle_close"


def test_bearish_bos_candle_strong():
    df = bearish_bos_df()
    original = with_swings(bearish_swings())
    try:
        events = scanner.detect_bos_events_with_displacement(df, direction="SHORT", evaluation_index=16)
    finally:
        restore_swings(original)
    assert len(events) == 1
    assert events[0]["direction"] == "SHORT"
    assert events[0]["bos_candle_only_strength"]["displacement"] == "STRONG"


def test_unknown_when_atr_history_is_insufficient():
    df = bullish_bos_df(older_than_three=False).iloc[:12]
    df.iloc[9, df.columns.get_loc("Close")] = 100.5
    df.iloc[10, df.columns.get_loc("Open")] = 100.8
    df.iloc[10, df.columns.get_loc("Close")] = 102.8
    df.iloc[10, df.columns.get_loc("High")] = 103.2
    original = with_swings([
        {"type": "high", "index": 2, "price": 101.0},
        {"type": "high", "index": 5, "price": 102.0},
    ])
    try:
        events = scanner.detect_bos_events_with_displacement(df, direction="LONG", evaluation_index=10)
    finally:
        restore_swings(original)
    assert events
    assert events[0]["bos_displacement"] == "UNKNOWN"


def test_strong_bos_older_than_three_candles_remains_valid_shadow_only():
    df = bullish_bos_df(older_than_three=True)
    original = with_swings(bullish_swings())
    row = {
        "ticker": "TEST",
        "direction": "LONG",
        "trend": "LONG",
        "timeframe": "1D",
        "bos_confirmed": True,
        "entry": 103.0,
        "sl": 101.0,
        "tp1": 107.0,
        "entryStatus": "Tradeable",
        "setupGrade": "A",
        "ob_low": 102.0,
        "ob_high": 103.2,
        "trade_eval": {
            "trade_stage": "BUILDING / WATCHLIST",
            "displacement": "NONE",
            "displacement_score": 0.1,
            "sweep_taken": True,
            "rejection_confirmed": True,
            "htf_aligned": True,
            "valid_zone": True,
            "structure_quality": "CLEAN BOS",
            "setup_type": "REVERSAL: sweep + rejection + displacement",
            "room_to_target": {"estimated_rr": 2.2, "clear": True, "blocked": False},
            "b_plus_tradeable": False,
            "missing_for_a_plus": ["Needs strong displacement"],
            "no_trade_reasons": [],
            "trigger_confirmed": False,
            "a_plus_ready": False,
        },
    }
    try:
        trace = scanner.bos_displacement_shadow_for_setup(row, df)
    finally:
        restore_swings(original)
    assert trace["current_displacement"] == "NONE"
    assert trace["bos_linked_displacement"] == "STRONG"
    assert trace["would_recover_strong_displacement"] is True
    assert trace["shadow_status_if_bos_displacement_used"] == "ENTER_NOW"
    assert row["trade_eval"]["displacement"] == "NONE"
    assert row["trade_eval"]["a_plus_ready"] is False


def test_other_blockers_still_block_enter_now():
    df = bullish_bos_df(older_than_three=True)
    original = with_swings(bullish_swings())
    row = {
        "ticker": "TEST",
        "direction": "LONG",
        "trend": "LONG",
        "timeframe": "1D",
        "bos_confirmed": True,
        "entry": 103.0,
        "sl": 101.0,
        "tp1": 107.0,
        "entryStatus": "Tradeable",
        "setupGrade": "C",
        "trade_eval": {
            "trade_stage": "RANGE / NO TRADE",
            "displacement": "NONE",
            "sweep_taken": True,
            "rejection_confirmed": True,
            "htf_aligned": True,
            "valid_zone": False,
            "structure_quality": "CLEAN BOS",
            "setup_type": "REVERSAL: sweep + rejection + displacement",
            "room_to_target": {"estimated_rr": 2.2, "clear": True, "blocked": False},
            "b_plus_tradeable": False,
            "missing_for_a_plus": ["Needs strong displacement", "Needs valid premium/discount location"],
            "no_trade_reasons": ["Poor premium/discount location"],
            "trigger_confirmed": False,
            "a_plus_ready": False,
        },
    }
    try:
        trace = scanner.bos_displacement_shadow_for_setup(row, df)
    finally:
        restore_swings(original)
    assert trace["would_recover_strong_displacement"] is True
    assert trace["shadow_status_if_bos_displacement_used"] != "ENTER_NOW"
    assert "Needs valid premium/discount location" in trace["shadow_missing_requirements"]


def test_opposite_bos_invalidates_prior_evidence():
    df = bullish_bos_df(older_than_three=True)
    df.iloc[20, df.columns.get_loc("Open")] = 105.0
    df.iloc[20, df.columns.get_loc("Close")] = 98.0
    df.iloc[20, df.columns.get_loc("Low")] = 97.0
    original = with_swings(bullish_swings() + [
        {"type": "low", "index": 16, "price": 100.0},
        {"type": "low", "index": 18, "price": 99.5},
    ])
    try:
        events = scanner.detect_bos_events_with_displacement(df, evaluation_index=27)
    finally:
        restore_swings(original)
    long_event = [event for event in events if event["direction"] == "LONG"][0]
    assert long_event["invalidation_reason"] == "OPPOSITE_BOS"


def test_ob_close_through_invalidates_evidence():
    df = bullish_bos_df(older_than_three=True)
    df.iloc[18, df.columns.get_loc("Open")] = 103.0
    df.iloc[18, df.columns.get_loc("Close")] = 101.0
    df.iloc[18, df.columns.get_loc("Low")] = 100.5
    original = with_swings(bullish_swings())
    try:
        events = scanner.detect_bos_events_with_displacement(df, direction="LONG", evaluation_index=27)
    finally:
        restore_swings(original)
    assert events[0]["invalidation_reason"] == "OB_INVALIDATED"


def test_new_same_direction_bos_supersedes_older_bos():
    df = bullish_bos_df(older_than_three=True)
    df.iloc[21, df.columns.get_loc("Open")] = 108.0
    df.iloc[21, df.columns.get_loc("Close")] = 112.0
    df.iloc[21, df.columns.get_loc("High")] = 113.0
    original = with_swings(bullish_swings() + [
        {"type": "high", "index": 17, "price": 109.0},
        {"type": "high", "index": 19, "price": 110.0},
    ])
    try:
        events = scanner.detect_bos_events_with_displacement(df, direction="LONG", evaluation_index=27)
    finally:
        restore_swings(original)
    assert len(events) >= 2
    assert events[0]["invalidation_reason"] == "SUPERSEDED_BY_NEWER_SAME_DIRECTION_BOS"


def test_future_following_candle_excluded_in_replay_window():
    df = bullish_bos_df(older_than_three=False)
    df.iloc[16, df.columns.get_loc("Open")] = 108.0
    df.iloc[16, df.columns.get_loc("Close")] = 114.0
    original = with_swings(bullish_swings())
    try:
        at_bos = scanner.detect_bos_events_with_displacement(df, direction="LONG", evaluation_index=15)[0]
        after_next = scanner.detect_bos_events_with_displacement(df, direction="LONG", evaluation_index=16)[0]
    finally:
        restore_swings(original)
    assert at_bos["displacement_window_end"] == 15
    assert after_next["displacement_window_end"] == 16


def test_shadow_report_counts_recoveries():
    df = bullish_bos_df(older_than_three=True)
    original = with_swings(bullish_swings())
    row = {
        "ticker": "TEST",
        "direction": "LONG",
        "timeframe": "1D",
        "setupGrade": "A",
        "bos_confirmed": True,
        "entry": 103.0,
        "sl": 101.0,
        "tp1": 107.0,
        "entryStatus": "Tradeable",
        "option_plan": {"available": True},
        "trade_eval": {
            "trade_stage": "BUILDING / WATCHLIST",
            "displacement": "NONE",
            "sweep_taken": True,
            "rejection_confirmed": True,
            "htf_aligned": True,
            "valid_zone": True,
            "structure_quality": "CLEAN BOS",
            "location": "NEAR DISCOUNT",
            "setup_type": "REVERSAL: sweep + rejection + displacement",
            "room_to_target": {"estimated_rr": 2.2, "clear": True, "blocked": False},
            "b_plus_tradeable": False,
            "missing_for_a_plus": ["Needs strong displacement"],
            "no_trade_reasons": [],
        },
    }
    try:
        report = scanner.build_bos_displacement_shadow_report([row], {"TEST": df})
    finally:
        restore_swings(original)
    assert report["message"] == "Shadow study only. Live strategy unchanged."
    assert report["recovered_strong_displacement_setups"] == 1
    assert report["enter_now_recoveries"] == 1


def main() -> int:
    test_bullish_bos_candle_strong()
    test_bearish_bos_candle_strong()
    test_unknown_when_atr_history_is_insufficient()
    test_strong_bos_older_than_three_candles_remains_valid_shadow_only()
    test_other_blockers_still_block_enter_now()
    test_opposite_bos_invalidates_prior_evidence()
    test_ob_close_through_invalidates_evidence()
    test_new_same_direction_bos_supersedes_older_bos()
    test_future_following_candle_excluded_in_replay_window()
    test_shadow_report_counts_recoveries()
    print("BOS displacement shadow v1 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
