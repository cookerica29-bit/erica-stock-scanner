#!/usr/bin/env python3
"""4H market-data forensic tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main  # noqa: E402
import provider_migration_audit as audit  # noqa: E402


TOKEN = "forensic-token"


def make_30m_frame(start: str, periods: int, base: float = 100.0) -> pd.DataFrame:
    index = pd.date_range(start=start, periods=periods, freq="30min", tz="America/New_York")
    rows = []
    for i, _ts in enumerate(index):
        value = base + i
        rows.append({
            "Open": value,
            "High": value + 0.5,
            "Low": value - 0.5,
            "Close": value + 0.25,
            "Volume": 1000 + i,
        })
    return pd.DataFrame(rows, index=index)


def test_regular_session_bucket_construction_and_short_final_bucket():
    frame = make_30m_frame("2026-07-20 09:30", 13)
    candles = audit.reconstruct_4h_candles("SPY", "test", frame)
    assert len(candles) == 2
    first, final = candles
    assert first["bucket_start_et"].endswith("09:30:00-04:00")
    assert first["bucket_end_et"].endswith("13:30:00-04:00")
    assert first["source_bar_count"] == 8
    assert first["expected_source_bar_count"] == 8
    assert first["is_complete"] is True
    assert final["bucket_start_et"].endswith("13:30:00-04:00")
    assert final["bucket_end_et"].endswith("16:00:00-04:00")
    assert final["source_bar_count"] == 5
    assert final["expected_source_bar_count"] == 5
    assert final["session_type"] == "regular_short_final"
    assert final["is_complete"] is True


def test_extended_hours_bucket_construction():
    frame = make_30m_frame("2026-07-20 04:00", 32)
    candles = audit.reconstruct_4h_candles("SPY", "test", frame, include_extended_hours=True)
    assert [c["session_type"] for c in candles] == ["pre_market", "regular_or_extended", "regular_or_extended", "after_hours"]
    assert all(c["source_bar_count"] == 8 for c in candles)
    assert all(c["is_complete"] for c in candles)


def test_incomplete_current_bucket_marks_partial():
    frame = make_30m_frame("2026-07-20 09:30", 6)
    candles = audit.reconstruct_4h_candles("SPY", "test", frame)
    assert len(candles) == 1
    assert candles[0]["source_bar_count"] == 6
    assert candles[0]["expected_source_bar_count"] == 8
    assert candles[0]["is_partial"] is True
    assert candles[0]["is_complete"] is False


def test_early_close_like_missing_final_source_bars():
    frame = make_30m_frame("2026-07-03 09:30", 7)
    candles = audit.reconstruct_4h_candles("SPY", "test", frame)
    assert candles[0]["source_bar_count"] == 7
    assert candles[0]["is_partial"] is True


def test_timezone_and_daylight_saving_offsets_are_preserved():
    summer = audit.reconstruct_4h_candles("SPY", "test", make_30m_frame("2026-07-20 09:30", 8))[0]
    winter = audit.reconstruct_4h_candles("SPY", "test", make_30m_frame("2026-11-09 09:30", 8))[0]
    assert summer["bucket_start_et"].endswith("-04:00")
    assert winter["bucket_start_et"].endswith("-05:00")
    assert summer["bucket_start"].endswith("Z")
    assert winter["bucket_start"].endswith("Z")


def test_interval_based_matching_detects_timestamp_label_only():
    left = audit.reconstruct_4h_candles("SPY", "left", make_30m_frame("2026-07-20 09:30", 8))
    right = [dict(left[0], provider="right", timestamp_label="2026-07-20T17:30:00Z", timestamp_label_et="2026-07-20T13:30:00-04:00")]
    comparison = audit.compare_candle_intervals(left, right, left_name="left", right_name="right")
    assert comparison["classification_counts"] == {"TIMESTAMP_LABEL_ONLY": 1}
    assert comparison["interval_matches"] == 1


def test_interval_based_matching_detects_source_aggregation_difference():
    left = audit.reconstruct_4h_candles("SPY", "left", make_30m_frame("2026-07-20 09:30", 8))
    right = [dict(left[0], provider="right", close=left[0]["close"] + 1.0)]
    comparison = audit.compare_candle_intervals(left, right, left_name="left", right_name="right")
    assert comparison["classification_counts"] == {"SOURCE_AGGREGATION_DIFFERENCE": 1}


def test_interval_based_matching_detects_missing_source_bars():
    left = audit.reconstruct_4h_candles("SPY", "left", make_30m_frame("2026-07-20 09:30", 8))
    comparison = audit.compare_candle_intervals(left, [], left_name="left", right_name="right")
    assert comparison["classification_counts"] == {"MISSING_SOURCE_BARS": 1}


def test_strategy_recommendation_blocks_on_significant_difference():
    interval_counts = {"SOURCE_AGGREGATION_DIFFERENCE": 1}
    strategy_counts = {"STRATEGY_SIGNIFICANT": 1}
    assert audit._four_h_recommendation(interval_counts, strategy_counts) == "BLOCK_ROUTING_DECISION"


def test_protected_4h_forensic_endpoint():
    previous_token = os.environ.get("JOURNAL_ADMIN_TOKEN")
    previous_report = main.four_h_forensics_report
    try:
        os.environ["JOURNAL_ADMIN_TOKEN"] = TOKEN
        main.four_h_forensics_report = lambda **kwargs: {
            "version": audit.FOUR_H_FORENSIC_VERSION,
            "production_routing_changed": False,
            "kwargs": kwargs,
        }
        client = TestClient(main.app)
        assert client.get("/api/dev/provider-comparison/4h-forensics").status_code == 403
        assert client.get("/api/dev/provider-comparison/4h-forensics", headers={"X-Kairos-Admin-Token": "wrong"}).status_code == 403
        response = client.get(
            "/api/dev/provider-comparison/4h-forensics?symbols=SPY,NVDA&start=2026-07-20&end=2026-07-21&include_extended_hours=true&include_strategy=false&limit=2",
            headers={"X-Kairos-Admin-Token": TOKEN},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["version"] == audit.FOUR_H_FORENSIC_VERSION
        assert payload["production_routing_changed"] is False
        assert payload["kwargs"]["symbols"] == ["SPY", "NVDA"]
        assert payload["kwargs"]["start"] == "2026-07-20"
        assert payload["kwargs"]["end"] == "2026-07-21"
        assert payload["kwargs"]["include_extended_hours"] is True
        assert payload["kwargs"]["include_strategy"] is False
        assert payload["kwargs"]["limit"] == 2
    finally:
        main.four_h_forensics_report = previous_report
        if previous_token is None:
            os.environ.pop("JOURNAL_ADMIN_TOKEN", None)
        else:
            os.environ["JOURNAL_ADMIN_TOKEN"] = previous_token


def main_test() -> int:
    test_regular_session_bucket_construction_and_short_final_bucket()
    test_extended_hours_bucket_construction()
    test_incomplete_current_bucket_marks_partial()
    test_early_close_like_missing_final_source_bars()
    test_timezone_and_daylight_saving_offsets_are_preserved()
    test_interval_based_matching_detects_timestamp_label_only()
    test_interval_based_matching_detects_source_aggregation_difference()
    test_interval_based_matching_detects_missing_source_bars()
    test_strategy_recommendation_blocks_on_significant_difference()
    test_protected_4h_forensic_endpoint()
    print("4H market-data forensics v1 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_test())

