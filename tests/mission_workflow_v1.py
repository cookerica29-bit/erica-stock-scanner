#!/usr/bin/env python3
"""Regression tests for mission workflow presentation metadata."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scanner  # noqa: E402


def setup(
    ticker: str,
    rank: int,
    ranking_bucket: str,
    grade: str = "A",
    *,
    lifecycle_state: str | None = None,
    tier: str = "TOP_OPPORTUNITY",
    entry: float = 100.0,
    stop: float = 95.0,
    tp1: float = 110.0,
) -> dict:
    row = {
        "ticker": ticker,
        "timeframe": "4H",
        "direction": "LONG",
        "entry": entry,
        "sl": stop,
        "tp1": tp1,
        "setupGrade": grade,
        "signal_timestamp": "2026-08-05T14:00:00Z",
        "entryStatus": "Tradeable",
        "ranking_status_bucket": ranking_bucket,
        "ranking": {
            "rank": rank,
            "tier": tier,
            "score": 90 - rank,
            "status_bucket": ranking_bucket,
            "priority_bucket": rank,
        },
    }
    if lifecycle_state:
        row["early_entry_shadow"] = {"state": lifecycle_state}
        row["execution_lifecycle_state"] = lifecycle_state
    if ranking_bucket == "ENTER_NOW" and lifecycle_state == "ENTRY_TRIGGERED":
        row["new_entry_signal"] = {
            "bucket": "ENTER_NOW",
            "current_strategy_status": "ENTER_NOW",
            "lifecycle_state": "ENTRY_TRIGGERED",
            "actionable": True,
        }
    return row


def test_bucket_classification_and_strategy_fields_are_read_only():
    rows = [
        setup("NOW", 1, "ENTER_NOW", lifecycle_state="ENTRY_TRIGGERED"),
        setup("AR", 2, "ALMOST_READY"),
        setup("EE", 3, "EARLY_ENTRY", lifecycle_state="EARLY_ENTRY_BUILDING"),
        setup("ET", 4, "EARLY_ENTRY", lifecycle_state="EARLY_TOUCH"),
        setup("RT", 5, "ALMOST_READY", lifecycle_state="WAITING_FOR_RETEST"),
        setup("MISS", 6, "EARLY_ENTRY", lifecycle_state="MISSED_ENTRY"),
        setup("INV", 7, "EARLY_ENTRY", lifecycle_state="INVALIDATED"),
        setup("EXP", 8, "EARLY_ENTRY", lifecycle_state="EXPIRED"),
        setup("WAIT", 9, "WAITING", tier="REVIEW"),
    ]
    original_rankings = [copy.deepcopy(row["ranking"]) for row in rows]

    assert scanner.stock_mission_workflow_bucket(rows[0]) == "TRADE_NOW"
    assert scanner.stock_mission_workflow_bucket(rows[1]) == "WATCH_CLOSELY"
    assert scanner.stock_mission_workflow_bucket(rows[2]) == "WATCH_CLOSELY"
    assert scanner.stock_mission_workflow_bucket(rows[3]) == "WATCH_CLOSELY"
    assert scanner.stock_mission_workflow_bucket(rows[4]) == "WATCH_CLOSELY"
    assert scanner.stock_mission_workflow_bucket(rows[5]) == "RESOLVED"
    assert scanner.stock_mission_workflow_bucket(rows[6]) == "RESOLVED"
    assert scanner.stock_mission_workflow_bucket(rows[7]) == "RESOLVED"
    assert scanner.stock_mission_workflow_bucket(rows[8]) == "WATCH_CLOSELY"

    snapshot = scanner.stock_mission_workflow_snapshot(rows, [], universe="test", update_movements=False)
    assert snapshot["enabled"] is False
    assert snapshot["trade_now_count"] == 1
    assert snapshot["watch_closely_count"] == 5
    assert snapshot["resolved_count"] == 3
    assert [row["ranking"] for row in rows] == original_rankings


def test_movement_diagnostics_are_generation_aware():
    scanner._stock_mission_workflow_previous.pop("mission-test", None)
    watch = setup("MOVE", 1, "ALMOST_READY", lifecycle_state="EARLY_ENTRY_BUILDING")
    first = scanner.stock_mission_workflow_snapshot([watch], [], universe="mission-test", update_movements=True)
    assert first["movements"]["entered_watch_closely"] == 1

    trade_now = setup("MOVE", 1, "ENTER_NOW", lifecycle_state="ENTRY_TRIGGERED")
    second = scanner.stock_mission_workflow_snapshot([trade_now], [], universe="mission-test", update_movements=True)
    assert second["movements"]["watch_closely_to_trade_now"] == 1

    replaced = setup("MOVE", 1, "EARLY_ENTRY", lifecycle_state="EARLY_ENTRY_BUILDING", entry=101.0)
    third = scanner.stock_mission_workflow_snapshot([replaced], [], universe="mission-test", update_movements=True)
    assert third["movements"]["entered_watch_closely"] == 1
    assert third["movements"]["plan_replacements_or_removed"] == 1


if __name__ == "__main__":
    test_bucket_classification_and_strategy_fields_are_read_only()
    test_movement_diagnostics_are_generation_aware()
    print("mission_workflow_v1.py passed")
