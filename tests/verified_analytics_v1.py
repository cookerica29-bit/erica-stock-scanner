#!/usr/bin/env python3
"""Verified trade analytics tests."""

from __future__ import annotations

import sys
import os
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import journal_store  # noqa: E402
import main  # noqa: E402
from verified_analytics import analytics_verification, pnl_taxonomy, verified_analytics_snapshot  # noqa: E402

TOKEN = "journal-secret"


def entry(**overrides):
    base = {
        "journal_id": "j-1",
        "position_id": "p-1",
        "ticker": "OXY",
        "direction": "SHORT",
        "result": "Win",
        "outcome": "TP1",
        "entry": 54.28,
        "exit": 51.04,
        "actual_underlying_entry": 54.28,
        "planned_underlying_entry": 54.28,
        "plannedStop": 55.9,
        "plannedTp1": 51.04,
        "contracts": 1,
        "actual_quantity": 1,
        "actual_option_premium": 1.88,
        "setupGrade": "A",
    }
    base.update(overrides)
    return base


def replay(**overrides):
    base = {
        "position_id": "p-1",
        "data_complete": True,
        "tp1_timestamp": "2026-07-21T14:00:00Z",
        "stop_timestamp": None,
        "outcome_category": "CALM_WINNER",
        "journal_replay_parity": {"status": "MATCH", "reason": "journal TP1 outcome reproduced by replay"},
    }
    base.update(overrides)
    return base


def test_verified_winner_is_included_in_verified_win_rate():
    snapshot = verified_analytics_snapshot([entry()], [replay()])
    assert snapshot["records"][0]["analytics_verification"]["status"] == "VERIFIED"
    assert snapshot["summary"]["verified"] == 1
    assert snapshot["summary"]["verified_wins"] == 1
    assert snapshot["summary"]["verified_win_rate"] == 100


def test_verified_loss_is_included_in_verified_loss_rate():
    loss = entry(result="Loss", outcome="Stop Loss", exit=55.9)
    stopped = replay(tp1_timestamp=None, stop_timestamp="2026-07-21T14:00:00Z", outcome_category="STOP_DETECTED")
    snapshot = verified_analytics_snapshot([loss], [stopped])
    assert snapshot["records"][0]["analytics_verification"]["status"] == "VERIFIED"
    assert snapshot["summary"]["verified_losses"] == 1
    assert snapshot["summary"]["verified_win_rate"] == 0


def test_oxy_style_mismatch_is_needs_review_not_verified_win():
    stopped = replay(
        tp1_timestamp=None,
        stop_timestamp="2026-07-21T04:00:00Z",
        outcome_category="STOP_DETECTED",
        journal_replay_parity={"status": "JOURNAL_EVENT_UNSUPPORTED", "reason": "journal TP1 outcome was not reproduced by replay candles"},
    )
    snapshot = verified_analytics_snapshot([entry()], [stopped])
    record = snapshot["records"][0]
    assert record["analytics_verification"]["status"] == "JOURNAL_REPLAY_MISMATCH"
    assert record["analytics_verification"]["journal_result"] == "Win"
    assert record["analytics_verification"]["replay_result"] == "Loss"
    assert snapshot["summary"]["needs_review"] == 1
    assert snapshot["summary"]["verified"] == 0
    assert snapshot["summary"]["verified_win_rate"] is None
    assert snapshot["summary"]["journal_win_rate"] == 100


def test_replay_pending_and_insufficient_data_are_not_verified_or_mismatches():
    pending = analytics_verification(entry(), None)
    assert pending["status"] == "REPLAY_PENDING"
    insufficient = analytics_verification(entry(), replay(data_complete=False, journal_replay_parity={"status": "INSUFFICIENT_REPLAY_DATA"}))
    assert insufficient["status"] == "INSUFFICIENT_REPLAY_DATA"
    snapshot = verified_analytics_snapshot([
        entry(position_id="p-1"),
        entry(journal_id="j-2", position_id="p-2"),
    ], [
        replay(position_id="p-2", data_complete=False, journal_replay_parity={"status": "INSUFFICIENT_REPLAY_DATA"}),
    ])
    assert snapshot["summary"]["replay_pending"] == 1
    assert snapshot["summary"]["insufficient_data"] == 1
    assert snapshot["summary"]["verified_win_rate"] is None


def test_missing_exit_option_premium_keeps_actual_pnl_unavailable():
    taxonomy = pnl_taxonomy(entry())
    assert taxonomy["actual_option_pnl"] is None
    assert round(taxonomy["underlying_plan_pnl"], 2) == 324
    assert taxonomy["pnl_source"] == "UNDERLYING_PLAN_LEVELS"
    assert taxonomy["pnl_verification_status"] == "JOURNAL_LEVEL_CALCULATION"


def test_actual_option_pnl_requires_exit_premium():
    taxonomy = pnl_taxonomy(entry(actual_exit_premium=3.12))
    assert round(taxonomy["actual_option_pnl"], 2) == 124
    assert taxonomy["pnl_source"] == "OPTION_PREMIUM_REALIZED"


def test_grouped_aggregation_has_no_double_counting_and_not_available_verified_rate():
    records = [
        entry(journal_id="j-1", position_id="p-1", ticker="OXY"),
        entry(journal_id="j-2", position_id="p-2", ticker="OXY", result="Loss", outcome="Stop Loss", exit=55.9),
    ]
    replays = [
        replay(position_id="p-1", tp1_timestamp=None, stop_timestamp="2026-07-21T04:00:00Z", outcome_category="STOP_DETECTED", journal_replay_parity={"status": "JOURNAL_EVENT_UNSUPPORTED"}),
        replay(position_id="p-2", tp1_timestamp=None, stop_timestamp="2026-07-22T04:00:00Z", outcome_category="STOP_DETECTED", journal_replay_parity={"status": "MATCH"}),
    ]
    snapshot = verified_analytics_snapshot(records, replays)
    oxy = snapshot["top_tickers"][0]
    assert oxy["key"] == "OXY"
    assert oxy["trades"] == 2
    assert oxy["completed"] == 2
    assert oxy["verified"] == 1
    assert oxy["needs_review"] == 1
    assert oxy["journal_win_rate"] == 50
    assert oxy["verified_win_rate"] == 0


def test_verified_analytics_endpoint_is_protected_and_uses_replay_parity():
    tmp = tempfile.TemporaryDirectory()
    previous_repo = main._journal_repository
    previous_replay = main._replay_positions
    previous_token = os.environ.get("JOURNAL_ADMIN_TOKEN")
    try:
        main._journal_repository = journal_store.SQLiteJournalRepository(str(Path(tmp.name) / "journal.sqlite3"))
        main._journal_repository.create_entry(entry())
        os.environ["JOURNAL_ADMIN_TOKEN"] = TOKEN

        def fake_replay(entries, summary_only=True):
            return [replay(
                position_id=entries[0]["position_id"],
                tp1_timestamp=None,
                stop_timestamp="2026-07-21T04:00:00Z",
                outcome_category="STOP_DETECTED",
                journal_replay_parity={"status": "JOURNAL_EVENT_UNSUPPORTED"},
            )]

        main._replay_positions = fake_replay
        client = TestClient(main.app)
        assert client.get("/api/dev/verified-analytics").status_code == 403
        payload = client.get("/api/dev/verified-analytics", headers={"X-Kairos-Admin-Token": TOKEN}).json()
        assert payload["ready"] is True
        assert payload["summary"]["completed"] == 1
        assert payload["summary"]["needs_review"] == 1
        assert payload["summary"]["verified_win_rate"] is None
        assert payload["records"][0]["analytics_verification"]["status"] == "JOURNAL_REPLAY_MISMATCH"
    finally:
        main._journal_repository = previous_repo
        main._replay_positions = previous_replay
        if previous_token is None:
            os.environ.pop("JOURNAL_ADMIN_TOKEN", None)
        else:
            os.environ["JOURNAL_ADMIN_TOKEN"] = previous_token
        tmp.cleanup()


if __name__ == "__main__":
    test_verified_winner_is_included_in_verified_win_rate()
    test_verified_loss_is_included_in_verified_loss_rate()
    test_oxy_style_mismatch_is_needs_review_not_verified_win()
    test_replay_pending_and_insufficient_data_are_not_verified_or_mismatches()
    test_missing_exit_option_premium_keeps_actual_pnl_unavailable()
    test_actual_option_pnl_requires_exit_premium()
    test_grouped_aggregation_has_no_double_counting_and_not_available_verified_rate()
    test_verified_analytics_endpoint_is_protected_and_uses_replay_parity()
    print("Verified analytics v1 tests passed")
