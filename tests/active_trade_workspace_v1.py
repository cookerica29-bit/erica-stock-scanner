#!/usr/bin/env python3
"""Active Trade Workspace tests."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import journal_store  # noqa: E402
import main  # noqa: E402
from smart_notifications import SQLiteNotificationRepository  # noqa: E402
from verified_history_store import SQLiteVerifiedHistoryRepository  # noqa: E402


TOKEN = "journal-secret"


def open_entry(**overrides):
    base = {
        "journal_id": "j-1",
        "position_id": "p-1",
        "ticker": "DOW",
        "direction": "SHORT",
        "result": "Open",
        "outcome": "Open",
        "tracking_status": "active",
        "tracking_started_at": "2026-07-23T14:00:00Z",
        "planned_underlying_entry": 31.20,
        "actual_underlying_entry": None,
        "original_stop": 32.15,
        "original_tp1": 29.50,
        "original_tp2": 28.70,
        "original_tp3": 27.90,
        "scanner_timeframe": "4H",
        "setup_grade": "A",
        "actual_option_type": "PUT",
        "actual_strike": 31,
        "actual_expiration": "2026-08-21",
        "actual_quantity": 1,
    }
    base.update(overrides)
    return base


def make_context():
    tmp = tempfile.TemporaryDirectory()
    db_path = str(Path(tmp.name) / "journal.sqlite3")
    journal = journal_store.SQLiteJournalRepository(db_path)
    history = SQLiteVerifiedHistoryRepository(db_path)
    notifications = SQLiteNotificationRepository(db_path)
    previous_journal = main._journal_repository
    previous_history = main._verified_history_repository
    previous_notifications = main._notification_repository
    previous_trade_intel = main._trade_intelligence_dataset
    previous_token = os.environ.get("JOURNAL_ADMIN_TOKEN")
    main._journal_repository = journal
    main._verified_history_repository = history
    main._notification_repository = notifications
    main._trade_intelligence_dataset = lambda force=False: {"verified_records": []}
    os.environ["JOURNAL_ADMIN_TOKEN"] = TOKEN
    client = TestClient(main.app)

    def cleanup():
        main._journal_repository = previous_journal
        main._verified_history_repository = previous_history
        main._notification_repository = previous_notifications
        main._trade_intelligence_dataset = previous_trade_intel
        if previous_token is None:
            os.environ.pop("JOURNAL_ADMIN_TOKEN", None)
        else:
            os.environ["JOURNAL_ADMIN_TOKEN"] = previous_token
        tmp.cleanup()

    return client, journal, history, cleanup


def headers():
    return {"X-Kairos-Admin-Token": TOKEN}


def test_active_trade_discovery_and_tracked_not_entered_label():
    client, journal, _history, cleanup = make_context()
    try:
        journal.create_entry(open_entry(actual_underlying_entry=None, entry_timestamp=""))
        payload = client.get("/api/active-trades", headers=headers()).json()
        assert payload["summary"]["active_records_found"] == 1
        assert payload["summary"]["tracked_but_not_entered"] == 1
        record = payload["records"][0]
        assert record["tracking_state"] == "WATCHING_FOR_ENTRY"
        assert record["plan"]["planned_entry"] == 31.20
        assert record["plan"]["actual_entry"] is None
        assert record["plan"]["stop"] == 32.15
        assert record["plan"]["tp1"] == 29.50
    finally:
        cleanup()


def test_multiple_trades_same_symbol_remain_separate():
    client, journal, _history, cleanup = make_context()
    try:
        journal.create_entry(open_entry(journal_id="j-1", position_id="p-1", actual_underlying_entry=31.15))
        journal.create_entry(open_entry(journal_id="j-2", position_id="p-2", actual_underlying_entry=30.80, planned_underlying_entry=30.90))
        payload = client.get("/api/active-trades", headers=headers()).json()
        ids = sorted(record["id"] for record in payload["records"])
        entries = sorted(record["plan"]["planned_entry"] for record in payload["records"])
        assert ids == ["p-1", "p-2"]
        assert entries == [30.90, 31.20]
    finally:
        cleanup()


def test_detail_includes_guided_chart_position_intelligence_verified_history_and_ti_progress():
    client, journal, _history, cleanup = make_context()
    try:
        journal.create_entry(open_entry(actual_underlying_entry=31.15, position_last_state="HEALTHY", position_max_progress_percent=42))
        detail = client.get("/api/active-trades/p-1", headers=headers()).json()
        assert detail["guided_chart"]["planned_entry"] == 31.20
        assert detail["guided_chart"]["actual_entry"] == 31.15
        assert detail["position_intelligence"]["last_state"] == "HEALTHY"
        assert detail["verified_history"]["pipeline_status"] == "OPEN"
        assert detail["trade_intelligence"]["available"] is False
        assert detail["timeline"]
    finally:
        cleanup()


def test_completion_requires_entry_evidence_and_required_fields():
    client, journal, _history, cleanup = make_context()
    try:
        journal.create_entry(open_entry(actual_underlying_entry=None, entry_timestamp=""))
        rejected = client.post("/api/active-trades/p-1/complete", headers=headers(), json={
            "exit_timestamp": "2026-07-24T14:00:00Z",
            "exit_price": 29.50,
            "exit_reason": "TP1",
        })
        assert rejected.status_code == 422
        entered = journal.update_entry("j-1", {"record_version": journal.get_entry("j-1")["record_version"], "actual_underlying_entry": 31.15, "entry_timestamp": "2026-07-23T14:00:00Z"})
        missing = client.post("/api/active-trades/p-1/complete", headers=headers(), json={"exit_price": 29.50})
        assert missing.status_code == 422
        assert journal.get_entry("j-1")["result"] == "Open"
    finally:
        cleanup()


def test_valid_completion_queues_verified_history_job_without_claiming_verification():
    client, journal, history, cleanup = make_context()
    try:
        journal.create_entry(open_entry(actual_underlying_entry=31.15, entry_timestamp="2026-07-23T14:00:00Z"))
        response = client.post("/api/active-trades/p-1/complete", headers=headers(), json={
            "exit_timestamp": "2026-07-24T14:00:00Z",
            "exit_price": 29.50,
            "exit_reason": "TP1",
            "actual_option_pnl": 125.0,
        })
        assert response.status_code == 200
        payload = response.json()
        assert payload["journal"]["result"] == "Win"
        assert payload["journal"]["outcome"] == "TP1"
        assert payload["verified_history"]["queued"] is True
        assert len(history.list_jobs()) == 1
        summary = client.get("/api/active-trades?include_completed=true", headers=headers()).json()
        completed = [record for record in summary["records"] if record["journal_id"] == "j-1"][0]
        assert completed["completed"] is True
        assert completed["verified_history"]["pipeline_status"] == "REPLAY_QUEUED"
        assert completed["trade_intelligence_eligible"] is False
    finally:
        cleanup()


def test_notes_endpoint_preserves_plan_and_does_not_queue_replay():
    client, journal, history, cleanup = make_context()
    try:
        journal.create_entry(open_entry(actual_underlying_entry=31.15))
        response = client.post("/api/active-trades/p-1/notes", headers=headers(), json={"note": "Reviewed contract fill."})
        assert response.status_code == 200
        updated = journal.get_entry("j-1")
        assert "Reviewed contract fill." in updated["notes"]
        assert updated["planned_underlying_entry"] == 31.20
        assert history.list_jobs() == []
    finally:
        cleanup()


def test_protected_diagnostics():
    client, journal, _history, cleanup = make_context()
    try:
        journal.create_entry(open_entry())
        assert client.get("/api/dev/active-trade-workspace").status_code == 403
        assert client.get("/api/dev/active-trade-workspace", headers={"X-Kairos-Admin-Token": "wrong"}).status_code == 403
        payload = client.get("/api/dev/active-trade-workspace", headers=headers()).json()
        assert payload["version"] == "active-trade-workspace-v1"
        assert payload["summary"]["active_records_found"] == 1
        assert payload["diagnostics"]["verified_history_links"] == 1
    finally:
        cleanup()


if __name__ == "__main__":
    test_active_trade_discovery_and_tracked_not_entered_label()
    test_multiple_trades_same_symbol_remain_separate()
    test_detail_includes_guided_chart_position_intelligence_verified_history_and_ti_progress()
    test_completion_requires_entry_evidence_and_required_fields()
    test_valid_completion_queues_verified_history_job_without_claiming_verification()
    test_notes_endpoint_preserves_plan_and_does_not_queue_replay()
    test_protected_diagnostics()
    print("Active Trade Workspace v1 tests passed")
