import os
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main
from journal_store import SQLiteJournalRepository
from smart_notifications import SQLiteNotificationRepository, build_setup_event, setup_identity


def setup(**overrides):
    base = {
        "ticker": "DOW",
        "direction": "SHORT",
        "setupGrade": "A",
        "scanner_status": "WAITING",
        "signal_timestamp": "2026-07-23T14:00:00Z",
        "entry": 31.20,
        "price": 31.80,
        "stop": 32.15,
        "tp1": 29.50,
    }
    base.update(overrides)
    return base


def temp_repo():
    handle = tempfile.NamedTemporaryFile(delete=False)
    handle.close()
    return SQLiteNotificationRepository(handle.name), handle.name


def test_enter_now_transition_creates_one_event_then_dedupes():
    repo, path = temp_repo()
    try:
        first = setup(scanner_status="ALMOST_READY")
        assert repo.evaluate_scan([first], {"scan_completed_at": "2026-07-23T14:00:00Z"})["events_created"] == 0

        current = setup(scanner_status="ENTER_NOW")
        created = repo.evaluate_scan([current], {"scan_completed_at": "2026-07-23T14:05:00Z"})
        assert created["events_created"] == 1
        assert created["created_events"][0]["event_type"] == "NEW_ENTER_NOW"
        assert "Set an alert at the planned entry and wait." in created["created_events"][0]["next_step"]

        duplicate = repo.evaluate_scan([current], {"scan_completed_at": "2026-07-23T14:10:00Z"})
        assert duplicate["events_created"] == 0
    finally:
        os.unlink(path)


def test_planned_entry_reached_is_direction_aware_and_not_execution_claim():
    repo, path = temp_repo()
    try:
        waiting_short = setup(scanner_status="ENTER_NOW", price=31.80)
        touched_short = setup(scanner_status="ENTER_NOW", price=31.18)
        repo.evaluate_scan([waiting_short], {})
        event = repo.evaluate_scan([touched_short], {})["created_events"][0]
        assert event["event_type"] == "PLANNED_ENTRY_REACHED"
        assert event["relevant_level"] == 31.20
        assert "before executing" in event["next_step"]
        assert "entered" not in event["message"].lower()

        waiting_long = setup(ticker="REGN", direction="LONG", scanner_status="ENTER_NOW", entry=653.07, price=650.0, tp1=670.13)
        touched_long = setup(ticker="REGN", direction="LONG", scanner_status="ENTER_NOW", entry=653.07, price=653.08, tp1=670.13)
        repo.evaluate_scan([waiting_long], {})
        long_event = repo.evaluate_scan([touched_long], {})["created_events"][0]
        assert long_event["event_type"] == "PLANNED_ENTRY_REACHED"
    finally:
        os.unlink(path)


def test_setup_invalidated_before_entry_is_not_trade_loss():
    repo, path = temp_repo()
    try:
        repo.evaluate_scan([setup(scanner_status="WAITING")], {})
        event = repo.evaluate_scan([setup(scanner_status="SKIP")], {})["created_events"][0]
        assert event["event_type"] == "SETUP_INVALIDATED"
        assert "trade loss" not in event["message"].lower()
        assert "Remove the entry alert" in event["next_step"]
    finally:
        os.unlink(path)


def test_dedupe_survives_repository_reopen():
    repo, path = temp_repo()
    try:
        event = build_setup_event("NEW_ENTER_NOW", setup(scanner_status="ENTER_NOW"), "WAITING", "ENTER_NOW", {})
        saved, inserted = repo.create_event(event)
        assert inserted is True
        reopened = SQLiteNotificationRepository(path)
        saved_again, inserted_again = reopened.create_event(event)
        assert saved_again is None
        assert inserted_again is False
        assert reopened.list_events()["total_count"] == 1
    finally:
        os.unlink(path)


def test_preferences_disable_in_app_without_affecting_state_transition():
    repo, path = temp_repo()
    try:
        prefs = repo.update_preferences({"NEW_ENTER_NOW": {"in_app": False}})
        assert prefs["NEW_ENTER_NOW"]["in_app"] is False
        repo.evaluate_scan([setup(scanner_status="WAITING")], {})
        result = repo.evaluate_scan([setup(scanner_status="ENTER_NOW")], {})
        assert result["events_created"] == 0
        assert repo.state_for(setup_identity(setup(scanner_status="ENTER_NOW")))["state"] == "ENTER_NOW"
    finally:
        os.unlink(path)


def test_notification_endpoints_require_token_and_accept_correct_token():
    repo, path = temp_repo()
    original_repo = main._notification_repository
    original_token = os.environ.get("JOURNAL_ADMIN_TOKEN")
    try:
        main._notification_repository = repo
        os.environ["JOURNAL_ADMIN_TOKEN"] = "test-token"
        client = TestClient(main.app)
        assert client.get("/api/notifications").status_code == 403
        assert client.get("/api/notifications", headers={"X-Kairos-Admin-Token": "wrong"}).status_code == 403
        ok = client.get("/api/notifications", headers={"X-Kairos-Admin-Token": "test-token"})
        assert ok.status_code == 200
        assert ok.json()["version"] == "smart-notifications-v1"
    finally:
        main._notification_repository = original_repo
        if original_token is None:
            os.environ.pop("JOURNAL_ADMIN_TOKEN", None)
        else:
            os.environ["JOURNAL_ADMIN_TOKEN"] = original_token
        os.unlink(path)


def test_journal_update_creates_tp_stop_and_position_transition_events():
    notif_repo, notif_path = temp_repo()
    journal_handle = tempfile.NamedTemporaryFile(delete=False)
    journal_path = journal_handle.name
    journal_handle.close()
    original_notif_repo = main._notification_repository
    original_journal_repo = main._journal_repository
    original_token = os.environ.get("JOURNAL_ADMIN_TOKEN")
    try:
        main._notification_repository = notif_repo
        main._journal_repository = SQLiteJournalRepository(journal_path)
        os.environ["JOURNAL_ADMIN_TOKEN"] = "test-token"
        client = TestClient(main.app)
        headers = {"X-Kairos-Admin-Token": "test-token"}
        created = client.post("/api/journal", headers=headers, json={
            "journal_id": "journal-dow",
            "position_id": "position-dow",
            "ticker": "DOW",
            "direction": "SHORT",
            "status": "Open",
            "entry_timestamp": "2026-07-23T14:00:00Z",
            "planned_underlying_entry": 31.2,
            "original_stop": 32.15,
            "original_tp1": 29.5,
            "position_last_state": "HEALTHY",
        })
        assert created.status_code == 200
        version = created.json()["record_version"]

        tp1 = client.patch("/api/journal/journal-dow", headers=headers, json={
            "record_version": version,
            "first_target_touch_at": "2026-07-23T15:00:00Z",
        })
        assert tp1.status_code == 200
        assert tp1.json()["smart_notifications"]["events_created"] == 1
        events = notif_repo.list_events()["events"]
        assert events[0]["event_type"] == "TP1_REACHED"
        assert "follow your management plan" in events[0]["next_step"]

        version = tp1.json()["record_version"]
        stop = client.patch("/api/journal/journal-dow", headers=headers, json={
            "record_version": version,
            "first_stop_touch_at": "2026-07-23T16:00:00Z",
        })
        assert stop.status_code == 200
        assert any(event["event_type"] == "STOP_REACHED" for event in notif_repo.list_events()["events"])

        version = stop.json()["record_version"]
        status = client.patch("/api/journal/journal-dow", headers=headers, json={
            "record_version": version,
            "position_last_state": "WATCH",
            "last_evaluated_at": "2026-07-23T16:05:00Z",
        })
        assert status.status_code == 200
        assert any(event["event_type"] == "POSITION_STATUS_CHANGE" for event in notif_repo.list_events()["events"])
    finally:
        main._notification_repository = original_notif_repo
        main._journal_repository = original_journal_repo
        if original_token is None:
            os.environ.pop("JOURNAL_ADMIN_TOKEN", None)
        else:
            os.environ["JOURNAL_ADMIN_TOKEN"] = original_token
        os.unlink(notif_path)
        os.unlink(journal_path)


if __name__ == "__main__":
    test_enter_now_transition_creates_one_event_then_dedupes()
    test_planned_entry_reached_is_direction_aware_and_not_execution_claim()
    test_setup_invalidated_before_entry_is_not_trade_loss()
    test_dedupe_survives_repository_reopen()
    test_preferences_disable_in_app_without_affecting_state_transition()
    test_notification_endpoints_require_token_and_accept_correct_token()
    test_journal_update_creates_tp_stop_and_position_transition_events()
    print("Smart notifications v1 tests passed")
