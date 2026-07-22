#!/usr/bin/env python3
"""Server-backed journal persistence tests."""

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


TOKEN = "journal-secret"


def make_repo():
    tmp = tempfile.TemporaryDirectory()
    db_path = str(Path(tmp.name) / "journal.sqlite3")
    return tmp, journal_store.SQLiteJournalRepository(db_path)


def client_with_repo(repo):
    previous_repo = main._journal_repository
    previous_journal_token = os.environ.get("JOURNAL_ADMIN_TOKEN")
    previous_discovery_token = os.environ.get("DISCOVERY_ADMIN_TOKEN")
    main._journal_repository = repo
    os.environ["JOURNAL_ADMIN_TOKEN"] = TOKEN
    client = TestClient(main.app)

    def cleanup():
        main._journal_repository = previous_repo
        if previous_journal_token is None:
            os.environ.pop("JOURNAL_ADMIN_TOKEN", None)
        else:
            os.environ["JOURNAL_ADMIN_TOKEN"] = previous_journal_token
        if previous_discovery_token is None:
            os.environ.pop("DISCOVERY_ADMIN_TOKEN", None)
        else:
            os.environ["DISCOVERY_ADMIN_TOKEN"] = previous_discovery_token

    return client, cleanup


def headers():
    return {"X-Kairos-Admin-Token": TOKEN}


def entry(**overrides):
    base = {
        "journal_id": "j-1",
        "position_id": "p-1",
        "ticker": "OXY",
        "direction": "LONG",
        "result": "Open",
        "setup_grade": "A",
        "entry_timestamp": "2026-07-20T14:00:00Z",
        "planned_underlying_entry": 54.5,
        "actual_underlying_entry": 54.6,
        "original_stop": 52.0,
        "original_tp1": 59.5,
        "target_price": 59.5,
        "actual_option_premium": 1.2,
        "actual_strike": 55,
        "actual_expiration": "2026-08-21",
        "actual_quantity": 1,
        "position_last_state": "HEALTHY",
        "position_best_price": 55.5,
        "position_max_progress_percent": 20,
        "position_tp1_reached": False,
        "position_state_history": [],
        "custom_legacy_field": "preserved",
    }
    base.update(overrides)
    return base


def test_create_read_list_and_unknown_field_preservation():
    tmp, repo = make_repo()
    try:
        created = repo.create_entry(entry())
        assert created["journal_id"] == "j-1"
        assert created["position_id"] == "p-1"
        assert created["custom_legacy_field"] == "preserved"
        assert created["record_version"] == 1
        assert repo.get_entry("j-1")["ticker"] == "OXY"
        assert len(repo.list_entries({"status": "open"})) == 1
        assert len(repo.list_entries({"ticker": "OXY", "direction": "LONG"})) == 1
    finally:
        tmp.cleanup()


def test_partial_update_preserves_original_plan_without_explicit_correction():
    tmp, repo = make_repo()
    try:
        created = repo.create_entry(entry())
        updated = repo.update_entry("j-1", {
            "record_version": created["record_version"],
            "position_last_state": "WATCH",
            "planned_underlying_entry": 99,
            "actual_underlying_entry": 88,
            "notes": "manual note",
        })
        assert updated["position_last_state"] == "WATCH"
        assert updated["planned_underlying_entry"] == 54.5
        assert updated["actual_underlying_entry"] == 54.6
        assert updated["notes"] == "manual note"
        corrected = repo.update_entry("j-1", {
            "record_version": updated["record_version"],
            "allow_plan_correction": True,
            "planned_underlying_entry": 54.7,
        })
        assert corrected["planned_underlying_entry"] == 54.7
    finally:
        tmp.cleanup()


def test_record_version_conflict_and_position_patch_isolation():
    tmp, repo = make_repo()
    try:
        created = repo.create_entry(entry())
        repo.update_entry("j-1", {"record_version": created["record_version"], "notes": "newer manual edit"})
        try:
            repo.update_entry("j-1", {"record_version": created["record_version"], "position_last_state": "WATCH"})
            raise AssertionError("Expected record-version conflict")
        except journal_store.JournalConflictError:
            pass
        latest = repo.get_entry("j-1")
        patched = repo.update_entry("j-1", {
            "record_version": latest["record_version"],
            "position_last_state": "WATCH",
            "position_best_price": 56.2,
        })
        assert patched["notes"] == "newer manual edit"
        assert patched["position_last_state"] == "WATCH"
        assert patched["position_best_price"] == 56.2
    finally:
        tmp.cleanup()


def test_history_append_and_milestone_deduplication():
    tmp, repo = make_repo()
    try:
        created = repo.create_entry(entry())
        event = {
            "event_id": "p-1:STATE:HEALTHY:WATCH",
            "timestamp": "2026-07-21T14:00:00Z",
            "previous_state": "HEALTHY",
            "new_state": "WATCH",
            "current_price": 53.8,
            "progress_to_tp1": 0,
            "current_r": -0.32,
            "reason_code": "POSITION_BELOW_ENTRY",
            "reason_text": "Current R is negative.",
        }
        milestone = {"event_id": "p-1:PROGRESS_25", "event_type": "PROGRESS_25", "timestamp": "2026-07-21T15:00:00Z"}
        updated = repo.update_entry("j-1", {
            "record_version": created["record_version"],
            "position_state_history": [event, milestone, milestone],
        })
        assert len(updated["position_state_history"]) == 2
        updated = repo.update_entry("j-1", {
            "record_version": updated["record_version"],
            "position_state_history": [event, {"event_id": "p-1:PROGRESS_50", "event_type": "PROGRESS_50"}],
        })
        assert len(updated["position_state_history"]) == 3
    finally:
        tmp.cleanup()


def test_soft_delete_export_and_diagnostics():
    tmp, repo = make_repo()
    try:
        repo.create_entry(entry())
        assert repo.diagnostics()["total_entries"] == 1
        diag = repo.diagnostics()
        assert diag["configured_db_path"].endswith("journal.sqlite3")
        assert diag["resolved_db_path"]
        assert diag["storage_directory_exists"] is True
        assert diag["storage_directory_writable"] is True
        assert diag["database_exists"] is True
        assert diag["database_size_bytes"] > 0
        assert diag["sqlite_wal_enabled"] is True
        assert diag["journal_entry_count"] == 1
        assert diag["open_entry_count"] == 1
        exported = repo.export_entries()
        assert exported["journal_schema_version"] == 1
        assert exported["entries"][0]["journal_id"] == "j-1"
        deleted = repo.delete_entry("j-1")
        assert deleted["deleted_at"]
        assert repo.get_entry("j-1") is None
        assert repo.list_entries({"status": "all"}) == []
    finally:
        tmp.cleanup()


def test_safe_sqlite_backup_retention_and_isolated_restore():
    tmp, repo = make_repo()
    try:
        repo.create_entry(entry(position_state_history=[
            {"event_id": "evt-1", "new_state": "WATCH"},
            {"event_id": "evt-2", "event_type": "TP1_REACHED"},
        ]))
        backup_dir = Path(tmp.name) / "journal_backups"
        first = repo.create_backup(backup_dir=backup_dir, keep_latest=2)
        assert first["filename"].startswith("kairos_journal_")
        assert first["size"] > 0
        assert len(first["sha256"]) == 64
        validation = repo.restore_validation(first["path"])
        assert validation["schema_version"] == 1
        assert validation["record_count"] == 1
        assert validation["journal_ids"] == ["j-1"]
        assert validation["position_ids"] == ["p-1"]
        assert validation["position_history_events"] == 2
        repo.create_backup(backup_dir=backup_dir, keep_latest=2)
        repo.create_backup(backup_dir=backup_dir, keep_latest=2)
        assert len(list(backup_dir.glob("kairos_journal_*.sqlite3"))) <= 2
    finally:
        tmp.cleanup()


def test_invalid_payload_rejection_and_missing_optional_fields():
    tmp, repo = make_repo()
    try:
        try:
            repo.create_entry(entry(direction="SIDEWAYS"))
            raise AssertionError("Expected invalid direction")
        except journal_store.JournalValidationError:
            pass
        try:
            repo.create_entry(entry(actual_underlying_entry="abc"))
            raise AssertionError("Expected invalid number")
        except journal_store.JournalValidationError:
            pass
        minimal = repo.create_entry({"ticker": "AAPL", "direction": "LONG"})
        assert minimal["journal_id"]
        assert minimal["position_id"]
    finally:
        tmp.cleanup()


def test_server_restart_and_multiple_device_simulation():
    tmp = tempfile.TemporaryDirectory()
    try:
        path = str(Path(tmp.name) / "journal.sqlite3")
        repo1 = journal_store.SQLiteJournalRepository(path)
        repo1.create_entry(entry(journal_id="j-1", position_id="p-1"))
        repo2 = journal_store.SQLiteJournalRepository(path)
        assert repo2.get_entry("j-1")["position_id"] == "p-1"
        client1, cleanup1 = client_with_repo(repo2)
        client2, cleanup2 = client_with_repo(repo2)
        try:
            assert client1.get("/api/journal/j-1", headers=headers()).json()["ticker"] == "OXY"
            assert client2.patch("/api/journal/j-1", headers=headers(), json={"record_version": 1, "notes": "device two"}).status_code == 200
            stale = client1.patch("/api/journal/j-1", headers=headers(), json={"record_version": 1, "notes": "stale"})
            assert stale.status_code == 409
        finally:
            cleanup1()
            cleanup2()
    finally:
        tmp.cleanup()


def test_api_crud_migrate_filters_and_auth():
    tmp, repo = make_repo()
    client, cleanup = client_with_repo(repo)
    try:
        assert client.get("/api/journal").status_code == 403
        created = client.post("/api/journal", headers=headers(), json=entry()).json()
        assert created["journal_id"] == "j-1"
        assert client.get("/api/journal?status=open", headers=headers()).json()["entries"][0]["journal_id"] == "j-1"
        assert client.get("/api/journal?status=closed", headers=headers()).json()["entries"] == []
        assert client.get("/api/journal/export", headers=headers()).json()["entries"][0]["journal_id"] == "j-1"
        diag = client.get("/api/journal/diagnostics", headers=headers()).json()
        assert diag["storage_backend"] == "sqlite"
        assert diag["durable_storage_confirmed"] is False
        backup = client.post("/api/journal/backup", headers=headers()).json()
        assert backup["size"] > 0
        restored = client.post("/api/journal/backup/validate", headers=headers(), json={"path": backup["path"]}).json()
        assert restored["record_count"] == 1
        migrated = client.post("/api/journal/migrate", headers=headers(), json={
            "entries": [
                entry(journal_id="j-1", notes="matching update", position_state_history=[{"event_id": "merge-event"}]),
                entry(journal_id="j-2", position_id="p-2", ticker="OXY", direction="SHORT", actual_underlying_entry=60, original_stop=63, original_tp1=55),
            ]
        }).json()
        assert migrated["created"] == 1
        assert migrated["updated"] == 0
        assert migrated["conflicts"] == 1
        assert len(client.get("/api/journal?ticker=OXY", headers=headers()).json()["entries"]) == 2
        current = client.get("/api/journal/j-1", headers=headers()).json()
        newer = client.post("/api/journal/migrate", headers=headers(), json={
            "entries": [entry(journal_id="j-1", updated_at="2099-01-01T00:00:00Z", notes="newer trusted update")]
        }).json()
        assert newer["updated"] == 1
        assert newer["conflicts"] == 0
        assert client.get("/api/journal/j-1", headers=headers()).json()["notes"] == "newer trusted update"
        assert client.delete("/api/journal/j-2", headers=headers()).json()["deleted"] is True
    finally:
        cleanup()
        tmp.cleanup()


def test_multiple_trades_same_ticker_long_short_closed_reopened():
    tmp, repo = make_repo()
    try:
        repo.create_entry(entry(journal_id="oxy-long-1", position_id="pos-long-1", direction="LONG", result="Win", tracking_status="completed", completion_reason="target"))
        repo.create_entry(entry(journal_id="oxy-short-1", position_id="pos-short-1", direction="SHORT", actual_underlying_entry=60, original_stop=63, original_tp1=55))
        repo.create_entry(entry(journal_id="oxy-long-2", position_id="pos-long-2", direction="LONG", actual_underlying_entry=58, original_stop=55, original_tp1=64))
        entries = repo.list_entries({"ticker": "OXY", "status": "all"})
        assert {item["position_id"] for item in entries} == {"pos-long-1", "pos-short-1", "pos-long-2"}
        assert len(repo.list_entries({"status": "open"})) == 2
        assert len(repo.list_entries({"status": "closed"})) == 1
    finally:
        tmp.cleanup()


def test_no_scanner_discovery_option_or_alert_side_effects():
    tmp, repo = make_repo()
    previous_scan_cached = main.scan_cached
    previous_submit_discovery = main._submit_discovery_universe_job
    try:
        def fail_scan(*args, **kwargs):
            raise AssertionError("journal API must not trigger scans")

        main.scan_cached = fail_scan
        main._submit_discovery_universe_job = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("journal API must not trigger discovery"))
        client, cleanup = client_with_repo(repo)
        try:
            assert client.post("/api/journal", headers=headers(), json=entry()).status_code == 200
            assert client.patch("/api/journal/j-1", headers=headers(), json={"record_version": 1, "position_last_state": "WATCH"}).status_code == 200
        finally:
            cleanup()
    finally:
        main.scan_cached = previous_scan_cached
        main._submit_discovery_universe_job = previous_submit_discovery
        tmp.cleanup()


if __name__ == "__main__":
    test_create_read_list_and_unknown_field_preservation()
    test_partial_update_preserves_original_plan_without_explicit_correction()
    test_record_version_conflict_and_position_patch_isolation()
    test_history_append_and_milestone_deduplication()
    test_soft_delete_export_and_diagnostics()
    test_safe_sqlite_backup_retention_and_isolated_restore()
    test_invalid_payload_rejection_and_missing_optional_fields()
    test_server_restart_and_multiple_device_simulation()
    test_api_crud_migrate_filters_and_auth()
    test_multiple_trades_same_ticker_long_short_closed_reopened()
    test_no_scanner_discovery_option_or_alert_side_effects()
    print("Journal persistence v1 tests passed")
