#!/usr/bin/env python3
"""Verified History pipeline tests."""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import journal_store  # noqa: E402
import main  # noqa: E402
from verified_history import (  # noqa: E402
    REPLAY_JOB_VERSION,
    backfill_preview,
    build_verified_history_snapshot,
    classify_pipeline_record,
    completion_readiness,
    replay_dedupe_key,
    replay_input_signature,
)
from verified_history_store import SQLiteVerifiedHistoryRepository  # noqa: E402


TOKEN = "journal-secret"


def completed_entry(**overrides):
    base = {
        "journal_id": "j-1",
        "position_id": "p-1",
        "ticker": "OXY",
        "direction": "SHORT",
        "result": "Win",
        "outcome": "TP1",
        "entry_timestamp": "2026-07-16T16:48:16Z",
        "exit_timestamp": "2026-07-21T00:29:30Z",
        "planned_underlying_entry": 54.28,
        "actual_underlying_entry": 54.28,
        "original_stop": 55.90,
        "original_tp1": 51.04,
        "original_tp2": 49.42,
        "original_tp3": 47.80,
        "setup_grade": "A",
    }
    base.update(overrides)
    return base


def open_entry(**overrides):
    base = completed_entry(journal_id="j-open", position_id="p-open", result="Open", outcome="", exit_timestamp="", tracking_status="active")
    base.update(overrides)
    return base


def replay(**overrides):
    base = {
        "journal_id": "j-1",
        "position_id": "p-1",
        "ticker": "OXY",
        "data_complete": True,
        "tp1_timestamp": "2026-07-21T00:00:00Z",
        "stop_timestamp": None,
        "outcome_category": "CALM_WINNER",
        "candles_evaluated": 8,
        "journal_replay_parity": {"status": "MATCH", "reason": "journal outcome reproduced"},
    }
    base.update(overrides)
    return base


def verification(status="VERIFIED", **overrides):
    base = {
        "status": status,
        "journal_result": "Win",
        "journal_outcome": "TP1",
        "replay_result": "Win",
        "replay_outcome": "TP1_REACHED",
        "replay_data_complete": True,
        "parity_detail": "MATCH",
    }
    base.update(overrides)
    return base


def make_store():
    tmp = tempfile.TemporaryDirectory()
    path = str(Path(tmp.name) / "journal.sqlite3")
    return tmp, journal_store.SQLiteJournalRepository(path), SQLiteVerifiedHistoryRepository(path)


def create_job(repo, entry):
    return repo.create_job_if_absent(
        entry["journal_id"],
        replay_input_signature(entry),
        replay_dedupe_key(entry),
        REPLAY_JOB_VERSION,
        {"source": "test"},
    )


def test_completion_readiness_and_missing_required_fields():
    ready = completion_readiness(completed_entry())
    assert ready["ready"] is True
    assert ready["pipeline_status"] == "COMPLETED_AWAITING_REPLAY"
    missing = completion_readiness(completed_entry(entry_timestamp=""))
    assert missing["ready"] is False
    assert missing["pipeline_status"] == "COMPLETION_PENDING"
    assert "entry_timestamp" in missing["missing_fields"]
    assert completion_readiness(open_entry())["pipeline_status"] == "OPEN"
    assert completion_readiness(open_entry(outcome="Open", tracking_status="active", exit_timestamp=None))["pipeline_status"] == "OPEN"


def test_replay_signature_ignores_notes_and_changes_for_material_inputs():
    entry = completed_entry(notes="first")
    same = completed_entry(notes="changed")
    changed = completed_entry(original_tp1=50.5)
    assert replay_input_signature(entry) == replay_input_signature(same)
    assert replay_input_signature(entry) != replay_input_signature(changed)


def test_queue_deduplication_and_multi_instance_claim():
    tmp, _journal, jobs = make_store()
    try:
        entry = completed_entry()
        first, created_first = create_job(jobs, entry)
        second, created_second = create_job(jobs, entry)
        assert created_first is True
        assert created_second is False
        assert first["job_id"] == second["job_id"]
        claimed = jobs.claim_next_job("worker-a")
        assert claimed["status"] == "RUNNING"
        assert jobs.claim_next_job("worker-b") is None
    finally:
        tmp.cleanup()


def test_expired_lease_is_reclaimable():
    tmp, _journal, jobs = make_store()
    try:
        entry = completed_entry()
        create_job(jobs, entry)
        old = datetime.now(timezone.utc) - timedelta(minutes=30)
        claimed = jobs.claim_next_job("worker-a", lease_seconds=1, now=old)
        assert claimed["status"] == "RUNNING"
        reclaimed = jobs.claim_next_job("worker-b", now=datetime.now(timezone.utc))
        assert reclaimed["job_id"] == claimed["job_id"]
        assert reclaimed["worker_id"] == "worker-b"
        assert reclaimed["attempt_count"] == 2
    finally:
        tmp.cleanup()


def test_retryable_and_nonretryable_failures_are_not_trade_outcomes():
    tmp, _journal, jobs = make_store()
    try:
        entry = completed_entry()
        job, _created = create_job(jobs, entry)
        claimed = jobs.claim_next_job("worker")
        failed = jobs.fail_job(claimed["job_id"], "TimeoutError", "temporary timeout", retryable=True)
        assert failed["status"] == "RETRY_PENDING"
        assert "replay" not in (failed.get("payload") or {})
        assert "verification" not in (failed.get("payload") or {})
        bad, _created = jobs.create_job_if_absent("bad", "sig", "bad|sig|1", REPLAY_JOB_VERSION, {})
        bad_claimed = jobs.claim_next_job("worker")
        failed_bad = jobs.fail_job(bad_claimed["job_id"], "invalid_timeframe", "invalid timeframe", retryable=False)
        assert failed_bad["status"] == "FAILED"
    finally:
        tmp.cleanup()


def test_pipeline_statuses_for_verified_mismatch_and_incomplete_replay():
    verified = classify_pipeline_record(completed_entry(), replay=replay(), verification=verification())
    assert verified["pipeline_status"] == "VERIFIED"
    assert verified["trade_intelligence_eligible"] is True
    mismatch = classify_pipeline_record(
        completed_entry(),
        replay=replay(tp1_timestamp=None, stop_timestamp="2026-07-17T00:00:00Z", outcome_category="STOP_DETECTED", journal_replay_parity={"status": "JOURNAL_EVENT_UNSUPPORTED"}),
        verification=verification("JOURNAL_REPLAY_MISMATCH", replay_result="Loss", replay_outcome="STOP_DETECTED", parity_detail="JOURNAL_EVENT_UNSUPPORTED"),
    )
    assert mismatch["pipeline_status"] == "NEEDS_REVIEW"
    assert mismatch["trade_intelligence_eligible"] is False
    incomplete = classify_pipeline_record(
        completed_entry(),
        replay=replay(data_complete=False, journal_replay_parity={"status": "INSUFFICIENT_REPLAY_DATA"}),
        verification=verification("INSUFFICIENT_REPLAY_DATA", replay_data_complete=False),
    )
    assert incomplete["pipeline_status"] == "REPLAY_DATA_INCOMPLETE"
    assert incomplete["trade_intelligence_eligible"] is False


def test_backfill_preview_does_not_queue_records():
    tmp, _journal, jobs = make_store()
    try:
        preview = backfill_preview([completed_entry(), completed_entry(journal_id="j-2", position_id="p-2", entry_timestamp=""), open_entry()], jobs.list_jobs())
        assert preview["counts"]["SAFE_TO_BACKFILL"] == 1
        assert preview["counts"]["MISSING_DATA"] == 1
        assert preview["counts"]["NOT_APPLICABLE"] == 1
        assert jobs.list_jobs() == []
    finally:
        tmp.cleanup()


def test_snapshot_reconciles_every_record_once():
    entries = [open_entry(), completed_entry()]
    replays = [replay()]
    analytics = [
        {"position_id": "p-open", "analytics_verification": verification("NOT_APPLICABLE")},
        {"position_id": "p-1", "analytics_verification": verification()},
    ]
    snapshot = build_verified_history_snapshot(entries, replays, analytics, [])
    assert snapshot["summary"]["journal_records_inspected"] == 2
    assert snapshot["summary"]["open"] == 1
    assert snapshot["summary"]["verified"] == 1
    assert snapshot["reconciliation"]["journal_reconciled"] is True
    assert snapshot["reconciliation"]["completed_reconciled"] is True
    assert snapshot["reconciliation"]["unclassified_record_count"] == 0


def test_protected_verified_history_endpoints_and_worker_flow():
    tmp, journal, jobs = make_store()
    previous_journal_repo = main._journal_repository
    previous_history_repo = main._verified_history_repository
    previous_replay = main._replay_positions
    previous_token = os.environ.get("JOURNAL_ADMIN_TOKEN")
    try:
        main._journal_repository = journal
        main._verified_history_repository = jobs
        os.environ["JOURNAL_ADMIN_TOKEN"] = TOKEN
        journal.create_entry(completed_entry())

        def fake_replay(entries, summary_only=True):
            return [replay(position_id=entries[0]["position_id"])]

        main._replay_positions = fake_replay
        client = TestClient(main.app)
        assert client.get("/api/dev/verified-history").status_code == 403
        assert client.get("/api/dev/verified-history", headers={"X-Kairos-Admin-Token": "wrong"}).status_code == 403
        payload = client.get("/api/dev/verified-history", headers={"X-Kairos-Admin-Token": TOKEN}).json()
        assert payload["summary"]["verified"] == 1
        queue = client.post("/api/dev/verified-history/replay/j-1", headers={"X-Kairos-Admin-Token": TOKEN}).json()
        assert queue["queued"] is True
        duplicate = client.post("/api/dev/verified-history/replay/j-1", headers={"X-Kairos-Admin-Token": TOKEN}).json()
        assert duplicate["queued"] is False
        worker = client.post("/api/dev/verified-history/worker/run?max_jobs=1", headers={"X-Kairos-Admin-Token": TOKEN}).json()
        assert worker["jobs_processed"] == 1
        after = client.get("/api/dev/verified-history", headers={"X-Kairos-Admin-Token": TOKEN}).json()
        assert after["summary"]["verified"] == 1
        assert after["summary"]["trade_intelligence_eligible"] == 1
        assert after["reconciliation"]["unclassified_record_count"] == 0
    finally:
        main._journal_repository = previous_journal_repo
        main._verified_history_repository = previous_history_repo
        main._replay_positions = previous_replay
        if previous_token is None:
            os.environ.pop("JOURNAL_ADMIN_TOKEN", None)
        else:
            os.environ["JOURNAL_ADMIN_TOKEN"] = previous_token
        tmp.cleanup()


if __name__ == "__main__":
    test_completion_readiness_and_missing_required_fields()
    test_replay_signature_ignores_notes_and_changes_for_material_inputs()
    test_queue_deduplication_and_multi_instance_claim()
    test_expired_lease_is_reclaimable()
    test_retryable_and_nonretryable_failures_are_not_trade_outcomes()
    test_pipeline_statuses_for_verified_mismatch_and_incomplete_replay()
    test_backfill_preview_does_not_queue_records()
    test_snapshot_reconciles_every_record_once()
    test_protected_verified_history_endpoints_and_worker_flow()
    print("Verified History pipeline v1 tests passed")
