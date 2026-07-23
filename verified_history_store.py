from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


class SQLiteVerifiedHistoryRepository:
    def __init__(self, db_path: str | os.PathLike[str]):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS verified_history_replay_jobs (
                    job_id TEXT PRIMARY KEY,
                    journal_id TEXT NOT NULL,
                    job_version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    dedupe_key TEXT NOT NULL UNIQUE,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    queued_at TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    next_attempt_at TEXT,
                    last_error_code TEXT,
                    last_error_message TEXT,
                    replay_signature TEXT NOT NULL,
                    worker_id TEXT,
                    lease_expires_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_verified_history_jobs_journal ON verified_history_replay_jobs(journal_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_verified_history_jobs_status ON verified_history_replay_jobs(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_verified_history_jobs_signature ON verified_history_replay_jobs(replay_signature)")

    def _row_to_job(self, row: sqlite3.Row) -> dict[str, Any]:
        payload = json.loads(row["payload"] or "{}")
        record = dict(payload)
        record.update({
            "job_id": row["job_id"],
            "journal_id": row["journal_id"],
            "job_version": row["job_version"],
            "status": row["status"],
            "dedupe_key": row["dedupe_key"],
            "attempt_count": row["attempt_count"],
            "queued_at": row["queued_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "next_attempt_at": row["next_attempt_at"],
            "last_error_code": row["last_error_code"],
            "last_error_message": row["last_error_message"],
            "replay_signature": row["replay_signature"],
            "worker_id": row["worker_id"],
            "lease_expires_at": row["lease_expires_at"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "payload": payload,
        })
        return record

    def list_jobs(self, limit: int = 1000, offset: int = 0) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM verified_history_replay_jobs ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (max(1, min(int(limit), 5000)), max(0, int(offset))),
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM verified_history_replay_jobs WHERE job_id=?", (str(job_id),)).fetchone()
        return self._row_to_job(row) if row else None

    def latest_job_for_journal(self, journal_id: str, replay_signature: str | None = None) -> dict[str, Any] | None:
        sql = "SELECT * FROM verified_history_replay_jobs WHERE journal_id=?"
        params: list[Any] = [str(journal_id)]
        if replay_signature:
            sql += " AND replay_signature=?"
            params.append(str(replay_signature))
        sql += " ORDER BY updated_at DESC LIMIT 1"
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return self._row_to_job(row) if row else None

    def create_job_if_absent(
        self,
        journal_id: str,
        replay_signature: str,
        dedupe_key: str,
        job_version: str,
        payload: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], bool]:
        now = utc_now_iso()
        job_id = f"vhjob:{uuid.uuid4()}"
        data = dict(payload or {})
        data.update({"created_by": data.get("created_by") or "verified_history_pipeline"})
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO verified_history_replay_jobs(
                        job_id, journal_id, job_version, status, dedupe_key, attempt_count,
                        queued_at, replay_signature, created_at, updated_at, payload
                    ) VALUES(?, ?, ?, 'QUEUED', ?, 0, ?, ?, ?, ?, ?)
                    """,
                    (
                        job_id,
                        str(journal_id),
                        str(job_version),
                        str(dedupe_key),
                        now,
                        str(replay_signature),
                        now,
                        now,
                        json.dumps(data, sort_keys=True),
                    ),
                )
            return self.get_job(job_id), True
        except sqlite3.IntegrityError:
            existing = self.latest_job_for_journal(journal_id, replay_signature)
            if existing:
                return existing, False
            with self._connect() as conn:
                row = conn.execute("SELECT * FROM verified_history_replay_jobs WHERE dedupe_key=?", (str(dedupe_key),)).fetchone()
            return self._row_to_job(row), False

    def claim_next_job(self, worker_id: str, lease_seconds: int = 900, now: datetime | None = None) -> dict[str, Any] | None:
        current = now or datetime.now(timezone.utc)
        now_iso = current.isoformat().replace("+00:00", "Z")
        lease_iso = (current + timedelta(seconds=lease_seconds)).isoformat().replace("+00:00", "Z")
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM verified_history_replay_jobs
                WHERE
                    status = 'QUEUED'
                    OR (status = 'RETRY_PENDING' AND (next_attempt_at IS NULL OR next_attempt_at <= ?))
                    OR (status = 'RUNNING' AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?)
                ORDER BY queued_at ASC, updated_at ASC
                LIMIT 1
                """,
                (now_iso, now_iso),
            ).fetchone()
            if not row:
                return None
            updated = conn.execute(
                """
                UPDATE verified_history_replay_jobs
                SET status='RUNNING',
                    worker_id=?,
                    started_at=COALESCE(started_at, ?),
                    lease_expires_at=?,
                    attempt_count=attempt_count + 1,
                    updated_at=?
                WHERE job_id=? AND (
                    status = 'QUEUED'
                    OR (status = 'RETRY_PENDING' AND (next_attempt_at IS NULL OR next_attempt_at <= ?))
                    OR (status = 'RUNNING' AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?)
                )
                """,
                (worker_id, now_iso, lease_iso, now_iso, row["job_id"], now_iso, now_iso),
            )
            if int(updated.rowcount or 0) != 1:
                return None
        return self.get_job(row["job_id"])

    def complete_job(self, job_id: str, replay: dict[str, Any], verification: dict[str, Any], pipeline_status: str) -> dict[str, Any] | None:
        existing = self.get_job(job_id)
        if not existing:
            return None
        payload = dict(existing.get("payload") or {})
        payload.update({
            "replay": replay,
            "verification": verification,
            "pipeline_status": pipeline_status,
        })
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE verified_history_replay_jobs
                SET status='COMPLETED',
                    completed_at=?,
                    lease_expires_at=NULL,
                    worker_id=NULL,
                    last_error_code=NULL,
                    last_error_message=NULL,
                    updated_at=?,
                    payload=?
                WHERE job_id=?
                """,
                (now, now, json.dumps(payload, sort_keys=True), str(job_id)),
            )
        return self.get_job(job_id)

    def fail_job(
        self,
        job_id: str,
        error_code: str,
        error_message: str,
        retryable: bool,
        max_attempts: int = 3,
        backoff_seconds: int = 300,
    ) -> dict[str, Any] | None:
        existing = self.get_job(job_id)
        if not existing:
            return None
        now = datetime.now(timezone.utc)
        attempts = int(existing.get("attempt_count") or 0)
        retry = retryable and attempts < max_attempts
        status = "RETRY_PENDING" if retry else "FAILED"
        next_attempt = (now + timedelta(seconds=backoff_seconds * max(1, attempts))).isoformat().replace("+00:00", "Z") if retry else None
        payload = dict(existing.get("payload") or {})
        payload["last_failure"] = {
            "error_code": error_code,
            "error_message": error_message,
            "retryable": retryable,
            "recorded_at": now.isoformat().replace("+00:00", "Z"),
        }
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE verified_history_replay_jobs
                SET status=?,
                    next_attempt_at=?,
                    lease_expires_at=NULL,
                    worker_id=NULL,
                    last_error_code=?,
                    last_error_message=?,
                    updated_at=?,
                    payload=?
                WHERE job_id=?
                """,
                (
                    status,
                    next_attempt,
                    str(error_code),
                    str(error_message)[:500],
                    now.isoformat().replace("+00:00", "Z"),
                    json.dumps(payload, sort_keys=True),
                    str(job_id),
                ),
            )
        return self.get_job(job_id)
