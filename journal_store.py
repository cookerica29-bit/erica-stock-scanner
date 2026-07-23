from __future__ import annotations

import json
import os
import sqlite3
import uuid
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


JOURNAL_SCHEMA_VERSION = 1
IMMUTABLE_PLAN_FIELDS = {
    "direction",
    "planned_underlying_entry",
    "actual_underlying_entry",
    "original_stop",
    "original_tp1",
    "original_tp2",
    "original_tp3",
    "entry_timestamp",
    "option_plan",
}
NUMERIC_FIELDS = {
    "planned_underlying_entry",
    "actual_underlying_entry",
    "original_stop",
    "original_tp1",
    "original_tp2",
    "original_tp3",
    "actual_option_premium",
    "actual_strike",
    "actual_quantity",
    "position_best_price",
    "position_max_progress_percent",
}
VALID_DIRECTIONS = {"LONG", "SHORT", "CALL", "PUT", "N/A", ""}
CANONICAL_REPLAY_FIELDS = {
    "journal_id",
    "position_id",
    "ticker",
    "direction",
    "setup_grade",
    "grade",
    "scanner_timeframe",
    "timeframe",
    "planned_underlying_entry",
    "actual_underlying_entry",
    "original_stop",
    "original_tp1",
    "original_tp2",
    "original_tp3",
    "entry_timestamp",
    "exit_timestamp",
    "result",
    "outcome",
    "completion_reason",
    "actual_option_type",
    "actual_option_premium",
    "actual_strike",
    "actual_expiration",
    "actual_quantity",
}
REPLAY_RELEVANT_FIELDS = {
    *CANONICAL_REPLAY_FIELDS,
    "entry_price",
    "entry",
    "stop_price",
    "plannedStop",
    "target_price",
    "plannedTp1",
    "plannedTp2",
    "plannedTp3",
    "tracking_started_at",
    "tracking_completed_at",
    "signal_timestamp",
    "setupGrade",
    "setupTf",
    "option_type",
    "optionType",
    "premium_paid",
    "strike_price",
    "strike",
    "expiration_date",
    "expiry",
    "contracts",
}


class JournalConflictError(Exception):
    pass


class JournalValidationError(Exception):
    pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_timestamp(value: Any, field: str) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    text = str(value)
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise JournalValidationError(f"Invalid timestamp for {field}") from exc
    return text


def normalize_number(value: Any, field: str) -> float | int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        raise JournalValidationError(f"Invalid numeric value for {field}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise JournalValidationError(f"Invalid numeric value for {field}") from exc
    return int(number) if number.is_integer() else number


def normalize_direction(value: Any) -> str:
    direction = str(value or "").upper()
    if direction not in VALID_DIRECTIONS:
        raise JournalValidationError("Invalid direction")
    return direction


def stable_id(value: Any = None) -> str:
    return str(value or uuid.uuid4())


def entry_status(entry: dict[str, Any]) -> str:
    result = str(entry.get("result") or entry.get("status") or "Open")
    tracking = str(entry.get("tracking_status") or "")
    completion = entry.get("completion_reason")
    if result.lower() == "open" and tracking != "completed" and not completion:
        return "open"
    return "closed"


def core_value(entry: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = entry.get(key)
        if value not in (None, ""):
            return value
    return None


def merge_history(existing: list[Any], incoming: list[Any]) -> list[Any]:
    output: list[Any] = []
    seen: set[str] = set()
    for raw in [*(existing or []), *(incoming or [])]:
        if not isinstance(raw, dict):
            continue
        event = dict(raw)
        event_id = str(event.get("event_id") or event.get("event_key") or "")
        if not event_id:
            event_id = str(uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(event, sort_keys=True, default=str)))
            event["event_id"] = event_id
        if event_id in seen:
            continue
        seen.add(event_id)
        output.append(event)
    return output


def merge_payload(existing: dict[str, Any], patch: dict[str, Any], allow_plan_correction: bool = False) -> dict[str, Any]:
    merged = dict(existing or {})
    for key, value in (patch or {}).items():
        if key in {"journal_id", "record_version", "created_at", "updated_at", "deleted_at"}:
            continue
        if key in IMMUTABLE_PLAN_FIELDS and key in merged and merged.get(key) not in (None, "") and not allow_plan_correction:
            continue
        if key == "position_state_history":
            merged[key] = merge_history(merged.get(key) or [], value or [])
        elif value not in (None, "") or merged.get(key) in (None, ""):
            merged[key] = value
    return merged


def parsed_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def validate_entry_payload(entry: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise JournalValidationError("Journal entry must be an object")
    payload = dict(entry)
    direction = core_value(payload, "direction", "option_type", "optionType")
    if direction is not None:
        payload["direction"] = normalize_direction(direction)
    for field in NUMERIC_FIELDS:
        if field in payload:
            payload[field] = normalize_number(payload[field], field)
    for field in ["entry_timestamp", "exit_timestamp", "created_at", "updated_at", "snapshot_timestamp", "tracking_started_at", "tracking_completed_at"]:
        if field in payload:
            payload[field] = normalize_timestamp(payload[field], field)
    if "position_state_history" in payload:
        payload["position_state_history"] = merge_history([], payload.get("position_state_history") or [])
    canonicalize_replay_fields(payload)
    return payload


def canonicalize_replay_fields(payload: dict[str, Any]) -> dict[str, Any]:
    ticker = core_value(payload, "ticker")
    if ticker not in (None, ""):
        payload["ticker"] = str(ticker).upper()
    direction = core_value(payload, "direction", "option_type", "optionType")
    if direction is not None:
        payload["direction"] = normalize_direction(direction)
    grade = core_value(payload, "setup_grade", "setupGrade", "grade")
    if grade not in (None, ""):
        payload["setup_grade"] = grade
    timeframe = core_value(payload, "scanner_timeframe", "timeframe", "setupTf")
    if timeframe not in (None, ""):
        payload["scanner_timeframe"] = str(timeframe).upper()
    entry_ts = core_value(payload, "entry_timestamp", "actual_entry_at", "position_opened_at", "tracking_started_at", "signal_timestamp", "created_at", "createdAt")
    if entry_ts not in (None, ""):
        payload["entry_timestamp"] = normalize_timestamp(entry_ts, "entry_timestamp")
    exit_ts = core_value(payload, "exit_timestamp", "tracking_completed_at")
    if exit_ts not in (None, ""):
        payload["exit_timestamp"] = normalize_timestamp(exit_ts, "exit_timestamp")
    number_aliases = {
        "planned_underlying_entry": ("planned_underlying_entry", "entry_price", "entry"),
        "actual_underlying_entry": ("actual_underlying_entry", "underlying_price_at_entry"),
        "original_stop": ("original_stop", "stop_price", "plannedStop"),
        "original_tp1": ("original_tp1", "target_price", "plannedTp1", "tp1"),
        "original_tp2": ("original_tp2", "plannedTp2", "tp2"),
        "original_tp3": ("original_tp3", "plannedTp3", "tp3"),
        "actual_option_premium": ("actual_option_premium", "actual_premium", "premium_paid", "askAtSelection"),
        "actual_strike": ("actual_strike", "strike_price", "strike"),
        "actual_quantity": ("actual_quantity", "contracts"),
    }
    for target, aliases in number_aliases.items():
        value = core_value(payload, *aliases)
        if value not in (None, ""):
            payload[target] = normalize_number(value, target)
    expiration = core_value(payload, "actual_expiration", "expiration_date", "expiry")
    if expiration not in (None, ""):
        payload["actual_expiration"] = str(expiration)
    option_type = core_value(payload, "actual_option_type", "option_type", "optionType")
    if option_type not in (None, ""):
        payload["actual_option_type"] = str(option_type).upper()
    return payload


def replay_relevant_patch(patch: dict[str, Any]) -> bool:
    return any(key in REPLAY_RELEVANT_FIELDS for key in (patch or {}))


def mark_replay_stale(payload: dict[str, Any], reason: str) -> dict[str, Any]:
    payload["replay_cache_status"] = "stale"
    payload["replay_cache_stale_at"] = utc_now_iso()
    payload["replay_cache_stale_reason"] = reason
    return payload


class JournalRepository:
    def list_entries(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError

    def get_entry(self, journal_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def create_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def update_entry(self, journal_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def delete_entry(self, journal_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def upsert_entries(self, entries: list[dict[str, Any]]) -> dict[str, Any]:
        raise NotImplementedError


class SQLiteJournalRepository(JournalRepository):
    def __init__(self, db_path: str | os.PathLike[str]):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS journal_schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS journal_entries (
                    journal_id TEXT PRIMARY KEY,
                    position_id TEXT NOT NULL,
                    ticker TEXT,
                    direction TEXT,
                    status TEXT,
                    grade TEXT,
                    entry_timestamp TEXT,
                    exit_timestamp TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    record_version INTEGER NOT NULL DEFAULT 1,
                    deleted_at TEXT,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_journal_position_id ON journal_entries(position_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_journal_ticker ON journal_entries(ticker)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_journal_status ON journal_entries(status)")
            conn.execute(
                "INSERT OR IGNORE INTO journal_schema_migrations(version, applied_at) VALUES(?, ?)",
                (JOURNAL_SCHEMA_VERSION, utc_now_iso()),
            )

    def _row_to_entry(self, row: sqlite3.Row) -> dict[str, Any]:
        payload = json.loads(row["payload"] or "{}")
        payload.update({
            "journal_id": row["journal_id"],
            "position_id": row["position_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "record_version": row["record_version"],
            "deleted_at": row["deleted_at"],
            "journal_schema_version": JOURNAL_SCHEMA_VERSION,
        })
        return payload

    def _core_tuple(self, payload: dict[str, Any], now: str, existing: dict[str, Any] | None = None):
        journal_id = stable_id(payload.get("journal_id") or payload.get("id") or (existing or {}).get("journal_id"))
        position_id = stable_id(payload.get("position_id") or (existing or {}).get("position_id") or journal_id)
        payload["journal_id"] = journal_id
        payload["position_id"] = position_id
        payload["journal_schema_version"] = JOURNAL_SCHEMA_VERSION
        ticker = str(core_value(payload, "ticker") or "").upper()
        direction = str(core_value(payload, "direction") or "").upper()
        status = entry_status(payload)
        grade = core_value(payload, "setup_grade", "setupGrade", "grade")
        entry_ts = core_value(payload, "entry_timestamp", "actual_entry_at", "position_opened_at", "tracking_started_at", "signal_timestamp")
        exit_ts = core_value(payload, "exit_timestamp", "tracking_completed_at")
        created_at = core_value(payload, "created_at", "createdAt") or now
        payload["created_at"] = created_at
        payload["updated_at"] = now
        return journal_id, position_id, ticker, direction, status, grade, entry_ts, exit_ts, created_at, now, json.dumps(payload, sort_keys=True)

    def list_entries(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        filters = filters or {}
        where = ["deleted_at IS NULL"]
        params: list[Any] = []
        status = str(filters.get("status") or "all").lower()
        if status in {"open", "closed"}:
            where.append("status = ?")
            params.append(status)
        if filters.get("ticker"):
            where.append("ticker = ?")
            params.append(str(filters["ticker"]).upper())
        if filters.get("direction"):
            where.append("direction = ?")
            params.append(str(filters["direction"]).upper())
        if filters.get("position_id"):
            where.append("position_id = ?")
            params.append(str(filters["position_id"]))
        limit = max(1, min(int(filters.get("limit") or 500), 1000))
        offset = max(0, int(filters.get("offset") or 0))
        sql = f"SELECT * FROM journal_entries WHERE {' AND '.join(where)} ORDER BY updated_at DESC LIMIT ? OFFSET ?"
        with self._connect() as conn:
            rows = conn.execute(sql, [*params, limit, offset]).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def get_entry(self, journal_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM journal_entries WHERE journal_id = ? AND deleted_at IS NULL", (str(journal_id),)).fetchone()
        return self._row_to_entry(row) if row else None

    def create_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        payload = validate_entry_payload(entry)
        mark_replay_stale(payload, "created")
        now = utc_now_iso()
        values = self._core_tuple(payload, now)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO journal_entries(
                    journal_id, position_id, ticker, direction, status, grade, entry_timestamp, exit_timestamp,
                    created_at, updated_at, payload
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
        return self.get_entry(values[0])

    def update_entry(self, journal_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        allow_plan_correction = bool((patch or {}).pop("allow_plan_correction", False))
        expected_version = (patch or {}).pop("record_version", None)
        should_mark_replay_stale = replay_relevant_patch(patch or {})
        existing = self.get_entry(journal_id)
        if not existing:
            raise KeyError(journal_id)
        if expected_version is not None and int(expected_version) != int(existing.get("record_version") or 0):
            raise JournalConflictError("record_version conflict")
        clean_patch = validate_entry_payload(patch or {})
        merged = merge_payload(existing, clean_patch, allow_plan_correction=allow_plan_correction)
        if should_mark_replay_stale:
            mark_replay_stale(merged, "journal_replay_field_updated")
        now = utc_now_iso()
        values = self._core_tuple(merged, now, existing=existing)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE journal_entries
                SET position_id=?, ticker=?, direction=?, status=?, grade=?, entry_timestamp=?, exit_timestamp=?,
                    updated_at=?, record_version=record_version + 1, payload=?
                WHERE journal_id=?
                """,
                (values[1], values[2], values[3], values[4], values[5], values[6], values[7], now, values[10], str(journal_id)),
            )
        return self.get_entry(journal_id)

    def delete_entry(self, journal_id: str) -> dict[str, Any] | None:
        existing = self.get_entry(journal_id)
        if not existing:
            return None
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                "UPDATE journal_entries SET deleted_at=?, updated_at=?, record_version=record_version + 1 WHERE journal_id=?",
                (now, now, str(journal_id)),
            )
        existing["deleted_at"] = now
        return existing

    def upsert_entries(self, entries: list[dict[str, Any]]) -> dict[str, Any]:
        created = updated = conflicts = 0
        conflict_entries = []
        results = []
        for raw in entries or []:
            payload = validate_entry_payload(raw)
            journal_id = str(payload.get("journal_id") or payload.get("id") or uuid.uuid4())
            payload["journal_id"] = journal_id
            existing = self.get_entry(journal_id)
            if not existing:
                results.append(self.create_entry(payload))
                created += 1
                continue
            incoming_updated = parsed_timestamp(payload.get("updated_at") or payload.get("updatedAt"))
            existing_updated = parsed_timestamp(existing.get("updated_at") or existing.get("updatedAt"))
            if not incoming_updated or not existing_updated or incoming_updated <= existing_updated:
                conflicts += 1
                conflict_entries.append(journal_id)
                results.append(existing)
                continue
            try:
                merged_patch = merge_payload(existing, payload, allow_plan_correction=False)
                merged_patch["record_version"] = existing.get("record_version")
                results.append(self.update_entry(journal_id, merged_patch))
                updated += 1
            except JournalConflictError:
                conflicts += 1
                conflict_entries.append(journal_id)
        return {"created": created, "updated": updated, "conflicts": conflicts, "conflict_ids": conflict_entries, "entries": results}

    def export_entries(self) -> dict[str, Any]:
        return {
            "journal_schema_version": JOURNAL_SCHEMA_VERSION,
            "exported_at": utc_now_iso(),
            "entries": self.list_entries({"status": "all", "limit": 1000}),
        }

    def create_backup(self, backup_dir: str | os.PathLike[str] | None = None, keep_latest: int = 10) -> dict[str, Any]:
        source = Path(self.db_path)
        directory = self._backup_directory(backup_dir)
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        destination = directory / f"kairos_journal_{timestamp}.sqlite3"
        with self._connect() as src, sqlite3.connect(str(destination)) as dst:
            src.backup(dst)
            dst.execute("PRAGMA wal_checkpoint(FULL)")
        data = destination.read_bytes()
        backups = sorted(directory.glob("kairos_journal_*.sqlite3"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in backups[max(1, int(keep_latest)):]:
            try:
                old.unlink()
            except OSError:
                pass
        return {
            "filename": destination.name,
            "path": str(destination),
            "timestamp": timestamp,
            "size_bytes": len(data),
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "retention_keep_latest": keep_latest,
        }

    def restore_validation(self, backup_path: str | os.PathLike[str]) -> dict[str, Any]:
        repo = SQLiteJournalRepository(str(backup_path))
        entries = repo.list_entries({"status": "all", "limit": 1000})
        return {
            "schema_version": JOURNAL_SCHEMA_VERSION,
            "record_count": len(entries),
            "journal_ids": [entry.get("journal_id") for entry in entries],
            "position_ids": [entry.get("position_id") for entry in entries],
            "position_history_events": sum(len(entry.get("position_state_history") or []) for entry in entries),
        }

    def _sqlite_wal_enabled(self) -> bool:
        try:
            with self._connect() as conn:
                row = conn.execute("PRAGMA journal_mode").fetchone()
            return bool(row and str(row[0]).lower() == "wal")
        except sqlite3.Error:
            return False

    def _storage_directory_writable(self, directory: Path) -> bool:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            probe = directory / f".kairos_write_probe_{uuid.uuid4().hex}"
            with open(probe, "w", encoding="utf-8") as handle:
                handle.write("ok")
                handle.flush()
                os.fsync(handle.fileno())
            renamed = directory / f"{probe.name}.renamed"
            probe.rename(renamed)
            renamed.unlink()
            return True
        except OSError:
            return False

    def _backup_directory(self, backup_dir: str | os.PathLike[str] | None = None) -> Path:
        source = Path(self.db_path)
        return Path(backup_dir or os.getenv("JOURNAL_BACKUP_DIR") or source.parent / "journal_backups")

    def _latest_backup_metadata(self) -> dict[str, Any] | None:
        directory = self._backup_directory()
        try:
            backups = sorted(directory.glob("kairos_journal_*.sqlite3"), key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            return None
        if not backups:
            return None
        latest = backups[0]
        try:
            stat = latest.stat()
        except OSError:
            return None
        return {
            "filename": latest.name,
            "path": str(latest),
            "size_bytes": stat.st_size,
        }

    def diagnostics(self) -> dict[str, Any]:
        entries = self.list_entries({"status": "all", "limit": 1000})
        ids = [entry.get("journal_id") for entry in entries]
        position_ids = [entry.get("position_id") for entry in entries]
        duplicate_ids = sorted({value for value in ids if ids.count(value) > 1 and value})
        path = Path(self.db_path)
        resolved = path.expanduser().resolve()
        directory = resolved.parent
        durable_expected = bool(os.getenv("DATABASE_URL")) or bool(os.getenv("JOURNAL_DB_PATH") or os.getenv("KAIROS_JOURNAL_DB_PATH"))
        durable_confirmed = bool(os.getenv("DATABASE_URL")) or any(
            str(resolved).startswith(str(Path(prefix).resolve()) + os.sep) and os.path.ismount(prefix)
            for prefix in ["/data", "/mnt", "/var/lib"]
            if Path(prefix).exists()
        )
        return {
            "storage_backend": "sqlite",
            "storage_location": self.db_path,
            "configured_db_path": self.db_path,
            "resolved_db_path": str(resolved),
            "storage_directory_exists": directory.exists(),
            "storage_directory_writable": self._storage_directory_writable(directory),
            "database_exists": resolved.exists(),
            "database_size_bytes": resolved.stat().st_size if resolved.exists() else 0,
            "sqlite_wal_enabled": self._sqlite_wal_enabled(),
            "durable_mount_expected": durable_expected,
            "durable_storage_confirmed": durable_confirmed,
            "schema_version": JOURNAL_SCHEMA_VERSION,
            "total_entries": len(entries),
            "journal_entry_count": len(entries),
            "open_entries": sum(1 for entry in entries if entry_status(entry) == "open"),
            "open_entry_count": sum(1 for entry in entries if entry_status(entry) == "open"),
            "closed_entries": sum(1 for entry in entries if entry_status(entry) == "closed"),
            "closed_entry_count": sum(1 for entry in entries if entry_status(entry) == "closed"),
            "entries_without_journal_id": sum(1 for entry in entries if not entry.get("journal_id")),
            "entries_without_position_id": sum(1 for entry in entries if not entry.get("position_id")),
            "duplicate_ids": duplicate_ids,
            "migration_conflicts": 0,
            "pending_sync_count": 0,
            "last_successful_backup": self._latest_backup_metadata(),
            "last_write_timestamp": max([entry.get("updated_at") or "" for entry in entries], default=None),
        }


def default_journal_db_path() -> str:
    return os.getenv("JOURNAL_DB_PATH") or os.getenv("KAIROS_JOURNAL_DB_PATH") or "/tmp/kairos_journal.sqlite3"
