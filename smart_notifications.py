from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SMART_NOTIFICATION_VERSION = "smart-notifications-v1"
DEFAULT_PREFERENCES = {
    "NEW_ENTER_NOW": {"in_app": True, "external": True},
    "PLANNED_ENTRY_REACHED": {"in_app": True, "external": True},
    "TP1_REACHED": {"in_app": True, "external": True},
    "STOP_REACHED": {"in_app": True, "external": True},
    "SETUP_INVALIDATED": {"in_app": True, "external": True},
    "POSITION_STATUS_CHANGE": {"in_app": True, "external": True},
}
EXTERNAL_FRESHNESS_MINUTES = {
    "NEW_ENTER_NOW": 60,
    "PLANNED_ENTRY_REACHED": 60,
    "TP1_REACHED": 240,
    "STOP_REACHED": 240,
    "SETUP_INVALIDATED": 60,
    "POSITION_STATUS_CHANGE": 60,
}
PRIORITY_BY_TYPE = {
    "STOP_REACHED": "HIGH",
    "SETUP_INVALIDATED": "HIGH",
    "PLANNED_ENTRY_REACHED": "HIGH",
    "NEW_ENTER_NOW": "MEDIUM",
    "TP1_REACHED": "MEDIUM",
    "POSITION_STATUS_CHANGE": "MEDIUM",
}


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


def first_present(*values):
    for value in values:
        if value not in (None, ""):
            return value
    return None


def finite_number(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def money(value) -> str:
    number = finite_number(value)
    return "unavailable" if number is None else f"${number:.2f}"


def normalize_direction(value: Any) -> str:
    direction = str(value or "").strip().upper()
    if direction in {"CALL", "LONG"}:
        return "LONG"
    if direction in {"PUT", "SHORT"}:
        return "SHORT"
    return direction


def normalize_bucket(value: Any) -> str:
    text = str(value or "").strip().upper().replace("+", "").replace(" ", "_").replace("-", "_")
    aliases = {
        "A_READY": "ENTER_NOW",
        "B_TRADEABLE": "ENTER_NOW",
        "TRADEABLE": "ENTER_NOW",
        "ENTRY_REACHED": "ENTER_NOW",
        "DEVELOPING": "WAITING",
        "WATCHLIST": "WAITING",
        "NO_TRADE": "SKIP",
        "INVALIDATED": "SKIP",
        "REJECTED": "SKIP",
    }
    return aliases.get(text, text or "UNKNOWN")


def setup_bucket(setup: dict[str, Any]) -> str:
    ranking = setup.get("ranking") if isinstance(setup.get("ranking"), dict) else {}
    trade_eval = setup.get("trade_eval") if isinstance(setup.get("trade_eval"), dict) else {}
    return normalize_bucket(first_present(
        ranking.get("status_bucket"),
        setup.get("progress_bucket"),
        setup.get("status_bucket"),
        setup.get("scanner_status_normalized"),
        setup.get("scanner_status"),
        trade_eval.get("trade_stage"),
        setup.get("setupStatus"),
        setup.get("setup_status"),
        setup.get("entryStatus"),
        setup.get("status"),
    ))


def stale_early_touch_blocks_enter_now(setup: dict[str, Any]) -> bool:
    shadow = setup.get("early_entry_shadow") if isinstance(setup.get("early_entry_shadow"), dict) else {}
    lifecycle = str(first_present(setup.get("execution_lifecycle_state"), shadow.get("state")) or "").strip().upper()
    if not lifecycle:
        return False
    if lifecycle == "ENTRY_TRIGGERED":
        return False
    return lifecycle in {
        "EARLY_TOUCH",
        "WAITING_FOR_RETEST",
        "MISSED_ENTRY",
        "TP1_BEFORE_CONFIRMATION",
        "INVALIDATED",
        "EXPIRED",
        "PLAN_REPLACED",
    }


def setup_identity(setup: dict[str, Any]) -> str:
    explicit = first_present(setup.get("setup_id"), setup.get("setupId"), setup.get("id"))
    if explicit:
        return str(explicit)
    parts = [
        str(setup.get("ticker") or "").upper(),
        normalize_direction(setup.get("direction")),
        first_present(setup.get("timeframe"), setup.get("scanner_timeframe"), setup.get("setupTf"), ""),
        first_present(setup.get("signal_timestamp"), setup.get("signalTimestamp"), setup.get("candle_timestamp"), setup.get("signal_market_date"), ""),
        finite_number(first_present(setup.get("entry"), setup.get("entry_price"))),
        finite_number(first_present(setup.get("sl"), setup.get("stop"), setup.get("stop_price"))),
        finite_number(first_present(setup.get("tp1"), setup.get("target"), setup.get("target_price"))),
    ]
    raw = "|".join("NA" if value in (None, "") else str(value) for value in parts)
    return f"setup:{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def source_timestamp(setup: dict[str, Any], meta: dict[str, Any] | None = None) -> str:
    meta = meta or {}
    return str(first_present(
        setup.get("signal_timestamp"),
        setup.get("signalTimestamp"),
        setup.get("candle_timestamp"),
        setup.get("candleTime"),
        meta.get("scan_completed_at"),
        meta.get("generated_at"),
        utc_now_iso(),
    ))


def current_price(setup: dict[str, Any]) -> float | None:
    return finite_number(first_present(
        setup.get("current_quote_price"),
        setup.get("currentQuotePrice"),
        setup.get("price"),
        setup.get("current_price"),
        setup.get("currentPrice"),
        setup.get("underlying_price"),
    ))


def planned_entry(setup: dict[str, Any]) -> float | None:
    return finite_number(first_present(setup.get("entry"), setup.get("entry_price"), setup.get("planned_underlying_entry")))


def target_one(setup: dict[str, Any]) -> float | None:
    return finite_number(first_present(setup.get("tp1"), setup.get("target"), setup.get("target_price"), setup.get("plannedTp1"), setup.get("original_tp1")))


def stop_level(setup: dict[str, Any]) -> float | None:
    return finite_number(first_present(setup.get("sl"), setup.get("stop"), setup.get("stop_price"), setup.get("plannedStop"), setup.get("original_stop")))


def entry_reached(setup: dict[str, Any]) -> bool:
    entry = planned_entry(setup)
    price = current_price(setup)
    direction = normalize_direction(setup.get("direction"))
    if entry is None or price is None:
        return False
    if direction == "SHORT":
        return price <= entry
    if direction == "LONG":
        return price >= entry
    return abs(price - entry) < 0.000001


def target_reached(setup: dict[str, Any]) -> bool:
    target = target_one(setup)
    price = current_price(setup)
    direction = normalize_direction(setup.get("direction"))
    if target is None or price is None:
        return False
    if direction == "SHORT":
        return price <= target
    if direction == "LONG":
        return price >= target
    return False


def invalidated(setup: dict[str, Any]) -> bool:
    bucket = setup_bucket(setup)
    return bucket in {"SKIP", "INVALIDATED", "REJECTED"}


def setup_too_far(setup: dict[str, Any]) -> bool:
    values = [
        setup.get("entryStatus"),
        setup.get("entry_timing_state"),
        setup.get("entryTimingState"),
        setup.get("timing_state"),
        setup.get("status"),
    ]
    return any("TOO" in str(value or "").upper() and "FAR" in str(value or "").upper() for value in values)


def stable_event_id(dedupe_key: str) -> str:
    return f"notif:{hashlib.sha1(dedupe_key.encode('utf-8')).hexdigest()}"


def planned_entry_cycle_key(setup: dict[str, Any], event_type: str, level: float | None) -> str:
    symbol = str(setup.get("ticker") or "UNKNOWN").upper()
    direction = normalize_direction(setup.get("direction"))
    setup_id = setup_identity(setup)
    signal_time = str(first_present(
        setup.get("signal_timestamp"),
        setup.get("signalTimestamp"),
        setup.get("candle_timestamp"),
        setup.get("candleTime"),
        setup.get("signal_market_date"),
        "",
    ))
    return "|".join([
        event_type,
        setup_id,
        symbol,
        direction,
        signal_time,
        str(level if level is not None else "NA"),
    ])


def event_is_fresh(event_type: str, event_time: str, now: datetime | None = None) -> bool:
    limit = EXTERNAL_FRESHNESS_MINUTES.get(event_type)
    if limit is None:
        return True
    parsed = parse_utc(event_time)
    if not parsed:
        return False
    now = now or datetime.now(timezone.utc)
    return (now - parsed).total_seconds() <= limit * 60


def notification_title(event_type: str, symbol: str) -> str:
    labels = {
        "NEW_ENTER_NOW": f"{symbol} is now Enter Now",
        "PLANNED_ENTRY_REACHED": f"{symbol} reached planned entry",
        "TP1_REACHED": f"{symbol} reached TP1",
        "STOP_REACHED": f"{symbol} crossed the planned stop",
        "SETUP_INVALIDATED": f"{symbol} is no longer valid",
        "POSITION_STATUS_CHANGE": f"{symbol} position needs review",
    }
    return labels.get(event_type, f"{symbol} notification")


def build_setup_event(
    event_type: str,
    setup: dict[str, Any],
    previous_state: str | None,
    current_state: str,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    symbol = str(setup.get("ticker") or "UNKNOWN").upper()
    direction = normalize_direction(setup.get("direction"))
    setup_id = setup_identity(setup)
    event_time = source_timestamp(setup, meta)
    level = planned_entry(setup) if event_type in {"NEW_ENTER_NOW", "PLANNED_ENTRY_REACHED"} else None
    level_name = "Planned Entry" if level is not None else None
    if event_type == "SETUP_INVALIDATED":
        level = stop_level(setup)
        level_name = "Stop Loss" if level is not None else None
    if event_type == "PLANNED_ENTRY_REACHED":
        dedupe_key = planned_entry_cycle_key(setup, event_type, level)
    else:
        dedupe_key = "|".join([symbol, setup_id, event_type, event_time, str(level if level is not None else "NA")])
    if event_type == "PLANNED_ENTRY_REACHED":
        message = f"{symbol} reached its planned entry. {direction.title()} setup. Planned Entry: {money(level)}. Current Price: {money(current_price(setup))}."
        next_step = "Review the option plan before executing."
    elif event_type == "SETUP_INVALIDATED":
        message = f"{symbol} is no longer valid. The setup was invalidated before entry."
        next_step = "Remove the entry alert and wait for a new setup."
    else:
        waiting = setup_bucket(setup) == "ENTER_NOW" and not entry_reached(setup)
        message = f"{symbol} setup is confirmed. Current Price: {money(current_price(setup))}. Planned Entry: {money(level)}." if waiting else f"{symbol} is now Enter Now. Direction: {direction.title()}. Grade: {first_present(setup.get('setupGrade'), setup.get('grade'), 'Unknown')}."
        next_step = "Set an alert at the planned entry and wait." if waiting else "Open the trade plan and confirm the current price before entering."
    return {
        "event_id": stable_event_id(dedupe_key),
        "version": SMART_NOTIFICATION_VERSION,
        "symbol": symbol,
        "direction": direction,
        "event_type": event_type,
        "priority": PRIORITY_BY_TYPE.get(event_type, "LOW"),
        "title": notification_title(event_type, symbol),
        "message": message,
        "next_step": next_step,
        "previous_state": previous_state,
        "current_state": current_state,
        "setup_id": setup_id,
        "position_id": None,
        "entity_type": "setup",
        "entity_id": setup_id,
        "deep_link": f"setup:{setup_id}",
        "event_time": event_time,
        "source_event_time": event_time,
        "detected_at": utc_now_iso(),
        "current_price": current_price(setup),
        "relevant_level": level,
        "level_name": level_name,
        "grade": first_present(setup.get("setupGrade"), setup.get("grade")),
        "status": current_state,
        "dedupe_key": dedupe_key,
        "source": "stock-scanner",
        "delivery_status": "pending",
        "external_delivery_status": "not_configured",
        "metadata": {
            "scan_timestamp": (meta or {}).get("scan_completed_at") or (meta or {}).get("generated_at"),
            "fresh_for_external": event_is_fresh(event_type, event_time),
        },
    }


class SQLiteNotificationRepository:
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
                CREATE TABLE IF NOT EXISTS smart_notifications (
                    id TEXT PRIMARY KEY,
                    dedupe_key TEXT NOT NULL UNIQUE,
                    event_type TEXT NOT NULL,
                    symbol TEXT,
                    direction TEXT,
                    priority TEXT,
                    entity_type TEXT,
                    entity_id TEXT,
                    source_event_time TEXT,
                    created_at TEXT NOT NULL,
                    read_at TEXT,
                    external_delivery_status TEXT,
                    external_delivered_at TEXT,
                    delivery_error TEXT,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_smart_notifications_created ON smart_notifications(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_smart_notifications_read ON smart_notifications(read_at)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS smart_notification_entity_states (
                    entity_id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    symbol TEXT,
                    direction TEXT,
                    state TEXT,
                    entry_reached INTEGER NOT NULL DEFAULT 0,
                    tp1_reached INTEGER NOT NULL DEFAULT 0,
                    stop_reached INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS smart_notification_preferences (
                    preference_scope TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def get_preferences(self, scope: str = "default") -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM smart_notification_preferences WHERE preference_scope=?", (scope,)).fetchone()
        if not row:
            return json.loads(json.dumps(DEFAULT_PREFERENCES))
        saved = json.loads(row["payload"] or "{}")
        merged = json.loads(json.dumps(DEFAULT_PREFERENCES))
        for key, value in saved.items():
            if key in merged and isinstance(value, dict):
                merged[key].update({k: bool(v) for k, v in value.items() if k in {"in_app", "external"}})
        return merged

    def update_preferences(self, preferences: dict[str, Any], scope: str = "default") -> dict[str, Any]:
        merged = self.get_preferences(scope)
        for key, value in (preferences or {}).items():
            if key in merged and isinstance(value, dict):
                merged[key].update({k: bool(v) for k, v in value.items() if k in {"in_app", "external"}})
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO smart_notification_preferences(preference_scope, payload, updated_at)
                VALUES(?, ?, ?)
                ON CONFLICT(preference_scope) DO UPDATE SET payload=excluded.payload, updated_at=excluded.updated_at
                """,
                (scope, json.dumps(merged, sort_keys=True), now),
            )
        return merged

    def state_for(self, entity_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM smart_notification_entity_states WHERE entity_id=?", (entity_id,)).fetchone()
        if not row:
            return None
        payload = json.loads(row["payload"] or "{}")
        payload.update({
            "entity_id": row["entity_id"],
            "entity_type": row["entity_type"],
            "symbol": row["symbol"],
            "direction": row["direction"],
            "state": row["state"],
            "entry_reached": bool(row["entry_reached"]),
            "tp1_reached": bool(row["tp1_reached"]),
            "stop_reached": bool(row["stop_reached"]),
            "updated_at": row["updated_at"],
        })
        return payload

    def save_state(self, entity_id: str, entity_type: str, symbol: str, direction: str, state: str, flags: dict[str, bool], payload: dict[str, Any] | None = None) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO smart_notification_entity_states(entity_id, entity_type, symbol, direction, state, entry_reached, tp1_reached, stop_reached, updated_at, payload)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_id) DO UPDATE SET
                    entity_type=excluded.entity_type,
                    symbol=excluded.symbol,
                    direction=excluded.direction,
                    state=excluded.state,
                    entry_reached=excluded.entry_reached,
                    tp1_reached=excluded.tp1_reached,
                    stop_reached=excluded.stop_reached,
                    updated_at=excluded.updated_at,
                    payload=excluded.payload
                """,
                (
                    entity_id,
                    entity_type,
                    symbol,
                    direction,
                    state,
                    1 if flags.get("entry_reached") else 0,
                    1 if flags.get("tp1_reached") else 0,
                    1 if flags.get("stop_reached") else 0,
                    now,
                    json.dumps(payload or {}, sort_keys=True),
                ),
            )

    def create_event(self, event: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
        preferences = self.get_preferences()
        pref = preferences.get(event.get("event_type")) or {}
        if not pref.get("in_app", True):
            return None, False
        now = utc_now_iso()
        payload = dict(event)
        payload.setdefault("created_at", now)
        payload.setdefault("read_at", None)
        payload.setdefault("external_delivery_status", "suppressed_by_preference" if not pref.get("external", True) else "not_configured")
        payload.setdefault("external_delivered_at", None)
        payload.setdefault("delivery_error", None)
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO smart_notifications(
                        id, dedupe_key, event_type, symbol, direction, priority, entity_type, entity_id,
                        source_event_time, created_at, read_at, external_delivery_status, external_delivered_at,
                        delivery_error, payload
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        payload["event_id"],
                        payload["dedupe_key"],
                        payload["event_type"],
                        payload.get("symbol"),
                        payload.get("direction"),
                        payload.get("priority"),
                        payload.get("entity_type"),
                        payload.get("entity_id"),
                        payload.get("source_event_time"),
                        now,
                        payload.get("read_at"),
                        payload.get("external_delivery_status"),
                        payload.get("external_delivered_at"),
                        payload.get("delivery_error"),
                        json.dumps(payload, sort_keys=True),
                    ),
                )
            return self.get_event(payload["event_id"]), True
        except sqlite3.IntegrityError:
            return None, False

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM smart_notifications WHERE id=?", (event_id,)).fetchone()
        return self._row_to_event(row) if row else None

    def _row_to_event(self, row: sqlite3.Row) -> dict[str, Any]:
        payload = json.loads(row["payload"] or "{}")
        payload.update({
            "id": row["id"],
            "event_id": row["id"],
            "created_at": row["created_at"],
            "read_at": row["read_at"],
            "external_delivery_status": row["external_delivery_status"],
            "external_delivered_at": row["external_delivered_at"],
            "delivery_error": row["delivery_error"],
        })
        return payload

    def list_events(self, unread_only: bool = False, limit: int = 50, offset: int = 0) -> dict[str, Any]:
        where = []
        params: list[Any] = []
        if unread_only:
            where.append("read_at IS NULL")
        sql = "SELECT * FROM smart_notifications"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([max(1, min(int(limit), 200)), max(0, int(offset))])
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            unread = conn.execute("SELECT COUNT(*) FROM smart_notifications WHERE read_at IS NULL").fetchone()[0]
            total = conn.execute("SELECT COUNT(*) FROM smart_notifications").fetchone()[0]
        return {
            "version": SMART_NOTIFICATION_VERSION,
            "events": [self._row_to_event(row) for row in rows],
            "unread_count": int(unread),
            "total_count": int(total),
            "limit": limit,
            "offset": offset,
        }

    def all_events(self, limit: int = 1000) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM smart_notifications ORDER BY created_at DESC LIMIT ?",
                (max(1, min(int(limit), 5000)),),
            ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def mark_read(self, event_id: str) -> dict[str, Any] | None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute("UPDATE smart_notifications SET read_at=COALESCE(read_at, ?) WHERE id=?", (now, event_id))
        return self.get_event(event_id)

    def mark_all_read(self) -> dict[str, Any]:
        now = utc_now_iso()
        with self._connect() as conn:
            cur = conn.execute("UPDATE smart_notifications SET read_at=COALESCE(read_at, ?) WHERE read_at IS NULL", (now,))
        return {"updated": int(cur.rowcount or 0), "read_at": now}

    def diagnostics(self) -> dict[str, Any]:
        with self._connect() as conn:
            rows = conn.execute("SELECT event_type, external_delivery_status, read_at FROM smart_notifications").fetchall()
            state_count = conn.execute("SELECT COUNT(*) FROM smart_notification_entity_states").fetchone()[0]
        by_type: dict[str, int] = {}
        delivery: dict[str, int] = {}
        unread = 0
        for row in rows:
            by_type[row["event_type"]] = by_type.get(row["event_type"], 0) + 1
            delivery[row["external_delivery_status"] or "unknown"] = delivery.get(row["external_delivery_status"] or "unknown", 0) + 1
            if not row["read_at"]:
                unread += 1
        return {
            "version": SMART_NOTIFICATION_VERSION,
            "events_total": len(rows),
            "unread": unread,
            "events_by_type": dict(sorted(by_type.items())),
            "delivery_status": dict(sorted(delivery.items())),
            "tracked_entity_states": int(state_count),
        }

    def _current_setup_indexes(self, current_setups: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        by_setup_id: dict[str, dict[str, Any]] = {}
        by_fallback: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for setup in current_setups or []:
            if not isinstance(setup, dict):
                continue
            explicit = first_present(setup.get("setup_id"), setup.get("setupId"), setup.get("id"))
            generated = setup_identity(setup)
            for key in {str(explicit) if explicit else "", generated}:
                if key:
                    by_setup_id[key] = setup
            fallback = "|".join([
                str(setup.get("ticker") or "").upper(),
                normalize_direction(setup.get("direction")),
                str(planned_entry(setup) if planned_entry(setup) is not None else "NA"),
            ])
            by_fallback[fallback].append(setup)
        return {"by_setup_id": by_setup_id, "by_fallback": by_fallback}

    def _matching_setup(self, event: dict[str, Any], indexes: dict[str, Any]) -> dict[str, Any] | None:
        for key in [event.get("setup_id"), event.get("entity_id")]:
            if key and key in indexes["by_setup_id"]:
                return indexes["by_setup_id"][key]
        fallback = "|".join([
            str(event.get("symbol") or "").upper(),
            normalize_direction(event.get("direction")),
            str(finite_number(event.get("relevant_level")) if finite_number(event.get("relevant_level")) is not None else "NA"),
        ])
        candidates = indexes["by_fallback"].get(fallback) or []
        return candidates[0] if len(candidates) == 1 else None

    def classify_event_actionability(self, event: dict[str, Any], current_setup: dict[str, Any] | None = None) -> dict[str, Any]:
        event_type = str(event.get("event_type") or "")
        if not current_setup:
            return {
                "state": "RESOLVED",
                "current_lifecycle_state": "No current setup",
                "reason": "No matching current scanner setup.",
                "has_current_setup": False,
            }
        bucket = setup_bucket(current_setup)
        lifecycle = current_setup.get("entryStatus") or current_setup.get("entry_timing_state") or bucket
        if invalidated(current_setup):
            return {
                "state": "RESOLVED",
                "current_lifecycle_state": str(lifecycle or "Invalidated"),
                "reason": "The setup is no longer valid in the current scanner output.",
                "has_current_setup": True,
            }
        if setup_too_far(current_setup):
            return {
                "state": "EARLIER",
                "current_lifecycle_state": str(lifecycle or "Too Far"),
                "reason": "Price moved beyond the execution window.",
                "has_current_setup": True,
            }
        if target_reached(current_setup):
            return {
                "state": "EARLIER",
                "current_lifecycle_state": "At or beyond TP1",
                "reason": "Price is already at or beyond the first target.",
                "has_current_setup": True,
            }
        if event_type in {"NEW_ENTER_NOW", "PLANNED_ENTRY_REACHED"} and bucket == "ENTER_NOW":
            return {
                "state": "ACTIONABLE",
                "current_lifecycle_state": str(lifecycle or "Enter Now"),
                "reason": "Current scanner setup remains valid for review.",
                "has_current_setup": True,
            }
        if event_type in {"TP1_REACHED", "STOP_REACHED"}:
            return {
                "state": "EARLIER",
                "current_lifecycle_state": str(lifecycle or bucket),
                "reason": "This is a lifecycle milestone, not a current entry action.",
                "has_current_setup": True,
            }
        return {
            "state": "EARLIER",
            "current_lifecycle_state": str(lifecycle or bucket),
            "reason": "The setup has moved into another lifecycle stage.",
            "has_current_setup": True,
        }

    def audit_events(self, current_setups: list[dict[str, Any]] | None = None, limit: int = 1000) -> dict[str, Any]:
        events = self.all_events(limit=limit)
        indexes = self._current_setup_indexes(current_setups or [])
        by_type = Counter(str(event.get("event_type") or "UNKNOWN") for event in events)
        unique_tickers = {str(event.get("symbol") or "").upper() for event in events if event.get("symbol")}
        unique_setup_ids = {str(first_present(event.get("setup_id"), event.get("entity_id"), "")) for event in events if first_present(event.get("setup_id"), event.get("entity_id"), "")}
        actionability = Counter()
        planned_entry_counts = Counter()
        duplicate_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in events:
            setup = self._matching_setup(event, indexes)
            classification = self.classify_event_actionability(event, setup)
            actionability[classification["state"]] += 1
            if event.get("event_type") == "PLANNED_ENTRY_REACHED":
                if setup_too_far(setup or {}):
                    planned_entry_counts["now_too_far"] += 1
                if invalidated(setup or {}):
                    planned_entry_counts["invalidated"] += 1
                if target_reached(setup or {}):
                    planned_entry_counts["already_at_or_beyond_tp1"] += 1
                if not setup:
                    planned_entry_counts["no_matching_current_setup"] += 1
                key = "|".join([
                    "PLANNED_ENTRY_REACHED",
                    str(first_present(event.get("setup_id"), event.get("entity_id"), "")),
                    str(event.get("symbol") or "").upper(),
                    normalize_direction(event.get("direction")),
                    str(finite_number(event.get("relevant_level")) if finite_number(event.get("relevant_level")) is not None else "NA"),
                ])
                duplicate_groups[key].append({
                    "event_id": event.get("event_id") or event.get("id"),
                    "ticker": event.get("symbol"),
                    "direction": event.get("direction"),
                    "plannedEntry": event.get("relevant_level"),
                    "setupId": first_present(event.get("setup_id"), event.get("entity_id")),
                    "signalTimestamp": event.get("source_event_time"),
                    "candleTime": (event.get("metadata") or {}).get("scan_timestamp"),
                    "lifecycleState": event.get("current_state") or event.get("status"),
                    "created_at": event.get("created_at"),
                    "read_at": event.get("read_at"),
                })
        duplicates = [
            {"dedupe_key": key, "count": len(group), "events": group[:8]}
            for key, group in duplicate_groups.items()
            if len(group) > 1
        ]
        duplicates.sort(key=lambda item: item["count"], reverse=True)
        unread = sum(1 for event in events if not event.get("read_at"))
        return {
            "version": SMART_NOTIFICATION_VERSION,
            "total_notifications": len(events),
            "total_unread": unread,
            "count_by_notification_type": dict(sorted(by_type.items())),
            "unique_tickers": len(unique_tickers),
            "unique_setup_ids": len(unique_setup_ids),
            "planned_entry_reached": by_type.get("PLANNED_ENTRY_REACHED", 0),
            "actionability_counts": dict(sorted(actionability.items())),
            "planned_entry_breakdown": dict(sorted(planned_entry_counts.items())),
            "suspected_duplicate_groups": duplicates,
            "duplicate_group_count": len(duplicates),
            "duplicate_records_beyond_first": sum(item["count"] - 1 for item in duplicates),
            "dedupe_identity_fields": ["notificationType", "setupId", "ticker", "direction", "plannedEntry", "signalTimestamp"],
        }

    def duplicate_cleanup_preview(self, current_setups: list[dict[str, Any]] | None = None, limit: int = 1000) -> dict[str, Any]:
        audit = self.audit_events(current_setups=current_setups, limit=limit)
        duplicate_groups = audit.get("suspected_duplicate_groups") or []
        return {
            "version": SMART_NOTIFICATION_VERSION,
            "action": "preview_duplicate_cleanup",
            "destructive": False,
            "duplicate_groups": len(duplicate_groups),
            "records_that_would_be_retained": len(duplicate_groups),
            "records_that_would_be_archived_or_marked_resolved": sum(max(0, int(group.get("count") or 0) - 1) for group in duplicate_groups),
            "dedupe_key_used": audit.get("dedupe_identity_fields"),
            "groups": duplicate_groups[:25],
        }

    def evaluate_scan(self, rows: list[dict[str, Any]], meta: dict[str, Any] | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        created = []
        deduped = 0
        evaluated = 0
        suppressed_stale = 0
        for setup in rows or []:
            if not isinstance(setup, dict) or not setup.get("ticker"):
                continue
            evaluated += 1
            entity_id = setup_identity(setup)
            state = setup_bucket(setup)
            direction = normalize_direction(setup.get("direction"))
            flags = {"entry_reached": entry_reached(setup), "tp1_reached": False, "stop_reached": False}
            previous = self.state_for(entity_id)
            previous_state = previous.get("state") if previous else None
            events = []
            enter_now_blocked_by_memory = state == "ENTER_NOW" and stale_early_touch_blocks_enter_now(setup)
            if previous:
                if previous_state != "ENTER_NOW" and state == "ENTER_NOW" and not enter_now_blocked_by_memory:
                    events.append(build_setup_event("NEW_ENTER_NOW", setup, previous_state, state, meta))
                if not previous.get("entry_reached") and flags["entry_reached"] and not enter_now_blocked_by_memory:
                    events.append(build_setup_event("PLANNED_ENTRY_REACHED", setup, previous_state, state, meta))
                if previous_state not in {"SKIP", "INVALIDATED", "REJECTED"} and invalidated(setup):
                    events.append(build_setup_event("SETUP_INVALIDATED", setup, previous_state, state, meta))
            self.save_state(
                entity_id,
                "setup",
                str(setup.get("ticker") or "").upper(),
                direction,
                state,
                flags,
                {"last_seen_at": utc_now_iso(), "entry": planned_entry(setup), "current_price": current_price(setup)},
            )
            for event in events:
                saved, inserted = self.create_event(event)
                if inserted and saved:
                    created.append(saved)
                else:
                    deduped += 1
                if not event.get("metadata", {}).get("fresh_for_external", True):
                    suppressed_stale += 1
        return {
            "version": SMART_NOTIFICATION_VERSION,
            "events_evaluated": evaluated,
            "events_created": len(created),
            "events_deduplicated": deduped,
            "events_suppressed_by_preference": 0,
            "events_suppressed_as_stale": suppressed_stale,
            "in_app_notifications_created": len(created),
            "external_notifications_attempted": 0,
            "external_notifications_delivered": 0,
            "external_delivery_failures": 0,
            "notification_evaluation_ms": round((time.perf_counter() - started) * 1000, 1),
            "notification_persistence_ms": None,
            "external_delivery_ms": 0,
            "dedupe_lookup_ms": None,
            "affected_tickers": [event.get("symbol") for event in created],
            "created_events": created,
        }
