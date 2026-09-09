"""Kairos Watch Contract -- persistence (2026-09 session).

A NEW, dedicated table (`watch_contracts`) in the SAME candidates database
every other Approved/Watch object already lives in (reuses the existing
persistence conventions -- `_get_db()`'s WAL-mode sqlite3 connection,
`sqlite3.Row` factory, idempotent `CREATE TABLE IF NOT EXISTS` -- rather
than a separate research-style db file like smc_shadow/hybrid_shadow use).
Deliberately NOT a stretch of `approved_setup_memories`/ManualCandidateIn:
this is a genuinely different semantic object (a frozen human chart read
Kairos monitors for an objective, multi-stage structural sequence), not a
trade plan with an optional price alert -- see the "Kairos Watch Contract"
architecture audit this session for the full reasoning.

Every function here takes `conn` as its first argument -- this module
owns no database connection/path itself (avoids a circular import with
candidates_router.py, which both calls into this module for the dashboard
read model AND supplies the connection via its own `_get_db()`).

Rows are otherwise treated as free to update in place (unlike
`approved_setup_memories`' own immutable-revision discipline) -- a Watch
Contract has no "supersede with a new generation" concept in this sprint;
its lifecycle is a single, one-way state machine per row, matching
`approved_setup_monitor_state`'s own mutable-row convention, not
`approved_setup_memories`' immutable one.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

# Real backend states -- see watch_contract_engine.py's own module comment
# for the full transition contract. WAITING_FOR_LOCATION/PULLBACK_REACHED
# are genuinely new; WATCHING/WAITING_FOR_PULLBACK/ENTRY_READY/INVALIDATED
# are reused, unchanged, from the existing dashboard_state.py vocabulary
# (per this sprint's explicit instruction: "use real backend states").
# NEEDS_REVIEW is reserved (schema-valid, never automatically emitted this
# sprint -- same "reserved, not invented" convention dashboard_state.py
# already uses for DISCOVERED/STALE).
WATCH_CONTRACT_STATES = (
    "WAITING_FOR_LOCATION",
    "WATCHING",
    "WAITING_FOR_PULLBACK",
    "PULLBACK_REACHED",
    "ENTRY_READY",
    "INVALIDATED",
    "NEEDS_REVIEW",
)
TERMINAL_WATCH_CONTRACT_STATES = frozenset({"ENTRY_READY", "INVALIDATED", "NEEDS_REVIEW"})

_COLUMNS = (
    "id", "watch_contract_id", "ticker", "source_review_id", "direction",
    "created_at", "updated_at",
    "approved_htf_thesis", "approved_current_leg",
    "location_type", "location_lower", "location_upper", "location_id",
    "location_status_at_approval", "location_reached_at",
    "confirmation_timeframe", "confirmation_direction", "confirmation_allowed_events",
    "confirmation_min_displacement",
    "confirmation_event_type", "confirmation_level", "confirmation_bar_time",
    "confirmation_prior_touch_count", "confirmation_bars_since_first_pierce",
    "pullback_reference_low", "pullback_reference_high", "pullback_cleared_at", "pullback_reached_at",
    "execution_timeframe", "execution_direction", "execution_allowed_events", "execution_min_displacement",
    "execution_event_type", "execution_level", "execution_bar_time",
    "execution_prior_touch_count", "execution_bars_since_first_pierce",
    "execution_rejection_event_type", "execution_rejection_level", "execution_rejection_bar_time",
    "approved_invalidation_rule", "approved_invalidation_level",
    "invalidated_at", "invalidation_reason",
    "state", "last_checked_at", "last_live_price",
)


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS watch_contracts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            watch_contract_id TEXT NOT NULL UNIQUE,
            ticker TEXT NOT NULL,
            source_review_id TEXT,
            direction TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            approved_htf_thesis TEXT,
            approved_current_leg TEXT,
            location_type TEXT,
            location_lower REAL,
            location_upper REAL,
            location_id TEXT,
            location_status_at_approval TEXT,
            location_reached_at TEXT,
            confirmation_timeframe TEXT,
            confirmation_direction TEXT,
            confirmation_allowed_events TEXT,
            confirmation_min_displacement TEXT,
            confirmation_event_type TEXT,
            confirmation_level REAL,
            confirmation_bar_time TEXT,
            confirmation_prior_touch_count INTEGER,
            confirmation_bars_since_first_pierce INTEGER,
            pullback_reference_low REAL,
            pullback_reference_high REAL,
            pullback_cleared_at TEXT,
            pullback_reached_at TEXT,
            execution_timeframe TEXT,
            execution_direction TEXT,
            execution_allowed_events TEXT,
            execution_min_displacement TEXT,
            execution_event_type TEXT,
            execution_level REAL,
            execution_bar_time TEXT,
            execution_prior_touch_count INTEGER,
            execution_bars_since_first_pierce INTEGER,
            execution_rejection_event_type TEXT,
            execution_rejection_level REAL,
            execution_rejection_bar_time TEXT,
            approved_invalidation_rule TEXT,
            approved_invalidation_level REAL,
            invalidated_at TEXT,
            invalidation_reason TEXT,
            state TEXT NOT NULL DEFAULT 'WAITING_FOR_LOCATION',
            last_checked_at TEXT,
            last_live_price REAL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_watch_contracts_ticker_state ON watch_contracts (ticker, state)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_watch_contracts_contract_id ON watch_contracts (watch_contract_id)")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = {k: row[k] for k in row.keys()}
    for field in ("confirmation_allowed_events", "execution_allowed_events"):
        if data.get(field):
            try:
                data[field] = json.loads(data[field])
            except (TypeError, ValueError):
                pass
    return data


def create_watch_contract(conn: sqlite3.Connection, **fields: Any) -> dict[str, Any]:
    """Creates one Watch Contract row. `fields` matches the columns above
    (unknown keys ignored, missing optional keys stored NULL -- "do not
    invent unavailable values": every field here is either explicitly
    supplied by the caller or left None, never guessed). `state` is
    computed by the caller (watch_contract_engine.initial_state) from
    `location_status_at_approval` BEFORE calling this -- this function
    persists exactly what it's given, it makes no lifecycle decision
    itself."""
    ensure_schema(conn)
    now = _utc_now_iso()
    watch_contract_id = fields.get("watch_contract_id") or f"wc_{uuid.uuid4().hex[:16]}"
    payload = {col: fields.get(col) for col in _COLUMNS if col not in ("id", "watch_contract_id", "created_at", "updated_at")}
    for field in ("confirmation_allowed_events", "execution_allowed_events"):
        if isinstance(payload.get(field), (list, tuple)):
            payload[field] = json.dumps(list(payload[field]))
    columns = ["watch_contract_id", "created_at", "updated_at", *payload.keys()]
    values = [watch_contract_id, now, now, *payload.values()]
    placeholders = ", ".join("?" for _ in columns)
    cur = conn.execute(
        f"INSERT INTO watch_contracts ({', '.join(columns)}) VALUES ({placeholders})",
        values,
    )
    conn.commit()
    return get_watch_contract(conn, watch_contract_id=watch_contract_id, row_id=cur.lastrowid)


def get_watch_contract(conn: sqlite3.Connection, watch_contract_id: Optional[str] = None, row_id: Optional[int] = None) -> Optional[dict[str, Any]]:
    ensure_schema(conn)
    if row_id is not None:
        row = conn.execute("SELECT * FROM watch_contracts WHERE id=?", (row_id,)).fetchone()
    elif watch_contract_id is not None:
        row = conn.execute("SELECT * FROM watch_contracts WHERE watch_contract_id=?", (watch_contract_id,)).fetchone()
    else:
        return None
    return _row_to_dict(row) if row is not None else None


def list_watch_contracts(conn: sqlite3.Connection, states: Optional[tuple[str, ...]] = None) -> list[dict[str, Any]]:
    ensure_schema(conn)
    if states:
        placeholders = ", ".join("?" for _ in states)
        rows = conn.execute(
            f"SELECT * FROM watch_contracts WHERE state IN ({placeholders}) ORDER BY created_at DESC", states,
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM watch_contracts ORDER BY created_at DESC").fetchall()
    return [_row_to_dict(row) for row in rows]


def update_watch_contract(conn: sqlite3.Connection, watch_contract_id: str, **fields: Any) -> Optional[dict[str, Any]]:
    ensure_schema(conn)
    if not fields:
        return get_watch_contract(conn, watch_contract_id=watch_contract_id)
    fields = dict(fields)
    fields["updated_at"] = _utc_now_iso()
    set_clause = ", ".join(f"{col}=?" for col in fields)
    conn.execute(
        f"UPDATE watch_contracts SET {set_clause} WHERE watch_contract_id=?",
        [*fields.values(), watch_contract_id],
    )
    conn.commit()
    return get_watch_contract(conn, watch_contract_id=watch_contract_id)
