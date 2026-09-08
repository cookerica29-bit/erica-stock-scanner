"""Kairos Sprint 4 -- Shadow SMC Strategy/State Engine: persistence
(2026-09 session).

Deliberately a COMPLETELY SEPARATE SQLite database from candidates.db /
journal.sqlite3 -- this file's repository class has no connection to, and
no code path that can reach, approved_setup_memories,
approved_setup_monitor_state, candidate_promotions, or journal_entries.
That separation is the actual enforcement mechanism behind "the shadow
engine must never alter positions/journal behavior" -- not just a
docstring promise, but an architectural fact (this module never opens
candidates.db/journal.sqlite3 at all).

Two tables, mirroring the same current-state + append-only-event-log
shape candidates_router.py already uses for
approved_setup_monitor_state/approved_setup_monitor_events:

  shadow_smc_state             -- ONE current row per (setup_key, strategy_version).
  shadow_smc_state_transitions -- every time that row's `state` actually
                                   changes, one more row is appended here
                                   with the from/to state, a timestamp,
                                   and the evidence snapshot that produced
                                   the new state. This is what lets a
                                   researcher answer "how did this setup
                                   get from CONFIRMED to INVALIDATED, and
                                   when" after the fact.

INVALIDATED is treated as terminal for a given (setup_key, strategy_version):
once persisted, record_evaluation() no longer overwrites that row (same
"terminal state" precedent as candidates_router.py's own
TERMINAL_MONITOR_STATES) -- a setup that has been marked invalid by this
research engine stays that way until a human explicitly resets it
(reset_state, below), rather than silently flapping back to WATCHING if a
later re-evaluation reads the structure differently.

Every row also snapshots the PRODUCTION lifecycle state at evaluation
time (production_state/production_legacy_state, passed in by the caller
-- this module does not read candidates.db itself, so it cannot silently
drift into depending on that schema) -- this is what makes "compare
production vs shadow decisions" possible without a live join.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from smc_shadow_engine import STRATEGY_VERSION

TERMINAL_SHADOW_STATES = frozenset({"INVALIDATED"})


def default_smc_shadow_db_path() -> str:
    configured = os.environ.get("KAIROS_SMC_SHADOW_DB")
    if configured:
        return configured
    mount_path = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH")
    if mount_path:
        return os.path.join(mount_path, "kairos_smc_shadow.sqlite3")
    return "/tmp/kairos_smc_shadow.sqlite3"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class SQLiteSmcShadowRepository:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or default_smc_shadow_db_path()
        parent = os.path.dirname(self.db_path)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 30000")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shadow_smc_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    setup_key TEXT NOT NULL,
                    strategy_version TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    direction TEXT,
                    state TEXT NOT NULL,
                    state_since TEXT NOT NULL,
                    production_state TEXT,
                    production_legacy_state TEXT,
                    htf_direction TEXT,
                    htf_direction_aligned INTEGER,
                    location_percentile REAL,
                    location_label TEXT,
                    zone_high REAL,
                    zone_low REAL,
                    confirmation_event_json TEXT,
                    displacement_evidence_json TEXT,
                    pullback_status TEXT,
                    pullback_detail_json TEXT,
                    execution_event_json TEXT,
                    invalidation_level REAL,
                    invalidation_reason TEXT,
                    unresolved_json TEXT,
                    evidence_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(setup_key, strategy_version)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shadow_smc_state_transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    setup_key TEXT NOT NULL,
                    strategy_version TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    from_state TEXT,
                    to_state TEXT NOT NULL,
                    transitioned_at TEXT NOT NULL,
                    production_state TEXT,
                    evidence_json TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_shadow_smc_transitions_setup "
                "ON shadow_smc_state_transitions(setup_key, strategy_version)"
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def record_evaluation(
        self,
        *,
        setup_key: str,
        ticker: str,
        direction: Optional[str],
        evaluation: dict[str, Any],
        production_state: Optional[str] = None,
        production_legacy_state: Optional[str] = None,
        strategy_version: str = STRATEGY_VERSION,
    ) -> dict[str, Any]:
        """Persists one evaluate_smc_shadow() result for this setup_key.
        Returns the resulting current-state row (dict). If the existing
        row for this (setup_key, strategy_version) is already terminal
        (INVALIDATED), this is a no-op that returns the unchanged
        existing row -- see TERMINAL_SHADOW_STATES's own docstring above."""
        now = _utc_now_iso()
        new_state = evaluation.get("state") or "WATCHING"
        with self._lock, self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM shadow_smc_state WHERE setup_key=? AND strategy_version=?",
                (setup_key, strategy_version),
            ).fetchone()

            if existing and existing["state"] in TERMINAL_SHADOW_STATES:
                return dict(existing)

            state_changed = (not existing) or (existing["state"] != new_state)
            state_since = now if state_changed else (existing["state_since"] if existing else now)

            confirmation_json = json.dumps(evaluation.get("confirmation_event") or {})
            displacement_json = json.dumps(evaluation.get("displacement_evidence") or {})
            pullback_detail_json = json.dumps(evaluation.get("pullback_detail")) if evaluation.get("pullback_detail") is not None else None
            execution_json = json.dumps(evaluation.get("execution_event") or {})
            unresolved_json = json.dumps(evaluation.get("unresolved") or [])
            evidence_json = json.dumps(evaluation, default=str)
            location = evaluation.get("location") or {}
            zone = evaluation.get("zone") or {}
            invalidation = evaluation.get("invalidation") or {}

            row_values = (
                setup_key, strategy_version, str(ticker or "").upper(), direction,
                new_state, state_since,
                production_state, production_legacy_state,
                evaluation.get("htf_direction"),
                1 if evaluation.get("htf_direction_aligned") else (0 if evaluation.get("htf_direction_aligned") is not None else None),
                location.get("percentile"), location.get("label"),
                zone.get("high"), zone.get("low"),
                confirmation_json, displacement_json,
                evaluation.get("pullback_status"), pullback_detail_json,
                execution_json,
                invalidation.get("level"), invalidation.get("reason"),
                unresolved_json, evidence_json,
                now,
            )

            if existing:
                conn.execute(
                    """
                    UPDATE shadow_smc_state SET
                        ticker=?, direction=?, state=?, state_since=?,
                        production_state=?, production_legacy_state=?,
                        htf_direction=?, htf_direction_aligned=?,
                        location_percentile=?, location_label=?,
                        zone_high=?, zone_low=?,
                        confirmation_event_json=?, displacement_evidence_json=?,
                        pullback_status=?, pullback_detail_json=?,
                        execution_event_json=?,
                        invalidation_level=?, invalidation_reason=?,
                        unresolved_json=?, evidence_json=?, updated_at=?
                    WHERE setup_key=? AND strategy_version=?
                    """,
                    row_values[2:] + (setup_key, strategy_version),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO shadow_smc_state(
                        setup_key, strategy_version, ticker, direction, state, state_since,
                        production_state, production_legacy_state,
                        htf_direction, htf_direction_aligned,
                        location_percentile, location_label,
                        zone_high, zone_low,
                        confirmation_event_json, displacement_evidence_json,
                        pullback_status, pullback_detail_json,
                        execution_event_json,
                        invalidation_level, invalidation_reason,
                        unresolved_json, evidence_json, created_at, updated_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    row_values[:23] + (now, now),
                )

            if state_changed:
                conn.execute(
                    """
                    INSERT INTO shadow_smc_state_transitions(
                        setup_key, strategy_version, ticker, from_state, to_state,
                        transitioned_at, production_state, evidence_json
                    ) VALUES (?,?,?,?,?,?,?,?)
                    """,
                    (
                        setup_key, strategy_version, str(ticker or "").upper(),
                        existing["state"] if existing else None, new_state,
                        now, production_state, evidence_json,
                    ),
                )

            conn.commit()
            result = conn.execute(
                "SELECT * FROM shadow_smc_state WHERE setup_key=? AND strategy_version=?",
                (setup_key, strategy_version),
            ).fetchone()
            return dict(result)

    def reset_state(self, setup_key: str, strategy_version: str = STRATEGY_VERSION) -> None:
        """Explicit human override to un-terminate a setup (e.g. a fresh
        approval cycle for the same setup_key) -- deletes the current-state
        row only; the transition history is kept for research."""
        with self._lock, self._connect() as conn:
            conn.execute(
                "DELETE FROM shadow_smc_state WHERE setup_key=? AND strategy_version=?",
                (setup_key, strategy_version),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_current(self, setup_key: str, strategy_version: str = STRATEGY_VERSION) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM shadow_smc_state WHERE setup_key=? AND strategy_version=?",
                (setup_key, strategy_version),
            ).fetchone()
            return dict(row) if row else None

    def list_current(self, strategy_version: str = STRATEGY_VERSION) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM shadow_smc_state WHERE strategy_version=? ORDER BY updated_at DESC",
                (strategy_version,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_transitions(self, setup_key: str, strategy_version: str = STRATEGY_VERSION) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM shadow_smc_state_transitions WHERE setup_key=? AND strategy_version=? ORDER BY transitioned_at ASC",
                (setup_key, strategy_version),
            ).fetchall()
            return [dict(r) for r in rows]
