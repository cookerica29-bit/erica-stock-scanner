"""Kairos Sprint 5 -- Historical + Paper Evaluation Harness: persistence
(2026-09 session). Sprint 5.2: schema renamed/hardened to match
smc_shadow_evaluation.py's own Sprint 5.2 outcome-semantics change -- see
that module's docstring. No primary outcome/win/loss/r_multiple column
exists anywhere in this schema anymore.

Same isolated database as smc_shadow_store.py (kairos_smc_shadow.sqlite3,
KAIROS_SMC_SHADOW_DB-overridable) -- this is still shadow-only research
data, with no code path to candidates.db or journal.sqlite3. New tables,
not new files, so every piece of shadow-engine data (current state,
transition log, and now entry-ready-signal evaluation results) lives in
one place, still completely separate from production.

Two tables:
  smc_shadow_evaluation_runs        -- one row per run_walk_forward_evaluation()
                                        call (metadata: ticker, direction,
                                        market, strategy_version,
                                        harness_version, counts).
  smc_shadow_entry_ready_signals    -- one row per entry-ready SIGNAL
                                        (Sprint 5.2 rename from
                                        "hypothetical trade" -- see
                                        smc_shadow_evaluation.py's module
                                        docstring), with every required
                                        output field as its own column
                                        PLUS the full evidence and
                                        transition-history JSON for
                                        complete traceability back to the
                                        exact state evidence that
                                        produced it. There is no outcome/
                                        win/loss/r_multiple column --
                                        trade_performance_status is a
                                        fixed constant column value.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from smc_shadow_store import default_smc_shadow_db_path
from smc_shadow_evaluation import (
    TRADE_PERFORMANCE_UNAVAILABLE,
    aggregate_evaluation_report,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# Sprint 5.2: columns whose values must round-trip through sqlite's
# INTEGER 0/1/NULL representation back into Python True/False/None
# (three-valued, never silently collapsed to a plain bool) when read
# back for report() aggregation.
_TRISTATE_BOOL_COLUMNS = (
    "post_entry_ready_invalidation_touched",
    "post_entry_ready_target_assumption_touched",
    "ambiguous_intrabar_between_invalidation_and_target_assumption",
)


def _normalize_signal_row(row: dict[str, Any]) -> dict[str, Any]:
    for col in _TRISTATE_BOOL_COLUMNS:
        if col in row and row[col] is not None:
            row[col] = bool(row[col])
    return row


class SQLiteSmcShadowEvaluationRepository:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or default_smc_shadow_db_path()
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
                CREATE TABLE IF NOT EXISTS smc_shadow_evaluation_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL UNIQUE,
                    kind TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    direction TEXT,
                    market TEXT,
                    strategy_version TEXT NOT NULL,
                    harness_version TEXT NOT NULL,
                    episode_count INTEGER NOT NULL DEFAULT 0,
                    signal_count INTEGER NOT NULL DEFAULT 0,
                    missed_due_to_no_pullback_count INTEGER NOT NULL DEFAULT 0,
                    invalidated_before_entry_ready_count INTEGER NOT NULL DEFAULT 0,
                    incomplete_episode_count INTEGER NOT NULL DEFAULT 0,
                    terminal_state_reached_json TEXT,
                    max_progression_reached_json TEXT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS smc_shadow_entry_ready_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    direction TEXT,
                    market TEXT,
                    entry_ready_bar_time TEXT,
                    assumed_entry_price REAL,
                    stop REAL,
                    assumed_target REAL,
                    assumed_target_r_multiple REAL,
                    assumed_risk REAL,
                    direction_provenance TEXT,
                    entry_ready_bar_time_provenance TEXT,
                    stop_provenance TEXT,
                    assumed_entry_price_provenance TEXT,
                    assumed_target_provenance TEXT,
                    trade_performance_status TEXT,
                    post_entry_ready_invalidation_touched INTEGER,
                    post_entry_ready_invalidation_touched_bar_time TEXT,
                    post_entry_ready_invalidation_fill_type TEXT,
                    bars_from_entry_ready_to_invalidation_touch INTEGER,
                    post_entry_ready_target_assumption_touched INTEGER,
                    post_entry_ready_target_assumption_touched_bar_time TEXT,
                    post_entry_ready_target_assumption_fill_type TEXT,
                    bars_from_entry_ready_to_target_assumption_touch INTEGER,
                    ambiguous_intrabar_between_invalidation_and_target_assumption INTEGER,
                    ambiguous_intrabar_bar_time TEXT,
                    research_mfe REAL,
                    research_mae REAL,
                    time_to_confirmation_bars INTEGER,
                    time_to_entry_ready_bars INTEGER,
                    location_type TEXT,
                    confirmation_event_type TEXT,
                    confirmation_displacement_label TEXT,
                    execution_event_type TEXT,
                    execution_area_source TEXT,
                    evidence_json TEXT,
                    transition_history_json TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_smc_shadow_signals_run ON smc_shadow_entry_ready_signals(run_id)"
            )
            conn.commit()

    def record_run(
        self, *, kind: str, ticker: str, direction: Optional[str], market: Optional[str],
        strategy_version: str, harness_version: str, episodes: list[dict[str, Any]], signals: list[dict[str, Any]],
    ) -> str:
        """Persists one run_walk_forward_evaluation() output. Returns the
        generated run_id. `kind` is "historical" or "paper" -- purely
        descriptive metadata, both paths use the identical evaluator and
        schema (see smc_shadow_evaluation.py's own module docstring on
        why historical and paper are the same computation)."""
        run_id = uuid.uuid4().hex
        now = _utc_now_iso()
        # Reuses the harness's own aggregate_evaluation_report() for the
        # terminal_state/max_progression/missed/invalidated counts rather
        # than recomputing that logic a second time here -- one
        # definition of "what counts as reaching a state" (in both of
        # Sprint 6.1's formally-distinguished senses) for both the
        # in-memory report path and the persisted-run summary.
        summary = aggregate_evaluation_report([{
            "ticker": ticker, "strategy_version": strategy_version, "episodes": episodes, "signals": signals,
        }])
        setup_stats = summary["setup_statistics"]
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO smc_shadow_evaluation_runs(
                    run_id, kind, ticker, direction, market, strategy_version, harness_version,
                    episode_count, signal_count, missed_due_to_no_pullback_count,
                    invalidated_before_entry_ready_count, incomplete_episode_count,
                    terminal_state_reached_json, max_progression_reached_json,
                    started_at, completed_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (run_id, kind, str(ticker or "").upper(), direction, market, strategy_version, harness_version,
                 len(episodes), len(signals), setup_stats["missed_due_to_no_pullback_count"],
                 setup_stats["invalidated_before_entry_ready_count"], setup_stats["incomplete_episode_count"],
                 json.dumps(setup_stats["terminal_state_reached_at_least"]),
                 json.dumps(setup_stats["max_progression_reached_at_least"]), now, now),
            )
            for s in signals:
                conn.execute(
                    """
                    INSERT INTO smc_shadow_entry_ready_signals(
                        run_id, ticker, direction, market, entry_ready_bar_time, assumed_entry_price, stop,
                        assumed_target, assumed_target_r_multiple, assumed_risk,
                        direction_provenance, entry_ready_bar_time_provenance, stop_provenance,
                        assumed_entry_price_provenance, assumed_target_provenance, trade_performance_status,
                        post_entry_ready_invalidation_touched, post_entry_ready_invalidation_touched_bar_time,
                        post_entry_ready_invalidation_fill_type, bars_from_entry_ready_to_invalidation_touch,
                        post_entry_ready_target_assumption_touched, post_entry_ready_target_assumption_touched_bar_time,
                        post_entry_ready_target_assumption_fill_type, bars_from_entry_ready_to_target_assumption_touch,
                        ambiguous_intrabar_between_invalidation_and_target_assumption, ambiguous_intrabar_bar_time,
                        research_mfe, research_mae, time_to_confirmation_bars, time_to_entry_ready_bars,
                        location_type, confirmation_event_type, confirmation_displacement_label,
                        execution_event_type, execution_area_source, evidence_json,
                        transition_history_json, created_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        run_id, str(s.get("ticker") or ticker or "").upper(), s.get("direction"), s.get("market"),
                        s.get("entry_ready_bar_time"), s.get("assumed_entry_price"), s.get("stop"),
                        s.get("assumed_target"), s.get("assumed_target_r_multiple"), s.get("assumed_risk"),
                        s.get("direction_provenance"), s.get("entry_ready_bar_time_provenance"), s.get("stop_provenance"),
                        s.get("assumed_entry_price_provenance"), s.get("assumed_target_provenance"),
                        s.get("trade_performance_status"),
                        s.get("post_entry_ready_invalidation_touched"),
                        s.get("post_entry_ready_invalidation_touched_bar_time"),
                        s.get("post_entry_ready_invalidation_fill_type"),
                        s.get("bars_from_entry_ready_to_invalidation_touch"),
                        s.get("post_entry_ready_target_assumption_touched"),
                        s.get("post_entry_ready_target_assumption_touched_bar_time"),
                        s.get("post_entry_ready_target_assumption_fill_type"),
                        s.get("bars_from_entry_ready_to_target_assumption_touch"),
                        s.get("ambiguous_intrabar_between_invalidation_and_target_assumption"),
                        s.get("ambiguous_intrabar_bar_time"),
                        s.get("research_mfe"), s.get("research_mae"),
                        s.get("time_to_confirmation_bars"), s.get("time_to_entry_ready_bars"),
                        s.get("location_type"), s.get("confirmation_event_type"), s.get("confirmation_displacement_label"),
                        s.get("execution_event_type"), s.get("execution_area_source"),
                        json.dumps(s.get("evidence"), default=str), json.dumps(s.get("transition_history"), default=str),
                        now,
                    ),
                )
            conn.commit()
        return run_id

    def list_runs(self, kind: Optional[str] = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if kind:
                rows = conn.execute("SELECT * FROM smc_shadow_evaluation_runs WHERE kind=? ORDER BY started_at DESC", (kind,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM smc_shadow_evaluation_runs ORDER BY started_at DESC").fetchall()
            return [dict(r) for r in rows]

    def list_signals(self, run_id: Optional[str] = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if run_id:
                rows = conn.execute(
                    "SELECT * FROM smc_shadow_entry_ready_signals WHERE run_id=? ORDER BY entry_ready_bar_time ASC", (run_id,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM smc_shadow_entry_ready_signals ORDER BY entry_ready_bar_time ASC"
                ).fetchall()
            return [_normalize_signal_row(dict(r)) for r in rows]

    def report(self, kind: Optional[str] = None) -> dict[str, Any]:
        """Rebuilds the full 3-layer aggregate report (same shape as
        smc_shadow_evaluation.aggregate_evaluation_report) from PERSISTED
        data alone -- setup_statistics summed from each run's own stored
        summary (computed once, at record_run() time, from the real
        episodes), post_signal_market_behavior segments rebuilt fresh from
        every stored signal row, trade_performance ALWAYS the same fixed
        TRADE_PERFORMANCE_UNAVAILABLE dict. Never re-evaluates anything --
        pure aggregation over what is already on disk. No code path here
        computes a win_rate or expectancy for smc_shadow_v1."""
        from smc_shadow_evaluation import SEGMENTATION_DIMENSIONS, _post_signal_market_behavior

        runs = self.list_runs(kind=kind)
        signals = self.list_signals()
        if kind:
            run_ids = {r["run_id"] for r in runs}
            signals = [s for s in signals if s["run_id"] in run_ids]

        terminal_state_reached = {}
        max_progression_reached = {}
        for r in runs:
            for state, count in json.loads(r["terminal_state_reached_json"] or "{}").items():
                terminal_state_reached[state] = terminal_state_reached.get(state, 0) + count
            for state, count in json.loads(r["max_progression_reached_json"] or "{}").items():
                max_progression_reached[state] = max_progression_reached.get(state, 0) + count

        segments: dict[str, dict[str, Any]] = {}
        for dim in SEGMENTATION_DIMENSIONS:
            buckets: dict[Any, list[dict[str, Any]]] = {}
            for s in signals:
                buckets.setdefault(s.get(dim), []).append(s)
            segments[dim] = {
                str(k): {
                    "post_signal_market_behavior": _post_signal_market_behavior(v),
                    "trade_performance": dict(TRADE_PERFORMANCE_UNAVAILABLE),
                }
                for k, v in buckets.items()
            }

        return {
            "runs_included": len(runs),
            "tickers_evaluated": sorted({r["ticker"] for r in runs}),
            "setup_statistics": {
                "detected": sum(r["episode_count"] for r in runs),
                "terminal_state_reached_at_least": terminal_state_reached,
                "max_progression_reached_at_least": max_progression_reached,
                "entry_ready_signal_count": sum(r["signal_count"] for r in runs),
                "invalidated_before_entry_ready_count": sum(r["invalidated_before_entry_ready_count"] for r in runs),
                "missed_due_to_no_pullback_count": sum(r["missed_due_to_no_pullback_count"] for r in runs),
                "incomplete_episode_count": sum(r["incomplete_episode_count"] for r in runs),
            },
            "post_signal_market_behavior": _post_signal_market_behavior(signals),
            "trade_performance": dict(TRADE_PERFORMANCE_UNAVAILABLE),
            "segments": segments,
        }

    def get_signal_evidence(self, signal_id: int) -> Optional[dict[str, Any]]:
        """Full traceability lookup: the exact evaluate_smc_shadow() result
        and transition history behind one specific entry-ready signal."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM smc_shadow_entry_ready_signals WHERE id=?", (signal_id,)).fetchone()
        if not row:
            return None
        record = _normalize_signal_row(dict(row))
        record["evidence"] = json.loads(record.pop("evidence_json") or "null")
        record["transition_history"] = json.loads(record.pop("transition_history_json") or "null")
        return record
