"""Kairos Sprint 4 -- Shadow SMC Strategy/State Engine: research endpoints
and orchestration (2026-09 session).

Wires smc_shadow_engine.evaluate_smc_shadow() (pure) and
smc_shadow_store.SQLiteSmcShadowRepository (isolated persistence) to real
bar data and to the existing production lifecycle, entirely behind the
KAIROS_SMC_SHADOW_ENABLED feature flag -- with the flag off (the default),
every endpoint here reports itself disabled and does no work at all, and
main.py never registers the periodic tick in the first place.

Guardrails enforced structurally, not just by comment:
  - No import of, or call into, any alert/notification/journal-write
    function anywhere in this file.
  - The only WRITE this file performs is via SQLiteSmcShadowRepository,
    which opens its own separate database file (see smc_shadow_store.py) --
    this file never calls candidates_router._get_db() for anything other
    than a plain SELECT to snapshot the current production state for
    comparison.
  - Never touches dashboard_state.py or candidates_router.py's
    /candidates/dashboard-state read model -- the Erica/Sprint-3 dashboards
    are unaware this module exists.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, Cookie, Header, HTTPException
from pydantic import BaseModel

import scanner
from market_data import AlpacaMarketDataProvider
from smc_shadow_engine import (
    STRATEGY_VERSION,
    SHADOW_STATES,
    SmcShadowConfig,
    evaluate_smc_shadow,
    smc_shadow_enabled,
)
from smc_shadow_store import SQLiteSmcShadowRepository, default_smc_shadow_db_path
from smc_shadow_evaluation import run_walk_forward_evaluation, HARNESS_VERSION
from smc_shadow_evaluation_store import SQLiteSmcShadowEvaluationRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/scanner", tags=["smc-shadow-research"])

_repository: Optional[SQLiteSmcShadowRepository] = None
_evaluation_repository: Optional[SQLiteSmcShadowEvaluationRepository] = None


def _repo() -> SQLiteSmcShadowRepository:
    global _repository
    if _repository is None:
        _repository = SQLiteSmcShadowRepository(default_smc_shadow_db_path())
    return _repository


def _evaluation_repo() -> SQLiteSmcShadowEvaluationRepository:
    global _evaluation_repository
    if _evaluation_repository is None:
        _evaluation_repository = SQLiteSmcShadowEvaluationRepository(default_smc_shadow_db_path())
    return _evaluation_repository


def _disabled_response() -> dict[str, Any]:
    return {
        "enabled": False,
        "strategy_version": STRATEGY_VERSION,
        "message": (
            "The shadow SMC engine is disabled (KAIROS_SMC_SHADOW_ENABLED "
            "is not set). This is a research-only, feature-flagged engine "
            "-- it never runs, persists, or is reachable unless explicitly "
            "enabled."
        ),
    }


# ---------------------------------------------------------------------------
# Auth -- same convention as candidates_router.py's scanner API key, so a
# researcher already signed in to the dashboards can use these endpoints
# without a separate credential. Imported lazily (inside functions) to
# avoid a hard import-time dependency between the two router modules.
# ---------------------------------------------------------------------------

def _check_auth(x_api_key: Optional[str], scanner_session: Optional[str]) -> None:
    import candidates_router
    candidates_router._check_api_key(x_api_key, scanner_session)


def _scanner_session_cookie_name() -> str:
    import candidates_router
    return candidates_router.SCANNER_SESSION_COOKIE


# ---------------------------------------------------------------------------
# Live bar fetching -- deliberately isolated in its own small function so
# tests can monkeypatch it (same convention as candidates_router.py's own
# _batch_download/_recent_4h_bars_for_execution_shadow mocking). Failures
# degrade to None (never raise) -- evaluate_smc_shadow already treats a
# missing frame as "unresolved", not a crash.
# ---------------------------------------------------------------------------

_INTERVAL_PERIODS = {"4h": "60d", "30m": "10d", "5m": "3d"}

# Sprint 5: a longer, disclosed lookback for HISTORICAL walk-forward
# evaluation specifically -- the live tick's own periods above stay
# unchanged (a rolling "current state" snapshot doesn't need months of
# history). These are provider-availability-limited, not strategy
# choices; a provider that can't return this much history for a given
# ticker simply yields fewer usable bars, handled the same "None ->
# unresolved, never fabricated" way every other missing-data case is.
_HISTORICAL_INTERVAL_PERIODS = {"4h": "2y", "30m": "60d", "5m": "10d"}


def fetch_smc_shadow_bars(ticker: str, interval: str, *, historical: bool = False) -> Optional[pd.DataFrame]:
    periods = _HISTORICAL_INTERVAL_PERIODS if historical else _INTERVAL_PERIODS
    period = periods.get(interval, "10d")
    try:
        raw = AlpacaMarketDataProvider().download([ticker], period=period, interval=interval, auto_adjust=True)
    except Exception as exc:
        logger.warning("[smc_shadow] bar fetch failed ticker=%s interval=%s error=%s", ticker, interval, exc)
        return None
    if raw is None or raw.empty:
        return None
    try:
        df = scanner._flatten_columns(raw.copy()).dropna()
    except Exception:
        return None
    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(set(df.columns)):
        return None
    return df


def fetch_smc_shadow_frames(ticker: str, *, historical: bool = False) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    return (
        fetch_smc_shadow_bars(ticker, "4h", historical=historical),
        fetch_smc_shadow_bars(ticker, "30m", historical=historical),
        fetch_smc_shadow_bars(ticker, "5m", historical=historical),
    )


# ---------------------------------------------------------------------------
# Production snapshot -- READ-ONLY. The only touch this file has on
# candidates.db at all.
# ---------------------------------------------------------------------------

def _production_snapshot(conn, setup_key: str) -> dict[str, Optional[str]]:
    import dashboard_state
    row = conn.execute(
        """
        SELECT s.state AS monitor_state, s.entry_reached_at AS entry_reached_at
        FROM approved_setup_monitor_state s
        WHERE s.setup_key = ?
        ORDER BY s.updated_at DESC LIMIT 1
        """,
        (setup_key,),
    ).fetchone()
    if not row:
        return {"production_state": None, "production_legacy_state": None}
    mapped = dashboard_state.map_approved_monitor_state(row["monitor_state"], row["entry_reached_at"])
    return {"production_state": mapped, "production_legacy_state": row["monitor_state"]}


def _active_production_setups(conn) -> list[dict[str, Any]]:
    """Every non-terminal (WITHDRAWN/SUPERSEDED-excluded) approved setup --
    the same population Sprint 1's dashboard-state read model considers
    active, reused here rather than redefined, so "which setups does the
    shadow engine evaluate" never silently drifts from "which setups the
    production dashboard considers active"."""
    import candidates_router
    rows = candidates_router._dashboard_rows_for_approved_monitor(conn)
    return [
        {"setup_key": r["setup_key"], "ticker": r["ticker"], "direction": r["direction"]}
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Orchestration -- the periodic research tick. Registered in main.py ONLY
# when smc_shadow_enabled() is true at startup; never scheduled otherwise.
# ---------------------------------------------------------------------------

def run_smc_shadow_tick(config: Optional[SmcShadowConfig] = None) -> dict[str, Any]:
    if not smc_shadow_enabled():
        return {"ran": False, "reason": "disabled"}
    import candidates_router
    conn = candidates_router._get_db()
    try:
        setups = _active_production_setups(conn)
        evaluated = 0
        for setup in setups:
            df_4h, df_30m, df_5m = fetch_smc_shadow_frames(setup["ticker"])
            evaluation = evaluate_smc_shadow(setup["ticker"], setup["direction"], df_4h, df_30m, df_5m, config=config)
            snapshot = _production_snapshot(conn, setup["setup_key"])
            _repo().record_evaluation(
                setup_key=setup["setup_key"],
                ticker=setup["ticker"],
                direction=setup["direction"],
                evaluation=evaluation,
                production_state=snapshot["production_state"],
                production_legacy_state=snapshot["production_legacy_state"],
            )
            evaluated += 1
        return {"ran": True, "evaluated": evaluated}
    finally:
        conn.close()


def safe_run_smc_shadow_tick(reason: str = "periodic") -> None:
    try:
        result = run_smc_shadow_tick()
        if result.get("ran"):
            logger.info("[smc_shadow] tick complete reason=%s evaluated=%s", reason, result.get("evaluated"))
    except Exception as exc:
        logger.warning("[smc_shadow] tick failed reason=%s error=%s", reason, exc)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/candidates/shadow-smc/status")
def shadow_smc_status():
    """Always reachable, regardless of the feature flag -- lets a
    researcher (or a test) confirm whether the engine is on without
    needing any data."""
    return {
        "enabled": smc_shadow_enabled(),
        "strategy_version": STRATEGY_VERSION,
        "states": list(SHADOW_STATES),
    }


@router.get("/candidates/shadow-smc/state")
def list_shadow_smc_state(
    x_api_key: Optional[str] = Header(default=None),
):
    """Every setup this engine has evaluated at least once, current shadow
    state alongside its production snapshot AT THE TIME OF THAT EVALUATION
    -- the comparison view this sprint's "compare production vs shadow
    decisions" requirement asks for."""
    scanner_session = None  # cookie auth intentionally not required for a research/debug view; API key only
    if not smc_shadow_enabled():
        return _disabled_response()
    _check_auth(x_api_key, scanner_session)
    rows = _repo().list_current()
    agreements = 0
    disagreements = 0
    for row in rows:
        if row.get("production_state") is not None:
            if (row["state"] == "ENTRY_READY") == (row.get("production_state") == "ENTRY_READY"):
                agreements += 1
            else:
                disagreements += 1
    return {
        "enabled": True,
        "strategy_version": STRATEGY_VERSION,
        "disclaimer": (
            "Research only. This shadow state has never influenced, and "
            "cannot influence, production ENTER_NOW, any dashboard row "
            "color, journal/position behavior, or any live alert."
        ),
        "count": len(rows),
        "entry_ready_agreement_count": agreements,
        "entry_ready_disagreement_count": disagreements,
        "setups": rows,
    }


@router.get("/candidates/shadow-smc/state/{setup_key}")
def get_shadow_smc_state(
    setup_key: str,
    x_api_key: Optional[str] = Header(default=None),
):
    if not smc_shadow_enabled():
        return _disabled_response()
    _check_auth(x_api_key, None)
    current = _repo().get_current(setup_key)
    if current is None:
        raise HTTPException(status_code=404, detail=f"No shadow state recorded for setup_key={setup_key}")
    transitions = _repo().get_transitions(setup_key)
    return {
        "enabled": True,
        "strategy_version": STRATEGY_VERSION,
        "current": current,
        "transitions": transitions,
    }


class ManualEvaluateRequest(BaseModel):
    ticker: str
    direction: str
    setup_key: Optional[str] = None


@router.post("/candidates/shadow-smc/evaluate")
def manual_shadow_smc_evaluate(
    payload: ManualEvaluateRequest,
    x_api_key: Optional[str] = Header(default=None),
):
    """On-demand research evaluation for any ticker -- does not require an
    existing approved_setup_memories row. Useful for testing the engine's
    read of a specific ticker/direction without waiting for the periodic
    tick. Persists under setup_key if given, else a synthetic
    research-only key (never collides with a real approved setup_key,
    which always includes an entry/stop/target in its own format)."""
    if not smc_shadow_enabled():
        return _disabled_response()
    _check_auth(x_api_key, None)
    ticker = str(payload.ticker or "").upper().strip()
    if not ticker:
        raise HTTPException(status_code=400, detail="ticker is required")
    setup_key = payload.setup_key or f"research::{ticker}::{payload.direction}"

    df_4h, df_30m, df_5m = fetch_smc_shadow_frames(ticker)
    evaluation = evaluate_smc_shadow(ticker, payload.direction, df_4h, df_30m, df_5m)

    production_state = None
    production_legacy_state = None
    if payload.setup_key:
        import candidates_router
        conn = candidates_router._get_db()
        try:
            snapshot = _production_snapshot(conn, payload.setup_key)
            production_state = snapshot["production_state"]
            production_legacy_state = snapshot["production_legacy_state"]
        finally:
            conn.close()

    row = _repo().record_evaluation(
        setup_key=setup_key, ticker=ticker, direction=payload.direction,
        evaluation=evaluation, production_state=production_state,
        production_legacy_state=production_legacy_state,
    )
    return {"enabled": True, "strategy_version": STRATEGY_VERSION, "evaluation": evaluation, "persisted": row}


# ---------------------------------------------------------------------------
# Sprint 5 -- Historical + Paper Evaluation Harness.
#
# Both endpoints below call run_walk_forward_evaluation() with its
# default strategy_config (STRATEGY_CONFIGS[STRATEGY_VERSION] --
# smc_shadow_v1 exactly as written, untouched) -- neither endpoint
# accepts a config override from the request. Measuring v1 "exactly as
# written" is enforced here structurally: there is no request field that
# could smuggle a modified threshold in.
# ---------------------------------------------------------------------------

class HistoricalEvaluateRequest(BaseModel):
    ticker: str
    direction: str
    market: str = "stock"
    kind: str = "historical"  # "historical" or "paper" -- descriptive only, see smc_shadow_evaluation_store.py


@router.post("/candidates/shadow-smc/evaluate-historical")
def historical_shadow_smc_evaluate(
    payload: HistoricalEvaluateRequest,
    x_api_key: Optional[str] = Header(default=None),
):
    """Runs the real, unmodified smc_shadow_v1 strategy over a longer
    historical window for one ticker/direction via a genuine no-lookahead
    walk-forward pass (see smc_shadow_evaluation.py's own module
    docstring), persists every resulting episode/entry-ready signal with
    full evidence, and returns the run's own summary. This is the
    concrete "historical evaluation" entry point -- the periodic tick
    (run_smc_shadow_tick) remains the "paper" (live, rolling-window)
    observation path; both ultimately produce data in the same schema
    (see smc_shadow_evaluation_store.py), because both are the same
    computation over different bar windows."""
    if not smc_shadow_enabled():
        return _disabled_response()
    _check_auth(x_api_key, None)
    ticker = str(payload.ticker or "").upper().strip()
    if not ticker:
        raise HTTPException(status_code=400, detail="ticker is required")
    if payload.kind not in ("historical", "paper"):
        raise HTTPException(status_code=400, detail="kind must be 'historical' or 'paper'")

    df_4h, df_30m, df_5m = fetch_smc_shadow_frames(ticker, historical=True)
    if df_30m is None or len(df_30m) == 0:
        return {
            "enabled": True, "strategy_version": STRATEGY_VERSION, "harness_version": HARNESS_VERSION,
            "ran": False, "reason": "no 30M historical data available for this ticker",
        }

    run = run_walk_forward_evaluation(ticker, payload.direction, df_4h, df_30m, df_5m, market=payload.market)
    run_id = _evaluation_repo().record_run(
        kind=payload.kind, ticker=ticker, direction=payload.direction, market=payload.market,
        strategy_version=run["strategy_version"], harness_version=run["harness_version"],
        episodes=run["episodes"], signals=run["signals"],
    )
    return {
        "enabled": True,
        "strategy_version": run["strategy_version"],
        "harness_version": run["harness_version"],
        "ran": True,
        "run_id": run_id,
        "episode_count": len(run["episodes"]),
        "signal_count": len(run["signals"]),
    }


@router.get("/candidates/shadow-smc/evaluation-report")
def shadow_smc_evaluation_report(
    kind: Optional[str] = None,
    x_api_key: Optional[str] = Header(default=None),
):
    """The required outputs, aggregated across every persisted evaluation
    run (historical and/or paper), as three explicitly separate layers
    (Sprint 5.2 -- see smc_shadow_evaluation.py's own module docstring):
    setup_statistics (setups detected, counts reaching each state,
    ENTRY_READY signal count, missed-due-to-no-pullback count,
    invalidated-before-entry-ready count), post_signal_market_behavior
    (invalidation-touched / assumed-target-touched rates, research MFE/
    MAE, same-bar ambiguity, gap counts -- segmented by every required
    dimension: market, direction, HTF location type, 30M event type,
    displacement strength, 5M execution type, and whether the execution
    area came from the 30M confirmation candle vs. a 5M displacement
    fallback), and trade_performance (ALWAYS unavailable/unresolved for
    smc_shadow_v1 -- there is no win rate or expectancy anywhere in this
    response). Pure aggregation over already-persisted data -- never
    re-evaluates anything, never touches a threshold."""
    if not smc_shadow_enabled():
        return _disabled_response()
    _check_auth(x_api_key, None)
    if kind is not None and kind not in ("historical", "paper"):
        raise HTTPException(status_code=400, detail="kind must be 'historical' or 'paper'")
    report = _evaluation_repo().report(kind=kind)
    return {
        "enabled": True,
        "strategy_version": STRATEGY_VERSION,
        "harness_version": HARNESS_VERSION,
        "disclaimer": (
            "Research only. Measures strategy_version=smc_shadow_v1 exactly as written -- no threshold "
            "in this report was tuned or optimized to produce it. A weak-looking rule here is a candidate "
            "for a future strategy_version (e.g. smc_shadow_v1_1), never a silent edit to v1 itself. This "
            "shadow evaluation has never influenced, and cannot influence, production ENTER_NOW, any "
            "dashboard row color, journal/position behavior, or any live alert."
        ),
        **report,
    }
