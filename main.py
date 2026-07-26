from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from datetime import datetime, timedelta, timezone
import logging
import os
import threading
import time
from typing import Optional

from fastapi import Body, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from scanner import (
    analysis_cache_snapshot,
    analysis_cache_status,
    build_bos_displacement_shadow_report,
    _batch_download,
    scan_cached,
    scan_ticker,
    debug_ticker,
    scan_trends,
    WATCHLIST,
    coverage_baseline_snapshot,
    register_background_periodic_task,
    start_market_cache_refresh,
)
from discovery import build_ranked_discovery_universe
from market_data import (
    ALPACA_PROVIDER_NAME,
    YAHOO_PROVIDER_NAME,
    alpaca_credentials_configured,
    comparison_diagnostics,
    configured_provider_name,
    configured_provider_profile_name,
    configured_timeframe_provider_profile,
    hybrid_strategy_diagnostics,
    validate_watchlist_candles,
)
from journal_store import (
    JournalConflictError,
    JournalValidationError,
    SQLiteJournalRepository,
    default_journal_db_path,
)
from market_data import build_market_data_provider, provider_name_for_timeframe
from position_intelligence import (
    aggregate_replays,
    classify_journal_replay_parity,
    evidence_guard,
    evidence_log_from_replays,
    real_evidence_counts,
    replay_position_intelligence,
    replay_readiness,
)
from verified_analytics import analytics_verification, verified_analytics_snapshot
from smart_notifications import SQLiteNotificationRepository, SMART_NOTIFICATION_VERSION, stable_event_id
from trade_intelligence import (
    TRADE_INTELLIGENCE_VERSION,
    build_trade_intelligence_snapshot,
    build_verified_trade_records,
    similar_trade_insight,
    trade_intelligence_eligibility_funnel,
)
from verified_history import (
    MAX_REPLAY_ATTEMPTS,
    REPLAY_JOB_VERSION,
    VERIFIED_HISTORY_PIPELINE_VERSION,
    backfill_preview,
    build_verified_history_snapshot,
    classify_pipeline_record,
    completion_readiness,
    is_open_entry,
    replay_dedupe_key,
    replay_input_signature,
    retryable_error,
    verification_to_pipeline_status,
)
from verified_history_store import SQLiteVerifiedHistoryRepository

app = FastAPI(title="Stock Options Scanner")
logger = logging.getLogger(__name__)

DISCOVERY_UNIVERSE_TTL_SECONDS = 24 * 60 * 60
_discovery_universe_cache = {
    "symbols": [],
    "generated_at": None,
    "expires_at": None,
    "pipeline_counts": {},
    "thresholds": {},
    "formula": {},
    "stage3": {},
    "stage4": {},
    "top_20": [],
    "bottom_20_selected": [],
    "watchlist_overlap": {},
    "last_error": None,
    "last_duration": None,
    "job_id": None,
    "running": False,
    "started_at": None,
    "completed_at": None,
    "metrics": {},
}
_discovery_universe_lock = threading.RLock()
_discovery_universe_executor = ThreadPoolExecutor(max_workers=1)
_journal_repository = SQLiteJournalRepository(default_journal_db_path())
_notification_repository = SQLiteNotificationRepository(default_journal_db_path())
_verified_history_repository = SQLiteVerifiedHistoryRepository(default_journal_db_path())
_trade_intelligence_cache = {
    "signature": None,
    "snapshot": None,
    "verified_records": [],
}
_trade_intelligence_lock = threading.RLock()
_verified_history_cache = {
    "signature": None,
    "snapshot": None,
}
_verified_history_lock = threading.RLock()

NO_STORE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}


def _format_timestamp(value):
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).replace(tzinfo=None).isoformat() + "Z"
    return None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_utc_datetime(value):
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _discovery_admin_token() -> str:
    return os.getenv("DISCOVERY_ADMIN_TOKEN", "").strip()


def _journal_admin_token() -> str:
    return os.getenv("JOURNAL_ADMIN_TOKEN", "").strip() or _discovery_admin_token()


def _discovery_status_snapshot() -> dict:
    now = _utc_now()
    with _discovery_universe_lock:
        cached = dict(_discovery_universe_cache)
    generated_at = _coerce_utc_datetime(cached.get("generated_at"))
    expires_at = _coerce_utc_datetime(cached.get("expires_at"))
    started_at = _coerce_utc_datetime(cached.get("started_at"))
    completed_at = _coerce_utc_datetime(cached.get("completed_at"))
    age_seconds = None
    if isinstance(generated_at, datetime):
        age_seconds = max(0, round((now - generated_at).total_seconds(), 1))
    expires_in_seconds = None
    if isinstance(expires_at, datetime):
        expires_in_seconds = round((expires_at - now).total_seconds(), 1)
    stale = not generated_at or (isinstance(expires_at, datetime) and expires_at <= now)
    status = "refreshing" if cached.get("running") else "ready" if cached.get("symbols") else "warming"
    if cached.get("last_error") and not cached.get("symbols") and not cached.get("running"):
        status = "error"
    return {
        "status": status,
        "enabled": bool(_discovery_admin_token()),
        "running": bool(cached.get("running")),
        "job_id": cached.get("job_id"),
        "started_at": _format_timestamp(started_at),
        "completed_at": _format_timestamp(completed_at),
        "generated_at": _format_timestamp(generated_at),
        "expires_at": _format_timestamp(expires_at),
        "age_seconds": age_seconds,
        "expires_in_seconds": expires_in_seconds,
        "stale": bool(stale),
        "has_cache": bool(cached.get("symbols")),
        "selected_count": len(cached.get("symbols") or []),
        "pipeline_counts": cached.get("pipeline_counts") or {},
        "thresholds": cached.get("thresholds") or {},
        "formula": cached.get("formula") or {},
        "stage3": cached.get("stage3") or {},
        "stage4": cached.get("stage4") or {},
        "top_20": cached.get("top_20") or [],
        "bottom_20_selected": cached.get("bottom_20_selected") or [],
        "watchlist_overlap": cached.get("watchlist_overlap") or {},
        "last_error": cached.get("last_error"),
        "last_duration": cached.get("last_duration"),
        "metrics": cached.get("metrics") or {},
    }


def _top_reasons(reason_counts: dict, limit: int = 8) -> dict:
    if not isinstance(reason_counts, dict):
        return {}
    items = sorted(reason_counts.items(), key=lambda item: (-int(item[1] or 0), str(item[0])))
    return {str(reason): int(count or 0) for reason, count in items[:limit]}


def _discovery_metrics_from_result(result: dict, started_at: datetime, completed_at: datetime, duration_ms: float) -> dict:
    result = result or {}
    counts = result.get("pipeline_counts") or {}
    thresholds = result.get("thresholds") or {}
    stage3 = result.get("stage3") or {}
    stage4 = result.get("stage4") or {}
    stage3_failures = stage3.get("failure_reasons") or {}
    stage4_failures = stage4.get("failure_reasons") or {}
    cap = thresholds.get("target_universe_size")
    return {
        "discovery_started_at": _format_timestamp(started_at),
        "discovery_completed_at": _format_timestamp(completed_at),
        "discovery_duration_ms": duration_ms,
        "raw_alpaca_asset_count": counts.get("raw_assets"),
        "tradable_optionable_count": counts.get("tradable_optionable"),
        "hygiene_passed_count": counts.get("hygiene_passed"),
        "dollar_volume_passed_count": counts.get("dollar_volume_passed"),
        "options_liquidity_passed_count": counts.get("options_liquidity_passed"),
        "ranked_universe_count_before_cap": counts.get("ranked"),
        "final_admitted_symbol_count": counts.get("selected"),
        "configured_cap": cap,
        "effective_cap": cap,
        "failure_count_by_stage": {
            "stage3_dollar_volume": sum(int(value or 0) for value in stage3_failures.values()) + int(stage3.get("fetch_failures") or 0),
            "stage4_options_liquidity": sum(int(value or 0) for value in stage4_failures.values()),
        },
        "top_blocker_or_failure_reasons": {
            "stage3_dollar_volume": _top_reasons(stage3_failures),
            "stage4_options_liquidity": _top_reasons(stage4_failures),
        },
    }


def _discovery_coverage_context() -> dict:
    with _discovery_universe_lock:
        cached = dict(_discovery_universe_cache)
    return {
        "universe_source": "discovered",
        "universe_generated_at": cached.get("generated_at"),
        "universe_symbol_count": len(cached.get("symbols") or []),
        "discovery": cached.get("metrics") or {},
    }


def _run_discovery_universe_job(job_id: str) -> None:
    started = time.perf_counter()
    started_at = _utc_now()
    try:
        result = build_ranked_discovery_universe(static_watchlist=WATCHLIST)
        now = _utc_now()
        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        metrics = _discovery_metrics_from_result(result, started_at, now, duration_ms)
        with _discovery_universe_lock:
            _discovery_universe_cache.update({
                "symbols": result.get("symbols") or [],
                "generated_at": now,
                "expires_at": now + timedelta(seconds=DISCOVERY_UNIVERSE_TTL_SECONDS),
                "pipeline_counts": result.get("pipeline_counts") or {},
                "thresholds": result.get("thresholds") or {},
                "formula": result.get("formula") or {},
                "stage3": result.get("stage3") or {},
                "stage4": result.get("stage4") or {},
                "top_20": result.get("top_20") or [],
                "bottom_20_selected": result.get("bottom_20_selected") or [],
                "watchlist_overlap": result.get("watchlist_overlap") or {},
                "last_error": None,
                "last_duration": round(duration_ms / 1000, 1),
                "job_id": job_id,
                "running": False,
                "started_at": None,
                "completed_at": now,
                "metrics": metrics,
            })
        counts = result.get("pipeline_counts") or {}
        logger.info(
            "coverage.discovery.complete duration_ms=%s raw_assets=%s tradable_optionable=%s hygiene_passed=%s dollar_volume_passed=%s options_liquidity_passed=%s ranked_before_cap=%s final_symbols=%s effective_cap=%s",
            duration_ms,
            counts.get("raw_assets"),
            counts.get("tradable_optionable"),
            counts.get("hygiene_passed"),
            counts.get("dollar_volume_passed"),
            counts.get("options_liquidity_passed"),
            counts.get("ranked"),
            counts.get("selected"),
            metrics.get("effective_cap"),
        )
    except Exception as exc:
        with _discovery_universe_lock:
            _discovery_universe_cache.update({
                "last_error": str(exc),
                "last_duration": round(time.perf_counter() - started, 1),
                "job_id": job_id,
                "running": False,
                "started_at": None,
                "completed_at": _utc_now(),
            })


def _submit_discovery_universe_job(force: bool = False) -> tuple[bool, str]:
    with _discovery_universe_lock:
        if _discovery_universe_cache.get("running"):
            return False, "already running"
        if not force and _discovery_universe_cache.get("symbols"):
            expires_at = _coerce_utc_datetime(_discovery_universe_cache.get("expires_at"))
            if isinstance(expires_at, datetime) and expires_at > _utc_now():
                return False, "cache fresh"
        job_id = f"discovery:{int(time.time())}"
        _discovery_universe_cache.update({
            "running": True,
            "started_at": _utc_now(),
            "job_id": job_id,
            "last_error": None,
        })
    try:
        _discovery_universe_executor.submit(_run_discovery_universe_job, job_id)
    except RuntimeError as exc:
        with _discovery_universe_lock:
            _discovery_universe_cache.update({
                "running": False,
                "started_at": None,
                "last_error": str(exc),
            })
        return False, "submit failed"
    return True, job_id


def _require_discovery_admin_token(header_value) -> None:
    token = _discovery_admin_token()
    if not token:
        raise HTTPException(status_code=503, detail="Discovery manual trigger is disabled; set DISCOVERY_ADMIN_TOKEN to enable it")
    if header_value != token:
        raise HTTPException(status_code=403, detail="Invalid discovery admin token")


def _require_journal_admin_token(header_value) -> None:
    token = _journal_admin_token()
    if not token:
        raise HTTPException(status_code=503, detail="Journal API is disabled; set JOURNAL_ADMIN_TOKEN or DISCOVERY_ADMIN_TOKEN to enable it")
    if header_value != token:
        raise HTTPException(status_code=403, detail="Invalid journal admin token")


def _discovery_cache_needs_refresh() -> bool:
    status = _discovery_status_snapshot()
    return not status.get("has_cache") or bool(status.get("stale"))


def _submit_discovery_universe_job_if_needed() -> tuple[bool, str]:
    if not _discovery_cache_needs_refresh():
        return False, "cache fresh"
    return _submit_discovery_universe_job(force=False)


def _register_discovery_background_refresh() -> None:
    register_background_periodic_task(
        "discovery_universe",
        DISCOVERY_UNIVERSE_TTL_SECONDS,
        _submit_discovery_universe_job_if_needed,
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="public"), name="static")


@app.middleware("http")
async def add_no_store_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers.update(NO_STORE_HEADERS)
    return response


@app.on_event("startup")
def startup_market_cache_refresh():
    _register_discovery_background_refresh()
    start_market_cache_refresh()
    _submit_discovery_universe_job_if_needed()


@app.get("/")
def index():
    return FileResponse(
        "public/index.html",
        headers=NO_STORE_HEADERS,
    )


def _discovery_symbols_ready():
    status = _discovery_status_snapshot()
    if not status.get("has_cache") or status.get("stale") or status.get("running"):
        return False, [], status
    with _discovery_universe_lock:
        symbols = list(_discovery_universe_cache.get("symbols") or [])
    return bool(symbols), symbols, status


def _discovery_scan_not_ready_response(status: dict) -> dict:
    return {
        "rows": [],
        "near_miss": [],
        "meta": {
            "cache": "miss",
            "cache_key": "discovered",
            "status": "warming",
            "stale": True,
            "refreshing": bool(status.get("running")),
            "has_cache": bool(status.get("has_cache")),
            "universe": "discovered",
            "configured_universe_count": status.get("selected_count") or 0,
            "symbols_attempted": 0,
            "symbols_successfully_processed": 0,
            "symbols_terminally_evaluated": 0,
            "symbols_with_setup": 0,
            "symbols_without_setup": 0,
            "symbols_intentionally_rejected": 0,
            "symbols_operationally_failed": 0,
            "symbols_not_evaluated": 0,
            "evaluation_coverage": None,
            "evaluation_coverage_percent": None,
            "result_yield": None,
            "result_yield_percent": None,
            "partial_result": None,
            "partial_result_reasons": [],
            "qualified_rows": 0,
            "near_miss_rows": 0,
            "message": "Discovery universe is not ready; trigger /api/discovery/run and wait for /api/discovery/status to become ready.",
            "discovery_status": status,
        },
    }


def _attach_notification_metrics(result: dict) -> dict:
    if not isinstance(result, dict):
        return result
    meta = result.get("meta") or {}
    if meta.get("status") == "warming":
        return result
    rows = [*(result.get("rows") or []), *(result.get("near_miss") or [])]
    try:
        notification_metrics = _notification_repository.evaluate_scan(rows, meta)
    except Exception as exc:
        notification_metrics = {
            "version": "smart-notifications-v1",
            "events_evaluated": len(rows),
            "events_created": 0,
            "events_deduplicated": 0,
            "external_delivery_failures": 0,
            "notification_error": exc.__class__.__name__,
        }
        logger.warning("smart_notifications.evaluate_failed error=%s", exc.__class__.__name__)
    result["meta"] = {
        **meta,
        "smart_notifications": {key: value for key, value in notification_metrics.items() if key != "created_events"},
    }
    return result


def _first_present(*values):
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _notification_price(value):
    try:
        if value in (None, ""):
            return None
        number = float(value)
        return number if number == number else None
    except (TypeError, ValueError):
        return None


def _journal_notification_event(event_type: str, entry: dict, previous_state=None, current_state=None, source_time=None) -> dict:
    symbol = str(entry.get("ticker") or "UNKNOWN").upper()
    direction = str(_first_present(entry.get("direction"), entry.get("actual_option_type"), "") or "").upper()
    position_id = str(_first_present(entry.get("position_id"), entry.get("journal_id"), symbol))
    source_event_time = str(source_time or _first_present(entry.get("updated_at"), entry.get("tracking_completed_at"), _format_timestamp(_utc_now())))
    level = None
    level_name = None
    if event_type == "TP1_REACHED":
        level = _notification_price(_first_present(entry.get("original_tp1"), entry.get("target_price"), entry.get("plannedTp1"), entry.get("tp1")))
        level_name = "TP1"
        title = f"{symbol} reached TP1"
        message = f"{symbol} reached TP1. TP1: {'unavailable' if level is None else f'${level:.2f}'}."
        next_step = "Review the position and follow your management plan."
    elif event_type == "STOP_REACHED":
        level = _notification_price(_first_present(entry.get("original_stop"), entry.get("stop_price"), entry.get("plannedStop")))
        level_name = "Stop Loss"
        title = f"{symbol} crossed the planned stop"
        message = f"{symbol} crossed the planned stop. Stop Loss: {'unavailable' if level is None else f'${level:.2f}'}."
        next_step = "Review the position and journal the outcome."
    else:
        title = f"{symbol} position needs review"
        message = f"{symbol} position status changed: {previous_state or 'Unknown'} to {current_state or 'Unknown'}."
        next_step = "Open Position Intelligence and review the latest evidence."
    dedupe_key = "|".join([symbol, position_id, event_type, source_event_time, str(level if level is not None else current_state or "NA")])
    return {
        "event_id": stable_event_id(dedupe_key),
        "version": SMART_NOTIFICATION_VERSION,
        "symbol": symbol,
        "direction": direction,
        "event_type": event_type,
        "priority": "HIGH" if event_type == "STOP_REACHED" else "MEDIUM",
        "title": title,
        "message": message,
        "next_step": next_step,
        "previous_state": previous_state,
        "current_state": current_state,
        "setup_id": entry.get("setup_id"),
        "position_id": position_id,
        "entity_type": "position",
        "entity_id": position_id,
        "deep_link": f"position:{position_id}",
        "event_time": source_event_time,
        "source_event_time": source_event_time,
        "detected_at": _format_timestamp(_utc_now()),
        "current_price": _notification_price(_first_present(entry.get("current_price"), entry.get("exit_price"), entry.get("underlying_price_at_signal"))),
        "relevant_level": level,
        "level_name": level_name,
        "grade": _first_present(entry.get("setup_grade"), entry.get("setupGrade"), entry.get("grade")),
        "status": current_state,
        "dedupe_key": dedupe_key,
        "source": "journal-position",
        "delivery_status": "pending",
        "external_delivery_status": "not_configured",
        "metadata": {"journal_id": entry.get("journal_id")},
    }


def _create_journal_notification_events(before: Optional[dict], after: dict) -> dict:
    created = []
    deduped = 0
    before = before or {}
    candidates = []
    if not before.get("first_target_touch_at") and after.get("first_target_touch_at"):
        candidates.append(_journal_notification_event("TP1_REACHED", after, source_time=after.get("first_target_touch_at")))
    if not before.get("first_stop_touch_at") and after.get("first_stop_touch_at"):
        candidates.append(_journal_notification_event("STOP_REACHED", after, source_time=after.get("first_stop_touch_at")))
    previous_state = before.get("position_last_state")
    current_state = after.get("position_last_state")
    if previous_state and current_state and previous_state != current_state and current_state in {"WATCH", "PROTECT", "EXIT"}:
        candidates.append(_journal_notification_event("POSITION_STATUS_CHANGE", after, previous_state, current_state, after.get("last_evaluated_at") or after.get("updated_at")))
    for event in candidates:
        saved, inserted = _notification_repository.create_event(event)
        if inserted and saved:
            created.append(saved)
        else:
            deduped += 1
    return {"events_created": len(created), "events_deduplicated": deduped, "created_events": created}


@app.get("/api/scan")
def api_scan(
    tickers: str = Query(default=""),
    refresh: bool = Query(default=False),
    discover: bool = Query(default=False),
    universe: str = Query(default="discovered"),
):
    """Scan the full watchlist or a custom comma-separated list of tickers."""
    if tickers:
        watchlist = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        return _attach_notification_metrics(scan_cached(watchlist, force_refresh=refresh))

    selected_universe = str(universe or "discovered").strip().lower()
    if discover:
        selected_universe = "finviz"
    if selected_universe == "discovered":
        ready, symbols, status = _discovery_symbols_ready()
        if not ready:
            return _discovery_scan_not_ready_response(status)
        result = scan_cached(
            symbols,
            force_refresh=refresh,
            universe="discovered",
            max_symbols=None,
            coverage_context=_discovery_coverage_context(),
            trusted_options_symbols=set(symbols),
        )
    elif selected_universe == "default":
        result = scan_cached(force_refresh=refresh, discover=False)
    else:
        use_finviz = selected_universe == "finviz"
        result = scan_cached(force_refresh=refresh, discover=use_finviz)
    return _attach_notification_metrics(result)


@app.get("/api/notifications")
def api_notifications(
    unread_only: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    return _notification_repository.list_events(unread_only=unread_only, limit=limit, offset=offset)


@app.post("/api/notifications/{event_id}/read")
def api_notification_mark_read(event_id: str, x_kairos_admin_token: str = Header(default="")):
    _require_journal_admin_token(x_kairos_admin_token)
    event = _notification_repository.mark_read(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Notification not found")
    return event


@app.post("/api/notifications/read-all")
def api_notifications_mark_all_read(x_kairos_admin_token: str = Header(default="")):
    _require_journal_admin_token(x_kairos_admin_token)
    return _notification_repository.mark_all_read()


@app.get("/api/notifications/preferences")
def api_notification_preferences(x_kairos_admin_token: str = Header(default="")):
    _require_journal_admin_token(x_kairos_admin_token)
    return {"version": "smart-notifications-v1", "preferences": _notification_repository.get_preferences()}


@app.patch("/api/notifications/preferences")
def api_notification_preferences_update(
    payload: dict = Body(default_factory=dict),
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    preferences = payload.get("preferences") if isinstance(payload, dict) else {}
    return {"version": "smart-notifications-v1", "preferences": _notification_repository.update_preferences(preferences or {})}


@app.get("/api/dev/smart-notifications/diagnostics")
def api_smart_notification_diagnostics(x_kairos_admin_token: str = Header(default="")):
    _require_journal_admin_token(x_kairos_admin_token)
    return _notification_repository.diagnostics()


@app.get("/api/scan/{ticker}")
def api_scan_single(ticker: str):
    result = scan_ticker(ticker)
    if result is None:
        return {"setup": None, "reason": "No valid setup found"}
    return result


@app.get("/api/debug/{ticker}")
def api_debug(ticker: str):
    """Return full filter reasoning for a single ticker — every step pass/fail."""
    return debug_ticker(ticker)


@app.get("/api/watchlist")
def api_watchlist():
    return {"watchlist": WATCHLIST}


@app.get("/api/cache/status")
def api_cache_status():
    return analysis_cache_status()


@app.post("/api/discovery/run")
def api_discovery_run(
    refresh: bool = Query(default=False),
    x_kairos_admin_token: str = Header(default=""),
):
    _require_discovery_admin_token(x_kairos_admin_token)
    accepted, reason_or_job = _submit_discovery_universe_job(force=refresh)
    return {
        "accepted": accepted,
        "reason": None if accepted else reason_or_job,
        "job_id": reason_or_job if accepted else None,
        "status": _discovery_status_snapshot(),
    }


@app.get("/api/discovery/status")
def api_discovery_status():
    return _discovery_status_snapshot()


@app.get("/api/discovery/symbols")
def api_discovery_symbols():
    status = _discovery_status_snapshot()
    with _discovery_universe_lock:
        symbols = list(_discovery_universe_cache.get("symbols") or [])
    return {
        "symbols": symbols,
        "count": len(symbols),
        "status": status,
    }


@app.get("/api/coverage/baseline")
def api_coverage_baseline():
    snapshot = coverage_baseline_snapshot()
    if not snapshot:
        return {
            "status": "warming",
            "ready": False,
            "message": "No discovered-universe scan coverage baseline has completed yet.",
            "generated_at": None,
            "discovery": {},
            "scan": {},
            "stage_distribution": {},
            "grade_distribution": {},
            "contract_distribution": {},
            "blocker_distribution": {},
            "provider_failures": {},
            "provider_diagnostics": {},
        }
    return {
        "status": "ready",
        "ready": True,
        **snapshot,
    }


@app.get("/api/journal")
def api_journal_list(
    status: str = Query(default="all"),
    ticker: str = Query(default=""),
    direction: str = Query(default=""),
    position_id: str = Query(default=""),
    limit: int = Query(default=500, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    return {
        "entries": _journal_repository.list_entries({
            "status": status,
            "ticker": ticker,
            "direction": direction,
            "position_id": position_id,
            "limit": limit,
            "offset": offset,
        }),
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/journal/export")
def api_journal_export(x_kairos_admin_token: str = Header(default="")):
    _require_journal_admin_token(x_kairos_admin_token)
    return _journal_repository.export_entries()


@app.get("/api/journal/diagnostics")
def api_journal_diagnostics(x_kairos_admin_token: str = Header(default="")):
    _require_journal_admin_token(x_kairos_admin_token)
    return _journal_repository.diagnostics()


@app.post("/api/journal/backup")
def api_journal_backup(
    keep_latest: int = Query(default=10, ge=1, le=100),
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    return _journal_repository.create_backup(keep_latest=keep_latest)


@app.post("/api/journal/backup/validate")
def api_journal_backup_validate(
    payload: dict = Body(default_factory=dict),
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    backup_path = payload.get("path") if isinstance(payload, dict) else None
    if not backup_path:
        raise HTTPException(status_code=422, detail="Backup path required")
    return _journal_repository.restore_validation(backup_path)


@app.get("/api/journal/{journal_id}")
def api_journal_get(journal_id: str, x_kairos_admin_token: str = Header(default="")):
    _require_journal_admin_token(x_kairos_admin_token)
    entry = _journal_repository.get_entry(journal_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Journal entry not found")
    return entry


@app.post("/api/journal")
def api_journal_create(
    entry: dict = Body(default_factory=dict),
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    try:
        created = _journal_repository.create_entry(entry)
        created["verified_history"] = _maybe_queue_verified_history_job(created, "journal_create")
        return created
    except JournalValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.patch("/api/journal/{journal_id}")
def api_journal_update(
    journal_id: str,
    patch: dict = Body(default_factory=dict),
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    try:
        before = _journal_repository.get_entry(journal_id)
        updated = _journal_repository.update_entry(journal_id, patch)
        updated["verified_history"] = _maybe_queue_verified_history_job(updated, "journal_update")
        notifications = _create_journal_notification_events(before, updated)
        if notifications.get("events_created") or notifications.get("events_deduplicated"):
            updated["smart_notifications"] = {key: value for key, value in notifications.items() if key != "created_events"}
        return updated
    except KeyError:
        raise HTTPException(status_code=404, detail="Journal entry not found")
    except JournalConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except JournalValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.delete("/api/journal/{journal_id}")
def api_journal_delete(journal_id: str, x_kairos_admin_token: str = Header(default="")):
    _require_journal_admin_token(x_kairos_admin_token)
    entry = _journal_repository.delete_entry(journal_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Journal entry not found")
    return {"deleted": True, "journal_id": journal_id, "deleted_at": entry.get("deleted_at")}


@app.post("/api/journal/migrate")
def api_journal_migrate(
    payload: dict = Body(default_factory=dict),
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise HTTPException(status_code=422, detail="Migration payload must include entries list")
    try:
        result = _journal_repository.upsert_entries(entries)
    except JournalValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {
        "accepted": True,
        "created": result["created"],
        "updated": result["updated"],
        "conflicts": result["conflicts"],
        "conflict_ids": result["conflict_ids"],
        "entries": result["entries"],
    }


def _replay_timeframe(position: dict) -> tuple[str, str]:
    timeframe = str(position.get("scanner_timeframe") or position.get("timeframe") or "").strip().upper()
    if timeframe:
        return timeframe, "journaled"
    return "4H", "inferred_default"


def _replay_period_interval(timeframe: str) -> tuple[str, str]:
    normalized = str(timeframe or "").strip().upper()
    if normalized in {"1D", "D", "DAILY"}:
        return "1y", "1d"
    if normalized in {"1W", "W", "WEEKLY"}:
        return "2y", "1wk"
    return "60d", "4h"


def _chart_period_interval(timeframe: str) -> tuple[str, str, str]:
    normalized = str(timeframe or "").strip().upper()
    if normalized in {"1D", "D", "DAILY"}:
        return "1D", "1y", "1d"
    if normalized in {"30M", "30MIN", "30", "30 MIN"}:
        return "30M", "60d", "30m"
    return "4H", "60d", "4h"


def _chart_candle_records(candles, limit: int = 120) -> list[dict]:
    records = []
    if candles is None or not hasattr(candles, "iterrows"):
        return records
    sliced = candles.tail(limit) if hasattr(candles, "tail") else candles
    for ts, row in sliced.iterrows():
        try:
            stamp = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
            if getattr(stamp, "tzinfo", None) is None:
                stamp = stamp.replace(tzinfo=timezone.utc)
            stamp = stamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        except Exception:
            stamp = str(ts)
        record = {
            "timestamp": stamp,
            "open": _safe_float(row.get("Open") if hasattr(row, "get") else None),
            "high": _safe_float(row.get("High") if hasattr(row, "get") else None),
            "low": _safe_float(row.get("Low") if hasattr(row, "get") else None),
            "close": _safe_float(row.get("Close") if hasattr(row, "get") else None),
            "volume": _safe_float(row.get("Volume") if hasattr(row, "get") else None),
        }
        if any(record.get(key) is not None for key in ("open", "high", "low", "close")):
            records.append(record)
    return records


def _safe_float(value):
    try:
        if value is None:
            return None
        numeric = float(value)
        if numeric != numeric:
            return None
        return numeric
    except Exception:
        return None


def _chart_provider_candidates(timeframe: str) -> list[str]:
    configured = provider_name_for_timeframe(timeframe)
    candidates = [configured]
    if ALPACA_PROVIDER_NAME not in candidates:
        candidates.append(ALPACA_PROVIDER_NAME)
    if YAHOO_PROVIDER_NAME not in candidates:
        candidates.append(YAHOO_PROVIDER_NAME)
    return candidates


def _download_chart_candles(ticker: str, period: str, interval: str, limit: int, providers: list[str]) -> tuple[list[dict], dict]:
    attempts = []
    for provider_name in providers:
        attempt_started = time.perf_counter()
        try:
            provider = build_market_data_provider(provider_name)
            candles = provider.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True, group_by="ticker")
            records = _chart_candle_records(candles, limit=limit)
            attempt = {
                "provider": provider.name,
                "status": "ready" if records else "unavailable",
                "candles_loaded": len(records),
                "duration_ms": round((time.perf_counter() - attempt_started) * 1000, 1),
            }
            if not records:
                attempt["failure_reason"] = "no_candles"
            attempts.append(attempt)
            if records:
                return records, {
                    "provider": provider.name,
                    "provider_attempts": attempts,
                    "fallback_used": len(attempts) > 1,
                    "failure_reason": None,
                }
        except Exception as exc:
            attempts.append({
                "provider": provider_name,
                "status": "error",
                "candles_loaded": 0,
                "duration_ms": round((time.perf_counter() - attempt_started) * 1000, 1),
                "failure_reason": exc.__class__.__name__,
            })
    failure_reason = attempts[-1].get("failure_reason") if attempts else "no_provider_attempted"
    return [], {
        "provider": attempts[-1].get("provider") if attempts else None,
        "provider_attempts": attempts,
        "fallback_used": len(attempts) > 1,
        "failure_reason": failure_reason,
    }


def _fetch_replay_candles(position: dict) -> tuple[object, dict]:
    ticker = str(position.get("ticker") or "").strip().upper()
    timeframe, timeframe_source = _replay_timeframe(position)
    period, interval = _replay_period_interval(timeframe)
    provider_name = provider_name_for_timeframe(timeframe)
    provider = build_market_data_provider(provider_name)
    meta = {
        "provider": provider.name,
        "timeframe": timeframe,
        "timeframe_source": timeframe_source,
        "period": period,
        "interval": interval,
        "requested_range": {
            "entry_timestamp": position.get("entry_timestamp") or position.get("tracking_started_at") or position.get("signal_timestamp"),
            "exit_timestamp": position.get("exit_timestamp") or position.get("tracking_completed_at"),
        },
        "candles_requested": None,
        "candles_returned": 0,
        "cache_status": "provider_fetch",
        "fetch_failure": None,
    }
    if not ticker:
        meta["fetch_failure"] = "missing ticker"
        return [], meta
    try:
        candles = provider.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True, group_by="ticker")
        meta["candles_returned"] = int(len(candles)) if hasattr(candles, "__len__") else 0
        return candles, meta
    except Exception as exc:
        meta["fetch_failure"] = exc.__class__.__name__
        return [], meta


def _replay_positions(entries: list[dict], summary_only: bool = False) -> list[dict]:
    results = []
    for entry in entries:
        candles, fetch_meta = _fetch_replay_candles(entry)
        try:
            replay = replay_position_intelligence(
                entry,
                candles,
                provider=fetch_meta["provider"],
                timeframe=fetch_meta["timeframe"],
                timeframe_source=fetch_meta["timeframe_source"],
            )
        except Exception as exc:
            replay = {
                "journal_id": entry.get("journal_id"),
                "position_id": entry.get("position_id"),
                "ticker": str(entry.get("ticker") or "").upper(),
                "data_complete": False,
                "data_gaps": ["replay failure"],
                "provider": fetch_meta.get("provider"),
                "provider_failure": exc.__class__.__name__,
                "final_state": "DATA_NEEDED",
                "outcome_category": "DATA_INCOMPLETE",
                "timeline": [],
            }
        replay["candle_fetch"] = fetch_meta
        replay["readiness"] = replay_readiness(
            entry,
            candles_available=bool(fetch_meta.get("candles_returned")),
            timeframe_source=fetch_meta.get("timeframe_source"),
        )
        if fetch_meta.get("fetch_failure"):
            replay["data_complete"] = False
            replay.setdefault("data_gaps", []).append(f"provider failure: {fetch_meta['fetch_failure']}")
            replay["outcome_category"] = "DATA_INCOMPLETE"
            replay["readiness"] = replay_readiness(entry, candles_available=False, timeframe_source=fetch_meta.get("timeframe_source"))
        replay["journal_replay_parity"] = classify_journal_replay_parity(entry, replay)
        if summary_only:
            replay = {key: value for key, value in replay.items() if key != "timeline"}
        results.append(replay)
    return results


def _readiness_summary(entries: list[dict], replays: list[dict]) -> dict:
    readiness = [replay.get("readiness") or replay_readiness(entry, candles_available=None, timeframe_source=_replay_timeframe(entry)[1]) for entry, replay in zip(entries, replays)]
    statuses = Counter(item.get("status") or "NOT_REPLAYABLE" for item in readiness)
    open_entries = [entry for entry in entries if str(entry.get("result") or entry.get("status") or "Open").lower() == "open"]
    closed_entries = [entry for entry in entries if entry not in open_entries]
    return {
        "total_durable_positions": len(entries),
        "replay_ready": statuses.get("REPLAY_READY", 0),
        "partially_ready": statuses.get("PARTIALLY_READY", 0),
        "not_replayable": statuses.get("NOT_REPLAYABLE", 0),
        "open_positions": len(open_entries),
        "closed_positions": len(closed_entries),
        "positions_with_journaled_timeframe": sum(1 for entry in entries if entry.get("scanner_timeframe") or entry.get("timeframe")),
        "positions_using_inferred_4h": sum(1 for replay in replays if (replay.get("readiness") or {}).get("timeframe_source") == "inferred_default"),
        "positions_missing_recorded_outcome": sum(1 for entry in entries if str(entry.get("result") or entry.get("status") or "Open").lower() != "open" and not (entry.get("result") or entry.get("outcome") or entry.get("completion_reason"))),
        "positions_missing_option_details": sum(1 for item in readiness if any(part.startswith("option ") for part in item.get("missing_optional") or [])),
        "positions": [
            {
                "journal_id": entry.get("journal_id"),
                "position_id": entry.get("position_id"),
                "ticker": entry.get("ticker"),
                "direction": entry.get("direction"),
                "status": item.get("status"),
                "available": item.get("available"),
                "missing_required": item.get("missing_required"),
                "missing_optional": item.get("missing_optional"),
                "invalid": item.get("invalid"),
            }
            for entry, item in zip(entries, readiness)
        ],
    }


def _replay_response(entries: list[dict], replays: list[dict], limit=None, offset=None) -> dict:
    aggregate = aggregate_replays(replays)
    aggregate.update(real_evidence_counts(replays))
    return {
        "status": "ready",
        "ready": True,
        "message": "Replay results are derived diagnostics only. They do not alter journal outcomes or live Position Intelligence history.",
        "replays": replays,
        "aggregate": aggregate,
        "evidence_readiness": _readiness_summary(entries, replays),
        "evidence_log": evidence_log_from_replays(replays),
        "evidence_guard": evidence_guard(aggregate),
        "synthetic_results_included": any(item.get("synthetic") for item in replays),
        **({"limit": limit} if limit is not None else {}),
        **({"offset": offset} if offset is not None else {}),
    }


def _completed_job_replays(jobs: list[dict]) -> list[dict]:
    replays = []
    for job in jobs or []:
        payload = job.get("payload") or {}
        replay = payload.get("replay")
        if isinstance(replay, dict):
            replays.append(replay)
    return replays


def _verified_history_replays(entries: list[dict], jobs: list[dict]) -> list[dict]:
    stored = _completed_job_replays(jobs)
    stored_by_position = {str(replay.get("position_id")): replay for replay in stored}
    missing_entries = [
        entry for entry in entries
        if completion_readiness(entry).get("ready") and str(entry.get("position_id")) not in stored_by_position
    ]
    derived = _replay_positions(missing_entries, summary_only=True) if missing_entries else []
    return [*stored, *derived]


def _verified_history_records_snapshot(force_replay: bool = False) -> dict:
    started = time.perf_counter()
    entries = _journal_repository.list_entries({"status": "all", "limit": 5000, "offset": 0})
    jobs = _verified_history_repository.list_jobs(limit=5000)
    signature = (
        len(entries),
        max([str(entry.get("updated_at") or "") for entry in entries], default=""),
        len(jobs),
        max([str(job.get("updated_at") or "") for job in jobs], default=""),
        bool(force_replay),
    )
    with _verified_history_lock:
        if not force_replay and _verified_history_cache.get("signature") == signature and _verified_history_cache.get("snapshot"):
            cached = dict(_verified_history_cache["snapshot"])
            diagnostics = dict(cached.get("diagnostics") or {})
            diagnostics["cache_status"] = "hit"
            cached["diagnostics"] = diagnostics
            return cached
    replays = _replay_positions(entries, summary_only=True) if force_replay else _verified_history_replays(entries, jobs)
    analytics = verified_analytics_snapshot(entries, replays)
    snapshot = build_verified_history_snapshot(entries, replays, analytics.get("records") or [], jobs)
    snapshot["status"] = "ready"
    snapshot["ready"] = True
    snapshot["message"] = "Verified History tracks how completed journal records move through replay, verification, and Trade Intelligence eligibility."
    diagnostics = dict(snapshot.get("diagnostics") or {})
    diagnostics.update({
        "queue_evaluation_duration_ms": round((time.perf_counter() - started) * 1000, 1),
        "journal_records_inspected": len(entries),
        "replay_records_found": len(replays),
        "verification_records_loaded": len(analytics.get("records") or []),
        "last_successful_worker_run": max([job.get("completed_at") or "" for job in jobs], default=None),
        "last_error": next((job.get("last_error_code") for job in jobs if job.get("last_error_code")), None),
    })
    snapshot["diagnostics"] = diagnostics
    with _verified_history_lock:
        _verified_history_cache.update({"signature": signature, "snapshot": snapshot})
    return snapshot


def _queue_verified_history_job(entry: dict, source: str) -> tuple[Optional[dict], bool]:
    readiness = completion_readiness(entry)
    if not readiness.get("ready"):
        return None, False
    signature = replay_input_signature(entry)
    return _verified_history_repository.create_job_if_absent(
        str(entry.get("journal_id")),
        signature,
        replay_dedupe_key(entry),
        REPLAY_JOB_VERSION,
        {
            "source": source,
            "ticker": entry.get("ticker"),
            "position_id": entry.get("position_id"),
            "readiness": readiness,
        },
    )


def _maybe_queue_verified_history_job(entry: dict, source: str) -> dict:
    job, created = _queue_verified_history_job(entry, source)
    return {
        "queued": bool(created),
        "job_id": (job or {}).get("job_id") if job else None,
        "pipeline_status": (job or {}).get("status") if job else completion_readiness(entry).get("pipeline_status"),
    }


def _create_history_notification(record: dict) -> None:
    status = record.get("pipeline_status")
    if status not in {"NEEDS_REVIEW", "REPLAY_FAILED", "COMPLETION_PENDING"}:
        return
    ticker = record.get("ticker") or "UNKNOWN"
    event_type = {
        "NEEDS_REVIEW": "VERIFIED_HISTORY_NEEDS_REVIEW",
        "REPLAY_FAILED": "VERIFIED_HISTORY_REPLAY_FAILED",
        "COMPLETION_PENDING": "VERIFIED_HISTORY_COMPLETION_PENDING",
    }[status]
    dedupe_key = "|".join([str(record.get("journal_id") or ""), event_type, str(record.get("verification_status") or status)])
    event = {
        "event_id": stable_event_id(dedupe_key),
        "version": VERIFIED_HISTORY_PIPELINE_VERSION,
        "symbol": ticker,
        "direction": None,
        "event_type": event_type,
        "priority": "HIGH" if status in {"NEEDS_REVIEW", "REPLAY_FAILED"} else "MEDIUM",
        "title": f"{ticker} history needs review" if status == "NEEDS_REVIEW" else f"{ticker} history pipeline update",
        "message": record.get("explanation") or "Verified History needs attention.",
        "next_step": record.get("next_step") or "Open Verified History.",
        "previous_state": None,
        "current_state": status,
        "entity_type": "journal",
        "entity_id": record.get("journal_id"),
        "deep_link": f"journal:{record.get('journal_id')}",
        "event_time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_event_time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "detected_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "dedupe_key": dedupe_key,
        "source": "verified-history",
        "delivery_status": "pending",
        "external_delivery_status": "suppressed_by_preference",
        "metadata": {"pipeline_status": status},
    }
    _notification_repository.create_event(event)


def _process_verified_history_jobs(max_jobs: int = 2) -> dict:
    started = time.perf_counter()
    worker_id = f"worker:{os.getpid()}:{int(time.time() * 1000)}"
    processed = []
    failures = []
    for _ in range(max(1, min(int(max_jobs), 10))):
        job_claim_started = time.perf_counter()
        job = _verified_history_repository.claim_next_job(worker_id)
        claim_duration_ms = round((time.perf_counter() - job_claim_started) * 1000, 1)
        if not job:
            break
        entry = _journal_repository.get_entry(str(job.get("journal_id")))
        if not entry:
            failed = _verified_history_repository.fail_job(
                str(job.get("job_id")),
                "missing_journal_record",
                "Journal record no longer exists.",
                retryable=False,
                max_attempts=MAX_REPLAY_ATTEMPTS,
            )
            failures.append(failed)
            continue
        readiness = completion_readiness(entry)
        if not readiness.get("ready"):
            failed = _verified_history_repository.fail_job(
                str(job.get("job_id")),
                "completion_not_ready",
                "Journal record is missing replay-required fields.",
                retryable=False,
                max_attempts=MAX_REPLAY_ATTEMPTS,
            )
            failures.append(failed)
            continue
        try:
            replay_started = time.perf_counter()
            replay = _replay_positions([entry], summary_only=False)[0]
            verification = analytics_verification(entry, replay, verified_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
            pipeline_status = verification_to_pipeline_status(verification, replay)
            completed = _verified_history_repository.complete_job(str(job.get("job_id")), replay, verification, pipeline_status)
            record = classify_pipeline_record(entry, replay=replay, verification=verification, job=completed)
            _create_history_notification(record)
            processed.append({
                "job_id": job.get("job_id"),
                "journal_id": entry.get("journal_id"),
                "ticker": entry.get("ticker"),
                "pipeline_status": pipeline_status,
                "verification_status": verification.get("status"),
                "replay_duration_ms": round((time.perf_counter() - replay_started) * 1000, 1),
                "job_claim_duration_ms": claim_duration_ms,
            })
        except Exception as exc:
            failed = _verified_history_repository.fail_job(
                str(job.get("job_id")),
                exc.__class__.__name__,
                str(exc),
                retryable=retryable_error(exc.__class__.__name__),
                max_attempts=MAX_REPLAY_ATTEMPTS,
            )
            failures.append(failed)
    return {
        "version": VERIFIED_HISTORY_PIPELINE_VERSION,
        "worker_id": worker_id,
        "jobs_processed": len(processed),
        "processed": processed,
        "failures": [item for item in failures if item],
        "worker_duration_ms": round((time.perf_counter() - started) * 1000, 1),
    }


ACTIVE_TRADE_WORKSPACE_VERSION = "active-trade-workspace-v1"


def _number(value):
    return _notification_price(value)


def _trade_direction(entry: dict) -> str:
    direction = str(_first_present(entry.get("direction"), entry.get("actual_option_type"), entry.get("option_type"), entry.get("optionType"), "") or "").upper()
    if direction == "CALL":
        return "LONG"
    if direction == "PUT":
        return "SHORT"
    return direction


def _active_trade_plan(entry: dict) -> dict:
    option_plan = entry.get("option_plan") if isinstance(entry.get("option_plan"), dict) else {}
    return {
        "ticker": str(entry.get("ticker") or "").upper(),
        "direction": _trade_direction(entry),
        "timeframe": _first_present(entry.get("scanner_timeframe"), entry.get("timeframe"), entry.get("setupTf")),
        "grade": _first_present(entry.get("setup_grade"), entry.get("setupGrade"), entry.get("grade")),
        "planned_entry": _number(_first_present(entry.get("planned_underlying_entry"), entry.get("entry_price"), entry.get("entry"))),
        "actual_entry": _number(_first_present(entry.get("actual_underlying_entry"), entry.get("underlying_price_at_entry"))),
        "stop": _number(_first_present(entry.get("original_stop"), entry.get("stop_price"), entry.get("plannedStop"), entry.get("stop"))),
        "tp1": _number(_first_present(entry.get("original_tp1"), entry.get("target_price"), entry.get("plannedTp1"), entry.get("tp1"))),
        "tp2": _number(_first_present(entry.get("original_tp2"), entry.get("plannedTp2"), entry.get("tp2"))),
        "tp3": _number(_first_present(entry.get("original_tp3"), entry.get("plannedTp3"), entry.get("tp3"))),
        "opportunity_remaining": _number(_first_present(entry.get("opportunity_remaining"), entry.get("opportunityRemaining"), entry.get("original_opportunity_remaining_pct"))),
        "initial_rr": _number(_first_present(entry.get("rr"), entry.get("riskReward"), entry.get("reward_risk"))),
        "option_plan": option_plan,
    }


def _active_trade_contract(entry: dict) -> dict:
    return {
        "instrument_type": "option" if _first_present(entry.get("actual_strike"), entry.get("strike_price"), entry.get("actual_expiration"), entry.get("expiration_date")) else "stock_or_underlying",
        "option_type": _first_present(entry.get("actual_option_type"), entry.get("option_type"), entry.get("optionType")),
        "strike": _number(_first_present(entry.get("actual_strike"), entry.get("strike_price"), entry.get("strike"))),
        "expiration": _first_present(entry.get("actual_expiration"), entry.get("expiration_date"), entry.get("expiry")),
        "quantity": _number(_first_present(entry.get("actual_quantity"), entry.get("contracts"))),
        "entry_premium": _number(_first_present(entry.get("actual_option_premium"), entry.get("premium_paid"), entry.get("askAtSelection"))),
        "exit_premium": _number(_first_present(entry.get("actual_exit_premium"), entry.get("exit_option_premium"), entry.get("exit_premium"))),
        "actual_option_pnl": _number(_first_present(entry.get("actual_option_pnl"), entry.get("manual_realized_pnl"), entry.get("realized_pnl"))),
    }


def _active_trade_tracking_state(entry: dict, history_record: Optional[dict] = None) -> str:
    if not is_open_entry(entry):
        status = (history_record or {}).get("pipeline_status")
        if status == "VERIFIED":
            return "VERIFIED"
        if status == "NEEDS_REVIEW":
            return "NEEDS_REVIEW"
        return "COMPLETED"
    if _first_present(entry.get("actual_underlying_entry"), entry.get("first_entry_touch_at"), entry.get("actual_entry_at"), entry.get("position_opened_at")):
        if entry.get("first_stop_touch_at"):
            return "STOP_REACHED"
        if entry.get("first_target_touch_at") or entry.get("position_tp1_reached"):
            return "TP1_REACHED"
        return "ENTERED"
    return "WATCHING_FOR_ENTRY"


def _active_status_guidance(state: str, entry: dict, history_record: Optional[dict] = None) -> dict:
    labels = {
        "WATCHING_FOR_ENTRY": ("Watching for planned entry", "This setup is being tracked, but no position has been recorded.", "Watch for the planned entry and keep the trade plan available."),
        "ENTERED": ("Entered", "The trade is active against the stored plan.", "Monitor progress toward TP1 and the stop."),
        "TP1_REACHED": ("TP1 reached", "The first target has been recorded by existing tracking data.", "Review the position and follow the existing management plan."),
        "STOP_REACHED": ("Stop reached", "The stored stop has been recorded by existing tracking data.", "Review the position and journal completion details."),
        "COMPLETED": ("Completed", "The journal says this trade is complete.", "Wait for replay and verification before treating the result as verified."),
        "VERIFIED": ("Verified", "Journal and replay agree.", "This record may contribute to historical knowledge."),
        "NEEDS_REVIEW": ("Needs Review", "The journaled result and replay result do not match.", "Review the trade record. Kairos will not use it for Trade Intelligence yet."),
    }
    title, happening, next_step = labels.get(state, labels["ENTERED"])
    if history_record and history_record.get("pipeline_status") in {"REPLAY_QUEUED", "REPLAY_RUNNING", "COMPLETED_AWAITING_REPLAY"}:
        title = "Awaiting replay"
        happening = history_record.get("explanation") or happening
        next_step = history_record.get("next_step") or next_step
    return {
        "status": state,
        "label": title,
        "what_is_happening": happening,
        "what_to_watch": "Use the stored entry, stop, targets, and Position Intelligence state. Do not treat journal outcomes as verified until replay agrees.",
        "what_happens_next": next_step,
    }


def _active_trade_timeline(entry: dict, history_record: Optional[dict] = None) -> list[dict]:
    items = [
        ("Setup Found", _first_present(entry.get("signal_timestamp"), entry.get("snapshot_timestamp"), entry.get("created_at")), True),
        ("Confirmation Started", _first_present(entry.get("confirmation_started_at"), entry.get("signal_timestamp")), bool(entry.get("confirmationStarted") or entry.get("confirmation_status"))),
        ("Entry Reached", entry.get("first_entry_touch_at"), bool(entry.get("first_entry_touch_at"))),
        ("Trade Entered", _first_present(entry.get("entry_timestamp"), entry.get("actual_entry_at"), entry.get("position_opened_at")), bool(_first_present(entry.get("actual_underlying_entry"), entry.get("first_entry_touch_at"), entry.get("actual_entry_at"), entry.get("position_opened_at")))),
        ("TP1 Reached", entry.get("first_target_touch_at"), bool(entry.get("first_target_touch_at") or entry.get("position_tp1_reached"))),
        ("Trade Completed", _first_present(entry.get("tracking_completed_at"), entry.get("exit_timestamp")), not is_open_entry(entry)),
        ("Replay", ((history_record or {}).get("job") or {}).get("completed_at"), (history_record or {}).get("pipeline_status") in {"VERIFIED", "NEEDS_REVIEW", "REPLAY_DATA_INCOMPLETE"}),
        ("Verification", _first_present(((history_record or {}).get("verification") or {}).get("verified_at"), ((history_record or {}).get("job") or {}).get("completed_at")), (history_record or {}).get("pipeline_status") in {"VERIFIED", "NEEDS_REVIEW"}),
    ]
    current_seen = False
    timeline = []
    for label, timestamp, complete in items:
        state = "complete" if complete else "future"
        if not complete and not current_seen:
            state = "current"
            current_seen = True
        timeline.append({"label": label, "timestamp": timestamp, "state": state})
    return timeline


def _active_trade_notifications(entry: dict, all_notifications: list[dict]) -> list[dict]:
    ids = {
        str(entry.get("journal_id") or ""),
        str(entry.get("position_id") or ""),
        str(entry.get("id") or ""),
    }
    ticker = str(entry.get("ticker") or "").upper()
    matched = []
    for event in all_notifications or []:
        entity_id = str(event.get("entity_id") or "")
        symbol = str(event.get("symbol") or "").upper()
        if entity_id in ids or (symbol and symbol == ticker and str(event.get("entity_type") or "") in {"position", "journal"}):
            matched.append(event)
    return matched[:10]


def _active_trade_record(entry: dict, history_record: Optional[dict], notifications: list[dict], include_detail: bool = False) -> dict:
    state = _active_trade_tracking_state(entry, history_record)
    plan = _active_trade_plan(entry)
    contract = _active_trade_contract(entry)
    notification_rows = _active_trade_notifications(entry, notifications)
    attention = []
    if state in {"STOP_REACHED", "NEEDS_REVIEW"}:
        attention.append(state)
    if state == "WATCHING_FOR_ENTRY":
        attention.append("TRACKED_NOT_ENTERED")
    if history_record and history_record.get("pipeline_status") in {"COMPLETION_PENDING", "REPLAY_FAILED", "REPLAY_DATA_INCOMPLETE", "NEEDS_REVIEW"}:
        attention.append(history_record.get("pipeline_status"))
    if notification_rows:
        attention.append("HAS_NOTIFICATIONS")
    record = {
        "workspace_version": ACTIVE_TRADE_WORKSPACE_VERSION,
        "id": str(_first_present(entry.get("position_id"), entry.get("journal_id"), entry.get("id"))),
        "journal_id": entry.get("journal_id"),
        "position_id": entry.get("position_id"),
        "ticker": plan["ticker"],
        "direction": plan["direction"],
        "tracking_state": state,
        "status_guidance": _active_status_guidance(state, entry, history_record),
        "grade": plan.get("grade"),
        "timeframe": plan.get("timeframe"),
        "plan": plan,
        "contract": contract,
        "verified_history": history_record,
        "notifications": notification_rows,
        "attention_items": sorted(set(attention)),
        "progress_percent": _number(entry.get("position_max_progress_percent")),
        "entered": state not in {"WATCHING_FOR_ENTRY"},
        "completed": not is_open_entry(entry),
        "trade_intelligence_eligible": bool((history_record or {}).get("trade_intelligence_eligible")),
    }
    if include_detail:
        record.update({
            "journal": entry,
            "guided_chart": {
                "symbol": plan["ticker"],
                "direction": plan["direction"],
                "timeframe": plan.get("timeframe") or "4H",
                "current_price": _number(_first_present(entry.get("current_price"), entry.get("current_quote_price"), entry.get("underlying_price_at_signal"))),
                "planned_entry": plan.get("planned_entry"),
                "actual_entry": plan.get("actual_entry"),
                "stop": plan.get("stop"),
                "targets": [plan.get("tp1"), plan.get("tp2"), plan.get("tp3")],
            },
            "position_intelligence": {
                "version": entry.get("position_intelligence_version"),
                "last_state": entry.get("position_last_state"),
                "best_price": entry.get("position_best_price"),
                "max_progress_percent": entry.get("position_max_progress_percent"),
                "tp1_reached": entry.get("position_tp1_reached"),
                "state_history": entry.get("position_state_history") or [],
            },
            "timeline": _active_trade_timeline(entry, history_record),
        })
    return record


def _active_trades_dataset(include_completed: bool = False) -> dict:
    started = time.perf_counter()
    entries = _journal_repository.list_entries({"status": "all", "limit": 5000, "offset": 0})
    history = _verified_history_records_snapshot(force_replay=False)
    history_by_journal = {str(record.get("journal_id")): record for record in history.get("records") or []}
    notification_payload = _notification_repository.list_events(unread_only=False, limit=200, offset=0)
    notifications = notification_payload.get("events") or []
    active_entries = [entry for entry in entries if is_open_entry(entry)]
    if include_completed:
        active_entries.extend(entry for entry in entries if not is_open_entry(entry))
    records = [_active_trade_record(entry, history_by_journal.get(str(entry.get("journal_id"))), notifications, include_detail=False) for entry in active_entries]
    need_attention = [record for record in records if record.get("attention_items") and record.get("tracking_state") != "WATCHING_FOR_ENTRY"]
    return {
        "version": ACTIVE_TRADE_WORKSPACE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "records": records,
        "summary": {
            "active_records_found": len(records),
            "entered_positions": sum(1 for record in records if record.get("entered") and not record.get("completed")),
            "tracked_but_not_entered": sum(1 for record in records if record.get("tracking_state") == "WATCHING_FOR_ENTRY"),
            "need_attention": len(need_attention),
            "approaching_milestone": sum(1 for record in records if _number(record.get("progress_percent")) is not None and _number(record.get("progress_percent")) >= 75),
            "awaiting_completion": sum(1 for record in records if record.get("tracking_state") == "STOP_REACHED"),
            "awaiting_replay": sum(1 for record in records if (record.get("verified_history") or {}).get("pipeline_status") in {"COMPLETED_AWAITING_REPLAY", "REPLAY_QUEUED", "REPLAY_RUNNING"}),
            "needs_review": sum(1 for record in records if (record.get("verified_history") or {}).get("pipeline_status") == "NEEDS_REVIEW"),
        },
        "diagnostics": {
            "workspace_version": ACTIVE_TRADE_WORKSPACE_VERSION,
            "journal_records_inspected": len(entries),
            "notification_links": sum(len(record.get("notifications") or []) for record in records),
            "verified_history_links": sum(1 for record in records if record.get("verified_history")),
            "aggregation_duration_ms": round((time.perf_counter() - started) * 1000, 1),
            "largest_payload_records": len(records),
            "ambiguous_associations": 0,
        },
    }


def _trade_intelligence_signature(entries: list[dict]) -> tuple:
    if not entries:
        return (0, None, None, None)
    newest_update = max(str(entry.get("updated_at") or "") for entry in entries)
    newest_replay = max(str(entry.get("replay_cache_generated_at") or "") for entry in entries)
    version_sum = sum(int(entry.get("record_version") or 0) for entry in entries)
    return (len(entries), newest_update, newest_replay, version_sum)


def _trade_intelligence_signature_label(signature: tuple) -> str:
    return "|".join(str(part) for part in signature)


def _trade_intelligence_dataset(force: bool = False) -> dict:
    entries = _journal_repository.list_entries({"status": "all", "limit": 5000, "offset": 0})
    signature = _trade_intelligence_signature(entries)
    with _trade_intelligence_lock:
        cached = (
            not force
            and _trade_intelligence_cache.get("signature") == signature
            and _trade_intelligence_cache.get("snapshot") is not None
        )
        if cached:
            snapshot = dict(_trade_intelligence_cache["snapshot"])
            diagnostics = dict(snapshot.get("diagnostics") or {})
            diagnostics["cache_status"] = "hit"
            diagnostics["cache_signature"] = _trade_intelligence_signature_label(signature)
            snapshot["diagnostics"] = diagnostics
            return {
                "entries": entries,
                "replays": [],
                "analytics": None,
                "verified_records": list(_trade_intelligence_cache.get("verified_records") or []),
                "snapshot": snapshot,
                "cache_status": "hit",
                "signature": signature,
            }

    replays = _replay_positions(entries, summary_only=True) if entries else []
    analytics = verified_analytics_snapshot(entries, replays)
    verified_records = build_verified_trade_records(entries, replays, analytics.get("records") or [])
    eligibility = trade_intelligence_eligibility_funnel(entries, replays, analytics.get("records") or [])
    snapshot = build_trade_intelligence_snapshot(verified_records)
    diagnostics = dict(snapshot.get("diagnostics") or {})
    diagnostics.update({
        "cache_status": "rebuilt",
        "cache_signature": _trade_intelligence_signature_label(signature),
        "cache_generated_at": snapshot.get("generated_at"),
        "excluded_trades": max(0, len(entries) - len(verified_records)),
        "journal_only_counts": eligibility.get("journal_only_count", 0),
        "mismatch_records_excluded": eligibility.get("journal_replay_mismatch_count", 0),
        "replay_pending_count": eligibility.get("replay_pending_count", 0),
        "insufficient_replay_data_count": eligibility.get("insufficient_replay_data_count", 0),
        "replay_only_counts": len(verified_records),
        "eligible_exact_groups": (snapshot.get("knowledge_growth") or {}).get("eligible_exact_groups", 0),
        "eligible_broader_groups": (snapshot.get("knowledge_growth") or {}).get("eligible_broader_groups", 0),
        "insufficient_data_groups": (
            (snapshot.get("knowledge_growth") or {}).get("insufficient_exact_groups", 0)
            + (snapshot.get("knowledge_growth") or {}).get("insufficient_broader_groups", 0)
        ),
    })
    snapshot["diagnostics"] = diagnostics
    snapshot["eligibility_funnel"] = eligibility
    snapshot["data_quality"] = {
        "eligible_verified_trades": eligibility.get("eligible_trade_intelligence_count", 0),
        "needs_review": eligibility.get("journal_replay_mismatch_count", 0),
        "replay_pending": eligibility.get("replay_pending_count", 0),
        "journal_only": eligibility.get("journal_only_count", 0),
        "insufficient_replay_data": eligibility.get("insufficient_replay_data_count", 0),
    }
    with _trade_intelligence_lock:
        _trade_intelligence_cache.update({
            "signature": signature,
            "snapshot": snapshot,
            "verified_records": verified_records,
        })
    return {
        "entries": entries,
        "replays": replays,
        "analytics": analytics,
        "verified_records": verified_records,
        "snapshot": snapshot,
        "cache_status": "rebuilt",
        "signature": signature,
    }


@app.get("/api/dev/trade-intelligence")
def api_dev_trade_intelligence(
    force: bool = Query(default=False),
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    dataset = _trade_intelligence_dataset(force=force)
    snapshot = dict(dataset["snapshot"])
    growth = snapshot.get("knowledge_growth") or {}
    has_eligible_groups = bool((growth.get("eligible_exact_groups") or 0) or (growth.get("eligible_broader_groups") or 0))
    snapshot["status"] = "ready" if has_eligible_groups else "not_ready"
    snapshot["ready"] = has_eligible_groups
    snapshot["message"] = (
        "Trade Intelligence uses replay-verified completed trades only."
        if has_eligible_groups
        else "Not enough verified historical data yet."
    )
    snapshot["journal_entry_count"] = len(dataset.get("entries") or [])
    snapshot["cache_signature"] = _trade_intelligence_signature_label(dataset.get("signature") or _trade_intelligence_signature(dataset.get("entries") or []))
    return snapshot


@app.post("/api/dev/trade-intelligence/similar")
def api_dev_trade_intelligence_similar(
    payload: dict = Body(default_factory=dict),
    exact_min_trades: int = Query(default=30, ge=1, le=10000),
    broad_min_trades: int = Query(default=100, ge=1, le=10000),
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    dataset = _trade_intelligence_dataset(force=False)
    setup = payload.get("setup") if isinstance(payload.get("setup"), dict) else payload
    insight = similar_trade_insight(
        setup or {},
        dataset.get("verified_records") or [],
        exact_min_trades=exact_min_trades,
        broad_min_trades=broad_min_trades,
    )
    insight["status"] = "ready" if insight.get("available") else "not_ready"
    insight["source"] = "verified_replay_history"
    return insight


@app.get("/api/dev/position-replay")
def api_dev_position_replay(
    position_id: str = Query(default=""),
    status: str = Query(default="all"),
    ticker: str = Query(default=""),
    direction: str = Query(default=""),
    outcome_category: str = Query(default=""),
    high_churn_only: bool = Query(default=False),
    ambiguous_only: bool = Query(default=False),
    data_incomplete_only: bool = Query(default=False),
    summary_only: bool = Query(default=True),
    limit: int = Query(default=50, ge=1, le=250),
    offset: int = Query(default=0, ge=0),
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    entries = _journal_repository.list_entries({
        "status": status,
        "ticker": ticker,
        "direction": direction,
        "position_id": position_id,
        "limit": limit,
        "offset": offset,
    })
    if not entries:
        aggregate = aggregate_replays([])
        return {
            "status": "not_ready",
            "ready": False,
            "message": "No server-backed positions are available for replay yet. Migrate or add journal positions to begin historical analysis.",
            "replays": [],
            "aggregate": {**aggregate, **real_evidence_counts([])},
            "evidence_readiness": _readiness_summary([], []),
            "evidence_log": [],
            "evidence_guard": evidence_guard(aggregate),
            "synthetic_results_included": False,
        }
    replays = _replay_positions(entries, summary_only=summary_only)
    if outcome_category:
        replays = [item for item in replays if str(item.get("outcome_category") or "").upper() == outcome_category.upper()]
    if high_churn_only:
        replays = [item for item in replays if item.get("high_churn")]
    if ambiguous_only:
        replays = [item for item in replays if item.get("outcome_order_ambiguous")]
    if data_incomplete_only:
        replays = [item for item in replays if not item.get("data_complete")]
    entries_by_position = {str(entry.get("position_id")): entry for entry in entries}
    filtered_entries = [entries_by_position.get(str(replay.get("position_id")), {}) for replay in replays]
    return _replay_response(filtered_entries, replays, limit=limit, offset=offset)


@app.post("/api/dev/position-replay/refresh")
def api_dev_position_replay_refresh(
    payload: dict = Body(default_factory=dict),
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    mode = str((payload or {}).get("mode") or "stale").lower()
    position_id = str((payload or {}).get("position_id") or "")
    filters = {"status": "all", "limit": 1000, "offset": 0}
    if position_id:
        filters["position_id"] = position_id
    elif mode in {"open", "closed"}:
        filters["status"] = mode
    entries = _journal_repository.list_entries(filters)
    if mode == "stale" and not position_id:
        entries = [entry for entry in entries if str(entry.get("replay_cache_status") or "").lower() == "stale"]
    replays = _replay_positions(entries, summary_only=False)
    refreshed = []
    for entry, replay in zip(entries, replays):
        patch = {
            "record_version": entry.get("record_version"),
            "replay_cache_status": "ready" if replay.get("data_complete") else "partial",
            "replay_cache_generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "replay_cache_replay_version": replay.get("replay_version"),
            "replay_cache_outcome_category": replay.get("outcome_category"),
        }
        try:
            refreshed.append(_journal_repository.update_entry(str(entry.get("journal_id")), patch))
        except (KeyError, JournalConflictError, JournalValidationError):
            continue
    response = _replay_response(entries, replays)
    response["refreshed_positions"] = len(refreshed)
    response["refresh_mode"] = mode
    return response


@app.get("/api/dev/position-replay/{position_id}")
def api_dev_position_replay_one(
    position_id: str,
    summary_only: bool = Query(default=False),
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    entries = _journal_repository.list_entries({
        "status": "all",
        "position_id": position_id,
        "limit": 1,
        "offset": 0,
    })
    if not entries:
        raise HTTPException(status_code=404, detail="Position not found")
    replays = _replay_positions(entries, summary_only=summary_only)
    return _replay_response(entries, replays)


@app.get("/api/dev/verified-analytics")
def api_dev_verified_analytics(
    limit: int = Query(default=1000, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    entries = _journal_repository.list_entries({"status": "all", "limit": limit, "offset": offset})
    replays = _replay_positions(entries, summary_only=True) if entries else []
    snapshot = verified_analytics_snapshot(entries, replays)
    snapshot["status"] = "ready" if entries else "not_ready"
    snapshot["ready"] = bool(entries)
    snapshot["message"] = "Verified analytics distinguish journal-recorded outcomes from replay-supported evidence."
    return snapshot


@app.get("/api/dev/verified-history")
def api_dev_verified_history(
    force_replay: bool = Query(default=False),
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    return _verified_history_records_snapshot(force_replay=force_replay)


@app.get("/api/dev/verified-history/records")
def api_dev_verified_history_records(
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    snapshot = _verified_history_records_snapshot(force_replay=False)
    return {
        "version": snapshot.get("version"),
        "records": snapshot.get("records") or [],
        "summary": snapshot.get("summary") or {},
        "reconciliation": snapshot.get("reconciliation") or {},
    }


@app.get("/api/verified-history/summary")
def api_verified_history_summary(x_kairos_admin_token: str = Header(default="")):
    _require_journal_admin_token(x_kairos_admin_token)
    snapshot = _verified_history_records_snapshot(force_replay=False)
    return {
        "version": snapshot.get("version"),
        "generated_at": snapshot.get("generated_at"),
        "summary": snapshot.get("summary") or {},
        "reconciliation": snapshot.get("reconciliation") or {},
    }


@app.post("/api/dev/verified-history/replay/{journal_id}")
def api_dev_verified_history_replay(
    journal_id: str,
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    entry = _journal_repository.get_entry(journal_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Journal entry not found")
    readiness = completion_readiness(entry)
    if not readiness.get("ready"):
        raise HTTPException(status_code=422, detail={"message": "Journal entry is not replay-ready", "readiness": readiness})
    job, created = _queue_verified_history_job(entry, "manual_replay")
    return {"queued": bool(created), "job": job}


@app.post("/api/dev/verified-history/retry/{journal_id}")
def api_dev_verified_history_retry(
    journal_id: str,
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    entry = _journal_repository.get_entry(journal_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Journal entry not found")
    job, created = _queue_verified_history_job(entry, "manual_retry")
    return {"queued": bool(created), "job": job}


@app.post("/api/dev/verified-history/backfill-preview")
def api_dev_verified_history_backfill_preview(x_kairos_admin_token: str = Header(default="")):
    _require_journal_admin_token(x_kairos_admin_token)
    entries = _journal_repository.list_entries({"status": "all", "limit": 5000, "offset": 0})
    jobs = _verified_history_repository.list_jobs(limit=5000)
    return backfill_preview(entries, jobs)


@app.post("/api/dev/verified-history/backfill")
def api_dev_verified_history_backfill(
    payload: dict = Body(default_factory=dict),
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    requested = set(str(item) for item in ((payload or {}).get("journal_ids") or []))
    if not requested:
        raise HTTPException(status_code=422, detail="Explicit journal_ids are required for backfill execution")
    entries = _journal_repository.list_entries({"status": "all", "limit": 5000, "offset": 0})
    jobs = _verified_history_repository.list_jobs(limit=5000)
    preview = backfill_preview(entries, jobs)
    safe = {str(record.get("journal_id")): record for record in preview.get("records") or [] if record.get("backfill_status") == "SAFE_TO_BACKFILL"}
    queued = []
    skipped = []
    by_id = {str(entry.get("journal_id")): entry for entry in entries}
    for journal_id in sorted(requested):
        if journal_id not in safe:
            skipped.append({"journal_id": journal_id, "reason": "not_safe_to_backfill"})
            continue
        job, created = _queue_verified_history_job(by_id[journal_id], "approved_backfill")
        queued.append({"journal_id": journal_id, "job_id": (job or {}).get("job_id"), "queued": bool(created)})
    return {
        "version": VERIFIED_HISTORY_PIPELINE_VERSION,
        "queued": queued,
        "skipped": skipped,
        "jobs_created": sum(1 for item in queued if item.get("queued")),
    }


@app.post("/api/dev/verified-history/worker/run")
def api_dev_verified_history_worker_run(
    max_jobs: int = Query(default=2, ge=1, le=10),
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    return _process_verified_history_jobs(max_jobs=max_jobs)


@app.post("/api/verified-history/{journal_id}/review-note")
def api_verified_history_review_note(
    journal_id: str,
    payload: dict = Body(default_factory=dict),
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    entry = _journal_repository.get_entry(journal_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Journal entry not found")
    note = str((payload or {}).get("note") or "")[:2000]
    patch = {
        "record_version": entry.get("record_version"),
        "verified_history_review_note": note,
        "verified_history_reviewed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    try:
        updated = _journal_repository.update_entry(journal_id, patch)
    except JournalConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"updated": True, "journal_id": journal_id, "verified_history_review_note": updated.get("verified_history_review_note")}


@app.post("/api/verified-history/{journal_id}/acknowledge")
def api_verified_history_acknowledge(
    journal_id: str,
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    entry = _journal_repository.get_entry(journal_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Journal entry not found")
    patch = {
        "record_version": entry.get("record_version"),
        "verified_history_acknowledged_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    try:
        updated = _journal_repository.update_entry(journal_id, patch)
    except JournalConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"acknowledged": True, "journal_id": journal_id, "verified_history_acknowledged_at": updated.get("verified_history_acknowledged_at")}


@app.get("/api/active-trades")
def api_active_trades(
    include_completed: bool = Query(default=False),
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    return _active_trades_dataset(include_completed=include_completed)


@app.get("/api/active-trades/{trade_id}")
def api_active_trade_detail(
    trade_id: str,
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    entries = _journal_repository.list_entries({"status": "all", "limit": 5000, "offset": 0})
    entry = next((
        item for item in entries
        if str(item.get("position_id")) == str(trade_id)
        or str(item.get("journal_id")) == str(trade_id)
        or str(item.get("id")) == str(trade_id)
    ), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Active trade not found")
    history = _verified_history_records_snapshot(force_replay=False)
    history_by_journal = {str(record.get("journal_id")): record for record in history.get("records") or []}
    notifications = _notification_repository.list_events(unread_only=False, limit=200, offset=0).get("events") or []
    record = _active_trade_record(entry, history_by_journal.get(str(entry.get("journal_id"))), notifications, include_detail=True)
    try:
        dataset = _trade_intelligence_dataset(force=False)
        record["trade_intelligence"] = similar_trade_insight(
            entry,
            dataset.get("verified_records") or [],
        )
    except Exception as exc:
        record["trade_intelligence"] = {"available": False, "error": exc.__class__.__name__}
    return record


@app.post("/api/active-trades/{trade_id}/notes")
def api_active_trade_notes(
    trade_id: str,
    payload: dict = Body(default_factory=dict),
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    entry = _journal_repository.get_entry(trade_id)
    if not entry:
        entries = _journal_repository.list_entries({"status": "all", "position_id": trade_id, "limit": 1, "offset": 0})
        entry = entries[0] if entries else None
    if not entry:
        raise HTTPException(status_code=404, detail="Active trade not found")
    note = str((payload or {}).get("note") or "")[:2000]
    existing_notes = str(_first_present(entry.get("notes"), entry.get("reviewNotes"), "") or "")
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    combined = f"{existing_notes}\n\n[{timestamp}] {note}".strip() if existing_notes else f"[{timestamp}] {note}"
    try:
        updated = _journal_repository.update_entry(str(entry.get("journal_id")), {
            "record_version": entry.get("record_version"),
            "notes": combined,
            "workspace_note_updated_at": timestamp,
        })
    except JournalConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"updated": True, "journal_id": updated.get("journal_id"), "notes": updated.get("notes")}


@app.post("/api/active-trades/{trade_id}/complete")
def api_active_trade_complete(
    trade_id: str,
    payload: dict = Body(default_factory=dict),
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    entry = _journal_repository.get_entry(trade_id)
    if not entry:
        entries = _journal_repository.list_entries({"status": "all", "position_id": trade_id, "limit": 1, "offset": 0})
        entry = entries[0] if entries else None
    if not entry:
        raise HTTPException(status_code=404, detail="Active trade not found")
    state = _active_trade_tracking_state(entry)
    if state == "WATCHING_FOR_ENTRY":
        raise HTTPException(status_code=422, detail="Complete Trade is only available after entry evidence exists")
    exit_timestamp = _first_present((payload or {}).get("exit_timestamp"), (payload or {}).get("tracking_completed_at"))
    exit_reason = str(_first_present((payload or {}).get("exit_reason"), (payload or {}).get("completion_reason"), "") or "").strip()
    exit_price = _number(_first_present((payload or {}).get("exit_price"), (payload or {}).get("exit_underlying_price")))
    if not exit_timestamp or not exit_reason:
        raise HTTPException(status_code=422, detail="Exit timestamp and exit reason are required")
    if exit_price is None:
        raise HTTPException(status_code=422, detail="Exit underlying price is required")
    reason_upper = exit_reason.upper().replace(" ", "_")
    result = str((payload or {}).get("result") or "").strip()
    if not result:
        if reason_upper in {"TP1", "TP2", "TP3", "TARGET"}:
            result = "Win"
        elif reason_upper in {"STOP", "STOP_LOSS"}:
            result = "Loss"
        else:
            result = "Closed"
    patch = {
        "record_version": entry.get("record_version"),
        "result": result,
        "outcome": exit_reason,
        "completion_reason": exit_reason,
        "tracking_status": "completed",
        "tracking_completed_at": exit_timestamp,
        "exit_timestamp": exit_timestamp,
        "exit_price": exit_price,
        "actual_exit_premium": _number(_first_present((payload or {}).get("actual_exit_premium"), (payload or {}).get("contract_exit_value"))),
        "actual_option_pnl": _number((payload or {}).get("actual_option_pnl")),
        "quantity_closed": _number((payload or {}).get("quantity_closed")),
        "completion_notes": str((payload or {}).get("notes") or "")[:2000],
    }
    try:
        updated = _journal_repository.update_entry(str(entry.get("journal_id")), patch)
    except JournalConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {
        "completed": True,
        "journal": updated,
        "verified_history": updated.get("verified_history") or _maybe_queue_verified_history_job(updated, "active_trade_completion"),
        "message": "Trade completion was journaled. Kairos will replay and verify the result through Verified History.",
    }


@app.get("/api/dev/active-trade-workspace")
def api_dev_active_trade_workspace(x_kairos_admin_token: str = Header(default="")):
    _require_journal_admin_token(x_kairos_admin_token)
    dataset = _active_trades_dataset(include_completed=True)
    return {
        "version": ACTIVE_TRADE_WORKSPACE_VERSION,
        "summary": dataset.get("summary") or {},
        "diagnostics": dataset.get("diagnostics") or {},
        "records": dataset.get("records") or [],
    }


@app.get("/api/chart/candles")
def api_chart_candles(
    symbol: str = Query(..., min_length=1, max_length=16),
    timeframe: str = Query(default="4H"),
    limit: int = Query(default=120, ge=20, le=240),
):
    started = time.perf_counter()
    ticker = str(symbol or "").strip().upper()
    normalized_tf, period, interval = _chart_period_interval(timeframe)
    providers = _chart_provider_candidates(normalized_tf)
    records, diagnostics = _download_chart_candles(ticker, period, interval, limit, providers)
    selected_provider = diagnostics.get("provider") or providers[-1]
    failure_reason = diagnostics.get("failure_reason")
    return {
        "status": "ready" if records else "unavailable",
        "chart_component_version": "guided-trade-chart-v1",
        "symbol": ticker,
        "requested_timeframe": timeframe,
        "normalized_timeframe": normalized_tf,
        "timeframe": normalized_tf,
        "period": period,
        "interval": interval,
        "provider": selected_provider,
        "selected_provider": selected_provider,
        "provider_attempts": diagnostics.get("provider_attempts") or [],
        "fallback_used": bool(diagnostics.get("fallback_used")),
        "failure_reason": failure_reason,
        "candles": records,
        "candles_loaded": len(records),
        "data_source": selected_provider,
        "cache_status": "provider_fetch" if records else "provider_unavailable",
        "chart_load_duration_ms": round((time.perf_counter() - started) * 1000, 1),
    }


def _cached_scan_snapshot_for_shadow(universe: str) -> tuple[Optional[dict], dict]:
    universe = str(universe or "default").strip().lower()
    if universe == "discovered":
        with _discovery_universe_lock:
            symbols = list(_discovery_universe_cache.get("symbols") or [])
        if not symbols:
            return None, {
                "status": "not_ready",
                "message": "Discovered universe cache is not ready. Shadow study does not trigger discovery or scanning.",
            }
        snapshot = analysis_cache_snapshot(symbols, universe="discovered")
        if not snapshot:
            return None, {
                "status": "not_ready",
                "message": "No completed discovered-universe scan cache is available. Shadow study does not trigger scanning.",
            }
        return snapshot, {}
    snapshot = analysis_cache_snapshot()
    if not snapshot:
        return None, {
            "status": "not_ready",
            "message": "No completed default scan cache is available. Shadow study does not trigger scanning.",
        }
    return snapshot, {}


def _shadow_candle_data_for_rows(rows: list[dict], limit: int) -> dict:
    selected = [row for row in rows[:limit] if isinstance(row, dict) and row.get("ticker")]
    by_interval = {}
    for row in selected:
        timeframe = str(row.get("timeframe") or "1D").upper()
        if timeframe == "4H":
            request = ("60d", "4h")
        else:
            request = ("1y", "1d")
        by_interval.setdefault(request, []).append(str(row.get("ticker")).upper())
    candle_data = {}
    for (period, interval), symbols in by_interval.items():
        fetched = _batch_download(list(dict.fromkeys(symbols)), period=period, interval=interval)
        candle_data.update({str(symbol).upper(): df for symbol, df in fetched.items()})
    return candle_data


@app.get("/api/dev/bos-displacement-shadow")
def api_dev_bos_displacement_shadow(
    universe: str = Query(default="default"),
    include_all_traces: bool = Query(default=False),
    limit: int = Query(default=250, ge=1, le=1000),
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    snapshot, not_ready = _cached_scan_snapshot_for_shadow(universe)
    if not snapshot:
        return {
            **not_ready,
            "ready": False,
            "message_guard": "Shadow study only. Live strategy unchanged.",
        }
    rows = [*(snapshot.get("rows") or []), *(snapshot.get("near_miss") or [])]
    candle_data = _shadow_candle_data_for_rows(rows, limit)
    report = build_bos_displacement_shadow_report(rows[:limit], candle_data)
    if not include_all_traces:
        report.pop("all_traces", None)
    return {
        **report,
        "ready": True,
        "universe": str(universe or "default").strip().lower(),
        "scan_cache_generated_at": _format_timestamp(snapshot.get("generated_at")),
        "scan_cache_meta": snapshot.get("scan_meta") or {},
        "candle_symbols_requested": min(len(rows), limit),
        "candle_symbols_returned": len(candle_data),
        "live_strategy_changed": False,
    }


@app.get("/api/data-provider/status")
def api_data_provider_status():
    return {
        "stock_data_provider": configured_provider_name(),
        "stock_data_provider_profile": configured_provider_profile_name(),
        "timeframe_provider_profile": configured_timeframe_provider_profile(),
        "default_provider": "yahoo",
        "alpaca_configured": alpaca_credentials_configured(),
        "note": "Scanner candle routing is controlled by STOCK_DATA_PROVIDER_PROFILE. Options and earnings remain Yahoo-backed.",
    }


@app.get("/api/data-provider/compare")
def api_data_provider_compare(
    ticker: str = Query(default="AAPL"),
    period: str = Query(default="60d"),
    interval: str = Query(default="4h"),
    full: bool = Query(default=False),
    hybrid: bool = Query(default=False),
    tickers: str = Query(default=""),
):
    if hybrid:
        symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if not symbols and full:
            symbols = WATCHLIST
        if not symbols:
            return {
                "success": False,
                "error": "hybrid diagnostics require tickers=... or full=true",
                "usage": "/api/data-provider/compare?hybrid=true&full=true",
            }
        return hybrid_strategy_diagnostics(symbols)
    if full:
        watchlist = [t.strip().upper() for t in tickers.split(",") if t.strip()] if tickers else WATCHLIST
        return validate_watchlist_candles(watchlist)
    return comparison_diagnostics(ticker=ticker, period=period, interval=interval)


@app.get("/api/trends")
def api_trends(
    direction: str = Query(default="all"),
    min_score: int = Query(default=0, ge=0, le=100),
    hide_choppy: bool = Query(default=False),
):
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "rows": scan_trends(direction=direction, min_score=min_score, hide_choppy=hide_choppy),
    }
