from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import logging
import os
import threading
import time

from fastapi import Body, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from scanner import (
    analysis_cache_status,
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
            "qualified_rows": 0,
            "near_miss_rows": 0,
            "message": "Discovery universe is not ready; trigger /api/discovery/run and wait for /api/discovery/status to become ready.",
            "discovery_status": status,
        },
    }


@app.get("/api/scan")
def api_scan(
    tickers: str = Query(default=""),
    refresh: bool = Query(default=False),
    discover: bool = Query(default=False),
    universe: str = Query(default="default"),
):
    """Scan the full watchlist or a custom comma-separated list of tickers."""
    if tickers:
        watchlist = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        result = scan_cached(watchlist, force_refresh=refresh)
    elif str(universe or "").strip().lower() == "discovered":
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
    else:
        use_finviz = bool(discover) or str(universe or "").strip().lower() == "finviz"
        result = scan_cached(force_refresh=refresh, discover=use_finviz)
    return result


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
        return _journal_repository.create_entry(entry)
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
        return _journal_repository.update_entry(journal_id, patch)
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
