from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from datetime import date, datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import Body, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from scanner import (
    analysis_cache_snapshot,
    analysis_cache_status,
    option_pricing_diagnostics,
    queue_option_pricing_for_contracts,
    stock_early_entry_shadow_diagnostics,
    stock_event_memory_presentation_enabled,
    stock_execution_lifecycle_presentation,
    stock_new_entry_signal,
    stock_mission_workflow_bucket,
    stock_mission_workflow_enabled,
    stock_mission_workflow_identity,
    stock_mission_workflow_snapshot,
    build_bos_displacement_shadow_report,
    build_stock_mtf_structure_shadow_report,
    build_stock_mtf_structure_shadow_v2_report,
    build_stock_mtf_structure_shadow_v3_report,
    build_stock_mtf_structure_shadow_v3_historical_outcome_report,
    persist_stock_mtf_structure_shadow_report,
    persist_stock_mtf_structure_shadow_v2_report,
    persist_stock_mtf_structure_shadow_v3_report,
    persist_stock_mtf_structure_shadow_v3_historical_outcome_report,
    _batch_download,
    _flatten_columns,
    scan_cached,
    scan_ticker,
    debug_ticker,
    scan_trends,
    WATCHLIST,
    coverage_baseline_snapshot,
    register_background_periodic_task,
    start_market_cache_refresh,
)
from discovery import build_ranked_discovery_universe, discovery_universe_max_symbols_resolution
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
from provider_migration_audit import (
    four_h_forensics_report,
    migration_state_report,
    provider_comparison_report,
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
from candidates_router import (
    CandidateIn,
    MONITOR_TICK_SECONDS,
    ShortlistIn,
    router as candidates_router,
    run_approved_setup_monitor_tick,
    upsert_candidate_shortlist,
)
from ma_pipeline import MA_PIPELINE_SOURCE, scan_ma_pipeline_candidates
import momentum_pullback_shadow as momentum_pullback
import momentum_pullback_short_lifecycle_experiment as short_lifecycle_experiment

app = FastAPI(title="Stock Options Scanner")
app.include_router(candidates_router)
logger = logging.getLogger(__name__)

DISCOVERY_POOL_VERSION = "kairos-weekly-discovery-pool-v1"
DISCOVERY_POOL_SOURCE = "alpaca"
DISCOVERY_POOL_RANKING_VERSION = "alpaca-liquidity-ranking-v1"
DISCOVERY_POOL_PATH_ENV = "KAIROS_DISCOVERY_POOL_PATH"
DISCOVERY_REJECTION_EVIDENCE_PATH_ENV = "KAIROS_DISCOVERY_REJECTION_EVIDENCE_PATH"
SCHEDULED_SCAN_STATE_PATH_ENV = "KAIROS_SCHEDULED_SCAN_STATE_PATH"
DISCOVERY_POOL_FILENAME = "kairos_weekly_discovery_pool_v1.json"
DISCOVERY_REJECTION_EVIDENCE_FILENAME = "kairos_weekly_discovery_rejection_evidence_v1.json"
SCHEDULED_SCAN_STATE_FILENAME = "kairos_scheduled_discovered_scan_v1.json"
DISCOVERY_REFRESH_WATCHDOG_SECONDS = 60 * 60
DISCOVERED_SCAN_SCHEDULE_HOUR_ET = 10
DISCOVERED_SCAN_SCHEDULE_MINUTE_ET = 0
EASTERN_TZ = ZoneInfo("America/New_York")
_discovery_universe_cache = {
    "version": DISCOVERY_POOL_VERSION,
    "source": DISCOVERY_POOL_SOURCE,
    "ranking_version": DISCOVERY_POOL_RANKING_VERSION,
    "symbols": [],
    "generated_at": None,
    "expires_at": None,
    "trading_week_id": None,
    "first_session_date": None,
    "expires_after_session_date": None,
    "refresh_reason": None,
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
    "loaded_from_disk": False,
    "persisted_at": None,
}
_discovery_universe_lock = threading.RLock()
_discovery_universe_executor = ThreadPoolExecutor(max_workers=1)
_discovered_scan_handoff_lock = threading.RLock()
_discovered_scan_handoff_state = {
    "version": "discovered-scan-handoff-v1",
    "handoff_triggered_at": None,
    "handoff_reason": None,
    "refresh_job_id": None,
    "refresh_started_at": None,
    "refresh_completed_at": None,
    "refresh_failed_at": None,
    "refresh_attempt_count": 0,
    "stale_job_recovered": False,
    "scanner_cache_generation": None,
    "last_checked_at": None,
    "last_result": None,
}
_scheduled_discovered_scan_lock = threading.RLock()
_scheduled_discovered_scan_state = {
    "version": "kairos-scheduled-discovered-scan-v1",
    "scheduled_for": None,
    "triggered_at": None,
    "started_at": None,
    "completed_at": None,
    "status": "idle",
    "universe": "discovered",
    "trading_week_id": None,
    "cache_generated_at": None,
    "refresh_job_id": None,
    "failure_reason": None,
    "last_checked_at": None,
}
_ma_pipeline_scan_lock = threading.RLock()
_ma_pipeline_scan_state = {
    "version": "kairos-ma-pipeline-auto-ingestion-v1",
    "status": "idle",
    "last_checked_at": None,
    "last_started_at": None,
    "last_completed_at": None,
    "last_result": None,
    "last_error": None,
    "last_run_key": None,
    "last_symbol_count": 0,
    "last_candidate_count": 0,
    "last_ingest": None,
    "last_meta": None,
}
_momentum_short_lifecycle_executor = ThreadPoolExecutor(max_workers=1)
_momentum_short_lifecycle_lock = threading.RLock()
_momentum_short_lifecycle_state = {
    "version": short_lifecycle_experiment.EXPERIMENT_VERSION,
    "startup_ready": False,
    "startup_ready_at": None,
    "deferred_before_ready": 0,
    "last_ingestion_run": None,
    "last_watcher_run": None,
    "last_success": None,
    "last_error": None,
    "ingestion": {
        "last_started_at": None,
        "last_completed_at": None,
        "signals_evaluated": 0,
        "newly_captured": 0,
        "duplicates_skipped": 0,
        "errors": [],
        "running": False,
    },
    "watcher": {
        "last_started_at": None,
        "last_completed_at": None,
        "records_checked": 0,
        "records_changed": 0,
        "intraday_fetch_failures": 0,
        "errors": [],
        "running": False,
    },
}
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
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if text.endswith("Z"):
                text = f"{text[:-1]}+00:00"
            try:
                parsed = datetime.fromisoformat(text)
            except ValueError:
                return None
            return parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _observed_market_holiday(actual: date) -> date:
    if actual.weekday() == 5:
        return actual - timedelta(days=1)
    if actual.weekday() == 6:
        return actual + timedelta(days=1)
    return actual


def _nth_weekday(year: int, month: int, weekday: int, nth: int) -> date:
    current = date(year, month, 1)
    offset = (weekday - current.weekday()) % 7
    return current + timedelta(days=offset + 7 * (nth - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    current = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    return current - timedelta(days=(current.weekday() - weekday) % 7)


def _easter_date(year: int) -> date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _us_market_holidays(year: int) -> set[date]:
    holidays = {
        _observed_market_holiday(date(year, 1, 1)),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _easter_date(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),
        _observed_market_holiday(date(year, 6, 19)),
        _observed_market_holiday(date(year, 7, 4)),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4),
        _observed_market_holiday(date(year, 12, 25)),
    }
    return {item for item in holidays if item.year == year}


def _is_us_trading_session(day: date) -> bool:
    return day.weekday() < 5 and day not in _us_market_holidays(day.year)


def _first_trading_session_for_week(day: date) -> date:
    current = day - timedelta(days=day.weekday())
    for offset in range(7):
        candidate = current + timedelta(days=offset)
        if _is_us_trading_session(candidate):
            return candidate
    return current


def _last_trading_session_for_week(day: date) -> date:
    current = day - timedelta(days=day.weekday()) + timedelta(days=6)
    for offset in range(7):
        candidate = current - timedelta(days=offset)
        if _is_us_trading_session(candidate):
            return candidate
    return current


def _trading_week_reference_date(now: Optional[datetime] = None) -> date:
    eastern = _coerce_utc_datetime(now or _utc_now()).astimezone(EASTERN_TZ)
    day = eastern.date()
    if _is_us_trading_session(day):
        return day
    if day.weekday() >= 5:
        current = day
        for _ in range(10):
            current += timedelta(days=1)
            if _is_us_trading_session(current):
                return current
    return day


def _discovery_trading_week(now: Optional[datetime] = None) -> dict:
    reference = _trading_week_reference_date(now)
    first_session = _first_trading_session_for_week(reference)
    last_session = _last_trading_session_for_week(reference)
    return {
        "trading_week_id": f"{first_session.isocalendar().year}-W{first_session.isocalendar().week:02d}",
        "first_session_date": first_session.isoformat(),
        "expires_after_session_date": last_session.isoformat(),
    }


def _datetime_et_for_session(day: date, hour: int, minute: int) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=EASTERN_TZ).astimezone(timezone.utc)


def _parse_iso_date(value) -> Optional[date]:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip())
        except ValueError:
            return None
    return None


def _next_trading_week_first_session(first_session: date) -> date:
    return _first_trading_session_for_week(first_session + timedelta(days=7))


def _next_discovery_refresh_at(now: Optional[datetime] = None) -> datetime:
    week = _discovery_trading_week(now)
    first_session = _parse_iso_date(week.get("first_session_date")) or _trading_week_reference_date(now)
    next_first_session = _next_trading_week_first_session(first_session)
    return _datetime_et_for_session(
        next_first_session,
        DISCOVERED_SCAN_SCHEDULE_HOUR_ET,
        DISCOVERED_SCAN_SCHEDULE_MINUTE_ET,
    )


def _persistent_data_dir() -> Path:
    mount_path = os.getenv("RAILWAY_VOLUME_MOUNT_PATH")
    if mount_path:
        return Path(mount_path)
    if (
        os.getenv("RAILWAY_ENVIRONMENT")
        or os.getenv("RAILWAY_PROJECT_ID")
        or os.getenv("RAILWAY_SERVICE_ID")
    ):
        return Path("/data")
    return Path(__file__).resolve().parent / "data"


def _discovery_pool_path() -> Path:
    configured = os.getenv(DISCOVERY_POOL_PATH_ENV)
    if configured:
        return Path(configured)
    return _persistent_data_dir() / DISCOVERY_POOL_FILENAME


def _scheduled_scan_state_path() -> Path:
    configured = os.getenv(SCHEDULED_SCAN_STATE_PATH_ENV)
    if configured:
        return Path(configured)
    return _persistent_data_dir() / SCHEDULED_SCAN_STATE_FILENAME


def _discovery_rejection_evidence_path() -> Path:
    configured = os.getenv(DISCOVERY_REJECTION_EVIDENCE_PATH_ENV)
    if configured:
        return Path(configured)
    configured_pool = os.getenv(DISCOVERY_POOL_PATH_ENV)
    if configured_pool:
        return Path(configured_pool).with_name(DISCOVERY_REJECTION_EVIDENCE_FILENAME)
    return _persistent_data_dir() / DISCOVERY_REJECTION_EVIDENCE_FILENAME


def _discovery_cache_defaults() -> dict:
    return {
        "version": DISCOVERY_POOL_VERSION,
        "source": DISCOVERY_POOL_SOURCE,
        "ranking_version": DISCOVERY_POOL_RANKING_VERSION,
        "symbols": [],
        "generated_at": None,
        "expires_at": None,
        "trading_week_id": None,
        "first_session_date": None,
        "expires_after_session_date": None,
        "refresh_reason": None,
        "pipeline_counts": {},
        "thresholds": {},
        "formula": {},
        "stage3": {},
        "stage4": {},
        "top_20": [],
        "bottom_20_selected": [],
        "watchlist_overlap": {},
        "rejection_evidence": {},
        "rejection_evidence_path": None,
        "rejection_evidence_counts": {},
        "rejection_evidence_persisted_at": None,
        "last_error": None,
        "last_duration": None,
        "job_id": None,
        "running": False,
        "started_at": None,
        "completed_at": None,
        "metrics": {},
        "loaded_from_disk": False,
        "persisted_at": None,
    }


def _json_safe_discovery_value(value):
    if isinstance(value, datetime):
        return _format_timestamp(value)
    if isinstance(value, dict):
        return {str(key): _json_safe_discovery_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe_discovery_value(item) for item in value]
    return value


def _discovery_pool_payload_from_cache(cached: dict, persisted_at: Optional[datetime] = None) -> dict:
    persisted_at = persisted_at or _utc_now()
    pipeline_counts = cached.get("pipeline_counts") or {}
    thresholds = cached.get("thresholds") or {}
    metrics = cached.get("metrics") or {}
    return {
        "version": DISCOVERY_POOL_VERSION,
        "source": cached.get("source") or DISCOVERY_POOL_SOURCE,
        "symbols": list(cached.get("symbols") or []),
        "generated_at": _format_timestamp(_coerce_utc_datetime(cached.get("generated_at"))),
        "expires_at": _format_timestamp(_coerce_utc_datetime(cached.get("expires_at"))),
        "next_scheduled_refresh": _format_timestamp(_coerce_utc_datetime(cached.get("expires_at"))),
        "trading_week_id": cached.get("trading_week_id"),
        "first_session_date": cached.get("first_session_date"),
        "expires_after_session_date": cached.get("expires_after_session_date"),
        "refresh_reason": cached.get("refresh_reason"),
        "persisted_at": _format_timestamp(persisted_at),
        "raw_source_count": pipeline_counts.get("raw_assets"),
        "optionable_tradable_count": pipeline_counts.get("tradable_optionable"),
        "hygiene_pass_count": pipeline_counts.get("hygiene_passed"),
        "dollar_volume_pass_count": pipeline_counts.get("dollar_volume_passed"),
        "options_liquidity_pass_count": pipeline_counts.get("options_liquidity_passed"),
        "selected_count": len(cached.get("symbols") or []),
        "ranking_version": cached.get("ranking_version") or metrics.get("ranking_version") or DISCOVERY_POOL_RANKING_VERSION,
        "ranking_methodology": (cached.get("formula") or {}).get("combined_liquidity_score"),
        "kairos_intake_cap": thresholds.get("kairos_intake_cap") or thresholds.get("target_universe_size"),
        "kairos_intake_cap_resolution": thresholds.get("kairos_intake_cap_resolution") or {},
        "kairos_intake_cap_warning": (thresholds.get("kairos_intake_cap_resolution") or {}).get("warning"),
        "pipeline_counts": pipeline_counts,
        "thresholds": thresholds,
        "formula": cached.get("formula") or {},
        "stage3": cached.get("stage3") or {},
        "stage4": cached.get("stage4") or {},
        "top_20": cached.get("top_20") or [],
        "bottom_20_selected": cached.get("bottom_20_selected") or [],
        "watchlist_overlap": cached.get("watchlist_overlap") or {},
        "rejection_evidence_path": cached.get("rejection_evidence_path"),
        "rejection_evidence_counts": cached.get("rejection_evidence_counts") or {},
        "rejection_evidence_persisted_at": _format_timestamp(_coerce_utc_datetime(cached.get("rejection_evidence_persisted_at"))),
        "last_duration": cached.get("last_duration"),
        "metrics": metrics,
    }


def _write_discovery_pool_locked() -> None:
    cached = dict(_discovery_universe_cache)
    if not cached.get("symbols") or not cached.get("generated_at") or not cached.get("expires_at"):
        return
    path = _discovery_pool_path()
    payload = _json_safe_discovery_value(_discovery_pool_payload_from_cache(cached))
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with tmp_path.open("w") as handle:
            handle.write(serialized)
            handle.write("\n")
        os.replace(tmp_path, path)
        _discovery_universe_cache["persisted_at"] = _coerce_utc_datetime(payload.get("persisted_at"))
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _rejection_evidence_payload(result: dict, generated_at: datetime, persisted_at: Optional[datetime] = None) -> dict:
    persisted_at = persisted_at or _utc_now()
    evidence = result.get("rejection_evidence") or {}
    generated_text = _format_timestamp(generated_at)

    def stamp_rows(rows: list[dict]) -> list[dict]:
        stamped = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            stamped.append({
                **row,
                "discovery_generated_at": generated_text,
                "discovery_timestamp": generated_text,
            })
        return stamped

    stage3 = stamp_rows(evidence.get("stage3_dollar_volume") or [])
    stage4 = stamp_rows(evidence.get("stage4_options_liquidity") or [])
    return {
        "version": DISCOVERY_POOL_VERSION,
        "source": result.get("source") or DISCOVERY_POOL_SOURCE,
        "ranking_version": result.get("ranking_version") or DISCOVERY_POOL_RANKING_VERSION,
        "generated_at": generated_text,
        "persisted_at": _format_timestamp(persisted_at),
        "counts": {
            "stage3_dollar_volume": len(stage3),
            "stage4_options_liquidity": len(stage4),
            "total": len(stage3) + len(stage4),
        },
        "stage3_dollar_volume": stage3,
        "stage4_options_liquidity": stage4,
    }


def _write_discovery_rejection_evidence(result: dict, generated_at: datetime) -> dict:
    path = _discovery_rejection_evidence_path()
    payload = _json_safe_discovery_value(_rejection_evidence_payload(result, generated_at))
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with tmp_path.open("w") as handle:
            handle.write(serialized)
            handle.write("\n")
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
    return {
        "path": str(path),
        "counts": payload.get("counts") or {},
        "persisted_at": _coerce_utc_datetime(payload.get("persisted_at")),
    }


def _load_discovery_pool_from_disk() -> bool:
    path = _discovery_pool_path()
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        with _discovery_universe_lock:
            _discovery_universe_cache.update(_discovery_cache_defaults())
            _discovery_universe_cache["last_error"] = f"persisted discovery pool invalid: {exc.__class__.__name__}"
        return False

    symbols = [str(symbol or "").strip().upper() for symbol in (payload.get("symbols") or []) if str(symbol or "").strip()]
    generated_at = _coerce_utc_datetime(payload.get("generated_at"))
    expires_at = _coerce_utc_datetime(payload.get("expires_at"))
    if payload.get("version") != DISCOVERY_POOL_VERSION or not symbols or not generated_at or not expires_at:
        with _discovery_universe_lock:
            _discovery_universe_cache.update(_discovery_cache_defaults())
            _discovery_universe_cache["last_error"] = "persisted discovery pool invalid: missing required fields"
        return False

    pipeline_counts = payload.get("pipeline_counts") if isinstance(payload.get("pipeline_counts"), dict) else {}
    thresholds = payload.get("thresholds") if isinstance(payload.get("thresholds"), dict) else {}
    if "kairos_intake_cap_resolution" not in thresholds and isinstance(payload.get("kairos_intake_cap_resolution"), dict):
        thresholds = {**thresholds, "kairos_intake_cap_resolution": payload.get("kairos_intake_cap_resolution")}
    formula = payload.get("formula") if isinstance(payload.get("formula"), dict) else {}
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    week = _discovery_trading_week(generated_at)
    with _discovery_universe_lock:
        _discovery_universe_cache.update({
            **_discovery_cache_defaults(),
            "version": payload.get("version") or DISCOVERY_POOL_VERSION,
            "source": payload.get("source") or DISCOVERY_POOL_SOURCE,
            "ranking_version": payload.get("ranking_version") or DISCOVERY_POOL_RANKING_VERSION,
            "symbols": symbols,
            "generated_at": generated_at,
            "expires_at": expires_at,
            "trading_week_id": payload.get("trading_week_id") or week.get("trading_week_id"),
            "first_session_date": payload.get("first_session_date") or week.get("first_session_date"),
            "expires_after_session_date": payload.get("expires_after_session_date") or week.get("expires_after_session_date"),
            "refresh_reason": payload.get("refresh_reason") or "loaded_from_legacy_or_persisted_pool",
            "pipeline_counts": pipeline_counts,
            "thresholds": thresholds,
            "formula": formula,
            "stage3": payload.get("stage3") if isinstance(payload.get("stage3"), dict) else {},
            "stage4": payload.get("stage4") if isinstance(payload.get("stage4"), dict) else {},
            "top_20": payload.get("top_20") if isinstance(payload.get("top_20"), list) else [],
            "bottom_20_selected": payload.get("bottom_20_selected") if isinstance(payload.get("bottom_20_selected"), list) else [],
            "watchlist_overlap": payload.get("watchlist_overlap") if isinstance(payload.get("watchlist_overlap"), dict) else {},
            "rejection_evidence_path": payload.get("rejection_evidence_path"),
            "rejection_evidence_counts": payload.get("rejection_evidence_counts") if isinstance(payload.get("rejection_evidence_counts"), dict) else {},
            "rejection_evidence_persisted_at": _coerce_utc_datetime(payload.get("rejection_evidence_persisted_at")),
            "last_error": None,
            "last_duration": payload.get("last_duration") or metrics.get("discovery_duration_ms"),
            "job_id": None,
            "running": False,
            "started_at": None,
            "completed_at": generated_at,
            "metrics": metrics,
            "loaded_from_disk": True,
            "persisted_at": _coerce_utc_datetime(payload.get("persisted_at")),
        })
    return True


def _discovery_admin_token() -> str:
    return os.getenv("DISCOVERY_ADMIN_TOKEN", "").strip()


def _journal_admin_token() -> str:
    return os.getenv("JOURNAL_ADMIN_TOKEN", "").strip() or _discovery_admin_token()


def _discovery_status_snapshot(now: Optional[datetime] = None) -> dict:
    now = _coerce_utc_datetime(now or _utc_now())
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
    current_week = _discovery_trading_week(now)
    pool_week_id = cached.get("trading_week_id")
    stale = not generated_at or not cached.get("symbols") or pool_week_id != current_week.get("trading_week_id")
    status = "refreshing" if cached.get("running") else "ready" if cached.get("symbols") else "warming"
    if cached.get("last_error") and not cached.get("symbols") and not cached.get("running"):
        status = "error"
    thresholds = cached.get("thresholds") or {}
    cap_resolution = thresholds.get("kairos_intake_cap_resolution") or discovery_universe_max_symbols_resolution()
    return {
        "status": status,
        "version": cached.get("version") or DISCOVERY_POOL_VERSION,
        "source": cached.get("source") or DISCOVERY_POOL_SOURCE,
        "pool_path": str(_discovery_pool_path()),
        "enabled": bool(_discovery_admin_token()),
        "running": bool(cached.get("running")),
        "job_id": cached.get("job_id"),
        "started_at": _format_timestamp(started_at),
        "completed_at": _format_timestamp(completed_at),
        "generated_at": _format_timestamp(generated_at),
        "expires_at": _format_timestamp(expires_at),
        "next_scheduled_refresh": _format_timestamp(expires_at),
        "trading_week_id": pool_week_id,
        "current_trading_week_id": current_week.get("trading_week_id"),
        "first_session_date": cached.get("first_session_date"),
        "expires_after_session_date": cached.get("expires_after_session_date"),
        "current_first_session_date": current_week.get("first_session_date"),
        "current_expires_after_session_date": current_week.get("expires_after_session_date"),
        "refresh_reason": cached.get("refresh_reason"),
        "age_seconds": age_seconds,
        "expires_in_seconds": expires_in_seconds,
        "stale": bool(stale),
        "has_cache": bool(cached.get("symbols")),
        "selected_count": len(cached.get("symbols") or []),
        "raw_source_count": (cached.get("pipeline_counts") or {}).get("raw_assets"),
        "optionable_tradable_count": (cached.get("pipeline_counts") or {}).get("tradable_optionable"),
        "hygiene_pass_count": (cached.get("pipeline_counts") or {}).get("hygiene_passed"),
        "dollar_volume_pass_count": (cached.get("pipeline_counts") or {}).get("dollar_volume_passed"),
        "options_liquidity_pass_count": (cached.get("pipeline_counts") or {}).get("options_liquidity_passed"),
        "kairos_intake_cap": thresholds.get("kairos_intake_cap") or thresholds.get("target_universe_size") or cap_resolution.get("resolved_value"),
        "kairos_intake_cap_resolution": cap_resolution,
        "kairos_intake_cap_warning": cap_resolution.get("warning"),
        "ranking_version": cached.get("ranking_version") or DISCOVERY_POOL_RANKING_VERSION,
        "ranking_methodology": (cached.get("formula") or {}).get("combined_liquidity_score"),
        "loaded_from_disk": bool(cached.get("loaded_from_disk")),
        "persisted_at": _format_timestamp(_coerce_utc_datetime(cached.get("persisted_at"))),
        "rejection_evidence_path": cached.get("rejection_evidence_path"),
        "rejection_evidence_counts": cached.get("rejection_evidence_counts") or {},
        "rejection_evidence_persisted_at": _format_timestamp(_coerce_utc_datetime(cached.get("rejection_evidence_persisted_at"))),
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
        "scheduled_discovered_scan": _scheduled_scan_snapshot(),
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
    cap_resolution = thresholds.get("kairos_intake_cap_resolution") or {}
    return {
        "discovery_started_at": _format_timestamp(started_at),
        "discovery_completed_at": _format_timestamp(completed_at),
        "discovery_duration_ms": duration_ms,
        "raw_alpaca_asset_count": counts.get("raw_assets"),
        "source": result.get("source") or DISCOVERY_POOL_SOURCE,
        "tradable_optionable_count": counts.get("tradable_optionable"),
        "hygiene_passed_count": counts.get("hygiene_passed"),
        "dollar_volume_passed_count": counts.get("dollar_volume_passed"),
        "options_liquidity_passed_count": counts.get("options_liquidity_passed"),
        "ranked_universe_count_before_cap": counts.get("ranked"),
        "final_admitted_symbol_count": counts.get("selected"),
        "configured_cap": cap,
        "effective_cap": cap,
        "kairos_intake_cap": cap,
        "kairos_intake_cap_resolution": cap_resolution,
        "kairos_intake_cap_warning": cap_resolution.get("warning"),
        "ranking_version": result.get("ranking_version") or DISCOVERY_POOL_RANKING_VERSION,
        "ranking_methodology": (result.get("formula") or {}).get("combined_liquidity_score"),
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
        "trading_week_id": cached.get("trading_week_id"),
        "discovery": cached.get("metrics") or {},
    }


def _scheduled_scan_state_defaults() -> dict:
    return {
        "version": "kairos-scheduled-discovered-scan-v1",
        "scheduled_for": None,
        "triggered_at": None,
        "started_at": None,
        "completed_at": None,
        "status": "idle",
        "universe": "discovered",
        "trading_week_id": None,
        "cache_generated_at": None,
        "refresh_job_id": None,
        "failure_reason": None,
        "last_checked_at": None,
    }


def _write_scheduled_scan_state_locked() -> None:
    path = _scheduled_scan_state_path()
    payload = _json_safe_discovery_value(dict(_scheduled_discovered_scan_state))
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with tmp_path.open("w") as handle:
            handle.write(serialized)
            handle.write("\n")
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _load_scheduled_scan_state_from_disk() -> bool:
    path = _scheduled_scan_state_path()
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        _update_scheduled_scan_state(
            status="state_load_error",
            failure_reason=f"persisted scheduled scan state invalid: {exc.__class__.__name__}",
            last_checked_at=_format_timestamp(_utc_now()),
        )
        return False
    if not isinstance(payload, dict):
        return False
    with _scheduled_discovered_scan_lock:
        merged = _scheduled_scan_state_defaults()
        merged.update({
            key: payload.get(key)
            for key in merged
            if key in payload
        })
        _scheduled_discovered_scan_state.update(merged)
    return True


def _update_scheduled_scan_state(**updates) -> None:
    with _scheduled_discovered_scan_lock:
        _scheduled_discovered_scan_state.update({
            key: value
            for key, value in updates.items()
            if key in _scheduled_discovered_scan_state
        })
        try:
            _write_scheduled_scan_state_locked()
        except Exception:
            logger.exception("scheduled_discovered_scan_state.persist_failed path=%s", _scheduled_scan_state_path())


def _scheduled_scan_snapshot() -> dict:
    with _scheduled_discovered_scan_lock:
        return dict(_scheduled_discovered_scan_state)


def _scheduled_scan_intended_run(now: Optional[datetime] = None) -> dict:
    checked_at = _coerce_utc_datetime(now or _utc_now())
    eastern = checked_at.astimezone(EASTERN_TZ)
    session_date = eastern.date()
    if not _is_us_trading_session(session_date):
        return {
            "due": False,
            "reason": "non_trading_day",
            "session_date": session_date.isoformat(),
            "scheduled_for": None,
            "checked_at": checked_at,
        }
    scheduled_for = _datetime_et_for_session(
        session_date,
        DISCOVERED_SCAN_SCHEDULE_HOUR_ET,
        DISCOVERED_SCAN_SCHEDULE_MINUTE_ET,
    )
    return {
        "due": checked_at >= scheduled_for,
        "reason": "due" if checked_at >= scheduled_for else "not_due",
        "session_date": session_date.isoformat(),
        "scheduled_for": scheduled_for,
        "checked_at": checked_at,
    }


def _scan_cache_generated_for_schedule(meta: dict, scheduled_for: datetime) -> bool:
    generated_at = _coerce_utc_datetime((meta or {}).get("generated_at"))
    return bool(generated_at and generated_at >= scheduled_for and (meta or {}).get("has_cache"))


def _submit_scheduled_discovered_scan_if_due(now: Optional[datetime] = None) -> tuple[bool, str]:
    run = _scheduled_scan_intended_run(now)
    checked_at = run["checked_at"]
    scheduled_for = run.get("scheduled_for")
    checked_text = _format_timestamp(checked_at)
    scheduled_text = _format_timestamp(scheduled_for)
    if run.get("reason") == "non_trading_day":
        _update_scheduled_scan_state(
            last_checked_at=checked_text,
            status="skipped_non_trading_day",
            failure_reason=None,
        )
        return False, "non trading day"
    if not run.get("due"):
        _update_scheduled_scan_state(
            scheduled_for=scheduled_text,
            last_checked_at=checked_text,
            status="waiting_for_schedule",
            failure_reason=None,
        )
        return False, "not due"

    with _scheduled_discovered_scan_lock:
        current_state = dict(_scheduled_discovered_scan_state)
    same_run = current_state.get("scheduled_for") == scheduled_text
    if same_run and current_state.get("status") == "completed":
        _update_scheduled_scan_state(last_checked_at=checked_text)
        return False, "scheduled scan already completed"
    if same_run and current_state.get("status") in {"running", "submitted", "observing_existing_scan"}:
        ready, symbols, discovery_status = _discovery_symbols_ready(checked_at)
        if not ready:
            _update_scheduled_scan_state(
                scheduled_for=scheduled_text,
                last_checked_at=checked_text,
                status="waiting_for_discovery",
                failure_reason=f"discovery_not_ready:{discovery_status.get('status')}",
            )
            return False, "waiting for discovery"
        meta = analysis_cache_status(symbols, universe="discovered")
        if _scan_cache_generated_for_schedule(meta, scheduled_for):
            _update_scheduled_scan_state(
                scheduled_for=scheduled_text,
                completed_at=meta.get("last_refresh_success_at") or meta.get("generated_at"),
                status="completed",
                trading_week_id=discovery_status.get("trading_week_id"),
                cache_generated_at=meta.get("generated_at"),
                refresh_job_id=meta.get("refresh_job_id") or current_state.get("refresh_job_id"),
                failure_reason=None,
                last_checked_at=checked_text,
            )
            _safe_submit_momentum_short_lifecycle_ingestion("scheduled_scan_completed_observed", symbols)
            return False, "scheduled scan completed"
        if meta.get("refreshing"):
            _update_scheduled_scan_state(
                scheduled_for=scheduled_text,
                status="observing_existing_scan",
                trading_week_id=discovery_status.get("trading_week_id"),
                started_at=meta.get("refresh_started_at") or current_state.get("started_at"),
                cache_generated_at=meta.get("generated_at"),
                refresh_job_id=meta.get("refresh_job_id") or current_state.get("refresh_job_id"),
                failure_reason=None,
                last_checked_at=checked_text,
            )
            return False, "scheduled scan already running"

    ready, symbols, discovery_status = _discovery_symbols_ready(checked_at)
    if not ready:
        _submit_discovery_universe_job_if_needed(now=checked_at)
        _update_scheduled_scan_state(
            scheduled_for=scheduled_text,
            triggered_at=checked_text,
            status="waiting_for_discovery",
            trading_week_id=discovery_status.get("current_trading_week_id") or discovery_status.get("trading_week_id"),
            failure_reason=f"discovery_not_ready:{discovery_status.get('status')}",
            last_checked_at=checked_text,
        )
        return False, "waiting for discovery"

    meta = analysis_cache_status(symbols, universe="discovered")
    if _scan_cache_generated_for_schedule(meta, scheduled_for):
        _update_scheduled_scan_state(
            scheduled_for=scheduled_text,
            triggered_at=current_state.get("triggered_at") if same_run else None,
            started_at=meta.get("refresh_started_at"),
            completed_at=meta.get("last_refresh_success_at") or meta.get("generated_at"),
            status="completed",
            trading_week_id=discovery_status.get("trading_week_id"),
            cache_generated_at=meta.get("generated_at"),
            refresh_job_id=meta.get("refresh_job_id"),
            failure_reason=None,
            last_checked_at=checked_text,
        )
        _safe_submit_momentum_short_lifecycle_ingestion("scheduled_scan_cache_ready", symbols)
        return False, "current cache already satisfies schedule"
    if meta.get("refreshing"):
        _update_scheduled_scan_state(
            scheduled_for=scheduled_text,
            triggered_at=checked_text,
            started_at=meta.get("refresh_started_at") or checked_text,
            completed_at=None,
            status="observing_existing_scan",
            trading_week_id=discovery_status.get("trading_week_id"),
            cache_generated_at=meta.get("generated_at"),
            refresh_job_id=meta.get("refresh_job_id"),
            failure_reason=None,
            last_checked_at=checked_text,
        )
        return False, "observing existing scan"

    _update_scheduled_scan_state(
        scheduled_for=scheduled_text,
        triggered_at=checked_text,
        started_at=checked_text,
        completed_at=None,
        status="running",
        trading_week_id=discovery_status.get("trading_week_id"),
        cache_generated_at=meta.get("generated_at"),
        refresh_job_id=meta.get("refresh_job_id"),
        failure_reason=None,
        last_checked_at=checked_text,
    )
    try:
        result = scan_cached(
            symbols,
            force_refresh=True,
            universe="discovered",
            max_symbols=None,
            coverage_context=_discovery_coverage_context(),
            trusted_options_symbols=set(symbols),
        )
        result_meta = (result or {}).get("meta") or analysis_cache_status(symbols, universe="discovered")
    except Exception as exc:
        _update_scheduled_scan_state(
            scheduled_for=scheduled_text,
            completed_at=_format_timestamp(_utc_now()),
            status="failed",
            failure_reason=str(exc),
            last_checked_at=checked_text,
        )
        return False, "scan failed"

    if _scan_cache_generated_for_schedule(result_meta, scheduled_for):
        status = "completed"
        completed_at = result_meta.get("last_refresh_success_at") or result_meta.get("generated_at") or _format_timestamp(_utc_now())
    elif result_meta.get("refreshing"):
        status = "submitted"
        completed_at = None
    elif result_meta.get("last_refresh_error"):
        status = "failed"
        completed_at = _format_timestamp(_utc_now())
    else:
        status = "submitted"
        completed_at = None
    _update_scheduled_scan_state(
        scheduled_for=scheduled_text,
        completed_at=completed_at,
        status=status,
        trading_week_id=discovery_status.get("trading_week_id"),
        cache_generated_at=result_meta.get("generated_at"),
        refresh_job_id=result_meta.get("refresh_job_id"),
        failure_reason=result_meta.get("last_refresh_error"),
        last_checked_at=checked_text,
    )
    if status in {"completed", "submitted"}:
        _safe_submit_momentum_short_lifecycle_ingestion("scheduled_scan_result", symbols)
    return True, result_meta.get("refresh_job_id") or status


def _discovered_scan_handoff_snapshot() -> dict:
    with _discovered_scan_handoff_lock:
        return dict(_discovered_scan_handoff_state)


def _update_discovered_scan_handoff_from_meta(meta: dict) -> None:
    meta = meta or {}
    now = _utc_now()
    updates = {
        "last_checked_at": _format_timestamp(now),
        "refresh_job_id": meta.get("refresh_job_id"),
        "refresh_started_at": meta.get("refresh_started_at"),
    }
    if meta.get("has_cache") and meta.get("generated_at"):
        updates.update({
            "refresh_completed_at": meta.get("last_refresh_success_at") or meta.get("generated_at"),
            "scanner_cache_generation": meta.get("generated_at"),
            "last_result": "cache_ready",
        })
    elif meta.get("refreshing"):
        updates["last_result"] = "refresh_running"
    elif meta.get("last_refresh_error"):
        updates.update({
            "refresh_failed_at": _format_timestamp(now),
            "last_result": "refresh_failed",
        })
        if "timeout" in str(meta.get("last_refresh_error") or "").lower():
            updates["stale_job_recovered"] = True
    with _discovered_scan_handoff_lock:
        _discovered_scan_handoff_state.update({k: v for k, v in updates.items() if v is not None})


def _maybe_enqueue_discovered_scan_handoff(reason: str = "discovery_ready_no_scanner_cache") -> tuple[bool, str]:
    ready, symbols, discovery_status = _discovery_symbols_ready()
    if not ready:
        with _discovered_scan_handoff_lock:
            _discovered_scan_handoff_state.update({
                "last_checked_at": _format_timestamp(_utc_now()),
                "last_result": f"discovery_not_ready:{discovery_status.get('status')}",
            })
        return False, "discovery not ready"

    meta = analysis_cache_status(symbols, universe="discovered")
    _update_discovered_scan_handoff_from_meta(meta)
    if meta.get("has_cache"):
        return False, "scanner cache ready"
    if meta.get("refreshing"):
        return False, "scanner refresh already running"

    started_at = _utc_now()
    with _discovered_scan_handoff_lock:
        _discovered_scan_handoff_state.update({
            "handoff_triggered_at": _format_timestamp(started_at),
            "handoff_reason": reason,
            "refresh_attempt_count": int(_discovered_scan_handoff_state.get("refresh_attempt_count") or 0) + 1,
            "last_checked_at": _format_timestamp(started_at),
            "last_result": "submitted",
        })
    logger.info("discovered_scan_handoff_started reason=%s selected_count=%s", reason, len(symbols))
    result = scan_cached(
        symbols,
        force_refresh=False,
        universe="discovered",
        max_symbols=None,
        coverage_context=_discovery_coverage_context(),
        trusted_options_symbols=set(symbols),
    )
    meta = (result or {}).get("meta") or analysis_cache_status(symbols, universe="discovered")
    _update_discovered_scan_handoff_from_meta(meta)
    if meta.get("has_cache") or meta.get("refreshing"):
        _safe_submit_momentum_short_lifecycle_ingestion("discovered_scan_handoff", symbols)
    return bool(meta.get("refreshing")), meta.get("refresh_job_id") or "submitted"


def _run_discovery_universe_job(job_id: str, refresh_reason: str = "weekly_pool_due") -> None:
    started = time.perf_counter()
    started_at = _utc_now()
    try:
        result = build_ranked_discovery_universe(static_watchlist=WATCHLIST)
        now = _utc_now()
        week = _discovery_trading_week(now)
        expires_at = _next_discovery_refresh_at(now)
        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        metrics = _discovery_metrics_from_result(result, started_at, now, duration_ms)
        with _discovery_universe_lock:
            _discovery_universe_cache.update({
                "version": result.get("version") or DISCOVERY_POOL_VERSION,
                "source": result.get("source") or DISCOVERY_POOL_SOURCE,
                "ranking_version": result.get("ranking_version") or DISCOVERY_POOL_RANKING_VERSION,
                "symbols": result.get("symbols") or [],
                "generated_at": now,
                "expires_at": expires_at,
                "trading_week_id": week.get("trading_week_id"),
                "first_session_date": week.get("first_session_date"),
                "expires_after_session_date": week.get("expires_after_session_date"),
                "refresh_reason": refresh_reason,
                "pipeline_counts": result.get("pipeline_counts") or {},
                "thresholds": result.get("thresholds") or {},
                "formula": result.get("formula") or {},
                "stage3": result.get("stage3") or {},
                "stage4": result.get("stage4") or {},
                "top_20": result.get("top_20") or [],
                "bottom_20_selected": result.get("bottom_20_selected") or [],
                "watchlist_overlap": result.get("watchlist_overlap") or {},
                "rejection_evidence_path": None,
                "rejection_evidence_counts": {},
                "rejection_evidence_persisted_at": None,
                "last_error": None,
                "last_duration": round(duration_ms / 1000, 1),
                "job_id": job_id,
                "running": False,
                "started_at": None,
                "completed_at": now,
                "metrics": metrics,
                "loaded_from_disk": False,
            })
            try:
                _write_discovery_pool_locked()
                evidence_metadata = _write_discovery_rejection_evidence(result, now)
                _discovery_universe_cache.update({
                    "rejection_evidence_path": evidence_metadata.get("path"),
                    "rejection_evidence_counts": evidence_metadata.get("counts") or {},
                    "rejection_evidence_persisted_at": evidence_metadata.get("persisted_at"),
                })
                _write_discovery_pool_locked()
            except Exception as persist_exc:
                logger.exception(
                    "discovery_persistence.persist_failed pool_path=%s evidence_path=%s",
                    _discovery_pool_path(),
                    _discovery_rejection_evidence_path(),
                )
                _discovery_universe_cache["last_error"] = f"discovery persistence failed: {persist_exc.__class__.__name__}"
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
        _maybe_enqueue_discovered_scan_handoff("discovery_completed_no_scanner_cache")
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


def _submit_discovery_universe_job(force: bool = False, reason: str = "weekly_pool_due", now: Optional[datetime] = None) -> tuple[bool, str]:
    with _discovery_universe_lock:
        if _discovery_universe_cache.get("running"):
            return False, "already running"
        if not force and _discovery_universe_cache.get("symbols"):
            status = _discovery_status_snapshot(now)
            if status.get("has_cache") and not status.get("stale"):
                return False, "cache fresh"
        job_id = f"discovery:{int(time.time())}"
        _discovery_universe_cache.update({
            "running": True,
            "started_at": _utc_now(),
            "job_id": job_id,
            "last_error": None,
            "refresh_reason": reason,
        })
    try:
        _discovery_universe_executor.submit(_run_discovery_universe_job, job_id, reason)
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


def _discovery_cache_needs_refresh(now: Optional[datetime] = None) -> bool:
    status = _discovery_status_snapshot(now)
    return not status.get("has_cache") or bool(status.get("stale"))


def _submit_discovery_universe_job_if_needed(now: Optional[datetime] = None) -> tuple[bool, str]:
    if not _discovery_cache_needs_refresh(now):
        return False, "cache fresh"
    return _submit_discovery_universe_job(force=False, reason="weekly_pool_missing_or_stale", now=now)


def _ma_pipeline_scan_schedules() -> list[tuple[int, int]]:
    raw = os.getenv("MA_PIPELINE_SCAN_TIMES_ET", "09:45,15:30")
    schedules = []
    for part in raw.split(","):
        try:
            hour, minute = [int(piece) for piece in part.strip().split(":", 1)]
        except Exception:
            continue
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            schedules.append((hour, minute))
    return schedules or [(9, 45), (15, 30)]


def _ma_pipeline_max_symbols() -> Optional[int]:
    raw = os.getenv("MA_PIPELINE_MAX_SYMBOLS", "").strip()
    default = int(discovery_universe_max_symbols_resolution().get("resolved_value") or 1000)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else None


def _ma_pipeline_state_snapshot() -> dict:
    with _ma_pipeline_scan_lock:
        return dict(_ma_pipeline_scan_state)


def _update_ma_pipeline_state(**updates) -> None:
    with _ma_pipeline_scan_lock:
        _ma_pipeline_scan_state.update({key: value for key, value in updates.items() if value is not None})


def _merge_curated_watchlist_into_universe(discovered_symbols: list[str]) -> tuple[list[str], dict[str, str]]:
    """Merge the legacy scanner's curated WATCHLIST (scanner.py, 113 hardcoded
    symbols) into the broker-fed discovery universe (~939 symbols from
    AlpacaAssetDiscoveryClient, see discovery.build_ranked_discovery_universe)
    for the twice-daily ma_pipeline scan.

    Verified against production /api/discovery/status before writing this:
    110 of WATCHLIST's 113 symbols already overlap the broker-fed universe --
    only REGN/SQ/UNG are genuinely new. So this is mostly NOT about expanding
    raw coverage; WATCHLIST was already being passed into
    build_ranked_discovery_universe(static_watchlist=WATCHLIST) for overlap
    *reporting* (see discovery.py's watchlist_overlap), but that was never
    wired into what actually gets scanned -- this closes that gap, and (the
    part that matters even for the 110 that already overlap) tags every
    resulting candidate with where it came from, so "trusted curated
    watchlist" doesn't silently blend into "broad broker-fed scan" with no
    way to tell them apart afterward.

    Curated-only additions are placed FIRST in the merged list, ahead of the
    broker feed -- so if a future, tighter MA_PIPELINE_MAX_SYMBOLS/
    DISCOVERY_UNIVERSE_MAX_SYMBOLS cap ever truncates the list,
    it eats into the broker feed's tail, never silently drops the handful of
    symbols the watchlist was curated for. Today's real numbers (939 + 3 new,
    cap 1000) mean this ordering doesn't currently change anything -- it's a
    defensive choice for when it might.

    Returns (merged_symbol_list, {symbol: "broker_feed" | "curated_watchlist" | "both"}).
    """
    discovered_norm = [str(s or "").strip().upper() for s in discovered_symbols if str(s or "").strip()]
    discovered_set = set(discovered_norm)
    watchlist_norm = list(dict.fromkeys(str(s or "").strip().upper() for s in WATCHLIST if str(s or "").strip()))
    watchlist_set = set(watchlist_norm)

    curated_only = [symbol for symbol in watchlist_norm if symbol not in discovered_set]
    merged = list(dict.fromkeys([*curated_only, *discovered_norm]))

    origins: dict[str, str] = {}
    for symbol in merged:
        if symbol in discovered_set and symbol in watchlist_set:
            origins[symbol] = "both"
        elif symbol in watchlist_set:
            origins[symbol] = "curated_watchlist"
        else:
            origins[symbol] = "broker_feed"
    return merged, origins


def _run_ma_pipeline_ingestion(reason: str = "manual") -> dict:
    started_at = _utc_now()
    ready, symbols, discovery_status = _discovery_symbols_ready(started_at)
    _update_ma_pipeline_state(
        status="running",
        last_checked_at=_format_timestamp(started_at),
        last_started_at=_format_timestamp(started_at),
        last_error=None,
        last_result=reason,
    )
    if not ready:
        message = f"discovery_not_ready:{discovery_status.get('status')}"
        _update_ma_pipeline_state(status="waiting_for_discovery", last_error=message, last_result=message)
        return {"status": "waiting_for_discovery", "message": message, "discovery_status": discovery_status}

    merged_symbols, symbol_origins = _merge_curated_watchlist_into_universe(symbols)
    max_symbols = _ma_pipeline_max_symbols()
    scan = scan_ma_pipeline_candidates(merged_symbols, max_symbols=max_symbols, symbol_origins=symbol_origins)
    candidates = [CandidateIn(**candidate) for candidate in scan.get("candidates") or []]
    payload = ShortlistIn(source=MA_PIPELINE_SOURCE, scanned_at=_utc_now(), candidates=candidates)
    ingest = upsert_candidate_shortlist(payload)
    completed_at = _utc_now()
    curated_only_count = sum(1 for origin in symbol_origins.values() if origin == "curated_watchlist")
    result = {
        "status": "completed",
        "source": MA_PIPELINE_SOURCE,
        "reason": reason,
        "discovery_symbol_count": len(symbols),
        "merged_symbol_count": len(merged_symbols),
        "curated_only_symbol_count": curated_only_count,
        "scanned_symbol_count": (scan.get("meta") or {}).get("requested"),
        "candidate_count": len(candidates),
        "ingest": ingest.dict(),
        "meta": scan.get("meta") or {},
        "completed_at": _format_timestamp(completed_at),
    }
    _update_ma_pipeline_state(
        status="completed",
        last_completed_at=_format_timestamp(completed_at),
        last_result="completed",
        last_symbol_count=len(merged_symbols),
        last_candidate_count=len(candidates),
        last_ingest=ingest.dict(),
        last_meta=result["meta"],
    )
    return result


def _submit_ma_pipeline_scan_if_due(now: Optional[datetime] = None) -> tuple[bool, str]:
    current = _coerce_utc_datetime(now or _utc_now()).astimezone(EASTERN_TZ)
    _update_ma_pipeline_state(last_checked_at=_format_timestamp(current.astimezone(timezone.utc)))
    if not _is_us_trading_session(current.date()):
        return False, "not a trading session"
    due_schedule = None
    for hour, minute in _ma_pipeline_scan_schedules():
        scheduled = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if scheduled <= current < scheduled + timedelta(minutes=10):
            due_schedule = scheduled
            break
    if due_schedule is None:
        return False, "not due"
    run_key = f"{due_schedule.date().isoformat()}T{due_schedule.hour:02d}:{due_schedule.minute:02d}"
    state = _ma_pipeline_state_snapshot()
    if state.get("last_run_key") == run_key and state.get("status") in {"running", "completed"}:
        return False, "already handled"
    _update_ma_pipeline_state(last_run_key=run_key)
    try:
        _run_ma_pipeline_ingestion(f"scheduled:{run_key}")
    except Exception as exc:
        _update_ma_pipeline_state(
            status="failed",
            last_error=str(exc),
            last_completed_at=_format_timestamp(_utc_now()),
            last_result="failed",
        )
        return False, "failed"
    return True, run_key


_promotion_outcome_lock = threading.Lock()
_promotion_outcome_state = {
    "last_started_at": None,
    "last_completed_at": None,
    "promotions_checked": 0,
    "promotions_resolved": 0,
    "fetch_failures": 0,
    "last_error": None,
    "running": False,
}


def _promotion_outcome_state_snapshot() -> dict:
    with _promotion_outcome_lock:
        return dict(_promotion_outcome_state)


def _update_promotion_outcome_state(**updates) -> None:
    with _promotion_outcome_lock:
        _promotion_outcome_state.update(updates)


def _watch_candidate_promotion_outcomes(reason: str = "periodic") -> dict:
    """Step 2 of real outcome tracking (Option C): resolve real outcomes for
    "taken" promotions from real bars. Structural precedent:
    _momentum_short_lifecycle_watch_open_records (load open records, skip
    terminal ones, fetch fresh data for what's still pending, update state) --
    registered the same simple way as ma_pipeline_candidate_ingestion below,
    not the momentum experiment's separate ThreadPoolExecutor/startup-gate
    machinery, since this has no equivalent startup-ordering dependency.

    Only taken=1 promotions with no outcome yet (or still "still_open") are
    ever examined -- this is what keeps phantom/hypothetical promotions
    (including all 7 that predate the taken column, defaulted to NULL) out
    of the dataset, with no special-casing needed.

    Uses 4h bars only (scanner._batch_download, period="60d") -- the
    DEFAULT_MAX_TRACKING_DAYS window (45 days, see outcome_resolver.py) is
    comfortably inside that 60-day fetch window, so 4h coverage is always
    available for anything not yet expired; daily bars are never used here
    since they're coarser, not better, for same-day stop/target ambiguity.
    """
    # _batch_download/_flatten_columns are module-level imports (top of this
    # file) deliberately, not local ones -- a local `from scanner import
    # _batch_download` here would create a fresh reference every call that
    # bypasses monkeypatch.setattr(main, "_batch_download", ...) in tests
    # entirely (this bit a first draft of this function's test coverage).
    from candidates_router import _get_db
    from outcome_resolver import resolve_outcome

    started = _utc_now()
    _update_promotion_outcome_state(running=True, last_started_at=_format_timestamp(started), last_error=None)
    metrics = {"reason": reason, "promotions_checked": 0, "promotions_resolved": 0, "fetch_failures": 0}
    try:
        conn = _get_db()
        try:
            rows = conn.execute(
                "SELECT * FROM candidate_promotions WHERE taken = 1 AND (outcome IS NULL OR outcome = 'still_open')"
            ).fetchall()
            if not rows:
                return metrics

            by_ticker: dict[str, list] = {}
            for row in rows:
                by_ticker.setdefault(str(row["ticker"]).upper(), []).append(row)

            # Chunked the same way ma_pipeline.py's own downloads are
            # (default chunk size 10) -- see market_data.py/ma_pipeline.py
            # comments on real measured Alpaca pagination density for why
            # that number, not an arbitrary one.
            tickers = list(by_ticker.keys())
            chunk_size = 10
            bars_by_ticker: dict[str, pd.DataFrame] = {}
            for i in range(0, len(tickers), chunk_size):
                chunk = tickers[i:i + chunk_size]
                try:
                    downloaded = _batch_download(chunk, period="60d", interval="4h")
                except Exception:
                    metrics["fetch_failures"] += len(chunk)
                    continue
                for ticker in chunk:
                    frame = downloaded.get(ticker)
                    if frame is None or getattr(frame, "empty", True):
                        metrics["fetch_failures"] += 1
                        continue
                    try:
                        bars_by_ticker[ticker] = _flatten_columns(frame.copy()).dropna().astype(float)
                    except Exception:
                        metrics["fetch_failures"] += 1

            now = _utc_now()
            for ticker, ticker_rows in by_ticker.items():
                bars = bars_by_ticker.get(ticker)
                if bars is None:
                    # Fetch failed or came back empty this cycle -- leave
                    # outcome untouched and retry next hour. Never guess.
                    continue
                bars_utc_index = pd.to_datetime(bars.index, utc=True)
                for row in ticker_rows:
                    metrics["promotions_checked"] += 1
                    promoted_at = _coerce_utc_datetime(row["promoted_at"])
                    if promoted_at is None:
                        continue
                    window = bars.loc[bars_utc_index > promoted_at]
                    result = resolve_outcome(
                        direction=row["direction"],
                        stop=row["stop"],
                        target=row["target"],
                        bars=window,
                        promoted_at=promoted_at,
                        now=now,
                    )
                    if result["outcome"] is None:
                        continue
                    conn.execute(
                        """
                        UPDATE candidate_promotions
                        SET outcome=?, outcome_resolved_at=?, outcome_bar_source=?,
                            outcome_hit_at=?, outcome_note=?
                        WHERE id=?
                        """,
                        (
                            result["outcome"], _format_timestamp(now), result["bar_source"],
                            result["hit_at"], result["note"], row["id"],
                        ),
                    )
                    if result["outcome"] != "still_open":
                        metrics["promotions_resolved"] += 1
            conn.commit()
        finally:
            conn.close()
        return metrics
    except Exception as exc:
        _update_promotion_outcome_state(last_error=str(exc))
        raise
    finally:
        completed = _utc_now()
        _update_promotion_outcome_state(
            running=False,
            last_completed_at=_format_timestamp(completed),
            promotions_checked=metrics.get("promotions_checked", 0),
            promotions_resolved=metrics.get("promotions_resolved", 0),
            fetch_failures=metrics.get("fetch_failures", 0),
        )


def _safe_watch_candidate_promotion_outcomes(reason: str = "periodic") -> None:
    try:
        _watch_candidate_promotion_outcomes(reason)
    except Exception as exc:
        logger.warning("[promotion_outcome_watcher] failed reason=%s error=%s", reason, exc)


def _safe_run_approved_setup_monitor_tick(reason: str = "periodic") -> None:
    # Execution Layer V1 (execution_layer_v1_implementation_plan.md
    # sections 5-9) -- registered the same simple way as
    # candidate_promotion_outcome_watcher above, same reasoning: no
    # startup-ordering dependency, scoped to a small (5-20) active set,
    # not the scanner universe.
    try:
        run_approved_setup_monitor_tick(reason)
    except Exception as exc:
        logger.warning("[approved_setup_monitor] failed reason=%s error=%s", reason, exc)


def _register_discovery_background_refresh() -> None:
    register_background_periodic_task(
        "discovery_universe",
        DISCOVERY_REFRESH_WATCHDOG_SECONDS,
        _submit_discovery_universe_job_if_needed,
    )
    register_background_periodic_task(
        "discovered_scan_handoff",
        30,
        lambda: _maybe_enqueue_discovered_scan_handoff("periodic_discovery_ready_no_scanner_cache"),
    )
    register_background_periodic_task(
        "scheduled_discovered_scan_10am_et",
        60,
        _submit_scheduled_discovered_scan_if_due,
    )
    register_background_periodic_task(
        "ma_pipeline_candidate_ingestion",
        60,
        _submit_ma_pipeline_scan_if_due,
    )
    register_background_periodic_task(
        "momentum_pullback_short_lifecycle_ingestion",
        60 * 60,
        lambda: _safe_submit_momentum_short_lifecycle_ingestion("periodic"),
    )
    register_background_periodic_task(
        "momentum_pullback_short_lifecycle_watcher",
        60 * 60,
        lambda: _submit_momentum_short_lifecycle_watcher("periodic"),
    )
    register_background_periodic_task(
        "candidate_promotion_outcome_watcher",
        60 * 60,
        lambda: _safe_watch_candidate_promotion_outcomes("periodic"),
    )
    register_background_periodic_task(
        "approved_setup_monitor",
        MONITOR_TICK_SECONDS,
        lambda: _safe_run_approved_setup_monitor_tick("periodic"),
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
    _load_scheduled_scan_state_from_disk()
    _load_discovery_pool_from_disk()
    _submit_discovery_universe_job_if_needed()
    _maybe_enqueue_discovered_scan_handoff("startup_discovery_ready_no_scanner_cache")
    _submit_scheduled_discovered_scan_if_due()
    _momentum_short_lifecycle_mark_startup_ready()


@app.get("/")
def index():
    # candidates.html (the SMC/confluence promotion dashboard) is the
    # default homepage as of 2026-08-29 -- the legacy scanner.py-driven
    # "Trade Readiness" card UI below is kept reachable at /legacy as a
    # read-only reference, not deleted (see /legacy). This is a real
    # routing change, not a cosmetic one: anything that assumed GET / meant
    # "the legacy scanner page" (e.g. tests/api_cache_headers_v1.py) needs
    # to follow the redirect, not read this response directly.
    return RedirectResponse(url="/candidates", status_code=307)


@app.get("/legacy")
def legacy_index():
    # Read-only reference copy of the pre-candidates.html scanner UI --
    # verified before this route was added that public/index.html was ALSO
    # already reachable at /static/index.html via the StaticFiles mount
    # below (an incidental side effect of serving all of public/ as static
    # assets, not an intentional route), so this isn't the only thing
    # keeping the legacy page alive -- it's the clean, discoverable,
    # intentional one, and what candidates.html's "Legacy Scanner" link
    # points to.
    return FileResponse(
        "public/index.html",
        headers=NO_STORE_HEADERS,
    )


@app.get("/candidates")
def candidates_dashboard():
    return FileResponse(
        "public/candidates.html",
        headers=NO_STORE_HEADERS,
    )


@app.get("/review-queue")
def review_queue_dashboard():
    # Stage C (2026-08-31 session, human-in-the-loop review-funnel
    # redesign): a standalone page, not a tab/filter inside candidates.html
    # -- see that session's design pass for why. Talks to GET
    # /candidates/review-queue and POST /candidates/{ticker}/visual-review
    # (Stage A/B), reusing the same same-origin session-cookie auth
    # candidates.html already establishes via POST /session.
    return FileResponse(
        "public/review_queue.html",
        headers=NO_STORE_HEADERS,
    )


@app.get("/approved-setups")
def approved_setups_dashboard():
    # Trader-facing board (2026-09 session, "bridge" build): presentation
    # only, no new backend route -- reads the exact same GET
    # /candidates/review-queue payload Review Queue already uses and
    # filters client-side to current_review.decision == "approve" (see
    # public/setup_board.js). Deliberately mirrors /review-queue's route
    # shape exactly.
    return FileResponse(
        "public/approved_setups.html",
        headers=NO_STORE_HEADERS,
    )


@app.get("/watch-setups")
def watch_setups_dashboard():
    # Same bridge build as /approved-setups, filtered to decision == "watch"
    # instead. See public/setup_board.js.
    return FileResponse(
        "public/watch_setups.html",
        headers=NO_STORE_HEADERS,
    )


def _discovery_symbols_ready(now: Optional[datetime] = None):
    status = _discovery_status_snapshot(now)
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


SUMMARY_SCAN_VIEW_VERSION = "scan-summary-v1"
SUMMARY_BUDGET_THRESHOLDS = (100, 250, 500, 1000)
SUMMARY_HEAVY_ROW_FIELDS = {
    "trade_eval",
    "quality",
    "market_regime_details",
    "marketRegimeDetails",
    "_scan_timing",
    "checklist",
}


def _first_present(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _safe_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _summary_contract_cost(contract: dict) -> Optional[float]:
    if not isinstance(contract, dict):
        return None
    explicit = _safe_number(contract.get("estimated_contract_cost"))
    if explicit is not None and explicit > 0:
        return round(explicit, 2)
    ask = _safe_number(contract.get("ask"))
    if ask is not None and ask > 0:
        return round(ask * 100, 2)
    midpoint = _safe_number(_first_present(contract.get("mid"), contract.get("mark")))
    if midpoint is not None and midpoint > 0:
        return round(midpoint * 100, 2)
    return None


def _summary_selected_contract(row: dict) -> dict:
    option = row.get("option") if isinstance(row.get("option"), dict) else {}
    pricing = row.get("option_pricing") if isinstance(row.get("option_pricing"), dict) else {}
    best_contract = row.get("best_contract") if isinstance(row.get("best_contract"), dict) else {}
    selected = {
        "type": _first_present(pricing.get("type"), option.get("type"), best_contract.get("type")),
        "strike": _first_present(pricing.get("strike"), option.get("strike"), best_contract.get("strike")),
        "expiration": _first_present(
            pricing.get("expiration"),
            pricing.get("expiry"),
            option.get("expiration"),
            option.get("expiry"),
            best_contract.get("expiration"),
            best_contract.get("expiry"),
        ),
        "expiry": _first_present(
            pricing.get("expiry"),
            pricing.get("expiration"),
            option.get("expiry"),
            option.get("expiration"),
            best_contract.get("expiry"),
            best_contract.get("expiration"),
        ),
        "requested_expiration": _first_present(
            pricing.get("requested_expiration"),
            option.get("requested_expiration"),
            best_contract.get("requested_expiration"),
            option.get("expiry"),
            option.get("expiration"),
        ),
        "resolved_expiration": _first_present(
            pricing.get("resolved_expiration"),
            option.get("resolved_expiration"),
            best_contract.get("resolved_expiration"),
            pricing.get("expiration"),
            option.get("expiration"),
            option.get("expiry"),
        ),
        "bid": _first_present(pricing.get("bid"), option.get("bid"), best_contract.get("bid")),
        "ask": _first_present(pricing.get("ask"), option.get("ask"), best_contract.get("ask")),
        "mid": _first_present(pricing.get("mid"), pricing.get("mark"), option.get("mid"), option.get("mark"), best_contract.get("mid"), best_contract.get("mark")),
        "mark": _first_present(pricing.get("mark"), pricing.get("mid"), option.get("mark"), option.get("mid"), best_contract.get("mark"), best_contract.get("mid")),
        "spread": _first_present(pricing.get("spread"), option.get("spread"), best_contract.get("spread")),
        "volume": _first_present(pricing.get("volume"), option.get("volume"), best_contract.get("volume")),
        "open_interest": _first_present(pricing.get("open_interest"), option.get("open_interest"), option.get("openInterest"), best_contract.get("open_interest"), best_contract.get("openInterest")),
        "estimated_contract_cost": _first_present(
            pricing.get("estimated_contract_cost"),
            option.get("estimated_contract_cost"),
            best_contract.get("estimated_contract_cost"),
        ),
        "pricing_status": _first_present(row.get("pricing_status"), pricing.get("status"), option.get("pricing_status")),
        "pricing_quality": _first_present(row.get("pricing_quality"), pricing.get("quality"), option.get("pricing_quality")),
        "symbol": _first_present(pricing.get("contract_symbol"), option.get("symbol"), best_contract.get("symbol")),
        "source": _first_present(pricing.get("source"), option.get("pricing_source"), option.get("source"), best_contract.get("source")),
        "quote_timestamp": _first_present(pricing.get("quote_timestamp"), option.get("quote_timestamp")),
    }
    cost = _summary_contract_cost(selected)
    if cost is not None:
        selected["estimated_contract_cost"] = cost
    return selected


def _summary_option_plan(row: dict) -> Optional[dict]:
    plan = row.get("option_plan") if isinstance(row.get("option_plan"), dict) else None
    if not plan:
        return None
    suggested_expiration = plan.get("suggested_expiration") if isinstance(plan.get("suggested_expiration"), dict) else {}
    expected_hold = plan.get("expected_hold") if isinstance(plan.get("expected_hold"), dict) else {}
    expected_move = plan.get("expected_move") if isinstance(plan.get("expected_move"), dict) else {}
    confidence = plan.get("confidence") if isinstance(plan.get("confidence"), dict) else {}
    return {
        "available": plan.get("available"),
        "type": plan.get("type"),
        "preferred_strike": plan.get("preferred_strike"),
        "raw_preferred_strike": plan.get("raw_preferred_strike"),
        "planned_entry": plan.get("planned_entry"),
        "tp1": plan.get("tp1"),
        "suggested_expiration": {
            "min_dte": suggested_expiration.get("min_dte"),
            "max_dte": suggested_expiration.get("max_dte"),
            "label": suggested_expiration.get("label"),
        },
        "expected_hold": {
            "min_trading_days": expected_hold.get("min_trading_days"),
            "max_trading_days": expected_hold.get("max_trading_days"),
            "label": expected_hold.get("label"),
            "fallback_used": expected_hold.get("fallback_used"),
            "fallback_speed": expected_hold.get("fallback_speed"),
        },
        "expected_move": {
            "dollars": expected_move.get("dollars"),
            "percent": expected_move.get("percent"),
            "label": expected_move.get("label"),
        },
        "confidence": {
            "stars": confidence.get("stars"),
            "label": confidence.get("label"),
            "note": confidence.get("note"),
        },
        "source": plan.get("source"),
        "reason": _first_present(plan.get("reason"), plan.get("unavailable_reason")),
    }


def _summary_accessibility(contract: dict, pricing_status: str = "") -> dict:
    status = str(pricing_status or "").lower()
    if status == "pending":
        return {"key": "unavailable", "label": "Pricing pending", "short_label": "Pending", "cost": None}
    if status == "not_requested":
        return {"key": "unavailable", "label": "Pricing not loaded", "short_label": "Not loaded", "cost": None}
    if status == "stale":
        return {"key": "unavailable", "label": "Stale pricing", "short_label": "Stale", "cost": None}
    cost = _summary_contract_cost(contract)
    if cost is None:
        return {"key": "unavailable", "label": "Accessibility unavailable", "short_label": "Unavailable", "cost": None}
    spread = _safe_number(contract.get("spread"))
    open_interest = _safe_number(_first_present(contract.get("open_interest"), contract.get("openInterest")))
    volume = _safe_number(contract.get("volume"))
    spread_pct_from_cost = ((spread * 100) / cost) * 100 if spread is not None and cost > 0 else None
    has_tight_spread = spread_pct_from_cost is not None and spread_pct_from_cost <= 20
    has_liquidity = (open_interest is not None and open_interest >= 100) or (volume is not None and volume >= 25)
    if cost <= 250 and (has_tight_spread or has_liquidity):
        return {"key": "easy", "label": "Easy", "short_label": "Easy", "cost": cost}
    if cost <= 600 or has_tight_spread or has_liquidity:
        return {"key": "moderate", "label": "Moderate", "short_label": "Moderate", "cost": cost}
    return {"key": "premium", "label": "Premium", "short_label": "Premium", "cost": cost}


def _summary_setup_id(row: dict, generation: Optional[str] = None) -> str:
    existing = row.get("setup_id")
    if existing:
        return str(existing)
    parts = [
        row.get("ticker") or row.get("symbol") or "",
        row.get("timeframe") or "",
        row.get("direction") or "",
        row.get("signal_timestamp") or row.get("scannedAt") or "",
        row.get("entry") or "",
        row.get("sl") or "",
        generation or "",
    ]
    return "|".join(str(part) for part in parts)


def _summary_status_bucket(row: dict) -> Optional[str]:
    ranking = row.get("ranking") if isinstance(row.get("ranking"), dict) else {}
    return _first_present(ranking.get("status_bucket"), row.get("status_bucket"), row.get("scanner_status"), row.get("entryStatus"))


def _summary_row(row: dict, generation: Optional[str]) -> dict:
    ranking = row.get("ranking") if isinstance(row.get("ranking"), dict) else {}
    contract = _summary_selected_contract(row)
    option_plan = _summary_option_plan(row)
    pricing_status = str(contract.get("pricing_status") or row.get("pricing_status") or "").lower()
    setup_id = _summary_setup_id(row, generation)
    status_bucket = _summary_status_bucket(row)
    execution_lifecycle = stock_execution_lifecycle_presentation(row)
    execution_lifecycle["ranking_status_bucket"] = status_bucket
    new_entry_signal = row.get("new_entry_signal") if isinstance(row.get("new_entry_signal"), dict) else stock_new_entry_signal(row)
    user_status_bucket = new_entry_signal.get("bucket") or status_bucket
    mission_identity = stock_mission_workflow_identity(row)
    mission_bucket = stock_mission_workflow_bucket(row)
    earnings = row.get("earnings") if isinstance(row.get("earnings"), dict) else {}
    summary = {
        "ticker": row.get("ticker") or row.get("symbol"),
        "setup_id": setup_id,
        "mission_identity": mission_identity,
        "mission_workflow_bucket": mission_bucket,
        "mission_workflow_enabled": stock_mission_workflow_enabled(),
        "scan_generation": generation,
        "timeframe": row.get("timeframe"),
        "direction": row.get("direction"),
        "price": row.get("price"),
        "current_price": _first_present(row.get("current_price"), row.get("price")),
        "current_bar_open": row.get("current_bar_open"),
        "current_bar_high": row.get("current_bar_high"),
        "current_bar_low": row.get("current_bar_low"),
        "current_bar_close": row.get("current_bar_close"),
        "current_candle_time": row.get("current_candle_time"),
        "current_candle_complete": row.get("current_candle_complete"),
        "entry": row.get("entry"),
        "planned_entry": row.get("entry"),
        "sl": row.get("sl"),
        "stop": row.get("sl"),
        "tp1": row.get("tp1"),
        "tp2": row.get("tp2"),
        "tp3": row.get("tp3"),
        "setupGrade": row.get("setupGrade"),
        "setupGradeReason": row.get("setupGradeReason"),
        "display_status": user_status_bucket,
        "status_bucket": user_status_bucket,
        "normalized_status_bucket": user_status_bucket,
        "raw_status_bucket": status_bucket,
        "ranking_status_bucket": execution_lifecycle.get("ranking_status_bucket") or status_bucket,
        "new_entry_signal": new_entry_signal,
        "new_entry_signal_bucket": new_entry_signal.get("bucket"),
        "new_entry_signal_label": new_entry_signal.get("label"),
        "execution_lifecycle": execution_lifecycle,
        "execution_lifecycle_state": execution_lifecycle.get("state"),
        "execution_lifecycle_display": execution_lifecycle.get("display"),
        "execution_lifecycle_reason": execution_lifecycle.get("reason"),
        "execution_lifecycle_presentation_enabled": execution_lifecycle.get("enabled"),
        "entryStatus": row.get("entryStatus"),
        "setupStatus": row.get("setupStatus"),
        "setupStatusReason": row.get("setupStatusReason"),
        "stockTrend": row.get("stockTrend"),
        "trendDirection": row.get("trendDirection"),
        "dailyTrendDirection": row.get("dailyTrendDirection"),
        "h4TrendDirection": row.get("h4TrendDirection"),
        "stockLocation": row.get("stockLocation"),
        "confirmationStarted": row.get("confirmationStarted"),
        "confirmationReason": row.get("confirmationReason"),
        "distanceFromEntryAtr": row.get("distanceFromEntryAtr"),
        "distanceFromEntryPercent": row.get("distanceFromEntryPercent"),
        "entryVisible": row.get("entryVisible"),
        "signal_timestamp": row.get("signal_timestamp"),
        "current_quote_price": row.get("current_quote_price"),
        "ranking": {
            "rank": ranking.get("rank"),
            "tier": ranking.get("tier"),
            "score": ranking.get("score"),
            "status_bucket": ranking.get("status_bucket"),
            "priority_bucket": ranking.get("priority_bucket"),
            "version": ranking.get("version"),
        },
        "earnings": {
            "status": earnings.get("status"),
            "date": earnings.get("date"),
            "days_until": earnings.get("days_until"),
            "source": earnings.get("source"),
            "loaded": earnings.get("loaded"),
            "loading": earnings.get("loading"),
        },
        "option": contract,
        "option_plan": option_plan,
        "option_pricing": {
            "status": contract.get("pricing_status"),
            "quality": contract.get("pricing_quality"),
            "estimated_contract_cost": contract.get("estimated_contract_cost"),
            "premium": contract.get("ask"),
            "bid": contract.get("bid"),
            "ask": contract.get("ask"),
            "mid": contract.get("mid"),
            "mark": contract.get("mark"),
            "spread": contract.get("spread"),
            "volume": contract.get("volume"),
            "open_interest": contract.get("open_interest"),
            "quote_timestamp": contract.get("quote_timestamp"),
            "source": contract.get("source"),
            "symbol": contract.get("symbol"),
            "type": contract.get("type"),
            "strike": contract.get("strike"),
            "expiration": contract.get("expiration"),
            "expiry": contract.get("expiry"),
            "requested_expiration": contract.get("requested_expiration"),
            "resolved_expiration": contract.get("resolved_expiration"),
        },
        "pricing_status": contract.get("pricing_status"),
        "pricing_quality": contract.get("pricing_quality"),
        "accessibility": _summary_accessibility(contract, pricing_status),
        "lazy_hydration": {
            "ticker": row.get("ticker") or row.get("symbol"),
            "type": contract.get("type"),
            "strike": contract.get("strike"),
            "expiration": contract.get("resolved_expiration") or contract.get("expiration") or contract.get("expiry"),
            "requested_expiration": contract.get("requested_expiration"),
            "setup_id": setup_id,
            "scan_generation": generation,
        },
    }
    return {key: value for key, value in summary.items() if key not in SUMMARY_HEAVY_ROW_FIELDS}


def _summary_contract_cost_for_budget(row: dict) -> Optional[float]:
    contract = row.get("option") if isinstance(row.get("option"), dict) else {}
    return _summary_contract_cost(contract)


def _summary_budget_counts(rows: list[dict]) -> dict:
    counts = {}
    for threshold in SUMMARY_BUDGET_THRESHOLDS:
        counts[f"under_{threshold}"] = len([
            row for row in rows
            if (_summary_contract_cost_for_budget(row) is not None and _summary_contract_cost_for_budget(row) <= threshold)
        ])
    return counts


def _summary_today_counts(rows: list[dict]) -> dict:
    buckets = Counter(str((row.get("new_entry_signal") or {}).get("bucket") or row.get("new_entry_signal_bucket") or row.get("normalized_status_bucket") or row.get("status_bucket") or "").upper() for row in rows)
    accessibility = Counter(str((row.get("accessibility") or {}).get("key") or "") for row in rows)
    return {
        "enter_now": buckets.get("ENTER_NOW", 0),
        "waiting_for_retest": buckets.get("WAITING_FOR_RETEST", 0),
        "early_entry": buckets.get("EARLY_ENTRY", 0),
        "early_touch": buckets.get("EARLY_TOUCH", 0),
        "almost_ready": buckets.get("ALMOST_READY", 0),
        "waiting": buckets.get("WAITING", 0),
        "missed_resolved": buckets.get("RESOLVED", 0),
        "budget_friendly": accessibility.get("easy", 0),
        "premium_only": accessibility.get("premium", 0),
    }


def _summarize_scan_response(result: dict) -> dict:
    started = time.perf_counter()
    meta = dict(result.get("meta") or {})
    generation = meta.get("generated_at") or meta.get("ranking_generated_at")
    rows = [_summary_row(row, generation) for row in result.get("rows") or [] if isinstance(row, dict)]
    near_miss = [_summary_row(row, generation) for row in result.get("near_miss") or [] if isinstance(row, dict)]
    all_rows = [*rows, *near_miss]
    summary_meta = {
        **meta,
        "view": "summary",
        "summary_version": SUMMARY_SCAN_VIEW_VERSION,
        "scan_generation": generation,
        "qualified_count": len(rows),
        "near_miss_count": len(near_miss),
        "stock_event_memory_presentation_v1": stock_event_memory_presentation_enabled(),
        "stock_mission_workflow_v1": stock_mission_workflow_enabled(),
        "today_market_counts": _summary_today_counts(all_rows),
        "budget_counts": _summary_budget_counts(all_rows),
        "mission_workflow": meta.get("mission_workflow") or stock_mission_workflow_snapshot(
            rows,
            near_miss,
            universe=str(meta.get("universe") or meta.get("cache_key") or "default"),
            update_movements=False,
        ),
        "summary_generation_ms": round((time.perf_counter() - started) * 1000, 1),
    }
    return {
        "rows": rows,
        "near_miss": near_miss,
        "meta": summary_meta,
    }


def _scan_snapshot_for_universe(universe: str) -> Optional[dict]:
    selected_universe = str(universe or "discovered").strip().lower()
    if selected_universe == "discovered":
        ready, symbols, _status = _discovery_symbols_ready()
        if not ready:
            return None
        return analysis_cache_snapshot(symbols, universe="discovered")
    if selected_universe == "default":
        return analysis_cache_snapshot(universe="default")
    if selected_universe == "finviz":
        return analysis_cache_snapshot(discover=True, universe="finviz")
    return None


def _detail_lookup_from_cache(ticker: str, universe: str, setup_id: str = "", generation: str = "") -> dict:
    snapshot = _scan_snapshot_for_universe(universe)
    if not snapshot:
        raise HTTPException(status_code=404, detail={"reason": "cache_missing", "message": "No scan cache is available for this universe."})
    current_generation = _format_timestamp(_coerce_utc_datetime(snapshot.get("generated_at"))) or _format_timestamp(snapshot.get("generated_at"))
    if generation and generation != current_generation:
        raise HTTPException(
            status_code=409,
            detail={
                "reason": "stale_generation",
                "requested_generation": generation,
                "current_generation": current_generation,
            },
        )
    ticker_key = str(ticker or "").upper()
    matches = []
    for row in [*(snapshot.get("rows") or []), *(snapshot.get("near_miss") or [])]:
        if not isinstance(row, dict):
            continue
        if str(row.get("ticker") or row.get("symbol") or "").upper() != ticker_key:
            continue
        row_setup_id = _summary_setup_id(row, current_generation)
        if setup_id and setup_id != row_setup_id and setup_id != str(row.get("setup_id") or ""):
            continue
        matches.append((row_setup_id, row))
    if not matches:
        raise HTTPException(
            status_code=404,
            detail={
                "reason": "setup_not_found",
                "ticker": ticker_key,
                "setup_id": setup_id or None,
                "generation": current_generation,
            },
        )
    if not setup_id and len(matches) > 1:
        raise HTTPException(
            status_code=409,
            detail={
                "reason": "ambiguous_ticker",
                "ticker": ticker_key,
                "matches": [{"setup_id": match_id} for match_id, _row in matches],
                "generation": current_generation,
            },
        )
    match_id, row = matches[0]
    return {
        "setup": row,
        "setup_id": match_id,
        "scan_generation": current_generation,
        "universe": str(universe or "discovered").strip().lower(),
        "detail": "full",
    }


def _notification_current_setups(universe: str = "discovered") -> list[dict]:
    requested = (universe or "discovered").lower()
    kwargs = {"discover": requested == "discovered", "universe": "discovered" if requested == "discovered" else requested}
    snapshot = analysis_cache_snapshot(None, **kwargs)
    if not snapshot and requested != "default":
        snapshot = analysis_cache_snapshot(WATCHLIST, discover=False, universe="default")
    if not snapshot:
        return []
    return [*(snapshot.get("rows") or []), *(snapshot.get("near_miss") or [])]


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
    view: str = Query(default="full"),
):
    """Scan the full watchlist or a custom comma-separated list of tickers."""
    requested_view = str(view or "full").strip().lower()
    if requested_view not in {"full", "summary"}:
        raise HTTPException(status_code=422, detail="Unsupported scan view")
    if tickers:
        watchlist = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        result = _attach_notification_metrics(scan_cached(watchlist, force_refresh=refresh))
        _safe_submit_momentum_short_lifecycle_ingestion("manual_custom_scan", watchlist)
        return _summarize_scan_response(result) if requested_view == "summary" else result

    selected_universe = str(universe or "discovered").strip().lower()
    if discover:
        selected_universe = "finviz"
    if selected_universe == "discovered":
        ready, symbols, status = _discovery_symbols_ready()
        if not ready:
            result = _discovery_scan_not_ready_response(status)
            return _summarize_scan_response(result) if requested_view == "summary" else result
        result = scan_cached(
            symbols,
            force_refresh=refresh,
            universe="discovered",
            max_symbols=None,
            coverage_context=_discovery_coverage_context(),
            trusted_options_symbols=set(symbols),
        )
        _safe_submit_momentum_short_lifecycle_ingestion("manual_discovered_scan", symbols)
    elif selected_universe == "default":
        result = scan_cached(force_refresh=refresh, discover=False)
        _safe_submit_momentum_short_lifecycle_ingestion("manual_default_scan", WATCHLIST)
    else:
        use_finviz = selected_universe == "finviz"
        result = scan_cached(force_refresh=refresh, discover=use_finviz)
        _safe_submit_momentum_short_lifecycle_ingestion("manual_finviz_scan")
    result = _attach_notification_metrics(result)
    return _summarize_scan_response(result) if requested_view == "summary" else result


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


@app.get("/api/dev/smart-notifications/audit")
def api_smart_notification_audit(
    universe: str = Query(default="discovered"),
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    current_setups = _notification_current_setups(universe)
    audit = _notification_repository.audit_events(current_setups=current_setups, limit=1000)
    audit["current_setup_count"] = len(current_setups)
    audit["universe"] = universe
    return audit


@app.post("/api/dev/smart-notifications/duplicate-cleanup-preview")
def api_smart_notification_duplicate_cleanup_preview(
    universe: str = Query(default="discovered"),
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    current_setups = _notification_current_setups(universe)
    preview = _notification_repository.duplicate_cleanup_preview(current_setups=current_setups, limit=1000)
    preview["current_setup_count"] = len(current_setups)
    preview["universe"] = universe
    return preview


@app.get("/api/scan/{ticker}")
def api_scan_single(
    ticker: str,
    universe: str = Query(default="default"),
    detail: str = Query(default="live"),
    setup_id: str = Query(default=""),
    generation: str = Query(default=""),
):
    if str(detail or "").strip().lower() == "full":
        return _detail_lookup_from_cache(ticker, universe, setup_id=setup_id, generation=generation)
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
def api_cache_status(
    universe: str = Query(default="discovered"),
    discover: bool = Query(default=False),
):
    selected_universe = str(universe or "discovered").strip().lower()
    if discover:
        selected_universe = "finviz"
    if selected_universe == "discovered":
        ready, symbols, discovery_status = _discovery_symbols_ready()
        if not ready:
            meta = _discovery_scan_not_ready_response(discovery_status).get("meta") or {}
        else:
            _maybe_enqueue_discovered_scan_handoff("cache_status_discovery_ready_no_scanner_cache")
            meta = analysis_cache_status(
                symbols,
                universe="discovered",
            )
            _update_discovered_scan_handoff_from_meta(meta)
            if not meta.get("has_cache"):
                meta = {
                    **meta,
                    "status": "scanner_warming",
                    "message": "Discovery ready; scanning the 1,000-symbol universe.",
                    "discovery_status": discovery_status,
                }
        meta = {
            **meta,
            "handoff": _discovered_scan_handoff_snapshot(),
            "scheduled_scan": _scheduled_scan_snapshot(),
        }
    elif selected_universe == "default":
        meta = analysis_cache_status(universe="default")
    elif selected_universe == "finviz":
        meta = analysis_cache_status(discover=True, universe="finviz")
    else:
        raise HTTPException(status_code=422, detail="Unsupported scanner universe")

    return {
        **meta,
        "universe": selected_universe,
        "cache_age_seconds": meta.get("age_seconds"),
        "fresh": not bool(meta.get("stale")) and bool(meta.get("generated_at")),
        "last_successful_refresh": meta.get("last_refresh_success_at"),
        "last_refresh_duration_ms": (
            round(float(meta.get("last_refresh_duration")) * 1000, 1)
            if meta.get("last_refresh_duration") is not None
            else None
        ),
        "qualified_count": meta.get("qualified_rows"),
        "near_miss_count": meta.get("near_miss_rows"),
    }


@app.get("/api/dev/option-pricing")
def api_dev_option_pricing(
    universe: str = Query(default="discovered"),
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    selected_universe = str(universe or "discovered").strip().lower()
    if selected_universe == "discovered":
        ready, symbols, discovery_status = _discovery_symbols_ready()
        if not ready:
            return {
                "version": "option-pricing-async-v1",
                "universe": "discovered",
                "has_cache": False,
                "discovery_status": discovery_status,
                "message": "Discovery universe is not ready.",
            }
        return option_pricing_diagnostics(symbols, universe="discovered")
    if selected_universe == "default":
        return option_pricing_diagnostics(universe="default")
    if selected_universe == "finviz":
        return option_pricing_diagnostics(discover=True, universe="finviz")
    raise HTTPException(status_code=422, detail="Unsupported scanner universe")


@app.get("/api/dev/stock-early-entry-memory")
def api_dev_stock_early_entry_memory(x_kairos_admin_token: str = Header(default="")):
    _require_journal_admin_token(x_kairos_admin_token)
    return stock_early_entry_shadow_diagnostics()


@app.get("/api/dev/momentum-pullback-short-lifecycle-experiment")
def api_dev_momentum_pullback_short_lifecycle_experiment(x_kairos_admin_token: str = Header(default="")):
    _require_journal_admin_token(x_kairos_admin_token)
    return _momentum_short_lifecycle_snapshot()


@app.post("/api/option-pricing/hydrate")
def api_option_pricing_hydrate(payload: dict = Body(default_factory=dict)):
    selected_universe = str((payload or {}).get("universe") or "discovered").strip().lower()
    contracts = (payload or {}).get("contracts")
    if not isinstance(contracts, list):
        raise HTTPException(status_code=422, detail="contracts must be a list")
    if selected_universe == "discovered":
        ready, symbols, discovery_status = _discovery_symbols_ready()
        if not ready:
            return {
                "accepted": False,
                "reason": "discovery_not_ready",
                "discovery_status": discovery_status,
                "queued": 0,
            }
        result = queue_option_pricing_for_contracts(contracts, symbols, universe="discovered")
    elif selected_universe == "default":
        result = queue_option_pricing_for_contracts(contracts, universe="default")
    elif selected_universe == "finviz":
        result = queue_option_pricing_for_contracts(contracts, discover=True, universe="finviz")
    else:
        raise HTTPException(status_code=422, detail="Unsupported scanner universe")
    return {"version": "option-pricing-async-v1", "universe": selected_universe, **result}


@app.post("/api/discovery/run")
def api_discovery_run(
    refresh: bool = Query(default=False),
    x_kairos_admin_token: str = Header(default=""),
):
    _require_discovery_admin_token(x_kairos_admin_token)
    accepted, reason_or_job = _submit_discovery_universe_job(force=refresh, reason="manual_refresh" if refresh else "manual_request")
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


@app.get("/api/v1/scanner/ma-pipeline/status")
def api_ma_pipeline_status():
    ready, symbols, discovery_status = _discovery_symbols_ready()
    return {
        **_ma_pipeline_state_snapshot(),
        "source": MA_PIPELINE_SOURCE,
        "discovery_ready": ready,
        "discovery_symbol_count": len(symbols),
        "discovery_status": discovery_status,
        "schedule_times_et": [f"{hour:02d}:{minute:02d}" for hour, minute in _ma_pipeline_scan_schedules()],
        "max_symbols": _ma_pipeline_max_symbols(),
        "alpaca_configured": alpaca_credentials_configured(),
    }


@app.post("/api/v1/scanner/ma-pipeline/run")
def api_ma_pipeline_run(x_kairos_admin_token: str = Header(default="")):
    _require_discovery_admin_token(x_kairos_admin_token)
    try:
        return _run_ma_pipeline_ingestion("manual_api")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/api/v1/scanner/promotion-outcomes/status")
def api_promotion_outcome_watcher_status():
    from outcome_resolver import DEFAULT_MAX_TRACKING_DAYS
    return {
        **_promotion_outcome_state_snapshot(),
        "max_tracking_days": DEFAULT_MAX_TRACKING_DAYS,
    }


@app.post("/api/v1/scanner/promotion-outcomes/run")
def api_promotion_outcome_watcher_run(x_kairos_admin_token: str = Header(default="")):
    _require_discovery_admin_token(x_kairos_admin_token)
    return _watch_candidate_promotion_outcomes("manual_api")


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


def _momentum_short_lifecycle_update_state(section: Optional[str] = None, **updates) -> None:
    with _momentum_short_lifecycle_lock:
        if section:
            current = dict(_momentum_short_lifecycle_state.get(section) or {})
            current.update(updates)
            _momentum_short_lifecycle_state[section] = current
        else:
            _momentum_short_lifecycle_state.update(updates)


def _momentum_short_lifecycle_is_startup_ready() -> bool:
    with _momentum_short_lifecycle_lock:
        return bool(_momentum_short_lifecycle_state.get("startup_ready"))


def _momentum_short_lifecycle_mark_startup_ready() -> None:
    _momentum_short_lifecycle_update_state(
        startup_ready=True,
        startup_ready_at=_format_timestamp(_utc_now()),
    )


def _momentum_short_lifecycle_defer_until_ready(kind: str, reason: str) -> tuple[bool, str]:
    with _momentum_short_lifecycle_lock:
        _momentum_short_lifecycle_state["deferred_before_ready"] = int(
            _momentum_short_lifecycle_state.get("deferred_before_ready") or 0
        ) + 1
    logger.info("[momentum-short-lifecycle] deferred %s reason=%s until startup ready", kind, reason)
    return False, "startup_not_ready"


def _momentum_short_lifecycle_submit(kind: str, fn, *args) -> tuple[bool, str]:
    section = "ingestion" if kind == "ingestion" else "watcher"
    with _momentum_short_lifecycle_lock:
        if (_momentum_short_lifecycle_state.get(section) or {}).get("running"):
            return False, f"{section}_already_running"
        _momentum_short_lifecycle_update_state(section, running=True)
    try:
        _momentum_short_lifecycle_executor.submit(fn, *args)
    except RuntimeError as exc:
        _momentum_short_lifecycle_update_state(section, running=False, errors=[exc.__class__.__name__])
        return False, f"{section}_submit_failed"
    return True, "submitted"


def _momentum_short_lifecycle_symbols(symbols: Optional[list[str]] = None) -> list[str]:
    if symbols:
        return list(dict.fromkeys(str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()))
    ready, discovered, _ = _discovery_symbols_ready()
    if ready:
        return list(dict.fromkeys(discovered))
    return list(dict.fromkeys(WATCHLIST))


def _momentum_short_lifecycle_fetch_daily(symbols: list[str]) -> dict:
    return _batch_download(symbols, "1y", "1d")


def _momentum_short_lifecycle_ingest(symbols: Optional[list[str]] = None, reason: str = "periodic") -> dict:
    started = _utc_now()
    _momentum_short_lifecycle_update_state("ingestion", last_started_at=_format_timestamp(started), running=True, errors=[])
    metrics = {
        "reason": reason,
        "symbols": 0,
        "signals_evaluated": 0,
        "newly_captured": 0,
        "duplicates_skipped": 0,
        "errors": [],
    }
    try:
        universe = _momentum_short_lifecycle_symbols(symbols)
        metrics["symbols"] = len(universe)
        if not universe:
            return metrics
        daily = _momentum_short_lifecycle_fetch_daily(universe)
        spy_daily = _momentum_short_lifecycle_fetch_daily(["SPY"]).get("SPY")
        ledger = short_lifecycle_experiment.load_ledger()
        for symbol, df in daily.items():
            try:
                report = momentum_pullback.replay_symbol(symbol, df, spy_daily)
            except Exception as exc:
                metrics["errors"].append({"symbol": symbol, "stage": "replay", "error": exc.__class__.__name__})
                continue
            for signal in report.get("signals") or []:
                if signal.get("direction") != momentum_pullback.SHORT:
                    continue
                metrics["signals_evaluated"] += 1
                try:
                    spy_context = (
                        short_lifecycle_experiment.spy_context_at(spy_daily, signal.get("signal_timestamp"))
                        if spy_daily is not None and not getattr(spy_daily, "empty", True)
                        else {"available": False, "reason": "spy_daily_unavailable"}
                    )
                    ledger, status = short_lifecycle_experiment.capture_signal(ledger, signal, spy_context=spy_context)
                    if status == "added":
                        metrics["newly_captured"] += 1
                    elif status == "duplicate":
                        metrics["duplicates_skipped"] += 1
                except Exception as exc:
                    metrics["errors"].append({"symbol": symbol, "stage": "capture", "error": exc.__class__.__name__})
        short_lifecycle_experiment.save_ledger(ledger)
        return metrics
    except Exception as exc:
        metrics["errors"].append({"stage": "ingestion", "error": exc.__class__.__name__, "message": str(exc)[:240]})
        return metrics
    finally:
        completed = _utc_now()
        success = not metrics.get("errors")
        _momentum_short_lifecycle_update_state(
            "ingestion",
            last_completed_at=_format_timestamp(completed),
            signals_evaluated=metrics.get("signals_evaluated", 0),
            newly_captured=metrics.get("newly_captured", 0),
            duplicates_skipped=metrics.get("duplicates_skipped", 0),
            errors=metrics.get("errors", [])[:20],
            running=False,
        )
        _momentum_short_lifecycle_update_state(
            last_ingestion_run=_format_timestamp(completed),
            last_success=_format_timestamp(completed) if success else _momentum_short_lifecycle_state.get("last_success"),
            last_error=None if success else (metrics.get("errors") or [{}])[-1],
        )


def _momentum_short_lifecycle_candles_from_records(records: list[dict]) -> pd.DataFrame:
    rows = []
    index = []
    for item in records or []:
        ts = item.get("timestamp")
        if not ts:
            continue
        rows.append({
            "Open": item.get("open"),
            "High": item.get("high"),
            "Low": item.get("low"),
            "Close": item.get("close"),
            "Volume": item.get("volume"),
        })
        index.append(pd.Timestamp(ts))
    if not rows:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    return pd.DataFrame(rows, index=pd.DatetimeIndex(index))


def _momentum_short_lifecycle_fetch_intraday(symbol: str) -> tuple[pd.DataFrame, dict]:
    providers = _chart_provider_candidates("30M")
    records, diagnostics = _download_chart_candles(symbol, "60d", "30m", 2000, providers)
    return _momentum_short_lifecycle_candles_from_records(records), diagnostics


def _momentum_short_lifecycle_watch_open_records(reason: str = "periodic") -> dict:
    started = _utc_now()
    _momentum_short_lifecycle_update_state("watcher", last_started_at=_format_timestamp(started), running=True, errors=[])
    metrics = {
        "reason": reason,
        "records_checked": 0,
        "records_changed": 0,
        "intraday_fetch_failures": 0,
        "errors": [],
    }
    try:
        ledger = short_lifecycle_experiment.load_ledger()
        records = ledger.get("records") or []
        if not records:
            return metrics
        symbols_needing_daily = list(dict.fromkeys(
            row.get("symbol")
            for row in records
            if row.get("state") == short_lifecycle_experiment.STATE_ENTRY_PENDING
        ))
        daily = _momentum_short_lifecycle_fetch_daily(symbols_needing_daily) if symbols_needing_daily else {}
        updated_ledger = dict(ledger)
        updated_ledger["records"] = []
        for record in records:
            current = record
            terminal = current.get("state") in {
                short_lifecycle_experiment.STATE_EXPERIMENT_COMPLETE,
                short_lifecycle_experiment.STATE_SEQUENCE_AMBIGUOUS,
            }
            if terminal:
                updated_ledger["records"].append(current)
                continue
            metrics["records_checked"] += 1
            before = json.dumps(current, sort_keys=True, default=str)
            symbol = str(current.get("symbol") or "").upper()
            try:
                if current.get("state") == short_lifecycle_experiment.STATE_ENTRY_PENDING:
                    current = short_lifecycle_experiment.resolve_next_session_entry(current, daily.get(symbol, pd.DataFrame()))
                if current.get("state") != short_lifecycle_experiment.STATE_ENTRY_PENDING:
                    intraday, diagnostics = _momentum_short_lifecycle_fetch_intraday(symbol)
                    if intraday.empty:
                        metrics["intraday_fetch_failures"] += 1
                        current.setdefault("diagnostics", {})["last_intraday_failure"] = diagnostics.get("failure_reason") or "no_30m_candles"
                    else:
                        current = short_lifecycle_experiment.evaluate_record_with_intraday(current, intraday)
            except Exception as exc:
                metrics["errors"].append({"symbol": symbol, "stage": "watch", "error": exc.__class__.__name__})
            after = json.dumps(current, sort_keys=True, default=str)
            if before != after:
                metrics["records_changed"] += 1
            updated_ledger["records"].append(current)
        short_lifecycle_experiment.save_ledger(updated_ledger)
        return metrics
    except Exception as exc:
        metrics["errors"].append({"stage": "watcher", "error": exc.__class__.__name__, "message": str(exc)[:240]})
        return metrics
    finally:
        completed = _utc_now()
        success = not metrics.get("errors")
        _momentum_short_lifecycle_update_state(
            "watcher",
            last_completed_at=_format_timestamp(completed),
            records_checked=metrics.get("records_checked", 0),
            records_changed=metrics.get("records_changed", 0),
            intraday_fetch_failures=metrics.get("intraday_fetch_failures", 0),
            errors=metrics.get("errors", [])[:20],
            running=False,
        )
        _momentum_short_lifecycle_update_state(
            last_watcher_run=_format_timestamp(completed),
            last_success=_format_timestamp(completed) if success else _momentum_short_lifecycle_state.get("last_success"),
            last_error=None if success else (metrics.get("errors") or [{}])[-1],
        )


def _submit_momentum_short_lifecycle_ingestion(reason: str = "scan_observed", symbols: Optional[list[str]] = None) -> tuple[bool, str]:
    if not _momentum_short_lifecycle_is_startup_ready():
        return _momentum_short_lifecycle_defer_until_ready("ingestion", reason)
    return _momentum_short_lifecycle_submit("ingestion", _momentum_short_lifecycle_ingest, symbols, reason)


def _safe_submit_momentum_short_lifecycle_ingestion(reason: str = "scan_observed", symbols: Optional[list[str]] = None) -> tuple[bool, str]:
    try:
        return _submit_momentum_short_lifecycle_ingestion(reason, symbols)
    except Exception as exc:
        error = {"stage": "submit_ingestion", "reason": reason, "error": exc.__class__.__name__}
        logger.warning("[momentum-short-lifecycle] ingestion submit failed reason=%s error=%s", reason, exc.__class__.__name__)
        _momentum_short_lifecycle_update_state(last_error=error)
        return False, "submit_failed"


def _submit_momentum_short_lifecycle_watcher(reason: str = "periodic") -> tuple[bool, str]:
    if not _momentum_short_lifecycle_is_startup_ready():
        return _momentum_short_lifecycle_defer_until_ready("watcher", reason)
    return _momentum_short_lifecycle_submit("watcher", _momentum_short_lifecycle_watch_open_records, reason)


def _momentum_short_lifecycle_snapshot() -> dict:
    ledger = short_lifecycle_experiment.load_ledger()
    status = short_lifecycle_experiment.experiment_status(ledger)
    metrics = short_lifecycle_experiment.comparative_metrics(ledger)
    with _momentum_short_lifecycle_lock:
        runtime = json.loads(json.dumps(_momentum_short_lifecycle_state, default=str))
    entry_pending = sum(
        1 for row in ledger.get("records") or []
        if row.get("state") == short_lifecycle_experiment.STATE_ENTRY_PENDING
    )
    return {
        **status,
        "entry_pending": entry_pending,
        "control_a_counts": status.get("control_a_outcome_counts") or {},
        "test_b_counts": status.get("test_b_outcome_counts") or {},
        "sacrificed_1R_continuations": status.get("be_then_control_win_1r"),
        "sacrificed_2R_continuations": status.get("be_then_later_2r"),
        "sacrificed_3R_continuations": status.get("be_then_later_3r"),
        "metrics": metrics,
        "runtime": runtime,
        "last_ingestion_run": runtime.get("last_ingestion_run"),
        "last_watcher_run": runtime.get("last_watcher_run"),
        "last_success": runtime.get("last_success"),
        "last_error": runtime.get("last_error"),
        "ledger": {
            "path": short_lifecycle_experiment.DEFAULT_LEDGER_PATH,
            "exists": Path(short_lifecycle_experiment.DEFAULT_LEDGER_PATH).exists(),
            "records": len(ledger.get("records") or []),
        },
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
    option_type = _first_present(entry.get("actual_option_type"), entry.get("option_type"), entry.get("optionType"))
    strike = _number(_first_present(entry.get("actual_option_strike"), entry.get("option_strike"), entry.get("actual_strike"), entry.get("strike_price"), entry.get("strike")))
    expiration = _first_present(entry.get("actual_option_expiration"), entry.get("option_expiration"), entry.get("actual_expiration"), entry.get("expiration_date"), entry.get("expiry"))
    quantity = _number(_first_present(entry.get("actual_option_quantity"), entry.get("option_quantity"), entry.get("actual_quantity"), entry.get("contracts"))) or 1
    entry_premium = _number(_first_present(entry.get("actual_option_entry_premium"), entry.get("option_entry_premium"), entry.get("actual_option_premium"), entry.get("premium_paid"), entry.get("askAtSelection")))
    entry_cost = _number(entry.get("option_entry_cost"))
    if entry_cost is None and entry_premium is not None and quantity:
        entry_cost = round(entry_premium * 100 * quantity, 2)
    current_premium = _number(entry.get("option_current_premium"))
    current_value = _number(entry.get("option_current_value"))
    if current_value is None and current_premium is not None and quantity:
        current_value = round(current_premium * 100 * quantity, 2)
    unrealized_pl = _number(entry.get("option_unrealized_pl"))
    if unrealized_pl is None and current_value is not None and entry_cost is not None:
        unrealized_pl = round(current_value - entry_cost, 2)
    unrealized_return_pct = _number(entry.get("option_unrealized_return_pct"))
    if unrealized_return_pct is None and unrealized_pl is not None and entry_cost and entry_cost > 0:
        unrealized_return_pct = round((unrealized_pl / entry_cost) * 100, 4)
    stop_premium = _number(entry.get("option_stop_premium"))
    protected_value = _number(entry.get("option_protected_value"))
    if protected_value is None and stop_premium is not None and quantity:
        protected_value = round(stop_premium * 100 * quantity, 2)
    protected_pl = _number(entry.get("option_protected_pl"))
    if protected_pl is None and protected_value is not None and entry_cost is not None:
        protected_pl = round(protected_value - entry_cost, 2)
    protected_return_pct = _number(entry.get("option_protected_return_pct"))
    if protected_return_pct is None and protected_pl is not None and entry_cost and entry_cost > 0:
        protected_return_pct = round((protected_pl / entry_cost) * 100, 4)
    exit_premium = _number(_first_present(entry.get("option_exit_premium"), entry.get("actual_exit_premium"), entry.get("exit_option_premium"), entry.get("exit_premium")))
    exit_quantity = _number(_first_present(entry.get("option_exit_quantity"), entry.get("quantity_closed"), quantity)) or quantity
    exit_value = _number(entry.get("option_exit_value"))
    if exit_value is None and exit_premium is not None and exit_quantity:
        exit_value = round(exit_premium * 100 * exit_quantity, 2)
    allocated_entry_cost = None
    if entry_premium is not None and exit_quantity:
        allocated_entry_cost = round(entry_premium * 100 * exit_quantity, 2)
    realized_pl = _number(_first_present(entry.get("option_realized_pl"), entry.get("actual_option_pnl"), entry.get("manual_realized_pnl"), entry.get("realized_pnl")))
    if realized_pl is None and exit_value is not None and allocated_entry_cost is not None:
        realized_pl = round(exit_value - allocated_entry_cost, 2)
    realized_return_pct = _number(entry.get("option_realized_return_pct"))
    if realized_return_pct is None and realized_pl is not None and allocated_entry_cost and allocated_entry_cost > 0:
        realized_return_pct = round((realized_pl / allocated_entry_cost) * 100, 4)
    return {
        "instrument_type": "option" if _first_present(strike, expiration, entry_premium) else "stock_or_underlying",
        "option_symbol": _first_present(entry.get("option_symbol"), entry.get("option")),
        "option_type": option_type,
        "strike": strike,
        "expiration": expiration,
        "quantity": quantity,
        "entry_premium": entry_premium,
        "entry_cost": entry_cost,
        "tier": _first_present(entry.get("actual_option_contract_tier"), entry.get("option_contract_tier"), entry.get("contract_tier_label"), entry.get("contract_tier")),
        "dte_at_entry": _number(_first_present(entry.get("option_dte_at_entry"), entry.get("days_to_expiration_at_entry"))),
        "breakeven": _number(_first_present(entry.get("option_breakeven_at_entry"), entry.get("breakeven_price"))),
        "current_premium": current_premium,
        "current_value": current_value,
        "unrealized_pl": unrealized_pl,
        "unrealized_return_pct": unrealized_return_pct,
        "stop_premium": stop_premium,
        "protected_value": protected_value,
        "protected_pl": protected_pl,
        "protected_return_pct": protected_return_pct,
        "exit_premium": exit_premium,
        "exit_quantity": exit_quantity,
        "exit_timestamp": _first_present(entry.get("option_exit_timestamp"), entry.get("exit_timestamp"), entry.get("tracking_completed_at")),
        "exit_value": exit_value,
        "actual_option_pnl": realized_pl,
        "realized_pl": realized_pl,
        "realized_return_pct": realized_return_pct,
        "exit_reason": _first_present(entry.get("option_exit_reason"), entry.get("outcome")),
        "exit_notes": _first_present(entry.get("option_exit_notes"), entry.get("completion_notes")),
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
        "position_handoff": entry.get("position_handoff") if isinstance(entry.get("position_handoff"), dict) else None,
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
    option_exit_premium = _number(_first_present((payload or {}).get("option_exit_premium"), (payload or {}).get("actual_exit_premium"), (payload or {}).get("contract_exit_value")))
    if option_exit_premium is not None and option_exit_premium < 0:
        raise HTTPException(status_code=422, detail="Option exit premium must be greater than or equal to 0")
    option_exit_quantity = _number(_first_present((payload or {}).get("option_exit_quantity"), (payload or {}).get("quantity_closed")))
    if option_exit_quantity is not None and (option_exit_quantity <= 0 or float(option_exit_quantity) != int(option_exit_quantity)):
        raise HTTPException(status_code=422, detail="Option exit quantity must be a positive integer")
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
        "option_exit_premium": option_exit_premium,
        "option_exit_quantity": option_exit_quantity,
        "option_exit_timestamp": exit_timestamp,
        "option_exit_reason": exit_reason,
        "option_exit_notes": str((payload or {}).get("option_exit_notes") or (payload or {}).get("notes") or "")[:2000],
        "actual_exit_premium": option_exit_premium,
        "actual_option_pnl": _number((payload or {}).get("actual_option_pnl")),
        "quantity_closed": _number((payload or {}).get("quantity_closed")),
        "completion_notes": str((payload or {}).get("notes") or "")[:2000],
    }
    quantity = _number(_first_present(entry.get("actual_option_quantity"), entry.get("option_quantity"), entry.get("actual_quantity"), entry.get("contracts"))) or 1
    exit_quantity = patch["option_exit_quantity"] or quantity
    patch["option_exit_quantity"] = exit_quantity
    entry_cost = _number(entry.get("option_entry_cost"))
    if entry_cost is None:
        entry_premium = _number(_first_present(entry.get("actual_option_entry_premium"), entry.get("option_entry_premium"), entry.get("actual_option_premium"), entry.get("premium_paid")))
        if entry_premium is not None:
            entry_cost = round(entry_premium * 100 * quantity, 2)
            patch["option_entry_cost"] = entry_cost
    allocated_entry_cost = None
    entry_premium = _number(_first_present(entry.get("actual_option_entry_premium"), entry.get("option_entry_premium"), entry.get("actual_option_premium"), entry.get("premium_paid")))
    if entry_premium is not None and exit_quantity:
        allocated_entry_cost = round(entry_premium * 100 * exit_quantity, 2)
    if patch["option_exit_premium"] is not None:
        patch["option_exit_value"] = round(patch["option_exit_premium"] * 100 * exit_quantity, 2)
        if allocated_entry_cost is not None:
            patch["option_realized_pl"] = round(patch["option_exit_value"] - allocated_entry_cost, 2)
            patch["option_realized_return_pct"] = round((patch["option_realized_pl"] / allocated_entry_cost) * 100, 4) if allocated_entry_cost > 0 else None
            patch["actual_option_pnl"] = patch["actual_option_pnl"] if patch["actual_option_pnl"] is not None else patch["option_realized_pl"]
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


def _mtf_shadow_candle_data_for_rows(rows: list[dict], limit: int) -> dict:
    selected = [row for row in rows[:limit] if isinstance(row, dict) and row.get("ticker")]
    symbols = list(dict.fromkeys(str(row.get("ticker")).upper() for row in selected))
    requests = {
        "1D": ("1y", "1d"),
        "4H": ("60d", "4h"),
        "1H": ("60d", "1h"),
        "30M": ("60d", "30m"),
    }
    candle_data: dict = {symbol: {} for symbol in symbols}
    for label, (period, interval) in requests.items():
        fetched = _batch_download(symbols, period=period, interval=interval) if symbols else {}
        for symbol, df in fetched.items():
            ticker = str(symbol).upper()
            candle_data.setdefault(ticker, {})[label] = df
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


@app.get("/api/dev/stock-mtf-structure-shadow")
def api_dev_stock_mtf_structure_shadow(
    universe: str = Query(default="default"),
    include_all_traces: bool = Query(default=False),
    limit: int = Query(default=1000, ge=1, le=1000),
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    snapshot, not_ready = _cached_scan_snapshot_for_shadow(universe)
    if not snapshot:
        return {
            **not_ready,
            "ready": False,
            "message_guard": "Shadow MTF hierarchy study only. Live strategy unchanged.",
        }
    rows = [*(snapshot.get("rows") or []), *(snapshot.get("near_miss") or [])]
    selected = rows[:limit]
    candle_data = _mtf_shadow_candle_data_for_rows(selected, limit)
    report = build_stock_mtf_structure_shadow_report(selected, candle_data)
    persistence = persist_stock_mtf_structure_shadow_report(
        report,
        source={
            "universe": str(universe or "default").strip().lower(),
            "scan_cache_generated_at": _format_timestamp(snapshot.get("generated_at")),
            "limit": limit,
            "selected_rows": len(selected),
        },
    )
    if not include_all_traces:
        report.pop("all_traces", None)
    return {
        **report,
        "ready": True,
        "universe": str(universe or "default").strip().lower(),
        "scan_cache_generated_at": _format_timestamp(snapshot.get("generated_at")),
        "scan_cache_meta": snapshot.get("scan_meta") or {},
        "candle_symbols_requested": len(selected),
        "candle_symbols_returned": len(candle_data),
        "shadow_persistence": persistence,
        "live_strategy_changed": False,
    }


@app.get("/api/dev/stock-mtf-structure-shadow-v2")
def api_dev_stock_mtf_structure_shadow_v2(
    universe: str = Query(default="default"),
    include_all_traces: bool = Query(default=False),
    limit: int = Query(default=1000, ge=1, le=1000),
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    snapshot, not_ready = _cached_scan_snapshot_for_shadow(universe)
    if not snapshot:
        return {
            **not_ready,
            "ready": False,
            "message_guard": "Shadow MTF hierarchy V2 study only. Live strategy unchanged.",
        }
    rows = [*(snapshot.get("rows") or []), *(snapshot.get("near_miss") or [])]
    selected = rows[:limit]
    candle_data = _mtf_shadow_candle_data_for_rows(selected, limit)
    v1_report = build_stock_mtf_structure_shadow_report(selected, candle_data)
    report = build_stock_mtf_structure_shadow_v2_report(selected, candle_data, v1_report=v1_report)
    persistence = persist_stock_mtf_structure_shadow_v2_report(
        report,
        source={
            "universe": str(universe or "default").strip().lower(),
            "scan_cache_generated_at": _format_timestamp(snapshot.get("generated_at")),
            "limit": limit,
            "selected_rows": len(selected),
            "v1_compared": True,
        },
    )
    if not include_all_traces:
        report.pop("all_traces", None)
    return {
        **report,
        "ready": True,
        "universe": str(universe or "default").strip().lower(),
        "scan_cache_generated_at": _format_timestamp(snapshot.get("generated_at")),
        "scan_cache_meta": snapshot.get("scan_meta") or {},
        "candle_symbols_requested": len(selected),
        "candle_symbols_returned": len(candle_data),
        "shadow_persistence": persistence,
        "live_strategy_changed": False,
    }


@app.get("/api/dev/stock-mtf-structure-shadow-v3")
def api_dev_stock_mtf_structure_shadow_v3(
    universe: str = Query(default="default"),
    include_all_traces: bool = Query(default=False),
    limit: int = Query(default=1000, ge=1, le=1000),
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    snapshot, not_ready = _cached_scan_snapshot_for_shadow(universe)
    if not snapshot:
        return {
            **not_ready,
            "ready": False,
            "message_guard": "Shadow MTF hierarchy V3 study only. Live strategy unchanged.",
        }
    rows = [*(snapshot.get("rows") or []), *(snapshot.get("near_miss") or [])]
    selected = rows[:limit]
    candle_data = _mtf_shadow_candle_data_for_rows(selected, limit)
    v1_report = build_stock_mtf_structure_shadow_report(selected, candle_data)
    v2_report = build_stock_mtf_structure_shadow_v2_report(selected, candle_data, v1_report=v1_report)
    report = build_stock_mtf_structure_shadow_v3_report(selected, candle_data, v2_report=v2_report)
    persistence = persist_stock_mtf_structure_shadow_v3_report(
        report,
        source={
            "universe": str(universe or "default").strip().lower(),
            "scan_cache_generated_at": _format_timestamp(snapshot.get("generated_at")),
            "limit": limit,
            "selected_rows": len(selected),
            "v2_compared": True,
        },
    )
    if not include_all_traces:
        report.pop("all_traces", None)
    return {
        **report,
        "ready": True,
        "universe": str(universe or "default").strip().lower(),
        "scan_cache_generated_at": _format_timestamp(snapshot.get("generated_at")),
        "scan_cache_meta": snapshot.get("scan_meta") or {},
        "candle_symbols_requested": len(selected),
        "candle_symbols_returned": len(candle_data),
        "shadow_persistence": persistence,
        "live_strategy_changed": False,
    }


@app.get("/api/dev/stock-mtf-structure-shadow-v3/history")
def api_dev_stock_mtf_structure_shadow_v3_history(
    universe: str = Query(default="default"),
    limit: int = Query(default=50, ge=1, le=250),
    max_events_per_symbol: Optional[int] = Query(default=None, ge=1, le=25),
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    snapshot, not_ready = _cached_scan_snapshot_for_shadow(universe)
    if not snapshot:
        return {
            **not_ready,
            "ready": False,
            "message_guard": "V3 historical outcome framework only. Live strategy unchanged.",
        }
    rows = [*(snapshot.get("rows") or []), *(snapshot.get("near_miss") or [])]
    symbols = list(dict.fromkeys(
        str(row.get("ticker") or "").upper()
        for row in rows
        if isinstance(row, dict) and row.get("ticker")
    ))[:limit]
    synthetic_rows = [{"ticker": symbol} for symbol in symbols]
    candle_data = _mtf_shadow_candle_data_for_rows(synthetic_rows, len(synthetic_rows))
    report = build_stock_mtf_structure_shadow_v3_historical_outcome_report(
        symbols,
        candle_data,
        max_events_per_symbol=max_events_per_symbol,
    )
    persistence = persist_stock_mtf_structure_shadow_v3_historical_outcome_report(
        report,
        source={
            "universe": str(universe or "default").strip().lower(),
            "scan_cache_generated_at": _format_timestamp(snapshot.get("generated_at")),
            "limit": limit,
            "symbols": symbols,
            "max_events_per_symbol": max_events_per_symbol,
        },
    )
    return {
        **report,
        "ready": True,
        "universe": str(universe or "default").strip().lower(),
        "scan_cache_generated_at": _format_timestamp(snapshot.get("generated_at")),
        "symbols_requested": len(symbols),
        "shadow_persistence": persistence,
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


@app.get("/api/dev/alpaca-migration-audit")
def api_dev_alpaca_migration_audit(x_kairos_admin_token: str = Header(default="")):
    _require_journal_admin_token(x_kairos_admin_token)
    return migration_state_report()


@app.get("/api/dev/provider-comparison")
def api_dev_provider_comparison(
    symbols: str = Query(default=""),
    timeframes: str = Query(default="4H,1D,1W"),
    include_strategy: bool = Query(default=True),
    limit: int = Query(default=12, ge=1, le=50),
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()] if symbols else None
    timeframe_list = [tf.strip().upper() for tf in timeframes.split(",") if tf.strip()] if timeframes else None
    return provider_comparison_report(
        symbols=symbol_list,
        timeframes=timeframe_list,
        include_strategy=include_strategy,
        limit=limit,
    )


@app.get("/api/dev/provider-comparison/4h-forensics")
def api_dev_provider_comparison_4h_forensics(
    symbols: str = Query(default="SPY,NVDA,DOW"),
    start: str = Query(default=""),
    end: str = Query(default=""),
    include_extended_hours: bool = Query(default=False),
    include_strategy: bool = Query(default=True),
    limit: int = Query(default=3, ge=1, le=10),
    x_kairos_admin_token: str = Header(default=""),
):
    _require_journal_admin_token(x_kairos_admin_token)
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()] if symbols else None
    return four_h_forensics_report(
        symbols=symbol_list,
        start=start or None,
        end=end or None,
        include_extended_hours=include_extended_hours,
        include_strategy=include_strategy,
        limit=limit,
    )


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
