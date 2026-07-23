# v2 — BOS + Order Block strategy (replaces EMA pullback)
import pandas as pd
import numpy as np
import contextlib
import io
import logging
import math
import os
import resource
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Optional

from market_data import (
    ALPACA_PROVIDER_NAME,
    MarketDataFacade,
    build_market_data_provider,
    provider_metrics_snapshot,
    provider_name_for_timeframe,
    reset_provider_metrics,
)

logger = logging.getLogger(__name__)
yf = MarketDataFacade()

STOCK_SCANNER_STRATEGY_VERSION = "v1.0"
STOCK_SCANNER_STRATEGY_BASELINE_COMMIT = "7441aac88d5cdf2bb479b85f0e73e4cec629ed57"
VERBOSE_SYMBOL_LOGS = str(os.getenv("KAIROS_VERBOSE_SYMBOL_LOGS") or "").strip().lower() in {"1", "true", "yes", "on"}


@contextlib.contextmanager
def _scan_symbol_stdout_context():
    if VERBOSE_SYMBOL_LOGS:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()):
        yield

WATCHLIST = [
    # ── Airlines ──────────────────────────────────────────────────────────────
    "DAL", "UAL", "AAL", "JBLU",
    # ── Cruise ────────────────────────────────────────────────────────────────
    "CCL", "RCL", "NCLH",
    # ── Tech / Mega-cap ───────────────────────────────────────────────────────
    "NVDA", "AMD", "META", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN",
    # ── Enterprise Software ──────────────────────────────────────────────────
    "ORCL", "CRM", "ADBE",
    # ── Communication Services ────────────────────────────────────────────────
    "NFLX", "DIS", "CMCSA", "T", "VZ", "WBD",
    # ── Semiconductors ────────────────────────────────────────────────────────
    "MU", "INTC", "QCOM", "AVGO", "TSM", "ARM",
    # ── Energy Majors ─────────────────────────────────────────────────────────
    "XOM", "CVX", "OXY",
    # ── Energy Mid-cap / Services ─────────────────────────────────────────────
    "DVN", "HAL", "SLB", "FANG",
    # ── Big Banks ─────────────────────────────────────────────────────────────
    "JPM", "BAC", "GS", "MS", "WFC",
    # ── Asset Managers / Brokers ──────────────────────────────────────────────
    "SCHW", "BLK",
    # ── Payments / Fintech ────────────────────────────────────────────────────
    "V", "MA", "AXP", "PYPL", "SQ",
    # ── Retail / Consumer Discretionary ──────────────────────────────────────
    "WMT", "TGT", "COST", "HD", "LOW", "MCD", "SBUX", "NKE", "F",
    # ── Consumer Staples ──────────────────────────────────────────────────────
    "PG", "KO", "PEP", "CL", "MO", "KHC",
    # ── Healthcare / Pharma ───────────────────────────────────────────────────
    "UNH", "JNJ", "ABBV", "LLY", "PFE", "MRK", "CVS",
    # ── Biotech ───────────────────────────────────────────────────────────────
    "GILD", "REGN", "MRNA", "BIIB",
    # ── Utilities ─────────────────────────────────────────────────────────────
    "NEE", "DUK", "SO", "XEL", "D", "AEP",
    # ── Real Estate ───────────────────────────────────────────────────────────
    "PLD", "AMT", "O", "SPG",
    # ── Industrials / Defense ─────────────────────────────────────────────────
    "CAT", "DE", "HON", "BA", "GE", "LMT", "RTX",
    # ── Materials / Commodities ───────────────────────────────────────────────
    "FCX", "NEM", "AA", "CLF", "LIN", "DOW",
    # ── Broad ETFs ────────────────────────────────────────────────────────────
    "SPY", "QQQ", "IWM", "DIA",
    # ── Sector ETFs ───────────────────────────────────────────────────────────
    "XLF", "XLE", "XLV", "XLU", "XLK", "XLI", "XLB",
    # ── Commodity ETFs ────────────────────────────────────────────────────────
    "GLD", "SLV", "USO", "UNG",
]

REGIME_RANGE_ATR_TOLERANCE = 1.25
REGIME_RECENT_SWING_POINTS = 5
REGIME_WEIGHTS = {
    "ema": 35,
    "swings": 35,
    "htf": 12,
    "volume": 8,
    "atr": 10,
}

TRENDING_UNIVERSE = [
    "IWM", "XLF", "XLE", "SOFI", "PLTR", "HOOD", "AFRM", "RIVN", "AAL",
    "DAL", "CCL", "AMD", "NVDA", "TSLA", "META", "AMZN", "MSFT", "AAPL",
]

NO_EARNINGS_SYMBOLS = {
    "SPY", "QQQ", "IWM", "DIA",
    "GLD", "SLV", "USO", "UNG",
    "XLF", "XLE", "XLV", "XLU", "XLK", "XLI", "XLB",
}

STOCK_SECTOR_MAP = {
    "DAL": "Industrials", "UAL": "Industrials", "AAL": "Industrials", "JBLU": "Industrials",
    "CCL": "Consumer Discretionary", "RCL": "Consumer Discretionary", "NCLH": "Consumer Discretionary",
    "NVDA": "Technology", "AMD": "Technology", "META": "Communication Services", "TSLA": "Consumer Discretionary",
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Communication Services", "AMZN": "Consumer Discretionary",
    "ORCL": "Technology", "CRM": "Technology", "ADBE": "Technology",
    "NFLX": "Communication Services", "DIS": "Communication Services", "CMCSA": "Communication Services",
    "T": "Communication Services", "VZ": "Communication Services", "WBD": "Communication Services",
    "MU": "Technology", "INTC": "Technology", "QCOM": "Technology", "AVGO": "Technology", "TSM": "Technology", "ARM": "Technology",
    "XOM": "Energy", "CVX": "Energy", "OXY": "Energy", "DVN": "Energy", "HAL": "Energy", "SLB": "Energy", "FANG": "Energy",
    "JPM": "Financials", "BAC": "Financials", "GS": "Financials", "MS": "Financials", "WFC": "Financials",
    "SCHW": "Financials", "BLK": "Financials",
    "V": "Financials", "MA": "Financials", "AXP": "Financials", "PYPL": "Financials", "SQ": "Financials",
    "WMT": "Consumer Staples", "TGT": "Consumer Discretionary", "COST": "Consumer Staples", "HD": "Consumer Discretionary",
    "LOW": "Consumer Discretionary", "MCD": "Consumer Discretionary", "SBUX": "Consumer Discretionary", "NKE": "Consumer Discretionary",
    "F": "Consumer Discretionary",
    "PG": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples", "CL": "Consumer Staples", "MO": "Consumer Staples", "KHC": "Consumer Staples",
    "UNH": "Healthcare", "JNJ": "Healthcare", "ABBV": "Healthcare", "LLY": "Healthcare", "PFE": "Healthcare", "MRK": "Healthcare", "CVS": "Healthcare",
    "GILD": "Healthcare", "REGN": "Healthcare", "MRNA": "Healthcare", "BIIB": "Healthcare",
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities", "XEL": "Utilities", "D": "Utilities", "AEP": "Utilities",
    "PLD": "Real Estate", "AMT": "Real Estate", "O": "Real Estate", "SPG": "Real Estate",
    "CAT": "Industrials", "DE": "Industrials", "HON": "Industrials", "BA": "Industrials", "GE": "Industrials", "LMT": "Industrials", "RTX": "Industrials",
    "FCX": "Materials", "NEM": "Materials", "AA": "Materials", "CLF": "Materials", "LIN": "Materials", "DOW": "Materials",
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF", "DIA": "ETF",
    "XLF": "ETF", "XLE": "ETF", "XLV": "ETF", "XLU": "ETF", "XLK": "ETF", "XLI": "ETF", "XLB": "ETF",
    "GLD": "ETF", "SLV": "ETF", "USO": "ETF", "UNG": "ETF",
}


def _sector_for_ticker(ticker: str) -> Optional[str]:
    return STOCK_SECTOR_MAP.get(str(ticker or "").strip().upper())

STOCK_TRADEABILITY_FILTER = {
    "enabled": True,
    "min_avg_volume": 0,
    "avg_volume_lookback": 30,
    "min_option_expirations": 1,
    "exclude_non_major_etfs": False,
    "allowlist": {
        "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV",
        "XLE", "XLF", "XLK", "XLV", "XLU", "XLI", "XLB",
        "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN", "GOOGL",
    },
    "blocklist": set(),
}

STOCK_UNIVERSE_FILTER = STOCK_TRADEABILITY_FILTER

MAJOR_LIQUID_ETFS = {
    "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV",
    "XLE", "XLF", "XLK", "XLV", "XLU", "XLI", "XLB",
}

_finviz_cache = {"tickers": [], "fetched_at": None}
FINVIZ_CACHE_TTL = timedelta(hours=6)
_price_cache = {}
PRICE_CACHE_TTL = timedelta(minutes=3)
_option_chain_cache = {}
OPTION_CHAIN_CACHE_TTL = timedelta(minutes=8)
OPTION_EXPIRATION_CACHE_TTL = timedelta(hours=24)
OPTION_EXPIRATION_FAILURE_RETRY_TTL = timedelta(minutes=5)
OPTION_EXPIRATION_EMPTY_RETRY_DELAY_SECONDS = 1.5
OPTION_YAHOO_RATE_LIMIT_COOLDOWN = timedelta(minutes=10)
_best_contract_cache = {}
BEST_CONTRACT_CACHE_TTL = timedelta(minutes=8)
_option_chain_fetch_semaphore = threading.BoundedSemaphore(4)
_option_yahoo_backoff_until = None
_earnings_cache = {}
EARNINGS_CACHE_TTL = timedelta(hours=12)
EARNINGS_UNAVAILABLE_CACHE_TTL = timedelta(hours=1)
EARNINGS_FAILURE_PRESERVE_TTL = timedelta(days=1)
EARNINGS_SCAN_DEFER_TTL = timedelta(minutes=10)
_cache_lock = threading.RLock()
_cache_stats = Counter()
_background_executor = ThreadPoolExecutor(max_workers=3)
_background_jobs = set()
_background_jobs_lock = threading.Lock()
_background_refresh_started = False
_background_last_refresh = {}
_earnings_deferred_until = {}
_background_periodic_tasks = {}
_active_scan_count = 0
_analysis_cache = {}
ANALYSIS_CACHE_STALE_SECONDS = 180
ANALYSIS_REFRESH_TIMEOUT_SECONDS = 300
_coverage_baseline_snapshot = None
_coverage_baseline_lock = threading.RLock()


def _parse_background_analysis_scan_workers(value: Optional[str], default: int = 4) -> int:
    try:
        parsed = int(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


BACKGROUND_ANALYSIS_SCAN_WORKERS = _parse_background_analysis_scan_workers(
    os.getenv("BACKGROUND_ANALYSIS_SCAN_WORKERS"),
    default=4,
)
_analysis_refresh_state = {}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_utc_datetime(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _format_utc_timestamp(value) -> Optional[str]:
    dt = _coerce_utc_datetime(value)
    if dt is None:
        return None
    # TODO: /api/trends still emits a legacy naive UTC timestamp string; migrate it to these shared helpers in a separate route cleanup.
    return dt.replace(tzinfo=None).isoformat() + "Z"


def _age_seconds(value, *, now: Optional[datetime] = None) -> Optional[float]:
    dt = _coerce_utc_datetime(value)
    if dt is None:
        return None
    current = _coerce_utc_datetime(now) or _utc_now()
    return max(0, (current - dt).total_seconds())


def _cache_record(name: str, outcome: str) -> None:
    with _cache_lock:
        _cache_stats[f"{name}_{outcome}"] += 1
    logger.debug("[cache] %s %s", name, outcome)


def _cache_record_value(name: str, value: float) -> None:
    with _cache_lock:
        _cache_stats[f"{name}_sum"] += float(value)
        _cache_stats[f"{name}_count"] += 1


def _cache_snapshot(reset: bool = True) -> dict:
    with _cache_lock:
        snapshot = dict(_cache_stats)
        if reset:
            _cache_stats.clear()
    return snapshot


def _cache_average(stats: Optional[dict], name: str) -> Optional[float]:
    stats = stats or {}
    count = float(stats.get(f"{name}_count", 0) or 0)
    if count <= 0:
        return None
    return round(float(stats.get(f"{name}_sum", 0) or 0) / count, 1)


def _cache_hit_rate(cache_stats: Optional[dict]) -> Optional[float]:
    stats = cache_stats or {}
    hits = int(stats.get("prices_hit", 0) or 0)
    misses = int(stats.get("prices_miss", 0) or 0)
    total = hits + misses
    if total <= 0:
        return None
    return round(hits / total, 4)


def _percentile(values: list[float], percentile: float) -> Optional[float]:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return round(clean[0], 1)
    rank = (len(clean) - 1) * float(percentile)
    lower = int(rank)
    upper = min(lower + 1, len(clean) - 1)
    weight = rank - lower
    return round(clean[lower] * (1 - weight) + clean[upper] * weight, 1)


def _process_memory_mb() -> Optional[float]:
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss = float(usage.ru_maxrss)
        if rss <= 0:
            return None
        # macOS reports bytes; Linux reports KiB.
        if rss > 10_000_000:
            rss = rss / (1024 * 1024)
        else:
            rss = rss / 1024
        return round(rss, 1)
    except Exception:
        return None


def _scan_partial_reasons(
    *,
    attempted: int,
    processed: int,
    tradeability_skipped: int,
    processing_failures: Optional[list[dict]] = None,
    provider_metrics: Optional[dict] = None,
) -> list[dict]:
    reasons = []
    failures = list(processing_failures or [])
    operational_failures = [
        item for item in failures
        if isinstance(item, dict) and str(item.get("reason") or "") != "symbol returned no setup"
    ]
    failure_counts = Counter(str(item.get("reason") or "symbol_processing_failed") for item in operational_failures)
    for reason, count in sorted(failure_counts.items()):
        reasons.append({
            "stage": "strategy_evaluation",
            "reason": reason,
            "count": int(count),
        })

    provider = provider_metrics or {}
    provider_failed = int(provider.get("alpaca_bar_symbols_failed", 0) or 0)
    if provider_failed > 0:
        reasons.append({
            "stage": "market_data",
            "reason": "provider_symbol_fetch_failed",
            "count": provider_failed,
        })
    max_pages = int(provider.get("alpaca_max_pages_exceeded_count", 0) or 0)
    if max_pages > 0:
        reasons.append({
            "stage": "market_data",
            "reason": "max_pages_exceeded",
            "count": max_pages,
        })

    unaccounted = max(
        0,
        int(attempted or 0)
        - int(processed or 0)
        - int(tradeability_skipped or 0)
        - len(failures)
        - provider_failed,
    )
    if unaccounted > 0:
        reasons.append({
            "stage": "strategy_evaluation",
            "reason": "symbol_not_evaluated",
            "count": unaccounted,
        })
    return reasons


def _submit_background_job(key: tuple, fn, *args, **kwargs) -> bool:
    with _background_jobs_lock:
        if key in _background_jobs:
            return False
        _background_jobs.add(key)

    def _run():
        try:
            while _scan_is_active():
                time.sleep(0.25)
            fn(*args, **kwargs)
        except Exception as exc:
            logger.warning("[background] refresh failed for %s: %s", key, exc)
        finally:
            with _background_jobs_lock:
                _background_jobs.discard(key)

    try:
        _background_executor.submit(_run)
    except RuntimeError as exc:
        with _background_jobs_lock:
            _background_jobs.discard(key)
        logger.warning("[background] could not submit refresh for %s: %s", key, exc)
        return False
    return True


def _is_yahoo_rate_limit_error(exc: Exception) -> bool:
    text = str(exc or "").lower()
    return (
        "too many requests" in text
        or "rate limited" in text
        or "yfratelimiterror" in text
    )


def _option_yahoo_backoff_active(now: Optional[datetime] = None) -> bool:
    current = now or datetime.utcnow()
    with _cache_lock:
        until = _option_yahoo_backoff_until
    return until is not None and current < until


def _option_expiration_retry_due(ticker: str, now: Optional[datetime] = None) -> bool:
    ticker = str(ticker or "").upper()
    current = now or datetime.utcnow()
    with _cache_lock:
        cached = _option_chain_cache.get(ticker) or {}
        failure_at = cached.get("expirations_failure_at")
        status = cached.get("expirations_status")
    if status != "unknown" or not failure_at:
        return True
    return current - failure_at >= OPTION_EXPIRATION_FAILURE_RETRY_TTL


def _mark_option_yahoo_rate_limited(now: Optional[datetime] = None) -> None:
    global _option_yahoo_backoff_until
    current = now or datetime.utcnow()
    until = current + OPTION_YAHOO_RATE_LIMIT_COOLDOWN
    with _cache_lock:
        if _option_yahoo_backoff_until is None or until > _option_yahoo_backoff_until:
            _option_yahoo_backoff_until = until
    logger.warning(
        "[options] Yahoo rate limit detected; pausing option expiration fetches for %.0fs",
        OPTION_YAHOO_RATE_LIMIT_COOLDOWN.total_seconds(),
    )


def _periodic_refresh_due(key: str, ttl_seconds: int) -> bool:
    now = time.monotonic()
    with _background_jobs_lock:
        last = _background_last_refresh.get(key, 0)
        if now - last < ttl_seconds:
            return False
        _background_last_refresh[key] = now
        return True


def register_background_periodic_task(key: str, ttl_seconds: int, callback) -> None:
    if not key or ttl_seconds <= 0 or not callable(callback):
        return
    with _background_jobs_lock:
        _background_periodic_tasks[str(key)] = {
            "ttl_seconds": int(ttl_seconds),
            "callback": callback,
        }


def _scan_activity_started() -> None:
    global _active_scan_count
    with _background_jobs_lock:
        _active_scan_count += 1


def _scan_activity_finished() -> None:
    global _active_scan_count
    with _background_jobs_lock:
        _active_scan_count = max(0, _active_scan_count - 1)


def _scan_is_active() -> bool:
    with _background_jobs_lock:
        return _active_scan_count > 0


def _analysis_cache_key(watchlist: Optional[list], discover: bool = False, universe: str = "default") -> tuple:
    universe = str(universe or "default").strip().lower()
    if watchlist is None:
        if discover:
            return ("discover",)
        return ("default",)
    symbols = tuple(sorted(dict.fromkeys([str(t).strip().upper() for t in watchlist if str(t).strip()])))
    if universe == "discovered":
        return ("universe", "discovered", symbols)
    return ("custom", symbols)


def _analysis_cache_label(key: tuple) -> str:
    if key == ("default",):
        return "default"
    if key == ("discover",):
        return "discover"
    if len(key) >= 2 and key[0] == "universe":
        return str(key[1])
    return "custom"


def _analysis_refresh_key(key: tuple) -> tuple:
    return ("analysis_refresh", key)


def _analysis_state_key(key: tuple) -> str:
    if key == ("default",):
        return "default"
    if key == ("discover",):
        return "discover"
    if len(key) >= 2 and key[0] == "universe":
        return ":".join(str(part) for part in key[:2])
    return repr(key)


def _analysis_refresh_snapshot(key: tuple) -> dict:
    state_key = _analysis_state_key(key)
    now = _utc_now()
    with _background_jobs_lock:
        refresh_key = _analysis_refresh_key(key)
        active = refresh_key in _background_jobs
        state = dict(_analysis_refresh_state.get(state_key, {}))
        started_at = _coerce_utc_datetime(state.get("refresh_started_at"))
        if active and started_at:
            duration = (now - started_at).total_seconds()
            if duration > ANALYSIS_REFRESH_TIMEOUT_SECONDS:
                _background_jobs.discard(refresh_key)
                active = False
                state["last_refresh_error"] = f"Refresh exceeded {ANALYSIS_REFRESH_TIMEOUT_SECONDS}s timeout"
                state["last_refresh_duration"] = round(duration, 1)
                state["refresh_timed_out_at"] = now
                _analysis_refresh_state[state_key] = state

    started_at = _coerce_utc_datetime(state.get("refresh_started_at"))
    active_duration = None
    if active and started_at:
        active_duration = round((now - started_at).total_seconds(), 1)

    return {
        "refreshing": active,
        "refresh_job_id": state.get("refresh_job_id"),
        "refresh_started_at": _format_utc_timestamp(started_at) if active and started_at else None,
        "refresh_duration": active_duration,
        "last_refresh_success_at": _format_utc_timestamp(state.get("last_refresh_success_at")),
        "last_refresh_error": state.get("last_refresh_error"),
        "last_refresh_duration": state.get("last_refresh_duration"),
    }


def _increment_counter(counter: dict, key) -> None:
    label = str(key or "unknown").strip() or "unknown"
    counter[label] = counter.get(label, 0) + 1


def _coverage_stage_bucket(row: dict) -> str:
    if not isinstance(row, dict):
        return "unknown"
    trade_stage = str((row.get("trade_eval") or {}).get("trade_stage") or "").strip()
    if trade_stage:
        return trade_stage
    setup_status = str(row.get("setupStatus") or row.get("setup_status") or "").strip()
    if setup_status:
        return setup_status
    entry_status = str(row.get("entryStatus") or "").strip()
    if entry_status:
        return entry_status
    return "unknown"


def _coverage_contract_bucket(row: dict) -> str:
    contract = row.get("best_contract") if isinstance(row, dict) else None
    if not isinstance(contract, dict) or not contract:
        return "contract unavailable"
    source = str(contract.get("source") or "").strip().lower()
    reason = str(contract.get("reason") or contract.get("fallback_reason") or "").strip().lower()
    if contract.get("available") is True:
        return "suggested contract available"
    if source == "fallback":
        return "potential/fallback contract only"
    if contract.get("loading") or source == "loading" or "loading" in reason or "refresh" in reason:
        return "option data unknown or temporarily failed"
    if source in {"", "not_evaluated"}:
        return "contract unavailable"
    if "unavailable" in source or "option" in source:
        return "contract unavailable"
    return "contract unavailable"


def _coverage_provider_failure_distribution(rows: list, scan_meta: dict) -> dict:
    provider_failures = {}
    for reason, count in (scan_meta.get("tradeability_skip_reasons") or {}).items():
        reason_key = str(reason or "").lower()
        if "unknown" in reason_key or "failed" in reason_key or "data" in reason_key:
            provider_failures[str(reason or "unknown")] = count
    for row in rows:
        contract = row.get("best_contract") or {}
        if isinstance(contract, dict) and _coverage_contract_bucket(row) == "option data unknown or temporarily failed":
            _increment_counter(provider_failures, contract.get("reason") or contract.get("source") or "option data unknown")
    return provider_failures


def _coverage_option_plan_diagnostics(rows: list) -> dict:
    metrics = {
        "option_plans_generated": 0,
        "option_plans_unavailable": 0,
        "missing_planned_entry": 0,
        "missing_tp1": 0,
        "invalid_projected_move": 0,
        "expected_hold_fallback_used": 0,
        "strike_rounding_distribution": {},
        "expiration_window_distribution": {},
        "confidence_distribution": {},
    }
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        plan = row.get("option_plan") or {}
        if not isinstance(plan, dict):
            metrics["option_plans_unavailable"] += 1
            continue
        if plan.get("available") is True:
            metrics["option_plans_generated"] += 1
            rounding = str(plan.get("strike_rounding_increment") or "unknown")
            _increment_counter(metrics["strike_rounding_distribution"], rounding)
            expiration = (plan.get("suggested_expiration") or {}).get("label") or "unknown"
            _increment_counter(metrics["expiration_window_distribution"], expiration)
            confidence = (plan.get("confidence") or {}).get("label") or "unknown"
            _increment_counter(metrics["confidence_distribution"], confidence)
            if (plan.get("expected_hold") or {}).get("fallback_used"):
                metrics["expected_hold_fallback_used"] += 1
        else:
            metrics["option_plans_unavailable"] += 1
            reason = str(plan.get("reason") or plan.get("unavailable_reason") or "").strip().lower()
            if "planned entry" in reason:
                metrics["missing_planned_entry"] += 1
            elif "tp1" in reason or "target" in reason:
                metrics["missing_tp1"] += 1
            elif "projected move" in reason:
                metrics["invalid_projected_move"] += 1
    metrics["strike_rounding_distribution"] = dict(sorted(metrics["strike_rounding_distribution"].items()))
    metrics["expiration_window_distribution"] = dict(sorted(metrics["expiration_window_distribution"].items()))
    metrics["confidence_distribution"] = dict(sorted(metrics["confidence_distribution"].items()))
    return metrics


def build_discovered_scan_coverage_snapshot(
    rows: list,
    near_miss: list,
    scan_meta: Optional[dict],
    coverage_context: Optional[dict],
) -> dict:
    scan_meta = scan_meta or {}
    coverage_context = coverage_context or {}
    all_rows = [item for item in [*(rows or []), *(near_miss or [])] if isinstance(item, dict)]
    stage_distribution = {}
    grade_distribution = {}
    contract_distribution = {}
    blocker_distribution = {}
    trade_stage_distribution = {}
    entry_status_distribution = {}
    setup_status_distribution = {}

    for row in all_rows:
        _increment_counter(stage_distribution, _coverage_stage_bucket(row))
        _increment_counter(grade_distribution, row.get("setupGrade") or "unknown")
        _increment_counter(contract_distribution, _coverage_contract_bucket(row))
        _increment_counter(trade_stage_distribution, (row.get("trade_eval") or {}).get("trade_stage") or "unknown")
        _increment_counter(entry_status_distribution, row.get("entryStatus") or "unknown")
        _increment_counter(setup_status_distribution, row.get("setupStatus") or row.get("setup_status") or "unknown")
        for reason in (row.get("trade_eval") or {}).get("no_trade_reasons") or []:
            _increment_counter(blocker_distribution, reason)
        entry_status = str(row.get("entryStatus") or "").strip()
        if entry_status == "Too Far":
            _increment_counter(blocker_distribution, "Too Far")
        contract = row.get("best_contract") or {}
        if isinstance(contract, dict) and contract.get("available") is not True:
            reason = contract.get("reason") or contract.get("fallback_reason")
            if reason:
                _increment_counter(blocker_distribution, reason)

    for reason, count in (scan_meta.get("tradeability_skip_reasons") or {}).items():
        blocker_distribution[str(reason or "unknown")] = blocker_distribution.get(str(reason or "unknown"), 0) + int(count or 0)

    requested = scan_meta.get("configured_universe_count")
    processed = scan_meta.get("symbols_successfully_processed", len(all_rows))
    with_setup = scan_meta.get("symbols_with_setup", processed)
    without_setup = scan_meta.get("symbols_without_setup")
    if without_setup is None:
        without_setup = max(0, int(scan_meta.get("no_setup_or_failed_count", 0) or 0) - int(scan_meta.get("symbols_operationally_failed", 0) or 0))
    skipped = scan_meta.get("tradeability_skipped", scan_meta.get("symbols_skipped", 0))
    failed = scan_meta.get("symbols_operationally_failed")
    if failed is None:
        failed = scan_meta.get("symbols_failed", 0)
    not_evaluated = scan_meta.get("symbols_not_evaluated", 0)
    terminally_evaluated = scan_meta.get("symbols_terminally_evaluated")
    if terminally_evaluated is None:
        terminally_evaluated = int(processed or 0) + int(without_setup or 0) + int(skipped or 0)
    returned = len(all_rows)
    cache_stats = scan_meta.get("cache_stats") or {}
    partial_reasons = list(scan_meta.get("partial_result_reasons") or [])
    if not partial_reasons and scan_meta.get("partial_result") and failed:
        partial_reasons = [{
            "stage": "strategy_evaluation",
            "reason": "symbol_processing_failed",
            "count": int(failed or 0),
        }]
    partial = bool(partial_reasons or scan_meta.get("partial_result"))
    completed_at = _coerce_utc_datetime(scan_meta.get("scan_completed_at")) or _utc_now()
    snapshot = {
        "generated_at": _format_utc_timestamp(completed_at),
        "discovery": coverage_context.get("discovery") or {},
        "scan": {
            "universe_source": coverage_context.get("universe_source") or "discovered",
            "universe_generated_at": _format_utc_timestamp(coverage_context.get("universe_generated_at")),
            "universe_symbol_count": coverage_context.get("universe_symbol_count"),
            "symbols_requested": requested,
            "symbols_processed": processed,
            "symbols_returned": returned,
            "symbols_terminally_evaluated": terminally_evaluated,
            "symbols_with_setup": with_setup,
            "symbols_without_setup": without_setup,
            "symbols_intentionally_rejected": scan_meta.get("symbols_intentionally_rejected", skipped),
            "symbols_operationally_failed": failed,
            "symbols_not_evaluated": not_evaluated,
            "evaluation_coverage": scan_meta.get("evaluation_coverage"),
            "evaluation_coverage_percent": scan_meta.get("evaluation_coverage_percent"),
            "result_yield": scan_meta.get("result_yield"),
            "result_yield_percent": scan_meta.get("result_yield_percent"),
            "symbols_failed": failed,
            "symbols_skipped": skipped,
            "scan_started_at": _format_utc_timestamp(scan_meta.get("scan_started_at")),
            "scan_completed_at": _format_utc_timestamp(completed_at),
            "scan_duration_ms": scan_meta.get("scan_duration_ms"),
            "partial_result": partial,
            "partial_result_reasons": partial_reasons,
            "partial_result_reason": scan_meta.get("partial_result_reason"),
            "performance": scan_meta.get("performance") or {},
        },
        "stage_distribution": dict(sorted(stage_distribution.items())),
        "canonical_field_distributions": {
            "trade_stage": dict(sorted(trade_stage_distribution.items())),
            "entry_status": dict(sorted(entry_status_distribution.items())),
            "setup_status": dict(sorted(setup_status_distribution.items())),
        },
        "grade_distribution": {
            "A": grade_distribution.get("A", 0),
            "B": grade_distribution.get("B", 0),
            "C": grade_distribution.get("C", 0),
            "unknown": grade_distribution.get("unknown", 0),
        },
        "contract_distribution": {
            "suggested contract available": contract_distribution.get("suggested contract available", 0),
            "potential/fallback contract only": contract_distribution.get("potential/fallback contract only", 0),
            "contract unavailable": contract_distribution.get("contract unavailable", 0),
            "option data unknown or temporarily failed": contract_distribution.get("option data unknown or temporarily failed", 0),
        },
        "option_plan_diagnostics": _coverage_option_plan_diagnostics(all_rows),
        "blocker_distribution": dict(sorted(blocker_distribution.items())),
        "provider_failures": dict(sorted(_coverage_provider_failure_distribution(all_rows, scan_meta).items())),
        "provider_diagnostics": {
            **(scan_meta.get("provider_metrics") or {}),
            "bar_cache_hits": cache_stats.get("prices_hit", 0),
            "bar_cache_misses": cache_stats.get("prices_miss", 0),
            "option_eligibility_from_discovery": scan_meta.get("option_eligibility_from_discovery", 0),
            "yahoo_expiration_requests": cache_stats.get("api_option_expirations_call", 0),
            "yahoo_expiration_unknown": (scan_meta.get("tradeability_skip_reasons") or {}).get("options data unknown", 0),
            "live_chain_requests": cache_stats.get("api_option_chain_call", 0),
            "live_chain_failures": sum(
                1
                for row in all_rows
                if _coverage_contract_bucket(row) == "option data unknown or temporarily failed"
            ),
        },
    }
    return snapshot


def _store_coverage_baseline_snapshot(snapshot: dict) -> None:
    global _coverage_baseline_snapshot
    with _coverage_baseline_lock:
        _coverage_baseline_snapshot = dict(snapshot)


def coverage_baseline_snapshot() -> Optional[dict]:
    with _coverage_baseline_lock:
        if _coverage_baseline_snapshot is None:
            return None
        return dict(_coverage_baseline_snapshot)


def _mark_analysis_refresh_started(key: tuple, job_id: str) -> None:
    with _background_jobs_lock:
        prior = _analysis_refresh_state.get(_analysis_state_key(key), {})
        _analysis_refresh_state[_analysis_state_key(key)] = {
            **prior,
            "refresh_job_id": job_id,
            "refresh_started_at": _utc_now(),
            "last_refresh_error": None,
            "last_refresh_duration": None,
        }


def _mark_analysis_refresh_finished(key: tuple, started: float, error: Optional[Exception] = None) -> None:
    duration = round(time.perf_counter() - started, 1)
    with _background_jobs_lock:
        state = _analysis_refresh_state.setdefault(_analysis_state_key(key), {})
        state["last_refresh_duration"] = duration
        if error is None:
            state["last_refresh_success_at"] = _utc_now()
            state["last_refresh_error"] = None
        else:
            state["last_refresh_error"] = str(error)


def _submit_analysis_refresh(
    key: tuple,
    watchlist: Optional[list],
    reason: str = "background",
    discover: bool = False,
    max_symbols: Optional[int] = 200,
    coverage_context: Optional[dict] = None,
    trusted_options_symbols: Optional[set[str]] = None,
) -> bool:
    """Submit an analysis refresh job.

    Returns True only when a new job was accepted by the background executor.
    Returns False when the refresh is already active or executor submission fails.
    """
    refresh_key = _analysis_refresh_key(key)
    snapshot = _analysis_refresh_snapshot(key)
    if snapshot.get("refreshing"):
        return False
    job_id = f"{_analysis_state_key(key)}:{int(time.time())}:{reason}"
    submitted = _submit_background_job(
        refresh_key,
        _refresh_analysis_cache,
        key,
        watchlist,
        job_id,
        discover,
        max_symbols,
        coverage_context,
        trusted_options_symbols,
    )
    if submitted:
        _mark_analysis_refresh_started(key, job_id)
    return submitted


def _analysis_cache_meta(key: tuple, cached: dict, refreshing: bool) -> dict:
    generated_at = cached.get("generated_at")
    age_seconds = _age_seconds(generated_at)
    refresh_snapshot = _analysis_refresh_snapshot(key)
    scan_meta = cached.get("scan_meta") or {}
    rows = cached.get("rows") or []
    near_miss = cached.get("near_miss") or []
    return {
        "cache": "hit",
        "generated_at": _format_utc_timestamp(generated_at),
        "age_seconds": round(age_seconds, 1) if age_seconds is not None else None,
        "stale": age_seconds is not None and age_seconds > ANALYSIS_CACHE_STALE_SECONDS,
        "refreshing": bool(refreshing and refresh_snapshot.get("refreshing")),
        "cache_key": _analysis_cache_label(key),
        "strategy_version": STOCK_SCANNER_STRATEGY_VERSION,
        "strategy_baseline": STOCK_SCANNER_STRATEGY_BASELINE_COMMIT,
        "configured_universe_count": scan_meta.get("configured_universe_count"),
        "symbols_attempted": scan_meta.get("symbols_attempted"),
        "symbols_successfully_processed": scan_meta.get("symbols_successfully_processed", len(rows) + len(near_miss)),
        "symbols_terminally_evaluated": scan_meta.get("symbols_terminally_evaluated"),
        "symbols_with_setup": scan_meta.get("symbols_with_setup"),
        "symbols_without_setup": scan_meta.get("symbols_without_setup"),
        "symbols_intentionally_rejected": scan_meta.get("symbols_intentionally_rejected"),
        "symbols_operationally_failed": scan_meta.get("symbols_operationally_failed"),
        "symbols_not_evaluated": scan_meta.get("symbols_not_evaluated"),
        "evaluation_coverage": scan_meta.get("evaluation_coverage"),
        "evaluation_coverage_percent": scan_meta.get("evaluation_coverage_percent"),
        "result_yield": scan_meta.get("result_yield"),
        "result_yield_percent": scan_meta.get("result_yield_percent"),
        "symbols_omitted_or_rejected": scan_meta.get("symbols_omitted_or_rejected"),
        "symbols_failed": scan_meta.get("symbols_failed"),
        "symbols_skipped": scan_meta.get("symbols_skipped"),
        "tradeability_skipped": scan_meta.get("tradeability_skipped"),
        "tradeability_skip_reasons": scan_meta.get("tradeability_skip_reasons", {}),
        "partial_result": scan_meta.get("partial_result"),
        "partial_result_reasons": scan_meta.get("partial_result_reasons", []),
        "partial_result_reason": scan_meta.get("partial_result_reason"),
        "scan_duration_ms": scan_meta.get("scan_duration_ms"),
        "scan_started_at": scan_meta.get("scan_started_at"),
        "scan_completed_at": scan_meta.get("scan_completed_at"),
        "symbols_per_second": (scan_meta.get("performance") or {}).get("symbols_per_second"),
        "provider_metrics": scan_meta.get("provider_metrics", {}),
        "cache_stats": scan_meta.get("cache_stats", {}),
        "performance": scan_meta.get("performance", {}),
        "processed_rows": len(rows) + len(near_miss),
        "qualified_rows": len(rows),
        "near_miss_rows": len(near_miss),
        **refresh_snapshot,
    }


def analysis_cache_status(watchlist: Optional[list] = None, *, discover: bool = False, universe: str = "default") -> dict:
    key = _analysis_cache_key(watchlist, discover=discover, universe=universe)
    with _cache_lock:
        cached = _analysis_cache.get(key)
    refresh_snapshot = _analysis_refresh_snapshot(key)
    refreshing = bool(refresh_snapshot.get("refreshing"))
    if cached:
        meta = _analysis_cache_meta(key, cached, refreshing)
        status = "stale" if meta.get("stale") else "fresh"
        return {**meta, "status": status, "has_cache": True}
    return {
        "cache": "miss",
        "generated_at": None,
        "age_seconds": None,
        "stale": True,
        "refreshing": refreshing,
        "cache_key": _analysis_cache_label(key),
        "status": "warming",
        "has_cache": False,
        "strategy_version": STOCK_SCANNER_STRATEGY_VERSION,
        "strategy_baseline": STOCK_SCANNER_STRATEGY_BASELINE_COMMIT,
        **refresh_snapshot,
    }


def analysis_cache_snapshot(watchlist: Optional[list] = None, *, discover: bool = False, universe: str = "default") -> Optional[dict]:
    key = _analysis_cache_key(watchlist, discover=discover, universe=universe)
    with _cache_lock:
        cached = _analysis_cache.get(key)
        if not cached:
            return None
        return {
            "key": key,
            "rows": [dict(row) for row in cached.get("rows") or [] if isinstance(row, dict)],
            "near_miss": [dict(row) for row in cached.get("near_miss") or [] if isinstance(row, dict)],
            "generated_at": cached.get("generated_at"),
            "scan_meta": dict(cached.get("scan_meta") or {}),
        }


def _store_analysis_cache(key: tuple, rows: list, near_miss: list, scan_meta: Optional[dict] = None) -> dict:
    cached = {
        "rows": rows,
        "near_miss": near_miss,
        "generated_at": _utc_now(),
        "scan_meta": scan_meta or {},
    }
    with _cache_lock:
        _analysis_cache[key] = cached
    return cached


def _hydrate_best_contracts_from_cache(rows: list) -> list:
    hydrated = []
    with _cache_lock:
        best_contract_cache = dict(_best_contract_cache)
    for row in rows or []:
        if not isinstance(row, dict):
            hydrated.append(row)
            continue
        item = dict(row)
        contract = item.get("best_contract") or {}
        source = str(contract.get("source") or "").lower()
        is_unresolved = (
            not contract
            or bool(contract.get("loading"))
            or source in {"", "loading", "not_evaluated"}
        )
        if is_unresolved:
            cache_key = _best_contract_cache_key(item.get("ticker") or "", item.get("direction") or "", item.get("entry") or 0)
            cached_contract = (best_contract_cache.get(cache_key) or {}).get("data")
            if cached_contract and str(cached_contract.get("source") or "").lower() != "loading":
                item["best_contract"] = dict(cached_contract)
        hydrated.append(item)
    return hydrated


def _hydrate_earnings_from_cache(rows: list) -> list:
    hydrated = []
    with _cache_lock:
        earnings_cache = dict(_earnings_cache)
    for row in rows or []:
        if not isinstance(row, dict):
            hydrated.append(row)
            continue
        item = dict(row)
        earnings = item.get("earnings") if isinstance(item.get("earnings"), dict) else {}
        is_unresolved = (
            not earnings
            or _earnings_is_loading(earnings)
            or not _earnings_has_date(earnings)
        )
        if is_unresolved:
            ticker = str(item.get("ticker") or "").upper()
            cached_earnings = (earnings_cache.get(ticker) or {}).get("data")
            if cached_earnings and not _earnings_is_loading(cached_earnings):
                item["earnings"] = dict(cached_earnings)
        hydrated.append(item)
    return hydrated


def _hydrate_scan_rows_from_cache(rows: list) -> list:
    return _hydrate_earnings_from_cache(_hydrate_best_contracts_from_cache(rows))


def _refresh_analysis_cache(
    key: tuple,
    watchlist: Optional[list],
    job_id: Optional[str] = None,
    discover: bool = False,
    max_symbols: Optional[int] = 200,
    coverage_context: Optional[dict] = None,
    trusted_options_symbols: Optional[set[str]] = None,
) -> None:
    started = time.perf_counter()
    try:
        logger.info("[analysis refresh] start key=%s job=%s", _analysis_state_key(key), job_id)
        rows, near_miss, scan_meta = scan_all(
            watchlist,
            max_workers=BACKGROUND_ANALYSIS_SCAN_WORKERS,
            discover=discover,
            max_symbols=max_symbols,
            trusted_options_symbols=trusted_options_symbols,
        )
        _store_analysis_cache(key, rows, near_miss, scan_meta)
        if isinstance(key, tuple) and len(key) >= 2 and key[0] == "universe" and key[1] == "discovered":
            coverage = build_discovered_scan_coverage_snapshot(rows, near_miss, scan_meta, coverage_context)
            _store_coverage_baseline_snapshot(coverage)
            scan = coverage.get("scan") or {}
            stages = coverage.get("stage_distribution") or {}
            logger.info(
                "coverage.scan.complete duration_ms=%s symbols_requested=%s symbols_processed=%s symbols_returned=%s symbols_failed=%s symbols_skipped=%s a_plus_ready=%s b_plus_tradeable=%s range_no_trade=%s building_watchlist=%s failed=%s partial_result=%s",
                scan.get("scan_duration_ms"),
                scan.get("symbols_requested"),
                scan.get("symbols_processed"),
                scan.get("symbols_returned"),
                scan.get("symbols_failed"),
                scan.get("symbols_skipped"),
                stages.get("A+ READY", 0),
                stages.get("B+ TRADEABLE", 0),
                stages.get("RANGE / NO TRADE", 0),
                stages.get("BUILDING / WATCHLIST", 0),
                scan.get("symbols_failed"),
                scan.get("partial_result"),
            )
        _mark_analysis_refresh_finished(key, started)
        logger.info("[analysis refresh] complete key=%s job=%s rows=%s near=%s", _analysis_state_key(key), job_id, len(rows), len(near_miss))
    except Exception as exc:
        _mark_analysis_refresh_finished(key, started, exc)
        logger.exception("[analysis refresh] failed key=%s job=%s", _analysis_state_key(key), job_id)
        raise


def get_finviz_watchlist() -> list:
    now = datetime.utcnow()
    cached_at = _finviz_cache.get("fetched_at")
    if cached_at and now - cached_at < FINVIZ_CACHE_TTL:
        return list(dict.fromkeys([*_finviz_cache.get("tickers", []), *WATCHLIST]))[:200]

    try:
        from finvizfinance.screener.overview import Overview

        screener = Overview()
        filters = {
            "Market Cap.": "+Mid (over $2bln)",
            "Average Volume": "Over 1M",
            "Price": "Over $10",
            "Country": "USA",
            "Industry": "Stocks only (ex-Funds)",
        }
        try:
            screener.set_filter(filters_dict=filters)
        except Exception:
            filters["Type"] = "Stock only"
            screener.set_filter(filters_dict=filters)

        df = screener.screener_view()
        tickers = []
        if df is not None and not df.empty and "Ticker" in df.columns:
            tickers = [
                str(t).strip().upper()
                for t in df["Ticker"].head(150).tolist()
                if str(t).strip()
            ]
        _finviz_cache["tickers"] = tickers[:150]
        _finviz_cache["fetched_at"] = now
        return list(dict.fromkeys([*tickers, *WATCHLIST]))[:200]
    except Exception as e:
        logger.error(f"[finviz] watchlist fetch failed: {e}")
        return list(WATCHLIST)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        ohlcv = {"open", "high", "low", "close", "adj close", "volume"}
        best_level = 0
        best_score = -1
        for level in range(df.columns.nlevels):
            values = [str(value).strip().lower() for value in df.columns.get_level_values(level)]
            score = sum(1 for value in values if value in ohlcv)
            if score > best_score:
                best_score = score
                best_level = level
        df.columns = df.columns.get_level_values(best_level)
    return df


def _compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    prev_close = df["Close"].astype(float).shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])


def _next_friday(days_out: int = 37) -> datetime:
    target = datetime.now() + timedelta(days=days_out)
    days_to_friday = (4 - target.weekday()) % 7
    return target + timedelta(days=days_to_friday)


def _fallback_option(ticker: str, direction: str, entry: float, reason: str = "option chain unavailable") -> dict:
    expiry = _next_friday(37)
    dte = (expiry - datetime.now()).days
    if entry < 20:
        increment = 0.5
    elif entry < 100:
        increment = 1.0
    elif entry < 500:
        increment = 5.0
    else:
        increment = 10.0
    atm_strike = round(entry / increment) * increment
    option_type = "CALL" if direction == "LONG" else "PUT"
    return {
        "type": option_type,
        "strike": atm_strike,
        "expiry": expiry.strftime("%Y-%m-%d"),
        "dte": dte,
        "symbol": f"{ticker} {expiry.strftime('%b %d').upper()} ${atm_strike:.0f} {'C' if option_type == 'CALL' else 'P'}",
        "source": "fallback",
        "fallback_reason": reason,
    }


def _option_expirations_for_ticker(ticker: str) -> list:
    ticker = str(ticker or "").upper()
    now = datetime.utcnow()
    with _cache_lock:
        cached = _option_chain_cache.get(ticker)
        if cached:
            expirations = list(cached.get("expirations", []))
            status = cached.get("expirations_status") or ("ready" if expirations else "unknown")
            fetched_at = cached.get("expirations_fetched_at") or cached.get("fetched_at", now)
            age = now - fetched_at
            if status in {"ready", "empty"} and age < OPTION_EXPIRATION_CACHE_TTL:
                _cache_record("option_chains", "hit")
                return expirations
            if status == "ready" and expirations:
                _cache_record("option_chains", "stale")
                _submit_background_job(("option_expirations", ticker), _refresh_option_expirations, ticker)
                return expirations
        _cache_record("option_chains", "miss")

    return _fetch_option_expirations(ticker)


def _cached_option_expirations_for_ticker(ticker: str) -> tuple[bool, list]:
    ticker = str(ticker or "").upper()
    with _cache_lock:
        cached = _option_chain_cache.get(ticker)
        if not cached:
            return False, []
        status = cached.get("expirations_status") or ("ready" if cached.get("expirations") else "unknown")
        if status == "unknown":
            return False, []
        return True, list(cached.get("expirations", []))


def _fetch_option_expirations(ticker: str) -> list:
    ticker = str(ticker or "").upper()
    now = datetime.utcnow()

    def _download_expirations_once() -> list:
        _cache_record("api_option_expirations", "call")
        return list(yf.Ticker(ticker).options or [])

    with _cache_lock:
        existing = _option_chain_cache.get(ticker) or {}
        existing_expirations = list(existing.get("expirations", []))
        existing_status = existing.get("expirations_status") or ("ready" if existing_expirations else "unknown")

    if _option_yahoo_backoff_active(now):
        with _cache_lock:
            cached = _option_chain_cache.setdefault(ticker, {"fetched_at": now, "expirations": [], "chains": {}})
            cached.setdefault("chains", {})
            if existing_status == "ready" and existing_expirations:
                cached["expirations"] = existing_expirations
                cached["expirations_status"] = "ready"
                cached.setdefault(
                    "expirations_fetched_at",
                    existing.get("expirations_fetched_at", existing.get("fetched_at", now)),
                )
                _cache_record("option_expirations", "preserve")
                return existing_expirations
            cached["expirations_status"] = "unknown"
            cached["expirations_failure_at"] = now
            cached["expirations_failure_reason"] = "rate_limited_backoff"
        _cache_record("option_expirations", "backoff")
        return []

    expirations = []
    try:
        expirations = _download_expirations_once()
        if not expirations:
            logger.info(
                "[options] empty expiration response for %s; retrying once after %.1fs",
                ticker,
                OPTION_EXPIRATION_EMPTY_RETRY_DELAY_SECONDS,
            )
            time.sleep(OPTION_EXPIRATION_EMPTY_RETRY_DELAY_SECONDS)
            expirations = _download_expirations_once()
    except Exception as e:
        logger.warning(f"[options] expiration fetch failed for {ticker}: {e}")
        rate_limited = _is_yahoo_rate_limit_error(e)
        if rate_limited:
            _mark_option_yahoo_rate_limited(now)
        with _cache_lock:
            cached = _option_chain_cache.setdefault(ticker, {"fetched_at": now, "expirations": [], "chains": {}})
            cached.setdefault("chains", {})
            cached["expirations_failure_at"] = now
            cached["expirations_failure_reason"] = "rate_limited" if rate_limited else "fetch_error"
            if existing_status == "ready" and existing_expirations:
                cached["expirations"] = existing_expirations
                cached["expirations_status"] = "ready"
                cached.setdefault(
                    "expirations_fetched_at",
                    existing.get("expirations_fetched_at", existing.get("fetched_at", now)),
                )
                _cache_record("option_expirations", "preserve")
                return existing_expirations
            cached["expirations_status"] = "unknown"
        return []

    with _cache_lock:
        cached = _option_chain_cache.setdefault(ticker, {"fetched_at": now, "expirations": [], "chains": {}})
        cached["expirations_fetched_at"] = now
        cached["expirations"] = expirations
        cached["expirations_status"] = "ready" if expirations else "empty"
        cached.pop("expirations_failure_at", None)
        cached.pop("expirations_failure_reason", None)
        cached.setdefault("chains", {})
    return expirations


def _refresh_option_expirations(ticker: str) -> None:
    _fetch_option_expirations(ticker)


def _select_expiration(expirations: list, target_days: int = 37) -> Optional[str]:
    today = datetime.now().date()
    parsed = []
    for expiry in expirations:
        try:
            expiry_date = datetime.strptime(str(expiry), "%Y-%m-%d").date()
        except Exception:
            continue
        dte = (expiry_date - today).days
        if dte >= 7:
            parsed.append((expiry, dte))
    if not parsed:
        return None
    return min(parsed, key=lambda item: (abs(item[1] - target_days), item[1]))[0]


def _option_chain_for_ticker(ticker: str, expiry: str):
    ticker = str(ticker or "").upper()
    now = datetime.utcnow()
    with _cache_lock:
        cached = _option_chain_cache.get(ticker)
        if cached:
            chains = cached.setdefault("chains", {})
            if expiry in chains:
                chain = chains[expiry]
                age = now - cached.get("fetched_at", now)
                if age < OPTION_CHAIN_CACHE_TTL:
                    _cache_record("option_chains", "hit")
                    return chain
                _cache_record("option_chains", "stale")
                _submit_background_job(("option_chain", ticker, expiry), _refresh_option_chain, ticker, expiry)
                return chain
        _cache_record("option_chains", "miss")

    return _fetch_option_chain(ticker, expiry)


def _cached_option_chain_for_ticker(ticker: str, expiry: str):
    ticker = str(ticker or "").upper()
    with _cache_lock:
        cached = _option_chain_cache.get(ticker)
        if not cached:
            return False, None
        chain = cached.setdefault("chains", {}).get(expiry)
        return chain is not None, chain


def _fetch_option_chain(ticker: str, expiry: str):
    ticker = str(ticker or "").upper()
    now = datetime.utcnow()

    try:
        with _option_chain_fetch_semaphore:
            _cache_record("api_option_chain", "call")
            chain = yf.Ticker(ticker).option_chain(expiry)
    except Exception as e:
        logger.warning(f"[options] chain fetch failed for {ticker} {expiry}: {e}")
        return None

    with _cache_lock:
        cached = _option_chain_cache.setdefault(ticker, {"fetched_at": now, "expirations": [], "chains": {}})
        cached["fetched_at"] = now
        cached.setdefault("chains", {})[expiry] = chain
    return chain


def _refresh_option_chain(ticker: str, expiry: str) -> None:
    _fetch_option_chain(ticker, expiry)


def _latest_close(df: Optional[pd.DataFrame]) -> Optional[float]:
    if df is None or df.empty or "Close" not in df.columns:
        return None
    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if close.empty:
        return None
    return float(close.iloc[-1])


def _average_volume(df: Optional[pd.DataFrame], lookback: int = 30) -> Optional[float]:
    if df is None or df.empty or "Volume" not in df.columns:
        return None
    volume = pd.to_numeric(df["Volume"], errors="coerce").dropna()
    if volume.empty:
        return None
    return float(volume.tail(max(int(lookback or 1), 1)).mean())


def _is_known_etf(ticker: str) -> bool:
    return ticker in NO_EARNINGS_SYMBOLS


def _stock_universe_skip_reason(ticker: str, daily_df: Optional[pd.DataFrame], trusted_options_eligible: bool = False) -> Optional[str]:
    config = STOCK_TRADEABILITY_FILTER
    if not config.get("enabled", True):
        return None

    ticker = str(ticker or "").upper()
    allowlist = set(config.get("allowlist") or set())
    blocklist = set(config.get("blocklist") or set())
    if ticker in blocklist:
        return "blocked symbol"

    if config.get("exclude_non_major_etfs", True) and _is_known_etf(ticker) and ticker not in MAJOR_LIQUID_ETFS:
        return "non-major ETF"

    price = _latest_close(daily_df)
    if price is None:
        return "no price data"

    min_avg_volume = float(config.get("min_avg_volume") or 0)
    if min_avg_volume > 0:
        avg_volume = _average_volume(daily_df, int(config.get("avg_volume_lookback") or 30))
        if (avg_volume is None or avg_volume < min_avg_volume) and ticker not in allowlist:
            return "low liquidity"

    min_expirations = int(config.get("min_option_expirations") or 0)
    if min_expirations > 0 and not trusted_options_eligible:
        known, expirations = _cached_option_expirations_for_ticker(ticker)
        if known:
            if not expirations:
                return "no options"
            if len(expirations) < min_expirations:
                return "thin options chain"
        else:
            if not _option_yahoo_backoff_active() and _option_expiration_retry_due(ticker):
                _submit_background_job(("option_expirations_prefilter", ticker), _refresh_option_expirations, ticker)

    return None


def _prefilter_stock_universe(watchlist: list, daily_data: dict, trusted_options_symbols: Optional[set[str]] = None) -> tuple[list, list]:
    trusted_options_symbols = {str(symbol or "").strip().upper() for symbol in (trusted_options_symbols or set()) if str(symbol or "").strip()}
    accepted = []
    skipped = []
    for ticker in watchlist:
        symbol = str(ticker or "").strip().upper()
        if not symbol:
            continue
        reason = _stock_universe_skip_reason(
            symbol,
            daily_data.get(symbol),
            trusted_options_eligible=symbol in trusted_options_symbols,
        )
        if reason:
            skipped.append({"ticker": symbol, "reason": reason})
        else:
            accepted.append(symbol)
    return accepted, skipped


def _safe_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value):
    try:
        if pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def _coerce_earnings_date(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and value:
        for item in value:
            parsed = _coerce_earnings_date(item)
            if parsed:
                return parsed
        return None
    if isinstance(value, pd.Series):
        for item in value.dropna().tolist():
            parsed = _coerce_earnings_date(item)
            if parsed:
                return parsed
        return None
    if isinstance(value, pd.DataFrame):
        for column in ["Earnings Date", "Earnings", "Date"]:
            if column in value.columns:
                parsed = _coerce_earnings_date(value[column])
                if parsed:
                    return parsed
        return None
    try:
        timestamp = pd.to_datetime(value, errors="coerce")
    except Exception:
        return None
    if pd.isna(timestamp):
        return None
    try:
        return timestamp.date()
    except Exception:
        return None


def _extract_earnings_from_calendar(calendar):
    if calendar is None:
        return None
    if isinstance(calendar, dict):
        for key in ["Earnings Date", "Earnings", "Next Earnings Date"]:
            parsed = _coerce_earnings_date(calendar.get(key))
            if parsed:
                return parsed
        return None
    if isinstance(calendar, pd.DataFrame):
        return _coerce_earnings_date(calendar)
    return None


def _next_future_date(values):
    dates = []
    today = datetime.now().date()
    try:
        for value in list(values):
            parsed = _coerce_earnings_date(value)
            if parsed and parsed >= today:
                dates.append(parsed)
    except Exception:
        return None
    return min(dates) if dates else None


def _earnings_loading_result() -> dict:
    return {
        "loaded": False,
        "loading": True,
        "status": "loading",
        "date": None,
        "days_until": None,
        "source": "background_refresh",
    }


def _earnings_has_date(data: Optional[dict]) -> bool:
    return bool(isinstance(data, dict) and data.get("loaded") and data.get("date"))


def _earnings_is_loading(data: Optional[dict]) -> bool:
    if not isinstance(data, dict):
        return False
    status = str(data.get("status") or "").lower()
    source = str(data.get("source") or "").lower()
    return bool(data.get("loading")) or status in {"loading", "pending", "refreshing"} or source == "background_refresh"


def _submit_earnings_refresh_if_safe(ticker: str, *, reason: str = "miss") -> bool:
    ticker = str(ticker or "").upper()
    if not ticker or ticker in NO_EARNINGS_SYMBOLS:
        return False
    now = datetime.utcnow()
    if _scan_is_active():
        with _cache_lock:
            deferred_until = _earnings_deferred_until.get(ticker)
            if not deferred_until or now >= deferred_until:
                _earnings_deferred_until[ticker] = now + EARNINGS_SCAN_DEFER_TTL
                _cache_record("earnings", "deferred_scan")
        return False
    submitted = _submit_background_job(("earnings", ticker), _refresh_earnings, ticker)
    if submitted:
        _cache_record("earnings", f"submitted_{reason}")
    return submitted


def _earnings_for_ticker(ticker: str, *, allow_fetch: bool = True) -> dict:
    ticker = str(ticker or "").upper()
    now = datetime.utcnow()
    with _cache_lock:
        cached = _earnings_cache.get(ticker)
        if cached:
            cached_data = cached.get("data", {})
            ttl = EARNINGS_CACHE_TTL if cached_data.get("loaded") else EARNINGS_UNAVAILABLE_CACHE_TTL
            if now - cached.get("fetched_at", now) < ttl:
                _cache_record("earnings", "hit")
                return dict(cached_data)
            if cached_data:
                _cache_record("earnings", "stale")
                _submit_earnings_refresh_if_safe(ticker, reason="stale")
                return dict(cached_data)
        _cache_record("earnings", "miss")

    if not allow_fetch:
        _submit_earnings_refresh_if_safe(ticker, reason="miss")
        return _earnings_loading_result()

    return _fetch_earnings_for_ticker(ticker)


def _fetch_earnings_for_ticker(ticker: str) -> dict:
    ticker = str(ticker or "").upper()
    now = datetime.utcnow()

    data = {
        "loaded": False,
        "date": None,
        "days_until": None,
        "source": "unavailable",
    }
    if ticker in NO_EARNINGS_SYMBOLS:
        with _cache_lock:
            _earnings_cache[ticker] = {"fetched_at": now, "data": data}
        return dict(data)

    try:
        _cache_record("api_earnings", "call")
        ticker_obj = yf.Ticker(ticker)
        earnings_date = None
        try:
            earnings_dates = ticker_obj.get_earnings_dates(limit=1)
            if earnings_dates is not None:
                earnings_date = _next_future_date(earnings_dates.index) or _coerce_earnings_date(earnings_dates.index)
        except Exception:
            earnings_date = None
        if not earnings_date:
            earnings_date = _extract_earnings_from_calendar(getattr(ticker_obj, "calendar", None))
        if earnings_date:
            today = datetime.now().date()
            days_until = (earnings_date - today).days
            data = {
                "loaded": True,
                "date": earnings_date.strftime("%Y-%m-%d"),
                "days_until": days_until,
                "source": "yfinance",
            }
    except Exception as e:
        logger.warning(f"[earnings] fetch failed for {ticker}: {e}")

    with _cache_lock:
        existing = _earnings_cache.get(ticker) or {}
        existing_data = existing.get("data") or {}
        existing_age = now - existing.get("fetched_at", now)
        if (
            not _earnings_has_date(data)
            and _earnings_has_date(existing_data)
            and existing_age < EARNINGS_FAILURE_PRESERVE_TTL
        ):
            _cache_record("earnings", "preserve")
            return dict(existing_data)
        _earnings_cache[ticker] = {"fetched_at": now, "data": data}
    return dict(data)


def _refresh_earnings(ticker: str) -> None:
    _fetch_earnings_for_ticker(ticker)


def _option_symbol_from_row(ticker: str, expiry: str, strike: float, option_type: str, row: pd.Series) -> str:
    contract_symbol = row.get("contractSymbol")
    if isinstance(contract_symbol, str) and contract_symbol.strip():
        return contract_symbol
    expiry_dt = datetime.strptime(expiry, "%Y-%m-%d")
    return f"{ticker} {expiry_dt.strftime('%b %d').upper()} ${strike:.0f} {'C' if option_type == 'CALL' else 'P'}"


def _suggest_option(ticker: str, direction: str, entry: float) -> dict:
    option_type = "CALL" if direction == "LONG" else "PUT"
    known_expirations, expirations = _cached_option_expirations_for_ticker(ticker)
    if not known_expirations:
        _submit_background_job(("option_expirations_suggested", str(ticker or "").upper()), _refresh_option_expirations, str(ticker or "").upper())
        return _fallback_option(ticker, direction, entry, "option data loading")
    expiry = _select_expiration(expirations)
    if not expiry:
        return _fallback_option(ticker, direction, entry, "no option expirations returned")

    known_chain, chain = _cached_option_chain_for_ticker(ticker, expiry)
    if not known_chain:
        _submit_background_job(("option_chain_suggested", str(ticker or "").upper(), expiry), _refresh_option_chain, str(ticker or "").upper(), expiry)
        return _fallback_option(ticker, direction, entry, f"option chain loading for {expiry}")
    if chain is None:
        return _fallback_option(ticker, direction, entry, f"option chain unavailable for {expiry}")

    contracts = chain.calls if option_type == "CALL" else chain.puts
    if contracts is None or contracts.empty or "strike" not in contracts.columns:
        return _fallback_option(ticker, direction, entry, f"no {option_type.lower()} contracts returned for {expiry}")

    contracts = contracts.copy()
    contracts["strike_distance"] = (contracts["strike"].astype(float) - float(entry)).abs()
    row = contracts.sort_values(["strike_distance", "strike"]).iloc[0]
    strike = _safe_float(row.get("strike"))
    if strike is None:
        return _fallback_option(ticker, direction, entry, f"selected {option_type.lower()} contract missing strike")

    expiry_date = datetime.strptime(expiry, "%Y-%m-%d").date()
    dte = (expiry_date - datetime.now().date()).days
    bid = _safe_float(row.get("bid"))
    ask = _safe_float(row.get("ask"))
    last_price = _safe_float(row.get("lastPrice"))
    mid = round((bid + ask) / 2, 2) if bid is not None and ask is not None and ask > 0 else None
    mark = mid if mid is not None else last_price
    spread = round(ask - bid, 2) if bid is not None and ask is not None and ask >= bid else None

    return {
        "type": option_type,
        "strike": strike,
        "expiry": expiry,
        "dte": dte,
        "symbol": _option_symbol_from_row(ticker, expiry, strike, option_type, row),
        "bid": bid,
        "ask": ask,
        "last": last_price,
        "mid": mid,
        "mark": mark,
        "spread": spread,
        "open_interest": _safe_int(row.get("openInterest")),
        "volume": _safe_int(row.get("volume")),
        "source": "option_chain",
    }


def _parsed_expirations(expirations: list) -> list:
    today = datetime.now().date()
    parsed = []
    for expiry in expirations:
        try:
            expiry_date = datetime.strptime(str(expiry), "%Y-%m-%d").date()
        except Exception:
            continue
        dte = (expiry_date - today).days
        if dte >= 7:
            parsed.append((str(expiry), dte))
    return parsed


def _price_region(entry: float) -> str:
    value = float(entry or 0)
    if value <= 0:
        return "0"
    bucket = max(value * 0.01, 0.5)
    region = round(value / bucket) * bucket
    return f"{region:.2f}"


def _best_contract_cache_key(ticker: str, direction: str, entry: float) -> tuple:
    return (str(ticker or "").upper(), str(direction or "").upper(), _price_region(entry))


def _unclean_contract(reason: str, source: str = "not_evaluated") -> dict:
    return {
        "available": False,
        "execution": "No Clean Contract",
        "reason": reason,
        "source": source,
    }


def _loading_contract(reason: str = "Options data is refreshing") -> dict:
    return {
        "available": False,
        "execution": "Loading",
        "reason": reason,
        "source": "loading",
        "loading": True,
    }


def _contract_score(contract: dict) -> Optional[float]:
    score = contract.get("score", contract.get("best_score"))
    return _safe_float(score)


def _contract_identity(contract: dict) -> tuple:
    return (
        contract.get("symbol"),
        contract.get("expiry"),
        contract.get("strike"),
        contract.get("type"),
    )


def _apply_contract_stability(cache_key: tuple, fresh: dict, now: datetime) -> dict:
    with _cache_lock:
        cached = _best_contract_cache.get(cache_key)
    if not cached:
        return fresh
    previous = dict(cached.get("data", {}))
    if not previous:
        return fresh
    previous_score = _contract_score(previous)
    fresh_score = _contract_score(fresh)
    previous_execution = previous.get("execution")
    fresh_execution = fresh.get("execution")

    if previous.get("available") and not fresh.get("available"):
        previous["cache"] = "stability_hold"
        previous["stability_reason"] = "kept previous clean contract after transient fresh no-contract result"
        return previous

    same_contract = _contract_identity(previous) == _contract_identity(fresh)
    if (
        same_contract
        and previous_execution == "Excellent"
        and fresh_execution == "Fair"
        and previous_score is not None
        and fresh_score is not None
        and previous_score - fresh_score < 8
    ):
        stable = dict(fresh)
        stable["execution"] = "Excellent"
        stable["cache"] = "stability_hold"
        stable["stability_reason"] = "held Excellent through minor score drift"
        return stable

    if (
        same_contract
        and previous_execution == "Fair"
        and fresh_execution == "Excellent"
        and previous_score is not None
        and fresh_score is not None
        and fresh_score - previous_score < 5
    ):
        stable = dict(fresh)
        stable["execution"] = "Fair"
        stable["cache"] = "stability_hold"
        stable["stability_reason"] = "held Fair until improvement is meaningful"
        return stable

    return fresh


def _store_best_contract(cache_key: tuple, result: dict, now: datetime) -> dict:
    stable = _apply_contract_stability(cache_key, result, now)
    with _cache_lock:
        _best_contract_cache[cache_key] = {"fetched_at": now, "data": stable}
    return dict(stable)


def _score_range(value, bands) -> int:
    if value is None:
        return 0
    for threshold, score in bands:
        if value >= threshold:
            return score
    return 0


def _score_contract_row(row: pd.Series, strike: float, entry: float, dte: int) -> tuple:
    bid = _safe_float(row.get("bid"))
    ask = _safe_float(row.get("ask"))
    last_price = _safe_float(row.get("lastPrice"))
    open_interest = _safe_int(row.get("openInterest")) or 0
    volume = _safe_int(row.get("volume")) or 0
    delta = _safe_float(row.get("delta"))
    mark = round((bid + ask) / 2, 2) if bid is not None and ask is not None and ask > 0 else last_price
    spread = round(ask - bid, 2) if bid is not None and ask is not None and ask >= bid else None
    spread_pct = round((spread / mark) * 100, 1) if spread is not None and mark and mark > 0 else None
    distance_pct = abs(strike - entry) / entry if entry else 1.0

    spread_score = 25 if spread_pct is not None and spread_pct <= 10 else 18 if spread_pct is not None and spread_pct <= 20 else 10 if spread_pct is not None and spread_pct <= 35 else 0
    oi_score = _score_range(open_interest, [(500, 20), (100, 14), (25, 8), (1, 4)])
    volume_score = _score_range(volume, [(100, 15), (20, 10), (1, 5)])
    dte_score = 15 if 21 <= dte <= 45 else 10 if 14 <= dte <= 60 else 5
    delta_score = 10 if delta is not None and 0.25 <= abs(delta) <= 0.65 else 5 if delta is None else 3
    distance_score = 15 if distance_pct <= 0.03 else 10 if distance_pct <= 0.06 else 5 if distance_pct <= 0.10 else 0
    score = spread_score + oi_score + volume_score + dte_score + delta_score + distance_score

    diagnostics = {
        "spread_score": spread_score,
        "open_interest_score": oi_score,
        "volume_score": volume_score,
        "dte_score": dte_score,
        "delta_score": delta_score,
        "distance_score": distance_score,
        "spread_pct": spread_pct,
        "distance_pct": round(distance_pct * 100, 2),
    }
    metrics = {
        "bid": bid,
        "ask": ask,
        "last": last_price,
        "mid": mark,
        "mark": mark,
        "spread": spread,
        "spread_pct": spread_pct,
        "open_interest": open_interest,
        "volume": volume,
        "delta": delta,
    }
    return score, metrics, diagnostics


def _contract_candidate_cost(candidate: dict) -> Optional[float]:
    premium = _safe_float(candidate.get("ask"))
    if premium is None:
        premium = _safe_float(candidate.get("mark"))
    if premium is None:
        premium = _safe_float(candidate.get("mid"))
    if premium is None:
        premium = _safe_float(candidate.get("last"))
    return round(premium * 100, 2) if premium is not None and premium > 0 else None


def _contract_candidate_rejection_reasons(candidate: dict) -> list:
    diagnostics = candidate.get("diagnostics") or {}
    reasons = []
    spread_pct = _safe_float(candidate.get("spread_pct"))
    if spread_pct is None:
        spread_pct = _safe_float(diagnostics.get("spread_pct"))
    distance_pct = _safe_float(diagnostics.get("distance_pct"))
    score = _safe_float(candidate.get("score"))
    dte = _safe_int(candidate.get("dte"))
    delta = _safe_float(candidate.get("delta"))
    open_interest = _safe_int(candidate.get("open_interest")) or 0
    volume = _safe_int(candidate.get("volume")) or 0
    cost = _contract_candidate_cost(candidate)

    if cost is None:
        reasons.append("missing quote")
    if spread_pct is None:
        reasons.append("missing spread")
    elif spread_pct > 35:
        reasons.append("spread above 35%")
    if open_interest < 25 and volume < 1:
        reasons.append("thin open interest and volume")
    if distance_pct is None:
        reasons.append("missing strike-distance diagnostic")
    elif distance_pct > 10:
        reasons.append("strike more than 10% from planned entry")
    if dte is None:
        reasons.append("missing DTE")
    elif dte < 14 or dte > 60:
        reasons.append("DTE outside 14-60 day safeguard")
    if delta is not None and not (0.20 <= abs(delta) <= 0.75):
        reasons.append("delta outside broad audit range")
    if score is None:
        reasons.append("missing contract score")
    elif score < 55:
        reasons.append("score below Fair threshold")
    return reasons


def _contract_quality_classification(candidate: dict) -> str:
    reasons = _contract_candidate_rejection_reasons(candidate)
    score = _safe_float(candidate.get("score")) or 0
    if reasons:
        return "Rejected"
    if score >= 75:
        return "Excellent"
    return "Acceptable"


def _contract_candidate_audit_row(candidate: dict, selected_identity: tuple) -> dict:
    diagnostics = candidate.get("diagnostics") or {}
    identity = _contract_identity(candidate)
    reasons = _contract_candidate_rejection_reasons(candidate)
    spread_pct = _safe_float(candidate.get("spread_pct"))
    if spread_pct is None:
        spread_pct = _safe_float(diagnostics.get("spread_pct"))
    return {
        "symbol": candidate.get("symbol"),
        "type": candidate.get("type"),
        "strike": candidate.get("strike"),
        "expiration": candidate.get("expiry"),
        "expiry": candidate.get("expiry"),
        "dte": candidate.get("dte"),
        "bid": candidate.get("bid"),
        "ask": candidate.get("ask"),
        "mid": candidate.get("mid", candidate.get("mark")),
        "mark": candidate.get("mark", candidate.get("mid")),
        "estimated_contract_cost": _contract_candidate_cost(candidate),
        "delta": candidate.get("delta"),
        "open_interest": candidate.get("open_interest"),
        "volume": candidate.get("volume"),
        "spread": candidate.get("spread"),
        "spread_pct": spread_pct,
        "distance_from_planned_entry_pct": diagnostics.get("distance_pct"),
        "score": candidate.get("score"),
        "quality_classification": _contract_quality_classification(candidate),
        "selected_status": "current_selected" if identity == selected_identity else "not_selected",
        "rejection_reasons": reasons,
    }


def _contract_candidate_audit(ticker: str, candidates: list, selected: dict) -> dict:
    selected_identity = _contract_identity(selected)
    rows = [_contract_candidate_audit_row(candidate, selected_identity) for candidate in candidates]
    acceptable = [
        row for row in rows
        if not row.get("rejection_reasons")
        and _safe_float(row.get("estimated_contract_cost")) is not None
    ]
    current = next((row for row in rows if row.get("selected_status") == "current_selected"), None)
    current_cost = _safe_float((current or {}).get("estimated_contract_cost"))
    best_quality = max(acceptable, key=lambda row: (_safe_float(row.get("score")) or 0, -(_safe_float(row.get("estimated_contract_cost")) or 10**9)), default=current)
    cheaper = [
        row for row in acceptable
        if current_cost is not None
        and (_safe_float(row.get("estimated_contract_cost")) or 10**9) <= current_cost * 0.85
    ]
    balanced = max(cheaper, key=lambda row: (_safe_float(row.get("score")) or 0, -(_safe_float(row.get("estimated_contract_cost")) or 10**9)), default=None)
    lowest = min(acceptable, key=lambda row: _safe_float(row.get("estimated_contract_cost")) or 10**9, default=None)
    potential_savings = None
    if current_cost is not None and lowest:
        lowest_cost = _safe_float(lowest.get("estimated_contract_cost"))
        if lowest_cost is not None and lowest_cost < current_cost:
            potential_savings = round(current_cost - lowest_cost, 2)
    return {
        "ticker": ticker,
        "production_selection_changed": False,
        "candidate_count": len(rows),
        "acceptable_candidate_count": len(acceptable),
        "current_selected_contract": current,
        "best_quality_contract": best_quality,
        "best_balanced_contract": balanced,
        "lowest_cost_acceptable_contract": lowest,
        "potential_savings": potential_savings,
        "rejected_candidates": [row for row in rows if row.get("rejection_reasons")],
        "candidates": rows,
        "minimum_quality_safeguards": [
            "score >= 55",
            "bid/ask spread <= 35%",
            "open interest >= 25 or volume >= 1",
            "strike within 10% of planned entry",
            "DTE between 14 and 60 days",
            "delta between 0.20 and 0.75 when available",
        ],
    }


def _refresh_best_contract(ticker: str, direction: str, entry: float) -> None:
    _best_contract(ticker, direction, entry, allow_stale=False, force_refresh=True)


def _best_contract(
    ticker: str,
    direction: str,
    entry: float,
    *,
    allow_stale: bool = True,
    force_refresh: bool = False,
    block_on_miss: bool = True,
) -> dict:
    option_type = "CALL" if direction == "LONG" else "PUT"
    cache_key = _best_contract_cache_key(ticker, direction, entry)
    now = datetime.utcnow()
    with _cache_lock:
        cached = _best_contract_cache.get(cache_key)
        if cached and not force_refresh:
            cached_data = dict(cached.get("data", {}))
            age = now - cached.get("fetched_at", now)
            if age < BEST_CONTRACT_CACHE_TTL:
                _cache_record("best_contract", "hit")
                cached_data["cache"] = "hit"
                return cached_data
            if allow_stale and cached_data:
                _cache_record("best_contract", "stale")
                cached_data["cache"] = "stale"
                _submit_background_job(("best_contract", ticker, direction, cache_key[2]), _refresh_best_contract, ticker, direction, entry)
                return cached_data
        _cache_record("best_contract", "miss")

    if not block_on_miss and not force_refresh:
        _submit_background_job(("best_contract", ticker, direction, cache_key[2]), _refresh_best_contract, ticker, direction, entry)
        return _loading_contract("Options data is loading in the background")

    expirations = _parsed_expirations(_option_expirations_for_ticker(ticker))
    if not expirations:
        result = _unclean_contract("No option expirations available", "unavailable")
        return _store_best_contract(cache_key, result, now)

    candidates = []
    target_dte = 37
    for expiry, dte in sorted(expirations, key=lambda item: (abs(item[1] - target_dte), item[1]))[:3]:
        chain = _option_chain_for_ticker(ticker, expiry)
        if chain is None:
            continue
        contracts = chain.calls if option_type == "CALL" else chain.puts
        if contracts is None or contracts.empty or "strike" not in contracts.columns:
            continue
        contracts = contracts.copy()
        contracts["strike_distance"] = (contracts["strike"].astype(float) - float(entry)).abs()
        contracts = contracts[contracts["strike_distance"] <= max(float(entry) * 0.10, 1.0)]
        for _, row in contracts.sort_values(["strike_distance", "strike"]).head(6).iterrows():
            strike = _safe_float(row.get("strike"))
            if strike is None:
                continue
            score, metrics, diagnostics = _score_contract_row(row, strike, entry, dte)
            candidates.append({
                "score": score,
                "expiry": expiry,
                "dte": dte,
                "type": option_type,
                "strike": strike,
                "symbol": _option_symbol_from_row(ticker, expiry, strike, option_type, row),
                **metrics,
                "diagnostics": diagnostics,
            })

    if not candidates:
        result = _unclean_contract("No contracts returned near the ideal strike", "unavailable")
        return _store_best_contract(cache_key, result, now)

    best = max(candidates, key=lambda item: item["score"])
    candidate_audit = _contract_candidate_audit(ticker, candidates, best)
    diagnostics = best.get("diagnostics", {})
    spread_pct = diagnostics.get("spread_pct")
    distance_pct = diagnostics.get("distance_pct")
    if distance_pct is not None and distance_pct > 10:
        result = {
            **_unclean_contract("Closest contract strike is too far from the ideal strike", "option_chain"),
            "best_score": best["score"],
            "diagnostics": diagnostics,
            "candidate_audit": candidate_audit,
        }
        return _store_best_contract(cache_key, result, now)
    if spread_pct is None or spread_pct > 35:
        result = {
            **_unclean_contract("Best contract spread is too wide or unavailable", "option_chain"),
            "best_score": best["score"],
            "diagnostics": diagnostics,
            "candidate_audit": candidate_audit,
        }
        return _store_best_contract(cache_key, result, now)
    if (best.get("open_interest") or 0) < 25 and (best.get("volume") or 0) < 1:
        result = {
            **_unclean_contract("Best contract liquidity is too thin", "option_chain"),
            "best_score": best["score"],
            "diagnostics": diagnostics,
            "candidate_audit": candidate_audit,
        }
        return _store_best_contract(cache_key, result, now)

    execution = "Excellent" if best["score"] >= 75 else "Fair" if best["score"] >= 55 else "No Clean Contract"
    if execution == "No Clean Contract":
        result = {
            **_unclean_contract("Spread, liquidity, DTE, or strike distance did not meet minimum quality", "option_chain"),
            "reason": "Spread, liquidity, DTE, or strike distance did not meet minimum quality",
            "best_score": best["score"],
            "diagnostics": best.get("diagnostics", {}),
            "candidate_audit": candidate_audit,
        }
        return _store_best_contract(cache_key, result, now)

    best.update({
        "available": True,
        "execution": execution,
        "source": "option_chain",
        "cache": "miss",
        "candidate_audit": candidate_audit,
    })
    return _store_best_contract(cache_key, best, now)


def _has_valid_trade_plan(result: dict) -> bool:
    direction = str(result.get("direction") or "").upper()
    if direction not in {"LONG", "SHORT"}:
        return False
    return all(_safe_float(result.get(key)) is not None for key in ("entry", "sl", "tp1"))


def _should_enrich_best_contract(result: dict, setup_grade: str, entry_status: str) -> bool:
    trade_stage = str((result.get("trade_eval") or {}).get("trade_stage") or "").upper()
    if setup_grade not in {"A", "B"}:
        return False
    if trade_stage in {"A+ READY", "B+ TRADEABLE"}:
        return _has_valid_trade_plan(result)
    if entry_status not in {"Tradeable", "Near Entry"}:
        return False
    return _has_valid_trade_plan(result)


OPTION_PLAN_SOURCE = "kairos_trade_plan"


def _option_plan_unavailable(reason: str) -> dict:
    return {
        "available": False,
        "reason": reason,
        "source": OPTION_PLAN_SOURCE,
    }


def _option_type_for_direction(direction: str) -> Optional[str]:
    raw = str(direction or "").upper()
    if raw in {"LONG", "CALL"}:
        return "CALL"
    if raw in {"SHORT", "PUT"}:
        return "PUT"
    return None


def _option_strike_increment(planned_entry: float) -> Optional[float]:
    price = _safe_float(planned_entry)
    if price is None or price <= 0:
        return None
    if price < 25:
        return 0.5
    if price < 100:
        return 1.0
    if price < 250:
        return 2.5
    return 5.0


def _round_option_plan_strike(value: float, increment: float, option_type: str) -> Optional[float]:
    raw = _safe_float(value)
    step = _safe_float(increment)
    if raw is None or step is None or step <= 0:
        return None
    scaled = raw / step
    if str(option_type or "").upper() == "PUT":
        rounded = math.floor(scaled + 1e-9) * step
    else:
        rounded = math.ceil(scaled - 1e-9) * step
    decimals = 2 if step < 1 else 1 if step % 1 else 0
    return round(rounded, decimals)


def _option_plan_expected_hold(setup: dict) -> dict:
    low = _safe_int(
        setup.get("expected_hold_min_days")
        or setup.get("expected_trading_days_low")
        or setup.get("expected_move_min_days")
    )
    high = _safe_int(
        setup.get("expected_hold_max_days")
        or setup.get("expected_trading_days_high")
        or setup.get("expected_move_max_days")
    )
    if low is not None and high is not None and low > 0 and high >= low:
        return {
            "min_trading_days": low,
            "max_trading_days": high,
            "label": f"{low}–{high} Trading Days",
            "fallback_used": False,
        }

    ev = setup.get("trade_eval") or {}
    entry_status = str(setup.get("entryStatus") or "").strip()
    if ev.get("trigger_confirmed") or ev.get("a_plus_ready"):
        low, high, speed = 3, 7, "fast"
    elif ev.get("b_plus_tradeable") or entry_status in {"Tradeable", "Near Entry"} or setup.get("confirmationStarted"):
        low, high, speed = 7, 12, "standard"
    else:
        low, high, speed = 10, 18, "extended"
    return {
        "min_trading_days": low,
        "max_trading_days": high,
        "label": f"{low}–{high} Trading Days",
        "fallback_used": True,
        "fallback_speed": speed,
    }


def _option_plan_expiration_window(expected_hold: dict) -> dict:
    high = _safe_int((expected_hold or {}).get("max_trading_days"))
    if high is None:
        min_dte, max_dte = 21, 35
    elif high <= 7:
        min_dte, max_dte = 21, 30
    elif high <= 12:
        min_dte, max_dte = 21, 35
    elif high <= 18:
        min_dte, max_dte = 30, 45
    else:
        min_dte, max_dte = 45, 60
    return {
        "min_dte": min_dte,
        "max_dte": max_dte,
        "label": f"{min_dte}–{max_dte} DTE",
    }


def _option_plan_confidence(setup: dict) -> dict:
    grade = str(setup.get("setupGrade") or setup.get("setup_grade") or "").strip().upper()
    ev = setup.get("trade_eval") or {}
    if grade.startswith("A") and (ev.get("trigger_confirmed") or ev.get("a_plus_ready")):
        stars = 5
    elif grade.startswith("A"):
        stars = 4
    elif grade.startswith("B"):
        stars = 3
    else:
        stars = 2
    return {
        "stars": stars,
        "label": "★" * stars + "☆" * (5 - stars),
        "note": "Confidence reflects Kairos setup quality, not a guaranteed probability of profit.",
    }


def _format_option_plan_move(dollars: float, percent: float) -> str:
    sign = "+" if dollars >= 0 else "-"
    return f"{sign}${abs(dollars):.2f} ({sign}{abs(percent):.1f}%)"


def build_option_plan(setup: dict) -> dict:
    if not isinstance(setup, dict):
        return _option_plan_unavailable("missing setup")
    option_type = _option_type_for_direction(setup.get("direction"))
    if not option_type:
        return _option_plan_unavailable("missing direction")

    planned_entry = _safe_float(setup.get("entry"))
    entry_source = "planned_entry"
    if planned_entry is None:
        planned_entry = _safe_float(
            setup.get("plannedEntry")
            or setup.get("entry_price")
            or setup.get("current_quote_price")
            or setup.get("price")
            or setup.get("current_price")
        )
        entry_source = "fallback_price" if planned_entry is not None else "missing"
    if planned_entry is None or planned_entry <= 0:
        return _option_plan_unavailable("missing planned entry")

    tp1 = _safe_float(setup.get("tp1") or setup.get("target") or setup.get("target_price"))
    if tp1 is None or tp1 <= 0:
        return _option_plan_unavailable("missing TP1")

    move_dollars = tp1 - planned_entry
    if (option_type == "CALL" and move_dollars <= 0) or (option_type == "PUT" and move_dollars >= 0):
        return _option_plan_unavailable("invalid projected move")
    move_percent = (move_dollars / planned_entry) * 100
    increment = _option_strike_increment(planned_entry)
    if increment is None:
        return _option_plan_unavailable("strike rounding unavailable")
    raw_strike = planned_entry + (move_dollars * 0.5)
    preferred_strike = _round_option_plan_strike(raw_strike, increment, option_type)
    if preferred_strike is None:
        return _option_plan_unavailable("preferred strike unavailable")

    expected_hold = _option_plan_expected_hold(setup)
    expiration = _option_plan_expiration_window(expected_hold)
    confidence = _option_plan_confidence(setup)
    return {
        "available": True,
        "type": option_type,
        "preferred_strike": preferred_strike,
        "raw_preferred_strike": round(raw_strike, 4),
        "strike_rounding_increment": increment,
        "entry_source": entry_source,
        "planned_entry": planned_entry,
        "tp1": tp1,
        "suggested_expiration": expiration,
        "expected_hold": expected_hold,
        "expected_move": {
            "dollars": round(move_dollars, 2),
            "percent": round(move_percent, 1),
            "label": _format_option_plan_move(move_dollars, move_percent),
        },
        "confidence": confidence,
        "source": OPTION_PLAN_SOURCE,
    }


# ── Price Action Functions ────────────────────────────────────────────────────

SWING_DAILY_PRICE_TOLERANCE = 0.006


def _price_level_tolerance(tolerance: Optional[float] = None) -> float:
    try:
        value = float(tolerance or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value) or value <= 0:
        return 0.0
    return value


def _swing_tolerance_for_pair(left: dict, right: dict) -> float:
    left_tol = left.get("tolerance")
    right_tol = right.get("tolerance")
    if left_tol is not None or right_tol is not None:
        return max(_price_level_tolerance(left_tol), _price_level_tolerance(right_tol))
    return 0.0


def _meaningfully_greater(left: float, right: float, tolerance: float) -> bool:
    return float(left) > float(right) + float(tolerance)


def _meaningfully_less(left: float, right: float, tolerance: float) -> bool:
    return float(left) < float(right) - float(tolerance)


def _find_swings(df: pd.DataFrame, margin: int = 4, tolerance: Optional[float] = None) -> list:
    highs = df["High"].values
    lows  = df["Low"].values
    price_tolerance = _price_level_tolerance(tolerance)
    swings = []
    for i in range(margin, len(df) - margin):
        high = float(highs[i])
        low = float(lows[i])
        window_h = highs[i - margin : i + margin + 1]
        window_l = lows[i  - margin : i + margin + 1]
        if high >= float(window_h.max()) - price_tolerance:
            swing = {"index": i, "price": high, "type": "high"}
            if price_tolerance:
                swing["tolerance"] = price_tolerance
            swings.append(swing)
        elif low <= float(window_l.min()) + price_tolerance:
            swing = {"index": i, "price": low, "type": "low"}
            if price_tolerance:
                swing["tolerance"] = price_tolerance
            swings.append(swing)
    return swings


def _get_trend(swings: list) -> str:
    highs = [s for s in swings if s["type"] == "high"]
    lows  = [s for s in swings if s["type"] == "low"]
    if len(highs) < 2 or len(lows) < 2:
        return "NEUTRAL"
    high_tolerance = _swing_tolerance_for_pair(highs[-1], highs[-2])
    low_tolerance = _swing_tolerance_for_pair(lows[-1], lows[-2])
    hh = _meaningfully_greater(highs[-1]["price"], highs[-2]["price"], high_tolerance)
    hl = _meaningfully_greater(lows[-1]["price"], lows[-2]["price"], low_tolerance)
    lh = _meaningfully_less(highs[-1]["price"], highs[-2]["price"], high_tolerance)
    ll = _meaningfully_less(lows[-1]["price"], lows[-2]["price"], low_tolerance)
    if hh and hl:
        return "LONG"
    if lh and ll:
        return "SHORT"
    return "NEUTRAL"


def _regime_direction_label(structure_trend: str) -> str:
    if structure_trend == "LONG":
        return "Bullish"
    if structure_trend == "SHORT":
        return "Bearish"
    return "Neutral"


def _regime_trend_matches(ema_direction: str, structure_trend: str) -> bool:
    return (
        (ema_direction == "Bullish" and structure_trend == "LONG")
        or (ema_direction == "Bearish" and structure_trend == "SHORT")
    )


def _last_swings(swings: list, swing_type: str, count: int) -> list:
    return [s for s in swings if s.get("type") == swing_type][-count:]


def _market_regime_range_context(df: pd.DataFrame, swings: list) -> Optional[dict]:
    highs = _last_swings(swings, "high", REGIME_RECENT_SWING_POINTS)
    lows = _last_swings(swings, "low", REGIME_RECENT_SWING_POINTS)
    if len(highs) < 2 or len(lows) < 2:
        return None

    range_high = max(float(s["price"]) for s in highs)
    range_low = min(float(s["price"]) for s in lows)
    atr = _compute_atr(df)
    price = float(df["Close"].iloc[-1])
    if not math.isfinite(atr) or atr <= 0:
        return None

    tolerance = REGIME_RANGE_ATR_TOLERANCE * atr
    inside_tolerance = (range_low - tolerance) <= price <= (range_high + tolerance)
    return {
        "range_low": round(range_low, 2),
        "range_high": round(range_high, 2),
        "price": round(price, 2),
        "atr": round(atr, 2),
        "atr_tolerance": REGIME_RANGE_ATR_TOLERANCE,
        "inside_tolerance": inside_tolerance,
        "distance_above_high_atr": round((price - range_high) / atr, 2),
        "distance_below_low_atr": round((range_low - price) / atr, 2),
    }


def _classify_market_regime_from_components(
    ema_direction: str,
    structure_trend: str,
    daily_aligned: bool,
    relative_volume: float,
    atr_expansion: float,
    range_context: Optional[dict],
) -> dict:
    trend_matches = _regime_trend_matches(ema_direction, structure_trend)
    ema_score = REGIME_WEIGHTS["ema"] if ema_direction != "Neutral" else 0
    swing_score = (
        REGIME_WEIGHTS["swings"]
        if trend_matches
        else 10 if structure_trend != "NEUTRAL"
        else 0
    )
    htf_score = REGIME_WEIGHTS["htf"] if daily_aligned else 6 if ema_direction != "Neutral" else 0
    volume_score = (
        min(REGIME_WEIGHTS["volume"], max(0, int((relative_volume - 0.8) / 0.7 * REGIME_WEIGHTS["volume"])))
        if relative_volume
        else 0
    )
    atr_score = (
        min(REGIME_WEIGHTS["atr"], max(0, int((atr_expansion - 0.9) / 0.5 * REGIME_WEIGHTS["atr"])))
        if atr_expansion
        else 0
    )
    score = max(0, min(100, ema_score + swing_score + htf_score + volume_score + atr_score))

    directional_conflict = ema_direction != "Neutral" and structure_trend != "NEUTRAL" and not trend_matches
    one_side_neutral = (ema_direction == "Neutral") != (structure_trend == "NEUTRAL")
    range_override = bool(
        range_context
        and range_context.get("inside_tolerance")
        and not trend_matches
        and (directional_conflict or one_side_neutral)
    )

    if ema_direction != "Neutral" and trend_matches and score >= 70:
        regime = "TRENDING"
    elif range_override:
        regime = "RANGING"
    elif score < 40 or (ema_direction == "Neutral" and structure_trend == "NEUTRAL"):
        regime = "RANGING"
    else:
        regime = "MIXED"

    return {
        "regime": regime,
        "score": score,
        "ema_direction": ema_direction,
        "swing_direction": _regime_direction_label(structure_trend),
        "structure_trend": structure_trend,
        "daily_aligned": daily_aligned,
        "relative_volume": relative_volume,
        "atr_expansion": atr_expansion,
        "range_override": range_override,
        "range_context": range_context,
        "components": {
            "ema": ema_score,
            "swings": swing_score,
            "htf": htf_score,
            "volume": volume_score,
            "atr": atr_score,
        },
    }


def _market_regime_for_df(df: pd.DataFrame, daily_df: Optional[pd.DataFrame] = None) -> dict:
    clean_df = _flatten_columns(df.copy()).dropna().astype(float)
    swings = _find_swings(clean_df, margin=4)
    structure_trend = _get_trend(swings)
    ema = _ema_trend_alignment(clean_df)
    ema_direction = ema["direction"]
    relative_volume = _relative_volume(clean_df)
    atr_expansion = _atr_expansion(clean_df)
    daily_aligned = _daily_aligns(daily_df, ema_direction) if daily_df is not None else False
    range_context = _market_regime_range_context(clean_df, swings)
    return _classify_market_regime_from_components(
        ema_direction=ema_direction,
        structure_trend=structure_trend,
        daily_aligned=daily_aligned,
        relative_volume=relative_volume,
        atr_expansion=atr_expansion,
        range_context=range_context,
    )


def _detect_bos(df: pd.DataFrame, swings: list, direction: str, lookback: int = 40):
    n        = len(df)
    opens    = df["Open"].values
    closes   = df["Close"].values
    highs_sw = [s for s in swings if s["type"] == "high"]
    lows_sw  = [s for s in swings if s["type"] == "low"]
    min_idx  = max(0, n - 1 - lookback)

    if direction == "LONG" and len(highs_sw) >= 2:
        prev_high = highs_sw[-2]
        for i in range(max(prev_high["index"] + 1, min_idx), n):
            if closes[i] > prev_high["price"] and closes[i] > opens[i]:
                return True, float(prev_high["price"])

    if direction == "SHORT" and len(lows_sw) >= 2:
        prev_low = lows_sw[-2]
        for i in range(max(prev_low["index"] + 1, min_idx), n):
            if closes[i] < prev_low["price"] and closes[i] < opens[i]:
                return True, float(prev_low["price"])

    return False, 0.0


def _find_order_block(df: pd.DataFrame, direction: str, swings: list) -> Optional[dict]:
    n      = len(df)
    opens  = df["Open"].values
    closes = df["Close"].values
    highs  = df["High"].values
    lows   = df["Low"].values

    if direction == "LONG":
        lows_sw = [s for s in swings if s["type"] == "low"]
        if not lows_sw:
            return None
        last_swing_low = lows_sw[-1]
        for i in range(n - 2, last_swing_low["index"] - 1, -1):
            if closes[i] < opens[i]:  # bearish candle before bullish impulse
                return {"high": float(highs[i]), "low": float(lows[i]), "index": i}

    if direction == "SHORT":
        highs_sw = [s for s in swings if s["type"] == "high"]
        if not highs_sw:
            return None
        last_swing_high = highs_sw[-1]
        for i in range(n - 2, last_swing_high["index"] - 1, -1):
            if closes[i] > opens[i]:  # bullish candle before bearish impulse
                return {"high": float(highs[i]), "low": float(lows[i]), "index": i}

    return None


def _macro_bias(price: float, df: pd.DataFrame) -> tuple:
    """
    Compute macro trend bias using 52-week high and 200-candle window high.
    Returns (bias, pct_from_52w_high, wk52_high, window_high).
      Macro Bearish : price > 15% below 52-week high
      Macro Bullish : price within 5% of 52-week high
      Macro Neutral : everything else
    """
    closes = df["Close"].astype(float)
    wk52_high    = float(closes.iloc[-252:].max()) if len(closes) >= 252 else float(closes.max())
    window_high  = float(closes.iloc[-200:].max()) if len(closes) >= 200 else float(closes.max())
    pct_from_52w = (wk52_high - price) / wk52_high if wk52_high > 0 else 0.0

    if pct_from_52w > 0.15:
        bias = "Macro Bearish"
    elif pct_from_52w < 0.05:
        bias = "Macro Bullish"
    else:
        bias = "Macro Neutral"

    return bias, round(pct_from_52w, 3), round(wk52_high, 2), round(window_high, 2)


def _market_structure(
    swings: list,
    price: float,
    df: pd.DataFrame,
    macro_bias: str = "",
    window_high: float = 0.0,
) -> tuple:
    """
    Classify structure as 'bullish', 'bearish', or 'ranging' using a weighted vote system.
    Returns (classification: str, reasons: list[str]).

    Weights:
      +4 bearish/bullish : macro bias (52w high check) — hard override
      +2 bearish/bullish : price vs EMA200
      +2 bearish/bullish : recent swing high comparison (LH vs HH)
      +2 bearish/bullish : recent swing low comparison (LL vs HL)
      +2 bearish/bullish : 3-swing LH/LL or HH/HL sequence
      +2 bearish         : price > 15% below 200-candle window high
    """
    reasons: list = []
    bearish_score = 0
    bullish_score = 0

    highs = [s for s in swings if s["type"] == "high"]
    lows  = [s for s in swings if s["type"] == "low"]

    def swing_price(swing: dict) -> float:
        return float(swing["price"])

    def swing_less(left: dict, right: dict) -> bool:
        return _meaningfully_less(swing_price(left), swing_price(right), _swing_tolerance_for_pair(left, right))

    def swing_greater(left: dict, right: dict) -> bool:
        return _meaningfully_greater(swing_price(left), swing_price(right), _swing_tolerance_for_pair(left, right))

    # ── Macro bias override (weight 4) ───────────────────────────────────────
    if macro_bias == "Macro Bearish":
        bearish_score += 4
        reasons.append(f"macro bias: price >15% below 52w high [bearish +4]")
    elif macro_bias == "Macro Bullish":
        bullish_score += 4
        reasons.append(f"macro bias: price within 5% of 52w high [bullish +4]")

    # ── 200-candle window high check (weight 2) ───────────────────────────────
    if window_high > 0:
        pct_below_window = (window_high - price) / window_high
        if pct_below_window > 0.15:
            bearish_score += 2
            reasons.append(
                f"price ${price:.2f} is {pct_below_window:.1%} below "
                f"200-bar high ${window_high:.2f} [bearish +2]"
            )

    # ── EMA200 bias (weight 2) ────────────────────────────────────────────────
    if len(df) >= 200:
        ema200 = float(df["Close"].ewm(span=200, adjust=False).mean().iloc[-1])
        if price < ema200:
            bearish_score += 2
            reasons.append(f"price ${price:.2f} below EMA200 ${ema200:.2f} [bearish +2]")
        else:
            bullish_score += 2
            reasons.append(f"price ${price:.2f} above EMA200 ${ema200:.2f} [bullish +2]")

    # ── Recent swing high comparison (weight 2) ───────────────────────────────
    if len(highs) >= 2:
        if swing_less(highs[-1], highs[-2]):
            bearish_score += 2
            reasons.append(f"LH: {swing_price(highs[-1]):.2f} < {swing_price(highs[-2]):.2f} [bearish +2]")
        else:
            bullish_score += 2
            reasons.append(f"HH: {swing_price(highs[-1]):.2f} >= {swing_price(highs[-2]):.2f} [bullish +2]")

    # ── Recent swing low comparison (weight 2) ────────────────────────────────
    if len(lows) >= 2:
        if swing_less(lows[-1], lows[-2]):
            bearish_score += 2
            reasons.append(f"LL: {swing_price(lows[-1]):.2f} < {swing_price(lows[-2]):.2f} [bearish +2]")
        else:
            bullish_score += 2
            reasons.append(f"HL: {swing_price(lows[-1]):.2f} >= {swing_price(lows[-2]):.2f} [bullish +2]")

    # ── 3-swing sequence (weight 2) ───────────────────────────────────────────
    if (len(highs) >= 3 and swing_greater(highs[-3], highs[-2]) and swing_greater(highs[-2], highs[-1])
            and len(lows) >= 2 and swing_greater(lows[-2], lows[-1])):
        bearish_score += 2
        reasons.append("3-swing LH/LL sequence confirmed [bearish +2]")
    elif (len(lows) >= 3 and swing_less(lows[-3], lows[-2]) and swing_less(lows[-2], lows[-1])
            and len(highs) >= 2 and swing_less(highs[-2], highs[-1])):
        bullish_score += 2
        reasons.append("3-swing HH/HL sequence confirmed [bullish +2]")

    if bearish_score > bullish_score:
        result = "bearish"
    elif bullish_score > bearish_score:
        result = "bullish"
    else:
        result = "ranging"

    reasons.append(f"→ scores: bearish={bearish_score} bullish={bullish_score} → {result}")
    return result, reasons


def _detect_choch(swings: list, direction: str) -> tuple:
    """
    Scan ALL swing pairs for CHoCH events and return the most recent one.
    Most recent CHoCH takes priority over earlier reversals (handles short-term bounces).
    Returns (suppress, reason, bearish_choch_level, last_choch_bar_index).
      suppress=True             → this CHoCH conflicts with `direction`, filter the setup.
      bearish_choch_level       → the prior swing-low that was broken (None if no bearish CHoCH).
      last_choch_bar_index      → bar index of the most recent CHoCH (-1 if none found).
    """
    highs = [s for s in swings if s["type"] == "high"]
    lows  = [s for s in swings if s["type"] == "low"]

    last_bearish_idx   = -1
    last_bullish_idx   = -1
    bearish_reason     = ""
    bullish_reason     = ""
    bearish_choch_lvl  = None   # prior swing low that was broken = CHoCH level

    # Bearish CHoCH: LL confirmed after a LH between the two lows
    for i in range(1, len(lows)):
        if lows[i]["price"] < lows[i - 1]["price"]:
            between_highs = [h for h in highs
                             if lows[i - 1]["index"] < h["index"] < lows[i]["index"]]
            prior_highs   = [h for h in highs if h["index"] < lows[i - 1]["index"]]
            if (between_highs and prior_highs
                    and between_highs[-1]["price"] < prior_highs[-1]["price"]):
                if lows[i]["index"] > last_bearish_idx:
                    last_bearish_idx  = lows[i]["index"]
                    bearish_choch_lvl = lows[i - 1]["price"]   # the level that was broken
                    bearish_reason = (
                        f"bearish CHoCH at swing-low {lows[i]['price']:.2f} "
                        f"(broke prior low {lows[i-1]['price']:.2f} after LH "
                        f"{between_highs[-1]['price']:.2f} < {prior_highs[-1]['price']:.2f})"
                    )

    # Bullish CHoCH: HH confirmed after a HL between the two highs
    for i in range(1, len(highs)):
        if highs[i]["price"] > highs[i - 1]["price"]:
            between_lows = [l for l in lows
                            if highs[i - 1]["index"] < l["index"] < highs[i]["index"]]
            prior_lows   = [l for l in lows if l["index"] < highs[i - 1]["index"]]
            if (between_lows and prior_lows
                    and between_lows[-1]["price"] > prior_lows[-1]["price"]):
                if highs[i]["index"] > last_bullish_idx:
                    last_bullish_idx = highs[i]["index"]
                    bullish_reason = (
                        f"bullish CHoCH at swing-high {highs[i]['price']:.2f} "
                        f"(broke prior high {highs[i-1]['price']:.2f} after HL "
                        f"{between_lows[-1]['price']:.2f} > {prior_lows[-1]['price']:.2f})"
                    )

    # No CHoCH found at all
    if last_bearish_idx == -1 and last_bullish_idx == -1:
        return False, "no CHoCH detected", None, None, -1

    # Track the bullish CHoCH level (prior swing high that was broken)
    bullish_choch_lvl: Optional[float] = None
    if last_bullish_idx >= 0:
        for i in range(1, len(highs)):
            if highs[i]["index"] == last_bullish_idx:
                bullish_choch_lvl = highs[i - 1]["price"]
                break

    # Most recent CHoCH wins regardless of short-term counter-moves
    if last_bearish_idx >= last_bullish_idx:
        if direction == "LONG":
            return True, f"[SUPPRESS] most recent CHoCH is bearish → {bearish_reason}", bearish_choch_lvl, bullish_choch_lvl, last_bearish_idx
        return False, f"bearish CHoCH present but not suppressing {direction} → {bearish_reason}", bearish_choch_lvl, bullish_choch_lvl, last_bearish_idx
    else:
        if direction == "SHORT":
            return True, f"[SUPPRESS] most recent CHoCH is bullish → {bullish_reason}", bearish_choch_lvl, bullish_choch_lvl, last_bullish_idx
        return False, f"bullish CHoCH present but not suppressing {direction} → {bullish_reason}", bearish_choch_lvl, bullish_choch_lvl, last_bullish_idx


def _safe_ratio(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    return numerator / denominator if denominator else fallback


def _grade_from_score(score: int) -> str:
    if score >= 90:
        return "A+"
    if score >= 84:
        return "A"
    if score >= 78:
        return "B+"
    if score >= 70:
        return "B"
    if score >= 62:
        return "C+"
    if score >= 54:
        return "C"
    return "D"


def _apply_grade_caps(
    score: int,
    *,
    bos_confirmed: bool,
    ob: Optional[dict],
    in_ob: bool,
    near_ob: bool,
    location: str,
    cleanliness: str,
    touches: int,
    bos_extension: Optional[float],
    room_to_target: str,
) -> tuple:
    cap = 100
    reasons = []

    if not bos_confirmed:
        cap = min(cap, 58)
        reasons.append("No confirmed BOS caps this at C.")
    if not ob:
        cap = min(cap, 66)
        reasons.append("No order block caps this at C+.")
    if ob and not (in_ob or near_ob):
        cap = min(cap, 72)
        reasons.append("Price has not returned to the OB, so this is capped at B.")
    if "late for fresh" in location:
        cap = min(cap, 84)
        reasons.append("Late location caps this at A.")
    if cleanliness == "Choppy / overlapping":
        cap = min(cap, 82)
        reasons.append("Choppy candles cap this at B+.")
    elif cleanliness == "Readable but mixed":
        cap = min(cap, 91)
        reasons.append("Mixed candles keep this below A+.")
    if touches == 2:
        cap = min(cap, 88)
        reasons.append("Tapped once keeps this below A+.")
    elif touches >= 3:
        cap = min(cap, 76)
        reasons.append("Repeated OB taps cap this at B.")
    if bos_extension is not None:
        if bos_extension > 4:
            cap = min(cap, 78)
            reasons.append("Very extended from BOS caps this at B+.")
        elif bos_extension > 3:
            cap = min(cap, 84)
            reasons.append("Extended from BOS caps this at A.")
    if room_to_target.startswith("Crowded"):
        cap = min(cap, 74)
        reasons.append("Crowded room to target caps this at B.")

    final_score = min(score, cap)
    grade_note = reasons[0] if reasons else "No grade cap applied."
    return final_score, cap, grade_note


def _latest_swing_range(swings: list) -> Optional[dict]:
    highs = [s for s in swings if s["type"] == "high"]
    lows = [s for s in swings if s["type"] == "low"]
    if not highs or not lows:
        return None
    high = max(highs[-3:], key=lambda s: s["price"])
    low = min(lows[-3:], key=lambda s: s["price"])
    if high["price"] <= low["price"]:
        return None
    return {"high": high["price"], "low": low["price"]}


def _location_read(price: float, direction: str, swings: list) -> tuple:
    swing_range = _latest_swing_range(swings)
    if not swing_range:
        return "Unclear", None

    low = swing_range["low"]
    high = swing_range["high"]
    percentile = max(0.0, min(1.0, _safe_ratio(price - low, high - low)))

    if percentile >= 0.67:
        zone = "Premium"
    elif percentile <= 0.33:
        zone = "Discount"
    else:
        zone = "Midrange"

    if direction == "LONG" and zone == "Premium":
        read = "Premium - late for fresh longs"
    elif direction == "LONG" and zone == "Discount":
        read = "Discount - better long location"
    elif direction == "SHORT" and zone == "Discount":
        read = "Discount - late for fresh shorts"
    elif direction == "SHORT" and zone == "Premium":
        read = "Premium - better short location"
    else:
        read = zone

    return read, round(percentile * 100, 1)


def _strict_location(price: float, swings: list) -> tuple:
    swing_range = _latest_swing_range(swings)
    if not swing_range:
        return "MIDRANGE", None

    low = swing_range["low"]
    high = swing_range["high"]
    percentile = max(0.0, min(1.0, _safe_ratio(price - low, high - low)))

    if percentile <= 0.18 or percentile >= 0.82:
        location = "AT EXTREME"
    elif percentile <= 0.35:
        location = "NEAR DISCOUNT"
    elif percentile >= 0.65:
        location = "NEAR PREMIUM"
    else:
        location = "MIDRANGE"
    return location, round(percentile * 100, 1)


def _structure_quality(trend: str, bos_confirmed: bool, cleanliness: str) -> str:
    if cleanliness == "Choppy / overlapping" or trend == "NEUTRAL":
        return "CHOPPY / INTERNAL ONLY"
    if bos_confirmed:
        return "CLEAN BOS"
    return "DEVELOPING"


def _directional_candles(df: pd.DataFrame, direction: str, lookback: int = 3) -> pd.DataFrame:
    recent = df.tail(lookback)
    if direction == "LONG":
        return recent[recent["Close"] > recent["Open"]]
    if direction == "SHORT":
        return recent[recent["Close"] < recent["Open"]]
    return recent.iloc[0:0]


def detect_displacement(df: pd.DataFrame, atr: float, direction: str, bos_confirmed: bool) -> tuple:
    if not bos_confirmed or direction not in ("LONG", "SHORT") or len(df) < 3 or atr <= 0:
        return "NONE", 0.0

    recent = df.tail(3)
    directional = _directional_candles(df, direction, lookback=3)
    if directional.empty:
        return "NONE", 0.0

    bodies = (directional["Close"] - directional["Open"]).abs()
    avg_body_atr = float(bodies.mean() / atr)
    last = recent.iloc[-1]
    last_is_directional = (
        (direction == "LONG" and float(last["Close"]) > float(last["Open"]))
        or (direction == "SHORT" and float(last["Close"]) < float(last["Open"]))
    )
    last_range_atr = float((last["High"] - last["Low"]) / atr) if last_is_directional else 0.0
    directional_majority = len(directional) >= 2

    if directional_majority and (avg_body_atr >= 0.7 or last_range_atr >= 1.2):
        return "STRONG", round(max(avg_body_atr, last_range_atr), 2)
    if avg_body_atr >= 0.35 or last_range_atr >= 0.75:
        return "WEAK", round(max(avg_body_atr, last_range_atr), 2)
    return "NONE", round(max(avg_body_atr, last_range_atr), 2)


def _timestamp_at(df: pd.DataFrame, index: int) -> Optional[str]:
    try:
        value = df.index[int(index)]
    except Exception:
        return None
    if hasattr(value, "tz_convert") and getattr(value, "tzinfo", None) is not None:
        return value.tz_convert("UTC").isoformat().replace("+00:00", "Z")
    if hasattr(value, "isoformat"):
        return value.isoformat() + ("Z" if getattr(value, "tzinfo", None) is None else "")
    return str(value)


def _atr_series_atr_context(df: pd.DataFrame, period: int = 14) -> pd.Series:
    highs = df["High"].astype(float)
    lows = df["Low"].astype(float)
    closes = df["Close"].astype(float)
    prev_close = closes.shift(1)
    tr = pd.concat([
        highs - lows,
        (highs - prev_close).abs(),
        (lows - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _candle_closes_in_direction(df: pd.DataFrame, index: int, direction: str) -> bool:
    if index < 0 or index >= len(df):
        return False
    open_ = float(df["Open"].iloc[index])
    close = float(df["Close"].iloc[index])
    if direction == "LONG":
        return close > open_
    if direction == "SHORT":
        return close < open_
    return False


def _bos_event_level_for_candle(df: pd.DataFrame, swings: list, index: int, direction: str) -> Optional[float]:
    if index <= 0 or direction not in ("LONG", "SHORT"):
        return None
    prior_swings = [
        s for s in swings
        if s.get("index", -1) < index and s.get("type") == ("high" if direction == "LONG" else "low")
    ]
    if len(prior_swings) < 2:
        return None
    level = float(prior_swings[-2]["price"])
    close = float(df["Close"].iloc[index])
    prior_close = float(df["Close"].iloc[index - 1])
    if direction == "LONG" and close > level and prior_close <= level and _candle_closes_in_direction(df, index, direction):
        return level
    if direction == "SHORT" and close < level and prior_close >= level and _candle_closes_in_direction(df, index, direction):
        return level
    return None


def _bos_impulse_window_indices(df: pd.DataFrame, bos_index: int, direction: str, evaluation_index: int) -> list[int]:
    start = int(bos_index)
    for i in range(bos_index - 1, max(-1, bos_index - 3), -1):
        if not _candle_closes_in_direction(df, i, direction):
            break
        start = i

    end = int(bos_index)
    max_follow = min(int(evaluation_index), bos_index + 2, len(df) - 1)
    for i in range(bos_index + 1, max_follow + 1):
        if not _candle_closes_in_direction(df, i, direction):
            break
        end = i
    return list(range(start, end + 1))


def _displacement_strength_for_indices(
    df: pd.DataFrame,
    indices: list[int],
    direction: str,
    atr: float,
    *,
    require_majority_for_strong: bool,
) -> dict:
    if not indices or direction not in ("LONG", "SHORT") or atr is None or not math.isfinite(float(atr)) or float(atr) <= 0:
        return {
            "displacement": "UNKNOWN",
            "score": 0.0,
            "avg_body_atr": None,
            "range_atr": None,
            "directional_count": 0,
        }
    directional = [i for i in indices if _candle_closes_in_direction(df, i, direction)]
    if not directional:
        return {
            "displacement": "NONE",
            "score": 0.0,
            "avg_body_atr": 0.0,
            "range_atr": 0.0,
            "directional_count": 0,
        }
    bodies = [
        abs(float(df["Close"].iloc[i]) - float(df["Open"].iloc[i]))
        for i in directional
    ]
    avg_body_atr = float(np.mean(bodies) / float(atr))
    last_directional = directional[-1]
    latest_range_atr = float((float(df["High"].iloc[last_directional]) - float(df["Low"].iloc[last_directional])) / float(atr))
    score = max(avg_body_atr, latest_range_atr)
    majority_ok = len(directional) >= 2 or not require_majority_for_strong
    if majority_ok and (avg_body_atr >= 0.70 or latest_range_atr >= 1.20):
        displacement = "STRONG"
    elif avg_body_atr >= 0.35 or latest_range_atr >= 0.75:
        displacement = "WEAK"
    else:
        displacement = "NONE"
    return {
        "displacement": displacement,
        "score": round(score, 2),
        "avg_body_atr": round(avg_body_atr, 2),
        "range_atr": round(latest_range_atr, 2),
        "directional_count": len(directional),
    }


def _linked_ob_for_bos_impulse(df: pd.DataFrame, direction: str, window_start: int) -> dict:
    opposing = []
    for i in range(int(window_start) - 1, -1, -1):
        if direction == "LONG" and float(df["Close"].iloc[i]) < float(df["Open"].iloc[i]):
            opposing.append(i)
            break
        if direction == "SHORT" and float(df["Close"].iloc[i]) > float(df["Open"].iloc[i]):
            opposing.append(i)
            break
    if not opposing:
        return {
            "linked_ob_low": None,
            "linked_ob_high": None,
            "linked_ob_candle_index": None,
            "linked_ob_timestamp": None,
            "linked_ob_method": "none_found",
            "linked_ob_ambiguous": True,
        }
    index = opposing[0]
    return {
        "linked_ob_low": round(float(df["Low"].iloc[index]), 2),
        "linked_ob_high": round(float(df["High"].iloc[index]), 2),
        "linked_ob_candle_index": index,
        "linked_ob_timestamp": _timestamp_at(df, index),
        "linked_ob_method": "preceding_opposing_candle",
        "linked_ob_ambiguous": False,
    }


def _bos_event_invalidations(df: pd.DataFrame, event: dict, all_events: list[dict], evaluation_index: int) -> tuple[str, Optional[str], Optional[int]]:
    direction = event.get("direction")
    bos_index = int(event.get("bos_candle_index"))
    candidates = []

    for other in all_events:
        other_index = int(other.get("bos_candle_index"))
        if other_index <= bos_index or other_index > evaluation_index:
            continue
        if other.get("direction") == direction:
            candidates.append((other_index, "SUPERSEDED_BY_NEWER_SAME_DIRECTION_BOS", other_index))
        else:
            candidates.append((other_index, "OPPOSITE_BOS", other_index))

    low = event.get("linked_ob_low")
    high = event.get("linked_ob_high")
    if low is not None and high is not None:
        for i in range(bos_index + 1, min(evaluation_index, len(df) - 1) + 1):
            close = float(df["Close"].iloc[i])
            if direction == "LONG" and close < float(low):
                candidates.append((i, "OB_INVALIDATED", None))
                break
            if direction == "SHORT" and close > float(high):
                candidates.append((i, "OB_INVALIDATED", None))
                break
            touches = (
                float(df["High"].iloc[i]) >= float(low)
                and float(df["Low"].iloc[i]) <= float(high)
            )
            if touches and i > bos_index:
                prior_touches = sum(
                    1
                    for j in range(bos_index + 1, i + 1)
                    if float(df["High"].iloc[j]) >= float(low) and float(df["Low"].iloc[j]) <= float(high)
                )
                if prior_touches >= 3:
                    candidates.append((i, "EXISTING_OB_FRESHNESS_INVALIDATION", None))
                    break

    if not candidates:
        return "VALID", None, None
    invalid_index, reason, superseded_by = min(candidates, key=lambda item: item[0])
    return "INVALID", reason, superseded_by


def detect_bos_events_with_displacement(
    df: pd.DataFrame,
    direction: Optional[str] = None,
    evaluation_index: Optional[int] = None,
) -> list[dict]:
    """Developer-only BOS-linked displacement study; does not affect live strategy."""
    if df is None or getattr(df, "empty", True):
        return []
    clean_df = _flatten_columns(df.copy()).dropna()
    if len(clean_df) < 5:
        return []
    evaluation_index = len(clean_df) - 1 if evaluation_index is None else min(int(evaluation_index), len(clean_df) - 1)
    if evaluation_index < 0:
        return []
    study_df = clean_df.iloc[:evaluation_index + 1].copy()
    swings = _find_swings(study_df, margin=2)
    atr_values = _atr_series_atr_context(study_df)
    directions = [direction] if direction in ("LONG", "SHORT") else ["LONG", "SHORT"]
    events = []
    seen = set()

    for bos_direction in directions:
        for i in range(1, len(study_df)):
            level = _bos_event_level_for_candle(study_df, swings, i, bos_direction)
            if level is None:
                continue
            key = (bos_direction, i, round(level, 6))
            if key in seen:
                continue
            seen.add(key)
            atr_at_bos = float(atr_values.iloc[i]) if i < len(atr_values) else float("nan")
            if i < 14 or not math.isfinite(atr_at_bos) or atr_at_bos <= 0:
                candle_strength = _displacement_strength_for_indices(study_df, [i], bos_direction, 0.0, require_majority_for_strong=False)
                impulse_indices = _bos_impulse_window_indices(study_df, i, bos_direction, evaluation_index)
                impulse_strength = _displacement_strength_for_indices(study_df, impulse_indices, bos_direction, 0.0, require_majority_for_strong=True)
            else:
                candle_strength = _displacement_strength_for_indices(study_df, [i], bos_direction, atr_at_bos, require_majority_for_strong=False)
                impulse_indices = _bos_impulse_window_indices(study_df, i, bos_direction, evaluation_index)
                impulse_strength = _displacement_strength_for_indices(study_df, impulse_indices, bos_direction, atr_at_bos, require_majority_for_strong=True)
            ob = _linked_ob_for_bos_impulse(study_df, bos_direction, impulse_indices[0] if impulse_indices else i)
            event = {
                "direction": bos_direction,
                "bos_level": round(float(level), 2),
                "bos_candle_index": i,
                "bos_timestamp": _timestamp_at(study_df, i),
                "bos_displacement": impulse_strength["displacement"],
                "bos_displacement_score": impulse_strength["score"],
                "bos_displacement_candle_index": i,
                "bos_displacement_timestamp": _timestamp_at(study_df, i),
                "displacement_window_start": impulse_indices[0] if impulse_indices else i,
                "displacement_window_end": impulse_indices[-1] if impulse_indices else i,
                "bos_candle_only_strength": candle_strength,
                "bos_impulse_window_strength": impulse_strength,
                "atr_period": 14,
                "atr_source": "bos_candle_close",
                "atr_at_bos": round(atr_at_bos, 4) if math.isfinite(atr_at_bos) else None,
                "linked_order_block": {
                    "low": ob.get("linked_ob_low"),
                    "high": ob.get("linked_ob_high"),
                    "index": ob.get("linked_ob_candle_index"),
                    "timestamp": ob.get("linked_ob_timestamp"),
                    "method": ob.get("linked_ob_method"),
                    "ambiguous": ob.get("linked_ob_ambiguous"),
                },
                **ob,
                "invalidation_state": "VALID",
                "invalidation_reason": None,
                "superseded_by_bos_index": None,
            }
            events.append(event)

    events.sort(key=lambda item: int(item.get("bos_candle_index", -1)))
    for event in events:
        state, reason, superseded_by = _bos_event_invalidations(study_df, event, events, evaluation_index)
        event["invalidation_state"] = state
        event["invalidation_reason"] = reason
        event["superseded_by_bos_index"] = superseded_by
    return events


def _select_active_bos_event(events: list[dict], row: dict) -> tuple[Optional[dict], str]:
    direction = str(row.get("direction") or row.get("trend") or "").upper()
    if direction not in ("LONG", "SHORT"):
        return None, "no_setup_direction"
    valid = [
        event for event in events
        if event.get("direction") == direction and event.get("invalidation_state") == "VALID"
    ]
    if not valid:
        return None, "no_valid_bos_in_setup_direction"
    row_ob_low = row.get("ob_low")
    row_ob_high = row.get("ob_high")
    if row_ob_low is not None and row_ob_high is not None:
        linked = [
            event for event in valid
            if event.get("linked_ob_low") is not None
            and abs(float(event.get("linked_ob_low")) - float(row_ob_low)) <= max(0.05, abs(float(row_ob_low)) * 0.002)
            and abs(float(event.get("linked_ob_high")) - float(row_ob_high)) <= max(0.05, abs(float(row_ob_high)) * 0.002)
        ]
        if linked:
            return linked[-1], "most_recent_valid_bos_linked_to_active_ob"
    return valid[-1], "most_recent_valid_bos_in_setup_direction"


def _shadow_status_from_trade_eval(row: dict, shadow_eval: dict) -> str:
    grade = str(row.get("setupGrade") or "").upper()
    direction = str(row.get("direction") or row.get("trend") or "").upper()
    setup_status = str(row.get("setupStatus") or row.get("setup_status") or "").upper()
    trade_stage = str(shadow_eval.get("trade_stage") or "").upper()
    entry_status = str(row.get("entryStatus") or "").strip()
    has_plan = row.get("entry") is not None and row.get("sl") is not None and row.get("tp1") is not None
    if direction not in ("LONG", "SHORT") or grade == "C" or setup_status == "SKIPPED":
        return "SKIP"
    if "NO TRADE" in trade_stage:
        return "SKIP"
    if entry_status == "Too Far":
        return "SKIP"
    if has_plan and (shadow_eval.get("trigger_confirmed") or shadow_eval.get("a_plus_ready")):
        return "ENTER_NOW"
    if has_plan and shadow_eval.get("b_plus_tradeable") and entry_status == "Tradeable":
        return "EARLY_ENTRY"
    if has_plan and entry_status in {"Tradeable", "Near Entry"}:
        return "ALMOST_READY"
    return "WAITING"


def _simulate_bos_displacement_status(row: dict, active_event: Optional[dict]) -> dict:
    live_eval = row.get("trade_eval") or {}
    current_displacement = str(live_eval.get("displacement") or "UNKNOWN").upper()
    bos_displacement = str((active_event or {}).get("bos_displacement") or "UNKNOWN").upper()
    bos_valid = bool(active_event and active_event.get("invalidation_state") == "VALID")
    would_recover = current_displacement != "STRONG" and bos_displacement == "STRONG" and bos_valid
    shadow_eval = dict(live_eval)
    live_missing = list(live_eval.get("missing_for_a_plus") or [])
    shadow_missing = list(live_missing)

    if would_recover:
        shadow_eval["displacement"] = "STRONG"
        shadow_eval["displacement_score"] = (active_event or {}).get("bos_displacement_score")
        shadow_missing = [item for item in shadow_missing if item != "Needs strong displacement"]
        shadow_eval["missing_for_a_plus"] = shadow_missing
        trigger_confirmed = bool(
            shadow_eval.get("sweep_taken")
            and shadow_eval.get("rejection_confirmed")
            and row.get("bos_confirmed")
        )
        shadow_eval["trigger_confirmed"] = trigger_confirmed
        room = shadow_eval.get("room_to_target") or {}
        rr_ok = room.get("estimated_rr") is not None and room.get("estimated_rr") >= 2.0
        room_clear = room.get("clear") is True and not room.get("blocked")
        a_plus_ready = bool(
            shadow_eval.get("htf_aligned")
            and shadow_eval.get("valid_zone")
            and shadow_eval.get("structure_quality") == "CLEAN BOS"
            and shadow_eval.get("sweep_taken")
            and shadow_eval.get("rejection_confirmed")
            and shadow_eval.get("setup_type") in ("CONTINUATION: BOS + retest", "REVERSAL: sweep + rejection + displacement")
            and room_clear
            and rr_ok
            and trigger_confirmed
        )
        shadow_eval["a_plus_ready"] = a_plus_ready
        if a_plus_ready:
            shadow_eval["trade_stage"] = "A+ READY"
        elif shadow_eval.get("b_plus_tradeable"):
            shadow_eval["trade_stage"] = "B+ TRADEABLE"
    else:
        shadow_eval["missing_for_a_plus"] = shadow_missing

    live_status = _shadow_status_from_trade_eval(row, live_eval)
    shadow_status = _shadow_status_from_trade_eval(row, shadow_eval)
    return {
        "current_displacement": current_displacement,
        "current_displacement_score": live_eval.get("displacement_score"),
        "bos_linked_displacement": bos_displacement,
        "bos_linked_displacement_score": (active_event or {}).get("bos_displacement_score"),
        "bos_linked_displacement_valid": bos_valid,
        "bos_linked_displacement_age_bars": (
            None if not active_event else max(0, int(row.get("_shadow_eval_index", 0)) - int(active_event.get("bos_candle_index", 0)))
        ),
        "bos_linked_displacement_source": "bos_impulse_window_at_bos_atr",
        "bos_linked_displacement_invalidation_reason": (active_event or {}).get("invalidation_reason"),
        "would_recover_strong_displacement": would_recover,
        "live_status": live_status,
        "shadow_status_if_bos_displacement_used": shadow_status,
        "live_missing_requirements": live_missing,
        "shadow_missing_requirements": shadow_missing,
        "would_change_status": live_status != shadow_status,
        "status_change_reason": (
            "BOS-linked STRONG displacement replaces only the current displacement input."
            if live_status != shadow_status else "No status change; other production requirements still control."
        ),
        "shadow_trade_eval": shadow_eval,
    }


def bos_displacement_shadow_for_setup(row: dict, df: pd.DataFrame) -> dict:
    clean_df = _flatten_columns(df.copy()).dropna() if df is not None and not getattr(df, "empty", True) else pd.DataFrame()
    if clean_df.empty:
        return {
            "ticker": row.get("ticker"),
            "error": "missing_candles",
            "current_displacement": str((row.get("trade_eval") or {}).get("displacement") or "UNKNOWN").upper(),
            "bos_linked_displacement": "UNKNOWN",
            "would_recover_strong_displacement": False,
            "would_change_status": False,
        }
    eval_index = len(clean_df) - 1
    shadow_row = dict(row)
    shadow_row["_shadow_eval_index"] = eval_index
    events = detect_bos_events_with_displacement(
        clean_df,
        direction=str(row.get("direction") or row.get("trend") or "").upper(),
        evaluation_index=eval_index,
    )
    active_event, selection_reason = _select_active_bos_event(events, row)
    simulation = _simulate_bos_displacement_status(shadow_row, active_event)
    active = active_event or {}
    return {
        "ticker": row.get("ticker"),
        "direction": row.get("direction") or row.get("trend"),
        "timeframe": row.get("timeframe"),
        "grade": row.get("setupGrade"),
        "live_trade_stage": (row.get("trade_eval") or {}).get("trade_stage"),
        "active_bos_index": active.get("bos_candle_index"),
        "active_bos_timestamp": active.get("bos_timestamp"),
        "active_bos_level": active.get("bos_level"),
        "active_bos_selection_reason": selection_reason,
        "bos_candle_displacement": (active.get("bos_candle_only_strength") or {}).get("displacement"),
        "bos_impulse_displacement": (active.get("bos_impulse_window_strength") or {}).get("displacement"),
        "linked_ob_low": active.get("linked_ob_low"),
        "linked_ob_high": active.get("linked_ob_high"),
        "linked_ob_ambiguous": active.get("linked_ob_ambiguous"),
        "bos_events": events,
        **simulation,
    }


def build_bos_displacement_shadow_report(rows: list, candle_data_by_symbol: dict) -> dict:
    traces = []
    current_counts = Counter()
    bos_counts = Counter()
    invalidation_counts = Counter()
    recovered_breakdown = {
        "by_grade": Counter(),
        "by_direction": Counter(),
        "by_timeframe": Counter(),
        "by_location_quality": Counter(),
        "by_structure_quality": Counter(),
        "by_no_trade_reason": Counter(),
        "by_option_plan_available": Counter(),
    }
    status_changes = Counter()
    ages = []

    for row in rows or []:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").upper()
        trace = bos_displacement_shadow_for_setup(row, candle_data_by_symbol.get(ticker, pd.DataFrame()))
        traces.append(trace)
        current_counts[trace.get("current_displacement") or "UNKNOWN"] += 1
        bos_counts[trace.get("bos_linked_displacement") or "UNKNOWN"] += 1
        for event in trace.get("bos_events") or []:
            reason = event.get("invalidation_reason")
            if reason:
                invalidation_counts[reason] += 1
        if trace.get("would_change_status"):
            status_changes[f"{trace.get('live_status')} -> {trace.get('shadow_status_if_bos_displacement_used')}"] += 1
        if trace.get("would_recover_strong_displacement"):
            ev = row.get("trade_eval") or {}
            recovered_breakdown["by_grade"][row.get("setupGrade") or "unknown"] += 1
            recovered_breakdown["by_direction"][row.get("direction") or "unknown"] += 1
            recovered_breakdown["by_timeframe"][row.get("timeframe") or "unknown"] += 1
            recovered_breakdown["by_location_quality"][ev.get("location") or "unknown"] += 1
            recovered_breakdown["by_structure_quality"][ev.get("structure_quality") or "unknown"] += 1
            recovered_breakdown["by_option_plan_available"][str((row.get("option_plan") or {}).get("available") is True)] += 1
            reasons = ev.get("no_trade_reasons") or ["none"]
            for reason in reasons:
                recovered_breakdown["by_no_trade_reason"][reason] += 1
            age = trace.get("bos_linked_displacement_age_bars")
            if isinstance(age, (int, float)):
                ages.append(age)

    recovered = [trace for trace in traces if trace.get("would_recover_strong_displacement")]
    changed = [trace for trace in traces if trace.get("would_change_status")]
    shadow_enter_now = [trace for trace in changed if trace.get("shadow_status_if_bos_displacement_used") == "ENTER_NOW"]
    shadow_early = [trace for trace in changed if trace.get("shadow_status_if_bos_displacement_used") == "EARLY_ENTRY"]
    shadow_almost = [trace for trace in changed if trace.get("shadow_status_if_bos_displacement_used") == "ALMOST_READY"]
    ages_sorted = sorted(ages)
    p75_index = int(math.ceil(len(ages_sorted) * 0.75) - 1) if ages_sorted else None
    median = ages_sorted[len(ages_sorted) // 2] if ages_sorted else None
    p75 = ages_sorted[p75_index] if p75_index is not None and p75_index >= 0 else None

    return {
        "status": "ready",
        "message": "Shadow study only. Live strategy unchanged.",
        "processed_setups": len(traces),
        "current_displacement_distribution": dict(sorted(current_counts.items())),
        "bos_linked_displacement_distribution": dict(sorted(bos_counts.items())),
        "recovered_strong_displacement_setups": len(recovered),
        "status_changes": sum(status_changes.values()),
        "status_change_distribution": dict(sorted(status_changes.items())),
        "enter_now_recoveries": len(shadow_enter_now),
        "early_entry_recoveries": len(shadow_early),
        "almost_ready_recoveries": len(shadow_almost),
        "unchanged_setups": len(traces) - len(changed),
        "invalidated_bos_evidence": sum(invalidation_counts.values()),
        "invalidation_counts": dict(sorted(invalidation_counts.items())),
        "ambiguous_bos_to_ob_links": sum(1 for trace in traces if trace.get("linked_ob_ambiguous")),
        "median_bos_age_bars": median,
        "p75_bos_age_bars": p75,
        "recovered_breakdown": {
            key: dict(sorted(counter.items()))
            for key, counter in recovered_breakdown.items()
        },
        "evidence_guard": (
            "Insufficient evidence to replace the current displacement model."
            if len(recovered) < 20 else "Shadow study only. Live strategy unchanged."
        ),
        "representative_traces": recovered[:25],
        "all_traces": traces,
    }


def _displacement_read(df: pd.DataFrame, atr: float, bos_confirmed: bool, direction: str = "NEUTRAL") -> tuple:
    return detect_displacement(df, atr, direction, bos_confirmed)


def detect_liquidity_sweep(df: pd.DataFrame, swings: list, direction: str, lookback: int = 12) -> tuple:
    if direction not in ("LONG", "SHORT") or len(df) < 3:
        return False, None
    recent = df.tail(lookback)
    highs = [s for s in swings if s["type"] == "high"]
    lows = [s for s in swings if s["type"] == "low"]

    if direction == "LONG" and lows:
        level = lows[-1]["price"]
        swept = bool((recent["Low"] < level).any())
        return bool(swept), float(level)
    if direction == "SHORT" and highs:
        level = highs[-1]["price"]
        swept = bool((recent["High"] > level).any())
        return bool(swept), float(level)
    return False, None


def _detect_liquidity_sweep(df: pd.DataFrame, swings: list, direction: str, lookback: int = 12) -> bool:
    swept, _ = detect_liquidity_sweep(df, swings, direction, lookback=lookback)
    return swept


def detect_rejection(df: pd.DataFrame, direction: str, sweep_level: Optional[float], lookback: int = 5) -> bool:
    if direction not in ("LONG", "SHORT") or sweep_level is None or len(df) < 2:
        return False

    recent = df.tail(lookback)
    for _, candle in recent.iterrows():
        high = float(candle["High"])
        low = float(candle["Low"])
        open_ = float(candle["Open"])
        close = float(candle["Close"])
        body = abs(close - open_)
        candle_range = high - low
        if candle_range <= 0:
            continue

        upper_wick = high - max(open_, close)
        lower_wick = min(open_, close) - low
        if direction == "LONG":
            closed_back_above = low < sweep_level and close > sweep_level
            lower_wick_failure = low < sweep_level and lower_wick >= max(body * 1.25, candle_range * 0.35)
            if closed_back_above or (lower_wick_failure and close > open_):
                return True
        elif direction == "SHORT":
            closed_back_below = high > sweep_level and close < sweep_level
            upper_wick_failure = high > sweep_level and upper_wick >= max(body * 1.25, candle_range * 0.35)
            if closed_back_below or (upper_wick_failure and close < open_):
                return True
    return False


def detect_structure_break(df: pd.DataFrame, swings: list, direction: str, lookback: int = 40) -> tuple:
    return _detect_bos(df, swings, direction, lookback=lookback)


def _nearest_target(price: float, direction: str, swings: list, fallback: float = 0.0, min_target: float = 0.0, atr: float = 0.0) -> Optional[float]:
    # Use entry as floor/ceiling so target is always beyond where we'd enter.
    # Also skip swings within 1×ATR of entry — minor structure that won't give real reward.
    if direction == "LONG":
        floor = max(price, min_target)
        skip_below = floor + atr if atr > 0 else floor
        highs = sorted([s["price"] for s in swings if s["type"] == "high" and s["price"] > skip_below])
        if highs:
            return float(highs[0])
        # Fallback: first swing above floor even if within 1 ATR (better than nothing)
        highs_any = sorted([s["price"] for s in swings if s["type"] == "high" and s["price"] > floor])
        if highs_any:
            return float(highs_any[0])
        return float(fallback) if fallback and fallback > floor else None
    if direction == "SHORT":
        ceiling = min(price, min_target) if min_target > 0 else price
        skip_above = ceiling - atr if atr > 0 else ceiling
        lows = sorted([s["price"] for s in swings if s["type"] == "low" and s["price"] < skip_above], reverse=True)
        if lows:
            return float(lows[0])
        lows_any = sorted([s["price"] for s in swings if s["type"] == "low" and s["price"] < ceiling], reverse=True)
        if lows_any:
            return float(lows_any[0])
        return float(fallback) if fallback and fallback < ceiling else None
    return None


def _room_to_target(
    price: float,
    direction: str,
    swings: list,
    entry: Optional[float] = None,
    stop: Optional[float] = None,
    fallback_target: float = 0.0,
    atr: float = 0.0,
) -> dict:
    target = _nearest_target(price, direction, swings, fallback_target, min_target=entry if entry is not None else 0.0, atr=atr)
    if target is None or direction not in ("LONG", "SHORT"):
        return {
            "target": None,
            "percent_to_target": None,
            "estimated_rr": None,
            "blocked": False,
            "clear": False,
            "label": "No clean structural target",
        }

    distance = (target - price) if direction == "LONG" else (price - target)
    pct = _safe_ratio(distance, price) * 100
    estimated_rr = None
    blocked = False
    if entry is not None and stop is not None:
        risk = abs(entry - stop)
        reward = (target - entry) if direction == "LONG" else (entry - target)
        if risk > 0 and reward > 0:
            estimated_rr = reward / risk
            blocked = estimated_rr < 1.5
        elif risk > 0:
            estimated_rr = 0.0
            blocked = True

    return {
        "target": round(target, 2),
        "percent_to_target": round(max(0.0, pct), 1),
        "estimated_rr": round(estimated_rr, 2) if estimated_rr is not None else None,
        "blocked": blocked,
        "clear": distance > 0 and not blocked,
        "label": "Blocked: RR < 1.5:1" if blocked else "Clear path to target",
    }


def _cap_quality_to_c(quality: dict, reason: str) -> dict:
    capped = dict(quality or {})
    if capped.get("score", 0) > 58:
        capped["score"] = 58
    capped["grade"] = "C"
    capped["grade_cap"] = min(capped.get("grade_cap", 100), 58)
    prior = capped.get("grade_note", "")
    capped["grade_note"] = reason if not prior or prior == "No grade cap applied." else f"{reason} {prior}"
    return capped


# STRATEGY FREEZE v1.0
# Do not modify qualification, direction, grading, entry, stop, target,
# confirmation, or trade-stage logic without an explicit strategy version change.
def _build_trade_stage_eval(
    *,
    df: pd.DataFrame,
    swings: list,
    trend: str,
    price: float,
    atr: float,
    bos_confirmed: bool,
    ob: Optional[dict],
    in_ob: bool,
    near_ob: bool,
    cleanliness: str,
    entry: Optional[float] = None,
    stop: Optional[float] = None,
    fallback_target: float = 0.0,
    macro_conflict: bool = False,
    context_conflict: bool = False,
) -> dict:
    location, location_pct = _strict_location(price, swings)
    bos_confirmed, bos_level = detect_structure_break(df, swings, trend)
    structure_quality = _structure_quality(trend, bos_confirmed, cleanliness)
    displacement, displacement_score = detect_displacement(df, atr, trend, bos_confirmed)
    sweep_taken, sweep_level = detect_liquidity_sweep(df, swings, trend)
    rejection_confirmed = detect_rejection(df, trend, sweep_level)
    room = _room_to_target(price, trend, swings, entry, stop, fallback_target, atr=atr)

    if location_pct is None:
        valid_zone = False
        preferred_fib_zone = False
    elif trend == "LONG":
        valid_zone = location_pct <= 50.0
        preferred_fib_zone = 21.0 <= location_pct <= 50.0
    elif trend == "SHORT":
        valid_zone = location_pct >= 50.0
        preferred_fib_zone = 50.0 <= location_pct <= 79.0
    else:
        valid_zone = False
        preferred_fib_zone = False

    htf_bias_clear = trend in ("LONG", "SHORT")
    htf_aligned = htf_bias_clear and not macro_conflict and not context_conflict
    rr_ok = room.get("estimated_rr") is not None and room.get("estimated_rr") >= 2.0
    room_clear = room.get("clear") is True and not room.get("blocked")
    structure_event_forming = bool(bos_confirmed or sweep_taken or rejection_confirmed or ob or near_ob or in_ob)

    if bos_confirmed and ob and (in_ob or near_ob):
        setup_type = "CONTINUATION: BOS + retest"
    elif sweep_taken and rejection_confirmed and displacement in ("WEAK", "STRONG"):
        setup_type = "REVERSAL: sweep + rejection + displacement"
    else:
        setup_type = "NONE"

    missing_for_a_plus = []
    if not sweep_taken:
        missing_for_a_plus.append("Needs liquidity sweep")
    if not rejection_confirmed:
        missing_for_a_plus.append("Needs rejection")
    if displacement != "STRONG":
        missing_for_a_plus.append("Needs strong displacement")
    if not bos_confirmed:
        missing_for_a_plus.append("Needs real BOS")
    if not valid_zone:
        missing_for_a_plus.append("Needs valid premium/discount location")
    if not rr_ok or room.get("blocked"):
        missing_for_a_plus.append("Needs 1:2 RR")
    if not htf_aligned:
        missing_for_a_plus.append("Needs HTF alignment")

    coaching = []
    if not valid_zone:
        coaching.append("Wait for valid premium/discount location.")
    elif not preferred_fib_zone and location_pct is not None:
        coaching.append("Best A+ location is usually the 50-79 fib retracement zone.")
    if not sweep_taken:
        coaching.append("Wait for a liquidity sweep before calling this A+.")
    if not rejection_confirmed:
        coaching.append("Wait for rejection back through the swept level.")
    if not bos_confirmed:
        coaching.append("Wait for strong candle close beyond structure.")
    if displacement == "NONE":
        coaching.append("Wait for impulse move showing control.")
    if room.get("blocked"):
        coaching.append("Wait for better entry or wider target.")
    if not coaching:
        coaching.append("Wait for all A+ conditions to remain true through the trigger candle.")

    no_trade_reasons = []
    if not htf_bias_clear:
        no_trade_reasons.append("No clear HTF bias")
    if not valid_zone:
        no_trade_reasons.append("Poor premium/discount location")
    if structure_quality == "CHOPPY / INTERNAL ONLY":
        no_trade_reasons.append("Choppy/internal structure")
    if room.get("blocked"):
        no_trade_reasons.append("RR < 1.5:1")
    if macro_conflict or context_conflict:
        no_trade_reasons.append("Macro/context conflict")
    if setup_type == "NONE" and not structure_event_forming:
        no_trade_reasons.append("No setup type or forming structure event")

    trigger_confirmed = bool(
        sweep_taken
        and rejection_confirmed
        and displacement == "STRONG"
        and bos_confirmed
    )

    a_plus_ready = (
        htf_aligned
        and valid_zone
        and structure_quality == "CLEAN BOS"
        and sweep_taken
        and rejection_confirmed
        and displacement == "STRONG"
        and setup_type in ("CONTINUATION: BOS + retest", "REVERSAL: sweep + rejection + displacement")
        and room_clear
        and rr_ok
        and trigger_confirmed
    )

    worth_watching = (
        htf_bias_clear
        and valid_zone
        and structure_event_forming
        and not room.get("blocked")
        and (room_clear or room.get("target") is not None or room.get("estimated_rr") is None)
        and 1 <= len(missing_for_a_plus) <= 3
    )

    # B+ TRADEABLE: BOS + OB at zone + at least 1 confirming signal.
    # Relaxed location: up to 70% percentile for longs, down to 30% for shorts.
    # A+ requires strict discount/premium (<=50% / >=50%); B+ allows midrange approach.
    b_plus_zone = (
        location_pct is not None and (
            (trend == "LONG"  and location_pct <= 70.0) or
            (trend == "SHORT" and location_pct >= 30.0)
        )
    )
    directional_signals = sum([
        bool(sweep_taken),
        bool(rejection_confirmed),
        displacement != "NONE",
        structure_quality == "CLEAN BOS",
    ])

    # RR is informational for B+ — options RR differs from underlying price RR.
    # Show the RR value on the card; do not gate B+ on it.
    b_plus_tradeable = (
        htf_bias_clear
        and bos_confirmed
        and ob is not None
        and (in_ob or near_ob)
        and not macro_conflict
        and not context_conflict
        and b_plus_zone
        and directional_signals >= 1
    )

    if a_plus_ready:
        trade_stage = "A+ READY"
    elif b_plus_tradeable:
        trade_stage = "B+ TRADEABLE"
    elif no_trade_reasons:
        trade_stage = "RANGE / NO TRADE"
    elif worth_watching:
        trade_stage = "BUILDING / WATCHLIST"
    else:
        trade_stage = "RANGE / NO TRADE"

    return {
        "trade_stage": trade_stage,
        "stage_badge": (
            "🟢 A+ READY"         if trade_stage == "A+ READY"
            else "🔵 B+ TRADEABLE"    if trade_stage == "B+ TRADEABLE"
            else "🟡 BUILDING / WATCHLIST" if trade_stage == "BUILDING / WATCHLIST"
            else "🔴 RANGE / NO TRADE"
        ),
        "b_plus_tradeable": b_plus_tradeable,
        "location": location,
        "location_percentile": location_pct,
        "structure_quality": structure_quality,
        "displacement": displacement,
        "displacement_score": displacement_score,
        "setup_type": setup_type,
        "room_to_target": room,
        "valid_zone": valid_zone,
        "preferred_fib_zone": preferred_fib_zone,
        "sweep_taken": sweep_taken,
        "sweep_level": round(sweep_level, 2) if sweep_level is not None else None,
        "rejection_confirmed": rejection_confirmed,
        "htf_bias_clear": htf_bias_clear,
        "htf_aligned": htf_aligned,
        "structure_event_forming": structure_event_forming,
        "trigger_confirmed": trigger_confirmed,
        "missing_for_a_plus": missing_for_a_plus,
        "no_trade_reasons": no_trade_reasons,
        "a_plus_ready": a_plus_ready,
        "coaching": coaching,
    }


def _ob_touch_count(df: pd.DataFrame, ob: Optional[dict]) -> int:
    if not ob:
        return 0
    touches = 0
    for i in range(ob["index"] + 1, len(df)):
        high = float(df["High"].iloc[i])
        low = float(df["Low"].iloc[i])
        if high >= ob["low"] and low <= ob["high"]:
            touches += 1
    return touches


def _cleanliness_read(df: pd.DataFrame, lookback: int = 12) -> tuple:
    closes = df["Close"].astype(float).iloc[-lookback - 1:]
    if len(closes) < 4:
        return "Unclear", 0.0
    net_move = abs(float(closes.iloc[-1] - closes.iloc[0]))
    path = float(closes.diff().abs().sum())
    efficiency = _safe_ratio(net_move, path)
    if efficiency >= 0.58:
        return "Clean impulse", round(efficiency, 2)
    if efficiency <= 0.32:
        return "Choppy / overlapping", round(efficiency, 2)
    return "Readable but mixed", round(efficiency, 2)


def _build_chart_coach(
    df: pd.DataFrame,
    swings: list,
    direction: str,
    price: float,
    atr: float,
    bos_confirmed: bool,
    bos_level: float,
    ob: Optional[dict],
    in_ob: bool,
    near_ob: bool,
    risk: Optional[float] = None,
    entry: Optional[float] = None,
) -> dict:
    score = 40
    warnings = []

    has_trend = direction in ("LONG", "SHORT")
    if has_trend:
        score += 12
    if bos_confirmed:
        score += 14
    if ob:
        score += 14
    if in_ob:
        score += 12
    elif near_ob:
        score += 7

    location, location_pct = _location_read(price, direction, swings)
    if "late for fresh" in location:
        score -= 8
        warnings.append("Location is late; avoid chasing without acceptance or a cleaner pullback.")
    elif "better" in location:
        score += 5

    cleanliness, efficiency = _cleanliness_read(df)
    if cleanliness == "Clean impulse":
        score += 7
    elif cleanliness == "Choppy / overlapping":
        score -= 7
        warnings.append("Recent candles are overlapping; structure may be harder to trust.")

    touches = _ob_touch_count(df, ob)
    if not ob:
        freshness = "No order block"
    elif touches <= 1:
        freshness = "Fresh OB"
        score += 6
    elif touches == 2:
        freshness = "Tapped once"
    else:
        freshness = "Heavily revisited"
        score -= 8
        warnings.append("Order block has been revisited multiple times; reaction quality matters more.")

    bos_extension = None
    if bos_confirmed and bos_level and atr:
        bos_extension = abs(price - bos_level) / atr
        if bos_extension > 3:
            score -= 7
            warnings.append("Price is extended from the BOS level; wait for a reset or proof of continuation.")

    room_to_target = "Unclear"
    if risk and entry and bos_level:
        if direction == "LONG" and bos_level > entry:
            room_r = (bos_level - entry) / risk
            room_to_target = "Crowded overhead" if room_r < 1.2 else "Clear path to BOS"
        elif direction == "SHORT" and bos_level < entry:
            room_r = (entry - bos_level) / risk
            room_to_target = "Crowded below" if room_r < 1.2 else "Clear path to BOS"
        else:
            room_r = 2.0
            room_to_target = "BOS already cleared"
        if room_to_target.startswith("Crowded"):
            score -= 6
            warnings.append("Nearby structure may limit room before the first reaction area.")

    raw_score = int(max(0, min(100, round(score))))
    score, grade_cap, grade_note = _apply_grade_caps(
        raw_score,
        bos_confirmed=bos_confirmed,
        ob=ob,
        in_ob=in_ob,
        near_ob=near_ob,
        location=location,
        cleanliness=cleanliness,
        touches=touches,
        bos_extension=bos_extension,
        room_to_target=room_to_target,
    )

    if not bos_confirmed:
        coach_note = "Trend has not produced a confirmed break yet; keep this on watch instead of forcing it."
        training_prompt = "What exact candle close would prove structure has actually broken?"
    elif not ob:
        coach_note = "BOS is present, but the scanner has not found a clean order block to anchor risk."
        training_prompt = "Where is the last opposite-color candle before the displacement, and is it meaningful?"
    elif not (in_ob or near_ob):
        coach_note = "The idea has direction and structure, but price has not returned to the order block yet."
        training_prompt = "What price would bring this from interesting to actionable?"
    elif "late for fresh" in location:
        coach_note = "Direction is valid, but location is stretched; wait for acceptance or a cleaner pullback."
        training_prompt = "Are you seeing continuation acceptance, or are you buying/selling into the reaction area?"
    elif cleanliness == "Choppy / overlapping":
        coach_note = "The setup passes rules, but the path is messy; demand a cleaner reaction before committing."
        training_prompt = "Which candle would show decisive control instead of overlap?"
    else:
        coach_note = "Structure, location, and order-block context are aligned enough to study closely."
        training_prompt = "What would invalidate the order-block defense before entry?"

    if not warnings:
        warnings.append("No major visual warning; still wait for the chart to confirm the plan.")

    return {
        "score": score,
        "raw_score": raw_score,
        "grade": _grade_from_score(score),
        "grade_cap": grade_cap,
        "grade_note": grade_note,
        "location": location,
        "location_percentile": location_pct,
        "cleanliness": cleanliness,
        "efficiency": efficiency,
        "freshness": freshness,
        "touches": touches,
        "room_to_target": room_to_target,
        "bos_extension_atr": round(bos_extension, 2) if bos_extension is not None else None,
        "warning": warnings[0],
        "warnings": warnings,
        "coach_note": coach_note,
        "training_prompt": training_prompt,
    }


# ── Batch data helpers ────────────────────────────────────────────────────────

def _timeframe_label_for_interval(interval: str) -> str:
    normalized = str(interval or "").strip().lower()
    if normalized == "1d":
        return "1D"
    if normalized == "1wk":
        return "1W"
    if normalized == "4h":
        return "4H"
    return normalized.upper()


def _price_provider_for_interval(interval: str):
    provider_name = provider_name_for_timeframe(_timeframe_label_for_interval(interval))
    return build_market_data_provider(provider_name)


def _download_price_batch_raw(tickers: list, period: str, interval: str, provider=None) -> dict:
    """
    Download OHLCV data for multiple tickers through the configured provider.
    Returns {ticker: DataFrame}.  Falls back gracefully on any per-ticker error.
    """
    if not tickers:
        return {}
    active_provider = provider or _price_provider_for_interval(interval)
    fetch_start = time.perf_counter()
    try:
        _cache_record("api_prices", "call")
        raw = active_provider.download(
            tickers, period=period, interval=interval,
            progress=False, auto_adjust=True, group_by="ticker",
        )
    except Exception as e:
        _cache_record("api_prices", "error")
        _cache_record_value("api_prices_duration_ms", round((time.perf_counter() - fetch_start) * 1000, 1))
        logger.warning("[batch_download] provider=%s period=%s interval=%s symbols=%s error=%s", active_provider.name, period, interval, len(tickers), e)
        return {}
    _cache_record_value("api_prices_duration_ms", round((time.perf_counter() - fetch_start) * 1000, 1))

    result: dict = {}
    single = len(tickers) == 1

    for t in tickers:
        try:
            if single:
                df = _flatten_columns(raw.copy())
            else:
                df = raw[t].copy()  # group_by='ticker' gives (Ticker, PriceType)
                df = _flatten_columns(df)
            df = df.dropna(how="all")
            if len(df) >= 10:
                result[t] = df
        except Exception:
            pass

    return result


def _price_cache_key(ticker: str, period: str, interval: str, provider_name: str) -> tuple:
    return (str(ticker or "").upper(), period, interval, provider_name)


def _refresh_price_cache(tickers: list, period: str, interval: str, provider_name: str) -> None:
    symbols = list(dict.fromkeys([str(t).strip().upper() for t in tickers if str(t).strip()]))
    if not symbols:
        return
    provider = build_market_data_provider(provider_name)
    fetched = _download_price_batch_raw(symbols, period, interval, provider=provider)
    now = datetime.utcnow()
    with _cache_lock:
        for ticker, df in fetched.items():
            _price_cache[_price_cache_key(ticker, period, interval, provider.name)] = {
                "fetched_at": now,
                "data": df,
            }
    logger.info(
        "[cache] refreshed prices provider=%s period=%s interval=%s tickers=%s/%s",
        provider.name,
        period,
        interval,
        len(fetched),
        len(symbols),
    )


def _batch_download(tickers: list, period: str, interval: str) -> dict:
    if not tickers:
        return {}
    symbols = list(dict.fromkeys([str(t).strip().upper() for t in tickers if str(t).strip()]))
    duplicate_count = max(0, len([t for t in tickers if str(t).strip()]) - len(symbols))
    if duplicate_count:
        with _cache_lock:
            _cache_stats["prices_duplicate_symbols_eliminated"] += duplicate_count
    with _cache_lock:
        _cache_stats[f"prices_request_{interval}_count"] += 1
        _cache_stats[f"prices_request_{interval}_symbols"] += len(symbols)
    provider = _price_provider_for_interval(interval)
    provider_name = provider.name
    now = datetime.utcnow()
    result = {}
    missing = []
    stale = []

    with _cache_lock:
        for ticker in symbols:
            key = _price_cache_key(ticker, period, interval, provider_name)
            cached = _price_cache.get(key)
            if not cached:
                missing.append(ticker)
                _cache_record("prices", "miss")
                continue
            age = now - cached.get("fetched_at", now)
            data = cached.get("data")
            if data is None or getattr(data, "empty", True):
                missing.append(ticker)
                _cache_record("prices", "miss")
                continue
            result[ticker] = data.copy()
            age_seconds = max(0.0, age.total_seconds())
            _cache_record_value("prices_age_seconds", age_seconds)
            with _cache_lock:
                _cache_stats["prices_oldest_age_seconds"] = max(
                    float(_cache_stats.get("prices_oldest_age_seconds", 0) or 0),
                    age_seconds,
                )
            if age <= PRICE_CACHE_TTL:
                _cache_record("prices", "hit")
            else:
                stale.append(ticker)
                _cache_record("prices", "stale")

    if stale:
        _submit_background_job(("prices", provider_name, period, interval, tuple(stale)), _refresh_price_cache, stale, period, interval, provider_name)

    if missing:
        fetched = _download_price_batch_raw(missing, period, interval, provider=provider)
        fetched_at = datetime.utcnow()
        with _cache_lock:
            for ticker, df in fetched.items():
                _price_cache[_price_cache_key(ticker, period, interval, provider_name)] = {
                    "fetched_at": fetched_at,
                    "data": df,
                }
                result[ticker] = df.copy()

    return result


def _attach_current_quotes(rows: list[dict]) -> None:
    """Attach display-only current quote prices without changing strategy price."""
    tickers = list(dict.fromkeys(
        str(row.get("ticker") or "").strip().upper()
        for row in rows
        if isinstance(row, dict) and str(row.get("ticker") or "").strip()
    ))
    if not tickers:
        return
    try:
        provider = build_market_data_provider(ALPACA_PROVIDER_NAME)
        quotes = provider.latest_quotes(tickers)
    except Exception as exc:
        logger.warning("[quotes] current quote enrichment failed symbols=%s error=%s", len(tickers), type(exc).__name__)
        return

    attached = 0
    for row in rows:
        ticker = str(row.get("ticker") or "").strip().upper()
        quote = quotes.get(ticker) or {}
        price = quote.get("price")
        try:
            current_quote_price = round(float(price), 2)
        except (TypeError, ValueError):
            continue
        row["current_quote_price"] = current_quote_price
        row["current_quote_source"] = quote.get("source") or "alpaca_latest_quote"
        row["current_quote_timestamp"] = quote.get("timestamp")
        attached += 1
    logger.info("[quotes] attached current quote prices rows=%s/%s", attached, len(rows))


def _refresh_cached_best_contracts(limit: int = 40) -> None:
    with _cache_lock:
        keys = list(_best_contract_cache.keys())[:limit]
    for key in keys:
        try:
            ticker, direction, price_region = key
            entry = float(price_region)
        except Exception:
            continue
        _submit_background_job(("best_contract_periodic", ticker, direction, price_region), _refresh_best_contract, ticker, direction, entry)


def _background_refresh_loop() -> None:
    time.sleep(2)
    while True:
        try:
            if _scan_is_active():
                time.sleep(5)
                continue
            symbols = list(dict.fromkeys(WATCHLIST))
            daily_provider = provider_name_for_timeframe("1D")
            weekly_provider = provider_name_for_timeframe("1W")
            h4_provider = provider_name_for_timeframe("4H")
            if _periodic_refresh_due("prices_1d", 180):
                _submit_background_job(("prices_periodic", daily_provider, "1y", "1d"), _refresh_price_cache, symbols, "1y", "1d", daily_provider)
            if _periodic_refresh_due("prices_1wk", 300):
                _submit_background_job(("prices_periodic", weekly_provider, "2y", "1wk"), _refresh_price_cache, symbols, "2y", "1wk", weekly_provider)
            if _periodic_refresh_due("prices_4h", 300):
                _submit_background_job(("prices_periodic", h4_provider, "60d", "4h"), _refresh_price_cache, symbols, "60d", "4h", h4_provider)

            if _periodic_refresh_due("option_expirations", 600):
                for ticker in list(STOCK_UNIVERSE_FILTER.get("allowlist", []))[:24]:
                    _submit_background_job(("option_expirations_periodic", ticker), _refresh_option_expirations, ticker)

            if _periodic_refresh_due("earnings", 6 * 60 * 60):
                for ticker in [s for s in symbols if s not in NO_EARNINGS_SYMBOLS][:40]:
                    _submit_earnings_refresh_if_safe(ticker, reason="periodic")

            with _background_jobs_lock:
                periodic_tasks = list(_background_periodic_tasks.items())
            for key, task in periodic_tasks:
                ttl_seconds = int(task.get("ttl_seconds") or 0)
                callback = task.get("callback")
                if ttl_seconds > 0 and callable(callback) and _periodic_refresh_due(f"periodic:{key}", ttl_seconds):
                    try:
                        callback()
                    except Exception as task_exc:
                        logger.warning("[background] periodic task failed key=%s error=%s", key, task_exc)
        except Exception as exc:
            logger.warning("[background] refresh loop error: %s", exc)
        time.sleep(30)


def _ensure_background_refresh_started() -> None:
    global _background_refresh_started
    with _background_jobs_lock:
        if _background_refresh_started:
            return
        _background_refresh_started = True
    thread = threading.Thread(target=_background_refresh_loop, daemon=True, name="kairos-market-cache-refresh")
    thread.start()


def start_market_cache_refresh() -> None:
    _ensure_background_refresh_started()
    key = _analysis_cache_key(None)
    _submit_analysis_refresh(key, None, reason="startup")


# ── Trending list analysis ───────────────────────────────────────────────────

def _atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    prev_close = df["Close"].astype(float).shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _relative_volume(df: pd.DataFrame, lookback: int = 20) -> float:
    if df is None or len(df) < lookback + 1:
        return 0.0
    current = float(df["Volume"].iloc[-1])
    avg = float(df["Volume"].iloc[-lookback - 1:-1].mean())
    return round(current / avg, 2) if avg > 0 else 0.0


def _atr_expansion(df: pd.DataFrame) -> float:
    atr = _atr_series(df).dropna()
    if len(atr) < 15:
        return 0.0
    current = float(atr.iloc[-1])
    baseline = float(atr.iloc[-15:-1].mean())
    return round(current / baseline, 2) if baseline > 0 else 0.0


def _ema_trend_alignment(df: pd.DataFrame) -> dict:
    close = df["Close"].astype(float)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    price = float(close.iloc[-1])
    e20 = float(ema20.iloc[-1])
    e50 = float(ema50.iloc[-1])
    e20_prev = float(ema20.iloc[-5]) if len(ema20) >= 5 else e20
    bullish = price > e20 > e50 and e20 > e20_prev
    bearish = price < e20 < e50 and e20 < e20_prev
    return {
        "price": price,
        "ema20": e20,
        "ema50": e50,
        "direction": "Bullish" if bullish else "Bearish" if bearish else "Neutral",
    }


def _clean_pullback(df: pd.DataFrame, direction: str, ema: dict) -> bool:
    if direction == "Neutral" or len(df) < 25:
        return False
    recent = df.tail(8)
    price = ema["price"]
    e20 = ema["ema20"]
    e50 = ema["ema50"]
    if direction == "Bullish":
        return (
            float(recent["Low"].min()) <= e20 * 1.015
            and float(recent["Close"].min()) >= e50 * 0.985
            and price <= e20 * 1.08
        )
    return (
        float(recent["High"].max()) >= e20 * 0.985
        and float(recent["Close"].max()) <= e50 * 1.015
        and price >= e20 * 0.92
    )


def _daily_aligns(df: Optional[pd.DataFrame], direction: str) -> bool:
    if df is None or len(df) < 55 or direction == "Neutral":
        return False
    daily_ema = _ema_trend_alignment(_flatten_columns(df.copy()).astype(float))
    return daily_ema["direction"] == direction


def _structure_direction_label(structure_trend: str) -> str:
    if structure_trend == "LONG":
        return "Bullish"
    if structure_trend == "SHORT":
        return "Bearish"
    return "Neutral"


def _resolve_trend_display_direction(ticker: str, ema_direction: str, structure_trend: str) -> tuple:
    structure_direction = _structure_direction_label(structure_trend)
    if ema_direction != "Neutral" and ema_direction == structure_direction:
        return ema_direction, None

    if ema_direction == "Neutral" and structure_direction == "Neutral":
        return "Neutral", None

    warning = (
        f"[trends] {ticker} mixed direction: EMA={ema_direction}, "
        f"structure={structure_direction}; displaying Neutral"
    )
    logger.warning(warning)
    return "Neutral", warning


def _validate_trend_direction_notes(ticker: str, direction: str, notes: list) -> Optional[str]:
    notes_text = " ".join(notes).lower()
    disagrees = (
        (direction == "Bullish" and ("bearish" in notes_text or "short" in notes_text))
        or (direction == "Bearish" and ("bullish" in notes_text or "long" in notes_text))
    )
    if not disagrees:
        return None
    warning = f"[trends] {ticker} direction/note disagreement: direction={direction}, notes={' | '.join(notes)}"
    logger.warning(warning)
    return warning


def analyze_trend_ticker(ticker: str, h4_df: Optional[pd.DataFrame] = None, daily_df: Optional[pd.DataFrame] = None) -> Optional[dict]:
    try:
        if h4_df is None:
            h4_df = _batch_download([ticker], period="60d", interval="4h").get(ticker, pd.DataFrame())
        df = _flatten_columns(h4_df.copy()).dropna().astype(float)
        if len(df) < 55:
            return None

        price = float(df["Close"].iloc[-1])
        if price < 5:
            return None

        swings = _find_swings(df, margin=4)
        structure_trend = _get_trend(swings)
        ema = _ema_trend_alignment(df)
        direction = ema["direction"]
        display_direction, direction_warning = _resolve_trend_display_direction(ticker, direction, structure_trend)
        trend_matches_structure = (
            (direction == "Bullish" and structure_trend == "LONG")
            or (direction == "Bearish" and structure_trend == "SHORT")
        )
        rel_vol = _relative_volume(df)
        atr_x = _atr_expansion(df)
        pullback = _clean_pullback(df, direction, ema)
        daily_ok = _daily_aligns(daily_df, direction)
        location, location_pct = _strict_location(price, swings)
        cleanliness, efficiency = _cleanliness_read(df)
        choppy = direction == "Neutral" or cleanliness == "Choppy / overlapping" or location == "MIDRANGE"

        ema_score = 25 if direction != "Neutral" else 0
        structure_score = 25 if trend_matches_structure else 8 if structure_trend != "NEUTRAL" else 0
        timeframe_score = 15 if daily_ok else 8 if direction != "Neutral" else 0
        volume_score = min(15, max(0, int((rel_vol - 0.8) / 0.7 * 15))) if rel_vol else 0
        atr_score = min(15, max(0, int((atr_x - 0.9) / 0.5 * 15))) if atr_x else 0
        pullback_score = 10 if pullback else 0
        choppy_penalty = 18 if choppy else 0
        score = max(0, min(100, ema_score + structure_score + timeframe_score + volume_score + atr_score + pullback_score - choppy_penalty))

        display_choppy = display_direction == "Neutral" or choppy
        status = "Strong Trend" if score >= 70 and not display_choppy else "Developing" if score >= 45 else "Choppy"
        notes = []
        if display_direction != "Neutral":
            notes.append("4H structure agrees with EMA trend")
        elif direction_warning:
            notes.append("EMA trend and 4H structure are mixed; wait for agreement")
        else:
            notes.append("No clean EMA20/EMA50 trend stack")
        if daily_ok and display_direction != "Neutral":
            notes.append("Daily confirms 4H direction")
        if pullback and display_direction != "Neutral":
            notes.append("Controlled pullback near EMA20")
        if choppy:
            notes.append("Avoid: choppy or mid-range chart")
        note_warning = _validate_trend_direction_notes(ticker, display_direction, notes)
        debug_warning = direction_warning or note_warning

        return {
            "ticker": ticker,
            "trend_direction": display_direction,
            "trend_score": score,
            "timeframe_alignment": "4H + Daily" if daily_ok and display_direction != "Neutral" else "4H only" if display_direction != "Neutral" else "Mixed",
            "relative_volume": rel_vol,
            "atr_expansion": atr_x,
            "clean_pullback": "Yes" if pullback and display_direction != "Neutral" else "No",
            "notes": "; ".join(notes),
            "status": status,
            "location": location,
            "location_percentile": location_pct,
            "cleanliness": cleanliness,
            "efficiency": efficiency,
            "debug_warning": debug_warning,
        }
    except Exception as e:
        logger.warning("[trends] ticker=%s error=%s", ticker, e)
        return None


def scan_trends(direction: str = "all", min_score: int = 0, hide_choppy: bool = False, limit: int = 10) -> list:
    universe = list(TRENDING_UNIVERSE)
    h4_data = _batch_download(universe, period="60d", interval="4h")
    daily_data = _batch_download(universe, period="1y", interval="1d")

    rows = []
    for ticker in universe:
        row = analyze_trend_ticker(ticker, h4_data.get(ticker), daily_data.get(ticker))
        if row:
            rows.append(row)

    direction = (direction or "all").lower()
    if direction == "bullish":
        rows = [r for r in rows if r["trend_direction"] == "Bullish"]
    elif direction == "bearish":
        rows = [r for r in rows if r["trend_direction"] == "Bearish"]
    rows = [r for r in rows if r["trend_score"] >= min_score]
    if hide_choppy:
        rows = [r for r in rows if r["status"] != "Choppy"]

    rows.sort(key=lambda r: (-r["trend_score"], r["ticker"]))
    return rows[:limit]


# ── Main Analysis ─────────────────────────────────────────────────────────────
# STRATEGY FREEZE v1.0
# Do not modify qualification, direction, grading, entry, stop, target,
# confirmation, or trade-stage logic without an explicit strategy version change.

def analyze_ticker(
    ticker: str,
    _daily_df: Optional[pd.DataFrame] = None,
    _weekly_df: Optional[pd.DataFrame] = None,
    timeframe: str = "1D",
) -> Optional[dict]:
    try:
        if _daily_df is not None:
            df = _flatten_columns(_daily_df.copy())
        else:
            raw = _batch_download([ticker], period="1y", interval="1d").get(ticker, pd.DataFrame())
            df  = _flatten_columns(raw)

        if len(df) < 50:
            return None

        df    = df.astype(float)
        close = df["Close"]
        price = float(close.iloc[-1])

        # Ticker quality filters
        if price < 5:
            return None
        avg_dollar_vol = float((df["Close"] * df["Volume"]).iloc[-20:].mean())
        if avg_dollar_vol < 5_000_000:
            return None
        daily_range_pct = (float(df["High"].iloc[-1]) - float(df["Low"].iloc[-1])) / price * 100
        if daily_range_pct < 0.5:
            return None

        atr = _compute_atr(df)
        rsi = round(_compute_rsi(close), 1)
        ema20 = round(float(close.ewm(span=20, adjust=False).mean().iloc[-1]), 2)
        ema50 = round(float(close.ewm(span=50, adjust=False).mean().iloc[-1]), 2)

        # Price action
        swing_tolerance = SWING_DAILY_PRICE_TOLERANCE if timeframe == "1D" else 0.0
        swings        = _find_swings(df, tolerance=swing_tolerance)
        trend         = _get_trend(swings)
        market_regime = _market_regime_for_df(df, df if timeframe == "1D" else _daily_df)
        bos_confirmed = False
        bos_level     = 0.0
        ob            = None
        in_ob = near_ob = False

        if trend != "NEUTRAL":
            bos_confirmed, bos_level = _detect_bos(df, swings, trend)
            if bos_confirmed:
                ob = _find_order_block(df, trend, swings)
                if ob:
                    in_ob = ob["low"] <= price <= ob["high"]
                    near_ob = not in_ob and (
                        (trend == "LONG"  and price < ob["high"] and price > ob["low"] - 0.75 * atr) or
                        (trend == "SHORT" and price > ob["low"]  and price < ob["high"] + 0.75 * atr)
                    )

        # ── Macro bias (52-week high context) ────────────────────────────────────
        macro_bias, pct_from_52w, wk52_high, window_high = _macro_bias(price, df)
        macro_label = (
            f"📉 {macro_bias} ({pct_from_52w:.0%} below 52w high ${wk52_high:.2f})"
            if macro_bias == "Macro Bearish"
            else f"📈 {macro_bias} (within {pct_from_52w:.0%} of 52w high ${wk52_high:.2f})"
            if macro_bias == "Macro Bullish"
            else f"〰️ {macro_bias} ({pct_from_52w:.0%} below 52w high ${wk52_high:.2f})"
        )
        print(f"[{ticker}] {macro_label}")

        # ── Macro bias hard block (LONG only) ────────────────────────────────
        # Suppress buying into macro downtrends. Never block shorts based on macro
        # bias — short signals use local structure detection only.
        macro_block = macro_bias == "Macro Bearish" and trend == "LONG"
        if macro_block:
            print(f"[{ticker}] Macro Bearish override — LONG marked no trade")

        # Market structure defaults
        structure = "ranging"
        choch = False
        struct_label = "🟡 Ranging"
        struct_aligned = False
        struct_note = "🟡 Ranging market — proceed with extra confirmation"

        choch_block = False
        if trend in ("LONG", "SHORT"):
            # Weekly EMA context (informational only — not a hard block)
            if _weekly_df is not None:
                weekly = _flatten_columns(_weekly_df.copy()).astype(float)
            else:
                w_raw = _batch_download([ticker], period="2y", interval="1wk").get(ticker, pd.DataFrame())
                weekly = _flatten_columns(w_raw).astype(float)
            if len(weekly) >= 50:
                w_close = weekly["Close"]
                w_e20 = float(w_close.ewm(span=20, adjust=False).mean().iloc[-1])
                w_e50 = float(w_close.ewm(span=50, adjust=False).mean().iloc[-1])
                print(
                    f"[{ticker}] weekly EMA20={w_e20:.2f} EMA50={w_e50:.2f} "
                    f"({'bearish cross' if w_e20 < w_e50 else 'bullish cross'})"
                )

            # Structure context — 200-candle lookback, margin=5 for confirmed swings
            # Informational only — structure/CHoCH labels show on cards but do NOT block signals.
            # Primary entry gate = BOS confirmed + price at/near OB.
            htf_df     = df.tail(200).reset_index(drop=True)
            htf_swing_tolerance = SWING_DAILY_PRICE_TOLERANCE if timeframe == "1D" else 0.0
            htf_swings = _find_swings(htf_df, margin=5, tolerance=htf_swing_tolerance)
            structure, struct_reasons = _market_structure(
                htf_swings, price, df, macro_bias=macro_bias, window_high=window_high
            )
            choch, choch_reason, bearish_choch_lvl, bullish_choch_lvl, choch_bar_idx = _detect_choch(htf_swings, trend)

            # Update structure label if CHoCH overrides the vote
            if bearish_choch_lvl is not None and price < bearish_choch_lvl:
                structure = "bearish"
                print(
                    f"[{ticker}] note: price {price:.2f} below bearish CHoCH level "
                    f"{bearish_choch_lvl:.2f} → structure reads bearish"
                )
            elif "bearish CHoCH" in choch_reason:
                structure = "bearish"
            elif "bullish CHoCH" in choch_reason:
                structure = "bullish"

            print(f"[{ticker}] trend={trend} structure={structure} choch={choch}")
            for r in struct_reasons:
                print(f"  {r}")
            print(f"  choch: {choch_reason}")

            # Price-based CHoCH block: only suppress if price has NOT cleared the CHoCH level.
            # A bearish CHoCH that price has already recovered above is no longer blocking.
            # A bullish CHoCH that price has already fallen below is no longer blocking.
            if choch and trend == "LONG":
                # Bearish CHoCH blocks longs only while price <= the broken swing-low level
                choch_block = bearish_choch_lvl is not None and price <= bearish_choch_lvl
                if choch and not choch_block:
                    print(f"[{ticker}] bearish CHoCH present but price ${price:.2f} > CHoCH level ${bearish_choch_lvl:.2f} — not blocking")
                elif choch_block:
                    print(f"[{ticker}] bearish CHoCH conflict — LONG marked no trade")
            elif choch and trend == "SHORT":
                # Bullish CHoCH blocks shorts only while price >= the broken swing-high level
                choch_block = bullish_choch_lvl is not None and price >= bullish_choch_lvl
                if choch and not choch_block:
                    print(f"[{ticker}] bullish CHoCH present but price ${price:.2f} < CHoCH level ${bullish_choch_lvl:.2f} — not blocking")
                elif choch_block:
                    print(f"[{ticker}] bullish CHoCH conflict — SHORT marked no trade")
            else:
                choch_block = False

            struct_label = (
                ("🔴 Bearish ChoCH" if trend == "LONG" else "🟢 Bullish ChoCH") if choch
                else ("🔴 Bearish Structure" if structure == "bearish"
                      else ("🟢 Bullish Structure" if structure == "bullish" else "🟡 Ranging"))
            )
            struct_aligned = (
                (trend == "LONG" and structure == "bullish")
                or (trend == "SHORT" and structure == "bearish")
            )
            struct_note = (
                "✅ Structure aligned with " + trend + " setup" if struct_aligned
                else "🟡 Ranging market — proceed with extra confirmation" if structure == "ranging"
                else "⚠️ Counter-trend setup — structure is " + structure + ", proceed with caution"
            )

        quality = _build_chart_coach(
            df=df,
            swings=swings,
            direction=trend,
            price=price,
            atr=atr,
            bos_confirmed=bos_confirmed,
            bos_level=bos_level,
            ob=ob,
            in_ob=in_ob,
            near_ob=near_ob,
        )
        trade_eval = _build_trade_stage_eval(
            df=df,
            swings=swings,
            trend=trend,
            price=price,
            atr=atr,
            bos_confirmed=bos_confirmed,
            ob=ob,
            in_ob=in_ob,
            near_ob=near_ob,
            cleanliness=quality.get("cleanliness", "Unclear"),
            fallback_target=window_high,
            macro_conflict=macro_block,
            context_conflict=choch_block,
        )
        if trade_eval["no_trade_reasons"] and not trade_eval.get("b_plus_tradeable"):
            quality = _cap_quality_to_c(quality, "Strict scout rules cap this at C.")
        if macro_block or choch_block:
            reason = "Macro/CHoCH conflict caps this at C."
            quality = _cap_quality_to_c(quality, reason)
            if macro_block:
                trade_eval["no_trade_reasons"].append("Macro bearish conflict")
                trade_eval["coaching"].append("Wait for macro context to stop fighting the long idea.")
            if choch_block:
                trade_eval["no_trade_reasons"].append("Counter CHoCH conflict")
                trade_eval["coaching"].append("Wait for structure to realign after the CHoCH.")
            trade_eval["trade_stage"] = "RANGE / NO TRADE"
            trade_eval["a_plus_ready"] = False
            trade_eval["b_plus_tradeable"] = False

        # ── Near-miss: has direction but setup is incomplete ──────────────────
        if trend == "NEUTRAL" or not bos_confirmed or ob is None:
            return {
                "ticker":        ticker,
                "timeframe":     timeframe,
                "direction":     trend if trend != "NEUTRAL" else None,
                "price":         round(price, 2),
                "atr":           round(atr, 2),
                "rsi":           rsi,
                "ema20":         ema20,
                "ema50":         ema50,
                "trend":         trend,
                "bos_confirmed": bos_confirmed,
                "bos_level":     round(bos_level, 2) if bos_level else None,
                "ob_high":       round(ob["high"], 2) if ob else None,
                "ob_low":        round(ob["low"],  2) if ob else None,
                "in_ob":         in_ob,
                "near_ob":       near_ob,
                "quality":        quality,
                "trade_eval":     trade_eval,
                "structure":      structure,
                "structureLabel": struct_label,
                "structureNote":  struct_note,
                "macroBias":      macro_bias,
                "macroLabel":     macro_label,
                "wk52High":       wk52_high,
                "pctFromHigh":    pct_from_52w,
                "setup_status":   "DEVELOPING" if trend != "NEUTRAL" else "SKIPPED",
                "market_regime":  market_regime.get("regime"),
                "market_regime_score": market_regime.get("score"),
                "market_regime_details": market_regime,
                "signal_timestamp": df.index[-1].tz_convert("UTC").isoformat().replace("+00:00", "Z") if df.index[-1].tzinfo is not None else df.index[-1].isoformat() + "Z",
                "scannedAt":      datetime.utcnow().isoformat() + "Z",
            }

        # ── Qualified setup: BOS confirmed + price at/near OB ────────────────
        ob_mid = (ob["high"] + ob["low"]) / 2
        entry  = round(ob_mid, 2)
        sl     = round(ob["low"] - 0.5 * atr, 2) if trend == "LONG" else round(ob["high"] + 0.5 * atr, 2)
        risk   = round(abs(entry - sl), 2)
        tp1    = round(entry + 2 * risk, 2) if trend == "LONG" else round(entry - 2 * risk, 2)
        tp2    = round(entry + 3 * risk, 2) if trend == "LONG" else round(entry - 3 * risk, 2)
        tp3    = round(entry + 4 * risk, 2) if trend == "LONG" else round(entry - 4 * risk, 2)

        option = _suggest_option(ticker, trend, entry)

        checklist = {
            "trendConfirmed":  True,
            "bosConfirmed":    bos_confirmed,
            "obFound":         ob is not None,
            "priceAtOb":       in_ob or near_ob,
            "structureAligned": struct_aligned,
            "chochClear":      not choch,
        }

        quality = _build_chart_coach(
            df=df,
            swings=swings,
            direction=trend,
            price=price,
            atr=atr,
            bos_confirmed=bos_confirmed,
            bos_level=bos_level,
            ob=ob,
            in_ob=in_ob,
            near_ob=near_ob,
            risk=risk,
            entry=entry,
        )
        trade_eval = _build_trade_stage_eval(
            df=df,
            swings=swings,
            trend=trend,
            price=price,
            atr=atr,
            bos_confirmed=bos_confirmed,
            ob=ob,
            in_ob=in_ob,
            near_ob=near_ob,
            cleanliness=quality.get("cleanliness", "Unclear"),
            entry=entry,
            stop=sl,
            fallback_target=window_high,
            macro_conflict=macro_block,
            context_conflict=choch_block,
        )
        if trade_eval["no_trade_reasons"] and not trade_eval.get("b_plus_tradeable"):
            quality = _cap_quality_to_c(quality, "Strict scout rules cap this at C.")
        if macro_block or choch_block:
            reason = "Macro/CHoCH conflict caps this at C."
            quality = _cap_quality_to_c(quality, reason)
            if macro_block:
                trade_eval["no_trade_reasons"].append("Macro bearish conflict")
                trade_eval["coaching"].append("Wait for macro context to stop fighting the long idea.")
            if choch_block:
                trade_eval["no_trade_reasons"].append("Counter CHoCH conflict")
                trade_eval["coaching"].append("Wait for structure to realign after the CHoCH.")
            trade_eval["trade_stage"] = "RANGE / NO TRADE"
            trade_eval["a_plus_ready"] = False
            trade_eval["b_plus_tradeable"] = False

        # Gate: only send to near-miss if no BOS or no OB — price location no longer blocks.
        # Setups with BOS + OB show as QUALIFIED regardless of in_ob/near_ob status.
        if not trade_eval["a_plus_ready"] and not trade_eval.get("b_plus_tradeable") and ob is None:
            return {
                "ticker":        ticker,
                "timeframe":     timeframe,
                "direction":     trend,
                "price":         round(price, 2),
                "atr":           round(atr, 2),
                "rsi":           rsi,
                "ema20":         ema20,
                "ema50":         ema50,
                "trend":         trend,
                "bos_confirmed": bos_confirmed,
                "bos_level":     round(bos_level, 2),
                "ob_high":       round(ob["high"], 2),
                "ob_low":        round(ob["low"],  2),
                "in_ob":         in_ob,
                "near_ob":       near_ob,
                "quality":        quality,
                "trade_eval":     trade_eval,
                "structure":      structure,
                "structureLabel": struct_label,
                "structureNote":  struct_note,
                "macroBias":      macro_bias,
                "macroLabel":     macro_label,
                "wk52High":       wk52_high,
                "pctFromHigh":    pct_from_52w,
                "setup_status":   "DEVELOPING",
                "market_regime":  market_regime.get("regime"),
                "market_regime_score": market_regime.get("score"),
                "market_regime_details": market_regime,
                "signal_timestamp": df.index[-1].tz_convert("UTC").isoformat().replace("+00:00", "Z") if df.index[-1].tzinfo is not None else df.index[-1].isoformat() + "Z",
                "scannedAt":      datetime.utcnow().isoformat() + "Z",
            }

        return {
            "ticker":        ticker,
            "timeframe":     timeframe,
            "direction":     trend,
            "price":         round(price, 2),
            "atr":           round(atr, 2),
            "rsi":           rsi,
            "ema20":         ema20,
            "ema50":         ema50,
            "trend":         trend,
            "bos_confirmed": bos_confirmed,
            "bos_level":     round(bos_level, 2),
            "ob_high":       round(ob["high"], 2),
            "ob_low":        round(ob["low"],  2),
            "in_ob":         in_ob,
            "near_ob":       near_ob,
            "entry":         entry,
            "sl":            sl,
            "tp1":           tp1,
            "tp2":           tp2,
            "tp3":           tp3,
            "risk":          risk,
            "option":        option,
            "checklist":      checklist,
            "quality":        quality,
            "trade_eval":     trade_eval,
            "structure":      structure,
            "structureLabel": struct_label,
            "structureNote":  struct_note,
            "macroBias":      macro_bias,
            "macroLabel":     macro_label,
            "wk52High":       wk52_high,
            "pctFromHigh":    pct_from_52w,
            "setup_status":   "QUALIFIED",
            "market_regime":  market_regime.get("regime"),
            "market_regime_score": market_regime.get("score"),
            "market_regime_details": market_regime,
            "signal_timestamp": df.index[-1].tz_convert("UTC").isoformat().replace("+00:00", "Z") if df.index[-1].tzinfo is not None else df.index[-1].isoformat() + "Z",
            "scannedAt":      datetime.utcnow().isoformat() + "Z",
        }

    except Exception as e:
        print(f"[scanner] {ticker} error: {e}")
        return None


def debug_ticker(ticker: str) -> dict:
    """
    Run every filter step on a single ticker and return the full reasoning.
    Used by /api/debug/{ticker}.
    """
    out: dict = {"ticker": ticker.upper(), "filters": [], "passed": False}
    try:
        raw = _batch_download([ticker], period="1y", interval="1d").get(ticker, pd.DataFrame())
        df  = _flatten_columns(raw)
        if len(df) < 50:
            out["filters"].append({"step": "data", "result": "FAIL", "reason": "< 50 bars"})
            return out
        df    = df.astype(float)
        price = float(df["Close"].iloc[-1])
        atr   = _compute_atr(df)

        # Quality gates
        if price < 5:
            out["filters"].append({"step": "price", "result": "FAIL", "reason": f"price ${price:.2f} < $5"})
            return out
        avg_dv = float((df["Close"] * df["Volume"]).iloc[-20:].mean())
        if avg_dv < 5_000_000:
            out["filters"].append({"step": "volume", "result": "FAIL", "reason": f"avg dollar vol ${avg_dv/1e6:.1f}M < $5M"})
            return out
        dr = (float(df["High"].iloc[-1]) - float(df["Low"].iloc[-1])) / price * 100
        if dr < 0.5:
            out["filters"].append({"step": "range", "result": "FAIL", "reason": f"daily range {dr:.2f}% < 0.5%"})
            return out
        out["price"] = round(price, 2)
        out["atr"]   = round(atr, 2)

        # Trend
        swings = _find_swings(df, tolerance=SWING_DAILY_PRICE_TOLERANCE)
        trend  = _get_trend(swings)
        out["trend"] = trend
        out["filters"].append({"step": "trend", "result": "OK", "reason": trend})

        # BOS
        bos_confirmed, bos_level = (False, 0.0)
        ob = None
        in_ob = near_ob = False
        if trend != "NEUTRAL":
            bos_confirmed, bos_level = _detect_bos(df, swings, trend)
            if bos_confirmed:
                ob = _find_order_block(df, trend, swings)
                if ob:
                    in_ob   = ob["low"] <= price <= ob["high"]
                    near_ob = not in_ob and (
                        (trend == "LONG"  and price < ob["high"] and price > ob["low"] - 0.75 * atr) or
                        (trend == "SHORT" and price > ob["low"]  and price < ob["high"] + 0.75 * atr)
                    )
        out["bos_confirmed"] = bos_confirmed
        out["bos_level"]     = round(bos_level, 2) if bos_level else None
        out["ob"]            = {"high": round(ob["high"], 2), "low": round(ob["low"], 2)} if ob else None
        out["in_ob"]         = in_ob
        out["near_ob"]       = near_ob
        out["filters"].append({
            "step": "bos_ob",
            "result": "OK" if bos_confirmed else "WARN",
            "reason": f"BOS={'yes' if bos_confirmed else 'no'} OB={'yes' if ob else 'no'} in_ob={in_ob} near_ob={near_ob}",
        })

        # Macro bias
        macro_bias, pct_from_52w, wk52_high, window_high = _macro_bias(price, df)
        out["macro_bias"]     = macro_bias
        out["wk52_high"]      = wk52_high
        out["pct_from_52w"]   = f"{pct_from_52w:.1%}"
        macro_block = macro_bias == "Macro Bearish" and trend == "LONG"
        out["filters"].append({
            "step": "macro_bias",
            "result": "FAIL" if macro_block else "OK",
            "reason": (
                f"{macro_bias} ({pct_from_52w:.1%} below 52w high ${wk52_high:.2f})"
                + (" → LONG suppressed" if macro_block else "")
            ),
        })
        if macro_block:
            return out

        # Weekly EMA
        w_raw  = _batch_download([ticker], period="2y", interval="1wk").get(ticker, pd.DataFrame())
        weekly = _flatten_columns(w_raw).astype(float)
        w_e20 = w_e50 = None
        if len(weekly) >= 50:
            w_close = weekly["Close"]
            w_e20 = round(float(w_close.ewm(span=20, adjust=False).mean().iloc[-1]), 2)
            w_e50 = round(float(w_close.ewm(span=50, adjust=False).mean().iloc[-1]), 2)
        out["weekly_ema20"] = w_e20
        out["weekly_ema50"] = w_e50
        out["filters"].append({
            "step": "weekly_ema",
            "result": "INFO",
            "reason": (
                f"EMA20={w_e20} EMA50={w_e50} "
                f"({'bearish cross' if w_e20 and w_e20 < w_e50 else 'bullish cross'})"
                if w_e20 else "not enough weekly data"
            ),
        })

        # Local structure + CHoCH
        htf_df     = df.tail(200).reset_index(drop=True)
        htf_swings = _find_swings(htf_df, margin=5, tolerance=SWING_DAILY_PRICE_TOLERANCE)
        structure, struct_reasons = _market_structure(
            htf_swings, price, df, macro_bias=macro_bias, window_high=window_high
        )
        choch, choch_reason, bearish_choch_lvl, bullish_choch_lvl, choch_bar_idx = _detect_choch(htf_swings, trend)
        if bearish_choch_lvl is not None and price < bearish_choch_lvl:
            structure = "bearish"
        elif "bearish CHoCH" in choch_reason:
            structure = "bearish"
        elif "bullish CHoCH" in choch_reason:
            structure = "bullish"
        out["structure"]        = structure
        out["choch"]            = choch
        out["choch_reason"]     = choch_reason
        out["struct_reasons"]   = struct_reasons
        out["filters"].append({
            "step": "structure",
            "result": "INFO",
            "reason": f"structure={structure} choch={choch} | {choch_reason}",
        })

        # Final gate: BOS + OB
        if not bos_confirmed:
            out["filters"].append({"step": "final", "result": "FAIL", "reason": "no BOS confirmed"})
            return out
        if not (in_ob or near_ob):
            out["filters"].append({
                "step": "final", "result": "FAIL",
                "reason": f"price not at/near OB (ob={out['ob']})",
            })
            return out

        out["passed"] = True
        out["filters"].append({"step": "final", "result": "PASS", "reason": "BOS + OB confirmed"})
    except Exception as e:
        out["filters"].append({"step": "error", "result": "ERROR", "reason": str(e)})
    return out


def _setup_rank(result: Optional[dict]) -> tuple:
    if not result:
        return (-1, -1, -1, -1)
    grade_order = {"A+": 6, "A": 5, "B+": 4, "B": 3, "C+": 2, "C": 1, "D": 0}
    status_rank = 1 if result.get("setup_status") == "QUALIFIED" else 0
    grade_rank = grade_order.get(result.get("quality", {}).get("grade"), 0)
    timeframe_rank = 1 if result.get("timeframe") == "1D" else 0
    score = result.get("quality", {}).get("score", 0)
    return (status_rank, grade_rank, timeframe_rank, score)


def _best_timeframe_result(daily_result: Optional[dict], h4_result: Optional[dict]) -> Optional[dict]:
    if not daily_result:
        return h4_result
    if not h4_result:
        return daily_result
    return max([daily_result, h4_result], key=_setup_rank)


def _stock_direction_label(raw: Optional[str]) -> str:
    if raw == "LONG":
        return "Bullish"
    if raw == "SHORT":
        return "Bearish"
    return "Mixed"


def _trend_from_ohlc(df: Optional[pd.DataFrame]) -> str:
    if df is None or len(df) < 20:
        return "Mixed"
    try:
        clean = _flatten_columns(df.copy()).dropna().astype(float)
        swings = _find_swings(clean)
        return _stock_direction_label(_get_trend(swings))
    except Exception:
        return "Mixed"


def _result_direction(result: Optional[dict], fallback_df: Optional[pd.DataFrame] = None) -> str:
    if result and result.get("trend"):
        return _stock_direction_label(result.get("trend"))
    return _trend_from_ohlc(fallback_df)


def _stock_phase(daily_direction: str, h4_direction: str) -> str:
    if daily_direction == "Bullish" and h4_direction == "Bullish":
        return "Trend Move"
    if daily_direction == "Bullish" and h4_direction == "Bearish":
        return "Pullback"
    if daily_direction == "Bearish" and h4_direction == "Bearish":
        return "Trend Move"
    if daily_direction == "Bearish" and h4_direction == "Bullish":
        return "Pullback"
    return "Transition"


def _stock_setup_status(result: dict) -> tuple:
    ev = result.get("trade_eval") or {}
    if ev.get("trigger_confirmed") or ev.get("a_plus_ready"):
        return "Trend Resumption Confirmed", "trigger confirmed after pullback"
    if ev.get("displacement") == "STRONG" and result.get("bos_confirmed"):
        return "Strong Confirmation", "strong displacement with confirmed BOS"
    if result.get("bos_confirmed") or ev.get("rejection_confirmed"):
        return "Early Confirmation", "structure confirmation is forming"
    if result.get("in_ob") or result.get("near_ob") or ev.get("sweep_taken"):
        return "Pullback Complete", "price reacted at the active pullback zone"
    return "Pullback Active", "waiting for pullback completion and confirmation"


def _stock_location_label(ev: dict) -> str:
    pct = ev.get("location_percentile")
    if pct is not None:
        if pct <= 30:
            return "Discount"
        if pct >= 70:
            return "Premium"
        return "Mid"
    raw = str(ev.get("location") or "").upper()
    if "DISCOUNT" in raw:
        return "Discount"
    if "PREMIUM" in raw:
        return "Premium"
    return "Mid"


def _stock_entry_status(result: dict) -> tuple:
    entry = result.get("entry")
    price = result.get("price")
    atr = result.get("atr")
    if entry is None or price is None or not atr:
        return "Waiting", None, None
    distance = abs(float(price) - float(entry))
    distance_atr = distance / float(atr) if float(atr) > 0 else None
    distance_pct = (distance / float(price)) * 100 if float(price) > 0 else None
    if distance_atr is None:
        status = "Waiting"
    elif distance_atr <= 0.25:
        status = "Tradeable"
    elif distance_atr <= 0.50:
        status = "Near Entry"
    elif distance_atr <= 1.00:
        status = "Waiting"
    else:
        status = "Too Far"
    return (
        status,
        round(distance_atr, 2) if distance_atr is not None else None,
        round(distance_pct, 2) if distance_pct is not None else None,
    )


def _stock_trade_direction(result: dict) -> str:
    raw = str(result.get("direction") or result.get("trend") or "").upper()
    if raw in {"LONG", "CALL", "BULLISH"}:
        return "LONG"
    if raw in {"SHORT", "PUT", "BEARISH"}:
        return "SHORT"
    return "NEUTRAL"


def _stock_confirmation(result: dict, setup_direction: str, setup_status: str) -> tuple:
    ev = result.get("trade_eval") or {}
    trade_direction = _stock_trade_direction(result)
    structure_label = str(result.get("structureLabel") or result.get("structure") or "").lower()
    displacement = str(ev.get("displacement") or "").upper()
    in_zone = bool(result.get("in_ob") or result.get("near_ob"))

    bullish_flow = setup_direction == "Bullish" or "bullish" in structure_label
    bearish_flow = setup_direction == "Bearish" or "bearish" in structure_label
    status_has_confirmation = (
        "Early Confirmation" in setup_status
        or "Strong Confirmation" in setup_status
        or "Trend Resumption" in setup_status
    )

    def _ema_relation_matches(direction: str) -> bool:
        try:
            price = float(result["price"])
            ema20 = float(result["ema20"])
        except (KeyError, TypeError, ValueError):
            return False
        if direction == "LONG":
            return price > ema20
        if direction == "SHORT":
            return price < ema20
        return False

    candidates = []

    def add_signal(condition: bool, priority: int, reason: str) -> None:
        if condition:
            candidates.append((priority, reason))

    if ev.get("trigger_confirmed"):
        return True, "trigger confirmed after pullback"

    if trade_direction == "LONG":
        add_signal(ev.get("rejection_confirmed") and in_zone, 90, "strong bullish reaction from demand/support")
        add_signal(displacement == "STRONG" and bullish_flow, 80, "strong bullish reaction candle from support")
        add_signal(result.get("bos_confirmed") and bullish_flow, 70, "bullish structure shift after pullback")
        add_signal(_ema_relation_matches("LONG") and bullish_flow, 60, "reclaimed EMA20 with bullish short-term flow")
        add_signal(status_has_confirmation, 50, "bullish confirmation started")
        if candidates:
            return True, max(candidates, key=lambda item: item[0])[1]
        return False, "waiting for bullish structure shift or support reaction"

    if trade_direction == "SHORT":
        add_signal(ev.get("rejection_confirmed") and in_zone, 90, "strong bearish reaction from supply/resistance")
        add_signal(displacement == "STRONG" and bearish_flow, 80, "strong bearish reaction candle from resistance")
        add_signal(result.get("bos_confirmed") and bearish_flow, 70, "bearish structure shift after pullback")
        add_signal(_ema_relation_matches("SHORT") and bearish_flow, 60, "rejected EMA20 with bearish short-term flow")
        add_signal(status_has_confirmation, 50, "bearish confirmation started")
        if candidates:
            return True, max(candidates, key=lambda item: item[0])[1]
        return False, "waiting for bearish structure shift or resistance rejection"

    return False, "direction unclear"


def _stock_setup_grade(result: dict, daily_direction: str, setup_direction: str, location: str, setup_status: str) -> tuple:
    ev = result.get("trade_eval") or {}
    trade_direction = _stock_trade_direction(result)
    confirmation_started, confirmation_reason = _stock_confirmation(result, setup_direction, setup_status)
    loc = str(location or "").lower()
    in_zone = bool(result.get("in_ob") or result.get("near_ob"))
    no_trade_reasons = ev.get("no_trade_reasons") or []
    has_no_trade_reasons = bool(no_trade_reasons)
    rejection_confirmed_at_zone = bool(ev.get("rejection_confirmed") and in_zone)
    confirmation_strength = bool(
        ev.get("trigger_confirmed")
        or ev.get("a_plus_ready")
        or rejection_confirmed_at_zone
        or ev.get("b_plus_tradeable")
    )

    long_idea = trade_direction == "LONG"
    short_idea = trade_direction == "SHORT"
    trend_aligned = (long_idea and daily_direction == "Bullish") or (short_idea and daily_direction == "Bearish")
    trend_mixed = daily_direction not in {"Bullish", "Bearish"}
    location_conflict = (long_idea and loc == "premium") or (short_idea and loc == "discount")
    location_aligned = (
        (long_idea and (loc == "discount" or (in_zone and loc != "premium"))) or
        (short_idea and (loc == "premium" or (in_zone and loc != "discount")))
    )

    if trend_mixed:
        return "C", "C Setup — Caution: mixed daily trend", confirmation_started, confirmation_reason
    if not trend_aligned:
        return "C", "C Setup — Caution: counter-trend idea", confirmation_started, confirmation_reason
    if location_conflict:
        return "C", "C Setup — Caution: location conflicts with direction", confirmation_started, confirmation_reason
    if not location_aligned:
        return "C", "C Setup — Caution: weak or unclear location context", confirmation_started, confirmation_reason
    if confirmation_strength and not has_no_trade_reasons:
        return "A", f"A Setup — Confirmation Started: {confirmation_reason}", confirmation_started, confirmation_reason
    if confirmation_started:
        return "B", f"B Setup — Promising but needs review: {confirmation_reason}", confirmation_started, confirmation_reason
    return "B", "B Setup — Wait: trend and location aligned, confirmation not started", confirmation_started, confirmation_reason


def _enrich_stock_scout_fields(
    result: Optional[dict],
    daily_result: Optional[dict],
    h4_result: Optional[dict],
    daily_df: Optional[pd.DataFrame],
    h4_df: Optional[pd.DataFrame],
) -> Optional[dict]:
    if not result:
        return None

    daily_direction = _result_direction(daily_result, daily_df)
    h4_direction = _result_direction(h4_result, h4_df)
    setup_direction = _result_direction(result)
    setup_status, setup_reason = _stock_setup_status(result)
    entry_status, distance_atr, distance_pct = _stock_entry_status(result)
    ev = result.get("trade_eval") or {}
    location = _stock_location_label(ev)
    setup_grade, setup_grade_reason, confirmation_started, confirmation_reason = _stock_setup_grade(
        result,
        daily_direction,
        setup_direction,
        location,
        setup_status,
    )

    earnings_start = time.perf_counter()
    earnings = _earnings_for_ticker(result.get("ticker") or "", allow_fetch=False)
    earnings_ms = round((time.perf_counter() - earnings_start) * 1000, 1)

    option_plan_start = time.perf_counter()
    option_plan = build_option_plan({
        **result,
        "setupGrade": setup_grade,
        "entryStatus": entry_status,
        "confirmationStarted": confirmation_started,
        "confirmationReason": confirmation_reason,
    })
    option_plan_ms = round((time.perf_counter() - option_plan_start) * 1000, 1)
    best_contract = _unclean_contract("Live contract selection replaced by Option Plan", "option_plan")

    timing = result.setdefault("_scan_timing", {})
    timing.update({
        "earnings_ms": earnings_ms,
        "best_contract_ms": 0,
        "option_plan_ms": option_plan_ms,
    })

    result.update({
        "stockTrend": daily_direction,
        "trendDirection": daily_direction,
        "dailyTrendDirection": daily_direction,
        "h4TrendDirection": h4_direction,
        "sector": _sector_for_ticker(result.get("ticker") or ""),
        "marketRegime": result.get("market_regime"),
        "marketRegimeScore": result.get("market_regime_score"),
        "marketRegimeDetails": result.get("market_regime_details"),
        "dailyMarketRegime": (daily_result or {}).get("market_regime"),
        "dailyMarketRegimeScore": (daily_result or {}).get("market_regime_score"),
        "h4MarketRegime": (h4_result or {}).get("market_regime"),
        "h4MarketRegimeScore": (h4_result or {}).get("market_regime_score"),
        "setupTimeframeDirection": setup_direction,
        "stockPhase": _stock_phase(daily_direction, h4_direction),
        "phase": _stock_phase(daily_direction, h4_direction),
        "stockSetupStatus": setup_status,
        "setupStatus": setup_status,
        "stockSetupStatusReason": setup_reason,
        "setupStatusReason": setup_reason,
        "stockLocation": location,
        "nearestSupportDemand": result.get("ob_low") if result.get("ob_low") is not None else result.get("sl"),
        "nearestResistanceSupply": result.get("ob_high") if result.get("ob_high") is not None else result.get("bos_level"),
        "entryStatus": entry_status,
        "distanceFromEntryAtr": distance_atr,
        "distanceFromEntryPercent": distance_pct,
        "entryVisible": distance_atr is not None and distance_atr <= 0.5,
        "setupGrade": setup_grade,
        "setupGradeReason": setup_grade_reason,
        "confirmationStarted": confirmation_started,
        "confirmationReason": confirmation_reason,
        "earnings": earnings,
        "option_plan": option_plan,
        "best_contract": best_contract,
    })
    return result


def scan_all(
    watchlist: Optional[list] = None,
    max_workers: int = 12,
    discover: bool = False,
    max_symbols: Optional[int] = 200,
    trusted_options_symbols: Optional[set[str]] = None,
) -> tuple:
    scan_start = time.perf_counter()
    scan_started_at = _utc_now()
    universe_resolution_start = time.perf_counter()
    reset_provider_metrics()
    _cache_snapshot(reset=True)
    _scan_activity_started()
    try:
        logger.info("Stock Scanner Strategy: %s", STOCK_SCANNER_STRATEGY_VERSION)
        _ensure_background_refresh_started()
        if watchlist is None and discover:
            watchlist = get_finviz_watchlist()
        elif watchlist is None:
            watchlist = list(WATCHLIST)
        else:
            watchlist = list(dict.fromkeys([str(t).strip().upper() for t in watchlist if str(t).strip()]))
            if max_symbols is not None:
                try:
                    watchlist = watchlist[:max(0, int(max_symbols))]
                except (TypeError, ValueError):
                    watchlist = watchlist[:200]

        original_count = len(watchlist)
        universe_resolution_ms = round((time.perf_counter() - universe_resolution_start) * 1000, 1)
        trusted_options_symbols = {
            str(symbol or "").strip().upper()
            for symbol in (trusted_options_symbols or set())
            if str(symbol or "").strip()
        }

        # ── Step 1: batch-download daily OHLCV for cheap tradeability filtering ───
        price_stage_start = time.perf_counter()
        daily_fetch_start = time.perf_counter()
        daily_data  = _batch_download(watchlist, period="1y",  interval="1d")
        daily_fetch_ms = round((time.perf_counter() - daily_fetch_start) * 1000, 1)
        prefilter_start = time.perf_counter()
        filtered_watchlist, skipped_symbols = _prefilter_stock_universe(
            watchlist,
            daily_data,
            trusted_options_symbols=trusted_options_symbols,
        )
        prefilter_ms = round((time.perf_counter() - prefilter_start) * 1000, 1)
        skip_counts = {}
        for item in skipped_symbols:
            reason = item.get("reason") or "unknown"
            skip_counts[reason] = skip_counts.get(reason, 0) + 1
        if skipped_symbols:
            examples = ", ".join(f"{item['ticker']}:{item['reason']}" for item in skipped_symbols[:20])
            logger.info(
                "[tradeability filter] skipped=%s/%s reasons=%s examples=%s",
                len(skipped_symbols),
                original_count,
                skip_counts,
                examples,
        )

        if filtered_watchlist:
            weekly_fetch_start = time.perf_counter()
            weekly_data = _batch_download(filtered_watchlist, period="2y",  interval="1wk")
            weekly_fetch_ms = round((time.perf_counter() - weekly_fetch_start) * 1000, 1)
            h4_fetch_start = time.perf_counter()
            h4_data     = _batch_download(filtered_watchlist, period="60d", interval="4h")
            h4_fetch_ms = round((time.perf_counter() - h4_fetch_start) * 1000, 1)
        else:
            weekly_data = {}
            h4_data = {}
            weekly_fetch_ms = 0.0
            h4_fetch_ms = 0.0
        price_stage_ms = round((time.perf_counter() - price_stage_start) * 1000, 1)

        # ── Step 2: parallel-process each ticker (pure CPU/logic, no I/O) ─────────
        rows, near_miss = [], []
        processing_failures = []
        process_stage_start = time.perf_counter()

        def _process(ticker: str):
            return scan_ticker(
                ticker,
                _daily_df=daily_data.get(ticker),
                _weekly_df=weekly_data.get(ticker),
                _h4_df=h4_data.get(ticker, pd.DataFrame()),
            )

        with _scan_symbol_stdout_context():
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_process, t): t for t in filtered_watchlist}
                for future in as_completed(futures):
                    ticker = futures[future]
                    try:
                        r = future.result()
                    except Exception as exc:
                        processing_failures.append({
                            "ticker": ticker,
                            "reason": "internal exception",
                            "error_type": type(exc).__name__,
                        })
                        logger.exception("[scan] symbol processing failed ticker=%s", ticker)
                        continue
                    if r is None:
                        processing_failures.append({
                            "ticker": ticker,
                            "reason": "symbol returned no setup",
                        })
                        continue
                    if r.get("setup_status") == "QUALIFIED":
                        rows.append(r)
                    else:
                        near_miss.append(r)

        rows.sort(key=lambda x: x.get("quality", {}).get("score", 0), reverse=True)
        near_miss.sort(key=lambda x: x.get("quality", {}).get("score", 0), reverse=True)
        process_stage_ms = round((time.perf_counter() - process_stage_start) * 1000, 1)
        all_results = [*rows, *near_miss]
        quote_stage_start = time.perf_counter()
        _attach_current_quotes(all_results)
        quote_stage_ms = round((time.perf_counter() - quote_stage_start) * 1000, 1)
        earnings_ms = round(sum((r.get("_scan_timing") or {}).get("earnings_ms", 0) for r in all_results), 1)
        best_contract_ms = round(sum((r.get("_scan_timing") or {}).get("best_contract_ms", 0) for r in all_results), 1)
        option_plan_ms = round(sum((r.get("_scan_timing") or {}).get("option_plan_ms", 0) for r in all_results), 1)
        price_trend_sum_ms = round(sum((r.get("_scan_timing") or {}).get("price_trend_ms", 0) for r in all_results), 1)
        symbol_durations = [
            float((r.get("_scan_timing") or {}).get("total_ticker_ms", 0) or 0)
            for r in all_results
            if isinstance(r.get("_scan_timing"), dict)
        ]
        evaluated_contracts = sum(1 for r in all_results if (r.get("best_contract") or {}).get("source") == "option_chain")
        option_plans_generated = sum(1 for r in all_results if (r.get("option_plan") or {}).get("available") is True)
        total_ms = round((time.perf_counter() - scan_start) * 1000, 1)
        scan_completed_at = _utc_now()
        cache_stats = _cache_snapshot()
        provider_metrics = provider_metrics_snapshot()
        attempted = len(filtered_watchlist)
        processed = len(all_results)
        raw_no_setup_count = sum(
            1 for item in processing_failures
            if isinstance(item, dict) and str(item.get("reason") or "") == "symbol returned no setup"
        )
        internal_failure_count = sum(
            1 for item in processing_failures
            if isinstance(item, dict) and str(item.get("reason") or "") != "symbol returned no setup"
        )
        partial_reasons = _scan_partial_reasons(
            attempted=attempted,
            processed=processed,
            tradeability_skipped=len(skipped_symbols),
            processing_failures=processing_failures,
            provider_metrics=provider_metrics,
        )
        provider_failed_count = int(provider_metrics.get("alpaca_bar_symbols_failed", 0) or 0)
        operational_failed_count = sum(int(reason.get("count") or 0) for reason in partial_reasons)
        no_setup_count = min(
            raw_no_setup_count,
            max(0, original_count - processed - len(skipped_symbols) - operational_failed_count),
        )
        not_evaluated_count = sum(
            int(reason.get("count") or 0)
            for reason in partial_reasons
            if reason.get("reason") == "symbol_not_evaluated"
        )
        terminally_evaluated = max(
            0,
            processed + no_setup_count + len(skipped_symbols),
        )
        evaluation_coverage = round(terminally_evaluated / original_count, 4) if original_count > 0 else None
        result_yield = round(processed / original_count, 4) if original_count > 0 else None
        symbols_per_second = round(processed / (total_ms / 1000), 2) if total_ms > 0 else None
        cache_hit_rate = _cache_hit_rate(cache_stats)
        candle_cache_requests = sum(
            int(cache_stats.get(f"prices_request_{interval}_count", 0) or 0)
            for interval in ("1d", "1wk", "4h")
        )
        candle_cache_symbols_requested = sum(
            int(cache_stats.get(f"prices_request_{interval}_symbols", 0) or 0)
            for interval in ("1d", "1wk", "4h")
        )
        market_data_engine = {
            "requests": candle_cache_requests,
            "symbols_requested": candle_cache_symbols_requested,
            "hits": int(cache_stats.get("prices_hit", 0) or 0),
            "misses": int(cache_stats.get("prices_miss", 0) or 0),
            "stale": int(cache_stats.get("prices_stale", 0) or 0),
            "hit_rate": cache_hit_rate,
            "average_cache_age_seconds": _cache_average(cache_stats, "prices_age_seconds"),
            "oldest_cache_age_seconds": round(float(cache_stats.get("prices_oldest_age_seconds", 0) or 0), 1),
            "duplicate_requests_eliminated": int(cache_stats.get("prices_duplicate_symbols_eliminated", 0) or 0),
            "average_fetch_time_ms": _cache_average(cache_stats, "api_prices_duration_ms"),
            "incremental_updates_used": 0,
            "http_retries": 0,
            "rate_limits": 0,
            "provider_errors": int(cache_stats.get("api_prices_error", 0) or 0),
            "earnings_requests_deferred_during_scan": int(cache_stats.get("earnings_deferred_scan", 0) or 0),
            "request_breakdown": {
                "daily": {
                    "requests": int(cache_stats.get("prices_request_1d_count", 0) or 0),
                    "symbols": int(cache_stats.get("prices_request_1d_symbols", 0) or 0),
                    "duration_ms": daily_fetch_ms,
                },
                "weekly": {
                    "requests": int(cache_stats.get("prices_request_1wk_count", 0) or 0),
                    "symbols": int(cache_stats.get("prices_request_1wk_symbols", 0) or 0),
                    "duration_ms": weekly_fetch_ms,
                },
                "h4": {
                    "requests": int(cache_stats.get("prices_request_4h_count", 0) or 0),
                    "symbols": int(cache_stats.get("prices_request_4h_symbols", 0) or 0),
                    "duration_ms": h4_fetch_ms,
                },
            },
        }
        slow_symbols = sorted(
            [
                {
                    "ticker": r.get("ticker"),
                    "duration_ms": round(float((r.get("_scan_timing") or {}).get("total_ticker_ms", 0) or 0), 1),
                }
                for r in all_results
                if isinstance(r, dict)
            ],
            key=lambda item: item.get("duration_ms") or 0,
            reverse=True,
        )[:10]
        slow_contracts = [
            f"{r.get('ticker')}:{(r.get('_scan_timing') or {}).get('best_contract_ms', 0):.0f}ms"
            for r in all_results
            if (r.get("_scan_timing") or {}).get("best_contract_ms", 0) >= 750
        ][:8]
        logger.info(
            "[scan timing] total=%.1fms price_trend=%.1fms processing=%.1fms earnings_sum=%.1fms best_contract_sum=%.1fms contract_evaluated=%s/%s option_plans=%s/%s universe=%s/%s skipped=%s skip_reasons=%s cache=%s slow_contracts=%s",
            total_ms,
            price_stage_ms,
            process_stage_ms,
            earnings_ms,
            best_contract_ms,
            evaluated_contracts,
            len(all_results),
            option_plans_generated,
            len(all_results),
            len(filtered_watchlist),
            original_count,
            len(skipped_symbols),
            skip_counts,
            cache_stats,
            ", ".join(slow_contracts) if slow_contracts else "none",
        )
        logger.info(
            "coverage.scan.performance total_ms=%s universe_resolution_ms=%s market_data_fetch_ms=%s strategy_evaluation_ms=%s option_plan_ms=%s quote_ms=%s symbols_per_second=%s cache_hit_rate=%s partial=%s partial_reasons=%s peak_workers=%s",
            total_ms,
            universe_resolution_ms,
            price_stage_ms,
            process_stage_ms,
            option_plan_ms,
            quote_stage_ms,
            symbols_per_second,
            cache_hit_rate,
            bool(partial_reasons),
            partial_reasons,
            max_workers,
        )
        scan_meta = {
            "configured_universe_count": original_count,
            "symbols_attempted": attempted,
            "symbols_successfully_processed": processed,
            "symbols_terminally_evaluated": terminally_evaluated,
            "symbols_with_setup": processed,
            "symbols_without_setup": no_setup_count,
            "symbols_intentionally_rejected": len(skipped_symbols),
            "symbols_operationally_failed": operational_failed_count,
            "symbols_not_evaluated": not_evaluated_count,
            "evaluation_coverage": evaluation_coverage,
            "evaluation_coverage_percent": round(evaluation_coverage * 100, 2) if evaluation_coverage is not None else None,
            "result_yield": result_yield,
            "result_yield_percent": round(result_yield * 100, 2) if result_yield is not None else None,
            "symbols_omitted_or_rejected": max(0, original_count - processed),
            "symbols_failed": operational_failed_count,
            "symbols_skipped": len(skipped_symbols),
            "tradeability_skipped": len(skipped_symbols),
            "tradeability_skip_reasons": skip_counts,
            "no_setup_or_failed_count": max(0, attempted - processed),
            "processing_no_setup_count": no_setup_count,
            "processing_internal_failure_count": internal_failure_count,
            "provider_symbol_failure_count": provider_failed_count,
            "processing_failures": processing_failures[:50],
            "contract_evaluated": evaluated_contracts,
            "contract_evaluation_pool": len(all_results),
            "option_plans_generated": option_plans_generated,
            "scan_started_at": _format_utc_timestamp(scan_started_at),
            "scan_completed_at": _format_utc_timestamp(scan_completed_at),
            "scan_duration_ms": total_ms,
            "partial_result": bool(partial_reasons),
            "partial_result_reasons": partial_reasons,
            "partial_result_reason": "; ".join(
                f"{item.get('stage')}:{item.get('reason')}={item.get('count')}"
                for item in partial_reasons
            ) or None,
            "option_eligibility_from_discovery": len([symbol for symbol in filtered_watchlist if symbol in trusted_options_symbols]),
            "provider_metrics": provider_metrics,
            "cache_stats": cache_stats,
            "performance": {
                "universe_resolution_ms": universe_resolution_ms,
                "market_data_fetch_ms": price_stage_ms,
                "market_data_cache_read_ms": None,
                "indicator_calculation_ms": price_trend_sum_ms,
                "strategy_evaluation_ms": process_stage_ms,
                "option_plan_ms": option_plan_ms,
                "serialization_ms": None,
                "quote_enrichment_ms": quote_stage_ms,
                "total_scan_duration_ms": total_ms,
                "daily_fetch_ms": daily_fetch_ms,
                "weekly_fetch_ms": weekly_fetch_ms,
                "h4_fetch_ms": h4_fetch_ms,
                "tradeability_prefilter_ms": prefilter_ms,
                "symbols_per_second": symbols_per_second,
                "provider_request_count": int(provider_metrics.get("alpaca_bar_requests", 0) or 0) + int(cache_stats.get("api_prices_call", 0) or 0),
                "provider_success_count": int(provider_metrics.get("alpaca_bar_symbols_succeeded", 0) or 0),
                "provider_failure_count": int(provider_metrics.get("alpaca_bar_symbols_failed", 0) or 0),
                "provider_timeout_count": 0,
                "provider_retry_count": 0,
                "cache_hit_count": int(cache_stats.get("prices_hit", 0) or 0),
                "cache_miss_count": int(cache_stats.get("prices_miss", 0) or 0),
                "cache_hit_rate": cache_hit_rate,
                "candle_cache_requests": candle_cache_requests,
                "candle_cache_hits": int(cache_stats.get("prices_hit", 0) or 0),
                "candle_cache_misses": int(cache_stats.get("prices_miss", 0) or 0),
                "candle_cache_stale": int(cache_stats.get("prices_stale", 0) or 0),
                "candle_cache_hit_rate": cache_hit_rate,
                "candle_cache_average_age_seconds": market_data_engine["average_cache_age_seconds"],
                "candle_cache_oldest_age_seconds": market_data_engine["oldest_cache_age_seconds"],
                "duplicate_requests_eliminated": market_data_engine["duplicate_requests_eliminated"],
                "average_fetch_time_ms": market_data_engine["average_fetch_time_ms"],
                "incremental_updates_used": 0,
                "http_retries": 0,
                "rate_limits": 0,
                "provider_errors": market_data_engine["provider_errors"],
                "earnings_requests_deferred_during_scan": market_data_engine["earnings_requests_deferred_during_scan"],
                "market_data_engine": market_data_engine,
                "peak_worker_count": max_workers,
                "memory_rss_mb": _process_memory_mb(),
                "median_symbol_duration_ms": _percentile(symbol_durations, 0.5),
                "p95_symbol_duration_ms": _percentile(symbol_durations, 0.95),
                "slowest_symbols": slow_symbols,
                "failure_breakdown_by_stage": partial_reasons,
            },
        }
        return rows, near_miss, scan_meta
    finally:
        _scan_activity_finished()


def scan_cached(
    watchlist: Optional[list] = None,
    *,
    force_refresh: bool = False,
    discover: bool = False,
    universe: str = "default",
    max_symbols: Optional[int] = 200,
    coverage_context: Optional[dict] = None,
    trusted_options_symbols: Optional[set[str]] = None,
) -> dict:
    key = _analysis_cache_key(watchlist, discover=discover, universe=universe)
    with _cache_lock:
        cached = _analysis_cache.get(key)

    if cached:
        generated_at = cached.get("generated_at")
        age_seconds = _age_seconds(generated_at)
        should_refresh = bool(force_refresh) or age_seconds is None or age_seconds > ANALYSIS_CACHE_STALE_SECONDS
        if should_refresh:
            _submit_analysis_refresh(
                key,
                watchlist,
                reason="manual" if force_refresh else "cache",
                discover=discover,
                max_symbols=max_symbols,
                coverage_context=coverage_context,
                trusted_options_symbols=trusted_options_symbols,
            )
        return {
            "rows": _hydrate_scan_rows_from_cache(list(cached.get("rows", []))),
            "near_miss": _hydrate_scan_rows_from_cache(list(cached.get("near_miss", []))),
            "meta": {
                **_analysis_cache_meta(key, cached, True),
                "refresh_requested": bool(force_refresh),
            },
        }

    if not cached:
        _submit_analysis_refresh(
            key,
            watchlist,
            reason="manual" if force_refresh else "cache",
            discover=discover,
            max_symbols=max_symbols,
            coverage_context=coverage_context,
            trusted_options_symbols=trusted_options_symbols,
        )
        return {
            "rows": [],
            "near_miss": [],
            "meta": analysis_cache_status(watchlist, discover=discover, universe=universe),
        }


def scan_ticker(
    ticker: str,
    _daily_df: Optional[pd.DataFrame] = None,
    _weekly_df: Optional[pd.DataFrame] = None,
    _h4_df: Optional[pd.DataFrame] = None,
) -> Optional[dict]:
    ticker_start = time.perf_counter()
    ticker = ticker.upper()
    price_trend_start = time.perf_counter()
    daily_result = analyze_ticker(
        ticker,
        _daily_df=_daily_df,
        _weekly_df=_weekly_df,
        timeframe="1D",
    )

    if _h4_df is not None:
        h4_source = _h4_df
    else:
        h4_source = _batch_download([ticker], period="60d", interval="4h").get(ticker, pd.DataFrame())

    h4_result = analyze_ticker(
        ticker,
        _daily_df=h4_source,
        _weekly_df=_weekly_df,
        timeframe="4H",
    )
    price_trend_ms = round((time.perf_counter() - price_trend_start) * 1000, 1)

    best = _best_timeframe_result(daily_result, h4_result)
    enriched = _enrich_stock_scout_fields(best, daily_result, h4_result, _daily_df, h4_source)
    if enriched is not None:
        timing = enriched.setdefault("_scan_timing", {})
        timing["price_trend_ms"] = price_trend_ms
        timing["total_ticker_ms"] = round((time.perf_counter() - ticker_start) * 1000, 1)
    return enriched
