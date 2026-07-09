# v2 — BOS + Order Block strategy (replaces EMA pullback)
import yfinance as yf
import pandas as pd
import numpy as np
import logging
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

WATCHLIST = [
    # ── Airlines ──────────────────────────────────────────────────────────────
    "DAL", "UAL", "AAL", "JBLU",
    # ── Cruise ────────────────────────────────────────────────────────────────
    "CCL", "RCL", "NCLH",
    # ── Tech / Mega-cap ───────────────────────────────────────────────────────
    "NVDA", "AMD", "META", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN",
    # ── Semiconductors ────────────────────────────────────────────────────────
    "MU", "INTC", "QCOM", "AVGO", "TSM", "ARM",
    # ── Energy Majors ─────────────────────────────────────────────────────────
    "XOM", "CVX", "OXY",
    # ── Energy Mid-cap / Services ─────────────────────────────────────────────
    "DVN", "MRO", "HAL", "SLB", "FANG",
    # ── Big Banks ─────────────────────────────────────────────────────────────
    "JPM", "BAC", "GS", "MS", "WFC",
    # ── Payments / Fintech ────────────────────────────────────────────────────
    "V", "MA", "AXP", "PYPL", "SQ",
    # ── Retail / Consumer Discretionary ──────────────────────────────────────
    "WMT", "TGT", "COST", "HD", "LOW", "MCD", "SBUX", "NKE",
    # ── Consumer Staples ──────────────────────────────────────────────────────
    "PG", "KO", "PEP", "CL", "MO",
    # ── Healthcare / Pharma ───────────────────────────────────────────────────
    "UNH", "JNJ", "ABBV", "LLY", "PFE", "MRK", "CVS",
    # ── Biotech ───────────────────────────────────────────────────────────────
    "GILD", "REGN", "MRNA", "BIIB",
    # ── Utilities ─────────────────────────────────────────────────────────────
    "NEE", "DUK", "SO", "XEL",
    # ── Industrials / Defense ─────────────────────────────────────────────────
    "CAT", "DE", "HON", "BA", "GE", "LMT", "RTX",
    # ── Materials / Commodities ───────────────────────────────────────────────
    "FCX", "NEM", "AA", "CLF",
    # ── Broad ETFs ────────────────────────────────────────────────────────────
    "SPY", "QQQ", "IWM", "DIA",
    # ── Sector ETFs ───────────────────────────────────────────────────────────
    "XLF", "XLE", "XLV", "XLU", "XLK", "XLI", "XLB",
    # ── Commodity ETFs ────────────────────────────────────────────────────────
    "GLD", "SLV", "USO", "UNG",
]

TRENDING_UNIVERSE = [
    "IWM", "XLF", "XLE", "SOFI", "PLTR", "HOOD", "AFRM", "RIVN", "AAL",
    "DAL", "CCL", "AMD", "NVDA", "TSLA", "META", "AMZN", "MSFT", "AAPL",
]

NO_EARNINGS_SYMBOLS = {
    "SPY", "QQQ", "IWM", "DIA",
    "GLD", "SLV", "USO", "UNG",
    "XLF", "XLE", "XLV", "XLU", "XLK", "XLI", "XLB",
}

STOCK_UNIVERSE_FILTER = {
    "enabled": True,
    "min_price": 10.0,
    "min_avg_volume": 1_000_000,
    "avg_volume_lookback": 30,
    "min_option_expirations": 1,
    "exclude_non_major_etfs": True,
    "allowlist": {
        "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV",
        "XLE", "XLF", "XLK", "XLV", "XLU", "XLI", "XLB",
        "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META", "AMZN", "GOOGL",
    },
    "blocklist": set(),
}

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
_best_contract_cache = {}
BEST_CONTRACT_CACHE_TTL = timedelta(minutes=8)
_option_chain_fetch_semaphore = threading.BoundedSemaphore(4)
_earnings_cache = {}
EARNINGS_CACHE_TTL = timedelta(hours=12)
EARNINGS_UNAVAILABLE_CACHE_TTL = timedelta(hours=24)
_cache_lock = threading.RLock()
_cache_stats = Counter()
_background_executor = ThreadPoolExecutor(max_workers=3)
_background_jobs = set()
_background_jobs_lock = threading.Lock()
_background_refresh_started = False
_background_last_refresh = {}
_active_scan_count = 0
_analysis_cache = {}
ANALYSIS_CACHE_STALE_SECONDS = 180


def _cache_record(name: str, outcome: str) -> None:
    with _cache_lock:
        _cache_stats[f"{name}_{outcome}"] += 1
    logger.debug("[cache] %s %s", name, outcome)


def _cache_snapshot(reset: bool = True) -> dict:
    with _cache_lock:
        snapshot = dict(_cache_stats)
        if reset:
            _cache_stats.clear()
    return snapshot


def _submit_background_job(key: tuple, fn, *args, **kwargs) -> None:
    with _background_jobs_lock:
        if key in _background_jobs:
            return
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

    _background_executor.submit(_run)


def _periodic_refresh_due(key: str, ttl_seconds: int) -> bool:
    now = time.monotonic()
    with _background_jobs_lock:
        last = _background_last_refresh.get(key, 0)
        if now - last < ttl_seconds:
            return False
        _background_last_refresh[key] = now
        return True


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


def _analysis_cache_key(watchlist: Optional[list]) -> tuple:
    if watchlist is None:
        return ("default",)
    symbols = tuple(sorted(dict.fromkeys([str(t).strip().upper() for t in watchlist if str(t).strip()])))
    return ("custom", symbols)


def _analysis_cache_meta(key: tuple, cached: dict, refreshing: bool) -> dict:
    generated_at = cached.get("generated_at")
    age_seconds = None
    if generated_at:
        age_seconds = max(0, (datetime.utcnow() - generated_at).total_seconds())
    return {
        "cache": "hit",
        "generated_at": generated_at.isoformat() + "Z" if generated_at else None,
        "age_seconds": round(age_seconds, 1) if age_seconds is not None else None,
        "stale": age_seconds is not None and age_seconds > ANALYSIS_CACHE_STALE_SECONDS,
        "refreshing": refreshing,
        "cache_key": "default" if key == ("default",) else "custom",
    }


def _store_analysis_cache(key: tuple, rows: list, near_miss: list) -> dict:
    cached = {
        "rows": rows,
        "near_miss": near_miss,
        "generated_at": datetime.utcnow(),
    }
    with _cache_lock:
        _analysis_cache[key] = cached
    return cached


def _refresh_analysis_cache(key: tuple, watchlist: Optional[list]) -> None:
    rows, near_miss = scan_all(watchlist)
    _store_analysis_cache(key, rows, near_miss)


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
        df.columns = df.columns.get_level_values(0)
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
            age = now - cached.get("fetched_at", now)
            if age < OPTION_CHAIN_CACHE_TTL:
                _cache_record("option_chains", "hit")
                return expirations
            if expirations:
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
        return True, list(cached.get("expirations", []))


def _fetch_option_expirations(ticker: str) -> list:
    ticker = str(ticker or "").upper()
    now = datetime.utcnow()

    expirations = []
    try:
        _cache_record("api_option_expirations", "call")
        expirations = list(yf.Ticker(ticker).options or [])
    except Exception as e:
        logger.warning(f"[options] expiration fetch failed for {ticker}: {e}")

    with _cache_lock:
        cached = _option_chain_cache.setdefault(ticker, {"fetched_at": now, "expirations": [], "chains": {}})
        cached["fetched_at"] = now
        cached["expirations"] = expirations
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


def _stock_universe_skip_reason(ticker: str, daily_df: Optional[pd.DataFrame]) -> Optional[str]:
    config = STOCK_UNIVERSE_FILTER
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
    min_price = float(config.get("min_price") or 0)
    if price < min_price and ticker not in allowlist:
        return "low price"

    avg_volume = _average_volume(daily_df, int(config.get("avg_volume_lookback") or 30))
    min_avg_volume = float(config.get("min_avg_volume") or 0)
    if (avg_volume is None or avg_volume < min_avg_volume) and ticker not in allowlist:
        return "low liquidity"

    min_expirations = int(config.get("min_option_expirations") or 0)
    if min_expirations > 0:
        known, expirations = _cached_option_expirations_for_ticker(ticker)
        if known:
            if not expirations:
                return "no options"
            if len(expirations) < min_expirations:
                return "thin options chain"
        else:
            _submit_background_job(("option_expirations_prefilter", ticker), _refresh_option_expirations, ticker)

    return None


def _prefilter_stock_universe(watchlist: list, daily_data: dict) -> tuple[list, list]:
    accepted = []
    skipped = []
    for ticker in watchlist:
        symbol = str(ticker or "").strip().upper()
        if not symbol:
            continue
        reason = _stock_universe_skip_reason(symbol, daily_data.get(symbol))
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
                _submit_background_job(("earnings", ticker), _refresh_earnings, ticker)
                return dict(cached_data)
        _cache_record("earnings", "miss")

    if not allow_fetch:
        _submit_background_job(("earnings", ticker), _refresh_earnings, ticker)
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
    diagnostics = best.get("diagnostics", {})
    spread_pct = diagnostics.get("spread_pct")
    distance_pct = diagnostics.get("distance_pct")
    if distance_pct is not None and distance_pct > 10:
        result = {
            **_unclean_contract("Closest contract strike is too far from the ideal strike", "option_chain"),
            "best_score": best["score"],
            "diagnostics": diagnostics,
        }
        return _store_best_contract(cache_key, result, now)
    if spread_pct is None or spread_pct > 35:
        result = {
            **_unclean_contract("Best contract spread is too wide or unavailable", "option_chain"),
            "best_score": best["score"],
            "diagnostics": diagnostics,
        }
        return _store_best_contract(cache_key, result, now)
    if (best.get("open_interest") or 0) < 25 and (best.get("volume") or 0) < 1:
        result = {
            **_unclean_contract("Best contract liquidity is too thin", "option_chain"),
            "best_score": best["score"],
            "diagnostics": diagnostics,
        }
        return _store_best_contract(cache_key, result, now)

    execution = "Excellent" if best["score"] >= 75 else "Fair" if best["score"] >= 55 else "No Clean Contract"
    if execution == "No Clean Contract":
        result = {
            **_unclean_contract("Spread, liquidity, DTE, or strike distance did not meet minimum quality", "option_chain"),
            "reason": "Spread, liquidity, DTE, or strike distance did not meet minimum quality",
            "best_score": best["score"],
            "diagnostics": best.get("diagnostics", {}),
        }
        return _store_best_contract(cache_key, result, now)

    best.update({
        "available": True,
        "execution": execution,
        "source": "option_chain",
        "cache": "miss",
    })
    return _store_best_contract(cache_key, best, now)


def _has_valid_trade_plan(result: dict) -> bool:
    direction = str(result.get("direction") or "").upper()
    if direction not in {"LONG", "SHORT"}:
        return False
    return all(_safe_float(result.get(key)) is not None for key in ("entry", "sl", "tp1"))


def _should_enrich_best_contract(result: dict, setup_grade: str, entry_status: str) -> bool:
    trade_stage = str((result.get("trade_eval") or {}).get("trade_stage") or "").upper()
    if trade_stage in {"A+ READY", "B+ TRADEABLE"}:
        return _has_valid_trade_plan(result)
    if setup_grade not in {"A", "B"}:
        return False
    if entry_status not in {"Tradeable", "Near Entry"}:
        return False
    return _has_valid_trade_plan(result)


# ── Price Action Functions ────────────────────────────────────────────────────

def _find_swings(df: pd.DataFrame, margin: int = 4) -> list:
    highs = df["High"].values
    lows  = df["Low"].values
    swings = []
    for i in range(margin, len(df) - margin):
        window_h = highs[i - margin : i + margin + 1]
        window_l = lows[i  - margin : i + margin + 1]
        if highs[i] == window_h.max():
            swings.append({"index": i, "price": float(highs[i]), "type": "high"})
        elif lows[i] == window_l.min():
            swings.append({"index": i, "price": float(lows[i]),  "type": "low"})
    return swings


def _get_trend(swings: list) -> str:
    highs = [s for s in swings if s["type"] == "high"]
    lows  = [s for s in swings if s["type"] == "low"]
    if len(highs) < 2 or len(lows) < 2:
        return "NEUTRAL"
    hh = highs[-1]["price"] > highs[-2]["price"]
    hl = lows[-1]["price"]  > lows[-2]["price"]
    lh = highs[-1]["price"] < highs[-2]["price"]
    ll = lows[-1]["price"]  < lows[-2]["price"]
    if hh and hl:
        return "LONG"
    if lh and ll:
        return "SHORT"
    return "NEUTRAL"


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

    highs = [s["price"] for s in swings if s["type"] == "high"]
    lows  = [s["price"] for s in swings if s["type"] == "low"]

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
        if highs[-1] < highs[-2]:
            bearish_score += 2
            reasons.append(f"LH: {highs[-1]:.2f} < {highs[-2]:.2f} [bearish +2]")
        else:
            bullish_score += 2
            reasons.append(f"HH: {highs[-1]:.2f} >= {highs[-2]:.2f} [bullish +2]")

    # ── Recent swing low comparison (weight 2) ────────────────────────────────
    if len(lows) >= 2:
        if lows[-1] < lows[-2]:
            bearish_score += 2
            reasons.append(f"LL: {lows[-1]:.2f} < {lows[-2]:.2f} [bearish +2]")
        else:
            bullish_score += 2
            reasons.append(f"HL: {lows[-1]:.2f} >= {lows[-2]:.2f} [bullish +2]")

    # ── 3-swing sequence (weight 2) ───────────────────────────────────────────
    if (len(highs) >= 3 and highs[-3] > highs[-2] > highs[-1]
            and len(lows) >= 2 and lows[-2] > lows[-1]):
        bearish_score += 2
        reasons.append("3-swing LH/LL sequence confirmed [bearish +2]")
    elif (len(lows) >= 3 and lows[-3] < lows[-2] < lows[-1]
            and len(highs) >= 2 and highs[-2] < highs[-1]):
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
        "label": "Blocked: RR < 1:2" if blocked else "Clear path to target",
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
        no_trade_reasons.append("RR < 1:2")
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

def _download_price_batch_raw(tickers: list, period: str, interval: str) -> dict:
    """
    Download OHLCV data for multiple tickers in a single yfinance call.
    Returns {ticker: DataFrame}.  Falls back gracefully on any per-ticker error.
    """
    if not tickers:
        return {}
    try:
        _cache_record("api_prices", "call")
        raw = yf.download(
            tickers, period=period, interval=interval,
            progress=False, auto_adjust=True, group_by="ticker",
        )
    except Exception as e:
        print(f"[batch_download] error: {e}")
        return {}

    result: dict = {}
    single = len(tickers) == 1

    for t in tickers:
        try:
            if single:
                df = raw.copy()
                if isinstance(df.columns, pd.MultiIndex):
                    # (PriceType, Ticker) layout — drop the ticker level
                    df.columns = df.columns.get_level_values(0)
            else:
                df = raw[t].copy()  # group_by='ticker' gives (Ticker, PriceType)
            df = df.dropna(how="all")
            if len(df) >= 10:
                result[t] = df
        except Exception:
            pass

    return result


def _price_cache_key(ticker: str, period: str, interval: str) -> tuple:
    return (str(ticker or "").upper(), period, interval)


def _refresh_price_cache(tickers: list, period: str, interval: str) -> None:
    symbols = list(dict.fromkeys([str(t).strip().upper() for t in tickers if str(t).strip()]))
    if not symbols:
        return
    fetched = _download_price_batch_raw(symbols, period, interval)
    now = datetime.utcnow()
    with _cache_lock:
        for ticker, df in fetched.items():
            _price_cache[_price_cache_key(ticker, period, interval)] = {
                "fetched_at": now,
                "data": df,
            }
    logger.info("[cache] refreshed prices period=%s interval=%s tickers=%s/%s", period, interval, len(fetched), len(symbols))


def _batch_download(tickers: list, period: str, interval: str) -> dict:
    if not tickers:
        return {}
    symbols = list(dict.fromkeys([str(t).strip().upper() for t in tickers if str(t).strip()]))
    now = datetime.utcnow()
    result = {}
    missing = []
    stale = []

    with _cache_lock:
        for ticker in symbols:
            key = _price_cache_key(ticker, period, interval)
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
            if age <= PRICE_CACHE_TTL:
                _cache_record("prices", "hit")
            else:
                stale.append(ticker)
                _cache_record("prices", "stale")

    if stale:
        _submit_background_job(("prices", period, interval, tuple(stale)), _refresh_price_cache, stale, period, interval)

    if missing:
        fetched = _download_price_batch_raw(missing, period, interval)
        fetched_at = datetime.utcnow()
        with _cache_lock:
            for ticker, df in fetched.items():
                _price_cache[_price_cache_key(ticker, period, interval)] = {
                    "fetched_at": fetched_at,
                    "data": df,
                }
                result[ticker] = df.copy()

    return result


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
            if _periodic_refresh_due("prices_1d", 180):
                _submit_background_job(("prices_periodic", "1y", "1d"), _refresh_price_cache, symbols, "1y", "1d")
            if _periodic_refresh_due("prices_1wk", 300):
                _submit_background_job(("prices_periodic", "2y", "1wk"), _refresh_price_cache, symbols, "2y", "1wk")
            if _periodic_refresh_due("prices_4h", 300):
                _submit_background_job(("prices_periodic", "60d", "4h"), _refresh_price_cache, symbols, "60d", "4h")

            if _periodic_refresh_due("option_expirations", 600):
                for ticker in list(STOCK_UNIVERSE_FILTER.get("allowlist", []))[:24]:
                    _submit_background_job(("option_expirations_periodic", ticker), _refresh_option_expirations, ticker)
                _refresh_cached_best_contracts()

            if _periodic_refresh_due("earnings", 6 * 60 * 60):
                for ticker in [s for s in symbols if s not in NO_EARNINGS_SYMBOLS][:40]:
                    _submit_background_job(("earnings_periodic", ticker), _refresh_earnings, ticker)
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
            h4_df = yf.download(ticker, period="60d", interval="4h", progress=False, auto_adjust=True)
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
        print(f"[trends] {ticker} error: {e}")
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
            raw = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
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
        swings        = _find_swings(df)
        trend         = _get_trend(swings)
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
                w_raw = yf.download(ticker, period="2y", interval="1wk", progress=False, auto_adjust=True)
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
            htf_swings = _find_swings(htf_df, margin=5)
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
        raw = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
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
        swings = _find_swings(df)
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
        w_raw  = yf.download(ticker, period="2y", interval="1wk", progress=False, auto_adjust=True)
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
        htf_swings = _find_swings(htf_df, margin=5)
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

    bullish_flow = setup_direction == "Bullish" or "bullish" in structure_label
    bearish_flow = setup_direction == "Bearish" or "bearish" in structure_label

    if ev.get("trigger_confirmed"):
        return True, "trigger confirmed after pullback"

    if trade_direction == "LONG":
        if result.get("bos_confirmed") and bullish_flow:
            return True, "bullish structure shift after pullback"
        if ev.get("rejection_confirmed") and (result.get("in_ob") or result.get("near_ob")):
            return True, "strong bullish reaction from demand/support"
        if displacement == "STRONG" and bullish_flow:
            return True, "strong bullish reaction candle from support"
        if result.get("price") and result.get("ema20") and float(result["price"]) > float(result["ema20"]) and bullish_flow:
            return True, "reclaimed EMA20 with bullish short-term flow"
        if "Early Confirmation" in setup_status or "Strong Confirmation" in setup_status or "Trend Resumption" in setup_status:
            return True, "bullish confirmation started"
        return False, "waiting for bullish structure shift or support reaction"

    if trade_direction == "SHORT":
        if result.get("bos_confirmed") and bearish_flow:
            return True, "bearish structure shift after pullback"
        if ev.get("rejection_confirmed") and (result.get("in_ob") or result.get("near_ob")):
            return True, "strong bearish reaction from supply/resistance"
        if displacement == "STRONG" and bearish_flow:
            return True, "strong bearish reaction candle from resistance"
        if result.get("price") and result.get("ema20") and float(result["price"]) < float(result["ema20"]) and bearish_flow:
            return True, "rejected EMA20 with bearish short-term flow"
        if "Early Confirmation" in setup_status or "Strong Confirmation" in setup_status or "Trend Resumption" in setup_status:
            return True, "bearish confirmation started"
        return False, "waiting for bearish structure shift or resistance rejection"

    return False, "direction unclear"


def _stock_setup_grade(result: dict, daily_direction: str, setup_direction: str, location: str, setup_status: str) -> tuple:
    trade_direction = _stock_trade_direction(result)
    confirmation_started, confirmation_reason = _stock_confirmation(result, setup_direction, setup_status)
    loc = str(location or "").lower()
    in_zone = bool(result.get("in_ob") or result.get("near_ob"))

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
    if confirmation_started:
        return "A", f"A Setup — Confirmation Started: {confirmation_reason}", confirmation_started, confirmation_reason
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

    best_contract_start = time.perf_counter()
    if _should_enrich_best_contract(result, setup_grade, entry_status):
        best_contract = _best_contract(
            result.get("ticker") or "",
            result.get("direction") or "",
            result.get("entry") or 0,
            block_on_miss=False,
        )
    else:
        best_contract = _unclean_contract("Best contract not evaluated for non-actionable setup", "not_evaluated")
    best_contract_ms = round((time.perf_counter() - best_contract_start) * 1000, 1)

    timing = result.setdefault("_scan_timing", {})
    timing.update({
        "earnings_ms": earnings_ms,
        "best_contract_ms": best_contract_ms,
    })

    result.update({
        "stockTrend": daily_direction,
        "trendDirection": daily_direction,
        "dailyTrendDirection": daily_direction,
        "h4TrendDirection": h4_direction,
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
        "best_contract": best_contract,
    })
    return result


def scan_all(watchlist: Optional[list] = None, max_workers: int = 12) -> tuple:
    scan_start = time.perf_counter()
    _scan_activity_started()
    try:
        _ensure_background_refresh_started()
        if watchlist is None:
            watchlist = get_finviz_watchlist()
        else:
            watchlist = list(dict.fromkeys([str(t).strip().upper() for t in watchlist if str(t).strip()]))[:200]

        original_count = len(watchlist)

        # ── Step 1: batch-download daily OHLCV for cheap universe filtering ───────
        price_stage_start = time.perf_counter()
        daily_data  = _batch_download(watchlist, period="1y",  interval="1d")
        filtered_watchlist, skipped_symbols = _prefilter_stock_universe(watchlist, daily_data)
        skip_counts = {}
        for item in skipped_symbols:
            reason = item.get("reason") or "unknown"
            skip_counts[reason] = skip_counts.get(reason, 0) + 1
        if skipped_symbols:
            examples = ", ".join(f"{item['ticker']}:{item['reason']}" for item in skipped_symbols[:20])
            logger.info(
                "[universe filter] skipped=%s/%s reasons=%s examples=%s",
                len(skipped_symbols),
                original_count,
                skip_counts,
                examples,
            )

        if filtered_watchlist:
            weekly_data = _batch_download(filtered_watchlist, period="2y",  interval="1wk")
            h4_data     = _batch_download(filtered_watchlist, period="60d", interval="4h")
        else:
            weekly_data = {}
            h4_data = {}
        price_stage_ms = round((time.perf_counter() - price_stage_start) * 1000, 1)

        # ── Step 2: parallel-process each ticker (pure CPU/logic, no I/O) ─────────
        rows, near_miss = [], []
        process_stage_start = time.perf_counter()

        def _process(ticker: str):
            return scan_ticker(
                ticker,
                _daily_df=daily_data.get(ticker),
                _weekly_df=weekly_data.get(ticker),
                _h4_df=h4_data.get(ticker, pd.DataFrame()),
            )

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_process, t): t for t in filtered_watchlist}
            for future in as_completed(futures):
                r = future.result()
                if r is None:
                    continue
                if r.get("setup_status") == "QUALIFIED":
                    rows.append(r)
                else:
                    near_miss.append(r)

        rows.sort(key=lambda x: x.get("quality", {}).get("score", 0), reverse=True)
        near_miss.sort(key=lambda x: x.get("quality", {}).get("score", 0), reverse=True)
        process_stage_ms = round((time.perf_counter() - process_stage_start) * 1000, 1)
        all_results = [*rows, *near_miss]
        earnings_ms = round(sum((r.get("_scan_timing") or {}).get("earnings_ms", 0) for r in all_results), 1)
        best_contract_ms = round(sum((r.get("_scan_timing") or {}).get("best_contract_ms", 0) for r in all_results), 1)
        evaluated_contracts = sum(1 for r in all_results if (r.get("best_contract") or {}).get("source") != "not_evaluated")
        total_ms = round((time.perf_counter() - scan_start) * 1000, 1)
        cache_stats = _cache_snapshot()
        slow_contracts = [
            f"{r.get('ticker')}:{(r.get('_scan_timing') or {}).get('best_contract_ms', 0):.0f}ms"
            for r in all_results
            if (r.get("_scan_timing") or {}).get("best_contract_ms", 0) >= 750
        ][:8]
        logger.info(
            "[scan timing] total=%.1fms price_trend=%.1fms processing=%.1fms earnings_sum=%.1fms best_contract_sum=%.1fms contract_evaluated=%s/%s universe=%s/%s skipped=%s skip_reasons=%s cache=%s slow_contracts=%s",
            total_ms,
            price_stage_ms,
            process_stage_ms,
            earnings_ms,
            best_contract_ms,
            evaluated_contracts,
            len(all_results),
            len(filtered_watchlist),
            original_count,
            len(skipped_symbols),
            skip_counts,
            cache_stats,
            ", ".join(slow_contracts) if slow_contracts else "none",
        )
        return rows, near_miss
    finally:
        _scan_activity_finished()


def scan_cached(watchlist: Optional[list] = None, *, force_refresh: bool = False) -> dict:
    key = _analysis_cache_key(watchlist)
    with _cache_lock:
        cached = _analysis_cache.get(key)
    had_cached_analysis = cached is not None

    if cached and not force_refresh:
        refreshing_key = ("analysis_refresh", key)
        _submit_background_job(refreshing_key, _refresh_analysis_cache, key, watchlist)
        return {
            "rows": list(cached.get("rows", [])),
            "near_miss": list(cached.get("near_miss", [])),
            "meta": _analysis_cache_meta(key, cached, True),
        }

    rows, near_miss = scan_all(watchlist)
    cached = _store_analysis_cache(key, rows, near_miss)
    return {
        "rows": rows,
        "near_miss": near_miss,
        "meta": {
            **_analysis_cache_meta(key, cached, False),
            "cache": "refresh" if had_cached_analysis or force_refresh else "miss",
            "refreshing": False,
        },
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
        h4_source = yf.download(ticker, period="60d", interval="4h", progress=False, auto_adjust=True)

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
