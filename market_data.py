"""Market data provider foundation for Kairos stock candles.

This module is infrastructure only. Yahoo remains the default provider and the
scanner continues to consume pandas DataFrames with the same OHLCV columns it
already expects: Open, High, Low, Close, Volume.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, time as datetime_time, timedelta, timezone
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yahoo_finance

logger = logging.getLogger(__name__)


YAHOO_PROVIDER_NAME = "yahoo"
ALPACA_PROVIDER_NAME = "alpaca"
DEFAULT_DATA_PROVIDER = YAHOO_PROVIDER_NAME
DEFAULT_ALPACA_DATA_BASE_URL = "https://data.alpaca.markets"
EASTERN_TZ = ZoneInfo("America/New_York")
SCANNER_TIMEFRAMES = [
    {"label": "1D", "period": "1y", "interval": "1d", "minimum_candles": 50},
    {"label": "1W", "period": "2y", "interval": "1wk", "minimum_candles": 50},
    {"label": "4H", "period": "60d", "interval": "4h", "minimum_candles": 55},
]


@dataclass(frozen=True)
class CandleRequest:
    symbols: list[str]
    period: str
    interval: str
    auto_adjust: bool = True
    group_by: str = "ticker"


class MarketDataProvider:
    name = "base"

    def download(
        self,
        tickers,
        period: str,
        interval: str,
        progress: bool = False,
        auto_adjust: bool = True,
        group_by: str = "ticker",
        **kwargs,
    ):
        raise NotImplementedError

    def normalize_symbol(self, symbol: str) -> str:
        return str(symbol or "").strip().upper()


def _normalize_ohlcv(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    normalized = df.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = normalized.columns.get_level_values(-1)

    rename = {}
    for column in normalized.columns:
        key = str(column).strip().lower()
        if key in {"open", "o"}:
            rename[column] = "Open"
        elif key in {"high", "h"}:
            rename[column] = "High"
        elif key in {"low", "l"}:
            rename[column] = "Low"
        elif key in {"close", "c"}:
            rename[column] = "Close"
        elif key in {"volume", "v"}:
            rename[column] = "Volume"
    normalized = normalized.rename(columns=rename)

    for column in ["Open", "High", "Low", "Close", "Volume"]:
        if column not in normalized.columns:
            normalized[column] = pd.NA
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    return normalized[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")


def _as_symbol_list(tickers) -> list[str]:
    if isinstance(tickers, str):
        return [t.strip().upper() for t in tickers.replace(",", " ").split() if t.strip()]
    return [str(t).strip().upper() for t in (tickers or []) if str(t).strip()]


def _period_start(period: str) -> datetime:
    now = datetime.now(timezone.utc)
    period = str(period or "").strip().lower()
    unit = period[-1:] if period else "d"
    try:
        value = int(period[:-1])
    except ValueError:
        value = 365
        unit = "d"
    if unit == "d":
        return now - timedelta(days=value)
    if unit == "w":
        return now - timedelta(weeks=value)
    if unit == "mo":
        return now - timedelta(days=value * 31)
    if unit == "y":
        return now - timedelta(days=value * 365)
    return now - timedelta(days=365)


def _alpaca_timeframe(interval: str) -> str:
    mapping = {
        "1d": "1Day",
        "1wk": "1Week",
        "4h": "4Hour",
        "1h": "1Hour",
        "30m": "30Min",
        "15m": "15Min",
        "5m": "5Min",
        "1m": "1Min",
    }
    return mapping.get(str(interval or "").strip().lower(), str(interval or "1Day"))


class YahooMarketDataProvider(MarketDataProvider):
    name = YAHOO_PROVIDER_NAME

    def download(
        self,
        tickers,
        period: str,
        interval: str,
        progress: bool = False,
        auto_adjust: bool = True,
        group_by: str = "ticker",
        **kwargs,
    ):
        return yahoo_finance.download(
            tickers,
            period=period,
            interval=interval,
            progress=progress,
            auto_adjust=auto_adjust,
            group_by=group_by,
            **kwargs,
        )


class AlpacaMarketDataProvider(MarketDataProvider):
    name = ALPACA_PROVIDER_NAME

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY", "")
        self.base_url = (base_url or os.getenv("ALPACA_DATA_BASE_URL") or DEFAULT_ALPACA_DATA_BASE_URL).rstrip("/")

    def _headers(self) -> dict[str, str]:
        if not self.api_key or not self.secret_key:
            raise RuntimeError("Alpaca credentials are not configured")
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }

    def normalize_symbol(self, symbol: str) -> str:
        return str(symbol or "").strip().upper().replace("-", ".")

    def download(
        self,
        tickers,
        period: str,
        interval: str,
        progress: bool = False,
        auto_adjust: bool = True,
        group_by: str = "ticker",
        **kwargs,
    ):
        symbols = _as_symbol_list(tickers)
        if not symbols:
            return pd.DataFrame()

        start = _period_start(period)
        params = {
            "symbols": ",".join(self.normalize_symbol(s) for s in symbols),
            "timeframe": _alpaca_timeframe(interval),
            "start": start.isoformat().replace("+00:00", "Z"),
            "limit": 10000,
            "adjustment": "all" if auto_adjust else "raw",
        }
        url = f"{self.base_url}/v2/stocks/bars?{urlencode(params)}"
        try:
            req = Request(url, headers=self._headers())
            with urlopen(req, timeout=int(os.getenv("ALPACA_DATA_TIMEOUT", "20"))) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
            logger.warning("[alpaca] candle request failed symbols=%s interval=%s: %s", symbols, interval, exc)
            return pd.DataFrame() if len(symbols) == 1 else _empty_multi_symbol_frame(symbols)

        bars_by_symbol = payload.get("bars") or {}
        frames = {}
        for symbol in symbols:
            alpaca_symbol = self.normalize_symbol(symbol)
            bars = bars_by_symbol.get(alpaca_symbol) or bars_by_symbol.get(symbol) or []
            frames[symbol] = _bars_to_frame(bars)

        if len(symbols) == 1:
            return frames.get(symbols[0], pd.DataFrame())
        return _multi_symbol_frame(frames)


def _bars_to_frame(bars: list[dict[str, Any]]) -> pd.DataFrame:
    if not bars:
        return pd.DataFrame()
    rows = []
    index = []
    for bar in bars:
        timestamp = bar.get("t")
        if not timestamp:
            continue
        index.append(pd.to_datetime(timestamp, utc=True))
        rows.append({
            "Open": bar.get("o"),
            "High": bar.get("h"),
            "Low": bar.get("l"),
            "Close": bar.get("c"),
            "Volume": bar.get("v"),
        })
    return _normalize_ohlcv(pd.DataFrame(rows, index=pd.DatetimeIndex(index)))


def _multi_symbol_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    keyed = {}
    for symbol, df in frames.items():
        if df is not None and not df.empty:
            keyed[symbol] = _normalize_ohlcv(df)
    if not keyed:
        return _empty_multi_symbol_frame(list(frames.keys()))
    return pd.concat(keyed, axis=1)


def _empty_multi_symbol_frame(symbols: list[str]) -> pd.DataFrame:
    columns = pd.MultiIndex.from_product([symbols, ["Open", "High", "Low", "Close", "Volume"]])
    return pd.DataFrame(columns=columns)


def configured_provider_name() -> str:
    return (os.getenv("STOCK_DATA_PROVIDER") or DEFAULT_DATA_PROVIDER).strip().lower() or DEFAULT_DATA_PROVIDER


def alpaca_credentials_configured() -> bool:
    return bool(os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY"))


def build_market_data_provider(name: Optional[str] = None) -> MarketDataProvider:
    provider_name = (name or configured_provider_name()).strip().lower()
    if provider_name == ALPACA_PROVIDER_NAME:
        return AlpacaMarketDataProvider()
    if provider_name != YAHOO_PROVIDER_NAME:
        logger.warning("[market-data] unknown provider %s; falling back to yahoo", provider_name)
    return YahooMarketDataProvider()


class MarketDataFacade:
    """Small yfinance-compatible facade used by scanner.py.

    `download` routes through the configured candle provider. `Ticker` remains
    Yahoo-backed so options and earnings behavior stays unchanged in this sprint.
    """

    def __init__(self) -> None:
        self.provider = build_market_data_provider()

    def download(self, *args, **kwargs):
        return self.provider.download(*args, **kwargs)

    def Ticker(self, *args, **kwargs):
        return yahoo_finance.Ticker(*args, **kwargs)

    @property
    def provider_name(self) -> str:
        return self.provider.name


def comparison_diagnostics(ticker: str, period: str = "60d", interval: str = "4h") -> dict:
    try:
        validation = validate_candle_pair(ticker=ticker, period=period, interval=interval)
        return {
            "success": validation["result"] != "FAILURE",
            "ticker": validation["ticker"],
            "period": validation["period"],
            "interval": validation["interval"],
            "production_provider": validation["production_provider"],
            "timing_context": validation["timing_context"],
            "providers": validation["providers"],
            "comparison": validation["comparison"],
            "result": validation["result"],
            "migration_readiness": validation["migration_readiness"],
            "readiness_category": validation["migration_readiness"],
            "classifications": validation["classifications"],
            "error_classification": None if validation["result"] != "FAILURE" else "; ".join(validation["classifications"]),
            "sanitized_error_message": None if validation["result"] != "FAILURE" else "Provider comparison did not complete. See provider status fields for sanitized details.",
        }
    except Exception as exc:
        logger.warning("[market-data] sanitized comparison failure ticker=%s interval=%s error=%s", ticker, interval, _classify_error(exc))
        return _comparison_failure_response(ticker=ticker, period=period, interval=interval, error_classification=_classify_error(exc))


def validate_candle_pair(ticker: str, period: str = "60d", interval: str = "4h") -> dict:
    symbol = str(ticker or "").strip().upper()
    timing = validation_timing_context()
    start_et = datetime.now(EASTERN_TZ)
    yahoo_provider = YahooMarketDataProvider()
    alpaca_provider = AlpacaMarketDataProvider()
    minimum = _minimum_candles_for_interval(interval)

    if not alpaca_credentials_configured():
        yahoo_df = pd.DataFrame()
        yahoo_error = "not_requested_alpaca_blocked"
        alpaca_df = pd.DataFrame()
        alpaca_error = "missing_credentials"
    else:
        yahoo_df, yahoo_error = _safe_provider_download(yahoo_provider, symbol, period, interval)
        alpaca_df, alpaca_error = _safe_provider_download(alpaca_provider, symbol, period, interval)

    yahoo_diag = _provider_diagnostics(
        provider_name="yahoo",
        df=yahoo_df,
        requested_symbol=symbol,
        provider_symbol=yahoo_provider.normalize_symbol(symbol),
        interval=interval,
        minimum_candles=minimum,
        error=("%s" % yahoo_error) if yahoo_error else None,
        timing_context=timing,
    )
    alpaca_diag = _provider_diagnostics(
        provider_name="alpaca",
        df=alpaca_df,
        requested_symbol=symbol,
        provider_symbol=alpaca_provider.normalize_symbol(symbol),
        interval=interval,
        minimum_candles=minimum,
        error=alpaca_error,
        timing_context=timing,
    )
    comparison = _compare_completed_frames(yahoo_df, alpaca_df)
    result, readiness, classifications = _classify_validation_result(yahoo_diag, alpaca_diag, comparison, timing)

    return {
        "success": result != "FAILURE",
        "ticker": symbol,
        "period": period,
        "interval": interval,
        "timeframe": _timeframe_label(period, interval),
        "production_provider": configured_provider_name(),
        "timing_context": timing,
        "providers": {
            "yahoo": yahoo_diag,
            "alpaca": alpaca_diag,
        },
        "comparison": comparison,
        "result": result,
        "migration_readiness": readiness,
        "classifications": classifications,
        "started_at_et": start_et.isoformat(),
        "finished_at_et": datetime.now(EASTERN_TZ).isoformat(),
    }


def validate_watchlist_candles(tickers: list[str]) -> dict:
    started = validation_timing_context()
    rows = []
    for ticker in tickers:
        symbol = str(ticker or "").strip().upper()
        if not symbol:
            continue
        for tf in SCANNER_TIMEFRAMES:
            rows.append(validate_candle_pair(symbol, period=tf["period"], interval=tf["interval"]))
    ended = validation_timing_context()
    counts = {}
    readiness_counts = {}
    for row in rows:
        counts[row["result"]] = counts.get(row["result"], 0) + 1
        readiness_counts[row["migration_readiness"]] = readiness_counts.get(row["migration_readiness"], 0) + 1
    return {
        "success": not any(row.get("result") == "FAILURE" for row in rows),
        "started": started,
        "ended": ended,
        "active_production_provider": configured_provider_name(),
        "alpaca_configured": alpaca_credentials_configured(),
        "timeframes": SCANNER_TIMEFRAMES,
        "ticker_count": len([t for t in tickers if str(t).strip()]),
        "combination_count": len(rows),
        "result_counts": counts,
        "readiness_counts": readiness_counts,
        "rows": rows,
    }


def validation_timing_context(now: Optional[datetime] = None) -> dict:
    now_et = (now or datetime.now(timezone.utc)).astimezone(EASTERN_TZ)
    session = market_session(now_et)
    return {
        "timestamp_et": now_et.isoformat(),
        "day_of_week": now_et.strftime("%A"),
        "market_session": session,
        "market_open": session == "regular session",
    }


def market_session(now_et: datetime) -> str:
    if now_et.weekday() >= 5:
        return "closed"
    current = now_et.time()
    if datetime_time(4, 0) <= current < datetime_time(9, 30):
        return "pre-market"
    if datetime_time(9, 30) <= current < datetime_time(16, 0):
        return "regular session"
    if datetime_time(16, 0) <= current < datetime_time(20, 0):
        return "after-hours"
    return "closed"


def _safe_provider_download(provider: MarketDataProvider, symbol: str, period: str, interval: str) -> tuple[pd.DataFrame, Optional[str]]:
    try:
        df = provider.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        normalized = _normalize_ohlcv(df)
        if normalized.empty:
            return normalized, "empty_result"
        return normalized, None
    except Exception as exc:
        logger.warning("[%s] sanitized comparison failure symbol=%s interval=%s error=%s", provider.name, symbol, interval, _classify_error(exc))
        return pd.DataFrame(), _classify_error(exc)


def _provider_diagnostics(
    *,
    provider_name: str,
    df: pd.DataFrame,
    requested_symbol: str,
    provider_symbol: str,
    interval: str,
    minimum_candles: int,
    error: Optional[str],
    timing_context: dict,
) -> dict:
    latest_raw = None
    latest_normalized = None
    latest_completed = None
    scanner_input_latest = None
    latest_completed_ohlcv = None
    if df is not None and not df.empty:
        latest_raw = _timestamp_to_iso(df.index[-1])
        latest_normalized = latest_raw
        forming = _is_forming_candle(df.index[-1], interval, timing_context)
        completed_df = df.iloc[:-1] if forming else df
        if not completed_df.empty:
            latest_completed = _timestamp_to_iso(completed_df.index[-1])
            last = completed_df.iloc[-1]
            latest_completed_ohlcv = _ohlcv_dict(last)
        scanner_input_latest = latest_completed
    else:
        forming = False
        completed_df = pd.DataFrame()

    status = "success"
    if error:
        status = "not_requested" if error == "not_requested_alpaca_blocked" else "blocked" if error == "missing_credentials" else "failure"
    elif df is None or df.empty:
        status = "failure"

    return {
        "provider": provider_name,
        "status": status,
        "error_classification": error,
        "sanitized_error_message": _sanitized_error_message(error),
        "requested_symbol": requested_symbol,
        "provider_symbol": provider_symbol,
        "candle_count": 0 if df is None else int(len(df)),
        "first_normalized_timestamp": _timestamp_to_iso(df.index[0]) if df is not None and not df.empty else None,
        "raw_latest_timestamp": latest_raw,
        "normalized_latest_timestamp": latest_normalized,
        "latest_completed_timestamp": latest_completed,
        "scanner_input_latest_timestamp": scanner_input_latest,
        "latest_completed_ohlcv": latest_completed_ohlcv,
        "incomplete_candle_excluded": bool(forming),
        "incomplete_candle_exclusion_reason": _incomplete_reason(forming, interval, timing_context),
        "duplicate_timestamps": _duplicate_timestamp_count(df),
        "out_of_order_timestamps": _out_of_order_timestamp_count(df),
        "detected_gap_count": _detected_gap_count(df, interval),
        "sufficient_history": (df is not None and len(df) >= minimum_candles),
        "minimum_required_candles": minimum_candles,
        "columns": [] if df is None else [str(c) for c in df.columns],
    }


def _compare_completed_frames(yahoo_df: pd.DataFrame, alpaca_df: pd.DataFrame) -> dict:
    if yahoo_df is None or alpaca_df is None or yahoo_df.empty or alpaca_df.empty:
        return {
            "completed_timestamp_matches": 0,
            "latest_timestamp_match": None,
            "latest_ohlcv_delta": None,
            "missing_in_alpaca": [],
            "missing_in_yahoo": [],
            "matched_latest_completed": None,
        }
    yahoo_lookup = {str(i): i for i in getattr(yahoo_df, "index", [])}
    alpaca_lookup = {str(i): i for i in getattr(alpaca_df, "index", [])}
    yahoo_index = set(yahoo_lookup.keys())
    alpaca_index = set(alpaca_lookup.keys())
    common = sorted(yahoo_index & alpaca_index)
    latest_match = None
    latest_ohlcv_delta = None
    matched_latest = None
    if common:
        matched_latest = common[-1]
        y = yahoo_df.loc[yahoo_lookup[matched_latest]]
        a = alpaca_df.loc[alpaca_lookup[matched_latest]]
        if isinstance(y, pd.DataFrame):
            y = y.iloc[-1]
        if isinstance(a, pd.DataFrame):
            a = a.iloc[-1]
        latest_match = str(yahoo_df.index[-1]) == str(alpaca_df.index[-1])
        latest_ohlcv_delta = _ohlcv_delta(y, a)
    return {
        "candle_count_delta": int((0 if alpaca_df is None else len(alpaca_df)) - (0 if yahoo_df is None else len(yahoo_df))),
        "completed_timestamp_matches": len(common),
        "latest_timestamp_match": latest_match,
        "matched_latest_completed": matched_latest,
        "latest_ohlcv_delta": latest_ohlcv_delta,
        "missing_in_alpaca": sorted(yahoo_index - alpaca_index),
        "missing_in_yahoo": sorted(alpaca_index - yahoo_index),
    }


def _classify_validation_result(yahoo_diag: dict, alpaca_diag: dict, comparison: dict, timing: dict) -> tuple[str, str, list[str]]:
    classifications = []
    if alpaca_diag.get("error_classification") == "missing_credentials":
        return "FAILURE", "BLOCKED BY DATA ACCESS", ["Alpaca credentials missing"]
    if alpaca_diag.get("status") != "success":
        return "FAILURE", "BLOCKED BY DATA ACCESS", [f"Alpaca {alpaca_diag.get('error_classification') or 'provider failure'}"]
    if yahoo_diag.get("status") != "success":
        return "FAILURE", "NOT READY", [f"Yahoo {yahoo_diag.get('error_classification') or 'provider failure'}"]
    if not alpaca_diag.get("sufficient_history"):
        classifications.append("insufficient history")
    if alpaca_diag.get("duplicate_timestamps") or yahoo_diag.get("duplicate_timestamps"):
        classifications.append("duplicate bars")
    if alpaca_diag.get("out_of_order_timestamps") or yahoo_diag.get("out_of_order_timestamps"):
        classifications.append("out-of-order bars")
    if comparison.get("completed_timestamp_matches", 0) == 0:
        classifications.append("timestamp alignment")
    if comparison.get("missing_in_alpaca"):
        classifications.append("missing bars")
    if comparison.get("latest_ohlcv_delta"):
        classifications.append("provider feed difference")
    if classifications:
        readiness = "NEEDS MANUAL REVIEW" if classifications == ["provider feed difference"] else "NOT READY"
        return "WARNING", readiness, classifications
    if timing.get("market_session") == "closed":
        return "WARNING", "PENDING LIVE-MARKET VALIDATION", ["closed-market run; forming candle behavior pending"]
    return "PASS", "READY", []


def _ohlcv_dict(row) -> dict:
    return {
        "open": _safe_float(row.get("Open")),
        "high": _safe_float(row.get("High")),
        "low": _safe_float(row.get("Low")),
        "close": _safe_float(row.get("Close")),
        "volume": _safe_float(row.get("Volume")),
    }


def _ohlcv_delta(yahoo_row, alpaca_row) -> dict:
    delta = {}
    for key in ["Open", "High", "Low", "Close", "Volume"]:
        y = _safe_float(yahoo_row.get(key))
        a = _safe_float(alpaca_row.get(key))
        if y is None or a is None:
            continue
        absolute = a - y
        pct = (absolute / y * 100) if y else None
        delta[key.lower()] = {
            "absolute": absolute,
            "percent": pct,
        }
    return delta


def _timestamp_to_iso(value) -> Optional[str]:
    if value is None:
        return None
    try:
        return pd.Timestamp(value).isoformat()
    except Exception:
        return str(value)


def _duplicate_timestamp_count(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    return int(pd.Index(df.index).duplicated().sum())


def _out_of_order_timestamp_count(df: pd.DataFrame) -> int:
    if df is None or df.empty or len(df.index) < 2:
        return 0
    values = list(pd.to_datetime(df.index))
    return sum(1 for prev, cur in zip(values, values[1:]) if cur < prev)


def _detected_gap_count(df: pd.DataFrame, interval: str) -> int:
    if df is None or df.empty or len(df.index) < 3:
        return 0
    expected = _interval_timedelta(interval)
    if expected is None:
        return 0
    values = list(pd.to_datetime(df.index))
    return sum(1 for prev, cur in zip(values, values[1:]) if cur - prev > expected * 1.75)


def _interval_timedelta(interval: str) -> Optional[timedelta]:
    interval = str(interval or "").lower()
    if interval.endswith("m"):
        return timedelta(minutes=int(interval[:-1]))
    if interval.endswith("h"):
        return timedelta(hours=int(interval[:-1]))
    if interval == "1d":
        return timedelta(days=1)
    if interval == "1wk":
        return timedelta(days=7)
    return None


def _is_forming_candle(timestamp, interval: str, timing_context: dict) -> bool:
    if timing_context.get("market_session") == "closed":
        return False
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone.utc)
    now_et = pd.Timestamp(timing_context["timestamp_et"])
    ts_et = ts.tz_convert(EASTERN_TZ)
    interval_delta = _interval_timedelta(interval)
    if interval_delta is None:
        return False
    return ts_et <= now_et < ts_et + interval_delta


def _incomplete_reason(forming: bool, interval: str, timing_context: dict) -> str:
    if forming:
        return f"{interval} candle is still forming during {timing_context.get('market_session')}"
    if timing_context.get("market_session") == "closed":
        return "market closed; latest normalized candle treated as completed"
    return "latest normalized candle is not currently forming"


def _minimum_candles_for_interval(interval: str) -> int:
    interval = str(interval or "").lower()
    if interval == "1wk":
        return 50
    if interval == "4h":
        return 55
    return 50


def _timeframe_label(period: str, interval: str) -> str:
    for tf in SCANNER_TIMEFRAMES:
        if tf["period"] == period and tf["interval"] == interval:
            return tf["label"]
    return f"{period}/{interval}"


def _classify_error(exc: Exception) -> str:
    text = str(exc).lower()
    if "credential" in text or "unauthorized" in text or "forbidden" in text or "401" in text or "403" in text:
        return "authorization"
    if "timed out" in text or "timeout" in text:
        return "timeout"
    if "resolve" in text or "dns" in text or "nodename" in text:
        return "network"
    return exc.__class__.__name__


def _sanitized_error_message(error: Optional[str]) -> Optional[str]:
    if not error:
        return None
    if error == "missing_credentials":
        return "Alpaca credentials are not configured for comparison diagnostics."
    if error == "not_requested_alpaca_blocked":
        return "Yahoo request was skipped because Alpaca comparison access is blocked."
    if error == "empty_result":
        return "Provider returned no usable normalized candles."
    if error == "authorization":
        return "Provider authorization failed or the data feed is unavailable for this request."
    if error == "timeout":
        return "Provider request timed out."
    if error == "network":
        return "Provider network request failed."
    return "Provider comparison failed before usable normalized candles were available."


def _comparison_failure_response(ticker: str, period: str, interval: str, error_classification: str) -> dict:
    symbol = str(ticker or "").strip().upper()
    timing = validation_timing_context()
    return {
        "success": False,
        "ticker": symbol,
        "period": period,
        "interval": interval,
        "production_provider": configured_provider_name(),
        "timing_context": timing,
        "providers": {
            "yahoo": {
                "provider": "yahoo",
                "status": "unknown",
                "error_classification": "comparison_not_completed",
                "sanitized_error_message": "Comparison failed before Yahoo provider diagnostics completed.",
                "requested_symbol": symbol,
                "provider_symbol": YahooMarketDataProvider().normalize_symbol(symbol),
            },
            "alpaca": {
                "provider": "alpaca",
                "status": "unknown",
                "error_classification": "comparison_not_completed",
                "sanitized_error_message": "Comparison failed before Alpaca provider diagnostics completed.",
                "requested_symbol": symbol,
                "provider_symbol": AlpacaMarketDataProvider().normalize_symbol(symbol),
            },
        },
        "comparison": {},
        "result": "FAILURE",
        "migration_readiness": "NOT READY",
        "readiness_category": "NOT READY",
        "classifications": ["implementation defect"],
        "error_classification": error_classification,
        "sanitized_error_message": "Provider comparison failed safely. No credentials, headers, upstream payloads, or stack traces are included.",
    }


def _safe_float(value) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None
