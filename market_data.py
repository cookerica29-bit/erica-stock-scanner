"""Market data provider foundation for Kairos stock candles.

This module is infrastructure only. Yahoo remains the default provider and the
scanner continues to consume pandas DataFrames with the same OHLCV columns it
already expects: Open, High, Low, Close, Volume.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from dataclasses import dataclass
from datetime import datetime, time as datetime_time, timedelta, timezone
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yahoo_finance

logger = logging.getLogger(__name__)


YAHOO_PROVIDER_NAME = "yahoo"
ALPACA_PROVIDER_NAME = "alpaca"
DEFAULT_DATA_PROVIDER = YAHOO_PROVIDER_NAME
DEFAULT_ALPACA_DATA_BASE_URL = "https://data.alpaca.markets"
DEFAULT_ALPACA_MAX_PAGES = 25
MAX_ALPACA_MAX_PAGES = 100
DEFAULT_ALPACA_BAR_SYMBOL_CHUNK_SIZE = 200
DEFAULT_ALPACA_QUOTE_CHUNK_SIZE = 200
EASTERN_TZ = ZoneInfo("America/New_York")
SCANNER_TIMEFRAMES = [
    {"label": "1D", "period": "1y", "interval": "1d", "minimum_candles": 50},
    {"label": "1W", "period": "2y", "interval": "1wk", "minimum_candles": 50},
    {"label": "4H", "period": "60d", "interval": "4h", "minimum_candles": 55},
]
PROVIDER_PROFILE_PRODUCTION_YAHOO = "production_yahoo"
PROVIDER_PROFILE_PROPOSED_HYBRID = "proposed_hybrid_alpaca_1d_1w_yahoo_4h"
DEFAULT_DATA_PROVIDER_PROFILE = PROVIDER_PROFILE_PRODUCTION_YAHOO
TIMEFRAME_PROVIDER_PROFILES = {
    PROVIDER_PROFILE_PRODUCTION_YAHOO: {
        "1D": YAHOO_PROVIDER_NAME,
        "1W": YAHOO_PROVIDER_NAME,
        "4H": YAHOO_PROVIDER_NAME,
    },
    PROVIDER_PROFILE_PROPOSED_HYBRID: {
        "1D": ALPACA_PROVIDER_NAME,
        "1W": ALPACA_PROVIDER_NAME,
        "4H": YAHOO_PROVIDER_NAME,
    },
}

_provider_metrics_lock = threading.RLock()
_provider_metrics = {
    "alpaca_bar_requests": 0,
    "alpaca_bar_pages": 0,
    "alpaca_bar_symbols_requested": 0,
    "alpaca_bar_symbols_succeeded": 0,
    "alpaca_bar_symbols_failed": 0,
    "alpaca_max_pages_exceeded_count": 0,
}


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


def _alpaca_period_start(period: str, interval: str) -> datetime:
    start = _period_start(period)
    interval = str(interval or "").strip().lower()
    if interval == "1wk":
        start = start - timedelta(days=start.weekday())
    return start.replace(hour=0, minute=0, second=0, microsecond=0)


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


def _normalize_alpaca_base_url(base_url: str) -> str:
    normalized = str(base_url or DEFAULT_ALPACA_DATA_BASE_URL).strip().strip("\"'")
    if not normalized:
        normalized = DEFAULT_ALPACA_DATA_BASE_URL
    if not normalized.startswith(("http://", "https://")):
        normalized = f"https://{normalized}"
    normalized = normalized.rstrip("/")
    for suffix in ("/v2/stocks/bars", "/v2/stocks", "/v2"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    parsed = urlparse(normalized)
    if not parsed.scheme or not parsed.netloc:
        return DEFAULT_ALPACA_DATA_BASE_URL
    return normalized


def _parse_positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(1, value)


def _parse_bounded_positive_int_env(name: str, default: int, maximum: int) -> int:
    return min(_parse_positive_int_env(name, default), max(1, int(maximum)))


def reset_provider_metrics() -> None:
    with _provider_metrics_lock:
        for key in _provider_metrics:
            _provider_metrics[key] = 0


def provider_metrics_snapshot() -> dict[str, int]:
    with _provider_metrics_lock:
        return dict(_provider_metrics)


def _record_provider_metric(name: str, amount: int = 1) -> None:
    with _provider_metrics_lock:
        _provider_metrics[name] = int(_provider_metrics.get(name, 0)) + int(amount or 0)


def _chunks(items: list[str], size: int):
    for index in range(0, len(items), size):
        yield items[index:index + size]


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
        self.base_url = _normalize_alpaca_base_url(base_url or os.getenv("ALPACA_DATA_BASE_URL") or DEFAULT_ALPACA_DATA_BASE_URL)

    def _headers(self) -> dict[str, str]:
        if not self.api_key or not self.secret_key:
            raise RuntimeError("Alpaca credentials are not configured")
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }

    def normalize_symbol(self, symbol: str) -> str:
        return str(symbol or "").strip().upper().replace("-", ".")

    def _request_bars_page(self, params: dict[str, Any]) -> dict:
        url = f"{self.base_url}/v2/stocks/bars?{urlencode(params)}"
        req = Request(url, headers=self._headers())
        with urlopen(req, timeout=int(os.getenv("ALPACA_DATA_TIMEOUT", "20"))) as response:
            return json.loads(response.read().decode("utf-8"))

    def _request_latest_quotes(self, params: dict[str, Any]) -> dict:
        url = f"{self.base_url}/v2/stocks/quotes/latest?{urlencode(params)}"
        req = Request(url, headers=self._headers())
        with urlopen(req, timeout=int(os.getenv("ALPACA_DATA_TIMEOUT", "20"))) as response:
            return json.loads(response.read().decode("utf-8"))

    def _request_bars_pages(self, params: dict[str, Any], symbols: list[str]) -> dict[str, list[dict[str, Any]]]:
        max_pages_env = "ALPACA_BARS_MAX_PAGES" if os.getenv("ALPACA_BARS_MAX_PAGES") else "ALPACA_MAX_PAGES"
        max_pages = _parse_bounded_positive_int_env(max_pages_env, DEFAULT_ALPACA_MAX_PAGES, MAX_ALPACA_MAX_PAGES)
        reverse_symbol_map = {self.normalize_symbol(symbol): symbol for symbol in symbols}
        bars_by_symbol = {symbol: [] for symbol in symbols}
        page_token = None
        seen_tokens = set()
        page_count = 0

        while True:
            if page_count >= max_pages:
                logger.warning(
                    "[alpaca] candle pagination failed pages_completed=%s symbols=%s timeframe=%s error=max_pages_exceeded",
                    page_count,
                    len(symbols),
                    params.get("timeframe"),
                )
                _record_provider_metric("alpaca_max_pages_exceeded_count")
                raise RuntimeError("Alpaca pagination exceeded max pages")

            page_params = dict(params)
            if page_token:
                if page_token in seen_tokens:
                    logger.warning(
                        "[alpaca] candle pagination failed pages_completed=%s symbols=%s timeframe=%s error=repeated_page_token",
                        page_count,
                        len(symbols),
                        params.get("timeframe"),
                    )
                    raise RuntimeError("Alpaca pagination repeated page token")
                seen_tokens.add(page_token)
                page_params["page_token"] = page_token

            try:
                payload = self._request_bars_page(page_params)
            except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
                logger.warning(
                    "[alpaca] candle pagination failed pages_completed=%s symbols=%s timeframe=%s error=%s",
                    page_count,
                    len(symbols),
                    params.get("timeframe"),
                    _classify_error(exc),
                )
                raise

            page_count += 1
            _record_provider_metric("alpaca_bar_pages")
            page_bars = payload.get("bars") or {}
            for provider_symbol, bars in page_bars.items():
                original_symbol = reverse_symbol_map.get(str(provider_symbol or "").strip().upper())
                if not original_symbol or not isinstance(bars, list):
                    continue
                bars_by_symbol.setdefault(original_symbol, []).extend(bars)

            page_token = payload.get("next_page_token")
            if not page_token:
                logger.info(
                    "[alpaca] candle pagination completed pages=%s symbols=%s timeframe=%s",
                    page_count,
                    len(symbols),
                    params.get("timeframe"),
                )
                return bars_by_symbol

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

        start = _alpaca_period_start(period, interval)
        chunk_size = _parse_positive_int_env("ALPACA_BAR_SYMBOL_CHUNK_SIZE", DEFAULT_ALPACA_BAR_SYMBOL_CHUNK_SIZE)
        bars_by_symbol = {symbol: [] for symbol in symbols}
        failed_symbols: set[str] = set()

        for chunk in _chunks(symbols, chunk_size):
            params = {
                "symbols": ",".join(self.normalize_symbol(s) for s in chunk),
                "timeframe": _alpaca_timeframe(interval),
                "start": start.isoformat().replace("+00:00", "Z"),
                "limit": 10000,
                "adjustment": "all" if auto_adjust else "raw",
            }
            _record_provider_metric("alpaca_bar_requests")
            _record_provider_metric("alpaca_bar_symbols_requested", len(chunk))
            try:
                chunk_bars = self._request_bars_pages(params, chunk)
            except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
                failed_symbols.update(chunk)
                _record_provider_metric("alpaca_bar_symbols_failed", len(chunk))
                logger.warning("[alpaca] candle request failed symbols=%s interval=%s error=%s", len(chunk), interval, _classify_error(exc))
                continue
            for symbol, bars in chunk_bars.items():
                bars_by_symbol[symbol] = bars

        frames = {}
        for symbol in symbols:
            bars = bars_by_symbol.get(symbol) or []
            frames[symbol] = _bars_to_frame(bars)
            if symbol in failed_symbols or frames[symbol].empty:
                if symbol not in failed_symbols:
                    _record_provider_metric("alpaca_bar_symbols_failed")
            else:
                _record_provider_metric("alpaca_bar_symbols_succeeded")

        if len(symbols) == 1:
            return frames.get(symbols[0], pd.DataFrame())
        return _multi_symbol_frame(frames)

    @staticmethod
    def _quote_price(quote: dict[str, Any]) -> Optional[float]:
        def positive_number(value) -> Optional[float]:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return None
            return number if number > 0 else None

        bid = positive_number(quote.get("bp"))
        ask = positive_number(quote.get("ap"))
        if bid is not None and ask is not None:
            return (bid + ask) / 2
        if ask is not None:
            return ask
        if bid is not None:
            return bid
        return None

    def latest_quotes(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        requested_symbols = list(dict.fromkeys(_as_symbol_list(symbols)))
        if not requested_symbols:
            return {}

        reverse_symbol_map = {self.normalize_symbol(symbol): symbol for symbol in requested_symbols}
        chunk_size = _parse_positive_int_env("ALPACA_QUOTE_CHUNK_SIZE", DEFAULT_ALPACA_QUOTE_CHUNK_SIZE)
        results: dict[str, dict[str, Any]] = {}
        for chunk in _chunks(requested_symbols, chunk_size):
            params = {"symbols": ",".join(self.normalize_symbol(symbol) for symbol in chunk)}
            try:
                payload = self._request_latest_quotes(params)
            except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
                logger.warning(
                    "[alpaca] latest quote request failed symbols=%s error=%s",
                    len(chunk),
                    _classify_error(exc),
                )
                continue

            quotes = payload.get("quotes") or {}
            for provider_symbol, quote in quotes.items():
                original_symbol = reverse_symbol_map.get(str(provider_symbol or "").strip().upper())
                if not original_symbol or not isinstance(quote, dict):
                    continue
                price = self._quote_price(quote)
                if price is None:
                    continue
                results[original_symbol] = {
                    "price": price,
                    "bid": quote.get("bp"),
                    "ask": quote.get("ap"),
                    "timestamp": quote.get("t"),
                    "source": "alpaca_latest_quote",
                }

        logger.info(
            "[alpaca] latest quotes completed symbols=%s quotes=%s chunks=%s",
            len(requested_symbols),
            len(results),
            math.ceil(len(requested_symbols) / chunk_size),
        )
        return results


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


def configured_provider_profile_name() -> str:
    profile = (os.getenv("STOCK_DATA_PROVIDER_PROFILE") or DEFAULT_DATA_PROVIDER_PROFILE).strip().lower()
    if profile not in TIMEFRAME_PROVIDER_PROFILES:
        logger.warning(
            "[market-data] unknown provider profile %s; falling back to %s",
            profile,
            DEFAULT_DATA_PROVIDER_PROFILE,
        )
        return DEFAULT_DATA_PROVIDER_PROFILE
    return profile


def configured_timeframe_provider_profile() -> dict:
    return timeframe_provider_profile(configured_provider_profile_name())


def provider_name_for_timeframe(label: str, profile: Optional[dict] = None) -> str:
    timeframe = str(label or "").strip().upper()
    active_profile = profile or configured_timeframe_provider_profile()
    return active_profile.get(timeframe, YAHOO_PROVIDER_NAME)


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
    comparison = _compare_completed_frames(yahoo_df, alpaca_df, interval)
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


def timeframe_provider_profile(profile: str) -> dict:
    profile_key = str(profile or PROVIDER_PROFILE_PRODUCTION_YAHOO).strip().lower()
    return dict(TIMEFRAME_PROVIDER_PROFILES.get(profile_key, TIMEFRAME_PROVIDER_PROFILES[PROVIDER_PROFILE_PRODUCTION_YAHOO]))


def hybrid_strategy_diagnostics(tickers: list[str]) -> dict:
    """Compare frozen scanner outputs for Yahoo-only versus proposed hybrid data.

    This is backend diagnostics only. It does not alter configured production
    provider selection and does not introduce fallback behavior.
    """
    symbols = [str(t or "").strip().upper() for t in (tickers or []) if str(t or "").strip()]
    started = validation_timing_context()
    yahoo_provider = YahooMarketDataProvider()
    alpaca_provider = AlpacaMarketDataProvider()
    production_profile = timeframe_provider_profile(PROVIDER_PROFILE_PRODUCTION_YAHOO)
    hybrid_profile = timeframe_provider_profile(PROVIDER_PROFILE_PROPOSED_HYBRID)
    rows = []

    for symbol in symbols:
        production, production_errors = _scanner_output_for_profile(
            symbol=symbol,
            profile=production_profile,
            yahoo_provider=yahoo_provider,
            alpaca_provider=alpaca_provider,
        )
        hybrid, hybrid_errors = _scanner_output_for_profile(
            symbol=symbol,
            profile=hybrid_profile,
            yahoo_provider=yahoo_provider,
            alpaca_provider=alpaca_provider,
        )
        comparison = _compare_strategy_outputs(production, hybrid)
        if production_errors or hybrid_errors:
            classification = "unresolved"
        elif comparison["material_differences"]:
            classification = "material difference"
        elif comparison["differences"]:
            classification = "minor explainable difference"
        else:
            classification = "exact strategy-output match"
        rows.append({
            "ticker": symbol,
            "production_profile": production_profile,
            "hybrid_profile": hybrid_profile,
            "production_errors": production_errors,
            "hybrid_errors": hybrid_errors,
            "production_output": production,
            "hybrid_output": hybrid,
            "comparison": comparison,
            "classification": classification,
        })

    ended = validation_timing_context()
    counts = {}
    for row in rows:
        counts[row["classification"]] = counts.get(row["classification"], 0) + 1
    return {
        "success": not any(row["classification"] in {"material difference", "unresolved"} for row in rows),
        "started": started,
        "ended": ended,
        "active_production_provider": configured_provider_name(),
        "production_profile": production_profile,
        "proposed_hybrid_profile": hybrid_profile,
        "alpaca_configured": alpaca_credentials_configured(),
        "ticker_count": len(symbols),
        "classification_counts": counts,
        "rows": rows,
        "note": "Comparison-only diagnostics. Production scanner output remains configured separately and is unchanged.",
    }


def _scanner_output_for_profile(
    symbol: str,
    profile: dict,
    yahoo_provider: YahooMarketDataProvider,
    alpaca_provider: AlpacaMarketDataProvider,
) -> tuple[Optional[dict], list[dict]]:
    from scanner import scan_ticker

    errors = []
    frames = {}
    requests = {
        "1D": ("1y", "1d"),
        "1W": ("2y", "1wk"),
        "4H": ("60d", "4h"),
    }
    for label, (period, interval) in requests.items():
        provider_name = profile.get(label)
        provider = alpaca_provider if provider_name == ALPACA_PROVIDER_NAME else yahoo_provider
        df, error = _safe_provider_download(provider, symbol, period, interval)
        if error:
            errors.append({
                "timeframe": label,
                "provider": provider.name,
                "error_classification": error,
                "sanitized_error_message": _sanitized_error_message(error),
            })
        elif df is None or df.empty:
            errors.append({
                "timeframe": label,
                "provider": provider.name,
                "error_classification": "empty_result",
                "sanitized_error_message": _sanitized_error_message("empty_result"),
            })
        frames[label] = df

    if errors:
        return None, errors

    result = scan_ticker(
        symbol,
        _daily_df=frames["1D"],
        _weekly_df=frames["1W"],
        _h4_df=frames["4H"],
    )
    return _strategy_output_snapshot(result), []


def _strategy_output_snapshot(result: Optional[dict]) -> Optional[dict]:
    if not result:
        return None
    trade_eval = result.get("trade_eval") or {}
    quality = result.get("quality") or {}
    return {
        "ticker": result.get("ticker"),
        "selected_timeframe": result.get("timeframe"),
        "setup_status": result.get("setup_status"),
        "trend": result.get("trend"),
        "direction": result.get("direction"),
        "setup_grade": result.get("setupGrade"),
        "quality_grade": quality.get("grade"),
        "quality_score": quality.get("score"),
        "confirmation_started": result.get("confirmationStarted"),
        "confirmation_reason": result.get("confirmationReason"),
        "entry_status": result.get("entryStatus"),
        "entry": result.get("entry"),
        "stop": result.get("sl"),
        "target_1": result.get("tp1"),
        "target_2": result.get("tp2"),
        "target_3": result.get("tp3"),
        "risk": result.get("risk"),
        "trade_stage": trade_eval.get("trade_stage"),
        "a_plus_ready": trade_eval.get("a_plus_ready"),
        "b_plus_tradeable": trade_eval.get("b_plus_tradeable"),
        "trigger_confirmed": trade_eval.get("trigger_confirmed"),
        "no_trade_reasons": trade_eval.get("no_trade_reasons"),
        "setup_status_reason": result.get("setupStatusReason"),
        "grade_reason": result.get("setupGradeReason"),
    }


def _compare_strategy_outputs(production: Optional[dict], hybrid: Optional[dict]) -> dict:
    fields = [
        "selected_timeframe",
        "setup_status",
        "trend",
        "direction",
        "setup_grade",
        "quality_grade",
        "quality_score",
        "confirmation_started",
        "entry_status",
        "entry",
        "stop",
        "target_1",
        "target_2",
        "target_3",
        "risk",
        "trade_stage",
        "a_plus_ready",
        "b_plus_tradeable",
        "trigger_confirmed",
    ]
    if production is None or hybrid is None:
        return {
            "differences": ["missing scanner output"],
            "material_differences": ["missing scanner output"],
        }
    differences = []
    material = []
    for field in fields:
        left = production.get(field)
        right = hybrid.get(field)
        if _values_equivalent(left, right):
            continue
        differences.append({
            "field": field,
            "production": left,
            "hybrid": right,
        })
        if field in {
            "selected_timeframe",
            "setup_status",
            "trend",
            "direction",
            "setup_grade",
            "quality_grade",
            "confirmation_started",
            "entry_status",
            "entry",
            "stop",
            "target_1",
            "target_2",
            "target_3",
            "risk",
            "trade_stage",
            "a_plus_ready",
            "b_plus_tradeable",
            "trigger_confirmed",
        }:
            material.append(field)
    return {
        "differences": differences,
        "material_differences": sorted(set(material)),
    }


def _values_equivalent(left, right) -> bool:
    if left == right:
        return True
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0, abs_tol=0.010000001)
    return False


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
        "requested_interval": interval,
        "provider_timeframe": _provider_timeframe(provider_name, interval),
        "bar_construction": "provider_native",
        "regular_session_filter": "provider_default",
        "recent_timestamps": _recent_timestamps(df),
        "recent_timestamps_et": _recent_timestamps_et(df),
        "columns": [] if df is None else [str(c) for c in df.columns],
    }


def _compare_completed_frames(yahoo_df: pd.DataFrame, alpaca_df: pd.DataFrame, interval: str) -> dict:
    if yahoo_df is None or alpaca_df is None or yahoo_df.empty or alpaca_df.empty:
        return {
            "completed_timestamp_matches": 0,
            "latest_timestamp_match": None,
            "latest_ohlcv_delta": None,
            "missing_in_alpaca": [],
            "missing_in_yahoo": [],
            "missing_in_alpaca_count": 0,
            "missing_in_yahoo_count": 0,
            "matched_latest_completed": None,
            "top_ohlc_differences": [],
        }
    yahoo_lookup = {_canonical_timestamp_key(i, interval): i for i in getattr(yahoo_df, "index", [])}
    alpaca_lookup = {_canonical_timestamp_key(i, interval): i for i in getattr(alpaca_df, "index", [])}
    yahoo_index = set(yahoo_lookup.keys())
    alpaca_index = set(alpaca_lookup.keys())
    common = sorted(yahoo_index & alpaca_index)
    latest_match = None
    latest_ohlcv_delta = None
    stats = _empty_comparison_stats()
    matched_latest = None
    if common:
        matched_latest = common[-1]
        y = yahoo_df.loc[yahoo_lookup[matched_latest]]
        a = alpaca_df.loc[alpaca_lookup[matched_latest]]
        if isinstance(y, pd.DataFrame):
            y = y.iloc[-1]
        if isinstance(a, pd.DataFrame):
            a = a.iloc[-1]
        latest_match = _canonical_timestamp_key(yahoo_df.index[-1], interval) == _canonical_timestamp_key(alpaca_df.index[-1], interval)
        latest_ohlcv_delta = _ohlcv_delta(y, a)
        stats = _comparison_stats(yahoo_df, alpaca_df, yahoo_lookup, alpaca_lookup, common)
    missing_in_alpaca = sorted(yahoo_index - alpaca_index)
    missing_in_yahoo = sorted(alpaca_index - yahoo_index)
    return {
        "candle_count_delta": int((0 if alpaca_df is None else len(alpaca_df)) - (0 if yahoo_df is None else len(yahoo_df))),
        "completed_timestamp_matches": len(common),
        "latest_timestamp_match": latest_match,
        "matched_latest_completed": matched_latest,
        "latest_ohlcv_delta": latest_ohlcv_delta,
        "stats": stats,
        "missing_in_alpaca": _sample_timestamps(missing_in_alpaca),
        "missing_in_yahoo": _sample_timestamps(missing_in_yahoo),
        "missing_in_alpaca_count": len(missing_in_alpaca),
        "missing_in_yahoo_count": len(missing_in_yahoo),
        "top_ohlc_differences": _top_ohlc_differences(yahoo_df, alpaca_df, yahoo_lookup, alpaca_lookup, common),
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


def _empty_comparison_stats() -> dict:
    return {
        "matched_candles": 0,
        "near_exact_price_matches": 0,
        "max_ohlc_percent_difference": None,
        "median_ohlc_percent_difference": None,
        "max_volume_percent_difference": None,
        "median_volume_percent_difference": None,
    }


def _comparison_stats(yahoo_df: pd.DataFrame, alpaca_df: pd.DataFrame, yahoo_lookup: dict, alpaca_lookup: dict, common: list[str]) -> dict:
    ohlc_pcts = []
    volume_pcts = []
    near_exact = 0
    for key in common:
        y = yahoo_df.loc[yahoo_lookup[key]]
        a = alpaca_df.loc[alpaca_lookup[key]]
        if isinstance(y, pd.DataFrame):
            y = y.iloc[-1]
        if isinstance(a, pd.DataFrame):
            a = a.iloc[-1]
        row_price_pcts = []
        for column in ["Open", "High", "Low", "Close"]:
            yv = _safe_float(y.get(column))
            av = _safe_float(a.get(column))
            if yv is None or av is None or yv == 0:
                continue
            pct = abs((av - yv) / yv * 100)
            ohlc_pcts.append(pct)
            row_price_pcts.append(pct)
        if row_price_pcts and max(row_price_pcts) <= 0.001:
            near_exact += 1
        yv = _safe_float(y.get("Volume"))
        av = _safe_float(a.get("Volume"))
        if yv is not None and av is not None and yv != 0:
            volume_pcts.append(abs((av - yv) / yv * 100))
    return {
        "matched_candles": len(common),
        "near_exact_price_matches": near_exact,
        "max_ohlc_percent_difference": _max_or_none(ohlc_pcts),
        "median_ohlc_percent_difference": _median_or_none(ohlc_pcts),
        "max_volume_percent_difference": _max_or_none(volume_pcts),
        "median_volume_percent_difference": _median_or_none(volume_pcts),
    }


def _top_ohlc_differences(
    yahoo_df: pd.DataFrame,
    alpaca_df: pd.DataFrame,
    yahoo_lookup: dict,
    alpaca_lookup: dict,
    common: list[str],
    limit: int = 8,
) -> list[dict]:
    differences = []
    for key in common:
        y = yahoo_df.loc[yahoo_lookup[key]]
        a = alpaca_df.loc[alpaca_lookup[key]]
        if isinstance(y, pd.DataFrame):
            y = y.iloc[-1]
        if isinstance(a, pd.DataFrame):
            a = a.iloc[-1]
        column_diffs = {}
        max_pct = 0.0
        max_column = None
        for column in ["Open", "High", "Low", "Close"]:
            yv = _safe_float(y.get(column))
            av = _safe_float(a.get(column))
            if yv is None or av is None or yv == 0:
                continue
            pct = abs((av - yv) / yv * 100)
            column_diffs[column.lower()] = {
                "yahoo": yv,
                "alpaca": av,
                "absolute": av - yv,
                "percent": pct,
            }
            if pct > max_pct:
                max_pct = pct
                max_column = column.lower()
        if max_column is None:
            continue
        differences.append({
            "timestamp": key,
            "max_field": max_column,
            "max_ohlc_percent_difference": max_pct,
            "yahoo_ohlcv": _ohlcv_dict(y),
            "alpaca_ohlcv": _ohlcv_dict(a),
            "ohlc_differences": column_diffs,
        })
    return sorted(differences, key=lambda row: row["max_ohlc_percent_difference"], reverse=True)[:limit]


def _max_or_none(values: list[float]) -> Optional[float]:
    return max(values) if values else None


def _median_or_none(values: list[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _canonical_timestamp_key(value, interval: str) -> str:
    ts = pd.Timestamp(value)
    interval = str(interval or "").lower()
    if ts.tzinfo is None:
        ts_et = ts.tz_localize(EASTERN_TZ)
    else:
        ts_et = ts.tz_convert(EASTERN_TZ)
    if interval in {"1d", "1wk"}:
        return ts_et.date().isoformat()
    return ts_et.isoformat()


def _sample_timestamps(values: list[str], limit: int = 12) -> list[str]:
    if len(values) <= limit:
        return values
    head = values[: limit // 2]
    tail = values[-(limit // 2):]
    return [*head, "...", *tail]


def _timestamp_to_iso(value) -> Optional[str]:
    if value is None:
        return None
    try:
        return pd.Timestamp(value).isoformat()
    except Exception:
        return str(value)


def _provider_timeframe(provider_name: str, interval: str) -> str:
    if provider_name == ALPACA_PROVIDER_NAME:
        return _alpaca_timeframe(interval)
    return interval


def _recent_timestamps(df: pd.DataFrame, limit: int = 10) -> list[str]:
    if df is None or df.empty:
        return []
    return [_timestamp_to_iso(value) for value in list(df.index)[-limit:]]


def _recent_timestamps_et(df: pd.DataFrame, limit: int = 10) -> list[str]:
    if df is None or df.empty:
        return []
    values = []
    for value in list(df.index)[-limit:]:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize(EASTERN_TZ)
        else:
            ts = ts.tz_convert(EASTERN_TZ)
        values.append(ts.isoformat())
    return values


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
