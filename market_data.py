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
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yahoo_finance

logger = logging.getLogger(__name__)


YAHOO_PROVIDER_NAME = "yahoo"
ALPACA_PROVIDER_NAME = "alpaca"
DEFAULT_DATA_PROVIDER = YAHOO_PROVIDER_NAME
DEFAULT_ALPACA_DATA_BASE_URL = "https://data.alpaca.markets"


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
    symbol = str(ticker or "").strip().upper()
    yahoo_provider = YahooMarketDataProvider()
    alpaca_provider = AlpacaMarketDataProvider()

    yahoo_df = _normalize_ohlcv(yahoo_provider.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True))
    alpaca_df = _normalize_ohlcv(alpaca_provider.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True))
    return {
        "ticker": symbol,
        "period": period,
        "interval": interval,
        "production_provider": configured_provider_name(),
        "providers": {
            "yahoo": _frame_diagnostics(yahoo_df, symbol, yahoo_provider.normalize_symbol(symbol)),
            "alpaca": _frame_diagnostics(alpaca_df, symbol, alpaca_provider.normalize_symbol(symbol)),
        },
        "comparison": _compare_frames(yahoo_df, alpaca_df),
    }


def _frame_diagnostics(df: pd.DataFrame, requested_symbol: str, provider_symbol: str) -> dict:
    latest = None
    latest_ohlcv = None
    if df is not None and not df.empty:
        last = df.iloc[-1]
        latest = df.index[-1].isoformat() if hasattr(df.index[-1], "isoformat") else str(df.index[-1])
        latest_ohlcv = {
            "open": _safe_float(last.get("Open")),
            "high": _safe_float(last.get("High")),
            "low": _safe_float(last.get("Low")),
            "close": _safe_float(last.get("Close")),
            "volume": _safe_float(last.get("Volume")),
        }
    return {
        "requested_symbol": requested_symbol,
        "provider_symbol": provider_symbol,
        "candle_count": 0 if df is None else int(len(df)),
        "latest_completed_timestamp": latest,
        "latest_ohlcv": latest_ohlcv,
        "columns": [] if df is None else [str(c) for c in df.columns],
    }


def _compare_frames(yahoo_df: pd.DataFrame, alpaca_df: pd.DataFrame) -> dict:
    yahoo_index = set(str(i) for i in getattr(yahoo_df, "index", []))
    alpaca_index = set(str(i) for i in getattr(alpaca_df, "index", []))
    latest_match = None
    latest_ohlcv_delta = None
    if yahoo_df is not None and alpaca_df is not None and not yahoo_df.empty and not alpaca_df.empty:
        latest_match = str(yahoo_df.index[-1]) == str(alpaca_df.index[-1])
        y = yahoo_df.iloc[-1]
        a = alpaca_df.iloc[-1]
        latest_ohlcv_delta = {
            key.lower(): _safe_float(a.get(key)) - _safe_float(y.get(key))
            for key in ["Open", "High", "Low", "Close", "Volume"]
            if _safe_float(a.get(key)) is not None and _safe_float(y.get(key)) is not None
        }
    return {
        "candle_count_delta": int((0 if alpaca_df is None else len(alpaca_df)) - (0 if yahoo_df is None else len(yahoo_df))),
        "latest_timestamp_match": latest_match,
        "latest_ohlcv_delta": latest_ohlcv_delta,
        "missing_in_alpaca": sorted(yahoo_index - alpaca_index)[-10:],
        "missing_in_yahoo": sorted(alpaca_index - yahoo_index)[-10:],
    }


def _safe_float(value) -> Optional[float]:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None
