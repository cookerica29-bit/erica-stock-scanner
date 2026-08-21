"""MA-based external scanner candidate generation for Kairos ingestion.

This module produces review candidates only. It does not open trades, promote
plans, or alter the legacy Kairos scanner cache.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from market_data import AlpacaMarketDataProvider, alpaca_credentials_configured


MA_PIPELINE_VERSION = "ma-pipeline-alpaca-v1"
MA_PIPELINE_SOURCE = "ma_pipeline"


def _frame_for_symbol(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    if isinstance(frame.columns, pd.MultiIndex):
        if symbol not in frame.columns.get_level_values(0):
            return pd.DataFrame()
        return frame[symbol].dropna()
    return frame.dropna()


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _candidate_from_frames(symbol: str, daily: pd.DataFrame, four_hour: pd.DataFrame) -> dict[str, Any] | None:
    if daily is None or four_hour is None or len(daily) < 220 or len(four_hour) < 30:
        return None
    daily = daily.dropna()
    four_hour = four_hour.dropna()
    if daily.empty or four_hour.empty:
        return None

    close_daily = daily["Close"].astype(float)
    close_4h = four_hour["Close"].astype(float)
    sma50 = float(close_daily.rolling(50).mean().iloc[-1])
    sma200 = float(close_daily.rolling(200).mean().iloc[-1])
    latest_daily = float(close_daily.iloc[-1])
    ema21_4h = float(_ema(close_4h, 21).iloc[-1])
    latest_4h = float(close_4h.iloc[-1])
    if not all(value > 0 for value in [sma50, sma200, latest_daily, ema21_4h, latest_4h]):
        return None

    signal = None
    if latest_daily > sma50 > sma200:
        signal = "long"
    elif latest_daily < sma50 < sma200:
        signal = "short"
    if signal is None:
        return None

    ema_distance_pct = abs(latest_4h - ema21_4h) / latest_4h
    if ema_distance_pct > 0.03:
        return None

    confidence = "high" if (signal == "long" and latest_daily > sma200 and latest_4h >= ema21_4h) or (signal == "short" and latest_daily < sma200 and latest_4h <= ema21_4h) else "medium"
    return {
        "ticker": symbol,
        "signal": signal,
        "entry_price": round(latest_4h, 4),
        "ema21_4h": round(ema21_4h, 4),
        "daily_regime": signal,
        "confidence": confidence,
        "sma50_daily": round(sma50, 4),
        "sma200_daily": round(sma200, 4),
    }


def scan_ma_pipeline_candidates(symbols: list[str], max_symbols: int | None = None) -> dict[str, Any]:
    requested = [str(symbol or "").strip().upper() for symbol in symbols if str(symbol or "").strip()]
    requested = list(dict.fromkeys(requested))
    if max_symbols is not None:
        requested = requested[: max(0, int(max_symbols))]
    if not requested:
        return {"version": MA_PIPELINE_VERSION, "source": MA_PIPELINE_SOURCE, "candidates": [], "meta": {"requested": 0}}
    if not alpaca_credentials_configured():
        raise RuntimeError("Alpaca credentials are not configured")

    provider = AlpacaMarketDataProvider()
    daily = provider.download(requested, period="1y", interval="1d", auto_adjust=True)
    four_hour = provider.download(requested, period="60d", interval="4h", auto_adjust=True)
    candidates = []
    failures = 0
    for symbol in requested:
        candidate = _candidate_from_frames(
            symbol,
            _frame_for_symbol(daily, symbol),
            _frame_for_symbol(four_hour, symbol),
        )
        if candidate:
            candidates.append(candidate)
        else:
            failures += 1

    return {
        "version": MA_PIPELINE_VERSION,
        "source": MA_PIPELINE_SOURCE,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "candidates": candidates,
        "meta": {
            "requested": len(requested),
            "candidate_count": len(candidates),
            "non_candidates_or_missing_data": failures,
            "daily_period": "1y",
            "entry_timeframe": "4h",
            "regime": "Daily 50/200 SMA",
            "entry_pullback": "4H EMA21 proximity within 3%",
        },
    }
