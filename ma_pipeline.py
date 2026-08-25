"""MA-based external scanner candidate generation for Kairos ingestion.

This module produces review candidates only. It does not open trades, promote
plans, or alter the legacy Kairos scanner cache.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
import time
from typing import Any

import pandas as pd

from market_data import (
    AlpacaMarketDataProvider,
    DEFAULT_ALPACA_REQUEST_PACING_SECONDS,
    alpaca_credentials_configured,
)


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


# Default chunk size is evidence-backed, not guessed: direct measurement against
# Alpaca's real /v2/stocks/bars pagination for a 60d/4h request (the interval this
# pipeline actually requests) showed page density of ~1.5-1.8 pages per symbol --
# 10-symbol chunks complete in 18 pages (comfortable margin under the 25-page
# ceiling), 12-symbol chunks in 20 (thin margin), and 15+-symbol chunks hit the
# ceiling outright and fail every time. The former default of 50 needed ~77 pages,
# more than 3x the ceiling -- that mismatch, not rate-limiting, was the root cause
# of most batch-level pagination failures during live scans.
def _chunk_size() -> int:
    try:
        value = int(os.getenv("MA_PIPELINE_ALPACA_CHUNK_SIZE", "10"))
    except ValueError:
        value = 10
    return min(max(value, 1), 100)


def _chunks(items: list[str], size: int):
    for index in range(0, len(items), size):
        yield items[index:index + size]


def _fallback_pacing_seconds() -> float:
    try:
        value = float(os.getenv("ALPACA_REQUEST_PACING_SECONDS", str(DEFAULT_ALPACA_REQUEST_PACING_SECONDS)))
    except (TypeError, ValueError):
        value = DEFAULT_ALPACA_REQUEST_PACING_SECONDS
    return max(0.0, value)


def _download_with_fallback(provider: AlpacaMarketDataProvider, symbols: list[str], period: str, interval: str) -> pd.DataFrame:
    frame = provider.download(symbols, period=period, interval=interval, auto_adjust=True)
    if frame is not None and not frame.empty:
        return frame
    if len(symbols) <= 1:
        return frame
    # Throttle between sequential single-symbol retries -- an unthrottled fallback
    # loop of this shape reproduced a genuine Alpaca HTTP 429 mid-burst; a clean,
    # isolated run at this same pace completed 250 consecutive single-symbol calls
    # with zero failures. See DEFAULT_ALPACA_REQUEST_PACING_SECONDS for the evidence.
    pacing_seconds = _fallback_pacing_seconds()
    frames = {}
    for index, symbol in enumerate(symbols):
        if index > 0 and pacing_seconds > 0:
            time.sleep(pacing_seconds)
        single = provider.download([symbol], period=period, interval=interval, auto_adjust=True)
        if single is not None and not single.empty:
            frames[symbol] = single
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1)


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

    candidates = []
    failures = 0
    provider = AlpacaMarketDataProvider()
    chunk_size = _chunk_size()
    for chunk in _chunks(requested, chunk_size):
        daily = _download_with_fallback(provider, chunk, period="1y", interval="1d")
        four_hour = _download_with_fallback(provider, chunk, period="60d", interval="4h")
        for symbol in chunk:
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
            "alpaca_chunk_size": chunk_size,
        },
    }
