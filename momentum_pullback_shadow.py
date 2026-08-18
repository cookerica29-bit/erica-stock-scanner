"""Momentum Pullback Shadow V1.0.

Shadow-only daily momentum pullback evaluator. This module intentionally does
not feed production scanner ranking, lifecycle, cards, alerts, journal, or
option selection.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


VERSION = "momentum-pullback-shadow-v1.0"

LONG = "LONG"
SHORT = "SHORT"

STATE_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
STATE_MACRO_REJECTED = "MACRO_REJECTED"
STATE_TREND_QUALIFIED = "TREND_QUALIFIED"
STATE_PULLBACK_WATCH = "PULLBACK_WATCH"
STATE_ARMED = "ARMED"
STATE_CONFIRMATION_WATCH = "CONFIRMATION_WATCH"
STATE_EXECUTION_CONFIRMED = "EXECUTION_CONFIRMED"
STATE_EXPIRED = "EXPIRED"
STATE_ENTRY_UNAVAILABLE = "ENTRY_UNAVAILABLE"

MIN_DAILY_BARS = 200
SMA_PERIOD = 200
EMA_PERIOD = 20
EMA_EXIT_PERIOD = 9
RSI_PERIOD = 14
ATR_PERIOD = 14
VOLUME_AVG_PERIOD = 20
RS_LOOKBACK_BARS = 126
PULLBACK_LOOKBACK_BARS = 10
RSI_RESET_LOOKBACK_BARS = 5
MAX_EMA_PENETRATION_ATR = 0.5
COHORT_SEED = 20260818
COHORT_SIZE = 50


@dataclass(frozen=True)
class MomentumSignal:
    symbol: str
    version: str
    direction: str
    state: str
    setup_id: str
    pullback_generation_id: str
    signal_generation_id: str
    signal_timestamp: str
    signal_close: float
    entry_timestamp: Optional[str]
    entry_price: Optional[float]
    entry_status: str
    sma200: float
    ema20: float
    ema9: float
    atr14: float
    rsi14: float
    prior_volume_avg20: float
    volume_ratio: float
    ema_interaction_timestamp: str
    ema_interaction_value: float
    ema_penetration_dollars: float
    ema_penetration_atr: float
    bars_since_ema_interaction: int
    stock_6m_return: Optional[float]
    spy_6m_return: Optional[float]
    relative_strength_excess: Optional[float]
    relative_strength_percentile: Optional[float]


def _iso(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def clean_daily_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError("daily frame missing columns: " + ", ".join(missing))
    out = df.loc[:, required].copy()
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["Open", "High", "Low", "Close"])


def historical_slice(df: pd.DataFrame, as_of: Any) -> pd.DataFrame:
    """Return candles observable at or before as_of."""
    clean = clean_daily_frame(df)
    ts = pd.Timestamp(as_of)
    return clean.loc[clean.index <= ts].copy()


def sma(series: pd.Series, period: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rolling(period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").ewm(span=period, adjust=False, min_periods=period).mean()


def atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")
    close = pd.to_numeric(df["Close"], errors="coerce")
    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period, min_periods=period).mean()


def rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    close = pd.to_numeric(series, errors="coerce")
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    values = 100 - (100 / (1 + rs))
    values = values.where(avg_loss != 0, 100.0)
    values = values.where(avg_gain != 0, 0.0)
    return values


def prior_average_volume(volume: pd.Series, period: int = VOLUME_AVG_PERIOD) -> pd.Series:
    return pd.to_numeric(volume, errors="coerce").shift(1).rolling(period, min_periods=period).mean()


def indicator_frame(df: pd.DataFrame, spy_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    out = clean_daily_frame(df)
    out["SMA200"] = sma(out["Close"], SMA_PERIOD)
    out["EMA20"] = ema(out["Close"], EMA_PERIOD)
    out["EMA9"] = ema(out["Close"], EMA_EXIT_PERIOD)
    out["RSI14"] = rsi(out["Close"], RSI_PERIOD)
    out["ATR14"] = atr(out, ATR_PERIOD)
    out["PRIOR_AVG_VOLUME20"] = prior_average_volume(out["Volume"], VOLUME_AVG_PERIOD)
    out["STOCK_6M_RETURN"] = out["Close"] / out["Close"].shift(RS_LOOKBACK_BARS) - 1.0
    out["SPY_6M_RETURN"] = np.nan
    if spy_df is not None and not spy_df.empty:
        spy = clean_daily_frame(spy_df)
        spy_return = spy["Close"] / spy["Close"].shift(RS_LOOKBACK_BARS) - 1.0
        out["SPY_6M_RETURN"] = spy_return.reindex(out.index)
    out["RELATIVE_STRENGTH_EXCESS"] = out["STOCK_6M_RETURN"] - out["SPY_6M_RETURN"]
    return out


def _macro_pass(row: pd.Series, direction: str) -> bool:
    close = _finite_float(row.get("Close"))
    sma200 = _finite_float(row.get("SMA200"))
    if close is None or sma200 is None:
        return False
    return close > sma200 if direction == LONG else close < sma200


def _ema_interaction_at(row: pd.Series, direction: str) -> Optional[Dict[str, Any]]:
    ema20 = _finite_float(row.get("EMA20"))
    atr14 = _finite_float(row.get("ATR14"))
    if ema20 is None or atr14 is None or atr14 <= 0:
        return None
    if direction == LONG:
        low = _finite_float(row.get("Low"))
        if low is None or low > ema20:
            return None
        penetration = max(0.0, ema20 - low)
    else:
        high = _finite_float(row.get("High"))
        if high is None or high < ema20:
            return None
        penetration = max(0.0, high - ema20)
    penetration_atr = penetration / atr14 if atr14 else math.inf
    if penetration_atr > MAX_EMA_PENETRATION_ATR:
        return None
    return {
        "timestamp": row.name,
        "ema20": ema20,
        "atr14": atr14,
        "penetration_dollars": penetration,
        "penetration_atr": penetration_atr,
    }


def _interaction_column(direction: str) -> str:
    return "_MP_LONG_EMA_INTERACTION" if direction == LONG else "_MP_SHORT_EMA_INTERACTION"


def _add_interaction_masks(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for direction in (LONG, SHORT):
        col = _interaction_column(direction)
        out[col] = [_ema_interaction_at(row, direction) is not None for _, row in out.iterrows()]
    return out


def _recent_ema_interaction(frame: pd.DataFrame, idx: int, direction: str) -> Optional[Dict[str, Any]]:
    start = max(0, idx - PULLBACK_LOOKBACK_BARS + 1)
    col = _interaction_column(direction)
    if col in frame.columns:
        mask = frame[col].iloc[start : idx + 1].to_numpy(dtype=bool)
        positions = np.flatnonzero(mask)
        if len(positions):
            pos = start + int(positions[-1])
            hit = _ema_interaction_at(frame.iloc[pos], direction)
            if hit:
                hit["bars_since"] = idx - pos
                return hit
        return None
    for pos in range(idx, start - 1, -1):
        hit = _ema_interaction_at(frame.iloc[pos], direction)
        if hit:
            hit["bars_since"] = idx - pos
            return hit
    return None


def _last_ema_interaction(frame: pd.DataFrame, idx: int, direction: str) -> Optional[Dict[str, Any]]:
    col = _interaction_column(direction)
    if col in frame.columns:
        mask = frame[col].iloc[: idx + 1].to_numpy(dtype=bool)
        positions = np.flatnonzero(mask)
        if len(positions):
            pos = int(positions[-1])
            hit = _ema_interaction_at(frame.iloc[pos], direction)
            if hit:
                hit["bars_since"] = idx - pos
                return hit
        return None
    for pos in range(idx, -1, -1):
        hit = _ema_interaction_at(frame.iloc[pos], direction)
        if hit:
            hit["bars_since"] = idx - pos
            return hit
    return None


def _rsi_reset_turn(frame: pd.DataFrame, idx: int, direction: str) -> Tuple[bool, Dict[str, Any]]:
    if idx <= 0:
        return False, {"reason": "no_previous_rsi"}
    current = _finite_float(frame["RSI14"].iloc[idx])
    previous = _finite_float(frame["RSI14"].iloc[idx - 1])
    if current is None or previous is None:
        return False, {"reason": "missing_rsi"}
    start = max(0, idx - RSI_RESET_LOOKBACK_BARS + 1)
    window = pd.to_numeric(frame["RSI14"].iloc[start : idx + 1], errors="coerce")
    if direction == LONG:
        entered = bool(((window >= 35.0) & (window <= 45.0)).any())
        turned = current > previous
    else:
        entered = bool(((window >= 55.0) & (window <= 65.0)).any())
        turned = current < previous
    return entered and turned, {
        "entered_reset_range": entered,
        "rsi_turned": turned,
        "rsi14": current,
        "rsi14_previous": previous,
    }


def _trigger_conditions(frame: pd.DataFrame, idx: int, direction: str) -> Tuple[bool, Dict[str, Any]]:
    if idx <= 0:
        return False, {"reason": "no_previous_bar"}
    row = frame.iloc[idx]
    prev = frame.iloc[idx - 1]
    open_ = _finite_float(row.get("Open"))
    close = _finite_float(row.get("Close"))
    volume = _finite_float(row.get("Volume"))
    prior_avg = _finite_float(row.get("PRIOR_AVG_VOLUME20"))
    if None in (open_, close, volume, prior_avg) or prior_avg <= 0:
        return False, {"reason": "missing_trigger_input"}
    if direction == LONG:
        candle_ok = close > open_
        break_ok = close > float(prev["High"])
    else:
        candle_ok = close < open_
        break_ok = close < float(prev["Low"])
    volume_ok = volume > prior_avg
    return candle_ok and break_ok and volume_ok, {
        "trigger_candle": candle_ok,
        "trigger_break": break_ok,
        "trigger_volume": volume_ok,
        "prior_volume_avg20": prior_avg,
        "volume_ratio": volume / prior_avg,
    }


def evaluate_indicator_frame_at(
    symbol: str,
    frame: pd.DataFrame,
    idx: int,
    direction: str,
) -> Tuple[str, Dict[str, Any]]:
    if idx < MIN_DAILY_BARS - 1:
        return STATE_INSUFFICIENT_DATA, {"required_bars": MIN_DAILY_BARS, "available_bars": idx + 1}
    row = frame.iloc[idx]
    if not _macro_pass(row, direction):
        return STATE_MACRO_REJECTED, {"macro_pass": False, "direction": direction}

    recent_interaction = _recent_ema_interaction(frame, idx, direction)
    last_interaction = _last_ema_interaction(frame, idx, direction)
    if not recent_interaction:
        if last_interaction and int(last_interaction["bars_since"]) >= PULLBACK_LOOKBACK_BARS:
            return STATE_EXPIRED, {
                "macro_pass": True,
                "expired_interaction": last_interaction,
                "bars_since_ema_interaction": int(last_interaction["bars_since"]),
            }
        ema20 = _finite_float(row.get("EMA20"))
        close = _finite_float(row.get("Close"))
        if ema20 is not None and close is not None:
            approaching = close >= ema20 if direction == LONG else close <= ema20
            if approaching:
                return STATE_PULLBACK_WATCH, {"macro_pass": True, "approaching_ema20": True}
        return STATE_TREND_QUALIFIED, {"macro_pass": True, "approaching_ema20": False}

    rsi_ok, rsi_details = _rsi_reset_turn(frame, idx, direction)
    trigger_ok, trigger_details = _trigger_conditions(frame, idx, direction)
    details: Dict[str, Any] = {
        "macro_pass": True,
        "ema_interaction": recent_interaction,
        "rsi": rsi_details,
        "trigger": trigger_details,
    }
    if rsi_ok and trigger_ok:
        return STATE_EXECUTION_CONFIRMED, details
    if rsi_ok:
        return STATE_CONFIRMATION_WATCH, details
    return STATE_ARMED, details


def evaluate_at(
    symbol: str,
    df: pd.DataFrame,
    idx: int,
    direction: Optional[str] = None,
    spy_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    frame = indicator_frame(df, spy_df)
    if direction in (LONG, SHORT):
        state, details = evaluate_indicator_frame_at(symbol, frame, idx, direction)
        return {"symbol": symbol, "direction": direction, "state": state, "details": details}
    long_state, long_details = evaluate_indicator_frame_at(symbol, frame, idx, LONG)
    short_state, short_details = evaluate_indicator_frame_at(symbol, frame, idx, SHORT)
    priority = [
        STATE_EXECUTION_CONFIRMED,
        STATE_CONFIRMATION_WATCH,
        STATE_ARMED,
        STATE_PULLBACK_WATCH,
        STATE_TREND_QUALIFIED,
        STATE_EXPIRED,
        STATE_MACRO_REJECTED,
        STATE_INSUFFICIENT_DATA,
    ]
    long_rank = priority.index(long_state)
    short_rank = priority.index(short_state)
    if long_rank <= short_rank:
        return {"symbol": symbol, "direction": LONG, "state": long_state, "details": long_details}
    return {"symbol": symbol, "direction": SHORT, "state": short_state, "details": short_details}


def setup_identity(symbol: str, direction: str, ema_touch_date: Any, signal_date: Any) -> Dict[str, str]:
    ema_date = _iso(pd.Timestamp(ema_touch_date).date())
    sig_date = _iso(pd.Timestamp(signal_date).date())
    pullback_generation_id = "|".join([symbol.upper(), direction, ema_date, VERSION])
    signal_generation_id = "|".join([pullback_generation_id, sig_date])
    setup_id = "|".join([symbol.upper(), direction, ema_date, sig_date, VERSION])
    return {
        "setup_id": setup_id,
        "pullback_generation_id": pullback_generation_id,
        "signal_generation_id": signal_generation_id,
    }


def _signal_from_evaluation(
    symbol: str,
    frame: pd.DataFrame,
    idx: int,
    direction: str,
    details: Dict[str, Any],
) -> MomentumSignal:
    row = frame.iloc[idx]
    interaction = details["ema_interaction"]
    ids = setup_identity(symbol, direction, interaction["timestamp"], row.name)
    next_idx = idx + 1
    if next_idx < len(frame):
        next_row = frame.iloc[next_idx]
        entry_timestamp: Optional[str] = _iso(next_row.name)
        entry_price: Optional[float] = float(next_row["Open"])
        entry_status = "NEXT_SESSION_OPEN"
    else:
        entry_timestamp = None
        entry_price = None
        entry_status = STATE_ENTRY_UNAVAILABLE
    return MomentumSignal(
        symbol=symbol.upper(),
        version=VERSION,
        direction=direction,
        state=STATE_EXECUTION_CONFIRMED,
        setup_id=ids["setup_id"],
        pullback_generation_id=ids["pullback_generation_id"],
        signal_generation_id=ids["signal_generation_id"],
        signal_timestamp=_iso(row.name),
        signal_close=float(row["Close"]),
        entry_timestamp=entry_timestamp,
        entry_price=entry_price,
        entry_status=entry_status,
        sma200=float(row["SMA200"]),
        ema20=float(row["EMA20"]),
        ema9=float(row["EMA9"]),
        atr14=float(row["ATR14"]),
        rsi14=float(row["RSI14"]),
        prior_volume_avg20=float(row["PRIOR_AVG_VOLUME20"]),
        volume_ratio=float(details["trigger"]["volume_ratio"]),
        ema_interaction_timestamp=_iso(interaction["timestamp"]),
        ema_interaction_value=float(interaction["ema20"]),
        ema_penetration_dollars=float(interaction["penetration_dollars"]),
        ema_penetration_atr=float(interaction["penetration_atr"]),
        bars_since_ema_interaction=int(interaction["bars_since"]),
        stock_6m_return=_finite_float(row.get("STOCK_6M_RETURN")),
        spy_6m_return=_finite_float(row.get("SPY_6M_RETURN")),
        relative_strength_excess=_finite_float(row.get("RELATIVE_STRENGTH_EXCESS")),
        relative_strength_percentile=None,
    )


def replay_symbol(symbol: str, df: pd.DataFrame, spy_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    frame = _add_interaction_masks(indicator_frame(df, spy_df))
    raw_signals: List[MomentumSignal] = []
    signals: List[MomentumSignal] = []
    duplicate_signals: List[MomentumSignal] = []
    states: Dict[str, int] = {}
    seen_pullbacks: set = set()
    for idx in range(len(frame)):
        for direction in (LONG, SHORT):
            state, details = evaluate_indicator_frame_at(symbol, frame, idx, direction)
            states[state] = states.get(state, 0) + 1
            if state != STATE_EXECUTION_CONFIRMED:
                continue
            signal = _signal_from_evaluation(symbol, frame, idx, direction, details)
            raw_signals.append(signal)
            if signal.pullback_generation_id in seen_pullbacks:
                duplicate_signals.append(signal)
                continue
            seen_pullbacks.add(signal.pullback_generation_id)
            signals.append(signal)
    return {
        "symbol": symbol.upper(),
        "version": VERSION,
        "bars": len(frame),
        "start": _iso(frame.index[0]) if len(frame) else None,
        "end": _iso(frame.index[-1]) if len(frame) else None,
        "sufficient_200_day_history": len(frame) >= MIN_DAILY_BARS,
        "raw_signals": [asdict(sig) for sig in raw_signals],
        "signals": [asdict(sig) for sig in signals],
        "duplicate_signals": [asdict(sig) for sig in duplicate_signals],
        "state_counts": states,
        "duplicate_suppression_count": len(duplicate_signals),
    }


def dedupe_independent_signals(signals: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    independent: List[Dict[str, Any]] = []
    duplicates: List[Dict[str, Any]] = []
    seen: set = set()
    for signal in sorted(signals, key=lambda item: (str(item.get("signal_timestamp")), str(item.get("setup_id")))):
        key = signal.get("pullback_generation_id") or signal.get("setup_id")
        if key in seen:
            duplicates.append(signal)
            continue
        seen.add(key)
        independent.append(signal)
    return independent, duplicates


def measure_underlying_outcome(
    signal: Dict[str, Any],
    df: pd.DataFrame,
    max_forward_days: int = 20,
    cleaned: bool = False,
) -> Dict[str, Any]:
    frame = df if cleaned else clean_daily_frame(df)
    entry_ts = signal.get("entry_timestamp")
    entry_price = _finite_float(signal.get("entry_price"))
    atr14 = _finite_float(signal.get("atr14"))
    if not entry_ts or entry_price is None or atr14 is None or atr14 <= 0:
        return {"status": "ENTRY_UNAVAILABLE"}
    index = frame.index
    matches = np.where(index == pd.Timestamp(entry_ts))[0]
    if len(matches) == 0:
        return {"status": "ENTRY_BAR_NOT_FOUND"}
    entry_idx = int(matches[0])
    direction = signal["direction"]
    direction_mult = 1.0 if direction == LONG else -1.0
    future = frame.iloc[entry_idx : min(len(frame), entry_idx + max_forward_days + 1)].copy()
    if future.empty:
        return {"status": "NO_FORWARD_BARS"}
    high_move = (future["High"] - entry_price) * direction_mult
    low_move = (future["Low"] - entry_price) * direction_mult
    if direction == LONG:
        favorable = future["High"] - entry_price
        adverse = future["Low"] - entry_price
    else:
        favorable = entry_price - future["Low"]
        adverse = entry_price - future["High"]
    mfe_pct = float((favorable.max() / entry_price) * 100.0)
    mae_pct = float((adverse.min() / entry_price) * 100.0)
    mfe_atr = float(favorable.max() / atr14)
    mae_atr = float(adverse.min() / atr14)

    def _close_return(days: int) -> Optional[float]:
        target_idx = entry_idx + days
        if target_idx >= len(frame):
            return None
        close = float(frame["Close"].iloc[target_idx])
        return float(((close - entry_price) / entry_price) * direction_mult * 100.0)

    def _threshold_before(fav_atr: float, adv_atr: float) -> Optional[bool]:
        fav_hit_idx = None
        adv_hit_idx = None
        for offset, (_, bar) in enumerate(future.iterrows()):
            if direction == LONG:
                fav_hit = float(bar["High"]) >= entry_price + fav_atr * atr14
                adv_hit = float(bar["Low"]) <= entry_price - adv_atr * atr14
            else:
                fav_hit = float(bar["Low"]) <= entry_price - fav_atr * atr14
                adv_hit = float(bar["High"]) >= entry_price + adv_atr * atr14
            if fav_hit and fav_hit_idx is None:
                fav_hit_idx = offset
            if adv_hit and adv_hit_idx is None:
                adv_hit_idx = offset
            if fav_hit_idx is not None or adv_hit_idx is not None:
                break
        if fav_hit_idx is None and adv_hit_idx is None:
            return None
        if fav_hit_idx is None:
            return False
        if adv_hit_idx is None:
            return True
        return fav_hit_idx <= adv_hit_idx

    def _pct_threshold_before(fav_pct: float, adv_pct: float) -> Optional[bool]:
        fav_hit_idx = None
        adv_hit_idx = None
        fav_price = entry_price * (1.0 + (fav_pct / 100.0) * direction_mult)
        adv_price = entry_price * (1.0 - (adv_pct / 100.0) * direction_mult)
        for offset, (_, bar) in enumerate(future.iterrows()):
            if direction == LONG:
                fav_hit = float(bar["High"]) >= fav_price
                adv_hit = float(bar["Low"]) <= adv_price
            else:
                fav_hit = float(bar["Low"]) <= fav_price
                adv_hit = float(bar["High"]) >= adv_price
            if fav_hit and fav_hit_idx is None:
                fav_hit_idx = offset
            if adv_hit and adv_hit_idx is None:
                adv_hit_idx = offset
            if fav_hit_idx is not None or adv_hit_idx is not None:
                break
        if fav_hit_idx is None and adv_hit_idx is None:
            return None
        if fav_hit_idx is None:
            return False
        if adv_hit_idx is None:
            return True
        return fav_hit_idx <= adv_hit_idx

    first5 = frame.iloc[entry_idx : min(len(frame), entry_idx + 6)]
    if direction == LONG:
        first5_fav = (first5["High"] - entry_price).max() / atr14
        first5_adv = (first5["Low"] - entry_price).min() / atr14
    else:
        first5_fav = (entry_price - first5["Low"]).max() / atr14
        first5_adv = (entry_price - first5["High"]).min() / atr14

    return {
        "status": "OK",
        "entry_timestamp": entry_ts,
        "entry_price": entry_price,
        "mfe_pct": mfe_pct,
        "mae_pct": mae_pct,
        "mfe_atr": mfe_atr,
        "mae_atr": mae_atr,
        "forward_return_1d_pct": _close_return(1),
        "forward_return_3d_pct": _close_return(3),
        "forward_return_5d_pct": _close_return(5),
        "forward_return_10d_pct": _close_return(10),
        "forward_return_20d_pct": _close_return(20),
        "one_atr_before_minus_one_atr": _threshold_before(1.0, 1.0),
        "two_atr_before_minus_one_atr": _threshold_before(2.0, 1.0),
        "three_atr_before_minus_one_atr": _threshold_before(3.0, 1.0),
        "plus_2pct_before_minus_2pct": _pct_threshold_before(2.0, 2.0),
        "plus_5pct_before_minus_5pct": _pct_threshold_before(5.0, 5.0),
        "plus_10pct_before_minus_10pct": _pct_threshold_before(10.0, 10.0),
        "first5_mfe_atr": float(first5_fav),
        "first5_mae_atr": float(first5_adv),
        "first5_plus_0_5_atr": bool(first5_fav >= 0.5),
        "first5_plus_1_atr": bool(first5_fav >= 1.0),
    }


def build_cohorts(symbols: Sequence[str], seed: int = COHORT_SEED, size: int = COHORT_SIZE) -> Dict[str, List[str]]:
    unique = sorted({str(symbol).upper() for symbol in symbols if str(symbol).strip()})
    first50 = unique[:size]
    remaining = [symbol for symbol in unique if symbol not in set(first50)]
    rng = random.Random(seed)
    shuffled = remaining[:]
    rng.shuffle(shuffled)
    cohorts = {"first50": first50}
    for idx in range(3):
        start = idx * size
        cohorts[f"random{idx + 1}"] = sorted(shuffled[start : start + size])
    return cohorts


def assign_relative_strength_percentiles(
    signals: Sequence[Dict[str, Any]],
    candle_data: Dict[str, pd.DataFrame],
    spy_df: Optional[pd.DataFrame],
) -> List[Dict[str, Any]]:
    by_date = sorted({pd.Timestamp(sig["signal_timestamp"]) for sig in signals if sig.get("signal_timestamp")})
    universe_frames: Dict[str, pd.DataFrame] = {}
    for symbol, df in candle_data.items():
        try:
            universe_frames[symbol.upper()] = indicator_frame(df, spy_df)
        except Exception:
            continue
    ranks: Dict[Tuple[str, str], Optional[float]] = {}
    for ts in by_date:
        values: List[Tuple[str, float]] = []
        for symbol, frame in universe_frames.items():
            if ts not in frame.index:
                continue
            value = _finite_float(frame.loc[ts].get("RELATIVE_STRENGTH_EXCESS"))
            if value is not None:
                values.append((symbol, value))
        if not values:
            continue
        ordered = sorted(values, key=lambda item: item[1])
        denom = max(1, len(ordered) - 1)
        for pos, (symbol, _) in enumerate(ordered):
            ranks[(symbol, ts.isoformat())] = pos / denom if denom else 1.0
    enriched: List[Dict[str, Any]] = []
    for signal in signals:
        item = dict(signal)
        ts = pd.Timestamp(item["signal_timestamp"]).isoformat()
        item["relative_strength_percentile"] = ranks.get((str(item["symbol"]).upper(), ts))
        enriched.append(item)
    return enriched


def replay_universe(
    candle_data: Dict[str, pd.DataFrame],
    spy_df: Optional[pd.DataFrame] = None,
    cohorts: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    symbol_reports: List[Dict[str, Any]] = []
    raw_signals: List[Dict[str, Any]] = []
    failures: Dict[str, str] = {}
    for symbol in sorted(candle_data):
        try:
            report = replay_symbol(symbol, candle_data[symbol], spy_df)
            symbol_reports.append(report)
            raw_signals.extend(report["raw_signals"])
        except Exception as exc:
            failures[symbol.upper()] = str(exc)
    raw_signals = assign_relative_strength_percentiles(raw_signals, candle_data, spy_df)
    independent, duplicates = dedupe_independent_signals(raw_signals)
    outcomes: List[Dict[str, Any]] = []
    clean_frames: Dict[str, pd.DataFrame] = {}
    for symbol, df in candle_data.items():
        try:
            clean_frames[symbol.upper()] = clean_daily_frame(df)
        except Exception:
            continue
    for signal in independent:
        symbol = signal["symbol"]
        outcome_frame = clean_frames.get(symbol)
        if outcome_frame is None:
            outcome = {"status": "CLEAN_FRAME_UNAVAILABLE"}
        else:
            outcome = measure_underlying_outcome(signal, outcome_frame, cleaned=True)
        outcomes.append({"signal": signal, "outcome": outcome})
    symbols = sorted({symbol.upper() for symbol in candle_data})
    cohort_membership = cohorts or build_cohorts(symbols)
    return {
        "version": VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cohort_seed": COHORT_SEED,
        "cohorts": cohort_membership,
        "symbols_attempted": len(symbols),
        "symbols_failed": failures,
        "symbol_reports": symbol_reports,
        "signals": raw_signals,
        "independent_signals": independent,
        "duplicate_signals": duplicates,
        "outcomes": outcomes,
    }


def _median(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return None
    return float(np.median(clean))


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return None
    return float(np.mean(clean))


def _rate(values: Iterable[Optional[bool]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return sum(1 for value in clean if value) / len(clean)


def summarize_outcomes(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [row for row in rows if row.get("outcome", {}).get("status") == "OK"]
    by_direction: Dict[str, Dict[str, Any]] = {}
    for direction in (LONG, SHORT):
        subset = [row for row in ok if row["signal"]["direction"] == direction]
        by_direction[direction] = _summary_block(subset)
    return {
        "event_count": len(rows),
        "outcome_count": len(ok),
        "all": _summary_block(ok),
        "by_direction": by_direction,
    }


def _summary_block(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    outcomes = [row["outcome"] for row in rows]
    signals = [row["signal"] for row in rows]
    return {
        "n": len(rows),
        "one_atr_before_minus_one_atr": _rate(o.get("one_atr_before_minus_one_atr") for o in outcomes),
        "two_atr_before_minus_one_atr": _rate(o.get("two_atr_before_minus_one_atr") for o in outcomes),
        "three_atr_before_minus_one_atr": _rate(o.get("three_atr_before_minus_one_atr") for o in outcomes),
        "median_mfe_atr": _median(o.get("mfe_atr") for o in outcomes),
        "median_mae_atr": _median(o.get("mae_atr") for o in outcomes),
        "median_forward_return_1d_pct": _median(o.get("forward_return_1d_pct") for o in outcomes),
        "median_forward_return_3d_pct": _median(o.get("forward_return_3d_pct") for o in outcomes),
        "median_forward_return_5d_pct": _median(o.get("forward_return_5d_pct") for o in outcomes),
        "median_forward_return_10d_pct": _median(o.get("forward_return_10d_pct") for o in outcomes),
        "median_forward_return_20d_pct": _median(o.get("forward_return_20d_pct") for o in outcomes),
        "median_rsi14": _median(s.get("rsi14") for s in signals),
        "median_volume_ratio": _median(s.get("volume_ratio") for s in signals),
        "median_ema_penetration_atr": _median(s.get("ema_penetration_atr") for s in signals),
        "median_pullback_age_bars": _median(s.get("bars_since_ema_interaction") for s in signals),
        "median_relative_strength_excess": _median(s.get("relative_strength_excess") for s in signals),
        "median_relative_strength_percentile": _median(s.get("relative_strength_percentile") for s in signals),
        "median_first5_mfe_atr": _median(o.get("first5_mfe_atr") for o in outcomes),
        "median_first5_mae_atr": _median(o.get("first5_mae_atr") for o in outcomes),
        "first5_plus_0_5_atr_rate": _rate(o.get("first5_plus_0_5_atr") for o in outcomes),
        "first5_plus_1_atr_rate": _rate(o.get("first5_plus_1_atr") for o in outcomes),
    }


def cohort_reports(report: Dict[str, Any]) -> Dict[str, Any]:
    outcomes = report.get("outcomes", [])
    cohorts = report.get("cohorts", {})
    output: Dict[str, Any] = {}
    for name, symbols in cohorts.items():
        symbol_set = set(symbols)
        rows = [row for row in outcomes if row.get("signal", {}).get("symbol") in symbol_set]
        output[name] = {
            "symbols": symbols,
            "summary": summarize_outcomes(rows),
        }
    return output


def data_completeness_report(report: Dict[str, Any]) -> Dict[str, Any]:
    symbol_reports = report.get("symbol_reports", [])
    sufficient = [row for row in symbol_reports if row.get("sufficient_200_day_history")]
    partial = [row for row in symbol_reports if not row.get("sufficient_200_day_history")]
    return {
        "symbols_attempted": report.get("symbols_attempted", 0),
        "symbols_complete_200_day_history": len(sufficient),
        "symbols_partial_history": len(partial),
        "symbols_failed": len(report.get("symbols_failed", {})),
        "failures": report.get("symbols_failed", {}),
        "date_ranges": {
            row["symbol"]: {"start": row.get("start"), "end": row.get("end"), "bars": row.get("bars")}
            for row in symbol_reports
        },
    }


def immutable_report_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def persist_report(path: str, payload: Dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    enriched = dict(payload)
    enriched["artifact_sha256"] = immutable_report_hash(payload)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(enriched, handle, indent=2, sort_keys=True, default=_json_default)
    os.replace(tmp_path, path)
    return path
