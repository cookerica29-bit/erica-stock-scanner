"""Kairos Watch Contract -- deterministic sensors (2026-09 session).

Pure, DataFrame-in/dict-out functions that power the live Watch Contract
monitor. Every detection primitive here is an EXISTING, already-tested
function from `scanner.py` / `displacement_score.py` -- nothing in this
module invents a new pattern-matching rule. What's genuinely new is:

  1. A STRONG-only displacement floor (raised from hybrid_shadow_v1's own
     MODERATE research default) for both the 30M confirmation layer and
     the 5M execution layer -- see the Sensor Calibration report this
     session produced: across 26 real cases, MODERATE was 0% clean/100%
     false-positive, STRONG was 0% false-positive. Both layers gate on it
     equally (NOT a CHoCH-only requirement -- the false positives in that
     study were an even BOS/CHoCH mix).
  2. `prior_touch_count`/`bars_since_first_pierce` diagnostics on every
     qualifying event -- LOG ONLY, never a blocker (a second, separate
     failure mode the calibration report found STRONG alone does not
     fix; no validated cutoff exists yet to gate on).
  3. `_resolve_pullback_reference`/`_windowed_clear_and_return` below are
     a deliberate, narrow COPY (not an import) of the already-audited
     functions of the same name in `hybrid_shadow_engine.py` -- per this
     session's explicit instruction, the live Watch Contract must not
     import or wire that module (a separate, feature-flagged RESEARCH
     engine that also autonomously infers HTF thesis/location, which the
     Watch Contract explicitly must never do). Copying just these two
     pullback functions (which take direction/zone-bounds as plain
     inputs and infer nothing) is the "narrow extraction" that session's
     audit specifically found safe to reuse. `min_execution_zone_atr`/
     `pullback_window_bars` are copied as plain module constants here
     for the same reason, not imported from `HybridShadowConfig`.

NEVER writes to any table, NEVER calls a live trade/alert function, NEVER
computes or infers HTF thesis/current-leg/location -- every function here
takes direction and (where relevant) a location/zone as a plain input it
trusts, exactly matching the Watch Contract's core boundary: Kairos
monitors an Erica-approved read, it does not form one.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

import scanner
from displacement_score import score_displacement

# ---------------------------------------------------------------------------
# Sensor policy constants -- the live Watch Contract's OWN explicit policy,
# deliberately separate from HYBRID_STRATEGY_CONFIGS (hybrid_shadow_v1_a/_b
# in hybrid_shadow_engine.py, both left byte-identical/untouched by this
# session -- see tests/watch_contract_v1.py's own config-untouched guard).
# ---------------------------------------------------------------------------

CONFIRMATION_TIMEFRAME = "30m"
EXECUTION_TIMEFRAME = "5m"
ALLOWED_CONFIRMATION_EVENTS = ("BOS", "CHoCH")
ALLOWED_EXECUTION_EVENTS = ("BOS", "CHoCH")
# Sensor Calibration report (2026-09 session): STRONG-only, for BOTH event
# types and BOTH layers -- the false positives that MODERATE let through
# were an even BOS/CHoCH mix (4 BOS, 5 CHoCH), not a CHoCH-specific defect,
# so the gate is not narrowed to one event type.
MIN_CONFIRMATION_DISPLACEMENT = "STRONG"
MIN_EXECUTION_DISPLACEMENT = "STRONG"

SWING_MARGIN_30M = 4
SWING_MARGIN_5M = 3
BOS_LOOKBACK_CONFIRMATION = 40
BOS_LOOKBACK_EXECUTION = 20
REJECTION_SWEEP_LOOKBACK = 12
REJECTION_LOOKBACK = 5

# Diagnostic-only lookback for prior_touch_count/bars_since_first_pierce --
# same window used in this session's own Sensor Calibration report, never
# a gate (see PULLBACK_REACHED handling in watch_contract_engine.py: these
# two fields are always computed and persisted, never blocking).
TOUCH_DIAGNOSTIC_LOOKBACK_BARS = 40

# Copied verbatim from hybrid_shadow_engine.HybridShadowConfig's own
# already-audited defaults (see this module's own docstring for why this
# is a copy, not an import).
PULLBACK_WINDOW_BARS = 60
PULLBACK_MIN_EXECUTION_ZONE_ATR = 0.15

_DISPLACEMENT_RANK = {"WEAK": 0, "MODERATE": 1, "STRONG": 2}


def _direction_upper(direction: str) -> str:
    d = str(direction or "").strip().upper()
    return d if d in ("LONG", "SHORT") else ""


def _direction_lower(direction: str) -> str:
    d = str(direction or "").strip().lower()
    return d if d in ("long", "short") else ""


def _displacement_meets_bar(label: Optional[str], minimum_label: str) -> bool:
    if label is None:
        return False
    return _DISPLACEMENT_RANK.get(label, -1) >= _DISPLACEMENT_RANK.get(minimum_label, 99)


def _is_fresher_than(bar_time: Optional[str], min_bar_time: Optional[str]) -> bool:
    """True iff `bar_time` is strictly after `min_bar_time`. Real datetime
    comparison via pandas.Timestamp, NOT a raw string comparison -- two
    ISO timestamps for the exact same instant can differ textually
    ("...Z" vs "...+00:00", scanner._timestamp_at's own output vs a
    caller-supplied timestamp built a different way), and a plain string
    `<=` would wrongly call an identical instant "fresher" or vice versa.
    Missing/unparseable input is never treated as fresh (fails closed)."""
    if bar_time is None or min_bar_time is None:
        return bar_time is not None and min_bar_time is None
    try:
        return pd.Timestamp(bar_time) > pd.Timestamp(min_bar_time)
    except (TypeError, ValueError):
        return False


def drop_forming_bar(df: pd.DataFrame, interval_minutes: int, now) -> pd.DataFrame:
    """A live provider fetch can return a still-forming final bar (the
    current, not-yet-closed period) -- structural-break detection must
    only ever look at COMPLETED bars, matching this codebase's own
    existing "completed candles only" discipline elsewhere (e.g.
    candidates_router._completed_rth_30m_bars). Drops the last row only
    if its own bar-start time plus the interval hasn't elapsed yet;
    otherwise returns df unchanged (never drops a genuinely closed bar)."""
    if df is None or len(df) == 0:
        return df
    last_time = df.index[-1]
    if hasattr(last_time, "to_pydatetime"):
        last_time = last_time.to_pydatetime()
    if getattr(last_time, "tzinfo", None) is None:
        return df  # can't safely compare naive/aware -- leave as-is rather than guess
    bar_end = last_time + pd.Timedelta(minutes=interval_minutes)
    if now < bar_end:
        return df.iloc[:-1]
    return df


def compute_touch_diagnostics(
    df: pd.DataFrame, direction_uc: str, level: float, qualifying_index: int,
    lookback_bars: int = TOUCH_DIAGNOSTIC_LOOKBACK_BARS,
) -> dict[str, Optional[int]]:
    """LOG-ONLY diagnostics (never a blocker -- see this module's own
    docstring and the Sensor Calibration report). prior_touch_count = how
    many of the `lookback_bars` bars before (and including) the qualifying
    bar already had a High/Low reaching past `level`; bars_since_first_pierce
    = how long ago the level was FIRST reached, relative to the qualifying
    bar. Both None if `level` or the window is unavailable."""
    if level is None or qualifying_index is None or qualifying_index < 0:
        return {"prior_touch_count": None, "bars_since_first_pierce": None}
    start = max(0, qualifying_index - lookback_bars)
    first_pierce_index = None
    touches = 0
    for i in range(start, qualifying_index + 1):
        bar = df.iloc[i]
        pierced = (float(bar["High"]) >= level) if direction_uc == "LONG" else (float(bar["Low"]) <= level)
        if pierced:
            touches += 1
            if first_pierce_index is None:
                first_pierce_index = i
    bars_since_first_pierce = (qualifying_index - first_pierce_index) if first_pierce_index is not None else None
    return {"prior_touch_count": touches, "bars_since_first_pierce": bars_since_first_pierce}


def _detect_structural_event(
    df: pd.DataFrame, direction_uc: str, allowed_events: tuple[str, ...],
    bos_lookback: int, swing_margin: int,
) -> tuple[Optional[str], Optional[float], Optional[int]]:
    """BOS-first, CHoCH-fallback -- same precedence hybrid_shadow_engine's
    own (already-audited) _evaluate_30m_confirmation/_evaluate_5m_execution
    use, reused here rather than invented fresh."""
    swings = scanner._find_swings(df, margin=swing_margin)
    if "BOS" in allowed_events:
        ok, level = scanner._detect_bos(df, swings, direction_uc, lookback=bos_lookback)
        if ok:
            idx = scanner._first_bos_close_index(df, swings, direction_uc, level, lookback=bos_lookback)
            if idx is not None:
                return "BOS", level, idx
    if "CHoCH" in allowed_events:
        suppress, _reason, bearish_lvl, bullish_lvl, choch_idx = scanner._detect_choch(swings, direction_uc)
        choch_ok = (
            (direction_uc == "LONG" and bullish_lvl is not None and not suppress)
            or (direction_uc == "SHORT" and bearish_lvl is not None and not suppress)
        )
        if choch_ok and isinstance(choch_idx, int) and choch_idx >= 0:
            level = bullish_lvl if direction_uc == "LONG" else bearish_lvl
            return "CHoCH", level, choch_idx
    return None, None, None


def evaluate_30m_confirmation(
    df_30m: pd.DataFrame, direction: str, min_bar_time: Optional[str],
) -> dict[str, Any]:
    """Fresh 30M BOS/CHoCH, STRONG displacement only, strictly AFTER
    `min_bar_time` (the contract's own location_reached_at) -- enforces
    the location < confirmation freshness ordering. `min_bar_time` may be
    None (no freshness floor yet, e.g. a contract that has never had a
    location touch persisted) -- callers must not invoke this before
    location_reached_at exists.
    """
    direction_uc = _direction_upper(direction)
    result: dict[str, Any] = {
        "confirmed": False, "event_type": None, "level": None, "bar_time": None,
        "prior_touch_count": None, "bars_since_first_pierce": None, "displacement_label": None,
        "reason": None,
    }
    if direction_uc not in ("LONG", "SHORT") or df_30m is None or len(df_30m) == 0:
        result["reason"] = "no direction or no 30M data"
        return result

    event_type, level, idx = _detect_structural_event(
        df_30m, direction_uc, ALLOWED_CONFIRMATION_EVENTS, BOS_LOOKBACK_CONFIRMATION, SWING_MARGIN_30M,
    )
    if event_type is None:
        result["reason"] = "no accepted 30M confirmation event (BOS/CHoCH) detected yet"
        return result

    bar_time = scanner._timestamp_at(df_30m, idx)
    if min_bar_time is not None and not _is_fresher_than(bar_time, min_bar_time):
        result["reason"] = f"most recent qualifying {event_type} ({bar_time}) is not fresher than the location touch ({min_bar_time})"
        return result

    direction_lc = _direction_lower(direction)
    displacement = score_displacement(df_30m, direction_lc, index=idx)
    label = displacement.get("label")
    if not _displacement_meets_bar(label, MIN_CONFIRMATION_DISPLACEMENT):
        result["reason"] = f"{event_type} detected but displacement label {label} does not clear the {MIN_CONFIRMATION_DISPLACEMENT} bar"
        result["displacement_label"] = label
        return result

    touches = compute_touch_diagnostics(df_30m, direction_uc, level, idx)
    result.update({
        "confirmed": True, "event_type": event_type, "level": level, "bar_time": bar_time,
        "displacement_label": label, "reason": f"{event_type} confirmed with {label} displacement",
        **touches,
    })
    return result


def evaluate_5m_execution(
    df_5m: pd.DataFrame, direction: str, min_bar_time: Optional[str],
) -> dict[str, Any]:
    """Fresh 5M BOS/CHoCH, STRONG displacement only, strictly AFTER
    `min_bar_time` (the contract's own pullback_reached_at) -- enforces
    the pullback < execution freshness ordering. Rejection is evaluated
    SEPARATELY by `evaluate_5m_rejection_diagnostic` below and never
    folded into this function's own confirmed/event_type result -- v1
    explicitly does not let rejection autonomously produce ENTRY_READY.
    """
    direction_uc = _direction_upper(direction)
    result: dict[str, Any] = {
        "confirmed": False, "event_type": None, "level": None, "bar_time": None,
        "prior_touch_count": None, "bars_since_first_pierce": None, "displacement_label": None,
        "reason": None,
    }
    if direction_uc not in ("LONG", "SHORT") or df_5m is None or len(df_5m) == 0:
        result["reason"] = "no direction or no 5M data"
        return result

    event_type, level, idx = _detect_structural_event(
        df_5m, direction_uc, ALLOWED_EXECUTION_EVENTS, BOS_LOOKBACK_EXECUTION, SWING_MARGIN_5M,
    )
    if event_type is None:
        result["reason"] = "no accepted 5M execution event (BOS/CHoCH) detected yet"
        return result

    bar_time = scanner._timestamp_at(df_5m, idx)
    if min_bar_time is not None and not _is_fresher_than(bar_time, min_bar_time):
        result["reason"] = f"most recent qualifying {event_type} ({bar_time}) is not fresher than the pullback ({min_bar_time})"
        return result

    direction_lc = _direction_lower(direction)
    displacement = score_displacement(df_5m, direction_lc, index=idx)
    label = displacement.get("label")
    if not _displacement_meets_bar(label, MIN_EXECUTION_DISPLACEMENT):
        result["reason"] = f"{event_type} detected but displacement label {label} does not clear the {MIN_EXECUTION_DISPLACEMENT} bar"
        result["displacement_label"] = label
        return result

    touches = compute_touch_diagnostics(df_5m, direction_uc, level, idx)
    result.update({
        "confirmed": True, "event_type": event_type, "level": level, "bar_time": bar_time,
        "displacement_label": label, "reason": f"{event_type} confirmed with {label} displacement",
        **touches,
    })
    return result


def _first_rejection_index(df: pd.DataFrame, direction_uc: str, sweep_level: Optional[float], lookback: int) -> Optional[int]:
    """Copied narrowly from hybrid_shadow_engine._first_rejection_index --
    built entirely from scanner.detect_liquidity_sweep/detect_rejection,
    no thesis/location inference, safe to copy for the same reason the
    pullback functions are (see this module's own docstring)."""
    if direction_uc not in ("LONG", "SHORT") or sweep_level is None or len(df) < 2:
        return None
    n = len(df)
    start = max(0, n - lookback)
    for i in range(start, n):
        high, low = float(df["High"].iloc[i]), float(df["Low"].iloc[i])
        open_, close = float(df["Open"].iloc[i]), float(df["Close"].iloc[i])
        body = abs(close - open_)
        candle_range = high - low
        if candle_range <= 0:
            continue
        upper_wick = high - max(open_, close)
        lower_wick = min(open_, close) - low
        if direction_uc == "LONG":
            closed_back_above = low < sweep_level and close > sweep_level
            lower_wick_failure = low < sweep_level and lower_wick >= max(body * 1.25, candle_range * 0.35)
            if closed_back_above or (lower_wick_failure and close > open_):
                return i
        else:
            closed_back_below = high > sweep_level and close < sweep_level
            upper_wick_failure = high > sweep_level and upper_wick >= max(body * 1.25, candle_range * 0.35)
            if closed_back_below or (upper_wick_failure and close < open_):
                return i
    return None


def evaluate_5m_rejection_diagnostic(df_5m: pd.DataFrame, direction: str, min_bar_time: Optional[str]) -> dict[str, Any]:
    """Rejection/sweep-and-reclaim, LOG ONLY -- per this sprint's explicit
    scope, a rejection event is persisted as diagnostic context but must
    NEVER independently advance PULLBACK_REACHED -> ENTRY_READY (that
    requires evaluate_5m_execution's own BOS/CHoCH result). Same freshness
    floor (must be after pullback_reached_at) applied for consistency,
    even though nothing currently acts on this result automatically."""
    direction_uc = _direction_upper(direction)
    result: dict[str, Any] = {"detected": False, "event_type": None, "level": None, "bar_time": None, "reason": None}
    if direction_uc not in ("LONG", "SHORT") or df_5m is None or len(df_5m) == 0:
        result["reason"] = "no direction or no 5M data"
        return result
    swings = scanner._find_swings(df_5m, margin=SWING_MARGIN_5M)
    swept, sweep_level = scanner.detect_liquidity_sweep(df_5m, swings, direction_uc, lookback=REJECTION_SWEEP_LOOKBACK)
    if not swept or not scanner.detect_rejection(df_5m, direction_uc, sweep_level, lookback=REJECTION_LOOKBACK):
        result["reason"] = "no rejection/sweep-and-reclaim detected"
        return result
    idx = _first_rejection_index(df_5m, direction_uc, sweep_level, REJECTION_LOOKBACK)
    if idx is None:
        result["reason"] = "no rejection/sweep-and-reclaim detected"
        return result
    bar_time = scanner._timestamp_at(df_5m, idx)
    if min_bar_time is not None and not _is_fresher_than(bar_time, min_bar_time):
        result["reason"] = f"rejection ({bar_time}) is not fresher than the pullback ({min_bar_time})"
        return result
    result.update({"detected": True, "event_type": "REJECTION", "level": sweep_level, "bar_time": bar_time,
                    "reason": "rejection/sweep-and-reclaim detected (diagnostic only -- does not produce ENTRY_READY)"})
    return result


# ---------------------------------------------------------------------------
# Pullback -- narrow copy of hybrid_shadow_engine._resolve_pullback_reference
# / _windowed_clear_and_return (see this module's own docstring for why a
# copy, not an import). Behavior is unchanged from the already-audited
# originals; only names/comments adapted to this module's own context.
#
# PARITY VERIFIED (2026-09 session, pre-deploy hardening pass): a copy can
# silently drift from its original during transcription -- see
# tests/watch_contract_pullback_parity_v1.py, which runs IDENTICAL inputs
# through both this copy AND the real, unmodified hybrid_shadow_engine
# functions (imported ONLY in that test file, never in production code)
# and asserts byte-identical outputs across clear-then-return, no-clear,
# clear-but-no-return, persisted arbitrary location bounds, and bounded-
# window expiry (on/just-past the boundary, both directions). Re-run that
# file after ever touching either copy of these two functions.
# ---------------------------------------------------------------------------

def resolve_pullback_reference(
    df_30m: pd.DataFrame, df_5m: Optional[pd.DataFrame], confirming_index: int,
    location: dict, direction_lc: str, unresolved: list[str],
) -> Optional[dict[str, Any]]:
    """Hierarchy: (1) the contract's own persisted location bounds, if
    supplied; (2) the 30M confirmation candle's own body; (3) the first
    materially-displaced 5M candle's body within the confirming 30M bar.
    Input `location` is a plain dict {"level_bounds": {"high","low"} |
    None, "level_id": ...} built by the caller from the PERSISTED
    contract -- this function infers no location itself."""
    if location.get("level_bounds") is not None:
        return {
            "high": location["level_bounds"]["high"], "low": location["level_bounds"]["low"],
            "reference_type": "location_level", "reference_provenance": location.get("level_id"),
        }

    open_ = float(df_30m["Open"].iloc[confirming_index])
    close = float(df_30m["Close"].iloc[confirming_index])
    body_high, body_low = max(open_, close), min(open_, close)
    atr_30m = scanner._compute_atr(df_30m) if len(df_30m) >= 15 else 0.0
    body_atr_multiple = ((body_high - body_low) / atr_30m) if atr_30m > 0 else None

    if body_atr_multiple is not None and body_atr_multiple >= PULLBACK_MIN_EXECUTION_ZONE_ATR:
        return {"high": body_high, "low": body_low, "reference_type": "confirmation_candle_body", "reference_provenance": None}

    if df_5m is None or len(df_5m) == 0:
        unresolved.append("pullback: 30M confirmation body too small and no 5M data available for the fallback reference")
        return None

    confirm_time = scanner._timestamp_at(df_30m, confirming_index)
    window = df_5m.loc[pd.Timestamp(confirm_time):] if confirm_time else df_5m
    if len(window) == 0:
        unresolved.append("pullback: no 5M bars available within the confirming 30M candle's own window")
        return None
    for i in range(len(window)):
        bar = window.iloc[i]
        o, h, l, c = float(bar["Open"]), float(bar["High"]), float(bar["Low"]), float(bar["Close"])
        rng = h - l
        if rng <= 0:
            continue
        directional = (c > o) if direction_lc == "long" else (c < o)
        atr_5m = scanner._compute_atr(window.iloc[: i + 1]) if i >= 14 else 0.0
        if directional and atr_5m > 0 and (abs(c - o) / atr_5m) >= PULLBACK_MIN_EXECUTION_ZONE_ATR:
            bh, bl = max(o, c), min(o, c)
            return {"high": bh, "low": bl, "reference_type": "5m_displacement_candle_body", "reference_provenance": None}
    unresolved.append("pullback: no materially-displaced 5M candle found within the confirming 30M bar to anchor a fallback reference")
    return None


def windowed_clear_and_return(
    df_30m: pd.DataFrame, entry_index: int, zone_low: float, zone_high: float, direction_uc: str,
    window_bars: int = PULLBACK_WINDOW_BARS,
) -> tuple[bool, Optional[int], Optional[int]]:
    """Bounded, episode-local pullback scan -- only ever looks at
    df_30m[entry_index+1 : entry_index+1+window_bars], never "anywhere in
    all of history since" (the smc_shadow_v1 whole-history sticky bug this
    was the fix for). Returns (found, clear_index, return_index)."""
    end = min(entry_index + 1 + window_bars, len(df_30m))
    cleared = False
    clear_index = None
    for i in range(entry_index + 1, end):
        low = float(df_30m["Low"].iloc[i])
        high = float(df_30m["High"].iloc[i])
        if direction_uc == "LONG":
            if not cleared:
                if low > zone_high:
                    cleared, clear_index = True, i
                continue
            if low <= zone_high and high >= zone_low:
                return True, clear_index, i
        else:
            if not cleared:
                if high < zone_low:
                    cleared, clear_index = True, i
                continue
            if high >= zone_low and low <= zone_high:
                return True, clear_index, i
    return False, clear_index, None


def is_invalidated(direction: str, current_price: Optional[float], rule: Optional[str], level: Optional[float]) -> bool:
    """The human-approved structural invalidation rule -- plain price-vs-
    level check, same close_above/close_below vocabulary already used
    throughout this codebase's other trigger/confirmation fields. Never
    infers a stop/invalidation level itself; `rule`/`level` are always
    supplied by the approved contract."""
    if current_price is None or level is None or rule not in ("close_above", "close_below"):
        return False
    if rule == "close_below":
        return current_price < level
    return current_price > level
