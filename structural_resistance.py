"""Structural resistance/support check for candidate target validation.

Flags a computed target (the output of candidates_router._nearest_structural_target)
when it lands near either:
  - a gap-day extreme: a bar that gapped hard and reversed -- a spike/shock
    level, not organic structure (news gaps, earnings surprises); or
  - a genuine swing pivot from scanner._find_swings -- real multi-bar
    structure, upgraded to "prior rejection" if price already moved away
    hard from it or came back and retested it.

When a target lands near such a level, clamp_target() pulls it back to just
this side of the nearest one -- the first real obstacle in price's path
toward the raw target, not necessarily the "strongest" evidence found nearby
-- and recomputes R:R off the adjusted value, while always preserving the
raw/unclamped target and R:R for the audit trail.

Column convention: dataframes here use the same capitalized OHLC columns
(Open/High/Low/Close) as the rest of this codebase (see scanner._find_swings,
scanner._compute_atr), and a plain positional (0..n-1) index, matching the
`index` field scanner._find_swings already returns for each swing.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

DEFAULT_GAP_PCT_THRESHOLD = 0.03
DEFAULT_REVERSAL_FRAC = 0.5
DEFAULT_FOLLOW_THROUGH_BARS = 3
DEFAULT_PROXIMITY_ATR = 0.3
DEFAULT_RETEST_ATR_FRAC = 0.15
DEFAULT_CLAMP_BUFFER_ATR = 0.1
# Minimum distance an order-block-derived stop must sit from entry, in ATR
# units, to be trusted as a real invalidation level rather than noise. See
# resolve_stop()'s docstring for the real-data case (AMZN) that motivated
# this. The cutoff is inclusive: exactly this distance away still passes.
MIN_STOP_DISTANCE_ATR = 0.25

_STRENGTH_RANK = {"strong": 3, "moderate": 2, "weak": 1}


def detect_gap_days(
    df: pd.DataFrame,
    gap_pct_threshold: float = DEFAULT_GAP_PCT_THRESHOLD,
    reversal_frac: float = DEFAULT_REVERSAL_FRAC,
    follow_through_bars: int = DEFAULT_FOLLOW_THROUGH_BARS,
) -> pd.DataFrame:
    """Flags rows where Open gapped >gap_pct_threshold from prior Close, AND
    price reversed -- either same-day (intraday round trip) or within
    follow_through_bars sessions (spike high on day 1, given back over the
    next 1-3 days). Either counts as unreliable structure: the level was
    created by a shock, not organic trading.

    Expects df indexed 0..n-1 (positional) with Open/High/Low/Close columns.
    Returns a copy with added columns: gap_pct, is_gap_day, is_gap_reversal.
    """
    df = df.reset_index(drop=True).copy()
    df["prior_close"] = df["Close"].shift(1)
    df["gap_pct"] = (df["Open"] - df["prior_close"]) / df["prior_close"]
    df["day_range"] = df["High"] - df["Low"]

    up_reversal = (df["High"] - df["Close"]) / df["day_range"].replace(0, np.nan)
    down_reversal = (df["Close"] - df["Low"]) / df["day_range"].replace(0, np.nan)
    intraday_reversal = (
        ((df["gap_pct"] > 0) & (up_reversal >= reversal_frac))
        | ((df["gap_pct"] < 0) & (down_reversal >= reversal_frac))
    )

    df["is_gap_day"] = df["gap_pct"].abs() >= gap_pct_threshold

    n = len(df)
    follow_through = pd.Series(False, index=df.index)
    for i in range(n):
        if not bool(df.loc[i, "is_gap_day"]):
            continue
        window_end = min(i + 1 + follow_through_bars, n)
        future = df.loc[i + 1 : window_end - 1]
        if future.empty:
            continue
        pre_gap_close = df.loc[i, "prior_close"]
        future_closes = future["Close"]
        if df.loc[i, "gap_pct"] > 0:
            move = df.loc[i, "High"] - pre_gap_close
            giveback_level = df.loc[i, "High"] - reversal_frac * move
            gave_back = bool((future_closes <= giveback_level).any())
        else:
            move = pre_gap_close - df.loc[i, "Low"]
            giveback_level = df.loc[i, "Low"] + reversal_frac * move
            gave_back = bool((future_closes >= giveback_level).any())
        follow_through.loc[i] = gave_back

    df["is_gap_reversal"] = df["is_gap_day"] & (intraday_reversal | follow_through)
    return df


def levels_near_target(
    df: pd.DataFrame,
    swings: list[dict[str, Any]],
    target_price: float,
    atr: float,
    direction: str,
    proximity_atr: float = DEFAULT_PROXIMITY_ATR,
    gap_pct_threshold: float = DEFAULT_GAP_PCT_THRESHOLD,
    retest_atr_frac: float = DEFAULT_RETEST_ATR_FRAC,
) -> list[dict[str, Any]]:
    """Find structural levels (gap-day extremes or swing pivots) within
    proximity_atr of target_price, classified by strength.

    `swings` is the same list scanner._find_swings already returns (reused
    here rather than re-deriving a separate pivot detector) -- each entry is
    {"index": int, "price": float, "type": "high"|"low"}.

    Returns a list of dicts, most-severe first (kind: "swing_pivot" |
    "gap_extreme"; strength: "weak" | "moderate" | "strong"), each with
    price, distance_atr, and a human-readable note.
    """
    if atr is None or atr <= 0:
        return []

    df = df.reset_index(drop=True)
    gap_df = detect_gap_days(df, gap_pct_threshold=gap_pct_threshold)
    tol = proximity_atr * atr
    findings: list[dict[str, Any]] = []
    gap_reversal_indices: set[int] = set()

    high_or_low = "High" if direction == "long" else "Low"
    for i in range(len(gap_df)):
        if not bool(gap_df.loc[i, "is_gap_reversal"]):
            continue
        gap_reversal_indices.add(i)
        level = float(gap_df.loc[i, high_or_low])
        if abs(level - target_price) > tol:
            continue
        findings.append({
            "index": i,
            "price": level,
            "kind": "gap_extreme",
            "distance_atr": abs(level - target_price) / atr,
            "strength": "weak",
            "note": (
                "Single-session gap/spike extreme with intraday reversal -- "
                "not organic structure, treat as unreliable resistance."
            ),
        })

    # A gap-reversal bar is intentionally NOT also evaluated as a swing pivot,
    # even if it happens to be a local max/min by the plain fractal
    # definition scanner._find_swings uses (a shock spike often is, by pure
    # coincidence of being the tallest bar in its own window). The two
    # classifications are mutually exclusive per bar: "this level is
    # unreliable because it's a spike" takes precedence over "this also
    # happens to look like real structure" for the same extreme.
    relevant_type = "high" if direction == "long" else "low"
    for swing in swings:
        if swing.get("type") != relevant_type:
            continue
        if int(swing["index"]) in gap_reversal_indices:
            continue
        level = float(swing["price"])
        if abs(level - target_price) > tol:
            continue
        idx = int(swing["index"])
        later = df.iloc[idx + 1 :]
        moved_away_hard = False
        retested_again = False
        if len(later) > 0:
            if direction == "long":
                post_low = later["Low"].min()
                moved_away_hard = (level - post_low) >= atr
                retested_again = bool((later["High"] >= level - retest_atr_frac * atr).any())
            else:
                post_high = later["High"].max()
                moved_away_hard = (post_high - level) >= atr
                retested_again = bool((later["Low"] <= level + retest_atr_frac * atr).any())

        if moved_away_hard or retested_again:
            strength = "strong"
            extreme_word = "high" if direction == "long" else "low"
            note = (
                f"Price moved away hard after making this {extreme_word} (prior rejection)."
                if moved_away_hard
                else "Price retested this level a second time -- confirmed resistance."
            )
        else:
            strength = "moderate"
            note = "Untested swing level -- genuine structure but not yet proven as resistance."

        findings.append({
            "index": idx,
            "price": level,
            "kind": "swing_pivot",
            "distance_atr": abs(level - target_price) / atr,
            "strength": strength,
            "note": note,
        })

    findings.sort(key=lambda f: (-_STRENGTH_RANK[f["strength"]], f["distance_atr"]))
    return findings


def clamp_target(
    entry: float,
    stop: float,
    target: float,
    atr: float,
    findings: list[dict[str, Any]],
    direction: str = "long",
    clamp_buffer_atr: float = DEFAULT_CLAMP_BUFFER_ATR,
    min_viable_rr: Optional[float] = None,
) -> dict[str, Any]:
    """Given the raw target and any findings from levels_near_target(), clamp
    the target to just below (long) / above (short) the nearest flagged
    level, and recompute R:R off the clamped value.

    Selection rule: clamps to whichever flagged level sits closest to ENTRY
    among the findings -- the first real obstacle in price's path toward the
    raw target -- not the "strongest" one. Every finding passed in is
    already within proximity_atr of the same raw target (that's what
    levels_near_target guarantees), so "nearest to entry" is equivalent to
    "first obstacle encountered on the way there": a stronger-but-farther
    level must not wave a weaker-but-closer one through unacknowledged.

    Floor guard: if clamping would push the adjusted target to/past `entry`,
    or would leave R:R below `min_viable_rr` (when given), the clamp is
    refused -- the raw target and R:R are kept, but the finding/badge are
    still surfaced (via clamp_refused_reason) so the caller can still warn
    the user instead of silently emitting a broken or degenerate trade plan.

    Returns: {adjusted_target, adjusted_rr, original_target, original_rr,
    clamped: bool, badge: str|None, nearest_finding: dict|None,
    clamp_refused_reason: str|None}.
    """
    original_rr = abs(target - entry) / abs(entry - stop)

    if not findings:
        return {
            "adjusted_target": target,
            "adjusted_rr": original_rr,
            "original_target": target,
            "original_rr": original_rr,
            "clamped": False,
            "badge": None,
            "nearest_finding": None,
            "clamp_refused_reason": None,
        }

    nearest = min(findings, key=lambda f: f["price"]) if direction == "long" \
        else max(findings, key=lambda f: f["price"])

    buf = clamp_buffer_atr * atr
    if direction == "long":
        candidate_target = min(target, nearest["price"] - buf)
    else:
        candidate_target = max(target, nearest["price"] + buf)

    candidate_rr = abs(candidate_target - entry) / abs(entry - stop)
    crosses_entry = candidate_target <= entry if direction == "long" else candidate_target >= entry
    below_floor = min_viable_rr is not None and candidate_rr < min_viable_rr

    badge_map = {
        ("swing_pivot", "strong"): "NEAR REJECTED HIGH" if direction == "long" else "NEAR REJECTED LOW",
        ("swing_pivot", "moderate"): "NEAR RESISTANCE" if direction == "long" else "NEAR SUPPORT",
        ("gap_extreme", "weak"): "TARGET NEAR GAP SPIKE",
    }
    badge = badge_map.get((nearest["kind"], nearest["strength"]), "NEAR RESISTANCE")

    if crosses_entry or below_floor:
        return {
            "adjusted_target": target,
            "adjusted_rr": original_rr,
            "original_target": target,
            "original_rr": original_rr,
            "clamped": False,
            "badge": badge,
            "nearest_finding": nearest,
            "clamp_refused_reason": (
                "adjusted target would cross entry" if crosses_entry
                else f"adjusted R:R would fall below {min_viable_rr}"
            ),
        }

    return {
        "adjusted_target": candidate_target,
        "adjusted_rr": candidate_rr,
        "original_target": target,
        "original_rr": original_rr,
        "clamped": True,
        "badge": badge,
        "nearest_finding": nearest,
        "clamp_refused_reason": None,
    }


def resolve_stop(
    entry: float,
    direction: str,
    atr: float,
    atr_multiplier: float,
    order_block: Optional[dict[str, Any]],
    buffer_atr: float = DEFAULT_CLAMP_BUFFER_ATR,
) -> dict[str, Any]:
    """Place the stop at the actual order-block invalidation level when a
    clean one exists, instead of always using the flat entry +/- atr_multiple
    fallback.

    `order_block` is scanner._find_order_block()'s return value directly
    (None, or {"high", "low", "index"}) -- the most recent bearish candle
    before the last swing low (longs) / bullish candle before the last swing
    high (shorts), i.e. the zone whose opposite edge is this codebase's own
    existing definition of the level that invalidates the setup (see the
    near_ob check in scanner.py's MTV3 order-block-interaction feature,
    which already uses "within 1 ATR of the order block edge" as its notion
    of proximity -- there is no separate proximity gate here beyond that
    established convention plus the sanity check below).

    Invalidation level: the order block's low for longs (price closing below
    it invalidates the bullish order block), its high for shorts. The stop is
    placed `buffer_atr` (same constant as clamp_target's buffer, so the two
    features read consistently) past that level, in the direction away from
    entry.

    "Clean" here means: a candle was actually found (order_block is not
    None), the resulting stop is still on the correct side of entry -- below
    entry for longs, above for shorts -- AND it's at least MIN_STOP_DISTANCE_ATR
    away from entry. That floor exists because an order block can sit almost
    exactly at entry (observed on real data: AMZN landed at 0.177 ATR before
    this floor existed), which produces a stop distance tight enough to be
    noise rather than a real invalidation level. The cutoff is inclusive of
    the floor itself: exactly MIN_STOP_DISTANCE_ATR away passes ("closer than"
    the floor is what's rejected, not "at or closer than"). If any of these
    three checks fails, this falls back to the flat ATR-multiple stop rather
    than ever emitting a stop that would make the trade un-computable (entry
    already past its own invalidation level, coincident with entry, or too
    tight to be a meaningful stop).

    Returns {stop, raw_stop, stop_source: "order_block" | "atr_multiple"}.
    raw_stop (the flat ATR-multiple value) is always populated, regardless of
    which stop is actually live, for the audit trail.
    """
    if direction == "short":
        raw_stop = entry + (atr_multiplier * atr)
    else:
        raw_stop = entry - (atr_multiplier * atr)

    if order_block is None:
        return {"stop": raw_stop, "raw_stop": raw_stop, "stop_source": "atr_multiple"}

    buf = buffer_atr * atr
    if direction == "long":
        candidate_stop = order_block["low"] - buf
        is_clean = candidate_stop < entry
    else:
        candidate_stop = order_block["high"] + buf
        is_clean = candidate_stop > entry

    if is_clean and atr > 0:
        stop_distance_atr = abs(entry - candidate_stop) / atr
        if stop_distance_atr < MIN_STOP_DISTANCE_ATR:
            is_clean = False

    if not is_clean:
        return {"stop": raw_stop, "raw_stop": raw_stop, "stop_source": "atr_multiple"}

    return {"stop": candidate_stop, "raw_stop": raw_stop, "stop_source": "order_block"}
