"""Automatic bar-replay outcome resolution for "taken" candidate promotions.

Step 2 of real outcome tracking (Option C, agreed): a promotion the user
marked as actually taken gets its real outcome resolved automatically from
real historical bars -- did price touch the target first, the stop first,
neither yet, or did tracking time out. This module is the pure
bar-classification logic only; no DB access, no network calls -- see
main._watch_candidate_promotion_outcomes for the periodic job that fetches
bars and calls this.

Modeled on the same shape of logic as scanner._mtf_v3_measure_underlying_outcome
(walk bars forward from an event time, check each one's High/Low against the
trade's levels, stop at the first bar that resolves anything) -- simplified
here to a first-touch classification (this doesn't need MFE/MAE or R-multiple
tracking, just "which happened first").

Column convention: same as displacement_score.py/structural_resistance.py --
capitalized Open/High/Low/Close, a DatetimeIndex. Direction convention:
lowercase "long"/"short", same as those two modules (NOT scanner._detect_bos's
uppercase "LONG"/"SHORT" -- caller must convert if reusing a direction string
from that call site).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd

# Placeholder, not validated -- same category of decision as
# structural_resistance.MIN_STOP_DISTANCE_ATR, displacement_score's weights,
# and the 1.5 R:R floor: a real call about how long to keep tracking an
# unresolved trade before giving up, not something to treat as settled just
# because it shipped. Chosen to comfortably exceed a typical option contract's
# DTE observed in this codebase's real data this session (~29-30 days), with
# margin -- but that's a reasonable-guess starting point, not evidence.
DEFAULT_MAX_TRACKING_DAYS = 45.0

OUTCOME_HIT_TARGET = "hit_target"
OUTCOME_HIT_STOP = "hit_stop"
OUTCOME_STILL_OPEN = "still_open"
OUTCOME_EXPIRED = "expired"
OUTCOME_AMBIGUOUS = "ambiguous"


def _iso(ts) -> str:
    if isinstance(ts, datetime):
        dt = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    else:
        dt = pd.Timestamp(ts).to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def resolve_outcome(
    direction: str,
    stop: float,
    target: Optional[float],
    bars: pd.DataFrame,
    promoted_at: datetime,
    now: datetime,
    bar_source: str = "4h",
    max_tracking_days: float = DEFAULT_MAX_TRACKING_DAYS,
) -> dict:
    """Classify what happened to a taken promotion using real bars.

    `bars` must already be filtered to strictly after `promoted_at` and
    sorted ascending by time -- this function doesn't do either. `target`
    may be None (the no_valid_target case): hit_target can never be reached,
    only hit_stop/still_open/expired are possible outcomes for those.

    Returns {"outcome", "hit_at" (iso str or None), "bar_source", "note"}.

    The honest limit this documents rather than hides: even a 4h bar can, in
    principle, have BOTH the stop and the target inside its [Low, High]
    range -- OHLC data alone can't tell you which was actually touched
    first within that bar. When that happens this returns "ambiguous" with
    a note explaining why, rather than guessing (e.g. via open/close
    position, which would just be a disguised guess, not a real answer).
    That ambiguity is about that specific bar's internal path and can't be
    resolved by more/later data, so it's effectively terminal once it
    occurs -- same as hit_target/hit_stop, unlike still_open.
    """
    direction = str(direction or "").strip().lower()
    for ts, row in bars.iterrows():
        high = float(row["High"])
        low = float(row["Low"])
        if direction == "long":
            hit_target = target is not None and high >= target
            hit_stop = low <= stop
        elif direction == "short":
            hit_target = target is not None and low <= target
            hit_stop = high >= stop
        else:
            return {
                "outcome": None, "hit_at": None, "bar_source": bar_source,
                "note": f"unsupported direction {direction!r}",
            }

        if hit_target and hit_stop:
            return {
                "outcome": OUTCOME_AMBIGUOUS,
                "hit_at": None,
                "bar_source": bar_source,
                "note": (
                    f"bar at {_iso(ts)} touched both target and stop within "
                    f"the same {bar_source} candle -- true order can't be "
                    f"determined from OHLC alone"
                ),
            }
        if hit_target:
            return {"outcome": OUTCOME_HIT_TARGET, "hit_at": _iso(ts), "bar_source": bar_source, "note": None}
        if hit_stop:
            return {"outcome": OUTCOME_HIT_STOP, "hit_at": _iso(ts), "bar_source": bar_source, "note": None}

    age_days = (now - promoted_at).total_seconds() / 86400.0
    if age_days > max_tracking_days:
        return {
            "outcome": OUTCOME_EXPIRED,
            "hit_at": None,
            "bar_source": bar_source,
            "note": f"no resolution within the {max_tracking_days:.0f}-day tracking window",
        }
    return {"outcome": OUTCOME_STILL_OPEN, "hit_at": None, "bar_source": bar_source, "note": None}
