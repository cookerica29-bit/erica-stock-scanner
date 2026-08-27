"""Continuous displacement score for candle-level conviction grading.

Replaces the legacy hard STRONG/WEAK/NONE gate (scanner.detect_displacement /
scanner._displacement_strength_for_indices -- fixed ATR-multiple thresholds,
no volume input at all) with a 0-100 score for use as a grading *input*, not
a gate. It never blocks ENTER_NOW eligibility on its own; see candidates_router
for where it's attached (informational only, alongside target_clamp_badge/
stop_source).

Named "score_displacement" (not "detect_displacement" or "displacement_score"
as a bare function) specifically to avoid colliding with the legacy names
already used throughout scanner.py's BOS-linked displacement shadow tracking
-- that code is untouched and out of scope; this is a new, separate signal.

Four components, each a percentile rank against the stock's OWN trailing
history (not a global constant) -- this is the actual fix for "hard to nail
down": a stock's own recent volatility regime sets the bar, rather than one
fixed ATR multiple applied to every stock alike.
  1. Relative body size  -- is this candle's body big FOR THIS STOCK lately?
  2. Relative range size -- same, for full high-low range.
  3. Close-location-value (CLV), direction-adjusted -- did price close near
     the extreme in the trade's direction (conviction), or wick back against
     it (indecision)?
  4. Relative volume -- did participation actually expand? (absent from the
     legacy gate entirely; confirmed present and populated in this codebase's
     real OHLCV data -- see the real-data check in tests/displacement_score_v1.py)

Column convention: same as structural_resistance.py -- capitalized
Open/High/Low/Close/Volume columns, a plain positional (0..n-1) index.
Direction convention: lowercase "long"/"short", matching structural_resistance.py
(NOT scanner._find_order_block's uppercase "LONG"/"SHORT" -- caller must
convert if reusing a direction string from that call site).
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

DEFAULT_LOOKBACK_BARS = 25

# Placeholder, not validated -- same category of decision as
# structural_resistance.MIN_STOP_DISTANCE_ATR and the 1.5 R:R floor: a real
# strategy call about how much each component should matter, not something to
# treat as settled just because it shipped. Flagging plainly rather than
# quietly picking a "final" split.
DEFAULT_DISPLACEMENT_WEIGHTS = {"body": 0.35, "range": 0.15, "clv": 0.25, "volume": 0.25}

# Also a placeholder: a big-bodied candle in the wrong direction is heavily
# discounted, not hard-zeroed (per the design brief -- it can still matter for
# context/logging), but *how much* to discount it by is exactly as unvalidated
# as the weights above.
WRONG_DIRECTION_PENALTY_MULTIPLIER = 0.3

# Label cutoffs -- informational display bucketing only, never a gate. Equally
# unvalidated; grouped here with the other placeholders rather than left as
# bare numbers in the return statement.
STRONG_LABEL_MIN_SCORE = 70.0
MODERATE_LABEL_MIN_SCORE = 40.0


def _percentile_rank(series: pd.Series, value: float) -> float:
    """Fraction of `series` that is <= value. Returns 0-1; 0.5 (neutral) if
    there's no history to compare against."""
    valid = series.dropna()
    if len(valid) == 0:
        return 0.5
    return float((valid <= value).sum()) / len(valid)


def score_displacement(
    df: pd.DataFrame,
    direction: str,
    index: Optional[int] = None,
    lookback: int = DEFAULT_LOOKBACK_BARS,
    weights: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """Score the candle at `index` (default: the last bar in df -- the real
    use case everywhere this is actually called from) against its own
    trailing `lookback` bars. Deliberately a single bar, not averaged across
    a window like the legacy gate -- averaging dilutes a single decisive
    candle's signal.

    Returns {score: 0-100, components: {...}, label} -- label is
    informational display only, never a gate; the continuous score is the
    real output.
    """
    weights = weights or DEFAULT_DISPLACEMENT_WEIGHTS
    if index is None:
        index = len(df) - 1
    if index < 1 or index >= len(df):
        return {"score": 0.0, "components": {}, "label": "N/A", "note": "index out of range"}

    window_start = max(0, index - lookback)
    hist = df.iloc[window_start:index]  # trailing history, EXCLUDING the bar itself
    if len(hist) < 5:
        return {"score": 0.0, "components": {}, "label": "N/A", "note": "insufficient history"}

    bar = df.iloc[index]
    body = abs(bar["Close"] - bar["Open"])
    rng = bar["High"] - bar["Low"]
    if rng <= 0:
        return {"score": 0.0, "components": {}, "label": "NONE", "note": "zero-range bar"}

    hist_bodies = (hist["Close"] - hist["Open"]).abs()
    hist_ranges = hist["High"] - hist["Low"]

    body_pct = _percentile_rank(hist_bodies, body)
    range_pct = _percentile_rank(hist_ranges, rng)

    if direction == "long":
        clv = (bar["Close"] - bar["Low"]) / rng
    else:
        clv = (bar["High"] - bar["Close"]) / rng
    clv = max(0.0, min(1.0, clv))

    if "Volume" in df.columns and len(hist["Volume"].dropna()) >= 5:
        volume_pct = _percentile_rank(hist["Volume"], bar["Volume"])
    else:
        volume_pct = 0.5

    # bool(...) here matters, not just style: bar["Close"]/bar["Open"] are
    # numpy scalars, so this comparison is a numpy.bool_, not a plain Python
    # bool -- json.dumps() (used when this dict gets persisted to the
    # candidate_plan_previews/candidate_promotions cache) rejects numpy.bool_.
    directional = bool((bar["Close"] > bar["Open"]) if direction == "long" else (bar["Close"] < bar["Open"]))
    directional_mult = 1.0 if directional else WRONG_DIRECTION_PENALTY_MULTIPLIER

    raw_score = (
        weights["body"] * body_pct
        + weights["range"] * range_pct
        + weights["clv"] * clv
        + weights["volume"] * volume_pct
    ) * directional_mult

    score = round(raw_score * 100, 1)

    if score >= STRONG_LABEL_MIN_SCORE:
        label = "STRONG"
    elif score >= MODERATE_LABEL_MIN_SCORE:
        label = "MODERATE"
    else:
        label = "WEAK"

    return {
        "score": score,
        "components": {
            "body_percentile": round(body_pct * 100, 1),
            "range_percentile": round(range_pct * 100, 1),
            "close_location_value": round(clv * 100, 1),
            "volume_percentile": round(volume_pct * 100, 1),
            "directional": directional,
        },
        "label": label,
    }
