"""Premium/discount location scoring for candle-level context.

Replaces legacy's dual, inconsistent bucketing -- scanner._location_read
uses Premium >=67%/Discount <=33%/Midrange; scanner._strict_location uses a
DIFFERENT scheme (AT EXTREME <=18% or >=82%, NEAR DISCOUNT <=35%, NEAR
PREMIUM >=65%, MIDRANGE) for the exact same underlying percentile. Two
different threshold schemes for one number is itself the evidence neither
was ever validated -- so this exposes the continuous 0-100 percentile as
the primary, authoritative field, with a single light categorical label
derived from it purely for display. Never a gate, never a second source of
truth: see candidates_router.py for where this is attached
(informational only, alongside target_clamp_badge/stop_source/
displacement_read/bos_details/macro_conflict/choch_details/sweep_details/
rejection_details).

Named "score_location" (not "_location_read" or "_strict_location") to
avoid colliding with scanner.py's existing private names -- same
precaution as displacement_score.score_displacement vs
scanner.detect_displacement.

Direction convention: lowercase "long"/"short", matching
structural_resistance.py/displacement_score.py (NOT scanner._detect_bos's
uppercase "LONG"/"SHORT" -- caller must convert if reusing a direction
string from that call site).
"""

from __future__ import annotations

from typing import Any, Optional


def _latest_swing_range(swings: list[dict]) -> Optional[dict]:
    """"Latest swing range" -- same definition as scanner._latest_swing_range:
    the HIGH of the last 3 swing highs, paired with the LOW of the last 3
    swing lows (not just the single most recent pair on each side).
    Reimplemented here rather than importing scanner.py's private helper
    into a new standalone module -- copied byte-for-byte and cross-checked
    against scanner._latest_swing_range's real output on real tickers
    before shipping (see session notes) to catch any drift between the
    two rather than trusting the copy blind.
    """
    highs = [s for s in swings if s["type"] == "high"]
    lows = [s for s in swings if s["type"] == "low"]
    if not highs or not lows:
        return None
    high = max(highs[-3:], key=lambda s: s["price"])
    low = min(lows[-3:], key=lambda s: s["price"])
    if high["price"] <= low["price"]:
        return None
    return {"high": high["price"], "low": low["price"]}


# Display-only bucket cutoffs -- legacy's own simpler 3-tier scheme
# (scanner._location_read), picked over its other 4-tier AT EXTREME
# variant (scanner._strict_location) for readability, per explicit
# instruction. Neither legacy scheme was ever validated; location_percentile
# below is the real, authoritative value -- this label is sugar on top of
# it, not a second source of truth, and never gates anything.
PREMIUM_THRESHOLD = 67.0
DISCOUNT_THRESHOLD = 33.0


def score_location(price: float, swings: list[dict], direction: str) -> dict[str, Any]:
    """Where does `price` sit (0-100) within the latest swing range.

    Returns {location_percentile, location_label, location_alignment,
    range_high, range_low}. location_percentile is the primary field --
    None only when there's no valid recent swing range to measure against
    (fewer than one swing on either side, or a degenerate/inverted range),
    in which case everything else is None too.

    location_alignment is a light, direction-relative reading (not a new
    computation -- purely derived from location_label): "favorable" when
    the zone favors this direction's entry (discount for long, premium for
    short -- legacy's own "better location" framing), "unfavorable" the
    opposite ("late for fresh longs/shorts" in legacy's wording), "neutral"
    for midrange.
    """
    direction = str(direction or "").strip().lower()
    swing_range = _latest_swing_range(swings)
    if swing_range is None:
        return {
            "location_percentile": None,
            "location_label": None,
            "location_alignment": None,
            "range_high": None,
            "range_low": None,
        }

    low = swing_range["low"]
    high = swing_range["high"]
    percentile = max(0.0, min(1.0, (price - low) / (high - low)))
    location_percentile = round(percentile * 100, 1)

    if location_percentile >= PREMIUM_THRESHOLD:
        location_label = "premium"
    elif location_percentile <= DISCOUNT_THRESHOLD:
        location_label = "discount"
    else:
        location_label = "midrange"

    if location_label == "midrange":
        location_alignment = "neutral"
    elif direction == "long":
        location_alignment = "favorable" if location_label == "discount" else "unfavorable"
    elif direction == "short":
        location_alignment = "favorable" if location_label == "premium" else "unfavorable"
    else:
        location_alignment = None

    return {
        "location_percentile": location_percentile,
        "location_label": location_label,
        "location_alignment": location_alignment,
        "range_high": round(float(high), 4),
        "range_low": round(float(low), 4),
    }
