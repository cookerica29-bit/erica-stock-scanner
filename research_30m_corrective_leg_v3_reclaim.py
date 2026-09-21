"""30m Corrective-Leg Anchoring Research -- Phase 3: reclaim test (2026-09
session). DEVELOPER-ONLY RESEARCH SCRIPT. Not imported by main.py,
candidates_router.py, or any production module. Makes zero writes to any
database table. Reads real historical 30M bars via scanner._batch_download
(the same Alpaca-backed provider production already uses) and
prints/saves developer-only diagnostic output only.

Phase 2's recommendation (see research_30m_corrective_leg_v2_report.md,
Part 15) was **C** -- the pivot-based "controlling swing" methods (A-F)
almost never had an internal opposing pivot to select in the first place,
because real corrections are usually too short-lived for fixed-window
swing detection to resolve internal structure. Phase 2 flagged one
concrete, untried, more promising lead instead of "more of the same":
CLH's final bar closing back above the original pre-correction level was
a real, visible signal none of methods A-F were built to check, because
none of them require an internal opposing pivot.

This phase tests EXACTLY that, and nothing more:

    Has price closed back beyond the level where the correction started
    (impulse_end / correction.start_price from Phase 2's
    reconstruct_correction, which was NOT the part that failed -- Phase 2's
    correction reconstruction itself worked; it was the internal-pivot
    "controlling swing" selection on top of it that had nothing to select)?

reconstruct_correction is reused completely unchanged from Phase 2 --
importing it, not reimplementing it. This phase adds exactly one new
function (check_reclaim) and reuses Phase 2's own _is_broken_by_cutoff
for the actual close-through-level test, since that primitive's semantics
(thesis_direction=LONG: broken means CLOSE > level) are already exactly
what "reclaimed" means here -- no new close-comparison logic invented.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from typing import Optional

import pandas as pd

sys.path.insert(0, ".")

import scanner  # noqa: E402
import research_30m_confirmation_audit as phase1  # noqa: E402
import research_30m_corrective_leg_v2 as v2  # noqa: E402

MARGINS = (1, 2, 3)


def check_reclaim(df: pd.DataFrame, correction, thesis_direction: str, atr: float) -> dict:
    """Has price, at any point from the correction's extreme onward, closed
    back beyond correction.start_price (the pre-correction swing point)?
    Reports both "ever reclaimed" (first bar that did, however long ago)
    and "currently holding" (is the MOST RECENT bar, as of cutoff, still
    beyond the level) -- these are different claims: price can reclaim and
    then fall back inside the correction, and only the second is an
    honest "as of right now" signal.
    """
    if correction.state not in ("CORRECTION_DEVELOPING", "CORRECTION_AMBIGUOUS"):
        return {"applicable": False, "reason": correction.state}
    if correction.extreme_timestamp is None or correction.start_price is None:
        return {"applicable": False, "reason": "no_extreme_or_start"}

    try:
        extreme_idx = df.index.get_loc(pd.Timestamp(correction.extreme_timestamp))
    except KeyError:
        return {"applicable": False, "reason": "extreme_timestamp_not_in_df"}

    reclaim = v2._is_broken_by_cutoff(df, correction.start_price, thesis_direction, extreme_idx)
    last_close = float(df["Close"].iloc[-1])
    currently_holding = (last_close > correction.start_price) if thesis_direction == "LONG" else (last_close < correction.start_price)

    out = {
        "applicable": True,
        "correction_start_price": correction.start_price,
        "correction_extreme_price": correction.extreme_price,
        "ever_reclaimed": bool(reclaim.get("close_through")),
        "currently_holding_above_start": currently_holding if thesis_direction == "LONG" else None,
        "currently_holding_below_start": currently_holding if thesis_direction == "SHORT" else None,
        "last_close": round(last_close, 4),
        "bars_since_extreme": int(len(df) - 1 - extreme_idx),
    }
    if reclaim.get("close_through"):
        out["first_reclaim_timestamp"] = reclaim.get("break_timestamp")
        out["first_reclaim_close"] = reclaim.get("break_close")
        if atr and atr > 0:
            out["reclaim_distance_atr"] = round(abs(reclaim["break_close"] - correction.start_price) / atr, 3)
        bars_to_reclaim = None
        try:
            reclaim_idx = df.index.get_loc(pd.Timestamp(reclaim["break_timestamp"]))
            bars_to_reclaim = int(reclaim_idx - extreme_idx)
        except (KeyError, TypeError):
            pass
        out["bars_from_extreme_to_reclaim"] = bars_to_reclaim
    return out


def run_for_ticker(ticker: str, direction: str, cutoff_iso: str, base_margin: int) -> dict:
    thesis_direction = "LONG" if direction.lower() == "long" else "SHORT"
    cutoff = phase1._to_utc(cutoff_iso)
    raw_30m = phase1.fetch_bars(ticker, interval="30m", period="60d")
    if raw_30m.empty:
        return {"ticker": ticker, "error": "no 30m data returned"}
    df = phase1.closed_candles_only(phase1.truncate_point_in_time(raw_30m, cutoff), cutoff, bar_minutes=30)
    if len(df) < 30:
        return {"ticker": ticker, "error": "insufficient point-in-time bars", "bars": len(df)}

    atr = scanner._compute_atr(df, period=14)
    pivots = v2.score_pivot_significance(v2.raw_pivots(df, base_margin), atr)
    correction = v2.reconstruct_correction(df, pivots, thesis_direction, atr)
    reclaim = check_reclaim(df, correction, thesis_direction, atr)

    return {
        "ticker": ticker,
        "cutoff": cutoff_iso,
        "bars_point_in_time": len(df),
        "atr": round(atr, 4),
        "correction_state": correction.state,
        "correction": asdict(correction),
        "reclaim": reclaim,
    }


def main():
    results = []
    for ticker, label, decision, reviewed_at, note in phase1.LABELED_EXAMPLES:
        print(f"--- {ticker} (label={label}, decision={decision}) ---", file=sys.stderr, flush=True)
        per_margin = {}
        for m in MARGINS:
            per_margin[m] = run_for_ticker(ticker, "long", reviewed_at, base_margin=m)
        results.append({
            "ticker": ticker, "label": label, "decision": decision,
            "cutoff": reviewed_at, "note": note, "per_margin": per_margin,
        })
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
