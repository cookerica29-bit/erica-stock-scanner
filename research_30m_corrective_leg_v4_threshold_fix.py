"""30m Corrective-Leg Anchoring Research -- Phase 4: reclaim threshold fix
(2026-09-21 session). DEVELOPER-ONLY RESEARCH SCRIPT. Not imported by
main.py, candidates_router.py, or any production module. Makes zero writes.

Phase 3 Batch 2 found two concrete misses on real human-confirmed examples:
  - XOM: last close ($164.025) was 8.5 cents short of the exact
    pre-correction peak ($164.113) -- essentially fully reclaimed, missed
    by a hair on the strict "close > start_price" test.
  - OXY: reconstruction landed on the right price zone (~$60.75-61.17,
    matching the human's own note almost exactly) but by cutoff price had
    only recovered about half the correction's depth, not the full
    round-trip back to the pre-correction peak.

Both misses are the SAME root cause: check_reclaim's bar (a full,
decisive close beyond the exact pre-correction level) is stricter than
what a human actually requires to call a correction "reclaimed." This
phase tests three progressively looser variants against the exact same
43 real labeled examples already gathered (no new data collection) to
see whether loosening the bar closes the gap without destroying the
approve/watch/reject separation Phase 3 already found:

  1. full_reclaim   -- Phase 3's original: close > correction.start_price
  2. near_reclaim    -- close > start_price - 0.15 ATR (mirrors Phase 1's
                         own detector_B pattern: "close beyond by >= 0.15
                         ATR" was already validated there as a reasonable
                         decisive-close tolerance, applied here in reverse
                         as a tolerance band instead of a minimum)
  3. half_reclaim    -- close > extreme + 0.5 * (start - extreme), i.e.
                         retraced at least half the correction's depth
"""

from __future__ import annotations

import json
import sys

import pandas as pd

sys.path.insert(0, ".")

import scanner  # noqa: E402
import research_30m_confirmation_audit as phase1  # noqa: E402
import research_30m_corrective_leg_v2 as v2  # noqa: E402
import research_30m_corrective_leg_v3_reclaim_batch2 as batch2  # noqa: E402

MARGINS = (1, 2, 3)
NEAR_RECLAIM_ATR_TOLERANCE = 0.15  # precedent: Phase 1's detector_B


def check_reclaim_variants(df: pd.DataFrame, correction, thesis_direction: str, atr: float) -> dict:
    if correction.state not in ("CORRECTION_DEVELOPING", "CORRECTION_AMBIGUOUS"):
        return {"applicable": False, "reason": correction.state}
    if correction.extreme_timestamp is None or correction.start_price is None:
        return {"applicable": False, "reason": "no_extreme_or_start"}

    last_close = float(df["Close"].iloc[-1])
    start = correction.start_price
    extreme = correction.extreme_price
    depth = abs(start - extreme)
    half_level = extreme + 0.5 * (start - extreme) if thesis_direction == "LONG" else extreme - 0.5 * (extreme - start)
    near_level = start - NEAR_RECLAIM_ATR_TOLERANCE * atr if thesis_direction == "LONG" else start + NEAR_RECLAIM_ATR_TOLERANCE * atr

    if thesis_direction == "LONG":
        full_reclaim = last_close > start
        near_reclaim = last_close > near_level
        half_reclaim = last_close > half_level
    else:
        full_reclaim = last_close < start
        near_reclaim = last_close < near_level
        half_reclaim = last_close < half_level

    return {
        "applicable": True,
        "last_close": round(last_close, 4),
        "start_price": start,
        "extreme_price": extreme,
        "correction_depth": round(depth, 4),
        "full_reclaim": full_reclaim,
        "near_reclaim": near_reclaim,
        "half_reclaim": half_reclaim,
        "distance_from_full_reclaim_atr": round((start - last_close) / atr, 3) if atr and thesis_direction == "LONG" else None,
    }


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
    reclaim = check_reclaim_variants(df, correction, thesis_direction, atr)

    return {
        "ticker": ticker,
        "correction_state": correction.state,
        "reclaim": reclaim,
    }


def main():
    combined = list(phase1.LABELED_EXAMPLES) + batch2.NEW_EXAMPLES
    results = []
    for ticker, label, decision, reviewed_at, note in combined:
        print(f"--- {ticker} ---", file=sys.stderr, flush=True)
        per_margin = {}
        for m in MARGINS:
            per_margin[m] = run_for_ticker(ticker, "long", reviewed_at, base_margin=m)
        results.append({"ticker": ticker, "label": label, "decision": decision, "cutoff": reviewed_at, "per_margin": per_margin})
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
