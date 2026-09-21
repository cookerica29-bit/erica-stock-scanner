"""Pullback screener -- surfaces tickers currently in a corrective leg of
their dominant trend, per the user's own stated strategy: "always looking
for the correction leg of a trend for entry."

Deliberately separate from morning_watchlist.py's aligned/watch_reversal
buckets (a different question: "where to look today" vs "who's pulling
back right now") and from scanner.py's A+/B+/C grading pipeline (no
entries/stops/targets/ENTER_NOW here either).

Built directly on real, validated research, not a fresh guess:
  - Direction bias: the same 200-day SMA read morning_watchlist.py uses,
    so a ticker's LONG/SHORT story is consistent across both tabs.
  - Correction detection: research_30m_corrective_leg_v2's
    reconstruct_correction, unchanged -- Phase 2 found this part (finding
    the impulse -> correction -> extreme) was not what failed.
  - Reclaim test: research_30m_corrective_leg_v4_threshold_fix's
    near_reclaim (0.15 ATR tolerance, borrowed from Phase 1's own
    detector_B precedent, not tuned to any specific dataset) -- verified
    at margin=2 against 43 real human-reviewed examples: 62% approve /
    24% watch / 17% reject separation, zero added false positives vs the
    stricter original threshold. See research_30m_corrective_leg_v4_
    threshold_fix.py and its companion reports for the full evidence.

Two buckets:
  - "reclaimed": a corrective leg was found AND price has closed back
    (within the validated tolerance) beyond the level where the
    correction started -- the strongest, most-actionable signal.
  - "correcting": a corrective leg was found but not yet reclaimed --
    still pulling back, worth watching for the reclaim.
Everything else (no clear 200SMA bias, or no corrective leg detected) is
omitted.
"""

from __future__ import annotations

from datetime import datetime, timezone

import scanner
from morning_watchlist import _sma200_bias
from research_30m_corrective_leg_v2 import raw_pivots, reconstruct_correction, score_pivot_significance
from research_30m_corrective_leg_v4_threshold_fix import check_reclaim_variants

PULLBACK_WATCHLIST_VERSION = "pullback-watchlist-v1"
BASE_PIVOT_MARGIN = 2  # the margin validated in Phase 4 -- see module docstring


def _frame_missing(frame) -> bool:
    return frame is None or getattr(frame, "empty", True)


def _pullback_entry_for_ticker(ticker: str, daily_raw, m30_raw) -> dict | None:
    daily_df = scanner._flatten_columns(daily_raw) if daily_raw is not None else None
    bias, price, sma200 = _sma200_bias(daily_df)
    if bias is None:
        return None

    m30_df = scanner._flatten_columns(m30_raw) if m30_raw is not None else None
    if m30_df is None or m30_df.empty or len(m30_df) < 30:
        return None

    atr = scanner._compute_atr(m30_df, period=14)
    if not atr or atr <= 0:
        return None

    pivots = score_pivot_significance(raw_pivots(m30_df, BASE_PIVOT_MARGIN), atr)
    correction = reconstruct_correction(m30_df, pivots, bias, atr)
    if correction.state not in ("CORRECTION_DEVELOPING", "CORRECTION_AMBIGUOUS"):
        return None

    reclaim = check_reclaim_variants(m30_df, correction, bias, atr)
    if not reclaim.get("applicable"):
        return None

    bucket = "reclaimed" if reclaim["near_reclaim"] else "correcting"
    direction_word = "bullish" if bias == "LONG" else "bearish"
    corrective_word = "bearish" if bias == "LONG" else "bullish"

    if bucket == "reclaimed":
        reason = (
            f"{direction_word.capitalize()} trend (200SMA ${sma200:.2f}) pulled back to "
            f"${correction.extreme_price:.2f}, then closed back near/beyond the pre-pullback level "
            f"(${correction.start_price:.2f}) on the 30M chart — correction may be resolving."
        )
    else:
        reason = (
            f"{direction_word.capitalize()} trend (200SMA ${sma200:.2f}) is in a {corrective_word} "
            f"30M pullback from ${correction.start_price:.2f} to ${correction.extreme_price:.2f} — "
            f"not yet reclaimed, watch for a close back beyond ${correction.start_price:.2f}."
        )

    return {
        "ticker": ticker,
        "bucket": bucket,
        "bias": bias,
        "reason": reason,
        "price": round(price, 2),
        "sma200_daily": round(sma200, 2),
        "correction_start_price": correction.start_price,
        "correction_extreme_price": correction.extreme_price,
        "correction_depth_pct": correction.correction_depth_pct,
        "bars_in_correction": correction.bars_in_correction,
        "last_close_30m": reclaim.get("last_close"),
        "distance_from_full_reclaim_atr": reclaim.get("distance_from_full_reclaim_atr"),
    }


def build_pullback_watchlist(tickers: list[str] | None = None) -> dict:
    """Scan `tickers` (defaults to the full stock WATCHLIST) for tickers
    currently in -- or just out of -- a corrective leg of their dominant
    trend. Read-only, side-effect-free, does not touch scanner.py's
    grading/caching machinery.
    """
    symbols = tickers if tickers is not None else scanner.WATCHLIST
    reclaimed: list[dict] = []
    correcting: list[dict] = []
    errors: list[dict] = []

    daily_frames = scanner._batch_download(symbols, period="1y", interval="1d")
    m30_frames = scanner._batch_download(symbols, period="60d", interval="30m")

    for ticker in symbols:
        daily_raw = daily_frames.get(ticker)
        m30_raw = m30_frames.get(ticker)

        if _frame_missing(daily_raw) and _frame_missing(m30_raw):
            errors.append({"ticker": ticker, "reason": "no_data", "detail": "Daily and 30M data both failed to fetch."})
            continue
        if _frame_missing(daily_raw):
            errors.append({"ticker": ticker, "reason": "no_daily_data", "detail": "Daily data failed to fetch (200SMA/bias unavailable)."})
            continue
        if _frame_missing(m30_raw):
            errors.append({"ticker": ticker, "reason": "no_30m_data", "detail": "30M data failed to fetch (correction/reclaim check unavailable)."})
            continue

        try:
            entry = _pullback_entry_for_ticker(ticker, daily_raw, m30_raw)
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not kill the scan
            errors.append({"ticker": ticker, "reason": "processing_error", "detail": str(exc)})
            continue
        if entry is None:
            continue
        (reclaimed if entry["bucket"] == "reclaimed" else correcting).append(entry)

    reclaimed.sort(key=lambda row: row["ticker"])
    correcting.sort(key=lambda row: row["ticker"])
    errors.sort(key=lambda row: row["ticker"])

    return {
        "version": PULLBACK_WATCHLIST_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scanned_count": len(symbols),
        "reclaimed": reclaimed,
        "correcting": correcting,
        "errors": errors,
    }
