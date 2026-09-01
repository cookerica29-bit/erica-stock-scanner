"""30-Minute Execution Confirmation Research Audit (2026-09 session).

DEVELOPER-ONLY RESEARCH SCRIPT. Not imported by main.py, candidates_router.py,
or any production module. Makes zero writes to any database table -- it
only reads real historical 30M/4H/Daily bars (via scanner._batch_download,
the same Alpaca-backed provider production already uses) and prints a
structured report to stdout. No candidate/candidate_visual_reviews/
approved_setup_memories/monitor_state/candidate_promotions row is ever
touched.

Purpose: determine whether Kairos can MECHANICALLY recognize the same
lower-timeframe (30M) confirmation event a human reviewer currently
recognizes visually -- using ONLY real, point-in-time-truncated historical
data (no lookahead), and ONLY existing, already-audited structural
primitives (_find_swings, detect_structure_break/_detect_bos, _detect_choch,
detect_liquidity_sweep, detect_rejection, _compute_atr) -- reused exactly
as they exist today, never modified, never given new "30m-tuned" thresholds
that would count as inventing new production logic.

This script does NOT decide anything. It measures and reports.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

sys.path.insert(0, ".")

import scanner  # noqa: E402


# ---------------------------------------------------------------------------
# Labeled dataset -- pulled directly from production candidate_visual_reviews
# via GET /candidate-visual-reviews (2026-09-01 session), not fabricated.
# Every ticker here is a REAL human-reviewed setup with a real
# lower_tf_confirmation answer and a real reviewed_at timestamp. All are
# direction="long" (confirmed via each row's own setup_key).
# ---------------------------------------------------------------------------
LABELED_EXAMPLES = [
    # ticker, lower_tf_confirmation label, decision, reviewed_at (ISO, UTC), note
    ("VUG", "not_yet", "watch", "2026-09-01T01:55:24.716456+00:00", None),
    ("KHC", "not_yet", "watch", "2026-09-01T01:53:39.818166+00:00", None),
    ("OXY", "not_yet", "watch", "2026-09-01T01:51:46.212679+00:00", None),
    ("LPLA", "not_yet", "watch", "2026-09-01T01:49:45.637402+00:00", None),
    ("QQQ", "not_yet", "watch", "2026-09-01T01:47:51.599376+00:00", None),
    ("QQQM", "not_yet", "watch", "2026-09-01T01:45:57.616282+00:00", None),
    ("VGT", "yes", "watch", "2026-09-01T01:44:06.880868+00:00", None),
    ("KMI", "yes", "reject", "2026-09-01T01:42:36.223017+00:00", None),
    ("NTNX", "not_yet", "watch", "2026-09-01T01:40:22.767610+00:00", None),
    ("BILL", "not_yet", "watch", "2026-09-01T01:38:05.066013+00:00", None),
    ("MTDR", "yes", "watch", "2026-09-01T01:36:34.195131+00:00", None),
    ("TTAN", "not_yet", "watch", "2026-09-01T01:34:40.424553+00:00", "$102-103 weak-high liquidity has to be cleared first"),
    ("XLK", "not_yet", "reject", "2026-09-01T01:32:32.910688+00:00", None),  # market_structure=range -- rejected on daily/4h grounds
    ("EQH", "not_yet", "watch", "2026-09-01T01:25:50.996241+00:00", None),
    ("CRBG", "not_yet", "watch", "2026-09-01T01:22:38.157435+00:00", None),
    ("WFC", "not_yet", "reject", "2026-09-01T01:18:04.318347+00:00", None),  # market_structure=range
    ("DVN", "yes", "watch", "2026-09-01T01:16:32.672142+00:00", None),
    ("NVDA", "yes", "approve", "2026-09-01T01:13:30.652359+00:00", "Bullish HTF structure; 30M bullish CHoCH after correction. Neutral entry location with overhead structure near 227-230."),
    ("FFIV", "yes", "approve", "2026-09-01T01:08:53.668785+00:00", None),  # weaker positive label -- no note describing the visual event
    ("CLH", "not_yet", "approve", "2026-09-01T00:43:10.213229+00:00", "Bullish HTF structure; bearish correction pulling back into prior breakout area. Wait for lower-TF bullish confirmation"),
    ("BKR", "not_yet", "reject", "2026-09-01T00:00:52.586262+00:00", None),  # market_structure=range
    ("CF", "not_yet", "reject", "2026-08-31T23:59:11.943627+00:00", None),  # market_structure=range
    ("DUOL", "not_yet", "watch", "2026-08-31T23:49:49.510390+00:00", None),
    ("IGV", "yes", "approve", "2026-08-31T20:21:03.468053+00:00", "Deploy verification -- real live submission"),
]

ANCHOR_TICKERS = {"NVDA", "FFIV", "CLH"}


def _to_utc(ts: str) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert("UTC")


def fetch_bars(ticker: str, interval: str, period: str) -> pd.DataFrame:
    """Reuses scanner._batch_download exactly as production does (same
    Alpaca-backed provider ma_pipeline.py/candidates_router.py already
    use) -- no new data-fetching code, no new provider call pattern."""
    result = scanner._batch_download([ticker], period=period, interval=interval)
    df = result.get(ticker.upper())
    if df is None or df.empty:
        return pd.DataFrame()
    try:
        return scanner._flatten_columns(df.copy()).dropna().astype(float)
    except Exception:
        return pd.DataFrame()


def truncate_point_in_time(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """CRITICAL: only bars whose timestamp is <= cutoff are kept -- this is
    what prevents lookahead bias. A bar's timestamp in this codebase's bar
    data represents the bar's OPEN/period-start; to be conservative and
    only use genuinely CLOSED candles as of cutoff, we require the bar's
    timestamp + interval to be <= cutoff. Since we don't have interval
    metadata on the df itself here, we instead require timestamp < cutoff
    AND drop the very last bar if it is within one interval-guess of the
    cutoff -- see the caller's explicit interval-aware trim for the exact
    logic; this function itself performs the simple timestamp filter and
    documents that the caller is responsible for closed-candle-only
    trimming on top of it.
    """
    if df.empty:
        return df
    idx = df.index
    if idx.tz is None:
        df = df.tz_localize("UTC") if hasattr(df, "tz_localize") else df
    return df[df.index <= cutoff]


def closed_candles_only(df: pd.DataFrame, cutoff: pd.Timestamp, bar_minutes: int) -> pd.DataFrame:
    """A bar is only "closed as of cutoff" if bar_open_time + bar_minutes <=
    cutoff. This is the actual no-lookahead guarantee: it discards any bar
    whose CLOSE would not yet have happened at the review moment, not just
    bars whose open is after the review moment."""
    if df.empty:
        return df
    close_times = df.index + pd.Timedelta(minutes=bar_minutes)
    return df[close_times <= cutoff]


def swings_and_events(df: pd.DataFrame, direction: str, margin: int) -> dict:
    """Runs the EXISTING, unmodified production primitives against a
    point-in-time-truncated dataframe. Nothing here is a new algorithm --
    every function called is the exact one already audited (see the audit
    report's Part 2)."""
    if df.empty or len(df) < (2 * margin + 5):
        return {"insufficient_data": True, "bars": len(df)}

    swings = scanner._find_swings(df, margin=margin)
    trend = scanner._get_trend(swings)
    bos_confirmed, bos_level = scanner._detect_bos(df, swings, direction.upper())
    bos_index = scanner._first_bos_close_index(df, swings, direction.upper(), bos_level) if bos_confirmed else None
    choch_suppress, choch_reason, bearish_lvl, bullish_lvl, choch_idx = scanner._detect_choch(swings, direction.upper())
    sweep_confirmed, sweep_level = scanner.detect_liquidity_sweep(df, swings, direction.upper())
    rejection_confirmed = scanner.detect_rejection(df, direction.upper(), sweep_level) if sweep_confirmed else False
    atr = scanner._compute_atr(df, period=14) if len(df) >= 15 else None

    event_type = None
    event_level = None
    event_index = None
    if bos_confirmed:
        event_type = "BOS"
        event_level = bos_level
        event_index = bos_index
    elif not choch_suppress and (bearish_lvl is not None or bullish_lvl is not None):
        # direction-favorable CHoCH exists and is NOT flagged as suppressing
        # our thesis direction -- i.e. it's a CHoCH pointed WITH direction.
        event_type = "CHoCH"
        event_level = bullish_lvl if direction.upper() == "LONG" else bearish_lvl
        event_index = choch_idx if choch_idx is not None and choch_idx >= 0 else None

    event_time = None
    event_bars_before_cutoff = None
    if event_index is not None and 0 <= event_index < len(df):
        event_time = df.index[event_index]
        event_bars_before_cutoff = len(df) - 1 - event_index

    close = float(df["Close"].iloc[-1])
    event_close_distance_atr = None
    if event_level is not None and atr and atr > 0 and event_index is not None:
        break_close = float(df["Close"].iloc[event_index])
        event_close_distance_atr = abs(break_close - event_level) / atr

    return {
        "insufficient_data": False,
        "bars": len(df),
        "margin": margin,
        "trend": trend,
        "bos_confirmed": bool(bos_confirmed),
        "choch_present": not (bearish_lvl is None and bullish_lvl is None),
        "choch_suppresses_direction": bool(choch_suppress),
        "choch_reason": choch_reason,
        "sweep_confirmed": bool(sweep_confirmed),
        "sweep_level": sweep_level,
        "rejection_confirmed": bool(rejection_confirmed),
        "atr": round(atr, 4) if atr else None,
        "event_type": event_type,
        "event_level": round(event_level, 4) if event_level is not None else None,
        "event_time": str(event_time) if event_time is not None else None,
        "event_bars_before_cutoff": event_bars_before_cutoff,
        "event_close_distance_atr": round(event_close_distance_atr, 3) if event_close_distance_atr is not None else None,
        "last_close": round(close, 4),
    }


def detector_A(analysis: dict) -> bool:
    """Close beyond relevant corrective swing (BOS or direction-favorable
    CHoCH, plain -- no quality filter)."""
    return bool(analysis.get("event_type") in ("BOS", "CHoCH"))


def detector_B(analysis: dict, min_atr_distance: float = 0.15) -> bool:
    """A + minimum decisive-close distance beyond the level, in ATR units."""
    if not detector_A(analysis):
        return False
    dist = analysis.get("event_close_distance_atr")
    return dist is not None and dist >= min_atr_distance


def detector_C(analysis: dict) -> bool:
    """A + sweep-and-reclaim occurred (liquidity swept, then rejected/
    reclaimed) ANYWHERE in the same window -- does not require the sweep to
    immediately precede the break."""
    return detector_A(analysis) and analysis.get("sweep_confirmed") and analysis.get("rejection_confirmed")


def detector_D(analysis: dict) -> bool:
    """BOS only (excludes CHoCH-only events) -- the stricter, decisive-close-
    through-a-confirmed-prior-swing-with-same-direction-body definition
    _detect_bos already enforces, with no separate distance/quality add-on."""
    return bool(analysis.get("bos_confirmed"))


DETECTORS = {
    "A_any_break": detector_A,
    "B_break_plus_0.15atr": detector_B,
    "C_sweep_and_break": detector_C,
    "D_bos_only": detector_D,
}


def run_for_ticker(
    ticker: str, direction: str, cutoff_iso: str, margins=(3, 4, 6), recent_windows=(40, 80, 150)
) -> dict:
    """recent_windows: in addition to running swing/event detection against
    the FULL point-in-time-truncated history (however many bars that is --
    up to 780 for a 60d/30m fetch), ALSO restrict the swing-computation
    INPUT to just the most recent N bars before cutoff. This directly tests
    Part 4's central question -- does constraining WHICH swings are even
    eligible (recent-window only, vs "anywhere in 60 days") change whether
    a detected break actually corresponds to the recent correction a human
    would describe, rather than an old, unrelated, already-resolved
    structural level from deep in the lookback."""
    cutoff = _to_utc(cutoff_iso)
    raw_30m = fetch_bars(ticker, interval="30m", period="60d")
    if raw_30m.empty:
        return {"ticker": ticker, "error": "no 30m data returned", "bars_fetched": 0}
    truncated = closed_candles_only(truncate_point_in_time(raw_30m, cutoff), cutoff, bar_minutes=30)

    per_margin = {}
    for margin in margins:
        per_margin[margin] = swings_and_events(truncated, direction, margin)

    per_window = {}
    for window in recent_windows:
        windowed = truncated.tail(window)
        # margin scales down for smaller windows so there's still room for
        # at least a few swing pivots -- an unscaled margin=4 on a 40-bar
        # window leaves almost nothing for _find_swings to work with.
        window_margin = 2 if window <= 40 else 3 if window <= 80 else 4
        per_window[window] = {
            **swings_and_events(windowed, direction, window_margin),
            "window_margin_used": window_margin,
        }

    return {
        "ticker": ticker,
        "cutoff": cutoff_iso,
        "bars_fetched_total": len(raw_30m),
        "bars_available_point_in_time": len(truncated),
        "first_bar": str(raw_30m.index[0]) if not raw_30m.empty else None,
        "last_bar_used": str(truncated.index[-1]) if not truncated.empty else None,
        "per_margin": per_margin,
        "per_recent_window": per_window,
    }


def main():
    results = []
    for ticker, label, decision, reviewed_at, note in LABELED_EXAMPLES:
        print(f"--- {ticker} (label={label}, decision={decision}) ---", file=sys.stderr, flush=True)
        r = run_for_ticker(ticker, "long", reviewed_at)
        r["label"] = label
        r["decision"] = decision
        r["note"] = note
        r["is_anchor"] = ticker in ANCHOR_TICKERS
        results.append(r)

    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
