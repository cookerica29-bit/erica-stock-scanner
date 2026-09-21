"""Morning watchlist screener.

Deliberately separate from the main grading engine (scanner.py's
quality/trade_eval/setupGrade machinery). This module answers one narrow
question — "which tickers are worth looking at today?" — using only
already-tested, completed-bar SMC structure (analyze_ticker's trend/BOS/CHoCH
detection) plus the 200-day SMA. It does not compute entries, stops, targets,
option strikes, or any live-quote-dependent execution status. Nothing here
feeds into or is fed by the A+/B+/C grading tiers.

Two buckets only:
  - "aligned": 200SMA bias, daily trend, and 4H trend/structure all agree.
  - "watch_reversal": 200SMA bias is clear but 4H structure has turned against
    it (a CHoCH/opposing structure forming) — early enough to be worth
    watching, not yet confirmed.
Anything else (no clear 200SMA bias yet, or nothing notable either way) is
omitted so the list stays short.
"""

from __future__ import annotations

from datetime import datetime, timezone

from scanner import WATCHLIST, _batch_download, _flatten_columns, analyze_ticker

MORNING_WATCHLIST_VERSION = "morning-watchlist-v2"

_DIRECTION_TO_STRUCTURE = {"LONG": "bullish", "SHORT": "bearish"}
_OPPOSITE_STRUCTURE = {"bullish": "bearish", "bearish": "bullish"}


def _sma200_bias(daily_df) -> tuple[str | None, float | None, float | None]:
    """Return (bias, price, sma200) from a daily OHLC frame, or (None, None, None)."""
    if daily_df is None or daily_df.empty or len(daily_df) < 200:
        return None, None, None
    close = daily_df["Close"].astype(float)
    sma200 = float(close.rolling(200).mean().iloc[-1])
    price = float(close.iloc[-1])
    if sma200 != sma200:  # NaN guard
        return None, None, None
    bias = "LONG" if price > sma200 else "SHORT"
    return bias, price, sma200


def _watchlist_entry_for_ticker(ticker: str, daily_raw, h4_raw) -> dict | None:
    daily_df = _flatten_columns(daily_raw) if daily_raw is not None else None
    sma_bias, price, sma200 = _sma200_bias(daily_df)
    if sma_bias is None:
        return None

    daily_result = analyze_ticker(ticker, _daily_df=daily_df, timeframe="1D")
    if not daily_result:
        return None

    h4_df = _flatten_columns(h4_raw) if h4_raw is not None else None
    h4_result = analyze_ticker(ticker, _daily_df=h4_df, timeframe="4H") if h4_df is not None and not h4_df.empty else None

    daily_trend = daily_result.get("trend") or "NEUTRAL"
    daily_structure = daily_result.get("structure") or "ranging"
    h4_trend = (h4_result or {}).get("trend") or "NEUTRAL"
    h4_structure = (h4_result or {}).get("structure") or "ranging"
    h4_price = (h4_result or {}).get("price")
    h4_ob_low = (h4_result or {}).get("ob_low")
    h4_ob_high = (h4_result or {}).get("ob_high")

    # Live-invalidation check (found via a real ticker, CLF, 2026-09-21):
    # daily_trend/h4_trend/h4_structure are all computed from completed-bar
    # swing/CHoCH detection, so a LONG thesis can still read "bullish
    # structure" even after the most recent 4H bar has already closed
    # BELOW the order block that structure depends on -- the labels don't
    # know the floor broke, only the raw price vs. level comparison does.
    # This is the same class of gap the original scanner audit found in
    # the old grading pipeline (completed-bar grade, no live-price
    # recheck); it applies here too since h4_result's own price/ob_low/
    # ob_high were already being computed and just never compared.
    structure_broken = False
    if h4_price is not None:
        if sma_bias == "LONG" and h4_ob_low is not None and h4_price < h4_ob_low:
            structure_broken = True
        elif sma_bias == "SHORT" and h4_ob_high is not None and h4_price > h4_ob_high:
            structure_broken = True

    bias_structure = _DIRECTION_TO_STRUCTURE[sma_bias]
    opposing_structure = _OPPOSITE_STRUCTURE[bias_structure]

    aligned = (
        daily_trend == sma_bias
        and daily_structure == bias_structure
        and h4_trend == sma_bias
        and h4_structure == bias_structure
        and not structure_broken
    )
    watch_reversal = (not aligned) and (h4_structure == opposing_structure or structure_broken)

    if not aligned and not watch_reversal:
        return None

    bucket = "aligned" if aligned else "watch_reversal"
    direction_word = "bullish" if sma_bias == "LONG" else "bearish"

    if aligned:
        reason = (
            f"Price {'above' if sma_bias == 'LONG' else 'below'} 200SMA "
            f"(${sma200:.2f}) with daily and 4H trend both {direction_word} — "
            f"structure confirms, no conflicting CHoCH."
        )
    elif structure_broken:
        ob_level = h4_ob_low if sma_bias == "LONG" else h4_ob_high
        reason = (
            f"Price still {'above' if sma_bias == 'LONG' else 'below'} 200SMA "
            f"(${sma200:.2f}, longer-term {direction_word} bias intact), but the 4H "
            f"order block (${ob_level:.2f}) has already broken — live price (${h4_price:.2f}) "
            f"is past the level this thesis depended on, structure labels haven't caught up yet."
        )
    else:
        reason = (
            f"Price still {'above' if sma_bias == 'LONG' else 'below'} 200SMA "
            f"(${sma200:.2f}, longer-term {direction_word} bias intact), but 4H "
            f"structure has turned {opposing_structure} — possible early reversal, "
            f"not yet confirmed."
        )

    return {
        "ticker": ticker,
        "bucket": bucket,
        "bias": sma_bias,
        "reason": reason,
        "price": round(price, 2),
        "sma200_daily": round(sma200, 2),
        "daily_trend": daily_trend,
        "daily_structure": daily_structure,
        "h4_trend": h4_trend,
        "h4_structure": h4_structure,
        "h4_bos_level": (h4_result or {}).get("bos_level"),
        "h4_ob_high": (h4_result or {}).get("ob_high"),
        "h4_ob_low": (h4_result or {}).get("ob_low"),
    }


def _frame_missing(frame) -> bool:
    return frame is None or getattr(frame, "empty", True)


def build_morning_watchlist(tickers: list[str] | None = None) -> dict:
    """Scan `tickers` (defaults to the full stock WATCHLIST) and bucket each
    into 'aligned' or 'watch_reversal', or omit it if there's nothing notable.
    Read-only, side-effect-free, does not touch the grading/caching machinery
    in scanner.py or main.py.

    `errors` reports only genuine fetch/processing failures -- a ticker with
    no clear signal today is just omitted, not an error. This distinction
    matters: without it, a ticker that failed to fetch from Yahoo looks
    identical to one that fetched fine and simply had nothing notable, and
    there's no way to tell "quiet market" from "the data feed broke."
    """
    symbols = tickers if tickers is not None else WATCHLIST
    aligned: list[dict] = []
    watch_reversal: list[dict] = []
    errors: list[dict] = []

    # Batch-fetch once for the whole watchlist (one multi-ticker request per
    # timeframe) rather than one network round-trip per ticker per timeframe.
    daily_frames = _batch_download(symbols, period="1y", interval="1d")
    h4_frames = _batch_download(symbols, period="60d", interval="4h")

    for ticker in symbols:
        daily_raw = daily_frames.get(ticker)
        h4_raw = h4_frames.get(ticker)

        if _frame_missing(daily_raw) and _frame_missing(h4_raw):
            errors.append({"ticker": ticker, "reason": "no_data", "detail": "Daily and 4H data both failed to fetch."})
            continue
        if _frame_missing(daily_raw):
            errors.append({"ticker": ticker, "reason": "no_daily_data", "detail": "Daily data failed to fetch (200SMA/daily trend unavailable)."})
            continue
        if _frame_missing(h4_raw):
            errors.append({"ticker": ticker, "reason": "no_4h_data", "detail": "4H data failed to fetch (structure/reversal check unavailable)."})
            continue

        try:
            entry = _watchlist_entry_for_ticker(ticker, daily_raw, h4_raw)
        except Exception as exc:  # noqa: BLE001 - one bad ticker must not kill the scan
            errors.append({"ticker": ticker, "reason": "processing_error", "detail": str(exc)})
            continue
        if entry is None:
            continue
        (aligned if entry["bucket"] == "aligned" else watch_reversal).append(entry)

    aligned.sort(key=lambda row: row["ticker"])
    watch_reversal.sort(key=lambda row: row["ticker"])
    errors.sort(key=lambda row: row["ticker"])

    return {
        "version": MORNING_WATCHLIST_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scanned_count": len(symbols),
        "aligned": aligned,
        "watch_reversal": watch_reversal,
        "errors": errors,
    }
