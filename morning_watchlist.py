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

Plus a "spotlight" list: aligned/watch_reversal tickers ALSO showing
unusually high volume today. This is deliberately a simple statistical
comparison (today's volume vs a trailing average), not a structural
pattern-match -- the Pullbacks tab was removed the same day this was added
specifically because a structural "which correction matters" judgment call
didn't hold up in live use (see project memory). Relative volume needs no
such judgment: it's one number vs another.

Plus an EXPERIMENTAL "liquidity_sweeps" list: aligned/watch_reversal tickers
where a real equal-highs/equal-lows liquidity pool (via the third-party
`smartmoneyconcepts` library) was swept in the last few 4H bars, price
hasn't run far from that level since (<= 1 ATR), split into "confirmed"
(price has already closed back through the level -- a real rejection) vs.
"forming" (swept but not yet rejected). This directly answers "closer to
entry", unlike prior attempts this same day (see project memory
entry_proximity_attempts) which were either wrong (a corrective-leg
detector) or uninformative (a static ATR-distance number). Marked
experimental because it has only been spot-checked against a handful of
real tickers, not run across many real mornings yet -- treat it as
something to watch and calibrate, not something to trust outright.
"""

from __future__ import annotations

import os

os.environ.setdefault("SMC_CREDIT", "0")  # suppress the library's stdout banner

from datetime import datetime, timezone

from smartmoneyconcepts import smc

from scanner import WATCHLIST, _batch_download, _compute_atr, _flatten_columns, analyze_ticker

MORNING_WATCHLIST_VERSION = "morning-watchlist-v4"

_DIRECTION_TO_STRUCTURE = {"LONG": "bullish", "SHORT": "bearish"}
_OPPOSITE_STRUCTURE = {"bullish": "bearish", "bearish": "bullish"}
RELATIVE_VOLUME_LOOKBACK_DAYS = 20
RELATIVE_VOLUME_SPOTLIGHT_THRESHOLD = 1.5  # 50% above the 20-day average

LIQUIDITY_SWEEP_LOOKBACK_PERIOD = "1y"  # 4H swing/liquidity detection needs real history, not just 60d
LIQUIDITY_SWEEP_SWING_LENGTH = 5
LIQUIDITY_SWEEP_RANGE_PERCENT = 0.02  # how close prior highs/lows must be to count as "equal"
LIQUIDITY_SWEEP_RECENT_BARS = 3  # how many bars ago the sweep itself must have happened
LIQUIDITY_SWEEP_MAX_DISTANCE_ATR = 1.0  # how far price may have already run from the level


def _relative_volume(daily_df) -> float | None:
    """Today's volume so far vs the average of the prior N full trading
    days. NOT time-of-day adjusted -- this will read low early in the
    session simply because less of today has happened yet, and become a
    fair comparison later in the day. Deliberately kept this simple for a
    first version rather than estimating a full-day pace projection.
    """
    if daily_df is None or "Volume" not in daily_df.columns:
        return None
    volume = daily_df["Volume"].astype(float)
    if len(volume) < RELATIVE_VOLUME_LOOKBACK_DAYS + 1:
        return None
    today_volume = volume.iloc[-1]
    avg_volume = volume.iloc[-(RELATIVE_VOLUME_LOOKBACK_DAYS + 1):-1].mean()
    if not avg_volume or avg_volume != avg_volume:  # zero or NaN guard
        return None
    return float(today_volume / avg_volume)


def _sma200_bias(daily_df) -> tuple[str | None, float | None, float | None]:
    """Return (bias, price, sma200) from a daily OHLC frame, or (None, None, None)."""
    if daily_df is None or daily_df.empty:
        return None, None, None
    # Yahoo sometimes returns a stub row for the current session (real
    # Volume, but NaN Open/High/Low/Close) during some after-hours window
    # before the daily bar is finalized. Found 2026-09-21 evening: this
    # silently zeroed out the entire Aligned/Watch list (not just this
    # ticker) because a NaN last-close poisons the rolling mean and trips
    # the NaN guard below. Drop NaN closes so a not-yet-finalized "today"
    # row is treated as not-yet-existing rather than corrupting the result.
    close = daily_df["Close"].astype(float).dropna()
    if len(close) < 200:
        return None, None, None
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

    relative_volume = _relative_volume(daily_df)
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
        "relative_volume_today": round(relative_volume, 2) if relative_volume is not None else None,
    }


def _frame_missing(frame) -> bool:
    return frame is None or getattr(frame, "empty", True)


def _find_liquidity_sweep(ticker: str, df) -> dict | None:
    """Look for a recent, still-fresh equal-highs/equal-lows liquidity sweep
    on this ticker's 4H data. Returns the single most relevant hit (most
    recent, ties broken by closest to the level) or None.

    Liquidity==1 from the library is built from equal HIGHS (buy-side
    liquidity) -- sweeping it (a new high) is the classic setup for a
    BEARISH reversal. Liquidity==-1 is equal LOWS (sell-side liquidity) --
    sweeping it (a new low) sets up a BULLISH reversal. The library's own
    "bullish"/"bearish" naming just labels which swing type formed the
    pool, not the expected direction after a sweep -- confirmed by reading
    its source (smc.py's liquidity()) after an initial pass here had the
    direction backwards.
    """
    min_bars = LIQUIDITY_SWEEP_SWING_LENGTH * 2 + 10
    if df is None or len(df) < min_bars:
        return None

    smc_df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    smc_df = smc_df[["open", "high", "low", "close", "volume"]].reset_index(drop=True)

    try:
        swing_highs_lows = smc.swing_highs_lows(smc_df, swing_length=LIQUIDITY_SWEEP_SWING_LENGTH)
        liq = smc.liquidity(smc_df, swing_highs_lows, range_percent=LIQUIDITY_SWEEP_RANGE_PERCENT)
    except Exception:  # noqa: BLE001 - experimental signal, must not kill the scan
        return None

    atr = _compute_atr(df)
    if not atr:
        return None

    last_idx = len(smc_df) - 1
    current_close = float(df["Close"].iloc[-1])
    swept_recently = liq[liq["Swept"] > 0]

    best = None
    for _, row in swept_recently.iterrows():
        swept_bar = int(row["Swept"])
        bars_ago = last_idx - swept_bar
        if bars_ago > LIQUIDITY_SWEEP_RECENT_BARS:
            continue

        level = float(row["Level"])
        distance_atr = abs(current_close - level) / atr
        if distance_atr > LIQUIDITY_SWEEP_MAX_DISTANCE_ATR:
            continue  # real sweep, but price already ran away from it -- stale

        candidate = {
            "level": round(level, 2),
            "bars_ago": bars_ago,
            "swept_at": df.index[swept_bar].isoformat() if swept_bar < len(df) else None,
            "distance_atr": round(distance_atr, 2),
            "direction": "bearish" if row["Liquidity"] == 1 else "bullish",
            "rejected": current_close < level if row["Liquidity"] == 1 else current_close > level,
            "current_price_4h": round(current_close, 2),
        }
        if best is None or (candidate["bars_ago"], candidate["distance_atr"]) < (best["bars_ago"], best["distance_atr"]):
            best = candidate

    return best


def _attach_liquidity_sweeps(entries: list[dict]) -> list[dict]:
    """Second pass, only for tickers that already made the Aligned/Watch
    list: fetch a full year of 4H data (swing/liquidity detection needs
    real history) and look for a fresh sweep. Returns a separate list --
    does not mutate the aligned/watch_reversal entries -- since this is an
    experimental, independent signal, not part of the aligned/reversal
    classification itself. Never raises; a ticker whose sweep-check fails
    just doesn't show up here.
    """
    if not entries:
        return []
    tickers = [e["ticker"] for e in entries]
    h4_1y_frames = _batch_download(tickers, period=LIQUIDITY_SWEEP_LOOKBACK_PERIOD, interval="4h")

    results = []
    for entry in entries:
        raw = h4_1y_frames.get(entry["ticker"])
        if _frame_missing(raw):
            continue
        df = _flatten_columns(raw)
        try:
            hit = _find_liquidity_sweep(entry["ticker"], df)
        except Exception:  # noqa: BLE001 - experimental signal, must not kill the scan
            hit = None
        if hit is None:
            continue

        status = "confirmed" if hit["rejected"] else "forming"
        pool_type = "equal-lows (sell-side)" if hit["direction"] == "bullish" else "equal-highs (buy-side)"
        if hit["rejected"]:
            outcome = "price has already closed back through the level, confirming the reversal."
        else:
            outcome = "reversal not yet confirmed -- watch for a close back through the level."
        reason = (
            f"Swept an {pool_type} liquidity level at ${hit['level']:.2f} "
            f"{hit['bars_ago']} bar(s) ago; price is still close ({hit['distance_atr']:.2f} ATR away) -- {outcome}"
        )

        results.append({
            "ticker": entry["ticker"],
            "bias": entry["bias"],
            "bucket": entry["bucket"],
            "price": hit["current_price_4h"],
            "direction": hit["direction"],
            "level": hit["level"],
            "bars_ago": hit["bars_ago"],
            "swept_at": hit["swept_at"],
            "distance_atr": hit["distance_atr"],
            "status": status,
            "reason": reason,
        })

    results.sort(key=lambda r: (r["bars_ago"], r["distance_atr"]))
    return results


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

    spotlight = sorted(
        (
            row for row in (aligned + watch_reversal)
            if row.get("relative_volume_today") is not None
            and row["relative_volume_today"] >= RELATIVE_VOLUME_SPOTLIGHT_THRESHOLD
        ),
        key=lambda row: row["relative_volume_today"],
        reverse=True,
    )

    liquidity_sweeps = _attach_liquidity_sweeps(aligned + watch_reversal)

    aligned.sort(key=lambda row: row["ticker"])
    watch_reversal.sort(key=lambda row: row["ticker"])
    errors.sort(key=lambda row: row["ticker"])

    return {
        "version": MORNING_WATCHLIST_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scanned_count": len(symbols),
        "aligned": aligned,
        "watch_reversal": watch_reversal,
        "spotlight": spotlight,
        "liquidity_sweeps": liquidity_sweeps,
        "errors": errors,
    }
