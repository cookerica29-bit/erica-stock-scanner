# v2 — BOS + Order Block strategy (replaces EMA pullback)
import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional

WATCHLIST = [
    # ── Airlines ──────────────────────────────────────────────────────────────
    "DAL", "UAL", "AAL", "JBLU",
    # ── Cruise ────────────────────────────────────────────────────────────────
    "CCL", "RCL", "NCLH",
    # ── Tech / Mega-cap ───────────────────────────────────────────────────────
    "NVDA", "AMD", "META", "TSLA", "AAPL", "MSFT", "GOOGL", "AMZN",
    # ── Semiconductors ────────────────────────────────────────────────────────
    "MU", "INTC", "QCOM", "AVGO", "TSM", "ARM",
    # ── Energy Majors ─────────────────────────────────────────────────────────
    "XOM", "CVX", "OXY",
    # ── Energy Mid-cap / Services ─────────────────────────────────────────────
    "DVN", "MRO", "HAL", "SLB", "FANG",
    # ── Big Banks ─────────────────────────────────────────────────────────────
    "JPM", "BAC", "GS", "MS", "WFC",
    # ── Payments / Fintech ────────────────────────────────────────────────────
    "V", "MA", "AXP", "PYPL", "SQ",
    # ── Retail / Consumer Discretionary ──────────────────────────────────────
    "WMT", "TGT", "COST", "HD", "LOW", "MCD", "SBUX", "NKE",
    # ── Consumer Staples ──────────────────────────────────────────────────────
    "PG", "KO", "PEP", "CL", "MO",
    # ── Healthcare / Pharma ───────────────────────────────────────────────────
    "UNH", "JNJ", "ABBV", "LLY", "PFE", "MRK", "CVS",
    # ── Biotech ───────────────────────────────────────────────────────────────
    "GILD", "REGN", "MRNA", "BIIB",
    # ── Utilities ─────────────────────────────────────────────────────────────
    "NEE", "DUK", "SO", "XEL",
    # ── Industrials / Defense ─────────────────────────────────────────────────
    "CAT", "DE", "HON", "BA", "GE", "LMT", "RTX",
    # ── Materials / Commodities ───────────────────────────────────────────────
    "FCX", "NEM", "AA", "CLF",
    # ── Broad ETFs ────────────────────────────────────────────────────────────
    "SPY", "QQQ", "IWM", "DIA",
    # ── Sector ETFs ───────────────────────────────────────────────────────────
    "XLF", "XLE", "XLV", "XLU", "XLK", "XLI", "XLB",
    # ── Commodity ETFs ────────────────────────────────────────────────────────
    "GLD", "SLV", "USO", "UNG",
]


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    prev_close = df["Close"].astype(float).shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])


def _next_friday(days_out: int = 37) -> datetime:
    target = datetime.now() + timedelta(days=days_out)
    days_to_friday = (4 - target.weekday()) % 7
    return target + timedelta(days=days_to_friday)


def _suggest_option(ticker: str, direction: str, entry: float) -> dict:
    expiry = _next_friday(37)
    dte = (expiry - datetime.now()).days
    if entry < 20:
        increment = 0.5
    elif entry < 100:
        increment = 1.0
    elif entry < 500:
        increment = 5.0
    else:
        increment = 10.0
    atm_strike = round(entry / increment) * increment
    return {
        "type":   "CALL" if direction == "LONG" else "PUT",
        "strike": atm_strike,
        "expiry": expiry.strftime("%Y-%m-%d"),
        "dte":    dte,
        "symbol": f"{ticker} {expiry.strftime('%b %d').upper()} ${atm_strike:.0f} {'C' if direction == 'LONG' else 'P'}",
    }


# ── Price Action Functions ────────────────────────────────────────────────────

def _find_swings(df: pd.DataFrame, margin: int = 3) -> list:
    highs = df["High"].values
    lows  = df["Low"].values
    swings = []
    for i in range(margin, len(df) - margin):
        window_h = highs[i - margin : i + margin + 1]
        window_l = lows[i  - margin : i + margin + 1]
        if highs[i] == window_h.max():
            swings.append({"index": i, "price": float(highs[i]), "type": "high"})
        elif lows[i] == window_l.min():
            swings.append({"index": i, "price": float(lows[i]),  "type": "low"})
    return swings


def _get_trend(swings: list) -> str:
    highs = [s for s in swings if s["type"] == "high"]
    lows  = [s for s in swings if s["type"] == "low"]
    if len(highs) < 2 or len(lows) < 2:
        return "NEUTRAL"
    hh = highs[-1]["price"] > highs[-2]["price"]
    hl = lows[-1]["price"]  > lows[-2]["price"]
    lh = highs[-1]["price"] < highs[-2]["price"]
    ll = lows[-1]["price"]  < lows[-2]["price"]
    if hh and hl:
        return "LONG"
    if lh and ll:
        return "SHORT"
    return "NEUTRAL"


def _detect_bos(df: pd.DataFrame, swings: list, direction: str, lookback: int = 40):
    n        = len(df)
    closes   = df["Close"].values
    highs_sw = [s for s in swings if s["type"] == "high"]
    lows_sw  = [s for s in swings if s["type"] == "low"]
    min_idx  = max(0, n - 1 - lookback)

    if direction == "LONG" and len(highs_sw) >= 2:
        prev_high = highs_sw[-2]
        for i in range(max(prev_high["index"] + 1, min_idx), n):
            if closes[i] > prev_high["price"]:
                return True, float(prev_high["price"])

    if direction == "SHORT" and len(lows_sw) >= 2:
        prev_low = lows_sw[-2]
        for i in range(max(prev_low["index"] + 1, min_idx), n):
            if closes[i] < prev_low["price"]:
                return True, float(prev_low["price"])

    return False, 0.0


def _find_order_block(df: pd.DataFrame, direction: str, swings: list) -> Optional[dict]:
    n      = len(df)
    opens  = df["Open"].values
    closes = df["Close"].values
    highs  = df["High"].values
    lows   = df["Low"].values

    if direction == "LONG":
        lows_sw = [s for s in swings if s["type"] == "low"]
        if not lows_sw:
            return None
        last_swing_low = lows_sw[-1]
        for i in range(n - 2, last_swing_low["index"] - 1, -1):
            if closes[i] < opens[i]:  # bearish candle before bullish impulse
                return {"high": float(highs[i]), "low": float(lows[i]), "index": i}

    if direction == "SHORT":
        highs_sw = [s for s in swings if s["type"] == "high"]
        if not highs_sw:
            return None
        last_swing_high = highs_sw[-1]
        for i in range(n - 2, last_swing_high["index"] - 1, -1):
            if closes[i] > opens[i]:  # bullish candle before bearish impulse
                return {"high": float(highs[i]), "low": float(lows[i]), "index": i}

    return None


def _macro_bias(price: float, df: pd.DataFrame) -> tuple:
    """
    Compute macro trend bias using 52-week high and 200-candle window high.
    Returns (bias, pct_from_52w_high, wk52_high, window_high).
      Macro Bearish : price > 15% below 52-week high
      Macro Bullish : price within 5% of 52-week high
      Macro Neutral : everything else
    """
    closes = df["Close"].astype(float)
    wk52_high    = float(closes.iloc[-252:].max()) if len(closes) >= 252 else float(closes.max())
    window_high  = float(closes.iloc[-200:].max()) if len(closes) >= 200 else float(closes.max())
    pct_from_52w = (wk52_high - price) / wk52_high if wk52_high > 0 else 0.0

    if pct_from_52w > 0.15:
        bias = "Macro Bearish"
    elif pct_from_52w < 0.05:
        bias = "Macro Bullish"
    else:
        bias = "Macro Neutral"

    return bias, round(pct_from_52w, 3), round(wk52_high, 2), round(window_high, 2)


def _market_structure(
    swings: list,
    price: float,
    df: pd.DataFrame,
    macro_bias: str = "",
    window_high: float = 0.0,
) -> tuple:
    """
    Classify structure as 'bullish', 'bearish', or 'ranging' using a weighted vote system.
    Returns (classification: str, reasons: list[str]).

    Weights:
      +4 bearish/bullish : macro bias (52w high check) — hard override
      +2 bearish/bullish : price vs EMA200
      +2 bearish/bullish : recent swing high comparison (LH vs HH)
      +2 bearish/bullish : recent swing low comparison (LL vs HL)
      +2 bearish/bullish : 3-swing LH/LL or HH/HL sequence
      +2 bearish         : price > 15% below 200-candle window high
    """
    reasons: list = []
    bearish_score = 0
    bullish_score = 0

    highs = [s["price"] for s in swings if s["type"] == "high"]
    lows  = [s["price"] for s in swings if s["type"] == "low"]

    # ── Macro bias override (weight 4) ───────────────────────────────────────
    if macro_bias == "Macro Bearish":
        bearish_score += 4
        reasons.append(f"macro bias: price >15% below 52w high [bearish +4]")
    elif macro_bias == "Macro Bullish":
        bullish_score += 4
        reasons.append(f"macro bias: price within 5% of 52w high [bullish +4]")

    # ── 200-candle window high check (weight 2) ───────────────────────────────
    if window_high > 0:
        pct_below_window = (window_high - price) / window_high
        if pct_below_window > 0.15:
            bearish_score += 2
            reasons.append(
                f"price ${price:.2f} is {pct_below_window:.1%} below "
                f"200-bar high ${window_high:.2f} [bearish +2]"
            )

    # ── EMA200 bias (weight 2) ────────────────────────────────────────────────
    if len(df) >= 200:
        ema200 = float(df["Close"].ewm(span=200, adjust=False).mean().iloc[-1])
        if price < ema200:
            bearish_score += 2
            reasons.append(f"price ${price:.2f} below EMA200 ${ema200:.2f} [bearish +2]")
        else:
            bullish_score += 2
            reasons.append(f"price ${price:.2f} above EMA200 ${ema200:.2f} [bullish +2]")

    # ── Recent swing high comparison (weight 2) ───────────────────────────────
    if len(highs) >= 2:
        if highs[-1] < highs[-2]:
            bearish_score += 2
            reasons.append(f"LH: {highs[-1]:.2f} < {highs[-2]:.2f} [bearish +2]")
        else:
            bullish_score += 2
            reasons.append(f"HH: {highs[-1]:.2f} >= {highs[-2]:.2f} [bullish +2]")

    # ── Recent swing low comparison (weight 2) ────────────────────────────────
    if len(lows) >= 2:
        if lows[-1] < lows[-2]:
            bearish_score += 2
            reasons.append(f"LL: {lows[-1]:.2f} < {lows[-2]:.2f} [bearish +2]")
        else:
            bullish_score += 2
            reasons.append(f"HL: {lows[-1]:.2f} >= {lows[-2]:.2f} [bullish +2]")

    # ── 3-swing sequence (weight 2) ───────────────────────────────────────────
    if (len(highs) >= 3 and highs[-3] > highs[-2] > highs[-1]
            and len(lows) >= 2 and lows[-2] > lows[-1]):
        bearish_score += 2
        reasons.append("3-swing LH/LL sequence confirmed [bearish +2]")
    elif (len(lows) >= 3 and lows[-3] < lows[-2] < lows[-1]
            and len(highs) >= 2 and highs[-2] < highs[-1]):
        bullish_score += 2
        reasons.append("3-swing HH/HL sequence confirmed [bullish +2]")

    if bearish_score > bullish_score:
        result = "bearish"
    elif bullish_score > bearish_score:
        result = "bullish"
    else:
        result = "ranging"

    reasons.append(f"→ scores: bearish={bearish_score} bullish={bullish_score} → {result}")
    return result, reasons


def _detect_choch(swings: list, direction: str) -> tuple:
    """
    Scan ALL swing pairs for CHoCH events and return the most recent one.
    Most recent CHoCH takes priority over earlier reversals (handles short-term bounces).
    Returns (suppress, reason, bearish_choch_level, last_choch_bar_index).
      suppress=True             → this CHoCH conflicts with `direction`, filter the setup.
      bearish_choch_level       → the prior swing-low that was broken (None if no bearish CHoCH).
      last_choch_bar_index      → bar index of the most recent CHoCH (-1 if none found).
    """
    highs = [s for s in swings if s["type"] == "high"]
    lows  = [s for s in swings if s["type"] == "low"]

    last_bearish_idx   = -1
    last_bullish_idx   = -1
    bearish_reason     = ""
    bullish_reason     = ""
    bearish_choch_lvl  = None   # prior swing low that was broken = CHoCH level

    # Bearish CHoCH: LL confirmed after a LH between the two lows
    for i in range(1, len(lows)):
        if lows[i]["price"] < lows[i - 1]["price"]:
            between_highs = [h for h in highs
                             if lows[i - 1]["index"] < h["index"] < lows[i]["index"]]
            prior_highs   = [h for h in highs if h["index"] < lows[i - 1]["index"]]
            if (between_highs and prior_highs
                    and between_highs[-1]["price"] < prior_highs[-1]["price"]):
                if lows[i]["index"] > last_bearish_idx:
                    last_bearish_idx  = lows[i]["index"]
                    bearish_choch_lvl = lows[i - 1]["price"]   # the level that was broken
                    bearish_reason = (
                        f"bearish CHoCH at swing-low {lows[i]['price']:.2f} "
                        f"(broke prior low {lows[i-1]['price']:.2f} after LH "
                        f"{between_highs[-1]['price']:.2f} < {prior_highs[-1]['price']:.2f})"
                    )

    # Bullish CHoCH: HH confirmed after a HL between the two highs
    for i in range(1, len(highs)):
        if highs[i]["price"] > highs[i - 1]["price"]:
            between_lows = [l for l in lows
                            if highs[i - 1]["index"] < l["index"] < highs[i]["index"]]
            prior_lows   = [l for l in lows if l["index"] < highs[i - 1]["index"]]
            if (between_lows and prior_lows
                    and between_lows[-1]["price"] > prior_lows[-1]["price"]):
                if highs[i]["index"] > last_bullish_idx:
                    last_bullish_idx = highs[i]["index"]
                    bullish_reason = (
                        f"bullish CHoCH at swing-high {highs[i]['price']:.2f} "
                        f"(broke prior high {highs[i-1]['price']:.2f} after HL "
                        f"{between_lows[-1]['price']:.2f} > {prior_lows[-1]['price']:.2f})"
                    )

    # No CHoCH found at all
    if last_bearish_idx == -1 and last_bullish_idx == -1:
        return False, "no CHoCH detected", None, None, -1

    # Track the bullish CHoCH level (prior swing high that was broken)
    bullish_choch_lvl: Optional[float] = None
    if last_bullish_idx >= 0:
        for i in range(1, len(highs)):
            if highs[i]["index"] == last_bullish_idx:
                bullish_choch_lvl = highs[i - 1]["price"]
                break

    # Most recent CHoCH wins regardless of short-term counter-moves
    if last_bearish_idx >= last_bullish_idx:
        if direction == "LONG":
            return True, f"[SUPPRESS] most recent CHoCH is bearish → {bearish_reason}", bearish_choch_lvl, bullish_choch_lvl, last_bearish_idx
        return False, f"bearish CHoCH present but not suppressing {direction} → {bearish_reason}", bearish_choch_lvl, bullish_choch_lvl, last_bearish_idx
    else:
        if direction == "SHORT":
            return True, f"[SUPPRESS] most recent CHoCH is bullish → {bullish_reason}", bearish_choch_lvl, bullish_choch_lvl, last_bullish_idx
        return False, f"bullish CHoCH present but not suppressing {direction} → {bullish_reason}", bearish_choch_lvl, bullish_choch_lvl, last_bullish_idx


def _safe_ratio(numerator: float, denominator: float, fallback: float = 0.0) -> float:
    return numerator / denominator if denominator else fallback


def _grade_from_score(score: int) -> str:
    if score >= 90:
        return "A+"
    if score >= 84:
        return "A"
    if score >= 78:
        return "B+"
    if score >= 70:
        return "B"
    if score >= 62:
        return "C+"
    if score >= 54:
        return "C"
    return "D"


def _apply_grade_caps(
    score: int,
    *,
    bos_confirmed: bool,
    ob: Optional[dict],
    in_ob: bool,
    near_ob: bool,
    location: str,
    cleanliness: str,
    touches: int,
    bos_extension: Optional[float],
    room_to_target: str,
) -> tuple:
    cap = 100
    reasons = []

    if not bos_confirmed:
        cap = min(cap, 58)
        reasons.append("No confirmed BOS caps this at C.")
    if not ob:
        cap = min(cap, 66)
        reasons.append("No order block caps this at C+.")
    if ob and not (in_ob or near_ob):
        cap = min(cap, 72)
        reasons.append("Price has not returned to the OB, so this is capped at B.")
    if "late for fresh" in location:
        cap = min(cap, 84)
        reasons.append("Late location caps this at A.")
    if cleanliness == "Choppy / overlapping":
        cap = min(cap, 82)
        reasons.append("Choppy candles cap this at B+.")
    elif cleanliness == "Readable but mixed":
        cap = min(cap, 91)
        reasons.append("Mixed candles keep this below A+.")
    if touches == 2:
        cap = min(cap, 88)
        reasons.append("Tapped once keeps this below A+.")
    elif touches >= 3:
        cap = min(cap, 76)
        reasons.append("Repeated OB taps cap this at B.")
    if bos_extension is not None:
        if bos_extension > 4:
            cap = min(cap, 78)
            reasons.append("Very extended from BOS caps this at B+.")
        elif bos_extension > 3:
            cap = min(cap, 84)
            reasons.append("Extended from BOS caps this at A.")
    if room_to_target.startswith("Crowded"):
        cap = min(cap, 74)
        reasons.append("Crowded room to target caps this at B.")

    final_score = min(score, cap)
    grade_note = reasons[0] if reasons else "No grade cap applied."
    return final_score, cap, grade_note


def _latest_swing_range(swings: list) -> Optional[dict]:
    highs = [s for s in swings if s["type"] == "high"]
    lows = [s for s in swings if s["type"] == "low"]
    if not highs or not lows:
        return None
    high = max(highs[-3:], key=lambda s: s["price"])
    low = min(lows[-3:], key=lambda s: s["price"])
    if high["price"] <= low["price"]:
        return None
    return {"high": high["price"], "low": low["price"]}


def _location_read(price: float, direction: str, swings: list) -> tuple:
    swing_range = _latest_swing_range(swings)
    if not swing_range:
        return "Unclear", None

    low = swing_range["low"]
    high = swing_range["high"]
    percentile = max(0.0, min(1.0, _safe_ratio(price - low, high - low)))

    if percentile >= 0.67:
        zone = "Premium"
    elif percentile <= 0.33:
        zone = "Discount"
    else:
        zone = "Midrange"

    if direction == "LONG" and zone == "Premium":
        read = "Premium - late for fresh longs"
    elif direction == "LONG" and zone == "Discount":
        read = "Discount - better long location"
    elif direction == "SHORT" and zone == "Discount":
        read = "Discount - late for fresh shorts"
    elif direction == "SHORT" and zone == "Premium":
        read = "Premium - better short location"
    else:
        read = zone

    return read, round(percentile * 100, 1)


def _strict_location(price: float, swings: list) -> tuple:
    swing_range = _latest_swing_range(swings)
    if not swing_range:
        return "MIDRANGE", None

    low = swing_range["low"]
    high = swing_range["high"]
    percentile = max(0.0, min(1.0, _safe_ratio(price - low, high - low)))

    if percentile <= 0.18 or percentile >= 0.82:
        location = "AT EXTREME"
    elif percentile <= 0.35:
        location = "NEAR DISCOUNT"
    elif percentile >= 0.65:
        location = "NEAR PREMIUM"
    else:
        location = "MIDRANGE"
    return location, round(percentile * 100, 1)


def _structure_quality(trend: str, bos_confirmed: bool, cleanliness: str) -> str:
    if cleanliness == "Choppy / overlapping" or trend == "NEUTRAL":
        return "CHOPPY / INTERNAL ONLY"
    if bos_confirmed:
        return "CLEAN BOS"
    return "DEVELOPING"


def _directional_candles(df: pd.DataFrame, direction: str, lookback: int = 3) -> pd.DataFrame:
    recent = df.tail(lookback)
    if direction == "LONG":
        return recent[recent["Close"] > recent["Open"]]
    if direction == "SHORT":
        return recent[recent["Close"] < recent["Open"]]
    return recent.iloc[0:0]


def detect_displacement(df: pd.DataFrame, atr: float, direction: str, bos_confirmed: bool) -> tuple:
    if not bos_confirmed or direction not in ("LONG", "SHORT") or len(df) < 3 or atr <= 0:
        return "NONE", 0.0

    recent = df.tail(3)
    directional = _directional_candles(df, direction, lookback=3)
    if directional.empty:
        return "NONE", 0.0

    bodies = (directional["Close"] - directional["Open"]).abs()
    avg_body_atr = float(bodies.mean() / atr)
    last = recent.iloc[-1]
    last_is_directional = (
        (direction == "LONG" and float(last["Close"]) > float(last["Open"]))
        or (direction == "SHORT" and float(last["Close"]) < float(last["Open"]))
    )
    last_range_atr = float((last["High"] - last["Low"]) / atr) if last_is_directional else 0.0
    directional_majority = len(directional) >= 2

    if directional_majority and (avg_body_atr >= 0.7 or last_range_atr >= 1.2):
        return "STRONG", round(max(avg_body_atr, last_range_atr), 2)
    if avg_body_atr >= 0.35 or last_range_atr >= 0.75:
        return "WEAK", round(max(avg_body_atr, last_range_atr), 2)
    return "NONE", round(max(avg_body_atr, last_range_atr), 2)


def _displacement_read(df: pd.DataFrame, atr: float, bos_confirmed: bool, direction: str = "NEUTRAL") -> tuple:
    return detect_displacement(df, atr, direction, bos_confirmed)


def detect_liquidity_sweep(df: pd.DataFrame, swings: list, direction: str, lookback: int = 12) -> tuple:
    if direction not in ("LONG", "SHORT") or len(df) < 3:
        return False, None
    recent = df.tail(lookback)
    highs = [s for s in swings if s["type"] == "high"]
    lows = [s for s in swings if s["type"] == "low"]

    if direction == "LONG" and lows:
        level = lows[-1]["price"]
        swept = bool((recent["Low"] < level).any())
        return bool(swept), float(level)
    if direction == "SHORT" and highs:
        level = highs[-1]["price"]
        swept = bool((recent["High"] > level).any())
        return bool(swept), float(level)
    return False, None


def _detect_liquidity_sweep(df: pd.DataFrame, swings: list, direction: str, lookback: int = 12) -> bool:
    swept, _ = detect_liquidity_sweep(df, swings, direction, lookback=lookback)
    return swept


def detect_rejection(df: pd.DataFrame, direction: str, sweep_level: Optional[float], lookback: int = 5) -> bool:
    if direction not in ("LONG", "SHORT") or sweep_level is None or len(df) < 2:
        return False

    recent = df.tail(lookback)
    for _, candle in recent.iterrows():
        high = float(candle["High"])
        low = float(candle["Low"])
        open_ = float(candle["Open"])
        close = float(candle["Close"])
        body = abs(close - open_)
        candle_range = high - low
        if candle_range <= 0:
            continue

        upper_wick = high - max(open_, close)
        lower_wick = min(open_, close) - low
        if direction == "LONG":
            closed_back_above = low < sweep_level and close > sweep_level
            lower_wick_failure = low < sweep_level and lower_wick >= max(body * 1.25, candle_range * 0.35)
            if closed_back_above or (lower_wick_failure and close > open_):
                return True
        elif direction == "SHORT":
            closed_back_below = high > sweep_level and close < sweep_level
            upper_wick_failure = high > sweep_level and upper_wick >= max(body * 1.25, candle_range * 0.35)
            if closed_back_below or (upper_wick_failure and close < open_):
                return True
    return False


def detect_structure_break(df: pd.DataFrame, swings: list, direction: str, lookback: int = 40) -> tuple:
    return _detect_bos(df, swings, direction, lookback=lookback)


def _nearest_target(price: float, direction: str, swings: list, fallback: float = 0.0) -> Optional[float]:
    highs = sorted([s["price"] for s in swings if s["type"] == "high" and s["price"] > price])
    lows = sorted([s["price"] for s in swings if s["type"] == "low" and s["price"] < price], reverse=True)
    if direction == "LONG":
        if highs:
            return float(highs[0])
        return float(fallback) if fallback and fallback > price else None
    if direction == "SHORT":
        if lows:
            return float(lows[0])
        return float(fallback) if fallback and fallback < price else None
    return None


def _room_to_target(
    price: float,
    direction: str,
    swings: list,
    entry: Optional[float] = None,
    stop: Optional[float] = None,
    fallback_target: float = 0.0,
) -> dict:
    target = _nearest_target(price, direction, swings, fallback_target)
    if target is None or direction not in ("LONG", "SHORT"):
        return {
            "target": None,
            "percent_to_target": None,
            "estimated_rr": None,
            "blocked": False,
            "clear": False,
            "label": "No clean structural target",
        }

    distance = (target - price) if direction == "LONG" else (price - target)
    pct = _safe_ratio(distance, price) * 100
    estimated_rr = None
    blocked = False
    if entry is not None and stop is not None:
        risk = abs(entry - stop)
        reward = (target - entry) if direction == "LONG" else (entry - target)
        if risk > 0 and reward > 0:
            estimated_rr = reward / risk
            blocked = estimated_rr < 2.0
        elif risk > 0:
            estimated_rr = 0.0
            blocked = True

    return {
        "target": round(target, 2),
        "percent_to_target": round(max(0.0, pct), 1),
        "estimated_rr": round(estimated_rr, 2) if estimated_rr is not None else None,
        "blocked": blocked,
        "clear": distance > 0 and not blocked,
        "label": "Blocked: RR < 1:2" if blocked else "Clear path to target",
    }


def _cap_quality_to_c(quality: dict, reason: str) -> dict:
    capped = dict(quality or {})
    if capped.get("score", 0) > 58:
        capped["score"] = 58
    capped["grade"] = "C"
    capped["grade_cap"] = min(capped.get("grade_cap", 100), 58)
    prior = capped.get("grade_note", "")
    capped["grade_note"] = reason if not prior or prior == "No grade cap applied." else f"{reason} {prior}"
    return capped


def _build_trade_stage_eval(
    *,
    df: pd.DataFrame,
    swings: list,
    trend: str,
    price: float,
    atr: float,
    bos_confirmed: bool,
    ob: Optional[dict],
    in_ob: bool,
    near_ob: bool,
    cleanliness: str,
    entry: Optional[float] = None,
    stop: Optional[float] = None,
    fallback_target: float = 0.0,
    macro_conflict: bool = False,
    context_conflict: bool = False,
) -> dict:
    location, location_pct = _strict_location(price, swings)
    bos_confirmed, bos_level = detect_structure_break(df, swings, trend)
    structure_quality = _structure_quality(trend, bos_confirmed, cleanliness)
    displacement, displacement_score = detect_displacement(df, atr, trend, bos_confirmed)
    sweep_taken, sweep_level = detect_liquidity_sweep(df, swings, trend)
    rejection_confirmed = detect_rejection(df, trend, sweep_level)
    room = _room_to_target(price, trend, swings, entry, stop, fallback_target)

    if location_pct is None:
        valid_zone = False
        preferred_fib_zone = False
    elif trend == "LONG":
        valid_zone = location_pct <= 50.0
        preferred_fib_zone = 21.0 <= location_pct <= 50.0
    elif trend == "SHORT":
        valid_zone = location_pct >= 50.0
        preferred_fib_zone = 50.0 <= location_pct <= 79.0
    else:
        valid_zone = False
        preferred_fib_zone = False

    htf_bias_clear = trend in ("LONG", "SHORT")
    htf_aligned = htf_bias_clear and not macro_conflict and not context_conflict
    rr_ok = room.get("estimated_rr") is not None and room.get("estimated_rr") >= 2.0
    room_clear = room.get("clear") is True and not room.get("blocked")
    structure_event_forming = bool(bos_confirmed or sweep_taken or rejection_confirmed or ob or near_ob or in_ob)

    if bos_confirmed and ob and (in_ob or near_ob):
        setup_type = "CONTINUATION: BOS + retest"
    elif sweep_taken and rejection_confirmed and displacement in ("WEAK", "STRONG"):
        setup_type = "REVERSAL: sweep + rejection + displacement"
    else:
        setup_type = "NONE"

    missing_for_a_plus = []
    if not sweep_taken:
        missing_for_a_plus.append("Needs liquidity sweep")
    if not rejection_confirmed:
        missing_for_a_plus.append("Needs rejection")
    if displacement != "STRONG":
        missing_for_a_plus.append("Needs strong displacement")
    if not bos_confirmed:
        missing_for_a_plus.append("Needs real BOS")
    if not valid_zone:
        missing_for_a_plus.append("Needs valid premium/discount location")
    if not rr_ok or room.get("blocked"):
        missing_for_a_plus.append("Needs 1:2 RR")
    if not htf_aligned:
        missing_for_a_plus.append("Needs HTF alignment")

    coaching = []
    if not valid_zone:
        coaching.append("Wait for valid premium/discount location.")
    elif not preferred_fib_zone and location_pct is not None:
        coaching.append("Best A+ location is usually the 50-79 fib retracement zone.")
    if not sweep_taken:
        coaching.append("Wait for a liquidity sweep before calling this A+.")
    if not rejection_confirmed:
        coaching.append("Wait for rejection back through the swept level.")
    if not bos_confirmed:
        coaching.append("Wait for strong candle close beyond structure.")
    if displacement == "NONE":
        coaching.append("Wait for impulse move showing control.")
    if room.get("blocked"):
        coaching.append("Wait for better entry or wider target.")
    if not coaching:
        coaching.append("Wait for all A+ conditions to remain true through the trigger candle.")

    no_trade_reasons = []
    if not htf_bias_clear:
        no_trade_reasons.append("No clear HTF bias")
    if not valid_zone:
        no_trade_reasons.append("Poor premium/discount location")
    if structure_quality == "CHOPPY / INTERNAL ONLY":
        no_trade_reasons.append("Choppy/internal structure")
    if room.get("blocked"):
        no_trade_reasons.append("RR < 1:2")
    if macro_conflict or context_conflict:
        no_trade_reasons.append("Macro/context conflict")
    if setup_type == "NONE" and not structure_event_forming:
        no_trade_reasons.append("No setup type or forming structure event")

    trigger_confirmed = bool(
        sweep_taken
        and rejection_confirmed
        and displacement == "STRONG"
        and bos_confirmed
    )

    a_plus_ready = (
        htf_aligned
        and valid_zone
        and structure_quality == "CLEAN BOS"
        and sweep_taken
        and rejection_confirmed
        and displacement == "STRONG"
        and setup_type in ("CONTINUATION: BOS + retest", "REVERSAL: sweep + rejection + displacement")
        and room_clear
        and rr_ok
        and trigger_confirmed
    )

    worth_watching = (
        htf_bias_clear
        and valid_zone
        and structure_event_forming
        and not room.get("blocked")
        and (room_clear or room.get("target") is not None or room.get("estimated_rr") is None)
        and 1 <= len(missing_for_a_plus) <= 3
    )

    # B+ TRADEABLE: BOS + OB at zone + at least 1 confirming signal.
    # Relaxed location: up to 70% percentile for longs, down to 30% for shorts.
    # A+ requires strict discount/premium (<=50% / >=50%); B+ allows midrange approach.
    b_plus_zone = (
        location_pct is not None and (
            (trend == "LONG"  and location_pct <= 70.0) or
            (trend == "SHORT" and location_pct >= 30.0)
        )
    )
    directional_signals = sum([
        bool(sweep_taken),
        bool(rejection_confirmed),
        displacement != "NONE",
        structure_quality == "CLEAN BOS",
    ])

    # RR is informational for B+ — options RR differs from underlying price RR.
    # Show the RR value on the card; do not gate B+ on it.
    b_plus_tradeable = (
        htf_bias_clear
        and bos_confirmed
        and ob is not None
        and (in_ob or near_ob)
        and not macro_conflict
        and not context_conflict
        and b_plus_zone
        and directional_signals >= 1
    )

    if a_plus_ready:
        trade_stage = "A+ READY"
    elif b_plus_tradeable:
        trade_stage = "B+ TRADEABLE"
    elif no_trade_reasons:
        trade_stage = "RANGE / NO TRADE"
    elif worth_watching:
        trade_stage = "BUILDING / WATCHLIST"
    else:
        trade_stage = "RANGE / NO TRADE"

    return {
        "trade_stage": trade_stage,
        "stage_badge": (
            "🟢 A+ READY"         if trade_stage == "A+ READY"
            else "🔵 B+ TRADEABLE"    if trade_stage == "B+ TRADEABLE"
            else "🟡 BUILDING / WATCHLIST" if trade_stage == "BUILDING / WATCHLIST"
            else "🔴 RANGE / NO TRADE"
        ),
        "b_plus_tradeable": b_plus_tradeable,
        "location": location,
        "location_percentile": location_pct,
        "structure_quality": structure_quality,
        "displacement": displacement,
        "displacement_score": displacement_score,
        "setup_type": setup_type,
        "room_to_target": room,
        "valid_zone": valid_zone,
        "preferred_fib_zone": preferred_fib_zone,
        "sweep_taken": sweep_taken,
        "sweep_level": round(sweep_level, 2) if sweep_level is not None else None,
        "rejection_confirmed": rejection_confirmed,
        "htf_bias_clear": htf_bias_clear,
        "htf_aligned": htf_aligned,
        "structure_event_forming": structure_event_forming,
        "trigger_confirmed": trigger_confirmed,
        "missing_for_a_plus": missing_for_a_plus,
        "no_trade_reasons": no_trade_reasons,
        "a_plus_ready": a_plus_ready,
        "coaching": coaching,
    }


def _ob_touch_count(df: pd.DataFrame, ob: Optional[dict]) -> int:
    if not ob:
        return 0
    touches = 0
    for i in range(ob["index"] + 1, len(df)):
        high = float(df["High"].iloc[i])
        low = float(df["Low"].iloc[i])
        if high >= ob["low"] and low <= ob["high"]:
            touches += 1
    return touches


def _cleanliness_read(df: pd.DataFrame, lookback: int = 12) -> tuple:
    closes = df["Close"].astype(float).iloc[-lookback - 1:]
    if len(closes) < 4:
        return "Unclear", 0.0
    net_move = abs(float(closes.iloc[-1] - closes.iloc[0]))
    path = float(closes.diff().abs().sum())
    efficiency = _safe_ratio(net_move, path)
    if efficiency >= 0.58:
        return "Clean impulse", round(efficiency, 2)
    if efficiency <= 0.32:
        return "Choppy / overlapping", round(efficiency, 2)
    return "Readable but mixed", round(efficiency, 2)


def _build_chart_coach(
    df: pd.DataFrame,
    swings: list,
    direction: str,
    price: float,
    atr: float,
    bos_confirmed: bool,
    bos_level: float,
    ob: Optional[dict],
    in_ob: bool,
    near_ob: bool,
    risk: Optional[float] = None,
    entry: Optional[float] = None,
) -> dict:
    score = 40
    warnings = []

    has_trend = direction in ("LONG", "SHORT")
    if has_trend:
        score += 12
    if bos_confirmed:
        score += 14
    if ob:
        score += 14
    if in_ob:
        score += 12
    elif near_ob:
        score += 7

    location, location_pct = _location_read(price, direction, swings)
    if "late for fresh" in location:
        score -= 8
        warnings.append("Location is late; avoid chasing without acceptance or a cleaner pullback.")
    elif "better" in location:
        score += 5

    cleanliness, efficiency = _cleanliness_read(df)
    if cleanliness == "Clean impulse":
        score += 7
    elif cleanliness == "Choppy / overlapping":
        score -= 7
        warnings.append("Recent candles are overlapping; structure may be harder to trust.")

    touches = _ob_touch_count(df, ob)
    if not ob:
        freshness = "No order block"
    elif touches <= 1:
        freshness = "Fresh OB"
        score += 6
    elif touches == 2:
        freshness = "Tapped once"
    else:
        freshness = "Heavily revisited"
        score -= 8
        warnings.append("Order block has been revisited multiple times; reaction quality matters more.")

    bos_extension = None
    if bos_confirmed and bos_level and atr:
        bos_extension = abs(price - bos_level) / atr
        if bos_extension > 3:
            score -= 7
            warnings.append("Price is extended from the BOS level; wait for a reset or proof of continuation.")

    room_to_target = "Unclear"
    if risk and entry and bos_level:
        if direction == "LONG" and bos_level > entry:
            room_r = (bos_level - entry) / risk
            room_to_target = "Crowded overhead" if room_r < 1.2 else "Clear path to BOS"
        elif direction == "SHORT" and bos_level < entry:
            room_r = (entry - bos_level) / risk
            room_to_target = "Crowded below" if room_r < 1.2 else "Clear path to BOS"
        else:
            room_r = 2.0
            room_to_target = "BOS already cleared"
        if room_to_target.startswith("Crowded"):
            score -= 6
            warnings.append("Nearby structure may limit room before the first reaction area.")

    raw_score = int(max(0, min(100, round(score))))
    score, grade_cap, grade_note = _apply_grade_caps(
        raw_score,
        bos_confirmed=bos_confirmed,
        ob=ob,
        in_ob=in_ob,
        near_ob=near_ob,
        location=location,
        cleanliness=cleanliness,
        touches=touches,
        bos_extension=bos_extension,
        room_to_target=room_to_target,
    )

    if not bos_confirmed:
        coach_note = "Trend has not produced a confirmed break yet; keep this on watch instead of forcing it."
        training_prompt = "What exact candle close would prove structure has actually broken?"
    elif not ob:
        coach_note = "BOS is present, but the scanner has not found a clean order block to anchor risk."
        training_prompt = "Where is the last opposite-color candle before the displacement, and is it meaningful?"
    elif not (in_ob or near_ob):
        coach_note = "The idea has direction and structure, but price has not returned to the order block yet."
        training_prompt = "What price would bring this from interesting to actionable?"
    elif "late for fresh" in location:
        coach_note = "Direction is valid, but location is stretched; wait for acceptance or a cleaner pullback."
        training_prompt = "Are you seeing continuation acceptance, or are you buying/selling into the reaction area?"
    elif cleanliness == "Choppy / overlapping":
        coach_note = "The setup passes rules, but the path is messy; demand a cleaner reaction before committing."
        training_prompt = "Which candle would show decisive control instead of overlap?"
    else:
        coach_note = "Structure, location, and order-block context are aligned enough to study closely."
        training_prompt = "What would invalidate the order-block defense before entry?"

    if not warnings:
        warnings.append("No major visual warning; still wait for the chart to confirm the plan.")

    return {
        "score": score,
        "raw_score": raw_score,
        "grade": _grade_from_score(score),
        "grade_cap": grade_cap,
        "grade_note": grade_note,
        "location": location,
        "location_percentile": location_pct,
        "cleanliness": cleanliness,
        "efficiency": efficiency,
        "freshness": freshness,
        "touches": touches,
        "room_to_target": room_to_target,
        "bos_extension_atr": round(bos_extension, 2) if bos_extension is not None else None,
        "warning": warnings[0],
        "warnings": warnings,
        "coach_note": coach_note,
        "training_prompt": training_prompt,
    }


# ── Batch data helpers ────────────────────────────────────────────────────────

def _batch_download(tickers: list, period: str, interval: str) -> dict:
    """
    Download OHLCV data for multiple tickers in a single yfinance call.
    Returns {ticker: DataFrame}.  Falls back gracefully on any per-ticker error.
    """
    if not tickers:
        return {}
    try:
        raw = yf.download(
            tickers, period=period, interval=interval,
            progress=False, auto_adjust=True, group_by="ticker",
        )
    except Exception as e:
        print(f"[batch_download] error: {e}")
        return {}

    result: dict = {}
    single = len(tickers) == 1

    for t in tickers:
        try:
            if single:
                df = raw.copy()
                if isinstance(df.columns, pd.MultiIndex):
                    # (PriceType, Ticker) layout — drop the ticker level
                    df.columns = df.columns.get_level_values(0)
            else:
                df = raw[t].copy()  # group_by='ticker' gives (Ticker, PriceType)
            df = df.dropna(how="all")
            if len(df) >= 10:
                result[t] = df
        except Exception:
            pass

    return result


# ── Main Analysis ─────────────────────────────────────────────────────────────

def analyze_ticker(
    ticker: str,
    _daily_df: Optional[pd.DataFrame] = None,
    _weekly_df: Optional[pd.DataFrame] = None,
) -> Optional[dict]:
    try:
        if _daily_df is not None:
            df = _flatten_columns(_daily_df.copy())
        else:
            raw = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
            df  = _flatten_columns(raw)

        if len(df) < 50:
            return None

        df    = df.astype(float)
        close = df["Close"]
        price = float(close.iloc[-1])

        # Ticker quality filters
        if price < 5:
            return None
        avg_dollar_vol = float((df["Close"] * df["Volume"]).iloc[-20:].mean())
        if avg_dollar_vol < 5_000_000:
            return None
        daily_range_pct = (float(df["High"].iloc[-1]) - float(df["Low"].iloc[-1])) / price * 100
        if daily_range_pct < 1.0:
            return None

        atr = _compute_atr(df)
        rsi = round(_compute_rsi(close), 1)

        # Price action
        swings        = _find_swings(df)
        trend         = _get_trend(swings)
        bos_confirmed = False
        bos_level     = 0.0
        ob            = None
        in_ob = near_ob = False

        if trend != "NEUTRAL":
            bos_confirmed, bos_level = _detect_bos(df, swings, trend)
            if bos_confirmed:
                ob = _find_order_block(df, trend, swings)
                if ob:
                    in_ob = ob["low"] <= price <= ob["high"]
                    near_ob = not in_ob and (
                        (trend == "LONG"  and price < ob["high"] and price > ob["low"] - atr) or
                        (trend == "SHORT" and price > ob["low"]  and price < ob["high"] + atr)
                    )

        # ── Macro bias (52-week high context) ────────────────────────────────────
        macro_bias, pct_from_52w, wk52_high, window_high = _macro_bias(price, df)
        macro_label = (
            f"📉 {macro_bias} ({pct_from_52w:.0%} below 52w high ${wk52_high:.2f})"
            if macro_bias == "Macro Bearish"
            else f"📈 {macro_bias} (within {pct_from_52w:.0%} of 52w high ${wk52_high:.2f})"
            if macro_bias == "Macro Bullish"
            else f"〰️ {macro_bias} ({pct_from_52w:.0%} below 52w high ${wk52_high:.2f})"
        )
        print(f"[{ticker}] {macro_label}")

        # ── Macro bias hard block (LONG only) ────────────────────────────────
        # Suppress buying into macro downtrends. Never block shorts based on macro
        # bias — short signals use local structure detection only.
        macro_block = macro_bias == "Macro Bearish" and trend == "LONG"
        if macro_block:
            print(f"[{ticker}] Macro Bearish override — LONG marked no trade")

        # Market structure defaults
        structure = "ranging"
        choch = False
        struct_label = "🟡 Ranging"
        struct_aligned = False
        struct_note = "🟡 Ranging market — proceed with extra confirmation"

        choch_block = False
        if trend in ("LONG", "SHORT"):
            # Weekly EMA context (informational only — not a hard block)
            if _weekly_df is not None:
                weekly = _flatten_columns(_weekly_df.copy()).astype(float)
            else:
                w_raw = yf.download(ticker, period="2y", interval="1wk", progress=False, auto_adjust=True)
                weekly = _flatten_columns(w_raw).astype(float)
            if len(weekly) >= 50:
                w_close = weekly["Close"]
                w_e20 = float(w_close.ewm(span=20, adjust=False).mean().iloc[-1])
                w_e50 = float(w_close.ewm(span=50, adjust=False).mean().iloc[-1])
                print(
                    f"[{ticker}] weekly EMA20={w_e20:.2f} EMA50={w_e50:.2f} "
                    f"({'bearish cross' if w_e20 < w_e50 else 'bullish cross'})"
                )

            # Structure context — 200-candle lookback, margin=5 for confirmed swings
            # Informational only — structure/CHoCH labels show on cards but do NOT block signals.
            # Primary entry gate = BOS confirmed + price at/near OB.
            htf_df     = df.tail(200).reset_index(drop=True)
            htf_swings = _find_swings(htf_df, margin=5)
            structure, struct_reasons = _market_structure(
                htf_swings, price, df, macro_bias=macro_bias, window_high=window_high
            )
            choch, choch_reason, bearish_choch_lvl, bullish_choch_lvl, choch_bar_idx = _detect_choch(htf_swings, trend)

            # Update structure label if CHoCH overrides the vote
            if bearish_choch_lvl is not None and price < bearish_choch_lvl:
                structure = "bearish"
                print(
                    f"[{ticker}] note: price {price:.2f} below bearish CHoCH level "
                    f"{bearish_choch_lvl:.2f} → structure reads bearish"
                )
            elif "bearish CHoCH" in choch_reason:
                structure = "bearish"
            elif "bullish CHoCH" in choch_reason:
                structure = "bullish"

            print(f"[{ticker}] trend={trend} structure={structure} choch={choch}")
            for r in struct_reasons:
                print(f"  {r}")
            print(f"  choch: {choch_reason}")

            # Price-based CHoCH block: only suppress if price has NOT cleared the CHoCH level.
            # A bearish CHoCH that price has already recovered above is no longer blocking.
            # A bullish CHoCH that price has already fallen below is no longer blocking.
            if choch and trend == "LONG":
                # Bearish CHoCH blocks longs only while price <= the broken swing-low level
                choch_block = bearish_choch_lvl is not None and price <= bearish_choch_lvl
                if choch and not choch_block:
                    print(f"[{ticker}] bearish CHoCH present but price ${price:.2f} > CHoCH level ${bearish_choch_lvl:.2f} — not blocking")
                elif choch_block:
                    print(f"[{ticker}] bearish CHoCH conflict — LONG marked no trade")
            elif choch and trend == "SHORT":
                # Bullish CHoCH blocks shorts only while price >= the broken swing-high level
                choch_block = bullish_choch_lvl is not None and price >= bullish_choch_lvl
                if choch and not choch_block:
                    print(f"[{ticker}] bullish CHoCH present but price ${price:.2f} < CHoCH level ${bullish_choch_lvl:.2f} — not blocking")
                elif choch_block:
                    print(f"[{ticker}] bullish CHoCH conflict — SHORT marked no trade")
            else:
                choch_block = False

            struct_label = (
                ("🔴 Bearish ChoCH" if trend == "LONG" else "🟢 Bullish ChoCH") if choch
                else ("🔴 Bearish Structure" if structure == "bearish"
                      else ("🟢 Bullish Structure" if structure == "bullish" else "🟡 Ranging"))
            )
            struct_aligned = (
                (trend == "LONG" and structure == "bullish")
                or (trend == "SHORT" and structure == "bearish")
            )
            struct_note = (
                "✅ Structure aligned with " + trend + " setup" if struct_aligned
                else "🟡 Ranging market — proceed with extra confirmation" if structure == "ranging"
                else "⚠️ Counter-trend setup — structure is " + structure + ", proceed with caution"
            )

        quality = _build_chart_coach(
            df=df,
            swings=swings,
            direction=trend,
            price=price,
            atr=atr,
            bos_confirmed=bos_confirmed,
            bos_level=bos_level,
            ob=ob,
            in_ob=in_ob,
            near_ob=near_ob,
        )
        trade_eval = _build_trade_stage_eval(
            df=df,
            swings=swings,
            trend=trend,
            price=price,
            atr=atr,
            bos_confirmed=bos_confirmed,
            ob=ob,
            in_ob=in_ob,
            near_ob=near_ob,
            cleanliness=quality.get("cleanliness", "Unclear"),
            fallback_target=window_high,
            macro_conflict=macro_block,
            context_conflict=choch_block,
        )
        if trade_eval["no_trade_reasons"] and not trade_eval.get("b_plus_tradeable"):
            quality = _cap_quality_to_c(quality, "Strict scout rules cap this at C.")
        if macro_block or choch_block:
            reason = "Macro/CHoCH conflict caps this at C."
            quality = _cap_quality_to_c(quality, reason)
            if macro_block:
                trade_eval["no_trade_reasons"].append("Macro bearish conflict")
                trade_eval["coaching"].append("Wait for macro context to stop fighting the long idea.")
            if choch_block:
                trade_eval["no_trade_reasons"].append("Counter CHoCH conflict")
                trade_eval["coaching"].append("Wait for structure to realign after the CHoCH.")
            trade_eval["trade_stage"] = "RANGE / NO TRADE"
            trade_eval["a_plus_ready"] = False
            trade_eval["b_plus_tradeable"] = False

        # ── Near-miss: has direction but setup is incomplete ──────────────────
        if trend == "NEUTRAL" or not bos_confirmed or (not in_ob and not near_ob):
            return {
                "ticker":        ticker,
                "direction":     trend if trend != "NEUTRAL" else None,
                "price":         round(price, 2),
                "atr":           round(atr, 2),
                "rsi":           rsi,
                "trend":         trend,
                "bos_confirmed": bos_confirmed,
                "bos_level":     round(bos_level, 2) if bos_level else None,
                "ob_high":       round(ob["high"], 2) if ob else None,
                "ob_low":        round(ob["low"],  2) if ob else None,
                "in_ob":         in_ob,
                "near_ob":       near_ob,
                "quality":        quality,
                "trade_eval":     trade_eval,
                "structure":      structure,
                "structureLabel": struct_label,
                "structureNote":  struct_note,
                "macroBias":      macro_bias,
                "macroLabel":     macro_label,
                "wk52High":       wk52_high,
                "pctFromHigh":    pct_from_52w,
                "setup_status":   "DEVELOPING" if trend != "NEUTRAL" else "SKIPPED",
                "scannedAt":      datetime.utcnow().isoformat() + "Z",
            }

        # ── Qualified setup: BOS confirmed + price at/near OB ────────────────
        ob_mid = (ob["high"] + ob["low"]) / 2
        entry  = round(ob_mid, 2)
        sl     = round(ob["low"] - 0.5 * atr, 2) if trend == "LONG" else round(ob["high"] + 0.5 * atr, 2)
        risk   = round(abs(entry - sl), 2)
        tp1    = round(entry + 2 * risk, 2) if trend == "LONG" else round(entry - 2 * risk, 2)
        tp2    = round(entry + 3 * risk, 2) if trend == "LONG" else round(entry - 3 * risk, 2)
        tp3    = round(entry + 4 * risk, 2) if trend == "LONG" else round(entry - 4 * risk, 2)

        option = _suggest_option(ticker, trend, entry)

        checklist = {
            "trendConfirmed":  True,
            "bosConfirmed":    bos_confirmed,
            "obFound":         ob is not None,
            "priceAtOb":       in_ob or near_ob,
            "structureAligned": struct_aligned,
            "chochClear":      not choch,
        }

        quality = _build_chart_coach(
            df=df,
            swings=swings,
            direction=trend,
            price=price,
            atr=atr,
            bos_confirmed=bos_confirmed,
            bos_level=bos_level,
            ob=ob,
            in_ob=in_ob,
            near_ob=near_ob,
            risk=risk,
            entry=entry,
        )
        trade_eval = _build_trade_stage_eval(
            df=df,
            swings=swings,
            trend=trend,
            price=price,
            atr=atr,
            bos_confirmed=bos_confirmed,
            ob=ob,
            in_ob=in_ob,
            near_ob=near_ob,
            cleanliness=quality.get("cleanliness", "Unclear"),
            entry=entry,
            stop=sl,
            fallback_target=window_high,
            macro_conflict=macro_block,
            context_conflict=choch_block,
        )
        if trade_eval["no_trade_reasons"] and not trade_eval.get("b_plus_tradeable"):
            quality = _cap_quality_to_c(quality, "Strict scout rules cap this at C.")
        if macro_block or choch_block:
            reason = "Macro/CHoCH conflict caps this at C."
            quality = _cap_quality_to_c(quality, reason)
            if macro_block:
                trade_eval["no_trade_reasons"].append("Macro bearish conflict")
                trade_eval["coaching"].append("Wait for macro context to stop fighting the long idea.")
            if choch_block:
                trade_eval["no_trade_reasons"].append("Counter CHoCH conflict")
                trade_eval["coaching"].append("Wait for structure to realign after the CHoCH.")
            trade_eval["trade_stage"] = "RANGE / NO TRADE"
            trade_eval["a_plus_ready"] = False
            trade_eval["b_plus_tradeable"] = False

        # Gate: only publish entry/option levels for A+ or B+ tradeable setups.
        # Everything else stays in near-miss / DEVELOPING.
        if not trade_eval["a_plus_ready"] and not trade_eval.get("b_plus_tradeable"):
            return {
                "ticker":        ticker,
                "direction":     trend,
                "price":         round(price, 2),
                "atr":           round(atr, 2),
                "rsi":           rsi,
                "trend":         trend,
                "bos_confirmed": bos_confirmed,
                "bos_level":     round(bos_level, 2),
                "ob_high":       round(ob["high"], 2),
                "ob_low":        round(ob["low"],  2),
                "in_ob":         in_ob,
                "near_ob":       near_ob,
                "quality":        quality,
                "trade_eval":     trade_eval,
                "structure":      structure,
                "structureLabel": struct_label,
                "structureNote":  struct_note,
                "macroBias":      macro_bias,
                "macroLabel":     macro_label,
                "wk52High":       wk52_high,
                "pctFromHigh":    pct_from_52w,
                "setup_status":   "DEVELOPING",
                "scannedAt":      datetime.utcnow().isoformat() + "Z",
            }

        return {
            "ticker":        ticker,
            "direction":     trend,
            "price":         round(price, 2),
            "atr":           round(atr, 2),
            "rsi":           rsi,
            "trend":         trend,
            "bos_confirmed": bos_confirmed,
            "bos_level":     round(bos_level, 2),
            "ob_high":       round(ob["high"], 2),
            "ob_low":        round(ob["low"],  2),
            "in_ob":         in_ob,
            "near_ob":       near_ob,
            "entry":         entry,
            "sl":            sl,
            "tp1":           tp1,
            "tp2":           tp2,
            "tp3":           tp3,
            "risk":          risk,
            "option":        option,
            "checklist":      checklist,
            "quality":        quality,
            "trade_eval":     trade_eval,
            "structure":      structure,
            "structureLabel": struct_label,
            "structureNote":  struct_note,
            "macroBias":      macro_bias,
            "macroLabel":     macro_label,
            "wk52High":       wk52_high,
            "pctFromHigh":    pct_from_52w,
            "setup_status":   "QUALIFIED",
            "scannedAt":      datetime.utcnow().isoformat() + "Z",
        }

    except Exception as e:
        print(f"[scanner] {ticker} error: {e}")
        return None


def debug_ticker(ticker: str) -> dict:
    """
    Run every filter step on a single ticker and return the full reasoning.
    Used by /api/debug/{ticker}.
    """
    out: dict = {"ticker": ticker.upper(), "filters": [], "passed": False}
    try:
        raw = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
        df  = _flatten_columns(raw)
        if len(df) < 50:
            out["filters"].append({"step": "data", "result": "FAIL", "reason": "< 50 bars"})
            return out
        df    = df.astype(float)
        price = float(df["Close"].iloc[-1])
        atr   = _compute_atr(df)

        # Quality gates
        if price < 5:
            out["filters"].append({"step": "price", "result": "FAIL", "reason": f"price ${price:.2f} < $5"})
            return out
        avg_dv = float((df["Close"] * df["Volume"]).iloc[-20:].mean())
        if avg_dv < 5_000_000:
            out["filters"].append({"step": "volume", "result": "FAIL", "reason": f"avg dollar vol ${avg_dv/1e6:.1f}M < $5M"})
            return out
        dr = (float(df["High"].iloc[-1]) - float(df["Low"].iloc[-1])) / price * 100
        if dr < 1.0:
            out["filters"].append({"step": "range", "result": "FAIL", "reason": f"daily range {dr:.2f}% < 1%"})
            return out
        out["price"] = round(price, 2)
        out["atr"]   = round(atr, 2)

        # Trend
        swings = _find_swings(df)
        trend  = _get_trend(swings)
        out["trend"] = trend
        out["filters"].append({"step": "trend", "result": "OK", "reason": trend})

        # BOS
        bos_confirmed, bos_level = (False, 0.0)
        ob = None
        in_ob = near_ob = False
        if trend != "NEUTRAL":
            bos_confirmed, bos_level = _detect_bos(df, swings, trend)
            if bos_confirmed:
                ob = _find_order_block(df, trend, swings)
                if ob:
                    in_ob   = ob["low"] <= price <= ob["high"]
                    near_ob = not in_ob and (
                        (trend == "LONG"  and price < ob["high"] and price > ob["low"] - atr) or
                        (trend == "SHORT" and price > ob["low"]  and price < ob["high"] + atr)
                    )
        out["bos_confirmed"] = bos_confirmed
        out["bos_level"]     = round(bos_level, 2) if bos_level else None
        out["ob"]            = {"high": round(ob["high"], 2), "low": round(ob["low"], 2)} if ob else None
        out["in_ob"]         = in_ob
        out["near_ob"]       = near_ob
        out["filters"].append({
            "step": "bos_ob",
            "result": "OK" if bos_confirmed else "WARN",
            "reason": f"BOS={'yes' if bos_confirmed else 'no'} OB={'yes' if ob else 'no'} in_ob={in_ob} near_ob={near_ob}",
        })

        # Macro bias
        macro_bias, pct_from_52w, wk52_high, window_high = _macro_bias(price, df)
        out["macro_bias"]     = macro_bias
        out["wk52_high"]      = wk52_high
        out["pct_from_52w"]   = f"{pct_from_52w:.1%}"
        macro_block = macro_bias == "Macro Bearish" and trend == "LONG"
        out["filters"].append({
            "step": "macro_bias",
            "result": "FAIL" if macro_block else "OK",
            "reason": (
                f"{macro_bias} ({pct_from_52w:.1%} below 52w high ${wk52_high:.2f})"
                + (" → LONG suppressed" if macro_block else "")
            ),
        })
        if macro_block:
            return out

        # Weekly EMA
        w_raw  = yf.download(ticker, period="2y", interval="1wk", progress=False, auto_adjust=True)
        weekly = _flatten_columns(w_raw).astype(float)
        w_e20 = w_e50 = None
        if len(weekly) >= 50:
            w_close = weekly["Close"]
            w_e20 = round(float(w_close.ewm(span=20, adjust=False).mean().iloc[-1]), 2)
            w_e50 = round(float(w_close.ewm(span=50, adjust=False).mean().iloc[-1]), 2)
        out["weekly_ema20"] = w_e20
        out["weekly_ema50"] = w_e50
        out["filters"].append({
            "step": "weekly_ema",
            "result": "INFO",
            "reason": (
                f"EMA20={w_e20} EMA50={w_e50} "
                f"({'bearish cross' if w_e20 and w_e20 < w_e50 else 'bullish cross'})"
                if w_e20 else "not enough weekly data"
            ),
        })

        # Local structure + CHoCH
        htf_df     = df.tail(200).reset_index(drop=True)
        htf_swings = _find_swings(htf_df, margin=5)
        structure, struct_reasons = _market_structure(
            htf_swings, price, df, macro_bias=macro_bias, window_high=window_high
        )
        choch, choch_reason, bearish_choch_lvl, bullish_choch_lvl, choch_bar_idx = _detect_choch(htf_swings, trend)
        if bearish_choch_lvl is not None and price < bearish_choch_lvl:
            structure = "bearish"
        elif "bearish CHoCH" in choch_reason:
            structure = "bearish"
        elif "bullish CHoCH" in choch_reason:
            structure = "bullish"
        out["structure"]        = structure
        out["choch"]            = choch
        out["choch_reason"]     = choch_reason
        out["struct_reasons"]   = struct_reasons
        out["filters"].append({
            "step": "structure",
            "result": "INFO",
            "reason": f"structure={structure} choch={choch} | {choch_reason}",
        })

        # Final gate: BOS + OB
        if not bos_confirmed:
            out["filters"].append({"step": "final", "result": "FAIL", "reason": "no BOS confirmed"})
            return out
        if not (in_ob or near_ob):
            out["filters"].append({
                "step": "final", "result": "FAIL",
                "reason": f"price not at/near OB (ob={out['ob']})",
            })
            return out

        out["passed"] = True
        out["filters"].append({"step": "final", "result": "PASS", "reason": "BOS + OB confirmed"})
    except Exception as e:
        out["filters"].append({"step": "error", "result": "ERROR", "reason": str(e)})
    return out


def scan_all(watchlist: list = WATCHLIST, max_workers: int = 12) -> tuple:
    # ── Step 1: batch-download all OHLCV data (2 network calls total) ─────────
    daily_data  = _batch_download(watchlist, period="1y",  interval="1d")
    weekly_data = _batch_download(watchlist, period="2y",  interval="1wk")

    # ── Step 2: parallel-process each ticker (pure CPU/logic, no I/O) ─────────
    rows, near_miss = [], []

    def _process(ticker: str):
        return analyze_ticker(
            ticker,
            _daily_df=daily_data.get(ticker),
            _weekly_df=weekly_data.get(ticker),
        )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process, t): t for t in watchlist}
        for future in as_completed(futures):
            r = future.result()
            if r is None:
                continue
            if r.get("setup_status") == "QUALIFIED":
                rows.append(r)
            else:
                near_miss.append(r)

    rows.sort(key=lambda x: x.get("quality", {}).get("score", 0), reverse=True)
    near_miss.sort(key=lambda x: x.get("quality", {}).get("score", 0), reverse=True)
    return rows, near_miss


def scan_ticker(ticker: str) -> Optional[dict]:
    return analyze_ticker(ticker.upper())
