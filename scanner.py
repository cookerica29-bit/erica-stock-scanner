import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

WATCHLIST = [
    "DAL", "UAL", "AAL", "CCL", "RCL", "IBM",
    "AAPL", "MSFT", "SPY", "QQQ", "AMZN", "NVDA", "NCLH",
]


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns that yfinance returns for single-ticker downloads."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


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
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    return float(atr.iloc[-1])


def _next_friday(days_out: int = 37) -> datetime:
    """Return the Friday closest to `days_out` days from today."""
    target = datetime.now() + timedelta(days=days_out)
    days_to_friday = (4 - target.weekday()) % 7
    return target + timedelta(days=days_to_friday)


def _suggest_option(ticker: str, direction: str, entry: float) -> dict:
    """Suggest an ATM option strike with 30-45 DTE."""
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


def analyze_ticker(ticker: str) -> Optional[dict]:
    try:
        daily_raw  = yf.download(ticker, period="1y",  interval="1d",  progress=False, auto_adjust=True)
        weekly_raw = yf.download(ticker, period="2y",  interval="1wk", progress=False, auto_adjust=True)

        daily  = _flatten_columns(daily_raw)
        weekly = _flatten_columns(weekly_raw)

        if len(daily) < 60 or len(weekly) < 20:
            return None

        # ── Daily indicators ─────────────────────────────────────────────────────
        close  = daily["Close"].astype(float)
        ema20  = _compute_ema(close, 20)
        ema50  = _compute_ema(close, 50)
        ema200 = _compute_ema(close, 200)
        atr    = _compute_atr(daily.astype(float))
        rsi    = _compute_rsi(close)

        price = float(close.iloc[-1])
        e20   = float(ema20.iloc[-1])
        e50   = float(ema50.iloc[-1])
        e200  = float(ema200.iloc[-1])

        # ── Daily trend ───────────────────────────────────────────────────────────
        if price > e20 and e20 > e50 and e50 > e200:
            direction = "LONG"
        elif price < e20 and e20 < e50 and e50 < e200:
            direction = "SHORT"
        else:
            return None  # no clean trend alignment

        # ── No-man's-land: price sandwiched between EMA20 and EMA50 ──────────────
        if direction == "LONG"  and e50 < price < e20:
            return None
        if direction == "SHORT" and e20 < price < e50:
            return None

        # ── Pullback to EMA20: price within 0.5×ATR ──────────────────────────────
        dist_to_ema20 = abs(price - e20)
        pullback = dist_to_ema20 <= 0.5 * atr
        if not pullback:
            return None

        # ── RSI filter ────────────────────────────────────────────────────────────
        rsi_ok = (40 <= rsi <= 68) if direction == "LONG" else (35 <= rsi <= 60)
        if not rsi_ok:
            return None

        # ── Weekly HTF alignment ──────────────────────────────────────────────────
        w_close = weekly["Close"].astype(float)
        w_ema20 = _compute_ema(w_close, 20)
        w_ema50 = _compute_ema(w_close, 50)
        w_price = float(w_close.iloc[-1])
        w_e20   = float(w_ema20.iloc[-1])
        w_e50   = float(w_ema50.iloc[-1])

        htf_aligned = (
            (w_price > w_e20 and w_price > w_e50) if direction == "LONG"
            else (w_price < w_e20 and w_price < w_e50)
        )
        if not htf_aligned:
            return None

        # ── Entry / SL / TP ───────────────────────────────────────────────────────
        entry = price
        risk  = 1.5 * atr  # SL distance = 1.5×ATR

        if direction == "LONG":
            sl  = entry - risk
            tp1 = entry + 2.0 * risk
            tp2 = entry + 3.0 * risk
            tp3 = entry + 4.0 * risk
        else:
            sl  = entry + risk
            tp1 = entry - 2.0 * risk
            tp2 = entry - 3.0 * risk
            tp3 = entry - 4.0 * risk

        option = _suggest_option(ticker, direction, entry)

        checklist = {
            "trendAligned":    True,          # passed above
            "emaStacked":      True,          # e20>e50>e200 (or inverse)
            "pullbackToEMA20": pullback,
            "htfAligned":      htf_aligned,
            "rsiInZone":       rsi_ok,
            "notSandwiched":   True,          # passed above
        }

        return {
            "ticker":    ticker,
            "direction": direction,
            "price":     round(price, 2),
            "ema20":     round(e20, 2),
            "ema50":     round(e50, 2),
            "ema200":    round(e200, 2),
            "atr":       round(atr, 2),
            "rsi":       round(rsi, 1),
            "entry":     round(entry, 2),
            "sl":        round(sl, 2),
            "tp1":       round(tp1, 2),
            "tp2":       round(tp2, 2),
            "tp3":       round(tp3, 2),
            "risk":      round(risk, 2),
            "distToEma20Pct": round(dist_to_ema20 / price * 100, 2),
            "option":    option,
            "checklist": checklist,
            "scannedAt": datetime.utcnow().isoformat() + "Z",
        }

    except Exception as e:
        print(f"[scanner] {ticker} error: {e}")
        return None


def scan_all(watchlist: list[str] = WATCHLIST) -> list[dict]:
    results = []
    for ticker in watchlist:
        r = analyze_ticker(ticker)
        if r:
            results.append(r)
    return results


def scan_ticker(ticker: str) -> Optional[dict]:
    return analyze_ticker(ticker.upper())
