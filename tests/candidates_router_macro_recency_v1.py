"""_recency_adjusted_macro_bias -- candidates_router wiring.

Real-world motivation (confirmed on live data before implementing, not
assumed): scanner._macro_bias reads "Macro Bearish" purely off distance from
the 52-week high, with no regard for how long ago that high was set or
whether price has genuinely recovered since. DASH was flagged "Macro
Bearish" (15.97% below a 52-week high set 10.5 months earlier) despite 5
straight months of sustained recovery and zero currently-visible bearish
structure. Since macro_conflict now feeds confluence_label, which gates
ENTER_NOW eligibility (see candidates_router_conflicted_exclusion tests),
this stale reading was a real gating bug, not just a cosmetic one.

Design (approved before implementing -- Option A): _recency_adjusted_macro_bias
wraps scanner._macro_bias unchanged. It only ever downgrades a raw "Macro
Bearish" reading to "Macro Neutral" when price has recovered to within
MACRO_RECENCY_PROXIMITY_PCT (5%) of its own MACRO_RECENCY_WINDOW_TRADING_DAYS
(63 trading days / ~1 quarter) high -- i.e. price is near ITS OWN recent
high, even if still far below a stale 52-week peak. It never upgrades a
Bearish reading straight to "Macro Bullish" (being near your own recent high
is weaker evidence than being near the actual 52-week high, which is what
the raw Bullish rule already means), and "Macro Neutral"/"Macro Bullish"
readings are untouched -- the recency check only ever looks at Bearish
readings in the first place. scanner._macro_bias and its own 3 call sites in
scanner.py are untouched; only this one port call site in
_compute_candidate_promotion was changed to call the wrapper instead.

Scope: the WIRING and the recency-correction LOGIC itself. Fixtures below
were verified directly against _macro_bias/_recency_adjusted_macro_bias
before being fixed here -- not hand-derived from assumptions about what
"should" flip.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _FakeRow(dict):
    def __getitem__(self, key):
        return dict.get(self, key)


def _recovered_bearish_frame(n: int = 280, peak: float = 150.0, bottom: float = 68.0, end: float = 126.0) -> pd.DataFrame:
    """The DASH-shaped calibration case: climbs to a stale peak long ago,
    crashes hard, then recovers steadily and consistently for the most
    recent ~170 bars, ending near ITS OWN 63-trading-day high (within 5%)
    but still meaningfully (>15%) below the old stale peak. Verified
    directly: raw _macro_bias -> "Macro Bearish" (19.0% below the 153.0
    stale high); _recency_adjusted_macro_bias -> "Macro Neutral" (1.8% below
    the 126.28 recent-window high)."""
    rows = []
    climb_days = 40
    crash_days = 70
    recover_days = n - climb_days - crash_days
    for i in range(n):
        if i < climb_days:
            close = 100.0 + (peak - 100.0) * (i / (climb_days - 1))
        elif i < climb_days + crash_days:
            j = i - climb_days
            cycle = j % 14
            close = peak - (peak - bottom) * (j / (crash_days - 1)) + (3 if cycle < 7 else -3)
        else:
            j = i - climb_days - crash_days
            cycle = j % 10
            close = bottom + (end - bottom) * (j / (recover_days - 1)) + (2 if cycle < 5 else -2)
        rows.append({
            "Open": close - 0.3, "High": close + 0.6, "Low": close - 0.6,
            "Close": close, "Volume": 1_000_000,
        })
    index = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(rows, index=index)


def _genuinely_still_bearish_frame(n: int = 280, peak: float = 150.0, bottom: float = 68.0, end: float = 95.0) -> pd.DataFrame:
    """Same stale-peak-and-crash shape, but the back half is weak and choppy
    and the series is forced to end on a sharp down-swing -- still
    meaningfully (>5%) below its OWN recent window high, not just the stale
    52-week peak. The negative control: the recency correction must NOT
    fire here. Verified directly: raw and recency-adjusted both read "Macro
    Bearish" (57.5% below the stale high, 36.4% below the 63-day high)."""
    rows = []
    climb_days = 40
    crash_days = 70
    recover_days = n - climb_days - crash_days
    for i in range(n):
        if i < climb_days:
            close = 100.0 + (peak - 100.0) * (i / (climb_days - 1))
        elif i < climb_days + crash_days:
            j = i - climb_days
            cycle = j % 14
            close = peak - (peak - bottom) * (j / (crash_days - 1)) + (3 if cycle < 7 else -3)
        else:
            j = i - climb_days - crash_days
            cycle = j % 20
            close = bottom + (end - bottom) * (j / (recover_days - 1)) + (8 if cycle < 10 else -8)
        rows.append({
            "Open": close - 0.3, "High": close + 0.6, "Low": close - 0.6,
            "Close": close, "Volume": 1_000_000,
        })
    # Force the last few bars into a sharp decline off a recent local high,
    # so the 63-day window high sits well above (>5%) the final close --
    # otherwise the synthetic cycle could coincidentally end near a local
    # peak and mask the case this fixture exists to cover.
    for k in range(5):
        idx = n - 5 + k
        close = end - 6.0 * (k + 1)
        rows[idx] = {
            "Open": close - 0.3, "High": close + 0.6, "Low": close - 0.6,
            "Close": close, "Volume": 1_000_000,
        }
    index = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(rows, index=index)


@pytest.fixture()
def router(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", str(tmp_path / "candidates.db"))
    monkeypatch.setenv("KAIROS_SCANNER_API_KEY", "test-scanner-key")
    import candidates_router
    return candidates_router


# -- _recency_adjusted_macro_bias, called directly -----------------------------

def test_downgrades_bearish_to_neutral_on_genuine_recovery(router):
    """The DASH calibration case: raw reading is Bearish off a stale
    52-week high, but price has recovered to within 5% of its own recent
    (63-trading-day) high -- must downgrade to Neutral."""
    df = _recovered_bearish_frame()
    price = float(df["Close"].iloc[-1])

    raw = router._macro_bias(price, df)
    adjusted = router._recency_adjusted_macro_bias(price, df)

    assert raw[0] == "Macro Bearish"
    assert adjusted[0] == "Macro Neutral"
    # pct_from_52w / wk52_high must pass through unchanged -- only the bias
    # label itself is corrected, not the underlying 52-week numbers.
    assert adjusted[1] == raw[1]
    assert adjusted[2] == raw[2]


def test_stays_bearish_when_no_genuine_recent_recovery(router):
    """Negative control: raw Bearish AND still meaningfully off its own
    recent high -- the correction must not fire just because a reading is
    old news; it requires an actual recent recovery."""
    df = _genuinely_still_bearish_frame()
    price = float(df["Close"].iloc[-1])

    raw = router._macro_bias(price, df)
    adjusted = router._recency_adjusted_macro_bias(price, df)

    assert raw[0] == "Macro Bearish"
    assert adjusted[0] == "Macro Bearish"
    assert adjusted == raw


def test_never_upgrades_past_neutral(router):
    """Structural guarantee, asserted explicitly: even a recovery all the
    way back to the recent high only ever reaches "Macro Neutral", never
    "Macro Bullish" -- that label is reserved for proximity to the actual
    52-week high, which is strictly stronger evidence."""
    df = _recovered_bearish_frame()
    price = float(df["Close"].iloc[-1])

    adjusted = router._recency_adjusted_macro_bias(price, df)

    assert adjusted[0] in {"Macro Bearish", "Macro Neutral"}
    assert adjusted[0] != "Macro Bullish"


def test_bullish_and_neutral_readings_pass_through_untouched(router):
    """The correction only ever inspects Bearish readings -- Bullish and
    Neutral pass through _macro_bias's raw output byte-for-byte, including
    the 52-week numbers."""
    from candidates_router_macro_choch_v1 import _quiet_daily_frame, _reclaimed_daily_frame

    quiet = _quiet_daily_frame()
    assert router._recency_adjusted_macro_bias(139.35, quiet) == router._macro_bias(139.35, quiet)

    reclaimed = _reclaimed_daily_frame()
    assert router._recency_adjusted_macro_bias(131.8, reclaimed) == router._macro_bias(131.8, reclaimed)


# -- wired through _compute_candidate_promotion --------------------------------

def test_promotion_macro_bias_reflects_recency_downgrade(router, monkeypatch):
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"RECOV": _recovered_bearish_frame()})
    candidate = _FakeRow(ticker="RECOV", signal="long", entry_price=124.0, source="test")

    result = router._compute_candidate_promotion(candidate)

    assert result["macro_bias"] == "Macro Neutral"
    assert result["macro_conflict"] is False


def test_promotion_macro_bias_stays_conflicted_without_real_recovery(router, monkeypatch):
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"BEAR": _genuinely_still_bearish_frame()})
    candidate = _FakeRow(ticker="BEAR", signal="long", entry_price=65.0, source="test")

    result = router._compute_candidate_promotion(candidate)

    assert result["macro_bias"] == "Macro Bearish"
    assert result["macro_conflict"] is True


def test_promotion_recency_downgrade_never_flags_shorts_either(router, monkeypatch):
    """Consistency check: macro bias already never flags shorts as
    conflicted (see candidates_router_macro_choch_v1). The recency
    correction changes the LABEL, not that short-side rule -- confirm both
    still hold together at the recovered-bearish fixture."""
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"RECOV": _recovered_bearish_frame()})
    candidate = _FakeRow(ticker="RECOV", signal="short", entry_price=124.0, source="test")

    result = router._compute_candidate_promotion(candidate)

    assert result["macro_bias"] == "Macro Neutral"
    assert result["macro_conflict"] is False


def test_existing_conflict_fixture_unaffected_by_recency_correction(router, monkeypatch):
    """Regression guard: the pre-existing _conflict_daily_frame fixture (from
    candidates_router_macro_choch_v1) is genuinely still declining with no
    recent recovery -- confirms this fix doesn't quietly soften that
    established test case's real behavior."""
    from candidates_router_macro_choch_v1 import _conflict_daily_frame

    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"CONF": _conflict_daily_frame()})
    candidate = _FakeRow(ticker="CONF", signal="long", entry_price=88.05, source="test")

    result = router._compute_candidate_promotion(candidate)

    assert result["macro_bias"] == "Macro Bearish"
    assert result["macro_conflict"] is True
