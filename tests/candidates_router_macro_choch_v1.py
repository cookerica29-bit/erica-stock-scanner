"""Macro bias / CHoCH conflict overlay -- candidates_router wiring.

Explicit design decision (confirmed before implementing, not assumed):
pure informational overlay, same as BOS/target-clamp/order-block-stop/
displacement. Legacy's version (scanner.analyze_ticker's macro_block/
choch_block feeding scanner._cap_quality_to_c) hard-caps score to <=58,
forces grade "C", and sets a_plus_ready=False/b_plus_tradeable=False/
trade_stage="RANGE / NO TRADE" -- a real gate, not a label. None of that is
replicated here: macro_conflict/choch_conflict never touch R:R, stop,
target, or promotion eligibility. Several tests below assert exactly that.

Scope: the WIRING of scanner._macro_bias/_detect_choch into
_compute_candidate_promotion -- not those functions' own detection logic,
which is scanner.py's existing, separately-covered responsibility. Fixtures
below were verified directly against _macro_bias/_detect_choch before being
fixed here (see session notes) -- not hand-derived from assumptions about
what "should" produce a conflict.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _FakeRow(dict):
    def __getitem__(self, key):
        return dict.get(self, key)


def _conflict_daily_frame(n: int = 260) -> pd.DataFrame:
    """Climbs to a peak, then declines with oscillation into a confirmed
    bearish CHoCH, ending well off the highs. Verified directly: for LONG,
    produces macro_bias="Macro Bearish" (36.7% below 52w high),
    macro_conflict=True, and an ACTIVE bearish CHoCH conflict (price still
    below the broken level: 88.05 <= 88.55). For SHORT, neither conflicts
    (macro bias never flags shorts; a bearish CHoCH doesn't conflict with a
    short)."""
    rows = []
    base = 100.0
    for i in range(n):
        if i < 60:
            close = base + i * 0.6
        elif i < 140:
            cycle = i % 14
            close = base + 60 * 0.6 - (i - 60) * 0.5 + (3 if cycle < 7 else -3)
        else:
            cycle = i % 14
            close = base + 60 * 0.6 - 80 * 0.5 - (i - 140) * 0.05 + (2 if cycle < 7 else -2)
        rows.append({
            "Open": close - 0.1, "High": close + 0.6, "Low": close - 0.6,
            "Close": close, "Volume": 1_000_000,
        })
    index = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(rows, index=index)


def _quiet_daily_frame(n: int = 260) -> pd.DataFrame:
    """Persistent uptrend, near highs. Verified directly: for LONG,
    macro_bias="Macro Bullish" (0.5% below 52w high), no conflicting CHoCH
    (the CHoCH present is bullish, which favors -- doesn't conflict with --
    a long). For SHORT: macro_conflict=False (never flags shorts), but the
    SAME bullish CHoCH is now an ACTIVE conflict (price 139.35 >= broken
    level 136.85) -- reused deliberately to cover the short-side CHoCH path
    without a fourth fixture."""
    rows = []
    base = 100.0
    for i in range(n):
        cycle = i % 12
        if cycle < 6:
            close = base + i * 0.15 + cycle * 0.3
        else:
            close = base + i * 0.15 + (12 - cycle) * 0.3 - 1.0
        rows.append({
            "Open": close - 0.1, "High": close + 0.4, "Low": close - 0.4,
            "Close": close, "Volume": 1_000_000,
        })
    index = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(rows, index=index)


def _reclaimed_daily_frame(n: int = 260) -> pd.DataFrame:
    """Climbs, forms a bearish CHoCH on the way down, bottoms out, then
    sharply recovers back above the broken CHoCH level by the last bar.
    Verified directly: for LONG, a bearish CHoCH conflict exists
    (categorically), but price (131.8) has already reclaimed the broken
    level (107.25) -- so it's no longer ACTIVE. This is the "conflict
    existed, price already recovered" case the price-relative refinement
    exists for."""
    rows = []
    base = 100.0
    for i in range(n):
        if i < 60:
            close = base + i * 0.6
        elif i < 100:
            cycle = i % 14
            close = base + 60 * 0.6 - (i - 60) * 0.5 + (3 if cycle < 7 else -3)
        elif i < 240:
            cycle = i % 14
            close = base + 60 * 0.6 - 40 * 0.5 - (i - 100) * 0.05 + (2 if cycle < 7 else -2)
        else:
            close = base + 60 * 0.6 - 40 * 0.5 - 140 * 0.05 + (i - 240) * 1.2
        rows.append({
            "Open": close - 0.1, "High": close + 0.6, "Low": close - 0.6,
            "Close": close, "Volume": 1_000_000,
        })
    index = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(rows, index=index)


@pytest.fixture()
def router(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", str(tmp_path / "candidates.db"))
    monkeypatch.setenv("KAIROS_SCANNER_API_KEY", "test-scanner-key")
    import candidates_router
    return candidates_router


# -- macro bias --------------------------------------------------------------

def test_macro_conflict_true_for_bearish_long(router, monkeypatch):
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"CONF": _conflict_daily_frame()})
    candidate = _FakeRow(ticker="CONF", signal="long", entry_price=88.05, source="test")

    result = router._compute_candidate_promotion(candidate)

    assert result["macro_bias"] == "Macro Bearish"
    assert result["macro_conflict"] is True


def test_macro_conflict_never_flags_shorts(router, monkeypatch):
    """Explicit legacy rule, confirmed by reading the real code: macro bias
    never blocks shorts, only longs -- 'short signals use local structure
    detection only.'"""
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"CONF": _conflict_daily_frame()})
    candidate = _FakeRow(ticker="CONF", signal="short", entry_price=88.05, source="test")

    result = router._compute_candidate_promotion(candidate)

    assert result["macro_bias"] == "Macro Bearish"  # bias itself is still reported...
    assert result["macro_conflict"] is False          # ...but never flagged as a conflict for a short


def test_macro_conflict_false_when_bullish(router, monkeypatch):
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"QUIET": _quiet_daily_frame()})
    candidate = _FakeRow(ticker="QUIET", signal="long", entry_price=139.35, source="test")

    result = router._compute_candidate_promotion(candidate)

    assert result["macro_bias"] == "Macro Bullish"
    assert result["macro_conflict"] is False


# -- CHoCH --------------------------------------------------------------------

def test_choch_conflict_active_for_long(router, monkeypatch):
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"CONF": _conflict_daily_frame()})
    candidate = _FakeRow(ticker="CONF", signal="long", entry_price=88.05, source="test")

    result = router._compute_candidate_promotion(candidate)

    assert result["choch_conflict"] is True
    assert result["choch_details"] is not None
    assert result["choch_details"]["direction"] == "bearish"
    assert result["choch_details"]["active"] is True
    assert result["choch_details"]["level"] == pytest.approx(88.55, abs=0.01)


def test_choch_conflict_active_for_short(router, monkeypatch):
    """Symmetric case, reusing the quiet uptrend frame's bullish CHoCH from
    the short side -- confirms the direction-mirrored branch, not just LONG."""
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"QUIET": _quiet_daily_frame()})
    candidate = _FakeRow(ticker="QUIET", signal="short", entry_price=139.35, source="test")

    result = router._compute_candidate_promotion(candidate)

    assert result["choch_conflict"] is True
    assert result["choch_details"]["direction"] == "bullish"
    assert result["choch_details"]["active"] is True


def test_choch_no_conflict_when_choch_favors_direction(router, monkeypatch):
    """The quiet frame's CHoCH is bullish -- for a LONG, that favors the
    trade, so it must not read as a conflict at all (details stay None,
    not just conflict=False)."""
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"QUIET": _quiet_daily_frame()})
    candidate = _FakeRow(ticker="QUIET", signal="long", entry_price=139.35, source="test")

    result = router._compute_candidate_promotion(candidate)

    assert result["choch_conflict"] is False
    assert result["choch_details"] is None


def test_choch_conflict_inactive_once_price_reclaims_level(router, monkeypatch):
    """A conflicting-direction CHoCH exists, but price has already recovered
    past the broken level -- choch_conflict must be False (matches legacy's
    price-relative refinement exactly), but choch_details is still
    populated with active=False rather than silently dropped."""
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"RECL": _reclaimed_daily_frame()})
    candidate = _FakeRow(ticker="RECL", signal="long", entry_price=131.8, source="test")

    result = router._compute_candidate_promotion(candidate)

    assert result["choch_conflict"] is False
    assert result["choch_details"] is not None
    assert result["choch_details"]["active"] is False
    assert result["choch_details"]["direction"] == "bearish"


# -- non-gating regression checks ---------------------------------------------

def test_macro_and_choch_conflict_never_block_promotion_or_change_rr(router, monkeypatch):
    """The whole point of the (A) decision: both conflicts firing
    simultaneously still produces a normal, successful promotion with a
    real stop/target/risk_reward -- exactly as if these fields didn't
    exist. No score cap, no forced 'no trade' stage, unlike legacy."""
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"CONF": _conflict_daily_frame()})
    candidate = _FakeRow(ticker="CONF", signal="long", entry_price=88.05, source="test")

    result = router._compute_candidate_promotion(candidate)

    assert result["macro_conflict"] is True
    assert result["choch_conflict"] is True
    assert result["stop"] is not None and result["stop"] > 0
    assert result["risk_reward"] is not None and result["risk_reward"] > 0
    assert result["no_valid_target"] is False


# -- pydantic models ------------------------------------------------------------

def test_promotion_out_model_accepts_conflict_fields(router):
    payload = {
        "id": 1, "ticker": "CONF", "source": "test", "direction": "long", "entry_price": 88.05,
        "stop": 85.0, "target": 92.0, "risk_reward": 1.6, "rr_warning": False,
        "no_valid_target": False, "promoted_at": "2026-08-27T00:00:00Z", "position_size": None,
        "atr14": 1.5, "atr_multiplier": 1.5, "rr_warning_threshold": 1.5,
        "min_target_atr_multiple": 2.0, "target_source": "daily_swing_structure",
        "macro_bias": "Macro Bearish", "macro_conflict": True,
        "choch_conflict": True, "choch_details": {"direction": "bearish", "level": 88.55, "active": True, "reason": "..."},
    }
    out = router.CandidatePromotionOut(**payload)
    assert out.macro_bias == "Macro Bearish"
    assert out.macro_conflict is True
    assert out.choch_conflict is True
    assert out.choch_details["level"] == 88.55


def test_promotion_out_model_defaults_conflict_fields_when_absent(router):
    payload = {
        "id": 2, "ticker": "QUIET", "source": "test", "direction": "long", "entry_price": 139.35,
        "stop": 135.0, "target": 145.0, "risk_reward": 2.0, "rr_warning": False,
        "no_valid_target": False, "promoted_at": "2026-08-27T00:00:00Z", "position_size": None,
        "atr14": 1.5, "atr_multiplier": 1.5, "rr_warning_threshold": 1.5,
        "min_target_atr_multiple": 2.0, "target_source": "daily_swing_structure",
    }
    out = router.CandidatePromotionOut(**payload)
    assert out.macro_bias is None
    assert out.macro_conflict is False
    assert out.choch_conflict is False
    assert out.choch_details is None


# -- DB round-trip --------------------------------------------------------------

def _open_db(router, monkeypatch, tmp_path):
    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "candidates.db"))
    conn.row_factory = sqlite3.Row
    router._initialize_candidates_schema(conn)
    return conn


def test_store_and_reload_promotion_round_trips_conflict_fields(router, tmp_path):
    conn = _open_db(router, None, tmp_path)
    promotion = {
        "ticker": "CONF", "source": "test", "direction": "long", "entry_price": 88.05,
        "stop": 85.0, "target": 92.0, "risk_reward": 1.6, "rr_warning": False,
        "no_valid_target": False, "promoted_at": "2026-08-27T00:00:00Z", "position_size": None,
        "atr14": 1.5, "atr_multiplier": 1.5, "rr_warning_threshold": 1.5,
        "min_target_atr_multiple": 2.0, "target_source": "daily_swing_structure",
        "raw_target": 92.0, "raw_risk_reward": 1.6, "target_clamped": False,
        "target_clamp_badge": None, "target_clamp_reason": None,
        "raw_stop": 85.0, "stop_source": "atr_multiple",
        "displacement_score": 50.0, "displacement_label": "MODERATE",
        "displacement_components": None, "raw_magnitude_score": 55.0,
        "displacement_read": "favorable", "bos_confirmed": False, "bos_details": None,
        "macro_bias": "Macro Bearish", "macro_conflict": True,
        "choch_conflict": True,
        "choch_details": {"direction": "bearish", "level": 88.55, "active": True, "reason": "test reason"},
    }
    router._store_promotion(conn, promotion)
    conn.commit()

    row = conn.execute("SELECT * FROM candidate_promotions WHERE ticker=? AND source=?", ("CONF", "test")).fetchone()
    reloaded = router._row_to_promotion(row)

    assert reloaded["macro_bias"] == "Macro Bearish"
    assert reloaded["macro_conflict"] is True
    assert reloaded["choch_conflict"] is True
    assert reloaded["choch_details"] == {"direction": "bearish", "level": 88.55, "active": True, "reason": "test reason"}
    conn.close()


def test_store_and_reload_promotion_round_trips_no_conflict_state(router, tmp_path):
    conn = _open_db(router, None, tmp_path)
    promotion = {
        "ticker": "QUIET", "source": "test", "direction": "long", "entry_price": 139.35,
        "stop": 135.0, "target": 145.0, "risk_reward": 2.0, "rr_warning": False,
        "no_valid_target": False, "promoted_at": "2026-08-27T00:00:00Z", "position_size": None,
        "atr14": 1.5, "atr_multiplier": 1.5, "rr_warning_threshold": 1.5,
        "min_target_atr_multiple": 2.0, "target_source": "daily_swing_structure",
        "raw_target": 145.0, "raw_risk_reward": 2.0, "target_clamped": False,
        "target_clamp_badge": None, "target_clamp_reason": None,
        "raw_stop": 135.0, "stop_source": "atr_multiple",
        "displacement_score": 20.0, "displacement_label": "WEAK",
        "displacement_components": None, "raw_magnitude_score": 15.0,
        "displacement_read": "quiet", "bos_confirmed": False, "bos_details": None,
        "macro_bias": "Macro Bullish", "macro_conflict": False,
        "choch_conflict": False, "choch_details": None,
    }
    router._store_promotion(conn, promotion)
    conn.commit()

    row = conn.execute("SELECT * FROM candidate_promotions WHERE ticker=? AND source=?", ("QUIET", "test")).fetchone()
    reloaded = router._row_to_promotion(row)

    assert reloaded["macro_bias"] == "Macro Bullish"
    assert reloaded["macro_conflict"] is False
    assert reloaded["choch_conflict"] is False
    assert reloaded["choch_details"] is None
    conn.close()


def test_plan_preview_success_path_threads_conflict_fields(router, monkeypatch):
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"CONF": _conflict_daily_frame()})
    monkeypatch.setattr(router, "_safe_option_contract_for_candidate", lambda ticker, direction, entry: None)
    candidate = _FakeRow(
        ticker="CONF", signal="long", entry_price=88.05, source="test",
        updated_at="2026-08-27T00:00:00Z",
    )

    preview = router._compute_candidate_plan_preview(candidate)

    assert preview["preview_error"] is None
    assert preview["macro_bias"] == "Macro Bearish"
    assert preview["macro_conflict"] is True
    assert preview["choch_conflict"] is True
    assert preview["choch_details"]["direction"] == "bearish"


def test_plan_preview_exception_fallback_defaults_conflict_fields_safely(router, monkeypatch):
    def _raise(tickers, period, interval):
        raise router.HTTPException(status_code=422, detail="No daily candles available for BAD")

    monkeypatch.setattr(router, "_batch_download", _raise)
    candidate = _FakeRow(
        ticker="BAD", signal="long", entry_price=100.0, source="test",
        updated_at="2026-08-27T00:00:00Z",
    )

    preview = router._compute_candidate_plan_preview(candidate)

    assert preview["preview_error"] is not None
    assert preview["macro_bias"] is None
    assert preview["macro_conflict"] is False
    assert preview["choch_conflict"] is False
    assert preview["choch_details"] is None
