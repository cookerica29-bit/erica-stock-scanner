"""Liquidity sweep / rejection overlay -- candidates_router wiring.

Explicit design decisions (confirmed before implementing, not assumed):
1. Pure informational overlay, same as BOS/macro/CHoCH/target-clamp/
   order-block-stop/displacement. Legacy's version (scanner.
   _build_trade_stage_eval) hard-gates the "A+ READY" tier on
   sweep_taken AND rejection_confirmed both being true (impossible
   without both, regardless of anything else) and soft-contributes to
   "B+ TRADEABLE". None of that gating is replicated: sweep_confirmed/
   rejection_confirmed never touch R:R, stop, target, or promotion
   eligibility.
2. Enriched-boolean approach, not a full continuous score: sweep is a
   clean binary structural fact (reused directly from
   scanner.detect_liquidity_sweep), rejection carries its raw computed
   numbers (wick_body_ratio, wick_range_pct, body_range_pct, which
   condition fired) in rejection_details rather than collapsing to a bare
   bool or rebuilding displacement's percentile-score machinery.

Scope: the WIRING into _compute_candidate_promotion, and (uniquely for
this port) the correctness of _evaluate_rejection itself, since it's a
REIMPLEMENTATION of scanner.detect_rejection's exact logic (not a direct
call) -- that function only returns a bare bool with no way to recover
which candle matched, so the enrichment requires re-deriving the same
math. This file's fixtures were verified directly against
scanner.detect_liquidity_sweep/detect_rejection before being fixed here,
and _evaluate_rejection was cross-checked against scanner.detect_rejection
's real boolean output across 60 real ticker x direction combinations with
zero mismatches (see session notes) before trusting it.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _FakeRow(dict):
    def __getitem__(self, key):
        return dict.get(self, key)


def _base_daily_frame(n: int = 60) -> pd.DataFrame:
    """Mild uptrend with a clear, stable recent swing low (~105.95) and
    swing high (~109.85) -- verified directly against _find_swings before
    being fixed here. Quiet by default (no sweep in the last 12 bars);
    the fixtures below mutate the final bar to engineer sweep/rejection
    scenarios without disturbing the swing structure itself (margin=4
    means the last few bars never become swing candidates anyway)."""
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
    index = pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(rows, index=index)


def _long_reclaim_frame() -> pd.DataFrame:
    """Last bar sweeps below the swing low (~105.95) then closes back
    above it -- verified: sweep_taken=True, rejection via "reclaim"."""
    df = _base_daily_frame()
    df.iloc[-1, df.columns.get_loc("Low")] = 104.0
    df.iloc[-1, df.columns.get_loc("Open")] = 104.5
    df.iloc[-1, df.columns.get_loc("Close")] = 107.0
    df.iloc[-1, df.columns.get_loc("High")] = 107.3
    return df


def _long_wick_failure_frame() -> pd.DataFrame:
    """Last bar sweeps well below the swing low, closes green, but does
    NOT reclaim the level -- the rejection comes from the wick-size
    threshold instead. Verified: sweep_taken=True, rejection via
    "wick_failure"."""
    df = _base_daily_frame()
    df.iloc[-1, df.columns.get_loc("Low")] = 103.0
    df.iloc[-1, df.columns.get_loc("Open")] = 105.0
    df.iloc[-1, df.columns.get_loc("Close")] = 105.5
    df.iloc[-1, df.columns.get_loc("High")] = 105.6
    return df


def _short_reclaim_frame() -> pd.DataFrame:
    """Symmetric SHORT case: last bar sweeps above the swing high
    (~109.85) then closes back below it."""
    df = _base_daily_frame()
    level = 109.85
    df.iloc[-1, df.columns.get_loc("High")] = level + 2.0
    df.iloc[-1, df.columns.get_loc("Open")] = level + 1.0
    df.iloc[-1, df.columns.get_loc("Close")] = level - 1.0
    df.iloc[-1, df.columns.get_loc("Low")] = level - 1.3
    return df


@pytest.fixture()
def router(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", str(tmp_path / "candidates.db"))
    monkeypatch.setenv("KAIROS_SCANNER_API_KEY", "test-scanner-key")
    import candidates_router
    return candidates_router


# -- pure logic: _evaluate_rejection vs scanner.detect_rejection -------------

def test_evaluate_rejection_matches_legacy_boolean_for_reclaim():
    import candidates_router
    from scanner import _find_swings, detect_liquidity_sweep, detect_rejection

    df = _long_reclaim_frame()
    swings = _find_swings(df)
    sweep_taken, sweep_level = detect_liquidity_sweep(df, swings, "LONG")
    assert sweep_taken is True

    legacy_result = detect_rejection(df, "LONG", sweep_level)
    mine = candidates_router._evaluate_rejection(df, "LONG", sweep_level)

    assert legacy_result is True
    assert mine is not None
    assert mine["condition"] == "reclaim"


def test_evaluate_rejection_matches_legacy_boolean_for_wick_failure():
    import candidates_router
    from scanner import _find_swings, detect_liquidity_sweep, detect_rejection

    df = _long_wick_failure_frame()
    swings = _find_swings(df)
    sweep_taken, sweep_level = detect_liquidity_sweep(df, swings, "LONG")
    assert sweep_taken is True

    legacy_result = detect_rejection(df, "LONG", sweep_level)
    mine = candidates_router._evaluate_rejection(df, "LONG", sweep_level)

    assert legacy_result is True
    assert mine is not None
    assert mine["condition"] == "wick_failure"
    assert mine["wick_body_ratio"] >= 1.25 or mine["wick_range_pct"] >= 0.35


def test_evaluate_rejection_matches_legacy_boolean_when_quiet():
    import candidates_router
    from scanner import _find_swings, detect_liquidity_sweep, detect_rejection

    df = _base_daily_frame()
    swings = _find_swings(df)
    sweep_taken, sweep_level = detect_liquidity_sweep(df, swings, "LONG")
    assert sweep_taken is False

    legacy_result = detect_rejection(df, "LONG", sweep_level)
    mine = candidates_router._evaluate_rejection(df, "LONG", sweep_level)

    assert legacy_result is False
    assert mine is None


def test_evaluate_rejection_short_direction_symmetric():
    import candidates_router
    from scanner import _find_swings, detect_liquidity_sweep, detect_rejection

    df = _short_reclaim_frame()
    swings = _find_swings(df)
    sweep_taken, sweep_level = detect_liquidity_sweep(df, swings, "SHORT")
    assert sweep_taken is True

    legacy_result = detect_rejection(df, "SHORT", sweep_level)
    mine = candidates_router._evaluate_rejection(df, "SHORT", sweep_level)

    assert legacy_result is True
    assert mine is not None
    assert mine["condition"] == "reclaim"


def test_evaluate_rejection_returns_none_without_sweep_level():
    import candidates_router
    df = _base_daily_frame()
    assert candidates_router._evaluate_rejection(df, "LONG", None) is None


def test_evaluate_rejection_returns_none_for_unsupported_direction():
    import candidates_router
    df = _base_daily_frame()
    assert candidates_router._evaluate_rejection(df, "SIDEWAYS", 100.0) is None


# -- wiring into _compute_candidate_promotion ---------------------------------

def test_promotion_surfaces_sweep_and_rejection_reclaim(router, monkeypatch):
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"SWPR": _long_reclaim_frame()})
    candidate = _FakeRow(ticker="SWPR", signal="long", entry_price=107.0, source="test")

    result = router._compute_candidate_promotion(candidate)

    assert result["sweep_confirmed"] is True
    assert result["sweep_details"] == {"level": pytest.approx(105.95, abs=0.01)}
    assert result["rejection_confirmed"] is True
    assert result["rejection_details"]["condition"] == "reclaim"


def test_promotion_surfaces_rejection_wick_failure(router, monkeypatch):
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"WICK": _long_wick_failure_frame()})
    candidate = _FakeRow(ticker="WICK", signal="long", entry_price=105.5, source="test")

    result = router._compute_candidate_promotion(candidate)

    assert result["sweep_confirmed"] is True
    assert result["rejection_confirmed"] is True
    assert result["rejection_details"]["condition"] == "wick_failure"


def test_promotion_quiet_case_has_no_sweep_or_rejection(router, monkeypatch):
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"QUIET": _base_daily_frame()})
    candidate = _FakeRow(ticker="QUIET", signal="long", entry_price=139.35, source="test")

    result = router._compute_candidate_promotion(candidate)

    assert result["sweep_confirmed"] is False
    assert result["sweep_details"] is None
    assert result["rejection_confirmed"] is False
    assert result["rejection_details"] is None


def test_sweep_and_rejection_never_block_promotion_or_change_rr(router, monkeypatch):
    """The explicit design point: both signals firing (or not firing)
    never affects stop/target/risk_reward -- unlike legacy, where their
    absence disqualifies the setup from A+ entirely."""
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"SWPR": _long_reclaim_frame()})
    candidate = _FakeRow(ticker="SWPR", signal="long", entry_price=107.0, source="test")

    result = router._compute_candidate_promotion(candidate)

    assert result["sweep_confirmed"] is True
    assert result["rejection_confirmed"] is True
    assert result["stop"] is not None and result["stop"] > 0
    assert result["risk_reward"] is not None and result["risk_reward"] > 0
    assert result["no_valid_target"] is False


# -- pydantic models ------------------------------------------------------------

def test_promotion_out_model_accepts_sweep_rejection_fields(router):
    payload = {
        "id": 1, "ticker": "SWPR", "source": "test", "direction": "long", "entry_price": 107.0,
        "stop": 104.0, "target": 112.0, "risk_reward": 1.7, "rr_warning": False,
        "no_valid_target": False, "promoted_at": "2026-08-27T00:00:00Z", "position_size": None,
        "atr14": 1.5, "atr_multiplier": 1.5, "rr_warning_threshold": 1.5,
        "min_target_atr_multiple": 2.0, "target_source": "daily_swing_structure",
        "sweep_confirmed": True, "sweep_details": {"level": 105.95},
        "rejection_confirmed": True,
        "rejection_details": {"condition": "reclaim", "wick_body_ratio": 0.2, "wick_range_pct": 0.15, "body_range_pct": 0.76, "time": "2026-03-01T00:00:00Z"},
    }
    out = router.CandidatePromotionOut(**payload)
    assert out.sweep_confirmed is True
    assert out.sweep_details == {"level": 105.95}
    assert out.rejection_confirmed is True
    assert out.rejection_details["condition"] == "reclaim"


def test_promotion_out_model_defaults_sweep_rejection_fields_when_absent(router):
    payload = {
        "id": 2, "ticker": "QUIET", "source": "test", "direction": "long", "entry_price": 139.35,
        "stop": 135.0, "target": 145.0, "risk_reward": 2.0, "rr_warning": False,
        "no_valid_target": False, "promoted_at": "2026-08-27T00:00:00Z", "position_size": None,
        "atr14": 1.5, "atr_multiplier": 1.5, "rr_warning_threshold": 1.5,
        "min_target_atr_multiple": 2.0, "target_source": "daily_swing_structure",
    }
    out = router.CandidatePromotionOut(**payload)
    assert out.sweep_confirmed is False
    assert out.sweep_details is None
    assert out.rejection_confirmed is False
    assert out.rejection_details is None


# -- DB round-trip --------------------------------------------------------------

def _open_db(router, tmp_path):
    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "candidates.db"))
    conn.row_factory = sqlite3.Row
    router._initialize_candidates_schema(conn)
    return conn


def test_store_and_reload_promotion_round_trips_sweep_and_rejection(router, tmp_path):
    conn = _open_db(router, tmp_path)
    promotion = {
        "ticker": "SWPR", "source": "test", "direction": "long", "entry_price": 107.0,
        "stop": 104.0, "target": 112.0, "risk_reward": 1.7, "rr_warning": False,
        "no_valid_target": False, "promoted_at": "2026-08-27T00:00:00Z", "position_size": None,
        "atr14": 1.5, "atr_multiplier": 1.5, "rr_warning_threshold": 1.5,
        "min_target_atr_multiple": 2.0, "target_source": "daily_swing_structure",
        "raw_target": 112.0, "raw_risk_reward": 1.7, "target_clamped": False,
        "target_clamp_badge": None, "target_clamp_reason": None,
        "raw_stop": 104.0, "stop_source": "atr_multiple",
        "displacement_score": 50.0, "displacement_label": "MODERATE",
        "displacement_components": None, "raw_magnitude_score": 55.0,
        "displacement_read": "favorable", "bos_confirmed": False, "bos_details": None,
        "macro_bias": "Macro Neutral", "macro_conflict": False,
        "choch_conflict": False, "choch_details": None,
        "sweep_confirmed": True, "sweep_details": {"level": 105.95},
        "rejection_confirmed": True,
        "rejection_details": {"condition": "reclaim", "wick_body_ratio": 0.2, "wick_range_pct": 0.15, "body_range_pct": 0.76, "time": "2026-03-01T00:00:00Z"},
    }
    router._store_promotion(conn, promotion)
    conn.commit()

    row = conn.execute("SELECT * FROM candidate_promotions WHERE ticker=? AND source=?", ("SWPR", "test")).fetchone()
    reloaded = router._row_to_promotion(row)

    assert reloaded["sweep_confirmed"] is True
    assert reloaded["sweep_details"] == {"level": 105.95}
    assert reloaded["rejection_confirmed"] is True
    assert reloaded["rejection_details"]["condition"] == "reclaim"
    assert reloaded["rejection_details"]["wick_body_ratio"] == 0.2
    conn.close()


def test_store_and_reload_promotion_round_trips_no_sweep_state(router, tmp_path):
    conn = _open_db(router, tmp_path)
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
        "sweep_confirmed": False, "sweep_details": None,
        "rejection_confirmed": False, "rejection_details": None,
    }
    router._store_promotion(conn, promotion)
    conn.commit()

    row = conn.execute("SELECT * FROM candidate_promotions WHERE ticker=? AND source=?", ("QUIET", "test")).fetchone()
    reloaded = router._row_to_promotion(row)

    assert reloaded["sweep_confirmed"] is False
    assert reloaded["sweep_details"] is None
    assert reloaded["rejection_confirmed"] is False
    assert reloaded["rejection_details"] is None
    conn.close()


def test_plan_preview_success_path_threads_sweep_rejection_fields(router, monkeypatch):
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"SWPR": _long_reclaim_frame()})
    monkeypatch.setattr(router, "_safe_option_contract_for_candidate", lambda ticker, direction, entry: None)
    candidate = _FakeRow(
        ticker="SWPR", signal="long", entry_price=107.0, source="test",
        updated_at="2026-08-27T00:00:00Z",
    )

    preview = router._compute_candidate_plan_preview(candidate)

    assert preview["preview_error"] is None
    assert preview["sweep_confirmed"] is True
    assert preview["rejection_confirmed"] is True
    assert preview["rejection_details"]["condition"] == "reclaim"


def test_plan_preview_exception_fallback_defaults_sweep_rejection_fields_safely(router, monkeypatch):
    def _raise(tickers, period, interval):
        raise router.HTTPException(status_code=422, detail="No daily candles available for BAD")

    monkeypatch.setattr(router, "_batch_download", _raise)
    candidate = _FakeRow(
        ticker="BAD", signal="long", entry_price=100.0, source="test",
        updated_at="2026-08-27T00:00:00Z",
    )

    preview = router._compute_candidate_plan_preview(candidate)

    assert preview["preview_error"] is not None
    assert preview["sweep_confirmed"] is False
    assert preview["sweep_details"] is None
    assert preview["rejection_confirmed"] is False
    assert preview["rejection_details"] is None
