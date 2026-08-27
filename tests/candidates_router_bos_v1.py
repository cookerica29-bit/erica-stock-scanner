"""BOS (break-of-structure) confirmation overlay -- candidates_router wiring.

Scope: this file tests the WIRING of scanner._detect_bos into
_compute_candidate_promotion / _compute_candidate_plan_preview / the SQLite
storage round-trip -- not _detect_bos's own break-detection logic, which is
scanner.py's existing, separately-covered responsibility (confirmed identical
to its public alias scanner.detect_structure_break -- a thin pass-through
wrapper, verified by direct read before this port).

Per the explicit design decision: bos_confirmed/bos_details is an additive,
informational overlay field. It must never change stop/target/risk_reward/
rr_warning/promotion success -- there is no gate here, on purpose. Several
tests below assert exactly that (a candidate with bos_confirmed=False still
promotes normally, with a normal stop/target/risk_reward).
"""

import math
import os
import sys
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _FakeRow(dict):
    """Minimal sqlite3.Row stand-in -- candidates_router reads candidate
    fields via both __getitem__ and .keys(), same as the existing ingestion
    tests' fixtures."""

    def __getitem__(self, key):
        return dict.get(self, key)


def _trending_daily_frame(n: int = 60) -> pd.DataFrame:
    """A persistently ascending daily series -- ascending swing highs/lows
    throughout, guaranteeing a real close-beyond-prior-swing-high with a
    bullish candle body somewhere in the lookback window. Verified directly
    against scanner._find_swings/_detect_bos before being fixed here (see
    session notes): produces bos_confirmed=True, break_level=108.05."""
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


def _sideways_daily_frame(n: int = 60) -> pd.DataFrame:
    """A pure oscillation with no net drift -- verified directly against
    scanner._find_swings/_detect_bos before being fixed here: never closes
    beyond a prior swing high (or low) with directional body confirmation,
    in either direction. produces bos_confirmed=False, break_level=0.0."""
    rows = []
    base = 100.0
    for i in range(n):
        close = base + 2.0 * math.sin(i / 4.0)
        rows.append({
            "Open": close - 0.05, "High": close + 0.3, "Low": close - 0.3,
            "Close": close, "Volume": 1_000_000,
        })
    index = pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(rows, index=index)


@pytest.fixture()
def router(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", str(tmp_path / "candidates.db"))
    monkeypatch.setenv("KAIROS_SCANNER_API_KEY", "test-scanner-key")
    import candidates_router
    return candidates_router


def test_promotion_surfaces_bos_confirmed_true_with_break_level(router, monkeypatch):
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"TRND": _trending_daily_frame()})
    candidate = _FakeRow(ticker="TRND", signal="long", entry_price=100.0, source="test")

    result = router._compute_candidate_promotion(candidate)

    assert result["bos_confirmed"] is True
    assert result["bos_details"] == {"break_level": 108.05}


def test_promotion_surfaces_bos_not_confirmed_as_false_with_none_details(router, monkeypatch):
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"FLAT": _sideways_daily_frame()})
    candidate = _FakeRow(ticker="FLAT", signal="long", entry_price=100.0, source="test")

    result = router._compute_candidate_promotion(candidate)

    assert result["bos_confirmed"] is False
    assert result["bos_details"] is None


def test_bos_confirmation_does_not_gate_promotion_or_change_rr(router, monkeypatch):
    """The whole point of the additive-overlay design: a candidate with no
    BOS confirmation still promotes normally -- real stop, real target, real
    risk_reward -- exactly as if the field didn't exist. This is the
    regression test for "never a hard gate"."""
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"FLAT": _sideways_daily_frame()})
    candidate = _FakeRow(ticker="FLAT", signal="long", entry_price=100.0, source="test")

    result = router._compute_candidate_promotion(candidate)

    assert result["bos_confirmed"] is False
    assert result["stop"] is not None and result["stop"] > 0
    assert result["target"] is not None
    assert result["risk_reward"] is not None and result["risk_reward"] > 0
    assert result["no_valid_target"] is False


def test_promotion_out_model_accepts_bos_fields(router):
    payload = {
        "ticker": "TRND", "source": "test", "direction": "long", "entry_price": 100.0,
        "stop": 98.5, "target": 105.0, "risk_reward": 2.0, "rr_warning": False,
        "no_valid_target": False, "promoted_at": "2026-08-27T00:00:00Z", "position_size": None,
        "atr14": 1.5, "atr_multiplier": 1.5, "rr_warning_threshold": 1.5,
        "min_target_atr_multiple": 2.0, "target_source": "daily_swing_structure",
        "bos_confirmed": True, "bos_details": {"break_level": 108.05},
    }
    out = router.CandidatePromotionOut(**payload)
    assert out.bos_confirmed is True
    assert out.bos_details == {"break_level": 108.05}


def test_promotion_out_model_defaults_bos_fields_when_absent(router):
    payload = {
        "ticker": "FLAT", "source": "test", "direction": "long", "entry_price": 100.0,
        "stop": 98.5, "target": 105.0, "risk_reward": 2.0, "rr_warning": False,
        "no_valid_target": False, "promoted_at": "2026-08-27T00:00:00Z", "position_size": None,
        "atr14": 1.5, "atr_multiplier": 1.5, "rr_warning_threshold": 1.5,
        "min_target_atr_multiple": 2.0, "target_source": "daily_swing_structure",
    }
    out = router.CandidatePromotionOut(**payload)
    assert out.bos_confirmed is False
    assert out.bos_details is None


def _open_db(router):
    conn = sqlite3.connect(os.environ["KAIROS_CANDIDATES_DB"])
    conn.row_factory = sqlite3.Row
    router._initialize_candidates_schema(conn)
    return conn


def test_store_and_reload_promotion_round_trips_bos_confirmed_true(router):
    conn = _open_db(router)
    # Build the promotion dict directly (no network/df dependency needed for
    # a pure storage round-trip test) -- mirrors the shape returned by
    # _compute_candidate_promotion.
    promotion = {
        "ticker": "TRND", "source": "test", "direction": "long", "entry_price": 100.0,
        "stop": 98.5, "target": 105.0, "risk_reward": 2.0, "rr_warning": False,
        "no_valid_target": False, "promoted_at": "2026-08-27T00:00:00Z", "position_size": None,
        "atr14": 1.5, "atr_multiplier": 1.5, "rr_warning_threshold": 1.5,
        "min_target_atr_multiple": 2.0, "target_source": "daily_swing_structure",
        "raw_target": 105.0, "raw_risk_reward": 2.0, "target_clamped": False,
        "target_clamp_badge": None, "target_clamp_reason": None,
        "raw_stop": 98.5, "stop_source": "atr_multiple",
        "displacement_score": 50.0, "displacement_label": "MODERATE",
        "displacement_components": {"body_percentile": 60.0}, "raw_magnitude_score": 55.0,
        "displacement_read": "favorable",
        "bos_confirmed": True, "bos_details": {"break_level": 108.05},
    }
    router._store_promotion(conn, promotion)
    conn.commit()

    row = conn.execute(
        "SELECT * FROM candidate_promotions WHERE ticker = ? AND source = ?", ("TRND", "test")
    ).fetchone()
    reloaded = router._row_to_promotion(row)

    assert reloaded["bos_confirmed"] is True
    assert reloaded["bos_details"] == {"break_level": 108.05}
    conn.close()


def test_store_and_reload_promotion_round_trips_bos_confirmed_false(router):
    conn = _open_db(router)
    promotion = {
        "ticker": "FLAT", "source": "test", "direction": "long", "entry_price": 100.0,
        "stop": 99.0, "target": 102.0, "risk_reward": 1.8, "rr_warning": False,
        "no_valid_target": False, "promoted_at": "2026-08-27T00:00:00Z", "position_size": None,
        "atr14": 0.65, "atr_multiplier": 1.5, "rr_warning_threshold": 1.5,
        "min_target_atr_multiple": 2.0, "target_source": "daily_swing_structure",
        "raw_target": 102.0, "raw_risk_reward": 1.8, "target_clamped": False,
        "target_clamp_badge": None, "target_clamp_reason": None,
        "raw_stop": 99.0, "stop_source": "atr_multiple",
        "displacement_score": 20.0, "displacement_label": "WEAK",
        "displacement_components": {"body_percentile": 10.0}, "raw_magnitude_score": 15.0,
        "displacement_read": "quiet",
        "bos_confirmed": False, "bos_details": None,
    }
    router._store_promotion(conn, promotion)
    conn.commit()

    row = conn.execute(
        "SELECT * FROM candidate_promotions WHERE ticker = ? AND source = ?", ("FLAT", "test")
    ).fetchone()
    reloaded = router._row_to_promotion(row)

    assert reloaded["bos_confirmed"] is False
    assert reloaded["bos_details"] is None
    conn.close()


def test_plan_preview_success_path_threads_bos_fields(router, monkeypatch):
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"TRND": _trending_daily_frame()})
    monkeypatch.setattr(router, "_safe_option_contract_for_candidate", lambda ticker, direction, entry: None)
    candidate = _FakeRow(
        ticker="TRND", signal="long", entry_price=100.0, source="test",
        updated_at="2026-08-27T00:00:00Z",
    )

    preview = router._compute_candidate_plan_preview(candidate)

    assert preview["preview_error"] is None
    assert preview["bos_confirmed"] is True
    assert preview["bos_details"] == {"break_level": 108.05}


def test_plan_preview_exception_fallback_defaults_bos_fields_safely(router, monkeypatch):
    def _raise(tickers, period, interval):
        raise router.HTTPException(status_code=422, detail="No daily candles available for BAD")

    monkeypatch.setattr(router, "_batch_download", _raise)
    candidate = _FakeRow(
        ticker="BAD", signal="long", entry_price=100.0, source="test",
        updated_at="2026-08-27T00:00:00Z",
    )

    preview = router._compute_candidate_plan_preview(candidate)

    assert preview["preview_error"] is not None
    assert preview["bos_confirmed"] is False
    assert preview["bos_details"] is None


def test_plan_preview_out_model_accepts_bos_fields(router):
    payload = {
        "ticker": "TRND", "source": "test", "signal": "long", "entry_price": 100.0,
        "stop": 98.5, "target": 105.0, "risk_reward": 2.0, "rr_warning": False,
        "no_valid_target": False, "atr14": 1.5, "atr_multiplier": 1.5,
        "rr_warning_threshold": 1.5, "min_target_atr_multiple": 2.0,
        "target_source": "daily_swing_structure", "bos_confirmed": True,
        "bos_details": {"break_level": 108.05}, "option_contract": None,
        "preview_error": None, "computed_at": "2026-08-27T00:00:00Z",
        "candidate_updated_at": "2026-08-27T00:00:00Z",
    }
    out = router.CandidatePlanPreviewOut(**payload)
    assert out.bos_confirmed is True
    assert out.bos_details == {"break_level": 108.05}
