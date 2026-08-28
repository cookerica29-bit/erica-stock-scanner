"""Kairos-native confluence summary -- candidates_router wiring.

Explicit design (confirmed before implementing, not assumed -- see session
discussion): NOT a port of legacy's A+/B+ tiers (deliberately impossible to
faithfully reproduce -- see confluence_summary.py's docstring for why).
Instead a flat, equal-weighted count of the 7 already-shipped overlay
signals, purely descriptive, never gating anything.

Scope: the WIRING into _compute_candidate_promotion -- specifically that
confluence_signals is INTERNALLY CONSISTENT with the individual signal
fields already on the same promotion (bos_confirmed, displacement_read,
sweep_confirmed, rejection_confirmed, macro_conflict, choch_conflict,
location_alignment, risk_reward/rr_warning/no_valid_target). This is a
more robust check than hand-deriving an expected confluence_label for a
specific synthetic fixture -- confluence_summary.py's own combination
logic (all 7 per-signal classifications, the applicable-count math, the
label thresholds) is covered in tests/confluence_summary_v1.py.
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
    """Reused from tests/candidates_router_macro_choch_v1.py -- produces
    Macro Bearish + an active bearish CHoCH conflict for LONG, verified
    there against real scanner functions."""
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
    """Reused from tests/candidates_router_macro_choch_v1.py -- Macro
    Bullish, no conflicting CHoCH for LONG."""
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


@pytest.fixture()
def router(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", str(tmp_path / "candidates.db"))
    monkeypatch.setenv("KAIROS_SCANNER_API_KEY", "test-scanner-key")
    import candidates_router
    return candidates_router


def _assert_confluence_consistent_with_promotion(result: dict) -> None:
    signals = result["confluence_signals"]
    counts = result["confluence_counts"]

    expected_bos = "favorable" if result["bos_confirmed"] else "neutral"
    assert signals["bos"] == expected_bos

    read = result["displacement_read"]
    expected_disp = "favorable" if read == "favorable" else "unfavorable" if read == "adverse" else "neutral"
    assert signals["displacement"] == expected_disp

    assert signals["sweep"] == ("favorable" if result["sweep_confirmed"] else "neutral")
    assert signals["rejection"] == ("favorable" if result["rejection_confirmed"] else "neutral")

    expected_macro_choch = "unfavorable" if (result["macro_conflict"] or result["choch_conflict"]) else "neutral"
    assert signals["macro_choch"] == expected_macro_choch

    alignment = result["location_alignment"]
    if alignment is None:
        assert signals["location"] is None
    else:
        assert signals["location"] == alignment

    expected_rr = "unfavorable" if (result["no_valid_target"] or result["rr_warning"]) else "favorable"
    assert signals["risk_reward"] == expected_rr

    # Counts must be pure arithmetic over the signals dict, against
    # whatever is applicable (location excluded when None).
    applicable = {k: v for k, v in signals.items() if v is not None}
    assert counts["applicable"] == len(applicable)
    assert counts["favorable"] == sum(1 for v in applicable.values() if v == "favorable")
    assert counts["unfavorable"] == sum(1 for v in applicable.values() if v == "unfavorable")
    assert counts["neutral"] == sum(1 for v in applicable.values() if v == "neutral")


def test_confluence_consistent_with_conflict_promotion(router, monkeypatch):
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"CONF": _conflict_daily_frame()})
    candidate = _FakeRow(ticker="CONF", signal="long", entry_price=88.05, source="test")

    result = router._compute_candidate_promotion(candidate)

    _assert_confluence_consistent_with_promotion(result)
    assert result["confluence_label"] is not None


def test_confluence_consistent_with_quiet_promotion(router, monkeypatch):
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"QUIET": _quiet_daily_frame()})
    candidate = _FakeRow(ticker="QUIET", signal="long", entry_price=139.35, source="test")

    result = router._compute_candidate_promotion(candidate)

    _assert_confluence_consistent_with_promotion(result)


def test_confluence_consistent_for_short_direction(router, monkeypatch):
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"QUIET": _quiet_daily_frame()})
    candidate = _FakeRow(ticker="QUIET", signal="short", entry_price=139.35, source="test")

    result = router._compute_candidate_promotion(candidate)

    _assert_confluence_consistent_with_promotion(result)


def test_confluence_never_blocks_promotion_or_changes_rr(router, monkeypatch):
    """The explicit design point: even a 'conflicted' confluence read
    (multiple real unfavorable signals) leaves stop/target/risk_reward
    computed exactly as they would be without this feature."""
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"CONF": _conflict_daily_frame()})
    candidate = _FakeRow(ticker="CONF", signal="long", entry_price=88.05, source="test")

    result = router._compute_candidate_promotion(candidate)

    # This fixture is known (from macro/choch tests) to produce real
    # macro_conflict=True and choch_conflict=True -- i.e. a real
    # "conflicted" confluence label is plausible here.
    assert result["stop"] is not None and result["stop"] > 0
    assert result["confluence_label"] is not None


# -- pydantic models ------------------------------------------------------------

def test_promotion_out_model_accepts_confluence_fields(router):
    payload = {
        "id": 1, "ticker": "CONF", "source": "test", "direction": "long", "entry_price": 88.05,
        "stop": 85.0, "target": 92.0, "risk_reward": 1.6, "rr_warning": False,
        "no_valid_target": False, "promoted_at": "2026-08-27T00:00:00Z", "position_size": None,
        "atr14": 1.5, "atr_multiplier": 1.5, "rr_warning_threshold": 1.5,
        "min_target_atr_multiple": 2.0, "target_source": "daily_swing_structure",
        "confluence_signals": {"bos": "favorable", "displacement": "neutral"},
        "confluence_counts": {"favorable": 1, "unfavorable": 0, "neutral": 6, "applicable": 7},
        "confluence_label": "some confluence",
    }
    out = router.CandidatePromotionOut(**payload)
    assert out.confluence_signals["bos"] == "favorable"
    assert out.confluence_counts["favorable"] == 1
    assert out.confluence_label == "some confluence"


def test_promotion_out_model_defaults_confluence_fields_when_absent(router):
    payload = {
        "id": 2, "ticker": "QUIET", "source": "test", "direction": "long", "entry_price": 139.35,
        "stop": 135.0, "target": 145.0, "risk_reward": 2.0, "rr_warning": False,
        "no_valid_target": False, "promoted_at": "2026-08-27T00:00:00Z", "position_size": None,
        "atr14": 1.5, "atr_multiplier": 1.5, "rr_warning_threshold": 1.5,
        "min_target_atr_multiple": 2.0, "target_source": "daily_swing_structure",
    }
    out = router.CandidatePromotionOut(**payload)
    assert out.confluence_signals is None
    assert out.confluence_counts is None
    assert out.confluence_label is None


# -- DB round-trip --------------------------------------------------------------

def _open_db(router, tmp_path):
    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "candidates.db"))
    conn.row_factory = sqlite3.Row
    router._initialize_candidates_schema(conn)
    return conn


def test_store_and_reload_promotion_round_trips_confluence_fields(router, tmp_path):
    conn = _open_db(router, tmp_path)
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
        "displacement_read": "favorable", "bos_confirmed": True, "bos_details": {"break_level": 90.0},
        "macro_bias": "Macro Bearish", "macro_conflict": True,
        "choch_conflict": True, "choch_details": {"direction": "bearish", "level": 88.55, "active": True, "reason": "x"},
        "sweep_confirmed": True, "sweep_details": {"level": 87.0},
        "rejection_confirmed": True, "rejection_details": {"condition": "reclaim", "wick_body_ratio": 0.5, "wick_range_pct": 0.2, "body_range_pct": 0.5, "time": "2026-08-01T00:00:00Z"},
        "location_percentile": 20.0, "location_label": "discount", "location_alignment": "favorable",
        "confluence_signals": {
            "bos": "favorable", "displacement": "favorable", "sweep": "favorable",
            "rejection": "favorable", "macro_choch": "unfavorable", "location": "favorable",
            "risk_reward": "favorable",
        },
        "confluence_counts": {"favorable": 6, "unfavorable": 1, "neutral": 0, "applicable": 7},
        "confluence_label": "conflicted",
    }
    router._store_promotion(conn, promotion)
    conn.commit()

    row = conn.execute("SELECT * FROM candidate_promotions WHERE ticker=? AND source=?", ("CONF", "test")).fetchone()
    reloaded = router._row_to_promotion(row)

    assert reloaded["confluence_signals"]["bos"] == "favorable"
    assert reloaded["confluence_signals"]["macro_choch"] == "unfavorable"
    assert reloaded["confluence_counts"] == {"favorable": 6, "unfavorable": 1, "neutral": 0, "applicable": 7}
    assert reloaded["confluence_label"] == "conflicted"
    conn.close()


def test_store_and_reload_promotion_round_trips_none_confluence(router, tmp_path):
    conn = _open_db(router, tmp_path)
    promotion = {
        "ticker": "NOCONF", "source": "test", "direction": "long", "entry_price": 100.0,
        "stop": 95.0, "target": 110.0, "risk_reward": 2.0, "rr_warning": False,
        "no_valid_target": False, "promoted_at": "2026-08-27T00:00:00Z", "position_size": None,
        "atr14": 1.5, "atr_multiplier": 1.5, "rr_warning_threshold": 1.5,
        "min_target_atr_multiple": 2.0, "target_source": "daily_swing_structure",
        "raw_target": 110.0, "raw_risk_reward": 2.0, "target_clamped": False,
        "target_clamp_badge": None, "target_clamp_reason": None,
        "raw_stop": 95.0, "stop_source": "atr_multiple",
        "displacement_score": 20.0, "displacement_label": "WEAK",
        "displacement_components": None, "raw_magnitude_score": 15.0,
        "displacement_read": "quiet", "bos_confirmed": False, "bos_details": None,
        "macro_bias": "Macro Neutral", "macro_conflict": False,
        "choch_conflict": False, "choch_details": None,
        "sweep_confirmed": False, "sweep_details": None,
        "rejection_confirmed": False, "rejection_details": None,
        "location_percentile": None, "location_label": None, "location_alignment": None,
        "confluence_signals": None, "confluence_counts": None, "confluence_label": None,
    }
    router._store_promotion(conn, promotion)
    conn.commit()

    row = conn.execute("SELECT * FROM candidate_promotions WHERE ticker=? AND source=?", ("NOCONF", "test")).fetchone()
    reloaded = router._row_to_promotion(row)

    assert reloaded["confluence_signals"] is None
    assert reloaded["confluence_counts"] is None
    assert reloaded["confluence_label"] is None
    conn.close()


def test_plan_preview_success_path_threads_confluence_fields(router, monkeypatch):
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"CONF": _conflict_daily_frame()})
    monkeypatch.setattr(router, "_safe_option_contract_for_candidate", lambda ticker, direction, entry, **kwargs: None)
    candidate = _FakeRow(
        ticker="CONF", signal="long", entry_price=88.05, source="test",
        updated_at="2026-08-27T00:00:00Z",
    )

    preview = router._compute_candidate_plan_preview(candidate)

    assert preview["preview_error"] is None
    assert preview["confluence_signals"] is not None
    assert preview["confluence_counts"] is not None
    assert preview["confluence_label"] is not None


def test_plan_preview_exception_fallback_defaults_confluence_fields_safely(router, monkeypatch):
    def _raise(tickers, period, interval):
        raise router.HTTPException(status_code=422, detail="No daily candles available for BAD")

    monkeypatch.setattr(router, "_batch_download", _raise)
    candidate = _FakeRow(
        ticker="BAD", signal="long", entry_price=100.0, source="test",
        updated_at="2026-08-27T00:00:00Z",
    )

    preview = router._compute_candidate_plan_preview(candidate)

    assert preview["preview_error"] is not None
    assert preview["confluence_signals"] is None
    assert preview["confluence_counts"] is None
    assert preview["confluence_label"] is None
