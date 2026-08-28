"""Premium/discount location overlay -- candidates_router wiring.

Explicit design decisions (confirmed before implementing, not assumed):
1. Pure informational overlay, same as every other port today. Legacy's
   scanner._strict_location output (valid_zone) is the single hardest gate
   found in any port this session -- appears directly in
   _build_trade_stage_eval's no_trade_reasons (can solo-cause "RANGE / NO
   TRADE"), gates BOTH the "A+ READY" and "BUILDING / WATCHLIST" tiers on a
   strict 50% cutoff, and a third, looser cutoff gates "B+ TRADEABLE" --
   three different threshold schemes for one number. None of that gating
   is replicated here: location_percentile/location_label/
   location_alignment never touch R:R, stop, target, or promotion
   eligibility.
2. Continuous score, not fixed buckets: location_percentile (0-100) is the
   primary/authoritative field, mirroring the raw_magnitude_score/
   displacement_label split exactly -- location_label/location_alignment
   are display-only sugar derived from it, picked from legacy's simpler
   3-tier scheme (not the 4-tier AT EXTREME variant), never a second
   source of truth.

Scope: the WIRING into _compute_candidate_promotion. location_score.py's
own math (including its parity with scanner._latest_swing_range/
_location_read/_strict_location's real percentile on real tickers -- zero
mismatches across 30 real tickers, see session notes) is covered in
tests/location_score_v1.py.
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
    """Mild uptrend -- verified directly against _find_swings/score_location
    before being fixed here: latest-3-highs/latest-3-lows range is
    [102.35, 109.85]. entry_price=102.35 -> discount/favorable-for-long
    (0th percentile); 106.1 -> midrange/neutral (50th); 109.85 ->
    premium/unfavorable-for-long (100th)."""
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


@pytest.fixture()
def router(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", str(tmp_path / "candidates.db"))
    monkeypatch.setenv("KAIROS_SCANNER_API_KEY", "test-scanner-key")
    import candidates_router
    return candidates_router


def test_promotion_surfaces_discount_favorable_for_long(router, monkeypatch):
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"DISC": _base_daily_frame()})
    candidate = _FakeRow(ticker="DISC", signal="long", entry_price=102.35, source="test")

    result = router._compute_candidate_promotion(candidate)

    assert result["location_percentile"] == 0.0
    assert result["location_label"] == "discount"
    assert result["location_alignment"] == "favorable"


def test_promotion_surfaces_premium_unfavorable_for_long(router, monkeypatch):
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"PREM": _base_daily_frame()})
    candidate = _FakeRow(ticker="PREM", signal="long", entry_price=109.85, source="test")

    result = router._compute_candidate_promotion(candidate)

    assert result["location_percentile"] == 100.0
    assert result["location_label"] == "premium"
    assert result["location_alignment"] == "unfavorable"


def test_promotion_surfaces_premium_favorable_for_short(router, monkeypatch):
    """Direction-mirrored check -- the same premium zone that's unfavorable
    for a long is favorable for a short."""
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"PREM": _base_daily_frame()})
    candidate = _FakeRow(ticker="PREM", signal="short", entry_price=109.85, source="test")

    result = router._compute_candidate_promotion(candidate)

    assert result["location_label"] == "premium"
    assert result["location_alignment"] == "favorable"


def test_promotion_surfaces_midrange_neutral(router, monkeypatch):
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"MID": _base_daily_frame()})
    candidate = _FakeRow(ticker="MID", signal="long", entry_price=106.1, source="test")

    result = router._compute_candidate_promotion(candidate)

    assert result["location_percentile"] == 50.0
    assert result["location_label"] == "midrange"
    assert result["location_alignment"] == "neutral"


def test_location_never_blocks_promotion_or_changes_rr(router, monkeypatch):
    """The explicit design point: legacy's valid_zone can solo-cause 'RANGE
    / NO TRADE' and gates two whole tiers -- confirm a deep-premium (worst
    possible location for a long) promotion still computes normally here,
    with a real stop, and doesn't error/short-circuit because of location.
    (entry_price sits exactly at the recent swing high for this fixture,
    so no_valid_target is naturally True here -- an unrelated structural
    fact, same real coexistence seen on live production data (e.g. BBY),
    not location gating anything.)
    """
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"PREM": _base_daily_frame()})
    candidate = _FakeRow(ticker="PREM", signal="long", entry_price=109.85, source="test")

    result = router._compute_candidate_promotion(candidate)

    assert result["location_alignment"] == "unfavorable"
    assert result["stop"] is not None and result["stop"] > 0
    assert (result["risk_reward"] is not None) or result["no_valid_target"] is True


def test_favorable_location_also_computes_full_rr_normally(router, monkeypatch):
    """Cleaner positive confirmation alongside the unfavorable case above:
    a favorable-location promotion computes a real, unremarkable
    stop/target/risk_reward -- location isn't secretly influencing the
    numbers in either direction."""
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"DISC": _base_daily_frame()})
    candidate = _FakeRow(ticker="DISC", signal="long", entry_price=102.35, source="test")

    result = router._compute_candidate_promotion(candidate)

    assert result["location_alignment"] == "favorable"
    assert result["stop"] is not None and result["stop"] > 0
    assert result["risk_reward"] is not None and result["risk_reward"] > 0
    assert result["no_valid_target"] is False


# -- pydantic models ------------------------------------------------------------

def test_promotion_out_model_accepts_location_fields(router):
    payload = {
        "id": 1, "ticker": "PREM", "source": "test", "direction": "long", "entry_price": 109.85,
        "stop": 105.0, "target": 115.0, "risk_reward": 1.5, "rr_warning": False,
        "no_valid_target": False, "promoted_at": "2026-08-27T00:00:00Z", "position_size": None,
        "atr14": 1.5, "atr_multiplier": 1.5, "rr_warning_threshold": 1.5,
        "min_target_atr_multiple": 2.0, "target_source": "daily_swing_structure",
        "location_percentile": 100.0, "location_label": "premium", "location_alignment": "unfavorable",
    }
    out = router.CandidatePromotionOut(**payload)
    assert out.location_percentile == 100.0
    assert out.location_label == "premium"
    assert out.location_alignment == "unfavorable"


def test_promotion_out_model_defaults_location_fields_when_absent(router):
    payload = {
        "id": 2, "ticker": "MID", "source": "test", "direction": "long", "entry_price": 106.1,
        "stop": 102.0, "target": 112.0, "risk_reward": 2.0, "rr_warning": False,
        "no_valid_target": False, "promoted_at": "2026-08-27T00:00:00Z", "position_size": None,
        "atr14": 1.5, "atr_multiplier": 1.5, "rr_warning_threshold": 1.5,
        "min_target_atr_multiple": 2.0, "target_source": "daily_swing_structure",
    }
    out = router.CandidatePromotionOut(**payload)
    assert out.location_percentile is None
    assert out.location_label is None
    assert out.location_alignment is None


# -- DB round-trip --------------------------------------------------------------

def _open_db(router, tmp_path):
    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "candidates.db"))
    conn.row_factory = sqlite3.Row
    router._initialize_candidates_schema(conn)
    return conn


def test_store_and_reload_promotion_round_trips_location_fields(router, tmp_path):
    conn = _open_db(router, tmp_path)
    promotion = {
        "ticker": "PREM", "source": "test", "direction": "long", "entry_price": 109.85,
        "stop": 105.0, "target": 115.0, "risk_reward": 1.5, "rr_warning": False,
        "no_valid_target": False, "promoted_at": "2026-08-27T00:00:00Z", "position_size": None,
        "atr14": 1.5, "atr_multiplier": 1.5, "rr_warning_threshold": 1.5,
        "min_target_atr_multiple": 2.0, "target_source": "daily_swing_structure",
        "raw_target": 115.0, "raw_risk_reward": 1.5, "target_clamped": False,
        "target_clamp_badge": None, "target_clamp_reason": None,
        "raw_stop": 105.0, "stop_source": "atr_multiple",
        "displacement_score": 50.0, "displacement_label": "MODERATE",
        "displacement_components": None, "raw_magnitude_score": 55.0,
        "displacement_read": "favorable", "bos_confirmed": False, "bos_details": None,
        "macro_bias": "Macro Neutral", "macro_conflict": False,
        "choch_conflict": False, "choch_details": None,
        "sweep_confirmed": False, "sweep_details": None,
        "rejection_confirmed": False, "rejection_details": None,
        "location_percentile": 100.0, "location_label": "premium", "location_alignment": "unfavorable",
    }
    router._store_promotion(conn, promotion)
    conn.commit()

    row = conn.execute("SELECT * FROM candidate_promotions WHERE ticker=? AND source=?", ("PREM", "test")).fetchone()
    reloaded = router._row_to_promotion(row)

    assert reloaded["location_percentile"] == 100.0
    assert reloaded["location_label"] == "premium"
    assert reloaded["location_alignment"] == "unfavorable"
    conn.close()


def test_store_and_reload_promotion_round_trips_no_valid_location(router, tmp_path):
    conn = _open_db(router, tmp_path)
    promotion = {
        "ticker": "NOLOC", "source": "test", "direction": "long", "entry_price": 100.0,
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
        "macro_bias": "Macro Bullish", "macro_conflict": False,
        "choch_conflict": False, "choch_details": None,
        "sweep_confirmed": False, "sweep_details": None,
        "rejection_confirmed": False, "rejection_details": None,
        "location_percentile": None, "location_label": None, "location_alignment": None,
    }
    router._store_promotion(conn, promotion)
    conn.commit()

    row = conn.execute("SELECT * FROM candidate_promotions WHERE ticker=? AND source=?", ("NOLOC", "test")).fetchone()
    reloaded = router._row_to_promotion(row)

    assert reloaded["location_percentile"] is None
    assert reloaded["location_label"] is None
    assert reloaded["location_alignment"] is None
    conn.close()


def test_plan_preview_success_path_threads_location_fields(router, monkeypatch):
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"PREM": _base_daily_frame()})
    monkeypatch.setattr(router, "_safe_option_contract_for_candidate", lambda ticker, direction, entry, **kwargs: None)
    candidate = _FakeRow(
        ticker="PREM", signal="long", entry_price=109.85, source="test",
        updated_at="2026-08-27T00:00:00Z",
    )

    preview = router._compute_candidate_plan_preview(candidate)

    assert preview["preview_error"] is None
    assert preview["location_percentile"] == 100.0
    assert preview["location_label"] == "premium"
    assert preview["location_alignment"] == "unfavorable"


def test_plan_preview_exception_fallback_defaults_location_fields_safely(router, monkeypatch):
    def _raise(tickers, period, interval):
        raise router.HTTPException(status_code=422, detail="No daily candles available for BAD")

    monkeypatch.setattr(router, "_batch_download", _raise)
    candidate = _FakeRow(
        ticker="BAD", signal="long", entry_price=100.0, source="test",
        updated_at="2026-08-27T00:00:00Z",
    )

    preview = router._compute_candidate_plan_preview(candidate)

    assert preview["preview_error"] is not None
    assert preview["location_percentile"] is None
    assert preview["location_label"] is None
    assert preview["location_alignment"] is None
