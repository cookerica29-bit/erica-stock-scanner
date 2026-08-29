"""_entry_proximity's fallback-to-last-close design -- candidates_router.

Real-world motivation (confirmed on live data before implementing, not
assumed): a one-sided live quote (Alpaca's quotes endpoint defaults to the
IEX feed, which frequently has no recent two-sided print for a given name)
was making entry_proximity_ok default straight to False -- despite the
reason string literally saying "not used for gating." Traced every real
consumer and found it WAS used for gating in three places: the actual
promotion gate (_mechanical_promotion_block_reason, via
update_candidate_status and track_candidate_outcome) and the frontend's
routeBlockReason (the READY/ENTER_NOW badge). Confirmed real via two live
tickers on 2026-08-29:

  - ACGL: entry 100.76, ATR14 1.793. Live quote was bid_only ($93.57,
    untrustworthy). The real last daily close ($98.84) shows a genuine,
    confirmable proximity FAILURE (1.91% / 1.07 ATR away, over the 1.5% /
    0.5 ATR tolerance) -- a real gap the old behavior was masking behind a
    generic "quote is one-sided" caveat.
  - AER: entry 149.345, ATR14 3.0845. Live quote was bid_only ($125.86, off
    by ~$22 from reality -- worse than the BRK.B case that motivated the
    original one-sided guard). The real last daily close ($147.49) shows
    proximity actually PASSES (1.24% / 0.60 ATR, within the OR-of-both-
    thresholds tolerance) -- the old behavior was falsely blocking a
    candidate that was, on trustworthy data, already near entry.

Design: _entry_proximity now accepts fallback_close/fallback_close_at
(sourced from _compute_candidate_promotion's already-fetched daily df --
not a new network call) and uses them whenever the live quote is missing or
one-sided, instead of defaulting to "not ok." entry_proximity_price_source
("live_quote" | "fallback_daily_close" | None) is threaded through every
consumer specifically so a candidate that reads ok=True only via the
fallback is never visually indistinguishable from one confirmed against a
live price -- see ENTRY_PROXIMITY_FALLBACK_CLOSE_MAX_AGE_HOURS's comment in
candidates_router.py for the full staleness-threshold rationale.

Fixtures below were verified directly against _entry_proximity before being
fixed here -- not hand-derived from assumptions about what "should" pass.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _FakeRow(dict):
    def __getitem__(self, key):
        return dict.get(self, key)


def _iso_hours_ago(hours: float) -> str:
    return (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)).isoformat()


ONE_SIDED_QUOTE = {
    "price": 93.57, "bid": 93.57, "ask": 0, "price_branch": "bid_only",
    "timestamp": "2026-08-28T20:00:01Z", "source": "alpaca_latest_quote",
}
TWO_SIDED_QUOTE = {
    "price": 149.5, "bid": 149.4, "ask": 149.6, "price_branch": "mid",
    "timestamp": "2026-08-29T15:00:00Z", "source": "alpaca_latest_quote",
}


@pytest.fixture()
def router(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", str(tmp_path / "candidates.db"))
    monkeypatch.setenv("KAIROS_SCANNER_API_KEY", "test-scanner-key")
    import candidates_router
    return candidates_router


# -- _entry_proximity, direct -------------------------------------------------

def test_acgl_shaped_fallback_fails_with_real_numbers(router):
    """The ACGL calibration case: one-sided live quote, but the last close
    genuinely fails proximity (1.91% / 1.07 ATR vs a 1.5% / 0.5 ATR
    tolerance) -- verified real numbers, not a hand-derived guess."""
    result = router._entry_proximity(
        entry_price=100.76, atr14=1.793, quote=ONE_SIDED_QUOTE,
        fallback_close=98.84, fallback_close_at=_iso_hours_ago(20),
    )
    assert result["entry_proximity_ok"] is False
    assert result["entry_proximity_price_source"] == "fallback_daily_close"
    assert result["entry_distance_pct"] == 1.91
    assert result["entry_distance_atr"] == 1.07
    assert "1.91%" in result["entry_proximity_reason"]
    assert "1.07 ATR" in result["entry_proximity_reason"]
    assert "based on last close" in result["entry_proximity_reason"]


def test_aer_shaped_fallback_passes_with_real_numbers(router):
    """The AER calibration case: one-sided live quote off by ~$22 from
    reality, but the last close genuinely PASSES proximity (1.24% / 0.60
    ATR -- passes on the pct leg via the max-of-both-thresholds rule) --
    the old behavior falsely blocked this."""
    result = router._entry_proximity(
        entry_price=149.345, atr14=3.0845, quote=ONE_SIDED_QUOTE,
        fallback_close=147.49, fallback_close_at=_iso_hours_ago(20),
    )
    assert result["entry_proximity_ok"] is True
    assert result["entry_proximity_price_source"] == "fallback_daily_close"
    assert result["entry_distance_pct"] == 1.24
    assert result["current_price"] == 147.49
    # A PASS achieved only via the fallback is never silently
    # indistinguishable from a live-confirmed one -- the reason text still
    # names the source even though it's not a block reason.
    assert "confirmed near entry via last close" in result["entry_proximity_reason"]


def test_one_sided_quote_no_fallback_available_stays_unresolved(router):
    result = router._entry_proximity(
        entry_price=100.0, atr14=1.0, quote=ONE_SIDED_QUOTE,
        fallback_close=None, fallback_close_at=None,
    )
    assert result["entry_proximity_ok"] is False
    assert result["entry_proximity_price_source"] is None
    assert "one-sided" in result["entry_proximity_reason"]


def test_stale_fallback_close_is_rejected(router):
    """Older than ENTRY_PROXIMITY_FALLBACK_CLOSE_MAX_AGE_HOURS (120h) --
    must not be trusted as a stand-in, same as no fallback at all."""
    result = router._entry_proximity(
        entry_price=100.76, atr14=1.793, quote=ONE_SIDED_QUOTE,
        fallback_close=98.84, fallback_close_at=_iso_hours_ago(200),
    )
    assert result["entry_proximity_ok"] is False
    assert result["entry_proximity_price_source"] is None
    assert "too old" in result["entry_proximity_reason"]


def test_fresh_fallback_close_at_the_edge_is_accepted(router):
    """Just inside the 120h threshold -- must still be trusted (this is the
    real 'Friday close still valid Saturday' case, not just an arbitrary
    cutoff check)."""
    result = router._entry_proximity(
        entry_price=149.345, atr14=3.0845, quote=ONE_SIDED_QUOTE,
        fallback_close=147.49, fallback_close_at=_iso_hours_ago(119),
    )
    assert result["entry_proximity_ok"] is True
    assert result["entry_proximity_price_source"] == "fallback_daily_close"


def test_missing_quote_entirely_also_uses_fallback(router):
    """No live quote at all (not just one-sided) hits the same fallback
    path -- confirmed via the missing-quote branch, not just bid_only."""
    result = router._entry_proximity(
        entry_price=149.345, atr14=3.0845, quote=None,
        fallback_close=147.49, fallback_close_at=_iso_hours_ago(20),
    )
    assert result["entry_proximity_ok"] is True
    assert result["entry_proximity_price_source"] == "fallback_daily_close"
    assert "Current quote unavailable" in result["entry_proximity_reason"]


def test_two_sided_live_quote_ignores_fallback_entirely(router):
    """A good live quote is used as-is -- the fallback is never consulted,
    even when supplied, and price_source reads live_quote."""
    result = router._entry_proximity(
        entry_price=149.345, atr14=3.0845, quote=TWO_SIDED_QUOTE,
        fallback_close=147.49, fallback_close_at=_iso_hours_ago(20),
    )
    assert result["entry_proximity_ok"] is True
    assert result["entry_proximity_price_source"] == "live_quote"
    assert result["current_price"] == 149.5
    assert result["entry_proximity_reason"] is None


def test_entry_price_unavailable_short_circuits_before_any_fallback(router):
    result = router._entry_proximity(
        entry_price=None, atr14=1.0, quote=ONE_SIDED_QUOTE,
        fallback_close=147.49, fallback_close_at=_iso_hours_ago(20),
    )
    assert result["entry_proximity_ok"] is False
    assert result["entry_proximity_price_source"] is None
    assert result["entry_proximity_reason"] == "Entry price unavailable"


# -- wired through _compute_candidate_promotion / _compute_candidate_plan_preview --

def _acgl_daily_frame(n: int = 260, last_close: float = 98.84) -> pd.DataFrame:
    """Mild, unremarkable daily drift ending at the real ACGL last-close
    value -- shape doesn't matter for this test, only that the final Close
    matches the real number _compute_candidate_promotion should surface as
    last_daily_close. Index ends at "now" (not a fixed past date) so the
    resulting last_daily_close_at passes _fallback_close_is_fresh's real
    staleness check the same way a genuinely fresh download would."""
    rows = []
    for i in range(n):
        close = last_close - (n - 1 - i) * 0.01
        rows.append({"Open": close - 0.1, "High": close + 0.3, "Low": close - 0.3, "Close": close, "Volume": 1_000_000})
    end = pd.Timestamp.now(tz="UTC").normalize()
    index = pd.date_range(end=end, periods=n, freq="D", tz="UTC")
    return pd.DataFrame(rows, index=index)


def test_promotion_surfaces_last_daily_close_from_the_same_df(router, monkeypatch):
    """last_daily_close/_at must come from the SAME df already downloaded
    for macro bias/BOS/target/stop -- not a second fetch."""
    df = _acgl_daily_frame()
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"ACGL": df})
    candidate = _FakeRow(ticker="ACGL", signal="long", entry_price=100.76, source="test")

    promotion = router._compute_candidate_promotion(candidate)

    assert promotion["last_daily_close"] == pytest.approx(98.84, abs=0.01)
    assert promotion["last_daily_close_at"] is not None


def test_plan_preview_threads_last_daily_close_through(router, monkeypatch):
    df = _acgl_daily_frame()
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {"ACGL": df})
    monkeypatch.setattr(router, "_safe_option_contract_for_candidate", lambda ticker, direction, entry, **kwargs: None)
    candidate = _FakeRow(ticker="ACGL", signal="long", entry_price=100.76, source="test", updated_at="2026-08-29T00:00:00Z")

    preview = router._compute_candidate_plan_preview(candidate)

    assert preview["last_daily_close"] == pytest.approx(98.84, abs=0.01)
    assert preview["last_daily_close_at"] is not None


def test_plan_preview_exception_fallback_leaves_last_daily_close_none(router, monkeypatch):
    def _raise(tickers, period, interval):
        raise router.HTTPException(status_code=422, detail="No daily candles available for BAD")

    monkeypatch.setattr(router, "_batch_download", _raise)
    candidate = _FakeRow(ticker="BAD", signal="long", entry_price=100.0, source="test", updated_at="2026-08-29T00:00:00Z")

    preview = router._compute_candidate_plan_preview(candidate)

    assert preview["last_daily_close"] is None
    assert preview["last_daily_close_at"] is None


# -- DB round-trip (the CACHED preview path, not just freshly-computed) -------

def _open_db(tmp_path):
    import sqlite3
    conn = sqlite3.connect(str(tmp_path / "candidates.db"))
    conn.row_factory = sqlite3.Row
    return conn


def test_store_and_reload_plan_preview_round_trips_last_daily_close(router, tmp_path):
    """Critical for the fallback to work on a CACHED preview, not just a
    freshly-computed one -- _stale_plan_preview_candidates loads cached
    previews straight from this table via _row_to_plan_preview."""
    conn = _open_db(tmp_path)
    router._initialize_candidates_schema(conn)
    preview = {
        "ticker": "ACGL", "source": "test", "signal": "long", "entry_price": 100.76,
        "stop": 99.9657, "target": 104.9102, "risk_reward": 5.22, "rr_warning": False,
        "no_valid_target": False, "atr14": 1.793, "atr_multiplier": 1.5,
        "rr_warning_threshold": 1.5, "min_target_atr_multiple": 2.0,
        "target_source": "daily_swing_structure", "option_contract": None,
        "preview_error": None, "computed_at": "2026-08-29T00:00:00Z",
        "candidate_updated_at": "2026-08-29T00:00:00Z",
        "last_daily_close": 98.84, "last_daily_close_at": "2026-08-28T21:00:00+00:00",
    }
    router._store_plan_preview(conn, preview)
    conn.commit()

    row = conn.execute("SELECT * FROM candidate_plan_previews WHERE ticker=? AND source=?", ("ACGL", "test")).fetchone()
    reloaded = router._row_to_plan_preview(row)

    assert reloaded["last_daily_close"] == pytest.approx(98.84, abs=0.001)
    assert reloaded["last_daily_close_at"] == "2026-08-28T21:00:00+00:00"
    conn.close()


# -- the real gating fix: _mechanical_promotion_block_reason ------------------

def _valid_promotion_shape(**overrides) -> dict:
    """A promotion dict with every OTHER mechanical condition already
    satisfied (regime aligned via the candidate, valid target, R:R above
    threshold) so these tests isolate the entry_proximity branch of
    _mechanical_promotion_block_reason specifically -- not entangled with
    whether a synthetic price series happens to produce a valid structural
    target."""
    base = {
        "direction": "long",
        "no_valid_target": False,
        "target": 104.9102,
        "risk_reward": 5.22,
        "rr_warning": False,
        "execution_shadow_ok": True,
    }
    base.update(overrides)
    return base


def test_mechanical_block_reason_no_longer_blocks_on_fallback_pass(router):
    """The actual bug: a one-sided quote used to make
    _mechanical_promotion_block_reason (the real gate behind
    update_candidate_status and track_candidate_outcome) reject a candidate
    outright, even when its own reason text said 'not used for gating'.
    AER-shaped: fallback close passes -> no longer blocked on proximity."""
    candidate = _FakeRow(ticker="AER", signal="long", daily_regime="long")
    proximity = router._entry_proximity(
        entry_price=149.345, atr14=3.0845, quote=ONE_SIDED_QUOTE,
        fallback_close=147.49, fallback_close_at=_iso_hours_ago(20),
    )
    promotion = {**_valid_promotion_shape(entry_price=149.345, atr14=3.0845), **proximity}

    assert promotion["entry_proximity_ok"] is True
    assert promotion["entry_proximity_price_source"] == "fallback_daily_close"
    reason = router._mechanical_promotion_block_reason(candidate, promotion)
    assert reason is None


def test_mechanical_block_reason_still_blocks_on_genuine_fallback_failure(router):
    """ACGL-shaped: fallback close genuinely fails proximity -> still
    correctly blocked, now with the real distance in the reason instead of
    a masked generic caveat."""
    candidate = _FakeRow(ticker="ACGL", signal="long", daily_regime="long")
    proximity = router._entry_proximity(
        entry_price=100.76, atr14=1.793, quote=ONE_SIDED_QUOTE,
        fallback_close=98.84, fallback_close_at=_iso_hours_ago(20),
    )
    promotion = {**_valid_promotion_shape(entry_price=100.76, atr14=1.793), **proximity}

    assert promotion["entry_proximity_ok"] is False
    assert promotion["entry_proximity_price_source"] == "fallback_daily_close"
    reason = router._mechanical_promotion_block_reason(candidate, promotion)
    assert reason is not None
    assert "1.91%" in reason
    assert "1.07 ATR" in reason
    assert "based on last close" in reason


# -- near-miss display fix: _gate_gap_report ----------------------------------

def test_gate_gap_report_shows_real_distance_for_fallback_failure(router, monkeypatch):
    """The secondary display fix: ACGL's near-miss card must show the real,
    confirmable distance-vs-tolerance numbers, not just the generic
    quote-caveat text, once the fallback resolves a real distance."""
    preview = {
        "signal": "long", "confluence_label": "some confluence",
        "no_valid_target": False, "target": 104.9102, "risk_reward": 5.22, "rr_warning": False,
        "entry_proximity_ok": False, "entry_distance_pct": 1.91, "entry_distance_atr": 1.07,
        "entry_proximity_threshold_pct": 1.5, "entry_proximity_threshold_atr": 0.5,
        "entry_proximity_price_source": "fallback_daily_close",
        "entry_proximity_reason": "Price moved 1.91% / 1.07 ATR away from entry (based on last close -- live quote unavailable)",
        "execution_shadow_checked": False, "execution_shadow_ok": None,
    }
    candidate = _FakeRow(ticker="ACGL", signal="long", daily_regime="long")

    report = router._gate_gap_report(candidate, preview)

    assert report["categorical_blocked"] is False
    proximity_gaps = [g for g in report["gaps"] if g["condition"] == "entry_proximity"]
    assert len(proximity_gaps) == 1
    detail = proximity_gaps[0]["detail"]
    assert "1.91%" in detail
    assert "1.07 ATR" in detail
    assert "last close" in detail
    assert "live quote unavailable" in detail


def test_gate_gap_report_no_proximity_gap_for_fallback_pass(router):
    """AER-shaped: once the fallback resolves proximity as OK, it must not
    appear as a gap at all."""
    preview = {
        "signal": "long", "confluence_label": "some confluence",
        "no_valid_target": False, "target": 152.0, "risk_reward": 2.02, "rr_warning": False,
        "entry_proximity_ok": True, "entry_distance_pct": 1.24, "entry_distance_atr": 0.60,
        "entry_proximity_threshold_pct": 1.5, "entry_proximity_threshold_atr": 0.5,
        "entry_proximity_price_source": "fallback_daily_close",
        "entry_proximity_reason": None,
        "execution_shadow_checked": False, "execution_shadow_ok": None,
    }
    candidate = _FakeRow(ticker="AER", signal="long", daily_regime="long")

    report = router._gate_gap_report(candidate, preview)

    proximity_gaps = [g for g in report["gaps"] if g["condition"] == "entry_proximity"]
    assert proximity_gaps == []
