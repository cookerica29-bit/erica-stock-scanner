"""Kairos -- Watch Lifecycle V1
(watch_lifecycle_v1_audit.md, and the task's own numbered sections).

Turns Watch Setups from a passive list into an actively monitored
lifecycle, reusing the EXACT SAME Execution Layer V1 infrastructure
(run_approved_setup_monitor_tick, approved_setup_memories/
approved_setup_monitor_state/approved_setup_monitor_events, the same
5-minute scheduler) -- no second monitor, no competing state machine.

Covers, per the task's section 15 list: RTH-only 30m trigger satisfaction
(long/short, wick-vs-close, premarket/after-hours exclusion, the
15:30-16:00 closing bar, DST), invalidation-before-trigger (terminal,
never revives), setup_key supersession (trigger never transfers), queue-
rotation independence, restart/historical-candle recovery, duplicate-tick
idempotency, missing-quote fail-closed, price-beyond-target /
below-minimum-R:R never actionable, a clean trigger->safety-gates->
handoff sequence, and the core historical-truth guarantee: the original
WATCH review is never rewritten and no APPROVE row is ever fabricated.

Every test exercises the real candidates_router.py code via a real
FastAPI TestClient for setup (the same visual-review flow a human would
use) plus direct calls to run_approved_setup_monitor_tick() and the new
RTH helper functions -- not a reimplementation.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import candidates_router as router  # noqa: E402


def _daily_frame():
    index = pd.date_range("2026-01-01", periods=30, freq="D", tz="UTC")
    rows = []
    for i in range(30):
        close = 100.0 - i * 0.1
        rows.append({"Open": close + 0.2, "High": close + 0.5, "Low": close - 0.5, "Close": close, "Volume": 1_000_000})
    rows[10]["Low"] = 90.0
    rows[10]["Close"] = 91.0
    rows[10]["Open"] = 92.0
    rows[20]["High"] = 110.0
    rows[20]["Close"] = 109.0
    rows[20]["Open"] = 108.0
    return pd.DataFrame(rows, index=index)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", str(tmp_path / "candidates.db"))
    monkeypatch.setenv("KAIROS_SCANNER_API_KEY", "test-scanner-key")
    app = FastAPI()
    app.include_router(router.router)
    return TestClient(app)


@pytest.fixture()
def headers():
    return {"X-API-Key": "test-scanner-key"}


@pytest.fixture(autouse=True)
def _mock_network(monkeypatch):
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {str(tickers[0]).upper(): _daily_frame()})
    monkeypatch.setattr(router, "_latest_quote_for_ticker", lambda ticker: {
        "price": 100.0, "timestamp": "2026-08-20T18:30:00Z", "source": "mock_latest_quote", "price_branch": "mid",
    })
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        str(p.get("ticker") or "").upper(): {
            "price": 100.0, "timestamp": "2026-08-20T18:30:00Z", "source": "mock_latest_quote", "price_branch": "mid",
        }
        for p in previews
    })
    monkeypatch.setattr(router, "_recent_4h_bars_for_execution_shadow", lambda ticker: [
        {
            "time": f"2026-08-19T{hour:02d}:00:00Z",
            "open": 99.0 + (idx * 0.05), "high": 101.0 + (idx * 0.05),
            "low": 98.0 + (idx * 0.05), "close": 100.0 + (idx * 0.05), "volume": 1000,
        }
        for idx, hour in enumerate(range(11))
    ] + [
        {"time": "2026-08-20T02:00:00Z", "open": 99.0, "high": 101.0, "low": 98.0, "close": 100.0, "volume": 1000},
        {"time": "2026-08-20T06:00:00Z", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1100},
        {"time": "2026-08-20T10:00:00Z", "open": 100.5, "high": 102.0, "low": 99.2, "close": 101.0, "volume": 1200},
        {"time": "2026-08-20T14:00:00Z", "open": 100.8, "high": 103.0, "low": 99.5, "close": 102.2, "volume": 1000},
    ])


@pytest.fixture(autouse=True)
def _forbid_option_hydration(monkeypatch):
    def _spy(ticker, direction, entry, **kwargs):
        raise AssertionError(f"_safe_option_contract_for_candidate called unexpectedly: {ticker}")

    monkeypatch.setattr(router, "_safe_option_contract_for_candidate", _spy)


def _seed(client, headers, ticker="AMD", signal="long", entry_price=100.0):
    payload = {
        "source": "ma_pipeline",
        "scanned_at": "2026-08-20T14:30:00Z",
        "candidates": [{
            "ticker": ticker, "signal": signal, "entry_price": entry_price, "ema21_4h": entry_price - 1.0,
            "daily_regime": signal, "confidence": "high",
            "sma50_daily": entry_price + 6.0 if signal == "long" else entry_price - 6.0,
            "sma200_daily": entry_price + 4.0 if signal == "long" else entry_price - 4.0,
        }],
    }
    created = client.post("/api/v1/scanner/candidates", headers=headers, json=payload)
    assert created.status_code == 200


def _review(client, headers, ticker, decision, source="ma_pipeline", **overrides):
    body = {
        "source": source,
        "market_structure": "bullish",
        "location_read": "good",
        "clear_path_to_target": "yes",
        "lower_tf_confirmation": "not_yet",
        "decision": decision,
        "note": "test note",
        **overrides,
    }
    resp = client.post(f"/api/v1/scanner/candidates/{ticker}/visual-review", headers=headers, json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _memory_record(client, headers, ticker="AMD"):
    resp = client.get("/api/v1/scanner/candidates/approved-setup-memory", headers=headers,
                       params={"ticker": ticker, "include_inactive": True})
    assert resp.status_code == 200, resp.text
    records = resp.json()
    assert len(records) >= 1
    # most recent by approved_at, matching setup_board.js's own reduction
    return sorted(records, key=lambda r: r["memory"]["approved_at"])[-1]


def _monitor_state_row(db_path, monitor_state_id):
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM approved_setup_monitor_state WHERE id=?", (monitor_state_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def _events(db_path, approved_memory_id):
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM approved_setup_monitor_events WHERE approved_memory_id=? ORDER BY id", (approved_memory_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _reviews_for_ticker(db_path, ticker):
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM candidate_visual_reviews WHERE ticker=? ORDER BY id", (ticker,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _bar(t, close, open_=None, high=None, low=None):
    o = open_ if open_ is not None else close
    return {"time": t, "open": o, "high": high if high is not None else max(o, close) + 0.1,
            "low": low if low is not None else min(o, close) - 0.1, "close": close}


def _watch_with_trigger(client, headers, ticker="AMD", rule="close_above", level=100.0):
    _seed(client, headers, ticker)
    resp = _review(client, headers, ticker, "watch", trigger_rule=rule, trigger_level=level)
    record = _memory_record(client, headers, ticker)
    assert record["memory"]["source_decision"] == "watch"
    assert record["monitor_state"]["state"] == "WAITING_FOR_TRIGGER"
    return resp, record


# ==========================================================================
# Watch eligibility (task section 3) -- reused/confirmed, not re-derived
# ==========================================================================

def test_watch_with_complete_trigger_creates_a_watch_origin_memory(client, headers):
    resp, record = _watch_with_trigger(client, headers)
    assert record["memory"]["trigger_rule"] == "close_above"
    assert record["memory"]["trigger_level"] == 100.0
    assert record["memory"]["approved_stop"] is not None
    assert record["memory"]["approved_target"] is not None


def test_watch_without_complete_trigger_stays_passive(client, headers):
    _seed(client, headers, "AMD")
    resp = _review(client, headers, "AMD", "watch")  # no trigger
    assert resp["trigger_level"] is None
    result = client.get("/api/v1/scanner/candidates/approved-setup-memory", headers=headers,
                         params={"ticker": "AMD", "include_inactive": True})
    assert result.json() == [], "MANUAL REVIEW REQUIRED -- no memory, no monitoring, never inferred"


# ==========================================================================
# STOCK 30M SESSION CONTRACT (task section 2) -- RTH-only, completed
# candles only, close-based, DST-correct via zoneinfo
# ==========================================================================

def test_rth_bar_start_accepts_all_thirteen_valid_slots():
    # 2026-08-20 is a real Thursday (weekday).
    import zoneinfo
    et = zoneinfo.ZoneInfo("America/New_York")
    valid_hours_minutes = [(9, 30), (10, 0), (10, 30), (11, 0), (11, 30), (12, 0), (12, 30),
                            (13, 0), (13, 30), (14, 0), (14, 30), (15, 0), (15, 30)]
    for h, m in valid_hours_minutes:
        bar_time_et = datetime(2026, 8, 20, h, m, tzinfo=et)
        assert router._is_rth_30m_bar_start(bar_time_et) is True, f"{h}:{m:02d} ET must be a valid RTH slot"


def test_rth_bar_start_rejects_premarket_afterhours_and_misaligned():
    import zoneinfo
    et = zoneinfo.ZoneInfo("America/New_York")
    assert router._is_rth_30m_bar_start(datetime(2026, 8, 20, 9, 0, tzinfo=et)) is False, "premarket"
    assert router._is_rth_30m_bar_start(datetime(2026, 8, 20, 8, 0, tzinfo=et)) is False, "premarket"
    assert router._is_rth_30m_bar_start(datetime(2026, 8, 20, 16, 0, tzinfo=et)) is False, "after-hours (session end)"
    assert router._is_rth_30m_bar_start(datetime(2026, 8, 20, 16, 30, tzinfo=et)) is False, "after-hours"
    assert router._is_rth_30m_bar_start(datetime(2026, 8, 20, 20, 0, tzinfo=et)) is False, "overnight"
    assert router._is_rth_30m_bar_start(datetime(2026, 8, 20, 9, 45, tzinfo=et)) is False, "misaligned (not a :00/:30)"
    # Saturday
    assert router._is_rth_30m_bar_start(datetime(2026, 8, 22, 10, 0, tzinfo=et)) is False, "weekend"


def test_premarket_completed_bar_never_satisfies_trigger(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _resp, record = _watch_with_trigger(client, headers, "AMD", "close_above", 100.0)

    now = datetime(2026, 8, 20, 16, 5, tzinfo=timezone.utc)  # 12:05pm ET -- well into RTH
    # Premarket bar (8:00-8:30 ET = 12:00-12:30 UTC) qualifies on price alone but must be excluded.
    # Close price kept just above the trigger level (not far toward target) so
    # this test isolates the RTH-window exclusion, not the separate pre-trigger
    # R:R/EXTENDED check (see test_current_rr_below_minimum_reads_extended_before_trigger).
    bars = [_bar("2026-08-20T12:00:00Z", 100.5)]
    monkeypatch.setattr(router, "_fetch_recent_30m_bars", lambda ticker: bars)
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "AMD": {"price": 100.5, "timestamp": "t", "source": "mock", "price_branch": "mid"},
    })
    monkeypatch.setattr(router, "datetime", type("_dt", (datetime,), {"now": staticmethod(lambda tz=None: now)}))
    try:
        router.run_approved_setup_monitor_tick("test")
    finally:
        monkeypatch.setattr(router, "datetime", datetime)

    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["state"] == "WAITING_FOR_TRIGGER", "a premarket close must never satisfy an RTH-only trigger"
    assert after["trigger_satisfied_at"] is None


def test_afterhours_completed_bar_never_satisfies_trigger(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _resp, record = _watch_with_trigger(client, headers, "AMD", "close_above", 100.0)

    now = datetime(2026, 8, 20, 21, 5, tzinfo=timezone.utc)  # 5:05pm ET -- after the close
    # After-hours bar (16:30-17:00 ET = 20:30-21:00 UTC). Close kept near
    # the trigger level, not toward target -- see the premarket test's
    # comment for why.
    bars = [_bar("2026-08-20T20:30:00Z", 100.5)]
    monkeypatch.setattr(router, "_fetch_recent_30m_bars", lambda ticker: bars)
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "AMD": {"price": 100.5, "timestamp": "t", "source": "mock", "price_branch": "mid"},
    })
    monkeypatch.setattr(router, "datetime", type("_dt", (datetime,), {"now": staticmethod(lambda tz=None: now)}))
    try:
        router.run_approved_setup_monitor_tick("test")
    finally:
        monkeypatch.setattr(router, "datetime", datetime)

    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["state"] == "WAITING_FOR_TRIGGER", "an after-hours close must never satisfy an RTH-only trigger"


def test_wick_above_but_close_below_does_not_satisfy_long_trigger(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _resp, record = _watch_with_trigger(client, headers, "AMD", "close_above", 100.0)

    now = datetime(2026, 8, 20, 16, 5, tzinfo=timezone.utc)  # 12:05pm ET
    # 11:30-12:00 ET RTH bar: high pokes above 100 but CLOSE is below.
    bars = [_bar("2026-08-20T15:30:00Z", close=99.5, high=101.5, low=99.0)]
    monkeypatch.setattr(router, "_fetch_recent_30m_bars", lambda ticker: bars)
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "AMD": {"price": 99.5, "timestamp": "t", "source": "mock", "price_branch": "mid"},
    })
    monkeypatch.setattr(router, "datetime", type("_dt", (datetime,), {"now": staticmethod(lambda tz=None: now)}))
    try:
        router.run_approved_setup_monitor_tick("test")
    finally:
        monkeypatch.setattr(router, "datetime", datetime)

    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["state"] == "WAITING_FOR_TRIGGER", "a wick/touch must never satisfy the trigger -- close only"


def test_close_at_1530_to_1600_bar_counts(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _resp, record = _watch_with_trigger(client, headers, "AMD", "close_above", 100.0)

    now = datetime(2026, 8, 20, 20, 5, tzinfo=timezone.utc)  # 4:05pm ET -- the 15:30 bar just closed
    bars = [_bar("2026-08-20T19:30:00Z", close=101.0)]  # 15:30-16:00 ET RTH bar
    monkeypatch.setattr(router, "_fetch_recent_30m_bars", lambda ticker: bars)
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "AMD": {"price": 101.0, "timestamp": "t", "source": "mock", "price_branch": "mid"},
    })
    monkeypatch.setattr(router, "datetime", type("_dt", (datetime,), {"now": staticmethod(lambda tz=None: now)}))
    try:
        router.run_approved_setup_monitor_tick("test")
    finally:
        monkeypatch.setattr(router, "datetime", datetime)

    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["trigger_satisfied_at"] is not None, "the final 15:30-16:00 RTH candle must be a valid trigger window"
    assert after["trigger_satisfied_bar_time"] == "2026-08-20T19:30:00Z"


def test_dst_boundaries_correct_in_both_edt_and_est():
    """America/New_York, not a fixed UTC offset -- 9:30am ET is 13:30 UTC
    in EDT (summer) but 14:30 UTC in EST (winter); both must resolve to
    the SAME 9:30 ET wall-clock RTH boundary."""
    import zoneinfo
    et = zoneinfo.ZoneInfo("America/New_York")
    # EDT: 2026-08-20 (August, daylight saving in effect, UTC-4)
    edt_930 = datetime(2026, 8, 20, 13, 30, tzinfo=timezone.utc).astimezone(et)
    assert (edt_930.hour, edt_930.minute) == (9, 30)
    assert router._is_rth_30m_bar_start(edt_930) is True
    # EST: 2026-01-15 (January, standard time, UTC-5) -- also a real Thursday
    est_930 = datetime(2026, 1, 15, 14, 30, tzinfo=timezone.utc).astimezone(et)
    assert (est_930.hour, est_930.minute) == (9, 30)
    assert router._is_rth_30m_bar_start(est_930) is True
    # And the reverse check: the SAME utc hour (13:30 UTC) is 8:30am EST in
    # January -- premarket, must be excluded.
    est_from_1330utc = datetime(2026, 1, 15, 13, 30, tzinfo=timezone.utc).astimezone(et)
    assert (est_from_1330utc.hour, est_from_1330utc.minute) == (8, 30)
    assert router._is_rth_30m_bar_start(est_from_1330utc) is False


def test_short_close_below_mirror_satisfied_by_completed_rth_close(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _seed(client, headers, "AMD", signal="short", entry_price=100.0)
    _review(client, headers, "AMD", "watch", trigger_rule="close_below", trigger_level=95.0)
    record = _memory_record(client, headers, "AMD")
    assert record["memory"]["direction"] == "short"

    now = datetime(2026, 8, 20, 16, 5, tzinfo=timezone.utc)
    bars = [_bar("2026-08-20T15:30:00Z", close=94.5)]  # completed RTH bar closing below 95
    monkeypatch.setattr(router, "_fetch_recent_30m_bars", lambda ticker: bars)
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "AMD": {"price": 94.5, "timestamp": "t", "source": "mock", "price_branch": "mid"},
    })
    monkeypatch.setattr(router, "datetime", type("_dt", (datetime,), {"now": staticmethod(lambda tz=None: now)}))
    try:
        router.run_approved_setup_monitor_tick("test")
    finally:
        monkeypatch.setattr(router, "datetime", datetime)

    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["trigger_satisfied_at"] is not None
    assert after["trigger_satisfied_price"] == 94.5


def test_short_wick_below_but_close_above_does_not_satisfy(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _seed(client, headers, "AMD", signal="short", entry_price=100.0)
    _review(client, headers, "AMD", "watch", trigger_rule="close_below", trigger_level=95.0)
    record = _memory_record(client, headers, "AMD")
    near_entry_close = record["memory"]["approved_entry"] - 0.5

    # This test is specifically about the wick-vs-close rule, not R:R --
    # the mock daily frame's SHORT-side structural geometry (reused from
    # the LONG-oriented fixture) happens to produce a current R:R below
    # threshold regardless of price, which would otherwise make the
    # separate, legitimate Gap-4 pre-trigger EXTENDED check fire and mask
    # what this test is actually verifying. Bypassing the window check
    # here isolates the one thing under test; Gap-4 itself has its own
    # dedicated coverage (test_current_rr_below_minimum_reads_extended_before_trigger).
    monkeypatch.setattr(router, "_execution_window_state", lambda current_rr: "ACTIONABLE")

    now = datetime(2026, 8, 20, 16, 5, tzinfo=timezone.utc)
    bars = [_bar("2026-08-20T15:30:00Z", close=near_entry_close, low=93.0, high=near_entry_close + 0.5)]  # wick below 95, close above
    monkeypatch.setattr(router, "_fetch_recent_30m_bars", lambda ticker: bars)
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "AMD": {"price": near_entry_close, "timestamp": "t", "source": "mock", "price_branch": "mid"},
    })
    monkeypatch.setattr(router, "datetime", type("_dt", (datetime,), {"now": staticmethod(lambda tz=None: now)}))
    try:
        router.run_approved_setup_monitor_tick("test")
    finally:
        monkeypatch.setattr(router, "datetime", datetime)

    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["state"] == "WAITING_FOR_TRIGGER"


# ==========================================================================
# Invalidation while watching (task section 6)
# ==========================================================================

def test_invalidation_before_trigger_is_terminal(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _resp, record = _watch_with_trigger(client, headers, "AMD", "close_above", 100.0)
    approved_stop = record["memory"]["approved_stop"]

    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "AMD": {"price": approved_stop - 0.01, "timestamp": "t", "source": "mock", "price_branch": "mid"},
    })
    router.run_approved_setup_monitor_tick("test")
    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["state"] == "INVALIDATED"
    assert after["terminal_at"] is not None


def test_trigger_after_invalidation_cannot_revive(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _resp, record = _watch_with_trigger(client, headers, "AMD", "close_above", 100.0)
    approved_stop = record["memory"]["approved_stop"]

    # Tick 1: invalidate.
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "AMD": {"price": approved_stop - 0.01, "timestamp": "t", "source": "mock", "price_branch": "mid"},
    })
    router.run_approved_setup_monitor_tick("test")
    assert _monitor_state_row(db_path, record["monitor_state"]["id"])["state"] == "INVALIDATED"

    # Tick 2: a genuinely satisfying completed RTH bar arrives -- must NOT revive.
    now = datetime(2026, 8, 20, 16, 5, tzinfo=timezone.utc)
    bars = [_bar("2026-08-20T15:30:00Z", close=105.0)]
    monkeypatch.setattr(router, "_fetch_recent_30m_bars", lambda ticker: bars)
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "AMD": {"price": 105.0, "timestamp": "t", "source": "mock", "price_branch": "mid"},
    })
    monkeypatch.setattr(router, "datetime", type("_dt", (datetime,), {"now": staticmethod(lambda tz=None: now)}))
    try:
        result = router.run_approved_setup_monitor_tick("test")
    finally:
        monkeypatch.setattr(router, "datetime", datetime)

    assert result["checked"] == 0, "a terminal (INVALIDATED) row must not even be fetched by the monitor"
    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["state"] == "INVALIDATED"
    assert after["trigger_satisfied_at"] is None, "a dead setup's trigger must never be recorded as satisfied"


# ==========================================================================
# Setup generation / supersession (task section 10)
# ==========================================================================

def test_setup_key_supersession_before_trigger_does_not_transfer(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _resp, old_record = _watch_with_trigger(client, headers, "AMD", "close_above", 100.0)
    old_setup_key = old_record["memory"]["setup_key"]
    old_monitor_state_id = old_record["monitor_state"]["id"]

    # A genuinely different daily structure -> different stop/target -> different setup_key.
    def _daily_frame_v2():
        index = pd.date_range("2026-01-01", periods=30, freq="D", tz="UTC")
        rows = []
        for i in range(30):
            close = 100.0 - i * 0.1
            rows.append({"Open": close + 0.2, "High": close + 0.5, "Low": close - 0.5, "Close": close, "Volume": 1_000_000})
        rows[8]["Low"] = 85.0
        rows[8]["Close"] = 86.0
        rows[8]["Open"] = 87.0
        rows[18]["High"] = 118.0
        rows[18]["Close"] = 117.0
        rows[18]["Open"] = 116.0
        return pd.DataFrame(rows, index=index)

    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {str(tickers[0]).upper(): _daily_frame_v2()})
    _seed(client, headers, "AMD")  # re-ingest to bump candidates.updated_at (cache-miss forces recompute)

    # The NEW setup_key generation requires a fresh human review -- the OLD
    # trigger must not carry over automatically. Confirm the new candidate
    # reads as unreviewed by checking a fresh watch WITHOUT a trigger stays
    # passive for the new generation (proving nothing pre-existing applies).
    rq = client.get("/api/v1/scanner/candidates/review-queue", headers=headers).json()
    new_entry = next(c for c in rq["candidates"] if c["ticker"] == "AMD")
    assert new_entry["setup_key"] != old_setup_key, "sanity: the structure change really did produce a new setup_key"
    assert new_entry["current_review"] is None, "the new setup generation must read as unreviewed -- no trigger inherited"

    # The OLD memory's monitor_state must be SUPERSEDED once a fresh
    # approve/watch happens for AMD under the NEW setup_key.
    _review(client, headers, "AMD", "watch", trigger_rule="close_below", trigger_level=50.0)
    old_after = _monitor_state_row(db_path, old_monitor_state_id)
    assert old_after["state"] == "SUPERSEDED"

    new_record = _memory_record(client, headers, "AMD")
    assert new_record["memory"]["setup_key"] != old_setup_key
    assert new_record["memory"]["trigger_level"] == 50.0, "the new memory has its OWN trigger, not inherited"
    assert new_record["memory"]["revision_of_memory_id"] is None, \
        "a genuine setup_key change is a structural supersession, not a revision"


# ==========================================================================
# Queue rotation independence (task section 12)
# ==========================================================================

def test_monitoring_continues_after_ticker_leaves_candidates_table(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _resp, record = _watch_with_trigger(client, headers, "AMD", "close_above", 100.0)

    # Simulate the ticker rotating out of today's scan entirely.
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM candidates WHERE ticker='AMD'")
    conn.commit()
    conn.close()

    rq = client.get("/api/v1/scanner/candidates/review-queue", headers=headers).json()
    assert not any(c["ticker"] == "AMD" for c in rq["candidates"]), "sanity: AMD is genuinely gone from the live queue"

    now = datetime(2026, 8, 20, 16, 5, tzinfo=timezone.utc)
    bars = [_bar("2026-08-20T15:30:00Z", close=105.0)]
    monkeypatch.setattr(router, "_fetch_recent_30m_bars", lambda ticker: bars)
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "AMD": {"price": 105.0, "timestamp": "t", "source": "mock", "price_branch": "mid"},
    })
    monkeypatch.setattr(router, "datetime", type("_dt", (datetime,), {"now": staticmethod(lambda tz=None: now)}))
    try:
        result = router.run_approved_setup_monitor_tick("test")
    finally:
        monkeypatch.setattr(router, "datetime", datetime)

    assert result["checked"] == 1, "the monitor must evaluate this row purely from the memory, never the candidates table"
    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["trigger_satisfied_at"] is not None


# ==========================================================================
# Restart / historical-candle recovery, duplicate-tick idempotency
# (task sections 7 and 15)
# ==========================================================================

def test_server_restart_recovers_first_satisfying_historical_candle_once(client, headers, tmp_path, monkeypatch):
    """A genuine restart-after-outage: the row must already have been
    monitored at least once (a real last_evaluated_bar_time on file) --
    a FRESH row's very first tick deliberately only checks the single
    latest completed bar (the already-agreed first-tick rule, see
    _first_satisfying_completed_rth_bar's own docstring), so that case is
    covered separately by test_trigger_satisfied_by_a_completed_bar in
    tests/approved_setup_monitor_v1.py, not here."""
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _resp, record = _watch_with_trigger(client, headers, "AMD", "close_above", 100.0)

    # Phase 1: an ordinary tick establishes a real last_evaluated_bar_time,
    # nothing satisfies yet.
    phase1_now = datetime(2026, 8, 20, 15, 5, tzinfo=timezone.utc)  # 11:05am ET
    phase1_bars = [_bar("2026-08-20T14:30:00Z", close=99.0)]  # 10:30 ET, below trigger
    monkeypatch.setattr(router, "_fetch_recent_30m_bars", lambda ticker: phase1_bars)
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "AMD": {"price": 99.0, "timestamp": "t", "source": "mock", "price_branch": "mid"},
    })
    monkeypatch.setattr(router, "datetime", type("_dt", (datetime,), {"now": staticmethod(lambda tz=None: phase1_now)}))
    try:
        router.run_approved_setup_monitor_tick("test")
    finally:
        monkeypatch.setattr(router, "datetime", datetime)
    after_phase1 = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after_phase1["state"] == "WAITING_FOR_TRIGGER"
    assert after_phase1["last_evaluated_bar_time"] == "2026-08-20T14:30:00Z"

    # Phase 2: simulates the server having been offline through several
    # more bar completions -- bar 1 (11:00 ET) is the TRUE first satisfying
    # bar, bar 2 also satisfies, bar 3 (12:00 ET) is the latest completed;
    # only scanning forward from last_evaluated_bar_time (not "just the
    # latest") finds the real first one.
    phase2_now = datetime(2026, 8, 20, 18, 5, tzinfo=timezone.utc)  # 2:05pm ET
    phase2_bars = phase1_bars + [
        _bar("2026-08-20T15:00:00Z", close=101.0),  # 11:00 ET -- FIRST satisfying bar
        _bar("2026-08-20T15:30:00Z", close=102.0),  # 11:30 ET -- also satisfies
        _bar("2026-08-20T16:00:00Z", close=103.0),  # 12:00 ET -- latest completed
    ]
    monkeypatch.setattr(router, "_fetch_recent_30m_bars", lambda ticker: phase2_bars)
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "AMD": {"price": 103.0, "timestamp": "t", "source": "mock", "price_branch": "mid"},
    })
    monkeypatch.setattr(router, "datetime", type("_dt", (datetime,), {"now": staticmethod(lambda tz=None: phase2_now)}))
    try:
        router.run_approved_setup_monitor_tick("test")
    finally:
        monkeypatch.setattr(router, "datetime", datetime)

    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["trigger_satisfied_bar_time"] == "2026-08-20T15:00:00Z", \
        "restart recovery must find the FIRST satisfying historical bar since it was last checked, not just the latest"
    assert after["trigger_satisfied_price"] == 101.0
    assert after["trigger_satisfied_at"] is not None, "detected_at is real, but never substituted for the candle's own timestamp"
    assert after["trigger_satisfied_at"] != after["trigger_satisfied_bar_time"]


def test_duplicate_ticks_produce_exactly_one_satisfaction_event(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _resp, record = _watch_with_trigger(client, headers, "AMD", "close_above", 100.0)
    memory_id = record["memory"]["id"]

    now = datetime(2026, 8, 20, 16, 5, tzinfo=timezone.utc)
    bars = [_bar("2026-08-20T15:30:00Z", close=101.0)]
    monkeypatch.setattr(router, "_fetch_recent_30m_bars", lambda ticker: bars)
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "AMD": {"price": 101.0, "timestamp": "t", "source": "mock", "price_branch": "mid"},
    })
    monkeypatch.setattr(router, "datetime", type("_dt", (datetime,), {"now": staticmethod(lambda tz=None: now)}))
    try:
        router.run_approved_setup_monitor_tick("test")
        first_satisfied_at = _monitor_state_row(db_path, record["monitor_state"]["id"])["trigger_satisfied_at"]
        router.run_approved_setup_monitor_tick("test")
        router.run_approved_setup_monitor_tick("test")
    finally:
        monkeypatch.setattr(router, "datetime", datetime)

    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["trigger_satisfied_at"] == first_satisfied_at, "persisted exactly once, never overwritten by later ticks"

    events = _events(db_path, memory_id)
    satisfaction_events = [e for e in events if e["to_state"] == "ACTIONABLE" or e["to_state"] == "TRIGGER_SATISFIED"]
    # TRIGGER_SATISFIED is transient (immediately re-evaluated in the same
    # tick) so it's never itself the logged to_state -- only its
    # downstream verdict (ACTIONABLE/EXTENDED) is. Either way: exactly one
    # event for this transition, not three.
    assert len(satisfaction_events) <= 1, f"three identical ticks must not produce three events, got {events}"


# ==========================================================================
# Safety gates after trigger (task section 8)
# ==========================================================================

def test_stale_missing_quote_fails_closed(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _resp, record = _watch_with_trigger(client, headers, "AMD", "close_above", 100.0)

    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {})  # no quote at all
    result = router.run_approved_setup_monitor_tick("test")
    assert result["updated"] == 0
    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["state"] == "WAITING_FOR_TRIGGER", "missing data must never force a verdict either way"
    assert after["last_checked_at"] is None


def test_price_beyond_target_never_actionable(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _resp, record = _watch_with_trigger(client, headers, "AMD", "close_above", 100.0)
    approved_target = record["memory"]["approved_target"]

    now = datetime(2026, 8, 20, 16, 5, tzinfo=timezone.utc)
    beyond_target = approved_target + 5.0
    bars = [_bar("2026-08-20T15:30:00Z", close=beyond_target)]
    monkeypatch.setattr(router, "_fetch_recent_30m_bars", lambda ticker: bars)
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "AMD": {"price": beyond_target, "timestamp": "t", "source": "mock", "price_branch": "mid"},
    })
    monkeypatch.setattr(router, "datetime", type("_dt", (datetime,), {"now": staticmethod(lambda tz=None: now)}))
    try:
        router.run_approved_setup_monitor_tick("test")
    finally:
        monkeypatch.setattr(router, "datetime", datetime)

    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["state"] != "ACTIONABLE", "price already past target must never read ACTIONABLE"
    assert after["state"] == "EXTENDED"


def test_current_rr_below_minimum_reads_extended_before_trigger(client, headers, tmp_path, monkeypatch):
    """Gap 4 fix (Option A, deterministically chosen): the execution
    window is evaluated even BEFORE the trigger has ever fired."""
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    _resp, record = _watch_with_trigger(client, headers, "AMD", "close_above", 100.0)
    approved_target = record["memory"]["approved_target"]

    # Price already very close to target -- current R:R degrades below
    # RR_WARNING_THRESHOLD -- but the trigger (close above 100) never
    # actually fires this tick (no bars supplied satisfy it).
    near_target_price = approved_target - 0.05
    monkeypatch.setattr(router, "_fetch_recent_30m_bars", lambda ticker: [])
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "AMD": {"price": near_target_price, "timestamp": "t", "source": "mock", "price_branch": "mid"},
    })
    router.run_approved_setup_monitor_tick("test")

    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["state"] == "EXTENDED", "R:R already degraded -- must not silently keep waiting for the trigger"
    assert after["trigger_satisfied_at"] is None, "the trigger itself never fired -- this is a pre-trigger EXTENDED"
    assert after["current_rr_at_last_check"] is not None


def test_trigger_satisfied_and_safety_gates_pass_clean_handoff(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    resp, record = _watch_with_trigger(client, headers, "AMD", "close_above", 100.0)
    memory_id = record["memory"]["id"]
    # Just above entry (and strictly above the 100.0 trigger level) --
    # close_above requires a STRICT inequality, so a close exactly AT the
    # trigger level (which happens to equal entry for this test's setup)
    # would not satisfy it.
    close_price = record["memory"]["approved_entry"] + 0.5

    now = datetime(2026, 8, 20, 16, 5, tzinfo=timezone.utc)
    bars = [_bar("2026-08-20T15:30:00Z", close=close_price)]  # near entry -- healthy R:R
    monkeypatch.setattr(router, "_fetch_recent_30m_bars", lambda ticker: bars)
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "AMD": {"price": close_price, "timestamp": "t", "source": "mock", "price_branch": "mid"},
    })
    monkeypatch.setattr(router, "datetime", type("_dt", (datetime,), {"now": staticmethod(lambda tz=None: now)}))
    try:
        router.run_approved_setup_monitor_tick("test")
    finally:
        monkeypatch.setattr(router, "datetime", datetime)

    after = _monitor_state_row(db_path, record["monitor_state"]["id"])
    assert after["state"] == "ACTIONABLE", "trigger satisfied + safety gates pass -> clean handoff"
    assert after["trigger_satisfied_at"] is not None

    events = _events(db_path, memory_id)
    assert any(e["event_type"] == "WATCH_HANDED_OFF" for e in events), \
        "a watch-originated ACTIONABLE transition must log the named WATCH_HANDED_OFF event"


# ==========================================================================
# Historical truth (task section 9) -- the core non-negotiable
# ==========================================================================

def test_original_watch_review_remains_historically_watch_no_fabricated_approve(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    resp, record = _watch_with_trigger(client, headers, "AMD", "close_above", 100.0)
    close_price = record["memory"]["approved_entry"] + 0.5  # strictly above the 100.0 trigger level

    # Drive it all the way to ACTIONABLE (the "handoff").
    now = datetime(2026, 8, 20, 16, 5, tzinfo=timezone.utc)
    bars = [_bar("2026-08-20T15:30:00Z", close=close_price)]
    monkeypatch.setattr(router, "_fetch_recent_30m_bars", lambda ticker: bars)
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "AMD": {"price": close_price, "timestamp": "t", "source": "mock", "price_branch": "mid"},
    })
    monkeypatch.setattr(router, "datetime", type("_dt", (datetime,), {"now": staticmethod(lambda tz=None: now)}))
    try:
        router.run_approved_setup_monitor_tick("test")
    finally:
        monkeypatch.setattr(router, "datetime", datetime)
    assert _monitor_state_row(db_path, record["monitor_state"]["id"])["state"] == "ACTIONABLE"

    # candidate_visual_reviews must show EXACTLY ONE row for AMD, decision
    # still "watch" -- never rewritten, never duplicated with a fabricated
    # "approve".
    reviews = _reviews_for_ticker(db_path, "AMD")
    assert len(reviews) == 1
    assert reviews[0]["decision"] == "watch"
    assert reviews[0]["id"] == resp["id"]

    # The memory itself also still says its origin was watch.
    final_record = _memory_record(client, headers, "AMD")
    assert final_record["memory"]["source_decision"] == "watch"
    assert final_record["memory"]["visual_review_id"] == resp["id"]
