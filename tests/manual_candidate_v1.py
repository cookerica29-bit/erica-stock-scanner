"""Manual Candidate Submission (2026-09 session) --
POST /candidates/manual and POST /candidates/{ticker}/manual-trigger in
candidates_router.py.

Same real-FastAPI-TestClient-for-setup + direct run_approved_setup_monitor_tick()
call convention as tests/approved_setup_monitor_v1.py (this file's fixtures
are deliberately duplicated from there, not imported -- see review_queue.js's
own comment on this codebase's "one self-contained file per feature" test
convention; there is no conftest.py in this repo).

Covers exactly the coverage list from the manual-submission task:
  1. manual submission creates a valid row
  2. trigger check fires correctly in both directions (long/short)
  3. trigger check doesn't fire falsely against stop/target logic -- all
     three checks (stop/target/trigger) are mutually exclusive per tick,
     same ordering discipline as the TARGET_HIT fix.
Plus the validation/duplicate/revision-path coverage a real submission
endpoint needs that a pure monitor-logic test file wouldn't otherwise touch.
"""

import sys
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
    monkeypatch.setattr(router, "_recent_4h_bars_for_execution_shadow", lambda ticker: [])


@pytest.fixture(autouse=True)
def _forbid_option_hydration(monkeypatch):
    def _spy(ticker, direction, entry, **kwargs):
        raise AssertionError(f"_safe_option_contract_for_candidate called unexpectedly: {ticker}")
    monkeypatch.setattr(router, "_safe_option_contract_for_candidate", _spy)


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


def _submit(client, headers, **overrides):
    body = {
        "ticker": "AMD", "direction": "long", "entry": 100.0, "stop": 95.0, "target": 115.0,
        "rationale": "Test rationale: clean daily structure, retesting a prior breakout zone.",
        **overrides,
    }
    return client.post("/api/v1/scanner/candidates/manual", headers=headers, json=body)


# ==========================================================================
# 1. Manual submission creates a valid row
# ==========================================================================

def test_manual_submission_creates_memory_and_monitor_state(client, headers):
    resp = _submit(client, headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["memory"]["ticker"] == "AMD"
    assert body["memory"]["source"] == "manual"
    assert body["memory"]["direction"] == "long"
    assert body["memory"]["approved_entry"] == 100.0
    assert body["memory"]["approved_stop"] == 95.0
    assert body["memory"]["approved_target"] == 115.0
    assert body["memory"]["review_note"] == "Test rationale: clean daily structure, retesting a prior breakout zone."
    assert body["memory"]["visual_review_id"] is None
    assert body["memory"]["market_structure"] is None, "no formal chart-review category was ever recorded for a manual entry"
    # No trigger given -> plain APPROVED, per _create_approved_setup_memory's
    # own existing initial-state logic (nothing new here).
    assert body["monitor_state"]["state"] == "APPROVED"


def test_manual_submission_with_trigger_starts_waiting_for_trigger(client, headers):
    resp = _submit(client, headers, trigger_rule="close_above", trigger_level=102.0)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["memory"]["trigger_rule"] == "close_above"
    assert body["memory"]["trigger_level"] == 102.0
    assert body["memory"]["trigger_timeframe"] == "30m"
    assert body["monitor_state"]["state"] == "WAITING_FOR_TRIGGER"


def test_manual_short_setup_accepted(client, headers):
    resp = _submit(client, headers, ticker="TLT", direction="short", entry=90.0, stop=95.0, target=80.0)
    assert resp.status_code == 200, resp.text
    assert resp.json()["memory"]["direction"] == "short"


# ==========================================================================
# Validation
# ==========================================================================

def test_long_with_stop_above_entry_is_rejected(client, headers):
    resp = _submit(client, headers, direction="long", entry=100.0, stop=101.0, target=115.0)
    assert resp.status_code == 422
    assert "stop" in resp.json()["detail"].lower()


def test_long_with_target_below_entry_is_rejected(client, headers):
    resp = _submit(client, headers, direction="long", entry=100.0, stop=95.0, target=99.0)
    assert resp.status_code == 422


def test_short_with_stop_below_entry_is_rejected(client, headers):
    resp = _submit(client, headers, direction="short", entry=90.0, stop=85.0, target=80.0)
    assert resp.status_code == 422


def test_short_with_target_above_entry_is_rejected(client, headers):
    resp = _submit(client, headers, direction="short", entry=90.0, stop=95.0, target=91.0)
    assert resp.status_code == 422


def test_partial_trigger_is_rejected(client, headers):
    resp = _submit(client, headers, trigger_rule="close_above")  # trigger_level omitted
    assert resp.status_code == 422
    assert "trigger_rule and trigger_level" in resp.json()["detail"]


def test_negative_or_zero_prices_rejected(client, headers):
    assert _submit(client, headers, entry=0).status_code == 422
    assert _submit(client, headers, stop=-5).status_code == 422


def test_duplicate_active_setup_key_rejected(client, headers):
    first = _submit(client, headers)
    assert first.status_code == 200
    second = _submit(client, headers)  # byte-identical ticker/direction/stop/target
    assert second.status_code == 409
    assert "already exists" in second.json()["detail"]


# ==========================================================================
# 2/3. Trigger check -- reuses the EXISTING WAITING_FOR_TRIGGER ->
# TRIGGER_SATISFIED monitor logic (candidates_router.py's already-shipped
# _first_satisfying_completed_rth_bar path) unchanged; these tests prove
# that existing mechanism already covers manually-submitted rows for free,
# in both directions, with no new intrabar check needed.
# ==========================================================================

def test_manual_long_trigger_satisfied_by_a_completed_bar(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    resp = _submit(client, headers, ticker="AMD", direction="long", entry=100.0, stop=95.0, target=115.0,
                    trigger_rule="close_above", trigger_level=102.0)
    memory_id = resp.json()["memory"]["id"]
    monitor_state_id = resp.json()["monitor_state"]["id"]

    import datetime as dt
    now = dt.datetime(2026, 8, 20, 15, 0, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(router, "_recent_4h_bars_for_execution_shadow", lambda ticker: [])
    monkeypatch.setattr(router, "_completed_rth_30m_bars", lambda bars, now: [
        {"time": "2026-08-20T14:30:00Z", "open": 101.5, "high": 102.5, "low": 101.0, "close": 102.3},
    ])
    monkeypatch.setattr(router, "datetime", type("_dt", (dt.datetime,), {"now": staticmethod(lambda tz=None: now)}))
    try:
        router.run_approved_setup_monitor_tick("test")
    finally:
        monkeypatch.setattr(router, "datetime", dt.datetime)

    after = _monitor_state_row(db_path, monitor_state_id)
    # TRIGGER_SATISFIED is transient by design (see dashboard_state.py's
    # own comment: the SAME tick immediately re-evaluates it to
    # ACTIONABLE/EXTENDED/STALE) -- trigger_satisfied_at being set is the
    # real, persisted evidence the trigger fired, not the resting state.
    assert after["trigger_satisfied_at"] is not None, "trigger_satisfied_at must be set once the completed bar satisfies the trigger"
    assert after["state"] in ("ACTIONABLE", "EXTENDED"), f"expected the post-trigger execution-window state, got {after['state']}"


def test_manual_short_trigger_satisfied_by_a_completed_bar(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    resp = _submit(client, headers, ticker="TLT", direction="short", entry=90.0, stop=95.0, target=80.0,
                    trigger_rule="close_below", trigger_level=88.0)
    memory_id = resp.json()["memory"]["id"]
    monitor_state_id = resp.json()["monitor_state"]["id"]

    # Override the default $100 mock quote -- above this short's stop (95),
    # which would invalidate it before the trigger check ever runs. 91 is
    # between entry (90) and stop (95): still a live, non-invalidated short.
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "TLT": {"price": 91.0, "timestamp": "2026-08-20T18:30:00Z", "source": "mock_latest_quote", "price_branch": "mid"},
    })
    import datetime as dt
    now = dt.datetime(2026, 8, 20, 15, 0, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(router, "_completed_rth_30m_bars", lambda bars, now: [
        {"time": "2026-08-20T14:30:00Z", "open": 88.5, "high": 88.7, "low": 87.5, "close": 87.8},
    ])
    monkeypatch.setattr(router, "datetime", type("_dt", (dt.datetime,), {"now": staticmethod(lambda tz=None: now)}))
    try:
        router.run_approved_setup_monitor_tick("test")
    finally:
        monkeypatch.setattr(router, "datetime", dt.datetime)

    after = _monitor_state_row(db_path, monitor_state_id)
    assert after["trigger_satisfied_at"] is not None, "trigger_satisfied_at must be set once the completed bar satisfies the trigger"
    assert after["state"] in ("ACTIONABLE", "EXTENDED"), f"expected the post-trigger execution-window state, got {after['state']}"


def test_invalidation_wins_over_trigger_satisfaction_in_the_same_tick(client, headers, tmp_path, monkeypatch):
    """Mutual exclusivity (task point 5): if price has ALSO breached the
    stop this tick, invalidation must win -- the trigger check must not
    even run, same ordering discipline the TARGET_HIT fix already
    established for stop-vs-target."""
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    resp = _submit(client, headers, ticker="AMD", direction="long", entry=100.0, stop=95.0, target=115.0,
                    trigger_rule="close_above", trigger_level=102.0)
    monitor_state_id = resp.json()["monitor_state"]["id"]

    # Price is now below the stop -- invalidation must fire regardless of
    # whether a qualifying trigger bar also exists.
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "AMD": {"price": 94.0, "timestamp": "2026-08-20T18:30:00Z", "source": "mock_latest_quote", "price_branch": "mid"},
    })
    monkeypatch.setattr(router, "_completed_rth_30m_bars", lambda bars, now: [
        {"time": "2026-08-20T14:30:00Z", "open": 101.5, "high": 102.5, "low": 101.0, "close": 102.3},
    ])
    import datetime as dt
    now = dt.datetime(2026, 8, 20, 18, 30, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(router, "datetime", type("_dt", (dt.datetime,), {"now": staticmethod(lambda tz=None: now)}))
    try:
        router.run_approved_setup_monitor_tick("test")
    finally:
        monkeypatch.setattr(router, "datetime", dt.datetime)

    after = _monitor_state_row(db_path, monitor_state_id)
    assert after["state"] == "INVALIDATED", f"invalidation must win over trigger satisfaction, got {after['state']}"


def test_target_hit_works_for_a_manual_setup_with_no_trigger(client, headers, tmp_path, monkeypatch):
    """Confirms TARGET_HIT (shipped earlier this session) applies to
    manually-submitted rows too -- it's an unconditional check on every
    active row, same as invalidation, with no source-specific gating."""
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    resp = _submit(client, headers, ticker="AMD", direction="long", entry=100.0, stop=95.0, target=115.0)
    monitor_state_id = resp.json()["monitor_state"]["id"]

    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        "AMD": {"price": 115.5, "timestamp": "2026-08-20T18:30:00Z", "source": "mock_latest_quote", "price_branch": "mid"},
    })
    router.run_approved_setup_monitor_tick("test")
    after = _monitor_state_row(db_path, monitor_state_id)
    assert after["state"] == "TARGET_HIT"


# ==========================================================================
# set_manual_candidate_trigger -- adding a trigger later
# ==========================================================================

def test_set_trigger_later_creates_a_revision_and_transitions_to_waiting(client, headers, tmp_path, monkeypatch):
    db_path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", db_path)
    submitted = _submit(client, headers, ticker="AMD")
    old_memory_id = submitted.json()["memory"]["id"]
    old_monitor_state_id = submitted.json()["monitor_state"]["id"]
    assert submitted.json()["monitor_state"]["state"] == "APPROVED"

    resp = client.post(
        "/api/v1/scanner/candidates/AMD/manual-trigger", headers=headers,
        json={"source": "manual", "trigger_rule": "close_above", "trigger_level": 103.0, "trigger_reason": "adding it after the fact"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["memory"]["id"] != old_memory_id, "adding a trigger creates a NEW memory generation, not an in-place mutation"
    assert body["memory"]["revision_of_memory_id"] == old_memory_id
    assert body["memory"]["trigger_rule"] == "close_above"
    assert body["memory"]["trigger_level"] == 103.0
    assert body["memory"]["approved_entry"] == 100.0, "entry/stop/target carry forward unchanged from the original submission"
    assert body["monitor_state"]["state"] == "WAITING_FOR_TRIGGER"

    old_state = _monitor_state_row(db_path, old_monitor_state_id)
    assert old_state["state"] == "SUPERSEDED"


def test_set_trigger_on_nonexistent_setup_is_rejected(client, headers):
    resp = client.post(
        "/api/v1/scanner/candidates/ZZZZ/manual-trigger", headers=headers,
        json={"source": "manual", "trigger_rule": "close_above", "trigger_level": 103.0},
    )
    assert resp.status_code == 422


def test_set_trigger_on_a_setup_that_already_has_one_is_rejected(client, headers):
    _submit(client, headers, ticker="AMD", trigger_rule="close_above", trigger_level=102.0)  # already WAITING_FOR_TRIGGER
    resp = client.post(
        "/api/v1/scanner/candidates/AMD/manual-trigger", headers=headers,
        json={"source": "manual", "trigger_rule": "close_above", "trigger_level": 105.0},
    )
    assert resp.status_code == 422
    assert "APPROVED" in resp.json()["detail"]


# ==========================================================================
# 5. Anchored plan: stop and/or target derived from marked structure
#
# The product rule this encodes: the human picks the ticker, the direction
# and the order block; Kairos does the arithmetic. A level she typed is
# always used verbatim -- deriving is strictly opt-in by omission -- and a
# block she marked is never silently swapped for a generic ATR stop.
#
# _daily_frame() has ATR14 == 1.0 exactly, so every expected stop below is
# arithmetic, not an approximation: order-block stop = OB edge -/+ 0.1 ATR
# (DEFAULT_CLAMP_BUFFER_ATR), flat ATR stop = entry -/+ 1.5 ATR.
# ==========================================================================

def test_omitted_stop_is_derived_from_the_marked_order_block(client, headers):
    resp = _submit(client, headers, ticker="NVDA", stop=None, order_block_high=98.0, order_block_low=97.0)
    assert resp.status_code == 200, resp.text
    memory = resp.json()["memory"]
    # 97.0 (OB low, the long's invalidation level) - 0.1 ATR buffer
    assert memory["approved_stop"] == pytest.approx(96.9)
    assert memory["approved_entry"] == 100.0
    assert memory["approved_target"] == 115.0, "a supplied target is still used verbatim"
    assert "order_block" in memory["review_note"], "provenance records which half Kairos computed"
    assert "98.5" in memory["review_note"], "the flat ATR stop is preserved in the note for comparison"
    assert "R:R" not in memory["review_note"], "no R:R warning on a healthy plan (15 / 3.1 = 4.8)"


def test_derived_short_stop_uses_the_order_block_high(client, headers):
    resp = _submit(
        client, headers, ticker="NVDA", direction="short", entry=100.0, stop=None, target=85.0,
        order_block_high=103.0, order_block_low=102.0,
    )
    assert resp.status_code == 200, resp.text
    # 103.0 (OB high) + 0.1 ATR buffer
    assert resp.json()["memory"]["approved_stop"] == pytest.approx(103.1)


def test_order_block_too_close_to_entry_is_refused_not_replaced(client, headers):
    # Stop would be 99.85 -- 0.15 ATR away, under resolve_stop's
    # MIN_STOP_DISTANCE_ATR floor of 0.25. On the scanner path that falls
    # back to the flat ATR stop; for a block she marked it must not.
    resp = _submit(client, headers, ticker="NVDA", stop=None, order_block_high=99.98, order_block_low=99.95)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "0.15x ATR" in detail and "0.25x ATR minimum" in detail
    assert "won't swap in a generic ATR stop" in detail


def test_order_block_on_the_wrong_side_of_entry_is_refused(client, headers):
    # A long whose marked block sits above entry: the "stop" would be 104.9.
    resp = _submit(client, headers, ticker="NVDA", stop=None, order_block_high=106.0, order_block_low=105.0)
    assert resp.status_code == 422
    assert "not below entry" in resp.json()["detail"]


def test_wrong_side_order_block_on_a_short_is_refused(client, headers):
    resp = _submit(
        client, headers, ticker="NVDA", direction="short", entry=100.0, stop=None, target=85.0,
        order_block_high=96.0, order_block_low=95.0,
    )
    assert resp.status_code == 422
    assert "not above entry" in resp.json()["detail"]


def test_order_block_with_an_explicit_stop_is_rejected(client, headers):
    # Accepting it would mean silently ignoring it -- the stored row would
    # never show that a block was sent.
    resp = _submit(client, headers, ticker="NVDA", order_block_high=98.0, order_block_low=97.0)
    assert resp.status_code == 422
    assert "can't be combined with an explicit stop" in resp.json()["detail"]


def test_omitting_stop_without_an_order_block_is_rejected(client, headers):
    resp = _submit(client, headers, stop=None)
    assert resp.status_code == 422
    assert "order_block_high" in resp.json()["detail"]


def test_half_specified_order_block_is_rejected(client, headers):
    resp = _submit(client, headers, stop=None, order_block_high=98.0)
    assert resp.status_code == 422
    assert "both" in resp.json()["detail"]


def test_inverted_order_block_is_rejected(client, headers):
    resp = _submit(client, headers, stop=None, order_block_high=97.0, order_block_low=98.0)
    assert resp.status_code == 422
    assert "must be below" in resp.json()["detail"]


def test_derived_stop_with_supplied_target_warns_on_low_rr(client, headers):
    # A wide structural stop (89.9) against her own close target (102):
    # R:R = 2 / 10.1 = 0.198. Still accepted -- it's her plan -- but the
    # weak R:R is recorded, same as when the target is the derived half.
    resp = _submit(client, headers, ticker="NVDA", stop=None, target=102.0, order_block_high=91.0, order_block_low=90.0)
    assert resp.status_code == 200, resp.text
    memory = resp.json()["memory"]
    assert memory["approved_stop"] == pytest.approx(89.9)
    assert "R:R 0.2 below 1.5" in memory["review_note"]


def test_omitted_target_is_derived_and_validated(client, headers, monkeypatch):
    # _nearest_structural_target itself is covered by structural_resistance_v1;
    # stubbed here so this test is about the WIRING -- that the derived target
    # reaches the stored row and the clamp/R:R math runs against the live stop.
    monkeypatch.setattr(router, "_nearest_structural_target", lambda entry, direction, swings, min_dist: 112.0)
    resp = _submit(client, headers, ticker="NVDA", target=None)
    assert resp.status_code == 200, resp.text
    memory = resp.json()["memory"]
    assert memory["approved_target"] == pytest.approx(112.0)
    assert memory["approved_stop"] == 95.0, "a supplied stop is still used verbatim"
    assert "target derived from structure" in memory["review_note"]


def test_derived_target_with_low_rr_is_warned(client, headers, monkeypatch):
    monkeypatch.setattr(router, "_nearest_structural_target", lambda entry, direction, swings, min_dist: 103.0)
    resp = _submit(client, headers, ticker="NVDA", target=None)
    assert resp.status_code == 200, resp.text
    # 3 / 5 = 0.6
    assert "R:R 0.6 below 1.5" in resp.json()["memory"]["review_note"]


def test_no_structural_target_is_refused_rather_than_invented(client, headers, monkeypatch):
    monkeypatch.setattr(router, "_nearest_structural_target", lambda entry, direction, swings, min_dist: None)
    resp = _submit(client, headers, target=None)
    assert resp.status_code == 422
    assert "won't invent one" in resp.json()["detail"]


def test_both_halves_derived_together(client, headers, monkeypatch):
    monkeypatch.setattr(router, "_nearest_structural_target", lambda entry, direction, swings, min_dist: 112.0)
    resp = _submit(
        client, headers, ticker="NVDA", stop=None, target=None,
        order_block_high=98.0, order_block_low=97.0,
    )
    assert resp.status_code == 200, resp.text
    memory = resp.json()["memory"]
    assert memory["approved_stop"] == pytest.approx(96.9)
    assert memory["approved_target"] == pytest.approx(112.0)
    note = memory["review_note"]
    assert "order_block" in note and "target derived" in note
    assert note.startswith("Test rationale:"), "the submitter's own rationale stays first and unedited"


def test_derived_values_drive_the_setup_key(client, headers):
    # Two submissions on the same ticker whose only difference is the marked
    # order block must not collide on setup_key -- it is built from the
    # resolved stop, not the (absent) submitted one.
    first = _submit(client, headers, ticker="NVDA", stop=None, order_block_high=98.0, order_block_low=97.0)
    second = _submit(client, headers, ticker="NVDA", stop=None, order_block_high=94.0, order_block_low=93.0)
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert first.json()["memory"]["approved_stop"] != second.json()["memory"]["approved_stop"]


def test_fully_specified_submission_fetches_no_market_data(client, headers, monkeypatch):
    # Regression guard on the pre-existing contract: when nothing needs
    # deriving, this endpoint still does no network I/O at all (the
    # "no live quote is fetched here" simplification in its docstring).
    def _boom(tickers, period, interval):
        raise AssertionError("market data was fetched for a fully-specified manual submission")
    monkeypatch.setattr(router, "_batch_download", _boom)
    resp = _submit(client, headers, ticker="NVDA")
    assert resp.status_code == 200, resp.text
    assert resp.json()["memory"]["review_note"] == "Test rationale: clean daily structure, retesting a prior breakout zone.", \
        "no provenance line is appended when nothing was derived"
