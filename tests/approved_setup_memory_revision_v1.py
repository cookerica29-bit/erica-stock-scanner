"""Execution Layer V1 -- Observed Confirmation Anchor fields
(confirmation_timeframe/rule/level/confirmed_candle_time/confirmation_note)
and Approved Setup Memory Revision (execution_layer_v1_implementation_plan.md
sections 1-4).

Two things this file is explicitly NOT about (still capture/persistence
only, same as tests/trigger_capture_v1.py before it): no 30m/4H monitoring,
no ENTER_NOW, no ACTIONABLE/EXTENDED/STALE derivation, no market-data call
to check whether a trigger or confirmation "already happened". Every test
here exercises the real candidates_router.py code via a real FastAPI
TestClient, same fixture pattern as tests/trigger_capture_v1.py -- not a
reimplementation.
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
    calls = []

    def _spy(ticker, direction, entry, **kwargs):
        calls.append((ticker, direction, entry, kwargs))
        raise AssertionError(f"_safe_option_contract_for_candidate called unexpectedly: {ticker}")

    monkeypatch.setattr(router, "_safe_option_contract_for_candidate", _spy)
    return calls


def _seed(client, headers, ticker="AMD"):
    payload = {
        "source": "ma_pipeline",
        "scanned_at": "2026-08-20T14:30:00Z",
        "candidates": [{
            "ticker": ticker, "signal": "long", "entry_price": 100.0, "ema21_4h": 99.0,
            "daily_regime": "bullish", "confidence": "high",
            "sma50_daily": 106.0, "sma200_daily": 104.0,
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
    return client.post(f"/api/v1/scanner/candidates/{ticker}/visual-review", headers=headers, json=body)


def _memories(client, headers, include_inactive=False, ticker=None):
    params = {"include_inactive": include_inactive}
    if ticker:
        params["ticker"] = ticker
    resp = client.get("/api/v1/scanner/candidates/approved-setup-memory", headers=headers, params=params)
    assert resp.status_code == 200, resp.text
    return resp.json()


# ==========================================================================
# Section 2: confirmation-anchor validation rules
# ==========================================================================

def test_confirmation_level_without_rule_rejected(client, headers):
    _seed(client, headers)
    resp = _review(client, headers, "AMD", "watch", lower_tf_confirmation="yes", confirmation_level=100.0)
    assert resp.status_code == 422
    assert "confirmation_rule and confirmation_level must both be set" in resp.text


def test_confirmation_rule_without_level_rejected(client, headers):
    _seed(client, headers)
    resp = _review(client, headers, "AMD", "watch", lower_tf_confirmation="yes", confirmation_rule="close_above")
    assert resp.status_code == 422
    assert "confirmation_rule and confirmation_level must both be set" in resp.text


def test_confirmation_timeframe_without_rule_and_level_rejected(client, headers):
    _seed(client, headers)
    resp = _review(client, headers, "AMD", "watch", lower_tf_confirmation="yes", confirmation_timeframe="30m")
    assert resp.status_code == 422
    assert "confirmation_timeframe requires confirmation_rule and confirmation_level" in resp.text


def test_invalid_confirmation_level_rejected(client, headers):
    _seed(client, headers)
    resp = _review(
        client, headers, "AMD", "watch", lower_tf_confirmation="yes",
        confirmation_rule="close_above", confirmation_level=-5.0,
    )
    assert resp.status_code == 422
    assert "confirmation_level must be a positive, finite number" in resp.text


def test_confirmed_candle_time_optional_when_rule_and_level_given(client, headers):
    _seed(client, headers)
    resp = _review(
        client, headers, "AMD", "watch", lower_tf_confirmation="yes",
        confirmation_rule="close_above", confirmation_level=100.5,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["confirmed_candle_time"] is None


def test_confirmed_candle_time_malformed_rejected(client, headers):
    _seed(client, headers)
    resp = _review(
        client, headers, "AMD", "watch", lower_tf_confirmation="yes",
        confirmation_rule="close_above", confirmation_level=100.5,
        confirmed_candle_time="not-a-real-timestamp",
    )
    assert resp.status_code == 422
    assert "confirmed_candle_time must be a parseable timestamp" in resp.text


def test_confirmed_candle_time_valid_iso8601_accepted(client, headers):
    _seed(client, headers)
    resp = _review(
        client, headers, "AMD", "watch", lower_tf_confirmation="yes",
        confirmation_rule="close_above", confirmation_level=100.5,
        confirmed_candle_time="2026-08-31T14:30:00Z",
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["confirmed_candle_time"] == "2026-08-31T14:30:00Z"


def test_confirmation_fields_forbidden_with_not_yet(client, headers):
    _seed(client, headers)
    resp = _review(
        client, headers, "AMD", "watch", lower_tf_confirmation="not_yet",
        confirmation_rule="close_above", confirmation_level=100.5,
    )
    assert resp.status_code == 422
    assert 'require lower_tf_confirmation="yes"' in resp.json()["detail"]


def test_trigger_fields_forbidden_with_yes(client, headers):
    _seed(client, headers)
    resp = _review(
        client, headers, "AMD", "watch", lower_tf_confirmation="yes",
        trigger_rule="close_above", trigger_level=100.5,
    )
    assert resp.status_code == 422
    assert 'require lower_tf_confirmation="not_yet"' in resp.json()["detail"]


def test_confirmation_fields_forbidden_with_practical_rejection(client, headers):
    _seed(client, headers)
    # All four visual fields (including lower_tf_confirmation) must stay
    # None for a practical rejection to pass the PRE-EXISTING visual-field
    # check -- confirmation_rule/level are independently complete
    # (rule+level both given), which is what makes this combination reach
    # the practical-rejection-specific check this test is actually about.
    resp = _review(
        client, headers, "AMD", "reject",
        market_structure=None, location_read=None, clear_path_to_target=None, lower_tf_confirmation=None,
        practical_rejection_reason="options_too_expensive",
        confirmation_rule="close_above", confirmation_level=100.5,
    )
    assert resp.status_code == 422
    assert "cannot be combined with practical_rejection_reason" in resp.json()["detail"]


def test_approve_with_yes_and_no_confirmation_anchor_rejected(client, headers):
    """The user's actual decision: a NEW approve with lower_tf_confirmation
    ="yes" must supply an objective confirmation anchor going forward."""
    _seed(client, headers)
    resp = _review(client, headers, "AMD", "approve", lower_tf_confirmation="yes")
    assert resp.status_code == 422
    assert "requires confirmation_rule and confirmation_level" in resp.text


def test_watch_with_yes_and_no_confirmation_anchor_still_allowed(client, headers):
    """Deliberately NOT required outside decision=approve -- a watch never
    creates a memory, so there is no Execution Layer consequence."""
    _seed(client, headers)
    resp = _review(client, headers, "AMD", "watch", lower_tf_confirmation="yes")
    assert resp.status_code == 200, resp.text


def test_approve_with_yes_and_confirmation_anchor_succeeds(client, headers):
    _seed(client, headers)
    resp = _review(
        client, headers, "AMD", "approve", lower_tf_confirmation="yes",
        confirmation_rule="close_above", confirmation_level=100.50,
        confirmed_candle_time="2026-08-31T14:30:00Z", confirmation_note="saw it confirm on the 30m chart",
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["confirmation_timeframe"] == "30m"
    assert body["confirmation_rule"] == "close_above"
    assert body["confirmation_level"] == 100.50
    assert body["confirmed_candle_time"] == "2026-08-31T14:30:00Z"
    assert body["confirmation_note"] == "saw it confirm on the 30m chart"


# ==========================================================================
# Approve copies the confirmation anchor exactly into the memory; initial
# state machine (CONFIRMED / WAITING_FOR_TRIGGER / APPROVED)
# ==========================================================================

def test_approve_copies_confirmation_anchor_into_memory_and_starts_confirmed(client, headers):
    _seed(client, headers)
    resp = _review(
        client, headers, "AMD", "approve", lower_tf_confirmation="yes",
        confirmation_rule="close_above", confirmation_level=100.50,
        confirmed_candle_time="2026-08-31T14:30:00Z", confirmation_note="observed",
    )
    assert resp.status_code == 200, resp.text
    record = _memories(client, headers, ticker="AMD")[0]
    memory = record["memory"]
    assert memory["confirmation_timeframe"] == "30m"
    assert memory["confirmation_rule"] == "close_above"
    assert memory["confirmation_level"] == 100.50
    assert memory["confirmed_candle_time"] == "2026-08-31T14:30:00Z"
    assert memory["confirmation_note"] == "observed"
    assert record["monitor_state"]["state"] == "CONFIRMED"


def test_approve_with_only_trigger_starts_waiting_for_trigger(client, headers):
    _seed(client, headers)
    resp = _review(
        client, headers, "AMD", "approve", lower_tf_confirmation="not_yet",
        trigger_rule="close_above", trigger_level=100.50,
    )
    assert resp.status_code == 200, resp.text
    record = _memories(client, headers, ticker="AMD")[0]
    assert record["monitor_state"]["state"] == "WAITING_FOR_TRIGGER"


def test_approve_with_neither_trigger_nor_confirmation_starts_approved(client, headers):
    """Legacy/incomplete evidence -- correctly stuck in APPROVED until a
    fresh human review supplies one or the other. Uses lower_tf_confirmation
    ="not_yet" with no trigger, mirroring a review submitted before this
    feature existed."""
    _seed(client, headers)
    resp = _review(client, headers, "AMD", "approve", lower_tf_confirmation="not_yet")
    assert resp.status_code == 200, resp.text
    record = _memories(client, headers, ticker="AMD")[0]
    assert record["monitor_state"]["state"] == "APPROVED"


# ==========================================================================
# Backfill never fabricates a confirmation anchor
# ==========================================================================

def test_backfill_never_populates_confirmation_fields(client, headers):
    _seed(client, headers)
    _review(client, headers, "AMD", "approve", lower_tf_confirmation="not_yet")
    conn = router._get_db()
    conn.execute("DELETE FROM approved_setup_monitor_state")
    conn.execute("DELETE FROM approved_setup_memories")
    conn.commit()
    conn.close()

    result = client.post("/api/v1/scanner/candidates/approved-setup-memory/backfill", headers=headers)
    assert result.status_code == 200, result.text

    memory = _memories(client, headers, ticker="AMD")[0]["memory"]
    assert memory["confirmation_timeframe"] is None
    assert memory["confirmation_rule"] is None
    assert memory["confirmation_level"] is None
    assert memory["confirmed_candle_time"] is None
    assert memory["confirmation_note"] is None
    assert memory["revision_of_memory_id"] is None


# ==========================================================================
# Section 4: Approved Setup Memory Revision -- the CLH failure mode,
# exact scenario (execution_layer_v1_implementation_plan.md section 4.5,
# requirement 10 verbatim).
# ==========================================================================

def test_clh_failure_mode_revision_on_new_confirmation_evidence(client, headers):
    _seed(client, headers, "CLH")

    # 1. Active memory exists for setup_key with lower_tf_confirmation=not_yet.
    approve_not_yet = _review(client, headers, "CLH", "approve", lower_tf_confirmation="not_yet")
    assert approve_not_yet.status_code == 200, approve_not_yet.text
    old_review_id = approve_not_yet.json()["id"]

    original = _memories(client, headers, include_inactive=True, ticker="CLH")
    assert len(original) == 1
    old_memory = original[0]["memory"]
    old_monitor_state_id = original[0]["monitor_state"]["id"]
    assert old_memory["lower_tf_confirmation"] == "not_yet"
    assert original[0]["monitor_state"]["state"] == "APPROVED"

    # 2. User submits new APPROVE for same setup_key with
    #    lower_tf_confirmation=yes and valid confirmation evidence.
    approve_yes = _review(
        client, headers, "CLH", "approve", lower_tf_confirmation="yes",
        confirmation_rule="close_above", confirmation_level=318.25,
        confirmed_candle_time="2026-09-01T14:30:00Z", confirmation_note="reclaimed prior range high",
    )
    assert approve_yes.status_code == 200, approve_yes.text
    new_review_id = approve_yes.json()["id"]
    assert new_review_id != old_review_id

    all_rows = _memories(client, headers, include_inactive=True, ticker="CLH")
    assert len(all_rows) == 2, "a new memory generation must be created, the old one preserved -- not overwritten"

    # -- old memory remains unchanged/historical --
    old_row = next(r for r in all_rows if r["memory"]["id"] == old_memory["id"])
    assert old_row["memory"]["lower_tf_confirmation"] == "not_yet"
    assert old_row["memory"]["visual_review_id"] == old_review_id
    assert old_row["memory"]["confirmation_rule"] is None
    assert old_row["memory"]["confirmation_level"] is None

    # -- new memory is created, points to new review, contains the new
    #    confirmation evidence, becomes authoritative --
    active_rows = _memories(client, headers, ticker="CLH")
    assert len(active_rows) == 1
    new_memory = active_rows[0]["memory"]
    assert new_memory["id"] != old_memory["id"]
    assert new_memory["visual_review_id"] == new_review_id
    assert new_memory["lower_tf_confirmation"] == "yes"
    assert new_memory["confirmation_rule"] == "close_above"
    assert new_memory["confirmation_level"] == 318.25
    assert new_memory["confirmed_candle_time"] == "2026-09-01T14:30:00Z"
    assert new_memory["confirmation_note"] == "reclaimed prior range high"
    assert new_memory["revision_of_memory_id"] == old_memory["id"]
    assert new_memory["revision_reason"] == "newer_approved_review_evidence"

    # -- monitor state attaches to new memory; old one is superseded --
    new_monitor_state = active_rows[0]["monitor_state"]
    assert new_monitor_state["approved_memory_id"] == new_memory["id"]
    assert new_monitor_state["state"] == "CONFIRMED"
    # fresh -- nothing carried forward from the old memory's monitor_state
    assert new_monitor_state["trigger_satisfied_at"] is None
    assert new_monitor_state["last_checked_at"] is None
    assert new_monitor_state["last_live_price"] is None

    old_state_after = old_row["monitor_state"]
    assert old_state_after["id"] == old_monitor_state_id
    assert old_state_after["state"] == "SUPERSEDED"
    assert old_state_after["superseded_by_memory_id"] == new_memory["id"]
    assert old_state_after["terminal_at"] is not None

    # -- identical repeated APPROVE remains idempotent --
    approve_yes_again = _review(
        client, headers, "CLH", "approve", lower_tf_confirmation="yes",
        confirmation_rule="close_above", confirmation_level=318.25,
        confirmed_candle_time="2026-09-01T14:30:00Z", confirmation_note="reclaimed prior range high",
    )
    assert approve_yes_again.status_code == 200, approve_yes_again.text
    all_rows_after_repeat = _memories(client, headers, include_inactive=True, ticker="CLH")
    assert len(all_rows_after_repeat) == 2, "a byte-identical resubmission must not create a third memory"
    still_active = _memories(client, headers, ticker="CLH")
    assert len(still_active) == 1
    assert still_active[0]["memory"]["id"] == new_memory["id"]


def test_revision_fires_on_non_trigger_evidence_change_alone(client, headers):
    """market_structure/location_read/clear_path_to_target/review_note
    changing, with trigger/confirmation fields unchanged, must still fire
    a revision -- confirms the comparison isn't accidentally scoped to
    only trigger/confirmation fields."""
    _seed(client, headers)
    _review(client, headers, "AMD", "approve", lower_tf_confirmation="not_yet", market_structure="bullish")
    first = _memories(client, headers, include_inactive=True)
    assert len(first) == 1

    _review(client, headers, "AMD", "approve", lower_tf_confirmation="not_yet", market_structure="range")
    second = _memories(client, headers, include_inactive=True)
    assert len(second) == 2, "a materially different market_structure alone must create a revision"


def test_genuinely_new_setup_key_still_takes_structural_supersession_path(client, headers, monkeypatch):
    """A materially different setup_key (stop/target actually changed)
    must still take the pre-existing structural-supersession path --
    revision_of_memory_id stays NULL on the new memory (requirement 9: no
    regression to existing behavior)."""
    _seed(client, headers, "AMD")
    _review(client, headers, "AMD", "approve", lower_tf_confirmation="not_yet")
    first = _memories(client, headers, ticker="AMD")
    assert len(first) == 1
    old_setup_key = first[0]["memory"]["setup_key"]

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
    _review(client, headers, "AMD", "approve", lower_tf_confirmation="not_yet")

    active = _memories(client, headers, ticker="AMD")
    assert len(active) == 1
    new_memory = active[0]["memory"]
    assert new_memory["setup_key"] != old_setup_key
    assert new_memory["revision_of_memory_id"] is None, \
        "a genuinely new setup_key generation must NOT be recorded as an evidence revision"


def test_reviewed_at_only_difference_does_not_create_a_revision(client, headers):
    _seed(client, headers)
    _review(client, headers, "AMD", "approve", lower_tf_confirmation="not_yet", note="same content")
    first = _memories(client, headers, include_inactive=True)
    assert len(first) == 1

    # Re-submitted moments later, identical content -- only reviewed_at differs.
    _review(client, headers, "AMD", "approve", lower_tf_confirmation="not_yet", note="same content")
    second = _memories(client, headers, include_inactive=True)
    assert len(second) == 1, "reviewed_at is never part of the evidence comparison"
