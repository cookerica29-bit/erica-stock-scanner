"""Approved Setup Memory (2026-09 session, memory-foundation build) --
candidates_router.py's approved_setup_memories / approved_setup_monitor_state
tables and their read/write paths.

Memory infrastructure only: no ENTER_NOW, no ACTIONABLE, no lower-timeframe
confirmation, no alerts. Every test here exercises the real
candidates_router.py code (via a real FastAPI TestClient, same fixture
pattern as tests/review_queue_v1.py) -- not a reimplementation.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import candidates_router as router  # noqa: E402


def _daily_frame(shift: float = 0.0):
    index = pd.date_range("2026-01-01", periods=30, freq="D", tz="UTC")
    rows = []
    for i in range(30):
        close = 100.0 - i * 0.1 + shift
        rows.append({"Open": close + 0.2, "High": close + 0.5, "Low": close - 0.5, "Close": close, "Volume": 1_000_000})
    rows[5]["Low"] = 99.0 + shift
    rows[10]["Low"] = 90.0 + shift
    rows[10]["Close"] = 91.0 + shift
    rows[10]["Open"] = 92.0 + shift
    rows[20]["High"] = 110.0 + shift
    rows[20]["Close"] = 109.0 + shift
    rows[20]["Open"] = 108.0 + shift
    return pd.DataFrame(rows, index=index)


# A genuinely different daily structure (different swing low/high days and
# levels) -- used to force a materially different setup_key for the same
# ticker, same convention as tests/review_queue_persistence_v1.py's
# scenario 7 and the earlier Approved/Watch Setups production verification.
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
    frame_holder = {"frame": _daily_frame()}
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {str(tickers[0]).upper(): frame_holder["frame"]})
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
    return frame_holder


@pytest.fixture(autouse=True)
def _forbid_option_hydration(monkeypatch):
    """Fails loudly if approved-setup-memory read/write/backfill ever
    triggers options hydration -- item L."""
    calls = []

    def _spy(ticker, direction, entry, **kwargs):
        calls.append((ticker, direction, entry, kwargs))
        raise AssertionError(f"_safe_option_contract_for_candidate called unexpectedly: {ticker}")

    monkeypatch.setattr(router, "_safe_option_contract_for_candidate", _spy)
    return calls


def _seed(client, headers, ticker="AMD", entry_price=100.0):
    payload = {
        "source": "ma_pipeline",
        "scanned_at": "2026-08-20T14:30:00Z",
        "candidates": [{
            "ticker": ticker, "signal": "long", "entry_price": entry_price, "ema21_4h": 99.0,
            "daily_regime": "bullish", "confidence": "high",
            "sma50_daily": 106.0, "sma200_daily": 104.0,
        }],
    }
    created = client.post("/api/v1/scanner/candidates", headers=headers, json=payload)
    assert created.status_code == 200


def _review(client, headers, ticker, decision, source="ma_pipeline", **overrides):
    # confirmation_rule/confirmation_level default alongside
    # lower_tf_confirmation="yes" -- required to approve since the
    # Execution Layer V1 session (a "yes" approve now needs an objective
    # confirmation anchor, not just the label). A caller overriding
    # lower_tf_confirmation to "not_yet" must also override these to None
    # (or use trigger_rule/trigger_level instead) -- mutual exclusivity is
    # enforced server-side.
    body = {
        "source": source,
        "market_structure": "bullish",
        "location_read": "good",
        "clear_path_to_target": "yes",
        "lower_tf_confirmation": "yes",
        "confirmation_rule": "close_above",
        "confirmation_level": 100.0,
        "decision": decision,
        "note": "test note",
        **overrides,
    }
    resp = client.post(f"/api/v1/scanner/candidates/{ticker}/visual-review", headers=headers, json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _memories(client, headers, include_inactive=False, ticker=None):
    params = {"include_inactive": include_inactive}
    if ticker:
        params["ticker"] = ticker
    resp = client.get("/api/v1/scanner/candidates/approved-setup-memory", headers=headers, params=params)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _setup_key_for(client, headers, ticker):
    rq = client.get("/api/v1/scanner/candidates/review-queue", headers=headers).json()
    entry = next(c for c in rq["candidates"] if c["ticker"] == ticker)
    return entry["setup_key"], entry


# --- A. approve creates immutable memory ---
def test_approve_creates_memory(client, headers):
    _seed(client, headers, "AMD")
    setup_key, entry = _setup_key_for(client, headers, "AMD")
    review = _review(client, headers, "AMD", "approve")

    records = _memories(client, headers)
    assert len(records) == 1
    memory = records[0]["memory"]
    assert memory["setup_key"] == setup_key
    assert memory["ticker"] == "AMD"
    assert memory["approved_entry"] == entry["entry_price"]
    assert memory["approved_stop"] == entry["stop"]
    assert memory["approved_target"] == entry["target"]
    assert memory["visual_review_id"] == review["id"]
    assert memory["snapshot_origin"] == "approval_event"
    assert memory["snapshot_exact"] is True
    # CONFIRMED, not the old hardcoded APPROVED -- _review()'s default body
    # carries a confirmation_rule/confirmation_level pair (Type A: the
    # reviewer already observed the confirming candle), and the Execution
    # Layer V1 initial-state rule reflects that immediately at creation.
    assert records[0]["monitor_state"]["state"] == "CONFIRMED"


# --- B. approved_entry does not change when live candidate entry changes ---
def test_approved_entry_immutable_across_rescans(client, headers):
    _seed(client, headers, "AMD", entry_price=100.0)
    _review(client, headers, "AMD", "approve")
    records = _memories(client, headers)
    frozen_entry = records[0]["memory"]["approved_entry"]
    assert frozen_entry == 100.0

    # Rescan with a DIFFERENT entry_price but the SAME daily structure --
    # same stop/target, same setup_key (mirrors ma_pipeline recomputing
    # entry_price fresh from the latest 4H close every scan).
    _seed(client, headers, "AMD", entry_price=101.5)
    setup_key_after, live_entry_after = _setup_key_for(client, headers, "AMD")
    assert live_entry_after["entry_price"] == 101.5, "sanity: the live candidate really did change"

    records_after = _memories(client, headers)
    assert len(records_after) == 1, "still exactly one memory -- no duplicate created by the rescan"
    assert records_after[0]["memory"]["setup_key"] == setup_key_after, "setup_key unchanged (same stop/target)"
    assert records_after[0]["memory"]["approved_entry"] == frozen_entry == 100.0, \
        "approved_entry must NOT drift just because the live candidate's entry changed"


# --- C. stop/target/setup_key identity remains preserved ---
def test_stop_target_setup_key_preserved(client, headers):
    _seed(client, headers, "AMD")
    setup_key, entry = _setup_key_for(client, headers, "AMD")
    _review(client, headers, "AMD", "approve")
    memory = _memories(client, headers)[0]["memory"]
    assert memory["setup_key"] == setup_key
    assert memory["approved_stop"] == entry["stop"]
    assert memory["approved_target"] == entry["target"]
    assert memory["approved_risk_reward"] == entry["risk_reward"]


# --- D. new setup_key does not inherit old memory ---
def test_new_setup_key_does_not_inherit_old_memory(client, headers, _mock_network):
    _seed(client, headers, "AMD")
    old_setup_key, _ = _setup_key_for(client, headers, "AMD")
    _review(client, headers, "AMD", "approve")
    assert len(_memories(client, headers, include_inactive=True, ticker="AMD")) == 1

    # Force a materially different daily structure -> different stop/target -> different setup_key.
    _mock_network["frame"] = _daily_frame_v2()
    _seed(client, headers, "AMD")  # re-ingest to bump candidates.updated_at (cache-miss forces recompute)
    new_setup_key, new_entry = _setup_key_for(client, headers, "AMD")
    assert new_setup_key != old_setup_key, "sanity: the structure change really did produce a new setup_key"
    assert new_entry["current_review"] is None, "the new setup generation must read as unreviewed"

    active = _memories(client, headers, ticker="AMD")
    assert len(active) == 1
    assert active[0]["memory"]["setup_key"] == old_setup_key, \
        "the old memory is still the only ACTIVE one -- the new setup_key has no memory of its own yet"


# --- E. approve -> watch preserves historical approval, deactivates monitoring ---
def test_approve_then_watch_withdraws_but_preserves(client, headers):
    _seed(client, headers, "AMD")
    setup_key, _ = _setup_key_for(client, headers, "AMD")
    _review(client, headers, "AMD", "approve")
    assert len(_memories(client, headers)) == 1

    _review(client, headers, "AMD", "watch")
    active = _memories(client, headers)
    assert active == [], "watch must deactivate the memory from the default (active-only) listing"

    historical = _memories(client, headers, include_inactive=True)
    assert len(historical) == 1, "the memory row itself must still exist -- never deleted"
    assert historical[0]["memory"]["setup_key"] == setup_key
    assert historical[0]["monitor_state"]["state"] == "WITHDRAWN"
    assert historical[0]["monitor_state"]["terminal_at"] is not None


# --- F. approve -> reject: same behavior ---
def test_approve_then_reject_withdraws_but_preserves(client, headers):
    _seed(client, headers, "AMD")
    _review(client, headers, "AMD", "approve")
    _review(client, headers, "AMD", "reject")

    assert _memories(client, headers) == []
    historical = _memories(client, headers, include_inactive=True)
    assert len(historical) == 1
    assert historical[0]["monitor_state"]["state"] == "WITHDRAWN"


# --- G. watch -> approve creates active memory ---
def test_watch_then_approve_creates_active_memory(client, headers):
    _seed(client, headers, "AMD")
    _review(client, headers, "AMD", "watch")
    assert _memories(client, headers, include_inactive=True) == [], "watch alone creates no memory at all"

    _review(client, headers, "AMD", "approve")
    active = _memories(client, headers)
    assert len(active) == 1
    assert active[0]["monitor_state"]["state"] == "CONFIRMED"


# --- H. re-approving the SAME setup_key: idempotent when evidence is
# unchanged, a new revision when it genuinely differs (Approved Setup
# Memory Revision, execution_layer_v1_implementation_plan.md section 4).
# Full CLH-failure-mode regression coverage lives in
# approved_setup_memory_revision_v1.py -- these two cover the base case
# this file already owned before that feature existed.
def test_identical_repeat_approve_is_idempotent(client, headers):
    _seed(client, headers, "AMD")
    _review(client, headers, "AMD", "approve", note="same note")
    first = _memories(client, headers, include_inactive=True)
    assert len(first) == 1

    # Byte-identical resubmission -- same setup_key, same everything.
    _review(client, headers, "AMD", "approve", note="same note")
    second = _memories(client, headers, include_inactive=True)
    assert len(second) == 1, "an identical resubmission must not create a second memory row"
    assert second[0]["memory"]["review_note"] == "same note"


def test_note_change_on_reapprove_creates_a_revision(client, headers):
    _seed(client, headers, "AMD")
    first = _review(client, headers, "AMD", "approve", note="first note")
    original = _memories(client, headers, include_inactive=True)
    assert len(original) == 1
    original_memory_id = original[0]["memory"]["id"]

    # Re-affirm the SAME setup_key with a genuinely different note --
    # per the Execution Layer V1 decision, review_note is real evidence,
    # not cosmetic, so this must now create a new memory revision (NOT the
    # old silent no-op).
    _review(client, headers, "AMD", "approve", note="edited note")
    all_rows = _memories(client, headers, include_inactive=True)
    assert len(all_rows) == 2, "a materially different review_note must create a new memory revision"

    active_rows = _memories(client, headers)
    assert len(active_rows) == 1, "exactly one memory stays active/authoritative"
    new_memory = active_rows[0]["memory"]
    assert new_memory["review_note"] == "edited note"
    assert new_memory["revision_of_memory_id"] == original_memory_id
    assert new_memory["revision_reason"] == "newer_approved_review_evidence"

    old_row = next(r for r in all_rows if r["memory"]["id"] == original_memory_id)
    assert old_row["memory"]["review_note"] == "first note", \
        "the ORIGINAL memory snapshot must not be overwritten -- immutability"
    assert old_row["monitor_state"]["state"] == "SUPERSEDED"
    assert old_row["monitor_state"]["superseded_by_memory_id"] == new_memory["id"]


# --- I. reload/database persistence ---
def test_persists_across_fresh_connections(client, headers, monkeypatch, tmp_path):
    _seed(client, headers, "AMD")
    _review(client, headers, "AMD", "approve")
    assert len(_memories(client, headers)) == 1

    # Every request already opens/closes its own sqlite3 connection to the
    # same on-disk db_path (see candidates_router._get_db) -- a second,
    # independent app/TestClient pointed at the SAME KAIROS_CANDIDATES_DB
    # is a clean "reload" simulation, not relying on any in-process cache.
    app2 = FastAPI()
    app2.include_router(router.router)
    client2 = TestClient(app2)
    reloaded = _memories(client2, headers)
    assert len(reloaded) == 1
    assert reloaded[0]["memory"]["ticker"] == "AMD"


# Watch Lifecycle V1: watch-with-a-complete-trigger is a brand new
# memory-creating code path (decision="watch" never created a memory at
# all before this work) -- this proves it persists across a fresh
# connection/process exactly like the approve path does, on-disk, with
# source_decision and the frozen trigger contract intact. Same "second
# independent app/TestClient against the same on-disk db_path" simulation
# as test_persists_across_fresh_connections above -- no in-process cache
# involved either way.
def test_watch_memory_persists_across_fresh_connections(client, headers):
    _seed(client, headers, "AMD")
    _review(
        client, headers, "AMD", "watch",
        lower_tf_confirmation="not_yet",
        confirmation_rule=None, confirmation_level=None,
        trigger_timeframe="30m", trigger_rule="close_above", trigger_level=100.0,
        trigger_reason="waiting for reclaim",
    )
    assert len(_memories(client, headers)) == 1

    app2 = FastAPI()
    app2.include_router(router.router)
    client2 = TestClient(app2)
    reloaded = _memories(client2, headers)
    assert len(reloaded) == 1
    assert reloaded[0]["memory"]["ticker"] == "AMD"
    assert reloaded[0]["memory"]["source_decision"] == "watch"
    assert reloaded[0]["memory"]["trigger_rule"] == "close_above"
    assert reloaded[0]["memory"]["trigger_level"] == 100.0
    assert reloaded[0]["monitor_state"]["state"] == "WAITING_FOR_TRIGGER"


# --- J. historical backfill is explicitly marked reconstructed ---
# --- K. new approvals are marked exact ---
def test_backfill_marks_reconstructed_vs_exact(client, headers):
    # Simulate a pre-existing production approval with NO memory row --
    # exactly today's real FFIV/NVDA/CLH situation: a real
    # candidate_visual_reviews row exists (decision=approve), but this
    # feature didn't exist when it was written.
    _seed(client, headers, "NVDA")
    setup_key, _ = _setup_key_for(client, headers, "NVDA")
    _review(client, headers, "NVDA", "approve")
    # Delete the memory this review just created, so the backfill has
    # something real to do (isolates "backfill of a pre-existing approval"
    # from "a normal approve just made one already").
    conn = router._get_db()
    conn.execute("DELETE FROM approved_setup_monitor_state")
    conn.execute("DELETE FROM approved_setup_memories")
    conn.commit()
    conn.close()
    assert _memories(client, headers, include_inactive=True) == []

    result = client.post("/api/v1/scanner/candidates/approved-setup-memory/backfill", headers=headers)
    assert result.status_code == 200, result.text
    body = result.json()
    assert any(entry["ticker"] == "NVDA" for entry in body["backfilled"])

    backfilled = _memories(client, headers, ticker="NVDA")[0]["memory"]
    assert backfilled["setup_key"] == setup_key
    assert backfilled["snapshot_origin"] == "live_backfill"
    assert backfilled["snapshot_exact"] is False
    assert backfilled["backfill_note"]
    assert "approved_stop" in backfilled["backfill_note"] or "stop" in backfilled["backfill_note"].lower()

    # K: a NORMAL approve on a different ticker must still be exact.
    _seed(client, headers, "MSFT")
    _review(client, headers, "MSFT", "approve")
    fresh = _memories(client, headers, ticker="MSFT")[0]["memory"]
    assert fresh["snapshot_origin"] == "approval_event"
    assert fresh["snapshot_exact"] is True

    # Backfill is idempotent -- running it again must not duplicate NVDA's memory.
    result2 = client.post("/api/v1/scanner/candidates/approved-setup-memory/backfill", headers=headers)
    assert "NVDA" in result2.json()["skipped_already_active"]
    assert len(_memories(client, headers, ticker="NVDA", include_inactive=True)) == 1


# --- L. no options hydration triggered (covered by the autouse
# _forbid_option_hydration fixture across every test above -- an explicit
# one here for the read/backfill paths specifically) ---
def test_no_option_hydration_on_memory_endpoints(client, headers):
    _seed(client, headers, "AMD")
    _review(client, headers, "AMD", "approve")
    _memories(client, headers, include_inactive=True)
    client.post("/api/v1/scanner/candidates/approved-setup-memory/backfill", headers=headers)
    # No AssertionError raised by the _forbid_option_hydration spy means this passed.


# --- M. existing Review Queue behavior unchanged ---
def test_review_queue_shape_unchanged(client, headers):
    _seed(client, headers, "AMD")
    rq = client.get("/api/v1/scanner/candidates/review-queue", headers=headers).json()
    entry = next(c for c in rq["candidates"] if c["ticker"] == "AMD")
    for field in ("setup_key", "current_review", "entry_price", "stop", "target", "risk_reward"):
        assert field in entry
    assert entry["current_review"] is None

    _review(client, headers, "AMD", "approve")
    rq2 = client.get("/api/v1/scanner/candidates/review-queue", headers=headers).json()
    entry2 = next(c for c in rq2["candidates"] if c["ticker"] == "AMD")
    assert entry2["current_review"]["decision"] == "approve"


# --- O. no writes to candidate_promotions / ENTER_NOW ---
def test_no_candidate_promotions_writes(client, headers):
    _seed(client, headers, "AMD")
    _review(client, headers, "AMD", "approve")
    conn = router._get_db()
    count = conn.execute("SELECT COUNT(*) AS c FROM candidate_promotions").fetchone()["c"]
    conn.close()
    assert count == 0
