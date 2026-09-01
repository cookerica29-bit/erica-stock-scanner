"""Explicit Lower-TF Trigger Capture (2026-09 session) --
candidates_router.py's trigger_timeframe/trigger_rule/trigger_level/
trigger_reason fields on candidate_visual_reviews and
approved_setup_memories.

This is capture/persistence only -- no monitoring, no ENTER_NOW, no
inferred BOS/CHoCH, no market-data validation. Every test here exercises
the real candidates_router.py code via a real FastAPI TestClient, same
fixture pattern as tests/approved_setup_memory_v1.py -- not a
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
    """Fails loudly if trigger-related read/write ever triggers options
    hydration -- item L of the test list."""
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


# --- NOT_YET review with close_above trigger ---
def test_not_yet_with_close_above_trigger(client, headers):
    _seed(client, headers, "AMD")
    resp = _review(
        client, headers, "AMD", "watch",
        lower_tf_confirmation="not_yet",
        trigger_timeframe="30m", trigger_rule="close_above", trigger_level=100.50, trigger_reason="waiting for reclaim",
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["trigger_timeframe"] == "30m"
    assert body["trigger_rule"] == "close_above"
    assert body["trigger_level"] == 100.50
    assert body["trigger_reason"] == "waiting for reclaim"


# --- NOT_YET with close_below ---
def test_not_yet_with_close_below_trigger(client, headers):
    _seed(client, headers, "AMD")
    resp = _review(
        client, headers, "AMD", "watch",
        lower_tf_confirmation="not_yet",
        trigger_timeframe="30m", trigger_rule="close_below", trigger_level=95.25,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["trigger_rule"] == "close_below"
    assert body["trigger_level"] == 95.25


# --- null trigger allowed ---
def test_null_trigger_allowed(client, headers):
    _seed(client, headers, "AMD")
    resp = _review(client, headers, "AMD", "watch", lower_tf_confirmation="not_yet")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["trigger_timeframe"] is None
    assert body["trigger_rule"] is None
    assert body["trigger_level"] is None
    assert body["trigger_reason"] is None


# --- invalid negative/zero trigger rejected ---
def test_negative_trigger_level_rejected(client, headers):
    _seed(client, headers, "AMD")
    resp = _review(client, headers, "AMD", "watch", trigger_rule="close_above", trigger_level=-5.0)
    assert resp.status_code == 422


def test_zero_trigger_level_rejected(client, headers):
    _seed(client, headers, "AMD")
    resp = _review(client, headers, "AMD", "watch", trigger_rule="close_above", trigger_level=0)
    assert resp.status_code == 422


# --- invalid rule rejected ---
def test_invalid_rule_rejected(client, headers):
    _seed(client, headers, "AMD")
    resp = _review(client, headers, "AMD", "watch", trigger_rule="close_at", trigger_level=100.0)
    assert resp.status_code == 422


# --- invalid timeframe rejected ---
def test_invalid_timeframe_rejected(client, headers):
    _seed(client, headers, "AMD")
    resp = _review(client, headers, "AMD", "watch", trigger_timeframe="5m", trigger_rule="close_above", trigger_level=100.0)
    assert resp.status_code == 422


def test_incomplete_trigger_rule_only_rejected(client, headers):
    _seed(client, headers, "AMD")
    resp = _review(client, headers, "AMD", "watch", trigger_rule="close_above")
    assert resp.status_code == 422


def test_incomplete_trigger_level_only_rejected(client, headers):
    _seed(client, headers, "AMD")
    resp = _review(client, headers, "AMD", "watch", trigger_level=100.0)
    assert resp.status_code == 422


def test_timeframe_without_rule_and_level_rejected(client, headers):
    _seed(client, headers, "AMD")
    resp = _review(client, headers, "AMD", "watch", trigger_timeframe="30m")
    assert resp.status_code == 422


def test_nan_trigger_level_rejected(client, headers):
    _seed(client, headers, "AMD")
    # Standard JSON has no NaN/Infinity literal, but Python's json module
    # accepts them as a non-standard extension by default -- FastAPI/
    # Starlette use it to parse the request body, so a client COULD send
    # this. Sent as raw bytes (not client.post(json=...), which would
    # round-trip through Python's json.dumps and refuse to serialize NaN
    # in the first place -- this test needs the literal wire bytes).
    resp = client.post(
        "/api/v1/scanner/candidates/AMD/visual-review",
        headers={**headers, "Content-Type": "application/json"},
        content='{"source":"ma_pipeline","market_structure":"bullish","location_read":"good",'
                '"clear_path_to_target":"yes","lower_tf_confirmation":"not_yet","decision":"watch",'
                '"trigger_rule":"close_above","trigger_level":NaN}',
    )
    assert resp.status_code == 422


# --- append-only review preserves old trigger; later review can use a different trigger ---
def test_append_only_preserves_old_trigger_and_allows_new_one(client, headers):
    _seed(client, headers, "AMD")
    first = _review(
        client, headers, "AMD", "watch",
        trigger_rule="close_above", trigger_level=100.50, trigger_reason="first",
    )
    assert first.status_code == 200
    first_id = first.json()["id"]

    second = _review(
        client, headers, "AMD", "watch",
        trigger_rule="close_below", trigger_level=97.00, trigger_reason="changed my mind",
    )
    assert second.status_code == 200
    second_id = second.json()["id"]
    assert second_id != first_id

    all_reviews = client.get("/api/v1/scanner/candidate-visual-reviews", headers=headers).json()
    first_row = next(r for r in all_reviews if r["id"] == first_id)
    second_row = next(r for r in all_reviews if r["id"] == second_id)
    assert first_row["trigger_rule"] == "close_above"
    assert first_row["trigger_level"] == 100.50
    assert second_row["trigger_rule"] == "close_below"
    assert second_row["trigger_level"] == 97.00


# --- approve copies trigger exactly into approved memory ---
def test_approve_copies_trigger_into_memory(client, headers):
    _seed(client, headers, "AMD")
    resp = _review(
        client, headers, "AMD", "approve",
        lower_tf_confirmation="not_yet",
        trigger_timeframe="30m", trigger_rule="close_above", trigger_level=100.50, trigger_reason="wait for reclaim",
    )
    assert resp.status_code == 200, resp.text

    memories = _memories(client, headers, ticker="AMD")
    assert len(memories) == 1
    memory = memories[0]["memory"]
    assert memory["trigger_timeframe"] == "30m"
    assert memory["trigger_rule"] == "close_above"
    assert memory["trigger_level"] == 100.50
    assert memory["trigger_reason"] == "wait for reclaim"


# --- approved memory trigger remains frozen if later live data changes ---
def test_approved_memory_trigger_frozen_across_rescans(client, headers):
    _seed(client, headers, "AMD", )
    _review(
        client, headers, "AMD", "approve",
        trigger_rule="close_above", trigger_level=100.50,
    )
    # Rescan with a different entry_price -- same setup_key (same daily structure).
    payload = {
        "source": "ma_pipeline",
        "scanned_at": "2026-08-20T18:30:00Z",
        "candidates": [{
            "ticker": "AMD", "signal": "long", "entry_price": 100.4, "ema21_4h": 99.0,
            "daily_regime": "bullish", "confidence": "high", "sma50_daily": 106.0, "sma200_daily": 104.0,
        }],
    }
    resp = client.post("/api/v1/scanner/candidates", headers=headers, json=payload)
    assert resp.status_code == 200

    memories = _memories(client, headers, ticker="AMD")
    assert len(memories) == 1
    assert memories[0]["memory"]["trigger_level"] == 100.50, "trigger must not drift with live candidate changes"


# --- watch can store trigger without creating approved memory ---
def test_watch_stores_trigger_without_creating_memory(client, headers):
    _seed(client, headers, "AMD")
    resp = _review(
        client, headers, "AMD", "watch",
        trigger_rule="close_above", trigger_level=100.50,
    )
    assert resp.status_code == 200
    assert resp.json()["trigger_level"] == 100.50
    assert _memories(client, headers, include_inactive=True, ticker="AMD") == []


# --- approve -> watch preserves old approved memory and its trigger snapshot ---
def test_approve_then_watch_preserves_memory_trigger(client, headers):
    _seed(client, headers, "AMD")
    _review(client, headers, "AMD", "approve", trigger_rule="close_above", trigger_level=100.50)
    assert len(_memories(client, headers, ticker="AMD")) == 1

    watch_resp = _review(client, headers, "AMD", "watch", trigger_rule="close_below", trigger_level=95.0)
    assert watch_resp.status_code == 200
    assert watch_resp.json()["trigger_rule"] == "close_below", "the NEW watch review has its own (different) trigger"

    active = _memories(client, headers, ticker="AMD")
    assert active == [], "the memory is withdrawn from the active listing"
    historical = _memories(client, headers, include_inactive=True, ticker="AMD")
    assert len(historical) == 1
    assert historical[0]["memory"]["trigger_rule"] == "close_above", "the frozen memory keeps the ORIGINAL approval-time trigger"
    assert historical[0]["memory"]["trigger_level"] == 100.50
    assert historical[0]["monitor_state"]["state"] == "WITHDRAWN"


# --- existing reviews with no trigger still work ---
def test_review_without_trigger_still_works(client, headers):
    _seed(client, headers, "AMD")
    resp = _review(client, headers, "AMD", "approve")
    assert resp.status_code == 200
    assert resp.json()["trigger_level"] is None
    memory = _memories(client, headers, ticker="AMD")[0]["memory"]
    assert memory["trigger_level"] is None


# --- FFIV/NVDA/CLH-style existing backfilled memories remain unchanged ---
def test_backfill_never_fabricates_trigger(client, headers):
    _seed(client, headers, "NVDA")
    # Simulate a pre-existing review written BEFORE this feature existed --
    # i.e. one with no trigger fields at all (exactly FFIV/NVDA/CLH's real
    # historical rows).
    approve_resp = _review(client, headers, "NVDA", "approve")
    assert approve_resp.status_code == 200
    assert approve_resp.json()["trigger_level"] is None

    # Delete the memory this created, to simulate "a pre-existing production
    # approval with no memory row yet" -- same setup as
    # tests/approved_setup_memory_v1.py's own backfill test.
    conn = router._get_db()
    conn.execute("DELETE FROM approved_setup_monitor_state")
    conn.execute("DELETE FROM approved_setup_memories")
    conn.commit()
    conn.close()

    result = client.post("/api/v1/scanner/candidates/approved-setup-memory/backfill", headers=headers)
    assert result.status_code == 200, result.text
    assert any(e["ticker"] == "NVDA" for e in result.json()["backfilled"])

    memory = _memories(client, headers, ticker="NVDA")[0]["memory"]
    assert memory["trigger_timeframe"] is None
    assert memory["trigger_rule"] is None
    assert memory["trigger_level"] is None
    assert memory["trigger_reason"] is None
    assert memory["snapshot_origin"] == "live_backfill"
    assert memory["snapshot_exact"] is False


# --- zero option hydration (covered by the autouse spy above; explicit check here too) ---
def test_no_option_hydration_on_trigger_endpoints(client, headers):
    _seed(client, headers, "AMD")
    _review(client, headers, "AMD", "approve", trigger_rule="close_above", trigger_level=100.50)
    _memories(client, headers, include_inactive=True)
    # No AssertionError raised by the _forbid_option_hydration spy means this passed.


# --- no candidate_promotions writes / no ENTER_NOW / no monitor-state trigger transitions ---
def test_no_promotions_writes_and_monitor_state_unaffected(client, headers):
    _seed(client, headers, "AMD")
    _review(client, headers, "AMD", "approve", trigger_rule="close_above", trigger_level=100.50)

    conn = router._get_db()
    promo_count = conn.execute("SELECT COUNT(*) FROM candidate_promotions").fetchone()[0]
    conn.close()
    assert promo_count == 0

    monitor_state = _memories(client, headers, ticker="AMD")[0]["monitor_state"]
    assert monitor_state["state"] == "APPROVED", "a trigger must not itself cause any state transition"
    assert monitor_state["terminal_at"] is None
    assert monitor_state["last_checked_at"] is None
