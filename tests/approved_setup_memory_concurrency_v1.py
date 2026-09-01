"""Approved Setup Memory -- pre-push safety review (2026-09 session), item
2: concurrency/duplicate safety.

The duplicate-prevention check in _sync_approved_setup_memory_on_review is
application-level lookup-before-insert (SELECT active memory, then INSERT
if none found) -- a classic TOCTOU shape. This test proves, empirically,
that two concurrent approve requests for the SAME setup_key cannot
produce two active memories in the CURRENT code, and explains why: every
write in record_candidate_visual_review happens inside one sqlite3
transaction that starts at the FIRST write statement (the
candidate_visual_reviews INSERT, since this codebase never sets
isolation_level=None / autocommit) -- and SQLite's WAL mode allows only
one writer transaction at a time. So by the time either request's
check-then-act memory logic runs, it has ALREADY acquired the single
writer lock for its entire remaining transaction; the other request's own
first write blocks (via PRAGMA busy_timeout=30000, set in _get_db) until
the first fully commits. The two requests are serialized well before the
memory-sync check ever runs -- not because of anything this module does
deliberately, but as a structural consequence of (a) one big transaction
per request, (b) WAL single-writer semantics, (c) a real write happening
before the check. This holds across multiple worker processes too (WAL's
writer lock is OS/file-level, not an in-process Python lock) -- unlike
e.g. _plan_preview_refresh_lock elsewhere in this file, which explicitly
does NOT claim that.

The race window is widened artificially here (a real production race
window is microseconds) specifically to prove the serialization holds
even in the worst case, not to claim races are rare in practice.
"""

import sys
import threading
import time
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
def db_path(tmp_path, monkeypatch):
    path = str(tmp_path / "candidates.db")
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", path)
    monkeypatch.setenv("KAIROS_SCANNER_API_KEY", "test-scanner-key")
    return path


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


def test_concurrent_approve_requests_never_duplicate_active_memory(db_path, monkeypatch):
    headers = {"X-API-Key": "test-scanner-key"}

    app = FastAPI()
    app.include_router(router.router)
    seed_client = TestClient(app)
    seed_client.post("/api/v1/scanner/candidates", headers=headers, json={
        "source": "ma_pipeline", "scanned_at": "2026-08-20T14:30:00Z",
        "candidates": [{
            "ticker": "AMD", "signal": "long", "entry_price": 100.0, "ema21_4h": 99.0,
            "daily_regime": "bullish", "confidence": "high", "sma50_daily": 106.0, "sma200_daily": 104.0,
        }],
    })

    # Artificially widen the check-then-act race window so an interleaving
    # is guaranteed to happen if the code were NOT already serialized by
    # the transaction structure described above.
    real_check = router._active_monitor_state_row_for_setup_key
    check_calls = []

    def _slow_check(conn, setup_key):
        result = real_check(conn, setup_key)
        check_calls.append(setup_key)
        time.sleep(0.3)
        return result

    monkeypatch.setattr(router, "_active_monitor_state_row_for_setup_key", _slow_check)

    results = []

    def _approve():
        local_app = FastAPI()
        local_app.include_router(router.router)
        c = TestClient(local_app)
        resp = c.post("/api/v1/scanner/candidates/AMD/visual-review", headers=headers, json={
            "source": "ma_pipeline", "market_structure": "bullish", "location_read": "good",
            "clear_path_to_target": "yes", "lower_tf_confirmation": "yes",
            "confirmation_rule": "close_above", "confirmation_level": 100.0, "decision": "approve",
        })
        results.append(resp.status_code)

    t1 = threading.Thread(target=_approve)
    t2 = threading.Thread(target=_approve)
    t1.start()
    time.sleep(0.05)  # let t1 begin its transaction (and its slow check) before t2 starts
    t2.start()
    t1.join()
    t2.join()

    assert results == [200, 200], "both concurrent requests must still succeed, not error out"
    assert len(check_calls) == 2, "sanity: both threads really did reach the check -- this is a genuine concurrent scenario"

    import sqlite3
    conn = sqlite3.connect(db_path)
    # CONFIRMED, not APPROVED -- both concurrent requests carry
    # confirmation_rule/confirmation_level (Type A). This also confirms
    # the revision-comparison logic correctly treats a byte-identical
    # concurrent resubmission as "no difference" (a no-op), not as new
    # evidence -- the two submitted bodies are exactly the same, so a
    # second memory revision must NOT be created just because they raced.
    active_count = conn.execute(
        "SELECT COUNT(*) FROM approved_setup_monitor_state WHERE state='CONFIRMED'"
    ).fetchone()[0]
    memory_count = conn.execute("SELECT COUNT(*) FROM approved_setup_memories").fetchone()[0]
    review_count = conn.execute("SELECT COUNT(*) FROM candidate_visual_reviews").fetchone()[0]
    conn.close()

    assert memory_count == 1, f"exactly one memory must exist even under a widened concurrent race -- got {memory_count}"
    assert active_count == 1, f"exactly one ACTIVE monitor_state must exist -- got {active_count}"
    assert review_count == 2, "both visual-review submissions ARE real, distinct events and both persist -- only the memory must not duplicate"


def test_concurrent_watch_requests_never_duplicate_active_memory(db_path, monkeypatch):
    """Watch Lifecycle V1: decision='watch' + a complete trigger contract is
    a BRAND NEW code path for memory creation -- before this work,
    decision='watch' never created a memory at all (see
    _sync_approved_setup_memory_on_review), so this exact race-condition
    protection had never been exercised for it. The underlying mechanism
    (one big transaction per request, SQLite WAL single-writer semantics)
    is unchanged and applies identically regardless of source_decision --
    this test proves that identity empirically for the watch path rather
    than just asserting it by analogy."""
    headers = {"X-API-Key": "test-scanner-key"}

    app = FastAPI()
    app.include_router(router.router)
    seed_client = TestClient(app)
    seed_client.post("/api/v1/scanner/candidates", headers=headers, json={
        "source": "ma_pipeline", "scanned_at": "2026-08-20T14:30:00Z",
        "candidates": [{
            "ticker": "AMD", "signal": "long", "entry_price": 100.0, "ema21_4h": 99.0,
            "daily_regime": "bullish", "confidence": "high", "sma50_daily": 106.0, "sma200_daily": 104.0,
        }],
    })

    real_check = router._active_monitor_state_row_for_setup_key
    check_calls = []

    def _slow_check(conn, setup_key):
        result = real_check(conn, setup_key)
        check_calls.append(setup_key)
        time.sleep(0.3)
        return result

    monkeypatch.setattr(router, "_active_monitor_state_row_for_setup_key", _slow_check)

    results = []

    def _watch():
        local_app = FastAPI()
        local_app.include_router(router.router)
        c = TestClient(local_app)
        resp = c.post("/api/v1/scanner/candidates/AMD/visual-review", headers=headers, json={
            "source": "ma_pipeline", "market_structure": "bullish", "location_read": "good",
            "clear_path_to_target": "yes", "lower_tf_confirmation": "not_yet", "decision": "watch",
            "trigger_timeframe": "30m", "trigger_rule": "close_above", "trigger_level": 100.0,
            "trigger_reason": "waiting for reclaim",
        })
        results.append(resp.status_code)

    t1 = threading.Thread(target=_watch)
    t2 = threading.Thread(target=_watch)
    t1.start()
    time.sleep(0.05)  # let t1 begin its transaction (and its slow check) before t2 starts
    t2.start()
    t1.join()
    t2.join()

    assert results == [200, 200], "both concurrent watch requests must still succeed, not error out"
    assert len(check_calls) == 2, "sanity: both threads really did reach the check -- this is a genuine concurrent scenario"

    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # WAITING_FOR_TRIGGER, not CONFIRMED -- decision='watch' with a
    # complete trigger contract and no confirmation_rule/confirmation_level
    # (Type B). Also confirms the byte-identical concurrent resubmission
    # is correctly treated as a no-op, not a spurious second revision.
    active_rows = conn.execute(
        "SELECT * FROM approved_setup_monitor_state WHERE state='WAITING_FOR_TRIGGER'"
    ).fetchall()
    memory_rows = conn.execute("SELECT * FROM approved_setup_memories").fetchall()
    review_count = conn.execute("SELECT COUNT(*) FROM candidate_visual_reviews").fetchone()[0]
    conn.close()

    assert len(memory_rows) == 1, f"exactly one memory must exist even under a widened concurrent watch race -- got {len(memory_rows)}"
    assert memory_rows[0]["source_decision"] == "watch", "the single created memory must be tagged source_decision='watch'"
    assert len(active_rows) == 1, f"exactly one ACTIVE monitor_state must exist -- got {len(active_rows)}"
    assert review_count == 2, "both visual-review submissions ARE real, distinct events and both persist -- only the memory must not duplicate"
