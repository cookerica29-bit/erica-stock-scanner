"""Approved Setup Memory -- pre-push safety review (2026-09 session).
Failure-injection tests for item 9 of that review: confirms the real
record_candidate_visual_review transaction rolls back cleanly (no
candidate_visual_reviews row, no approved_setup_memories row, no
approved_setup_monitor_state row) when a failure is injected at each of
the three write points. Same fixture pattern as
tests/approved_setup_memory_v1.py -- real FastAPI TestClient against
candidates_router.router, not a reimplementation.
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


def _row_counts(db_path):
    import sqlite3
    conn = sqlite3.connect(db_path)
    counts = {}
    for table in ("candidate_visual_reviews", "approved_setup_memories", "approved_setup_monitor_state"):
        counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    conn.close()
    return counts


def test_monitor_state_insert_failure_rolls_back_everything(client, headers, monkeypatch, tmp_path):
    """Inject a failure specifically at the approved_setup_monitor_state
    insert point -- the approved_setup_memories INSERT has already
    executed (real, not mocked) earlier in the SAME transaction. Expected:
    the WHOLE transaction rolls back -- zero rows in ALL THREE tables,
    including candidate_visual_reviews and the already-executed
    approved_setup_memories insert. A memory row must never be observable
    without its paired monitor_state row."""
    _seed(client, headers, "AMD")

    real_create = router._create_approved_setup_memory

    def _boom(conn, **kwargs):
        # Let the real approved_setup_memories INSERT actually execute
        # first (so this genuinely tests mid-transaction failure, not a
        # no-op), then fail before the paired monitor_state INSERT.
        confluence_counts = kwargs["preview"].get("confluence_counts")
        conn.execute(
            "INSERT INTO approved_setup_memories (ticker, source, direction, setup_key, approved_at, "
            "snapshot_origin, snapshot_exact) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (kwargs["ticker"], kwargs["source"], kwargs["direction"], kwargs["setup_key"],
             kwargs["approved_at"], kwargs["snapshot_origin"], 1 if kwargs["snapshot_exact"] else 0),
        )
        raise RuntimeError("injected failure: monitor_state insert never happens")

    monkeypatch.setattr(router, "_create_approved_setup_memory", _boom)

    with pytest.raises(RuntimeError, match="monitor_state insert never happens"):
        client.post(
            "/api/v1/scanner/candidates/AMD/visual-review",
            headers=headers,
            json={
                "source": "ma_pipeline", "market_structure": "bullish", "location_read": "good",
                "clear_path_to_target": "yes", "lower_tf_confirmation": "yes", "decision": "approve",
            },
        )

    db_path = str(tmp_path / "candidates.db")
    counts = _row_counts(db_path)
    assert counts["candidate_visual_reviews"] == 0, (
        "the visual-review INSERT from earlier in the SAME transaction must also roll back -- "
        f"got {counts}"
    )
    assert counts["approved_setup_memories"] == 0, (
        "the partial approved_setup_memories INSERT (before the injected failure) must roll back -- "
        f"got {counts}"
    )
    assert counts["approved_setup_monitor_state"] == 0
    monkeypatch.setattr(router, "_create_approved_setup_memory", real_create)


def test_memory_insert_failure_rolls_back_visual_review_too(client, headers, monkeypatch, tmp_path):
    """Inject a failure at _create_approved_setup_memory itself -- fails
    immediately, before any INSERT into approved_setup_memories or
    approved_setup_monitor_state executes at all. Expected: the earlier
    candidate_visual_reviews INSERT (same transaction, already executed)
    still rolls back -- a review must never be observably recorded if the
    memory sync that was supposed to accompany it never actually ran."""
    _seed(client, headers, "AMD")

    def _boom(conn, **kwargs):
        raise RuntimeError("injected failure: memory insert fails outright")

    monkeypatch.setattr(router, "_create_approved_setup_memory", _boom)

    with pytest.raises(RuntimeError, match="memory insert fails outright"):
        client.post(
            "/api/v1/scanner/candidates/AMD/visual-review",
            headers=headers,
            json={
                "source": "ma_pipeline", "market_structure": "bullish", "location_read": "good",
                "clear_path_to_target": "yes", "lower_tf_confirmation": "yes", "decision": "approve",
            },
        )

    db_path = str(tmp_path / "candidates.db")
    counts = _row_counts(db_path)
    assert counts == {"candidate_visual_reviews": 0, "approved_setup_memories": 0, "approved_setup_monitor_state": 0}, (
        f"a mid-transaction failure at the memory-insert entry point must roll back the whole transaction -- got {counts}"
    )


def test_review_change_sync_failure_rolls_back_review_too(client, headers, monkeypatch, tmp_path):
    """First approve normally (succeeds, real baseline state). Then submit
    a watch on the SAME setup_key with a failure injected inside
    _withdraw_active_memory_for_setup_key -- expected: the watch's
    candidate_visual_reviews row ALSO rolls back (must not end up with
    "review says watch" while monitor_state still reads APPROVED)."""
    _seed(client, headers, "AMD")
    approve_resp = client.post(
        "/api/v1/scanner/candidates/AMD/visual-review",
        headers=headers,
        json={
            "source": "ma_pipeline", "market_structure": "bullish", "location_read": "good",
            "clear_path_to_target": "yes", "lower_tf_confirmation": "yes", "decision": "approve",
        },
    )
    assert approve_resp.status_code == 200
    db_path = str(tmp_path / "candidates.db")
    baseline = _row_counts(db_path)
    assert baseline == {"candidate_visual_reviews": 1, "approved_setup_memories": 1, "approved_setup_monitor_state": 1}

    def _boom(conn, setup_key, at):
        raise RuntimeError("injected failure: withdrawal never happens")

    monkeypatch.setattr(router, "_withdraw_active_memory_for_setup_key", _boom)

    with pytest.raises(RuntimeError, match="withdrawal never happens"):
        client.post(
            "/api/v1/scanner/candidates/AMD/visual-review",
            headers=headers,
            json={
                "source": "ma_pipeline", "market_structure": "range", "location_read": "neutral",
                "clear_path_to_target": "no", "lower_tf_confirmation": "not_yet", "decision": "watch",
            },
        )

    after = _row_counts(db_path)
    assert after["candidate_visual_reviews"] == 1, (
        "the SECOND (watch) review write must roll back too -- there must be no 'orphan' review "
        f"recorded as watch while monitor_state was never actually withdrawn -- got {after}"
    )
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    remaining_review = conn.execute("SELECT decision FROM candidate_visual_reviews").fetchone()
    monitor_state = conn.execute("SELECT state FROM approved_setup_monitor_state").fetchone()
    conn.close()
    assert remaining_review["decision"] == "approve", "the original approve review must be the only one left"
    assert monitor_state["state"] == "APPROVED", (
        "monitor_state must still read APPROVED -- never allowed to observe "
        "'review changed to watch' with monitor still active, or vice versa"
    )
