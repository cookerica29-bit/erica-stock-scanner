"""source_universe DB round-trip through the public candidate-ingestion API.

Complements tests/ma_pipeline_curated_watchlist_v1.py (the merge/tagging
logic itself) and the source_universe assertion already updated in
tests/scanner_candidates_ingestion_v1.py (the None-default case). This file
covers the non-None cases end-to-end through the real
POST /candidates -> GET /candidates round trip, and the schema-migration
guard for a pre-existing table that predates this column.
"""

import os
import sqlite3
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture()
def router(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", str(tmp_path / "candidates.db"))
    monkeypatch.setenv("KAIROS_SCANNER_API_KEY", "test-scanner-key")
    import candidates_router
    return candidates_router


@pytest.fixture()
def client(router):
    app = FastAPI()
    app.include_router(router.router)
    return TestClient(app)


def _post_candidate(client, ticker, source_universe):
    payload = {
        "source": "ma_pipeline",
        "scanned_at": "2026-08-27T14:30:00Z",
        "candidates": [{
            "ticker": ticker,
            "signal": "long",
            "entry_price": 100.0,
            "source_universe": source_universe,
        }],
    }
    return client.post(
        "/api/v1/scanner/candidates",
        headers={"X-API-Key": "test-scanner-key"},
        json=payload,
    )


@pytest.mark.parametrize("source_universe", ["broker_feed", "curated_watchlist", "both", None])
def test_source_universe_round_trips_through_post_and_get(client, source_universe):
    resp = _post_candidate(client, "ZZZZ", source_universe)
    assert resp.status_code == 200

    listed = client.get(
        "/api/v1/scanner/candidates",
        headers={"X-API-Key": "test-scanner-key"},
    ).json()
    row = next(c for c in listed if c["ticker"] == "ZZZZ")
    assert row["source_universe"] == source_universe


def test_source_universe_rejects_unknown_value(client):
    resp = _post_candidate(client, "ZZZZ", "made_up_universe")
    assert resp.status_code == 422


def test_ensure_candidates_schema_adds_missing_column(router):
    """Simulates a production DB created before source_universe existed:
    CREATE TABLE IF NOT EXISTS is a no-op against it, so the ALTER-based
    migration guard has to actually add the column."""
    conn = sqlite3.connect(os.environ["KAIROS_CANDIDATES_DB"])
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE candidates (
            ticker TEXT NOT NULL,
            source TEXT NOT NULL,
            signal TEXT NOT NULL,
            entry_price REAL,
            ema21_4h REAL,
            daily_regime TEXT,
            confidence TEXT,
            sma50_daily REAL,
            sma200_daily REAL,
            status TEXT NOT NULL DEFAULT 'new',
            scanned_at TEXT NOT NULL,
            expires_at TEXT,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (ticker, source)
        )
        """
    )
    conn.commit()
    columns_before = {info[1] for info in conn.execute("PRAGMA table_info(candidates)").fetchall()}
    assert "source_universe" not in columns_before

    router._ensure_candidates_schema(conn)
    conn.commit()

    columns_after = {info[1] for info in conn.execute("PRAGMA table_info(candidates)").fetchall()}
    assert "source_universe" in columns_after
    conn.close()
