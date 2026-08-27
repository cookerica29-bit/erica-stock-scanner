"""candidate_promotions: mutable-upsert -> append-only event log.

Scope: the schema migration itself (both the old composite-PK production
shape and the even-older narrow pre-annotation-fields shape), the write path
(_store_promotion is now a plain INSERT, never an overwrite), and the one
read call site (list_candidate_promotions) that has to keep its existing
"one row per ticker/source, the latest" API contract on top of a table that
no longer enforces that as a constraint -- see candidates_router.py's
_ensure_candidate_promotions_schema for why this matters: every previous
promotion of the same ticker used to vanish, silently, with zero trace.

Does not touch outcome tracking (Step 2) -- that's the next, separate change
on top of this one.
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


def _promotion_payload(ticker="AAPL", source="ma_pipeline", entry_price=100.0, promoted_at="2026-08-20T14:00:00Z"):
    return {
        "ticker": ticker, "source": source, "direction": "long", "entry_price": entry_price,
        "stop": entry_price - 5, "target": entry_price + 10, "risk_reward": 2.0,
        "rr_warning": False, "no_valid_target": False, "promoted_at": promoted_at,
        "position_size": None, "atr14": 1.5, "atr_multiplier": 1.5, "rr_warning_threshold": 1.5,
        "min_target_atr_multiple": 2.0, "target_source": "daily_swing_structure",
        "raw_target": entry_price + 10, "raw_risk_reward": 2.0, "target_clamped": False,
        "target_clamp_badge": None, "target_clamp_reason": None,
        "raw_stop": entry_price - 5, "stop_source": "atr_multiple",
        "displacement_score": 50.0, "displacement_label": "MODERATE",
        "displacement_components": {"body_percentile": 60.0}, "raw_magnitude_score": 55.0,
        "displacement_read": "favorable", "bos_confirmed": False, "bos_details": None,
    }


def _open_db(router):
    conn = sqlite3.connect(os.environ["KAIROS_CANDIDATES_DB"])
    conn.row_factory = sqlite3.Row
    router._initialize_candidates_schema(conn)
    return conn


# -- write path: append-only, never overwrites --------------------------------

def test_promote_same_ticker_twice_creates_two_rows_neither_overwritten(router):
    conn = _open_db(router)
    first_id = router._store_promotion(conn, _promotion_payload(entry_price=100.0, promoted_at="2026-08-20T14:00:00Z"))
    second_id = router._store_promotion(conn, _promotion_payload(entry_price=105.0, promoted_at="2026-08-24T14:00:00Z"))
    conn.commit()

    assert first_id != second_id

    rows = conn.execute(
        "SELECT * FROM candidate_promotions WHERE ticker=? AND source=? ORDER BY id",
        ("AAPL", "ma_pipeline"),
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["entry_price"] == 100.0
    assert rows[0]["promoted_at"] == "2026-08-20T14:00:00Z"
    assert rows[1]["entry_price"] == 105.0
    assert rows[1]["promoted_at"] == "2026-08-24T14:00:00Z"
    conn.close()


def test_store_promotion_returns_incrementing_row_ids(router):
    conn = _open_db(router)
    id_a = router._store_promotion(conn, _promotion_payload(ticker="AAA"))
    id_b = router._store_promotion(conn, _promotion_payload(ticker="BBB"))
    id_c = router._store_promotion(conn, _promotion_payload(ticker="AAA"))  # re-promote AAA
    conn.commit()

    assert id_a < id_b < id_c
    conn.close()


# -- read path: list_candidate_promotions keeps its old contract --------------

@pytest.fixture()
def client(router):
    app = FastAPI()
    app.include_router(router.router)
    return TestClient(app)


def test_list_candidate_promotions_returns_only_latest_per_ticker_despite_history(router, client):
    conn = _open_db(router)
    router._store_promotion(conn, _promotion_payload(entry_price=100.0, promoted_at="2026-08-20T14:00:00Z"))
    router._store_promotion(conn, _promotion_payload(entry_price=105.0, promoted_at="2026-08-24T14:00:00Z"))
    router._store_promotion(conn, _promotion_payload(entry_price=110.0, promoted_at="2026-08-27T14:00:00Z"))
    conn.commit()
    conn.close()

    resp = client.get("/api/v1/scanner/candidate-promotions", headers={"X-API-Key": "test-scanner-key"})
    assert resp.status_code == 200
    rows = resp.json()

    aapl_rows = [r for r in rows if r["ticker"] == "AAPL"]
    assert len(aapl_rows) == 1, "API contract unchanged: one row per ticker/source, even though 3 exist in the table"
    assert aapl_rows[0]["entry_price"] == 110.0  # the latest one, not the first or a random one
    assert aapl_rows[0]["promoted_at"] == "2026-08-27T14:00:00Z"


def test_list_candidate_promotions_latest_pick_is_independent_across_tickers(router, client):
    conn = _open_db(router)
    router._store_promotion(conn, _promotion_payload(ticker="AAA", entry_price=1.0, promoted_at="2026-08-20T14:00:00Z"))
    router._store_promotion(conn, _promotion_payload(ticker="AAA", entry_price=2.0, promoted_at="2026-08-24T14:00:00Z"))
    router._store_promotion(conn, _promotion_payload(ticker="BBB", entry_price=3.0, promoted_at="2026-08-26T14:00:00Z"))
    conn.commit()
    conn.close()

    resp = client.get("/api/v1/scanner/candidate-promotions", headers={"X-API-Key": "test-scanner-key"})
    rows = {r["ticker"]: r for r in resp.json()}

    assert len(rows) == 2
    assert rows["AAA"]["entry_price"] == 2.0
    assert rows["BBB"]["entry_price"] == 3.0


# -- schema migration: composite PK (today's real production shape) ----------

def test_composite_pk_migration_preserves_rows_and_assigns_ids(router):
    """Simulates today's actual production candidate_promotions table --
    composite PRIMARY KEY (ticker, source), already has every annotation
    column from earlier this session (raw_target, stop_source,
    displacement_*, bos_*) -- with real-shaped values resembling the 7
    live promotions on record, and confirms the migration is lossless."""
    conn = sqlite3.connect(os.environ["KAIROS_CANDIDATES_DB"])
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE candidate_promotions (
            ticker TEXT NOT NULL,
            source TEXT NOT NULL,
            direction TEXT NOT NULL,
            entry_price REAL NOT NULL,
            stop REAL NOT NULL,
            target REAL,
            risk_reward REAL,
            rr_warning INTEGER NOT NULL,
            no_valid_target INTEGER NOT NULL DEFAULT 0,
            promoted_at TEXT NOT NULL,
            position_size REAL,
            atr14 REAL NOT NULL,
            atr_multiplier REAL NOT NULL,
            rr_warning_threshold REAL NOT NULL,
            min_target_atr_multiple REAL NOT NULL DEFAULT 2.0,
            target_source TEXT NOT NULL,
            raw_target REAL,
            raw_risk_reward REAL,
            target_clamped INTEGER NOT NULL DEFAULT 0,
            target_clamp_badge TEXT,
            target_clamp_reason TEXT,
            raw_stop REAL,
            stop_source TEXT,
            displacement_score REAL,
            displacement_label TEXT,
            displacement_components_json TEXT,
            raw_magnitude_score REAL,
            displacement_read TEXT,
            bos_confirmed INTEGER NOT NULL DEFAULT 0,
            bos_level REAL,
            PRIMARY KEY (ticker, source)
        )
        """
    )
    # Real-shaped rows, mirroring the 7 live promotions pulled from
    # production this session (VZ, ORCL, XLF, AAPL, CVNA, LPX, RYN).
    seed_rows = [
        ("VZ", "ma_pipeline", "long", 49.9, 48.6007, None, None, 0, 1, "2026-08-24T14:55:05.868552+00:00", "daily_swing_structure"),
        ("ORCL", "ma_pipeline", "short", 141.295, 150.9577, 114.5, 1.62, 0, 0, "2026-08-24T14:54:22.121386+00:00", "daily_swing_structure"),
        ("AAPL", "ma_pipeline", "long", 311.645, 301.5956, 334.7, 2.29, 0, 0, "2026-08-24T14:53:44.147570+00:00", "daily_swing_structure"),
    ]
    for ticker, source, direction, entry, stop, target, rr, rr_warn, no_target, promoted_at, target_source in seed_rows:
        conn.execute(
            """
            INSERT INTO candidate_promotions
                (ticker, source, direction, entry_price, stop, target, risk_reward,
                 rr_warning, no_valid_target, promoted_at, atr14, atr_multiplier,
                 rr_warning_threshold, target_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1.5, 1.5, 1.5, ?)
            """,
            (ticker, source, direction, entry, stop, target, rr, rr_warn, no_target, promoted_at, target_source),
        )
    conn.commit()

    columns_before = {info["name"] for info in conn.execute("PRAGMA table_info(candidate_promotions)").fetchall()}
    assert "id" not in columns_before

    router._ensure_candidate_promotions_schema(conn)
    conn.commit()

    ddl = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='candidate_promotions'"
    ).fetchone()["sql"]
    assert "PRIMARY KEY (ticker, source)" not in ddl

    rows = conn.execute("SELECT * FROM candidate_promotions ORDER BY ticker").fetchall()
    assert len(rows) == 3
    by_ticker = {r["ticker"]: r for r in rows}
    assert by_ticker["VZ"]["entry_price"] == 49.9
    assert by_ticker["VZ"]["target"] is None  # no_valid_target case, preserved as None not 0
    assert by_ticker["ORCL"]["target"] == 114.5
    assert by_ticker["AAPL"]["risk_reward"] == 2.29
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == 3  # all distinct, all assigned
    assert all(isinstance(i, int) and i > 0 for i in ids)
    conn.close()


def test_composite_pk_migration_is_idempotent(router):
    """Running schema init twice (e.g. two app workers starting up against
    the same fresh-migrated DB) must not re-trigger the rebuild or duplicate
    rows -- the id-based table no longer matches the composite-PK DDL check,
    so the second call should be a pure no-op ALTER-guard pass."""
    conn = sqlite3.connect(os.environ["KAIROS_CANDIDATES_DB"])
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE candidate_promotions (
            ticker TEXT NOT NULL, source TEXT NOT NULL, direction TEXT NOT NULL,
            entry_price REAL NOT NULL, stop REAL NOT NULL, target REAL, risk_reward REAL,
            rr_warning INTEGER NOT NULL, no_valid_target INTEGER NOT NULL DEFAULT 0,
            promoted_at TEXT NOT NULL, position_size REAL, atr14 REAL NOT NULL,
            atr_multiplier REAL NOT NULL, rr_warning_threshold REAL NOT NULL,
            min_target_atr_multiple REAL NOT NULL DEFAULT 2.0, target_source TEXT NOT NULL,
            PRIMARY KEY (ticker, source)
        )
        """
    )
    conn.execute(
        """
        INSERT INTO candidate_promotions
            (ticker, source, direction, entry_price, stop, target, risk_reward,
             rr_warning, no_valid_target, promoted_at, atr14, atr_multiplier,
             rr_warning_threshold, target_source)
        VALUES ('AAPL', 'ma_pipeline', 'long', 100.0, 95.0, 110.0, 2.0, 0, 0,
                '2026-08-20T14:00:00Z', 1.5, 1.5, 1.5, 'daily_swing_structure')
        """
    )
    conn.commit()

    router._ensure_candidate_promotions_schema(conn)
    conn.commit()
    router._ensure_candidate_promotions_schema(conn)
    conn.commit()

    rows = conn.execute("SELECT * FROM candidate_promotions").fetchall()
    assert len(rows) == 1
    assert rows[0]["ticker"] == "AAPL"
    conn.close()


def test_migration_failure_mid_rebuild_leaves_original_table_untouched(router, monkeypatch):
    """The specific risk this migration carries: RENAME/CREATE/DROP are DDL,
    which sqlite3's default connection mode auto-commits immediately even
    with no explicit conn.commit() and even after the connection is closed
    post-exception (confirmed empirically before adding the explicit
    BEGIN/rollback in _ensure_candidate_promotions_schema). Forces a failure
    between the RENAME and the DROP and confirms the explicit transaction
    actually protects the real data -- not just that the happy path works."""
    conn = sqlite3.connect(os.environ["KAIROS_CANDIDATES_DB"])
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE candidate_promotions (
            ticker TEXT NOT NULL, source TEXT NOT NULL, direction TEXT NOT NULL,
            entry_price REAL NOT NULL, stop REAL NOT NULL, target REAL, risk_reward REAL,
            rr_warning INTEGER NOT NULL, no_valid_target INTEGER NOT NULL DEFAULT 0,
            promoted_at TEXT NOT NULL, position_size REAL, atr14 REAL NOT NULL,
            atr_multiplier REAL NOT NULL, rr_warning_threshold REAL NOT NULL,
            min_target_atr_multiple REAL NOT NULL DEFAULT 2.0, target_source TEXT NOT NULL,
            PRIMARY KEY (ticker, source)
        )
        """
    )
    conn.execute(
        """
        INSERT INTO candidate_promotions
            (ticker, source, direction, entry_price, stop, target, risk_reward,
             rr_warning, no_valid_target, promoted_at, atr14, atr_multiplier,
             rr_warning_threshold, target_source)
        VALUES ('REAL', 'ma_pipeline', 'long', 100.0, 95.0, 110.0, 2.0, 0, 0,
                '2026-08-20T14:00:00Z', 1.5, 1.5, 1.5, 'daily_swing_structure')
        """
    )
    conn.commit()

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated failure mid-rebuild")

    monkeypatch.setattr(router, "_rebuild_candidate_promotions_table", _boom)

    with pytest.raises(RuntimeError, match="simulated failure mid-rebuild"):
        router._ensure_candidate_promotions_schema(conn)

    # The original table (and its real row) must still be there, untouched --
    # not renamed to candidate_promotions_old, not replaced by an empty table.
    tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "candidate_promotions" in tables
    assert "candidate_promotions_old" not in tables

    rows = conn.execute("SELECT * FROM candidate_promotions").fetchall()
    assert len(rows) == 1
    assert rows[0]["ticker"] == "REAL"
    assert rows[0]["entry_price"] == 100.0
    conn.close()


# -- schema migration: even-older narrow pre-annotation-columns shape --------

def test_legacy_narrow_schema_migration_still_works(router):
    """Protects the pre-existing narrow-schema rebuild path (predates
    raw_target/stop_source/displacement_*/bos_* entirely) -- confirms
    extending this function for the id-PK migration didn't regress it."""
    conn = sqlite3.connect(os.environ["KAIROS_CANDIDATES_DB"])
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE candidate_promotions (
            ticker TEXT NOT NULL,
            source TEXT NOT NULL,
            direction TEXT NOT NULL,
            entry_price REAL NOT NULL,
            stop REAL NOT NULL,
            target REAL NOT NULL,
            risk_reward REAL NOT NULL,
            rr_warning INTEGER NOT NULL,
            promoted_at TEXT NOT NULL,
            position_size REAL,
            atr14 REAL NOT NULL,
            atr_multiplier REAL NOT NULL,
            rr_warning_threshold REAL NOT NULL,
            target_source TEXT NOT NULL,
            PRIMARY KEY (ticker, source)
        )
        """
    )
    conn.execute(
        """
        INSERT INTO candidate_promotions
            (ticker, source, direction, entry_price, stop, target, risk_reward,
             rr_warning, promoted_at, atr14, atr_multiplier, rr_warning_threshold, target_source)
        VALUES ('OLD', 'ma_pipeline', 'long', 50.0, 47.0, 56.0, 2.0, 0,
                '2026-08-01T14:00:00Z', 1.0, 1.5, 1.5, 'daily_swing_structure')
        """
    )
    conn.commit()

    router._ensure_candidate_promotions_schema(conn)
    conn.commit()

    row = conn.execute("SELECT * FROM candidate_promotions WHERE ticker='OLD'").fetchone()
    assert row is not None
    assert row["entry_price"] == 50.0
    assert row["id"] > 0
    assert row["no_valid_target"] == 0
    assert row["raw_target"] is None  # backfilled by the ALTER checks, not copied (didn't exist yet)
    assert row["bos_confirmed"] == 0
    conn.close()


def test_fresh_db_gets_id_schema_directly_no_rebuild_needed(router):
    """A genuinely new install should never take the rebuild path at all --
    CREATE TABLE IF NOT EXISTS should already produce the id-PK schema."""
    conn = _open_db(router)
    ddl = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='candidate_promotions'"
    ).fetchone()["sql"]
    assert "id INTEGER PRIMARY KEY AUTOINCREMENT" in ddl
    assert "PRIMARY KEY (ticker, source)" not in ddl
    conn.close()


# -- end-to-end through the real promotion API, not just direct DB calls -----

def test_promoting_same_ticker_twice_via_status_update_api_creates_two_promotion_rows(router, monkeypatch):
    monkeypatch.setattr(
        router, "_batch_download",
        lambda tickers, period, interval: {"TRND": _trending_frame_for_promotion()},
    )
    # Promotion goes through the live-gate/execution-shadow checks
    # (_promotion_with_live_gate_context / _promotion_block_reason), which
    # need current-quote and recent-4h-bar data too -- same mocks as
    # scanner_candidates_ingestion_v1.py's ingestion lifecycle test, since a
    # real network call here (no live Alpaca creds in the test env) would
    # otherwise get treated as a block reason and 422 the promotion.
    # Entry-proximity gate needs the mocked quote to actually be near
    # whatever entry_price the candidate has -- mutable so it can track the
    # second (different) entry_price used for the re-promotion below.
    quote_price = {"value": 100.0}
    monkeypatch.setattr(
        router, "_latest_quote_for_ticker",
        lambda ticker: {
            "price": quote_price["value"], "timestamp": "2026-08-20T18:30:00Z",
            "source": "mock_latest_quote", "price_branch": "mid",
        },
    )
    monkeypatch.setattr(
        router, "_best_contract",
        lambda ticker, direction, entry, **kwargs: {
            "available": True, "execution": "Fair",
            "type": "PUT" if direction == "SHORT" else "CALL",
            "strike": 100.0, "expiry": "2026-09-18", "dte": 29, "symbol": "MOCK",
            "source": "option_chain", "bid": 1.10, "ask": 1.20, "mid": 1.15, "mark": 1.15,
            "estimated_contract_cost": 120.0,
        },
    )
    # Offset by the same amount as quote_price so the execution-shadow "hold
    # zone" check (computed from these bars relative to entry_price) still
    # passes for whichever entry_price is currently active.
    def _execution_shadow_bars(ticker):
        offset = quote_price["value"] - 100.0
        bars = [
            {
                "time": f"2026-08-19T{hour:02d}:00:00Z",
                "open": 99.0 + (idx * 0.05) + offset, "high": 101.0 + (idx * 0.05) + offset,
                "low": 98.0 + (idx * 0.05) + offset, "close": 100.0 + (idx * 0.05) + offset, "volume": 1000,
            }
            for idx, hour in enumerate(range(11))
        ] + [
            {"time": "2026-08-20T02:00:00Z", "open": 99.0 + offset, "high": 101.0 + offset, "low": 98.0 + offset, "close": 100.0 + offset, "volume": 1000},
            {"time": "2026-08-20T06:00:00Z", "open": 100.0 + offset, "high": 101.0 + offset, "low": 99.0 + offset, "close": 100.5 + offset, "volume": 1100},
            {"time": "2026-08-20T10:00:00Z", "open": 100.5 + offset, "high": 102.0 + offset, "low": 99.2 + offset, "close": 101.0 + offset, "volume": 1200},
            {"time": "2026-08-20T14:00:00Z", "open": 100.8 + offset, "high": 103.0 + offset, "low": 99.5 + offset, "close": 102.2 + offset, "volume": 1000},
        ]
        return bars

    monkeypatch.setattr(router, "_recent_4h_bars_for_execution_shadow", _execution_shadow_bars)
    app = FastAPI()
    app.include_router(router.router)
    client = TestClient(app)
    headers = {"X-API-Key": "test-scanner-key"}

    ingest_payload = {
        "source": "ma_pipeline",
        "scanned_at": "2026-08-20T14:30:00Z",
        "candidates": [{"ticker": "TRND", "signal": "long", "entry_price": 100.0, "daily_regime": "bullish"}],
    }
    client.post("/api/v1/scanner/candidates", headers=headers, json=ingest_payload)

    first = client.patch(
        "/api/v1/scanner/candidates/TRND?source=ma_pipeline",
        headers=headers, json={"status": "active"},
    )
    assert first.status_code == 200

    # Dismiss, re-ingest (simulating a later scan re-surfacing it), promote again.
    client.patch("/api/v1/scanner/candidates/TRND?source=ma_pipeline", headers=headers, json={"status": "dismissed"})
    quote_price["value"] = 105.0
    client.post("/api/v1/scanner/candidates", headers=headers, json={
        "source": "ma_pipeline",
        "scanned_at": "2026-08-24T14:30:00Z",
        "candidates": [{"ticker": "TRND", "signal": "long", "entry_price": 105.0, "daily_regime": "bullish"}],
    })
    second = client.patch(
        "/api/v1/scanner/candidates/TRND?source=ma_pipeline",
        headers=headers, json={"status": "active"},
    )
    assert second.status_code == 200

    conn = sqlite3.connect(os.environ["KAIROS_CANDIDATES_DB"])
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM candidate_promotions WHERE ticker='TRND' AND source='ma_pipeline' ORDER BY id"
    ).fetchall()
    conn.close()
    assert len(rows) == 2, "re-promoting the same ticker must append, not overwrite"
    assert rows[0]["entry_price"] != rows[1]["entry_price"] or rows[0]["promoted_at"] != rows[1]["promoted_at"]


def _trending_frame_for_promotion():
    import pandas as pd
    n = 60
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
