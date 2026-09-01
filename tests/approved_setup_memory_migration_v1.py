"""Approved Setup Memory -- pre-push safety review (2026-09 session), item
8: existing-production DB migration safety.

Simulates a database shaped like the current, already-deployed production
database -- candidates / candidate_visual_reviews / candidate_promotions
(among others) populated with real-shaped rows -- that does NOT yet
contain approved_setup_memories / approved_setup_monitor_state. Confirms
that booting against it (the real _get_db() -> _initialize_candidates_schema
path, not a reimplementation) adds the two new tables without altering a
single existing row anywhere.
"""

import sys
import sqlite3
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import candidates_router as router  # noqa: E402

PRE_EXISTING_TABLES = [
    "candidates", "candidate_visual_reviews", "candidate_promotions", "candidate_plan_previews",
    "candidate_status_history", "candidate_ranking_snapshots", "candidate_ai_chart_reviews",
]


def _snapshot(conn):
    result = {}
    for table in PRE_EXISTING_TABLES:
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        result[table] = [dict(row) for row in rows]
    return result


@pytest.fixture()
def pre_migration_db(tmp_path):
    """A real database built by the CURRENT schema function, then
    physically stripped of the two new tables/indexes -- the closest
    faithful reproduction of "production before this feature shipped"
    achievable without a second, frozen copy of the schema function to
    maintain and let drift out of sync."""
    db_path = str(tmp_path / "candidates.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    router._initialize_candidates_schema(conn)
    for index in (
        "idx_approved_setup_monitor_state_memory",
        "idx_approved_setup_monitor_state_ticker_source",
        "idx_approved_setup_monitor_state_setup_key",
        "idx_approved_setup_memories_ticker_source",
        "idx_approved_setup_memories_setup_key",
    ):
        conn.execute(f"DROP INDEX IF EXISTS {index}")
    conn.execute("DROP TABLE approved_setup_monitor_state")
    conn.execute("DROP TABLE approved_setup_memories")
    conn.commit()

    conn.execute(
        "INSERT INTO candidates (ticker, source, signal, entry_price, ema21_4h, daily_regime, confidence, "
        "sma50_daily, sma200_daily, status, scanned_at, expires_at, updated_at, source_universe) "
        "VALUES ('FFIV','ma_pipeline','long',405.76,400.0,'bullish','high',390.0,380.0,'new',"
        "'2026-08-20T14:30:00Z',NULL,'2026-08-31T14:00:07Z',NULL)"
    )
    conn.execute(
        "INSERT INTO candidate_visual_reviews (ticker, source, setup_key, review_type, market_structure, "
        "location_read, clear_path_to_target, lower_tf_confirmation, practical_rejection_reason, decision, "
        "note, reviewed_at) VALUES ('FFIV','ma_pipeline','FFIV|ma_pipeline|long|392.73|433.66','visual',"
        "'bullish','neutral','yes','yes',NULL,'approve','pre-existing production approval',"
        "'2026-08-31T21:08:53Z')"
    )
    conn.execute(
        "INSERT INTO candidate_promotions (ticker, source, direction, entry_price, stop, target, risk_reward, "
        "rr_warning, no_valid_target, promoted_at, atr14, atr_multiplier, rr_warning_threshold, target_source) "
        "VALUES ('IGV','ma_pipeline','long',109.75,107.95,117.02,4.04,0,0,'2026-08-31T18:00:28Z',"
        "2.7934,1.5,1.5,'daily_swing_structure')"
    )
    conn.commit()

    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "approved_setup_memories" not in tables
    assert "approved_setup_monitor_state" not in tables
    conn.close()
    return db_path


def test_migration_adds_new_tables_without_touching_existing_rows(pre_migration_db, monkeypatch):
    conn_before = sqlite3.connect(pre_migration_db)
    conn_before.row_factory = sqlite3.Row
    before = _snapshot(conn_before)
    conn_before.close()
    assert before["candidates"], "sanity: real pre-existing data actually seeded"
    assert before["candidate_visual_reviews"]
    assert before["candidate_promotions"]

    # A real restart bypasses the in-process _schema_ready_db_paths cache
    # too -- discard it explicitly so this test exercises the actual
    # migration path, not a short-circuited no-op.
    router._schema_ready_db_paths.discard(pre_migration_db)
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", pre_migration_db)
    monkeypatch.setenv("KAIROS_SCANNER_API_KEY", "test-scanner-key")

    conn_after = router._get_db()
    tables_after = {row[0] for row in conn_after.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "approved_setup_memories" in tables_after
    assert "approved_setup_monitor_state" in tables_after

    after = _snapshot(conn_after)
    assert before == after, "migration must not alter a single existing row in any pre-existing table"

    new_counts = {
        "approved_setup_memories": conn_after.execute("SELECT COUNT(*) FROM approved_setup_memories").fetchone()[0],
        "approved_setup_monitor_state": conn_after.execute("SELECT COUNT(*) FROM approved_setup_monitor_state").fetchone()[0],
    }
    assert new_counts == {"approved_setup_memories": 0, "approved_setup_monitor_state": 0}
    conn_after.close()
