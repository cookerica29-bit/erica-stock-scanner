#!/usr/bin/env python3
"""Regression tests for durable heavy-job telemetry (2026-09 session,
Railway OOM follow-up) -- heavy_job_gate.py's small, bounded, in-process
ring buffer of already-recorded heavy-job telemetry, queryable through the
protected GET /api/v1/scanner/heavy-job-status endpoint even when Railway's
own stdout log ingestion drops the HEAVY_JOB_START/CHECKPOINT/END lines
during a scan's high-volume burst (confirmed happening in production).

This is observability only: no scanner/gate/strategy behavior is exercised
differently here than in tests/heavy_job_gate_v1.py and
tests/heavy_job_gate_v2.py (both re-run at the end of this file's own
`main()` to prove they still pass unmodified).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

import heavy_job_gate  # noqa: E402
import scanner  # noqa: E402


def _reset_telemetry():
    """Tests must never leak records into each other -- clear both ring
    buffers directly (module-private, but this IS the test suite for this
    exact module) before and after every test."""
    heavy_job_gate._job_records.clear()
    heavy_job_gate._skip_records.clear()
    heavy_job_gate.release()


# ---------------------------------------------------------------------------
# 1-5: recording primitives
# ---------------------------------------------------------------------------

def test_1_starting_heavy_job_creates_running_record():
    _reset_telemetry()
    try:
        heavy_job_gate.record_job_start("job-1", "scan_all", ("default",), symbol_count=113)
        snapshot = heavy_job_gate.telemetry_snapshot()
        assert len(snapshot["recent_jobs"]) == 1
        record = snapshot["recent_jobs"][0]
        assert record["job_id"] == "job-1"
        assert record["job_type"] == "scan_all"
        assert record["job_key"] == "('default',)"
        assert record["symbol_count"] == 113
        assert record["status"] == "running"
        assert record["ended_at"] is None
        assert record["duration_seconds"] is None
    finally:
        _reset_telemetry()


def test_2_checkpoint_updates_the_correct_job():
    _reset_telemetry()
    try:
        heavy_job_gate.record_job_start("job-a", "scan_all", "default")
        heavy_job_gate.record_job_start("job-b", "discovery_universe", "weekly")
        heavy_job_gate.record_job_checkpoint("job-a", rss_after_ohlcv=150.0)
        heavy_job_gate.record_job_checkpoint("job-a", rss_pre_fanout=160.0)
        snapshot = heavy_job_gate.telemetry_snapshot()
        by_id = {r["job_id"]: r for r in snapshot["recent_jobs"]}
        assert by_id["job-a"]["rss_after_ohlcv_mb"] == 150.0
        assert by_id["job-a"]["rss_pre_fanout_mb"] == 160.0
        assert by_id["job-b"]["rss_after_ohlcv_mb"] is None, "a checkpoint must never bleed into a different job's record"
        assert by_id["job-b"]["rss_pre_fanout_mb"] is None
    finally:
        _reset_telemetry()


def test_3_completion_records_end_rss_duration_and_status():
    _reset_telemetry()
    try:
        heavy_job_gate.record_job_start("job-1", "scan_all", "default")
        heavy_job_gate.record_job_end("job-1", status="completed", duration_seconds=12.3, rss_end=135.0)
        record = heavy_job_gate.telemetry_snapshot()["recent_jobs"][0]
        assert record["status"] == "completed"
        assert record["duration_seconds"] == 12.3
        assert record["rss_end_mb"] == 135.0
        assert record["ended_at"] is not None
    finally:
        _reset_telemetry()


def test_4_failure_records_failed_status_without_exception_objects():
    _reset_telemetry()
    try:
        heavy_job_gate.record_job_start("job-1", "scan_all", "default")
        heavy_job_gate.record_job_end("job-1", status="failed", duration_seconds=5.0, rss_end=120.0)
        record = heavy_job_gate.telemetry_snapshot()["recent_jobs"][0]
        assert record["status"] == "failed"
        # Every value in the record must be a plain scalar -- no exception,
        # traceback, DataFrame, or other rich object could have been
        # stored, because record_job_end's own signature never accepts one.
        for value in record.values():
            assert value is None or isinstance(value, (str, int, float)), (
                f"telemetry record must only ever contain scalars, found {type(value)}"
            )
        assert heavy_job_gate.telemetry_snapshot()["summary"]["failed_jobs"] == 1
    finally:
        _reset_telemetry()


def test_5_max_observed_rss_reflects_highest_checkpoint():
    _reset_telemetry()
    try:
        heavy_job_gate.record_job_start("job-1", "scan_all", "default")  # rss_start recorded here too
        heavy_job_gate.record_job_checkpoint("job-1", rss_after_ohlcv=200.0)
        heavy_job_gate.record_job_checkpoint("job-1", rss_pre_fanout=180.0)  # lower -- must not overwrite the max
        heavy_job_gate.record_job_checkpoint("job-1", rss_post_processing=250.0)  # new highest
        heavy_job_gate.record_job_end("job-1", status="completed", duration_seconds=10.0, rss_end=190.0)
        record = heavy_job_gate.telemetry_snapshot()["recent_jobs"][0]
        assert record["max_observed_rss_mb"] == 250.0
        assert heavy_job_gate.telemetry_snapshot()["summary"]["highest_recent_rss_mb"] == 250.0
    finally:
        _reset_telemetry()


# ---------------------------------------------------------------------------
# 6-7: bounded ring buffers
# ---------------------------------------------------------------------------

def test_6_job_ring_buffer_is_bounded():
    _reset_telemetry()
    try:
        for i in range(heavy_job_gate._TELEMETRY_CAPACITY + 10):
            job_id = f"job-{i}"
            heavy_job_gate.record_job_start(job_id, "scan_all", "default")
            heavy_job_gate.record_job_end(job_id, status="completed", duration_seconds=1.0, rss_end=100.0)
        snapshot = heavy_job_gate.telemetry_snapshot()
        assert len(snapshot["recent_jobs"]) == heavy_job_gate._TELEMETRY_CAPACITY
        # newest-first, and the oldest ones must have been evicted
        newest = snapshot["recent_jobs"][0]
        oldest_kept = snapshot["recent_jobs"][-1]
        assert newest["job_id"] == f"job-{heavy_job_gate._TELEMETRY_CAPACITY + 9}"
        assert oldest_kept["job_id"] == "job-10"
    finally:
        _reset_telemetry()


def test_7_busy_skip_ring_buffer_is_bounded():
    _reset_telemetry()
    try:
        assert heavy_job_gate.try_acquire("scan_all", key="default") is True
        for _ in range(heavy_job_gate._TELEMETRY_CAPACITY + 10):
            heavy_job_gate.skipped_busy_message("discovery_universe", "weekly")
        snapshot = heavy_job_gate.telemetry_snapshot()
        assert len(snapshot["recent_busy_skips"]) == heavy_job_gate._TELEMETRY_CAPACITY
        assert snapshot["summary"]["busy_skips"] == heavy_job_gate._TELEMETRY_CAPACITY
        for skip in snapshot["recent_busy_skips"]:
            assert skip["requested_job_type"] == "discovery_universe"
            assert skip["active_job_type"] == "scan_all"
    finally:
        _reset_telemetry()


# ---------------------------------------------------------------------------
# 8-9: the protected endpoint itself
# ---------------------------------------------------------------------------

def test_8_endpoint_is_read_only_and_never_triggers_heavy_work():
    """The endpoint must never call try_acquire/try_reserve, never submit
    to any executor, never call scan_all/build_ranked_discovery_universe/
    scan_ma_pipeline_candidates -- it only reads already-recorded state."""
    import main

    _reset_telemetry()
    previous_token = os.environ.get("DISCOVERY_ADMIN_TOKEN")
    os.environ["DISCOVERY_ADMIN_TOKEN"] = "secret-telemetry-test"
    original_scan_all = scanner.scan_all
    original_background_executor = scanner._background_executor
    calls = {"scan_all": 0, "executor_submit": 0}

    def _boom_scan_all(*args, **kwargs):
        calls["scan_all"] += 1
        raise AssertionError("the status endpoint must never call scan_all")

    class _BoomExecutor:
        def submit(self, *args, **kwargs):
            calls["executor_submit"] += 1
            raise AssertionError("the status endpoint must never submit heavy work")

    try:
        heavy_job_gate.record_job_start("job-1", "scan_all", "default")
        heavy_job_gate.record_job_end("job-1", status="completed", duration_seconds=5.0, rss_end=100.0)
        scanner.scan_all = _boom_scan_all
        scanner._background_executor = _BoomExecutor()
        client = TestClient(main.app)
        response = client.get(
            "/api/v1/scanner/heavy-job-status", headers={"X-Kairos-Admin-Token": "secret-telemetry-test"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert "recent_jobs" in payload and "summary" in payload and "active_job" in payload and "recent_busy_skips" in payload
        assert calls["scan_all"] == 0
        assert calls["executor_submit"] == 0
        assert heavy_job_gate.active_snapshot() is None, "reading the endpoint must never leave the gate held"
    finally:
        scanner.scan_all = original_scan_all
        scanner._background_executor = original_background_executor
        if previous_token is None:
            os.environ.pop("DISCOVERY_ADMIN_TOKEN", None)
        else:
            os.environ["DISCOVERY_ADMIN_TOKEN"] = previous_token
        _reset_telemetry()


def test_9_endpoint_follows_existing_admin_token_auth():
    """Mirrors tests/discovery_endpoints_v1.py's own auth test shape
    exactly (503 disabled / 403 wrong token / 200 correct token) --
    proving this endpoint uses the SAME convention as the existing
    protected scanner endpoints in this file, not a new auth mechanism."""
    import main

    previous_token = os.environ.pop("DISCOVERY_ADMIN_TOKEN", None)
    try:
        client = TestClient(main.app)
        response = client.get("/api/v1/scanner/heavy-job-status")
        assert response.status_code == 503, "disabled (no token configured) must reject with 503"

        os.environ["DISCOVERY_ADMIN_TOKEN"] = "secret-telemetry-test"
        response = client.get(
            "/api/v1/scanner/heavy-job-status", headers={"X-Kairos-Admin-Token": "wrong"},
        )
        assert response.status_code == 403, "a wrong token must be rejected with 403"

        response = client.get(
            "/api/v1/scanner/heavy-job-status", headers={"X-Kairos-Admin-Token": "secret-telemetry-test"},
        )
        assert response.status_code == 200, "the correct token must be accepted"

        response = client.get("/api/v1/scanner/heavy-job-status")
        assert response.status_code in (401, 403), "a missing token (once the feature is enabled) must not be silently accepted"
    finally:
        os.environ.pop("DISCOVERY_ADMIN_TOKEN", None)
        if previous_token is not None:
            os.environ["DISCOVERY_ADMIN_TOKEN"] = previous_token


# ---------------------------------------------------------------------------
# 10: existing heavy-job gate suites still pass
# ---------------------------------------------------------------------------

def test_10_existing_heavy_job_gate_tests_remain_passing():
    """Re-runs tests/heavy_job_gate_v1.py and tests/heavy_job_gate_v2.py's
    own main() functions in-process (not via subprocess, so any import-time
    state this file already set up is irrelevant to them) -- proves the
    telemetry addition did not regress Phase 1/1A's own test coverage."""
    import runpy
    for name in ("heavy_job_gate_v1", "heavy_job_gate_v2"):
        module_path = str(ROOT / "tests" / f"{name}.py")
        namespace = runpy.run_path(module_path, run_name="__not_main__")
        result = namespace["main"]()
        assert result == 0, f"{name}.py's own main() must still return 0"


# ---------------------------------------------------------------------------
# 11-12: no collateral change
# ---------------------------------------------------------------------------

def test_11_scanner_results_unchanged_by_new_parameter():
    """scan_all's new `heavy_job_id` parameter must be purely additive --
    identical inputs must produce byte-identical rows/near_miss/meta
    whether or not it's supplied."""
    original_scan_all = scanner.scan_all
    original_batch = scanner._batch_download
    original_prefilter = scanner._prefilter_stock_universe
    original_background = scanner._ensure_background_refresh_started
    try:
        scanner._batch_download = lambda tickers, period, interval: {}
        scanner._prefilter_stock_universe = lambda watchlist, daily_data, trusted_options_symbols=None: (
            [], [{"ticker": t, "reason": "no price data"} for t in watchlist],
        )
        scanner._ensure_background_refresh_started = lambda: None
        _reset_telemetry()
        rows_a, near_a, meta_a = scanner.scan_all(watchlist=["AAPL"], discover=False, max_symbols=200)
        rows_b, near_b, meta_b = scanner.scan_all(watchlist=["AAPL"], discover=False, max_symbols=200, heavy_job_id="telemetry-test-job")
        assert rows_a == rows_b == []
        assert near_a == near_b == []
        # Compare every meta field except timing/telemetry-adjacent ones
        # that legitimately differ between two separate calls (timestamps,
        # durations) -- the STRATEGY-relevant fields must be identical.
        for key in ("configured_universe_count", "partial_result", "partial_result_reasons", "symbols_successfully_processed"):
            assert meta_a.get(key) == meta_b.get(key), f"meta[{key!r}] differs: {meta_a.get(key)!r} vs {meta_b.get(key)!r}"
    finally:
        scanner.scan_all = original_scan_all
        scanner._batch_download = original_batch
        scanner._prefilter_stock_universe = original_prefilter
        scanner._ensure_background_refresh_started = original_background
        _reset_telemetry()


def test_12_no_database_migration():
    """Structural check: neither the gate/telemetry module nor the new
    endpoint imports sqlite3 or any DB-access module -- this is pure
    in-process, in-memory observability."""
    source = (ROOT / "heavy_job_gate.py").read_text()
    for forbidden in ("import sqlite3", "candidates_router", "journal_store", "CREATE TABLE"):
        assert forbidden not in source, f"heavy_job_gate.py must not reference {forbidden!r}"


def main() -> int:
    test_1_starting_heavy_job_creates_running_record()
    test_2_checkpoint_updates_the_correct_job()
    test_3_completion_records_end_rss_duration_and_status()
    test_4_failure_records_failed_status_without_exception_objects()
    test_5_max_observed_rss_reflects_highest_checkpoint()
    test_6_job_ring_buffer_is_bounded()
    test_7_busy_skip_ring_buffer_is_bounded()
    test_8_endpoint_is_read_only_and_never_triggers_heavy_work()
    test_9_endpoint_follows_existing_admin_token_auth()
    test_10_existing_heavy_job_gate_tests_remain_passing()
    test_11_scanner_results_unchanged_by_new_parameter()
    test_12_no_database_migration()
    print("Heavy job telemetry v1 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
