#!/usr/bin/env python3
"""Regression tests for the process-wide heavy-job gate (2026-09 session,
Railway OOM Phase 1 fix) -- see heavy_job_gate.py's own module docstring
for the root cause this contains: multiple independently-scheduled heavy
scan/discovery jobs (scan_all, build_ranked_discovery_universe,
scan_ma_pipeline_candidates, the momentum-short-lifecycle whole-universe
ingestion) were able to run concurrently, each holding a full watchlist's
OHLCV data in memory at once.

Two layers of test, deliberately different in style:
  - GATE-level tests (1-8) call heavy_job_gate directly -- this is the new
    primitive itself, and every invariant it must uphold (single-flight
    across DIFFERENT job types/keys, non-blocking, finally-safe release)
    is provable in isolation without touching scanner.py/main.py at all.
  - INTEGRATION-level tests (9-10, plus re-proving 6/7/3 against the real
    wiring) drive scanner._refresh_analysis_cache (the real scan_all
    entrypoint) and main._run_discovery_universe_job (the real discovery
    entrypoint) with their own heavy internals monkeypatched, proving the
    gate is actually wired into production code, not just correct as a
    standalone primitive.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import heavy_job_gate  # noqa: E402
import scanner  # noqa: E402


def _reset_gate():
    """Tests must never leak a held gate into the next test -- release
    unconditionally (safe/no-op if nothing is held, see release()'s own
    docstring) before AND after every test."""
    heavy_job_gate.release()


# ---------------------------------------------------------------------------
# 1-8: pure gate-primitive tests
# ---------------------------------------------------------------------------

def test_1_heavy_scan_a_running_blocks_heavy_scan_b():
    _reset_gate()
    try:
        assert heavy_job_gate.try_acquire("scan_all", key="default") is True
        assert heavy_job_gate.try_acquire("discovery_universe", key="weekly") is False, (
            "a second heavy job must not be able to acquire the gate while the first holds it"
        )
    finally:
        _reset_gate()


def test_2_different_cache_keys_still_cannot_overlap():
    """Same job_type, two DIFFERENT cache keys -- cross-key exclusion is
    the whole point of this gate (same-key exclusion already existed via
    scanner._submit_analysis_refresh's own guard, see test 8)."""
    _reset_gate()
    try:
        assert heavy_job_gate.try_acquire("scan_all", key=("universe", "discovered")) is True
        assert heavy_job_gate.try_acquire("scan_all", key=("default",)) is False
    finally:
        _reset_gate()


def test_3_scan_all_and_discovery_cannot_overlap():
    _reset_gate()
    try:
        assert heavy_job_gate.try_acquire("scan_all", key="default") is True
        assert heavy_job_gate.try_acquire("discovery_universe", key="job-1") is False
    finally:
        _reset_gate()


def test_4_busy_trigger_is_skipped_not_queued():
    """A busy try_acquire must return immediately -- never block waiting
    for the holder to finish (that would let waiting triggers accumulate
    into a backlog, exactly what this gate must prevent)."""
    _reset_gate()
    try:
        assert heavy_job_gate.try_acquire("scan_all", key="default") is True
        start = time.monotonic()
        acquired = heavy_job_gate.try_acquire("ma_pipeline", key="scheduled:09:45")
        elapsed = time.monotonic() - start
        assert acquired is False
        assert elapsed < 0.05, "try_acquire must never block waiting for the gate to free up"
    finally:
        _reset_gate()


def test_5_gate_releases_after_successful_job():
    _reset_gate()
    try:
        assert heavy_job_gate.try_acquire("scan_all", key="default") is True
        heavy_job_gate.release()
        assert heavy_job_gate.try_acquire("discovery_universe", key="job-2") is True
    finally:
        _reset_gate()


def test_6_gate_releases_after_exception():
    _reset_gate()
    try:
        assert heavy_job_gate.try_acquire("scan_all", key="default") is True
        try:
            try:
                raise RuntimeError("simulated heavy-job failure")
            finally:
                heavy_job_gate.release()
        except RuntimeError:
            pass
        assert heavy_job_gate.active_snapshot() is None, "gate must be free after an exception + finally-release"
    finally:
        _reset_gate()


def test_7_next_job_can_run_after_exception():
    _reset_gate()
    try:
        assert heavy_job_gate.try_acquire("scan_all", key="default") is True
        try:
            raise ValueError("simulated failure")
        except ValueError:
            pass
        finally:
            heavy_job_gate.release()
        assert heavy_job_gate.try_acquire("ma_pipeline", key="scheduled:09:45") is True, (
            "one failing heavy job must never leave the gate permanently busy"
        )
    finally:
        _reset_gate()


def test_8_existing_same_key_dedupe_still_works():
    """scanner._submit_analysis_refresh's own pre-existing same-key guard
    (independent of the new cross-job gate) must be completely unaffected
    by this change -- a second submission for the SAME key while the first
    is still marked active must still be rejected, exactly as before."""
    key = ("default",)
    state_key = scanner._analysis_state_key(key)
    original_state = dict(scanner._analysis_refresh_state)
    original_jobs = set(scanner._background_jobs)
    try:
        scanner._analysis_refresh_state.pop(state_key, None)
        scanner._background_jobs.add(scanner._analysis_refresh_key(key))
        submitted = scanner._submit_analysis_refresh(key, None, reason="test")
        assert submitted is False, "same-key resubmission while already active must still be rejected"
    finally:
        scanner._analysis_refresh_state.clear()
        scanner._analysis_refresh_state.update(original_state)
        scanner._background_jobs.clear()
        scanner._background_jobs.update(original_jobs)


# ---------------------------------------------------------------------------
# 9-10 (+ re-proving 6/7/3): integration tests against the real wiring
# ---------------------------------------------------------------------------

def _patch_scan_all(monkeypatch_calls, rows=None, near_miss=None, meta=None, raise_exc=None):
    def _fake_scan_all(watchlist, max_workers=12, discover=False, max_symbols=200, trusted_options_symbols=None, heavy_job_id=None):
        monkeypatch_calls.append({
            "watchlist": watchlist, "max_workers": max_workers, "discover": discover, "max_symbols": max_symbols,
        })
        if raise_exc is not None:
            raise raise_exc
        return rows or [], near_miss or [], meta or {"symbols_requested": 0}
    return _fake_scan_all


def test_9_lightweight_job_unaffected_by_heavy_gate_being_held():
    """A single-symbol background job (the same _submit_background_job
    path earnings/option-expirations/best-contract refreshes all use)
    must complete normally even while the heavy gate is held -- it never
    touches heavy_job_gate at all, by design (see this module's own
    docstring: 'Do not gate lightweight price refreshes unnecessarily')."""
    _reset_gate()
    ran = {"value": False}

    def _light_job():
        ran["value"] = True

    try:
        assert heavy_job_gate.try_acquire("scan_all", key="default") is True
        key = ("light_job_test", time.time())
        submitted = scanner._submit_background_job(key, _light_job)
        assert submitted is True
        deadline = time.monotonic() + 2.0
        while not ran["value"] and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ran["value"] is True, "a lightweight background job must run to completion regardless of the heavy gate"
    finally:
        _reset_gate()


def test_10_results_unchanged_when_scan_runs_alone():
    """With the gate free, _refresh_analysis_cache's success-path
    behavior (scan_all is called, results land in _analysis_cache exactly
    as before this fix) must be byte-for-byte the same as pre-gate
    behavior -- the gate must add zero observable difference to an
    uncontended run. NOTE (Phase 1A): gate admission now happens in
    _submit_analysis_refresh, not here -- this test passes its own
    reservation directly to exercise _refresh_analysis_cache's release-of-
    a-given-token behavior in isolation, mirroring exactly what
    _submit_analysis_refresh does in production."""
    _reset_gate()
    original_scan_all = scanner.scan_all
    original_cache = dict(scanner._analysis_cache)
    calls = []
    key = ("default",)
    fake_rows = [{"ticker": "AAPL", "setup_status": "QUALIFIED", "quality": {"score": 1}}]
    fake_near_miss = [{"ticker": "MSFT", "setup_status": "WATCH", "quality": {"score": 0.5}}]
    fake_meta = {"symbols_requested": 2}
    try:
        scanner.scan_all = _patch_scan_all(calls, rows=fake_rows, near_miss=fake_near_miss, meta=fake_meta)
        reservation = heavy_job_gate.try_reserve("scan_all", key="default")
        assert reservation is not None
        scanner._refresh_analysis_cache(key, None, job_id="test-job", heavy_reservation=reservation)
        assert len(calls) == 1, "scan_all must be called exactly once for an uncontended run"
        cached = scanner._analysis_cache.get(key)
        assert cached is not None, "a successful, uncontended run must populate the analysis cache exactly as before"
        assert [r["ticker"] for r in cached["rows"]] == ["AAPL"]
        assert [r["ticker"] for r in cached["near_miss"]] == ["MSFT"]
        assert heavy_job_gate.active_snapshot() is None, "gate must be released after a successful run"
    finally:
        scanner.scan_all = original_scan_all
        scanner._analysis_cache.clear()
        scanner._analysis_cache.update(original_cache)
        _reset_gate()


def test_refresh_analysis_cache_skips_when_gate_busy():
    """Re-proves case 1/3 against the REAL admission point (Phase 1A moved
    this from _refresh_analysis_cache to _submit_analysis_refresh -- see
    tests/heavy_job_gate_v2.py for the full admission-path suite): scan_all
    must never even be CALLED when another heavy job already holds the
    gate, and the skip must be silent (no exception) so the caller's own
    tick/loop continues normally."""
    _reset_gate()
    original_scan_all = scanner.scan_all
    calls = []
    try:
        assert heavy_job_gate.try_acquire("discovery_universe", key="occupied") is True
        scanner.scan_all = _patch_scan_all(calls)
        submitted = scanner._submit_analysis_refresh(("default",), None, reason="test-busy")
        assert submitted is False
        assert calls == [], "scan_all must never be invoked while a different heavy job holds the gate"
    finally:
        scanner.scan_all = original_scan_all
        _reset_gate()


def test_refresh_analysis_cache_releases_gate_after_exception():
    """Re-proves case 6/7 against the real scan_all entrypoint: an
    exception raised inside scan_all itself must still release the
    reservation it was handed (via _refresh_analysis_cache's own finally)
    and must still propagate to the caller (scanner._submit_background_job's
    own except clause is what actually swallows it in production -- this
    test calls the function directly, so the exception must surface here)."""
    _reset_gate()
    original_scan_all = scanner.scan_all
    calls = []
    try:
        scanner.scan_all = _patch_scan_all(calls, raise_exc=RuntimeError("simulated scan_all failure"))
        reservation = heavy_job_gate.try_reserve("scan_all", key="default")
        assert reservation is not None
        raised = False
        try:
            scanner._refresh_analysis_cache(("default",), None, job_id="test-job-fail", heavy_reservation=reservation)
        except RuntimeError:
            raised = True
        assert raised, "the original exception must still propagate -- the gate must not swallow it"
        assert heavy_job_gate.active_snapshot() is None, "gate must be released even when scan_all raises"
        assert heavy_job_gate.try_acquire("scan_all", key="default") is True, (
            "a subsequent job must be able to acquire the gate after a failed run"
        )
    finally:
        scanner.scan_all = original_scan_all
        _reset_gate()


def test_discovery_job_skipped_when_scan_all_holds_gate():
    """Cross-module proof for case 3: main._run_discovery_universe_job
    (the real discovery entrypoint) must also skip -- never even call
    build_ranked_discovery_universe -- when scanner.py's scan_all path
    already holds the SAME shared gate. Imported lazily here (main.py
    pulls in the full FastAPI app + its startup-time module state) so
    this file's other tests never pay that import cost unless this one
    specific test runs."""
    import main  # noqa: PLC0415 -- deliberately lazy, see docstring above

    _reset_gate()
    original_build = main.build_ranked_discovery_universe
    original_cache = dict(main._discovery_universe_cache)
    calls = []

    def _fake_build(*args, **kwargs):
        calls.append(True)
        raise AssertionError("build_ranked_discovery_universe must not be called while scan_all holds the gate")

    try:
        assert heavy_job_gate.try_acquire("scan_all", key="default") is True
        main.build_ranked_discovery_universe = _fake_build
        main._discovery_universe_cache["running"] = True  # mirrors _submit_discovery_universe_job's own pre-set state
        main._run_discovery_universe_job("job-cross-module-test", refresh_reason="test")
        assert calls == [], "the heavy discovery pipeline must never run while scan_all's gate is held"
        assert main._discovery_universe_cache.get("running") is False, (
            "a busy-skip must reset 'running' so the next hourly watchdog tick can retry"
        )
    finally:
        main.build_ranked_discovery_universe = original_build
        main._discovery_universe_cache.clear()
        main._discovery_universe_cache.update(original_cache)
        _reset_gate()


def main() -> int:
    test_1_heavy_scan_a_running_blocks_heavy_scan_b()
    test_2_different_cache_keys_still_cannot_overlap()
    test_3_scan_all_and_discovery_cannot_overlap()
    test_4_busy_trigger_is_skipped_not_queued()
    test_5_gate_releases_after_successful_job()
    test_6_gate_releases_after_exception()
    test_7_next_job_can_run_after_exception()
    test_8_existing_same_key_dedupe_still_works()
    test_9_lightweight_job_unaffected_by_heavy_gate_being_held()
    test_10_results_unchanged_when_scan_runs_alone()
    test_refresh_analysis_cache_skips_when_gate_busy()
    test_refresh_analysis_cache_releases_gate_after_exception()
    test_discovery_job_skipped_when_scan_all_holds_gate()
    print("Heavy job gate v1 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
