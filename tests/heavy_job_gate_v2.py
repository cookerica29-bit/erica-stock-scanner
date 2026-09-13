#!/usr/bin/env python3
"""Regression tests for Phase 1A of the heavy-job gate (2026-09 session,
Railway OOM fix): atomic admission BEFORE a heavy scan-analysis refresh is
ever submitted to `scanner._background_executor`.

Phase 1 (tests/heavy_job_gate_v1.py) put the gate check inside the WORKER
function (scanner._refresh_analysis_cache), which correctly capped
concurrent heavy EXECUTION at 1 but still let a busy heavy refresh reach
`_background_executor.submit()` first -- different cache keys pass
scanner._background_jobs' own same-key dedupe independently, so several
could be admitted to the shared, multi-worker executor (and sit
busy-waiting on scanner._scan_is_active() there) before any of them
actually reached the gate check. Phase 1A moves the reservation to the
SUBMITTER (scanner._submit_analysis_refresh), atomically, before
executor.submit() is ever called -- this file proves that admission-time
behavior specifically; tests/heavy_job_gate_v1.py's own suite (still
passing, updated only where the moved gate check required it) continues to
prove the worker-side release/telemetry/lifecycle behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import heavy_job_gate  # noqa: E402
import scanner  # noqa: E402


def _reset_gate():
    heavy_job_gate.release()


class _RecordingExecutor:
    """Stands in for scanner._background_executor. Runs the submitted
    callable SYNCHRONOUSLY on the calling thread (tests need no real
    threading/polling to observe outcomes) while recording exactly how
    many times submit() was actually called -- the one fact every
    admission-path test in this file cares about."""

    def __init__(self):
        self.submit_calls = 0

    def submit(self, fn, *args, **kwargs):
        self.submit_calls += 1
        fn(*args, **kwargs)
        return None


class _FailingExecutor:
    """Simulates _background_executor.submit() itself raising (e.g. the
    executor is shutting down) -- scanner._submit_background_job already
    catches exactly this (a bare RuntimeError) and returns False."""

    def submit(self, fn, *args, **kwargs):
        raise RuntimeError("executor is shutting down")


def _patch_scan_all(calls, rows=None, near_miss=None, meta=None, raise_exc=None):
    def _fake_scan_all(watchlist, max_workers=12, discover=False, max_symbols=200, trusted_options_symbols=None, heavy_job_id=None):
        calls.append(True)
        if raise_exc is not None:
            raise raise_exc
        return rows or [], near_miss or [], meta or {"symbols_requested": 0}
    return _fake_scan_all


# ---------------------------------------------------------------------------
# 1-2: busy gate -- executor.submit must never be called
# ---------------------------------------------------------------------------

def test_1_heavy_slot_occupied_blocks_submission():
    _reset_gate()
    original_executor = scanner._background_executor
    original_scan_all = scanner.scan_all
    fake_executor = _RecordingExecutor()
    calls = []
    try:
        assert heavy_job_gate.try_acquire("scan_all", key="default") is True
        scanner._background_executor = fake_executor
        scanner.scan_all = _patch_scan_all(calls)
        submitted = scanner._submit_analysis_refresh(("default",), None, reason="test")
        assert submitted is False
        assert fake_executor.submit_calls == 0, "executor.submit() must never be called while the heavy slot is occupied"
        assert calls == [], "scan_all must never run either"
    finally:
        scanner._background_executor = original_executor
        scanner.scan_all = original_scan_all
        _reset_gate()


def test_2_different_cache_key_while_busy_also_not_submitted():
    """The SAME concern as test 1, but proving it specifically for a
    DIFFERENT cache key than whatever (hypothetically) holds the slot --
    this is exactly the gap Phase 1A closes: scanner._background_jobs'
    own same-key dedupe would have let a different key straight through
    to the executor under Phase 1's design."""
    _reset_gate()
    original_executor = scanner._background_executor
    original_scan_all = scanner.scan_all
    fake_executor = _RecordingExecutor()
    calls = []
    try:
        assert heavy_job_gate.try_acquire("scan_all", key=("universe", "discovered")) is True
        scanner._background_executor = fake_executor
        scanner.scan_all = _patch_scan_all(calls)
        submitted = scanner._submit_analysis_refresh(("default",), None, reason="test")
        assert submitted is False
        assert fake_executor.submit_calls == 0
    finally:
        scanner._background_executor = original_executor
        scanner.scan_all = original_scan_all
        _reset_gate()


# ---------------------------------------------------------------------------
# 3-6: uncontended admission, and every release path
# ---------------------------------------------------------------------------

def test_3_successful_reservation_submits_exactly_one_future():
    _reset_gate()
    original_executor = scanner._background_executor
    original_scan_all = scanner.scan_all
    original_cache = dict(scanner._analysis_cache)
    fake_executor = _RecordingExecutor()
    calls = []
    try:
        scanner._background_executor = fake_executor
        scanner.scan_all = _patch_scan_all(calls)
        submitted = scanner._submit_analysis_refresh(("default",), None, reason="test")
        assert submitted is True
        assert fake_executor.submit_calls == 1, "exactly one future must be submitted for an uncontended request"
        assert len(calls) == 1
    finally:
        scanner._background_executor = original_executor
        scanner.scan_all = original_scan_all
        scanner._analysis_cache.clear()
        scanner._analysis_cache.update(original_cache)
        scanner._background_jobs.clear()
        _reset_gate()


def test_4_reservation_released_after_worker_completion():
    _reset_gate()
    original_executor = scanner._background_executor
    original_scan_all = scanner.scan_all
    original_cache = dict(scanner._analysis_cache)
    fake_executor = _RecordingExecutor()
    calls = []
    try:
        scanner._background_executor = fake_executor
        scanner.scan_all = _patch_scan_all(calls)
        submitted = scanner._submit_analysis_refresh(("default",), None, reason="test")
        assert submitted is True
        assert heavy_job_gate.active_snapshot() is None, "the reservation must be released once the worker completes"
    finally:
        scanner._background_executor = original_executor
        scanner.scan_all = original_scan_all
        scanner._analysis_cache.clear()
        scanner._analysis_cache.update(original_cache)
        scanner._background_jobs.clear()
        _reset_gate()


def test_5_reservation_released_after_worker_exception():
    _reset_gate()
    original_executor = scanner._background_executor
    original_scan_all = scanner.scan_all
    fake_executor = _RecordingExecutor()
    calls = []
    try:
        scanner._background_executor = fake_executor
        scanner.scan_all = _patch_scan_all(calls, raise_exc=RuntimeError("simulated scan_all failure"))
        # _submit_background_job's own _run() wrapper catches and logs any
        # exception fn() raises (see scanner.py) -- it must not propagate
        # out of submit() itself, and the reservation must still be freed.
        submitted = scanner._submit_analysis_refresh(("default",), None, reason="test")
        assert submitted is True, "submission itself succeeds; the failure happens inside the worker"
        assert heavy_job_gate.active_snapshot() is None, "the reservation must be released even when the worker raises"
    finally:
        scanner._background_executor = original_executor
        scanner.scan_all = original_scan_all
        scanner._background_jobs.clear()
        _reset_gate()


def test_6_reservation_released_when_executor_submit_itself_fails():
    _reset_gate()
    original_executor = scanner._background_executor
    try:
        scanner._background_executor = _FailingExecutor()
        submitted = scanner._submit_analysis_refresh(("default",), None, reason="test")
        assert submitted is False
        assert heavy_job_gate.active_snapshot() is None, (
            "a reservation whose executor.submit() call itself raised must still be released, "
            "not leaked -- otherwise one transient executor failure would permanently wedge the gate"
        )
    finally:
        scanner._background_executor = original_executor
        scanner._background_jobs.clear()
        _reset_gate()


def test_7_next_heavy_refresh_admitted_after_release():
    _reset_gate()
    original_executor = scanner._background_executor
    original_scan_all = scanner.scan_all
    original_cache = dict(scanner._analysis_cache)
    fake_executor = _RecordingExecutor()
    calls = []
    try:
        scanner._background_executor = fake_executor
        scanner.scan_all = _patch_scan_all(calls)
        first = scanner._submit_analysis_refresh(("default",), None, reason="first")
        assert first is True
        scanner._background_jobs.clear()  # simulate the first job's own key having already cleared
        second = scanner._submit_analysis_refresh(("universe", "discovered"), None, reason="second")
        assert second is True, "a fresh request must be admitted once the previous reservation released"
        assert fake_executor.submit_calls == 2
    finally:
        scanner._background_executor = original_executor
        scanner.scan_all = original_scan_all
        scanner._analysis_cache.clear()
        scanner._analysis_cache.update(original_cache)
        scanner._background_jobs.clear()
        _reset_gate()


# ---------------------------------------------------------------------------
# 8-9: pre-existing behavior this fix must not disturb
# ---------------------------------------------------------------------------

def test_8_existing_same_key_dedupe_remains_correct():
    """scanner._background_jobs' own same-key guard runs BEFORE the new
    heavy-slot reservation is even attempted -- a same-key resubmission
    must still be rejected exactly as before, and must never touch (or
    leak) a reservation at all."""
    key = ("default",)
    state_key = scanner._analysis_state_key(key)
    original_state = dict(scanner._analysis_refresh_state)
    original_jobs = set(scanner._background_jobs)
    _reset_gate()
    try:
        scanner._analysis_refresh_state.pop(state_key, None)
        scanner._background_jobs.add(scanner._analysis_refresh_key(key))
        submitted = scanner._submit_analysis_refresh(key, None, reason="test")
        assert submitted is False
        assert heavy_job_gate.active_snapshot() is None, "a same-key-dedupe rejection must never reserve the heavy slot"
    finally:
        scanner._analysis_refresh_state.clear()
        scanner._analysis_refresh_state.update(original_state)
        scanner._background_jobs.clear()
        scanner._background_jobs.update(original_jobs)
        _reset_gate()


def test_9_stale_cache_response_unchanged_while_heavy_slot_busy():
    """scan_cached() must still return whatever is already cached
    immediately, completely unaffected by the heavy slot being busy --
    the safe stale-response behavior scan_cached() already had must never
    be replaced by blocking or an overlapping heavy workload."""
    key = ("default",)
    original_cache = dict(scanner._analysis_cache)
    _reset_gate()
    try:
        scanner._analysis_cache[key] = {
            "rows": [{"ticker": "OLD", "setup_status": "QUALIFIED"}],
            "near_miss": [],
            "generated_at": scanner._utc_now(),
            "scan_meta": {},
        }
        assert heavy_job_gate.try_acquire("discovery_universe", key="occupied") is True
        result = scanner.scan_cached(None, force_refresh=False)
        assert [r["ticker"] for r in result["rows"]] == ["OLD"], (
            "a cached response must be returned as-is regardless of heavy-slot contention"
        )
    finally:
        scanner._analysis_cache.clear()
        scanner._analysis_cache.update(original_cache)
        _reset_gate()


# ---------------------------------------------------------------------------
# 10-11: cross-module exclusion, both directions
# ---------------------------------------------------------------------------

def test_10_discovery_owns_slot_scanner_refresh_never_enters_executor_queue():
    """Cross-module proof, admission-path direction: with main.py's
    discovery job holding the shared gate, scanner._submit_analysis_refresh
    must reject BEFORE ever calling _background_executor.submit()."""
    _reset_gate()
    original_executor = scanner._background_executor
    original_scan_all = scanner.scan_all
    fake_executor = _RecordingExecutor()
    calls = []
    try:
        assert heavy_job_gate.try_acquire("discovery_universe", key="weekly-job") is True
        scanner._background_executor = fake_executor
        scanner.scan_all = _patch_scan_all(calls)
        submitted = scanner._submit_analysis_refresh(("default",), None, reason="test")
        assert submitted is False
        assert fake_executor.submit_calls == 0
        assert calls == []
    finally:
        scanner._background_executor = original_executor
        scanner.scan_all = original_scan_all
        _reset_gate()


def test_11_scanner_owns_slot_discovery_ma_pipeline_momentum_all_skip():
    """Cross-module proof, the other direction: with scanner's scan_all
    path holding the shared gate, all three main.py heavy jobs must skip
    their own inner heavy call without ever invoking it."""
    import main  # noqa: PLC0415 -- lazy import, see tests/heavy_job_gate_v1.py's own note on why

    _reset_gate()
    original_build = main.build_ranked_discovery_universe
    original_scan_ma = main.scan_ma_pipeline_candidates
    original_fetch_daily = main._momentum_short_lifecycle_fetch_daily
    original_discovery_ready = main._discovery_symbols_ready
    original_discovery_cache = dict(main._discovery_universe_cache)

    def _boom(*args, **kwargs):
        raise AssertionError("this heavy inner call must not run while scan_all holds the gate")

    try:
        assert heavy_job_gate.try_acquire("scan_all", key="default") is True

        # discovery_universe
        main.build_ranked_discovery_universe = _boom
        main._discovery_universe_cache["running"] = True
        main._run_discovery_universe_job("job-1", refresh_reason="test")
        assert main._discovery_universe_cache.get("running") is False

        # ma_pipeline
        main.scan_ma_pipeline_candidates = _boom
        main._discovery_symbols_ready = lambda now=None: (True, ["AAPL", "MSFT"], {"status": "ready"})
        result = main._run_ma_pipeline_ingestion("test_reason")
        assert result.get("status") == "skipped_busy"

        # momentum_short_lifecycle_ingestion
        main._momentum_short_lifecycle_fetch_daily = _boom
        metrics = main._momentum_short_lifecycle_ingest(symbols=["AAPL"], reason="test")
        assert any(err.get("error") == "skipped_busy" for err in metrics.get("errors", []))
    finally:
        main.build_ranked_discovery_universe = original_build
        main.scan_ma_pipeline_candidates = original_scan_ma
        main._momentum_short_lifecycle_fetch_daily = original_fetch_daily
        main._discovery_symbols_ready = original_discovery_ready
        main._discovery_universe_cache.clear()
        main._discovery_universe_cache.update(original_discovery_cache)
        _reset_gate()


# ---------------------------------------------------------------------------
# 12: lightweight jobs on the same shared executor are unaffected
# ---------------------------------------------------------------------------

def test_12_lightweight_background_job_unaffected_by_admission_change():
    """A single-symbol background job (earnings/option-expirations/best-
    contract style) never touches heavy_job_gate at all and must continue
    to submit and run normally regardless of the heavy slot's state or of
    _background_executor being swapped out for this test file's fakes
    elsewhere -- this uses the REAL executor to prove that."""
    _reset_gate()
    ran = {"value": False}

    def _light_job():
        ran["value"] = True

    try:
        assert heavy_job_gate.try_acquire("scan_all", key="default") is True
        key = ("light_job_phase1a_test", id(_light_job))
        submitted = scanner._submit_background_job(key, _light_job)
        assert submitted is True
        import time as _time
        deadline = _time.monotonic() + 2.0
        while not ran["value"] and _time.monotonic() < deadline:
            _time.sleep(0.01)
        assert ran["value"] is True, "lightweight jobs must run regardless of heavy-slot state"
    finally:
        _reset_gate()


def main() -> int:
    test_1_heavy_slot_occupied_blocks_submission()
    test_2_different_cache_key_while_busy_also_not_submitted()
    test_3_successful_reservation_submits_exactly_one_future()
    test_4_reservation_released_after_worker_completion()
    test_5_reservation_released_after_worker_exception()
    test_6_reservation_released_when_executor_submit_itself_fails()
    test_7_next_heavy_refresh_admitted_after_release()
    test_8_existing_same_key_dedupe_remains_correct()
    test_9_stale_cache_response_unchanged_while_heavy_slot_busy()
    test_10_discovery_owns_slot_scanner_refresh_never_enters_executor_queue()
    test_11_scanner_owns_slot_discovery_ma_pipeline_momentum_all_skip()
    test_12_lightweight_background_job_unaffected_by_admission_change()
    print("Heavy job gate v2 (Phase 1A admission-path) tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
