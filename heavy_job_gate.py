"""Process-wide single-flight gate for heavyweight scanner/discovery
pipelines (2026-09 session, Railway OOM Phase 1 fix).

Root cause this exists to contain (see the 2026-09-12 Railway diagnosis):
a production runtime OOM traced to multiple independently-scheduled heavy
jobs -- scan_all() (scanner.py), build_ranked_discovery_universe()
(discovery.py, via main._run_discovery_universe_job), the M&A pipeline scan
(ma_pipeline.scan_ma_pipeline_candidates, via main._run_ma_pipeline_ingestion),
and the momentum-short-lifecycle whole-universe daily ingestion
(main._momentum_short_lifecycle_ingest) -- each independently holding a
full watchlist/universe's OHLCV data (up to 1000 symbols, multiple
timeframes) in memory at once, with nothing preventing two or more of them
from running concurrently. Same-cache-key de-duplication already existed
for scan_all's own callers (scanner._submit_analysis_refresh) and for the
discovery/momentum jobs' own single-worker executors, but nothing prevented
DIFFERENT job types (or different cache keys within the same job type) from
overlapping -- that cross-job overlap, landing at the moment of peak
concurrent per-ticker processing, is the confirmed OOM mechanism.

This module is intentionally tiny and dependency-free: a single process-wide
lock (this is a single uvicorn process, no ASGI worker multiplication -- see
the diagnosis report -- so in-process state is the correct scope, not a
DB-backed or file-backed lock) that at most one heavy job may hold at a time,
plus a stdlib-only current-RSS helper for the telemetry this same fix adds
around every heavy job.

Two admission styles are supported, both backed by the exact same
underlying single-holder state -- a reservation made through one API is
fully visible to (and blocks) the other API, and vice versa:

1. Same-function acquire/release (main._run_discovery_universe_job,
   main._run_ma_pipeline_ingestion, main._momentum_short_lifecycle_ingest):
   these are already, individually, self-serialized against THEMSELVES
   before this gate is ever reached (a "running" flag checked at their own
   submission point, or -- for ma_pipeline -- no executor/queue at all, a
   direct synchronous call), so there is no pre-gate-queueing concern to
   design around for them. They acquire and release within one function
   body:

       if not heavy_job_gate.try_acquire("discovery_universe", key=job_id):
           logger.info(heavy_job_gate.skipped_busy_message("discovery_universe", job_id))
           return  # never queue/backlog -- the next scheduled cycle tries again
       try:
           ... the actual heavy work ...
       finally:
           heavy_job_gate.release()

2. Reserve-then-submit (scanner._submit_analysis_refresh /
   scanner._refresh_analysis_cache -- Phase 1A, 2026-09 session): scan_all's
   own path is different in a way that made style 1 insufficient: MULTIPLE
   DIFFERENT cache keys can each independently pass scanner._background_jobs'
   own same-key dedupe and get submitted to the SHARED, multi-worker
   `_background_executor` (max_workers=3, also used by ~10 lightweight
   single-symbol job types) before any of them ever reached a style-1 gate
   check placed inside the worker function itself -- Python's
   ThreadPoolExecutor work queue is an unbounded queue.SimpleQueue with no
   backpressure, and the worker additionally busy-waits on
   scanner._scan_is_active() before doing anything else, so a queued
   heavy refresh could occupy one of only 3 shared workers (starving
   lightweight jobs) for an entire other scan's duration before ever
   discovering the gate was busy. Style 2 closes that gap by making the
   SUBMITTER (not the worker) atomically reserve the slot BEFORE
   `_background_executor.submit()` is even called, and hands the
   resulting token to the worker to release -- the worker never calls
   try_acquire()/try_reserve() itself, so the same non-reentrant gate is
   never independently touched by both ends of one job:

       reservation = heavy_job_gate.try_reserve("scan_all", key=state_key)
       if reservation is None:
           logger.info(heavy_job_gate.skipped_busy_message("scan_all", state_key))
           return False  # executor.submit() is never called -- zero queue footprint
       submitted = _submit_background_job(key, worker_fn, ..., _heavy_reservation=reservation)
       if not submitted:
           reservation.release()  # executor rejected the submission itself
       ...
       # inside worker_fn, in a finally:
       reservation.release()

Deliberately NOT a blocking context manager anywhere in this module: every
caller is either a periodic background trigger (which must skip-and-retry-
later, never block its own scheduling thread waiting for someone else's
scan to finish) or an async-submitted job (which must never grow an
unbounded backlog of waiters). Explicit try_acquire/try_reserve calls make
that skip decision visible at every call site rather than hiding it behind
a context manager some future caller could accidentally treat as blocking.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_active: "Optional[HeavyJobReservation]" = None


class HeavyJobReservation:
    """An atomic, single-use reservation of the global heavy-job slot,
    returned by `try_reserve()`. This is the ONE thing that actually holds
    the gate -- `try_acquire()`/`release()` below are thin, fully
    interoperable convenience wrappers around the same mechanism, not a
    second, separate gate.

    Exactly-once release semantics (Phase 1A safety requirement): `release()`
    is idempotent -- safe to call more than once (e.g. once from an
    executor.submit()-failure handler and again from the worker's own
    `finally`, in whichever order actually happens for a given job), and
    safe to call on a token that no longer holds the gate (a newer
    reservation already replaced it) -- it only ever clears `_active` if
    THIS token is still the current holder, so one job's cleanup can never
    clobber a different job's now-active reservation."""

    __slots__ = ("job_type", "key", "started_monotonic", "_released")

    def __init__(self, job_type: str, key: Any):
        self.job_type = job_type
        self.key = key
        self.started_monotonic = time.monotonic()
        self._released = False

    def release(self) -> None:
        global _active
        with _lock:
            if self._released:
                return
            self._released = True
            if _active is self:
                _active = None

    @property
    def active_duration_s(self) -> float:
        return round(time.monotonic() - self.started_monotonic, 1)


def try_reserve(job_type: str, key: Any = None) -> "Optional[HeavyJobReservation]":
    """Atomic check-and-set admission (Phase 1A): returns a new
    HeavyJobReservation -- and makes it the current holder, in the SAME
    locked section as the check -- iff no reservation is currently held;
    returns None immediately otherwise. There is no separate 'is it busy'
    step for a caller to race against between checking and reserving; the
    check and the reservation happen atomically under `_lock`, which is
    exactly the "avoid check-then-act" requirement this exists to satisfy."""
    global _active
    with _lock:
        if _active is not None:
            return None
        token = HeavyJobReservation(job_type, key)
        _active = token
        return token


def try_acquire(job_type: str, key: Any = None) -> bool:
    """Style-1 convenience (see module docstring): acquire-and-release
    within one function body. Internally just `try_reserve()` with the
    token discarded -- callers that never need to hand the reservation to
    a different function (no submitter/worker split) don't have to touch
    HeavyJobReservation at all. Returns True/False exactly as before this
    refactor; behavior and callers are unchanged."""
    return try_reserve(job_type, key) is not None


def release() -> None:
    """Style-1 convenience: releases whatever reservation is currently
    active, if any. Always safe to call, including when nothing is held
    (defensive -- see HeavyJobReservation.release()'s own idempotency).
    Every style-1 caller only calls this after a True try_acquire(), in a
    `finally`; style-2 callers (scan_all's path) release their own token
    object directly instead of calling this module-level function."""
    with _lock:
        current = _active
    if current is not None:
        current.release()


def active_snapshot() -> Optional[dict]:
    """A dict snapshot of the current holder's info plus a freshly-computed
    `active_duration_s`, or None if the gate is free. Safe to call from any
    thread; never mutates state. Works identically regardless of whether
    the current holder was reserved via try_acquire() or try_reserve()."""
    with _lock:
        current = _active
        if current is None:
            return None
        job_type, key, started = current.job_type, current.key, current.started_monotonic
    return {
        "job_type": job_type, "key": key, "started_monotonic": started,
        "active_duration_s": round(time.monotonic() - started, 1),
    }


def skipped_busy_message(requested_job_type: str, requested_key: Any = None) -> str:
    """One consistent HEAVY_JOB_SKIPPED_BUSY log line, shared by every call
    site so the four heavy-job types never drift into four different log
    shapes. Callers pass this straight to their own module's `logger.info`
    so the log source still reflects where the skip actually happened."""
    busy = active_snapshot()
    return (
        "HEAVY_JOB_SKIPPED_BUSY requested_job_type=%s requested_key=%s "
        "active_job_type=%s active_key=%s active_duration_s=%s"
    ) % (
        requested_job_type, requested_key,
        busy.get("job_type") if busy else None,
        busy.get("key") if busy else None,
        busy.get("active_duration_s") if busy else None,
    )


def current_rss_mb() -> Optional[float]:
    """Best-effort CURRENT process RSS in MB, stdlib-only (no psutil or
    other dependency added for this).

    On Linux (Railway's actual runtime) this reads /proc/self/status'
    VmRSS line directly -- a real, current (not peak) resident-set size,
    from one cheap small-file read.

    Anywhere /proc is unavailable (local dev on macOS, this repo's own
    test runs) this falls back to resource.getrusage(RUSAGE_SELF).ru_maxrss,
    which is PEAK (not current) RSS and platform-inconsistent in units
    (kilobytes on Linux, bytes on macOS) -- handled here, but still only a
    peak-so-far approximation, never mistaken for a Linux measurement by
    any caller since it's the same field name either way.

    Returns None (never raises, never fabricates a number) if neither
    source is readable."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return round(int(parts[1]) / 1024, 1)  # kB -> MB
    except (OSError, ValueError):
        pass
    try:
        import resource
        ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if ru_maxrss <= 0:
            return None
        divisor = (1024 * 1024) if sys.platform == "darwin" else 1024
        return round(ru_maxrss / divisor, 1)
    except Exception:
        return None
