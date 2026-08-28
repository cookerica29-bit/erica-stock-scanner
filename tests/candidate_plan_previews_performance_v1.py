"""Three targeted fixes to /candidate-plan-previews' real ~45-85s slowness
(diagnosed 2026-08-28 via railway run profiling against live infra: ~97-98%
of that time was per-candidate network latency -- daily-bar and
option-contract fetches done one ticker at a time, sequentially; SQLite
writes and all 8 signal computations combined were negligible).

1. Batched daily-bar cache pre-warming before the per-candidate loop
   (_stale_plan_preview_candidates + the pre-warm block in
   _enriched_previews_for_candidates).
2. Non-blocking option-contract fetch (block_on_miss=False) for the bulk
   path only -- the single-candidate real-promotion path
   (update_candidate_status) is untouched and still blocks.
3. Single-flight coordination (_plan_preview_refresh_lock) so a concurrent
   request waits on an in-flight refresh instead of redoing it.

Scope: the wiring/behavior of these three fixes specifically. The 8 signal
computations' own correctness is covered by their own dedicated test files
(candidates_router_bos_v1.py, _macro_choch_v1.py, _sweep_rejection_v1.py,
_location_v1.py, _confluence_v1.py) -- none of that changed here.
"""

import os
import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _client():
    import candidates_router
    app = FastAPI()
    app.include_router(candidates_router.router)
    return TestClient(app)


def _daily_frame():
    index = pd.date_range("2026-01-01", periods=30, freq="D", tz="UTC")
    rows = []
    for i in range(30):
        close = 100.0 - i * 0.1
        rows.append({"Open": close + 0.2, "High": close + 0.5, "Low": close - 0.5,
                      "Close": close, "Volume": 1_000_000})
    rows[5]["Low"] = 99.0
    rows[10]["Low"] = 90.0
    rows[10]["Close"] = 91.0
    rows[10]["Open"] = 92.0
    rows[20]["High"] = 110.0
    rows[20]["Close"] = 109.0
    rows[20]["Open"] = 108.0
    return pd.DataFrame(rows, index=index)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", str(tmp_path / "candidates.db"))
    monkeypatch.setenv("KAIROS_SCANNER_API_KEY", "test-scanner-key")
    import candidates_router
    return candidates_router


def _seed_candidates(router, n, prefix="TICK", direction="long"):
    conn = sqlite3.connect(os.environ["KAIROS_CANDIDATES_DB"])
    conn.row_factory = sqlite3.Row
    router._initialize_candidates_schema(conn)
    for i in range(n):
        conn.execute(
            """INSERT INTO candidates (ticker, source, signal, entry_price, ema21_4h,
               daily_regime, confidence, sma50_daily, sma200_daily, status, scanned_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (f"{prefix}{i}", "ma_pipeline", direction, 100.0, 101.0,
             "bullish" if direction == "long" else "bearish", "high", 98.0, 95.0,
             "new", "2026-08-28T12:00:00Z", "2026-08-28T12:00:00Z"),
        )
    conn.commit()
    conn.close()


def _fake_batch_download_factory(call_log, delay=0.0):
    """Logs every call (for asserting on batching/dedup shape), and returns
    real-looking data for every requested ticker every time -- appropriate
    for tests that only care about CALL SHAPE (how many tickers per call),
    not about whether real network round-trips were actually avoided."""
    def fake(tickers, period, interval):
        call_log.append(tuple(sorted(str(t).upper() for t in tickers)))
        if delay:
            time.sleep(delay)
        return {str(t).upper(): _daily_frame() for t in tickers}
    return fake


# --- 1. Batched pre-warming ------------------------------------------------

def test_stale_candidates_trigger_one_batched_prewarm_call_not_one_per_ticker(env, monkeypatch):
    router = env
    _seed_candidates(router, 8, prefix="WARM")
    call_log = []
    monkeypatch.setattr(router, "_batch_download", _fake_batch_download_factory(call_log))
    monkeypatch.setattr(router, "_safe_option_contract_for_candidate", lambda *a, **k: None)
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {})

    client = _client()
    headers = {"X-API-Key": "test-scanner-key"}
    resp = client.get("/api/v1/scanner/candidate-plan-previews", headers=headers)
    assert resp.status_code == 200
    assert len(resp.json()) == 8

    # One pre-warm call carrying all 8 tickers together, PLUS each candidate's
    # own unchanged single-ticker call inside _compute_candidate_promotion
    # (that one is untouched by design -- pre-warming just makes it a cache
    # hit in the real code; this fake has no cache, so it still fires, which
    # is fine -- the assertion here is specifically that the FIRST call is
    # batched, not that the inner per-candidate call was removed).
    assert len(call_log) >= 1
    first_call = call_log[0]
    assert len(first_call) == 8, f"expected the pre-warm call to carry all 8 tickers at once, got {first_call}"


def test_prewarm_chunks_at_the_configured_size(env, monkeypatch):
    router = env
    n = router.CANDIDATE_PREVIEW_PREWARM_CHUNK_SIZE + 5
    _seed_candidates(router, n, prefix="CHK")
    call_log = []
    monkeypatch.setattr(router, "_batch_download", _fake_batch_download_factory(call_log))
    monkeypatch.setattr(router, "_safe_option_contract_for_candidate", lambda *a, **k: None)
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {})

    client = _client()
    headers = {"X-API-Key": "test-scanner-key"}
    resp = client.get("/api/v1/scanner/candidate-plan-previews", headers=headers)
    assert resp.status_code == 200
    assert len(resp.json()) == n

    prewarm_calls = [c for c in call_log if len(c) > 1]
    assert len(prewarm_calls) == 2, f"expected 2 chunked pre-warm calls for {n} tickers, got {len(prewarm_calls)}"
    assert len(prewarm_calls[0]) == router.CANDIDATE_PREVIEW_PREWARM_CHUNK_SIZE
    assert len(prewarm_calls[1]) == 5


def test_prewarm_failure_does_not_break_the_request(env, monkeypatch):
    """Best-effort only -- if the batched pre-warm call itself raises, the
    request must still succeed (each candidate's own existing per-ticker
    fetch inside _compute_candidate_promotion still runs and handles its own
    failure exactly as it always has)."""
    router = env
    _seed_candidates(router, 3, prefix="FAIL")

    def flaky_batch_download(tickers, period, interval):
        if len(tickers) > 1:
            raise RuntimeError("simulated provider outage during pre-warm")
        return {str(tickers[0]).upper(): _daily_frame()}

    monkeypatch.setattr(router, "_batch_download", flaky_batch_download)
    monkeypatch.setattr(router, "_safe_option_contract_for_candidate", lambda *a, **k: None)
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {})

    client = _client()
    headers = {"X-API-Key": "test-scanner-key"}
    resp = client.get("/api/v1/scanner/candidate-plan-previews", headers=headers)
    assert resp.status_code == 200
    assert len(resp.json()) == 3


# --- 2. Non-blocking option-contract fetch ---------------------------------

def test_bulk_preview_path_requests_non_blocking_contract_fetch(env, monkeypatch):
    router = env
    _seed_candidates(router, 1, prefix="OPT")
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {
        str(tickers[0]).upper(): _daily_frame()
    })
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {})

    seen_kwargs = {}
    def fake_contract(ticker, direction, entry_price, **kwargs):
        seen_kwargs.update(kwargs)
        return {"available": False, "execution": "Loading", "reason": "loading", "source": "loading"}
    monkeypatch.setattr(router, "_safe_option_contract_for_candidate", fake_contract)

    client = _client()
    headers = {"X-API-Key": "test-scanner-key"}
    resp = client.get("/api/v1/scanner/candidate-plan-previews", headers=headers)
    assert resp.status_code == 200
    assert seen_kwargs.get("block_on_miss") is False


def test_single_promotion_path_still_blocks_on_miss(env, monkeypatch):
    """The real-promotion path (PATCH /candidates/{ticker}, status=active)
    is untouched -- it's one explicit user action, not a 550-candidate
    sweep, and should still return a complete contract synchronously."""
    router = env
    _seed_candidates(router, 1, prefix="BLK")
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {
        str(tickers[0]).upper(): _daily_frame()
    })

    seen_kwargs = {"called": False}
    def fake_contract(ticker, direction, entry_price, **kwargs):
        seen_kwargs["called"] = True
        seen_kwargs.update(kwargs)
        return {"available": True, "execution": "Excellent", "strike": 100.0}
    monkeypatch.setattr(router, "_safe_option_contract_for_candidate", fake_contract)

    client = _client()
    headers = {"X-API-Key": "test-scanner-key"}
    # The contract fetch happens before _promotion_block_reason is even
    # checked -- whether this specific fake candidate clears every other
    # ENTER_NOW gate (regime/target/R:R/proximity/execution) isn't the
    # point of this test and isn't fully mocked here, so the response may
    # legitimately be a 422 from some later, unrelated check. What matters
    # is what block_on_miss was passed to the contract fetch itself.
    client.patch(
        "/api/v1/scanner/candidates/BLK0?source=ma_pipeline",
        headers=headers, json={"status": "active"},
    )
    assert seen_kwargs["called"] is True
    # block_on_miss not passed at all here -> defaults to True (the function's
    # own default), same as before this change existed.
    assert "block_on_miss" not in seen_kwargs or seen_kwargs["block_on_miss"] is True


def test_loading_placeholder_is_treated_as_transient_and_gets_rechecked(env):
    router = env
    row = {
        "option_contract_json": '{"available": false, "execution": "Loading", "reason": "loading", "source": "loading"}',
    }
    assert router._preview_has_transient_option_unavailable(row) is True


# --- 3. Single-flight coordination ------------------------------------------

def test_concurrent_requests_do_not_each_redo_the_full_computation(env, monkeypatch):
    """The exact bug confirmed by direct testing before this fix existed:
    firing 2 concurrent requests against the same cold 5-candidate set fired
    10 network calls, not 5 -- each request independently redid the full
    computation with zero awareness of the other in flight.

    Made deterministic (not just "usually passes"): request A's first
    _batch_download call blocks until request B has definitely been fired,
    guaranteeing A is still holding _plan_preview_refresh_lock when B
    arrives and tries to acquire it. B must then wait for A, re-check the
    (now fresh) cache, and do zero further fetching of its own.
    """
    router = env
    n = 5
    _seed_candidates(router, n, prefix="LOCK")
    call_log: list[tuple] = []
    call_lock = threading.Lock()
    request_b_fired = threading.Event()

    def fake_batch_download(tickers, period, interval):
        with call_lock:
            call_log.append(tuple(sorted(str(t).upper() for t in tickers)))
        # Only A's very first call (the pre-warm call, made while holding
        # the lock) needs to stall -- without this wait, A could finish
        # before B's thread even starts, and the test would trivially pass
        # without ever exercising the lock at all.
        request_b_fired.wait(timeout=2)
        return {str(t).upper(): _daily_frame() for t in tickers}

    monkeypatch.setattr(router, "_batch_download", fake_batch_download)
    monkeypatch.setattr(router, "_safe_option_contract_for_candidate", lambda *a, **k: None)
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {})

    client = _client()
    headers = {"X-API-Key": "test-scanner-key"}
    results = {}

    def request_a():
        results["a"] = client.get("/api/v1/scanner/candidate-plan-previews", headers=headers).status_code

    def request_b():
        # Give A a moment to actually enter the lock and block inside
        # fake_batch_download before B fires -- then release A.
        time.sleep(0.1)
        request_b_fired.set()
        results["b"] = client.get("/api/v1/scanner/candidate-plan-previews", headers=headers).status_code

    ta = threading.Thread(target=request_a)
    tb = threading.Thread(target=request_b)
    ta.start()
    tb.start()
    ta.join()
    tb.join()

    assert results["a"] == 200
    assert results["b"] == 200
    # A alone (5 candidates, none cached yet) does exactly 1 batched
    # pre-warm call (all 5 tickers) + 5 individual per-candidate calls
    # inside _compute_candidate_promotion (unchanged, always fires -- the
    # fake has no cache of its own, only the real scanner._batch_download
    # would turn these into cache hits) = 6 calls. B, arriving while A
    # holds the lock, should wait, find A's committed rows fresh on
    # re-check, and contribute ZERO calls of its own.
    assert len(call_log) == n + 1, (
        f"expected exactly {n + 1} total _batch_download calls (A's own work "
        f"only -- 1 batched pre-warm + {n} per-candidate calls), got "
        f"{len(call_log)}: B should have waited on A and found a warm cache, "
        f"not redone the computation"
    )


def test_lock_is_held_during_the_recompute_work(env, monkeypatch):
    """Direct proof the lock is real and actually held while the expensive
    work runs, not just present and unused: from inside a slow pre-warm
    call (which only happens while the lock is held), a non-blocking
    acquire attempt from the SAME thread must fail."""
    router = env
    _seed_candidates(router, 2, prefix="SER")
    observed = {"acquired_while_busy": None}

    def slow_batch_download(tickers, period, interval):
        observed["acquired_while_busy"] = router._plan_preview_refresh_lock.acquire(blocking=False)
        if observed["acquired_while_busy"]:
            router._plan_preview_refresh_lock.release()
        return {str(t).upper(): _daily_frame() for t in tickers}

    monkeypatch.setattr(router, "_batch_download", slow_batch_download)
    monkeypatch.setattr(router, "_safe_option_contract_for_candidate", lambda *a, **k: None)
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {})

    client = _client()
    headers = {"X-API-Key": "test-scanner-key"}
    resp = client.get("/api/v1/scanner/candidate-plan-previews", headers=headers)

    assert resp.status_code == 200
    assert observed["acquired_while_busy"] is False, (
        "the lock should already be held by this same request's own refresh "
        "work at the point _batch_download runs -- a reentrant acquire "
        "attempt reporting success would mean the lock isn't guarding this "
        "code path at all"
    )
