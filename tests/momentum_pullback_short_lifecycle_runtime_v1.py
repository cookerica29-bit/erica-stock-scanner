#!/usr/bin/env python3
"""Runtime wiring tests for the SHORT lifecycle prospective experiment."""

from __future__ import annotations

import os
import tempfile
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import main
import momentum_pullback_shadow as mp
import momentum_pullback_short_lifecycle_experiment as exp


def _wait_future(future):
    return future.result(timeout=5)


def signal(**overrides):
    base = {
        "symbol": "ABC",
        "version": mp.VERSION,
        "direction": mp.SHORT,
        "state": mp.STATE_EXECUTION_CONFIRMED,
        "setup_id": "ABC|SHORT|2026-08-19|2026-08-20|momentum-pullback-shadow-v1.0",
        "pullback_generation_id": "ABC|SHORT|2026-08-19|momentum-pullback-shadow-v1.0",
        "signal_generation_id": "ABC|SHORT|2026-08-19|momentum-pullback-shadow-v1.0|2026-08-20",
        "signal_timestamp": "2026-08-20T00:00:00",
        "signal_close": 101.0,
        "entry_timestamp": None,
        "entry_price": None,
        "entry_status": mp.STATE_ENTRY_UNAVAILABLE,
        "atr14": 4.0,
        "ema_interaction_timestamp": "2026-08-19T00:00:00",
    }
    base.update(overrides)
    return base


def intraday(rows):
    return pd.DataFrame(rows, index=pd.date_range("2026-08-21T13:30:00", periods=len(rows), freq="30min"))


def daily_with_next_open():
    return pd.DataFrame(
        [
            {"Open": 101.0, "High": 102.0, "Low": 99.0, "Close": 101.0, "Volume": 1000},
            {"Open": 100.0, "High": 100.5, "Low": 98.5, "Close": 99.0, "Volume": 1000},
        ],
        index=pd.to_datetime(["2026-08-20T00:00:00", "2026-08-21T00:00:00"]),
    )


def reset_runtime_state():
    with main._momentum_short_lifecycle_lock:
        main._momentum_short_lifecycle_state["ingestion"].update({
            "running": False,
            "signals_evaluated": 0,
            "newly_captured": 0,
            "duplicates_skipped": 0,
            "errors": [],
        })
        main._momentum_short_lifecycle_state["watcher"].update({
            "running": False,
            "records_checked": 0,
            "records_changed": 0,
            "intraday_fetch_failures": 0,
            "errors": [],
        })
        main._momentum_short_lifecycle_state["last_error"] = None


def with_temp_ledger(fn):
    def wrapper():
        previous_path = exp.DEFAULT_LEDGER_PATH
        previous_env = os.environ.get("JOURNAL_ADMIN_TOKEN")
        with tempfile.TemporaryDirectory() as temp_dir:
            exp.DEFAULT_LEDGER_PATH = str(Path(temp_dir) / "ledger.json")
            os.environ["JOURNAL_ADMIN_TOKEN"] = "test-token"
            reset_runtime_state()
            try:
                fn()
            finally:
                exp.DEFAULT_LEDGER_PATH = previous_path
                if previous_env is None:
                    os.environ.pop("JOURNAL_ADMIN_TOKEN", None)
                else:
                    os.environ["JOURNAL_ADMIN_TOKEN"] = previous_env
                reset_runtime_state()
    return wrapper


@with_temp_ledger
def test_post_freeze_signal_captured_automatically_and_duplicate_skipped():
    previous_fetch = main._momentum_short_lifecycle_fetch_daily
    previous_replay = main.momentum_pullback.replay_symbol
    try:
        main._momentum_short_lifecycle_fetch_daily = lambda symbols: {"ABC": pd.DataFrame(), "SPY": pd.DataFrame()}
        main.momentum_pullback.replay_symbol = lambda symbol, df, spy: {"signals": [signal()]} if symbol == "ABC" else {"signals": []}
        metrics = main._momentum_short_lifecycle_ingest(["ABC"], "test")
        assert metrics["newly_captured"] == 1
        metrics2 = main._momentum_short_lifecycle_ingest(["ABC"], "test")
        assert metrics2["duplicates_skipped"] == 1
        ledger = exp.load_ledger()
        assert len(ledger["records"]) == 1
        assert ledger["records"][0]["state"] == exp.STATE_ENTRY_PENDING
    finally:
        main._momentum_short_lifecycle_fetch_daily = previous_fetch
        main.momentum_pullback.replay_symbol = previous_replay


@with_temp_ledger
def test_pre_freeze_signal_rejected_by_runtime_ingestion():
    previous_fetch = main._momentum_short_lifecycle_fetch_daily
    previous_replay = main.momentum_pullback.replay_symbol
    try:
        main._momentum_short_lifecycle_fetch_daily = lambda symbols: {"ABC": pd.DataFrame(), "SPY": pd.DataFrame()}
        main.momentum_pullback.replay_symbol = (
            lambda symbol, df, spy: {"signals": [signal(signal_timestamp="2026-08-18T00:00:00")]} if symbol == "ABC" else {"signals": []}
        )
        metrics = main._momentum_short_lifecycle_ingest(["ABC"], "test")
        assert metrics["signals_evaluated"] == 1
        assert metrics["newly_captured"] == 0
        assert len(exp.load_ledger()["records"]) == 0
    finally:
        main._momentum_short_lifecycle_fetch_daily = previous_fetch
        main.momentum_pullback.replay_symbol = previous_replay


@with_temp_ledger
def test_restart_does_not_duplicate_setup():
    ledger = exp.new_ledger()
    ledger, status = exp.capture_signal(ledger, signal())
    assert status == "added"
    exp.save_ledger(ledger)
    loaded = exp.load_ledger()
    loaded, status = exp.capture_signal(loaded, signal())
    assert status == "duplicate"
    assert len(loaded["records"]) == 1


@with_temp_ledger
def test_next_session_entry_resolves_and_intraday_update_runs():
    ledger = exp.new_ledger()
    ledger, _ = exp.capture_signal(ledger, signal())
    exp.save_ledger(ledger)
    previous_daily = main._momentum_short_lifecycle_fetch_daily
    previous_intraday = main._momentum_short_lifecycle_fetch_intraday
    try:
        main._momentum_short_lifecycle_fetch_daily = lambda symbols: {"ABC": daily_with_next_open()}
        main._momentum_short_lifecycle_fetch_intraday = lambda symbol: (
            intraday([
                {"Open": 99.8, "High": 99.9, "Low": 98.0, "Close": 98.5},
                {"Open": 98.5, "High": 100.0, "Low": 98.2, "Close": 99.8},
                {"Open": 99.8, "High": 99.9, "Low": 96.0, "Close": 96.5},
                {"Open": 96.5, "High": 97.0, "Low": 91.0, "Close": 91.5},
                {"Open": 91.5, "High": 92.0, "Low": 87.0, "Close": 88.0},
            ]),
            {"provider": "test"},
        )
        metrics = main._momentum_short_lifecycle_watch_open_records("test")
        record = exp.load_ledger()["records"][0]
        assert metrics["records_checked"] == 1
        assert record["entry_price"] == 100.0
        assert record["state"] == exp.STATE_EXPERIMENT_COMPLETE
        assert record["test"]["outcome"] == exp.OUTCOME_BREAKEVEN
        assert record["control"]["outcome"] == exp.OUTCOME_TARGET_1R
        assert record["diagnostics"]["be_then_later_2r"]
        assert record["diagnostics"]["be_then_later_3r"]
    finally:
        main._momentum_short_lifecycle_fetch_daily = previous_daily
        main._momentum_short_lifecycle_fetch_intraday = previous_intraday


@with_temp_ledger
def test_weekend_without_next_session_does_not_create_entry():
    ledger = exp.new_ledger()
    ledger, _ = exp.capture_signal(ledger, signal())
    exp.save_ledger(ledger)
    previous_daily = main._momentum_short_lifecycle_fetch_daily
    try:
        main._momentum_short_lifecycle_fetch_daily = lambda symbols: {"ABC": pd.DataFrame(
            [{"Open": 101.0, "High": 102.0, "Low": 99.0, "Close": 101.0}],
            index=pd.to_datetime(["2026-08-20T00:00:00"]),
        )}
        metrics = main._momentum_short_lifecycle_watch_open_records("test")
        record = exp.load_ledger()["records"][0]
        assert metrics["records_checked"] == 1
        assert record["state"] == exp.STATE_ENTRY_PENDING
        assert record["entry_price"] is None
    finally:
        main._momentum_short_lifecycle_fetch_daily = previous_daily


@with_temp_ledger
def test_terminal_record_is_not_mutated_by_watcher():
    record = exp.evaluate_record_with_intraday(
        exp.create_record(signal(entry_timestamp="2026-08-21T13:30:00", entry_price=100.0)),
        intraday([{"Open": 99.5, "High": 99.8, "Low": 95.8, "Close": 96.2}]),
    )
    ledger = exp.new_ledger()
    ledger, _ = exp.merge_record(ledger, record)
    exp.save_ledger(ledger)
    previous_intraday = main._momentum_short_lifecycle_fetch_intraday
    try:
        main._momentum_short_lifecycle_fetch_intraday = lambda symbol: (_ for _ in ()).throw(AssertionError("terminal record must not fetch"))
        metrics = main._momentum_short_lifecycle_watch_open_records("test")
        assert metrics["records_checked"] == 0
        assert exp.load_ledger()["records"][0]["history"] == record["history"]
    finally:
        main._momentum_short_lifecycle_fetch_intraday = previous_intraday


@with_temp_ledger
def test_ambiguous_sequence_remains_ambiguous():
    ledger = exp.new_ledger()
    rec = exp.create_record(signal(entry_timestamp="2026-08-21T13:30:00", entry_price=100.0))
    rec = exp.evaluate_record_with_intraday(
        rec,
        intraday([{"Open": 100.0, "High": 100.5, "Low": 97.8, "Close": 98.0}]),
    )
    ledger, _ = exp.merge_record(ledger, rec)
    exp.save_ledger(ledger)
    metrics = main._momentum_short_lifecycle_watch_open_records("test")
    assert metrics["records_checked"] == 0
    assert exp.load_ledger()["records"][0]["state"] == exp.STATE_SEQUENCE_AMBIGUOUS


@with_temp_ledger
def test_missing_intraday_data_does_not_fabricate_outcome():
    ledger = exp.new_ledger()
    ledger, _ = exp.capture_signal(ledger, signal(entry_timestamp="2026-08-21T13:30:00", entry_price=100.0))
    exp.save_ledger(ledger)
    previous_intraday = main._momentum_short_lifecycle_fetch_intraday
    try:
        main._momentum_short_lifecycle_fetch_intraday = lambda symbol: (pd.DataFrame(), {"failure_reason": "no_candles"})
        metrics = main._momentum_short_lifecycle_watch_open_records("test")
        record = exp.load_ledger()["records"][0]
        assert metrics["intraday_fetch_failures"] == 1
        assert record["control"]["outcome"] == exp.OUTCOME_OPEN
        assert record["test"]["outcome"] == exp.OUTCOME_OPEN
    finally:
        main._momentum_short_lifecycle_fetch_intraday = previous_intraday


@with_temp_ledger
def test_shadow_ingestion_failure_does_not_fail_scan_path():
    previous_submit = main._submit_momentum_short_lifecycle_ingestion
    previous_scan = main.scan_cached
    try:
        main._submit_momentum_short_lifecycle_ingestion = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("shadow failed"))
        main.scan_cached = lambda *args, **kwargs: {"rows": [], "near_miss": [], "meta": {}}
        response = TestClient(main.app).get("/api/scan?tickers=ABC")
        assert response.status_code == 200
    finally:
        main._submit_momentum_short_lifecycle_ingestion = previous_submit
        main.scan_cached = previous_scan


@with_temp_ledger
def test_status_endpoint_reports_counts():
    ledger = exp.new_ledger()
    ledger, _ = exp.capture_signal(ledger, signal())
    exp.save_ledger(ledger)
    client = TestClient(main.app)
    response = client.get(
        "/api/dev/momentum-pullback-short-lifecycle-experiment",
        headers={"x-kairos-admin-token": "test-token"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["experiment_version"] == exp.EXPERIMENT_VERSION
    assert payload["captured_setups"] == 1
    assert payload["entry_pending"] == 1
    assert payload["target_sample"] == 500
    assert payload["interim_sample"] == 200


def run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")


if __name__ == "__main__":
    run_all()
