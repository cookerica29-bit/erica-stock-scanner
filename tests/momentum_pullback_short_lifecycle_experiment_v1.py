#!/usr/bin/env python3
"""Provider-free tests for the SHORT lifecycle prospective experiment."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import momentum_pullback_shadow as mp
import momentum_pullback_short_lifecycle_experiment as exp


def _assert_close(actual, expected, tolerance=1e-9):
    assert abs(float(actual) - float(expected)) <= tolerance, (actual, expected)


def short_signal(**overrides):
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
        "entry_timestamp": "2026-08-21T13:30:00",
        "entry_price": 100.0,
        "entry_status": "NEXT_SESSION_OPEN",
        "atr14": 4.0,
        "ema_interaction_timestamp": "2026-08-19T00:00:00",
    }
    base.update(overrides)
    return base


def bars(rows, start="2026-08-21T13:30:00"):
    return pd.DataFrame(rows, index=pd.date_range(start, periods=len(rows), freq="30min"))


def test_short_atr_arithmetic():
    th = exp.short_thresholds(100.0, 4.0)
    _assert_close(th["favorable_plus_0_5_atr"], 98.0)
    _assert_close(th["target_plus_1_atr"], 96.0)
    _assert_close(th["diagnostic_plus_2_atr"], 92.0)
    _assert_close(th["diagnostic_plus_3_atr"], 88.0)
    _assert_close(th["control_stop_minus_1_atr"], 104.0)
    _assert_close(th["breakeven"], 100.0)


def test_stable_setup_identity_and_no_pre_freeze_setup_admitted():
    signal = short_signal()
    assert exp.experiment_setup_identity(signal) == exp.experiment_setup_identity(dict(signal))
    old = short_signal(signal_timestamp="2026-08-18T00:00:00")
    eligible, reason = exp.signal_is_eligible(old)
    assert not eligible
    assert reason == "pre_freeze_signal"


def test_non_short_rejected_and_missing_entry_is_pending():
    eligible, reason = exp.signal_is_eligible(short_signal(direction=mp.LONG))
    assert not eligible
    assert reason == "not_short"
    eligible, reason = exp.signal_is_eligible(short_signal(entry_timestamp=None, entry_price=None))
    assert eligible
    assert reason == "eligible"
    record = exp.create_record(short_signal(entry_timestamp=None, entry_price=None))
    assert record["state"] == exp.STATE_ENTRY_PENDING
    assert record["thresholds"] is None


def test_next_session_open_entry_is_recorded_and_option_missing_ok():
    record = exp.create_record(short_signal(), option_observation=None)
    assert record["entry_timestamp"] == "2026-08-21T13:30:00"
    assert record["entry_timestamp"] != record["signal_timestamp"]
    assert record["option_observation"]["available"] is False


def test_duplicate_setup_rejected_on_capture():
    ledger = exp.new_ledger()
    ledger, status1 = exp.capture_signal(ledger, short_signal())
    ledger, status2 = exp.capture_signal(ledger, short_signal())
    assert status1 == "added"
    assert status2 == "duplicate"
    assert len(ledger["records"]) == 1


def test_half_atr_trigger_and_be_arming_then_control_still_minus1():
    record = exp.create_record(short_signal())
    updated = exp.evaluate_record_with_intraday(
        record,
        bars([
            {"Open": 99.8, "High": 99.9, "Low": 98.0, "Close": 98.5},
            {"Open": 98.5, "High": 99.5, "Low": 98.2, "Close": 99.0},
        ]),
    )
    assert updated["events"]["plus_0_5_atr_at"] is not None
    assert updated["state"] == exp.STATE_BE_ARMED
    assert updated["control"]["outcome"] == exp.OUTCOME_OPEN
    assert updated["thresholds"]["control_stop_minus_1_atr"] == 104.0


def test_control_loses_while_test_breakeven_prevents_full_loss():
    record = exp.create_record(short_signal())
    updated = exp.evaluate_record_with_intraday(
        record,
        bars([
            {"Open": 99.8, "High": 99.9, "Low": 98.0, "Close": 98.5},
            {"Open": 98.5, "High": 100.0, "Low": 98.3, "Close": 100.0},
            {"Open": 100.0, "High": 104.0, "Low": 100.0, "Close": 103.5},
        ]),
    )
    assert updated["test"]["outcome"] == exp.OUTCOME_BREAKEVEN
    assert updated["control"]["outcome"] == exp.OUTCOME_STOP_MINUS_1R
    assert updated["diagnostics"]["full_loss_prevented_by_be"]


def test_test_b_breakeven_then_control_later_wins_and_reaches_plus2_plus3():
    record = exp.create_record(short_signal())
    updated = exp.evaluate_record_with_intraday(
        record,
        bars([
            {"Open": 99.8, "High": 99.9, "Low": 98.0, "Close": 98.2},
            {"Open": 98.2, "High": 100.0, "Low": 98.1, "Close": 99.8},
            {"Open": 99.8, "High": 99.9, "Low": 96.0, "Close": 96.5},
            {"Open": 96.5, "High": 97.0, "Low": 91.0, "Close": 91.5},
            {"Open": 91.5, "High": 92.0, "Low": 87.0, "Close": 88.0},
        ]),
    )
    assert updated["test"]["outcome"] == exp.OUTCOME_BREAKEVEN
    assert updated["control"]["outcome"] == exp.OUTCOME_TARGET_1R
    assert updated["diagnostics"]["be_then_control_win_1r"]
    assert updated["diagnostics"]["be_then_later_2r"]
    assert updated["diagnostics"]["be_then_later_3r"]


def test_target_before_be():
    record = exp.create_record(short_signal())
    updated = exp.evaluate_record_with_intraday(
        record,
        bars([
            {"Open": 99.5, "High": 99.8, "Low": 95.8, "Close": 96.2},
        ]),
    )
    assert updated["control"]["outcome"] == exp.OUTCOME_TARGET_1R
    assert updated["test"]["outcome"] == exp.OUTCOME_TARGET_1R


def test_be_before_target():
    record = exp.create_record(short_signal())
    updated = exp.evaluate_record_with_intraday(
        record,
        bars([
            {"Open": 99.7, "High": 99.8, "Low": 98.0, "Close": 98.5},
            {"Open": 98.5, "High": 100.0, "Low": 98.2, "Close": 99.9},
            {"Open": 99.9, "High": 100.2, "Low": 96.0, "Close": 96.5},
        ]),
    )
    assert updated["test"]["outcome"] == exp.OUTCOME_BREAKEVEN
    assert updated["control"]["outcome"] == exp.OUTCOME_TARGET_1R


def test_unresolved_same_bar_ordering_is_sequence_ambiguous():
    record = exp.create_record(short_signal())
    updated = exp.evaluate_record_with_intraday(
        record,
        bars([
            {"Open": 100.0, "High": 100.5, "Low": 97.8, "Close": 98.0},
        ]),
    )
    assert updated["state"] == exp.STATE_SEQUENCE_AMBIGUOUS
    assert updated["diagnostics"]["sequence_ambiguous_reason"] == "plus_0_5_and_entry_same_intraday_bar"


def test_restart_persistence_roundtrip(tmp_path: Path = None):
    path = (tmp_path or Path("/tmp")) / "mp_short_lifecycle_test_ledger.json"
    ledger = exp.new_ledger(str(path))
    ledger, status = exp.capture_signal(ledger, short_signal())
    assert status == "added"
    exp.save_ledger(ledger, str(path))
    loaded = exp.load_ledger(str(path))
    assert loaded["records"][0]["experiment_setup_id"] == ledger["records"][0]["experiment_setup_id"]
    path.unlink(missing_ok=True)


def test_spy_context_frozen_through_signal_close():
    idx = pd.date_range("2025-10-01", periods=230, freq="B")
    close = np.linspace(400.0, 450.0, len(idx))
    spy = pd.DataFrame(
        {
            "Open": close,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(len(idx), 1000.0),
        },
        index=idx,
    )
    signal_ts = idx[220]
    future = spy.copy()
    future.loc[idx[221]:, "Close"] = 999.0
    context = exp.spy_context_at(future, signal_ts)
    assert context["feature_timestamp"] == signal_ts.isoformat()
    assert context["spy_close"] != 999.0


def test_200_interim_and_500_final_governance():
    g199 = exp.governance_state(199)
    assert g199["phase"] == "ACCUMULATING"
    assert not g199["decision_allowed"]
    g200 = exp.governance_state(200)
    assert g200["phase"] == "INTERIM"
    assert g200["message"] == "DECISION PROHIBITED - EXPERIMENT INCOMPLETE"
    assert not g200["decision_allowed"]
    g500 = exp.governance_state(500)
    assert g500["phase"] == "FINAL_SAMPLE_REACHED"
    assert g500["decision_allowed"]


def test_status_counts_and_metrics():
    ledger = exp.new_ledger()
    rec = exp.evaluate_record_with_intraday(
        exp.create_record(short_signal()),
        bars([
            {"Open": 99.5, "High": 99.8, "Low": 95.8, "Close": 96.2},
        ]),
    )
    ledger, _ = exp.merge_record(ledger, rec)
    status = exp.experiment_status(ledger)
    metrics = exp.comparative_metrics(ledger)
    assert status["captured_setups"] == 1
    assert status["evaluable_setups"] == 1
    assert metrics["control_a"]["target_count"] == 1
    assert metrics["test_b"]["target_count"] == 1


def run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")


if __name__ == "__main__":
    run_all()
