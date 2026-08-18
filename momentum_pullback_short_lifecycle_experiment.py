"""Shadow-only SHORT lifecycle prospective experiment.

This module is intentionally inert: production scanner code does not import it.
It observes frozen Momentum Pullback Shadow V1.0 SHORT signals and maintains an
append-only research ledger for the breakeven-after-+0.5 ATR hypothesis.
"""

from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import momentum_pullback_shadow as mp


EXPERIMENT_VERSION = "momentum-pullback-short-lifecycle-prospective-v1.0"
EXPERIMENT_FROZEN_AT = "2026-08-18T21:17:23Z"
TARGET_SAMPLE = 500
INTERIM_SAMPLE = 200
DEFAULT_LEDGER_PATH = os.getenv(
    "MOMENTUM_PULLBACK_SHORT_LIFECYCLE_LEDGER_PATH",
    "/tmp/kairos_momentum_pullback_short_lifecycle_prospective_v1.json",
)

STATE_SIGNAL_CAPTURED = "SIGNAL_CAPTURED"
STATE_ENTRY_PENDING = "ENTRY_PENDING"
STATE_ENTRY_OPEN = "ENTRY_OPEN"
STATE_HALF_ATR_REACHED = "HALF_ATR_REACHED"
STATE_BE_ARMED = "BE_ARMED"
STATE_SEQUENCE_AMBIGUOUS = "SEQUENCE_AMBIGUOUS"
STATE_DATA_INCOMPLETE = "DATA_INCOMPLETE"
STATE_EXPERIMENT_COMPLETE = "EXPERIMENT_COMPLETE"

OUTCOME_TARGET_1R = "TARGET_+1R"
OUTCOME_STOP_MINUS_1R = "STOP_-1R"
OUTCOME_BREAKEVEN = "BREAKEVEN"
OUTCOME_OPEN = "OPEN"
OUTCOME_SEQUENCE_AMBIGUOUS = "SEQUENCE_AMBIGUOUS"
OUTCOME_DATA_INCOMPLETE = "DATA_INCOMPLETE"

IMMUTABLE_RECORD_FIELDS = {
    "experiment_version",
    "experiment_frozen_at",
    "experiment_setup_id",
    "setup_id",
    "pullback_generation_id",
    "signal_generation_id",
    "symbol",
    "direction",
    "signal_timestamp",
    "ema_interaction_timestamp",
    "entry_timestamp",
    "entry_price",
    "frozen_atr14",
    "thresholds",
    "spy_context",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _ts(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        return ts.tz_convert(None)
    return ts


def _iso(value: Any) -> str:
    return _ts(value).isoformat()


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    return str(value)


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Open", "High", "Low", "Close"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError("frame missing columns: " + ", ".join(missing))
    out = df.copy()
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_convert(None)
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    for col in [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in out.columns]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=required)


def experiment_setup_identity(signal: Dict[str, Any]) -> str:
    return "|".join(
        [
            str(signal.get("symbol") or "").upper(),
            mp.SHORT,
            str(signal.get("ema_interaction_timestamp") or ""),
            str(signal.get("signal_timestamp") or ""),
            str(signal.get("entry_timestamp") or ""),
            EXPERIMENT_VERSION,
        ]
    )


def short_thresholds(entry_price: float, atr14: float) -> Dict[str, float]:
    return {
        "favorable_plus_0_5_atr": entry_price - 0.5 * atr14,
        "target_plus_1_atr": entry_price - 1.0 * atr14,
        "diagnostic_plus_2_atr": entry_price - 2.0 * atr14,
        "diagnostic_plus_3_atr": entry_price - 3.0 * atr14,
        "control_stop_minus_1_atr": entry_price + 1.0 * atr14,
        "breakeven": entry_price,
    }


def signal_is_eligible(signal: Dict[str, Any], freeze_at: str = EXPERIMENT_FROZEN_AT) -> Tuple[bool, str]:
    if str(signal.get("direction")) != mp.SHORT:
        return False, "not_short"
    if str(signal.get("state")) != mp.STATE_EXECUTION_CONFIRMED:
        return False, "not_execution_confirmed"
    if str(signal.get("version")) != mp.VERSION:
        return False, "not_frozen_v1_signal"
    signal_ts = signal.get("signal_timestamp")
    if not signal_ts:
        return False, "missing_signal_timestamp"
    if _ts(signal_ts) <= _ts(freeze_at):
        return False, "pre_freeze_signal"
    atr14 = _finite_float(signal.get("atr14"))
    if atr14 is None or atr14 <= 0:
        return False, "invalid_atr14"
    return True, "eligible"


def spy_context_at(spy_df: pd.DataFrame, signal_timestamp: Any) -> Dict[str, Any]:
    frame = mp.indicator_frame(mp.historical_slice(spy_df, signal_timestamp))
    if len(frame) < 200:
        return {"available": False, "reason": "insufficient_spy_history"}
    row = frame.iloc[-1]
    close = _finite_float(row.get("Close"))
    sma20 = _finite_float(mp.sma(frame["Close"], 20).iloc[-1])
    sma200 = _finite_float(row.get("SMA200"))
    ret5 = close / float(frame["Close"].iloc[-6]) - 1.0 if close is not None and len(frame) >= 6 else None
    ret20 = close / float(frame["Close"].iloc[-21]) - 1.0 if close is not None and len(frame) >= 21 else None
    return {
        "available": True,
        "feature_timestamp": _iso(frame.index[-1]),
        "spy_close": close,
        "spy_sma20": sma20,
        "spy_sma200": sma200,
        "spy_5_session_return": ret5,
        "spy_20_session_return": ret20,
        "spy_close_vs_sma20": "ABOVE" if close is not None and sma20 is not None and close >= sma20 else "BELOW",
        "spy_close_vs_sma200": "ABOVE" if close is not None and sma200 is not None and close >= sma200 else "BELOW",
    }


def create_record(
    signal: Dict[str, Any],
    *,
    freeze_at: str = EXPERIMENT_FROZEN_AT,
    spy_context: Optional[Dict[str, Any]] = None,
    option_observation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    eligible, reason = signal_is_eligible(signal, freeze_at)
    if not eligible:
        raise ValueError(f"signal not eligible for experiment: {reason}")
    entry_price = _finite_float(signal.get("entry_price"))
    atr14 = float(signal["atr14"])
    entry_timestamp = signal.get("entry_timestamp")
    record = {
        "experiment_version": EXPERIMENT_VERSION,
        "experiment_frozen_at": freeze_at,
        "first_eligible_signal_after_freeze": None,
        "captured_at": _utc_now(),
        "last_observed_at": None,
        "experiment_setup_id": experiment_setup_identity(signal),
        "setup_id": signal.get("setup_id"),
        "pullback_generation_id": signal.get("pullback_generation_id"),
        "signal_generation_id": signal.get("signal_generation_id"),
        "symbol": str(signal.get("symbol") or "").upper(),
        "direction": mp.SHORT,
        "signal_timestamp": signal.get("signal_timestamp"),
        "ema_interaction_timestamp": signal.get("ema_interaction_timestamp"),
        "entry_timestamp": entry_timestamp,
        "entry_price": entry_price,
        "frozen_atr14": atr14,
        "thresholds": short_thresholds(entry_price, atr14) if entry_price is not None else None,
        "state": STATE_ENTRY_OPEN if entry_timestamp and entry_price is not None else STATE_ENTRY_PENDING,
        "control": {"policy": "Experimental Control A", "outcome": OUTCOME_OPEN, "outcome_at": None},
        "test": {"policy": "Experimental Breakeven Test B", "outcome": OUTCOME_OPEN, "outcome_at": None},
        "events": {
            "plus_0_5_atr_at": None,
            "plus_1_atr_at": None,
            "plus_2_atr_at": None,
            "plus_3_atr_at": None,
            "entry_after_plus_0_5_atr_at": None,
            "minus_1_atr_at": None,
        },
        "timing": {},
        "diagnostics": {
            "be_then_control_win_1r": False,
            "be_then_later_2r": False,
            "be_then_later_3r": False,
            "full_loss_prevented_by_be": False,
            "sequence_ambiguous_reason": None,
            "mfe_atr": 0.0,
            "mae_atr": 0.0,
        },
        "spy_context": spy_context or {"available": False, "reason": "not_provided"},
        "option_observation": option_observation or {"available": False, "reason": "not_provided"},
        "observation_count": 0,
        "history": [],
    }
    return record


def resolve_next_session_entry(record: Dict[str, Any], daily_df: pd.DataFrame) -> Dict[str, Any]:
    updated = deepcopy(record)
    if updated.get("entry_timestamp") and _finite_float(updated.get("entry_price")) is not None:
        if not updated.get("thresholds"):
            updated["thresholds"] = short_thresholds(float(updated["entry_price"]), float(updated["frozen_atr14"]))
        updated["state"] = STATE_ENTRY_OPEN if updated.get("state") not in {STATE_EXPERIMENT_COMPLETE, STATE_SEQUENCE_AMBIGUOUS} else updated.get("state")
        return updated
    frame = _clean_frame(daily_df)
    if frame.empty:
        updated["state"] = STATE_DATA_INCOMPLETE
        updated.setdefault("diagnostics", {})["entry_resolution_reason"] = "daily_frame_empty"
        return updated
    signal_ts = _ts(updated.get("signal_timestamp"))
    future = frame.loc[frame.index > signal_ts]
    if future.empty:
        updated["state"] = STATE_ENTRY_PENDING
        updated.setdefault("diagnostics", {})["entry_resolution_reason"] = "next_session_not_available_yet"
        return updated
    entry_row = future.iloc[0]
    entry_price = _finite_float(entry_row.get("Open"))
    if entry_price is None:
        updated["state"] = STATE_DATA_INCOMPLETE
        updated.setdefault("diagnostics", {})["entry_resolution_reason"] = "next_session_open_missing"
        return updated
    updated["entry_timestamp"] = _iso(future.index[0])
    updated["entry_price"] = entry_price
    updated["thresholds"] = short_thresholds(entry_price, float(updated["frozen_atr14"]))
    updated["state"] = STATE_ENTRY_OPEN
    updated.setdefault("history", []).append({
        "observed_at": _utc_now(),
        "state": STATE_ENTRY_OPEN,
        "event": "next_session_entry_resolved",
        "entry_timestamp": updated["entry_timestamp"],
        "entry_price": entry_price,
    })
    return updated


def _bar_contains(bar: pd.Series, price: float) -> bool:
    return float(bar["Low"]) <= price <= float(bar["High"])


def _set_event(events: Dict[str, Any], key: str, when: str) -> None:
    if events.get(key) is None:
        events[key] = when


def _session_delta(start: Optional[str], end: Optional[str]) -> Optional[int]:
    if not start or not end:
        return None
    return int(np.busday_count(_ts(start).date(), _ts(end).date()))


def evaluate_record_with_intraday(
    record: Dict[str, Any],
    intraday_df: pd.DataFrame,
    *,
    observed_at: Optional[str] = None,
) -> Dict[str, Any]:
    updated = deepcopy(record)
    if not updated.get("entry_timestamp") or _finite_float(updated.get("entry_price")) is None or not updated.get("thresholds"):
        updated["state"] = STATE_ENTRY_PENDING
        return updated
    frame = _clean_frame(intraday_df)
    entry_ts = _ts(updated["entry_timestamp"])
    frame = frame.loc[frame.index >= entry_ts]
    if frame.empty:
        updated["state"] = STATE_DATA_INCOMPLETE
        updated["control"]["outcome"] = OUTCOME_DATA_INCOMPLETE
        updated["test"]["outcome"] = OUTCOME_DATA_INCOMPLETE
        return updated

    th = updated["thresholds"]
    events = updated["events"]
    be_armed = events.get("plus_0_5_atr_at") is not None
    control_done = updated["control"]["outcome"] not in {OUTCOME_OPEN, OUTCOME_DATA_INCOMPLETE}
    test_done = updated["test"]["outcome"] not in {OUTCOME_OPEN, OUTCOME_DATA_INCOMPLETE}
    max_fav = 0.0
    max_adv = 0.0

    for ts, bar in frame.iterrows():
        when = _iso(ts)
        high = float(bar["High"])
        low = float(bar["Low"])
        fav = max(0.0, updated["entry_price"] - low) / updated["frozen_atr14"]
        adv = min(0.0, (updated["entry_price"] - high) / updated["frozen_atr14"])
        max_fav = max(max_fav, fav)
        max_adv = min(max_adv, adv)

        hits_half = low <= th["favorable_plus_0_5_atr"]
        hits_target = low <= th["target_plus_1_atr"]
        hits_plus2 = low <= th["diagnostic_plus_2_atr"]
        hits_plus3 = low <= th["diagnostic_plus_3_atr"]
        hits_entry = _bar_contains(bar, th["breakeven"])
        hits_stop = high >= th["control_stop_minus_1_atr"]

        if hits_plus3:
            _set_event(events, "plus_3_atr_at", when)
        if hits_plus2:
            _set_event(events, "plus_2_atr_at", when)
        if hits_target:
            _set_event(events, "plus_1_atr_at", when)
        if hits_half:
            _set_event(events, "plus_0_5_atr_at", when)
        if hits_stop:
            _set_event(events, "minus_1_atr_at", when)

        if not be_armed and hits_half and hits_entry:
            updated["state"] = STATE_SEQUENCE_AMBIGUOUS
            updated["control"]["outcome"] = OUTCOME_SEQUENCE_AMBIGUOUS
            updated["test"]["outcome"] = OUTCOME_SEQUENCE_AMBIGUOUS
            updated["diagnostics"]["sequence_ambiguous_reason"] = "plus_0_5_and_entry_same_intraday_bar"
            break
        if not control_done and hits_target and hits_stop:
            updated["state"] = STATE_SEQUENCE_AMBIGUOUS
            updated["control"]["outcome"] = OUTCOME_SEQUENCE_AMBIGUOUS
            updated["test"]["outcome"] = OUTCOME_SEQUENCE_AMBIGUOUS
            updated["diagnostics"]["sequence_ambiguous_reason"] = "target_and_stop_same_intraday_bar"
            break

        if not control_done:
            if hits_target:
                updated["control"]["outcome"] = OUTCOME_TARGET_1R
                updated["control"]["outcome_at"] = when
                control_done = True
            elif hits_stop:
                updated["control"]["outcome"] = OUTCOME_STOP_MINUS_1R
                updated["control"]["outcome_at"] = when
                control_done = True

        if not test_done:
            if be_armed and hits_target:
                updated["test"]["outcome"] = OUTCOME_TARGET_1R
                updated["test"]["outcome_at"] = when
                test_done = True
            elif be_armed and hits_entry:
                _set_event(events, "entry_after_plus_0_5_atr_at", when)
                updated["test"]["outcome"] = OUTCOME_BREAKEVEN
                updated["test"]["outcome_at"] = when
                test_done = True
            elif not be_armed and hits_target:
                updated["test"]["outcome"] = OUTCOME_TARGET_1R
                updated["test"]["outcome_at"] = when
                test_done = True
            elif not be_armed and hits_stop:
                updated["test"]["outcome"] = OUTCOME_STOP_MINUS_1R
                updated["test"]["outcome_at"] = when
                test_done = True

        if not be_armed and hits_half:
            be_armed = True
            updated["state"] = STATE_BE_ARMED

    updated["diagnostics"]["mfe_atr"] = max(updated["diagnostics"].get("mfe_atr") or 0.0, max_fav)
    updated["diagnostics"]["mae_atr"] = min(updated["diagnostics"].get("mae_atr") or 0.0, max_adv)
    updated["diagnostics"]["be_then_control_win_1r"] = (
        updated["test"]["outcome"] == OUTCOME_BREAKEVEN
        and updated["control"]["outcome"] == OUTCOME_TARGET_1R
    )
    updated["diagnostics"]["be_then_later_2r"] = (
        updated["test"]["outcome"] == OUTCOME_BREAKEVEN and updated["events"].get("plus_2_atr_at") is not None
    )
    updated["diagnostics"]["be_then_later_3r"] = (
        updated["test"]["outcome"] == OUTCOME_BREAKEVEN and updated["events"].get("plus_3_atr_at") is not None
    )
    updated["diagnostics"]["full_loss_prevented_by_be"] = (
        updated["test"]["outcome"] == OUTCOME_BREAKEVEN
        and updated["control"]["outcome"] == OUTCOME_STOP_MINUS_1R
    )
    updated["timing"] = {
        "entry_to_plus_0_5_sessions": _session_delta(updated["entry_timestamp"], events.get("plus_0_5_atr_at")),
        "entry_to_plus_1_sessions": _session_delta(updated["entry_timestamp"], events.get("plus_1_atr_at")),
        "plus_0_5_to_entry_sessions": _session_delta(events.get("plus_0_5_atr_at"), events.get("entry_after_plus_0_5_atr_at")),
        "plus_0_5_to_plus_1_sessions": _session_delta(events.get("plus_0_5_atr_at"), events.get("plus_1_atr_at")),
        "entry_to_control_stop_sessions": _session_delta(updated["entry_timestamp"], events.get("minus_1_atr_at")),
    }
    updated["last_observed_at"] = observed_at or _utc_now()
    policy_outcomes_done = (
        updated["control"]["outcome"] not in {OUTCOME_OPEN, OUTCOME_DATA_INCOMPLETE}
        and updated["test"]["outcome"] not in {OUTCOME_OPEN, OUTCOME_DATA_INCOMPLETE}
    )
    sacrificed_path_done = (
        updated["test"]["outcome"] != OUTCOME_BREAKEVEN
        or updated["control"]["outcome"] == OUTCOME_STOP_MINUS_1R
        or updated["events"].get("plus_3_atr_at") is not None
        or updated["events"].get("minus_1_atr_at") is not None
    )
    if updated["state"] != STATE_SEQUENCE_AMBIGUOUS and policy_outcomes_done and sacrificed_path_done:
        updated["state"] = STATE_EXPERIMENT_COMPLETE
    updated["observation_count"] = int(updated.get("observation_count") or 0) + 1
    updated["history"].append(
        {
            "observed_at": updated["last_observed_at"],
            "state": updated["state"],
            "control_outcome": updated["control"]["outcome"],
            "test_outcome": updated["test"]["outcome"],
        }
    )
    return updated


def new_ledger(path: Optional[str] = None, *, freeze_at: str = EXPERIMENT_FROZEN_AT) -> Dict[str, Any]:
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "experiment_frozen_at": freeze_at,
        "first_eligible_signal_after_freeze": None,
        "target_sample": TARGET_SAMPLE,
        "interim_sample": INTERIM_SAMPLE,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "persistence_path": path or DEFAULT_LEDGER_PATH,
        "records": [],
        "history": [],
    }


def load_ledger(path: Optional[str] = None) -> Dict[str, Any]:
    ledger_path = Path(path or DEFAULT_LEDGER_PATH)
    if not ledger_path.exists():
        return new_ledger(str(ledger_path))
    return json.loads(ledger_path.read_text())


def save_ledger(ledger: Dict[str, Any], path: Optional[str] = None) -> str:
    ledger_path = Path(path or ledger.get("persistence_path") or DEFAULT_LEDGER_PATH)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    payload = deepcopy(ledger)
    payload["updated_at"] = _utc_now()
    tmp = ledger_path.with_suffix(ledger_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    os.replace(tmp, ledger_path)
    return str(ledger_path)


def immutable_subset(record: Dict[str, Any]) -> Dict[str, Any]:
    return {key: deepcopy(record.get(key)) for key in sorted(IMMUTABLE_RECORD_FIELDS)}


def _immutable_compatible(existing: Dict[str, Any], incoming: Dict[str, Any]) -> bool:
    for key in sorted(IMMUTABLE_RECORD_FIELDS):
        old = existing.get(key)
        new = incoming.get(key)
        if old in (None, {}) and new not in (None, {}):
            continue
        if old != new:
            return False
    return True


def merge_record(ledger: Dict[str, Any], record: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    merged = deepcopy(ledger)
    records = merged.setdefault("records", [])
    setup_id = record["experiment_setup_id"]
    for idx, existing in enumerate(records):
        if existing.get("experiment_setup_id") != setup_id:
            continue
        if not _immutable_compatible(existing, record):
            existing.setdefault("identity_mutation_failures", []).append(
                {"observed_at": _utc_now(), "incoming_immutable": immutable_subset(record)}
            )
        mutable = deepcopy(existing)
        for key in ["entry_timestamp", "entry_price", "thresholds", "last_observed_at", "state", "control", "test", "events", "timing", "diagnostics", "observation_count", "history"]:
            if key in {"entry_timestamp", "entry_price", "thresholds"} and mutable.get(key) not in (None, {}):
                continue
            mutable[key] = deepcopy(record.get(key, mutable.get(key)))
        records[idx] = mutable
        merged.setdefault("history", []).append({"at": _utc_now(), "event": "duplicate_observation_merged", "setup_id": setup_id})
        return merged, "duplicate"
    records.append(deepcopy(record))
    first = merged.get("first_eligible_signal_after_freeze")
    signal_ts = record.get("signal_timestamp")
    if signal_ts and (not first or _ts(signal_ts) < _ts(first)):
        merged["first_eligible_signal_after_freeze"] = signal_ts
    merged.setdefault("history", []).append({"at": _utc_now(), "event": "record_added", "setup_id": setup_id})
    return merged, "added"


def capture_signal(
    ledger: Dict[str, Any],
    signal: Dict[str, Any],
    *,
    spy_context: Optional[Dict[str, Any]] = None,
    option_observation: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], str]:
    eligible, reason = signal_is_eligible(signal, ledger.get("experiment_frozen_at") or EXPERIMENT_FROZEN_AT)
    if not eligible:
        return deepcopy(ledger), reason
    record = create_record(
        signal,
        freeze_at=ledger.get("experiment_frozen_at") or EXPERIMENT_FROZEN_AT,
        spy_context=spy_context,
        option_observation=option_observation,
    )
    return merge_record(ledger, record)


def governance_state(evaluable_count: int) -> Dict[str, Any]:
    if evaluable_count < INTERIM_SAMPLE:
        return {
            "phase": "ACCUMULATING",
            "decision_allowed": False,
            "remaining_to_interim": INTERIM_SAMPLE - evaluable_count,
            "remaining_to_final": TARGET_SAMPLE - evaluable_count,
        }
    if evaluable_count < TARGET_SAMPLE:
        return {
            "phase": "INTERIM",
            "decision_allowed": False,
            "message": "DECISION PROHIBITED - EXPERIMENT INCOMPLETE",
            "remaining_to_interim": 0,
            "remaining_to_final": TARGET_SAMPLE - evaluable_count,
        }
    return {
        "phase": "FINAL_SAMPLE_REACHED",
        "decision_allowed": True,
        "remaining_to_interim": 0,
        "remaining_to_final": 0,
    }


def experiment_status(ledger: Dict[str, Any]) -> Dict[str, Any]:
    records = ledger.get("records") or []
    evaluable = [
        row for row in records
        if row.get("control", {}).get("outcome") not in {OUTCOME_OPEN, OUTCOME_DATA_INCOMPLETE}
        and row.get("test", {}).get("outcome") not in {OUTCOME_OPEN, OUTCOME_DATA_INCOMPLETE}
    ]
    ambiguous = [row for row in records if row.get("state") == STATE_SEQUENCE_AMBIGUOUS]
    incomplete = [
        row for row in records
        if row.get("state") == STATE_DATA_INCOMPLETE
        or row.get("control", {}).get("outcome") == OUTCOME_DATA_INCOMPLETE
        or row.get("test", {}).get("outcome") == OUTCOME_DATA_INCOMPLETE
    ]
    open_setups = [row for row in records if row not in evaluable and row not in ambiguous and row not in incomplete]

    def counts(policy: str) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for row in records:
            outcome = str(row.get(policy, {}).get("outcome") or OUTCOME_OPEN)
            out[outcome] = out.get(outcome, 0) + 1
        return out

    return {
        "experiment_version": ledger.get("experiment_version") or EXPERIMENT_VERSION,
        "experiment_frozen_at": ledger.get("experiment_frozen_at") or EXPERIMENT_FROZEN_AT,
        "target_sample": TARGET_SAMPLE,
        "interim_sample": INTERIM_SAMPLE,
        "captured_setups": len(records),
        "open_setups": len(open_setups),
        "evaluable_setups": len(evaluable),
        "ambiguous_setups": len(ambiguous),
        "incomplete_setups": len(incomplete),
        **governance_state(len(evaluable)),
        "control_a_outcome_counts": counts("control"),
        "test_b_outcome_counts": counts("test"),
        "full_losses_prevented_by_be": sum(1 for row in records if row.get("diagnostics", {}).get("full_loss_prevented_by_be")),
        "be_then_control_win_1r": sum(1 for row in records if row.get("diagnostics", {}).get("be_then_control_win_1r")),
        "be_then_later_2r": sum(1 for row in records if row.get("diagnostics", {}).get("be_then_later_2r")),
        "be_then_later_3r": sum(1 for row in records if row.get("diagnostics", {}).get("be_then_later_3r")),
    }


@dataclass(frozen=True)
class ExperimentalMetrics:
    completed_setups: int
    target_count: int
    stop_count: int
    breakeven_count: int
    ambiguous_count: int
    incomplete_count: int
    target_rate: Optional[float]
    loss_rate: Optional[float]
    breakeven_rate: Optional[float]
    mean_outcome_atr: Optional[float]
    median_outcome_atr: Optional[float]
    cumulative_experimental_r: float
    maximum_experimental_drawdown: float
    longest_losing_sequence: int


def _policy_outcome_value(policy: str, outcome: str) -> Optional[float]:
    if outcome == OUTCOME_TARGET_1R:
        return 1.0
    if outcome == OUTCOME_STOP_MINUS_1R:
        return -1.0
    if policy == "test" and outcome == OUTCOME_BREAKEVEN:
        return 0.0
    return None


def policy_metrics(records: Iterable[Dict[str, Any]], policy: str) -> Dict[str, Any]:
    rows = list(records)
    values: List[float] = []
    target_count = stop_count = be_count = ambiguous_count = incomplete_count = 0
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    losing_streak = longest_losing = 0
    for row in rows:
        outcome = str(row.get(policy, {}).get("outcome") or OUTCOME_OPEN)
        target_count += outcome == OUTCOME_TARGET_1R
        stop_count += outcome == OUTCOME_STOP_MINUS_1R
        be_count += outcome == OUTCOME_BREAKEVEN
        ambiguous_count += outcome == OUTCOME_SEQUENCE_AMBIGUOUS
        incomplete_count += outcome in {OUTCOME_OPEN, OUTCOME_DATA_INCOMPLETE}
        value = _policy_outcome_value(policy, outcome)
        if value is None:
            continue
        values.append(value)
        equity += value
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)
        losing_streak = losing_streak + 1 if value < 0 else 0
        longest_losing = max(longest_losing, losing_streak)
    completed = len(values)
    return {
        "completed_setups": completed,
        "target_count": target_count,
        "stop_count": stop_count,
        "breakeven_count": be_count,
        "ambiguous_count": ambiguous_count,
        "incomplete_count": incomplete_count,
        "target_rate": target_count / completed if completed else None,
        "loss_rate": stop_count / completed if completed else None,
        "breakeven_rate": be_count / completed if completed else None,
        "mean_outcome_atr": float(np.mean(values)) if values else None,
        "median_outcome_atr": float(np.median(values)) if values else None,
        "cumulative_experimental_r": float(sum(values)),
        "maximum_experimental_drawdown": float(max_dd),
        "longest_losing_sequence": longest_losing,
    }


def comparative_metrics(ledger: Dict[str, Any]) -> Dict[str, Any]:
    records = ledger.get("records") or []
    return {
        "control_a": policy_metrics(records, "control"),
        "test_b": policy_metrics(records, "test"),
        "test_b_diagnostics": {
            "full_losses_prevented_by_be": sum(1 for row in records if row.get("diagnostics", {}).get("full_loss_prevented_by_be")),
            "eventual_1r_winners_sacrificed": sum(1 for row in records if row.get("diagnostics", {}).get("be_then_control_win_1r")),
            "eventual_2r_continuations_sacrificed": sum(1 for row in records if row.get("diagnostics", {}).get("be_then_later_2r")),
            "eventual_3r_continuations_sacrificed": sum(1 for row in records if row.get("diagnostics", {}).get("be_then_later_3r")),
        },
    }
