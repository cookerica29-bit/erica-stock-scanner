from __future__ import annotations

import math
import statistics
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd


POSITION_INTELLIGENCE_VERSION = 1
POSITION_REPLAY_VERSION = 1
DEFAULT_REPLAY_TIMEFRAME = "4H"
REAL_EVIDENCE_MIN_CLOSED_COMPLETE = 20

STATES = ["HEALTHY", "WATCH", "PROTECT", "EXIT", "DATA_NEEDED"]
MILESTONE_LEVELS = [25, 50, 75]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def coerce_number(value: Any) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def normalize_direction(value: Any) -> str | None:
    direction = str(value or "").upper()
    if direction in {"LONG", "CALL"}:
        return "LONG"
    if direction in {"SHORT", "PUT"}:
        return "SHORT"
    return None


def parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def iso_timestamp(value: Any) -> str | None:
    parsed = parse_timestamp(value)
    if not parsed:
        return None
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def count_weekdays_between(start: Any, end: Any) -> int | None:
    start_dt = parse_timestamp(start)
    end_dt = parse_timestamp(end)
    if not start_dt or not end_dt:
        return None
    if end_dt < start_dt:
        return 0
    days = 0
    cursor = start_dt.date()
    finish = end_dt.date()
    while cursor <= finish:
        if cursor.weekday() < 5:
            days += 1
        cursor = cursor + timedelta(days=1)
    return max(0, days - 1)


def position_entry_price(position: dict[str, Any]) -> float | None:
    return coerce_number(first_present(
        position.get("actual_underlying_entry"),
        position.get("planned_underlying_entry"),
        position.get("entry_price"),
        position.get("entry"),
    ))


def position_stop_price(position: dict[str, Any]) -> float | None:
    return coerce_number(first_present(position.get("stop_price"), position.get("plannedStop"), position.get("original_stop")))


def position_tp_price(position: dict[str, Any], number: int) -> float | None:
    keys = {
        1: ("target_price", "plannedTp1", "tp1", "original_tp1"),
        2: ("plannedTp2", "tp2", "original_tp2"),
        3: ("plannedTp3", "tp3", "original_tp3"),
    }.get(number, ())
    return coerce_number(first_present(*(position.get(key) for key in keys)))


def replay_readiness(position: dict[str, Any], candles_available: bool | None = None, timeframe_source: str | None = None) -> dict[str, Any]:
    direction = normalize_direction(first_present(position.get("direction"), position.get("option_type"), position.get("optionType")))
    entry_ts = parse_timestamp(first_present(position.get("entry_timestamp"), position.get("tracking_started_at"), position.get("signal_timestamp"), position.get("created_at"), position.get("createdAt")))
    entry = position_entry_price(position)
    stop = position_stop_price(position)
    tp1 = position_tp_price(position, 1)
    missing_required = []
    invalid = []
    if not direction:
        missing_required.append("direction")
    if not entry_ts:
        missing_required.append("entry timestamp")
    if entry is None:
        missing_required.append("underlying entry")
    if stop is None:
        missing_required.append("stop")
    if tp1 is None:
        missing_required.append("TP1")
    if direction and entry is not None and stop is not None:
        if direction == "LONG" and not entry > stop:
            invalid.append("invalid stop geometry")
        if direction == "SHORT" and not stop > entry:
            invalid.append("invalid stop geometry")
    if direction and entry is not None and tp1 is not None:
        if direction == "LONG" and not tp1 > entry:
            invalid.append("invalid target geometry")
        if direction == "SHORT" and not entry > tp1:
            invalid.append("invalid target geometry")

    optional_missing = []
    if not first_present(position.get("setup_grade"), position.get("setupGrade"), position.get("grade")):
        optional_missing.append("grade")
    if not first_present(position.get("scanner_timeframe"), position.get("timeframe")):
        optional_missing.append("setup timeframe")
    if position_tp_price(position, 2) is None:
        optional_missing.append("TP2")
    if position_tp_price(position, 3) is None:
        optional_missing.append("TP3")
    if not first_present(position.get("actual_strike"), position.get("strike_price"), position.get("strike")):
        optional_missing.append("option strike")
    if not first_present(position.get("actual_expiration"), position.get("expiration_date"), position.get("expiry")):
        optional_missing.append("option expiration")
    if not first_present(position.get("actual_option_premium"), position.get("premium_paid"), position.get("askAtSelection")):
        optional_missing.append("option premium")
    if str(first_present(position.get("result"), position.get("status"), "Open") or "").lower() != "open" and not first_present(position.get("result"), position.get("outcome"), position.get("completion_reason")):
        optional_missing.append("recorded outcome")

    available = {
        "direction": bool(direction),
        "entry": entry is not None,
        "stop": stop is not None,
        "TP1": tp1 is not None,
        "entry timestamp": bool(entry_ts),
    }
    if candles_available is False:
        missing_required.append("historical candles")
    if missing_required or invalid:
        status = "NOT_REPLAYABLE"
    elif optional_missing or timeframe_source == "inferred_default":
        status = "PARTIALLY_READY"
    else:
        status = "REPLAY_READY"
    return {
        "status": status,
        "available": available,
        "missing_required": sorted(set(missing_required)),
        "missing_optional": sorted(set(optional_missing)),
        "invalid": sorted(set(invalid)),
        "timeframe_source": timeframe_source,
    }


def progress_to_tp1(direction: str | None, entry: float | None, tp1: float | None, price: float | None) -> dict[str, float | int] | None:
    if not direction or entry is None or tp1 is None or price is None:
        return None
    denominator = entry - tp1 if direction == "SHORT" else tp1 - entry
    if denominator <= 0:
        return None
    numerator = entry - price if direction == "SHORT" else price - entry
    raw = numerator / denominator * 100
    return {"raw_percent": raw, "display_percent": max(0, min(100, round(raw)))}


def r_multiple(direction: str | None, entry: float | None, stop: float | None, price: float | None) -> float | None:
    if not direction or entry is None or stop is None or price is None:
        return None
    initial_risk = stop - entry if direction == "SHORT" else entry - stop
    if initial_risk <= 0:
        return None
    reward = entry - price if direction == "SHORT" else price - entry
    return reward / initial_risk


def best_price_for(position: dict[str, Any], direction: str | None, current_price: float | None) -> float | None:
    stored = coerce_number(first_present(
        position.get("position_best_price"),
        position.get("lowest_price_reached") if direction == "SHORT" else position.get("highest_price_reached"),
    ))
    if stored is None:
        return current_price
    if current_price is None:
        return stored
    return min(stored, current_price) if direction == "SHORT" else max(stored, current_price)


def data_issue(position: dict[str, Any], current_price: float | None) -> dict[str, str] | None:
    direction = normalize_direction(first_present(position.get("direction"), position.get("option_type"), position.get("optionType")))
    entry = position_entry_price(position)
    stop = position_stop_price(position)
    tp1 = position_tp_price(position, 1)
    if not direction:
        return {"code": "MISSING_POSITION_DATA", "message": "Missing direction"}
    if entry is None:
        return {"code": "MISSING_POSITION_DATA", "message": "Missing underlying entry price"}
    if stop is None:
        return {"code": "MISSING_POSITION_DATA", "message": "Missing original stop"}
    if tp1 is None:
        return {"code": "MISSING_POSITION_DATA", "message": "Missing TP1"}
    if current_price is None:
        return {"code": "MISSING_POSITION_DATA", "message": "Current price unavailable"}
    if direction == "LONG" and not entry > stop:
        return {"code": "INVALID_POSITION_GEOMETRY", "message": "Invalid stop geometry"}
    if direction == "SHORT" and not stop > entry:
        return {"code": "INVALID_POSITION_GEOMETRY", "message": "Invalid stop geometry"}
    if direction == "LONG" and not tp1 > entry:
        return {"code": "INVALID_POSITION_GEOMETRY", "message": "Invalid target geometry"}
    if direction == "SHORT" and not entry > tp1:
        return {"code": "INVALID_POSITION_GEOMETRY", "message": "Invalid target geometry"}
    return None


def state_label(state: str) -> str:
    return {
        "HEALTHY": "🟢 Healthy",
        "WATCH": "🟡 Watch",
        "PROTECT": "🟠 Protect",
        "EXIT": "🔴 Exit",
        "DATA_NEEDED": "⚪ Data Needed",
    }.get(state, "⚪ Data Needed")


def next_action(state: str) -> tuple[str, str]:
    return {
        "HEALTHY": ("HOLD", "Hold"),
        "WATCH": ("MONITOR", "Monitor"),
        "PROTECT": ("PROTECT_PROFIT", "Protect Profit"),
        "EXIT": ("EXIT", "Exit"),
        "DATA_NEEDED": ("REVIEW_POSITION", "Review Position"),
    }.get(state, ("REVIEW_POSITION", "Review Position"))


SAFE_STATIC_INPUT = "SAFE_STATIC_INPUT"
RANGE_BOUND_ONLY = "RANGE_BOUND_ONLY"
STRIP_FROM_REPLAY = "STRIP_FROM_REPLAY"
RECOMPUTE_CHRONOLOGICALLY = "RECOMPUTE_CHRONOLOGICALLY"
UNUSED = "UNUSED"

REPLAY_STATIC_INPUT_FIELDS = {
    "ticker",
    "direction",
    "option_type",
    "optionType",
    "actual_underlying_entry",
    "planned_underlying_entry",
    "entry_price",
    "entry",
    "stop_price",
    "plannedStop",
    "original_stop",
    "target_price",
    "plannedTp1",
    "tp1",
    "original_tp1",
    "plannedTp2",
    "tp2",
    "original_tp2",
    "plannedTp3",
    "tp3",
    "original_tp3",
    "actual_entry_at",
    "position_opened_at",
    "tracking_started_at",
    "entry_timestamp",
    "signal_timestamp",
    "createdAt",
    "created_at",
    "expected_hold_min_days",
    "expected_hold_max_days",
    "expected_trading_days_low",
    "expected_trading_days_high",
}

REPLAY_FIELD_CLASSIFICATION = {
    **{field: SAFE_STATIC_INPUT for field in REPLAY_STATIC_INPUT_FIELDS},
    "exit_timestamp": RANGE_BOUND_ONLY,
    "tracking_completed_at": RANGE_BOUND_ONLY,
    "position_best_price": RECOMPUTE_CHRONOLOGICALLY,
    "best_price": RECOMPUTE_CHRONOLOGICALLY,
    "lowest_price_reached": RECOMPUTE_CHRONOLOGICALLY,
    "highest_price_reached": RECOMPUTE_CHRONOLOGICALLY,
    "position_max_progress_percent": RECOMPUTE_CHRONOLOGICALLY,
    "best_progress": RECOMPUTE_CHRONOLOGICALLY,
    "current_progress": RECOMPUTE_CHRONOLOGICALLY,
    "current_r": RECOMPUTE_CHRONOLOGICALLY,
    "position_tp1_reached": RECOMPUTE_CHRONOLOGICALLY,
    "tp1_reached": RECOMPUTE_CHRONOLOGICALLY,
    "first_target_touch_at": RECOMPUTE_CHRONOLOGICALLY,
    "tp1_reached_at": RECOMPUTE_CHRONOLOGICALLY,
    "second_target_touch_at": RECOMPUTE_CHRONOLOGICALLY,
    "tp2_reached_at": RECOMPUTE_CHRONOLOGICALLY,
    "third_target_touch_at": RECOMPUTE_CHRONOLOGICALLY,
    "tp3_reached_at": RECOMPUTE_CHRONOLOGICALLY,
    "target_hit_at": RECOMPUTE_CHRONOLOGICALLY,
    "first_stop_touch_at": RECOMPUTE_CHRONOLOGICALLY,
    "stop_hit_at": RECOMPUTE_CHRONOLOGICALLY,
    "last_market_price": RECOMPUTE_CHRONOLOGICALLY,
    "last_evaluated_at": RECOMPUTE_CHRONOLOGICALLY,
    "position_last_evaluated_at": RECOMPUTE_CHRONOLOGICALLY,
    "maximum_favorable_excursion": RECOMPUTE_CHRONOLOGICALLY,
    "maximum_favorable_excursion_atr": RECOMPUTE_CHRONOLOGICALLY,
    "maximum_favorable_excursion_percent": RECOMPUTE_CHRONOLOGICALLY,
    "maximum_favorable_excursion_r": RECOMPUTE_CHRONOLOGICALLY,
    "maximum_adverse_excursion": RECOMPUTE_CHRONOLOGICALLY,
    "maximum_adverse_excursion_atr": RECOMPUTE_CHRONOLOGICALLY,
    "maximum_adverse_excursion_percent": RECOMPUTE_CHRONOLOGICALLY,
    "maximum_adverse_excursion_r": RECOMPUTE_CHRONOLOGICALLY,
    "mfe": RECOMPUTE_CHRONOLOGICALLY,
    "mae": RECOMPUTE_CHRONOLOGICALLY,
    "reached_0_5r": RECOMPUTE_CHRONOLOGICALLY,
    "reached_1r": RECOMPUTE_CHRONOLOGICALLY,
    "reached_2r": RECOMPUTE_CHRONOLOGICALLY,
    "time_to_0_5r": RECOMPUTE_CHRONOLOGICALLY,
    "time_to_1r": RECOMPUTE_CHRONOLOGICALLY,
    "time_to_2r": RECOMPUTE_CHRONOLOGICALLY,
    "bars_elapsed": RECOMPUTE_CHRONOLOGICALLY,
    "bars_to_entry": RECOMPUTE_CHRONOLOGICALLY,
    "bars_to_stop": RECOMPUTE_CHRONOLOGICALLY,
    "bars_to_target": RECOMPUTE_CHRONOLOGICALLY,
    "trading_days_elapsed": RECOMPUTE_CHRONOLOGICALLY,
    "trading_days_to_entry": RECOMPUTE_CHRONOLOGICALLY,
    "trading_days_to_stop": RECOMPUTE_CHRONOLOGICALLY,
    "trading_days_to_target": RECOMPUTE_CHRONOLOGICALLY,
    "last_tracking_bar_time": RECOMPUTE_CHRONOLOGICALLY,
    "final_trading_days": STRIP_FROM_REPLAY,
    "final_bar_count": STRIP_FROM_REPLAY,
    "position_last_state": STRIP_FROM_REPLAY,
    "position_state_history": STRIP_FROM_REPLAY,
    "exit_price": STRIP_FROM_REPLAY,
    "recorded_outcome": STRIP_FROM_REPLAY,
    "result": STRIP_FROM_REPLAY,
    "outcome": STRIP_FROM_REPLAY,
    "completion_reason": STRIP_FROM_REPLAY,
    "closed_reason": STRIP_FROM_REPLAY,
    "reviewResult": STRIP_FROM_REPLAY,
    "tracking_status": STRIP_FROM_REPLAY,
}


def _point_in_time_replay_position(original: dict[str, Any], best_price: float | None, tp1_reached: bool) -> dict[str, Any]:
    replay_position = {key: original.get(key) for key in REPLAY_STATIC_INPUT_FIELDS if key in original}
    replay_position["position_best_price"] = best_price
    replay_position["position_tp1_reached"] = tp1_reached
    replay_position["tp1_reached"] = tp1_reached
    return replay_position


def build_position_intelligence(position: dict[str, Any], market_snapshot: dict[str, Any]) -> dict[str, Any]:
    current_price = coerce_number(first_present(market_snapshot.get("current_price"), market_snapshot.get("price")))
    direction = normalize_direction(first_present(position.get("direction"), position.get("option_type"), position.get("optionType")))
    entry = position_entry_price(position)
    stop = position_stop_price(position)
    tp1 = position_tp_price(position, 1)
    issue = data_issue(position, current_price)
    base = {
        "available": not bool(issue),
        "position_intelligence_version": POSITION_INTELLIGENCE_VERSION,
        "ticker": str(position.get("ticker") or "").upper(),
        "direction": direction or str(first_present(position.get("direction"), position.get("option_type"), position.get("optionType"), "") or "").upper(),
        "entry_price": entry,
        "current_price": current_price,
        "original_stop": stop,
        "tp1": tp1,
        "source": "journal_trade_plan",
    }
    if issue:
        action, action_label = next_action("DATA_NEEDED")
        return {
            **base,
            "state": "DATA_NEEDED",
            "state_label": state_label("DATA_NEEDED"),
            "next_action": action,
            "next_action_label": action_label,
            "progress_to_tp1": None,
            "position_opportunity_remaining": None,
            "current_r": None,
            "best_price": best_price_for(position, direction, current_price),
            "max_progress_percent": None,
            "pullback_from_best_progress": None,
            "tp1_reached": False,
            "time_in_trade": position_time_in_trade(position, market_snapshot.get("timestamp")),
            "reason_code": issue["code"],
            "triggered_rule": issue["message"],
            "reason": [
                f"Kairos cannot calculate position health because {issue['message'].lower()}.",
                "Add the stock entry price to the journal." if issue["message"] == "Missing underlying entry price" else "Review the original trade-plan fields in the journal.",
            ],
        }

    progress = progress_to_tp1(direction, entry, tp1, current_price)
    current_r = r_multiple(direction, entry, stop, current_price)
    best = best_price_for(position, direction, current_price)
    best_progress = progress_to_tp1(direction, entry, tp1, best)
    max_progress = (best_progress or progress)["raw_percent"]
    pullback = max(0, max_progress - progress["raw_percent"])
    tp1_reached = bool(
        position.get("tp1_reached")
        or position.get("position_tp1_reached")
        or position.get("first_target_touch_at")
        or progress["raw_percent"] >= 100
        or max_progress >= 100
    )
    stop_invalidated = current_price >= stop if direction == "SHORT" else current_price <= stop
    scanner_invalidated = bool(market_snapshot.get("strategy_invalidated") and normalize_direction(market_snapshot.get("direction")) == direction)

    state = "HEALTHY"
    reason_code = "NORMAL_PROGRESS"
    reason = ["Price remains above the original invalidation level.", "The trade is progressing normally toward TP1."]
    if stop_invalidated or scanner_invalidated:
        state = "EXIT"
        reason_code = "STOP_INVALIDATED"
        reason = [f"Price crossed the original stop at ${stop:.2f}.", "The original trade thesis is invalidated."]
    elif tp1_reached:
        state = "PROTECT"
        reason_code = "TP1_REACHED"
        reason = [f"The trade reached TP1 with {round(max_progress)}% maximum progress.", "Consider protecting gains while the thesis remains valid."]
    elif max_progress >= 70 and progress["raw_percent"] <= max_progress - 25:
        state = "PROTECT"
        reason_code = "LARGE_RETRACE_AFTER_70"
        reason = [f"The trade reached {round(max_progress)}% progress and has retraced to {round(progress['raw_percent'])}%.", "Consider protecting gains while the thesis remains valid."]
    elif current_r is not None and current_r < 0:
        state = "WATCH"
        reason_code = "POSITION_BELOW_ENTRY"
        reason = [f"Current R is {format_r(current_r)} while the original stop remains intact.", "Monitor the trade against the original plan."]
    elif max_progress >= 40 and progress["raw_percent"] <= max_progress - 20:
        state = "WATCH"
        reason_code = "RETRACE_AFTER_40"
        reason = [f"Price has pulled back {round(pullback)} percentage points from its best progress.", "The original stop remains intact."]
    elif market_snapshot.get("opposing_confirmation") is True:
        state = "WATCH"
        reason_code = "OPPOSING_CONFIRMATION"
        reason = ["The current scanner read shows opposing confirmation.", "The original stop remains intact."]

    action, action_label = next_action(state)
    return {
        **base,
        "available": True,
        "state": state,
        "state_label": state_label(state),
        "next_action": action,
        "next_action_label": action_label,
        "progress_to_tp1": progress,
        "position_opportunity_remaining": max(0, min(100, 100 - progress["display_percent"])),
        "current_r": current_r,
        "best_price": best,
        "max_progress_percent": max_progress,
        "pullback_from_best_progress": pullback,
        "tp1_reached": tp1_reached,
        "time_in_trade": position_time_in_trade(position, market_snapshot.get("timestamp")),
        "reason_code": reason_code,
        "triggered_rule": reason_code,
        "reason": reason,
    }


def position_time_in_trade(position: dict[str, Any], now: Any = None) -> dict[str, Any]:
    start = first_present(position.get("actual_entry_at"), position.get("position_opened_at"), position.get("tracking_started_at"), position.get("entry_timestamp"), position.get("signal_timestamp"), position.get("createdAt"))
    return {
        "trading_days": count_weekdays_between(start, now or utc_now_iso()) if start else None,
        "expected_min": coerce_number(first_present(position.get("expected_hold_min_days"), position.get("expected_trading_days_low"))),
        "expected_max": coerce_number(first_present(position.get("expected_hold_max_days"), position.get("expected_trading_days_high"))),
    }


def format_r(value: float | None) -> str:
    if value is None:
        return "—"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}R"


def _candle_records(candles: Any) -> list[dict[str, Any]]:
    if candles is None:
        return []
    if isinstance(candles, pd.DataFrame):
        records = []
        for idx, row in candles.iterrows():
            records.append({
                "timestamp": iso_timestamp(idx),
                "open": coerce_number(row.get("Open")),
                "high": coerce_number(row.get("High")),
                "low": coerce_number(row.get("Low")),
                "close": coerce_number(row.get("Close")),
            })
        return records
    return [dict(item) for item in candles if isinstance(item, dict)]


def _coarse_replay_timeframe(timeframe: str | None) -> bool:
    normalized = str(timeframe or "").strip().upper()
    return normalized in {"1D", "D", "DAILY", "1W", "W", "WEEKLY"}


def _in_replay_window(ts: datetime, entry_ts: datetime | None, exit_ts: datetime | None, timeframe: str | None) -> bool:
    if _coarse_replay_timeframe(timeframe):
        day = ts.astimezone(timezone.utc).date()
        if entry_ts and day < entry_ts.astimezone(timezone.utc).date():
            return False
        if exit_ts and day > exit_ts.astimezone(timezone.utc).date():
            return False
        return True
    if entry_ts and ts < entry_ts:
        return False
    if exit_ts and ts > exit_ts:
        return False
    return True


def _timestamp_covered_by_candles(claimed_ts: datetime | None, evaluated_timestamps: list[datetime], timeframe: str | None) -> bool:
    if not claimed_ts or not evaluated_timestamps:
        return False
    if _coarse_replay_timeframe(timeframe):
        claimed_day = claimed_ts.astimezone(timezone.utc).date()
        return any(ts.astimezone(timezone.utc).date() == claimed_day for ts in evaluated_timestamps)
    return max(evaluated_timestamps) >= claimed_ts


def _recorded_target_touch(position: dict[str, Any], target_number: int = 1) -> datetime | None:
    keys = {
        1: ("first_target_touch_at", "tp1_reached_at", "target_hit_at"),
        2: ("second_target_touch_at", "tp2_reached_at"),
        3: ("third_target_touch_at", "tp3_reached_at"),
    }.get(target_number, ())
    return parse_timestamp(first_present(*(position.get(key) for key in keys)))


def _recorded_stop_touch(position: dict[str, Any]) -> datetime | None:
    return parse_timestamp(first_present(position.get("first_stop_touch_at"), position.get("stop_hit_at")))


def _journal_claims_target(position: dict[str, Any], target_number: int = 1) -> bool:
    if _recorded_target_touch(position, target_number):
        return True
    if target_number == 1:
        outcome = str(first_present(position.get("outcome"), position.get("completion_reason"), position.get("result"), "") or "").upper()
        return outcome in {"TP1", "TARGET", "WIN"}
    return False


def _journal_claims_stop(position: dict[str, Any]) -> bool:
    if _recorded_stop_touch(position):
        return True
    outcome = str(first_present(position.get("outcome"), position.get("completion_reason"), position.get("result"), "") or "").upper()
    return outcome in {"STOP", "STOP LOSS", "LOSS", "STOPPED"}


def classify_journal_replay_parity(position: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    status = str(first_present(position.get("result"), position.get("status"), "Open") or "Open")
    if status.lower() == "open":
        return {"status": "UNUSED", "reason": "open position"}
    if not replay:
        return {"status": "MISSING_REPLAY", "reason": "no replay result"}
    if not replay.get("data_complete"):
        return {
            "status": "INSUFFICIENT_REPLAY_DATA",
            "reason": "; ".join(replay.get("data_gaps") or ["replay data incomplete"]),
        }

    journal_target = _journal_claims_target(position, 1)
    journal_stop = _journal_claims_stop(position)
    replay_target = bool(replay.get("tp1_timestamp"))
    replay_stop = bool(replay.get("stop_timestamp"))
    if journal_target and replay_target:
        return {"status": "MATCH", "reason": "journal TP1 outcome reproduced by replay"}
    if journal_stop and replay_stop:
        return {"status": "MATCH", "reason": "journal stop outcome reproduced by replay"}
    if journal_target and not replay_target:
        return {"status": "JOURNAL_EVENT_UNSUPPORTED", "reason": "journal TP1 outcome was not reproduced by replay candles"}
    if journal_stop and not replay_stop:
        return {"status": "JOURNAL_EVENT_UNSUPPORTED", "reason": "journal stop outcome was not reproduced by replay candles"}
    return {"status": "NEEDS_INVESTIGATION", "reason": "completed journal outcome has no comparable replay event"}


def _event_id(position_id: str, event_type: str, timestamp: str | None, suffix: str = "") -> str:
    return ":".join([str(position_id or ""), event_type, str(timestamp or ""), str(suffix or "")])


def _timeline_event(position: dict[str, Any], event_type: str, timestamp: str | None, intel: dict[str, Any], previous_state: str | None = None, suffix: str = "") -> dict[str, Any]:
    progress = intel.get("progress_to_tp1") or {}
    return {
        "event_id": _event_id(str(position.get("position_id") or position.get("journal_id") or ""), event_type, timestamp, suffix or intel.get("state")),
        "timestamp": timestamp,
        "event_type": event_type,
        "previous_state": previous_state,
        "new_state": intel.get("state"),
        "price": intel.get("current_price"),
        "progress_to_tp1": progress.get("raw_percent"),
        "current_r": intel.get("current_r"),
        "best_progress": intel.get("max_progress_percent"),
        "pullback_from_best": intel.get("pullback_from_best_progress"),
        "reason_code": suffix or intel.get("reason_code"),
        "reason_text": " ".join(intel.get("reason") or []),
        "candle_timestamp": timestamp,
    }


def replay_position_intelligence(position: dict[str, Any], candles: Any, provider: str = "test", timeframe: str | None = None, timeframe_source: str | None = None) -> dict[str, Any]:
    original = dict(position)
    records = _candle_records(candles)
    entry_ts = parse_timestamp(first_present(position.get("entry_timestamp"), position.get("tracking_started_at"), position.get("signal_timestamp"), position.get("created_at")))
    exit_ts = parse_timestamp(first_present(position.get("exit_timestamp"), position.get("tracking_completed_at")))
    tf = timeframe or str(first_present(position.get("scanner_timeframe"), position.get("timeframe"), DEFAULT_REPLAY_TIMEFRAME)).upper()
    tf_source = timeframe_source or ("journaled" if first_present(position.get("scanner_timeframe"), position.get("timeframe")) else "inferred_default")
    direction = normalize_direction(first_present(position.get("direction"), position.get("option_type"), position.get("optionType")))
    entry_price = position_entry_price(position)
    stop = position_stop_price(position)
    targets = {1: position_tp_price(position, 1), 2: position_tp_price(position, 2), 3: position_tp_price(position, 3)}

    data_gaps = []
    if not position.get("position_id"):
        data_gaps.append("missing position ID")
    if not entry_ts:
        data_gaps.append("missing entry timestamp")
    if not direction:
        data_gaps.append("missing direction")
    if entry_price is None:
        data_gaps.append("missing underlying entry")
    if stop is None:
        data_gaps.append("missing stop")
    if targets[1] is None:
        data_gaps.append("missing TP1")
    if not records:
        data_gaps.append("missing candles")

    input_timestamps = []
    filtered = []
    rejected_candles = 0
    for record in records:
        ts = parse_timestamp(record.get("timestamp") or record.get("Datetime") or record.get("Date"))
        if not ts:
            rejected_candles += 1
            continue
        input_timestamps.append(ts)
        if not _in_replay_window(ts, entry_ts, exit_ts, tf):
            rejected_candles += 1
            continue
        filtered.append((ts, record))
    filtered.sort(key=lambda item: item[0])
    if records and not filtered:
        data_gaps.append("insufficient history after entry")

    timeline = []
    states_seen = []
    durations = {state: {"candle_count": 0, "elapsed_market_seconds": 0, "percent_of_candles": 0} for state in STATES}
    milestones_seen: set[str] = set()
    previous_state = None
    first_watch = first_protect = first_exit = None
    target_timestamps = {1: None, 2: None, 3: None}
    stop_timestamp = None
    ambiguous = False
    mfe = mae = max_r = min_r = max_progress = final_progress = final_r = None
    best_price = entry_price
    last_ts = None
    evaluated_timestamps = []

    for idx, (ts, candle) in enumerate(filtered):
        evaluated_timestamps.append(ts)
        timestamp = ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        high = coerce_number(first_present(candle.get("high"), candle.get("High")))
        low = coerce_number(first_present(candle.get("low"), candle.get("Low")))
        close = coerce_number(first_present(candle.get("close"), candle.get("Close")))
        if high is None or low is None or close is None:
            data_gaps.append(f"missing candle OHLC at {timestamp}")
            continue
        favorable = low if direction == "SHORT" else high
        adverse = high if direction == "SHORT" else low
        best_price = favorable if best_price is None else (min(best_price, favorable) if direction == "SHORT" else max(best_price, favorable))
        replay_position = _point_in_time_replay_position(original, best_price, bool(target_timestamps[1]))
        intel = build_position_intelligence(replay_position, {"current_price": close, "timestamp": timestamp})
        state = intel.get("state") or "DATA_NEEDED"
        states_seen.append(state)
        durations[state]["candle_count"] += 1
        if last_ts is not None:
            durations[previous_state or state]["elapsed_market_seconds"] += max(0, int((ts - last_ts).total_seconds()))
        last_ts = ts

        if idx == 0:
            timeline.append(_timeline_event(position, "ENTRY", timestamp, intel, None, "ENTRY"))
        elif previous_state and previous_state != state:
            timeline.append(_timeline_event(position, "STATE_TRANSITION", timestamp, intel, previous_state))
            if state == "WATCH" and not first_watch:
                first_watch = timestamp
            if state == "PROTECT" and not first_protect:
                first_protect = timestamp
            if state == "EXIT" and not first_exit:
                first_exit = timestamp

        progress = progress_to_tp1(direction, entry_price, targets[1], favorable)
        current_progress = intel.get("progress_to_tp1") or {}
        current_r = intel.get("current_r")
        extreme_r = r_multiple(direction, entry_price, stop, favorable)
        adverse_r = r_multiple(direction, entry_price, stop, adverse)
        if progress:
            max_progress = progress["raw_percent"] if max_progress is None else max(max_progress, progress["raw_percent"])
            for level in MILESTONE_LEVELS:
                key = f"PROGRESS_{level}"
                if progress["raw_percent"] >= level and key not in milestones_seen:
                    milestones_seen.add(key)
                    timeline.append(_timeline_event(position, key, timestamp, intel, previous_state, key))
        for target_number, target in targets.items():
            if target is None or target_timestamps[target_number]:
                continue
            touched = favorable <= target if direction == "SHORT" else favorable >= target
            if touched:
                target_timestamps[target_number] = timestamp
                event_type = f"TP{target_number}_REACHED"
                timeline.append(_timeline_event(position, event_type, timestamp, intel, previous_state, event_type))
        stop_touched = False
        if stop is not None and direction:
            stop_touched = adverse >= stop if direction == "SHORT" else adverse <= stop
            if stop_touched and not stop_timestamp:
                stop_timestamp = timestamp
                timeline.append(_timeline_event(position, "STOP_INVALIDATED", timestamp, intel, previous_state, "STOP_INVALIDATED"))
        if stop_touched and any(target_timestamps.values()):
            ambiguous = True
            timeline.append(_timeline_event(position, "AMBIGUOUS_TARGET_STOP", timestamp, intel, previous_state, "AMBIGUOUS_TARGET_STOP"))
        if exit_ts and timestamp == exit_ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"):
            timeline.append(_timeline_event(position, "EXIT_RECORDED", timestamp, intel, previous_state, "EXIT_RECORDED"))

        if state == "DATA_NEEDED" and not any(event["event_type"] == "DATA_NEEDED" for event in timeline):
            timeline.append(_timeline_event(position, "DATA_NEEDED", timestamp, intel, previous_state, "DATA_NEEDED"))
        if extreme_r is not None:
            max_r = extreme_r if max_r is None else max(max_r, extreme_r)
            mfe = extreme_r if mfe is None else max(mfe, extreme_r)
        if adverse_r is not None:
            min_r = adverse_r if min_r is None else min(min_r, adverse_r)
            mae = adverse_r if mae is None else min(mae, adverse_r)
        final_progress = current_progress.get("raw_percent")
        final_r = current_r
        previous_state = state

    for target_number in (1, 2, 3):
        claimed_ts = _recorded_target_touch(original, target_number)
        if claimed_ts and not target_timestamps[target_number] and not _timestamp_covered_by_candles(claimed_ts, evaluated_timestamps, tf):
            data_gaps.append(f"insufficient candles through recorded TP{target_number} timestamp")
    claimed_stop_ts = _recorded_stop_touch(original)
    if claimed_stop_ts and not stop_timestamp and not _timestamp_covered_by_candles(claimed_stop_ts, evaluated_timestamps, tf):
        data_gaps.append("insufficient candles through recorded stop timestamp")

    total_candles = len(states_seen)
    for state in STATES:
        if total_candles:
            durations[state]["percent_of_candles"] = durations[state]["candle_count"] / total_candles * 100

    transitions = [event for event in timeline if event["event_type"] == "STATE_TRANSITION"]
    rapid_reversal_count = _rapid_reversal_count(states_seen)
    same_day_changes = _same_day_state_changes(transitions)
    changes_per_10 = (len(transitions) / total_candles * 10) if total_candles else 0
    high_churn = _high_churn(transitions, total_candles)
    category, unsupported = outcome_category(target_timestamps, stop_timestamp, states_seen, high_churn, rapid_reversal_count, ambiguous, data_gaps, transitions)
    result = {
        "replay_version": POSITION_REPLAY_VERSION,
        "position_intelligence_version": POSITION_INTELLIGENCE_VERSION,
        "synthetic": bool(position.get("synthetic")),
        "journal_id": position.get("journal_id"),
        "position_id": position.get("position_id"),
        "ticker": str(position.get("ticker") or "").upper(),
        "direction": direction,
        "grade": first_present(position.get("setup_grade"), position.get("setupGrade"), position.get("grade")),
        "status": first_present(position.get("status"), position.get("result")),
        "entry_timestamp": iso_timestamp(entry_ts),
        "exit_timestamp": iso_timestamp(exit_ts),
        "recorded_outcome": first_present(position.get("result"), position.get("outcome"), position.get("completion_reason")),
        "entry_price": entry_price,
        "stop": stop,
        "targets": {"tp1": targets[1], "tp2": targets[2], "tp3": targets[3]},
        "timeframe": tf,
        "timeframe_source": tf_source,
        "provider": provider,
        "candle_input_count": len(records),
        "candles_accepted": len(filtered),
        "candles_rejected": rejected_candles,
        "first_input_candle_timestamp": iso_timestamp(min(input_timestamps)) if input_timestamps else None,
        "last_input_candle_timestamp": iso_timestamp(max(input_timestamps)) if input_timestamps else None,
        "first_evaluated_candle_timestamp": iso_timestamp(filtered[0][0]) if filtered else None,
        "last_evaluated_candle_timestamp": iso_timestamp(filtered[-1][0]) if filtered else None,
        "replay_window_filter": "date_inclusive" if _coarse_replay_timeframe(tf) else "timestamp_inclusive",
        "candles_evaluated": total_candles,
        "data_complete": not data_gaps,
        "data_gaps": sorted(set(data_gaps)),
        "outcome_order_ambiguous": ambiguous,
        "final_state": previous_state or "DATA_NEEDED",
        "final_progress": final_progress,
        "final_r": final_r,
        "maximum_progress": max_progress,
        "maximum_r": max_r,
        "minimum_r": min_r,
        "MFE": mfe,
        "MAE": mae,
        "state_transition_count": len(transitions),
        "state_changes_per_10_candles": changes_per_10,
        "same_day_state_changes": same_day_changes,
        "rapid_reversal_count": rapid_reversal_count,
        "watch_recovery_count": _recovery_count(states_seen, "WATCH", "HEALTHY"),
        "protect_recovery_count": _recovery_count(states_seen, "PROTECT", "WATCH"),
        "high_churn": high_churn,
        "time_in_each_state": durations,
        "first_watch_timestamp": first_watch,
        "first_protect_timestamp": first_protect,
        "first_exit_timestamp": first_exit,
        "tp1_timestamp": target_timestamps[1],
        "tp2_timestamp": target_timestamps[2],
        "tp3_timestamp": target_timestamps[3],
        "stop_timestamp": stop_timestamp,
        "timeline": _dedupe_timeline(timeline),
        "outcome_category": category,
    }
    if unsupported:
        result["unsupported_reason"] = unsupported
    return result


def _dedupe_timeline(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []
    for event in events:
        event_id = event.get("event_id") or str(uuid.uuid5(uuid.NAMESPACE_URL, repr(event)))
        if event_id in seen:
            continue
        seen.add(event_id)
        event["event_id"] = event_id
        output.append(event)
    return output


def _rapid_reversal_count(states: list[str]) -> int:
    count = 0
    for index in range(len(states) - 2):
        if states[index] == states[index + 2] and states[index] != states[index + 1]:
            count += 1
    return count


def _same_day_state_changes(transitions: list[dict[str, Any]]) -> int:
    counts = Counter((event.get("timestamp") or "")[:10] for event in transitions)
    return sum(count for day, count in counts.items() if day)


def _high_churn(transitions: list[dict[str, Any]], candle_count: int) -> bool:
    if not transitions:
        return False
    same_day_counts = Counter((event.get("timestamp") or "")[:10] for event in transitions)
    if any(count > 2 for count in same_day_counts.values()):
        return True
    return candle_count <= 10 and len(transitions) > 3


def _recovery_count(states: list[str], from_state: str, to_state: str) -> int:
    return sum(1 for prev, cur in zip(states, states[1:]) if prev == from_state and cur == to_state)


def outcome_category(targets: dict[int, str | None], stop_timestamp: str | None, states: list[str], high_churn: bool, rapid_reversal_count: int, ambiguous: bool, data_gaps: list[str], transitions: list[dict[str, Any]]) -> tuple[str, str | None]:
    if data_gaps:
        return "DATA_INCOMPLETE", None
    if ambiguous:
        return "AMBIGUOUS", None
    if stop_timestamp and "EXIT" in states:
        return "STOP_DETECTED", None
    if high_churn:
        return "CHURNY_BUT_VALID", None
    if rapid_reversal_count:
        return "EARLY_WARNING", None
    if targets.get(1) and "WATCH" in states and states[-1] == "HEALTHY":
        return "WATCH_RECOVERED", None
    if targets.get(1) and len([event for event in transitions if (event.get("timestamp") or "") <= targets[1]]) <= 2:
        return "CALM_WINNER", None
    if "PROTECT" in states and (targets.get(1) or stop_timestamp):
        return "PROTECT_USEFUL", None
    return "UNCLASSIFIED", "no_existing_material_adverse_move_definition"


def aggregate_replays(replays: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(replays)
    complete = [r for r in replays if r.get("data_complete")]
    entered_watch = [r for r in replays if r.get("first_watch_timestamp")]
    entered_protect = [r for r in replays if r.get("first_protect_timestamp")]
    entered_exit = [r for r in replays if r.get("first_exit_timestamp")]
    transitions = [r.get("state_transition_count") or 0 for r in replays]
    category_counts = Counter(r.get("outcome_category") or "UNCLASSIFIED" for r in replays)

    def rate(numerator: int, denominator: int) -> dict[str, Any]:
        return {"count": numerator, "sample_size": denominator, "percent": (numerator / denominator * 100) if denominator else None}

    def average(values: list[Any]) -> float | None:
        nums = [coerce_number(v) for v in values]
        nums = [n for n in nums if n is not None]
        return sum(nums) / len(nums) if nums else None

    return {
        "positions_replayed": total,
        "open_positions_replayed": sum(1 for r in replays if str(r.get("status") or "").lower() == "open"),
        "closed_positions_replayed": sum(1 for r in replays if str(r.get("status") or "").lower() != "open"),
        "complete_replays": len(complete),
        "incomplete_replays": total - len(complete),
        "ambiguous_replays": sum(1 for r in replays if r.get("outcome_order_ambiguous")),
        "percent_never_left_healthy": rate(sum(1 for r in replays if not r.get("state_transition_count") and r.get("final_state") == "HEALTHY"), total),
        "percent_entered_watch": rate(len(entered_watch), total),
        "watch_recovery_rate": rate(sum(1 for r in replays if (r.get("watch_recovery_count") or 0) > 0), len(entered_watch)),
        "percent_entered_protect": rate(len(entered_protect), total),
        "protect_to_tp1_rate": rate(sum(1 for r in entered_protect if r.get("tp1_timestamp")), len(entered_protect)),
        "protect_then_stop_rate": rate(sum(1 for r in entered_protect if r.get("stop_timestamp")), len(entered_protect)),
        "percent_entered_exit": rate(len(entered_exit), total),
        "stop_detection_rate": rate(sum(1 for r in replays if r.get("stop_timestamp") and r.get("first_exit_timestamp")), sum(1 for r in replays if r.get("stop_timestamp"))),
        "average_transitions_per_trade": average(transitions),
        "median_transitions_per_trade": statistics.median(transitions) if transitions else None,
        "average_state_changes_per_10_candles": average([r.get("state_changes_per_10_candles") for r in replays]),
        "rapid_reversal_rate": rate(sum(1 for r in replays if (r.get("rapid_reversal_count") or 0) > 0), total),
        "high_churn_rate": rate(sum(1 for r in replays if r.get("high_churn")), total),
        "average_maximum_r": average([r.get("maximum_r") for r in replays]),
        "average_minimum_r": average([r.get("minimum_r") for r in replays]),
        "average_MFE": average([r.get("MFE") for r in replays]),
        "average_MAE": average([r.get("MAE") for r in replays]),
        "state_distribution": dict(Counter(r.get("final_state") or "DATA_NEEDED" for r in replays)),
        "outcome_category_distribution": dict(category_counts),
        "segments": {
            "direction": _segment_counts(replays, "direction"),
            "grade": _segment_counts(replays, "grade"),
            "recorded_outcome": _segment_counts(replays, "recorded_outcome"),
            "status": _segment_counts(replays, "status"),
            "outcome_category": dict(category_counts),
        },
    }


def evidence_guard(aggregate: dict[str, Any]) -> dict[str, Any]:
    closed_complete = int(aggregate.get("closed_complete_real_replays") or 0)
    return {
        "minimum_closed_complete": REAL_EVIDENCE_MIN_CLOSED_COMPLETE,
        "closed_complete_real_replays": closed_complete,
        "message": "Evidence sample is still developing. No threshold recommendations should be made." if closed_complete < REAL_EVIDENCE_MIN_CLOSED_COMPLETE else "",
    }


def real_evidence_counts(replays: list[dict[str, Any]]) -> dict[str, Any]:
    real = [replay for replay in replays if not replay.get("synthetic")]
    closed = [replay for replay in real if str(replay.get("status") or "").lower() != "open"]
    complete = [replay for replay in real if replay.get("data_complete")]
    closed_complete = [replay for replay in closed if replay.get("data_complete")]
    return {
        "real_positions_replayed": len(real),
        "closed_real_positions_replayed": len(closed),
        "complete_real_replays": len(complete),
        "closed_complete_real_replays": len(closed_complete),
        "incomplete_real_replays": len(real) - len(complete),
        "ambiguous_real_replays": sum(1 for replay in real if replay.get("outcome_order_ambiguous")),
    }


def evidence_log_from_replays(replays: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    seen = set()

    def add(replay: dict[str, Any], observation_type: str, observation: str, metrics: dict[str, Any]):
        key = f"{replay.get('position_id')}|{observation_type}|{replay.get('replay_version') or POSITION_REPLAY_VERSION}"
        if key in seen:
            return
        seen.add(key)
        output.append({
            "evidence_id": str(uuid.uuid5(uuid.NAMESPACE_URL, key)),
            "position_id": replay.get("position_id"),
            "journal_id": replay.get("journal_id"),
            "ticker": replay.get("ticker"),
            "direction": replay.get("direction"),
            "observation_type": observation_type,
            "observation": observation,
            "supporting_replay_metrics": metrics,
            "generated_at": utc_now_iso(),
            "replay_version": replay.get("replay_version") or POSITION_REPLAY_VERSION,
        })

    for replay in replays:
        metrics = {
            "final_state": replay.get("final_state"),
            "outcome_category": replay.get("outcome_category"),
            "maximum_r": replay.get("maximum_r"),
            "maximum_progress": replay.get("maximum_progress"),
            "state_transition_count": replay.get("state_transition_count"),
        }
        if not replay.get("data_complete") or replay.get("data_gaps"):
            add(replay, "DATA_GAP", "Replay data incomplete.", {**metrics, "data_gaps": replay.get("data_gaps") or []})
        if replay.get("watch_recovery_count"):
            add(replay, "WATCH_RECOVERY", "WATCH recovered to a calmer state.", {**metrics, "watch_recovery_count": replay.get("watch_recovery_count")})
        if replay.get("stop_timestamp"):
            add(replay, "STOP_DETECTED", "Original stop invalidation was detected by replay.", {**metrics, "stop_timestamp": replay.get("stop_timestamp")})
        if replay.get("high_churn"):
            add(replay, "HIGH_CHURN", "Replay changed states frequently.", {**metrics, "state_changes_per_10_candles": replay.get("state_changes_per_10_candles")})
        if replay.get("outcome_order_ambiguous"):
            add(replay, "AMBIGUOUS_CANDLE", "Target and stop order was ambiguous inside a candle.", metrics)
        if replay.get("recorded_outcome") and replay.get("outcome_category") == "DATA_INCOMPLETE":
            add(replay, "REPLAY_DISCREPANCY", "Recorded outcome exists but replay data is incomplete.", metrics)
    return output


def _segment_counts(replays: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(str(r.get(key) or "unknown") for r in replays))
