from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from position_intelligence import coerce_number, first_present

VERIFIED_ANALYTICS_VERSION = "verified-analytics-v1"


def _result(value: Any) -> str:
    return str(value or "").strip()


def _upper(value: Any) -> str:
    return _result(value).upper()


def _is_open(entry: dict[str, Any]) -> bool:
    return _upper(first_present(entry.get("result"), entry.get("status"), "Open")) == "OPEN"


def _is_completed(entry: dict[str, Any]) -> bool:
    return not _is_open(entry) and bool(first_present(entry.get("result"), entry.get("outcome"), entry.get("completion_reason")))


def _journal_outcome(entry: dict[str, Any]) -> str:
    return _result(first_present(entry.get("outcome"), entry.get("completion_reason"), entry.get("reviewResult"), entry.get("result"), ""))


def _replay_result_and_outcome(replay: dict[str, Any] | None) -> tuple[str | None, str | None]:
    if not replay:
        return None, None
    if replay.get("tp1_timestamp"):
        return "Win", "TP1_REACHED"
    if replay.get("stop_timestamp") or _upper(replay.get("outcome_category")) == "STOP_DETECTED":
        return "Loss", "STOP_DETECTED"
    if not replay.get("data_complete"):
        return None, "DATA_INCOMPLETE"
    return None, _result(replay.get("outcome_category") or "UNCLASSIFIED")


def analytics_verification(entry: dict[str, Any], replay: dict[str, Any] | None = None, verified_at: str | None = None) -> dict[str, Any]:
    if not _is_completed(entry):
        return {
            "status": "NOT_APPLICABLE",
            "journal_result": first_present(entry.get("result"), entry.get("status")),
            "journal_outcome": _journal_outcome(entry),
            "replay_result": None,
            "replay_outcome": None,
            "replay_data_complete": None,
            "parity_detail": None,
            "verified_at": verified_at,
            "version": VERIFIED_ANALYTICS_VERSION,
        }
    if not replay:
        status = "REPLAY_PENDING"
        parity = None
    else:
        parity = replay.get("journal_replay_parity") or {}
        detail = _upper(parity.get("status"))
        if not replay.get("data_complete"):
            status = "INSUFFICIENT_REPLAY_DATA"
        elif detail == "MATCH":
            status = "VERIFIED"
        elif detail in {"JOURNAL_EVENT_UNSUPPORTED", "NEEDS_INVESTIGATION"}:
            status = "JOURNAL_REPLAY_MISMATCH"
        elif detail == "UNUSED":
            status = "NOT_APPLICABLE"
        else:
            status = "REPLAY_PENDING"
    replay_result, replay_outcome = _replay_result_and_outcome(replay)
    return {
        "status": status,
        "journal_result": first_present(entry.get("result"), entry.get("status")),
        "journal_outcome": _journal_outcome(entry),
        "replay_result": replay_result,
        "replay_outcome": replay_outcome,
        "replay_data_complete": replay.get("data_complete") if replay else None,
        "parity_detail": (parity or {}).get("status") if parity else None,
        "parity_reason": (parity or {}).get("reason") if parity else None,
        "verified_at": verified_at,
        "version": VERIFIED_ANALYTICS_VERSION,
    }


def _option_premium_pnl(entry: dict[str, Any]) -> float | None:
    entry_premium = coerce_number(first_present(
        entry.get("actual_option_premium"),
        entry.get("premium_paid"),
        entry.get("entry_premium"),
        entry.get("askAtSelection"),
    ))
    exit_premium = coerce_number(first_present(
        entry.get("actual_exit_premium"),
        entry.get("exit_option_premium"),
        entry.get("option_exit_premium"),
        entry.get("exit_premium"),
        entry.get("premium_received"),
    ))
    quantity = coerce_number(first_present(entry.get("actual_quantity"), entry.get("contracts"), 1))
    if entry_premium is None or exit_premium is None or quantity is None:
        return None
    return (exit_premium - entry_premium) * quantity * 100


def _underlying_plan_pnl(entry: dict[str, Any]) -> float | None:
    entry_price = coerce_number(first_present(
        entry.get("actual_underlying_entry"),
        entry.get("planned_underlying_entry"),
        entry.get("entry_price"),
        entry.get("entry"),
    ))
    exit_price = coerce_number(first_present(entry.get("exit_price"), entry.get("exit")))
    quantity = coerce_number(first_present(entry.get("actual_quantity"), entry.get("contracts"), 1))
    direction = _upper(first_present(entry.get("direction"), entry.get("option_type"), entry.get("optionType")))
    if entry_price is None or exit_price is None or quantity is None:
        return None
    move = exit_price - entry_price if direction in {"LONG", "CALL"} else entry_price - exit_price
    return move * quantity * 100


def pnl_taxonomy(entry: dict[str, Any]) -> dict[str, Any]:
    actual = _option_premium_pnl(entry)
    underlying = _underlying_plan_pnl(entry)
    manual = coerce_number(first_present(entry.get("manual_realized_pnl"), entry.get("realized_pnl")))
    if actual is not None:
        source = "OPTION_PREMIUM_REALIZED"
        status = "VERIFIED_ACTUAL"
    elif manual is not None:
        source = "MANUAL_REALIZED_PNL"
        status = "MANUAL_ACTUAL"
        actual = manual
    elif underlying is not None:
        source = "UNDERLYING_PLAN_LEVELS"
        status = "JOURNAL_LEVEL_CALCULATION"
    else:
        source = "UNAVAILABLE"
        status = "UNAVAILABLE"
    return {
        "actual_option_pnl": actual,
        "underlying_plan_pnl": underlying,
        "pnl_source": source,
        "pnl_verification_status": status,
    }


def _journal_win(entry: dict[str, Any]) -> bool:
    return _result(entry.get("result")) == "Win"


def _journal_loss(entry: dict[str, Any]) -> bool:
    return _result(entry.get("result")) == "Loss"


def _verification_counts(records: list[dict[str, Any]]) -> Counter:
    return Counter((record.get("analytics_verification") or {}).get("status") or "REPLAY_PENDING" for record in records)


def _rate(wins: int, total: int) -> int | None:
    return round(wins / total * 100) if total else None


def _group_records(records: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        entry = record.get("entry") or {}
        groups[_result(entry.get(key) or "Untracked")].append(record)
    rows = []
    for group_key, items in groups.items():
        rows.append({**_aggregate_records(items), "key": group_key})
    return sorted(rows, key=lambda row: (-row["trades"], -(row.get("underlying_plan_result_total") or 0), row["key"]))


def _aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if _is_completed(record.get("entry") or {})]
    journal_wins = sum(1 for record in completed if _journal_win(record.get("entry") or {}))
    journal_losses = sum(1 for record in completed if _journal_loss(record.get("entry") or {}))
    verified = [record for record in completed if (record.get("analytics_verification") or {}).get("status") == "VERIFIED"]
    verified_wins = sum(1 for record in verified if _journal_win(record.get("entry") or {}))
    verified_losses = sum(1 for record in verified if _journal_loss(record.get("entry") or {}))
    counts = _verification_counts(completed)
    actual_values = [record["pnl"]["actual_option_pnl"] for record in completed if (record.get("pnl") or {}).get("actual_option_pnl") is not None]
    underlying_values = [record["pnl"]["underlying_plan_pnl"] for record in completed if (record.get("pnl") or {}).get("underlying_plan_pnl") is not None]
    return {
        "trades": len(records),
        "completed": len(completed),
        "verified": len(verified),
        "needs_review": counts.get("JOURNAL_REPLAY_MISMATCH", 0),
        "replay_pending": counts.get("REPLAY_PENDING", 0),
        "insufficient_data": counts.get("INSUFFICIENT_REPLAY_DATA", 0),
        "journal_only": counts.get("JOURNAL_ONLY", 0),
        "journal_wins": journal_wins,
        "journal_losses": journal_losses,
        "journal_win_rate": _rate(journal_wins, journal_wins + journal_losses),
        "verified_wins": verified_wins,
        "verified_losses": verified_losses,
        "verified_win_rate": _rate(verified_wins, verified_wins + verified_losses),
        "actual_option_pnl_total": sum(actual_values) if actual_values else None,
        "underlying_plan_result_total": sum(underlying_values) if underlying_values else None,
        "actual_option_pnl_count": len(actual_values),
        "underlying_plan_pnl_count": len(underlying_values),
        "missing_exit_premium_count": sum(1 for record in completed if (record.get("pnl") or {}).get("actual_option_pnl") is None),
    }


def verified_analytics_snapshot(entries: list[dict[str, Any]], replays: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    replay_by_position = {str(replay.get("position_id")): replay for replay in (replays or [])}
    records = []
    for entry in entries:
        replay = replay_by_position.get(str(entry.get("position_id")))
        verification = analytics_verification(entry, replay, verified_at=started.isoformat().replace("+00:00", "Z"))
        records.append({
            "journal_id": entry.get("journal_id"),
            "position_id": entry.get("position_id"),
            "ticker": _upper(entry.get("ticker")),
            "entry": entry,
            "analytics_verification": verification,
            "pnl": pnl_taxonomy(entry),
            "replay_summary": {
                "final_state": replay.get("final_state") if replay else None,
                "outcome_category": replay.get("outcome_category") if replay else None,
                "tp1_timestamp": replay.get("tp1_timestamp") if replay else None,
                "stop_timestamp": replay.get("stop_timestamp") if replay else None,
                "candles_evaluated": replay.get("candles_evaluated") if replay else None,
                "data_complete": replay.get("data_complete") if replay else None,
                "data_gaps": replay.get("data_gaps") if replay else [],
            },
        })
    aggregate = _aggregate_records(records)
    duration_ms = (datetime.now(timezone.utc) - started).total_seconds() * 1000
    completed = [record for record in records if _is_completed(record.get("entry") or {})]
    return {
        "version": VERIFIED_ANALYTICS_VERSION,
        "generated_at": started.isoformat().replace("+00:00", "Z"),
        "verification_duration_ms": duration_ms,
        "summary": aggregate,
        "records": records,
        "top_tickers": _group_records(records, "ticker")[:8],
        "by_grade": _group_records(records, "setupGrade"),
        "by_direction": _group_records(records, "direction"),
        "diagnostics": {
            "version": VERIFIED_ANALYTICS_VERSION,
            "completed_journal_trades": len(completed),
            "verified": aggregate["verified"],
            "mismatches": aggregate["needs_review"],
            "replay_pending": aggregate["replay_pending"],
            "insufficient_data": aggregate["insufficient_data"],
            "journal_only": aggregate["journal_only"],
            "actual_option_pnl_count": aggregate["actual_option_pnl_count"],
            "underlying_plan_pnl_count": aggregate["underlying_plan_pnl_count"],
            "missing_exit_premium_count": aggregate["missing_exit_premium_count"],
            "verification_generation_duration_ms": duration_ms,
            "affected_tickers": sorted({record["ticker"] for record in completed if (record.get("analytics_verification") or {}).get("status") == "JOURNAL_REPLAY_MISMATCH"}),
        },
    }
