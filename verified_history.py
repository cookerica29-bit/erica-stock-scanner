from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from position_intelligence import POSITION_REPLAY_VERSION, coerce_number, first_present


VERIFIED_HISTORY_PIPELINE_VERSION = "verified-history-v1"
REPLAY_JOB_VERSION = "verified-history-replay-job-v1"
MAX_REPLAY_ATTEMPTS = 3

PIPELINE_STATES = {
    "OPEN",
    "COMPLETION_PENDING",
    "COMPLETED_AWAITING_REPLAY",
    "REPLAY_QUEUED",
    "REPLAY_RUNNING",
    "REPLAY_RETRY_PENDING",
    "REPLAY_FAILED",
    "REPLAY_DATA_INCOMPLETE",
    "VERIFICATION_PENDING",
    "VERIFIED",
    "NEEDS_REVIEW",
    "NOT_APPLICABLE",
}

REQUIRED_REPLAY_FIELDS = {
    "journal_id": ("journal_id",),
    "position_id": ("position_id",),
    "ticker": ("ticker", "symbol"),
    "direction": ("direction", "option_type", "optionType"),
    "entry_timestamp": ("entry_timestamp", "tracking_started_at", "position_opened_at", "actual_entry_at"),
    "completion_timestamp": ("exit_timestamp", "tracking_completed_at", "completed_at"),
    "entry_price": ("actual_underlying_entry", "planned_underlying_entry", "entry_price", "entry"),
    "stop": ("original_stop", "stop_price", "plannedStop", "stop"),
    "tp1": ("original_tp1", "target_price", "plannedTp1", "tp1"),
    "journal_result": ("result", "outcome", "completion_reason", "reviewResult"),
}

REPLAY_SIGNATURE_FIELDS = {
    "symbol": ("ticker", "symbol"),
    "direction": ("direction", "option_type", "optionType"),
    "timeframe": ("scanner_timeframe", "timeframe", "setupTf"),
    "entry_timestamp": ("entry_timestamp", "tracking_started_at", "position_opened_at", "actual_entry_at"),
    "completion_timestamp": ("exit_timestamp", "tracking_completed_at", "completed_at"),
    "entry_price": ("actual_underlying_entry", "planned_underlying_entry", "entry_price", "entry"),
    "stop": ("original_stop", "stop_price", "plannedStop", "stop"),
    "tp1": ("original_tp1", "target_price", "plannedTp1", "tp1"),
    "tp2": ("original_tp2", "plannedTp2", "tp2"),
    "tp3": ("original_tp3", "plannedTp3", "tp3"),
}

RETRYABLE_ERROR_CODES = {
    "provider_timeout",
    "temporary_dns_failure",
    "rate_limit",
    "temporary_empty_response",
    "network_interruption",
    "database_lock",
    "TimeoutError",
    "ConnectionError",
    "YFRateLimitError",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _text(value: Any) -> str:
    return str(value or "").strip()


def _upper(value: Any) -> str:
    return _text(value).upper()


def _value(entry: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    return first_present(*(entry.get(alias) for alias in aliases))


def _terminal_value(*values: Any) -> Any:
    for value in values:
        if value in (None, ""):
            continue
        if _upper(value) == "OPEN":
            continue
        return value
    return None


def is_open_entry(entry: dict[str, Any]) -> bool:
    result = _upper(first_present(entry.get("result"), entry.get("status"), "Open"))
    tracking = _upper(entry.get("tracking_status"))
    completion = _terminal_value(entry.get("completion_reason"), entry.get("outcome"), entry.get("exit_timestamp"), entry.get("tracking_completed_at"))
    return result == "OPEN" and tracking != "COMPLETED" and not completion


def is_completed_entry(entry: dict[str, Any]) -> bool:
    return not is_open_entry(entry) and bool(_terminal_value(entry.get("result"), entry.get("outcome"), entry.get("completion_reason"), entry.get("tracking_completed_at"), entry.get("exit_timestamp")))


def completion_readiness(entry: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(entry, dict):
        return {
            "ready": False,
            "pipeline_status": "NOT_APPLICABLE",
            "missing_fields": ["journal_record"],
            "invalid_fields": [],
            "message": "Journal record is unavailable.",
        }
    if not is_completed_entry(entry):
        return {
            "ready": False,
            "pipeline_status": "OPEN",
            "missing_fields": [],
            "invalid_fields": [],
            "message": "Trade is still open or not completed.",
        }
    missing = [name for name, aliases in REQUIRED_REPLAY_FIELDS.items() if _value(entry, aliases) in (None, "")]
    invalid = []
    direction = _upper(_value(entry, REQUIRED_REPLAY_FIELDS["direction"]))
    if direction not in {"LONG", "SHORT", "CALL", "PUT"}:
        invalid.append("direction")
    entry_price = coerce_number(_value(entry, REQUIRED_REPLAY_FIELDS["entry_price"]))
    stop = coerce_number(_value(entry, REQUIRED_REPLAY_FIELDS["stop"]))
    tp1 = coerce_number(_value(entry, REQUIRED_REPLAY_FIELDS["tp1"]))
    if entry_price is None:
        invalid.append("entry_price")
    if stop is None:
        invalid.append("stop")
    if tp1 is None:
        invalid.append("tp1")
    normalized_direction = "LONG" if direction == "CALL" else "SHORT" if direction == "PUT" else direction
    if entry_price is not None and stop is not None and tp1 is not None:
        if normalized_direction == "LONG" and not (stop < entry_price < tp1):
            invalid.append("long_stop_entry_tp1_geometry")
        if normalized_direction == "SHORT" and not (tp1 < entry_price < stop):
            invalid.append("short_tp1_entry_stop_geometry")
    ready = not missing and not invalid
    return {
        "ready": ready,
        "pipeline_status": "COMPLETED_AWAITING_REPLAY" if ready else "COMPLETION_PENDING",
        "missing_fields": missing,
        "invalid_fields": invalid,
        "message": "Completed trade is ready for replay." if ready else "Completed trade is missing replay-required evidence.",
    }


def replay_input_material(entry: dict[str, Any]) -> dict[str, Any]:
    material = {name: _value(entry, aliases) for name, aliases in REPLAY_SIGNATURE_FIELDS.items()}
    material["replay_engine_version"] = POSITION_REPLAY_VERSION
    material["replay_job_version"] = REPLAY_JOB_VERSION
    return material


def replay_input_signature(entry: dict[str, Any]) -> str:
    raw = json.dumps(replay_input_material(entry), sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def replay_dedupe_key(entry: dict[str, Any]) -> str:
    return "|".join([
        str(entry.get("journal_id") or ""),
        replay_input_signature(entry),
        str(POSITION_REPLAY_VERSION),
    ])


def retryable_error(error_code: str | None) -> bool:
    return str(error_code or "") in RETRYABLE_ERROR_CODES


def verification_to_pipeline_status(verification: dict[str, Any] | None, replay: dict[str, Any] | None = None) -> str:
    status = _upper((verification or {}).get("status"))
    if status == "VERIFIED":
        return "VERIFIED"
    if status == "JOURNAL_REPLAY_MISMATCH":
        return "NEEDS_REVIEW"
    if status == "INSUFFICIENT_REPLAY_DATA" or (replay and replay.get("data_complete") is False):
        return "REPLAY_DATA_INCOMPLETE"
    if status == "NOT_APPLICABLE":
        return "NOT_APPLICABLE"
    if status == "REPLAY_PENDING":
        return "VERIFICATION_PENDING" if replay else "COMPLETED_AWAITING_REPLAY"
    return "VERIFICATION_PENDING" if replay else "COMPLETED_AWAITING_REPLAY"


def trade_intelligence_eligibility(verification: dict[str, Any] | None, has_required_features: bool = True) -> dict[str, Any]:
    status = _upper((verification or {}).get("status"))
    if status == "VERIFIED" and has_required_features:
        return {"trade_intelligence_eligible": True, "eligibility_reason": "VERIFIED"}
    if status == "VERIFIED":
        return {"trade_intelligence_eligible": False, "eligibility_reason": "MISSING_REQUIRED_FEATURES"}
    reasons = {
        "JOURNAL_REPLAY_MISMATCH": "JOURNAL_REPLAY_MISMATCH",
        "INSUFFICIENT_REPLAY_DATA": "REPLAY_DATA_INCOMPLETE",
        "REPLAY_PENDING": "REPLAY_PENDING",
        "JOURNAL_ONLY": "JOURNAL_ONLY",
        "NOT_APPLICABLE": "NOT_COMPLETED",
    }
    return {"trade_intelligence_eligible": False, "eligibility_reason": reasons.get(status, "REPLAY_PENDING")}


def classify_pipeline_record(
    entry: dict[str, Any],
    replay: dict[str, Any] | None = None,
    verification: dict[str, Any] | None = None,
    job: dict[str, Any] | None = None,
    has_required_features: bool = True,
) -> dict[str, Any]:
    readiness = completion_readiness(entry)
    job_status = _upper((job or {}).get("status"))
    if readiness["pipeline_status"] == "OPEN":
        status = "OPEN"
        explanation = "Trade is still open or not completed."
        next_step = "Complete the trade record when the position is closed."
    elif not readiness["ready"]:
        status = "COMPLETION_PENDING"
        explanation = "Completion details are incomplete."
        next_step = "Add the missing completion fields before replay can run."
    elif job_status == "QUEUED":
        status = "REPLAY_QUEUED"
        explanation = "Replay job is queued."
        next_step = "Wait for the replay worker."
    elif job_status == "RUNNING":
        status = "REPLAY_RUNNING"
        explanation = "Replay job is running."
        next_step = "Wait for replay to finish."
    elif job_status == "RETRY_PENDING":
        status = "REPLAY_RETRY_PENDING"
        explanation = "Replay hit a temporary error and is scheduled to retry."
        next_step = "Retry after the scheduled time or inspect provider health."
    elif job_status == "FAILED":
        status = "REPLAY_FAILED"
        explanation = "Replay failed for a technical reason."
        next_step = "Review the error and retry only after the cause is resolved."
    else:
        status = verification_to_pipeline_status(verification, replay)
        explanations = {
            "COMPLETED_AWAITING_REPLAY": "Completed trade is ready for replay.",
            "REPLAY_DATA_INCOMPLETE": "Replay ran but did not have sufficient chronological market data.",
            "VERIFICATION_PENDING": "Replay exists and is waiting for verification.",
            "VERIFIED": "Journal and replay agree.",
            "NEEDS_REVIEW": "Journal and replay disagree.",
            "NOT_APPLICABLE": "History verification does not apply to this record.",
        }
        next_steps = {
            "COMPLETED_AWAITING_REPLAY": "Queue replay for this completed trade.",
            "REPLAY_DATA_INCOMPLETE": "Retry when historical data coverage improves.",
            "VERIFICATION_PENDING": "Run verification.",
            "VERIFIED": "This record may be used as verified evidence.",
            "NEEDS_REVIEW": "Review the journal result against replay evidence.",
            "NOT_APPLICABLE": "No history action is required.",
        }
        explanation = explanations.get(status, "Pipeline status is pending.")
        next_step = next_steps.get(status, "Review this record.")
    eligibility = trade_intelligence_eligibility(verification, has_required_features=has_required_features)
    return {
        "journal_id": entry.get("journal_id"),
        "position_id": entry.get("position_id"),
        "ticker": _upper(entry.get("ticker")),
        "pipeline_status": status,
        "readiness": readiness,
        "missing_fields": readiness.get("missing_fields") or [],
        "invalid_fields": readiness.get("invalid_fields") or [],
        "explanation": explanation,
        "next_step": next_step,
        "job": _safe_job(job),
        "verification_status": (verification or {}).get("status"),
        "verification": verification,
        "replay_summary": _safe_replay(replay),
        **eligibility,
    }


def _safe_job(job: dict[str, Any] | None) -> dict[str, Any] | None:
    if not job:
        return None
    return {
        "job_id": job.get("job_id"),
        "status": job.get("status"),
        "attempt_count": job.get("attempt_count"),
        "queued_at": job.get("queued_at"),
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "next_attempt_at": job.get("next_attempt_at"),
        "last_error_code": job.get("last_error_code"),
        "replay_signature": job.get("replay_signature"),
    }


def _safe_replay(replay: dict[str, Any] | None) -> dict[str, Any] | None:
    if not replay:
        return None
    return {
        "journal_id": replay.get("journal_id"),
        "position_id": replay.get("position_id"),
        "ticker": replay.get("ticker"),
        "data_complete": replay.get("data_complete"),
        "outcome_category": replay.get("outcome_category"),
        "tp1_timestamp": replay.get("tp1_timestamp"),
        "stop_timestamp": replay.get("stop_timestamp"),
        "candles_evaluated": replay.get("candles_evaluated"),
        "data_gaps": replay.get("data_gaps") or [],
        "journal_replay_parity": replay.get("journal_replay_parity"),
    }


def build_verified_history_snapshot(
    entries: list[dict[str, Any]],
    replays: list[dict[str, Any]],
    analytics_records: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    replay_by_position = {str(replay.get("position_id")): replay for replay in replays or []}
    analytics_by_position = {str(record.get("position_id")): record for record in analytics_records or []}
    latest_job_by_journal = {}
    for job in sorted(jobs or [], key=lambda item: str(item.get("updated_at") or ""), reverse=True):
        latest_job_by_journal.setdefault(str(job.get("journal_id")), job)
    records = []
    for entry in entries or []:
        position_id = str(entry.get("position_id"))
        analytics_record = analytics_by_position.get(position_id) or {}
        verification = analytics_record.get("analytics_verification")
        record = classify_pipeline_record(
            entry,
            replay=replay_by_position.get(position_id),
            verification=verification,
            job=latest_job_by_journal.get(str(entry.get("journal_id"))),
        )
        records.append(record)
    counts = Counter(record["pipeline_status"] for record in records)
    completed_records = [record for record in records if record["pipeline_status"] != "OPEN"]
    processing_states = {"COMPLETED_AWAITING_REPLAY", "REPLAY_QUEUED", "REPLAY_RUNNING", "REPLAY_RETRY_PENDING", "REPLAY_DATA_INCOMPLETE", "VERIFICATION_PENDING"}
    needs_attention_states = {"COMPLETION_PENDING", "REPLAY_FAILED", "NEEDS_REVIEW"}
    unclassified = [record for record in records if record["pipeline_status"] not in PIPELINE_STATES]
    return {
        "version": VERIFIED_HISTORY_PIPELINE_VERSION,
        "generated_at": utc_now_iso(),
        "records": records,
        "summary": {
            "journal_records_inspected": len(entries or []),
            "open": counts.get("OPEN", 0),
            "completion_pending": counts.get("COMPLETION_PENDING", 0),
            "completed_awaiting_replay": counts.get("COMPLETED_AWAITING_REPLAY", 0),
            "queued": counts.get("REPLAY_QUEUED", 0),
            "running": counts.get("REPLAY_RUNNING", 0),
            "retry_pending": counts.get("REPLAY_RETRY_PENDING", 0),
            "failed": counts.get("REPLAY_FAILED", 0),
            "replay_incomplete": counts.get("REPLAY_DATA_INCOMPLETE", 0),
            "verification_pending": counts.get("VERIFICATION_PENDING", 0),
            "verified": counts.get("VERIFIED", 0),
            "needs_review": counts.get("NEEDS_REVIEW", 0),
            "not_applicable": counts.get("NOT_APPLICABLE", 0),
            "trade_intelligence_eligible": sum(1 for record in records if record.get("trade_intelligence_eligible")),
            "open_bucket": counts.get("OPEN", 0),
            "processing_bucket": sum(counts.get(state, 0) for state in processing_states),
            "needs_attention_bucket": sum(counts.get(state, 0) for state in needs_attention_states),
            "verified_bucket": counts.get("VERIFIED", 0),
        },
        "reconciliation": {
            "journal_reconciled": len(records) == len(entries or []),
            "completed_reconciled": len(completed_records) == sum(counts.get(state, 0) for state in PIPELINE_STATES if state != "OPEN"),
            "unclassified_record_count": len(unclassified),
            "unclassified_record_ids": [record.get("journal_id") for record in unclassified],
        },
        "diagnostics": {
            "pipeline_version": VERIFIED_HISTORY_PIPELINE_VERSION,
            "jobs_total": len(jobs or []),
            "duplicate_active_job_count": duplicate_active_job_count(jobs or []),
            "replay_jobs_by_status": dict(sorted(Counter(_upper(job.get("status")) for job in jobs or []).items())),
        },
    }


def duplicate_active_job_count(jobs: list[dict[str, Any]]) -> int:
    active = [job for job in jobs if _upper(job.get("status")) in {"QUEUED", "RUNNING", "RETRY_PENDING"}]
    counts = Counter((job.get("journal_id"), job.get("replay_signature")) for job in active)
    return sum(1 for count in counts.values() if count > 1)


def backfill_preview(entries: list[dict[str, Any]], jobs: list[dict[str, Any]]) -> dict[str, Any]:
    active_signatures = {str(job.get("replay_signature")) for job in jobs or [] if _upper(job.get("status")) in {"QUEUED", "RUNNING", "RETRY_PENDING", "COMPLETED"}}
    rows = []
    counts = Counter()
    for entry in entries or []:
        readiness = completion_readiness(entry)
        signature = replay_input_signature(entry) if readiness.get("ready") else None
        if not is_completed_entry(entry):
            status = "NOT_APPLICABLE"
        elif not readiness.get("ready"):
            status = "MISSING_DATA"
        elif signature in active_signatures:
            status = "ALREADY_PROCESSED"
        else:
            status = "SAFE_TO_BACKFILL"
        counts[status] += 1
        rows.append({
            "journal_id": entry.get("journal_id"),
            "position_id": entry.get("position_id"),
            "ticker": _upper(entry.get("ticker")),
            "backfill_status": status,
            "missing_fields": readiness.get("missing_fields") or [],
            "invalid_fields": readiness.get("invalid_fields") or [],
            "replay_signature": signature,
        })
    return {
        "version": VERIFIED_HISTORY_PIPELINE_VERSION,
        "created_jobs": 0,
        "counts": dict(sorted(counts.items())),
        "records": rows,
    }
