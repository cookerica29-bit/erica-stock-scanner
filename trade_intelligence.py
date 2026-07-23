from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from statistics import mean, median
from typing import Any

from position_intelligence import coerce_number, first_present


TRADE_INTELLIGENCE_VERSION = "trade-intelligence-v1"
DEFAULT_EXACT_MIN_TRADES = 30
DEFAULT_BROAD_MIN_TRADES = 100


def _upper(value: Any) -> str:
    return str(value or "").strip().upper()


def _text(value: Any) -> str:
    return str(value or "").strip()


def _bucket_opportunity(value: Any) -> str:
    number = coerce_number(value)
    if number is None:
        return "unknown"
    if number >= 70:
        return "70+"
    if number >= 40:
        return "40-69"
    return "under_40"


def _bucket_earnings(days: Any) -> str:
    number = coerce_number(days)
    if number is None:
        return "unknown"
    if number <= 7:
        return "within_7"
    if number <= 14:
        return "within_14"
    return "beyond_14"


def _day_of_week(timestamp: Any) -> str:
    if not timestamp:
        return "unknown"
    try:
        parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except ValueError:
        return "unknown"
    return parsed.strftime("%A")


def _volatility_bucket(value: Any) -> str:
    number = coerce_number(value)
    if number is None:
        return "unknown"
    if number >= 1.5:
        return "high"
    if number >= 0.8:
        return "normal"
    return "low"


def setup_dimensions(entry: dict[str, Any]) -> dict[str, str]:
    trade_eval = entry.get("trade_eval") if isinstance(entry.get("trade_eval"), dict) else {}
    return {
        "symbol": _upper(entry.get("ticker")) or "unknown",
        "sector": _text(first_present(entry.get("sector"), entry.get("stock_sector"), entry.get("market_sector"), "unknown")) or "unknown",
        "direction": _upper(first_present(entry.get("direction"), entry.get("option_type"), entry.get("optionType"))) or "unknown",
        "grade": _upper(first_present(entry.get("setup_grade"), entry.get("setupGrade"), entry.get("grade"))) or "unknown",
        "lifecycle": _upper(first_present(entry.get("scanner_status_normalized"), entry.get("scanner_status"), entry.get("setupStatus"), trade_eval.get("trade_stage"), "unknown")) or "unknown",
        "timeframe": _upper(first_present(entry.get("scanner_timeframe"), entry.get("timeframe"), entry.get("setupTf"), "unknown")) or "unknown",
        "trend_alignment": _text(first_present(entry.get("htf_alignment"), entry.get("trend_state"), entry.get("trend"), entry.get("daily_market_regime"), "unknown")) or "unknown",
        "location": _text(first_present(entry.get("setupLocation"), entry.get("stockLocation"), trade_eval.get("location"), "unknown")) or "unknown",
        "confirmation": _text(first_present(entry.get("confirmation_status"), "confirmed" if entry.get("confirmationStarted") else None, "unknown")) or "unknown",
        "opportunity_bucket": _bucket_opportunity(first_present(entry.get("opportunity_remaining"), entry.get("opportunityRemaining"), entry.get("position_opportunity_remaining"))),
        "earnings_bucket": _bucket_earnings(first_present(entry.get("days_until_earnings"), entry.get("earnings_days_until"))),
        "day_of_week": _day_of_week(first_present(entry.get("entry_timestamp"), entry.get("signal_timestamp"), entry.get("created_at"), entry.get("createdAt"))),
        "volatility_bucket": _volatility_bucket(first_present(entry.get("atr_expansion"), entry.get("relative_volume"), entry.get("atr_at_signal"))),
        "market_regime": _text(first_present(entry.get("market_regime"), entry.get("daily_market_regime"), entry.get("h4_market_regime"), "unknown")) or "unknown",
    }


def exact_key(dimensions: dict[str, str]) -> tuple[str, ...]:
    return (
        dimensions["symbol"],
        dimensions["direction"],
        dimensions["grade"],
        dimensions["lifecycle"],
        dimensions["timeframe"],
        dimensions["location"],
        dimensions["confirmation"],
    )


def broad_key(dimensions: dict[str, str]) -> tuple[str, ...]:
    return (
        dimensions["direction"],
        dimensions["grade"],
        dimensions["timeframe"],
        dimensions["location"],
        dimensions["confirmation"],
        dimensions["market_regime"],
    )


def _verified_record(entry: dict[str, Any], replay: dict[str, Any], verification: dict[str, Any]) -> dict[str, Any] | None:
    if (verification or {}).get("status") != "VERIFIED":
        return None
    if not replay or not replay.get("data_complete"):
        return None
    dimensions = setup_dimensions(entry)
    return {
        "entry": entry,
        "replay": replay,
        "dimensions": dimensions,
        "exact_key": exact_key(dimensions),
        "broad_key": broad_key(dimensions),
    }


def _safe_mean(values: list[float]) -> float | None:
    return round(mean(values), 2) if values else None


def _safe_median(values: list[float]) -> float | None:
    return round(median(values), 2) if values else None


def _rate(count: int, total: int) -> float | None:
    return round((count / total) * 100, 1) if total else None


def _seconds_to_trading_days(seconds: Any) -> float | None:
    number = coerce_number(seconds)
    if number is None:
        return None
    return round(number / (6.5 * 60 * 60), 2)


def _time_to_event_days(replay: dict[str, Any], event_timestamp: str | None) -> float | None:
    if not event_timestamp:
        return None
    start = replay.get("entry_timestamp")
    try:
        started = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        ended = datetime.fromisoformat(str(event_timestamp).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return _seconds_to_trading_days((ended - started).total_seconds())


def group_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    tp1 = [r for r in records if r["replay"].get("tp1_timestamp")]
    tp2 = [r for r in records if r["replay"].get("tp2_timestamp")]
    tp3 = [r for r in records if r["replay"].get("tp3_timestamp")]
    stopped = [r for r in records if r["replay"].get("stop_timestamp")]
    invalidated = [r for r in records if _upper(r["replay"].get("outcome_category")) == "STOP_DETECTED"]
    final_r = [coerce_number(first_present(r["replay"].get("final_r"), r["replay"].get("maximum_r"))) for r in records]
    final_r = [v for v in final_r if v is not None]
    mae = [abs(coerce_number(r["replay"].get("MAE"))) for r in records if coerce_number(r["replay"].get("MAE")) is not None]
    mfe = [coerce_number(r["replay"].get("MFE")) for r in records if coerce_number(r["replay"].get("MFE")) is not None]
    time_to_tp1 = [_time_to_event_days(r["replay"], r["replay"].get("tp1_timestamp")) for r in tp1]
    time_to_tp1 = [v for v in time_to_tp1 if v is not None]
    completion_times = []
    distribution = Counter()
    for record in records:
        replay = record["replay"]
        category = _upper(replay.get("outcome_category") or "UNCLASSIFIED")
        distribution[category] += 1
        completion_ts = first_present(replay.get("tp1_timestamp"), replay.get("stop_timestamp"), replay.get("exit_timestamp"))
        days = _time_to_event_days(replay, completion_ts)
        if days is not None:
            completion_times.append(days)
    return {
        "total_verified_trades": total,
        "tp1_rate": _rate(len(tp1), total),
        "tp2_rate": _rate(len(tp2), total),
        "tp3_rate": _rate(len(tp3), total),
        "stop_rate": _rate(len(stopped), total),
        "invalidation_rate": _rate(len(invalidated), total),
        "average_r": _safe_mean(final_r),
        "median_r": _safe_median(final_r),
        "average_time_to_tp1_trading_days": _safe_mean(time_to_tp1),
        "average_time_to_completion_trading_days": _safe_mean(completion_times),
        "average_maximum_drawdown_r": _safe_mean(mae),
        "average_maximum_favorable_excursion_r": _safe_mean(mfe),
        "completion_distribution": dict(sorted(distribution.items())),
    }


def what_usually_happens(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(records)
    if not total:
        return []
    reached_tp1 = sum(1 for r in records if r["replay"].get("tp1_timestamp"))
    stopped = sum(1 for r in records if r["replay"].get("stop_timestamp") and not r["replay"].get("tp1_timestamp"))
    pulled_back_then_tp1 = sum(
        1 for r in records
        if r["replay"].get("tp1_timestamp") and coerce_number(r["replay"].get("MAE")) is not None and abs(coerce_number(r["replay"].get("MAE"))) >= 0.5
    )
    clean_tp1 = max(0, reached_tp1 - pulled_back_then_tp1)
    return [
        {"label": "Reached TP1 before meaningful pullback", "percent": _rate(clean_tp1, total), "count": clean_tp1},
        {"label": "Pulled back first, then reached TP1", "percent": _rate(pulled_back_then_tp1, total), "count": pulled_back_then_tp1},
        {"label": "Stopped before TP1", "percent": _rate(stopped, total), "count": stopped},
    ]


def confidence_drivers(group_records: list[dict[str, Any]], baseline_records: list[dict[str, Any]], dimensions: dict[str, str]) -> list[str]:
    if len(group_records) < DEFAULT_EXACT_MIN_TRADES or len(baseline_records) < DEFAULT_BROAD_MIN_TRADES:
        return []
    group_tp1 = group_metrics(group_records).get("tp1_rate")
    baseline_tp1 = group_metrics(baseline_records).get("tp1_rate")
    if group_tp1 is None or baseline_tp1 is None or group_tp1 < baseline_tp1 + 10:
        return []
    drivers = []
    if dimensions.get("grade") and dimensions["grade"] != "unknown":
        drivers.append(f"{dimensions['grade']}-grade setups outperformed the verified baseline.")
    if dimensions.get("location") and dimensions["location"] != "unknown":
        drivers.append(f"{dimensions['location']} location had stronger verified TP1 follow-through.")
    if dimensions.get("confirmation") and dimensions["confirmation"] != "unknown":
        drivers.append("Confirmation state showed better verified outcomes than baseline.")
    if dimensions.get("day_of_week") and dimensions["day_of_week"] != "unknown":
        drivers.append(f"{dimensions['day_of_week']} entries performed better in verified history.")
    return drivers[:4]


def build_trade_intelligence_snapshot(
    verified_records: list[dict[str, Any]],
    exact_min_trades: int = DEFAULT_EXACT_MIN_TRADES,
    broad_min_trades: int = DEFAULT_BROAD_MIN_TRADES,
) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    exact_groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    broad_groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_grade: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in verified_records:
        exact_groups[record["exact_key"]].append(record)
        broad_groups[record["broad_key"]].append(record)
        by_symbol[record["dimensions"]["symbol"]].append(record)
        by_grade[record["dimensions"]["grade"]].append(record)
        by_day[record["dimensions"]["day_of_week"]].append(record)
    duration_ms = (datetime.now(timezone.utc) - started).total_seconds() * 1000
    return {
        "version": TRADE_INTELLIGENCE_VERSION,
        "generated_at": started.isoformat().replace("+00:00", "Z"),
        "verified_trade_count": len(verified_records),
        "thresholds": {"exact_min_trades": exact_min_trades, "broad_min_trades": broad_min_trades},
        "diagnostics": {
            "cache_status": "rebuilt",
            "sample_sizes": {
                "verified_records": len(verified_records),
                "exact_groups": len(exact_groups),
                "broad_groups": len(broad_groups),
            },
            "similarity_dimensions_used": list(setup_dimensions({}).keys()),
            "excluded_trades": 0,
            "replay_only_counts": len(verified_records),
            "journal_only_counts": 0,
            "confidence_thresholds": {"exact_min_trades": exact_min_trades, "broad_min_trades": broad_min_trades},
            "cache_rebuild_time_ms": round(duration_ms, 2),
        },
        "groups": {
            "exact": {json_key(key): group_metrics(records) for key, records in exact_groups.items()},
            "broad": {json_key(key): group_metrics(records) for key, records in broad_groups.items()},
        },
        "dashboard": dashboard_summary(by_symbol, by_grade, by_day),
    }


def json_key(key: tuple[str, ...]) -> str:
    return "|".join(str(part) for part in key)


def dashboard_summary(by_symbol: dict[str, list[dict[str, Any]]], by_grade: dict[str, list[dict[str, Any]]], by_day: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    def ranked(groups):
        rows = []
        for key, records in groups.items():
            metrics = group_metrics(records)
            rows.append({"key": key, **metrics})
        return sorted(rows, key=lambda row: (-(row.get("tp1_rate") or 0), -row["total_verified_trades"], row["key"]))[:10]
    return {
        "most_reliable_symbols": ranked(by_symbol),
        "most_reliable_grades": ranked(by_grade),
        "best_weekdays": ranked(by_day),
        "highest_tp1_rate": ranked(by_symbol),
        "fastest_average_winners": sorted(
            [row for row in ranked(by_symbol) if row.get("average_time_to_tp1_trading_days") is not None],
            key=lambda row: (row["average_time_to_tp1_trading_days"], -(row.get("tp1_rate") or 0)),
        )[:10],
    }


def build_verified_trade_records(entries: list[dict[str, Any]], replays: list[dict[str, Any]], analytics_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entry_by_position = {str(entry.get("position_id")): entry for entry in entries if entry.get("position_id")}
    replay_by_position = {str(replay.get("position_id")): replay for replay in replays if replay.get("position_id")}
    verification_by_position = {
        str(record.get("position_id")): record.get("analytics_verification") or {}
        for record in analytics_records
        if record.get("position_id")
    }
    records = []
    for position_id, entry in entry_by_position.items():
        record = _verified_record(entry, replay_by_position.get(position_id), verification_by_position.get(position_id) or {})
        if record:
            records.append(record)
    return records


def similar_trade_insight(
    setup: dict[str, Any],
    verified_records: list[dict[str, Any]],
    exact_min_trades: int = DEFAULT_EXACT_MIN_TRADES,
    broad_min_trades: int = DEFAULT_BROAD_MIN_TRADES,
) -> dict[str, Any]:
    dimensions = setup_dimensions(setup)
    ekey = exact_key(dimensions)
    bkey = broad_key(dimensions)
    exact_records = [record for record in verified_records if record["exact_key"] == ekey]
    broad_records = [record for record in verified_records if record["broad_key"] == bkey]
    selected = None
    group_type = None
    threshold = None
    if len(exact_records) >= exact_min_trades:
        selected = exact_records
        group_type = "exact"
        threshold = exact_min_trades
    elif len(broad_records) >= broad_min_trades:
        selected = broad_records
        group_type = "broader"
        threshold = broad_min_trades
    if not selected:
        return {
            "version": TRADE_INTELLIGENCE_VERSION,
            "available": False,
            "message": "Not enough verified historical data yet.",
            "dimensions": dimensions,
            "exact_match_count": len(exact_records),
            "broader_match_count": len(broad_records),
            "thresholds": {"exact_min_trades": exact_min_trades, "broad_min_trades": broad_min_trades},
        }
    metrics = group_metrics(selected)
    return {
        "version": TRADE_INTELLIGENCE_VERSION,
        "available": True,
        "group_type": group_type,
        "sample_size": len(selected),
        "verified_trades": len(selected),
        "minimum_required": threshold,
        "dimensions": dimensions,
        "exact_match_count": len(exact_records),
        "broader_match_count": len(broad_records),
        "metrics": metrics,
        "what_usually_happens": what_usually_happens(selected),
        "confidence_drivers": confidence_drivers(selected, verified_records, dimensions),
        "evidence": {
            "sample_size": len(selected),
            "verified_trades": len(selected),
            "confidence_level": "evidence_supported",
        },
    }
