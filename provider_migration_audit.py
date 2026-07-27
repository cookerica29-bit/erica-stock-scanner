"""Developer-only Alpaca migration audit and comparison helpers.

This module is diagnostic only. It does not change provider routing, scanner
strategy, journal behavior, replay logic, or production market-data defaults.
"""

from __future__ import annotations

import os
import time
from collections import Counter
from datetime import datetime, time as datetime_time, timedelta, timezone
from typing import Any

import market_data
import pandas as pd


AUDIT_VERSION = "alpaca-migration-audit-v1"
FOUR_H_FORENSIC_VERSION = "4h-market-data-forensics-v1"
RAW_DISCREPANCY_CLASSES = {
    "MATCH",
    "EXPECTED_PROVIDER_VARIATION",
    "HARMLESS_DIFFERENCE",
    "DATA_QUALITY_CONCERN",
    "STRATEGY_SIGNIFICANT",
    "IMPLEMENTATION_DEFECT",
    "UNRESOLVED",
}
REPRESENTATIVE_SYMBOLS = [
    "SPY",
    "QQQ",
    "AAPL",
    "MSFT",
    "NVDA",
    "TSLA",
    "AMZN",
    "GOOGL",
    "META",
    "KO",
    "JNJ",
    "XOM",
    "OXY",
    "DVN",
    "DOW",
    "UNP",
    "ENB",
    "CMG",
    "WMT",
    "BAC",
]
TIMEFRAME_REQUESTS = {
    "30M": {"period": "60d", "interval": "30m"},
    "4H": {"period": "60d", "interval": "4h"},
    "1D": {"period": "1y", "interval": "1d"},
    "1W": {"period": "2y", "interval": "1wk"},
}
ALL_ALPACA_PROFILE = {
    "1D": market_data.ALPACA_PROVIDER_NAME,
    "1W": market_data.ALPACA_PROVIDER_NAME,
    "4H": market_data.ALPACA_PROVIDER_NAME,
}
ALL_YAHOO_PROFILE = {
    "1D": market_data.YAHOO_PROVIDER_NAME,
    "1W": market_data.YAHOO_PROVIDER_NAME,
    "4H": market_data.YAHOO_PROVIDER_NAME,
}
FOUR_H_MISMATCH_CLASSES = {
    "MATCH",
    "TIMESTAMP_LABEL_ONLY",
    "SESSION_BOUNDARY_DIFFERENCE",
    "EXTENDED_HOURS_DIFFERENCE",
    "SOURCE_AGGREGATION_DIFFERENCE",
    "PARTIAL_BAR_DIFFERENCE",
    "TIMEZONE_DEFECT",
    "DAYLIGHT_SAVING_DEFECT",
    "MISSING_SOURCE_BARS",
    "IMPLEMENTATION_DEFECT",
    "UNRESOLVED",
}
STRATEGY_IMPACT_CLASSES = {
    "MATCH",
    "HARMLESS_DIFFERENCE",
    "STRATEGY_SIGNIFICANT",
    "IMPLEMENTATION_DEFECT",
    "UNRESOLVED",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _env_flag_status() -> dict[str, Any]:
    active_profile = market_data.configured_provider_profile_name()
    return {
        "flag": "STOCK_DATA_PROVIDER_PROFILE",
        "configured_value": os.getenv("STOCK_DATA_PROVIDER_PROFILE"),
        "active_profile": active_profile,
        "default_profile": market_data.DEFAULT_DATA_PROVIDER_PROFILE,
        "available_profiles": {
            name: market_data.timeframe_provider_profile(name)
            for name in sorted(market_data.TIMEFRAME_PROVIDER_PROFILES)
        },
        "current_timeframe_routing": market_data.configured_timeframe_provider_profile(),
        "effect": "Controls scanner candle provider by timeframe. The proposed hybrid profile routes 1D and 1W to Alpaca while leaving 4H on Yahoo. It does not control options, earnings, or journal/replay logic directly except where those systems call provider_name_for_timeframe().",
    }


def _consumer(
    *,
    consumer: str,
    endpoint_module: str,
    data_type: str,
    timeframe: str,
    provider: str,
    fallback: str,
    cache: str,
    controlling_flag: str,
    adjusted: str = "auto_adjust=True for candles unless noted",
    session: str = "provider_default",
    timestamps: str = "provider-native timestamps normalized to pandas/UTC where serialized",
    latest_completed: str = "not explicitly changed by this audit",
    trust_critical: bool = False,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "consumer": consumer,
        "endpoint_or_module": endpoint_module,
        "data_type": data_type,
        "timeframe": timeframe,
        "primary_provider": provider,
        "fallback_provider": fallback,
        "cache": cache,
        "controlling_flag": controlling_flag,
        "adjusted_or_unadjusted_behavior": adjusted,
        "regular_vs_extended_hours": session,
        "timestamp_timezone_handling": timestamps,
        "latest_completed_candle_logic": latest_completed,
        "trust_critical": trust_critical,
        "notes": notes,
    }


def consumer_dependency_map() -> list[dict[str, Any]]:
    profile = market_data.configured_timeframe_provider_profile()
    tf_provider = lambda label: profile.get(label, market_data.YAHOO_PROVIDER_NAME)
    scanner_cache = "scanner._price_cache keyed by ticker, period, interval, provider"
    return [
        _consumer(
            consumer="production stock scanner",
            endpoint_module="/api/scan -> scanner.scan_cached -> scanner._batch_download",
            data_type="OHLCV candles",
            timeframe="1D, 1W, 4H",
            provider=f"1D={tf_provider('1D')}, 1W={tf_provider('1W')}, 4H={tf_provider('4H')}",
            fallback="none in scanner candle path; failed provider returns missing data/partial telemetry",
            cache=scanner_cache,
            controlling_flag="STOCK_DATA_PROVIDER_PROFILE",
            session="provider_default; no explicit extended-hours filter in scanner batch helper",
            latest_completed="scanner consumes provider-native returned frame; diagnostics separately identify forming candles",
            trust_critical=True,
        ),
        _consumer(
            consumer="scanner cards and setup endpoints",
            endpoint_module="/api/scan response rows",
            data_type="derived setup fields from scanner candles plus display quote enrichment",
            timeframe="setup selected timeframe plus current quote",
            provider=f"setup fields from scanner routing; current_quote_price={market_data.ALPACA_PROVIDER_NAME}",
            fallback="quote enrichment fails silently; setup fields do not fall back",
            cache="scan cache plus Alpaca latest quote request",
            controlling_flag="STOCK_DATA_PROVIDER_PROFILE for setup candles only",
            latest_completed="inherits scanner candle behavior",
            trust_critical=True,
        ),
        _consumer(
            consumer="Guided Trade Charts",
            endpoint_module="/api/chart/candles -> main._download_chart_candles",
            data_type="visualization-only OHLCV candles",
            timeframe="4H, 1D, 30M",
            provider="provider_name_for_timeframe(timeframe)",
            fallback="configured provider, then Alpaca, then Yahoo",
            cache="no durable chart cache; provider request per expansion",
            controlling_flag="STOCK_DATA_PROVIDER_PROFILE selects first chart provider for 4H/1D; 30M falls back to Yahoo unless profile names it",
            session="provider_default; chart is display-only",
            latest_completed="serializes last returned rows; does not alter trade levels",
            trust_critical=True,
        ),
        _consumer(
            consumer="Active Trade Workspace",
            endpoint_module="/api/active-trades and /api/active-trades/{id}",
            data_type="journal plan, Guided Chart payload, Position Intelligence, notifications, Verified History, Trade Intelligence progress",
            timeframe="journal scanner_timeframe where available; chart defaults through Guided Chart endpoint",
            provider="no direct candle fetch except Guided Chart component",
            fallback="inherits Guided Chart fallback when chart is opened",
            cache="journal DB, notification DB, Verified History cache, Trade Intelligence cache",
            controlling_flag="indirect through Guided Chart/replay systems",
            latest_completed="does not evaluate candles directly",
            trust_critical=True,
        ),
        _consumer(
            consumer="Position Intelligence",
            endpoint_module="position_intelligence.build_position_intelligence",
            data_type="position plan plus point-in-time market snapshot",
            timeframe="caller supplied",
            provider="none internally",
            fallback="none internally",
            cache="none internally",
            controlling_flag="none",
            latest_completed="caller responsibility",
            trust_critical=True,
            notes="Live callers and replay callers supply market_snapshot; this builder is provider-agnostic.",
        ),
        _consumer(
            consumer="Smart Notifications",
            endpoint_module="smart_notifications plus main._attach_notification_metrics",
            data_type="scanner row transitions, journal position transitions, notification persistence",
            timeframe="inherits source event",
            provider="no independent candle provider",
            fallback="none",
            cache="SQLiteNotificationRepository",
            controlling_flag="none directly",
            latest_completed="inherits scanner/journal source event",
            trust_critical=False,
        ),
        _consumer(
            consumer="journal snapshots",
            endpoint_module="browser journal capture and /api/journal",
            data_type="stored setup plan fields, option fields, scanner snapshot fields",
            timeframe="scanner_timeframe stored when available",
            provider="inherits scanner setup fields at capture time",
            fallback="none at journal write",
            cache="durable SQLite journal",
            controlling_flag="indirect through scanner at setup creation time",
            latest_completed="snapshot preserves scanner output; does not refetch candles",
            trust_critical=True,
        ),
        _consumer(
            consumer="trade completion",
            endpoint_module="/api/active-trades/{id}/complete and journal update paths",
            data_type="user-entered completion fields and existing journal plan",
            timeframe="journal timeframe",
            provider="none during completion save",
            fallback="none",
            cache="durable SQLite journal",
            controlling_flag="none",
            latest_completed="replay queue later fetches candles",
            trust_critical=True,
        ),
        _consumer(
            consumer="replay engine",
            endpoint_module="main._fetch_replay_candles -> replay_position_intelligence",
            data_type="historical underlying OHLCV candles",
            timeframe="journaled timeframe or inferred 4H",
            provider="provider_name_for_timeframe(replay timeframe)",
            fallback="none in replay fetch path",
            cache="replay cache fields/jobs; provider fetch on demand",
            controlling_flag="STOCK_DATA_PROVIDER_PROFILE",
            session="provider_default",
            latest_completed="replay filters chronological window by timestamp/date and reports coverage; provider fetch itself uses period request",
            trust_critical=True,
        ),
        _consumer(
            consumer="Verified History",
            endpoint_module="verified_history pipeline -> main._process_verified_history_jobs",
            data_type="stored replay results and verification metadata",
            timeframe="inherits replay",
            provider="inherits replay",
            fallback="inherits replay; no independent fallback",
            cache="SQLiteVerifiedHistoryRepository jobs/results",
            controlling_flag="indirect through replay",
            latest_completed="inherits replay",
            trust_critical=True,
        ),
        _consumer(
            consumer="Trade Intelligence",
            endpoint_module="trade_intelligence.py via replay-verified records",
            data_type="verified historical trade records and grouped metrics",
            timeframe="stored setup timeframe from verified records",
            provider="inherits replay-verified evidence",
            fallback="none",
            cache="Trade Intelligence in-process cache/signature",
            controlling_flag="indirect through replay evidence",
            latest_completed="inherits replay",
            trust_critical=True,
        ),
        _consumer(
            consumer="earnings data",
            endpoint_module="scanner earnings helpers and yfinance.Ticker through MarketDataFacade.Ticker",
            data_type="earnings dates/calendar",
            timeframe="not candle-based",
            provider=market_data.YAHOO_PROVIDER_NAME,
            fallback="cached stale earnings where existing code permits",
            cache="scanner earnings cache",
            controlling_flag="none; not controlled by STOCK_DATA_PROVIDER_PROFILE",
            adjusted="not applicable",
            session="not applicable",
            timestamps="provider date/calendar fields",
            latest_completed="not applicable",
        ),
        _consumer(
            consumer="option chains and contract selection",
            endpoint_module="scanner option chain/contract helpers through yfinance.Ticker",
            data_type="option expirations, chains, bid/ask/open interest/volume",
            timeframe="option expiration chain",
            provider=market_data.YAHOO_PROVIDER_NAME,
            fallback="cached/deferred behavior for rate-limit safety",
            cache="option expiration/best contract caches",
            controlling_flag="none; MarketDataFacade.Ticker remains Yahoo-backed",
            adjusted="not applicable",
            session="provider_default option chain snapshot",
            timestamps="provider option snapshot timestamps where available",
            latest_completed="not applicable",
        ),
        _consumer(
            consumer="current-price or quote endpoints",
            endpoint_module="scanner._attach_current_quotes",
            data_type="latest quote midpoint/bid/ask",
            timeframe="latest quote",
            provider=market_data.ALPACA_PROVIDER_NAME,
            fallback="no Yahoo quote fallback; quote enrichment may be omitted",
            cache="none for latest quotes",
            controlling_flag="ALPACA_API_KEY/ALPACA_SECRET_KEY only",
            adjusted="raw quote data",
            session="Alpaca latest quote feed behavior",
            timestamps="Alpaca quote timestamp",
            latest_completed="not candle-based",
        ),
    ]


def migration_state_report() -> dict[str, Any]:
    consumers = consumer_dependency_map()
    counts = Counter()
    for row in consumers:
        provider = str(row.get("primary_provider") or "").lower()
        if "alpaca" in provider and "yahoo" in provider:
            counts["hybrid"] += 1
        elif "alpaca" in provider:
            counts["alpaca"] += 1
        elif "yahoo" in provider:
            counts["yahoo"] += 1
        else:
            counts["provider_agnostic"] += 1
    return {
        "version": AUDIT_VERSION,
        "generated_at": utc_now_iso(),
        "production_routing_changed": False,
        "alpaca_configured": market_data.alpaca_credentials_configured(),
        "stock_data_provider": market_data.configured_provider_name(),
        "provider_profile": _env_flag_status(),
        "migration_summary": {
            "migrated": [
                "Alpaca provider abstraction exists for candle downloads.",
                "Current quote enrichment uses Alpaca latest quotes.",
                "Discovery universe and optionable asset discovery are Alpaca-backed.",
                "Guided Chart endpoint can fall back to Alpaca without changing scanner routing.",
            ],
            "not_migrated": [
                "Default production scanner profile remains Yahoo-only unless STOCK_DATA_PROVIDER_PROFILE is changed.",
                "Options and earnings remain Yahoo-backed by design.",
                "Replay uses the active timeframe provider profile, so it remains Yahoo under the default profile.",
            ],
            "hybrid": [
                "A proposed 1D/1W Alpaca, 4H Yahoo profile exists but is not the default.",
                "Guided Charts use configured provider first and then provider fallback.",
                "Scanner setup fields may be Yahoo while display-only current quotes are Alpaca.",
            ],
        },
        "consumer_provider_counts": dict(counts),
        "dependency_map": consumers,
        "trust_critical_flows": trust_critical_data_flow_report(consumers),
        "remaining_yahoo_paths": remaining_yahoo_paths(consumers),
        "migration_risk_register": migration_risk_register(),
        "recommended_completion_plan": recommended_completion_plan(),
        "definition_of_done": definition_of_done(),
    }


def trust_critical_data_flow_report(consumers: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    rows = [row for row in (consumers or consumer_dependency_map()) if row.get("trust_critical")]
    return [
        {
            "consumer": row["consumer"],
            "provider": row["primary_provider"],
            "timeframe": row["timeframe"],
            "compatibility_note": _trust_compatibility_note(row["consumer"]),
            "requires_completion_validation": row["consumer"] in {
                "production stock scanner",
                "Guided Trade Charts",
                "replay engine",
                "Verified History",
                "Trade Intelligence",
            },
        }
        for row in rows
    ]


def _trust_compatibility_note(consumer: str) -> str:
    notes = {
        "production stock scanner": "Source of original setup decisions; any Alpaca migration must prove strategy-output parity on identical timestamps.",
        "Guided Trade Charts": "Visualization-only, but should use candle definitions compatible with scanner/replay to avoid confusing displays.",
        "journal snapshots": "Preserves original scanner output; future provider migrations should store provider/profile metadata with snapshots.",
        "replay engine": "Independent validation source; must not evaluate a different candle/session definition without explicit provider metadata.",
        "Verified History": "Consumes replay; provider compatibility depends on replay fetch integrity.",
        "Trade Intelligence": "Learns only from verified replay records; provider mismatch can affect historical eligibility and outcomes.",
    }
    return notes.get(consumer, "Provider-agnostic or derived from another trust-critical source.")


def remaining_yahoo_paths(consumers: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    rows = []
    for row in consumers or consumer_dependency_map():
        provider = str(row.get("primary_provider") or "").lower()
        fallback = str(row.get("fallback_provider") or "").lower()
        fallback_mentions_yahoo = "yahoo" in fallback and "no yahoo" not in fallback
        if "yahoo" not in provider and not fallback_mentions_yahoo:
            continue
        rows.append({
            "consumer": row["consumer"],
            "current_yahoo_usage": row["primary_provider"],
            "fallback": row["fallback_provider"],
            "why_remaining": _why_yahoo_remaining(row["consumer"]),
            "known_concern": _known_yahoo_concern(row["consumer"]),
        })
    return rows


def _why_yahoo_remaining(consumer: str) -> str:
    if consumer in {"earnings data", "option chains and contract selection"}:
        return "Deliberate hybrid design: MarketDataFacade.Ticker remains Yahoo-backed for options and earnings."
    if consumer == "production stock scanner":
        return "Default STOCK_DATA_PROVIDER_PROFILE is production_yahoo; proposed hybrid flag exists but is not active by default."
    if consumer in {"replay engine", "Verified History", "Trade Intelligence"}:
        return "These inherit the active timeframe provider profile through replay."
    if consumer == "Guided Trade Charts":
        return "Fallback chain intentionally includes Yahoo for visualization availability."
    return "Inherited from scanner or fallback behavior."


def _known_yahoo_concern(consumer: str) -> list[str]:
    concerns = {
        "earnings data": ["rate limiting", "fallback masking failures"],
        "option chains and contract selection": ["rate limiting", "stale data", "symbol compatibility"],
        "production stock scanner": ["rate limiting", "missing candles", "timestamp/session differences", "split or dividend adjustments"],
        "Guided Trade Charts": ["fallback masking failures", "session-boundary differences"],
        "replay engine": ["replay may not match original setup provider if provider profile changes later"],
        "Verified History": ["inherits replay provider consistency risk"],
        "Trade Intelligence": ["inherits replay provider consistency risk"],
    }
    return concerns.get(consumer, ["no meaningful reliability problem identified yet"])


def _classify_raw_discrepancy(validation: dict[str, Any]) -> str:
    providers = validation.get("providers") or {}
    yahoo = providers.get("yahoo") or {}
    alpaca = providers.get("alpaca") or {}
    comparison = validation.get("comparison") or {}
    if alpaca.get("error_classification") == "missing_credentials":
        return "UNRESOLVED"
    if yahoo.get("status") != "success" or alpaca.get("status") != "success":
        return "DATA_QUALITY_CONCERN"
    if yahoo.get("duplicate_timestamps") or alpaca.get("duplicate_timestamps") or yahoo.get("out_of_order_timestamps") or alpaca.get("out_of_order_timestamps"):
        return "DATA_QUALITY_CONCERN"
    if int(comparison.get("completed_timestamp_matches") or 0) == 0:
        return "STRATEGY_SIGNIFICANT"
    stats = comparison.get("stats") or {}
    max_ohlc = stats.get("max_ohlc_percent_difference")
    missing = int(comparison.get("missing_in_alpaca_count") or 0) + int(comparison.get("missing_in_yahoo_count") or 0)
    if max_ohlc is None and missing == 0:
        return "MATCH"
    if max_ohlc is not None and float(max_ohlc) > 1.0:
        return "STRATEGY_SIGNIFICANT"
    if missing > 5:
        return "DATA_QUALITY_CONCERN"
    if max_ohlc is not None and float(max_ohlc) > 0.05:
        return "EXPECTED_PROVIDER_VARIATION"
    if missing:
        return "HARMLESS_DIFFERENCE"
    if max_ohlc is not None and float(max_ohlc) > 0:
        return "HARMLESS_DIFFERENCE"
    return "MATCH"


def _strategy_classification(row: dict[str, Any]) -> str:
    if row.get("production_errors") or row.get("comparison_errors"):
        return "UNRESOLVED"
    comparison = row.get("comparison") or {}
    material = comparison.get("material_differences") or []
    if material:
        return "STRATEGY_SIGNIFICANT"
    if comparison.get("differences"):
        return "HARMLESS_DIFFERENCE"
    return "MATCH"


def provider_comparison_report(
    *,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    include_strategy: bool = True,
    limit: int = 12,
) -> dict[str, Any]:
    started = time.perf_counter()
    selected_symbols = [str(s or "").strip().upper() for s in (symbols or REPRESENTATIVE_SYMBOLS) if str(s or "").strip()]
    selected_symbols = list(dict.fromkeys(selected_symbols))[: max(1, min(int(limit or 12), 50))]
    selected_timeframes = [str(tf or "").strip().upper() for tf in (timeframes or ["4H", "1D", "1W"]) if str(tf or "").strip()]
    selected_timeframes = [tf for tf in selected_timeframes if tf in TIMEFRAME_REQUESTS]
    raw_rows = []
    for symbol in selected_symbols:
        for timeframe in selected_timeframes:
            request = TIMEFRAME_REQUESTS[timeframe]
            validation = market_data.validate_candle_pair(
                ticker=symbol,
                period=request["period"],
                interval=request["interval"],
            )
            raw_rows.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "period": request["period"],
                "interval": request["interval"],
                "classification": _classify_raw_discrepancy(validation),
                "validation": validation,
            })
    strategy_rows = []
    if include_strategy:
        strategy_rows = _strategy_comparison_rows(selected_symbols)
    raw_counts = Counter(row["classification"] for row in raw_rows)
    strategy_counts = Counter(row["classification"] for row in strategy_rows)
    return {
        "version": AUDIT_VERSION,
        "generated_at": utc_now_iso(),
        "production_routing_changed": False,
        "symbols": selected_symbols,
        "timeframes": selected_timeframes,
        "classification_values": sorted(RAW_DISCREPANCY_CLASSES),
        "raw_candle_comparison": {
            "rows": raw_rows,
            "classification_counts": dict(raw_counts),
        },
        "downstream_interpretation_comparison": {
            "enabled": bool(include_strategy),
            "rows": strategy_rows,
            "classification_counts": dict(strategy_counts),
            "note": "Runs existing scanner interpretation on explicit provider profiles without changing production routing.",
        },
        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
    }


def _parse_date_bound(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    try:
        return pd.Timestamp(value)
    except Exception:
        return None


def _to_et_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize(market_data.EASTERN_TZ)
    return ts.tz_convert(market_data.EASTERN_TZ)


def _to_utc_iso(value) -> str | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(market_data.EASTERN_TZ)
    return ts.tz_convert(timezone.utc).isoformat().replace("+00:00", "Z")


def _to_et_iso(value) -> str | None:
    if value is None:
        return None
    return _to_et_timestamp(value).isoformat()


def _filter_frame_window(df: pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    result = df.copy()
    start_ts = _parse_date_bound(start)
    end_ts = _parse_date_bound(end)
    if start_ts is not None:
        start_et = _to_et_timestamp(start_ts)
        result = result[[(_to_et_timestamp(idx) >= start_et) for idx in result.index]]
    if end_ts is not None:
        end_et = _to_et_timestamp(end_ts)
        result = result[[(_to_et_timestamp(idx) <= end_et) for idx in result.index]]
    return result


def _session_bucket_for_timestamp(ts_et: pd.Timestamp, include_extended_hours: bool = False) -> tuple[pd.Timestamp, pd.Timestamp, str, int] | None:
    day = ts_et.date()

    def stamp(hour: int, minute: int = 0) -> pd.Timestamp:
        return pd.Timestamp(datetime.combine(day, datetime_time(hour, minute)), tz=market_data.EASTERN_TZ)

    if include_extended_hours:
        buckets = [
            (stamp(4), stamp(8), "pre_market", 8),
            (stamp(8), stamp(12), "regular_or_extended", 8),
            (stamp(12), stamp(16), "regular_or_extended", 8),
            (stamp(16), stamp(20), "after_hours", 8),
        ]
    else:
        buckets = [
            (stamp(9, 30), stamp(13, 30), "regular", 8),
            (stamp(13, 30), stamp(16), "regular_short_final", 5),
        ]
    for start, end, session_type, expected_count in buckets:
        if start <= ts_et < end:
            return start, end, session_type, expected_count
    return None


def _ohlcv_from_group(group: pd.DataFrame) -> dict[str, float | None]:
    if group is None or group.empty:
        return {"open": None, "high": None, "low": None, "close": None, "volume": None}
    return {
        "open": _safe_float(group.iloc[0].get("Open")),
        "high": _safe_float(group["High"].max()),
        "low": _safe_float(group["Low"].min()),
        "close": _safe_float(group.iloc[-1].get("Close")),
        "volume": _safe_float(group["Volume"].sum()),
    }


def _safe_float(value):
    try:
        if value is None:
            return None
        number = float(value)
        return number if number == number else None
    except Exception:
        return None


def _frame_to_native_4h_candles(symbol: str, provider: str, df: pd.DataFrame, start: str | None = None, end: str | None = None) -> list[dict[str, Any]]:
    filtered = _filter_frame_window(market_data._normalize_ohlcv(df), start=start, end=end)
    candles = []
    for ts, row in filtered.iterrows():
        ts_et = _to_et_timestamp(ts)
        bucket_end = ts_et + timedelta(hours=4)
        candles.append({
            "symbol": symbol,
            "provider": provider,
            "trading_date": ts_et.date().isoformat(),
            "market_timezone": "America/New_York",
            "session_type": "provider_native",
            "source_interval": "4h",
            "native_or_aggregated": "provider_native",
            "bucket_start": _to_utc_iso(ts_et),
            "bucket_end": _to_utc_iso(bucket_end),
            "bucket_start_et": ts_et.isoformat(),
            "bucket_end_et": bucket_end.isoformat(),
            "timestamp_label": _to_utc_iso(ts),
            "timestamp_label_et": ts_et.isoformat(),
            "timestamp_label_convention": "provider_label_preserved_assumed_bucket_start_for_native_4h",
            "first_source_bar_time": _to_utc_iso(ts),
            "last_source_bar_time": _to_utc_iso(ts),
            "source_bar_count": 1,
            "expected_source_bar_count": 1,
            "open": _safe_float(row.get("Open")),
            "high": _safe_float(row.get("High")),
            "low": _safe_float(row.get("Low")),
            "close": _safe_float(row.get("Close")),
            "volume": _safe_float(row.get("Volume")),
            "is_partial": False,
            "is_complete": True,
            "cache_used": False,
            "data_as_of": utc_now_iso(),
        })
    return candles


def reconstruct_4h_candles(
    symbol: str,
    provider: str,
    source_df: pd.DataFrame,
    *,
    include_extended_hours: bool = False,
    start: str | None = None,
    end: str | None = None,
) -> list[dict[str, Any]]:
    source = _filter_frame_window(market_data._normalize_ohlcv(source_df), start=start, end=end)
    grouped: dict[tuple[str, str], list[tuple[pd.Timestamp, Any]]] = {}
    for ts, row in source.iterrows():
        ts_et = _to_et_timestamp(ts)
        bucket = _session_bucket_for_timestamp(ts_et, include_extended_hours=include_extended_hours)
        if not bucket:
            continue
        bucket_start, bucket_end, session_type, expected_count = bucket
        key = (bucket_start.isoformat(), bucket_end.isoformat())
        grouped.setdefault(key, []).append((ts_et, row))

    candles = []
    for key, rows in sorted(grouped.items()):
        bucket_start = pd.Timestamp(key[0])
        bucket_end = pd.Timestamp(key[1])
        session_type = _session_bucket_for_timestamp(bucket_start, include_extended_hours=include_extended_hours)[2]
        expected_count = _session_bucket_for_timestamp(bucket_start, include_extended_hours=include_extended_hours)[3]
        rows = sorted(rows, key=lambda item: item[0])
        frame = pd.DataFrame([row for _, row in rows], index=[ts for ts, _ in rows])
        ohlcv = _ohlcv_from_group(frame)
        source_count = len(rows)
        candles.append({
            "symbol": symbol,
            "provider": provider,
            "trading_date": bucket_start.date().isoformat(),
            "market_timezone": "America/New_York",
            "session_type": session_type,
            "source_interval": "30m",
            "native_or_aggregated": "canonical_kairos_aggregated",
            "bucket_start": _to_utc_iso(bucket_start),
            "bucket_end": _to_utc_iso(bucket_end),
            "bucket_start_et": bucket_start.isoformat(),
            "bucket_end_et": bucket_end.isoformat(),
            "timestamp_label": _to_utc_iso(bucket_start),
            "timestamp_label_et": bucket_start.isoformat(),
            "timestamp_label_convention": "canonical_bucket_start",
            "first_source_bar_time": _to_utc_iso(rows[0][0]) if rows else None,
            "last_source_bar_time": _to_utc_iso(rows[-1][0]) if rows else None,
            "source_bar_count": source_count,
            "expected_source_bar_count": expected_count,
            **ohlcv,
            "is_partial": source_count < expected_count,
            "is_complete": source_count >= expected_count,
            "cache_used": False,
            "data_as_of": utc_now_iso(),
        })
    return candles


def _interval_key(candle: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(candle.get("trading_date") or ""),
        str(candle.get("session_type") or ""),
        str(candle.get("bucket_start") or ""),
        str(candle.get("bucket_end") or ""),
    )


def _label_key(candle: dict[str, Any]) -> str:
    return str(candle.get("timestamp_label") or "")


def _ohlcv_differences(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    differences = {}
    for field in ["open", "high", "low", "close", "volume"]:
        lv = _safe_float(left.get(field))
        rv = _safe_float(right.get(field))
        if lv is None or rv is None:
            differences[field] = {"left": lv, "right": rv, "absolute": None, "percent": None}
            continue
        absolute = rv - lv
        percent = (absolute / lv * 100) if lv else None
        differences[field] = {"left": lv, "right": rv, "absolute": absolute, "percent": percent}
    return differences


def _max_price_percent_difference(differences: dict[str, Any]) -> float:
    values = []
    for field in ["open", "high", "low", "close"]:
        percent = (differences.get(field) or {}).get("percent")
        if percent is not None:
            values.append(abs(float(percent)))
    return max(values) if values else 0.0


def compare_candle_intervals(left: list[dict[str, Any]], right: list[dict[str, Any]], *, left_name: str, right_name: str) -> dict[str, Any]:
    left_by_interval = {_interval_key(candle): candle for candle in left}
    right_by_interval = {_interval_key(candle): candle for candle in right}
    left_by_label = {_label_key(candle): candle for candle in left if _label_key(candle)}
    right_by_label = {_label_key(candle): candle for candle in right if _label_key(candle)}
    interval_keys = sorted(set(left_by_interval) | set(right_by_interval))
    rows = []
    counts = Counter()
    for key in interval_keys:
        l = left_by_interval.get(key)
        r = right_by_interval.get(key)
        if l and r:
            differences = _ohlcv_differences(l, r)
            max_price_pct = _max_price_percent_difference(differences)
            if _label_key(l) != _label_key(r):
                classification = "TIMESTAMP_LABEL_ONLY" if max_price_pct <= 0.05 else "SOURCE_AGGREGATION_DIFFERENCE"
            elif max_price_pct > 0.05:
                classification = "SOURCE_AGGREGATION_DIFFERENCE"
            else:
                classification = "TIMESTAMP_LABEL_ONLY" if _label_key(l) != _label_key(r) else "MATCH"
            counts[classification] += 1
            rows.append({
                "interval_key": key,
                "classification": classification,
                "left_timestamp_label": _label_key(l),
                "right_timestamp_label": _label_key(r),
                "ohlcv_differences": differences,
                "max_price_percent_difference": max_price_pct,
                "left": l,
                "right": r,
            })
            continue
        missing = l or r
        label = _label_key(missing)
        label_counterpart = right_by_label.get(label) if l else left_by_label.get(label)
        if label_counterpart:
            classification = "SESSION_BOUNDARY_DIFFERENCE"
        elif missing and missing.get("is_partial"):
            classification = "PARTIAL_BAR_DIFFERENCE"
        else:
            classification = "MISSING_SOURCE_BARS"
        counts[classification] += 1
        rows.append({
            "interval_key": key,
            "classification": classification,
            "left_timestamp_label": _label_key(l) if l else None,
            "right_timestamp_label": _label_key(r) if r else None,
            "missing_side": right_name if l else left_name,
            "left": l,
            "right": r,
        })
    return {
        "left": left_name,
        "right": right_name,
        "candles_compared": len(rows),
        "interval_matches": counts.get("MATCH", 0) + counts.get("TIMESTAMP_LABEL_ONLY", 0) + counts.get("SOURCE_AGGREGATION_DIFFERENCE", 0),
        "classification_counts": dict(counts),
        "rows": rows[:120],
    }


def _candles_to_frame(candles: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    index = []
    for candle in candles:
        if not candle.get("bucket_start"):
            continue
        index.append(pd.Timestamp(candle["bucket_start"]))
        rows.append({
            "Open": candle.get("open"),
            "High": candle.get("high"),
            "Low": candle.get("low"),
            "Close": candle.get("close"),
            "Volume": candle.get("volume"),
        })
    if not rows:
        return pd.DataFrame()
    return market_data._normalize_ohlcv(pd.DataFrame(rows, index=pd.DatetimeIndex(index)))


def _strategy_impact_rows(symbol: str, series: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    from scanner import scan_ticker

    yahoo_provider = market_data.YahooMarketDataProvider()
    alpaca_provider = market_data.AlpacaMarketDataProvider()
    active_profile = market_data.configured_timeframe_provider_profile()
    daily_provider = alpaca_provider if active_profile.get("1D") == market_data.ALPACA_PROVIDER_NAME else yahoo_provider
    weekly_provider = alpaca_provider if active_profile.get("1W") == market_data.ALPACA_PROVIDER_NAME else yahoo_provider
    daily_df, daily_error = market_data._safe_provider_download(daily_provider, symbol, "1y", "1d")
    weekly_df, weekly_error = market_data._safe_provider_download(weekly_provider, symbol, "2y", "1wk")
    if daily_error or weekly_error:
        return [{
            "symbol": symbol,
            "baseline": "production_yahoo_4h",
            "variant": "all",
            "classification": "UNRESOLVED",
            "reason": "daily or weekly provider data unavailable",
            "errors": {"daily": daily_error, "weekly": weekly_error},
        }]
    outputs = {}
    for name, candles in series.items():
        h4_df = _candles_to_frame(candles)
        output = scan_ticker(symbol, _daily_df=daily_df, _weekly_df=weekly_df, _h4_df=h4_df)
        outputs[name] = market_data._strategy_output_snapshot(output)
    baseline = outputs.get("production_yahoo_4h")
    rows = []
    for name, output in outputs.items():
        if name == "production_yahoo_4h":
            continue
        comparison = market_data._compare_strategy_outputs(baseline, output)
        material = comparison.get("material_differences") or []
        classification = "STRATEGY_SIGNIFICANT" if material else "HARMLESS_DIFFERENCE" if comparison.get("differences") else "MATCH"
        rows.append({
            "symbol": symbol,
            "baseline": "production_yahoo_4h",
            "variant": name,
            "classification": classification,
            "comparison": comparison,
            "baseline_output": baseline,
            "variant_output": output,
        })
    return rows


def _provider_download(provider_name: str, symbol: str, period: str, interval: str) -> tuple[pd.DataFrame, str | None]:
    provider = market_data.AlpacaMarketDataProvider() if provider_name == market_data.ALPACA_PROVIDER_NAME else market_data.YahooMarketDataProvider()
    return market_data._safe_provider_download(provider, symbol, period, interval)


def _construction_map() -> list[dict[str, Any]]:
    return [
        {
            "provider": "Yahoo",
            "source_interval": "4h via yfinance.download(interval='4h')",
            "native_or_aggregated": "provider/yfinance native from Kairos perspective",
            "session_policy": "provider_default; no Kairos regular-session filter before scanner consumption",
            "bucket_anchor": "provider-defined",
            "timestamp_label": "provider timestamp preserved in pandas index",
            "partial_bar_policy": "scanner consumes returned frame; diagnostics can identify forming bars separately",
            "completed_bar_policy": "not centrally enforced in scanner batch path",
        },
        {
            "provider": "Alpaca",
            "source_interval": "4Hour via /v2/stocks/bars",
            "native_or_aggregated": "provider native 4Hour bars",
            "session_policy": "Alpaca provider default feed/session behavior",
            "bucket_anchor": "provider-defined by Alpaca timeframe",
            "timestamp_label": "Alpaca bar timestamp converted to pandas UTC index",
            "partial_bar_policy": "scanner consumes returned frame when configured; diagnostics can identify forming bars separately",
            "completed_bar_policy": "not centrally enforced in scanner batch path",
        },
        {
            "provider": "Canonical Kairos Reconstruction",
            "source_interval": "30m",
            "native_or_aggregated": "diagnostic-only aggregation",
            "session_policy": "regular: 09:30-13:30 and 13:30-16:00 ET; extended optional: 04:00-08:00, 08:00-12:00, 12:00-16:00, 16:00-20:00 ET",
            "bucket_anchor": "explicit bucket start in America/New_York",
            "timestamp_label": "canonical bucket start",
            "partial_bar_policy": "source_bar_count < expected_source_bar_count marks partial",
            "completed_bar_policy": "complete only when expected 30m source bars exist for the bucket",
        },
    ]


def four_h_forensics_report(
    *,
    symbols: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    include_extended_hours: bool = False,
    include_strategy: bool = True,
    limit: int = 3,
) -> dict[str, Any]:
    started = time.perf_counter()
    selected = [str(s or "").strip().upper() for s in (symbols or ["SPY", "NVDA", "DOW"]) if str(s or "").strip()]
    selected = list(dict.fromkeys(selected))[: max(1, min(int(limit or 3), 10))]
    reports = []
    interval_totals = Counter()
    strategy_totals = Counter()
    for symbol in selected:
        yahoo_4h_df, yahoo_4h_error = _provider_download(market_data.YAHOO_PROVIDER_NAME, symbol, "60d", "4h")
        alpaca_4h_df, alpaca_4h_error = _provider_download(market_data.ALPACA_PROVIDER_NAME, symbol, "60d", "4h")
        yahoo_30m_df, yahoo_30m_error = _provider_download(market_data.YAHOO_PROVIDER_NAME, symbol, "60d", "30m")
        alpaca_30m_df, alpaca_30m_error = _provider_download(market_data.ALPACA_PROVIDER_NAME, symbol, "60d", "30m")
        series = {
            "production_yahoo_4h": _frame_to_native_4h_candles(symbol, market_data.YAHOO_PROVIDER_NAME, yahoo_4h_df, start=start, end=end),
            "current_alpaca_4h": _frame_to_native_4h_candles(symbol, market_data.ALPACA_PROVIDER_NAME, alpaca_4h_df, start=start, end=end),
            "reconstructed_yahoo_4h": reconstruct_4h_candles(symbol, market_data.YAHOO_PROVIDER_NAME, yahoo_30m_df, include_extended_hours=include_extended_hours, start=start, end=end),
            "reconstructed_alpaca_4h": reconstruct_4h_candles(symbol, market_data.ALPACA_PROVIDER_NAME, alpaca_30m_df, include_extended_hours=include_extended_hours, start=start, end=end),
        }
        comparisons = {
            "native_yahoo_vs_native_alpaca": compare_candle_intervals(series["production_yahoo_4h"], series["current_alpaca_4h"], left_name="production_yahoo_4h", right_name="current_alpaca_4h"),
            "reconstructed_yahoo_vs_reconstructed_alpaca": compare_candle_intervals(series["reconstructed_yahoo_4h"], series["reconstructed_alpaca_4h"], left_name="reconstructed_yahoo_4h", right_name="reconstructed_alpaca_4h"),
            "native_yahoo_vs_reconstructed_yahoo": compare_candle_intervals(series["production_yahoo_4h"], series["reconstructed_yahoo_4h"], left_name="production_yahoo_4h", right_name="reconstructed_yahoo_4h"),
            "native_alpaca_vs_reconstructed_alpaca": compare_candle_intervals(series["current_alpaca_4h"], series["reconstructed_alpaca_4h"], left_name="current_alpaca_4h", right_name="reconstructed_alpaca_4h"),
        }
        for comparison in comparisons.values():
            interval_totals.update(comparison.get("classification_counts") or {})
        strategy_rows = _strategy_impact_rows(symbol, series) if include_strategy else []
        strategy_totals.update(row.get("classification") for row in strategy_rows)
        reports.append({
            "symbol": symbol,
            "provider_errors": {
                "yahoo_4h": yahoo_4h_error,
                "alpaca_4h": alpaca_4h_error,
                "yahoo_30m": yahoo_30m_error,
                "alpaca_30m": alpaca_30m_error,
            },
            "series_counts": {name: len(candles) for name, candles in series.items()},
            "sample_candles": {name: candles[-8:] for name, candles in series.items()},
            "interval_comparisons": comparisons,
            "strategy_impact": strategy_rows,
        })
    recommendation = _four_h_recommendation(interval_totals, strategy_totals)
    return {
        "version": FOUR_H_FORENSIC_VERSION,
        "generated_at": utc_now_iso(),
        "production_routing_changed": False,
        "symbols": selected,
        "start": start,
        "end": end,
        "include_extended_hours": bool(include_extended_hours),
        "construction_map": _construction_map(),
        "mismatch_classification_values": sorted(FOUR_H_MISMATCH_CLASSES),
        "strategy_impact_classification_values": sorted(STRATEGY_IMPACT_CLASSES),
        "reports": reports,
        "interval_comparison_totals": dict(interval_totals),
        "strategy_impact_totals": dict(strategy_totals),
        "completed_bar_findings": {
            "regular_first_bucket": "09:30-13:30 ET expects 8 30m source bars in canonical reconstruction.",
            "regular_final_bucket": "13:30-16:00 ET is a shortened final session bucket and expects 5 30m source bars.",
            "incomplete_current_bucket": "Diagnostic reconstruction marks buckets partial when expected 30m bars are missing.",
            "holidays_and_early_closes": "Not fully calendar-aware yet; early close support is represented by partial bucket evidence rather than an exchange calendar.",
            "daylight_saving": "Buckets are anchored in America/New_York; UTC offsets change with DST through zoneinfo conversion.",
        },
        "root_cause_summary": _four_h_root_cause(interval_totals),
        "recommendation": recommendation,
        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
    }


def _four_h_root_cause(interval_totals: Counter) -> str:
    if interval_totals.get("SESSION_BOUNDARY_DIFFERENCE") or interval_totals.get("MISSING_SOURCE_BARS"):
        return "Native Yahoo and Alpaca 4H bars are not directly timestamp-comparable because provider-native buckets use different anchors/session construction; interval-based reconstructed candles are required before strategy conclusions."
    if interval_totals.get("TIMESTAMP_LABEL_ONLY"):
        return "Provider labels differ for otherwise equivalent intervals in at least part of the sample."
    return "Root cause remains unresolved from the current bounded sample."


def _four_h_recommendation(interval_totals: Counter, strategy_totals: Counter) -> str:
    if strategy_totals.get("STRATEGY_SIGNIFICANT") or interval_totals.get("SOURCE_AGGREGATION_DIFFERENCE"):
        return "BLOCK_ROUTING_DECISION"
    if interval_totals.get("SESSION_BOUNDARY_DIFFERENCE") or interval_totals.get("MISSING_SOURCE_BARS"):
        return "USE_CANONICAL_KAIROS_AGGREGATION"
    if interval_totals.get("UNRESOLVED"):
        return "RETAIN_HYBRID_PENDING_MORE_DATA"
    return "RETAIN_HYBRID_PENDING_MORE_DATA"


def _strategy_comparison_rows(symbols: list[str]) -> list[dict[str, Any]]:
    rows = []
    yahoo_provider = market_data.YahooMarketDataProvider()
    alpaca_provider = market_data.AlpacaMarketDataProvider()
    active_profile = market_data.configured_timeframe_provider_profile()
    profile_specs = [
        ("active_profile", active_profile),
        ("all_yahoo", ALL_YAHOO_PROFILE),
        ("proposed_hybrid_1d_1w", market_data.timeframe_provider_profile(market_data.PROVIDER_PROFILE_PROPOSED_HYBRID)),
        ("all_alpaca", ALL_ALPACA_PROFILE),
    ]
    for symbol in symbols:
        production_output, production_errors = market_data._scanner_output_for_profile(
            symbol=symbol,
            profile=active_profile,
            yahoo_provider=yahoo_provider,
            alpaca_provider=alpaca_provider,
        )
        for profile_name, profile in profile_specs[1:]:
            comparison_output, comparison_errors = market_data._scanner_output_for_profile(
                symbol=symbol,
                profile=profile,
                yahoo_provider=yahoo_provider,
                alpaca_provider=alpaca_provider,
            )
            comparison = market_data._compare_strategy_outputs(production_output, comparison_output)
            row = {
                "symbol": symbol,
                "production_profile": active_profile,
                "comparison_profile_name": profile_name,
                "comparison_profile": profile,
                "production_errors": production_errors,
                "comparison_errors": comparison_errors,
                "production_output": production_output,
                "comparison_output": comparison_output,
                "comparison": comparison,
            }
            row["classification"] = _strategy_classification(row)
            row["difference_reason"] = _strategy_difference_reason(row)
            rows.append(row)
    return rows


def _strategy_difference_reason(row: dict[str, Any]) -> str:
    if row.get("production_errors") or row.get("comparison_errors"):
        return "provider-quality issue or unavailable data"
    material = ((row.get("comparison") or {}).get("material_differences") or [])
    if not material:
        return "no material strategy difference"
    if any(field in material for field in {"entry", "stop", "target_1", "target_2", "target_3", "risk"}):
        return "strategy-significant unresolved difference"
    if any(field in material for field in {"selected_timeframe", "trend", "direction", "setup_status", "trade_stage"}):
        return "timestamp/session or provider-data variation"
    return "expected provider-data variation"


def migration_risk_register() -> list[dict[str, Any]]:
    return [
        {
            "risk": "Replay may use a different provider profile than the original scanner snapshot after a future routing change.",
            "affected_systems": ["replay", "Verified History", "Trade Intelligence"],
            "mitigation": "Persist provider/profile metadata with new journal snapshots and use it explicitly for replay where historically required.",
        },
        {
            "risk": "1D/1W Alpaca adjustment/session behavior may differ from Yahoo and alter trend, structure, or levels.",
            "affected_systems": ["scanner", "journal snapshots", "replay"],
            "mitigation": "Run parallel raw candle and strategy-output comparison over representative symbols and recent sessions before routing changes.",
        },
        {
            "risk": "Guided Chart fallback can mask provider failure and show a different provider than scanner/replay.",
            "affected_systems": ["Guided Trade Charts", "Active Trade Workspace"],
            "mitigation": "Keep provider attempts visible in developer diagnostics and display selected provider unobtrusively.",
        },
        {
            "risk": "Yahoo remains rate-limited for options/earnings even if candle migration completes.",
            "affected_systems": ["Option Plan", "earnings cautions"],
            "mitigation": "Treat options/earnings as a deliberate hybrid dependency unless Alpaca replacements are separately validated.",
        },
        {
            "risk": "Latest incomplete candle handling is not uniformly enforced across all consumers.",
            "affected_systems": ["scanner", "charts", "replay"],
            "mitigation": "Define and test a shared completed-bar policy before final migration.",
        },
    ]


def recommended_completion_plan() -> list[str]:
    return [
        "Run developer-only Yahoo-vs-Alpaca raw candle comparison on the agreed representative symbol set for 30M, 4H, 1D, and 1W.",
        "Run downstream scanner interpretation comparison for active, all-Yahoo, proposed hybrid, and all-Alpaca profiles.",
        "Classify every raw and downstream discrepancy; resolve implementation defects before any routing change.",
        "Persist provider/profile metadata on new journal setup snapshots before changing replay-critical routing.",
        "Align replay provider selection with original setup provider metadata or explicitly document replay as current-provider validation.",
        "Promote the proposed hybrid profile only after no unexplained strategy-significant discrepancies remain for 1D/1W.",
        "Keep Yahoo for options/earnings unless a separate replacement is validated.",
        "Add Railway validation commands for status, compare, scan parity, replay parity, and Strategy Freeze.",
        "Retain the environment flag as emergency rollback until production evidence covers several market sessions.",
    ]


def definition_of_done() -> list[str]:
    return [
        "Every market-data consumer has a documented provider, fallback, cache, and controlling flag.",
        "Every intended candle timeframe uses Alpaca in production, or has a documented reason to remain Yahoo.",
        "Any remaining Yahoo usage is deliberate, documented, and limited to data Alpaca cannot suitably replace.",
        "Scanner, Guided Charts, journal snapshots, replay, Verified History, and Trade Intelligence use compatible candle definitions and adjustment rules.",
        "Parallel comparison covers the representative symbol set, supported timeframes, and sufficient market sessions.",
        "No unexplained strategy-significant discrepancies remain.",
        "Provider failures and fallbacks are visible in developer diagnostics.",
        "The temporary migration flag is removed or intentionally retained only as an emergency rollback mechanism.",
        "Local and Railway validation pass.",
        "STRATEGY_FREEZE passes with no strategy-rule or threshold changes.",
    ]
