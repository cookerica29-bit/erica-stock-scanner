"""External scanner candidate ingestion endpoints.

Receives shortlist candidates from external scanners and stores them as
reviewable candidates. Promotion to active trade management remains a separate
explicit workflow.
"""

from __future__ import annotations

import os
import json
import sqlite3
import hmac
import math
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import APIRouter, Cookie, Header, HTTPException, Query, Response
from pydantic import BaseModel, Field

from market_data import AlpacaMarketDataProvider
from scanner import _batch_download, _best_contract, _compute_atr, _detect_bos, _find_order_block, _find_swings, _flatten_columns
from structural_resistance import clamp_target, levels_near_target, resolve_stop
from displacement_score import score_displacement


router = APIRouter(prefix="/api/v1/scanner", tags=["scanner"])
ATR_MULTIPLIER = 1.5
RR_WARNING_THRESHOLD = 1.5
MIN_TARGET_ATR_MULTIPLE_DEFAULT = 2.0
ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_MODEL_DEFAULT = "claude-sonnet-4-5"
AI_CHART_REVIEW_RUBRIC_VERSION = "kairos-chart-note-v1"
CANDIDATE_PREVIEW_TRANSIENT_OPTION_REFRESH_TTL = timedelta(minutes=10)
SCANNER_SESSION_COOKIE = "kairos_scanner_session"
ENTRY_PROXIMITY_MAX_PCT_DEFAULT = 1.5
ENTRY_PROXIMITY_MAX_ATR_MULTIPLE_DEFAULT = 0.5
MIN_CLEAN_OPTION_PREMIUM_DEFAULT = 0.50
MIN_CLEAN_OPTION_CONTRACT_COST_DEFAULT = 50.0
EXECUTION_SHADOW_MIN_REACTION_ATR = 0.10
EXECUTION_SHADOW_RECENT_RANGE_BARS = 15
EXECUTION_SHADOW_CONFIRMATION_BARS = 5
EXECUTION_SHADOW_VOLUME_LOOKBACK_BARS = EXECUTION_SHADOW_RECENT_RANGE_BARS - EXECUTION_SHADOW_CONFIRMATION_BARS
EXECUTION_SHADOW_MIN_DIRECTIONAL_EXPANSION_ATR = 0.75
EXECUTION_SHADOW_MIN_VOLUME_RATIO = 0.60
EXECUTION_SHADOW_LOW_VOL_ATR_PCT_MAX = 0.015
EXECUTION_SHADOW_LOW_VOL_MIN_NET_MOVE_PCT = 0.01


def default_candidates_db_path() -> str:
    configured = os.environ.get("KAIROS_CANDIDATES_DB")
    if configured:
        return configured
    mount_path = os.environ.get("RAILWAY_VOLUME_MOUNT_PATH")
    if mount_path:
        return str(Path(mount_path) / "kairos_candidates.sqlite3")
    journal_path = os.environ.get("JOURNAL_DB_PATH") or os.environ.get("KAIROS_JOURNAL_DB_PATH")
    if journal_path:
        return str(Path(journal_path).parent / "kairos_candidates.sqlite3")
    if (
        os.environ.get("RAILWAY_ENVIRONMENT")
        or os.environ.get("RAILWAY_PROJECT_ID")
        or os.environ.get("RAILWAY_SERVICE_ID")
    ):
        return "/data/kairos_candidates.sqlite3"
    return str(Path(__file__).resolve().parent / "data" / "kairos_candidates.sqlite3")


def _get_api_key() -> Optional[str]:
    return os.environ.get("KAIROS_SCANNER_API_KEY")


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _min_target_atr_multiple() -> float:
    return _float_env("MIN_TARGET_ATR_MULTIPLE", MIN_TARGET_ATR_MULTIPLE_DEFAULT)


def _entry_proximity_max_pct() -> float:
    return _float_env("ENTER_NOW_ENTRY_PROXIMITY_MAX_PCT", ENTRY_PROXIMITY_MAX_PCT_DEFAULT)


def _entry_proximity_max_atr_multiple() -> float:
    return _float_env("ENTER_NOW_ENTRY_PROXIMITY_MAX_ATR_MULTIPLE", ENTRY_PROXIMITY_MAX_ATR_MULTIPLE_DEFAULT)


def _min_clean_option_premium() -> float:
    return _float_env("ENTER_NOW_MIN_OPTION_PREMIUM", MIN_CLEAN_OPTION_PREMIUM_DEFAULT)


def _min_clean_option_contract_cost() -> float:
    return _float_env("ENTER_NOW_MIN_OPTION_CONTRACT_COST", MIN_CLEAN_OPTION_CONTRACT_COST_DEFAULT)


def _ensure_candidate_promotions_schema(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='candidate_promotions'"
    ).fetchone()
    needs_rebuild = bool(row and ("target REAL NOT NULL" in row["sql"] or "risk_reward REAL NOT NULL" in row["sql"]))
    if needs_rebuild:
        conn.execute("ALTER TABLE candidate_promotions RENAME TO candidate_promotions_old")
        conn.execute(
            """
            CREATE TABLE candidate_promotions (
                ticker TEXT NOT NULL,
                source TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop REAL NOT NULL,
                target REAL,
                risk_reward REAL,
                rr_warning INTEGER NOT NULL,
                no_valid_target INTEGER NOT NULL DEFAULT 0,
                promoted_at TEXT NOT NULL,
                position_size REAL,
                atr14 REAL NOT NULL,
                atr_multiplier REAL NOT NULL,
                rr_warning_threshold REAL NOT NULL,
                min_target_atr_multiple REAL NOT NULL DEFAULT 2.0,
                target_source TEXT NOT NULL,
                PRIMARY KEY (ticker, source)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO candidate_promotions
                (ticker, source, direction, entry_price, stop, target, risk_reward,
                 rr_warning, no_valid_target, promoted_at, position_size, atr14,
                 atr_multiplier, rr_warning_threshold, min_target_atr_multiple,
                 target_source)
            SELECT ticker, source, direction, entry_price, stop, target, risk_reward,
                   rr_warning, 0, promoted_at, position_size, atr14, atr_multiplier,
                   rr_warning_threshold, 2.0, target_source
            FROM candidate_promotions_old
            """
        )
        conn.execute("DROP TABLE candidate_promotions_old")
        # Deliberately no early return here: fall through to the ALTER-based
        # checks below so any columns added to this schema *after* the rebuild
        # path was written (like the structural-target-clamp fields) still get
        # added post-rebuild instead of silently missing until the next deploy.

    columns = {info["name"] for info in conn.execute("PRAGMA table_info(candidate_promotions)").fetchall()}
    if "no_valid_target" not in columns:
        conn.execute("ALTER TABLE candidate_promotions ADD COLUMN no_valid_target INTEGER NOT NULL DEFAULT 0")
    if "min_target_atr_multiple" not in columns:
        conn.execute(
            "ALTER TABLE candidate_promotions ADD COLUMN min_target_atr_multiple REAL NOT NULL DEFAULT 2.0"
        )
    if "raw_target" not in columns:
        conn.execute("ALTER TABLE candidate_promotions ADD COLUMN raw_target REAL")
    if "raw_risk_reward" not in columns:
        conn.execute("ALTER TABLE candidate_promotions ADD COLUMN raw_risk_reward REAL")
    if "target_clamped" not in columns:
        conn.execute("ALTER TABLE candidate_promotions ADD COLUMN target_clamped INTEGER NOT NULL DEFAULT 0")
    if "target_clamp_badge" not in columns:
        conn.execute("ALTER TABLE candidate_promotions ADD COLUMN target_clamp_badge TEXT")
    if "target_clamp_reason" not in columns:
        conn.execute("ALTER TABLE candidate_promotions ADD COLUMN target_clamp_reason TEXT")
    if "raw_stop" not in columns:
        conn.execute("ALTER TABLE candidate_promotions ADD COLUMN raw_stop REAL")
    if "stop_source" not in columns:
        conn.execute("ALTER TABLE candidate_promotions ADD COLUMN stop_source TEXT")
    if "displacement_score" not in columns:
        conn.execute("ALTER TABLE candidate_promotions ADD COLUMN displacement_score REAL")
    if "displacement_label" not in columns:
        conn.execute("ALTER TABLE candidate_promotions ADD COLUMN displacement_label TEXT")
    if "displacement_components_json" not in columns:
        conn.execute("ALTER TABLE candidate_promotions ADD COLUMN displacement_components_json TEXT")
    if "raw_magnitude_score" not in columns:
        conn.execute("ALTER TABLE candidate_promotions ADD COLUMN raw_magnitude_score REAL")
    if "displacement_read" not in columns:
        conn.execute("ALTER TABLE candidate_promotions ADD COLUMN displacement_read TEXT")
    if "bos_confirmed" not in columns:
        conn.execute("ALTER TABLE candidate_promotions ADD COLUMN bos_confirmed INTEGER NOT NULL DEFAULT 0")
    if "bos_level" not in columns:
        conn.execute("ALTER TABLE candidate_promotions ADD COLUMN bos_level REAL")


# CREATE TABLE IF NOT EXISTS is a no-op against an existing production table,
# even when its own column list has grown since the table was first created --
# so new columns need an explicit ADD COLUMN migration here, or _store_plan_preview's
# INSERT (which names every column) fails immediately with "no such column" on
# an existing database.
def _ensure_candidate_plan_previews_schema(conn: sqlite3.Connection) -> None:
    columns = {info["name"] for info in conn.execute("PRAGMA table_info(candidate_plan_previews)").fetchall()}
    if "raw_target" not in columns:
        conn.execute("ALTER TABLE candidate_plan_previews ADD COLUMN raw_target REAL")
    if "raw_risk_reward" not in columns:
        conn.execute("ALTER TABLE candidate_plan_previews ADD COLUMN raw_risk_reward REAL")
    if "target_clamped" not in columns:
        conn.execute("ALTER TABLE candidate_plan_previews ADD COLUMN target_clamped INTEGER NOT NULL DEFAULT 0")
    if "target_clamp_badge" not in columns:
        conn.execute("ALTER TABLE candidate_plan_previews ADD COLUMN target_clamp_badge TEXT")
    if "target_clamp_reason" not in columns:
        conn.execute("ALTER TABLE candidate_plan_previews ADD COLUMN target_clamp_reason TEXT")
    if "raw_stop" not in columns:
        conn.execute("ALTER TABLE candidate_plan_previews ADD COLUMN raw_stop REAL")
    if "stop_source" not in columns:
        conn.execute("ALTER TABLE candidate_plan_previews ADD COLUMN stop_source TEXT")
    if "displacement_score" not in columns:
        conn.execute("ALTER TABLE candidate_plan_previews ADD COLUMN displacement_score REAL")
    if "displacement_label" not in columns:
        conn.execute("ALTER TABLE candidate_plan_previews ADD COLUMN displacement_label TEXT")
    if "displacement_components_json" not in columns:
        conn.execute("ALTER TABLE candidate_plan_previews ADD COLUMN displacement_components_json TEXT")
    if "raw_magnitude_score" not in columns:
        conn.execute("ALTER TABLE candidate_plan_previews ADD COLUMN raw_magnitude_score REAL")
    if "displacement_read" not in columns:
        conn.execute("ALTER TABLE candidate_plan_previews ADD COLUMN displacement_read TEXT")
    if "bos_confirmed" not in columns:
        conn.execute("ALTER TABLE candidate_plan_previews ADD COLUMN bos_confirmed INTEGER NOT NULL DEFAULT 0")
    if "bos_level" not in columns:
        conn.execute("ALTER TABLE candidate_plan_previews ADD COLUMN bos_level REAL")


# Schema setup (CREATE TABLE IF NOT EXISTS x7 plus a migration check) used to run
# on every single call to _get_db() -- i.e. on every API request. Each of those
# statements needs a write-capable lock even when it's a no-op, so every request
# (including simple reads like /candidates) was contending for a write lock on
# every call. Guarded here to run once per process instead. WAL mode is enabled
# so that even the rare real writer (plan-preview cache refresh) never blocks
# concurrent readers -- see _enriched_previews_for_candidates for the other half
# of this fix (that loop used to hold one uncommitted write transaction open
# across an unbounded, network-call-heavy loop; see incident notes there).
_schema_ready_db_paths: set[str] = set()
_schema_ready_lock = threading.Lock()


def _get_db():
    db_path = Path(default_candidates_db_path())
    if db_path.parent and str(db_path.parent) not in {"", "."}:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA journal_mode = WAL")
    db_key = str(db_path)
    if db_key in _schema_ready_db_paths:
        return conn
    with _schema_ready_lock:
        if db_key in _schema_ready_db_paths:
            return conn
        _initialize_candidates_schema(conn)
        _schema_ready_db_paths.add(db_key)
    return conn


def _ensure_candidates_schema(conn: sqlite3.Connection) -> None:
    columns = {info["name"] for info in conn.execute("PRAGMA table_info(candidates)").fetchall()}
    if "source_universe" not in columns:
        conn.execute("ALTER TABLE candidates ADD COLUMN source_universe TEXT")


def _initialize_candidates_schema(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS candidates (
            ticker TEXT NOT NULL,
            source TEXT NOT NULL,
            signal TEXT NOT NULL,
            entry_price REAL,
            ema21_4h REAL,
            daily_regime TEXT,
            confidence TEXT,
            sma50_daily REAL,
            sma200_daily REAL,
            status TEXT NOT NULL DEFAULT 'new',
            scanned_at TEXT NOT NULL,
            expires_at TEXT,
            updated_at TEXT NOT NULL,
            source_universe TEXT,
            PRIMARY KEY (ticker, source)
        )
        """
    )
    _ensure_candidates_schema(conn)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS candidate_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            source TEXT NOT NULL,
            signal TEXT NOT NULL,
            entry_price REAL,
            ema21_4h REAL,
            daily_regime TEXT,
            confidence TEXT,
            sma50_daily REAL,
            sma200_daily REAL,
            scanned_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS candidate_promotions (
            ticker TEXT NOT NULL,
            source TEXT NOT NULL,
            direction TEXT NOT NULL,
            entry_price REAL NOT NULL,
            stop REAL NOT NULL,
            target REAL,
            risk_reward REAL,
            rr_warning INTEGER NOT NULL,
            no_valid_target INTEGER NOT NULL DEFAULT 0,
            promoted_at TEXT NOT NULL,
            position_size REAL,
            atr14 REAL NOT NULL,
            atr_multiplier REAL NOT NULL,
            rr_warning_threshold REAL NOT NULL,
            min_target_atr_multiple REAL NOT NULL DEFAULT 2.0,
            target_source TEXT NOT NULL,
            raw_target REAL,
            raw_risk_reward REAL,
            target_clamped INTEGER NOT NULL DEFAULT 0,
            target_clamp_badge TEXT,
            target_clamp_reason TEXT,
            raw_stop REAL,
            stop_source TEXT,
            displacement_score REAL,
            displacement_label TEXT,
            displacement_components_json TEXT,
            raw_magnitude_score REAL,
            displacement_read TEXT,
            bos_confirmed INTEGER NOT NULL DEFAULT 0,
            bos_level REAL,
            PRIMARY KEY (ticker, source)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS candidate_ai_chart_reviews (
            ticker TEXT NOT NULL,
            source TEXT NOT NULL,
            signal TEXT NOT NULL,
            classification TEXT NOT NULL,
            rationale TEXT NOT NULL,
            caveat TEXT NOT NULL,
            reviewed_at TEXT NOT NULL,
            model TEXT NOT NULL,
            rubric_version TEXT NOT NULL,
            data_source TEXT NOT NULL,
            bars_start TEXT,
            bars_end TEXT,
            raw_response TEXT,
            PRIMARY KEY (ticker, source)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS candidate_plan_previews (
            ticker TEXT NOT NULL,
            source TEXT NOT NULL,
            signal TEXT NOT NULL,
            entry_price REAL,
            stop REAL,
            target REAL,
            risk_reward REAL,
            rr_warning INTEGER NOT NULL DEFAULT 0,
            no_valid_target INTEGER NOT NULL DEFAULT 0,
            atr14 REAL,
            atr_multiplier REAL NOT NULL DEFAULT 1.5,
            rr_warning_threshold REAL NOT NULL DEFAULT 1.5,
            min_target_atr_multiple REAL NOT NULL DEFAULT 2.0,
            target_source TEXT,
            option_contract_json TEXT,
            preview_error TEXT,
            computed_at TEXT NOT NULL,
            candidate_updated_at TEXT,
            raw_target REAL,
            raw_risk_reward REAL,
            target_clamped INTEGER NOT NULL DEFAULT 0,
            target_clamp_badge TEXT,
            target_clamp_reason TEXT,
            raw_stop REAL,
            stop_source TEXT,
            displacement_score REAL,
            displacement_label TEXT,
            displacement_components_json TEXT,
            raw_magnitude_score REAL,
            displacement_read TEXT,
            bos_confirmed INTEGER NOT NULL DEFAULT 0,
            bos_level REAL,
            PRIMARY KEY (ticker, source)
        )
        """
    )
    _ensure_candidate_plan_previews_schema(conn)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS candidate_status_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            source TEXT NOT NULL,
            previous_status TEXT,
            new_status TEXT NOT NULL,
            changed_at TEXT NOT NULL,
            trigger TEXT NOT NULL
        )
        """
    )
    _ensure_candidate_promotions_schema(conn)
    conn.commit()


class CandidateIn(BaseModel):
    ticker: str
    signal: Literal["long", "short"]
    entry_price: Optional[float] = None
    ema21_4h: Optional[float] = None
    daily_regime: Optional[str] = None
    confidence: Optional[Literal["high", "medium", "low"]] = None
    sma50_daily: Optional[float] = None
    sma200_daily: Optional[float] = None
    # Which symbol universe produced this candidate -- see
    # main._merge_curated_watchlist_into_universe. None for sources that
    # don't distinguish universes (e.g. smc_forex), same as before this
    # field existed.
    source_universe: Optional[Literal["broker_feed", "curated_watchlist", "both"]] = None


class ShortlistIn(BaseModel):
    source: str = Field(..., description="e.g. 'ma_pipeline', 'smc_forex'")
    scanned_at: datetime
    candidates: list[CandidateIn]


class CandidateOut(BaseModel):
    ticker: str
    source: str
    signal: str
    entry_price: Optional[float]
    ema21_4h: Optional[float]
    daily_regime: Optional[str]
    confidence: Optional[str]
    sma50_daily: Optional[float]
    sma200_daily: Optional[float]
    status: str
    scanned_at: str
    updated_at: str
    source_universe: Optional[str] = None


class CandidatePromotionOut(BaseModel):
    ticker: str
    source: str
    direction: str
    entry_price: float
    stop: float
    target: Optional[float]
    risk_reward: Optional[float]
    rr_warning: bool
    no_valid_target: bool
    promoted_at: str
    position_size: Optional[float]
    atr14: float
    atr_multiplier: float
    rr_warning_threshold: float
    min_target_atr_multiple: float
    target_source: str
    raw_target: Optional[float] = None
    raw_risk_reward: Optional[float] = None
    target_clamped: bool = False
    target_clamp_badge: Optional[str] = None
    target_clamp_reason: Optional[str] = None
    raw_stop: Optional[float] = None
    stop_source: Optional[str] = None
    displacement_score: Optional[float] = None
    displacement_label: Optional[str] = None
    displacement_components: Optional[dict[str, Any]] = None
    raw_magnitude_score: Optional[float] = None
    displacement_read: Optional[str] = None
    bos_confirmed: bool = False
    bos_details: Optional[dict[str, Any]] = None


class CandidateChartReviewOut(BaseModel):
    ticker: str
    source: str
    signal: str
    classification: str
    rationale: str
    caveat: str
    reviewed_at: str
    model: str
    rubric_version: str
    data_source: str
    bars_start: Optional[str]
    bars_end: Optional[str]


class CandidatePlanPreviewOut(BaseModel):
    ticker: str
    source: str
    signal: str
    entry_price: Optional[float]
    stop: Optional[float]
    target: Optional[float]
    risk_reward: Optional[float]
    rr_warning: bool
    no_valid_target: bool
    atr14: Optional[float]
    atr_multiplier: float
    rr_warning_threshold: float
    min_target_atr_multiple: float
    target_source: Optional[str]
    raw_target: Optional[float] = None
    raw_risk_reward: Optional[float] = None
    target_clamped: bool = False
    target_clamp_badge: Optional[str] = None
    target_clamp_reason: Optional[str] = None
    raw_stop: Optional[float] = None
    stop_source: Optional[str] = None
    displacement_score: Optional[float] = None
    displacement_label: Optional[str] = None
    displacement_components: Optional[dict[str, Any]] = None
    raw_magnitude_score: Optional[float] = None
    displacement_read: Optional[str] = None
    bos_confirmed: bool = False
    bos_details: Optional[dict[str, Any]] = None
    option_contract: Optional[dict[str, Any]]
    current_price: Optional[float] = None
    current_quote_timestamp: Optional[str] = None
    current_quote_source: Optional[str] = None
    current_quote_price_branch: Optional[str] = None
    entry_distance: Optional[float] = None
    entry_distance_pct: Optional[float] = None
    entry_distance_atr: Optional[float] = None
    entry_proximity_ok: bool = False
    entry_proximity_reason: Optional[str] = None
    entry_proximity_threshold_pct: float = ENTRY_PROXIMITY_MAX_PCT_DEFAULT
    entry_proximity_threshold_atr: float = ENTRY_PROXIMITY_MAX_ATR_MULTIPLE_DEFAULT
    execution_shadow_checked: bool = False
    execution_shadow_ok: Optional[bool] = None
    execution_shadow_reason: Optional[str] = None
    execution_shadow_version: Optional[str] = None
    execution_shadow_candle_time: Optional[str] = None
    preview_error: Optional[str]
    computed_at: str
    candidate_updated_at: Optional[str]


class RejectedEntry(BaseModel):
    ticker: str
    reason: str


class IngestResponse(BaseModel):
    received: int
    created: int
    updated: int
    rejected: list[RejectedEntry]


CandidateStatus = Literal["active", "dismissed", "new"]


class StatusUpdate(BaseModel):
    status: CandidateStatus


class ScannerSessionIn(BaseModel):
    api_key: str = Field(min_length=1)


def _check_api_key(
    x_api_key: Optional[str],
    scanner_session: Optional[str] = None,
) -> None:
    api_key = _get_api_key()
    if not api_key:
        raise HTTPException(status_code=500, detail="KAIROS_SCANNER_API_KEY not configured on server")
    supplied = x_api_key or scanner_session
    if not supplied or not hmac.compare_digest(supplied, api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _valid_ticker(ticker: str) -> bool:
    return bool(ticker) and ticker.replace(".", "").replace("-", "").isalnum()


def _parse_displacement_components_json(raw: Optional[str]) -> Optional[dict]:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _bos_level_for_storage(payload: dict) -> Optional[float]:
    details = payload.get("bos_details")
    if not details:
        return None
    return details.get("break_level")


def _bos_details_from_row(output: dict) -> Optional[dict]:
    # bos_level is stored as its own flat column (not JSON) -- it's a single
    # number, unlike displacement_components' genuinely multi-valued payload --
    # but exposed via the API as a small bos_details dict per the agreed field
    # shape, so a future addition (e.g. bars-since-break) doesn't need another
    # migration on top of a bare scalar field.
    bos_level = output.pop("bos_level", None)
    if not output.get("bos_confirmed") or bos_level is None:
        return None
    return {"break_level": bos_level}


def _row_to_promotion(row: sqlite3.Row) -> dict:
    output = dict(row)
    output["rr_warning"] = bool(output.get("rr_warning"))
    output["no_valid_target"] = bool(output.get("no_valid_target"))
    output["target_clamped"] = bool(output.get("target_clamped"))
    output["displacement_components"] = _parse_displacement_components_json(
        output.pop("displacement_components_json", None)
    )
    output["bos_confirmed"] = bool(output.get("bos_confirmed"))
    output["bos_details"] = _bos_details_from_row(output)
    return output


def _row_to_chart_review(row: sqlite3.Row) -> dict:
    output = dict(row)
    output.pop("raw_response", None)
    return output


def _row_to_plan_preview(row: sqlite3.Row | dict) -> dict:
    output = dict(row)
    output["rr_warning"] = bool(output.get("rr_warning"))
    output["no_valid_target"] = bool(output.get("no_valid_target"))
    output["target_clamped"] = bool(output.get("target_clamped"))
    output["displacement_components"] = _parse_displacement_components_json(
        output.pop("displacement_components_json", None)
    )
    output["bos_confirmed"] = bool(output.get("bos_confirmed"))
    output["bos_details"] = _bos_details_from_row(output)
    raw_contract = output.pop("option_contract_json", None)
    if raw_contract:
        try:
            parsed_contract = json.loads(raw_contract)
            output["option_contract"] = _normalize_preview_option_contract(parsed_contract) if isinstance(parsed_contract, dict) else parsed_contract
        except json.JSONDecodeError:
            output["option_contract"] = {"available": False, "reason": "Stored option contract JSON is not usable"}
    else:
        output["option_contract"] = None
    return output


def _record_status_change(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    source: str,
    previous_status: Optional[str],
    new_status: str,
    changed_at: str,
    trigger: str,
) -> None:
    conn.execute(
        """
        INSERT INTO candidate_status_history
            (ticker, source, previous_status, new_status, changed_at, trigger)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (ticker, source, previous_status, new_status, changed_at, trigger),
    )


def _safe_option_contract_for_candidate(ticker: str, direction: str, entry_price: Optional[float]) -> dict:
    if entry_price is None or entry_price <= 0:
        return {"available": False, "execution": "No Clean Contract", "reason": "Missing candidate entry price", "source": "candidate_preview"}
    try:
        contract = _best_contract(ticker, "LONG" if direction == "long" else "SHORT", float(entry_price), block_on_miss=True)
    except Exception as exc:
        return {
            "available": False,
            "execution": "Contract Data Unavailable",
            "reason": f"Existing option selector failed: {exc.__class__.__name__}",
            "source": "option_chain",
        }
    if not isinstance(contract, dict):
        return {
            "available": False,
            "execution": "Contract Data Unavailable",
            "reason": "Existing option selector returned no contract",
            "source": "option_chain",
        }
    return _normalize_preview_option_contract(contract)


def _normalize_preview_option_contract(contract: dict) -> dict:
    normalized = dict(contract)
    # Optionability floor: chain_available=False is the ONLY genuine
    # "this ticker has no listed options chain" case (see _best_contract in
    # scanner.py). Every other unavailable/low-quality result means a real
    # chain was found -- display it plainly, not as a transient data error.
    if normalized.get("chain_available") is False:
        normalized["execution"] = "No Options Chain"
        normalized["reason"] = "No options chain available"
        normalized["source"] = "unavailable"
        normalized.pop("transient_unavailable", None)
    return normalized


def _safe_float_value(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _option_contract_cost(contract: dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    cost = _safe_float_value(contract.get("estimated_contract_cost"))
    premium = cost / 100 if cost is not None and cost > 0 else None
    if premium is not None:
        return cost, premium

    for key in ("ask", "mark", "mid", "last", "last_price"):
        candidate_premium = _safe_float_value(contract.get(key))
        if candidate_premium is not None and candidate_premium > 0:
            return round(candidate_premium * 100, 2), candidate_premium
    return None, None


def _contract_block_reason(contract: Optional[dict[str, Any]]) -> Optional[str]:
    if not contract or not contract.get("available"):
        if contract:
            return str(contract.get("reason") or contract.get("execution") or "No clean options contract")
        return "No clean options contract."
    execution = str(contract.get("execution") or "").strip().lower()
    if any(grade in execution for grade in ("excellent", "good", "fair")):
        cost, premium = _option_contract_cost(contract)
        min_cost = _min_clean_option_contract_cost()
        min_premium = _min_clean_option_premium()
        if cost is None or premium is None:
            return "Option premium is unavailable for clean dashboard."
        if cost < min_cost or premium < min_premium:
            return (
                f"Option premium is too thin for clean dashboard "
                f"(${premium:.2f} / ${cost:.2f} contract cost; minimum ${min_premium:.2f} / ${min_cost:.2f})."
            )
        return None
    return f"Option contract quality is {contract.get('execution') or 'unknown'}."


def _latest_quote_for_ticker(ticker: str) -> Optional[dict[str, Any]]:
    try:
        quotes = AlpacaMarketDataProvider().latest_quotes([ticker])
    except Exception:
        return None
    quote = quotes.get(str(ticker).upper()) if isinstance(quotes, dict) else None
    return quote if isinstance(quote, dict) else None


def _entry_proximity(
    *,
    entry_price: Optional[float],
    atr14: Optional[float],
    quote: Optional[dict[str, Any]],
) -> dict[str, Any]:
    max_pct = _entry_proximity_max_pct()
    max_atr = _entry_proximity_max_atr_multiple()
    base = {
        "current_price": None,
        "current_quote_timestamp": None,
        "current_quote_source": None,
        "current_quote_price_branch": None,
        "entry_distance": None,
        "entry_distance_pct": None,
        "entry_distance_atr": None,
        "entry_proximity_ok": False,
        "entry_proximity_reason": "Current quote unavailable",
        "entry_proximity_threshold_pct": max_pct,
        "entry_proximity_threshold_atr": max_atr,
    }
    try:
        entry = float(entry_price)
    except (TypeError, ValueError):
        base["entry_proximity_reason"] = "Entry price unavailable"
        return base
    if entry <= 0:
        base["entry_proximity_reason"] = "Entry price unavailable"
        return base
    if not quote:
        return base
    price_branch = quote.get("price_branch")
    if price_branch in ("bid_only", "ask_only"):
        # A one-sided quote is a real, legitimate value for DISPLAY purposes
        # elsewhere in the app (see AlpacaMarketDataProvider._quote_price,
        # exercised by tests/current_quote_price_v1.py) -- but it is not
        # reliable enough to gate a live pass/fail decision on. Root cause:
        # Alpaca's quotes endpoint defaults to the IEX feed (no `feed=sip`
        # param is sent anywhere in this codebase), and IEX alone carries a
        # small fraction of total market volume, so at any given instant a
        # meaningful share of names have no recent two-sided IEX print --
        # the stale/resting single side that's left can be far from the real
        # NBBO (confirmed live: BRK.B showed "5.81% / 3.94 ATR away from
        # entry" off a bid_only quote of 473.31, while independently checked
        # real price was ~0.35% from entry -- a bogus $29 gap). Treat it the
        # same as no quote at all rather than asserting a distance that
        # might be wrong, especially now that this number is surfaced
        # prominently in the near-miss ranking view.
        base["entry_proximity_reason"] = "Current quote is one-sided (no reliable two-sided price); not used for gating"
        return base
    try:
        current_price = float(quote.get("price"))
    except (TypeError, ValueError):
        return base
    if current_price <= 0:
        return base

    atr = float(atr14 or 0)
    distance = abs(current_price - entry)
    pct_distance = (distance / entry) * 100
    atr_distance = (distance / atr) if atr > 0 else None
    pct_threshold = entry * (max_pct / 100)
    atr_threshold = atr * max_atr if atr > 0 else 0
    allowed_distance = max(pct_threshold, atr_threshold)
    ok = distance <= allowed_distance
    reason = None
    if not ok:
        if atr_distance is not None:
            reason = f"Price moved {pct_distance:.2f}% / {atr_distance:.2f} ATR away from entry"
        else:
            reason = f"Price moved {pct_distance:.2f}% away from entry"

    return {
        **base,
        "current_price": round(current_price, 4),
        "current_quote_timestamp": str(quote.get("timestamp")) if quote.get("timestamp") is not None else None,
        "current_quote_source": quote.get("source"),
        "current_quote_price_branch": quote.get("price_branch"),
        "entry_distance": round(distance, 4),
        "entry_distance_pct": round(pct_distance, 2),
        "entry_distance_atr": round(atr_distance, 2) if atr_distance is not None else None,
        "entry_proximity_ok": ok,
        "entry_proximity_reason": reason,
    }


def _attach_entry_proximity(preview: dict, quote: Optional[dict[str, Any]]) -> dict:
    return {
        **preview,
        **_entry_proximity(
            entry_price=preview.get("entry_price"),
            atr14=preview.get("atr14"),
            quote=quote,
        ),
    }


def _latest_quotes_for_previews(previews: list[dict]) -> dict[str, dict[str, Any]]:
    tickers = sorted({str(preview.get("ticker") or "").upper() for preview in previews if preview.get("ticker")})
    if not tickers:
        return {}
    try:
        quotes = AlpacaMarketDataProvider().latest_quotes(tickers)
    except Exception:
        return {}
    return quotes if isinstance(quotes, dict) else {}


def _as_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bar_time_iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return str(value)


def _recent_4h_bars_for_execution_shadow(ticker: str) -> list[dict[str, Any]]:
    try:
        raw = AlpacaMarketDataProvider().download([ticker], period="30d", interval="4h", auto_adjust=True)
    except Exception:
        return []
    if raw is None or raw.empty:
        return []
    try:
        df = _flatten_columns(raw.copy()).dropna().astype(float)
    except Exception:
        return []
    required = {"Open", "High", "Low", "Close"}
    if df.empty or not required.issubset(set(df.columns)):
        return []
    rows: list[dict[str, Any]] = []
    for index, row in df.tail(EXECUTION_SHADOW_RECENT_RANGE_BARS).iterrows():
        rows.append(
            {
                "time": _bar_time_iso(index),
                "open": _as_float(row.get("Open")),
                "high": _as_float(row.get("High")),
                "low": _as_float(row.get("Low")),
                "close": _as_float(row.get("Close")),
                "volume": _as_float(row.get("Volume")) if "Volume" in row else None,
            }
        )
    return rows


def _execution_shadow_from_bars(
    candidate: sqlite3.Row | dict,
    preview: dict,
    bars: list[dict[str, Any]],
) -> dict[str, Any]:
    version = "4h-confirmed-recently-still-intact-shadow-v10"
    base = {
        "execution_shadow_checked": True,
        "execution_shadow_ok": False,
        "execution_shadow_reason": "Recent 4H bars unavailable",
        "execution_shadow_version": version,
        "execution_shadow_candle_time": None,
        "execution_shadow_failures": [],
        "execution_shadow_diagnostics": {},
    }
    if len(bars) < EXECUTION_SHADOW_RECENT_RANGE_BARS:
        base["execution_shadow_reason"] = (
            f"Need {EXECUTION_SHADOW_RECENT_RANGE_BARS} recent 4H bars for execution check"
        )
        return base

    latest = bars[-1]
    prior = bars[-2]
    prior_lows = [_as_float(bar.get("low")) for bar in bars[-4:-1]]
    prior_lows = [low for low in prior_lows if low is not None]
    range_bars = bars[-EXECUTION_SHADOW_RECENT_RANGE_BARS:]
    baseline_bars = range_bars[:EXECUTION_SHADOW_VOLUME_LOOKBACK_BARS]
    confirmation_bars = range_bars[EXECUTION_SHADOW_VOLUME_LOOKBACK_BARS:]
    early_range_closes = [_as_float(bar.get("close")) for bar in baseline_bars]
    recent_range_closes = [_as_float(bar.get("close")) for bar in confirmation_bars]
    early_range_closes = [close_value for close_value in early_range_closes if close_value is not None]
    recent_range_closes = [close_value for close_value in recent_range_closes if close_value is not None]
    prior_volumes = [_as_float(bar.get("volume")) for bar in baseline_bars]
    prior_volumes = [volume for volume in prior_volumes if volume is not None and volume > 0]
    qualifying_confirmation_volumes = []
    recent_confirmations = []

    open_price = _as_float(latest.get("open"))
    high = _as_float(latest.get("high"))
    low = _as_float(latest.get("low"))
    close = _as_float(latest.get("close"))
    prior_close = _as_float(prior.get("close"))
    entry = _as_float(preview.get("entry_price"))
    atr = _as_float(preview.get("atr14"))
    ema21 = _as_float(candidate["ema21_4h"] if "ema21_4h" in candidate.keys() else None)

    base["execution_shadow_candle_time"] = latest.get("time")
    if (
        None in (open_price, high, low, close, prior_close, entry, atr)
        or not prior_lows
        or not early_range_closes
        or not recent_range_closes
    ):
        base["execution_shadow_reason"] = "4H execution inputs unavailable"
        return base
    if atr is None or atr <= 0:
        base["execution_shadow_reason"] = "ATR unavailable for execution check"
        return base

    hold_floor = entry - (0.5 * atr)
    if ema21 is not None:
        hold_floor = max(hold_floor, ema21 - (0.5 * atr))
    holds_zone = close >= hold_floor
    no_fresh_lower_low = low >= min(prior_lows)
    range_start_close = early_range_closes[0]
    directional_expansion_atr = (close - range_start_close) / atr
    direction_expanded = directional_expansion_atr >= EXECUTION_SHADOW_MIN_DIRECTIONAL_EXPANSION_ATR
    atr_pct_of_entry = atr / entry if entry > 0 else None
    net_move_pct = (close - range_start_close) / range_start_close if range_start_close > 0 else None
    low_vol_bucket = atr_pct_of_entry is not None and atr_pct_of_entry <= EXECUTION_SHADOW_LOW_VOL_ATR_PCT_MAX
    low_vol_net_move_ok = (
        not low_vol_bucket
        or (
            net_move_pct is not None
            and net_move_pct >= EXECUTION_SHADOW_LOW_VOL_MIN_NET_MOVE_PCT
        )
    )
    confirmation_reaction_atr = 0.0
    confirmation_directional_atr = 0.0
    for idx, bar in enumerate(confirmation_bars):
        bar_open = _as_float(bar.get("open"))
        bar_high = _as_float(bar.get("high"))
        bar_low = _as_float(bar.get("low"))
        bar_close = _as_float(bar.get("close"))
        bar_volume = _as_float(bar.get("volume"))
        previous_bar = range_bars[EXECUTION_SHADOW_VOLUME_LOOKBACK_BARS + idx - 1] if idx > 0 else baseline_bars[-1]
        previous_close = _as_float(previous_bar.get("close")) if previous_bar else None
        if None in (bar_open, bar_high, bar_low, bar_close, previous_close):
            continue
        strong_confirmation = (
            bar_close > bar_open
            and bar_close > ((bar_high + bar_low) / 2)
        )
        if not strong_confirmation:
            continue
        bar_reaction_atr = max(bar_close - bar_open, bar_close - previous_close) / atr
        bar_directional_atr = (bar_close - early_range_closes[0]) / atr
        confirmation_reaction_atr = max(confirmation_reaction_atr, bar_reaction_atr)
        confirmation_directional_atr = max(confirmation_directional_atr, bar_directional_atr)
        if (
            bar_reaction_atr >= EXECUTION_SHADOW_MIN_REACTION_ATR
            and bar_directional_atr >= EXECUTION_SHADOW_MIN_DIRECTIONAL_EXPANSION_ATR
        ):
            recent_confirmations.append(bar)
            if bar_volume is not None and bar_volume > 0:
                qualifying_confirmation_volumes.append(bar_volume)
    has_recent_confirmation = bool(recent_confirmations)
    volume_ratio = None
    volume_confirmed = True
    if prior_volumes:
        sorted_prior_volumes = sorted(prior_volumes)
        mid = len(sorted_prior_volumes) // 2
        median_prior_volume = (
            sorted_prior_volumes[mid]
            if len(sorted_prior_volumes) % 2
            else (sorted_prior_volumes[mid - 1] + sorted_prior_volumes[mid]) / 2
        )
        if median_prior_volume > 0:
            confirmation_volume = max(qualifying_confirmation_volumes) if qualifying_confirmation_volumes else 0.0
            volume_ratio = confirmation_volume / median_prior_volume
            volume_confirmed = volume_ratio >= EXECUTION_SHADOW_MIN_VOLUME_RATIO

    failures = []
    if not has_recent_confirmation:
        failures.append(
            f"no recent bullish confirmation "
            f"(reaction {confirmation_reaction_atr:.2f} ATR, expansion {confirmation_directional_atr:.2f} ATR)"
        )
    if not holds_zone:
        failures.append(f"lost hold zone close {close:.2f} < floor {hold_floor:.2f}")
    if not no_fresh_lower_low:
        failures.append("fresh lower low vs prior 3 bars")
    if not direction_expanded:
        failures.append(f"directional expansion only {directional_expansion_atr:.2f} ATR")
    if not low_vol_net_move_ok:
        if net_move_pct is None:
            failures.append("low-vol net move unavailable")
        else:
            failures.append(
                f"low-vol net move only {net_move_pct * 100:.2f}% "
                f"< {EXECUTION_SHADOW_LOW_VOL_MIN_NET_MOVE_PCT * 100:.2f}%"
            )
    if not volume_confirmed and volume_ratio is not None:
        failures.append(f"thin bullish confirmation volume {volume_ratio:.2f}x prior median")

    ok = not failures
    return {
        **base,
        "execution_shadow_ok": ok,
        "execution_shadow_reason": "Recent 4H confirmation remains structurally intact" if ok else "; ".join(failures),
        # Structured, additive -- surfaces the same numeric values already
        # computed above (not re-derived) as real data instead of only a
        # joined string, so callers like the near-miss ranking view can
        # read and rank on the actual gap sizes without parsing prose.
        "execution_shadow_failures": failures,
        "execution_shadow_diagnostics": {
            "has_recent_confirmation": has_recent_confirmation,
            "confirmation_reaction_atr": round(confirmation_reaction_atr, 4),
            "confirmation_directional_atr": round(confirmation_directional_atr, 4),
            "holds_zone": holds_zone,
            "hold_floor": round(hold_floor, 4),
            "close": round(close, 4),
            "no_fresh_lower_low": no_fresh_lower_low,
            "direction_expanded": direction_expanded,
            "directional_expansion_atr": round(directional_expansion_atr, 4),
            "directional_expansion_required_atr": EXECUTION_SHADOW_MIN_DIRECTIONAL_EXPANSION_ATR,
            "low_vol_bucket": low_vol_bucket,
            "low_vol_net_move_ok": low_vol_net_move_ok,
            "net_move_pct": round(net_move_pct, 4) if net_move_pct is not None else None,
            "low_vol_min_net_move_pct": EXECUTION_SHADOW_LOW_VOL_MIN_NET_MOVE_PCT,
            "volume_confirmed": volume_confirmed,
            "volume_ratio": round(volume_ratio, 4) if volume_ratio is not None else None,
            "volume_ratio_required": EXECUTION_SHADOW_MIN_VOLUME_RATIO,
        },
    }


def _preview_base_enter_now_ready(candidate: sqlite3.Row | dict, preview: dict) -> bool:
    direction = str(preview.get("signal") or candidate["signal"]).strip().lower()
    if direction != "long":
        return False
    if not _candidate_regime_aligned(candidate, direction):
        return False
    if preview.get("preview_error"):
        return False
    if preview.get("no_valid_target") or preview.get("target") is None or preview.get("risk_reward") is None:
        return False
    if preview.get("rr_warning") or float(preview.get("risk_reward") or 0) < RR_WARNING_THRESHOLD:
        return False
    # Contract quality (spread/liquidity/DTE/delta) is informational only --
    # demoted from a hard gate, same pattern as the AI chart-read demotion.
    # The user verifies the real chain in her broker; Kairos just needs to
    # show a suggested strike, not validate one. See _contract_block_reason
    # for the (now unused-for-gating) quality assessment, still exposed via
    # option_contract fields for display.
    return bool(preview.get("entry_proximity_ok"))


def _gate_gap_report(candidate: sqlite3.Row | dict, preview: dict) -> dict:
    """Evaluate every GRADABLE ENTER_NOW gate condition independently (no
    short-circuiting) and return the real, already-computed numeric gap for
    each one -- for the near-miss ranking view. This reuses the exact
    values the real gates compute (routeBlockReason's backend twin
    _preview_base_enter_now_ready, _entry_proximity, and
    _execution_shadow_from_bars' new structured diagnostics); it does not
    re-derive any threshold comparison itself.

    Direction (long-only) and regime alignment are categorical, not
    gradable -- a short candidate is never "close" to being long no matter
    how good its other numbers are, so those two disqualify a candidate
    from the near-miss list entirely rather than counting as one gap among
    several.
    """
    direction = str(preview.get("signal") or candidate["signal"] or "").strip().lower()
    regime_aligned = _candidate_regime_aligned(candidate, direction)
    if direction == "short":
        return {"categorical_blocked": True, "categorical_reason": "Shorts are research-only", "failing_count": None, "gaps": []}
    if direction != "long":
        return {"categorical_blocked": True, "categorical_reason": "Unsupported direction", "failing_count": None, "gaps": []}
    if not regime_aligned:
        return {"categorical_blocked": True, "categorical_reason": "Regime is not aligned", "failing_count": None, "gaps": []}

    gaps: list[dict[str, str]] = []
    no_target = bool(preview.get("no_valid_target")) or preview.get("target") is None
    if no_target:
        gaps.append({"condition": "valid_target", "detail": "No valid structural target found"})

    rr = preview.get("risk_reward")
    if not no_target:
        if rr is None:
            gaps.append({"condition": "risk_reward", "detail": "R:R unavailable"})
        elif preview.get("rr_warning") or float(rr) < RR_WARNING_THRESHOLD:
            gaps.append({
                "condition": "risk_reward",
                "detail": f"R:R {float(rr):.2f} -- needs {RR_WARNING_THRESHOLD:.2f} (off by {RR_WARNING_THRESHOLD - float(rr):.2f})",
            })

    if not preview.get("entry_proximity_ok"):
        pct = preview.get("entry_distance_pct")
        atr_d = preview.get("entry_distance_atr")
        max_pct = preview.get("entry_proximity_threshold_pct", ENTRY_PROXIMITY_MAX_PCT_DEFAULT)
        max_atr = preview.get("entry_proximity_threshold_atr", ENTRY_PROXIMITY_MAX_ATR_MULTIPLE_DEFAULT)
        if pct is not None:
            atr_txt = f" / {atr_d:.2f} ATR" if atr_d is not None else ""
            gaps.append({
                "condition": "entry_proximity",
                "detail": f"Entry moved {pct:.2f}%{atr_txt} from scan -- outside proximity tolerance (max {max_pct:.1f}% / {max_atr:.2f} ATR)",
            })
        else:
            gaps.append({
                "condition": "entry_proximity",
                "detail": preview.get("entry_proximity_reason") or "Entry proximity unavailable",
            })

    # Execution-shadow sub-conditions only get evaluated once the real
    # system would actually reach them -- reuse the real gate function
    # rather than re-checking target/RR/proximity a second time here.
    if _preview_base_enter_now_ready(candidate, preview) and preview.get("execution_shadow_checked") and preview.get("execution_shadow_ok") is not True:
        diagnostics = preview.get("execution_shadow_diagnostics") or {}
        if diagnostics.get("has_recent_confirmation") is False:
            gaps.append({
                "condition": "execution_confirmation",
                "detail": (
                    f"Execution: no recent bullish confirmation "
                    f"(best reaction {diagnostics.get('confirmation_reaction_atr', 0):.2f} ATR, "
                    f"expansion {diagnostics.get('confirmation_directional_atr', 0):.2f} ATR)"
                ),
            })
        if diagnostics.get("direction_expanded") is False:
            atr_val = diagnostics.get("directional_expansion_atr")
            req = diagnostics.get("directional_expansion_required_atr", EXECUTION_SHADOW_MIN_DIRECTIONAL_EXPANSION_ATR)
            if atr_val is not None:
                gaps.append({
                    "condition": "directional_expansion",
                    "detail": f"Execution: directional expansion {atr_val:+.2f} ATR -- needs +{req:.2f} ATR (off by {req - atr_val:.2f})",
                })
        if diagnostics.get("volume_confirmed") is False:
            ratio = diagnostics.get("volume_ratio")
            req = diagnostics.get("volume_ratio_required", EXECUTION_SHADOW_MIN_VOLUME_RATIO)
            if ratio is not None:
                gaps.append({
                    "condition": "bullish_volume",
                    "detail": f"Execution: bullish-window volume {ratio:.2f}x prior median -- needs {req:.2f}x (off by {req - ratio:.2f})",
                })
        if diagnostics.get("holds_zone") is False:
            gaps.append({
                "condition": "hold_zone",
                "detail": f"Execution: close {diagnostics.get('close')} lost hold zone (floor {diagnostics.get('hold_floor')})",
            })
        if diagnostics.get("no_fresh_lower_low") is False:
            gaps.append({"condition": "fresh_lower_low", "detail": "Execution: fresh lower low vs prior 3 bars"})
        if diagnostics.get("low_vol_bucket") and diagnostics.get("low_vol_net_move_ok") is False:
            net = diagnostics.get("net_move_pct")
            req = diagnostics.get("low_vol_min_net_move_pct", EXECUTION_SHADOW_LOW_VOL_MIN_NET_MOVE_PCT)
            if net is not None:
                gaps.append({
                    "condition": "low_vol_guard",
                    "detail": f"Execution: low-vol net move {net * 100:.2f}% -- needs {req * 100:.2f}% (off by {(req - net) * 100:.2f} pts)",
                })
        if not gaps and preview.get("execution_shadow_reason"):
            # Fallback: execution failed but none of the structured diagnostics
            # matched (shouldn't normally happen) -- surface the real reason text
            # rather than silently reporting zero gaps for a failing candidate.
            gaps.append({"condition": "execution_confirmation", "detail": f"Execution: {preview['execution_shadow_reason']}"})

    return {"categorical_blocked": False, "categorical_reason": None, "failing_count": len(gaps), "gaps": gaps}


def _attach_execution_shadow(candidate: sqlite3.Row | dict, preview: dict) -> dict:
    if not _preview_base_enter_now_ready(candidate, preview):
        return {
            **preview,
            "execution_shadow_checked": False,
            "execution_shadow_ok": None,
            "execution_shadow_reason": "Not checked until base ENTER_NOW gate passes",
            "execution_shadow_version": "4h-confirmed-recently-still-intact-shadow-v10",
            "execution_shadow_candle_time": None,
        }
    bars = _recent_4h_bars_for_execution_shadow(str(preview.get("ticker") or candidate["ticker"]))
    return {**preview, **_execution_shadow_from_bars(candidate, preview, bars)}


def _entry_proximity_block_reason(entry_price: Optional[float], atr14: Optional[float], ticker: str) -> Optional[str]:
    proximity = _entry_proximity(
        entry_price=entry_price,
        atr14=atr14,
        quote=_latest_quote_for_ticker(ticker),
    )
    if proximity.get("entry_proximity_ok"):
        return None
    return str(proximity.get("entry_proximity_reason") or "Price is not near entry")


def _promotion_with_live_gate_context(
    candidate: sqlite3.Row | dict[str, Any],
    promotion: dict[str, Any],
    option_contract: Optional[dict[str, Any]],
) -> dict[str, Any]:
    ticker = str(promotion.get("ticker") or candidate["ticker"]).strip().upper()
    enriched = {
        **promotion,
        "ticker": ticker,
        "signal": promotion.get("direction") or candidate["signal"],
        "option_contract": option_contract,
    }
    enriched = _attach_entry_proximity(enriched, _latest_quote_for_ticker(ticker))
    return _attach_execution_shadow(candidate, enriched)


def _preview_has_transient_option_unavailable(row: sqlite3.Row) -> bool:
    raw_contract = row["option_contract_json"] if "option_contract_json" in row.keys() else None
    if not raw_contract:
        return False
    try:
        contract = json.loads(raw_contract)
    except json.JSONDecodeError:
        return True
    reason = str(contract.get("reason") or "").lower()
    source = str(contract.get("source") or "").lower()
    return (
        bool(contract.get("transient_unavailable"))
        or source in {"unavailable", "data_unavailable"}
        or "no option expirations available" in reason
        or "option expiration data unavailable" in reason
    )


def _preview_transient_refresh_due(row: sqlite3.Row) -> bool:
    if not _preview_has_transient_option_unavailable(row):
        return False
    computed = _coerce_iso_datetime(row["computed_at"] if "computed_at" in row.keys() else None)
    if computed is None:
        return True
    return datetime.now(timezone.utc) - computed >= CANDIDATE_PREVIEW_TRANSIENT_OPTION_REFRESH_TTL


def _coerce_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _anthropic_api_key() -> Optional[str]:
    return os.environ.get("ANTHROPIC_API_KEY")


def _anthropic_model() -> str:
    return os.environ.get("ANTHROPIC_MODEL") or ANTHROPIC_MODEL_DEFAULT


def _candidate_chart_review_caveat() -> str:
    return "Informational pattern read only, not a recommendation or automated approval."


def _parse_chart_review_text(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def _alpaca_daily_bars_for_review(ticker: str):
    try:
        raw = AlpacaMarketDataProvider().download([ticker], period="1y", interval="1d", auto_adjust=True)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"Alpaca price data unavailable: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Alpaca price data request failed: {exc.__class__.__name__}")
    if raw is None or getattr(raw, "empty", True):
        raise HTTPException(status_code=422, detail=f"No Alpaca daily candles available for {ticker}")
    try:
        df = _flatten_columns(raw.copy()).dropna().astype(float)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Alpaca daily candles for {ticker} are not usable: {exc.__class__.__name__}")
    if len(df) < 60:
        raise HTTPException(status_code=422, detail=f"Not enough Alpaca daily candles for {ticker}")
    return df


def _compact_chart_bars(df, limit: int = 90) -> list[dict]:
    bars = []
    for index, row in df.tail(limit).iterrows():
        bars.append({
            "date": str(getattr(index, "date", lambda: index)()),
            "open": round(float(row["Open"]), 4),
            "high": round(float(row["High"]), 4),
            "low": round(float(row["Low"]), 4),
            "close": round(float(row["Close"]), 4),
            "volume": int(float(row["Volume"])) if "Volume" in row else None,
        })
    return bars


def _chart_review_prompt(candidate: sqlite3.Row, bars: list[dict]) -> str:
    payload = {
        "ticker": str(candidate["ticker"]).upper(),
        "signal": str(candidate["signal"]).lower(),
        "entry_price": candidate["entry_price"],
        "daily_regime": candidate["daily_regime"],
        "confidence": candidate["confidence"],
        "sma50_daily": candidate["sma50_daily"],
        "sma200_daily": candidate["sma200_daily"],
        "bars": bars,
    }
    return (
        "You are giving Erica a second-opinion chart note for a Kairos scanner candidate. "
        "This is pattern-reading on OHLCV data, not a recommendation, not a probability, and not approval/rejection. "
        "Use the rubric from prior manual reviews: fresh clean structural break/retest is strongest; genuine orderly trend is good; "
        "choppy range-bound action should be flagged; already-played-out gap/decline that has stabilized is lower quality; "
        "persistent grinding continuation with no fresh event is lower priority. "
        "Return strict JSON only with keys classification and rationale. "
        "classification must be one of: fresh_clean_structural_break, genuine_trending_move, choppy_range_bound, "
        "played_out_stabilized, grinding_no_fresh_event, mixed_unclear. "
        "rationale must be 2-3 short sentences grounded in the provided bars. "
        f"Candidate data: {json.dumps(payload, separators=(',', ':'))}"
    )


def _call_anthropic_chart_review(prompt: str) -> tuple[dict, str, str]:
    api_key = _anthropic_api_key()
    if not api_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured on server")
    model = _anthropic_model()
    body = {
        "model": model,
        "max_tokens": 450,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }
    req = Request(
        ANTHROPIC_MESSAGES_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=int(os.environ.get("ANTHROPIC_TIMEOUT_SECONDS", "45"))) as response:
            raw_response = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8")[:500]
        raise HTTPException(status_code=502, detail=f"Anthropic request failed with HTTP {exc.code}: {detail}")
    except (URLError, TimeoutError) as exc:
        raise HTTPException(status_code=502, detail=f"Anthropic request failed: {exc.__class__.__name__}")

    try:
        payload = json.loads(raw_response)
        text = "\n".join(
            block.get("text", "")
            for block in payload.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
        parsed = _parse_chart_review_text(text)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Anthropic response was not usable JSON: {exc.__class__.__name__}")
    return parsed, raw_response, model


def _compute_candidate_chart_review(candidate: sqlite3.Row) -> dict:
    ticker = str(candidate["ticker"] or "").strip().upper()
    signal = str(candidate["signal"] or "").strip().lower()
    if signal not in {"long", "short"}:
        raise HTTPException(status_code=422, detail=f"Unsupported candidate signal for {ticker}: {signal}")
    df = _alpaca_daily_bars_for_review(ticker)
    bars = _compact_chart_bars(df)
    parsed, raw_response, model = _call_anthropic_chart_review(_chart_review_prompt(candidate, bars))
    allowed = {
        "fresh_clean_structural_break",
        "genuine_trending_move",
        "choppy_range_bound",
        "played_out_stabilized",
        "grinding_no_fresh_event",
        "mixed_unclear",
    }
    classification = str(parsed.get("classification") or "").strip()
    rationale = str(parsed.get("rationale") or "").strip()
    if classification not in allowed:
        raise HTTPException(status_code=502, detail=f"Anthropic returned unsupported classification: {classification}")
    if not rationale:
        raise HTTPException(status_code=502, detail="Anthropic returned empty rationale")
    reviewed_at = datetime.now(timezone.utc).isoformat()
    return {
        "ticker": ticker,
        "source": str(candidate["source"]),
        "signal": signal,
        "classification": classification,
        "rationale": rationale[:1200],
        "caveat": _candidate_chart_review_caveat(),
        "reviewed_at": reviewed_at,
        "model": model,
        "rubric_version": AI_CHART_REVIEW_RUBRIC_VERSION,
        "data_source": "alpaca_adjusted_daily_ohlcv",
        "bars_start": bars[0]["date"] if bars else None,
        "bars_end": bars[-1]["date"] if bars else None,
        "raw_response": raw_response,
    }


def _store_chart_review(conn, review: dict) -> None:
    conn.execute(
        """
        INSERT INTO candidate_ai_chart_reviews
            (ticker, source, signal, classification, rationale, caveat, reviewed_at,
             model, rubric_version, data_source, bars_start, bars_end, raw_response)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, source) DO UPDATE SET
            signal=excluded.signal,
            classification=excluded.classification,
            rationale=excluded.rationale,
            caveat=excluded.caveat,
            reviewed_at=excluded.reviewed_at,
            model=excluded.model,
            rubric_version=excluded.rubric_version,
            data_source=excluded.data_source,
            bars_start=excluded.bars_start,
            bars_end=excluded.bars_end,
            raw_response=excluded.raw_response
        """,
        (
            review["ticker"],
            review["source"],
            review["signal"],
            review["classification"],
            review["rationale"],
            review["caveat"],
            review["reviewed_at"],
            review["model"],
            review["rubric_version"],
            review["data_source"],
            review["bars_start"],
            review["bars_end"],
            review["raw_response"],
        ),
    )


def _nearest_structural_target(
    entry: float,
    direction: str,
    swings: list,
    min_distance: float,
) -> Optional[float]:
    if direction == "long":
        highs = sorted(float(swing["price"]) for swing in swings if swing.get("type") == "high" and float(swing["price"]) > entry)
        return next((price for price in highs if abs(price - entry) >= min_distance), None)
    if direction == "short":
        lows = sorted(
            (float(swing["price"]) for swing in swings if swing.get("type") == "low" and float(swing["price"]) < entry),
            reverse=True,
        )
        return next((price for price in lows if abs(price - entry) >= min_distance), None)
    return None


def _compute_candidate_promotion(candidate: sqlite3.Row) -> dict:
    ticker = str(candidate["ticker"] or "").strip().upper()
    direction = str(candidate["signal"] or "").strip().lower()
    entry = candidate["entry_price"]
    if direction not in {"long", "short"}:
        raise HTTPException(status_code=422, detail=f"Unsupported candidate signal for {ticker}: {direction}")
    if entry is None:
        raise HTTPException(status_code=422, detail=f"Candidate {ticker} has no entry_price")

    try:
        entry_price = float(entry)
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail=f"Candidate {ticker} has invalid entry_price")
    if entry_price <= 0:
        raise HTTPException(status_code=422, detail=f"Candidate {ticker} entry_price must be positive")

    daily = _batch_download([ticker], period="1y", interval="1d").get(ticker)
    if daily is None or getattr(daily, "empty", True):
        raise HTTPException(status_code=422, detail=f"No daily candles available for {ticker}")
    try:
        df = _flatten_columns(daily.copy()).dropna().astype(float)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Daily candles for {ticker} are not usable: {exc.__class__.__name__}")
    if len(df) < 20:
        raise HTTPException(status_code=422, detail=f"Not enough daily candles for {ticker}")

    atr14 = float(_compute_atr(df, period=14))
    if atr14 <= 0:
        raise HTTPException(status_code=422, detail=f"ATR14 is not usable for {ticker}")

    # swings computed once, up front -- shared by the order-block stop check
    # below, the structural target lookup, the target-resistance check
    # further down, and the BOS confirmation check.
    swings = _find_swings(df)

    # BOS (break-of-structure) confirmation -- additive overlay/flag ONLY.
    # Explicitly does NOT gate ENTER_NOW eligibility or block promotion; that
    # call stays with the human, same category as the R:R/stop-distance/
    # displacement-weight decisions already flagged elsewhere in this file.
    # Reuses scanner._detect_bos (its public alias scanner.detect_structure_break
    # is confirmed to be a thin pass-through wrapper around this exact function --
    # nothing different to reuse instead), which requires a close beyond the
    # prior (second-most-recent) same-type swing WITH same-direction candle
    # body confirmation -- a wick poke through the level alone doesn't count.
    # direction.upper() here for the same reason as _find_order_block below:
    # scanner.py's BOS/order-block functions use uppercase "LONG"/"SHORT",
    # while this module and structural_resistance.py use lowercase.
    bos_confirmed, bos_level = _detect_bos(df, swings, direction.upper())
    bos_details = {"break_level": round(float(bos_level), 4)} if bos_confirmed else None

    # Continuous displacement/conviction score for the most recent daily
    # candle -- informational grading input only, never a gate (see
    # displacement_score.score_displacement's docstring). Independent of the
    # stop/target logic below; computed here purely for locality with the
    # other df-derived signals.
    displacement = score_displacement(df, direction)

    # Order-block stop: replaces the flat entry +/- ATR_MULTIPLIER stop with
    # the actual invalidation level of the most recent order block, when one
    # exists on the correct side of entry. raw_stop (the flat ATR value) is
    # always preserved for the audit trail regardless of which stop is live
    # -- see resolve_stop's docstring for the "clean" fallback rule.
    order_block = _find_order_block(df, direction.upper(), swings)
    stop_resolution = resolve_stop(entry_price, direction, atr14, ATR_MULTIPLIER, order_block)
    stop = stop_resolution["stop"]
    raw_stop = stop_resolution["raw_stop"]
    stop_source = stop_resolution["stop_source"]

    risk = abs(entry_price - stop)
    if risk <= 0:
        raise HTTPException(status_code=422, detail=f"Computed risk is not usable for {ticker}")

    min_target_atr_multiple = _min_target_atr_multiple()
    min_target_distance = min_target_atr_multiple * atr14
    target = _nearest_structural_target(entry_price, direction, swings, min_target_distance)
    no_valid_target = target is None
    risk_reward = None
    rr_warning = True
    if target is not None:
        reward = abs(target - entry_price)
        risk_reward = reward / risk
        rr_warning = risk_reward < RR_WARNING_THRESHOLD

    # Structural resistance/support check: does the raw target land near a
    # gap-day spike (unreliable) or a genuine swing pivot (real structure,
    # possibly already rejected)? If so, clamp the target back to just this
    # side of it and recompute R:R -- but always preserve the raw/unclamped
    # values below for the audit trail, and never let the clamp itself
    # produce a broken or degenerate plan (see clamp_target's floor guard).
    raw_target = target
    raw_risk_reward = risk_reward
    target_clamped = False
    target_clamp_badge = None
    target_clamp_reason = None
    if target is not None:
        findings = levels_near_target(df, swings, target, atr14, direction)
        if findings:
            clamp = clamp_target(
                entry_price, stop, target, atr14, findings,
                direction=direction, min_viable_rr=RR_WARNING_THRESHOLD,
            )
            target_clamp_badge = clamp["badge"]
            target_clamp_reason = clamp["clamp_refused_reason"]
            if clamp["clamped"]:
                target = clamp["adjusted_target"]
                risk_reward = clamp["adjusted_rr"]
                rr_warning = risk_reward < RR_WARNING_THRESHOLD
                target_clamped = True

    promoted_at = datetime.now(timezone.utc).isoformat()
    return {
        "ticker": ticker,
        "source": str(candidate["source"]),
        "direction": direction,
        "entry_price": round(entry_price, 4),
        "stop": round(stop, 4),
        "target": round(float(target), 4) if target is not None else None,
        "risk_reward": round(risk_reward, 2) if risk_reward is not None else None,
        "rr_warning": rr_warning,
        "no_valid_target": no_valid_target,
        "promoted_at": promoted_at,
        "position_size": None,
        "atr14": round(atr14, 4),
        "atr_multiplier": ATR_MULTIPLIER,
        "rr_warning_threshold": RR_WARNING_THRESHOLD,
        "min_target_atr_multiple": min_target_atr_multiple,
        "target_source": "daily_swing_structure",
        "raw_target": round(float(raw_target), 4) if raw_target is not None else None,
        "raw_risk_reward": round(raw_risk_reward, 2) if raw_risk_reward is not None else None,
        "target_clamped": target_clamped,
        "target_clamp_badge": target_clamp_badge,
        "target_clamp_reason": target_clamp_reason,
        "raw_stop": round(float(raw_stop), 4),
        "stop_source": stop_source,
        "displacement_score": displacement["score"],
        "displacement_label": displacement["label"],
        "displacement_components": displacement["components"],
        "raw_magnitude_score": displacement["raw_magnitude_score"],
        "displacement_read": displacement["displacement_read"],
        "bos_confirmed": bos_confirmed,
        "bos_details": bos_details,
    }


def _candidate_regime_aligned(candidate: sqlite3.Row, direction: str) -> bool:
    regime = str(candidate["daily_regime"] if "daily_regime" in candidate.keys() else "").strip().lower()
    if direction == "long":
        return "long" in regime or "bull" in regime
    if direction == "short":
        return "short" in regime or "bear" in regime
    return False


def _promotion_block_reason(
    candidate: sqlite3.Row | dict[str, Any],
    promotion: dict,
    option_contract: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    direction = str(promotion.get("direction") or "").strip().lower()
    if direction == "short":
        return "Short candidates are research-only and cannot be promoted to the clean dashboard."
    if direction != "long":
        return "Unsupported candidate direction."
    if not _candidate_regime_aligned(candidate, direction):
        return "Candidate regime is not aligned, so it is not ENTER_NOW dashboard-ready."
    if promotion.get("no_valid_target") or promotion.get("target") is None or promotion.get("risk_reward") is None:
        return "Candidate has no valid target, so it is not ENTER_NOW dashboard-ready."
    if promotion.get("rr_warning") or float(promotion.get("risk_reward") or 0) < RR_WARNING_THRESHOLD:
        return f"Candidate R:R is below {RR_WARNING_THRESHOLD}:1, so it is not ENTER_NOW dashboard-ready."
    # Contract quality is informational only, not an ENTER_NOW/promotion gate
    # -- see _preview_base_enter_now_ready for the matching change.
    if "entry_proximity_ok" in promotion:
        proximity_reason = None if promotion.get("entry_proximity_ok") else str(
            promotion.get("entry_proximity_reason") or "Price is not near entry"
        )
    else:
        proximity_reason = _entry_proximity_block_reason(
            promotion.get("entry_price"),
            promotion.get("atr14"),
            str(promotion.get("ticker") or candidate["ticker"]),
        )
    if proximity_reason:
        return f"Candidate is not ENTER_NOW-ready: {proximity_reason}."
    if promotion.get("execution_shadow_ok") is not True:
        execution_reason = str(promotion.get("execution_shadow_reason") or "Recent 4H confirmation is not ready")
        return f"Candidate execution confirmation is not ENTER_NOW-ready: {execution_reason}."
    return None


def _store_promotion(conn, promotion: dict) -> None:
    conn.execute(
        """
        INSERT INTO candidate_promotions
            (ticker, source, direction, entry_price, stop, target, risk_reward,
             rr_warning, no_valid_target, promoted_at, position_size, atr14,
             atr_multiplier, rr_warning_threshold, min_target_atr_multiple,
             target_source, raw_target, raw_risk_reward, target_clamped,
             target_clamp_badge, target_clamp_reason, raw_stop, stop_source,
             displacement_score, displacement_label, displacement_components_json,
             raw_magnitude_score, displacement_read, bos_confirmed, bos_level)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, source) DO UPDATE SET
            direction=excluded.direction,
            entry_price=excluded.entry_price,
            stop=excluded.stop,
            target=excluded.target,
            risk_reward=excluded.risk_reward,
            rr_warning=excluded.rr_warning,
            no_valid_target=excluded.no_valid_target,
            promoted_at=excluded.promoted_at,
            position_size=excluded.position_size,
            atr14=excluded.atr14,
            atr_multiplier=excluded.atr_multiplier,
            rr_warning_threshold=excluded.rr_warning_threshold,
            min_target_atr_multiple=excluded.min_target_atr_multiple,
            target_source=excluded.target_source,
            raw_target=excluded.raw_target,
            raw_risk_reward=excluded.raw_risk_reward,
            target_clamped=excluded.target_clamped,
            target_clamp_badge=excluded.target_clamp_badge,
            target_clamp_reason=excluded.target_clamp_reason,
            raw_stop=excluded.raw_stop,
            stop_source=excluded.stop_source,
            displacement_score=excluded.displacement_score,
            displacement_label=excluded.displacement_label,
            displacement_components_json=excluded.displacement_components_json,
            raw_magnitude_score=excluded.raw_magnitude_score,
            displacement_read=excluded.displacement_read,
            bos_confirmed=excluded.bos_confirmed,
            bos_level=excluded.bos_level
        """,
        (
            promotion["ticker"],
            promotion["source"],
            promotion["direction"],
            promotion["entry_price"],
            promotion["stop"],
            promotion["target"],
            promotion["risk_reward"],
            1 if promotion["rr_warning"] else 0,
            1 if promotion["no_valid_target"] else 0,
            promotion["promoted_at"],
            promotion["position_size"],
            promotion["atr14"],
            promotion["atr_multiplier"],
            promotion["rr_warning_threshold"],
            promotion["min_target_atr_multiple"],
            promotion["target_source"],
            promotion.get("raw_target"),
            promotion.get("raw_risk_reward"),
            1 if promotion.get("target_clamped") else 0,
            promotion.get("target_clamp_badge"),
            promotion.get("target_clamp_reason"),
            promotion.get("raw_stop"),
            promotion.get("stop_source"),
            promotion.get("displacement_score"),
            promotion.get("displacement_label"),
            json.dumps(promotion["displacement_components"], separators=(",", ":")) if promotion.get("displacement_components") else None,
            promotion.get("raw_magnitude_score"),
            promotion.get("displacement_read"),
            1 if promotion.get("bos_confirmed") else 0,
            _bos_level_for_storage(promotion),
        ),
    )


def _compute_candidate_plan_preview(candidate: sqlite3.Row) -> dict:
    computed_at = datetime.now(timezone.utc).isoformat()
    try:
        promotion_like = _compute_candidate_promotion(candidate)
        direction = str(promotion_like["direction"])
        option_contract = _safe_option_contract_for_candidate(
            promotion_like["ticker"],
            direction,
            promotion_like["entry_price"],
        )
        return {
            "ticker": promotion_like["ticker"],
            "source": promotion_like["source"],
            "signal": direction,
            "entry_price": promotion_like["entry_price"],
            "stop": promotion_like["stop"],
            "target": promotion_like["target"],
            "risk_reward": promotion_like["risk_reward"],
            "rr_warning": promotion_like["rr_warning"],
            "no_valid_target": promotion_like["no_valid_target"],
            "atr14": promotion_like["atr14"],
            "atr_multiplier": promotion_like["atr_multiplier"],
            "rr_warning_threshold": promotion_like["rr_warning_threshold"],
            "min_target_atr_multiple": promotion_like["min_target_atr_multiple"],
            "target_source": promotion_like["target_source"],
            "raw_target": promotion_like.get("raw_target"),
            "raw_risk_reward": promotion_like.get("raw_risk_reward"),
            "target_clamped": promotion_like.get("target_clamped", False),
            "target_clamp_badge": promotion_like.get("target_clamp_badge"),
            "target_clamp_reason": promotion_like.get("target_clamp_reason"),
            "raw_stop": promotion_like.get("raw_stop"),
            "stop_source": promotion_like.get("stop_source"),
            "displacement_score": promotion_like.get("displacement_score"),
            "displacement_label": promotion_like.get("displacement_label"),
            "displacement_components": promotion_like.get("displacement_components"),
            "raw_magnitude_score": promotion_like.get("raw_magnitude_score"),
            "displacement_read": promotion_like.get("displacement_read"),
            "bos_confirmed": promotion_like.get("bos_confirmed", False),
            "bos_details": promotion_like.get("bos_details"),
            "option_contract": option_contract,
            "preview_error": None,
            "computed_at": computed_at,
            "candidate_updated_at": candidate["updated_at"],
        }
    except HTTPException as exc:
        return {
            "ticker": str(candidate["ticker"] or "").strip().upper(),
            "source": str(candidate["source"]),
            "signal": str(candidate["signal"] or "").strip().lower(),
            "entry_price": candidate["entry_price"],
            "stop": None,
            "target": None,
            "risk_reward": None,
            "rr_warning": True,
            "no_valid_target": True,
            "atr14": None,
            "atr_multiplier": ATR_MULTIPLIER,
            "rr_warning_threshold": RR_WARNING_THRESHOLD,
            "min_target_atr_multiple": _min_target_atr_multiple(),
            "target_source": "daily_swing_structure",
            "raw_target": None,
            "raw_risk_reward": None,
            "target_clamped": False,
            "target_clamp_badge": None,
            "target_clamp_reason": None,
            "raw_stop": None,
            "stop_source": None,
            "displacement_score": None,
            "displacement_label": None,
            "displacement_components": None,
            "raw_magnitude_score": None,
            "displacement_read": None,
            "bos_confirmed": False,
            "bos_details": None,
            "option_contract": None,
            "preview_error": str(exc.detail),
            "computed_at": computed_at,
            "candidate_updated_at": candidate["updated_at"],
        }


def _store_plan_preview(conn, preview: dict) -> None:
    option_contract = preview.get("option_contract")
    conn.execute(
        """
        INSERT INTO candidate_plan_previews
            (ticker, source, signal, entry_price, stop, target, risk_reward,
             rr_warning, no_valid_target, atr14, atr_multiplier, rr_warning_threshold,
             min_target_atr_multiple, target_source, option_contract_json, preview_error,
             computed_at, candidate_updated_at, raw_target, raw_risk_reward,
             target_clamped, target_clamp_badge, target_clamp_reason,
             raw_stop, stop_source, displacement_score, displacement_label,
             displacement_components_json, raw_magnitude_score, displacement_read,
             bos_confirmed, bos_level)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, source) DO UPDATE SET
            signal=excluded.signal,
            entry_price=excluded.entry_price,
            stop=excluded.stop,
            target=excluded.target,
            risk_reward=excluded.risk_reward,
            rr_warning=excluded.rr_warning,
            no_valid_target=excluded.no_valid_target,
            atr14=excluded.atr14,
            atr_multiplier=excluded.atr_multiplier,
            rr_warning_threshold=excluded.rr_warning_threshold,
            min_target_atr_multiple=excluded.min_target_atr_multiple,
            target_source=excluded.target_source,
            option_contract_json=excluded.option_contract_json,
            preview_error=excluded.preview_error,
            computed_at=excluded.computed_at,
            candidate_updated_at=excluded.candidate_updated_at,
            raw_target=excluded.raw_target,
            raw_risk_reward=excluded.raw_risk_reward,
            target_clamped=excluded.target_clamped,
            target_clamp_badge=excluded.target_clamp_badge,
            target_clamp_reason=excluded.target_clamp_reason,
            raw_stop=excluded.raw_stop,
            stop_source=excluded.stop_source,
            displacement_score=excluded.displacement_score,
            displacement_label=excluded.displacement_label,
            displacement_components_json=excluded.displacement_components_json,
            raw_magnitude_score=excluded.raw_magnitude_score,
            displacement_read=excluded.displacement_read,
            bos_confirmed=excluded.bos_confirmed,
            bos_level=excluded.bos_level
        """,
        (
            preview["ticker"],
            preview["source"],
            preview["signal"],
            preview["entry_price"],
            preview["stop"],
            preview["target"],
            preview["risk_reward"],
            1 if preview["rr_warning"] else 0,
            1 if preview["no_valid_target"] else 0,
            preview["atr14"],
            preview["atr_multiplier"],
            preview["rr_warning_threshold"],
            preview["min_target_atr_multiple"],
            preview["target_source"],
            json.dumps(option_contract, separators=(",", ":")) if option_contract else None,
            preview["preview_error"],
            preview["computed_at"],
            preview["candidate_updated_at"],
            preview.get("raw_target"),
            preview.get("raw_risk_reward"),
            1 if preview.get("target_clamped") else 0,
            preview.get("target_clamp_badge"),
            preview.get("target_clamp_reason"),
            preview.get("raw_stop"),
            preview.get("stop_source"),
            preview.get("displacement_score"),
            preview.get("displacement_label"),
            json.dumps(preview["displacement_components"], separators=(",", ":")) if preview.get("displacement_components") else None,
            preview.get("raw_magnitude_score"),
            preview.get("displacement_read"),
            1 if preview.get("bos_confirmed") else 0,
            _bos_level_for_storage(preview),
        ),
    )


@router.post("/candidates", response_model=IngestResponse)
def ingest_candidates(payload: ShortlistIn, x_api_key: Optional[str] = Header(default=None)):
    _check_api_key(x_api_key)
    return upsert_candidate_shortlist(payload)


@router.post("/session")
def create_scanner_session(payload: ScannerSessionIn, response: Response):
    _check_api_key(payload.api_key)
    response.set_cookie(
        key=SCANNER_SESSION_COOKIE,
        value=payload.api_key,
        max_age=60 * 60 * 24 * 365,
        httponly=True,
        secure=True,
        samesite="lax",
        path="/",
    )
    return {"authenticated": True}


@router.delete("/session")
def clear_scanner_session(response: Response):
    response.delete_cookie(key=SCANNER_SESSION_COOKIE, path="/")
    return {"authenticated": False}


def upsert_candidate_shortlist(payload: ShortlistIn) -> IngestResponse:
    conn = _get_db()
    created, updated = 0, 0
    rejected: list[RejectedEntry] = []
    now = datetime.now(timezone.utc).isoformat()
    scanned_at = payload.scanned_at.astimezone(timezone.utc).isoformat()

    try:
        for candidate in payload.candidates:
            ticker = candidate.ticker.strip().upper()
            if not _valid_ticker(ticker):
                rejected.append(RejectedEntry(ticker=candidate.ticker, reason="invalid ticker format"))
                continue

            existing = conn.execute(
                "SELECT 1 FROM candidates WHERE ticker=? AND source=?",
                (ticker, payload.source),
            ).fetchone()

            conn.execute(
                """
                INSERT INTO candidates
                    (ticker, source, signal, entry_price, ema21_4h, daily_regime,
                     confidence, sma50_daily, sma200_daily, status, scanned_at, updated_at,
                     source_universe)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'new', ?, ?, ?)
                ON CONFLICT(ticker, source) DO UPDATE SET
                    signal=excluded.signal,
                    entry_price=excluded.entry_price,
                    ema21_4h=excluded.ema21_4h,
                    daily_regime=excluded.daily_regime,
                    confidence=excluded.confidence,
                    sma50_daily=excluded.sma50_daily,
                    sma200_daily=excluded.sma200_daily,
                    scanned_at=excluded.scanned_at,
                    updated_at=excluded.updated_at,
                    source_universe=excluded.source_universe
                """,
                (
                    ticker,
                    payload.source,
                    candidate.signal,
                    candidate.entry_price,
                    candidate.ema21_4h,
                    candidate.daily_regime,
                    candidate.confidence,
                    candidate.sma50_daily,
                    candidate.sma200_daily,
                    scanned_at,
                    now,
                    candidate.source_universe,
                ),
            )

            conn.execute(
                """
                INSERT INTO candidate_history
                    (ticker, source, signal, entry_price, ema21_4h, daily_regime,
                     confidence, sma50_daily, sma200_daily, scanned_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker,
                    payload.source,
                    candidate.signal,
                    candidate.entry_price,
                    candidate.ema21_4h,
                    candidate.daily_regime,
                    candidate.confidence,
                    candidate.sma50_daily,
                    candidate.sma200_daily,
                    scanned_at,
                ),
            )

            if existing:
                updated += 1
            else:
                created += 1

        conn.commit()
    finally:
        conn.close()

    return IngestResponse(
        received=len(payload.candidates),
        created=created,
        updated=updated,
        rejected=rejected,
    )


@router.get("/candidates", response_model=list[CandidateOut])
def list_candidates(
    status: Optional[CandidateStatus] = Query(default=None),
    x_api_key: Optional[str] = Header(default=None),
    scanner_session: Optional[str] = Cookie(default=None, alias=SCANNER_SESSION_COOKIE),
):
    _check_api_key(x_api_key, scanner_session)
    conn = _get_db()
    try:
        if status:
            rows = conn.execute(
                "SELECT * FROM candidates WHERE status=? ORDER BY updated_at DESC",
                (status,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM candidates ORDER BY updated_at DESC").fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


@router.get("/candidate-promotions", response_model=list[CandidatePromotionOut])
def list_candidate_promotions(
    x_api_key: Optional[str] = Header(default=None),
    scanner_session: Optional[str] = Cookie(default=None, alias=SCANNER_SESSION_COOKIE),
):
    _check_api_key(x_api_key, scanner_session)
    conn = _get_db()
    try:
        rows = conn.execute("SELECT * FROM candidate_promotions ORDER BY promoted_at DESC").fetchall()
        return [_row_to_promotion(row) for row in rows]
    finally:
        conn.close()


def _enriched_previews_for_candidates(conn, candidates: list) -> tuple[list[dict], list]:
    """Shared by /candidate-plan-previews and /candidate-near-misses: computes
    (or reuses cached) plan previews for the given candidate rows, then
    attaches live entry-proximity and execution-shadow the same way for
    both -- so the two views can never compute gate state differently."""
    previews: list[dict] = []
    for candidate in candidates:
        existing = conn.execute(
            "SELECT * FROM candidate_plan_previews WHERE ticker=? AND source=?",
            (candidate["ticker"], candidate["source"]),
        ).fetchone()
        if (
            existing
            and existing["candidate_updated_at"] == candidate["updated_at"]
            and not _preview_transient_refresh_due(existing)
        ):
            previews.append(_row_to_plan_preview(existing))
            continue
        # _compute_candidate_plan_preview() makes real network calls (regime bars,
        # option chain lookups) per candidate. Committing right after each row's
        # write -- instead of deferring one commit to the end of this loop -- keeps
        # the write lock held for one upsert's duration instead of across every
        # remaining candidate's network round-trip. With hundreds of candidates
        # needing a refresh right after a scan (their cache keys off
        # candidate_updated_at, and a scan can touch hundreds of rows at once),
        # a single end-of-loop commit held the lock open long enough to exceed the
        # busy_timeout for every other endpoint reading this database concurrently.
        preview = _compute_candidate_plan_preview(candidate)
        try:
            _store_plan_preview(conn, preview)
            conn.commit()
        except sqlite3.OperationalError as exc:
            if "database is locked" not in str(exc).lower():
                raise
            preview["preview_error"] = (
                preview.get("preview_error")
                or "Plan preview cache is temporarily busy; showing uncached computed preview."
            )
        previews.append(preview)
    quotes = _latest_quotes_for_previews(previews)
    candidates_by_key = {
        (str(candidate["ticker"]).upper(), str(candidate["source"])): candidate
        for candidate in candidates
    }
    enriched = []
    for preview in previews:
        with_proximity = _attach_entry_proximity(
            preview,
            quotes.get(str(preview.get("ticker") or "").upper()),
        )
        candidate = candidates_by_key.get(
            (str(preview.get("ticker") or "").upper(), str(preview.get("source") or ""))
        )
        enriched.append(
            _attach_execution_shadow(candidate, with_proximity) if candidate is not None else with_proximity
        )
    return enriched, candidates


@router.get("/candidate-plan-previews", response_model=list[CandidatePlanPreviewOut])
def list_candidate_plan_previews(
    x_api_key: Optional[str] = Header(default=None),
    scanner_session: Optional[str] = Cookie(default=None, alias=SCANNER_SESSION_COOKIE),
):
    _check_api_key(x_api_key, scanner_session)
    conn = _get_db()
    try:
        candidates = conn.execute("SELECT * FROM candidates ORDER BY updated_at DESC").fetchall()
        enriched, _ = _enriched_previews_for_candidates(conn, candidates)
        return enriched
    finally:
        conn.close()


class CandidateNearMissOut(BaseModel):
    ticker: str
    source: str
    signal: str
    tier: int
    failing_count: int
    gaps: list[dict[str, str]]
    entry_price: Optional[float]
    risk_reward: Optional[float]
    current_price: Optional[float]
    scanned_at: Optional[str]


@router.get("/candidate-near-misses", response_model=list[CandidateNearMissOut])
def list_candidate_near_misses(
    status: CandidateStatus = Query(default="new"),
    limit: int = Query(default=10, ge=1, le=50),
    x_api_key: Optional[str] = Header(default=None),
    scanner_session: Optional[str] = Cookie(default=None, alias=SCANNER_SESSION_COOKIE),
):
    """Ranked near-miss view: candidates failing exactly 1 (tier 1) or 2
    (tier 2) GRADABLE gate conditions, tier 1 first, capped at `limit`.
    Reuses the same enriched preview pipeline and the same gate functions
    as /candidate-plan-previews and the ENTER_NOW gate itself
    (_preview_base_enter_now_ready, _entry_proximity,
    _execution_shadow_from_bars) via _gate_gap_report() -- it does not
    recompute or approximate any gate condition, only re-labels the real,
    already-computed pass/fail + numeric gap as a rank instead of a binary
    filter. This is additive: it does not change what "Actionable only"
    (the strict all-gates-pass toggle) does.
    """
    _check_api_key(x_api_key, scanner_session)
    conn = _get_db()
    try:
        candidates = conn.execute(
            "SELECT * FROM candidates WHERE status=? ORDER BY updated_at DESC",
            (status,),
        ).fetchall()
        enriched, candidate_rows = _enriched_previews_for_candidates(conn, candidates)
        candidates_by_key = {
            (str(row["ticker"]).upper(), str(row["source"])): row for row in candidate_rows
        }
        ranked: list[dict[str, Any]] = []
        for preview in enriched:
            key = (str(preview.get("ticker") or "").upper(), str(preview.get("source") or ""))
            candidate = candidates_by_key.get(key)
            if candidate is None:
                continue
            report = _gate_gap_report(candidate, preview)
            if report["categorical_blocked"]:
                continue
            count = report["failing_count"]
            if not count or count > 2:
                continue
            ranked.append({
                "ticker": str(preview.get("ticker") or "").upper(),
                "source": str(preview.get("source") or ""),
                "signal": str(preview.get("signal") or ""),
                "tier": count,
                "failing_count": count,
                "gaps": report["gaps"],
                "entry_price": preview.get("entry_price"),
                "risk_reward": preview.get("risk_reward"),
                "current_price": preview.get("current_price"),
                "scanned_at": candidate["scanned_at"] if "scanned_at" in candidate.keys() else None,
            })
        ranked.sort(key=lambda item: (item["tier"], item["ticker"]))
        tier1 = [item for item in ranked if item["tier"] == 1]
        tier2 = [item for item in ranked if item["tier"] == 2]
        return (tier1 + tier2)[:limit]
    finally:
        conn.close()


@router.get("/candidate-chart-reviews", response_model=list[CandidateChartReviewOut])
def list_candidate_chart_reviews(
    x_api_key: Optional[str] = Header(default=None),
    scanner_session: Optional[str] = Cookie(default=None, alias=SCANNER_SESSION_COOKIE),
):
    _check_api_key(x_api_key, scanner_session)
    conn = _get_db()
    try:
        rows = conn.execute("SELECT * FROM candidate_ai_chart_reviews ORDER BY reviewed_at DESC").fetchall()
        return [_row_to_chart_review(row) for row in rows]
    finally:
        conn.close()


@router.post("/candidates/{ticker}/ai-chart-review", response_model=CandidateChartReviewOut)
def request_candidate_chart_review(
    ticker: str,
    source: str,
    x_api_key: Optional[str] = Header(default=None),
    scanner_session: Optional[str] = Cookie(default=None, alias=SCANNER_SESSION_COOKIE),
):
    _check_api_key(x_api_key, scanner_session)
    normalized_ticker = ticker.strip().upper()
    conn = _get_db()
    try:
        candidate = conn.execute(
            "SELECT * FROM candidates WHERE ticker=? AND source=?",
            (normalized_ticker, source),
        ).fetchone()
        if not candidate:
            raise HTTPException(status_code=404, detail=f"No candidate found for {ticker} / {source}")
        review = _compute_candidate_chart_review(candidate)
        _store_chart_review(conn, review)
        conn.commit()
        return _row_to_chart_review(review)
    finally:
        conn.close()


@router.patch("/candidates/{ticker}")
def update_candidate_status(
    ticker: str,
    source: str,
    update: StatusUpdate,
    x_api_key: Optional[str] = Header(default=None),
    scanner_session: Optional[str] = Cookie(default=None, alias=SCANNER_SESSION_COOKIE),
):
    """Change review status only; this does not open or manage a live trade."""
    _check_api_key(x_api_key, scanner_session)
    normalized_ticker = ticker.strip().upper()
    conn = _get_db()
    try:
        candidate = conn.execute(
            "SELECT * FROM candidates WHERE ticker=? AND source=?",
            (normalized_ticker, source),
        ).fetchone()
        if not candidate:
            raise HTTPException(status_code=404, detail=f"No candidate found for {ticker} / {source}")

        promotion = None
        option_contract = None
        if update.status == "active":
            promotion = _compute_candidate_promotion(candidate)
            if str(promotion.get("direction") or "").strip().lower() == "long":
                option_contract = _safe_option_contract_for_candidate(
                    promotion["ticker"],
                    promotion["direction"],
                    promotion["entry_price"],
                )
            promotion = _promotion_with_live_gate_context(candidate, promotion, option_contract)
            block_reason = _promotion_block_reason(candidate, promotion, option_contract)
            if block_reason:
                raise HTTPException(status_code=422, detail=block_reason)
            _store_promotion(conn, promotion)

        changed_at = datetime.now(timezone.utc).isoformat()
        previous_status = candidate["status"] if "status" in candidate.keys() else None
        result = conn.execute(
            "UPDATE candidates SET status=?, updated_at=? WHERE ticker=? AND source=?",
            (
                update.status,
                changed_at,
                normalized_ticker,
                source,
            ),
        )
        if result.rowcount:
            _record_status_change(
                conn,
                ticker=normalized_ticker,
                source=source,
                previous_status=previous_status,
                new_status=update.status,
                changed_at=changed_at,
                trigger="api_status_update",
            )
        conn.commit()
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail=f"No candidate found for {ticker} / {source}")
        response = {"ticker": normalized_ticker, "source": source, "status": update.status}
        if promotion:
            response["promotion"] = promotion
        if option_contract:
            response["option_contract"] = option_contract
        return response
    finally:
        conn.close()
