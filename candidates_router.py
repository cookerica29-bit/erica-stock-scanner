"""External scanner candidate ingestion endpoints.

Receives shortlist candidates from external scanners and stores them as
reviewable candidates. Promotion to active trade management remains a separate
explicit workflow.
"""

from __future__ import annotations

import os
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, Field

from market_data import AlpacaMarketDataProvider
from scanner import _batch_download, _best_contract, _compute_atr, _find_swings, _flatten_columns


router = APIRouter(prefix="/api/v1/scanner", tags=["scanner"])
ATR_MULTIPLIER = 1.5
RR_WARNING_THRESHOLD = 1.5
MIN_TARGET_ATR_MULTIPLE_DEFAULT = 2.0
ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_MODEL_DEFAULT = "claude-sonnet-4-5"
AI_CHART_REVIEW_RUBRIC_VERSION = "kairos-chart-note-v1"
CANDIDATE_PREVIEW_TRANSIENT_OPTION_REFRESH_TTL = timedelta(minutes=10)


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
    return "/data/kairos_candidates.sqlite3"


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
        return

    columns = {info["name"] for info in conn.execute("PRAGMA table_info(candidate_promotions)").fetchall()}
    if "no_valid_target" not in columns:
        conn.execute("ALTER TABLE candidate_promotions ADD COLUMN no_valid_target INTEGER NOT NULL DEFAULT 0")
    if "min_target_atr_multiple" not in columns:
        conn.execute(
            "ALTER TABLE candidate_promotions ADD COLUMN min_target_atr_multiple REAL NOT NULL DEFAULT 2.0"
        )


def _get_db():
    db_path = Path(default_candidates_db_path())
    if db_path.parent and str(db_path.parent) not in {"", "."}:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
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
            PRIMARY KEY (ticker, source)
        )
        """
    )
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
            PRIMARY KEY (ticker, source)
        )
        """
    )
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
    return conn


class CandidateIn(BaseModel):
    ticker: str
    signal: Literal["long", "short"]
    entry_price: Optional[float] = None
    ema21_4h: Optional[float] = None
    daily_regime: Optional[str] = None
    confidence: Optional[Literal["high", "medium", "low"]] = None
    sma50_daily: Optional[float] = None
    sma200_daily: Optional[float] = None


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
    option_contract: Optional[dict[str, Any]]
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


def _check_api_key(x_api_key: Optional[str]) -> None:
    api_key = _get_api_key()
    if not api_key:
        raise HTTPException(status_code=500, detail="KAIROS_SCANNER_API_KEY not configured on server")
    if x_api_key != api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _valid_ticker(ticker: str) -> bool:
    return bool(ticker) and ticker.replace(".", "").replace("-", "").isalnum()


def _row_to_promotion(row: sqlite3.Row) -> dict:
    output = dict(row)
    output["rr_warning"] = bool(output.get("rr_warning"))
    output["no_valid_target"] = bool(output.get("no_valid_target"))
    return output


def _row_to_chart_review(row: sqlite3.Row) -> dict:
    output = dict(row)
    output.pop("raw_response", None)
    return output


def _row_to_plan_preview(row: sqlite3.Row | dict) -> dict:
    output = dict(row)
    output["rr_warning"] = bool(output.get("rr_warning"))
    output["no_valid_target"] = bool(output.get("no_valid_target"))
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
    reason = str(normalized.get("reason") or "").strip().lower()
    source = str(normalized.get("source") or "").strip().lower()
    if not normalized.get("available") and (
        "no option expirations available" in reason
        or "no option expirations returned" in reason
        or source == "unavailable"
    ):
        normalized["execution"] = "Contract Data Unavailable"
        normalized["reason"] = "Option expiration data unavailable from the legacy option-chain provider; retry later."
        normalized["source"] = "data_unavailable"
        normalized["transient_unavailable"] = True
    return normalized


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

    if direction == "short":
        stop = entry_price + (ATR_MULTIPLIER * atr14)
    else:
        stop = entry_price - (ATR_MULTIPLIER * atr14)

    risk = abs(entry_price - stop)
    if risk <= 0:
        raise HTTPException(status_code=422, detail=f"Computed risk is not usable for {ticker}")

    min_target_atr_multiple = _min_target_atr_multiple()
    min_target_distance = min_target_atr_multiple * atr14
    swings = _find_swings(df)
    target = _nearest_structural_target(entry_price, direction, swings, min_target_distance)
    no_valid_target = target is None
    risk_reward = None
    rr_warning = True
    if target is not None:
        reward = abs(target - entry_price)
        risk_reward = reward / risk
        rr_warning = risk_reward < RR_WARNING_THRESHOLD

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
    }


def _store_promotion(conn, promotion: dict) -> None:
    conn.execute(
        """
        INSERT INTO candidate_promotions
            (ticker, source, direction, entry_price, stop, target, risk_reward,
             rr_warning, no_valid_target, promoted_at, position_size, atr14,
             atr_multiplier, rr_warning_threshold, min_target_atr_multiple,
             target_source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            target_source=excluded.target_source
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
             computed_at, candidate_updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            candidate_updated_at=excluded.candidate_updated_at
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
        ),
    )


@router.post("/candidates", response_model=IngestResponse)
def ingest_candidates(payload: ShortlistIn, x_api_key: Optional[str] = Header(default=None)):
    _check_api_key(x_api_key)
    return upsert_candidate_shortlist(payload)


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
                     confidence, sma50_daily, sma200_daily, status, scanned_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'new', ?, ?)
                ON CONFLICT(ticker, source) DO UPDATE SET
                    signal=excluded.signal,
                    entry_price=excluded.entry_price,
                    ema21_4h=excluded.ema21_4h,
                    daily_regime=excluded.daily_regime,
                    confidence=excluded.confidence,
                    sma50_daily=excluded.sma50_daily,
                    sma200_daily=excluded.sma200_daily,
                    scanned_at=excluded.scanned_at,
                    updated_at=excluded.updated_at
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
):
    _check_api_key(x_api_key)
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
def list_candidate_promotions(x_api_key: Optional[str] = Header(default=None)):
    _check_api_key(x_api_key)
    conn = _get_db()
    try:
        rows = conn.execute("SELECT * FROM candidate_promotions ORDER BY promoted_at DESC").fetchall()
        return [_row_to_promotion(row) for row in rows]
    finally:
        conn.close()


@router.get("/candidate-plan-previews", response_model=list[CandidatePlanPreviewOut])
def list_candidate_plan_previews(x_api_key: Optional[str] = Header(default=None)):
    _check_api_key(x_api_key)
    conn = _get_db()
    try:
        candidates = conn.execute("SELECT * FROM candidates ORDER BY updated_at DESC").fetchall()
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
            preview = _compute_candidate_plan_preview(candidate)
            _store_plan_preview(conn, preview)
            previews.append(preview)
        conn.commit()
        return previews
    finally:
        conn.close()


@router.get("/candidate-chart-reviews", response_model=list[CandidateChartReviewOut])
def list_candidate_chart_reviews(x_api_key: Optional[str] = Header(default=None)):
    _check_api_key(x_api_key)
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
):
    _check_api_key(x_api_key)
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
):
    """Change review status only; this does not open or manage a live trade."""
    _check_api_key(x_api_key)
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
        if update.status == "active":
            promotion = _compute_candidate_promotion(candidate)
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
        return response
    finally:
        conn.close()
