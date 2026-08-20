"""External scanner candidate ingestion endpoints.

Receives shortlist candidates from external scanners and stores them as
reviewable candidates. Promotion to active trade management remains a separate
explicit workflow.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, Field

from scanner import _batch_download, _compute_atr, _find_swings, _flatten_columns


router = APIRouter(prefix="/api/v1/scanner", tags=["scanner"])
ATR_MULTIPLIER = 1.5
RR_WARNING_THRESHOLD = 1.5
MIN_TARGET_ATR_MULTIPLE_DEFAULT = 2.0


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


@router.post("/candidates", response_model=IngestResponse)
def ingest_candidates(payload: ShortlistIn, x_api_key: Optional[str] = Header(default=None)):
    _check_api_key(x_api_key)

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

        result = conn.execute(
            "UPDATE candidates SET status=?, updated_at=? WHERE ticker=? AND source=?",
            (
                update.status,
                datetime.now(timezone.utc).isoformat(),
                normalized_ticker,
                source,
            ),
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
