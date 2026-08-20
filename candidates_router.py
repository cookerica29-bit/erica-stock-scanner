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


router = APIRouter(prefix="/api/v1/scanner", tags=["scanner"])


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


@router.patch("/candidates/{ticker}")
def update_candidate_status(
    ticker: str,
    source: str,
    update: StatusUpdate,
    x_api_key: Optional[str] = Header(default=None),
):
    """Change review status only; this does not open or manage a live trade."""
    _check_api_key(x_api_key)
    conn = _get_db()
    try:
        result = conn.execute(
            "UPDATE candidates SET status=?, updated_at=? WHERE ticker=? AND source=?",
            (
                update.status,
                datetime.now(timezone.utc).isoformat(),
                ticker.strip().upper(),
                source,
            ),
        )
        conn.commit()
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail=f"No candidate found for {ticker} / {source}")
        return {"ticker": ticker.strip().upper(), "source": source, "status": update.status}
    finally:
        conn.close()
