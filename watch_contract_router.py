"""Kairos Watch Contract -- dev/test creation + listing endpoints
(2026-09 session).

Deliberately NOT the "Frozen Human Review Result -> Watch Contract" bridge
-- that integration is explicitly out of scope this sprint (Codex is
separately building the Unified Chart Review workflow on its own
branch/worktree; this file must not be edited to reach into it). This
router exists so a Watch Contract can be created directly from a
structured, already-frozen fixture -- e.g. in a test, or by hand against
a dev/staging deployment -- while that bridge is designed and reviewed
separately.

Auth reuses candidates_router._check_api_key verbatim (same convention
smc_shadow_router.py already established for a sibling feature router).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Cookie, Header, HTTPException
from pydantic import BaseModel, Field

import candidates_router as _candidates_router
import watch_contract_sensors as sensors
import watch_contract_store as store
from watch_contract_engine import initial_state

router = APIRouter(prefix="/api/v1/scanner", tags=["watch-contracts"])

# Pre-deploy hardening (2026-09 session): a valid scanner API key alone
# must NOT be enough to manufacture a Watch Contract in production --
# this route exists for tests and dev/staging fixture creation, not as a
# generally-available production capability. Independent of, and NOT a
# substitute for, KAIROS_WATCH_CONTRACT_ENABLED (that flag gates the
# periodic MONITOR tick in main.py; this one gates only this CREATE
# route). The future real "Frozen Chart Review -> Watch Contract" bridge
# will be a separate production endpoint that does NOT require this flag
# -- see this file's own module docstring.
WATCH_CONTRACT_DEV_CREATE_ENABLED_ENV_VAR = "KAIROS_WATCH_CONTRACT_DEV_CREATE_ENABLED"


def dev_create_enabled() -> bool:
    return str(os.environ.get(WATCH_CONTRACT_DEV_CREATE_ENABLED_ENV_VAR) or "").strip().lower() in ("1", "true", "yes", "on")


class WatchContractLocationIn(BaseModel):
    type: str = Field(min_length=1)
    lower: float
    upper: float
    id: Optional[str] = None
    status_at_approval: Literal["REACHED", "NOT_REACHED"]
    # Freshness-anchor semantics (2026-09 session, pre-deploy hardening
    # pass, section 6) -- exact and important:
    #
    # `reached_at` is only meaningful when status_at_approval == "REACHED",
    # and it becomes `location_reached_at` (the freshness floor
    # confirmation_30m must be strictly AFTER -- see
    # watch_contract_sensors.evaluate_30m_confirmation's own min_bar_time
    # parameter). It should be the REAL, human-observed moment price
    # touched the approved location -- i.e. Track B's own review
    # timestamp for that fact, once that bridge exists -- NOT derived from
    # market data here.
    #
    # If omitted on a REACHED contract, this endpoint falls back to "now"
    # (the moment this request is handled), NEVER to any earlier,
    # already-chart-visible timestamp. This is deliberately the SAFE
    # direction to default toward: "now" is always >= the true review
    # moment, so any 30M structural event Erica could already see on the
    # chart at review time (necessarily at or before the true review
    # moment) is guaranteed to be at or before "now" too, and is therefore
    # correctly excluded as stale -- the exact failure this section exists
    # to prevent ("do not accidentally consume a structural event that
    # was already visible on the chart when Erica approved confirmation
    # = NONE"). The tradeoff, disclosed: if a contract is created well
    # after the real review happened (e.g. a batch backfill), a genuine
    # confirmation that already fired in that gap is invisible forever
    # (its bar_time is before the "now" fallback) -- safe (never a false
    # accept), but not maximally sensitive. The real fix, once the Track B
    # bridge is built, is for that bridge to always supply the review's
    # own real `reached_at` explicitly, never rely on this fallback -- see
    # this module's own docstring for why that bridge isn't built yet.
    reached_at: Optional[str] = None


class WatchContractInvalidationIn(BaseModel):
    rule: Literal["close_above", "close_below"]
    level: float


class WatchContractDevCreateIn(BaseModel):
    ticker: str = Field(min_length=1)
    direction: Literal["long", "short"]
    source_review_id: Optional[str] = None
    approved_htf_thesis: str = Field(min_length=1)
    approved_current_leg: str = Field(min_length=1)
    location: WatchContractLocationIn
    invalidation: WatchContractInvalidationIn


@router.post("/candidates/watch-contracts/dev-create")
def dev_create_watch_contract(
    body: WatchContractDevCreateIn,
    x_api_key: Optional[str] = Header(default=None),
    scanner_session: Optional[str] = Cookie(default=None, alias=_candidates_router.SCANNER_SESSION_COOKIE),
):
    """Creates one Watch Contract from an already-frozen, structured
    review result -- ticker/direction/thesis/leg/location/invalidation
    are all human-approved input, taken exactly as supplied, never
    re-derived here. Initial state is WATCHING if the location was
    already reached at approval time, else WAITING_FOR_LOCATION -- see
    watch_contract_engine.initial_state's own docstring."""
    if not dev_create_enabled():
        # Fails closed as a plain 404 -- deliberately BEFORE the auth
        # check, so a valid API key alone reveals nothing about whether
        # this route exists in this environment. GET .../watch-contracts
        # (read) and the monitor tick are both completely unaffected --
        # this flag gates ONLY the ability to manufacture a new contract.
        raise HTTPException(status_code=404, detail="Not Found")
    _candidates_router._check_api_key(x_api_key, scanner_session)
    ticker = body.ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=422, detail="ticker must not be blank.")
    if body.location.lower >= body.location.upper:
        raise HTTPException(status_code=422, detail="location.lower must be below location.upper.")

    state = initial_state(body.location.status_at_approval)
    location_reached_at = body.location.reached_at if state == "WATCHING" else None
    if state == "WATCHING" and not location_reached_at:
        location_reached_at = datetime.now(timezone.utc).isoformat()

    conn = _candidates_router._get_db()
    try:
        record = store.create_watch_contract(
            conn,
            ticker=ticker,
            source_review_id=body.source_review_id,
            direction=body.direction,
            approved_htf_thesis=body.approved_htf_thesis,
            approved_current_leg=body.approved_current_leg,
            location_type=body.location.type,
            location_lower=body.location.lower,
            location_upper=body.location.upper,
            location_id=body.location.id,
            location_status_at_approval=body.location.status_at_approval,
            location_reached_at=location_reached_at,
            confirmation_timeframe=sensors.CONFIRMATION_TIMEFRAME,
            confirmation_direction=body.direction,
            confirmation_allowed_events=list(sensors.ALLOWED_CONFIRMATION_EVENTS),
            confirmation_min_displacement=sensors.MIN_CONFIRMATION_DISPLACEMENT,
            execution_timeframe=sensors.EXECUTION_TIMEFRAME,
            execution_direction=body.direction,
            execution_allowed_events=list(sensors.ALLOWED_EXECUTION_EVENTS),
            execution_min_displacement=sensors.MIN_EXECUTION_DISPLACEMENT,
            approved_invalidation_rule=body.invalidation.rule,
            approved_invalidation_level=body.invalidation.level,
            state=state,
        )
        return record
    finally:
        conn.close()


@router.get("/candidates/watch-contracts")
def list_watch_contracts_endpoint(
    x_api_key: Optional[str] = Header(default=None),
    scanner_session: Optional[str] = Cookie(default=None, alias=_candidates_router.SCANNER_SESSION_COOKIE),
):
    _candidates_router._check_api_key(x_api_key, scanner_session)
    conn = _candidates_router._get_db()
    try:
        return {"watch_contracts": store.list_watch_contracts(conn)}
    finally:
        conn.close()
