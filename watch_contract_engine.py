"""Kairos Watch Contract -- live lifecycle monitor (2026-09 session).

Reads Erica-approved Watch Contracts (watch_contract_store.py) and advances
each one's state using ONLY the deterministic sensors in
watch_contract_sensors.py -- this module never infers HTF thesis, current
leg, or location; every one of those is a frozen field the contract was
CREATED with (see watch_contract_store.create_watch_contract's own
docstring). This is a lifecycle monitor, the same role
candidates_router.run_approved_setup_monitor_tick already plays for the
price-trigger Approved/Watch object -- not a second strategy engine.

State-specific data cost (per this sprint's explicit instruction -- never
fetch 4H/30M/5M for every row regardless of state):
  WAITING_FOR_LOCATION -> one quote per ticker, no bar fetch at all
  WATCHING             -> 30M bars only
  WAITING_FOR_PULLBACK -> 30M bars only (same timeframe the pullback
                          primitive itself operates on)
  PULLBACK_REACHED     -> 5M bars only
  ENTRY_READY / INVALIDATED / NEEDS_REVIEW -> excluded from the active
                          query entirely; no fetch, no further scanning.
Per-tick caches (quote/30M/5M, keyed by ticker) mean two contracts sharing
a ticker+state never trigger two provider calls in the same tick.

Deliberately does NOT import hybrid_shadow_engine.py at all -- see
watch_contract_sensors.py's own module docstring for why (that module is
a separate, feature-flagged RESEARCH engine that also autonomously infers
thesis/location; this monitor's whole job is the opposite of that).
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

import scanner
import watch_contract_sensors as sensors
import watch_contract_store as store
from market_data import AlpacaMarketDataProvider

logger = logging.getLogger(__name__)

MONITOR_TICK_SECONDS = 300  # same 5-minute cadence run_approved_setup_monitor_tick uses

# Feature flag -- OFF unless explicitly set, same convention
# smc_shadow_engine.smc_shadow_enabled()/hybrid_shadow_engine.hybrid_shadow_enabled()
# already use. Gates ONLY the periodic-tick registration in main.py; the
# router (dev-create/list) is always mounted and auth-gated the same way
# every other endpoint in this app already is, matching the smc_shadow
# precedent of "inert until explicitly enabled, never a second auth model."
WATCH_CONTRACT_ENABLED_ENV_VAR = "KAIROS_WATCH_CONTRACT_ENABLED"


def watch_contract_enabled() -> bool:
    return str(os.environ.get(WATCH_CONTRACT_ENABLED_ENV_VAR) or "").strip().lower() in ("1", "true", "yes", "on")

# Provider fetch windows -- generous enough that a confirmation/pullback
# event several hours old is still inside the freshly fetched window (the
# pullback scan below needs to re-locate the ORIGINAL confirmation bar in
# each new fetch), without pulling more history than any sensor here
# actually looks at.
BARS_30M_PERIOD = "5d"
BARS_5M_PERIOD = "5d"


def initial_state(location_status_at_approval: Optional[str]) -> str:
    """WATCHING if the approved review already recorded the location as
    reached; WAITING_FOR_LOCATION otherwise. Called by the creation
    endpoint/service BEFORE the row is written -- this module makes the
    one-time initial-state decision, but never re-derives it later."""
    return "WATCHING" if str(location_status_at_approval or "").strip().upper() == "REACHED" else "WAITING_FOR_LOCATION"


def _now_iso(now: datetime) -> str:
    return now.astimezone(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Narrow market-data fetch -- deliberately NOT imported from
# candidates_router.py (that module imports watch_contract_store.py for
# the dashboard read model; importing back would be circular). Same
# provider call, same cleanup (scanner._flatten_columns), as
# candidates_router._fetch_recent_30m_bars/_recent_4h_bars_for_execution_shadow
# use for their own, differently-shaped, purposes -- but returns the RAW
# DataFrame (Open/High/Low/Close, DatetimeIndex) the sensors need, not
# their lowercase-dict row shape.
# ---------------------------------------------------------------------------

def _fetch_quote(ticker: str) -> Optional[float]:
    try:
        quotes = AlpacaMarketDataProvider().latest_quotes([ticker])
    except Exception as exc:
        logger.warning("[watch_contract] quote fetch failed ticker=%s error=%s", ticker, exc)
        return None
    quote = (quotes or {}).get(ticker) if isinstance(quotes, dict) else None
    if not quote:
        return None
    try:
        price = float(quote.get("price"))
    except (TypeError, ValueError):
        return None
    return price if price > 0 else None


def _fetch_bars_df(ticker: str, interval: str, period: str, interval_minutes: int, now: datetime) -> Optional[pd.DataFrame]:
    try:
        raw = AlpacaMarketDataProvider().download([ticker], period=period, interval=interval, auto_adjust=True)
    except Exception as exc:
        logger.warning("[watch_contract] %s bar download failed ticker=%s error=%s", interval, ticker, exc)
        return None
    if raw is None or raw.empty:
        return None
    try:
        df = scanner._flatten_columns(raw.copy()).dropna().astype(float)
    except Exception:
        return None
    required = {"Open", "High", "Low", "Close"}
    if df.empty or not required.issubset(set(df.columns)):
        return None
    return sensors.drop_forming_bar(df, interval_minutes, now)


class _TickCaches:
    """Per-tick memoization -- two active contracts on the same ticker in
    the same state share one provider call (section 13's "do not create
    one provider request per row per timeframe blindly")."""

    def __init__(self, now: datetime):
        self.now = now
        self._quotes: dict[str, Optional[float]] = {}
        self._bars_30m: dict[str, Optional[pd.DataFrame]] = {}
        self._bars_5m: dict[str, Optional[pd.DataFrame]] = {}

    def quote(self, ticker: str) -> Optional[float]:
        if ticker not in self._quotes:
            self._quotes[ticker] = _fetch_quote(ticker)
        return self._quotes[ticker]

    def bars_30m(self, ticker: str) -> Optional[pd.DataFrame]:
        if ticker not in self._bars_30m:
            self._bars_30m[ticker] = _fetch_bars_df(ticker, "30m", BARS_30M_PERIOD, 30, self.now)
        return self._bars_30m[ticker]

    def bars_5m(self, ticker: str) -> Optional[pd.DataFrame]:
        if ticker not in self._bars_5m:
            self._bars_5m[ticker] = _fetch_bars_df(ticker, "5m", BARS_5M_PERIOD, 5, self.now)
        return self._bars_5m[ticker]


def _locate_bar_index(df: pd.DataFrame, bar_time_iso: Optional[str]) -> Optional[int]:
    if not bar_time_iso:
        return None
    try:
        target = pd.Timestamp(bar_time_iso)
    except (TypeError, ValueError):
        return None
    if target.tzinfo is None:
        target = target.tz_localize("UTC")
    for i, idx in enumerate(df.index):
        candidate = idx if idx.tzinfo is not None else idx.tz_localize("UTC")
        if candidate == target:
            return i
    return None


def _check_invalidation(contract: dict[str, Any], current_price: Optional[float]) -> bool:
    return sensors.is_invalidated(
        contract["direction"], current_price,
        contract.get("approved_invalidation_rule"), contract.get("approved_invalidation_level"),
    )


def _invalidate(conn: sqlite3.Connection, contract: dict[str, Any], as_of_iso: str, current_price: Optional[float]) -> None:
    rule = contract.get("approved_invalidation_rule")
    level = contract.get("approved_invalidation_level")
    verb = "below" if rule == "close_below" else "above"
    store.update_watch_contract(
        conn, contract["watch_contract_id"],
        state="INVALIDATED", invalidated_at=as_of_iso,
        invalidation_reason=f"Price ({current_price}) closed {verb} the approved invalidation level ({level}).",
        last_checked_at=as_of_iso, last_live_price=current_price,
    )


def _process_waiting_for_location(conn: sqlite3.Connection, contract: dict[str, Any], caches: _TickCaches, now_iso: str) -> None:
    ticker = contract["ticker"]
    price = caches.quote(ticker)
    if price is None:
        return  # no fresh price this tick -- leave untouched, never guess
    if _check_invalidation(contract, price):
        _invalidate(conn, contract, now_iso, price)
        return
    lower, upper = contract.get("location_lower"), contract.get("location_upper")
    if lower is None or upper is None:
        store.update_watch_contract(conn, contract["watch_contract_id"], last_checked_at=now_iso, last_live_price=price)
        return
    if lower <= price <= upper:
        store.update_watch_contract(
            conn, contract["watch_contract_id"],
            location_reached_at=now_iso, state="WATCHING", last_checked_at=now_iso, last_live_price=price,
        )
    else:
        store.update_watch_contract(conn, contract["watch_contract_id"], last_checked_at=now_iso, last_live_price=price)


def _process_watching(conn: sqlite3.Connection, contract: dict[str, Any], caches: _TickCaches, now_iso: str) -> None:
    ticker = contract["ticker"]
    df_30m = caches.bars_30m(ticker)
    if df_30m is None or len(df_30m) == 0:
        return
    current_price = float(df_30m["Close"].iloc[-1])
    if _check_invalidation(contract, current_price):
        _invalidate(conn, contract, now_iso, current_price)
        return
    result = sensors.evaluate_30m_confirmation(df_30m, contract["direction"], contract.get("location_reached_at"))
    if not result["confirmed"]:
        store.update_watch_contract(conn, contract["watch_contract_id"], last_checked_at=now_iso, last_live_price=current_price)
        return

    confirming_index = _locate_bar_index(df_30m, result["bar_time"])
    location = {
        "level_bounds": (
            {"high": contract["location_upper"], "low": contract["location_lower"]}
            if contract.get("location_lower") is not None and contract.get("location_upper") is not None
            else None
        ),
        "level_id": contract.get("location_id"),
    }
    unresolved: list[str] = []
    reference = None
    if confirming_index is not None:
        reference = sensors.resolve_pullback_reference(
            df_30m, None, confirming_index, location, sensors._direction_lower(contract["direction"]), unresolved,
        )
    store.update_watch_contract(
        conn, contract["watch_contract_id"],
        confirmation_event_type=result["event_type"], confirmation_level=result["level"],
        confirmation_bar_time=result["bar_time"],
        confirmation_prior_touch_count=result["prior_touch_count"],
        confirmation_bars_since_first_pierce=result["bars_since_first_pierce"],
        pullback_reference_low=reference["low"] if reference else None,
        pullback_reference_high=reference["high"] if reference else None,
        state="WAITING_FOR_PULLBACK", last_checked_at=now_iso, last_live_price=current_price,
    )


def _process_waiting_for_pullback(conn: sqlite3.Connection, contract: dict[str, Any], caches: _TickCaches, now_iso: str) -> None:
    ticker = contract["ticker"]
    df_30m = caches.bars_30m(ticker)
    if df_30m is None or len(df_30m) == 0:
        return
    current_price = float(df_30m["Close"].iloc[-1])
    if _check_invalidation(contract, current_price):
        _invalidate(conn, contract, now_iso, current_price)
        return
    if contract.get("pullback_reference_low") is None or contract.get("pullback_reference_high") is None:
        store.update_watch_contract(conn, contract["watch_contract_id"], last_checked_at=now_iso, last_live_price=current_price)
        return
    confirming_index = _locate_bar_index(df_30m, contract.get("confirmation_bar_time"))
    if confirming_index is None:
        store.update_watch_contract(conn, contract["watch_contract_id"], last_checked_at=now_iso, last_live_price=current_price)
        return
    found, clear_index, return_index = sensors.windowed_clear_and_return(
        df_30m, confirming_index, contract["pullback_reference_low"], contract["pullback_reference_high"],
        sensors._direction_upper(contract["direction"]),
    )
    if not found:
        store.update_watch_contract(conn, contract["watch_contract_id"], last_checked_at=now_iso, last_live_price=current_price)
        return
    pullback_cleared_at = scanner._timestamp_at(df_30m, clear_index)
    pullback_reached_at = scanner._timestamp_at(df_30m, return_index)
    store.update_watch_contract(
        conn, contract["watch_contract_id"],
        pullback_cleared_at=pullback_cleared_at, pullback_reached_at=pullback_reached_at,
        state="PULLBACK_REACHED", last_checked_at=now_iso, last_live_price=current_price,
    )


def _process_pullback_reached(conn: sqlite3.Connection, contract: dict[str, Any], caches: _TickCaches, now_iso: str) -> None:
    ticker = contract["ticker"]
    df_5m = caches.bars_5m(ticker)
    if df_5m is None or len(df_5m) == 0:
        return
    current_price = float(df_5m["Close"].iloc[-1])
    if _check_invalidation(contract, current_price):
        _invalidate(conn, contract, now_iso, current_price)
        return

    min_bar_time = contract.get("pullback_reached_at")
    execution = sensors.evaluate_5m_execution(df_5m, contract["direction"], min_bar_time)
    # Rejection diagnostic (2026-09 session, v1 scope): ALWAYS computed and
    # persisted when present, but never gates ENTRY_READY on its own --
    # see evaluate_5m_rejection_diagnostic's own docstring.
    rejection = sensors.evaluate_5m_rejection_diagnostic(df_5m, contract["direction"], min_bar_time)

    updates: dict[str, Any] = {"last_checked_at": now_iso, "last_live_price": current_price}
    if rejection["detected"]:
        updates.update({
            "execution_rejection_event_type": rejection["event_type"],
            "execution_rejection_level": rejection["level"],
            "execution_rejection_bar_time": rejection["bar_time"],
        })
    if execution["confirmed"]:
        updates.update({
            "execution_event_type": execution["event_type"], "execution_level": execution["level"],
            "execution_bar_time": execution["bar_time"],
            "execution_prior_touch_count": execution["prior_touch_count"],
            "execution_bars_since_first_pierce": execution["bars_since_first_pierce"],
            "state": "ENTRY_READY",
        })
    store.update_watch_contract(conn, contract["watch_contract_id"], **updates)


_STATE_HANDLERS = {
    "WAITING_FOR_LOCATION": _process_waiting_for_location,
    "WATCHING": _process_watching,
    "WAITING_FOR_PULLBACK": _process_waiting_for_pullback,
    "PULLBACK_REACHED": _process_pullback_reached,
}

_ACTIVE_STATES = tuple(_STATE_HANDLERS.keys())


def run_watch_contract_monitor_tick(conn: sqlite3.Connection, reason: str = "periodic", now: Optional[datetime] = None) -> dict[str, Any]:
    """One tick, over every NON-terminal Watch Contract. Terminal states
    (ENTRY_READY/INVALIDATED/NEEDS_REVIEW) are excluded from the query
    itself -- no fetch, no re-evaluation, ever, for a row already
    resolved (structurally guarantees "no duplicate transition": a
    contract that already advanced past a stage is never handed to that
    stage's handler again, regardless of how many times this tick runs).
    """
    now = now or datetime.now(timezone.utc)
    contracts = store.list_watch_contracts(conn, states=_ACTIVE_STATES)
    caches = _TickCaches(now)
    now_iso = _now_iso(now)
    checked = 0
    for contract in contracts:
        handler = _STATE_HANDLERS.get(contract["state"])
        if handler is None:
            continue
        checked += 1
        handler(conn, contract, caches, now_iso)
    return {"checked": checked, "reason": reason}


def run_watch_contract_monitor_tick_standalone(reason: str = "periodic") -> dict[str, Any]:
    """Zero-arg entrypoint for periodic-task registration (main.py) --
    opens its own connection via candidates_router's existing
    default_candidates_db_path()/schema-init, imported lazily HERE (not at
    module load) specifically to avoid a circular import: candidates_router.py
    imports watch_contract_store.py at its own module load time (for the
    dashboard read model), so this module cannot import candidates_router.py
    at ITS module load time -- only safely, lazily, inside a function that
    runs after both modules have already finished loading."""
    from candidates_router import _get_db  # local import -- see docstring above

    conn = _get_db()
    try:
        return run_watch_contract_monitor_tick(conn, reason=reason)
    finally:
        conn.close()
