"""Alpaca-backed ticker discovery primitives.

This module is intentionally separate from scanner.py. It builds candidate
universes for later review; it does not feed the live scanner path directly.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from market_data import AlpacaMarketDataProvider


DEFAULT_ALPACA_TRADING_BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_ASSETS_ENDPOINT = "/v2/assets"
ALPACA_OPTION_CONTRACTS_ENDPOINT = "/v2/options/contracts"
SUPPORTED_US_EQUITY_EXCHANGES = {
    "AMEX",
    "ARCA",
    "BATS",
    "NASDAQ",
    "NYSE",
    "NYSEARCA",
}
OPTIONS_ENABLED_ATTRIBUTES = {"has_options", "options_enabled"}
DISCOVERY_DOLLAR_VOLUME_LOOKBACK_BARS = 30
DISCOVERY_MIN_VALID_DAILY_BARS = 25
DISCOVERY_MIN_AVG_DOLLAR_VOLUME = 100_000_000
DISCOVERY_DAILY_BARS_PERIOD = "60d"
DISCOVERY_DAILY_BARS_BATCH_SIZE = 1000
DISCOVERY_OPTIONS_MIN_DTE = 14
DISCOVERY_OPTIONS_MAX_DTE = 60
DISCOVERY_OPTIONS_STRIKE_BAND_PCT = 0.10
DISCOVERY_OPTIONS_MIN_OPEN_INTEREST = 100
DISCOVERY_OPTIONS_CONTRACT_LIMIT = 10000
DISCOVERY_OPTIONS_MAX_PAGES = 10
DISCOVERY_OPTIONS_MAX_WORKERS = 2
DISCOVERY_OPTIONS_MAX_ATTEMPTS = 4
DISCOVERY_OPTIONS_RETRY_BACKOFF_SECONDS = 1.5
DISCOVERY_DEFAULT_UNIVERSE_MAX_SYMBOLS = 550
DISCOVERY_UNIVERSE_MAX_SYMBOLS_ENV = "DISCOVERY_UNIVERSE_MAX_SYMBOLS"
DISCOVERY_RANK_DOLLAR_VOLUME_WEIGHT = 0.50
DISCOVERY_RANK_CALL_OI_WEIGHT = 0.25
DISCOVERY_RANK_PUT_OI_WEIGHT = 0.25


def discovery_universe_max_symbols(value: Optional[str] = None) -> int:
    raw = os.getenv(DISCOVERY_UNIVERSE_MAX_SYMBOLS_ENV) if value is None else value
    if raw is None or str(raw).strip() == "":
        return DISCOVERY_DEFAULT_UNIVERSE_MAX_SYMBOLS
    try:
        parsed = int(str(raw).strip())
    except (TypeError, ValueError):
        return DISCOVERY_DEFAULT_UNIVERSE_MAX_SYMBOLS
    return parsed if parsed > 0 else DISCOVERY_DEFAULT_UNIVERSE_MAX_SYMBOLS

WARRANT_NAME_RE = re.compile(r"\bwarrants?\b|\bwt\b", re.IGNORECASE)
UNIT_NAME_RE = re.compile(r"\bunits?\b", re.IGNORECASE)
PREFERRED_NAME_RE = re.compile(
    r"\bpreferred\s+(stock|shares?|securities)\b"
    r"|\b(pfd|preference shares?)\b"
    r"|\bdepositary shares\b.*\bpreferred\b"
    r"|\bpreferred\b.*\bdepositary shares\b",
    re.IGNORECASE,
)
RIGHT_NAME_RE = re.compile(r"\bcontingent value rights?\b|\brights?\b(?!\s+to\s+receive)|\brt pur\b", re.IGNORECASE)
SPAC_UNIT_NAME_RE = re.compile(
    r"\b(acquisition|acquisitions|spac|blank check)\b.*\bunits?\b"
    r"|\bunits?\b.*\b(acquisition|acquisitions|spac|blank check)\b",
    re.IGNORECASE,
)
LEVERAGED_ETF_DIRECT_RE = re.compile(
    r"\b(2x|3x|1\.5x|1x\s+short|leveraged|inverse|ultrapro|bear|bull)\b",
    re.IGNORECASE,
)
LEVERAGED_ETF_PROSHARES_RE = re.compile(
    r"\bproshares\b.*\b(ultrashort|ultrapro|ultra|short)\b",
    re.IGNORECASE,
)
DERIVATIVE_SYMBOL_RE = re.compile(r"([.\-/](WS?|WT|U|UN|R|RT|P|PR|PF|PRA|PRB|PRC|PRD|PRE|PRF|PRG|PRH|PRI|PRJ|PRK|PRL|PRM|PRN|PRO|PRP|PRQ|PRR|PRS|PRT|PRU|PRV|PRW|PRX|PRY|PRZ))$")


@dataclass(frozen=True)
class AssetFilterCounts:
    raw_assets: int
    active_tradable_non_otc_optionable: int
    hygiene_passed: int


@dataclass(frozen=True)
class DollarVolumeMetrics:
    symbol: str
    latest_close: Optional[float]
    average_daily_volume: Optional[float]
    average_daily_dollar_volume: Optional[float]
    valid_daily_bars: int
    passed: bool
    rejection_reason: Optional[str] = None


@dataclass(frozen=True)
class OptionsLiquidityMetrics:
    symbol: str
    latest_close: Optional[float]
    near_atm_call_open_interest: Optional[int]
    near_atm_put_open_interest: Optional[int]
    near_atm_call_contract: Optional[str]
    near_atm_put_contract: Optional[str]
    near_atm_contracts_checked: int
    pages_fetched: int
    passed: bool
    rejection_reason: Optional[str] = None


@dataclass(frozen=True)
class RankedDiscoveryCandidate:
    symbol: str
    latest_close: Optional[float]
    average_daily_dollar_volume: Optional[float]
    near_atm_call_open_interest: Optional[int]
    near_atm_put_open_interest: Optional[int]
    dollar_volume_percentile: float
    call_open_interest_percentile: float
    put_open_interest_percentile: float
    combined_liquidity_score: float
    rank: int
    selected: bool


class AlpacaAssetDiscoveryClient:
    """Small Trading API client for Alpaca asset discovery."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY", "")
        self.base_url = (base_url or os.getenv("ALPACA_TRADING_BASE_URL") or DEFAULT_ALPACA_TRADING_BASE_URL).rstrip("/")
        self.timeout = int(timeout or os.getenv("ALPACA_TRADING_TIMEOUT", "30"))

    def _headers(self) -> dict[str, str]:
        if not self.api_key or not self.secret_key:
            raise RuntimeError("Alpaca credentials are not configured")
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }

    def fetch_assets(self, *, status: str = "active", asset_class: str = "us_equity") -> list[dict[str, Any]]:
        params = urlencode({"status": status, "asset_class": asset_class})
        url = f"{self.base_url}{ALPACA_ASSETS_ENDPOINT}?{params}"
        request = Request(url, headers=self._headers())
        with urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, list):
            raise RuntimeError("Alpaca assets response was not a list")
        return payload

    def _request_option_contracts_page(self, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{ALPACA_OPTION_CONTRACTS_ENDPOINT}?{urlencode(params)}"
        request = Request(url, headers=self._headers())
        with urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def fetch_option_contracts_pages(
        self,
        params: dict[str, Any],
        *,
        max_pages: int = DISCOVERY_OPTIONS_MAX_PAGES,
    ) -> tuple[list[dict[str, Any]], int]:
        contracts: list[dict[str, Any]] = []
        page_token = None
        seen_tokens = set()
        page_count = 0

        while True:
            if page_count >= max(int(max_pages or 1), 1):
                raise RuntimeError("Alpaca option-contract pagination exceeded max pages")

            page_params = dict(params)
            if page_token:
                if page_token in seen_tokens:
                    raise RuntimeError("Alpaca option-contract pagination repeated page token")
                seen_tokens.add(page_token)
                page_params["page_token"] = page_token

            payload = self._request_option_contracts_page(page_params)
            page_count += 1
            page_contracts = payload.get("option_contracts")
            if not isinstance(page_contracts, list):
                raise RuntimeError("Alpaca option-contract response missing option_contracts list")
            contracts.extend(page_contracts)

            page_token = payload.get("next_page_token")
            if not page_token:
                return contracts, page_count


def asset_symbol(asset: dict[str, Any]) -> str:
    return str(asset.get("symbol") or "").strip().upper()


def asset_name(asset: dict[str, Any]) -> str:
    return str(asset.get("name") or "").strip()


def asset_class_name(asset: dict[str, Any]) -> str:
    return str(asset.get("class") or asset.get("asset_class") or "").strip().lower()


def asset_attributes(asset: dict[str, Any]) -> set[str]:
    attributes = asset.get("attributes") or []
    if not isinstance(attributes, list):
        return set()
    return {str(item).strip().lower() for item in attributes if str(item).strip()}


def asset_has_options(asset: dict[str, Any]) -> bool:
    if bool(asset.get("has_options") or asset.get("options_enabled")):
        return True
    return bool(asset_attributes(asset) & OPTIONS_ENABLED_ATTRIBUTES)


def is_active_tradable_non_otc_optionable_us_equity(asset: dict[str, Any]) -> bool:
    exchange = str(asset.get("exchange") or "").strip().upper()
    return (
        str(asset.get("status") or "").strip().lower() == "active"
        and asset_class_name(asset) == "us_equity"
        and bool(asset.get("tradable")) is True
        and exchange in SUPPORTED_US_EQUITY_EXCHANGES
        and asset_has_options(asset)
    )


def symbol_hygiene_rejection_reason(asset: dict[str, Any]) -> Optional[str]:
    symbol = asset_symbol(asset)
    name = asset_name(asset)

    if not symbol:
        return "missing symbol"
    if DERIVATIVE_SYMBOL_RE.search(symbol):
        return "derivative symbol suffix"
    if WARRANT_NAME_RE.search(name):
        return "warrant"
    if RIGHT_NAME_RE.search(name):
        return "right"
    if PREFERRED_NAME_RE.search(name):
        return "preferred"
    if SPAC_UNIT_NAME_RE.search(name):
        return "spac unit"
    if is_leveraged_or_inverse_fund_name(name):
        return "leveraged/inverse etf"
    return None


def is_leveraged_or_inverse_fund_name(name: str) -> bool:
    lower_name = str(name or "").lower()
    if "short duration" in lower_name:
        return False
    if LEVERAGED_ETF_PROSHARES_RE.search(lower_name):
        return True
    is_fund = bool(re.search(r"\b(etf|etn|fund|trust|shares)\b", lower_name))
    if not is_fund:
        return False
    if LEVERAGED_ETF_DIRECT_RE.search(lower_name):
        return True
    return False


def passes_symbol_hygiene(asset: dict[str, Any]) -> bool:
    return symbol_hygiene_rejection_reason(asset) is None


def stage1_optionable_assets(assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [asset for asset in assets if is_active_tradable_non_otc_optionable_us_equity(asset)]


def stage2_hygiene_assets(assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [asset for asset in assets if passes_symbol_hygiene(asset)]


def discovery_stage_counts(assets: list[dict[str, Any]]) -> AssetFilterCounts:
    stage1 = stage1_optionable_assets(assets)
    stage2 = stage2_hygiene_assets(stage1)
    return AssetFilterCounts(
        raw_assets=len(assets),
        active_tradable_non_otc_optionable=len(stage1),
        hygiene_passed=len(stage2),
    )


def _symbol_batches(symbols: list[str], batch_size: int = DISCOVERY_DAILY_BARS_BATCH_SIZE):
    size = max(int(batch_size or 1), 1)
    for index in range(0, len(symbols), size):
        yield symbols[index : index + size]


def _frame_for_symbol(downloaded: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if downloaded is None or downloaded.empty:
        return pd.DataFrame()
    if isinstance(downloaded.columns, pd.MultiIndex):
        top_level = downloaded.columns.get_level_values(0)
        if symbol in top_level:
            return downloaded[symbol].dropna(how="all")
        return pd.DataFrame()
    return downloaded.dropna(how="all")


def fetch_discovery_daily_bars(
    symbols: list[str],
    *,
    provider: Optional[AlpacaMarketDataProvider] = None,
    batch_size: int = DISCOVERY_DAILY_BARS_BATCH_SIZE,
    period: str = DISCOVERY_DAILY_BARS_PERIOD,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Fetch daily bars for discovery candidates.

    This uses dollar volume only. There is intentionally no flat price floor in
    the discovery pipeline.
    """
    data_provider = provider or AlpacaMarketDataProvider()
    frames: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    unique_symbols = list(dict.fromkeys(str(symbol or "").strip().upper() for symbol in symbols if str(symbol or "").strip()))

    for batch in _symbol_batches(unique_symbols, batch_size):
        try:
            downloaded = data_provider.download(batch, period=period, interval="1d", progress=False, auto_adjust=True, group_by="ticker")
        except Exception:
            for symbol in batch:
                failures[symbol] = "bar fetch failed"
            continue

        for symbol in batch:
            frame = _frame_for_symbol(downloaded, symbol)
            if frame.empty:
                failures[symbol] = "no price data"
            else:
                frames[symbol] = frame
    return frames, failures


def dollar_volume_metrics_for_frame(
    symbol: str,
    df: Optional[pd.DataFrame],
    *,
    lookback_bars: int = DISCOVERY_DOLLAR_VOLUME_LOOKBACK_BARS,
    min_valid_bars: int = DISCOVERY_MIN_VALID_DAILY_BARS,
    min_avg_dollar_volume: float = DISCOVERY_MIN_AVG_DOLLAR_VOLUME,
) -> DollarVolumeMetrics:
    symbol = str(symbol or "").strip().upper()
    if df is None or df.empty or "Close" not in df.columns or "Volume" not in df.columns:
        return DollarVolumeMetrics(symbol, None, None, None, 0, False, "no price data")

    close = pd.to_numeric(df["Close"], errors="coerce")
    volume = pd.to_numeric(df["Volume"], errors="coerce")
    valid = pd.DataFrame({"close": close, "volume": volume}).dropna()
    valid = valid[(valid["close"] > 0) & (valid["volume"] >= 0)]
    window = valid.tail(max(int(lookback_bars or 1), 1))
    valid_daily_bars = int(len(window))
    if window.empty:
        return DollarVolumeMetrics(symbol, None, None, None, 0, False, "no valid daily bars")

    latest_close = float(window["close"].iloc[-1])
    average_daily_volume = float(window["volume"].mean())
    average_daily_dollar_volume = float((window["close"] * window["volume"]).mean())
    if valid_daily_bars < int(min_valid_bars):
        return DollarVolumeMetrics(
            symbol,
            latest_close,
            average_daily_volume,
            average_daily_dollar_volume,
            valid_daily_bars,
            False,
            "insufficient valid daily bars",
        )
    if average_daily_dollar_volume < float(min_avg_dollar_volume):
        return DollarVolumeMetrics(
            symbol,
            latest_close,
            average_daily_volume,
            average_daily_dollar_volume,
            valid_daily_bars,
            False,
            "low dollar volume",
        )
    return DollarVolumeMetrics(
        symbol,
        latest_close,
        average_daily_volume,
        average_daily_dollar_volume,
        valid_daily_bars,
        True,
    )


def apply_dollar_volume_filter(
    symbols: list[str],
    daily_bars: dict[str, pd.DataFrame],
    fetch_failures: Optional[dict[str, str]] = None,
    *,
    lookback_bars: int = DISCOVERY_DOLLAR_VOLUME_LOOKBACK_BARS,
    min_valid_bars: int = DISCOVERY_MIN_VALID_DAILY_BARS,
    min_avg_dollar_volume: float = DISCOVERY_MIN_AVG_DOLLAR_VOLUME,
) -> list[DollarVolumeMetrics]:
    failures = fetch_failures or {}
    metrics = []
    for symbol in list(dict.fromkeys(str(item or "").strip().upper() for item in symbols if str(item or "").strip())):
        if symbol in failures and symbol not in daily_bars:
            metrics.append(DollarVolumeMetrics(symbol, None, None, None, 0, False, failures[symbol]))
            continue
        metrics.append(dollar_volume_metrics_for_frame(
            symbol,
            daily_bars.get(symbol),
            lookback_bars=lookback_bars,
            min_valid_bars=min_valid_bars,
            min_avg_dollar_volume=min_avg_dollar_volume,
        ))
    return metrics


def stage3_dollar_volume_filter(
    assets: list[dict[str, Any]],
    *,
    provider: Optional[AlpacaMarketDataProvider] = None,
    batch_size: int = DISCOVERY_DAILY_BARS_BATCH_SIZE,
) -> tuple[list[DollarVolumeMetrics], dict[str, str]]:
    symbols = [asset_symbol(asset) for asset in assets if asset_symbol(asset)]
    daily_bars, failures = fetch_discovery_daily_bars(symbols, provider=provider, batch_size=batch_size)
    return apply_dollar_volume_filter(symbols, daily_bars, failures), failures


def _safe_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not pd.notna(parsed):
        return None
    return parsed


def _safe_int(value: Any) -> Optional[int]:
    parsed = _safe_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _options_expiration_window(today: Optional[date] = None) -> tuple[str, str]:
    base = today or date.today()
    return (
        (base + timedelta(days=DISCOVERY_OPTIONS_MIN_DTE)).isoformat(),
        (base + timedelta(days=DISCOVERY_OPTIONS_MAX_DTE)).isoformat(),
    )


def option_contract_request_params(symbol: str, latest_close: float, *, today: Optional[date] = None) -> dict[str, Any]:
    min_expiry, max_expiry = _options_expiration_window(today)
    lower_strike = float(latest_close) * (1 - DISCOVERY_OPTIONS_STRIKE_BAND_PCT)
    upper_strike = float(latest_close) * (1 + DISCOVERY_OPTIONS_STRIKE_BAND_PCT)
    return {
        "underlying_symbols": str(symbol or "").strip().upper(),
        "status": "active",
        "expiration_date_gte": min_expiry,
        "expiration_date_lte": max_expiry,
        "strike_price_gte": f"{lower_strike:.4f}",
        "strike_price_lte": f"{upper_strike:.4f}",
        "limit": DISCOVERY_OPTIONS_CONTRACT_LIMIT,
    }


def options_liquidity_metrics_from_contracts(
    symbol: str,
    latest_close: Optional[float],
    contracts: list[dict[str, Any]],
    *,
    pages_fetched: int = 0,
    min_open_interest: int = DISCOVERY_OPTIONS_MIN_OPEN_INTEREST,
) -> OptionsLiquidityMetrics:
    symbol = str(symbol or "").strip().upper()
    if latest_close is None:
        return OptionsLiquidityMetrics(symbol, None, None, None, None, None, 0, pages_fetched, False, "missing current price")

    lower_strike = float(latest_close) * (1 - DISCOVERY_OPTIONS_STRIKE_BAND_PCT)
    upper_strike = float(latest_close) * (1 + DISCOVERY_OPTIONS_STRIKE_BAND_PCT)
    best_call = None
    best_put = None
    checked = 0

    for contract in contracts:
        if str(contract.get("underlying_symbol") or "").strip().upper() != symbol:
            continue
        if str(contract.get("status") or "").strip().lower() != "active":
            continue
        if contract.get("tradable") is not True:
            continue
        strike = _safe_float(contract.get("strike_price"))
        open_interest = _safe_int(contract.get("open_interest"))
        if strike is None or open_interest is None:
            continue
        if strike < lower_strike or strike > upper_strike:
            continue
        checked += 1
        contract_type = str(contract.get("type") or "").strip().lower()
        candidate = {
            "symbol": contract.get("symbol"),
            "open_interest": open_interest,
            "strike_distance": abs(strike - float(latest_close)),
        }
        if contract_type == "call" and (
            best_call is None
            or candidate["open_interest"] > best_call["open_interest"]
            or (
                candidate["open_interest"] == best_call["open_interest"]
                and candidate["strike_distance"] < best_call["strike_distance"]
            )
        ):
            best_call = candidate
        elif contract_type == "put" and (
            best_put is None
            or candidate["open_interest"] > best_put["open_interest"]
            or (
                candidate["open_interest"] == best_put["open_interest"]
                and candidate["strike_distance"] < best_put["strike_distance"]
            )
        ):
            best_put = candidate

    call_oi = best_call["open_interest"] if best_call else None
    put_oi = best_put["open_interest"] if best_put else None
    call_ok = call_oi is not None and call_oi >= int(min_open_interest)
    put_ok = put_oi is not None and put_oi >= int(min_open_interest)
    if call_ok and put_ok:
        return OptionsLiquidityMetrics(
            symbol,
            float(latest_close),
            call_oi,
            put_oi,
            str(best_call.get("symbol") or ""),
            str(best_put.get("symbol") or ""),
            checked,
            pages_fetched,
            True,
        )
    if not best_call and not best_put:
        reason = "no near-atm contracts"
    elif not call_ok and not put_ok:
        reason = "thin call and put open interest"
    elif not call_ok:
        reason = "thin call open interest"
    else:
        reason = "thin put open interest"
    return OptionsLiquidityMetrics(
        symbol,
        float(latest_close),
        call_oi,
        put_oi,
        str((best_call or {}).get("symbol") or "") or None,
        str((best_put or {}).get("symbol") or "") or None,
        checked,
        pages_fetched,
        False,
        reason,
    )


def _retryable_options_error(exc: Exception) -> bool:
    if isinstance(exc, HTTPError):
        return exc.code == 429 or 500 <= exc.code <= 599
    return isinstance(exc, (URLError, TimeoutError))


def options_liquidity_for_symbol(
    metric: DollarVolumeMetrics,
    *,
    client: Optional[AlpacaAssetDiscoveryClient] = None,
) -> OptionsLiquidityMetrics:
    symbol = metric.symbol
    if not metric.passed:
        return OptionsLiquidityMetrics(symbol, metric.latest_close, None, None, None, None, 0, 0, False, "dollar volume not passed")
    if metric.latest_close is None or metric.latest_close <= 0:
        return OptionsLiquidityMetrics(symbol, metric.latest_close, None, None, None, None, 0, 0, False, "missing current price")

    discovery_client = client or AlpacaAssetDiscoveryClient()
    params = option_contract_request_params(symbol, metric.latest_close)
    attempts = max(int(os.getenv("DISCOVERY_OPTIONS_MAX_ATTEMPTS", str(DISCOVERY_OPTIONS_MAX_ATTEMPTS)) or 1), 1)
    backoff = max(float(os.getenv("DISCOVERY_OPTIONS_RETRY_BACKOFF_SECONDS", str(DISCOVERY_OPTIONS_RETRY_BACKOFF_SECONDS)) or 0), 0)
    last_error = None
    for attempt in range(attempts):
        try:
            contracts, pages = discovery_client.fetch_option_contracts_pages(params)
            return options_liquidity_metrics_from_contracts(symbol, metric.latest_close, contracts, pages_fetched=pages)
        except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
            last_error = exc
            if attempt >= attempts - 1 or not _retryable_options_error(exc):
                break
            time.sleep(backoff * (attempt + 1))
    reason = "option contracts fetch failed"
    if isinstance(last_error, HTTPError) and last_error.code == 429:
        reason = "option contracts rate limited"
    return OptionsLiquidityMetrics(symbol, metric.latest_close, None, None, None, None, 0, 0, False, reason)


def stage4_options_liquidity_filter(
    dollar_volume_metrics: list[DollarVolumeMetrics],
    *,
    client: Optional[AlpacaAssetDiscoveryClient] = None,
    max_workers: int = DISCOVERY_OPTIONS_MAX_WORKERS,
) -> list[OptionsLiquidityMetrics]:
    candidates = [metric for metric in dollar_volume_metrics if metric.passed]
    if not candidates:
        return []
    worker_count = max(int(max_workers or 1), 1)
    results: list[OptionsLiquidityMetrics] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(options_liquidity_for_symbol, metric, client=client): metric.symbol
            for metric in candidates
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                results.append(future.result())
            except Exception:
                results.append(OptionsLiquidityMetrics(symbol, None, None, None, None, None, 0, 0, False, "option contracts fetch failed"))
    return sorted(results, key=lambda item: item.symbol)


def _percentile_ranks(values_by_symbol: dict[str, float]) -> dict[str, float]:
    if not values_by_symbol:
        return {}
    ordered = sorted(values_by_symbol.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    if count == 1:
        return {ordered[0][0]: 1.0}

    ranks = {}
    index = 0
    while index < count:
        value = ordered[index][1]
        end = index
        while end + 1 < count and ordered[end + 1][1] == value:
            end += 1
        percentile = ((index + end) / 2) / (count - 1)
        for position in range(index, end + 1):
            ranks[ordered[position][0]] = percentile
        index = end + 1
    return ranks


def rank_discovery_candidates(
    dollar_volume_metrics: list[DollarVolumeMetrics],
    options_metrics: list[OptionsLiquidityMetrics],
    *,
    target_size: Optional[int] = None,
) -> list[RankedDiscoveryCandidate]:
    effective_target_size = discovery_universe_max_symbols() if target_size is None else target_size
    dollar_by_symbol = {
        metric.symbol: metric
        for metric in dollar_volume_metrics
        if metric.passed and metric.average_daily_dollar_volume is not None
    }
    option_by_symbol = {
        metric.symbol: metric
        for metric in options_metrics
        if metric.passed
        and metric.near_atm_call_open_interest is not None
        and metric.near_atm_put_open_interest is not None
        and metric.symbol in dollar_by_symbol
    }
    if not option_by_symbol:
        return []

    dollar_percentiles = _percentile_ranks({
        symbol: float(dollar_by_symbol[symbol].average_daily_dollar_volume or 0)
        for symbol in option_by_symbol
    })
    call_percentiles = _percentile_ranks({
        symbol: float(metric.near_atm_call_open_interest or 0)
        for symbol, metric in option_by_symbol.items()
    })
    put_percentiles = _percentile_ranks({
        symbol: float(metric.near_atm_put_open_interest or 0)
        for symbol, metric in option_by_symbol.items()
    })

    candidates = []
    for symbol, option_metric in option_by_symbol.items():
        dollar_metric = dollar_by_symbol[symbol]
        dollar_rank = dollar_percentiles[symbol]
        call_rank = call_percentiles[symbol]
        put_rank = put_percentiles[symbol]
        combined_score = (
            DISCOVERY_RANK_DOLLAR_VOLUME_WEIGHT * dollar_rank
            + DISCOVERY_RANK_CALL_OI_WEIGHT * call_rank
            + DISCOVERY_RANK_PUT_OI_WEIGHT * put_rank
        )
        candidates.append(RankedDiscoveryCandidate(
            symbol=symbol,
            latest_close=dollar_metric.latest_close,
            average_daily_dollar_volume=dollar_metric.average_daily_dollar_volume,
            near_atm_call_open_interest=option_metric.near_atm_call_open_interest,
            near_atm_put_open_interest=option_metric.near_atm_put_open_interest,
            dollar_volume_percentile=dollar_rank,
            call_open_interest_percentile=call_rank,
            put_open_interest_percentile=put_rank,
            combined_liquidity_score=combined_score,
            rank=0,
            selected=False,
        ))

    selected_cutoff = max(int(effective_target_size or 0), 0)
    ranked = []
    for index, candidate in enumerate(sorted(
        candidates,
        key=lambda item: (
            item.combined_liquidity_score,
            item.average_daily_dollar_volume or 0,
            min(item.near_atm_call_open_interest or 0, item.near_atm_put_open_interest or 0),
            item.symbol,
        ),
        reverse=True,
    ), start=1):
        ranked.append(RankedDiscoveryCandidate(
            symbol=candidate.symbol,
            latest_close=candidate.latest_close,
            average_daily_dollar_volume=candidate.average_daily_dollar_volume,
            near_atm_call_open_interest=candidate.near_atm_call_open_interest,
            near_atm_put_open_interest=candidate.near_atm_put_open_interest,
            dollar_volume_percentile=candidate.dollar_volume_percentile,
            call_open_interest_percentile=candidate.call_open_interest_percentile,
            put_open_interest_percentile=candidate.put_open_interest_percentile,
            combined_liquidity_score=candidate.combined_liquidity_score,
            rank=index,
            selected=index <= selected_cutoff,
        ))
    return ranked


def _sample_metric(metric: DollarVolumeMetrics) -> dict[str, Any]:
    return {
        "symbol": metric.symbol,
        "latest_close": metric.latest_close,
        "average_daily_volume": metric.average_daily_volume,
        "average_daily_dollar_volume": metric.average_daily_dollar_volume,
        "valid_daily_bars": metric.valid_daily_bars,
        "passed": metric.passed,
        "rejection_reason": metric.rejection_reason,
    }


def _sample_options_metric(metric: OptionsLiquidityMetrics) -> dict[str, Any]:
    return {
        "symbol": metric.symbol,
        "latest_close": metric.latest_close,
        "near_atm_call_open_interest": metric.near_atm_call_open_interest,
        "near_atm_put_open_interest": metric.near_atm_put_open_interest,
        "near_atm_call_contract": metric.near_atm_call_contract,
        "near_atm_put_contract": metric.near_atm_put_contract,
        "near_atm_contracts_checked": metric.near_atm_contracts_checked,
        "pages_fetched": metric.pages_fetched,
        "passed": metric.passed,
        "rejection_reason": metric.rejection_reason,
    }


def _sample_ranked_candidate(candidate: RankedDiscoveryCandidate) -> dict[str, Any]:
    return {
        "rank": candidate.rank,
        "symbol": candidate.symbol,
        "latest_close": candidate.latest_close,
        "average_daily_dollar_volume": candidate.average_daily_dollar_volume,
        "near_atm_call_open_interest": candidate.near_atm_call_open_interest,
        "near_atm_put_open_interest": candidate.near_atm_put_open_interest,
        "dollar_volume_percentile": round(candidate.dollar_volume_percentile, 4),
        "call_open_interest_percentile": round(candidate.call_open_interest_percentile, 4),
        "put_open_interest_percentile": round(candidate.put_open_interest_percentile, 4),
        "combined_liquidity_score": round(candidate.combined_liquidity_score, 4),
        "selected": candidate.selected,
    }


def build_ranked_discovery_universe(static_watchlist: Optional[list[str]] = None) -> dict[str, Any]:
    """Run the full discovery pipeline and return a cache-ready universe report.

    This is intentionally opt-in and expensive. Callers should run it from a
    background/manual job, never synchronously from a user-facing scan request.
    """
    started = time.perf_counter()
    client = AlpacaAssetDiscoveryClient()
    assets = client.fetch_assets()
    stage1 = stage1_optionable_assets(assets)
    stage2 = stage2_hygiene_assets(stage1)

    stage3_started = time.perf_counter()
    dollar_metrics, stage3_failures = stage3_dollar_volume_filter(stage2)
    stage3_elapsed = round(time.perf_counter() - stage3_started, 2)
    stage3_passed = [metric for metric in dollar_metrics if metric.passed]
    stage3_failed = [metric for metric in dollar_metrics if not metric.passed]

    stage4_started = time.perf_counter()
    options_metrics = stage4_options_liquidity_filter(dollar_metrics)
    stage4_elapsed = round(time.perf_counter() - stage4_started, 2)
    stage4_passed = [metric for metric in options_metrics if metric.passed]
    stage4_failed = [metric for metric in options_metrics if not metric.passed]

    effective_cap = discovery_universe_max_symbols()
    ranked = rank_discovery_candidates(dollar_metrics, options_metrics, target_size=effective_cap)
    selected = [candidate for candidate in ranked if candidate.selected]
    selected_symbols = [candidate.symbol for candidate in selected]
    selected_symbol_set = set(selected_symbols)
    watchlist = {str(symbol or "").strip().upper() for symbol in (static_watchlist or []) if str(symbol or "").strip()}
    watchlist_overlap = {
        "watchlist_count": len(watchlist),
        "overlap": len(watchlist & selected_symbol_set),
        "missing": sorted(watchlist - selected_symbol_set),
    }

    stage3_failure_reasons = {
        reason: sum(1 for metric in stage3_failed if metric.rejection_reason == reason)
        for reason in {metric.rejection_reason for metric in stage3_failed}
    }
    stage4_failure_reasons = {
        reason: sum(1 for metric in stage4_failed if metric.rejection_reason == reason)
        for reason in {metric.rejection_reason for metric in stage4_failed}
    }
    return {
        "symbols": selected_symbols,
        "pipeline_counts": {
            "raw_assets": len(assets),
            "tradable_optionable": len(stage1),
            "hygiene_passed": len(stage2),
            "dollar_volume_passed": len(stage3_passed),
            "options_liquidity_passed": len(stage4_passed),
            "ranked": len(ranked),
            "selected": len(selected),
        },
        "thresholds": {
            "average_daily_dollar_volume_floor": DISCOVERY_MIN_AVG_DOLLAR_VOLUME,
            "minimum_valid_daily_bars": DISCOVERY_MIN_VALID_DAILY_BARS,
            "flat_price_floor": None,
            "minimum_dte": DISCOVERY_OPTIONS_MIN_DTE,
            "maximum_dte": DISCOVERY_OPTIONS_MAX_DTE,
            "near_atm_strike_band_pct": DISCOVERY_OPTIONS_STRIKE_BAND_PCT,
            "minimum_call_open_interest": DISCOVERY_OPTIONS_MIN_OPEN_INTEREST,
            "minimum_put_open_interest": DISCOVERY_OPTIONS_MIN_OPEN_INTEREST,
            "target_universe_size": effective_cap,
            "default_target_universe_size": DISCOVERY_DEFAULT_UNIVERSE_MAX_SYMBOLS,
            "target_universe_size_env": DISCOVERY_UNIVERSE_MAX_SYMBOLS_ENV,
        },
        "formula": {
            "combined_liquidity_score": (
                f"{DISCOVERY_RANK_DOLLAR_VOLUME_WEIGHT:.2f} * dollar_volume_percentile"
                f" + {DISCOVERY_RANK_CALL_OI_WEIGHT:.2f} * call_open_interest_percentile"
                f" + {DISCOVERY_RANK_PUT_OI_WEIGHT:.2f} * put_open_interest_percentile"
            ),
        },
        "stage3": {
            "elapsed_seconds": stage3_elapsed,
            "fetch_failures": len(stage3_failures),
            "failure_reasons": dict(sorted(stage3_failure_reasons.items())),
        },
        "stage4": {
            "elapsed_seconds": stage4_elapsed,
            "failure_reasons": dict(sorted(stage4_failure_reasons.items())),
            "total_pages_fetched": sum(metric.pages_fetched for metric in options_metrics),
        },
        "top_20": [_sample_ranked_candidate(candidate) for candidate in ranked[:20]],
        "bottom_20_selected": [_sample_ranked_candidate(candidate) for candidate in selected[-20:]],
        "watchlist_overlap": watchlist_overlap,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }


def _sample_asset(asset: dict[str, Any]) -> dict[str, Any]:
    return {
        "symbol": asset.get("symbol"),
        "name": asset.get("name"),
        "status": asset.get("status"),
        "class": asset.get("class"),
        "asset_class": asset.get("asset_class"),
        "exchange": asset.get("exchange"),
        "tradable": asset.get("tradable"),
        "attributes": asset.get("attributes"),
    }


def main() -> int:
    run_stage3 = "--stage3" in sys.argv
    run_stage4 = "--stage4" in sys.argv
    run_stage5 = "--stage5" in sys.argv
    client = AlpacaAssetDiscoveryClient()
    assets = client.fetch_assets()
    stage1 = stage1_optionable_assets(assets)
    stage2 = stage2_hygiene_assets(stage1)
    counts = discovery_stage_counts(assets)
    report = {
        "request": {
            "method": "GET",
            "url": f"{client.base_url}{ALPACA_ASSETS_ENDPOINT}?status=active&asset_class=us_equity",
            "headers": ["APCA-API-KEY-ID", "APCA-API-SECRET-KEY"],
        },
        "counts": counts.__dict__,
        "sample_stage1_assets": [_sample_asset(asset) for asset in stage1[:8]],
        "sample_hygiene_rejections": [
            {
                **_sample_asset(asset),
                "hygiene_rejection_reason": symbol_hygiene_rejection_reason(asset),
            }
            for asset in stage1
            if symbol_hygiene_rejection_reason(asset)
        ][:12],
        "sample_stage2_symbols": [asset_symbol(asset) for asset in stage2[:25]],
    }
    if run_stage3 or run_stage4 or run_stage5:
        started = time.perf_counter()
        metrics, failures = stage3_dollar_volume_filter(stage2)
        elapsed_seconds = round(time.perf_counter() - started, 2)
        passed = [metric for metric in metrics if metric.passed]
        failed = [metric for metric in metrics if not metric.passed]
        report["stage3"] = {
            "thresholds": {
                "average_daily_dollar_volume_floor": DISCOVERY_MIN_AVG_DOLLAR_VOLUME,
                "lookback_valid_daily_bars": DISCOVERY_DOLLAR_VOLUME_LOOKBACK_BARS,
                "minimum_valid_daily_bars": DISCOVERY_MIN_VALID_DAILY_BARS,
                "flat_price_floor": None,
            },
            "elapsed_seconds": elapsed_seconds,
            "input_symbols": len(stage2),
            "passed": len(passed),
            "failed_or_unverifiable": len(failed),
            "fetch_failures": len(failures),
            "failure_reasons": dict(sorted({reason: sum(1 for metric in failed if metric.rejection_reason == reason) for reason in {metric.rejection_reason for metric in failed}}.items())),
            "sample_passed": [_sample_metric(metric) for metric in sorted(passed, key=lambda item: item.average_daily_dollar_volume or 0, reverse=True)[:20]],
            "sample_failed": [_sample_metric(metric) for metric in sorted(failed, key=lambda item: item.average_daily_dollar_volume or 0, reverse=True)[:20]],
            "ford": _sample_metric(next((metric for metric in metrics if metric.symbol == "F"), DollarVolumeMetrics("F", None, None, None, 0, False, "not in input"))),
        }
        if run_stage4 or run_stage5:
            options_started = time.perf_counter()
            options_metrics = stage4_options_liquidity_filter(metrics)
            options_elapsed_seconds = round(time.perf_counter() - options_started, 2)
            options_passed = [metric for metric in options_metrics if metric.passed]
            options_failed = [metric for metric in options_metrics if not metric.passed]
            report["stage4"] = {
                "thresholds": {
                    "minimum_dte": DISCOVERY_OPTIONS_MIN_DTE,
                    "maximum_dte": DISCOVERY_OPTIONS_MAX_DTE,
                    "near_atm_strike_band_pct": DISCOVERY_OPTIONS_STRIKE_BAND_PCT,
                    "minimum_call_open_interest": DISCOVERY_OPTIONS_MIN_OPEN_INTEREST,
                    "minimum_put_open_interest": DISCOVERY_OPTIONS_MIN_OPEN_INTEREST,
                    "snapshot_spread_checks": False,
                },
                "elapsed_seconds": options_elapsed_seconds,
                "input_symbols": len(passed),
                "passed": len(options_passed),
                "failed_or_unverifiable": len(options_failed),
                "failure_reasons": dict(sorted({reason: sum(1 for metric in options_failed if metric.rejection_reason == reason) for reason in {metric.rejection_reason for metric in options_failed}}.items())),
                "total_pages_fetched": sum(metric.pages_fetched for metric in options_metrics),
                "sample_passed": [_sample_options_metric(metric) for metric in sorted(options_passed, key=lambda item: min(item.near_atm_call_open_interest or 0, item.near_atm_put_open_interest or 0), reverse=True)[:20]],
                "sample_failed": [_sample_options_metric(metric) for metric in sorted(options_failed, key=lambda item: min(item.near_atm_call_open_interest or 0, item.near_atm_put_open_interest or 0), reverse=True)[:20]],
                "ford": _sample_options_metric(next((metric for metric in options_metrics if metric.symbol == "F"), OptionsLiquidityMetrics("F", None, None, None, None, None, 0, 0, False, "not in input"))),
            }
            if run_stage5:
                effective_cap = discovery_universe_max_symbols()
                ranked = rank_discovery_candidates(metrics, options_metrics, target_size=effective_cap)
                selected = [candidate for candidate in ranked if candidate.selected]
                report["stage5"] = {
                    "target_universe_size": effective_cap,
                    "default_target_universe_size": DISCOVERY_DEFAULT_UNIVERSE_MAX_SYMBOLS,
                    "target_universe_size_env": DISCOVERY_UNIVERSE_MAX_SYMBOLS_ENV,
                    "formula": {
                        "combined_liquidity_score": (
                            f"{DISCOVERY_RANK_DOLLAR_VOLUME_WEIGHT:.2f} * dollar_volume_percentile"
                            f" + {DISCOVERY_RANK_CALL_OI_WEIGHT:.2f} * call_open_interest_percentile"
                            f" + {DISCOVERY_RANK_PUT_OI_WEIGHT:.2f} * put_open_interest_percentile"
                        ),
                        "flat_price_floor": None,
                    },
                    "input_symbols": len(options_passed),
                    "ranked": len(ranked),
                    "selected": len(selected),
                    "top_20": [_sample_ranked_candidate(candidate) for candidate in ranked[:20]],
                    "bottom_20_selected": [_sample_ranked_candidate(candidate) for candidate in selected[-20:]],
                    "full_selected_symbols": [candidate.symbol for candidate in selected],
                }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
