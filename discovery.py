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
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from market_data import AlpacaMarketDataProvider


DEFAULT_ALPACA_TRADING_BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_ASSETS_ENDPOINT = "/v2/assets"
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
    if run_stage3:
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
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
