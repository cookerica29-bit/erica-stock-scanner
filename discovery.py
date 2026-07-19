"""Alpaca-backed ticker discovery primitives.

This module is intentionally separate from scanner.py. It builds candidate
universes for later review; it does not feed the live scanner path directly.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen


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
    client = AlpacaAssetDiscoveryClient()
    assets = client.fetch_assets()
    stage1 = stage1_optionable_assets(assets)
    stage2 = stage2_hygiene_assets(stage1)
    counts = discovery_stage_counts(assets)
    print(json.dumps({
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
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
