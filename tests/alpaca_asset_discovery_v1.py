#!/usr/bin/env python3
"""Tests for Alpaca asset discovery stages 1-2."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import discovery  # noqa: E402


def asset(symbol, name, **overrides):
    base = {
        "symbol": symbol,
        "name": name,
        "status": "active",
        "class": "us_equity",
        "exchange": "NASDAQ",
        "tradable": True,
        "attributes": ["fractional_eh_enabled", "has_options", "overnight_tradable"],
    }
    base.update(overrides)
    return base


def test_stage1_accepts_active_tradable_non_otc_optionable_common_stocks():
    assets = [
        asset("AMAT", "Applied Materials, Inc. Common Stock"),
        asset("BP", "BP p.l.c.", exchange="NYSE"),
        asset("BF.A", "Brown-Forman Corporation Class A", exchange="NYSE"),
    ]
    accepted = discovery.stage1_optionable_assets(assets)
    assert [discovery.asset_symbol(item) for item in accepted] == ["AMAT", "BP", "BF.A"]


def test_stage1_rejects_inactive_non_tradable_otc_and_non_optionable_assets():
    assets = [
        asset("AMAT", "Applied Materials, Inc. Common Stock"),
        asset("OLD", "Old Corp Common Stock", status="inactive"),
        asset("HALT", "Halted Corp Common Stock", tradable=False),
        asset("OTCQ", "OTC Corp Common Stock", exchange="OTC"),
        asset("NOOPT", "No Options Corp Common Stock", attributes=["overnight_tradable"]),
    ]
    accepted = discovery.stage1_optionable_assets(assets)
    assert [discovery.asset_symbol(item) for item in accepted] == ["AMAT"]


def test_hygiene_accepts_real_common_stock_shapes():
    for item in [
        asset("AMAT", "Applied Materials, Inc. Common Stock"),
        asset("BP", "BP p.l.c.", exchange="NYSE"),
        asset("BF.A", "Brown-Forman Corporation Class A", exchange="NYSE"),
    ]:
        assert discovery.passes_symbol_hygiene(item), item


def test_hygiene_accepts_adr_ads_and_lp_unit_shapes_that_are_not_derivatives():
    for item in [
        asset("CMCM", "Cheetah Mobile Inc. American Depositary Shares, each representing fifty (50) Class A Ordinary Shares", exchange="NYSE"),
        asset("AZUL", "Azul S.A. American Depositary Shares, each representing two (2) Common Shares", exchange="NYSE"),
        asset("ZLAB", "Zai Lab Limited American Depositary Shares"),
        asset("ERIC", "Ericsson American Depositary Shares"),
        asset("AMX", "America Movil S.A.B de C.V American Depositary Shares (each representing the right to receive twenty (20) Series B Shares)", exchange="NYSE"),
        asset("ET", "Energy Transfer LP Common Units representing limited partner interests", exchange="NYSE"),
        asset("PAA", "Plains All American Pipeline, L.P. Common Units representing Limited Partner Interests"),
        asset("ARLP", "Alliance Resource Partners, L.P. Common Units representing Limited Partners Interests"),
        asset("UNG", "United States Natural Gas Fund, LP Unit", exchange="ARCA"),
    ]:
        assert discovery.passes_symbol_hygiene(item), item


def test_hygiene_rejects_warrants_preferreds_rights_and_spac_units():
    cases = [
        (asset("KEYYW", "Keystone Acquisition Corp. Warrants"), "warrant"),
        (asset("CDAUF", "COMPASS DIGITAL ACQUISITION CORP Unit"), "spac unit"),
        (asset("HCDPQ", "HARBOR CUSTOM DEV INC Preferred Stock Cumulative Series A 8%"), "preferred"),
        (asset("VHAQR", "VIVEON HEALTH ACQUISITION CORP Rights Exp 09/30/2024"), "right"),
        (asset("GENVR", "Gen Digital Inc. Contingent Value Rights"), "right"),
        (asset("FPE", "First Trust Preferred Securities and Income ETF", exchange="ARCA"), "preferred"),
    ]
    for item, expected_reason in cases:
        assert discovery.symbol_hygiene_rejection_reason(item) == expected_reason
        assert not discovery.passes_symbol_hygiene(item)


def test_hygiene_rejects_derivative_symbol_suffixes():
    cases = [
        asset("ABC.WS", "ABC Corp Warrants"),
        asset("ABC.U", "ABC Corp Unit"),
        asset("ABC.RT", "ABC Corp Rights"),
        asset("ABC.PR", "ABC Corp Preferred"),
    ]
    for item in cases:
        assert discovery.symbol_hygiene_rejection_reason(item) == "derivative symbol suffix"


def test_hygiene_rejects_leveraged_inverse_etfs_without_blocking_plain_bull_company_names():
    leveraged = [
        asset("NIOG", "Themes ETF Trust Leverage Shares 2X Long NIO Daily ETF"),
        asset("CBRZ", "Tradr 2X Short CBRS Daily ETF"),
        asset("ASMU", "Direxion Shares ETF Trust Direxion Daily ASML Bull 2X ETF"),
        asset("UMDD", "ProShares UltraPro MidCap400", exchange="ARCA"),
        asset("SQQQ", "ProShares UltraPro Short QQQ", exchange="ARCA"),
        asset("SPXU", "ProShares UltraPro Short S&P 500", exchange="ARCA"),
        asset("QLD", "ProShares Ultra QQQ", exchange="ARCA"),
        asset("YCS", "ProShares UltraShort Yen", exchange="ARCA"),
        asset("DOG", "ProShares Short Dow30", exchange="ARCA"),
        asset("SARK", "Investment Managers Series Trust II Tradr 1X Short Innovation Daily ETF"),
        asset("SKHZ", "Leverage Shares 1X Short SK Hynix Daily ETF", exchange="BATS"),
    ]
    for item in leveraged:
        assert discovery.symbol_hygiene_rejection_reason(item) == "leveraged/inverse etf"

    plain_company = asset("SVBL", "SILVER BULL RES INC Common Stock")
    assert discovery.passes_symbol_hygiene(plain_company)

    short_duration_fund = asset("XBOX", "Roundhill Ultra Short Duration No Dividend Target ETF")
    assert discovery.passes_symbol_hygiene(short_duration_fund)


def test_discovery_stage_counts():
    assets = [
        asset("AMAT", "Applied Materials, Inc. Common Stock"),
        asset("NOOPT", "No Options Corp Common Stock", attributes=[]),
        asset("NIOG", "Themes ETF Trust Leverage Shares 2X Long NIO Daily ETF"),
        asset("OTCQ", "OTC Corp Common Stock", exchange="OTC"),
    ]
    counts = discovery.discovery_stage_counts(assets)
    assert counts.raw_assets == 4
    assert counts.active_tradable_non_otc_optionable == 2
    assert counts.hygiene_passed == 1


def main() -> int:
    test_stage1_accepts_active_tradable_non_otc_optionable_common_stocks()
    test_stage1_rejects_inactive_non_tradable_otc_and_non_optionable_assets()
    test_hygiene_accepts_real_common_stock_shapes()
    test_hygiene_accepts_adr_ads_and_lp_unit_shapes_that_are_not_derivatives()
    test_hygiene_rejects_warrants_preferreds_rights_and_spac_units()
    test_hygiene_rejects_derivative_symbol_suffixes()
    test_hygiene_rejects_leveraged_inverse_etfs_without_blocking_plain_bull_company_names()
    test_discovery_stage_counts()
    print("Alpaca asset discovery v1 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
