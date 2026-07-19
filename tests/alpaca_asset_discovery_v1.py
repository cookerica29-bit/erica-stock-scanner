#!/usr/bin/env python3
"""Tests for Alpaca asset discovery stages 1-2."""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.error import HTTPError

import pandas as pd

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


def bars(closes, volumes):
    return pd.DataFrame({
        "Close": closes,
        "Volume": volumes,
    }, index=pd.date_range("2026-06-01", periods=len(closes), freq="B"))


def test_dollar_volume_metrics_pass_without_flat_price_floor():
    df = bars([2.0] * 30, [60_000_000] * 30)
    metrics = discovery.dollar_volume_metrics_for_frame("LOWP", df)
    assert metrics.passed is True
    assert metrics.latest_close == 2.0
    assert metrics.average_daily_volume == 60_000_000
    assert metrics.average_daily_dollar_volume == 120_000_000
    assert metrics.valid_daily_bars == 30
    assert metrics.rejection_reason is None


def test_dollar_volume_metrics_reject_low_dollar_volume_and_insufficient_bars():
    low_volume = discovery.dollar_volume_metrics_for_frame("THIN", bars([20.0] * 30, [1_000_000] * 30))
    assert low_volume.passed is False
    assert low_volume.rejection_reason == "low dollar volume"

    insufficient = discovery.dollar_volume_metrics_for_frame("SHORT", bars([100.0] * 24, [2_000_000] * 24))
    assert insufficient.passed is False
    assert insufficient.rejection_reason == "insufficient valid daily bars"
    assert insufficient.average_daily_dollar_volume == 200_000_000


def test_apply_dollar_volume_filter_tracks_unverifiable_fetch_failures():
    metrics = discovery.apply_dollar_volume_filter(
        ["GOOD", "MISS"],
        {"GOOD": bars([100.0] * 30, [2_000_000] * 30)},
        {"MISS": "no price data"},
    )
    by_symbol = {metric.symbol: metric for metric in metrics}
    assert by_symbol["GOOD"].passed is True
    assert by_symbol["MISS"].passed is False
    assert by_symbol["MISS"].rejection_reason == "no price data"


class FakeBarsProvider:
    def __init__(self):
        self.calls = []

    def download(self, tickers, **kwargs):
        symbols = list(tickers)
        self.calls.append((symbols, dict(kwargs)))
        frames = {
            symbol: bars([10.0] * 30, [20_000_000] * 30)
            for symbol in symbols
        }
        return pd.concat(frames, axis=1)


def test_fetch_discovery_daily_bars_batches_through_provider():
    provider = FakeBarsProvider()
    frames, failures = discovery.fetch_discovery_daily_bars(
        ["AAA", "BBB", "CCC"],
        provider=provider,
        batch_size=2,
    )
    assert [call[0] for call in provider.calls] == [["AAA", "BBB"], ["CCC"]]
    assert failures == {}
    assert sorted(frames) == ["AAA", "BBB", "CCC"]


def dollar_metric(symbol="AAA", latest_close=100.0):
    return discovery.DollarVolumeMetrics(
        symbol=symbol,
        latest_close=latest_close,
        average_daily_volume=2_000_000,
        average_daily_dollar_volume=200_000_000,
        valid_daily_bars=30,
        passed=True,
    )


def contract(symbol, contract_type, strike, open_interest, *, underlying="AAA", tradable=True, status="active"):
    return {
        "symbol": symbol,
        "underlying_symbol": underlying,
        "type": contract_type,
        "strike_price": str(strike),
        "open_interest": str(open_interest),
        "tradable": tradable,
        "status": status,
    }


class FakeOptionsClient:
    def __init__(self, pages_by_symbol):
        self.pages_by_symbol = pages_by_symbol
        self.calls = []

    def fetch_option_contracts_pages(self, params):
        symbol = params["underlying_symbols"]
        self.calls.append(dict(params))
        page = self.pages_by_symbol[symbol]
        if isinstance(page, list):
            next_item = page.pop(0)
            if isinstance(next_item, Exception):
                raise next_item
            return next_item
        if isinstance(page, Exception):
            raise page
        return page


def test_option_contract_request_params_use_dte_window_and_strike_band():
    params = discovery.option_contract_request_params("AAPL", 200.0, today=__import__("datetime").date(2026, 7, 19))
    assert params["underlying_symbols"] == "AAPL"
    assert params["status"] == "active"
    assert params["expiration_date_gte"] == "2026-08-02"
    assert params["expiration_date_lte"] == "2026-09-17"
    assert params["strike_price_gte"] == "180.0000"
    assert params["strike_price_lte"] == "220.0000"
    assert params["limit"] == discovery.DISCOVERY_OPTIONS_CONTRACT_LIMIT


def test_options_liquidity_requires_call_and_put_open_interest():
    contracts = [
        contract("AAA260821C00100000", "call", 100, 250),
        contract("AAA260821P00100000", "put", 100, 125),
    ]
    metrics = discovery.options_liquidity_metrics_from_contracts("AAA", 100.0, contracts, pages_fetched=1)
    assert metrics.passed is True
    assert metrics.near_atm_call_open_interest == 250
    assert metrics.near_atm_put_open_interest == 125
    assert metrics.near_atm_contracts_checked == 2


def test_options_liquidity_rejects_when_either_side_is_thin():
    call_thin = discovery.options_liquidity_metrics_from_contracts(
        "AAA",
        100.0,
        [
            contract("AAA260821C00100000", "call", 100, 99),
            contract("AAA260821P00100000", "put", 100, 200),
        ],
    )
    assert call_thin.passed is False
    assert call_thin.rejection_reason == "thin call open interest"

    put_thin = discovery.options_liquidity_metrics_from_contracts(
        "AAA",
        100.0,
        [
            contract("AAA260821C00100000", "call", 100, 200),
            contract("AAA260821P00100000", "put", 100, 99),
        ],
    )
    assert put_thin.passed is False
    assert put_thin.rejection_reason == "thin put open interest"


def test_options_liquidity_filters_to_near_atm_tradable_active_contracts():
    metrics = discovery.options_liquidity_metrics_from_contracts(
        "AAA",
        100.0,
        [
            contract("AAA260821C00150000", "call", 150, 999),
            contract("AAA260821P00050000", "put", 50, 999),
            contract("AAA260821C00100000", "call", 100, 175, tradable=False),
            contract("AAA260821C00100001", "call", 101, 180),
            contract("AAA260821P00099000", "put", 99, 190, status="inactive"),
            contract("AAA260821P00100001", "put", 101, 195),
        ],
    )
    assert metrics.passed is True
    assert metrics.near_atm_call_contract == "AAA260821C00100001"
    assert metrics.near_atm_put_contract == "AAA260821P00100001"
    assert metrics.near_atm_contracts_checked == 2


def test_options_liquidity_fetch_failure_fails_closed():
    client = FakeOptionsClient({"AAA": RuntimeError("timeout")})
    metrics = discovery.options_liquidity_for_symbol(dollar_metric("AAA"), client=client)
    assert metrics.passed is False
    assert metrics.rejection_reason == "option contracts fetch failed"


def test_options_liquidity_retries_rate_limit_then_recovers():
    import os

    previous_attempts = os.environ.get("DISCOVERY_OPTIONS_MAX_ATTEMPTS")
    previous_backoff = os.environ.get("DISCOVERY_OPTIONS_RETRY_BACKOFF_SECONDS")
    os.environ["DISCOVERY_OPTIONS_MAX_ATTEMPTS"] = "2"
    os.environ["DISCOVERY_OPTIONS_RETRY_BACKOFF_SECONDS"] = "0"
    try:
        client = FakeOptionsClient({
            "AAA": [
                HTTPError(url="https://example.test", code=429, msg="rate limited", hdrs=None, fp=None),
                ([
                    contract("AAA260821C00100000", "call", 100, 200),
                    contract("AAA260821P00100000", "put", 100, 200),
                ], 1),
            ],
        })
        metrics = discovery.options_liquidity_for_symbol(dollar_metric("AAA"), client=client)
        assert metrics.passed is True
        assert len(client.calls) == 2
    finally:
        if previous_attempts is None:
            os.environ.pop("DISCOVERY_OPTIONS_MAX_ATTEMPTS", None)
        else:
            os.environ["DISCOVERY_OPTIONS_MAX_ATTEMPTS"] = previous_attempts
        if previous_backoff is None:
            os.environ.pop("DISCOVERY_OPTIONS_RETRY_BACKOFF_SECONDS", None)
        else:
            os.environ["DISCOVERY_OPTIONS_RETRY_BACKOFF_SECONDS"] = previous_backoff


def test_stage4_options_liquidity_filter_uses_only_stage3_passed_symbols():
    client = FakeOptionsClient({
        "AAA": ([
            contract("AAA260821C00100000", "call", 100, 200),
            contract("AAA260821P00100000", "put", 100, 200),
        ], 1),
    })
    failed_dollar = discovery.DollarVolumeMetrics("MISS", None, None, None, 0, False, "low dollar volume")
    metrics = discovery.stage4_options_liquidity_filter(
        [dollar_metric("AAA"), failed_dollar],
        client=client,
        max_workers=1,
    )
    assert [metric.symbol for metric in metrics] == ["AAA"]
    assert metrics[0].passed is True
    assert [call["underlying_symbols"] for call in client.calls] == ["AAA"]


def test_percentile_ranks_handle_ties_and_single_values():
    ranks = discovery._percentile_ranks({"LOW": 10.0, "MID1": 20.0, "MID2": 20.0, "HIGH": 40.0})
    assert ranks["LOW"] == 0.0
    assert ranks["MID1"] == ranks["MID2"]
    assert ranks["MID1"] == (1 + 2) / 2 / 3
    assert ranks["HIGH"] == 1.0
    assert discovery._percentile_ranks({"ONLY": 99.0}) == {"ONLY": 1.0}


def test_rank_discovery_candidates_combines_dollar_volume_and_both_options_sides():
    dollar_metrics = [
        dollar_metric("DOLLAR", latest_close=100.0),
        dollar_metric("BALANCED", latest_close=100.0),
        dollar_metric("OPTIONS", latest_close=100.0),
        dollar_metric("WEAK", latest_close=100.0),
    ]
    dollar_metrics[0] = discovery.DollarVolumeMetrics("DOLLAR", 100.0, 10_000_000, 1_000_000_000, 30, True)
    dollar_metrics[1] = discovery.DollarVolumeMetrics("BALANCED", 100.0, 8_000_000, 800_000_000, 30, True)
    dollar_metrics[2] = discovery.DollarVolumeMetrics("OPTIONS", 100.0, 2_000_000, 200_000_000, 30, True)
    dollar_metrics[3] = discovery.DollarVolumeMetrics("WEAK", 100.0, 1_000_000, 100_000_000, 30, True)
    option_metrics = [
        discovery.OptionsLiquidityMetrics("DOLLAR", 100.0, 100, 100, "DOLLARC", "DOLLARP", 2, 1, True),
        discovery.OptionsLiquidityMetrics("BALANCED", 100.0, 600, 600, "BALANCEDC", "BALANCEDP", 2, 1, True),
        discovery.OptionsLiquidityMetrics("OPTIONS", 100.0, 1_000, 1_000, "OPTIONSC", "OPTIONSP", 2, 1, True),
        discovery.OptionsLiquidityMetrics("WEAK", 100.0, 100, 100, "WEAKC", "WEAKP", 2, 1, True),
    ]
    ranked = discovery.rank_discovery_candidates(dollar_metrics, option_metrics, target_size=2)
    assert ranked[0].symbol == "BALANCED"
    assert [candidate.symbol for candidate in ranked if candidate.selected] == ["BALANCED", "OPTIONS"]
    assert ranked[0].combined_liquidity_score == ranked[1].combined_liquidity_score
    assert ranked[1].combined_liquidity_score > ranked[2].combined_liquidity_score
    assert ranked[-1].symbol == "WEAK"


def test_rank_discovery_candidates_ignores_unpassed_or_unmatched_metrics():
    dollar_metrics = [
        discovery.DollarVolumeMetrics("GOOD", 100.0, 2_000_000, 200_000_000, 30, True),
        discovery.DollarVolumeMetrics("LOWVOL", 100.0, 1_000, 100_000, 30, False, "low dollar volume"),
    ]
    option_metrics = [
        discovery.OptionsLiquidityMetrics("GOOD", 100.0, 200, 200, "GOODC", "GOODP", 2, 1, True),
        discovery.OptionsLiquidityMetrics("LOWVOL", 100.0, 999, 999, "LOWVOLC", "LOWVOLP", 2, 1, True),
        discovery.OptionsLiquidityMetrics("NO_DOLLAR", 100.0, 999, 999, "NOC", "NOP", 2, 1, True),
        discovery.OptionsLiquidityMetrics("THIN", 100.0, 20, 20, "THINC", "THINP", 2, 1, False, "thin call and put open interest"),
    ]
    ranked = discovery.rank_discovery_candidates(dollar_metrics, option_metrics, target_size=10)
    assert [candidate.symbol for candidate in ranked] == ["GOOD"]
    assert ranked[0].rank == 1
    assert ranked[0].selected is True


def main() -> int:
    test_stage1_accepts_active_tradable_non_otc_optionable_common_stocks()
    test_stage1_rejects_inactive_non_tradable_otc_and_non_optionable_assets()
    test_hygiene_accepts_real_common_stock_shapes()
    test_hygiene_accepts_adr_ads_and_lp_unit_shapes_that_are_not_derivatives()
    test_hygiene_rejects_warrants_preferreds_rights_and_spac_units()
    test_hygiene_rejects_derivative_symbol_suffixes()
    test_hygiene_rejects_leveraged_inverse_etfs_without_blocking_plain_bull_company_names()
    test_discovery_stage_counts()
    test_dollar_volume_metrics_pass_without_flat_price_floor()
    test_dollar_volume_metrics_reject_low_dollar_volume_and_insufficient_bars()
    test_apply_dollar_volume_filter_tracks_unverifiable_fetch_failures()
    test_fetch_discovery_daily_bars_batches_through_provider()
    test_option_contract_request_params_use_dte_window_and_strike_band()
    test_options_liquidity_requires_call_and_put_open_interest()
    test_options_liquidity_rejects_when_either_side_is_thin()
    test_options_liquidity_filters_to_near_atm_tradable_active_contracts()
    test_options_liquidity_fetch_failure_fails_closed()
    test_options_liquidity_retries_rate_limit_then_recovers()
    test_stage4_options_liquidity_filter_uses_only_stage3_passed_symbols()
    test_percentile_ranks_handle_ties_and_single_values()
    test_rank_discovery_candidates_combines_dollar_volume_and_both_options_sides()
    test_rank_discovery_candidates_ignores_unpassed_or_unmatched_metrics()
    print("Alpaca asset discovery v1 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
