"""Kairos -- Review Queue asset-type eligibility (2026-09 session, Sprint 1
follow-up). Excludes ETFs/funds from GET /candidates/review-queue
eligibility, BEFORE global Review Value ranking, scoped entirely to that
endpoint. Does not touch ma_pipeline.py, GET /candidates/ranked, scanner
strategy, Review Value weights, or any lifecycle code.

Classification: "ETF" or "FUND" (case-insensitive) in the real Alpaca
asset name, plus a small curated supplement (QQQ/GLD/SLV) for legacy-
named funds that predate that naming convention. A bare "TRUST" keyword
was deliberately rejected -- tests/test_ntrs... below proves why (a real,
confirmed false positive on Northern Trust Corporation, a genuine
individual bank stock).

Every test exercises the real candidates_router.py code via a real
FastAPI TestClient -- not a reimplementation. _review_queue_asset_names
is monkeypatched to real, verified Alpaca asset names pulled during this
work (not invented), so these tests reflect genuine classification
behavior, not synthetic data.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import candidates_router as router  # noqa: E402

# Captured at import time, before any test monkeypatches it away -- lets
# one test (below) restore the REAL implementation to exercise its own
# internal try/except directly, matching the _REAL_EXECUTION_WINDOW_STATE
# pattern already established elsewhere this session.
_REAL_REVIEW_QUEUE_ASSET_NAMES = router._review_queue_asset_names

# Real Alpaca asset names, verified live against production Alpaca
# credentials during this work (2026-09) -- not invented.
REAL_ASSET_NAMES = {
    "VGK": "Vanguard FTSE Europe ETF",
    "VYM": "Vanguard High Dividend Yield ETF",
    "SHY": "iShares 1-3 Year Treasury Bond ETF",
    "ACWI": "iShares MSCI ACWI ETF",
    "VEA": "Vanguard FTSE Developed Markets ETF",
    "VTI": "Vanguard Morningstar Total Stock Market ETF",
    "IVV": "iShares Core S&P 500 ETF",
    "EEM": "iShares MSCI Emerging Markets ETF",
    "VOO": "Vanguard S&P 500 ETF",
    "USO": "United States Oil Fund, LP",
    "UNG": "United States Natural Gas Fund, LP Unit",
    "QQQ": "Invesco QQQ Trust, Series 1",
    "GLD": "SPDR Gold Trust, SPDR Gold Shares",
    "SLV": "iShares Silver Trust",
    "DEO": "Diageo plc",
    "EMR": "Emerson Electric Co.",
    "COF": "Capital One Financial",
    "BKNG": "Booking Holdings Inc. Common Stock",
    "GKOS": "Glaukos Corporation",
    "ZBRA": "Zebra Technologies Corporation Class A Common Stock",
    "SWK": "Stanley Black & Decker, Inc.",
    "ALLE": "Allegion Public Limited Company",
    "OXY": "Occidental Petroleum Corporation",
    "AAPL": "Apple Inc. Common Stock",
    "ASND": "Ascendis Pharma A/S Ordinary Share",
    "NTRS": "Northern Trust Corporation Common Stock",
}


def _daily_frame():
    # Same verified values as tests/review_queue_evolution_sprint1_v1.py.
    index = pd.date_range("2026-01-01", periods=30, freq="D", tz="UTC")
    rows = []
    for i in range(30):
        close = 100.0 - i * 0.1
        rows.append({"Open": close + 0.2, "High": close + 0.5, "Low": close - 0.5, "Close": close, "Volume": 1_000_000})
    rows[5]["Low"] = 99.0
    rows[10]["Low"] = 90.0
    rows[10]["Close"] = 91.0
    rows[10]["Open"] = 92.0
    rows[20]["High"] = 110.0
    rows[20]["Close"] = 109.0
    rows[20]["Open"] = 108.0
    return pd.DataFrame(rows, index=index)


PASSING_SHADOW_BARS = [
    {
        "time": f"2026-08-19T{hour:02d}:00:00Z",
        "open": 99.0 + (idx * 0.05), "high": 101.0 + (idx * 0.05),
        "low": 98.0 + (idx * 0.05), "close": 100.0 + (idx * 0.05), "volume": 1000,
    }
    for idx, hour in enumerate(range(11))
] + [
    {"time": "2026-08-20T02:00:00Z", "open": 99.0, "high": 101.0, "low": 98.0, "close": 100.0, "volume": 1000},
    {"time": "2026-08-20T06:00:00Z", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1100},
    {"time": "2026-08-20T10:00:00Z", "open": 100.5, "high": 102.0, "low": 99.2, "close": 101.0, "volume": 1200},
    {"time": "2026-08-20T14:00:00Z", "open": 100.8, "high": 103.0, "low": 99.5, "close": 102.2, "volume": 1000},
]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_CANDIDATES_DB", str(tmp_path / "candidates.db"))
    monkeypatch.setenv("KAIROS_SCANNER_API_KEY", "test-scanner-key")
    app = FastAPI()
    app.include_router(router.router)
    return TestClient(app)


@pytest.fixture()
def headers():
    return {"X-API-Key": "test-scanner-key"}


@pytest.fixture(autouse=True)
def _mock_network(monkeypatch):
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {str(t).upper(): _daily_frame() for t in tickers})
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        str(p.get("ticker") or "").upper(): {"price": 100.0, "timestamp": "2026-08-20T18:30:00Z", "source": "mock", "price_branch": "mid"}
        for p in previews
    })
    monkeypatch.setattr(router, "_recent_4h_bars_for_execution_shadow", lambda ticker: PASSING_SHADOW_BARS)
    # Deterministic, real-shaped asset-name lookup -- no live Alpaca call
    # in tests. Individual test cases override with variants (empty,
    # raising) to exercise fail-open behavior.
    monkeypatch.setattr(router, "_review_queue_asset_names", lambda now: dict(REAL_ASSET_NAMES))


@pytest.fixture(autouse=True)
def _forbid_option_hydration(monkeypatch):
    def _spy(ticker, direction, entry, **kwargs):
        raise AssertionError(f"_safe_option_contract_for_candidate called unexpectedly: {ticker}")

    monkeypatch.setattr(router, "_safe_option_contract_for_candidate", _spy)


def _seed(client, headers, ticker, entry_price=100.0):
    payload = {
        "source": "ma_pipeline", "scanned_at": "2026-08-20T14:30:00Z",
        "candidates": [{
            "ticker": ticker, "signal": "long", "entry_price": entry_price, "ema21_4h": entry_price - 1.0,
            "daily_regime": "long", "confidence": "high",
            "sma50_daily": entry_price + 6.0, "sma200_daily": entry_price + 4.0,
        }],
    }
    resp = client.post("/api/v1/scanner/candidates", headers=headers, json=payload)
    assert resp.status_code == 200


def _get_queue(client, headers):
    resp = client.get("/api/v1/scanner/candidates/review-queue", headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _find(payload, ticker):
    for c in payload["candidates"]:
        if c["ticker"] == ticker:
            return c
    return None


# ==========================================================================
# Real ETFs/funds excluded
# ==========================================================================

@pytest.mark.parametrize("ticker", ["VGK", "VYM", "SHY", "ACWI", "VEA", "VTI", "IVV", "EEM", "VOO"])
def test_modern_etf_named_with_etf_is_excluded(client, headers, ticker):
    _seed(client, headers, ticker)
    payload = _get_queue(client, headers)
    assert _find(payload, ticker) is None, f"{ticker} ({REAL_ASSET_NAMES[ticker]!r}) must be excluded -- real ETF"


@pytest.mark.parametrize("ticker", ["USO", "UNG"])
def test_fund_named_with_fund_is_excluded(client, headers, ticker):
    _seed(client, headers, ticker)
    payload = _get_queue(client, headers)
    assert _find(payload, ticker) is None, f"{ticker} ({REAL_ASSET_NAMES[ticker]!r}) must be excluded -- real fund, name says 'Fund' not 'ETF'"


@pytest.mark.parametrize("ticker", ["QQQ", "GLD", "SLV"])
def test_legacy_named_fund_excluded_via_curated_supplement(client, headers, ticker):
    """QQQ/GLD/SLV real names contain neither 'ETF' nor 'FUND' -- only
    'Trust', which is deliberately NOT used as a keyword (see the NTRS
    false-positive test below). These rely on
    REVIEW_QUEUE_LEGACY_FUND_TICKERS instead."""
    assert "ETF" not in REAL_ASSET_NAMES[ticker].upper()
    assert "FUND" not in REAL_ASSET_NAMES[ticker].upper()
    _seed(client, headers, ticker)
    payload = _get_queue(client, headers)
    assert _find(payload, ticker) is None, f"{ticker} ({REAL_ASSET_NAMES[ticker]!r}) must be excluded via the curated legacy-fund list"


# ==========================================================================
# Real individual equities are NOT excluded -- including the NTRS
# false-positive regression case
# ==========================================================================

@pytest.mark.parametrize("ticker", ["DEO", "EMR", "COF", "BKNG", "GKOS", "ZBRA", "SWK", "ALLE", "OXY", "AAPL", "ASND"])
def test_real_individual_equity_not_excluded(client, headers, ticker):
    _seed(client, headers, ticker)
    payload = _get_queue(client, headers)
    entry = _find(payload, ticker)
    assert entry is not None, f"{ticker} ({REAL_ASSET_NAMES[ticker]!r}) is a genuine individual equity -- must not be excluded"
    assert entry["is_fund_or_etf"] is False


def test_ntrs_false_positive_regression(client, headers):
    """Northern Trust Corporation -- a REAL, individual, publicly-traded
    bank holding company -- has 'Trust' literally in its name. This is
    the concrete false positive that ruled out a bare 'TRUST' keyword;
    this test pins that decision so it can never silently regress."""
    _seed(client, headers, "NTRS")
    payload = _get_queue(client, headers)
    entry = _find(payload, "NTRS")
    assert entry is not None, "NTRS is a real bank stock, not a fund -- 'Trust' in the name must not cause exclusion"
    assert entry["is_fund_or_etf"] is False
    assert entry["asset_name"] == "Northern Trust Corporation Common Stock"


# ==========================================================================
# Diagnostics, asset-type fields on the response, ordering vs. quote fetch
# ==========================================================================

def test_excluded_fund_count_diagnostic(client, headers):
    _seed(client, headers, "VGK")
    _seed(client, headers, "AAPL")
    payload = _get_queue(client, headers)
    assert payload["diagnostics"]["excluded_fund_or_etf_count"] == 1
    assert _find(payload, "AAPL") is not None
    assert _find(payload, "VGK") is None


def test_asset_type_fields_present_on_displayed_candidates(client, headers):
    _seed(client, headers, "AAPL")
    payload = _get_queue(client, headers)
    entry = _find(payload, "AAPL")
    assert entry["is_fund_or_etf"] is False
    assert entry["asset_name"] == "Apple Inc. Common Stock"
    assert entry["asset_name_available"] is True


def test_excluded_fund_never_reaches_live_quote_fetch(client, headers, monkeypatch):
    """Performance: an excluded fund must not consume a live-quote fetch
    -- same 'don't pay for what's already excluded' philosophy as the
    existing mechanical prechecks (Gate 3)."""
    quoted_tickers = []

    def _spy_quotes(previews):
        tickers = sorted({str(p.get("ticker") or "").upper() for p in previews})
        quoted_tickers.extend(tickers)
        return {t: {"price": 100.0, "timestamp": "t", "source": "mock", "price_branch": "mid"} for t in tickers}

    monkeypatch.setattr(router, "_latest_quotes_for_previews", _spy_quotes)
    _seed(client, headers, "VGK")
    _seed(client, headers, "AAPL")
    _get_queue(client, headers)
    assert "VGK" not in quoted_tickers, "an excluded ETF must never reach the live-quote fetch"
    assert "AAPL" in quoted_tickers


# ==========================================================================
# Fail-open behavior
# ==========================================================================

def test_fails_open_when_asset_name_lookup_entirely_unavailable(client, headers, monkeypatch):
    """If the whole asset-name fetch fails (e.g. a real Alpaca outage),
    the review queue must not empty out -- real candidates still appear,
    simply un-classified (asset_name_available=False), never wrongly
    excluded."""
    monkeypatch.setattr(router, "_review_queue_asset_names", lambda now: {})
    _seed(client, headers, "AAPL")
    payload = _get_queue(client, headers)
    entry = _find(payload, "AAPL")
    assert entry is not None, "a total asset-name lookup failure must fail OPEN, never empty the queue"
    assert entry["is_fund_or_etf"] is False
    assert entry["asset_name_available"] is False


def test_fails_open_per_ticker_when_name_missing_but_fetch_succeeded(client, headers, monkeypatch):
    """The whole-fetch succeeds, but this ONE ticker isn't in the
    returned map (e.g. genuinely absent from Alpaca's active us_equity
    list) -- must still fail open for that ticker specifically, not be
    excluded."""
    monkeypatch.setattr(router, "_review_queue_asset_names", lambda now: {"OTHER": "Some Other Company Inc."})
    _seed(client, headers, "AAPL")
    payload = _get_queue(client, headers)
    entry = _find(payload, "AAPL")
    assert entry is not None
    assert entry["is_fund_or_etf"] is False
    assert entry["asset_name_available"] is False


def test_real_fetch_exception_is_caught_and_fails_open(client, headers, monkeypatch):
    """Exercises the ACTUAL _review_queue_asset_names function (restored
    over this file's own autouse mock) with a real exception raised
    inside the fetch -- confirms the try/except inside the real
    implementation itself, not just the test fixture's mock."""
    monkeypatch.setattr(router, "_review_queue_asset_names", _REAL_REVIEW_QUEUE_ASSET_NAMES)
    # Reset the module-level cache explicitly -- it's a shared global, so
    # this test must not depend on incidentally running before anything
    # else ever populates it (test order is not guaranteed).
    monkeypatch.setitem(router._review_queue_asset_name_cache, "fetched_at", None)
    monkeypatch.setitem(router._review_queue_asset_name_cache, "names", {})

    import discovery

    def _raise(*a, **k):
        raise RuntimeError("Alpaca credentials are not configured")

    monkeypatch.setattr(discovery.AlpacaAssetDiscoveryClient, "fetch_assets", _raise)

    _seed(client, headers, "AAPL")
    payload = _get_queue(client, headers)
    entry = _find(payload, "AAPL")
    assert entry is not None, "a real exception inside the asset-name fetch must be caught, never crash the review queue"
    assert entry["asset_name_available"] is False
