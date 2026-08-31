"""Stage-2 ranking (Option A) -- candidates_router.py.

_stage1_mechanical_ready: the redefined stage 1 (regime, valid target, R:R,
entry proximity, execution confirmation only -- confluence-conflicted
deliberately excluded, see its own comment in candidates_router.py). This is
an ADDITIVE read used only by the new ranking endpoint; it does not touch
_promotion_block_reason/routeBlockReason, so real ENTER_NOW promotion still
excludes conflicted candidates exactly as it does today.

rank_stage1_candidates / _confluence_sort_key: Option A from the design
pass -- confluence favorable desc, unfavorable asc, tiebroken by R:R desc
then entry-proximity asc. Zero new weights, reuses confluence_counts
exactly as confluence_summary.py already computes it.

Real motivation for the KMI-shaped edge case tested below (confirmed live,
2026-08-31): KMI cleared every stage-1 mechanical gate that day but had
confluence_signals/counts/label all null (confluence never computed for
it -- location_percentile was also null). A ranking mechanism that crashes
on that (None can't compare to int) or silently drops the candidate is a
real bug, not a hypothetical -- tested directly here.

The "some confluence" ordering test below replays real counts observed
live for IGV/FDS/ELF that same day (5/1, 3/1, 3/1 favorable/unfavorable,
R:R 4.04/5.83/4.07) as a concrete sanity check, not just synthetic data.
"""

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import candidates_router as router  # noqa: E402


class _FakeRow(dict):
    def __getitem__(self, key):
        return dict.get(self, key)


def _candidate(signal="long", daily_regime="long"):
    return _FakeRow({"signal": signal, "daily_regime": daily_regime, "ticker": "X"})


def _preview(
    ticker="X",
    source="ma_pipeline",
    signal="long",
    no_valid_target=False,
    target=100.0,
    risk_reward=2.0,
    rr_warning=False,
    entry_proximity_ok=True,
    execution_shadow_ok=True,
    confluence_counts=None,
    entry_distance_pct=None,
    preview_error=None,
):
    return {
        "ticker": ticker,
        "source": source,
        "signal": signal,
        "no_valid_target": no_valid_target,
        "target": target,
        "risk_reward": risk_reward,
        "rr_warning": rr_warning,
        "entry_proximity_ok": entry_proximity_ok,
        "execution_shadow_ok": execution_shadow_ok,
        "confluence_counts": confluence_counts,
        "entry_distance_pct": entry_distance_pct,
        "preview_error": preview_error,
    }


# ---------------------------------------------------------------------------
# _stage1_mechanical_ready
# ---------------------------------------------------------------------------

def test_stage1_ready_when_every_mechanical_gate_passes():
    assert router._stage1_mechanical_ready(_candidate(), _preview()) is True


def test_stage1_blocks_on_regime_misalignment():
    cand = _candidate(daily_regime="short")
    assert router._stage1_mechanical_ready(cand, _preview()) is False


def test_stage1_blocks_on_invalid_target():
    assert router._stage1_mechanical_ready(_candidate(), _preview(no_valid_target=True)) is False


def test_stage1_blocks_on_rr_below_threshold():
    assert router._stage1_mechanical_ready(_candidate(), _preview(risk_reward=1.2)) is False


def test_stage1_blocks_on_proximity_not_ok():
    assert router._stage1_mechanical_ready(_candidate(), _preview(entry_proximity_ok=False)) is False


def test_stage1_blocks_on_execution_shadow_not_true():
    assert router._stage1_mechanical_ready(_candidate(), _preview(execution_shadow_ok=False)) is False
    assert router._stage1_mechanical_ready(_candidate(), _preview(execution_shadow_ok=None)) is False


def test_stage1_ignores_confluence_conflicted():
    # The whole point of the redefinition: a mechanically-clean candidate
    # stays stage-1-ready regardless of confluence_label/counts -- that
    # exclusion moved to ranking, not stage-1 filtering.
    preview = _preview(confluence_counts={"favorable": 0, "unfavorable": 3, "neutral": 4, "applicable": 7})
    assert router._stage1_mechanical_ready(_candidate(), preview) is True


# ---------------------------------------------------------------------------
# _confluence_sort_key / rank_stage1_candidates
# ---------------------------------------------------------------------------

def test_ranking_orders_by_favorable_count_descending():
    a = _preview(ticker="A", confluence_counts={"favorable": 1, "unfavorable": 0, "neutral": 6, "applicable": 7})
    b = _preview(ticker="B", confluence_counts={"favorable": 4, "unfavorable": 0, "neutral": 3, "applicable": 7})
    ranked = router.rank_stage1_candidates([(_candidate(), a), (_candidate(), b)])
    assert [p["ticker"] for p in ranked] == ["B", "A"]
    assert [p["rank"] for p in ranked] == [1, 2]


def test_ranking_breaks_favorable_tie_by_unfavorable_ascending():
    a = _preview(ticker="A", confluence_counts={"favorable": 2, "unfavorable": 2, "neutral": 3, "applicable": 7})
    b = _preview(ticker="B", confluence_counts={"favorable": 2, "unfavorable": 0, "neutral": 5, "applicable": 7})
    ranked = router.rank_stage1_candidates([(_candidate(), a), (_candidate(), b)])
    assert [p["ticker"] for p in ranked] == ["B", "A"]


def test_ranking_breaks_confluence_tie_by_risk_reward_descending():
    a = _preview(ticker="A", risk_reward=1.6, confluence_counts={"favorable": 2, "unfavorable": 1, "neutral": 4, "applicable": 7})
    b = _preview(ticker="B", risk_reward=3.2, confluence_counts={"favorable": 2, "unfavorable": 1, "neutral": 4, "applicable": 7})
    ranked = router.rank_stage1_candidates([(_candidate(), a), (_candidate(), b)])
    assert [p["ticker"] for p in ranked] == ["B", "A"]


def test_ranking_breaks_final_tie_by_proximity_ascending():
    a = _preview(
        ticker="A", risk_reward=2.0, entry_distance_pct=1.4,
        confluence_counts={"favorable": 2, "unfavorable": 1, "neutral": 4, "applicable": 7},
    )
    b = _preview(
        ticker="B", risk_reward=2.0, entry_distance_pct=0.2,
        confluence_counts={"favorable": 2, "unfavorable": 1, "neutral": 4, "applicable": 7},
    )
    ranked = router.rank_stage1_candidates([(_candidate(), a), (_candidate(), b)])
    assert [p["ticker"] for p in ranked] == ["B", "A"]


def test_ranking_confluence_unavailable_sorts_after_all_available_and_does_not_crash():
    # KMI-shaped: confluence never computed (None), not just zero counts.
    unavailable = _preview(ticker="KMI", confluence_counts=None)
    available = _preview(ticker="WFC", confluence_counts={"favorable": 1, "unfavorable": 0, "neutral": 6, "applicable": 7})
    ranked = router.rank_stage1_candidates([(_candidate(), unavailable), (_candidate(), available)])
    assert [p["ticker"] for p in ranked] == ["WFC", "KMI"]
    assert ranked[0]["confluence_available"] is True
    assert ranked[1]["confluence_available"] is False


def test_ranking_excludes_candidates_that_fail_stage1():
    ready = _preview(ticker="READY")
    not_ready = _preview(ticker="NOTREADY", execution_shadow_ok=False)
    ranked = router.rank_stage1_candidates([(_candidate(), ready), (_candidate(), not_ready)])
    assert [p["ticker"] for p in ranked] == ["READY"]


def test_ranking_real_data_igv_fds_elf_2026_08_31():
    # Real confluence_counts/risk_reward observed live for these three
    # tickers on 2026-08-31 (see module docstring) -- IGV's higher
    # favorable count outranks FDS/ELF's tied 3/1, and FDS's higher R:R
    # (5.83 vs 4.07) breaks that tie ahead of ELF.
    igv = _preview(ticker="IGV", risk_reward=4.04, confluence_counts={"favorable": 5, "unfavorable": 1, "neutral": 1, "applicable": 7})
    fds = _preview(ticker="FDS", risk_reward=5.83, confluence_counts={"favorable": 3, "unfavorable": 1, "neutral": 3, "applicable": 7})
    elf = _preview(ticker="ELF", risk_reward=4.07, confluence_counts={"favorable": 3, "unfavorable": 1, "neutral": 3, "applicable": 7})
    ranked = router.rank_stage1_candidates([(_candidate(), elf), (_candidate(), fds), (_candidate(), igv)])
    assert [p["ticker"] for p in ranked] == ["IGV", "FDS", "ELF"]


def test_ranking_is_stable_on_a_full_tie():
    a = _preview(ticker="FIRST", confluence_counts={"favorable": 2, "unfavorable": 1, "neutral": 4, "applicable": 7})
    b = _preview(ticker="SECOND", confluence_counts={"favorable": 2, "unfavorable": 1, "neutral": 4, "applicable": 7})
    ranked = router.rank_stage1_candidates([(_candidate(), a), (_candidate(), b)])
    assert [p["ticker"] for p in ranked] == ["FIRST", "SECOND"]


# ---------------------------------------------------------------------------
# GET /candidates/ranked -- end-to-end wiring (real ingestion + preview
# computation, same fixture/monkeypatch scaffold as
# scanner_candidates_ingestion_v1.py's NVDA case). Ordering itself is
# already proven above at the unit level; this proves the endpoint's own
# wiring -- auth, candidate/preview key-matching, response shape.
# ---------------------------------------------------------------------------

def _promotion_daily_frame():
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


def test_ranked_endpoint_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["KAIROS_CANDIDATES_DB"] = os.path.join(tmp, "candidates.db")
        os.environ["KAIROS_SCANNER_API_KEY"] = "test-scanner-key"

        previous = {
            "_batch_download": router._batch_download,
            "_best_contract": router._best_contract,
            "_latest_quote_for_ticker": router._latest_quote_for_ticker,
            "_latest_quotes_for_previews": router._latest_quotes_for_previews,
            "_recent_4h_bars_for_execution_shadow": router._recent_4h_bars_for_execution_shadow,
        }
        router._batch_download = lambda tickers, period, interval: {str(tickers[0]).upper(): _promotion_daily_frame()}
        router._best_contract = lambda ticker, direction, entry, **kwargs: {
            "available": True, "execution": "Fair", "type": "CALL", "strike": 100.0,
            "expiry": "2026-09-18", "dte": 29, "symbol": "MOCK", "source": "option_chain",
            "bid": 1.10, "ask": 1.20, "mid": 1.15, "mark": 1.15, "estimated_contract_cost": 120.0,
        }
        router._latest_quote_for_ticker = lambda ticker: {
            "price": 100.0, "timestamp": "2026-08-20T18:30:00Z", "source": "mock_latest_quote", "price_branch": "mid",
        }
        router._latest_quotes_for_previews = lambda previews: {
            str(preview.get("ticker") or "").upper(): {
                "price": 100.0, "timestamp": "2026-08-20T18:30:00Z", "source": "mock_latest_quote", "price_branch": "mid",
            }
            for preview in previews
        }
        router._recent_4h_bars_for_execution_shadow = lambda ticker: [
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

        try:
            app = FastAPI()
            app.include_router(router.router)
            client = TestClient(app)
            headers = {"X-API-Key": "test-scanner-key"}

            unauthorized = client.get("/api/v1/scanner/candidates/ranked")
            assert unauthorized.status_code == 401

            payload = {
                "source": "ma_pipeline",
                "scanned_at": "2026-08-20T14:30:00Z",
                "candidates": [{
                    "ticker": "NVDA", "signal": "long", "entry_price": 100.0, "ema21_4h": 99.0,
                    "daily_regime": "bullish", "confidence": "high", "sma50_daily": 106.0, "sma200_daily": 104.0,
                }],
            }
            created = client.post("/api/v1/scanner/candidates", headers=headers, json=payload)
            assert created.status_code == 200

            response = client.get("/api/v1/scanner/candidates/ranked", headers=headers)
            assert response.status_code == 200
            body = response.json()
            assert body["mechanism"] == router.RANKING_MECHANISM_VERSION
            assert body["disclaimer"] == router.RANKING_DISCLAIMER
            assert "unvalidated" in body["disclaimer"].lower()
            assert body["count"] == len(body["candidates"]) == 1
            entry = body["candidates"][0]
            assert entry["ticker"] == "NVDA"
            assert entry["rank"] == 1
            assert "confluence_available" in entry
            assert body["snapshot_id"]
        finally:
            for name, fn in previous.items():
                setattr(router, name, fn)
