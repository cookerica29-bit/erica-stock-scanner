"""Curated-watchlist merge + source_universe tagging.

Scope:
  - main._merge_curated_watchlist_into_universe: dedup/overlap logic and
    curated-only-first ordering, using scanner.WATCHLIST directly (not a
    mocked stand-in) so a real change to WATCHLIST is exercised here too.
  - ma_pipeline.scan_ma_pipeline_candidates: symbol_origins is purely
    informational bookkeeping -- it must never change which candidates get
    produced (that's still what "symbols" contains) or how they're scored,
    only what source_universe ends up stamped on the result.

Does not touch candidates_router's storage layer here -- see
scanner_candidates_ingestion_v1.py (existing file, updated this session) for
the source_universe DB round-trip through CandidateIn/CandidateOut/
upsert_candidate_shortlist/list_candidates.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ma_pipeline
from scanner import WATCHLIST


def _valid_daily_frame() -> pd.DataFrame:
    """Monotonically ascending daily closes -- satisfies
    _candidate_from_frames' long-signal regime check (latest > sma50 > sma200)
    for any symbol it's handed. Verified directly against
    ma_pipeline._candidate_from_frames before being fixed here."""
    n = 250
    index = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    close = pd.Series([100.0 + i * 0.05 for i in range(n)], index=index)
    return pd.DataFrame({"Close": close})


def _valid_4h_frame() -> pd.DataFrame:
    n = 40
    index = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
    close = pd.Series([112.0 + i * 0.01 for i in range(n)], index=index)
    return pd.DataFrame({"Close": close})


@pytest.fixture()
def mock_alpaca(monkeypatch):
    monkeypatch.setattr(ma_pipeline, "alpaca_credentials_configured", lambda: True)

    def _fake_download(provider, symbols, period, interval):
        return _valid_daily_frame() if interval == "1d" else _valid_4h_frame()

    monkeypatch.setattr(ma_pipeline, "_download_with_fallback", _fake_download)


def test_scan_stamps_source_universe_from_origin_map(mock_alpaca):
    origins = {"AAPL": "broker_feed", "REGN": "curated_watchlist", "MSFT": "both"}

    result = ma_pipeline.scan_ma_pipeline_candidates(["AAPL", "REGN", "MSFT"], symbol_origins=origins)

    by_ticker = {c["ticker"]: c for c in result["candidates"]}
    assert by_ticker["AAPL"]["source_universe"] == "broker_feed"
    assert by_ticker["REGN"]["source_universe"] == "curated_watchlist"
    assert by_ticker["MSFT"]["source_universe"] == "both"


def test_scan_defaults_source_universe_to_none_without_origin_map(mock_alpaca):
    """Backward compatibility: a caller that never passes symbol_origins
    (there isn't one today, but nothing should require it) gets the exact
    same candidates as before this field existed, just with
    source_universe=None instead of the key being absent."""
    result = ma_pipeline.scan_ma_pipeline_candidates(["AAPL"])

    assert result["candidates"][0]["source_universe"] is None


def test_scan_defaults_unmapped_symbol_to_none(mock_alpaca):
    """A symbol_origins map that's missing an entry for a requested symbol
    (shouldn't happen given how main._merge_curated_watchlist_into_universe
    builds it, but the function shouldn't crash or mislabel if it does)."""
    result = ma_pipeline.scan_ma_pipeline_candidates(["AAPL"], symbol_origins={"MSFT": "broker_feed"})

    assert result["candidates"][0]["source_universe"] is None


def test_symbol_origins_never_changes_candidate_count_or_scoring(mock_alpaca):
    """The core non-negotiable from the spec: source_universe is bookkeeping
    only. Same symbols in, with or without an origin map, produce identical
    candidates (same fields, same values) except for the new key."""
    without_origins = ma_pipeline.scan_ma_pipeline_candidates(["AAPL", "MSFT"])
    with_origins = ma_pipeline.scan_ma_pipeline_candidates(
        ["AAPL", "MSFT"], symbol_origins={"AAPL": "curated_watchlist", "MSFT": "both"}
    )

    def _without_origin_field(candidates):
        return [{k: v for k, v in c.items() if k != "source_universe"} for c in candidates]

    assert _without_origin_field(without_origins["candidates"]) == _without_origin_field(with_origins["candidates"])


# -- main._merge_curated_watchlist_into_universe ------------------------------

def _import_merge_fn():
    import main
    return main._merge_curated_watchlist_into_universe


def test_merge_tags_broker_only_symbols_correctly():
    merge = _import_merge_fn()
    discovered = ["AAPL", "TSLA", "ZZZZ_NOT_ON_WATCHLIST"]

    merged, origins = merge(discovered)

    assert origins["ZZZZ_NOT_ON_WATCHLIST"] == "broker_feed"
    assert "ZZZZ_NOT_ON_WATCHLIST" in merged


def test_merge_tags_overlap_symbols_as_both():
    merge = _import_merge_fn()
    # AAPL is on WATCHLIST (scanner.py) and also passed in as broker-discovered.
    assert "AAPL" in WATCHLIST
    merged, origins = merge(["AAPL"])

    assert origins["AAPL"] == "both"
    assert merged.count("AAPL") == 1  # deduped, not listed twice


def test_merge_adds_curated_only_symbols_not_in_discovered_universe():
    merge = _import_merge_fn()
    # Real production numbers (confirmed via /api/discovery/status this
    # session): REGN, SQ, UNG are on WATCHLIST but were NOT in the broker-fed
    # universe. Use a discovered list that deliberately excludes them.
    discovered = [s for s in WATCHLIST if s not in {"REGN", "SQ", "UNG"}]

    merged, origins = merge(discovered)

    for symbol in ("REGN", "SQ", "UNG"):
        assert symbol in merged
        assert origins[symbol] == "curated_watchlist"


def test_merge_places_curated_only_symbols_before_discovered_symbols():
    """Defensive ordering: if a future tighter max_symbols cap ever
    truncates the merged list, it should eat into the broker feed's tail,
    never silently drop the handful of symbols WATCHLIST was curated for."""
    merge = _import_merge_fn()
    discovered = ["ZZZZ1", "ZZZZ2"]  # neither on WATCHLIST -- pure broker_feed

    merged, origins = merge(discovered)

    watchlist_set = set(WATCHLIST)
    curated_positions = [i for i, s in enumerate(merged) if s in watchlist_set]
    broker_positions = [i for i, s in enumerate(merged) if s in {"ZZZZ1", "ZZZZ2"}]
    assert curated_positions, "expected at least one curated-only symbol in a fresh merge"
    assert max(curated_positions) < min(broker_positions)


def test_merge_handles_empty_discovered_list():
    """Discovery not ready / empty cache shouldn't crash the merge -- the
    curated watchlist should still be scannable on its own."""
    merge = _import_merge_fn()
    merged, origins = merge([])

    assert set(merged) == set(WATCHLIST)
    assert all(origin == "curated_watchlist" for origin in origins.values())


def test_merge_is_case_and_whitespace_normalizing():
    merge = _import_merge_fn()
    merged, origins = merge([" aapl ", "tsla"])

    assert "AAPL" in merged and "TSLA" in merged
    assert origins["AAPL"] == "both"  # AAPL is on WATCHLIST
