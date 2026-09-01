"""Stage-3 tracking infrastructure -- candidates_router.py.

Two append-only tables, both modeled on patterns already proven in this
codebase, not invented from scratch:

- candidate_visual_reviews: a human's structured stage-3 visual-review
  decision (approve/watch/reject), captured independent of promotion/taken
  -- so a review is recorded even for a candidate never promoted. Bound to
  setup_key, not just ticker (see _compute_setup_key's own docstring in
  candidates_router.py for the full 2026-08-31 investigation this is built
  on -- candidates.updated_at is NOT usable for this, since it changes on
  every routine rescan AND every status change, neither of which has
  anything to do with the trade thesis). "Not yet reviewed" is the absence
  of any row for a setup_key, not a stored null -- consistent with an
  append-only event log. GET /candidate-visual-reviews returns full
  history; "current decision" is a setup_key's latest row, same pattern
  GET /candidate-promotions already uses for its own append-only table.

- candidate_ranking_snapshots: one row per ranked candidate per ranking
  computation, all sharing one snapshot_id -- so what the ranking actually
  surfaced at a given moment can be compared later against what was
  visually approved and what was taken, without re-deriving ranking state
  from live data that's since moved.
"""

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import candidates_router as router  # noqa: E402


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


def _alternate_daily_frame():
    # A genuinely different structural picture from _promotion_daily_frame:
    # the swing-low/order-block candle and the gap-spike sit at different
    # bars with different magnitudes, so _find_order_block/
    # _nearest_structural_target pick different real levels -- a materially
    # different thesis, not the same setup with noisy price drift.
    index = pd.date_range("2026-01-01", periods=30, freq="D", tz="UTC")
    rows = []
    for i in range(30):
        close = 100.0 - i * 0.08
        rows.append({"Open": close + 0.2, "High": close + 0.5, "Low": close - 0.5, "Close": close, "Volume": 1_000_000})
    rows[3]["Low"] = 97.5
    rows[15]["Low"] = 85.0
    rows[15]["Close"] = 86.5
    rows[15]["Open"] = 88.0
    rows[25]["High"] = 118.0
    rows[25]["Close"] = 116.0
    rows[25]["Open"] = 114.0
    return pd.DataFrame(rows, index=index)


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


def _mock_network(monkeypatch, daily_frame_fn):
    monkeypatch.setattr(router, "_batch_download", lambda tickers, period, interval: {str(tickers[0]).upper(): daily_frame_fn()})
    monkeypatch.setattr(router, "_best_contract", lambda ticker, direction, entry, **kwargs: {
        "available": True, "execution": "Fair", "type": "CALL", "strike": 100.0,
        "expiry": "2026-09-18", "dte": 29, "symbol": "MOCK", "source": "option_chain",
        "bid": 1.10, "ask": 1.20, "mid": 1.15, "mark": 1.15, "estimated_contract_cost": 120.0,
    })
    monkeypatch.setattr(router, "_latest_quote_for_ticker", lambda ticker: {
        "price": 100.0, "timestamp": "2026-08-20T18:30:00Z", "source": "mock_latest_quote", "price_branch": "mid",
    })
    monkeypatch.setattr(router, "_latest_quotes_for_previews", lambda previews: {
        str(preview.get("ticker") or "").upper(): {
            "price": 100.0, "timestamp": "2026-08-20T18:30:00Z", "source": "mock_latest_quote", "price_branch": "mid",
        }
        for preview in previews
    })
    monkeypatch.setattr(router, "_recent_4h_bars_for_execution_shadow", lambda ticker: [
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
    ])


def _seed_candidate(client, headers, entry_price=100.0, scanned_at="2026-08-20T14:30:00Z"):
    payload = {
        "source": "ma_pipeline",
        "scanned_at": scanned_at,
        "candidates": [{
            "ticker": "NVDA", "signal": "long", "entry_price": entry_price, "ema21_4h": 99.0,
            "daily_regime": "bullish", "confidence": "high", "sma50_daily": 106.0, "sma200_daily": 104.0,
        }],
    }
    created = client.post("/api/v1/scanner/candidates", headers=headers, json=payload)
    assert created.status_code == 200
    return created


def _compute_preview(client, headers):
    """Forces a fresh plan-preview computation (via the ranked endpoint,
    same as the real review-queue flow) and returns it."""
    ranked = client.get("/api/v1/scanner/candidates/ranked", headers=headers).json()
    matches = [c for c in ranked["candidates"] if c["ticker"] == "NVDA"]
    if matches:
        return matches[0]
    # Not stage-1-ready in this fixture (e.g. execution_shadow/proximity) --
    # fall back to the raw preview, which is all setup_key needs.
    previews = client.get("/api/v1/scanner/candidate-plan-previews", headers=headers).json()
    return next(p for p in previews if p["ticker"] == "NVDA")


# ---------------------------------------------------------------------------
# _compute_setup_key -- unit level, no network
# ---------------------------------------------------------------------------

class _FakeRow(dict):
    def __getitem__(self, key):
        return dict.get(self, key)


def _preview(ticker="NVDA", source="ma_pipeline", signal="long", stop=90.0, target=110.0):
    return {"ticker": ticker, "source": source, "signal": signal, "stop": stop, "target": target}


def test_setup_key_stable_across_ordinary_refresh_signals():
    # entry_price is exactly what changes on an ordinary rescan -- it's not
    # even a parameter setup_key looks at, which is the point.
    candidate = _FakeRow({"ticker": "NVDA", "source": "ma_pipeline", "signal": "long"})
    key_a = router._compute_setup_key(candidate, _preview())
    key_b = router._compute_setup_key(candidate, _preview())
    assert key_a == key_b


def test_setup_key_changes_when_stop_or_target_materially_moves():
    candidate = _FakeRow({"ticker": "NVDA", "source": "ma_pipeline", "signal": "long"})
    original = router._compute_setup_key(candidate, _preview(stop=90.0, target=110.0))
    moved_stop = router._compute_setup_key(candidate, _preview(stop=85.0, target=110.0))
    moved_target = router._compute_setup_key(candidate, _preview(stop=90.0, target=118.0))
    assert moved_stop != original
    assert moved_target != original


def test_setup_key_ignores_float_noise_in_stop_target():
    candidate = _FakeRow({"ticker": "NVDA", "source": "ma_pipeline", "signal": "long"})
    key_a = router._compute_setup_key(candidate, _preview(stop=90.001, target=110.004))
    key_b = router._compute_setup_key(candidate, _preview(stop=89.999, target=110.001))
    assert key_a == key_b  # both round to 90.00 / 110.00


def test_setup_key_handles_missing_target():
    candidate = _FakeRow({"ticker": "NVDA", "source": "ma_pipeline", "signal": "long"})
    key = router._compute_setup_key(candidate, _preview(target=None))
    assert "no_target" in key


def test_setup_key_includes_ticker_source_direction():
    candidate = _FakeRow({"ticker": "NVDA", "source": "ma_pipeline", "signal": "long"})
    a = router._compute_setup_key(candidate, _preview(ticker="NVDA", source="ma_pipeline", signal="long"))
    b = router._compute_setup_key(candidate, _preview(ticker="AMD", source="ma_pipeline", signal="long"))
    c = router._compute_setup_key(candidate, _preview(ticker="NVDA", source="ma_pipeline", signal="short"))
    assert a != b
    assert a != c


# ---------------------------------------------------------------------------
# candidate_visual_reviews -- endpoint level
# ---------------------------------------------------------------------------

def test_visual_review_requires_auth(client):
    response = client.post(
        "/api/v1/scanner/candidates/AAPL/visual-review",
        json={
            "source": "ma_pipeline", "market_structure": "bullish", "location_read": "good",
            "clear_path_to_target": "yes", "lower_tf_confirmation": "yes", "confirmation_rule": "close_above", "confirmation_level": 100.0, "decision": "approve",
        },
    )
    assert response.status_code == 401


def test_visual_review_requires_a_real_candidate(client, headers, monkeypatch):
    _mock_network(monkeypatch, _promotion_daily_frame)
    # No candidate ingested at all -- no candidates row.
    response = client.post(
        "/api/v1/scanner/candidates/NVDA/visual-review",
        headers=headers,
        json={
            "source": "ma_pipeline", "market_structure": "bullish", "location_read": "good",
            "clear_path_to_target": "yes", "lower_tf_confirmation": "yes", "confirmation_rule": "close_above", "confirmation_level": 100.0, "decision": "approve",
        },
    )
    assert response.status_code == 404  # no candidate row at all


def test_visual_review_computes_setup_key_fresh_without_a_cached_preview(client, headers, monkeypatch):
    # Real gap found via Stage B testing: the review queue (Stage B) never
    # writes candidate_plan_previews, so requiring a cached row here would
    # make a review impossible for anyone using only the queue. This must
    # succeed -- computed fresh via _compute_review_queue_preview, the same
    # lightweight path the queue itself uses -- not 422.
    _mock_network(monkeypatch, _promotion_daily_frame)
    _seed_candidate(client, headers)
    # Deliberately no GET /candidate-plan-previews call first -- nothing
    # has cached anything for this candidate yet.
    response = client.post(
        "/api/v1/scanner/candidates/NVDA/visual-review",
        headers=headers,
        json={
            "source": "ma_pipeline", "market_structure": "bullish", "location_read": "good",
            "clear_path_to_target": "yes", "lower_tf_confirmation": "yes", "confirmation_rule": "close_above", "confirmation_level": 100.0, "decision": "approve",
        },
    )
    assert response.status_code == 200
    assert response.json()["setup_key"]


def test_visual_review_records_a_structured_decision(client, headers, monkeypatch):
    _mock_network(monkeypatch, _promotion_daily_frame)
    _seed_candidate(client, headers)
    _compute_preview(client, headers)

    response = client.post(
        "/api/v1/scanner/candidates/nvda/visual-review",
        headers=headers,
        json={
            "source": "ma_pipeline",
            "market_structure": "bullish",
            "location_read": "good",
            "clear_path_to_target": "yes",
            "lower_tf_confirmation": "yes",
            "confirmation_rule": "close_above",
            "confirmation_level": 100.0,
            "decision": "approve",
            "note": "Clean structure, taking it.",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "NVDA"
    assert body["source"] == "ma_pipeline"
    assert body["market_structure"] == "bullish"
    assert body["location_read"] == "good"
    assert body["clear_path_to_target"] == "yes"
    assert body["lower_tf_confirmation"] == "yes"
    assert body["decision"] == "approve"
    assert body["note"] == "Clean structure, taking it."
    assert body["setup_key"]
    assert body["reviewed_at"]
    assert isinstance(body["id"], int)



# ---------------------------------------------------------------------------
# Early practical-disqualification path (2026-09-01 session): a candidate
# can be practically untradeable (options too expensive, poor liquidity/
# spread) before a human ever reaches chart review. Real usage gap found:
# the form previously required all four visual-read fields before allowing
# ANY submission, forcing a reviewer to fabricate chart observations just
# to record "not trading this, the contract is unaffordable."
# ---------------------------------------------------------------------------

def test_practical_rejection_succeeds_with_all_visual_fields_null(client, headers, monkeypatch):
    _mock_network(monkeypatch, _promotion_daily_frame)
    _seed_candidate(client, headers)
    _compute_preview(client, headers)

    response = client.post(
        "/api/v1/scanner/candidates/NVDA/visual-review",
        headers=headers,
        json={
            "source": "ma_pipeline",
            "decision": "reject",
            "practical_rejection_reason": "options_too_expensive",
            "note": "Nearest liquid strike is $8+, not worth the risk.",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["review_type"] == "practical_rejection"
    assert body["practical_rejection_reason"] == "options_too_expensive"
    assert body["decision"] == "reject"
    # No fake/inferred/defaulted visual values -- real None, not a string
    # placeholder or a guessed enum value.
    assert body["market_structure"] is None
    assert body["location_read"] is None
    assert body["clear_path_to_target"] is None
    assert body["lower_tf_confirmation"] is None
    assert body["setup_key"]

    # And confirm what actually landed in the DB, not just the response
    # shape -- the response is built from the same row, but this closes
    # the loop against a hypothetical response-serialization bug hiding a
    # real fabricated value in storage.
    listed = client.get("/api/v1/scanner/candidate-visual-reviews", headers=headers).json()
    stored = next(r for r in listed if r["id"] == body["id"])
    assert stored["market_structure"] is None
    assert stored["location_read"] is None
    assert stored["clear_path_to_target"] is None
    assert stored["lower_tf_confirmation"] is None
    assert stored["review_type"] == "practical_rejection"


@pytest.mark.parametrize("reason", ["options_too_expensive", "poor_option_liquidity", "other"])
def test_practical_rejection_accepts_every_real_reason(client, headers, monkeypatch, reason):
    _mock_network(monkeypatch, _promotion_daily_frame)
    _seed_candidate(client, headers)
    _compute_preview(client, headers)

    response = client.post(
        "/api/v1/scanner/candidates/NVDA/visual-review",
        headers=headers,
        json={"source": "ma_pipeline", "decision": "reject", "practical_rejection_reason": reason},
    )
    assert response.status_code == 200
    assert response.json()["practical_rejection_reason"] == reason


def test_practical_rejection_rejects_invalid_reason_enum(client, headers, monkeypatch):
    _mock_network(monkeypatch, _promotion_daily_frame)
    _seed_candidate(client, headers)
    _compute_preview(client, headers)

    response = client.post(
        "/api/v1/scanner/candidates/NVDA/visual-review",
        headers=headers,
        json={"source": "ma_pipeline", "decision": "reject", "practical_rejection_reason": "just_dont_like_it"},
    )
    assert response.status_code == 422


@pytest.mark.parametrize("decision", ["approve", "watch"])
def test_practical_rejection_requires_decision_reject(client, headers, monkeypatch, decision):
    # There's no "approve, but the contract is unaffordable" case.
    _mock_network(monkeypatch, _promotion_daily_frame)
    _seed_candidate(client, headers)
    _compute_preview(client, headers)

    response = client.post(
        "/api/v1/scanner/candidates/NVDA/visual-review",
        headers=headers,
        json={"source": "ma_pipeline", "decision": decision, "practical_rejection_reason": "options_too_expensive"},
    )
    assert response.status_code == 422

    # Confirm nothing was actually written for the rejected request.
    listed = client.get("/api/v1/scanner/candidate-visual-reviews", headers=headers).json()
    assert listed == []


@pytest.mark.parametrize("field,value", [
    ("market_structure", "bullish"),
    ("location_read", "good"),
    ("clear_path_to_target", "yes"),
    ("lower_tf_confirmation", "yes"),
])
def test_practical_rejection_rejects_mixing_with_visual_fields(client, headers, monkeypatch, field, value):
    # A request can never produce a row with SOME real chart observations
    # and some None ones -- caught outright, not silently dropped.
    _mock_network(monkeypatch, _promotion_daily_frame)
    _seed_candidate(client, headers)
    _compute_preview(client, headers)

    payload = {"source": "ma_pipeline", "decision": "reject", "practical_rejection_reason": "options_too_expensive"}
    payload[field] = value
    response = client.post("/api/v1/scanner/candidates/NVDA/visual-review", headers=headers, json=payload)
    assert response.status_code == 422

    listed = client.get("/api/v1/scanner/candidate-visual-reviews", headers=headers).json()
    assert listed == []


def test_normal_rejection_still_requires_visual_fields(client, headers, monkeypatch):
    _mock_network(monkeypatch, _promotion_daily_frame)
    _seed_candidate(client, headers)
    _compute_preview(client, headers)

    response = client.post(
        "/api/v1/scanner/candidates/NVDA/visual-review",
        headers=headers,
        json={"source": "ma_pipeline", "decision": "reject"},  # no visual fields, no practical_rejection_reason
    )
    assert response.status_code == 422


@pytest.mark.parametrize("decision", ["approve", "watch"])
def test_approve_and_watch_still_require_visual_fields(client, headers, monkeypatch, decision):
    _mock_network(monkeypatch, _promotion_daily_frame)
    _seed_candidate(client, headers)
    _compute_preview(client, headers)

    response = client.post(
        "/api/v1/scanner/candidates/NVDA/visual-review",
        headers=headers,
        json={"source": "ma_pipeline", "decision": decision},
    )
    assert response.status_code == 422


def test_practical_rejection_binds_to_correct_setup_key(client, headers, monkeypatch):
    _mock_network(monkeypatch, _promotion_daily_frame)
    _seed_candidate(client, headers)
    preview = _compute_preview(client, headers)

    response = client.post(
        "/api/v1/scanner/candidates/NVDA/visual-review",
        headers=headers,
        json={"source": "ma_pipeline", "decision": "reject", "practical_rejection_reason": "poor_option_liquidity"},
    )
    body = response.json()

    conn_check = client.get("/api/v1/scanner/candidates", headers=headers).json()
    candidate_row = next(c for c in conn_check if c["ticker"] == "NVDA")
    expected_key = router._compute_setup_key(candidate_row, preview)
    assert body["setup_key"] == expected_key

    # And it shows up as the current review for that setup in the queue.
    queue = client.get("/api/v1/scanner/candidates/review-queue", headers=headers).json()
    entry = next((c for c in queue["candidates"] if c["ticker"] == "NVDA"), None)
    if entry is not None:  # only if NVDA is stage-1-ready in this fixture
        assert entry["setup_key"] == expected_key
        assert entry["current_review"]["review_type"] == "practical_rejection"
        assert entry["current_review"]["practical_rejection_reason"] == "poor_option_liquidity"


def test_practical_rejection_does_not_change_queue_qualification_or_ranking(client, headers, monkeypatch):
    _mock_network(monkeypatch, _promotion_daily_frame)
    _seed_candidate(client, headers)

    before = client.get("/api/v1/scanner/candidates/review-queue", headers=headers).json()

    client.post(
        "/api/v1/scanner/candidates/NVDA/visual-review",
        headers=headers,
        json={"source": "ma_pipeline", "decision": "reject", "practical_rejection_reason": "options_too_expensive"},
    )

    after = client.get("/api/v1/scanner/candidates/review-queue", headers=headers).json()
    assert before["count"] == after["count"]
    before_order = [(c["rank"], c["ticker"]) for c in before["candidates"]]
    after_order = [(c["rank"], c["ticker"]) for c in after["candidates"]]
    assert before_order == after_order


def test_existing_visual_reviews_remain_readable_after_migration(client, headers):
    """Simulates the real already-deployed pre-migration schema (NOT NULL
    visual columns, no review_type/practical_rejection_reason) with a row
    shaped exactly like the real 2026-08-31 IGV production verification
    review, then confirms schema init upgrades it correctly and the row
    stays fully readable with its original values plus review_type
    defaulted to "visual" -- not lost, not blanked, not miscategorized."""
    db_path = os.environ["KAIROS_CANDIDATES_DB"]
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE candidate_visual_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                source TEXT NOT NULL,
                setup_key TEXT NOT NULL,
                market_structure TEXT NOT NULL,
                location_read TEXT NOT NULL,
                clear_path_to_target TEXT NOT NULL,
                lower_tf_confirmation TEXT NOT NULL,
                decision TEXT NOT NULL,
                note TEXT,
                reviewed_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO candidate_visual_reviews
                (ticker, source, setup_key, market_structure, location_read,
                 clear_path_to_target, lower_tf_confirmation, decision, note, reviewed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "IGV", "ma_pipeline", "IGV|ma_pipeline|long|107.95|117.02",
                "bullish", "good", "yes", "yes", "approve",
                "Deploy verification — real live submission, 2026-08-31",
                "2026-08-31T20:21:03.468053+00:00",
            ),
        )
        conn.commit()
    finally:
        conn.close()

    # Force a fresh schema check against this pre-migration DB (mirrors
    # what a real server boot against the real production DB does).
    router._schema_ready_db_paths.discard(str(Path(db_path)))

    listed = client.get("/api/v1/scanner/candidate-visual-reviews", headers=headers).json()
    igv_rows = [r for r in listed if r["ticker"] == "IGV"]
    assert len(igv_rows) == 1
    row = igv_rows[0]
    assert row["setup_key"] == "IGV|ma_pipeline|long|107.95|117.02"
    assert row["market_structure"] == "bullish"
    assert row["location_read"] == "good"
    assert row["clear_path_to_target"] == "yes"
    assert row["lower_tf_confirmation"] == "yes"
    assert row["decision"] == "approve"
    assert row["note"] == "Deploy verification — real live submission, 2026-08-31"
    assert row["review_type"] == "visual"  # correctly backfilled, not left null/blank
    assert row["practical_rejection_reason"] is None


@pytest.mark.parametrize("field,bad_value", [
    ("market_structure", "sideways"),
    ("location_read", "great"),
    ("clear_path_to_target", "maybe"),
    ("lower_tf_confirmation", "no"),
    ("decision", "maybe_later"),
])
def test_visual_review_rejects_invalid_enum_values(client, headers, monkeypatch, field, bad_value):
    _mock_network(monkeypatch, _promotion_daily_frame)
    _seed_candidate(client, headers)
    _compute_preview(client, headers)

    payload = {
        "source": "ma_pipeline", "market_structure": "bullish", "location_read": "good",
        "clear_path_to_target": "yes", "lower_tf_confirmation": "yes", "confirmation_rule": "close_above", "confirmation_level": 100.0, "decision": "approve",
    }
    payload[field] = bad_value
    response = client.post("/api/v1/scanner/candidates/NVDA/visual-review", headers=headers, json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize("decision", ["approve", "watch", "reject"])
def test_visual_review_accepts_every_real_decision(client, headers, monkeypatch, decision):
    _mock_network(monkeypatch, _promotion_daily_frame)
    _seed_candidate(client, headers)
    _compute_preview(client, headers)

    response = client.post(
        "/api/v1/scanner/candidates/NVDA/visual-review",
        headers=headers,
        json={
            "source": "ma_pipeline", "market_structure": "range", "location_read": "neutral",
            "clear_path_to_target": "no", "lower_tf_confirmation": "not_yet", "decision": decision,
        },
    )
    assert response.status_code == 200
    assert response.json()["decision"] == decision


def test_visual_review_is_append_only_not_overwritten(client, headers, monkeypatch):
    _mock_network(monkeypatch, _promotion_daily_frame)
    _seed_candidate(client, headers)
    _compute_preview(client, headers)

    first = client.post(
        "/api/v1/scanner/candidates/NVDA/visual-review", headers=headers,
        json={
            "source": "ma_pipeline", "market_structure": "bearish", "location_read": "bad",
            "clear_path_to_target": "no", "lower_tf_confirmation": "not_yet", "decision": "reject",
        },
    )
    second = client.post(
        "/api/v1/scanner/candidates/NVDA/visual-review", headers=headers,
        json={
            "source": "ma_pipeline", "market_structure": "bullish", "location_read": "good",
            "clear_path_to_target": "yes", "lower_tf_confirmation": "yes", "confirmation_rule": "close_above", "confirmation_level": 100.0, "decision": "approve",
        },
    )
    assert first.json()["id"] != second.json()["id"]

    listed = client.get("/api/v1/scanner/candidate-visual-reviews", headers=headers).json()
    nvda_reviews = [r for r in listed if r["ticker"] == "NVDA"]
    assert len(nvda_reviews) == 2
    decisions = {r["decision"] for r in nvda_reviews}
    assert decisions == {"reject", "approve"}
    # Newest first, and the earlier reject is still there, not overwritten.
    assert listed[0]["decision"] == "approve"
    assert listed[1]["decision"] == "reject"


def test_visual_review_list_requires_auth(client):
    response = client.get("/api/v1/scanner/candidate-visual-reviews")
    assert response.status_code == 401


def test_same_setup_reviewed_twice_shares_setup_key(client, headers, monkeypatch):
    _mock_network(monkeypatch, _promotion_daily_frame)
    _seed_candidate(client, headers)
    _compute_preview(client, headers)

    first = client.post(
        "/api/v1/scanner/candidates/NVDA/visual-review", headers=headers,
        json={
            "source": "ma_pipeline", "market_structure": "bullish", "location_read": "neutral",
            "clear_path_to_target": "yes", "lower_tf_confirmation": "not_yet", "decision": "watch",
        },
    ).json()
    second = client.post(
        "/api/v1/scanner/candidates/NVDA/visual-review", headers=headers,
        json={
            "source": "ma_pipeline", "market_structure": "bullish", "location_read": "good",
            "clear_path_to_target": "yes", "lower_tf_confirmation": "yes", "confirmation_rule": "close_above", "confirmation_level": 100.0, "decision": "approve",
        },
    ).json()
    assert first["setup_key"] == second["setup_key"]

    reviews = client.get("/api/v1/scanner/candidate-visual-reviews", headers=headers).json()
    same_setup = [r for r in reviews if r["setup_key"] == first["setup_key"]]
    assert len(same_setup) == 2
    # Current state for this setup_key is the latest row: approve, not watch.
    current = max(same_setup, key=lambda r: r["reviewed_at"])
    assert current["decision"] == "approve"


def test_ordinary_refresh_does_not_create_a_new_generation(client, headers, monkeypatch):
    """Re-ingesting the SAME structural setup with a different entry_price
    (exactly what a routine rescan does -- see the candidate_updated_at
    investigation) must NOT change setup_key, and must NOT strand the
    existing review under an orphaned key."""
    _mock_network(monkeypatch, _promotion_daily_frame)
    _seed_candidate(client, headers, entry_price=100.0, scanned_at="2026-08-20T14:30:00Z")
    _compute_preview(client, headers)

    reviewed = client.post(
        "/api/v1/scanner/candidates/NVDA/visual-review", headers=headers,
        json={
            "source": "ma_pipeline", "market_structure": "bullish", "location_read": "good",
            "clear_path_to_target": "yes", "lower_tf_confirmation": "yes", "decision": "reject",
        },
    ).json()
    original_key = reviewed["setup_key"]

    # Ordinary refresh: same daily structure, a different live entry_price,
    # a later scan timestamp -- exactly what the next ma_pipeline scan cycle
    # does, per the investigation.
    _seed_candidate(client, headers, entry_price=101.35, scanned_at="2026-08-20T18:00:00Z")
    refreshed_preview = _compute_preview(client, headers)

    conn_check = client.get("/api/v1/scanner/candidates", headers=headers).json()
    candidate_row = next(c for c in conn_check if c["ticker"] == "NVDA")
    new_key = router._compute_setup_key(candidate_row, refreshed_preview)

    assert new_key == original_key

    reviews = client.get("/api/v1/scanner/candidate-visual-reviews", headers=headers).json()
    matching = [r for r in reviews if r["setup_key"] == new_key]
    assert len(matching) == 1
    assert matching[0]["decision"] == "reject"


def test_materially_new_setup_does_not_inherit_a_previous_review(client, headers, monkeypatch):
    """A genuinely different structural setup for the same ticker (real
    stop/target move) must start with zero reviews -- Needs Review, never
    silently inheriting the prior setup's decision."""
    _mock_network(monkeypatch, _promotion_daily_frame)
    _seed_candidate(client, headers)
    original_preview = _compute_preview(client, headers)

    rejected = client.post(
        "/api/v1/scanner/candidates/NVDA/visual-review", headers=headers,
        json={
            "source": "ma_pipeline", "market_structure": "bearish", "location_read": "bad",
            "clear_path_to_target": "no", "lower_tf_confirmation": "not_yet", "decision": "reject",
        },
    ).json()
    original_key = rejected["setup_key"]

    # Swap in a materially different daily structure -- a real new setup,
    # not routine drift -- and force recomputation (candidate_updated_at
    # bumps here too, but that's irrelevant to setup_key by design).
    _mock_network(monkeypatch, _alternate_daily_frame)
    _seed_candidate(client, headers, entry_price=100.0, scanned_at="2026-08-21T14:30:00Z")
    new_preview = _compute_preview(client, headers)

    conn_check = client.get("/api/v1/scanner/candidates", headers=headers).json()
    candidate_row = next(c for c in conn_check if c["ticker"] == "NVDA")
    new_key = router._compute_setup_key(candidate_row, new_preview)

    assert new_key != original_key

    reviews = client.get("/api/v1/scanner/candidate-visual-reviews", headers=headers).json()
    matching_new = [r for r in reviews if r["setup_key"] == new_key]
    matching_old = [r for r in reviews if r["setup_key"] == original_key]
    assert matching_new == []  # Needs Review -- no inherited decision
    assert len(matching_old) == 1
    assert matching_old[0]["decision"] == "reject"  # old review still there, untouched


# ---------------------------------------------------------------------------
# candidate_ranking_snapshots
# ---------------------------------------------------------------------------

def test_ranked_call_writes_a_ranking_snapshot(client, headers, monkeypatch):
    _mock_network(monkeypatch, _promotion_daily_frame)
    _seed_candidate(client, headers)

    response = client.get("/api/v1/scanner/candidates/ranked", headers=headers)
    assert response.status_code == 200
    snapshot_id = response.json()["snapshot_id"]
    assert snapshot_id

    snapshots = client.get("/api/v1/scanner/candidate-ranking-snapshots", headers=headers).json()
    rows = [row for row in snapshots if row["snapshot_id"] == snapshot_id]
    assert len(rows) == 1
    assert rows[0]["ticker"] == "NVDA"
    assert rows[0]["rank"] == 1
    assert rows[0]["mechanism"] == router.RANKING_MECHANISM_VERSION
    assert rows[0]["computed_at"]


def test_each_ranked_call_writes_a_distinct_snapshot(client, headers, monkeypatch):
    _mock_network(monkeypatch, _promotion_daily_frame)
    _seed_candidate(client, headers)

    first = client.get("/api/v1/scanner/candidates/ranked", headers=headers).json()
    second = client.get("/api/v1/scanner/candidates/ranked", headers=headers).json()
    assert first["snapshot_id"] != second["snapshot_id"]

    all_snapshots = client.get("/api/v1/scanner/candidate-ranking-snapshots", headers=headers).json()
    distinct_ids = {row["snapshot_id"] for row in all_snapshots}
    assert {first["snapshot_id"], second["snapshot_id"]} <= distinct_ids


def test_ranking_snapshots_filterable_by_snapshot_id(client, headers, monkeypatch):
    _mock_network(monkeypatch, _promotion_daily_frame)
    _seed_candidate(client, headers)

    result = client.get("/api/v1/scanner/candidates/ranked", headers=headers).json()
    snapshot_id = result["snapshot_id"]

    filtered = client.get(
        "/api/v1/scanner/candidate-ranking-snapshots",
        params={"snapshot_id": snapshot_id},
        headers=headers,
    ).json()
    assert len(filtered) == 1
    assert filtered[0]["snapshot_id"] == snapshot_id


def test_ranking_snapshots_list_requires_auth(client):
    response = client.get("/api/v1/scanner/candidate-ranking-snapshots")
    assert response.status_code == 401
