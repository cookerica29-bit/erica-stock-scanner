#!/usr/bin/env python3
"""Tests for manual discovery-universe cache endpoints."""

from __future__ import annotations

import os
import sys
import tempfile
import json
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main  # noqa: E402
import discovery  # noqa: E402
from discovery import DollarVolumeMetrics, OptionsLiquidityMetrics, discovery_universe_max_symbols, discovery_universe_max_symbols_resolution, rank_discovery_candidates  # noqa: E402


def reset_discovery_cache():
    with main._discovery_universe_lock:
        main._discovery_universe_cache.update(main._discovery_cache_defaults())


def reset_handoff_state():
    with main._discovered_scan_handoff_lock:
        main._discovered_scan_handoff_state.update({
            "handoff_triggered_at": None,
            "handoff_reason": None,
            "refresh_job_id": None,
            "refresh_started_at": None,
            "refresh_completed_at": None,
            "refresh_failed_at": None,
            "refresh_attempt_count": 0,
            "stale_job_recovered": False,
            "scanner_cache_generation": None,
            "last_checked_at": None,
            "last_result": None,
        })


class InlineExecutor:
    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)

        class Done:
            pass

        return Done()


def fake_discovery_result(static_watchlist=None):
    cap_resolution = {
        "resolved_value": 1000,
        "source": "KAIROS_INTAKE_CAP",
        "env_var_used": "KAIROS_INTAKE_CAP",
        "used_default": False,
        "default_value": 1000,
        "warning": None,
    }
    return {
        "symbols": ["AAPL", "MSFT", "F"],
        "source": "alpaca",
        "ranking_version": "alpaca-liquidity-ranking-v1",
        "pipeline_counts": {
            "raw_assets": 10,
            "tradable_optionable": 8,
            "hygiene_passed": 7,
            "dollar_volume_passed": 6,
            "options_liquidity_passed": 5,
            "ranked": 5,
            "selected": 3,
        },
        "thresholds": {
            "target_universe_size": 1000,
            "kairos_intake_cap": 1000,
            "kairos_intake_cap_resolution": cap_resolution,
        },
        "formula": {"combined_liquidity_score": "test"},
        "stage3": {"elapsed_seconds": 1.2},
        "stage4": {"elapsed_seconds": 2.3},
        "rejection_evidence": {
            "stage3_dollar_volume": [
                {
                    "ticker": "LOWDV",
                    "symbol": "LOWDV",
                    "stage": "dollar_volume",
                    "rejection_reason": "low dollar volume",
                    "measured_values": {
                        "latest_close": 10,
                        "average_daily_volume": 1000,
                        "average_daily_dollar_volume": 10000,
                        "valid_daily_bars": 30,
                    },
                    "thresholds_used": {"average_daily_dollar_volume_floor": 100_000_000},
                }
            ],
            "stage4_options_liquidity": [
                {
                    "ticker": "THINOI",
                    "symbol": "THINOI",
                    "stage": "options_liquidity",
                    "rejection_reason": "thin call open interest",
                    "measured_values": {
                        "latest_close": 25,
                        "near_atm_call_open_interest": 12,
                        "near_atm_put_open_interest": 180,
                        "near_atm_contracts_checked": 8,
                        "pages_fetched": 1,
                    },
                    "thresholds_used": {"minimum_call_open_interest": 100, "minimum_put_open_interest": 100},
                }
            ],
        },
        "top_20": [{"symbol": "AAPL", "rank": 1}],
        "bottom_20_selected": [{"symbol": "F", "rank": 3}],
        "watchlist_overlap": {
            "watchlist_count": len(static_watchlist or []),
            "overlap": 3,
            "missing": [],
        },
        "elapsed_seconds": 3.5,
    }


def assert_utc_z_timestamp(value):
    assert isinstance(value, str)
    assert value.endswith("Z")
    parsed = __import__("datetime").datetime.fromisoformat(value.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None


def test_discovery_status_warming_without_cache():
    previous_token = os.environ.pop("DISCOVERY_ADMIN_TOKEN", None)
    reset_discovery_cache()
    try:
        client = TestClient(main.app)
        response = client.get("/api/discovery/status")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "warming"
        assert payload["enabled"] is False
        assert payload["has_cache"] is False
        assert payload["selected_count"] == 0
        symbols = client.get("/api/discovery/symbols").json()
        assert symbols["symbols"] == []
        assert symbols["count"] == 0
        assert symbols["status"]["status"] == "warming"
    finally:
        if previous_token is not None:
            os.environ["DISCOVERY_ADMIN_TOKEN"] = previous_token


def test_discovery_run_disabled_when_token_unset():
    previous_token = os.environ.pop("DISCOVERY_ADMIN_TOKEN", None)
    reset_discovery_cache()
    try:
        client = TestClient(main.app)
        response = client.post("/api/discovery/run")
        assert response.status_code == 503
        assert "disabled" in response.json()["detail"]
    finally:
        if previous_token is not None:
            os.environ["DISCOVERY_ADMIN_TOKEN"] = previous_token


def test_discovery_run_rejects_wrong_token():
    previous_token = os.environ.get("DISCOVERY_ADMIN_TOKEN")
    os.environ["DISCOVERY_ADMIN_TOKEN"] = "secret"
    reset_discovery_cache()
    try:
        client = TestClient(main.app)
        response = client.post("/api/discovery/run", headers={"X-Kairos-Admin-Token": "wrong"})
        assert response.status_code == 403
        assert "Invalid" in response.json()["detail"]
    finally:
        if previous_token is None:
            os.environ.pop("DISCOVERY_ADMIN_TOKEN", None)
        else:
            os.environ["DISCOVERY_ADMIN_TOKEN"] = previous_token


def test_discovery_run_populates_cache_with_valid_token():
    previous_token = os.environ.get("DISCOVERY_ADMIN_TOKEN")
    previous_path = os.environ.get(main.DISCOVERY_POOL_PATH_ENV)
    previous_executor = main._discovery_universe_executor
    previous_builder = main.build_ranked_discovery_universe
    previous_handoff = main._maybe_enqueue_discovered_scan_handoff
    os.environ["DISCOVERY_ADMIN_TOKEN"] = "secret"
    main._discovery_universe_executor = InlineExecutor()
    main.build_ranked_discovery_universe = fake_discovery_result
    handoff_calls = []
    main._maybe_enqueue_discovered_scan_handoff = lambda reason="": handoff_calls.append(reason) or (False, "stubbed")
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ[main.DISCOVERY_POOL_PATH_ENV] = str(Path(temp_dir) / "pool.json")
        reset_discovery_cache()
        try:
            client = TestClient(main.app)
            response = client.post("/api/discovery/run?refresh=true", headers={"X-Kairos-Admin-Token": "secret"})
            assert response.status_code == 200
            payload = response.json()
            assert payload["accepted"] is True
            assert payload["status"]["status"] == "ready"
            assert payload["status"]["selected_count"] == 3
            assert payload["status"]["pipeline_counts"]["selected"] == 3
            assert payload["status"]["metrics"]["final_admitted_symbol_count"] == 3
            assert payload["status"]["metrics"]["effective_cap"] == 1000
            assert payload["status"]["kairos_intake_cap"] == 1000
            assert payload["status"]["kairos_intake_cap_resolution"]["source"] == "KAIROS_INTAKE_CAP"
            assert payload["status"]["kairos_intake_cap_warning"] is None
            assert_utc_z_timestamp(payload["status"]["metrics"]["discovery_started_at"])
            assert_utc_z_timestamp(payload["status"]["metrics"]["discovery_completed_at"])
            assert payload["status"]["last_error"] is None

            symbols = client.get("/api/discovery/symbols").json()
            assert symbols["symbols"] == ["AAPL", "MSFT", "F"]
            assert symbols["count"] == 3
            assert symbols["status"]["has_cache"] is True

            status = client.get("/api/discovery/status").json()
            assert status["enabled"] is True
            assert status["status"] == "ready"
            assert status["stale"] is False
            assert status["generated_at"]
            assert status["expires_at"]
            assert handoff_calls == ["discovery_completed_no_scanner_cache"]
        finally:
            main._discovery_universe_executor = previous_executor
            main.build_ranked_discovery_universe = previous_builder
            main._maybe_enqueue_discovered_scan_handoff = previous_handoff
            reset_discovery_cache()
            if previous_token is None:
                os.environ.pop("DISCOVERY_ADMIN_TOKEN", None)
            else:
                os.environ["DISCOVERY_ADMIN_TOKEN"] = previous_token
            if previous_path is None:
                os.environ.pop(main.DISCOVERY_POOL_PATH_ENV, None)
            else:
                os.environ[main.DISCOVERY_POOL_PATH_ENV] = previous_path


def test_discovery_pool_persists_and_hydrates_after_restart():
    previous_path = os.environ.get(main.DISCOVERY_POOL_PATH_ENV)
    previous_executor = main._discovery_universe_executor
    previous_builder = main.build_ranked_discovery_universe
    previous_handoff = main._maybe_enqueue_discovered_scan_handoff
    main._discovery_universe_executor = InlineExecutor()
    main.build_ranked_discovery_universe = fake_discovery_result
    main._maybe_enqueue_discovered_scan_handoff = lambda reason="": (False, "stubbed")
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ[main.DISCOVERY_POOL_PATH_ENV] = str(Path(temp_dir) / "pool.json")
        reset_discovery_cache()
        try:
            accepted, job_id = main._submit_discovery_universe_job(force=True)
            assert accepted is True
            assert job_id.startswith("discovery:")
            assert Path(os.environ[main.DISCOVERY_POOL_PATH_ENV]).exists()

            reset_discovery_cache()
            loaded = main._load_discovery_pool_from_disk()
            assert loaded is True
            status = main._discovery_status_snapshot()
            assert status["status"] == "ready"
            assert status["loaded_from_disk"] is True
            assert status["selected_count"] == 3
            assert status["source"] == "alpaca"
            assert status["kairos_intake_cap"] == 1000
            assert status["kairos_intake_cap_resolution"]["resolved_value"] == 1000
            assert status["next_scheduled_refresh"] == status["expires_at"]
        finally:
            main._discovery_universe_executor = previous_executor
            main.build_ranked_discovery_universe = previous_builder
            main._maybe_enqueue_discovered_scan_handoff = previous_handoff
            reset_discovery_cache()
            if previous_path is None:
                os.environ.pop(main.DISCOVERY_POOL_PATH_ENV, None)
            else:
                os.environ[main.DISCOVERY_POOL_PATH_ENV] = previous_path


def test_legacy_persisted_pool_exposes_live_cap_resolution_metadata():
    previous_new = os.environ.get("KAIROS_INTAKE_CAP")
    previous_old = os.environ.get("DISCOVERY_UNIVERSE_MAX_SYMBOLS")
    try:
        os.environ["KAIROS_INTAKE_CAP"] = "1000"
        os.environ.pop("DISCOVERY_UNIVERSE_MAX_SYMBOLS", None)
        reset_discovery_cache()
        with main._discovery_universe_lock:
            main._discovery_universe_cache.update({
                **main._discovery_cache_defaults(),
                "symbols": ["AAPL"],
                "generated_at": main._utc_now(),
                "expires_at": main._utc_now() + __import__("datetime").timedelta(days=7),
                "thresholds": {"kairos_intake_cap": 1000, "target_universe_size": 1000},
                "loaded_from_disk": True,
            })
        status = main._discovery_status_snapshot()
        assert status["kairos_intake_cap"] == 1000
        assert status["kairos_intake_cap_resolution"]["resolved_value"] == 1000
        assert status["kairos_intake_cap_resolution"]["source"] == "KAIROS_INTAKE_CAP"
        assert status["kairos_intake_cap_resolution"]["used_default"] is False
        assert status["kairos_intake_cap_warning"] is None
    finally:
        reset_discovery_cache()
        if previous_new is None:
            os.environ.pop("KAIROS_INTAKE_CAP", None)
        else:
            os.environ["KAIROS_INTAKE_CAP"] = previous_new
        if previous_old is None:
            os.environ.pop("DISCOVERY_UNIVERSE_MAX_SYMBOLS", None)
        else:
            os.environ["DISCOVERY_UNIVERSE_MAX_SYMBOLS"] = previous_old


def test_successful_discovery_persists_rejection_evidence():
    previous_pool_path = os.environ.get(main.DISCOVERY_POOL_PATH_ENV)
    previous_evidence_path = os.environ.get(main.DISCOVERY_REJECTION_EVIDENCE_PATH_ENV)
    previous_executor = main._discovery_universe_executor
    previous_builder = main.build_ranked_discovery_universe
    previous_handoff = main._maybe_enqueue_discovered_scan_handoff
    main._discovery_universe_executor = InlineExecutor()
    main.build_ranked_discovery_universe = fake_discovery_result
    main._maybe_enqueue_discovered_scan_handoff = lambda reason="": (False, "stubbed")
    with tempfile.TemporaryDirectory() as temp_dir:
        evidence_path = Path(temp_dir) / "evidence.json"
        os.environ[main.DISCOVERY_POOL_PATH_ENV] = str(Path(temp_dir) / "pool.json")
        os.environ[main.DISCOVERY_REJECTION_EVIDENCE_PATH_ENV] = str(evidence_path)
        reset_discovery_cache()
        try:
            accepted, _job_id = main._submit_discovery_universe_job(force=True)
            assert accepted is True
            assert evidence_path.exists()
            payload = json.loads(evidence_path.read_text())
            assert payload["counts"]["stage3_dollar_volume"] == 1
            assert payload["counts"]["stage4_options_liquidity"] == 1
            assert payload["counts"]["total"] == 2
            assert payload["stage3_dollar_volume"][0]["ticker"] == "LOWDV"
            assert payload["stage3_dollar_volume"][0]["discovery_generated_at"]
            assert payload["stage4_options_liquidity"][0]["measured_values"]["near_atm_call_open_interest"] == 12
            status = main._discovery_status_snapshot()
            assert status["rejection_evidence_path"] == str(evidence_path)
            assert status["rejection_evidence_counts"]["total"] == 2
            assert status["rejection_evidence_persisted_at"]
        finally:
            main._discovery_universe_executor = previous_executor
            main.build_ranked_discovery_universe = previous_builder
            main._maybe_enqueue_discovered_scan_handoff = previous_handoff
            reset_discovery_cache()
            if previous_pool_path is None:
                os.environ.pop(main.DISCOVERY_POOL_PATH_ENV, None)
            else:
                os.environ[main.DISCOVERY_POOL_PATH_ENV] = previous_pool_path
            if previous_evidence_path is None:
                os.environ.pop(main.DISCOVERY_REJECTION_EVIDENCE_PATH_ENV, None)
            else:
                os.environ[main.DISCOVERY_REJECTION_EVIDENCE_PATH_ENV] = previous_evidence_path


def test_failed_discovery_does_not_overwrite_last_good_rejection_evidence():
    previous_pool_path = os.environ.get(main.DISCOVERY_POOL_PATH_ENV)
    previous_evidence_path = os.environ.get(main.DISCOVERY_REJECTION_EVIDENCE_PATH_ENV)
    previous_executor = main._discovery_universe_executor
    previous_builder = main.build_ranked_discovery_universe
    previous_handoff = main._maybe_enqueue_discovered_scan_handoff
    main._discovery_universe_executor = InlineExecutor()
    main._maybe_enqueue_discovered_scan_handoff = lambda reason="": (False, "stubbed")
    with tempfile.TemporaryDirectory() as temp_dir:
        evidence_path = Path(temp_dir) / "evidence.json"
        os.environ[main.DISCOVERY_POOL_PATH_ENV] = str(Path(temp_dir) / "pool.json")
        os.environ[main.DISCOVERY_REJECTION_EVIDENCE_PATH_ENV] = str(evidence_path)
        reset_discovery_cache()
        try:
            main.build_ranked_discovery_universe = fake_discovery_result
            assert main._submit_discovery_universe_job(force=True)[0] is True
            before = evidence_path.read_text()

            def failing_discovery(*_args, **_kwargs):
                raise RuntimeError("provider failure")

            main.build_ranked_discovery_universe = failing_discovery
            assert main._submit_discovery_universe_job(force=True)[0] is True
            assert evidence_path.read_text() == before
        finally:
            main._discovery_universe_executor = previous_executor
            main.build_ranked_discovery_universe = previous_builder
            main._maybe_enqueue_discovered_scan_handoff = previous_handoff
            reset_discovery_cache()
            if previous_pool_path is None:
                os.environ.pop(main.DISCOVERY_POOL_PATH_ENV, None)
            else:
                os.environ[main.DISCOVERY_POOL_PATH_ENV] = previous_pool_path
            if previous_evidence_path is None:
                os.environ.pop(main.DISCOVERY_REJECTION_EVIDENCE_PATH_ENV, None)
            else:
                os.environ[main.DISCOVERY_REJECTION_EVIDENCE_PATH_ENV] = previous_evidence_path


def test_valid_persisted_pool_prevents_startup_rediscovery():
    previous_path = os.environ.get(main.DISCOVERY_POOL_PATH_ENV)
    previous_executor = main._discovery_universe_executor
    previous_builder = main.build_ranked_discovery_universe
    previous_handoff = main._maybe_enqueue_discovered_scan_handoff
    calls = []
    main._discovery_universe_executor = InlineExecutor()
    main.build_ranked_discovery_universe = lambda *args, **kwargs: calls.append("builder") or fake_discovery_result()
    main._maybe_enqueue_discovered_scan_handoff = lambda reason="": (False, "stubbed")
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ[main.DISCOVERY_POOL_PATH_ENV] = str(Path(temp_dir) / "pool.json")
        reset_discovery_cache()
        try:
            assert main._submit_discovery_universe_job(force=True)[0] is True
            assert calls == ["builder"]
            reset_discovery_cache()
            assert main._load_discovery_pool_from_disk() is True
            accepted, reason = main._submit_discovery_universe_job_if_needed()
            assert accepted is False
            assert reason == "cache fresh"
            assert calls == ["builder"]
        finally:
            main._discovery_universe_executor = previous_executor
            main.build_ranked_discovery_universe = previous_builder
            main._maybe_enqueue_discovered_scan_handoff = previous_handoff
            reset_discovery_cache()
            if previous_path is None:
                os.environ.pop(main.DISCOVERY_POOL_PATH_ENV, None)
            else:
                os.environ[main.DISCOVERY_POOL_PATH_ENV] = previous_path


def test_weekly_expired_persisted_pool_triggers_discovery():
    previous_path = os.environ.get(main.DISCOVERY_POOL_PATH_ENV)
    previous_executor = main._discovery_universe_executor
    previous_builder = main.build_ranked_discovery_universe
    previous_handoff = main._maybe_enqueue_discovered_scan_handoff
    calls = []
    main._discovery_universe_executor = InlineExecutor()
    main.build_ranked_discovery_universe = lambda *args, **kwargs: calls.append("builder") or fake_discovery_result()
    main._maybe_enqueue_discovered_scan_handoff = lambda reason="": (False, "stubbed")
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "pool.json"
        os.environ[main.DISCOVERY_POOL_PATH_ENV] = str(path)
        now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
        payload = main._discovery_pool_payload_from_cache({
            **main._discovery_cache_defaults(),
            "symbols": ["OLD"],
            "generated_at": now - __import__("datetime").timedelta(days=8),
            "expires_at": now - __import__("datetime").timedelta(hours=1),
            "pipeline_counts": {"selected": 1},
            "thresholds": {"kairos_intake_cap": 1000, "target_universe_size": 1000},
            "formula": {"combined_liquidity_score": "test"},
        })
        path.write_text(json.dumps(payload))
        reset_discovery_cache()
        try:
            assert main._load_discovery_pool_from_disk() is True
            assert main._discovery_cache_needs_refresh() is True
            accepted, job_id = main._submit_discovery_universe_job_if_needed()
            assert accepted is True
            assert job_id.startswith("discovery:")
            assert calls == ["builder"]
            assert main._discovery_status_snapshot()["selected_count"] == 3
        finally:
            main._discovery_universe_executor = previous_executor
            main.build_ranked_discovery_universe = previous_builder
            main._maybe_enqueue_discovered_scan_handoff = previous_handoff
            reset_discovery_cache()
            if previous_path is None:
                os.environ.pop(main.DISCOVERY_POOL_PATH_ENV, None)
            else:
                os.environ[main.DISCOVERY_POOL_PATH_ENV] = previous_path


def test_corrupt_or_missing_persisted_pool_safely_rebuilds():
    previous_path = os.environ.get(main.DISCOVERY_POOL_PATH_ENV)
    previous_executor = main._discovery_universe_executor
    previous_builder = main.build_ranked_discovery_universe
    previous_handoff = main._maybe_enqueue_discovered_scan_handoff
    calls = []
    main._discovery_universe_executor = InlineExecutor()
    main.build_ranked_discovery_universe = lambda *args, **kwargs: calls.append("builder") or fake_discovery_result()
    main._maybe_enqueue_discovered_scan_handoff = lambda reason="": (False, "stubbed")
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "pool.json"
        os.environ[main.DISCOVERY_POOL_PATH_ENV] = str(path)
        try:
            reset_discovery_cache()
            assert main._load_discovery_pool_from_disk() is False
            accepted, _ = main._submit_discovery_universe_job_if_needed()
            assert accepted is True
            assert calls == ["builder"]

            path.write_text("{not-json")
            reset_discovery_cache()
            assert main._load_discovery_pool_from_disk() is False
            accepted, _ = main._submit_discovery_universe_job_if_needed()
            assert accepted is True
            assert calls == ["builder", "builder"]
        finally:
            main._discovery_universe_executor = previous_executor
            main.build_ranked_discovery_universe = previous_builder
            main._maybe_enqueue_discovered_scan_handoff = previous_handoff
            reset_discovery_cache()
            if previous_path is None:
                os.environ.pop(main.DISCOVERY_POOL_PATH_ENV, None)
            else:
                os.environ[main.DISCOVERY_POOL_PATH_ENV] = previous_path


def test_scanner_analyzes_persisted_pool_normally():
    previous_path = os.environ.get(main.DISCOVERY_POOL_PATH_ENV)
    previous_scan_cached = main.scan_cached
    calls = []
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "pool.json"
        os.environ[main.DISCOVERY_POOL_PATH_ENV] = str(path)
        now = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
        payload = main._discovery_pool_payload_from_cache({
            **main._discovery_cache_defaults(),
            "symbols": ["AAPL", "MSFT", "F"],
            "generated_at": now,
            "expires_at": now + __import__("datetime").timedelta(days=7),
            "pipeline_counts": {"selected": 3},
            "thresholds": {"kairos_intake_cap": 1000, "target_universe_size": 1000},
            "formula": {"combined_liquidity_score": "test"},
        })
        path.write_text(json.dumps(payload))
        reset_discovery_cache()

        def fake_scan_cached(watchlist=None, **kwargs):
            calls.append((watchlist, kwargs))
            return {"rows": [], "near_miss": [], "meta": {"configured_universe_count": len(watchlist or []), "cache_key": "discovered"}}

        main.scan_cached = fake_scan_cached
        try:
            assert main._load_discovery_pool_from_disk() is True
            client = TestClient(main.app)
            response = client.get("/api/scan?universe=discovered")
            assert response.status_code == 200
            assert calls[0][0] == ["AAPL", "MSFT", "F"]
            assert calls[0][1]["universe"] == "discovered"
            assert calls[0][1]["max_symbols"] is None
            assert calls[0][1]["trusted_options_symbols"] == {"AAPL", "MSFT", "F"}
        finally:
            main.scan_cached = previous_scan_cached
            reset_discovery_cache()
            if previous_path is None:
                os.environ.pop(main.DISCOVERY_POOL_PATH_ENV, None)
            else:
                os.environ[main.DISCOVERY_POOL_PATH_ENV] = previous_path


def test_intake_cap_only_truncates_when_eligible_candidates_exceed_cap():
    dollars = [
        DollarVolumeMetrics("AAA", 10, 1_000_000, 100_000_000, 30, True),
        DollarVolumeMetrics("BBB", 20, 1_000_000, 200_000_000, 30, True),
        DollarVolumeMetrics("CCC", 30, 1_000_000, 300_000_000, 30, True),
    ]
    options = [
        OptionsLiquidityMetrics("AAA", 10, 100, 100, "AAAC", "AAAP", 2, 1, True),
        OptionsLiquidityMetrics("BBB", 20, 200, 200, "BBBC", "BBBP", 2, 1, True),
        OptionsLiquidityMetrics("CCC", 30, 300, 300, "CCCC", "CCCP", 2, 1, True),
    ]
    under_cap = rank_discovery_candidates(dollars, options, target_size=5)
    at_cap = rank_discovery_candidates(dollars, options, target_size=3)
    over_cap = rank_discovery_candidates(dollars, options, target_size=2)
    assert sum(1 for candidate in under_cap if candidate.selected) == 3
    assert sum(1 for candidate in at_cap if candidate.selected) == 3
    assert sum(1 for candidate in over_cap if candidate.selected) == 2


def test_discovery_selected_symbols_unchanged_for_identical_inputs():
    previous_client = discovery.AlpacaAssetDiscoveryClient
    previous_stage3 = discovery.stage3_dollar_volume_filter
    previous_stage4 = discovery.stage4_options_liquidity_filter
    previous_new = os.environ.get("KAIROS_INTAKE_CAP")
    previous_old = os.environ.get("DISCOVERY_UNIVERSE_MAX_SYMBOLS")

    class FakeClient:
        def fetch_assets(self):
            return [
                {"symbol": "AAA", "status": "active", "class": "us_equity", "tradable": True, "exchange": "NYSE", "attributes": ["options_enabled"]},
                {"symbol": "BBB", "status": "active", "class": "us_equity", "tradable": True, "exchange": "NYSE", "attributes": ["options_enabled"]},
                {"symbol": "CCC", "status": "active", "class": "us_equity", "tradable": True, "exchange": "NYSE", "attributes": ["options_enabled"]},
                {"symbol": "LOW", "status": "active", "class": "us_equity", "tradable": True, "exchange": "NYSE", "attributes": ["options_enabled"]},
            ]

    dollars = [
        DollarVolumeMetrics("AAA", 10, 1_000_000, 100_000_000, 30, True),
        DollarVolumeMetrics("BBB", 20, 1_000_000, 200_000_000, 30, True),
        DollarVolumeMetrics("CCC", 30, 1_000_000, 300_000_000, 30, True),
        DollarVolumeMetrics("LOW", 5, 10_000, 50_000, 30, False, "low dollar volume"),
    ]
    options = [
        OptionsLiquidityMetrics("AAA", 10, 100, 100, "AAAC", "AAAP", 2, 1, True),
        OptionsLiquidityMetrics("BBB", 20, 200, 200, "BBBC", "BBBP", 2, 1, True),
        OptionsLiquidityMetrics("CCC", 30, 300, 300, "CCCC", "CCCP", 2, 1, True),
    ]
    try:
        os.environ["KAIROS_INTAKE_CAP"] = "2"
        os.environ.pop("DISCOVERY_UNIVERSE_MAX_SYMBOLS", None)
        discovery.AlpacaAssetDiscoveryClient = FakeClient
        discovery.stage3_dollar_volume_filter = lambda _assets: (dollars, {})
        discovery.stage4_options_liquidity_filter = lambda _metrics: options
        first = discovery.build_ranked_discovery_universe()
        second = discovery.build_ranked_discovery_universe()
        assert first["symbols"] == ["CCC", "BBB"]
        assert second["symbols"] == first["symbols"]
        assert [(row["symbol"], row["rank"], row["selected"]) for row in first["top_20"]] == [
            ("CCC", 1, True),
            ("BBB", 2, True),
            ("AAA", 3, False),
        ]
        assert first["rejection_evidence"]["stage3_dollar_volume"][0]["ticker"] == "LOW"
    finally:
        discovery.AlpacaAssetDiscoveryClient = previous_client
        discovery.stage3_dollar_volume_filter = previous_stage3
        discovery.stage4_options_liquidity_filter = previous_stage4
        if previous_new is None:
            os.environ.pop("KAIROS_INTAKE_CAP", None)
        else:
            os.environ["KAIROS_INTAKE_CAP"] = previous_new
        if previous_old is None:
            os.environ.pop("DISCOVERY_UNIVERSE_MAX_SYMBOLS", None)
        else:
            os.environ["DISCOVERY_UNIVERSE_MAX_SYMBOLS"] = previous_old


def test_kairos_intake_cap_preserves_legacy_env_compatibility():
    previous_new = os.environ.get("KAIROS_INTAKE_CAP")
    previous_old = os.environ.get("DISCOVERY_UNIVERSE_MAX_SYMBOLS")
    try:
        os.environ["KAIROS_INTAKE_CAP"] = "1000"
        os.environ["DISCOVERY_UNIVERSE_MAX_SYMBOLS"] = "750"
        resolution = discovery_universe_max_symbols_resolution()
        assert discovery_universe_max_symbols() == 1000
        assert resolution["resolved_value"] == 1000
        assert resolution["source"] == "KAIROS_INTAKE_CAP"
        assert resolution["env_var_used"] == "KAIROS_INTAKE_CAP"
        assert resolution["used_default"] is False
        assert resolution["warning"] is None

        os.environ.pop("KAIROS_INTAKE_CAP", None)
        os.environ["DISCOVERY_UNIVERSE_MAX_SYMBOLS"] = "1000"
        legacy_resolution = discovery_universe_max_symbols_resolution()
        assert discovery_universe_max_symbols() == 1000
        assert legacy_resolution["source"] == "DISCOVERY_UNIVERSE_MAX_SYMBOLS"
        assert legacy_resolution["env_var_used"] == "DISCOVERY_UNIVERSE_MAX_SYMBOLS"

        os.environ["KAIROS_INTAKE_CAP"] = "750"
        assert discovery_universe_max_symbols() == 750

        os.environ["KAIROS_INTAKE_CAP"] = "invalid"
        assert discovery_universe_max_symbols() == 1000

        os.environ["DISCOVERY_UNIVERSE_MAX_SYMBOLS"] = "invalid"
        default_resolution = discovery_universe_max_symbols_resolution()
        assert discovery_universe_max_symbols() == 1000
        assert default_resolution["resolved_value"] == 1000
        assert default_resolution["source"] == "code_default"
        assert default_resolution["env_var_used"] is None
        assert default_resolution["used_default"] is True
        assert default_resolution["default_value"] == 1000
        assert "KAIROS_INTAKE_CAP" in default_resolution["warning"]
    finally:
        if previous_new is None:
            os.environ.pop("KAIROS_INTAKE_CAP", None)
        else:
            os.environ["KAIROS_INTAKE_CAP"] = previous_new
        if previous_old is None:
            os.environ.pop("DISCOVERY_UNIVERSE_MAX_SYMBOLS", None)
        else:
            os.environ["DISCOVERY_UNIVERSE_MAX_SYMBOLS"] = previous_old


def test_scan_discovered_universe_returns_warming_when_cache_missing():
    reset_discovery_cache()
    client = TestClient(main.app)
    response = client.get("/api/scan?universe=discovered")
    assert response.status_code == 200
    payload = response.json()
    assert payload["rows"] == []
    assert payload["near_miss"] == []
    assert payload["meta"]["status"] == "warming"
    assert payload["meta"]["universe"] == "discovered"
    assert payload["meta"]["cache_key"] == "discovered"
    assert "Discovery universe is not ready" in payload["meta"]["message"]


def test_scan_discovered_universe_uses_cached_symbols_without_touching_default_or_finviz():
    original_scan_cached = main.scan_cached
    calls = []
    with main._discovery_universe_lock:
        main._discovery_universe_cache.update({
            "symbols": ["AAPL", "F", "KMI"],
            "generated_at": __import__("datetime").datetime.utcnow(),
            "expires_at": __import__("datetime").datetime.utcnow() + __import__("datetime").timedelta(hours=1),
            "running": False,
            "last_error": None,
        })

    def fake_scan_cached(watchlist=None, **kwargs):
        calls.append((watchlist, kwargs))
        return {
            "rows": [],
            "near_miss": [],
            "meta": {
                "cache_key": kwargs.get("universe") or "default",
                "configured_universe_count": len(watchlist or []),
            },
        }

    main.scan_cached = fake_scan_cached
    try:
        client = TestClient(main.app)
        response = client.get("/api/scan?universe=discovered")
        assert response.status_code == 200
        assert len(calls) == 1
        assert calls[0][0] == ["AAPL", "F", "KMI"]
        assert calls[0][1]["force_refresh"] is False
        assert calls[0][1]["universe"] == "discovered"
        assert calls[0][1]["max_symbols"] is None
        assert calls[0][1]["coverage_context"]["universe_source"] == "discovered"
        assert calls[0][1]["coverage_context"]["universe_symbol_count"] == 3
        assert response.json()["meta"]["cache_key"] == "discovered"
    finally:
        main.scan_cached = original_scan_cached
        reset_discovery_cache()


def test_scan_discovered_universe_passes_full_cached_symbol_list_without_truncation():
    original_scan_cached = main.scan_cached
    calls = []
    symbols = [f"T{i}" for i in range(550)]
    with main._discovery_universe_lock:
        main._discovery_universe_cache.update({
            "symbols": symbols,
            "generated_at": __import__("datetime").datetime.utcnow(),
            "expires_at": __import__("datetime").datetime.utcnow() + __import__("datetime").timedelta(hours=1),
            "running": False,
            "last_error": None,
        })

    def fake_scan_cached(watchlist=None, **kwargs):
        calls.append((watchlist, kwargs))
        return {
            "rows": [],
            "near_miss": [],
            "meta": {
                "cache_key": kwargs.get("universe") or "default",
                "configured_universe_count": len(watchlist or []),
            },
        }

    main.scan_cached = fake_scan_cached
    try:
        client = TestClient(main.app)
        response = client.get("/api/scan?universe=discovered")
        assert response.status_code == 200
        assert len(calls) == 1
        assert calls[0][0] == symbols
        assert calls[0][1]["force_refresh"] is False
        assert calls[0][1]["universe"] == "discovered"
        assert calls[0][1]["max_symbols"] is None
        assert calls[0][1]["coverage_context"]["universe_symbol_count"] == 550
        assert response.json()["meta"]["configured_universe_count"] == 550
    finally:
        main.scan_cached = original_scan_cached
        reset_discovery_cache()


def test_scan_default_route_uses_discovered_universe_by_default():
    original_scan_cached = main.scan_cached
    calls = []
    symbols = [f"T{i}" for i in range(750)]
    with main._discovery_universe_lock:
        main._discovery_universe_cache.update({
            "symbols": symbols,
            "generated_at": __import__("datetime").datetime.utcnow(),
            "expires_at": __import__("datetime").datetime.utcnow() + __import__("datetime").timedelta(hours=1),
            "running": False,
            "last_error": None,
        })

    def fake_scan_cached(watchlist=None, **kwargs):
        calls.append((watchlist, kwargs))
        return {
            "rows": [],
            "near_miss": [],
            "meta": {
                "cache_key": kwargs.get("universe") or "default",
                "configured_universe_count": len(watchlist or []),
            },
        }

    main.scan_cached = fake_scan_cached
    try:
        client = TestClient(main.app)
        response = client.get("/api/scan")
        assert response.status_code == 200
        assert len(calls) == 1
        assert calls[0][0] == symbols
        assert calls[0][1]["universe"] == "discovered"
        assert calls[0][1]["max_symbols"] is None
        assert calls[0][1]["coverage_context"]["universe_symbol_count"] == 750
        assert response.json()["meta"]["configured_universe_count"] == 750
    finally:
        main.scan_cached = original_scan_cached
        reset_discovery_cache()


def test_scan_explicit_default_and_finviz_modes_remain_available():
    original_scan_cached = main.scan_cached
    calls = []

    def fake_scan_cached(watchlist=None, **kwargs):
        calls.append((watchlist, kwargs))
        return {"rows": [], "near_miss": [], "meta": {}}

    main.scan_cached = fake_scan_cached
    try:
        client = TestClient(main.app)
        client.get("/api/scan?universe=default")
        client.get("/api/scan?discover=true")
        client.get("/api/scan?universe=finviz")
        assert calls == [
            (None, {"force_refresh": False, "discover": False}),
            (None, {"force_refresh": False, "discover": True}),
            (None, {"force_refresh": False, "discover": True}),
        ]
    finally:
        main.scan_cached = original_scan_cached


def test_discovery_cache_refresh_needed_for_missing_and_stale_cache():
    reset_discovery_cache()
    assert main._discovery_cache_needs_refresh() is True

    now = __import__("datetime").datetime.utcnow()
    with main._discovery_universe_lock:
        main._discovery_universe_cache.update({
            "symbols": ["AAPL"],
            "generated_at": now,
            "expires_at": now + __import__("datetime").timedelta(hours=3),
            "running": False,
            "last_error": None,
        })
    assert main._discovery_cache_needs_refresh() is False

    with main._discovery_universe_lock:
        main._discovery_universe_cache["expires_at"] = now + __import__("datetime").timedelta(minutes=30)
    assert main._discovery_cache_needs_refresh() is True

    with main._discovery_universe_lock:
        main._discovery_universe_cache["expires_at"] = now - __import__("datetime").timedelta(seconds=1)
    assert main._discovery_cache_needs_refresh() is True
    reset_discovery_cache()


def test_discovery_auto_submit_skips_fresh_cache_and_running_job():
    previous_path = os.environ.get(main.DISCOVERY_POOL_PATH_ENV)
    previous_executor = main._discovery_universe_executor
    previous_builder = main.build_ranked_discovery_universe
    previous_handoff = main._maybe_enqueue_discovered_scan_handoff
    main._discovery_universe_executor = InlineExecutor()
    main.build_ranked_discovery_universe = fake_discovery_result
    main._maybe_enqueue_discovered_scan_handoff = lambda reason="": (False, "stubbed")
    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ[main.DISCOVERY_POOL_PATH_ENV] = str(Path(temp_dir) / "pool.json")
        reset_discovery_cache()
        try:
            accepted, job_id = main._submit_discovery_universe_job_if_needed()
            assert accepted is True
            assert job_id.startswith("discovery:")
            assert main._discovery_status_snapshot()["status"] == "ready"

            accepted, reason = main._submit_discovery_universe_job_if_needed()
            assert accepted is False
            assert reason == "cache fresh"

            with main._discovery_universe_lock:
                near_expiry = __import__("datetime").datetime.utcnow() + __import__("datetime").timedelta(minutes=30)
                main._discovery_universe_cache["expires_at"] = near_expiry
            accepted, job_id = main._submit_discovery_universe_job_if_needed()
            assert accepted is True
            assert job_id.startswith("discovery:")
            assert main._discovery_status_snapshot()["status"] == "ready"

            with main._discovery_universe_lock:
                main._discovery_universe_cache.update({
                    "running": True,
                    "symbols": [],
                    "generated_at": None,
                    "expires_at": None,
                })
            accepted, reason = main._submit_discovery_universe_job_if_needed()
            assert accepted is False
            assert reason == "already running"
        finally:
            main._discovery_universe_executor = previous_executor
            main.build_ranked_discovery_universe = previous_builder
            main._maybe_enqueue_discovered_scan_handoff = previous_handoff
            reset_discovery_cache()
            if previous_path is None:
                os.environ.pop(main.DISCOVERY_POOL_PATH_ENV, None)
            else:
                os.environ[main.DISCOVERY_POOL_PATH_ENV] = previous_path


def test_startup_registers_and_submits_discovery_refresh():
    previous_register = main.register_background_periodic_task
    previous_submit = main._submit_discovery_universe_job_if_needed
    previous_load = main._load_discovery_pool_from_disk
    previous_start_market_cache = main.start_market_cache_refresh
    previous_handoff = main._maybe_enqueue_discovered_scan_handoff
    calls = []
    main.register_background_periodic_task = lambda key, ttl, callback: calls.append(("register", key, ttl, callback))
    main._load_discovery_pool_from_disk = lambda: calls.append(("load_pool",)) or False
    main._submit_discovery_universe_job_if_needed = lambda: calls.append(("submit",)) or (True, "job")
    main.start_market_cache_refresh = lambda: calls.append(("market_cache",))
    main._maybe_enqueue_discovered_scan_handoff = lambda reason="": calls.append(("handoff", reason)) or (False, "stubbed")
    try:
        main.startup_market_cache_refresh()
        assert calls[0][0:3] == ("register", "discovery_universe", main.DISCOVERY_REFRESH_WATCHDOG_SECONDS)
        assert callable(calls[0][3])
        assert calls[1][0:3] == ("register", "discovered_scan_handoff", 30)
        assert callable(calls[1][3])
        assert calls[2] == ("market_cache",)
        assert calls[3] == ("load_pool",)
        assert calls[4] == ("submit",)
        assert calls[5] == ("handoff", "startup_discovery_ready_no_scanner_cache")
    finally:
        main.register_background_periodic_task = previous_register
        main._submit_discovery_universe_job_if_needed = previous_submit
        main._load_discovery_pool_from_disk = previous_load
        main.start_market_cache_refresh = previous_start_market_cache
        main._maybe_enqueue_discovered_scan_handoff = previous_handoff


def test_discovered_scan_handoff_queues_when_discovery_ready_and_scan_cache_missing():
    reset_discovery_cache()
    reset_handoff_state()
    previous_status = main.analysis_cache_status
    previous_scan_cached = main.scan_cached
    calls = []
    now = __import__("datetime").datetime.utcnow()
    with main._discovery_universe_lock:
        main._discovery_universe_cache.update({
            "symbols": ["AAPL", "MSFT"],
            "generated_at": now,
            "expires_at": now + __import__("datetime").timedelta(hours=1),
            "running": False,
            "last_error": None,
        })

    def fake_status(watchlist=None, **kwargs):
        return {
            "has_cache": False,
            "status": "warming",
            "refreshing": False,
            "generated_at": None,
            "last_refresh_error": None,
        }

    def fake_scan_cached(watchlist=None, **kwargs):
        calls.append((watchlist, kwargs))
        return {
            "rows": [],
            "near_miss": [],
            "meta": {
                "has_cache": False,
                "status": "warming",
                "refreshing": True,
                "refresh_job_id": "universe:discovered:1:test",
                "refresh_started_at": "2026-08-03T00:00:00Z",
            },
        }

    main.analysis_cache_status = fake_status
    main.scan_cached = fake_scan_cached
    try:
        accepted, job = main._maybe_enqueue_discovered_scan_handoff("test_missing_cache")
        assert accepted is True
        assert job == "universe:discovered:1:test"
        assert len(calls) == 1
        assert calls[0][0] == ["AAPL", "MSFT"]
        assert calls[0][1]["universe"] == "discovered"
        assert calls[0][1]["max_symbols"] is None
        snapshot = main._discovered_scan_handoff_snapshot()
        assert snapshot["handoff_reason"] == "test_missing_cache"
        assert snapshot["refresh_attempt_count"] == 1
        assert snapshot["refresh_job_id"] == "universe:discovered:1:test"
    finally:
        main.analysis_cache_status = previous_status
        main.scan_cached = previous_scan_cached
        reset_discovery_cache()
        reset_handoff_state()


def test_discovered_scan_handoff_does_not_duplicate_running_or_ready_cache():
    reset_discovery_cache()
    reset_handoff_state()
    previous_status = main.analysis_cache_status
    previous_scan_cached = main.scan_cached
    calls = []
    now = __import__("datetime").datetime.utcnow()
    with main._discovery_universe_lock:
        main._discovery_universe_cache.update({
            "symbols": ["AAPL"],
            "generated_at": now,
            "expires_at": now + __import__("datetime").timedelta(hours=1),
            "running": False,
            "last_error": None,
        })

    main.scan_cached = lambda *args, **kwargs: calls.append((args, kwargs))
    try:
        main.analysis_cache_status = lambda *args, **kwargs: {
            "has_cache": False,
            "status": "warming",
            "refreshing": True,
            "refresh_job_id": "already-running",
        }
        accepted, reason = main._maybe_enqueue_discovered_scan_handoff("test_running")
        assert accepted is False
        assert reason == "scanner refresh already running"
        assert calls == []

        main.analysis_cache_status = lambda *args, **kwargs: {
            "has_cache": True,
            "status": "fresh",
            "refreshing": False,
            "generated_at": "2026-08-03T00:00:00Z",
        }
        accepted, reason = main._maybe_enqueue_discovered_scan_handoff("test_ready")
        assert accepted is False
        assert reason == "scanner cache ready"
        assert calls == []
        assert main._discovered_scan_handoff_snapshot()["scanner_cache_generation"] == "2026-08-03T00:00:00Z"
    finally:
        main.analysis_cache_status = previous_status
        main.scan_cached = previous_scan_cached
        reset_discovery_cache()
        reset_handoff_state()


def test_discovered_cache_status_triggers_handoff_and_exposes_diagnostics():
    reset_discovery_cache()
    reset_handoff_state()
    previous_status = main.analysis_cache_status
    previous_scan_cached = main.scan_cached
    calls = []
    now = __import__("datetime").datetime.utcnow()
    with main._discovery_universe_lock:
        main._discovery_universe_cache.update({
            "symbols": ["AAPL"],
            "generated_at": now,
            "expires_at": now + __import__("datetime").timedelta(hours=1),
            "running": False,
            "last_error": None,
        })

    def fake_status(watchlist=None, **kwargs):
        return {
            "has_cache": False,
            "status": "warming",
            "refreshing": bool(calls),
            "generated_at": None,
            "refresh_job_id": "handoff-job" if calls else None,
            "refresh_started_at": "2026-08-03T00:00:00Z" if calls else None,
        }

    def fake_scan_cached(watchlist=None, **kwargs):
        calls.append((watchlist, kwargs))
        return {
            "rows": [],
            "near_miss": [],
            "meta": {
                "has_cache": False,
                "status": "warming",
                "refreshing": True,
                "refresh_job_id": "handoff-job",
                "refresh_started_at": "2026-08-03T00:00:00Z",
            },
        }

    main.analysis_cache_status = fake_status
    main.scan_cached = fake_scan_cached
    try:
        client = TestClient(main.app)
        payload = client.get("/api/cache/status?universe=discovered").json()
        assert len(calls) == 1
        assert payload["status"] == "scanner_warming"
        assert payload["refreshing"] is True
        assert payload["handoff"]["handoff_reason"] == "cache_status_discovery_ready_no_scanner_cache"
        assert payload["handoff"]["refresh_job_id"] == "handoff-job"
        assert payload["message"] == "Discovery ready; scanning the 1,000-symbol universe."
    finally:
        main.analysis_cache_status = previous_status
        main.scan_cached = previous_scan_cached
        reset_discovery_cache()
        reset_handoff_state()


def test_coverage_baseline_endpoint_warms_without_completed_discovered_scan():
    reset_discovery_cache()
    previous_snapshot = main.coverage_baseline_snapshot
    main.coverage_baseline_snapshot = lambda: None
    try:
        client = TestClient(main.app)
        payload = client.get("/api/coverage/baseline").json()
        assert payload["status"] == "warming"
        assert payload["ready"] is False
        assert payload["generated_at"] is None
        assert payload["scan"] == {}
    finally:
        main.coverage_baseline_snapshot = previous_snapshot


def test_coverage_baseline_endpoint_returns_latest_snapshot_without_starting_work():
    previous_snapshot = main.coverage_baseline_snapshot
    previous_scan_cached = main.scan_cached
    calls = []
    snapshot = {
        "generated_at": "2026-07-22T12:00:00Z",
        "discovery": {"final_admitted_symbol_count": 550},
        "scan": {"universe_source": "discovered", "symbols_requested": 550},
        "stage_distribution": {"Enter Now": 1},
        "grade_distribution": {"A": 1, "B": 0, "C": 0, "unknown": 0},
        "contract_distribution": {"suggested contract available": 1},
        "blocker_distribution": {},
        "provider_failures": {},
    }
    main.coverage_baseline_snapshot = lambda: snapshot
    main.scan_cached = lambda *args, **kwargs: calls.append((args, kwargs))
    try:
        client = TestClient(main.app)
        payload = client.get("/api/coverage/baseline").json()
        assert payload["status"] == "ready"
        assert payload["ready"] is True
        assert payload["scan"]["symbols_requested"] == 550
        assert payload["stage_distribution"]["Enter Now"] == 1
        assert calls == []
    finally:
        main.coverage_baseline_snapshot = previous_snapshot
        main.scan_cached = previous_scan_cached


def main_test() -> int:
    test_discovery_status_warming_without_cache()
    test_discovery_run_disabled_when_token_unset()
    test_discovery_run_rejects_wrong_token()
    test_discovery_run_populates_cache_with_valid_token()
    test_discovery_pool_persists_and_hydrates_after_restart()
    test_legacy_persisted_pool_exposes_live_cap_resolution_metadata()
    test_successful_discovery_persists_rejection_evidence()
    test_failed_discovery_does_not_overwrite_last_good_rejection_evidence()
    test_valid_persisted_pool_prevents_startup_rediscovery()
    test_weekly_expired_persisted_pool_triggers_discovery()
    test_corrupt_or_missing_persisted_pool_safely_rebuilds()
    test_scanner_analyzes_persisted_pool_normally()
    test_intake_cap_only_truncates_when_eligible_candidates_exceed_cap()
    test_discovery_selected_symbols_unchanged_for_identical_inputs()
    test_kairos_intake_cap_preserves_legacy_env_compatibility()
    test_scan_discovered_universe_returns_warming_when_cache_missing()
    test_scan_discovered_universe_uses_cached_symbols_without_touching_default_or_finviz()
    test_scan_discovered_universe_passes_full_cached_symbol_list_without_truncation()
    test_scan_default_route_uses_discovered_universe_by_default()
    test_scan_explicit_default_and_finviz_modes_remain_available()
    test_discovery_cache_refresh_needed_for_missing_and_stale_cache()
    test_discovery_auto_submit_skips_fresh_cache_and_running_job()
    test_startup_registers_and_submits_discovery_refresh()
    test_discovered_scan_handoff_queues_when_discovery_ready_and_scan_cache_missing()
    test_discovered_scan_handoff_does_not_duplicate_running_or_ready_cache()
    test_discovered_cache_status_triggers_handoff_and_exposes_diagnostics()
    test_coverage_baseline_endpoint_warms_without_completed_discovered_scan()
    test_coverage_baseline_endpoint_returns_latest_snapshot_without_starting_work()
    print("Discovery endpoints v1 tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_test())
