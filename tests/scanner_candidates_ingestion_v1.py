import os
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _client():
    import candidates_router

    app = FastAPI()
    app.include_router(candidates_router.router)
    return TestClient(app)


def test_scanner_candidate_ingestion_lifecycle():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "candidates.db")
        os.environ["KAIROS_CANDIDATES_DB"] = db_path
        os.environ["KAIROS_SCANNER_API_KEY"] = "test-scanner-key"

        client = _client()
        headers = {"X-API-Key": "test-scanner-key"}
        payload = {
            "source": "ma_pipeline",
            "scanned_at": "2026-08-20T14:30:00Z",
            "candidates": [
                {
                    "ticker": "nvda",
                    "signal": "long",
                    "entry_price": 217.25,
                    "ema21_4h": 214.1,
                    "daily_regime": "bullish",
                    "confidence": "high",
                    "sma50_daily": 207.3,
                    "sma200_daily": 195.08,
                },
                {"ticker": "BAD/FORMAT", "signal": "short"},
            ],
        }

        missing_key = client.post("/api/v1/scanner/candidates", json=payload)
        assert missing_key.status_code == 401

        wrong_key = client.post(
            "/api/v1/scanner/candidates",
            headers={"X-API-Key": "wrong"},
            json=payload,
        )
        assert wrong_key.status_code == 401

        created = client.post("/api/v1/scanner/candidates", headers=headers, json=payload)
        assert created.status_code == 200
        assert created.json() == {
            "received": 2,
            "created": 1,
            "updated": 0,
            "rejected": [{"ticker": "BAD/FORMAT", "reason": "invalid ticker format"}],
        }

        listed = client.get("/api/v1/scanner/candidates", headers=headers).json()
        assert len(listed) == 1
        assert listed[0] == {
            "ticker": "NVDA",
            "source": "ma_pipeline",
            "signal": "long",
            "entry_price": 217.25,
            "ema21_4h": 214.1,
            "daily_regime": "bullish",
            "confidence": "high",
            "sma50_daily": 207.3,
            "sma200_daily": 195.08,
            "status": "new",
            "scanned_at": "2026-08-20T14:30:00+00:00",
            "updated_at": listed[0]["updated_at"],
        }

        update_payload = {
            "source": "ma_pipeline",
            "scanned_at": "2026-08-20T18:30:00Z",
            "candidates": [
                {
                    "ticker": "NVDA",
                    "signal": "short",
                    "entry_price": 210.5,
                    "ema21_4h": 212.0,
                    "daily_regime": "bearish",
                    "confidence": "medium",
                    "sma50_daily": 206.0,
                    "sma200_daily": 194.0,
                },
            ],
        }
        updated = client.post("/api/v1/scanner/candidates", headers=headers, json=update_payload)
        assert updated.status_code == 200
        assert updated.json() == {"received": 1, "created": 0, "updated": 1, "rejected": []}

        after_update = client.get("/api/v1/scanner/candidates", headers=headers).json()
        assert len(after_update) == 1
        assert after_update[0]["ticker"] == "NVDA"
        assert after_update[0]["signal"] == "short"
        assert after_update[0]["entry_price"] == 210.5
        assert after_update[0]["daily_regime"] == "bearish"

        import sqlite3

        conn = sqlite3.connect(db_path)
        try:
            current_count = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
            history_count = conn.execute("SELECT COUNT(*) FROM candidate_history").fetchone()[0]
        finally:
            conn.close()
        assert current_count == 1
        assert history_count == 2

        promoted = client.patch(
            "/api/v1/scanner/candidates/NVDA?source=ma_pipeline",
            headers=headers,
            json={"status": "active"},
        )
        assert promoted.status_code == 200
        assert promoted.json() == {"ticker": "NVDA", "source": "ma_pipeline", "status": "active"}

        active = client.get("/api/v1/scanner/candidates?status=active", headers=headers).json()
        assert len(active) == 1
        assert active[0]["ticker"] == "NVDA"

        restarted_client = _client()
        after_restart = restarted_client.get("/api/v1/scanner/candidates", headers=headers).json()
        assert len(after_restart) == 1
        assert after_restart[0]["ticker"] == "NVDA"
        assert after_restart[0]["status"] == "active"


if __name__ == "__main__":
    test_scanner_candidate_ingestion_lifecycle()
    print("scanner_candidates_ingestion_v1 passed")
