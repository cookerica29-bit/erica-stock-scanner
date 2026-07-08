from datetime import datetime
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from scanner import scan_all, scan_ticker, debug_ticker, scan_trends, WATCHLIST, start_market_cache_refresh

app = FastAPI(title="Stock Options Scanner")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="public"), name="static")


@app.on_event("startup")
def startup_market_cache_refresh():
    start_market_cache_refresh()


@app.get("/")
def index():
    return FileResponse(
        "public/index.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


@app.get("/api/scan")
def api_scan(tickers: str = Query(default="")):
    """Scan the full watchlist or a custom comma-separated list of tickers."""
    if tickers:
        watchlist = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        rows, near_miss = scan_all(watchlist)
    else:
        rows, near_miss = scan_all()
    return {"rows": rows, "near_miss": near_miss}


@app.get("/api/scan/{ticker}")
def api_scan_single(ticker: str):
    result = scan_ticker(ticker)
    if result is None:
        return {"setup": None, "reason": "No valid setup found"}
    return result


@app.get("/api/debug/{ticker}")
def api_debug(ticker: str):
    """Return full filter reasoning for a single ticker — every step pass/fail."""
    return debug_ticker(ticker)


@app.get("/api/watchlist")
def api_watchlist():
    return {"watchlist": WATCHLIST}


@app.get("/api/trends")
def api_trends(
    direction: str = Query(default="all"),
    min_score: int = Query(default=0, ge=0, le=100),
    hide_choppy: bool = Query(default=False),
):
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "rows": scan_trends(direction=direction, min_score=min_score, hide_choppy=hide_choppy),
    }
