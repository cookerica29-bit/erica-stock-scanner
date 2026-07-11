from datetime import datetime
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from scanner import analysis_cache_status, scan_cached, scan_ticker, debug_ticker, scan_trends, WATCHLIST, start_market_cache_refresh
from market_data import alpaca_credentials_configured, comparison_diagnostics, configured_provider_name, validate_watchlist_candles

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
def api_scan(tickers: str = Query(default=""), refresh: bool = Query(default=False)):
    """Scan the full watchlist or a custom comma-separated list of tickers."""
    if tickers:
        watchlist = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        result = scan_cached(watchlist, force_refresh=refresh)
    else:
        result = scan_cached(force_refresh=refresh)
    return result


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


@app.get("/api/cache/status")
def api_cache_status():
    return analysis_cache_status()


@app.get("/api/data-provider/status")
def api_data_provider_status():
    return {
        "stock_data_provider": configured_provider_name(),
        "default_provider": "yahoo",
        "alpaca_configured": alpaca_credentials_configured(),
        "note": "Yahoo remains the default scanner provider. Alpaca is available only for backend comparison diagnostics.",
    }


@app.get("/api/data-provider/compare")
def api_data_provider_compare(
    ticker: str = Query(default="AAPL"),
    period: str = Query(default="60d"),
    interval: str = Query(default="4h"),
    full: bool = Query(default=False),
    tickers: str = Query(default=""),
):
    if full:
        watchlist = [t.strip().upper() for t in tickers.split(",") if t.strip()] if tickers else WATCHLIST
        return validate_watchlist_candles(watchlist)
    return comparison_diagnostics(ticker=ticker, period=period, interval=interval)


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
