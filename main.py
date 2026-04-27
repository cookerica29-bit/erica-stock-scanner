from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from scanner import scan_all, scan_ticker, debug_ticker, WATCHLIST

app = FastAPI(title="Stock Options Scanner")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="public"), name="static")


@app.get("/")
def index():
    return FileResponse("public/index.html")


@app.get("/api/scan")
def api_scan(tickers: str = Query(default="")):
    """Scan the full watchlist or a custom comma-separated list of tickers."""
    if tickers:
        watchlist = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    else:
        watchlist = WATCHLIST
    rows, near_miss = scan_all(watchlist)
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
