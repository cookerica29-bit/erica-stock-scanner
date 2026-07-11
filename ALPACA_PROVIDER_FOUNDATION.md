# Alpaca Provider Foundation

This sprint adds provider infrastructure only. It does not migrate the scanner
to Alpaca and does not change Stock Scanner Strategy v1.0.

## Current Yahoo Data Flow

- Candle requests are made through `scanner.py` calls shaped like
  `yf.download(ticker_or_list, period=..., interval=..., auto_adjust=True)`.
- `yf` is now a market-data facade from `market_data.py`.
- Default provider remains Yahoo through `STOCK_DATA_PROVIDER=yahoo`.
- Options and earnings still use Yahoo via `yf.Ticker(...)`.

## Candle Normalization

The scanner consumes pandas DataFrames with these columns:

- `Open`
- `High`
- `Low`
- `Close`
- `Volume`

`market_data.py` normalizes provider responses into that schema. Yahoo behavior
is preserved by delegating to `yfinance.download` and returning the same shape
that the scanner already expected.

## Ticker Mapping

- Yahoo symbols are passed through unchanged.
- Alpaca symbols are uppercased and `-` is mapped to `.` for provider requests.
- Diagnostics report both the requested symbol and provider symbol.

## Timeframe Handling

Current scanner periods and intervals:

- Daily: `period="1y"`, `interval="1d"`
- Weekly: `period="2y"`, `interval="1wk"`
- 4H: `period="60d"`, `interval="4h"`

Alpaca comparison mode maps those to:

- `1d` -> `1Day`
- `1wk` -> `1Week`
- `4h` -> `4Hour`

## Completed Candle Handling

Existing Yahoo behavior does not explicitly drop an in-progress candle. The
provider foundation preserves that behavior for production scanner output.
Comparison diagnostics report the latest provider timestamp so differences can
be reviewed before any future migration decision.

## Retry And Caching Behavior

Existing scanner cache behavior is unchanged:

- price cache TTL: 3 minutes
- options chain TTL: 8 minutes
- best contract TTL: 8 minutes
- earnings TTL: 12 hours
- unavailable earnings TTL: 24 hours

Price cache refreshes still use the scanner's existing background job mechanism.
Provider comparison is backend-only and does not write scanner analysis output.

## Environment Variables

- `STOCK_DATA_PROVIDER=yahoo`
- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `ALPACA_DATA_BASE_URL`

Yahoo remains the default. Alpaca is not activated unless
`STOCK_DATA_PROVIDER=alpaca` is explicitly set, and this sprint does not set it.

## Diagnostics Endpoints

- `GET /api/data-provider/status`
- `GET /api/data-provider/compare?ticker=AAPL&period=60d&interval=4h`

Comparison mode reports:

- candle count
- latest completed timestamp
- latest OHLCV
- missing candle timestamps
- symbol mapping

It compares raw provider market data only. It does not compare strategy output.
