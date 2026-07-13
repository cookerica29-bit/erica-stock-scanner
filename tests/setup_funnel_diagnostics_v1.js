const assert = require('assert');
const funnel = require('../public/setup_funnel_diagnostics.js');

function setup(overrides = {}) {
  return {
    ticker: 'ATO',
    direction: 'SHORT',
    timeframe: '4H',
    setupGrade: 'B+ TRADEABLE',
    grade_value: 'B',
    progress_bucket: 'ALMOST_READY',
    entryStatus: 'Near Entry',
    stockLocation: 'Premium',
    setupTimeframeDirection: 'Bearish',
    price: 112,
    entry: 111,
    sl: 114,
    tp1: 105,
    signal_timestamp: '2026-07-13T14:00:00Z',
    trade_eval: {},
    ...overrides,
  };
}

function snapshot(rows, meta = {}) {
  return funnel.createScanSnapshot({
    rows,
    meta: { symbols_requested: rows.length, symbols_processed: rows.length, symbols_failed: 0, market_date: '2026-07-13', ...meta },
    timestamp: '2026-07-13T15:00:00Z',
    scanId: meta.scan_id,
  });
}

// One setup is counted once per stage.
const enter = setup({
  progress_bucket: 'ENTER_NOW',
  entryStatus: 'Tradeable',
  confirmationStarted: true,
  trade_eval: { trigger_confirmed: true },
});
const snapOne = snapshot([enter]);
assert.strictEqual(snapOne.stage_counts.directional_setup_found, 1);
assert.strictEqual(snapOne.stage_counts.enter_now, 1);
assert.strictEqual(snapOne.setups.length, 1);

// Multiple blockers are preserved and primary blocker is deterministic.
const blocked = setup({
  direction: 'LONG',
  stockLocation: 'Premium',
  setupTimeframeDirection: 'Bearish CONFLICT',
  progress_bucket: 'WAITING',
  entryStatus: 'Waiting',
  confirmationStarted: false,
});
const diag = funnel.setupDiagnostic(blocked);
assert.ok(diag.all_blocking_reasons.includes('HTF_CONFLICT'));
assert.ok(diag.all_blocking_reasons.includes('LOCATION_CONFLICT'));
assert.strictEqual(diag.primary_blocking_reason, 'HTF_CONFLICT');

// Data failures remain separate from strategy blockers.
const missingData = funnel.setupDiagnostic(setup({ price: null, current_price: null, underlying_price: null }));
assert.strictEqual(missingData.reason_types.MISSING_MARKET_DATA, funnel.REASON_TYPES.DATA);

// Cautions are not blockers unless the setup explicitly says they block entry.
const cautionOnly = funnel.setupDiagnostic(setup({ earnings_blocks_entry: false }));
assert.ok(!cautionOnly.all_blocking_reasons.includes('EARNINGS_CAUTION'));
const cautionBlocking = funnel.setupDiagnostic(setup({ earnings_blocks_entry: true }));
assert.strictEqual(cautionBlocking.reason_types.EARNINGS_CAUTION, funnel.REASON_TYPES.CAUTION);

// Latest scan funnel counts and conversion rates are accurate.
const rows = [
  enter,
  setup({ ticker: 'BMY', progress_bucket: 'ALMOST_READY', entryStatus: 'Near Entry' }),
  setup({ ticker: 'CCL', direction: '', progress_bucket: 'SKIP', setupGrade: 'C', grade_value: 'C' }),
];
const snap = snapshot(rows);
assert.strictEqual(snap.symbols_processed, 3);
assert.strictEqual(snap.stage_counts.enter_now, 1);
assert.strictEqual(snap.stage_counts.grade_eligible, 2);
assert.strictEqual(snap.conversion_rates.overall_enter_now_rate, 1 / 3);

// Same setup does not duplicate on repeated scan of the same signal in daily aggregation.
const first = snapshot([enter], { scan_id: 'scan-1' });
const second = snapshot([enter], { scan_id: 'scan-2' });
const daily = funnel.dailyAggregation([first, second]);
assert.strictEqual(daily[0].unique_setups_seen, 1);
assert.strictEqual(daily[0].unique_enter_now_setups, 1);

// A genuinely new signal candle counts as a new setup.
const laterSignal = setup({ progress_bucket: 'ENTER_NOW', entryStatus: 'Tradeable', confirmationStarted: true, signal_timestamp: '2026-07-14T14:00:00Z' });
const dailyNew = funnel.dailyAggregation([first, snapshot([laterSignal], { scan_id: 'scan-3', market_date: '2026-07-13' })]);
assert.strictEqual(dailyNew[0].unique_setups_seen, 2);

// A+ and B+, Long and Short, and timeframes stay visible in diagnostics.
const mixed = snapshot([
  setup({ ticker: 'AAA', setupGrade: 'A+ READY', grade_value: 'A', direction: 'LONG', timeframe: 'Daily', progress_bucket: 'ENTER_NOW', entryStatus: 'Tradeable', confirmationStarted: true }),
  setup({ ticker: 'BBB', setupGrade: 'B+ TRADEABLE', grade_value: 'B', direction: 'SHORT', timeframe: '4H', progress_bucket: 'ENTER_NOW', entryStatus: 'Tradeable', confirmationStarted: true }),
]);
const mixedDaily = funnel.dailyAggregation([mixed])[0];
assert.strictEqual(mixedDaily.a_plus_enter_now, 1);
assert.strictEqual(mixedDaily.b_plus_tradeable_enter_now, 1);
assert.strictEqual(mixedDaily.long_enter_now, 1);
assert.strictEqual(mixedDaily.short_enter_now, 1);
assert.deepStrictEqual(mixedDaily.enter_now_by_timeframe, { Daily: 1, '4H': 1 });

// History retention limit is enforced.
const history = Array.from({ length: 25 }, (_, i) => snapshot([setup({ ticker: `T${i}` })], { scan_id: `scan-${i}` }));
const retained = history.reduce((acc, item) => funnel.appendHistory(acc, item, 20), []);
assert.strictEqual(retained.length, 20);

// Scarcity and bottleneck classifications are descriptive only.
const lowOppRows = Array.from({ length: 30 }, (_, i) => setup({ ticker: `L${i}`, direction: '', setupGrade: 'C', grade_value: 'C', progress_bucket: 'SKIP' }));
assert.strictEqual(snapshot(lowOppRows).diagnostic_classification, 'LOW_OPPORTUNITY_ENVIRONMENT');

const bottleneckRows = Array.from({ length: 30 }, (_, i) => setup({
  ticker: `B${i}`,
  progress_bucket: i < 2 ? 'ENTER_NOW' : 'ALMOST_READY',
  entryStatus: i < 2 ? 'Tradeable' : 'Tradeable',
  confirmationStarted: true,
  trade_eval: i < 2 ? { trigger_confirmed: true } : {},
}));
assert.strictEqual(snapshot(bottleneckRows).diagnostic_classification, 'POSSIBLE_FINAL_GATE_BOTTLENECK');

console.log('Setup funnel diagnostics v1 tests passed');
