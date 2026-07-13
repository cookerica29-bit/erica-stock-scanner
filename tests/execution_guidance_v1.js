const assert = require('assert');
const guidance = require('../public/execution_guidance.js');

function setup(overrides = {}) {
  return {
    ticker: 'NKE',
    direction: 'SHORT',
    price: 75,
    entry: 70,
    sl: 72,
    tp1: 65,
    entryStatus: 'Near Entry',
    trade_eval: { trade_stage: 'BUILDING / WATCHLIST' },
    ...overrides,
  };
}

// Label data for cards: Current Price and Planned Entry remain separate.
const rows = guidance.executionPlanRows(setup());
assert.deepStrictEqual(rows.map(row => row.label), ['Current Price', 'Planned Entry', 'Stop', 'Target']);
assert.strictEqual(rows.find(row => row.label === 'Current Price').value, 75);
assert.strictEqual(rows.find(row => row.label === 'Planned Entry').value, 70);
assert.notStrictEqual(guidance.currentPrice(setup()), guidance.plannedEntry(setup()));

// Almost Ready tells the trader to wait and optionally set an alert at the planned entry.
const almostReady = guidance.nextStep(setup(), { bucket: 'ALMOST_READY' }, 'pending');
assert.strictEqual(almostReady.label, 'Next Step');
assert.ok(almostReady.lines.includes('Waiting for price to reach the planned entry.'));
assert.ok(almostReady.lines.includes('Set an alert at $70.00.'));

// Enter Now only uses execute wording when a suggested contract is available.
const enterWithContract = guidance.nextStep(setup({ entryStatus: 'Tradeable' }), { bucket: 'ENTER_NOW' }, 'available');
assert.deepStrictEqual(enterWithContract.lines, ['Price is in the planned entry zone. You can execute this trade.']);

const enterWithoutContract = guidance.nextStep(setup({ entryStatus: 'Tradeable' }), { bucket: 'ENTER_NOW' }, 'confirmed_unavailable');
assert.deepStrictEqual(enterWithoutContract.lines, ['Price is in the planned entry zone.', 'Select an option contract before executing.']);

// Tradeable but not Enter Now keeps the wording grounded in the existing production meaning.
const tradeable = guidance.nextStep(setup({ entryStatus: 'Tradeable' }), { bucket: 'ALMOST_READY' }, 'available');
assert.deepStrictEqual(tradeable.lines, ['Monitor price near the planned entry.']);

// Watchlist/building setups remain monitoring-only.
const building = guidance.nextStep(setup({ entryStatus: 'Waiting', trade_eval: { trade_stage: 'BUILDING / WATCHLIST' } }), { bucket: 'WAITING' }, 'pending');
assert.deepStrictEqual(building.lines, ['Continue monitoring. The setup is still developing.']);

// Invalidated/no-trade setups explicitly say no entry.
const invalid = guidance.nextStep(setup({ trade_eval: { trade_stage: 'RANGE / NO TRADE' } }), { bucket: 'SKIP' }, 'pending');
assert.deepStrictEqual(invalid.lines, ['No entry. The setup is not currently valid.']);

// Nike-style unfilled planned entry remains unfilled: price and planned entry are not forced together.
const nike = setup({ ticker: 'NKE', price: 77.25, entry: 70.5, direction: 'SHORT' });
assert.strictEqual(guidance.currentPrice(nike), 77.25);
assert.strictEqual(guidance.plannedEntry(nike), 70.5);

// Helpers do not mutate scanner status or eligibility fields.
const original = setup({ progress_bucket: 'ALMOST_READY', setupGrade: 'B+ TRADEABLE' });
const before = JSON.stringify(original);
guidance.nextStep(original, { bucket: 'ALMOST_READY' }, 'pending');
guidance.executionPlanRows(original);
assert.strictEqual(JSON.stringify(original), before);

console.log('Execution guidance v1 tests passed');
