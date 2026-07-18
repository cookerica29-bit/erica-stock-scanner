const assert = require('assert');
const mix = require('../public/opportunity_price_mix.js');

function setup(overrides = {}) {
  return {
    ticker: 'TEST',
    direction: 'LONG',
    entry: 50,
    price: 51,
    setupGrade: 'B+ TRADEABLE',
    status: 'ENTER_NOW',
    best_contract: {
      available: true,
      source: 'option_chain',
      type: 'CALL',
      strike: 50,
      expiry: '2026-08-21',
      bid: 1.1,
      ask: 1.2,
      spread: 0.1,
      open_interest: 250,
      volume: 60,
    },
    ...overrides,
  };
}

const budget = setup({ ticker: 'F', best_contract: { available: true, source: 'option_chain', ask: 2.5 } });
const mid = setup({ ticker: 'PLTR', status: 'ALMOST_READY', best_contract: { available: true, source: 'option_chain', ask: 4.5 } });
const premium = setup({ ticker: 'NVDA', status: 'ENTER_NOW', best_contract: { available: true, source: 'option_chain', ask: 9.5 } });
const early = setup({ ticker: 'BKR', status: 'EARLY_ENTRY', best_contract: { available: true, source: 'option_chain', ask: 2.1 } });
const unavailable = setup({
  ticker: 'BMY',
  status: 'WAITING',
  best_contract: { available: false, source: 'unavailable', reason: 'No option expirations available' },
});
const skip = setup({ ticker: 'NOPE', status: 'SKIP', direction: '', setupGrade: 'C', best_contract: {} });

assert.strictEqual(mix.contractCost(budget), 250);
assert.strictEqual(mix.costBand(250), 'BUDGET');
assert.strictEqual(mix.costBand(251), 'MID_RANGE');
assert.strictEqual(mix.costBand(600), 'MID_RANGE');
assert.strictEqual(mix.costBand(601), 'PREMIUM');
assert.strictEqual(mix.costBand(null), 'UNAVAILABLE');

const diag = mix.summarize([budget, mid, premium, early, unavailable, skip], {
  progressResolver: row => ({ bucket: row.status }),
});

assert.strictEqual(diag.stage_matrix.universe.BUDGET, 2);
assert.strictEqual(diag.stage_matrix.universe.MID_RANGE, 1);
assert.strictEqual(diag.stage_matrix.universe.PREMIUM, 1);
assert.strictEqual(diag.stage_matrix.universe.UNAVAILABLE, 2);
assert.strictEqual(diag.stage_matrix.enter_now.BUDGET, 1);
assert.strictEqual(diag.stage_matrix.enter_now.PREMIUM, 1);
assert.strictEqual(diag.stage_matrix.early_entry.BUDGET, 1);
assert.strictEqual(diag.stage_matrix.developing.BUDGET, 1);
assert.strictEqual(diag.stage_matrix.developing.MID_RANGE, 1);
assert.strictEqual(diag.stage_matrix.developing.UNAVAILABLE, 1);
assert.strictEqual(diag.stage_matrix.skip.UNAVAILABLE, 1);
assert.strictEqual(diag.stage_matrix.suggested_contract_found.BUDGET, 2);
assert.strictEqual(diag.stage_matrix.suggested_contract_found.MID_RANGE, 1);
assert.strictEqual(diag.stage_matrix.suggested_contract_found.PREMIUM, 1);

assert.strictEqual(diag.by_section['Enter Now'].BUDGET, 1);
assert.strictEqual(diag.by_section['Enter Now'].PREMIUM, 1);
assert.strictEqual(diag.by_section['Early Entry'].BUDGET, 1);
assert.strictEqual(diag.by_section.Developing.MID_RANGE, 1);
assert.ok(diag.by_failure_reason['No option expirations available'] >= 1);

const row = diag.rows.find(item => item.ticker === 'PLTR');
assert.strictEqual(row.cost_band, 'MID_RANGE');
assert.strictEqual(row.section, 'Developing');
assert.strictEqual(row.suggested_contract_found, true);

assert.ok(diag.proposals.some(item => item.bucket === 'Budget-likely'));
assert.ok(diag.proposals.some(item => item.bucket === 'Mid-Range-likely'));
assert.ok(diag.safeguards.some(item => item.includes('far-OTM')));

console.log('Opportunity price mix v1 tests passed');
