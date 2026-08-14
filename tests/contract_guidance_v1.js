const assert = require('assert');
const guidance = require('../public/contract_guidance.js');
const execution = require('../public/execution_guidance.js');

function setup(overrides = {}) {
  return {
    ticker: 'NKE',
    direction: 'SHORT',
    price: 320.12,
    entry: 314.54,
    setupGrade: 'B',
    best_contract: {
      available: false,
      source: 'unavailable',
      reason: 'no contract passed filters',
    },
    ...overrides,
  };
}

const readyExpiration = {
  card_ready: true,
  suggested_expiration_min_dte: 21,
  suggested_expiration_max_dte: 45,
};

const learningExpiration = {
  card_ready: false,
  suggested_expiration_min_dte: null,
  suggested_expiration_max_dte: null,
};

// Validated live contracts always take priority over planning estimates.
const live = setup({
  best_contract: { available: true, type: 'PUT', strike: 315, expiry: '2026-08-21' },
});
assert.deepStrictEqual(guidance.guidanceState(live, 'available', readyExpiration).state, 'validated_live');

// Bearish estimates round at or above Planned Entry, not current price.
const bearish = guidance.potentialContract(setup(), readyExpiration);
assert.strictEqual(bearish.type, 'PUT');
assert.strictEqual(bearish.strike, 315);
assert.strictEqual(bearish.planned_entry, 314.54);
assert.notStrictEqual(bearish.strike, 320);

// Bullish estimates round at or below Planned Entry.
const bullish = guidance.potentialContract(setup({ direction: 'LONG', entry: 57.47 }), readyExpiration);
assert.strictEqual(bullish.type, 'CALL');
assert.strictEqual(bullish.strike, 57);

// Real strike increment metadata is preferred when available.
const metadataIncrement = guidance.potentialContract(setup({
  direction: 'LONG',
  entry: 57.47,
  option_metadata: { strike_increment: 0.5 },
}), readyExpiration);
assert.strictEqual(metadataIncrement.strike_increment, 0.5);
assert.strictEqual(metadataIncrement.strike_increment_source, 'metadata');
assert.strictEqual(metadataIncrement.strike, 57);

// Fallback strike increments follow the centralized price buckets.
assert.strictEqual(guidance.strikeIncrementForPrice(24.9), 0.5);
assert.strictEqual(guidance.strikeIncrementForPrice(57.47), 1);
assert.strictEqual(guidance.strikeIncrementForPrice(150), 2.5);
assert.strictEqual(guidance.strikeIncrementForPrice(314.54), 5);

// Qualified Expected Move produces Suggested Expiration; learning does not invent a DTE range.
assert.strictEqual(guidance.expirationGuidance(readyExpiration).label, '21–45 DTE');
assert.strictEqual(guidance.expirationGuidance(learningExpiration).label, 'Learning');
assert.strictEqual(guidance.potentialContract(setup(), learningExpiration).expiration_label, 'Learning');

// Potential Contract is distinct from Suggested Contract and remains estimated.
const potentialState = guidance.guidanceState(setup(), 'confirmed_unavailable', readyExpiration);
assert.strictEqual(potentialState.state, 'potential');
assert.strictEqual(potentialState.potential.contract_guidance_source, 'estimated');

// Missing direction and missing Planned Entry fail safely.
assert.strictEqual(guidance.potentialContract(setup({ direction: '' }), readyExpiration).reason, 'missing direction');
assert.strictEqual(guidance.potentialContract(setup({ entry: null }), readyExpiration).reason, 'missing planned entry');

// Next Step never says execute when only a fallback contract exists.
const fallbackStep = execution.nextStep(setup({ price: 314.54, entryStatus: 'Tradeable' }), { bucket: 'ENTER_NOW' }, 'potential');
assert.ok(fallbackStep.lines.join(' ').includes('Use the Contract Candidate as planning guidance and verify the contract in your broker'));
assert.ok(!fallbackStep.lines.join(' ').includes('You can execute this trade'));

// Estimated fields are separate from verified contract fields.
const estimatedJournalFields = {
  option_type: 'N/A',
  strike_price: null,
  expiration_date: '',
  potential_option_type: potentialState.potential.type,
  potential_strike: potentialState.potential.strike,
  potential_expiration_min_dte: potentialState.potential.potential_expiration_min_dte,
  potential_expiration_max_dte: potentialState.potential.potential_expiration_max_dte,
  contract_guidance_source: potentialState.potential.contract_guidance_source,
};
assert.strictEqual(estimatedJournalFields.option_type, 'N/A');
assert.strictEqual(estimatedJournalFields.strike_price, null);
assert.strictEqual(estimatedJournalFields.expiration_date, '');
assert.strictEqual(estimatedJournalFields.potential_option_type, 'PUT');
assert.strictEqual(estimatedJournalFields.potential_strike, 315);
assert.strictEqual(estimatedJournalFields.contract_guidance_source, 'estimated');

console.log('Contract guidance v1 tests passed');
