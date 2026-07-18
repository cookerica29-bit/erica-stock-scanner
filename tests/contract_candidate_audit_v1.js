const assert = require('assert');
const audit = require('../public/contract_candidate_audit.js');

function setup(overrides = {}) {
  return {
    ticker: 'TEST',
    direction: 'LONG',
    setupGrade: 'A',
    status: 'ENTER_NOW',
    best_contract: {
      available: true,
      candidate_audit: {
        candidate_count: 3,
        acceptable_candidate_count: 2,
        current_selected_contract: {
          type: 'CALL',
          strike: 100,
          expiration: '2026-08-21',
          estimated_contract_cost: 885,
          score: 88,
          rejection_reasons: [],
        },
        best_quality_contract: {
          type: 'CALL',
          strike: 100,
          expiration: '2026-08-21',
          estimated_contract_cost: 885,
          score: 88,
          rejection_reasons: [],
        },
        best_balanced_contract: {
          type: 'CALL',
          strike: 102.5,
          expiration: '2026-08-21',
          estimated_contract_cost: 510,
          score: 76,
          rejection_reasons: [],
        },
        lowest_cost_acceptable_contract: {
          type: 'CALL',
          strike: 105,
          expiration: '2026-08-21',
          estimated_contract_cost: 390,
          score: 64,
          rejection_reasons: [],
        },
        potential_savings: 495,
        rejected_candidates: [
          { type: 'CALL', strike: 110, estimated_contract_cost: 150, score: 42, rejection_reasons: ['score below Fair threshold'] },
        ],
        candidates: [],
      },
    },
    ...overrides,
  };
}

assert.strictEqual(audit.costBand(250), 'BUDGET');
assert.strictEqual(audit.costBand(600), 'MID_RANGE');
assert.strictEqual(audit.costBand(601), 'PREMIUM');
assert.strictEqual(audit.costBand(null), 'UNAVAILABLE');

const rows = [
  setup({ ticker: 'UAL' }),
  setup({ ticker: 'XOM', status: 'ALMOST_READY', best_contract: { candidate_audit: {
    candidate_count: 2,
    acceptable_candidate_count: 1,
    current_selected_contract: { estimated_contract_cost: 355, score: 80 },
    best_quality_contract: { estimated_contract_cost: 355, score: 80 },
    lowest_cost_acceptable_contract: { estimated_contract_cost: 355, score: 80 },
    rejected_candidates: [],
    candidates: [],
  } } }),
  setup({ ticker: 'BKR', status: 'EARLY_ENTRY', best_contract: { candidate_audit: {
    candidate_count: 2,
    acceptable_candidate_count: 1,
    current_selected_contract: { estimated_contract_cost: 245, score: 72 },
    best_quality_contract: { estimated_contract_cost: 245, score: 72 },
    lowest_cost_acceptable_contract: { estimated_contract_cost: 245, score: 72 },
    rejected_candidates: [],
    candidates: [],
  } } }),
  setup({ ticker: 'AAPL', status: 'WAITING', best_contract: {} }),
  setup({ ticker: 'SKIP', status: 'SKIP', best_contract: {} }),
];

const summary = audit.summarize(rows, { progressResolver: row => ({ bucket: row.status }) });
assert.strictEqual(summary.setup_count, 4);
assert.strictEqual(summary.setups_audited, 3);
assert.strictEqual(summary.cheaper_acceptable_candidate_found, 1);
assert.strictEqual(summary.no_cheaper_acceptable_candidate, 2);
assert.strictEqual(summary.candidate_data_unavailable, 1);
assert.strictEqual(summary.average_potential_savings, 495);
assert.strictEqual(summary.distribution_by_current_selection_band.PREMIUM, 1);
assert.strictEqual(summary.distribution_by_current_selection_band.MID_RANGE, 1);

const ual = summary.rows.find(row => row.ticker === 'UAL');
assert.strictEqual(ual.cheaperAcceptableFound, true);
assert.strictEqual(ual.potentialSavings, 495);
assert.strictEqual(ual.productionSelectionChanged, false);

console.log('Contract candidate audit v1 tests passed');
