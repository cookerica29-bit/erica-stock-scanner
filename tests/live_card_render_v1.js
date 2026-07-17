const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const executionGuidance = require('../public/execution_guidance.js');
const contractGuidance = require('../public/contract_guidance.js');
const cardData = require('../public/card_data.js');

const html = fs.readFileSync('public/index.html', 'utf8');
const inline = [...html.matchAll(/<script(?![^>]*src=)[^>]*>([\s\S]*?)<\/script>/gi)][0][1];

function elementStub() {
  return {
    innerHTML: '',
    textContent: '',
    value: '',
    checked: false,
    dataset: {},
    style: {},
    selectedOptions: [{ textContent: '' }],
    classList: { add() {}, remove() {}, toggle() {} },
    addEventListener() {},
    removeEventListener() {},
    setAttribute() {},
    appendChild() {},
  };
}

const storage = {};
const context = {
  console,
  Date,
  Math,
  Number,
  String,
  Boolean,
  Array,
  Object,
  JSON,
  RegExp,
  encodeURIComponent,
  decodeURIComponent,
  setTimeout: () => 1,
  clearTimeout: () => {},
  setInterval: () => 1,
  clearInterval: () => {},
  alert: () => {},
  confirm: () => false,
  fetch: () => Promise.reject(new Error('network disabled in render test')),
  localStorage: {
    getItem: key => storage[key] || null,
    setItem: (key, value) => { storage[key] = String(value); },
    removeItem: key => { delete storage[key]; },
  },
  document: {
    getElementById: () => elementStub(),
    querySelectorAll: () => [],
    createElement: () => elementStub(),
  },
  navigator: { clipboard: { writeText: () => Promise.resolve() } },
};
context.window = {
  KairosExecutionGuidance: executionGuidance,
  KairosContractGuidance: contractGuidance,
  KairosCardData: cardData,
  open: () => {},
  addEventListener: () => {},
  removeEventListener: () => {},
};
context.self = context.window;
context.globalThis = context;

vm.createContext(context);
vm.runInContext(inline, context);

function setup(overrides = {}) {
  return {
    ticker: 'BAM',
    direction: 'SHORT',
    price: 46.34,
    entry: 46.05,
    sl: 47,
    tp1: 44,
    timeframe: '4H',
    entryStatus: 'Near Entry',
    distanceFromEntryAtr: 0.5,
    setupGrade: 'A',
    stockTrend: 'Bearish',
    stockPhase: 'Pullback',
    stockSetupStatus: 'Pullback Active',
    stockLocation: 'Premium',
    confirmationStarted: true,
    trade_eval: { trade_stage: 'A+ READY', trigger_confirmed: true },
    best_contract: { available: true, type: 'PUT', strike: 45, expiry: '2026-08-21', ask: 1.2, bid: 1.1, spread: 0.1 },
    ...overrides,
  };
}

function htmlFor(fixture, lifecycleContract = undefined) {
  const input = lifecycleContract ? { ...fixture, best_contract: lifecycleContract } : fixture;
  return context.renderCard(input);
}

const enterWaiting = htmlFor(setup());
assert.ok(enterWaiting.includes('data-normalized-status="ENTER_NOW"'));
assert.ok(enterWaiting.includes('data-execution-state="SETUP_CONFIRMED_WAITING_FOR_ENTRY"'));
assert.ok(enterWaiting.includes('data-execute-visual-state="waiting"'));
assert.ok(enterWaiting.includes('execute-waiting-entry'));
assert.ok(enterWaiting.includes('Setup confirmed. Wait for price to reach the planned entry at $46.05.'));
assert.ok(enterWaiting.includes('Set an alert at $46.05.'));
assert.ok(!enterWaiting.includes('Continue monitoring. The setup is still developing.'));
assert.ok(enterWaiting.includes('Suggested Contract'));
assert.ok(enterWaiting.includes('Best Quality'));
assert.ok(enterWaiting.includes('Highest quality contract'));
assert.ok(enterWaiting.includes('$120.00'));
assert.ok(!enterWaiting.includes('Kairos Confidence'));
assert.ok(!enterWaiting.includes('View contract details'));

const reachedLive = htmlFor(setup({ price: 46.05, entryStatus: 'Tradeable', distanceFromEntryAtr: 0.1 }));
assert.ok(reachedLive.includes('data-execution-state="SETUP_CONFIRMED_ENTRY_REACHED"'));
assert.ok(reachedLive.includes('data-execute-visual-state="ready"'));
assert.ok(reachedLive.includes('execute-entry-ready'));
assert.ok(reachedLive.includes('Price is at the planned entry. You can execute this trade.'));

const reachedPotential = htmlFor(
  setup({ price: 46.05, entryStatus: 'Tradeable', distanceFromEntryAtr: 0.1 }),
  { available: false, source: 'unavailable', reason: 'no contract passed filters' }
);
assert.ok(reachedPotential.includes('execute-entry-ready'));
assert.ok(reachedPotential.includes('Price is at the planned entry.'));
assert.ok(reachedPotential.includes('Verify and select the live option contract before executing.'));

const almostReady = htmlFor(setup({ trade_eval: { trade_stage: 'BUILDING / WATCHLIST' }, setupGrade: 'B+ TRADEABLE', entryStatus: 'Near Entry', confirmationStarted: false }));
assert.ok(almostReady.includes('data-normalized-status="ALMOST_READY"'));
assert.ok(almostReady.includes('execute-not-ready'));
assert.ok(almostReady.includes('Setup is still developing. Wait for full confirmation.'));

const building = htmlFor(setup({ trade_eval: { trade_stage: 'BUILDING / WATCHLIST' }, setupGrade: 'B+ TRADEABLE', entryStatus: 'Waiting', confirmationStarted: false, direction: 'LONG' }));
assert.ok(building.includes('data-normalized-status="BUILDING"'));
assert.ok(building.includes('execute-not-ready'));
assert.ok(building.includes('Continue monitoring. The setup is still developing.'));

const loadingEarnings = htmlFor(setup({ earnings: { status: 'loading', started_at: new Date().toISOString() } }));
assert.ok(loadingEarnings.includes('Loading...'));
const staleLoadingEarnings = htmlFor(setup({ earnings: { loaded: false, loading: true, status: 'loading', date: null, days_until: null, source: 'background_refresh' } }));
assert.ok(staleLoadingEarnings.includes('Data unavailable'));
assert.ok(!staleLoadingEarnings.includes('Loading...'));
const datedEarnings = htmlFor(setup({
  earnings: { status: 'loaded', loaded: true, date: '2026-07-30', days_until: 17, source: 'cache' },
}));
assert.ok(datedEarnings.includes('Jul 30'));
assert.ok(datedEarnings.includes('17 days away'));
assert.ok(datedEarnings.includes('Setup confirmed. Wait for price to reach the planned entry at $46.05.'));

const topLevelEarnings = htmlFor(setup({ earningsDate: '2026-08-04', daysUntilEarnings: 22 }));
assert.ok(topLevelEarnings.includes('Aug 4'));
assert.ok(topLevelEarnings.includes('22 days away'));

const unavailableEarnings = htmlFor(setup({ earnings: { status: 'unavailable', loaded: true, date: null } }));
assert.ok(unavailableEarnings.includes('Earnings'));
assert.ok(unavailableEarnings.includes('Unavailable'));
assert.ok(!unavailableEarnings.includes('Loading...'));

const failedEarnings = htmlFor(setup({ earnings: { status: 'failed', error: 'provider timeout' } }));
assert.ok(failedEarnings.includes('Data unavailable'));
assert.ok(!failedEarnings.includes('Loading...'));

const singleTarget = htmlFor(setup({ tp1: 44, tp2: null, tp3: null, final_target: null }));
assert.ok(singleTarget.includes('<span>Target</span><span>$44.00</span>'));
assert.ok(!singleTarget.includes('<span>TP2</span>'));
assert.ok(!singleTarget.includes('<span>TP3</span>'));

const threeTargets = htmlFor(setup({ tp1: 45, tp2: 44, tp3: 43 }));
assert.ok(threeTargets.includes('<span>TP1</span><span>$45.00</span>'));
assert.ok(threeTargets.includes('<span>TP2</span><span>$44.00</span>'));
assert.ok(threeTargets.includes('<span>TP3</span><span>$43.00</span>'));
assert.ok(!threeTargets.includes('<span>Final Target</span><span>$43.00</span>'));

const finalTarget = htmlFor(setup({ tp1: 45, tp2: null, tp3: null, final_target: 42 }));
assert.ok(finalTarget.includes('<span>TP1</span><span>$45.00</span>'));
assert.ok(finalTarget.includes('<span>Final Target</span><span>$42.00</span>'));

const duplicateTargets = htmlFor(setup({ tp1: 44, tp2: 44, tp3: 43, final_target: 43 }));
assert.strictEqual((duplicateTargets.match(/<span>TP2<\/span>/g) || []).length, 0);
assert.strictEqual((duplicateTargets.match(/<span>Final Target<\/span>/g) || []).length, 0);
assert.ok(duplicateTargets.includes('<span>TP3</span><span>$43.00</span>'));

const snapshot = context.scannerSnapshotFromSetup(setup({
  tp1: 45,
  tp2: 44,
  tp3: 43,
  earnings: { loaded: true, date: '2026-07-30', days_until: 17, source: 'cache' },
}), { trade_stage: 'A+ READY' });
assert.strictEqual(snapshot.earnings_date, '2026-07-30');
assert.strictEqual(snapshot.days_until_earnings, 17);
assert.strictEqual(snapshot.earnings_source, 'cache');

// Opportunity Remaining is hidden until the existing analytics engine says confidence is sufficient.
assert.ok(!htmlFor(setup()).includes('Opportunity Remaining'));
context.window.KairosOpportunityAnalytics = {
  opportunityAnalytics: () => ({
    available: true,
    confidence: 'MODERATE',
    opportunityRemainingPct: 82.4,
    status: 'ON_TIME',
    calculationVersion: 'test',
  }),
};
const opportunityCard = htmlFor(setup());
assert.ok(opportunityCard.includes('Opportunity Remaining'));
assert.ok(opportunityCard.includes('<div class="opportunity-value">82%</div>'));
context.window.KairosOpportunityAnalytics = {
  opportunityAnalytics: () => ({
    available: false,
    confidence: 'INSUFFICIENT',
    opportunityRemainingPct: null,
    status: 'INSUFFICIENT_DATA',
  }),
};
assert.ok(!htmlFor(setup()).includes('Opportunity Remaining'));

// Contract candidate tiers render compactly and update the journal snapshot selection.
const tieredSetup = setup({
  setupGrade: 'A',
  best_contract: {
    available: true,
    type: 'PUT',
    strike: 45,
    expiry: '2026-08-21',
    bid: 7.9,
    ask: 8,
    mid: 7.95,
    spread: 0.1,
    candidate_audit: {
      current_selected_contract: { type: 'PUT', strike: 45, expiration: '2026-08-21', bid: 7.9, ask: 8, mid: 7.95, estimated_contract_cost: 800, score: 95, rejection_reasons: [] },
      best_quality_contract: { type: 'PUT', strike: 45, expiration: '2026-08-21', bid: 7.9, ask: 8, mid: 7.95, estimated_contract_cost: 800, score: 95, rejection_reasons: [] },
      best_balanced_contract: { type: 'PUT', strike: 44, expiration: '2026-08-21', bid: 5.4, ask: 5.5, mid: 5.45, estimated_contract_cost: 550, score: 88, rejection_reasons: [] },
      lowest_cost_acceptable_contract: { type: 'PUT', strike: 43, expiration: '2026-08-21', bid: 3.85, ask: 3.95, mid: 3.9, estimated_contract_cost: 395, score: 80, rejection_reasons: [] },
    },
  },
});
const tieredHtml = htmlFor(tieredSetup);
assert.ok(tieredHtml.includes('Best Quality'));
assert.ok(tieredHtml.includes('Balanced'));
assert.ok(tieredHtml.includes('Budget'));
assert.ok(tieredHtml.includes('$800.00'));
assert.ok(tieredHtml.includes('$550.00'));
assert.ok(tieredHtml.includes('$395.00'));
assert.ok(tieredHtml.includes('Aug 21 · $45 Put'));
const tieredId = context.setupIdFromSetup(tieredSetup);
context.selectContractTier(tieredId, 'balanced');
const balancedSnapshot = context.selectedContractFromSetup(tieredSetup);
assert.strictEqual(balancedSnapshot.strike_price, 44);
assert.strictEqual(balancedSnapshot.expiration_date, '2026-08-21');
assert.strictEqual(balancedSnapshot.option_ask_at_entry, 5.5);
assert.strictEqual(balancedSnapshot.premium_paid, 5.5);
assert.strictEqual(balancedSnapshot.contract_guidance_source, 'balanced');
assert.strictEqual(balancedSnapshot.contract_selection_source, 'balanced');
assert.strictEqual(balancedSnapshot.contract_tier, 'balanced');
assert.strictEqual(balancedSnapshot.contract_tier_label, 'Balanced');
assert.strictEqual(balancedSnapshot.estimated_contract_cost_at_entry, 550);

context.selectContractTier(tieredId, 'best_quality');
const bestQualitySnapshot = context.selectedContractFromSetup(tieredSetup);
assert.strictEqual(bestQualitySnapshot.contract_tier, 'best_quality');
assert.strictEqual(bestQualitySnapshot.contract_tier_label, 'Best Quality');
assert.strictEqual(bestQualitySnapshot.strike_price, 45);
assert.strictEqual(bestQualitySnapshot.estimated_contract_cost_at_entry, 800);

context.selectContractTier(tieredId, 'budget');
const budgetSnapshot = context.selectedContractFromSetup(tieredSetup);
assert.strictEqual(budgetSnapshot.contract_tier, 'budget');
assert.strictEqual(budgetSnapshot.contract_tier_label, 'Budget');
assert.strictEqual(budgetSnapshot.strike_price, 43);
assert.strictEqual(budgetSnapshot.estimated_contract_cost_at_entry, 395);

const multiContractSnapshot = context.selectedContractFromSetup({ ...tieredSetup, contracts: 2 });
assert.strictEqual(multiContractSnapshot.contract_tier, 'budget');
assert.strictEqual(multiContractSnapshot.contracts, 2);
assert.strictEqual(multiContractSnapshot.premium_paid, 3.95);
assert.strictEqual(multiContractSnapshot.estimated_contract_cost_at_entry, 790);

const scannerStatusSnapshot = context.scannerSnapshotFromSetup(tieredSetup, { trade_stage: 'B+ TRADEABLE', trigger_confirmed: true });
assert.strictEqual(scannerStatusSnapshot.journal_snapshot_version, 'options-v2');
assert.ok(scannerStatusSnapshot.snapshot_timestamp);
assert.strictEqual(scannerStatusSnapshot.scanner_status_raw, 'B+ TRADEABLE');
assert.strictEqual(scannerStatusSnapshot.scanner_status_normalized, 'WATCH');
assert.strictEqual(scannerStatusSnapshot.scanner_status, 'WATCH');
assert.strictEqual(scannerStatusSnapshot.trade_stage, 'ENTER_NOW');
assert.strictEqual(scannerStatusSnapshot.entry_trigger_state, 'TRIGGER_CONFIRMED');
assert.ok(scannerStatusSnapshot.entry_timing_state);

const cGradeReadySnapshot = context.scannerSnapshotFromSetup(
  { ...tieredSetup, setupGrade: 'C' },
  { trade_stage: 'A+ READY', trigger_confirmed: true }
);
assert.strictEqual(cGradeReadySnapshot.scanner_status_normalized, 'SKIP');
assert.strictEqual(cGradeReadySnapshot.scanner_status, 'SKIP');
assert.strictEqual(cGradeReadySnapshot.scanner_status_raw, 'A+ READY');

const staleContract = {
  available: true,
  source: 'option_chain',
  strike: 45,
  type: 'PUT',
  expiry: '2026-08-21',
  score: 82,
};
const normalizedCGradeRows = context.normalizeScanRowsForClient([
  { ticker: 'CGD', setupGrade: 'C', best_contract: staleContract },
]);
assert.strictEqual(normalizedCGradeRows[0].best_contract.available, false);
assert.strictEqual(normalizedCGradeRows[0].best_contract.source, 'not_evaluated');

const aGradeRow = { ticker: 'AGD', setupGrade: 'A', best_contract: staleContract };
const bGradeRow = { ticker: 'BGD', setupGrade: 'B', best_contract: staleContract };
const missingGradeRow = { ticker: 'MGD', best_contract: staleContract };
const normalizedEligibleRows = context.normalizeScanRowsForClient([aGradeRow, bGradeRow, missingGradeRow]);
assert.strictEqual(normalizedEligibleRows[0], aGradeRow);
assert.strictEqual(normalizedEligibleRows[1], bGradeRow);
assert.strictEqual(normalizedEligibleRows[2], missingGradeRow);
assert.strictEqual(normalizedEligibleRows[2].best_contract.available, true);

const migratedOld = context.migrateJournalEntry({
  ticker: 'OLD',
  optionType: 'PUT',
  strike: 55,
  expiry: '2026-08-21',
  askAtSelection: 2.5,
  contracts: 3,
  scannerStatus: 'WATCH',
});
assert.strictEqual(migratedOld.strike_price, 55);
assert.strictEqual(migratedOld.expiration_date, '2026-08-21');
assert.strictEqual(migratedOld.estimated_contract_cost_at_entry, 750);
assert.strictEqual(migratedOld.contract_tier_label, '');
assert.strictEqual(migratedOld.scanner_status_normalized, 'WATCH');

const tracked = context.updateTrackedSetupWithObservation({
  ticker: 'ATO',
  direction: 'SHORT',
  tracking_status: 'active',
  tracking_started_at: '2026-07-14T13:30:00Z',
  entry_price: 179.2,
  stop_price: 181,
  target_price: 176.15,
  plannedTp1: 176.15,
}, {
  observed_at: '2026-07-14T18:30:00Z',
  bar_time: '2026-07-14T18:30:00Z',
  price: 176.08,
  high: 179.25,
  low: 176.08,
});
assert.strictEqual(tracked.first_entry_touch_at, '2026-07-14T18:30:00Z');
assert.strictEqual(tracked.entry_reached_at, '2026-07-14T18:30:00Z');
assert.strictEqual(tracked.first_target_touch_at, '2026-07-14T18:30:00Z');
assert.strictEqual(tracked.tp1_reached_at, '2026-07-14T18:30:00Z');
assert.strictEqual(tracked.completion_reason, 'target');

console.log('Live card render v1 tests passed');
