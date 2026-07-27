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
    remove() {},
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
  URLSearchParams,
  encodeURIComponent,
  decodeURIComponent,
  setTimeout: () => 1,
  clearTimeout: () => {},
  setInterval: () => 1,
  clearInterval: () => {},
  alert: () => {},
  confirm: () => false,
  fetch: () => { throw new Error('Position Intelligence must not fetch during deterministic evaluation'); },
  localStorage: {
    getItem: key => storage[key] || null,
    setItem: (key, value) => { storage[key] = String(value); },
    removeItem: key => { delete storage[key]; },
  },
  document: {
    body: { appendChild() {} },
    getElementById: () => elementStub(),
    querySelector: () => null,
    querySelectorAll: () => [],
    createElement: () => elementStub(),
    addEventListener() {},
    removeEventListener() {},
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

function position(overrides = {}) {
  return {
    id: 'pos-1',
    ticker: 'UAL',
    result: 'Open',
    tracking_status: 'active',
    direction: 'LONG',
    actual_underlying_entry: 100,
    planned_underlying_entry: 99,
    entry_price: 99,
    actual_option_premium: 2.75,
    stop_price: 95,
    target_price: 110,
    plannedTp2: 115,
    plannedTp3: 120,
    tracking_started_at: '2026-07-15T14:00:00Z',
    expected_hold_min_days: 7,
    expected_hold_max_days: 12,
    option_type: 'CALL',
    preferred_strike: 105,
    suggested_min_dte: 21,
    suggested_max_dte: 35,
    expected_move_dollars: 10,
    expected_move_percent: 10,
    option_plan_confidence: 4,
    ...overrides,
  };
}

function intel(entry, currentPrice) {
  return context.build_position_intelligence(entry, { current_price: currentPrice });
}

let healthy = intel(position(), 104);
assert.strictEqual(healthy.state, 'HEALTHY');
assert.strictEqual(healthy.next_action_label, 'Hold');
assert.strictEqual(Math.round(healthy.progress_to_tp1.raw_percent), 40);
assert.strictEqual(healthy.progress_to_tp1.display_percent, 40);
assert.strictEqual(healthy.position_opportunity_remaining, 60);
assert.ok(Math.abs(healthy.current_r - 0.8) < 0.001);

let short = intel(position({ direction: 'SHORT', option_type: 'PUT', actual_underlying_entry: 100, stop_price: 105, target_price: 90 }), 96);
assert.strictEqual(short.state, 'HEALTHY');
assert.strictEqual(Math.round(short.progress_to_tp1.raw_percent), 40);
assert.ok(Math.abs(short.current_r - 0.8) < 0.001);

let belowZero = intel(position(), 98);
assert.strictEqual(Math.round(belowZero.progress_to_tp1.raw_percent), -20);
assert.strictEqual(belowZero.progress_to_tp1.display_percent, 0);
assert.strictEqual(belowZero.position_opportunity_remaining, 100);
assert.strictEqual(belowZero.state, 'WATCH');
assert.strictEqual(belowZero.reason_code, 'POSITION_BELOW_ENTRY');

let beyondTp1 = intel(position(), 112);
assert.strictEqual(Math.round(beyondTp1.progress_to_tp1.raw_percent), 120);
assert.strictEqual(beyondTp1.progress_to_tp1.display_percent, 100);
assert.strictEqual(beyondTp1.position_opportunity_remaining, 0);
assert.strictEqual(beyondTp1.tp1_reached, true);
assert.strictEqual(beyondTp1.state, 'PROTECT');
assert.strictEqual(beyondTp1.reason_code, 'TP1_REACHED');

let longExit = intel(position(), 95);
assert.strictEqual(longExit.state, 'EXIT');
assert.strictEqual(longExit.next_action_label, 'Exit');
assert.strictEqual(longExit.reason_code, 'STOP_INVALIDATED');

let shortExit = intel(position({ direction: 'SHORT', option_type: 'PUT', actual_underlying_entry: 100, stop_price: 105, target_price: 90 }), 105);
assert.strictEqual(shortExit.state, 'EXIT');

let watchRetrace = intel(position({ position_best_price: 105 }), 102.5);
assert.strictEqual(watchRetrace.state, 'WATCH');
assert.strictEqual(watchRetrace.reason_code, 'RETRACE_AFTER_40');

let protectRetrace = intel(position({ position_best_price: 108 }), 105);
assert.strictEqual(protectRetrace.state, 'PROTECT');
assert.strictEqual(protectRetrace.reason_code, 'LARGE_RETRACE_AFTER_70');

let priorityExit = intel(position({ position_best_price: 112 }), 94);
assert.strictEqual(priorityExit.state, 'EXIT', 'hard stop invalidation must outrank profit-protection state');

assert.strictEqual(intel(position({ direction: '', option_type: '', optionType: '' }), 104).state, 'DATA_NEEDED');
assert.ok(intel(position({ actual_underlying_entry: null, planned_underlying_entry: null, entry_price: null, entry: null }), 104).reason[0].includes('underlying entry price'));
assert.ok(intel(position({ stop_price: null, plannedStop: null }), 104).reason[0].includes('original stop'));
assert.ok(intel(position({ target_price: null, plannedTp1: null, tp1: null }), 104).reason[0].includes('tp1'));
assert.strictEqual(intel(position({ stop_price: 101 }), 104).reason[0], 'Kairos cannot calculate position health because invalid stop geometry.');
assert.strictEqual(intel(position({ direction: 'SHORT', stop_price: 99, target_price: 90 }), 96).reason[0], 'Kairos cannot calculate position health because invalid stop geometry.');
assert.strictEqual(intel(position({ target_price: 99 }), 104).reason[0], 'Kairos cannot calculate position health because invalid target geometry.');
assert.strictEqual(intel(position(), null).reason[0], 'Kairos cannot calculate position health because current price unavailable.');

const premiumTrap = intel(position({
  actual_underlying_entry: null,
  planned_underlying_entry: null,
  entry_price: null,
  entry: null,
  actual_option_premium: 1.25,
}), 104);
assert.strictEqual(premiumTrap.state, 'DATA_NEEDED');
assert.ok(premiumTrap.reason[0].includes('underlying entry price'), 'option premium must never be used as underlying entry');

const preservedPlan = position({ actual_underlying_entry: 101, planned_underlying_entry: 100, entry_price: 99, stop_price: 95, target_price: 110 });
const beforePlan = JSON.stringify({ entry: preservedPlan.entry_price, stop: preservedPlan.stop_price, target: preservedPlan.target_price });
const computed = intel(preservedPlan, 104);
assert.strictEqual(computed.entry_price, 101);
assert.strictEqual(JSON.stringify({ entry: preservedPlan.entry_price, stop: preservedPlan.stop_price, target: preservedPlan.target_price }), beforePlan);

storage.stock_scanner_journal = JSON.stringify([position({
  id: 'hist-1',
  ticker: 'HIST',
  position_last_state: 'HEALTHY',
  position_best_price: 105,
  position_state_history: [],
})]);
vm.runInContext(`scannerRows = [{ ticker: 'HIST', current_quote_price: 102.5, direction: 'LONG' }]; scannerNearMiss = [];`, context);
let models = context.buildPositionModels();
let sync = context.syncPositionIntelligenceHistory(models);
assert.strictEqual(sync.transitionsCreated, 1);
assert.strictEqual(sync.deduplicated, 0);
let saved = JSON.parse(storage.stock_scanner_journal)[0];
assert.strictEqual(saved.position_last_state, 'WATCH');
assert.strictEqual(saved.position_state_history.length, 3, 'state change plus 25% and 50% max-progress milestones should be saved');
const firstHistory = saved.position_state_history.length;
models = context.buildPositionModels();
sync = context.syncPositionIntelligenceHistory(models);
saved = JSON.parse(storage.stock_scanner_journal)[0];
assert.strictEqual(saved.position_state_history.length, firstHistory);
assert.ok(sync.deduplicated >= 1);

storage.stock_scanner_journal = JSON.stringify([position({ id: 'render-1', ticker: 'REND' })]);
vm.runInContext(`scannerRows = [{ ticker: 'REND', current_quote_price: 106, direction: 'LONG' }]; scannerNearMiss = [];`, context);
const positions = context.buildPositionModels();
const rendered = context.renderPositionManagerBlock(positions[0]);
assert.ok(rendered.includes('📊 Position Manager'));
assert.ok(rendered.includes('Progress to TP1'));
assert.ok(rendered.includes('Current R'));
assert.ok(rendered.includes('Opportunity Remaining'));
assert.ok(rendered.includes('Next Action: Hold'));
assert.ok(rendered.includes('Position Details'));
assert.ok(rendered.includes('Option Plan Snapshot'));
assert.ok(rendered.includes('Actual Robinhood Contract'));

const diag = context.positionDiagnostics(positions);
assert.strictEqual(diag.open_positions_evaluated, 1);
assert.strictEqual(diag.healthy_positions, 1);

assert.strictEqual(typeof context._best_contract, 'undefined', 'frontend Position Intelligence must not call live option-chain helpers');
assert.strictEqual(context.evaluatePositionAlertNotifications().sent, 0, 'Position Intelligence must not send Telegram or management alerts');

console.log('Position Intelligence v1 tests passed');
