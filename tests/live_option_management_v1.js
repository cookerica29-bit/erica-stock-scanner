const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const executionGuidance = require('../public/execution_guidance.js');
const cardData = require('../public/card_data.js');

const html = fs.readFileSync('public/index.html', 'utf8');
const inline = [...html.matchAll(/<script(?![^>]*src=)[^>]*>([\s\S]*?)<\/script>/gi)][0][1];

const elements = new Map();
function elementStub(id = '') {
  return {
    id,
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
function getElement(id) {
  if (!elements.has(id)) elements.set(id, elementStub(id));
  return elements.get(id);
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
  performance: { now: () => 100 },
  URLSearchParams,
  encodeURIComponent,
  decodeURIComponent,
  setTimeout: () => 1,
  clearTimeout: () => {},
  setInterval: () => 1,
  clearInterval: () => {},
  alert: () => {},
  confirm: () => true,
  fetch: () => Promise.reject(new Error('network disabled in live option test')),
  localStorage: {
    getItem: key => storage[key] || null,
    setItem: (key, value) => { storage[key] = String(value); },
    removeItem: key => { delete storage[key]; },
  },
  document: {
    body: { appendChild() {} },
    getElementById: getElement,
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
  KairosCardData: cardData,
  open: () => {},
  addEventListener: () => {},
  removeEventListener: () => {},
};
context.self = context.window;
context.globalThis = context;

vm.createContext(context);
vm.runInContext(inline, context);

function optionEntry(overrides = {}) {
  return context.migrateJournalEntry({
    id: 101,
    ticker: 'DOW',
    direction: 'SHORT',
    result: 'Open',
    option_symbol: 'DOW260731P00029000',
    option_type: 'PUT',
    option_strike: 29,
    option_expiration: '2026-07-31',
    option_entry_premium: 0.21,
    option_quantity: 1,
    option_contract_tier: 'Manual',
    entry: 30,
    plannedStop: 31,
    plannedTp1: 28,
    ...overrides,
  });
}

function modelFor(entry) {
  return {
    id: entry.id,
    entry,
    plan: context.positionPlanSide(entry),
    live: context.positionLiveSide(entry, null),
  };
}

const dowOpen = optionEntry({ option_current_premium: 0.68, option_stop_premium: 0.59 });
const dowContract = context.normalizeJournalOptionContract(dowOpen);
assert.strictEqual(dowContract.option_current_value, 68);
assert.strictEqual(dowContract.option_unrealized_pl, 47);
assert.strictEqual(Number(dowContract.option_unrealized_return_pct.toFixed(2)), 223.81);
assert.strictEqual(dowContract.option_protected_value, 59);
assert.strictEqual(dowContract.option_protected_pl, 38);
assert.strictEqual(Number(dowContract.option_protected_return_pct.toFixed(2)), 180.95);

const dowHtml = context.renderOptionPositionSection(modelFor(dowOpen));
assert.ok(dowHtml.includes('OPTION POSITION') || dowHtml.includes('Option Position'));
assert.ok(dowHtml.includes('DOW Jul 31 $29 Put'));
assert.ok(dowHtml.includes('$68.00'));
assert.ok(dowHtml.includes('+$47.00'));
assert.ok(dowHtml.includes('+223.81%'));
assert.ok(dowHtml.includes('Estimated Protected Profit'));
assert.ok(dowHtml.includes('+$38.00'));
assert.ok(dowHtml.includes('+180.95%'));
assert.ok(dowHtml.includes('Based on the stop premium. Actual fill may differ.'));
assert.ok(!dowHtml.includes('Underlying Stop $0.59'));

const stopBelow = context.normalizeJournalOptionContract(optionEntry({ option_current_premium: 0.42, option_stop_premium: 0.12 }));
assert.strictEqual(context.optionPositionState({
  attached: true,
  entryPremium: stopBelow.option_entry_premium,
  currentPremium: stopBelow.option_current_premium,
  stopPremium: stopBelow.option_stop_premium,
}), 'OPTION STOP BELOW ENTRY');
assert.strictEqual(context.optionPositionGuidance({
  attached: true,
  entryPremium: 0.21,
  currentPremium: 0.42,
  stopPremium: 0.12,
}), 'Option stop remains below entry premium.');

assert.strictEqual(context.optionPositionState({ attached: true, entryPremium: 0.21, currentPremium: 0.05 }), 'OPTION LOSING');
assert.strictEqual(context.optionPositionGuidance({ attached: true, entryPremium: 0.21 }), 'Current premium has not been updated.');
assert.strictEqual(context.optionPositionState({ attached: false }), 'NO OPTION DATA');

const closedWinner = context.normalizeJournalOptionContract(optionEntry({
  result: 'Win',
  option_exit_premium: 0.59,
  option_exit_quantity: 1,
  option_exit_reason: 'Target reached',
  option_exit_timestamp: '2026-07-27T15:30:00Z',
}));
assert.strictEqual(closedWinner.option_exit_value, 59);
assert.strictEqual(closedWinner.option_realized_pl, 38);
assert.strictEqual(Number(closedWinner.option_realized_return_pct.toFixed(2)), 180.95);
assert.strictEqual(closedWinner.returnLabel, '+$38.00 · +180.95%');

const closedLoser = context.normalizeJournalOptionContract(optionEntry({ result: 'Loss', option_exit_premium: 0.04, option_exit_reason: 'Manual loss cut' }));
assert.strictEqual(closedLoser.option_realized_pl, -17);
assert.strictEqual(closedLoser.returnLabel, '-$17.00 · -80.95%');

const stopActualFill = context.normalizeJournalOptionContract(optionEntry({
  result: 'Win',
  option_stop_premium: 0.59,
  option_exit_premium: 0.52,
  option_exit_reason: 'Stop hit',
}));
assert.strictEqual(stopActualFill.option_protected_pl, 38);
assert.strictEqual(stopActualFill.option_realized_pl, 31);
assert.strictEqual(Number(stopActualFill.option_realized_return_pct.toFixed(2)), 147.62);

const missingQuantity = context.normalizeJournalOptionContract(optionEntry({ option_quantity: null, contracts: null, option_current_premium: 0.30 }));
assert.strictEqual(missingQuantity.option_quantity, 1);
assert.strictEqual(missingQuantity.option_current_value, 30);

const oldRecord = context.renderOptionPositionSection(modelFor({ id: 12, ticker: 'OLD', result: 'Open' }));
assert.ok(oldRecord.includes('No option contract attached'));
assert.ok(!oldRecord.includes('$0.00'));

context.localStorage.setItem('stock_scanner_journal', JSON.stringify([optionEntry({ id: 202 })]));
getElement('optionCurrent-202').value = '0.68';
getElement('optionStop-202').value = '0.59';
getElement('optionQty-202').value = '1';
getElement('optionNotes-202').value = 'Raised option stop after TP1.';
context.updateOptionPositionFromButton({ closest: () => ({ dataset: { positionId: '202' } }) }, 'save');
const saved = JSON.parse(storage.stock_scanner_journal)[0];
assert.strictEqual(saved.option_current_premium, 0.68);
assert.strictEqual(saved.option_current_value, 68);
assert.strictEqual(saved.option_unrealized_pl, 47);
assert.strictEqual(Number(saved.option_unrealized_return_pct.toFixed(2)), 223.81);
assert.strictEqual(saved.option_stop_premium, 0.59);
assert.strictEqual(saved.option_protected_pl, 38);
assert.ok(saved.option_update_notes.includes('Raised option stop'));

context.updateOptionPositionFromButton({ closest: () => ({ dataset: { positionId: '202' } }) }, 'clearCurrent');
const clearedCurrent = JSON.parse(storage.stock_scanner_journal)[0];
assert.strictEqual(clearedCurrent.option_current_premium, null);
assert.strictEqual(clearedCurrent.option_current_value, null);
assert.strictEqual(clearedCurrent.option_unrealized_pl, null);

context.updateOptionPositionFromButton({ closest: () => ({ dataset: { positionId: '202' } }) }, 'clearStop');
const clearedStop = JSON.parse(storage.stock_scanner_journal)[0];
assert.strictEqual(clearedStop.option_stop_premium, null);
assert.strictEqual(clearedStop.option_protected_value, null);
assert.strictEqual(clearedStop.option_protected_pl, null);

console.log('live_option_management_v1 passed');
