const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const executionGuidance = require('../public/execution_guidance.js');
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
  fetch: () => Promise.reject(new Error('network disabled in option contract test')),
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
  KairosCardData: cardData,
  open: () => {},
  addEventListener: () => {},
  removeEventListener: () => {},
};
context.self = context.window;
context.globalThis = context;

vm.createContext(context);
vm.runInContext(inline, context);

const dow = context.optionContractFieldsFromValues({
  optionSymbol: 'DOW260731P00029000',
  optionType: 'PUT',
  strike: 29,
  expiry: '2026-07-31',
  entryPremium: 0.21,
  quantity: 1,
  contractTier: 'Manual',
  stopPremium: 0.59,
  exitPremium: 0.59,
  exitReason: 'TP1',
  direction: 'SHORT',
}, { signalTimestamp: '2026-07-23T14:00:00Z' });
assert.strictEqual(dow.attached, true);
assert.strictEqual(dow.option_type, 'PUT');
assert.strictEqual(dow.option_entry_cost, 21);
assert.strictEqual(dow.option_breakeven_at_entry, 28.79);
assert.strictEqual(dow.option_exit_value, 59);
assert.strictEqual(dow.option_realized_pl, 38);
assert.ok(Math.abs(dow.option_realized_return_pct - 180.9523) < 0.01);

const normalizedDow = context.normalizeJournalOptionContract({ ticker: 'DOW', result: 'Win', ...dow });
assert.strictEqual(normalizedDow.compactLabel, 'Jul 31 $29 Put');
assert.strictEqual(normalizedDow.premiumLabel, 'Entry $0.21');
assert.strictEqual(normalizedDow.returnLabel, '+180.95%');

const positionPlan = context.positionPlanSide({
  ticker: 'DOW',
  direction: 'SHORT',
  result: 'Open',
  planned_underlying_entry: 30,
  actual_underlying_entry: 30,
  stop_price: 31,
  target_price: 28,
  ...dow,
});
assert.strictEqual(context.positionContractLabel(positionPlan), 'DOW Jul 31 $29 Put');
const contractHtml = context.renderActualContractSnapshot(positionPlan);
assert.ok(contractHtml.includes('Entry Premium'));
assert.ok(contractHtml.includes('$0.21'));
assert.ok(contractHtml.includes('Entry Cost'));
assert.ok(contractHtml.includes('$21.00'));
assert.ok(contractHtml.includes('Breakeven'));
assert.ok(contractHtml.includes('$28.79'));
assert.ok(contractHtml.includes('Realized P/L'));
assert.ok(contractHtml.includes('+$38.00'));
assert.ok(!contractHtml.includes('$0.00'));
assert.ok(!contractHtml.includes('N/A'));

const oldRecord = context.normalizeJournalOptionContract({ ticker: 'OLD', result: 'Open' });
assert.strictEqual(oldRecord.attached, false);
assert.strictEqual(oldRecord.compactLabel, 'No option contract attached');
assert.strictEqual(oldRecord.returnLabel, 'Unavailable');
assert.strictEqual(context.optionContractFieldsFromValues({ optionType: 'N/A', quantity: 1 }).attached, false);

const suggestedVsActual = context.migrateJournalEntry({
  ticker: 'UNP',
  direction: 'SHORT',
  option_type: 'PUT',
  suggested_option_type: 'PUT',
  suggested_option_strike: 300,
  suggested_option_expiration: '2026-07-31',
  suggested_option_tier: 'Budget',
  suggested_option_premium: 1.9,
  actual_option_type: 'PUT',
  actual_option_strike: 297.5,
  actual_option_expiration: '2026-08-07',
  actual_option_entry_premium: 2.15,
  actual_option_quantity: 1,
  option_contract_tier: 'Manual',
  entry: 300,
  plannedStop: 303,
  plannedTp1: 294,
});
assert.strictEqual(suggestedVsActual.suggested_option_tier, 'Budget');
assert.strictEqual(suggestedVsActual.actual_option_strike, 297.5);
assert.strictEqual(suggestedVsActual.option_entry_premium, 2.15);

context.saveJournal([context.migrateJournalEntry({ ticker: 'DOW', direction: 'SHORT', entry: 30, plannedStop: 31, plannedTp1: 28, ...dow })]);
const reloaded = context.loadJournal()[0];
assert.strictEqual(reloaded.option_symbol, 'DOW260731P00029000');
assert.strictEqual(reloaded.option_entry_cost, 21);
assert.strictEqual(reloaded.option_stop_premium, 0.59);

console.log('Option contract persistence v1 tests passed');
