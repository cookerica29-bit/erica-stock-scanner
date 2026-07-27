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
    scrollIntoView() {},
  };
}

const storage = {};
const session = {};
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
  fetch: () => Promise.reject(new Error('network disabled in active trade dashboard test')),
  localStorage: {
    getItem: key => storage[key] || null,
    setItem: (key, value) => { storage[key] = String(value); },
    removeItem: key => { delete storage[key]; },
  },
  sessionStorage: {
    getItem: key => session[key] || null,
    setItem: (key, value) => { session[key] = String(value); },
    removeItem: key => { delete session[key]; },
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

function entry(overrides = {}) {
  return context.migrateJournalEntry({
    id: overrides.id || Math.floor(Math.random() * 100000),
    ticker: 'DOW',
    direction: 'SHORT',
    result: 'Open',
    tracking_status: 'active',
    entry: 30,
    plannedStop: 31,
    plannedTp1: 28,
    first_entry_touch_at: '2026-07-27T14:00:00Z',
    current_price: 28.9,
    option_symbol: 'DOW260731P00029000',
    option_type: 'PUT',
    option_strike: 29,
    option_expiration: '2026-07-31',
    option_entry_premium: 0.21,
    option_quantity: 1,
    option_current_premium: 0.68,
    option_stop_premium: 0.59,
    option_contract_tier: 'Manual',
    updatedAt: '2026-07-27T15:00:00Z',
    ...overrides,
  });
}

function models(entries, latestRows = []) {
  return context.buildPositionModels(entries, new Map(latestRows.map(row => [row.ticker, row])));
}

const dow = models([entry({ id: 1 })])[0];
const dowVm = context.activeDashboardViewModel(dow);
assert.strictEqual(dowVm.ticker, 'DOW');
assert.strictEqual(dowVm.option_contract_label, 'DOW Jul 31 $29 Put');
assert.strictEqual(dowVm.option_unrealized_pl, 47);
assert.strictEqual(Number(dowVm.option_unrealized_return_pct.toFixed(2)), 223.81);
assert.strictEqual(dowVm.option_protected_pl, 38);
assert.strictEqual(Number(dowVm.option_protected_return_pct.toFixed(2)), 180.95);
assert.strictEqual(dowVm.option_unprotected_pl, 9);
assert.strictEqual(dowVm.needs_action, true);
assert.ok(dowVm.attention_reasons.some(reason => reason.code === 'EXPIRING_SOON'));

const losing = models([entry({
  id: 2,
  ticker: 'UNP',
  option_symbol: 'UNP260731P00300000',
  option_strike: 300,
  option_entry_premium: 2.15,
  option_current_premium: 1.5,
  option_stop_premium: 1.2,
  option_expiration: '2026-08-31',
  updatedAt: '2026-07-27T14:00:00Z',
})])[0];
const noStop = models([entry({
  id: 3,
  ticker: 'XOM',
  option_current_premium: 0.42,
  option_stop_premium: null,
  option_expiration: '2026-08-31',
})])[0];
const noCurrent = models([entry({
  id: 4,
  ticker: 'DVN',
  option_current_premium: null,
  option_stop_premium: 0.18,
  option_expiration: '2026-08-31',
})])[0];
const atStop = models([entry({
  id: 5,
  ticker: 'ENB',
  option_current_premium: 0.12,
  option_stop_premium: 0.12,
  option_expiration: '2026-08-31',
})])[0];
const noContract = models([entry({
  id: 6,
  ticker: 'OLD',
  option_symbol: '',
  option_type: 'N/A',
  option_strike: null,
  option_expiration: '',
  option_entry_premium: null,
  option_current_premium: null,
  option_stop_premium: null,
})])[0];
const scannerNoTrade = models([entry({
  id: 7,
  ticker: 'WPM',
  option_expiration: '2026-08-31',
})], [{
  ticker: 'WPM',
  direction: 'SHORT',
  setupGrade: 'C',
  entryStatus: 'Too Far',
  scannerStatus: 'NO TRADE',
}])[0];

const losingVm = context.activeDashboardViewModel(losing);
assert.strictEqual(losingVm.option_unrealized_pl, -65);
assert.strictEqual(Math.max(losingVm.option_protected_pl || 0, 0), 0);

const noStopVm = context.activeDashboardViewModel(noStop);
assert.ok(noStopVm.attention_reasons.some(reason => reason.code === 'MISSING_OPTION_STOP'));
assert.strictEqual(noStopVm.dashboard_status, 'Needs Attention');

const noCurrentVm = context.activeDashboardViewModel(noCurrent);
assert.ok(noCurrentVm.attention_reasons.some(reason => reason.code === 'MISSING_CURRENT_PREMIUM'));
assert.strictEqual(noCurrentVm.dashboard_status, 'Missing Data');

const atStopVm = context.activeDashboardViewModel(atStop);
assert.ok(atStopVm.attention_reasons.some(reason => reason.code === 'OPTION_STOP_TRIGGERED'));
assert.strictEqual(atStopVm.dashboard_status, 'At Risk');

const oldVm = context.activeDashboardViewModel(noContract);
assert.strictEqual(oldVm.dashboard_status, 'Missing Data');

const scannerVm = context.activeDashboardViewModel(scannerNoTrade);
assert.ok(context.currentScannerViewLabel(scannerNoTrade).includes('Grade C'));
assert.strictEqual(scannerVm.position_lifecycle.includes('NO TRADE'), false);

const closed = entry({ id: 8, result: 'Win', tracking_status: 'completed', completion_reason: 'target' });
assert.strictEqual(models([closed]).length, 0);

const allVms = [dow, losing, noStop, noCurrent, atStop, noContract, scannerNoTrade].map(context.activeDashboardViewModel);
const portfolio = context.activeDashboardPortfolio(allVms);
assert.strictEqual(portfolio.open_positions, 7);
assert.strictEqual(portfolio.current_option_pl, 41);
assert.strictEqual(portfolio.protected_profit, 76);
assert.strictEqual(portfolio.unprotected_profit, 39);
assert.strictEqual(portfolio.capital_in_open_trades, 320);
assert.strictEqual(portfolio.positions_needing_action, 5);

const sorted = context.sortActiveDashboardCards(allVms);
assert.strictEqual(sorted[0].ticker, 'ENB');

context.activeTradeDashboardFilter = 'missingData';
const missing = allVms.filter(vm => context.activeDashboardFilterMatches(vm, 'missingData')).map(vm => vm.ticker).sort();
assert.deepStrictEqual(missing, ['DVN', 'OLD']);

const htmlOut = context.renderActiveTradeDashboard(allVms.map(vm => vm.model));
assert.ok(htmlOut.includes('ACTIVE TRADES') || htmlOut.includes('Active Trades'));
assert.ok(htmlOut.includes('Current Return'));
assert.ok(htmlOut.includes('+$47.00'));
assert.ok(htmlOut.includes('+$38.00'));
assert.ok(htmlOut.includes('+$9.00') || htmlOut.includes('+9'));
assert.ok(htmlOut.includes('No option contract attached'));
assert.ok(htmlOut.includes('Current premium has not been updated.'));
assert.ok(htmlOut.includes('Option P/L') || htmlOut.includes('Option Performance'));

console.log('active_trade_dashboard_v1 passed');
