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
    disabled: false,
    dataset: {},
    style: {},
    selectedOptions: [{ textContent: '' }],
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    addEventListener() {},
    removeEventListener() {},
    setAttribute() {},
    appendChild() {},
    remove() {},
    closest() { return null; },
  };
}

const storage = {};
const elements = {};
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
  Promise,
  performance: { now: () => 100 },
  URLSearchParams,
  encodeURIComponent,
  decodeURIComponent,
  setTimeout: () => 1,
  clearTimeout: () => {},
  setInterval: () => 1,
  clearInterval: () => {},
  alert: () => {},
  confirm: () => false,
  fetch: () => Promise.reject(new Error('network disabled in scanner summary frontend test')),
  localStorage: {
    getItem: key => storage[key] || null,
    setItem: (key, value) => { storage[key] = String(value); },
    removeItem: key => { delete storage[key]; },
  },
  document: {
    body: { appendChild() {} },
    getElementById: id => elements[id] || elementStub(),
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

elements.universeFilter = { ...elementStub(), value: 'discovered' };
assert.strictEqual(context.scannerScanUrl(), '/api/scan');
elements.universeFilter.value = 'default';
assert.strictEqual(context.scannerScanUrl(), '/api/scan?universe=default&view=summary');
assert.strictEqual(context.scannerScanUrl({ refresh: true }), '/api/scan?universe=default&view=summary&refresh=true');

const summarySetup = context.normalizeScanRowForClient({
  ticker: 'MO',
  setup_id: 'MO|4H|LONG|signal|68.39|66.88|gen-1',
  scan_generation: 'gen-1',
  timeframe: '4H',
  direction: 'LONG',
  price: 68.26,
  current_price: 68.26,
  entry: 68.39,
  planned_entry: 68.39,
  sl: 66.88,
  stop: 66.88,
  tp1: 71.41,
  tp2: 72.92,
  tp3: 74.43,
  setupGrade: 'A',
  status_bucket: 'ENTER_NOW',
  normalized_status_bucket: 'ENTER_NOW',
  display_status: 'ENTER_NOW',
  entryStatus: 'Tradeable',
  confirmationStarted: true,
  stockTrend: 'Bullish',
  stockLocation: 'Discount',
  ranking: { rank: 1, tier: 'TOP_OPPORTUNITY', score: 92 },
  earnings: { loaded: true, status: 'clear', source: 'fixture' },
  option: {
    type: 'CALL',
    strike: 68,
    expiration: '2026-09-18',
    expiry: '2026-09-18',
    bid: 0.95,
    ask: 0.99,
    mid: 0.97,
    volume: 120,
    open_interest: 500,
    estimated_contract_cost: 99,
    pricing_status: 'ready',
    pricing_quality: 'live_ask',
  },
  option_pricing: {
    status: 'ready',
    quality: 'live_ask',
    estimated_contract_cost: 99,
    bid: 0.95,
    ask: 0.99,
    mid: 0.97,
    volume: 120,
    open_interest: 500,
    type: 'CALL',
    strike: 68,
    expiration: '2026-09-18',
  },
  pricing_status: 'ready',
  pricing_quality: 'live_ask',
  lazy_hydration: {
    ticker: 'MO',
    type: 'CALL',
    strike: 68,
    expiration: '2026-09-18',
    setup_id: 'MO|4H|LONG|signal|68.39|66.88|gen-1',
    scan_generation: 'gen-1',
  },
});

const fullSetup = {
  ...summarySetup,
  trade_eval: { trade_stage: 'A+ READY', trigger_confirmed: true, displacement: 'Bullish' },
  quality: { grade: 'A', coach_note: 'clean pullback', rr: 2.0 },
};

const summaryHtml = context.renderCard(summarySetup);
const fullHtml = context.renderCard(fullSetup);
['MO', 'ENTER NOW', '$68.26', '$68.39', '$66.88', 'Budget Friendly'].forEach(fragment => {
  assert.ok(summaryHtml.includes(fragment), `summary card should include ${fragment}`);
  assert.ok(fullHtml.includes(fragment), `full card should include ${fragment}`);
});
assert.ok(summaryHtml.includes('Full setup details load when this section opens.'));

const detailKey = context.scannerDetailKey(summarySetup);
const targetId = `scannerDetail-${context.stableHash(detailKey)}`;
elements[targetId] = elementStub();
context.__summarySetup = summarySetup;
context.__fullSetup = fullSetup;
vm.runInContext('scannerRows = [__summarySetup]; scannerNearMiss = []; latestScannerMeta = { universe: "default", view: "summary" };', context);

let fetchCount = 0;
let fetchedUrl = '';
context.fetchWithTimeout = url => {
  fetchCount += 1;
  fetchedUrl = url;
  return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({ setup: fullSetup }) });
};

async function run() {
  const detailsEl = { open: true, dataset: { detailKey } };
  const first = context.handleScannerDetailToggle(detailsEl);
  const second = context.handleScannerDetailToggle({ open: true, dataset: { detailKey } });
  await Promise.all([first, second]);
  assert.strictEqual(fetchCount, 1, 'simultaneous detail opens should dedupe');
  assert.ok(fetchedUrl.includes('/api/scan/MO?'));
  assert.ok(fetchedUrl.includes('detail=full'));
  assert.ok(fetchedUrl.includes('setup_id='));
  assert.ok(elements[targetId].innerHTML.includes('Legacy status'));

  await context.fetchScannerDetail(summarySetup);
  assert.strictEqual(fetchCount, 1, 'reopening should use cached detail');

  let refreshCalled = false;
  vm.runInContext('runScan = () => { globalThis.__refreshCalled = true; return Promise.resolve(); };', context);
  context.__refreshCalled = false;
  context.fetchWithTimeout = () => Promise.resolve({
    ok: false,
    status: 409,
    json: () => Promise.resolve({ detail: { reason: 'stale_generation' } }),
  });
  const staleSetup = { ...summarySetup, setup_id: 'MO|stale|gen-1' };
  try {
    await context.fetchScannerDetail(staleSetup);
  } catch (error) {
    assert.strictEqual(error.reason, 'stale_generation');
    refreshCalled = context.__refreshCalled;
  }
  assert.strictEqual(refreshCalled, true, 'stale detail should trigger summary refresh');

  context.fetchWithTimeout = url => {
    fetchCount += 1;
    return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({ setup: fullSetup }) });
  };
  await context.fetchScannerDetail(summarySetup);
  assert.strictEqual(fetchCount, 2, 'stale generation should clear cached details');

  vm.runInContext('renderScannerResults = () => {}; runScan = () => {}; scannerDetailCache.set("manual", { ticker: "OLD" });', context);
  context.handleUniverseChange();
  assert.strictEqual(vm.runInContext('scannerDetailCache.size', context), 0, 'universe switch clears detail cache');

  console.log('scanner_summary_frontend_v1 passed');
}

run().catch(error => {
  console.error(error);
  process.exit(1);
});
