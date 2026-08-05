const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const executionGuidance = require('../public/execution_guidance.js');
const contractGuidance = require('../public/contract_guidance.js');
const cardData = require('../public/card_data.js');

const html = fs.readFileSync('public/index.html', 'utf8');
const inline = [...html.matchAll(/<script(?![^>]*src=)[^>]*>([\s\S]*?)<\/script>/gi)][0][1];

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
    scrollIntoView() {},
  };
}

const elements = new Map();
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
  confirm: () => false,
  fetch: () => Promise.reject(new Error('network disabled in mission workflow test')),
  localStorage: {
    getItem: key => storage[key] || null,
    setItem: (key, value) => { storage[key] = String(value); },
    removeItem: key => { delete storage[key]; },
  },
  sessionStorage: {
    getItem: key => storage[`session:${key}`] || null,
    setItem: (key, value) => { storage[`session:${key}`] = String(value); },
    removeItem: key => { delete storage[`session:${key}`]; },
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
  KairosContractGuidance: contractGuidance,
  KairosCardData: cardData,
  open: () => {},
  addEventListener: () => {},
  removeEventListener: () => {},
  matchMedia: () => ({ matches: false, addEventListener() {}, removeEventListener() {} }),
};
context.self = context.window;
context.globalThis = context;

vm.createContext(context);
vm.runInContext(inline, context);
vm.runInContext('latestScannerMeta = { stock_mission_workflow_v1: true, stock_event_memory_presentation_v1: false };', context);

function setup(overrides = {}) {
  return {
    ticker: 'AAA',
    timeframe: '4H',
    direction: 'LONG',
    price: 100,
    entry: 99,
    sl: 95,
    tp1: 108,
    tp2: 112,
    tp3: 118,
    setupGrade: 'A',
    signal_timestamp: '2026-08-05T14:00:00Z',
    ranking_status_bucket: 'ALMOST_READY',
    normalized_status_bucket: 'ALMOST_READY',
    ranking: {
      rank: 1,
      tier: 'TOP_OPPORTUNITY',
      score: 95,
      status_bucket: 'ALMOST_READY',
      priority_bucket: 1,
    },
    ...overrides,
  };
}

const almostReady = setup({ ticker: 'AR', ranking: { rank: 2, tier: 'HIGH_PRIORITY', score: 90, status_bucket: 'ALMOST_READY' } });
const earlyEntry = setup({ ticker: 'EE', ranking_status_bucket: 'EARLY_ENTRY', normalized_status_bucket: 'EARLY_ENTRY', ranking: { rank: 3, tier: 'HIGH_PRIORITY', score: 88, status_bucket: 'EARLY_ENTRY' }, execution_lifecycle_state: 'EARLY_ENTRY_BUILDING' });
const earlyTouch = setup({ ticker: 'ET', ranking_status_bucket: 'EARLY_ENTRY', normalized_status_bucket: 'EARLY_ENTRY', ranking: { rank: 4, tier: 'HIGH_PRIORITY', score: 86, status_bucket: 'EARLY_ENTRY' }, execution_lifecycle_state: 'EARLY_TOUCH' });
const waitingRetest = setup({ ticker: 'RT', execution_lifecycle_state: 'WAITING_FOR_RETEST', ranking: { rank: 5, tier: 'HIGH_PRIORITY', score: 84, status_bucket: 'ALMOST_READY' } });
const tradeNow = setup({ ticker: 'NOW', ranking_status_bucket: 'ENTER_NOW', normalized_status_bucket: 'ENTER_NOW', ranking: { rank: 1, tier: 'TOP_OPPORTUNITY', score: 97, status_bucket: 'ENTER_NOW' }, execution_lifecycle_state: 'ENTRY_TRIGGERED' });
const resolved = setup({ ticker: 'MISS', ranking_status_bucket: 'EARLY_ENTRY', normalized_status_bucket: 'EARLY_ENTRY', ranking: { rank: 6, tier: 'HIGH_PRIORITY', score: 82, status_bucket: 'EARLY_ENTRY' }, execution_lifecycle_state: 'MISSED_ENTRY' });

assert.strictEqual(context.stockMissionWorkflowEnabled(), true);
assert.strictEqual(context.stockEventMemoryPresentationEnabled(), true, 'mission flag activates lifecycle presentation labels');
assert.strictEqual(context.missionWorkflowBucket(almostReady), 'WATCH_CLOSELY');
assert.strictEqual(context.missionWorkflowBucket(earlyEntry), 'WATCH_CLOSELY');
assert.strictEqual(context.missionWorkflowBucket(earlyTouch), 'WATCH_CLOSELY');
assert.strictEqual(context.missionWorkflowBucket(waitingRetest), 'WATCH_CLOSELY');
assert.strictEqual(context.missionWorkflowBucket(tradeNow), 'TRADE_NOW');
assert.strictEqual(context.missionWorkflowBucket(resolved), 'RESOLVED');
const samePlanEarly = setup({ ticker: 'SAME', execution_lifecycle_state: 'EARLY_TOUCH' });
const samePlanRetest = setup({ ticker: 'SAME', execution_lifecycle_state: 'WAITING_FOR_RETEST' });
assert.strictEqual(context.missionSetupIdentity(samePlanEarly), context.missionSetupIdentity(samePlanRetest), 'same setup identity survives lifecycle-only changes');
const replacedPlan = setup({ ticker: 'SAME', entry: 101, execution_lifecycle_state: 'EARLY_TOUCH' });
assert.notStrictEqual(context.missionSetupIdentity(samePlanEarly), context.missionSetupIdentity(replacedPlan), 'material plan change gets a new identity');

const rows = [almostReady, earlyEntry, earlyTouch, waitingRetest, tradeNow, resolved, tradeNow];
const rankingBefore = JSON.stringify(rows.map(row => row.ranking));
const buckets = context.missionRows(rows);
assert.strictEqual(buckets.tradeNow.length, 1, 'no duplicate card across Trade Now');
assert.strictEqual(buckets.watchClosely.length, 4, 'resolved setup leaves Watch Closely');
assert.strictEqual(buckets.watchClosely[0].ticker, 'RT', 'Waiting for Retest is top Watch Closely priority');
assert.strictEqual(JSON.stringify(rows.map(row => row.ranking)), rankingBefore, 'mission helpers do not mutate frozen ranking');

context.renderMissionWorkflow(rows);
const htmlOut = getElement('missionWorkflow').innerHTML;
assert.ok(htmlOut.includes('Trade Now'));
assert.ok(htmlOut.includes('No fully confirmed trades right now') === false);
assert.ok(htmlOut.includes('Watch Closely'));
assert.ok(htmlOut.includes('WAITING FOR RETEST'));
assert.ok(!htmlOut.includes('MISS</span>'), 'resolved setup is not rendered as a current opportunity');

vm.runInContext('latestScannerMeta = { stock_mission_workflow_v1: false };', context);
context.renderMissionWorkflow(rows);
assert.strictEqual(getElement('missionWorkflow').style.display, 'none');

console.log('mission_workflow_v1.js passed');
