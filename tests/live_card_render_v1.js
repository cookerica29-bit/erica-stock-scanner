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
  fetch: () => Promise.reject(new Error('network disabled in render test')),
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
    option_plan: {
      available: true,
      type: 'PUT',
      preferred_strike: 45,
      suggested_expiration: { min_dte: 21, max_dte: 35, label: '21–35 DTE' },
      expected_hold: { min_trading_days: 7, max_trading_days: 12, label: '7–12 Trading Days' },
      expected_move: { dollars: -2.05, percent: -4.5, label: '-$2.05 (-4.5%)' },
      confidence: { stars: 4, label: '★★★★☆' },
      source: 'kairos_trade_plan',
    },
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
assert.ok(!enterWaiting.includes('price-meta'), 'Scanner cards should not render redundant readiness stars/status meta');
assert.ok(!enterWaiting.includes('class="price-meta"'), 'Scanner cards should not repeat status beside the price');
assert.strictEqual(context.scanStatusText({ cache: 'hit', age_seconds: 16 }), '');
assert.strictEqual(context.scanStatusText({ cache: 'refresh', age_seconds: 16 }), '');
assert.strictEqual(context.scanStatusText({ cache: 'hit', age_seconds: 180, stale: true }), '');
assert.strictEqual(context.scanStatusText({ cache: 'hit', age_seconds: 180, refreshing: true }), 'Refreshing cached data...');
assert.strictEqual(context.scanStatusText({ cache: 'hit', age_seconds: 180, last_refresh_error: 'timeout', refreshing: false }), 'Refresh failed');
assert.ok(!context.scanStatusText({ cache: 'hit', age_seconds: 16 }).includes('Last updated'));
let fetchOptions = null;
const originalFetch = context.fetch;
context.AbortController = class {
  constructor() {
    this.signal = {};
  }
  abort() {}
};
context.fetch = (_url, options) => {
  fetchOptions = options;
  return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
};
context.fetchWithTimeout('/api/scan', {}, 12000);
assert.strictEqual(fetchOptions.cache, 'no-store');
context.fetch = originalFetch;
const originalGetElementByIdForUniverse = context.document.getElementById;
const universeElements = {
  universeFilter: { ...elementStub(), value: 'discovered' },
};
context.document.getElementById = id => universeElements[id] || elementStub();
assert.strictEqual(context.currentScannerUniverse(), 'discovered');
assert.strictEqual(context.scannerScanUrl(), '/api/scan?view=summary');
assert.strictEqual(context.scannerScanUrl({ refresh: true }), '/api/scan?view=summary&refresh=true');
universeElements.universeFilter.value = 'default';
assert.strictEqual(context.currentScannerUniverse(), 'default');
assert.strictEqual(context.scannerScanUrl(), '/api/scan?universe=default&view=summary');
assert.strictEqual(context.scannerScanUrl({ refresh: true }), '/api/scan?universe=default&view=summary&refresh=true');
context.document.getElementById = originalGetElementByIdForUniverse;
assert.ok(html.includes('id="verifiedHistorySummary"'), 'Analytics should include Verified History summary panel');
assert.ok(html.includes('Verified History Diagnostics'), 'Diagnostics should include Verified History diagnostics');
const verifiedHistoryEl = elementStub();
const verifiedHistoryDiag = elementStub();
const originalGetElementByIdForHistory = context.document.getElementById;
context.document.getElementById = id => ({
  verifiedHistorySummary: verifiedHistoryEl,
  verifiedHistoryDiagnostics: verifiedHistoryDiag,
}[id] || elementStub());
context.renderVerifiedHistoryPayload({
  version: 'verified-history-v1',
  summary: {
    open_bucket: 4,
    processing_bucket: 0,
    needs_attention_bucket: 1,
    needs_review: 1,
    verified: 0,
    trade_intelligence_eligible: 0,
  },
  reconciliation: { journal_reconciled: true, completed_reconciled: true, unclassified_record_count: 0 },
  diagnostics: { replay_jobs_by_status: {}, duplicate_active_job_count: 0 },
  records: [{
    ticker: 'OXY',
    pipeline_status: 'NEEDS_REVIEW',
    explanation: 'Journal and replay disagree.',
    next_step: 'Review the journal result against replay evidence.',
    verification: { journal_result: 'Win', journal_outcome: 'TP1', replay_result: 'Loss', replay_outcome: 'STOP_DETECTED' },
    replay_summary: { outcome_category: 'STOP_DETECTED' },
    trade_intelligence_eligible: false,
  }],
});
assert.ok(verifiedHistoryEl.innerHTML.includes('Verified History'));
assert.ok(verifiedHistoryEl.innerHTML.includes('Needs Attention'));
assert.ok(verifiedHistoryEl.innerHTML.includes('OXY'));
assert.ok(verifiedHistoryEl.innerHTML.includes('Needs Review'));
assert.ok(verifiedHistoryEl.innerHTML.includes('Journal and replay disagree.'));
assert.ok(verifiedHistoryDiag.innerHTML.includes('Duplicate active jobs'));
context.document.getElementById = originalGetElementByIdForHistory;
assert.ok(html.includes('data-tab="positions"') && html.includes('Active Trades'), 'Navigation should include Active Trades via the Positions tab');
assert.ok(html.includes('id="active-trades-panel"'), 'Active Trade Workspace panel should be present');
const activeTradesSummary = elementStub();
const activeTradesBody = elementStub();
const originalGetElementByIdForActiveTrades = context.document.getElementById;
context.document.getElementById = id => ({
  activeTradesSummary,
  activeTradesBody,
}[id] || elementStub());
delete storage.kairos_journal_admin_token;
context.renderActiveTradeWorkspace();
assert.ok(activeTradesBody.innerHTML.includes('Enter the journal admin token'));
storage.kairos_journal_admin_token = 'test-token';
context.__activeTradePayload = {
  summary: {
    active_records_found: 1,
    entered_positions: 0,
    tracked_but_not_entered: 1,
    need_attention: 0,
    awaiting_replay: 0,
    needs_review: 0,
  },
  records: [{
    id: 'pos-dow-1',
    journal_id: 'journal-dow-1',
    position_id: 'pos-dow-1',
    ticker: 'DOW',
    direction: 'SHORT',
    tracking_state: 'WATCHING_FOR_ENTRY',
    status_guidance: {
      label: 'Watching for planned entry',
      what_is_happening: 'This setup is being tracked, but no position has been recorded.',
      what_to_watch: 'Use the stored entry, stop, targets, and Position Intelligence state.',
      what_happens_next: 'Watch for the planned entry and keep the trade plan available.',
    },
    grade: 'A',
    timeframe: '4H',
    plan: {
      ticker: 'DOW',
      direction: 'SHORT',
      timeframe: '4H',
      grade: 'A',
      planned_entry: 31.2,
      actual_entry: null,
      stop: 32.15,
      tp1: 29.5,
      tp2: 28.7,
      tp3: 27.9,
      opportunity_remaining: 76,
      initial_rr: 2.4,
    },
    contract: {
      instrument_type: 'option',
      option_type: 'PUT',
      strike: 31,
      expiration: '2026-08-21',
      quantity: 1,
      entry_premium: 1.25,
      actual_option_pnl: null,
    },
    attention_items: ['TRACKED_NOT_ENTERED'],
    entered: false,
    completed: false,
    verified_history: { pipeline_status: 'OPEN', verification_status: 'NOT_APPLICABLE', trade_intelligence_eligible: false },
  }],
};
vm.runInContext('activeTradeWorkspaceState = { loading: false, error: null, token: "test-token", payload: globalThis.__activeTradePayload, selectedId: null, details: new Map() };', context);
context.renderActiveTradeWorkspace();
assert.ok(activeTradesSummary.innerHTML.includes('Open Trades'));
assert.ok(activeTradesBody.innerHTML.includes('DOW'));
assert.ok(activeTradesBody.innerHTML.includes('Watching for planned entry'));
context.__activeTradeDetail = {
  ...context.__activeTradePayload.records[0],
  guided_chart: {
    symbol: 'DOW',
    direction: 'SHORT',
    timeframe: '4H',
    current_price: 31.4,
    planned_entry: 31.2,
    actual_entry: null,
    stop: 32.15,
    targets: [29.5, 28.7, 27.9],
  },
  position_intelligence: {
    last_state: 'HEALTHY',
    best_price: 31.4,
    max_progress_percent: 0,
    tp1_reached: false,
    state_history: [],
  },
  timeline: [{ label: 'Setup Found', timestamp: '2026-07-23T14:00:00Z', state: 'complete' }],
  trade_intelligence: {
    available: false,
    message: 'Not enough verified historical data yet.',
    exact_progress: { current: 0, required: 30, label: '0 of 30' },
    broader_progress: { current: 0, required: 100, label: '0 of 100' },
  },
};
vm.runInContext('activeTradeWorkspaceState.details.set("pos-dow-1", globalThis.__activeTradeDetail);', context);
context.renderActiveTradeWorkspace();
assert.ok(activeTradesBody.innerHTML.includes('View Trade Chart'));
assert.ok(activeTradesBody.innerHTML.includes('Position Intelligence'));
assert.ok(activeTradesBody.innerHTML.includes('Historical Intelligence'));
assert.ok(activeTradesBody.innerHTML.includes('Underlying plan levels and option performance are kept separate.'));
assert.ok(activeTradesBody.innerHTML.includes('Complete Trade is available only after entry evidence exists.'));
context.document.getElementById = originalGetElementByIdForActiveTrades;
assert.ok(enterWaiting.includes('execute-waiting-entry'));
assert.ok(enterWaiting.includes('Setup confirmed. Wait for price to reach the planned entry at $46.05.'));
assert.ok(enterWaiting.includes('Set an alert at $46.05.'));
assert.ok(enterWaiting.includes('<div class="index-simple-label">Why</div>'));
assert.ok(enterWaiting.includes('Short setup with bearish trend; entry area is active.'));
assert.ok(!enterWaiting.includes('Short setup with bearish trend; Setup confirmed. Wait'), 'Why summary should not duplicate Next Step wording verbatim');
assert.ok(!enterWaiting.includes('Continue monitoring. The setup is still developing.'));
assert.ok(enterWaiting.includes('🎯 Option Plan'));
assert.ok(enterWaiting.includes('🔴 PUT'));
assert.ok(enterWaiting.includes('Preferred Strike'));
assert.ok(enterWaiting.includes('$45.00'));
assert.ok(enterWaiting.includes('21–35 DTE'));
assert.ok(enterWaiting.includes('7–12 Trading Days'));
assert.ok(enterWaiting.includes('-$2.05 (-4.5%)'));
assert.ok(enterWaiting.includes('★★★★☆'));
assert.ok(!enterWaiting.includes('Suggested Contract'));
assert.ok(!enterWaiting.includes('Best Quality'));
assert.ok(!enterWaiting.includes('top pick'));
assert.ok(enterWaiting.includes('Budget Friendly'));
assert.ok(enterWaiting.includes('Est. $120.00'));
assert.ok(!enterWaiting.includes('Kairos Confidence'));
assert.ok(!enterWaiting.includes('View contract details'));
assert.ok(enterWaiting.includes('data-plan-visual="risk-reward"'));
assert.ok(enterWaiting.includes('>Now</text>'));
assert.ok(enterWaiting.includes('>Target</text>'));
assert.ok(enterWaiting.includes("openTerms('term-risk-reward')"));

const longBar = context.riskRewardBarData({
  direction: 'LONG',
  price: 103,
  entry: 100,
  sl: 95,
  tp1: 110,
});
assert.strictEqual(longBar.positions.stop, 0);
assert.ok(Math.abs(longBar.positions.entry - 33.3333) < 0.01);
assert.ok(Math.abs(longBar.positions.current - 53.3333) < 0.01);
assert.strictEqual(longBar.positions.tp1, 100);
assert.ok(longBar.rawXPositions);
assert.ok(longBar.xPositions);

const shortBar = context.riskRewardBarData({
  direction: 'SHORT',
  price: 97,
  entry: 100,
  sl: 105,
  tp1: 90,
});
assert.strictEqual(shortBar.positions.stop, 0);
assert.ok(Math.abs(shortBar.positions.entry - 33.3333) < 0.01);
assert.ok(Math.abs(shortBar.positions.current - 53.3333) < 0.01);
assert.strictEqual(shortBar.positions.tp1, 100);

const closeEntryNowBar = context.riskRewardBarData({
  direction: 'SHORT',
  price: 170.06,
  entry: 169.05,
  sl: 172.15,
  tp1: 162.85,
});
assert.ok(Math.abs(closeEntryNowBar.positions.entry - 33.3333) < 0.01);
assert.ok(closeEntryNowBar.positions.current < closeEntryNowBar.positions.entry);
assert.ok(Math.abs(closeEntryNowBar.rawXPositions.current - closeEntryNowBar.rawXPositions.entry) < 28);
assert.ok(Math.abs(closeEntryNowBar.xPositions.current - closeEntryNowBar.xPositions.entry) >= 15);

const screenshotExampleBar = context.riskRewardBarData({
  direction: 'LONG',
  price: 495.76,
  entry: 505.27,
  sl: 472.15,
  tp1: 571.51,
  tp2: 604.63,
  tp3: 637.75,
});
assert.ok(Math.abs(screenshotExampleBar.rawXPositions.current - screenshotExampleBar.rawXPositions.entry) < 15);
assert.ok(Math.abs(screenshotExampleBar.xPositions.current - screenshotExampleBar.xPositions.entry) >= 15);
assert.ok(screenshotExampleBar.xPositions.current < screenshotExampleBar.xPositions.entry);
const closeEntryNowHtml = htmlFor(setup({ price: 170.06, entry: 169.05, sl: 172.15, tp1: 162.85 }));
assert.ok(closeEntryNowHtml.includes('viewBox="0 0 300 86"'));
assert.ok(closeEntryNowHtml.includes('x="24" y="8">Risk</text>'));
assert.ok(closeEntryNowHtml.includes('x="276" y="8" text-anchor="end">Reward</text>'));
assert.ok(closeEntryNowHtml.includes('y="24" text-anchor="middle">Entry</text>'));
assert.ok(closeEntryNowHtml.includes('y="80" text-anchor="middle">Now</text>'));
const riskAxisY = Number(closeEntryNowHtml.match(/<text class="rr-axis-label" x="24" y="([^"]+)">Risk<\/text>/)[1]);
const closeEntryLabelY = Number(closeEntryNowHtml.match(/<text class="rr-point-label" x="[^"]+" y="([^"]+)" text-anchor="middle">Entry<\/text>/)[1]);
assert.ok(closeEntryLabelY - riskAxisY >= 12, 'close Entry label must clear the Risk/Reward axis row');

assert.ok(enterWaiting.includes('View Trade Chart'), 'Scanner setup cards should expose the guided trade chart expander');
assert.ok(enterWaiting.includes('Planned Entry'));
assert.ok(enterWaiting.includes('Stop Loss'));
assert.ok(enterWaiting.includes('First Target'));
assert.ok(enterWaiting.includes('Price has not reached the planned entry yet.'));
assert.ok(enterWaiting.includes('Set an alert at the planned entry and wait.'));
assert.ok(enterWaiting.includes('Why Kairos Likes This Trade'));

const guidedLongPlan = context.guidedChartPlan(setup({
  ticker: 'LONG',
  direction: 'LONG',
  price: 101,
  entry: 100,
  sl: 95,
  tp1: 110,
  tp2: 115,
  tp3: 120,
}));
assert.strictEqual(guidedLongPlan.direction, 'LONG');
assert.ok(guidedLongPlan.stop < guidedLongPlan.entry, 'Long guided chart must keep stop below entry');
assert.ok(guidedLongPlan.targets.every(([, value]) => value > guidedLongPlan.entry), 'Long guided chart must keep targets above entry');
const guidedLongSvg = context.renderGuidedTradeChartSvg(guidedLongPlan, [
  { timestamp: '2026-07-20T14:00:00Z', open: 98, high: 102, low: 97, close: 101 },
  { timestamp: '2026-07-20T18:00:00Z', open: 101, high: 104, low: 100, close: 103 },
]);
assert.ok(guidedLongSvg.includes('Planned Entry'));
assert.ok(guidedLongSvg.includes('Current Price'));
assert.ok(guidedLongSvg.includes('guided-risk-zone'));
assert.ok(guidedLongSvg.includes('guided-reward-zone'));
assert.ok(!guidedLongSvg.includes('NaN'), 'Guided chart SVG should not emit NaN coordinates');

const guidedShortPlan = context.guidedChartPlan(setup({
  ticker: 'SHORT',
  direction: 'SHORT',
  price: 99,
  entry: 100,
  sl: 105,
  tp1: 90,
  tp2: 85,
  tp3: 80,
}));
assert.strictEqual(guidedShortPlan.direction, 'SHORT');
assert.ok(guidedShortPlan.stop > guidedShortPlan.entry, 'Short guided chart must keep stop above entry');
assert.ok(guidedShortPlan.targets.every(([, value]) => value < guidedShortPlan.entry), 'Short guided chart must keep targets below entry');
assert.ok(context.guidedPriceContext(guidedShortPlan).includes('moving toward TP1'));

const guidedMissingTargets = context.renderGuidedTradeChartBlock(setup({ tp1: 44, tp2: null, tp3: null }), 'setup');
assert.ok(guidedMissingTargets.includes('First Target'));
assert.ok(!guidedMissingTargets.includes('TP2'));
assert.ok(!guidedMissingTargets.includes('TP3'));
assert.ok(!guidedMissingTargets.includes('null'));
assert.ok(!guidedMissingTargets.includes('NaN'));

const guidedWaiting = context.renderGuidedTradeChartBlock(setup({
  trade_eval: { trade_stage: 'BUILDING / WATCHLIST' },
  confirmationStarted: false,
  entryStatus: 'Waiting',
  guided_status: 'Waiting for Planned Entry',
}), 'setup');
assert.ok(guidedWaiting.includes('Wait for the planned entry or confirmation.'));
assert.ok(!guidedWaiting.includes('Review the option plan before executing.'));

const guidedMismatch = context.renderGuidedTradeChartBlock(setup({
  result: 'Win',
  outcome: 'TP1',
  analytics_verification: {
    status: 'JOURNAL_REPLAY_MISMATCH',
    journal_result: 'Win',
    journal_outcome: 'TP1',
    replay_result: 'Loss',
    replay_outcome: 'STOP_DETECTED',
  },
}), 'completed');
assert.ok(guidedMismatch.includes('Completed'));
assert.ok(guidedMismatch.includes('Review the trade outcome.'));
assert.ok(guidedMismatch.includes('Needs Review: journal outcome and replay evidence do not agree.'));
assert.ok(!guidedMismatch.includes('Verified by replay'));
assert.ok(context.formatSignalTimestamp('2026-07-20T14:08:00Z').includes('2026') === false);
assert.ok(guidedMismatch.includes('Historical Intelligence'));

const tradeIntelEmpty = context.renderTradeIntelligenceInsight({
  available: false,
  message: 'Not enough verified historical data yet.',
  exact_match_count: 2,
  broader_match_count: 14,
  thresholds: { exact_min_trades: 30, broad_min_trades: 100 },
});
assert.ok(tradeIntelEmpty.includes('Not enough verified historical data yet.'));
assert.ok(tradeIntelEmpty.includes('2 of the 30 verified similar trades required for this exact group.'));
assert.ok(tradeIntelEmpty.includes('Exact group: 2 of 30'));
assert.ok(tradeIntelEmpty.includes('Broader comparison: 14 of 100'));
assert.ok(!tradeIntelEmpty.includes('TP1 Success'), 'insufficient Trade Intelligence should not show premature rates');

const tradeIntelReady = context.renderTradeIntelligenceInsight({
  available: true,
  sample_size: 84,
  verified_trades: 84,
  metrics: {
    tp1_rate: 76,
    average_time_to_tp1_trading_days: 2.4,
    average_maximum_drawdown_r: 0.58,
    average_r: 1.82,
  },
  what_usually_happens: [
    { label: 'Reached TP1 before meaningful pullback', percent: 68, count: 57 },
    { label: 'Stopped before TP1', percent: 13, count: 11 },
  ],
  confidence_drivers: ['A-grade setups outperformed the verified baseline.'],
});
assert.ok(tradeIntelReady.includes('Verified Similar Trades: 84'));
assert.ok(tradeIntelReady.includes('TP1 Success'));
assert.ok(tradeIntelReady.includes('76%'));
assert.ok(tradeIntelReady.includes('Sample Size: 84 verified trades'));

const tradeIntelPayload = context.tradeIntelligenceSubjectPayload(setup({
  ticker: 'OXY',
  direction: 'SHORT',
  setupGrade: 'A',
  scanner_status_normalized: 'ENTER_NOW',
  scanner_timeframe: '4H',
  setupLocation: 'premium',
  confirmationStarted: true,
}));
assert.strictEqual(tradeIntelPayload.ticker, 'OXY');
assert.strictEqual(tradeIntelPayload.direction, 'SHORT');
assert.strictEqual(tradeIntelPayload.grade, 'A');
assert.strictEqual(tradeIntelPayload.scanner_timeframe, '4H');

const tradeIntelDashboard = elementStub();
const tradeIntelDiag = elementStub();
const previousTradeIntelGetElementById = context.document.getElementById;
storage.stock_scanner_journal_server_authoritative = 'true';
storage.kairos_journal_admin_token = 'secret';
context.document.getElementById = id => {
  if (id === 'tradeIntelligenceDashboard') return tradeIntelDashboard;
  if (id === 'tradeIntelligenceDiagnostics') return tradeIntelDiag;
  return previousTradeIntelGetElementById(id);
};
context.renderTradeIntelligenceDashboard({
  version: 'trade-intelligence-v1',
  verified_trade_count: 7,
  message: 'Not enough verified historical data yet.',
  thresholds: { exact_min_trades: 30, broad_min_trades: 100 },
  knowledge_growth: {
    verified_trades_collected: 7,
    exact_threshold_progress: { label: '7 of 30' },
    broader_threshold_progress: { label: '7 of 100' },
    status: 'Building verified history',
    clarification: 'Insights unlock when enough verified trades share similar setup characteristics.',
    closest_exact_groups: [{ label: 'OXY · Short · A Grade', verified_trades: 7, threshold: 30, progress_percent: 23.3 }],
    closest_broader_groups: [{ label: 'Short · A Grade', verified_trades: 7, threshold: 100, progress_percent: 7 }],
    eligible_exact_groups: 0,
    eligible_broader_groups: 0,
  },
  data_quality: { eligible_verified_trades: 7, needs_review: 1, replay_pending: 3, journal_only: 2 },
  diagnostics: {
    cache_status: 'hit',
    sample_sizes: { verified_records: 7, exact_groups: 1, broad_groups: 1 },
    similarity_dimensions_used: ['symbol', 'direction'],
  },
  eligibility_funnel: {
    completed_journal_count: 13,
    replay_available_count: 10,
    replay_data_complete_count: 8,
    verified_match_count: 7,
    eligible_trade_intelligence_count: 7,
    reconciliation: { journal_records_reconciled: true, completed_records_reconciled: true },
    exclusion_reasons: { JOURNAL_REPLAY_MISMATCH: 1 },
  },
});
assert.ok(tradeIntelDashboard.innerHTML.includes('Kairos Knowledge Growth'));
assert.ok(tradeIntelDashboard.innerHTML.includes('7 of 30'));
assert.ok(tradeIntelDashboard.innerHTML.includes('Closest Exact Groups to Unlocking'));
assert.ok(tradeIntelDashboard.innerHTML.includes('Trade Intelligence Data Quality'));
assert.ok(!tradeIntelDashboard.innerHTML.includes('Most Reliable Symbols'), 'dashboard should not publish conclusion tables below thresholds');
context.document.getElementById = previousTradeIntelGetElementById;
delete storage.stock_scanner_journal_server_authoritative;
delete storage.kairos_journal_admin_token;

const missingPlanFallback = htmlFor(setup({ entry: null, sl: null, tp1: null }));
assert.ok(!missingPlanFallback.includes('data-plan-visual="risk-reward"'));
assert.ok(missingPlanFallback.includes('<div class="index-plan-row current-price"><span>Current Price</span>'));

// A setup with only b_plus_tradeable (no true trigger confirmation) must
// NOT be labeled ENTER NOW. It gets its own lower-confirmation Early Entry
// tier instead of falling through to the fully confirmed workflow.
const bPlusOnlySetup = {
  direction: 'SHORT',
  setupGrade: 'A',
  entryStatus: 'Tradeable',
  entry: 50,
  sl: 52,
  tp1: 45,
  trade_eval: {
    trigger_confirmed: false,
    a_plus_ready: false,
    b_plus_tradeable: true,
    trade_stage: 'B+ TRADEABLE',
  },
};
const bPlusStatus = context.simpleStatus(bPlusOnlySetup);
const bPlusProgress = context.setupProgressState(bPlusOnlySetup);
assert.strictEqual(bPlusStatus.label, 'EARLY ENTRY', 'B+-only setups must show Early Entry, not Enter Now');
assert.strictEqual(bPlusProgress.bucket, 'EARLY_ENTRY');
assert.notStrictEqual(bPlusProgress.bucket, 'ENTER_NOW');
assert.notStrictEqual(bPlusProgress.bucket, 'ALMOST_READY');

// True trigger confirmation still correctly produces ENTER NOW.
const aPlusSetup = {
  ...bPlusOnlySetup,
  trade_eval: { ...bPlusOnlySetup.trade_eval, trigger_confirmed: true, a_plus_ready: true },
};
const aPlusStatus = context.simpleStatus(aPlusSetup);
assert.strictEqual(aPlusStatus.label, 'ENTER NOW', 'true trigger-confirmed setups must still show ENTER NOW');

const rankingExecutableLifecycleWaiting = {
  ...aPlusSetup,
  new_entry_signal: {
    bucket: 'WAITING_FOR_CONFIRMATION',
    label: 'WAITING FOR CONFIRMATION',
    current_strategy_status: 'ENTER_NOW',
    current_strategy_executable: true,
    lifecycle_state: 'WAITING_FOR_CONFIRMATION',
    lifecycle_entry_triggered: false,
    actionable: false,
  },
};
assert.strictEqual(context.simpleStatus(rankingExecutableLifecycleWaiting).label, 'WAITING FOR CONFIRMATION');
assert.strictEqual(context.setupProgressState(rankingExecutableLifecycleWaiting).bucket, 'WAITING_FOR_CONFIRMATION');
assert.ok(!htmlFor(rankingExecutableLifecycleWaiting).includes('data-normalized-status="ENTER_NOW"'));

const lifecycleOnlyTriggered = {
  ...aPlusSetup,
  setupGrade: 'C',
  setup_status: 'SKIPPED',
  new_entry_signal: {
    bucket: 'NO_CURRENT_ENTRY',
    label: 'Entry Triggered Previously',
    current_strategy_status: 'SKIP',
    current_strategy_executable: false,
    lifecycle_state: 'ENTRY_TRIGGERED',
    lifecycle_entry_triggered: true,
    actionable: false,
  },
};
assert.strictEqual(context.simpleStatus(lifecycleOnlyTriggered).label, 'ENTRY TRIGGERED PREVIOUSLY');
assert.strictEqual(context.setupProgressState(lifecycleOnlyTriggered).bucket, 'NO_CURRENT_ENTRY');

const authoritativeEnterNow = {
  ...aPlusSetup,
  new_entry_signal: {
    bucket: 'ENTER_NOW',
    label: 'ENTER NOW',
    current_strategy_status: 'ENTER_NOW',
    current_strategy_executable: true,
    lifecycle_state: 'ENTRY_TRIGGERED',
    lifecycle_entry_triggered: true,
    actionable: true,
  },
};
assert.strictEqual(context.simpleStatus(authoritativeEnterNow).label, 'ENTER NOW');
assert.strictEqual(context.setupProgressState(authoritativeEnterNow).bucket, 'ENTER_NOW');

const earlyEntryCard = htmlFor(bPlusOnlySetup);
assert.ok(earlyEntryCard.includes('data-normalized-status="EARLY_ENTRY"'));
assert.ok(earlyEntryCard.includes('data-execution-state="SETUP_EARLY_ENTRY"'));
assert.ok(earlyEntryCard.includes('data-execute-visual-state="early-entry"'));
assert.ok(earlyEntryCard.includes('execute-early-entry'));
assert.ok(earlyEntryCard.includes('EARLY ENTRY'));
assert.ok(earlyEntryCard.includes("Structure has broken, but full confirmation hasn&#39;t happened yet."));
assert.ok(earlyEntryCard.includes('Consider smaller size given lower confirmation.'));
assert.ok(earlyEntryCard.includes('Short setup with mixed trend; structure has started, but full confirmation is not complete.'));
assert.ok(!earlyEntryCard.includes('Price is at the planned entry. You can execute this trade.'));
assert.ok(context.passesFrameworkFilters(bPlusOnlySetup, { status: 'EARLY_ENTRY', tickerSearch: [], direction: 'all', quality: 'all', contractType: 'all' }));
assert.ok(!context.passesFrameworkFilters(bPlusOnlySetup, { status: 'ENTER_NOW', tickerSearch: [], direction: 'all', quality: 'all', contractType: 'all' }));

const snapshotElements = {
  statusFilter: elementStub(),
  qualityFilter: elementStub(),
  directionFilter: elementStub(),
  contractTypeFilter: elementStub(),
  sortFilter: elementStub(),
};
snapshotElements.statusFilter.value = 'ACTIONABLE';
snapshotElements.qualityFilter.value = 'all';
snapshotElements.directionFilter.value = 'all';
snapshotElements.contractTypeFilter.value = 'all';
const originalGetElementById = context.document.getElementById;
const originalRenderScannerResults = context.renderScannerResults;
let snapshotRenderCount = 0;
context.document.getElementById = id => snapshotElements[id] || elementStub();
context.renderScannerResults = () => { snapshotRenderCount += 1; };

const inactiveSnapshotBadge = context.renderSnapshotBadge('Almost Ready', 3, 'timing-area', 'status', 'ALMOST_READY');
assert.ok(inactiveSnapshotBadge.includes('<button type="button"'));
assert.ok(inactiveSnapshotBadge.includes("onclick=\"applySnapshotFilter('status', 'ALMOST_READY')\""));
assert.ok(inactiveSnapshotBadge.includes('aria-pressed="false"'));

context.applySnapshotFilter('status', 'ALMOST_READY');
assert.strictEqual(snapshotElements.statusFilter.value, 'ALMOST_READY');
assert.strictEqual(snapshotElements.statusFilter.dataset.userTouched, 'true');
assert.strictEqual(snapshotRenderCount, 1);
const activeSnapshotBadge = context.renderSnapshotBadge('Almost Ready', 3, 'timing-area', 'status', 'ALMOST_READY');
assert.ok(activeSnapshotBadge.includes(' active'));
assert.ok(activeSnapshotBadge.includes('aria-pressed="true"'));

context.applySnapshotFilter('status', 'ALMOST_READY');
assert.strictEqual(snapshotElements.statusFilter.value, 'all');
assert.strictEqual(snapshotRenderCount, 2);

context.applySnapshotFilter('quality', 'B');
context.applySnapshotFilter('direction', 'SHORT');
context.applySnapshotFilter('contractType', 'PUT');
assert.strictEqual(snapshotElements.qualityFilter.value, 'B');
assert.strictEqual(snapshotElements.directionFilter.value, 'SHORT');
assert.strictEqual(snapshotElements.contractTypeFilter.value, 'PUT');
assert.strictEqual(snapshotElements.qualityFilter.dataset.userTouched, undefined);

context.document.getElementById = originalGetElementById;
context.renderScannerResults = originalRenderScannerResults;

const scannerRenderElements = {
  results: elementStub(),
  summary: elementStub(),
  marketSnapshot: elementStub(),
  statusFilter: elementStub(),
  qualityFilter: elementStub(),
  directionFilter: elementStub(),
  contractTypeFilter: elementStub(),
  tickerInput: elementStub(),
  marketCoverage: elementStub(),
  marketIntelligence: elementStub(),
  topOpportunities: elementStub(),
  sortFilter: elementStub(),
  'near-miss-section': elementStub(),
  'near-miss-results': elementStub(),
  'near-miss-header': elementStub(),
};
scannerRenderElements.statusFilter.value = 'all';
scannerRenderElements.statusFilter.selectedOptions = [{ textContent: 'All Statuses' }];
scannerRenderElements.qualityFilter.value = 'B';
scannerRenderElements.qualityFilter.selectedOptions = [{ textContent: 'Strong Setup' }];
scannerRenderElements.directionFilter.value = 'all';
scannerRenderElements.directionFilter.selectedOptions = [{ textContent: 'All Directions' }];
scannerRenderElements.contractTypeFilter.value = 'all';
scannerRenderElements.contractTypeFilter.selectedOptions = [{ textContent: 'All Contracts' }];
scannerRenderElements.sortFilter.value = 'RANK';
scannerRenderElements.sortFilter.selectedOptions = [{ textContent: 'Opportunity Rank' }];
scannerRenderElements.tickerInput.value = '';
scannerRenderElements.universeFilter = elementStub();
scannerRenderElements.universeFilter.value = 'default';
context.document.getElementById = id => scannerRenderElements[id] || elementStub();

const snapshotFullA = setup({
  ticker: 'TOP',
  setupGrade: 'A',
  direction: 'LONG',
  entryStatus: 'Near Entry',
  option: { type: 'CALL' },
  trade_eval: { trade_stage: 'B+ TRADEABLE', b_plus_tradeable: true, no_trade_reasons: [] },
  ranking: {
    rank: 1,
    tier: 'TOP_OPPORTUNITY',
    score: 86.4,
    positive_reasons: ['A-grade setup', 'Near planned entry', 'Clean structure'],
    cautions: [],
  },
});
const snapshotFullB = setup({
  ticker: 'STRONG',
  setupGrade: 'B',
  direction: 'SHORT',
  entryStatus: 'Near Entry',
  option: { type: 'PUT' },
  trade_eval: { trade_stage: 'B+ TRADEABLE', b_plus_tradeable: true, no_trade_reasons: ['RR < 1.5:1'] },
  ranking: {
    rank: 2,
    tier: 'HIGH_PRIORITY',
    score: 74.2,
    positive_reasons: ['B-grade setup', 'Near planned entry'],
    cautions: ['RR < 1.5:1'],
  },
});
context.__scannerRenderRows = [snapshotFullA, snapshotFullB];
vm.runInContext(`scannerRows = __scannerRenderRows; scannerNearMiss = []; latestScannerMeta = {
  configured_universe_count: 750,
  symbols_attempted: 750,
  symbols_successfully_processed: 715,
  symbols_terminally_evaluated: 750,
  symbols_with_setup: 715,
  symbols_without_setup: 35,
  symbols_operationally_failed: 0,
  evaluation_coverage: 1,
  result_yield: 0.9533,
  symbols_failed: 0,
  scan_duration_ms: 67869,
  partial_result: false,
  partial_result_reasons: [],
  provider_metrics: { alpaca_bar_symbols_requested: 1278, alpaca_bar_symbols_succeeded: 1278, alpaca_bar_symbols_failed: 0 },
  cache_stats: { prices_hit: 10, prices_miss: 30 },
  performance: {
    symbols_per_second: 10.54,
    cache_hit_rate: 0.25,
    market_data_fetch_ms: 42000,
    strategy_evaluation_ms: 21000,
    quote_enrichment_ms: 900,
    median_symbol_duration_ms: 110,
    p95_symbol_duration_ms: 480,
    peak_worker_count: 4,
    memory_rss_mb: 256.4
  }
};`, context);
context.renderScannerResults();

assert.ok(scannerRenderElements.marketCoverage.innerHTML.includes('Market Coverage'));
assert.ok(scannerRenderElements.marketCoverage.innerHTML.includes('Coverage Diagnostics'));
assert.ok(scannerRenderElements.marketCoverage.innerHTML.includes('715'));
assert.ok(scannerRenderElements.marketCoverage.innerHTML.includes('95%'));
assert.ok(scannerRenderElements.marketCoverage.innerHTML.includes('Evaluation Coverage'));
assert.ok(scannerRenderElements.marketCoverage.innerHTML.includes('100%'));
assert.ok(scannerRenderElements.marketCoverage.innerHTML.includes('No Setup'));
assert.ok(scannerRenderElements.marketCoverage.innerHTML.includes('Operational Failures'));
assert.ok(scannerRenderElements.marketCoverage.innerHTML.includes('67.9s'));
assert.ok(scannerRenderElements.marketCoverage.innerHTML.includes('10.5/s'));
assert.ok(scannerRenderElements.marketCoverage.innerHTML.includes('Cache Hit Rate'));
assert.ok(scannerRenderElements.marketCoverage.innerHTML.includes('Complete'));
assert.ok(scannerRenderElements.marketCoverage.innerHTML.includes('Opportunity Ranking'));
assert.ok(!scannerRenderElements.marketCoverage.innerHTML.includes('35 symbol returned no setup'));
assert.ok(scannerRenderElements.marketCoverage.innerHTML.includes('256.4 MB'));
assert.ok(scannerRenderElements.marketIntelligence.innerHTML.includes('Market Intelligence'));
assert.ok(scannerRenderElements.marketIntelligence.innerHTML.includes('2 Ranked Opportunities'));
assert.ok(scannerRenderElements.marketIntelligence.innerHTML.includes('Top Opportunity'));
assert.ok(scannerRenderElements.marketIntelligence.innerHTML.includes('Direction Bias'));
assert.ok(scannerRenderElements.marketIntelligence.innerHTML.includes('Market Breadth'));
assert.ok(scannerRenderElements.marketIntelligence.innerHTML.includes('Market Intelligence Diagnostics'));
assert.ok(scannerRenderElements.results.innerHTML.includes('STRONG'));
assert.ok(!scannerRenderElements.results.innerHTML.includes('TOP'));
assert.ok(scannerRenderElements.topOpportunities.innerHTML.includes('Top Opportunities'));
assert.ok(scannerRenderElements.topOpportunities.innerHTML.includes('#2'));
assert.ok(scannerRenderElements.topOpportunities.innerHTML.includes('STRONG'));
assert.ok(!scannerRenderElements.topOpportunities.innerHTML.includes('TOP'));
assert.ok(scannerRenderElements.summary.innerHTML.includes('1 qualified setup'));
assert.ok(scannerRenderElements.summary.innerHTML.includes('Showing 1 of 2 setups'));
assert.ok(scannerRenderElements.summary.innerHTML.includes('Opportunity Rank'));
assert.ok(
  scannerRenderElements.marketSnapshot.innerHTML.includes(
    '<span class="label">Top Setup</span><span class="count">1</span>'
  ),
  'Market Snapshot should keep full-universe Top Setup count while quality filter is active'
);
assert.ok(
  scannerRenderElements.marketSnapshot.innerHTML.includes(
    '<span class="label">Strong Setup</span><span class="count">1</span>'
  ),
  'Market Snapshot should keep full-universe Strong Setup count while quality filter is active'
);
context.document.getElementById = originalGetElementById;

const warmingElements = {
  results: elementStub(),
  summary: elementStub(),
  marketCoverage: elementStub(),
  marketSnapshot: elementStub(),
  marketIntelligence: elementStub(),
  dataStatus: { ...elementStub(), textContent: '🟡 Warming Building market cache' },
  statusFilter: elementStub(),
  qualityFilter: elementStub(),
  directionFilter: elementStub(),
  contractTypeFilter: elementStub(),
  sortFilter: elementStub(),
  tickerInput: elementStub(),
  topOpportunities: elementStub(),
  'near-miss-section': elementStub(),
  'near-miss-results': elementStub(),
  'near-miss-header': elementStub(),
};
warmingElements.statusFilter.value = 'all';
warmingElements.statusFilter.selectedOptions = [{ textContent: 'All Statuses' }];
warmingElements.qualityFilter.value = 'all';
warmingElements.qualityFilter.selectedOptions = [{ textContent: 'All Setup Quality' }];
warmingElements.directionFilter.value = 'all';
warmingElements.directionFilter.selectedOptions = [{ textContent: 'All Directions' }];
warmingElements.contractTypeFilter.value = 'all';
warmingElements.contractTypeFilter.selectedOptions = [{ textContent: 'All Contracts' }];
warmingElements.tickerInput.value = '';
context.document.getElementById = id => warmingElements[id] || elementStub();
vm.runInContext('scannerRows = []; scannerNearMiss = []; latestScannerMeta = { cache_key: "discovered", universe: "discovered", status: "warming", refreshing: true, running: true, has_cache: false };', context);
context.renderScannerResults();
assert.ok(
  warmingElements.results.innerHTML.includes('Discovery warming'),
  'Discovered warming state should use discovered-specific user-facing copy'
);
context.document.getElementById = originalGetElementById;

const originalFetchWithTimeoutForContractPoll = context.fetchWithTimeout;
const originalSetTimeoutForContractPoll = context.setTimeout;
let scheduledContractPoll = null;
let contractPollUrl = null;
const pollElements = {
  universeFilter: { ...elementStub(), value: 'discovered' },
  statusFilter: elementStub(),
  qualityFilter: elementStub(),
  directionFilter: elementStub(),
  contractTypeFilter: elementStub(),
  sortFilter: elementStub(),
  tickerInput: elementStub(),
  results: elementStub(),
  summary: elementStub(),
  marketCoverage: elementStub(),
  marketSnapshot: elementStub(),
  marketIntelligence: elementStub(),
  topOpportunities: elementStub(),
  dataStatus: elementStub(),
  'near-miss-section': elementStub(),
  'near-miss-results': elementStub(),
  'near-miss-header': elementStub(),
};
pollElements.statusFilter.value = 'all';
pollElements.statusFilter.selectedOptions = [{ textContent: 'All Statuses' }];
pollElements.qualityFilter.value = 'all';
pollElements.qualityFilter.selectedOptions = [{ textContent: 'All Setup Quality' }];
pollElements.directionFilter.value = 'all';
pollElements.directionFilter.selectedOptions = [{ textContent: 'All Directions' }];
pollElements.contractTypeFilter.value = 'all';
pollElements.contractTypeFilter.selectedOptions = [{ textContent: 'All Contracts' }];
pollElements.tickerInput.value = '';
context.document.getElementById = id => pollElements[id] || elementStub();
context.setTimeout = cb => { scheduledContractPoll = cb; return 1; };
context.fetchWithTimeout = (url) => {
  contractPollUrl = url;
  return Promise.resolve({ ok: true, json: () => Promise.resolve({ rows: [], near_miss: [], meta: { cache_key: 'discovered' } }) });
};
vm.runInContext('scannerRows = [{ ticker: "PENDING", direction: "LONG", setupGrade: "B", best_contract: { available: true } }]; scannerNearMiss = []; contractRefreshAttempts = 0;', context);
context.scheduleContractRefreshPoll();
assert.strictEqual(typeof scheduledContractPoll, 'function');
scheduledContractPoll();
assert.strictEqual(contractPollUrl, '/api/scan?view=summary');
context.fetchWithTimeout = originalFetchWithTimeoutForContractPoll;
context.setTimeout = originalSetTimeoutForContractPoll;
context.document.getElementById = originalGetElementById;

vm.runInContext(`
  globalThis.__savedRunScan = runScan;
  globalThis.__savedRenderScannerResults = renderScannerResults;
  globalThis.__renderCountForUniverseChange = 0;
  globalThis.__runScanCountForUniverseChange = 0;
  renderScannerResults = () => { globalThis.__renderCountForUniverseChange += 1; };
  runScan = () => { globalThis.__runScanCountForUniverseChange += 1; };
`, context);
vm.runInContext('scannerRows = [{ ticker: "OLD" }]; scannerNearMiss = [{ ticker: "OLDMISS" }]; latestScannerMeta = { cache_key: "default" }; contractRefreshAttempts = 2; contractPendingTimedOutKeys = new Set(["OLD"]);', context);
context.handleUniverseChange();
assert.strictEqual(vm.runInContext('scannerRows.length', context), 0);
assert.strictEqual(vm.runInContext('scannerNearMiss.length', context), 0);
assert.strictEqual(vm.runInContext('latestScannerMeta', context), null);
assert.strictEqual(vm.runInContext('contractRefreshAttempts', context), 0);
assert.strictEqual(vm.runInContext('contractPendingTimedOutKeys.size', context), 0);
assert.strictEqual(vm.runInContext('__renderCountForUniverseChange', context), 1);
assert.strictEqual(vm.runInContext('__runScanCountForUniverseChange', context), 1);
vm.runInContext('runScan = globalThis.__savedRunScan; renderScannerResults = globalThis.__savedRenderScannerResults;', context);

const reachedLive = htmlFor(setup({ price: 46.05, entryStatus: 'Tradeable', distanceFromEntryAtr: 0.1 }));
assert.ok(reachedLive.includes('data-execution-state="SETUP_CONFIRMED_ENTRY_REACHED"'));
assert.ok(reachedLive.includes('data-execute-visual-state="ready"'));
assert.ok(reachedLive.includes('execute-entry-ready'));
assert.ok(reachedLive.includes('Price is at the planned entry. Use the Option Plan and confirm your selected contract in your broker.'));

const reachedPotential = htmlFor(
  setup({ price: 46.05, entryStatus: 'Tradeable', distanceFromEntryAtr: 0.1, option_plan: null }),
  { available: false, source: 'unavailable', reason: 'no contract passed filters' }
);
assert.ok(reachedPotential.includes('execute-entry-ready'));
assert.ok(reachedPotential.includes('Price is at the planned entry.'));
assert.ok(reachedPotential.includes('Use the Option Plan to select and confirm a contract in your broker before executing.'));

const almostReady = htmlFor(setup({ trade_eval: { trade_stage: 'BUILDING / WATCHLIST' }, setupGrade: 'B', entryStatus: 'Near Entry', confirmationStarted: false }));
assert.ok(almostReady.includes('data-normalized-status="ALMOST_READY"'));
assert.ok(almostReady.includes('execute-not-ready'));
assert.ok(almostReady.includes('Setup is still developing. Wait for full confirmation.'));
assert.ok(almostReady.includes('Short setup with bearish trend; near the entry area and waiting on final confirmation.'));

const building = htmlFor(setup({ trade_eval: { trade_stage: 'BUILDING / WATCHLIST' }, setupGrade: 'B', entryStatus: 'Waiting', confirmationStarted: false, direction: 'LONG' }));
assert.ok(building.includes('data-normalized-status="BUILDING"'));
assert.ok(building.includes('execute-not-ready'));
assert.ok(building.includes('Continue monitoring. The setup is still developing.'));
assert.ok(building.includes('Long setup with bearish trend; forming, but not ready yet.'));

const skipCard = htmlFor(setup({
  setupGrade: 'C',
  trade_eval: { trade_stage: 'RANGE / NO TRADE' },
}));
assert.ok(skipCard.includes('Short setup with bearish trend; not valid right now.'));

const highQualityTooFar = htmlFor(setup({
  setupGrade: 'A',
  entryStatus: 'Too Far',
  trade_eval: { trade_stage: 'B+ TRADEABLE', b_plus_tradeable: true, no_trade_reasons: [] },
}));
assert.ok(highQualityTooFar.includes('Short setup with bearish trend; still Grade A — just too far from the ideal entry zone to act on right now.'));

const strongTooFar = htmlFor(setup({
  setupGrade: 'B',
  entryStatus: 'Too Far',
  confirmationStarted: false,
  trade_eval: { trade_stage: 'BUILDING / WATCHLIST', no_trade_reasons: [] },
}));
assert.ok(strongTooFar.includes('Short setup with bearish trend; still Grade B — just too far from the ideal entry zone to act on right now.'));

const structurallyBlockedTooFar = htmlFor(setup({
  setupGrade: 'A',
  entryStatus: 'Too Far',
  trade_eval: { trade_stage: 'RANGE / NO TRADE', no_trade_reasons: ['Macro/context conflict'] },
}));
assert.ok(structurallyBlockedTooFar.includes('Short setup with bearish trend; not valid right now.'));
assert.ok(!structurallyBlockedTooFar.includes('still Grade A — just too far'));

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

const priorDatedEarnings = setup({
  ticker: 'AAPL',
  direction: 'LONG',
  timeframe: '1D',
  signal_timestamp: '2026-07-17T00:00:00Z',
  entry: 100,
  sl: 96,
  tp1: 108,
  earnings: {
    loaded: true,
    date: '2026-07-30',
    days_until: 13,
    source: 'yfinance',
  },
});

const nextLoadingSameSetup = setup({
  ...priorDatedEarnings,
  earnings: {
    loaded: false,
    loading: true,
    status: 'loading',
    date: null,
    days_until: null,
    source: 'background_refresh',
  },
});

const preservedRows = context.preservePriorEarningsRows([nextLoadingSameSetup], [priorDatedEarnings]);
assert.strictEqual(preservedRows[0].earnings.date, '2026-07-30');
assert.strictEqual(preservedRows[0].earnings.days_until, 13);
assert.strictEqual(preservedRows[0].earnings.source, 'yfinance');

const nextUnavailableSameSetup = setup({
  ...priorDatedEarnings,
  earnings: {
    loaded: false,
    date: null,
    days_until: null,
    source: 'unavailable',
  },
});
const preservedUnavailable = context.preservePriorEarningsRows([nextUnavailableSameSetup], [priorDatedEarnings]);
assert.strictEqual(preservedUnavailable[0].earnings.date, '2026-07-30');

const nextDatedWins = setup({
  ...priorDatedEarnings,
  earnings: {
    loaded: true,
    date: '2026-08-04',
    days_until: 18,
    source: 'yfinance',
  },
});
const noOverride = context.preservePriorEarningsRows([nextDatedWins], [priorDatedEarnings]);
assert.strictEqual(noOverride[0].earnings.date, '2026-08-04');

const differentSignalSameTicker = setup({
  ...nextLoadingSameSetup,
  signal_timestamp: '2026-07-18T00:00:00Z',
});
const notPreservedForDifferentSetup = context.preservePriorEarningsRows([differentSignalSameTicker], [priorDatedEarnings]);
assert.strictEqual(notPreservedForDifferentSetup[0].earnings.date, null);
assert.strictEqual(notPreservedForDifferentSetup[0].earnings.source, 'background_refresh');

const singleTarget = htmlFor(setup({ tp1: 44, tp2: null, tp3: null, final_target: null }));
assert.ok(singleTarget.includes('<div class="rr-value-label">TP1</div>'));
assert.ok(singleTarget.includes('<div class="rr-value-price">$44.00</div>'));
assert.ok(!singleTarget.includes('<div class="rr-value-label">TP2</div>'));
assert.ok(!singleTarget.includes('<div class="rr-value-label">TP3</div>'));
assert.ok(singleTarget.includes('data-current-pct="22.00"'), 'Risk/reward Now marker should fall back to candle-close price');

const quoteCurrent = htmlFor(setup({ current_quote_price: 46.5, tp1: 44, tp2: null, tp3: null, final_target: null }));
assert.ok(quoteCurrent.includes('data-current-pct="16.67"'), 'Risk/reward Now marker should prefer current quote price over candle-close price');

const planValueLabels = [...singleTarget.matchAll(/<div class="rr-value-label">([^<]+)<\/div>/g)].map(match => match[1]);
assert.deepStrictEqual(planValueLabels, ['Entry', 'Stop', 'TP1']);
assert.ok(singleTarget.includes('<div class="rr-value-item entry">'));
assert.ok(!singleTarget.includes('<div class="rr-value-label">Now</div>'));

const threeTargets = htmlFor(setup({ tp1: 45, tp2: 44, tp3: 43 }));
const threeTargetLabels = [...threeTargets.matchAll(/<div class="rr-value-label">([^<]+)<\/div>/g)].map(match => match[1]);
assert.deepStrictEqual(threeTargetLabels, ['Entry', 'Stop', 'TP1', 'TP2', 'TP3']);
assert.ok(threeTargets.includes('<div class="rr-value-label">TP1</div>'));
assert.ok(threeTargets.includes('<div class="rr-value-price">$45.00</div>'));
assert.ok(threeTargets.includes('<div class="rr-value-label">TP2</div>'));
assert.ok(threeTargets.includes('<div class="rr-value-price">$44.00</div>'));
assert.ok(threeTargets.includes('<div class="rr-value-label">TP3</div>'));
assert.ok(threeTargets.includes('<div class="rr-value-price">$43.00</div>'));
assert.ok(!threeTargets.includes('<div class="rr-value-label">Final Target</div>'));

const finalTarget = htmlFor(setup({ tp1: 45, tp2: null, tp3: null, final_target: 42 }));
assert.ok(finalTarget.includes('<div class="rr-value-label">TP1</div>'));
assert.ok(finalTarget.includes('<div class="rr-value-price">$45.00</div>'));
assert.ok(finalTarget.includes('<div class="rr-value-label">Final Target</div>'));
assert.ok(finalTarget.includes('<div class="rr-value-price">$42.00</div>'));

const duplicateTargets = htmlFor(setup({ tp1: 44, tp2: 44, tp3: 43, final_target: 43 }));
assert.strictEqual((duplicateTargets.match(/<div class="rr-value-label">TP2<\/div>/g) || []).length, 0);
assert.strictEqual((duplicateTargets.match(/<div class="rr-value-label">Final Target<\/div>/g) || []).length, 0);
assert.ok(duplicateTargets.includes('<div class="rr-value-label">TP3</div>'));
assert.ok(duplicateTargets.includes('<div class="rr-value-price">$43.00</div>'));

const snapshot = context.scannerSnapshotFromSetup(setup({
  tp1: 45,
  tp2: 44,
  tp3: 43,
  earnings: { loaded: true, date: '2026-07-30', days_until: 17, source: 'cache' },
}), { trade_stage: 'A+ READY' });
assert.strictEqual(snapshot.earnings_date, '2026-07-30');
assert.strictEqual(snapshot.days_until_earnings, 17);
assert.strictEqual(snapshot.earnings_source, 'cache');
assert.strictEqual(snapshot.option_type, 'PUT');
assert.strictEqual(snapshot.preferred_strike, 45);
assert.strictEqual(snapshot.suggested_min_dte, 21);
assert.strictEqual(snapshot.suggested_max_dte, 35);
assert.strictEqual(snapshot.expected_hold_min_days, 7);
assert.strictEqual(snapshot.expected_hold_max_days, 12);
assert.strictEqual(snapshot.expected_move_dollars, -2.05);
assert.strictEqual(snapshot.expected_move_percent, -4.5);
assert.strictEqual(snapshot.option_plan_confidence, 4);
assert.strictEqual(snapshot.actual_strike, null);

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
const tieredHtml = context.renderBestContractBlock(tieredSetup);
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
assert.strictEqual(scannerStatusSnapshot.sector, null);
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

const sectorSnapshot = context.scannerSnapshotFromSetup(
  { ...tieredSetup, sector: 'Technology' },
  { trade_stage: 'B+ TRADEABLE', trigger_confirmed: true }
);
assert.strictEqual(sectorSnapshot.sector, 'Technology');

const regimeSnapshot = context.scannerSnapshotFromSetup(
  {
    ...tieredSetup,
    marketRegime: 'TRENDING',
    dailyMarketRegime: 'RANGING',
    h4MarketRegime: 'TRENDING',
    marketRegimeScore: 82,
  },
  { trade_stage: 'B+ TRADEABLE', trigger_confirmed: true }
);
assert.strictEqual(regimeSnapshot.market_regime, 'TRENDING');
assert.strictEqual(regimeSnapshot.daily_market_regime, 'RANGING');
assert.strictEqual(regimeSnapshot.h4_market_regime, 'TRENDING');
assert.strictEqual(regimeSnapshot.market_regime_score, 82);

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
assert.strictEqual(
  context.positionUrgency({ live: { rNow: 1.21, contractHealth: { status: 'OK' } } }).label,
  'Up 1R+'
);
assert.strictEqual(
  context.positionUrgency({ live: { rNow: 0.62, contractHealth: { status: 'OK' } } }).label,
  'Move Started'
);

const defaultPositionPrefs = context.positionAlertPreferences({});
assert.deepStrictEqual(Object.values(defaultPositionPrefs), Object.values(defaultPositionPrefs).map(() => false));

context.renderJournal = () => {};
context.renderAnalytics = () => {};
storage.stock_scanner_journal = JSON.stringify([{
  id: 9001,
  ticker: 'OXY',
  result: 'Open',
  tracking_status: 'active',
  direction: 'CALL',
  entry: 60,
  plannedStop: 57,
  plannedTp1: 66,
  position_alert_preferences: { UP_1R: true },
}]);
context.Notification = function Notification() {};
context.Notification.permission = 'granted';
context.window.Notification = context.Notification;
context.togglePositionAlertPreferenceFromInput({
  checked: false,
  closest: () => ({ dataset: { positionId: '9001' } }),
}, 'UP_1R');
let alertJournal = JSON.parse(storage.stock_scanner_journal);
assert.strictEqual(alertJournal[0].position_alert_preferences.UP_1R, false);
assert.strictEqual(alertJournal[0].position_alert_preferences.TP1_HIT, false);
alertJournal[0].position_alert_preferences.UP_1R = true;
storage.stock_scanner_journal = JSON.stringify(alertJournal);

let sentNotifications = 0;
context.Notification = function Notification(title, options) {
  sentNotifications += 1;
  this.title = title;
  this.options = options;
};
context.Notification.permission = 'granted';
context.window.Notification = context.Notification;
vm.runInContext(`scannerRows = [{
  ticker: 'OXY',
  current_quote_price: 63.7,
  price: 62.5,
  setupGrade: 'B',
  entryStatus: 'Near Entry',
  trade_eval: { trade_stage: 'B+ TRADEABLE' }
}]; scannerNearMiss = [];`, context);
assert.strictEqual(context.evaluatePositionAlertNotifications().sent, 1);
assert.strictEqual(context.evaluatePositionAlertNotifications().sent, 0);
alertJournal = JSON.parse(storage.stock_scanner_journal);
assert.ok(alertJournal[0].position_alert_notified_keys.UP_1R);
assert.strictEqual(sentNotifications, 1);

storage.stock_scanner_journal = JSON.stringify([{
  id: 9002,
  ticker: 'WAIT',
  result: 'Open',
  tracking_status: 'active',
  direction: 'CALL',
  entry: 60,
  plannedStop: 57,
  plannedTp1: 66,
  position_alert_preferences: {},
}]);
vm.runInContext(`scannerRows = []; scannerNearMiss = [];`, context);
assert.strictEqual(context.evaluatePositionAlertNotifications().sent, 0);

context.Notification = function Notification() {};
context.Notification.permission = 'denied';
context.window.Notification = context.Notification;
const blockedHtml = context.renderPositionAlertControls({
  id: 9003,
  entry: { id: 9003, position_alert_preferences: { UP_1R: true } },
});
assert.ok(blockedHtml.includes('Notifications blocked'));
assert.ok(blockedHtml.includes('disabled'));

assert.ok(html.includes('positionReplayTabButton'));
assert.ok(html.includes('Position Intelligence Replay'));
assert.ok(html.includes('Developer Authentication'));
assert.ok(html.includes('journalAdminTokenInput'));
assert.ok(html.includes('Connect'));
assert.ok(html.includes('Clear Token'));
const replaySummary = elementStub();
const replayBody = elementStub();
const replayReadiness = elementStub();
const replayButton = elementStub();
const replayTokenInput = elementStub();
const replayAuthStatus = elementStub();
const previousGetElementById = context.document.getElementById;
context.document.getElementById = id => ({
  positionReplaySummary: replaySummary,
  positionReplayBody: replayBody,
  positionReplayReadiness: replayReadiness,
  positionReplayTabButton: replayButton,
  journalAdminTokenInput: replayTokenInput,
  journalAuthStatus: replayAuthStatus,
}[id] || elementStub());
delete storage.kairos_journal_admin_token;
context.updateDeveloperReplayAccess();
assert.strictEqual(replayButton.style.display, '');
storage.kairos_journal_admin_token = 'secret';
context.updateDeveloperReplayAccess();
assert.strictEqual(replayButton.style.display, '');
context.initializeJournalAdminTokenInput();
assert.strictEqual(replayTokenInput.value, 'secret');
assert.strictEqual(replayAuthStatus.textContent, '🟡 Checking token...');
replayTokenInput.value = 'new-secret';
context.syncJournalTokenInputState();
assert.strictEqual(storage.kairos_journal_admin_token, 'new-secret');
assert.strictEqual(replayAuthStatus.textContent, '🟡 Checking token...');
context.clearJournalAdminToken();
assert.strictEqual(storage.kairos_journal_admin_token, undefined);
assert.strictEqual(replayTokenInput.value, '');
assert.strictEqual(replayAuthStatus.textContent, '🔴 Not Authenticated');

replayTokenInput.value = 'bad-token';
let authCalls = [];
const previousAuthFetch = context.fetch;
context.fetch = (url, options = {}) => {
  authCalls.push({ url, options });
  return Promise.resolve({ status: 403, ok: false, json: () => Promise.resolve({}), text: () => Promise.resolve('forbidden') });
};

(async () => {
const staleKey = 'setup-STALE-LONG';
context.renderGuidedTradeChartBlock(setup({ ticker: 'STALE', direction: 'LONG', price: 101, entry: 100, sl: 95, tp1: 110 }), 'setup');
const chartCanvas = elementStub();
const chartBody = elementStub();
const chartButtons = ['4H', '1D', '30M'].map(tf => ({
  dataset: { timeframe: tf },
  active: false,
  classList: {
    toggle(cls, value) {
      if (cls === 'active') this.active = Boolean(value);
    },
  },
}));
chartBody.querySelectorAll = selector => selector === '.guided-timeframes button' ? chartButtons : [];
const previousGuidedGetElementById = context.document.getElementById;
const previousGuidedFetchWithTimeout = context.fetchWithTimeout;
context.document.getElementById = id => {
  if (id === `guidedChartCanvas-${staleKey}`) return chartCanvas;
  if (id === `guidedChart-${staleKey}`) return chartBody;
  return previousGuidedGetElementById(id);
};
let resolveFirstChart;
let resolveSecondChart;
const chartRequests = [];
context.fetchWithTimeout = url => {
  chartRequests.push(String(url));
  return new Promise(resolve => {
    if (chartRequests.length === 1) resolveFirstChart = resolve;
    else resolveSecondChart = resolve;
  });
};
const firstChartLoad = context.loadGuidedTradeChart(staleKey, '4H');
const secondChartLoad = context.loadGuidedTradeChart(staleKey, '1D');
resolveSecondChart({
  ok: true,
  json: () => Promise.resolve({
    timeframe: '1D',
    provider: 'alpaca',
    chart_load_duration_ms: 12,
    candles: [
      { open: 100, high: 104, low: 98, close: 103 },
      { open: 103, high: 106, low: 102, close: 105 },
    ],
  }),
});
await secondChartLoad;
const afterDaily = chartCanvas.innerHTML;
assert.ok(afterDaily.includes('1D'));
resolveFirstChart({
  ok: true,
  json: () => Promise.resolve({
    timeframe: '4H',
    provider: 'yahoo',
    chart_load_duration_ms: 40,
    candles: [
      { open: 98, high: 99, low: 95, close: 96 },
      { open: 96, high: 97, low: 94, close: 95 },
    ],
  }),
});
await firstChartLoad;
assert.strictEqual(chartCanvas.innerHTML, afterDaily, 'stale guided chart response must not overwrite latest timeframe selection');
assert.ok(chartRequests[0].includes('timeframe=4H'));
assert.ok(chartRequests[1].includes('timeframe=1D'));
assert.ok(chartButtons.find(button => button.dataset.timeframe === '1D').classList.active);
context.fetchWithTimeout = previousGuidedFetchWithTimeout;
context.document.getElementById = previousGuidedGetElementById;

const scannerLifecycleElements = {
  scanBtn: elementStub(),
  status: elementStub(),
  results: elementStub(),
  summary: elementStub(),
  universeFilter: { ...elementStub(), value: 'discovered' },
  dataStatus: elementStub(),
  marketCoverage: elementStub(),
  marketSnapshot: elementStub(),
  marketIntelligence: elementStub(),
  topOpportunities: elementStub(),
  statusFilter: elementStub(),
  qualityFilter: elementStub(),
  directionFilter: elementStub(),
  contractTypeFilter: elementStub(),
  sortFilter: elementStub(),
  tickerInput: elementStub(),
  'near-miss-section': elementStub(),
  'near-miss-results': elementStub(),
  'near-miss-header': elementStub(),
};
scannerLifecycleElements.statusFilter.value = 'all';
scannerLifecycleElements.statusFilter.selectedOptions = [{ textContent: 'All Statuses' }];
scannerLifecycleElements.qualityFilter.value = 'all';
scannerLifecycleElements.qualityFilter.selectedOptions = [{ textContent: 'All Setup Quality' }];
scannerLifecycleElements.directionFilter.value = 'all';
scannerLifecycleElements.directionFilter.selectedOptions = [{ textContent: 'All Directions' }];
scannerLifecycleElements.contractTypeFilter.value = 'all';
scannerLifecycleElements.contractTypeFilter.selectedOptions = [{ textContent: 'All Contracts' }];
scannerLifecycleElements.sortFilter.value = 'RANK';
scannerLifecycleElements.sortFilter.selectedOptions = [{ textContent: 'Opportunity Rank' }];
scannerLifecycleElements.tickerInput.value = '';
const originalFetchWithTimeoutForScannerLifecycle = context.fetchWithTimeout;
const originalSetTimeoutForScannerLifecycle = context.setTimeout;
const previousGetElementByIdForScannerLifecycle = context.document.getElementById;
context.document.getElementById = id => scannerLifecycleElements[id] || elementStub();

vm.runInContext(`
  globalThis.__savedLifecycleRenderScannerResults = renderScannerResults;
  globalThis.__savedLifecycleRenderExecutionTab = renderExecutionTab;
  globalThis.__savedLifecycleRenderFurtherAnalysisTab = renderFurtherAnalysisTab;
  globalThis.__savedLifecycleRenderPositionsTab = renderPositionsTab;
  globalThis.__savedLifecycleDetectStockAlertEvents = detectStockAlertEvents;
  globalThis.__savedLifecycleDeliverStockAlertEvent = deliverStockAlertEvent;
  globalThis.__savedLifecycleEvaluatePositionAlertNotifications = evaluatePositionAlertNotifications;
  globalThis.__savedLifecycleRefreshExpectedMoveTrackingFromRows = refreshExpectedMoveTrackingFromRows;
  globalThis.__savedLifecycleRecordEnterNowFunnelSnapshot = recordEnterNowFunnelSnapshot;
  globalThis.__savedLifecycleFetchWithTimeout = fetchWithTimeout;
  globalThis.__savedLifecycleSetTimeout = setTimeout;
  renderScannerResults = () => {
    globalThis.__scannerLifecycleRenderCount += 1;
    document.getElementById('results').innerHTML = 'completed result rendered';
  };
  renderExecutionTab = () => {};
  renderFurtherAnalysisTab = () => {};
  renderPositionsTab = () => {};
  detectStockAlertEvents = () => [];
  deliverStockAlertEvent = () => {};
  evaluatePositionAlertNotifications = () => {};
  refreshExpectedMoveTrackingFromRows = () => {};
  recordEnterNowFunnelSnapshot = () => {};
`, context);
context.__scannerLifecycleRenderCount = 0;

const completedNoSetupPayload = {
  rows: Array.from({ length: 715 }, (_, index) => ({ ticker: `T${index}`, direction: 'LONG', setupGrade: 'B', best_contract: { available: false, source: 'option_plan' } })),
  near_miss: [],
  meta: {
    status: 'complete',
    universe: 'discovered',
    cache_key: 'discovered',
    symbols_attempted: 750,
    symbols_terminally_evaluated: 750,
    symbols_successfully_processed: 715,
    symbols_with_setup: 715,
    symbols_without_setup: 35,
    symbols_operationally_failed: 0,
    evaluation_coverage_percent: 100,
    partial_result: false,
    partial_result_reasons: [],
  },
};
let scannerLifecyclePollScheduled = false;
context.setTimeout = () => { scannerLifecyclePollScheduled = true; return 1; };
context.fetchWithTimeout = (url) => {
  assert.strictEqual(url, '/api/scan?view=summary');
  return Promise.resolve({ ok: true, json: () => Promise.resolve(completedNoSetupPayload) });
};
await context.runScan();
assert.strictEqual(scannerLifecycleElements.scanBtn.disabled, false);
assert.strictEqual(scannerLifecyclePollScheduled, false);
assert.ok(context.__scannerLifecycleRenderCount >= 1);
assert.strictEqual(vm.runInContext('scannerRows.length', context), 715);
assert.strictEqual(vm.runInContext('latestScannerMeta.partial_result', context), false);
assert.strictEqual(vm.runInContext('latestScannerMeta.symbols_terminally_evaluated', context), 750);
assert.ok(!scannerLifecycleElements.status.innerHTML.includes('spinner'), 'completed scan should clear spinner even when rows returned are fewer than attempted');

let warmingPollCallback = null;
let warmingFetchCount = 0;
context.__scannerLifecycleRenderCount = 0;
context.setTimeout = cb => { warmingPollCallback = cb; return 1; };
context.fetchWithTimeout = (url) => {
  assert.strictEqual(url, '/api/scan?view=summary', 'warming poll must not use refresh=true');
  warmingFetchCount += 1;
  const payload = warmingFetchCount === 1
    ? { rows: [], near_miss: [], meta: { status: 'warming', universe: 'discovered', cache_key: 'discovered', cache: 'miss', has_cache: false, refreshing: true, generated_at: null } }
    : { ...completedNoSetupPayload, rows: completedNoSetupPayload.rows.slice(0, 1), meta: { ...completedNoSetupPayload.meta, symbols_successfully_processed: 1, symbols_with_setup: 1, symbols_without_setup: 749 } };
  return Promise.resolve({ ok: true, json: () => Promise.resolve(payload) });
};
await context.runScan();
assert.strictEqual(scannerLifecycleElements.scanBtn.disabled, true);
assert.strictEqual(typeof warmingPollCallback, 'function');
assert.ok(scannerLifecycleElements.status.innerHTML.includes('Discovering the market'));
warmingPollCallback();
await new Promise(resolve => setTimeout(resolve, 0));
assert.strictEqual(warmingFetchCount, 2);
assert.strictEqual(scannerLifecycleElements.scanBtn.disabled, false);
assert.ok(!scannerLifecycleElements.status.innerHTML.includes('spinner'), 'warming poll should clear spinner after complete response');

context.__scannerLifecycleRenderCount = 0;
context.setTimeout = () => { throw new Error('partial scans should not schedule warming polls'); };
context.fetchWithTimeout = () => Promise.resolve({
  ok: true,
  json: () => Promise.resolve({
    ...completedNoSetupPayload,
    meta: {
      ...completedNoSetupPayload.meta,
      partial_result: true,
      partial_result_reasons: [{ stage: 'market_data', reason: 'provider_timeout', count: 1 }],
      symbols_operationally_failed: 1,
    },
  }),
});
await context.runScan();
assert.strictEqual(scannerLifecycleElements.scanBtn.disabled, false);
assert.ok(context.__scannerLifecycleRenderCount >= 1);
assert.ok(!scannerLifecycleElements.status.innerHTML.includes('spinner'), 'operationally partial scans are finished requests and should clear spinner');

vm.runInContext('renderScannerResults = () => { throw new Error("render exploded"); };', context);
context.fetchWithTimeout = () => Promise.resolve({ ok: true, json: () => Promise.resolve(completedNoSetupPayload) });
await context.runScan();
assert.strictEqual(scannerLifecycleElements.scanBtn.disabled, false);
assert.ok(scannerLifecycleElements.status.textContent.includes('Error: render exploded'));
assert.ok(scannerLifecycleElements.results.innerHTML.includes('Scan completed, but results could not be displayed.'));
vm.runInContext(`
  renderScannerResults = () => {
    globalThis.__scannerLifecycleRenderCount += 1;
    document.getElementById('results').innerHTML = 'completed result rendered';
  };
`, context);

let duplicateFetchCount = 0;
let releaseDuplicateFetch;
const duplicateFetchPromise = new Promise(resolve => { releaseDuplicateFetch = resolve; });
context.fetchWithTimeout = () => {
  duplicateFetchCount += 1;
  return duplicateFetchPromise;
};
const firstDuplicateScan = context.runScan();
const secondDuplicateScan = context.runScan();
assert.strictEqual(await secondDuplicateScan, null);
assert.strictEqual(duplicateFetchCount, 1, 'duplicate runScan calls should not overlap full discovered scans');
releaseDuplicateFetch({ ok: true, json: () => Promise.resolve(completedNoSetupPayload) });
await firstDuplicateScan;
assert.strictEqual(scannerLifecycleElements.scanBtn.disabled, false);

context.fetchWithTimeout = originalFetchWithTimeoutForScannerLifecycle;
context.setTimeout = originalSetTimeoutForScannerLifecycle;
context.document.getElementById = previousGetElementByIdForScannerLifecycle;
vm.runInContext(`
  renderScannerResults = globalThis.__savedLifecycleRenderScannerResults;
  renderExecutionTab = globalThis.__savedLifecycleRenderExecutionTab;
  renderFurtherAnalysisTab = globalThis.__savedLifecycleRenderFurtherAnalysisTab;
  renderPositionsTab = globalThis.__savedLifecycleRenderPositionsTab;
  detectStockAlertEvents = globalThis.__savedLifecycleDetectStockAlertEvents;
  deliverStockAlertEvent = globalThis.__savedLifecycleDeliverStockAlertEvent;
  evaluatePositionAlertNotifications = globalThis.__savedLifecycleEvaluatePositionAlertNotifications;
  refreshExpectedMoveTrackingFromRows = globalThis.__savedLifecycleRefreshExpectedMoveTrackingFromRows;
  recordEnterNowFunnelSnapshot = globalThis.__savedLifecycleRecordEnterNowFunnelSnapshot;
  fetchWithTimeout = globalThis.__savedLifecycleFetchWithTimeout;
  setTimeout = globalThis.__savedLifecycleSetTimeout;
`, context);
context.fetchWithTimeout = (url, options = {}) => context.fetch(url, options);
vm.runInContext('fetchWithTimeout = globalThis.fetchWithTimeout;', context);

const result = await context.connectJournalAdminToken();
assert.strictEqual(result.authenticated, false);
assert.strictEqual(result.status, 403);
assert.strictEqual(storage.kairos_journal_admin_token, 'bad-token');
assert.strictEqual(replayTokenInput.value, 'bad-token');
assert.strictEqual(replayAuthStatus.textContent, '🔴 Invalid token');
assert.ok(authCalls[0].url.includes('/api/journal/diagnostics'));
assert.strictEqual(authCalls[0].options.headers['X-Kairos-Admin-Token'], 'bad-token');
context.fetch = previousAuthFetch;

context.renderPositionReplay({
  ready: false,
  message: 'No server-backed positions are available for replay yet. Migrate or add journal positions to begin historical analysis.',
  aggregate: { positions_replayed: 0, complete_replays: 0, incomplete_replays: 0, ambiguous_replays: 0 },
  replays: [],
});
assert.ok(replayBody.innerHTML.includes('No server-backed positions are available for replay yet'));

context.renderPositionReplay({
  ready: true,
  synthetic_results_included: true,
  aggregate: {
    positions_replayed: 1,
    complete_replays: 1,
    incomplete_replays: 0,
    ambiguous_replays: 0,
    closed_complete_real_replays: 0,
    percent_entered_watch: { percent: 100, sample_size: 1 },
    high_churn_rate: { percent: 0, sample_size: 1 },
  },
  evidence_guard: { message: 'Evidence sample is still developing. No threshold recommendations should be made.' },
  evidence_log: [{ ticker: 'SYN', observation_type: 'WATCH_RECOVERY', observation: 'WATCH recovered to a calmer state.', position_id: 'syn-pos' }],
  evidence_readiness: {
    total_durable_positions: 1,
    replay_ready: 0,
    partially_ready: 1,
    not_replayable: 0,
    open_positions: 1,
    closed_positions: 0,
    positions_with_journaled_timeframe: 0,
    positions_using_inferred_4h: 1,
    positions_missing_recorded_outcome: 0,
    positions_missing_option_details: 1,
    positions: [{ ticker: 'SYN', direction: 'LONG', status: 'PARTIALLY_READY', available: { Direction: true, Entry: true }, missing_required: [], missing_optional: ['setup timeframe'], invalid: [] }],
  },
  replays: [{
    synthetic: true,
    ticker: 'SYN',
    direction: 'LONG',
    timeframe: '4H',
    timeframe_source: 'inferred_default',
    outcome_category: 'WATCH_RECOVERED',
    final_state: 'HEALTHY',
    candles_evaluated: 3,
    state_transition_count: 2,
    maximum_r: 1.2,
    minimum_r: -0.4,
    maximum_progress: 80,
    data_gaps: [],
    timeline: [{ timestamp: '2026-07-22T14:00:00Z', event_type: 'STATE_TRANSITION', previous_state: 'WATCH', new_state: 'HEALTHY', current_r: 0.4, reason_code: 'NORMAL_PROGRESS' }],
    time_in_each_state: { HEALTHY: { candle_count: 2, percent_of_candles: 66.7 }, WATCH: { candle_count: 1, percent_of_candles: 33.3 } },
  }],
});
assert.ok(replayBody.innerHTML.includes('Synthetic Fixture'));
assert.ok(replayBody.innerHTML.includes('STATE_TRANSITION'));
assert.ok(replayBody.innerHTML.includes('State Duration'));
assert.ok(replayReadiness.innerHTML.includes('Evidence Readiness'));
assert.ok(replayReadiness.innerHTML.includes('PARTIALLY_READY'));
assert.ok(replayReadiness.innerHTML.includes('Evidence sample is still developing'));

let refreshCalled = false;
context.fetch = (url, options = {}) => {
  refreshCalled = String(url).includes('/api/dev/position-replay/refresh') && options.method === 'POST';
  return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({ ready: true, aggregate: {}, evidence_readiness: { total_durable_positions: 0 }, replays: [] }) });
};
await context.refreshPositionReplay('stale');
assert.ok(refreshCalled);
context.fetch = previousAuthFetch;
context.document.getElementById = previousGetElementById;

console.log('Live card render v1 tests passed');
})().catch(error => {
  console.error(error);
  process.exit(1);
});
