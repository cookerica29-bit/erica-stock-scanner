const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const executionGuidance = require('../public/execution_guidance.js');
const contractGuidance = require('../public/contract_guidance.js');
const cardData = require('../public/card_data.js');

const html = fs.readFileSync('public/index.html', 'utf8');
const inline = [...html.matchAll(/<script(?![^>]*src=)[^>]*>([\s\S]*?)<\/script>/gi)][0][1];

function elementStub(overrides = {}) {
  return {
    innerHTML: '',
    textContent: '',
    value: '',
    checked: false,
    dataset: {},
    style: {},
    options: [],
    selectedOptions: [{ textContent: '' }],
    classList: { add() {}, remove() {}, toggle() {} },
    addEventListener() {},
    removeEventListener() {},
    setAttribute() {},
    appendChild() {},
    remove() {},
    scrollIntoView() {},
    ...overrides,
  };
}

const storage = {};
const elements = {
  tradingBudgetFilter: elementStub({
    value: '250',
    options: [{ value: 'NO_LIMIT' }, { value: '100' }, { value: '250' }, { value: '500' }, { value: '1000' }],
    selectedOptions: [{ textContent: 'Under $250' }],
  }),
  statusFilter: elementStub({ value: 'ACTIONABLE', selectedOptions: [{ textContent: 'Actionable' }] }),
  directionFilter: elementStub({ value: 'all', selectedOptions: [{ textContent: 'All Directions' }] }),
  qualityFilter: elementStub({ value: 'all', selectedOptions: [{ textContent: 'All Setup Quality' }] }),
  contractTypeFilter: elementStub({ value: 'all', selectedOptions: [{ textContent: 'All Contracts' }] }),
  sortFilter: elementStub({ value: 'RANK', selectedOptions: [{ textContent: 'Best Opportunities' }] }),
  tickerInput: elementStub({ value: '' }),
  topOpportunities: elementStub(),
};

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
  fetch: () => Promise.reject(new Error('network disabled in budget discovery test')),
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
  matchMedia: () => ({ matches: false, addEventListener() {}, removeEventListener() {} }),
};
context.self = context.window;
context.globalThis = context;

vm.createContext(context);
vm.runInContext(inline, context);

function contract({ premium, strike = 30, type = 'PUT', expiration = '2026-08-21', openInterest = 250, volume = 80, spread = 0.1 }) {
  return {
    type,
    strike,
    expiration,
    ask: premium,
    bid: Math.max(0.01, premium - spread),
    spread,
    open_interest: openInterest,
    volume,
  };
}

function setup(overrides = {}) {
  const baseContract = contract({ premium: 9.25, strike: 30, spread: 0.15, openInterest: 1000, volume: 300 });
  return {
    ticker: 'TER',
    direction: 'SHORT',
    price: 31,
    entry: 30,
    sl: 32,
    tp1: 27,
    tp2: 25,
    tp3: 23,
    timeframe: '4H',
    entryStatus: 'Near Entry',
    setupGrade: 'A',
    setupScore: 90,
    stockTrend: 'Bearish',
    stockPhase: 'Pullback',
    stockSetupStatus: 'Pullback Active',
    stockLocation: 'Premium',
    confirmationStarted: true,
    trade_eval: { trade_stage: 'A+ READY', trigger_confirmed: true },
    ranking: { rank: 1, tier: 'TOP_OPPORTUNITY', score: 95, positive_reasons: ['A-grade setup'], cautions: [] },
    best_contract: {
      available: true,
      ...baseContract,
      candidate_audit: {
        best_quality_contract: baseContract,
        best_balanced_contract: contract({ premium: 3.2, strike: 29, spread: 0.12, openInterest: 400, volume: 100 }),
        lowest_cost_acceptable_contract: contract({ premium: 0.85, strike: 28, spread: 0.08, openInterest: 120, volume: 50 }),
      },
    },
    ...overrides,
  };
}

const premiumTop = setup();
const budgetFriendly = setup({
  ticker: 'FE',
  entryStatus: 'Tradeable',
  ranking: { rank: 2, tier: 'HIGH_PRIORITY', score: 88, positive_reasons: ['Clean pullback'], cautions: [] },
  best_contract: {
    available: true,
    ...contract({ premium: 0.85, strike: 25, spread: 0.05, openInterest: 500, volume: 140 }),
  },
});
const premiumOnly = setup({
  ticker: 'EXP',
  ranking: { rank: 3, tier: 'HIGH_PRIORITY', score: 84, positive_reasons: ['Room to target'], cautions: [] },
  best_contract: {
    available: true,
    ...contract({ premium: 8.75, strike: 80, spread: 3.0, openInterest: 5, volume: 0 }),
  },
});
const hydratedAffordable = setup({
  ticker: 'HYD',
  option_pricing: {
    status: 'ready',
    quality: 'live_ask',
    ask: 1.85,
    estimated_contract_cost: 185,
    open_interest: 300,
    volume: 100,
  },
  pricing_status: 'ready',
  pricing_quality: 'live_ask',
  best_contract: {
    available: false,
    source: 'option_plan',
  },
});
const pendingPricing = setup({
  ticker: 'PND',
  pricing_status: 'pending',
  option_pricing: { status: 'pending', quality: 'pending' },
});
const notRequestedPricing = setup({
  ticker: 'NRQ',
  pricing_status: 'not_requested',
  option_pricing: { status: 'not_requested', quality: 'not_requested', reason: 'outside_auto_hydration_cap' },
});
const notRequestedWithRawAsk = setup({
  ticker: 'RAW',
  pricing_status: 'not_requested',
  option_pricing: { status: 'not_requested', quality: 'not_requested', reason: 'outside_auto_hydration_cap' },
  best_contract: {
    available: true,
    ...contract({ premium: 0.75, strike: 30, type: 'CALL', spread: 0.05 }),
  },
});
const liveBudgetFriendly = setup({
  ticker: 'LIV',
  direction: 'LONG',
  pricing_status: 'ready',
  pricing_quality: 'live_ask',
  option_pricing: { status: 'ready', quality: 'live_ask', ask: 0.75, estimated_contract_cost: 75, open_interest: 300, volume: 100 },
  best_contract: { available: false, source: 'option_plan' },
});
const livePremiumTrade = setup({
  ticker: 'XOM',
  direction: 'LONG',
  pricing_status: 'ready',
  pricing_quality: 'live_ask',
  option_pricing: { status: 'ready', quality: 'live_ask', ask: 5.85, estimated_contract_cost: 585, open_interest: 300, volume: 100 },
  best_contract: { available: false, source: 'option_plan' },
});
const stalePremiumTrade = setup({
  ticker: 'XOM',
  direction: 'LONG',
  pricing_status: 'stale',
  pricing_quality: 'stale_quote',
  option_pricing: { status: 'stale', quality: 'stale_quote', estimated_contract_cost: 585, open_interest: 300, volume: 100 },
  best_contract: {
    available: true,
    ...contract({ premium: 5.85, strike: 105, type: 'CALL', spread: 0.15, openInterest: 300, volume: 100 }),
  },
});
const unavailablePricing = setup({
  ticker: 'NA',
  pricing_status: 'unavailable',
  option_pricing: { status: 'unavailable', quality: 'unavailable', reason: 'contract_not_found' },
  best_contract: { available: false },
});
const zeroQuote = setup({
  ticker: 'ZERO',
  best_contract: {
    available: true,
    type: 'CALL',
    strike: 25,
    expiry: '2026-08-21',
    ask: 0,
    mid: 0,
    estimated_contract_cost: 0,
  },
});

assert.strictEqual(context.tradingBudgetValue(), 250);
assert.strictEqual(context.tradingBudgetLabel(250), 'Under $250');
assert.strictEqual(context.selectedEstimatedContractCost(premiumTop), 925);
assert.strictEqual(context.bestAffordableContract(premiumTop, 250).cost, 85);
assert.strictEqual(context.selectedEstimatedContractCost(hydratedAffordable), 185);
assert.strictEqual(context.setupFitsTradingBudget(hydratedAffordable, 250), true);
assert.strictEqual(context.setupFitsTradingBudget(premiumTop, 250), true, 'Budget candidate keeps premium setup eligible for budget view');
assert.strictEqual(context.accessibilityScore(budgetFriendly).key, 'easy');
assert.strictEqual(context.accessibilityScore(hydratedAffordable).key, 'easy');
assert.strictEqual(context.accessibilityScore(pendingPricing).label, 'Pricing pending');
assert.strictEqual(context.accessibilityScore(notRequestedPricing).label, 'Pricing not loaded');
assert.strictEqual(context.selectedEstimatedContractCost(liveBudgetFriendly), 75);
assert.strictEqual(context.budgetBadge(liveBudgetFriendly).displayLabel, 'Lower Indicative Cost');
assert.strictEqual(context.setupFitsTradingBudget(liveBudgetFriendly, 100), true);
assert.strictEqual(context.selectedEstimatedContractCost(livePremiumTrade), 585);
assert.strictEqual(context.budgetBadge(livePremiumTrade).displayLabel, 'Mid Indicative Cost');
assert.strictEqual(context.setupFitsTradingBudget(livePremiumTrade, 500), false);
assert.strictEqual(context.selectedEstimatedContractCost(stalePremiumTrade), 585);
assert.strictEqual(context.budgetBadge(stalePremiumTrade).displayLabel, 'Stale Indicative Cost');
assert.strictEqual(context.budgetBadge(stalePremiumTrade).costText, 'Stale $585.00');
assert.strictEqual(context.renderBudgetBadge(stalePremiumTrade).includes('Cost Unavailable'), false);
assert.strictEqual(context.selectedEstimatedContractCost(notRequestedWithRawAsk), null);
assert.strictEqual(context.budgetBadge(notRequestedWithRawAsk).costText, 'Pricing not loaded');
assert.strictEqual(context.setupFitsTradingBudget(notRequestedWithRawAsk, 100), false);
assert.strictEqual(context.budgetBadge(pendingPricing).costText, 'Pricing loading...');
assert.strictEqual(context.budgetBadge(unavailablePricing).displayLabel, 'Cost Unavailable');
assert.strictEqual(context.setupFitsTradingBudget(unavailablePricing, 1000), false);
assert.strictEqual(context.selectedEstimatedContractCost(zeroQuote), null);
assert.strictEqual(context.setupFitsTradingBudget(zeroQuote, 100), false);
assert.strictEqual(context.accessibilityScore(premiumOnly).key, 'premium');
assert.strictEqual(context.renderBudgetBadge(budgetFriendly).includes('Lower Indicative Cost'), true);
assert.strictEqual(context.renderBudgetBadge(pendingPricing).includes('Pricing loading...'), true);
assert.strictEqual(context.renderBudgetBadge(notRequestedPricing).includes('Pricing not loaded'), true);

const defaultSorted = context.sortScannerCardsForDisplay([budgetFriendly, premiumTop], 'RANK');
assert.strictEqual(defaultSorted[0].ticker, 'TER', 'Best Overall ranking remains unchanged');
const budgetSorted = context.sortScannerCardsForDisplay([premiumOnly, budgetFriendly], 'BUDGET');
assert.strictEqual(budgetSorted[0].ticker, 'FE', 'Budget sort prioritizes affordable setups');

const filters = { status: 'all', direction: 'all', quality: 'all', contractType: 'all', tickerSearch: [] };
assert.strictEqual(context.passesFrameworkFilters(premiumOnly, filters), true, 'Budget preference must not filter expensive setups');

const summary = context.opportunityBudgetSummary([premiumTop, budgetFriendly, premiumOnly]);
assert.strictEqual(summary.enterNow, 3);
assert.strictEqual(summary.budgetFriendly, 1);
assert.strictEqual(summary.premiumOnly, 1);

context.renderTopOpportunities([premiumTop, budgetFriendly, premiumOnly]);
assert.ok(elements.topOpportunities.innerHTML.includes("Today's Market"));
assert.ok(elements.topOpportunities.innerHTML.includes('Best Overall'));
assert.ok(elements.topOpportunities.innerHTML.includes('Best Within My Budget'));
assert.ok(elements.topOpportunities.innerHTML.includes('TER'));
assert.ok(elements.topOpportunities.innerHTML.includes('FE'));
assert.ok(!elements.topOpportunities.innerHTML.includes('&lt;span class=&quot;direction-value'), 'Winner panels must not render escaped direction HTML');
assert.ok(!elements.topOpportunities.innerHTML.includes('<span class="direction-value direction-short"><span'), 'Top rows must not nest direction markup');
context.renderTopOpportunities([liveBudgetFriendly, livePremiumTrade]);
assert.ok(elements.topOpportunities.innerHTML.includes('<span class="direction-value direction-long">Long</span>'));
assert.ok(!elements.topOpportunities.innerHTML.includes('&lt;span class=&quot;direction-value'), 'Long winner direction must render as styled text, not raw markup');

elements.tradingBudgetFilter.value = '100';
context.renderTopOpportunities([premiumOnly]);
assert.ok(elements.topOpportunities.innerHTML.includes('No currently priced opportunities fit this budget.'));
elements.tradingBudgetFilter.value = '250';

const cardHtml = context.renderCard(budgetFriendly);
assert.ok(cardHtml.includes('Lower Indicative Cost'));
assert.ok(cardHtml.includes('Indicative $85.00'));

console.log('budget_opportunity_discovery_v1 passed');
