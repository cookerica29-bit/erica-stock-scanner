// Review Queue auth-state UI (2026-09-01 session): the production
// regression traced to GET /candidates/review-queue returning a real 401
// on a fresh /review-queue load (no bug in the auth mechanism itself --
// investigated and confirmed the same localStorage key name and the same
// path="/" session cookie are already shared with candidates.html) -- but
// the old handling was a small one-line status message, easy to mistake
// for a broken/empty queue. This tests the actual review_queue.js code
// (via a minimal hand-built DOM/fetch/storage fake, matching this repo's
// existing plain-Node .js test convention -- no jsdom dependency
// available), not a reimplementation of its logic.
'use strict';

const assert = require('assert');

function makeElement() {
  const classes = new Set();
  return {
    _html: '',
    get innerHTML() { return this._html; },
    set innerHTML(value) { this._html = value; },
    value: '',
    classList: {
      add: (cls) => classes.add(cls),
      remove: (cls) => classes.delete(cls),
      contains: (cls) => classes.has(cls),
    },
    focus() {},
  };
}

const elements = {
  apiBand: makeElement(),
  mainContent: makeElement(),
  apiKeyInput: makeElement(),
};

const storageListeners = [];
const storageData = {};
const KEY = 'kairos_scanner_api_key';

global.document = {
  getElementById: (id) => elements[id] || null,
  // DOMContentLoaded registration only -- deliberately never fired in
  // these tests, so requiring the module doesn't auto-trigger loadQueue().
  addEventListener: () => {},
};

global.window = {
  addEventListener: (type, callback) => {
    if (type === 'storage') storageListeners.push(callback);
  },
};

global.localStorage = {
  getItem: (k) => (Object.prototype.hasOwnProperty.call(storageData, k) ? storageData[k] : null),
  setItem: (k, v) => { storageData[k] = v; },
  removeItem: (k) => { delete storageData[k]; },
};
global.sessionStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };

let fetchQueue = [];
global.fetch = async (url) => {
  const next = fetchQueue.shift();
  if (!next) throw new Error(`Unexpected fetch call with nothing queued: ${url}`);
  return {
    ok: next.status >= 200 && next.status < 300,
    status: next.status,
    text: async () => JSON.stringify(next.body),
  };
};

const reviewQueue = require('../public/review_queue.js');

function goodQueuePayload(overrides = {}) {
  return {
    mechanism: 'stage1_mechanical_then_confluence_count_v1',
    disclaimer: 'Ranking is unvalidated -- based on signal counts, not track record.',
    snapshot_id: 'test-snapshot',
    count: 1,
    candidates: [{
      ticker: 'TEST', source: 'ma_pipeline', signal: 'long', rank: 1,
      entry_price: 100, stop: 95, target: 110, risk_reward: 2,
      confluence_available: true,
      confluence_counts: { favorable: 2, unfavorable: 1, neutral: 4, applicable: 7 },
      confluence_label: 'some confluence',
      bos_confirmed: false, displacement_label: 'WEAK', displacement_score: 20,
      location_label: 'midrange', location_alignment: 'neutral', macro_bias: 'Macro Neutral',
      sweep_confirmed: false, rejection_confirmed: false,
      execution_shadow_ok: true, execution_shadow_reason: 'Recent 4H confirmation remains structurally intact',
      entry_distance_pct: 0.5, setup_key: 'TEST|ma_pipeline|long|95.00|110.00', current_review: null,
    }],
    ...overrides,
  };
}

async function run() {
  // --- 1. First load returns 401 -> prominent auth-required panel renders ---
  elements.apiBand.classList.add('hidden'); // simulate a stale/expired saved key -- band was hidden
  fetchQueue = [{ status: 401, body: { detail: 'Invalid or missing API key' } }];
  await reviewQueue.loadQueue();

  assert.ok(
    elements.mainContent.innerHTML.includes('Sign in required'),
    'a 401 should render the prominent auth-required panel, not a small status line',
  );
  assert.strictEqual(
    elements.apiBand.classList.contains('hidden'), false,
    'the API key band should be shown again on a 401, even if previously hidden',
  );
  assert.strictEqual(reviewQueue.state.loaded, false, 'state.loaded must stay false on a 401');
  assert.strictEqual(reviewQueue.state.authRequired, true);

  // --- 2. Subsequent authenticated retry returns 200 -> queue renders normally ---
  storageData[KEY] = 'a-real-key';
  fetchQueue = [{ status: 200, body: goodQueuePayload() }];
  await reviewQueue.loadQueue();

  assert.ok(
    elements.mainContent.innerHTML.includes('TEST'),
    'a successful retry should render the real candidate',
  );
  assert.ok(
    !elements.mainContent.innerHTML.includes('Sign in required'),
    'the auth panel must not remain after a successful load',
  );
  assert.strictEqual(reviewQueue.state.loaded, true);
  assert.strictEqual(reviewQueue.state.authRequired, false);
  assert.strictEqual(reviewQueue.state.queue.length, 1);

  // --- 3. No duplicate candidate rendering or stale auth message on a
  // second successful load ---
  fetchQueue = [{ status: 200, body: goodQueuePayload() }];
  await reviewQueue.loadQueue();

  const cardCount = (elements.mainContent.innerHTML.match(/class="candidate-card"/g) || []).length;
  assert.strictEqual(cardCount, 1, 'a second successful load must replace, not append, the candidate card');
  assert.strictEqual(reviewQueue.state.queue.length, 1, 'state.queue must be replaced, not concatenated');
  assert.ok(
    !elements.mainContent.innerHTML.includes('Sign in required'),
    'no stale auth message should remain after a normal repeat load',
  );

  // --- 4. Bonus: the cross-tab storage-event retry actually fires
  // loadQueue() when a key appears while authRequired is true ---
  elements.mainContent.innerHTML = '';
  fetchQueue = [{ status: 401, body: { detail: 'Invalid or missing API key' } }];
  await reviewQueue.loadQueue();
  assert.strictEqual(reviewQueue.state.authRequired, true);

  fetchQueue = [{ status: 200, body: goodQueuePayload() }];
  storageData[KEY] = 'a-different-real-key';
  assert.ok(storageListeners.length > 0, 'a storage listener should have been registered on load');
  storageListeners.forEach((cb) => cb({ key: KEY, newValue: storageData[KEY] }));
  await new Promise((resolve) => setTimeout(resolve, 0)); // flush the async loadQueue() the listener kicked off

  assert.ok(
    elements.mainContent.innerHTML.includes('TEST'),
    'a storage event (another tab signing in) should auto-retry and render the queue',
  );
  assert.strictEqual(reviewQueue.state.authRequired, false);

  console.log('Review queue auth v1 tests passed');
}

run().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
