// Review Queue persistence + review editing (2026-09-01 session). Root
// cause: GET /candidates/review-queue already returns current_review per
// candidate (Stage A/B, unchanged) -- the frontend just ignored it,
// tracking progress in a session-only counter that reset on every load
// and always starting navigation at index 0. This tests the real
// review_queue.js code (same hand-built DOM/fetch/storage fake as
// tests/review_queue_auth_v1.js, this repo's existing plain-Node .js test
// convention -- no jsdom dependency available), not a reimplementation.
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

function makeFormElement(initialValues) {
  return { _formValues: initialValues };
}

const elements = {
  apiBand: makeElement(),
  mainContent: makeElement(),
  apiKeyInput: makeElement(),
  reviewNote: makeElement(),
  submitRow: makeElement(),
  practicalRejectRow: makeElement(),
  practicalRejectNote: makeElement(),
  // submitReview() reads this via `new FormData(form)` -- real Node has a
  // native FormData that requires a genuine HTML form element, so
  // global.FormData is mocked below to read straight off _formValues
  // instead, letting this test drive the exact same submitReview() code a
  // real edit would call without needing a full form/DOM implementation.
  reviewForm: makeFormElement({
    market_structure: 'bullish', location_read: 'good',
    clear_path_to_target: 'yes', lower_tf_confirmation: 'yes', decision: 'approve',
  }),
};

global.FormData = class {
  constructor(form) {
    this._data = (form && form._formValues) || {};
  }
  get(key) {
    return Object.prototype.hasOwnProperty.call(this._data, key) ? this._data[key] : null;
  }
};

global.document = {
  getElementById: (id) => elements[id] || null,
  // querySelector is only used by submitReview/submitPracticalRejection to
  // read the checked radio out of a REAL form -- not exercised by these
  // persistence/editing tests (which assert on the rendered HTML string
  // and on direct applyReviewResult-driven state, not on live form
  // interaction), so a minimal stub that's never actually called here is
  // enough.
  querySelector: () => null,
  addEventListener: () => {},
};

global.window = { addEventListener: () => {} };
global.localStorage = { getItem: () => 'test-key', setItem: () => {}, removeItem: () => {} };
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

function candidate(overrides = {}) {
  return {
    ticker: 'AAA', source: 'ma_pipeline', signal: 'long', rank: 1,
    entry_price: 100, stop: 95, target: 110, risk_reward: 2,
    confluence_available: true,
    confluence_counts: { favorable: 2, unfavorable: 1, neutral: 4, applicable: 7 },
    confluence_label: 'some confluence',
    bos_confirmed: false, displacement_label: 'WEAK', displacement_score: 20,
    location_label: 'midrange', location_alignment: 'neutral', macro_bias: 'Macro Neutral',
    sweep_confirmed: false, rejection_confirmed: false,
    execution_shadow_ok: true, execution_shadow_reason: 'Recent 4H confirmation remains structurally intact',
    entry_distance_pct: 0.5, setup_key: 'AAA|ma_pipeline|long|95.00|110.00', current_review: null,
    ...overrides,
  };
}

function visualReview(overrides = {}) {
  return {
    id: 1, ticker: 'AAA', source: 'ma_pipeline', setup_key: 'AAA|ma_pipeline|long|95.00|110.00',
    review_type: 'visual', market_structure: 'bullish', location_read: 'good',
    clear_path_to_target: 'yes', lower_tf_confirmation: 'yes',
    practical_rejection_reason: null, decision: 'watch', note: null,
    reviewed_at: '2026-09-01T10:00:00Z',
    ...overrides,
  };
}

function queuePayload(candidates) {
  return {
    mechanism: 'stage1_mechanical_then_confluence_count_v1',
    disclaimer: 'Ranking is unvalidated -- based on signal counts, not track record.',
    snapshot_id: 'test-snapshot',
    count: candidates.length,
    candidates,
  };
}

function candidateCardCount() {
  return (elements.mainContent.innerHTML.match(/class="candidate-card"/g) || []).length;
}

async function run() {
  // --- 1 & 2. Counters/progress hydrate from PERSISTED reviews on load,
  // and navigation resumes at the first unreviewed candidate -- none of
  // this comes from any submission made during this test run. ---
  const bbb = candidate({ ticker: 'BBB', rank: 2, setup_key: 'BBB|ma_pipeline|long|50.00|60.00' });
  const ccc = candidate({
    ticker: 'CCC', rank: 3, setup_key: 'CCC|ma_pipeline|long|30.00|40.00',
    current_review: visualReview({ id: 2, ticker: 'CCC', setup_key: 'CCC|ma_pipeline|long|30.00|40.00', decision: 'approve' }),
  });
  const aaa = candidate({
    current_review: visualReview({ decision: 'watch' }), // already reviewed, WATCH
  });
  fetchQueue = [{ status: 200, body: queuePayload([aaa, bbb, ccc]) }];
  await reviewQueue.loadQueue();

  const counts = reviewQueue.computeCounts(reviewQueue.state.queue);
  assert.strictEqual(counts.watch, 1, 'persisted watch review must be counted on load, not just session submissions');
  assert.strictEqual(counts.approve, 1, 'persisted approve review must be counted on load');
  assert.strictEqual(counts.unreviewed, 1, 'exactly one candidate (BBB) is genuinely unreviewed');
  assert.ok(
    elements.mainContent.innerHTML.includes('Reviewed 2 / 3'),
    'progress header must reflect persisted reviews, not reset to 0 on load',
  );

  assert.strictEqual(reviewQueue.state.index, 1, 'navigation must resume at BBB (index 1), the first unreviewed candidate, not index 0');
  assert.ok(elements.mainContent.innerHTML.includes('BBB'), 'the unreviewed candidate (BBB) should be the one shown');
  assert.strictEqual(candidateCardCount(), 1, 'exactly one candidate card, not the already-reviewed ones too');

  // --- 3. Previously reviewed candidates must not silently re-enter the
  // unreviewed workflow (AAA and CCC, both already reviewed, must never
  // appear in the single-candidate unreviewed-filter stepper). ---
  assert.ok(!elements.mainContent.innerHTML.includes('AAA'), 'AAA is already reviewed -- must not appear in the unreviewed stepper');
  assert.ok(!elements.mainContent.innerHTML.includes('CCC'), 'CCC is already reviewed -- must not appear in the unreviewed stepper');

  // --- 4. Reviewed filter shows a list; opening one pre-populates its
  // existing fields. ---
  reviewQueue.setFilter('reviewed');
  assert.ok(elements.mainContent.innerHTML.includes('AAA'), 'reviewed list should show AAA');
  assert.ok(elements.mainContent.innerHTML.includes('CCC'), 'reviewed list should show CCC');
  assert.ok(!elements.mainContent.innerHTML.includes('BBB'), 'reviewed list must not show the still-unreviewed BBB');

  reviewQueue.openReviewedItem(0); // AAA, watch
  assert.strictEqual(reviewQueue.state.editing, true);
  const editHtml = elements.mainContent.innerHTML;
  assert.ok(/name="decision" value="watch" checked/.test(editHtml), 'the existing decision (watch) must be pre-selected');
  assert.ok(/name="market_structure" value="bullish" checked/.test(editHtml), 'the existing market_structure must be pre-selected');
  assert.ok(/name="location_read" value="good" checked/.test(editHtml), 'the existing location_read must be pre-selected');
  assert.strictEqual(candidateCardCount(), 1);

  // --- 5. Changing a decision updates counters immediately (Watch -> Approve). ---
  const beforeChange = reviewQueue.computeCounts(reviewQueue.state.queue);
  assert.strictEqual(beforeChange.watch, 1);
  assert.strictEqual(beforeChange.approve, 1);

  fetchQueue = [{
    status: 200,
    body: visualReview({ id: 3, decision: 'approve', reviewed_at: '2026-09-01T11:00:00Z' }),
  }];
  await reviewQueue.submitReview('AAA', 'ma_pipeline');

  const afterChange = reviewQueue.computeCounts(reviewQueue.state.queue);
  assert.strictEqual(afterChange.watch, 0, 'Watch count must decrement when AAA changes from watch to approve');
  assert.strictEqual(afterChange.approve, 2, 'Approved count must increment (CCC was already approve, AAA just became approve)');
  assert.strictEqual(reviewQueue.state.editing, false, 'submitting an edit should close editing mode');

  // The reviewed list (now shown again post-edit) should reflect the new
  // decision -- no duplicate row, no stale "watch" label surviving.
  assert.ok(elements.mainContent.innerHTML.includes('decision-approve'), 'AAA row should now show approve styling');
  const aaaRowMatches = (elements.mainContent.innerHTML.match(/AAA/g) || []).length;
  assert.ok(aaaRowMatches <= 2, 'AAA should appear once as a row (ticker + edit label at most), not duplicated'); // ticker span only, no dup rows

  // --- 6. "Reload the page" -- a fresh loadQueue() call reflecting the
  // now-current (latest) review server-side, exactly like a real reload
  // since this page always re-fetches, never trusts client cache. ---
  const aaaReloaded = candidate({ current_review: visualReview({ id: 3, decision: 'approve' }) });
  const bbbReloaded = candidate({ ticker: 'BBB', rank: 2, setup_key: 'BBB|ma_pipeline|long|50.00|60.00' });
  const cccReloaded = candidate({
    ticker: 'CCC', rank: 3, setup_key: 'CCC|ma_pipeline|long|30.00|40.00',
    current_review: visualReview({ id: 2, ticker: 'CCC', setup_key: 'CCC|ma_pipeline|long|30.00|40.00', decision: 'approve' }),
  });
  fetchQueue = [{ status: 200, body: queuePayload([aaaReloaded, bbbReloaded, cccReloaded]) }];
  await reviewQueue.loadQueue();

  const afterReload = reviewQueue.computeCounts(reviewQueue.state.queue);
  assert.strictEqual(afterReload.approve, 2, 'reload must show the updated (approve) decision for AAA, not the stale watch');
  assert.strictEqual(afterReload.watch, 0);
  assert.strictEqual(reviewQueue.state.index, 1, 'reload must again resume at the first unreviewed candidate (BBB)');
  assert.strictEqual(candidateCardCount(), 1);

  // --- 7. A materially new setup_key for the same ticker starts unreviewed. ---
  const aaaNewSetup = candidate({ setup_key: 'AAA|ma_pipeline|long|80.00|130.00', current_review: null });
  fetchQueue = [{ status: 200, body: queuePayload([aaaNewSetup]) }];
  await reviewQueue.loadQueue();
  assert.strictEqual(reviewQueue.computeCounts(reviewQueue.state.queue).unreviewed, 1, 'a new setup_key for AAA must be treated as unreviewed, not inherit the old review');
  assert.strictEqual(reviewQueue.state.index, 0);
  assert.ok(elements.mainContent.innerHTML.includes('AAA'), 'the new-setup AAA should render in the unreviewed stepper');
  // Note: trigger_rule intentionally carries a direction-based convenience
  // default (LONG -> close_above) even on a genuinely unreviewed candidate
  // -- see Kairos trigger-capture spec Section 3 -- so the assertion below
  // is scoped to the actual review-decision fields (the ones this test's
  // stale-carryover bug was about), not a blanket "nothing checked" check.
  const reviewFieldNames = ['market_structure', 'location_read', 'clear_path_to_target', 'lower_tf_confirmation', 'practical_reason', 'decision'];
  for (const name of reviewFieldNames) {
    const fieldGroup = new RegExp(`name="${name}"[^>]*checked`, 'g');
    assert.ok(!fieldGroup.test(elements.mainContent.innerHTML), `a genuinely unreviewed candidate must not have ${name} pre-checked`);
  }

  console.log('Review queue persistence v1 tests passed');
}

run().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
