// Kairos Approved Setups / Watch Setups boards (2026-09 session).
//
// This is presentation/workflow only, built on top of the Review Queue's
// existing, unchanged data path -- GET /candidates/review-queue already
// returns every candidate's setup_key and current_review (the latest
// candidate_visual_reviews row for that exact setup generation, or null),
// see review_queue.js's own header comment for the full history. Both
// boards call that SAME endpoint and filter client-side by
// current_review.decision -- deliberately no new backend endpoint, no
// duplicated strategy/ranking calculation, and this is what makes the
// setup_key identity boundary (a materially new setup_key starts
// unreviewed, old review stays in history) apply here for free: it's
// already true of the field this reads.
//
// One module drives both pages -- decision ('approve' | 'watch') comes
// from <body data-decision="..."> at init time, set per-page in
// approved_setups.html / watch_setups.html.
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  const KEY = 'kairos_scanner_api_key';
  const API_BASE = '/api/v1/scanner';

  const state = {
    decision: 'approve', // 'approve' | 'watch' -- set by initBoard()
    queue: [], // full, unfiltered review-queue payload
    board: [], // queue filtered to state.decision (+ state-grouped for approve)
    disclaimer: '',
    loaded: false,
    authRequired: false,
    // setup_key -> {memory, monitor_state} (ApprovedSetupMemoryRecordOut,
    // Approved board only -- see loadBoard). The frozen anchor
    // computeDisplayState uses instead of live candidate fields
    // (Execution Layer V1 "Finding A" fix), and the real server-computed
    // execution state (monitor_state.state) the card's execution banner
    // reads directly rather than re-deriving anything client-side.
    recordsBySetupKey: new Map(),
  };
  let loadBoardInFlight = false;

  // ---- pure helpers, no DOM -- exported for tests ----
  function isReviewed(item) {
    return !!(item && item.current_review);
  }

  function decisionFor(item) {
    return isReviewed(item) ? item.current_review.decision : null;
  }

  // Approved-setup display state. Informational only -- never written back
  // to the server, never touches candidate_promotions or ENTER_NOW.
  //
  // INVALIDATED and EXTENDED are the only two states this computes.
  // ACTIONABLE is deliberately NOT implemented client-side: the real
  // Execution Layer verdict belongs server-side in
  // approved_setup_monitor_state.state once a monitor exists (see
  // execution_layer_v1_implementation_plan.md) -- approximating it here
  // from a partial signal would be exactly the kind of premature ENTER_NOW/
  // Stage D shortcut this build is explicitly scoped to not take.
  //
  // Execution Layer V1 fix (design audit "Finding A"): this used to anchor
  // INVALIDATED/EXTENDED to the LIVE, every-scan-recomputed candidate
  // fields (item.stop/item.entry_price) -- so if a rescan ever produced a
  // different order-block stop for the same setup_key window, INVALIDATED
  // could be computed against a stop the human never actually approved.
  // Now anchored to the FROZEN approved_setup_memories snapshot (via
  // `memory`, looked up by setup_key -- see loadBoard) instead: approved_
  // stop/approved_entry/entry_proximity_threshold_pct_at_approval, exactly
  // what was true the moment the human approved, never drifting
  // afterward. current_price stays LIVE on purpose (freshness matters for
  // "is this still true right now"; only the ANCHOR needed to stop
  // drifting). Falls back to the live candidate fields only when no
  // memory exists yet for this setup_key -- a defensive path that
  // shouldn't occur for a real approval going forward, since every
  // approve now creates a memory+monitor_state row in the same
  // transaction, but kept so a board render never breaks outright if it
  // does.
  function computeDisplayState(item, memory) {
    const direction = String((item && item.signal) || '').toLowerCase();
    const entry = memory ? memory.approved_entry : (item ? item.entry_price : null);
    const stop = memory ? memory.approved_stop : (item ? item.stop : null);
    const thresholdPct = memory
      ? memory.entry_proximity_threshold_pct_at_approval
      : (item ? item.entry_proximity_threshold_pct : null);
    const price = item ? item.current_price : null;
    if (entry == null || stop == null || price == null || (direction !== 'long' && direction !== 'short')) {
      return 'WAITING_FOR_ENTRY';
    }
    if (direction === 'long' && price <= stop) return 'INVALIDATED';
    if (direction === 'short' && price >= stop) return 'INVALIDATED';

    const distance = Math.abs(price - entry);
    const distPct = entry > 0 ? (distance / entry) * 100 : null;
    if (distPct != null && thresholdPct != null) {
      if (direction === 'long' && price > entry && distPct > thresholdPct) return 'EXTENDED';
      if (direction === 'short' && price < entry && distPct > thresholdPct) return 'EXTENDED';
    }
    return 'WAITING_FOR_ENTRY';
  }

  const STATE_ORDER = ['WAITING_FOR_ENTRY', 'EXTENDED', 'INVALIDATED'];

  // Filters queue to the board's decision, preserving existing rank order.
  // For 'approve', additionally groups by display state (waiting / extended
  // / invalidated) -- a stable partition, so rank order is preserved WITHIN
  // each group, per the task's explicit sorting rule. 'watch' preserves
  // plain rank order with no grouping -- Watch Setups now has real display
  // states too (Watch Lifecycle V1), this is a deliberate scope decision to
  // not also add grouping there, not a leftover assumption that display
  // state is approve-only. recordsBySetupKey (Map<setup_key, {memory, monitor_state}>)
  // is optional -- omitted entirely by any existing caller/test that
  // doesn't care about memory-anchored state, falling back to
  // computeDisplayState's own live-field fallback for every item.
  function filterAndOrder(queue, decision, recordsBySetupKey) {
    const matches = (queue || []).filter((item) => decisionFor(item) === decision);
    if (decision !== 'approve') return matches;
    const groups = { WAITING_FOR_ENTRY: [], EXTENDED: [], INVALIDATED: [] };
    matches.forEach((item) => {
      const record = recordsBySetupKey ? recordsBySetupKey.get(item.setup_key) : null;
      groups[computeDisplayState(item, record ? record.memory : null)].push(item);
    });
    return STATE_ORDER.reduce((acc, s) => acc.concat(groups[s]), []);
  }

  function apiKey() {
    return localStorage.getItem(KEY) || sessionStorage.getItem(KEY) || '';
  }

  function persistApiKey(value) {
    localStorage.setItem(KEY, value);
    sessionStorage.setItem(KEY, value);
  }

  function headers() {
    const key = apiKey();
    return { 'Content-Type': 'application/json', ...(key ? { 'X-API-Key': key } : {}) };
  }

  async function fetchJson(url, options = {}) {
    const response = await fetch(url, {
      ...options,
      credentials: 'same-origin',
      headers: { ...headers(), ...(options.headers || {}) },
    });
    const text = await response.text();
    let payload = null;
    try { payload = text ? JSON.parse(text) : null; } catch { payload = { detail: text }; }
    if (!response.ok) {
      const message = payload && payload.detail ? payload.detail : `Request failed (${response.status})`;
      const err = new Error(Array.isArray(message) ? message.map(m => m.msg || String(m)).join(', ') : String(message));
      err.status = response.status;
      throw err;
    }
    return payload;
  }

  function escapeHtml(value) {
    return String(value == null ? '' : value).replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    })[c]);
  }

  function fmtMoney(value) {
    return value == null ? '--' : `$${Number(value).toFixed(2)}`;
  }

  function fmtNumber(value, digits = 2) {
    return value == null ? '--' : Number(value).toFixed(digits);
  }

  // Entry-Reached Alert V1: the SERVER's persisted verdict
  // (monitor_state.entry_reached_at/entry_reached_price) is the only
  // source of truth here -- never inferred ad hoc in the browser from
  // item.current_price vs. item.entry_price, which would be re-deriving
  // a judgment the monitor already makes deterministically (and could
  // disagree with it, e.g. right after a price briefly wicks back). "The
  // reviewed location is available" is explicitly NOT "confirmation
  // satisfied" -- the pending-confirmation suffix is added only when the
  // row is still WAITING_FOR_TRIGGER, regardless of source_decision.
  function entryReachedLine(record) {
    const ms = record && record.monitor_state;
    if (!ms || !ms.entry_reached_at) return 'Entry not reached';
    const when = new Date(ms.entry_reached_at).toLocaleString(undefined, {
      month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit',
    });
    const price = ms.entry_reached_price != null ? ` at ${fmtMoney(ms.entry_reached_price)}` : '';
    const pending = ms.state === 'WAITING_FOR_TRIGGER' ? ' — confirmation still pending' : '';
    return `Entry reached ${when}${price}${pending}`;
  }

  function fmtPct(value) {
    return value == null ? '--' : `${Number(value).toFixed(2)}%`;
  }

  async function submitApiKey() {
    const input = document.getElementById('apiKeyInput');
    const value = (input.value || '').trim();
    if (!value) return;
    try {
      await fetchJson(`${API_BASE}/session`, {
        method: 'POST',
        body: JSON.stringify({ api_key: value }),
      });
      persistApiKey(value);
      document.getElementById('apiBand').classList.add('hidden');
      await loadBoard();
    } catch (err) {
      setStatus(`Could not connect: ${err.message}`, true);
    }
  }
  window.submitApiKey = submitApiKey;

  function setStatus(message, isError = false) {
    const main = document.getElementById('mainContent');
    main.innerHTML = `<div class="status-line${isError ? ' error' : ''}">${escapeHtml(message)}</div>`;
  }

  function renderAuthRequiredPanel(boardLabel) {
    return `
      <div class="auth-required-panel" role="alert">
        <div class="auth-required-icon" aria-hidden="true">🔒</div>
        <h2>Sign in required</h2>
        <p>Your scanner API key is needed to load ${escapeHtml(boardLabel)}. Paste it in the field above and click Connect.</p>
        <p class="auth-required-hint">Already signed in on another tab? This page will pick that up automatically -- or click Retry.</p>
        <button type="button" onclick="loadBoard()">Retry</button>
      </div>`;
  }

  // Watch Lifecycle V1 section 12: a card built ENTIRELY from the frozen
  // approved_setup_memories snapshot, for a watch-originated setup_key
  // that's no longer in today's live Review Queue response. Every field
  // is deliberately sourced from the memory, never invented -- rank is
  // null (renderCard shows "Not in today's scan" instead of a rank when
  // null), current_price comes from monitor_state.last_live_price (the
  // real last price the server-side monitor itself observed -- no new
  // network call from the board), never from a live candidate row that
  // may not exist anymore.
  function syntheticWatchItem(record) {
    const m = record.memory;
    const ms = record.monitor_state;
    return {
      ticker: m.ticker,
      source: m.source,
      signal: m.direction,
      rank: null,
      entry_price: m.approved_entry,
      stop: m.approved_stop,
      target: m.approved_target,
      risk_reward: m.approved_risk_reward,
      current_price: ms ? ms.last_live_price : null,
      entry_distance_pct: null,
      entry_proximity_threshold_pct: null,
      setup_key: m.setup_key,
      offQueue: true,
      current_review: {
        decision: 'watch',
        review_type: 'visual',
        market_structure: m.market_structure,
        location_read: m.location_read,
        clear_path_to_target: m.clear_path_to_target,
        lower_tf_confirmation: m.lower_tf_confirmation,
        note: m.review_note,
        reviewed_at: m.approved_at,
        trigger_timeframe: m.trigger_timeframe,
        trigger_rule: m.trigger_rule,
        trigger_level: m.trigger_level,
        trigger_reason: m.trigger_reason,
        confirmation_timeframe: m.confirmation_timeframe,
        confirmation_rule: m.confirmation_rule,
        confirmation_level: m.confirmation_level,
        confirmed_candle_time: m.confirmed_candle_time,
        confirmation_note: m.confirmation_note,
      },
    };
  }

  async function loadBoard() {
    if (loadBoardInFlight) return;
    loadBoardInFlight = true;
    const boardLabel = state.decision === 'approve' ? 'approved setups' : 'watch setups';
    setStatus(`Loading ${boardLabel}…`);
    try {
      const result = await fetchJson(`${API_BASE}/candidates/review-queue`);
      state.queue = result.candidates || [];
      state.disclaimer = result.disclaimer || '';
      // Watch Lifecycle V1: fetched for BOTH boards now -- a watch review
      // with a complete trigger contract creates a real
      // approved_setup_memories row too (source_decision="watch"), see
      // watch_lifecycle_v1_audit.md section 2/3. No cross-contamination
      // between boards needed here: a watch-originated record's paired
      // candidate always has current_review.decision==="watch" (that's
      // literally what created it), so filterAndOrder's own
      // decisionFor(item)===decision check already keeps it off the
      // Approved board for free -- same for an approve-originated record
      // never showing on Watch. A failure fetching memories must never
      // break the whole board -- falls back to computeDisplayState's own
      // live-field fallback for every item, same as before this fetch
      // existed.
      // include_inactive=true + keep only the MOST RECENT record per
      // setup_key (by approved_at): the default (active-only) fetch would
      // silently hide a REAL, server-confirmed terminal verdict
      // (INVALIDATED/SUPERSEDED) the instant it happens, since neither is
      // in ACTIVE_MONITOR_STATES -- the board would then fall all the way
      // back to computeDisplayState's live-field heuristic instead of
      // showing what the monitor actually determined. Taking the latest
      // row per setup_key (not the full history) keeps this bounded and
      // correct: an active lifecycle shows its real live state; a
      // just-terminated one still shows its real terminal state until a
      // fresh human review creates a newer memory for the same setup_key,
      // at which point that newer one naturally becomes "most recent".
      try {
        const records = await fetchJson(`${API_BASE}/candidates/approved-setup-memory?include_inactive=true`);
        const bySetupKey = new Map();
        (records || []).forEach((record) => {
          const key = record.memory.setup_key;
          const existing = bySetupKey.get(key);
          if (!existing || new Date(record.memory.approved_at) > new Date(existing.memory.approved_at)) {
            bySetupKey.set(key, record);
          }
        });
        state.recordsBySetupKey = bySetupKey;
      } catch (memErr) {
        state.recordsBySetupKey = new Map();
      }
      state.board = filterAndOrder(state.queue, state.decision, state.recordsBySetupKey);
      // Watch Lifecycle V1 section 12: Watch monitoring must NOT depend on
      // the ticker remaining in today's live Review Queue -- the queue
      // rotates intraday on mechanical gates that have nothing to do with
      // whether Kairos is still watching a human-stated condition. Any
      // active (non-WITHDRAWN) watch-originated record whose setup_key
      // isn't already represented by a queue-driven card gets a
      // synthesized one, built entirely from the frozen memory snapshot
      // -- never from a live candidate row that may no longer exist.
      if (state.decision === 'watch') {
        const representedKeys = new Set(state.board.map((item) => item.setup_key));
        const offQueueItems = [];
        state.recordsBySetupKey.forEach((record) => {
          if (record.memory.source_decision !== 'watch') return;
          if (representedKeys.has(record.memory.setup_key)) return;
          if (!record.monitor_state || record.monitor_state.state === 'WITHDRAWN') return;
          offQueueItems.push(syntheticWatchItem(record));
        });
        state.board = state.board.concat(offQueueItems);
      }
      state.loaded = true;
      state.authRequired = false;
      render();
    } catch (err) {
      if (err.status === 401) {
        state.authRequired = true;
        document.getElementById('apiBand').classList.remove('hidden');
        document.getElementById('mainContent').innerHTML = renderAuthRequiredPanel(boardLabel);
        const input = document.getElementById('apiKeyInput');
        if (input && input.focus) input.focus();
      } else {
        setStatus(`Could not load ${boardLabel}: ${err.message}`, true);
      }
    } finally {
      loadBoardInFlight = false;
    }
  }

  // Same cross-tab auto-retry as review_queue.js.
  if (typeof window !== 'undefined' && window.addEventListener) {
    window.addEventListener('storage', (event) => {
      if (event.key === KEY && event.newValue && state.authRequired) {
        loadBoard();
      }
    });
  }

  const MARKET_STRUCTURE_LABELS = { bullish: 'Bullish', bearish: 'Bearish', range: 'Range' };
  const LOCATION_LABELS = { good: 'Good', neutral: 'Neutral', bad: 'Bad' };
  const CLEAR_PATH_LABELS = { yes: 'Yes', no: 'No' };
  const LOWER_TF_LABELS = { yes: 'Yes', not_yet: 'Not yet' };
  const STATE_LABELS = {
    WAITING_FOR_ENTRY: 'Waiting',
    EXTENDED: 'Extended',
    INVALIDATED: 'Invalidated',
    ACTIONABLE: 'Actionable',
    STALE: 'Needs Fresh Review',
  };

  // Maps the REAL server-computed execution state
  // (approved_setup_monitor_state.state, Execution Layer V1) onto this
  // board's existing display vocabulary. This is now the SOURCE OF TRUTH
  // for ACTIONABLE/STALE (states the old client-side heuristic never had
  // at all) as well as EXTENDED/INVALIDATED (states the old heuristic
  // only ever approximated from live, drifting fields -- see
  // computeDisplayState's own "Finding A" comment). APPROVED/
  // WAITING_FOR_TRIGGER/CONFIRMED/TRIGGER_SATISFIED are four real,
  // distinct server states that collapse to ONE product-facing "waiting"
  // bucket here -- execution_layer_v1_implementation_plan.md section 5:
  // "APPROVED == the product-facing WAITING label".
  const MONITOR_STATE_DISPLAY = {
    APPROVED: 'WAITING_FOR_ENTRY',
    WAITING_FOR_TRIGGER: 'WAITING_FOR_ENTRY',
    CONFIRMED: 'WAITING_FOR_ENTRY',
    TRIGGER_SATISFIED: 'WAITING_FOR_ENTRY',
    ACTIONABLE: 'ACTIONABLE',
    EXTENDED: 'EXTENDED',
    STALE: 'STALE',
    INVALIDATED: 'INVALIDATED',
    SUPERSEDED: 'SUPERSEDED',
  };

  // The single source of truth for a card's display state: prefers the
  // REAL server-computed monitor_state.state when a record exists (see
  // loadBoard); falls back to computeDisplayState's own live-field
  // heuristic only when it doesn't -- shouldn't happen for any setup
  // approved going forward (every approve creates a memory+monitor_state
  // row in the same transaction), but kept so a render never breaks
  // outright if it somehow does.
  function displayStateFor(item, record) {
    const monitorState = record && record.monitor_state ? record.monitor_state.state : null;
    if (monitorState && MONITOR_STATE_DISPLAY[monitorState]) {
      return MONITOR_STATE_DISPLAY[monitorState];
    }
    return computeDisplayState(item, record ? record.memory : null);
  }

  // Watch Lifecycle V1 section 3/13: a watch review with NO complete
  // trigger contract stays passive by design (see
  // watch_lifecycle_v1_audit.md section 9) -- no memory is ever created
  // for it, so `record` is undefined here. Distinguishing this from a
  // genuinely off-record item matters: it must never be silently
  // conflated with "DO NOT ENTER -- Kairos is monitoring" (which implies
  // real monitoring exists).
  function isManualReviewRequired(item, record) {
    if (state.decision !== 'watch' || record) return false;
    const cr = item.current_review;
    return !!cr && cr.decision === 'watch' && cr.lower_tf_confirmation === 'not_yet'
      && (cr.trigger_rule == null || cr.trigger_level == null);
  }

  function stateBadge(item) {
    const record = state.recordsBySetupKey.get(item.setup_key);
    if (isManualReviewRequired(item, record)) {
      return `<span class="state-pill state-MANUAL_REVIEW">Manual Review Required</span>`;
    }
    const s = displayStateFor(item, record);
    return `<span class="state-pill state-${s}">${STATE_LABELS[s]}</span>`;
  }

  function stateNotice(item) {
    const record = state.recordsBySetupKey.get(item.setup_key);
    const isWatch = state.decision === 'watch';
    if (isManualReviewRequired(item, record)) {
      return `<div class="state-notice notice-manual-review">MANUAL REVIEW REQUIRED &mdash; no objective trigger stored. Kairos is not monitoring this setup.</div>`;
    }
    const s = displayStateFor(item, record);
    if (s === 'EXTENDED') {
      return isWatch
        ? `<div class="state-notice notice-extended">Confirmation came too late, or the execution window is gone. Do not chase.</div>`
        : `<div class="state-notice notice-extended">Extended &mdash; do not chase</div>`;
    }
    if (s === 'INVALIDATED') {
      return isWatch
        ? `<div class="state-notice notice-invalidated">Original thesis invalidated before confirmation.</div>`
        : `<div class="state-notice notice-invalidated">Setup invalidated</div>`;
    }
    if (s === 'SUPERSEDED') {
      return `<div class="state-notice notice-superseded">Scanner structure changed for this ticker. New review required.</div>`;
    }
    if (s === 'STALE') {
      return `<div class="state-notice notice-stale">Execution evidence has gone stale &mdash; needs a fresh human review before this can become actionable again</div>`;
    }
    if (s === 'WAITING_FOR_ENTRY') {
      const monitored = !!(record && record.monitor_state);
      const text = monitored
        ? 'DO NOT ENTER &mdash; Kairos is monitoring for your stated execution condition.'
        : 'DO NOT ENTER &mdash; this setup has no recorded execution evidence yet.';
      return `<div class="state-notice notice-waiting">${text}</div>`;
    }
    return '';
  }

  // ACTIONABLE must visually DOMINATE the card
  // (execution_layer_v1_implementation_plan.md section 13) -- a
  // full-width, success-colored banner, not a small pill/notice like the
  // other states. Only ever rendered from a REAL server-computed
  // ACTIONABLE monitor_state -- never approximated client-side.
  function actionableBanner(item, record) {
    if (displayStateFor(item, record) !== 'ACTIONABLE') return '';
    const monitorState = record.monitor_state;
    const rrText = monitorState.current_rr_at_last_check != null ? fmtNumber(monitorState.current_rr_at_last_check) : '--';
    // Watch Lifecycle V1 section 9/13: for a watch-originated record,
    // this IS the "TRIGGER SATISFIED / HANDED OFF" moment -- the human's
    // stated condition occurred AND the safety gates independently still
    // say the opportunity is real. Distinct copy from the approve-
    // origin banner (which never had a "trigger" to report satisfying),
    // same box/styling, same underlying ACTIONABLE state -- no
    // fabricated approve review, database history still shows the
    // original decision was watch (candidate_visual_reviews is
    // untouched).
    const isWatchHandoff = record.memory.source_decision === 'watch';
    const label = isWatchHandoff ? 'TRIGGER SATISFIED' : 'ACTIONABLE';
    // triggerSummaryText/confirmationSummaryText already escapeHtml their
    // own output -- do not double-escape here.
    const conditionText = triggerSummaryText(record.memory) || confirmationSummaryText(record.memory) || '30m condition';
    const text = isWatchHandoff
      ? `Your stated ${conditionText} occurred. Kairos is evaluating this setup -- it currently reads within an acceptable execution window.`
      : 'Kairos has confirmed your stated execution condition, and price remains within an acceptable window.';
    return `
        <div class="actionable-banner">
          <div class="actionable-banner-label">${label}</div>
          <div class="actionable-banner-text">${text}</div>
          <div class="actionable-banner-metrics">Current ${fmtMoney(item.current_price)} &middot; Stop ${fmtMoney(record.memory.approved_stop)} &middot; Target ${fmtMoney(record.memory.approved_target)} &middot; Current R:R ${rrText}</div>
          <div class="actionable-banner-disclaimer">This is not investment advice &mdash; you decide whether and how to act.</div>
        </div>`;
  }

  function editHref(item) {
    return `/review-queue?ticker=${encodeURIComponent(item.ticker)}&source=${encodeURIComponent(item.source || '')}`;
  }

  const TRIGGER_RULE_LABELS = { close_above: 'close above', close_below: 'close below' };

  function triggerSummaryText(cr) {
    if (!cr || cr.trigger_rule == null || cr.trigger_level == null) return '';
    const rule = TRIGGER_RULE_LABELS[cr.trigger_rule] || cr.trigger_rule;
    return `${escapeHtml(cr.trigger_timeframe || '')} ${escapeHtml(rule)} ${fmtMoney(cr.trigger_level)}`;
  }

  // Status line reflects the REAL server monitor now (Execution Layer
  // V1) -- "Kairos is monitoring" only when a real, active monitor_state
  // record exists for this setup; "Not monitored yet" stays the honest
  // fallback for the defensive no-record case (shouldn't occur for any
  // setup approved going forward, but never overclaim if it does).
  function monitorStatusLine(record) {
    return record && record.monitor_state
      ? 'Status: Kairos is monitoring this'
      : 'Status: Not monitored yet';
  }

  // Watch Lifecycle V1 (found via real-browser verification, not in the
  // original spec list): once the trigger has actually fired, this block
  // MUST stop claiming "waiting for this to happen" / "Kairos is
  // monitoring this" -- that directly contradicts the TRIGGER
  // SATISFIED/HANDED OFF banner rendered right above it by
  // actionableBanner(). monitor_state.trigger_satisfied_at is the
  // authoritative signal (frozen once written server-side, see
  // candidates_router.py's trigger_satisfied_at/_bar_time/_price columns)
  // -- same pattern confirmationBlock already uses for "already
  // happened" framing, applied here for the SAME reason.
  function triggerBlock(cr, record) {
    if (!cr || cr.trigger_rule == null || cr.trigger_level == null) return '';
    const ms = record && record.monitor_state;
    if (ms && ms.trigger_satisfied_at) {
      const when = ms.trigger_satisfied_bar_time || ms.trigger_satisfied_at;
      return `
          <div class="execution-trigger">
            <div class="execution-trigger-label">Execution Trigger &mdash; satisfied</div>
            <div class="execution-trigger-value">${triggerSummaryText(cr)}</div>
            ${cr.trigger_reason ? `<div class="execution-trigger-reason">${escapeHtml(cr.trigger_reason)}</div>` : ''}
            <div class="execution-trigger-when">Satisfied ${escapeHtml(new Date(when).toLocaleString())}${ms.trigger_satisfied_price != null ? ` at ${fmtMoney(ms.trigger_satisfied_price)}` : ''}</div>
          </div>`;
    }
    return `
          <div class="execution-trigger">
            <div class="execution-trigger-label">Execution Trigger &mdash; waiting for this to happen</div>
            <div class="execution-trigger-value">${triggerSummaryText(cr)}</div>
            ${cr.trigger_reason ? `<div class="execution-trigger-reason">${escapeHtml(cr.trigger_reason)}</div>` : ''}
            <div class="execution-trigger-status">${monitorStatusLine(record)}</div>
          </div>`;
  }

  const CONFIRMATION_RULE_LABELS = { close_above: 'close above', close_below: 'close below' };

  function confirmationSummaryText(cr) {
    if (!cr || cr.confirmation_rule == null || cr.confirmation_level == null) return '';
    const rule = CONFIRMATION_RULE_LABELS[cr.confirmation_rule] || cr.confirmation_rule;
    return `${escapeHtml(cr.confirmation_timeframe || '')} ${escapeHtml(rule)} ${fmtMoney(cr.confirmation_level)}`;
  }

  // Observed Confirmation Anchor -- a DIFFERENT concept from the trigger
  // block above (already-occurred, past tense, vs. a future condition) --
  // deliberately distinct label/copy so a novice never reads the two the
  // same way. See execution_layer_v1_implementation_plan.md section 1.
  function confirmationBlock(cr, record) {
    if (!cr || cr.confirmation_rule == null || cr.confirmation_level == null) return '';
    const when = cr.confirmed_candle_time || cr.reviewed_at;
    return `
          <div class="execution-confirmation">
            <div class="execution-confirmation-label">Observed Confirmation &mdash; already happened</div>
            <div class="execution-confirmation-value">${confirmationSummaryText(cr)}</div>
            ${when ? `<div class="execution-confirmation-when">Observed ${escapeHtml(new Date(when).toLocaleString())}</div>` : ''}
            ${cr.confirmation_note ? `<div class="execution-confirmation-note">${escapeHtml(cr.confirmation_note)}</div>` : ''}
            <div class="execution-confirmation-status">${monitorStatusLine(record)}</div>
          </div>`;
  }

  function renderCard(item) {
    const cr = item.current_review || {};
    const record = state.recordsBySetupKey.get(item.setup_key);
    const chartUrl = `https://www.tradingview.com/symbols/${encodeURIComponent(item.ticker)}/`;
    return `
      <div class="setup-card">
        <div class="setup-head">
          <div>
            <span class="setup-ticker">${escapeHtml(item.ticker)}</span>
            <span class="direction-pill">${escapeHtml((item.signal || '').toUpperCase())}</span>
            ${stateBadge(item)}
          </div>
          <span class="setup-rank">${item.rank != null ? `Rank #${item.rank}` : 'Not in today’s scan'} &middot; ${escapeHtml(item.source || '')}</span>
        </div>

        ${actionableBanner(item, record)}

        <div class="current-price-row">
          <span class="current-price-label">Current Price</span>
          <span class="current-price-value">${fmtMoney(item.current_price)}</span>
        </div>

        <div class="entry-reached-line">${escapeHtml(entryReachedLine(record))}</div>

        <div class="metric-grid primary-metrics">
          <div class="metric"><div class="metric-label">Entry</div><div class="metric-value">${fmtMoney(item.entry_price)}</div></div>
          <div class="metric"><div class="metric-label">Stop Loss</div><div class="metric-value">${fmtMoney(item.stop)}</div></div>
          <div class="metric"><div class="metric-label">Target</div><div class="metric-value">${fmtMoney(item.target)}</div></div>
          <div class="metric"><div class="metric-label">R:R</div><div class="metric-value">${fmtNumber(item.risk_reward)}</div></div>
        </div>
        <div class="metric-grid secondary-metrics">
          <div class="metric"><div class="metric-label">Entry distance</div><div class="metric-value">${fmtPct(item.entry_distance_pct)}</div></div>
        </div>

        ${stateNotice(item)}

        <div class="review-context">
          <div class="review-meta">Reviewed ${escapeHtml(new Date(cr.reviewed_at).toLocaleString())}</div>
          ${cr.note ? `<div class="review-note">${escapeHtml(cr.note)}</div>` : ''}
          <div class="signal-grid">
            <div class="signal-row"><span>Market structure</span><span class="read">${escapeHtml(MARKET_STRUCTURE_LABELS[cr.market_structure] || cr.market_structure || '--')}</span></div>
            <div class="signal-row"><span>Location</span><span class="read">${escapeHtml(LOCATION_LABELS[cr.location_read] || cr.location_read || '--')}</span></div>
            <div class="signal-row"><span>Clear path to target</span><span class="read">${escapeHtml(CLEAR_PATH_LABELS[cr.clear_path_to_target] || cr.clear_path_to_target || '--')}</span></div>
            <div class="signal-row"><span>Lower-TF confirmation</span><span class="read">${escapeHtml(LOWER_TF_LABELS[cr.lower_tf_confirmation] || cr.lower_tf_confirmation || '--')}</span></div>
          </div>
          ${triggerBlock(cr, record)}
          ${confirmationBlock(cr, record)}
        </div>

        <div class="card-actions">
          <a class="nav-link" href="${chartUrl}" target="_blank" rel="noopener">Open Chart on TradingView &#8599;</a>
          <a class="nav-link" href="${editHref(item)}">Edit Review</a>
        </div>
      </div>`;
  }

  function emptyStateMessage() {
    if (state.decision === 'approve') {
      return `No approved setups yet. Approve a candidate in the <a class="nav-link" href="/review-queue">Review Queue</a> to see it here.`;
    }
    return `No setups on watch yet. Mark a candidate Watch in the <a class="nav-link" href="/review-queue">Review Queue</a> to see it here.`;
  }

  function render() {
    const main = document.getElementById('mainContent');
    if (!state.loaded) return;
    const countLabel = state.decision === 'approve' ? 'approved setup' : 'watch setup';
    const header = `<div class="disclaimer">${escapeHtml(state.disclaimer)}</div>
      <div class="progress-row"><span class="progress-count">${state.board.length} ${countLabel}${state.board.length === 1 ? '' : 's'}</span></div>`;
    if (!state.board.length) {
      main.innerHTML = `${header}<div class="status-line">${emptyStateMessage()}</div>`;
      return;
    }
    main.innerHTML = `${header}<div class="setup-list">${state.board.map(renderCard).join('')}</div>`;
  }

  // Root cause of the "Approved Setups not loading" production bug
  // (2026-09 session): this used to be an exported initBoard(decision)
  // function that EACH board page had to call itself from a SEPARATE
  // inline <script>initBoard('approve')</script> tag after the external
  // setup_board.js tag. review_queue.js never has this problem -- it
  // wires its own DOMContentLoaded listener internally, as part of the
  // SAME script, so there is nothing else that has to load or run for it
  // to work. The two-script split here meant that if the external
  // setup_board.js tag failed to execute for ANY reason (slow/flaky
  // network, a blocked/failed script load, a browser extension) -- as
  // opposed to a normal successful load, which this session's own
  // testing repeatedly reproduced working -- the second, inline script
  // would throw "initBoard is not defined" with nothing to catch it, and
  // the page would stay frozen on its static "Loading approved setups…"
  // HTML forever: no error shown, no retry offered, matching exactly
  // what was reported. Also: initBoard(decision) took the decision as an
  // explicit argument the caller had to pass correctly -- despite this
  // file's own header comment claiming decision "comes from <body
  // data-decision=...>", the code never actually read that attribute.
  // Fixed the same way review_queue.js already does it: one script,
  // self-initializing, reading <body data-decision> itself, wrapped so
  // an unexpected init-time exception is surfaced as visible text
  // instead of leaving the page stuck on its loading state.
  if (typeof document !== 'undefined' && document.addEventListener) {
    document.addEventListener('DOMContentLoaded', () => {
      try {
        const decision = (document.body && document.body.dataset && document.body.dataset.decision) || 'approve';
        state.decision = decision;
        if (apiKey()) {
          document.getElementById('apiBand').classList.add('hidden');
        }
        loadBoard();
      } catch (err) {
        setStatus(`Could not initialize this page: ${err.message}`, true);
      }
    });
  }
  window.loadBoard = loadBoard;

  // Exposed for tests (tests/setup_board_v1.js) -- real browser usage never
  // touches this return value, it only matters under module.exports.
  return {
    state,
    loadBoard,
    renderAuthRequiredPanel,
    submitApiKey,
    isReviewed,
    decisionFor,
    computeDisplayState,
    filterAndOrder,
  };
});
