// Stage C: Kairos Review Queue frontend. Deliberately a standalone page,
// not a filter/tab bolted onto candidate_dashboard.js's card wall -- see
// the 2026-08-31 session's design pass for why. Talks only to the two
// Stage A/B endpoints (GET /candidates/review-queue, POST
// /candidates/{ticker}/visual-review) plus the existing session-cookie
// auth flow (POST /session) already used by candidates.html -- same-origin,
// so a session established there is reused here automatically.
(function (root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    factory();
  }
})(typeof self !== 'undefined' ? self : this, function () {
  const KEY = 'kairos_scanner_api_key';
  const API_BASE = '/api/v1/scanner';

  // Persistence + editing fix (2026-09-01 session): root cause was that
  // GET /candidates/review-queue was ALREADY returning current_review per
  // candidate (Stage A/B, unchanged, no backend work needed here) -- the
  // frontend just never looked at it. Progress/tallies were tracked in a
  // session-only state.tally that reset on every load, and navigation
  // always started at index 0 regardless of what was already reviewed.
  // Fixed by deriving everything from state.queue[i].current_review (the
  // real, persisted source of truth) instead of separate mutable counters,
  // and adding an editing path for already-reviewed candidates. Setup-key
  // binding itself needed zero changes: current_review is already looked
  // up by setup_key server-side, so a materially different setup_key
  // (structural stop/target actually moved) already has no matching row
  // and is already correctly treated as unreviewed -- this file just
  // needs to keep trusting that field, not re-derive it from ticker.
  const state = {
    queue: [],
    filter: 'unreviewed', // 'unreviewed' | 'reviewed'
    index: 0, // index into state.queue of the candidate currently shown
    editing: false, // true only while viewing a reviewed candidate's form pre-populated for editing
    disclaimer: '',
    loaded: false,
    authRequired: false,
  };
  // Guards against two overlapping loadQueue() calls (e.g. a storage-event
  // retry firing while the Connect button's own retry is still in flight)
  // racing to write mainContent.innerHTML -- whichever resolved last would
  // still win either way (full replace, never append), so this isn't a
  // correctness bug on its own, just wasted duplicate fetches worth
  // skipping.
  let loadQueueInFlight = false;

  // Early practical-disqualification path (2026-09-01 session): a real
  // usage gap -- a candidate can be practically untradeable (options too
  // expensive, poor liquidity/spread) before a human ever reaches chart
  // review. Backend already refuses to accept these mixed with any
  // visual-read field (candidates_router.py's record_candidate_visual_review),
  // so this UI path submits ONLY {decision:"reject", practical_rejection_reason}
  // -- never touches the visual-review form's fields at all.
  const PRACTICAL_REASON_LABELS = {
    options_too_expensive: 'Options too expensive',
    poor_option_liquidity: 'Poor option liquidity/spread',
    other: 'Other practical disqualifier',
  };

  // ---- pure helpers over state.queue, no DOM -- exported for tests ----
  function isReviewed(item) {
    return !!(item && item.current_review);
  }

  function decisionFor(item) {
    return isReviewed(item) ? item.current_review.decision : null;
  }

  function computeCounts(queue) {
    const counts = { approve: 0, watch: 0, reject: 0, unreviewed: 0 };
    (queue || []).forEach((item) => {
      const d = decisionFor(item);
      if (d === 'approve') counts.approve += 1;
      else if (d === 'watch') counts.watch += 1;
      else if (d === 'reject') counts.reject += 1;
      else counts.unreviewed += 1;
    });
    return counts;
  }

  function firstUnreviewedIndex(queue) {
    return (queue || []).findIndex((item) => !isReviewed(item));
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

  function checkedIf(condition) {
    return condition ? 'checked' : '';
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
      await loadQueue();
    } catch (err) {
      setStatus(`Could not connect: ${err.message}`, true);
    }
  }
  window.submitApiKey = submitApiKey;

  function setStatus(message, isError = false) {
    const main = document.getElementById('mainContent');
    main.innerHTML = `<div class="status-line${isError ? ' error' : ''}">${escapeHtml(message)}</div>`;
  }

  // Production regression (2026-09-01 session): a fresh /review-queue load
  // with no session/key yet established on that browser got a real 401 from
  // GET /candidates/review-queue -- correct, expected behavior (the auth
  // mechanism itself was investigated and found to have no bug: same
  // localStorage key name and the same session cookie, path "/", are
  // already shared with candidates.html), but the OLD handling here just
  // set a small one-line status message, easy to mistake for a broken or
  // empty queue rather than "please sign in." This is a UI-prominence fix
  // only -- auth mechanism/API behavior untouched.
  function renderAuthRequiredPanel() {
    return `
      <div class="auth-required-panel" role="alert">
        <div class="auth-required-icon" aria-hidden="true">🔒</div>
        <h2>Sign in required</h2>
        <p>Your scanner API key is needed to load the review queue. Paste it in the field above and click Connect.</p>
        <p class="auth-required-hint">Already signed in on another tab? This page will pick that up automatically -- or click Retry.</p>
        <button type="button" onclick="loadQueue()">Retry</button>
      </div>`;
  }

  async function loadQueue() {
    if (loadQueueInFlight) return;
    loadQueueInFlight = true;
    setStatus('Loading review queue…');
    try {
      const result = await fetchJson(`${API_BASE}/candidates/review-queue`);
      state.queue = result.candidates || [];
      state.disclaimer = result.disclaimer || '';
      state.filter = 'unreviewed';
      state.editing = false;
      // Resume at the first unreviewed candidate (persisted, not session-
      // only) -- falls back to index 0 only when literally nothing is
      // unreviewed, in which case render() shows the completion state.
      const firstUnreviewed = firstUnreviewedIndex(state.queue);
      state.index = firstUnreviewed >= 0 ? firstUnreviewed : 0;
      state.loaded = true;
      state.authRequired = false;

      // Deep-link support (2026-09 Approved/Watch Setups build): "Edit
      // Review" on those boards links here as
      // /review-queue?ticker=X&source=Y. If that candidate is already
      // reviewed, jump straight to its edit view instead of the default
      // first-unreviewed resume point. No-ops (and is a no-op in the
      // plain-Node test fixtures, which never set window.location) when
      // there's no query string, or the ticker doesn't resolve to a
      // reviewed candidate in the current queue.
      const search = (typeof window !== 'undefined' && window.location && window.location.search) || '';
      const deepTicker = search && typeof URLSearchParams !== 'undefined'
        ? new URLSearchParams(search).get('ticker')
        : null;
      if (deepTicker) {
        const deepSource = new URLSearchParams(search).get('source') || '';
        const deepIdx = state.queue.findIndex((item) => item.ticker === deepTicker && (item.source || '') === deepSource);
        if (deepIdx >= 0 && isReviewed(state.queue[deepIdx])) {
          state.filter = 'reviewed';
          state.index = deepIdx;
          state.editing = true;
        }
      }

      render();
    } catch (err) {
      if (err.status === 401) {
        state.authRequired = true;
        document.getElementById('apiBand').classList.remove('hidden');
        document.getElementById('mainContent').innerHTML = renderAuthRequiredPanel();
        const input = document.getElementById('apiKeyInput');
        if (input && input.focus) input.focus();
      } else {
        setStatus(`Could not load review queue: ${err.message}`, true);
      }
    } finally {
      loadQueueInFlight = false;
    }
  }

  // Auto-retry when a valid key becomes available through ANY means this
  // page didn't itself trigger -- most commonly: the user has
  // candidates.html open in another tab and signs in there. The browser's
  // storage event fires in OTHER tabs of the same origin when localStorage
  // changes (never in the tab that made the change itself, which is
  // already covered by submitApiKey()'s own retry after a successful
  // /session call) -- exactly the cross-tab case worth covering here,
  // with zero polling.
  if (typeof window !== 'undefined' && window.addEventListener) {
    window.addEventListener('storage', (event) => {
      if (event.key === KEY && event.newValue && state.authRequired) {
        loadQueue();
      }
    });
  }

  function progressHeader() {
    const counts = computeCounts(state.queue);
    const reviewedTotal = counts.approve + counts.watch + counts.reject;
    return `
      <div class="progress-row">
        <span class="progress-count">Reviewed ${reviewedTotal} / ${state.queue.length} &middot; Remaining ${counts.unreviewed}</span>
        <div class="tally">
          <span class="tally-chip approve">Approved ${counts.approve}</span>
          <span class="tally-chip watch">Watch ${counts.watch}</span>
          <span class="tally-chip reject">Rejected ${counts.reject}</span>
        </div>
      </div>
      <div class="filter-row" role="tablist">
        <button type="button" class="filter-tab ${state.filter === 'unreviewed' ? 'active' : ''}" onclick="setFilter('unreviewed')">Unreviewed (${counts.unreviewed})</button>
        <button type="button" class="filter-tab ${state.filter === 'reviewed' ? 'active' : ''}" onclick="setFilter('reviewed')">Reviewed (${reviewedTotal})</button>
      </div>`;
  }

  function setFilter(filter) {
    state.filter = filter;
    state.editing = false;
    if (filter === 'unreviewed') {
      const idx = firstUnreviewedIndex(state.queue);
      state.index = idx >= 0 ? idx : 0;
    }
    render();
  }
  window.setFilter = setFilter;

  function openReviewedItem(queueIndex) {
    state.index = queueIndex;
    state.editing = true;
    render();
  }
  window.openReviewedItem = openReviewedItem;

  function closeEditing() {
    state.editing = false;
    render();
  }
  window.closeEditing = closeEditing;

  function renderReviewedList() {
    const reviewedEntries = state.queue
      .map((item, idx) => ({ item, idx }))
      .filter(({ item }) => isReviewed(item));
    if (!reviewedEntries.length) {
      return `<div class="status-line">No candidates reviewed yet.</div>`;
    }
    const rows = reviewedEntries.map(({ item, idx }) => {
      const cr = item.current_review;
      const label = cr.review_type === 'practical_rejection'
        ? `Practically rejected — ${PRACTICAL_REASON_LABELS[cr.practical_rejection_reason] || cr.practical_rejection_reason}`
        : cr.decision.charAt(0).toUpperCase() + cr.decision.slice(1);
      const triggerText = triggerSummaryText(cr);
      const confirmationText = confirmationSummaryText(cr);
      return `
        <div class="reviewed-row" onclick="openReviewedItem(${idx})">
          <span class="reviewed-ticker">${escapeHtml(item.ticker)}</span>
          <span class="reviewed-decision decision-${escapeHtml(cr.decision)}">${escapeHtml(label)}</span>
          ${triggerText ? `<span class="reviewed-trigger">Waiting for: ${escapeHtml(triggerText)}</span>` : ''}
          ${confirmationText ? `<span class="reviewed-confirmation">Confirmed: ${escapeHtml(confirmationText)}</span>` : ''}
          <span class="reviewed-time">${escapeHtml(new Date(cr.reviewed_at).toLocaleString())}</span>
          <span class="reviewed-edit">Edit &rsaquo;</span>
        </div>`;
    }).join('');
    return `<div class="reviewed-list">${rows}</div>`;
  }

  // "Waiting for: 30m close above $X" -- the human-readable rendering of
  // a stored trigger, shared by the reviewed list and the candidate card
  // itself. A trigger is a FUTURE condition (lower_tf_confirmation=
  // "not_yet") -- Kairos genuinely monitors it now (Execution Layer V1),
  // but this copy stays deliberately informational/instruction-framed
  // rather than claiming real-time precision until that monitor has real
  // production track record.
  function triggerSummaryText(cr) {
    if (!cr || cr.trigger_rule == null || cr.trigger_level == null) return null;
    const ruleText = cr.trigger_rule === 'close_above' ? 'above' : 'below';
    const timeframe = cr.trigger_timeframe || '30m';
    return `${timeframe} close ${ruleText} ${fmtMoney(cr.trigger_level)}`;
  }

  // "Confirmed: 30m close above $X" -- the human-readable rendering of a
  // stored OBSERVED CONFIRMATION ANCHOR (lower_tf_confirmation="yes").
  // Deliberately worded in the PAST tense and a distinct label from
  // triggerSummaryText's future-tense "Waiting for:" -- these are two
  // different concepts (an already-occurred event the reviewer observed,
  // vs. a future event Kairos is waiting for) and must never read the
  // same way to a novice. See execution_layer_v1_implementation_plan.md
  // section 1.
  function confirmationSummaryText(cr) {
    if (!cr || cr.confirmation_rule == null || cr.confirmation_level == null) return null;
    const ruleText = cr.confirmation_rule === 'close_above' ? 'above' : 'below';
    const timeframe = cr.confirmation_timeframe || '30m';
    return `${timeframe} close ${ruleText} ${fmtMoney(cr.confirmation_level)}`;
  }

  function signalRow(label, read, detail) {
    if (read == null) return '';
    const cls = read === true ? 'favorable' : read === false ? 'neutral' : read;
    const text = typeof read === 'boolean' ? (read ? 'Confirmed' : 'Not confirmed') : String(read);
    return `<div class="signal-row"><span>${escapeHtml(label)}</span><span class="read ${escapeHtml(cls)}">${escapeHtml(text)}${detail ? ` — ${escapeHtml(detail)}` : ''}</span></div>`;
  }

  // Explicit Lower-TF Trigger Capture (2026-09 session): a human-defined
  // objective 30m close condition, optional, recorded alongside a
  // review -- NOT monitoring, NOT an inferred BOS/CHoCH, NOT ENTER_NOW.
  // Only shown/relevant when lower_tf_confirmation is "not_yet"
  // (toggleTriggerFields, wired to the radio's onchange below); when
  // "yes" is picked, no future trigger is required or shown. Direction
  // only supplies a DEFAULT rule (LONG->close_above, SHORT->close_below)
  // for a fresh, never-reviewed candidate -- never enforced, and never
  // used as a default once an existing trigger is being edited (the
  // stored rule always wins there, whatever it is).
  // Also toggles the confirmation-anchor block (#confirmationFields,
  // shown under lower_tf_confirmation="yes") -- kept in the same
  // function/onchange wiring as the trigger block since the two are
  // mutually exclusive by construction (radio group), same trigger event.
  function toggleTriggerFields() {
    const selected = document.querySelector('input[name="lower_tf_confirmation"]:checked');
    const value = selected ? selected.value : null;
    const triggerFields = document.getElementById('triggerFields');
    if (triggerFields) triggerFields.style.display = value === 'not_yet' ? 'block' : 'none';
    const confirmationFields = document.getElementById('confirmationFields');
    if (confirmationFields) confirmationFields.style.display = value === 'yes' ? 'block' : 'none';
  }
  window.toggleTriggerFields = toggleTriggerFields;

  function renderCandidate(item, editing) {
    const cc = item.confluence_counts || {};
    const confluenceLine = item.confluence_available
      ? `${cc.favorable ?? 0} favorable / ${cc.unfavorable ?? 0} unfavorable / ${cc.neutral ?? 0} neutral (${item.confluence_label || 'confluence'})`
      : 'Confluence unavailable for this candidate';
    const cr = editing ? item.current_review : null;
    const storedTriggerText = editing ? triggerSummaryText(cr) : null;
    const storedConfirmationText = editing ? confirmationSummaryText(cr) : null;
    const editingBanner = editing
      ? `<div class="already-reviewed">Editing the existing review from ${escapeHtml(new Date(cr.reviewed_at).toLocaleString())}. Submitting below records a NEW review for this same setup -- the old one stays in history, this becomes current.
          <button type="button" class="cancel-edit-btn" onclick="closeEditing()">Cancel</button>
          ${storedTriggerText ? `<div class="stored-trigger">Waiting for: ${escapeHtml(storedTriggerText)}</div>` : ''}
          ${storedConfirmationText ? `<div class="stored-confirmation">Confirmed: ${escapeHtml(storedConfirmationText)}</div>` : ''}
          </div>`
      : '';
    const chartUrl = `https://www.tradingview.com/symbols/${encodeURIComponent(item.ticker)}/`;
    const wasPractical = cr && cr.review_type === 'practical_rejection';
    const wasVisual = cr && cr.review_type === 'visual';
    const submitLabel = editing ? 'Save Updated Review' : 'Submit &amp; Next';
    const practicalSubmitLabel = editing ? 'Save Updated Rejection' : 'Reject &amp; Next';

    const hasStoredTrigger = wasVisual && cr.trigger_rule != null && cr.trigger_level != null;
    const defaultTriggerRule = (item.signal || '').toLowerCase() === 'short' ? 'close_below' : 'close_above';
    const triggerRuleValue = hasStoredTrigger ? cr.trigger_rule : defaultTriggerRule;
    const triggerFieldsVisible = wasVisual && cr.lower_tf_confirmation === 'not_yet';

    // Observed Confirmation Anchor (execution_layer_v1_implementation_plan.md
    // section 8) -- same direction-default-only pattern as the trigger
    // fields above: a pre-selected rule is convenience only, never
    // enforced; the LEVEL input (not the radio) is what decides whether a
    // real confirmation anchor is being submitted (see submitReview).
    const hasStoredConfirmation = wasVisual && cr.confirmation_rule != null && cr.confirmation_level != null;
    const confirmationRuleValue = hasStoredConfirmation ? cr.confirmation_rule : defaultTriggerRule;
    const confirmationFieldsVisible = wasVisual && cr.lower_tf_confirmation === 'yes';

    return `
      <div class="candidate-card">
        <div class="candidate-head">
          <div>
            <span class="candidate-ticker">${escapeHtml(item.ticker)}</span>
            <span class="direction-pill">${escapeHtml((item.signal || '').toUpperCase())}</span>
          </div>
          <span class="candidate-rank">Rank #${item.rank} · ${escapeHtml(item.source || '')}</span>
        </div>
        ${editingBanner}

        <div class="metric-grid">
          <div class="metric"><div class="metric-label">Entry</div><div class="metric-value">${fmtMoney(item.entry_price)}</div></div>
          <div class="metric"><div class="metric-label">Stop</div><div class="metric-value">${fmtMoney(item.stop)}</div></div>
          <div class="metric"><div class="metric-label">Target</div><div class="metric-value">${fmtMoney(item.target)}</div></div>
          <div class="metric"><div class="metric-label">R:R</div><div class="metric-value">${fmtNumber(item.risk_reward)}</div></div>
          <div class="metric"><div class="metric-label">Entry distance</div><div class="metric-value">${item.entry_distance_pct != null ? item.entry_distance_pct.toFixed(2) + '%' : '--'}</div></div>
        </div>

        <div class="signal-grid">
          ${signalRow('Confluence', confluenceLine)}
          ${signalRow('BOS', item.bos_confirmed)}
          ${signalRow('Displacement', item.displacement_label, item.displacement_score != null ? fmtNumber(item.displacement_score, 1) : null)}
          ${signalRow('Location', item.location_label, item.location_alignment)}
          ${signalRow('Macro bias', item.macro_bias)}
          ${signalRow('Sweep', item.sweep_confirmed)}
          ${signalRow('Rejection', item.rejection_confirmed)}
          ${signalRow('Execution confirmation', item.execution_shadow_ok, item.execution_shadow_reason)}
        </div>

        <div class="chart-action">
          <a class="nav-link" href="${chartUrl}" target="_blank" rel="noopener">Open Chart on TradingView ↗</a>
        </div>

        <fieldset class="practical-reject">
          <legend>Practically untradeable? Skip chart review</legend>
          <div class="option-row">
            <label><input type="radio" name="practical_reason" value="options_too_expensive" ${checkedIf(wasPractical && cr.practical_rejection_reason === 'options_too_expensive')}> Options too expensive</label>
            <label><input type="radio" name="practical_reason" value="poor_option_liquidity" ${checkedIf(wasPractical && cr.practical_rejection_reason === 'poor_option_liquidity')}> Poor option liquidity/spread</label>
            <label><input type="radio" name="practical_reason" value="other" ${checkedIf(wasPractical && cr.practical_rejection_reason === 'other')}> Other practical disqualifier</label>
          </div>
          <textarea id="practicalRejectNote" placeholder="Optional note">${wasPractical && cr.note ? escapeHtml(cr.note) : ''}</textarea>
          <div class="submit-row" id="practicalRejectRow">
            <button type="button" id="practicalRejectBtn"
              onclick="submitPracticalRejection('${escapeHtml(item.ticker)}', '${escapeHtml(item.source || '')}')">${practicalSubmitLabel}</button>
          </div>
        </fieldset>

        <div class="or-divider">— or complete a full chart review —</div>

        <form id="reviewForm" onsubmit="return false;">
          <fieldset>
            <legend>Market structure</legend>
            <div class="option-row">
              <label><input type="radio" name="market_structure" value="bullish" ${checkedIf(wasVisual && cr.market_structure === 'bullish')}> Bullish</label>
              <label><input type="radio" name="market_structure" value="bearish" ${checkedIf(wasVisual && cr.market_structure === 'bearish')}> Bearish</label>
              <label><input type="radio" name="market_structure" value="range" ${checkedIf(wasVisual && cr.market_structure === 'range')}> Range</label>
            </div>
          </fieldset>
          <fieldset>
            <legend>Location</legend>
            <div class="option-row">
              <label><input type="radio" name="location_read" value="good" ${checkedIf(wasVisual && cr.location_read === 'good')}> Good</label>
              <label><input type="radio" name="location_read" value="neutral" ${checkedIf(wasVisual && cr.location_read === 'neutral')}> Neutral</label>
              <label><input type="radio" name="location_read" value="bad" ${checkedIf(wasVisual && cr.location_read === 'bad')}> Bad</label>
            </div>
          </fieldset>
          <fieldset>
            <legend>Clear path to target</legend>
            <div class="option-row">
              <label><input type="radio" name="clear_path_to_target" value="yes" ${checkedIf(wasVisual && cr.clear_path_to_target === 'yes')}> Yes</label>
              <label><input type="radio" name="clear_path_to_target" value="no" ${checkedIf(wasVisual && cr.clear_path_to_target === 'no')}> No</label>
            </div>
          </fieldset>
          <fieldset>
            <legend>Lower-timeframe confirmation</legend>
            <div class="option-row">
              <label><input type="radio" name="lower_tf_confirmation" value="yes" onchange="toggleTriggerFields()" ${checkedIf(wasVisual && cr.lower_tf_confirmation === 'yes')}> Yes</label>
              <label><input type="radio" name="lower_tf_confirmation" value="not_yet" onchange="toggleTriggerFields()" ${checkedIf(wasVisual && cr.lower_tf_confirmation === 'not_yet')}> Not yet</label>
            </div>
            <div class="trigger-fields" id="triggerFields" style="display:${triggerFieldsVisible ? 'block' : 'none'}">
              <div class="trigger-label">Waiting for 30m close:</div>
              <div class="option-row">
                <label><input type="radio" name="trigger_rule" value="close_above" ${checkedIf(triggerRuleValue === 'close_above')}> Above</label>
                <label><input type="radio" name="trigger_rule" value="close_below" ${checkedIf(triggerRuleValue === 'close_below')}> Below</label>
              </div>
              <div class="trigger-inputs">
                <input type="number" id="triggerLevel" step="0.01" min="0" placeholder="Price"
                  value="${hasStoredTrigger ? cr.trigger_level : ''}">
                <input type="text" id="triggerReason" placeholder="Optional note (why this level)"
                  value="${wasVisual && cr.trigger_reason ? escapeHtml(cr.trigger_reason) : ''}">
              </div>
              <div class="trigger-hint">Optional -- not required to submit a "Not yet" review. Kairos monitors this going forward, watching for a completed 30m close matching what you describe here.</div>
            </div>
            <div class="confirmation-fields" id="confirmationFields" style="display:${confirmationFieldsVisible ? 'block' : 'none'}">
              <div class="confirmation-label">You saw a 30m candle already close:</div>
              <div class="option-row">
                <label><input type="radio" name="confirmation_rule" value="close_above" ${checkedIf(confirmationRuleValue === 'close_above')}> Above</label>
                <label><input type="radio" name="confirmation_rule" value="close_below" ${checkedIf(confirmationRuleValue === 'close_below')}> Below</label>
              </div>
              <div class="confirmation-inputs">
                <input type="number" id="confirmationLevel" step="0.01" min="0" placeholder="Price"
                  value="${hasStoredConfirmation ? cr.confirmation_level : ''}">
                <input type="text" id="confirmedCandleTime" placeholder="When did that candle close? (optional)"
                  value="${wasVisual && cr.confirmed_candle_time ? escapeHtml(cr.confirmed_candle_time) : ''}">
                <input type="text" id="confirmationNote" placeholder="Optional note (what you saw)"
                  value="${wasVisual && cr.confirmation_note ? escapeHtml(cr.confirmation_note) : ''}">
              </div>
              <div class="confirmation-hint">Required to Approve -- describes an ALREADY-OBSERVED confirmation, not a future condition. The candle-close time is optional but helpful if you remember it.</div>
            </div>
          </fieldset>
          <fieldset>
            <legend>Decision</legend>
            <div class="option-row">
              <label><input type="radio" name="decision" value="approve" ${checkedIf(wasVisual && cr.decision === 'approve')}> Approve</label>
              <label><input type="radio" name="decision" value="watch" ${checkedIf(wasVisual && cr.decision === 'watch')}> Watch</label>
              <label><input type="radio" name="decision" value="reject" ${checkedIf(wasVisual && cr.decision === 'reject')}> Reject</label>
            </div>
          </fieldset>
          <textarea id="reviewNote" placeholder="Optional note">${wasVisual && cr.note ? escapeHtml(cr.note) : ''}</textarea>
          <div class="submit-row" id="submitRow">
            <button type="button" onclick="submitReview('${escapeHtml(item.ticker)}', '${escapeHtml(item.source || '')}')" class="primary">${submitLabel}</button>
          </div>
        </form>
      </div>`;
  }

  function renderAllReviewedPanel() {
    const counts = computeCounts(state.queue);
    return `
      <div class="summary-card">
        <h2>All candidates reviewed</h2>
        <p class="subtitle">Every candidate currently in the queue has a review on file.</p>
        <div class="summary-tally">
          <div><span class="big" style="color:var(--pass)">${counts.approve}</span>Approved</div>
          <div><span class="big" style="color:var(--warn)">${counts.watch}</span>Watch</div>
          <div><span class="big" style="color:var(--fail)">${counts.reject}</span>Rejected</div>
        </div>
        <div class="summary-actions">
          <a class="nav-link" href="/approved-setups">View Approved Setups</a>
          <a class="nav-link" href="/watch-setups">View Watch Setups</a>
          <button type="button" onclick="setFilter('reviewed')">Browse Reviewed</button>
          <button type="button" onclick="loadQueue()">Reload Queue</button>
        </div>
      </div>`;
  }

  function render() {
    const main = document.getElementById('mainContent');
    if (!state.loaded) return;
    if (!state.queue.length) {
      main.innerHTML = `<div class="disclaimer">${escapeHtml(state.disclaimer)}</div><div class="status-line">No candidates in the review queue right now.</div>`;
      return;
    }

    const header = `<div class="disclaimer">${escapeHtml(state.disclaimer)}</div>${progressHeader()}`;

    if (state.filter === 'reviewed') {
      if (state.editing && state.queue[state.index] && isReviewed(state.queue[state.index])) {
        main.innerHTML = `${header}${renderCandidate(state.queue[state.index], true)}`;
      } else {
        main.innerHTML = `${header}${renderReviewedList()}`;
      }
      return;
    }

    // filter === 'unreviewed'
    const idx = firstUnreviewedIndex(state.queue);
    if (idx < 0) {
      main.innerHTML = `${header}${renderAllReviewedPanel()}`;
      return;
    }
    state.index = idx;
    main.innerHTML = `${header}${renderCandidate(state.queue[idx], false)}`;
  }

  // Applies a fresh CandidateVisualReviewOut response to the in-memory
  // queue entry it belongs to (by ticker+source, unique within one
  // queue response) -- this is what makes counters/progress update
  // immediately without a full reload, from the SAME response the POST
  // already returns.
  function applyReviewResult(ticker, source, reviewResult) {
    const idx = state.queue.findIndex((item) => item.ticker === ticker && (item.source || '') === (source || ''));
    if (idx >= 0) {
      state.queue[idx].current_review = reviewResult;
    }
    return idx;
  }

  function afterSubmit() {
    if (state.editing) {
      state.editing = false;
      // stay on the reviewed list, now showing the updated decision
      render();
      return;
    }
    // Normal flow: advance to whatever is now the first unreviewed
    // candidate (search, not a blind +1 -- correct regardless of where in
    // the ranked list this candidate was).
    render();
  }

  async function submitPracticalRejection(ticker, source) {
    const selected = document.querySelector('input[name="practical_reason"]:checked');
    if (!selected) {
      alert('Select a reason before rejecting.');
      return;
    }
    const row = document.getElementById('practicalRejectRow');
    row.innerHTML = '<button type="button" disabled>Saving…</button>';
    try {
      const result = await fetchJson(`${API_BASE}/candidates/${encodeURIComponent(ticker)}/visual-review`, {
        method: 'POST',
        body: JSON.stringify({
          source,
          decision: 'reject',
          practical_rejection_reason: selected.value,
          // Deliberately no market_structure/location_read/clear_path_to_target/
          // lower_tf_confirmation here at all -- the backend rejects the
          // combination outright if any is present, and this path never
          // touches the visual-review form's fields to begin with.
          note: document.getElementById('practicalRejectNote').value || null,
        }),
      });
      applyReviewResult(ticker, source, result);
      afterSubmit();
    } catch (err) {
      alert(`Could not save rejection: ${err.message}`);
      row.innerHTML = '<button type="button" id="practicalRejectBtn" onclick="submitPracticalRejection(\'' +
        ticker.replace(/'/g, "\\'") + '\', \'' + (source || '').replace(/'/g, "\\'") + '\')">Reject &amp; Next</button>';
    }
  }
  window.submitPracticalRejection = submitPracticalRejection;

  async function submitReview(ticker, source) {
    const form = document.getElementById('reviewForm');
    const data = new FormData(form);
    const decision = data.get('decision');
    const required = ['market_structure', 'location_read', 'clear_path_to_target', 'lower_tf_confirmation', 'decision'];
    for (const field of required) {
      if (!data.get(field)) {
        alert('Please answer every field before submitting.');
        return;
      }
    }
    // Required-when-approving (execution_layer_v1_implementation_plan.md
    // section 2 rule D): the server enforces this regardless -- this is
    // UX only, so a reviewer gets a clear, immediate message instead of a
    // round-trip 422.
    if (decision === 'approve' && data.get('lower_tf_confirmation') === 'yes') {
      const confirmationLevelInput = document.getElementById('confirmationLevel');
      const confirmationLevelRaw = confirmationLevelInput ? confirmationLevelInput.value.trim() : '';
      if (confirmationLevelRaw === '') {
        alert('Approving a "Yes" (already confirmed) setup requires describing the completed candle you observed -- fill in the confirmation price below.');
        return;
      }
    }
    const submitRow = document.getElementById('submitRow');
    submitRow.innerHTML = '<button type="button" disabled>Saving…</button>';
    try {
      const body = {
        source,
        market_structure: data.get('market_structure'),
        location_read: data.get('location_read'),
        clear_path_to_target: data.get('clear_path_to_target'),
        lower_tf_confirmation: data.get('lower_tf_confirmation'),
        decision,
        note: document.getElementById('reviewNote').value || null,
      };
      // Trigger fields (Explicit Lower-TF Trigger Capture, 2026-09
      // session): the trigger_rule radio always has a pre-selected
      // default (direction-based, or the stored value when editing) even
      // when the reviewer never touches the trigger section at all -- so
      // the LEVEL field, not the radio, is what decides whether a real
      // trigger is being submitted. An empty level means "no trigger,"
      // full stop, regardless of which rule radio happens to be checked.
      const triggerLevelInput = document.getElementById('triggerLevel');
      const triggerLevelRaw = triggerLevelInput ? triggerLevelInput.value.trim() : '';
      if (triggerLevelRaw !== '') {
        body.trigger_timeframe = '30m';
        body.trigger_rule = data.get('trigger_rule');
        body.trigger_level = parseFloat(triggerLevelRaw);
        const triggerReasonInput = document.getElementById('triggerReason');
        const triggerReasonRaw = triggerReasonInput ? triggerReasonInput.value.trim() : '';
        if (triggerReasonRaw) body.trigger_reason = triggerReasonRaw;
      }
      // Confirmation fields (Observed Confirmation Anchor, Execution
      // Layer V1 session): same "the LEVEL field decides, not the radio"
      // rule as trigger fields above.
      const confirmationLevelInput = document.getElementById('confirmationLevel');
      const confirmationLevelRaw = confirmationLevelInput ? confirmationLevelInput.value.trim() : '';
      if (confirmationLevelRaw !== '') {
        body.confirmation_timeframe = '30m';
        body.confirmation_rule = data.get('confirmation_rule');
        body.confirmation_level = parseFloat(confirmationLevelRaw);
        const confirmedCandleTimeInput = document.getElementById('confirmedCandleTime');
        const confirmedCandleTimeRaw = confirmedCandleTimeInput ? confirmedCandleTimeInput.value.trim() : '';
        if (confirmedCandleTimeRaw) body.confirmed_candle_time = confirmedCandleTimeRaw;
        const confirmationNoteInput = document.getElementById('confirmationNote');
        const confirmationNoteRaw = confirmationNoteInput ? confirmationNoteInput.value.trim() : '';
        if (confirmationNoteRaw) body.confirmation_note = confirmationNoteRaw;
      }
      const result = await fetchJson(`${API_BASE}/candidates/${encodeURIComponent(ticker)}/visual-review`, {
        method: 'POST',
        body: JSON.stringify(body),
      });
      applyReviewResult(ticker, source, result);
      afterSubmit();
    } catch (err) {
      alert(`Could not save review: ${err.message}`);
      submitRow.innerHTML = '<button type="button" onclick="submitReview(\'' + ticker.replace(/'/g, "\\'") + '\', \'' + (source || '').replace(/'/g, "\\'") + '\')" class="primary">Submit &amp; Next</button>';
    }
  }
  window.submitReview = submitReview;
  window.loadQueue = loadQueue;

  document.addEventListener('DOMContentLoaded', () => {
    if (apiKey()) {
      document.getElementById('apiBand').classList.add('hidden');
    }
    loadQueue();
  });

  // Exposed for tests (tests/review_queue_auth_v1.js,
  // tests/review_queue_persistence_v1.js) -- real browser usage never
  // touches this return value, it only matters under module.exports.
  return {
    state,
    loadQueue,
    renderAuthRequiredPanel,
    submitApiKey,
    submitReview,
    submitPracticalRejection,
    setFilter,
    openReviewedItem,
    closeEditing,
    isReviewed,
    decisionFor,
    computeCounts,
    firstUnreviewedIndex,
    triggerSummaryText,
    confirmationSummaryText,
  };
});
