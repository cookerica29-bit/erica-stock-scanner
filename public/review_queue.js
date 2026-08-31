// Stage C: Kairos Review Queue frontend. Deliberately a standalone page,
// not a filter/tab bolted onto candidate_dashboard.js's card wall -- see
// the 2026-08-31 session's design pass for why. Talks only to the two
// Stage A/B endpoints (GET /candidates/review-queue, POST
// /candidates/{ticker}/visual-review) plus the existing session-cookie
// auth flow (POST /session) already used by candidates.html -- same-origin,
// so a session established there is reused here automatically.
(function () {
  const KEY = 'kairos_scanner_api_key';
  const API_BASE = '/api/v1/scanner';

  const state = {
    queue: [],
    index: 0,
    disclaimer: '',
    tally: { approve: 0, watch: 0, reject: 0 },
    loaded: false,
  };

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

  async function loadQueue() {
    setStatus('Loading review queue…');
    try {
      const result = await fetchJson(`${API_BASE}/candidates/review-queue`);
      state.queue = result.candidates || [];
      state.disclaimer = result.disclaimer || '';
      state.index = 0;
      state.tally = { approve: 0, watch: 0, reject: 0 };
      state.loaded = true;
      render();
    } catch (err) {
      if (err.status === 401) {
        document.getElementById('apiBand').classList.remove('hidden');
        setStatus('Enter your scanner API key above to load the review queue.');
      } else {
        setStatus(`Could not load review queue: ${err.message}`, true);
      }
    }
  }

  function tallyChips() {
    return `
      <div class="tally">
        <span class="tally-chip approve">Approved ${state.tally.approve}</span>
        <span class="tally-chip watch">Watch ${state.tally.watch}</span>
        <span class="tally-chip reject">Rejected ${state.tally.reject}</span>
      </div>`;
  }

  function signalRow(label, read, detail) {
    if (read == null) return '';
    const cls = read === true ? 'favorable' : read === false ? 'neutral' : read;
    const text = typeof read === 'boolean' ? (read ? 'Confirmed' : 'Not confirmed') : String(read);
    return `<div class="signal-row"><span>${escapeHtml(label)}</span><span class="read ${escapeHtml(cls)}">${escapeHtml(text)}${detail ? ` — ${escapeHtml(detail)}` : ''}</span></div>`;
  }

  function renderCandidate(item) {
    const cc = item.confluence_counts || {};
    const confluenceLine = item.confluence_available
      ? `${cc.favorable ?? 0} favorable / ${cc.unfavorable ?? 0} unfavorable / ${cc.neutral ?? 0} neutral (${item.confluence_label || 'confluence'})`
      : 'Confluence unavailable for this candidate';
    const current = item.current_review;
    const alreadyReviewed = current
      ? `<div class="already-reviewed">Last reviewed ${escapeHtml(new Date(current.reviewed_at).toLocaleString())} — decision: <strong>${escapeHtml(current.decision)}</strong>${current.note ? ` — "${escapeHtml(current.note)}"` : ''}. Submitting below records a new review for this same setup.</div>`
      : '';
    const chartUrl = `https://www.tradingview.com/symbols/${encodeURIComponent(item.ticker)}/`;

    return `
      <div class="candidate-card">
        <div class="candidate-head">
          <div>
            <span class="candidate-ticker">${escapeHtml(item.ticker)}</span>
            <span class="direction-pill">${escapeHtml((item.signal || '').toUpperCase())}</span>
          </div>
          <span class="candidate-rank">Rank #${item.rank} · ${escapeHtml(item.source || '')}</span>
        </div>
        ${alreadyReviewed}

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

        <form id="reviewForm" onsubmit="return false;">
          <fieldset>
            <legend>Market structure</legend>
            <div class="option-row">
              <label><input type="radio" name="market_structure" value="bullish" required> Bullish</label>
              <label><input type="radio" name="market_structure" value="bearish"> Bearish</label>
              <label><input type="radio" name="market_structure" value="range"> Range</label>
            </div>
          </fieldset>
          <fieldset>
            <legend>Location</legend>
            <div class="option-row">
              <label><input type="radio" name="location_read" value="good" required> Good</label>
              <label><input type="radio" name="location_read" value="neutral"> Neutral</label>
              <label><input type="radio" name="location_read" value="bad"> Bad</label>
            </div>
          </fieldset>
          <fieldset>
            <legend>Clear path to target</legend>
            <div class="option-row">
              <label><input type="radio" name="clear_path_to_target" value="yes" required> Yes</label>
              <label><input type="radio" name="clear_path_to_target" value="no"> No</label>
            </div>
          </fieldset>
          <fieldset>
            <legend>Lower-timeframe confirmation</legend>
            <div class="option-row">
              <label><input type="radio" name="lower_tf_confirmation" value="yes" required> Yes</label>
              <label><input type="radio" name="lower_tf_confirmation" value="not_yet"> Not yet</label>
            </div>
          </fieldset>
          <fieldset>
            <legend>Decision</legend>
            <div class="option-row">
              <label><input type="radio" name="decision" value="approve" required> Approve</label>
              <label><input type="radio" name="decision" value="watch"> Watch</label>
              <label><input type="radio" name="decision" value="reject"> Reject</label>
            </div>
          </fieldset>
          <textarea id="reviewNote" placeholder="Optional note"></textarea>
          <div class="submit-row" id="submitRow">
            <button type="button" onclick="submitReview('${escapeHtml(item.ticker)}', '${escapeHtml(item.source || '')}')" class="primary">Submit &amp; Next</button>
          </div>
        </form>
      </div>`;
  }

  function renderSummary() {
    const total = state.tally.approve + state.tally.watch + state.tally.reject;
    return `
      <div class="summary-card">
        <h2>Queue complete</h2>
        <p class="subtitle">Reviewed ${total} of ${state.queue.length} candidates this pass.</p>
        <div class="summary-tally">
          <div><span class="big" style="color:var(--pass)">${state.tally.approve}</span>Approved</div>
          <div><span class="big" style="color:var(--warn)">${state.tally.watch}</span>Watch</div>
          <div><span class="big" style="color:var(--fail)">${state.tally.reject}</span>Rejected</div>
          <div><span class="big">${state.queue.length - total}</span>Remaining</div>
        </div>
        <button type="button" onclick="loadQueue()">Reload Queue</button>
      </div>`;
  }

  function render() {
    const main = document.getElementById('mainContent');
    if (!state.loaded) return;
    if (!state.queue.length) {
      main.innerHTML = `<div class="disclaimer">${escapeHtml(state.disclaimer)}</div><div class="status-line">No candidates in the review queue right now.</div>`;
      return;
    }
    if (state.index >= state.queue.length) {
      main.innerHTML = renderSummary();
      return;
    }
    const item = state.queue[state.index];
    main.innerHTML = `
      <div class="disclaimer">${escapeHtml(state.disclaimer)}</div>
      <div class="progress-row">
        <span class="progress-count">Candidate ${state.index + 1} of ${state.queue.length}</span>
        ${tallyChips()}
      </div>
      ${renderCandidate(item)}`;
  }

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
    const submitRow = document.getElementById('submitRow');
    submitRow.innerHTML = '<button type="button" disabled>Submitting…</button>';
    try {
      await fetchJson(`${API_BASE}/candidates/${encodeURIComponent(ticker)}/visual-review`, {
        method: 'POST',
        body: JSON.stringify({
          source,
          market_structure: data.get('market_structure'),
          location_read: data.get('location_read'),
          clear_path_to_target: data.get('clear_path_to_target'),
          lower_tf_confirmation: data.get('lower_tf_confirmation'),
          decision,
          note: document.getElementById('reviewNote').value || null,
        }),
      });
      state.tally[decision] = (state.tally[decision] || 0) + 1;
      state.index += 1;
      render();
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
})();
