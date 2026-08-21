(function () {
  const KEY = 'kairos_scanner_api_key';
  const state = {
    loaded: false,
    loading: false,
    view: 'new',
    candidates: [],
    promotions: [],
    planPreviews: [],
    chartReviews: [],
    error: null,
  };

  function escapeHtml(value) {
    return String(value ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function fmtMoney(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '—';
    return `$${n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  }

  function fmtNumber(value, digits = 2) {
    const n = Number(value);
    if (!Number.isFinite(n)) return '—';
    return n.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
  }

  function fmtTime(value) {
    if (!value) return '—';
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) return String(value);
    return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
  }

  function apiKey() {
    return localStorage.getItem(KEY) || '';
  }

  function headers() {
    return { 'Content-Type': 'application/json', 'X-API-Key': apiKey() };
  }

  async function fetchJson(url, options = {}) {
    const response = await fetch(url, { ...options, headers: { ...headers(), ...(options.headers || {}) } });
    const text = await response.text();
    let payload = null;
    try { payload = text ? JSON.parse(text) : null; } catch { payload = { detail: text }; }
    if (!response.ok) {
      const message = payload?.detail || `Request failed (${response.status})`;
      throw new Error(Array.isArray(message) ? message.map(item => item.msg || item.reason || String(item)).join(', ') : String(message));
    }
    return payload;
  }

  function promotionKey(item) {
    return `${String(item.ticker || '').toUpperCase()}|${item.source || ''}`;
  }

  function promotionsByKey() {
    const map = new Map();
    state.promotions.forEach(item => map.set(promotionKey(item), item));
    return map;
  }

  function chartReviewsByKey() {
    const map = new Map();
    state.chartReviews.forEach(item => map.set(promotionKey(item), item));
    return map;
  }

  function planPreviewsByKey() {
    const map = new Map();
    state.planPreviews.forEach(item => map.set(promotionKey(item), item));
    return map;
  }

  function setStatus(message, kind = '') {
    const el = document.getElementById('candidateStatus');
    if (!el) return;
    el.textContent = message;
    el.style.color = kind === 'error' ? 'var(--fail)' : kind === 'ok' ? 'var(--pass)' : 'var(--muted)';
  }

  function saveCandidateApiKey() {
    const input = document.getElementById('candidateApiKeyInput');
    const value = String(input?.value || '').trim();
    if (value) localStorage.setItem(KEY, value);
    else localStorage.removeItem(KEY);
    state.loaded = false;
    setStatus(value ? 'Scanner API key saved in this browser.' : 'Scanner API key cleared.', value ? 'ok' : '');
  }

  function syncInput() {
    const input = document.getElementById('candidateApiKeyInput');
    if (input && !input.value) input.value = apiKey();
  }

  async function loadCandidateDashboard(force = false) {
    syncInput();
    if (!apiKey()) {
      render();
      setStatus('Enter the scanner API key, then refresh.');
      return;
    }
    if (state.loading || (state.loaded && !force)) {
      render();
      return;
    }
    state.loading = true;
    state.error = null;
    setStatus('Loading candidate inbox...');
    render();
    try {
      const [candidates, promotions, planPreviews, chartReviews] = await Promise.all([
        fetchJson('/api/v1/scanner/candidates'),
        fetchJson('/api/v1/scanner/candidate-promotions'),
        fetchJson('/api/v1/scanner/candidate-plan-previews'),
        fetchJson('/api/v1/scanner/candidate-chart-reviews'),
      ]);
      state.candidates = Array.isArray(candidates) ? candidates : [];
      state.promotions = Array.isArray(promotions) ? promotions : [];
      state.planPreviews = Array.isArray(planPreviews) ? planPreviews : [];
      state.chartReviews = Array.isArray(chartReviews) ? chartReviews : [];
      state.loaded = true;
      setStatus(`Loaded ${state.candidates.length} candidates, ${state.promotions.length} active plans, ${state.planPreviews.length} plan previews, and ${state.chartReviews.length} AI chart notes.`, 'ok');
    } catch (error) {
      state.error = error.message || String(error);
      setStatus(state.error, 'error');
    } finally {
      state.loading = false;
      render();
    }
  }

  function setCandidateView(view) {
    state.view = view || 'new';
    document.querySelectorAll('.candidate-tab[data-candidate-view]').forEach(button => {
      button.classList.toggle('active', button.dataset.candidateView === state.view);
    });
    render();
  }

  function candidatesForView() {
    if (state.view === 'all') return state.candidates;
    return state.candidates.filter(item => String(item.status || 'new').toLowerCase() === state.view);
  }

  function renderStats() {
    const el = document.getElementById('candidateStats');
    if (!el) return;
    const counts = state.candidates.reduce((acc, item) => {
      const status = String(item.status || 'new').toLowerCase();
      acc[status] = (acc[status] || 0) + 1;
      return acc;
    }, {});
    const newest = state.candidates
      .map(item => item.scanned_at || item.updated_at)
      .filter(Boolean)
      .sort()
      .at(-1);
    const stats = [
      ['New', counts.new || 0],
      ['Active', counts.active || 0],
      ['Dismissed', counts.dismissed || 0],
      ['Last Scan', newest ? fmtTime(newest) : '—'],
    ];
    el.innerHTML = stats.map(([label, value]) => `
      <div class="candidate-stat">
        <div class="candidate-stat-label">${escapeHtml(label)}</div>
        <div class="candidate-stat-value">${escapeHtml(value)}</div>
      </div>
    `).join('');
  }

  function render() {
    renderStats();
    const list = document.getElementById('candidateList');
    if (!list) return;
    if (!apiKey()) {
      list.innerHTML = '<div class="candidate-empty">Paste the scanner API key to review candidates pushed by the local scanner.</div>';
      return;
    }
    if (state.loading) {
      list.innerHTML = '<div class="candidate-empty">Loading candidates...</div>';
      return;
    }
    if (state.error) {
      list.innerHTML = `<div class="candidate-empty">${escapeHtml(state.error)}</div>`;
      return;
    }
    const items = candidatesForView();
    if (!items.length) {
      list.innerHTML = '<div class="candidate-empty">No candidates in this view.</div>';
      return;
    }
    const promoMap = promotionsByKey();
    const previewMap = planPreviewsByKey();
    const reviewMap = chartReviewsByKey();
    list.innerHTML = items.map(item => renderCandidateCard(item, promoMap.get(promotionKey(item)), previewMap.get(promotionKey(item)), reviewMap.get(promotionKey(item)))).join('');
  }

  function renderCandidateCard(item, promotion, planPreview, chartReview) {
    const status = String(item.status || 'new').toLowerCase();
    const direction = String(item.signal || '').toLowerCase();
    const confidence = String(item.confidence || 'unknown').toLowerCase();
    const regime = String(item.daily_regime || 'unknown').toLowerCase();
    const aligned = regime && direction && regime.includes(direction);
    return `
      <article class="candidate-card ${status === 'active' ? 'active-plan' : ''} ${status === 'dismissed' ? 'dismissed' : ''}">
        <div class="candidate-card-head">
          <div>
            <div class="candidate-ticker">${escapeHtml(item.ticker)}</div>
            <div class="candidate-meta">${escapeHtml(item.source || 'source unknown')} · scanned ${escapeHtml(fmtTime(item.scanned_at))}</div>
          </div>
          <span class="candidate-pill ${escapeHtml(direction)}">${escapeHtml(direction || '—')}</span>
        </div>
        <div class="candidate-pill-row">
          <span class="candidate-pill ${escapeHtml(confidence)}">${escapeHtml(confidence)}</span>
          <span class="candidate-pill ${aligned ? 'high' : 'medium'}">${aligned ? 'regime aligned' : `regime ${regime || 'unknown'}`}</span>
          <span class="candidate-pill">${escapeHtml(status)}</span>
        </div>
        <div class="candidate-kv">
          <div><span>Entry</span><strong>${escapeHtml(fmtMoney(item.entry_price))}</strong></div>
          <div><span>4H EMA21</span><strong>${escapeHtml(fmtMoney(item.ema21_4h))}</strong></div>
          <div><span>SMA50 Daily</span><strong>${escapeHtml(fmtMoney(item.sma50_daily))}</strong></div>
          <div><span>SMA200 Daily</span><strong>${escapeHtml(fmtMoney(item.sma200_daily))}</strong></div>
        </div>
        ${planPreview ? renderPlanPreview(planPreview, Boolean(promotion)) : ''}
        ${promotion ? renderPromotion(promotion) : ''}
        ${chartReview ? renderChartReview(chartReview) : ''}
        ${renderActions(item)}
      </article>
    `;
  }

  function renderPromotion(promotion) {
    const warnings = [];
    if (promotion.rr_warning) warnings.push('<span class="candidate-pill warn">R:R warning</span>');
    if (promotion.no_valid_target) warnings.push('<span class="candidate-pill bad">No valid target</span>');
    if (promotion.position_size == null) warnings.push('<span class="candidate-pill">Options sizing pending</span>');
    return `
      <div class="candidate-pill-row">${warnings.join('')}</div>
      <div class="candidate-kv">
        <div><span>Stop</span><strong>${escapeHtml(fmtMoney(promotion.stop))}</strong></div>
        <div><span>Target</span><strong>${escapeHtml(fmtMoney(promotion.target))}</strong></div>
        <div><span>R:R</span><strong>${escapeHtml(promotion.risk_reward == null ? '—' : fmtNumber(promotion.risk_reward, 2))}</strong></div>
        <div><span>ATR14</span><strong>${escapeHtml(fmtNumber(promotion.atr14, 2))}</strong></div>
      </div>
      <div class="candidate-meta">Promoted ${escapeHtml(fmtTime(promotion.promoted_at))} · target source ${escapeHtml(promotion.target_source || '—')}</div>
    `;
  }

  function renderPlanPreview(preview, hasPromotion) {
    const warnings = [];
    if (preview.preview_error) warnings.push(`<span class="candidate-pill bad">${escapeHtml(preview.preview_error)}</span>`);
    if (preview.rr_warning) warnings.push('<span class="candidate-pill warn">R:R warning</span>');
    if (preview.no_valid_target) warnings.push('<span class="candidate-pill bad">No valid target</span>');
    const contract = preview.option_contract || {};
    const contractAvailable = Boolean(contract.available);
    const contractLabel = contractAvailable
      ? `${fmtMoney(contract.strike)} ${String(contract.type || '').toUpperCase()}`
      : (contract.execution || 'No Clean Contract');
    const contractMeta = contractAvailable
      ? `${contract.expiry || contract.expiration || '—'} · ${contract.dte ?? '—'} DTE · ${contract.execution || '—'}`
      : (contract.reason || 'Contract unavailable');
    return `
      <div class="candidate-pill-row">
        <span class="candidate-pill">${hasPromotion ? 'Current Plan Math' : 'Plan Preview'}</span>
        ${warnings.join('')}
      </div>
      <div class="candidate-kv">
        <div><span>Stop</span><strong>${escapeHtml(fmtMoney(preview.stop))}</strong></div>
        <div><span>Target</span><strong>${escapeHtml(fmtMoney(preview.target))}</strong></div>
        <div><span>R:R</span><strong>${escapeHtml(preview.risk_reward == null ? '—' : fmtNumber(preview.risk_reward, 2))}</strong></div>
        <div><span>ATR14</span><strong>${escapeHtml(fmtNumber(preview.atr14, 2))}</strong></div>
        <div><span>Contract</span><strong>${escapeHtml(contractLabel)}</strong></div>
        <div><span>Expiry</span><strong>${escapeHtml(contractMeta)}</strong></div>
      </div>
      <div class="candidate-meta">Preview computed ${escapeHtml(fmtTime(preview.computed_at))} · target source ${escapeHtml(preview.target_source || '—')}</div>
    `;
  }

  function classificationLabel(value) {
    return String(value || 'mixed_unclear').replace(/_/g, ' ');
  }

  function renderChartReview(review) {
    return `
      <div class="candidate-ai-note">
        <div class="candidate-pill-row">
          <span class="candidate-pill">AI Chart Note</span>
          <span class="candidate-pill medium">${escapeHtml(classificationLabel(review.classification))}</span>
        </div>
        <div class="candidate-meta">${escapeHtml(review.caveat || 'Informational only, not a recommendation.')}</div>
        <div class="candidate-ai-rationale">${escapeHtml(review.rationale)}</div>
        <div class="candidate-meta">Reviewed ${escapeHtml(fmtTime(review.reviewed_at))} · ${escapeHtml(review.data_source || 'price data')}</div>
      </div>
    `;
  }

  function renderActions(item) {
    const status = String(item.status || 'new').toLowerCase();
    const source = encodeURIComponent(item.source || '');
    const ticker = encodeURIComponent(item.ticker || '');
    const promote = status !== 'active'
      ? `<button onclick="updateCandidateStatus('${ticker}','${source}','active','${status}')">Promote</button>`
      : '';
    const chartRead = `<button class="btn-ghost" onclick="requestChartReview('${ticker}','${source}')">Get AI Chart Read</button>`;
    const dismiss = status !== 'dismissed' && status !== 'active'
      ? `<button class="btn-secondary" onclick="updateCandidateStatus('${ticker}','${source}','dismissed','${status}')">Dismiss</button>`
      : '';
    const restore = status !== 'new'
      ? `<button class="btn-ghost" onclick="updateCandidateStatus('${ticker}','${source}','new','${status}')">Back to Inbox</button>`
      : '';
    return `<div class="candidate-actions">${promote}${chartRead}${dismiss}${restore}</div>`;
  }

  async function updateCandidateStatus(ticker, source, status, currentStatus = '') {
    const decodedTicker = decodeURIComponent(ticker);
    const decodedSource = decodeURIComponent(source);
    if (status === 'dismissed') {
      const ok = window.confirm(`Dismiss ${decodedTicker} from the candidate inbox?`);
      if (!ok) return;
    }
    if (String(currentStatus).toLowerCase() === 'active' && status === 'new') {
      const ok = window.confirm(`Move active plan ${decodedTicker} back to the inbox? The saved promotion math will remain available.`);
      if (!ok) return;
    }
    setStatus(`${status === 'active' ? 'Promoting' : 'Updating'} ${decodedTicker}...`);
    try {
      await fetchJson(`/api/v1/scanner/candidates/${encodeURIComponent(decodedTicker)}?source=${encodeURIComponent(decodedSource)}`, {
        method: 'PATCH',
        body: JSON.stringify({ status }),
      });
      state.loaded = false;
      await loadCandidateDashboard(true);
    } catch (error) {
      setStatus(error.message || String(error), 'error');
    }
  }

  async function requestChartReview(ticker, source) {
    const decodedTicker = decodeURIComponent(ticker);
    const decodedSource = decodeURIComponent(source);
    setStatus(`Requesting informational AI chart note for ${decodedTicker}...`);
    try {
      await fetchJson(`/api/v1/scanner/candidates/${encodeURIComponent(decodedTicker)}/ai-chart-review?source=${encodeURIComponent(decodedSource)}`, {
        method: 'POST',
      });
      state.loaded = false;
      await loadCandidateDashboard(true);
    } catch (error) {
      setStatus(error.message || String(error), 'error');
    }
  }

  window.saveCandidateApiKey = saveCandidateApiKey;
  window.loadCandidateDashboard = loadCandidateDashboard;
  window.setCandidateView = setCandidateView;
  window.updateCandidateStatus = updateCandidateStatus;
  window.requestChartReview = requestChartReview;
})();
