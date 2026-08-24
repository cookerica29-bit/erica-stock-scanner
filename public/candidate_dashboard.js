(function () {
  const KEY = 'kairos_scanner_api_key';
  const ALERT_KEY = 'kairos_candidate_alerts';
  const API_KEY_DB = 'kairos_candidate_dashboard';
  const API_KEY_STORE = 'settings';
  const ENTER_NOW_MIN_RR = 1.5;
  const ENTER_NOW_MAX_SCAN_AGE_MS = 5 * 60 * 60 * 1000;
  const ACCEPTABLE_CONTRACT_GRADES = ['excellent', 'good', 'fair'];
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
  let knownCandidateIds = new Set();
  let candidateAlertsEnabled = storageGet(localStorage, ALERT_KEY) === 'on';
  let candidateAlertsInitialized = false;
  let apiKeyPanelExpanded = false;
  let cachedApiKey = '';
  let candidateSessionAuthenticated = false;

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

  function storageGet(storage, key) {
    try {
      return storage.getItem(key) || '';
    } catch {
      return '';
    }
  }

  function storageSet(storage, key, value) {
    try {
      storage.setItem(key, value);
    } catch {}
  }

  function storageRemove(storage, key) {
    try {
      storage.removeItem(key);
    } catch {}
  }

  function openApiKeyDb() {
    if (!('indexedDB' in window)) return Promise.resolve(null);
    return new Promise(resolve => {
      const request = indexedDB.open(API_KEY_DB, 1);
      request.onupgradeneeded = () => {
        request.result.createObjectStore(API_KEY_STORE);
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => resolve(null);
      request.onblocked = () => resolve(null);
    });
  }

  async function readApiKeyFromDb() {
    const db = await openApiKeyDb();
    if (!db) return '';
    return new Promise(resolve => {
      const tx = db.transaction(API_KEY_STORE, 'readonly');
      const request = tx.objectStore(API_KEY_STORE).get(KEY);
      request.onsuccess = () => resolve(String(request.result || ''));
      request.onerror = () => resolve('');
      tx.oncomplete = () => db.close();
      tx.onerror = () => db.close();
    });
  }

  async function writeApiKeyToDb(value) {
    const db = await openApiKeyDb();
    if (!db) return;
    await new Promise(resolve => {
      const tx = db.transaction(API_KEY_STORE, 'readwrite');
      const store = tx.objectStore(API_KEY_STORE);
      if (value) store.put(value, KEY);
      else store.delete(KEY);
      tx.oncomplete = () => {
        db.close();
        resolve();
      };
      tx.onerror = () => {
        db.close();
        resolve();
      };
    });
  }

  function persistApiKey(value) {
    cachedApiKey = value;
    storageSet(localStorage, KEY, value);
    storageSet(sessionStorage, KEY, value);
  }

  async function hydrateApiKey() {
    const existing = apiKey();
    if (existing) return existing;
    const stored = await readApiKeyFromDb();
    if (stored) persistApiKey(stored);
    return stored;
  }

  function clearApiKey() {
    cachedApiKey = '';
    storageRemove(localStorage, KEY);
    storageRemove(sessionStorage, KEY);
  }

  function apiKey() {
    cachedApiKey = cachedApiKey || storageGet(localStorage, KEY) || storageGet(sessionStorage, KEY);
    return cachedApiKey;
  }

  function headers() {
    const key = apiKey();
    return {
      'Content-Type': 'application/json',
      ...(key ? { 'X-API-Key': key } : {}),
    };
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

  function effectivePlanForCandidate(item, promoMap, previewMap) {
    const key = promotionKey(item);
    const promotion = promoMap.get(key);
    const preview = previewMap.get(key);
    if (promotion) {
      return {
        ...preview,
        ...promotion,
        option_contract: preview?.option_contract || promotion.option_contract || null,
      };
    }
    return preview || null;
  }

  function isRegimeAligned(item) {
    const direction = String(item.signal || '').toLowerCase();
    const regime = String(item.daily_regime || '').toLowerCase();
    if (direction === 'long') return regime.includes('long') || regime.includes('bull');
    if (direction === 'short') return regime.includes('short') || regime.includes('bear');
    return false;
  }

  function contractBlockReason(plan) {
    const contract = plan?.option_contract || {};
    if (!contract.available) return contract.reason || contract.execution || 'No clean options contract';
    const execution = String(contract.execution || '').toLowerCase();
    const acceptable = ACCEPTABLE_CONTRACT_GRADES.some(grade => execution.includes(grade));
    return acceptable ? '' : `Contract quality is ${contract.execution || 'unknown'}`;
  }

  function scanFreshnessBlockReason(item) {
    const rawTime = item?.scanned_at || item?.updated_at;
    if (!rawTime) return 'Scan time unavailable';
    const scannedAt = new Date(rawTime);
    if (Number.isNaN(scannedAt.getTime())) return 'Scan time unavailable';
    if (Date.now() - scannedAt.getTime() > ENTER_NOW_MAX_SCAN_AGE_MS) {
      return 'Scan is stale; refresh before entry';
    }
    return '';
  }

  function routeBlockReason(item, plan) {
    const direction = String(item.signal || '').toLowerCase();
    if (direction === 'short') return 'Shorts are research-only';
    if (direction !== 'long') return 'Unsupported direction';
    if (!isRegimeAligned(item)) return 'Regime is not aligned';
    if (!plan) return 'Plan math pending';
    if (plan.preview_error) return plan.preview_error;
    if (plan.no_valid_target || plan.target == null || plan.risk_reward == null) return 'No valid target';
    if (plan.rr_warning || Number(plan.risk_reward) < ENTER_NOW_MIN_RR) return `R:R is below ${ENTER_NOW_MIN_RR}:1`;
    const contractReason = contractBlockReason(plan);
    if (contractReason) return contractReason;
    const freshnessReason = scanFreshnessBlockReason(item);
    if (freshnessReason) return freshnessReason;
    return '';
  }

  function isEnterNowCandidate(item, promoMap, previewMap) {
    return !routeBlockReason(item, effectivePlanForCandidate(item, promoMap, previewMap));
  }

  function enterNowCandidates() {
    const promoMap = promotionsByKey();
    const previewMap = planPreviewsByKey();
    return state.candidates.filter(item => isEnterNowCandidate(item, promoMap, previewMap));
  }

  function setStatus(message, kind = '') {
    const el = document.getElementById('candidateStatus');
    if (!el) return;
    el.textContent = message;
    el.style.color = kind === 'error' ? 'var(--fail)' : kind === 'ok' ? 'var(--pass)' : 'var(--muted)';
  }

  async function saveCandidateApiKey() {
    const input = document.getElementById('candidateApiKeyInput');
    const value = String(input?.value || '').trim();
    if (value) {
      persistApiKey(value);
      await fetchJson('/api/v1/scanner/session', {
        method: 'POST',
        body: JSON.stringify({ api_key: value }),
      });
      candidateSessionAuthenticated = true;
    } else {
      clearApiKey();
      await fetchJson('/api/v1/scanner/session', { method: 'DELETE' }).catch(() => null);
      candidateSessionAuthenticated = false;
    }
    await writeApiKeyToDb(value);
    state.loaded = false;
    apiKeyPanelExpanded = !value;
    updateApiKeyPanel();
    setStatus(value ? 'Scanner session saved on this device.' : 'Scanner API key cleared.', value ? 'ok' : '');
  }

  function syncInput() {
    const input = document.getElementById('candidateApiKeyInput');
    if (input && !input.value) input.value = apiKey();
    updateApiKeyPanel();
  }

  function updateApiKeyPanel() {
    const band = document.getElementById('candidateApiBand');
    const toggle = document.getElementById('candidateApiToggle');
    if (!band) return;
    const hasKey = Boolean(apiKey()) || candidateSessionAuthenticated;
    const collapsed = hasKey && !apiKeyPanelExpanded;
    band.classList.toggle('collapsed', collapsed);
    if (toggle) toggle.textContent = collapsed ? 'Change Key' : hasKey ? 'Hide Key' : 'Change Key';
  }

  function toggleCandidateApiKeyPanel() {
    apiKeyPanelExpanded = !apiKeyPanelExpanded;
    updateApiKeyPanel();
  }

  async function loadCandidateDashboard(force = false) {
    await hydrateApiKey();
    syncInput();
    if (state.loading || (state.loaded && !force)) {
      render();
      return;
    }
    state.loading = true;
    state.error = null;
    setStatus('Loading candidate inbox...');
    render();
    try {
      const [candidates, promotions, planPreviewsResult, chartReviews] = await Promise.all([
        fetchJson('/api/v1/scanner/candidates'),
        fetchJson('/api/v1/scanner/candidate-promotions'),
        fetchJson('/api/v1/scanner/candidate-plan-previews').catch((error) => ({ __previewError: error.message || String(error) })),
        fetchJson('/api/v1/scanner/candidate-chart-reviews'),
      ]);
      state.candidates = Array.isArray(candidates) ? candidates : [];
      state.promotions = Array.isArray(promotions) ? promotions : [];
      const previewError = planPreviewsResult && planPreviewsResult.__previewError;
      state.planPreviews = Array.isArray(planPreviewsResult) ? planPreviewsResult : [];
      state.chartReviews = Array.isArray(chartReviews) ? chartReviews : [];
      state.loaded = true;
      candidateSessionAuthenticated = true;
      updateApiKeyPanel();
      notifyForNewCandidates(state.candidates);
      const previewStatus = previewError
        ? ` Plan previews unavailable temporarily: ${previewError}`
        : '';
      const cleanCount = enterNowCandidates().length;
      setStatus(`Loaded ${cleanCount} ENTER_NOW cards from ${state.candidates.length} scanned candidates. Raw Medium/High candidates stay in audit/legacy.${previewStatus}`, previewError ? 'error' : 'ok');
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
    const clean = enterNowCandidates();
    if (state.view === 'all') return clean;
    return clean.filter(item => String(item.status || 'new').toLowerCase() === state.view);
  }

  function renderStats() {
    const el = document.getElementById('candidateStats');
    if (!el) return;
    const clean = enterNowCandidates();
    const counts = clean.reduce((acc, item) => {
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
    updateCandidateAlertToggle();
    renderStats();
    const list = document.getElementById('candidateList');
    if (!list) return;
    if (!apiKey() && !candidateSessionAuthenticated && !state.loaded && !state.error) {
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
    const promoMap = promotionsByKey();
    const previewMap = planPreviewsByKey();
    const blockReason = routeBlockReason(item, effectivePlanForCandidate(item, promoMap, previewMap));
    const promote = status !== 'active' && !blockReason
      ? `<button onclick="updateCandidateStatus('${ticker}','${source}','active','${status}')">Promote</button>`
      : '';
    const chartRead = `<button class="btn-ghost" onclick="requestChartReview('${ticker}','${source}')">Get AI Chart Read</button>`;
    const dismiss = status !== 'dismissed' && status !== 'active'
      ? `<button class="btn-secondary" onclick="updateCandidateStatus('${ticker}','${source}','dismissed','${status}')">Dismiss</button>`
      : '';
    const restore = status !== 'new'
      ? `<button class="btn-ghost" onclick="updateCandidateStatus('${ticker}','${source}','new','${status}')">Back to Inbox</button>`
      : '';
    const blocked = blockReason
      ? `<span class="candidate-pill bad" title="${escapeHtml(blockReason)}">Not dashboard-ready</span>`
      : '';
    return `<div class="candidate-actions">${promote}${chartRead}${dismiss}${restore}${blocked}</div>`;
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

  function candidateAlertId(item) {
    return [
      String(item.ticker || '').toUpperCase(),
      item.source || '',
      item.signal || '',
      item.entry_price ?? '',
      item.scanned_at || item.updated_at || '',
    ].join('|');
  }

  function candidateNotificationTitle(item) {
    const direction = String(item.signal || '').toUpperCase();
    return `${String(item.ticker || 'Ticker').toUpperCase()} ${direction || 'Candidate'}`;
  }

  function candidateNotificationBody(item) {
    const confidence = item.confidence ? `${String(item.confidence).toUpperCase()} confidence` : 'New dashboard candidate';
    return `${confidence} · Entry ${fmtMoney(item.entry_price)} · scanned ${fmtTime(item.scanned_at)}`;
  }

  function playCandidateAlertTone() {
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      const ctx = new AudioCtx();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.value = 820;
      gain.gain.setValueAtTime(0.0001, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.08, ctx.currentTime + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.45);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.5);
    } catch (_) {}
  }

  function sendCandidateTestNotification() {
    if (!('Notification' in window) || Notification.permission !== 'granted') return;
    new Notification('Kairos Stock Alerts On', {
      body: 'You will be notified when new inbox candidates appear.',
      tag: 'kairos-stock-alerts-enabled',
    });
  }

  function updateCandidateAlertToggle() {
    const btn = document.getElementById('candidateAlertToggle');
    if (!btn) return;
    const blocked = 'Notification' in window && Notification.permission === 'denied';
    btn.className = 'alert-toggle' + (blocked ? ' blocked' : candidateAlertsEnabled ? ' on' : '');
    btn.textContent = blocked ? 'Alerts Blocked' : candidateAlertsEnabled ? 'Alerts On' : 'Alerts Off';
  }

  async function toggleCandidateAlerts() {
    if (!('Notification' in window)) {
      candidateAlertsEnabled = !candidateAlertsEnabled;
      localStorage.setItem(ALERT_KEY, candidateAlertsEnabled ? 'on' : 'off');
      updateCandidateAlertToggle();
      if (candidateAlertsEnabled) playCandidateAlertTone();
      return;
    }

    if (!candidateAlertsEnabled && Notification.permission === 'default') {
      const permission = await Notification.requestPermission();
      if (permission !== 'granted') {
        candidateAlertsEnabled = false;
        localStorage.setItem(ALERT_KEY, 'off');
        updateCandidateAlertToggle();
        return;
      }
    }

    if (Notification.permission === 'denied') {
      candidateAlertsEnabled = false;
      localStorage.setItem(ALERT_KEY, 'off');
      updateCandidateAlertToggle();
      return;
    }

    candidateAlertsEnabled = !candidateAlertsEnabled;
    localStorage.setItem(ALERT_KEY, candidateAlertsEnabled ? 'on' : 'off');
    if (candidateAlertsEnabled) {
      playCandidateAlertTone();
      sendCandidateTestNotification();
    }
    updateCandidateAlertToggle();
  }

  function notifyForNewCandidates(candidates) {
    const promoMap = promotionsByKey();
    const previewMap = planPreviewsByKey();
    const inbox = candidates.filter(item => (
      String(item.status || 'new').toLowerCase() === 'new'
      && isEnterNowCandidate(item, promoMap, previewMap)
    ));
    const nextIds = new Set(inbox.map(candidateAlertId));
    const newItems = inbox.filter(item => !knownCandidateIds.has(candidateAlertId(item)));

    if (candidateAlertsInitialized && candidateAlertsEnabled && newItems.length) {
      playCandidateAlertTone();
      document.title = `(${newItems.length}) New Kairos Candidate`;
      if ('Notification' in window && Notification.permission === 'granted') {
        newItems.slice(0, 3).forEach(item => {
          new Notification(candidateNotificationTitle(item), {
            body: candidateNotificationBody(item),
            tag: candidateAlertId(item),
          });
        });
      }
    }

    knownCandidateIds = nextIds;
    candidateAlertsInitialized = true;
  }

  window.saveCandidateApiKey = saveCandidateApiKey;
  window.toggleCandidateApiKeyPanel = toggleCandidateApiKeyPanel;
  window.toggleCandidateAlerts = toggleCandidateAlerts;
  window.loadCandidateDashboard = loadCandidateDashboard;
  window.setCandidateView = setCandidateView;
  window.updateCandidateStatus = updateCandidateStatus;
  window.requestChartReview = requestChartReview;
})();
