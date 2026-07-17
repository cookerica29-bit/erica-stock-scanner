(function(root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.KairosSetupFunnel = factory();
  }
})(typeof self !== 'undefined' ? self : this, function() {
  const CONFIG = {
    historyLimit: 20,
    storageKey: 'kairos_enter_now_funnel_history_v1',
    lowOpportunityGradeEligibleMaxRate: 0.10,
    bottleneckReactionStartedMin: 8,
    bottleneckEnterNowMaxRate: 0.20,
    bottleneckPrimaryShareMin: 0.45,
    insufficientProcessedSymbols: 20,
  };

  const STAGES = [
    'directional_setup_found',
    'bias_aligned',
    'location_aligned',
    'grade_eligible',
    'area_reached',
    'reaction_started',
    'enter_now',
    'tradeable',
    'watchlist',
    'no_trade',
  ];

  const REASON_TYPES = {
    STRATEGY: 'strategy',
    DATA: 'data_quality',
    OPTION: 'option_contract',
    CAUTION: 'informational_caution',
  };

  const PRIMARY_REASON_PRIORITY = [
    'MISSING_MARKET_DATA',
    'NO_DIRECTIONAL_BIAS',
    'HTF_CONFLICT',
    'LOCATION_CONFLICT',
    'GRADE_NOT_ELIGIBLE',
    'ENTRY_AREA_NOT_REACHED',
    'REACTION_NOT_STARTED',
    'ENTRY_TRIGGER_NOT_CONFIRMED',
    'STALE_OR_EXTENDED',
    'POOR_RISK_REWARD',
    'OPTION_LIQUIDITY',
    'EARNINGS_CAUTION',
    'OTHER_EXISTING_BLOCK',
  ];

  function normalizeText(value) {
    return String(value || '').trim();
  }

  function upper(value) {
    return normalizeText(value).toUpperCase();
  }

  function firstPresent(...values) {
    return values.find(value => value !== undefined && value !== null && value !== '');
  }

  function finiteNumber(value) {
    if (value === null || value === undefined || value === '') return null;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function marketDateFrom(timestamp = new Date().toISOString()) {
    return String(timestamp).slice(0, 10);
  }

  function stableHash(input) {
    let hash = 2166136261;
    const text = String(input || '');
    for (let i = 0; i < text.length; i += 1) {
      hash ^= text.charCodeAt(i);
      hash = Math.imul(hash, 16777619);
    }
    return (hash >>> 0).toString(36);
  }

  function setupIdentity(setup = {}) {
    const explicit = firstPresent(setup.setup_id, setup.setupId, setup.funnel_setup_id, setup.id);
    if (explicit) return String(explicit);
    const parts = [
      upper(setup.ticker),
      upper(setup.direction),
      upper(firstPresent(setup.timeframe, setup.scanner_timeframe, setup.setupTf)),
      firstPresent(setup.signal_timestamp, setup.candleTime, setup.scannedAt, setup.updated_at),
      firstPresent(setup.entry, setup.entry_price),
      firstPresent(setup.sl, setup.stop, setup.stop_price),
      firstPresent(setup.tp1, setup.target, setup.target_price),
    ];
    return `funnel:${parts[0] || 'UNKNOWN'}:${stableHash(parts.join('|'))}`;
  }

  function gradeValue(setup = {}) {
    const raw = upper(firstPresent(setup.grade_value, setup.setupGrade, setup.setup_grade, setup.scanner_status, setup.trade_eval?.trade_stage));
    if (raw.includes('A')) return 'A';
    if (raw.includes('B')) return 'B';
    if (raw.includes('C')) return 'C';
    return '';
  }

  function stageBucket(setup = {}) {
    return upper(firstPresent(setup.progress_bucket, setup.simple_status, setup.readiness_bucket));
  }

  function hasDirectionalBias(setup = {}) {
    return ['LONG', 'SHORT'].includes(upper(setup.direction));
  }

  function hasMarketData(setup = {}) {
    return finiteNumber(firstPresent(setup.price, setup.current_price, setup.underlying_price)) !== null;
  }

  function htfConflict(setup = {}) {
    if (setup.htf_conflict === true || setup.htfConflict === true || setup.trade_eval?.htf_conflict === true) return true;
    const direction = upper(setup.direction);
    const htf = upper(firstPresent(setup.higherTimeframeAlignment, setup.htfAlignment, setup.setupTimeframeDirection, setup.trend_state, setup.stockTrend));
    if (direction === 'LONG' && htf.includes('BEARISH') && htf.includes('CONFLICT')) return true;
    if (direction === 'SHORT' && htf.includes('BULLISH') && htf.includes('CONFLICT')) return true;
    return false;
  }

  function biasAligned(setup = {}) {
    if (setup.bias_aligned === true || setup.htf_aligned === true || setup.trade_eval?.bias_aligned === true) return true;
    if (htfConflict(setup)) return false;
    return hasDirectionalBias(setup);
  }

  function locationConflict(setup = {}) {
    if (setup.location_conflict === true || setup.trade_eval?.location_conflict === true) return true;
    const direction = upper(setup.direction);
    const loc = upper(firstPresent(setup.stockLocation, setup.location, setup.trade_eval?.location));
    if (direction === 'LONG' && loc.includes('PREMIUM')) return true;
    if (direction === 'SHORT' && loc.includes('DISCOUNT')) return true;
    return false;
  }

  function locationAligned(setup = {}) {
    if (setup.location_aligned === true || setup.trade_eval?.location_aligned === true) return true;
    if (locationConflict(setup)) return false;
    const direction = upper(setup.direction);
    const loc = upper(firstPresent(setup.stockLocation, setup.location, setup.trade_eval?.location));
    if (direction === 'LONG' && loc.includes('DISCOUNT')) return true;
    if (direction === 'SHORT' && loc.includes('PREMIUM')) return true;
    return false;
  }

  function gradeEligible(setup = {}) {
    const grade = gradeValue(setup);
    if (grade === 'C') return false;
    const tradeStage = upper(setup.trade_eval?.trade_stage);
    return grade === 'A' || grade === 'B' || tradeStage.includes('A+ READY') || tradeStage.includes('B+ TRADEABLE');
  }

  function areaReached(setup = {}) {
    const bucket = stageBucket(setup);
    const entryStatus = normalizeText(firstPresent(setup.entryStatus, setup.entry_status));
    return ['ENTER_NOW', 'ALMOST_READY'].includes(bucket)
      || ['Tradeable', 'Near Entry'].includes(entryStatus)
      || setup.area_reached === true
      || setup.trade_eval?.area_reached === true;
  }

  function reactionStarted(setup = {}) {
    const bucket = stageBucket(setup);
    const entryStatus = normalizeText(firstPresent(setup.entryStatus, setup.entry_status));
    return bucket === 'ENTER_NOW'
      || setup.confirmationStarted === true
      || setup.reaction_started === true
      || setup.trade_eval?.rejection_confirmed === true
      || setup.trade_eval?.trigger_confirmed === true
      || entryStatus === 'Tradeable';
  }

  function enterNow(setup = {}) {
    if (gradeValue(setup) === 'C') return false;
    return stageBucket(setup) === 'ENTER_NOW'
      || setup.trade_eval?.a_plus_ready === true
      || (setup.trade_eval?.trigger_confirmed === true && normalizeText(setup.entryStatus) === 'Tradeable');
  }

  function tradeable(setup = {}) {
    if (gradeValue(setup) === 'C') return false;
    return enterNow(setup) || normalizeText(setup.entryStatus) === 'Tradeable' || setup.trade_eval?.b_plus_tradeable === true;
  }

  function watchlist(setup = {}) {
    const stage = upper(setup.trade_eval?.trade_stage);
    return stage.includes('WATCHLIST') || stage.includes('BUILDING') || stageBucket(setup) === 'WAITING';
  }

  function noTrade(setup = {}) {
    return stageBucket(setup) === 'SKIP' || gradeValue(setup) === 'C' || upper(setup.trade_eval?.trade_stage).includes('NO TRADE');
  }

  function riskRewardPoor(setup = {}) {
    if (setup.risk_reward_blocked === true || setup.trade_eval?.risk_reward_blocked === true) return true;
    const rr = finiteNumber(firstPresent(setup.rr, setup.riskReward, setup.reward_risk));
    return rr !== null && rr > 0 && rr < 2;
  }

  function staleOrExtended(setup = {}) {
    const entryStatus = upper(firstPresent(setup.entryStatus, setup.entry_status));
    return entryStatus.includes('TOO FAR') || entryStatus.includes('LATE') || upper(setup.setupStatus).includes('STALE');
  }

  function optionLimitation(setup = {}) {
    const lifecycle = upper(firstPresent(setup.contract_lifecycle, setup.best_contract?.source));
    return setup.option_blocks_entry === true || lifecycle === 'CONFIRMED_UNAVAILABLE';
  }

  function earningsCaution(setup = {}) {
    return setup.earnings_blocks_entry === true || setup.trade_eval?.earnings_blocks_entry === true;
  }

  function stageFlags(setup = {}) {
    const isEnterNow = enterNow(setup);
    return {
      directional_setup_found: hasDirectionalBias(setup),
      bias_aligned: biasAligned(setup),
      location_aligned: locationAligned(setup),
      grade_eligible: gradeEligible(setup),
      area_reached: areaReached(setup),
      reaction_started: reactionStarted(setup),
      enter_now: isEnterNow,
      tradeable: tradeable(setup),
      watchlist: watchlist(setup),
      no_trade: isEnterNow ? false : noTrade(setup),
    };
  }

  function lastStageReached(flags = {}) {
    const order = ['enter_now', 'reaction_started', 'area_reached', 'grade_eligible', 'location_aligned', 'bias_aligned', 'directional_setup_found'];
    const found = order.find(stage => flags[stage]);
    return found ? found.toUpperCase() : 'NO_TRADE';
  }

  function reasonType(code) {
    if (code === 'MISSING_MARKET_DATA') return REASON_TYPES.DATA;
    if (code === 'OPTION_LIQUIDITY') return REASON_TYPES.OPTION;
    if (code === 'EARNINGS_CAUTION') return REASON_TYPES.CAUTION;
    return REASON_TYPES.STRATEGY;
  }

  function blockingReasons(setup = {}, flags = stageFlags(setup)) {
    if (flags.enter_now) return [];
    const reasons = [];
    if (!hasMarketData(setup)) reasons.push('MISSING_MARKET_DATA');
    if (!flags.directional_setup_found) reasons.push('NO_DIRECTIONAL_BIAS');
    if (htfConflict(setup) || !flags.bias_aligned) reasons.push('HTF_CONFLICT');
    if (locationConflict(setup) || !flags.location_aligned) reasons.push('LOCATION_CONFLICT');
    if (!flags.grade_eligible) reasons.push('GRADE_NOT_ELIGIBLE');
    if (flags.grade_eligible && !flags.area_reached) reasons.push('ENTRY_AREA_NOT_REACHED');
    if (flags.area_reached && !flags.reaction_started) reasons.push('REACTION_NOT_STARTED');
    if (flags.reaction_started && !flags.enter_now) reasons.push('ENTRY_TRIGGER_NOT_CONFIRMED');
    if (staleOrExtended(setup)) reasons.push('STALE_OR_EXTENDED');
    if (riskRewardPoor(setup)) reasons.push('POOR_RISK_REWARD');
    if (optionLimitation(setup)) reasons.push('OPTION_LIQUIDITY');
    if (earningsCaution(setup)) reasons.push('EARNINGS_CAUTION');
    return Array.from(new Set(reasons.length ? reasons : ['OTHER_EXISTING_BLOCK']));
  }

  function primaryBlockingReason(reasons = []) {
    if (!reasons.length) return null;
    return reasons.slice().sort((a, b) => {
      const aRank = PRIMARY_REASON_PRIORITY.indexOf(a);
      const bRank = PRIMARY_REASON_PRIORITY.indexOf(b);
      return (aRank === -1 ? 999 : aRank) - (bRank === -1 ? 999 : bRank) || a.localeCompare(b);
    })[0];
  }

  function setupDiagnostic(setup = {}) {
    const flags = stageFlags(setup);
    const reasons = blockingReasons(setup, flags);
    const primary = primaryBlockingReason(reasons);
    return {
      setup_id: setupIdentity(setup),
      ticker: upper(setup.ticker),
      direction: upper(setup.direction),
      timeframe: firstPresent(setup.timeframe, setup.scanner_timeframe, setup.setupTf) || '',
      grade: firstPresent(setup.setupGrade, setup.setup_grade, setup.scanner_status, setup.entryStatus) || '',
      status: firstPresent(setup.progress_label, setup.entryStatus, setup.trade_eval?.trade_stage) || '',
      signal_timestamp: firstPresent(setup.signal_timestamp, setup.candleTime, setup.scannedAt, setup.updated_at) || '',
      entry: firstPresent(setup.entry, setup.entry_price),
      stop: firstPresent(setup.sl, setup.stop, setup.stop_price),
      target: firstPresent(setup.tp1, setup.target, setup.target_price),
      atr: firstPresent(setup.atr, setup.atr_at_signal),
      price_at_scan: firstPresent(setup.price, setup.current_price, setup.underlying_price),
      stage_flags: flags,
      last_stage_reached: lastStageReached(flags),
      enter_now_eligible: flags.enter_now,
      primary_blocking_reason: primary,
      all_blocking_reasons: reasons,
      reason_types: Object.fromEntries(reasons.map(reason => [reason, reasonType(reason)])),
    };
  }

  function countBy(items, keyFn) {
    return items.reduce((acc, item) => {
      const key = keyFn(item);
      if (!key) return acc;
      acc[key] = (acc[key] || 0) + 1;
      return acc;
    }, {});
  }

  function conversionRate(numerator, denominator) {
    return denominator ? numerator / denominator : null;
  }

  function largestEntry(counts = {}) {
    return Object.entries(counts).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))[0] || null;
  }

  function classifyScan(stageCounts, primaryCounts, symbolsProcessed) {
    if (symbolsProcessed < CONFIG.insufficientProcessedSymbols) return 'INSUFFICIENT_DATA';
    const gradeRate = conversionRate(stageCounts.grade_eligible || 0, symbolsProcessed) || 0;
    const enterFromReaction = conversionRate(stageCounts.enter_now || 0, stageCounts.reaction_started || 0) || 0;
    const largest = largestEntry(primaryCounts);
    const nonEnter = symbolsProcessed - (stageCounts.enter_now || 0);
    const largestShare = largest && nonEnter ? largest[1] / nonEnter : 0;
    if (gradeRate <= CONFIG.lowOpportunityGradeEligibleMaxRate) return 'LOW_OPPORTUNITY_ENVIRONMENT';
    if ((stageCounts.reaction_started || 0) >= CONFIG.bottleneckReactionStartedMin
      && enterFromReaction <= CONFIG.bottleneckEnterNowMaxRate
      && largestShare >= CONFIG.bottleneckPrimaryShareMin) {
      return 'POSSIBLE_FINAL_GATE_BOTTLENECK';
    }
    return 'NORMAL_SELECTIVITY';
  }

  function createScanSnapshot({ rows = [], nearMiss = [], meta = {}, scanId = null, timestamp = new Date().toISOString() } = {}) {
    const allRows = [...rows, ...nearMiss];
    const unique = new Map();
    allRows.forEach(setup => unique.set(setupIdentity(setup), setup));
    const setups = Array.from(unique.values()).map(setupDiagnostic);
    const stageCounts = Object.fromEntries(STAGES.map(stage => [stage, 0]));
    setups.forEach(item => {
      STAGES.forEach(stage => {
        if (item.stage_flags[stage]) stageCounts[stage] += 1;
      });
    });
    const nonEnter = setups.filter(item => !item.enter_now_eligible);
    const primaryCounts = countBy(nonEnter, item => item.primary_blocking_reason);
    const allReasonCounts = {};
    nonEnter.forEach(item => {
      item.all_blocking_reasons.forEach(reason => {
        allReasonCounts[reason] = (allReasonCounts[reason] || 0) + 1;
      });
    });
    const symbolsRequested = finiteNumber(meta.symbols_requested) ?? finiteNumber(meta.requested_count) ?? allRows.length;
    const symbolsProcessed = finiteNumber(meta.symbols_processed) ?? unique.size;
    const symbolsFailed = finiteNumber(meta.symbols_failed) ?? Math.max(0, symbolsRequested - symbolsProcessed);
    const largest = largestEntry(primaryCounts);
    return {
      scan_id: scanId || `scan_${timestamp}_${stableHash(allRows.map(setupIdentity).join('|'))}`,
      scan_timestamp: timestamp,
      market_date: meta.market_date || marketDateFrom(timestamp),
      symbols_requested: symbolsRequested,
      symbols_processed: symbolsProcessed,
      symbols_failed: symbolsFailed,
      stage_counts: stageCounts,
      conversion_rates: {
        directional_to_grade_eligible: conversionRate(stageCounts.grade_eligible, stageCounts.directional_setup_found),
        grade_eligible_to_area_reached: conversionRate(stageCounts.area_reached, stageCounts.grade_eligible),
        area_reached_to_reaction_started: conversionRate(stageCounts.reaction_started, stageCounts.area_reached),
        reaction_started_to_enter_now: conversionRate(stageCounts.enter_now, stageCounts.reaction_started),
        overall_enter_now_rate: conversionRate(stageCounts.enter_now, symbolsProcessed),
      },
      primary_blocking_reason_counts: primaryCounts,
      all_blocking_reason_counts: allReasonCounts,
      largest_bottleneck: largest ? { reason: largest[0], count: largest[1] } : null,
      diagnostic_classification: classifyScan(stageCounts, primaryCounts, symbolsProcessed),
      setups,
      missed_opportunity_candidates: setups.filter(item => item.stage_flags.grade_eligible && !item.enter_now_eligible).map(item => ({
        setup_id: item.setup_id,
        ticker: item.ticker,
        direction: item.direction,
        timeframe: item.timeframe,
        grade: item.grade,
        status: item.status,
        signal_timestamp: item.signal_timestamp,
        entry: item.entry,
        stop: item.stop,
        target: item.target,
        ATR: item.atr,
        last_stage_reached: item.last_stage_reached,
        primary_blocking_reason: item.primary_blocking_reason,
        all_blocking_reasons: item.all_blocking_reasons,
        price_at_scan: item.price_at_scan,
      })),
    };
  }

  function appendHistory(history = [], snapshot, limit = CONFIG.historyLimit) {
    const merged = [snapshot, ...history.filter(item => item.scan_id !== snapshot.scan_id)];
    return merged.slice(0, limit);
  }

  function loadHistory(storage) {
    if (!storage) return [];
    try {
      const parsed = JSON.parse(storage.getItem(CONFIG.storageKey) || '[]');
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }

  function saveHistory(storage, history = []) {
    if (!storage) return;
    try { storage.setItem(CONFIG.storageKey, JSON.stringify(history.slice(0, CONFIG.historyLimit))); } catch {}
  }

  function recordScan(storage, snapshot) {
    const history = appendHistory(loadHistory(storage), snapshot);
    saveHistory(storage, history);
    return history;
  }

  function dailyAggregation(history = []) {
    const byDate = new Map();
    history.forEach(scan => {
      const date = scan.market_date || marketDateFrom(scan.scan_timestamp);
      if (!byDate.has(date)) {
        byDate.set(date, {
          market_date: date,
          setupIds: new Set(),
          enterNowIds: new Set(),
          aEnterIds: new Set(),
          bEnterIds: new Set(),
          longEnterIds: new Set(),
          shortEnterIds: new Set(),
          enter_now_by_timeframe: {},
        });
      }
      const row = byDate.get(date);
      (scan.setups || []).forEach(setup => {
        row.setupIds.add(setup.setup_id);
        if (!setup.enter_now_eligible) return;
        row.enterNowIds.add(setup.setup_id);
        const grade = upper(setup.grade);
        if (grade.startsWith('A')) row.aEnterIds.add(setup.setup_id);
        if (grade.startsWith('B')) row.bEnterIds.add(setup.setup_id);
        if (setup.direction === 'LONG') row.longEnterIds.add(setup.setup_id);
        if (setup.direction === 'SHORT') row.shortEnterIds.add(setup.setup_id);
        const tf = setup.timeframe || 'Unknown';
        row.enter_now_by_timeframe[tf] = row.enter_now_by_timeframe[tf] || new Set();
        row.enter_now_by_timeframe[tf].add(setup.setup_id);
      });
    });
    return Array.from(byDate.values()).map(row => ({
      market_date: row.market_date,
      unique_setups_seen: row.setupIds.size,
      unique_enter_now_setups: row.enterNowIds.size,
      a_plus_enter_now: row.aEnterIds.size,
      b_plus_tradeable_enter_now: row.bEnterIds.size,
      long_enter_now: row.longEnterIds.size,
      short_enter_now: row.shortEnterIds.size,
      enter_now_by_timeframe: Object.fromEntries(Object.entries(row.enter_now_by_timeframe).map(([tf, ids]) => [tf, ids.size])),
    })).sort((a, b) => b.market_date.localeCompare(a.market_date));
  }

  return {
    CONFIG,
    STAGES,
    REASON_TYPES,
    setupIdentity,
    stageFlags,
    blockingReasons,
    primaryBlockingReason,
    setupDiagnostic,
    createScanSnapshot,
    appendHistory,
    loadHistory,
    saveHistory,
    recordScan,
    dailyAggregation,
  };
});
