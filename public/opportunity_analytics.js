(function(root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.KairosOpportunityAnalytics = factory(root.KairosExpectedMove);
  }
})(typeof self !== 'undefined' ? self : this, function(expectedMoveModule) {
  const CALCULATION_VERSION = 'opportunity-v1.0-shadow';
  const CONFIG = {
    minimumTargetSamples: 30,
    bands: {
      earlyMinRemaining: 75,
      onTimeMinRemaining: 50,
      midMoveMinRemaining: 30,
      lateMinRemaining: 1,
    },
    confidenceMap: {
      high: 'HIGH',
      moderate: 'MODERATE',
      low: 'LOW',
      insufficient: 'INSUFFICIENT',
    },
  };

  function finiteNumber(value) {
    if (value === null || value === undefined || value === '') return null;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function firstPresent(...values) {
    return values.find(value => value !== undefined && value !== null && value !== '');
  }

  function upper(value) {
    return String(value || '').trim().toUpperCase();
  }

  function clamp(value, min = 0, max = 100) {
    const number = finiteNumber(value);
    if (number === null) return null;
    return Math.min(max, Math.max(min, number));
  }

  function normalizeDirection(value) {
    const raw = upper(value);
    if (raw.includes('SHORT') || raw === 'PUT') return 'SHORT';
    if (raw.includes('LONG') || raw === 'CALL') return 'LONG';
    return 'UNKNOWN';
  }

  function normalizeTimeframe(value) {
    const raw = String(value || '').trim();
    if (!raw) return 'UNKNOWN';
    const up = raw.toUpperCase();
    if (['1D', 'D', 'DAY', 'DAILY'].includes(up)) return 'Daily';
    if (['1W', 'W', 'WEEK', 'WEEKLY'].includes(up)) return 'Weekly';
    return up;
  }

  function normalizeGrade(value) {
    const raw = upper(value);
    if (raw.includes('B_PLUS') || raw.includes('B+') || raw.includes('TRADEABLE')) return 'B_PLUS_TRADEABLE';
    if (raw.includes('A_PLUS') || raw.includes('A+') || raw.includes('READY')) return 'A_PLUS_READY';
    return raw || 'UNKNOWN';
  }

  function setupCriteria(subject = {}) {
    return {
      ticker: upper(subject.ticker) || null,
      grade: normalizeGrade(firstPresent(subject.setup_grade, subject.setupGrade, subject.scanner_status, subject.scannerStatus, subject.entryStatus)),
      direction: normalizeDirection(subject.direction),
      timeframe: normalizeTimeframe(firstPresent(subject.scanner_timeframe, subject.timeframe, subject.setupTf)),
      setup_family: firstPresent(subject.setup_family, subject.setupFamily, subject.entryType, subject.entry_type, subject.phase, subject.setupStatus, null),
      volatility_bucket: firstPresent(subject.volatility_bucket, subject.volatilityBucket, null),
    };
  }

  function countWeekdaysBetween(startValue, endValue) {
    if (!startValue || !endValue) return null;
    const start = new Date(startValue);
    const end = new Date(endValue);
    if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) return null;
    const s = new Date(start.getFullYear(), start.getMonth(), start.getDate());
    const e = new Date(end.getFullYear(), end.getMonth(), end.getDate());
    if (e < s) return null;
    let days = 0;
    const cursor = new Date(s);
    while (cursor < e) {
      cursor.setDate(cursor.getDate() + 1);
      const weekday = cursor.getDay();
      if (weekday !== 0 && weekday !== 6) days += 1;
    }
    return days;
  }

  function priceLevels(subject = {}) {
    return {
      direction: normalizeDirection(subject.direction),
      currentPrice: finiteNumber(firstPresent(subject.current_price, subject.currentPrice, subject.price, subject.underlying_price, subject.last_price)),
      entry: finiteNumber(firstPresent(subject.actual_entry_price, subject.entry_price, subject.entry, subject.plannedEntry)),
      plannedEntry: finiteNumber(firstPresent(subject.planned_entry, subject.plannedEntry, subject.entry_price, subject.entry)),
      stop: finiteNumber(firstPresent(subject.stop_price, subject.plannedStop, subject.sl, subject.stop)),
      tp1: finiteNumber(firstPresent(subject.target_price, subject.plannedTp1, subject.tp1, subject.target, subject.target_price)),
      tp2: finiteNumber(firstPresent(subject.plannedTp2, subject.tp2)),
      tp3: finiteNumber(firstPresent(subject.plannedTp3, subject.tp3)),
    };
  }

  function rawPriceProgress(levels) {
    const { direction, currentPrice, plannedEntry, tp1 } = levels;
    if (direction === 'UNKNOWN' || currentPrice === null || plannedEntry === null || tp1 === null) return null;
    const denominator = direction === 'SHORT' ? plannedEntry - tp1 : tp1 - plannedEntry;
    if (!denominator || denominator <= 0) return null;
    const numerator = direction === 'SHORT' ? plannedEntry - currentPrice : currentPrice - plannedEntry;
    return (numerator / denominator) * 100;
  }

  function remainingDistance(level, currentPrice, direction) {
    const target = finiteNumber(level);
    const price = finiteNumber(currentPrice);
    if (target === null || price === null || direction === 'UNKNOWN') return null;
    return direction === 'SHORT' ? price - target : target - price;
  }

  function hasEntryTriggered(subject = {}, levels = priceLevels(subject)) {
    if (subject.first_entry_touch_at || subject.entry_reached_at || subject.position_opened_at || subject.position_open_at) return true;
    const status = upper(firstPresent(subject.entryStatus, subject.entry_status, subject.trade_progress_status, ''));
    if (status.includes('ENTRY REACHED') || status.includes('TRADEABLE') || status.includes('OPEN')) return true;
    const price = levels.currentPrice;
    const entry = levels.plannedEntry;
    if (price === null || entry === null) return false;
    const tolerance = finiteNumber(firstPresent(subject.entry_tolerance, subject.entryTolerance));
    if (tolerance !== null) return Math.abs(price - entry) <= tolerance;
    return false;
  }

  function isCompleted(subject = {}, levels = priceLevels(subject), entryTriggered = hasEntryTriggered(subject, levels)) {
    if (subject.completion_reason === 'target' || subject.first_target_touch_at || subject.tp1_reached_at) return true;
    const raw = rawPriceProgress(levels);
    return entryTriggered && raw !== null && raw >= 100;
  }

  function isInvalidated(subject = {}) {
    const text = [
      subject.completion_reason,
      subject.tracking_status,
      subject.status,
      subject.entryStatus,
      subject.setupStatus,
      subject.scanner_status,
      subject.trade_progress_status,
      subject.reviewResult,
      subject.outcome,
    ].map(upper).join(' ');
    return text.includes('STOP')
      || text.includes('INVALID')
      || text.includes('NO TRADE')
      || text.includes('STALE')
      || text.includes('EXPIRED');
  }

  function confidenceFromSampleSize(samples, fallbackUsed = false) {
    if (fallbackUsed) return samples >= CONFIG.minimumTargetSamples ? 'LOW' : 'INSUFFICIENT';
    if (samples >= 300) return 'HIGH';
    if (samples >= 100) return 'MODERATE';
    if (samples >= CONFIG.minimumTargetSamples) return 'LOW';
    return 'INSUFFICIENT';
  }

  function analyticsFromCohorts(subject = {}, options = {}) {
    const historyModule = options.opportunityHistoryModule || (typeof self !== 'undefined' ? self.KairosOpportunityHistory : null);
    const summaries = Array.isArray(options.cohortSummaries) ? options.cohortSummaries : [];
    if (!historyModule || typeof historyModule.selectCohortForSetup !== 'function' || !summaries.length) return null;
    const selected = historyModule.selectCohortForSetup(subject, summaries, { allowFallback: true });
    const summary = selected?.summary;
    if (!summary || summary.readiness !== 'QUALIFIED') {
      return {
        result: {
          group_used: null,
          fallback_used: false,
          fallback_level: null,
          cohort_key: selected?.requestedKey || null,
          cohort_readiness: summary?.readiness || 'INSUFFICIENT',
        },
        readiness: null,
        confidence: 'INSUFFICIENT',
        sampleSize: summary?.winsToTp1 || 0,
        blockers: [`${summary?.winsToTp1 || 0} of ${CONFIG.minimumTargetSamples} required TP1 completions`],
      };
    }
    const confidence = confidenceFromSampleSize(summary.winsToTp1, selected.fallbackUsed);
    return {
      result: {
        expected_move_min_days: summary.p25TradingDaysToTp1 !== null ? Math.max(1, Math.floor(summary.p25TradingDaysToTp1)) : null,
        expected_move_max_days: summary.p75TradingDaysToTp1 !== null ? Math.max(1, Math.ceil(summary.p75TradingDaysToTp1)) : null,
        median_days_to_target: summary.medianTradingDaysToTp1,
        p25_days_to_target: summary.p25TradingDaysToTp1,
        p75_days_to_target: summary.p75TradingDaysToTp1,
        target_sample_count: summary.winsToTp1,
        sample_count: summary.qualifiedSamples,
        group_used: { cohortKey: summary.cohortKey, cohortLevel: summary.cohortLevelLabel },
        fallback_used: selected.fallbackUsed,
        fallback_level: selected.fallbackLevel,
        cohort_key: summary.cohortKey,
        cohort_level: summary.cohortLevel,
        cohort_level_label: summary.cohortLevelLabel,
        cohort_readiness: summary.readiness,
      },
      readiness: null,
      confidence,
      sampleSize: summary.qualifiedSamples,
      blockers: selected.fallbackUsed ? ['Fallback cohort used; not a ticker-specific release claim.'] : [],
    };
  }

  function analyticsFromHistory(subject = {}, entries = [], options = {}) {
    const cohortHistory = analyticsFromCohorts(subject, options);
    if (cohortHistory && cohortHistory.confidence !== 'INSUFFICIENT') return cohortHistory;
    const expectedMove = options.expectedMoveModule || expectedMoveModule;
    if (!expectedMove || typeof expectedMove.expectedMoveAnalytics !== 'function') {
      return {
        result: null,
        readiness: null,
        confidence: 'INSUFFICIENT',
        sampleSize: 0,
        blockers: ['expected move analytics module unavailable'],
      };
    }
    const criteria = setupCriteria(subject);
    const result = expectedMove.expectedMoveAnalytics(entries, criteria);
    const expiration = expectedMove.suggestedExpirationAnalytics ? expectedMove.suggestedExpirationAnalytics(result) : {};
    const readiness = expectedMove.expectedMoveReleaseReadiness ? expectedMove.expectedMoveReleaseReadiness(result, expiration) : null;
    const cardReady = readiness ? readiness.card_ready === true : result?.target_sample_count >= CONFIG.minimumTargetSamples;
    const confidence = CONFIG.confidenceMap[result?.confidence] || 'INSUFFICIENT';
    const blockers = [];
    if (!cardReady) blockers.push(...(readiness?.reasons || [`${result?.target_sample_count || 0} of ${CONFIG.minimumTargetSamples} required target completions`]));
    return {
      result,
      readiness,
      confidence: cardReady ? confidence : 'INSUFFICIENT',
      sampleSize: result?.target_sample_count || 0,
      blockers: cohortHistory?.blockers?.length ? cohortHistory.blockers : blockers,
    };
  }

  function expectedDurations(history) {
    const result = history.result || {};
    return {
      low: finiteNumber(firstPresent(result.expected_move_min_days, result.p25_days_to_target)),
      high: finiteNumber(firstPresent(result.expected_move_max_days, result.p75_days_to_target)),
      median: finiteNumber(result.median_days_to_target),
    };
  }

  function entryTimestamp(subject = {}) {
    return firstPresent(subject.first_entry_touch_at, subject.entry_reached_at, subject.position_opened_at, subject.position_open_at, subject.actual_entry_at, null);
  }

  function classifyRemaining(remainingPct) {
    const remaining = finiteNumber(remainingPct);
    if (remaining === null) return 'INSUFFICIENT_DATA';
    if (remaining <= 0) return 'COMPLETED';
    if (remaining >= CONFIG.bands.earlyMinRemaining) return 'EARLY';
    if (remaining >= CONFIG.bands.onTimeMinRemaining) return 'ON_TIME';
    if (remaining >= CONFIG.bands.midMoveMinRemaining) return 'MID_MOVE';
    if (remaining >= CONFIG.bands.lateMinRemaining) return 'LATE';
    return 'COMPLETED';
  }

  function opportunityAnalytics(subject = {}, entries = [], options = {}) {
    const now = options.now || new Date().toISOString();
    const levels = priceLevels(subject);
    const history = analyticsFromHistory(subject, entries, options);
    const durations = expectedDurations(history);
    const blockers = [...history.blockers];
    const triggered = hasEntryTriggered(subject, levels);
    const completed = isCompleted(subject, levels, triggered);
    const invalidated = isInvalidated(subject);
    const rawProgress = rawPriceProgress(levels);
    const priceProgress = triggered ? clamp(rawProgress) : null;
    const sufficientHistory = history.confidence !== 'INSUFFICIENT';
    const moveCompletedPct = completed ? 100 : invalidated || !sufficientHistory ? null : triggered ? priceProgress : 0;
    const opportunityRemainingPct = completed ? 0 : invalidated || !sufficientHistory ? null : triggered ? clamp(100 - (priceProgress ?? 0)) : 100;
    const entryAt = entryTimestamp(subject);
    const elapsedTradingDays = triggered ? firstPresent(finiteNumber(subject.trading_days_elapsed), countWeekdaysBetween(entryAt, now)) : null;
    const median = durations.median;
    const timeProgressPct = triggered && median ? clamp((elapsedTradingDays / median) * 100) : null;
    const remainingLow = triggered && durations.low !== null && elapsedTradingDays !== null ? Math.max(0, Math.ceil(durations.low - elapsedTradingDays)) : durations.low;
    const remainingHigh = triggered && durations.high !== null && elapsedTradingDays !== null ? Math.max(0, Math.ceil(durations.high - elapsedTradingDays)) : durations.high;

    let status = 'INSUFFICIENT_DATA';
    if (invalidated) status = 'INVALIDATED';
    else if (completed) status = 'COMPLETED';
    else if (sufficientHistory) status = classifyRemaining(opportunityRemainingPct);

    const remainingToTp1 = remainingDistance(levels.tp1, levels.currentPrice, levels.direction);
    const remainingToTp2 = remainingDistance(levels.tp2, levels.currentPrice, levels.direction);
    const remainingToTp3 = remainingDistance(levels.tp3, levels.currentPrice, levels.direction);
    const expectedMoveRemaining = remainingToTp1 !== null ? Math.max(0, remainingToTp1) : null;

    const explanationParts = [];
    if (status === 'INSUFFICIENT_DATA') explanationParts.push('Historical sample is not yet release-ready for this setup group.');
    else if (!triggered && !completed && !invalidated) explanationParts.push('Entry has not been reached; opportunity is measured from the planned entry, not current price drift.');
    else if (completed) explanationParts.push('TP1 has been reached or the target completion state is already recorded.');
    else if (invalidated) explanationParts.push('Setup or position is invalidated, stopped, stale, or expired.');
    else explanationParts.push(`${Math.round(opportunityRemainingPct)}% of the TP1 move remains based on price progress from entry to TP1.`);
    if (timeProgressPct !== null && priceProgress !== null) explanationParts.push(`Time progress is ${Math.round(timeProgressPct)}% while price progress is ${Math.round(priceProgress)}%.`);

    return {
      available: status !== 'INSUFFICIENT_DATA' && status !== 'INVALIDATED',
      status,
      opportunityRemainingPct,
      moveCompletedPct,
      expectedTradingDaysTotal: {
        low: durations.low,
        high: durations.high,
        median,
      },
      expectedTradingDaysRemaining: {
        low: remainingLow,
        high: remainingHigh,
      },
      priceProgressPct: priceProgress,
      rawPriceProgressPct: rawProgress,
      timeProgressPct,
      expectedMoveRemaining,
      remainingToTp1,
      remainingToTp2,
      remainingToTp3,
      entryTriggered: triggered,
      entryTimestamp: entryAt || null,
      tradingDaysElapsed: elapsedTradingDays,
      sampleSize: history.sampleSize,
      confidence: history.confidence,
      explanation: explanationParts.join(' '),
      blockers,
      calculationVersion: CALCULATION_VERSION,
      grouping: {
        requested: setupCriteria(subject),
        used: history.result?.group_used || null,
        fallbackUsed: Boolean(history.result?.fallback_used),
        fallbackLevel: history.result?.fallback_level || null,
        cohortKey: history.result?.cohort_key || null,
        cohortLevel: history.result?.cohort_level || null,
        cohortReadiness: history.result?.cohort_readiness || null,
      },
    };
  }

  function daysUntilExpiration(expiration, now = new Date().toISOString()) {
    if (!expiration) return null;
    const start = new Date(now);
    const end = new Date(`${String(expiration).slice(0, 10)}T00:00:00`);
    if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) return null;
    return Math.ceil((end - start) / 86400000);
  }

  function contractHealth(subject = {}, opportunity = opportunityAnalytics(subject, []), options = {}) {
    const expiration = firstPresent(subject.expiration_date, subject.expiry, subject.option?.expiry, subject.best_contract?.expiry, subject.best_contract?.expiration);
    const explicitDte = finiteNumber(firstPresent(subject.days_to_expiration_at_entry, subject.dte, subject.contractDte));
    const contractDte = explicitDte !== null ? explicitDte : daysUntilExpiration(expiration, options.now || new Date().toISOString());
    const expectedHigh = finiteNumber(opportunity.expectedTradingDaysRemaining?.high);
    const opportunityRemaining = finiteNumber(opportunity.opportunityRemainingPct);
    if (opportunity.confidence === 'INSUFFICIENT' || expectedHigh === null) {
      return {
        available: false,
        status: 'INSUFFICIENT_DATA',
        contractDte,
        expectedTradingDaysRemaining: expectedHigh,
        timeBufferTradingDays: null,
        explanation: 'Expected move timing is still learning for this setup group.',
      };
    }
    if (contractDte === null) {
      return {
        available: false,
        status: 'INSUFFICIENT_DATA',
        contractDte: null,
        expectedTradingDaysRemaining: expectedHigh,
        timeBufferTradingDays: null,
        explanation: 'Contract expiration or DTE is unavailable.',
      };
    }
    const buffer = contractDte - expectedHigh;
    let status = 'GOOD_MATCH';
    if (contractDte <= 2) status = 'EXPIRING_SOON';
    else if (buffer < 0) status = 'TOO_SHORT';
    else if (opportunity.status === 'LATE' || (opportunityRemaining !== null && opportunityRemaining < 30)) status = 'LATE_MOVE';
    else if (buffer < 5) status = 'ADEQUATE';
    const explanation = {
      GOOD_MATCH: `Contract has ${contractDte} DTE and the expected move requires up to ${expectedHigh} trading days.`,
      ADEQUATE: `Contract has ${contractDte} DTE with a modest buffer over the expected ${expectedHigh} trading-day window.`,
      TOO_SHORT: `Contract has ${contractDte} DTE while similar setups usually require up to ${expectedHigh} trading days.`,
      LATE_MOVE: 'Contract duration may be adequate, but most of the expected move has already occurred.',
      EXPIRING_SOON: `Contract has ${contractDte} DTE and is close to expiration.`,
    }[status];
    return {
      available: true,
      status,
      contractDte,
      expectedTradingDaysRemaining: expectedHigh,
      timeBufferTradingDays: buffer,
      explanation,
    };
  }

  function diagnostics(subjects = [], entries = [], options = {}) {
    return subjects.map(subject => {
      const opportunity = opportunityAnalytics(subject, entries, options);
      const contract = contractHealth(subject, opportunity, options);
      const levels = priceLevels(subject);
      return {
        ticker: upper(subject.ticker) || 'UNKNOWN',
        direction: levels.direction,
        lifecycleState: opportunity.status,
        entryTriggered: opportunity.entryTriggered,
        plannedEntry: levels.plannedEntry,
        currentPrice: levels.currentPrice,
        tp1: levels.tp1,
        priceProgressPct: opportunity.priceProgressPct,
        timeProgressPct: opportunity.timeProgressPct,
        opportunityRemainingPct: opportunity.opportunityRemainingPct,
        expectedTradingDaysTotal: opportunity.expectedTradingDaysTotal,
        expectedTradingDaysRemaining: opportunity.expectedTradingDaysRemaining,
        sampleSize: opportunity.sampleSize,
        confidence: opportunity.confidence,
        contractDte: contract.contractDte,
        contractHealth: contract.status,
        explanation: opportunity.explanation,
        blockers: opportunity.blockers,
        historicalSource: opportunity.grouping?.used?.cohortLevel || opportunity.grouping?.used?.cohortLevelLabel || 'Expected Move journal group',
        cohortKey: opportunity.grouping?.cohortKey || '—',
        cohortLevel: opportunity.grouping?.cohortLevel || '—',
        cohortReadiness: opportunity.grouping?.cohortReadiness || '—',
        fallbackUsed: opportunity.grouping?.fallbackUsed || false,
        calculationVersion: opportunity.calculationVersion,
      };
    });
  }

  return {
    CONFIG,
    CALCULATION_VERSION,
    finiteNumber,
    countWeekdaysBetween,
    setupCriteria,
    priceLevels,
    rawPriceProgress,
    hasEntryTriggered,
    opportunityAnalytics,
    contractHealth,
    diagnostics,
  };
});
