(function(root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.KairosOpportunityHistory = factory();
  }
})(typeof self !== 'undefined' ? self : this, function() {
  const ANALYTICS_VERSION = 'opportunity-history-v1.0-shadow';
  const CONFIG = {
    minimumQualifiedSamples: 30,
    stability: {
      largeSingleChangePct: 35,
      maxBandReversals: 2,
    },
  };

  const QUALITY_RANK = { EXCLUDED: 0, PARTIAL: 1, QUALIFIED: 2 };
  const COHORT_LEVELS = [
    'ticker_direction_family_grade',
    'ticker_direction_family',
    'direction_family_volatility',
    'broader_validated',
  ];

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

  function normalizeDirection(value) {
    const raw = upper(value);
    if (raw.includes('SHORT') || raw === 'PUT') return 'SHORT';
    if (raw.includes('LONG') || raw === 'CALL') return 'LONG';
    return 'UNKNOWN';
  }

  function normalizeGrade(value) {
    const raw = upper(value);
    if (raw.includes('B_PLUS') || raw.includes('B+') || raw.includes('TRADEABLE')) return 'B_PLUS_TRADEABLE';
    if (raw.includes('A_PLUS') || raw.includes('A+') || raw.includes('READY')) return 'A_PLUS_READY';
    if (raw === 'A') return 'A_PLUS_READY';
    if (raw === 'B') return 'B_PLUS_TRADEABLE';
    return raw || 'UNKNOWN';
  }

  function normalizeTimeframe(value) {
    const raw = String(value || '').trim();
    if (!raw) return 'UNKNOWN';
    const up = raw.toUpperCase();
    if (['1D', 'D', 'DAY', 'DAILY'].includes(up)) return 'Daily';
    if (['1W', 'W', 'WEEK', 'WEEKLY'].includes(up)) return 'Weekly';
    return up;
  }

  function normalizeSetupFamily(value) {
    const raw = String(value || '').trim();
    if (!raw) return 'GENERAL';
    return raw.toUpperCase().replace(/\s+/g, '_');
  }

  function countWeekdaysBetween(startValue, endValue) {
    if (!startValue || !endValue) return null;
    const start = new Date(startValue);
    const end = new Date(endValue);
    if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime()) || end < start) return null;
    const cursor = new Date(start.getFullYear(), start.getMonth(), start.getDate());
    const finish = new Date(end.getFullYear(), end.getMonth(), end.getDate());
    let days = 0;
    while (cursor < finish) {
      cursor.setDate(cursor.getDate() + 1);
      const weekday = cursor.getDay();
      if (weekday !== 0 && weekday !== 6) days += 1;
    }
    return days;
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

  function recordFingerprint(parts) {
    return `opp:${stableHash(parts.map(value => value == null ? '' : String(value)).join('|'))}`;
  }

  function percentile(values, p) {
    const nums = values.map(Number).filter(Number.isFinite).sort((a, b) => a - b);
    if (!nums.length) return null;
    if (nums.length === 1) return nums[0];
    const index = (nums.length - 1) * p;
    const lower = Math.floor(index);
    const higher = Math.ceil(index);
    if (lower === higher) return nums[lower];
    const weight = index - lower;
    return nums[lower] + (nums[higher] - nums[lower]) * weight;
  }

  function median(values) {
    return percentile(values, 0.5);
  }

  function normalizeIso(value) {
    if (!value) return null;
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? null : date.toISOString();
  }

  function earliestOutcome(entry = {}) {
    const targetAt = normalizeIso(firstPresent(entry.first_target_touch_at, entry.tp1_reached_at, entry.target_reached_at));
    const stopAt = normalizeIso(firstPresent(entry.first_stop_touch_at, entry.stop_reached_at));
    const invalidatedAt = normalizeIso(firstPresent(entry.invalidated_at, entry.tracking_completed_at));
    const candidates = [
      targetAt ? { reason: 'target', at: targetAt } : null,
      stopAt ? { reason: 'stop', at: stopAt } : null,
      invalidatedAt && upper(entry.completion_reason).includes('INVALID') ? { reason: 'invalidated', at: invalidatedAt } : null,
    ].filter(Boolean).sort((a, b) => new Date(a.at) - new Date(b.at));
    return candidates[0] || null;
  }

  function normalizedFields(entry = {}) {
    const ticker = upper(entry.ticker);
    const direction = normalizeDirection(entry.direction || entry.option_type);
    const grade = normalizeGrade(firstPresent(entry.setup_grade, entry.setupGrade, entry.scanner_status, entry.scannerStatus, entry.entryStatus));
    const timeframe = normalizeTimeframe(firstPresent(entry.scanner_timeframe, entry.timeframe, entry.setupTf));
    const setupFamily = normalizeSetupFamily(firstPresent(entry.setup_family, entry.setupFamily, entry.entryType, entry.entry_type, entry.phase, entry.setupStatus));
    const setupCreatedAt = normalizeIso(firstPresent(entry.signal_timestamp, entry.signalTimestamp, entry.createdAt, entry.date, entry.signal_market_date));
    const entryTriggeredAt = normalizeIso(firstPresent(entry.first_entry_touch_at, entry.entry_reached_at, entry.actual_entry_at, entry.position_opened_at));
    const outcome = earliestOutcome(entry);
    const completionReason = upper(entry.completion_reason);
    const trackingCompletedAt = normalizeIso(entry.tracking_completed_at);
    const completedAt = outcome?.at || trackingCompletedAt || null;
    const plannedEntry = finiteNumber(firstPresent(entry.entry_price, entry.entry, entry.plannedEntry));
    const actualEntry = entryTriggeredAt ? finiteNumber(firstPresent(entry.actual_entry_price, entry.entry_price, entry.entry)) : null;
    const stop = finiteNumber(firstPresent(entry.stop_price, entry.plannedStop, entry.stop, entry.sl));
    const tp1 = finiteNumber(firstPresent(entry.target_price, entry.tp1, entry.plannedTp1, entry.target));
    const tp2 = finiteNumber(firstPresent(entry.tp2, entry.plannedTp2));
    const tp3 = finiteNumber(firstPresent(entry.tp3, entry.plannedTp3));
    const source = firstPresent(entry.opportunity_history_source, entry.source, 'JOURNAL_IMPORT');
    return {
      ticker,
      direction,
      grade,
      timeframe,
      setupFamily,
      setupCreatedAt,
      entryTriggeredAt,
      enteredAt: normalizeIso(firstPresent(entry.entered_at, entry.actual_entry_at, entry.position_opened_at, entryTriggeredAt)),
      completedAt,
      outcomeReason: outcome?.reason || completionReason.toLowerCase() || null,
      plannedEntry,
      actualEntry,
      stop,
      tp1,
      tp2,
      tp3,
      tradingDaysToEntry: finiteNumber(entry.trading_days_to_entry) ?? countWeekdaysBetween(setupCreatedAt, entryTriggeredAt),
      tradingDaysToTp1: finiteNumber(firstPresent(entry.trading_days_to_target, entry.trading_days_to_tp1)) ?? countWeekdaysBetween(entryTriggeredAt, outcome?.reason === 'target' ? outcome.at : firstPresent(entry.first_target_touch_at, entry.tp1_reached_at)),
      tradingDaysToCompletion: finiteNumber(entry.final_trading_days) ?? countWeekdaysBetween(entryTriggeredAt, completedAt),
      mfe: finiteNumber(firstPresent(entry.maximum_favorable_excursion, entry.mfe)),
      mae: finiteNumber(firstPresent(entry.maximum_adverse_excursion, entry.mae)),
      volatilityBucket: firstPresent(entry.volatility_bucket, entry.volatilityBucket, 'GENERAL'),
      source: ['SCANNER_REPLAY', 'LIVE_TRACKING', 'MANUAL_IMPORT'].includes(upper(source)) ? upper(source) : 'JOURNAL_IMPORT',
    };
  }

  function classifyRecord(fields) {
    const reasons = [];
    if (!fields.ticker) reasons.push('MISSING_TICKER');
    if (!['LONG', 'SHORT'].includes(fields.direction)) reasons.push('MISSING_DIRECTION');
    if (fields.plannedEntry === null) reasons.push('MISSING_ENTRY');
    if (fields.tp1 === null) reasons.push('MISSING_TP1');
    if (!fields.entryTriggeredAt) reasons.push('MISSING_ENTRY_TRIGGER');
    if (!fields.completedAt) reasons.push('MISSING_COMPLETION');
    if (fields.outcomeReason === 'target' && (fields.tradingDaysToTp1 === null || fields.tradingDaysToTp1 < 0)) reasons.push('INVALID_TP1_DURATION');
    if (fields.tradingDaysToCompletion !== null && fields.tradingDaysToCompletion < 0) reasons.push('NEGATIVE_COMPLETION_DURATION');
    if (reasons.some(reason => ['MISSING_TICKER', 'MISSING_DIRECTION', 'MISSING_ENTRY', 'MISSING_TP1'].includes(reason))) return { quality: 'EXCLUDED', reasons };
    if (reasons.length) return { quality: 'PARTIAL', reasons };
    return { quality: 'QUALIFIED', reasons };
  }

  function normalizeJournalEntry(entry = {}) {
    const fields = normalizedFields(entry);
    const classification = classifyRecord(fields);
    const setupId = firstPresent(entry.setup_id, entry.setupId, null);
    const recordId = firstPresent(entry.opportunity_record_id, entry.recordId, recordFingerprint([
      setupId,
      fields.ticker,
      fields.direction,
      fields.timeframe,
      fields.setupCreatedAt,
      fields.plannedEntry,
      fields.stop,
      fields.tp1,
      fields.entryTriggeredAt,
      fields.completedAt,
    ]));
    return {
      recordId,
      setupId,
      ticker: fields.ticker,
      direction: fields.direction,
      grade: fields.grade,
      timeframe: fields.timeframe,
      setupFamily: fields.setupFamily,
      volatilityBucket: fields.volatilityBucket,
      setupCreatedAt: fields.setupCreatedAt,
      entryTriggeredAt: fields.entryTriggeredAt,
      enteredAt: fields.enteredAt,
      completedAt: fields.completedAt,
      plannedEntry: fields.plannedEntry,
      actualEntry: fields.actualEntry,
      stop: fields.stop,
      tp1: fields.tp1,
      tp2: fields.tp2,
      tp3: fields.tp3,
      tp1Hit: fields.outcomeReason === 'target',
      tp2Hit: Boolean(entry.tp2_reached_at),
      tp3Hit: Boolean(entry.tp3_reached_at),
      stopped: fields.outcomeReason === 'stop',
      invalidated: fields.outcomeReason === 'invalidated' || upper(entry.completion_reason).includes('INVALID'),
      expiredUnresolved: fields.outcomeReason === 'tracking_window_expired' || upper(entry.completion_reason).includes('EXPIRED'),
      tradingDaysToEntry: fields.tradingDaysToEntry,
      tradingDaysToTp1: fields.tradingDaysToTp1,
      tradingDaysToCompletion: fields.tradingDaysToCompletion,
      mfe: fields.mfe,
      mae: fields.mae,
      source: fields.source,
      dataQuality: classification.quality,
      exclusionReasons: classification.reasons,
      analyticsVersion: ANALYTICS_VERSION,
    };
  }

  function previewJournalImport(entries = []) {
    const records = entries.map(normalizeJournalEntry);
    return {
      total: entries.length,
      qualified: records.filter(record => record.dataQuality === 'QUALIFIED').length,
      partial: records.filter(record => record.dataQuality === 'PARTIAL').length,
      excluded: records.filter(record => record.dataQuality === 'EXCLUDED').length,
      records,
    };
  }

  function richerRecord(existing, incoming) {
    const existingRank = QUALITY_RANK[existing.dataQuality] || 0;
    const incomingRank = QUALITY_RANK[incoming.dataQuality] || 0;
    if (incomingRank !== existingRank) return incomingRank > existingRank ? incoming : existing;
    const existingFields = Object.values(existing).filter(value => value !== null && value !== undefined && value !== '').length;
    const incomingFields = Object.values(incoming).filter(value => value !== null && value !== undefined && value !== '').length;
    return incomingFields > existingFields ? incoming : existing;
  }

  function mergeOpportunityRecords(existingRecords = [], incomingRecords = []) {
    const byId = new Map();
    existingRecords.forEach(record => byId.set(record.recordId, record));
    const stats = { imported: 0, skipped: 0, duplicates: 0, partial: 0, invalid: 0 };
    incomingRecords.forEach(record => {
      if (record.dataQuality === 'PARTIAL') stats.partial += 1;
      if (record.dataQuality === 'EXCLUDED') stats.invalid += 1;
      if (record.dataQuality === 'EXCLUDED') {
        stats.skipped += 1;
        return;
      }
      const existing = byId.get(record.recordId);
      if (!existing) {
        byId.set(record.recordId, record);
        stats.imported += 1;
        return;
      }
      stats.duplicates += 1;
      const chosen = richerRecord(existing, record);
      if (chosen === existing) stats.skipped += 1;
      else {
        byId.set(record.recordId, chosen);
        stats.imported += 1;
      }
    });
    return { records: Array.from(byId.values()), ...stats };
  }

  function cohortKey(record = {}, level = 'broader_validated') {
    const ticker = upper(record.ticker) || 'UNKNOWN';
    const direction = normalizeDirection(record.direction);
    const family = normalizeSetupFamily(record.setupFamily);
    const grade = normalizeGrade(record.grade);
    const volatility = upper(record.volatilityBucket) || 'GENERAL';
    if (level === 'ticker_direction_family_grade') return `TICKER|${ticker}|${direction}|${family}|${grade}`;
    if (level === 'ticker_direction_family') return `TICKER|${ticker}|${direction}|${family}`;
    if (level === 'direction_family_volatility') return `COHORT|${direction}|${family}|${volatility}`;
    return `BROADER|${direction}|${family}|${grade}`;
  }

  function cohortLevelLabel(level) {
    return {
      ticker_direction_family_grade: 'Exact ticker + direction + setup family + grade',
      ticker_direction_family: 'Ticker + direction + setup family',
      direction_family_volatility: 'Direction + setup family + volatility cohort',
      broader_validated: 'Broader validated scanner cohort',
    }[level] || level;
  }

  function buildSummary(records, key, level, now) {
    const qualified = records.filter(record => record.dataQuality === 'QUALIFIED');
    const targetRecords = qualified.filter(record => record.tp1Hit && record.tradingDaysToTp1 !== null && record.tradingDaysToTp1 >= 0);
    const durations = targetRecords.map(record => record.tradingDaysToTp1);
    return {
      cohortKey: key,
      cohortLevel: level,
      cohortLevelLabel: cohortLevelLabel(level),
      qualifiedSamples: qualified.length,
      winsToTp1: targetRecords.length,
      stoppedBeforeTp1: qualified.filter(record => record.stopped && !record.tp1Hit).length,
      invalidatedBeforeTp1: qualified.filter(record => record.invalidated && !record.tp1Hit).length,
      medianTradingDaysToTp1: median(durations),
      p25TradingDaysToTp1: percentile(durations, 0.25),
      p75TradingDaysToTp1: percentile(durations, 0.75),
      medianMoveToTp1: median(targetRecords.map(record => Math.abs(record.tp1 - record.plannedEntry))),
      medianMfe: median(qualified.map(record => record.mfe).filter(value => value !== null)),
      medianMae: median(qualified.map(record => record.mae).filter(value => value !== null)),
      readiness: targetRecords.length >= CONFIG.minimumQualifiedSamples ? 'QUALIFIED' : qualified.length > 0 ? 'FALLBACK_ONLY' : 'INSUFFICIENT',
      updatedAt: now,
      analyticsVersion: ANALYTICS_VERSION,
    };
  }

  function buildCohortSummaries(records = [], options = {}) {
    const now = options.now || new Date().toISOString();
    const summaries = [];
    COHORT_LEVELS.forEach(level => {
      const buckets = new Map();
      records.forEach(record => {
        if (record.dataQuality === 'EXCLUDED') return;
        const key = cohortKey(record, level);
        if (!buckets.has(key)) buckets.set(key, []);
        buckets.get(key).push(record);
      });
      buckets.forEach((bucketRecords, key) => summaries.push(buildSummary(bucketRecords, key, level, now)));
    });
    return summaries.sort((a, b) => COHORT_LEVELS.indexOf(a.cohortLevel) - COHORT_LEVELS.indexOf(b.cohortLevel) || b.qualifiedSamples - a.qualifiedSamples);
  }

  function setupToRecordLike(setup = {}) {
    const fields = normalizedFields(setup);
    return {
      ticker: fields.ticker,
      direction: fields.direction,
      grade: fields.grade,
      timeframe: fields.timeframe,
      setupFamily: fields.setupFamily,
      volatilityBucket: fields.volatilityBucket,
    };
  }

  function selectCohortForSetup(setup = {}, summaries = [], options = {}) {
    const allowFallback = options.allowFallback !== false;
    const setupRecord = setupToRecordLike(setup);
    const requestedKeys = COHORT_LEVELS.map(level => ({ level, key: cohortKey(setupRecord, level) }));
    const exact = summaries.find(summary => summary.cohortKey === requestedKeys[0].key);
    for (const requested of requestedKeys) {
      if (requested.level !== 'ticker_direction_family_grade' && !allowFallback) break;
      const summary = summaries.find(item => item.cohortKey === requested.key);
      if (summary && summary.readiness === 'QUALIFIED') {
        return {
          summary,
          fallbackUsed: requested.level !== 'ticker_direction_family_grade',
          fallbackLevel: requested.level,
          exactGroupUsed: requested.level === 'ticker_direction_family_grade',
          requestedKey: requestedKeys[0].key,
        };
      }
    }
    return {
      summary: exact || null,
      fallbackUsed: false,
      fallbackLevel: null,
      exactGroupUsed: false,
      requestedKey: requestedKeys[0].key,
    };
  }

  function lifecycleBand(remaining) {
    if (remaining === null || remaining === undefined) return 'INSUFFICIENT';
    if (remaining <= 0) return 'COMPLETED';
    if (remaining >= 75) return 'EARLY';
    if (remaining >= 50) return 'ON_TIME';
    if (remaining >= 30) return 'MID_MOVE';
    return 'LATE';
  }

  function lifecycleReplay(record = {}) {
    if (!record || record.dataQuality !== 'QUALIFIED') return [];
    const target = record.tp1Hit;
    const stopped = record.stopped || record.invalidated;
    const base = target
      ? [
          ['Entry trigger', 91],
          ['25% elapsed', 73],
          ['50% elapsed', 47],
          ['75% elapsed', 22],
          ['Immediately before TP1', 7],
          ['TP1 reached', 0],
        ]
      : stopped
        ? [
            ['Entry trigger', 91],
            ['25% elapsed', 78],
            ['50% elapsed', 70],
            ['Immediately before stop/invalidation', null],
          ]
        : [
            ['Entry trigger', 91],
            ['25% elapsed', 80],
            ['50% elapsed', 64],
            ['Tracking window checkpoint', 52],
          ];
    return base.map(([label, remaining], index) => ({
      recordId: record.recordId,
      ticker: record.ticker,
      label,
      sequence: index,
      opportunityRemainingPct: remaining,
      lifecycleBand: lifecycleBand(remaining),
      source: 'JOURNAL_IMPORT_APPROXIMATION',
      note: 'Snapshot replay is approximated from persisted outcome milestones until candle-level replay data is available.',
    }));
  }

  function stabilityDiagnostics(snapshots = []) {
    const values = snapshots.map(item => finiteNumber(item.opportunityRemainingPct)).filter(value => value !== null);
    if (values.length < 2) return { largestSingleChange: 0, averageScanToScanChange: 0, lifecycleBandReversals: 0, unstable: false, warnings: [] };
    const changes = [];
    let reversals = 0;
    for (let i = 1; i < values.length; i += 1) {
      const delta = values[i] - values[i - 1];
      changes.push(Math.abs(delta));
      if (delta > 0) reversals += 1;
    }
    const largestSingleChange = Math.max(...changes);
    const averageScanToScanChange = changes.reduce((sum, value) => sum + value, 0) / changes.length;
    const warnings = [];
    if (largestSingleChange > CONFIG.stability.largeSingleChangePct) warnings.push('LARGE_SINGLE_STEP_CHANGE');
    if (reversals > CONFIG.stability.maxBandReversals) warnings.push('FREQUENT_BAND_REVERSALS');
    return {
      largestSingleChange,
      averageScanToScanChange,
      lifecycleBandReversals: reversals,
      unstable: warnings.length > 0,
      warnings,
    };
  }

  function sourceFieldReport() {
    return [
      ['Scanner setup snapshots', 'scannerRows/scannerNearMiss in browser memory plus card setup payloads', 'browser memory', 'No', 'Yes', 'Not durable server-side'],
      ['Setup ID', 'setup_id or deterministic setupIdFromSetup/setupIdFromJournalEntry', 'localStorage journal', 'No', 'Yes', 'Available'],
      ['Ticker', 'scanner row and journal ticker', 'scanner memory/localStorage', 'No', 'Yes', 'Available'],
      ['Direction', 'scanner row and journal direction', 'scanner memory/localStorage', 'No', 'Yes', 'Available'],
      ['Grade', 'setup_grade/setupGrade/scanner_status', 'scanner memory/localStorage', 'No', 'Yes', 'Available'],
      ['Setup family/lifecycle type', 'entryType/setupStatus/phase/setup_family when present', 'scanner memory/localStorage', 'No', 'Yes', 'Partially available'],
      ['Planned entry', 'entry_price/entry', 'scanner memory/localStorage', 'No', 'Yes', 'Available'],
      ['Stop', 'stop_price/plannedStop/stop', 'scanner memory/localStorage', 'No', 'Yes', 'Available'],
      ['TP1/TP2/TP3', 'target_price/tp1/tp2/tp3/plannedTp fields', 'scanner memory/localStorage', 'No', 'Yes', 'TP1 available; TP2/TP3 partial'],
      ['Entry-trigger time', 'first_entry_touch_at/entry_reached_at', 'localStorage journal', 'No', 'Yes', 'Available after tracking'],
      ['Target-hit time', 'first_target_touch_at/tp1_reached_at', 'localStorage journal', 'No', 'Yes', 'Available after tracking'],
      ['Stop or invalidation time', 'first_stop_touch_at/tracking_completed_at/completion_reason', 'localStorage journal', 'No', 'Yes', 'Partially available'],
      ['MFE/MAE', 'maximum_favorable_excursion/maximum_adverse_excursion', 'localStorage journal', 'No', 'Yes', 'Available after tracking'],
      ['Replay dataset', 'none found as durable server-side candle replay store', 'missing', 'No', 'No', 'Missing'],
    ].map(([field, source, persistence, serverSide, browserSide, status]) => ({ field, source, persistence, serverSide, browserSide, status }));
  }

  function diagnostics(records = [], summaries = []) {
    const qualified = records.filter(record => record.dataQuality === 'QUALIFIED');
    const replays = qualified.slice(0, 20).map(record => ({ record, snapshots: lifecycleReplay(record) }));
    const stability = replays.map(item => ({ recordId: item.record.recordId, ticker: item.record.ticker, ...stabilityDiagnostics(item.snapshots) }));
    return {
      counts: {
        totalRecords: records.length,
        qualified: qualified.length,
        partial: records.filter(record => record.dataQuality === 'PARTIAL').length,
        excluded: records.filter(record => record.dataQuality === 'EXCLUDED').length,
        completed: qualified.filter(record => record.completedAt).length,
        invalidated: qualified.filter(record => record.invalidated || record.stopped).length,
        fallbackOnlyCohorts: summaries.filter(summary => summary.readiness === 'FALLBACK_ONLY').length,
        qualifiedCohorts: summaries.filter(summary => summary.readiness === 'QUALIFIED').length,
        insufficientCohorts: summaries.filter(summary => summary.readiness === 'INSUFFICIENT').length,
        unstableCalculations: stability.filter(item => item.unstable).length,
      },
      stability,
      sampleReplays: replays,
      fieldReport: sourceFieldReport(),
    };
  }

  return {
    ANALYTICS_VERSION,
    CONFIG,
    COHORT_LEVELS,
    finiteNumber,
    countWeekdaysBetween,
    normalizeJournalEntry,
    previewJournalImport,
    mergeOpportunityRecords,
    cohortKey,
    buildCohortSummaries,
    selectCohortForSetup,
    lifecycleReplay,
    stabilityDiagnostics,
    sourceFieldReport,
    diagnostics,
  };
});
