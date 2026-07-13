(function(root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.KairosExpectedMove = factory();
  }
})(typeof self !== 'undefined' ? self : this, function() {
  const CONFIG = {
    minimumTargetSamples: 30,
    confidenceThresholds: [
      { min: 300, status: 'ready', confidence: 'high' },
      { min: 100, status: 'ready', confidence: 'moderate' },
      { min: 30, status: 'early_estimate', confidence: 'low' },
      { min: 0, status: 'learning', confidence: 'insufficient' },
    ],
    downgrade: {
      iqrToMedianRatio: 1.5,
      targetCompletionRate: 0.35,
      exclusionRate: 0.25,
    },
    expiration: {
      tradingToCalendarRatio: 7 / 5,
      upperBoundMultiplier: 2,
      safetyBufferCalendarDays: 7,
      minimumDte: 21,
      rangeWidth: 21,
    },
  };

  const READINESS_CONFIG = {
    minimumTargetSamples: 30,
    maximumExclusionRate: 0.20,
    minimumTargetCompletionRate: 0.35,
    maximumIqrToMedianRatio: 1.25,
    maximumStddevToMedianRatio: 1.50,
    minimumConfidence: 'moderate',
  };

  const READINESS_FLAGS = {
    INSUFFICIENT_SAMPLES: 'INSUFFICIENT_SAMPLES',
    FALLBACK_USED: 'FALLBACK_USED',
    LOW_CONFIDENCE: 'LOW_CONFIDENCE',
    HIGH_EXCLUSION_RATE: 'HIGH_EXCLUSION_RATE',
    LOW_TARGET_COMPLETION_RATE: 'LOW_TARGET_COMPLETION_RATE',
    BROAD_IQR: 'BROAD_IQR',
    HIGH_VARIABILITY: 'HIGH_VARIABILITY',
    INVALID_EXPECTED_MOVE: 'INVALID_EXPECTED_MOVE',
    UNSAFE_EXPIRATION: 'UNSAFE_EXPIRATION',
    DATA_INCONSISTENCY: 'DATA_INCONSISTENCY',
  };

  function finiteNumber(value) {
    if (value === null || value === undefined || value === '') return null;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function normalizeText(value) {
    return String(value || '').trim().toUpperCase();
  }

  function normalizeGrade(entry = {}) {
    const raw = [
      entry.setup_grade,
      entry.setupGrade,
      entry.scanner_status,
      entry.scannerStatus,
      entry.entryStatus,
    ].map(normalizeText).filter(Boolean).join(' ');
    if (/\bB\+?\b/.test(raw) || raw.includes('TRADEABLE')) return 'B_PLUS_TRADEABLE';
    if (/\bA\+?\b/.test(raw) || raw.includes('READY')) return 'A_PLUS_READY';
    return 'UNKNOWN';
  }

  function normalizeDirection(value) {
    const raw = normalizeText(value);
    if (raw.includes('SHORT') || raw === 'PUT') return 'SHORT';
    if (raw.includes('LONG') || raw === 'CALL') return 'LONG';
    return 'UNKNOWN';
  }

  function normalizeTimeframe(value) {
    const raw = String(value || '').trim();
    if (!raw) return 'UNKNOWN';
    const upper = raw.toUpperCase();
    if (['1D', 'D', 'DAY', 'DAILY'].includes(upper)) return 'Daily';
    if (['1W', 'W', 'WEEK', 'WEEKLY'].includes(upper)) return 'Weekly';
    return upper;
  }

  function normalizeCriteria(input = {}) {
    const rawGrade = normalizeText(input.grade);
    return {
      grade: rawGrade.includes('_') ? rawGrade : normalizeGrade({ ...input, setup_grade: input.grade || input.setup_grade || input.setupGrade }),
      direction: input.direction ? normalizeDirection(input.direction) : normalizeDirection(input.direction || input.option_type),
      timeframe: normalizeTimeframe(input.timeframe || input.scanner_timeframe || input.setupTf),
    };
  }

  function percentile(values, p) {
    const nums = values.map(Number).filter(Number.isFinite).sort((a, b) => a - b);
    if (!nums.length) return null;
    if (nums.length === 1) return nums[0];
    const index = (nums.length - 1) * p;
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    if (lower === upper) return nums[lower];
    const weight = index - lower;
    return nums[lower] + (nums[upper] - nums[lower]) * weight;
  }

  function median(values) {
    return percentile(values, 0.5);
  }

  function mean(values) {
    const nums = values.map(Number).filter(Number.isFinite);
    return nums.length ? nums.reduce((sum, value) => sum + value, 0) / nums.length : null;
  }

  function standardDeviation(values) {
    const nums = values.map(Number).filter(Number.isFinite);
    if (nums.length < 2) return 0;
    const avg = mean(nums);
    const variance = nums.reduce((sum, value) => sum + Math.pow(value - avg, 2), 0) / nums.length;
    return Math.sqrt(variance);
  }

  function medianAbsoluteDeviation(values) {
    const nums = values.map(Number).filter(Number.isFinite);
    const med = median(nums);
    if (med === null) return null;
    return median(nums.map(value => Math.abs(value - med)));
  }

  function min(values) {
    const nums = values.map(Number).filter(Number.isFinite);
    return nums.length ? Math.min(...nums) : null;
  }

  function max(values) {
    const nums = values.map(Number).filter(Number.isFinite);
    return nums.length ? Math.max(...nums) : null;
  }

  function round(value, places = 4) {
    const num = finiteNumber(value);
    if (num === null) return null;
    const scale = Math.pow(10, places);
    return Math.round(num * scale) / scale;
  }

  function confidenceRank(confidence) {
    if (confidence === 'high') return 3;
    if (confidence === 'moderate') return 2;
    if (confidence === 'low') return 1;
    return 0;
  }

  function sameGroup(a, b) {
    if (!a || !b) return false;
    return a.grade === b.grade && a.direction === b.direction && a.timeframe === b.timeframe;
  }

  function percent(value) {
    const num = finiteNumber(value);
    return num === null ? null : Math.round(num * 100);
  }

  function isCompleted(entry = {}) {
    return entry.tracking_status === 'completed'
      && ['target', 'stop', 'manually_closed', 'tracking_window_expired'].includes(entry.completion_reason);
  }

  function exclusionReason(entry = {}) {
    if (!isCompleted(entry)) return 'not_completed_or_invalid_completion';
    if (entry.completion_reason !== 'target') return 'non_target_completion';
    const duration = finiteNumber(entry.trading_days_to_target);
    if (duration === null) return 'missing_duration';
    if (duration < 0) return 'negative_duration';
    if (finiteNumber(entry.entry_price) === null && finiteNumber(entry.entry) === null) return 'invalid_entry';
    if (finiteNumber(entry.target_price) === null && finiteNumber(entry.plannedTp1) === null) return 'invalid_target';
    if (!entry.first_target_touch_at && entry.completion_reason !== 'target') return 'target_not_reached';
    return null;
  }

  function criteriaMatches(entryCriteria, requested, level) {
    if (entryCriteria.grade !== requested.grade) return false;
    if (level === 'grade') return true;
    if (level === 'grade_timeframe') return entryCriteria.timeframe === requested.timeframe;
    if (level === 'grade_direction') return entryCriteria.direction === requested.direction;
    return entryCriteria.direction === requested.direction && entryCriteria.timeframe === requested.timeframe;
  }

  function groupUsedForLevel(requested, level) {
    if (level === 'grade') return { grade: requested.grade, direction: null, timeframe: null };
    if (level === 'grade_timeframe') return { grade: requested.grade, direction: null, timeframe: requested.timeframe };
    if (level === 'grade_direction') return { grade: requested.grade, direction: requested.direction, timeframe: null };
    return { grade: requested.grade, direction: requested.direction, timeframe: requested.timeframe };
  }

  function confidenceFor(targetCount) {
    return CONFIG.confidenceThresholds.find(row => targetCount >= row.min);
  }

  function downgradeConfidence(confidence) {
    if (confidence === 'high') return 'moderate';
    if (confidence === 'moderate') return 'low';
    if (confidence === 'low') return 'insufficient';
    return confidence;
  }

  function statusForConfidence(confidence, targetCount) {
    if (targetCount < CONFIG.minimumTargetSamples || confidence === 'insufficient') return 'learning';
    return targetCount >= 100 ? 'ready' : 'early_estimate';
  }

  function buildGroupStats(entries, requested, level = 'exact') {
    const groupEntries = entries.filter(entry => criteriaMatches(normalizeCriteria(entry), requested, level) && isCompleted(entry));
    const targetEntries = groupEntries.filter(entry => entry.completion_reason === 'target');
    const validTargetEntries = [];
    const exclusions = {};

    targetEntries.forEach(entry => {
      const reason = exclusionReason(entry);
      if (reason) exclusions[reason] = (exclusions[reason] || 0) + 1;
      else validTargetEntries.push(entry);
    });

    const durations = validTargetEntries.map(entry => finiteNumber(entry.trading_days_to_target));
    const p25 = percentile(durations, 0.25);
    const p75 = percentile(durations, 0.75);
    const med = median(durations);
    const iqr = p25 !== null && p75 !== null ? p75 - p25 : null;
    const targetCount = validTargetEntries.length;
    const completedCount = groupEntries.length;
    let base = confidenceFor(targetCount);
    let confidence = base.confidence;
    const exclusionCount = Object.values(exclusions).reduce((sum, count) => sum + count, 0);
    const targetRate = completedCount ? targetCount / completedCount : 0;
    const exclusionRate = targetEntries.length ? exclusionCount / targetEntries.length : 0;

    if (targetCount >= CONFIG.minimumTargetSamples) {
      if (med && iqr !== null && iqr / med > CONFIG.downgrade.iqrToMedianRatio) confidence = downgradeConfidence(confidence);
      if (targetRate < CONFIG.downgrade.targetCompletionRate) confidence = downgradeConfidence(confidence);
      if (exclusionRate > CONFIG.downgrade.exclusionRate) confidence = downgradeConfidence(confidence);
    }

    const qualified = targetCount >= CONFIG.minimumTargetSamples && confidence !== 'insufficient';
    return {
      level,
      group_used: groupUsedForLevel(requested, level),
      sample_count: completedCount,
      target_sample_count: targetCount,
      stopped_sample_count: groupEntries.filter(entry => entry.completion_reason === 'stop').length,
      manual_close_count: groupEntries.filter(entry => entry.completion_reason === 'manually_closed').length,
      expired_count: groupEntries.filter(entry => entry.completion_reason === 'tracking_window_expired').length,
      exclusion_count: exclusionCount,
      exclusion_reasons: exclusions,
      target_completion_rate: completedCount ? targetCount / completedCount : null,
      stop_completion_rate: completedCount ? groupEntries.filter(entry => entry.completion_reason === 'stop').length / completedCount : null,
      manual_close_rate: completedCount ? groupEntries.filter(entry => entry.completion_reason === 'manually_closed').length / completedCount : null,
      expiry_rate: completedCount ? groupEntries.filter(entry => entry.completion_reason === 'tracking_window_expired').length / completedCount : null,
      median_days_to_target: med,
      mean_days_to_target: mean(durations),
      min_days_to_target: min(durations),
      max_days_to_target: max(durations),
      p10_days_to_target: percentile(durations, 0.10),
      p25_days_to_target: p25,
      p75_days_to_target: p75,
      p90_days_to_target: percentile(durations, 0.90),
      standard_deviation_days_to_target: standardDeviation(durations),
      median_absolute_deviation_days_to_target: medianAbsoluteDeviation(durations),
      iqr_days_to_target: iqr,
      iqr_to_median_ratio: med ? iqr / med : null,
      stddev_to_median_ratio: med ? standardDeviation(durations) / med : null,
      target_durations: durations.slice().sort((a, b) => a - b),
      median_mfe_r: median(groupEntries.map(entry => entry.maximum_favorable_excursion_r)),
      median_mae_r: median(groupEntries.map(entry => entry.maximum_adverse_excursion_r)),
      median_target_distance_atr: median(groupEntries.map(entry => entry.target_distance_atr)),
      median_days_to_stop: median(groupEntries.filter(entry => entry.completion_reason === 'stop').map(entry => entry.trading_days_to_stop)),
      median_days_to_entry: median(groupEntries.map(entry => entry.trading_days_to_entry)),
      median_bars_to_target: median(validTargetEntries.map(entry => entry.bars_to_target)),
      median_bars_to_stop: median(groupEntries.filter(entry => entry.completion_reason === 'stop').map(entry => entry.bars_to_stop)),
      expected_move_min_days: qualified && p25 !== null ? Math.max(1, Math.floor(p25)) : null,
      expected_move_max_days: qualified && p75 !== null ? Math.max(1, Math.ceil(p75)) : null,
      status: statusForConfidence(confidence, targetCount),
      confidence,
      qualified,
    };
  }

  function expectedMoveAnalytics(entries = [], criteria = {}) {
    const requested = normalizeCriteria(criteria);
    const levels = [
      { key: 'exact', label: 'grade_direction_timeframe' },
      { key: 'grade_direction', label: 'grade_direction' },
      { key: 'grade_timeframe', label: 'grade_timeframe' },
      { key: 'grade', label: 'grade' },
    ];
    let exactStats = null;
    for (const level of levels) {
      const stats = buildGroupStats(entries, requested, level.key);
      if (level.key === 'exact') exactStats = stats;
      if (stats.qualified) {
        return {
          status: stats.status,
          confidence: stats.confidence,
          sample_count: stats.sample_count,
          target_sample_count: stats.target_sample_count,
          expected_move_min_days: stats.expected_move_min_days,
          expected_move_max_days: stats.expected_move_max_days,
          median_days_to_target: stats.median_days_to_target,
          p25_days_to_target: stats.p25_days_to_target,
          p75_days_to_target: stats.p75_days_to_target,
          group_requested: requested,
          group_used: stats.group_used,
          fallback_used: level.key !== 'exact',
          fallback_level: level.key === 'exact' ? null : level.label,
          exclusion_count: stats.exclusion_count,
          stats,
        };
      }
    }
    return {
      status: 'learning',
      confidence: 'insufficient',
      sample_count: exactStats ? exactStats.sample_count : 0,
      target_sample_count: exactStats ? exactStats.target_sample_count : 0,
      expected_move_min_days: null,
      expected_move_max_days: null,
      median_days_to_target: null,
      p25_days_to_target: null,
      p75_days_to_target: null,
      group_requested: requested,
      group_used: null,
      fallback_used: false,
      fallback_level: null,
      exclusion_count: exactStats ? exactStats.exclusion_count : 0,
      stats: exactStats,
    };
  }

  function uniquePrimaryCriteria(entries = []) {
    const seen = new Map();
    entries.forEach(entry => {
      const criteria = normalizeCriteria(entry);
      if (criteria.grade === 'UNKNOWN' || criteria.direction === 'UNKNOWN' || criteria.timeframe === 'UNKNOWN') return;
      const key = [criteria.grade, criteria.direction, criteria.timeframe].join('|');
      seen.set(key, criteria);
    });
    return Array.from(seen.values()).sort((a, b) =>
      a.grade.localeCompare(b.grade) || a.direction.localeCompare(b.direction) || a.timeframe.localeCompare(b.timeframe)
    );
  }

  function buildGroupDiagnostics(entries = []) {
    return uniquePrimaryCriteria(entries)
      .map(criteria => {
        const result = expectedMoveAnalytics(entries, criteria);
        const exact = buildGroupStats(entries, criteria, 'exact');
        return {
          key: `${criteria.grade} | ${criteria.direction} | ${criteria.timeframe}`,
          criteria,
          result,
          ...exact,
        };
      })
      .sort((a, b) => b.target_sample_count - a.target_sample_count || a.key.localeCompare(b.key));
  }

  function suggestedExpirationAnalytics(expectedMoveResult) {
    if (!expectedMoveResult || expectedMoveResult.status === 'learning' || expectedMoveResult.confidence === 'insufficient') {
      return { status: 'learning', min_dte: null, max_dte: null };
    }
    const upper = finiteNumber(expectedMoveResult.expected_move_max_days);
    if (upper === null) return { status: 'learning', min_dte: null, max_dte: null };
    const rawMinimumDte = (
      upper * CONFIG.expiration.upperBoundMultiplier * CONFIG.expiration.tradingToCalendarRatio
      + CONFIG.expiration.safetyBufferCalendarDays
    );
    const roundedMinimumDte = Math.ceil(rawMinimumDte);
    const minDte = Math.max(CONFIG.expiration.minimumDte, roundedMinimumDte, upper);
    const maxDte = minDte + CONFIG.expiration.rangeWidth;
    return {
      status: 'internal_only',
      min_dte: minDte,
      max_dte: maxDte,
      policy: {
        upper_expected_trading_days: upper,
        trading_day_to_calendar_day_conversion: CONFIG.expiration.tradingToCalendarRatio,
        safety_multiplier: CONFIG.expiration.upperBoundMultiplier,
        minimum_calendar_dte_formula: 'ceil(upper * 2 * 7/5 + safety_buffer)',
        safety_buffer_calendar_days: CONFIG.expiration.safetyBufferCalendarDays,
        raw_minimum_dte: round(rawMinimumDte, 2),
        rounded_minimum_dte: roundedMinimumDte,
        recommended_maximum_dte: maxDte,
      },
    };
  }

  function durationDistributionBuckets(values = []) {
    const buckets = [
      { label: '1-3', min: 1, max: 3, count: 0 },
      { label: '4-5', min: 4, max: 5, count: 0 },
      { label: '6-8', min: 6, max: 8, count: 0 },
      { label: '9-12', min: 9, max: 12, count: 0 },
      { label: '13-20', min: 13, max: 20, count: 0 },
      { label: '21+', min: 21, max: Infinity, count: 0 },
    ];
    values.map(Number).filter(Number.isFinite).forEach(value => {
      const bucket = buckets.find(row => value >= row.min && value <= row.max);
      if (bucket) bucket.count += 1;
    });
    return buckets.map(({ label, count }) => ({ label, count }));
  }

  function expectedMoveReleaseReadiness(expectedMoveResult = {}, expirationResult = {}) {
    const stats = expectedMoveResult.stats || {};
    const targetCount = finiteNumber(expectedMoveResult.target_sample_count) || 0;
    const minimumRequired = READINESS_CONFIG.minimumTargetSamples;
    const reasons = [];
    const warnings = [];
    const flags = [];
    const criticalFlags = new Set([
      READINESS_FLAGS.INVALID_EXPECTED_MOVE,
      READINESS_FLAGS.UNSAFE_EXPIRATION,
      READINESS_FLAGS.DATA_INCONSISTENCY,
    ]);

    const expectedMin = finiteNumber(expectedMoveResult.expected_move_min_days);
    const expectedMax = finiteNumber(expectedMoveResult.expected_move_max_days);
    const expirationMin = finiteNumber(expirationResult.min_dte);
    const expirationMax = finiteNumber(expirationResult.max_dte);
    const groupRequested = expectedMoveResult.group_requested || null;
    const groupUsed = expectedMoveResult.group_used || null;
    const fallbackUsed = Boolean(expectedMoveResult.fallback_used);
    const exactGroupUsed = !fallbackUsed && sameGroup(groupRequested, groupUsed);

    if (!expectedMoveResult || !groupRequested || !stats) {
      flags.push(READINESS_FLAGS.DATA_INCONSISTENCY);
      reasons.push('analytics result inconsistent');
    }

    if (targetCount < minimumRequired) {
      flags.push(READINESS_FLAGS.INSUFFICIENT_SAMPLES);
      reasons.push(`${targetCount} of ${minimumRequired} required target completions`);
    }

    if (fallbackUsed) {
      flags.push(READINESS_FLAGS.FALLBACK_USED);
      reasons.push('fallback group required');
    }

    if (confidenceRank(expectedMoveResult.confidence) < confidenceRank(READINESS_CONFIG.minimumConfidence)) {
      flags.push(READINESS_FLAGS.LOW_CONFIDENCE);
      reasons.push(`confidence below ${READINESS_CONFIG.minimumConfidence}`);
    }

    const exclusionRate = stats.exclusion_count || stats.exclusion_count === 0
      ? (stats.exclusion_count / Math.max(1, (stats.target_sample_count || 0) + stats.exclusion_count))
      : null;
    if (exclusionRate !== null && exclusionRate > READINESS_CONFIG.maximumExclusionRate) {
      flags.push(READINESS_FLAGS.HIGH_EXCLUSION_RATE);
      reasons.push(`exclusion rate ${percent(exclusionRate)}% exceeds ${percent(READINESS_CONFIG.maximumExclusionRate)}%`);
    }

    const targetRate = finiteNumber(stats.target_completion_rate);
    if (targetRate !== null && targetRate < READINESS_CONFIG.minimumTargetCompletionRate) {
      flags.push(READINESS_FLAGS.LOW_TARGET_COMPLETION_RATE);
      reasons.push(`target completion rate ${percent(targetRate)}% is below ${percent(READINESS_CONFIG.minimumTargetCompletionRate)}%`);
    }

    const iqrRatio = finiteNumber(stats.iqr_to_median_ratio);
    if (iqrRatio !== null && iqrRatio > READINESS_CONFIG.maximumIqrToMedianRatio) {
      flags.push(READINESS_FLAGS.BROAD_IQR);
      reasons.push('duration IQR is too broad');
    }

    const stddevRatio = finiteNumber(stats.stddev_to_median_ratio);
    if (stddevRatio !== null && stddevRatio > READINESS_CONFIG.maximumStddevToMedianRatio) {
      flags.push(READINESS_FLAGS.HIGH_VARIABILITY);
      reasons.push('duration standard deviation is unstable');
    }

    const expectedBoundsValid = expectedMin !== null && expectedMax !== null && expectedMin >= 1 && expectedMax >= expectedMin;
    if (!expectedBoundsValid && targetCount >= minimumRequired) {
      flags.push(READINESS_FLAGS.INVALID_EXPECTED_MOVE);
      reasons.push('invalid Expected Move bounds');
    }

    const expirationBoundsValid = expirationMin !== null && expirationMax !== null && expirationMin >= 1 && expirationMax >= expirationMin;
    if (expectedBoundsValid && (!expirationBoundsValid || expirationMin < expectedMax)) {
      flags.push(READINESS_FLAGS.UNSAFE_EXPIRATION);
      reasons.push('Suggested Expiration shorter than Expected Move');
    }

    if (targetCount > (stats.sample_count || 0)) {
      flags.push(READINESS_FLAGS.DATA_INCONSISTENCY);
      reasons.push('analytics result inconsistent');
    }

    if (stats.target_durations && stats.target_durations.length && targetCount >= minimumRequired) {
      const buckets = durationDistributionBuckets(stats.target_durations);
      const nonEmptyBuckets = buckets.filter(bucket => bucket.count > 0).length;
      if (nonEmptyBuckets > 4) warnings.push('duration samples are widely distributed');
    }

    const uniqueFlags = Array.from(new Set(flags));
    const hasCritical = uniqueFlags.some(flag => criticalFlags.has(flag));
    let releaseStatus = 'ready_for_release';
    if (hasCritical) releaseStatus = 'blocked';
    else if (targetCount < minimumRequired) releaseStatus = 'learning';
    else if (reasons.length) releaseStatus = 'testing';

    const cardReady = releaseStatus === 'ready_for_release'
      && exactGroupUsed
      && !fallbackUsed
      && targetCount >= minimumRequired
      && expectedMoveResult.status === 'ready'
      && confidenceRank(expectedMoveResult.confidence) >= confidenceRank(READINESS_CONFIG.minimumConfidence)
      && expectedBoundsValid
      && expirationBoundsValid
      && expirationMin >= expectedMax
      && (exclusionRate === null || exclusionRate <= READINESS_CONFIG.maximumExclusionRate)
      && (targetRate === null || targetRate >= READINESS_CONFIG.minimumTargetCompletionRate)
      && (iqrRatio === null || iqrRatio <= READINESS_CONFIG.maximumIqrToMedianRatio)
      && (stddevRatio === null || stddevRatio <= READINESS_CONFIG.maximumStddevToMedianRatio);

    return {
      release_status: releaseStatus,
      card_ready: cardReady,
      reasons: Array.from(new Set(reasons)),
      warnings,
      flags: uniqueFlags,
      group_requested: groupRequested,
      group_used: groupUsed,
      exact_group_used: exactGroupUsed,
      fallback_used: fallbackUsed,
      target_sample_count: targetCount,
      minimum_required: minimumRequired,
      confidence: expectedMoveResult.confidence || 'insufficient',
      expected_move_min_days: expectedBoundsValid ? expectedMin : null,
      expected_move_max_days: expectedBoundsValid ? expectedMax : null,
      median_days_to_target: expectedMoveResult.median_days_to_target ?? null,
      p25_days_to_target: expectedMoveResult.p25_days_to_target ?? null,
      p75_days_to_target: expectedMoveResult.p75_days_to_target ?? null,
      suggested_expiration_min_dte: expirationBoundsValid ? expirationMin : null,
      suggested_expiration_max_dte: expirationBoundsValid ? expirationMax : null,
      target_completion_rate: targetRate,
      exclusion_rate: exclusionRate,
      iqr_to_median_ratio: iqrRatio,
      stddev_to_median_ratio: stddevRatio,
      expiration_audit: expirationResult.policy || null,
    };
  }

  function buildAnalyticsSnapshot(readinessResult = {}, timestamp = new Date().toISOString()) {
    const group = readinessResult.group_requested || {};
    return {
      analytics_snapshot_timestamp: timestamp,
      group_key: [group.grade, group.direction, group.timeframe].filter(Boolean).join('|') || null,
      target_sample_count: readinessResult.target_sample_count || 0,
      median_days: readinessResult.median_days_to_target ?? null,
      p25_days: readinessResult.p25_days_to_target ?? null,
      p75_days: readinessResult.p75_days_to_target ?? null,
      target_completion_rate: readinessResult.target_completion_rate ?? null,
      exclusion_rate: readinessResult.exclusion_rate ?? null,
      confidence: readinessResult.confidence || 'insufficient',
      release_status: readinessResult.release_status || 'learning',
    };
  }

  function buildExpectedMoveCardFields(readinessResult = {}) {
    if (!readinessResult.card_ready) return null;
    const minDays = finiteNumber(readinessResult.expected_move_min_days);
    const maxDays = finiteNumber(readinessResult.expected_move_max_days);
    const minDte = finiteNumber(readinessResult.suggested_expiration_min_dte);
    const maxDte = finiteNumber(readinessResult.suggested_expiration_max_dte);
    if (minDays === null || maxDays === null || minDte === null || maxDte === null) return null;
    return {
      expected_move_label: `${minDays}\u2013${maxDays} trading days`,
      suggested_expiration_label: `${minDte}\u2013${maxDte} DTE`,
    };
  }

  return {
    CONFIG,
    READINESS_CONFIG,
    READINESS_FLAGS,
    normalizeGrade,
    normalizeDirection,
    normalizeTimeframe,
    normalizeCriteria,
    percentile,
    median,
    expectedMoveAnalytics,
    buildGroupDiagnostics,
    suggestedExpirationAnalytics,
    expectedMoveReleaseReadiness,
    durationDistributionBuckets,
    buildAnalyticsSnapshot,
    buildExpectedMoveCardFields,
  };
});
