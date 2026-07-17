(function(root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.KairosFurtherAnalysis = factory();
  }
})(typeof self !== 'undefined' ? self : this, function() {
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

  function isExplicitCGradeSetup(setup = {}) {
    return upper(firstPresent(setup.setupGrade, setup.setup_grade, setup.quality?.grade)) === 'C';
  }

  function setupIdentity(setup = {}) {
    const explicit = firstPresent(setup.setup_id, setup.setupId, setup.id);
    if (explicit) return String(explicit);
    return [
      setup.ticker,
      setup.direction,
      setup.timeframe || setup.scanner_timeframe || setup.setupTf,
      setup.signal_timestamp || setup.candleTime || setup.scannedAt || setup.updated_at,
      setup.entry,
      setup.sl,
      setup.tp1,
    ].map(value => normalizeText(value) || 'NA').join('|');
  }

  function contractType(setup = {}) {
    const explicit = upper(firstPresent(setup.option_type, setup.option?.type, setup.best_contract?.type, setup.suggested_contract?.type));
    if (explicit === 'CALL' || explicit === 'PUT') return explicit;
    const direction = upper(setup.direction);
    if (direction === 'LONG') return 'CALL';
    if (direction === 'SHORT') return 'PUT';
    return '';
  }

  function selectorLabel(setup = {}) {
    return [
      upper(setup.ticker) || 'UNKNOWN',
      contractType(setup) || upper(setup.direction) || 'N/A',
      firstPresent(setup.setupGrade, setup.setup_grade, setup.scanner_status, setup.entryStatus, 'Ungraded'),
      firstPresent(setup.timeframe, setup.scanner_timeframe, setup.setupTf, 'No TF'),
      firstPresent(setup.signal_timestamp, setup.candleTime) ? normalizeText(firstPresent(setup.signal_timestamp, setup.candleTime)).slice(0, 10) : '',
    ].filter(Boolean).join(' · ');
  }

  function setupCriteria(setup = {}) {
    return {
      grade: firstPresent(setup.setupGrade, setup.setup_grade, setup.scanner_status, setup.entryStatus),
      direction: setup.direction,
      timeframe: firstPresent(setup.timeframe, setup.scanner_timeframe, setup.setupTf),
    };
  }

  function directionWord(setup = {}) {
    const direction = upper(setup.direction);
    if (direction === 'LONG') return 'bullish';
    if (direction === 'SHORT') return 'bearish';
    return 'directional';
  }

  function buildTradeThesis(setup = {}) {
    const direction = directionWord(setup);
    const trend = firstPresent(setup.stockTrend, setup.trendDirection, setup.trend_state, setup.trendState, setup.trend);
    const htf = firstPresent(setup.higherTimeframeAlignment, setup.htfAlignment, setup.setupTimeframeDirection);
    const location = firstPresent(setup.stockLocation, setup.location, setup.trade_eval?.location);
    const phase = firstPresent(setup.stockPhase, setup.phase);
    const structure = firstPresent(setup.structureState, setup.structure_state, setup.trade_eval?.structure_state);
    const confirmation = firstPresent(setup.confirmationReason, setup.confirmation_status, setup.confirmationStatus);
    const gradeReason = firstPresent(setup.setupGradeReason, setup.stockSetupStatusReason, setup.setupStatusReason, setup.quality?.coach_note);
    const actionReason = firstPresent(setup.action_reason, setup.actionLayerReason, setup.trade_eval?.reason);
    const lines = [];

    lines.push(`${upper(setup.ticker) || 'This setup'} is currently a ${direction} setup based on the scanner output.`);
    if (trend || htf || location) {
      lines.push([
        trend ? `Trend: ${trend}` : '',
        htf ? `HTF: ${htf}` : '',
        location ? `Location: ${location}` : '',
      ].filter(Boolean).join(' · '));
    }
    if (phase || structure || confirmation) {
      lines.push([
        phase ? `Price phase: ${phase}` : '',
        structure ? `Structure: ${structure}` : '',
        confirmation ? `Confirmation: ${confirmation}` : '',
      ].filter(Boolean).join(' · '));
    }
    if (gradeReason || actionReason) {
      lines.push(firstPresent(gradeReason, actionReason));
    }
    return lines.filter(Boolean);
  }

  function keyLevels(setup = {}) {
    const targetRows = typeof self !== 'undefined' && self.KairosCardData
      ? self.KairosCardData.targetRows(setup)
      : localTargetRows(setup);
    const levels = [
      ['Current Price', firstPresent(setup.price, setup.current_price, setup.underlying_price)],
      ['Entry', firstPresent(setup.entry, setup.entry_price)],
      ['Stop', firstPresent(setup.sl, setup.stop, setup.stop_price)],
      ...targetRows.map(row => [row.label, row.value]),
      ['Risk/Reward', firstPresent(setup.rr, setup.riskReward, setup.reward_risk)],
      ['Nearest Resistance', firstPresent(setup.nearestResistance, setup.resistance, setup.supply)],
      ['Nearest Support', firstPresent(setup.nearestSupport, setup.support, setup.demand)],
      ['Supply/Demand Zone', firstPresent(setup.active_zone, setup.zone, setup.supplyDemandZone)],
      ['Invalidation Level', firstPresent(setup.invalidation, setup.invalidation_level)],
    ];
    return levels
      .map(([label, value]) => ({ label, value }))
      .filter(row => row.value !== undefined && row.value !== null && row.value !== '' && row.value !== 0);
  }

  function localTargetRows(setup = {}) {
    const rows = [];
    const pushDistinct = (label, value) => {
      const numeric = finiteNumber(value);
      if (numeric === null) return;
      if (rows.some(row => Math.abs(Number(row.value) - numeric) < 0.000001)) return;
      rows.push({ label, value: numeric });
    };
    const tp1 = finiteNumber(firstPresent(setup.tp1, setup.target, setup.target_price));
    const tp2 = finiteNumber(firstPresent(setup.tp2, setup.target2, setup.target_2));
    const tp3 = finiteNumber(firstPresent(setup.tp3, setup.target3, setup.target_3));
    const finalTarget = finiteNumber(firstPresent(setup.final_target, setup.finalTarget, setup.finalTargetPrice));
    if (tp2 === null && tp3 === null && finalTarget === null) {
      pushDistinct('Target', tp1);
    } else {
      pushDistinct('TP1', tp1);
      pushDistinct('TP2', tp2);
      pushDistinct('TP3', tp3);
      pushDistinct('Final Target', finalTarget);
    }
    return rows;
  }

  function optionContract(setup = {}) {
    const best = setup.best_contract || {};
    const option = setup.option || {};
    const saved = setup.suggested_contract || setup.contract || {};
    const source = [best, option, saved].find(item => item && Object.keys(item).length) || {};
    return {
      type: firstPresent(setup.option_type, source.type, contractType(setup)),
      strike: firstPresent(setup.strike_price, source.strike),
      expiration: firstPresent(setup.expiration_date, source.expiry, source.expiration),
      dte: firstPresent(setup.days_to_expiration_at_entry, source.dte, source.days_to_expiration),
      bid: firstPresent(setup.option_bid_at_entry, source.bid),
      ask: firstPresent(setup.option_ask_at_entry, source.ask),
      spread: firstPresent(setup.option_spread_at_entry, source.spread),
      openInterest: firstPresent(setup.open_interest_at_entry, source.open_interest, source.openInterest),
      volume: firstPresent(setup.option_volume_at_entry, source.volume),
      impliedVolatility: firstPresent(setup.implied_volatility_at_entry, source.implied_volatility, source.iv),
      premium: firstPresent(setup.premium_paid, source.mark, source.mid, source.ask),
      breakeven: firstPresent(setup.breakeven_price, source.breakeven),
      underlying: firstPresent(setup.underlying_price_at_entry, setup.underlying_price, setup.price),
      source: firstPresent(setup.contract_guidance_source, best.available === true ? 'validated_live' : ''),
      potentialType: firstPresent(setup.potential_option_type),
      potentialStrike: firstPresent(setup.potential_strike),
      potentialExpirationMinDte: firstPresent(setup.potential_expiration_min_dte),
      potentialExpirationMaxDte: firstPresent(setup.potential_expiration_max_dte),
    };
  }

  function expirationComparison(contract = {}, readiness = {}) {
    const dte = finiteNumber(contract.dte);
    const min = finiteNumber(readiness.suggested_expiration_min_dte);
    const max = finiteNumber(readiness.suggested_expiration_max_dte);
    if (min === null || max === null) return { selected: dte, range: null, enoughTime: 'Learning' };
    if (dte === null) return { selected: null, range: `${min}-${max} DTE`, enoughTime: 'Learning' };
    return {
      selected: dte,
      range: `${min}-${max} DTE`,
      enoughTime: dte >= min ? 'Yes' : 'No',
    };
  }

  function historicalBehavior(stats = {}) {
    return [
      ['Median MFE in R', stats.median_mfe_r],
      ['Median MAE in R', stats.median_mae_r],
      ['Median target distance in ATR', stats.median_target_distance_atr],
      ['Median days to entry', stats.median_days_to_entry],
      ['Median days to target', stats.median_days_to_target],
      ['Median days to stop', stats.median_days_to_stop],
      ['Target completion rate', stats.target_completion_rate],
      ['Stop completion rate', stats.stop_completion_rate],
      ['Manual close rate', stats.manual_close_rate],
      ['Tracking-window expiry rate', stats.expiry_rate],
    ].filter(([, value]) => value !== undefined && value !== null && value !== '');
  }

  function riskAndTiming(setup = {}, readiness = {}, contract = optionContract(setup)) {
    const risks = [];
    const earningsDays = finiteNumber(firstPresent(setup.days_until_earnings, setup.daysUntilEarnings));
    if (earningsDays !== null) {
      if (earningsDays <= 7) risks.push('Earnings are within 7 days.');
      else if (earningsDays <= 14) risks.push('Earnings are within 14 days.');
    }
    const entryStatus = normalizeText(firstPresent(setup.entryStatus, setup.entry_status));
    if (entryStatus && entryStatus !== 'Tradeable') risks.push(`Entry status: ${entryStatus}.`);
    if (upper(setup.setupStatus).includes('STALE')) risks.push('Setup may be stale.');
    if (readiness && readiness.card_ready === false) risks.push(`Expected Move: ${(readiness.reasons || [])[0] || 'Learning'}.`);
    const spread = finiteNumber(contract.spread);
    if (spread !== null && spread > 1) risks.push('Option spread may need review.');
    if (!contract.type || !contract.strike || !contract.expiration) risks.push('No complete option contract selected.');
    return Array.from(new Set(risks));
  }

  function helpsHurts(setup = {}, readiness = {}, contract = optionContract(setup)) {
    const helps = [];
    const hurts = [];
    const isCGrade = isExplicitCGradeSetup(setup);
    const direction = upper(setup.direction);
    const location = upper(firstPresent(setup.stockLocation, setup.location, setup.trade_eval?.location));
    const confirmationStarted = !isCGrade && Boolean(setup.confirmationStarted || setup.trade_eval?.trigger_confirmed || setup.trade_eval?.rejection_confirmed);
    const htf = upper(firstPresent(setup.higherTimeframeAlignment, setup.htfAlignment, setup.setupTimeframeDirection));
    const entryStatus = normalizeText(firstPresent(setup.entryStatus, setup.entry_status));

    if (htf.includes('ALIGN') || htf.includes('BULLISH') || htf.includes('BEARISH')) helps.push('Higher-timeframe context is defined.');
    if ((direction === 'LONG' && location.includes('DISCOUNT')) || (direction === 'SHORT' && location.includes('PREMIUM'))) helps.push(`Price is in ${location.toLowerCase()} for the setup direction.`);
    if (confirmationStarted) helps.push('Confirmation has started.');
    if (!isCGrade && entryStatus === 'Tradeable') helps.push('Entry conditions are marked tradeable.');
    if (contract.type && contract.strike && contract.expiration) helps.push('A contract is selected for review.');
    if (readiness.card_ready) helps.push('Expected Move is release-ready for this exact group.');

    if (!confirmationStarted) hurts.push('Setup is still awaiting confirmation.');
    if (readiness.card_ready === false) hurts.push('Expected Move is still learning or not release-ready.');
    if (!contract.type || !contract.strike || !contract.expiration) hurts.push('Option contract details are incomplete.');
    const earningsDays = finiteNumber(firstPresent(setup.days_until_earnings, setup.daysUntilEarnings));
    if (earningsDays !== null && earningsDays <= 14) hurts.push('Earnings are near the selected window.');

    return {
      helps: Array.from(new Set(helps)).slice(0, 5),
      hurts: Array.from(new Set(hurts)).slice(0, 5),
    };
  }

  function currentSetupStatus(setup = {}) {
    if (isExplicitCGradeSetup(setup)) return 'SKIP';
    return firstPresent(
      setup.trade_progress_status,
      setup.stockSetupStatus,
      setup.setupStatus,
      setup.entryStatus,
      setup.trade_eval?.trade_stage,
      'Watching'
    );
  }

  function bottomLine(setup = {}, readiness = {}, contract = optionContract(setup)) {
    const ticker = upper(setup.ticker) || 'This setup';
    const direction = directionWord(setup);
    const status = currentSetupStatus(setup);
    const lines = [`${ticker} remains a ${direction} setup with current status: ${status}.`];
    if (readiness.card_ready) {
      lines.push(`Expected Move is release-ready for the exact setup group: ${readiness.expected_move_min_days}-${readiness.expected_move_max_days} trading days.`);
    } else {
      lines.push(`Expected Move is still learning for this exact setup group${(readiness.reasons || []).length ? `: ${readiness.reasons[0]}` : '.'}`);
    }
    if (contract.expiration && readiness.suggested_expiration_min_dte) {
      const comparison = expirationComparison(contract, readiness);
      lines.push(`Selected expiration comparison: ${comparison.enoughTime}.`);
    }
    return lines.slice(0, 4);
  }

  function buildAnalysisModel(setup = {}, readiness = {}, stats = {}) {
    const contract = optionContract(setup);
    return {
      id: setupIdentity(setup),
      selector_label: selectorLabel(setup),
      criteria: setupCriteria(setup),
      thesis: buildTradeThesis(setup),
      key_levels: keyLevels(setup),
      contract,
      expiration_comparison: expirationComparison(contract, readiness),
      historical_behavior: historicalBehavior(stats),
      risks: riskAndTiming(setup, readiness, contract),
      helps_hurts: helpsHurts(setup, readiness, contract),
      status: currentSetupStatus(setup),
      bottom_line: bottomLine(setup, readiness, contract),
    };
  }

  return {
    setupIdentity,
    selectorLabel,
    setupCriteria,
    buildTradeThesis,
    keyLevels,
    optionContract,
    expirationComparison,
    historicalBehavior,
    riskAndTiming,
    helpsHurts,
    currentSetupStatus,
    bottomLine,
    buildAnalysisModel,
  };
});
