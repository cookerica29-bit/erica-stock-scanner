(function(root, factory) {
  if (typeof module === 'object' && module.exports) {
    module.exports = factory();
  } else {
    root.KairosExecutionGuidance = factory();
  }
})(typeof self !== 'undefined' ? self : this, function() {
  function upper(value) {
    return String(value || '').trim().toUpperCase();
  }

  function finiteNumber(value) {
    if (value === null || value === undefined || value === '') return null;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  function plannedEntry(setup = {}) {
    return finiteNumber(setup.entry ?? setup.entry_price);
  }

  function currentPrice(setup = {}) {
    return finiteNumber(setup.price ?? setup.current_price ?? setup.underlying_price);
  }

  function normalizedStatusValues(setup = {}, readiness = {}) {
    return [
      readiness.bucket,
      readiness.label,
      readiness.status,
      setup.progress_bucket,
      setup.progress_label,
      setup.simple_status,
      setup.status,
      setup.scanner_status,
      setup.scannerStatus,
      setup.trade_eval && setup.trade_eval.trade_stage,
      setup.setupGrade,
      setup.setup_grade,
    ].map(upper);
  }

  function isConfirmedSetup(setup = {}, readiness = {}) {
    const values = normalizedStatusValues(setup, readiness);
    if (values.some(value => ['ENTER_NOW', 'ENTER NOW', 'A+ READY', 'READY'].includes(value))) return true;
    const tradeStage = upper(setup.trade_eval && setup.trade_eval.trade_stage);
    return tradeStage === 'B+ TRADEABLE' && upper(setup.entryStatus) === 'TRADEABLE';
  }

  function isNoTrade(setup = {}, readiness = {}) {
    const bucket = upper(readiness.bucket);
    const tradeStage = upper(setup.trade_eval && setup.trade_eval.trade_stage);
    const setupStatus = upper(setup.setupStatus || setup.stockSetupStatus);
    return bucket === 'SKIP'
      || tradeStage.includes('NO TRADE')
      || tradeStage.includes('INVALID')
      || setupStatus.includes('INVALID');
  }

  function isDeveloping(setup = {}, readiness = {}) {
    const bucket = upper(readiness.bucket);
    const tradeStage = upper(setup.trade_eval && setup.trade_eval.trade_stage);
    return bucket === 'WAITING'
      || tradeStage.includes('BUILDING')
      || tradeStage.includes('WATCHLIST')
      || upper(setup.entryStatus) === 'WAITING';
  }

  function hasAvailableContract(contractLifecycle) {
    return contractLifecycle === 'available';
  }

  function executionState(setup = {}, readiness = {}) {
    if (!isConfirmedSetup(setup, readiness)) return 'SETUP_NOT_CONFIRMED';
    const entryStatus = upper(setup.entryStatus);
    const tradeStage = upper(setup.trade_eval && setup.trade_eval.trade_stage);
    const setupStatus = upper(setup.setupStatus || setup.stockSetupStatus);
    if (
      entryStatus === 'TOO FAR'
      || tradeStage.includes('STALE')
      || tradeStage.includes('EXTENDED')
      || setupStatus.includes('STALE')
      || setupStatus.includes('EXTENDED')
    ) {
      return 'SETUP_CONFIRMED_ENTRY_PASSED';
    }
    const distanceAtr = finiteNumber(setup.distanceFromEntryAtr);
    if (entryStatus === 'TRADEABLE' || (distanceAtr !== null && distanceAtr <= 0.25)) {
      return 'SETUP_CONFIRMED_ENTRY_REACHED';
    }
    const price = currentPrice(setup);
    const entry = plannedEntry(setup);
    if (price !== null && entry !== null) {
      const tolerance = Math.max(Math.abs(entry) * 0.0025, 0.01);
      if (Math.abs(price - entry) <= tolerance) return 'SETUP_CONFIRMED_ENTRY_REACHED';
    }
    return 'SETUP_CONFIRMED_WAITING_FOR_ENTRY';
  }

  function nextStep(setup = {}, readiness = {}, contractLifecycle = 'pending') {
    const bucket = upper(readiness.bucket);
    const entryStatus = String(setup.entryStatus || '').trim();
    const entry = plannedEntry(setup);

    if (isNoTrade(setup, readiness)) {
      return {
        label: 'Next Step',
        lines: ['No entry. The setup is not currently valid.'],
      };
    }

    if (isConfirmedSetup(setup, readiness)) {
      const state = executionState(setup, readiness);
      const entryText = entry !== null ? `$${entry.toFixed(2)}` : 'the planned entry';
      if (state === 'SETUP_CONFIRMED_ENTRY_PASSED') {
        return {
          label: 'Next Step',
          lines: ['Do not chase. Price moved beyond the planned entry.'],
        };
      }
      if (state === 'SETUP_CONFIRMED_ENTRY_REACHED') {
        if (hasAvailableContract(contractLifecycle)) {
          return {
            label: 'Next Step',
            lines: ['Price is at the planned entry. You can execute this trade.'],
          };
        }
        return {
          label: 'Next Step',
          lines: ['Price is at the planned entry.', 'Verify and select the live option contract before executing.'],
        };
      }
      return {
        label: 'Next Step',
        lines: [`Setup confirmed. Wait for price to reach the planned entry at ${entryText}.`, entry !== null ? `Set an alert at ${entryText}.` : null].filter(Boolean),
      };
    }

    if (entryStatus === 'Tradeable') {
      return {
        label: 'Next Step',
        lines: ['Monitor price near the planned entry.'],
      };
    }

    if (bucket === 'ALMOST_READY' || entryStatus === 'Near Entry') {
      return {
        label: 'Next Step',
        lines: ['Setup is still developing. Wait for full confirmation.'],
      };
    }

    if (isDeveloping(setup, readiness)) {
      return {
        label: 'Next Step',
        lines: ['Continue monitoring. The setup is still developing.'],
      };
    }

    return {
      label: 'Next Step',
      lines: ['Continue monitoring. The setup is still developing.'],
    };
  }

  function executionPlanRows(setup = {}) {
    return [
      { label: 'Current Price', value: currentPrice(setup) },
      { label: 'Planned Entry', value: plannedEntry(setup) },
      { label: 'Stop', value: finiteNumber(setup.sl ?? setup.stop ?? setup.stop_price) },
      { label: 'Target', value: finiteNumber(setup.tp1 ?? setup.target ?? setup.target_price) },
    ];
  }

  return {
    plannedEntry,
    currentPrice,
    nextStep,
    executionState,
    isConfirmedSetup,
    executionPlanRows,
  };
});
