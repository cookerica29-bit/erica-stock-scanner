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

  function setupGradeValue(setup = {}) {
    const grade = upper(setup.setupGrade);
    return ['A', 'B', 'C'].includes(grade) ? grade : 'C';
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
    if (setupGradeValue(setup) === 'C') return false;
    const bucket = upper(readiness.bucket);
    if (bucket) return bucket === 'ENTER_NOW';
    const values = normalizedStatusValues(setup, readiness);
    if (values.some(value => ['ENTER_NOW', 'ENTER NOW', 'A+ READY', 'READY'].includes(value))) return true;
    return false;
  }

  function isEarlyEntrySetup(setup = {}, readiness = {}) {
    if (setupGradeValue(setup) === 'C') return false;
    const bucket = upper(readiness.bucket);
    if (bucket) return bucket === 'EARLY_ENTRY' || bucket === 'EARLY ENTRY';
    const tradeEval = setup.trade_eval || {};
    return tradeEval.b_plus_tradeable === true
      && upper(setup.entryStatus) === 'TRADEABLE'
      && tradeEval.trigger_confirmed !== true
      && tradeEval.a_plus_ready !== true;
  }

  function executionLifecyclePresentation(setup = {}) {
    if (setup.execution_lifecycle_presentation_enabled !== true) return null;
    const state = upper(setup.execution_lifecycle_state || (setup.execution_lifecycle && setup.execution_lifecycle.state));
    const map = {
      EARLY_ENTRY_BUILDING: {
        label: 'EARLY ENTRY',
        bucket: 'EARLY_ENTRY',
        className: 'early-entry',
        executionState: 'SETUP_EARLY_ENTRY',
        nextLines: ['Price has not reached the planned entry. Full confirmation is still developing.'],
      },
      EARLY_TOUCH: {
        label: 'EARLY TOUCH',
        bucket: 'EARLY_TOUCH',
        className: 'early-touch',
        executionState: 'SETUP_EARLY_TOUCH',
        nextLines: ['Price reached the entry before confirmation. Wait - this is not Enter Now.'],
      },
      WAITING_FOR_RETEST: {
        label: 'WAITING FOR RETEST',
        bucket: 'WAITING_FOR_RETEST',
        className: 'waiting-retest',
        executionState: 'SETUP_WAITING_FOR_RETEST',
        nextLines: ['Confirmation completed after an early touch. Wait for a fresh retest of the entry area.'],
      },
      ENTRY_TRIGGERED: {
        label: 'ENTER NOW',
        bucket: 'ENTER_NOW',
        className: 'enter-now',
        executionState: 'SETUP_CONFIRMED_ENTRY_REACHED',
        nextLines: ['Price is at a valid post-confirmation entry. Review the option plan before executing.'],
      },
      MISSED_ENTRY: {
        label: 'MISSED ENTRY',
        bucket: 'RESOLVED',
        className: 'too-late',
        executionState: 'SETUP_CONFIRMED_ENTRY_PASSED',
        nextLines: ['The confirmed setup moved away without a valid retest. Do not chase.'],
      },
      TP1_BEFORE_CONFIRMATION: {
        label: 'TP1 BEFORE CONFIRMATION',
        bucket: 'RESOLVED',
        className: 'too-late',
        executionState: 'SETUP_CONFIRMED_ENTRY_PASSED',
        nextLines: ['The move reached the first target before confirmation completed. This setup is no longer actionable.'],
      },
      INVALIDATED: {
        label: 'INVALIDATED',
        bucket: 'RESOLVED',
        className: 'skip',
        executionState: 'SETUP_NOT_CONFIRMED',
        nextLines: ['The setup invalidated before a valid entry.'],
      },
      EXPIRED: {
        label: 'EXPIRED',
        bucket: 'RESOLVED',
        className: 'skip',
        executionState: 'SETUP_NOT_CONFIRMED',
        nextLines: ['The setup aged out before producing a valid entry.'],
      },
      PLAN_REPLACED: {
        label: 'PLAN REPLACED',
        bucket: 'RESOLVED',
        className: 'skip',
        executionState: 'SETUP_NOT_CONFIRMED',
        nextLines: ['The trade plan changed materially; use the current plan.'],
      },
    };
    return map[state] || null;
  }

  function isNoTrade(setup = {}, readiness = {}) {
    const bucket = upper(readiness.bucket);
    if (bucket === 'SKIP') return true;
    const tradeStage = upper(setup.trade_eval && setup.trade_eval.trade_stage);
    const setupStatus = upper(setup.setupStatus || setup.stockSetupStatus);
    if (tradeStage.includes('NO TRADE')
      || tradeStage.includes('INVALID')
      || setupStatus.includes('INVALID')) {
      return true;
    }
    return bucket === 'SKIP' && !isConfirmedSetup(setup, readiness) && !hasDevelopingSetupSignal(setup, readiness);
  }

  function isDeveloping(setup = {}, readiness = {}) {
    const bucket = upper(readiness.bucket);
    const tradeStage = upper(setup.trade_eval && setup.trade_eval.trade_stage);
    return bucket === 'WAITING'
      || tradeStage.includes('BUILDING')
      || tradeStage.includes('WATCHLIST')
      || upper(setup.entryStatus) === 'WAITING';
  }

  function hasDevelopingSetupSignal(setup = {}, readiness = {}) {
    const entryStatus = upper(setup.entryStatus);
    const tradeStage = upper(setup.trade_eval && setup.trade_eval.trade_stage);
    const setupGrade = upper(setup.setupGrade || setup.setup_grade);
    return ['ALMOST_READY', 'ALMOST READY', 'WAITING'].includes(upper(readiness.bucket))
      || ['NEAR ENTRY', 'TRADEABLE', 'WAITING'].includes(entryStatus)
      || tradeStage.includes('BUILDING')
      || tradeStage.includes('WATCHLIST')
      || setupGrade.includes('B+')
      || setupGrade === 'B';
  }

  function hasAvailableContract(contractLifecycle) {
    return contractLifecycle === 'available';
  }

  function executionState(setup = {}, readiness = {}) {
    const lifecycle = executionLifecyclePresentation(setup);
    if (lifecycle) return lifecycle.executionState;
    if (isEarlyEntrySetup(setup, readiness)) return 'SETUP_EARLY_ENTRY';
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

  function executeVisualState(executionStateValue) {
    if (executionStateValue === 'SETUP_CONFIRMED_ENTRY_REACHED') return 'ready';
    if (executionStateValue === 'SETUP_CONFIRMED_WAITING_FOR_ENTRY') return 'waiting';
    if (executionStateValue === 'SETUP_EARLY_ENTRY') return 'early-entry';
    if (executionStateValue === 'SETUP_EARLY_TOUCH') return 'early-entry';
    if (executionStateValue === 'SETUP_WAITING_FOR_RETEST') return 'waiting';
    if (executionStateValue === 'SETUP_CONFIRMED_ENTRY_PASSED') return 'passed';
    return 'not-ready';
  }

  function cardStatus(setup = {}, readiness = {}) {
    const lifecycle = executionLifecyclePresentation(setup);
    if (lifecycle) return { label: lifecycle.label, className: lifecycle.className };
    if (isNoTrade(setup, readiness)) return { label: 'NO TRADE', className: 'skip' };
    if (isEarlyEntrySetup(setup, readiness)) return { label: 'EARLY ENTRY', className: 'early-entry' };
    if (isConfirmedSetup(setup, readiness)) return { label: 'ENTER NOW', className: 'enter-now' };
    const bucket = upper(readiness.bucket);
    if (bucket === 'ALMOST_READY' || upper(setup.entryStatus) === 'NEAR ENTRY') return { label: 'ALMOST READY', className: 'almost-ready' };
    return { label: 'BUILDING', className: 'wait' };
  }

  function readinessStages(setup = {}, readiness = {}) {
    const lifecycle = executionLifecyclePresentation(setup);
    if (lifecycle) {
      if (lifecycle.bucket === 'ENTER_NOW') {
        return [
          { label: 'Trend', state: 'complete', status: 'Complete' },
          { label: 'Zone', state: 'complete', status: 'Complete' },
          { label: 'Confirm', state: 'complete', status: 'Complete' },
          { label: 'Execute', state: 'complete execute-entry-ready', status: 'Ready' },
        ];
      }
      if (lifecycle.bucket === 'WAITING_FOR_RETEST') {
        return [
          { label: 'Trend', state: 'complete', status: 'Complete' },
          { label: 'Zone', state: 'complete', status: 'Complete' },
          { label: 'Confirm', state: 'complete', status: 'Complete' },
          { label: 'Execute', state: 'current execute-waiting-entry', status: 'Retest' },
        ];
      }
      if (lifecycle.bucket === 'EARLY_TOUCH') {
        return [
          { label: 'Trend', state: 'complete', status: 'Complete' },
          { label: 'Zone', state: 'complete', status: 'Complete' },
          { label: 'Confirm', state: 'current', status: 'Waiting' },
          { label: 'Execute', state: 'pending execute-not-ready', status: 'Touched Early' },
        ];
      }
      if (lifecycle.bucket === 'RESOLVED') {
        return [
          { label: 'Trend', state: 'complete', status: 'Complete' },
          { label: 'Zone', state: 'complete', status: 'Complete' },
          { label: 'Confirm', state: 'pending skip', status: 'Resolved' },
          { label: 'Execute', state: 'pending skip execute-not-ready', status: lifecycle.label },
        ];
      }
    }
    if (isNoTrade(setup, readiness)) {
      return [
        { label: 'Trend', state: 'pending skip', status: 'No Trade' },
        { label: 'Zone', state: 'pending skip', status: 'Muted' },
        { label: 'Confirm', state: 'pending skip', status: 'Muted' },
        { label: 'Execute', state: 'pending skip execute-not-ready', status: 'Not Ready' },
      ];
    }
    const entryState = executionState(setup, readiness);
    if (isConfirmedSetup(setup, readiness)) {
      const executeState = entryState === 'SETUP_CONFIRMED_ENTRY_REACHED'
        ? { state: 'complete execute-entry-ready', status: 'Ready' }
        : entryState === 'SETUP_CONFIRMED_ENTRY_PASSED'
          ? { state: 'current execute-entry-passed', status: 'Passed' }
          : { state: 'current execute-waiting-entry', status: 'Waiting' };
      return [
        { label: 'Trend', state: 'complete', status: 'Complete' },
        { label: 'Zone', state: 'complete', status: 'Complete' },
        { label: 'Confirm', state: 'complete', status: 'Complete' },
        { label: 'Execute', ...executeState },
      ];
    }
    if (isEarlyEntrySetup(setup, readiness)) {
      return [
        { label: 'Trend', state: 'complete', status: 'Complete' },
        { label: 'Zone', state: 'complete', status: 'Complete' },
        { label: 'Confirm', state: 'current', status: 'Early' },
        { label: 'Execute', state: 'current execute-early-entry', status: 'Caution' },
      ];
    }
    const bucket = upper(readiness.bucket);
    const entryStatus = upper(setup.entryStatus);
    const confirmationStarted = Boolean(setup.confirmationStarted || (setup.trade_eval && (setup.trade_eval.trigger_confirmed || setup.trade_eval.rejection_confirmed)));
    if (bucket === 'ALMOST_READY' || entryStatus === 'NEAR ENTRY' || entryStatus === 'TRADEABLE') {
      const confirmComplete = confirmationStarted && entryStatus === 'TRADEABLE';
      return [
        { label: 'Trend', state: 'complete', status: 'Complete' },
        { label: 'Zone', state: 'complete', status: 'Complete' },
        { label: 'Confirm', state: confirmComplete ? 'complete' : 'current', status: confirmComplete ? 'Complete' : 'Waiting' },
        { label: 'Execute', state: 'pending execute-not-ready', status: 'Not Ready' },
      ];
    }
    return [
      { label: 'Trend', state: setup.direction ? 'complete' : 'current', status: setup.direction ? 'Complete' : 'Waiting' },
      { label: 'Zone', state: 'current', status: 'Building' },
      { label: 'Confirm', state: 'pending', status: 'Not Ready' },
      { label: 'Execute', state: 'pending execute-not-ready', status: 'Not Ready' },
    ];
  }

  function nextStep(setup = {}, readiness = {}, contractLifecycle = 'pending') {
    const lifecycle = executionLifecyclePresentation(setup);
    if (lifecycle) {
      return {
        label: 'Next Step',
        lines: lifecycle.nextLines,
      };
    }
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
            lines: ['Price is at the planned entry. Review the Contract Candidate and verify the selected contract in your broker.'],
          };
        }
        return {
          label: 'Next Step',
          lines: ['Price is at the planned entry.', 'Use the Contract Candidate as planning guidance and verify the contract in your broker before executing.'],
        };
      }
      return {
        label: 'Next Step',
        lines: [`Setup confirmed. Wait for price to reach the planned entry at ${entryText}.`, entry !== null ? `Set an alert at ${entryText}.` : null].filter(Boolean),
      };
    }

    if (isEarlyEntrySetup(setup, readiness)) {
      return {
        label: 'Next Step',
        lines: [
          "Structure has broken, but full confirmation hasn't happened yet.",
          'Consider smaller size given lower confirmation.',
        ],
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
    const tp1 = finiteNumber(setup.tp1 ?? setup.target ?? setup.target_price);
    const tp2 = finiteNumber(setup.tp2 ?? setup.target2 ?? setup.target_2);
    const tp3 = finiteNumber(setup.tp3 ?? setup.target3 ?? setup.target_3);
    const finalTarget = finiteNumber(setup.final_target ?? setup.finalTarget ?? setup.finalTargetPrice);
    const targetRows = [];
    const pushDistinct = (label, value, key) => {
      if (value === null) return;
      if (targetRows.some(row => Math.abs(Number(row.value) - Number(value)) < 0.000001)) return;
      targetRows.push({ label, value, key });
    };
    if (tp2 === null && tp3 === null && finalTarget === null) {
      pushDistinct('Planned TP1', tp1, 'target');
    } else {
      pushDistinct('Planned TP1', tp1, 'tp1');
      pushDistinct('Planned TP2', tp2, 'tp2');
      pushDistinct('Planned TP3', tp3, 'tp3');
      if (tp2 === null && tp3 === null) pushDistinct('Planned Final Target', finalTarget, 'final-target');
      else pushDistinct('Planned Final Target', finalTarget, 'final-target');
    }
    return [
      { label: 'Quote Check / Last Close', value: currentPrice(setup), key: 'current-price' },
      { label: 'Planned Entry', value: plannedEntry(setup), key: 'planned-entry' },
      { label: 'Model Stop', value: finiteNumber(setup.sl ?? setup.stop ?? setup.stop_price), key: 'stop' },
      ...targetRows,
    ];
  }

  return {
    plannedEntry,
    currentPrice,
    nextStep,
    executionState,
    executionLifecyclePresentation,
    executeVisualState,
    isConfirmedSetup,
    isEarlyEntrySetup,
    cardStatus,
    readinessStages,
    executionPlanRows,
  };
});
