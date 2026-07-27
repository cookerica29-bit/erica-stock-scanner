const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const executionGuidance = require('../public/execution_guidance.js');
const cardData = require('../public/card_data.js');

const html = fs.readFileSync('public/index.html', 'utf8');
const inline = [...html.matchAll(/<script(?![^>]*src=)[^>]*>([\s\S]*?)<\/script>/gi)][0][1];

function elementStub() {
  return {
    innerHTML: '',
    textContent: '',
    value: '',
    checked: false,
    dataset: {},
    style: {},
    selectedOptions: [{ textContent: '' }],
    classList: { add() {}, remove() {}, toggle() {} },
    addEventListener() {},
    removeEventListener() {},
    setAttribute() {},
    appendChild() {},
    remove() {},
  };
}

const storage = {};
const context = {
  console,
  Date,
  Math,
  Number,
  String,
  Boolean,
  Array,
  Object,
  JSON,
  RegExp,
  performance: { now: () => 100 },
  URLSearchParams,
  encodeURIComponent,
  decodeURIComponent,
  setTimeout: () => 1,
  clearTimeout: () => {},
  setInterval: () => 1,
  clearInterval: () => {},
  alert: () => {},
  confirm: () => true,
  fetch: () => Promise.reject(new Error('network disabled in contract optional journal test')),
  localStorage: {
    getItem: key => storage[key] || null,
    setItem: (key, value) => { storage[key] = String(value); },
    removeItem: key => { delete storage[key]; },
  },
  document: {
    body: { appendChild() {} },
    getElementById: () => elementStub(),
    querySelector: () => null,
    querySelectorAll: () => [],
    createElement: () => elementStub(),
    addEventListener() {},
    removeEventListener() {},
  },
  navigator: { clipboard: { writeText: () => Promise.resolve() } },
};
context.window = {
  KairosExecutionGuidance: executionGuidance,
  KairosCardData: cardData,
  open: () => {},
  addEventListener: () => {},
  removeEventListener: () => {},
};
context.self = context.window;
context.globalThis = context;

vm.createContext(context);
vm.runInContext(inline, context);

const noContract = context.optionContractFieldsFromValues({
  optionType: 'N/A',
  strike: null,
  expiry: '',
  entryPremium: null,
  quantity: 1,
  direction: 'SHORT',
}, { contractValidationMode: 'journal_setup_save' });
assert.strictEqual(noContract.attached, false);
assert.strictEqual(noContract.invalid_reason, undefined);

const noContractJournalFields = context.journalContractFieldsFromForm({
  option: '',
  optionType: 'N/A',
  strike: null,
  expiry: '',
  askAtSelection: null,
  contracts: 1,
  direction: 'SHORT',
  entry: 30,
});
assert.strictEqual(noContractJournalFields.option_contract_invalid_reason, '');
assert.strictEqual(noContractJournalFields.option_type, 'N/A');
assert.strictEqual(noContractJournalFields.option_strike, null);
assert.strictEqual(noContractJournalFields.option_expiration, '');
assert.strictEqual(noContractJournalFields.option_entry_premium, null);
assert.strictEqual(noContractJournalFields.option_quantity, null);

const completeContract = context.optionContractFieldsFromValues({
  optionType: 'PUT',
  strike: 29,
  expiry: '2026-07-31',
  entryPremium: 0.21,
  quantity: 1,
}, { contractValidationMode: 'journal_setup_save' });
assert.strictEqual(completeContract.attached, true);
assert.strictEqual(completeContract.option_entry_cost, 21);

const partialContract = context.optionContractFieldsFromValues({
  optionType: 'N/A',
  strike: 29,
  expiry: '',
  entryPremium: null,
  quantity: 1,
}, { contractValidationMode: 'journal_setup_save' });
assert.strictEqual(partialContract.attached, false);
assert.ok(partialContract.invalid_reason.includes('Complete the option contract or clear the partial contract details.'));
assert.ok(partialContract.invalid_reason.includes('type'));
assert.ok(partialContract.invalid_reason.includes('expiration'));
assert.ok(partialContract.invalid_reason.includes('entry premium'));

const positionOpenBlank = context.validateOptionContractForMode({
  optionType: 'N/A',
  strike: null,
  expiry: '',
  entryPremium: null,
  quantity: 1,
}, 'position_open');
assert.strictEqual(positionOpenBlank.valid, false);
assert.strictEqual(positionOpenBlank.message, 'Attach or confirm the option contract before opening this option position.');

const positionOpenComplete = context.validateOptionContractForMode({
  optionType: 'CALL',
  strike: 100,
  expiry: '2026-08-21',
  entryPremium: 1.25,
  quantity: 2,
}, 'position_open');
assert.strictEqual(positionOpenComplete.valid, true);

const oldRecord = context.normalizeJournalOptionContract({ ticker: 'OLD', direction: 'SHORT', result: 'Open' });
assert.strictEqual(oldRecord.attached, false);
assert.strictEqual(oldRecord.compactLabel, 'No option contract attached');

console.log('contract_optional_journal_saves_v1 passed');
