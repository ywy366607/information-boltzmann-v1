import test from 'node:test';
import assert from 'node:assert/strict';
import { createMetricsClient } from '../metrics-state.mjs';

test('production life_card_data saved event works without renaming original family event', async () => {
  let state = { connected: true, state: { cycle: 0, events: [] } };
  const listeners = new Set();
  const bridge = { controller: { getSnapshot: () => state, subscribe(listener) { listeners.add(listener); return () => listeners.delete(listener); } } };
  const card = { branch_id: 'selected', source: { cycle: 3509 }, age: { cycles: 3509 }, latest: { momentum_norms: [0, 0, 0, 0] } };
  const client = createMetricsClient({ getBridge: () => bridge, fetcher: async () => ({ ok: true, json: async () => ({ schema: 1,
    source: { cycle: 3509, branch_id: 'selected', kind: 'saved_snapshot' },
    status: { loaded: false, mode: 'paused', busy: false }, life: { card, state: 'saved' },
    history: { rows: [{ cycle: 3509, updates: 4 }], source: 'selected/metrics.jsonl' }, latest_event: null }) }) });
  await client.open();
  const committed = { status: 'committed', cursor: 1, cycle: 3510, branch_id: 'selected',
    life_card_data: { ...card, source: { cycle: 3510 }, age: { cycles: 3510 } }, row: { cycle: 3510, updates: 0 } };
  state = { connected: true, state: { cycle: 3510, events: [committed] } };
  for (const listener of listeners) listener();
  assert.equal(client.getSnapshot().card.age.cycles, 3510);
  assert.equal(client.getSnapshot().rows.at(-1).updates, 0);
  // A replayed older committed event cannot rewind the saved read model.
  state = { ...state, state: { cycle: 3510, events: [{ ...committed, cursor: 0, cycle: 3508 }, committed] } };
  for (const listener of listeners) listener();
  assert.equal(client.getSnapshot().source.cycle, 3510);
  assert.equal(client.getSnapshot().rows.length, 2);
  client.dispose();
});

test('controller appearing via bridge publish is subscribed without polling', async () => {
  const bridgeListeners = new Set();
  const bridge = { controller: null, subscribe(fn) { bridgeListeners.add(fn); return () => bridgeListeners.delete(fn); } };
  let subscribed = 0;
  const client = createMetricsClient({ getBridge: () => bridge, fetcher: async () => ({ ok: true, json: async () => ({ source: { cycle: 3509 }, history: { rows: [] } }) }) });
  await client.open();
  bridge.controller = { getSnapshot: () => ({ connected: true, state: { loaded: false, cycle: 0, events: [] } }),
    subscribe() { subscribed++; return () => subscribed--; } };
  for (const fn of bridgeListeners) fn();
  assert.equal(subscribed, 1);
  assert.equal(client.getSnapshot().source.cycle, 3509);
  client.dispose();
  assert.equal(subscribed, 0); assert.equal(bridgeListeners.size, 0);
});
