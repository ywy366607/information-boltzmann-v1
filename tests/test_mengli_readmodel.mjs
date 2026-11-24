import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { careHistory, checkpointMetrics, familyLife, familyTelemetry } from '../scripts/ib_local/mengli_harness/family_readmodel.mjs';

test('original receipt date follows the matching individual and preserves an explicit card date', () => {
  const card={identity:{individual_id:'mengli-native-18ce4392d36ca574'},age:{byte_events:2699511},dynamics:{}};
  const result=familyLife({card},null);
  assert.equal(result.card.age.born_at,'2026-11-15T21:03:31.2311383+08:00');
  assert.equal(result.card.age.byte_events,2699511);
  assert.equal(card.age.born_at,undefined);
  assert.equal(familyLife({card:{...card,identity:{individual_id:'other'}}},null).card.age.born_at,undefined);
  const dated={...card,age:{...card.age,born_at:'2026-01-01T00:00:00Z'}};
  assert.equal(familyLife({card:dated},null).card,dated);
});

test('checkpoint oscillator state survives an older event with the same cycle number', () => {
  const life = { card: { source: { cycle: 2612 }, latest: { momentum_norms: [4, 7] },
    dynamics: { frequency: [[.3, 39], [.4, 38]], source_cycle: 2612,
      source_kind: 'committed_checkpoint_state' } } };
  const result = familyLife(life, null, { row: { cycle: 2612,
    adaptive_dynamics: { frequency: [[.8]], coupling: [.005] } } });
  assert.equal(result, life);
  assert.deepEqual(result.card.dynamics.frequency, [[.3, 39], [.4, 38]]);
});

test('learned dynamics use only same-cycle evidence and override birth defaults without modifying the card', () => {
  const card = { source: { cycle: 1600 }, latest: { momentum_norms: [2, 4] },
    dynamics: { frequency: .8, coupling: .005 } };
  const adaptive = { frequency: [[.79, .81], [.78, .82]], coupling: [.004, .006],
    context_gain: [0, .01], content_coefficients: [[0, 0], [0, .01]] };
  const row = { cycle: 1600, adaptive_dynamics: adaptive,
    momentum_wood: { choice_probability_tv_without_p: .006 } };
  const life = familyLife({ card }, null, { row });
  assert.deepEqual(life.card.dynamics.frequency, adaptive.frequency);
  assert.equal(life.card.dynamics.source_cycle, 1600);
  assert.equal(life.card.latest.momentum_wood.choice_probability_tv_without_p, .006);
  assert.equal(card.dynamics.frequency, .8);
  assert.equal(familyLife({ card }, null, { row: { ...row, cycle: 1601 } }).card, card);
  assert.equal(familyLife({ card }, null, { row: { ...row, cycle: 1599 } }).card, card);
});

const snapshot = { branch_id: 'selected', checkpoint_ref: { session_id: 'care-log', turn_end_seq: 55, result_seq: 55 },
  turns: Array.from({ length: 16 }, (_, i) => ({ role: i % 2 ? 'assistant' : 'user', content: `record ${i}` })) };

test('shared mother history counts actual messages/pairs, not event sequence or new family cursor', () => {
  const history = careHistory(snapshot, { limit: 8 });
  assert.equal(history.total_messages, 16);
  assert.equal(history.recorded_pairs, 8);
  assert.equal(history.turns.length, 8);
  assert.equal(history.turns[0].actor, 'mengli');
  assert.equal(history.turns[1].actor, 'mother');
  assert.equal(history.next_before, 8);
  assert.match(history.reference_meaning, /not conversation-round counts/);
  assert.match(history.authority, /never a new parent authorization/);
  const older = careHistory(snapshot, { before: history.next_before, limit: 8 });
  assert.equal(older.turns[0].record_index, 0);
  assert.equal(older.next_before, null);
  assert.equal(snapshot.turns[0].role, 'user');
});

test('only explicitly bound checkpoint-adjacent matching metrics are exposed', t => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'mengli-readmodel-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const checkpoint = path.join(root, 'last.pt.gz');
  fs.writeFileSync(path.join(root, 'metrics.jsonl'), [
    JSON.stringify({ cycle: 3508, updates: 4 }), JSON.stringify({ cycle: 3509, updates: 0 }),
    'incomplete-json', JSON.stringify({ cycle: 3510, updates: 6 }), '',
  ].join('\n'));
  const read = checkpointMetrics(checkpoint, 3509, { maxRows: 1 });
  assert.deepEqual(read.rows, [{ cycle: 3509, updates: 0 }]);
  assert.equal(read.limited, true);
  assert.equal(read.invalid_lines, 1);
  assert.equal(read.through_cycle, 3509);
  assert.match(read.meaning, /not lifetime totals/);
  assert.equal(checkpointMetrics(path.join(root, 'other/last.pt.gz'), 3509).state, 'not_recorded');
  assert.equal(checkpointMetrics(null, null).state, 'unbound');
});

test('telemetry uses committed card and row without loading a checkpoint', () => {
  const card = { source: { cycle: 12 }, latest: { updates: 0 } };
  const latest = { status: 'committed', cycle: 12, row: { cycle: 12, updates: 0 }, life_card_data: card };
  const family = { inspect: () => ({ checkpoint: null, cycle: 12, branch_id: 'live', mode: 'paused', loaded: true,
    busy: false, events: [latest] }) };
  const result = familyTelemetry(family, { card: null, state: 'unbound' });
  assert.equal(result.source.kind, 'committed');
  assert.equal(result.life.card, card);
  assert.equal(result.latest_event.row.updates, 0);
  assert.equal(result.status.mode, 'paused');
});

test('life dynamics projects only bound config metadata and reuses recorded metadata first', t => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'mengli-dynamics-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const checkpoint = path.join(root, 'never-opened.pt.gz');
  const file = path.join(root, 'config.json');
  const dynamics = { dt: 0, frequency: [0.8, 1.2], coupling: 0, steps: 0, enabled: false };
  const config = JSON.stringify({ bfss_dynamics: { ...dynamics, schema: 1, migrated_at_cycle: 5 }, private: 'not exposed' });
  fs.writeFileSync(file, config);
  const card = { source: { checkpoint, cycle: 12 } };
  const life = { card, state: 'saved' };
  const result = familyLife(life);
  assert.deepEqual(result.card.dynamics, dynamics);
  assert.deepEqual(result.dynamics_source, { kind: 'bound_checkpoint_config', checkpoint, file });
  assert.equal(card.dynamics, undefined);
  assert.equal(fs.readFileSync(file, 'utf8'), config);
  assert.equal(fs.existsSync(checkpoint), false);
  const recorded = { dt: 0.02, frequency: [2, 3], enabled: true };
  const saved = { card: { ...card, dynamics: recorded }, state: 'saved' };
  assert.equal(familyLife(saved), saved);
  assert.equal(familyLife({ ...life, dynamics: recorded }).card.dynamics, recorded);
  const family = { inspect: () => ({ checkpoint, cycle: 12, events: [{ life_card_data: saved.card }] }) };
  assert.equal(familyTelemetry(family, life).life.card.dynamics, recorded);
});

test('missing, malformed, oversized or absent dynamics stay unknown without searching other runs', t => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'mengli-dynamics-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  fs.mkdirSync(path.join(root, 'bound'));
  fs.writeFileSync(path.join(root, 'config.json'), JSON.stringify({ bfss_dynamics: { dt: 99 } }));
  const checkpoint = path.join(root, 'bound', 'last.pt');
  const life = { card: { source: { checkpoint } }, state: 'saved' };
  const file = path.join(root, 'bound', 'config.json');
  assert.equal(familyLife(life).card.dynamics, undefined);
  for (const content of ['{broken', '{}', '{"bfss_dynamics":[]}', ' '.repeat(129)]) {
    fs.writeFileSync(file, content);
    const result = familyLife(life, checkpoint, { maxBytes: 128 });
    assert.equal(result.card.dynamics, undefined);
    assert.equal(result.dynamics_source, undefined);
  }
  fs.writeFileSync(file, JSON.stringify({ bfss_dynamics: { enabled: false } }));
  assert.deepEqual(familyLife(life).card.dynamics,
    { dt: null, frequency: null, coupling: null, steps: null, enabled: false });
  const unbound = { card: null, state: 'unbound' };
  assert.equal(familyLife(unbound), unbound);
});

test('life routes and saved/committed telemetry expose the same dynamics without writing the raw card', async t => {
  const { createMengliServer } = await import('../scripts/ib_local/mengli_harness/mengli_service.mjs');
  const { HARNESS_ROOT } = await import('../scripts/ib_local/mengli_harness/dsh_runtime.mjs');
  const root = fs.mkdtempSync(path.join(HARNESS_ROOT, 'readmodel-test-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const checkpoint = path.join(root, 'not-loaded.pt.gz');
  const lifeCardPath = path.join(root, 'life.json');
  const dynamics = { dt: 0.01, frequency: 0.8, coupling: 0.01, steps: 1, enabled: true };
  fs.writeFileSync(path.join(root, 'config.json'), JSON.stringify({ bfss_dynamics: dynamics }));
  const card = { schema: 1, identity: { name: 'fixture' }, source: { checkpoint, cycle: 12, sha256: 'fixture' }, age: { cycles: 12 } };
  const original = JSON.stringify(card);
  fs.writeFileSync(lifeCardPath, original);
  const state = { checkpoint, cycle: 12, events: [] };
  const family = { state, inspect: () => state };
  const server = createMengliServer({}, { lifeCardPath, family });
  t.after(() => { server.closeAllConnections(); return new Promise(resolve => server.close(resolve)); });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const get = async route => {
    const response = await fetch(`http://127.0.0.1:${server.address().port}${route}`);
    assert.equal(response.status, 200);
    return response.json();
  };
  const life = await get('/api/family/life');
  assert.deepEqual(life.card.dynamics, dynamics);
  assert.deepEqual(await get('/api/life'), life);
  assert.deepEqual((await get('/api/family/telemetry')).life, life);
  state.events = [{ status: 'committed', life_card_data: card }];
  assert.deepEqual((await get('/api/family/telemetry')).life, life);
  assert.equal(card.dynamics, undefined);
  assert.equal(fs.readFileSync(lifeCardPath, 'utf8'), original);
  assert.equal(fs.existsSync(checkpoint), false);
});
