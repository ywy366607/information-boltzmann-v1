import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { randomUUID } from 'node:crypto';
import { TrainingObserver } from '../scripts/ib_local/mengli_harness/training_observer.mjs';
import { FamilyRuntime } from '../scripts/ib_local/mengli_harness/family_runtime.mjs';
import { familyTelemetry } from '../scripts/ib_local/mengli_harness/family_readmodel.mjs';
import { createMengliServer } from '../scripts/ib_local/mengli_harness/mengli_service.mjs';
import { HARNESS_ROOT } from '../scripts/ib_local/mengli_harness/dsh_runtime.mjs';
import { lifeStatus, createMetricsClient } from '../scripts/ib_local/mengli_harness/native-metrics-client/metrics-state.mjs';
import { projectFamilyEvents } from '../scripts/ib_local/mengli_harness/native-family-client/family-state.mjs';

function fixture(t) {
  const root = path.join(HARNESS_ROOT, `observer-contract-${randomUUID()}`), body = path.join(root, 'body');
  fs.mkdirSync(body, { recursive: true });
  const write = (file, value) => fs.writeFileSync(file, JSON.stringify(value));
  const bindingPath = path.join(root, 'binding.json'), cardPath = path.join(root, 'life.json');
  const guard = path.join(root, 'status.json'), metrics = path.join(body, 'metrics.jsonl');
  const binding = { schema: 1, run_id: 'offline-observer-contract', individual_id: 'same-individual', branch_id: 'same-branch',
    body, life_card: cardPath, guard_status: guard, segments: [body], start_cycle: 4208, target_cycle: 4720 };
  const card = { schema: 1, identity: { name: '梦漓', individual_id: binding.individual_id }, branch_id: binding.branch_id,
    source: { checkpoint: path.join(body, 'last.pt.gz'), cycle: 4211, sha256: 'a'.repeat(64) }, age: { cycles: 4211 } };
  write(bindingPath, binding); write(cardPath, card);
  write(path.join(body, 'config.json'), { life_identity: card.identity, mother_context_harness: { branch_id: binding.branch_id } });
  write(guard, { status: 'running', pid: process.pid, child_pid: process.pid });
  const row = { cycle: 4210, updates: 6, response_nll: 1.2, world: { child_text: 'My actual words', mother_reply: 'Actual full mother reply', mother_bytes: 12 }, external_bytes: 480 };
  fs.writeFileSync(metrics, JSON.stringify(row) + '\n');
  const observer = new TrainingObserver({ bindingPath });
  const calls = [];
  const family = new FamilyRuntime({ dataRoot: root, observer, worker: { request: async (...a) => { calls.push(a); }, close: async () => {} } });
  t.after(async () => { await family.close(); fs.rmSync(root, { recursive: true, force: true }); });
  return { root, body, write, binding, bindingPath, card, cardPath, guard, metrics, row, observer, family, calls };
}

test('explicit observation separates complete metrics from saved boundary without loading any body', t => {
  const f = fixture(t), a = f.observer.snapshot(), b = f.family.inspect();
  assert.equal(a.observation.latest_completed_cycle, 4210);
  assert.equal(a.observation.saved_cycle, 4211);
  assert.equal(a.observation.status, 'running');
  assert.equal(b.mode, 'observing'); assert.equal(b.read_only, true); assert.equal(b.loaded, false);
  assert.equal(b.cycle, 4210); assert.deepEqual(f.calls, []);
  assert.equal(a.events[0].delivered, undefined);
  assert.equal(a.events[0].observation.mother_bytes, 12);
  const text = projectFamilyEvents(a.events);
  assert.equal(text[1].family.label, '妈妈 · 回应原文');
  assert.equal(text[1].family.timing, 'recorded_reply');
  const telemetry = familyTelemetry(f.family, { card: { source: { cycle: 3509 } } });
  assert.equal(telemetry.source.cycle, 4210);
  assert.equal(telemetry.life.card.source.cycle, 4211);
  assert.equal(telemetry.history.rows.length, 1);
});

test('partial final line stays private and disk appends update the same live source', t => {
  const f = fixture(t), next = { ...f.row, cycle: 4211 };
  fs.appendFileSync(f.metrics, JSON.stringify(next));
  assert.equal(f.observer.snapshot().rows.length, 1);
  fs.appendFileSync(f.metrics, '\n');
  assert.equal(f.observer.snapshot().rows.length, 2);
  f.write(f.guard, { status: 'running', pid: 2147483647, child_pid: 2147483646 });
  assert.equal(f.observer.snapshot().observation.status, 'interrupted');
});

test('native text body uses its own branch and explicit external storage root', t => {
  const f = fixture(t), root = fs.mkdtempSync(path.join(os.tmpdir(), 'mengli-observer-native-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const body = path.join(root, 'body'); fs.mkdirSync(body);
  const binding = { ...f.binding, body, segments: [body], storage_roots: [root] };
  f.write(path.join(body, 'config.json'), { life_identity: f.card.identity,
    native_body: { schema: 1 }, life_card: { owner_branch_id: binding.branch_id } });
  fs.copyFileSync(f.metrics, path.join(body, 'metrics.jsonl'));
  f.write(f.cardPath, { ...f.card, source: { ...f.card.source, checkpoint: path.join(body, 'last.pt.gz') } });
  f.write(f.bindingPath, binding);
  assert.equal(f.observer.snapshot().observation.latest_completed_cycle, 4210);
  f.write(f.bindingPath, { ...binding, storage_roots: [] });
  assert.equal(f.observer.snapshot().observation.status, 'unavailable');
  f.write(f.bindingPath, { ...binding, storage_roots: [path.parse(root).root] });
  assert.equal(f.observer.snapshot().observation.status, 'unavailable');
});

test('natural text is displayed as actual world input, never a fabricated mother reply', t => {
  const f = fixture(t);
  const row = { ...f.row, world: { source: 'text', child_text: 'actual suffix',
    text_reply: 'actual natural text', mother_bytes: 0, provider_calls: 0 } };
  fs.writeFileSync(f.metrics, JSON.stringify(row)+'\n');
  const event = f.observer.snapshot().events[0];
  assert.equal(event.actor, 'world');
  assert.equal(event.observation.kind, 'natural_text_reply');
  assert.equal(event.observation.text, 'actual natural text');
  assert.equal(projectFamilyEvents([event])[1].family.label, '文本世界 · 实际输入');
});

test('configured invalid or mismatched binding fails closed and all body mutations reject', async t => {
  const f = fixture(t);
  f.write(f.cardPath, { ...f.card, identity: { individual_id: 'other-individual' } });
  assert.equal(f.family.inspect().observation.status, 'unavailable');
  assert.equal(f.family.inspect().cycle, null);
  for (const action of [() => f.family.load('http://127.0.0.1:1'), () => f.family.control('mother'),
    () => f.family.say({ text: 'hello' }), () => f.family.step()]) await assert.rejects(action, /网页仅观察/);
  assert.throws(() => f.family.updatePlan({ text: 'new' }), /网页仅观察/);
  assert.deepEqual(f.calls, []);
});

test('no directory discovery, paths bounded, explicit log segments capped and deduplicated', t => {
  const f = fixture(t);
  const other = path.join(f.root, 'other'); fs.mkdirSync(other);
  fs.writeFileSync(path.join(other, 'metrics.jsonl'), JSON.stringify({ cycle: 99999 }) + '\n');
  assert.equal(f.observer.snapshot().rows.at(-1).cycle, 4210);
  const rows = Array.from({ length: 180 }, (_, i) => ({ ...f.row, cycle: 4031 + i }));
  fs.writeFileSync(f.metrics, rows.map(r => JSON.stringify(r)).join('\n') + '\n');
  f.write(f.bindingPath, { ...f.binding, segments: [f.body, f.body] });
  assert.equal(f.observer.snapshot().rows.length, 128);
  assert.equal(f.observer.snapshot().history.limited, true);
  f.write(f.bindingPath, { ...f.binding, body: path.resolve('D:/outside-this-repo') });
  assert.equal(f.observer.snapshot().observation.status, 'unavailable');
});

test('recent curves do not pad a bounded tail with a disconnected retired segment', t => {
  const f = fixture(t);
  const old = path.join(f.root, 'retired'); fs.mkdirSync(old);
  fs.writeFileSync(path.join(old, 'metrics.jsonl'), [357, 358, 359, 360, 361]
    .map(cycle => JSON.stringify({ ...f.row, cycle })).join('\n') + '\n');
  fs.writeFileSync(f.metrics, [976, 977, 978]
    .map(cycle => JSON.stringify({ ...f.row, cycle })).join('\n') + '\n');
  f.write(f.bindingPath, { ...f.binding, segments: [old, f.body] });
  const observed = f.observer.snapshot();
  assert.deepEqual(observed.history.rows.map(row => row.cycle), [976, 977, 978]);
  assert.equal(observed.history.from_cycle, 976);
  assert.equal(observed.history.limited, true);
  assert.equal(fs.readFileSync(path.join(old, 'metrics.jsonl'), 'utf8').split('\n').length, 6);
});

test('continuous cycles still join across explicitly bound segment boundaries', t => {
  const f = fixture(t), old = path.join(f.root, 'previous'); fs.mkdirSync(old);
  fs.writeFileSync(path.join(old, 'metrics.jsonl'), JSON.stringify({ ...f.row, cycle: 4209 }) + '\n');
  f.write(f.bindingPath, { ...f.binding, segments: [old, f.body] });
  assert.deepEqual(f.observer.snapshot().history.rows.map(row => row.cycle), [4209, 4210]);
});

test('HTTP family/life/telemetry use same source and writes cannot invoke a worker', async t => {
  const f = fixture(t), runtime = { branchFile: () => path.join(f.root, 'not-a-branch') };
  const server = createMengliServer(runtime, { family: f.family });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise(resolve => { server.closeAllConnections(); server.close(resolve); }));
  const base = `http://127.0.0.1:${server.address().port}`;
  const state = await (await fetch(base + '/api/family')).json();
  const telemetry = await (await fetch(base + '/api/family/telemetry')).json();
  const life = await (await fetch(base + '/api/life')).json();
  assert.equal(state.cycle, telemetry.source.cycle);
  assert.deepEqual(life.card, telemetry.life.card);
  for (const route of ['load', 'control', 'step', 'say', 'plan']) {
    const response = await fetch(base + '/api/family/' + route, { method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ mode: 'mother', text: 'not authorized to drive observer' }) });
    assert.equal(response.status, 409);
  }
  assert.deepEqual(f.calls, []);
});

test('client recognizes observer state and fresh SSE advances cards, curves and shared-question rows', async t => {
  const f = fixture(t); let notify;
  let state = f.family.inspect();
  const controller = { getSnapshot: () => ({ connected: true, state }), subscribe: fn => { notify = fn; return () => {}; } };
  const client = createMetricsClient({ getBridge: () => ({ controller }), fetcher: async () => ({ ok: true,
    json: async () => familyTelemetry(f.family, null) }) });
  await client.open();
  assert.equal(lifeStatus(client.getSnapshot()), '训练中 · 只读观察');
  fs.appendFileSync(f.metrics, JSON.stringify({ ...f.row, cycle: 4211, shared_unknown: { identity: 'unknown:1', recalled: true } }) + '\n');
  state = f.family.inspect(); notify();
  assert.equal(client.getSnapshot().source.cycle, 4211);
  assert.equal(client.getSnapshot().rows.at(-1).shared_unknown.recalled, true);
  assert.equal(client.getSnapshot().card.source.cycle, 4211);
  client.dispose();
});

test('a transient wrong card fails closed then recovers from the same saved event without reloading', async t => {
  const f = fixture(t); let notify;
  f.write(f.cardPath, { ...f.card, source: { ...f.card.source, checkpoint: path.join(f.root, 'other.pt.gz') } });
  let state = f.family.inspect();
  const controller = { getSnapshot: () => ({ connected: true, state }),
    subscribe: fn => { notify = fn; return () => {}; } };
  const client = createMetricsClient({ getBridge: () => ({ controller }), fetcher: async () => ({ ok: true,
    json: async () => familyTelemetry(f.family, null) }) });
  await client.open();
  assert.match(client.getSnapshot().error, /Life card does not match/);
  assert.equal(client.getSnapshot().card, null);
  assert.equal(client.getSnapshot().rows.length, 0);
  f.write(f.cardPath, f.card);
  state = f.family.inspect(); notify();
  const restored = client.getSnapshot();
  assert.equal(restored.error, null);
  assert.equal(restored.source.branch_id, f.binding.branch_id);
  assert.equal(restored.source.checkpoint, path.join(f.body, 'last.pt.gz'));
  assert.equal(restored.source.cycle, f.row.cycle);
  assert.equal(restored.card.source.cycle, f.card.source.cycle);
  assert.equal(restored.rows.length, 1);
  assert.deepEqual(f.calls, []);
  client.dispose();
});

test('an older failed HTTP response cannot overwrite recovered observer SSE', async t => {
  const f = fixture(t); let notify, reject;
  let state = { read_only: true, error: 'Life card does not match selected body' };
  const controller = { getSnapshot: () => ({ connected: true, state }),
    subscribe: fn => { notify = fn; return () => {}; } };
  const client = createMetricsClient({ getBridge: () => ({ controller }),
    fetcher: () => new Promise((_, failed) => { reject = failed; }) });
  const opened = client.open();
  state = f.family.inspect(); notify();
  reject(new Error('Earlier request failed'));
  await opened;
  assert.equal(client.getSnapshot().error, null);
  assert.equal(client.getSnapshot().source.cycle, f.row.cycle);
  assert.equal(client.getSnapshot().rows.length, 1);
  client.dispose();
});
