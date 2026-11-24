import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { createDshRuntime, HARNESS_ROOT, REPO_ROOT } from '../scripts/ib_local/mengli_harness/dsh_runtime.mjs';
import { createMengliServer } from '../scripts/ib_local/mengli_harness/mengli_service.mjs';
import { normalizeUsage, publicUsage } from '../scripts/ib_local/mengli_harness/deepseek_chat_adapter.mjs';

// No network: even the real official loop and summarizer use this fake HTTP transport.
process.env.DEEPSEEK_API_KEY = 'offline-test-key';
const persona = 'Caregiver offline interface fixture; no real child experiences are claimed.';
function fakeTransport({ finish = 'stop', reply = '  A gentle reply.\n' } = {}) {
  const calls = [];
  return { calls, fetchImpl: async (url, options) => {
    assert.equal(url, 'https://api.deepseek.com/chat/completions');
    const body = JSON.parse(options.body);
    calls.push(body);
    const summary = body.max_tokens === 512;
    return { ok: true, json: async () => ({ choices: [{ finish_reason: finish,
      message: { content: summary ? '## Shared experiences\n- Preserve the recorded conversation; ambiguous sounds remain uncertain.' : reply } }],
      usage: { prompt_tokens: 100, completion_tokens: summary ? 21 : 7, prompt_cache_hit_tokens: 60 } }) };
  } };
}
async function fixture(t, transport = fakeTransport()) {
  const dataRoot = path.join(HARNESS_ROOT, `node-test-${randomUUID()}`);
  let runtime = await createDshRuntime({ dataRoot, fetchImpl: transport.fetchImpl });
  t.after(async () => { await runtime?.close(); fs.rmSync(dataRoot, { recursive: true, force: true }); });
  return { transport, dataRoot, get runtime() { return runtime; }, restart: async () => {
    await runtime.close(); runtime = null;
    runtime = await createDshRuntime({ dataRoot, fetchImpl: transport.fetchImpl });
    return runtime;
  } };
}
function restore(branch_id = 'main', turns = [], settings = {}) {
  return { branch_id, settings, migration: { system_prompt: persona, turns, provenance: { source_checkpoint: 'offline-fixture', cycle: 0 } } };
}
async function turn(runtime, branch_id, checkpoint_ref, step_id, transcript) {
  const prepared = await runtime.prepare({ branch_id, checkpoint_ref, step_id, transcript });
  const committed = await runtime.commit({ branch_id, step_id });
  return { prepared, ...committed };
}

test('real official loop/persistence: pending survives restart, raw preserved, commit is idempotent', async t => {
  const f = await fixture(t);
  let r = f.runtime;
  const base = (await r.restore(restore())).checkpoint_ref;
  const request = { branch_id: 'main', checkpoint_ref: base, step_id: 'step-one', transcript: 'A sound?' };
  const [a, b] = await Promise.all([r.prepare(request), r.prepare(request)]);
  assert.deepEqual(a, b);
  assert.equal(a.raw, '  A gentle reply.\n');
  assert.equal(a.delivered, 'A gentle reply.');
  assert.equal(f.transport.calls.length, 1);
  assert.deepEqual(f.transport.calls[0], { model: 'deepseek-flash', messages: [
    { role: 'system', content: persona }, { role: 'user', content: 'A sound?' },
  ], temperature: 0.7, max_tokens: 120, stream: false, thinking: { type: 'disabled' } });
  assert.equal((await r.inspect('main')).turns.length, 0);
  await assert.rejects(r.prepare({ ...request, step_id: 'different' }), { code: 'PENDING_CONFLICT' });
  await assert.rejects(r.compactNow('main', base), { code: 'PENDING_CONFLICT' });
  r = await f.restart();
  assert.deepEqual((await r.restore({ branch_id: 'main', checkpoint_ref: base })).checkpoint_ref, base);
  assert.deepEqual(await r.prepare(request), a);
  const committed = await r.commit({ branch_id: 'main', step_id: 'step-one' });
  assert.deepEqual(await r.commit({ branch_id: 'main', step_id: 'step-one' }), committed);
  assert.equal(f.transport.calls.length, 1);
  assert.equal(committed.checkpoint_ref.turn_end_seq, committed.checkpoint_ref.result_seq);
  const events = await r.events(committed.checkpoint_ref.session_id);
  assert.equal(events[committed.checkpoint_ref.turn_end_seq].type, 'turn/end');
  assert(events.slice(committed.checkpoint_ref.turn_end_seq + 1).every(event => event.type === 'session/end-seed'));
  assert.equal(events.find(event => event.type === 'assistant/message').data.message.content[0].text, a.raw);
  assert.equal((await r.inspect('main')).turns.length, 2);
  assert.equal((await r.inspect('main')).usage_totals.reply_usage.prompt_tokens, 100);
  assert.equal(committed.telemetry.api_usage.cache_miss_tokens, 40);
});

test('official prefix fork excludes future, retains lineage, rejects mismatched checkpoint versions', async t => {
  const f = await fixture(t);
  const r = f.runtime;
  const initial = (await r.restore(restore())).checkpoint_ref;
  const old = (await turn(r, 'main', initial, 'one', 'past input')).checkpoint_ref;
  const future = (await turn(r, 'main', old, 'two', 'FUTURE MUST NOT BE READ')).checkpoint_ref;
  const branch = (await r.restore({ branch_id: 'research', checkpoint_ref: old })).checkpoint_ref;
  assert.notEqual(branch.session_id, old.session_id);
  assert.equal(branch.assistant_seq, old.assistant_seq);
  const pending = await r.prepare({ branch_id: 'research', checkpoint_ref: branch, step_id: 'next', transcript: 'new input' });
  assert.equal(f.transport.calls.at(-1).messages.some(message => message.content.includes('FUTURE MUST NOT BE READ')), false);
  assert.equal((await r.events(future.session_id)).some(event => JSON.stringify(event).includes('FUTURE MUST NOT BE READ')), true);
  const handle = await r.ctx.sessionPersistence.open(branch.session_id, 'read');
  assert.equal(handle.header.parentSession, old.session_id);
  assert.equal(handle.header.isSeeded, true);
  assert.equal(handle.inheritedEventCount, old.result_seq + 1);
  await handle.close();
  assert.equal(pending.source, 'api');
  await assert.rejects(r.restore({ branch_id: 'bad-ref', checkpoint_ref: { ...old, summary_version: 99 } }), { code: 'INVALID_CHECKPOINT' });
});

test('official Basic caregiver summarize hook: forced compact, persistent restore, identical next wire/cache', async t => {
  const f = await fixture(t);
  let r = f.runtime;
  const turns = Array.from({ length: 20 }, (_, i) => ({
    input: `fixture shared thread ${i}: ${'garden '.repeat(35)}`,
    raw: `  fixture actual raw ${i}: ${'tree '.repeat(45)}\n`,
    delivered: `fixture actual raw ${i}: ${'tree '.repeat(45)}`.trim(),
  }));
  const migrated = await r.restore(restore('main', turns));
  const compacted = await r.compactNow('main', migrated.checkpoint_ref);
  assert.equal(compacted.compacted, true);
  assert.equal(compacted.summary_version, 1);
  assert.equal(f.transport.calls.length, 1);
  assert.equal(f.transport.calls[0].max_tokens, 512);
  assert.match(f.transport.calls[0].messages.at(-1).content, /caregiver memory/);
  assert.equal(compacted.telemetry.summary_usage.length, 1);
  assert.equal(compacted.telemetry.api_usage, null);
  const events = await r.events(compacted.checkpoint_ref.session_id);
  assert(events.some(event => event.type === 'compaction/summary' && event.data.llmStreamCall === true));
  assert(events.some(event => event.surfaceOp?.op === 'replace'));
  assert.equal(events.at(-1).type, 'compaction/end');
  assert.equal(compacted.checkpoint_ref.result_seq, events.at(-1).seq);
  const next = await r.prepare({ branch_id: 'main', checkpoint_ref: compacted.checkpoint_ref, step_id: 'next', transcript: 'continue recorded thread' });
  const expectedWire = f.transport.calls.at(-1);
  r = await f.restart();
  const fork = await r.restore({ branch_id: 'after-restart', checkpoint_ref: compacted.checkpoint_ref });
  const restored = await r.prepare({ branch_id: 'after-restart', checkpoint_ref: fork.checkpoint_ref, step_id: 'next', transcript: 'continue recorded thread' });
  assert.equal(restored.telemetry.request_key, next.telemetry.request_key);
  assert.equal(restored.source, 'shared');
  assert.equal(restored.usage, null);
  assert.equal(restored.raw, next.raw);
  assert.equal(f.transport.calls.length, 2);
  assert(expectedWire.messages.some(message => message.content.includes('Preserve the recorded conversation')));
  assert(expectedWire.messages.some(message => message.content.includes('fixture actual raw 19')));
  assert((await r.inspect('after-restart')).summary.includes('Shared experiences'));
});

test('official AgentLoop automatically compacts token pressure before the next normal reply', async t => {
  const f = await fixture(t);
  const r = f.runtime;
  const migration = Array.from({ length: 12 }, (_, i) => ({
    input: `fixture ${i}: ${'interface child '.repeat(40)}`,
    raw: `fixture ${i}: ${'interface caregiver '.repeat(40)}`,
    delivered: `fixture ${i}: ${'interface caregiver '.repeat(40)}`.trim(),
  }));
  const base = (await r.restore(restore('auto', migration))).checkpoint_ref;
  const full = await turn(r, 'auto', base, 'fill', 'x '.repeat(4400));
  assert(full.telemetry.meter.total_tokens > 6553);
  assert.equal(f.transport.calls.filter(call => call.max_tokens === 512).length, 0);
  const before = await r.events(full.checkpoint_ref.session_id);
  const next = await turn(r, 'auto', full.checkpoint_ref, 'after-pressure', 'next interface input');
  assert.equal(f.transport.calls.filter(call => call.max_tokens === 512).length, 1);
  assert.equal(next.checkpoint_ref.summary_version, 1);
  assert.equal(next.telemetry.summary_usage.length, 1);
  assert.equal(next.telemetry.summary_usage[0].source, 'api');
  assert.equal(next.usage_totals.summary_usage.completion_tokens, 21);
  assert.equal(next.usage_totals.api_calls, 3);
  assert(next.prepared.delivered.length > 0);
  const after = await r.events(next.checkpoint_ref.session_id);
  assert.deepEqual(after.slice(0, before.length), before);
  assert.equal(after.filter(event => event.type === 'compaction/summary').length, 1);
  assert(after.some(event => event.surfaceOp?.op === 'replace'));
  assert.equal(after.at(-1).data.reason.kind, 'completed');
});

test('diagnostic retain is scoped, failures never become replies, final heuristic budget blocks dispatch', async t => {
  const f = await fixture(t, fakeTransport({ finish: 'length' }));
  const r = f.runtime;
  const base = (await r.restore(restore('main'))).checkpoint_ref;
  await assert.rejects(r.prepare({ branch_id: 'main', checkpoint_ref: base, step_id: 'bad', transcript: 'test truncation' }), { code: 'MAX_TOKENS' });
  assert.deepEqual((await r.inspect('main')).checkpoint_ref, base);
  await assert.rejects(r.commit({ branch_id: 'main', step_id: 'bad' }), { code: 'NO_PREPARED_RECEIPT' });
  await assert.rejects(r.prepare({ branch_id: 'main', checkpoint_ref: base, step_id: 'bad', transcript: 'test truncation' }), { code: 'MAX_TOKENS' });
  assert.equal(f.transport.calls.length, 1);
  assert.equal((await r.inspect('main')).usage_totals.api_calls, 1);
  assert.equal((await r.inspect('main')).usage_totals.failures, 1);
  const empty = (await r.restore(restore('huge', [], { retain_tokens: 128 }))).checkpoint_ref;
  await assert.rejects(r.prepare({ branch_id: 'huge', checkpoint_ref: empty, step_id: 'huge', transcript: 'x'.repeat(50000) }), { code: 'CONTEXT_BUDGET' });
  assert.equal(f.transport.calls.length, 1);
  assert.equal((await r.inspect('huge')).telemetry.meter.retain_tokens, 128);
  await assert.rejects(createDshRuntime({ dataRoot: path.dirname(HARNESS_ROOT) }), { code: 'DATA_ROOT' });
});

test('HTTP bridge is loopback/same-origin JSON only and reports real state without demo metrics', async t => {
  const f = await fixture(t);
  const server = createMengliServer(f.runtime);
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => { server.closeAllConnections(); return new Promise(resolve => server.close(resolve)); });
  const origin = `http://127.0.0.1:${server.address().port}`;
  const post = (route, body, headers = {}) => fetch(origin + route, { method: 'POST',
    headers: { 'content-type': 'application/json', ...headers }, body: JSON.stringify(body) });
  assert.equal((await fetch(origin + '/health')).status, 200);
  assert.equal((await (await fetch(origin)).json()).service, 'mengli-body-bridge');
  for (const asset of ['intro.css', 'life.css', 'intro.js', 'family.js', 'family.css', 'mengli-mark.svg']) {
    assert.equal((await fetch(`${origin}/${asset}`)).status, 404, 'Retired custom frontend is not mounted');
  }
  assert.equal((await fetch(origin + '/dsh_runtime.mjs')).status, 404);
  assert.equal((await fetch(origin + '/web/intro.css')).status, 404);
  assert.equal((await post('/api/mengli/restore', restore(), { origin: 'https://other.invalid' })).status, 403);
  assert.equal((await post('/api/mengli/restore', restore(), { 'content-type': 'text/plain' })).status, 415);
  const restored = await (await post('/api/mengli/restore', restore())).json();
  const prepared = await (await post('/api/mengli/prepare', { branch_id: 'main', checkpoint_ref: restored.checkpoint_ref, step_id: 'http', transcript: 'input' })).json();
  assert.equal(prepared.raw, '  A gentle reply.\n');
  assert.equal((await (await fetch(origin + '/api/session')).json()).total_turns, 0);
  assert.equal((await post('/api/mengli/commit', { branch_id: 'main', step_id: 'http' })).status, 200);
  assert.equal((await (await fetch(origin + '/api/session')).json()).total_turns, 1);
  const vitals = await (await fetch(origin + '/api/vitals')).json();
  assert.equal(vitals.s_state_norm, null);
  assert.equal(vitals.metadata.training_status, 'unknown');
  assert.equal(vitals.reply_usage.prompt_tokens, 100);
  assert.equal(vitals.source_counts.api, 1);
  assert.equal(vitals.provider_hit_rate, 0.6);
  assert.equal(vitals.api_calls, 1);
});

test('null checkpoint restore returns bootstrap migration prefix, not the advanced branch future', async t => {
  const f = await fixture(t);
  const request = restore('bootstrap', [{ input: 'recorded child', delivered: 'recorded delivered only' }]);
  const initial = await f.runtime.restore(request);
  const advanced = await turn(f.runtime, 'bootstrap', initial.checkpoint_ref, 'future', 'UNCOMMITTED TO INITIAL CHECKPOINT');
  const restored = await f.runtime.restore(request);
  assert.notEqual(restored.checkpoint_ref.session_id, advanced.checkpoint_ref.session_id);
  const inspected = await f.runtime.inspect('bootstrap');
  assert.equal(inspected.turns.length, 2);
  assert(!JSON.stringify(inspected.turns).includes('UNCOMMITTED TO INITIAL CHECKPOINT'));
  assert.deepEqual((await f.runtime.restore(request)).checkpoint_ref, restored.checkpoint_ref);
  const events = await f.runtime.events(restored.checkpoint_ref.session_id);
  assert.equal(events.find(event => event.type === 'assistant/message').data.message.source.replayState.provenance, 'delivered-only');
});

test('repository root is anchored independently and fork usage uses only the supplied checkpoint snapshot', async t => {
  assert.equal(REPO_ROOT, path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..'));
  assert.equal(HARNESS_ROOT, path.join(REPO_ROOT, 'data', 'harness'));
  const f = await fixture(t);
  const base = (await f.runtime.restore(restore())).checkpoint_ref;
  const first = await turn(f.runtime, 'main', base, 'one', 'old known input');
  await turn(f.runtime, 'main', first.checkpoint_ref, 'two', 'later input');
  const restored = await f.runtime.restore({ branch_id: 'accounting-fork', checkpoint_ref: first.checkpoint_ref, usage_totals: first.usage_totals });
  assert.equal(restored.usage_totals.api_calls, 1);
  assert.equal((await f.runtime.inspect('accounting-fork')).usage_totals.api_calls, 1);
});

test('life endpoint reads only an explicitly selected saved card and never chooses a branch', async t => {
  const f = await fixture(t);
  const lifeCardPath = path.join(f.dataRoot, 'life.json');
  const card = { schema: 1, identity: { name: 'offline fixture', individual_id: 'fixture' },
    branch_id: 'fixture-branch', source: { cycle: 0, sha256: 'fixture-only', checkpoint: 'fixture.pt.gz' },
    age: { cycles: 0, byte_events: 0 }, memory: { occupied: 0 } };
  fs.writeFileSync(lifeCardPath, JSON.stringify(card));
  const server = createMengliServer(f.runtime, { lifeCardPath });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => { server.closeAllConnections(); return new Promise(resolve => server.close(resolve)); });
  const origin = `http://127.0.0.1:${server.address().port}`;
  const get = async route => (await fetch(origin + route)).json();
  assert.deepEqual(await get('/api/life'), { card, state: 'saved' });
  const vitals = await get('/api/vitals');
  assert.deepEqual(vitals.life, card);
  assert.equal(vitals.cycle, null); // Saved body facts do not masquerade as live measurements.
  assert.equal(vitals.metadata.arm, null);
  fs.writeFileSync(lifeCardPath, JSON.stringify({ ...card, age: { cycles: 1 } }));
  assert.deepEqual(await get('/api/life'), { card: null, state: 'invalid' });
  fs.unlinkSync(lifeCardPath);
  assert.deepEqual(await get('/api/life'), { card: null, state: 'unbound' });
  assert.equal(f.transport.calls.length, 0);
  assert.throws(() => createMengliServer(f.runtime, { lifeCardPath: path.dirname(HARNESS_ROOT) }), { code: 'LIFE_PATH' });
});

test('usage normalization counts provider cache tokens disjointly, preserving genuine zero and unknown', () => {
  assert.deepEqual(normalizeUsage(null), null);
  assert.deepEqual(publicUsage(normalizeUsage({ prompt_tokens: 0, completion_tokens: 0, prompt_cache_hit_tokens: 0 })), {
    prompt_tokens: 0, completion_tokens: 0, cache_hit_tokens: 0, cache_miss_tokens: 0, total_tokens: 0,
  });
  assert.equal(normalizeUsage({ prompt_tokens: 100, completion_tokens: 2, prompt_cache_hit_tokens: 30 }).inputTokens, 70);
});
