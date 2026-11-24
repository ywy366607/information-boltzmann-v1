import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import { FamilyRuntime, BodyWorker } from '../scripts/ib_local/mengli_harness/family_runtime.mjs';
import { createDshRuntime, HARNESS_ROOT } from '../scripts/ib_local/mengli_harness/dsh_runtime.mjs';
import { createMengliServer } from '../scripts/ib_local/mengli_harness/mengli_service.mjs';

function fixture(t, hook) {
  const root = path.join(HARNESS_ROOT, `family-test-${randomUUID()}`);
  fs.mkdirSync(root, { recursive: true });
  const lifeCardPath = path.join(root, 'life.json');
  fs.writeFileSync(lifeCardPath, JSON.stringify({ identity: { individual_id: 'offline-fixture' }, branch_id: 'fixture',
    source: { checkpoint: 'fixture-only-not-a-real-body', cycle: 10 } }));
  const requests = [];
  let cycle = 10;
  const worker = { request: async (command, value) => {
    requests.push({ command, ...value });
    if (command === 'load') return { cycle, branch_id: 'fixture', checkpoint: path.join(root, 'body/last.pt.gz'), response_bytes: 128 };
    await hook?.(value);
    cycle++;
    return { cycle, branch_id: 'fixture', checkpoint: path.join(root, 'body/last.pt.gz'), caregiver: value.caregiver,
      utterance: { text: '', spoke: false, timing: 'before_caregiver_input' }, caregiver_text: value.text ?? 'Fixture mother',
      delivered: { text: value.text ?? 'Fixture mother', byte_end: 12, candidate_bytes: 12 }, plan_version: value.plan.version, choice: 'silence', updates: 1 };
  }, close: async () => {} };
  const family = new FamilyRuntime({ dataRoot: root, lifeCardPath, worker });
  t.after(async () => { await family.close(); fs.rmSync(root, { recursive: true, force: true }); });
  return { root, family, requests, lifeCardPath, worker };
}

test('startup binds saved individual, is paused, and does not load or advance a body', t => {
  const { family, requests } = fixture(t);
  assert.equal(family.inspect().cycle, 10);
  assert.equal(family.inspect().mode, 'paused');
  assert.equal(family.inspect().loaded, false);
  assert.equal(requests.length, 0);
});

test('plan editing does not step; explicit parent delivery commits once and preserves identity after restart', async t => {
  const f = fixture(t);
  f.family.updatePlan({ text: 'Fixture parenting intention', version: 0 });
  assert.equal(f.requests.length, 0);
  await f.family.load('http://127.0.0.1:1');
  await f.family.control('parent');
  const result = await f.family.say({ text: 'Actual parent fixture' });
  assert.equal(result.busy, false);
  assert.equal(result.pending_input, null);
  assert.equal(result.events[0].actor, 'parent');
  assert.equal(result.events[0].utterance.timing, 'before_caregiver_input');
  assert.deepEqual(f.requests.at(-1).plan, { version: 1, text: 'Fixture parenting intention' });
  assert.equal(result.plan.effective_version, 1);
  const restarted = new FamilyRuntime({ dataRoot: f.root, lifeCardPath: f.lifeCardPath, worker: f.worker });
  assert.equal(restarted.inspect().cycle, 11);
  assert.equal(restarted.inspect().mode, 'paused');
  assert.equal(restarted.inspect().events.length, 1);
  assert.equal(f.requests.filter(value => value.command === 'step').length, 1);
});

test('pause during a body cycle closes admission immediately and waits for the saved boundary', async t => {
  let release, started;
  const waiting = new Promise(resolve => { started = resolve; });
  const f = fixture(t, async () => { started(); await new Promise(resolve => { release = resolve; }); });
  await f.family.load('http://127.0.0.1:1');
  await f.family.control('mother');
  const run = f.family.step({ cycles: 4 });
  await waiting;
  const state = await f.family.control('parent');
  assert.equal(state.mode, 'pausing');
  await assert.rejects(f.family.step(), { code: 'CARE_PAUSED' });
  await assert.rejects(f.family.say({ text: 'Too early' }), { code: 'NOT_PARENT' });
  release();
  const done = await run;
  assert.equal(done.mode, 'parent');
  assert.equal(done.busy, false);
  assert.equal(done.committed.length, 1);
});

test('plan changes during a cycle apply on the next complete cycle', async t => {
  let release, started;
  const waiting = new Promise(resolve => { started = resolve; });
  let first = true;
  const f = fixture(t, async () => {
    if (first) { first = false; started(); await new Promise(resolve => { release = resolve; }); }
  });
  await f.family.load('http://127.0.0.1:1');
  await f.family.control('mother');
  const run = f.family.step();
  await waiting;
  f.family.updatePlan({ text: 'new version', version: 0 });
  release();
  await run;
  assert.equal(f.family.inspect().plan.effective_version, 0);
  await f.family.step();
  assert.equal(f.family.inspect().plan.effective_version, 1);
  assert.equal(f.requests.at(-1).plan.text, 'new version');
});

test('management pause/resume does not share the body request queue', async t => {
  const f = fixture(t);
  const controls = [];
  f.family.setWorkbench({ pauseCare: async () => controls.push('pause'), resumeCare: async () => controls.push('resume') }, { origin: 'fixture' });
  await f.family.control('parent');
  await f.family.control('mother');
  assert.deepEqual(controls, ['pause', 'resume']);
  assert.equal(f.requests.length, 0);
});

test('real Python protocol can start and close without loading torch or the body', async () => {
  const worker = new BodyWorker();
  assert.deepEqual(await worker.request('close'), { closed: true, cycle: null });
});

test('official caregiver context is append-only, external, restorable, and not a fabricated assistant', async t => {
  const root = path.join(HARNESS_ROOT, `family-context-test-${randomUUID()}`);
  const wires = [];
  const runtime = await createDshRuntime({ dataRoot: root, fetchImpl: async (_url, options) => {
    wires.push(JSON.parse(options.body));
    return { ok: true, json: async () => ({ choices: [{ finish_reason: 'stop', message: { content: 'Fixture reply' } }] }) };
  } });
  t.after(async () => { await runtime.close(); fs.rmSync(root, { recursive: true, force: true }); });
  const initial = await runtime.restore({ branch_id: 'main', migration: { system_prompt: 'Offline caregiver fixture', turns: [] } });
  const context = await runtime.context({ branch_id: 'main', checkpoint_ref: initial.checkpoint_ref,
    plan: { version: 1, text: 'Recorded plan fixture' }, observations: [{ event_id: 'parent-1', caregiver: 'parent', caregiver_text: 'Actual parent fixture' }] });
  assert.equal(wires.length, 0);
  assert.equal((await runtime.inspect('main')).turns.length, 0);
  await runtime.restore({ branch_id: 'main', checkpoint_ref: context.checkpoint_ref });
  process.env.DEEPSEEK_API_KEY ||= 'offline-test-key';
  const reply = await runtime.prepare({ branch_id: 'main', checkpoint_ref: context.checkpoint_ref, step_id: 'one', transcript: 'Actual child fixture' });
  assert.equal(wires.length, 1);
  assert.match(wires[0].messages[1].content, /Actual parent fixture/);
  assert.equal(wires[0].messages[1].role, 'user');
  assert.equal(wires[0].messages.at(-1).content, 'Actual child fixture');
  await runtime.commit({ branch_id: 'main', step_id: 'one' });
  assert.equal((await runtime.inspect('main', initial.checkpoint_ref)).turns.length, 0,
    'The body-bound older ref must not expose a caregiver commit before the complete body save');
  const events = await runtime.events(reply.result_ref.session_id);
  assert.equal(events.filter(event => event.type === 'assistant/message').length, 1);
  assert.equal(events.find(event => event.type === 'user/message' && event.data.source?.kind === 'mengli-child').data.content[0].text, 'Actual child fixture');
});

test('family state exposes the same inherited dialogue as the child view without starting care', async t => {
  const f = fixture(t);
  const historyRoot = path.join(HARNESS_ROOT, `history-read-test-${randomUUID()}`);
  const runtime = await createDshRuntime({ dataRoot: historyRoot, fetchImpl: async () => { throw new Error('No provider call permitted'); } });
  const initial = await runtime.restore({ branch_id: 'saved-care', migration: { system_prompt: 'Offline mother',
    turns: [{ input: 'Child saved words', raw: 'Mother saved words', delivered: 'Mother saved words' }] } });
  Object.assign(f.family.state, { source_branch_id: 'saved-care', source_mother_checkpoint_ref: initial.checkpoint_ref });
  const server = createMengliServer(runtime, { family: f.family, lifeCardPath: f.lifeCardPath });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(async () => { server.closeAllConnections(); server.close(); await runtime.close(); fs.rmSync(historyRoot, { recursive: true, force: true }); });
  const origin = `http://127.0.0.1:${server.address().port}`;
  const state = await (await fetch(origin + '/api/family')).json();
  assert.deepEqual(state.events, []);
  assert.equal(state.cursor, 0);
  assert.equal(state.recent_care_history.total_messages, 2);
  assert.equal(state.recent_care_history.recorded_pairs, 1);
  assert.equal(state.recent_care_history.turns[0].actor, 'mengli');
  assert.equal(state.recent_care_history.turns[1].content, 'Mother saved words');
  const history = await (await fetch(origin + '/api/family/history?limit=1')).json();
  assert.equal(history.turns[0].content, 'Mother saved words');
  assert.equal(history.next_before, 1);
  assert.equal((await fetch(origin + '/api/family/history?limit=999')).status, 400);
  assert.equal(f.requests.length, 0);
});

test('family HTTP interface exposes state and rejects cross-origin writes', async t => {
  const f = fixture(t);
  const server = createMengliServer({ branchFile: () => '/not-a-branch' }, { family: f.family });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => { server.closeAllConnections(); server.close(); });
  const origin = `http://127.0.0.1:${server.address().port}`;
  assert.equal((await (await fetch(origin + '/api/family')).json()).mode, 'paused');
  const abort = new AbortController();
  const events = await fetch(origin + '/api/family/events', { signal: abort.signal });
  assert.match(events.headers.get('content-type'), /text\/event-stream/);
  const reader = events.body.getReader();
  const initial = new TextDecoder().decode((await reader.read()).value);
  assert.match(initial, /event: family/);
  assert.match(initial, /"cycle":10/);
  const denied = await fetch(origin + '/api/family/plan', { method: 'POST',
    headers: { origin: 'http://untrusted.invalid', 'content-type': 'application/json' }, body: JSON.stringify({ text: 'No' }) });
  assert.equal(denied.status, 403);
  const result = await fetch(origin + '/api/family/plan', { method: 'POST',
    headers: { 'content-type': 'application/json' }, body: JSON.stringify({ text: 'Plan' }) });
  assert.equal(result.status, 200);
  assert.equal((await result.json()).plan.version, 1);
  const updated = new TextDecoder().decode((await reader.read()).value);
  assert.match(updated, /"version":1/);
  abort.abort();
  await reader.cancel().catch(() => {});
  assert.equal(f.requests.length, 0);
});
