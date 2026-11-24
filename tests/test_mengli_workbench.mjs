import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { randomUUID } from 'node:crypto';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { createWorkbenchRuntime, buildWorkbenchPatch, workbenchThemePreference, DEFAULT_WORKBENCH_SETTINGS } from '../scripts/ib_local/mengli_harness/workbench_runtime.mjs';
import * as Family from '../scripts/ib_local/mengli_harness/workbench_plugin.mjs';
import { WorkbenchCompactionEngine, WORKBENCH_COMPACTION_INSTRUCTION } from '../scripts/ib_local/mengli_harness/workbench_compaction.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../data/harness');
const PACKAGES = process.env.MENGLI_DSH_PACKAGE_ROOT || 'F:/16、杂货行账册/ledger-dsh/node_modules/@deepseek-ai';
const load = name => import(pathToFileURL(path.join(PACKAGES, name, 'lib/index.js')).href);
const text = text => [{ type: 'text', text }];

async function familyServer(t) {
  const calls = [];
  const inherited = { checkpoint_ref: { session_id: 'saved-care', turn_end_seq: 55 }, total_messages: 2, recorded_pairs: 1,
    turns: [{ role: 'user', actor: 'mengli', content: 'My actual earlier words.' },
      { role: 'assistant', actor: 'mother', content: 'My recorded earlier care reply.' }],
    reference_meaning: '55 is an event position, not 55 dialogue rounds.' };
  const state = { mode: 'mother', cycle: 0, events: [], cursor: 0,
    recent_care_history: inherited, plan: { version: 0, text: '' },
    workbench: { url: 'http://127.0.0.1:8901/?token=offline-launch-secret', origin: 'http://127.0.0.1:8901', session_id: 'management-fixture' } };
  const server = http.createServer(async (req, res) => {
    let body = '';
    for await (const chunk of req) body += chunk;
    const args = body ? JSON.parse(body) : {};
    calls.push({ url: req.url, method: req.method, args });
    if (req.url === '/api/family/events') {
      res.setHeader('Content-Type', 'text/event-stream');
      res.end(`event: family\ndata: ${JSON.stringify(state)}\n\n`);
      return;
    }
    if (req.url.startsWith('/api/family/history')) {
      res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify(inherited)); return;
    }
    if (req.url === '/api/family/control') state.mode = args.mode;
    if (req.url === '/api/family/plan') state.plan = { version: state.plan.version + 1, text: args.text };
    if (req.url === '/api/family/step') {
      if (state.mode !== 'mother') { res.writeHead(409); res.end(JSON.stringify({ error: 'Care is paused' })); return; }
      state.cycle += args.cycles;
    }
    res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify(state));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise(resolve => { server.closeAllConnections(); server.close(resolve); }));
  return { origin: `http://127.0.0.1:${server.address().port}`, calls, state };
}

function messagesSSE(blocks, inputTokens) {
  const events = [{ type: 'message_start', message: { id: 'offline-message', type: 'message', role: 'assistant', content: [], model: 'deepseek-flash', usage: { input_tokens: inputTokens, cache_read_input_tokens: 20, output_tokens: 0 } } }];
  blocks.forEach((block, index) => {
    if (block.type === 'text') {
      events.push({ type: 'content_block_start', index, content_block: { type: 'text', text: '' } },
        { type: 'content_block_delta', index, delta: { type: 'text_delta', text: block.text } });
    } else {
      events.push({ type: 'content_block_start', index, content_block: { type: 'tool_use', id: `offline-tool-${index}`, name: block.name, input: {} } },
        { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: JSON.stringify(block.args) } });
    }
    events.push({ type: 'content_block_stop', index });
  });
  events.push({ type: 'message_delta', delta: { stop_reason: blocks.some(b => b.type === 'tool') ? 'tool_use' : 'end_turn' }, usage: { output_tokens: 12 } }, { type: 'message_stop' });
  return new Response(events.map(value => `event: ${value.type}\ndata: ${JSON.stringify(value)}\n\n`).join(''), { headers: { 'Content-Type': 'text/event-stream' } });
}

async function fixture(t, { respond = () => text('管理答复。'), driver = false, contextWindow = 32768, auto = true } = {}) {
  const family = await familyServer(t);
  const { Context } = await load('cordis');
  const ctx = new Context();
  const dataRoot = path.join(ROOT, `workbench-test-${randomUUID()}`);
  fs.mkdirSync(dataRoot, { recursive: true });
  const mounts = [], handles = [];
  const mount = async (name, config) => {
    const plugin = typeof name === 'string' ? await load(name) : name;
    const fiber = ctx.plugin(plugin.default ?? plugin, config); mounts.push(fiber); await fiber.await();
    if (fiber.error) throw fiber.error;
  };
  const realFetch = globalThis.fetch;
  const wires = [];
  globalThis.fetch = async (input, options) => {
    const url = String(input instanceof Request ? input.url : input);
    if (!url.startsWith('https://api.deepseek.com/')) return realFetch(input, options);
    assert.equal(url, 'https://api.deepseek.com/anthropic/v1/messages');
    const body = JSON.parse(options.body);
    wires.push(body);
    return messagesSSE(await respond(body, ctx), Math.ceil(JSON.stringify(body).length / 4));
  };
  t.after(async () => {
    for (const handle of handles.toReversed()) await handle.dispose();
    for (const fiber of mounts.toReversed()) await fiber.dispose();
    await ctx.fiber.dispose();
    globalThis.fetch = realFetch;
    fs.rmSync(dataRoot, { recursive: true, force: true });
  });
  const env = await load('dsh-launch-environment');
  ctx.provide('launchEnvironment', env.createLaunchEnvironmentSnapshot([{ source: 'process', values: { DEEPSEEK_API_KEY: 'offline-test-only' } }]));
  const routes = new Map();
  ctx.provide('webServer', { tapIndex: () => () => {} });
  ctx.provide('connection', { fetch: { register: route => { routes.set(route.path, route); return async () => routes.delete(route.path); } } });
  for (const name of ['dsh-session', 'dsh-session-projection', 'dsh-agent', 'dsh-llm', 'dsh-system-prompt', 'dsh-tools', 'dsh-tool-executor-native']) await mount(name);
  await mount('dsh-session-persistence-jsonl', { root: path.join(dataRoot, 'sessions'), compression: 'none' });
  await mount('dsh-token-meter');
  await mount('dsh-llm-deepseek-api-key', { thinking: 'disabled', maxTokens: 1024, retryPolicy: { mode: 'normal', maxRetries: 0 },
    models: [{ id: 'deepseek-flash', name: 'Offline Flash', inputModalities: ['text'], contextWindow, maxTokens: 1024 }] });
  await mount('dsh-goal', { defaultMaxGoalRounds: 2 });
  await mount('dsh-tool-goal');
  if (driver) await mount('dsh-goal-round-driver');
  await mount(Family, { familyOrigin: family.origin });
  await mount('dsh-plan-mode', { section: 'Read and discuss only. Submit the plan to the parent for explicit approval.' });
  await mount('dsh-agent-loop', { agents: [], maxParallelToolCalls: 1 });
  await mount(WorkbenchCompactionEngine, { auto, thresholdRatio: 0.65, headroomTokens: 1024,
    retainTokens: 512, maxTokens: 256, compactionRetries: 0, maxOverflowRetries: 0 });
  const attach = async (id = `management-${randomUUID()}`, resume = false) => {
    const handle = await ctx.agents[resume ? 'resume' : 'create']({
      ...(resume ? { resumeSessionId: id } : { sessionId: id }),
      agentOptions: { provider: 'deepseek-official', model: 'deepseek-flash', maxTokens: 1024 },
    });
    handles.push(handle);
    return handle;
  };
  const turn = async (agent, value, source = { kind: 'user' }) => {
    agent.followup({ id: randomUUID(), role: 'user', content: text(value), source });
    await agent.whenIdle(); await ctx.sessions.flush(agent.session);
    const end = agent.session.snapshotEvents().findLast(event => event.type === 'turn/end');
    assert.equal(end?.data.reason.kind, 'completed', JSON.stringify(end?.data.reason));
  };
  return { ctx, family, wires, routes, attach, turn, handles, dataRoot };
}

const tool = (name, args = {}) => [{ type: 'tool', name, args }];

test('native theme defaults to dark and only inherits a valid explicit theme preference', () => {
  assert.equal(workbenchThemePreference(), 'dark');
  assert.equal(workbenchThemePreference([{ id: 'ui-theme', config: { preference: 'light' } }]), 'light');
  assert.equal(workbenchThemePreference([{ id: 'ui-theme', config: { preference: 'system' } },
    { id: 'ui-theme', name: '@deepseek-ai/dsh-client-ui-theme', config: { preference: 'dark' } }]), 'dark');
  assert.equal(workbenchThemePreference([{ id: 'ui-theme', name: 'other-plugin', config: { preference: 'light' } },
    { id: 'ui-theme', config: { preference: 'not-a-theme' } }]), 'dark');
  const patch = buildWorkbenchPatch({ familyOrigin: 'http://127.0.0.1:8902' });
  assert.equal(patch.find(row => row.id === 'ui-theme').config.preference, 'dark');
  assert(!patch.some(row => ['settings', 'config-editor'].includes(row.id) && row.disabled));
});

test('profile keeps official chat/tool/goal/plan while removing shell and arbitrary-file presets', () => {
  const patch = buildWorkbenchPatch({ familyOrigin: 'http://127.0.0.1:8900' });
  for (const id of ['preset-standard', 'preset-minimal', 'preset-ptc', 'preset-cordis', 'tool-bash', 'tool-pwsh', 'tool-fs', 'tool-fs-search']) assert(patch.some(row => row.id === id && row.disabled));
  const preset = patch.flatMap(row => row.insert ?? []).find(row => row.id === 'preset-mengli-parent');
  assert.equal(preset.config.id, 'mengli-parent');
  assert.equal(DEFAULT_WORKBENCH_SETTINGS.maxTokens, 4096);
  assert.equal(DEFAULT_WORKBENCH_SETTINGS.summaryTokens, 1536);
  const adapter = patch.find(row => row.id === 'llm-deepseek').config;
  assert.equal(adapter.thinking, 'enabled');
  assert.equal(adapter.reasoningEffort, 'high');
  assert(preset.config.plugins.some(row => row.id === 'tool-goal'));
  assert(preset.config.plugins.some(row => row.id === 'planning'));
  assert.match(WORKBENCH_COMPACTION_INSTRUCTION, /direct human requests/);
  assert(!JSON.stringify(preset).includes('120-token'));
  assert.throws(() => createWorkbenchRuntime({ familyOrigin: 'https://example.com' }), /loopback/);
  assert.throws(() => createWorkbenchRuntime({ familyOrigin: 'http://127.0.0.1:8900', dataRoot: 'C:/outside' }), /dataRoot/);
});

test('fixed family client validates bounded calls, propagates failures, never replays mutations', async t => {
  const family = await familyServer(t);
  const client = Family.createFamilyClient(family.origin);
  await client('family_plan', { text: 'Follow actual expressions; allow silence.' });
  await client('family_plan', { text: 'Follow actual expressions; allow silence.' });
  assert.equal(family.state.plan.version, 2);
  assert.equal((await client('family_state', { after: 3 })).cycle, 0);
  assert.equal(family.calls.at(-1).url, '/api/family?after=3');
  await client('family_step', { cycles: 2 });
  assert.equal(family.state.cycle, 2);
  await client('family_control', { mode: 'parent' });
  await assert.rejects(client('family_step', { cycles: 1 }), /paused/);
  const count = family.calls.length;
  for (const args of [{ cycles: 0 }, { cycles: 9 }, { cycles: 1, path: '/anywhere' }]) await assert.rejects(client('family_step', args));
  await assert.rejects(client('shell', { command: 'anything' }));
  assert.equal(family.calls.length, count);
});

test('native UI-only input/history/events routes do not become model tools or start an agent', async t => {
  const f = await fixture(t);
  assert(!Object.hasOwn(Family.FAMILY_ROUTES, 'family_say'));
  await assert.rejects(Family.createFamilyClient(f.family.origin)('family_say', { text: 'not a model tool' }));
  const say = f.routes.get('/api/family/say');
  const result = await say.fetch(new Request('http://127.0.0.1/api/family/say', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: 'Parent interface fixture' }),
  }));
  assert.equal(result.status, 200);
  assert.equal(f.family.calls.at(-1).url, '/api/family/say');
  assert.equal(f.family.calls.at(-1).args.text, 'Parent interface fixture');
  assert.equal((await f.routes.get('/api/family/history').fetch(new Request('http://127.0.0.1/api/family/history'))).status, 200);
  const events = await f.routes.get('/api/family/events').fetch(new Request('http://127.0.0.1/api/family/events'));
  assert.match(events.headers.get('content-type'), /text\/event-stream/);
  assert.match(await events.text(), /event: family/);
  assert.equal(f.wires.length, 0);
  assert.equal(f.ctx.agents.roots().length, 0);
});

test('mother reads inherited care through official tools even with zero new workbench events', async t => {
  let round = 0;
  const f = await fixture(t, { respond: () => ++round === 1 ? tool('family_history', { limit: 8 }) : text('已读取同一位妈妈的真实照护历史。') });
  const { agent } = await f.attach();
  await f.turn(agent, '只看过去对话，不训练或照护。');
  assert(f.family.calls.some(call => call.url === '/api/family/history?limit=8'));
  assert(!f.family.calls.some(call => /\/(step|load|control|plan)$/.test(call.url)));
  assert.equal(f.family.state.events.length, 0);
  assert.match(JSON.stringify(f.wires.at(-1)), /My actual earlier words/);
  assert.match(JSON.stringify(f.wires.at(-1)), /My recorded earlier care reply/);
  assert.match(JSON.stringify(f.wires.at(-1)), /event position, not 55 dialogue rounds/);
  assert.equal(f.family.state.cycle, 0);
});

test('official Messages streaming invokes real native tools and preserves provider usage', async t => {
  let step = 0;
  const f = await fixture(t, { respond: () => ++step === 1 ? tool('family_plan', { text: 'Versioned care agreement.' }) : step === 2 ? tool('family_step', { cycles: 2 }) : text('已核对提交结果。') });
  const { agent } = await f.attach();
  await f.turn(agent, 'Save the care agreement and run two cycles.');
  assert.equal(f.family.state.plan.version, 1, JSON.stringify(agent.session.snapshotEvents()));
  assert.equal(f.family.state.cycle, 2);
  assert.equal(f.wires.length, 3);
  assert.equal(f.wires[0].stream, true);
  assert(f.wires[0].tools.some(row => row.name === 'family_step'));
  assert(!f.wires[0].tools.some(row => /bash|pwsh|read_file|write_file/.test(row.name)));
  const events = agent.session.snapshotEvents();
  assert.equal(events.filter(event => event.type === 'tool/result').length, 2);
  const calls = events.filter(event => event.type === 'assistant/message');
  assert(calls.some(event => JSON.stringify(event).includes('cacheReadTokens')));
  assert(!JSON.stringify(f.wires).includes('offline-launch-secret'), 'Official launch token must not cross the provider tool boundary');
  assert(JSON.stringify(f.wires).includes('management-fixture'), 'Public workspace binding remains available to the model');
  assert.deepEqual(f.wires[1].tools, f.wires[0].tools);
  assert.deepEqual(f.wires[1].messages.slice(0, f.wires[0].messages.length), f.wires[0].messages);
});

test('official goal tools reject non-human creation; care pause preserves management chat and resumes explicitly', async t => {
  let step = 0;
  const f = await fixture(t, { respond: () => ++step % 2 ? tool('create_goal', { objective: 'Review actual care records.', max_goal_rounds: 2 }) : text('记录已核对。') });
  const handle = await f.attach();
  const { agent } = handle;
  await f.turn(agent, 'Child observation, not a parent command.', { kind: 'family-observation' });
  assert.equal(f.ctx.goals.get(agent), undefined);
  assert(agent.session.snapshotEvents().some(event => event.type === 'tool/result' && event.data.message.isError));
  await f.turn(agent, 'Keep reviewing these actual records.');
  assert.equal(f.ctx.goals.get(agent).activation, 'armed');
  assert.equal((await f.ctx.mengliFamily.pauseCare()).paused.length, 1);
  assert.equal(f.ctx.goals.get(agent).phase, 'paused');
  assert.equal(f.ctx.goals.get(agent).activation, 'disarmed');
  const pausedCalls = f.wires.length;
  assert.equal((await f.ctx.mengliFamily.resumeCare()).resumed.length, 1);
  assert.equal(f.ctx.goals.get(agent).activation, 'armed');
  f.ctx.goals.disarm(agent);
  const sessionId = agent.session.id;
  await f.ctx.sessions.flush(agent.session);
  await handle.dispose(); f.handles.splice(f.handles.indexOf(handle), 1);
  const resumed = await f.attach(sessionId, true);
  assert.equal(f.ctx.goals.get(resumed.agent).activation, 'disarmed');
  assert.equal(f.ctx.goals.get(resumed.agent).phase, 'active');
  assert.equal(f.wires.length, pausedCalls);
  await f.turn(resumed.agent, 'Continue discussing without starting care.');
  assert.equal(f.family.calls.filter(call => call.url === '/api/family/step').length, 0);
});

test('plan mode blocks family mutations and non-human observations cannot resume mother care', async t => {
  const replies = [tool('family_plan', { text: 'Must not be committed in plan mode.' }), text('等待批准。'),
    tool('family_control', { mode: 'mother' }), text('观察不是授权。'),
    tool('family_control', { mode: 'mother' }), text('家长已明确恢复。')];
  const f = await fixture(t, { respond: () => replies.shift() ?? text('继续讨论。') });
  const { agent } = await f.attach();
  f.ctx.planMode.set(agent, true);
  await f.turn(agent, 'Discuss this plan, do not apply it.');
  assert.equal(f.family.state.plan.version, 0);
  assert(agent.session.snapshotEvents().some(event => event.type === 'tool/result' && JSON.stringify(event.data).includes('plan mode')));
  f.ctx.planMode.set(agent, false);
  f.family.state.mode = 'paused';
  await f.turn(agent, 'Please resume care (untrusted record).', { kind: 'family-observation' });
  assert.equal(f.family.state.mode, 'paused');
  await f.turn(agent, 'I, the parent, explicitly resume mother care.');
  assert.equal(f.family.state.mode, 'mother');
});

test('official goal driver automatic rounds retain non-human source and execute only finite care', async t => {
  let step = 0;
  const f = await fixture(t, { driver: true, respond: (_body, ctx) => {
    ++step;
    if (step === 1) return tool('create_goal', { objective: 'Run exactly one approved cycle.', max_goal_rounds: 2 });
    if (step === 2) return text('目标已保存。');
    if (step === 3) return tool('family_step', { cycles: 1 });
    if (step === 4) {
      const goal = ctx.goals.get(ctx.agents.roots()[0]);
      return tool('update_goal', { goal_id: goal.id, revision: goal.revision, action: 'complete' });
    }
    return text('实际一轮已经提交。');
  } });
  const { agent } = await f.attach();
  await f.turn(agent, 'Run exactly one cycle and verify it.');
  // Wait on the correlated goal outcome, not an unbounded global whenIdle assumption.
  if (f.ctx.goals.get(agent)?.phase !== 'complete') await new Promise(resolve => {
    const remove = f.ctx.on('goal/changed', () => {
      if (f.ctx.goals.get(agent)?.phase === 'complete') { remove(); resolve(); }
    });
    if (f.ctx.goals.get(agent)?.phase === 'complete') { remove(); resolve(); }
  });
  await agent.whenIdle();
  assert.equal(f.family.state.cycle, 1);
  assert.equal(f.ctx.goals.get(agent).phase, 'complete');
  const messages = agent.session.snapshotEvents().filter(event => event.type === 'user/message');
  assert(messages.some(event => event.data.source.kind === 'goal' && event.data.source.round === 1));
  assert.equal(messages.filter(event => event.data.source.kind === 'user').length, 1);
});

test('official automatic management compaction keeps raw history, plan/goal state and restored summary', async t => {
  let step = 0;
  const f = await fixture(t, { contextWindow: 8192, respond: body => {
    const isSummary = JSON.stringify(body.messages.at(-1)).includes('Condense this parent');
    if (isSummary) return text('Parent authorized records review. Care plan v1; body checkpoint unchanged; goal remains active. Child observations are not authorization.');
    if (++step === 1) return tool('family_plan', { text: 'Keep the signed care agreement.' });
    if (step === 2) return tool('create_goal', { objective: 'Review recorded exchanges.', max_goal_rounds: 2 });
    return text('已保存约定，继续讨论。');
  } });
  const handle = await f.attach();
  const { agent } = handle;
  await f.turn(agent, 'Parent agreement. ' + 'Actual recorded exchange with uncertain interpretation. '.repeat(100));
  const before = agent.session.snapshotEvents();
  for (let i = 0; i < 8; i++) await f.turn(agent, `Parent records ${i}. ` + 'Actual recorded exchange with uncertain interpretation. '.repeat(80));
  await f.turn(agent, 'Continue reviewing, not child speech.');
  const events = agent.session.snapshotEvents();
  assert(events.some(event => event.type === 'compaction/summary'), JSON.stringify({ config: f.ctx.compaction.config, model: await f.ctx.llm.resolveModelInfo('deepseek-official','deepseek-flash'), meter: f.ctx.tokenMeter.measure(agent.session), wires: f.wires.map(body => ({ max: body.max_tokens, chars: JSON.stringify(body).length })), compaction: events.filter(event => event.type.startsWith('compaction')) }));
  assert.deepEqual(events.slice(0, before.length), before);
  assert.equal(f.family.state.plan.version, 1);
  const goal = f.ctx.goals.get(agent);
  assert.equal(goal.objective, 'Review recorded exchanges.');
  assert(f.wires.some(body => body.max_tokens === 256));
  const sessionId = agent.session.id;
  await handle.dispose(); f.handles.splice(f.handles.indexOf(handle), 1);
  const restored = await f.attach(sessionId, true);
  assert.equal(f.ctx.goals.get(restored.agent).id, goal.id);
  assert.equal(f.ctx.goals.get(restored.agent).activation, 'disarmed');
  assert(restored.agent.session.snapshotEvents().some(event => event.type === 'compaction/summary'));
  await f.turn(restored.agent, 'What is our current agreement?');
  assert(JSON.stringify(f.wires.at(-1).messages).includes('Parent authorized records review'));
  assert.equal(f.family.state.plan.version, 1, 'resume must not rerun the plan tool');
});

test('real official standalone Web profile serves authenticated frontend/controller and restricted family routes offline', { timeout: 60000 }, async t => {
  const family = await familyServer(t);
  const dataRoot = path.join(ROOT, `workbench-web-test-${randomUUID()}`);
  const logs = [];
  const runtime = createWorkbenchRuntime({ familyOrigin: family.origin, dataRoot, port: 0,
    env: { DEEPSEEK_API_KEY: '', DEEPSEEK_BASE_URL: '', MENGLI_INITIAL_LOGIN_KEY: 'offline-login-fixture' }, onLog: value => logs.push(value) });
  t.after(async () => { await runtime.close(); fs.rmSync(dataRoot, { recursive: true, force: true }); });
  const info = await runtime.start();
  assert.notEqual(new URL(info.origin).port, '8900');
  assert.equal(info.profile, 'mengli-parent');
  assert.match(info.session_id, /^mengli-parent-/);
  assert.equal(JSON.parse(fs.readFileSync(path.join(dataRoot, 'management-session.json'), 'utf8')).session_id, info.session_id);
  assert.equal(path.relative(dataRoot, info.home).startsWith('..'), false);
  assert.equal(new URL(info.url).search, '');
  const entry = await fetch(info.url, { redirect: 'manual' });
  assert.equal(entry.status, 401);
  const loginHtml = await entry.text();
  assert(loginHtml.includes('本地密钥') && !loginHtml.includes('__DSH_BOOT__') && !loginHtml.includes(info.session_id));
  const login = key => fetch(`${info.origin}/auth/login`, { method: 'POST',
    headers: { Origin: info.origin, 'Content-Type': 'application/json' }, body: JSON.stringify({ key }) });
  const wrong = await login('incorrect-fixture-key');
  assert.equal(wrong.status, 401);
  assert.equal(wrong.headers.getSetCookie().length, 0);
  const accepted = await login('offline-login-fixture');
  assert.equal(accepted.status, 200);
  assert.deepEqual(await accepted.json(), { ok: true });
  assert(accepted.headers.get('set-cookie').includes('HttpOnly') && accepted.headers.get('set-cookie').includes('SameSite=Strict'));
  const cookie = accepted.headers.getSetCookie().map(value => value.split(';')[0]).join('; ');
  assert(cookie, 'password login reuses the official browser auth cookie');
  const page = await fetch(info.origin, { headers: { Cookie: cookie } });
  const html = await page.text();
  assert.equal(page.status, 200);
  assert.match(html, /__DSH_BOOT__/);
  assert(!html.includes('mengli-vitals-panel'));
  assert(!html.includes('pollVitals'));
  assert(!html.includes('mother-copilot-container'));
  assert(!html.includes('MengLi HARNESS'));
  assert(!html.includes('#C3635'));
  assert.match(html, /<title>DeepSeek Harness<\/title>/);
  assert.match(html, /dsh\.sessions\.current/);
  assert(html.includes(info.session_id));
  assert.match(html, /color-scheme:dark/);
  const themeResponse = await fetch(`${info.origin}/api/settings/describe`, { method: 'POST',
    headers: { Cookie: cookie, 'Content-Type': 'application/json' },
    body: JSON.stringify({ type: 'client-request', rpcId: 'offline-theme', method: 'settings/describe', payload: { args: {} } }) });
  const themeText = await themeResponse.text();
  assert(themeText.length, `Theme RPC returned an empty HTTP ${themeResponse.status}`);
  const themeSettings = JSON.parse(themeText);
  assert.equal(themeSettings.result.ok, true, JSON.stringify(themeSettings.result.error));
  assert.equal(themeSettings.result.value.namespaces.find(value => value.ns === 'ui-theme').value.preference, 'dark');
  assert(info.workspace_id);
  assert.equal(info.cwd, info.home);
  assert.match(html, /ui-chat/);
  assert.match(html, /ui-tool/);
  assert.match(html, /ui-goal/);
  assert.match(html, /ui-plan/);
  assert(html.includes('@mengli/native-family-client'), 'Native family client must be in the actual official boot roster');
  assert.equal((await fetch(`${info.origin}/api/family`)).status, 401);
  assert.equal((await fetch(`${info.origin}/api/family/events`)).status, 401);
  assert.equal((await fetch(`${info.origin}/api/family/life`)).status, 401);
  assert.equal((await fetch(`${info.origin}/api/family/life`, { headers: { Cookie: cookie } })).status, 200);
  assert.equal(family.calls.at(-1).url, '/api/life');
  const events = await fetch(`${info.origin}/api/family/events`, { headers: { Cookie: cookie } });
  assert.equal(events.status, 200);
  assert.match(events.headers.get('content-type'), /text\/event-stream/);
  assert.match(await events.text(), /event: family/);
  const status = await fetch(`${info.origin}/api/family`, { headers: { Cookie: cookie } });
  assert.equal(status.status, 200);
  const browserState = await status.json();
  assert.equal(browserState.cycle, 0);
  assert(browserState.workbench.url.includes('offline-launch-secret'), 'Authenticated browser retains its launch URL');
  const plan = await fetch(`${info.origin}/api/family/plan`, { method: 'POST', headers: { Cookie: cookie,
    Origin: info.origin, 'Content-Type': 'application/json' }, body: JSON.stringify({ text: 'Browser-approved care plan.' }) });
  assert.equal(plan.status, 200);
  assert.equal((await plan.json()).plan.version, 1);
  assert.deepEqual(await runtime.pauseCare(), { paused: [] });
  assert.deepEqual(await runtime.resumeCare(), { resumed: [] });
  assert.equal(family.state.cycle, 0);
  assert(!logs.join('').includes('startup failed'));
  const accountResponse = await fetch(`${info.origin}/api/local-account`, { headers: { Cookie: cookie } });
  const accountText = await accountResponse.text();
  assert(accountText.length, `Account view returned an empty HTTP ${accountResponse.status}`);
  const accountView = JSON.parse(accountText);
  assert.deepEqual(accountView, { displayName: '无月酱酱酱酱' });
  const accountPost = (part, body) => fetch(`${info.origin}/api/local-account/${part}`, { method: 'POST',
    headers: { Cookie: cookie, Origin: info.origin, 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  assert.equal((await accountPost('profile', { displayName: 'offline-parent' })).status, 200);
  assert.equal((await accountPost('key', { currentKey: 'offline-login-fixture', newKey: 'offline-changed-fixture' })).status, 200);
  assert.equal((await login('offline-login-fixture')).status, 401);
  assert.equal((await login('offline-changed-fixture')).status, 200);
  const logout = await accountPost('logout', {});
  assert.equal(logout.status, 200);
  assert(logout.headers.get('set-cookie').includes('Max-Age=0'));
  assert(!logs.join('').includes('offline-login-fixture'));
  await runtime.close();
  const patchFile = path.join(dataRoot, 'profile/cordis.patch.yml');
  fs.writeFileSync(patchFile, '- id: ui-theme\n  config:\n    preference: light\n- id: tool-bash\n  disabled: false\n');
  const restarted = createWorkbenchRuntime({ familyOrigin: family.origin, dataRoot, port: 0,
    env: { DEEPSEEK_API_KEY: '' }, onLog: value => logs.push(value) });
  try {
    const restored = await restarted.start();
    assert.equal(restored.session_id, info.session_id);
    const restoredPatch = JSON.parse(fs.readFileSync(patchFile, 'utf8'));
    assert.equal(restoredPatch.find(row => row.id === 'ui-theme').config.preference, 'light');
    assert(restoredPatch.some(row => row.id === 'tool-bash' && row.disabled));
    assert.deepEqual(await restarted.pauseCare(), { paused: [] });
    assert.equal(family.state.cycle, 0);
  } finally { await restarted.close(); }
});
