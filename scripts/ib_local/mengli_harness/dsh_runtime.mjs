import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createHash, randomUUID } from 'node:crypto';
import { isDeepStrictEqual } from 'node:util';
import { CaregiverMemoryCompactionEngine } from './caregiver_compaction.mjs';
import { DeepSeekChatAdapter } from './deepseek_chat_adapter.mjs';

export const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
export const HARNESS_ROOT = path.join(REPO_ROOT, 'data/harness');
export const DEFAULT_SETTINGS = Object.freeze({
  model: 'deepseek-flash', temperature: 0.7, max_tokens: 120,
  context_tokens: 8192, headroom_tokens: 1024, retain_tokens: 2048, summary_tokens: 512,
});
const ALLOWED_PROFILES = Object.freeze({
  8192: { headroom_tokens: 1024, retain_tokens: [64, 128, 2048], summary_tokens: [512] },
  32768: { headroom_tokens: 2048, retain_tokens: [null, 4096, 5242, 6144, 8192], summary_tokens: [512, 1024] },
  65536: { headroom_tokens: 2048, retain_tokens: [null, 6144, 8192, 10485, 12288], summary_tokens: [512, 1024, 1536] },
  131072: { headroom_tokens: 4096, retain_tokens: [null, 16384, 20971, 24576], summary_tokens: [1024, 1536, 2048] },
  262144: { headroom_tokens: 4096, retain_tokens: [null, 24576, 32768, 41943], summary_tokens: [1024, 1536, 2048] },
  524288: { headroom_tokens: 4096, retain_tokens: [null, 49152, 65536, 83886], summary_tokens: [1024, 1536, 2048] },
});
const packageRoot = process.env.MENGLI_DSH_PACKAGE_ROOT || 'F:/16、杂货行账册/ledger-dsh/node_modules/@deepseek-ai';
const load = name => import(pathToFileURL(path.join(packageRoot, name, 'lib/index.js')).href);
const hash = value => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const text = value => [{ type: 'text', text: value }];
const same = isDeepStrictEqual;
export class BridgeError extends Error {
  constructor(code, message, status = 409) { super(message); this.code = code; this.status = status; }
}
const requireValue = (condition, code, message, status = 400) => {
  if (!condition) throw new BridgeError(code, message, status);
};
function writeJson(file, value) {
  const temporary = `${file}.tmp`;
  const fd = fs.openSync(temporary, 'w');
  try { fs.writeFileSync(fd, JSON.stringify(value)); fs.fsyncSync(fd); } finally { fs.closeSync(fd); }
  fs.renameSync(temporary, file);
}
function readJson(file) { return fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, 'utf8')) : null; }
function settingsOf(settings = {}) {
  const resolved = { ...DEFAULT_SETTINGS, ...settings };
  const profile = ALLOWED_PROFILES[resolved.context_tokens];
  const valid = Boolean(
    profile &&
    resolved.model === DEFAULT_SETTINGS.model &&
    [0.7, 1.0, 1.3].includes(resolved.temperature) &&
    [120, 180, 240].includes(resolved.max_tokens) &&
    resolved.headroom_tokens === profile.headroom_tokens &&
    profile.retain_tokens.includes(resolved.retain_tokens) &&
    profile.summary_tokens.includes(resolved.summary_tokens)
  );
  requireValue(valid, 'INVALID_SETTINGS',
    'Supported caregiver profiles: deepseek-flash temp 0.7/1.0/1.3, context 8192..524288 with matching headroom/retain/summary budgets');
  return resolved;
}
function boundaryOf(ref) { return ref.result_seq ?? ref.turn_end_seq; }
function summaryVersion(events) { return events.filter(event => event.type === 'compaction/summary').length; }
function refFor(sessionId, events) {
  const end = events.findLast(event => event.type === 'turn/end');
  const assistant = events.findLast(event => event.type === 'assistant/message');
  return { session_id: sessionId, turn_end_seq: end?.seq ?? null, assistant_seq: assistant?.seq ?? null,
    result_seq: events.length ? events.at(-1).seq : null, summary_version: summaryVersion(events) };
}
function validateRef(ref, events) {
  const boundary = boundaryOf(ref);
  requireValue(boundary === null || (Number.isSafeInteger(boundary) && boundary >= 0 && boundary < events.length),
    'INVALID_CHECKPOINT', 'Checkpoint boundary does not exist');
  const prefix = boundary === null ? [] : events.slice(0, boundary + 1);
  const expected = refFor(ref.session_id, prefix);
  requireValue(expected.turn_end_seq === ref.turn_end_seq && expected.assistant_seq === ref.assistant_seq &&
    expected.summary_version === ref.summary_version, 'INVALID_CHECKPOINT', 'Checkpoint turn, assistant or summary version does not match its prefix');
  if (boundary !== null) requireValue(['turn/end', 'compaction/end', 'session/end-seed'].includes(events[boundary].type) ||
    (events[boundary].type === 'user/message' && events[boundary].data.source?.kind === 'caregiver-context'),
    'INVALID_CHECKPOINT', 'Checkpoint must end at a settled turn, context or compaction boundary');
  return prefix;
}

export async function createDshRuntime({ dataRoot = process.env.MENGLI_DSH_DATA_ROOT || path.join(HARNESS_ROOT, 'dsh'), fetchImpl, adapterFactory } = {}) {
  dataRoot = path.resolve(dataRoot);
  const relative = path.relative(HARNESS_ROOT, dataRoot);
  requireValue(relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative),
    'DATA_ROOT', 'Service data root must be inside repository data/harness');
  // Refuse links out of the allowed tree before creating any files.
  for (let parent = dataRoot; parent !== path.dirname(parent); parent = path.dirname(parent)) {
    if (fs.existsSync(parent)) {
      requireValue(!fs.lstatSync(parent).isSymbolicLink(), 'DATA_ROOT', 'Service data path must not traverse symbolic links');
    }
    if (parent === REPO_ROOT) break;
  }
  fs.mkdirSync(dataRoot, { recursive: true });
  for (const dir of ['branches', 'session-config', 'sessions']) fs.mkdirSync(path.join(dataRoot, dir), { recursive: true });
  const [{ Context }, sessions, projections, agents, llm, prompt, tools, persistence, meter, loop] = await Promise.all([
    load('cordis'), load('dsh-session'), load('dsh-session-projection'), load('dsh-agent'), load('dsh-llm'),
    load('dsh-system-prompt'), load('dsh-tools'), load('dsh-session-persistence-jsonl'), load('dsh-token-meter'), load('dsh-agent-loop'),
  ]);
  const ctx = new Context();
  const mounts = [];
  const mount = async (plugin, config) => { const fiber = ctx.plugin(plugin, config); mounts.push(fiber); await fiber.await(); };
  await mount(sessions.default);
  await mount(projections.default);
  await mount(agents.default);
  await mount(llm.default);
  await mount(prompt.default);
  await mount(tools.default);
  // Registry/driver dependency only: no callable tools or tool plugins are installed.
  ctx.tools.setExecutor({ describe: () => ({ schemas: [] }), execute: async () => {
    throw new BridgeError('TOOLS_DISABLED', 'Caregiver tools are disabled');
  } });
  await mount(persistence, { root: path.join(dataRoot, 'sessions'), compression: 'none' });
  await mount(meter.default);
  const calls = [];
  const onCall = call => { calls.push(call); };
  const adapter = adapterFactory ? adapterFactory({ tokenMeter: ctx.tokenMeter, onCall }) : new DeepSeekChatAdapter({
    cachePath: path.join(dataRoot, 'chat-wire-v1.sqlite'), tokenMeter: ctx.tokenMeter, fetchImpl, onCall,
  });
  ctx.llm.registerAdapter(['mengli-chat'], adapter);
  await mount(loop.default, { agents: [], maxParallelToolCalls: 1 });
  await mount(CaregiverMemoryCompactionEngine, {
    thresholdRatio: 0.8, headroomTokens: 1024, retainTokens: 2048,
    maxTokens: 512, compactionRetries: 0, maxOverflowRetries: 0, auto: false,
  });
  return new DshRuntime({ ctx, mounts, adapter, dataRoot, calls, buildForkSeed: sessions.buildForkSeed });
}

class DshRuntime {
  constructor(options) { Object.assign(this, options); this.handles = new Map(); this.engines = new Map(); this.serial = Promise.resolve(); }
  run(operation) {
    const pending = this.serial.then(operation);
    this.serial = pending.catch(() => {});
    return pending;
  }
  branchFile(branch) {
    requireValue(typeof branch === 'string' && branch.length > 0 && branch.length <= 256, 'INVALID_BRANCH', 'branch_id must be a nonempty string up to 256 characters');
    return path.join(this.dataRoot, 'branches', `${hash(branch)}.json`);
  }
  configFile(id) { return path.join(this.dataRoot, 'session-config', `${hash(id)}.json`); }
  state(branch) {
    const state = readJson(this.branchFile(branch));
    requireValue(state, 'NOT_RESTORED', 'Restore this branch before preparing a reply', 409);
    return state;
  }
  save(state) { writeJson(this.branchFile(state.branch_id), state); }
  async events(id) {
    const live = this.handles.get(id)?.agent;
    if (live) await this.ctx.sessions.flush(live.session);
    const handle = await this.ctx.sessionPersistence.open(id, 'read');
    try { return (await handle.read()).events; } finally { await handle.close(); }
  }
  async attach(id, config, creation) {
    if (this.handles.has(id)) return this.handles.get(id).agent;
    this.adapter.setContextWindow?.(id, config.settings.context_tokens);
    const common = {
      agentOptions: { provider: 'mengli-chat', model: config.settings.model, maxTokens: config.settings.max_tokens },
      setup: async agentCtx => {
        agentCtx.systemPrompt.section({ name: 'caregiver', order: 0,
          text: config.system_prompt, interpolate: false, complete: true });
        const scoped = agentCtx.isolate('compaction');
        // Reuse official dsh-compaction-basic defaults (thresholdRatio 0.8, retainRatio 0.16, retries 1)
        // whenever retain_tokens is null; legacy 8K checkpoints keep their exact retainTokens.
        const compactionCfg = {
          headroomTokens: config.settings.headroom_tokens,
          maxTokens: config.settings.summary_tokens,
          ...(config.settings.retain_tokens == null ? {} : { retainTokens: config.settings.retain_tokens }),
        };
        const engine = scoped.plugin(CaregiverMemoryCompactionEngine, compactionCfg);
        await engine.await();
        this.engines.set(id, scoped.get('compaction'));
      },
    };
    const handle = creation ? await this.ctx.agents.create({ ...common, sessionId: id, ...creation }) :
      await this.ctx.agents.resume({ ...common, resumeSessionId: id });
    this.handles.set(id, handle);
    return handle.agent;
  }
  meter(agent) {
    const value = this.ctx.tokenMeter.measure(agent.session);
    const cfg = readJson(this.configFile(agent.session.id)).settings;
    return { total_tokens: value.totalTokens, surface_tokens: value.surfaceTokens,
      baseline: value.baseline.kind, heuristic: true,
      context_tokens: cfg.context_tokens, headroom_tokens: cfg.headroom_tokens,
      retain_tokens: cfg.retain_tokens, summary_tokens: cfg.summary_tokens };
  }
  telemetry(agent, callStart = this.calls.length) {
    const calls = this.calls.slice(callStart).filter(call => call.session_id === agent.session.id);
    const normal = calls.filter(call => call.purpose !== 'compaction');
    const summaries = calls.filter(call => call.purpose === 'compaction');
    return { source: normal.at(-1)?.source ?? null, api_usage: normal.at(-1)?.usage ?? null,
      cached_usage: normal.at(-1)?.cached_usage ?? null, summary_usage: summaries.map(call => ({ source: call.source, usage: call.usage, cached_usage: call.cached_usage })),
      request_key: normal.at(-1)?.cache_key ?? null,
      summary_request_keys: summaries.map(call => call.cache_key),
      latency_seconds: normal.at(-1)?.latency_seconds ?? null, meter: this.meter(agent) };
  }
  account(state) {
    const totals = state.usage_totals ?? { reply_usage: null, summary_usage: null, source_counts: { api: 0, local: 0, shared: 0 } };
    totals.api_calls ??= 0;
    totals.failures ??= 0;
    for (const call of this.calls.filter(call => call.session_id === state.session_id)) {
      totals.source_counts[call.source] += 1;
      if (call.source === 'api') totals.api_calls += 1;
      if (call.failed) totals.failures += 1;
      if (!call.usage) continue;
      const name = call.purpose === 'compaction' ? 'summary_usage' : 'reply_usage';
      const sum = totals[name] ?? { prompt_tokens: 0, completion_tokens: 0, cache_hit_tokens: 0, cache_miss_tokens: 0, total_tokens: 0 };
      for (const key of Object.keys(sum)) {
        sum[key] = sum[key] === null || call.usage[key] === null ? null : sum[key] + call.usage[key];
      }
      totals[name] = sum;
    }
    state.usage_totals = totals;
  }
  async createSession(config, prefix, parent, migrationTurns = []) {
    const id = `mengli-${randomUUID()}`;
    writeJson(this.configFile(id), config);
    let creation = migrationTurns.length ? { seed: this.migrationSeed(config, migrationTurns) } : {};
    if (prefix?.length) {
      creation = { seed: this.buildForkSeed(prefix, prefix.at(-1).seq),
        meta: { parentSession: parent, isSeeded: true }, inheritedEventCount: prefix.length };
    }
    const agent = await this.attach(id, config, creation);
    await this.ctx.sessions.flush(agent.session);
    return agent;
  }
  restore(request) { return this.run(() => this._restore(request)); }
  async _restore({ branch_id, checkpoint_ref = null, settings, migration, usage_totals = null }) {
    const file = this.branchFile(branch_id);
    const existing = readJson(file);
    const resolved = settingsOf(settings);
    const bootstrapRestore = checkpoint_ref === null;
    if (existing && bootstrapRestore) {
      requireValue(existing.bootstrap_ref && (!migration || hash(migration) === existing.bootstrap_hash),
        'MIGRATION_MISMATCH', 'Null checkpoint restore must use the original bootstrap migration');
      checkpoint_ref = existing.bootstrap_ref;
    }
    // A repeated restore of the same base must preserve a prepared receipt.
    if (existing && same(existing.checkpoint_ref, checkpoint_ref) && same(existing.settings, resolved)) {
      const events = await this.events(existing.session_id);
      const tail = events.slice((boundaryOf(existing.checkpoint_ref) ?? -1) + 1);
      if (existing.pending?.status === 'prepared' || (!existing.pending && tail.every(event => event.type === 'session/end-seed'))) {
        const config = readJson(this.configFile(existing.session_id));
        const agent = await this.attach(existing.session_id, config);
        return { checkpoint_ref: existing.checkpoint_ref, telemetry: existing.last_telemetry ?? this.telemetry(agent), usage_totals: existing.usage_totals ?? usage_totals };
      }
    }
    let agent, config;
    if (checkpoint_ref) {
      requireValue(typeof checkpoint_ref.session_id === 'string', 'INVALID_CHECKPOINT', 'Missing session identity');
      config = readJson(this.configFile(checkpoint_ref.session_id));
      requireValue(config, 'UNKNOWN_SESSION', 'Session does not belong to this harness data root', 404);
      const windowUpgradeOnly = (
        config.owner_branch !== branch_id &&
        resolved.context_tokens > config.settings.context_tokens &&
        same({ ...config.settings, context_tokens: resolved.context_tokens }, resolved)
      );
      requireValue(same(config.settings, resolved) || windowUpgradeOnly, 'SETTINGS_MISMATCH', 'Restore settings differ from the stored session');
      const events = await this.events(checkpoint_ref.session_id);
      const prefix = validateRef(checkpoint_ref, events);
      const anotherBranch = config.owner_branch !== branch_id;
      const hasTail = events.slice(prefix.length).some(event => event.type !== 'session/end-seed');
      if (hasTail || anotherBranch || windowUpgradeOnly) {
        config = { ...config, owner_branch: branch_id, settings: resolved };
        agent = await this.createSession(config, prefix, checkpoint_ref.session_id);
      } else agent = await this.attach(checkpoint_ref.session_id, config);
    } else {
      requireValue(typeof migration?.system_prompt === 'string' && migration.system_prompt.length > 0,
        'SYSTEM_PROMPT_REQUIRED', 'New sessions require migration.system_prompt from the Python provider');
      config = { owner_branch: branch_id, settings: resolved, system_prompt: migration.system_prompt,
        provenance: migration.provenance ?? null };
      requireValue(Array.isArray(migration.turns ?? []), 'INVALID_MIGRATION', 'migration.turns must be an array');
      agent = await this.createSession(config, undefined, undefined, migration.turns ?? []);
      await this.ctx.sessions.flush(agent.session);
    }
    const events = await this.events(agent.session.id);
    const state = { branch_id, session_id: agent.session.id, settings: resolved,
      checkpoint_ref: refFor(agent.session.id, events), pending: null, last_receipt: null,
      bootstrap_ref: bootstrapRestore ? refFor(agent.session.id, events) : existing?.bootstrap_ref ?? refFor(agent.session.id, events),
      bootstrap_hash: existing?.bootstrap_hash ?? hash(migration ?? null),
      usage_totals,
      last_telemetry: this.telemetry(agent) };
    this.save(state);
    return { checkpoint_ref: state.checkpoint_ref, telemetry: state.last_telemetry, usage_totals: state.usage_totals };
  }
  migrationSeed(config, turns) {
    const seed = [];
    const session = { append(type, data, options = {}) {
      seed.push({ seq: seed.length, time: Date.now(), type, data, ...options });
    } };
    for (const [index, value] of turns.entries()) {
      requireValue(typeof value.input === 'string' && typeof value.delivered === 'string' &&
        (typeof value.raw === 'string' || value.raw == null), 'INVALID_MIGRATION', 'Each migrated turn needs input, delivered and optional real raw text');
      const raw = value.raw ?? value.delivered;
      const turn = index + 1, step = 1;
      session.append('turn/start', { turn });
      session.append('step/start', { turn, step });
      if (index === 0) session.append('system/message', { turn, step,
        message: { id: randomUUID(), role: 'system', content: text(config.system_prompt), source: { kind: 'system-prompt' } } }, { surfaceOp: 'append' });
      session.append('user/message', { id: randomUUID(), role: 'user', content: text(value.input),
        source: { kind: 'checkpoint-migration', provenance: value.provenance ?? (value.raw == null ? 'delivered-only' : 'checkpoint-raw') } }, { surfaceOp: 'append' });
      session.append('assistant/message', { turn, step,
        message: { id: randomUUID(), role: 'assistant', content: text(raw), source: { kind: 'model', provider: 'checkpoint-migration', model: 'imported',
          replayState: { provenance: value.provenance ?? (value.raw == null ? 'delivered-only' : 'checkpoint-raw'),
            raw_source: value.raw == null ? 'delivered-only' : 'checkpoint-raw', delivered: value.delivered } } },
        stream: [] }, { surfaceOp: 'append' });
      session.append('step/end', { turn, step });
      session.append('turn/end', { turn, reason: { kind: 'completed' } });
    }
    return seed;
  }
  prepare(request) { return this.run(() => this._prepare(request)); }
  async _prepare({ branch_id, checkpoint_ref, step_id, transcript }) {
    requireValue(typeof step_id === 'string' && step_id.length > 0 && typeof transcript === 'string', 'INVALID_REQUEST', 'prepare requires step_id and transcript strings');
    const state = this.state(branch_id);
    requireValue(same(state.checkpoint_ref, checkpoint_ref), 'CHECKPOINT_MISMATCH', 'Prepare must use the current committed checkpoint');
    const inputHash = hash(transcript);
    if (state.pending) {
      requireValue(state.pending.step_id === step_id && state.pending.input_hash === inputHash, 'PENDING_CONFLICT', 'A different turn is already pending');
      if (state.pending.status === 'prepared') return state.pending.response;
      throw new BridgeError(state.pending.error_code ?? 'INCOMPLETE_PENDING', 'The pending generation did not complete; explicitly restore its checkpoint before retrying');
    }
    if (state.last_receipt?.step_id === step_id) {
      requireValue(state.last_receipt.input_hash === inputHash, 'STEP_CONFLICT', 'Step identity was reused with different input');
      return state.last_receipt.response;
    }
    const config = readJson(this.configFile(state.session_id));
    const agent = await this.attach(state.session_id, config);
    const baseEvents = await this.events(state.session_id);
    // No unseen durable tail may silently enter a new request.
    requireValue(baseEvents.slice((boundaryOf(state.checkpoint_ref) ?? -1) + 1).every(event => event.type === 'session/end-seed'),
      'UNCOMMITTED_TAIL', 'Restore the committed checkpoint before generating');
    const pending = { step_id, input_hash: inputHash, base_ref: state.checkpoint_ref, status: 'generating' };
    state.pending = pending;
    this.save(state);
    this.calls.length = 0;
    const callStart = 0;
    try {
      agent.followup({ id: randomUUID(), role: 'user', content: text(transcript), source: { kind: 'mengli-child' } });
      await agent.whenIdle();
      await this.ctx.sessions.flush(agent.session);
      const events = await this.events(state.session_id);
      const fresh = events.slice(baseEvents.length);
      const end = fresh.findLast(event => event.type === 'turn/end');
      const assistant = fresh.findLast(event => event.type === 'assistant/message');
      requireValue(end?.data.reason.kind === 'completed' && assistant && !assistant.data.interrupted,
        end?.data.reason.kind === 'max-tokens' ? 'MAX_TOKENS' : end?.data.reason.error?.code ?? 'GENERATION_FAILED',
        'Caregiver turn did not finish normally', 502);
      const raw = assistant.data.message.content.filter(block => block.type === 'text').map(block => block.text).join('');
      requireValue(raw.trim().length > 0, 'EMPTY_REPLY', 'Caregiver returned no usable text', 502);
      const telemetry = this.telemetry(agent, callStart);
      const response = { step_id, receipt_id: `${branch_id}:${step_id}`, raw, delivered: raw.trim(),
        source: telemetry.source, usage: telemetry.api_usage, result_ref: refFor(state.session_id, events), telemetry };
      Object.assign(pending, { status: 'prepared', raw, delivered: response.delivered, result_ref: response.result_ref, usage: response.usage, response });
      this.account(state);
      response.usage_totals = state.usage_totals;
      this.save(state);
      return response;
    } catch (error) {
      pending.status = 'failed';
      pending.error_code = error.code ?? 'GENERATION_FAILED';
      this.account(state);
      this.save(state);
      throw error;
    }
  }
  commit(request) { return this.run(() => this._commit(request)); }
  async _commit({ branch_id, step_id }) {
    const state = this.state(branch_id);
    if (state.last_receipt?.step_id === step_id && !state.pending) {
      return { checkpoint_ref: state.checkpoint_ref, telemetry: state.last_telemetry, usage_totals: state.usage_totals };
    }
    requireValue(state.pending?.step_id === step_id && state.pending.status === 'prepared', 'NO_PREPARED_RECEIPT', 'No successful prepared reply matches this step');
    const pending = state.pending;
    state.checkpoint_ref = pending.result_ref;
    state.last_telemetry = pending.response.telemetry;
    state.last_receipt = { step_id, input_hash: pending.input_hash, response: pending.response };
    state.pending = null;
    this.save(state);
    return { checkpoint_ref: state.checkpoint_ref, telemetry: state.last_telemetry, usage_totals: state.usage_totals };
  }
  context(request) { return this.run(async () => {
    const { branch_id, checkpoint_ref, plan, observations = [] } = request;
    const state = this.state(branch_id);
    requireValue(!state.pending, 'PENDING_CONFLICT', 'Context changes apply between delivered replies');
    requireValue(same(state.checkpoint_ref, checkpoint_ref), 'CHECKPOINT_MISMATCH', 'Context must use the committed checkpoint');
    requireValue(Array.isArray(observations) && observations.every(value => typeof value.event_id === 'string' && value.caregiver === 'parent'),
      'INVALID_CONTEXT', 'Observations must identify actual parent care events');
    requireValue(plan === undefined || (Number.isSafeInteger(plan.version) && typeof plan.text === 'string'),
      'INVALID_CONTEXT', 'A care plan requires a version and text');
    const fresh = observations.filter(value => !(state.observed_events ?? []).includes(value.event_id));
    const changedPlan = plan && plan.version !== (state.plan_version ?? 0);
    if (fresh.length || changedPlan) {
      const agent = await this.attach(state.session_id, readJson(this.configFile(state.session_id)));
      const base = await this.events(state.session_id);
      requireValue(base.slice((boundaryOf(state.checkpoint_ref) ?? -1) + 1).every(event => event.type === 'session/end-seed'),
        'UNCOMMITTED_TAIL', 'Restore the committed checkpoint before appending context');
      if (!base.some(event => event.type === 'system/message')) {
        // The official format protects the initial system surface. Seed it
        // before a context-only message, without inventing an assistant reply.
        agent.session.append('turn/start', { turn: 1 });
        agent.session.append('step/start', { turn: 1, step: 1 });
        agent.session.append('system/message', { turn: 1, step: 1, message: {
          id: randomUUID(), role: 'system', content: text(readJson(this.configFile(state.session_id)).system_prompt),
          source: { kind: 'system-prompt' },
        } }, { surfaceOp: 'append' });
        agent.session.append('step/end', { turn: 1, step: 1 });
        agent.session.append('turn/end', { turn: 1, reason: { kind: 'completed' } });
      }
      const context = { ...(changedPlan ? { care_plan: plan } : {}), ...(fresh.length ? { actual_parent_care: fresh } : {}) };
      agent.session.append('user/message', { id: randomUUID(), role: 'user',
        content: text('External caregiver context. These are records, not words spoken by you. Follow the care plan in future replies; parent speech is not your own speech.\n' + JSON.stringify(context)),
        source: { kind: 'caregiver-context' } }, { surfaceOp: 'append' });
      await this.ctx.sessions.flush(agent.session);
      state.checkpoint_ref = refFor(state.session_id, await this.events(state.session_id));
      if (changedPlan) state.plan_version = plan.version;
      state.observed_events = [...(state.observed_events ?? []), ...fresh.map(value => value.event_id)];
      this.save(state);
      // Resume from the newly committed surface so the official agent derives
      // its next turn counter from the context boundary, including an initial seed.
      await this.handles.get(state.session_id).dispose();
      this.handles.delete(state.session_id);
      this.engines.delete(state.session_id);
    }
    return { checkpoint_ref: state.checkpoint_ref, plan_version: state.plan_version ?? 0 };
  }); }
  compactNow(branch_id, checkpoint_ref) { return this.run(async () => {
    const state = this.state(branch_id);
    requireValue(checkpoint_ref === undefined || same(state.checkpoint_ref, checkpoint_ref), 'CHECKPOINT_MISMATCH', 'Compact must use the committed checkpoint');
    requireValue(!state.pending, 'PENDING_CONFLICT', 'Commit or restore the pending reply before compacting');
    const agent = await this.attach(state.session_id, readJson(this.configFile(state.session_id)));
    this.calls.length = 0;
    const callStart = 0;
    let result;
    try { result = await this.engines.get(agent.session.id).compactNow(agent, new AbortController().signal); }
    catch (error) { this.account(state); this.save(state); throw error; }
    await this.ctx.sessions.flush(agent.session);
    state.checkpoint_ref = refFor(state.session_id, await this.events(state.session_id));
    state.last_telemetry = this.telemetry(agent, callStart);
    this.account(state);
    this.save(state);
    return { compacted: result !== null, checkpoint_ref: state.checkpoint_ref,
      summary_version: state.checkpoint_ref.summary_version, telemetry: state.last_telemetry, usage_totals: state.usage_totals };
  }); }
  reflect(request) { return this.run(async () => {
    const { branch_id, checkpoint_ref, dossier } = request;
    requireValue(typeof dossier === 'string' && dossier.trim().length > 0 && dossier.length <= 24000,
      'INVALID_REQUEST', 'reflect requires a non-empty dossier string <= 24000 chars');
    const state = this.state(branch_id);
    requireValue(!state.pending, 'PENDING_CONFLICT', 'Commit or restore the pending reply before reflecting');
    requireValue(checkpoint_ref === undefined || same(state.checkpoint_ref, checkpoint_ref),
      'CHECKPOINT_MISMATCH', 'Reflect must use the committed checkpoint');
    const config = readJson(this.configFile(state.session_id));
    const agent = await this.attach(state.session_id, config);
    const events = validateRef(state.checkpoint_ref, await this.events(state.session_id));
    // Reconstruct the exact active surface messages (after latest compaction replace if any)
    // so this side-query hits the exact same DeepSeek KV cache prefix without polluting the dialogue!
    const lastReplaceIdx = events.findLastIndex(ev => ev.surfaceOp?.op === 'replace');
    const active = lastReplaceIdx >= 0 ? events.slice(lastReplaceIdx) : events;
    const messages = [];
    for (const ev of active) {
      if (ev.type === 'system/message') {
        const msg = ev.data.message ?? ev.data;
        messages.push({ role: 'system', content: msg.content });
      } else if (ev.type === 'user/message') {
        messages.push({ role: 'user', content: ev.data.content });
      } else if (ev.type === 'assistant/message') {
        const msg = ev.data.message ?? ev.data;
        messages.push({ role: 'assistant', content: msg.content });
      }
    }
    if (!messages.some(m => m.role === 'system')) {
      messages.unshift({ role: 'system', content: text(config.system_prompt) });
    }
    messages.push({ role: 'user', content: text(dossier) });
    this.calls.length = 0;
    const callStart = 0;
    let raw = '';
    try {
      for await (const chunk of this.adapter.stream({
        model: config.settings.model,
        messages,
        temperature: 0.4,
        maxTokens: 600,
        contextWindow: config.settings.context_tokens,
        sessionId: state.session_id,
        purpose: 'reflection',
      })) {
        if (chunk.type === 'text-delta') raw += chunk.text;
      }
    } finally {
      this.account(state);
      this.save(state);
    }
    const telemetry = this.telemetry(agent, callStart);
    return { raw: raw.trim(), telemetry, usage_totals: state.usage_totals };
  }); }
  inspect(branch_id, checkpoint_ref) { return this.run(async () => {
    const state = this.state(branch_id);
    const ref = checkpoint_ref ?? state.checkpoint_ref;
    // Read only the selected committed prefix, never the generated future.
    const events = validateRef(ref, await this.events(ref.session_id));
    const summary = events.findLast(event => event.type === 'user/message' && event.surfaceOp?.op === 'replace' && event.sourceEventSeqs?.some(seq => events[seq]?.type === 'compaction/summary'));
    return { branch_id, checkpoint_ref: ref, summary_version: ref.summary_version,
      summary: summary ? summary.data.content.filter(block => block.type === 'text').map(block => block.text).join('') : null,
      usage_totals: state.usage_totals ?? null, provenance: readJson(this.configFile(state.session_id)).provenance,
      pending: state.pending ? { step_id: state.pending.step_id, status: state.pending.status } : null,
      turns: events.filter(event => ['user/message', 'assistant/message'].includes(event.type) &&
        event.data.source?.kind !== 'caregiver-context').map(event => ({
        role: event.type === 'user/message' ? 'user' : 'assistant',
        content: (event.data.message ?? event.data).content.filter(block => block.type === 'text').map(block => block.text).join(''),
      })), telemetry: state.last_telemetry };
  }); }
  inspectSession(checkpoint_ref, branch_id = null) { return this.run(async () => {
    if (!checkpoint_ref?.session_id) return { branch_id, checkpoint_ref: null, summary_version: 0, summary: null, turns: [] };
    const allEvents = await this.events(checkpoint_ref.session_id);
    const events = checkpoint_ref.turn_end_seq != null ? validateRef(checkpoint_ref, allEvents) : allEvents;
    const summary = events.findLast(event => event.type === 'user/message' && event.surfaceOp?.op === 'replace' && event.sourceEventSeqs?.some(seq => events[seq]?.type === 'compaction/summary'));
    return { branch_id, checkpoint_ref, summary_version: checkpoint_ref.summary_version ?? 0,
      summary: summary ? summary.data.content.filter(block => block.type === 'text').map(block => block.text).join('') : null,
      turns: events.filter(event => ['user/message', 'assistant/message'].includes(event.type) &&
        event.data.source?.kind !== 'caregiver-context').map(event => ({
        role: event.type === 'user/message' ? 'user' : 'assistant',
        content: (event.data.message ?? event.data).content.filter(block => block.type === 'text').map(block => block.text).join(''),
      })) };
  }); }
  async close() {
    await this.serial;
    for (const handle of this.handles.values()) await handle.dispose();
    this.handles.clear();
    for (const mount of this.mounts.reverse()) await mount.dispose();
    this.adapter.close?.();
    await this.ctx.fiber.dispose();
  }
}
