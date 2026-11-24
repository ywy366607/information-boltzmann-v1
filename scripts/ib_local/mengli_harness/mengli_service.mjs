import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { careHistory, familyLife, familyTelemetry } from './family_readmodel.mjs';
const packageArgument = process.argv.indexOf('--package-root');
if (packageArgument >= 0) process.env.MENGLI_DSH_PACKAGE_ROOT = process.argv[packageArgument + 1];
const { BridgeError, createDshRuntime, HARNESS_ROOT } = await import('./dsh_runtime.mjs');

const json = (res, status, value) => {
  res.writeHead(status, { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' });
  res.end(JSON.stringify(value));
};

export function createMengliServer(runtime, { lifeCardPath = null, family = null } = {}) {
  const lifeFile = lifeCardPath ? path.resolve(lifeCardPath) : null;
  const insideHarness = file => {
    const relative = path.relative(HARNESS_ROOT, file);
    return relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
  };
  if (lifeFile && !insideHarness(lifeFile)) throw new BridgeError('LIFE_PATH', 'Life cards belong under data/harness', 400);
  function readLife() {
    const observed = family?.observer?.snapshot();
    if (observed) return observed.card ? familyLife({ card: observed.card, state: 'saved' }, observed.checkpoint)
      : { card: null, state: 'unavailable' };
    if (!lifeFile || !fs.existsSync(lifeFile)) return { card: null, state: 'unbound' };
    try {
      if (!insideHarness(fs.realpathSync(lifeFile)) || fs.statSync(lifeFile).size > 65536) {
        return { card: null, state: 'invalid' };
      }
      const card = JSON.parse(fs.readFileSync(lifeFile, 'utf8'));
      if (card.schema !== 1 || typeof card.identity?.name !== 'string' ||
          !Number.isSafeInteger(card.source?.cycle) || card.source.cycle < 0 ||
          card.age?.cycles !== card.source.cycle || typeof card.source?.sha256 !== 'string') {
        return { card: null, state: 'invalid' };
      }
      return familyLife({ card, state: 'saved' }, card.source.checkpoint ?? family?.state.checkpoint);
    } catch { return { card: null, state: 'invalid' }; }
  }
  let cachedHistory;
  const sourceHistory = async () => {
    // Unified Mother memory: when observing a live Umwelt run, read the EXACT live
    // DSH caregiver session for the bound branch (falling back to the card's session_id
    // or live observer events) so the Mother in "和妈妈商量" and the Mother accompanying
    // MengLi are always ONE single Mother with full shared conversation history.
    if (family?.observer?.bound) {
      const snap = family.inspect();
      const bId = snap.branch_id;
      const bodyRef = readLife().card?.mother?.checkpoint_ref ?? null;
      try {
        if (bId && fs.existsSync(runtime.branchFile(bId))) {
          const liveState = await runtime.inspect(bId);
          if (liveState?.turns?.length) return liveState;
        }
        if (activeBranch && fs.existsSync(runtime.branchFile(activeBranch))) {
          const activeState = await runtime.inspect(activeBranch);
          if (activeState?.turns?.length) return activeState;
        }
        if (bodyRef?.session_id) {
          const bySession = await runtime.inspectSession(bodyRef, bId);
          if (bySession?.turns?.length) return bySession;
        }
      } catch { /* Fall back cleanly to observer events below if session JSONL is mid-write. */ }
      const obsTurns = [];
      for (const ev of snap.events ?? []) {
        if (ev.actor !== 'mother' && ev.actor !== 'parent') continue;
        const child = ev.utterance?.text ? ev.utterance.text : (ev.utterance?.spoke ? '[咿呀声]' : '[安静未开口]');
        const reply = ev.observation?.text ?? '';
        if (reply) {
          obsTurns.push({ role: 'user', actor: 'mengli', content: `[cycle ${ev.cycle}] ${child}` });
          obsTurns.push({ role: 'assistant', actor: ev.actor, content: reply });
        }
      }
      return { turns: obsTurns, branch_id: bId, checkpoint_ref: bodyRef };
    }
    const ref = family?.state.source_mother_checkpoint_ref;
    const key = JSON.stringify(ref ?? null);
    if (cachedHistory?.key !== key) {
      const snapshot = ref ? await runtime.inspect(family.state.source_branch_id, ref) : { turns: [] };
      cachedHistory = { key, snapshot };
    }
    return cachedHistory.snapshot;
  };
  const subscribers = new Set();
  let activeBranch = null;
  const broadcast = (event, value) => {
    for (const subscriber of subscribers) subscriber.res.write(`event: ${event}\ndata: ${JSON.stringify(value)}\n\n`);
  };
  if (family) family.onEvent = value => broadcast('family', value);
  const server = http.createServer(async (req, res) => {
    try {
      const host = req.headers.host;
      if (!host || !/^(127\.0\.0\.1|localhost)(:\d+)?$/.test(host)) {
        throw new BridgeError('INVALID_HOST', 'Only loopback hosts are accepted', 403);
      }
      const url = new URL(req.url, `http://${host}`);
      const boundBranch = family?.inspect().branch_id;
      const branch = url.searchParams.get('branch_id') ??
        (boundBranch && fs.existsSync(runtime.branchFile(boundBranch)) ? boundBranch : activeBranch);
      if (req.method === 'GET') {
        if (url.pathname === '/health') return json(res, 200, { status: 'ready', service: 'mengli-dsh', tools_enabled: false });
        if (url.pathname === '/' || url.pathname === '/index.html') {
          return json(res, 200, { service: 'mengli-membrane-bridge', ui: 'Use the MengLi Umwelt interface (梦漓环世界)' });
        }
        if (url.pathname === '/events' || (url.pathname === '/api/family/events' && family)) {
          res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
          res.write('event: service\ndata: {"service_status":"ready"}\n\n');
          if (url.pathname === '/api/family/events') res.write(`event: family\ndata: ${JSON.stringify(family.inspect())}\n\n`);
          const entry = { res, branch, family: url.pathname === '/api/family/events' };
          subscribers.add(entry);
          req.on('close', () => subscribers.delete(entry));
          return;
        }
        if (url.pathname === '/api/session') {
          if (!branch) return json(res, 200, { summary: null, summary_version: 0, turns: [], total_turns: 0, session_ref: null, checkpoint_ref: null });
          const bodyRef = !url.searchParams.has('branch_id')
            ? (family?.observer?.bound ? readLife().card?.mother?.checkpoint_ref : family?.state.mother_checkpoint_ref) : undefined;
          const state = await runtime.inspect(branch, bodyRef ?? undefined);
          return json(res, 200, { ...state, summary: state.summary ?? null, session_ref: state.checkpoint_ref,
            total_turns: state.turns.filter(turn => turn.role === 'assistant').length });
        }
        if (url.pathname === '/api/family/history' && family) {
          const before = url.searchParams.has('before') ? Number(url.searchParams.get('before')) : undefined;
          const limit = url.searchParams.has('limit') ? Number(url.searchParams.get('limit')) : 32;
          if ((before !== undefined && (!Number.isSafeInteger(before) || before < 0)) ||
              !Number.isSafeInteger(limit) || limit < 1 || limit > 64) throw new BridgeError('HISTORY_RANGE', 'History accepts before>=0 and limit 1..64', 400);
          return json(res, 200, careHistory(await sourceHistory(), { before, limit }));
        }
        if (url.pathname === '/api/family/telemetry' && family) return json(res, 200, familyTelemetry(family, readLife()));
        if (url.pathname === '/api/family' && family) {
          const snapshot = family.inspect(url.searchParams.has('after') ? Number(url.searchParams.get('after')) : null);
          return json(res, 200, { ...snapshot, recent_care_history: careHistory(await sourceHistory(), { limit: 8 }) });
        }
        if (url.pathname === '/api/life' || url.pathname === '/api/family/life') return json(res, 200, readLife());
        if (url.pathname === '/api/vitals') {
          if (family?.observer?.bound) {
            const telemetry = familyTelemetry(family, null), row = telemetry.history.rows?.at(-1) ?? {}, event = telemetry.latest_event;
            return json(res, 200, { cycle: telemetry.source.cycle, choice: row.choice ?? null,
              child_text: event?.utterance?.text ?? null, mother_reply: event?.observation?.text ?? null,
              metal_g: row.metal_g ?? null, life: telemetry.life.card, life_state: telemetry.life.state,
              metadata: { ...telemetry.source, training_status: telemetry.source.observation?.status,
                reply_meaning: 'Recorded full caregiver reply; actual prefix byte count is in event.observation.mother_bytes.' } });
          }
          const life = readLife();
          const state = branch ? await runtime.inspect(branch) : null;
          const usage = state?.usage_totals?.reply_usage;
          const knownCache = typeof usage?.cache_hit_tokens === 'number' && typeof usage?.cache_miss_tokens === 'number';
          const cacheTotal = knownCache ? usage.cache_hit_tokens + usage.cache_miss_tokens : 0;
          return json(res, 200, {
            cycle: null, choice: null, child_text: null, mother_reply: null,
            metal_g: null, s_state_norm: null, p_momentum_norm: null, q_invariant_norm: null,
            layer_momentums: null, active_relations: null, unknown_identity: null,
            current_evidence_count: null, mother_prequential_mse: null, question_norm: null,
            provider_hit_rate: cacheTotal > 0 ? usage.cache_hit_tokens / cacheTotal : null,
            api_calls: state?.usage_totals?.api_calls ?? null, failures: state?.usage_totals?.failures ?? null,
            latency_seconds: state?.telemetry?.latency_seconds ?? null,
            prompt_tokens: state?.usage_totals?.reply_usage?.prompt_tokens ?? null,
            completion_tokens: state?.usage_totals?.reply_usage?.completion_tokens ?? null,
            cache_hit_tokens: state?.usage_totals?.reply_usage?.cache_hit_tokens ?? null,
            cache_miss_tokens: state?.usage_totals?.reply_usage?.cache_miss_tokens ?? null,
            reply_usage: state?.usage_totals?.reply_usage ?? null, summary_usage: state?.usage_totals?.summary_usage ?? null,
            source_counts: state?.usage_totals?.source_counts ?? { api: 0, local: 0, shared: 0 },
            life: life.card, life_state: life.state,
            metadata: { run_id: null, arm: branch, source_checkpoint: family?.state.checkpoint ?? state?.provenance?.source_checkpoint ?? null,
              metrics_source: null, service_status: 'ready', training_status: 'unknown' },
          });
        }
        throw new BridgeError('NOT_FOUND', 'Route not found', 404);
      }
      if (req.method !== 'POST') throw new BridgeError('METHOD_NOT_ALLOWED', 'Use GET or JSON POST', 405);
      if (req.headers.origin && req.headers.origin !== `http://${host}`) {
        throw new BridgeError('CROSS_ORIGIN', 'Cross-origin writes are not accepted', 403);
      }
      if (req.headers['sec-fetch-site'] && !['same-origin', 'none'].includes(req.headers['sec-fetch-site'])) {
        throw new BridgeError('CROSS_ORIGIN', 'Cross-origin writes are not accepted', 403);
      }
      if (!/^application\/json(?:\s*;|$)/i.test(req.headers['content-type'] ?? '')) {
        throw new BridgeError('CONTENT_TYPE', 'Write requests require application/json', 415);
      }
      let body = '';
      req.setEncoding('utf8');
      for await (const chunk of req) {
        body += chunk;
        if (Buffer.byteLength(body) > 4 * 1024 * 1024) throw new BridgeError('BODY_TOO_LARGE', 'JSON body exceeds 4 MiB', 413);
      }
      let payload;
      try { payload = JSON.parse(body); } catch { throw new BridgeError('INVALID_JSON', 'Invalid JSON request', 400); }
      if (!payload || Array.isArray(payload) || typeof payload !== 'object') throw new BridgeError('INVALID_JSON', 'Expected a JSON object', 400);
      const operations = {
        '/api/mengli/restore': request => runtime.restore(request),
        '/api/mengli/prepare': request => runtime.prepare(request),
        '/api/mengli/commit': request => runtime.commit(request),
        '/api/mengli/compact': request => runtime.compactNow(request.branch_id, request.checkpoint_ref),
        '/api/mengli/context': request => runtime.context(request),
        '/api/mengli/reflect': request => runtime.reflect(request),
        ...(family ? {
          '/api/family/load': () => family.load(`http://${host}`),
          '/api/family/control': request => family.control(request.mode),
          '/api/family/plan': request => family.updatePlan(request),
          '/api/family/step': request => family.step(request),
          '/api/family/say': request => family.say(request),
        } : {}),
      };
      if (!operations[url.pathname]) throw new BridgeError('NOT_FOUND', 'Route not found', 404);
      const response = await operations[url.pathname](payload);
      if (payload.branch_id) activeBranch = payload.branch_id;
      if (url.pathname.endsWith('/commit')) {
        for (const subscriber of subscribers) {
          if (subscriber.branch && subscriber.branch !== payload.branch_id) continue;
          subscriber.res.write(`event: commit\ndata: ${JSON.stringify({ branch_id: payload.branch_id, ...response })}\n\n`);
        }
      }
      json(res, 200, response);
    } catch (error) {
      // Never forward provider response bodies, request headers or arbitrary SDK errors.
      json(res, error instanceof BridgeError ? error.status : 502, {
        error: { code: error instanceof BridgeError ? error.code : 'BACKEND_FAILED',
          message: error instanceof BridgeError ? error.message : 'Caregiver backend request failed' },
      });
    }
  });
  // No checkpoint polling or large payload broadcast when no browser is watching.
  let lastObserved;
  const observationTimer = setInterval(() => {
    if (!family?.observer?.bound || ![...subscribers].some(s => s.family)) return;
    const snapshot = family.inspect();
    // observed_at is a read timestamp, not a state change.
    const stable = JSON.stringify({ ...snapshot.observation, observed_at: undefined });
    if (stable !== lastObserved) { lastObserved = stable; broadcast('family', snapshot); }
  }, 2000);
  observationTimer.unref();
  server.on('close', () => { clearInterval(observationTimer); for (const subscriber of subscribers) subscriber.res.end(); subscribers.clear(); });
  return server;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const args = process.argv.slice(2);
  const option = name => { const i = args.indexOf(name); return i < 0 ? undefined : args[i + 1]; };
  if (option('--package-root')) process.env.MENGLI_DSH_PACKAGE_ROOT = option('--package-root');
  const initialKey = process.env.MENGLI_INITIAL_LOGIN_KEY;
  delete process.env.MENGLI_INITIAL_LOGIN_KEY;
  const { openLocalAccount } = await import('./local-account.mjs');
  await openLocalAccount(path.join(HARNESS_ROOT, 'workbench/local-account.json'), { initialKey, displayName: '无月酱酱酱酱' });
  const runtime = await createDshRuntime({ dataRoot: option('--data-root') });
  const lifeCardPath = option('--life-card') ?? path.join(HARNESS_ROOT, 'life/current.json');
  const { FamilyRuntime } = await import('./family_runtime.mjs');
  const { TrainingObserver } = await import('./training_observer.mjs');
  const family = new FamilyRuntime({ lifeCardPath, observer: new TrainingObserver() });
  const server = createMengliServer(runtime, { lifeCardPath, family });
  const port = Number(option('--port') ?? process.env.MENGLI_DSH_PORT ?? 8900);
  const bridgePort = Number(option('--bridge-port') ?? 8902);
  if (bridgePort === port) throw new Error('The native workbench and internal body bridge need separate ports');
  server.listen(bridgePort, '127.0.0.1', async () => {
    console.log(`MengLi Membrane Bridge (梦漓界膜桥) http://127.0.0.1:${bridgePort}`);
    try {
      const { createWorkbenchRuntime } = await import('./workbench_runtime.mjs');
      const workbench = await createWorkbenchRuntime({ familyOrigin: `http://127.0.0.1:${bridgePort}`,
        port, dataRoot: path.join(HARNESS_ROOT, 'workbench') });
      const info = await workbench.start();
      family.setWorkbench(workbench, info);
      console.log(`MengLi Umwelt (梦漓环世界) ${info.url}`);
    } catch (error) {
      family.workbenchInfo = { status: 'failed', error: error.message };
      family.publish();
      console.error('Mother workbench startup failed:', error.message);
    }
  });
  let closing = false;
  const close = async () => {
    if (closing) return;
    closing = true;
    if (!family.observer.bound) await family.control('paused');
    await family.close();
    server.closeAllConnections();
    server.close(async () => { await runtime.close(); });
  };
  process.once('SIGINT', close);
  process.once('SIGTERM', close);
}
