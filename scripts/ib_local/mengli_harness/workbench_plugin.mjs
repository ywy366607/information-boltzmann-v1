import { randomUUID } from 'node:crypto';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const packageRoot = process.env.MENGLI_DSH_PACKAGE_ROOT || 'F:/16、杂货行账册/ledger-dsh/node_modules/@deepseek-ai';
const { defineTool } = await import(pathToFileURL(path.join(packageRoot, 'dsh-tools/lib/index.js')).href);
const { CommandDefinitionId } = await import(pathToFileURL(path.join(packageRoot, 'dsh-commands/lib/brand.js')).href);

export const name = 'mengli-family';
export const inject = ['tools', 'agents', 'goals', 'sessions', 'connection', 'webServer', 'commands'];

/** Remove only the retired inline MengLi dashboard previously injected into this installed index. */
export function cleanOfficialIndex(html) {
  const removeBlock = (startMarker, endMarker, requiredMarker) => {
    const start = html.indexOf(startMarker);
    if (start < 0) return;
    const end = html.indexOf(endMarker, start);
    if (end < 0 || !html.slice(start, end).includes(requiredMarker)) throw new Error('Unrecognized retired MengLi index injection');
    html = html.slice(0, start) + html.slice(end + endMarker.length);
  };
  html = html.replace('<title>MengLi HARNESS</title>', '<title>DeepSeek Harness</title>');
  removeBlock('    <script>\n      // Safe title branding without touching React DOM', '</script>', 'MengLi HARNESS');
  removeBlock('    <style>\n      /* Ensure DSH Header & Tab Bar ALWAYS stay on top of any custom views */', '</style>', '/* MengLi Training Dashboard styles */');
  removeBlock('    <!-- Injected MengLi Training Vital Signs Dashboard -->', '    </script>', 'async function clearMotherWhispers()');
  return html;
}
export const FAMILY_ROUTES = Object.freeze({
  family_state: { path: '/api/family', method: 'GET' },
  family_history: { path: '/api/family/history', method: 'GET' },
  family_plan: { path: '/api/family/plan', method: 'POST' },
  family_step: { path: '/api/family/step', method: 'POST' },
  family_control: { path: '/api/family/control', method: 'POST' },
});

// UI-only operations deliberately stay outside the model tool registry.
export const FAMILY_UI_ROUTES = Object.freeze({
  family_load: { path: '/api/family/load', method: 'POST' },
  family_say: { path: '/api/family/say', method: 'POST' },
  family_telemetry: { path: '/api/family/telemetry', method: 'GET' },
  family_life: { path: '/api/family/life', upstream: '/api/life', method: 'GET' },
});

export function validateFamilyArguments(tool, args, { ui = false } = {}) {
  const keys = { family_state: ['after'], family_history: ['before', 'limit'], family_plan: ['text', 'version'], family_step: ['cycles'], family_control: ['mode'],
    ...(ui ? { family_load: [], family_say: ['text'], family_life: [], family_telemetry: [] } : {}) }[tool];
  if (!keys || !args || typeof args !== 'object' || Array.isArray(args) || Object.keys(args).some(key => !keys.includes(key))) {
    throw new Error('Only fixed family API arguments are allowed');
  }
  if (tool === 'family_state' && args.after !== undefined && (!Number.isSafeInteger(args.after) || args.after < 0)) throw new Error('after must be a nonnegative event cursor');
  if (tool === 'family_history' && ((args.before !== undefined && (!Number.isSafeInteger(args.before) || args.before < 0)) ||
    (args.limit !== undefined && (!Number.isSafeInteger(args.limit) || args.limit < 1 || args.limit > 64)))) throw new Error('History accepts before>=0 and limit 1..64');
  if (tool === 'family_plan' && (typeof args.text !== 'string' || !args.text.trim() || args.text.length > 16000)) throw new Error('Plan text must contain 1..16000 characters');
  if (tool === 'family_plan' && args.version !== undefined && (!Number.isSafeInteger(args.version) || args.version < 0)) throw new Error('Plan version must be a nonnegative integer');
  if (tool === 'family_say' && (typeof args.text !== 'string' || !args.text.trim() || Buffer.byteLength(args.text) > 16384)) throw new Error('Parent text must contain 1..16384 UTF-8 bytes');
  if (tool === 'family_step' && (!Number.isInteger(args.cycles) || args.cycles < 1 || args.cycles > 8)) throw new Error('cycles must be an integer from 1 to 8');
  if (tool === 'family_control' && !['parent', 'mother', 'paused'].includes(args.mode)) throw new Error('Unknown family control mode');
  return args;
}

/** No retry, arbitrary URL, filesystem operation, or local result replay. */
export function createFamilyClient(familyOrigin, { fetchImpl = fetch, ui = false } = {}) {
  const origin = new URL(familyOrigin);
  if (origin.protocol !== 'http:' || !['127.0.0.1', '[::1]', 'localhost'].includes(origin.hostname) || origin.pathname !== '/') throw new Error('Family endpoint must be a loopback origin');
  return async (tool, args = {}, signal) => {
    validateFamilyArguments(tool, args, { ui });
    const route = FAMILY_ROUTES[tool] ?? (ui ? FAMILY_UI_ROUTES[tool] : undefined);
    const url = new URL(route.upstream ?? route.path, origin);
    if (tool === 'family_state' && args.after !== undefined) url.searchParams.set('after', args.after);
    if (tool === 'family_history') for (const key of ['before', 'limit']) {
      if (args[key] !== undefined) url.searchParams.set(key, args[key]);
    }
    const response = await fetchImpl(url, { method: route.method, signal, redirect: 'error',
      ...(route.method === 'POST' ? { headers: { 'Content-Type': 'application/json', Origin: origin.origin }, body: JSON.stringify(args) } : {}) });
    const value = await response.json();
    if (!response.ok) throw new Error(value.error?.message ?? value.error ?? `Family API returned ${response.status}`);
    return value;
  };
}

/** Browser launch URLs carry credentials; the provider-facing tool boundary must not receive them. */
export function modelFamilyResult(value) {
  if (!value || typeof value !== 'object') return value;
  const result = { ...value };
  if (result.workbench && typeof result.workbench === 'object') {
    const { url: _launchUrl, ...workbench } = result.workbench;
    result.workbench = workbench;
  }
  // Care dialogue is useful to the mother; full tensor/metrics diagnostics are
  // available to the separate dashboard, not repeated in every tool response.
  for (const name of ['events', 'committed']) if (Array.isArray(result[name])) result[name] = result[name].map(event => {
    const { row: _metrics, life_card_data: _card, ...care } = event;
    return care;
  });
  return result;
}

export function directHumanTurn(ctx, agent) {
  if (!agent || !ctx.agents.roots().includes(agent) || ctx.agents.currentInitiator() !== agent || agent.status !== 'running') return false;
  const start = ctx.agents.turnBoundaryOf(agent.session).openTurnStartSeq;
  return start !== null && agent.session.snapshotEvents().slice(start + 1).some(event =>
    event.type === 'user/message' && event.data.source.kind === 'user');
}

export async function apply(ctx, config) {
  const displayRead = createFamilyClient(config.familyOrigin, { ui: true });
  if (config.accountFile) {
    const { openLocalAccount } = await import('./local-account.mjs');
    const { installLocalLogin } = await import('./local-login.mjs');
    installLocalLogin(ctx, { account: await openLocalAccount(config.accountFile),
      getEntranceCard: async () => (await displayRead('family_life', {}, AbortSignal.timeout(2000))).card });
  }
  ctx.webServer.tapIndex(cleanOfficialIndex);
  const request = createFamilyClient(config.familyOrigin);
  const pauseCare = async () => {
    const agents = ctx.agents.roots();
    const paused = [];
    // Synchronous disarm/pause precedes every await, so another driver cannot reserve a round.
    for (const agent of agents) {
      let goal = ctx.goals.disarm(agent);
      if (goal?.phase === 'active') {
        goal = ctx.goals.pause(agent, { id: goal.id, revision: goal.revision });
        paused.push({ sessionId: agent.session.id, goal });
      }
    }
    await Promise.all(agents.map(agent => ctx.sessions.flush(agent.session)));
    return { paused };
  };
  const resumeCare = async () => {
    const resumed = [];
    for (const agent of ctx.agents.roots()) {
      const goal = ctx.goals.get(agent);
      if (goal && ['paused', 'active'].includes(goal.phase) && goal.activation === 'disarmed' && goal.roundsStarted < goal.maxGoalRounds) {
        resumed.push({ sessionId: agent.session.id, goal: ctx.goals.resume(agent, { id: goal.id, revision: goal.revision }) });
      }
    }
    await Promise.all(ctx.agents.roots().map(agent => ctx.sessions.flush(agent.session)));
    return { resumed };
  };
  ctx.provide('mengliFamily', { pauseCare, resumeCare });

  // Register `/clear` slash command so parents can cleanly reset the workbench chat context in-place.
  ctx.effect(function* () {
    yield ctx.commands.register({
      definitionId: CommandDefinitionId('mengli-family/command-clear'),
      name: 'clear',
      description: '清空当前工作台对话上下文（不影响梦漓本体与照护记忆）',
      handler: async invocation => {
        const session = invocation.agent.session;
        const events = session.snapshotEvents();
        const lastReplace = events.findLastIndex(ev => ev.surfaceOp?.op === 'replace');
        const active = lastReplace >= 0 ? events.slice(lastReplace) : events;
        const clearable = active.filter(ev =>
          ['user/message', 'assistant/message', 'tool/result', 'tool/call'].includes(ev.type),
        );
        if (!clearable.length) {
          return { kind: 'success', text: '当前对话上下文已是空的。' };
        }
        const startSeq = clearable[0].seq;
        const endSeq = clearable[clearable.length - 1].seq;
        session.append('user/message', {
          id: randomUUID(),
          role: 'user',
          content: [{ type: 'text', text: '[家长执行了 /clear：工作台会话上下文已清空，请从这里开始新的对话；如需了解梦漓最新状态与聊天记录，随时调用 family_state / family_history。]' }],
          source: { kind: 'system' },
        }, {
          surfaceOp: { op: 'replace', startSeq, endSeq },
        });
        await ctx.sessions.flush(session);
        return { kind: 'success', text: `已清空 ${clearable.length} 条工作台上下文记录（梦漓与妈妈的共享照护记忆保留完整）。` };
      },
    });
  }, 'mengli-command-clear');

  const parameters = {
    family_state: { after: { type: 'integer', description: 'Optional new workbench-event cursor. Inherited care dialogue is separate and remains available in recent_care_history/family_history even at cursor 0.' } },
    family_history: { before: { type: 'integer', description: 'Optional exclusive saved message index for reading earlier pages.' },
      limit: { type: 'integer', description: 'Saved caregiver/child messages per page, 1..64; default 32.' } },
    family_plan: { text: { type: 'string', required: true } },
    family_step: { cycles: { type: 'integer', required: true, description: 'One through eight complete mother-care cycles.' } },
    family_control: { mode: { type: 'string', required: true, enum: ['parent', 'mother', 'paused'] } },
  };
  const descriptions = {
    family_state: 'Read the shared mother-care state, plan and recent inherited dialogue as well as new workbench events. events=[] means no NEW workbench cycles, not no lifetime experience. *_seq values are event positions, not rounds. Read before discussing Mengli or planning care; observation is not authorization.',
    family_history: 'Read the same bound saved child–mother conversation shown in the Mengli view, with real speaker names, actual message counts and bounded pagination. You are the same mother across care and management workspaces. Read even when family_state.events is empty. This never starts care or training.',
    family_plan: 'Explicitly update the versioned external care plan, not the child’s intrinsic purpose or reward.',
    family_step: 'Execute 1..8 complete real care cycles through the separate caregiver session. Only mother mode permits this; report actual committed results.',
    family_control: 'Pause care, hand over to the parent, or explicitly resume mother care. Resuming requires a direct human turn. Pause also disarms management goals.',
  };
  for (const tool of Object.keys(FAMILY_ROUTES)) ctx.tools.register(defineTool({
    name: tool, description: descriptions[tool], parameters: parameters[tool],
    output: { schema: { type: 'json' }, render: (_args, value) => [{ type: 'text', text: JSON.stringify(value) }] },
    execute: async (args, exec) => {
      validateFamilyArguments(tool, args);
      const plan = ctx.get('agentPresets')?.serviceFor(exec.agent, 'planMode') ?? ctx.get('planMode');
      if (!['family_state', 'family_history'].includes(tool) && plan?.get(exec.agent)?.active) throw new Error('Family mutation is unavailable in plan mode');
      if (tool === 'family_control' && args.mode === 'mother' && !directHumanTurn(ctx, exec.agent)) {
        throw new Error('Resuming mother care requires a direct human turn');
      }
      // The family owner first commits its control boundary, then invokes our IPC pause/resume hook.
      // Pausing a goal can cancel this very tool's signal; control must still reach the body owner.
      return modelFamilyResult(await request(tool, args, tool === 'family_control' ? undefined : exec.signal));
    },
  }));

  // Use the official authenticated Connection route registry, not a bypass around its Host/Origin/cookie checks.
  const uiRequest = createFamilyClient(config.familyOrigin, { ui: true });
  for (const [tool, route] of Object.entries({ ...FAMILY_ROUTES, ...FAMILY_UI_ROUTES })) ctx.connection.fetch.register({
    path: route.path, methods: [route.method], requestBody: 'buffered',
    fetch: async incoming => {
      try {
        const args = route.method === 'POST' ? await incoming.json() : (() => {
          const after = new URL(incoming.url).searchParams.get('after');
          return after === null ? {} : { after: Number(after) };
        })();
        validateFamilyArguments(tool, args, { ui: true });
        const result = await uiRequest(tool, args, tool === 'family_control' ? undefined : incoming.signal);
        return Response.json(result);
      } catch (error) { return Response.json({ error: error.message }, { status: 400 }); }
    },
  });
  ctx.connection.fetch.register({
    path: '/api/family/events', methods: ['GET'], requestBody: 'buffered',
    fetch: async incoming => {
      const response = await fetch(new URL('/api/family/events', config.familyOrigin), {
        signal: incoming.signal, redirect: 'error', headers: { Accept: 'text/event-stream' },
      });
      return new Response(response.body, { status: response.status,
        headers: { 'Content-Type': response.headers.get('content-type') || 'text/event-stream', 'Cache-Control': 'no-cache' } });
    },
  });
}
