import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { DEFAULT_PACKAGE_ROOT, metricsPatchRows } from '../build.mjs';
import { cleanOfficialIndex } from '../../workbench_plugin.mjs';
const here = path.dirname(fileURLToPath(import.meta.url));

/** Real official web app with a single authenticated fixture GET. No family worker/model. */
export async function startMetricsBrowserFixture() {
  const packageRoot = process.env.MENGLI_DSH_PACKAGE_ROOT || DEFAULT_PACKAGE_ROOT;
  const root = path.join(here, '.fixture-runtime', String(Date.now()));
  fs.mkdirSync(root, { recursive: true });
  // Official persistence resolves its home during import, before runProfile().
  process.env.DSH_HOME = root;
  process.env.DSH_TELEMETRY_DISABLED = '1';
  const load = (name, entry = 'lib/index.js') => import(pathToFileURL(path.join(packageRoot, name, entry)).href);
  const [boot, launcher, environment] = await Promise.all([load('dsh-app-boot'), load('dsh', 'lib/profile-boot.js'), load('dsh-launch-environment')]);
  fs.writeFileSync(path.join(root, 'package.json'), JSON.stringify({ name: '@mengli/metrics-ui-fixture-profile', private: true, type: 'module',
    dsh: { profile: { bundles: ['@deepseek-ai/dsh-base', '@deepseek-ai/dsh-web-app'] } } }));
  // Cordis discovers the nearest package identity. Keep this server-only fixture
  // outside the production client's identity and omit dsh.client entirely.
  const dataPlugin = path.join(root, 'fixture-data');
  fs.mkdirSync(dataPlugin);
  fs.writeFileSync(path.join(dataPlugin, 'package.json'), JSON.stringify({ name: '@mengli/metrics-ui-fixture-data', private: true, type: 'module' }));
  fs.copyFileSync(path.join(here, 'fixture-plugin.mjs'), path.join(dataPlugin, 'index.mjs'));
  const disabled = ['preset-standard', 'preset-minimal', 'preset-ptc', 'preset-cordis', 'plugin-manager', 'hmr',
    'config-editor', 'settings', 'deepseek-account', 'llm-deepseek-account', 'account-controller', 'permission',
    'ui-permission', 'ui-deliverables', 'session-title-llm', 'session-telemetry-otel', 'desktop-product-telemetry', 'product-analytics',
    'subprocess', 'bash-sandbox', 'pwsh-sandbox', 'tool-working-directory', 'tool-bash', 'tool-pwsh', 'tool-fs',
    'tool-fs-search', 'tool-jobs', 'tool-skill', 'skill-filesystem', 'tool-subagent', 'tool-subagent-fork',
    'tool-subagent-control', 'workflow-ptc', 'ptc-runtime', 'subagent-spawn-in-process', 'subagent-fork-in-process',
    'tool-web', 'web-search-deepseek', 'web-deepseek-api', 'open-in-app', 'directory-picker', 'terminal-controller',
    'workspace-files', 'file-reference-local', 'session-reference', 'workspace-changes', 'schedule',
    'harness-dev-context', 'cordis-inspect-providers', 'ui-sidebar-terminal', 'ui-sidebar-files', 'ui-plugin-manager',
    'ui-settings-shell', 'ui-settings-subagent', 'ui-settings-web-search', 'ui-settings-plugins', 'ui-settings-plugin-inventory',
    'ui-cordis', 'ui-schedule'];
  const patch = [...disabled.map(id => ({ id, disabled: true })),
    { id: 'credentials', config: { watch: false } },
    { id: 'session-controller', config: { nativeOpen: false } },
    { id: 'agent-preset-registry', config: { default: 'metrics-ui-fixture' } },
    { id: 'agent-default-model', config: { provider: 'deepseek-official', model: 'fixture-never-called' } },
    { id: 'web-runtime', config: { openBrowser: false, printUrl: false, surfaceContext: false } },
    { id: 'llm-deepseek', config: { apiKeyEnv: 'MENGLI_FIXTURE_NO_KEY', models: [{ id: 'fixture-never-called', name: 'Offline fixture', contextWindow: 32768, maxTokens: 1024 }] } },
    { insert: [{ id: 'metrics-fixture-data', name: path.join(dataPlugin, 'index.mjs') },
      { id: 'preset-metrics-fixture', name: '@deepseek-ai/dsh-agent-preset', config: { id: 'metrics-ui-fixture', order: 0, plugins: [] } }] },
    ...metricsPatchRows({ packageRoot })];
  fs.writeFileSync(path.join(root, 'cordis.yml'), '[]\n');
  fs.writeFileSync(path.join(root, 'cordis.patch.yml'), JSON.stringify(patch));
  const installAnchor = path.join(packageRoot, 'dsh/package.json');
  const profile = boot.loadProfileDirectory('metrics-browser-fixture', root, installAnchor);
  const { ctx } = await launcher.runProfile({ profile: 'metrics-browser-fixture', resolvedProfile: { profile, installAnchor }, patchFiles: [],
    args: ['--host', '127.0.0.1', '--port', '0', '--no-open'],
    environment: environment.createLaunchEnvironmentSnapshot([{ source: 'process', values: { ...process.env, DSH_HOME: root, DSH_TELEMETRY_DISABLED: '1' } }]) });
  ctx.webServer.tapIndex(cleanOfficialIndex);
  const sessionId = 'metrics-fixture-only';
  const handle = await ctx.agents.create({ sessionId, meta: { cwd: root, agentPreset: 'metrics-ui-fixture' },
    setup: async agentCtx => { await ctx.agentPresets.mount(agentCtx, 'metrics-ui-fixture'); } });
  // A persisted UI-only message makes the stock blank-session header visible.
  // Appending an observation does not submit a prompt or execute the agent loop.
  handle.agent.session.append('turn/start', { turn: 1 });
  handle.agent.session.append('user/message', { id: 'metrics-fixture-static-message', role: 'user',
    content: [{ type: 'text', text: 'Offline dashboard UI fixture — no model or body is running.' }],
    source: { kind: 'metrics-ui-fixture' } }, { surfaceOp: 'append' });
  handle.agent.session.append('turn/end', { turn: 1, reason: { kind: 'completed' } });
  await ctx.sessions.flush(handle.agent.session);
  const workspace = await ctx.workspaceRegistry.create(root, 'Metrics · Offline fixture');
  await workspace.attachSession(sessionId);
  ctx.on('webserver/index-inject', rows => rows.push({ kind: 'script', placement: 'head',
    text: `localStorage.setItem('dsh.sessions.current',${JSON.stringify(JSON.stringify({ sessionId }))});` }));
  const origin = ctx.webServer.getOrigin();
  return { url: ctx.connection.authenticatedUrl(origin), close: async () => {
    await ctx.fiber.dispose();
    try { fs.rmSync(root, { recursive: true, force: true }); } catch {}
  } };
}
