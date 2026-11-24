import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { buildIdentity, patchIdentityConversation, patchIdentityLayout } from './identity-build.mjs';
const root = path.dirname(fileURLToPath(import.meta.url));
export const DEFAULT_PACKAGE_ROOT = 'F:/16、杂货行账册/ledger-dsh/node_modules/@deepseek-ai';
const bridgeSource = `
const familyBridge = window.__MENGLI_NATIVE_FAMILY__ ??= {
  views: new Map(), drafts: new Map(), controller: null, listeners: new Set(), revision: 0,
  subscribe: listener => { familyBridge.listeners.add(listener); return () => familyBridge.listeners.delete(listener); },
  publish: () => { familyBridge.revision++; for (const listener of familyBridge.listeners) listener(); },
};
const familyView = id => familyBridge.views.get(id) === 'mengli';
`;
function copyDirectory(from, to) {
  fs.mkdirSync(to, { recursive: true });
  for (const entry of fs.readdirSync(from, { withFileTypes: true })) {
    const source = path.join(from, entry.name), target = path.join(to, entry.name);
    if (entry.isDirectory()) copyDirectory(source, target);
    else fs.writeFileSync(target, fs.readFileSync(source));
  }
}
function replace(source, before, after) {
  if (source.split(before).length !== 2) throw new Error(`Official hook anchor changed: ${before.slice(0, 90)}`);
  return source.replace(before, after);
}
export function patchConversation(source) {
  source = replace(source, 'var module = { exports: {} };', `var module = { exports: {} };\n${bridgeSource}`);
  source = replace(source, 'submissionState: () => {\n', 'familyView: () => familyView(session.sessionId),\n\t\t\t\t\tsubmissionState: () => {\n');
  source = replace(source, '...state.blank ? {} : { sessionId: state.sessionId },', 'familyView: familyView(session.sessionId),\n\t\t\t\t\t\t\t...state.blank ? {} : { sessionId: state.sessionId },');
  source = replace(source, 'submit(mode = "queue", source) {\n\t\t\t\tif (this.disposed) return;', `submit(mode = "queue", source) {
    if (this.disposed) return;
    if (this.deps.familyView?.() && (this.snapshot.phase !== 'plain' || this.attachmentIds.length || this.projection.occurrences.length || this.projection.quotes.length)) {
      this.notify('error', '梦漓视图只接受普通家长文字，不执行模型命令或附件'); return;
    }`);
  source = replace(source, 'adjudicate(attempt, draft) {\n\t\t\t\tconst inputTriggers = this.deps.inputTriggers?.();', `adjudicate(attempt, draft) {
    const inputTriggers = attempt.submission?.state?.familyView ? undefined : this.deps.inputTriggers?.();`);
  source = replace(source, 'sinkSerialized(attempt, draft, mode) {\n', `sinkSerialized(attempt, draft, mode) {
    if (attempt.submission?.state?.familyView) {
      this.settleSink(attempt, this.deps.defaultSink(draft, attempt.attachmentIds, mode, attempt.signal, () => this.acceptSink(attempt), true));
      return;
    }
`);
  source = replace(source, 'defaultSink: (text, attachmentIds, mode, signal, accept) => this.sink(session, text, attachmentIds, mode, signal, accept),', 'defaultSink: (text, attachmentIds, mode, signal, accept, body = false) => this.sink(session, text, attachmentIds, mode, signal, accept, body),');
  source = replace(source, 'sink(session, text, attachmentIds, mode, signal, accept) {', `sink(session, text, attachmentIds, mode, signal, accept, body = false) {
    if (body) {
      const family = familyBridge.controller;
      if (!family) return Promise.reject(new Error('家庭客户端不可用；已阻止默认模型发送'));
      const canonical = family.getSnapshot?.().state?.management_session_id;
      if (canonical && canonical !== session.sessionId) return Promise.reject(new Error('梦漓输入只接受家庭主会话'));
      if (attachmentIds.length) return Promise.reject(new Error('梦漓视图不接受模型附件'));
      return family.say(text, signal);
    }`);
  source = replace(source, 'inputTriggers: () => this.controller(actx),', 'inputTriggers: () => familyView(session.sessionId) ? undefined : this.controller(actx),');
  source = replace(source, 'const activateView = (sessionId, preferred) => {\n', `const activateView = (sessionId, preferred) => {
    const oldView = familyBridge.views.get(sessionId) ?? 'chat';
    const nextView = preferred ?? 'chat';
    if (familyBridge.views.has(sessionId) && oldView !== nextView && (oldView === 'mengli' || nextView === 'mengli')) {
      const shell = inputHub.shell(sessionId);
      if (['adjudicating', 'submitting', 'sending'].includes(shell.snapshot.phase)) {
        shell.notify('info', '请等本次提交完成后切换会话视图'); return false;
      }
      familyBridge.drafts.set(sessionId + ':' + oldView, shell.draftSnapshot);
      shell.setDraft(familyBridge.drafts.get(sessionId + ':' + nextView) ?? '');
    }
    familyBridge.views.set(sessionId, nextView); familyBridge.publish();
`);
  source = replace(source, 'activateView(sessionId, view);\n\t\t\t\t\t\tactions.openView', 'if (activateView(sessionId, view) === false) return;\n\t\t\t\t\t\tactions.openView');
  source = replace(source, 'activateView(sessionId, view);\n\t\t\t\t\t\tactions.setView', 'if (activateView(sessionId, view) === false) return;\n\t\t\t\t\t\tactions.setView');
  source = replace(source, 'const hero = sessionId === void 0 || shellPhase === "blank" && (openState === "open" || summaryBlank === true);', `react.useSyncExternalStore(familyBridge.subscribe, () => familyBridge.revision);
    const hero = sessionId === void 0 || !familyView(sessionId) && shellPhase === "blank" && (openState === "open" || summaryBlank === true);`);
  source = replace(source, 'function ConversationHeader({ sessionId, useSession, useConversation, renderSlot }) {', `function ConversationHeader({ sessionId, useSession, useConversation, renderSlot }) {
    react.useSyncExternalStore(familyBridge.subscribe, () => familyBridge.revision);`);
  source = replace(source, 'renderSlot("conversation.session.header", { hideChrome: blank })', 'renderSlot("conversation.session.header", { hideChrome: blank && !familyBridge.controller })');
  // Keep the original blank-session gate for mom, but a body view is meaningful even before mom has a model turn.
  source = replace(source, 'if (session.blank && conversationPhase(session, conversation) === "blank") return null;', 'if (active?.id !== "mengli" && session.blank && conversationPhase(session, conversation) === "blank") return null;');
  source = replace(source, 'const stop = (sessionId) => {', 'const stop = (sessionId) => {\n if (familyView(sessionId)) return;');
  source = replace(source, 'const promptError = useSession((s) => s.promptError) ?? null;', 'const promptError = useSession((s) => familyView(sessionId) ? null : s.promptError) ?? null;');
  source = replace(source, 'const running = useSession((s) => s.running) ?? false;', 'const running = useSession((s) => familyView(sessionId) ? false : s.running) ?? false;');
  source = replace(source, 'const planActive = useProjection("plan", (plan) => plan !== void 0 && (plan.pending ? !plan.active : plan.active));', 'const planActive = useProjection("plan", (plan) => !familyView(sessionId) && plan !== void 0 && (plan.pending ? !plan.active : plan.active));');
  source = replace(source, 'const hasGoal = useProjection("goal", (goal) => goal != null);', 'const hasGoal = useProjection("goal", (goal) => !familyView(sessionId) && goal != null);');
  source = replace(source, 'const disabled = removed || inert || !live || blocked !== void 0 || parentOffline;', `react.useSyncExternalStore(familyBridge.subscribe, () => familyBridge.revision);
    const familyActive = familyView(sessionId);
    const familyState = familyBridge.controller?.getSnapshot();
    const liveSteward = Boolean(familyState?.connected && familyState.state?.live_steward);
    const familyReady = liveSteward || (familyState?.connected && familyState.state?.loaded && familyState.state.mode === 'parent' && !familyState.state.busy && !familyState.state.pending_input && (!familyState.state.management_session_id || familyState.state.management_session_id === sessionId));
    if (familyActive) {
      placeholder = liveSteward ? '对梦漓说话（实时进入界膜）' : familyReady ? '对梦漓说话' : '先连接并接管，再对梦漓说话';
      toggleCommandMenu = undefined;
      const nativeRenderSlot = renderSlot;
      renderSlot = (name, ...args) => ['conversation.input.plan', 'conversation.input.permission', 'conversation.input.activity', 'conversation.input.overlay'].includes(name) ? null : nativeRenderSlot(name, ...args);
    }
    const disabled = removed || inert || !live || blocked !== void 0 || parentOffline || (familyActive && !familyReady);`);
  source = replace(source, 'renderSlot("conversation.input.left", {})', 'renderSlot("conversation.input.left", {}, familyActive ? { only: "mengli-family-controls" } : {})');
  source = replace(source, 'renderSlot("conversation.input.right", {})', 'renderSlot("conversation.input.right", {}, familyActive ? { only: "mengli-family-controls" } : {})');
  source = replace(source, 'sessionId === void 0 ? null : renderSlot("conversation.input.model", { locked: modelSeatLocked })', 'sessionId === void 0 || familyActive ? null : renderSlot("conversation.input.model", { locked: modelSeatLocked })');
  source = replace(source, 'variant === "composer" && input !== void 0 && sessionId !== void 0 ? renderSlot("conversation.composer.dock", {}) : null', 'variant === "composer" && input !== void 0 && sessionId !== void 0 && !familyActive ? renderSlot("conversation.composer.dock", {}) : null');
  source = replace(source, 'addFiles: (files, directories = /* @__PURE__ */ new Set()) => {', `addFiles: (files, directories = /* @__PURE__ */ new Set()) => {
    if (familyView(sessionId)) return '梦漓视图只接受家长文字';`);
  return patchIdentityConversation(source);
}
export function patchChat(source) {
  const hook = fs.readFileSync(path.join(root, 'chat-hook.js'), 'utf8');
  source = replace(source, 'function ChatView({ useSession,', 'function ChatView({ familyProjection, useSession,');
  source = replace(source, 'const { getTriggerProps } = useContextMenu();\n\t\t\tconst getMarkdownTriggerProps', `if (familyProjection) {
      ({ useChat, useSession, useConversation, useProjection, useInteractionIds, dispatchInteraction, chatScroll } = familyProjection);
      forkAt = undefined; inspectCall = undefined; loadOlder = () => {}; loadThrough = async () => {};
    }
    const { getTriggerProps } = useContextMenu();\n\t\t\tconst getMarkdownTriggerProps`);
  source = replace(source, 'children: renderSlot("conversation.chat.flow", {\n\t\t\t\t\t\t\t\t\t\t\tentries,', 'children: renderSlot("conversation.chat.flow", {\n familyProjection,\n\t\t\t\t\t\t\t\t\t\t\tentries,');
  const viewStart = source.indexOf('\n\t\t\tctx.slots.inject("conversation.view", () => {');
  const viewEndMarker = '}, ChatView);\n\t\t\t});';
  const viewEnd = source.indexOf(viewEndMarker, viewStart);
  if (viewStart < 0 || viewEnd < 0) throw new Error('Official ChatView registration anchor changed');
  let factory = source.slice(viewStart, viewEnd + viewEndMarker.length);
  factory = replace(factory, 'ctx.slots.inject("conversation.view", () => {\n\t\t\t\treturn ctx.slots.register({\n\t\t\t\t\tname: "conversation.view",\n\t\t\t\t\tid: "chat",\n\t\t\t\t\torder: 0,\n\t\t\t\t\tlabel: () => t("view.chat"),', 'ctx.slots.registerFactory({\n name: "conversation.chat.native-view", scope: "session",');
  factory = replace(factory, viewEndMarker, '}, ChatView);');
  factory += `\nctx.slots.inject('conversation.view', () => ctx.slots.register({ name: 'conversation.view', id: 'chat', order: 2, label: () => '和妈妈商量' }, props => props.renderFactorySlot('conversation.chat.native-view', { inspectCall: props.inspectCall, viewRequest: props.viewRequest, openView: props.openView, completeViewRequest: props.completeViewRequest })));\n`;
  source = source.slice(0, viewStart) + factory + source.slice(viewEnd + viewEndMarker.length);
  source = replace(source, 'const ChatFlow = (0, react.memo)(function ChatFlow({ entries,', 'const ChatFlow = (0, react.memo)(function ChatFlow({ familyProjection, entries,');
  source = replace(source, 'const nodeStore = useChat((snapshot) => snapshot.nodes);', `if (familyProjection) ({ useChat, useSession, useChatNode, useChatNodeBottom, useChatNodeProcess } = familyProjection);
    const nodeStore = useChat((snapshot) => snapshot.nodes);`);
  return replace(source, '\n\t\t\tctx.slots.inject("shell.overlay", () => ctx.slots.register({', `\n${hook}\n\t\t\tctx.slots.inject("shell.overlay", () => ctx.slots.register({`);
}
export function patchLayout(source) {
  source = replace(source, 'function apply(ctx) {', `function apply(ctx) {
    const bootTheme = document.documentElement.getAttribute('data-ds-theme-source');
    if (['light', 'dark', 'system'].includes(bootTheme)) ctx.theme.setTheme(bootTheme);`);
  return patchIdentityLayout(replace(source, 'openRightbar(track, fullscreen) {\n\t\t\t\tthis.panels.openRightbar(track, fullscreen);', 'setRightbarWidth(px) { this.panels.setRightbar(px); }\n\t\t\topenRightbar(track, fullscreen) {\n\t\t\t\tthis.panels.openRightbar(track, fullscreen);'));
}
export function patchRenderer(source) {
  // Keep the root outlet subscribed across a native registration handoff.
  return replace(source,
    `throw new SlotAssemblyError("renderSlot('root') before any 'root' registration (boot order)");`,
    'return null;');
}
/** The footer belongs to the local browser login, independently of provider accounts. */
export function patchAccountMenu(source) {
  const start = source.indexOf('\t\t\tconst requestSignOut = async () => {');
  const end = source.indexOf('\t\t\tconst beginSignIn =', start);
  if (start < 0 || end < 0) throw new Error('Official account menu logout hook changed');
  source = source.slice(0, start) + `
            const [logoutError, setLogoutError] = (0, react.useState)(null);
            const requestSignOut = async () => {
                if (busy) return;
                setBusy(true); setLogoutError(null);
                try {
                    const response = await fetch('/api/local-account/logout', {
                        method: 'POST', credentials: 'same-origin',
                        headers: { 'Content-Type': 'application/json' }, body: '{}'
                    });
                    if (!response.ok) {
                        const value = await response.json().catch(() => ({}));
                        throw new Error(value.error?.message ?? value.error ?? '退出失败，请重试');
                    }
                    window.location.replace('/');
                } catch (error) {
                    setLogoutError(error.message); setBusy(false);
                }
            };
` + source.slice(end);
  const menuStart = source.indexOf('\t\t\t\t\t\t\t...signedIn ? [{', start);
  const menuEnd = source.indexOf('\n\t\t\t\t\t\t],', menuStart);
  if (menuStart < 0 || menuEnd < 0) throw new Error('Official account menu items hook changed');
  source = source.slice(0, menuStart) + `
                            { id: "signout", label: "退出登录",
                              icon: (0, react_jsx_runtime.jsx)(LogoutIcon, {}), disabled: busy }
` + source.slice(menuEnd);
  return replace(source, '\t\t\t\t\tsignInNotice > 0 &&', `
                    logoutError && (0, react_jsx_runtime.jsx)(_deepseek_ai_dsh_client_ui_primitives.Toast, {
                        text: logoutError, onDone: () => setLogoutError(null)
                    }),
                    signInNotice > 0 &&`);
}
export function ensureNativeFamilyBuild({ packageRoot = DEFAULT_PACKAGE_ROOT } = {}) {
  const owned = path.join(root, 'official');
  for (const [name, patch] of [['ui-layout', patchLayout], ['ui-conversation', patchConversation], ['ui-chat', patchChat], ['ui-renderer', patchRenderer], ['ui-settings-account', patchAccountMenu]]) {
    const officialName = `@deepseek-ai/dsh-client-${name}`;
    const from = path.join(packageRoot, `dsh-client-${name}`);
    const to = path.join(owned, name);
    fs.mkdirSync(path.join(to, 'lib'), { recursive: true });
    const originalPackage = JSON.parse(fs.readFileSync(path.join(from, 'package.json'), 'utf8'));
    const manifest = { name: officialName, version: originalPackage.version, type: 'module', main: 'lib/index.js', exports: { '.': './lib/index.js', './client': './lib/client.js', './package.json': './package.json', './locale/*.json': './locale/*.json' }, dsh: originalPackage.dsh, license: originalPackage.license };
    fs.writeFileSync(path.join(to, 'package.json'), JSON.stringify(manifest, null, 2) + '\n');
    fs.writeFileSync(path.join(to, 'lib/client.js'), patch(fs.readFileSync(path.join(from, 'lib/client.js'), 'utf8')));
    fs.writeFileSync(path.join(to, 'lib/index.js'), `export * from ${JSON.stringify(pathToFileURL(path.join(from, 'lib/index.js')).href)};\n`);
    fs.copyFileSync(path.join(from, 'LICENSE'), path.join(to, 'LICENSE'));
    if (fs.existsSync(path.join(from, 'locale'))) copyDirectory(path.join(from, 'locale'), path.join(to, 'locale'));
    if (fs.existsSync(path.join(from, 'lib/assets'))) copyDirectory(path.join(from, 'lib/assets'), path.join(to, 'lib/assets'));
  }
  const stateSource = fs.readFileSync(path.join(root, 'family-state.mjs'), 'utf8').replaceAll('export ', '');
  const clientSource = fs.readFileSync(path.join(root, 'account-client.js'), 'utf8') + '\n'
    + fs.readFileSync(path.join(root, 'client-source.js'), 'utf8');
  // Reuse the loaded official settings form's own field classes; no new stylesheet.
  const officialForm = fs.readFileSync(path.join(packageRoot, 'dsh-client-ui-settings-models/lib/client.js'), 'utf8');
  const classMap = officialForm.match(/var ModelsSection_module_css_default = (\{[\s\S]*?\n\s*\});/);
  if (!classMap) throw new Error('Official settings form class map changed');
  const formClasses = JSON.parse(classMap[1]);
  fs.mkdirSync(path.join(root, 'lib'), { recursive: true });
  fs.writeFileSync(path.join(root, 'lib/client.js'), `window.__ModuleLoader__.load({id:'@mengli/native-family-client',factory(require){\nconst OFFICIAL_FORM_CLASSES = ${JSON.stringify(formClasses)};\n${stateSource}\n${clientSource}\nreturn {apply, inject};\n}});\n`);
  buildIdentity({ packageRoot });
  return { root, owned };
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const built = ensureNativeFamilyBuild({ packageRoot: process.env.MENGLI_DSH_PACKAGE_ROOT || DEFAULT_PACKAGE_ROOT });
  console.log(`Native family bundles built in ${built.root}`);
}
