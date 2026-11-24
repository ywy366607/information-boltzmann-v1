const React = require('react');
const { Button, Menu, StateDot } = require('@deepseek-ai/dsh-client-ui-primitives');
const chatModule = require('@deepseek-ai/dsh-client-ui-chat');
require('@deepseek-ai/dsh-client-ui-settings-models'); // Owns OFFICIAL_FORM_CLASSES and its stylesheet.
const inject = ['slots', 'sessions', 'uiConversation', 'sidebarRight', 'sidebarRightTabs', 'layout'];
function apply(ctx) {
  const family = createFamilyClient();
  const bridge = window.__MENGLI_NATIVE_FAMILY__;
  if (!bridge || !chatModule.registerNativeFamilyView) throw new Error('Native family official module hooks are not installed');
  bridge.controller = family;
  const unsubscribe = family.subscribe(bridge.publish);
  ctx.effect(() => () => { unsubscribe(); family.dispose(); bridge.controller = null; bridge.publish(); });
  chatModule.registerNativeFamilyView(ctx, family, projectFamilyEvents, projectFamilyHistory);
  family.connect();
  const h = React.createElement;
  const account = installLocalAccountUI(ctx);
  const useFamily = () => React.useSyncExternalStore(family.subscribe, family.getSnapshot);
  const command = (name, body) => family.command(name, body).catch(() => {});
  const openPlan = () => { ctx.layout.setRightbarWidth(380); ctx.sidebarRight.openTab('mengli-family-plan', { destination: 'sidebar' }); };

  const umweltStyle = document.createElement('style');
  umweltStyle.dataset.plugin = '@mengli/umwelt-ui';
  umweltStyle.textContent = `
    .umwelt-source-row { display: inline-flex; align-items: center; gap: 6px; flex-wrap: wrap; font-size: 12px; color: var(--dsh-text-secondary, #9ca3af); }
    .umwelt-badge { display: inline-flex; align-items: center; padding: 1px 6px; border-radius: 999px; font-size: 11px; line-height: 18px; border: 1px solid rgba(148,163,184,0.28); background: rgba(148,163,184,0.08); color: inherit; }
    .umwelt-cite { position: relative; display: inline-flex; align-items: center; padding: 1px 7px; border-radius: 999px; font-size: 11px; line-height: 18px; cursor: default; border: 1px solid rgba(99,179,237,0.35); background: rgba(99,179,237,0.10); color: var(--dsh-text-primary, inherit); }
    .umwelt-cite.stumble { border-color: rgba(245,158,11,0.40); background: rgba(245,158,11,0.12); }
    .umwelt-cite-pop { display: none; position: absolute; left: 0; top: calc(100% + 6px); z-index: 80; width: min(380px, 72vw); padding: 10px 12px; border-radius: 10px; border: 1px solid rgba(148,163,184,0.28); background: var(--dsh-surface-elevated, #181b20); color: var(--dsh-text-primary, #e5e7eb); box-shadow: 0 10px 28px rgba(0,0,0,0.35); font-size: 12px; line-height: 1.55; white-space: pre-wrap; word-break: break-word; pointer-events: none; }
    .umwelt-cite:hover .umwelt-cite-pop, .umwelt-cite:focus-within .umwelt-cite-pop { display: block; }
    .umwelt-cite-title { font-weight: 600; font-size: 11px; opacity: 0.72; margin-bottom: 4px; }
    .umwelt-traj { height: 100%; overflow: auto; padding: 18px 22px 32px; box-sizing: border-box; font-family: var(--dsh-font-family-text, var(--dsw-font-family)); color: var(--dsh-text-primary, #e5e7eb); }
    .umwelt-traj-bar { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 14px; }
    .umwelt-traj-stats { display: flex; flex-wrap: wrap; gap: 10px; }
    .umwelt-stat { padding: 6px 10px; border-radius: 8px; border: 1px solid rgba(148,163,184,0.20); background: rgba(148,163,184,0.06); font-size: 12px; }
    .umwelt-stat b { font-weight: 600; margin-left: 6px; }
    .umwelt-filters { display: inline-flex; gap: 6px; }
    .umwelt-filter { padding: 4px 10px; border-radius: 999px; font-size: 12px; border: 1px solid rgba(148,163,184,0.28); background: transparent; color: inherit; cursor: pointer; }
    .umwelt-filter.active { background: rgba(99,179,237,0.18); border-color: rgba(99,179,237,0.55); font-weight: 600; }
    .umwelt-traj-list { display: flex; flex-direction: column; gap: 8px; }
    .umwelt-traj-row { padding: 10px 12px; border-radius: 10px; border: 1px solid rgba(148,163,184,0.18); background: rgba(148,163,184,0.04); display: flex; flex-direction: column; gap: 6px; }
    .umwelt-traj-row.social { border-color: rgba(99,179,237,0.32); background: rgba(99,179,237,0.05); }
    .umwelt-traj-top { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; font-size: 12px; }
    .umwelt-traj-cyc { font-variant-numeric: tabular-nums; font-weight: 600; opacity: 0.78; }
    .umwelt-traj-route { font-weight: 600; }
    .umwelt-traj-metrics { margin-left: auto; display: inline-flex; gap: 10px; font-size: 11px; font-variant-numeric: tabular-nums; opacity: 0.85; }
    .umwelt-pos { color: #34d399; }
    .umwelt-neg { color: #f87171; }
    .umwelt-traj-body { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.4fr); gap: 10px; font-size: 12.5px; line-height: 1.5; }
    .umwelt-traj-cell { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; opacity: 0.9; }
    .umwelt-traj-cell span { opacity: 0.55; margin-right: 6px; font-size: 11px; }
    @media (max-width: 720px) { .umwelt-traj-body { grid-template-columns: 1fr; } }
  `;
  document.head.appendChild(umweltStyle);
  ctx.effect(() => () => umweltStyle.remove());

  const STAGE_NAMES = {
    infant_direct_modeling: '启蒙示范',
    emerging_scaffolded: '支架引导',
    reflective_inquiry: '自主探究',
  };

  function CitationPill({ label, title, text, stumble = false }) {
    if (!text) return null;
    return h('span', { className: 'umwelt-cite' + (stumble ? ' stumble' : ''), tabIndex: 0 },
      label,
      h('span', { className: 'umwelt-cite-pop', role: 'tooltip' },
        h('div', { className: 'umwelt-cite-title' }, title),
        text));
  }

  function Controls({ sessionId, compact = false }) {
    const snapshot = useFamily();
    const state = snapshot.state;
    const [pending, setPending] = React.useState(false);
    const [open, setOpen] = React.useState(false);
    if (state?.management_session_id && sessionId !== state.management_session_id) return null;
    const run = async (name, body) => { setPending(true); try { await command(name, body); } finally { setPending(false); } };
    const busy = pending || !snapshot.connected;
    const observedStatus = { running: '环世界运行中', completed: '已完成', complete: '已完成', stopped: '已保存停止', interrupted: '中断', failed: '失败', waiting_resources: '等待资源', unavailable: '观察不可用' };
    const label = !snapshot.connected ? '离线' : state?.live_steward ? '环世界 · 实时协作' : state?.read_only ? (observedStatus[state.observation?.status] ?? '观察中')
      : !state?.loaded ? '身体未连接' : state.mode === 'parent' ? '家长接管' : state.mode === 'mother' ? '妈妈陪读' : state.mode === 'pausing' ? '正在交接' : '已暂停';
    const items = (state?.read_only || state?.live_steward) ? [
      { id: 'observer', label: state?.live_steward ? '实时环世界 · 可调整计划与发言' : '环世界已保存 · 观察模式', disabled: true },
      { id: 'plan', label: state?.live_steward ? '调整育儿计划' : '查看育儿计划' },
    ] : [
      ...(!state?.loaded ? [{ id: 'load', label: '连接身体', disabled: busy || state?.busy }] : []),
      ...(state?.mode !== 'parent' ? [{ id: 'parent', label: '家长接管', disabled: busy }] : []),
      ...(state?.mode !== 'paused' ? [{ id: 'paused', label: '暂停', disabled: busy }] : []),
      ...(state?.mode !== 'mother' ? [{ id: 'mother', label: '恢复妈妈陪读', disabled: busy || !state?.loaded }] : []),
      { id: 'plan', label: '育儿计划' },
    ];
    return h(React.Fragment, null,
      h(Menu, { open, onClose: () => setOpen(false), items, align: 'end', portal: true,
        onSelect: id => {
          setOpen(false);
          if (id === 'plan') openPlan();
          else if (id === 'load') run('load', {});
          else run('control', { mode: id });
        },
        anchor: h(Button, { size: 'sm', className: 'mengli-care-menu', 'aria-label': '环世界状态', 'aria-haspopup': 'menu', 'aria-expanded': open,
          title: snapshot.error ?? label, onClick: () => setOpen(value => !value) },
          h(StateDot, { state: snapshot.connected ? 'success' : 'error' }), ' ', compact ? label : label),
      }),
      !compact && h(Button, { size: 'sm', className: 'mengli-plan-shortcut', onClick: () => openPlan() }, '育儿计划'));
  }
  ctx.slots.inject('conversation.session.header.actions', () => ctx.slots.register({ name: 'conversation.session.header.actions', id: 'mengli-family-controls', order: -10 }, Controls));
  ctx.slots.inject('conversation.input.right', () => ctx.slots.register({ name: 'conversation.input.right', id: 'mengli-family-controls', order: -10 }, props => h(Controls, { ...props, compact: true })));

  function SourceLabel({ source }) {
    const profile = account.useAccount();
    const parent = source.kind === 'user' || (source.kind === 'family' && source.actor === 'parent');
    const u = source.umwelt;
    const perspectiveLabel = u?.perspective_mode === 'mirror_visible' ? '可见视界' : u?.perspective_mode === 'book_only' ? '仅见书页' : null;
    const triggerBadge = u?.trigger_kind === 'mother_proactive' ? '主动开口' : u?.trigger_kind === 'child_replied' ? '接续对话' : null;
    return h('span', { className: 'umwelt-source-row' },
      h('small', null, parent ? profile.displayName : source.label),
      perspectiveLabel && h('span', { className: 'umwelt-badge' }, perspectiveLabel),
      triggerBadge && h('span', { className: 'umwelt-badge' }, triggerBadge),
      u?.mirror_excerpt && h(CitationPill, { label: '卡顿片段', title: '梦漓刚刚卡顿的 64 字节书页片段', text: u.mirror_excerpt, stumble: true }),
      u?.book_excerpt && h(CitationPill, { label: '共读书页', title: '当前共读书页引文', text: u.book_excerpt }));
  }
  ctx.slots.inject('conversation.chat.user.before', () => ctx.slots.register({
    name: 'conversation.chat.user.before', id: 'mengli-source', order: -10,
    select: props => ['user', 'family', 'family-history'].includes(props.source?.kind) ? { source: props.source } : null,
  }, SourceLabel));

  function UmweltTrajectoryView() {
    const snapshot = useFamily();
    const [filter, setFilter] = React.useState('all');
    const items = React.useMemo(() => projectUmweltTrajectory(snapshot.state?.events ?? []), [snapshot.state?.events]);
    const filtered = React.useMemo(() => {
      if (filter === 'social') return items.filter(it => it.isSocial);
      if (filter === 'book') return items.filter(it => !it.isSocial);
      return items;
    }, [items, filter]);

    const socialCount = items.filter(it => it.isSocial).length;
    const mirrorCount = items.filter(it => it.perspective === 'mirror_visible').length;
    const bookOnlyCount = items.filter(it => it.perspective === 'book_only').length;
    const latestStage = items.find(it => it.stage);
    const stageText = latestStage ? `${STAGE_NAMES[latestStage.stage] ?? latestStage.stage}${latestStage.planVersion ? ' v' + latestStage.planVersion : ''}` : '—';

    return h('div', { className: 'umwelt-traj' },
      h('div', { className: 'umwelt-traj-bar' },
        h('div', { className: 'umwelt-traj-stats' },
          h('div', { className: 'umwelt-stat' }, '轮次', h('b', null, snapshot.state?.cycle ?? '—')),
          h('div', { className: 'umwelt-stat' }, '育儿阶段', h('b', null, stageText)),
          h('div', { className: 'umwelt-stat' }, '社交交互', h('b', null, `${socialCount} / ${items.length}`)),
          h('div', { className: 'umwelt-stat' }, '视界', h('b', null, `可见 ${mirrorCount} · 仅书页 ${bookOnlyCount}`))),
        h('div', { className: 'umwelt-filters' },
          [['all', '全部'], ['social', '社交与工具'], ['book', '书本脉冲']].map(([id, text]) =>
            h('button', { key: id, type: 'button', className: 'umwelt-filter' + (filter === id ? ' active' : ''), onClick: () => setFilter(id) }, text)))),
      !filtered.length
        ? h('div', { style: { opacity: 0.6, padding: '24px 0', fontSize: 13 } }, '暂无界膜轨迹记录')
        : h('div', { className: 'umwelt-traj-list' },
            filtered.map(it => h('div', { key: it.cycle, className: 'umwelt-traj-row' + (it.isSocial ? ' social' : '') },
              h('div', { className: 'umwelt-traj-top' },
                h('span', { className: 'umwelt-traj-cyc' }, `#${it.cycle}`),
                h('span', { className: 'umwelt-traj-route' }, it.route),
                it.perspective && h('span', { className: 'umwelt-badge' }, it.perspective === 'mirror_visible' ? '可见视界' : '仅见书页'),
                it.mirrorExcerpt && h(CitationPill, { label: '卡顿片段', title: '触发停顿的书页片段', text: it.mirrorExcerpt, stumble: true }),
                it.bookExcerpt && h(CitationPill, { label: '书页引文', title: '共读书页原文', text: it.bookExcerpt }),
                h('span', { className: 'umwelt-traj-metrics' },
                  it.metalG !== null && h('span', { className: it.metalG >= 0 ? 'umwelt-pos' : 'umwelt-neg' }, `ΔG ${it.metalG >= 0 ? '+' : ''}${it.metalG.toFixed(4)}`),
                  it.nll !== null && h('span', null, `NLL ${it.nll.toFixed(2)}`),
                  it.hitRate !== null && h('span', null, `缓存 ${(it.hitRate * 100).toFixed(0)}%`))),
              h('div', { className: 'umwelt-traj-body' },
                h('div', { className: 'umwelt-traj-cell', title: it.utterance || '（安静）' }, h('span', null, '梦漓'), it.utterance || '（安静）'),
                h('div', { className: 'umwelt-traj-cell', title: it.reply }, h('span', null, it.isSocial ? '回应' : '书页'), it.reply || '—'))))));
  }
  ctx.slots.inject('conversation.view', () => ctx.slots.register({
    name: 'conversation.view', id: 'umwelt-trajectory', order: 1, label: () => '界膜轨迹',
  }, UmweltTrajectoryView));

  const planId = '@mengli/native-family-client/plan';
  ctx.effect(() => ctx.sidebarRightTabs.register({ id: planId, kind: 'mengli-family-plan', keepMounted: true, title: () => '育儿计划' }));
  function Plan() {
    const snapshot = useFamily();
    const state = snapshot.state;
    const [draft, setDraft] = React.useState('');
    const [baseVersion, setBaseVersion] = React.useState(null);
    const [dirty, setDirty] = React.useState(false);
    const [saving, setSaving] = React.useState(false);
    const [error, setError] = React.useState(null);
    React.useEffect(() => {
      if (!dirty && state?.plan) { setDraft(state.plan.text ?? ''); setBaseVersion(state.plan.version); }
    }, [state?.plan?.version, dirty]);
    const save = async () => {
      setSaving(true); setError(null);
      try { await family.command('plan', { text: draft, version: baseVersion }); setDirty(false); }
      catch (failure) { setError(failure.message); }
      finally { setSaving(false); }
    };
    const canEdit = Boolean(state?.live_steward || !state?.read_only);
    const form = OFFICIAL_FORM_CLASSES;
    return h('section', { className: form.section, style: { padding: 16, fontFamily: 'var(--dsh-font-family-text,var(--dsw-font-family))', fontSize: 14, lineHeight: '22px', overflow: 'auto', height: '100%', boxSizing: 'border-box' } },
      h('div', { className: form.editorTitle }, '育儿计划'),
      h('p', { className: form.intro }, state?.live_steward ? '实时环世界已接通，保存后下一轮立即生效；也可以在「和妈妈商量」里直接让妈妈代为修改。' : !canEdit ? '当前为已完成存档的只读观察视图。' : '和妈妈约定接下来的陪伴方式，新计划从下一轮开始'),
      h('label', { className: form.field }, h('span', { className: form.fieldLabel }, '计划内容'), h('textarea', { className: form.input, 'aria-label': '育儿计划内容', readOnly: !canEdit, value: draft, rows: 12, onChange: event => { setDraft(event.target.value); setDirty(true); }, style: { height: 240, minHeight: 160, padding: '8px 10px', resize: 'vertical' } })),
      h('p', { className: form.intro }, `已保存 v${state?.plan?.version ?? '—'} / 生效 v${state?.plan?.effective_version ?? '—'}`),
      dirty && baseVersion !== state?.plan?.version && h('p', { role: 'alert' }, '计划已由另一处更新；当前草稿尚未覆盖新版本。'),
      h('div', { className: form.editorActions },
        h(Button, { size: 'sm', disabled: !canEdit || !snapshot.connected || !dirty || saving, onClick: save }, saving ? '保存中' : '保存计划'),
        h(Button, { size: 'sm', disabled: !dirty || saving, onClick: () => setDirty(false) }, '放弃草稿')),
      error && h('p', { className: form.error, role: 'alert' }, error));
  }
  ctx.slots.inject('sidebar.right.pane.tab', () => ctx.slots.register({ name: 'sidebar.right.pane.tab', key: planId }, Plan));
}
