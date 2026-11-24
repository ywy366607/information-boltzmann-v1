const React = require('react');
const { Button, Menu } = require('@deepseek-ai/dsh-client-ui-primitives');
const h = React.createElement;
const inject = ['slots', 'sidebarRight', 'sidebarRightTabs', 'layout'];
const LIFE_KIND = 'mengli-life-metrics';
const LEARNING_KIND = 'mengli-learning-metrics';

function apply(ctx) {
  const client = createMetricsClient();
  const style = document.createElement('style');
  style.dataset.plugin = '@mengli/native-metrics-client';
  style.textContent = METRICS_CSS;
  document.head.appendChild(style);
  let lastCard;
  const unbindLifeState = client.subscribe(() => {
    const card = client.getSnapshot().card;
    if (card && card !== lastCard) {
      lastCard = card;
      window.dispatchEvent(new CustomEvent('mengli:life-state', { detail: { card } }));
    }
  });
  ctx.effect(() => () => { unbindLifeState(); client.dispose(); style.remove(); });
  const useMetrics = () => {
    const snapshot = React.useSyncExternalStore(client.subscribe, client.getSnapshot);
    React.useEffect(() => { ctx.layout.setRightbarWidth?.(420); client.open(); }, []);
    return snapshot;
  };
  let lifeOpened = false;
  const open = kind => {
    if (kind === LIFE_KIND) lifeOpened = true;
    // Existing native-family layout exposes this native action; unpatched DSH keeps its own width.
    ctx.layout.setRightbarWidth?.(420);
    ctx.sidebarRight.openTab(kind, { destination: 'sidebar' });
  };
  const tabs = [
    { kind: LIFE_KIND, title: '生命体征', description: '记忆与分层动量' },
    { kind: LEARNING_KIND, title: '学习训练', description: '更新、预测误差与学习进展' },
  ];
  for (const [index, tab] of tabs.entries()) {
    const id = '@mengli/native-metrics-client/' + tab.kind;
    ctx.effect(() => ctx.sidebarRightTabs.register({ id, kind: tab.kind, keepMounted: true,
      title: () => tab.title,
      guide: [{ id: tab.kind, order: 30 + index, title: () => tab.title, description: () => tab.description }],
    }));
    ctx.slots.inject('sidebar.right.pane.tab', () => ctx.slots.register({ name: 'sidebar.right.pane.tab', key: id },
      () => h(MetricsPanel, { kind: tab.kind })));
  }
  // Keep both desktop shortcuts; the mobile doorway is one official Menu.
  function HeaderControls() {
    const [menuOpen, setMenuOpen] = React.useState(false);
    // The native session seat exists by this commit; plugin apply can precede it.
    React.useEffect(() => {
      const enter = () => { if (!lifeOpened) open(LIFE_KIND); };
      window.addEventListener('mengli:enter', enter);
      // Covers earlier entry, including the first launch with no intro playback.
      if (window.MengliIntro?.entered) enter();
      return () => window.removeEventListener('mengli:enter', enter);
    }, []);
    return h('span', { className: 'mengli-metrics-actions' },
      h('span', { className: 'mm-header-desktop' }, tabs.map(tab => h(Button, { key: tab.kind, size: 'sm',
        onClick: () => open(tab.kind), 'aria-label': '打开' + tab.title },
        tab.kind === LIFE_KIND && h(CoreAnchor), tab.title))),
      h('span', { className: 'mm-header-mobile' }, h(Menu, { open: menuOpen, onClose: () => setMenuOpen(false),
        items: tabs.map(tab => ({ id: tab.kind, label: tab.title })), align: 'end', portal: true,
        onSelect: kind => { setMenuOpen(false); open(kind); },
        anchor: h(Button, { size: 'sm', 'aria-label': '生命与学习面板', 'aria-haspopup': 'menu', 'aria-expanded': menuOpen,
          onClick: () => setMenuOpen(value => !value) }, h(CoreAnchor), '面板') })));
  }
  ctx.slots.inject('conversation.session.header.actions', () => ctx.slots.register({
    name: 'conversation.session.header.actions', id: 'mengli-metrics-open', order: 20,
  }, HeaderControls));

  function CoreAnchor({ dock = false }) {
    const ref = React.useRef(null);
    React.useEffect(() => {
      const notify = () => window.dispatchEvent(new Event('mengli:core-anchor'));
      const observer = new ResizeObserver(notify);
      observer.observe(ref.current);
      notify();
      return () => { observer.disconnect(); notify(); };
    }, []);
    return h('span', { ref, className: dock ? 'mm-core-dock' : 'mm-core-park',
      [dock ? 'data-mengli-core-dock' : 'data-mengli-core-park']: '', 'aria-hidden': true });
  }
  function Hero({ label, value, unit, integer = false }) {
    return h('div', null, h('div', { className: 'mm-label' }, label),
      h('div', { className: 'mm-value', title: String(value ?? '未测') }, formatMetric(value, integer)),
      h('div', { className: 'mm-unit' }, unit));
  }
  function Facts({ rows }) {
    return h('dl', { className: 'mm-facts' }, rows.map(([label, value]) => h(React.Fragment, { key: label },
      h('dt', null, label), h('dd', null, value))));
  }
  function Section({ title, note, children }) {
    return h('section', { className: 'mm-section' }, h('div', { className: 'mm-section-head' },
      h('h3', null, title), note && h('span', { className: 'mm-note' }, note)), children);
  }
  function NormBars({ norms, title, note }) {
    if (!norms?.length) return null;
    const measured = norms.filter(isMeasured);
    const max = measured.length ? Math.max(0, ...measured) : null;
    return h(Section, { title, note }, h('div', { role: 'group', 'aria-label': title },
      norms.map((value, index) => h('div', { className: 'mm-bar-row', key: index, tabIndex: 0,
        title: `第${index + 1}层 · ${value ?? '未测'}`,
        'aria-label': `第${index + 1}层，${formatMetric(value)}` },
        h('span', null, 'L' + (index + 1)),
        h('div', { className: 'mm-track', 'aria-hidden': true }, isMeasured(value) && h('div', {
          className: 'mm-bar', style: { width: (max > 0 ? Math.max(0, value) / max * 100 : 0) + '%' },
        })), h('span', { className: 'mm-bar-value' }, formatMetric(value))))),
      );
  }
  function LinePlot({ rows, path, title, unit }) {
    const allPoints = metricPoints(rows, path);
    const [hover, setHover] = React.useState(null);
    const [range,setRange]=React.useState({});
    const [draft,setDraft]=React.useState({from:'',to:'',low:'',high:''});
    const [error,setError]=React.useState('');
    if (!allPoints.length) return null;
    if (allPoints.length === 1) {
      const point = allPoints[0];
      return h('section', { className: 'mm-section mm-single-record', 'data-metric': path,
        title: `${title} · cycle ${point.x} · ${point.y} ${unit}` },
        h('div', { className: 'mm-single-heading' }, h('h3', null, title)),
        h('div', { className: 'mm-single-value' }, formatMetric(point.y)),
        h('p', { className: 'mm-note' }, `cycle ${point.x}${unit?' · '+unit:''}`));
    }
    const requested=metricViewport(allPoints,range);
    const viewport=requested.error?metricViewport(allPoints):requested;
    const {points,minX,maxX,low,high}=viewport;
    const rangeControl=h('details',{className:'mm-plot-range'},h('summary',null,'范围'),
      h('form',{'aria-label':title+'范围',onSubmit:event=>{
        event.preventDefault();const next=metricViewport(allPoints,draft);
        if(next.error){setError(next.error);return;}setRange({...draft});setHover(null);setError('');
      }},h('div',{className:'mm-range-grid'},[['from','cycle 起点',minX],['to','cycle 终点',maxX],['low','数值下限',low],['high','数值上限',high]].map(([key,label,value])=>
        h('div',{key},h('label',{htmlFor:`mm-${path}-${key}`},label),h('input',{id:`mm-${path}-${key}`,name:key,type:'number',step:key==='from'||key==='to'?'1':'any',value:draft[key],placeholder:String(Math.round(value*10000)/10000),'aria-invalid':error?'true':undefined,'aria-describedby':error?`mm-${path}-error`:undefined,
          onChange:event=>{setDraft({...draft,[key]:event.target.value});setError('');}})))),
        h('div',{className:'mm-actions'},h('button',{type:'submit'},'应用'),h('button',{type:'button',onClick:()=>{setDraft({from:'',to:'',low:'',high:''});setRange({});setHover(null);setError('');}},'自动')),
        error&&h('p',{id:`mm-${path}-error`,className:'mm-error',role:'alert'},error)));
    if(!points.length)return h(Section,{title,note:unit},rangeControl,h('p',{className:'mm-empty'},'此范围暂无记录'));
    const span = high - low || 1;
    const x = point => minX === maxX ? 183 : 42 + (point.x - minX) / (maxX - minX) * 282;
    const y = point => 18 + (high - point.y) / span * 90;
    const active = hover && points.find(point => point.index === hover.index) || points.at(-1);
    const line = metricLineSegments(points).map(segment => segment.map((point, index) =>
      `${index ? 'L' : 'M'}${x(point)},${y(point)}`).join(' ')).join(' ');
    const xLabel = rows.every(row => isMeasured(row.cycle)) ? 'cycle' : '记录序号';
    return h(Section, { title, note: unit },
      rangeControl,
      h('figure', { 'aria-label': `${title}，${points.length}条记录${unit?'，'+unit:''}` },
        h('svg', { viewBox: '0 0 344 144', className: 'mm-plot', role: 'group', 'aria-label': title },
          h('defs',null,h('clipPath',{id:`mm-clip-${path}`},h('rect',{x:42,y:18,width:282,height:90}))),
          h('line', { x1: 42, y1: 18, x2: 42, y2: 108, className: 'mm-axis' }),
          h('line', { x1: 42, y1: 108, x2: 324, y2: 108, className: 'mm-axis' }),
          h('text', { x: 35, y: 22, textAnchor: 'end', className: 'mm-axis-label' }, formatMetric(high)),
          h('text', { x: 35, y: 110, textAnchor: 'end', className: 'mm-axis-label' }, formatMetric(low)),
          h('text', { x: 42, y: 128, className: 'mm-axis-label' }, formatMetric(minX, true)),
          minX !== maxX && h('text', { x: 324, y: 128, textAnchor: 'end', className: 'mm-axis-label' }, formatMetric(maxX, true)),
          h('text', { x: 183, y: 142, textAnchor: 'middle', className: 'mm-axis-label' }, xLabel),
          h('path', { d: line, className: 'mm-series', 'aria-hidden': true, vectorEffect: 'non-scaling-stroke',clipPath:`url(#mm-clip-${path})` }),
          h('circle', { cx: x(active), cy: y(active), r: 3, className: 'mm-point', 'aria-hidden': true,clipPath:`url(#mm-clip-${path})` }),
          points.map(point => h('g', { key: point.index,clipPath:`url(#mm-clip-${path})` },
            h('circle', { cx: x(point), cy: y(point), r: 12, className: 'mm-hit', tabIndex: 0,
              'aria-label': `${title} · cycle ${point.x} · ${point.y} ${unit}`,
              onMouseEnter: () => setHover(point), onMouseLeave: () => setHover(null),
              onFocus: () => setHover(point), onBlur: () => setHover(null) },
              h('title', null, `cycle ${point.x} · ${point.y} ${unit}`))))),
        hover&&h('figcaption', { className: 'mm-readout', role: 'status' }, `cycle ${active.x} · ${formatMetric(active.y)} ${unit}`)));
  }
  function RawTable({ snapshot, life }) {
    const [show, setShow] = React.useState(false);
    const fields=life?[
      ['周期',snapshot.source?.cycle??snapshot.card?.age?.cycles],['累计字节',snapshot.card?.age?.byte_events],
      ['经历占用',snapshot.card?.memory?.occupied],['容量',snapshot.card?.memory?.capacity],
      ['经历提交',snapshot.card?.memory?.episodes_committed],['最老留痕',snapshot.card?.memory?.oldest_since_write_events],
      ...(snapshot.card?.latest?.momentum_norms??[]).map((v,i)=>['L'+(i+1)+' 动量',v]),
    ]:null;
    const columns=[['cycle','cycle'],['updates','更新次数'],['response_nll','回应误差'],['mother_response_nll','交互回应误差'],['ambient_response_nll','环境回应误差'],['progress','学习进展'],['joint_progress','联合学习进展']];
    return h(React.Fragment,null,
      h('div',{className:'mm-actions'},h(Button,{size:'sm','aria-expanded':show,onClick:()=>setShow(v=>!v)},show?'收起数据表':'数据表')),
      show&&h('div',{className:'mm-table-wrap'},h('table',null,
        h('caption',null,life?'状态数据':'学习数据'),
        h('thead',null,h('tr',null,(life?['项目','数值']:columns.map(([,label])=>label)).map(label=>h('th',{key:label,scope:'col'},label)))),
        h('tbody',null,life?fields.map(([label,value])=>h('tr',{key:label},h('th',{scope:'row'},label),h('td',null,formatMetric(value)))):
          snapshot.rows.map((row,index)=>h('tr',{key:index},columns.map(([key])=>h('td',{key},formatMetric(row[key],key==='cycle'||key==='updates')))))))));
  }
  function PhaseDashboard({ rows, memory }) {
    const [selected, select] = React.useState('latest');
    const row = rows.find(r => String(r.cycle) === selected) ?? rows.at(-1);
    if (!row) return null;
    const p = row.phases ?? {}, t = p.topology, n = v => formatMetric(v, true);
    const stop = { handover: '自行交话', budget: '达到长度上限', silence: '沉默', reused_payload: '复用原话' };
    const action = { silence: '沉默', generate: '生成', reuse: '复用' };
    return h(Section, { title: '交互仪表盘' },
      h('label', null, '交互时间线 ', h('select', { value: rows.some(r => String(r.cycle) === selected) ? selected : 'latest', onChange: e => select(e.target.value) },
        h('option', { value: 'latest' }, '最新一轮'), rows.map(r => h('option', { key: r.cycle, value: String(r.cycle) }, `${r.cycle} · ${action[r.choice] ?? '未测'}`)))),
      h(Facts, { rows: [
        ['归档经历 / 关系 / 引用', `${n(p.episodes)} / ${n(p.relations)} / ${n(p.references)}`],
        ['经历占用 / 容量', `${n(memory?.occupied)} / ${n(memory?.capacity)}`],
        ['未决项 / 角色', `${row.shared_unknown?.identity?.replace(/^unknown:/,'') ?? '—'} / ${{condition:'条件',cause:'原因',effect:'结果',action:'行动'}[row.shared_unknown?.role] ?? row.shared_unknown?.role ?? '—'}`],
        ['候选 / 草稿', `${n(p.candidates)} / ${n(p.drafts)}`],
        ['行动 / 表达字节', `${action[row.choice] ?? '—'} / ${n(row.self_bytes)}`],
        ['交话原因', stop[p.reason] ?? '—'],
        ['摄入字节 / 外界字节', `${n(p.received)} / ${n(row.external_bytes)}`],
        ['证据', p.fact?.replace(/^episode:/,'') ?? '—'],
        ['预期 / 实际收益', `${formatMetric(p.expected)} / ${formatMetric(row.metal_g)}`],
        ['提交核对', p.verified == null ? '—' : p.verified ? '已核对' : '未通过'],
        ['外界轮 / 内部步 / 节律钟', `${n(row.cycle)} / ${n(row.internal_ticks)} / ${n(p.clock)}`],
        ['间隙安静步', n(p.quiet)],
      ] }),
      t && h(Section,{title:'水库结构'},h(Facts, { rows: [['未知项', n(t.unknowns)], ['最大连通群 / 关系', `${n(t.giant)} / ${n(t.relations)}`],
        ['β₁', n(t.beta1)], ['边 / 面', `${n(t.edges)} / ${n(t.faces)}`]] })));
  }
  function MetricsPanel({ kind }) {
    const snapshot = useMetrics();
    const life = kind === LIFE_KIND;
    // Opening through the native guide also consumes the one-time default.
    React.useEffect(() => { if (life) lifeOpened = true; }, [life]);
    const card = snapshot.card;
    const row = snapshot.rows.at(-1) ?? card?.latest ?? {};
    const memory = card?.memory;
    const omega = (card?.dynamics?.frequency ?? row.adaptive_dynamics?.frequency ?? []);
    const frequencies = (Array.isArray(omega) ? omega.flat() : [omega]).filter(isMeasured);
    const pEffect = card?.latest?.momentum_wood?.choice_probability_tv_without_p ?? row.momentum_wood?.choice_probability_tv_without_p;
    const responsePath = snapshot.rows.some(value => isMeasured(value.response_nll)) ? 'response_nll' : 'mother_response_nll';
    const progressPath = snapshot.rows.some(value => isMeasured(value.joint_progress)) ? 'joint_progress' : 'progress';
    const observation = snapshot.source?.observation;
    return h('section', { className: 'mengli-metrics', 'data-mengli-metrics': life ? 'life' : 'learning', 'aria-label': life ? '生命体征' : '学习训练' },
      h('header', null,
        h('div', { className: 'mm-title-row' }, h('h2', null, life ? '生命体征' : '学习训练'),
          life && h(CoreAnchor, { dock: true }))),
      h('div', { className: 'mm-state' }, snapshot.loading ? '正在读取绑定存档…' : lifeStatus(snapshot)),
      observation && (!life || observation.saved_cycle !== observation.latest_completed_cycle || observation.target_cycle !== observation.saved_cycle) && h('div', { className: 'mm-run-observation' },
        h(Facts, { rows: [
          ...(!life ? [['已完成事件', formatMetric(observation.latest_completed_cycle, true)]] : []),
          ...(observation.saved_cycle !== observation.latest_completed_cycle ? [['已保存检查点', formatMetric(observation.saved_cycle, true)]] : []),
          ...(observation.target_cycle !== observation.saved_cycle ? [['本段目标', formatMetric(observation.target_cycle, true)]] : []),
        ] }),
        isMeasured(observation.start_cycle) && isMeasured(observation.target_cycle) && observation.target_cycle > observation.start_cycle && observation.latest_completed_cycle < observation.target_cycle &&
          h('progress', { 'aria-label': '当前研究训练进度', max: observation.target_cycle - observation.start_cycle,
            value: Math.max(0, (observation.latest_completed_cycle ?? observation.start_cycle) - observation.start_cycle), style: { width: '100%', accentColor: 'var(--color-text-primary)' } }),
        ),
      snapshot.error && h('p', { role: 'alert', className: 'mm-error' }, '遥测读取失败 · ' + snapshot.error),
      !snapshot.loading && !snapshot.error && (life ? h(React.Fragment, null,
        h(PhaseDashboard, { rows: snapshot.rows, memory }),
        h('details', { className: 'mm-life-details' }, h('summary', null, '状态与记忆明细'),
        h('div', { className: 'mm-heroes' },
          h(Hero, { label: '周期', value: snapshot.source?.cycle ?? card?.age?.cycles ?? card?.source?.cycle, unit: 'cycle', integer: true }),
          h(Hero, { label: '累计字节', value: card?.age?.byte_events, unit: 'byte', integer: true })),
        h(Facts, { rows: [
          ['经历槽位', `${formatMetric(memory?.occupied, true)} / ${formatMetric(memory?.capacity, true)}`],
          ['经历提交', formatMetric(memory?.episodes_committed, true)],
          ['最老留痕', `${formatMetric(memory?.oldest_since_write_events, true)} byte`],
          ...(frequencies.length ? [['内部角频率范围', `${Math.min(...frequencies).toFixed(6)} — ${Math.max(...frequencies).toFixed(6)}`]] : []),
          ...(isMeasured(pEffect) ? [['P读出对当前选择的影响', `TV ${(pEffect * 100).toFixed(3)}%`]] : []),
        ] }),
        h(NormBars, { norms: card?.latest?.momentum_norms ?? row.momentum_norms, title: '分层动量' }),
        !card && h('p', { className: 'mm-empty' }, '绑定来源尚未提供生命卡。'),
        h(RawTable, { snapshot, life }))) : h(React.Fragment, null,
        h('div', { className: 'mm-heroes' },
          h(Hero, { label: '本轮更新', value: row.updates, unit: '次', integer: true }),
          h(Hero, { label: '回应误差', value: row[responsePath], unit: 'NLL' })),
        snapshot.rows.length ? h(React.Fragment, null,
          h(LinePlot, { rows: snapshot.rows, path: responsePath, title: '回应预测误差', unit: 'NLL' }),
          h(LinePlot, { rows: snapshot.rows, path: progressPath, title: progressPath === 'joint_progress' ? '联合学习进展' : '学习进展', unit: '' }),
          row.causal_program && h(Section, { title: '已记录关系' }, h(Facts, { rows: [
            ['活跃关系', formatMetric(row.causal_program.active_relations, true)],
            ['证据数', formatMetric(row.causal_program.evidence_count, true)],
            ...(isMeasured(row.causal_program.mother_prequential?.mse) ? [['后果预测误差 · MSE', formatMetric(row.causal_program.mother_prequential.mse)]] : []),
          ] })),
          row.learned_head_coupling?.per_layer_gradient_norm && h('details', null,
            h('summary', null, '梯度诊断'), h(NormBars, { title: '分层梯度范数', note: '非生命状态', norms: row.learned_head_coupling.per_layer_gradient_norm }))) :
          h('p', { className: 'mm-empty' }, '当前绑定来源尚无训练记录。'))),
      !life && h(RawTable, { snapshot, life }));
  }
}
