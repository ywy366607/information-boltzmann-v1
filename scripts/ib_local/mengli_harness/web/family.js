(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  let family = null;
  let planVersion = null;
  let streamCursor = null;
  let frameURL = null;
  const api = window.MengliFamily = { ownsStream: false };
  const modes = { mother: '妈妈照护', parent: '家长接管', paused: '照护已暂停', pausing: '正在保存当前轮 · 稍后交接' };

  function notice(text, error = false) {
    $('family-notice').textContent = text;
    $('family-notice').dataset.error = String(error);
    $('family-notice').hidden = !text;
  }
  async function request(route, payload) {
    const response = await fetch('/api/family' + route, payload === undefined ? { cache: 'no-store' } : {
      method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(payload),
    });
    const value = await response.json();
    if (!response.ok) throw new Error(value.error?.message || `HTTP ${response.status}`);
    return value;
  }
  function showView(view) {
    document.body.dataset.familyView = view;
    document.querySelectorAll('.family-tabs [data-view]').forEach(button => button.setAttribute('aria-selected', String(button.dataset.view === view)));
    $('mother-panel').hidden = view !== 'mother';
    $('family-plan-panel').hidden = view !== 'plan';
    if (view === 'mother' && family?.workbench?.url && frameURL !== family.workbench.url) {
      frameURL = family.workbench.url;
      $('mother-frame').src = frameURL;
    }
  }
  document.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => showView(button.dataset.view)));

  function bubble(stream, actor, text, metadata, note) {
    const turn = document.createElement('div');
    turn.className = `dialogue-turn turn-${actor === 'mengli' ? 'child' : actor}`;
    const box = document.createElement('div');
    box.className = `bubble bubble-${actor === 'mengli' ? 'child' : actor}`;
    const meta = document.createElement('div');
    meta.className = 'bubble-meta';
    const name = document.createElement('span');
    name.textContent = { mengli: '梦漓 (MengLi)', mother: '妈妈 (DS)', parent: '家长 · 你' }[actor];
    const label = document.createElement('span');
    label.textContent = metadata;
    meta.append(name, label);
    const body = document.createElement('div');
    body.textContent = text;
    box.append(meta, body);
    if (note) {
      const detail = document.createElement('div');
      detail.className = 'delivery-note';
      detail.textContent = note;
      box.append(detail);
    }
    turn.append(box);
    stream.append(turn);
  }
  function renderCare(value) {
    const events = value.events || [];
    if (!events.length || streamCursor === value.cursor) return;
    api.ownsStream = true;
    streamCursor = value.cursor;
    const stream = $('dialogue-stream');
    const atBottom = stream.scrollHeight - stream.scrollTop - stream.clientHeight < 40;
    stream.replaceChildren();
    events.forEach(event => {
      bubble(stream, 'mengli', event.utterance?.text || '（沉默）', `C${event.cycle} · ${event.choice ?? '行动'}`,
        '本轮输入送达前的自主行动，不是对下方话语的回答。');
      const delivered = event.delivered || {};
      bubble(stream, event.caregiver || event.actor, delivered.text ?? '', `已提交 · 计划 v${event.plan_version}`,
        `实际摄入 ${delivered.byte_end ?? '?'} / ${delivered.candidate_bytes ?? '?'} 字节` +
        (delivered.truncated ? ' · 超出容量的部分未送达' : '') + (delivered.utf8_complete === false ? ' · 末尾含不完整 UTF-8 字节，原字节已保留' : ''));
    });
    $('stream-turn-count').textContent = `本次续接已提交 ${value.cursor} 轮 · 显示最近 ${events.length} 轮`;
    if (atBottom) stream.scrollTop = stream.scrollHeight;
  }
  function render(value) {
    family = value;
    $('care-mode').textContent = (modes[value.mode] || value.mode) + (value.loaded ? ' · 身体已连接' : ' · 身体未加载');
    $('family-binding').textContent = `${value.individual_id || '梦漓'} · 已历 ${value.cycle ?? '未绑定'} 轮`;
    $('family-checkpoint').textContent = `分支 ${value.branch_id || '未绑定'}\n${value.checkpoint || '没有完整存档'}`;
    $('body-load').disabled = value.loaded || value.busy;
    $('body-load').textContent = value.loaded ? '续接体已连接' : value.busy ? '正在连接 / 完整保存…' : '连接续接体 · 不推进轮次';
    $('care-step').disabled = !value.loaded || value.busy || value.mode !== 'mother';
    $('care-resume').disabled = value.busy || value.mode === 'mother';
    $('care-takeover').disabled = value.mode === 'parent' || value.mode === 'pausing';
    const canSpeak = value.loaded && !value.busy && value.mode === 'parent' && !value.pending_input;
    $('parent-text').disabled = !canSpeak;
    $('parent-send').disabled = !canSpeak;
    $('parent-text').placeholder = canSpeak ? '以家长身份对梦漓说话…' : '先连接续接体，再暂停妈妈并接管';
    const capacity = value.response_bytes;
    $('input-capacity').textContent = capacity ? `每轮实际输入容量 ${capacity} UTF-8字节（含换行）` : '实际送达受身体字节容量限制';
    $('delivery-status').textContent = value.pending_input
      ? `${value.pending_input.status === 'queued' ? '待送达 · 正在完成整轮' : '未确认交付 · 请核查存档'}：${value.pending_input.text}`
      : '整轮保存后才标为已提交；后继自主行动才可能回应这次输入，也可以沉默。';
    if (planVersion !== value.plan.version && document.activeElement !== $('care-plan')) {
      $('care-plan').value = value.plan.text;
      planVersion = value.plan.version;
    }
    $('plan-version').textContent = `已保存 v${value.plan.version} · 身体已采用 v${value.plan.effective_version}`;
    if (value.workbench?.url) {
      $('mother-open').href = value.workbench.url;
      $('mother-open').hidden = false;
      $('mother-loading').hidden = true;
      if (document.body.dataset.familyView === 'mother') showView('mother');
    } else if (value.workbench?.error) $('mother-loading').textContent = `官方工作区未启动：${value.workbench.error}`;
    if (value.error) notice(value.error, true);
    renderCare(value);
  }
  async function action(route, payload, message) {
    notice(message || '正在处理…');
    try {
      const result = await request(route, payload);
      render(result);
      notice('');
      return result;
    } catch (error) { notice(error.message, true); return null; }
  }
  $('body-load').addEventListener('click', () => action('/load', {}, '正在恢复完整续接态；不推进轮次…'));
  $('care-takeover').addEventListener('click', () => action('/control', { mode: 'parent' }, '暂停自动照护，等待当前轮完整保存…'));
  $('care-resume').addEventListener('click', () => action('/control', { mode: 'mother' }, '恢复妈妈照护与已暂停目标…'));
  $('care-step').addEventListener('click', () => action('/step', { cycles: 1 }, '妈妈照护一轮：真实回应、学习并完整保存…'));
  $('parent-form').addEventListener('submit', async event => {
    event.preventDefault();
    const text = $('parent-text').value;
    const result = await action('/say', { text }, '家长输入待送达；正在完成真实交付与保存…');
    if (result) $('parent-text').value = '';
  });
  $('plan-form').addEventListener('submit', event => {
    event.preventDefault();
    action('/plan', { text: $('care-plan').value, version: planVersion }, '保存新计划；下一完整轮生效…');
  });
  request('').then(render).catch(error => notice(`家庭绑定读取失败：${error.message}`, true));
  request('/history').then(history => {
    const turns = history.turns || [];
    $('care-archive').hidden = !turns.length;
    turns.forEach(turn => bubble($('care-history'), turn.role === 'user' ? 'mengli' : 'mother', turn.content, '已保存历史'));
  }).catch(() => { $('care-archive').hidden = true; });
  const events = new EventSource('/events');
  events.addEventListener('family', event => render(JSON.parse(event.data)));
  api.render = render;
  api.showView = showView;
  api.updateLife = card => {
    const norms = card?.latest?.momentum_norms;
    if (!Array.isArray(norms) || norms.length !== 4 || !norms.every(value => Number.isFinite(value) && value >= 0)) return;
    const maximum = Math.max(...norms);
    document.querySelectorAll('.family-background i').forEach((orbit, index) => {
      const relative = maximum > 0 ? norms[index] / maximum : 0;
      orbit.style.opacity = String(.25 + relative * .65);
      orbit.title = `Layer ${index + 1} P = ${norms[index]}`;
    });
  };
  api.updateLife(window.__mengliLife);
})();
