/* MengLi visual entrance. No model/session operations.
 * Wait/release/skip lifecycle adapted from dsh-startup-screen v1.3.0,
 * MIT Copyright (c) 2026 沈宏宏. See THIRD_PARTY_NOTICES.md.
 */
(function () {
  'use strict';

  const SEEN_KEY = 'mengli:intro:seen:v1';
  let active = null;
  let replayButton;
  let lifeCard = null;
  let headerPulseTimer = null;
  const known = value => typeof value === 'number' && Number.isFinite(value) && value >= 0;
  const count = value => known(value) ? value.toLocaleString('en-US') : '未测';

  function validLife(card) {
    return card?.schema === 1 && typeof card.identity?.name === 'string' &&
      typeof card.identity?.individual_id === 'string' && typeof card.branch_id === 'string' &&
      known(card.source?.cycle) && card.age?.cycles === card.source.cycle ? card : null;
  }

  function updateLife(value) {
    const next = validLife(value);
    const sameBranch = lifeCard && next && lifeCard.identity.individual_id === next.identity.individual_id &&
      lifeCard.branch_id === next.branch_id;
    const pulse = Boolean(sameBranch && next.source.cycle > lifeCard.source.cycle &&
      known(next.latest?.updates) && next.latest.updates > 0);
    lifeCard = next;
    const name = document.getElementById('life-name');
    const reading = document.getElementById('life-reading');
    if (name) name.textContent = next?.identity.name ?? '梦漓';
    if (reading) {
      reading.textContent = next ? `存档第 ${count(next.age.cycles)} 轮 · ${count(next.memory?.occupied)} 处留痕` : '妈妈世界 · 连续工作台';
      reading.title = next ? `${next.branch_id}\n${next.source.checkpoint}\nSHA256 ${next.source.sha256}` : '';
    }
    const brand = document.querySelector('.mengli-brand-mark');
    if (!sameBranch || pulse) {
      if (headerPulseTimer !== null) window.clearTimeout(headerPulseTimer);
      headerPulseTimer = null;
      brand?.classList.remove('has-life-update');
    }
    if (pulse && brand && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      brand.classList.add('has-life-update');
      headerPulseTimer = window.setTimeout(() => {
        brand.classList.remove('has-life-update'); headerPulseTimer = null;
      }, 1100);
    }
    active?.showLife(next, pulse);
  }

  function element(tag, className, content) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (content !== undefined) node.textContent = content;
    return node;
  }

  function wasSeen() {
    try { return window.sessionStorage.getItem(SEEN_KEY) === 'yes'; }
    catch { return false; }
  }

  function markSeen() {
    try { window.sessionStorage.setItem(SEEN_KEY, 'yes'); }
    catch { /* Storage is optional; never block the workbench. */ }
  }

  function mount() {
    const root = element('section', 'mengli-intro');
    root.id = 'mengli-intro';
    root.dataset.phase = 'seed';
    root.setAttribute('role', 'dialog');
    root.setAttribute('aria-modal', 'true');
    root.setAttribute('aria-labelledby', 'mengli-intro-title');
    root.tabIndex = -1;

    for (const name of ['intro-grain', 'intro-ambient']) {
      const decoration = element('div', name);
      decoration.setAttribute('aria-hidden', 'true');
      root.appendChild(decoration);
    }
    const top = element('div', 'intro-topline');
    top.appendChild(element('span', 'brand-en', 'MENG LI'));
    top.appendChild(element('span', 'subline', 'CONTINUITY, NOT RESTART'));
    root.appendChild(top);

    const field = element('div', 'intro-field');
    field.setAttribute('aria-hidden', 'true');
    const orbits = ['a', 'b', 'c', 'd'].map(letter => element('div', `intro-orbit orbit-${letter}`));
    for (const orbit of orbits) field.appendChild(orbit);
    field.appendChild(element('div', 'intro-seed'));
    const mark = element('img', 'intro-mark');
    mark.src = '/mengli-mark.svg';
    mark.alt = '';
    mark.width = 160;
    mark.height = 160;
    field.appendChild(mark);
    for (let i = 0; i < 16; i++) {
      const speck = element('i', 'intro-speck');
      speck.style.setProperty('--angle', `${(i * 137.508) % 360}deg`);
      speck.style.setProperty('--distance', `${92 + (i % 5) * 23}px`);
      speck.style.setProperty('--delay', `${(i % 4) * 0.12}s`);
      field.appendChild(speck);
    }
    root.appendChild(field);

    const copy = element('div', 'intro-copy');
    const title = element('h1', '', '梦漓');
    title.id = 'mengli-intro-title';
    copy.appendChild(title);
    copy.appendChild(element('div', 'intro-latin', 'MENG LI'));
    const life = element('div', 'intro-life');
    life.hidden = true;
    const age = element('p', 'intro-life-age');
    const memory = element('p', 'intro-life-memory');
    const oldest = element('p', 'intro-life-oldest');
    const layers = element('p', 'intro-life-layers');
    const source = element('p', 'intro-life-source');
    for (const line of [age, memory, oldest, layers, source]) life.appendChild(line);
    copy.appendChild(life);
    copy.appendChild(element('p', 'intro-verse', '于流中，成其形。'));
    root.appendChild(copy);
    root.appendChild(element('div', 'intro-edge-label', 'ONE CONTINUOUS ENCOUNTER'));
    const bottom = element('div', 'intro-bottom');
    bottom.appendChild(element('span', 'intro-continuity', '接续，而非重启。'));
    const skip = element('button', 'intro-skip', '跳过开场');
    skip.type = 'button';
    skip.appendChild(element('span', '', 'Esc'));
    bottom.appendChild(skip);
    root.appendChild(bottom);
    document.body.appendChild(root);
    return { root, skip, title, orbits, life, age, memory, oldest, layers, source };
  }

  function start(force) {
    const styles = document.getElementById('mengli-intro-styles');
    if (!styles?.sheet || (!force && wasSeen())) return;
    active?.dispose('replaced');
    const view = mount();
    const { root, skip } = view;
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const previousFocus = document.activeElement;
    const background = [document.querySelector('header'), document.querySelector('.workspace'),
      document.querySelector('.family-tabs'), document.querySelector('.family-notice')]
      .filter(Boolean).map(node => ({ node, inert: node.inert }));
    for (const entry of background) entry.node.inert = true;
    if (reduced) root.classList.add('is-reduced');

    const timers = new Set();
    const waiters = new Set();
    let ending = false;
    let disposed = false;

    function after(delay, callback) {
      const timer = window.setTimeout(() => { timers.delete(timer); callback(); }, delay);
      timers.add(timer);
      return timer;
    }
    function wait(delay) {
      return new Promise(resolve => {
        if (ending) return resolve();
        const done = () => { waiters.delete(done); resolve(); };
        waiters.add(done);
        after(delay, done);
      });
    }
    function releaseAll() {
      for (const timer of timers) window.clearTimeout(timer);
      timers.clear();
      for (const done of [...waiters]) done();
      waiters.clear();
    }
    function dispose(reason) {
      if (disposed) return;
      disposed = ending = true;
      releaseAll();
      document.removeEventListener('keydown', onKey, true);
      skip.removeEventListener('click', onSkip);
      root.remove();
      for (const entry of background) entry.node.inert = entry.inert;
      if (active === instance) active = null;
      if (reason !== 'replaced') {
        markSeen();
        const target = previousFocus?.isConnected && previousFocus !== document.body
          ? previousFocus : replayButton;
        target?.focus({ preventScroll: true });
        window.dispatchEvent(new CustomEvent('mengli-intro:done', { detail: { reason } }));
      }
    }
    function leave(reason) {
      if (ending) return;
      ending = true;
      releaseAll();
      root.classList.add('is-leaving');
      after(reduced ? 160 : 650, () => dispose(reason));
    }
    function onKey(event) {
      if (event.key === 'Escape') { event.preventDefault(); leave('skipped'); }
    }
    function onSkip() { leave('skipped'); }
    function showLife(card, pulse = false) {
      root.dataset.life = card ? 'bound' : 'unbound';
      root.dataset.activity = 'saved';
      view.life.hidden = !card;
      view.title.textContent = card?.identity.name ?? '梦漓';
      if (!card) { root.dataset.momentum = 'unmeasured'; return; }
      root.dataset.lifeSource = card.source.sha256;
      view.age.textContent = `已历 ${count(card.age.cycles)} 轮 · ${count(card.age.byte_events)} 字节事件（含自听）`;
      view.age.title = '继承的事件轮次；字节事件时钟包含自听，不是秒龄';
      view.memory.textContent = `${count(card.memory?.occupied)} 处留痕 · ${count(card.memory?.episodes_committed)} 次经历提交`;
      view.memory.title = `当前容量 ${count(card.memory?.capacity)} 槽；经历提交包括合并，不是现存条数`;
      view.oldest.textContent = `最老留痕 · ${count(card.memory?.oldest_since_write_events)} 字节事件`;
      view.oldest.title = '占用记录距末次写入的最大事件间隔，不是自然时间';
      const norms = card.latest?.momentum_norms;
      const measured = Array.isArray(norms) && norms.length === 4 && norms.every(known);
      root.dataset.momentum = measured ? 'measured' : 'unmeasured';
      view.layers.textContent = measured ? `四层 P · ${norms.map(value => value.toFixed(2)).join(' / ')}` : '四层 P · 未测';
      view.layers.title = '层1→4的模型坐标范数；膜仅按本快照各层相对值作艺术映射，没有心跳或健康刻度';
      if (measured) {
        const maximum = Math.max(...norms);
        norms.forEach((value, index) => {
          const ratio = maximum > 0 ? value / maximum : 0;
          view.orbits[index].style.setProperty('--layer-drive', String(ratio));
          view.orbits[index].style.setProperty('--life-scale', String(.85 + .2 * ratio));
        });
      }
      view.source.textContent = `已保存 · 第 ${count(card.source.cycle)} 轮 · 本轮 ${count(card.latest?.updates)} 次参数更新`;
      view.source.title = `${card.branch_id}\n${card.source.checkpoint}\nSHA256 ${card.source.sha256}`;
      if (pulse && !ending && !reduced) {
        root.classList.add('is-learning');
        after(1100, () => root.classList.remove('is-learning'));
      }
    }
    const instance = { dispose, leave, showLife };
    active = instance;
    showLife(lifeCard);
    document.addEventListener('keydown', onKey, true);
    skip.addEventListener('click', onSkip);
    root.focus({ preventScroll: true });
    // Independent of the data connection; even a failed API cannot hold the curtain.
    after(7600, () => dispose('timeout'));

    (async () => {
      if (reduced) {
        root.dataset.phase = 'name';
        await wait(600);
      } else {
        await wait(900);
        if (ending) return;
        root.dataset.phase = 'unfurl';
        await wait(1900);
        if (ending) return;
        root.dataset.phase = 'name';
        await wait(1900);
        if (ending) return;
        root.dataset.phase = 'handoff';
        await wait(1500);
      }
      if (!ending) leave('complete');
    })();
  }

  function initialize() {
    replayButton = document.getElementById('intro-replay');
    if (replayButton && document.getElementById('mengli-intro-styles')?.sheet) {
      replayButton.hidden = false;
      replayButton.addEventListener('click', () => start(true));
    }
    window.MengliIntro = Object.freeze({ replay: () => start(true), skip: () => active?.leave('skipped'), updateLife });
    updateLife(window.__mengliLife ?? null);
    start(false);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize, { once: true });
  } else initialize();
})();
