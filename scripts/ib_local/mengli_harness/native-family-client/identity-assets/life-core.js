function installMengliIntro() {
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  let card = null, target = coreTarget(null), current = { ...target };
  let phase = 0, frame = 0, lastTime = 0, pulseUntil = 0, motionPaused = false;
  let visualSeconds = 0, focusStep = 0, focus = null, beatAt = -10, nextScanAt = 0;
  let lastDraw = 0, renderSkipped = false;
  let active = null, entered = false, disposed = false, placement = '', observedAnchor = null;
  const host = document.createElement('div');
  host.id = 'mengli-life-core'; host.hidden = true; host.style.transition = 'none';
  host.setAttribute('role', 'figure'); host.setAttribute('aria-label', '梦漓的分层活动与保存态读数');
  const renderer = createEntranceSky(host);
  const companion = document.createElement('aside'); companion.id = 'mengli-companion';
  companion.setAttribute('aria-label', '梦漓');
  const characterCanvas = document.createElement('canvas'); characterCanvas.className = 'ml-character-matte';
  characterCanvas.setAttribute('aria-hidden', 'true'); companion.appendChild(characterCanvas);
  const characterRenderer = createCharacterPortrait(characterCanvas, { motion: true });
  document.body.appendChild(companion);
  function wakeCharacter() {
    characterRenderer.setPaused(disposed || document.hidden || reduced.matches || motionPaused);
  }
  const compact = document.createElement('img'); compact.className = 'ml-nebula-compact';
  compact.src = MENGLI_NEBULA; compact.alt = ''; host.appendChild(compact);
  const ns = 'http://www.w3.org/2000/svg';
  const guides = document.createElementNS(ns, 'svg'); guides.classList.add('ml-nebula-guides');
  guides.setAttribute('viewBox', '0 0 680 480'); guides.setAttribute('aria-hidden', 'true');
  const leaderLines = [];
  for (const {path:d} of telemetryLayout(0)) {
    const line = document.createElementNS(ns, 'path'); line.setAttribute('d', d); guides.appendChild(line);
    leaderLines.push(line);
  }
  host.appendChild(guides);
  const hud = document.createElement('dl'); hud.className = 'ml-nebula-hud'; hud.hidden = true;
  const readouts = {};
  for (const [key, label] of [['cycle', 'cycle'], ['depth', 'layer / head'],
    ['momentum', 'P'], ['frequency', 'ω']]) {
    const group = document.createElement('div'); group.className = `ml-readout ml-readout-${key}`;
    const name = document.createElement('dt'); name.textContent = label;
    const value = document.createElement('dd'); value.textContent = '—';
    group.append(name, value); hud.appendChild(group); readouts[key] = value;
  }
  host.appendChild(hud);
  const byteReadout=document.createElement('small'); byteReadout.className='ml-readout-bytes';
  byteReadout.hidden=true; readouts.cycle.after(byteReadout);
  function placeReadouts() {
    const heads = target.surfaces.reduce((count, surface) => Math.max(count, surface.headFrequencies.length), 0);
    const port = telemetryPort(focus, target.depth, heads);
    telemetryLayout(focusStep, port, true).forEach(({key,x,y,side,path},index)=>{
      const group=readouts[key].parentElement;
      group.style.left='0';group.style.top='0';group.style.right='auto';
      group.style.width='132px';group.style.textAlign=side>0?'left':'right';
      group.style.transform=`translate(${x}px,${y}px)`;
      leaderLines[index].setAttribute('d',path);
    });
    host.dataset.layout=String(focusStep);
  }
  placeReadouts();
  const strip = document.createElement('div'); strip.className = 'ml-nebula-strip';
  strip.setAttribute('aria-hidden', 'true'); host.appendChild(strip);
  const paths = [], layerPhases = [], headPhases = [];
  document.body.appendChild(host);
  function syncSurfaces() {
    const count = target.surfaces.length;
    while (paths.length > count) { paths.pop().remove(); layerPhases.pop(); headPhases.pop(); }
    while (paths.length < count) {
      const layer = document.createElement('i'); layer.className = 'ml-nebula-layer';
      strip.appendChild(layer); paths.push(layer); layerPhases.push(0); headPhases.push([]);
    }
    paths.forEach((layer, index) => {
      const surface = target.surfaces[index];
      layer.dataset.layer = String(surface.layer); layer.dataset.measured = String(surface.measured);
      layer.dataset.momentum = surface.mean === null ? 'missing' : String(surface.mean);
      layer.dataset.frequency = surface.frequency === null ? 'missing' : String(surface.frequency);
      layer.dataset.headFrequencies = JSON.stringify(surface.headFrequencies);
      layer.style.setProperty('--layer-colour', surface.colour);
      layer.style.setProperty('--layer-height', `${2 + surface.amplitude * 13}px`);
    });
    renderer.sync(target.surfaces); hud.hidden = !target.depth;
    const fmt = (value, digits = 2) => Number.isFinite(value) ? value.toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: digits }) : '—';
    readouts.cycle.textContent = Number.isFinite(card?.source?.cycle) ? card.source.cycle.toLocaleString('en-US') : '—';
    const bytes=card?.age?.byte_events;
    byteReadout.hidden=!Number.isSafeInteger(bytes)||bytes<0;
    byteReadout.textContent=byteReadout.hidden?'':`${bytes.toLocaleString('en-US')} byte`;
    focus = telemetryFocus(target.surfaces, focusStep);
    characterRenderer.setDynamics(focus);
    readouts.depth.textContent = focus ? `${String(focus.layer).padStart(2,'0')} / ${String(focus.head).padStart(2,'0')}` : '—';
    readouts.momentum.textContent = focus ? fmt(focus.momentum,3) : '—';
    readouts.frequency.textContent = focus ? fmt(focus.frequency,3) : '—';
    if (focus) { host.dataset.focusLayer=String(focus.layer);host.dataset.focusHead=String(focus.head); }
  }
  function visible(node) {
    if (!node) return false;
    const rect = node.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0 && rect.bottom > 0 && rect.top < innerHeight && rect.right > 0 && rect.left < innerWidth;
  }
  function stopFrame() { cancelAnimationFrame(frame); frame = 0; lastTime = 0; }
  function locate() {
    if (disposed) return;
    wakeCharacter();
    const dock = [...document.querySelectorAll('[data-mengli-core-dock]')].find(visible);
    const park = [...document.querySelectorAll('[data-mengli-core-park]')].find(visible);
    const anchor = active ? active.anchor : dock ?? park;
    if (anchor !== observedAnchor) {
      if (observedAnchor) resize.unobserve(observedAnchor);
      if (anchor) resize.observe(anchor);
      observedAnchor = anchor;
    }
    host.hidden = !anchor;
    if (!anchor) { placement = ''; stopFrame(); return; }
    const rect = anchor.getBoundingClientRect();
    const scale = Math.min(rect.width / 680, rect.height / 480);
    const next = `${rect.x + rect.width / 2},${rect.y + rect.height / 2},${scale}`;
    if (next !== placement) {
      if (!placement) requestAnimationFrame(() => { host.style.transition = ''; });
      placement = next;
      host.style.transform = `translate(${rect.x + rect.width / 2 - 340}px,${rect.y + rect.height / 2 - 240}px) scale(${scale})`;
    }
    host.dataset.position = active ? 'entrance' : dock ? 'panel' : 'park';
    host.setAttribute('aria-hidden', String(!active && !dock));
    if (host.dataset.position === 'park') stopFrame(); else wake();
  }
  function draw() {
    const remaining = Math.max(0, pulseUntil - performance.now());
    const learning = remaining > 0;
    const age = visualSeconds - beatAt;
    const scanBeat = !reduced.matches && age >= 0 && age < .16 ? Math.sin(Math.PI * age / .16) ** 2 : 0;
    const beat = renderer.draw(current.surfaces, headPhases, Math.sin(Math.PI * remaining / 1100) ** 2, visualSeconds,
      remaining > 0 ? 1 - remaining / 1100 : -1, scanBeat, focus);
    const scanning = String(scanBeat > .1);
    if (host.dataset.scanBeat !== scanning) host.dataset.scanBeat = scanning;
    paths.forEach((layer, index) => { layer.dataset.phase = layerPhases[index].toFixed(4); });
    host.dataset.learning = String(learning); host.dataset.phase = phase.toFixed(4);
    host.dataset.mean = target.mean === null ? 'missing' : String(target.mean);
    host.dataset.layers = String(target.count); host.dataset.depth = String(target.depth);
    host.dataset.activity = String(beat ?? 0);
  }
  function tick(time) {
    frame = 0;
    if (disposed || document.hidden || host.hidden || renderSkipped || host.dataset.position === 'park' || reduced.matches || motionPaused) return;
    if (lastDraw && time - lastDraw < 1000 / 60 - 1) { frame = requestAnimationFrame(tick); return; }
    lastDraw = time;
    const dt = lastTime ? Math.min((time - lastTime) / 1000, .1) : 0; lastTime = time;
    current = easeCore(current, target, dt); phase += dt * Math.PI * 2 * (.55 + .4 * current.speed);
    visualSeconds += dt;
    if (visualSeconds >= nextScanAt) {
      if (nextScanAt > 0) focusStep++;
      syncSurfaces(); placeReadouts(); beatAt=visualSeconds; nextScanAt=visualSeconds+(focus?.period??4);
    }
    current.surfaces.forEach((surface, index) => {
      layerPhases[index] += dt * surface.speed;
      headPhases[index] = surface.headFrequencies.map((frequency, head) =>
        (headPhases[index][head] ?? 0) + dt * .45 * Math.log1p(frequency));
    });
    draw(); frame = requestAnimationFrame(tick);
  }
  function wake() {
    wakeCharacter();
    if (disposed || document.hidden || host.hidden || renderSkipped || host.dataset.position === 'park') return;
    if (reduced.matches || motionPaused) { current = { ...target }; draw(); return; }
    if (!frame) { lastTime = 0; frame = requestAnimationFrame(tick); }
  }
  function renderVisibility(event) {
    renderSkipped = event.skipped;
    if (renderSkipped) stopFrame(); else wake();
  }
  // Direct listener: this event does not bubble in all implementations.
  host.addEventListener('contentvisibilityautostatechange', renderVisibility);
  function updateLife(next, { committed = false } = {}) {
    if (!next || next.schema !== 1) return;
    const learning = isLearningCommit(card, next, committed);
    if (card && card.branch_id === next.branch_id && card.source?.cycle > next.source?.cycle) return;
    card = next.dynamics || card?.branch_id !== next.branch_id ? next : { ...next, dynamics: card?.dynamics };
    target = coreTarget(card); syncSurfaces(); placeReadouts();
    updateBirth();
    host.title = target.depth ? `${target.depth}层 · P 动量 · ω 角频率` : '等待活动读数';
    if (learning && !reduced.matches) pulseUntil = performance.now() + 1100;

    wake();
  }
  function node(tag, className, text) {
    const element = document.createElement(tag); element.className = className;
    if (text) element.textContent = text;
    return element;
  }
  function updateBirth() {
    const birth=active?.root.querySelector('.ml-entry-birth');
    if(!birth)return;
    const date=new Date(card?.age?.born_at??NaN);
    birth.hidden=!Number.isFinite(date.getTime());
    if(birth.hidden){birth.textContent='';birth.removeAttribute('datetime');return;}
    const parts=new Intl.DateTimeFormat('en-GB',{timeZone:'Asia/Shanghai',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hourCycle:'h23'}).formatToParts(date);
    const value=Object.fromEntries(parts.map(part=>[part.type,part.value]));
    birth.textContent=value.year+value.month+value.day+value.hour+value.minute;
    birth.dateTime=card.age.born_at;
    birth.setAttribute('aria-label',`出生时间 ${value.year}年${value.month}月${value.day}日${value.hour}时${value.minute}分，北京时间`);
  }
  function enter() {
    if (!active) return;
    const view = active; active = null; entered = true;
    view.root.dataset.phase = 'handoff';
    view.background.forEach(([element, inert]) => { element.inert = inert; });
    document.body.classList.remove('mengli-intro-open');
    document.removeEventListener('keydown', view.key, true);
    window.dispatchEvent(new Event('mengli:enter'));
    requestAnimationFrame(locate);
    setTimeout(() => view.root.remove(), reduced.matches ? 150 : 1000);
    const focus = view.focus?.isConnected && view.focus !== document.body ? view.focus :
      document.querySelector('[aria-label="梦漓菜单"], [contenteditable="true"], textarea:not([disabled])');
    focus?.focus({ preventScroll: true });
  }
  function replay() {
    if (active || disposed) return;
    document.getElementById('mengli-intro')?.remove();
    entered = false;
    const root = node('section', 'mengli-intro' + (reduced.matches ? ' is-reduced' : ''));
    root.id = 'mengli-intro'; root.dataset.phase = 'name';
    root.dataset.motionPaused=String(motionPaused);
    root.setAttribute('role', 'dialog'); root.setAttribute('aria-modal', 'true');
    root.setAttribute('aria-label', '梦漓 · 实验个体');
    const anchor = node('div', 'ml-intro-core-anchor');
    const copy = node('div', 'ml-intro-copy');
    const heading = node('div', 'ml-entry-heading');
    const birth = node('time', 'ml-entry-birth'); birth.hidden=true;
    const name=node('h1','ml-entry-name','梦'); name.setAttribute('aria-label','梦漓');
    const li=node('img','ml-name-li'); li.src=MENGLI_LI_GLYPH;li.alt='';li.setAttribute('aria-hidden','true');name.appendChild(li);
    heading.append(name, birth); copy.appendChild(heading);
    const form = node('form', 'ml-entry-form'); form.method = 'post'; form.action = '/auth/login';
    const label = node('label', 'ml-key-label'); label.htmlFor = 'key';
    const keyTag=node('span','ml-key-tag','本地密钥'); label.append(keyTag);
    const field = node('input', ''); field.id = 'key'; field.name = 'key'; field.type = 'password';
    field.autocomplete = 'current-password'; field.required = true; field.maxLength = 512;
    field.setAttribute('enterkeyhint', 'go'); field.setAttribute('aria-describedby', 'error');
    const inputLine=node('div','ml-key-input');
    const caret=node('span','ml-key-caret'); caret.setAttribute('aria-hidden','true');
    const prefix=node('span','ml-key-prefix'),cursor=node('span','ml-key-cursor','−');
    caret.append(prefix,cursor); inputLine.append(field,caret);
    const syncCaret=()=>{
      prefix.textContent='•'.repeat(field.selectionStart??field.value.length);
      caret.style.transform=`translateX(${-field.scrollLeft}px)`;
    };
    for(const event of ['input','click','keyup','select','focus','scroll'])field.addEventListener(event,syncCaret);
    const identity=node('dl','ml-entry-identity');
    const identityRow=node('div','');
    identityRow.append(node('dt','','身份名称'),node('dd','','无月')); identity.appendChild(identityRow);
    const permission=node('dl','ml-entry-identity ml-entry-permission');
    const permissionRow=node('div','');
    permissionRow.append(node('dt','','权限'),node('dd','','∞')); permission.appendChild(permissionRow);
    const button = node('button', 'ml-entry-submit'); button.type = 'submit';
    const water = node('span', 'ml-confirm-water'); water.setAttribute('aria-hidden','true');
    const cn = node('span','ml-confirm-cn','确认身份'), en = node('span','ml-confirm-en','CONFIRM IDENTITY');
    const icon = document.createElementNS(ns, 'svg'); icon.classList.add('ml-confirm-icon');
    icon.setAttribute('viewBox', '0 0 24 24'); icon.setAttribute('aria-hidden', 'true');
    const arrow = document.createElementNS(ns, 'path'); arrow.classList.add('ml-confirm-morph');
    arrow.setAttribute('d', 'M18 6C15 9 9 15 6 18C9 15 15 9 18 6C16 6 13 6 11 6C14 6 18 6 18 6C18 8 18 11 18 13');
    const check = document.createElementNS(ns, 'path'); check.classList.add('ml-confirm-check');
    check.setAttribute('d', 'M8 12l3 3 5-6'); check.setAttribute('pathLength','1');
    icon.append(arrow,check); button.append(water,cn,en,icon);
    const error = node('p', 'ml-entry-error'); error.id = 'error'; error.setAttribute('role', 'alert');
    error.setAttribute('aria-live', 'polite');
    form.append(identity, label, inputLine, permission, button, error); copy.appendChild(form);
    field.addEventListener('input', () => { error.textContent = ''; field.removeAttribute('aria-invalid'); });
    form.addEventListener('submit', async event => {
      event.preventDefault();
      if (button.disabled) return;
      button.disabled = true; error.textContent = '';
      button.dataset.state = 'checking';
      let verified = false;
      try {
        const response = await fetch('/auth/login', { method: 'POST', credentials: 'same-origin',
          headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ key: field.value }) });
        field.value = ''; syncCaret();
        if (response.ok) {
          verified = true;
          button.style.width = `${button.offsetWidth}px`;
          button.dataset.state = 'filling';
          if (!reduced.matches && !motionPaused) await new Promise(resolve => setTimeout(resolve, 460));
          button.dataset.state = 'verified';
          cn.textContent = '验证成功'; en.textContent = 'VERIFIED';
          if (!reduced.matches && !motionPaused) await new Promise(resolve => setTimeout(resolve, 950));
          if (MENGLI_LOGIN_PAGE) {
            try { sessionStorage.setItem('mengli:entry-complete', 'yes'); } catch {}
            location.replace('/');
          } else enter();
          return;
        }
        error.textContent = response.status === 401 ? '密钥不正确' : response.status === 413 ? '密钥过长' : '暂时无法进入';
        field.setAttribute('aria-invalid', 'true');
      } catch { error.textContent = '连接失败，请稍后再试'; }
      finally { if (!verified) { button.disabled = false; delete button.dataset.state; field.focus(); } }
    });
    renderer.attach(root);
    root.append(anchor, copy);
    const background = [...document.querySelectorAll('#root')].map(element => [element, element.inert]);
    const focus = document.activeElement;
    background.forEach(([element]) => { element.inert = true; });
    document.body.classList.add('mengli-intro-open'); document.body.appendChild(root);
    const key = event => {
      if (event.key === 'Escape' && !MENGLI_LOGIN_PAGE) { event.preventDefault(); event.stopPropagation(); enter(); }
      if (event.key === 'Tab') {
        const first = field, last = button.disabled ? field : button;
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
      }
    };
    document.addEventListener('keydown', key, true);
    active = { root, anchor, background, focus, key };
    if (card) updateLife(card);
    field.focus({ preventScroll: true }); locate();
  }
  function visibility() {
    if (document.hidden) {
      cancelAnimationFrame(frame); frame = 0; lastTime = 0;
      characterRenderer.setPaused(true);
    }
    else { locate(); wake(); }
  }
  const observer = new MutationObserver(locate);
  observer.observe(document.getElementById('root') ?? document.body, { childList: true, subtree: true });
  const resize = new ResizeObserver(locate); resize.observe(document.body);
  window.addEventListener('resize', locate);
  document.addEventListener('scroll', locate, true);
  // Native docks animate their position without resizing the anchor itself.
  document.addEventListener('transitionend', locate, true);
  document.addEventListener('animationend', locate, true);
  window.addEventListener('mengli:core-anchor', locate);
  document.addEventListener('visibilitychange', visibility);
  reduced.addEventListener('change', locate);
  window.MengliIntro = { replay, skip: enter, updateLife,
    get motionPaused() { return motionPaused; },
    setMotionPaused(value) {
      motionPaused = Boolean(value); pulseUntil = 0;
      if(active)active.root.dataset.motionPaused=String(motionPaused);
      cancelAnimationFrame(frame); frame = 0; lastTime = 0;
      characterRenderer.setPaused(true);
      host.dataset.paused = String(motionPaused);
      window.dispatchEvent(new Event('mengli:motion'));
      wake();
      return motionPaused;
    },
    get entered() { return entered; },
    dispose() {
      enter(); disposed = true; clearTimeout(active?.timer); cancelAnimationFrame(frame);
      characterRenderer.dispose(); companion.remove();
      observer.disconnect(); resize.disconnect(); host.remove();
      window.removeEventListener('resize', locate); document.removeEventListener('scroll', locate, true);
      document.removeEventListener('transitionend', locate, true); document.removeEventListener('animationend', locate, true);
      window.removeEventListener('mengli:core-anchor', locate); document.removeEventListener('visibilitychange', visibility);
      reduced.removeEventListener('change', locate); window.removeEventListener('pageshow', pageShow);
    },
  };
  function pageShow(event) {
    if (!event.persisted) return;
    if (MENGLI_LOGIN_PAGE) replay(); else locate();
  }
  if (MENGLI_LOGIN_PAGE) replay();
  else {
    // The server has already authenticated this document. Never mount a second
    // password scene while the native modules and their life dock are starting.
    entered = true;
    try { sessionStorage.removeItem('mengli:entry-complete'); } catch {}
    locate();
    window.dispatchEvent(new Event('mengli:enter'));
  }
  window.addEventListener('pageshow', pageShow);
}

// Original-art joint mesh. Local knee, head, ear and arm bindings retain the
// complete source texture; eye closure is a local eyelid deformation.
function createEntranceSky(host) {
  const eye = document.createElement('div'); eye.className = 'ml-helix';
  const photo = new Image(); photo.src = MENGLI_NEBULA; photo.alt = '';
  const swells = Array.from({length: 2}, () => {
    const veil = document.createElement('div'); veil.className = 'ml-helix-swell';
    veil.appendChild(photo.cloneNode()); return veil;
  });
  eye.append(photo,...swells); host.appendChild(eye);
  const stars = document.createElement('canvas'); stars.className = 'ml-star-flight';
  stars.setAttribute('aria-hidden', 'true');
  const ctx = stars.getContext('2d');
  const rand = n => { const value = Math.sin(n * 127.1 + 19.2) * 43758.5453; return value - Math.floor(value); };
  const field = Array.from({length: 240}, (_, i) => [rand(i*5), rand(i*5+1), rand(i*5+2), rand(i*5+3), rand(i*5+4)]);
  let last = 0, breath = 0, transport = 0;
  return {
    attach(root) { if (MENGLI_LOGIN_PAGE) root.appendChild(stars); },
    sync(surfaces) { host.dataset.flowLayers = String(surfaces.length); },
    draw(_surfaces, _phases, _learning, time, _commit, beat, focus) {
      const dt = Math.max(0,time-last); last = time;
      const rate = bodyVisualDynamics(focus);
      breath += dt * rate.clockRate; transport += dt * rate.transportRate * .42;
      eye.style.setProperty('--helix-scale', 1 + .009*Math.sin(breath*.55));
      swells.forEach((veil,index) => {
        const wave = (transport + index*.5) % 1;
        veil.style.setProperty('--swell-radius', `${8 + 94*wave}%`);
        veil.style.setProperty('--swell-scale', 1.035 + .12*wave);
        veil.style.opacity = transport > 0 ? .93*Math.min(1,wave*6,(1-wave)*7) : 0;
      });
      host.dataset.transportRate = rate.transportRate.toFixed(5);
      if (!ctx) return beat;
      const w = stars.clientWidth, h = stars.clientHeight, dpr = Math.min(2, devicePixelRatio || 1);
      if (!w || !h) return beat;
      if (stars.width !== Math.round(w*dpr) || stars.height !== Math.round(h*dpr)) {
        stars.width = Math.round(w*dpr); stars.height = Math.round(h*dpr);
      }
      ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,w,h);
      const cx=w*.51, cy=h*.5;
      for (const [u,v,depth,seed,tone] of field) {
        const z=(depth+1-(time*.021)%1)%1;
        const dx=(u-.5)*w*.71, dy=(v-.5)*h*.68, inv=1/(.23+1.36*z);
        const x=cx+dx*inv, y=cy+dy*inv;
        if (x<0 || x>w || y<0 || y>h) continue;
        const twinkle=.42+.58*(.5+.5*Math.sin(time*(.45+.85*seed)+tone*37))**1.6;
        const alpha=(.12+.63*(1-z))*twinkle*Math.min(1,z*16,(1-z)*13);
        ctx.fillStyle = tone>.8 ? `rgba(171,204,255,${alpha})` : `rgba(244,242,249,${alpha})`;
        const size=(.42+seed*.65)*Math.min(1.75,inv);
        ctx.beginPath(); ctx.arc(x,y,size,0,Math.PI*2); ctx.fill();
      }
      return beat;
    },
  };
}

function createCharacterPortrait(canvas, { motion = false } = {}) {
  canvas.width=512;canvas.height=768;
  const gl=canvas.getContext('webgl',{alpha:true,premultipliedAlpha:true,antialias:true,depth:false});
  function staticPortrait(){
    const im=document.createElement('img');im.src=MENGLI_CHARACTER;im.alt='';im.className=canvas.className;im.dataset.ready='true';im.style.objectFit='cover';im.style.objectPosition='top';canvas.replaceWith(im);
  }
  if(!gl){staticPortrait();return{setPaused(){},setPose(){},setDynamics(){},dispose(){}};}
  const program=gl.createProgram();
  for(const[type,source]of[
    [gl.VERTEX_SHADER,`attribute vec2 p;attribute vec4 binding;attribute vec4 upper;attribute vec2 flexBinding;
      uniform vec2 angles,flexAngles;uniform vec4 gestures;uniform float blink,part;varying vec2 uv;
      vec2 rotateAt(vec2 q,vec2 pivot,float a){vec2 d=q-pivot;return pivot+vec2(cos(a)*d.x-sin(a)*d.y,sin(a)*d.x+cos(a)*d.y);}
      vec2 eyelid(vec2 q,vec2 centre,vec2 radius){
        vec2 d=q-centre;float a=-.48;vec2 e=vec2(cos(a)*d.x+sin(a)*d.y,-sin(a)*d.x+cos(a)*d.y);
        float w=1.-smoothstep(.72,1.5,length(e/radius));e.y*=1.-blink*w;
        return centre+vec2(cos(a)*e.x-sin(a)*e.y,sin(a)*e.x+cos(a)*e.y);
      }
      void main(){uv=p/vec2(1024.,1536.);
      float rigid=smoothstep(1080.,1160.,p.y),side=mix(binding.x,part,rigid);
      vec2 pivot=mix(binding.zw,mix(vec2(709.,960.),vec2(616.,990.),part),rigid);
      float a=mix(angles.y,angles.x,side)*binding.y;
      vec2 d=p-pivot;vec2 q=pivot+vec2(cos(a)*d.x-sin(a)*d.y,sin(a)*d.x+cos(a)*d.y);
      q=eyelid(q,vec2(328.,375.),vec2(25.,20.));q=eyelid(q,vec2(380.,345.),vec2(18.,18.));
      q=rotateAt(q,vec2(185.,360.),gestures.z*upper.z);q=rotateAt(q,vec2(307.,280.),gestures.w*upper.w);
      q=rotateAt(q,vec2(345.,490.),gestures.x*upper.x);
      q=rotateAt(q,vec2(618.,255.),flexAngles.y*flexBinding.y);
      q=rotateAt(q,vec2(540.,402.),flexAngles.x*flexBinding.x);
      q=rotateAt(q,vec2(450.,530.),gestures.y*upper.y);
      gl_Position=vec4((q.x+24.)/536.-1.,1.-(q.y+36.)/804.,0.,1.);}`],
    [gl.FRAGMENT_SHADER,`precision highp float;varying vec2 uv;uniform sampler2D portrait;uniform vec2 gaze;
      uniform float part,atlasRows;uniform vec2 legSeam[12];
      vec4 original(vec2 q){return texture2D(portrait,vec2(q.x,q.y/atlasRows));}
      void main(){vec2 pixel=uv*vec2(1024.,1536.);
        float boundary=legSeam[0].y;
        for(int i=1;i<12;i++){
          float t=clamp((pixel.y-legSeam[i-1].x)/(legSeam[i].x-legSeam[i-1].x),0.,1.);
          boundary+=(legSeam[i].y-legSeam[i-1].y)*t;
        }
        bool nearPiece=pixel.y>=1080.&&pixel.x>=515.&&pixel.x<boundary;
        bool hiddenHeel=nearPiece&&pixel.x>728.&&pixel.y>1240.&&pixel.y<1440.&&atlasRows>1.5;
        if(part>.5&&!nearPiece)discard;
        if(part<.5&&nearPiece&&!hiddenHeel)discard;
        float eyes=exp(-dot((pixel-vec2(327.,374.))/vec2(13.,15.),(pixel-vec2(327.,374.))/vec2(13.,15.)))
          +exp(-dot((pixel-vec2(380.,345.))/vec2(9.,12.),(pixel-vec2(380.,345.))/vec2(9.,12.)));
        vec4 colour=original(uv-gaze/vec2(1024.,1536.)*min(eyes,1.));
        if(part<.5&&hiddenHeel){
          colour=texture2D(portrait,vec2(uv.x,(uv.y+1.)/atlasRows));
          float matte=smoothstep(.65,.96,colour.a);
          colour=vec4(colour.rgb*matte/max(colour.a,.0001),matte);
        }
        gl_FragColor=colour;}`],
  ]){
    const shader=gl.createShader(type);gl.shaderSource(shader,source);gl.compileShader(shader);
    if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(shader));gl.attachShader(program,shader);
  }
  gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));gl.useProgram(program);
  const columns=128,rows=192,vertices=[],indices=[];
  for(let y=0;y<=rows;y++)for(let x=0;x<=columns;x++){
    const px=x*1024/columns,py=y*1536/rows,b=originalPortraitBinding(px,py);
    vertices.push(px,py,b.near,b.blend,...b.pivot,...portraitUpperBinding(px,py));
  }
  for(let y=0;y<rows;y++)for(let x=0;x<columns;x++){
    const a=y*(columns+1)+x,b=a+1,c=a+columns+1,d=c+1;indices.push(a,b,c,b,d,c);
  }
  const vertexBuffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,vertexBuffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(vertices),gl.STATIC_DRAW);
  for(const[name,size,offset]of[['p',2,0],['binding',4,8],['upper',4,24],['flexBinding',2,40]]){const loc=gl.getAttribLocation(program,name);gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,size,gl.FLOAT,false,48,offset);}
  const indexBuffer=gl.createBuffer();gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,indexBuffer);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint16Array(indices),gl.STATIC_DRAW);
  const angleUniform=gl.getUniformLocation(program,'angles');
  const gestureUniform=gl.getUniformLocation(program,'gestures'),blinkUniform=gl.getUniformLocation(program,'blink'),gazeUniform=gl.getUniformLocation(program,'gaze');
  const flexUniform=gl.getUniformLocation(program,'flexAngles');
  const partUniform=gl.getUniformLocation(program,'part');
  const atlasUniform=gl.getUniformLocation(program,'atlasRows');
  gl.uniform2fv(gl.getUniformLocation(program,'legSeam[0]'),new Float32Array(portraitLegSeam().flat()));
  // Each piece owns its vertices and source pixels; no triangle connects the
  // oppositely rotating shoes. Premultiplied alpha preserves their occlusion.
  gl.enable(gl.BLEND);gl.blendFunc(gl.ONE,gl.ONE_MINUS_SRC_ALPHA);
  const texture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,texture);
  for(const name of [gl.TEXTURE_MIN_FILTER,gl.TEXTURE_MAG_FILTER])gl.texParameteri(gl.TEXTURE_2D,name,gl.LINEAR);
  for(const name of [gl.TEXTURE_WRAP_S,gl.TEXTURE_WRAP_T])gl.texParameteri(gl.TEXTURE_2D,name,gl.CLAMP_TO_EDGE);
  let ready=false,disposed=false,paused=true,frame=0,lastTime=0,seconds=0,pose=null,clockRate=1,targetClockRate=1;
  function contextLost(){disposed=true;cancelAnimationFrame(frame);staticPortrait();}
  canvas.addEventListener('webglcontextlost',contextLost,{once:true});
  function draw(){
    if(!ready||disposed)return;
    const angles=motion?originalPortraitAngles(pose??seconds):{near:0,far:0};
    const g=originalPortraitGestures(motion?(pose??seconds):0);
    gl.useProgram(program);gl.uniform2f(angleUniform,angles.near,angles.far);gl.uniform2f(flexUniform,g.elbow,g.wrist);gl.uniform4f(gestureUniform,g.head,g.arm,g.leftEar,g.rightEar);gl.uniform1f(blinkUniform,g.blink);gl.uniform2f(gazeUniform,...g.gaze);gl.viewport(0,0,512,768);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT);
    for(const piece of [0,1]){gl.uniform1f(partUniform,piece);gl.drawElements(gl.TRIANGLES,indices.length,gl.UNSIGNED_SHORT,0);}
    canvas.dataset.ready='true';canvas.dataset.rig=motion?'original-art-joint-mesh':'original-static-portrait';
    canvas.dataset.legPieces='independent';
    canvas.dataset.nearAngle=angles.near.toFixed(5);canvas.dataset.farAngle=angles.far.toFixed(5);
    canvas.dataset.headAngle=g.head.toFixed(5);canvas.dataset.armAngle=g.arm.toFixed(5);canvas.dataset.blink=g.blink.toFixed(3);
    canvas.dataset.elbowAngle=g.elbow.toFixed(5);canvas.dataset.wristAngle=g.wrist.toFixed(5);canvas.dataset.clockRate=clockRate.toFixed(4);
  }
  function tick(time){frame=0;if(disposed||paused||!motion)return;if(lastTime){const dt=Math.min(.1,(time-lastTime)/1000);clockRate+=(targetClockRate-clockRate)*(1-Math.exp(-dt/.7));seconds+=dt*clockRate;}lastTime=time;draw();frame=requestAnimationFrame(tick);}
  function queue(){if(ready&&motion&&!paused&&!frame){lastTime=0;frame=requestAnimationFrame(tick);}}
  const image=new Image();image.onload=()=>{
    if(disposed)return;
    gl.bindTexture(gl.TEXTURE_2D,texture);gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL,true);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,image);gl.uniform1f(atlasUniform,image.naturalHeight/1536);ready=true;draw();queue();
  };image.src=MENGLI_CHARACTER;
  return{
    setPaused(value){paused=Boolean(value);canvas.dataset.paused=String(paused);if(paused){cancelAnimationFrame(frame);frame=0;lastTime=0;}else queue();draw();},
    setPose(value){pose=value;draw();},
    setDynamics(focus){targetClockRate=bodyVisualDynamics(focus).clockRate;canvas.dataset.frequency=String(focus?.frequency??'missing');canvas.dataset.momentum=String(focus?.momentum??'missing');},
    dispose(){disposed=true;cancelAnimationFrame(frame);image.onload=null;canvas.removeEventListener('webglcontextlost',contextLost);gl.deleteTexture(texture);gl.deleteBuffer(vertexBuffer);gl.deleteBuffer(indexBuffer);gl.deleteProgram(program);},
  };
}
