"""Generate a standalone, interactive HTML5 Canvas animated monitor for 2D Phase Space."""
import json
from pathlib import Path

trajectory_file = Path("results/published/phase_space_trajectory.json")
output_file = Path("results/phase_space_monitor.html")

data = json.loads(trajectory_file.read_text(encoding="utf-8"))
data_json_str = json.dumps(data)

html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Information Boltzmann 2D Phase Space Live Monitor</title>
  <style>
    :root {{
      --bg: #0d1117;
      --card-bg: rgba(22, 27, 34, 0.85);
      --border: #30363d;
      --accent: #58a6ff;
      --accent-green: #3fb950;
      --accent-purple: #bc8cff;
      --accent-red: #f85149;
      --accent-orange: #d29922;
      --text: #c9d1d9;
      --text-bright: #f0f6fc;
      --text-muted: #8b949e;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; }}
    body {{ background: var(--bg); color: var(--text); overflow: hidden; height: 100vh; display: flex; flex-direction: column; }}

    /* Header */
    header {{
      background: #161b22; border-bottom: 1px solid var(--border);
      padding: 10px 20px; display: flex; justify-content: space-between; align-items: center; z-index: 10;
    }}
    .header-title {{ font-size: 16px; font-weight: 700; color: var(--text-bright); display: flex; align-items: center; gap: 10px; }}
    .badge {{ background: rgba(88, 166, 255, 0.15); color: var(--accent); border: 1px solid rgba(88, 166, 255, 0.4); padding: 2px 8px; border-radius: 12px; font-size: 11px; }}
    .badge-ness {{ background: rgba(63, 185, 80, 0.15); color: var(--accent-green); border-color: rgba(63, 185, 80, 0.4); }}
    .header-stats {{ font-size: 13px; color: var(--text-muted); display: flex; gap: 20px; }}
    .header-stats span strong {{ color: var(--text-bright); }}

    /* Main Container */
    .main-container {{
      flex: 1; display: grid; grid-template-columns: 1fr 380px; gap: 12px; padding: 12px; height: calc(100vh - 110px);
    }}

    /* Viewport Panel */
    .viewport-panel {{
      background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px;
      position: relative; display: flex; flex-direction: column; overflow: hidden;
    }}
    .viewport-toolbar {{
      position: absolute; top: 10px; left: 10px; z-index: 5;
      display: flex; gap: 8px; background: rgba(13, 17, 23, 0.75); backdrop-filter: blur(8px);
      padding: 6px 12px; border-radius: 6px; border: 1px solid var(--border); font-size: 12px;
    }}
    .viewport-btn {{
      background: #21262d; border: 1px solid var(--border); color: var(--text); padding: 4px 10px;
      border-radius: 4px; cursor: pointer; font-size: 12px; transition: all 0.15s;
    }}
    .viewport-btn:hover {{ background: #30363d; color: var(--text-bright); }}
    .viewport-btn.active {{ background: rgba(88, 166, 255, 0.2); color: var(--accent); border-color: var(--accent); }}

    canvas#phaseCanvas {{ width: 100%; height: 100%; display: block; }}

    /* Legend Overlay */
    .canvas-legend {{
      position: absolute; bottom: 10px; left: 10px; z-index: 5;
      background: rgba(13, 17, 23, 0.75); backdrop-filter: blur(8px);
      padding: 8px 12px; border-radius: 6px; border: 1px solid var(--border); font-size: 11px;
      display: flex; flex-direction: column; gap: 4px; pointer-events: none;
    }}
    .legend-row {{ display: flex; align-items: center; gap: 8px; }}
    .legend-dot {{ width: 8px; height: 8px; border-radius: 50%; }}

    /* Sidebar / Diagnostics */
    .sidebar {{
      display: flex; flex-direction: column; gap: 12px; overflow-y: auto;
    }}
    .card {{
      background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 14px;
    }}
    .card-header {{
      font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
      color: var(--text-muted); margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;
    }}

    /* Token Card */
    .token-display {{ display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }}
    .token-char {{
      width: 48px; height: 48px; background: #21262d; border: 1px solid var(--border);
      border-radius: 6px; display: flex; align-items: center; justify-content: center;
      font-size: 22px; font-weight: 700; color: var(--accent); font-family: monospace;
    }}
    .token-meta {{ flex: 1; }}
    .token-prob {{ font-size: 13px; color: var(--text-bright); }}
    .token-loss {{ font-size: 12px; color: var(--text-muted); }}

    .top5-list {{ display: flex; flex-direction: column; gap: 4px; margin-top: 8px; }}
    .top5-item {{ display: flex; justify-content: space-between; font-size: 11px; padding: 3px 6px; background: #161b22; border-radius: 4px; }}
    .top5-bar {{ height: 3px; background: var(--accent); border-radius: 2px; margin-top: 2px; }}

    /* Metrics Grid */
    .metrics-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
    .metric-box {{ background: #161b22; padding: 8px 10px; border-radius: 6px; border: 1px solid rgba(48, 54, 61, 0.5); }}
    .metric-label {{ font-size: 11px; color: var(--text-muted); }}
    .metric-value {{ font-size: 15px; font-weight: 700; color: var(--text-bright); margin-top: 2px; }}

    /* Charts */
    .mini-chart {{ width: 100%; height: 55px; margin-top: 6px; }}

    /* Footer / Controls */
    footer {{
      background: #161b22; border-top: 1px solid var(--border);
      padding: 10px 20px; display: flex; align-items: center; gap: 20px; z-index: 10;
    }}
    .control-btn {{
      background: #238636; border: 1px solid rgba(240, 246, 252, 0.1); color: #fff;
      padding: 6px 16px; border-radius: 6px; font-size: 13px; font-weight: 600; cursor: pointer;
    }}
    .control-btn:hover {{ background: #2ea043; }}
    .control-btn.paused {{ background: #d29922; }}
    .btn-perturb {{
      background: #da3633; border: 1px solid rgba(240, 246, 252, 0.1); color: #fff;
      padding: 6px 14px; border-radius: 6px; font-size: 12px; font-weight: 600; cursor: pointer;
    }}
    .btn-perturb:hover {{ background: #b62324; }}

    .slider-container {{ flex: 1; display: flex; align-items: center; gap: 12px; }}
    input[type=range] {{ flex: 1; accent-color: var(--accent); cursor: pointer; }}
    .speed-select {{
      background: #21262d; border: 1px solid var(--border); color: var(--text);
      padding: 4px 8px; border-radius: 4px; font-size: 12px; cursor: pointer;
    }}
  </style>
</head>
<body>

  <header>
    <div class="header-title">
      <span>Information Boltzmann 2D Phase Space Live Monitor</span>
      <span class="badge">N = 16 Particles</span>
      <span class="badge badge-ness">Active NESS (T=0.1)</span>
      <span class="badge">38,336 Params</span>
    </div>
    <div class="header-stats">
      <span>Dataset: <strong>OpenWebText</strong></span>
      <span>Damping: <strong>&gamma; = 1.0</strong></span>
      <span>Harmonic Trap: <strong>&kappa; = 1.0</strong></span>
    </div>
  </header>

  <div class="main-container">
    <!-- Viewport Panel -->
    <div class="viewport-panel">
      <div class="viewport-toolbar">
        <button class="viewport-btn active" id="btnViewPos" onclick="setViewMode('pos')">Position Space (x₁, x₂)</button>
        <button class="viewport-btn" id="btnViewPhase" onclick="setViewMode('phase')">Phase Slice (x₁, v₁)</button>
        <button class="viewport-btn" id="btnToggleTrail" onclick="toggleTrails()">Trails: ON</button>
        <button class="viewport-btn" id="btnToggleFlow" onclick="toggleFlow()">Current J: ON</button>
      </div>

      <canvas id="phaseCanvas"></canvas>

      <div class="canvas-legend">
        <div class="legend-row"><div class="legend-dot" style="background:#58a6ff;"></div><span>Continuous Particle (color = Kinetic Energy)</span></div>
        <div class="legend-row"><div class="legend-dot" style="background:#f85149;"></div><span>Expanding Shockwave (Elastic Binary Collision)</span></div>
        <div class="legend-row"><div class="legend-dot" style="background:rgba(88,166,255,0.4);"></div><span>Vector Field (J_x = v, J_v = a_I)</span></div>
        <div class="legend-row"><div class="legend-dot" style="background:rgba(255,255,255,0.2);"></div><span>Harmonic Confinement Boundary (|x| &le; 1.0)</span></div>
      </div>
    </div>

    <!-- Sidebar Diagnostics -->
    <div class="sidebar">
      <!-- Token Card -->
      <div class="card">
        <div class="card-header">
          <span>Incoming Token &amp; Prediction</span>
          <span id="stepCounter">Event 0 / 150</span>
        </div>
        <div class="token-display">
          <div class="token-char" id="tokenChar">-</div>
          <div class="token-meta">
            <div class="token-prob">P(Target) = <span id="tokenProb" style="color:var(--accent-green);font-weight:700;">-</span></div>
            <div class="token-loss">Loss (CE) = <span id="tokenCE" style="color:var(--accent-orange);">-</span></div>
          </div>
        </div>
        <div style="font-size:11px;color:var(--text-muted);margin-bottom:4px;">Top-5 Model Predictions:</div>
        <div class="top5-list" id="top5List"></div>
      </div>

      <!-- Live Physics Metrics -->
      <div class="card">
        <div class="card-header">
          <span>NESS Thermodynamics</span>
          <span style="color:var(--accent-green);">Non-Collapse Verified</span>
        </div>
        <div class="metrics-grid">
          <div class="metric-box">
            <div class="metric-label">Effective Area A_eff</div>
            <div class="metric-value" id="valAeff" style="color:var(--accent);">1.54</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">Eff. Temp T_eff</div>
            <div class="metric-value" id="valTeff" style="color:var(--accent-green);">0.095</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">Position Var(x)</div>
            <div class="metric-value" id="valVarX">0.170</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">Velocity Var(v)</div>
            <div class="metric-value" id="valVarV">0.189</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">Total Energy E(t)</div>
            <div class="metric-value" id="valEnergy">0.22</div>
          </div>
          <div class="metric-box">
            <div class="metric-label">Accepted Collisions</div>
            <div class="metric-value" id="valCollisions" style="color:var(--accent-purple);">0</div>
          </div>
        </div>
      </div>

      <!-- Real-Time Charts -->
      <div class="card" style="flex:1;">
        <div class="card-header">Effective Area A_eff(t) [Non-Collapse Signature]</div>
        <canvas id="chartAeff" class="mini-chart"></canvas>
        <div class="card-header" style="margin-top:10px;">Total Energy E(t) &amp; Dissipation Balance</div>
        <canvas id="chartEnergy" class="mini-chart"></canvas>
      </div>
    </div>
  </div>

  <!-- Footer Controls -->
  <footer>
    <button class="control-btn" id="btnPlay" onclick="togglePlay()">Pause</button>
    <button class="btn-perturb" onclick="triggerPerturbation()">⚡ Inject Perturbation (&delta;X = 1.0)</button>

    <div class="slider-container">
      <span style="font-size:12px;color:var(--text-muted);">Event Scrubber:</span>
      <input type="range" id="frameSlider" min="0" max="{len(data['frames'])-1}" value="0" oninput="seekFrame(this.value)">
    </div>

    <div style="display:flex;align-items:center;gap:8px;">
      <span style="font-size:12px;color:var(--text-muted);">Speed:</span>
      <select class="speed-select" id="speedSelect" onchange="setSpeed(this.value)">
        <option value="0.5">0.5x</option>
        <option value="1.0" selected>1.0x</option>
        <option value="2.0">2.0x</option>
        <option value="4.0">4.0x</option>
      </select>
    </div>
  </footer>

  <script>
    // Embedded Trajectory Data
    const simData = {data_json_str};
    const frames = simData.frames;
    const flowGrid = simData.flow_grid;

    let currentFrameIdx = 0;
    let isPlaying = true;
    let playbackSpeed = 1.0;
    let viewMode = 'pos'; // 'pos' or 'phase'
    let showTrails = true;
    let showFlow = true;

    // Perturbation injection offset
    let perturbationOffset = [0, 0];
    let perturbationDecay = 0;

    // Collision shockwaves
    let shockwaves = [];

    // Historical trajectories for trails
    let trails = [];
    const MAX_TRAIL = 12;

    const canvas = document.getElementById('phaseCanvas');
    const ctx = canvas.getContext('2d');

    function resizeCanvas() {{
      const rect = canvas.parentElement.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    }}
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    function setViewMode(mode) {{
      viewMode = mode;
      document.getElementById('btnViewPos').classList.toggle('active', mode === 'pos');
      document.getElementById('btnViewPhase').classList.toggle('active', mode === 'phase');
      trails = [];
    }}

    function toggleTrails() {{
      showTrails = !showTrails;
      document.getElementById('btnToggleTrail').innerText = `Trails: ${{showTrails ? 'ON' : 'OFF'}}`;
    }}

    function toggleFlow() {{
      showFlow = !showFlow;
      document.getElementById('btnToggleFlow').innerText = `Current J: ${{showFlow ? 'ON' : 'OFF'}}`;
    }}

    function togglePlay() {{
      isPlaying = !isPlaying;
      const btn = document.getElementById('btnPlay');
      btn.innerText = isPlaying ? 'Pause' : 'Play';
      btn.classList.toggle('paused', !isPlaying);
    }}

    function setSpeed(val) {{
      playbackSpeed = parseFloat(val);
    }}

    function seekFrame(val) {{
      currentFrameIdx = parseInt(val);
      updateUI();
    }}

    function triggerPerturbation() {{
      perturbationOffset = [1.2, -0.9];
      perturbationDecay = 1.0;
      shockwaves.push({{ x: 0, y: 0, r: 10, maxR: 180, alpha: 1.0, color: '#da3633' }});
    }}

    // Coordinate mapping: [-2.0, 2.0] -> canvas pixels
    function toCanvasCoords(valX, valY, w, h) {{
      const scale = Math.min(w, h) / 3.8;
      const cx = w / 2;
      const cy = h / 2;
      return [cx + valX * scale, cy - valY * scale];
    }}

    function render() {{
      const w = canvas.width / window.devicePixelRatio;
      const h = canvas.height / window.devicePixelRatio;

      ctx.clearRect(0, 0, w, h);

      // Background Grid
      ctx.strokeStyle = '#21262d';
      ctx.lineWidth = 1;
      const [c0x, c0y] = toCanvasCoords(0, 0, w, h);
      ctx.beginPath();
      ctx.moveTo(0, c0y); ctx.lineTo(w, c0y);
      ctx.moveTo(c0x, 0); ctx.lineTo(c0x, h);
      ctx.stroke();

      // Harmonic Trap circle (|x| <= 1.0)
      const [r1x, r1y] = toCanvasCoords(1.0, 0, w, h);
      const radiusPx = r1x - c0x;
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.arc(c0x, c0y, radiusPx, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.setLineDash([]);

      const frame = frames[currentFrameIdx];
      if (!frame) return;

      // Draw Flow field / Current J
      if (showFlow && viewMode === 'pos') {{
        ctx.fillStyle = 'rgba(88, 166, 255, 0.25)';
        ctx.strokeStyle = 'rgba(88, 166, 255, 0.25)';
        const coords = flowGrid.x;
        for (let i = 0; i < coords.length; i += 2) {{
          for (let j = 0; j < coords.length; j += 2) {{
            const gx = coords[i];
            const gy = coords[j];
            const fx = flowGrid.fx[j][i] * 0.15;
            const fy = flowGrid.fy[j][i] * 0.15;
            const [px, py] = toCanvasCoords(gx, gy, w, h);
            const [pEndx, pEndy] = toCanvasCoords(gx + fx, gy + fy, w, h);
            ctx.beginPath();
            ctx.moveTo(px, py);
            ctx.lineTo(pEndx, pEndy);
            ctx.stroke();
          }}
        }}
      }}

      // Decay perturbation
      if (perturbationDecay > 0) {{
        perturbationDecay -= 0.03;
        perturbationOffset[0] *= 0.96;
        perturbationOffset[1] *= 0.96;
      }}

      // Particles
      const particles = frame.particles;
      const numP = particles.x.length;

      // Update trails
      if (showTrails) {{
        const currentPos = [];
        for (let i = 0; i < numP; i++) {{
          const px = (viewMode === 'pos' ? particles.x[i][0] : particles.x[i][0]) + perturbationOffset[0];
          const py = (viewMode === 'pos' ? particles.x[i][1] : particles.v[i][0]) + perturbationOffset[1];
          currentPos.push([px, py]);
        }}
        trails.push(currentPos);
        if (trails.length > MAX_TRAIL) trails.shift();

        // Draw Trails
        ctx.lineWidth = 1.5;
        for (let i = 0; i < numP; i++) {{
          ctx.beginPath();
          for (let k = 0; k < trails.length; k++) {{
            const [tx, ty] = toCanvasCoords(trails[k][i][0], trails[k][i][1], w, h);
            if (k === 0) ctx.moveTo(tx, ty);
            else ctx.lineTo(tx, ty);
          }}
          const alpha = 0.25;
          ctx.strokeStyle = `rgba(88, 166, 255, ${{alpha}})`;
          ctx.stroke();
        }}
      }}

      // Draw Particles
      for (let i = 0; i < numP; i++) {{
        const px = (viewMode === 'pos' ? particles.x[i][0] : particles.x[i][0]) + perturbationOffset[0];
        const py = (viewMode === 'pos' ? particles.x[i][1] : particles.v[i][0]) + perturbationOffset[1];
        const vx = particles.v[i][0];
        const vy = particles.v[i][1];
        const speed = Math.sqrt(vx * vx + vy * vy);

        const [cx, cy] = toCanvasCoords(px, py, w, h);

        // Velocity Arrow
        const [vex, vey] = toCanvasCoords(px + vx * 0.25, py + vy * 0.25, w, h);
        ctx.strokeStyle = 'rgba(240, 246, 252, 0.6)';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(vex, vey);
        ctx.stroke();

        // Glowing particle sphere
        const rad = 6;
        const grad = ctx.createRadialGradient(cx, cy, 1, cx, cy, rad * 2);
        const hue = Math.min(60, speed * 40); // color by kinetic speed
        grad.addColorStop(0, '#f0f6fc');
        grad.addColorStop(0.4, `hsl(${{200 + hue}}, 100%, 65%)`);
        grad.addColorStop(1, 'rgba(88, 166, 255, 0)');

        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(cx, cy, rad * 2, 0, 2 * Math.PI);
        ctx.fill();

        ctx.fillStyle = '#fff';
        ctx.beginPath();
        ctx.arc(cx, cy, rad * 0.6, 0, 2 * Math.PI);
        ctx.fill();
      }}

      // Spawn collision shockwaves if collisions occurred
      if (frame.collisions > 0 && Math.random() < 0.4) {{
        const randP = Math.floor(Math.random() * numP);
        const [sx, sy] = toCanvasCoords(particles.x[randP][0], particles.x[randP][1], w, h);
        shockwaves.push({{ x: sx, y: sy, r: 4, maxR: 45, alpha: 0.9, color: '#f85149' }});
      }}

      // Draw and update shockwaves
      for (let i = shockwaves.length - 1; i >= 0; i--) {{
        const sw = shockwaves[i];
        ctx.strokeStyle = sw.color;
        ctx.globalAlpha = sw.alpha;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(sw.x, sw.y, sw.r, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.globalAlpha = 1.0;

        sw.r += 2.5;
        sw.alpha -= 0.04;
        if (sw.alpha <= 0 || sw.r >= sw.maxR) {{
          shockwaves.splice(i, 1);
        }}
      }}
    }}

    function updateUI() {{
      const frame = frames[currentFrameIdx];
      if (!frame) return;

      document.getElementById('stepCounter').innerText = `Event ${{frame.step + 1}} / ${{frames.length}}`;
      document.getElementById('frameSlider').value = currentFrameIdx;

      // Token Display
      document.getElementById('tokenChar').innerText = frame.token_char.replace(/\\n/, '↵').replace(/ /, '␣');
      document.getElementById('tokenProb').innerText = `${{(frame.target_prob * 100).toFixed(1)}}%`;
      document.getElementById('tokenCE').innerText = frame.ce_loss.toFixed(3);

      // Top-5 list
      const top5Html = frame.top5.map(item => `
        <div class="top5-item">
          <span>${{item.token.replace(/\\n/, '↵').replace(/ /, '␣')}}</span>
          <span style="color:var(--text-bright);font-weight:600;">${{(item.prob * 100).toFixed(1)}}%</span>
        </div>
      `).join('');
      document.getElementById('top5List').innerHTML = top5Html;

      // Metrics
      const m = frame.metrics;
      document.getElementById('valAeff').innerText = m.a_eff.toFixed(3);
      document.getElementById('valTeff').innerText = m.t_eff.toFixed(3);
      document.getElementById('valVarX').innerText = m.var_x.toFixed(3);
      document.getElementById('valVarV').innerText = m.var_v.toFixed(3);
      document.getElementById('valEnergy').innerText = m.total_energy.toFixed(3);
      document.getElementById('valCollisions').innerText = frame.collisions;

      renderMiniCharts();
    }}

    // Draw Mini Time Series Strip Charts
    function drawMiniChart(canvasId, values, color, label, targetVal = null) {{
      const c = document.getElementById(canvasId);
      const cx = c.getContext('2d');
      const w = c.width = c.clientWidth * window.devicePixelRatio;
      const h = c.height = c.clientHeight * window.devicePixelRatio;

      cx.clearRect(0, 0, w, h);

      if (values.length < 2) return;
      const minV = Math.min(...values) * 0.9;
      const maxV = Math.max(...values) * 1.1 + 0.01;

      cx.strokeStyle = color;
      cx.lineWidth = 2 * window.devicePixelRatio;
      cx.beginPath();

      for (let i = 0; i < values.length; i++) {{
        const x = (i / (values.length - 1)) * w;
        const y = h - ((values[i] - minV) / (maxV - minV)) * h;
        if (i === 0) cx.moveTo(x, y);
        else cx.lineTo(x, y);
      }}
      cx.stroke();

      // Current playhead dot
      const curX = (currentFrameIdx / (frames.length - 1)) * w;
      const curY = h - ((values[currentFrameIdx] - minV) / (maxV - minV)) * h;
      cx.fillStyle = '#fff';
      cx.beginPath();
      cx.arc(curX, curY, 4 * window.devicePixelRatio, 0, 2 * Math.PI);
      cx.fill();
    }}

    function renderMiniCharts() {{
      const aeffVals = frames.map(f => f.metrics.a_eff);
      const energyVals = frames.map(f => f.metrics.total_energy);
      drawMiniChart('chartAeff', aeffVals, '#58a6ff', 'A_eff');
      drawMiniChart('chartEnergy', energyVals, '#3fb950', 'E(t)');
    }}

    let lastTime = 0;
    const interval = 120; // ms per event at 1x
    function animLoop(time) {{
      if (isPlaying && time - lastTime > interval / playbackSpeed) {{
        currentFrameIdx = (currentFrameIdx + 1) % frames.length;
        updateUI();
        lastTime = time;
      }}
      render();
      requestAnimationFrame(animLoop);
    }}

    // Initial setup
    updateUI();
    requestAnimationFrame(animLoop);
  </script>
</body>
</html>
"""

output_file.write_text(html_content, encoding="utf-8")
print(f"Successfully generated standalone live animation monitor: {output_file}")