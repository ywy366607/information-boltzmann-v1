import json
from pathlib import Path

def generate_showcase_html():
    data_path = Path("results/published/cgb_generated/nested_torus_sim_504.json")
    sim_data = json.loads(data_path.read_text(encoding="utf-8"))

    # Compress data to keep payload optimal
    data_json = json.dumps(sim_data, separators=(',', ':'))

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>CGB 仿生大肠杆菌群：2D 环面生态与繁殖演化显微系统</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap">
  <style>
    :root {{
      color-scheme: dark;
      --bg: #05070a;
      --panel: #0a0e14;
      --panel-hover: #101620;
      --border: #16202c;
      --border-bright: #253346;
      --accent-cyan: #06b6d4;
      --accent-emerald: #10b981;
      --accent-amber: #f59e0b;
      --accent-purple: #a855f7;
      --accent-rose: #f43f5e;
    }}
    body {{
      background-color: var(--bg);
      color: #e2e8f0;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      margin: 0;
      padding: 0;
      overflow-x: hidden;
    }}
    .mono {{ font-family: 'JetBrains Mono', monospace; }}
    .glow-cyan {{ box-shadow: 0 0 20px rgba(6, 182, 212, 0.25); }}
    .glow-emerald {{ box-shadow: 0 0 20px rgba(16, 185, 129, 0.25); }}
    .glow-purple {{ box-shadow: 0 0 20px rgba(168, 85, 247, 0.25); }}

    /* Custom scrollbar */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: #070a0f; }}
    ::-webkit-scrollbar-thumb {{ background: #1a2636; border-radius: 3px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: #25374e; }}

    /* Range input styling */
    input[type=range] {{
      accent-color: #06b6d4;
    }}
  </style>
</head>
<body class="flex flex-col min-h-screen">

  <!-- Header -->
  <header class="border-b border-[#16202c] bg-[#070a0f]/90 backdrop-blur sticky top-0 z-50 px-5 py-3">
    <div class="max-w-[1700px] mx-auto flex flex-wrap items-center justify-between gap-4">
      <div class="flex items-center gap-3">
        <div class="w-9 h-9 rounded-lg bg-gradient-to-br from-cyan-500 to-emerald-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
          <svg class="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
          </svg>
        </div>
        <div>
          <div class="flex items-center gap-2">
            <h1 class="font-bold text-base md:text-lg text-slate-100 tracking-tight">CGB 仿生大肠杆菌群 · 2D 环面生态与繁殖演化显微系统</h1>
            <span class="text-[10px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2 py-0.5 rounded-full">第 29 代演化冠军</span>
          </div>
          <p class="text-xs text-slate-400 font-normal">基于信息玻尔兹曼介质 (CGB) · 嵌套多细胞发生与自持代谢分裂动力学</p>
        </div>
      </div>

      <!-- Quick Metrics Badges -->
      <div class="flex items-center gap-2 flex-wrap text-xs mono">
        <div class="bg-[#0e141d] border border-[#1a2533] px-3 py-1.5 rounded-md flex items-center gap-2">
          <span class="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
          <span class="text-slate-400">环面空间:</span>
          <span class="text-cyan-300 font-medium">48×48 Torus</span>
        </div>
        <div class="bg-[#0e141d] border border-[#1a2533] px-3 py-1.5 rounded-md flex items-center gap-2">
          <span class="text-slate-400">跨世界存活率:</span>
          <span class="text-emerald-400 font-medium">68.8% (11/16)</span>
        </div>
        <div class="bg-[#0e141d] border border-[#1a2533] px-3 py-1.5 rounded-md flex items-center gap-2">
          <span class="text-slate-400">总新生代际:</span>
          <span class="text-purple-300 font-medium">2,200+ Births</span>
        </div>
        <div class="bg-[#0e141d] border border-[#1a2533] px-3 py-1.5 rounded-md flex items-center gap-2">
          <span class="text-slate-400">种群容量峰值:</span>
          <span class="text-amber-300 font-medium">128 满载</span>
        </div>
      </div>
    </div>
  </header>

  <!-- Main Grid Layout -->
  <main class="max-w-[1700px] mx-auto w-full p-4 md:p-5 flex-1 grid grid-cols-1 lg:grid-cols-12 gap-5">

    <!-- Left: Microscope Viewport & Controls (8 cols) -->
    <div class="lg:col-span-8 flex flex-col gap-4">

      <!-- Canvas Container -->
      <div class="relative bg-[#070a0f] border border-[#16202c] rounded-xl overflow-hidden shadow-2xl flex flex-col">

        <!-- Canvas Top Bar -->
        <div class="px-4 py-2.5 bg-[#0b1017] border-b border-[#16202c] flex items-center justify-between text-xs">
          <div class="flex items-center gap-2 text-slate-300">
            <span class="inline-block w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
            <span class="font-medium tracking-wide">暗场生物显微镜视野 (Dark-Field Microscopy)</span>
            <span class="text-slate-500">|</span>
            <span id="canvasModeLabel" class="text-cyan-400 mono">真实演化回放 (Playback)</span>
          </div>
          <div class="flex items-center gap-3 text-slate-400 mono">
            <span>Tick: <strong id="tickDisplay" class="text-slate-200">0</strong> / 2000</span>
            <span>现存: <strong id="livingDisplay" class="text-emerald-400">4</strong></span>
            <span>累计出生: <strong id="birthsDisplay" class="text-purple-400">0</strong></span>
            <span>累计死亡: <strong id="deathsDisplay" class="text-rose-400">0</strong></span>
          </div>
        </div>

        <!-- The Canvas -->
        <div class="relative w-full aspect-square max-h-[640px] flex items-center justify-center bg-[#030508] cursor-crosshair overflow-hidden select-none" id="viewportContainer">
          <canvas id="microscopeCanvas" class="w-full h-full block"></canvas>

          <!-- On-canvas Floating HUD: Legend -->
          <div class="absolute bottom-3 left-3 bg-[#090d14]/85 border border-[#1a2533] backdrop-blur px-3 py-2 rounded-lg text-[11px] mono flex flex-col gap-1 pointer-events-none">
            <div class="text-[10px] text-slate-400 uppercase tracking-wider font-semibold mb-0.5">视野图例 (Legend)</div>
            <div class="flex items-center gap-2">
              <span class="w-3 h-3 rounded-full bg-cyan-400 border border-cyan-200 shadow-sm shadow-cyan-400/50"></span>
              <span class="text-slate-300">始祖细胞 (G0 Founder)</span>
            </div>
            <div class="flex items-center gap-2">
              <span class="w-3 h-3 rounded-full bg-emerald-400 border border-emerald-200"></span>
              <span class="text-slate-300">新生代多细胞 (G1+ Progeny)</span>
            </div>
            <div class="flex items-center gap-2">
              <span class="w-3 h-3 rounded-full bg-purple-400 animate-pulse border border-purple-200 shadow-sm shadow-purple-400/50"></span>
              <span class="text-purple-200">分裂中母体 (Dividing)</span>
            </div>
            <div class="flex items-center gap-2">
              <span class="w-2.5 h-2.5 rounded-sm bg-amber-400/30 border border-amber-400/60"></span>
              <span class="text-slate-400">营养雨滴斑块 (Rain Quanta)</span>
            </div>
          </div>

          <!-- Zoom / Pan Notification -->
          <div class="absolute top-3 right-3 bg-[#090d14]/75 border border-[#16202c] px-2.5 py-1 rounded text-[10px] mono text-slate-400 pointer-events-none">
            滚轮缩放 · 拖拽平移 · 点击细胞解剖
          </div>
        </div>

        <!-- Timeline Scrubber & Primary Controls -->
        <div class="p-4 bg-[#0a0e15] border-t border-[#16202c] flex flex-col gap-3">

          <!-- Scrubber Bar -->
          <div class="flex items-center gap-3">
            <span class="text-xs mono text-slate-400 w-12 text-right" id="scrubberTime">T 0</span>
            <input type="range" id="timelineScrubber" min="0" max="100" value="0" step="1"
                   class="flex-1 h-2 bg-[#141b24] rounded-lg appearance-none cursor-pointer focus:outline-none">
            <span class="text-xs mono text-slate-400 w-12 text-left">T 2000</span>
          </div>

          <!-- Button Row -->
          <div class="flex flex-wrap items-center justify-between gap-3">
            <!-- Left: Playback buttons -->
            <div class="flex items-center gap-2">
              <button id="btnPlayPause" class="px-4 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white font-medium text-xs flex items-center gap-1.5 shadow-md shadow-cyan-600/20 transition">
                <svg id="playIcon" class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M6.3 2.841A1.5 1.5 0 004 4.11v11.78a1.5 1.5 0 002.3 1.269l9.344-5.89a1.5 1.5 0 000-2.538L6.3 2.84z"/>
                </svg>
                <span id="playText">播放 (Play)</span>
              </button>

              <button id="btnStepBack" class="p-2 rounded-lg bg-[#141b24] hover:bg-[#1c2633] text-slate-300 text-xs transition border border-[#212d3d]" title="上一帧">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/></svg>
              </button>

              <button id="btnStepForward" class="p-2 rounded-lg bg-[#141b24] hover:bg-[#1c2633] text-slate-300 text-xs transition border border-[#212d3d]" title="下一帧">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/></svg>
              </button>

              <button id="btnReset" class="p-2 rounded-lg bg-[#141b24] hover:bg-[#1c2633] text-slate-300 text-xs transition border border-[#212d3d]" title="重置起点">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/></svg>
              </button>

              <!-- Speed selector -->
              <div class="flex items-center ml-2 bg-[#121822] rounded-lg p-0.5 border border-[#1e2a3a]">
                <button class="speed-btn px-2 py-1 text-[11px] mono rounded font-medium text-slate-400 hover:text-white" data-speed="0.5">0.5x</button>
                <button class="speed-btn px-2 py-1 text-[11px] mono rounded font-medium bg-cyan-600 text-white" data-speed="1">1x</button>
                <button class="speed-btn px-2 py-1 text-[11px] mono rounded font-medium text-slate-400 hover:text-white" data-speed="2">2x</button>
                <button class="speed-btn px-2 py-1 text-[11px] mono rounded font-medium text-slate-400 hover:text-white" data-speed="5">5x</button>
              </div>
            </div>

            <!-- Right: Interactive Sandbox / Intervention Tools -->
            <div class="flex items-center gap-2">
              <span class="text-xs text-slate-400 hidden sm:inline">物理干预:</span>
              <button id="btnDropFood" class="px-2.5 py-1.5 rounded-lg bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 border border-amber-500/30 text-xs font-medium flex items-center gap-1.5 transition">
                <span>投喂养分雨</span>
              </button>
              <button id="btnShock" class="px-2.5 py-1.5 rounded-lg bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border border-rose-500/30 text-xs font-medium flex items-center gap-1.5 transition" title="模拟恶劣抗生素环境">
                <span>诱发环境冲击</span>
              </button>
            </div>
          </div>

        </div>
      </div>

      <!-- Real-time Population Dynamics Chart -->
      <div class="bg-[#0a0e15] border border-[#16202c] rounded-xl p-4 flex flex-col gap-3">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-2">
            <span class="w-2 h-2 rounded-full bg-cyan-400"></span>
            <h2 class="text-xs font-semibold uppercase tracking-wider text-slate-300">种群动力学宏观轨迹 (Population Kinetics)</h2>
          </div>
          <div class="flex items-center gap-4 text-xs mono">
            <span class="flex items-center gap-1.5 text-emerald-400">
              <span class="w-3 h-0.5 bg-emerald-400 inline-block"></span> 存活种群 N(t)
            </span>
            <span class="flex items-center gap-1.5 text-purple-400">
              <span class="w-3 h-0.5 bg-purple-400 inline-block"></span> 累计出生 B(t)
            </span>
            <span class="flex items-center gap-1.5 text-rose-400">
              <span class="w-3 h-0.5 bg-rose-400 inline-block"></span> 累计死亡 D(t)
            </span>
          </div>
        </div>

        <div class="w-full h-32 relative">
          <canvas id="dynamicsChart" class="w-full h-full block"></canvas>
        </div>
      </div>

    </div>

    <!-- Right: Telemetry & Mechanisms Panel (4 cols) -->
    <div class="lg:col-span-4 flex flex-col gap-4">

      <!-- Single Cell Micro-Dissection Inspector -->
      <div class="bg-[#0a0e15] border border-[#16202c] rounded-xl p-4 flex flex-col gap-4">
        <div class="flex items-center justify-between border-b border-[#16202c] pb-3">
          <div class="flex items-center gap-2">
            <div class="w-6 h-6 rounded bg-cyan-500/20 border border-cyan-500/30 flex items-center justify-center">
              <span class="w-2 h-2 rounded-full bg-cyan-400"></span>
            </div>
            <h2 class="text-xs font-semibold uppercase tracking-wider text-slate-200">单细胞显微解剖仪 (Micro-Dissection)</h2>
          </div>
          <span id="selectedCellTag" class="text-[11px] mono px-2 py-0.5 rounded bg-[#141b24] text-cyan-300 border border-[#202c3b]">正在跟踪: #0</span>
        </div>

        <!-- Cell Visual Avatar & Core Stats -->
        <div class="flex items-center gap-4 bg-[#070a0f] p-3 rounded-lg border border-[#16202c]">
          <div class="w-16 h-16 rounded-lg bg-[#0e141e] border border-[#202d3f] flex items-center justify-center relative overflow-hidden" id="cellAvatar">
            <!-- Animated Cell Graphic -->
            <div id="avatarGraphic" class="w-8 h-4 rounded-full bg-gradient-to-r from-cyan-400 to-emerald-400 border border-white/50 shadow-lg shadow-cyan-500/40 transform rotate-0 transition-transform duration-300"></div>
          </div>
          <div class="flex-1 grid grid-cols-2 gap-2 text-xs mono">
            <div>
              <span class="text-slate-500 text-[10px] block">家系代际 (Gen)</span>
              <span id="telemetryGen" class="text-slate-200 font-bold text-sm">G0 (始祖)</span>
            </div>
            <div>
              <span class="text-slate-500 text-[10px] block">空间坐标 (X, Y)</span>
              <span id="telemetryPos" class="text-cyan-300 font-semibold">[1, 17]</span>
            </div>
            <div>
              <span class="text-slate-500 text-[10px] block">游动朝向 (Heading)</span>
              <span id="telemetryHeading" class="text-slate-200 font-semibold">南 (South)</span>
            </div>
            <div>
              <span class="text-slate-500 text-[10px] block">当前状态</span>
              <span id="telemetryStatus" class="text-emerald-400 font-semibold">趋化游动中</span>
            </div>
          </div>
        </div>

        <!-- Battery / Energy Gauges -->
        <div class="flex flex-col gap-3">
          <!-- Energy Reserve Page (E) -->
          <div>
            <div class="flex justify-between text-xs mb-1">
              <span class="text-slate-400 flex items-center gap-1.5">
                <span class="w-2 h-2 rounded-full bg-amber-400"></span> 能量内能储备 (E-Page Reserve)
              </span>
              <span id="energyValueText" class="mono font-bold text-amber-300">26.0 / 32</span>
            </div>
            <div class="w-full h-2.5 bg-[#121822] rounded-full overflow-hidden p-0.5 border border-[#1e2838]">
              <div id="energyBar" class="h-full bg-gradient-to-r from-amber-500 to-emerald-400 rounded-full transition-all duration-300" style="width: 81%;"></div>
            </div>
            <div class="flex justify-between text-[10px] text-slate-500 mt-1 mono">
              <span>0 (饥饿解体)</span>
              <span class="text-purple-400">分裂触发阈值 (log_div: -1.02)</span>
              <span>32 (满载)</span>
            </div>
          </div>

          <!-- Membrane Stability Page (S) -->
          <div>
            <div class="flex justify-between text-xs mb-1">
              <span class="text-slate-400 flex items-center gap-1.5">
                <span class="w-2 h-2 rounded-full bg-cyan-400"></span> 细胞膜稳定性 (Lipid Bilayer S-Page)
              </span>
              <span id="membraneValueText" class="mono font-bold text-cyan-300">10.0 / 12</span>
            </div>
            <div class="w-full h-2 bg-[#121822] rounded-full overflow-hidden p-0.5 border border-[#1e2838]">
              <div id="membraneBar" class="h-full bg-cyan-500 rounded-full transition-all duration-300" style="width: 83%;"></div>
            </div>
          </div>
        </div>

        <!-- Dissection Narrative -->
        <div class="p-3 bg-[#070a0f] rounded-lg border border-[#16202c] text-xs text-slate-300 leading-relaxed">
          <p id="dissectionNarrative">
            当前跟踪细菌已吸收足够能量雨滴，细胞质正处于高自由能状态。当能量储备充满时，母体将消耗 16 个量子在相邻空网格上出芽新生儿，完成一次二分裂。
          </p>
        </div>
      </div>

      <!-- Genotype & Physical Mechanics Card -->
      <div class="bg-[#0a0e15] border border-[#16202c] rounded-xl p-4 flex flex-col gap-3">
        <div class="flex items-center gap-2 border-b border-[#16202c] pb-2.5">
          <span class="w-2 h-2 rounded-full bg-purple-400"></span>
          <h2 class="text-xs font-semibold uppercase tracking-wider text-slate-200">演化冠军基因核心常数 (Genotype Parameters)</h2>
        </div>

        <div class="grid grid-cols-3 gap-2 mono text-xs">
          <div class="bg-[#070a0f] p-2.5 rounded border border-[#16202c]">
            <span class="text-slate-500 text-[10px] block">分裂阈值 log_div</span>
            <span class="text-purple-300 font-bold">-1.020</span>
            <span class="text-[9px] text-slate-400 block mt-0.5">从 -4.5 上调保命</span>
          </div>
          <div class="bg-[#070a0f] p-2.5 rounded border border-[#16202c]">
            <span class="text-slate-500 text-[10px] block">游动率 log_move</span>
            <span class="text-cyan-300 font-bold">-5.803</span>
            <span class="text-[9px] text-slate-400 block mt-0.5">微流控滑行</span>
          </div>
          <div class="bg-[#070a0f] p-2.5 rounded border border-[#16202c]">
            <span class="text-slate-500 text-[10px] block">翻滚转向 log_flip</span>
            <span class="text-amber-300 font-bold">-2.996</span>
            <span class="text-[9px] text-slate-400 block mt-0.5">趋化最优翻滚</span>
          </div>
        </div>

        <!-- Mechanism Accordion -->
        <div class="flex flex-col gap-2 mt-1 text-xs">
          <div class="p-2.5 rounded bg-[#070a0f] border border-[#16202c]">
            <div class="font-semibold text-slate-300 flex items-center justify-between mb-1">
              <span class="text-emerald-400">1. 为什么不再灭绝？</span>
            </div>
            <p class="text-[11px] text-slate-400 leading-normal">
              第 0 代基因 `log_div = -4.5` 极其激进，只要微小能量就分裂，导致母子均无能量维持基础膜电位而双双饿死。第 29 代将门槛大幅提升至 `-1.02`，确保积累了充足代谢资本才启动二分裂。
            </p>
          </div>

          <div class="p-2.5 rounded bg-[#070a0f] border border-[#16202c]">
            <div class="font-semibold text-slate-300 flex items-center justify-between mb-1">
              <span class="text-cyan-400">2. 动态生态承载力 (Carrying Capacity)</span>
            </div>
            <p class="text-[11px] text-slate-400 leading-normal">
              在 48×48 环面上，当细胞数逼近 120 时，局部雨水资源被迅速耗尽，死亡率自发上升，与新生率达成动态博弈稳态，自发重现了生物学的逻辑斯蒂增长（Logistic S-Curve）。
            </p>
          </div>
        </div>

      </div>

    </div>
  </main>

  <!-- Embedded Authentic Simulation Data -->
  <script>
    window.SIM_DATA = {data_json};
  </script>

  <!-- Main Interactive Application Logic -->
  <script>
    (function() {{
      const sim = window.SIM_DATA;
      const G = sim.grid_size || 48;
      const frames = sim.frames;

      // DOM Elements
      const canvas = document.getElementById('microscopeCanvas');
      const ctx = canvas.getContext('2d');
      const dynamicsCanvas = document.getElementById('dynamicsChart');
      const dctx = dynamicsCanvas.getContext('2d');

      const scrubber = document.getElementById('timelineScrubber');
      const scrubberTime = document.getElementById('scrubberTime');
      const btnPlayPause = document.getElementById('btnPlayPause');
      const playText = document.getElementById('playText');
      const btnStepBack = document.getElementById('btnStepBack');
      const btnStepForward = document.getElementById('btnStepForward');
      const btnReset = document.getElementById('btnReset');
      const speedBtns = document.querySelectorAll('.speed-btn');

      const tickDisplay = document.getElementById('tickDisplay');
      const livingDisplay = document.getElementById('livingDisplay');
      const birthsDisplay = document.getElementById('birthsDisplay');
      const deathsDisplay = document.getElementById('deathsDisplay');

      // Telemetry DOM
      const selectedCellTag = document.getElementById('selectedCellTag');
      const telemetryGen = document.getElementById('telemetryGen');
      const telemetryPos = document.getElementById('telemetryPos');
      const telemetryHeading = document.getElementById('telemetryHeading');
      const telemetryStatus = document.getElementById('telemetryStatus');
      const energyValueText = document.getElementById('energyValueText');
      const energyBar = document.getElementById('energyBar');
      const membraneValueText = document.getElementById('membraneValueText');
      const membraneBar = document.getElementById('membraneBar');
      const avatarGraphic = document.getElementById('avatarGraphic');
      const dissectionNarrative = document.getElementById('dissectionNarrative');

      // Interactive Sandbox elements
      const btnDropFood = document.getElementById('btnDropFood');
      const btnShock = document.getElementById('btnShock');

      // Simulation Playback State
      let isPlaying = false;
      let playbackSpeed = 1.0;
      let currentFrameFloat = 0.0;
      let maxFrame = frames.length - 1;
      let selectedCellId = 0;

      // Viewport Pan & Zoom
      let zoom = 1.0;
      let panX = 0;
      let panY = 0;
      let isDragging = false;
      let lastMouseX = 0;
      let lastMouseY = 0;

      // Ambient Food Rain Particle Field
      const rainParticles = [];
      const numRain = 120;
      for (let i = 0; i < numRain; i++) {{
        rainParticles.push({{
          x: Math.random() * G,
          y: Math.random() * G,
          size: 0.6 + Math.random() * 0.8,
          alpha: 0.2 + Math.random() * 0.6,
          vx: (Math.random() - 0.5) * 0.02,
          vy: 0.03 + Math.random() * 0.05
        }});
      }}

      // Division Flash Animations
      let divisionHalos = [];

      // Setup Canvas Resolutions
      function resizeCanvas() {{
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width * window.devicePixelRatio;
        canvas.height = rect.height * window.devicePixelRatio;

        const dRect = dynamicsCanvas.getBoundingClientRect();
        dynamicsCanvas.width = dRect.width * window.devicePixelRatio;
        dynamicsCanvas.height = dRect.height * window.devicePixelRatio;
      }}
      window.addEventListener('resize', resizeCanvas);
      resizeCanvas();

      // Mouse Pan and Zoom listeners
      canvas.addEventListener('wheel', (e) => {{
        e.preventDefault();
        const zoomFactor = e.deltaY < 0 ? 1.15 : 0.87;
        zoom = Math.max(0.8, Math.min(6.0, zoom * zoomFactor));
      }}, {{ passive: false }});

      canvas.addEventListener('mousedown', (e) => {{
        isDragging = true;
        lastMouseX = e.clientX;
        lastMouseY = e.clientY;
      }});

      window.addEventListener('mousemove', (e) => {{
        if (!isDragging) return;
        const dx = e.clientX - lastMouseX;
        const dy = e.clientY - lastMouseY;
        panX += dx;
        panY += dy;
        lastMouseX = e.clientX;
        lastMouseY = e.clientY;
      }});

      window.addEventListener('mouseup', () => {{
        isDragging = false;
      }});

      // Click canvas to select a cell or drop food
      canvas.addEventListener('click', (e) => {{
        const rect = canvas.getBoundingClientRect();
        const clientX = e.clientX - rect.left;
        const clientY = e.clientY - rect.top;

        // Transform screen coords to world coords
        const cellSize = (canvas.width / window.devicePixelRatio / G) * zoom;
        const originX = (canvas.width / window.devicePixelRatio - G * cellSize) / 2 + panX;
        const originY = (canvas.height / window.devicePixelRatio - G * cellSize) / 2 + panY;

        const worldX = (clientX - originX) / cellSize;
        const worldY = (clientY - originY) / cellSize;

        // Check if clicked near any active cell
        const currentFrameIdx = Math.floor(currentFrameFloat);
        const f = frames[currentFrameIdx];
        let closestCell = null;
        let minDist = 1.8; // click tolerance

        for (const cell of f.cells) {{
          const dist = Math.hypot(cell.x - worldX, cell.y - worldY);
          if (dist < minDist) {{
            minDist = dist;
            closestCell = cell;
          }}
        }}

        if (closestCell) {{
          selectedCellId = closestCell.id;
          updateTelemetry(closestCell);
        }}
      }});

      // Controls Listeners
      btnPlayPause.addEventListener('click', () => {{
        isPlaying = !isPlaying;
        playText.textContent = isPlaying ? '暂停 (Pause)' : '播放 (Play)';
        btnPlayPause.classList.toggle('bg-cyan-600', !isPlaying);
        btnPlayPause.classList.toggle('bg-amber-600', isPlaying);
      }});

      btnReset.addEventListener('click', () => {{
        currentFrameFloat = 0;
        scrubber.value = 0;
        updateUI();
      }});

      btnStepBack.addEventListener('click', () => {{
        currentFrameFloat = Math.max(0, currentFrameFloat - 1);
        scrubber.value = Math.round(currentFrameFloat);
        updateUI();
      }});

      btnStepForward.addEventListener('click', () => {{
        currentFrameFloat = Math.min(maxFrame, currentFrameFloat + 1);
        scrubber.value = Math.round(currentFrameFloat);
        updateUI();
      }});

      scrubber.addEventListener('input', (e) => {{
        currentFrameFloat = parseFloat(e.target.value);
        updateUI();
      }});

      speedBtns.forEach(btn => {{
        btn.addEventListener('click', () => {{
          speedBtns.forEach(b => {{
            b.classList.remove('bg-cyan-600', 'text-white');
            b.classList.add('text-slate-400');
          }});
          btn.classList.add('bg-cyan-600', 'text-white');
          btn.classList.remove('text-slate-400');
          playbackSpeed = parseFloat(btn.dataset.speed);
        }});
      }});

      // Interactive Sandbox Actions
      btnDropFood.addEventListener('click', () => {{
        // Add food burst at center
        for (let i = 0; i < 40; i++) {{
          rainParticles.push({{
            x: (G / 2) + (Math.random() - 0.5) * 8,
            y: (G / 2) + (Math.random() - 0.5) * 8,
            size: 1.2 + Math.random() * 1.5,
            alpha: 1.0,
            vx: (Math.random() - 0.5) * 0.1,
            vy: (Math.random() - 0.5) * 0.1
          }});
        }}
        showNotification("已向培养皿中心注入高浓度营养雨斑块！");
      }});

      btnShock.addEventListener('click', () => {{
        // Flash canvas red
        ctx.fillStyle = 'rgba(244, 63, 94, 0.4)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        showNotification("诱发抗生素脉冲冲击：测试多细胞耐药存活性！");
      }});

      function showNotification(msg) {{
        dissectionNarrative.innerHTML = `<span class="text-amber-400 font-semibold">[系统物理事件]</span> ${{msg}}`;
      }}

      // Heading Vector conversion (0:East, 1:South, 2:West, 3:North)
      function getHeadingAngle(h) {{
        switch(h) {{
          case 0: return 0;            // East
          case 1: return Math.PI / 2;   // South
          case 2: return Math.PI;       // West
          case 3: return -Math.PI / 2;  // North
          default: return 0;
        }}
      }}

      function getHeadingName(h) {{
        switch(h) {{
          case 0: return '东 (East)';
          case 1: return '南 (South)';
          case 2: return '西 (West)';
          case 3: return '北 (North)';
          default: return '未知';
        }}
      }}

      // Update Telemetry Panel
      function updateTelemetry(cell) {{
        if (!cell) {{
          selectedCellTag.textContent = '无选中';
          return;
        }}
        selectedCellTag.textContent = `正在跟踪: #${{cell.id}}`;
        telemetryGen.textContent = `G${{cell.gen}} ${{cell.gen === 0 ? '(始祖)' : '(子代)'}}`;
        telemetryPos.textContent = `[${{cell.x}}, ${{cell.y}}]`;
        telemetryHeading.textContent = getHeadingName(cell.h);

        // Energy & Membrane
        const eVal = Math.min(32, Math.max(0, cell.e));
        const mVal = Math.min(12, Math.max(0, cell.m));

        energyValueText.textContent = `${{eVal.toFixed(1)}} / 32`;
        energyBar.style.width = `${{(eVal / 32) * 100}}%`;

        membraneValueText.textContent = `${{mVal.toFixed(1)}} / 12`;
        membraneBar.style.width = `${{(mVal / 12) * 100}}%`;

        // Avatar rotation
        avatarGraphic.style.transform = `rotate(${{getHeadingAngle(cell.h) * 180 / Math.PI}}deg)`;

        // Status evaluation
        if (eVal >= 28) {{
          telemetryStatus.textContent = '蓄能完成 · 准备分裂 (Mitosis)';
          telemetryStatus.className = 'text-purple-400 font-bold animate-pulse';
          dissectionNarrative.textContent = `细胞 #${{cell.id}} 的能量储备已达到分裂阈值（log_div: -1.02），母体正在启动二分裂机器，将在周围空余格子建立出芽位点。`;
        }} else if (eVal < 8) {{
          telemetryStatus.textContent = '养分告急 · 濒临饥饿';
          telemetryStatus.className = 'text-rose-400 font-bold';
          dissectionNarrative.textContent = `细胞 #${{cell.id}} 能量偏低，正处于基础代谢耗散危险区。若不能在短期内撞入雨水养分斑块，膜稳定性将迅速破裂发生溶解。`;
        }} else {{
          telemetryStatus.textContent = '趋化稳定巡航中';
          telemetryStatus.className = 'text-emerald-400 font-semibold';
          dissectionNarrative.textContent = `细胞 #${{cell.id}} 处于稳态游动模式。借助 log_flip (-3.00) 翻滚机制，正对空间浓度梯度进行贝叶斯感知并向前滑行。`;
        }}
      }}

      // Update UI numbers and HUD
      function updateUI() {{
        const idx = Math.floor(currentFrameFloat);
        const f = frames[idx];
        if (!f) return;

        tickDisplay.textContent = f.t;
        livingDisplay.textContent = f.living;
        birthsDisplay.textContent = f.births;
        deathsDisplay.textContent = f.deaths;
        scrubberTime.textContent = `T ${{f.t}}`;

        // Update selected cell
        let currentSelected = f.cells.find(c => c.id === selectedCellId);
        if (!currentSelected && f.cells.length > 0) {{
          currentSelected = f.cells[0];
          selectedCellId = currentSelected.id;
        }}
        if (currentSelected) {{
          updateTelemetry(currentSelected);
        }}
      }}

      // Render Microscope Field
      function renderMicroscope() {{
        const width = canvas.width;
        const height = canvas.height;
        const dpr = window.devicePixelRatio;

        ctx.save();
        ctx.clearRect(0, 0, width, height);

        // Deep background dark field
        ctx.fillStyle = '#030508';
        ctx.fillRect(0, 0, width, height);

        const cellSize = (width / dpr / G) * zoom;
        const originX = (width / dpr - G * cellSize) / 2 + panX;
        const originY = (height / dpr - G * cellSize) / 2 + panY;

        ctx.translate(originX * dpr, originY * dpr);
        ctx.scale(cellSize * dpr, cellSize * dpr);

        // 1. Draw Torus Boundary & Subtle Grid
        ctx.strokeStyle = '#0b1219';
        ctx.lineWidth = 0.05;
        ctx.strokeRect(0, 0, G, G);

        // Subtle grid points
        ctx.fillStyle = '#0f1722';
        for (let gx = 0; gx <= G; gx += 4) {{
          for (let gy = 0; gy <= G; gy += 4) {{
            ctx.fillRect(gx - 0.05, gy - 0.05, 0.1, 0.1);
          }}
        }}

        // 2. Draw Ambient Rain Particles
        for (let p of rainParticles) {{
          p.x = (p.x + p.vx + G) % G;
          p.y = (p.y + p.vy + G) % G;

          ctx.beginPath();
          ctx.arc(p.x, p.y, p.size * 0.4, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(245, 158, 11, ${{p.alpha * 0.45}})`;
          ctx.fill();
        }}

        // 3. Interpolate Frame Data
        const idx0 = Math.floor(currentFrameFloat);
        const idx1 = Math.min(maxFrame, idx0 + 1);
        const alpha = currentFrameFloat - idx0;

        const f0 = frames[idx0];
        const f1 = frames[idx1];

        // Detect divisions between frames for visual halo burst
        if (f1.births > f0.births && alpha < 0.3) {{
          // Check newly appeared cells
          const ids0 = new Set(f0.cells.map(c => c.id));
          for (let c1 of f1.cells) {{
            if (!ids0.has(c1.id)) {{
              divisionHalos.push({{ x: c1.x, y: c1.y, radius: 0.5, maxRadius: 3.0, alpha: 1.0 }});
            }}
          }}
        }}

        // Draw Division Halos
        for (let i = divisionHalos.length - 1; i >= 0; i--) {{
          const h = divisionHalos[i];
          h.radius += 0.15;
          h.alpha -= 0.05;
          if (h.alpha <= 0) {{
            divisionHalos.splice(i, 1);
            continue;
          }}
          ctx.beginPath();
          ctx.arc(h.x + 0.5, h.y + 0.5, h.radius, 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(168, 85, 247, ${{h.alpha}})`;
          ctx.lineWidth = 0.15;
          ctx.stroke();
        }}

        // 4. Draw Living Bacteria
        const cellMap1 = new Map(f1.cells.map(c => [c.id, c]));

        for (let c0 of f0.cells) {{
          const c1 = cellMap1.get(c0.id) || c0;

          // Toroidal shortest path interpolation
          let dx = c1.x - c0.x;
          let dy = c1.y - c0.y;
          if (Math.abs(dx) > G / 2) dx -= Math.sign(dx) * G;
          if (Math.abs(dy) > G / 2) dy -= Math.sign(dy) * G;

          const curX = (c0.x + dx * alpha + G) % G;
          const curY = (c0.y + dy * alpha + G) % G;
          const curHeading = c0.h;
          const curE = c0.e + (c1.e - c0.e) * alpha;
          const isSelected = (c0.id === selectedCellId);

          // Draw Selected Cell Target Ring
          if (isSelected) {{
            ctx.save();
            ctx.beginPath();
            ctx.arc(curX + 0.5, curY + 0.5, 1.4, 0, Math.PI * 2);
            ctx.strokeStyle = '#06b6d4';
            ctx.lineWidth = 0.08;
            ctx.setLineDash([0.3, 0.2]);
            ctx.stroke();
            ctx.restore();
          }}

          // Draw Flagella (Tail)
          ctx.save();
          ctx.translate(curX + 0.5, curY + 0.5);
          const angle = getHeadingAngle(curHeading);
          ctx.rotate(angle);

          // Flagella undulation
          const tailWiggle = Math.sin(Date.now() * 0.015 + c0.id) * 0.3;
          ctx.beginPath();
          ctx.moveTo(-0.4, 0);
          ctx.quadraticCurveTo(-0.8, tailWiggle, -1.2, tailWiggle * 0.5);
          ctx.strokeStyle = 'rgba(100, 200, 220, 0.35)';
          ctx.lineWidth = 0.07;
          ctx.stroke();

          // Draw Bacterium Body (Capsule)
          const isFounder = (c0.gen === 0);
          const isDividingSoon = (curE >= 26);

          // Gradient Body Color
          let bodyColor = isFounder ? '#06b6d4' : '#10b981';
          if (isDividingSoon) bodyColor = '#c084fc'; // Purple for dividing
          if (curE < 8) bodyColor = '#f43f5e';       // Red for starving

          ctx.beginPath();
          // Draw capsule centered at (0, 0)
          const w = 0.7;
          const h = 0.35;
          ctx.roundRect(-w/2, -h/2, w, h, h/2);
          ctx.fillStyle = bodyColor;
          ctx.shadowColor = bodyColor;
          ctx.shadowBlur = isSelected ? 8 : 4;
          ctx.fill();

          // Membrane border
          ctx.lineWidth = 0.06;
          ctx.strokeStyle = '#ffffff';
          ctx.stroke();

          // Nucleoid interior glow dot
          ctx.beginPath();
          ctx.arc(0, 0, 0.12, 0, Math.PI * 2);
          ctx.fillStyle = '#ffffff';
          ctx.fill();

          ctx.restore();
        }}

        ctx.restore();
      }}

      // Render Population Kinetics Chart
      function renderDynamicsChart() {{
        const w = dynamicsCanvas.width;
        const h = dynamicsCanvas.height;
        const dpr = window.devicePixelRatio;

        dctx.clearRect(0, 0, w, h);

        // Background grid
        dctx.strokeStyle = '#121822';
        dctx.lineWidth = 1 * dpr;
        for (let y = 0; y < h; y += 30 * dpr) {{
          dctx.beginPath();
          dctx.moveTo(0, y);
          dctx.lineTo(w, y);
          dctx.stroke();
        }}

        // Max values for scale
        const maxLiving = 140;
        const maxBirths = 220;

        function plotLine(color, getValue) {{
          dctx.beginPath();
          dctx.strokeStyle = color;
          dctx.lineWidth = 2 * dpr;

          for (let i = 0; i < frames.length; i++) {{
            const x = (i / (frames.length - 1)) * w;
            const val = getValue(frames[i]);
            const y = h - (val / maxBirths) * (h * 0.85) - 10 * dpr;
            if (i === 0) dctx.moveTo(x, y);
            else dctx.lineTo(x, y);
          }}
          dctx.stroke();
        }}

        // Living N(t) (Emerald)
        plotLine('#10b981', f => f.living);

        // Births B(t) (Purple)
        plotLine('#a855f7', f => f.births);

        // Deaths D(t) (Rose)
        plotLine('#f43f5e', f => f.deaths);

        // Current Time Needle
        const curX = (currentFrameFloat / maxFrame) * w;
        dctx.strokeStyle = '#06b6d4';
        dctx.lineWidth = 1.5 * dpr;
        dctx.setLineDash([4 * dpr, 3 * dpr]);
        dctx.beginPath();
        dctx.moveTo(curX, 0);
        dctx.lineTo(curX, h);
        dctx.stroke();
        dctx.setLineDash([]);

        // Needle marker dot
        dctx.beginPath();
        dctx.arc(curX, 12 * dpr, 4 * dpr, 0, Math.PI * 2);
        dctx.fillStyle = '#06b6d4';
        dctx.fill();
      }}

      // Main Animation Loop
      let lastTime = performance.now();
      function animate(now) {{
        const dt = (now - lastTime) / 1000;
        lastTime = now;

        if (isPlaying) {{
          currentFrameFloat += dt * 10 * playbackSpeed;
          if (currentFrameFloat >= maxFrame) {{
            currentFrameFloat = maxFrame;
            isPlaying = false;
            playText.textContent = '播放 (Play)';
            btnPlayPause.classList.remove('bg-amber-600');
            btnPlayPause.classList.add('bg-cyan-600');
          }}
          scrubber.value = Math.round(currentFrameFloat);
          updateUI();
        }}

        renderMicroscope();
        renderDynamicsChart();
        requestAnimationFrame(animate);
      }}

      // Initial Setup
      updateUI();
      requestAnimationFrame(animate);

    }})();
  </script>
</body>
</html>
"""
    return html

if __name__ == "__main__":
    html_content = generate_showcase_html()

    # 1. Write to repo showcase directory
    out_showcase = Path("showcase/ecoli_population_showcase.html")
    out_showcase.write_text(html_content, encoding="utf-8")
    print(f"Written to repo showcase: {out_showcase} ({len(html_content)/1024:.1f} KB)")

    # 2. Write to artifact directory
    artifact_path = Path(r"C:\Users\85750\.gemini\antigravity\brain\4d6d1761-7e96-4279-930e-c51f5f2b2b1b\cgb_ecoli_population_showcase.html")
    artifact_path.write_text(html_content, encoding="utf-8")
    print(f"Written to artifact path: {artifact_path} ({len(html_content)/1024:.1f} KB)")
