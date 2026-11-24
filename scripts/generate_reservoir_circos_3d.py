"""Generate a High-End Cinematic Dark-Field 3D Circos Connectome with WebGL GPU Post-Processing:
- Multi-Pass Bloom (Kawase Dual Blur)
- CineStill 800T Warm Red-Amber Halation (光渗扩散)
- Photochemical Midtone Film Grain (胶片颗粒)
- ACES Filmic Tone Mapping
for Information Boltzmann / MengLi at Cycle 2401.

Tech Stack: TypeScript + Three.js + WebGL GPU Custom Shaders.
"""
import gzip
import itertools
import json
import math
import sqlite3
from pathlib import Path
import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_OBS = Path(r"C:\Users\85750\.codex\cgb_native_topology_20261116\work\collect_observations_from_2383\operations.jsonl.gz")
DB_PATH = Path(r"C:\Users\85750\.codex\cgb_native_topology_20261116\body\causal_archive_511e944037da1e9128b452aba47402f5da5f8bb688919b97f896d0582dded02a.sqlite")
OUTPUT_HTML = ROOT / "results/published/cgb_reservoir_circos_3d.html"
ARTIFACT_DIR = Path(r"C:\Users\85750\.gemini\antigravity\brain\b52ce350-543f-4727-bcda-4a922c34066a")
ARTIFACT_HTML = ARTIFACT_DIR / "cgb_reservoir_circos_3d.html"
THREE_JS_SRC = ROOT / "results/published/assets/three.min.js"
ORBIT_JS_SRC = ROOT / "results/published/assets/OrbitControls.js"

def main():
    print("Reading genuine operations observations...")
    last = None
    with gzip.open(SOURCE_OBS, 'rt', encoding='utf8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['kind'] == 'structural_relation_dependencies':
                last = row
    raw = last['data']['edges']
    relations = sorted({(r, g) for r, u, p, g, v in raw})
    unknowns = sorted({(u, p) for r, u, p, g, v in raw})
    ri = {x: i for i, x in enumerate(relations)}
    ui = {x: i for i, x in enumerate(unknowns)}
    supports = [set() for _ in unknowns]
    for r, u, p, g, v in raw:
        supports[ui[u, p]].add(ri[r, g])

    witness_edges = {}
    for u, s in enumerate(supports):
        for c in itertools.combinations(sorted(s), 2):
            witness_edges.setdefault(c, u)
    sorted_edges = sorted(witness_edges.keys())

    num_nodes = len(relations)
    print(f"Total relations: {num_nodes}, edges: {len(sorted_edges)}")

    # Read SQLite
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT relation_id, payload FROM relation_program;")
    rel_supports_map = {}
    for r_id, payload_str in c.fetchall():
        try:
            p = json.loads(payload_str)
            sups = [ep for ep in p.get('support_episode_ids', []) if ep > 0]
            rel_supports_map[r_id] = sups
        except:
            rel_supports_map[r_id] = []
    conn.close()

    # Build Graph
    G = nx.Graph()
    for i in range(num_nodes):
        G.add_node(i)
    for u, v in sorted_edges:
        G.add_edge(u, v)

    degrees = [G.degree(i) for i in range(num_nodes)]
    comps = list(nx.connected_components(G))
    comps.sort(key=len, reverse=True)
    comp_map = {}
    for ci, comp in enumerate(comps):
        for node in comp:
            comp_map[node] = (ci, len(comp))

    norm_factor = (num_nodes - 1) * (num_nodes - 2)

    # Centralities
    sp_betweenness_all = nx.betweenness_centrality(G, normalized=True)

    rw_betweenness_all = {}
    for comp in comps:
        sub = G.subgraph(comp)
        if len(comp) > 2:
            res = nx.current_flow_betweenness_centrality(sub, normalized=False)
            for n, raw_val in res.items():
                rw_betweenness_all[n] = raw_val / norm_factor
        else:
            for n in comp:
                rw_betweenness_all[n] = 0.0

    rw_sorted_all = sorted(range(num_nodes), key=lambda n: (rw_betweenness_all[n], G.degree(n), relations[n][0]), reverse=True)
    rw_global_ranks = {n: r + 1 for r, n in enumerate(rw_sorted_all)}
    top_10_rw_nodes = set(rw_sorted_all[:10])

    sp_sorted_all = sorted(range(num_nodes), key=lambda n: (sp_betweenness_all[n], G.degree(n), relations[n][0]), reverse=True)
    sp_global_ranks = {n: r + 1 for r, n in enumerate(sp_sorted_all)}

    # Special markers
    target_quartet = {202, 204, 221, 254}
    bridge_ids = {213, 237}
    harmonic_eids = {59, 78, 79, 80, 81}
    harmonic_nodes = set()
    for eid in harmonic_eids:
        if eid < len(sorted_edges):
            harmonic_nodes.add(sorted_edges[eid][0])
            harmonic_nodes.add(sorted_edges[eid][1])

    # 3D Spring Layout for Morphing
    pos_3d = nx.spring_layout(G, dim=3, iterations=180, seed=42, k=0.18)
    coords_3d = np.array([pos_3d[i] for i in range(num_nodes)])
    c_min, c_max = coords_3d.min(axis=0), coords_3d.max(axis=0)
    coords_3d_norm = (coords_3d - c_min) / (c_max - c_min + 1e-9) * 80.0 - 40.0
    coords_3d_norm[:, 1] += 15.0

    sectors = [
        {"name": "巨组件核心", "short": "巨组件", "color": "#38bdf8", "desc": "包含62个同调洞与主依赖"},
        {"name": "四重奏群", "short": "四重奏", "color": "#f43f5e", "desc": "独立孵化群"},
        {"name": "次级拓扑簇", "short": "卫星簇", "color": "#818cf8", "desc": "二元/三元小群"},
        {"name": "外围单节点", "short": "孤立点", "color": "#94a3b8", "desc": "外围事实节点"}
    ]

    node_sector = [0] * num_nodes
    for i in range(num_nodes):
        ci, csize = comp_map[i]
        rel_id = relations[i][0]
        if rel_id in target_quartet or (csize == 5):
            node_sector[i] = 1
        elif csize == 253:
            node_sector[i] = 0
        elif csize in (2, 3):
            node_sector[i] = 2
        else:
            node_sector[i] = 3

    # Sort nodes for circular placement
    def sort_key(i):
        sec = node_sector[i]
        rel_id, gen = relations[i]
        rw = rw_betweenness_all[i]
        return (sec, gen, -rw, rel_id)

    sorted_indices = sorted(range(num_nodes), key=sort_key)

    # Circos Angle Calculations
    R_circos = 50.0
    gap_rad = 0.08
    total_gaps = len(sectors) * gap_rad
    usable_rad = 2.0 * math.pi - total_gaps

    node_angles = [0.0] * num_nodes
    current_angle = 0.0
    current_sec = -1
    for rank, idx in enumerate(sorted_indices):
        sec = node_sector[idx]
        if sec != current_sec:
            if current_sec != -1:
                current_angle += gap_rad
            current_sec = sec
        step = usable_rad / num_nodes
        node_angles[idx] = current_angle
        current_angle += step

    for s in range(len(sectors)):
        sec_nodes = [i for i in range(num_nodes) if node_sector[i] == s]
        if sec_nodes:
            s_min = min(node_angles[i] for i in sec_nodes)
            s_max = max(node_angles[i] for i in sec_nodes)
            sectors[s]["angle_start"] = s_min
            sectors[s]["angle_end"] = s_max
            sectors[s]["count"] = len(sec_nodes)

    coords_circos = np.zeros((num_nodes, 3))
    for i in range(num_nodes):
        a = node_angles[i]
        coords_circos[i, 0] = R_circos * math.cos(a)
        coords_circos[i, 1] = 0.0
        coords_circos[i, 2] = R_circos * math.sin(a)

    gen_colors = {
        1: "#22d3ee",
        2: "#2dd4bf",
        3: "#34d399",
        4: "#facc15",
        5: "#fb923c",
        6: "#f87171"
    }

    # Build JSON nodes
    nodes_payload = []
    for i, (rel_id, gen) in enumerate(relations):
        sec = node_sector[i]
        is_h = (i in harmonic_nodes)
        is_q = (rel_id in target_quartet)
        is_top_rw = (i in top_10_rw_nodes)
        sups = rel_supports_map.get(rel_id, [])
        ci, csize = comp_map[i]
        rw_val = rw_betweenness_all[i]
        sp_val = sp_betweenness_all[i]
        rw_rank = rw_global_ranks[i]
        sp_rank = sp_global_ranks[i]

        nodes_payload.append({
            "id": i,
            "rel_id": rel_id,
            "gen": gen,
            "gen_color": gen_colors.get(gen, "#94a3b8"),
            "sector": sec,
            "sector_name": sectors[sec]["short"],
            "comp_size": csize,
            "degree": degrees[i],
            "rw_betweenness": round(float(rw_val), 5),
            "rw_rank": rw_rank,
            "sp_betweenness": round(float(sp_val), 5),
            "sp_rank": sp_rank,
            "is_top_rw": is_top_rw,
            "supports": sups,
            "supports_count": len(sups),
            "is_harmonic": is_h,
            "is_quartet": is_q,
            "is_bridge": (rel_id in bridge_ids),
            "angle": round(float(node_angles[i]), 4),
            "circos_pos": [round(float(coords_circos[i, 0]), 2), round(float(coords_circos[i, 1]), 2), round(float(coords_circos[i, 2]), 2)],
            "brain_pos": [round(float(coords_3d_norm[i, 0]), 2), round(float(coords_3d_norm[i, 1]), 2), round(float(coords_3d_norm[i, 2]), 2)]
        })

    # Build JSON edges
    edges_payload = []
    for eid, (u, v) in enumerate(sorted_edges):
        is_h = (eid in harmonic_eids)
        u_gen = relations[u][1]
        v_gen = relations[v][1]
        is_cross_gen = (u_gen != v_gen)
        is_bottleneck_edge = (u in top_10_rw_nodes or v in top_10_rw_nodes)
        da = abs(node_angles[u] - node_angles[v])
        if da > math.pi:
            da = 2.0 * math.pi - da
        span_factor = da / math.pi
        edges_payload.append({
            "id": eid,
            "u": u,
            "v": v,
            "is_harmonic": is_h,
            "is_cross_gen": is_cross_gen,
            "is_bottleneck": is_bottleneck_edge,
            "u_gen": u_gen,
            "v_gen": v_gen,
            "span": round(float(span_factor), 3)
        })

    with open(THREE_JS_SRC, 'r', encoding='utf-8') as f:
        three_js_code = f.read()
    with open(ORBIT_JS_SRC, 'r', encoding='utf-8') as f:
        orbit_js_code = f.read()

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Information Boltzmann · 2401 Causal Reservoir Connectome (Bloom/Halation/Grain FX)</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; user-select: none; }}
  body {{
    background: #02050e;
    color: #e2e8f0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", "PingFang SC", sans-serif;
    overflow: hidden;
    height: 100vh;
  }}
  #canvas-container {{
    width: 100vw;
    height: 100vh;
    position: absolute;
    top: 0; left: 0;
  }}

  /* Minimalist Atmospheric HUD */
  .hud {{
    transition: opacity 0.5s ease, transform 0.5s ease;
    pointer-events: auto;
  }}
  .hud.zen-hidden {{
    opacity: 0 !important;
    pointer-events: none !important;
    transform: translateY(6px);
  }}

  /* Top Minimal Brand */
  .brand {{
    position: absolute;
    top: 24px; left: 28px;
    z-index: 10;
    pointer-events: none;
  }}
  .brand-title {{
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 2.5px;
    color: #f1f5f9;
    text-shadow: 0 0 16px rgba(56, 189, 248, 0.35);
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .brand-title::before {{
    content: "";
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #38bdf8;
    box-shadow: 0 0 8px #38bdf8;
  }}
  .brand-sub {{
    font-size: 10px;
    letter-spacing: 1.2px;
    color: #64748b;
    margin-top: 4px;
    text-transform: uppercase;
  }}

  /* Zen Mode Floating Button */
  .zen-toggle {{
    position: absolute;
    top: 24px;
    right: 28px;
    z-index: 20;
    background: rgba(15, 23, 42, 0.55);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(56, 189, 248, 0.25);
    color: #94a3b8;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.5px;
    cursor: pointer;
    transition: all 0.25s;
    display: flex;
    align-items: center;
    gap: 6px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
  }}
  .zen-toggle:hover {{
    background: rgba(56, 189, 248, 0.18);
    color: #38bdf8;
    border-color: #38bdf8;
    box-shadow: 0 0 16px rgba(56, 189, 248, 0.3);
  }}

  /* Bottom Controls Floating Glass Dock */
  .bottom-dock {{
    position: absolute;
    bottom: 24px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 10;
    display: flex;
    align-items: center;
    gap: 10px;
    background: rgba(6, 12, 26, 0.76);
    backdrop-filter: blur(24px);
    border: 1px solid rgba(56, 189, 248, 0.18);
    padding: 6px 14px;
    border-radius: 30px;
    box-shadow: 0 16px 40px rgba(0,0,0,0.7);
  }}
  .dock-group {{
    display: flex;
    align-items: center;
    gap: 5px;
  }}
  .dock-btn {{
    background: transparent;
    border: 1px solid transparent;
    color: #94a3b8;
    padding: 4px 9px;
    border-radius: 16px;
    font-size: 11px;
    cursor: pointer;
    transition: all 0.2s;
  }}
  .dock-btn:hover {{
    color: #f1f5f9;
    background: rgba(255, 255, 255, 0.06);
  }}
  .dock-btn.active {{
    background: rgba(56, 189, 248, 0.2);
    border-color: rgba(56, 189, 248, 0.4);
    color: #38bdf8;
    font-weight: 600;
  }}
  .dock-divider {{
    width: 1px;
    height: 14px;
    background: rgba(255, 255, 255, 0.12);
  }}
  .dock-slider {{
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 0 2px;
  }}
  .dock-slider span {{
    font-size: 10px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}
  .slider-slim {{
    -webkit-appearance: none;
    width: 55px;
    height: 3px;
    border-radius: 2px;
    background: rgba(255, 255, 255, 0.14);
    outline: none;
  }}
  .slider-slim::-webkit-slider-thumb {{
    -webkit-appearance: none;
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #38bdf8;
    cursor: pointer;
    box-shadow: 0 0 6px #38bdf8;
  }}

  /* Bottom Left Streamlined Legend Pill */
  .legend-pill {{
    position: absolute;
    bottom: 24px; left: 28px;
    z-index: 10;
    display: flex;
    align-items: center;
    gap: 12px;
    background: rgba(6, 12, 26, 0.65);
    backdrop-filter: blur(18px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 10.5px;
    color: #94a3b8;
  }}
  .leg-item {{ display: flex; align-items: center; gap: 5px; }}
  .leg-dot {{ width: 5px; height: 5px; border-radius: 50%; }}

  /* Floating Hover Tooltip */
  .node-tooltip {{
    position: absolute;
    bottom: 24px; right: 28px;
    z-index: 15;
    background: rgba(6, 12, 26, 0.88);
    backdrop-filter: blur(24px);
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: 12px;
    padding: 12px 14px;
    width: 250px;
    box-shadow: 0 16px 40px rgba(0,0,0,0.8);
    pointer-events: none;
    opacity: 0;
    transform: translateY(6px);
    transition: opacity 0.25s, transform 0.25s;
  }}
  .node-tooltip.visible {{
    opacity: 1;
    transform: translateY(0);
  }}
  .tip-title {{
    font-size: 13px;
    font-weight: 700;
    color: #38bdf8;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 4px;
  }}
  .tip-tag {{
    font-size: 9px;
    padding: 1px 5px;
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.08);
    color: #cbd5e1;
  }}
  .tip-stat {{
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    margin-bottom: 4px;
    color: #94a3b8;
  }}
  .tip-val {{ font-weight: 600; color: #f1f5f9; }}
</style>
</head>
<body>

<div id="canvas-container"></div>

<!-- Top Minimal Brand -->
<div class="brand hud" id="hud-brand">
  <div class="brand-title">CGB · CAUSAL RESERVOIR 2401</div>
  <div class="brand-sub">TypeScript · Three.js · Bloom / Halation / Grain Shaders</div>
</div>

<!-- Zen Mode Toggle Button -->
<button class="zen-toggle" id="zen-btn">
  <span id="zen-text">✦ 纯净背景</span>
</button>

<!-- Bottom Streamlined Legend -->
<div class="legend-pill hud" id="hud-legend">
  <div class="leg-item"><div class="leg-dot" style="background:#22d3ee; box-shadow:0 0 5px #22d3ee;"></div>轨1 代次</div>
  <div class="leg-item"><div class="leg-dot" style="background:#0284c7; box-shadow:0 0 5px #0284c7;"></div>轨2 度数</div>
  <div class="leg-item"><div class="leg-dot" style="background:#c084fc; box-shadow:0 0 5px #c084fc;"></div>轨3 漫游中介</div>
  <div class="leg-item"><div class="leg-dot" style="background:#10b981; box-shadow:0 0 5px #10b981;"></div>轨4 经历</div>
  <div class="leg-item"><div class="leg-dot" style="background:#fbbf24; box-shadow:0 0 5px #fbbf24;"></div>调和环流</div>
</div>

<!-- Floating Hover Tooltip -->
<div class="node-tooltip" id="node-tooltip">
  <div class="tip-title">
    <span id="tip-id">Relation #192</span>
    <span class="tip-tag" id="tip-sector">巨组件</span>
  </div>
  <div class="tip-stat">
    <span>漫游中介 (RW):</span>
    <span class="tip-val" id="tip-rw" style="color:#c084fc;">0.12264 (#1/280)</span>
  </div>
  <div class="tip-stat">
    <span>拓扑连边 (Deg):</span>
    <span class="tip-val" id="tip-deg" style="color:#38bdf8;">16 条</span>
  </div>
  <div class="tip-stat">
    <span>代次 / 经历:</span>
    <span class="tip-val" id="tip-gen">Gen 4 · 8 经历</span>
  </div>
  <div class="tip-stat" id="tip-status-row">
    <span>同调状态:</span>
    <span class="tip-val" id="tip-harmonic" style="color:#fbbf24;">主干道</span>
  </div>
</div>

<!-- Bottom Minimal Dock with Real-Time Bloom, Halation, Grain & Exposure Controls -->
<div class="bottom-dock hud" id="hud-dock">
  <div class="dock-group">
    <button class="dock-btn active" id="btn-tour">巡游</button>
    <button class="dock-btn" id="btn-top">俯视</button>
    <button class="dock-btn" id="btn-front">前缘</button>
  </div>
  <div class="dock-divider"></div>
  <div class="dock-group">
    <button class="dock-btn active" id="btn-all">全部</button>
    <button class="dock-btn" id="btn-bottleneck">咽喉</button>
    <button class="dock-btn" id="btn-harmonic">同调洞</button>
  </div>
  <div class="dock-divider"></div>
  <div class="dock-slider">
    <span>3D</span>
    <input type="range" min="0" max="100" value="0" class="slider-slim" id="morph-slider" title="平滑变形: Circos 环 <-> 3D 解剖网格">
  </div>
  <div class="dock-divider"></div>
  <div class="dock-slider">
    <span>曝光</span>
    <input type="range" min="15" max="150" value="65" class="slider-slim" id="exposure-slider" title="调节曝光度">
  </div>
  <div class="dock-slider">
    <span>辉光</span>
    <input type="range" min="0" max="200" value="85" class="slider-slim" id="bloom-slider" title="Bloom 辉光强度">
  </div>
  <div class="dock-slider">
    <span>光渗</span>
    <input type="range" min="0" max="200" value="65" class="slider-slim" id="halation-slider" title="CineStill 800T 红晕光渗 (Halation)">
  </div>
  <div class="dock-slider">
    <span>颗粒</span>
    <input type="range" min="0" max="100" value="35" class="slider-slim" id="grain-slider" title="胶片有机微颗粒 (Film Grain)">
  </div>
</div>

<script>
{three_js_code}
</script>
<script>
{orbit_js_code}
</script>

<script>
const NODES_DATA = {json.dumps(nodes_payload)};
const EDGES_DATA = {json.dumps(edges_payload)};
const SECTORS_DATA = {json.dumps(sectors)};

// Scene, Camera, Renderer
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x02050e, 0.0035);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.5, 2000);
camera.position.set(0, 66.0, 86.0);

const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true, powerPreference: 'high-performance' }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
container.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.maxDistance = 350;
controls.minDistance = 15;
controls.target.set(0, 8, 0);

controls.autoRotate = true;
controls.autoRotateSpeed = 0.5;

// Root World Group (for Zero-G Floating decoupled from OrbitControls)
const worldGroup = new THREE.Group();
scene.add(worldGroup);

// Calibrated Scene Lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.32);
scene.add(ambientLight);

const pointLight1 = new THREE.PointLight(0x38bdf8, 1.2, 300);
pointLight1.position.set(30, 70, 40);
scene.add(pointLight1);

const pointLight2 = new THREE.PointLight(0xa855f7, 0.9, 300);
pointLight2.position.set(-40, 50, -40);
scene.add(pointLight2);

const pointLightCenter = new THREE.PointLight(0xfbbf24, 0.55, 150);
pointLightCenter.position.set(0, 15, 0);
scene.add(pointLightCenter);

// Pure Chromatic Radial Glow Texture
function hexToRgba(hex, alpha) {{
  let c = hex.replace('#', '');
  if (c.length === 3) c = c.split('').map(x => x + x).join('');
  const num = parseInt(c, 16);
  const r = (num >> 16) & 255;
  const g = (num >> 8) & 255;
  const b = num & 255;
  return `rgba(${{r}}, ${{g}}, ${{b}}, ${{alpha}})`;
}}

function createGlowTexture(colorHex, size = 128) {{
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  const grad = ctx.createRadialGradient(size/2, size/2, 0, size/2, size/2, size/2);
  grad.addColorStop(0, hexToRgba(colorHex, 0.9));
  grad.addColorStop(0.25, hexToRgba(colorHex, 0.65));
  grad.addColorStop(0.6, hexToRgba(colorHex, 0.12));
  grad.addColorStop(1.0, hexToRgba(colorHex, 0.0));
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(canvas);
}}

const glowTexPurple = createGlowTexture('#c084fc');
const glowTexGold = createGlowTexture('#fbbf24');
const glowTexRose = createGlowTexture('#f43f5e');
const glowTexWhite = createGlowTexture('#ffffff');

// 1. Holographic Floor Reticles & Sector Arcs
const floorGroup = new THREE.Group();
worldGroup.add(floorGroup);

[25, 40, 50.0, 52.5, 55.0, 57.5, 60.0, 64.0].forEach(r => {{
  const pts = [];
  for (let a = 0; a <= Math.PI * 2.01; a += Math.PI / 64) {{
    pts.push(new THREE.Vector3(r * Math.cos(a), 0, r * Math.sin(a)));
  }}
  const ringGeo = new THREE.BufferGeometry().setFromPoints(pts);
  const ringMat = new THREE.LineBasicMaterial({{
    color: (r === 50.0) ? 0x38bdf8 : 0x1e293b,
    transparent: true,
    opacity: (r === 50.0) ? 0.30 : 0.15
  }});
  floorGroup.add(new THREE.Line(ringGeo, ringMat));
}});

for (let deg = 0; deg < 360; deg += 45) {{
  const rad = deg * Math.PI / 180;
  const rayGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(45 * Math.cos(rad), 0, 45 * Math.sin(rad)),
    new THREE.Vector3(65 * Math.cos(rad), 0, 65 * Math.sin(rad))
  ]);
  const rayMat = new THREE.LineBasicMaterial({{ color: 0x1e293b, transparent: true, opacity: 0.18 }});
  floorGroup.add(new THREE.Line(rayGeo, rayMat));
}}

const sectorsGroup = new THREE.Group();
worldGroup.add(sectorsGroup);

SECTORS_DATA.forEach(s => {{
  const arcShape = new THREE.Shape();
  const innerR = 48.5;
  const outerR = 50.5;
  arcShape.absarc(0, 0, outerR, s.angle_start, s.angle_end, false);
  arcShape.absarc(0, 0, innerR, s.angle_end, s.angle_start, true);
  arcShape.closePath();

  const geom = new THREE.ShapeGeometry(arcShape, 48);
  const mat = new THREE.MeshBasicMaterial({{
    color: new THREE.Color(s.color),
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.20
  }});
  const mesh = new THREE.Mesh(geom, mat);
  mesh.rotation.x = Math.PI / 2;
  mesh.position.y = 0.02;
  sectorsGroup.add(mesh);
}});

// 2. Bioluminescent Crystal Nodes
const nodesGroup = new THREE.Group();
worldGroup.add(nodesGroup);

const aurasGroup = new THREE.Group();
worldGroup.add(aurasGroup);

const nodeObjects = [];
const auraObjects = [];

const baseNodeGeom = new THREE.SphereGeometry(0.70, 18, 18);
const harmonicNodeGeom = new THREE.SphereGeometry(1.15, 20, 20);
const topRwNodeGeom = new THREE.SphereGeometry(1.35, 20, 20);

NODES_DATA.forEach((d, idx) => {{
  const baseCol = new THREE.Color(d.gen_color);
  let color = baseCol;
  let geom = baseNodeGeom;
  let emissiveIntensity = 0.28;
  let auraTex = null;
  let auraScale = 3.2;

  if (d.is_top_rw) {{
    color = new THREE.Color(0xc084fc);
    geom = topRwNodeGeom;
    emissiveIntensity = 0.65;
    auraTex = glowTexPurple;
    auraScale = 4.8;
  }} else if (d.is_harmonic) {{
    color = new THREE.Color(0xfbbf24);
    geom = harmonicNodeGeom;
    emissiveIntensity = 0.55;
    auraTex = glowTexGold;
    auraScale = 4.2;
  }} else if (d.is_quartet) {{
    color = new THREE.Color(0xf43f5e);
    emissiveIntensity = 0.45;
    auraTex = glowTexRose;
    auraScale = 3.4;
  }}

  const mat = new THREE.MeshStandardMaterial({{
    color: color,
    emissive: color,
    emissiveIntensity: emissiveIntensity,
    roughness: 0.15,
    metalness: 0.20,
    transparent: true,
    opacity: 0.88
  }});

  const mesh = new THREE.Mesh(geom, mat);
  mesh.position.set(d.circos_pos[0], d.circos_pos[1], d.circos_pos[2]);
  mesh.userData = {{ nodeIndex: idx, data: d }};
  nodesGroup.add(mesh);
  nodeObjects.push(mesh);

  if (auraTex) {{
    const spriteMat = new THREE.SpriteMaterial({{
      map: auraTex,
      blending: THREE.AdditiveBlending,
      transparent: true,
      opacity: 0.45,
      depthWrite: false
    }});
    const sprite = new THREE.Sprite(spriteMat);
    sprite.scale.set(auraScale, auraScale, 1);
    sprite.position.copy(mesh.position);
    aurasGroup.add(sprite);
    auraObjects.push({{ sprite: sprite, baseScale: auraScale, nodeIdx: idx }});
  }}
}});

// 3. Optical Light-Pipes (Sleek Micro-Rods)
const tracksGroup = new THREE.Group();
worldGroup.add(tracksGroup);

const pipeGeom = new THREE.CylinderGeometry(0.16, 0.20, 1.0, 14);
pipeGeom.translate(0, 0.5, 0);
const capGeom = new THREE.SphereGeometry(0.24, 12, 12);

NODES_DATA.forEach(d => {{
  // Track 1 (R=52.5): Gen
  const genH = d.gen * 1.35;
  const t1Mat = new THREE.MeshStandardMaterial({{
    color: new THREE.Color(d.gen_color),
    emissive: new THREE.Color(d.gen_color),
    emissiveIntensity: 0.32,
    roughness: 0.2,
    transparent: true,
    opacity: 0.72
  }});
  const p1 = new THREE.Mesh(pipeGeom, t1Mat);
  p1.position.set(52.5 * Math.cos(d.angle), 0, 52.5 * Math.sin(d.angle));
  p1.scale.set(1, genH, 1);
  tracksGroup.add(p1);

  const c1 = new THREE.Mesh(capGeom, t1Mat);
  c1.position.set(52.5 * Math.cos(d.angle), genH, 52.5 * Math.sin(d.angle));
  tracksGroup.add(c1);

  // Track 2 (R=55.0): Degree
  const degH = Math.max(0.3, (d.degree / 22.0) * 7.5);
  const t2Mat = new THREE.MeshStandardMaterial({{
    color: new THREE.Color(0x0284c7),
    emissive: new THREE.Color(0x38bdf8),
    emissiveIntensity: 0.35,
    roughness: 0.2,
    transparent: true,
    opacity: 0.72
  }});
  const p2 = new THREE.Mesh(pipeGeom, t2Mat);
  p2.position.set(55.0 * Math.cos(d.angle), 0, 55.0 * Math.sin(d.angle));
  p2.scale.set(1, degH, 1);
  tracksGroup.add(p2);

  const c2 = new THREE.Mesh(capGeom, t2Mat);
  c2.position.set(55.0 * Math.cos(d.angle), degH, 55.0 * Math.sin(d.angle));
  tracksGroup.add(c2);

  // Track 3 (R=57.5): RW
  const rwH = Math.max(0.2, (d.rw_betweenness / 0.125) * 12.0);
  const t3Color = d.is_top_rw ? 0xd946ef : 0xa855f7;
  const t3Mat = new THREE.MeshStandardMaterial({{
    color: new THREE.Color(t3Color),
    emissive: new THREE.Color(t3Color),
    emissiveIntensity: d.is_top_rw ? 0.65 : 0.38,
    roughness: 0.2,
    transparent: true,
    opacity: d.is_top_rw ? 0.90 : 0.72
  }});
  const p3 = new THREE.Mesh(pipeGeom, t3Mat);
  p3.position.set(57.5 * Math.cos(d.angle), 0, 57.5 * Math.sin(d.angle));
  p3.scale.set(d.is_top_rw ? 1.4 : 1.0, rwH, d.is_top_rw ? 1.4 : 1.0);
  tracksGroup.add(p3);

  const c3 = new THREE.Mesh(capGeom, t3Mat);
  c3.position.set(57.5 * Math.cos(d.angle), rwH, 57.5 * Math.sin(d.angle));
  tracksGroup.add(c3);

  // Track 4 (R=60.0): Supports
  const supH = Math.max(0.25, (d.supports_count / 8.0) * 4.5);
  const t4Mat = new THREE.MeshStandardMaterial({{
    color: new THREE.Color(0x10b981),
    emissive: new THREE.Color(0x34d399),
    emissiveIntensity: 0.35,
    roughness: 0.2,
    transparent: true,
    opacity: 0.72
  }});
  const p4 = new THREE.Mesh(pipeGeom, t4Mat);
  p4.position.set(60.0 * Math.cos(d.angle), 0, 60.0 * Math.sin(d.angle));
  p4.scale.set(1, supH, 1);
  tracksGroup.add(p4);

  const c4 = new THREE.Mesh(capGeom, t4Mat);
  c4.position.set(60.0 * Math.cos(d.angle), supH, 60.0 * Math.sin(d.angle));
  tracksGroup.add(c4);
}});

// 4. Fiber-Optic Arcing Chords with Additive Blending
const chordsGroup = new THREE.Group();
worldGroup.add(chordsGroup);

const chordCurves = [];
const chordMeshes = [];

EDGES_DATA.forEach(e => {{
  const uNode = NODES_DATA[e.u];
  const vNode = NODES_DATA[e.v];
  const maxArch = 6.0 + e.span * 38.0;
  
  const p0 = new THREE.Vector3(uNode.circos_pos[0], uNode.circos_pos[1], uNode.circos_pos[2]);
  const p3 = new THREE.Vector3(vNode.circos_pos[0], vNode.circos_pos[1], vNode.circos_pos[2]);
  
  const centerPull = 0.52 - e.span * 0.32;
  const midX = (p0.x + p3.x) * centerPull;
  const midZ = (p0.z + p3.z) * centerPull;
  const pMid = new THREE.Vector3(midX, maxArch, midZ);
  
  const curve = new THREE.QuadraticBezierCurve3(p0, pMid, p3);
  chordCurves.push({{ curve: curve, edge: e, p0: p0, pMid: pMid, p3: p3 }});

  const points = curve.getPoints(28);
  const geom = new THREE.BufferGeometry().setFromPoints(points);

  let col = new THREE.Color(0x0284c7);
  let opacity = 0.14;
  if (e.is_harmonic) {{
    col = new THREE.Color(0xfbbf24);
    opacity = 0.65;
  }} else if (e.is_bottleneck) {{
    col = new THREE.Color(0xc084fc);
    opacity = 0.32;
  }}

  const lineMat = new THREE.LineBasicMaterial({{
    color: col,
    transparent: true,
    opacity: opacity,
    blending: THREE.AdditiveBlending,
    depthWrite: false
  }});

  const line = new THREE.Line(geom, lineMat);
  line.userData = {{ edge: e, baseCol: col, baseOpacity: opacity }};
  chordsGroup.add(line);
  chordMeshes.push(line);
}});

// 5. Dynamic Photon Energy Pulses (Flowing Light)
const PHOTON_COUNT = 160;
const photonGeo = new THREE.BufferGeometry();
const photonPositions = new Float32Array(PHOTON_COUNT * 3);
const photonColors = new Float32Array(PHOTON_COUNT * 3);

const activeCurveIndices = [];
chordCurves.forEach((c, idx) => {{
  if (c.edge.is_harmonic) {{
    for(let k=0; k<6; k++) activeCurveIndices.push(idx);
  }} else if (c.edge.is_bottleneck) {{
    for(let k=0; k<2; k++) activeCurveIndices.push(idx);
  }} else if (idx % 7 === 0) {{
    activeCurveIndices.push(idx);
  }}
}});

const photons = [];
for (let p = 0; p < PHOTON_COUNT; p++) {{
  const curveIdx = activeCurveIndices[p % activeCurveIndices.length];
  const t = Math.random();
  const speed = 0.003 + Math.random() * 0.005;
  const isHarmonic = chordCurves[curveIdx].edge.is_harmonic;
  const isBottleneck = chordCurves[curveIdx].edge.is_bottleneck;
  
  let col = isHarmonic ? new THREE.Color(0xfbbf24) : (isBottleneck ? new THREE.Color(0xc084fc) : new THREE.Color(0x38bdf8));

  photons.push({{
    curveIdx: curveIdx,
    t: t,
    speed: speed,
    color: col
  }});

  photonColors[p * 3] = col.r;
  photonColors[p * 3 + 1] = col.g;
  photonColors[p * 3 + 2] = col.b;
}}

photonGeo.setAttribute('position', new THREE.BufferAttribute(photonPositions, 3));
photonGeo.setAttribute('color', new THREE.BufferAttribute(photonColors, 3));

const photonMat = new THREE.PointsMaterial({{
  size: 2.3,
  map: glowTexWhite,
  vertexColors: true,
  blending: THREE.AdditiveBlending,
  transparent: true,
  opacity: 0.75,
  depthWrite: false
}});

const photonSystem = new THREE.Points(photonGeo, photonMat);
worldGroup.add(photonSystem);

// 6. Smooth Morphing
let morphT = 0.0;
function updateMorph(t) {{
  morphT = t;
  NODES_DATA.forEach((d, idx) => {{
    const mesh = nodeObjects[idx];
    mesh.position.x = (1 - t) * d.circos_pos[0] + t * d.brain_pos[0];
    mesh.position.y = (1 - t) * d.circos_pos[1] + t * d.brain_pos[1];
    mesh.position.z = (1 - t) * d.circos_pos[2] + t * d.brain_pos[2];
  }});

  auraObjects.forEach(a => {{
    const pos = nodeObjects[a.nodeIdx].position;
    a.sprite.position.copy(pos);
  }});

  sectorsGroup.position.y = (1 - t) * 0.02 - t * 40.0;
  tracksGroup.position.y = (1 - t) * 0.0 - t * 40.0;
  floorGroup.position.y = (1 - t) * 0.0 - t * 40.0;
  
  sectorsGroup.children.forEach(c => c.material.opacity = (1 - t) * 0.20);
  tracksGroup.children.forEach(c => {{
    if (c.material) c.material.opacity = (1 - t) * 0.72;
  }});

  chordCurves.forEach((item, idx) => {{
    const e = item.edge;
    const uPos = nodeObjects[e.u].position;
    const vPos = nodeObjects[e.v].position;

    const circosMid = item.pMid;
    const brainMid = new THREE.Vector3().addVectors(uPos, vPos).multiplyScalar(0.5);
    const currMid = new THREE.Vector3().lerpVectors(circosMid, brainMid, t);

    const dynamicCurve = new THREE.QuadraticBezierCurve3(uPos, currMid, vPos);
    item.dynamicCurve = dynamicCurve;
    chordMeshes[idx].geometry.setFromPoints(dynamicCurve.getPoints(24));
  }});
}}

document.getElementById('morph-slider').addEventListener('input', e => {{
  updateMorph(parseFloat(e.target.value) / 100.0);
}});

// 7. GPU Post-Processing Pipeline (Bloom + CineStill 800T Halation + Film Grain + ACES)
const isWebGL2 = renderer.capabilities.isWebGL2;
const targetType = isWebGL2 ? THREE.HalfFloatType : THREE.UnsignedByteType;
const targetOptions = {{
  minFilter: THREE.LinearFilter,
  magFilter: THREE.LinearFilter,
  format: THREE.RGBAFormat,
  type: targetType
}};

let w = window.innerWidth;
let h = window.innerHeight;
let downW = Math.max(1, Math.floor(w / 2));
let downH = Math.max(1, Math.floor(h / 2));

const sceneTarget = new THREE.WebGLRenderTarget(w, h, targetOptions);
const brightTarget = new THREE.WebGLRenderTarget(downW, downH, targetOptions);
const blurTargetA = new THREE.WebGLRenderTarget(downW, downH, targetOptions);
const blurTargetB = new THREE.WebGLRenderTarget(downW, downH, targetOptions);

const postCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
const quadGeo = new THREE.PlaneGeometry(2, 2);

// Shaders
const brightPassMat = new THREE.ShaderMaterial({{
  uniforms: {{
    tDiffuse: {{ value: null }},
    uThreshold: {{ value: 0.32 }},
    uKnee: {{ value: 0.15 }}
  }},
  vertexShader: `
    varying vec2 vUv;
    void main() {{
      vUv = uv;
      gl_Position = vec4(position.xy, 0.0, 1.0);
    }}
  `,
  fragmentShader: `
    uniform sampler2D tDiffuse;
    uniform float uThreshold;
    uniform float uKnee;
    varying vec2 vUv;
    void main() {{
      vec4 tex = texture2D(tDiffuse, vUv);
      float luma = dot(tex.rgb, vec3(0.2126, 0.7152, 0.0722));
      float soft = clamp((luma - uThreshold + uKnee) / (2.0 * uKnee + 1e-4), 0.0, 1.0);
      float weight = max(soft * soft, step(uThreshold, luma));
      gl_FragColor = vec4(tex.rgb * weight, 1.0);
    }}
  `,
  depthTest: false,
  depthWrite: false
}});

const blurMat = new THREE.ShaderMaterial({{
  uniforms: {{
    tDiffuse: {{ value: null }},
    uResolution: {{ value: new THREE.Vector2(downW, downH) }},
    uOffset: {{ value: 1.0 }}
  }},
  vertexShader: `
    varying vec2 vUv;
    void main() {{
      vUv = uv;
      gl_Position = vec4(position.xy, 0.0, 1.0);
    }}
  `,
  fragmentShader: `
    uniform sampler2D tDiffuse;
    uniform vec2 uResolution;
    uniform float uOffset;
    varying vec2 vUv;
    void main() {{
      vec2 d = (1.0 / uResolution) * uOffset;
      vec3 sum = texture2D(tDiffuse, vUv).rgb * 0.25;
      sum += texture2D(tDiffuse, vUv + vec2(-d.x, -d.y)).rgb * 0.125;
      sum += texture2D(tDiffuse, vUv + vec2( d.x, -d.y)).rgb * 0.125;
      sum += texture2D(tDiffuse, vUv + vec2(-d.x,  d.y)).rgb * 0.125;
      sum += texture2D(tDiffuse, vUv + vec2( d.x,  d.y)).rgb * 0.125;
      sum += texture2D(tDiffuse, vUv + vec2(-d.x * 2.0, 0.0)).rgb * 0.0625;
      sum += texture2D(tDiffuse, vUv + vec2( d.x * 2.0, 0.0)).rgb * 0.0625;
      sum += texture2D(tDiffuse, vUv + vec2(0.0, -d.y * 2.0)).rgb * 0.0625;
      sum += texture2D(tDiffuse, vUv + vec2(0.0,  d.y * 2.0)).rgb * 0.0625;
      gl_FragColor = vec4(sum, 1.0);
    }}
  `,
  depthTest: false,
  depthWrite: false
}});

const compositeMat = new THREE.ShaderMaterial({{
  uniforms: {{
    tDiffuse: {{ value: null }},
    tBloom: {{ value: null }},
    uResolution: {{ value: new THREE.Vector2(w, h) }},
    uTime: {{ value: 0.0 }},
    uExposure: {{ value: 0.65 }},
    uBloomIntensity: {{ value: 0.85 }},
    uHalationIntensity: {{ value: 0.65 }},
    uHalationSpread: {{ value: 0.005 }},
    uGrainIntensity: {{ value: 0.035 }}
  }},
  vertexShader: `
    varying vec2 vUv;
    void main() {{
      vUv = uv;
      gl_Position = vec4(position.xy, 0.0, 1.0);
    }}
  `,
  fragmentShader: `
    uniform sampler2D tDiffuse;
    uniform sampler2D tBloom;
    uniform vec2 uResolution;
    uniform float uTime;
    uniform float uExposure;
    uniform float uBloomIntensity;
    uniform float uHalationIntensity;
    uniform float uHalationSpread;
    uniform float uGrainIntensity;
    varying vec2 vUv;

    // ACES Filmic Tone Mapping Curve
    vec3 ACESFilm(vec3 x) {{
      float a = 2.51;
      float b = 0.03;
      float c = 2.43;
      float d = 0.59;
      float e = 0.14;
      return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
    }}

    // High-speed pseudo-random hash
    float hash21(vec2 p) {{
      p = fract(p * vec2(234.34, 435.345));
      p += dot(p, p + 34.23);
      return fract(p.x * p.y);
    }}

    // Organic triangular distribution film grain
    float filmGrain(vec2 uv, float time) {{
      vec2 seed = uv * uResolution * 0.85 + vec2(time * 45.0, time * 85.0);
      float h1 = hash21(seed);
      float h2 = hash21(seed + vec2(1.7, 4.3));
      return (h1 + h2) * 0.5 - 0.5;
    }}

    void main() {{
      vec4 baseColor = texture2D(tDiffuse, vUv);
      vec3 sceneColor = baseColor.rgb;

      // 1. Bloom Sampling
      vec3 bloom = texture2D(tBloom, vUv).rgb * uBloomIntensity;

      // 2. Cinematic Halation (CineStill 800T warm red-amber scatter)
      vec2 hDir = (vUv - vec2(0.5)) * uHalationSpread;
      vec3 hSample = texture2D(tBloom, vUv - hDir * 1.5).rgb * 0.5 +
                     texture2D(tBloom, vUv + hDir * 1.5).rgb * 0.5;
      vec3 halationTint = vec3(1.35, 0.40, 0.10);
      vec3 halation = pow(hSample, vec3(1.15)) * halationTint * uHalationIntensity;

      // Composite Light Energy
      vec3 composite = sceneColor + bloom + halation;

      // 3. Sensor Exposure Scaling
      composite *= (uExposure / 0.65);

      // 4. Physical Film Grain (Luminance Midtone Weighted)
      float luma = dot(composite, vec3(0.2126, 0.7152, 0.0722));
      float grainWeight = 1.0 - 3.8 * pow(clamp(luma, 0.0, 1.0) - 0.5, 2.0);
      grainWeight = clamp(grainWeight, 0.20, 1.0);

      float grain = filmGrain(vUv, uTime) * uGrainIntensity * grainWeight;
      composite += vec3(grain);

      // 5. ACES Filmic Tone Mapping & Gamma Correction
      vec3 graded = ACESFilm(composite);
      graded = pow(graded, vec3(1.0 / 2.2)); // Linear to sRGB gamma

      gl_FragColor = vec4(graded, baseColor.a);
    }}
  `,
  depthTest: false,
  depthWrite: false
}});

const quadMesh = new THREE.Mesh(quadGeo, compositeMat);

// Real-Time Controls for Exposure, Bloom, Halation, Grain
document.getElementById('exposure-slider').addEventListener('input', e => {{
  compositeMat.uniforms.uExposure.value = parseFloat(e.target.value) / 100.0;
}});
document.getElementById('bloom-slider').addEventListener('input', e => {{
  compositeMat.uniforms.uBloomIntensity.value = parseFloat(e.target.value) / 100.0;
}});
document.getElementById('halation-slider').addEventListener('input', e => {{
  compositeMat.uniforms.uHalationIntensity.value = parseFloat(e.target.value) / 100.0;
}});
document.getElementById('grain-slider').addEventListener('input', e => {{
  compositeMat.uniforms.uGrainIntensity.value = parseFloat(e.target.value) / 1000.0;
}});

// 8. Intentional Raycast Hovering (Anti-Flicker & Anti-Strobe)
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2(-9999, -9999);
let hoveredNode = null;
let lastMouseMoveTime = 0;
const tooltip = document.getElementById('node-tooltip');

window.addEventListener('mousemove', e => {{
  mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
  lastMouseMoveTime = performance.now();
}});

function clearHoverState() {{
  mouse.set(-9999, -9999);
  if (hoveredNode !== null) {{
    hoveredNode = null;
    hideTooltip();
  }}
}}

window.addEventListener('mouseleave', clearHoverState);
window.addEventListener('blur', clearHoverState);
document.addEventListener('visibilitychange', () => {{
  if (document.hidden) clearHoverState();
}});

function checkHover() {{
  if (controls.autoRotate && (performance.now() - lastMouseMoveTime > 1200)) {{
    if (hoveredNode !== null) {{
      hoveredNode = null;
      hideTooltip();
    }}
    return;
  }}

  if (mouse.x < -1 || mouse.x > 1 || mouse.y < -1 || mouse.y > 1) return;

  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObjects(nodeObjects);

  if (intersects.length > 0) {{
    const hit = intersects[0].object;
    if (hoveredNode !== hit) {{
      hoveredNode = hit;
      showTooltip(hit.userData.nodeIndex);
    }}
  }} else {{
    if (hoveredNode !== null) {{
      hoveredNode = null;
      hideTooltip();
    }}
  }}
}}

function showTooltip(idx) {{
  const d = NODES_DATA[idx];
  document.getElementById('tip-id').innerText = `Relation #${{d.rel_id}}`;
  document.getElementById('tip-sector').innerText = d.sector_name;
  document.getElementById('tip-rw').innerText = `${{d.rw_betweenness}} (#${{d.rw_rank}}/280)`;
  document.getElementById('tip-deg').innerText = `${{d.degree}} 条连边`;
  document.getElementById('tip-gen').innerText = `Gen ${{d.gen}} · ${{d.supports_count}} 经历`;

  const hRow = document.getElementById('tip-status-row');
  if (d.is_harmonic) {{
    hRow.style.display = 'flex';
    document.getElementById('tip-harmonic').innerText = '18.32% 调和洞循环';
    document.getElementById('tip-harmonic').style.color = '#fbbf24';
  }} else if (d.is_top_rw) {{
    hRow.style.display = 'flex';
    document.getElementById('tip-harmonic').innerText = 'TOP 10 漫游咽喉';
    document.getElementById('tip-harmonic').style.color = '#c084fc';
  }} else if (d.is_quartet) {{
    hRow.style.display = 'flex';
    document.getElementById('tip-harmonic').innerText = '独立四重奏群';
    document.getElementById('tip-harmonic').style.color = '#f43f5e';
  }} else {{
    hRow.style.display = 'none';
  }}

  tooltip.classList.add('visible');

  chordMeshes.forEach(m => {{
    const e = m.userData.edge;
    if (e.u === idx || e.v === idx) {{
      m.material.opacity = 0.95;
      m.material.color.setHex(0x38bdf8);
    }} else {{
      m.material.opacity = 0.02;
    }}
  }});
}}

function hideTooltip() {{
  tooltip.classList.remove('visible');
  chordMeshes.forEach(m => {{
    const u = m.userData;
    m.material.opacity = u.baseOpacity;
    m.material.color.copy(u.baseCol);
  }});
}}

// 9. Zen Mode (纯净背景模式)
let isZen = false;
const zenBtn = document.getElementById('zen-btn');
const zenText = document.getElementById('zen-text');
const hudElements = document.querySelectorAll('.hud');

function toggleZen() {{
  isZen = !isZen;
  hudElements.forEach(el => el.classList.toggle('zen-hidden', isZen));
  zenText.innerText = isZen ? '✦ 显示控制' : '✦ 纯净背景';
  zenBtn.style.background = isZen ? 'rgba(56, 189, 248, 0.25)' : 'rgba(15, 23, 42, 0.55)';
  zenBtn.style.borderColor = isZen ? '#38bdf8' : 'rgba(56, 189, 248, 0.25)';
}}

zenBtn.onclick = toggleZen;
window.addEventListener('keydown', e => {{
  if (e.key === 'h' || e.key === 'H' || e.key === 'Escape') {{
    toggleZen();
  }}
}});

// 10. Minimal Dock Presets & Filters
document.getElementById('btn-tour').onclick = () => {{
  controls.autoRotate = !controls.autoRotate;
  document.getElementById('btn-tour').classList.toggle('active', controls.autoRotate);
}};
document.getElementById('btn-top').onclick = () => {{
  camera.position.set(0, 110, 0.1);
  controls.target.set(0, 0, 0);
  setActiveBtn('btn-top');
}};
document.getElementById('btn-front').onclick = () => {{
  camera.position.set(0, 15, 100);
  controls.target.set(0, 8, 0);
  setActiveBtn('btn-front');
}};

function setActiveBtn(id) {{
  ['btn-top', 'btn-front'].forEach(b => {{
    document.getElementById(b).classList.toggle('active', b === id);
  }});
}}

document.getElementById('btn-all').onclick = () => {{
  setActiveFilter('btn-all');
  chordMeshes.forEach(m => m.visible = true);
}};
document.getElementById('btn-bottleneck').onclick = () => {{
  setActiveFilter('btn-bottleneck');
  chordMeshes.forEach(m => {{
    const e = m.userData.edge;
    m.visible = NODES_DATA[e.u].is_top_rw || NODES_DATA[e.v].is_top_rw;
  }});
}};
document.getElementById('btn-harmonic').onclick = () => {{
  setActiveFilter('btn-harmonic');
  chordMeshes.forEach(m => m.visible = m.userData.edge.is_harmonic);
}};

function setActiveFilter(id) {{
  ['btn-all', 'btn-bottleneck', 'btn-harmonic'].forEach(b => {{
    document.getElementById(b).classList.toggle('active', b === id);
  }});
}}

// Robust Window Resize & RenderTarget resizing
window.addEventListener('resize', () => {{
  w = window.innerWidth;
  h = window.innerHeight;
  if (w > 10 && h > 10) {{
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);

    downW = Math.max(1, Math.floor(w / 2));
    downH = Math.max(1, Math.floor(h / 2));
    sceneTarget.setSize(w, h);
    brightTarget.setSize(downW, downH);
    blurTargetA.setSize(downW, downH);
    blurTargetB.setSize(downW, downH);

    blurMat.uniforms.uResolution.value.set(downW, downH);
    compositeMat.uniforms.uResolution.value.set(w, h);
  }}
}});

// 11. Animation Loop with 4-Pass GPU Post-Processing
let clockTime = 0;
function animate() {{
  requestAnimationFrame(animate);
  clockTime += 0.02;

  // Zero-G Gentle Floating
  if (controls.autoRotate) {{
    worldGroup.position.y = Math.sin(clockTime * 0.35) * 1.5;
  }}

  controls.update();
  checkHover();

  // Subtle breathing pulsation for Auras
  auraObjects.forEach((a, i) => {{
    const s = a.baseScale * (1.0 + 0.08 * Math.sin(clockTime * 2.5 + i));
    a.sprite.scale.set(s, s, 1);
  }});

  // Animate Photons along Curves
  const posArr = photonGeo.attributes.position.array;
  photons.forEach((p, idx) => {{
    p.t += p.speed;
    if (p.t > 1.0) p.t = 0.0;
    
    const curveItem = chordCurves[p.curveIdx];
    const curve = curveItem.dynamicCurve || curveItem.curve;
    const pt = curve.getPoint(p.t);
    posArr[idx * 3] = pt.x;
    posArr[idx * 3 + 1] = pt.y;
    posArr[idx * 3 + 2] = pt.z;
  }});
  photonGeo.attributes.position.needsUpdate = true;

  // ==========================================
  // GPU POST-PROCESSING PIPELINE
  // ==========================================
  
  // Pass 1: Render 3D Scene to HDR Scene Target
  renderer.setRenderTarget(sceneTarget);
  renderer.clear();
  renderer.render(scene, camera);

  // Pass 2: Bright-pass threshold into Downsampled Target
  brightPassMat.uniforms.tDiffuse.value = sceneTarget.texture;
  quadMesh.material = brightPassMat;
  renderer.setRenderTarget(brightTarget);
  renderer.clear();
  renderer.render(quadMesh, postCamera);

  // Pass 3A: Kawase Blur Iteration 1
  blurMat.uniforms.tDiffuse.value = brightTarget.texture;
  blurMat.uniforms.uOffset.value = 1.25;
  quadMesh.material = blurMat;
  renderer.setRenderTarget(blurTargetA);
  renderer.clear();
  renderer.render(quadMesh, postCamera);

  // Pass 3B: Kawase Blur Iteration 2 (Wider spread for deep halo)
  blurMat.uniforms.tDiffuse.value = blurTargetA.texture;
  blurMat.uniforms.uOffset.value = 2.6;
  quadMesh.material = blurMat;
  renderer.setRenderTarget(blurTargetB);
  renderer.clear();
  renderer.render(quadMesh, postCamera);

  // Pass 4: Final Screen Composite (Scene + Bloom + CineStill Halation + Film Grain + ACES)
  compositeMat.uniforms.tDiffuse.value = sceneTarget.texture;
  compositeMat.uniforms.tBloom.value = blurTargetB.texture;
  compositeMat.uniforms.uTime.value = clockTime;
  quadMesh.material = compositeMat;

  renderer.setRenderTarget(null);
  renderer.render(quadMesh, postCamera);
}}
animate();
</script>
</body>
</html>
"""

    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Written: {OUTPUT_HTML} ({OUTPUT_HTML.stat().st_size} bytes)")

    with open(ARTIFACT_HTML, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Artifact Written: {ARTIFACT_HTML} ({ARTIFACT_HTML.stat().st_size} bytes)")

if __name__ == '__main__':
    main()
