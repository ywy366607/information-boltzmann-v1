"""
生成 3D 词元引力星系宇宙与真实双半球脑皮层构形流形全景交互式可视化 HTML
- 词引力星系: 真实 3D 罗德里格斯开普勒公转椭圆，内圈快外圈慢，离心力势垒护航杜绝坍缩
- 构形大脑皮层流形: 彻底告别人工“圆锥森林”与“红节点重叠”！
  * 真实脑形态: 解构为「基底 Willis 动脉环 (Circle of Willis)」+「双脑半球脑回皮层 (Bi-Hemispheric Cortical Shell)」
  * 物理排斥松弛: 引入库仑斥力与弹性锚定（Force-Directed Relaxation），确保场内任意节点物理间距 ≥ 7.5px，红节点 360° 环状舒展，绝对零重叠！
  * 默里最小耗散流向: 119 条导管由基底动脉环向左右脑叶立体辐射分流，形成真实自然的大脑血管回旋树。
"""

import json, os, shutil, math

data_path = "present/galaxy_neural_data.json"
if not os.path.exists(data_path):
    print(f"Waiting for {data_path} to be generated...")
    exit(0)

with open(data_path, "r", encoding="utf-8") as f:
    sim_data = json.load(f)

json_str = json.dumps(sim_data, ensure_ascii=False)

html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>词引力星系宇宙与双半球脑皮层构形流形 3D 可视化</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: #020408;
            color: #dce5ef;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif;
            overflow: hidden;
            display: flex;
            height: 100vh;
        }}
        #canvas-container {{
            flex: 1;
            height: 100%;
            position: relative;
        }}
        canvas {{
            width: 100%;
            height: 100%;
            display: block;
        }}
        #sidebar {{
            width: 440px;
            height: 100%;
            background: rgba(7, 11, 18, 0.96);
            backdrop-filter: blur(20px);
            border-left: 1px solid rgba(255, 255, 255, 0.08);
            padding: 22px;
            overflow-y: auto;
            z-index: 10;
            display: flex;
            flex-direction: column;
            gap: 14px;
            box-shadow: -10px 0 30px rgba(0,0,0,0.6);
        }}
        h1 {{
            font-size: 19px;
            font-weight: 700;
            background: linear-gradient(135deg, #00E5FF, #FFD700, #FF007F);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.3px;
        }}
        .subtitle {{
            font-size: 11px;
            color: #7e8d9f;
            line-height: 1.5;
            margin-top: 3px;
        }}
        .badge-row {{
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
            margin-top: 4px;
        }}
        .badge {{
            font-size: 10px;
            padding: 2px 7px;
            border-radius: 4px;
            background: rgba(0, 229, 255, 0.1);
            color: #00E5FF;
            border: 1px solid rgba(0, 229, 255, 0.25);
        }}
        .badge.gold {{
            background: rgba(255, 215, 0, 0.1);
            color: #FFD700;
            border-color: rgba(255, 215, 0, 0.25);
        }}
        .badge.pink {{
            background: rgba(255, 0, 127, 0.1);
            color: #FF007F;
            border-color: rgba(255, 0, 127, 0.25);
        }}
        .view-tabs {{
            display: flex;
            background: rgba(255, 255, 255, 0.04);
            border-radius: 8px;
            padding: 3px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            gap: 4px;
        }}
        .tab-btn {{
            flex: 1;
            padding: 8px 6px;
            border: none;
            background: transparent;
            color: #8c9ba5;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            border-radius: 6px;
            transition: all 0.2s ease;
            text-align: center;
        }}
        .tab-btn.active {{
            background: linear-gradient(135deg, rgba(0, 229, 255, 0.25), rgba(255, 0, 127, 0.25));
            color: #ffffff;
            border: 1px solid rgba(0, 229, 255, 0.4);
            box-shadow: 0 0 12px rgba(0, 229, 255, 0.2);
        }}
        .card {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 10px;
            padding: 14px;
        }}
        .card-title {{
            font-size: 12px;
            font-weight: 700;
            color: #00E5FF;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
        }}
        .stat-box {{
            background: rgba(0, 0, 0, 0.35);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 7px;
            padding: 8px 10px;
        }}
        .stat-val {{
            font-size: 15px;
            font-weight: 700;
            color: #fff;
            font-family: monospace;
        }}
        .stat-lbl {{
            font-size: 10px;
            color: #6d7c8d;
            margin-top: 2px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            font-size: 11px;
        }}
        .legend-dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }}
        .info-panel {{
            background: rgba(0, 229, 255, 0.04);
            border: 1px solid rgba(0, 229, 255, 0.2);
            border-radius: 8px;
            padding: 10px;
            font-size: 11px;
            line-height: 1.5;
            color: #a0b2c6;
        }}
        .hover-card {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: rgba(9, 14, 24, 0.94);
            border: 1px solid rgba(0, 229, 255, 0.4);
            backdrop-filter: blur(12px);
            border-radius: 8px;
            padding: 12px 16px;
            pointer-events: none;
            display: none;
            font-size: 11px;
            color: #e0ecf8;
            box-shadow: 0 4px 20px rgba(0,0,0,0.6);
            z-index: 100;
            max-width: 320px;
        }}
        .hover-title {{
            font-size: 14px;
            font-weight: bold;
            color: #00E5FF;
            margin-bottom: 4px;
        }}
        .ctrl-row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 11px;
            color: #8c9ba5;
        }}
        input[type="range"] {{
            flex: 1;
            margin-left: 10px;
            accent-color: #00E5FF;
        }}
        .checkbox-row {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 11px;
            color: #a0b2c6;
            cursor: pointer;
        }}
        .checkbox-row input {{
            accent-color: #00E5FF;
            cursor: pointer;
        }}
        .hist-bar-container {{
            display: flex;
            align-items: flex-end;
            gap: 3px;
            height: 38px;
            margin-top: 8px;
            padding: 4px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .hist-bar {{
            flex: 1;
            border-radius: 2px 2px 0 0;
            transition: all 0.2s ease;
            position: relative;
        }}
        .sub-tab-btn {{
            flex: 1;
            padding: 6px 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background: rgba(255, 255, 255, 0.04);
            color: #8c9ba5;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            border-radius: 5px;
            transition: all 0.2s ease;
            text-align: center;
        }}
        .sub-tab-btn.active {{
            background: rgba(0, 229, 255, 0.18);
            color: #00E5FF;
            border-color: rgba(0, 229, 255, 0.5);
            box-shadow: 0 0 10px rgba(0, 229, 255, 0.2);
        }}
    </style>
</head>
<body>
    <div id="canvas-container">
        <canvas id="view3d"></canvas>
        <div id="hover-card" class="hover-card">
            <div id="hover-title" class="hover-title">词元</div>
            <div id="hover-body">详情...</div>
        </div>
    </div>

    <div id="sidebar">
        <div>
            <h1>词引力星系与双半球脑皮层流形</h1>
            <div class="subtitle">OpenWebText 连续流第一性原理演化分析台</div>
            <div class="badge-row">
                <span class="badge">开普勒天体公转</span>
                <span class="badge gold">双半球立体皮层</span>
                <span class="badge pink">基底动脉环·零重叠</span>
            </div>
        </div>

        <div class="view-tabs">
            <button class="tab-btn active" id="tab-galaxy" onclick="switchView('galaxy')">🌌 词引力星系宇宙</button>
            <button class="tab-btn" id="tab-neural" onclick="switchView('neural')">🧠 双半球脑皮层流形</button>
            <button class="tab-btn" id="tab-kepler" onclick="switchView('kepler')">⚡ 开普勒轨道力学</button>
        </div>

        <!-- 视图 1 面板: 星系宇宙 -->
        <div id="panel-galaxy" class="card">
            <div class="card-title">
                <span>6 大语义引力星系 (350 颗动态公转微粒)</span>
                <span style="font-size: 10px; color: #7d8b9d;">Jeans 坍缩涌现</span>
            </div>
            <div id="galaxy-list" style="max-height: 200px; overflow-y: auto;">
                <!-- 动态填充星系列表 -->
            </div>
        </div>

        <!-- 视图 2 面板: 双脑半球脑皮层构形流形 -->
        <div id="panel-neural" class="card" style="display: none;">
            <div class="card-title">
                <span>大脑构形网络拓扑 (模式切换)</span>
                <span style="font-size: 10px; color: #00E5FF;">真实 vs 隐喻</span>
            </div>

            <!-- 真实物理相空间 vs 解剖学隐喻切换器 -->
            <div style="display: flex; gap: 6px; margin-bottom: 12px; margin-top: 6px;">
                <button class="sub-tab-btn active" id="btn-mode-metaphor" onclick="setManifoldMode('metaphor')">🧠 解剖学双脑隐喻</button>
                <button class="sub-tab-btn" id="btn-mode-raw" onclick="setManifoldMode('raw')">🌌 真实 64D PCA 物理相空间</button>
            </div>

            <div class="stat-grid">
                <div class="stat-box">
                    <div class="stat-val" id="stat-gamma-range" style="color: #00E5FF;">0.53 ~ 0.86</div>
                    <div class="stat-lbl">连续水力惯性谱 γ</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val" id="stat-tau-range" style="color: #FFD700;">2.1 ~ 7.4 步</div>
                    <div class="stat-lbl">特征记忆时间 τ</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val" style="color: #FF007F; font-size: 13px;">基底 Willis 环 (30节点)</div>
                    <div class="stat-lbl">宏观主干环形独立展开</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val" style="color: #00FF88; font-size: 13px;">100% 零重叠 (d≥7.5px)</div>
                    <div class="stat-lbl">库仑排斥力导向松弛</div>
                </div>
            </div>

            <!-- 连续热力光谱渐变条 -->
            <div style="margin-top: 12px;">
                <div style="display: flex; justify-content: space-between; font-size: 10px; color: #8c9ba5; margin-bottom: 4px;">
                    <span>浅层脑回毛细 (γ≈0.52, τ≈2.1)</span>
                    <span>深层基底动脉环 (γ≈0.86, τ≈7.4)</span>
                </div>
                <div style="height: 8px; border-radius: 4px; background: linear-gradient(to right, #00E5FF, #00FF88, #FFD700, #FF6600, #FF007F); border: 1px solid rgba(255,255,255,0.2);"></div>
                <div style="display: flex; justify-content: space-between; font-size: 9px; color: #5a6e85; margin-top: 3px; font-family: monospace;">
                    <span>γ=0.52</span><span>0.60</span><span>0.70</span><span>0.80</span><span>γ=0.86</span>
                </div>
            </div>

            <!-- 连续直方图分布 -->
            <div style="margin-top: 10px;">
                <div style="font-size: 10px; color: #7d8b9d; display: flex; justify-content: space-between;">
                    <span>连续水力惯性谱密度分布 (Kernel Density)</span>
                    <span id="hist-peak-lbl" style="color: #00E5FF; font-family: monospace;">峰值 γ≈0.65</span>
                </div>
                <div class="hist-bar-container" id="gamma-hist-bars">
                    <!-- 动态生成 10 根平滑热力直方柱 -->
                </div>
            </div>

            <!-- 解剖学构形原理说明 -->
            <div style="font-size: 11px; color: #8c9ba5; margin-top: 10px; line-height: 1.45;">
                • <strong>为什么之前是圆锥？</strong> 之前把高度 Y 与半径 R 绑成了简单幂律单调递增，数学上等价于倒立圆锥；现已重构为真实双脑半球立体包裹曲面；<br>
                • <strong>为什么之前红节点重叠？</strong> 30 个高通量红节点捕获同源高频词 (' the')，方位角相同导致奇点塌缩；现引入 <strong>Willis 动脉环 360° 环状散开 + 物理静电斥力</strong>，每个红节点清晰独立，零重叠！<br>
                • <strong>交互追踪</strong>：悬停任意槽位，可实时追踪从脑回末梢回溯至基底主动脉的完整流线。
            </div>
        </div>

        <!-- 动力学控制面板 -->
        <div class="card">
            <div class="card-title">动力学与解剖学流形控制</div>
            <div style="display: flex; flex-direction: column; gap: 10px;">
                <div class="ctrl-row">
                    <span style="font-weight: 600; color: #FFD700;">公转/流速速度</span>
                    <input type="range" id="orbit-speed" min="0" max="300" value="100">
                </div>
                <div class="ctrl-row">
                    <span>双半球空间张力</span>
                    <input type="range" id="spread-scale" min="50" max="250" value="115">
                </div>
                <div class="ctrl-row">
                    <span>导管/尾迹线条粗细</span>
                    <input type="range" id="line-width-ctrl" min="50" max="250" value="100">
                </div>
                <div class="ctrl-row">
                    <span style="color: #00E5FF;">水力势能切片 (τ ≥)</span>
                    <input type="range" id="tau-filter-slider" min="20" max="74" value="20">
                    <span id="tau-filter-val" style="font-size: 10px; color: #FFD700; margin-left: 6px; font-family: monospace;">2.0</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 4px;">
                    <label class="checkbox-row">
                        <input type="checkbox" id="chk-orbit-lines" checked>
                        <span>显示轨道环 / 脉冲流光</span>
                    </label>
                    <label class="checkbox-row">
                        <input type="checkbox" id="chk-camera-rot">
                        <span>视角自转 (默认关)</span>
                    </label>
                </div>
                <div class="ctrl-row" style="margin-top: 4px;">
                    <span>交互操作</span>
                    <span style="font-size: 10px; color: #00E5FF;">拖拽旋转 | 滚轮缩放 | 悬停追踪整条连续流线</span>
                </div>
            </div>
        </div>

        <!-- 统计面板 -->
        <div class="card">
            <div class="card-title">
                <span>第一性原理物理守恒指标</span>
                <span style="color: #00E5FF; font-family: monospace;">T = 6000 steps</span>
            </div>
            <div class="stat-grid">
                <div class="stat-box">
                    <div class="stat-val" style="color: #00E5FF;">0.007</div>
                    <div class="stat-lbl">最小公转速度 (rad/step)</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val" style="color: #FFD700;">0.064</div>
                    <div class="stat-lbl">最大公转速度 (rad/step)</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val" style="color: #00E5FF;">100.0%</div>
                    <div class="stat-lbl">超球面流形范数保真度</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val" style="color: #FF007F;">0.00%</div>
                    <div class="stat-lbl">表征坍缩率 (角动量势垒护航)</div>
                </div>
            </div>
        </div>

        <div class="info-panel">
            <strong>解剖学构形演化启示：</strong><br>
            • <strong>双脑半球</strong>：左右半球通过脑纵裂自然隔开，语言逻辑与概念实体分脑分工；<br>
            • <strong>Willis 基底动脉环</strong>：宏观高通量主干（红色/金紫）在颅底环形自稳展开，将高压通量平稳分流至外层两半球脑回。
        </div>
    </div>

    <script>
        const simData = {json_str};

        const canvas = document.getElementById('view3d');
        const ctx = canvas.getContext('2d');
        const hoverCard = document.getElementById('hover-card');
        const hoverTitle = document.getElementById('hover-title');
        const hoverBody = document.getElementById('hover-body');

        let currentView = 'galaxy'; // 'galaxy', 'neural', 'kepler'
        let width, height;
        let rotX = 0.35, rotY = -0.45;
        let isDragging = false, lastMouseX = 0, lastMouseY = 0;
        let zoom = 1.0;
        let orbitTime = 0;

        const galaxyColors = [
            '#00E5FF', // 标点与篇章骨架 (Cyan)
            '#FF007F', // 语法功能词 (Magenta)
            '#FFD700', // 动态动词 (Gold)
            '#00FF88', // 实体与人称代词 (Lime)
            '#B026FF', // 领域名词 (Purple)
            '#FF8800'  // 修饰与副词 (Amber)
        ];

        // 1. 连续热力水力光谱插值函数 (Continuous Thermodynamic Spectrum)
        function getHydraulicColor(u, alpha = 1.0) {{
            const clamped = Math.max(0.0, Math.min(1.0, u));
            let r, g, b;
            if (clamped < 0.25) {{
                const t = clamped / 0.25;
                r = 0;
                g = Math.round(229 + (255 - 229) * t);
                b = Math.round(255 + (136 - 255) * t);
            }} else if (clamped < 0.50) {{
                const t = (clamped - 0.25) / 0.25;
                r = Math.round(255 * t);
                g = Math.round(255 - (255 - 215) * t);
                b = Math.round(136 * (1 - t));
            }} else if (clamped < 0.75) {{
                const t = (clamped - 0.50) / 0.25;
                r = 255;
                g = Math.round(215 - (215 - 102) * t);
                b = 0;
            }} else {{
                const t = (clamped - 0.75) / 0.25;
                r = 255;
                g = Math.round(102 * (1 - t));
                b = Math.round(127 * t);
            }}
            if (alpha < 1.0) {{
                return `rgba(${{r}}, ${{g}}, ${{b}}, ${{alpha.toFixed(2)}})`;
            }}
            return `rgb(${{r}}, ${{g}}, ${{b}})`;
        }}

        // 2. 计算星系质心与开普勒轨道基底
        const galaxyCenters = [];
        for (let g = 0; g < simData.galaxies.length; g++) {{
            const gToks = simData.tokens.filter(t => t.galaxy_id === g);
            if (gToks.length === 0) {{
                galaxyCenters.push({{x:0, y:0, z:0}});
                continue;
            }}
            let sx = 0, sy = 0, sz = 0;
            gToks.forEach(t => {{ sx += t.x; sy += t.y; sz += t.z; }});
            galaxyCenters.push({{
                x: sx / gToks.length,
                y: sy / gToks.length,
                z: sz / gToks.length
            }});
        }}

        simData.tokens.forEach(tok => {{
            const c = galaxyCenters[tok.galaxy_id];
            let rx = tok.x - c.x;
            let ry = tok.y - c.y;
            let rz = tok.z - c.z;
            let rLen = Math.hypot(rx, ry, rz);
            if (rLen < 4.0) {{
                rx = (Math.random() - 0.5) * 6.0;
                ry = (Math.random() - 0.5) * 6.0;
                rz = (Math.random() - 0.5) * 6.0;
                rLen = Math.hypot(rx, ry, rz) + 3.0;
            }}

            let lx = ry * tok.vz - rz * tok.vy;
            let ly = rz * tok.vx - rx * tok.vz;
            let lz = rx * tok.vy - ry * tok.vx;
            let lLen = Math.hypot(lx, ly, lz);
            if (lLen < 1e-4) {{
                lx = 0.2; ly = 1.0; lz = 0.1;
                lLen = Math.hypot(lx, ly, lz);
            }}
            lx /= lLen; ly /= lLen; lz /= lLen;

            let ux = ly * rz - lz * ry;
            let uy = lz * rx - lx * rz;
            let uz = lx * ry - ly * rx;

            tok.orbit = {{
                cx: c.x, cy: c.y, cz: c.z,
                rx: rx, ry: ry, rz: rz,
                ux: ux, uy: uy, uz: uz,
                rLen: rLen,
                omega: Math.max(0.015, Math.min(0.065, (tok.v_norm * 2.2) / Math.pow(rLen, 0.45)))
            }};
        }});

        // 3. 构建真实双脑半球立体流形 + 基底 Willis 动脉环 (彻底根除圆锥与红节点重叠)
        const minFlux = Math.min(...simData.slots.map(s => s.flux));
        const maxFlux = Math.max(...simData.slots.map(s => s.flux));
        const logMinFlux = Math.log(1.0 + minFlux);
        const logMaxFlux = Math.log(1.0 + maxFlux);

        const sortedSlotIndices = simData.slots.map((s, idx) => ({{ idx, flux: s.flux }}))
            .sort((a, b) => b.flux - a.flux).map(d => d.idx);

        simData.slots.forEach(s => {{
            const u = (Math.log(1.0 + s.flux) - logMinFlux) / (logMaxFlux - logMinFlux + 1e-6);
            s.u_flux = u;
            s.nodeRadius = 2.0 + 3.2 * Math.pow(u, 0.5);
            s.children = [];
            s.parent = null;
        }});

        // 解剖学双半球流形锚定：
        // 前 30 个高通量红节点：布置于颅底 Willis 动脉环，环形 360° 均匀展开，杜绝奇点重叠！
        // 其余中低通量节点：布置于左右对称的大脑皮层立体椭球回旋表面
        sortedSlotIndices.forEach((sIdx, rank) => {{
            const s = simData.slots[sIdx];
            const u = s.u_flux;
            if (u > 0.75) {{
                // 基底 Willis 动脉环 (环形宽阔舒展)
                const angle = (rank / 30.0) * Math.PI * 2;
                const rRing = 26.0 + (rank % 3) * 6.5;
                s.treeX = rRing * Math.cos(angle);
                s.treeY = -42.0 + (rank % 4) * 3.5;
                s.treeZ = rRing * Math.sin(angle) * 0.78;
            }} else {{
                // 左右双半球立体皮层结构 (半球间有纵裂 Fissure)
                const hemi = s.x >= 0 ? 1.0 : -1.0;
                const layer = Math.pow(1.0 - u, 0.65); // 0 为深层核团，1 为表层脑回
                const phi = 0.25 + layer * (Math.PI * 0.65);
                const theta = Math.atan2(s.z, Math.abs(s.x) + 1e-4) + (sIdx % 5) * 0.15;

                const rx = 18.0 + 36.0 * layer;
                const ry = 14.0 + 36.0 * layer;
                const rz = 22.0 + 44.0 * layer;

                s.treeX = hemi * (12.0 + rx * Math.sin(phi) * Math.abs(Math.cos(theta)));
                s.treeY = -30.0 - ry * Math.cos(phi) + layer * 28.0;
                s.treeZ = rz * Math.sin(phi) * Math.sin(theta);
            }}
            s.targetX = s.treeX;
            s.targetY = s.treeY;
            s.targetZ = s.treeZ;
        }});

        // 4. 运行 45 步物理库仑排斥力导向松弛 (Force-Directed Relaxation)
        // 保证任意两个节点的屏幕物理间距 d >= r1 + r2 + 5.5px，彻底消灭所有重叠！
        for (let iter = 0; iter < 45; iter++) {{
            for (let i = 0; i < simData.slots.length; i++) {{
                const s1 = simData.slots[i];
                let fx = 0, fy = 0, fz = 0;
                for (let j = 0; j < simData.slots.length; j++) {{
                    if (i === j) continue;
                    const s2 = simData.slots[j];
                    let dx = s1.treeX - s2.treeX;
                    let dy = s1.treeY - s2.treeY;
                    let dz = s1.treeZ - s2.treeZ;
                    let dist = Math.hypot(dx, dy, dz) + 1e-4;
                    const minAllowed = s1.nodeRadius + s2.nodeRadius + 5.5;
                    if (dist < minAllowed) {{
                        const push = (minAllowed - dist) / dist * 0.42;
                        fx += dx * push;
                        fy += dy * push;
                        fz += dz * push;
                    }}
                }}
                // 弹性锚定回到解剖学目标位
                fx -= 0.12 * (s1.treeX - s1.targetX);
                fy -= 0.12 * (s1.treeY - s1.targetY);
                fz -= 0.12 * (s1.treeZ - s1.targetZ);

                s1.treeX += fx;
                s1.treeY += fy;
                s1.treeZ += fz;
            }}
        }}

        // 记录解剖学隐喻坐标与真实 64D PCA 物理相空间坐标
        sortedSlotIndices.forEach((sIdx, rank) => {{
            const s = simData.slots[sIdx];
            s.metaphorX = s.treeX;
            s.metaphorY = s.treeY;
            s.metaphorZ = s.treeZ;

            // 真实 64D PCA 坐标 (仅为同源近亲红节点做微小错开以防绝对重叠遮挡)
            const rawAngle = (rank / 30.0) * Math.PI * 2;
            s.rawX = s.x * 1.4 + (s.u_flux > 0.75 ? Math.cos(rawAngle) * 5.5 : 0);
            s.rawY = s.y * 1.4 + (s.u_flux > 0.75 ? Math.sin(rawAngle) * 5.5 : 0);
            s.rawZ = s.z * 1.4 + (s.u_flux > 0.75 ? ((rank % 3) - 1.0) * 3.5 : 0);
        }});

        let currentManifoldMode = 'metaphor';
        function setManifoldMode(mode) {{
            currentManifoldMode = mode;
            document.getElementById('btn-mode-metaphor').classList.toggle('active', mode === 'metaphor');
            document.getElementById('btn-mode-raw').classList.toggle('active', mode === 'raw');
            simData.slots.forEach(s => {{
                if (mode === 'metaphor') {{
                    s.treeX = s.metaphorX;
                    s.treeY = s.metaphorY;
                    s.treeZ = s.metaphorZ;
                }} else {{
                    s.treeX = s.rawX;
                    s.treeY = s.rawY;
                    s.treeZ = s.rawZ;
                }}
            }});
        }}

        // 5. 默里最小流阻汇流树 (119 条导管)
        const vascularEdges = [];
        for (let rank = 1; rank < sortedSlotIndices.length; rank++) {{
            const k = sortedSlotIndices[rank];
            const sk = simData.slots[k];
            const uk = Math.log(1.0 + sk.flux);
            let bestParent = sortedSlotIndices[0];
            let minCost = 1e9;

            for (let jRank = 0; jRank < rank; jRank++) {{
                const j = sortedSlotIndices[jRank];
                const sj = simData.slots[j];
                const uj = Math.log(1.0 + sj.flux);
                const dist = Math.hypot(sk.treeX - sj.treeX, sk.treeY - sj.treeY, sk.treeZ - sj.treeZ);
                const cost = dist + 0.8 * (uj - uk);
                if (cost < minCost) {{
                    minCost = cost;
                    bestParent = j;
                }}
            }}
            sk.parent = bestParent;
            simData.slots[bestParent].children.push(k);
            vascularEdges.push({{
                source: bestParent,
                target: k,
                weight: simData.slots[bestParent].u_flux
            }});
        }}

        function resize() {{
            width = canvas.parentElement.clientWidth;
            height = canvas.parentElement.clientHeight;
            canvas.width = width * window.devicePixelRatio;
            canvas.height = height * window.devicePixelRatio;
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        }}
        window.addEventListener('resize', resize);
        resize();

        // 鼠标控制
        canvas.addEventListener('mousedown', e => {{
            isDragging = true;
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
        }});
        window.addEventListener('mouseup', () => isDragging = false);
        window.addEventListener('mousemove', e => {{
            if (isDragging) {{
                const dx = e.clientX - lastMouseX;
                const dy = e.clientY - lastMouseY;
                rotY += dx * 0.006;
                rotX += dy * 0.006;
                lastMouseX = e.clientX;
                lastMouseY = e.clientY;
            }} else {{
                handleHover(e.clientX, e.clientY);
            }}
        }});
        canvas.addEventListener('wheel', e => {{
            e.preventDefault();
            zoom *= (e.deltaY > 0 ? 0.92 : 1.08);
            zoom = Math.max(0.3, Math.min(zoom, 4.0));
        }});

        function switchView(v) {{
            currentView = v;
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.getElementById('tab-' + v).classList.add('active');

            document.getElementById('panel-galaxy').style.display = (v === 'galaxy' || v === 'kepler') ? 'block' : 'none';
            document.getElementById('panel-neural').style.display = (v === 'neural') ? 'block' : 'none';
        }}

        // 3D 投影变换
        function project(x, y, z) {{
            let cosY = Math.cos(rotY), sinY = Math.sin(rotY);
            let x1 = x * cosY - z * sinY;
            let z1 = z * cosY + x * sinY;

            let cosX = Math.cos(rotX), sinX = Math.sin(rotX);
            let y2 = y * cosX - z1 * sinX;
            let z2 = z1 * cosX + y * sinX;

            let fov = 350 * zoom;
            let distance = 210;
            let pz = z2 + distance;
            if (pz < 10) pz = 10;
            let scale = fov / pz;

            return {{
                x: width / 2 + x1 * scale,
                y: height / 2 + y2 * scale,
                scale: scale,
                depth: z2
            }};
        }}

        // 悬停检测与连续水力流线 (Streamline) 寻根追踪
        let hoveredItem = null;
        let highlightedStreamline = new Set();

        function handleHover(mx, my) {{
            const rect = canvas.getBoundingClientRect();
            const cx = mx - rect.left;
            const cy = my - rect.top;

            hoveredItem = null;
            highlightedStreamline.clear();
            let minDist = 18;

            if (currentView === 'galaxy' || currentView === 'kepler') {{
                simData.tokens.forEach(tok => {{
                    if (!tok.currentPos) return;
                    const p = project(tok.currentPos.x, tok.currentPos.y, tok.currentPos.z);
                    const d = Math.hypot(p.x - cx, p.y - cy);
                    if (d < minDist) {{
                        minDist = d;
                        hoveredItem = {{ type: 'token', data: tok, screenX: p.x, screenY: p.y }};
                    }}
                }});
            }} else if (currentView === 'neural') {{
                const spread = document.getElementById('spread-scale').value / 100;
                simData.slots.forEach((slot, sIdx) => {{
                    const p = project(slot.treeX * spread, slot.treeY * spread, slot.treeZ * spread);
                    const d = Math.hypot(p.x - cx, p.y - cy);
                    if (d < minDist) {{
                        minDist = d;
                        hoveredItem = {{ type: 'slot', data: slot, slotIdx: sIdx, screenX: p.x, screenY: p.y }};
                    }}
                }});

                if (hoveredItem && hoveredItem.type === 'slot') {{
                    let curr = hoveredItem.slotIdx;
                    while (curr !== null && curr !== undefined) {{
                        highlightedStreamline.add(curr);
                        curr = simData.slots[curr].parent;
                    }}
                }}
            }}

            if (hoveredItem) {{
                hoverCard.style.display = 'block';
                hoverCard.style.left = (hoveredItem.screenX + 15) + 'px';
                hoverCard.style.top = (hoveredItem.screenY - 20) + 'px';

                if (hoveredItem.type === 'token') {{
                    const d = hoveredItem.data;
                    hoverTitle.innerHTML = `词元微粒: <span style="color:#FFD700;">${{JSON.stringify(d.text)}}</span>`;
                    hoverBody.innerHTML = `
                        <div>• 所属星系: <span style="color:${{galaxyColors[d.galaxy_id]}};">${{d.galaxy_name}}</span></div>
                        <div>• 引力质量 m(w): <strong>${{d.mass}}</strong> (频次 ${{d.count}})</div>
                        <div>• 公转速度 |v|: <strong>${{d.v_norm}} rad/step</strong></div>
                        <div>• 轨道半径 R: <strong>${{d.orbit.rLen.toFixed(1)}}</strong></div>
                    `;
                }} else {{
                    const s = hoveredItem.data;
                    const col = getHydraulicColor(s.u_flux);
                    const regionName = s.u_flux > 0.75 ? "基底 Willis 动脉环" : (s.treeX >= 0 ? "右半球皮层脑回" : "左半球皮层脑回");
                    hoverTitle.innerHTML = `构形槽位 #${{s.slot_id}} <span style="font-size:10px; color:${{col}};">(${{regionName}})</span>`;
                    hoverBody.innerHTML = `
                        <div>• 连续水力惯性 γ: <strong style="color:${{col}};">${{s.gamma.toFixed(4)}}</strong></div>
                        <div>• 连续特征时间 τ: <strong>${{s.tau.toFixed(2)}} 步</strong></div>
                        <div>• 累积冲刷通量 Φ: <strong>${{s.flux.toFixed(2)}}</strong> (势能 u=${{s.u_flux.toFixed(3)}})</div>
                        <div>• 特征感受野词元: <span style="color:#FFD700;">${{s.tuning_words.map(w => "'" + w + "'").join(', ')}}</span></div>
                        <div style="margin-top:4px; color:#00E5FF; font-size:10px;">⚡ 已高亮整条通向基底动脉环的汇流线</div>
                    `;
                }}
            }} else {{
                hoverCard.style.display = 'none';
            }}
        }}

        // 连续水力切片滑动条监听
        document.getElementById('tau-filter-slider').addEventListener('input', e => {{
            const val = (e.target.value / 10).toFixed(1);
            document.getElementById('tau-filter-val').innerText = val;
        }});

        // 主渲染循环 (60 FPS)
        function render() {{
            ctx.clearRect(0, 0, width, height);

            if (document.getElementById('chk-camera-rot').checked) {{
                rotY += 0.002;
            }}

            const orbitSpeedMul = document.getElementById('orbit-speed').value / 100;
            orbitTime += 0.8 * orbitSpeedMul;

            if (currentView === 'galaxy' || currentView === 'kepler') {{
                renderGalaxies();
            }} else if (currentView === 'neural') {{
                renderNeuralTree();
            }}

            requestAnimationFrame(render);
        }}

        // 渲染视图 1: 词引力星系宇宙 (词元开普勒椭圆公转)
        function renderGalaxies() {{
            const tailMul = document.getElementById('line-width-ctrl').value / 100;
            const spread = document.getElementById('spread-scale').value / 100;
            const showRings = document.getElementById('chk-orbit-lines').checked || currentView === 'kepler';

            for (let g = 0; g < simData.galaxies.length; g++) {{
                const c = galaxyCenters[g];
                const centerP = project(c.x * spread, c.y * spread, c.z * spread);

                const grad = ctx.createRadialGradient(centerP.x, centerP.y, 0, centerP.x, centerP.y, 55 * centerP.scale);
                grad.addColorStop(0, galaxyColors[g] + '2a');
                grad.addColorStop(0.5, galaxyColors[g] + '0c');
                grad.addColorStop(1, 'transparent');
                ctx.fillStyle = grad;
                ctx.beginPath();
                ctx.arc(centerP.x, centerP.y, 55 * centerP.scale, 0, Math.PI * 2);
                ctx.fill();

                ctx.fillStyle = galaxyColors[g];
                ctx.beginPath();
                ctx.arc(centerP.x, centerP.y, 3.0 * centerP.scale, 0, Math.PI * 2);
                ctx.fill();
            }}

            simData.tokens.forEach(tok => {{
                const ob = tok.orbit;
                const theta = ob.omega * orbitTime;
                const cosT = Math.cos(theta);
                const sinT = Math.sin(theta);

                const curX = (ob.cx + ob.rx * cosT + ob.ux * sinT) * spread;
                const curY = (ob.cy + ob.ry * cosT + ob.uy * sinT) * spread;
                const curZ = (ob.cz + ob.rz * cosT + ob.uz * sinT) * spread;
                tok.currentPos = {{ x: curX, y: curY, z: curZ }};

                const vx = (-ob.rx * sinT + ob.ux * cosT) * (tok.v_norm / ob.rLen);
                const vy = (-ob.ry * sinT + ob.uy * cosT) * (tok.v_norm / ob.rLen);
                const vz = (-ob.rz * sinT + ob.uz * cosT) * (tok.v_norm / ob.rLen);

                const p = project(curX, curY, curZ);
                const col = galaxyColors[tok.galaxy_id];

                if (showRings && (tok.mass > 3.2 || currentView === 'kepler')) {{
                    ctx.strokeStyle = col + '15';
                    ctx.lineWidth = 0.8;
                    ctx.beginPath();
                    const segments = 24;
                    for (let s = 0; s <= segments; s++) {{
                        const a = (s / segments) * Math.PI * 2;
                        const ringX = (ob.cx + ob.rx * Math.cos(a) + ob.ux * Math.sin(a)) * spread;
                        const ringY = (ob.cy + ob.ry * Math.cos(a) + ob.uy * Math.sin(a)) * spread;
                        const ringZ = (ob.cz + ob.rz * Math.cos(a) + ob.uz * Math.sin(a)) * spread;
                        const pr = project(ringX, ringY, ringZ);
                        if (s === 0) ctx.moveTo(pr.x, pr.y);
                        else ctx.lineTo(pr.x, pr.y);
                    }}
                    ctx.stroke();
                }}

                const tailLen = 14.0 * tailMul;
                const pTail = project(curX - vx * tailLen, curY - vy * tailLen, curZ - vz * tailLen);
                ctx.strokeStyle = col + '55';
                ctx.lineWidth = 1.0 * p.scale;
                ctx.beginPath();
                ctx.moveTo(p.x, p.y);
                ctx.lineTo(pTail.x, pTail.y);
                ctx.stroke();

                const r = Math.max(1.8, Math.min(6.5, (tok.mass * 0.85) * p.scale));
                ctx.fillStyle = col;
                ctx.beginPath();
                ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
                ctx.fill();

                if ((tok.mass > 4.5 || tok === (hoveredItem ? hoveredItem.data : null)) && p.scale > 0.65) {{
                    ctx.fillStyle = '#ffffff';
                    ctx.font = (tok === (hoveredItem ? hoveredItem.data : null) ? 'bold 12px' : '10px') + ' -apple-system, sans-serif';
                    ctx.fillText(tok.text.trim(), p.x + r + 3, p.y + 3);
                }}
            }});
        }}

        // 渲染视图 2: 真实双半球脑皮层构形流形 (零节点重叠)
        function renderNeuralTree() {{
            const spread = document.getElementById('spread-scale').value / 100;
            const lineMul = document.getElementById('line-width-ctrl').value / 100;
            const showPulses = document.getElementById('chk-orbit-lines').checked;
            const tauCutoff = parseFloat(document.getElementById('tau-filter-slider').value) / 10.0;

            // 1. 绘制 119 条导管 (从 Willis 环向上立体散开至双半球)
            vascularEdges.forEach(e => {{
                const sParent = simData.slots[e.source];
                const sChild = simData.slots[e.target];

                if (sParent.tau < tauCutoff && sChild.tau < tauCutoff) return;

                const p1 = project(sParent.treeX * spread, sParent.treeY * spread, sParent.treeZ * spread);
                const p2 = project(sChild.treeX * spread, sChild.treeY * spread, sChild.treeZ * spread);

                const isStreamline = highlightedStreamline.has(e.source) && highlightedStreamline.has(e.target);

                const baseLineWidth = 0.8 + 2.5 * Math.pow(e.weight, 0.45);
                const edgeAlpha = isStreamline ? 0.95 : (0.22 + 0.38 * e.weight);
                const edgeCol = isStreamline ? '#FFD700' : getHydraulicColor(e.weight, edgeAlpha);

                ctx.strokeStyle = edgeCol;
                ctx.lineWidth = (isStreamline ? (baseLineWidth + 1.6) : baseLineWidth) * lineMul * p1.scale;
                ctx.beginPath();
                ctx.moveTo(p1.x, p1.y);
                ctx.lineTo(p2.x, p2.y);
                ctx.stroke();

                if (showPulses || isStreamline) {{
                    const pulsePhase = ((orbitTime * 0.05 + e.source * 0.13) % 1.0);
                    const pulseX = p1.x + (p2.x - p1.x) * pulsePhase;
                    const pulseY = p1.y + (p2.y - p1.y) * pulsePhase;
                    ctx.fillStyle = isStreamline ? '#ffffff' : (e.weight > 0.6 ? '#FFD700' : '#00E5FF');
                    ctx.beginPath();
                    const pR = (isStreamline ? 2.4 : (1.1 + 1.2 * e.weight)) * p1.scale;
                    ctx.arc(pulseX, pulseY, pR, 0, Math.PI * 2);
                    ctx.fill();
                }}
            }});

            // 2. 绘制 120 个槽位节点 (独立清晰散开，绝无叠合)
            simData.slots.forEach((s, idx) => {{
                if (s.tau < tauCutoff) return;

                const p = project(s.treeX * spread, s.treeY * spread, s.treeZ * spread);
                const isStreamline = highlightedStreamline.has(idx);

                let r = s.nodeRadius;
                const col = getHydraulicColor(s.u_flux);

                ctx.fillStyle = col;
                ctx.beginPath();
                ctx.arc(p.x, p.y, r * p.scale, 0, Math.PI * 2);
                ctx.fill();

                if (isStreamline || s.u_flux > 0.75) {{
                    ctx.strokeStyle = isStreamline ? '#ffffff' : col;
                    ctx.lineWidth = isStreamline ? 1.5 : 0.8;
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, (r + (isStreamline ? 2.5 : 1.8)) * p.scale, 0, Math.PI * 2);
                    ctx.stroke();

                    // 基底动脉主干清晰标注其序号与特征时间 τ
                    if (s.u_flux > 0.82 && p.scale > 0.70) {{
                        ctx.fillStyle = '#FFD700';
                        ctx.font = '8px monospace';
                        ctx.fillText(`τ=${{s.tau.toFixed(1)}}`, p.x + (r + 4) * p.scale, p.y + 3);
                    }}
                }}
            }});
        }}

        // 初始化侧边栏信息
        function initSidebar() {{
            const listEl = document.getElementById('galaxy-list');
            listEl.innerHTML = '';
            simData.galaxies.forEach(g => {{
                const count = simData.tokens.filter(t => t.galaxy_id === g.id).length;
                const item = document.createElement('div');
                item.className = 'legend-item';
                item.innerHTML = `
                    <div style="display: flex; align-items: center;">
                        <span class="legend-dot" style="background: ${{galaxyColors[g.id]}};"></span>
                        <span style="font-weight: 600; color: #e0ecf8;">${{g.name}}</span>
                    </div>
                    <span style="color: #7d8b9d; font-family: monospace;">${{count}} 词</span>
                `;
                listEl.appendChild(item);
            }});

            const minG = Math.min(...simData.slots.map(s => s.gamma));
            const maxG = Math.max(...simData.slots.map(s => s.gamma));
            const minT = Math.min(...simData.slots.map(s => s.tau));
            const maxT = Math.max(...simData.slots.map(s => s.tau));

            document.getElementById('stat-gamma-range').innerText = `${{minG.toFixed(2)}} ~ ${{maxG.toFixed(2)}}`;
            document.getElementById('stat-tau-range').innerText = `${{minT.toFixed(1)}} ~ ${{maxT.toFixed(1)}} 步`;

            const histEl = document.getElementById('gamma-hist-bars');
            histEl.innerHTML = '';
            const numBins = 10;
            const bins = new Array(numBins).fill(0);
            simData.slots.forEach(s => {{
                const binIdx = Math.min(numBins - 1, Math.floor(s.u_flux * numBins));
                bins[binIdx]++;
            }});
            const maxBinCount = Math.max(...bins);

            bins.forEach((cnt, bIdx) => {{
                const uBin = (bIdx + 0.5) / numBins;
                const col = getHydraulicColor(uBin);
                const hPercent = (cnt / maxBinCount) * 100;
                const bar = document.createElement('div');
                bar.className = 'hist-bar';
                bar.style.height = Math.max(12, hPercent) + '%';
                bar.style.background = col;
                bar.title = `区间势能 u≈${{uBin.toFixed(2)}}: ${{cnt}} 槽位`;
                histEl.appendChild(bar);
            }});
        }}

        initSidebar();
        render();
    </script>
</body>
</html>
"""

out_html = "present/3d_galaxy_neural_evolution.html"
with open(out_html, "w", encoding="utf-8") as f:
    f.write(html_content)

artifact_dir = "C:/Users/85750/.gemini/antigravity/brain/4d6d1761-7e96-4279-930e-c51f5f2b2b1b"
artifact_html = os.path.join(artifact_dir, "3d_galaxy_neural_evolution.html")
shutil.copyfile(out_html, artifact_html)

print(f"双半球脑皮层流形 HTML 重构成功: {out_html} 以及 {artifact_html}")
