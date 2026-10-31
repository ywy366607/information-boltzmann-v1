import json

with open('C:/Users/85750/.gemini/antigravity/brain/4d6d1761-7e96-4279-930e-c51f5f2b2b1b/scratch/source_sink_evolution_data.json') as f:
    sim_data = json.load(f)

json_str = json.dumps(sim_data)

html_code = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>源-汇任务流驱动 3D 构形场演化交互模拟器 (Constructal Law Simulation)</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: #04060a;
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
        #sidebar {{
            width: 440px;
            height: 100%;
            background: rgba(9, 13, 22, 0.96);
            backdrop-filter: blur(18px);
            border-left: 1px solid rgba(255, 255, 255, 0.08);
            padding: 22px;
            overflow-y: auto;
            z-index: 10;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        h1 {{
            font-size: 19px;
            font-weight: 700;
            background: linear-gradient(135deg, #00E5FF, #FFD700, #00E676);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtitle {{
            font-size: 12px;
            color: #7d8b9d;
            line-height: 1.4;
            margin-top: 2px;
        }}
        .card {{
            background: rgba(255, 255, 255, 0.035);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 10px;
            padding: 13px;
        }}
        .card-title {{
            font-size: 13px;
            font-weight: 600;
            color: #00E5FF;
            margin-bottom: 9px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .timeline-slider {{
            width: 100%;
            -webkit-appearance: none;
            background: #151c2c;
            height: 6px;
            border-radius: 3px;
            outline: none;
            margin: 6px 0 10px 0;
        }}
        .timeline-slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #00E5FF;
            cursor: pointer;
            box-shadow: 0 0 12px #00E5FF;
        }}
        .btn-row {{
            display: flex;
            gap: 8px;
        }}
        .btn {{
            flex: 1;
            padding: 8px 12px;
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            background: rgba(255, 255, 255, 0.05);
            color: #ccd6f6;
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
            transition: all 0.2s;
            text-align: center;
        }}
        .btn:hover {{
            background: rgba(0, 229, 255, 0.15);
            border-color: #00E5FF;
            color: #00E5FF;
        }}
        .btn.active {{
            background: #00E5FF;
            color: #080a10;
            border-color: #00E5FF;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }}
        .metric-box {{
            background: rgba(0, 0, 0, 0.35);
            padding: 8px;
            border-radius: 6px;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, 0.04);
        }}
        .metric-val {{
            font-size: 16px;
            font-weight: 700;
            color: #fff;
        }}
        .metric-lbl {{
            font-size: 10px;
            color: #7d8b9d;
            margin-top: 2px;
        }}
        .task-panel {{
            background: rgba(0, 0, 0, 0.45);
            border-left: 3px solid #00E5FF;
            padding: 10px 12px;
            border-radius: 0 6px 6px 0;
            font-size: 12px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        .task-row {{
            display: flex;
            justify-content: space-between;
        }}
        .event-ticker {{
            font-family: Consolas, monospace;
            font-size: 11px;
            color: #64ffda;
            background: rgba(0, 0, 0, 0.5);
            padding: 8px 10px;
            border-radius: 6px;
            max-height: 55px;
            overflow-y: auto;
            border: 1px solid rgba(100, 255, 218, 0.2);
        }}
        #hud {{
            position: absolute;
            top: 20px;
            left: 20px;
            pointer-events: none;
            background: rgba(6, 10, 18, 0.88);
            backdrop-filter: blur(10px);
            padding: 14px 20px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            font-size: 12px;
            line-height: 1.6;
        }}
        .node-tag {{
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 6px;
        }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="canvas-container">
        <div id="hud">
            <b>贝让构形定律验证：外加贯穿任务流 (Imposed Currents)</b><br>
            • 左侧两个青色节点: <b>源端 (Source A, B)</b> 持续输入题目高压<br>
            • 右侧红/绿光球: <b>汇端 (Sink Drain)</b> 任务考核出口 (匹配则畅通，偏离则高阻反压)<br>
            • 中间流道: <b>受压差强行雕刻的树状流道</b> (粗细表示管径 D_{{ij}}，颜色表示能流活性)
        </div>
    </div>

    <div id="sidebar">
        <div>
            <h1>源-汇任务流驱动构形场</h1>
            <div class="subtitle">贝让构形定律 • 任务反压阻抗 • 黏菌流道强化</div>
        </div>

        <!-- 演化时间控制器 -->
        <div class="card">
            <div class="card-title">
                <span>演化时钟</span>
                <span id="epoch-label" style="color: #00E5FF; font-weight: bold;">Epoch 0 / 120</span>
            </div>
            <input type="range" class="timeline-slider" id="time-slider" min="0" max="119" value="0" step="1" oninput="onSlideStep(this.value)">
            <div class="btn-row">
                <button class="btn" id="btn-play" onclick="togglePlay()">▶ 连续演化流回放</button>
                <button class="btn" onclick="stepTime(-1)">◀ 单步</button>
                <button class="btn" onclick="stepTime(1)">单步 ▶</button>
            </div>
        </div>

        <!-- 外部任务考核面板 -->
        <div class="card">
            <div class="card-title">外部任务流与做功考核 (A × B)</div>
            <div class="task-panel" id="task-box">
                <div class="task-row">
                    <span style="color: #8892b0;">源端输入流:</span>
                    <span style="font-family: monospace; color: #00E5FF;" id="val-inputs">A = 1.60, B = 2.00</span>
                </div>
                <div class="task-row">
                    <span style="color: #8892b0;">汇端目标 (A × B):</span>
                    <span style="font-family: monospace; font-weight: bold; color: #FFD700;" id="val-target">3.20</span>
                </div>
                <div class="task-row">
                    <span style="color: #8892b0;">汇端实际流出:</span>
                    <span style="font-family: monospace; font-weight: bold;" id="val-output">0.04</span>
                </div>
                <div class="task-row" style="margin-top: 4px; padding-top: 4px; border-top: 1px dashed rgba(255,255,255,0.1);">
                    <span style="color: #8892b0;">当前任务误差:</span>
                    <span style="font-family: monospace; font-weight: bold;" id="val-error">3.16</span>
                </div>
            </div>
        </div>

        <!-- 贯穿阻抗仪表盘 -->
        <div class="card">
            <div class="card-title">物理贯穿阻抗与流道统计</div>
            <div class="metric-grid">
                <div class="metric-box">
                    <div class="metric-val" id="metric-resistance" style="color: #FF0055;">84.39</div>
                    <div class="metric-lbl">全流道贯穿阻抗 R (→1.4)</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val" id="metric-slots" style="color: #00E676;">10</div>
                    <div class="metric-lbl">活跃流道槽位数 (10→26)</div>
                </div>
            </div>
        </div>

        <!-- 拓扑演化事件 -->
        <div class="card">
            <div class="card-title">流道重构事件</div>
            <div class="event-ticker" id="event-log">系统初始化就绪，源端开始加压...</div>
        </div>

        <!-- 构形定律图例 -->
        <div class="card" style="font-size: 11px; line-height: 1.6; color: #8892b0;">
            <div class="card-title" style="margin-bottom: 6px;">构形定律如何雕刻出算术回路？</div>
            1. <b>初始阻塞 (Epoch 0)</b>: 内部没有能算出乘法的通道，汇端形成高反压，贯穿阻抗高达 <b>84.4</b>；<br>
            2. <b>反压爆裂 (Epoch 20)</b>: 巨大反压逼迫中间槽位发生 <b>Hopf 余乘 Δ</b>，槽位数由 10 激增至 26，炸开大量并联旁路；<br>
            3. <b>流道定型 (Epoch 40~119)</b>: 能够算出 $A \times B$ 的非线性二次流道被通量<b>自发增粗加固 (黏菌定律)</b>，无效管道萎缩。贯穿阻抗断崖式跌落至 <b>1.40 (-86.9%)</b>，误差骤降 <b>-84.5%</b>！
        </div>
    </div>

    <script>
        const simData = {json_str};
        const snapshots = simData.snapshots;
        let currentStep = 0;
        let isPlaying = false;
        let playTimer = null;

        const container = document.getElementById('canvas-container');
        const scene = new THREE.Scene();
        scene.fog = new THREE.FogExp2(0x04060a, 0.4);

        const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 100);
        camera.position.set(0, 0.6, 2.3);

        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.target.set(0, 0, 0);

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
        scene.add(ambientLight);

        let nodesGroup = new THREE.Group();
        scene.add(nodesGroup);
        let tubesGroup = new THREE.Group();
        scene.add(tubesGroup);

        function updateScene(idx) {{
            const snap = snapshots[idx];
            if (!snap) return;

            // 1. 刷新节点
            while (nodesGroup.children.length > 0) {{
                nodesGroup.remove(nodesGroup.children[0]);
            }}

            snap.nodes.forEach(n => {{
                let colHex = 0x2979FF; // 内部线性节点 蓝色
                let r = 0.038;

                if (n.is_source) {{
                    colHex = 0x00E5FF; // 源节点 青色
                    r = 0.065;
                }} else if (n.is_sink) {{
                    colHex = (snap.error < 0.6) ? 0x00E676 : 0xFF1744; // 汇节点 低误差绿，高误差红
                    r = 0.07;
                }} else if (n.type === 'quadratic') {{
                    colHex = 0xFFD700; // 二次整流非线性节点 金色
                    r = 0.045;
                }}

                const geom = new THREE.SphereGeometry(r, 16, 16);
                const mat = new THREE.MeshBasicMaterial({{ color: colHex }});
                const sphere = new THREE.Mesh(geom, mat);
                sphere.position.set(n.x * 1.3, n.y * 1.3, n.z * 1.3);
                nodesGroup.add(sphere);

                // 源和汇加光环
                if (n.is_source || n.is_sink) {{
                    const ringGeom = new THREE.RingGeometry(r * 1.25, r * 1.5, 16);
                    const ringMat = new THREE.MeshBasicMaterial({{
                        color: colHex,
                        transparent: true,
                        opacity: 0.75,
                        side: THREE.DoubleSide
                    }});
                    const ring = new THREE.Mesh(ringGeom, ringMat);
                    ring.position.copy(sphere.position);
                    ring.lookAt(camera.position);
                    nodesGroup.add(ring);
                }}
            }});

            // 2. 刷新导通管道
            while (tubesGroup.children.length > 0) {{
                tubesGroup.remove(tubesGroup.children[0]);
            }}

            const nodeMap = {{}};
            snap.nodes.forEach(n => {{ nodeMap[n.id] = n; }});

            snap.tubes.forEach(t => {{
                const n1 = nodeMap[t.source];
                const n2 = nodeMap[t.target];
                if (!n1 || !n2) return;

                const p1 = new THREE.Vector3(n1.x * 1.3, n1.y * 1.3, n1.z * 1.3);
                const p2 = new THREE.Vector3(n2.x * 1.3, n2.y * 1.3, n2.z * 1.3);

                const positions = [p1.x, p1.y, p1.z, p2.x, p2.y, p2.z];
                const geom = new THREE.BufferGeometry();
                geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));

                // 管径越大越亮越粗
                const opacity = Math.min(0.9, Math.max(0.15, t.weight * 0.7));
                const mat = new THREE.LineBasicMaterial({{
                    color: (snap.error < 0.6) ? 0x00E5FF : 0xFF5252,
                    transparent: true,
                    opacity: opacity,
                    blending: THREE.AdditiveBlending
                }});

                const line = new THREE.Line(geom, mat);
                tubesGroup.add(line);
            }});

            // 3. 刷新 UI
            document.getElementById('time-slider').value = idx;
            document.getElementById('epoch-label').innerText = `Epoch ${{snap.epoch}} / 120`;
            document.getElementById('val-inputs').innerText = `A = ${{snap.a.toFixed(2)}}, B = ${{snap.b.toFixed(2)}}`;
            document.getElementById('val-target').innerText = snap.target.toFixed(2);

            const outElem = document.getElementById('val-output');
            outElem.innerText = snap.output.toFixed(2);
            outElem.style.color = (snap.error < 0.6) ? '#00E676' : '#FF5252';

            const errElem = document.getElementById('val-error');
            errElem.innerText = snap.error.toFixed(3);
            errElem.style.color = (snap.error < 0.6) ? '#00E676' : '#FF5252';

            const resElem = document.getElementById('metric-resistance');
            resElem.innerText = snap.resistance.toFixed(2);
            resElem.style.color = (snap.resistance < 3.0) ? '#00E676' : '#FF0055';

            document.getElementById('metric-slots').innerText = snap.num_slots;
            document.getElementById('event-log').innerText = snap.recent_event;
        }}

        function onSlideStep(val) {{
            currentStep = parseInt(val);
            updateScene(currentStep);
        }}

        function stepTime(delta) {{
            let next = currentStep + delta;
            if (next >= 0 && next < snapshots.length) {{
                currentStep = next;
                updateScene(currentStep);
            }}
        }}

        function togglePlay() {{
            isPlaying = !isPlaying;
            const btn = document.getElementById('btn-play');
            if (isPlaying) {{
                btn.innerText = '⏸ 暂停演化回放';
                btn.classList.add('active');
                playTimer = setInterval(() => {{
                    currentStep = (currentStep + 1) % snapshots.length;
                    updateScene(currentStep);
                }}, 80);
            }} else {{
                btn.innerText = '▶ 连续演化流回放';
                btn.classList.remove('active');
                clearInterval(playTimer);
            }}
        }}

        updateScene(0);

        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();

        window.addEventListener('resize', () => {{
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        }});
    </script>
</body>
</html>
"""

target_html = 'C:/Users/85750/.gemini/antigravity/brain/4d6d1761-7e96-4279-930e-c51f5f2b2b1b/3d_source_sink_constructal_demo.html'
with open(target_html, 'w', encoding='utf-8') as f:
    f.write(html_code)

print(f"Source-Sink Constructal HTML demo built at {target_html}")
