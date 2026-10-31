import json

with open('C:/Users/85750/.gemini/antigravity/brain/4d6d1761-7e96-4279-930e-c51f5f2b2b1b/scratch/extended_source_sink_evolution_data.json') as f:
    sim_data = json.load(f)

json_str = json.dumps(sim_data)

html_code = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>构形场引擎 (CFE) 交互式实验台与 300 轮演化全景</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: #030508;
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
            width: 460px;
            height: 100%;
            background: rgba(8, 12, 20, 0.96);
            backdrop-filter: blur(18px);
            border-left: 1px solid rgba(255, 255, 255, 0.08);
            padding: 20px;
            overflow-y: auto;
            z-index: 10;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}
        h1 {{
            font-size: 18px;
            font-weight: 700;
            background: linear-gradient(135deg, #00E5FF, #FFD700, #FF007F);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtitle {{
            font-size: 11px;
            color: #7d8b9d;
            line-height: 1.4;
            margin-top: 2px;
        }}
        .tab-row {{
            display: flex;
            background: rgba(255, 255, 255, 0.04);
            border-radius: 8px;
            padding: 3px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            gap: 4px;
        }}
        .tab-btn {{
            flex: 1;
            padding: 8px 10px;
            border: none;
            background: transparent;
            color: #8892b0;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            border-radius: 6px;
            transition: all 0.2s;
        }}
        .tab-btn.active {{
            background: rgba(0, 229, 255, 0.15);
            color: #00E5FF;
            box-shadow: 0 0 10px rgba(0, 229, 255, 0.2);
        }}
        .card {{
            background: rgba(255, 255, 255, 0.035);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 10px;
            padding: 12px;
        }}
        .card-title {{
            font-size: 12px;
            font-weight: 600;
            color: #00E5FF;
            margin-bottom: 8px;
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
            background: #FF007F;
            cursor: pointer;
            box-shadow: 0 0 12px #FF007F;
        }}
        .btn-row {{
            display: flex;
            gap: 8px;
        }}
        .btn {{
            flex: 1;
            padding: 7px 10px;
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            background: rgba(255, 255, 255, 0.05);
            color: #ccd6f6;
            cursor: pointer;
            font-size: 11px;
            transition: all 0.2s;
        }}
        .btn:hover {{
            background: rgba(0, 229, 255, 0.2);
            border-color: #00E5FF;
        }}
        .preset-btn {{
            padding: 5px 8px;
            border-radius: 5px;
            border: 1px solid rgba(255, 215, 0, 0.3);
            background: rgba(255, 215, 0, 0.08);
            color: #FFD700;
            cursor: pointer;
            font-size: 11px;
            font-family: Consolas, monospace;
        }}
        .preset-btn:hover {{
            background: rgba(255, 215, 0, 0.25);
        }}
        .task-panel {{
            display: flex;
            flex-direction: column;
            gap: 6px;
            font-size: 12px;
        }}
        .task-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }}
        .metric-box {{
            background: rgba(0, 0, 0, 0.4);
            border-radius: 8px;
            padding: 8px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            text-align: center;
        }}
        .metric-val {{
            font-size: 16px;
            font-weight: bold;
            font-family: Consolas, monospace;
        }}
        .metric-lbl {{
            font-size: 10px;
            color: #7d8b9d;
            margin-top: 2px;
        }}
        .event-ticker {{
            font-family: Consolas, monospace;
            font-size: 11px;
            color: #64ffda;
            background: rgba(0, 0, 0, 0.5);
            padding: 8px 10px;
            border-radius: 6px;
            max-height: 48px;
            overflow-y: auto;
            border: 1px solid rgba(100, 255, 218, 0.2);
        }}
        #hud {{
            position: absolute;
            top: 20px;
            left: 20px;
            pointer-events: none;
            background: rgba(6, 10, 18, 0.90);
            backdrop-filter: blur(10px);
            padding: 14px 20px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            font-size: 12px;
            line-height: 1.6;
        }}
        .legend-tag {{
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
            vertical-align: middle;
        }}
        .slider-group {{
            margin: 6px 0;
        }}
        .slider-header {{
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: #8892b0;
            margin-bottom: 3px;
        }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="canvas-container">
        <div id="hud">
            <b>构形场引擎 (CFE)：实时自定义输入与计算流场</b><br>
            • <span class="legend-tag" style="background:#00E5FF;"></span><b>源端 A, B (x = -0.85)</b>: 注入你指定的输入数值 (光球大小随输入动态伸缩)<br>
            • <span class="legend-tag" style="background:#FF007F;"></span><b>汇端 Drain (x = +0.85)</b>: 极化相消后的实际物理泄流结果<br>
            • <span class="legend-tag" style="background:#FFD700;"></span><b>金黄色管道</b>: 同相相长通路 (+W) | <span class="legend-tag" style="background:#00E5FF;"></span><b>青色管道</b>: 反相相消通路 (-W)
        </div>
    </div>

    <div id="sidebar">
        <div>
            <h1>构形场引擎 (CFE) 控制台</h1>
            <div class="subtitle">极化相消 • 任意输入实时交互 • 300 轮演化回放</div>
        </div>

        <!-- 模式切换选项卡 -->
        <div class="tab-row">
            <button class="tab-btn active" id="tab-custom" onclick="switchMode('custom')">🎮 实时自定义计算台</button>
            <button class="tab-btn" id="tab-timeline" onclick="switchMode('timeline')">⏱ 300 轮演化全景回放</button>
        </div>

        <!-- 自定义交互计算面板 -->
        <div id="panel-custom" class="card" style="border-color: rgba(0, 229, 255, 0.3);">
            <div class="card-title">
                <span>自定义输入变量 (A, B)</span>
                <span style="font-size: 10px; color: #64ffda;">⚡ 实时零延迟响应</span>
            </div>

            <div class="slider-group">
                <div class="slider-header">
                    <span>输入 A (源端槽位 0):</span>
                    <span id="label-custom-a" style="color: #00E5FF; font-weight: bold; font-family: monospace;">1.80</span>
                </div>
                <input type="range" class="timeline-slider" id="slider-a" min="0.2" max="3.0" step="0.05" value="1.80" oninput="onCustomInputChange()">
            </div>

            <div class="slider-group">
                <div class="slider-header">
                    <span>输入 B (源端槽位 1):</span>
                    <span id="label-custom-b" style="color: #00E5FF; font-weight: bold; font-family: monospace;">1.50</span>
                </div>
                <input type="range" class="timeline-slider" id="slider-b" min="0.2" max="3.0" step="0.05" value="1.50" oninput="onCustomInputChange()">
            </div>

            <div style="margin: 8px 0 4px 0;">
                <div style="font-size: 11px; color: #7d8b9d; margin-bottom: 5px;">快速预设典型算例：</div>
                <div style="display: flex; flex-wrap: wrap; gap: 6px;">
                    <button class="preset-btn" onclick="setPreset(1.8, 1.5)">1.8 × 1.5</button>
                    <button class="preset-btn" onclick="setPreset(2.0, 2.0)">2.0 × 2.0</button>
                    <button class="preset-btn" onclick="setPreset(1.2, 2.5)">1.2 × 2.5</button>
                    <button class="preset-btn" onclick="setPreset(1.5, 1.5)">1.5 × 1.5</button>
                    <button class="preset-btn" onclick="setPreset(1.0, 2.0)">1.0 × 2.0</button>
                </div>
            </div>
        </div>

        <!-- 演化时间控制器 (在时间轴模式下显示) -->
        <div id="panel-timeline" class="card" style="display: none;">
            <div class="card-title">
                <span>演化时间轴</span>
                <span id="epoch-label" style="color: #FF007F; font-weight: bold;">Epoch 299 / 300</span>
            </div>
            <input type="range" class="timeline-slider" id="time-slider" min="0" max="299" value="299" step="1" oninput="onSlideStep(this.value)">
            <div class="btn-row">
                <button class="btn" id="btn-play" onclick="togglePlay()">▶ 演化回放</button>
                <button class="btn" onclick="stepTime(-1)">◀ 单步</button>
                <button class="btn" onclick="stepTime(1)">单步 ▶</button>
            </div>
        </div>

        <!-- 实时计算结果面板 -->
        <div class="card">
            <div class="card-title">计算考核与输出对齐</div>
            <div class="task-panel">
                <div class="task-row">
                    <span style="color: #8892b0;">当前输入对 (A, B):</span>
                    <span style="font-family: monospace; color: #00E5FF;" id="val-inputs">A = 1.80, B = 1.50</span>
                </div>
                <div class="task-row">
                    <span style="color: #8892b0;">目标真值 (A × B):</span>
                    <span style="font-family: monospace; font-weight: bold; color: #FFD700; font-size: 14px;" id="val-target">2.700</span>
                </div>
                <div class="task-row">
                    <span style="color: #8892b0;">构形网络输出 (Drain):</span>
                    <span style="font-family: monospace; font-weight: bold; color: #00E676; font-size: 15px;" id="val-output">2.736</span>
                </div>
                <div class="task-row" style="margin-top: 4px; padding-top: 4px; border-top: 1px dashed rgba(255,255,255,0.1);">
                    <span style="color: #8892b0;">物理残差 (误差):</span>
                    <span style="font-family: monospace; font-weight: bold;" id="val-error">0.0360</span>
                </div>
                <div class="task-row">
                    <span style="color: #8892b0;">计算匹配度 (准确率):</span>
                    <span style="font-family: monospace; font-weight: bold; color: #00E5FF;" id="val-acc">98.7%</span>
                </div>
            </div>
        </div>

        <!-- 物理指标 -->
        <div class="card">
            <div class="card-title">网络构形与流阻</div>
            <div class="metric-grid">
                <div class="metric-box">
                    <div class="metric-val" id="metric-resistance" style="color: #00E5FF;">1.09</div>
                    <div class="metric-lbl">全流通阻抗 R (低阻层流态)</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val" id="metric-slots" style="color: #00E676;">24</div>
                    <div class="metric-lbl">活跃极化槽位数</div>
                </div>
            </div>
        </div>

        <!-- 演化事件记录 -->
        <div class="card">
            <div class="card-title">系统状态动态</div>
            <div class="event-ticker" id="event-ticker">
                已进入交互模式：可自由拖动 A 与 B 滑块，观察流经构形网络的瞬时响应。
            </div>
        </div>
    </div>

    <script>
        const simData = {json_str};
        const snapshots = simData.snapshots;
        const totalEpochs = snapshots.length;

        let activeMode = 'custom'; // 'custom' or 'timeline'
        let currentEpochIdx = totalEpochs - 1; // 默认使用成熟的最终轮次权重
        let customA = 1.80;
        let customB = 1.50;

        let isPlaying = false;
        let playTimer = null;

        // Three.js 场景初始化
        const container = document.getElementById('canvas-container');
        const scene = new THREE.Scene();
        scene.fog = new THREE.FogExp2(0x030508, 0.18);

        const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 100);
        camera.position.set(0, 1.8, 3.2);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;

        // 灯光
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
        scene.add(ambientLight);
        const pointLight1 = new THREE.PointLight(0x00E5FF, 2.0, 10);
        pointLight1.position.set(-2, 2, 2);
        scene.add(pointLight1);
        const pointLight2 = new THREE.PointLight(0xFF007F, 2.0, 10);
        pointLight2.position.set(2, 2, -1);
        scene.add(pointLight2);

        // 参考框
        const boxGeo = new THREE.BoxGeometry(2.4, 1.2, 1.2);
        const boxEdges = new THREE.EdgesGeometry(boxGeo);
        const boxLine = new THREE.LineSegments(boxEdges, new THREE.LineBasicMaterial({{ color: 0x1f293d, transparent: true, opacity: 0.3 }}));
        scene.add(boxLine);

        // 动态物体组
        const nodeGroup = new THREE.Group();
        const tubeGroup = new THREE.Group();
        const particleGroup = new THREE.Group();
        scene.add(nodeGroup);
        scene.add(tubeGroup);
        scene.add(particleGroup);

        // 粒子流
        const particleCount = 200;
        const particleGeo = new THREE.BufferGeometry();
        const particlePos = new Float32Array(particleCount * 3);
        const particleProgress = new Float32Array(particleCount);
        const particleTubes = new Int32Array(particleCount);

        for (let i = 0; i < particleCount; i++) {{
            particlePos[i*3] = -0.85;
            particlePos[i*3+1] = 0;
            particlePos[i*3+2] = 0;
            particleProgress[i] = Math.random();
            particleTubes[i] = -1;
        }}
        particleGeo.setAttribute('position', new THREE.BufferAttribute(particlePos, 3));
        const particleMat = new THREE.PointsMaterial({{
            color: 0x00E5FF,
            size: 0.025,
            transparent: true,
            opacity: 0.8,
            blending: THREE.AdditiveBlending
        }});
        const particles = new THREE.Points(particleGeo, particleMat);
        particleGroup.add(particles);

        // 构形网络前向推理核心
        function computeNetworkInference(a, b, epochIdx) {{
            const snap = snapshots[epochIdx];
            const w_a = {{}};
            const w_b = {{}};
            const w_out = {{}};

            snap.tubes.forEach(t => {{
                if (t.source === 0) w_a[t.target] = t.weight;
                else if (t.source === 1) w_b[t.target] = t.weight;
                else if (t.target === 2) w_out[t.source] = t.weight;
            }});

            let out = 0.0;
            const nodeActivations = {{}};
            snap.nodes.forEach(n => {{
                if (n.is_source || n.is_sink) return;
                const k = n.id;
                const wa = w_a[k] || 0.0;
                const wb = w_b[k] || 0.0;
                const u = wa * a + wb * b;
                const v = 0.25 * (u * u);
                nodeActivations[k] = v;
                const wo = w_out[k] || 0.0;
                out += v * wo;
            }});

            const target = a * b;
            const error = Math.abs(target - out);
            const acc = Math.max(0.0, 1.0 - error / Math.max(0.01, Math.abs(target))) * 100.0;

            return {{
                output: out,
                target: target,
                error: error,
                acc: acc,
                activations: nodeActivations
            }};
        }}

        function updateDisplay() {{
            const snap = snapshots[currentEpochIdx];
            let inA, inB, result;

            if (activeMode === 'custom') {{
                inA = customA;
                inB = customB;
                result = computeNetworkInference(inA, inB, currentEpochIdx);
            }} else {{
                inA = snap.a;
                inB = snap.b;
                result = {{
                    output: snap.output,
                    target: snap.target,
                    error: snap.train_error,
                    acc: Math.max(0, (1 - snap.train_error / Math.max(0.01, snap.target)) * 100),
                    activations: {{}}
                }};
            }}

            // 重绘 3D 节点
            while(nodeGroup.children.length > 0) nodeGroup.remove(nodeGroup.children[0]);
            while(tubeGroup.children.length > 0) tubeGroup.remove(tubeGroup.children[0]);

            const nodeMap = {{}};

            snap.nodes.forEach(n => {{
                let color = 0xFFD700;
                let size = 0.055;

                if (n.id === 0) {{ // Source A
                    color = 0x00E5FF;
                    size = 0.06 + Math.min(0.06, inA * 0.03);
                }} else if (n.id === 1) {{ // Source B
                    color = 0x00E5FF;
                    size = 0.06 + Math.min(0.06, inB * 0.03);
                }} else if (n.id === 2) {{ // Sink
                    color = result.error < 0.15 ? 0x00E676 : 0xFF0055;
                    size = 0.08 + Math.min(0.06, Math.abs(result.output) * 0.02);
                }} else {{
                    const act = result.activations[n.id] || 0.5;
                    size = 0.045 + Math.min(0.04, act * 0.03);
                }}

                const sphereGeo = new THREE.SphereGeometry(size, 20, 20);
                const sphereMat = new THREE.MeshStandardMaterial({{
                    color: color,
                    emissive: color,
                    emissiveIntensity: 0.5,
                    roughness: 0.2,
                    metalness: 0.8
                }});
                const mesh = new THREE.Mesh(sphereGeo, sphereMat);
                mesh.position.set(n.x, n.y, n.z);
                nodeGroup.add(mesh);
                nodeMap[n.id] = mesh.position;
            }});

            // 重绘管道
            snap.tubes.forEach(t => {{
                const p1 = nodeMap[t.source];
                const p2 = nodeMap[t.target];
                if (!p1 || !p2) return;

                const dist = p1.distanceTo(p2);
                const mid = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);

                const isNeg = t.is_inhibitory;
                const tubeColor = isNeg ? 0x00E5FF : 0xFFD700;
                const radius = Math.max(0.006, Math.min(0.032, Math.abs(t.weight) * 0.028));

                const cylGeo = new THREE.CylinderGeometry(radius, radius, dist, 12);
                const cylMat = new THREE.MeshStandardMaterial({{
                    color: tubeColor,
                    emissive: tubeColor,
                    emissiveIntensity: 0.45,
                    transparent: true,
                    opacity: 0.85
                }});
                const cyl = new THREE.Mesh(cylGeo, cylMat);
                cyl.position.copy(mid);
                cyl.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), p2.clone().sub(p1).normalize());
                tubeGroup.add(cyl);
            }});

            // 更新侧边栏界面数值
            document.getElementById('val-inputs').innerText = `A = ${{inA.toFixed(2)}}, B = ${{inB.toFixed(2)}}`;
            document.getElementById('val-target').innerText = result.target.toFixed(3);

            const outEl = document.getElementById('val-output');
            outEl.innerText = result.output.toFixed(3);
            outEl.style.color = result.error < 0.15 ? '#00E676' : '#FF0055';

            const errEl = document.getElementById('val-error');
            errEl.innerText = result.error.toFixed(4);
            errEl.style.color = result.error < 0.15 ? '#00E676' : '#FF0055';

            document.getElementById('val-acc').innerText = `${{result.acc.toFixed(1)}}%`;
            document.getElementById('metric-resistance').innerText = snap.resistance.toFixed(2);
            document.getElementById('metric-slots').innerText = snap.num_slots;
            document.getElementById('epoch-label').innerText = `Epoch ${{snap.epoch}} / ${{totalEpochs}}`;
            document.getElementById('time-slider').value = currentEpochIdx;
        }}

        function onCustomInputChange() {{
            customA = parseFloat(document.getElementById('slider-a').value);
            customB = parseFloat(document.getElementById('slider-b').value);
            document.getElementById('label-custom-a').innerText = customA.toFixed(2);
            document.getElementById('label-custom-b').innerText = customB.toFixed(2);
            updateDisplay();
        }}

        function setPreset(a, b) {{
            customA = a;
            customB = b;
            document.getElementById('slider-a').value = a;
            document.getElementById('slider-b').value = b;
            document.getElementById('label-custom-a').innerText = a.toFixed(2);
            document.getElementById('label-custom-b').innerText = b.toFixed(2);
            updateDisplay();
        }}

        function switchMode(mode) {{
            activeMode = mode;
            if (mode === 'custom') {{
                document.getElementById('tab-custom').classList.add('active');
                document.getElementById('tab-timeline').classList.remove('active');
                document.getElementById('panel-custom').style.display = 'block';
                document.getElementById('panel-timeline').style.display = 'none';
                document.getElementById('event-ticker').innerText = "当前模式：自由输入 (A, B)，实时由神经网络解出乘积并渲染物理流场。";
            }} else {{
                document.getElementById('tab-custom').classList.remove('active');
                document.getElementById('tab-timeline').classList.add('active');
                document.getElementById('panel-custom').style.display = 'none';
                document.getElementById('panel-timeline').style.display = 'block';
                document.getElementById('event-ticker').innerText = snapshots[currentEpochIdx].recent_event;
            }}
            updateDisplay();
        }}

        function onSlideStep(val) {{
            currentEpochIdx = parseInt(val);
            updateDisplay();
        }}

        function stepTime(delta) {{
            let next = currentEpochIdx + delta;
            if (next < 0) next = 0;
            if (next >= totalEpochs) next = totalEpochs - 1;
            currentEpochIdx = next;
            updateDisplay();
        }}

        function togglePlay() {{
            isPlaying = !isPlaying;
            const btn = document.getElementById('btn-play');
            if (isPlaying) {{
                btn.innerText = '⏸ 暂停演化';
                btn.style.background = 'rgba(255, 0, 127, 0.3)';
                playTimer = setInterval(() => {{
                    if (currentEpochIdx >= totalEpochs - 1) {{
                        currentEpochIdx = 0;
                    }} else {{
                        currentEpochIdx++;
                    }}
                    updateDisplay();
                }}, 70);
            }} else {{
                btn.innerText = '▶ 演化回放';
                btn.style.background = 'rgba(255, 255, 255, 0.05)';
                clearInterval(playTimer);
            }}
        }}

        // 渲染与粒子流动循环
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();

            const snap = snapshots[currentEpochIdx];
            if (snap && snap.tubes.length > 0) {{
                const posArr = particleGeo.attributes.position.array;
                const tubes = snap.tubes;
                const nodeMap = {{}};
                snap.nodes.forEach(n => {{ nodeMap[n.id] = new THREE.Vector3(n.x, n.y, n.z); }});

                for (let i = 0; i < particleCount; i++) {{
                    particleProgress[i] += 0.02;
                    if (particleProgress[i] >= 1.0 || particleTubes[i] < 0 || particleTubes[i] >= tubes.length) {{
                        particleProgress[i] = 0;
                        particleTubes[i] = Math.floor(Math.random() * tubes.length);
                    }}
                    const tube = tubes[particleTubes[i]];
                    const p1 = nodeMap[tube.source];
                    const p2 = nodeMap[tube.target];
                    if (p1 && p2) {{
                        const cur = new THREE.Vector3().lerpVectors(p1, p2, particleProgress[i]);
                        posArr[i*3] = cur.x;
                        posArr[i*3+1] = cur.y;
                        posArr[i*3+2] = cur.z;
                    }}
                }}
                particleGeo.attributes.position.needsUpdate = true;
            }}

            renderer.render(scene, camera);
        }}

        window.addEventListener('resize', () => {{
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        }});

        updateDisplay();
        animate();
    </script>
</body>
</html>
"""

paths = [
    'd:/information-boltzmann-v1/present/3d_extended_constructal_demo.html',
    'C:/Users/85750/Downloads/3d_extended_constructal_demo.html',
    'C:/Users/85750/.gemini/antigravity/brain/4d6d1761-7e96-4279-930e-c51f5f2b2b1b/3d_extended_constructal_demo.html'
]

for p in paths:
    with open(p, 'w', encoding='utf-8') as f:
        f.write(html_code)
    print(f"成功生成实时交互式 3D 演化网页: {p}")
