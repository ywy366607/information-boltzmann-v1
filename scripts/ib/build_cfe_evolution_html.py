import json

with open('C:/Users/85750/.gemini/antigravity/brain/4d6d1761-7e96-4279-930e-c51f5f2b2b1b/scratch/cfe_stream_evolution_data.json') as f:
    cfe_data = json.load(f)

json_str = json.dumps(cfe_data)

html_code = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>构形场引擎 (CFE) - 双层神经生物流形模拟器</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: #05070c;
            color: #dbe4ef;
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
            background: rgba(10, 14, 22, 0.96);
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
            background: linear-gradient(135deg, #FFD700, #00E5FF, #FF0055);
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
            background: #FFD700;
            cursor: pointer;
            box-shadow: 0 0 12px #FFD700;
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
            background: rgba(255, 215, 0, 0.15);
            border-color: #FFD700;
            color: #FFD700;
        }}
        .btn.active {{
            background: #FFD700;
            color: #080a10;
            border-color: #FFD700;
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
        .badge-core {{
            background: rgba(255, 215, 0, 0.15);
            color: #FFD700;
            border: 1px solid #FFD700;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
        }}
        .badge-spine {{
            background: rgba(0, 229, 255, 0.15);
            color: #00E5FF;
            border: 1px solid #00E5FF;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
        }}
        .event-ticker {{
            font-family: Consolas, monospace;
            font-size: 11px;
            color: #64ffda;
            background: rgba(0, 0, 0, 0.5);
            padding: 8px 10px;
            border-radius: 6px;
            max-height: 60px;
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
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="canvas-container">
        <div id="hud">
            <b>双层神经本体论 (Two-Tier Neural Ontology)</b><br>
            • 🌟 <span class="badge-core">永久核心胞体 (Core Soma)</span>: <b>终生永存</b> (带金色光晕)，不可被 Merge 销毁<br>
            • 🌿 <span class="badge-spine">可塑树突棘 (Plastic Spines)</span>: 随非线性阻力<b>萌发(Δ)</b>与<b>收缩并合(μ)</b><br>
            • <b>物理流形</b>: 质心归零 + 椭球体空间有界势能，彻底消除了纵向漂移失真！
        </div>
    </div>

    <div id="sidebar">
        <div>
            <h1>CFE 双层神经流形架构</h1>
            <div class="subtitle">终生骨干胞体 • 动态树突分枝 • Hopf 双代数</div>
        </div>

        <!-- 演化时间控制器 -->
        <div class="card">
            <div class="card-title">
                <span>演化时钟</span>
                <span id="step-label" style="color: #FFD700; font-weight: bold;">Step 0 / 180</span>
            </div>
            <input type="range" class="timeline-slider" id="time-slider" min="0" max="179" value="0" step="1" oninput="onSlideStep(this.value)">
            <div class="btn-row">
                <button class="btn" id="btn-play" onclick="togglePlay()">▶ 连续演化流回放</button>
                <button class="btn" onclick="stepTime(-1)">◀ 单步</button>
                <button class="btn" onclick="stepTime(1)">单步 ▶</button>
            </div>
        </div>

        <!-- 双层神经代谢指标 -->
        <div class="card">
            <div class="card-title">神经元代谢与构形指标</div>
            <div class="metric-grid">
                <div class="metric-box">
                    <div class="metric-val" id="metric-core" style="color: #FFD700;">6</div>
                    <div class="metric-lbl">🌟 永久核心胞体 (永不灭)</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val" id="metric-spines" style="color: #00E5FF;">0</div>
                    <div class="metric-lbl">🌿 动态可塑分枝 (自发增减)</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val" id="metric-hopf" style="color: #76FF03;">0.21</div>
                    <div class="metric-lbl">Hopf 抵消残差 (→0)</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val" id="metric-resistance" style="color: #FF0055;">0.60</div>
                    <div class="metric-lbl">全局构形阻力 R</div>
                </div>
            </div>
        </div>

        <!-- 输入流状态 -->
        <div class="card">
            <div class="card-title">当前非线性外部驱动流</div>
            <div style="background: rgba(0,0,0,0.4); border-left: 3px solid #FFD700; padding: 10px; border-radius: 0 6px 6px 0; font-size: 12px;">
                <div id="stage-name" style="font-weight: bold; color: #FFD700; margin-bottom: 4px;">阶段 1: 谐波线性加法流</div>
                <div style="color: #8892b0; font-family: monospace;" id="stream-vals">
                    A: 0.00 | B: 1.00 | 耦合: 1.00
                </div>
            </div>
        </div>

        <!-- 事件日志 -->
        <div class="card">
            <div class="card-title">Hopf 算子事件 (Δ 裂变 / μ 并合)</div>
            <div class="event-ticker" id="event-log">Hopf 真空稳态基底就绪...</div>
        </div>

        <!-- 真实大脑代谢原理说明 -->
        <div class="card" style="font-size: 11px; line-height: 1.6; color: #8892b0;">
            <div class="card-title" style="margin-bottom: 6px;">神经生物代谢真实机制</div>
            1. <b>神经元胞体是不可再生的后有丝分裂细胞 (Post-mitotic)</b>，一旦发育完成，与个体同生共死，绝不能随意合并消失；<br>
            2. <b>真正动态代谢的是突触与树突棘 (Dendritic Spines)</b>：当非线性流冲击时，母体胞体沿切向萌发出新的树突棘 (Δ)；冗余时<b>树突棘缩回母体并合 (μ)</b>；<br>
            3. <b>保护核心地基</b>：核心 6 个骨干胞体（带金光）终生锚定坐标系，系统既有永恒的记忆锚点，又有适应未知的无限伸缩力！
        </div>
    </div>

    <script>
        const cfeData = {json_str};
        const snapshots = cfeData.snapshots;
        let currentStep = 0;
        let isPlaying = false;
        let playTimer = null;

        const container = document.getElementById('canvas-container');
        const scene = new THREE.Scene();
        scene.fog = new THREE.FogExp2(0x05070c, 0.4);

        const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 100);
        camera.position.set(0, 0.7, 2.3);

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

        let slotsGroup = new THREE.Group();
        scene.add(slotsGroup);
        let edgesMesh = null;

        function updateSceneForStep(stepIdx) {{
            const snap = snapshots[stepIdx];
            if (!snap) return;

            while (slotsGroup.children.length > 0) {{
                const obj = slotsGroup.children[0];
                slotsGroup.remove(obj);
            }}

            // 渲染当前槽位
            snap.nodes.forEach(node => {{
                const isCore = node.is_persistent;
                // 核心骨干胞体更大更显著
                const radius = isCore
                    ? Math.max(0.055, Math.min(0.09, 0.055 + (node.energy - 1.0) * 0.015))
                    : Math.max(0.035, Math.min(0.065, 0.035 + (node.energy - 1.0) * 0.01));

                const geom = new THREE.SphereGeometry(radius, 16, 16);
                const col = new THREE.Color(`hsl(${{node.hue}}, 100%, 55%)`);
                const mat = new THREE.MeshBasicMaterial({{ color: col }});
                const sphere = new THREE.Mesh(geom, mat);

                sphere.position.set(node.x * 1.3, node.y * 1.3, node.z * 1.3);
                slotsGroup.add(sphere);

                // 核心永久神经元带有金色光晕双环 (象征终生恒星)
                if (isCore) {{
                    const coreHaloGeom = new THREE.RingGeometry(radius * 1.3, radius * 1.6, 20);
                    const coreHaloMat = new THREE.MeshBasicMaterial({{
                        color: 0xFFD700,
                        transparent: true,
                        opacity: 0.85,
                        side: THREE.DoubleSide
                    }});
                    const halo = new THREE.Mesh(coreHaloGeom, coreHaloMat);
                    halo.position.copy(sphere.position);
                    halo.lookAt(camera.position);
                    slotsGroup.add(halo);
                }}

                // 不应期呼吸光环
                if (node.refractory > 0.1) {{
                    const ringGeom = new THREE.RingGeometry(radius * 1.25, radius * 1.45, 16);
                    const ringMat = new THREE.MeshBasicMaterial({{
                        color: 0xffffff,
                        transparent: true,
                        opacity: node.refractory * 0.5,
                        side: THREE.DoubleSide
                    }});
                    const ring = new THREE.Mesh(ringGeom, ringMat);
                    ring.position.copy(sphere.position);
                    ring.lookAt(camera.position);
                    slotsGroup.add(ring);
                }}
            }});

            // 渲染导通连线 W_ij
            if (edgesMesh) scene.remove(edgesMesh);

            const linePositions = [];
            const lineColors = [];

            const nodeMap = {{}};
            snap.nodes.forEach(n => {{ nodeMap[n.id] = n; }});

            snap.edges.forEach(e => {{
                const n1 = nodeMap[e.source];
                const n2 = nodeMap[e.target];
                if (!n1 || !n2) return;

                linePositions.push(n1.x * 1.3, n1.y * 1.3, n1.z * 1.3);
                linePositions.push(n2.x * 1.3, n2.y * 1.3, n2.z * 1.3);

                const c1 = new THREE.Color(`hsl(${{n1.hue}}, 90%, 50%)`);
                const c2 = new THREE.Color(`hsl(${{n2.hue}}, 90%, 50%)`);

                lineColors.push(c1.r, c1.g, c1.b);
                lineColors.push(c2.r, c2.g, c2.b);
            }});

            if (linePositions.length > 0) {{
                const lineGeom = new THREE.BufferGeometry();
                lineGeom.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
                lineGeom.setAttribute('color', new THREE.Float32BufferAttribute(lineColors, 3));

                const lineMat = new THREE.LineBasicMaterial({{
                    vertexColors: true,
                    transparent: true,
                    opacity: 0.65,
                    blending: THREE.AdditiveBlending
                }});

                edgesMesh = new THREE.LineSegments(lineGeom, lineMat);
                scene.add(edgesMesh);
            }}

            // 更新 UI 面板
            document.getElementById('time-slider').value = stepIdx;
            document.getElementById('step-label').innerText = `Step ${{snap.step}} / 180`;
            document.getElementById('stage-name').innerText = snap.stage;
            document.getElementById('stream-vals').innerText =
                `A: ${{snap.input_vals[0].toFixed(2)}} | B: ${{snap.input_vals[1].toFixed(2)}} | 耦合: ${{snap.input_vals[2].toFixed(2)}}`;

            document.getElementById('metric-core').innerText = snap.num_core || 6;
            document.getElementById('metric-spines').innerText = snap.num_spines || (snap.num_slots - 6);
            document.getElementById('metric-hopf').innerText = snap.hopf_residual.toFixed(3);
            document.getElementById('metric-resistance').innerText = snap.resistance.toFixed(2);

            if (snap.recent_event) {{
                document.getElementById('event-log').innerText = snap.recent_event;
            }}
        }}

        function onSlideStep(val) {{
            currentStep = parseInt(val);
            updateSceneForStep(currentStep);
        }}

        function stepTime(delta) {{
            let next = currentStep + delta;
            if (next >= 0 && next < snapshots.length) {{
                currentStep = next;
                updateSceneForStep(currentStep);
            }}
        }}

        function togglePlay() {{
            isPlaying = !isPlaying;
            const btn = document.getElementById('btn-play');
            if (isPlaying) {{
                btn.innerText = '⏸ 暂停演化流';
                btn.classList.add('active');
                playTimer = setInterval(() => {{
                    currentStep = (currentStep + 1) % snapshots.length;
                    updateSceneForStep(currentStep);
                }}, 75);
            }} else {{
                btn.innerText = '▶ 连续演化流回放';
                btn.classList.remove('active');
                clearInterval(playTimer);
            }}
        }}

        updateSceneForStep(0);

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

target_html = 'C:/Users/85750/.gemini/antigravity/brain/4d6d1761-7e96-4279-930e-c51f5f2b2b1b/3d_cfe_stream_evolution_demo.html'
with open(target_html, 'w', encoding='utf-8') as f:
    f.write(html_code)

print(f"Updated 3D CFE Two-Tier HTML built at {target_html}")
