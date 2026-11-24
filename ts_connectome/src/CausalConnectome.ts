import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { CGBNode, CGBEdge, CGBSector } from './types';
import { CinematicPostProcessor } from './pipeline/CinematicPostProcessor';

export class CausalConnectome {
  private container: HTMLElement;
  private renderer: THREE.WebGLRenderer;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private postProcessor: CinematicPostProcessor;

  // Scene Objects
  private worldGroup: THREE.Group;
  private nodesGroup: THREE.Group;
  private aurasGroup: THREE.Group;
  private tracksGroup: THREE.Group;
  private chordsGroup: THREE.Group;
  private floorGroup: THREE.Group;
  private sectorsGroup: THREE.Group;

  private nodeObjects: THREE.Mesh[] = [];
  private auraObjects: { sprite: THREE.Sprite; baseScale: number; nodeIdx: number }[] = [];
  private chordCurves: { curve: THREE.QuadraticBezierCurve3; dynamicCurve?: THREE.QuadraticBezierCurve3; edge: CGBEdge; p0: THREE.Vector3; pMid: THREE.Vector3; p3: THREE.Vector3 }[] = [];
  private chordMeshes: THREE.Line[] = [];

  // Photons
  private photonGeo: THREE.BufferGeometry;
  private photonMat: THREE.PointsMaterial;
  private photons: { curveIdx: number; t: number; speed: number }[] = [];

  // Data
  private nodes: CGBNode[];
  private edges: CGBEdge[];
  private sectors: CGBSector[];

  // State
  private clockTime = 0;
  private isAutoRotate = true;
  private morphProgress = 0;
  private hoveredNodeIdx: number | null = null;
  private lastMouseMoveTime = 0;
  private mouse = new THREE.Vector2(-9999, -9999);
  private raycaster = new THREE.Raycaster();

  constructor(
    container: HTMLElement,
    nodes: CGBNode[],
    edges: CGBEdge[],
    sectors: CGBSector[]
  ) {
    this.container = container;
    this.nodes = nodes;
    this.edges = edges;
    this.sectors = sectors;

    // Renderer
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance'
    });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.container.appendChild(this.renderer.domElement);

    // Scene & Camera
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x02050e, 0.0035);

    this.camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.5, 2000);
    this.camera.position.set(0, 66.0, 86.0);

    // Orbit Controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.maxDistance = 350;
    this.controls.minDistance = 15;
    this.controls.target.set(0, 8, 0);
    this.controls.autoRotate = this.isAutoRotate;
    this.controls.autoRotateSpeed = 0.5;

    // Root World Group (for Zero-G Floating decoupled from OrbitControls)
    this.worldGroup = new THREE.Group();
    this.scene.add(this.worldGroup);

    this.floorGroup = new THREE.Group();
    this.sectorsGroup = new THREE.Group();
    this.nodesGroup = new THREE.Group();
    this.aurasGroup = new THREE.Group();
    this.tracksGroup = new THREE.Group();
    this.chordsGroup = new THREE.Group();

    this.worldGroup.add(this.floorGroup);
    this.worldGroup.add(this.sectorsGroup);
    this.worldGroup.add(this.nodesGroup);
    this.worldGroup.add(this.aurasGroup);
    this.worldGroup.add(this.tracksGroup);
    this.worldGroup.add(this.chordsGroup);

    // Lights
    this.setupLights();

    // Scene Geometry & Shaders
    this.buildFloorAndSectors();
    this.buildNodes();
    this.buildLightPipes();
    this.buildChords();
    this.buildPhotonStream();

    // Post-Processing Pipeline (Bloom + Halation + Film Grain)
    this.postProcessor = new CinematicPostProcessor(
      this.renderer,
      window.innerWidth,
      window.innerHeight
    );

    // Event Listeners
    this.bindEvents();

    // Start Loop
    this.animate = this.animate.bind(this);
    requestAnimationFrame(this.animate);
  }

  private setupLights(): void {
    const ambient = new THREE.AmbientLight(0xffffff, 0.32);
    this.scene.add(ambient);

    const p1 = new THREE.PointLight(0x38bdf8, 1.2, 300);
    p1.position.set(30, 70, 40);
    this.scene.add(p1);

    const p2 = new THREE.PointLight(0xa855f7, 0.9, 300);
    p2.position.set(-40, 50, -40);
    this.scene.add(p2);

    const pCenter = new THREE.PointLight(0xfbbf24, 0.55, 150);
    pCenter.position.set(0, 15, 0);
    this.scene.add(pCenter);
  }

  private createGlowTexture(colorHex: string): THREE.CanvasTexture {
    const size = 128;
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d')!;
    const grad = ctx.createRadialGradient(size/2, size/2, 0, size/2, size/2, size/2);
    
    // Parse hex
    const hex = colorHex.replace('#', '');
    const num = parseInt(hex.length === 3 ? hex.split('').map(x=>x+x).join('') : hex, 16);
    const r = (num >> 16) & 255;
    const g = (num >> 8) & 255;
    const b = num & 255;

    grad.addColorStop(0, `rgba(${r}, ${g}, ${b}, 0.9)`);
    grad.addColorStop(0.25, `rgba(${r}, ${g}, ${b}, 0.65)`);
    grad.addColorStop(0.6, `rgba(${r}, ${g}, ${b}, 0.12)`);
    grad.addColorStop(1.0, `rgba(${r}, ${g}, ${b}, 0.0)`);
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, size, size);
    return new THREE.CanvasTexture(canvas);
  }

  private buildFloorAndSectors(): void {
    [25, 40, 50.0, 52.5, 55.0, 57.5, 60.0, 64.0].forEach(r => {
      const pts: THREE.Vector3[] = [];
      for (let a = 0; a <= Math.PI * 2.01; a += Math.PI / 64) {
        pts.push(new THREE.Vector3(r * Math.cos(a), 0, r * Math.sin(a)));
      }
      const ringGeo = new THREE.BufferGeometry().setFromPoints(pts);
      const ringMat = new THREE.LineBasicMaterial({
        color: r === 50.0 ? 0x38bdf8 : 0x1e293b,
        transparent: true,
        opacity: r === 50.0 ? 0.30 : 0.15
      });
      this.floorGroup.add(new THREE.Line(ringGeo, ringMat));
    });

    for (let deg = 0; deg < 360; deg += 45) {
      const rad = deg * Math.PI / 180;
      const rayGeo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(45 * Math.cos(rad), 0, 45 * Math.sin(rad)),
        new THREE.Vector3(65 * Math.cos(rad), 0, 65 * Math.sin(rad))
      ]);
      const rayMat = new THREE.LineBasicMaterial({ color: 0x1e293b, transparent: true, opacity: 0.18 });
      this.floorGroup.add(new THREE.Line(rayGeo, rayMat));
    }

    this.sectors.forEach(s => {
      if (s.angle_start === undefined || s.angle_end === undefined) return;
      const arcShape = new THREE.Shape();
      arcShape.absarc(0, 0, 50.5, s.angle_start, s.angle_end, false);
      arcShape.absarc(0, 0, 48.5, s.angle_end, s.angle_start, true);
      arcShape.closePath();

      const geom = new THREE.ShapeGeometry(arcShape, 48);
      const mat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(s.color),
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.20
      });
      const mesh = new THREE.Mesh(geom, mat);
      mesh.rotation.x = Math.PI / 2;
      mesh.position.y = 0.02;
      this.sectorsGroup.add(mesh);
    });
  }

  private buildNodes(): void {
    const baseNodeGeom = new THREE.SphereGeometry(0.70, 18, 18);
    const harmonicNodeGeom = new THREE.SphereGeometry(1.15, 20, 20);
    const topRwNodeGeom = new THREE.SphereGeometry(1.35, 20, 20);

    const glowTexPurple = this.createGlowTexture('#c084fc');
    const glowTexGold = this.createGlowTexture('#fbbf24');
    const glowTexRose = this.createGlowTexture('#f43f5e');

    this.nodes.forEach((d, idx) => {
      let color = new THREE.Color(d.gen_color);
      let geom = baseNodeGeom;
      let emissiveIntensity = 0.28;
      let auraTex: THREE.CanvasTexture | null = null;
      let auraScale = 3.2;

      if (d.is_top_rw) {
        color = new THREE.Color(0xc084fc);
        geom = topRwNodeGeom;
        emissiveIntensity = 0.65;
        auraTex = glowTexPurple;
        auraScale = 4.8;
      } else if (d.is_harmonic) {
        color = new THREE.Color(0xfbbf24);
        geom = harmonicNodeGeom;
        emissiveIntensity = 0.55;
        auraTex = glowTexGold;
        auraScale = 4.2;
      } else if (d.is_quartet) {
        color = new THREE.Color(0xf43f5e);
        emissiveIntensity = 0.45;
        auraTex = glowTexRose;
        auraScale = 3.4;
      }

      const mat = new THREE.MeshStandardMaterial({
        color: color,
        emissive: color,
        emissiveIntensity: emissiveIntensity,
        roughness: 0.15,
        metalness: 0.20,
        transparent: true,
        opacity: 0.88
      });

      const mesh = new THREE.Mesh(geom, mat);
      mesh.position.set(d.circos_pos[0], d.circos_pos[1], d.circos_pos[2]);
      mesh.userData = { nodeIndex: idx, data: d };
      this.nodesGroup.add(mesh);
      this.nodeObjects.push(mesh);

      if (auraTex) {
        const spriteMat = new THREE.SpriteMaterial({
          map: auraTex,
          blending: THREE.AdditiveBlending,
          transparent: true,
          opacity: 0.45,
          depthWrite: false
        });
        const sprite = new THREE.Sprite(spriteMat);
        sprite.scale.set(auraScale, auraScale, 1);
        sprite.position.copy(mesh.position);
        this.aurasGroup.add(sprite);
        this.auraObjects.push({ sprite, baseScale: auraScale, nodeIdx: idx });
      }
    });
  }

  private buildLightPipes(): void {
    const pipeGeom = new THREE.CylinderGeometry(0.16, 0.20, 1.0, 14);
    pipeGeom.translate(0, 0.5, 0);
    const capGeom = new THREE.SphereGeometry(0.24, 12, 12);

    this.nodes.forEach(d => {
      // Track 1: Generation
      const genH = d.gen * 1.35;
      const t1Mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(d.gen_color),
        emissive: new THREE.Color(d.gen_color),
        emissiveIntensity: 0.32,
        roughness: 0.2,
        transparent: true,
        opacity: 0.72
      });
      const p1 = new THREE.Mesh(pipeGeom, t1Mat);
      p1.position.set(52.5 * Math.cos(d.angle), 0, 52.5 * Math.sin(d.angle));
      p1.scale.set(1, genH, 1);
      this.tracksGroup.add(p1);

      const c1 = new THREE.Mesh(capGeom, t1Mat);
      c1.position.set(52.5 * Math.cos(d.angle), genH, 52.5 * Math.sin(d.angle));
      this.tracksGroup.add(c1);

      // Track 2: Degree
      const degH = Math.max(0.3, (d.degree / 22.0) * 7.5);
      const t2Mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(0x0284c7),
        emissive: new THREE.Color(0x38bdf8),
        emissiveIntensity: 0.35,
        roughness: 0.2,
        transparent: true,
        opacity: 0.72
      });
      const p2 = new THREE.Mesh(pipeGeom, t2Mat);
      p2.position.set(55.0 * Math.cos(d.angle), 0, 55.0 * Math.sin(d.angle));
      p2.scale.set(1, degH, 1);
      this.tracksGroup.add(p2);

      const c2 = new THREE.Mesh(capGeom, t2Mat);
      c2.position.set(55.0 * Math.cos(d.angle), degH, 55.0 * Math.sin(d.angle));
      this.tracksGroup.add(c2);

      // Track 3: RW Betweenness
      const rwH = Math.max(0.2, (d.rw_betweenness / 0.125) * 12.0);
      const t3Color = d.is_top_rw ? 0xd946ef : 0xa855f7;
      const t3Mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(t3Color),
        emissive: new THREE.Color(t3Color),
        emissiveIntensity: d.is_top_rw ? 0.65 : 0.38,
        roughness: 0.2,
        transparent: true,
        opacity: d.is_top_rw ? 0.90 : 0.72
      });
      const p3 = new THREE.Mesh(pipeGeom, t3Mat);
      p3.position.set(57.5 * Math.cos(d.angle), 0, 57.5 * Math.sin(d.angle));
      p3.scale.set(d.is_top_rw ? 1.4 : 1.0, rwH, d.is_top_rw ? 1.4 : 1.0);
      this.tracksGroup.add(p3);

      const c3 = new THREE.Mesh(capGeom, t3Mat);
      c3.position.set(57.5 * Math.cos(d.angle), rwH, 57.5 * Math.sin(d.angle));
      this.tracksGroup.add(c3);

      // Track 4: Episodes
      const supH = Math.max(0.25, (d.supports_count / 8.0) * 4.5);
      const t4Mat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(0x10b981),
        emissive: new THREE.Color(0x34d399),
        emissiveIntensity: 0.35,
        roughness: 0.2,
        transparent: true,
        opacity: 0.72
      });
      const p4 = new THREE.Mesh(pipeGeom, t4Mat);
      p4.position.set(60.0 * Math.cos(d.angle), 0, 60.0 * Math.sin(d.angle));
      p4.scale.set(1, supH, 1);
      this.tracksGroup.add(p4);

      const c4 = new THREE.Mesh(capGeom, t4Mat);
      c4.position.set(60.0 * Math.cos(d.angle), supH, 60.0 * Math.sin(d.angle));
      this.tracksGroup.add(c4);
    });
  }

  private buildChords(): void {
    this.edges.forEach(e => {
      const uNode = this.nodes[e.u];
      const vNode = this.nodes[e.v];
      const maxArch = 6.0 + e.span * 38.0;

      const p0 = new THREE.Vector3(uNode.circos_pos[0], uNode.circos_pos[1], uNode.circos_pos[2]);
      const p3 = new THREE.Vector3(vNode.circos_pos[0], vNode.circos_pos[1], vNode.circos_pos[2]);

      const centerPull = 0.52 - e.span * 0.32;
      const midX = (p0.x + p3.x) * centerPull;
      const midZ = (p0.z + p3.z) * centerPull;
      const pMid = new THREE.Vector3(midX, maxArch, midZ);

      const curve = new THREE.QuadraticBezierCurve3(p0, pMid, p3);
      this.chordCurves.push({ curve, edge: e, p0, pMid, p3 });

      const points = curve.getPoints(28);
      const geom = new THREE.BufferGeometry().setFromPoints(points);

      let col = new THREE.Color(0x0284c7);
      let opacity = 0.14;
      if (e.is_harmonic) {
        col = new THREE.Color(0xfbbf24);
        opacity = 0.65;
      } else if (e.is_bottleneck) {
        col = new THREE.Color(0xc084fc);
        opacity = 0.32;
      }

      const lineMat = new THREE.LineBasicMaterial({
        color: col,
        transparent: true,
        opacity: opacity,
        blending: THREE.AdditiveBlending,
        depthWrite: false
      });

      const line = new THREE.Line(geom, lineMat);
      line.userData = { edge: e, baseCol: col, baseOpacity: opacity };
      this.chordsGroup.add(line);
      this.chordMeshes.push(line);
    });
  }

  private buildPhotonStream(): void {
    const PHOTON_COUNT = 160;
    this.photonGeo = new THREE.BufferGeometry();
    const positions = new Float32Array(PHOTON_COUNT * 3);
    const colors = new Float32Array(PHOTON_COUNT * 3);

    const activeIndices: number[] = [];
    this.chordCurves.forEach((c, idx) => {
      if (c.edge.is_harmonic) {
        for (let k = 0; k < 6; k++) activeIndices.push(idx);
      } else if (c.edge.is_bottleneck) {
        for (let k = 0; k < 2; k++) activeIndices.push(idx);
      } else if (idx % 7 === 0) {
        activeIndices.push(idx);
      }
    });

    for (let p = 0; p < PHOTON_COUNT; p++) {
      const curveIdx = activeIndices[p % activeIndices.length];
      const t = Math.random();
      const speed = 0.003 + Math.random() * 0.005;
      const isHarmonic = this.chordCurves[curveIdx].edge.is_harmonic;
      const isBottleneck = this.chordCurves[curveIdx].edge.is_bottleneck;

      const col = isHarmonic ? new THREE.Color(0xfbbf24) : (isBottleneck ? new THREE.Color(0xc084fc) : new THREE.Color(0x38bdf8));
      this.photons.push({ curveIdx, t, speed });

      colors[p * 3] = col.r;
      colors[p * 3 + 1] = col.g;
      colors[p * 3 + 2] = col.b;
    }

    this.photonGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    this.photonGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const glowTexWhite = this.createGlowTexture('#ffffff');
    this.photonMat = new THREE.PointsMaterial({
      size: 2.3,
      map: glowTexWhite,
      vertexColors: true,
      blending: THREE.AdditiveBlending,
      transparent: true,
      opacity: 0.75,
      depthWrite: false
    });

    const photonPoints = new THREE.Points(this.photonGeo, this.photonMat);
    this.worldGroup.add(photonPoints);
  }

  public setMorph(t: number): void {
    this.morphProgress = t;
    this.nodes.forEach((d, idx) => {
      const mesh = this.nodeObjects[idx];
      mesh.position.x = (1 - t) * d.circos_pos[0] + t * d.brain_pos[0];
      mesh.position.y = (1 - t) * d.circos_pos[1] + t * d.brain_pos[1];
      mesh.position.z = (1 - t) * d.circos_pos[2] + t * d.brain_pos[2];
    });

    this.auraObjects.forEach(a => {
      a.sprite.position.copy(this.nodeObjects[a.nodeIdx].position);
    });

    this.sectorsGroup.position.y = (1 - t) * 0.02 - t * 40.0;
    this.tracksGroup.position.y = (1 - t) * 0.0 - t * 40.0;
    this.floorGroup.position.y = (1 - t) * 0.0 - t * 40.0;

    this.sectorsGroup.children.forEach(c => ((c as THREE.Mesh).material as THREE.Material).opacity = (1 - t) * 0.20);
    this.tracksGroup.children.forEach(c => {
      const m = (c as THREE.Mesh).material as THREE.Material;
      if (m) m.opacity = (1 - t) * 0.72;
    });

    this.chordCurves.forEach((item, idx) => {
      const e = item.edge;
      const uPos = this.nodeObjects[e.u].position;
      const vPos = this.nodeObjects[e.v].position;

      const circosMid = item.pMid;
      const brainMid = new THREE.Vector3().addVectors(uPos, vPos).multiplyScalar(0.5);
      const currMid = new THREE.Vector3().lerpVectors(circosMid, brainMid, t);

      const dynamicCurve = new THREE.QuadraticBezierCurve3(uPos, currMid, vPos);
      item.dynamicCurve = dynamicCurve;
      this.chordMeshes[idx].geometry.setFromPoints(dynamicCurve.getPoints(24));
    });
  }

  public setExposure(exposure: number): void {
    this.postProcessor.params.exposure = exposure;
  }

  public setBloom(intensity: number): void {
    this.postProcessor.params.bloomIntensity = intensity;
  }

  public setHalation(intensity: number): void {
    this.postProcessor.params.halationIntensity = intensity;
  }

  public setGrain(intensity: number): void {
    this.postProcessor.params.grainIntensity = intensity;
  }

  private bindEvents(): void {
    window.addEventListener('resize', () => {
      const w = window.innerWidth;
      const h = window.innerHeight;
      if (w > 10 && h > 10) {
        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);
        this.postProcessor.setSize(w, h);
      }
    });

    window.addEventListener('mousemove', e => {
      this.mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
      this.mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
      this.lastMouseMoveTime = performance.now();
    });

    const clearHover = () => {
      this.mouse.set(-9999, -9999);
      if (this.hoveredNodeIdx !== null) {
        this.hoveredNodeIdx = null;
        this.resetChordHighlight();
      }
    };

    window.addEventListener('mouseleave', clearHover);
    window.addEventListener('blur', clearHover);
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) clearHover();
    });
  }

  private checkHover(): void {
    if (this.controls.autoRotate && performance.now() - this.lastMouseMoveTime > 1200) {
      if (this.hoveredNodeIdx !== null) {
        this.hoveredNodeIdx = null;
        this.resetChordHighlight();
      }
      return;
    }

    if (this.mouse.x < -1 || this.mouse.x > 1 || this.mouse.y < -1 || this.mouse.y > 1) return;

    this.raycaster.setFromCamera(this.mouse, this.camera);
    const intersects = this.raycaster.intersectObjects(this.nodeObjects);

    if (intersects.length > 0) {
      const idx = intersects[0].object.userData.nodeIndex as number;
      if (this.hoveredNodeIdx !== idx) {
        this.hoveredNodeIdx = idx;
        this.highlightNode(idx);
      }
    } else {
      if (this.hoveredNodeIdx !== null) {
        this.hoveredNodeIdx = null;
        this.resetChordHighlight();
      }
    }
  }

  private highlightNode(idx: number): void {
    this.chordMeshes.forEach(m => {
      const e = m.userData.edge as CGBEdge;
      const mat = m.material as THREE.LineBasicMaterial;
      if (e.u === idx || e.v === idx) {
        mat.opacity = 0.95;
        mat.color.setHex(0x38bdf8);
      } else {
        mat.opacity = 0.02;
      }
    });
  }

  private resetChordHighlight(): void {
    this.chordMeshes.forEach(m => {
      const u = m.userData;
      const mat = m.material as THREE.LineBasicMaterial;
      mat.opacity = u.baseOpacity;
      mat.color.copy(u.baseCol);
    });
  }

  private animate(): void {
    requestAnimationFrame(this.animate);
    this.clockTime += 0.02;

    // Zero-G Gentle Floating (Applies cleanly to worldGroup.position.y)
    if (this.controls.autoRotate) {
      this.worldGroup.position.y = Math.sin(this.clockTime * 0.35) * 1.5;
    }

    this.controls.update();
    this.checkHover();

    // Breathing pulsation for auras
    this.auraObjects.forEach((a, i) => {
      const s = a.baseScale * (1.0 + 0.08 * Math.sin(this.clockTime * 2.5 + i));
      a.sprite.scale.set(s, s, 1);
    });

    // Animate Photons
    const posArr = this.photonGeo.attributes.position.array as Float32Array;
    this.photons.forEach((p, idx) => {
      p.t += p.speed;
      if (p.t > 1.0) p.t = 0.0;
      const curveItem = this.chordCurves[p.curveIdx];
      const curve = curveItem.dynamicCurve || curveItem.curve;
      const pt = curve.getPoint(p.t);
      posArr[idx * 3] = pt.x;
      posArr[idx * 3 + 1] = pt.y;
      posArr[idx * 3 + 2] = pt.z;
    });
    this.photonGeo.attributes.position.needsUpdate = true;

    // GPU Cinematic Post-Processing Pass (Scene + Bloom + CineStill Halation + Film Grain)
    this.postProcessor.render(this.scene, this.camera, this.clockTime);
  }
}
