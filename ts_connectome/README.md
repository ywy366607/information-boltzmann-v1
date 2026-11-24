# CGB Causal Reservoir Connectome 3D (TypeScript + Three.js + WebGL/GPU)

Production-grade 3D Causal Connectome visualization of Information Boltzmann / MengLi (Cycle 2401 Causal Reservoir), engineered with **TypeScript**, **Three.js**, and custom **WebGL GPU Post-Processing Shaders**.

---

## 🎨 Visual FX Pipeline (GLSL Post-Processing)

The post-processing pipeline is implemented as a 4-pass GPU chain executed on full-screen quads without heavy external bundle dependencies:

1. **Pass 1 (Scene Render)**:
   - Renders 280 bioluminescent crystal nodes, 976 fiber-optic causal chords, optical light-pipes, and 160 dynamic flowing photons into an HDR render target (`THREE.HalfFloatType` in WebGL2).
2. **Pass 2 (Luminance Bright-Pass)**:
   - Evaluates photometric luminance $L = 0.2126R + 0.7152G + 0.0722B$ with a quadratic soft knee thresholding curve to isolate high-energy specular and emissive highlights.
3. **Pass 3 (Dual-Iteration Kawase Blur)**:
   - Rapid 9-tap Kawase cross-filtering on downsampled half-resolution targets to generate silky, continuous high-order bloom.
4. **Pass 4 (Cinematic Composite)**:
   - **Bloom Blend**: Blends the filtered light buffer back onto the scene.
   - **CineStill 800T Halation (光渗扩散)**: Simulates photochemical red-layer halation. When intense light penetrates the film emulsion, it scatters and reflects off the red backing base, generating a warm red-amber chromatic corona (`vec3(1.35, 0.40, 0.10)`) around bright nodes and fiber bundles.
   - **Photochemical Film Grain (胶片微颗粒)**: Dual-hash triangular distribution noise weighted by midtone luminance ($L \approx 0.5$). Peak grain in zone V midtones, suppressed in deep specular highlights and deep blacks, eliminating digital color banding.
   - **ACES Filmic Tone Mapping**: Industry-standard curve mapping high dynamic range into rich, non-clipping sRGB space.

---

## 📁 Project Architecture

```
ts_connectome/
├── src/
│   ├── types.ts                        # TypeScript interfaces (Node, Edge, Sector, FXParams)
│   ├── shaders/
│   │   └── CinematicPostShader.ts      # GLSL Shaders (BrightPass, KawaseBlur, Composite)
│   ├── pipeline/
│   │   └── CinematicPostProcessor.ts   # GPU RenderTarget & Post-processing controller
│   ├── CausalConnectome.ts             # 3D Scene Graph, OrbitControls, Anti-strobe Raycasting
│   └── main.ts                         # Application entry & DOM event binding
├── package.json                        # Vite + Three.js + TypeScript setup
├── tsconfig.json                       # Strict ESNext compiler configuration
└── README.md
```

---

## 🚀 Running Locally

```bash
cd ts_connectome
npm install
npm run dev
```

To compile production distribution:
```bash
npm run build
```

---

## 🌐 Instant Standalone HTML

The standalone compiled HTML bundle is always available at:
- `results/published/cgb_reservoir_circos_3d.html`
- Artifact: `C:\Users\85750\.gemini\antigravity\brain\b52ce350-543f-4727-bcda-4a922c34066a\cgb_reservoir_circos_3d.html`

Contains the complete WebGL post-processing shaders, OrbitControls, and interactive UI sliders (曝光, 辉光, 光渗, 颗粒, 3D形态, 巡游, 纯净背景).
