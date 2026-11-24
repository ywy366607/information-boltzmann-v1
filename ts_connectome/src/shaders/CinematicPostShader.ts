/**
 * GLSL Shaders for Cinematic Post-Processing:
 * - Dual-Threshold Luminance Bright-Pass
 * - Dual Kawase / Tent Blur Pass
 * - Cinematic Composite Shader (Bloom + CineStill 800T Halation + Midtone Film Grain + ACES Filmic Tone Mapping)
 */

export const BrightPassShader = {
  uniforms: {
    tDiffuse: { value: null },
    uThreshold: { value: 0.35 },
    uKnee: { value: 0.15 }
  },
  vertexShader: /* glsl */ `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = vec4(position.xy, 0.0, 1.0);
    }
  `,
  fragmentShader: /* glsl */ `
    uniform sampler2D tDiffuse;
    uniform float uThreshold;
    uniform float uKnee;
    varying vec2 vUv;

    void main() {
      vec4 texel = texture2D(tDiffuse, vUv);
      float luma = dot(texel.rgb, vec3(0.2126, 0.7152, 0.0722));
      
      // Soft knee thresholding
      float soft = clamp((luma - uThreshold + uKnee) / (2.0 * uKnee + 1e-4), 0.0, 1.0);
      float weight = max(soft * soft, step(uThreshold, luma));
      
      gl_FragColor = vec4(texel.rgb * weight, 1.0);
    }
  `
};

export const KawaseBlurShader = {
  uniforms: {
    tDiffuse: { value: null },
    uResolution: { value: null },
    uOffset: { value: 1.0 }
  },
  vertexShader: /* glsl */ `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = vec4(position.xy, 0.0, 1.0);
    }
  `,
  fragmentShader: /* glsl */ `
    uniform sampler2D tDiffuse;
    uniform vec2 uResolution;
    uniform float uOffset;
    varying vec2 vUv;

    void main() {
      vec2 texel = 1.0 / uResolution;
      vec2 d = texel * uOffset;

      // 9-Tap Kawase Cross Filter
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
    }
  `
};

export const CinematicCompositeShader = {
  uniforms: {
    tDiffuse: { value: null },
    tBloom: { value: null },
    uResolution: { value: null },
    uTime: { value: 0.0 },
    uExposure: { value: 0.65 },
    uBloomIntensity: { value: 0.85 },
    uHalationIntensity: { value: 0.65 },
    uHalationSpread: { value: 0.005 },
    uGrainIntensity: { value: 0.035 }
  },
  vertexShader: /* glsl */ `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = vec4(position.xy, 0.0, 1.0);
    }
  `,
  fragmentShader: /* glsl */ `
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
    vec3 ACESFilm(vec3 x) {
      float a = 2.51;
      float b = 0.03;
      float c = 2.43;
      float d = 0.59;
      float e = 0.14;
      return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
    }

    // High-speed pseudo-random hash
    float hash21(vec2 p) {
      p = fract(p * vec2(234.34, 435.345));
      p += dot(p, p + 34.23);
      return fract(p.x * p.y);
    }

    // Organic triangular distribution film grain
    float filmGrain(vec2 uv, float time) {
      vec2 seed = uv * uResolution * 0.85 + vec2(time * 45.0, time * 85.0);
      float h1 = hash21(seed);
      float h2 = hash21(seed + vec2(1.7, 4.3));
      return (h1 + h2) * 0.5 - 0.5; // Centered at 0, soft triangular roll-off
    }

    void main() {
      vec4 baseColor = texture2D(tDiffuse, vUv);
      vec3 sceneColor = baseColor.rgb;

      // 1. Bloom Sampling
      vec3 bloom = texture2D(tBloom, vUv).rgb * uBloomIntensity;

      // 2. Cinematic Halation (CineStill 800T warm red-amber scatter)
      vec2 hDir = (vUv - vec2(0.5)) * uHalationSpread;
      vec3 hSample = texture2D(tBloom, vUv - hDir * 1.5).rgb * 0.5 +
                     texture2D(tBloom, vUv + hDir * 1.5).rgb * 0.5;
      
      // Photochemical red-layer halation curve: warm reddish-orange corona
      vec3 halationTint = vec3(1.35, 0.40, 0.10);
      vec3 halation = pow(hSample, vec3(1.15)) * halationTint * uHalationIntensity;

      // Composite Light Energy
      vec3 composite = sceneColor + bloom + halation;

      // 3. Sensor Exposure Scaling
      composite *= (uExposure / 0.65);

      // 4. Physical Film Grain (Luminance Midtone Weighted)
      float luma = dot(composite, vec3(0.2126, 0.7152, 0.0722));
      // Peak at zone V (18% gray / midtone 0.5), subdued at deep black (0.0) and blown highlights (1.0)
      float grainWeight = 1.0 - 3.8 * pow(clamp(luma, 0.0, 1.0) - 0.5, 2.0);
      grainWeight = clamp(grainWeight, 0.20, 1.0);

      float grain = filmGrain(vUv, uTime) * uGrainIntensity * grainWeight;
      composite += vec3(grain);

      // 5. ACES Filmic Tone Mapping & Gamma Correction
      vec3 graded = ACESFilm(composite);
      graded = pow(graded, vec3(1.0 / 2.2)); // Linear to sRGB gamma

      gl_FragColor = vec4(graded, baseColor.a);
    }
  `
};
