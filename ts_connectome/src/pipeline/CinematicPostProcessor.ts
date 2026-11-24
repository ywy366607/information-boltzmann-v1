import * as THREE from 'three';
import { BrightPassShader, KawaseBlurShader, CinematicCompositeShader } from '../shaders/CinematicPostShader';
import { CinematicFXParams } from '../types';

export class CinematicPostProcessor {
  private renderer: THREE.WebGLRenderer;
  private width: number;
  private height: number;
  
  // Render Targets
  private sceneTarget: THREE.WebGLRenderTarget;
  private brightTarget: THREE.WebGLRenderTarget;
  private blurTargetA: THREE.WebGLRenderTarget;
  private blurTargetB: THREE.WebGLRenderTarget;

  // Orthographic Post Camera & Quad
  private postCamera: THREE.OrthographicCamera;
  private quadGeo: THREE.PlaneGeometry;

  // Materials
  private brightPassMat: THREE.ShaderMaterial;
  private blurMat: THREE.ShaderMaterial;
  private compositeMat: THREE.ShaderMaterial;
  private quadMesh: THREE.Mesh;

  public params: CinematicFXParams;

  constructor(renderer: THREE.WebGLRenderer, width: number, height: number, initialParams?: Partial<CinematicFXParams>) {
    this.renderer = renderer;
    this.width = width;
    this.height = height;

    this.params = {
      exposure: 0.65,
      bloomIntensity: 0.85,
      bloomRadius: 1.25,
      bloomThreshold: 0.35,
      halationIntensity: 0.65,
      halationSpread: 0.005,
      grainIntensity: 0.035,
      ...initialParams
    };

    // HalfFloat HDR Buffer for smooth light blending
    const isHalfFloatSupported = renderer.capabilities.isWebGL2;
    const targetType = isHalfFloatSupported ? THREE.HalfFloatType : THREE.UnsignedByteType;

    const baseOptions: THREE.RenderTargetOptions = {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      type: targetType
    };

    this.sceneTarget = new THREE.WebGLRenderTarget(width, height, baseOptions);
    
    // Half-res downsample for silky bloom and fast GPU execution
    const downW = Math.max(1, Math.floor(width / 2));
    const downH = Math.max(1, Math.floor(height / 2));
    this.brightTarget = new THREE.WebGLRenderTarget(downW, downH, baseOptions);
    this.blurTargetA = new THREE.WebGLRenderTarget(downW, downH, baseOptions);
    this.blurTargetB = new THREE.WebGLRenderTarget(downW, downH, baseOptions);

    this.postCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.quadGeo = new THREE.PlaneGeometry(2, 2);

    // 1. Bright Pass Material
    this.brightPassMat = new THREE.ShaderMaterial({
      uniforms: THREE.UniformsUtils.clone(BrightPassShader.uniforms),
      vertexShader: BrightPassShader.vertexShader,
      fragmentShader: BrightPassShader.fragmentShader,
      depthTest: false,
      depthWrite: false
    });

    // 2. Kawase Blur Material
    this.blurMat = new THREE.ShaderMaterial({
      uniforms: THREE.UniformsUtils.clone(KawaseBlurShader.uniforms),
      vertexShader: KawaseBlurShader.vertexShader,
      fragmentShader: KawaseBlurShader.fragmentShader,
      depthTest: false,
      depthWrite: false
    });
    this.blurMat.uniforms.uResolution.value = new THREE.Vector2(downW, downH);

    // 3. Composite Material (Bloom + Halation + Film Grain + ACES)
    this.compositeMat = new THREE.ShaderMaterial({
      uniforms: THREE.UniformsUtils.clone(CinematicCompositeShader.uniforms),
      vertexShader: CinematicCompositeShader.vertexShader,
      fragmentShader: CinematicCompositeShader.fragmentShader,
      depthTest: false,
      depthWrite: false
    });
    this.compositeMat.uniforms.uResolution.value = new THREE.Vector2(width, height);

    this.quadMesh = new THREE.Mesh(this.quadGeo, this.compositeMat);
  }

  public setSize(width: number, height: number): void {
    if (width <= 0 || height <= 0) return;
    this.width = width;
    this.height = height;

    this.sceneTarget.setSize(width, height);
    const downW = Math.max(1, Math.floor(width / 2));
    const downH = Math.max(1, Math.floor(height / 2));
    this.brightTarget.setSize(downW, downH);
    this.blurTargetA.setSize(downW, downH);
    this.blurTargetB.setSize(downW, downH);

    this.blurMat.uniforms.uResolution.value.set(downW, downH);
    this.compositeMat.uniforms.uResolution.value.set(width, height);
  }

  public render(scene: THREE.Scene, camera: THREE.Camera, time: number): void {
    const gl = this.renderer;

    // Pass 1: Render 3D Scene into HDR Scene Target
    gl.setRenderTarget(this.sceneTarget);
    gl.clear();
    gl.render(scene, camera);

    // Pass 2: Bright-pass threshold into Downsampled Target
    this.brightPassMat.uniforms.tDiffuse.value = this.sceneTarget.texture;
    this.brightPassMat.uniforms.uThreshold.value = this.params.bloomThreshold;
    this.quadMesh.material = this.brightPassMat;

    gl.setRenderTarget(this.brightTarget);
    gl.clear();
    gl.render(this.quadMesh, this.postCamera);

    // Pass 3: Dual Kawase Iterative Blur (2 Iterations ping-pong)
    this.blurMat.uniforms.tDiffuse.value = this.brightTarget.texture;
    this.blurMat.uniforms.uOffset.value = 1.0 * this.params.bloomRadius;
    this.quadMesh.material = this.blurMat;

    gl.setRenderTarget(this.blurTargetA);
    gl.clear();
    gl.render(this.quadMesh, this.postCamera);

    // Iteration 2 (Wider spread for cinematic glow)
    this.blurMat.uniforms.tDiffuse.value = this.blurTargetA.texture;
    this.blurMat.uniforms.uOffset.value = 2.4 * this.params.bloomRadius;

    gl.setRenderTarget(this.blurTargetB);
    gl.clear();
    gl.render(this.quadMesh, this.postCamera);

    // Pass 4: Final Composite (Scene + Bloom + CineStill Halation + Photochemical Grain)
    this.compositeMat.uniforms.tDiffuse.value = this.sceneTarget.texture;
    this.compositeMat.uniforms.tBloom.value = this.blurTargetB.texture;
    this.compositeMat.uniforms.uTime.value = time;
    this.compositeMat.uniforms.uExposure.value = this.params.exposure;
    this.compositeMat.uniforms.uBloomIntensity.value = this.params.bloomIntensity;
    this.compositeMat.uniforms.uHalationIntensity.value = this.params.halationIntensity;
    this.compositeMat.uniforms.uHalationSpread.value = this.params.halationSpread;
    this.compositeMat.uniforms.uGrainIntensity.value = this.params.grainIntensity;

    this.quadMesh.material = this.compositeMat;

    // Render directly to screen framebuffer
    gl.setRenderTarget(null);
    gl.render(this.quadMesh, this.postCamera);
  }

  public dispose(): void {
    this.sceneTarget.dispose();
    this.brightTarget.dispose();
    this.blurTargetA.dispose();
    this.blurTargetB.dispose();
    this.quadGeo.dispose();
    this.brightPassMat.dispose();
    this.blurMat.dispose();
    this.compositeMat.dispose();
  }
}
