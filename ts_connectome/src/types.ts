/**
 * Type Definitions for Information Boltzmann / MengLi
 * 2401 Causal Reservoir Connectome & Cinematic GPU Pipeline
 */

export interface CGBNode {
  id: number;
  rel_id: number;
  gen: number;
  gen_color: string;
  sector: number;
  sector_name: string;
  comp_size: number;
  degree: number;
  rw_betweenness: number;
  rw_rank: number;
  sp_betweenness: number;
  sp_rank: number;
  is_top_rw: boolean;
  supports: number[];
  supports_count: number;
  is_harmonic: boolean;
  is_quartet: boolean;
  is_bridge: boolean;
  angle: number;
  circos_pos: [number, number, number];
  brain_pos: [number, number, number];
}

export interface CGBEdge {
  id: number;
  u: number;
  v: number;
  is_harmonic: boolean;
  is_cross_gen: boolean;
  is_bottleneck: boolean;
  u_gen: number;
  v_gen: number;
  span: number;
}

export interface CGBSector {
  name: string;
  short: string;
  color: string;
  desc: string;
  count?: number;
  angle_start?: number;
  angle_end?: number;
}

export interface CinematicFXParams {
  exposure: number;           // Camera sensor exposure (0.1 ~ 1.5, default: 0.65)
  bloomIntensity: number;     // Multi-tap bloom gain (0.0 ~ 2.0, default: 0.85)
  bloomRadius: number;        // Kawase blur spread radius (0.5 ~ 3.0, default: 1.25)
  bloomThreshold: number;     // Luminance cutoff knee (0.1 ~ 0.8, default: 0.35)
  halationIntensity: number;  // CineStill 800T warm red-orange scatter (0.0 ~ 1.5, default: 0.65)
  halationSpread: number;     // Chromatic dispersion distance (0.001 ~ 0.015, default: 0.005)
  grainIntensity: number;     // Triangular photochemical grain amplitude (0.0 ~ 0.15, default: 0.035)
}
