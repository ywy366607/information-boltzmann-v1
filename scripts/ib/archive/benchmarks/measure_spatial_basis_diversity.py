"""Quantitative measurement of spatial basis functions in the biological fly connectome vs. a 3D periodic cube.

Quantifies:
1. Intrinsic Dimension (TwoNN, Facco et al. 2017) across biological brain regions vs. 3D Cube.
2. Spectral Dimension d_s(t) across temporal/spatial diffusion scales via heat trace P(t) ~ t^(-d_s/2).
3. Spatial Localization: Inverse Participation Ratio (IPR) of graph Laplacian eigenmodes vs. flat 3D Fourier modes.
4. Functional Neuropil Specificity & Shannon Compartmental Entropy of the spatial basis functions.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pyarrow.feather as feather

ROOT = Path(__file__).resolve().parents[2]


def compute_twonn_dimension(coords: np.ndarray, sample_size: int = 5000) -> float:
    """Compute local intrinsic dimension using TwoNN (Facco et al., Sci Rep 2017).
    
    Robust, parameter-free estimator based on the ratio of distances to the 2nd vs 1st nearest neighbor.
    """
    n = len(coords)
    if n < 20:
        return float("nan")
    
    # Subsample if large
    if n > sample_size:
        idx = np.random.choice(n, sample_size, replace=False)
        pts = coords[idx]
    else:
        pts = coords
        
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=3, algorithm="auto").fit(pts)
    distances, _ = nbrs.kneighbors(pts)
    
    # r1 = distance to 1st nearest neighbor, r2 = distance to 2nd
    r1 = distances[:, 1]
    r2 = distances[:, 2]
    
    # Filter identical/coincident points
    valid = (r1 > 1e-6) & (r2 > r1)
    if np.sum(valid) < 10:
        return float("nan")
        
    mu = r2[valid] / r1[valid]
    # Maximum likelihood estimator for TwoNN
    # d = N / sum(log(mu_i))
    d_mle = float(len(mu) / np.sum(np.log(mu)))
    return round(d_mle, 3)


def main():
    print("=" * 70)
    print("QUANTITATIVE BENCHMARK: FLY CONNECTOME SPATIAL BASIS VS. 3D PERIODIC CUBE")
    print("=" * 70)
    
    # 1. Load connectome
    t0 = time.perf_counter()
    graph_path = ROOT / "data/malecns_v1/fly_reservoir_biological.npz"
    packed = np.load(graph_path)
    coords_um = packed["coords_um"]
    body_ids = packed["neuron_body_ids"]
    pre = packed["edge_pre"]
    post = packed["edge_post"]
    weight = np.abs(packed["edge_weight"])
    n = len(body_ids)
    print(f"Loaded connectome: {n:,} neurons, {len(pre):,} synapses in {time.perf_counter() - t0:.2f}s")
    
    # Load cell class annotations
    ann = feather.read_table(ROOT / "data/malecns_v1/body-annotations.feather").to_pandas()
    ann_dict = ann.set_index("bodyId").to_dict("index")
    
    classes = {}
    for idx, bid in enumerate(body_ids):
        row = ann_dict.get(bid, {})
        cls = row.get("class", "unknown")
        if cls not in classes:
            classes[cls] = []
        classes[cls].append(idx)
        
    major_regions = {
        "Central_Complex_CX (Navigation Ring/Cylinder)": classes.get("CX", []),
        "Visual_Optic_Lobe (Compound Eye Retinotopy)": classes.get("visual", []),
        "Mushroom_Body_KC (Kenyon Cells Associative)": classes.get("Kenyon_Cell", []),
        "Olfactory_Antennal_Lobe (Chemo-topic Glomeruli)": classes.get("olfactory", []),
        "Mechanosensory_Tactile (Body Somatotopy)": classes.get("mechanosensory_tactile", []),
        "Mechanosensory_Proprioceptive (Joint Kinematics)": classes.get("mechanosensory_proprioceptive", []),
        "Whole_Brain (Full 165k Network)": list(range(n)),
    }
    
    # -------------------------------------------------------------
    # EXPERIMENT 1: Intrinsic Dimensionality (TwoNN) per Region
    # -------------------------------------------------------------
    print("\n--- 1. INTRINSIC DIMENSION (TwoNN) BY BRAIN REGION ---")
    print("Benchmark: Theoretical 3D Periodic Cube has D = 3.000 everywhere.")
    
    # Create 3D Periodic Cube baseline
    np.random.seed(42)
    cube_pts = np.random.uniform(0, 1000, size=(10000, 3))
    cube_dim = compute_twonn_dimension(cube_pts)
    print(f"  Control 3D Cube:                    D = {cube_dim:.3f} (Expected: 3.0)")
    
    regional_dimensions = {"Control_3D_Cube": cube_dim}
    for name, indices in major_regions.items():
        if len(indices) < 30:
            continue
        reg_coords = coords_um[indices]
        reg_dim = compute_twonn_dimension(reg_coords)
        regional_dimensions[name] = reg_dim
        print(f"  {name:50s} N={len(indices):5d}, D = {reg_dim:.3f}")
        
    # -------------------------------------------------------------
    # EXPERIMENT 2: Spectral Dimension d_s(t) via Diffusion Trace
    # -------------------------------------------------------------
    print("\n--- 2. SPECTRAL DIMENSION d_s(t) ACROSS DIFFUSION TIME SCALES ---")
    print("For a periodic 3D cube: return probability P(t) ~ t^(-3/2) => d_s = 3.00 at all scales.")
    
    # Symmetrized normalized adjacency for diffusion
    A = sp.coo_matrix((weight, (pre, post)), shape=(n, n)).tocsr()
    A_sym = A + A.T
    deg = np.array(A_sym.sum(axis=1)).flatten()
    deg[deg == 0] = 1.0
    # Random walk transition matrix T = D^(-1) A
    T_rw = sp.diags(1.0 / deg) @ A_sym
    
    # Stochastic Hutchinson trace estimator for P(t) = (1/N) Tr(T^t)
    M = 10  # Rademacher random vectors
    V = np.random.choice([-1.0, 1.0], size=(n, M)).astype(np.float32)
    
    # Convert T_rw to PyTorch CPU CSR for 30x OpenMP multi-threaded acceleration
    import torch
    crow = torch.from_numpy(T_rw.indptr)
    col = torch.from_numpy(T_rw.indices)
    val = torch.from_numpy(T_rw.data.astype(np.float32))
    T_torch = torch.sparse_csr_tensor(crow, col, val, (n, n))
    X_torch = torch.from_numpy(V)
    V_torch = torch.from_numpy(V)
    
    diffusion_steps = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    heat_trace = {}
    
    # Multi-step propagation
    current_step = 0
    t_diff0 = time.perf_counter()
    for target_t in diffusion_steps:
        steps_to_take = target_t - current_step
        for _ in range(steps_to_take):
            X_torch = torch.matmul(T_torch, X_torch)
        current_step = target_t
        # Return probability estimate: (1 / (N * M)) sum(V * X)
        tr = float(torch.sum(V_torch * X_torch).item() / (n * M))
        heat_trace[target_t] = max(tr, 1e-12)
        
    print(f"  Completed 256 diffusion steps in {time.perf_counter() - t_diff0:.2f}s", flush=True)
    spectral_dims = {}
    for i in range(len(diffusion_steps) - 1):
        t1, t2 = diffusion_steps[i], diffusion_steps[i+1]
        p1, p2 = heat_trace[t1], heat_trace[t2]
        # d_s = -2 * (log(p2) - log(p1)) / (log(t2) - log(t1))
        d_s = -2.0 * (math.log(p2) - math.log(p1)) / (math.log(t2) - math.log(t1))
        spectral_dims[f"t={t1}->{t2}"] = round(d_s, 3)
        print(f"  Diffusion Scale t={t1:3d}->{t2:3d} ms:  Return Prob = {p2:.4e},  Spectral Dim d_s = {d_s:.3f}", flush=True)
        
    # -------------------------------------------------------------
    # EXPERIMENT 3: Graph Laplacian Eigenfunctions & IPR Localization
    # -------------------------------------------------------------
    print("\n--- 3. GRAPH LAPLACIAN BASIS LOCALIZATION (IPR) & COMPARTMENT ENTROPY ---", flush=True)
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(deg))
    M_sym = D_inv_sqrt @ A_sym @ D_inv_sqrt
    
    # L_norm = I - M_sym, so eigenvectors of L_norm are identically the eigenvectors of M_sym.
    # The smallest eigenvalues of L_norm correspond to the largest algebraic eigenvalues (which="LA")
    # of M_sym, converging in seconds with ARPACK!
    k_modes = 50
    t_eig0 = time.perf_counter()
    mu_vals, vecs = spla.eigsh(M_sym, k=k_modes, which="LA", tol=1e-3, maxiter=300)
    sort_idx = np.argsort(1.0 - mu_vals)
    vals = 1.0 - mu_vals[sort_idx]
    vecs = vecs[:, sort_idx]
    print(f"  Computed 50 Laplacian eigenmodes in {time.perf_counter() - t_eig0:.2f}s", flush=True)
    
    ipr = np.sum(vecs ** 4, axis=0)
    ipr_cube = 1.0 / n  # For a flat 3D Fourier wave, sum(u^4) = 1/N
    
    print(f"  3D Periodic Cube Fourier wave IPR:     {ipr_cube:.6e} (completely delocalized)", flush=True)
    print(f"  Connectome Eigenmodes: Median IPR =    {np.median(ipr):.6e} ({np.median(ipr)/ipr_cube:.1f}x more localized)", flush=True)
    print(f"  Connectome Eigenmodes: Max IPR    =    {ipr.max():.6e} ({ipr.max()/ipr_cube:.1f}x more localized)", flush=True)
    
    # Compute regional energy distribution for each eigenmode
    # p(c) = sum_{i in c} u_{k, i}^2
    reg_names = ["visual", "Kenyon_Cell", "CX", "olfactory", "mechanosensory_tactile", "mechanosensory_proprioceptive"]
    mode_energies = []
    mode_entropies = []
    max_entropy = math.log2(len(reg_names))
    
    for k in range(k_modes):
        u_sq = vecs[:, k] ** 2
        p_c = np.array([np.sum(u_sq[classes.get(c, [])]) for c in reg_names])
        p_c = p_c / (np.sum(p_c) + 1e-12)
        # Shannon entropy of spatial distribution
        ent = -np.sum(p_c[p_c > 0] * np.log2(p_c[p_c > 0]))
        mode_entropies.append(ent)
        top_reg = reg_names[int(np.argmax(p_c))]
        top_frac = float(np.max(p_c))
        mode_energies.append({"mode": k, "eigenvalue": float(vals[k]), "top_region": top_reg,
                              "top_fraction": top_frac, "entropy_bits": float(ent), "ipr": float(ipr[k])})
        
    print(f"\n  Spatial Shannon Entropy of Basis Modes: Mean = {np.mean(mode_entropies):.2f} bits (Max uniform = {max_entropy:.2f} bits)")
    print("  Selected Specialized Eigen-Basis Functions:")
    for m in [4, 5, 10, 15, 20, 30, 40]:
        if m < len(mode_energies):
            info = mode_energies[m]
            print(f"    Mode {m:2d} (lambda={info['eigenvalue']:.4f}): Dedicated to {info['top_region']:25s} ({info['top_fraction']*100:4.1f}% energy), IPR={info['ipr']:.4e}")

    # Save quantitative report
    out_file = ROOT / "results/spatial_basis_quantitative_analysis.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "n_neurons": n,
        "n_synapses": int(len(pre)),
        "control_3d_cube_properties": {
            "intrinsic_dimension": cube_dim,
            "spectral_dimension": 3.0,
            "eigenmode_ipr": ipr_cube,
            "spatial_entropy_bits": max_entropy,
            "geometry": "homogeneous flat Euclidean 3-torus (R=0)",
        },
        "regional_intrinsic_dimensions": regional_dimensions,
        "spectral_dimensions_by_scale": spectral_dims,
        "eigenmode_localization": {
            "median_ipr": float(np.median(ipr)),
            "max_ipr": float(ipr.max()),
            "localization_ratio_vs_3d_cube_median": float(np.median(ipr) / ipr_cube),
            "localization_ratio_vs_3d_cube_max": float(ipr.max() / ipr_cube),
        },
        "spatial_entropy": {
            "mean_entropy_bits": float(np.mean(mode_entropies)),
            "max_uniform_entropy_bits": max_entropy,
        },
        "selected_basis_modes": mode_energies[:25],
    }
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved full quantitative analysis to {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
