import zipfile
import numpy as np

# Load MaleCNS graph to get neuron coordinates and superclasses
graph_path = "data/malecns_v1/fly_reservoir_coba.npz"
packed = np.load(graph_path, allow_pickle=False)

n_neurons = int(packed["neuron_body_ids"].shape[0])
superclass_id = packed["superclass_id"]
superclass_names = [s.decode() if isinstance(s, bytes) else str(s) for s in packed["superclass_names"]]

SENSORY_CLASSES = ("cb_sensory", "ol_sensory", "vnc_sensory",
                   "sensory_ascending", "sensory_descending",
                   "cb_sensory_tbc", "vnc_sensory_tbc",
                   "sensory_ascending_tbc")
OUTPUT_CLASSES = ("cb_motor", "vnc_motor", "descending_neuron",
                  "cb_efferent", "vnc_efferent", "efferent_ascending",
                  "efferent_descending", "cb_endocrine", "vnc_endocrine")

# Injection and Read indices (same as in fly_reservoir.py)
sensory = np.zeros(n_neurons, dtype=bool)
for name in SENSORY_CLASSES:
    if name in superclass_names:
        sensory |= superclass_id == superclass_names.index(name)
sens_idx = np.flatnonzero(sensory).astype(np.int64)

read_mask = np.zeros(n_neurons, dtype=bool)
for name in OUTPUT_CLASSES:
    if name in superclass_names:
        read_mask |= superclass_id == superclass_names.index(name)
read_idx = np.flatnonzero(read_mask).astype(np.int64)

# Anatomical distances
coords = packed["coords_um"]
sens_center = np.median(coords[sens_idx], axis=0)
dists_um = np.linalg.norm(coords[read_idx] - sens_center, axis=1)
d_min, d_max = float(dists_um.min()), float(dists_um.max())
norm_dist = (dists_um - d_min) / max(d_max - d_min, 1e-6)

# Initial tau and gamma
init_tau = 1.0 + 4.0 * norm_dist
init_gamma = np.clip(init_tau / (1.0 + init_tau), 0.02, 0.98)
u_init = (init_gamma - 0.01) / 0.98
init_logit = np.log(u_init / (1.0 - u_init))

# Read learned logits from last.pt and best.pt
z_last = zipfile.ZipFile("artifacts/active_training/q8_fly_bptt32_gamma_2333_100k/last.pt")
learned_logit_last = np.frombuffer(z_last.read("last.pt/data/11"), dtype=np.float32)

z_best = zipfile.ZipFile("artifacts/active_training/q8_fly_bptt32_gamma_2333_100k/best.pt")
learned_logit_best = np.frombuffer(z_best.read("best.pt/data/11"), dtype=np.float32)

def to_gamma(logit):
    sig = 1.0 / (1.0 + np.exp(-logit))
    return 0.01 + 0.98 * sig

def to_tau(gamma):
    return gamma / np.maximum(1.0 - gamma, 1e-6)

gamma_last = to_gamma(learned_logit_last)
tau_last = to_tau(gamma_last)

gamma_best = to_gamma(learned_logit_best)
tau_best = to_tau(gamma_best)

# Superclass of each read neuron
read_superclasses = [superclass_names[superclass_id[i]] for i in read_idx]

print("=" * 80)
print("ANALYSIS OF 2,333 MOTOR NEURON GAMMA DIFFERENTIATION")
print("=" * 80)

print(f"\n1. Overall Statistics (N={len(gamma_last)}):")
print(f"  Init  Gamma: Mean={init_gamma.mean():.4f}, Std={init_gamma.std():.4f}, Min={init_gamma.min():.4f}, Max={init_gamma.max():.4f}")
print(f"  Learned Last: Mean={gamma_last.mean():.4f}, Std={gamma_last.std():.4f}, Min={gamma_last.min():.4f}, Max={gamma_last.max():.4f}")
print(f"  Learned Best: Mean={gamma_best.mean():.4f}, Std={gamma_best.std():.4f}, Min={gamma_best.min():.4f}, Max={gamma_best.max():.4f}")

delta_gamma = gamma_last - init_gamma
delta_tau = tau_last - init_tau
print(f"\n  Mean Delta Gamma: {delta_gamma.mean():+.4f} (Max: {delta_gamma.max():+.4f}, Min: {delta_gamma.min():+.4f})")
print(f"  Abs Delta Gamma > 0.01: {np.sum(np.abs(delta_gamma) > 0.01)} / {len(delta_gamma)} ({np.mean(np.abs(delta_gamma) > 0.01)*100:.1f}%)")
print(f"  Abs Delta Gamma > 0.03: {np.sum(np.abs(delta_gamma) > 0.03)} / {len(delta_gamma)} ({np.mean(np.abs(delta_gamma) > 0.03)*100:.1f}%)")
print(f"  Abs Delta Gamma > 0.05: {np.sum(np.abs(delta_gamma) > 0.05)} / {len(delta_gamma)} ({np.mean(np.abs(delta_gamma) > 0.05)*100:.1f}%)")

print("\n2. Differentiation by Anatomical Superclass (last.pt):")
print(f"{'Class':<22} | {'Count':<5} | {'Init Gamma':<10} | {'Learned Gamma':<14} | {'Delta':<8} | {'Learned Tau (ticks)':<18}")
print("-" * 88)

classes = sorted(list(set(read_superclasses)))
for cls in classes:
    mask = np.array([c == cls for c in read_superclasses])
    cnt = int(mask.sum())
    g_init_mean = init_gamma[mask].mean()
    g_learn_mean = gamma_last[mask].mean()
    g_learn_std = gamma_last[mask].std()
    d_mean = delta_gamma[mask].mean()
    t_mean = tau_last[mask].mean()
    print(f"{cls:<22} | {cnt:<5} | {g_init_mean:<10.4f} | {g_learn_mean:.4f}±{g_learn_std:.4f} | {d_mean:<+8.4f} | {t_mean:.2f} ticks")

print("\n3. Spatial Correlation with 3D Distance (D_i to Sensory Center):")
corr_init = np.corrcoef(dists_um, init_gamma)[0, 1]
corr_learned_last = np.corrcoef(dists_um, gamma_last)[0, 1]
corr_learned_best = np.corrcoef(dists_um, gamma_best)[0, 1]
print(f"  Initial Pearson r(Distance, Gamma): {corr_init:.4f}")
print(f"  Learned Last  r(Distance, Gamma): {corr_learned_last:.4f}")
print(f"  Learned Best  r(Distance, Gamma): {corr_learned_best:.4f}")

# Distribution of shifts: fastest vs slowest adapting neurons
top_accelerated = np.argsort(delta_gamma)[:5]
top_decelerated = np.argsort(delta_gamma)[-5:]

print("\n4. Top 5 Accelerated Neurons (Shifted to faster/shorter memory, smaller gamma):")
for idx in top_accelerated:
    print(f"  Neuron {idx} ({read_superclasses[idx]}): Init {init_gamma[idx]:.4f} -> Learned {gamma_last[idx]:.4f} (Delta: {delta_gamma[idx]:+.4f}, Tau: {tau_last[idx]:.2f} ticks, Dist: {dists_um[idx]:.1f} um)")

print("\n5. Top 5 Decelerated Neurons (Shifted to longer memory/slower integration, larger gamma):")
for idx in reversed(top_decelerated):
    print(f"  Neuron {idx} ({read_superclasses[idx]}): Init {init_gamma[idx]:.4f} -> Learned {gamma_last[idx]:.4f} (Delta: {delta_gamma[idx]:+.4f}, Tau: {tau_last[idx]:.2f} ticks, Dist: {dists_um[idx]:.1f} um)")
