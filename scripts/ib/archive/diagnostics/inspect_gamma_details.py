import zipfile
import numpy as np

# Load MaleCNS
graph_path = "data/malecns_v1/fly_reservoir_coba.npz"
packed = np.load(graph_path, allow_pickle=False)

n_neurons = int(packed["neuron_body_ids"].shape[0])
superclass_id = packed["superclass_id"]
superclass_names = [s.decode() if isinstance(s, bytes) else str(s) for s in packed["superclass_names"]]

OUTPUT_CLASSES = ("cb_motor", "vnc_motor", "descending_neuron",
                  "cb_efferent", "vnc_efferent", "efferent_ascending",
                  "efferent_descending", "cb_endocrine", "vnc_endocrine")

read_mask = np.zeros(n_neurons, dtype=bool)
for name in OUTPUT_CLASSES:
    if name in superclass_names:
        read_mask |= superclass_id == superclass_names.index(name)
read_idx = np.flatnonzero(read_mask).astype(np.int64)

coords = packed["coords_um"]
SENSORY_CLASSES = ("cb_sensory", "ol_sensory", "vnc_sensory",
                   "sensory_ascending", "sensory_descending",
                   "cb_sensory_tbc", "vnc_sensory_tbc",
                   "sensory_ascending_tbc")
sensory = np.zeros(n_neurons, dtype=bool)
for name in SENSORY_CLASSES:
    if name in superclass_names:
        sensory |= superclass_id == superclass_names.index(name)
sens_idx = np.flatnonzero(sensory).astype(np.int64)
sens_center = np.median(coords[sens_idx], axis=0)
dists_um = np.linalg.norm(coords[read_idx] - sens_center, axis=1)

d_min, d_max = float(dists_um.min()), float(dists_um.max())
norm_dist = (dists_um - d_min) / max(d_max - d_min, 1e-6)
init_tau = 1.0 + 4.0 * norm_dist
init_gamma = np.clip(init_tau / (1.0 + init_tau), 0.02, 0.98)

z_last = zipfile.ZipFile("artifacts/active_training/q8_fly_bptt32_gamma_2333_100k/last.pt")
learned_logit_last = np.frombuffer(z_last.read("last.pt/data/11"), dtype=np.float32)

def to_gamma(logit):
    sig = 1.0 / (1.0 + np.exp(-logit))
    return 0.01 + 0.98 * sig

gamma_last = to_gamma(learned_logit_last)
delta_gamma = gamma_last - init_gamma
tau_last = gamma_last / (1.0 - gamma_last)

read_superclasses = [superclass_names[superclass_id[i]] for i in read_idx]

# Check degree of read neurons in edge_post / edge_pre
edge_pre = packed["edge_pre"]
edge_post = packed["edge_post"]

# In-degree of read neurons
in_degree = np.bincount(edge_post, minlength=n_neurons)[read_idx]
out_degree = np.bincount(edge_pre, minlength=n_neurons)[read_idx]

print("Correlation of Delta Gamma with Anatomical Properties:")
corr_dist = np.corrcoef(dists_um, delta_gamma)[0, 1]
corr_in = np.corrcoef(in_degree, delta_gamma)[0, 1]
corr_out = np.corrcoef(out_degree, delta_gamma)[0, 1]
print(f"  r(Delta Gamma, Distance)   : {corr_dist:+.4f}")
print(f"  r(Delta Gamma, In-Degree)  : {corr_in:+.4f}")
print(f"  r(Delta Gamma, Out-Degree) : {corr_out:+.4f}")

# Histogram of delta_gamma
hist, bin_edges = np.histogram(delta_gamma, bins=10)
print("\nHistogram of Delta Gamma across 2,333 motor neurons:")
for i in range(len(hist)):
    print(f"  [{bin_edges[i]:+.4f}, {bin_edges[i+1]:+.4f}) : {hist[i]:>4} neurons {'*' * (hist[i] // 30)}")

# Check within-class differentiation variance
print("\nWithin-class standard deviation of Delta Gamma:")
classes = sorted(list(set(read_superclasses)))
for cls in classes:
    mask = np.array([c == cls for c in read_superclasses])
    d_sub = delta_gamma[mask]
    print(f"  {cls:<22} (N={len(d_sub):>4}): Delta std={d_sub.std():.4f}, Range=[{d_sub.min():+.4f}, {d_sub.max():+.4f}]")
