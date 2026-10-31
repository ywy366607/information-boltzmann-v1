import sys, os, time, math
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.abspath("."))
import torch
import numpy as np
from fine_grain.information_boltzmann.owt_cfe import OWTConstructalEngine

device = "cuda" if torch.cuda.is_available() else "cpu"
train_tokens = np.load("data/ib_owt_gpt2/train.npy", mmap_mode="r")

engine = OWTConstructalEngine(
    vocab_size=50257,
    dim=64,
    init_slots=2048,
    max_slots=8192,
    k_sparse=256,
    pipe_lr=4.0,
    J_crit=3.2,
    eta_oja=0.05,
    device=device
)

print("Running 1,000 steps with Emergent Vascular Hierarchy & Hamiltonian Orbits...")
t0 = time.time()
surps = []
for step in range(1000):
    c_tok = int(train_tokens[step])
    n_tok = int(train_tokens[step + 1])
    metrics = engine.step_stream(c_tok, n_tok)
    surps.append(metrics["surprisal_nats"])

torch.cuda.synchronize()
dt = time.time() - t0
print(f"1,000 steps completed in {dt:.2f}s ({dt/1000*1000:.2f} ms/step)")

# Inspect gamma distribution
act_idx = torch.where(engine.active_mask)[0]
gammas = engine.gamma[act_idx].cpu().numpy()
fluxes = engine.flux_accum[act_idx].cpu().numpy()

print(f"\nActive slots: {len(act_idx)}")
print(f"Gamma stats: min = {gammas.min():.4f}, median = {np.median(gammas):.4f}, max = {gammas.max():.4f}")
print("Gamma percentiles [5%, 25%, 50%, 75%, 95%]:", np.percentile(gammas, [5, 25, 50, 75, 95]))
print(f"Flux stats: min = {fluxes.min():.2f}, median = {np.median(fluxes):.2f}, max = {fluxes.max():.2f}")

# Inspect token velocities
vel_norms = torch.norm(engine.token_velocity, p=2, dim=-1).cpu().numpy()
active_vel = vel_norms[vel_norms > 1e-6]
print(f"\nActive token velocity count: {len(active_vel)}")
print(f"Velocity norms: min = {active_vel.min():.4f}, median = {np.median(active_vel):.4f}, max = {active_vel.max():.4f}")

# Check embedding diversity (ensure NO representation collapse)
E_norm = engine.embedding.weight.detach().cpu().numpy()
dots = np.sum(E_norm[active_vel > 0][:50] * E_norm[active_vel > 0][:50], axis=-1)
print(f"Norm of embeddings: mean = {np.mean(np.linalg.norm(E_norm[:100], axis=-1)):.4f} (strictly 1.0)")
print(f"Recent 100-step NLL: {np.mean(surps[-100:]):.3f} nats")
