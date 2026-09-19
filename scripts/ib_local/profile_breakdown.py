import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
import time
from scripts.ib_local.cbim_torus3d import CBIMTorus3D

model = CBIMTorus3D(
    shape=(8, 8, 4), velocities=8, content_dim=16,
    v2_coordinate_components=True, readout_type="kernel_r1", write_type="w2_impedance",
    micro_steps=1, adaptive_clock=True, continuous_velocities=True, dissipation_type="unified",
    dissipation_rank=4, three_clock=True, tau_mem=3.0, nu_s_init=0.020, decouple_source_feedback=True,
    reversible_ponder=True
).cuda()
state = model.initial_state(1, "cuda")
inp = torch.tensor([100], device="cuda")
tok_embed = model.source.embedding(inp)

# Warmup
for _ in range(10):
    model.step(state, inp, micro_steps=1)

N_ITERS = 200

def time_op(name, fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000 / N_ITERS
    return ms

print("=" * 60)
print(f"OPERATOR BREAKDOWN PER SINGLE INVOCATION (averaged over {N_ITERS} runs)")
print("=" * 60)

# 1. Source (write)
t_src = time_op("Source Write", lambda: model.source(state, inp))
print(f"1. Source Write               : {t_src:6.3f} ms")

# 2. Clock
t_clk = time_op("Adaptive Clock", lambda: model.clock(state, tok_embed))
print(f"2. Adaptive Clock             : {t_clk:6.3f} ms")

# 3. Direction
t_dir = time_op("Continuous Direction", lambda: model.direction_controller(state, tok_embed))
print(f"3. Continuous Direction       : {t_dir:6.3f} ms")

# 4. Transport Multiplier
cached_learned = model.transport.learned_symbol()
dir_k = model.direction_controller(state, tok_embed)
t_mult = time_op("Transport Multiplier", lambda: model.transport.multiplier(1.0, direction=dir_k, learned=cached_learned))
print(f"4. Transport Multiplier       : {t_mult:6.3f} ms")

# 5. Transport FFT + IFFT (apply_multiplier)
mult, _ = model.transport.multiplier(1.0, direction=dir_k, learned=cached_learned)
t_fft = time_op("Transport Apply (2x FFT3D)", lambda: model.transport.apply_multiplier(state, mult))
print(f"5. Transport Apply (2x FFT3D) : {t_fft:6.3f} ms")

# 6. Collision Operator
t_coll = time_op("Collision (Nullspace+Givens)", lambda: model.collision(state, 1.0))
print(f"6. Collision (Nullspace+Givens): {t_coll:6.3f} ms")

# 7. Readout Probe
t_ro = time_op("Characteristic Kernel Readout", lambda: model.readout(state, tok_embed))
print(f"7. Readout Probe              : {t_ro:6.3f} ms")

# 8. Memory Aging Bath
t_bath = time_op("Memory Aging Bath", lambda: model.bath(state, 3.0, tok_embed=tok_embed, disable_viscosity=True, disable_subspace=False))
print(f"8. Memory Aging Bath          : {t_bath:6.3f} ms")

# 9. Decoder (Vocab projection)
feat = model.readout(state, tok_embed)[0]
t_dec = time_op("Decoder Projection (50k vocab)", lambda: model.decoder(feat))
print(f"9. Decoder Projection (50k)   : {t_dec:6.3f} ms")

print("=" * 60)
print("Projected times per token:")
t_micro = t_clk + t_dir + t_mult + t_fft + t_coll
print(f"  Single microstep (T+C+Clock+Dir): {t_micro:6.3f} ms")
for k in [1, 2, 4, 8]:
    t_tok = t_src + k * t_micro + t_ro + t_bath + t_dec
    print(f"  Full Token with K={k:<2d}         : {t_tok:6.3f} ms ({1000/t_tok:5.1f} tok/s) -> 128 tok batch: {t_tok * 128 / 1000:.3f} s")
print("=" * 60)
