"""Profile one K=64 belief chunk component by component on the real checkpoint."""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from information_boltzmann.core.torus3d import CBIMTorus3D

CKPT = ROOT / "results/q8_predictive_ports_k64_3000_v2/last.pt"


def timeit(fn, repeat=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeat


def main() -> None:
    torch.manual_seed(11)
    saved = torch.load(CKPT, map_location="cuda", weights_only=False)
    model = CBIMTorus3D(
        shape=(8, 8, 4), velocities=8, content_dim=16, collision_layers=2,
        relative_address=True, readout_type="belief_agent",
        write_type="w4_predictive_agent", micro_steps=64,
        dissipation_type="quadratic",
    ).cuda()
    model.load_state_dict(saved["model"])
    belief = model.initial_belief(1, warm_start=False)
    belief.field.copy_(saved["state"])
    belief.precision.copy_(saved["precision"])

    ids = torch.randint(0, 50257, (1, 8), device="cuda")
    targets = torch.randint(0, 50257, (1, 8), device="cuda")

    transport, collision, bath = model.transport, model.collision, model.bath
    field = belief.field
    dt = model.tau_0_tensor

    # Does the triton Givens path actually engage on this box?
    coefficient = torch.randn(1, 256, collision.nullity, device="cuda")
    angles = torch.randn(1, 256, collision.layers, collision.nullity // 2, device="cuda")
    try:
        from information_boltzmann.core.triton_givens import triton_givens
        out = triton_givens(coefficient, angles)
        torch.cuda.synchronize()
        print(f"triton_givens: ENGAGED, out {tuple(out.shape)}")
    except Exception as error:  # pragma: no cover - diagnostic only
        print(f"triton_givens: FALLBACK ({type(error).__name__}: {error})")

    # Multiplier recomputation (parameter-only work done 8192x per update today)
    mult_cache, omega = transport.multiplier(dt)
    print(f"multiplier shape {tuple(mult_cache.shape)} "
          f"| recompute {timeit(lambda: transport.multiplier(dt))*1e3:.2f} ms "
          f"| apply-only {timeit(lambda: transport.apply_multiplier(field, mult_cache))*1e3:.2f} ms")

    print(f"transport microstep full  {timeit(lambda: transport.forward(field)[0])*1e3:.3f} ms")
    print(f"collision microstep       {timeit(lambda: collision(field, dt)[0])*1e3:.3f} ms")
    print(f"bath microstep            {timeit(lambda: bath(field, dt)[0])*1e3:.3f} ms")

    # Full chunk fwd+bwd, then component ablations
    def chunk(disable_transport=False, disable_collision=False, disable_bath=False):
        loss, _, _ = model.forward_belief(
            ids, targets, model.initial_belief(1, warm_start=False),
            disable_transport=disable_transport,
            disable_collision=disable_collision,
            disable_bath=disable_bath)
        loss.backward()
        model.zero_grad(set_to_none=True)

    base = timeit(lambda: chunk(), repeat=5, warmup=2)
    no_bath = timeit(lambda: chunk(disable_bath=True), repeat=5, warmup=2)
    no_coll = timeit(lambda: chunk(disable_collision=True), repeat=5, warmup=2)
    no_trans = timeit(lambda: chunk(disable_transport=True), repeat=5, warmup=2)
    print(f"chunk 8 tokens fwd+bwd: full {base*1e3:.0f} ms | -bath {no_bath*1e3:.0f} "
          f"| -collision {no_coll*1e3:.0f} | -transport {no_trans*1e3:.0f} ms")
    print(f"implied per update (16 chunks): full {base*16:.2f} s"
          f" | bath {max(base-no_bath,0)*16:.2f} | collision {max(base-no_coll,0)*16:.2f}"
          f" | transport {max(base-no_trans,0)*16:.2f} s")


if __name__ == "__main__":
    main()
