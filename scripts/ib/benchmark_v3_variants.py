"""Benchmark unified-bath chunk variants to pick the v3 trainer form."""
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from information_boltzmann.core.torus3d import CBIMTorus3D


def build(unified=True):
    torch.manual_seed(11)
    model = CBIMTorus3D(
        shape=(8, 8, 4), velocities=8, content_dim=16, collision_layers=2,
        relative_address=True, readout_type="belief_agent",
        write_type="w4_predictive_agent", micro_steps=64,
        dissipation_type="unified" if unified else "quadratic",
        dissipation_rank=4,
    ).cuda()
    return model


def timeit(fn, repeat=5, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeat


def chunk_loss(model, ids, targets, autocast_dtype=None):
    def run():
        belief = model.initial_belief(1, warm_start=False)
        if autocast_dtype is not None:
            with torch.autocast("cuda", dtype=autocast_dtype):
                loss, _, _ = model.forward_belief(ids, targets, belief)
        else:
            loss, _, _ = model.forward_belief(ids, targets, belief)
        loss.backward()
        model.zero_grad(set_to_none=True)
    return run


def main() -> None:
    ids = torch.randint(0, 50257, (1, 8), device="cuda")
    targets = torch.randint(0, 50257, (1, 8), device="cuda")

    model = build()
    base = timeit(chunk_loss(model, ids, targets), repeat=3, warmup=2)
    print(f"eager unified chunk           : {base*1e3:7.0f} ms  (x16 = {base*16:5.2f} s/update)")

    model.collision.forward = torch.compile(model.collision.forward)
    model.bath.forward = torch.compile(model.bath.forward)
    model.transport.apply_multiplier = torch.compile(model.transport.apply_multiplier)
    c3 = timeit(chunk_loss(model, ids, targets))
    print(f"compile x3 operators          : {c3*1e3:7.0f} ms  (x16 = {c3*16:5.2f} s/update)")

    model.write_agent.forward = torch.compile(model.write_agent.forward)
    model.readout.forward = torch.compile(model.readout.forward)
    c5 = timeit(chunk_loss(model, ids, targets))
    print(f"compile +write_agent +readout : {c5*1e3:7.0f} ms  (x16 = {c5*16:5.2f} s/update)")

    amp = timeit(chunk_loss(model, ids, targets, autocast_dtype=torch.float16))
    print(f"compile x5 + autocast fp16    : {amp*1e3:7.0f} ms  (x16 = {amp*16:5.2f} s/update)")

    loss_check = chunk_loss(model, ids, targets, autocast_dtype=torch.float16)
    loss_check()
    torch.cuda.synchronize()
    print("fp16 chunk executed without runtime failure")


if __name__ == "__main__":
    main()
