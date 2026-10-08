"""Reproducible world numerical audit, with config and no learned agent claims."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from information_boltzmann.world import GenerativeWorldLaw, PersistentWorld, WorldConfig
from information_boltzmann.world.model import DEFAULT_MATERIALS


def run(duration: float, rate: int, cutoff: float) -> dict:
    world = PersistentWorld(GenerativeWorldLaw(WorldConfig(sample_rate=rate, cutoff_hz=cutoff)))
    world.step(1)  # compilation excluded from throughput, state sample retained
    start = time.perf_counter()
    peaks, rms, residuals = [], [], []
    frames = []
    target = int(duration * rate)
    while world.steps < target:
        audio = world.step(min(256, target - world.steps))
        peaks.append(float(np.abs(audio).max()))
        rms.append(float(np.mean(audio ** 2)))
        residuals.append(abs(world.snapshot()["balance_error"]))
        if world.steps % 2048 < 256:
            frames.append(world.snapshot())
    elapsed = time.perf_counter() - start
    snapshot = world.snapshot()
    return {"manifest": world.law.manifest(), "physical_seconds": world.time,
            "wall_compute_seconds": elapsed, "realtime_factor": duration / elapsed,
            "initial_energy_j": world.initial_energy,
            "max_ledger_error_j": max(residuals),
            "max_relative_ledger_error": max(residuals) / world.initial_energy,
            "pressure_peak_pa": max(peaks), "pressure_block_rms_mean_pa": float(np.sqrt(np.mean(rms))),
            "final": snapshot, "samples": frames,
            "claim": "initialized numerical world closure; not real-world calibration or agent capability"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=12)
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--cutoff-hz", type=float, default=480)
    parser.add_argument("--output", type=Path, default=ROOT / "results/published/generative_world_numerical_audit.json")
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("duration must be positive")
    audit = run(args.duration, args.sample_rate, args.cutoff_hz)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("physical_seconds", "wall_compute_seconds", "realtime_factor",
                                         "max_ledger_error_j", "max_relative_ledger_error", "pressure_peak_pa")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
