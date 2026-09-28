"""Jointly train the predictive write/read Q8 belief system on OpenWebText.

This is the canonical trainer for the active Q8 port branch.  Every observed
token updates one persistent posterior belief ``(field, precision)`` through
the predictive write port; K=64 is the numerical resolution of its interior
flow.  The run never resets the belief between training chunks.  Truncation
limits gradient history only and is recorded explicitly in the run config.

The training objective is the joint realized port objective already defined by
``CBIMTorus3D.forward_belief``: next-token negative log likelihood plus the
write-port realized free energy.  Read-action KL remains diagnostic until its
categorical action likelihood is marginalized exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from information_boltzmann.core.torus3d import CBIMTorus3D, KineticBeliefState
from information_boltzmann.evaluation import WarmSiteSpec
from scripts.ib.evaluate_continuous_owt import evaluate_warm_sites


def atomic_json(path: Path, payload: dict[str, Any], *, required: bool = False) -> None:
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    encoded = json.dumps(payload, allow_nan=False)
    for attempt in range(60):
        try:
            temporary.write_text(encoded, encoding="utf-8")
            os.replace(temporary, path)
            return
        except PermissionError:
            time.sleep(0.05 * min(attempt + 1, 4))
    if required:
        raise PermissionError(f"Could not update {path}")


class TruncatedBeliefGraphTrainer:
    """CUDA-graph trainer with a persistent field and posterior precision.

    Capture removes Python launch overhead from the 64 transport/collision
    microsteps.  The graph covers a fixed BPTT chunk.  State is copied forward
    after each replay and detached at that boundary; it is never reset.
    """

    def __init__(self, model: CBIMTorus3D, *, tokens: int, chunk_tokens: int,
                 lr: float, max_grad_norm: float) -> None:
        if tokens < 1 or chunk_tokens < 1 or tokens % chunk_tokens:
            raise ValueError("tokens must be a positive multiple of chunk-tokens")
        self.model = model
        self.tokens = int(tokens)
        self.chunk_tokens = int(chunk_tokens)
        self.chunks = self.tokens // self.chunk_tokens
        self.max_grad_norm = float(max_grad_norm)
        self.ids = torch.zeros(1, chunk_tokens, dtype=torch.long, device="cuda")
        self.targets = torch.zeros_like(self.ids)
        belief = model.initial_belief(1, device="cuda", warm_start=False)
        self.field = belief.field
        self.precision = belief.precision
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, foreach=True)

        named = list(model.named_parameters())
        lexical_names = {"source.embedding.weight", "decoder.weight", "decoder.bias"}
        self.gradient_groups = (
            [parameter for name, parameter in named if name in lexical_names],
            [parameter for name, parameter in named if name not in lexical_names],
        )
        if not all(self.gradient_groups):
            raise RuntimeError("Expected nonempty lexical and dynamics parameter groups")

        original_parameters = [parameter.detach().clone() for parameter in model.parameters()]
        original_field = self.field.detach().clone()
        original_precision = self.precision.detach().clone()

        def backward_chunk():
            loss, next_belief, diagnostics = model.forward_belief(
                self.ids, self.targets,
                KineticBeliefState(self.field, self.precision),
            )
            (loss / self.chunks).backward()
            return loss, next_belief, diagnostics

        # Warm allocations must happen on a side stream before capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.optimizer.zero_grad(set_to_none=True)
                backward_chunk()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.empty_cache()
        self.optimizer.zero_grad(set_to_none=True)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.loss, next_belief, self.diagnostics = backward_chunk()
            self.next_field = next_belief.field
            self.next_precision = next_belief.precision

        with torch.no_grad():
            for parameter, original in zip(model.parameters(), original_parameters):
                parameter.copy_(original)
            self.field.copy_(original_field)
            self.precision.copy_(original_precision)
        self.optimizer.zero_grad(set_to_none=False)

    def belief(self) -> KineticBeliefState:
        return KineticBeliefState(self.field, self.precision)

    def step(self, ids: torch.Tensor, targets: torch.Tensor):
        if ids.shape != (1, self.tokens) or targets.shape != ids.shape:
            raise ValueError(f"Expected one persistent stream of {self.tokens} tokens")
        self.optimizer.zero_grad(set_to_none=False)
        loss_sum = self.loss.detach().new_zeros(())
        diagnostics: dict[str, torch.Tensor] = {}
        for offset in range(0, self.tokens, self.chunk_tokens):
            self.ids.copy_(ids[:, offset:offset + self.chunk_tokens])
            self.targets.copy_(targets[:, offset:offset + self.chunk_tokens])
            self.graph.replay()
            loss_sum.add_(self.loss.detach())
            diagnostics = {key: value.detach().clone()
                           for key, value in self.diagnostics.items()}
            with torch.no_grad():
                self.field.copy_(self.next_field.detach())
                self.precision.copy_(self.next_precision.detach())
        self.group_grad_norms = torch.stack(tuple(
            torch.nn.utils.clip_grad_norm_(group, self.max_grad_norm, foreach=True)
            for group in self.gradient_groups
        ))
        self.grad_norm = self.group_grad_norms.square().sum().sqrt()
        if not torch.isfinite(self.grad_norm):
            raise FloatingPointError("Non-finite gradient norm")
        self.optimizer.step()
        return loss_sum / self.chunks, self.belief(), diagnostics


def _batch(data: np.ndarray, offset: int, tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    return tuple(
        torch.as_tensor(np.array(data[offset + shift:offset + shift + tokens]),
                        dtype=torch.long, device="cuda")[None]
        for shift in (0, 1)
    )  # type: ignore[return-value]


def _float(value: torch.Tensor | float | int | None) -> float | None:
    if value is None:
        return None
    return float(value.detach().float().mean().cpu()) if isinstance(value, torch.Tensor) else float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/ib_owt_gpt2"))
    parser.add_argument("--output", type=Path, default=Path("results/q8_port_agents_k64_3000"))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--chunk-tokens", type=int, default=8)
    parser.add_argument("--micro-steps", type=int, default=64)
    parser.add_argument("--shape", type=int, nargs=3, default=(8, 8, 4))
    parser.add_argument("--velocities", type=int, default=8, choices=(8, 27))
    parser.add_argument("--content-dim", type=int, default=16)
    parser.add_argument("--collision-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--site-starts", type=str, default="8192,12288,16384,20480")
    parser.add_argument("--warm-in-tokens", type=int, default=256)
    parser.add_argument("--score-tokens", type=int, default=128)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    args.shape = tuple(args.shape)
    if args.micro_steps != 64:
        parser.error("The active Q8 port line fixes K=64; use a new registered branch for another K.")
    if not torch.cuda.is_available():
        parser.error("CUDA is required for the captured K=64 trainer")
    if args.steps < 1 or args.tokens < 1:
        parser.error("steps and tokens must be positive")

    torch.set_num_threads(2)
    torch.manual_seed(11)
    torch.cuda.manual_seed_all(11)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "last.pt").exists() and args.resume is None:
        parser.error("Output already contains a run; pass --resume or choose a new directory")
    dashboard = Path("present/cbim_torus3d_live.html")
    if dashboard.exists():
        shutil.copy2(dashboard, args.output / "index.html")

    train = np.load(args.data / "train.npy", mmap_mode="r")
    validation = np.load(args.data / "validation.npy", mmap_mode="r")
    if args.steps * args.tokens + 1 > len(train):
        parser.error("Training stream is shorter than requested budget")
    spec = WarmSiteSpec(
        site_starts=tuple(int(item) for item in args.site_starts.split(",") if item),
        warm_in_tokens=args.warm_in_tokens,
        score_tokens=args.score_tokens,
    )

    model = CBIMTorus3D(
        shape=args.shape, velocities=args.velocities, content_dim=args.content_dim,
        collision_layers=args.collision_layers, relative_address=True,
        readout_type="belief_agent", write_type="w4_predictive_agent",
        micro_steps=args.micro_steps, dissipation_type="quadratic",
    ).cuda()
    runner = TruncatedBeliefGraphTrainer(
        model, tokens=args.tokens, chunk_tokens=args.chunk_tokens,
        lr=args.lr, max_grad_norm=args.max_grad_norm)
    source_paths = [
        Path(__file__), Path("information_boltzmann/core/torus3d.py"),
        Path("information_boltzmann/core/readout_probes.py"),
        Path("scripts/ib/evaluate_continuous_owt.py"),
    ]
    config = {
        "architecture": model.architecture,
        "data": str(args.data), "shape": args.shape,
        "velocities": args.velocities, "content_dim": args.content_dim,
        "channels": model.d, "collision_layers": args.collision_layers,
        "write_type": "w4_predictive_agent", "readout_type": "belief_agent",
        "relative_address": True, "dissipation_type": "quadratic",
        "micro_steps": args.micro_steps, "K": args.micro_steps,
        "integration_resolution": "fixed kinetic quadrature",
        "tokens": args.tokens, "bptt_chunk_tokens": args.chunk_tokens,
        "state_policy": "one_never_reset_posterior_field_and_channel_precision",
        "objective": "next_token_nll_plus_realized_write_port_free_energy",
        "read_action_kl": "diagnostic_only_pending_categorical_action_likelihood_marginalization",
        "optimizer": "AdamW", "lr": args.lr, "max_grad_norm": args.max_grad_norm,
        "seed": 11, "steps": args.steps,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "validation_protocol": "IB-warm-local-language-v1",
        "validation_sites": spec.site_starts,
        "manifest": json.loads((args.data / "manifest.json").read_text()),
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_paths},
    }
    atomic_json(args.output / "config.json", config, required=True)
    step, best = 0, float("inf")

    if args.resume is not None:
        saved = torch.load(args.resume, map_location="cuda", weights_only=False)
        for key in ("architecture", "shape", "channels", "tokens", "K", "write_type", "readout_type"):
            if saved["config"].get(key) != config.get(key):
                parser.error(f"Resume mismatch: {key}")
        if "precision" not in saved:
            parser.error("Resume checkpoint has no posterior precision")
        model.load_state_dict(saved["model"])
        runner.optimizer.load_state_dict(saved["optimizer"])
        runner.field.copy_(saved["state"])
        runner.precision.copy_(saved["precision"])
        step, best = int(saved["step"]), float(saved["best_validation_nll"])

    def save(name: str) -> None:
        model.set_ness_prior(runner.field)
        temporary = args.output / f"{name}.{os.getpid()}.tmp"
        torch.save({
            "model": model.state_dict(), "optimizer": runner.optimizer.state_dict(),
            "state": runner.field.detach().clone(),
            "precision": runner.precision.detach().clone(), "step": step,
            "events": step * args.tokens, "best_validation_nll": best,
            "config": config,
        }, temporary)
        os.replace(temporary, args.output / name)

    def log(row: dict[str, Any]) -> None:
        with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)

    @torch.no_grad()
    def validate() -> dict[str, Any]:
        model.set_ness_prior(runner.field)
        terminal = KineticBeliefState(
            runner.field.detach().clone(), runner.precision.detach().clone())
        return evaluate_warm_sites(
            model, terminal, validation, spec, args.micro_steps, ness_phase_seed=11)

    try:
        if args.resume is None:
            atomic_json(args.output / "progress.json", {
                "status": "capturing", "step": 0, "target_steps": args.steps}, required=True)
            initial = validate()
            best = float(initial["nll"])
            log({"kind": "validation", "step": 0, "validation_nll": best,
                 "sites": initial["sites_count"]})
            save("BBest.pt")
            save("last.pt")

        while step < args.steps:
            ids, targets = _batch(train, step * args.tokens, args.tokens)
            torch.cuda.synchronize()
            started = time.perf_counter()
            loss, belief, diagnostics = runner.step(ids, targets)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            step += 1
            if not math.isfinite(float(loss)):
                raise FloatingPointError("Non-finite training loss")
            if step == 1 or step % 10 == 0 or step == args.steps:
                field = belief.field.detach()
                row = {
                    "kind": "train", "step": step, "events": step * args.tokens,
                    "objective": _float(loss), "token_nll": _float(diagnostics.get("token_nll")),
                    "write_free_energy": _float(diagnostics.get("write_free_energy_mean")),
                    "read_action_complexity": _float(diagnostics.get("read_action_complexity_mean")),
                    "seconds": elapsed, "energy": _float(diagnostics.get("energy")),
                    "incident_energy": _float(diagnostics.get("incident_energy")),
                    "reflected_energy": _float(diagnostics.get("reflected_energy")),
                    "accepted_fraction": _float(diagnostics.get("accepted_fraction")),
                    "write_angle_abs_mean": _float(diagnostics.get("write_angle_abs_mean")),
                    "collision_angle_abs_mean": _float(diagnostics.get("collision_angle_abs_mean")),
                    "transport_angle_abs_mean": _float(diagnostics.get("transport_angle_abs_mean")),
                    "bath_out_energy": _float(diagnostics.get("bath_out_energy")),
                    "read_attention_entropy": _float(diagnostics.get("read_attention_entropy")),
                    "read_action_entropy": _float(diagnostics.get("read_action_entropy")),
                    "read_action_kl": _float(diagnostics.get("read_action_kl")),
                    "read_temperature_mean": _float(diagnostics.get("read_temperature_mean")),
                    "prior_precision_mean": _float(diagnostics.get("prior_precision_mean")),
                    "posterior_precision_mean": _float(diagnostics.get("posterior_precision_mean")),
                    "grad_norm": _float(runner.grad_norm),
                    "allocated_mib": torch.cuda.memory_allocated() / 2**20,
                    "reserved_mib": torch.cuda.memory_reserved() / 2**20,
                }
                log(row)
                atomic_json(args.output / "progress.json", {
                    "status": "running", "target_steps": args.steps, **row})
                atomic_json(args.output / "live_state.json", {
                    **row, "run_id": args.output.name,
                    "volume": field[0].square().mean(-1).sqrt().cpu().tolist(),
                })
            if step % args.validate_every == 0 or step == args.steps:
                report = validate()
                score = float(report["nll"])
                if score < best:
                    best = score
                    save("BBest.pt")
                log({"kind": "validation", "step": step,
                     "validation_nll": score, "best_validation_nll": best,
                     "sites": report["sites_count"]})
                save("last.pt")
        atomic_json(args.output / "progress.json", {
            "status": "complete", "step": step, "target_steps": args.steps,
            "best_validation_nll": best}, required=True)
    except BaseException as error:
        atomic_json(args.output / "progress.json", {
            "status": "failed", "step": step, "target_steps": args.steps,
            "error": str(error)}, required=True)
        raise


if __name__ == "__main__":
    main()
