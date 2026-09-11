"""Train or evaluate the same persistent event loop; no prefix resets."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fine_grain.information_boltzmann import InformationBoltzmann
from fine_grain.information_boltzmann.streaming import StreamRunner
from fine_grain.information_boltzmann.async_monitor import AsyncMonitor


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="Prepared data directory")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--targets", type=int, help="Total observed events, including resumed events")
    parser.add_argument("--output", type=Path, default=Path("results/information_boltzmann"))
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument("--resume", type=Path, help="Exact continuation of the same stream")
    checkpoint.add_argument("--weights", type=Path, help="Load weights and start a separate evaluation stream")
    parser.add_argument("--frozen", action="store_true", help="No online optimizer")
    parser.add_argument("--monitor", action="store_true", help="Independent TensorBoard/animation worker")
    parser.add_argument("--serve", action="store_true", help="Launch live web telemetry server at http://localhost:8080")
    parser.add_argument("--port", type=int, default=8080, help="Web telemetry port (default 8080)")
    parser.add_argument("--telemetry-interval", type=int, default=1, help="Events between telemetry updates (default 1)")
    parser.add_argument("--allow-smoke-data", action="store_true", help="Explicitly allow a tiny fixture, never a quality benchmark")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use the vox environment or --device cpu")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest_path = args.data / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["vocab_size"] != config["data"]["vocab_size"]:
        raise ValueError("Config vocabulary does not match the prepared tokenizer")
    if manifest["source"] == "openwebtext" and not manifest.get("revision"):
        raise ValueError("OpenWebText requires a pinned revision")
    if config["data"]["dataset"] != manifest["source"]:
        raise ValueError("Config dataset does not match prepared source")
    if manifest["source"] == "openwebtext" and not args.allow_smoke_data:
        if len(manifest.get("documents", [])) < 1000 or min(manifest["token_counts"].get(s, 0) for s in ("validation", "test")) == 0:
            raise ValueError("Tiny OpenWebText fixture rejected. Prepare >=1000 documents with held-out streams, or explicitly --allow-smoke-data")
    if (args.output / "last.pt").exists() and not args.resume:
        raise ValueError("Existing training run: use --resume to preserve cursor, or a new output directory")
    tokens_path = args.data / f"{args.split}.npy"
    if file_hash(tokens_path) != manifest["files"][tokens_path.name]:
        raise ValueError("Prepared token stream checksum differs")
    tokens = np.load(tokens_path, mmap_mode="r")
    count = args.targets if args.targets is not None else config["train"]["supervised_targets"]
    if count < 1 or count > len(tokens):
        raise ValueError(f"Requested {count} targets but stream contains {len(tokens)}; no silent repetition")
    torch.manual_seed(config["seed"])
    model = InformationBoltzmann.from_config(config).to(args.device)
    if args.weights:
        saved = torch.load(args.weights, map_location=args.device, weights_only=True)
        if saved["metadata"]["manifest_sha256"] != file_hash(manifest_path):
            raise ValueError("Evaluation data/tokenizer manifest differs from checkpoint")
        model.load_state_dict(saved["model"])
        model.force.gamma = float(saved.get("force_gamma", model.force.gamma))
        model.force.kappa = float(saved.get("force_kappa", model.force.kappa))
    settings = config["train"]
    optimizer = None if args.frozen else torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                                         weight_decay=settings["weight_decay"])
    runner = StreamRunner(model, manifest["bos_token"], config["seed"], optimizer,
                          settings["update_every_real_tokens"], settings["gradient_clip"])
    metadata = {"config": config, "manifest_sha256": file_hash(manifest_path),
                "split": args.split}
    if args.resume:
        old = runner.load(args.resume)
        for key in metadata:
            if old[key] != metadata[key]:
                raise ValueError(f"Resume provenance mismatch: {key}")
    if count <= runner.events:
        raise ValueError("Target budget must exceed checkpoint cursor")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started, initial_events = time.perf_counter(), runner.events
    metrics = []
    nll_before = runner.total_nll

    telemetry = AsyncMonitor(args.output, port=args.port if args.serve else 0) if (args.monitor or args.serve) else None

    def token_to_char(t_id: int) -> str:
        if t_id < 4:
            return ["<pad>", "<bos>", "<eos>", "<unk>"][t_id]
        byte_val = t_id - 4
        try:
            return bytes([byte_val]).decode("utf-8")
        except Exception:
            return f"\\x{byte_val:02x}"

    for cursor in range(runner.events, count):
        if optimizer is not None and not runner.loss_terms:
            warmup = max(1, int(settings["supervised_targets"] * settings["warmup_fraction"]))
            fraction = min(1.0, (cursor + settings["update_every_real_tokens"]) / warmup)
            for group in optimizer.param_groups:
                group["lr"] = settings["learning_rate"] * fraction

        pred_logits = runner.predict()  # no access to tokens[cursor] before this call

        tok = int(tokens[cursor])
        ce = runner.observe(tok)

        # Real telemetry update from live state
        if telemetry is not None and telemetry.due() and ((cursor + 1) % args.telemetry_interval == 0 or cursor + 1 == count):
            ke = float(runner.state.moments()["kinetic_energy"].item())
            pe = 0.5 * model.force.kappa * float(runner.state.moments()["position_second_moment"].item())
            var_x = float(runner.state.moments()["position_variance"].item())
            var_v = float(runner.state.moments()["velocity_variance"].item())
            dim = model.force.net[-1].out_features

            # Expensive eigendecomposition belongs to offline diagnostics.
            probs = torch.softmax(pred_logits,dim=-1)
            top = torch.topk(probs,5)
            top5_tokens = [{"token":token_to_char(i),"prob":p} for i,p in zip(top.indices.cpu().tolist(),top.values.cpu().tolist())]
            telemetry.push_frame({
                "status": "training",
                "event": cursor + 1,
                "token_id": tok,
                "token_char": token_to_char(tok),
                "ce": ce,
                "mean_nll": runner.total_nll / runner.events,
                "perplexity": math.exp(runner.total_nll / runner.events),
                "lr": optimizer.param_groups[0]["lr"] if optimizer is not None else 0.0,
                "particles": {
                    "x": runner.state.x[:128].detach().cpu().numpy().tolist(),
                    "v": runner.state.v[:128].detach().cpu().numpy().tolist(),
                },
                "collision_pairs": getattr(runner, "last_collision_pairs", []),
                "accepted_collisions": runner.accepted,
                "metrics": {
                    "var_x": var_x,
                    "var_v": var_v,
                    "energy": ke + pe,
                    "ke": ke,
                    "t_eff": var_v / dim,
                    "gamma": float(model.force.damping(runner.state.x).mean().detach()),
                },
                "top5": top5_tokens,
            })

        if (cursor + 1) % settings["update_every_real_tokens"] == 0 or cursor + 1 == count:
            row = {"event": cursor + 1, "ce": ce, "mean_nll": runner.total_nll / runner.events,
                   "accepted_collisions": runner.accepted,
                   **{k: float(v.detach()) for k, v in runner.state.moments().items()}}
            if not all(math.isfinite(v) for v in row.values()):
                raise FloatingPointError("Nonfinite phase diagnostic")
            metrics.append(row)
            print(json.dumps(row), flush=True)

    runner.flush()
    if args.device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if telemetry is not None:
        telemetry.close()
    metadata["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    runner.save(args.output / "last.pt", metadata)
    summary = {
        "status": "smoke_only_not_language_quality_or_NESS_evidence",
        "device": args.device, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "parameters": sum(p.numel() for p in model.parameters()),
        "events": runner.events, "new_events": count - initial_events, "updates": runner.updates,
        "prequential_nll": runner.total_nll / runner.events,
        "new_events_nll": (runner.total_nll - nll_before) / (count - initial_events),
        "perplexity": math.exp(runner.total_nll / runner.events),
        "seconds": elapsed, "targets_per_second": (count - initial_events) / elapsed,
        "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else 0,
        "collision_candidates": runner.candidates, "accepted_collisions": runner.accepted,
        "collision_cross_moment_abs_sum": runner.cross_moment_abs_sum,
        "last_gradient_norm": runner.last_gradient_norm,
        "phase_time": runner.state.time, "metrics": metrics, "metadata": metadata,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("metadata", "metrics")}), flush=True)


if __name__ == "__main__":
    main()
