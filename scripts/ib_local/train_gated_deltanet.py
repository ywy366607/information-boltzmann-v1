"""Train the capacity-matched Gated DeltaNet baseline on OpenWebText BPE."""
from __future__ import annotations

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(os.path.dirname(script_dir))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import subprocess

import numpy as np
import torch

from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
from scripts.ib_local.train_cbim_malecns_internal_time import (
    TruncatedInternalTimeGraphTrainer,
)


def atomic_json(path: Path, data, required=False):
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    payload = json.dumps(data, allow_nan=False)
    for attempt in range(60):
        try:
            temporary.write_text(payload, encoding="utf-8")
            os.replace(temporary, path)
            return True
        except PermissionError:
            time.sleep(0.05 * min(attempt + 1, 4))
    if required:
        raise PermissionError(f"Unable to update {path}")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/ib_owt_gpt2"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/gated_deltanet_d128_3000"),
    )
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--validation-tokens", type=int, default=4096)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--continuous-bytes", action="store_true",
                        help="Continuous UTF-8 A/B/A/fresh protocol; original BPE mode unchanged")
    parser.add_argument("--recurrence", choices=["decay_then_delta", "feedback_delta", "evidence_exchange", "contact_exchange"],
                        default="decay_then_delta")
    parser.add_argument("--content-budget", type=float, default=1.0)
    parser.add_argument("--association-budget", type=float, default=0.,
                        help="Same row/column L1 cap for delta and feedback_delta; 0 disables")
    parser.add_argument("--contact-availability", action="store_true",
                        help="Use the same contact share as available capacity; contact_exchange only")
    parser.add_argument("--chunk-tokens", type=int, default=32)
    parser.add_argument("--mlp-hidden", type=int, default=256)
    parser.add_argument("--conv-kernel", type=int, default=4)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--paired-feedback", action="store_true",
                        help="Sequential matched delta/feedback runs and existing frozen evaluations")
    parser.add_argument("--decay-init", choices=["constant", "log_timescale"], default="constant",
                        help="log_timescale: official Gated DeltaNet A_log/dt_bias retention")
    parser.add_argument("--feedback-init", choices=["constant", "threshold_spread"], default="constant",
                        help="threshold_spread: feedback heads born across the latch threshold")
    parser.add_argument("--fused-recurrence", action="store_true",
                        help="Triton window kernels (gdn_fused.py); exact up to float32 rounding")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    if args.fused_recurrence:
        os.environ["CGB_FUSED_DELTA"] = "1"  # inherited by paired arms and evaluators
        from scripts.ib_local import gdn_fused
        gdn_fused.enable()
    if args.paired_feedback:
        if not args.continuous_bytes or args.calibrate or args.resume:
            raise ValueError("paired feedback requires continuous bytes without calibration/resume")
        run_feedback_pair(args)
        return
    if args.continuous_bytes:
        run_continuous_bytes(args)
        return

    torch.set_num_threads(2)
    torch.manual_seed(11)
    torch.cuda.set_per_process_memory_fraction(0.90)
    args.output.mkdir(parents=True, exist_ok=True)

    train = np.load(args.data / "train.npy", mmap_mode="r")
    valid = np.load(args.data / "validation.npy", mmap_mode="r")

    model = GatedDeltaNetLM(
        vocab_size=50257,
        d=args.d,
        layers=args.layers,
        heads=args.heads,
    ).cuda()

    runner = TruncatedInternalTimeGraphTrainer(
        model, tokens=args.tokens, chunk_tokens=args.tokens
    )

    sources = [Path(__file__), Path("scripts/ib_local/gated_deltanet.py")]
    non_emb_params = sum(
        p.numel()
        for n, p in model.named_parameters()
        if "embedding" not in n and "decoder" not in n
    )

    config = {
        "architecture": model.architecture,
        "data": str(args.data),
        "output": str(args.output),
        "steps": args.steps,
        "tokens": args.tokens,
        "d": args.d,
        "layers": args.layers,
        "heads": args.heads,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "non_embedding_parameters": non_emb_params,
        "cbim_reference_non_embedding": 332724,
        "parameter_delta_ratio": abs(non_emb_params - 332724) / 332724,
        "precision": "FP32",
        "seed": 11,
        "lr": 3e-4,
        "validate_every": args.validate_every,
        "validation_tokens": args.validation_tokens,
        "manifest": json.loads((args.data / "manifest.json").read_text()),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
    }

    atomic_json(args.output / "config.json", config, required=True)
    atomic_json(
        args.output / "progress.json",
        {"status": "capturing", "step": 0, "target_steps": args.steps},
    )

    step, best = 0, float("inf")
    if args.resume:
        saved = torch.load(args.resume, map_location="cuda", weights_only=False)
        for key in ("architecture", "d", "layers", "tokens"):
            if saved["config"][key] != config[key]:
                raise ValueError(f"Resume mismatch: {key}")
        model.load_state_dict(saved["model"])
        runner.optimizer.load_state_dict(saved["optimizer"])
        runner.state.copy_(saved["state"])
        step, best = saved["step"], saved["best_validation_nll"]

    def batch(data, offset):
        return tuple(
            torch.as_tensor(
                np.array(data[offset + shift : offset + shift + args.tokens]),
                dtype=torch.long,
                device="cuda",
            )[None]
            for shift in (0, 1)
        )

    def save(name):
        temporary = args.output / f"{name}.{os.getpid()}.tmp"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": runner.optimizer.state_dict(),
                "state": runner.state.detach(),
                "step": step,
                "events": step * args.tokens,
                "best_validation_nll": best,
                "config": config,
            },
            temporary,
        )
        os.replace(temporary, args.output / name)

    def log(row):
        with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)

    @torch.no_grad()
    def validate():
        state, total = model.initial_state(1, "cuda"), 0.0
        for offset in range(0, args.validation_tokens + args.tokens, args.tokens):
            ids, targets = batch(valid, offset)
            loss, state, _ = model(ids, targets, state)
            if offset:
                total += float(loss) * args.tokens
        return total / args.validation_tokens

    try:
        if args.resume is None:
            best = validate()
            log({"kind": "validation", "step": 0, "validation_nll": best, "best_validation_nll": best})
            save("BBest.pt")
            save("last.pt")

        while step < args.steps:
            ids, targets = batch(train, step * args.tokens)
            torch.cuda.synchronize()
            started = time.perf_counter()
            loss, state, diagnostics = runner.step(ids, targets)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            step += 1

            if not math.isfinite(float(loss)):
                raise FloatingPointError("Non-finite training loss")
            if elapsed > 1.5 and step > 2:
                raise RuntimeError(f"Training step exceeded 1.5 seconds: {elapsed:.3f}")

            if step == 1 or step % 10 == 0 or step == args.steps:
                row = {
                    "kind": "train",
                    "step": step,
                    "events": step * args.tokens,
                    "nll": float(loss),
                    "seconds": elapsed,
                    "alpha_mean": float(diagnostics["alpha_mean"]),
                    "beta_mean": float(diagnostics["beta_mean"]),
                    "state_norm": float(diagnostics["state_norm"]),
                    "error_norm": float(diagnostics["error_norm"]),
                    "grad_norm": float(runner.grad_norm),
                    "grad_norm_lexical": float(runner.group_grad_norms[0]),
                    "grad_norm_dynamics": float(runner.group_grad_norms[1]),
                    "allocated_mib": torch.cuda.memory_allocated() / 2**20,
                    "reserved_mib": torch.cuda.memory_reserved() / 2**20,
                }
                log(row)
                atomic_json(
                    args.output / "progress.json",
                    {"status": "running", "target_steps": args.steps, **row},
                )

            if step % args.validate_every == 0 or step == args.steps:
                score = validate()
                improved = score < best
                if improved:
                    best = score
                log(
                    {
                        "kind": "validation",
                        "step": step,
                        "validation_nll": score,
                        "best_validation_nll": best,
                    }
                )
                if improved:
                    save("BBest.pt")
                save("last.pt")

        atomic_json(
            args.output / "progress.json",
            {
                "status": "complete",
                "step": step,
                "target_steps": args.steps,
                "best_validation_nll": best,
            },
        )
    except BaseException as error:
        atomic_json(
            args.output / "progress.json",
            {"status": "failed", "step": step, "error": str(error)},
        )
        raise


def byte_protocol(raw, steps, tokens):
    """All phase boundaries score their actual next input, including jumps.

    A/B are text intervals, not asserted semantic domains. 70% initial traffic,
    10% new B, 10% revisit of A's final interval, 10% fresh C. No state resets.
    Source is already decoded to bytes, without tokenizer control symbols.
    """
    if steps < 10 or steps % 10:
        raise ValueError("continuous protocol requires steps divisible by ten")
    unit = steps // 10 * tokens
    specs = [("A_first", 0, 7*unit), ("B_fresh", 7*unit, 8*unit),
             ("A_revisit", 6*unit, 7*unit), ("C_fresh", 8*unit, 9*unit)]
    if len(raw) < 9*unit+1:
        raise ValueError("insufficient source bytes")
    stream = np.concatenate([raw[start:stop] for _, start, stop in specs]
                            + [raw[9*unit:9*unit+1]])
    phases, cursor = [], 0
    for name, start, stop in specs:
        end = cursor + (stop-start)//tokens
        phases.append(dict(name=name, start_update=cursor, stop_update=end,
                           source_byte_start=start, source_byte_stop=stop,
                           replay=name == "A_revisit"))
        cursor = end
    return stream, phases


class GraphByteEvaluator:
    """Independent frozen-weight validation, with causal state within the split.

    This sidecar never resets or modifies the live training state/optimizer.
    It is secondary to the continuous prequential training protocol.
    """
    def __init__(self, model, chunk):
        self.model, self.chunk = model, chunk
        self.ids = torch.zeros(1, chunk, dtype=torch.long, device="cuda")
        self.targets = torch.zeros_like(self.ids)
        self.state = model.initial_state(1)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(stream):
            for _ in range(3):
                model(self.ids, self.targets, self.state)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.graph):
            self.loss, self.next_state, self.diagnostics = model(self.ids, self.targets, self.state)

    @torch.no_grad()
    def score(self, data, count, *, erase_at_chunk=False):
        self.state.copy_(self.model.initial_state(1))
        total = torch.zeros((), device="cuda")
        for offset in range(0, count+self.chunk, self.chunk):
            if erase_at_chunk:
                # Remove old matrix contents, retaining convolution cache and shares.
                self.state[..., :self.model.layers[0].matrix_size].zero_()
            self.ids.copy_(data[:, offset:offset+self.chunk])
            self.targets.copy_(data[:, offset+1:offset+self.chunk+1])
            self.graph.replay()
            self.state.copy_(self.next_state)
            if offset:  # common unscored warmup, never optimized on validation
                total.add_(self.loss)
        return float(total) / (count//self.chunk)


def run_continuous_bytes(args):
    """Single-seed, bounded-window training of the existing full neural stack."""
    if args.tokens % args.chunk_tokens or args.validation_tokens % args.chunk_tokens:
        raise ValueError("update and validation lengths must divide by chunk length")
    if args.resume:
        raise ValueError("this new protocol does not yet expose CLI resume; complete snapshots are saved")
    if (args.output / "config.json").exists():
        raise FileExistsError("refusing to overwrite an existing run")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.cuda.set_per_process_memory_fraction(0.90)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
    tokenizer = json.loads((args.data / "tokenizer.json").read_text(encoding="utf-8"))
    if manifest["source"] != "openwebtext" or tokenizer["type"] != "utf8_bytes":
        raise ValueError("requires registered OpenWebText byte source")
    offset = tokenizer["offset"]
    source_hashes = {}
    arrays = {}
    for split in ("train", "validation"):
        path = args.data / f"{split}.npy"
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 << 20), b""):
                digest.update(block)
        source_hashes[split] = digest.hexdigest()
        if source_hashes[split] != manifest["files"][f"{split}.npy"]:
            raise ValueError(f"source hash mismatch: {split}")
        raw = np.load(path, mmap_mode="r")
        # Bound preparation memory; only a prefix is needed. Control IDs are dropped,
        # document text is concatenated without invented bytes or explicit resets.
        need = args.steps*args.tokens+1 if split == "train" else args.validation_tokens+args.chunk_tokens+1
        prefix = np.asarray(raw[:min(len(raw), 2*need)])
        decoded = prefix[prefix >= offset] - offset
        if len(decoded) < need or decoded.max() > 255:
            raise ValueError("invalid/insufficient UTF-8 byte source")
        arrays[split] = decoded[:need].astype(np.uint8)
    stream, phases = byte_protocol(arrays["train"], args.steps, args.tokens)
    data = torch.from_numpy(stream.astype(np.int64)).cuda()[None]
    validation = torch.from_numpy(arrays["validation"].astype(np.int64)).cuda()[None]
    model = GatedDeltaNetLM(vocab_size=256, d=args.d, layers=args.layers, heads=args.heads,
                           recurrence=args.recurrence, mlp_hidden=args.mlp_hidden,
                           conv_kernel=args.conv_kernel,
                           contact_availability=args.contact_availability,
                           content_budget=args.content_budget if args.recurrence in
                           ("evidence_exchange", "contact_exchange") else args.association_budget,
                           decay_init=args.decay_init,
                           feedback_init=args.feedback_init if args.recurrence == "feedback_delta"
                           else "constant").cuda()
    sources = [Path(__file__), Path("scripts/ib_local/gated_deltanet.py"),
               Path("scripts/ib_local/train_cbim_malecns_internal_time.py")]
    if args.fused_recurrence:
        sources.append(Path("scripts/ib_local/gdn_fused.py"))
    config = dict(model=model.config_dict(), seed=args.seed, fused_recurrence=args.fused_recurrence, steps=args.steps, tokens=args.tokens,
                  chunk_tokens=args.chunk_tokens, lr=3e-4, weight_decay=0.01,
                  precision="FP32", optimizer="AdamW, foreach; lexical/dynamics norm clip 1",
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  state_bytes=model.initial_state(1).numel()*4,
                  state_encoding=("conditional_content_C_plus_share_a" if args.recurrence == "contact_exchange"
                                  else "weighted_content_M_plus_share_a" if args.recurrence == "evidence_exchange"
                                  else "ordinary_association_matrix"),
                  source_revision=manifest["revision"], split_rule=manifest["split_rule"],
                  source_sha256=source_hashes, data_path=str(args.data),
                  tokenizer_sha256=hashlib.sha256((args.data/"tokenizer.json").read_bytes()).hexdigest(),
                  source_conversion="drop control IDs; byte=ID-offset; concatenate documents; no reset",
                  stream_sha256=hashlib.sha256(stream.tobytes()).hexdigest(),
                  validation_sha256=hashlib.sha256(arrays["validation"].tobytes()).hexdigest(),
                  code_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  phases=phases, device=torch.cuda.get_device_name(),
                  torch_version=torch.__version__, validation_tokens=args.validation_tokens,
                  calibrate=args.calibrate, review="not independently reviewed",
                  stopping_rule="fixed update budget; no convergence claim; abort nonfinite",
                  comparison_scope="shared initialization/data/optimizer; feedback adds 2 scalars per head; explicit actual matrix budget",
                  feedback_parameterization=("eta=sigmoid(head_logit), gain=softplus(head_raw), cue=value; init eta=.1,gain=2"
                                             if args.recurrence == "feedback_delta" else None))
    atomic_json(args.output/"config.json", config, required=True)
    atomic_json(args.output/"progress.json", dict(status="capturing", step=0), required=True)
    started = time.perf_counter()
    runner = TruncatedInternalTimeGraphTrainer(model, tokens=args.tokens,
                                              chunk_tokens=args.chunk_tokens, lr=3e-4)
    evaluator = GraphByteEvaluator(model, args.chunk_tokens)
    torch.cuda.synchronize()
    capture_seconds = time.perf_counter()-started
    initial_validation = evaluator.score(validation, args.validation_tokens)
    step, best = 0, initial_validation
    phase_sums, times, window_losses = {}, [], []

    def log(row):
        with (args.output/"metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False)+"\n")
        print(json.dumps(row), flush=True)

    def save(name):
        tmp = args.output/(name+".tmp")
        torch.save(dict(config=config, model=model.state_dict(), optimizer=runner.optimizer.state_dict(),
                        state=runner.state.detach(), step=step, events=step*args.tokens,
                        best_validation_nll=best, phase_sums=phase_sums,
                        rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all()), tmp)
        tmp.replace(args.output/name)

    log(dict(kind="initial", validation_nll=best, capture_seconds=capture_seconds,
             parameters=config["parameter_count"], state_bytes=config["state_bytes"]))
    torch.cuda.reset_peak_memory_stats()
    run_started = time.perf_counter()
    try:
        for index in range(args.steps):
            phase = next(p["name"] for p in phases if p["start_update"] <= index < p["stop_update"])
            at = index*args.tokens
            torch.cuda.synchronize()
            tick = time.perf_counter()
            loss, state, diag = runner.step(data[:, at:at+args.tokens], data[:, at+1:at+args.tokens+1])
            torch.cuda.synchronize()
            duration = time.perf_counter()-tick
            value = float(loss)
            if not math.isfinite(value) or not bool(torch.isfinite(state).all()):
                raise FloatingPointError("nonfinite loss or recurrent state")
            step = index+1
            times.append(duration)
            window_losses.append(value)
            totals = phase_sums.setdefault(phase, dict(nll_sum=0., updates=0, bytes=0))
            totals["nll_sum"] += value*args.tokens
            totals["updates"] += 1
            totals["bytes"] += args.tokens
            if args.steps*6//10 <= index < args.steps*7//10:
                aligned = phase_sums.setdefault("A_matched_first", dict(nll_sum=0., updates=0, bytes=0))
                aligned["nll_sum"] += value*args.tokens
                aligned["updates"] += 1
                aligned["bytes"] += args.tokens
            if step == 1 or step % 50 == 0 or step == args.steps:
                diagnostics = {k: float(v) for k, v in diag.items()}
                row = dict(kind="train", step=step, phase=phase, events=step*args.tokens,
                           window_nll=float(np.mean(window_losses)), window_updates=len(window_losses),
                           recent_bytes_per_second=args.tokens/float(np.median(times[-50:])),
                           gradient_norm=float(runner.grad_norm), diagnostics=diagnostics,
                           allocated_mib=torch.cuda.memory_allocated()/2**20,
                           peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
                log(row)
                window_losses.clear()
                atomic_json(args.output/"progress.json", dict(status="running", target_steps=args.steps, **row))
            if step % args.validate_every == 0 or step == args.steps:
                score = evaluator.score(validation, args.validation_tokens)
                log(dict(kind="validation_frozen_sidecar", step=step, validation_nll=score))
                if score < best:
                    best = score
                    save("best.pt")
                save("last.pt")
        cut_score = evaluator.score(validation, args.validation_tokens, erase_at_chunk=True)
        report = dict(status="complete", step=step, config=config, initial_validation_nll=initial_validation,
                      final_validation_nll=score, best_validation_nll=best,
                      final_validation_erased_each_chunk=cut_score,
                      cross_chunk_memory_nll_cost=cut_score-score,
                      phase_scores={p: dict(**v, mean_nll=v["nll_sum"]/v["bytes"]) for p,v in phase_sums.items()},
                      capture_seconds=capture_seconds, wall_seconds=time.perf_counter()-run_started,
                      median_update_seconds=float(np.median(times[1:])),
                      training_bytes_per_second=args.tokens/float(np.median(times[1:])),
                      peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                      peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                      convergence_established=False, organism_admitted=False)
        atomic_json(args.output/"summary.json", report, required=True)
        atomic_json(args.output/"progress.json", dict(status="complete", step=step, best_validation_nll=best))
        log(dict(kind="complete", **{k: v for k,v in report.items() if k not in ("config", "phase_scores")}))
    except BaseException as error:
        atomic_json(args.output/"progress.json", dict(status="failed", step=step, error=str(error)), required=True)
        raise


def run_feedback_pair(args):
    """Use existing trainers/evaluators, with immutable-source paired provenance."""
    from scripts.ib_local.evaluate_gated_deltanet import aligned_revisit_blocks
    root = Path(repo_root)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__).resolve(), root/"scripts/ib_local/gated_deltanet.py",
                    root/"scripts/ib_local/train_cbim_malecns_internal_time.py",
                    root/"scripts/ib_local/evaluate_gated_deltanet.py",
                    root/"scripts/ib_local/evaluate_lambada.py"]
    if args.fused_recurrence:
        source_paths.append(root/"scripts/ib_local/gdn_fused.py")
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    plan = dict(protocol="same_body_delta_vs_readback_v1", status="registered", seed=args.seed,
                steps=args.steps, bytes_per_arm=args.steps*args.tokens,
                settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                code_sha256=hashes, model_difference="two learned scalars per head; no new recurrent storage",
                target="language prediction, matched revisit, state retrieval, plasticity and resource cost",
                not_claimed="whole-life closure, trained latch or convergence from a fixed budget",
                primary="continuous prequential A/B/A/fresh; identical bytes and optimizer cadence",
                secondary="independent validation, memory erasure, paired context and full LAMBADA development",
                stop="nonfinite; source mismatch; command failure; otherwise registered budget")
    atomic_json(output/"plan.json", plan, required=True)

    def check_sources():
        for name,digest in hashes.items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
                raise RuntimeError("paired source changed: "+name)

    def execute(label, command):
        check_sources()
        atomic_json(output/"progress.json", dict(status="running",stage=label), required=True)
        with (output/(label+".log")).open("w",encoding="utf-8") as handle:
            subprocess.run([sys.executable,"-u",*map(str,command)], cwd=root,
                           stdout=handle,stderr=subprocess.STDOUT,check=True)
        check_sources()

    try:
        summaries = {}
        for arm in ("decay_then_delta", "feedback_delta"):
            run = output/arm
            execute(arm,[Path(__file__).resolve(),"--continuous-bytes","--data",args.data.resolve(),
                         "--output",run,"--recurrence",arm,"--association-budget",args.association_budget,
                         "--steps",args.steps,"--tokens",args.tokens,"--chunk-tokens",args.chunk_tokens,
                         "--d",args.d,"--layers",args.layers,"--heads",args.heads,
                         "--mlp-hidden",args.mlp_hidden,"--conv-kernel",args.conv_kernel,
                         "--validate-every",args.validate_every,"--validation-tokens",args.validation_tokens,
                         "--seed",args.seed,"--decay-init",args.decay_init,"--feedback-init",args.feedback_init,
                         *(["--fused-recurrence"] if args.fused_recurrence else [])])
            summary=json.loads((run/"summary.json").read_text(encoding="utf-8"))
            rows=[json.loads(line) for line in (run/"metrics.jsonl").read_text().splitlines()]
            summary["aligned_revisit"]=aligned_revisit_blocks(summary["config"],rows)
            summaries[arm]=summary
        left,right=(summaries[arm]["config"] for arm in ("decay_then_delta","feedback_delta"))
        for field in ("stream_sha256","validation_sha256","state_bytes","steps","tokens","chunk_tokens"):
            if left[field]!=right[field]:raise ValueError("pair mismatch: "+field)
        atomic_json(output/"training_comparison.json",dict(status="training_complete_evaluations_pending",arms=summaries),required=True)
        for arm in summaries:
            checkpoint=output/arm/"last.pt"
            execute(arm+"_context",[root/"scripts/ib_local/evaluate_gated_deltanet.py",
                    "--checkpoint",checkpoint,"--data",args.data.resolve(),"--output",output/(arm+"_context.json"),
                    "--prefixes",32,128,512,2048,8192,32768,"--samples",8,"--scored",4096,
                    "--exclude",args.validation_tokens+args.chunk_tokens,"--chunk",min(args.chunk_tokens,32)])
            execute(arm+"_lambada",[root/"scripts/ib_local/evaluate_lambada.py","--checkpoint",checkpoint,
                    "--output",output/(arm+"_lambada.json"),"--chunk",min(args.chunk_tokens,32)])
        atomic_json(output/"progress.json",dict(status="complete",training_comparison="training_comparison.json",
                    convergence_established=False,organism_admitted=False),required=True)
    except BaseException as error:
        atomic_json(output/"progress.json",dict(status="failed",error=str(error)),required=True)
        raise


if __name__ == "__main__":
    main()
