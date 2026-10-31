"""Held-out accounting or a fresh local-port numerical wiring audit.

Default checkpoint measurement is read-only. The local-port fixture is not
a selected brain architecture or a capability study, including when edits run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_core(name: str):
    path = ROOT / "information_boltzmann" / "core" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unigram_nlls(train_tokens, targets, vocab_size: int, alpha: float):
    if alpha <= 0.0 or not math.isfinite(alpha):
        raise ValueError("alpha must be finite and positive")
    counts = np.zeros(vocab_size, dtype=np.float64)
    for start in range(0, len(train_tokens), 1_000_000):
        counts += np.bincount(train_tokens[start:start + 1_000_000], minlength=vocab_size)
    probabilities = (counts + alpha) / (float(len(train_tokens)) + alpha * vocab_size)
    return (-math.log(float(probabilities[int(token)])) for token in targets)


def consume_prequential(bridge, events, observation_type, count, write_units, dissipation_units):
    """Prepare before pulling each event from the external iterator."""
    source = iter(events)
    total_nll = 0.0
    last_event = None
    for _ in range(count):
        bridge.prepare()
        last_event = next(source)
        result = bridge.step(observation_type(last_event.tick, last_event.payload),
                             write_units=write_units, dissipation_units=dissipation_units)
        total_nll += result.model_nll_nats
    return total_nll, last_event


def fixture_rule(code, hidden, bit, context, depth, prefix):
    return ((code + 9 if hidden else 1) * (3 if hidden == bit else 1)
            * (2 if hidden == ((context[0] >> depth) & 1) else 1))


def fixture_advance(context, token, posterior_hidden):
    return (token,)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learner", choices=("checkpoint", "local-port"), default="checkpoint")
    parser.add_argument("--alphabet", choices=("byte", "gpt2"))
    parser.add_argument("--transition", choices=("joint", "legacy"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--train-tokens", type=Path)
    parser.add_argument("--validation-tokens", type=Path)
    parser.add_argument("--events", type=int, default=1024)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--start-tick", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--local-learning", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-rng-device-mismatch", action="store_true")
    parser.add_argument("--write-units", type=int, default=1)
    parser.add_argument("--dissipation-units", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    local = args.learner == "local-port"
    alphabet = args.alphabet or ("byte" if local else "gpt2")
    transition = args.transition or ("joint" if alphabet == "byte" else "legacy")
    if not local and (alphabet != "gpt2" or args.transition is not None):
        raise ValueError("old checkpoints require GPT2 and their existing transition")
    folder = ROOT / "data" / ("ib_owt_10000" if alphabet == "byte" else "ib_owt_gpt2")
    args.train_tokens = args.train_tokens or folder / "train.npy"
    args.validation_tokens = args.validation_tokens or folder / "validation.npy"
    if (args.events < 1 or args.start < 0 or args.start_tick < 0
            or (local and (args.events > 256 or args.start_tick != 0))):
        raise ValueError("invalid range; fresh local-port audit requires tick zero and 1..256 events")
    if local and (args.config or args.checkpoint or args.device != "cpu"):
        raise ValueError("fresh local-port audit takes no old config/checkpoint and uses CPU")
    if not local and (not args.config or not args.checkpoint or args.local_learning):
        raise ValueError("checkpoint mode needs --config/--checkpoint and is read-only")
    life = load_core("cgb_low_resource_life")
    stream = load_core("cgb_observation_stream")
    train, validation, manifest, source_hashes = stream.load_owt_sources(
        args.train_tokens, args.validation_tokens, alphabet)
    vocab_size = 256 if alphabet == "byte" else manifest["vocab_size"]
    if args.start + args.events + int(local) >= len(validation):
        raise ValueError("validation slice needs a boot token followed by all requested targets")
    boot = int(validation[args.start])
    targets = (None if alphabet == "byte" else
               validation[args.start + 1:args.start + args.events + 1])
    ledger_module = load_core("cgb_information_ledger")
    load_core("cgb_language_kernel")
    bridge_module = load_core("cgb_prequential_bridge")
    checkpoint_hash, metadata = None, {}
    if local:
        rule_id = f"owt-{alphabet}-{transition}-local-port-interface-v2:" + sha256(Path(__file__))

        def fresh():
            return bridge_module.PackedTreePort(vocab_size, fixture_rule,
                fixture_advance if transition == "legacy" else None,
                context=(boot,), seed=args.seed, learning=args.local_learning, rule_id=rule_id)

        runner = fresh()
        metadata = {"reference_protocol_only": True, "capability_claim": False,
                    "model_architecture_selected": False, "rule_id": rule_id,
                    "transition": transition, "joint_law_identified": False,
                    "persistent_context": "sampled_hidden_mask" if transition == "joint" else "last_observed_symbol",
                    "initial_context": "registered boot-derived parameter; not a world posterior",
                    "seed": args.seed, "learning_enabled": args.local_learning, "warm_boot_symbols": 1}
    else:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if (config["data"]["vocab_size"] != vocab_size
                or config["data"].get("revision", manifest["revision"]) != manifest["revision"]):
            raise ValueError("checkpoint configuration disagrees with source manifest")
        model_module = __import__("fine_grain.information_boltzmann.model", fromlist=["InformationBoltzmann"])
        streaming_module = __import__("fine_grain.information_boltzmann.streaming", fromlist=["StreamRunner"])
        model = model_module.InformationBoltzmann.from_config(config).to(args.device).eval()
        runner = streaming_module.StreamRunner(model, bos_token=boot, seed=int(config.get("seed", 11)),
            optimizer=None, update_every=int(config["train"].get("update_every_real_tokens", 16)),
            gradient_clip=float(config["train"].get("gradient_clip", 1.0)),
            boot_tokens=torch.tensor([boot], device=args.device, dtype=torch.long))
        checkpoint_hash = sha256(args.checkpoint)
        metadata = runner.load(args.checkpoint, read_only=True,
                              allow_rng_device_mismatch=args.allow_rng_device_mismatch)
        if runner.current_token != boot:
            raise ValueError("checkpoint current token does not match the validation boot token")
    ledger = ledger_module.PredictiveInformationLedger(start_tick=args.start_tick, reference_name="train_unigram")
    reference_nll = tuple(unigram_nlls(train, range(vocab_size), vocab_size, args.alpha))
    bridge = bridge_module.CGBPrequentialBridge(
        runner, ledger, lambda token: reference_nll[token], strict_scoring=True)
    events = (validation.observations(start=args.start + 1, limit=args.events, start_tick=args.start_tick)
              if alphabet == "byte" else stream.token_observations(
                  targets, source="owt_validation", start_tick=args.start_tick))
    total_nll, event = consume_prequential(
        bridge, events, life.Observation, args.events, args.write_units, args.dissipation_units)
    if checkpoint_hash and sha256(args.checkpoint) != checkpoint_hash:
        raise RuntimeError("checkpoint changed during read-only measurement")
    if local:
        # Clone-only continuation checks do not advance the scored source or ledger.
        left, right = fresh(), fresh()
        state = runner.state_dict()
        left.load_state_dict(state)
        right.load_state_dict(json.loads(json.dumps(state)))
        sampled = [left.sample(), right.sample()]
        next_token = int(validation[args.start + args.events + 1])
        if alphabet == "byte":
            cursor = json.loads(json.dumps(event.metadata["next_cursor"]))
            restored = next(validation.observations(cursor=cursor, limit=1, start_tick=runner.tick))
            if restored.payload != next_token or restored.tick != runner.tick:
                raise RuntimeError("native source cursor failed exact continuation")
            metadata.update(source_cursor=cursor, source_cursor_restore_equal=True)
        nlls = [left.observe(next_token), right.observe(next_token)]
        if (sampled[0] != sampled[1] or not 0 <= sampled[0] < vocab_size
                or nlls[0] != nlls[1] or left.state_dict() != right.state_dict()
                or state != runner.state_dict()):
            raise RuntimeError("local-port restore continuation disagrees")
        metadata.update(restore_state_dict_continuation_equal=True, json_restore_equal=True, generated_token_ids=sampled,
                        head_nibbles=runner.depths, packed_bytes=runner.packed.numel(),
                        next_target_offset=args.start + args.events + 1, external_tick=runner.tick)
    report = {"status": "local_port_interface_audit" if local else "read_only_heldout_measurement",
        "training_performed": bool(local and args.local_learning), "learner": args.learner,
        "metadata": metadata, "checkpoint_sha256": checkpoint_hash,
        "config": str(args.config.resolve()) if args.config else None,
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "model_parameters": runner.depths if local else sum(p.numel() for p in model.parameters()),
        "source_revision": manifest["revision"], "source_sha256": source_hashes,
        "implementation_sha256": {name: sha256(ROOT / "information_boltzmann/core" / (name + ".py"))
                                  for name in ("cgb_language_kernel", "cgb_prequential_bridge", "cgb_observation_stream")},
        "validation_tokens": str(args.validation_tokens.resolve()), "alphabet": alphabet,
        "train_reference_tokens": str(args.train_tokens.resolve()),
        "unit": "UTF8_bytes" if alphabet == "byte" else "GPT2_tokens",
        "boundary_policy": "EOS/BOS retained as source metadata; byte ticks, no state reset" if alphabet == "byte" else manifest.get("boundary_policy"),
        "start": args.start, "start_tick": args.start_tick, "events": ledger.rows,
        "boot_token": boot, "learner_updates": runner.updates,
        "mean_prequential_nll_nats": total_nll / ledger.rows, "ledger": ledger.summary(),
        "scope": "source/score/restore wiring only; no language capability or world-action acceptance"}
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
