"""Separate conserved S-return transfer's future effect from non-S learning.

This is one causal 2-by-2 continuation, not a new optimizer or a parameter
search.  Four bodies begin from the same saved continuous ``s_on`` state and
receive one identical unseen OpenWebText continuation:

* all ordinary parameter learners enabled, S slow write enabled/disabled;
* one declared set of non-S parameter learners held, S slow write
  enabled/disabled.

H/C/S *state dynamics* remain active in every arm.  The difference between the
two within-pair S effects therefore audits whether concurrent non-S parameter
adaptation materially changes the S effect.  It cannot identify an individual
member of that aggregate, prove an optimal update, or establish capability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_cgb_column_language_admission import CudaGraphValueCredit, load_split


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


HELD_GROUPS = {
    "non_s_all": ("H-contact", "C-value", "route", "readout"),
    # This is the shortest existing causal path to S teaching: the immediate
    # language coefficient in `_s_contact_return_from_tensors` is innovation @
    # readout.weight.  Holding its *write* retains the observed boundary error.
    "readout": ("readout",),
}


def profile_state(state: dict, *, s_contact_step: float, held_group: str) -> dict:
    """Alter only declared update rates; all physical H/C/S transitions persist."""
    result = dict(state)
    profile = dict(state["profile"])
    profile["s_contact_step"] = float(s_contact_step)
    if held_group == "none":
        pass
    elif held_group == "non_s_all":
        profile.update(head_step=0.0, value_step=0.0, route_step=0.0, contact_step=0.0)
    elif held_group == "readout":
        profile["head_step"] = 0.0
    else:
        raise ValueError(f"unknown held group: {held_group}")
    result["profile"] = profile
    return result


@torch.no_grad()
def run_arm(state: dict, values: np.ndarray, *, device: torch.device,
            report_every: int) -> dict:
    engine = CudaGraphValueCredit.from_checkpoint(state, device=device)
    windows = []
    for start in range(0, len(values), report_every):
        metrics = engine.phase(values[start:start + report_every], report_every=report_every + 1)
        windows.append({"raw": [start, min(start + report_every, len(values))], **metrics})
    result = {"total_scored_bytes": sum(window["scored_bytes"] for window in windows),
              "total_nll_nats": sum(window["mean_nll_nats"] * window["scored_bytes"] for window in windows),
              "windows": windows}
    result["mean_nll_nats"] = result["total_nll_nats"] / max(result["total_scored_bytes"], 1)
    result["final_lifetime"] = engine.checkpoint_state()["lifetime"]
    del engine
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "results/cgb_column_language_living_20261101/latest.pt")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/ib_owt_10000")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/published/cgb_column_s_conserved_transfer_interaction_owt_20261101.json")
    parser.add_argument("--raw-start", type=int, default=None)
    parser.add_argument("--raw-count", type=int, default=64_000)
    parser.add_argument("--report-every", type=int, default=8_000)
    parser.add_argument("--held-group", choices=tuple(HELD_GROUPS), default="non_s_all",
                        help="Existing non-S parameter write path held in the second pair")
    parser.add_argument("--device", choices=("cpu", "cuda"),
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # CLI paths may be relative while ROOT is absolute; normalize once so
    # provenance serialization and file access share one coordinate system.
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    args.data_root = args.data_root.resolve()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit: {args.output}")
    if args.raw_count < 1 or args.report_every < 1:
        raise ValueError("raw count and report interval must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "s_on" not in checkpoint.get("engines", {}):
        raise ValueError("checkpoint lacks the saved s_on individual")
    raw_start = checkpoint["experiment"]["next_raw_cursor"] if args.raw_start is None else args.raw_start
    raw, provenance = load_split(args.data_root, "train", raw_start + args.raw_count)
    values = raw[raw_start:raw_start + args.raw_count]
    if len(values) != args.raw_count:
        raise ValueError("registered source is too short for the requested continuation")
    expected = checkpoint["experiment"].get("provenance", {}).get("train", {})
    if expected and provenance["revision"] != expected.get("revision"):
        raise ValueError("checkpoint and audit source revisions differ")
    base = checkpoint["engines"]["s_on"]
    device = torch.device(args.device)
    arms = {
        "all_s_on": profile_state(base, s_contact_step=base["profile"]["s_contact_step"],
                                  held_group="none"),
        "all_s_off": profile_state(base, s_contact_step=0.0, held_group="none"),
        "held_s_on": profile_state(base, s_contact_step=base["profile"]["s_contact_step"],
                                   held_group=args.held_group),
        "held_s_off": profile_state(base, s_contact_step=0.0, held_group=args.held_group),
    }
    results = {name: run_arm(state, values, device=device, report_every=args.report_every)
               for name, state in arms.items()}
    all_effect = results["all_s_on"]["mean_nll_nats"] - results["all_s_off"]["mean_nll_nats"]
    held_effect = results["held_s_on"]["mean_nll_nats"] - results["held_s_off"]["mean_nll_nats"]
    result = {
        "kind": "s_conserved_transfer_non_s_parameter_interaction_audit",
        "scope": (
            "A common-state real-continuation difference-in-differences audit of the current "
            "one-receipt conserved S transfer. Positive language return admits from implicit "
            "local free capacity and negative return releases the indicated S factor. H/C/S "
            "state dynamics remain active in all arms; the held condition disables only "
            "declared non-S parameter learners. This checkpoint predates the transfer replacement, "
            "so the result is a post-replacement intervention audit, not a new-from-birth run."
        ),
        "checkpoint": {"path": str(args.checkpoint.relative_to(ROOT)), "sha256": sha256(args.checkpoint),
                       "arm": "s_on", "raw_start": raw_start},
        "source": provenance,
        "protocol": {"raw_range": [raw_start, raw_start + args.raw_count],
                     "report_every_raw": args.report_every, "s_step": base["profile"]["s_contact_step"],
                     "held_group": args.held_group,
                     "held_parameter_learners": list(HELD_GROUPS[args.held_group])},
        "results": results,
        "effects_nats_per_byte": {
            "all_other_parameter_learning": all_effect,
            "other_parameter_learning_held": held_effect,
            "interaction_contrast_all_minus_held": all_effect - held_effect,
        },
        "model_source": {
            path.name: sha256(path) for path in (
                ROOT / "information_boltzmann/core/cgb_column_body.py",
                ROOT / "scripts/run_cgb_column_language_admission.py",
                Path(__file__),
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["effects_nats_per_byte"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
