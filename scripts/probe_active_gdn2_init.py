"""Audit that active GDN-2 starts as an exact champion-preserving residual."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.capability_tasks import collate_capability, fixed_capability_bank
from fine_grain.omni_model import DualStreamOmni
from scripts.train_northstar_capabilities import load_compatible


def config(active: bool) -> dict:
    return {
        "d_model": 64, "n_slices": 16, "n_layers": 4, "n_heads": 4, "res": 16,
        "surprise_mode": "v1_bayes", "s_update": "raw", "prior_loss_coef": 0.1,
        "sigreg_coef": 0.0, "use_stiefel": False, "deslice_topk": 0,
        "use_null_slice": False, "use_residual_read": False, "gate_on": "u",
        "deslice_write": "increment", "gate_h_local": False, "vfe_coef": 0.1,
        "prior_write": 1.0, "prior_write_by_t": False,
        "pixel_loss_mode": "balanced_bce", "spatial_prompt_vocab": True,
        "capability_vocab": True, "use_modal_precision": True,
        "use_target_time": True, "use_target_time_adaln": False,
        "use_horizon_tokens": False, "gate_action_by_horizon": True,
        "history_size": 2, "action_dim": 2, "use_action_adaln": False,
        "use_action_tokens": True, "use_action_rel_bias": True,
        "use_action_transport": False, "use_action_slice_transition": False,
        "use_goal_adaln": False, "use_active_gdn2": active,
        "seg_classes": 2, "seg_loss_coef": 1.0, "transition_loss_coef": 0.5,
        "transition_posterior_loss_coef": 1.0, "transition_detach_q": True,
        "causal_memory_loss_coef": 0.25, "s0_acc_coef": 0.0,
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--init",
        default=str(ROOT / "checkpoints" / "omni_d64_northstar_omni_active_f2_grid_best.pt"),
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "results" / "published" / "active_gdn2_init_identity.json"),
    )
    args = parser.parse_args()
    torch.manual_seed(42)
    base = DualStreamOmni(**config(False)).eval()
    torch.manual_seed(42)
    active = DualStreamOmni(**config(True)).eval()
    init = Path(args.init)
    base_report = load_compatible(base, init)
    active_report = load_compatible(active, init)

    samples = fixed_capability_bank(16)
    # One item per boundary is enough for exact graph identity.
    chosen = [next(x for x in samples if x["case"] == case) for case in (
        "text_to_both", "image_to_current", "image_text_edit", "image_to_future",
    )]
    batch = collate_capability(chosen)
    kwargs = {
        "t": torch.zeros(4),
        "image_precision": batch["image_precision"],
        "text_precision": batch["text_precision"],
        "target_time": batch["target_time"],
        "history_images": batch["history_images"],
        "history_precision": batch["history_precision"],
        "action": batch["action"],
        "action_precision": batch["action_precision"],
    }
    a = base(batch["image"], batch["prompt"], **kwargs)
    b = active(batch["image"], batch["prompt"], **kwargs)
    deltas = {
        key: float((a[key] - b[key]).abs().max())
        for key in ("X", "logits", "rgb", "seg_logits")
    }
    record = {
        "schema": "active-gdn2-init-identity-v1",
        "checkpoint": str(init.resolve()),
        "base_load": base_report,
        "active_load": active_report,
        "max_abs_delta": deltas,
        "exact_identity": all(value == 0.0 for value in deltas.values()),
        "causal_projection_norm": float(
            active.mot_stack.active_gdn2.prior_channel_gain.norm()
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
