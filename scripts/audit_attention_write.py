"""Controlled head-use and prior-write diagnosis on an exact saved graph."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.capability_tasks import fixed_capability_bank, collate_capability
from fine_grain.omni_model import DualStreamOmni
from scripts.train_northstar_capabilities import evaluate_samples
from scripts.train_pythia_capabilities import make_static_bank, sample_digit_group


def spectrum(x):
    x = x.detach().float().flatten(1)
    x = x - x.mean(0)
    energy = torch.linalg.svdvals(x).square()
    p = energy / energy.sum().clamp_min(1e-12)
    return {"rank": float(1 / p.square().sum().clamp_min(1e-12)),
            "difference_rms": float(x.square().mean().sqrt())}


def forward_batch(model, samples, device):
    b = collate_capability(samples)
    return model(b["image"].to(device), b["prompt"],
                 t=torch.zeros(len(samples), device=device),
                 **{k: b[k].to(device) for k in (
                     "image_precision", "text_precision", "target_time",
                     "history_images", "history_precision", "action",
                     "action_precision", "task_id")})


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/northstar_static32_unified_task_best.pt")
    parser.add_argument("--out", default="results/published/attention_write_audit.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--diagnostics-only", action="store_true")
    args = parser.parse_args()
    raw = torch.load(args.checkpoint, map_location="cpu")
    model = DualStreamOmni(**raw["config"]).to(args.device).eval()
    model.load_state_dict(raw["state_dict"])
    bank = [s for s in fixed_capability_bank(model.res) if s["case"] == "text_to_both"]
    group = sample_digit_group(make_static_bank(model.res, "text_to_both"), np.random.default_rng(20260905))
    records, handles = {}, []
    for i, layer in enumerate(model.mot_stack.layers):
        def read_heads(module, inputs, kwargs, output, index=i):
            s, h = inputs[:2]
            val = torch.cat([module._shape(module.Wv_v(module.rms_v(s))),
                             module._shape(module.Wv_t(module.rms_t(h)))], dim=2)
            av = module.last_av
            v = av @ val
            # Each head's contribution after its actual block of W_o.
            projected = torch.einsum("bhqd,ohd->bhqo", v,
                                    module.Wo_v.weight.reshape(module.d, module.h, module.dh))
            rms = projected.square().mean((0, 2, 3)).sqrt()
            shares = rms / rms.sum().clamp_min(1e-12)
            task_index = val.shape[2] - 1
            values = val.norm(dim=-1).mean((0, 1))
            records[f"layer_{index}"] = {
                "projected_head_rms": rms.tolist(),
                "effective_heads": float(1 / shares.square().sum()),
                "max_mean_key_mass_per_head": av.mean((0, 2)).max(-1).values.tolist(),
                "task_mass_per_head": av[..., task_index].mean((0, 2)).tolist(),
                "task_value_norm": float(values[task_index]),
                "mean_value_norm": float(values.mean()),
                "text_reads_task_mass": float(module.last_at[:, :, :h.shape[1]-2, -1].mean()),
                "non_sink_mass_per_head": av.sum(-1).mean((0, 2)).tolist(),
            }
        handles.append(layer.mot.register_forward_hook(read_heads, with_kwargs=True))
        def read_local(module, inputs, output, index=i):
            records[f"local_{index}"] = {"before": spectrum(inputs[0]), "after": spectrum(output)}
        handles.append(layer.local.register_forward_hook(read_local))
    model.mot_stack.set_record_field_trace(True)
    out = forward_batch(model, group, args.device)
    report = {"checkpoint": args.checkpoint, "config": raw["config"],
              "heads_and_local": records.copy(),
              "field": [spectrum(x) for x in model.mot_stack._last_X_steps],
              "rgb": spectrum(out["rgb"])}
    for h in handles:
        h.remove()
    model.mot_stack.set_record_field_trace(False)
    report["interventions"] = {}
    originals = [l.prior_write for l in model.mot_stack.layers]
    interventions = [] if args.diagnostics_only else [
        ("baseline", originals, False),
        ("late_prior_half", originals[:2] + [0.5 * p for p in originals[2:]], False),
        ("late_prior_off", originals[:2] + [0.0, 0.0], False),
        ("late_local_off", originals, True),
    ]
    for name, gains, no_local in interventions:
        local_hooks = []
        for i, (layer, gain) in enumerate(zip(model.mot_stack.layers, gains)):
            layer.prior_write = gain
            if no_local and i >= 2:
                local_hooks.append(layer.local.register_forward_hook(lambda m, inp, out: inp[0]))
        report["interventions"][name] = evaluate_samples(model, bank, torch.device(args.device))["text_to_both"]
        for hook in local_hooks:
            hook.remove()
    for layer, gain in zip(model.mot_stack.layers, originals):
        layer.prior_write = gain
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "config"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
