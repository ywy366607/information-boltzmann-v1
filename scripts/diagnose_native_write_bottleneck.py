"""Measure where native text-to-image detail is lost, without training the model.

This is a diagnostic, not a capability run.  It measures three distinct
objects on the same fixed checkpoint:

* target / reconstruction / T2I image bandwidth;
* the spatial selectivity and bandwidth of each live Slice/Deslice write;
* two target-aware *oracles*.  A free point-field oracle checks the RGB
  readout, while a fixed-address Slice oracle checks what one or four actual
  Deslice writes could express if their slice increments were chosen perfectly.

The latter never feeds a target into the deployed forward pass and must never
be reported as a generated image.  Its only purpose is to distinguish a
write-address capacity limit from a learned language-to-write failure.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.omni_model import DualStreamOmni
from fine_grain.real_capacity import (
    add_zero_horizon_controls,
    build_real_bank,
    collate_real_capacity,
    forward_real_capacity,
)
from fine_grain.unified_capacity import extend_unified_bank
from scripts.train_real256_capacity import edge_metrics, load_graph_weights, sha256


def _rms(x: torch.Tensor) -> float:
    return float(x.detach().float().square().mean().sqrt().cpu())


def _field_image(x: torch.Tensor) -> torch.Tensor:
    """[B,N,C] -> [B,C,R,R], validating the native square-grid contract."""
    n = x.shape[1]
    side = math.isqrt(n)
    if side * side != n:
        raise ValueError(f"expected a square field, got {n} points")
    return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], side, side)


def _bandwidth(x: torch.Tensor) -> Dict[str, float]:
    """Scale-invariant high-frequency and neighbour-difference statistics."""
    if x.ndim == 3:
        x = _field_image(x)
    if x.ndim != 4:
        raise ValueError(f"expected BCHW or BNC, got {tuple(x.shape)}")
    y = x.detach().float()
    y = y - y.mean(dim=(-2, -1), keepdim=True)
    spatial_energy = y.square().mean()
    dx = y[..., :, 1:] - y[..., :, :-1]
    dy = y[..., 1:, :] - y[..., :-1, :]
    grad_rms = torch.cat([dx.flatten(), dy.flatten()]).square().mean().sqrt()
    # Frequencies are in cycles/pixel.  r >= .25 is the upper half of the
    # one-dimensional Nyquist band; DC is excluded before normalisation.
    h, w = y.shape[-2:]
    fy = torch.fft.fftfreq(h, device=y.device).abs()[:, None]
    fx = torch.fft.fftfreq(w, device=y.device).abs()[None, :]
    radius = torch.sqrt(fx.square() + fy.square())
    power = torch.fft.fft2(y, dim=(-2, -1)).abs().square()
    non_dc = radius > 0
    high = radius >= 0.25
    total = power[..., non_dc].sum().clamp_min(1e-12)
    return {
        "signal_rms": float(spatial_energy.sqrt().cpu()),
        "neighbor_gradient_rms": float(grad_rms.cpu()),
        "fft_highband_fraction_r_ge_0_25": float((power[..., high].sum() / total).cpu()),
    }


def _effective_rank(values: torch.Tensor) -> float:
    values = values.detach().float()
    values = values[values > 1e-10]
    if not values.numel():
        return 0.0
    p = values / values.sum().clamp_min(1e-12)
    return float(torch.exp(-(p * p.log()).sum()).cpu())


def _address_metrics(w: torch.Tensor) -> Dict[str, float]:
    """Measure address diversity and locality of one [1,N,M] write map."""
    if w.shape[0] != 1:
        raise ValueError("diagnostic uses one example at a time")
    _, n, m = w.shape
    side = math.isqrt(n)
    if side * side != n:
        raise ValueError("address field is not square")
    a = w[0].detach().float().clamp_min(0)
    row = a.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    p = a / row
    entropy = -(p * p.clamp_min(1e-12).log()).sum(dim=-1) / math.log(max(m, 2))
    mass = a.sum(dim=0)
    coords_1d = torch.linspace(-1.0, 1.0, side, device=a.device)
    yy, xx = torch.meshgrid(coords_1d, coords_1d, indexing="ij")
    xy = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)
    centers = torch.einsum("nm,nc->mc", a, xy) / mass[:, None].clamp_min(1e-12)
    radius_sq = ((xy[:, None, :] - centers[None, :, :]).square() * a[:, :, None]).sum((0, 2))
    radius_sq = radius_sq / mass.clamp_min(1e-12)
    grid = a.transpose(0, 1).reshape(m, side, side)
    tv = (grid[:, 1:, :] - grid[:, :-1, :]).abs().mean()
    tv = tv + (grid[:, :, 1:] - grid[:, :, :-1]).abs().mean()
    singular = torch.linalg.svdvals(a)
    # Uniform weight on [-1,1]^2 has RMS radius sqrt(2/3), so this ratio is
    # 1 for a spatially unselective slice and approaches 0 for a compact one.
    uniform_radius = math.sqrt(2.0 / 3.0)
    return {
        "mean_row_entropy_normalized": float(entropy.mean().cpu()),
        "slice_mass_effective_count": _effective_rank(mass),
        "address_matrix_effective_rank": _effective_rank(singular),
        "mean_slice_radius_over_uniform": float(radius_sq.sqrt().mean().cpu() / uniform_radius),
        "address_total_variation": float(tv.cpu()),
        **{f"address_{k}": v for k, v in _bandwidth(grid).items()},
    }


@torch.no_grad()
def _collect(model: DualStreamOmni, sample: Dict, device: str) -> Dict:
    """Run one deployed forward pass and retain only target-free internals."""
    batch = collate_real_capacity(model.lm_tok, [sample], device)
    model.mot_stack.record_field_trace = True
    out = forward_real_capacity(model, batch)
    fields = [x.detach().clone() for x in model.mot_stack._last_X_steps]
    layers = []
    for layer in model.mot_stack.layers:
        layers.append({
            "w_write": layer.last_w_write.detach().clone(),
            "dW": layer.last_dW.detach().clone(),
            "x_after": layer.last_X.detach().clone(),
        })
    model.mot_stack.record_field_trace = False
    pred = out["rgb"].detach().clone()
    target = sample["target_rgb"].unsqueeze(0).to(device)
    return {"pred": pred, "target": target, "fields": fields, "layers": layers}


def _one_report(model: DualStreamOmni, sample: Dict, control: Dict | None, device: str) -> Dict:
    run = _collect(model, sample, device)
    edge_error, edge_corr = edge_metrics(run["pred"].cpu(), run["target"].cpu())
    answer = {
        "id": sample["id"],
        "task": sample["task"],
        "family": sample.get("family", "real"),
        "pred_bandwidth": _bandwidth(run["pred"]),
        "target_bandwidth": _bandwidth(run["target"]),
        "edge_correlation": float(edge_corr),
        "edge_relative_mse": float(edge_error),
        "layers": [],
    }
    for idx, (before, detail) in enumerate(zip(run["fields"], run["layers"])):
        written = before + detail["dW"]
        after = detail["x_after"]
        rgb_before = model.decode_rgb(before)
        rgb_written_delta = model.decode_rgb(written) - rgb_before
        rgb_full_delta = model.decode_rgb(after) - rgb_before
        local_delta = after - written
        answer["layers"].append({
            "layer": idx,
            "address": _address_metrics(detail["w_write"]),
            "deslice_latent_delta": _bandwidth(detail["dW"]),
            "local_latent_delta": _bandwidth(local_delta),
            "deslice_rgb_effect": _bandwidth(rgb_written_delta),
            "full_layer_rgb_effect": _bandwidth(rgb_full_delta),
            "deslice_latent_rms": _rms(detail["dW"]),
            "local_latent_rms": _rms(local_delta),
        })
    if control is not None:
        alternate = _collect(model, control, device)
        answer["prompt_counterfactual"] = {
            "control_id": control["id"],
            "final_rgb_rms": _rms(run["pred"] - alternate["pred"]),
            "per_layer": [
                {
                    "layer": i,
                    "write_delta_rms": _rms(a["dW"] - b["dW"]),
                    "write_address_rms": _rms(a["w_write"] - b["w_write"]),
                    "field_rms": _rms(run["fields"][i + 1] - alternate["fields"][i + 1]),
                }
                for i, (a, b) in enumerate(zip(run["layers"], alternate["layers"]))
            ],
        }
    return answer | {"_oracle_state": run}


def _oracle_fit(
    model: DualStreamOmni,
    state: Dict,
    steps: int,
    lr: float,
    kind: str,
) -> Dict[str, float]:
    """Target-aware capacity probe; never a deployed prediction.

    ``field`` has a free persistent value at every point. ``one_write`` and
    ``four_writes`` instead expose only M*d controls per retained layer and
    use the actual prompt-conditioned addresses from the preceding forward.
    """
    target = state["target"].detach()
    stem = state["fields"][0].detach()
    layers = model.mot_stack.layers
    if kind == "field":
        controls: Sequence[torch.nn.Parameter] = [torch.nn.Parameter(stem.clone())]
        retained = 0
    elif kind == "one_write":
        controls = [torch.nn.Parameter(torch.zeros_like(layers[0].last_S))]
        retained = 1
    elif kind == "four_writes":
        controls = [torch.nn.Parameter(torch.zeros_like(layer.last_S)) for layer in layers]
        retained = len(layers)
    else:
        raise ValueError(kind)
    opt = torch.optim.Adam(controls, lr=lr)
    first = last = None
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        if kind == "field":
            x = controls[0]
        else:
            x = stem
            for index in range(retained):
                write = layers[index].deslice.write_delta(
                    controls[index], state["layers"][index]["w_write"],
                )
                # This is the deployed Deslice + LocalVisual sequence, with
                # only the target-aware Slice increment substituted.
                x = layers[index].local(x + write)
        pred = model.decode_rgb(x)
        loss = F.mse_loss(pred, target)
        if first is None:
            first = float(loss.detach())
        loss.backward()
        opt.step()
        last = float(loss.detach())
    with torch.no_grad():
        if kind == "field":
            x = controls[0]
        else:
            x = stem
            for index in range(retained):
                x = layers[index].local(x + layers[index].deslice.write_delta(
                    controls[index], state["layers"][index]["w_write"]
                ))
        pred = model.decode_rgb(x)
        mse = float(F.mse_loss(pred, target))
        edge_error, edge_corr = edge_metrics(pred.cpu(), target.cpu())
    return {
        "oracle": kind,
        "controls": int(sum(control.numel() for control in controls)),
        "steps": int(steps),
        "loss_first": float(first if first is not None else mse),
        "loss_last_pre_step": float(last if last is not None else mse),
        "mse": mse,
        "psnr": -10.0 * math.log10(max(mse, 1e-12)),
        "edge_correlation": float(edge_corr),
        "edge_relative_mse": float(edge_error),
        "pred_bandwidth": _bandwidth(pred),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", default="checkpoints/mixed_native_full_64_256_resume.pt")
    parser.add_argument("--sharegpt-manifest", default="D:/ml_cache/sharegpt4o/pilot_manifest.json")
    parser.add_argument("--davis-manifest", default="D:/ml_cache/davis_micro/manifest.json")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--ids", nargs="+", default=["freedom-t2i-17808", "freedom-t2i-32449", "t2i1px-7"])
    parser.add_argument("--oracle-ids", nargs="+", default=["freedom-t2i-17808", "t2i1px-7"])
    parser.add_argument("--oracle-kinds", nargs="+", choices=("field", "one_write", "four_writes"),
                        default=("field", "one_write", "four_writes"),
                        help="Run a subset when extending a converged oracle probe.")
    parser.add_argument("--oracle-steps", type=int, default=240)
    parser.add_argument("--oracle-lr", type=float, default=0.08)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tag", default="native_write_bottleneck_6400")
    args = parser.parse_args()
    if args.oracle_steps < 1 or args.oracle_lr <= 0:
        parser.error("oracle steps and lr must be positive")
    output = ROOT / "results" / "published" / f"{args.tag}.json"
    if output.exists():
        parser.error(f"refusing to overwrite existing report: {output}")

    payload = torch.load(ROOT / args.init, map_location="cpu")
    model = DualStreamOmni(**payload["config"]).to(args.device)
    load_graph_weights(model, payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    bank = extend_unified_bank(add_zero_horizon_controls(build_real_bank(
        args.sharegpt_manifest, args.davis_manifest, args.resolution
    )), args.resolution)
    by_id = {sample["id"]: sample for sample in bank}
    missing = sorted(set(args.ids) - set(by_id))
    if missing:
        raise ValueError(f"unknown IDs: {missing}")
    all_t2i = [sample for sample in bank if sample["task"] == "t2i"]
    rows: List[Dict] = []
    saved_state: Dict[str, Dict] = {}
    for ident in args.ids:
        sample = by_id[ident]
        control = next((s for s in all_t2i if s["id"] != ident and s.get("family") == sample.get("family")), None)
        row = _one_report(model, sample, control, args.device)
        saved_state[ident] = row.pop("_oracle_state")
        rows.append(row)
        print(json.dumps({"measured": ident, "edge": row["edge_correlation"]}), flush=True)
    oracles: Dict[str, List[Dict[str, float]]] = {}
    for ident in args.oracle_ids:
        if ident not in saved_state:
            state_row = _one_report(model, by_id[ident], None, args.device)
            saved_state[ident] = state_row.pop("_oracle_state")
        oracles[ident] = []
        for kind in args.oracle_kinds:
            report = _oracle_fit(model, saved_state[ident], args.oracle_steps, args.oracle_lr, kind)
            oracles[ident].append(report)
            print(json.dumps({"oracle": ident, **{k: report[k] for k in ("oracle", "psnr", "edge_correlation")}}), flush=True)
    report = {
        "status": "completed_diagnostic_not_a_capability_claim",
        "checkpoint": str(Path(args.init)),
        "checkpoint_sha256": sha256(ROOT / args.init),
        "resolution": args.resolution,
        "scope": (
            "Deployed forward metrics use no target input. Oracle rows intentionally optimise "
            "against the target and quantify only field/readout or fixed-address write capacity."
        ),
        "frequency_definition": "FFT radial frequency >= 0.25 cycles/pixel; no DC energy",
        "address_definition": "actual last_w_write per independent native layer; no address retraining",
        "examples": rows,
        "oracles": oracles,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
