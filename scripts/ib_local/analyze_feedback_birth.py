"""Summarize results/cgb_feedback_birth_20261103 against its PREREG.md (2026-11-03)."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/cgb_feedback_birth_20261103"
ARMS = ("decay_then_delta", "feedback_delta")


def spread(values):
    values = sorted(values)
    return dict(min=values[0], median=statistics.median(values),
                mean=sum(values) / len(values), max=values[-1])


def latch_ratios(checkpoint: Path):
    """Birth-formula proxy per layer and head: beta*eta*g / (1 - alpha(1-beta)), a(x)=0."""
    model = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    rows = []
    for layer in range(4):
        p = lambda name: model[f"layers.{layer}.{name}"].float()
        alpha = torch.exp(-p("A_log").exp() * F.softplus(p("dt_bias")))
        beta = torch.sigmoid(p("beta_proj.bias"))
        eta, gain = torch.sigmoid(p("feedback_logit")), F.softplus(p("feedback_gain_raw"))
        rows.append(dict(alpha=alpha.tolist(), eta=eta.tolist(), gain=gain.tolist(),
                         ratio=(beta * eta * gain / (1 - alpha * (1 - beta))).tolist()))
    return rows


def seed_metrics(folder: Path):
    arms = json.loads((folder / "training_comparison.json").read_text())["arms"]
    out = {}
    for arm in ARMS:
        summary = arms[arm]
        context = json.loads((folder / f"{arm}_context.json").read_text())
        by_prefix = {r["prefix_bytes"]: r["nll_nats_per_byte"] for r in context["results"]}
        lambada = json.loads((folder / f"{arm}_lambada.json").read_text())["results"]["full"]
        phases = summary["phase_scores"]
        out[arm] = dict(
            final_validation=summary["final_validation_nll"],
            revisit_saving=summary["aligned_revisit"]["nll_saving"],
            B_fresh=phases["B_fresh"]["mean_nll"], C_fresh=phases["C_fresh"]["mean_nll"],
            A_revisit=phases["A_revisit"]["mean_nll"],
            context_gain_2048_to_32768=by_prefix[2048] - by_prefix[32768],
            context_32768=by_prefix[32768],
            lambada_nll=lambada["mean_target_nll"], lambada_acc=lambada["exact_prefix_accuracy"],
            bytes_per_second=summary["training_bytes_per_second"])
    out["latch"] = latch_ratios(folder / "feedback_delta/last.pt")
    return out


def main():
    seeds = sorted(p for p in RUN.glob("seed*") if p.is_dir())
    per_seed = {p.name: seed_metrics(p) for p in seeds}
    keys = list(next(iter(per_seed.values()))["decay_then_delta"].keys())
    table = {}
    for key in keys:
        control = [s["decay_then_delta"][key] for s in per_seed.values()]
        feedback = [s["feedback_delta"][key] for s in per_seed.values()]
        table[key] = dict(control=spread(control), feedback=spread(feedback),
                          paired_difference=spread([f - c for f, c in zip(feedback, control)]),
                          per_seed_difference=[f - c for f, c in zip(feedback, control)])
    latch = {}
    for name, s in per_seed.items():
        top = [layer["ratio"][-1] for layer in s["latch"]]
        latch[name] = dict(top_head_ratio=top, layers_above=sum(r > 1 for r in top),
                           all_ratios=[layer["ratio"] for layer in s["latch"]],
                           eta=[layer["eta"] for layer in s["latch"]])
    verdict = dict(
        P1_latch_kept=sum(v["layers_above"] >= 3 for v in latch.values()) >= 2,
        P2_revisit=sum(d > 0 for d in table["revisit_saving"]["per_seed_difference"]) >= 2,
        P3_no_fresh_cost=max(table["B_fresh"]["paired_difference"]["median"],
                             table["C_fresh"]["paired_difference"]["median"]) <= 0.02,
        P4_long_context=sum(d > 0 for d in table["context_gain_2048_to_32768"]["per_seed_difference"]) >= 2)
    report = dict(seeds=[p.name for p in seeds], metrics=table, latch=latch, predictions=verdict)
    (RUN / "analysis.json").write_text(json.dumps(report, indent=1))
    for key, row in table.items():
        c, f, d = row["control"], row["feedback"], row["paired_difference"]
        print(f"{key:28s} control {c['median']:.4f}  feedback {f['median']:.4f}  "
              f"diff [{d['min']:+.4f} | {d['median']:+.4f} | {d['mean']:+.4f} | {d['max']:+.4f}]")
    for name, v in latch.items():
        print(name, "top-head ratio per layer", [round(r, 2) for r in v["top_head_ratio"]],
              "eta", [[round(e, 3) for e in layer] for layer in v["eta"]])
    print(verdict)


if __name__ == "__main__":
    sys.exit(main())
