"""Sequential S1 comparison; never run beside another model process."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_real256_capacity import sha256

EXPECTED = "5718aee71281bccb93203c84e0c7ac9fbfeff275772f9537b625df42418f974a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sharegpt-manifest", required=True)
    parser.add_argument("--davis-manifest", required=True)
    parser.add_argument("--completed-run", required=True,
                        help="Preceding training report, which must show terminal successful evaluation")
    args = parser.parse_args()
    prior = json.loads(Path(args.completed_run).read_text(encoding="utf-8"))
    if prior["status"] != "completed_budget_candidate_only" or "reloaded" not in prior:
        raise RuntimeError("preceding run is not complete; do not overlap heavy processes")
    checkpoint = ROOT / "checkpoints/mixed_native_full_64_256.pt"
    if sha256(checkpoint) != EXPECTED:
        raise RuntimeError("preregistered checkpoint hash mismatch")
    results = ROOT / "results/published"
    report_path = results / "spatial_measure_ab.json"
    tags = {arm: f"spatial_measure_{arm}" for arm in ("uniform", "stratified")}
    if report_path.exists() or any((results / f"{tag}.json").exists() for tag in tags.values()):
        raise RuntimeError("existing S1 experiment protected; inspect before resuming")
    report = {"status": "running", "checkpoint": str(checkpoint), "sha256": EXPECTED,
              "scope": "S1 screening only; all 42 cases at both sizes; no gate relaxation",
              "updates_per_arm": 800, "original_schedule_steps": 6400,
              "arms": {}, "comparison": {}}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for arm, tag in tags.items():
        command = [sys.executable, str(ROOT / "scripts/train_mixed_native.py"),
                   "--init", str(checkpoint), "--resume-training", "--full-bank",
                   "--steps", "6400", "--stop-after-updates", "800", "--eval-every", "400",
                   "--lr", "0.0001", "--rgb-measure", arm, "--tag", tag,
                   "--sharegpt-manifest", args.sharegpt_manifest, "--davis-manifest", args.davis_manifest]
        print(f"Starting sequential S1 arm: {arm}", flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode != 0:
            report.update(status="failed", failed_arm=arm, exit_code=completed.returncode)
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            raise SystemExit(completed.returncode)
        data = json.loads((results / f"{tag}.json").read_text(encoding="utf-8"))
        if data["current_step"] != 1600 or "reloaded" not in data:
            raise RuntimeError("trial did not independently reload at the registered end step")
        report["arms"][arm] = {"report": f"{tag}.json", "checkpoint_sha256": data["checkpoint_sha256"],
                               "seen": data["seen"], "reloaded": data["reloaded"]}
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["arms"]["uniform"]["seen"] != report["arms"]["stratified"]["seen"]:
        raise RuntimeError("arm exposure mismatch")
    keys = ("psnr", "edge_correlation", "rgb_stroke_iou", "rgb_background_flood", "rgb_near_background_flood",
            "seg_iou", "token_nll", "image_shuffle_nll_gap", "text_shuffle_nll_gap",
            "improvement_over_copy", "zero_horizon_mse_gap")
    for res in ("64", "256"):
        base = {r["id"]: r for r in report["arms"]["uniform"]["reloaded"][res]["metrics"]["samples"]}
        changed = report["arms"]["stratified"]["reloaded"][res]["metrics"]["samples"]
        report["comparison"][res] = [dict(
            id=r["id"], task=r["task"], family=r["family"],
            delta_stratified_minus_uniform={k: r[k]-base[r["id"]][k] for k in keys if k in r},
            uniform_answer=base[r["id"]].get("decoded"), stratified_answer=r.get("decoded"),
        ) for r in changed]
    if sha256(checkpoint) != EXPECTED:
        raise RuntimeError("protected start changed")
    report["status"] = "completed_comparison_requires_review"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Both arms reloaded; inspect per-example comparison, not weighted training loss.", flush=True)


if __name__ == "__main__":
    main()
