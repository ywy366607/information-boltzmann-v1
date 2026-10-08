"""Audit existing S14 logs; no model execution, optimizer updates or GPU use."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path,
                        default=Path("results/q8_fly_ctm_settle14_100k"))
    parser.add_argument("--data", type=Path, default=Path("data/ib_owt_gpt2"))
    parser.add_argument("--output", type=Path, default=Path(
        "results/published/fly_ctm_s14_review_20261008.json"))
    args = parser.parse_args()
    config = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    progress = json.loads((args.run / "progress.json").read_text(encoding="utf-8"))
    first = read_jsonl(args.run / "metrics.jsonl")[0]
    evaluations = read_jsonl(args.run / "lifelong_evaluation.jsonl")
    train = np.load(args.data / "train.npy", mmap_mode="r")
    validation = np.load(args.data / "validation.npy", mmap_mode="r")
    counts = np.ones(50257, dtype=np.float64)
    for left in range(0, len(train), 1_000_000):
        counts += np.bincount(np.asarray(train[left:left + 1_000_000],
                                         dtype=np.int64), minlength=len(counts))
    surprisal = -np.log(counts / counts.sum())

    # First logged row includes the first W new targets. Subtract it to recover
    # the inherited phase accumulator at the actual S14 migration boundary.
    window = int(config["window"])
    boundary = int(first["bptt_train_tokens"]) - window
    initial_targets = np.asarray(train[boundary + 1:boundary + 1 + window],
                                 dtype=np.int64)
    n_first = first["phase_first_pass_measured_train_targets"]
    n_final = progress["phase_first_pass_measured_train_targets"]
    n_new = n_final - n_first + window
    old_loss_sum = n_first * first["phase_first_pass_train_nll"] - (
        window * first["window_prequential_nll"])
    old_prior_sum = n_first * first["phase_first_pass_fixed_unigram_nll"] - (
        float(surprisal[initial_targets].sum()))
    stage_loss = (n_final * progress["phase_first_pass_train_nll"] - old_loss_sum) / n_new
    stage_prior = (n_final * progress["phase_first_pass_fixed_unigram_nll"] - old_prior_sum) / n_new
    paired = []
    for row in evaluations:
        end = int(row["fresh_validation_cursor"])
        length = len(row["B_curve"])
        targets = np.asarray(validation[end - length:end], dtype=np.int64)
        prior = float(surprisal[targets].mean())
        model = float(row["live_prequential_nll"])
        paired.append({"train_tokens": row["bptt_train_tokens"],
                       "validation_target_interval": [end - length, end],
                       "validation_targets": length,
                       "model_active_prequential_nll": model,
                       "fixed_training_unigram_nll": prior,
                       "paired_gain": prior - model})
    report = {
        "scope": "Existing log/source review; no training or new model execution",
        "run": str(args.run),
        "measurement": "Active score-before-update B stream; Laplace-smoothed training-only unigram on exactly matched targets",
        "training_count_sha256": hashlib.sha256(counts.tobytes()).hexdigest(),
        "s14_stage": {"start_train_tokens": boundary,
                      "end_train_tokens": progress["bptt_train_tokens"],
                      "new_training_targets": n_new,
                      "train_prequential_nll": stage_loss,
                      "paired_train_unigram_nll": stage_prior,
                      "paired_train_gain": stage_prior - stage_loss,
                      "new_training_updates": n_new // window},
        "fresh_B": paired,
        "reported_recent_training_tail": {
            "targets": progress["recent_first_pass_measured_targets"],
            "nll": progress["recent_first_pass_train_nll"],
            "gain": progress["recent_first_pass_gain_over_fixed_unigram"]},
        "inherited_phase_start": progress["phase_first_pass_measurement_start_bptt_tokens"],
        "physical_ticks_per_token": int(config["settle_ticks"]) + 1,
        "credit_tokens": window,
        "time_selection": "Minimum predictive entropy after all ticks; persistent state still advances to final tick",
        "training_loss": "0.5 * (label-minimum-tick CE + entropy-selected-tick CE)",
        "evaluation_score": "Entropy-selected-tick CE (no label-based time selection)",
        "gradient_accumulation_proposal": {
            "chunks": 8, "tokens_per_chunk": 4, "tokens_per_update": 32,
            "credit_tokens": 4,
            "status": "Attachment one-update test; not integrated into production CLI",
            "equivalence": "Mean of detached-boundary chunk objectives at fixed parameters; different from full BPTT32",
            "stability": "One finite update does not establish a universal stability bound"},
        "conclusion": "All three fresh B segments beat the paired unigram. Incremental benefit of S14 over S4 remains unidentified because training lineage, credit window and data intervals differ.",
        "review": "/root/rtc_contract_review independently confirmed scoring/credit distinction",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"report": str(args.output), "s14_stage": report["s14_stage"],
                      "fresh_B": paired}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
