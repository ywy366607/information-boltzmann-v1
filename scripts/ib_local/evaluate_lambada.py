"""Frozen LAMBADA development evaluation of existing CGB byte checkpoints."""
from __future__ import annotations

import os
import sys
if sys.path and os.path.abspath(sys.path[0]) == os.path.dirname(os.path.abspath(__file__)):
    sys.path.pop(0)

import argparse
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pyarrow.parquet as pq
import torch

from scripts.ib_local.evaluate_gated_deltanet import FrozenScorer, sha256, paired_summary
from scripts.ib_local.gated_deltanet import GatedDeltaNetLM


def split_target(text):
    """Match lm-evaluation-harness lambada_standard; no text normalization."""
    prompt, target = text.rsplit(" ", 1)
    if not prompt or not target:
        raise ValueError("LAMBADA requires nonempty context and last word")
    return prompt.encode("utf-8"), (" " + target).encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/cgb_eval_lambada"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.data/"manifest.json").read_text(encoding="utf-8"))
    path = args.data/"validation.parquet"
    if manifest["split"] != "validation" or sha256(path) != manifest["sha256"]:
        raise ValueError("registered development source required")
    texts = pq.read_table(path, columns=["text"])["text"].to_pylist()
    if len(texts) != manifest["rows"]:
        raise ValueError("row count mismatch")
    items = [split_target(text) for text in texts]
    # Length bucketing reduces padding; every original sample ID is retained.
    order = sorted(range(len(items)), key=lambda i: sum(map(len, items[i])))
    torch.set_num_threads(2)
    snapshot = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = GatedDeltaNetLM(**snapshot["config"]["model"]).to(args.device).eval()
    model.load_state_dict(snapshot["model"], strict=True)
    if model.vocab_size != 256:
        raise ValueError("UTF-8 byte model required")
    scorer = FrozenScorer(model, args.batch, args.chunk)
    code_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in
                   [Path(__file__), ROOT/"scripts/ib_local/evaluate_gated_deltanet.py",
                    ROOT/"scripts/ib_local/gated_deltanet.py"]}
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    results = {}
    for label, limit in [("last32", 32), ("last128", 128), ("full", None)]:
        nll = np.zeros(len(items))
        exact = np.zeros(len(items), dtype=bool)
        for start in range(0, len(order), args.batch):
            ids = order[start:start+args.batch]
            prompts = [items[i][0][-limit:] if limit else items[i][0] for i in ids]
            targets = [items[i][1] for i in ids]
            scores, matches = scorer.continuations(prompts, targets)
            nll[ids] = scores
            exact[ids] = matches
            if start % (args.batch*32) == 0:
                print(json.dumps(dict(context=label, done=start+len(ids), total=len(items))), flush=True)
        results[label] = dict(exact_prefix_accuracy=float(exact.mean()),
                              correct=int(exact.sum()), count=len(items),
                              mean_target_nll=float(nll.mean()),
                              target_perplexity=math.exp(float(nll.mean())),
                              target_bits_per_byte=float(nll.sum()/sum(len(t) for _, t in items)/math.log(2)),
                              sample_nll=nll.tolist(), sample_exact=exact.tolist())
        print(json.dumps(dict(context=label, **{k:v for k,v in results[label].items()
                                               if not k.startswith("sample_")})), flush=True)
    gains = {}
    for short in ("last32", "last128"):
        gain = paired_summary(np.array(results[short]["sample_nll"])-results["full"]["sample_nll"])
        gain["uncertainty"] = "paired passage bootstrap; book clustering, training-seed and corpus uncertainty not covered"
        old = np.array(results[short]["sample_exact"])
        new = np.array(results["full"]["sample_exact"])
        gains[short] = dict(target_nll_gain=gain, wrong_to_right=int((~old & new).sum()),
                            right_to_wrong=int((old & ~new).sum()))
    out = dict(protocol="lambada_standard_development_byte_v1", dataset=manifest,
               source_protocol="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/lambada/lambada_standard.yaml",
               checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint),
               checkpoint_events=snapshot["events"], checkpoint_step=snapshot["step"],
               config=model.config_dict(), code_sha256=code_hashes,
               parameter_count=sum(p.numel() for p in model.parameters()),
               batch=args.batch, execution_chunk=args.chunk, results=results, full_context_gains=gains,
               prompt_bytes=dict(min=min(len(p) for p,t in items), max=max(len(p) for p,t in items),
                                 mean=float(np.mean([len(p) for p,t in items]))),
               scoring="sum causal log probabilities of leading-space plus final word; all target bytes greedy-correct; gold length, no delimiter check",
               sample_order="original parquet row; length buckets affect execution only",
               state_policy="reset per passage/context arm, continuous within passage, frozen weights",
               training=False, test_split_used=False, live_learning_evaluated=False,
               architecture_verdict="undetermined", independent_review=False,
               contamination="No training on this evaluation set; cross-corpus overlap with OWT not audited",
               seconds=time.perf_counter()-started,
               peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if args.device.startswith("cuda") else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
