"""Frozen-weight, live-state evaluation for runs of train_gdn_fast_table.py --stream rich.

From the run's final trained state, on unseen text (train split from byte 12M):
  * accuracy of the long state M by distance from its marker (segments 131072 bytes, 4x the
    longest trained), and of the medium state P by distance (its training distribution);
  * M accuracy within 256 bytes after a decoy versus the same distances without one;
  * flip effects: log p(answer) minus the same with every M (or P) marker letter swapped;
  * where each state lives: per-head state contrast between the true and the flipped stream
    at chunk boundaries, binned by distance from the marker;
  * which heads it needs: memory ablation, the state of one head zeroed at every 512-byte
    boundary (it can hold nothing longer), and the change in M and P accuracy;
  * text NLL off the queries;
  * for runs with latch slots (latch_slot.py): the same per-head contrast read along each
    slot direction (S^T u), and M accuracy with every slot's renewal switched off (rho = 0).
Accuracies are balanced: the mean of the accuracy on the true stream and on the stream with
that state's marker letters swapped (where the correct answer swaps too), so a model that
always gives one letter scores exactly 0.5 whatever the mix of letters in the few long
segments (the unbalanced version misled on the 8-segment mode evaluation, 2026-11-04).
"""

from __future__ import annotations

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)  # ib_local/types.py would shadow the standard library
ROOT_DIR = os.path.dirname(os.path.dirname(script_dir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from scripts.ib_local import gdn_fused  # noqa: E402
from scripts.ib_local.eval_recall import load  # noqa: E402
from scripts.ib_local.rich_task import EVAL_LONG, eval_stream, flip_markers  # noqa: E402

CHUNK = 512
M_BINS = (0, 1024, 4096, 16384, 32768, 65536, 131072)
P_BINS = (0, 64, 256, 1024)
M_LETTERS, P_LETTERS = (ord("A"), ord("B")), (ord("x"), ord("y"))


@torch.no_grad()
def run(model, state, stream, zero_head=None):
    """Per-position log-probabilities of the four answer letters, text NLL per position, and
    per-head memory matrices at chunk boundaries."""
    n = (len(stream) - 1) // CHUNK * CHUNK
    data = torch.from_numpy(stream[:n + 1].astype(np.int64)).cuda()[None]
    uniform = torch.full((1, CHUNK, 256), math.log(1 / 256), device="cuda")
    letters = torch.tensor(M_LETTERS + P_LETTERS, device="cuda")
    base = model.base
    layer = base.layers[0]
    width = layer.dk * layer.dv
    answers = np.empty((n, 4), np.float32)
    nll = np.empty(n, np.float32)
    heads = {}
    for at in range(0, n, CHUNK):
        if zero_head is not None:
            l, h = zero_head
            state = state.clone()
            state[l, :, h * width:(h + 1) * width] = 0
        logits, state, _ = model.predict(data[:, at:at + CHUNK], state, uniform)
        logp = torch.log_softmax(logits[0], -1)
        answers[at:at + CHUNK] = logp[:, letters].cpu().numpy()
        nll[at:at + CHUNK] = F.nll_loss(logp, data[0, at + 1:at + CHUNK + 1], reduction="none").cpu().numpy()
        flat = state[:, 0, :layer.matrix_size].reshape(base.num_layers, base.heads, width)
        heads[at + CHUNK] = flat.float().cpu()
    return answers, nll, heads, n


def correct_margin(answers, queries):
    out = []
    for position, kind, answer, _, _ in queries:
        a = answers[position]
        pair = (0, 1) if kind == 0 else (2, 3)
        right = pair[0] if answer in (M_LETTERS[0], P_LETTERS[0]) else pair[1]
        wrong = pair[1] if right == pair[0] else pair[0]
        out.append((a[right] - a[wrong], a[right]))
    return np.array(out)


def balanced(true_margin, flip_margin):
    """Per query: 1 if right on the true stream, plus 1 if right on the flipped one, over 2."""
    return 0.5 * ((true_margin[:, 0] > 0).astype(float) + (flip_margin[:, 0] < 0).astype(float))


def binned(score, values, distance, bins):
    rows = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (distance >= lo) & (distance < hi)
        if mask.any():
            rows[f"{lo}-{hi}"] = dict(n=int(mask.sum()), accuracy=float(score[mask].mean()),
                                      margin=float(values[mask, 0].mean()))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--segments", type=int, default=6)
    parser.add_argument("--no-ablation", action="store_true")
    args = parser.parse_args()
    gdn_fused.enable()
    stream, _, queries = eval_stream(args.segments * EVAL_LONG)
    n_max = (len(stream) - 1) // CHUNK * CHUNK
    queries = [q for q in queries if q[0] < n_max]
    kind = np.array([q[1] for q in queries])
    distance = np.array([q[3] for q in queries])
    since_decoy = np.array([-1 if q[4] is None else q[4] for q in queries])
    keep = np.ones(n_max, bool)
    keep[[q[0] for q in queries]] = False
    flips = {name: flip_markers(stream, name) for name in ("M", "P")}
    for path in args.runs:
        model, state0, summary = load(path)
        slots = None
        if summary.get("latch_slots"):
            layer0 = model.base.layers[0]
            slots = torch.stack([F.normalize(l.slot_key, dim=-1) for l in model.base.layers]).float().cpu()
            slot_shape = (layer0.dk, layer0.dv)
        answers, nll, heads, _ = run(model, state0.clone(), stream)

        def slot_read(flat, u):  # flat [L, H, dk*dv] -> |S^T u| per head, [L, H, dv]
            return torch.einsum("lhkv,lhk->lhv", flat.reshape(*flat.shape[:2], *slot_shape), u)
        margin = correct_margin(answers, queries)
        report = dict(text_nll=float(nll[keep[:len(nll)]].mean()))
        mk, pk = kind == 0, kind == 1
        report["flip"], report["head_contrast"] = {}, {}
        flip_margin = {}
        for name, mask, bins in (("M", mk, M_BINS), ("P", pk, P_BINS)):
            f_answers, _, f_heads, _ = run(model, state0.clone(), flips[name])
            f_margin = correct_margin(f_answers, queries)
            flip_margin[name] = f_margin
            effect = margin[:, 1] - f_margin[:, 1]
            report["flip"][name] = {f"{lo}-{hi}": float(effect[mask & (distance >= lo) & (distance < hi)].mean())
                                    for lo, hi in zip(bins[:-1], bins[1:])
                                    if (mask & (distance >= lo) & (distance < hi)).any()}
            # per-head contrast at chunk boundaries, by distance from the latest marker of that state
            marker = b"\n<mode:" if name == "M" else b"\n<who:"
            raw = stream.tobytes()
            starts = []
            at = raw.find(marker)
            while at >= 0:
                starts.append(at)
                at = raw.find(marker, at + 1)
            starts = np.array(starts)
            rows, slot_rows = {}, {}
            for lo, hi in zip(bins[:-1], bins[1:]):
                picks, slot_picks = [], []
                for k in heads:
                    prior = starts[starts < k]
                    if len(prior) and lo <= k - prior[-1] < hi:
                        a, b = heads[k], f_heads[k]
                        picks.append((a - b).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-9))
                        if slots is not None:
                            ra, rb = slot_read(a, slots), slot_read(b, slots)
                            slot_picks.append((ra - rb).norm(dim=-1) / ra.norm(dim=-1).clamp_min(1e-9))
                if picks:
                    rows[f"{lo}-{hi}"] = torch.stack(picks).mean(0).tolist()
                if slot_picks:
                    slot_rows[f"{lo}-{hi}"] = torch.stack(slot_picks).mean(0).tolist()
            report["head_contrast"][name] = rows
            if slots is not None:
                report.setdefault("slot_contrast", {})[name] = slot_rows
        score = np.where(mk, balanced(margin, flip_margin["M"]), balanced(margin, flip_margin["P"]))
        report["M"] = binned(score[mk], margin[mk], distance[mk], M_BINS)
        report["P"] = binned(score[pk], margin[pk], distance[pk], P_BINS)
        recent = mk & (since_decoy >= 0) & (since_decoy < 256) & (distance >= 4096)
        calm = mk & ~((since_decoy >= 0) & (since_decoy < 256)) & (distance >= 4096)
        report["M_after_decoy"] = dict(n=int(recent.sum()), accuracy=float(score[recent].mean()),
                                       calm_n=int(calm.sum()), calm_accuracy=float(score[calm].mean()))
        if slots is not None:
            saved = [l.slot_rate_logit.detach().clone() for l in model.base.layers]
            for l in model.base.layers:
                l.slot_rate_logit.data.fill_(-1e4)
            o_margin = correct_margin(run(model, state0.clone(), stream)[0], queries)
            of_margin = correct_margin(run(model, state0.clone(), flips["M"])[0], queries)
            for l, value in zip(model.base.layers, saved):
                l.slot_rate_logit.data.copy_(value)
            o_score = balanced(o_margin, of_margin)
            report["M_slots_off"] = binned(o_score[mk], o_margin[mk], distance[mk], M_BINS)
            report["slot_rates"] = [(l.slot_rate_logit.sigmoid() * l.slot_mask).tolist()
                                    for l in model.base.layers]
        if not args.no_ablation:
            report["ablation"] = {}
            base = model.base
            far_m = mk & (distance >= 4096)
            for l in range(base.num_layers):
                for h in range(base.heads):
                    # balanced: the head ablated on the true and on the M-flipped stream
                    a_margin = correct_margin(run(model, state0.clone(), stream, zero_head=(l, h))[0], queries)
                    af_margin = correct_margin(run(model, state0.clone(), flips["M"], zero_head=(l, h))[0], queries)
                    m_score = balanced(a_margin, af_margin)
                    report["ablation"][f"L{l}H{h}"] = dict(
                        M_far=float(m_score[far_m].mean()),
                        P=float((a_margin[pk, 0] > 0).mean()))
            report["ablation"]["none"] = dict(M_far=float(score[far_m].mean()),
                                              P=float((margin[pk, 0] > 0).mean()))
        (path / "rich_eval.json").write_text(json.dumps(report, indent=1))
        print(path.name, "text_nll %.4f" % report["text_nll"])
        print("  M", {k: round(v["accuracy"], 2) for k, v in report["M"].items()})
        print("  P", {k: round(v["accuracy"], 2) for k, v in report["P"].items()})
        print("  M after decoy %.2f (n=%d) vs calm %.2f" % (report["M_after_decoy"]["accuracy"],
              report["M_after_decoy"]["n"], report["M_after_decoy"]["calm_accuracy"]))
        print("  flip M", {k: round(v, 2) for k, v in report["flip"]["M"].items()})
        if slots is not None:
            print("  M slots off", {k: round(v["accuracy"], 2) for k, v in report["M_slots_off"].items()})
        if "ablation" in report:
            none = report["ablation"]["none"]
            drops = sorted(((none["M_far"] - v["M_far"], none["P"] - v["P"], k)
                            for k, v in report["ablation"].items() if k != "none"), reverse=True)
            print("  ablation drops (M_far, P, head):", [(round(a, 2), round(b, 2), k) for a, b, k in drops[:6]])
        sys.stdout.flush()


if __name__ == "__main__":
    main()
