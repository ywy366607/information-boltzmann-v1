"""E2 of the mode task: does a constant key component let a trained latch head hold the bit?
No training (2026-11-03).

Cause 2 of results/cgb_mode_task_20261103/RESULT.md: feedback_delta renews only r = S^T k_t,
the read along the current key, so content written at the marker's key decays with alpha.
Two checks on the trained latch runs:

(a) In the model.  For every head whose birth-formula latch ratio exceeds 1, the key becomes
    normalize(normalize(k) + c u) with a fixed random unit u per head, so every event reads and
    renews (mostly) the same direction.  c = 0 reproduces the trained model.  Reported: mode
    accuracy and flip_effect by distance (eval_mode bins), contrast of the patched heads, and
    text NLL off the queries (the price paid by the head's ordinary job).

(b) The mechanism alone.  For one latching head, the read along a key that is u at every
    event obeys r' = a r + b (target - a r), target = v + eta (softsign(g r + v) - v), with the
    head's real per-event a, b and value v on the evaluation text.  Two copies start at the
    two no-input fixed points +r* and -r*; their distance |r_A - r_B| / |r*| is tracked over
    32768 events, with v scaled by lambda in {1, 0.3, 0.1, 0}.  If the bit survives only at
    small lambda, the text value written at every event (weight 1 - eta) is what erases it,
    not the missing read direction.
Prediction written before running: (b) holds at lambda = 0 and fails at lambda = 1, because
eta is about 0.1 so the value term dominates the target; (a) therefore gains little at
small c and costs text NLL at large c.
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

import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from scripts.ib_local import gdn_fused  # noqa: E402
from scripts.ib_local.eval_mode import BINS, flipped, margins  # noqa: E402
from scripts.ib_local.eval_recall import load  # noqa: E402
from scripts.ib_local.mode_task import EVAL_SEGMENT, eval_stream  # noqa: E402
from scripts.ib_local.probe_mode_heads import latch_ratio, states  # noqa: E402

CHUNK = 512
SEGMENTS = 12
OFFSETS = (0.0, 1.0, 4.0)
LAMBDAS = (1.0, 0.3, 0.1, 0.0)
SELF_KEY = {}      # layer -> (head mask [H], u [H, dk], c)
CAPTURE = {}       # layer -> list of (alpha, beta, v) per chunk, when capturing


def patched_forward(self, x, state):
    """gdn_fused.fused_forward with the self-key offset and optional capture."""
    B, T, d = x.shape
    x_norm = self.norm(x)
    q = self.q_proj(x_norm).view(B, T, self.heads, self.dk)
    k = self.k_proj(x_norm).view(B, T, self.heads, self.dk)
    v = self.v_proj(x_norm).view(B, T, self.heads, self.dv)
    S = state[:, :self.matrix_size].reshape(B, self.heads, self.dk, self.dv)
    history = state[:, self.matrix_size:].reshape(B, 3*d, self.conv_kernel-1)
    projected = torch.cat((q.reshape(B, T, d), k.reshape(B, T, d),
                           v.reshape(B, T, d)), dim=-1).transpose(1, 2)
    mixed, next_history = self._mix_qkv(projected, history)
    q, k, v = [part.reshape(B, T, self.heads, self.dk) for part in mixed.transpose(1, 2).split(d, dim=-1)]
    g = self.g_proj(x_norm)
    alpha = self.retention(x_norm)
    beta = torch.sigmoid(self.beta_proj(x_norm))
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    if self in SELF_KEY and SELF_KEY[self][2] > 0:
        mask, u, c = SELF_KEY[self]
        k = torch.where(mask[None, None, :, None], F.normalize(k + c * u[None, None], dim=-1), k)
    if self in CAPTURE:
        CAPTURE[self].append((alpha[0].cpu(), beta[0].cpu(), v[0].cpu()))
    strength = self.feedback_logit.sigmoid()
    feedback_gain = F.softplus(self.feedback_gain_raw)
    out, S, _, _ = gdn_fused.FusedDelta.apply(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        alpha.transpose(1, 2), beta.transpose(1, 2), strength, feedback_gain, S,
        self.content_budget, True)
    out = out.transpose(1, 2).reshape(B, T, d)
    result = x + self.o_proj(out * F.silu(g))
    if self.mlp is not None:
        normalized = self.mlp_norm(result)
        result = result + self.mlp["down"](F.silu(self.mlp["gate"](normalized)) * self.mlp["up"](normalized))
    pieces = [S.reshape(B, self.matrix_size), next_history.reshape(B, self.history_size)]
    return result, torch.cat(pieces, dim=-1), {"alpha_mean": alpha.detach().mean()}


@torch.no_grad()
def text_nll(model, state, stream, query_positions):
    n = (len(stream) - 1) // CHUNK * CHUNK
    data = torch.from_numpy(stream[:n + 1].astype(np.int64)).cuda()[None]
    uniform = torch.full((1, CHUNK, 256), math.log(1 / 256), device="cuda")
    keep = np.ones(n, bool)
    keep[[p for p in query_positions if p < n]] = False
    total = []
    for at in range(0, n, CHUNK):
        logits, state, _ = model.predict(data[:, at:at + CHUNK], state, uniform)
        nll = F.cross_entropy(logits[0], data[0, at + 1:at + CHUNK + 1], reduction="none")
        total.append(nll.cpu().numpy())
    return float(np.concatenate(total)[keep].mean())


def mode_rows(true, distance):
    rows = {}
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        mask = (distance[:len(true)] >= lo) & (distance[:len(true)] < hi)
        rows[f"{lo}-{hi}"] = dict(accuracy=float((true[mask, 0] > 0).mean()))
    return rows


def simulate(alpha, beta, v, eta, gain, lam):
    """Read along a constant key, two copies from the no-input fixed points +-r*."""
    r = torch.full((v.shape[-1],), 0.5)
    for _ in range(2000):
        a, b = alpha.mean(), beta.mean()
        r = a * r + b * (eta * F.softsign(gain * r) - a * r)
    star = r.norm()
    pair = torch.stack([r, -r])
    out = {}
    marks = {64, 512, 2048, 8192, 32767}
    for t in range(len(alpha)):
        a, b, vt = alpha[t], beta[t], lam * v[t]
        target = vt + eta * (F.softsign(gain * pair + vt) - vt)
        pair = a * pair + b * (target - a * pair)
        if t in marks:
            out[t + 1] = float((pair[0] - pair[1]).norm() / (2 * star))
    return float(star), out


def main():
    gdn_fused.enable()
    runs = [Path(p) for p in sys.argv[1:]]
    stream, _, queries = eval_stream(SEGMENTS * EVAL_SEGMENT)
    flip = flipped(stream)
    distance = np.array([q[2] for q in queries])
    positions = [q[0] for q in queries]
    torch.manual_seed(0)
    report = {}
    for run in runs:
        model, state0, _ = load(run)
        layers = list(model.base.layers)
        for layer in layers:
            layer.forward = patched_forward.__get__(layer)
        ratio = torch.tensor(latch_ratio(model))
        units = [F.normalize(torch.randn(l.heads, l.dk, device="cuda"), dim=-1) for l in layers]
        report[run.name] = {"latch_ratio": ratio.tolist(), "model": {}, "mechanism": {}}
        for c in OFFSETS:
            for i, layer in enumerate(layers):
                SELF_KEY[layer] = ((ratio[i] > 1).cuda(), units[i], c)
            a = margins(model, state0.clone(), stream, queries)
            b = margins(model, state0.clone(), flip, queries)
            rows = mode_rows(a, distance)
            for (lo, hi), name in zip(zip(BINS[:-1], BINS[1:]), rows):
                mask = (distance[:len(a)] >= lo) & (distance[:len(a)] < hi)
                rows[name]["flip_effect"] = float((a[mask, 1] - b[mask, 1]).mean())
            sa, sb = states(model, state0.clone(), stream), states(model, state0.clone(), flip)
            contrast = {}
            for d in (512, 4096, 16384):
                keys = [k for k in sa if k % EVAL_SEGMENT == d]
                cval = torch.stack([(sa[k] - sb[k]).flatten(-2).norm(dim=-1)
                                    / sa[k].flatten(-2).norm(dim=-1).clamp_min(1e-9) for k in keys]).mean(0)
                contrast[d] = float(cval[ratio > 1].max())
            nll = text_nll(model, state0.clone(), stream, positions)
            report[run.name]["model"][c] = dict(mode=rows, patched_head_contrast_max=contrast, text_nll=nll)
            print(run.name, f"c={c}", "text_nll=%.4f" % nll,
                  "acc " + " ".join(f"{k}:{r['accuracy']:.2f}" for k, r in rows.items() if k in
                                    ("256-512", "2048-4096", "4096-8192", "8192-16384", "16384-32768")),
                  "flip16k+ %.3f" % rows["16384-32768"]["flip_effect"],
                  "contrast", {k: round(v, 3) for k, v in contrast.items()}, flush=True)
        for layer in layers:
            SELF_KEY[layer] = (torch.zeros(layer.heads, dtype=torch.bool, device="cuda"), units[0], 0.0)
        li, hi_ = divmod(int(ratio.flatten().argmax()), ratio.shape[1])
        layer = layers[li]
        CAPTURE[layer] = []
        margins(model, state0.clone(), stream[:EVAL_SEGMENT + CHUNK], [])
        captured = CAPTURE.pop(layer)
        alpha = torch.cat([x[0] for x in captured])[:EVAL_SEGMENT, hi_].double()
        beta = torch.cat([x[1] for x in captured])[:EVAL_SEGMENT, hi_].double()
        v = torch.cat([x[2] for x in captured])[:EVAL_SEGMENT, hi_].double()
        eta = float(layer.feedback_logit.sigmoid()[hi_])
        gain = float(F.softplus(layer.feedback_gain_raw)[hi_])
        mech = dict(layer=li, head=hi_, eta=eta, gain=gain, alpha_mean=float(alpha.mean()),
                    beta_mean=float(beta.mean()), v_norm_mean=float(v.norm(dim=-1).mean()))
        for lam in LAMBDAS:
            star, sep = simulate(alpha, beta, v, eta, gain, lam)
            mech[f"lambda={lam}"] = dict(r_star_norm=star, separation=sep)
            print(f"  mechanism L{li}H{hi_} eta={eta:.3f} g={gain:.2f} lambda={lam}: |r*|={star:.3f}",
                  {k: round(x, 3) for k, x in sep.items()}, flush=True)
        report[run.name]["mechanism"] = mech
        for layer in layers:
            del layer.forward
    out = Path("results/cgb_mode_task_20261103/selfkey.json")
    out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
