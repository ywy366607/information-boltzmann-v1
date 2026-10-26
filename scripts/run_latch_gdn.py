"""Gated DeltaNet vs GDN + latch slots on flip-flop under bounded noisy state.

Prereg: docs/information_boltzmann/CGB_LINEAR_MEMORY_LATCH_PREREG_20261026.md
usage: python scripts/run_latch_gdn.py --out results/published/latch_gdn/latch_gdn_20261026.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

W0, W1, I0, I1, R = range(5)
GAPS = (16, 64, 256, 1024, 4096)


def train_batch(B, T, gen):
    u = torch.rand(B, T, generator=gen)
    bit = torch.randint(0, 2, (B, T), generator=gen)
    tok = torch.where(u < 0.1, W0 + bit, torch.where(u < 0.9, I0 + bit, torch.full_like(bit, R)))
    tok[:, 0] = W0 + bit[:, 0]
    last = torch.zeros(B, T, dtype=torch.long)
    cur = bit[:, 0].clone()
    for t in range(T):
        w = tok[:, t] <= W1
        cur = torch.where(w, tok[:, t] - W0, cur)
        last[:, t] = cur
    return tok, last, tok == R


def gap_batch(B, G, gen):
    bit = torch.randint(0, 2, (B,), generator=gen)
    noise = torch.randint(0, 2, (B, G), generator=gen)
    tok = torch.cat([(W0 + bit)[:, None], I0 + noise, torch.full((B, 1), R)], 1)
    return tok, bit


class Model(nn.Module):
    def __init__(self, latch, d=16, m=4, emb=32):
        super().__init__()
        self.d, self.m, self.latch = d, m, latch
        self.emb = nn.Embedding(5, emb)
        self.qkv = nn.Linear(emb, 3 * d)
        self.ab = nn.Linear(emb, 2)
        nn.init.constant_(self.ab.bias, 0.0)
        if latch:
            self.U = nn.Linear(emb, m)
            self.a = nn.Parameter(torch.full((m,), 2.0))
        self.head = nn.Sequential(nn.Linear(d + emb + (m if latch else 0), 32), nn.GELU(), nn.Linear(32, 2))

    def forward(self, tok, sigma, zero_latch=False, record=False):
        B, T = tok.shape
        x = self.emb(tok)
        q, k, v = self.qkv(x).chunk(3, -1)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        ga, gb = self.ab(x).unbind(-1)
        alpha, beta = torch.exp(-F.softplus(ga)), torch.sigmoid(gb)
        S = x.new_zeros(B, self.d, self.d)
        z = x.new_zeros(B, self.m) if self.latch else None
        outs, alphas = [], []
        for t in range(T):
            kt, vt, bt, at = k[:, t], v[:, t], beta[:, t, None], alpha[:, t, None, None]
            Sk = torch.einsum("bij,bj->bi", S, kt)
            S = at * (S - bt[..., None] * Sk[..., None] * kt[:, None, :]) + bt[..., None] * vt[..., None] * kt[:, None, :]
            if sigma > 0:
                S = S + sigma * torch.randn_like(S)
            S = S.clamp(-1, 1)
            feats = [torch.einsum("bij,bj->bi", S, q[:, t]), x[:, t]]
            if self.latch:
                z = torch.tanh(self.a * z + self.U(x[:, t]))
                if sigma > 0:
                    z = z + sigma * torch.randn_like(z)
                z = z.clamp(-1, 1)
                feats.append(torch.zeros_like(z) if zero_latch else z)
            outs.append(torch.cat(feats, -1))
            if record:
                alphas.append(alpha[:, t])
        return self.head(torch.stack(outs, 1)), (torch.stack(alphas, 1) if record else None)


@torch.no_grad()
def gap_acc(model, G, sigma, gen, B=256, **kw):
    tok, bit = gap_batch(B, G, gen)
    logits, _ = model(tok.to(DEV), sigma, **kw)
    return float((logits[:, -1].argmax(-1).cpu() == bit).float().mean())


TRAIN_LEN = 128
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def run(latch, sigma, seed, steps):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = Model(latch).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for step in range(steps):
        tok, last, mask = train_batch(64, TRAIN_LEN, gen)
        logits, _ = model(tok.to(DEV), sigma)
        loss = F.cross_entropy(logits[mask.to(DEV)], last.to(DEV)[mask.to(DEV)])
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    tgen = torch.Generator().manual_seed(1000 + seed)
    out = dict(latch=latch, sigma=sigma, seed=seed, final_loss=float(loss.detach()),
               acc={G: gap_acc(model, G, sigma, tgen) for G in GAPS})
    if latch:
        out["gain"] = model.a.detach().cpu().tolist()
        out["acc_latch_zeroed"] = {G: gap_acc(model, G, sigma, tgen, zero_latch=True) for G in GAPS}
    with torch.no_grad():
        tok, _ = gap_batch(64, 256, tgen)
        _, al = model(tok.to(DEV), sigma, record=True)
        out["alpha_on_ignore_mean"] = float(al[:, 1:-1].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.02, 0.05])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = []
    for sigma in a.sigmas:
        for latch in (False, True):
            for seed in range(a.seeds):
                r = run(latch, sigma, seed, a.steps)
                res.append(r)
                print(json.dumps(r), flush=True)
                Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                Path(a.out).write_text(json.dumps(dict(
                    prereg="docs/information_boltzmann/CGB_LINEAR_MEMORY_LATCH_PREREG_20261026.md",
                    args=vars(a), runs=res), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
