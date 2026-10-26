"""Multi-register flip-flop: GDN, GDN + tanh latch slots, GDN + conserved sector latches.

Prereg: docs/information_boltzmann/CGB_LINEAR_MEMORY_LATCH_PREREG_20261026.md (part 2).
usage: python scripts/run_latch_gdn_multi.py --out results/published/latch_gdn/latch_multi_20261026.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

J = 4                      # registers
WT = 0                     # write tokens 0..2J-1: register j, bit b -> 2j+b
IG = 2 * J                 # ignore tokens IG+b
RD = 2 * J + 2             # read tokens RD+j
V = 3 * J + 2
M = 8                      # latch slots / sectors
TRAIN_LEN, GAPS, SIGMA = 128, (16, 64, 256, 1024), 0.05
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def train_batch(B, T, gen):
    u = torch.rand(B, T, generator=gen)
    bit = torch.randint(0, 2, (B, T), generator=gen)
    reg = torch.randint(0, J, (B, T), generator=gen)
    tok = torch.where(u < 0.15, WT + 2 * reg + bit, torch.where(u < 0.85, IG + bit, RD + reg))
    tok[:, :J] = WT + 2 * torch.arange(J) + bit[:, :J]
    state = torch.zeros(B, J, dtype=torch.long)
    tgt = torch.zeros(B, T, dtype=torch.long)
    for t in range(T):
        w = tok[:, t] < IG
        j = (tok[:, t] - WT) // 2
        upd = torch.zeros(B, J, dtype=torch.bool)
        upd[torch.arange(B), j.clamp(0, J - 1)] = w
        state = torch.where(upd, ((tok[:, t] - WT) % 2)[:, None].expand(B, J), state)
        r = tok[:, t] >= RD
        tgt[:, t] = state[torch.arange(B), (tok[:, t] - RD).clamp(0, J - 1)]
        tgt[:, t] = torch.where(r, tgt[:, t], 0)
    return tok, tgt, tok >= RD


def gap_batch(B, G, gen):
    bits = torch.randint(0, 2, (B, J), generator=gen)
    writes = WT + 2 * torch.arange(J) + bits
    noise = IG + torch.randint(0, 2, (B, G), generator=gen)
    reads = (RD + torch.arange(J)).expand(B, J)
    return torch.cat([writes, noise, reads], 1), bits


def flip_batch(B, gen, rounds=12, gap=8):
    """Every register rewritten with alternating bits; used to find slots that never flip."""
    parts, b0 = [], torch.randint(0, 2, (B, J), generator=gen)
    for k in range(rounds):
        bits = (b0 + k) % 2
        parts += [WT + 2 * torch.arange(J) + bits, IG + torch.randint(0, 2, (B, gap), generator=gen)]
    return torch.cat(parts, 1)


class Model(nn.Module):
    def __init__(self, kind, d=16, emb=32):
        super().__init__()
        self.kind, self.d = kind, d
        self.emb = nn.Embedding(V, emb)
        self.qkv = nn.Linear(emb, 3 * d)
        self.ab = nn.Linear(emb, 2)
        extra = 0
        if kind == "tanh":
            self.U = nn.Linear(emb, M)
            self.a = nn.Parameter(torch.full((M,), 2.0))
            extra = M
        if kind == "sector":
            self.U = nn.Linear(emb, 2 * M)
            self.a = nn.Parameter(torch.full((M,), 2.0))
            self.eps_raw = nn.Parameter(torch.full((M,), -4.0))
            self.rho_raw = nn.Parameter(torch.full((M,), -3.0))
            extra = M
        self.head = nn.Sequential(nn.Linear(d + emb + extra, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, tok, sigma, zero=None, trace=False):
        B, T = tok.shape
        x = self.emb(tok)
        q, k, v = self.qkv(x).chunk(3, -1)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        ga, gb = self.ab(x).unbind(-1)
        alpha, beta = torch.exp(-F.softplus(ga)), torch.sigmoid(gb)
        S = x.new_zeros(B, self.d, self.d)
        if self.kind == "tanh":
            z = x.new_zeros(B, M)
        if self.kind == "sector":
            z = x.new_full((B, M, 2), 0.5)
            r = x.new_ones(B, M, 2)
            eps, rho = 0.05 * torch.sigmoid(self.eps_raw), 0.05 * torch.sigmoid(self.rho_raw)
        outs, tr = [], []
        for t in range(T):
            kt, vt, bt, at = k[:, t], v[:, t], beta[:, t, None, None], alpha[:, t, None, None]
            Sk = torch.einsum("bij,bj->bi", S, kt)
            S = at * (S - bt * Sk[..., None] * kt[:, None, :]) + bt * vt[..., None] * kt[:, None, :]
            S = (S + sigma * torch.randn_like(S)).clamp(-1, 1)
            feats = [torch.einsum("bij,bj->bi", S, q[:, t]), x[:, t]]
            if self.kind == "tanh":
                z = torch.tanh(self.a * z + self.U(x[:, t]))
                z = (z + sigma * torch.randn_like(z)).clamp(-1, 1)
                lat = z
            elif self.kind == "sector":
                logit = self.a[:, None] * torch.log(z + 1e-4) + torch.log(r + 1e-4) + self.U(x[:, t]).view(B, M, 2)
                z = torch.softmax(logit, -1)
                z = (z + sigma * torch.randn_like(z)).clamp(min=0)
                z = z / z.sum(-1, keepdim=True).clamp(min=1e-6)
                r = (r - eps[:, None] * z * r + rho[:, None] * (1 - z) * (1 - r)).clamp(0, 1)
                lat = z[..., 0] - 0.5
            if self.kind != "gdn":
                if zero is not None:
                    lat = lat.clone()
                    lat[:, zero] = 0
                feats.append(lat)
                if trace:
                    tr.append(lat.detach())
            outs.append(torch.cat(feats, -1))
        return self.head(torch.stack(outs, 1)), (torch.stack(tr, 1) if trace else None)


@torch.no_grad()
def gap_acc(model, G, gen, B=256, zero=None):
    tok, bits = gap_batch(B, G, gen)
    logits, _ = model(tok.to(DEV), SIGMA, zero=zero)
    return float((logits[:, -J:].argmax(-1).cpu() == bits).float().mean())


def gini(x):
    x = sorted(max(v, 0.0) for v in x)
    n, s = len(x), sum(x)
    if s == 0:
        return 0.0
    return float(sum((2 * (i + 1) - n - 1) * v for i, v in enumerate(x)) / (n * s))


def run(kind, seed, steps):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = Model(kind).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        tok, tgt, mask = train_batch(64, TRAIN_LEN, gen)
        logits, _ = model(tok.to(DEV), SIGMA)
        m = mask.to(DEV)
        loss = F.cross_entropy(logits[m], tgt.to(DEV)[m])
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    tg = torch.Generator().manual_seed(1000 + seed)
    out = dict(kind=kind, seed=seed, final_loss=float(loss.detach()), acc={G: gap_acc(model, G, tg) for G in GAPS})
    if kind != "gdn":
        base = gap_acc(model, 64, torch.Generator().manual_seed(7))
        contrib = [base - gap_acc(model, 64, torch.Generator().manual_seed(7), zero=s) for s in range(M)]
        with torch.no_grad():
            _, tr = model(flip_batch(128, tg).to(DEV), SIGMA, trace=True)
        sign = torch.sign(tr)
        flips = (sign[:, 1:] != sign[:, :-1]).any(1).float().mean(0).cpu()   # per slot, fraction of sequences
        out.update(contrib=contrib, gini=gini(contrib), never_flip=float((flips < 0.05).float().mean()),
                   used_never_flip=float(((flips < 0.05) & (torch.tensor(contrib) > 0.02)).float().sum()
                                         / max(1, int((torch.tensor(contrib) > 0.02).sum()))),
                   gain=model.a.detach().cpu().tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--kinds", nargs="+", default=["gdn", "tanh", "sector"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = []
    for kind in a.kinds:
        for seed in range(a.seeds):
            res.append(run(kind, seed, a.steps))
            print(json.dumps(res[-1]), flush=True)
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(dict(
                prereg="docs/information_boltzmann/CGB_LINEAR_MEMORY_LATCH_PREREG_20261026.md#part-2",
                args=vars(a), runs=res), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
