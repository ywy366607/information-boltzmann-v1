"""Mechanism checks for CLO on a synthetic quadratic that is not a registered task.

Checks: curvature and noise estimates against the truth in the closed loop (efference copy),
bounded rho with fresh scoring, latch holds a short burst and releases after a long one.
"""
import sys, json, math
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_optimizer_continual as R

def regression(opt, W, Ws, lam, steps, gen, sig=0.3, shift=None, B=32):
    D = W.shape[1]
    for t in range(steps):
        X = lam.sqrt()[:, None] * torch.randn(D, B, generator=gen)
        T = Ws if shift is None else Ws + shift(t)
        Y = T @ X + sig * torch.randn(W.shape[0], B, generator=gen)
        opt.step(W, (W @ X - Y) @ X.T / B, X, R.lin_fg(X, Y))

out = {}
gen = torch.Generator().manual_seed(7)
K, D = 16, 8
lam = torch.logspace(-1, 0, D)
Ws = torch.randn(K, D, generator=gen)
W = torch.zeros(K, D)
opt = R.CLO((K, D), latch=False)
regression(opt, W, Ws, lam, 300, gen)
H = opt.Q.T @ torch.diag(lam) @ opt.Q
htrue = torch.diagonal(H)[None].expand(K, D)
h = (opt.sxy / (opt.sxx + 1e-12)).abs()
# true gradient-noise variance per element at the current W, in the basis
gs = []
for _ in range(400):
    X = lam.sqrt()[:, None] * torch.randn(D, 32, generator=gen)
    Y = Ws @ X + 0.3 * torch.randn(K, 32, generator=gen)
    gs.append(((W @ X - Y) @ X.T / 32) @ opt.Q)
rtrue = torch.stack(gs).var(0)
c11 = (1 - 0.95 ** (opt.nd - 1)).clamp(min=0.05)
rhat = (-opt.e11 / c11).clamp(min=0)
q = lambda v: [round(float(x), 3) for x in torch.quantile(v.flatten(), torch.tensor([0.1, 0.5, 0.9]))]
out["h_over_true_q10_50_90"] = q(h / htrue)
out["rhat_over_true_q10_50_90"] = q(rhat / rtrue)
out["rhat_zero_fraction"] = round(float((rhat == 0).float().mean()), 3)
out["rho_final"] = opt.rho
out["rho_counts"] = {k: v for k, v in opt.stats.items() if k.startswith("rho")}
out["excess_after_300"] = round(0.5 * float(torch.trace((W - Ws) @ torch.diag(lam) @ (W - Ws).T)), 4)

# latch: train to rest, then a burst of length 4 (held) and of length 24 (released, adopted)
for L in (4, 24):
    gen = torch.Generator().manual_seed(11)
    W = torch.zeros(K, D)
    opt = R.CLO((K, D))
    regression(opt, W, Ws, lam, 400, gen)
    cons_before = float(opt.cons.float().mean())
    W0 = W.clone()
    delta = torch.zeros(K, D); delta[:4] = 1.0
    regression(opt, W, Ws, lam, L, gen, shift=lambda t: delta)
    moved = float((W - W0)[:4].norm() / delta[:4].norm())
    regression(opt, W, Ws, lam, 30, gen, shift=lambda t: delta)   # the change lasts
    adopted = float(1 - (W - Ws - delta)[:4].norm() / delta[:4].norm())
    out[f"latch_burst{L}"] = dict(consolidated_before=round(cons_before, 3), moved_during_burst=round(moved, 3),
                                   adopted_after_30_more=round(adopted, 3), releases=opt.stats["releases"])
print(json.dumps(out, indent=1))
