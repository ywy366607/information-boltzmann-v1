"""Requirement R gate for CLO6 (synthetic problems, not registered tasks): scale each estimate the
step depends on (step curvature, drift variance, population signal power) by 0.5 and by 2 and check
that the loop stays finite and within 3x of the unperturbed loss, on a stationary rotated regression
(D=48, seeds 77-79) and a drifting one (K=16, D=8, seed 5)."""
import sys, json, math
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_optimizer_continual as V3, run_optimizer_continual_v6 as V6, run_optimizer_shell_kalman as V1

class Perturbed(V6.CLO6):
    def __init__(self, shape, which=None, f=1.0, **kw):
        super().__init__(shape, **kw); self.which, self.f = which, f
    def _step_curvature(self, h):
        x = super()._step_curvature(h); return x * self.f if self.which == "curv" else x
    def _drift(self, r, c2):
        x = super()._drift(r, c2); return x * self.f if self.which == "drift" else x
    def _frac(self, P):
        if self.which != "tau2":
            return super()._frac(P)
        tau2 = (self.m * self.m - P).mean(0, keepdim=True)
        self.tau2 = tau2 if self.tau2 is None else 0.9 * self.tau2 + 0.1 * tau2
        t2 = self.tau2.clamp(min=0) * self.f
        return t2 / (t2 + P + V6.EPS)

def stationary(mk):
    out = []
    for seed in (77, 78, 79):
        gen = torch.Generator().manual_seed(seed); D = 48
        lam = torch.logspace(-2, 0, D); sig = torch.logspace(math.log10(0.03), 0, D)
        U, Vr = V1.rot(D, gen), V1.rot(D, gen); Sig = U @ torch.diag(lam) @ U.T
        Ws = torch.randn(D, D, generator=gen) / D ** 0.5; W = torch.zeros(D, D); opt = mk((D, D))
        for t in range(500):
            X = U @ (lam.sqrt()[:, None] * torch.randn(D, 24, generator=gen)); Y = Ws @ X + Vr @ (sig[:, None] * torch.randn(D, 24, generator=gen))
            opt.step(W, (W @ X - Y) @ X.T / 24, X, V3.lin_fg(X, Y))
        E = W - Ws; out.append(0.5 * float(torch.trace(E @ Sig @ E.T)))
    return out

def drifting(mk):
    K, D = 16, 8; lam = torch.logspace(-1, 0, D)
    gen = torch.Generator().manual_seed(5); Ws = torch.randn(K, D, generator=gen); W = torch.zeros(K, D); opt = mk((K, D)); errs = []
    for t in range(600):
        Ws += 0.01 * torch.randn(K, D, generator=gen)
        X = lam.sqrt()[:, None] * torch.randn(D, 32, generator=gen); Y = Ws @ X + 0.3 * torch.randn(K, 32, generator=gen)
        opt.step(W, (W @ X - Y) @ X.T / 32, X, V3.lin_fg(X, Y))
        errs.append(0.5 * float(torch.trace((W - Ws) @ torch.diag(lam) @ (W - Ws).T)))
    return [sum(errs[300:]) / 300]

res = {}
for prob, fn in (("stationary", stationary), ("drifting", drifting)):
    base = fn(lambda s: Perturbed(s))
    res[prob] = {"nominal": [round(x, 4) for x in base]}
    for which in ("curv", "drift", "tau2"):
        for f in (0.5, 2.0):
            v = fn(lambda s: Perturbed(s, which, f))
            ok = all(math.isfinite(a) and a <= 3 * b for a, b in zip(v, base))
            res[prob][f"{which}x{f}"] = dict(loss=[round(x, 4) for x in v], within_3x=ok)
res["gate_passed"] = all(v["within_3x"] for p in ("stationary", "drifting") for k, v in res[p].items() if k != "nominal")
print(json.dumps(res, indent=1))
