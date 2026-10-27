"""Calibration check for CLO4 (not registered tasks): fresh decrease / predicted decrease on
evidence steps, and the bound rho, on (a) a noisy quadratic (K=16, D=8, seeds 7-9) and (b) softmax
with Zipf classes (K=256, D=32, B=128, seed 99). Accuracy and loss are not read."""
import sys, json
from pathlib import Path
import torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_optimizer_continual as V3, run_optimizer_continual_v4 as V4

def spy(opt, store):
    orig = opt._trust
    def f(a, p, l):
        if abs(p) > max(V3.RESOLUTION * abs(l), V3.EVIDENCE_Z * opt.pred_sd, V3.EPS):
            store.append(a / p)
        orig(a, p, l)
    opt._trust = f

def q(r):
    return [round(float(x), 3) for x in torch.quantile(torch.tensor(r), torch.tensor([.25, .5, .75]))] if r else None

out = {}
for cls_name, cls in (("clo_v3", V3.CLO), ("clo4", V4.CLO4)):
    for seed in (7, 8, 9):
        gen = torch.Generator().manual_seed(seed); K, D = 16, 8
        lam = torch.logspace(-1, 0, D); Ws = torch.randn(K, D, generator=gen)
        W = torch.zeros(K, D); opt = cls((K, D)); rat = []; spy(opt, rat)
        for t in range(300):
            X = lam.sqrt()[:, None] * torch.randn(D, 32, generator=gen); Y = Ws @ X + 0.3 * torch.randn(K, 32, generator=gen)
            opt.step(W, (W @ X - Y) @ X.T / 32, X, V3.lin_fg(X, Y))
        out[f"{cls_name}_quadratic_seed{seed}"] = dict(n=len(rat), ratio_q25_50_75=q(rat), rho_final=f"{opt.rho:.1e}")
    gen = torch.Generator().manual_seed(99); K, D, B = 256, 32, 128
    E = F.normalize(torch.randn(K, D, generator=gen), dim=-1)
    p = torch.arange(1, K + 1, dtype=torch.float).pow(-1.2); p /= p.sum()
    W = torch.zeros(K, D); opt = cls((K, D)); rat = []; spy(opt, rat); rhos = {}
    for t in range(150):
        y = torch.multinomial(p, B, replacement=True, generator=gen)
        def fg(Wc, need, y=y):
            Wg = Wc.clone().requires_grad_(need)
            with torch.enable_grad():
                L = F.cross_entropy(E[y] @ Wg.T, y)
                if need: L.backward()
            return float(L.detach()), (Wg.grad if need else None)
        G = fg(W, True)[1]
        with torch.no_grad(): opt.step(W, G, E[y].T, fg)
        if t in (10, 30, 60, 149): rhos[t] = f"{opt.rho:.1e}"
    out[f"{cls_name}_softmax"] = dict(n=len(rat), ratio_q25_50_75=q(rat), rho=rhos)
print(json.dumps(out, indent=1))
