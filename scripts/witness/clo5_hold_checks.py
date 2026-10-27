"""Mechanism checks for the CLO5 hold (synthetic problems, not registered tasks).

(1) pure drift: tracking error with the hold vs without (B1: the hold must pass drift);
(2) a 3-step burst after 400 quiet steps: damage with vs without the hold (B2/B3);
(3) a 30-step burst: accepted, then the return after it reverts (a new value is held shallow);
(4) k*(age) grows with age (B3); (5) a small two-layer ReLU net (D=16, K=4, seed 77) keeps learning
through its plateau with the hold (E's v4 failure)."""
import sys, json
from pathlib import Path
import torch, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_optimizer_continual as V3, run_optimizer_continual_v5 as V5

K, D = 16, 8
lam = torch.logspace(-1, 0, D)

def run(opt, steps, gen, Ws, drift=0.0, shift=None, track=None):
    W = track
    errs = []
    for t in range(steps):
        if drift:
            Ws += drift * torch.randn(K, D, generator=gen)
        T = Ws + (shift(t) if shift else 0)
        X = lam.sqrt()[:, None] * torch.randn(D, 32, generator=gen)
        Y = T @ X + 0.3 * torch.randn(K, 32, generator=gen)
        opt.step(W, (W @ X - Y) @ X.T / 32, X, V3.lin_fg(X, Y))
        errs.append(0.5 * float(torch.trace((W - Ws) @ torch.diag(lam) @ (W - Ws).T)))
    return errs

out = {}
for hold in (True, False):
    gen = torch.Generator().manual_seed(5); Ws = torch.randn(K, D, generator=gen)
    W = torch.zeros(K, D); opt = V5.CLO5((K, D), hold=hold)
    e = run(opt, 600, gen, Ws, drift=0.01, track=W)
    out[f"drift_hold{hold}"] = round(sum(e[300:]) / 300, 4)
for L in (3, 30):
    for hold in (True, False):
        gen = torch.Generator().manual_seed(6); Ws = torch.randn(K, D, generator=gen)
        W = torch.zeros(K, D); opt = V5.CLO5((K, D), hold=hold)
        delta = torch.zeros(K, D); delta[:4] = 1.5
        e = run(opt, 400 + L + 40, gen, Ws, shift=lambda t: delta if 400 <= t < 400 + L else 0, track=W)
        out[f"burst{L}_hold{hold}"] = dict(peak_during=round(max(e[400:400 + L]), 3), mean_10_after=round(sum(e[400 + L:410 + L]) / 10, 3),
                                           mean_30_40_after=round(sum(e[430 + L:440 + L]) / 10, 3), before=round(sum(e[390:400]) / 10, 4))
        if hold and L == 3:
            ks = opt.kstar()
            out["kstar_q10_50_90_at_end"] = [round(float(x), 2) for x in torch.quantile(ks.flatten(), torch.tensor([.1, .5, .9]))]
            out["pi_b_s_b"] = [round(opt.pi_b, 4), round(opt.s_b, 3)]
            out["stats"] = {k: v for k, v in opt.stats.items() if k in ("jumps", "accepted", "burst_runs")}
# k* versus age at fixed pooled statistics
opt = V5.CLO5((1, 4)); opt.pi_b, opt.s_b = 0.02, 0.6
opt.age = torch.tensor([[0., 10., 100., 1000.]])
out["kstar_at_age_0_10_100_1000"] = [round(float(x), 2) for x in opt.kstar().flatten()]
# small two-layer net
for hold in (True, False):
    gen = torch.Generator().manual_seed(77); De, Ke = 16, 4
    Fq = torch.linalg.qr(torch.randn(De, Ke, generator=gen))[0].T
    amp = torch.arange(1, Ke + 1, dtype=torch.float).pow(-0.7); teach = lambda X: amp @ F.relu(Fq @ X)
    W1 = torch.randn(Ke, De, generator=gen) / De ** 0.5; W2 = torch.randn(1, Ke, generator=gen) / Ke ** 0.5
    o1, o2 = V5.CLO5(W1.shape, hold=hold), V5.CLO5(W2.shape, hold=hold)
    def grads(A, B, X, y):
        A, B = A.clone().requires_grad_(True), B.clone().requires_grad_(True)
        with torch.enable_grad():
            L = 0.5 * ((B @ F.relu(A @ X) - y) ** 2).mean(); L.backward()
        return float(L.detach()), A.grad, B.grad
    Ls = []
    for t in range(800):
        X = torch.randn(De, 64, generator=gen); y = teach(X)
        L, G1, G2 = grads(W1, W2, X, y); Ls.append(L)
        with torch.no_grad():
            H0 = F.relu(W1 @ X); W1o, W2o = W1.clone(), W2.clone()
            o1.step(W1, G1, X, lambda Wc, need: (lambda r: (r[0], r[1] if need else None))(grads(Wc, W2o, X, y)))
            o2.step(W2, G2, H0, lambda Wc, need: (lambda r: (r[0], r[2] if need else None))(grads(W1o, Wc, X, y)))
    out[f"twolayer_hold{hold}_loss_at_100_400_800"] = [round(sum(Ls[i - 20:i]) / 20, 4) for i in (100, 400, 800)]
print(json.dumps(out, indent=1))
