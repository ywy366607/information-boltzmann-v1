"""Can the brain know, at tick t, whether waiting will help THIS token?  Fixed 12 ticks per token; at tick 6 we keep the
motor read input (2,333 membranes) and the read's own confidence (entropy of its prediction); target = how much the
prediction improves by waiting to tick 10 (CE_6 - CE_10).  Held-out predictability (Spearman) of that benefit from:
  entropy only, motor membranes (ridge), both.  If nothing predicts it, a go neuron reading the motor surface cannot
learn to think on demand, however long it trains."""
import sys, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from scipy.stats import spearmanr
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scratch/fly_stream/think_and_speak.py', encoding='utf-8').read().split('report = {')[0])
T, AT, LATER = 12, 6, 10
torch.manual_seed(0)
targets = np.array(val[cursor:cursor + 4256], dtype=np.int64)
inputs = np.concatenate(([L['previous_token']], targets[:-1]))
X, H, D = [], [], []
s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
with torch.no_grad():
    for i, tok in enumerate(inputs):
        s, ticks = fl.advance_fly_input_event(model, s, torch.tensor([int(tok)], device='cuda'), settle_ticks=T - 1,
                                              writer_baseline_clock='physical', return_ticks=True, **opts)
        if i >= 256:
            st, later = ticks[AT - 1], ticks[LATER - 1]
            lp = F.log_softmax(model.decoder(model.read_norm(fl.read_fly_motor(model, st))).float(), -1)[0]
            lq = F.log_softmax(model.decoder(model.read_norm(fl.read_fly_motor(model, later))).float(), -1)[0]
            X.append(st.h[0, model.read_indices].cpu().numpy())
            H.append(float(-(lp.exp() * lp).sum()))
            D.append(float(-lp[int(targets[i])] + lq[int(targets[i])]))
X, H, D = np.array(X), np.array(H), np.array(D)
n = len(D); tr, te = slice(0, 3000), slice(3000, n)
print(f'benefit of waiting 6 -> 10 ticks: mean {D.mean():.4f}, sd {D.std():.4f}, fraction of tokens helped {np.mean(D > 0):.3f}')
res = {'entropy_only_spearman': float(spearmanr(H[te], D[te]).correlation)}
def ridge(Z):
    Z = (Z - Z[tr].mean(0)) / (Z[tr].std(0) + 1e-6)
    best = None
    for lam in (1, 10, 100, 1e3, 1e4):
        w = np.linalg.solve(Z[:2500].T @ Z[:2500] + lam * np.eye(Z.shape[1]), Z[:2500].T @ (D[:2500] - D[:2500].mean()))
        r = spearmanr(Z[2500:3000] @ w, D[2500:3000]).correlation
        if best is None or r > best[0]:
            best = (r, lam)
    w = np.linalg.solve(Z[tr].T @ Z[tr] + best[1] * np.eye(Z.shape[1]), Z[tr].T @ (D[tr] - D[tr].mean()))
    return float(spearmanr(Z[te] @ w, D[te]).correlation)
res['motor_ridge_spearman'] = ridge(X)
res['motor_plus_entropy_spearman'] = ridge(np.concatenate([X, H[:, None]], 1))
# what a go rule using the predictor could gain: wait to 10 only when predicted benefit > 0 (held-out)
print('held-out predictability of the benefit of waiting:', {k: round(v, 3) for k, v in res.items()})
json.dump(res, open(fork / 'wait_predictability.json', 'w'), indent=1)
