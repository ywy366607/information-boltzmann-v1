"""P2 and P3 of the representation-level surprise hypothesis (FLY_ACTIVE_INFERENCE_GOAL_20261014_ACTS.md).  The Kenyon
code is a context code (c7: same last-3 words overlap 0.47 vs 0.08), so ask what it carries beyond the token.
Reads a stretch of validation text as kenyon_repeat_probe.py does (gap 0), and per word t:
- token surprise: the brain's CE on w_t (read after w_{t-1}), and the token's rarity (unigram -log p);
- Kenyon novelty: 1 - the best cosine of word t's Kenyon active set with any of the previous `memory` words'
  (familiarity of the context pattern, the alpha'3 route, no predictor needed);
- Kenyon prediction error: 1 - cosine(predicted, actual) Kenyon set, predicted by ridge from the Kenyon + ALPN sets at
  t-1 (fitted on the first half, scored on the second);
P2: rank correlations of novelty / error with rarity (token surprise ~ rarity was rho 0.97), with the brain's CE, and
with the CE residual after rarity.
P3 (same meaning, different words): context vectors from distributional semantics of the training text (PPMI-SVD,
+-2 words, top vocabulary), the last 3 words weighted 1, 0.5, 0.25.  For anchor words, the nearest context sharing NO
token among the last 3 words vs a random context, both >= 16 words away: overlap per layer.  If the Kenyon code
carries meaning, neighbour > random beyond what the receptors and ALPN already share.
Distributions as [min | median | mean | max].
Usage: python kenyon_semantic_probe.py <checkpoint dir> [words=3000] [out.json]
"""
import sys, json, numpy as np, pandas as pd, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork = Path(sys.argv[1])
N_WORDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
model, cfg, L = load(fork)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
writer = model.topographic_writer
HOLD = bool(cfg.get('hold_input'))
physical_clock = cfg.get('writer_baseline_clock', 'physical') == 'physical'
quiet = torch.zeros(1, model.n_neurons, device='cuda')
graph = np.load(ROOT / cfg['graph'], allow_pickle=True)
ann = pd.read_feather(ROOT / 'data/malecns_v1/body-annotations.feather').set_index('bodyId').reindex(graph['neuron_body_ids'])
cls = ann['class'].astype(str).values
LAYERS = {'olfactory receptors': cls == 'olfactory', 'ALPN': cls == 'ALPN', 'Kenyon': cls == 'Kenyon_Cell', 'MBON': cls == 'MBON'}
LT = {k: torch.as_tensor(v, device='cuda') for k, v in LAYERS.items()}
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
TICKS = int(cfg.get('stream_ticks') or 8)
words = np.array(val[40_000:40_000 + N_WORDS], dtype=np.int64)
nxt = np.array(val[40_000 + 1:40_000 + N_WORDS + 1], dtype=np.int64)

sets = {k: [] for k in LAYERS}
ce_next = []                                    # CE of the read after word t on word t+1
with torch.no_grad():
    s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
    for i, w in enumerate(words):
        token = torch.tensor([int(w)], device='cuda')
        act = torch.zeros(model.n_neurons, dtype=torch.bool, device='cuda')
        for t in range(TICKS):
            if t == 0 or HOLD:
                dr, baseline = fl.write_input(model, s, token)
                s = fl.step_fly_physical_tick(model, s, token, dr, baseline, opts, base_rates=opts['base_rates'])
            else:
                baseline = writer.lambda_adapt * s.baseline if physical_clock else s.baseline
                s = fl.step_fly_physical_tick(model, s, None, quiet, baseline, opts, base_rates=opts['base_rates'])
            act |= s.ring[0][0] > 0
        z = fl.read_fly_motor(model, s)
        ce_next.append(float(F.cross_entropy(model.decoder(model.read_norm(z)).float(),
                                             torch.tensor([int(nxt[i])], device='cuda'))))
        s = fl.expect_next(model, s, z)
        for k, m in LT.items():
            sets[k].append(act[m].cpu().numpy())
del s
torch.cuda.empty_cache()
A = {k: np.stack(v).astype(np.float32) for k, v in sets.items()}
BURN = 32
T = len(words)
ce = np.full(T, np.nan); ce[1:] = np.array(ce_next[:-1])     # brain CE on w_t

# unigram and distributional semantics from the training text
train = np.load(ROOT / cfg['data'] / 'train.npy', mmap_mode='r')
text = np.array(train[:20_000_000], dtype=np.int64)
counts = np.bincount(text, minlength=50257).astype(np.float64) + 1.0
rarity_all = -np.log(counts / counts.sum())
rarity = rarity_all[words]
V = 4096
top = np.argsort(-counts)[:V]
rank = np.full(50257, -1, dtype=np.int64); rank[top] = np.arange(V)
r = rank[text]
co = np.zeros(V * V, dtype=np.float64)
for off in (1, 2):
    a, b = r[:-off], r[off:]
    ok = (a >= 0) & (b >= 0)
    co += np.bincount(a[ok] * V + b[ok], minlength=V * V)
del text, r
C = torch.tensor(co.reshape(V, V), device='cuda', dtype=torch.float32)
C = C + C.T
tot = C.sum()
pmi = torch.log((C * tot).clamp_min(1e-12) / (C.sum(1, keepdim=True) * C.sum(0, keepdim=True)).clamp_min(1e-12))
ppmi = torch.where(C > 0, pmi.clamp_min(0), torch.zeros_like(pmi))
U, S, _ = torch.linalg.svd(ppmi)
emb = (U[:, :128] * S[:128].sqrt()).cpu().numpy()
del C, pmi, ppmi, U, S
torch.cuda.empty_cache()
wvec = np.zeros((T, 128), dtype=np.float32)
for t in range(T):
    if rank[words[t]] >= 0:
        wvec[t] = emb[rank[words[t]]] / (np.linalg.norm(emb[rank[words[t]]]) + 1e-9)
ctx = np.zeros_like(wvec)
for t in range(2, T):
    ctx[t] = wvec[t] + 0.5 * wvec[t - 1] + 0.25 * wvec[t - 2]
ctx /= np.linalg.norm(ctx, axis=1, keepdims=True) + 1e-9


def q(v):
    v = np.asarray(v, dtype=np.float64); v = v[np.isfinite(v)]
    return [round(float(x), 4) for x in (v.min(), np.median(v), v.mean(), v.max())] if v.size else None


def spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return round(float(pd.Series(a[m]).rank().corr(pd.Series(b[m]).rank())), 4)


rep = {'words': T - BURN}
# --- P2 ---
K = A['Kenyon']
kn = np.linalg.norm(K, axis=1)
MEM = 256
novelty = np.full(T, np.nan)
for t in range(BURN, T):
    if kn[t] == 0:
        continue
    past = K[max(0, t - MEM):t]
    pn = np.linalg.norm(past, axis=1)
    ok = pn > 0
    if ok.any():
        novelty[t] = 1.0 - float(((past[ok] @ K[t]) / (kn[t] * pn[ok])).max())
X = np.concatenate([A['Kenyon'], A['ALPN']], 1)
half = BURN + (T - BURN) // 2
tr = np.arange(BURN, half); te = np.arange(half, T)
Xtr, Ytr = torch.tensor(X[tr - 1], device='cuda'), torch.tensor(K[tr], device='cuda')
mu_x, mu_y = Xtr.mean(0), Ytr.mean(0)
Xc, Yc = Xtr - mu_x, Ytr - mu_y
lam = 0.1 * float((Xc * Xc).sum(1).mean())
alpha = torch.linalg.solve(Xc @ Xc.T + lam * torch.eye(len(tr), device='cuda'), Yc)
pred = ((torch.tensor(X[te - 1], device='cuda') - mu_x) @ (Xc.T @ alpha) + mu_y).cpu().numpy()
pred_err = np.full(T, np.nan)
for j, t in enumerate(te):
    if kn[t] > 0:
        pred_err[t] = 1.0 - float(pred[j] @ K[t] / (np.linalg.norm(pred[j]) * kn[t] + 1e-9))
mean_err = np.nanmean([1.0 - float(mu_y.cpu().numpy() @ K[t] / (np.linalg.norm(mu_y.cpu().numpy()) * kn[t] + 1e-9))
                       for t in te if kn[t] > 0])
sl = np.arange(BURN, T)
fit = np.polyfit(rarity[sl][np.isfinite(ce[sl])], ce[sl][np.isfinite(ce[sl])], 1)
resid = ce - np.polyval(fit, rarity)
rep['P2'] = {
    'kenyon_silent_words': round(float((kn[sl] == 0).mean()), 4),
    'brain_ce': q(ce[sl]), 'novelty': q(novelty[sl]), 'prediction_error_test_half': q(pred_err[te]),
    'prediction_error_of_mean_pattern': round(float(mean_err), 4),
    'rho_ce_rarity': spearman(ce[sl], rarity[sl]),
    'rho_novelty_rarity': spearman(novelty[sl], rarity[sl]),
    'rho_novelty_ce': spearman(novelty[sl], ce[sl]),
    'rho_novelty_ce_residual': spearman(novelty[sl], resid[sl]),
    'rho_error_rarity': spearman(pred_err[te], rarity[te]),
    'rho_error_ce': spearman(pred_err[te], ce[te]),
    'rho_error_ce_residual': spearman(pred_err[te], resid[te]),
    'rho_novelty_error': spearman(novelty[te], pred_err[te]),
}
print('P2', json.dumps(rep['P2']), flush=True)

# --- P3 ---
rng = np.random.default_rng(0)
anchors = rng.choice(np.arange(BURN + 2, T), size=min(600, T - BURN - 2), replace=False)
last3 = [set(int(x) for x in words[t - 2:t + 1]) for t in range(T)]
nb_pairs, rnd_pairs, nb_cos, rnd_cos = [], [], [], []
idx_all = np.arange(BURN + 2, T)
for a in anchors:
    far = idx_all[np.abs(idx_all - a) >= 16]
    far = far[[not (last3[a] & last3[j]) for j in far]]
    if far.size == 0:
        continue
    c = ctx[far] @ ctx[a]
    j = int(far[np.argmax(c)]); jr = int(rng.choice(far))
    nb_pairs.append((int(a), j)); rnd_pairs.append((int(a), jr))
    nb_cos.append(float(c.max())); rnd_cos.append(float(ctx[jr] @ ctx[a]))


def overlap(k, pairs):
    out = []
    for a, b in pairs:
        x, y = A[k][a], A[k][b]
        if x.sum() > 0 and y.sum() > 0:
            out.append(float((x * y).sum() / np.sqrt(x.sum() * y.sum())))
    return np.array(out)


rep['P3'] = {'pairs': len(nb_pairs), 'semantic_cosine_neighbour': q(nb_cos), 'semantic_cosine_random': q(rnd_cos)}
for k in LAYERS:
    nb, rd = overlap(k, nb_pairs), overlap(k, rnd_pairs)
    rep['P3'][k] = {'neighbour': q(nb), 'random': q(rd),
                    'neighbour_minus_random': round(float(nb.mean() - rd.mean()), 4) if nb.size and rd.size else None,
                    'ratio': round(float(nb.mean() / max(rd.mean(), 1e-9)), 3) if nb.size and rd.size else None}
print('P3', json.dumps(rep['P3']), flush=True)
print(json.dumps(rep), flush=True)
if len(sys.argv) > 3:
    Path(sys.argv[3]).write_text(json.dumps(rep, indent=1), encoding='utf-8')
