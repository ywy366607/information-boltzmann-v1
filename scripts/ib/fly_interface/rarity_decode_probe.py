"""Can the acts see what makes a word worth thinking about?  Fixed weights, the checkpoint's own state, pulse input,
8 ticks per word, 256 burn-in + 1,024 scored validation words.  Features at a word's tick 1 and tick 4:
  muscles      the output cells' transmitted spikes through the muscle filter (tau 8 ticks) -- what the acts read
  out_membrane the output cells' membranes (what the language read uses)
  core         membranes of 3,000 random non-output cells
  kc           the Kenyon cells' membranes, kc_spk their spikes summed since the word's onset (with compartment
               memory: does the text reach the mushroom body in absolute terms, not just as a share of the mass?)
Targets: rarity of the current word and of the next word (unigram CE, the listener's prior), the read's CE of the next
word at tick 1 (difficulty as it is), and the oracle value of waiting (tick-1 CE minus the best CE over ticks 1-8).
Ridge regression, 5 contiguous folds, ridge strength picked per target on the folds (slightly optimistic); reported as
cross-validated R^2.  Usage: python rarity_decode_probe.py <checkpoint dir> [out.json]
"""
import sys, json, math, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork = Path(sys.argv[1])
model, cfg, L = load(fork)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
writer = model.topographic_writer
quiet = torch.zeros(1, model.n_neurons, device='cuda')
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
train = np.load(ROOT / cfg['data'] / 'train.npy', mmap_mode='r')
counts = np.bincount(np.asarray(train[:20_000_000], dtype=np.int64), minlength=50257).astype(np.float64) + 0.5
unigram = -np.log(counts / counts.sum())
seq = np.array(val[60_000 - 1:60_000 + BURN + SCORE + 1], dtype=np.int64)
idx = model.read_indices
rng = np.random.default_rng(3)
others = np.setdiff1d(np.arange(model.n_neurons), idx.cpu().numpy())
core = torch.as_tensor(np.sort(rng.choice(others, 3000, replace=False)), device='cuda')
lam = math.exp(-1 / 8)
s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
m = torch.zeros(1, idx.numel(), device='cuda')
KC = getattr(model, 'comp_kc', None)
feats = {f'{k}@{t}': [] for k in ('muscles', 'out_membrane', 'core') + (('kc', 'kc_spk') if KC is not None else ()) for t in (1, 4)}
kc_active = []
ce_curves = []
with torch.no_grad():
    for i in range(len(seq) - 1):
        token = torch.tensor([int(seq[i])], device='cuda'); target = torch.tensor([int(seq[i + 1])], device='cuda')
        ces = []
        spk = None
        for t in range(1, 9):
            if t == 1:
                drive, baseline = fl.write_input(model, s, token)
                s = fl.step_fly_physical_tick(model, s, token, drive, baseline, opts, base_rates=opts['base_rates'])
            else:
                s = fl.step_fly_physical_tick(model, s, None, quiet, writer.lambda_adapt * s.baseline, opts, base_rates=opts['base_rates'])
            m = lam * m + (1 - lam) * s.ring[0].index_select(1, idx)
            if KC is not None:
                now = (s.ring[0].index_select(1, KC) > 0).float()
                spk = now if spk is None else spk + now
            lat = fl.read_fly_motor(model, s)
            ces.append(float(F.cross_entropy(model.decoder(model.read_norm(lat)).float(), target)))
            if i >= BURN and t in (1, 4):
                feats[f'muscles@{t}'].append(m[0].cpu().numpy())
                feats[f'out_membrane@{t}'].append(s.h[0, idx].cpu().numpy())
                feats[f'core@{t}'].append(s.h[0, core].cpu().numpy())
                if KC is not None:
                    feats[f'kc@{t}'].append(s.h[0, KC].cpu().numpy())
                    feats[f'kc_spk@{t}'].append(spk[0].cpu().numpy())
        if KC is not None and i >= BURN:
            kc_active.append(float((spk > 0).float().mean()))
        s = fl.expect_next(model, s, lat)
        if i >= BURN:
            ce_curves.append(ces)
C = np.array(ce_curves)
words = seq[BURN:BURN + len(C)]; nxt = seq[BURN + 1:BURN + 1 + len(C)]
targets = {'current_rarity': unigram[words], 'next_rarity': unigram[nxt], 'difficulty_ce1': C[:, 0],
           'wait_value': C[:, 0] - C.min(1)}


def cv_r2(X, y, folds=5):
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    keep = X.std(0) > 0
    X = X[:, keep]
    n = len(y); edges = np.linspace(0, n, folds + 1).astype(int)
    best = -np.inf
    for alpha in (1.0, 10.0, 100.0, 1000.0, 10000.0):
        pred = np.zeros(n)
        for f in range(folds):
            te = np.arange(edges[f], edges[f + 1]); tr = np.setdiff1d(np.arange(n), te)
            Xt, yt = X[tr], y[tr] - y[tr].mean()
            # dual ridge (n < p): w = X^T (X X^T + a I)^-1 y
            K = Xt @ Xt.T
            coef = np.linalg.solve(K + alpha * np.eye(len(tr)), yt)
            pred[te] = X[te] @ (Xt.T @ coef) + y[tr].mean()
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        best = max(best, r2)
    return float(best)


rep = {'checkpoint': str(fork), 'words': int(len(C)), 'target_sd': {k: float(v.std()) for k, v in targets.items()},
       'corr_between_targets': {f'{a}~{b}': float(np.corrcoef(targets[a], targets[b])[0, 1])
                                for a in targets for b in targets if a < b},
       'r2': {}}
for fname, rows in feats.items():
    X = np.array(rows, dtype=np.float64)
    rep['r2'][fname] = {t: round(cv_r2(X, y), 3) for t, y in targets.items()}
    print(fname, rep['r2'][fname], flush=True)
print('target correlations', {k: round(v, 3) for k, v in rep['corr_between_targets'].items()})
if kc_active:
    rep['kc_active_frac_per_word_min_median_mean_max'] = [float(np.min(kc_active)), float(np.median(kc_active)), float(np.mean(kc_active)), float(np.max(kc_active))]
    print('Kenyon cells firing at least once per word', rep['kc_active_frac_per_word_min_median_mean_max'])
json.dump(rep, open(Path(sys.argv[2]) if len(sys.argv) > 2 else fork / 'rarity_decode.json', 'w'), indent=1)
