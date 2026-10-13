"""Does text written on the photoreceptors get past the lamina?  (2026-10-13: photoreceptors and lamina monopolar cells
are graded, tonically releasing cells in the fly; as spiking point cells the photoreceptors' histamine only inhibits
silent L1-L3.)  Zero-shot, from the individual's reading state, 4 places x 128 words read as the world does vs the same
state with the words withheld.  Reports per group: output per tick (spikes, or graded release for graded cells) reading
and withheld; the photoreceptors' mean (v - E_I) / theta (the operating point that sets the tonic gain); and how much
of the current word the medulla carries: ridge R2 (5-fold) of the word embedding's top 8 principal components from the
medulla cells' per-word output.  Env as world_eval (ZS_GRADED_ADD='~R1-R6,~R7*,~R8*,~L1,~L2,~L3', ZS_TONIC_GAIN).
Usage: python visual_relay_probe.py <checkpoint dir> [out.json]
"""
import os, sys, json, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork = Path(sys.argv[1])
model, cfg, L = load(fork)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
writer = model.topographic_writer
HOLD = bool(cfg.get('hold_input'))
TICKS = int(cfg.get('stream_ticks') or 8)
graph = np.load(ROOT / cfg['graph'], allow_pickle=True)
ann = pd.read_feather(ROOT / 'data/malecns_v1/body-annotations.feather').set_index('bodyId').reindex(graph['neuron_body_ids'])
typ = ann['type'].astype(str).values.astype(str)
R = np.char.startswith(typ, 'R1-R6') | np.char.startswith(typ, 'R7') | np.char.startswith(typ, 'R8')
LMC = np.isin(typ, ['L1', 'L2', 'L3'])
MED = np.isin(typ, ['Mi1', 'Mi4', 'Mi9', 'Tm1', 'Tm2', 'Tm3', 'Tm4', 'Tm9', 'Tm20', 'Dm8a', 'Dm8b', 'Dm9', 'Mi15', 'Tm5c',
                    'C2', 'C3', 'T1'])
G = {'photoreceptors': R, 'L1-L3': LMC, 'medulla targets': MED}
G = {k: torch.as_tensor(v, device='cuda') for k, v in G.items()}
med_idx = torch.as_tensor(np.flatnonzero(MED), device='cuda')
quiet = torch.zeros(1, model.n_neurons, device='cuda')
E_I = float(model.E_I)
thr = opts['thresholds'].detach().expand(1, model.n_neurons)[0]
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')


def read(s, words, text=True):
    """Per tick: every cell's transmitted output (spikes; graded release for graded cells)."""
    acc = {k: 0.0 for k in G}
    op, n, feats = 0.0, 0, []
    for w in words:
        token = torch.tensor([int(w)], device='cuda')
        f = torch.zeros(len(med_idx), device='cuda')
        for t in range(TICKS):
            if text and (t == 0 or HOLD):
                dr, baseline = fl.write_input(model, s, token)
                s = fl.step_fly_physical_tick(model, s, token, dr, baseline, opts, base_rates=opts['base_rates'])
            else:
                baseline = writer.lambda_adapt * s.baseline
                s = fl.step_fly_physical_tick(model, s, None, quiet, baseline, opts, base_rates=opts['base_rates'])
            o = s.ring[0][0]
            for k, m in G.items():
                acc[k] += float(o[m].mean())
            op += float(((s.h[0] - E_I) / thr)[G['photoreceptors']].mean())
            f += o[med_idx]
            n += 1
        feats.append(f.cpu().numpy())
        if text:
            s = fl.expect_next(model, s, fl.read_fly_motor(model, s))
    return s, {k: v / n for k, v in acc.items()}, op / n, np.array(feats)


def ridge_r2(Xs, Y, lam=100.0, k=5):
    idx = np.arange(len(Xs))
    folds = np.array_split(np.random.default_rng(0).permutation(idx), k)
    ss_res = ss_tot = 0.0
    for f in folds:
        tr = np.setdiff1d(idx, f)
        A, mu = Xs[tr], Y[tr].mean(0)
        beta = A.T @ np.linalg.solve(A @ A.T + lam * np.eye(len(A)), Y[tr] - mu)
        pred = Xs[f] @ beta + mu
        ss_res += ((Y[f] - pred) ** 2).sum()
        ss_tot += ((Y[f] - mu) ** 2).sum()
    return 1 - ss_res / ss_tot


rows, X, words_all = [], [], []
with torch.no_grad():
    for place in range(4):
        s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
        start = 70_000 + place * 3_000
        s, _, _, _ = read(s, np.array(val[start:start + 32], dtype=np.int64))
        words = np.array(val[start + 32:start + 160], dtype=np.int64)
        _, on, op, f = read(s, words, True)
        _, off, _, _ = read(s, words, False)
        rows.append((on, off, op))
        X.append(f)
        words_all.append(words)
X, W = np.concatenate(X), np.concatenate(words_all)
emb = model.embedding.weight.detach().float().cpu().numpy()
Vt = np.linalg.svd(emb[:20000] - emb[:20000].mean(0), full_matrices=False)[2]
Y = (emb[W] - emb.mean(0)) @ Vt[:8].T
Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
rep = {'graded_add': os.environ.get('ZS_GRADED_ADD'), 'tonic_gain': os.environ.get('ZS_TONIC_GAIN')}
for k in G:
    on, off = np.mean([r[0][k] for r in rows]), np.mean([r[1][k] for r in rows])
    rep[k] = {'reading': on, 'withheld': off, 'text_driven': on - off}
    print(f'{k:>16}: output per tick reading {on:.4f}, withheld {off:.4f}, text-driven {on - off:+.4f}', flush=True)
rep['photoreceptor_operating_point'] = float(np.mean([r[2] for r in rows]))
rep['medulla_word_R2_top8pc'] = float(ridge_r2(Xs, Y))
print(f"photoreceptor (v - E_I)/theta while reading: {rep['photoreceptor_operating_point']:.3f}", flush=True)
print(f"current word from the medulla (ridge R2, top 8 PCs, {len(W)} words): {rep['medulla_word_R2_top8pc']:.3f}", flush=True)
if len(sys.argv) > 2:
    json.dump(rep, open(sys.argv[2], 'w'), indent=1)
