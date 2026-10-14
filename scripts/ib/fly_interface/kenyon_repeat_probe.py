"""Does a word's code repeat when the word repeats?  The precondition of per-cell habituation as word familiarity (the
alpha'3 novelty route) and of any Kenyon -> MBON memory of a word.  Reads a stretch of validation text as the trainer
does (the same writer, held input, ticks per word), keeps every word's active set per layer (at least one spike over
its ticks), and compares, per layer:
- same word: the overlap |a & b| / |a| between two occurrences of the same token (successive occurrences);
- different words: the same measure between occurrences of two different tokens the same distance apart (control);
- chance: the layer's coding level.
A repeatable code has same >> different ~ chance.  Distributions over word pairs as [min | median | mean | max].
Usage: python kenyon_repeat_probe.py <checkpoint dir> [words=1500] [out.json]
"""
import sys, json, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork = Path(sys.argv[1])
N_WORDS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
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
sets = {k: [] for k in LAYERS}
with torch.no_grad():
    s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
    for w in words:
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
        s = fl.expect_next(model, s, fl.read_fly_motor(model, s))
        for k, m in LT.items():
            sets[k].append(act[m].cpu().numpy())

BURN = 32
idx = np.arange(BURN, len(words))
last = {}
same_pairs = []                                   # (earlier position, later position) of the same token
for i in idx:
    w = int(words[i])
    if w in last:
        same_pairs.append((last[w], i))
    last[w] = i
rng = np.random.default_rng(0)
diff_pairs = []                                   # different tokens at the same distances
for a, b in same_pairs:
    d = b - a
    for _ in range(20):
        j = int(rng.integers(BURN, len(words) - d))
        if words[j] != words[j + d]:
            diff_pairs.append((j, j + d))
            break


def overlap(k, pairs):
    out = []
    for a, b in pairs:
        A, B = sets[k][a], sets[k][b]
        if A.any():
            out.append(float((A & B).sum() / A.sum()))
    return np.array(out)


def q(v):
    return [round(float(x), 4) for x in (v.min(), np.median(v), v.mean(), v.max())] if v.size else None


report = {'words': int(len(idx)), 'same_word_pairs': len(same_pairs), 'distinct_tokens': int(len(set(words[idx].tolist())))}
for k in LAYERS:
    level = float(np.mean([x.mean() for x in sets[k][BURN:]]))
    same, diff = overlap(k, same_pairs), overlap(k, diff_pairs)
    report[k] = {'level': round(level, 4), 'same_word_min_median_mean_max': q(same),
                 'different_word_min_median_mean_max': q(diff),
                 'same_minus_different': round(float(same.mean() - diff.mean()), 4) if same.size and diff.size else None}
    print(f'{k:>20}: level {level:.3f}; same word {q(same)}; different words {q(diff)}', flush=True)
print(json.dumps(report), flush=True)
if len(sys.argv) > 3:
    Path(sys.argv[3]).write_text(json.dumps(report, indent=1), encoding='utf-8')
