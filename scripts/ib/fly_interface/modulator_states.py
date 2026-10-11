"""Steps 2-3 measurements on a fork with neuromodulatory precision (fixed weights).

1. Inert or not: NLL change in each world when (a) precision is neutralised (gains = 1) and (b) the neuromodulatory cells
   are silenced (their outgoing edges and intrinsic current off; gains then also see no modulator activity).
2. Tracking surprise: during normal reading, Spearman between the token's prediction error and the dopamine level /
   attention gain right after it.
3. Emotion-like states: normal text, then a 'chaos' stretch of L shuffled words (L = 8, 16, 32: a run of prediction
   errors), then normal text again.  Time courses (per word) of dopamine, mean attention gain and thinking ticks,
   relative to the pre-chaos baseline: persistence (words until back within 1 sd) and grading with L.
"""
import sys, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from scipy.stats import spearmanr
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scratch/fly_stream/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork = Path(sys.argv[1])
model, cfg, L = load(fork)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
max_ticks = int(L.get('go_max_ticks', 32))
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
start = lambda: move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
has_precision = getattr(model, 'precision_u', None) is not None


def gains(state):
    if not has_precision or state.modulator.numel() == 0:
        return np.ones(3), 0.0
    m = state.modulator
    g = (2 * torch.sigmoid(m @ model.precision_u.T + model.precision_b))[0]
    return g.cpu().numpy(), float(m[0, model.dopamine_mask].mean())


def read(tokens, nexts, s, gap_every=0, gap=0, trace=False):
    torch.manual_seed(0)
    out, rows = [], []
    with torch.no_grad():
        for i, tok in enumerate(tokens):
            g = gap if (gap_every and i % gap_every == 0) else 0
            drive, _ = fl.write_input(model, s, torch.tensor([int(tok)], device='cuda'))
            innovation = float(drive[0, model.topographic_writer.injection_index].norm())
            s, lat, _, _, _, after = fl.advance_fly_token_go(model, s, torch.tensor([int(tok)], device='cuda'), [],
                                                              max_ticks=max_ticks, writer_baseline_clock='physical',
                                                              silence_after=g, **opts)
            z = after if g else lat[-1:]
            ce = float(F.cross_entropy(model.decoder(model.read_norm(z)).float(), torch.tensor([int(nexts[i])], device='cuda')))
            out.append(ce)
            if trace:
                gn, da = gains(s)
                rows.append((ce, da, float(gn.mean()), lat.shape[0], innovation))
    return np.array(out), rows, s


def listen(tokens, nexts, s, k):
    out = []
    with torch.no_grad():
        for i, tok in enumerate(tokens):
            s, z = fl.advance_fly_token_stream(model, s, torch.tensor([int(tok)], device='cuda'), ticks=k,
                                               writer_baseline_clock='physical', **opts)
            out.append(float(F.cross_entropy(model.decoder(model.read_norm(z)).float(), torch.tensor([int(nexts[i])], device='cuda'))))
    return np.array(out)


region = 60_000
BURN, SCORE = 256, 768
targets = np.array(val[region:region + BURN + SCORE], dtype=np.int64)
inputs = np.concatenate(([int(val[region - 1])], targets[:-1]))


def worlds():
    r, _, _ = read(inputs, targets, start())
    li = listen(inputs, targets, start(), 4)
    gp, _, _ = read(inputs, targets, start(), gap_every=4, gap=16)
    sl = slice(BURN, None)
    return {'read': float(r[sl].mean()), 'listen_k4': float(li[sl].mean()),
            'gap16_after': float(gp[sl][np.arange(SCORE) % 4 == 0].mean())}


report = {'fork': str(fork), 'precision': has_precision}
if '--only-episodes' in sys.argv or '--only-tracking' in sys.argv:
    report['worlds_full'] = {}
else:
    report['worlds_full'] = worlds()
if report['worlds_full']: print('worlds (full):', {k: round(v, 4) for k, v in report['worlds_full'].items()}, flush=True)
if has_precision and '--only-episodes' not in sys.argv and '--only-tracking' not in sys.argv:
    saved_u, saved_b = model.precision_u.detach().clone(), model.precision_b.detach().clone()
    model.precision_u.data.zero_(); model.precision_b.data.zero_()
    report['worlds_neutral_precision'] = worlds()
    model.precision_u.data.copy_(saved_u); model.precision_b.data.copy_(saved_b)
    # silence the neuromodulatory cells: no outgoing transmission, no intrinsic current, no concentration
    mod = model.modulatory_index
    backup = {}
    for name, pre_name in (('edge_weight_e', 'edge_pre_e'), ('edge_weight_i', 'edge_pre_i')):
        w = getattr(model, name)
        backup[name] = w.detach().clone()
        silent = torch.zeros(model.n_neurons, dtype=torch.bool, device='cuda'); silent[mod] = True
        w.data.mul_((~silent[getattr(model, pre_name).long()]).float())
    kap = None
    if getattr(model, 'intrinsic_kappa', None) is not None:
        kap = model.intrinsic_kappa.detach().clone()
        is_mod = torch.isin(model.intrinsic_index, mod)
        model.intrinsic_kappa.data[is_mod] = 0.0
    th = model.get_thresholds().clone()
    th[0, mod] = 1e6                                  # they cannot fire, so the concentration decays to zero
    opts['thresholds'] = th
    report['worlds_modulators_silenced'] = worlds()
    opts['thresholds'] = model.get_thresholds()
    for name, w in backup.items():
        getattr(model, name).data.copy_(w)
    if kap is not None:
        model.intrinsic_kappa.data.copy_(kap)
    for k in ('worlds_neutral_precision', 'worlds_modulators_silenced'):
        print(k, {w: round(report[k][w] - report['worlds_full'][w], 4) for w in report['worlds_full']}, flush=True)

# 2. tracking surprise during normal reading
_, rows, _ = read(inputs, targets, start(), trace=True)
a = np.array(rows[BURN:])
report['surprise_tracking'] = {'spearman_error_vs_dopamine_after': float(spearmanr(a[:, 0], a[:, 1]).correlation),
                               'spearman_error_vs_gain_after': float(spearmanr(a[:, 0], a[:, 2]).correlation),
                               'spearman_error_vs_ticks': float(spearmanr(a[:, 0], a[:, 3]).correlation)} if has_precision else {
                               'spearman_error_vs_ticks': float(spearmanr(a[:, 0], a[:, 3]).correlation)}
# is surprise a neural signal?  the innovation written for word i vs the error of the prediction it met (row i-1)
report['surprise_tracking']['spearman_prediction_error_vs_written_innovation'] = float(
    spearmanr(a[:-1, 0], a[1:, 4]).correlation)
report['surprise_tracking']['spearman_prediction_error_vs_dopamine_next_word'] = float(
    spearmanr(a[:-1, 0], a[1:, 1]).correlation) if has_precision else None
print('surprise tracking:', report['surprise_tracking'], flush=True)

if '--only-tracking' in sys.argv:
    json.dump(report, open(fork / 'modulator_tracking.json', 'w'), indent=1)
    sys.exit()
# 3. chaos episodes
rng = np.random.default_rng(7)
report['episodes'] = {}
for Lc in (8, 16, 32):
    pre, post = 160, 96
    base = np.array(val[region + 5000:region + 5000 + pre + Lc + post + 1], dtype=np.int64)
    seq = base.copy()
    chaos = (rng.integers(0, 50257, Lc) if '--only-episodes' in sys.argv
             else rng.permutation(np.array(val[region + 20000:region + 20000 + 4000], dtype=np.int64))[:Lc])
    seq[pre:pre + Lc] = chaos
    _, rows, _ = read(seq[:-1], seq[1:], start(), trace=True)
    a = np.array(rows)
    ref = a[64:pre]
    mu, sd = ref.mean(0), ref.std(0) + 1e-9
    z = (a - mu) / sd
    after = z[pre + Lc:]
    persist = {}
    for j, name in ((0, 'error'), (1, 'dopamine'), (2, 'gain'), (3, 'ticks'), (4, 'innovation')):
        inside = float(z[pre:pre + Lc, j].mean())
        back = next((t for t in range(len(after)) if abs(after[t:t + 4, j].mean()) < 1.0), len(after))
        persist[name] = {'z_during_chaos': inside, 'z_first_8_after': float(after[:8, j].mean()),
                         'words_until_back': int(back)}
    report['episodes'][Lc] = persist
    print(f'chaos L={Lc}:', {k: {kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                              for k, v in persist.items()}, flush=True)
json.dump(report, open(fork / ('modulator_states_uniform_chaos.json' if '--only-episodes' in sys.argv else 'modulator_states.json'), 'w'), indent=1)
