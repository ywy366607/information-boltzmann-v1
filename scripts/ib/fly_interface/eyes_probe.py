"""M1 probe (GOAL: docs/information_boltzmann/FLY_ACTIVE_INFERENCE_GOAL_20261013.md): the eyes in a world-paced stream.
Fixed weights, the checkpoint's own physical state and act integrator, the same validation text for every condition
(256 words burn-in, then 1,024 scored), each word's interval K ~ Geom(1/mu) with the same seed per mu.

Eye modes
  own     the act head decides the eye gain before each word is written (closed eyes miss the word)
  open    eyes forced open (gain 1): the same weights, so the difference is the eyes' own effect on language

Reported per (mu, mode): eye gain at word onsets [min | median | mean | max], open fraction (gain > 0.5), arrival CE and
survival-weighted CE [min | median | mean | max] (per-word values kept for paired comparisons); for 'own': Spearman correlations of the eye gain with the difficulty of
the word about to be written (its arrival CE, known before it is written), with its lifetime novelty
1 / sqrt(1 + n_seen), and the mean gain in the hardest vs easiest quartile.
Usage: python eyes_probe.py <checkpoint dir> [out .jsonl]
"""
import sys, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from scipy.stats import spearmanr
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])
from information_boltzmann.core.fly_active_inference import ai_eye, ai_step
import os
BURN, SCORE = int(os.environ.get('EYES_BURN', BURN)), int(os.environ.get('EYES_SCORE', SCORE))
MUS = tuple(int(x) for x in os.environ.get('EYES_MUS', '16,8,4,2').split(','))

fork = Path(sys.argv[1])
out = Path(sys.argv[2] if len(sys.argv) > 2 else fork / 'eyes_probe.jsonl')
model, cfg, L = load(fork)
assert getattr(model, 'ai_w', None) is not None, 'no act head in this checkpoint'
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
N = model.n_neurons
writer = model.topographic_writer
physical_clock = cfg.get('writer_baseline_clock', 'physical') == 'physical'
quiet = torch.zeros(1, N, device='cuda')
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
region = 60_000
targets = np.array(val[region:region + BURN + SCORE], dtype=np.int64)
inputs = np.concatenate(([int(val[region - 1])], targets[:-1]))
fam = L.get('familiarity_n')
fam = np.zeros(50257) if fam is None else np.asarray(fam)
g0 = L.get('ai_g')
g0 = torch.tensor([[np.log(0.95 / 0.05), 0.0]]) if g0 is None else g0
d = lambda x: [float(np.min(x)), float(np.median(x)), float(np.mean(x)), float(np.max(x))]


def run(mu, mode):
    rng = np.random.default_rng(1000 + mu)
    s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
    g = g0.clone().float().cuda()
    eyes, arrival, survival, prev_arrival = [], [], [], []
    last_ce = None                       # the arrival CE of the word about to be written (scored when it arrived)
    with torch.no_grad():
        for i, (tok, tgt) in enumerate(zip(inputs, targets)):
            score = i >= BURN
            token = torch.tensor([int(tok)], device='cuda'); target = torch.tensor([int(tgt)], device='cuda')
            K = int(rng.geometric(1.0 / mu))
            ces = []
            for t in range(1, K + 1):
                if t == 1:
                    e = ai_eye(model, g, 1.0 if mode == 'open' else None)
                    drive, baseline = fl.write_input(model, s, token)
                    # as in training: the surface receives e * the word (drive and adaptation); e = 0 is a quiet tick
                    quiet_baseline = writer.lambda_adapt * s.baseline if physical_clock else s.baseline
                    baseline = e * baseline + (1.0 - e) * quiet_baseline
                    s = fl.step_fly_physical_tick(model, s, token, drive * e, baseline, opts, base_rates=opts['base_rates'])
                    if score:
                        eyes.append(float(e)); prev_arrival.append(last_ce)
                else:
                    baseline = writer.lambda_adapt * s.baseline if (writer is not None and physical_clock) else s.baseline
                    s = fl.step_fly_physical_tick(model, s, None, quiet, baseline, opts, base_rates=opts['base_rates'])
                _, g, _ = ai_step(model, s.h, g)
                lat = fl.read_fly_motor(model, s)
                ces.append(float(F.cross_entropy(model.decoder(model.read_norm(lat)).float(), target)))
            s = fl.expect_next(model, s, lat)
            last_ce = ces[-1]
            if score:
                arrival.append(ces[-1]); survival.append(sum(ces) / mu)
    eyes, arrival = np.array(eyes), np.array(arrival)
    rep = {'checkpoint': str(fork), 'mu': mu, 'mode': mode, 'eye_min_median_mean_max': d(eyes),
           'open_frac': float((eyes > 0.5).mean()), 'arrival_ce_min_median_mean_max': d(arrival),
           'survival_ce_min_median_mean_max': d(survival),
           'per_word_arrival': arrival.tolist(), 'per_word_survival': list(survival), 'per_word_eye': eyes.tolist()}
    if mode == 'own' and eyes.std() > 0:
        # the word about to be written: how hard it was to predict (its arrival CE) and how new it is in this life
        written = inputs[BURN:BURN + SCORE]
        hard = np.array([np.nan if c is None else c for c in prev_arrival])
        ok = ~np.isnan(hard)
        nov = 1.0 / np.sqrt(1.0 + fam[written])
        q1, q3 = np.quantile(hard[ok], [0.25, 0.75])
        rep.update({'rho_eye_written_word_ce': float(spearmanr(eyes[ok], hard[ok])[0]),
                    'rho_eye_novelty': float(spearmanr(eyes, nov)[0]),
                    'rho_eye_next_arrival_ce': float(spearmanr(eyes, arrival)[0]),
                    'eye_mean_hardest_quartile': float(eyes[ok][hard[ok] >= q3].mean()),
                    'eye_mean_easiest_quartile': float(eyes[ok][hard[ok] <= q1].mean())})
    print(json.dumps({k: v for k, v in rep.items() if k != 'checkpoint' and not k.startswith('per_word')}), flush=True)
    with open(out, 'a') as f:
        f.write(json.dumps(rep) + '\n')


# both modes on every arm: on the forced-open arm the head was never trained, so its 'own' run is the fresh-head
# baseline for the correlations (the eye and the CE are read from the same membranes)
modes = ('own', 'open')
for mu in MUS:
    for mode in modes:
        run(mu, mode)
