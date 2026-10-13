"""Before training the dopamine-precision arm (GOAL section 'D3b'): does the brain's own dopamine tone fluctuate enough
to matter?  Zero-shot, fixed weights, world-paced reading (pulse input, 8 ticks per word, eyes open), the physical
state carried as in life.  Per tick: the population mean transmitted spike of the 392 dopamine cells d, low-passed on
the brain's own slow dopamine clock (sigmoid(dopamine_logit_slow)); gamma = d / mean(d).  Reports gamma's distribution
[min | median | mean | max], its CV over words and over 512-tick blocks, its autocorrelation time; and -- from a
checkpoint with a trained motor readout (D1) -- the spread of the learned commit value v = W m_std + b over ticks: the
reviewer's criterion, gamma only reshapes behaviour if sd(v) is not small (> ~0.5) AND gamma's CV is not small
(> ~0.2).
Usage: python gamma_tone_probe.py <checkpoint dir> [out.json] [words]
"""
import sys, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork = Path(sys.argv[1])
out = Path(sys.argv[2]) if len(sys.argv) > 2 else fork / 'gamma_tone.json'
WORDS = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
model, cfg, L = load(fork)
import os
if os.environ.get('INTRINSIC_KAPPA') and getattr(model, 'intrinsic_kappa', None) is None:
    # zero-shot: release the dopamine cells' own tonic activity (attach_intrinsic_drive, the fly's spontaneously
    # active types; here the dopaminergic PAM / PPL only), kappa in units of their rheobase
    fl.attach_intrinsic_drive(model, ROOT / cfg['graph'], ROOT / 'data/malecns_v1/body-annotations.feather',
                              kappa=float(os.environ['INTRINSIC_KAPPA']), pattern=r'^(PAM\d|PPL\d)')
    print('intrinsic drive on PAM / PPL, kappa', os.environ['INTRINSIC_KAPPA'], flush=True)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
writer = model.topographic_writer
physical_clock = cfg.get('writer_baseline_clock', 'physical') == 'physical'
quiet = torch.zeros(1, model.n_neurons, device='cuda')
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
seq = np.array(val[50_000:50_000 + WORDS], dtype=np.int64)
K = 8
rho = float(torch.sigmoid(model.dopamine_logit_slow)) if getattr(model, 'dopamine_logit_slow', None) is not None else 0.998
dan = model.dan_indices
motor = getattr(model, 'ai_w', None) is not None and getattr(model, 'ai_mu', None) is not None
lam = float(model.ai_lambda) if motor else None
rate, value = [], []
s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
m = torch.zeros(1, model.read_indices.numel(), device='cuda')
with torch.no_grad():
    for i, w in enumerate(seq):
        token = torch.tensor([int(w)], device='cuda')
        for t in range(K):
            if t == 0:
                drive, baseline = fl.write_input(model, s, token)
                s = fl.step_fly_physical_tick(model, s, token, drive, baseline, opts, base_rates=opts['base_rates'])
            else:
                baseline = writer.lambda_adapt * s.baseline if physical_clock else s.baseline
                s = fl.step_fly_physical_tick(model, s, None, quiet, baseline, opts, base_rates=opts['base_rates'])
            rate.append(float(s.ring[0].index_select(1, dan).mean()))
            if motor:
                m = lam * m + (1 - lam) * s.ring[0].index_select(1, model.read_indices)
                v = ((m - model.ai_mu) / model.ai_sigma) @ model.ai_w[1] + model.ai_b[1]
                value.append(float(v))
        s = fl.expect_next(model, s, fl.read_fly_motor(model, s))
rate = np.array(rate)
d = np.zeros_like(rate); acc = rate[:512].mean()                 # start the slow trace at its early mean
for k, r in enumerate(rate):
    acc = rho * acc + (1 - rho) * r; d[k] = acc
burn = min(len(d) // 4, 2048)
d = d[burn:]
gamma = d / d.mean()
words = gamma[K - 1::K]
blocks = gamma[:len(gamma) // 512 * 512].reshape(-1, 512).mean(1)
x = gamma - gamma.mean()
ac = np.correlate(x[:8192], x[:8192], 'full')[8191:] / (x[:8192] ** 2).sum()
tau = int(np.argmax(ac < np.exp(-1))) if (ac < np.exp(-1)).any() else len(ac)
q = lambda v: [float(np.min(v)), float(np.median(v)), float(np.mean(v)), float(np.max(v))]
rep = {'fork': str(fork), 'words': WORDS, 'ticks_per_word': K, 'slow_clock_rho': rho, 'tau_slow_ticks': 1 / (1 - rho),
       'da_rate_raw_min_median_mean_max': q(rate), 'da_rate_raw_cv': float(rate.std() / max(rate.mean(), 1e-12)),
       'gamma_min_median_mean_max': q(words), 'gamma_cv_words': float(words.std()), 'gamma_cv_512_blocks': float(blocks.std()),
       'gamma_autocorr_ticks': tau}
if motor:
    v = np.array(value)[burn:]
    rep['commit_value_min_median_mean_max'] = q(v)
    rep['commit_value_sd'] = float(v.std())
    # how much gamma would move the commit hazard: sd of sigmoid(habit + gamma v) - sigmoid(habit + v) over ticks
    habit = float(np.log(float(cfg.get('act_habit') or 0.2) / (1 - float(cfg.get('act_habit') or 0.2))))
    sig = lambda z: 1 / (1 + np.exp(-z))
    vv = v - habit                                                # D1's b holds habit + value bias: value = v - ln E
    rep['hazard_change_by_gamma_sd'] = float(np.std(sig(habit + gamma[:len(vv)] * vv) - sig(habit + vv)))
print(json.dumps(rep, indent=1))
json.dump(rep, open(out, 'w'), indent=1)
