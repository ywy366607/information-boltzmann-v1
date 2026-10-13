"""Instead of training D2b: is there an STP setting under which this brain accumulates evidence?  If no change of the
release dynamics makes a noisy word's prediction improve with ticks, evolving STP (D2b) has no direction to find.
Zero-shot, fixed weights.  World: sustained noisy input as in D2 (the word written every tick as a noisy sample,
sigma ~ U(0, 2) x the embedding norm per dimension), 8 ticks per word, 384 scored words after 64 burn-in, the
physical state carried.  Per setting: CE of the next word at ticks 1..8, for clear (sigma < 0.5) and noisy (sigma >
1.5) words; accumulation = CE(tick 1) - CE(tick 8) on noisy words.  Settings scale every class's STP parameters:
baseline; tau_rec x4 / x0.25; u0 0.10 (facilitating) / 0.50 (depressing); tau_fac x4.
Usage: python stp_sweep_probe.py <checkpoint dir> [out.json]
"""
import sys, json, math, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork = Path(sys.argv[1])
out = Path(sys.argv[2]) if len(sys.argv) > 2 else fork / 'stp_sweep.json'
model, cfg, L = load(fork)
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
BURN, SCORE, K, SMAX = 64, 384, 8, 2.0
seq = np.array(val[90_000:90_000 + BURN + SCORE + 1], dtype=np.int64)
base = {n: getattr(model, n).detach().clone() for n in ('logit_u0', 'log_tau_rec', 'log_tau_fac')}
logit = lambda p: math.log(p / (1 - p))
settings = {'baseline': {}, 'tau_rec_x4': {'log_tau_rec': math.log(4)}, 'tau_rec_x0.25': {'log_tau_rec': math.log(0.25)},
            'u0_0.10': {'logit_u0': ('set', logit(0.10))}, 'u0_0.50': {'logit_u0': ('set', logit(0.50))},
            'tau_fac_x4': {'log_tau_fac': math.log(4)}}
writer = model.topographic_writer
q = lambda v: [float(np.min(v)), float(np.median(v)), float(np.mean(v)), float(np.max(v))]
rep = {'fork': str(fork), 'ticks': K, 'sigma_max': SMAX}
for name, change in settings.items():
    with torch.no_grad():
        for n, v in base.items():
            getattr(model, n).copy_(v)
        for n, c in change.items():
            p = getattr(model, n)
            if isinstance(c, tuple):
                p.fill_(c[1])
            else:
                p.add_(c)
    opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
                alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
    rng = np.random.default_rng(5)
    s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
    curves, sig = [], []
    with torch.no_grad():
        for i in range(len(seq) - 1):
            token = torch.tensor([int(seq[i])], device='cuda'); target = torch.tensor([int(seq[i + 1])], device='cuda')
            emb = model.embedding(token)
            sigma = SMAX * rng.random()
            ces = []
            for t in range(K):
                noisy = emb + torch.as_tensor(rng.standard_normal(emb.shape), dtype=emb.dtype, device='cuda') \
                    * (sigma * float(emb.norm()) / math.sqrt(emb.shape[-1]))
                drive, baseline = fl.write_input(model, s, token, noisy)
                s = fl.step_fly_physical_tick(model, s, token, drive, baseline, opts, base_rates=opts['base_rates'])
                lat = fl.read_fly_motor(model, s)
                ces.append(float(F.cross_entropy(model.decoder(model.read_norm(lat)).float(), target)))
            s = fl.expect_next(model, s, lat)
            if i >= BURN:
                curves.append(ces); sig.append(sigma)
    C, sig = np.array(curves), np.array(sig)
    noisy, clear = sig > 1.5, sig < 0.5
    r = {'ce_by_tick_noisy': [round(float(x), 3) for x in C[noisy].mean(0)],
         'ce_by_tick_clear': [round(float(x), 3) for x in C[clear].mean(0)],
         'accumulation_noisy': q(C[noisy, 0] - C[noisy, -1]), 'mean_ce': float(C.mean())}
    rep[name] = r
    print(name, 'noisy', r['ce_by_tick_noisy'], 'clear', r['ce_by_tick_clear'],
          'accum mean', round(r['accumulation_noisy'][2], 3), 'mean CE', round(r['mean_ce'], 3), flush=True)
json.dump(rep, open(out, 'w'), indent=1)
