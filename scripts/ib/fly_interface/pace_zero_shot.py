"""Zero-shot: an individual in a world that sets its own pace.  Fixed weights, the checkpoint's own physical state,
the same validation text for every condition (256 words burn-in, then 1,024 scored).

Conditions
  native         the checkpoint's own flux admission (pulse, then 3..14 quiet ticks until the flux settles)
  fixed k        every word takes k ticks (pulse at the first)
  geo mu pulse   each word's interval K ~ Geom(1/mu) >= 1, memoryless; pulse at the first tick, quiet afterwards
  geo mu hold    the same intervals (same seed), the word held on the sensory surface for all its ticks

The motor read is taken after every tick, so every tick has a prediction of the next word.  Reported per condition:
arrival CE (the read when the next word arrives) [min | median | mean | max], the survival-weighted CE
sum_{t<=K} p CE_t (geometric only; its expectation is the expected arrival CE), CE by ticks since the word's onset,
firing per tick (whole brain, Kenyon), mean |h|, non-finite ticks, Kenyon burst onsets and the realized ticks.
Usage: python pace_zero_shot.py <checkpoint dir> [out .jsonl]
"""
import sys, json, math, time, numpy as np, pandas as pd, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork = Path(sys.argv[1])
out = Path(sys.argv[2] if len(sys.argv) > 2 else 'results/published/fly_consequence_20261012/pace_zero_shot.jsonl')
model, cfg, L = load(fork)
import os
if os.environ.get('NO_CHEMO_WRITER'):
    # words are seen and heard, not smelled or tasted: the writer no longer drives the chemosensory cells
    with torch.no_grad():
        model.topographic_writer.proj_chemo.weight.zero_()
        if getattr(model.topographic_writer.proj_chemo, 'bias', None) is not None:
            model.topographic_writer.proj_chemo.bias.zero_()
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
N = model.n_neurons
graph = ROOT / cfg['graph']; g = np.load(graph, allow_pickle=True)
ann = pd.read_feather(graph.parent / 'body-annotations.feather').set_index('bodyId').reindex(g['neuron_body_ids'])
kenyon = torch.as_tensor(ann['class'].astype(str).values == 'Kenyon_Cell', device='cuda')
KC_RENORM = bool(os.environ.get('KC_RENORM'))
if KC_RENORM:
    # the birth normalization divided every input of a Kenyon cell by sqrt(K_in) counting its axo-axonal Kenyon
    # partners (median 162), which are not dendritic input and are pinned at 0; with the real dendritic partners
    # (median 13) each remaining input onto a Kenyon cell is sqrt(K_all / K_dendritic) larger (median 3.6)
    kc_mask = ann['class'].astype(str).values == 'Kenyon_Cell'
    k_all = np.bincount(g['edge_post'], minlength=N)
    k_den = np.bincount(g['edge_post'][~kc_mask[g['edge_pre']]], minlength=N)
    factor = torch.as_tensor(np.where(kc_mask, np.sqrt(k_all / np.maximum(k_den, 1)), 1.0), dtype=torch.float32, device='cuda')
    with torch.no_grad():
        for sfx in ('e', 'i'):
            w = getattr(model, f'edge_weight_{sfx}'); pre_ = getattr(model, f'edge_pre_{sfx}').long(); post_ = getattr(model, f'edge_post_{sfx}').long()
            keep = ~torch.as_tensor(kc_mask, device='cuda')[pre_]
            w.mul_(torch.where(keep, factor[post_], torch.ones_like(w)))
writer = model.topographic_writer
physical_clock = cfg.get('writer_baseline_clock', 'physical') == 'physical'
quiet = torch.zeros(1, N, device='cuda')
BURN, SCORE, MAXT = 256, 1024, 16
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
region = 60_000
targets = np.array(val[region:region + BURN + SCORE], dtype=np.int64)
inputs = np.concatenate(([int(val[region - 1])], targets[:-1]))
min_settle, max_settle, flux_baseline = int(cfg.get('min_settle_ticks', 3)), int(cfg.get('settle_ticks', 14)), float(cfg.get('flux_baseline', 0.048))


def run(name, fixed=None, mu=None, hold=False, native=False):
    rng = np.random.default_rng(1000 + int(mu or 0))
    s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
    arrival, survival, ticks = [], [], []
    by_tick = [[] for _ in range(MAXT + 1)]
    fire_all, fire_kc, habs = [], [], []
    nonfinite = 0
    begin = time.perf_counter()
    with torch.no_grad():
        for i, (tok, tgt) in enumerate(zip(inputs, targets)):
            score = i >= BURN
            token = torch.tensor([int(tok)], device='cuda'); target = torch.tensor([int(tgt)], device='cuda')
            K = None if native else (fixed if fixed else int(rng.geometric(1.0 / mu)))
            ces, t, prev_h, prev_flux = [], 0, None, None
            while True:
                t += 1
                if t == 1 or (hold and writer is not None):
                    drive, baseline = fl.write_input(model, s, token)
                    s = fl.step_fly_physical_tick(model, s, token, drive, baseline, opts, base_rates=opts['base_rates'])
                else:
                    baseline = writer.lambda_adapt * s.baseline if (writer is not None and physical_clock) else s.baseline
                    s = fl.step_fly_physical_tick(model, s, None, quiet, baseline, opts, base_rates=opts['base_rates'])
                if score:
                    lat = fl.read_fly_motor(model, s)
                    ce = float(F.cross_entropy(model.decoder(model.read_norm(lat)).float(), target))
                    ces.append(ce)
                    if t <= MAXT:
                        by_tick[t].append(ce)
                    spikes = (s.ring[0][0] > 0).reshape(-1)
                    fire_all.append(float(spikes.float().mean())); fire_kc.append(float(spikes[kenyon].float().mean()))
                    habs.append(float(s.h.abs().mean()))
                    if not torch.isfinite(s.h).all():
                        nonfinite += 1
                if native:
                    # the checkpoint's flux admission: quiet ticks q = t - 1, stop when the flux settles or stops falling
                    q = t - 1
                    if q >= 1:
                        flux = torch.linalg.vector_norm(s.h - prev_h).item() / math.sqrt(N)
                        if (q >= min_settle and (flux <= flux_baseline or (prev_flux is not None and flux >= prev_flux))) \
                                or q >= max_settle:
                            break
                        prev_flux = flux
                    prev_h = s.h.clone()
                elif t >= K:
                    break
            lat = fl.read_fly_motor(model, s)
            s = fl.expect_next(model, s, lat)
            if score:
                arrival.append(ces[-1]); ticks.append(t)
                if mu:
                    survival.append(sum(ces) / mu)
    a = np.array(arrival); kc = np.array(fire_kc)
    onsets = [u for u in range(50, len(kc)) if kc[u] > 0.01 and kc[u - 50:u].max() <= 0.01]
    d = lambda x: [float(np.min(x)), float(np.median(x)), float(np.mean(x)), float(np.max(x))]
    rep = {'checkpoint': str(fork), 'no_chemo_writer': bool(os.environ.get('NO_CHEMO_WRITER')), 'kc_renorm': KC_RENORM, 'condition': name, 'hold': hold, 'mu': mu, 'fixed': fixed, 'native': native,
           'arrival_ce_min_median_mean_max': d(a), 'survival_ce_mean': float(np.mean(survival)) if survival else None,
           'ticks_min_median_mean_max': d(ticks), 'total_scored_ticks': int(sum(ticks)),
           'ce_by_tick': {t: [float(np.mean(v)), len(v)] for t, v in enumerate(by_tick) if v},
           'fire_all_mean_p99': [float(np.mean(fire_all)), float(np.percentile(fire_all, 99))],
           'fire_kenyon_mean_p99': [float(kc.mean()), float(np.percentile(kc, 99))],
           'h_abs_mean': float(np.mean(habs)), 'nonfinite_ticks': nonfinite, 'kenyon_burst_onsets': len(onsets),
           'seconds': round(time.perf_counter() - begin, 1)}
    short = {k: v for k, v in rep.items() if k not in ('ce_by_tick', 'checkpoint')}
    print(json.dumps(short), flush=True)
    print('  ce_by_tick', {t: round(v[0], 3) for t, v in rep['ce_by_tick'].items()}, flush=True)
    with open(out, 'a') as f:
        f.write(json.dumps(rep) + '\n')


run('native', native=True)
run('fixed1', fixed=1)
run('fixed6', fixed=6)
for mu in (8, 4, 2):
    run(f'geo{mu}_pulse', mu=mu)
    run(f'geo{mu}_hold', mu=mu, hold=True)
