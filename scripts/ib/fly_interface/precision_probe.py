"""Does the brain turn familiarity into precision?  Fixed weights, self-paced reading.

Per word, against its rarity (-log unigram frequency in the training text; rare = unfamiliar):
  sensory gain     spikes per tick of the direct targets of the sensory (injection) neurons, and that response
                   divided by the written innovation (response per unit of input)
  learning rate    the dopamine plasticity scale (the learner's own rule) averaged over the neurons that fired in the
                   word: at those neurons' synapses the update is multiplied by it
  thinking time    ticks the go neuron took
  energy           spikes x outgoing synapses, per word and per tick
  dopamine         mean relative phasic dopamine at the targets at the read
Reported as the Spearman with rarity and the mean (in sd of the measure) of each rarity quintile, Q1 familiar ..
Q5 rare; once as learned and once with the dopamine sensitivities at 0 (no dopamine effect on excitability), so the
difference is what dopamine adds.
Usage: python precision_probe.py <fork dir>
"""
import sys, json, numpy as np, torch
from pathlib import Path
from scipy.stats import spearmanr
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scratch/fly_stream/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])

fork = Path(sys.argv[1])
model, cfg, L = load(fork)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
max_ticks = int(L.get('go_max_ticks', 32))
N = model.n_neurons
inj = model.topographic_writer.injection_index
pre_all = torch.cat([model.edge_pre_e.long(), model.edge_pre_i.long()])
post_all = torch.cat([model.edge_post_e.long(), model.edge_post_i.long()])
is_inj = torch.zeros(N, dtype=torch.bool, device='cuda'); is_inj[inj] = True
sensory_targets = torch.zeros(N, dtype=torch.bool, device='cuda'); sensory_targets[post_all[is_inj[pre_all]]] = True
sensory_targets &= ~is_inj
outdeg = torch.bincount(pre_all, minlength=N).float()
adapted = bool(getattr(model, 'dopamine_adapted', False))
acc = {}
original_step = fl.step_fly_physical_tick


def step(model_, current, *a, **k):
    out = original_step(model_, current, *a, **k)
    s = (out.ring[0][0] > 0)
    acc['energy'] = acc.get('energy', 0.0) + float((s.float() * outdeg).sum())
    acc['sensory'] = acc.get('sensory', 0.0) + float(s[sensory_targets].float().sum())
    acc['fired'] = s if 'fired' not in acc else (acc['fired'] | s)
    return out


fl.step_fly_physical_tick = step
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
region, BURN, SCORE = 60_000, 256, 1024
targets = np.array(val[region:region + BURN + SCORE], dtype=np.int64)
inputs = np.concatenate(([int(val[region - 1])], targets[:-1]))
counts = np.bincount(np.load(ROOT / cfg['data'] / 'train.npy').astype(np.int64), minlength=50257) + 1.0
rarity = -np.log(counts / counts.sum())[inputs[BURN:]]
n_sens = float(sensory_targets.sum())


def plasticity_scale(dopamine):
    """The learner's dopamine plasticity rule (FlyBPTTLearner.observe), per neuron."""
    conc = dopamine[0]
    tg = conc > 0
    local = torch.ones_like(conc)
    if bool(tg.any()):
        level = conc[tg].mean().clamp_min(1e-9)
        if adapted:
            slow = dopamine[1][tg]
            local[tg] = (1.0 + (conc[tg] - slow) / (slow + level)).clamp(0.25, 4.0)
        else:
            local[tg] = (conc[tg] / level).clamp(0.25, 4.0)
    return local


def run():
    torch.manual_seed(0)
    s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
    rows = []
    with torch.no_grad():
        for i, tok in enumerate(inputs):
            t = torch.tensor([int(tok)], device='cuda')
            drive, _ = fl.write_input(model, s, t)
            innov = float(drive[0, inj].norm())
            acc.clear()
            s, lat, _, _, _, _ = fl.advance_fly_token_go(model, s, t, [], max_ticks=max_ticks, writer_baseline_clock='physical', **opts)
            if i < BURN:
                continue
            ticks = lat.shape[0]
            row = {'thinking ticks': ticks, 'energy per word': acc['energy'], 'energy per tick': acc['energy'] / ticks,
                   'sensory response per tick': acc['sensory'] / ticks / n_sens,
                   'sensory gain (response / innovation)': acc['sensory'] / ticks / n_sens / max(innov, 1e-9),
                   'written innovation': innov}
            if s.dopamine.numel():
                fast, slow = s.dopamine[0], s.dopamine[1]
                tg = slow > 0
                level = slow[tg].mean()
                row['phasic dopamine'] = float(((fast - slow) / (slow + level) if adapted else (fast - slow) / level)[tg].mean())
                row['learning-rate scale at fired neurons'] = float(plasticity_scale(s.dopamine)[acc['fired']].mean())
            rows.append(row)
    out = {}
    for k in rows[0]:
        v = np.array([r[k] for r in rows], dtype=np.float64)
        z = (v - v.mean()) / (v.std() + 1e-12)
        q = np.quantile(rarity, [0.2, 0.4, 0.6, 0.8])
        bins = np.digitize(rarity, q)
        out[k] = {'spearman_with_rarity': float(spearmanr(rarity, v).correlation),
                  'quintile_z_familiar_to_rare': [float(z[bins == b].mean()) for b in range(5)],
                  'mean': float(v.mean())}
    return out


report = {'fork': str(fork), 'adapted': adapted}
report['as_learned'] = run()
for k, v in report['as_learned'].items():
    print(f'{k:40s} rho={v["spearman_with_rarity"]:+.3f}  Q1..Q5 z=', [round(x, 2) for x in v['quintile_z_familiar_to_rare']], flush=True)
if getattr(model, 'dopamine_sensitivity', None) is not None:
    model.dopamine_sensitivity.data.zero_()
    report['no_dopamine_effect'] = run()
    print('-- dopamine sensitivities at 0 --', flush=True)
    for k, v in report['no_dopamine_effect'].items():
        print(f'{k:40s} rho={v["spearman_with_rarity"]:+.3f}  Q1..Q5 z=', [round(x, 2) for x in v['quintile_z_familiar_to_rare']],
              f' mean change {report["as_learned"][k]["mean"] - v["mean"]:+.4g}', flush=True)
json.dump(report, open(fork / 'precision_probe.json', 'w'), indent=1)
