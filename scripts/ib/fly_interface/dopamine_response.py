"""Zero-shot: does surprise reach the dopamine cells?  Fixed weights of a predictive-coding individual.

For every word read with its own go neuron: surprise = norm of the innovation written for it (predictive coding) and
the error of the prediction it met; response = spikes of the 392 dopaminergic cells (the graph's dan_indices) and the
dopamine they deliver along their 241,701 modulatory projections (summed over the word's ticks), in this word's window
and the next.  Compared with the released intrinsic current as learned (kappa ~ 1.2) and with it removed (kappa = 0:
driven by the connectome like every other cell).
"""
import sys, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
from scipy.stats import spearmanr
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scratch/fly_stream/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])
from information_boltzmann.core.triton_synapse import execute_delayed_synaptic_transmission

fork = Path(sys.argv[1])
model, cfg, L = load(fork)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
max_ticks = int(L.get('go_max_ticks', 32))
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
dan = model.dan_indices
tick_log = []
original_step = fl.step_fly_physical_tick


def step(model_, current, *a, **k):
    out = original_step(model_, current, *a, **k)
    with torch.no_grad():
        arrival = execute_delayed_synaptic_transmission(tuple(p.detach() for p in out.ring), model.dan_edge_pre,
                                                        model.dan_edge_post, model.dan_edge_weight, model.dan_delay_splits)
        tick_log.append((float((out.ring[0][0, dan] > 0).float().sum()), float(arrival.sum() / model.dan_scale)))
    return out


fl.step_fly_physical_tick = step
region, BURN, SCORE = 60_000, 256, 1024
targets = np.array(val[region:region + BURN + SCORE], dtype=np.int64)
inputs = np.concatenate(([int(val[region - 1])], targets[:-1]))


def run():
    torch.manual_seed(0)
    s = move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')
    rows = []
    with torch.no_grad():
        for i, tok in enumerate(inputs):
            t = torch.tensor([int(tok)], device='cuda')
            drive, _ = fl.write_input(model, s, t)
            innovation = float(drive[0, model.topographic_writer.injection_index].norm())
            tick_log.clear()
            s, lat, _, _, _, _ = fl.advance_fly_token_go(model, s, t, [], max_ticks=max_ticks,
                                                         writer_baseline_clock='physical', **opts)
            ce = float(F.cross_entropy(model.decoder(model.read_norm(lat[-1:])).float(), torch.tensor([int(targets[i])], device='cuda')))
            spikes = sum(x[0] for x in tick_log) / len(tick_log)
            arrival = sum(x[1] for x in tick_log) / len(tick_log)
            rows.append((innovation, ce, spikes, arrival))
    a = np.array(rows[BURN:])
    met_error = a[:-1, 1]                    # the error of the prediction word i+1 met = row i's error
    surprise, spikes_now, arrival_now = a[1:, 0], a[1:, 2], a[1:, 3]
    spikes_next, arrival_next = np.r_[a[2:, 2], np.nan], np.r_[a[2:, 3], np.nan]
    ok = ~np.isnan(spikes_next)
    hi, lo = surprise > np.quantile(surprise, .75), surprise < np.quantile(surprise, .25)
    return {'dopamine_spikes_per_tick': float(a[:, 2].mean()), 'dopamine_arrival_per_tick': float(a[:, 3].mean()),
            'spearman_innovation_vs_spikes_same_word': float(spearmanr(surprise, spikes_now).correlation),
            'spearman_innovation_vs_spikes_next_word': float(spearmanr(surprise[ok], spikes_next[ok]).correlation),
            'spearman_innovation_vs_arrival_same_word': float(spearmanr(surprise, arrival_now).correlation),
            'spearman_error_met_vs_spikes_same_word': float(spearmanr(met_error, spikes_now).correlation),
            'spikes_high_minus_low_surprise_in_sd': float((spikes_now[hi].mean() - spikes_now[lo].mean()) / (spikes_now.std() + 1e-9)),
            'arrival_high_minus_low_surprise_in_sd': float((arrival_now[hi].mean() - arrival_now[lo].mean()) / (arrival_now.std() + 1e-9)),
            'spearman_prediction_error_vs_innovation': float(spearmanr(met_error, surprise).correlation)}


report = {'fork': str(fork)}
report['kappa_as_learned'] = run()
print('kappa as learned:', {k: round(v, 3) for k, v in report['kappa_as_learned'].items()}, flush=True)
if getattr(model, 'intrinsic_kappa', None) is not None:
    model.intrinsic_kappa.data.zero_()
    report['kappa_zero'] = run()
    print('kappa = 0       :', {k: round(v, 3) for k, v in report['kappa_zero'].items()}, flush=True)
json.dump(report, open(fork / 'dopamine_response.json', 'w'), indent=1)
