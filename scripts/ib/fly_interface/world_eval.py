"""Specialised measures for the worlds (fixed weights, same fresh text for every individual).

read     self-paced: the go neuron decides when to speak and move on (sampling seeded)
listen k externally paced: every word takes k ticks, the prediction of the next word is read just before it arrives
gap G    self-paced, but after every 4th word the world is silent for G ticks before the next word; the prediction
         that word meets is the read at the end of the silence.  Reported as the NLL of those words after the gap and
         its difference to the same words read without a gap.
Usage: python world_eval.py <fork dir> [<fork dir> ...]
"""
import sys, json, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
import information_boltzmann.core.fly_bptt_learning as fl
from scripts.ib.inspect_fly_bptt_credit import move_state

ROOT = Path('D:/information-boltzmann-v1')
BURN, SCORE = 256, 1024


def load(fork):
    saved = torch.load(fork / 'last.pt', map_location='cpu', mmap=True, weights_only=False)
    cfg, L = saved['config'], saved['learner']
    graph = ROOT / cfg['graph']
    model = FlyReservoirLM(graph, vocab_size=50257, d_model=cfg['d_model'], injection='topographic', read_surface='output',
                           synapse_model='coba', use_alif=True, use_stp=True, decoder_bias=cfg['decoder_bias'],
                           detach_reset=cfg.get('detach_reset', False), surrogate_mode=cfg.get('surrogate_mode', 'absolute'),
                           transmission_mode=cfg.get('transmission_mode', 'atomic')).cuda()
    if cfg.get('go_neuron') == 'virtual':
        fl.attach_go_neuron(model)
    if cfg.get('read_bank'):
        fl.attach_read_bank(model)
    if cfg.get('intrinsic'):
        fl.attach_intrinsic_drive(model, graph, graph.parent / 'body-annotations.feather')
    if cfg.get('dopamine_modulation'):
        fl.attach_dopamine_modulation(model, adapted=bool(cfg.get('dopamine_adapted', False)), affinity=cfg.get('dopamine_affinity'))
        if cfg.get('online_da_plasticity'):
            fl.attach_online_da_plasticity(model, graph, graph.parent / 'body-annotations.feather')
    if cfg.get('regime_homeostasis'):
        fl.attach_regime_homeostasis(model, graph, seed=int(cfg.get('regime_seed') or 0))
    elif cfg.get('homeostasis'):
        fl.attach_homeostasis(model, graph, seed=int(cfg.get('homeostasis_seed') or 0))
    if cfg.get('predictive_writer'):
        fl.attach_predictive_writer(model)
    if cfg.get('precision'):
        fl.attach_precision(model, graph, graph.parent / 'body-annotations.feather')
    if cfg.get('tied_writer'):
        fl.attach_tied_writer(model, graph, graph.parent / 'body-annotations.feather', convert=False)
    edges = ('edge_weight_e', 'edge_weight_i')
    with torch.no_grad():
        for k in edges:
            getattr(model, k).copy_(saved['model'][k])
        assert not model.load_state_dict({k: v for k, v in saved['model'].items() if k not in edges}, strict=False).unexpected_keys
    model.eval().requires_grad_(False)
    return model, cfg, L


def evaluate(fork):
    model, cfg, L = load(fork)
    opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
                alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
    max_ticks = int(L.get('go_max_ticks', 32))
    val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
    region = 60_000                            # a fixed stretch of validation text no evaluation has used, the same for every individual
    targets = np.array(val[region:region + BURN + SCORE], dtype=np.int64)
    inputs = np.concatenate(([int(val[region - 1])], targets[:-1]))
    start = lambda: move_state(fl.FlyPhysicalState(**L['physical']), 'cuda')

    def read_world(gap=0):
        torch.manual_seed(0)
        s, out, ticks = start(), [], 0
        with torch.no_grad():
            for i, tok in enumerate(inputs):
                g = gap if (gap and i >= BURN and (i - BURN) % 4 == 0) else 0
                s, lat, _, _, _, after = fl.advance_fly_token_go(model, s, torch.tensor([int(tok)], device='cuda'), [],
                                                                  max_ticks=max_ticks, writer_baseline_clock='physical',
                                                                  silence_after=g, **opts)
                ticks += lat.shape[0] + g
                if i >= BURN:
                    z = after if g else lat[-1:]
                    out.append(float(F.cross_entropy(model.decoder(model.read_norm(z)).float(),
                                                     torch.tensor([int(targets[i])], device='cuda'))))
        return np.array(out), ticks / len(inputs)

    def listen_world(k):
        s, out = start(), []
        with torch.no_grad():
            for i, tok in enumerate(inputs):
                s, z = fl.advance_fly_token_stream(model, s, torch.tensor([int(tok)], device='cuda'), ticks=k,
                                                   writer_baseline_clock='physical', **opts)
                if i >= BURN:
                    out.append(float(F.cross_entropy(model.decoder(model.read_norm(z)).float(),
                                                     torch.tensor([int(targets[i])], device='cuda'))))
        return np.array(out)

    r = {'fork': str(fork)}
    read, tpt = read_world()
    r['read_nll'], r['read_ticks_per_word'] = float(read.mean()), tpt
    for k in (3, 4, 6):
        r[f'listen_k{k}_nll'] = float(listen_world(k).mean())
        r[f'listen_k{k}_cost_vs_read'] = r[f'listen_k{k}_nll'] - r['read_nll']
    mask = (np.arange(SCORE) % 4 == 0)
    for gap in (8, 16, 32):
        g, _ = read_world(gap)
        r[f'gap{gap}_after_nll'] = float(g[mask].mean())
        r[f'gap{gap}_cost_vs_no_gap'] = float((g[mask] - read[mask]).mean())
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    return r


results = [evaluate(Path(f)) for f in sys.argv[1:]]
json.dump(results, open(ROOT / ('scratch/fly_stream/world_eval_' + '_'.join(Path(f).name for f in sys.argv[1:]) + '.json'), 'w'), indent=1)
