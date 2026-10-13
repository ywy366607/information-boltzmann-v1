import numpy as np
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


def _read_extra(graph_npz, classes):
    import pandas as pd
    if not classes:
        return None
    ids = np.load(graph_npz, allow_pickle=True)['neuron_body_ids']
    cls = pd.read_feather(Path(graph_npz).parent / 'body-annotations.feather').set_index('bodyId').reindex(ids)['class']
    return np.flatnonzero(cls.astype(str).isin([c.strip() for c in classes.split(',')]).values)


def load(fork):
    saved = torch.load(fork / 'last.pt', map_location='cpu', mmap=True, weights_only=False)
    cfg, L = saved['config'], saved['learner']
    graph = ROOT / cfg['graph']
    model = FlyReservoirLM(graph, vocab_size=50257, d_model=cfg['d_model'], injection='topographic', read_surface='output',
                           synapse_model='coba', use_alif=True, decoder_bias=cfg['decoder_bias'],
                           E_E=cfg.get('E_E', 1.0), E_I=cfg.get('E_I', -0.2),
                           use_stp=cfg.get('use_stp', True) or bool(__import__('os').environ.get('ZS_STP')),
                           spike_consumes_drive=cfg.get('spike_consumes_drive', False), refractory_ticks=cfg.get('refractory_ticks', 0),
                           inhibitory_tau_ratio=cfg.get('inhibitory_tau_ratio', 2.0),
                           detach_reset=cfg.get('detach_reset', False), surrogate_mode=cfg.get('surrogate_mode', 'absolute'),
                           transmission_mode=cfg.get('transmission_mode', 'atomic'),
                           use_read_gamma_trace=bool(cfg.get('use_read_gamma_trace', False)), init_read_gamma=0.7,
                           extra_read_index=_read_extra(graph, cfg.get('read_extra_classes'))).cuda()
    model.read_spikes = bool(cfg.get('read_spikes', False))
    if cfg.get('go_neuron') == 'virtual':
        fl.attach_go_neuron(model)
    if cfg.get('read_bank'):
        fl.attach_read_bank(model)
    if cfg.get('intrinsic'):
        fl.attach_intrinsic_drive(model, graph, graph.parent / 'body-annotations.feather',
                                  **({'pattern': cfg['intrinsic_pattern']} if cfg.get('intrinsic_pattern') else {}))
    if cfg.get('dopamine_modulation'):
        fl.attach_dopamine_modulation(model, adapted=bool(cfg.get('dopamine_adapted', False)), affinity=cfg.get('dopamine_affinity'))
        if cfg.get('online_da_plasticity'):
            fl.attach_online_da_plasticity(model, graph, graph.parent / 'body-annotations.feather')
    if cfg.get('regime_homeostasis'):
        fl.attach_regime_homeostasis(model, graph, seed=int(cfg.get('regime_seed') or 0), kenyon_z=cfg.get('regime_kenyon_z'),
                                     annotations=graph.parent / 'body-annotations.feather')
    elif cfg.get('homeostasis'):
        fl.attach_homeostasis(model, graph, seed=int(cfg.get('homeostasis_seed') or 0))
    if cfg.get('predictive_writer'):
        fl.attach_predictive_writer(model)
    if cfg.get('precision'):
        fl.attach_precision(model, graph, graph.parent / 'body-annotations.feather')
    if cfg.get('writer') in ('receptor', 'receptor_all'):
        fl.attach_receptor_writer(model, graph, graph.parent / 'body-annotations.feather',
                                  sense_class='all' if cfg.get('writer') == 'receptor_all' else 'olfactory')
    if cfg.get('tied_writer'):
        fl.attach_tied_writer(model, graph, graph.parent / 'body-annotations.feather', convert=False)
    if cfg.get('graded_types'):
        fl.attach_graded_cells(model, graph, graph.parent / 'body-annotations.feather', types=tuple(cfg['graded_types'].split(',')))
    edges = ('edge_weight_e', 'edge_weight_i')
    from information_boltzmann.core import fly_odour_world as ow
    act = ow.ACT_PARAMETER_NAMES + ('act_mu', 'act_sigma', 'act_w_scale') + ow.SENSE_PARAMETER_NAMES
    from information_boltzmann.core import fly_active_inference as fai
    late = fl.COMPARTMENT_PARAMETER_NAMES + fl.TASTE_PARAMETER_NAMES + act + fai.AI_STATE_NAMES
    if cfg.get('type_classes'):
        fl.split_by_type(model, graph, graph.parent / 'body-annotations.feather', record=False)
    elif cfg.get('kc_subtypes') or saved['model']['log_tau_m'].numel() > model.n_superclasses:
        fl.split_kenyon_subtypes(model, graph, graph.parent / 'body-annotations.feather', record=False)   # a split lineage
    if cfg.get('modulatory_classes') and not cfg.get('type_classes'):
        fl.split_modulatory_classes(model, graph, graph.parent / 'body-annotations.feather', record=False)
    with torch.no_grad():
        for k in edges:
            getattr(model, k).copy_(saved['model'][k])
        assert not model.load_state_dict({k: v for k, v in saved['model'].items() if k not in edges and k not in late}, strict=False).unexpected_keys
        if cfg.get('compartment_memory'):
            # the plastic Kenyon -> MBON synapses live outside the edge list: attach, then load their own weights
            fl.attach_compartment_memory(model, graph, graph.parent / 'body-annotations.feather', affinity=cfg.get('dopamine_affinity'))
            fl.attach_taste_heads(model, graph, graph.parent / 'body-annotations.feather')
            model.load_state_dict({k: v for k, v in saved['model'].items() if k in late}, strict=False)
        if 'act_w' in saved['model']:
            # the act head (odour world): attach with the saved tensors, then load them exactly
            init = torch.load(ROOT / cfg.get('act_init', 'scratch/fly_stream/act_init.pt'), map_location='cpu')
            ow.attach_act_head(model, init)            # constants (mu, sigma, read scale) from the init; learned below
            model.load_state_dict({k: v for k, v in saved['model'].items() if k in act}, strict=False)
        if 'ai_w' in saved['model']:
            if cfg.get('motor_acts'):
                fai.attach_motor_acts(model, tau_muscle=float(cfg.get('act_tau') or 8.0), eye_open=float(cfg.get('eye_init') or 0.9),
                                      habit=float(cfg.get('act_habit') or 0.2),
                                      dopamine_precision=bool(cfg.get('act_dopamine_precision')))
                fai.set_motor_constants(model, saved['model'])
            else:
                fai.attach_ai_head(model, torch.load(ROOT / cfg.get('act_init', 'scratch/fly_stream/act_init.pt'), map_location='cpu'))
            model.load_state_dict({k: v for k, v in saved['model'].items() if k in fai.AI_STATE_NAMES}, strict=False)
        if 'sense_head.weight' in saved['model']:
            ow.attach_sense_head(model, saved['model']['sense_head.weight'].shape[0])
            model.load_state_dict({k: v for k, v in saved['model'].items() if k in ow.SENSE_PARAMETER_NAMES}, strict=False)
    import os
    # zero-shot interventions (measure a fix before training it): ZS_DAN_RENORM=1 renormalizes the dopamine cells'
    # inputs without their pinned Kenyon partners; INTRINSIC_KAPPA releases the tonically active modulatory cells
    # (INTRINSIC_PATTERN, default PAM / PPL / OA / 5-HT / DPM) with that intrinsic current
    if os.environ.get('ZS_DAN_RENORM') and not cfg.get('dan_renorm'):
        print(f'zero-shot: dopamine inputs renormalized (median x{fl.renormalize_dopamine_inputs(model, graph, graph.parent / "body-annotations.feather"):.2f})', flush=True)
    if os.environ.get('ZS_DA_INPUT_GAIN'):
        # dose-response: every excitatory input onto the dopamine cells x gain (how far is the route from working?)
        gain = float(os.environ['ZS_DA_INPUT_GAIN'])
        dan_t = torch.zeros(model.n_neurons, dtype=torch.bool, device='cuda')
        dan_t[torch.as_tensor(np.load(graph, allow_pickle=True)['dan_indices'].astype(np.int64), device='cuda')] = True
        with torch.no_grad():
            model.edge_weight_e[dan_t[model.edge_post_e.long()]] *= gain
        print(f'zero-shot: excitatory inputs onto dopamine cells x{gain}', flush=True)
    if os.environ.get('INTRINSIC_KAPPA') and getattr(model, 'intrinsic_kappa', None) is None:
        released = fl.attach_intrinsic_drive(model, graph, graph.parent / 'body-annotations.feather', kappa=float(os.environ['INTRINSIC_KAPPA']),
                                             pattern=os.environ.get('INTRINSIC_PATTERN', r'^(PAM\d|PPL\d|OA-|5-HT|DPM$)'))
        print(f'zero-shot: intrinsic drive kappa {os.environ["INTRINSIC_KAPPA"]} on {int(model.intrinsic_index.numel())} cells', flush=True)
    if os.environ.get('ZS_GRADED') and getattr(model, 'graded_mask', None) is None:
        n = fl.attach_graded_cells(model, graph, graph.parent / 'body-annotations.feather', types=tuple(os.environ['ZS_GRADED'].split(',')),
                                   gain=float(os.environ.get('ZS_GRADED_GAIN', 1.0)))
        print(f'zero-shot: graded cells {n}, gain {os.environ.get("ZS_GRADED_GAIN", 1.0)}', flush=True)
    if os.environ.get('ZS_TIED_WRITER') and not cfg.get('tied_writer'):
        n = fl.attach_tied_writer(model, graph, graph.parent / 'body-annotations.feather', convert=True,
                                  scale=float(os.environ['ZS_TIED_WRITER']))
        print(f'zero-shot: writer tied by chemosensory receptor type, scale {os.environ["ZS_TIED_WRITER"]}: {n}', flush=True)
    if os.environ.get('ZS_CLAW_GAIN'):
        # zero-shot: excitatory inputs onto Kenyon cells from non-Kenyon cells (their dendritic claws) x gain
        import pandas as pd
        cls_t = pd.read_feather(graph.parent / 'body-annotations.feather').set_index('bodyId').reindex(
            np.load(graph, allow_pickle=True)['neuron_body_ids'])['class'].astype(str).values
        kc_t = torch.as_tensor(cls_t == 'Kenyon_Cell', device='cuda')
        src = torch.as_tensor(np.isin(cls_t, os.environ.get('ZS_CLAW_FROM', 'ALPN').split(',')), device='cuda')             if os.environ.get('ZS_CLAW_FROM') != 'all' else ~kc_t
        sel = kc_t[model.edge_post_e.long()] & src[model.edge_pre_e.long()]
        with torch.no_grad():
            model.edge_weight_e[sel] *= float(os.environ['ZS_CLAW_GAIN'])
        print(f'zero-shot: {int(sel.sum())} claw inputs onto Kenyon cells x{os.environ["ZS_CLAW_GAIN"]}', flush=True)
    if os.environ.get('ZS_GRADED_SCALE') and getattr(model, 'log_graded_gain', None) is not None:
        with torch.no_grad():
            model.log_graded_gain.add_(float(np.log(float(os.environ['ZS_GRADED_SCALE']))))
        print(f'zero-shot: graded gain x{os.environ["ZS_GRADED_SCALE"]} -> {model.log_graded_gain.exp().tolist()}', flush=True)
    if os.environ.get('ZS_STP'):
        # zero-shot short-term plasticity 'u0,tau_fac,tau_rec' on every synapse (tau_fac 5 = depression only)
        u0, tf, tr = (float(x) for x in os.environ['ZS_STP'].split(','))
        with torch.no_grad():
            model.logit_u0.fill_(float(np.log(u0 / (1 - u0)))); model.log_tau_fac.fill_(float(np.log(tf))); model.log_tau_rec.fill_(float(np.log(tr)))
        print(f'zero-shot: STP u0 {u0} tau_fac {tf} tau_rec {tr}', flush=True)
    if os.environ.get('ZS_NO_STP'):
        model.use_stp = False
        print('zero-shot: STP off', flush=True)
    if os.environ.get('ZS_ALIF_BETA'):
        with torch.no_grad():
            model.log_beta.add_(float(np.log(float(os.environ['ZS_ALIF_BETA']))))
        print(f'zero-shot: adaptation strength x{os.environ["ZS_ALIF_BETA"]}', flush=True)
    if os.environ.get('ZS_SILENCE_OUTPUT'):
        # zero-shot: the output synapses of the given cell types removed (comma-separated types), e.g. APL
        import pandas as pd
        ann_t = pd.read_feather(graph.parent / 'body-annotations.feather').set_index('bodyId').reindex(
            np.load(graph, allow_pickle=True)['neuron_body_ids'])['type'].astype(str).values
        quiet_cells = torch.as_tensor(np.isin(ann_t, os.environ['ZS_SILENCE_OUTPUT'].split(',')), device='cuda')
        with torch.no_grad():
            for sfx in ('e', 'i'):
                getattr(model, f'edge_weight_{sfx}')[quiet_cells[getattr(model, f'edge_pre_{sfx}').long()]] = 0.0
        print(f'zero-shot: output synapses of {os.environ["ZS_SILENCE_OUTPUT"]} removed ({int(quiet_cells.sum())} cells)', flush=True)
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
