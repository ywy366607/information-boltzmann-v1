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
    if cfg.get('writer') in fl.WRITER_SENSE:
        fl.attach_receptor_writer(model, graph, graph.parent / 'body-annotations.feather',
                                  sense_class=fl.WRITER_SENSE[cfg.get('writer')])
    if cfg.get('tied_writer'):
        fl.attach_tied_writer(model, graph, graph.parent / 'body-annotations.feather', convert=False)
    if cfg.get('graded_types'):
        fl.attach_graded_cells(model, graph, graph.parent / 'body-annotations.feather', types=tuple(cfg['graded_types'].split(',')))
    edges = ('edge_weight_e', 'edge_weight_i')
    from information_boltzmann.core import fly_odour_world as ow
    act = ow.ACT_PARAMETER_NAMES + ('act_mu', 'act_sigma', 'act_w_scale') + ow.SENSE_PARAMETER_NAMES
    from information_boltzmann.core import fly_active_inference as fai
    late = fl.COMPARTMENT_PARAMETER_NAMES + fl.TASTE_PARAMETER_NAMES + act + fai.AI_STATE_NAMES + fl.STREAM_JEPA_PARAMETER_NAMES
    if cfg.get('input_gain_size_exponent') is not None:
        fl.attach_input_gain(model)
    if cfg.get('kc_dan_release'):
        fl.attach_kenyon_terminal_release(model, graph, graph.parent / 'body-annotations.feather')
        if cfg.get('kc_habituation_tau'):
            fl.attach_kenyon_habituation(model, cfg['kc_habituation_tau'], cfg.get('kc_habituation_use') or 0.2)
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
            model.comp_adapted = bool(cfg.get('compartment_adapted', False))
            fl.attach_taste_heads(model, graph, graph.parent / 'body-annotations.feather')
            model.load_state_dict({k: v for k, v in saved['model'].items() if k in late}, strict=False)
        if cfg.get('stream_jepa'):
            fl.attach_stream_jepa(model, graph, graph.parent / 'body-annotations.feather')
            model.load_state_dict({k: v for k, v in saved['model'].items() if k in fl.STREAM_JEPA_PARAMETER_NAMES}, strict=False)
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
    if os.environ.get('ZS_COMPARTMENT') and not cfg.get('compartment_dendritic'):
        # the general compartment rule applied now, on the trained weights (--compartment-dendritic, zero-shot)
        print(f'zero-shot: compartment rule {fl.attach_compartment_dendritic(model, graph, graph.parent / "body-annotations.feather", graph.parent, rescale=True)}', flush=True)
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
    if os.environ.get('ZS_GRADED_ADD'):
        # zero-shot: more graded cells on top of the individual's own ('~T' = tonic operating point), learned gains kept
        types = tuple(filter(None, (cfg.get('graded_types') or '').split(','))) + tuple(os.environ['ZS_GRADED_ADD'].split(','))
        n = fl.attach_graded_cells(model, graph, graph.parent / 'body-annotations.feather', types=types,
                                   tonic_gain=float(os.environ.get('ZS_TONIC_GAIN', 1.0)))
        print(f'zero-shot: graded cells {n}, tonic gain {os.environ.get("ZS_TONIC_GAIN", 1.0)}', flush=True)
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
    if os.environ.get('ZS_SILENCE_MASK'):
        # zero-shot: a set of cells (bool .npy over the graph's neurons, e.g. the latched ones) removed from the brain:
        # their output synapses cut and, where they are read cells, their read weights zeroed
        quiet_cells = torch.as_tensor(np.load(os.environ['ZS_SILENCE_MASK']), device='cuda')
        with torch.no_grad():
            for sfx in ('e', 'i'):
                getattr(model, f'edge_weight_{sfx}')[quiet_cells[getattr(model, f'edge_pre_{sfx}').long()]] = 0.0
            if model.read_surface != 'all':
                model.output_read.weight[:, quiet_cells[model.read_indices]] = 0.0
        print(f'zero-shot: {int(quiet_cells.sum())} cells silenced ({os.environ["ZS_SILENCE_MASK"]})', flush=True)
    if os.environ.get('ZS_WRITER_GAIN') and not getattr(fl, '_zs_writer_gain', None):
        # zero-shot (2026-10-15): the text writer's drive x gain (the senses' strength), every caller of write_input
        g_w = float(os.environ['ZS_WRITER_GAIN'])
        _orig_write = fl.write_input

        def _gained_write(*a, **k):
            drive, baseline = _orig_write(*a, **k)
            return (drive * g_w if drive is not None else drive), baseline
        fl.write_input = _gained_write
        fl._zs_writer_gain = g_w
        print(f'zero-shot: writer drive x{g_w}', flush=True)
    if os.environ.get('ZS_RESTORE_KAPPA') and getattr(model, 'intrinsic_kappa', None) is not None:
        # zero-shot (2026-10-15): the pacemaker drive of the spontaneously active modulatory cells back at birth
        # (BPTT turned it down in m_main: median 1.21 -> 0.39)
        birth_sd = torch.load(fork / 'birth.pt', map_location='cpu', mmap=True, weights_only=False)['model']
        with torch.no_grad():
            before = float(model.intrinsic_kappa.median())
            model.intrinsic_kappa.copy_(birth_sd['intrinsic_kappa'].to(model.intrinsic_kappa.device))
        print(f'zero-shot: intrinsic kappa restored to birth (median {before:.3f} -> {float(model.intrinsic_kappa.median()):.3f})', flush=True)
    if os.environ.get('ZS_TIME_SCALE'):
        # zero-shot (2026-10-15, user: drop the biological rate limit): the whole dynamics run k times faster per tick
        # -- membrane and synaptic time constants / k, conductance gains and the senses' drive x k (one tick carries
        # the charge of k), thresholds, wiring and learned weights unchanged.  Delays (1-4 ticks) cannot shrink.
        k_t = float(os.environ['ZS_TIME_SCALE'])
        import math as _m
        with torch.no_grad():
            for name in ('log_tau_m', 'log_tau_s_e', 'log_tau_s_i', 'log_tau_a'):
                if getattr(model, name, None) is not None:
                    getattr(model, name).sub_(_m.log(k_t))
            for name in ('log_g_e', 'log_g_i'):
                if getattr(model, name, None) is not None:
                    getattr(model, name).add_(_m.log(k_t))
        _orig_write_t = fl.write_input

        def _fast_write(*a, **k):
            drive, baseline = _orig_write_t(*a, **k)
            return (drive * k_t if drive is not None else drive), baseline
        fl.write_input = _fast_write
        print(f'zero-shot: time scale x{k_t} (tau / {k_t}, conductance and input x {k_t})', flush=True)
    if os.environ.get('ZS_KC_POOLED') and getattr(model, 'log_input_gain', None) is not None:
        # zero-shot, at birth only (2026-10-15): the input gains re-initialised from size with the Kenyon subtypes
        # pooled (on birth.pt every other class gets back exactly its birth value)
        print('zero-shot: Kenyon input gain pooled', fl.init_input_gain_from_size(model, graph, float(cfg.get('input_gain_size_exponent') or 0.5), kenyon_pooled=True), flush=True)
    if os.environ.get('ZS_MBON_KAPPA'):
        import pandas as pd
        # zero-shot (2026-10-15): the MBONs join the spontaneously active cells with intrinsic current kappa (x their
        # rheobase), appended to the pacemakers' (MBONs are tonically active in the fly; at birth they were silent)
        k_m = float(os.environ['ZS_MBON_KAPPA'])
        ids_ = np.load(graph, allow_pickle=True)['neuron_body_ids']
        cls_ = pd.read_feather(graph.parent / 'body-annotations.feather').set_index('bodyId').reindex(ids_)['class'].astype(str).values
        mb = torch.as_tensor(np.nonzero(cls_ == 'MBON')[0], dtype=torch.long, device='cuda')
        model.register_buffer('mbon_index', mb, persistent=False)
        model.mbon_kappa = torch.nn.Parameter(torch.full((mb.numel(),), k_m, device='cuda'))
        print(f'zero-shot: {mb.numel()} MBONs spontaneously active, kappa {k_m}', flush=True)
    if os.environ.get('ZS_COMP_TIMING') and getattr(model, 'comp_eta', None) is not None:
        # zero-shot (2026-10-15): the timing-signed compartment rule, trace-gated forgetting off (attach_comp_timing)
        print(f'zero-shot: compartment timing rule (potentiation start {fl.attach_comp_timing(model):.3f}), '
              f'use-forgetting off', flush=True)
    if os.environ.get('ZS_COMP_TONIC') and getattr(model, 'comp_eta', None) is not None:
        # zero-shot (2026-10-15): active forgetting of the Kenyon -> MBON memory at this many ticks (tonic dopamine)
        print(f'zero-shot: active forgetting, tau {fl.attach_comp_tonic_forget(model, float(os.environ["ZS_COMP_TONIC"])):.0f} ticks', flush=True)
    if os.environ.get('ZS_RESTORE_DAN_INHIB'):
        # zero-shot (2026-10-15): the inhibitory inputs onto the dopamine cells back at their birth weights (training
        # grew inhibition that silenced PPL1: birth +0.77 theta, c21 -1.0 theta); excitation and everything else as is
        birth_sd = torch.load(fork / 'birth.pt', map_location='cpu', mmap=True, weights_only=False)['model']
        dan_m = torch.zeros(model.n_neurons, dtype=torch.bool, device='cuda')
        dan_m[torch.as_tensor(np.load(graph, allow_pickle=True)['dan_indices'].astype(np.int64), device='cuda')] = True
        with torch.no_grad():
            w = model.edge_weight_i
            m = dan_m[model.edge_post_i.long()] & (w != 0)
            before = float(w[m].abs().sum())
            w[m] = birth_sd['edge_weight_i'].to(w.device, w.dtype)[m]
        print(f'zero-shot: inhibition onto the dopamine cells restored to birth ({int(m.sum())} edges, '
              f'|w| sum {before:.1f} -> {float(w[m].abs().sum()):.1f})', flush=True)
    if os.environ.get('ZS_RESTORE_TASTE'):
        # zero-shot (2026-10-15): where were the innate taste lines lost?  The output synapses of the taste receptors
        # (sugar / bitter / water / salt GRNs, Vicente 2025) and of every cell fewer than K hops downstream of them
        # back at their birth weights (edges the connectome rules pinned at 0 stay 0)
        import pandas as pd
        K = int(os.environ['ZS_RESTORE_TASTE'])
        gnpz = np.load(graph, allow_pickle=True)
        typ_t = pd.read_feather(graph.parent / 'body-annotations.feather').set_index('bodyId').reindex(
            gnpz['neuron_body_ids'])['type'].astype(str).values
        src = np.isin(typ_t, ['LB3b', 'LB3c', 'dorsal_tpGRN', 'LB1a', 'LB1b', 'LB1c', 'LB1d', 'LB3a', 'LB3d'])
        pre_all = np.concatenate([gnpz['edge_pre_e'], gnpz['edge_pre_i']]); post_all = np.concatenate([gnpz['edge_post_e'], gnpz['edge_post_i']])
        sel, front = src.copy(), src.copy()
        for _ in range(K - 1):
            nxt = np.zeros_like(src); nxt[post_all[front[pre_all]]] = True
            front = nxt & ~sel; sel |= nxt
        birth_sd = torch.load(fork / 'birth.pt', map_location='cpu', mmap=True, weights_only=False)['model']
        sel_t = torch.as_tensor(sel, device='cuda')
        n_restored = 0
        with torch.no_grad():
            for sfx in ('e', 'i'):
                w = getattr(model, f'edge_weight_{sfx}')
                m = sel_t[getattr(model, f'edge_pre_{sfx}').long()] & (w != 0)
                w[m] = birth_sd[f'edge_weight_{sfx}'].to(w.device, w.dtype)[m]
                n_restored += int(m.sum())
        print(f'zero-shot: taste lines restored to birth weights, {int(src.sum())} receptors + {int(sel.sum() - src.sum())} '
              f'cells within {K - 1} hops, {n_restored} output edges', flush=True)
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
