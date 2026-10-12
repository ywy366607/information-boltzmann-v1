"""Where does 'taste once, choose differently next time' break?  Memory -> MBON -> output, link by link.
The adult with the Kenyon fixes, dopamine modulation and compartment memory (rate 2), fixed weights except the online
memory (it runs, as in life).  Text streams throughout (mean 8 ticks/word, pulses).
Two odours A, B (8 receptor types each, whiffs p = 0.5 at 32 rheobase for 24 ticks).
  baseline: A and B presented 4 times each (no outcome)
  pairing:  A + sweet (taste cells, and the dopamine cells 3 ticks later), B + bitter, 3 times each
  test:     A and B presented 4 times each (no outcome)
Measured per presentation (odour window minus the matched window before it):
  (a) phi change on the Kenyon -> MBON edges of the Kenyon cells each odour drives (pairing vs baseline)
  (b) every MBON's response; the valence-weighted sum sum_j v_j dMBON_j with v_j = (PPL1 - PAM)/(PPL1 + PAM) input
      (in the fly, MBONs of PPL1 compartments promote approach, of PAM compartments avoidance); learning predicts the
      test - baseline change to be > 0 for A (sweet: PAM depresses avoid-MBONs) and < 0 for B
  (c) the output cells' membranes (the act head's input): the A-vs-B difference of the test - baseline change, against
      the spread over repeated presentations of the same odour (a d' over the output surface)
Usage: python memory_chain_probe.py <checkpoint dir> [out .jsonl]
"""
import sys, os, json, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])
from information_boltzmann.core import fly_odour_world as ow

fork = Path(sys.argv[1])
out = Path(sys.argv[2] if len(sys.argv) > 2 else 'results/published/fly_consequence_20261013/memory_chain_probe.jsonl')
out.parent.mkdir(parents=True, exist_ok=True)
model, cfg, L = load(fork)
import os
NO_CHEMO_WRITER = bool(os.environ.get('NO_CHEMO_WRITER'))
if NO_CHEMO_WRITER:
    # the clean bound: words are not written into the chemosensory cells
    with torch.no_grad():
        model.topographic_writer.proj_chemo.weight.zero_()
graph = ROOT / cfg['graph']; ann = graph.parent / 'body-annotations.feather'; dend = graph.parent / 'kc_kc_dendritic_synapses.feather'
if not cfg.get('kc_dendritic_only'):
    fl.attach_kenyon_dendritic_only(model, graph, ann, dend, rescale=True)
if not cfg.get('kc_renorm'):
    fl.renormalize_kenyon_inputs(model, graph, ann, dend)
KC_DAN_PIN = bool(os.environ.get('KC_DAN_PIN'))
if KC_DAN_PIN:
    print('Kenyon -> dopamine edges pinned:', fl.pin_kenyon_to_dopamine(model, graph, ann), flush=True)
if getattr(model, 'dopamine_sensitivity', None) is None:
    fl.attach_dopamine_modulation(model, site='pre')
if getattr(model, 'comp_pre', None) is None:
    fl.attach_compartment_memory(model, graph, ann, eta=-2.0, forget=0.02)
if getattr(model, 'taste_valence', None) is None:
    fl.attach_taste_heads(model, graph, ann)
if os.environ.get('AL_RESET'):
    print('antennal-lobe inputs reset to birth:', fl.reset_inputs_to_birth(model, ROOT / cfg['graph'], (ROOT / cfg['graph']).parent / 'body-annotations.feather'), flush=True)
if os.environ.get('CONNECTOME_V2'):
    # the full set; the Kenyon fixes above already ran in this script, so only the pins (idempotent) are added here
    print('connectome v2 pins:', {k: v for k, v in {
        'kenyon_to_dopamine': fl.pin_kenyon_to_dopamine(model, ROOT / cfg['graph'], (ROOT / cfg['graph']).parent / 'body-annotations.feather'),
        'sensory_terminals': fl.pin_sensory_terminals(model, ROOT / cfg['graph']),
        'octopamine_serotonin_fast': fl.pin_modulatory_fast_edges(model, ROOT / cfg['graph'], (ROOT / cfg['graph']).parent / 'modulatory_neurons.feather')}.items()}, flush=True)
model.eval().requires_grad_(False)
world = ow.OdourWorld(model, graph, ann, seed=0)
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
N = model.n_neurons; writer = model.topographic_writer; quiet = torch.zeros(1, N, device='cuda')
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r'); text = np.array(val[30_000:50_000], dtype=np.int64)
rng = np.random.default_rng(4)
DA_LOG = []
st = {'s': move_state(fl.FlyPhysicalState(**L['physical']), 'cuda'), 'i': 0, 'left': 0}
mbon, v = model.taste_mbon, model.taste_valence
read = model.read_indices


def tick(drive=None):
    s = st['s']; model.external_drive = drive
    if st['left'] == 0:
        st['left'] = int(rng.geometric(1 / 8))
        tok = torch.tensor([int(text[st['i'] % len(text)])], device='cuda'); st['i'] += 1
        src, base = fl.write_input(model, s, tok)
        s = fl.step_fly_physical_tick(model, s, tok, src, base, opts, base_rates=opts['base_rates'])
    else:
        base = writer.lambda_adapt * s.baseline if writer is not None else s.baseline
        s = fl.step_fly_physical_tick(model, s, None, quiet, base, opts, base_rates=opts['base_rates'])
    st['left'] -= 1; model.external_drive = None; st['s'] = s
    return s, (s.ring[0][0] > 0).float()


def current(cells, level):
    d = torch.zeros(1, N, device='cuda'); d[0, cells] = level * world.rheobase[cells]
    return d


def present(cells, outcome=None, ticks=24):
    """Matched control window, then the odour window; returns (dMBON rate [n_mbon], dOutput membrane [n_read], the
    cells that fired in the odour window but not in the control window)."""
    for _ in range(40):
        tick()
    acc_m = torch.zeros(mbon.numel(), device='cuda'); acc_o = torch.zeros(read.numel(), device='cuda')
    ctrl = torch.zeros(N, dtype=torch.bool, device='cuda')
    for _ in range(ticks):
        s, sp = tick(); acc_m -= sp[mbon]; acc_o -= s.h[0, read]; ctrl |= sp.bool()
    drove = torch.zeros(N, dtype=torch.bool, device='cuda')
    for t in range(ticks):
        d = current(cells, 32.0) if rng.random() < 0.5 else None
        if outcome is not None and 12 <= t < 16:
            d = (d if d is not None else 0) + current(world.taste_t[outcome], 8.0)
        if outcome is not None and 15 <= t < 19:
            d = (d if d is not None else 0) + current(world.dopamine_t[outcome], 8.0)
        s, sp = tick(d); acc_m += sp[mbon]; acc_o += s.h[0, read]; drove |= sp.bool()
        if s.dopamine.numel():
            dp = s.dopamine.index_select(1, mbon)
            burst = torch.relu(fl.dopamine_occupancy(dp[0], model.comp_affinity) - fl.dopamine_occupancy(dp[1], model.comp_affinity))
            DA_LOG.append(('paired' if outcome is not None else 'odour', float(burst.mean())))
    return acc_m / ticks, acc_o / ticks, drove & ~ctrl


rep = {'checkpoint': str(fork), 'no_chemo_writer': NO_CHEMO_WRITER, 'kc_dan_pin': KC_DAN_PIN, 'al_reset': bool(os.environ.get('AL_RESET')), 'connectome_v2': bool(os.environ.get('CONNECTOME_V2'))}
kc_all = torch.as_tensor(np.flatnonzero(pd.read_feather(ann).set_index('bodyId').reindex(np.load(graph, allow_pickle=True)['neuron_body_ids'])['class'].astype(str).values == 'Kenyon_Cell'), device='cuda')
gr = np.load(graph, allow_pickle=True); typ_all = pd.read_feather(ann).set_index('bodyId').reindex(gr['neuron_body_ids'])['type'].astype(str).values
with torch.no_grad():
    for _ in range(500):
        tick()
    types = rng.choice(world.pool, 16, replace=False)
    odour = {'A': torch.as_tensor(np.concatenate([world.receptor[x] for x in types[:8]]), device='cuda'),
             'B': torch.as_tensor(np.concatenate([world.receptor[x] for x in types[8:]]), device='cuda')}
    res = {'baseline': {'A': [], 'B': []}, 'test': {'A': [], 'B': []}}
    kc_mask, kc_sets = {}, {'A': [], 'B': []}
    for k in range(4):
        for o in 'AB':
            m, outp, drove = present(odour[o]); res['baseline'][o].append((m, outp))
            kc_sets[o].append(drove[kc_all])
            kc_mask[o] = drove if o not in kc_mask else (kc_mask[o] | drove)
    jac = lambda a, b: float((a & b).sum() / max(int((a | b).sum()), 1))
    rep['kc_odour_fraction'] = float(torch.stack([x.float().mean() for x in kc_sets['A'] + kc_sets['B']]).mean())
    rep['kc_jaccard_same_odour'] = float(np.mean([jac(kc_sets[o][i], kc_sets[o][j]) for o in 'AB' for i in range(4) for j in range(i + 1, 4)]))
    rep['kc_jaccard_A_vs_B'] = float(np.mean([jac(a, b) for a in kc_sets['A'] for b in kc_sets['B']]))
    phi0 = st['s'].comp_memory.clone()
    for k in range(3):
        present(odour['A'], 'sweet'); present(odour['B'], 'bitter')
    dphi = st['s'].comp_memory - phi0
    for k in range(4):
        for o in 'AB':
            res['test'][o].append(present(odour[o])[:2])
    edges_kc = model.comp_edge_kc
    dpre, dpost = gr['dan_edge_pre'], gr['dan_edge_post']
    for o, val_ in (('A', 'sweet'), ('B', 'bitter')):
        on = kc_mask[o][model.comp_kc][edges_kc]
        rep[f'dphi_{o}_edges'] = float(dphi[0, on].mean()) if bool(on.any()) else None
        dan_cells = world.dopamine[val_]
        reached = torch.zeros(N, dtype=torch.bool, device='cuda')
        reached[torch.as_tensor(np.unique(dpost[np.isin(dpre, dan_cells)]), device='cuda')] = True
        sel = on & reached[model.comp_post.long()]
        other = (~on) & reached[model.comp_post.long()]
        rep[f'dphi_{o}_edges_in_its_outcome_compartments'] = float(dphi[0, sel].mean()) if bool(sel.any()) else None
        rep[f'dphi_other_edges_in_{o}_outcome_compartments'] = float(dphi[0, other].mean()) if bool(other.any()) else None
    rep['dphi_all_edges'] = float(dphi.mean())
    for o in 'AB':
        b = torch.stack([m for m, _ in res['baseline'][o]]); t = torch.stack([m for m, _ in res['test'][o]])
        rep[f'mbon_{o}_valence_shift'] = float(((t.mean(0) - b.mean(0)) * v).sum())
        rep[f'mbon_{o}_change_norm'] = float((t.mean(0) - b.mean(0)).norm())
        rep[f'mbon_{o}_repeat_sd_norm'] = float(torch.cat([b, t]).std(0).norm() / 2 ** 0.5)
    ob = {o: torch.stack([x for _, x in res['baseline'][o]]) for o in 'AB'}
    ot = {o: torch.stack([x for _, x in res['test'][o]]) for o in 'AB'}
    diff = (ot['A'].mean(0) - ob['A'].mean(0)) - (ot['B'].mean(0) - ob['B'].mean(0))
    noise = torch.cat([ob['A'] - ob['A'].mean(0), ob['B'] - ob['B'].mean(0), ot['A'] - ot['A'].mean(0), ot['B'] - ot['B'].mean(0)])
    sd = noise.std(0).clamp_min(1e-6)
    rep['output_A_vs_B_dprime_rms'] = float(((diff / (sd * (2 / 4) ** 0.5)) ** 2).mean().sqrt())
    rep['output_A_vs_B_cells_over_2'] = int(((diff.abs() / (sd * (2 / 4) ** 0.5)) > 2).sum())
    rep['output_cells'] = int(read.numel())
for kind in ('odour', 'paired'):
    xs = [x for k, x in DA_LOG if k == kind]
    rep[f'da_burst_{kind}_mean_sd'] = [float(np.mean(xs)), float(np.std(xs))] if xs else None
print(json.dumps(rep, indent=1))
open(out, 'a').write(json.dumps(rep) + chr(10))
