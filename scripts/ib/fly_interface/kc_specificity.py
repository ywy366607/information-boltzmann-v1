"""Is the Kenyon odour code specific?  The adult with the Kenyon fixes and Kenyon -> dopamine pinned; fixed weights.
Two odours A, B (8 receptor types each, whiffs p = 0.5 for 24 ticks), each presented 8 times, each presentation with a
matched control window just before it.  Per Kenyon cell: P(fires in an A window), P(fires in a control window).
  reliable responders of A: fire in >= 6/8 A windows and <= 2/8 controls (likewise B); their overlap
  population d': mean over KCs of (P_A - P_B)^2 / pooled Bernoulli variance -> how separable A and B are
Env: NO_TEXT=1 no words at all (diagnostic upper bound), NO_CHEMO_WRITER=1, LEVEL=<rheobase units> (default 32).
Usage: python kc_specificity.py <checkpoint dir> [out .jsonl]
"""
import sys, os, json, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])
from information_boltzmann.core import fly_odour_world as ow

fork = Path(sys.argv[1])
out = Path(sys.argv[2] if len(sys.argv) > 2 else 'results/published/fly_consequence_20261013/kc_specificity.jsonl')
NO_TEXT, NO_CHEMO = bool(os.environ.get('NO_TEXT')), bool(os.environ.get('NO_CHEMO_WRITER'))
LEVEL = float(os.environ.get('LEVEL', '32'))
model, cfg, L = load(fork)
graph = ROOT / cfg['graph']; ann = graph.parent / 'body-annotations.feather'; dend = graph.parent / 'kc_kc_dendritic_synapses.feather'
if not cfg.get('kc_dendritic_only'):
    fl.attach_kenyon_dendritic_only(model, graph, ann, dend, rescale=True)
if not cfg.get('kc_renorm'):
    fl.renormalize_kenyon_inputs(model, graph, ann, dend)
fl.pin_kenyon_to_dopamine(model, graph, ann)
if NO_CHEMO:
    with torch.no_grad():
        model.topographic_writer.proj_chemo.weight.zero_()
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
g = np.load(graph, allow_pickle=True)
cls = pd.read_feather(ann).set_index('bodyId').reindex(g['neuron_body_ids'])['class'].astype(str).values
kc = torch.as_tensor(np.flatnonzero(cls == 'Kenyon_Cell'), device='cuda')
pn = torch.as_tensor(np.flatnonzero(cls == 'ALPN'), device='cuda')
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r'); text = np.array(val[20_000:40_000], dtype=np.int64)
rng = np.random.default_rng(9)
st = {'s': move_state(fl.FlyPhysicalState(**L['physical']), 'cuda'), 'i': 0, 'left': 0}


def tick(drive=None):
    s = st['s']; model.external_drive = drive
    if st['left'] == 0 and not NO_TEXT:
        st['left'] = int(rng.geometric(1 / 8))
        tok = torch.tensor([int(text[st['i'] % len(text)])], device='cuda'); st['i'] += 1
        src, base = fl.write_input(model, s, tok)
        s = fl.step_fly_physical_tick(model, s, tok, src, base, opts, base_rates=opts['base_rates'])
    else:
        base = writer.lambda_adapt * s.baseline if writer is not None else s.baseline
        s = fl.step_fly_physical_tick(model, s, None, quiet, base, opts, base_rates=opts['base_rates'])
    st['left'] = max(0, st['left'] - 1); model.external_drive = None; st['s'] = s
    return (s.ring[0][0] > 0)


with torch.no_grad():
    for _ in range(500):
        tick()
    types = rng.choice(world.pool, 16, replace=False)
    cells = {'A': torch.as_tensor(np.concatenate([world.receptor[x] for x in types[:8]]), device='cuda'),
             'B': torch.as_tensor(np.concatenate([world.receptor[x] for x in types[8:]]), device='cuda')}
    fired = {'A': [], 'B': [], 'ctrl': []}; pn_rate = {'A': [], 'B': [], 'ctrl': []}
    for k in range(8):
        for o in 'AB':
            for _ in range(40):
                tick()
            c = torch.zeros(N, dtype=torch.bool, device='cuda'); pc = 0.0
            for _ in range(24):
                sp = tick(); c |= sp; pc += float(sp[pn].float().mean())
            fired['ctrl'].append(c[kc]); pn_rate['ctrl'].append(pc / 24)
            f = torch.zeros(N, dtype=torch.bool, device='cuda'); pf = 0.0
            for _ in range(24):
                d = None
                if rng.random() < 0.5:
                    d = torch.zeros(1, N, device='cuda'); d[0, cells[o]] = LEVEL * world.rheobase[cells[o]]
                sp = tick(d); f |= sp; pf += float(sp[pn].float().mean())
            fired[o].append(f[kc]); pn_rate[o].append(pf / 24)
    P = {k: torch.stack(v).float().mean(0) for k, v in fired.items()}
    rel = {o: (P[o] >= 6 / 8) & (P['ctrl'] <= 2 / 8) for o in 'AB'}
    pooled = ((P['A'] + P['B']) / 2).clamp(1e-3, 1 - 1e-3)
    dprime = ((P['A'] - P['B']) ** 2 / (pooled * (1 - pooled) * 2 / 8)).mean()
    rep = {'checkpoint': str(fork), 'al_reset': bool(os.environ.get('AL_RESET')), 'connectome_v2': bool(os.environ.get('CONNECTOME_V2')), 'no_text': NO_TEXT, 'no_chemo_writer': NO_CHEMO, 'level': LEVEL,
           'kc_fire_window_A_B_ctrl': [float(P[k].mean()) for k in ('A', 'B', 'ctrl')],
           'reliable_A': int(rel['A'].sum()), 'reliable_B': int(rel['B'].sum()), 'reliable_overlap': int((rel['A'] & rel['B']).sum()),
           'kc_total': int(kc.numel()), 'population_dprime_mean': float(dprime),
           'pn_rate_A_B_ctrl': [float(np.mean(pn_rate[k])) for k in ('A', 'B', 'ctrl')]}
print(json.dumps(rep), flush=True)
open(out, 'a').write(json.dumps(rep) + chr(10))
