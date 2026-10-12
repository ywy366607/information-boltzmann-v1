"""Where does an odour stop?  Fixed weights, text streaming at mean 8 ticks/word (pulses).
1  Descending neurons: each one's firing per tick during text; the act pools and the DNs with > 1% MBON input.
2  Odour chain: the receptor cells of K glomeruli (K = 2, 6) driven for 8 consecutive ticks at L rheobase (L = 4, 16);
   the spike-rate change (driven minus matched undriven window, 12 ticks from onset) of the driven ORNs, the PNs of
   those glomeruli, all ALPNs, the Kenyon cells and APL; also the conductance the Kenyon cells receive from PNs.
Usage: python odour_chain.py <checkpoint dir> [out .json]
"""
import sys, json, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scripts/ib/fly_interface/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])
from information_boltzmann.core import fly_odour_world as ow
POOL12 = ('DM3', 'DM4', 'DM6', 'DL5', 'VA3', 'VA4', 'VA5', 'VM2', 'VM3', 'DC1', 'DC2', 'DC4')

fork = Path(sys.argv[1])
out = Path(sys.argv[2] if len(sys.argv) > 2 else 'results/published/fly_consequence_20261012/odour_chain.json')
model, cfg, L = load(fork)
import os
if os.environ.get('NO_CHEMO_WRITER'):
    # words are seen and heard, not smelled or tasted: the writer no longer drives the chemosensory cells
    with torch.no_grad():
        model.topographic_writer.proj_chemo.weight.zero_()
        if getattr(model.topographic_writer.proj_chemo, 'bias', None) is not None:
            model.topographic_writer.proj_chemo.bias.zero_()
graph = ROOT / cfg['graph']; ann_path = graph.parent / 'body-annotations.feather'
g = np.load(graph, allow_pickle=True)
ann = pd.read_feather(ann_path).set_index('bodyId').reindex(g['neuron_body_ids'])
typ, cls = ann['type'].astype(str).values, ann['class'].astype(str).values
sup = np.array(list(g['superclass_names']))[g['superclass_id']]
N = model.n_neurons
opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(), conductance_gains=model.get_conductance_gains(),
            alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
with torch.no_grad():
    rates0 = model.get_decay_rates()[0].expand(1, N)
    rheobase = (model.get_thresholds().expand(1, N) * -torch.log(rates0.clamp(1e-5, 1 - 1e-7)))[0].float()
writer = model.topographic_writer
quiet = torch.zeros(1, N, device='cuda')
val = np.load(ROOT / cfg['data'] / 'validation.npy', mmap_mode='r')
text = np.array(val[80_000:95_000], dtype=np.int64)
rng = np.random.default_rng(1)
st = {'s': move_state(fl.FlyPhysicalState(**L['physical']), 'cuda'), 'i': 0, 'left': 0}


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


SENSORY_PIN = bool(os.environ.get('SENSORY_PIN'))
if SENSORY_PIN:
    # option A of the sensory audit: the connectome's synapses onto sensory axon terminals do not act on the membrane
    _se, _si = fl.sensory_axonal_edges(model, graph)
    with torch.no_grad():
        model.edge_weight_e[_se] = 0.0; model.edge_weight_i[_si] = 0.0
rep = {'checkpoint': str(fork), 'sensory_pin': SENSORY_PIN, 'no_chemo_writer': bool(os.environ.get('NO_CHEMO_WRITER'))}
with torch.no_grad():
    for _ in range(500):
        tick()
    # 1 descending neurons
    dn = np.flatnonzero(sup == 'descending_neuron')
    acc = torch.zeros(N, device='cuda'); T = 4000
    for _ in range(T):
        acc += tick()[1]
    r = (acc / T).cpu().numpy()
    paths = json.load(open(ROOT / 'results/published/fly_consequence_20261012/mbon_paths.json'))
    mbon_dn_types = sorted({d['type'] for d in paths['top_DN']})
    rep['DN_rate_quantiles'] = {q: float(np.quantile(r[dn], q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
    rep['DN_fraction_below_5pct'] = float((r[dn] < 0.05).mean())
    rep['read_surface_rate_median'] = float(np.median(r[model.read_indices.cpu().numpy()]))
    rep['pool_cells'] = {t: [round(float(x), 3) for x in r[typ == t]] for t in ('DNa02', 'DNa03', 'DNp09', 'MDN', 'DNp42')}
    rep['mbon_receiving_DN_rates'] = {t: [round(float(x), 3) for x in r[typ == t]] for t in mbon_dn_types}
    quiet_dn = dn[r[dn] < 0.05]
    rep['quiet_DN_types_sample'] = sorted(set(typ[quiet_dn]))[:60]
    print(json.dumps({k: rep[k] for k in ('DN_rate_quantiles', 'DN_fraction_below_5pct', 'read_surface_rate_median', 'pool_cells')}), flush=True)
    print('MBON-receiving DNs', rep['mbon_receiving_DN_rates'], flush=True)
    # 2 odour chain
    alpn = cls == 'ALPN'; kc = cls == 'Kenyon_Cell'; apl = typ == 'APL'
    chain = {}
    for K in (2, 6):
        for level in (4.0, 16.0):
            rows = []
            for trial in range(8):
                gl = list(rng.choice(POOL12, K, replace=False))
                orn = np.concatenate([np.flatnonzero(typ == f'ORN_{x}') for x in gl])
                pn = np.flatnonzero(alpn & np.array([any(t.startswith(x + '_') for x in gl) for t in typ]))
                groups = {'ORN': orn, 'PN_matched': pn, 'ALPN_all': np.flatnonzero(alpn), 'KC': np.flatnonzero(kc), 'APL': np.flatnonzero(apl)}
                drive = torch.zeros(1, N, device='cuda'); idx = torch.as_tensor(orn, device='cuda')
                drive[0, idx] = level * rheobase[idx]
                res = {}
                for phase in ('control', 'odour'):
                    for _ in range(60):
                        tick()
                    spk = torch.zeros(N, device='cuda'); union = torch.zeros(N, device='cuda')
                    for t in range(12):
                        _, sp = tick(drive if (phase == 'odour' and t < 8) else None)
                        spk += sp; union = torch.maximum(union, sp)
                    res[phase] = {k: float(spk[torch.as_tensor(v, device='cuda')].mean() / 12) if len(v) else None for k, v in groups.items()}
                    res[phase]['KC_any'] = float(union[torch.as_tensor(groups['KC'], device='cuda')].mean())
                rows.append({k: (res['odour'][k] - res['control'][k]) if res['odour'][k] is not None else None for k in res['odour']}
                            | {'n_PN': int(len(pn)), 'control_ORN': res['control']['ORN'], 'control_PN': res['control']['PN_matched']})
            summ = {k: float(np.mean([x[k] for x in rows if x[k] is not None])) for k in rows[0]}
            chain[f'K{K}_L{int(level)}'] = summ
            print('chain', K, level, {k: round(v, 4) for k, v in summ.items()}, flush=True)
    rep['odour_chain'] = chain
out.write_text(json.dumps(rep, indent=1))
print('saved', out)
