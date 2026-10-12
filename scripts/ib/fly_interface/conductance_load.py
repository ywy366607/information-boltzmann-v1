"""Operating point of each cell class at the saved living state: total conductance over leak conductance
(g_total / g_L; > 1 = shunted, every extra input counts less), the ALIF threshold increment beta_a * b_alif in units of
the threshold, the homeostatic drive (rheobase units) and the distance to threshold.  Fixed weights; no ticks run.
Usage: python conductance_load.py <fork dir> [...]"""
import sys, json, numpy as np, pandas as pd, torch
from pathlib import Path
sys.path.insert(0, 'D:/information-boltzmann-v1')
exec(open('D:/information-boltzmann-v1/scratch/fly_stream/world_eval.py', encoding='utf-8').read().split('def evaluate(fork):')[0])
out = {}
for fork in map(Path, sys.argv[1:]):
    model, cfg, L = load(fork)
    P = L['physical']; N = model.n_neurons
    graph = ROOT / cfg['graph']; g = np.load(graph, allow_pickle=True)
    ann = pd.read_feather(graph.parent / 'body-annotations.feather').set_index('bodyId').reindex(g['neuron_body_ids'])
    cls, sup = ann['class'].astype(str).values, ann['superclass'].astype(str).values
    is_dan = np.zeros(N, bool); is_dan[g['dan_indices']] = True
    groups = {'olfactory receptor': cls == 'olfactory', 'ALPN': cls == 'ALPN', 'Kenyon': cls == 'Kenyon_Cell', 'MBON': cls == 'MBON',
              'dopamine': is_dan, 'core': sup == 'cb_intrinsic', 'all': np.ones(N, bool)}
    with torch.no_grad():
        leak_m, leak_se, leak_si = model.get_decay_rates()
        g_e, g_i = model.get_conductance_gains()
        ge, gi, h = (P[k].float().cuda().reshape(1, -1) for k in ('ge', 'gi', 'h'))
        G_E, G_I = (g_e * ge).expand(1, N), (g_i * gi).expand(1, N)
        g_L = -torch.log(leak_m.clamp(1e-5, 1 - 1e-7)).expand(1, N)
        load_ = ((g_L + G_E + G_I) / g_L)[0].cpu().numpy()
        theta = model.get_thresholds().expand(1, N)[0]
        rho_a, beta_a = model.get_alif_params()
        alif = (beta_a * P['b'].float().cuda().reshape(1, -1)).expand(1, N)[0] / theta
        hb = P.get('homeostasis')
        hb = hb.reshape(-1).float().numpy() if hb is not None and hb.numel() == N else np.zeros(N)
        dist = ((theta - h[0]) / theta.abs()).cpu().numpy()
        alif = alif.cpu().numpy()
    rep = {}
    for k, m in groups.items():
        q = lambda x: [round(float(v), 3) for v in np.percentile(x[m], [10, 50, 90])]
        rep[k] = {'g_total/g_L p10/50/90': q(load_), 'ALIF threshold increment / theta': q(alif), 'homeostatic b': q(hb),
                  'distance to threshold / theta': q(dist)}
        print(fork.name, f'{k:18s}', rep[k], flush=True)
    out[fork.name] = rep
    del model; torch.cuda.empty_cache()
json.dump(out, open('scratch/fly_stream/conductance_load.json', 'w'), indent=1)
