"""How far is each neuron from the Markov blanket?  Graph only (CPU).
Hops from the sensory injection neurons (shortest directed path) and hops to the motor read neurons (shortest directed
path to any of them), over all synapses (excitatory and inhibitory), per neuron; summarized by class.  The classes'
activity (spikes per neuron per tick, reading vs silence; energy share) comes from self_model_probe.json."""
import json, sys, numpy as np, pandas as pd, collections
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from pathlib import Path
ROOT = Path('D:/information-boltzmann-v1')
g = np.load(ROOT / 'data/malecns_v1/fly_reservoir_coba.npz', allow_pickle=True)
N = len(g['neuron_body_ids'])
ann = pd.read_feather(ROOT / 'data/malecns_v1/body-annotations.feather').set_index('bodyId').reindex(g['neuron_body_ids'])
cls, sup = ann['class'].astype(str).values, ann['superclass'].astype(str).values
s = np.load(ROOT / 'data/malecns_v1/sensory_partitions.npz')
inj = np.unique(np.concatenate([s['visual_idx'], s['chemo_idx'], s['mechano_idx']]))
probe = json.load(open(ROOT / 'scratch/fly_stream/dasite_pre/self_model_probe.json'))
sys.path.insert(0, str(ROOT))
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
names = [str(x) for x in g['superclass_names']]
read = np.flatnonzero(np.isin(g['superclass_id'], [names.index(n) for n in FlyReservoirLM.OUTPUT_CLASSES if n in names]))
pre = np.concatenate([g['edge_pre_e'], g['edge_pre_i']]); post = np.concatenate([g['edge_post_e'], g['edge_post_i']])
# super-source / super-sink node N
A = csr_matrix((np.ones(len(pre) + len(inj)), (np.r_[pre, np.full(len(inj), N)], np.r_[post, inj])), shape=(N + 1, N + 1))
d_in = shortest_path(A, indices=N, unweighted=True, directed=True)[:N] - 1        # injection neurons: 0
B = csr_matrix((np.ones(len(pre) + len(read)), (np.r_[post, np.full(len(read), N)], np.r_[pre, read])), shape=(N + 1, N + 1))
d_out = shortest_path(B, indices=N, unweighted=True, directed=True)[:N] - 1       # read neurons: 0
core = sup == 'cb_intrinsic'
is_read = np.zeros(N, bool); is_read[read] = True
is_dan = np.zeros(N, bool); is_dan[g['dan_indices']] = True
groups = {'core: central complex': core & (cls == 'CX'), 'core: Kenyon cells': core & (cls == 'Kenyon_Cell'),
          'core: MB output neurons': core & (cls == 'MBON'), 'core: dopamine cells': is_dan,
          'core: other': core & ~np.isin(cls, ['CX', 'Kenyon_Cell', 'MBON', 'DAN']) & ~is_dan,
          'motor read surface': is_read, 'rest of the brain': ~core & ~is_read & ~is_dan}
fin = lambda x: x[np.isfinite(x)]
out = {}
for k, m in groups.items():
    a, b = d_in[m], d_out[m]
    act = probe['activity'].get(k, {})
    out[k] = {'neurons': int(m.sum()),
              'hops_from_sensory_median': float(np.median(fin(a))) if len(fin(a)) else None,
              'hops_from_sensory_mean': float(fin(a).mean()) if len(fin(a)) else None,
              'unreachable_from_sensory': float((~np.isfinite(a)).mean()),
              'hops_to_motor_median': float(np.median(fin(b))) if len(fin(b)) else None,
              'hops_to_motor_mean': float(fin(b).mean()) if len(fin(b)) else None,
              'cannot_reach_motor': float((~np.isfinite(b)).mean()),
              'rate_reading': act.get('spikes_per_neuron_tick_reading'), 'rate_silence': act.get('spikes_per_neuron_tick_silence'),
              'energy_share_silence': act.get('energy_share_silence')}
    print(f'{k:24s}', {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in out[k].items()})
both = d_in + d_out
dist = collections.Counter(np.where(np.isfinite(both), both, -1).astype(int))
print('blanket distance (hops in + hops out) over all neurons:', sorted(dist.items())[:12])
np.savez_compressed(ROOT / 'scratch/fly_stream/blanket_distance.npz', hops_from_sensory=d_in, hops_to_motor=d_out)
json.dump(out, open(ROOT / 'scratch/fly_stream/blanket_distance.json', 'w'), indent=1)
