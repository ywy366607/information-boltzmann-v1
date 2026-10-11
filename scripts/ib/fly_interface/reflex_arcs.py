"""Which read neurons are one synapse from the sensory injection surface (reflex-like arcs), by superclass."""
import sys, numpy as np, scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
sys.path.insert(0, 'D:/information-boltzmann-v1')
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
g = np.load('D:/information-boltzmann-v1/data/malecns_v1/fly_reservoir_coba.npz')
m = FlyReservoirLM('D:/information-boltzmann-v1/data/malecns_v1/fly_reservoir_coba.npz', d_model=768,
                   injection='topographic', read_surface='output', synapse_model='coba', use_alif=True, use_stp=True)
inj, read, N = m.topographic_writer.injection_index.numpy(), m.read_indices.numpy(), m.n_neurons
names = [str(x) for x in g['superclass_names']]; sc = g['superclass_id']
pre = np.concatenate([g['edge_pre_e'], g['edge_pre_i']]); post = np.concatenate([g['edge_post_e'], g['edge_post_i']])
delay = np.concatenate([g['edge_delay_e'], g['edge_delay_i']]).astype(float)
A = sp.csr_matrix((delay, (pre, post)), shape=(N, N))
d = dijkstra(A, directed=True, indices=inj, min_only=True)[read]
print('injection classes:', {names[k]: int((sc[inj] == k).sum()) for k in np.unique(sc[inj])})
print(f'{"read superclass":24s} {"n":>5s} {"d=1":>5s} {"d=2":>5s} {"d=3":>4s} {"inf":>4s}')
for k in np.unique(sc[read]):
    sel = sc[read] == k
    dd = d[sel]
    print(f'{names[k]:24s} {sel.sum():5d} {(dd == 1).sum():5d} {(dd == 2).sum():5d} {(dd == 3).sum():4d} {np.isinf(dd).sum():4d}')
