"""Neuron sizes from neuPrint male-cns v1.0 (voxel count, n.size) for every body of the graph, saved to
data/malecns_v1/neuron_sizes.feather (bodyId, size).  Used for the per-cell synaptic efficacy (input resistance falls
with membrane area: a small cell such as a Kenyon cell is depolarized more by one synapse).
Usage: python neuprint_sizes.py [keys]   ('keys' prints the property names of one Kenyon cell node and exits)"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, 'D:/information-boltzmann-v1/scripts/ib/fly_interface')
from neuprint_query import cypher
if len(sys.argv) > 1 and sys.argv[1] == 'keys':
    r = cypher("MATCH (n:Neuron) WHERE n.class = 'Kenyon_Cell' RETURN keys(n) LIMIT 1")
    print(sorted(r['data'][0][0]))
    sys.exit()
ids = np.load('D:/information-boltzmann-v1/data/malecns_v1/fly_reservoir_coba_shiu.npz')['neuron_body_ids']
rows = []
for k in range(0, len(ids), 20000):
    chunk = ','.join(str(int(b)) for b in ids[k:k + 20000])
    r = cypher(f"MATCH (n:Neuron) WHERE n.bodyId IN [{chunk}] RETURN n.bodyId, n.size")
    rows += r['data']
df = pd.DataFrame(rows, columns=['bodyId', 'size'])
print(len(df), 'of', len(ids), 'bodies; size quantiles', df['size'].quantile([0, .1, .5, .9, 1]).tolist())
df.to_feather('D:/information-boltzmann-v1/data/malecns_v1/neuron_sizes.feather')
