"""Kenyon -> Kenyon synapses by brain region from neuPrint male-cns v1.0 (server-side aggregation, ~0.6 MB): writes
data/malecns_v1/kc_kc_dendritic_synapses.feather.  Needs NEUPRINT_TOKEN in the environment (never stored)."""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from neuprint_query import cypher
# which primary regions hold the KC->KC synapses that are not in MB(L)/MB(R)
q = """
MATCH (a:Neuron)-[c:ConnectsTo]->(b:Neuron)
WHERE a.class = 'Kenyon_Cell' AND b.class = 'Kenyon_Cell'
WITH c, apoc.convert.fromJsonMap(c.roiInfo) AS info
UNWIND keys(info) AS roi
WITH roi, info[roi].post AS n WHERE roi IN ['ACA(L)','ACA(R)','INP','CentralBrain-unspecified','LH(L)','LH(R)','SNP(L)','SNP(R)','SLP(L)','SLP(R)','PLP(L)','PLP(R)']
RETURN roi, sum(n) ORDER BY sum(n) DESC
"""
for row in cypher(q)['data']:
    print(row)
# per pair: dendritic (calyx + accessory calyx) synapses and total
q = """
MATCH (a:Neuron)-[c:ConnectsTo]->(b:Neuron)
WHERE a.class = 'Kenyon_Cell' AND b.class = 'Kenyon_Cell'
WITH a, b, c, apoc.convert.fromJsonMap(c.roiInfo) AS info
WITH a.bodyId AS pre, b.bodyId AS post, c.weight AS w,
     coalesce(info['CA(L)'].post, 0) + coalesce(info['CA(R)'].post, 0) + coalesce(info['ACA(L)'].post, 0) + coalesce(info['ACA(R)'].post, 0) AS dendritic
WHERE dendritic > 0
RETURN pre, post, dendritic, w
"""
rows = cypher(q)['data']
df = pd.DataFrame(rows, columns=['body_pre', 'body_post', 'dendritic_synapses', 'total_synapses'])
out = 'data/malecns_v1/kc_kc_dendritic_synapses.feather'
df.to_feather(out)
print(len(df), 'pairs saved to', out, 'dendritic', int(df.dendritic_synapses.sum()), 'of their', int(df.total_synapses.sum()))
