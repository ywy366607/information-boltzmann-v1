"""Neuromodulatory neurons of MaleCNS v1.0 from neuPrint: every neuron whose consensus transmitter is octopamine,
serotonin or dopamine (and, for reference, the cell-type and the neuron-level predictions).  Written to
data/malecns_v1/modulatory_neurons.feather (bodyId, type, consensusNt, predictedNt, celltypePredictedNt).
Token from NEUPRINT_TOKEN or NEUPRINT_APPLICATION_CREDENTIALS (never printed or stored).
Usage: python neuprint_modulatory.py
"""
import sys, pandas as pd
sys.path.insert(0, 'D:/information-boltzmann-v1/scripts/ib/fly_interface')
from neuprint_query import cypher

q = ("MATCH (n:Neuron) WHERE n.consensusNt IN ['octopamine', 'serotonin', 'dopamine'] "
     "RETURN n.bodyId, n.type, n.consensusNt, n.predictedNt, n.celltypePredictedNt")
r = cypher(q)
df = pd.DataFrame(r['data'], columns=['bodyId', 'type', 'consensusNt', 'predictedNt', 'celltypePredictedNt'])
out = 'D:/information-boltzmann-v1/data/malecns_v1/modulatory_neurons.feather'
df.to_feather(out)
print(df['consensusNt'].value_counts().to_dict())
print(df[df.consensusNt != 'dopamine'].groupby('consensusNt')['type'].apply(lambda s: sorted(set(map(str, s)))[:25]).to_dict())
print('saved', out, len(df))
