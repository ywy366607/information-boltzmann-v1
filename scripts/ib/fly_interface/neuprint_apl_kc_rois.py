"""APL <-> Kenyon synapses per pair, calyx (CA + ACA: Kenyon dendrites) vs elsewhere (lobes / peduncle: Kenyon axons),
from neuPrint male-cns v1.0 (server-side, ~0.3 MB): writes data/malecns_v1/apl_kc_dendritic_synapses.feather with
columns body_pre, body_post, dendritic_synapses, total_synapses for APL -> KC and KC -> APL.  In the point model every
APL -> KC synapse acts on the Kenyon membrane, but most sit on the axons (mb_feedback_audit: ~17% of APL -> KC in the
calyx).  Needs NEUPRINT_TOKEN or NEUPRINT_APPLICATION_CREDENTIALS in the environment (never printed or stored)."""
import sys
import pandas as pd
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
from neuprint_query import cypher
frames = []
for wa, wb in (("a.type = 'APL'", "b.class = 'Kenyon_Cell'"), ("a.class = 'Kenyon_Cell'", "b.type = 'APL'")):
    q = f"""
MATCH (a:Neuron)-[c:ConnectsTo]->(b:Neuron) WHERE {wa} AND {wb}
WITH a, b, c, apoc.convert.fromJsonMap(c.roiInfo) AS info
RETURN a.bodyId AS pre, b.bodyId AS post, c.weight AS w,
       coalesce(info['CA(L)'].post, 0) + coalesce(info['CA(R)'].post, 0) + coalesce(info['ACA(L)'].post, 0) + coalesce(info['ACA(R)'].post, 0) AS dendritic
"""
    rows = cypher(q)['data']
    frames.append(pd.DataFrame(rows, columns=['body_pre', 'body_post', 'total_synapses', 'dendritic_synapses']))
df = pd.concat(frames, ignore_index=True)
out = 'data/malecns_v1/apl_kc_dendritic_synapses.feather'
df.to_feather(out)
print(len(df), 'pairs ->', out)
for name, sel in (('APL -> KC', df.body_pre.isin([10540, 10977])), ('KC -> APL', df.body_post.isin([10540, 10977]))):
    d = df[sel]
    print(f'{name}: {int(d.dendritic_synapses.sum())} of {int(d.total_synapses.sum())} in the calyx '
          f'({d.dendritic_synapses.sum() / d.total_synapses.sum():.3f}); per pair calyx fraction median '
          f'{(d.dendritic_synapses / d.total_synapses).median():.3f}')
