"""Per connection a -> b on the route, the share of its synapses that land outside b's axonal territory (b's
output-enriched primary ROIs: those holding a larger share of b's output sites than of its input sites; the criterion
of compartment_audit.py, validated on KC -> KC 0.88 vs truth 0.97 and APL -> KC 0.73 vs 0.845).  For the general
compartment rule: every edge scaled by its dendritic share (the point model puts every synapse on the soma).
Writes data/malecns_v1/dendritic_share_<group>.feather (body_pre, body_post, total, dendritic) per target group.
neuPrint male-cns v1.0, server-side per pair; token from the environment, never printed; retries on network errors.
Usage: python neuprint_dendritic_share.py [group ...]   (default: every group below)
"""
import sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from neuprint_query import cypher

ROOT = Path('D:/information-boltzmann-v1')
GROUPS = {
    'Kenyon_Cell': "b.class = 'Kenyon_Cell' AND NOT coalesce(a.class, '') = 'Kenyon_Cell'",   # KC -> KC: own table
    'MBON': "b.class = 'MBON'", 'DAN': "b.class = 'DAN'", 'ALPN': "b.class = 'ALPN'",
    'descending_neuron': "b.superclass = 'descending_neuron'",
}
prim = cypher("MATCH (m:Meta) RETURN m.primaryRois")['data'][0][0]
prim_list = '[' + ','.join("'" + r.replace("'", "\\'") + "'" for r in prim) + ']'
for name in (sys.argv[1:] or list(GROUPS)):
    where = GROUPS[name]
    if name == 'Kenyon_Cell':
        # the Kenyon dendrites are exactly the calyx (CA, ACA): the anatomical truth, sharper than the ROI-enrichment
        # criterion (APL -> KC 0.845 axonal by calyx vs 0.73 by enrichment)
        q = f"""
MATCH (a:Neuron)-[c:ConnectsTo]->(b:Neuron) WHERE {where}
WITH a, b, c, apoc.convert.fromJsonMap(c.roiInfo) AS ci
RETURN a.bodyId AS pre, b.bodyId AS post, c.weight AS w, c.weight AS inroi,
       coalesce(ci['CA(L)'].post, 0) + coalesce(ci['CA(R)'].post, 0) + coalesce(ci['ACA(L)'].post, 0) + coalesce(ci['ACA(R)'].post, 0) AS dend"""
    else:
        q = f"""
MATCH (a:Neuron)-[c:ConnectsTo]->(b:Neuron) WHERE {where}
WITH a, b, c, apoc.convert.fromJsonMap(c.roiInfo) AS ci, apoc.convert.fromJsonMap(b.roiInfo) AS bi
UNWIND [r IN keys(ci) WHERE r IN {prim_list}] AS roi
WITH a.bodyId AS pre, b.bodyId AS post, c.weight AS w, coalesce(ci[roi].post, 0) AS n,
     toFloat(coalesce(bi[roi].pre, 0)) / CASE WHEN coalesce(b.pre, 0) > 0 THEN b.pre ELSE 1 END AS sp,
     toFloat(coalesce(bi[roi].post, 0)) / CASE WHEN coalesce(b.post, 0) > 0 THEN b.post ELSE 1 END AS sq
WITH pre, post, w, sum(n) AS inroi, sum(CASE WHEN sp > sq THEN 0 ELSE n END) AS dend
RETURN pre, post, w, inroi, dend"""
    for attempt in range(4):
        try:
            rows = cypher(q)['data']
            break
        except Exception as e:
            print(f'{name}: attempt {attempt + 1} failed ({type(e).__name__}); retrying', flush=True)
            time.sleep(20 * (attempt + 1))
    else:
        print(f'{name}: FAILED', flush=True)
        continue
    df = pd.DataFrame(rows, columns=['body_pre', 'body_post', 'total', 'in_primary', 'dendritic'])
    out = ROOT / f'data/malecns_v1/dendritic_share_{name}.feather'
    df.to_feather(out)
    share = df.dendritic.sum() / max(1, df.in_primary.sum())
    print(f'{name}: {len(df)} pairs, {int(df.total.sum())} synapses, dendritic share {share:.3f} -> {out.name}', flush=True)
