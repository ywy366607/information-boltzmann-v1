"""Systematic compartment audit (user, 2026-10-14: check every circuit we use, after the APL -> Kenyon artifact).
The point model puts every synapse on the soma.  For each target group on our route (antennal lobe, mushroom body,
the read surface: descending / motor / endocrine), and each presynaptic group, the share of synapses that land in the
target cell's axonal territory: a primary ROI that holds a larger share of that very cell's output sites than of its
input sites (output-enriched: 'axonal' when the output share is > 1.5x the input share, 'mixed' when > 1x).  (A first
version asked for more output than input sites in the ROI; Kenyon axons, which receive heavy axo-axonal input, failed
it, so it found nothing in the mushroom body.)  Inputs there are axo-axonal (presynaptic modulation of the output) and
should not drive the soma.  Server-side sums from neuPrint male-cns v1.0 (a few kB per group; token from the
environment, never printed).
Usage: python compartment_audit.py [out.json]
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from neuprint_query import cypher

prim = cypher("MATCH (m:Meta) RETURN m.primaryRois")['data'][0][0]
prim_list = '[' + ','.join("'" + r.replace("'", "\\'") + "'" for r in prim) + ']'
GROUPS = {
    'Kenyon_Cell': "b.class = 'Kenyon_Cell'", 'MBON': "b.class = 'MBON'", 'DAN': "b.class = 'DAN'",
    'APL': "b.type = 'APL'", 'DPM': "b.type = 'DPM'", 'ALPN': "b.class = 'ALPN'", 'ALLN': "b.class = 'ALLN'",
    'descending_neuron': "b.superclass = 'descending_neuron'", 'cb_motor': "b.superclass = 'cb_motor'",
    'vnc_motor': "b.superclass = 'vnc_motor'", 'cb_endocrine': "b.superclass = 'cb_endocrine'",
}
SRC = "CASE WHEN a.type IN ['APL', 'DPM'] THEN a.type ELSE coalesce(a.class, a.superclass, 'none') END"
report = {'primary_rois': len(prim)}
for name, where in GROUPS.items():
    q = f"""
MATCH (a:Neuron)-[c:ConnectsTo]->(b:Neuron) WHERE {where}
WITH a, b, apoc.convert.fromJsonMap(c.roiInfo) AS ci, apoc.convert.fromJsonMap(b.roiInfo) AS bi
UNWIND [r IN keys(ci) WHERE r IN {prim_list}] AS roi
WITH {SRC} AS src, coalesce(ci[roi].post, 0) AS n,
     toFloat(coalesce(bi[roi].pre, 0)) / CASE WHEN coalesce(b.pre, 0) > 0 THEN b.pre ELSE 1 END AS sp,
     toFloat(coalesce(bi[roi].post, 0)) / CASE WHEN coalesce(b.post, 0) > 0 THEN b.post ELSE 1 END AS sq
RETURN src, sum(n) AS total, sum(CASE WHEN sp > 1.5 * sq THEN n ELSE 0 END) AS axonal,
       sum(CASE WHEN sp > sq THEN n ELSE 0 END) AS mixed
ORDER BY total DESC"""
    try:
        rows = cypher(q)['data']
    except Exception as e:
        report[name] = f'query failed: {type(e).__name__}'
        print(name, report[name], flush=True)
        continue
    tot = sum(r[1] for r in rows) or 1
    ax = sum(r[2] for r in rows); mx = sum(r[3] for r in rows)
    report[name] = {'input_synapses': tot, 'axonal_share': round(ax / tot, 4), 'mixed_share': round(mx / tot, 4),
                    'by_source': {r[0]: {'share_of_input': round(r[1] / tot, 4), 'axonal': round(r[2] / max(1, r[1]), 3),
                                         'mixed': round(r[3] / max(1, r[1]), 3)} for r in rows[:12]}}
    print(f"{name}: {tot} input synapses, axonal {ax / tot:.3f}, mixed {mx / tot:.3f}", flush=True)
    for r in rows[:12]:
        print(f"    from {r[0]:<24} {r[1] / tot:6.3f} of input; axonal {r[2] / max(1, r[1]):.3f}, mixed {r[3] / max(1, r[1]):.3f}", flush=True)
if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(json.dumps(report, indent=1), encoding='utf-8')
