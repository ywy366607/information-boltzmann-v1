"""Static connectome audit (no simulation): what could carry 'explaining away' (iP-VAE, arXiv:2410.19315) into the
Kenyon layer, i.e. a pattern-specific subtraction of what is already predicted?  User question (2026-10-14): could the
free-energy route be built from the fly's own mechanisms instead of a JEPA loss?

Offline, from data/malecns_v1/connectome-weights.feather (raw synapse counts, streamed in Arrow batches):
1. the Kenyon cells' input synapses by presynaptic class / type (PN, APL, DPM, Kenyon, DAN, MBON, other);
2. APL's input and output by class; per Kenyon cell, APL -> KC synapses vs KC -> APL synapses (reciprocity);
3. Kenyon -> Kenyon: total vs dendritic (calyx), from kc_kc_dendritic_synapses.feather (neuPrint ROI split);
4. feedback into the projection neurons (ALPN inputs from Kenyon / APL / MBON / DPM / LH types);
5. the iP-VAE overlap term Phi^T Phi: for random Kenyon pairs, the overlap of their PN input vectors, and whether a
   pair-specific inhibitory path exists (it can only run through APL / DPM: Kenyon cells are cholinergic).
Online (neuPrint, server-side sums, a few kB; token from the environment, never printed): APL <-> Kenyon and
APL -> ALPN synapses by region, calyx (CA, ACA: Kenyon dendrites, PN boutons) vs lobes / peduncle (Kenyon axons).
Usage: python mb_feedback_audit.py [out.json] [--no-network]
"""
import sys, json
import numpy as np, pandas as pd, pyarrow as pa
from pathlib import Path
ROOT = Path('D:/information-boltzmann-v1')
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else None
NET = '--no-network' not in sys.argv

ann = pd.read_feather(ROOT / 'data/malecns_v1/body-annotations.feather')
cls = dict(zip(ann.bodyId, ann['class'].astype(str)))
typ = dict(zip(ann.bodyId, ann['type'].astype(str)))
sub = dict(zip(ann.bodyId, ann['subclass'].astype(str)))
kc = set(ann.bodyId[ann['class'] == 'Kenyon_Cell'])
apl = set(ann.bodyId[ann['type'] == 'APL']); dpm = set(ann.bodyId[ann['type'] == 'DPM'])
alpn = set(ann.bodyId[ann['class'] == 'ALPN'])
watch_post = kc | apl | dpm | alpn
watch_pre = apl | kc | dpm
kc_arr, wpost, wpre = (np.fromiter(s, dtype=np.int64) for s in (kc, watch_post, watch_pre))

rows = []
with pa.memory_map(str(ROOT / 'data/malecns_v1/connectome-weights.feather')) as src:
    reader = pa.ipc.open_file(src)
    for i in range(reader.num_record_batches):
        b = reader.get_batch(i)
        pre = b.column('body_pre').to_numpy(); post = b.column('body_post').to_numpy(); w = b.column('weight').to_numpy()
        keep = np.isin(post, wpost) | np.isin(pre, wpre)
        if keep.any():
            rows.append(pd.DataFrame({'pre': pre[keep], 'post': post[keep], 'w': w[keep]}))
E = pd.concat(rows, ignore_index=True)
E['pre_class'] = E.pre.map(cls).fillna('?'); E['pre_type'] = E.pre.map(typ).fillna('?')
E['post_class'] = E.post.map(cls).fillna('?'); E['post_type'] = E.post.map(typ).fillna('?')


def group(t, c):
    if t == 'APL': return 'APL'
    if t == 'DPM': return 'DPM'
    return c


E['pre_g'] = [group(t, c) for t, c in zip(E.pre_type, E.pre_class)]
E['post_g'] = [group(t, c) for t, c in zip(E.post_type, E.post_class)]
rep = {}
q = lambda v: [round(float(x), 3) for x in (np.min(v), np.median(v), np.mean(v), np.max(v))] if len(v) else None

# 1. Kenyon inputs
into_kc = E[E.post.isin(kc)]
tot = into_kc.w.sum()
by = into_kc.groupby('pre_g').w.sum().sort_values(ascending=False)
rep['kenyon_input_share_by_class'] = {k: round(float(v / tot), 4) for k, v in by.head(12).items()}
other = into_kc[~into_kc.pre_g.isin(['ALPN', 'APL', 'DPM', 'Kenyon_Cell', 'DAN', 'MBON'])]
rep['kenyon_input_other_top_types'] = {k: int(v) for k, v in other.groupby('pre_type').w.sum().sort_values(ascending=False).head(12).items()}
per_kc = into_kc.pivot_table(index='post', columns='pre_g', values='w', aggfunc='sum', fill_value=0)
for g in ('ALPN', 'APL', 'DPM', 'Kenyon_Cell', 'DAN', 'MBON'):
    if g in per_kc:
        rep[f'per_kenyon_synapses_from_{g}'] = q(per_kc[g].values)

# 2. APL
from_apl = E[E.pre.isin(apl)]; into_apl = E[E.post.isin(apl)]
rep['apl_output_share_by_class'] = {k: round(float(v / from_apl.w.sum()), 4) for k, v in from_apl.groupby('post_g').w.sum().sort_values(ascending=False).head(8).items()}
rep['apl_input_share_by_class'] = {k: round(float(v / into_apl.w.sum()), 4) for k, v in into_apl.groupby('pre_g').w.sum().sort_values(ascending=False).head(8).items()}
a2k = from_apl[from_apl.post.isin(kc)].groupby('post').w.sum()
k2a = into_apl[into_apl.pre.isin(kc)].groupby('pre').w.sum()
both = pd.concat([a2k.rename('apl_to_kc'), k2a.rename('kc_to_apl')], axis=1).fillna(0).reindex(kc_arr).fillna(0)
rep['apl_to_kc_per_kenyon'] = q(both.apl_to_kc.values); rep['kc_to_apl_per_kenyon'] = q(both.kc_to_apl.values)
rep['rho_apl_to_kc_vs_kc_to_apl'] = round(float(both.rank().corr().iloc[0, 1]), 3)
pn_in = per_kc['ALPN'].reindex(kc_arr).fillna(0) if 'ALPN' in per_kc else None
if pn_in is not None:
    rep['rho_apl_to_kc_vs_pn_input'] = round(float(pd.concat([both.apl_to_kc, pn_in], axis=1).rank().corr().iloc[0, 1]), 3)
# by Kenyon subtype: is APL output spread evenly over the lobes' cell types?
kt = pd.Series({b: typ[b] for b in kc_arr})
rep['apl_to_kc_by_kenyon_type_median'] = {t: float(both.apl_to_kc[kt[kt == t].index].median()) for t in kt.value_counts().index[:9]}

# 3. Kenyon -> Kenyon
kk = E[E.pre.isin(kc) & E.post.isin(kc)]
rep['kc_kc_synapses_total'] = int(kk.w.sum())
dend = pd.read_feather(ROOT / 'data/malecns_v1/kc_kc_dendritic_synapses.feather')
rep['kc_kc_dendritic_synapses'] = int(dend.dendritic_synapses.sum())
rep['kc_kc_dendritic_fraction'] = round(float(dend.dendritic_synapses.sum() / max(1, kk.w.sum())), 4)
rep['kc_kc_same_type_fraction'] = round(float(kk.w[kk.pre_type.values == kk.post_type.values].sum() / max(1, kk.w.sum())), 4)

# 4. feedback into the projection neurons
into_pn = E[E.post.isin(alpn)]
rep['alpn_input_share_by_class'] = {k: round(float(v / into_pn.w.sum()), 4) for k, v in into_pn.groupby('pre_g').w.sum().sort_values(ascending=False).head(12).items()}
mb_fb = into_pn[into_pn.pre_g.isin(['Kenyon_Cell', 'APL', 'DPM', 'MBON', 'DAN'])]
rep['alpn_input_from_mushroom_body'] = {k: int(v) for k, v in mb_fb.groupby('pre_g').w.sum().items()}

# 5. Phi^T Phi: overlap of PN input vectors for Kenyon pairs
pk = into_kc[into_kc.pre_g == 'ALPN']
mat = pk.pivot_table(index='post', columns='pre', values='w', aggfunc='sum', fill_value=0)
X = mat.values.astype(np.float64); X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
rng = np.random.default_rng(0)
i, j = rng.integers(0, len(X), 20000), rng.integers(0, len(X), 20000)
ok = i != j
ov = (X[i[ok]] * X[j[ok]]).sum(1)
rep['kenyon_pair_pn_overlap_cosine'] = q(ov)
rep['kenyon_pair_pn_overlap_fraction_above_0.2'] = round(float((ov > 0.2).mean()), 4)
rep['kenyon_cells_with_pn_input'] = int(len(X))

# online: where APL meets the Kenyon cells and the PN boutons
if NET:
    sys.path.insert(0, str(ROOT / 'scripts/ib/fly_interface'))
    from neuprint_query import cypher
    pairs = {'apl_to_kenyon': ("a.type = 'APL'", "b.class = 'Kenyon_Cell'"),
             'kenyon_to_apl': ("a.class = 'Kenyon_Cell'", "b.type = 'APL'"),
             'apl_to_alpn': ("a.type = 'APL'", "b.class = 'ALPN'"),
             'alpn_to_apl': ("a.class = 'ALPN'", "b.type = 'APL'"),
             'alpn_to_kenyon': ("a.class = 'ALPN'", "b.class = 'Kenyon_Cell'")}
    for name, (wa, wb) in pairs.items():
        cy = f"""
MATCH (a:Neuron)-[c:ConnectsTo]->(b:Neuron) WHERE {wa} AND {wb}
WITH apoc.convert.fromJsonMap(c.roiInfo) AS info
UNWIND keys(info) AS roi
RETURN roi, sum(info[roi].post) AS n ORDER BY n DESC"""
        try:
            data = cypher(cy)['data']
            tot_n = sum(int(n or 0) for _, n in data)
            rep[f'roi_{name}'] = {r: round(int(n or 0) / max(1, tot_n), 4) for r, n in data[:14]}
            rep[f'roi_{name}_total'] = tot_n
        except Exception as e:                                      # report, never the token
            rep[f'roi_{name}'] = f'query failed: {type(e).__name__}'

print(json.dumps(rep, indent=1))
if OUT:
    OUT.write_text(json.dumps(rep, indent=1), encoding='utf-8')
