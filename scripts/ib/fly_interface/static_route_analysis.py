"""Where does the text go, and does the mushroom body reach action?  Static, from the weights only (no simulation, CPU).
Linear influence: x_{k+1} = A x_k, A[post, pre] = g_e |w_e| - g_i |w_i| (signed; reported as absolute mass per unit
source activity -- never shares, never |w|: memory feedback-static-routes-signed-absolute),
Kenyon -> MBON weights from the compartment memory put back.  For each source distribution (mass 1) and hop k = 1..8:
the share of the hop-k mass on each group, and the mass landing on the output cells (what the language read and the
acts read) and on the descending neurons.  Removing the Kenyon cells (they transmit nothing) gives the share of the
text's influence on the outputs that passes through the mushroom body.  Run on the birth weights and on a trained
checkpoint, to see whether evolution weakened the route.
Usage: python static_route_analysis.py <checkpoint dir> [out.json]
"""
import sys, json, numpy as np, pandas as pd, torch, scipy.sparse as sp
from pathlib import Path

ROOT = Path('D:/information-boltzmann-v1')
ck = Path(sys.argv[1])
saved = torch.load(ck / 'last.pt', map_location='cpu', weights_only=False, mmap=True)
cfg, M = saved['config'], saved['model']
g = np.load(ROOT / cfg['graph'], allow_pickle=True)
N = len(g['neuron_body_ids'])
ann = pd.read_feather(ROOT / 'data/malecns_v1/body-annotations.feather').set_index('bodyId').reindex(g['neuron_body_ids'])
cls, typ = ann['class'].astype(str).values, ann['type'].astype(str).values
sc_names = list(g['superclass_names']); sc = g['superclass_id']
OUT = ('cb_motor', 'vnc_motor', 'descending_neuron', 'cb_efferent', 'vnc_efferent', 'efferent_ascending',
       'efferent_descending', 'cb_endocrine', 'vnc_endocrine')
is_out = np.isin(np.array(sc_names)[sc], OUT)
starts = lambda *p: np.any([np.char.startswith(typ.astype(str), x) for x in p], axis=0)
groups = {
    'writer_sensory': None,                                        # filled below (text-driven cells)
    'ALPN': cls == 'ALPN', 'KC_all': cls == 'Kenyon_Cell', 'KC_ab_core': starts('KCab-c'),
    'MBON': cls == 'MBON', 'DAN': cls == 'DAN', 'convergence_SMP_SIP_CRE': starts('SMP', 'SIP', 'CRE'),
    'CX': cls == 'CX', 'LAL': starts('LAL'), 'descending': np.array(sc_names)[sc] == 'descending_neuron',
    'output_cells': is_out, 'visual_projection': np.array(sc_names)[sc] == 'visual_projection',
    'cb_intrinsic': np.array(sc_names)[sc] == 'cb_intrinsic'}
part = np.load(ROOT / 'data/malecns_v1/sensory_partitions.npz')
drive = np.zeros(N)
for key, name in (('visual_idx', 'proj_vis'), ('chemo_idx', 'proj_chemo'), ('mechano_idx', 'proj_mech')):
    w = M[f'topographic_writer.{name}.weight'].detach().float().numpy()
    drive[part[key]] += np.linalg.norm(w, axis=1)                  # how strongly text drives each sensory cell
groups['writer_sensory'] = drive > 0
read_use = np.zeros(N)
read_idx = np.where(is_out)[0]
read_w = M['output_read.weight'].detach().float().numpy()                    # [d, n_out]: which output cells the read uses
if read_w.shape[1] == len(read_idx):
    read_use[read_idx] = np.linalg.norm(read_w, axis=0)


def matrix(trained, signed=True):
    """Signed by default (excitation +, inhibition -): |w| counts an inhibited target as driven (memory:
    feedback-static-routes-signed-absolute).  signed=False only to compare with old reports."""
    we = (M['edge_weight_e'].detach().float().numpy() if trained else g['edge_weight_e']).astype(np.float64).copy()
    wi = (M['edge_weight_i'].detach().float().numpy() if trained else g['edge_weight_i']).astype(np.float64)
    pre_e, post_e = g['edge_pre_e'], g['edge_post_e']
    if trained and 'comp_weight' in M:
        kc_mbon = np.nonzero((cls == 'Kenyon_Cell')[pre_e] & (cls == 'MBON')[post_e])[0]
        if len(kc_mbon) == M['comp_weight'].numel():
            we[kc_mbon] = M['comp_weight'].detach().float().numpy()         # the plastic Kenyon -> MBON synapses
    ge = float(np.exp(M['log_g_e'].item())) if trained else 1.0
    gi = float(np.exp(M['log_g_i'].item())) if trained else 1.0
    A = sp.coo_matrix((np.concatenate([ge * np.abs(we), (-1.0 if signed else 1.0) * gi * np.abs(wi)]),
                       (np.concatenate([post_e, g['edge_post_i']]), np.concatenate([pre_e, g['edge_pre_i']]))),
                      shape=(N, N)).tocsr()
    return A


def spread(A, x0, hops=8, silence=None):
    x = x0 / x0.sum()
    rows = []
    for k in range(1, hops + 1):
        if silence is not None:
            x = x * (~silence)
        x = A @ x
        tot = x.sum()
        # absolute signed mass per unit source activity (shares mislead across weight sets: every excitatory weight
        # grew 4.4x in training; and a signed total can be ~0)
        rows.append({'hop': k, 'total': float(tot),
                     **{f'mass_{n}': float(x[m].sum()) for n, m in groups.items()},
                     'to_outputs': float(x[is_out].sum()), 'to_descending': float(x[groups['descending']].sum()),
                     'to_read_weighted': float((x * read_use).sum())})
    return rows


rep = {'checkpoint': str(ck), 'group_sizes': {n: int(m.sum()) for n, m in groups.items()}}
for label, trained in (('birth', False), ('trained', True)):
    A = matrix(trained)
    out = {}
    out['text'] = spread(A, drive)
    out['text_without_KC'] = spread(A, drive, silence=groups['KC_all'])
    for src in ('KC_ab_core', 'KC_all', 'MBON', 'ALPN', 'visual_projection', 'cb_intrinsic'):
        out[src] = spread(A, groups[src].astype(float))
    rep[label] = out
    t, n = out['text'], out['text_without_KC']
    print(f'== {label}')
    print('  text -> outputs by hop:', [f"{r['to_outputs']:.2e}" for r in t])
    print('  share through Kenyon cells:', [f"{1 - b['to_outputs'] / max(a['to_outputs'], 1e-30):.3f}" for a, b in zip(t, n)])
    print('  text mass on KC by hop (signed, absolute):', [f"{r['mass_KC_all']:.4f}" for r in t])
    for src in ('KC_ab_core', 'MBON', 'ALPN', 'visual_projection', 'cb_intrinsic'):
        r = out[src]
        print(f'  {src:>18} -> descending (signed mass):', [f"{x['mass_descending']:.4f}" for x in r][:6],
              '| outputs (abs):', [f"{x['to_outputs']:.1e}" for x in r][:6])
    r = out['KC_ab_core']
    print('  alpha/beta core route, share by hop: MBON', [f"{x['mass_MBON']:.4f}" for x in r][:5],
          'DAN', [f"{x['mass_DAN']:.4f}" for x in r][:5], 'convergence', [f"{x['mass_convergence_SMP_SIP_CRE']:.3f}" for x in r][:5],
          'CX', [f"{x['mass_CX']:.3f}" for x in r][:5], 'LAL', [f"{x['mass_LAL']:.3f}" for x in r][:5])
rep['read_use_share'] = {n: float(read_use[m].sum() / max(read_use.sum(), 1e-30)) for n, m in groups.items()}
print('language read weight share:', {n: round(v, 3) for n, v in rep['read_use_share'].items() if v > 0})
json.dump(rep, open(Path(sys.argv[2]) if len(sys.argv) > 2 else ck / 'static_routes.json', 'w'), indent=1)
