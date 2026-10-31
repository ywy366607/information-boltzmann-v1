"""Assemble the 62 verified sensory geometry blocks into one executable terminal archive.

Inputs are the already verified batch outputs (FLY_BATCH_CONNECTOME_20261019):
morphology forests, actual release sites, actual incoming contacts and their
cross-block release-site bindings. Nothing is pooled: every supplied node,
site and contact is kept with its original delay.

Units: one contact carries one synapse of the coba_r2 calibration
(excitatory 0.00178178, inhibitory 0.00653913 per synapse; the r2 builder's
per-synapse Shiu scale). Outgoing contacts split each live model edge evenly
over its actual release sites (contacts == synapses, checked here), so with no
terminal input the site route reproduces the point-cell delivery.

Contacts from octopamine/serotonin/dopamine cells are excluded and counted:
those transmitters act through metabotropic receptors whose fast edges are
pinned elsewhere; they await an amine receptor template, not a fast conductance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SENSORY_MARK = 'sensory'
AMINE_TYPES = r'^(?:OA-|5-HT|PAM\d|PPL\d)'


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def keys(pre, post, n):
    return np.asarray(pre, np.int64) * n + np.asarray(post, np.int64)


def lookup(sorted_keys, order, query):
    pos = np.searchsorted(sorted_keys, query)
    pos = np.minimum(pos, max(len(sorted_keys) - 1, 0))
    hit = sorted_keys[pos] == query if len(sorted_keys) else np.zeros(len(query), bool)
    out = np.full(len(query), -1, np.int64)
    out[hit] = order[pos[hit]]
    return out


def main(args):
    manifest = json.loads((args.geometry / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete':
        raise ValueError('Geometry linker did not complete')
    graph = np.load(args.graph, allow_pickle=False)
    n = len(graph['neuron_body_ids'])
    sign = graph['nt_sign']
    names = np.array(graph['superclass_names']).astype(str)[graph['superclass_id']]
    ids = graph['neuron_body_ids']
    ann = pd.read_feather(args.annotations).set_index('bodyId').reindex(ids)
    types = ann['type'].astype(str).values
    mod = pd.read_feather(args.modulatory)
    amine_bodies = set(mod[mod.consensusNt.isin(['octopamine', 'serotonin', 'dopamine'])].bodyId.astype(int))
    amine = np.isin(ids, list(amine_bodies)) | pd.Series(types).str.contains(AMINE_TYPES, regex=True).values
    amine |= np.isin(np.arange(n), graph['dan_indices'])
    model_keys = {}
    for channel in 'ei':
        k = keys(graph[f'edge_pre_{channel}'], graph[f'edge_post_{channel}'], n)
        order = np.argsort(k, kind='stable')
        if np.any(k[order][1:] == k[order][:-1]):
            raise ValueError('Duplicate model edge pair')
        model_keys[channel] = (k[order], order)
    master = np.load(args.master, allow_pickle=False)
    m_pre, m_post = master['edge_pre'], master['edge_post']
    returns = pd.read_feather(args.pairs, columns=['body_pre', 'body_post', 'delay_tier'])
    files = sorted(args.geometry.glob('geometry_*.npz'))
    if len(files) != manifest['completed_batches']:
        raise ValueError('Geometry block count differs from manifest')
    parts = {k: [] for k in ('volume', 'u', 'v', 'g', 'owner', 'site_node', 'site_owner', 'cell_index',
                              'in_node', 'in_source', 'in_delay', 'in_kind', 'in_site_block', 'in_site_local',
                              'out_edge', 'out_channel', 'out_site', 'out_delay', 'out_pair_key')}
    node_off = cell_off = site_off = 0
    site_offsets = []
    excluded = {'amine_source_contacts': 0, 'returnless_outgoing_contacts': 0}
    unlinked_out = 0
    identities = []
    for block, path in enumerate(files):
        note = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        if file_sha(path) != note['sha256']:
            raise ValueError('Geometry digest mismatch: ' + path.name)
        identities.append(note['sha256'])
        d = np.load(path, allow_pickle=False)
        b = np.load(args.geometry / 'bindings' / path.name, allow_pickle=False)
        if bytes(b['geometry_sha256'].tolist()).hex() != note['sha256']:
            raise ValueError('Binding belongs to another geometry block')
        nn_, nc, ns = len(d['volume']), len(d['cell_index']), len(d['site_node'])
        parts['volume'].append(d['volume'])
        parts['u'].append(d['u'].astype(np.int64) + node_off)
        parts['v'].append(d['v'].astype(np.int64) + node_off)
        parts['g'].append(d['g'])
        parts['owner'].append(d['owner'].astype(np.int64) + cell_off)
        parts['site_node'].append(d['site_node'].astype(np.int64) + node_off)
        parts['site_owner'].append(d['site_owner'].astype(np.int64) + cell_off)
        parts['cell_index'].append(d['cell_index'])
        src = d['incoming_source'].astype(np.int64)
        tier = returns.delay_tier.values[d['incoming_pair']]
        if not np.array_equal(tier, d['incoming_delay']):
            raise ValueError('Incoming contact delay differs from the return catalog')
        if not np.array_equal(ids[src], returns.body_pre.values[d['incoming_pair']]):
            raise ValueError('Incoming contact source identity differs from the return catalog')
        keep = ~amine[src]
        excluded['amine_source_contacts'] += int((~keep).sum())
        parts['in_node'].append(d['incoming_node'][keep].astype(np.int64) + node_off)
        parts['in_source'].append(src[keep])
        parts['in_delay'].append(d['incoming_delay'][keep])
        parts['in_kind'].append(np.where(sign[src[keep]] < 0, 1, 0).astype(np.int8))
        parts['in_site_block'].append(b['incoming_source_block'][keep].astype(np.int64))
        parts['in_site_local'].append(b['incoming_source_site'][keep].astype(np.int64))
        oe = d['outgoing_master_edge'].astype(np.int64)
        pre, post = m_pre[oe].astype(np.int64), m_post[oe].astype(np.int64)
        if not np.array_equal(post, d['outgoing_post']) or not np.isin(pre, d['cell_index']).all():
            raise ValueError('Outgoing contact endpoint mismatch')
        q = keys(pre, post, n)
        edge = np.full(len(q), -1, np.int64)
        channel = np.full(len(q), -1, np.int8)
        for c, label in enumerate('ei'):
            found = lookup(*model_keys[label], q)
            hit = (found >= 0) & (edge < 0)
            edge[hit], channel[hit] = found[hit], c
        live = edge >= 0
        # Outgoing contacts without a model edge are contacts onto another
        # sensory terminal: the same synapses enter as incoming native contacts.
        unlinked_out += int((~live).sum())
        parts['out_edge'].append(edge[live])
        parts['out_channel'].append(channel[live])
        parts['out_site'].append(d['outgoing_site'][live].astype(np.int64) + site_off)
        parts['out_delay'].append(d['outgoing_delay'][live].astype(np.int64))
        parts['out_pair_key'].append(q[live])
        site_offsets.append(site_off)
        node_off += nn_
        cell_off += nc
        site_off += ns
        print(json.dumps({'block': block, 'nodes': node_off, 'cells': cell_off, 'sites': site_off}), flush=True)
    data = {k: np.concatenate(v) for k, v in parts.items()}
    site_offsets = np.asarray(site_offsets, np.int64)
    native = data['in_site_block'] >= 0
    in_site = np.full(len(data['in_node']), -1, np.int64)
    in_site[native] = site_offsets[data['in_site_block'][native]] + data['in_site_local'][native]
    cell_index = data['cell_index'].astype(np.int64)
    if len(np.unique(cell_index)) != len(cell_index):
        raise ValueError('A native cell appears in two blocks')
    slot = np.full(n, -1, np.int64)
    slot[cell_index] = np.arange(len(cell_index))
    if native.any():
        owner_of_site = data['site_owner'][in_site[native]]
        if not np.array_equal(cell_index[owner_of_site], data['in_source'][native]):
            raise ValueError('Native incoming contact is bound to a site of another cell')
    if np.any(data['owner'][data['u']] != data['owner'][data['v']]):
        raise ValueError('Axial edge crosses cells')
    # Live model edges from native cells: each must have actual release sites.
    out_key = data['out_pair_key']
    verify = {}
    for c, label in enumerate('ei'):
        pre = graph[f'edge_pre_{label}']
        live = np.flatnonzero(slot[pre] >= 0)
        covered = np.unique(data['out_edge'][data['out_channel'] == c])
        missing = np.setdiff1d(live, covered)
        if missing.size:
            raise ValueError(f'{missing.size} {label} edges of native sensory cells lack actual release sites')
        verify[label] = live
        delay = graph[f'edge_delay_{label}']
        sel = data['out_channel'] == c
        if not np.array_equal(delay[data['out_edge'][sel]], data['out_delay'][sel]):
            raise ValueError('Outgoing contact delay differs from the model edge delay')
    # contacts == synapses: the r2 weight divided by contacts is the per-synapse unit
    units = {}
    share = np.zeros(len(out_key), np.float32)
    for c, label in enumerate('ei'):
        sel = data['out_channel'] == c
        edge = data['out_edge'][sel]
        count = np.bincount(edge, minlength=len(graph[f'edge_pre_{label}']))
        share[sel] = 1.0 / count[edge]
        w = graph[f'edge_weight_{label}'][verify[label]] / count[verify[label]]
        units[label] = float(np.median(w))
        if not np.allclose(np.quantile(w, [0.0, 1.0]), units[label], rtol=1e-4):
            raise ValueError(f'{label} per-contact birth weight is not one synapse unit')
    unit_e, unit_i = units['e'], units['i']
    in_weight = np.where(data['in_kind'] == 1, unit_i, unit_e).astype(np.float32)
    volume = data['volume']
    totals = np.bincount(data['owner'], weights=volume, minlength=len(cell_index))
    out = {
        'cell_index': cell_index, 'cell_body_ids': ids[cell_index],
        'volume': volume, 'u': data['u'].astype(np.int32), 'v': data['v'].astype(np.int32),
        'g': data['g'], 'owner': data['owner'].astype(np.int32),
        'density': (totals[data['owner']] / volume).astype(np.float32),
        'site_node': data['site_node'].astype(np.int32), 'site_owner': data['site_owner'].astype(np.int32),
        'in_node': data['in_node'].astype(np.int32), 'in_source': data['in_source'].astype(np.int32),
        'in_site': in_site.astype(np.int32), 'in_delay': data['in_delay'].astype(np.int8),
        'in_kind': data['in_kind'], 'in_weight': in_weight,
        'out_edge': data['out_edge'].astype(np.int32), 'out_channel': data['out_channel'],
        'out_site': data['out_site'].astype(np.int32), 'out_delay': data['out_delay'].astype(np.int8),
        'out_share': share,
        'verify_e_edge': verify['e'].astype(np.int32), 'verify_i_edge': verify['i'].astype(np.int32),
        'verify_e_pre': graph['edge_pre_e'][verify['e']], 'verify_e_post': graph['edge_post_e'][verify['e']],
        'verify_i_pre': graph['edge_pre_i'][verify['i']], 'verify_i_post': graph['edge_post_i'][verify['i']],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix('.tmp.npz')
    np.savez(temporary, **out)
    temporary.replace(args.output)
    sizes = {k: int(v.nbytes) for k, v in out.items()}
    report = {
        'at': datetime.now(timezone.utc).isoformat(), 'version': 1,
        'geometry': str(args.geometry.resolve()), 'geometry_sha256_by_block': identities,
        'graph': str(args.graph.resolve()), 'graph_sha256': file_sha(args.graph),
        'return_catalog': str(args.pairs.resolve()),
        'archive_sha256': file_sha(args.output),
        'cells': int(len(cell_index)), 'nodes': int(len(volume)), 'axial_edges': int(len(out['u'])),
        'release_sites': int(len(out['site_node'])),
        'incoming_contacts_executed': int(len(out['in_node'])),
        'incoming_contacts_native_source_site': int(native.sum()),
        'incoming_contacts_by_kind': {'excitatory': int((out['in_kind'] == 0).sum()),
                                      'inhibitory': int((out['in_kind'] == 1).sum())},
        'excluded': {**excluded, 'note': 'amine/DA source contacts await a metabotropic receptor template'},
        'outgoing_contacts_on_model_edges': int(len(out['out_edge'])),
        'outgoing_contacts_onto_sensory_terminals': unlinked_out,
        'live_model_edges_site_routed': {'e': int(len(verify['e'])), 'i': int(len(verify['i']))},
        'synapse_unit': {'e': unit_e, 'i': unit_i},
        'array_bytes': sizes, 'total_array_bytes': int(sum(sizes.values())),
        'scope': 'Anatomical execution operands only; receptor identity is not known. '
                 'Terminal effect = conductance shunting of release (mechanism hypothesis, see module).',
    }
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('geometry_sha256_by_block', 'array_bytes')}, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--geometry', type=Path, default=Path('F:/fly_stream_overflow/native_sensory_geometry_20261019'))
    p.add_argument('--graph', type=Path, default=Path('data/malecns_v1/fly_reservoir_coba_r2.npz'))
    p.add_argument('--master', type=Path, default=Path('data/malecns_v1/fly_reservoir_delayed.npz'))
    p.add_argument('--pairs', type=Path,
                   default=Path('F:/fly_stream_overflow/malecns_sensory_return_catalog_20261019/sensory_return_edges.feather'))
    p.add_argument('--annotations', type=Path, default=Path('data/malecns_v1/body-annotations.feather'))
    p.add_argument('--modulatory', type=Path, default=Path('data/malecns_v1/modulatory_neurons.feather'))
    p.add_argument('--output', type=Path,
                   default=Path('F:/fly_stream_overflow/native_sensory_terminals_20261019/sensory_terminals.npz'))
    main(p.parse_args())
