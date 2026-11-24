"""Frozen action-bank roles and same-context archived teacher contrasts."""
import argparse
from collections import defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.ib_local.latent_action_planner import features


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path('scratch/cgb_math_contracts_20261122/source_ports_2537.pt'))
    parser.add_argument('--output', type=Path, default=Path('results/published/cgb_action_role_geometry_20261122.json'))
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = torch.load(args.input, weights_only=False, map_location='cpu')
    w = source['encoder']['encoder.weight'].double()
    encoder = torch.nn.Linear(w.shape[1], w.shape[0], bias=False).double()
    encoder.weight.copy_(w)
    engine = SimpleNamespace(model=SimpleNamespace(encoder=encoder))
    records = source['engine']['latent_action_bank']['records']
    groups = defaultdict(list)
    for i, record in enumerate(records):
        if record['kind'] != 'native':
            continue
        p = record['packet']
        raw, scalars = p['raw'].clone(), list(p['scalars'])
        raw[1] = 0
        scalars[3:6] = [0., 0., 0.]
        identity = (p['relation'], p['unknown'], p['role'], record['revision'])
        key = hashlib.sha256(raw.numpy().tobytes()+repr((scalars, identity)).encode()).hexdigest()
        groups[key].append(i)
    right = torch.linalg.solve(w@w.T, w).T
    rows = []
    for key, group in groups.items():
        for ia, ib in combinations(group, 2):
            a, b = records[ia], records[ib]
            pa, pb = a['packet'], b['packet']
            fa, fb = features(engine, [pa, pb])
            delta = pb['raw'][1].double()-pa['raw'][1].double()
            visible = right@(w@delta)
            hidden = delta-visible
            ta, tb = a['target'].double(), b['target'].double()
            kernel2 = (1+fa@fa).square()+(1+fb@fb).square()-2*(1+fa@fb).square()
            rows.append(dict(context_sha256=key, indices=[ia, ib],
                record_keys=[a['key'], b['key']], teacher_bank_revision=a['revision'],
                relation=list(pa['relation']), unknown=pa['unknown'], role=pa['role'],
                options=[pa['option'], pb['option']],
                nominal_difference_l2=float(delta.norm()),
                encoder_visible_difference_l2=float(visible.norm()),
                encoder_null_difference_l2=float(hidden.norm()),
                encoder_null_energy_fraction=float(hidden.square().sum()/delta.square().sum()),
                encoded_difference_l2=float((w@delta).norm()),
                shared_post_target_max_difference=float((ta[:256]-tb[:256]).abs().max()),
                endpoint_target_difference_l2=float((ta[256:512]-tb[256:512]).norm()),
                archived_local_q=[float(ta[-1]), float(tb[-1])],
                archived_local_q_difference=float(tb[-1]-ta[-1]),
                feature_distance=float((fa-fb).norm()),
                kernel_distance=float(kernel2.sqrt()),
                local_q_interpolant_pair_norm_lower_bound=float(abs(tb[-1]-ta[-1])/kernel2.sqrt())))
    output = dict(schema=1, cycle=source['cycle'],
        source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        role_map=dict(pre=0, nominal=1, pre_momentum=6), native_records=sum(
            r['kind']=='native' for r in records), matched_context_pairs=rows,
        target='archived native q and endpoint targets in the current bank representation',
        feature_encoder='saved cycle2537 encoder',
        context_matching='all non-nominal raw blocks, non-option scalars, relation, unknown, role and teacher revision',
        contrast_span='local native q; total continuation and policy supplied by the live four-corner observer',
        training_updates=0, world_calls=0)
    args.output.write_text(json.dumps(output, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(cycle=source['cycle'], matched_pairs=len(rows), pairs=rows)))


if __name__ == '__main__':
    main()
