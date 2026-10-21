"""Check V3d's real-data feature/split identity and old finite-step null.

This CPU-only diagnostic leaves the concurrently running GPU sweep alone.
Historical test targets are not scored before the main selection is locked.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.ib.generative.v3d_stable_matching import prepare, write_json
from information_boltzmann.cgb.matching import TypeMatching, linear_fit


def main(args):
    started = time.perf_counter()
    data = prepare(args.work)
    train, val = (data['splits'][name] for name in ('train','val'))
    meta = np.load(args.work/'e1_types3000.npz', allow_pickle=True)
    sizes = meta['sizes']
    recovered = data['features']*data['feature_scale']+data['feature_mean']
    errors = []
    for block, scale in enumerate((4,2)):
        matrix = np.load(args.work/f'e1c_overlap_trainlabels_{scale}um.npy')
        for direction, m in enumerate((matrix, matrix.T)):
            old = m/(sizes[:,None]*sizes[None,:])
            floor = old[old>0].min()*.5
            old_log = np.log(np.maximum(old, floor))[data['pre'],data['post']]
            error = float(np.max(np.abs(old_log-recovered[:,block*4+direction*2])))
            errors.append(error)
        del matrix, old, old_log
    n = data['n']
    model = TypeMatching(n,0,0)
    t = torch.tensor(data['target'][train], dtype=torch.float32)
    pre = torch.tensor(data['pre'][train]); post = torch.tensor(data['post'][train])
    empty = torch.empty(len(train),0)
    with torch.no_grad():
        model.bias.copy_(t.mean())
    # Re-expression of the original degree-only Adam solve on selected entries.
    # The value and derivative identity of this expression is tested separately.
    opt = torch.optim.Adam(model.parameters(), lr=.05)
    for step in range(1500):
        opt.zero_grad(set_to_none=True)
        loss = (model(pre,post,empty)-t).square().mean()
        loss.backward(); opt.step()
        if step == 750:
            opt.param_groups[0]['lr'] = .01
    with torch.no_grad():
        legacy = model(torch.tensor(data['pre']),torch.tensor(data['post']),torch.empty(len(data['pre']),0)).numpy()
    _, converged, solver = linear_fit(n,data['pre'],data['post'],data['features'][:,:0],data['target'],train)
    mse = {}
    for split in ('train','val'):
        idx = data['splits'][split]
        a = float(np.mean((legacy[idx]-data['target'][idx])**2))
        b = float(np.mean((converged[idx]-data['target'][idx])**2))
        mse[split] = {'legacy_1500_mse': a, 'converged_mse': b, 'relative_difference': (a-b)/b}
    args.out.mkdir(parents=True,exist_ok=True)
    np.save(args.out/'legacy_null_prediction.npy',legacy)
    report = {'geometry_feature_absmax_by_direction_scale':errors,
        'geometry_exact_within_float64':max(errors)<1e-10,
        'split_sha256':data['split_sha256'], 'pairs':{k:len(v) for k,v in data['splits'].items()},
        'degree_null_comparison':mse, 'linear_solver':solver,
        'test_scored':False, 'seconds':time.perf_counter()-started}
    write_json(args.out/'data_audit.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--work',type=Path,default=Path('F:/fly_stream_overflow/cgb_v1'))
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
