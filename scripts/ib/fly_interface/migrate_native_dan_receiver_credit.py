"""Add source-resolved receiver sensitivities at an explicit saved boundary."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.core.fly_dan_receiver_credit import DAN_RECEIVER_CREDIT_VERSION
from scripts.ib.train_fly_bptt_stream import atomic_save
from scripts.ib.fly_interface.migrate_native_kc_terminals import same
from scripts.ib.fly_interface.native_causal_audit import sha256


def migrate(checkpoint, output):
    if output.exists():
        raise ValueError('Use a new isolated full-state candidate')
    saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    cfg = saved['config']
    if not cfg.get('dan_carried_credit') or cfg.get('dan_receiver_credit'):
        raise ValueError('Require the source credit extension without receiver history')
    if saved['learner']['physical'].get('dan_credit_dopamine', torch.empty(0)).numel():
        raise ValueError('Refuse to reset receiver derivative history')
    with np.load(cfg['dan_release_geometry'], allow_pickle=False) as data:
        edges = np.unique(data['out_edge'])
    if not np.array_equal(edges, np.arange(len(edges))):
        raise ValueError('Actual DAN source geometry must cover every receiver edge')
    history = saved['learner']['physical']['h'].new_zeros(2, 7, len(edges))
    size = history.numel()*history.element_size()
    if shutil.disk_usage(output.parent).free < 2*(checkpoint.stat().st_size+size)+1024**3:
        raise OSError('Need complete candidate, atomic replacement and1GiB reserve')
    payload = {**saved, 'learner': {**saved['learner'], 'physical': {
        **saved['learner']['physical'], 'dan_credit_dopamine':history}},
        'config': {**cfg, 'dan_receiver_credit':True,
            'dan_receiver_credit_version':DAN_RECEIVER_CREDIT_VERSION,
            'window':1, 'checkpoint_dir':str(output.resolve())}}
    output.mkdir(parents=True)
    atomic_save(output/'last.pt', payload)
    del saved, payload, history
    gc.collect()
    before = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    after = torch.load(output/'last.pt', map_location='cpu', mmap=True, weights_only=False)
    assert same(before['model'], after['model'])
    assert all(same(v, after['learner']['physical'][k]) for k,v in before['learner']['physical'].items() if k!='dan_credit_dopamine')
    assert all(same(v, after['learner'][k]) for k,v in before['learner'].items() if k!='physical')
    assert all(same(v, after[k]) for k,v in before.items() if k not in ('model','learner','config'))
    report = {'source':str(checkpoint), 'source_sha256':sha256(checkpoint),
        'candidate_sha256':sha256(output/'last.pt'), 'trained':before['bptt_train_tokens'],
        'physical_ticks':before['learner'].get('physical_ticks'), 'new_inputs':0, 'new_optimizer_updates':0,
        'window_tokens':1, 'edges':len(edges), 'additional_state_bytes':size,
        'existing_complete_state_exact':True,
        'initialization':'Previously unrecorded source-resolved receiver derivatives start at this explicit boundary. All existing source tangents, physical state, optimizer, world, pending credits and RNG retained.',
        'scope':'Source material history through global DA filters; separate spatial KC field, phi and recurrent spike history not claimed',
        'status':'zero_exposure_DAN_receiver_credit_extension_verified'}
    (output/'migration.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    migrate(args.checkpoint,args.output)
