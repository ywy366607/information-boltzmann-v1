"""Verify an extension using one mapped archive per process on small hosts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.core.fly_kc_lateral import (
    KC_LATERAL_PARAMETER_NAMES, KC_LATERAL_STATE_FIELDS, initial_parameters,
)


def digest(value):
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        array = value.detach().cpu().numpy() if torch.is_tensor(value) else value
        checksum = hashlib.sha256()
        # nditer also bounds copies for strided tensors. Never duplicate a
        # language head or a complete optimizer just to compare its bytes.
        for block in np.nditer(array, flags=['external_loop', 'buffered', 'zerosize_ok'],
                               op_flags=['readonly'], order='C', buffersize=1024**2):
            checksum.update(np.ascontiguousarray(block).tobytes())
        return {'kind': 'tensor' if torch.is_tensor(value) else 'numpy',
                'dtype': str(array.dtype), 'shape': list(array.shape), 'sha256': checksum.hexdigest()}
    if isinstance(value, dict):
        return {str(k): digest(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return {'sequence': type(value).__name__, 'items': [digest(v) for v in value]}
    return value


def file_hash(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            checksum.update(block)
    return checksum.hexdigest()


def snapshot(saved):
    optimizer = saved['learner']['optimizer']
    moments = {name: optimizer['state'].get(index)
               for group in optimizer['param_groups']
               for name, index in zip(group['parameter_names'], group['params'])}
    return {
        'model': digest(saved['model']),
        'physical': digest(saved['learner']['physical']),
        'learner': digest({k:v for k,v in saved['learner'].items()
                           if k not in ('physical', 'optimizer', 'adam_names')}),
        'other': digest({k:v for k,v in saved.items() if k not in ('config','model','learner')}),
        'moments': digest(moments),
        'groups': [{k:v for k,v in group.items() if k != 'params'} for group in optimizer['param_groups']],
        'adam_names': saved['learner']['adam_names'], 'config': saved['config'],
    }


def verify(checkpoint, source_record, output):
    old = json.loads(source_record.read_text(encoding='utf-8'))
    saved = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    new = snapshot(saved)
    for group in ('model','physical','moments'):
        for key,value in old[group].items():
            assert new[group][key] == value, (group,key)
    assert new['learner'] == old['learner'] and new['other'] == old['other']
    for old_group,new_group in zip(old['groups'],new['groups']):
        reduced = dict(new_group)
        reduced['parameter_names'] = [n for n in reduced['parameter_names'] if n not in KC_LATERAL_PARAMETER_NAMES]
        assert json.loads(json.dumps(reduced)) == old_group
    assert len(new['groups']) == len(old['groups'])
    assert [n for n in new['adam_names'] if n not in KC_LATERAL_PARAMETER_NAMES] == old['adam_names']
    allowed = {'kc_lateral','kc_lateral_version','kc_lateral_geometry','checkpoint_dir'}
    assert json.loads(json.dumps({k:v for k,v in new['config'].items() if k not in allowed})) == {
        k:v for k,v in old['config'].items() if k not in allowed}
    assert new['config']['kc_lateral'] and new['config']['kc_lateral_version'] == 1
    assert set(new['model']) - set(old['model']) == set(KC_LATERAL_PARAMETER_NAMES) | {'kc_lateral_digest'}
    assert set(new['physical']) - set(old['physical']) <= set(KC_LATERAL_STATE_FIELDS)
    count = saved['model'][KC_LATERAL_PARAMETER_NAMES[0]].numel()
    for name,value in initial_parameters(count).items():
        assert torch.equal(saved['model'][name],value),name
    geometry = Path(new['config']['kc_lateral_geometry'])
    assert bytes(saved['model']['kc_lateral_digest'].tolist()).hex() == file_hash(geometry)
    for name in KC_LATERAL_STATE_FIELDS:
        state = saved['learner']['physical'][name]
        assert state.shape == (1, saved['learner']['physical']['kc_terminal_gaba'].shape[1])
        assert torch.count_nonzero(state) == 0
    report = {'source':old['source'], 'source_sha256':old['source_sha256'],
        'candidate':str(checkpoint), 'candidate_sha256':file_hash(checkpoint),
        'trained':int(saved['bptt_train_tokens']), 'new_formal_inputs':0, 'new_formal_updates':0,
        'old_model_physical_optimizer_credit_world_rng_exact':True,
        'verification':'Sequential archive SHA256 fingerprints; one mapped archive per process',
        'new_state_bytes':sum(saved['learner']['physical'][n].numel()*4 for n in KC_LATERAL_STATE_FIELDS),
        'new_parameters':sum(saved['model'][n].numel() for n in KC_LATERAL_PARAMETER_NAMES),
        'archive_only':True,'training_adoption_approved':False,
        'initialization':'New receptor/effector coordinates at rest; existing packets and history retained',
        'status':'zero_exposure_KC_lateral_extension_verified'}
    for name in ('migration.json','checkpoint_metadata.json'):
        (output/name).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--source-record',type=Path,required=True)
    parser.add_argument('--capture-source',action='store_true')
    args=parser.parse_args()
    if args.capture_source:
        saved=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
        record=snapshot(saved)
        record.update(source=str(args.checkpoint),source_sha256=file_hash(args.checkpoint))
        args.source_record.write_text(json.dumps(record)+'\n',encoding='utf-8')
        print('Source fingerprints saved',flush=True)
    else:
        verify(args.checkpoint,args.source_record,args.checkpoint.parent)
