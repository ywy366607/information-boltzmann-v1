"""Prepare disjoint real OWT exposures for the continuing old-native-credit pair.

Local UTF8-byte corpus is decoded then tokenized by the actual pretrained model.
No synthetic sequences or world labels enter the learner.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from scripts.ib_local.run_pretrained_gdn_cross_window import read_text,write_json


def pack_stream(first,second,validation, *, phase_tokens,validation_tokens):
    if phase_tokens < 128 or phase_tokens%128:
        raise ValueError('phase size must be a positive multiple of128')
    if (min(len(first),len(second)) < phase_tokens+validation_tokens+2 or
            len(validation) < validation_tokens+1):
        raise ValueError('real-text slices do not cover training and disjoint continuations')
    x,y = first[:phase_tokens],second[:phase_tokens]
    return dict(stream=torch.cat((x,y,x,y[:1])),window=128,
        heldouts=dict(validation=validation[:validation_tokens+1],
            a_unseen=first[phase_tokens+1:phase_tokens+validation_tokens+2],
            b_unseen=second[phase_tokens+1:phase_tokens+validation_tokens+2]),
        warm_tokens=0,phase_steps=phase_tokens//128,
        phase_labels=['X_continue','Y_continue','X_revisit'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('data/ib_owt_10000'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--segment-steps',type=int,default=32)
    p.add_argument('--validation-tokens',type=int,default=512)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve prepared real stream')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    offset = json.loads((args.data/'tokenizer.json').read_text())['offset']
    manifest = json.loads((args.data/'manifest.json').read_text())
    locations = dict(x=('train',4_000_000),y=('train',6_000_000),validation=('validation',250_000))
    blocks,hashes = {},{}
    for name,(split,start) in locations.items():
        text,hashes[name] = read_text(args.data/f'{split}.npy',offset,start)
        blocks[name] = tokenizer(text,return_tensors='pt',add_special_tokens=False).input_ids[0]
    data = pack_stream(blocks['x'],blocks['y'],blocks['validation'],
        phase_tokens=args.segment_steps*128,validation_tokens=args.validation_tokens)
    data.update(dataset_revision=manifest['revision'],source_hashes=hashes,
        source_locations={k:dict(split=s,array_start=i,array_length=196608) for k,(s,i) in locations.items()},
        stream_sha256=hashlib.sha256(data['stream'].numpy().tobytes()).hexdigest(),
        note='fresh relative to the491-update online branch (prior corpus starts0 and2000000); '
             'not claimed unseen during foundation pretraining. X and Y are corpus slices, '
             'not certified semantic domains. Heldouts are disjoint continuations; score real bridges.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(data,args.output)
    metadata={k:v for k,v in data.items() if k not in ('stream','heldouts')}
    metadata['stream_tokens'] = len(data['stream'])
    metadata['heldout_tokens'] = {k:len(v) for k,v in data['heldouts'].items()}
    write_json(args.output.with_suffix('.json'),metadata)
    print(json.dumps(metadata),flush=True)


if __name__=='__main__':
    main()
