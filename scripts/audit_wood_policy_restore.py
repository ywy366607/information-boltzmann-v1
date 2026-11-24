"""Reconstruct the original live cache and verify the saved decision, without a world step."""
import argparse
from contextlib import contextmanager
import gc
import gzip
import json
from pathlib import Path
import re
import sys


def original_projection(checkpoint, output):
    import torch
    with gzip.open(checkpoint, 'rb') as stream:
        saved=torch.load(stream,map_location='cpu',weights_only=False)
    weights=saved['model']; parts=[]; names=[]
    for i in range(32):
        keys=[k for k in weights if re.search(rf'(?:^|\.)layers\.{i}\.q_proj\.weight$',k)]
        key=sorted(keys,key=len)[0]
        norm=key.replace('q_proj.weight','norm.weight')
        # Same GPU product/concatenation used by the original packed read.
        parts.append(weights[key].cuda()*weights[norm].cuda()[None])
        names.append((key,norm))
    packed=torch.cat(parts)
    torch.save(dict(weight=packed.cpu(),source=str(checkpoint),keys=names),output)
    del saved,weights,parts
    gc.collect()
    return packed


@contextmanager
def inherited_projection(weight):
    from scripts.ib_local import navigation_momentum_wood as momentum
    original=momentum._packed_momentum_read
    installed=set()
    def read(model,momenta,source):
        if id(model) not in installed:
            # Construct today's identity/version key, then restore the actual
            # cached value carried by the uninterrupted original life line.
            original(model,momenta,source)
            cache=model._packed_momentum_query
            assert cache['weight'].shape==weight.shape
            cache['weight']=weight.to(cache['weight'])
            installed.add(id(model))
        return original(model,momenta,source)
    momentum._packed_momentum_read=read
    try:yield
    finally:momentum._packed_momentum_read=original


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--temporary',type=Path,required=True)
    args=p.parse_args()
    import torch
    from scripts import run_wood_policy_use_worker as worker
    from scripts.ib_local import action_value_pair_audit as audit
    protocol=json.loads((args.campaign/'protocol.json').read_text(encoding='utf8'))
    projection_path=args.campaign/'original_live_projection.pt'
    weight=original_projection(protocol['source'],projection_path)
    observation=json.loads((args.campaign/'natural/observations.json').read_text(encoding='utf8'))['4']['action']
    original=audit.first_action
    class PreviewComplete(Exception):pass
    @contextmanager
    def stop_at_choice(engine,option,publish,**kw):
        def capture(receipt):
            keys=('source_sha256','endpoint_sha256','recurrent_sha256','unknown','questions',
                  'proposals','selected_proposal','local','future','total','gamma','habit','offset')
            checks={key:receipt[key]==observation[key] for key in keys}
            checks['nominal_probability']=max(abs(a-b) for a,b in zip(receipt['nominal_probability'],observation['production']))<1e-7
            result=dict(checks=checks,passed=all(checks.values()),receipt=receipt,
                        external_rounds=0,updates=0,projection=str(projection_path))
            (args.campaign/'restore_projection_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
            raise PreviewComplete()
        with original(engine,option,capture,**kw) as used:yield used
    audit.first_action=stop_at_choice
    argv=sys.argv
    sys.argv=[argv[0],'--mode','branch','--option','1',
        '--checkpoint',protocol['prefix_paths']['4']+'/last.pt.gz',
        '--work',str(args.work),'--temporary',str(args.temporary),
        '--protocol',str(args.campaign/'protocol.json')]
    try:
        with inherited_projection(weight):
            try:worker.main()
            except PreviewComplete:pass
        result=json.loads((args.campaign/'restore_projection_audit.json').read_text())
        assert result['passed'],result['checks']
        print(json.dumps(dict(passed=True,checks=result['checks'],external_rounds=0,updates=0)))
    finally:
        audit.first_action=original
        sys.argv=argv


if __name__=='__main__':main()
