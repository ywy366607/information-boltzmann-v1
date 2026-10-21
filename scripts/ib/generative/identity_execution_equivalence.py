"""Same-clock archived-state check for disposable identity memoization."""
from __future__ import annotations

import argparse
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from information_boltzmann.cgb.identity_material import IdentityMobileExchange
from information_boltzmann.cgb.exchange_checkpoint import load_exchange
from scripts.ib.generative.driven_ports_audit import input_voltage


def main(args):
    source = args.reference/'source_snapshot/identity_material.py'
    report = json.loads((args.reference/'report.json').read_text(encoding='utf-8'))
    key = next(k for k in report['code_sha256'] if k.replace('\\','/').endswith('/identity_material.py'))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == report['code_sha256'][key]
    name = 'information_boltzmann.cgb._actual_identity_reference'
    spec = importlib.util.spec_from_file_location(name,source)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    arms={}
    for label in report['arms']:
        path=args.reference/(label+'_half.npz')
        with np.load(path,allow_pickle=False) as z:
            meta=json.loads(str(z['metadata']));s=meta['state']
            s.update({k:z[k].copy() for k in ('pre','post','capacity')})
            s['arrays']={k:z['state_'+k].copy() for k in IdentityMobileExchange._arrays}
        old,new=module.IdentityMobileExchange.from_state_dict(s),IdentityMobileExchange.from_state_dict(s)
        data=bytes.fromhex(meta['provenance']['byte_stream_hex'])
        times=[]
        for model in (old,new):
            started=time.perf_counter()
            for step in range(64,128): model.step(2.,input_voltage(data,step,2.))
            times.append(time.perf_counter()-started)
        for k in new._arrays+('pre','post','capacity'):
            np.testing.assert_array_equal(getattr(old,k),getattr(new,k))
        for k in ('rng_motion','rng_reaction','rng_field'):
            assert getattr(old,k).bit_generator.state==getattr(new,k).bit_generator.state
        assert old.reaction_counts==new.reaction_counts
        error=max(abs(getattr(old,k)-getattr(new,k)) for k in new._ledgers)
        assert error==0.
        arms[label]=dict(physical_arrays_rng_and_ledgers_bitwise=True,
            direct_seconds=times[0],memoized_seconds=times[1],peak_retained_pair_entries=len(new._pair_cache))
    args.out.write_text(json.dumps(dict(reference=str(args.reference),
        clock='128->256ms at dt2ms, identical actual real-byte stream',arms=arms,passed=True),indent=2),encoding='utf-8')
    print(json.dumps(arms))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
