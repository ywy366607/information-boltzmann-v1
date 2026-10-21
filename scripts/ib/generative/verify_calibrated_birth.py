"""Verify real saved candidate state and two-program deployment artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from information_boltzmann.cgb.checkpoint import load_medium
from information_boltzmann.cgb.development import SpatialDevelopment
from information_boltzmann.cgb.morphology import SpatialIdentityProgram
from information_boltzmann.cgb.type_generator import TypeStrengthGenerator
from information_boltzmann.cgb.tree_development import ConditionalTreeDevelopment


def same(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a: same(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): same(x,y)
    elif isinstance(a,np.ndarray): np.testing.assert_array_equal(a,b)
    else: assert a==b,(a,b)


def main(args):
    saved=json.loads((args.run/'development_state.json').read_text(encoding='utf-8'))
    spatial=SpatialIdentityProgram.load(args.programs/'spatial_identity_program.npz')
    strength=TypeStrengthGenerator.load(args.generator)
    types=np.sort(np.random.default_rng(21).choice(np.flatnonzero(spatial.values['known']),48,replace=False))
    tc=np.repeat(types,4)
    plan=spatial.describe(tc,np.tile(['L','L','R','R'],len(types)))
    if args.tree:
        with np.load(args.tree,allow_pickle=False) as z: tree={k:z[k] for k in z.files}
        replay=ConditionalTreeDevelopment(strength.values['code'],tc,plan,tree)
    else:
        replay=SpatialDevelopment(strength.values['code'],tc,spatial_plan=plan)
    for _ in range(saved['generation']): replay.step()
    restored=SpatialDevelopment.from_state_dict(saved)
    same(replay.state_dict(),restored.state_dict())
    for _ in range(3): replay.step();restored.step()
    same(replay.state_dict(),restored.state_dict())
    a,provenance=load_medium(args.run/'medium_complete.npz')
    b,_=load_medium(args.run/'medium_complete.npz')
    for _ in range(120):
        a.step(I_ext=np.zeros(a.N));b.step(I_ext=np.zeros(b.N))
    same(a.state_dict(),b.state_dict())
    report={'growth_real_rebuild_bitwise':True,'growth_future_replay_steps':3,
            'medium_restore_future_bitwise_steps':120,'material_error_after_resume':float(np.abs(a.material.balance(a.g)).max()),
            'source_provenance':provenance,
            'payload_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.run.glob('*.npz')}}
    (args.run/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--programs',type=Path,required=True)
    p.add_argument('--generator',type=Path,required=True)
    p.add_argument('--tree',type=Path)
    main(p.parse_args())
