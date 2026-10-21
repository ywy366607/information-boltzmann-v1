"""Post-primary inventory-matched seed intervention; physical diagnosis."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from information_boltzmann.cgb.exchange_checkpoint import load_exchange,save_exchange
from information_boltzmann.cgb.spatial_exchange import SpatialExchangeParams
from scripts.ib.generative.compare_support_transport import digest,same
from scripts.ib.generative.driven_ports_audit import input_voltage
from scripts.ib.generative.spatial_growth_audit import construct,geometry


def main(args):
    args.out.mkdir(parents=True,exist_ok=False)
    primary=json.loads((args.primary/'report.json').read_text(encoding='utf-8'))
    data=bytes.fromhex(primary['byte_stream_hex'])
    p=SpatialExchangeParams(**primary['arms']['J0.6_D0.005']['params'])
    m,seed=construct(p)
    seed_mass=float(m.capacity.sum())
    installed_initial_energy=m.energy()
    installed_initial_g=m.free_energy()
    m.capacity[:]=0.
    m.material+=seed_mass/m.n
    m.initial_material=m.material+m._sum(m.post,m.capacity)
    m.initial_energy,m.initial_free_energy=m.energy(),m.free_energy()
    assert max(float(np.max(np.abs(v),initial=0.)) for v in m.balances().values())<1e-10
    initial=geometry(m,seed) if m.capacity.sum()>0 else dict(installed=0.)
    maximum={k:0. for k in m.balances()}
    least,fork=0.,None
    started=time.perf_counter()
    for k in range(128):
        v=input_voltage(data,k,2.)
        result=m.step(2.,world_voltage=v)
        least=min(least,result['free_energy_drop'])
        if fork is not None:fork.step(2.,world_voltage=v)
        for name,value in m.balances().items():
            maximum[name]=max(maximum[name],float(np.max(np.abs(value),initial=0.)))
        if k+1==64:
            save_exchange(args.out/'no_seed_half.npz',m,dict(next_step=64,byte_stream_hex=data.hex(),dt_ms=2.,byte_ms=4.))
            fork,_=load_exchange(args.out/'no_seed_half.npz')
    same(m.state_dict(),fork.state_dict())
    original,_=load_exchange(args.primary/'J0.6_D0.005_complete.npz')
    assert abs(original.initial_material.sum()-m.initial_material.sum())<1e-10
    path=args.out/'no_seed_complete.npz'
    save_exchange(path,m,dict(scope='post-primary material-inventory-matched seed diagnosis',params=asdict(p)))
    report=dict(date='2026-10-22',post_primary_exploratory=True,
        preregistered_before_this_run='docs/information_boltzmann/CGB_SPATIAL_SEED_CONTROL_20261022.md',
        primary_report=str(args.primary/'report.json'),primary_report_sha256=digest(args.primary/'report.json'),
        material_total=float(m.initial_material.sum()),seed_material_returned_uniformly=seed_mass,
        initial=initial,initial_energy=m.initial_energy,initial_free_energy=m.initial_free_energy,
        installed_seed_initial_energy=installed_initial_energy,installed_seed_initial_free_energy=installed_initial_g,
        final=geometry(m,seed),final_charged=float(m.charged.sum()),source_work=m.source_work,
        source_work_seeded=original.source_work,charge_absmax_from_seeded=float(np.max(np.abs(m.charge-original.charge))),
        density_absmax_from_seeded=float(np.max(np.abs(m.capacity-original.capacity))),
        maximum_balance_errors=maximum,minimum_irreversible_g_drop=least,
        seconds_with_restore=time.perf_counter()-started,full_restore_future_bitwise=True,
        archive=str(path),sha256=digest(path),passed=True)
    assert max(maximum.values())<1e-8 and least>=-1e-9
    (args.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--primary',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
