"""Exact local reaction response of phase rates/bounds, not a reward rule."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from information_boltzmann.cgb.exchange_checkpoint import load_exchange
from information_boltzmann.cgb.material_fork import fork_material
from information_boltzmann.cgb.learning_observation import parameter_quench
from scripts.ib.generative.compare_support_transport import digest,same
from scripts.ib.generative.native_learning_stream import atomic_json


def response(model):
    initial=model.domain_generator_rates()
    features=('recovery_accepted_rate','birth_attempt_rate_upper_bound')
    total={key:0. for key in features};groups={};rows=[]
    degree=model.degrees()
    for event in model.events(degree=degree):
        rate=float(model.event_rate(event,degree))
        if not rate:continue
        branch=fork_material(model)
        branch.apply_event(event,check=False)
        after=branch.domain_generator_rates()
        contribution={key:rate*(after[key]-initial[key]) for key in features}
        label=('port_' if event.interface>=0 else '')+event.kind
        group=groups.setdefault(label,{key:0. for key in features})
        for key,value in contribution.items():total[key]+=value;group[key]+=value
        rows.append(dict(kind=event.kind,sign=event.sign,i=event.i,j=event.j,interface=event.interface,
                         native_rate=rate,contribution=contribution))
    return dict(initial_phase_rates=initial,native_generator_response=total,by_kind=groups,
                nonzero_channels=[row for row in rows if any(value!=0 for value in row['contribution'].values())],
                evaluated_channels=len(rows))


def main(args):
    model,metadata=load_exchange(args.archive)
    initial=model.state_dict();source_hash=digest(args.archive)
    intact=response(model)
    cut=fork_material(model)
    intervention=parameter_quench(cut,capture_coupling=0.)
    assert model.domain_generator_rates()==cut.domain_generator_rates()
    changed=response(cut)
    same(initial,model.state_dict())
    assert digest(args.archive)==source_hash
    report=dict(source=str(args.archive),source_sha256=source_hash,source_unchanged=True,
        intact=intact,no_capture=changed,intervention=intervention,
        instantaneous_phase_rates_equal=True,
        difference={key:changed['native_generator_response'][key]-intact['native_generator_response'][key]
                    for key in intact['native_generator_response']},
        scope='Exploratory exact bridge/pool subgenerator applied to current recovery hazard and birth-attempt upper bound. Geometry/source/OU and phase jumps are not included in this local generator. The bound is not an integrated birth rate. Hypothetical permitted native jumps are enumerated with actual rates/energy; they do not change the primary state or become a learning score.')
    atomic_json(args.out,report)
    print(json.dumps({key:report[key] for key in ('instantaneous_phase_rates_equal','difference','scope')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--archive',type=Path,default=Path('F:/fly_stream_overflow/cgb_domain_birth_20261022/growing_complete.npz'))
    p.add_argument('--out',type=Path,default=Path('F:/fly_stream_overflow/cgb_domain_rate_r1_20261022/capture_generator.json'))
    main(p.parse_args())
