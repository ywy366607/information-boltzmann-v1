"""Match native write-/active- jumps to isolate actual boundary memory."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from information_boltzmann.cgb.boundary_material import BoundaryIdentityMobileExchange
from information_boltzmann.cgb.mobile_exchange import MaterialEvent
from information_boltzmann.cgb.exchange_checkpoint import load_exchange,save_exchange
from scripts.ib.generative.native_learning_stream import advance_byte,atomic_json
from scripts.ib.generative.compare_support_transport import digest,same


def main(args):
    args.out.mkdir(parents=True,exist_ok=False)
    source,meta=load_exchange(args.archive);source_hash=digest(args.archive)
    blocks=np.load(args.run/'stream_blocks.npz',allow_pickle=False)
    assert digest(args.run/'stream_blocks.npz')==meta['sources']['selected_stream_sha256']
    phase,cursor=meta['session']['phase'],meta['session']['next_phase_byte']
    records=[]
    for port,node in enumerate(source.port_nodes):
        node=int(node)
        if not (source.port_installed[port] and source.port_trace[port] and source.depleted[node]):continue
        write=MaterialEvent('write',node,node,port,-1,interface=port)
        active=MaterialEvent('active',node,node,port,-1,interface=port)
        assert write in source.events() and active in source.events()
        models={name:BoundaryIdentityMobileExchange.from_state_dict(source.state_dict()) for name in ('write','active','sham')}
        jumps={}
        for name,model in models.items():
            event=write if name=='write' else active
            du,dg=model.event_delta(event)
            model.apply_event(event)
            jumps[name]=dict(delta_u=du,delta_g=dg)
        for name in source._arrays:
            if name!='port_trace':np.testing.assert_array_equal(getattr(models['write'],name),getattr(models['active'],name))
        for rng in ('rng_reaction','rng_motion','rng_field'):
            assert getattr(models['write'],rng).bit_generator.state==getattr(models['active'],rng).bit_generator.state
        bound=int(source.port_bound[port])
        expected=-(source.params.trace_energy-source.params.capture_coupling*bound)
        assert abs(jumps['write']['delta_u']-jumps['active']['delta_u']-expected)<1e-12
        unbind=MaterialEvent('binding',node,node,port,-1,interface=port)
        response={name:dict(port_trace=int(model.port_trace[port]),
            conductance=float(model.effective_port_conductance()[port]),
            native_unbind_rate=model.event_rate(unbind) if unbind in model.events() else None)
            for name,model in models.items()}
        predictions={}
        for name,model in models.items():
            session=copy.deepcopy(meta['session'])
            predictions[name]=advance_byte(model,session,int(blocks[phase][cursor]),phase)
            assert max(abs(float(v)) for v in model.balances().values())<1e-8
            if not records and name in ('write','active'):
                save_exchange(args.out/(name+'_complete.npz'),model,dict(session=session,sources=meta['sources'],
                    conditional_native_jump=dict(kind=name,sign=-1,port=port),probe_not_training_candidate=True))
        same(models['active'].state_dict(),models['sham'].state_dict())
        assert predictions['active']==predictions['sham']
        assert predictions['write']['scores']==predictions['active']['scores']
        records.append(dict(port=port,resource_node=node,source_time_ms=source.t,end_time_ms=models['write'].t,
            jumps=jumps,only_memory_configuration_differs_before_future=True,prearrival_scores_equal=True,
            response_before_future=response,sham_exact=True,
            brain_charge_difference_norm=float(np.linalg.norm(models['write'].charge-models['active'].charge)),
            world_charge_difference_norm=float(np.linalg.norm(models['write'].world_charge-models['active'].world_charge))))
    assert source_hash==digest(args.archive)
    atomic_json(args.out/'report.json',dict(measured=bool(records),passed=True,source=str(args.archive),
        source_sha256=source_hash,source_unchanged=True,records=records,
        scope='Conditional native-jump memory intervention on all naturally eligible ports; no direct trace reset, no predictive utility claim'))
    for path in (Path(__file__),ROOT/'docs/information_boltzmann/CGB_BOUNDARY_MEMORY_PROBE_20261022.md'):
        (args.out/path.name).write_bytes(path.read_bytes())
    print({'ports_measured':len(records),'source_unchanged':True},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run',type=Path,default=Path('F:/fly_stream_overflow/cgb_native_learning_20261022'))
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
