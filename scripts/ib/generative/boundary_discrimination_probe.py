"""Physical discrimination after matched native memory jumps, no training."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from information_boltzmann.cgb.exchange_checkpoint import load_exchange
from information_boltzmann.cgb.mobile_exchange import MaterialEvent
from information_boltzmann.cgb.boundary_material import BoundaryIdentityMobileExchange
from information_boltzmann.cgb.sensory_channel import exact_field_reference,uniform_information_bound
from scripts.ib.generative.sensory_exposure_audit import rooted_nodes
from scripts.ib.generative.compare_support_transport import digest
from scripts.ib.generative.native_learning_stream import atomic_json


def main(args):
    args.out.mkdir(parents=True,exist_ok=False)
    source,_=load_exchange(args.archive);source_hash=digest(args.archive)
    records=[]
    for port,node in enumerate(source.port_nodes):
        if not(source.port_installed[port] and source.port_trace[port] and source.depleted[node]):continue
        record=dict(port=port,conditional_duration_ms=4.,observed_brain_nodes=rooted_nodes(source).tolist(),states={})
        for kind in ('write','active'):
            model=BoundaryIdentityMobileExchange.from_state_dict(source.state_dict())
            event=MaterialEvent(kind,int(node),int(node),port,-1,interface=port)
            assert event in model.events()
            du,dg=model.event_delta(event);model.apply_event(event)
            channel=exact_field_reference(model,4.,rooted_nodes(model))
            whitened=np.linalg.solve(np.linalg.cholesky(channel['covariance']),channel['gain'][:,port])
            mahalanobis_squared=4*9*float(whitened@whitened)
            record['states'][kind]=dict(trace=int(model.port_trace[port]),
                conductance=float(model.effective_port_conductance()[port]),
                native_jump_delta_u=du,native_jump_delta_g=dg,source_flip_kl_nats=.5*mahalanobis_squared,
                equal_prior_source_flip_error=.5*math.erfc(math.sqrt(mahalanobis_squared)/(2*math.sqrt(2))),
                uniform_source_gaussian_information_bound_nats=uniform_information_bound(channel))
        records.append(record)
    assert source_hash==digest(args.archive)
    paths=[Path(__file__),ROOT/'information_boltzmann/cgb/boundary_material.py',
           ROOT/'information_boltzmann/cgb/sensory_channel.py',
           ROOT/'docs/information_boltzmann/CGB_BOUNDARY_MEMORY_PROBE_20261022.md']
    atomic_json(args.out/'report.json',dict(source=str(args.archive),source_sha256=source_hash,
        scope='Exact frozen 4ms field after matched native jumps; one source bit inverted, other bits fixed. Instrument discrimination, not organism knowledge or text prediction.',
        records=records,code_sha256={str(path.relative_to(ROOT)):digest(path) for path in paths}))
    for path in paths:(args.out/path.name).write_bytes(path.read_bytes())
    print({'naturally_eligible_ports':len(records),'source_unchanged':True},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
