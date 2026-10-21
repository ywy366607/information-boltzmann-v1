"""Matched finite trace/capture/channel controls on retained real input."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.cgb.checkpoint import load_medium
from information_boltzmann.cgb.exchange_checkpoint import load_exchange, save_exchange
from information_boltzmann.cgb.trace_exchange import TraceExchange, TraceExchangeParams
from scripts.ib.generative.compare_support_transport import digest, same
from scripts.ib.generative.driven_ports_audit import input_voltage


def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    m, inherited = load_medium(args.archive)
    source_hash = digest(args.archive)
    shelf_path = ROOT/'data/harness/reading/shelf.json'
    story = json.loads(shelf_path.read_text(encoding='utf-8'))['entries'][0]
    data = story['text'].encode()[:16]
    paths = [Path(__file__)] + [ROOT/'information_boltzmann/cgb'/name for name in (
        'trace_exchange.py','driven_exchange.py','flow_exchange.py','structural_exchange.py',
        'finite_exchange.py','material_reactions.py','exchange_checkpoint.py')]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    snapshot = args.out/'source_snapshot'
    snapshot.mkdir()
    for p in paths:
        (snapshot/p.name).write_bytes(p.read_bytes())
    reports, states = {}, {}
    p = TraceExchangeParams()
    arms = dict(coupled=p, no_capture=replace(p,capture_coupling=0.),
                flat_channel=replace(p,gate_low=1.,gate_high=1.), no_write=replace(p,write_rate=0.))
    for label, params in arms.items():
        state = TraceExchange(m.pre,m.post,np.ones(m.E),m.Nf,m.T,np.zeros(m.N),
            np.full(m.N,.5),np.full(m.N,.5),np.full(m.E,2.),np.full(m.N,.5),np.zeros(m.N),
            params=params,port_nodes=np.arange(8),world_capacitance=2.,port_conductance=.05,
            source_conductance=.1)
        errors = {name:0. for name in state.balances()}
        fork = None
        least = 0.
        started = time.perf_counter()
        for k in range(4*len(data)):
            voltage = input_voltage(data,k,1.)
            update = state.step(1.,world_voltage=voltage)
            least = min(least,update['free_energy_drop'])
            if fork is not None:
                fork.step(1.,world_voltage=voltage)
            for name,value in state.balances().items():
                errors[name] = max(errors[name],float(np.max(np.abs(value),initial=0.)))
            if k+1 == 2*len(data):
                save_exchange(args.out/(label+'_half.npz'),state,
                    dict(next_step=k+1,byte_stream_hex=data.hex(),byte_ms=4.,dt_ms=1.))
                fork,_ = load_exchange(args.out/(label+'_half.npz'))
        same(state.state_dict(),fork.state_dict())
        assert least >= -1e-10 and max(errors.values()) < 1e-8
        path = args.out/(label+'_complete.npz')
        save_exchange(path,state,dict(source_sha256=source_hash,params=asdict(params),
            next_step=4*len(data),byte_stream_hex=data.hex(),byte_ms=4.,dt_ms=1.))
        reports[label] = dict(elapsed_ms=state.t,seconds_with_restore=time.perf_counter()-started,
            trace=float(state.trace.sum()),written=float(state.trace_written.sum()),
            bound=float(state.bound.sum()),installed=float(state.capacity.sum()),
            charged=float(state.charged.sum()),source_work=state.source_work,trace_heat=state.trace_heat,
            maximum_balance_errors=errors,minimum_irreversible_g_drop=least,
            complete_restore_future_bitwise=True,archive=str(path),sha256=digest(path))
        states[label] = state
        print(json.dumps(dict(arm=label,**reports[label])),flush=True)
    differences = {}
    for label in ('no_capture','flat_channel','no_write'):
        differences[label] = {name:float(np.max(np.abs(getattr(states['coupled'],name)
            -getattr(states[label],name)),initial=0.)) for name in ('trace','bound','capacity','charge')}
    assert differences['no_capture']['bound'] > 1e-8
    assert differences['flat_channel']['charge'] > 1e-8
    assert differences['no_write']['trace'] > 1e-8
    a = states['coupled']
    report = dict(date='2026-10-22',scope='finite physical trace-to-capture/rate/turnover delivery',
        baseline_git=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),
        preregistered='docs/information_boltzmann/CGB_TRACE_GROWTH_PREREG_20261022.md',
        source_archive=str(args.archive),source_sha256=source_hash,inherited_provenance=inherited,
        source_unchanged=digest(args.archive)==source_hash,cells=m.N,contacts=m.E,
        story=story['title'],story_source=story['source'],reading_shelf_sha256=digest(shelf_path),
        byte_stream_hex=data.hex(),bytes_presented=len(data),byte_ms=4.,dt_ms=1.,
        arms=reports,paired_differences=differences,code_sha256=hashes,
        state_array_bytes=sum(getattr(a,k).nbytes for k in a._arrays)+sum(v.nbytes for v in (a.pre,a.post,a.capacity,a.port_nodes)),
        full_restore_future_bitwise=True,learning_or_behavior_benefit='unmeasured',
        biological_E5_signed_tag_equivalence='not claimed; physical implementation of trace/capture/retention route',passed=True)
    assert report['source_unchanged']
    assert hashes == {str(p.relative_to(ROOT)):digest(p) for p in paths}
    (args.out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
