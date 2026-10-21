"""Paired real-stream execution of the same field generator, fully saved."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.cgb.exchange_checkpoint import load_exchange, save_exchange
from information_boltzmann.cgb.sensory_channel import frozen_field_channel, exact_field_reference
from scripts.ib.generative.compare_support_transport import digest, same
from scripts.ib.generative.native_learning_stream import advance_byte, atomic_json
from scripts.ib.generative.sensory_exposure_audit import rooted_nodes


def resolution(model):
    original = model.field_scheme
    nodes = rooted_nodes(model)
    exact = exact_field_reference(model, 4., nodes, np.arange(8))
    table = []
    for scheme in ('sequential_ou', 'symmetric_ou'):
        model.field_scheme = scheme
        for k in (2,4,8,16):
            current = frozen_field_channel(model, [4./k]*k, nodes, np.arange(8))
            table.append(dict(scheme=scheme,k=k,exposure_ms=4.,
                gain_relative_error=float(np.linalg.norm(current['gain']-exact['gain'])/np.linalg.norm(exact['gain'])),
                covariance_relative_error=float(np.linalg.norm(current['covariance']-exact['covariance'])/np.linalg.norm(exact['covariance']))))
    model.field_scheme = original
    return dict(time_ms=model.t,scope='same complete checkpoint, frozen-field reference only',rows=table)


def main(args):
    args.out.mkdir(parents=True,exist_ok=False)
    original_hash = digest(args.archive)
    source_model, metadata = load_exchange(args.archive)
    if digest(args.run/'stream_blocks.npz') != metadata['sources']['selected_stream_sha256']:
        raise ValueError('Original external experience changed')
    blocks = np.load(args.run/'stream_blocks.npz',allow_pickle=False)
    phase = metadata['session']['phase']
    cursor = metadata['session']['next_phase_byte']
    block = blocks[phase][cursor:cursor+128]
    if len(block)!=128:
        raise ValueError('Need 128 recorded future bytes in the current phase')
    old_rows = [json.loads(line) for line in (args.run/'native_scores.jsonl').read_text().splitlines()]
    old_rows = {row['phase_byte']:row for row in old_rows if row['phase']==phase}
    code_paths = [Path(__file__), ROOT/'information_boltzmann/cgb/mobile_exchange.py',
        ROOT/'information_boltzmann/cgb/identity_material.py',ROOT/'information_boltzmann/cgb/sensory_channel.py',
        ROOT/'docs/information_boltzmann/CGB_FIELD_SYMMETRY_PREREG_20261022.md']
    hashes = {str(path.relative_to(ROOT)):digest(path) for path in code_paths}
    snapshot = args.out/'source_snapshot'
    snapshot.mkdir()
    for path in code_paths:
        (snapshot/path.name).write_bytes(path.read_bytes())
    initial = source_model.state_dict()
    results = {}
    for scheme in ('sequential_ou','symmetric_ou'):
        model = type(source_model).from_state_dict(initial)
        session = copy.deepcopy(metadata['session'])
        model.field_scheme = scheme
        assert model.energy()==source_model.energy() and model.free_energy()==source_model.free_energy()
        session['interventions'].append(dict(time_ms=model.t,field_scheme=scheme,
            scope='same physical generator, numerical composition choice',state_erased=False,instantaneous_energy_change=0.))
        diagnostics = [resolution(model)]
        metrics = []
        counts = dict(model.reaction_counts)
        errors = {key:0. for key in model.balances()}
        work0, heat0 = model.work_in, model.heat_out
        fork = fork_session = None
        started = time.perf_counter()
        for index,target in enumerate(block):
            row=advance_byte(model,session,int(target),phase)
            row.update(phase=phase,phase_byte=cursor+index)
            if scheme=='sequential_ou':
                assert row==old_rows[cursor+index]
            if fork is not None:
                other=advance_byte(fork,fork_session,int(target),phase)
                other.update(phase=phase,phase_byte=cursor+index)
                assert row==other
            metrics.append(row)
            for key,value in model.balances().items():
                errors[key]=max(errors[key],abs(float(value)))
            if index==63:
                path=args.out/(scheme+'_mid.npz')
                save_exchange(path,model,dict(session=copy.deepcopy(session),sources=metadata['sources'],
                    numerical_composition=scheme,byte_ms=4.,dt_ms=2.))
                fork,restored=load_exchange(path)
                fork_session=copy.deepcopy(restored['session'])
                diagnostics.append(resolution(model))
        same(model.state_dict(),fork.state_dict())
        assert session==fork_session
        seconds=time.perf_counter()-started
        assert max(errors.values())<1e-8
        diagnostics.append(resolution(model))
        path=args.out/(scheme+'_complete.npz')
        save_exchange(path,model,dict(session=session,sources=metadata['sources'],
            numerical_composition=scheme,restored_future_bytes_exact=64,byte_ms=4.,dt_ms=2.))
        with (args.out/(scheme+'_scores.jsonl')).open('w',encoding='utf-8') as output:
            for row in metrics:
                output.write(json.dumps(row,separators=(',',':'))+'\n')
        results[scheme]=dict(actual_bytes=128,start_ms=source_model.t,end_ms=model.t,
            nll_nats_per_byte={key:float(np.mean([row['scores'][key] for row in metrics])) for key in metrics[0]['scores']},
            seconds_including_restored_future64_and_diagnostics=seconds,
            actual_reactions={key:model.reaction_counts.get(key,0)-counts.get(key,0)
                              for key in set(model.reaction_counts)|set(counts)},
            errors=errors,source_work=model.work_in-work0,heat_out=model.heat_out-heat0,
            final_bridges=len(model.pre),final_trace=int(model.trace.sum()),final_support=int(model.bound.sum()),
            archive_schema=model.state_dict()['schema'],archive_sha256=digest(path),
            restored_future_bytes=64,all_future_scores_full_state_three_rngs_and_session_exact=True,
            frozen_resolution=diagnostics,
            field_base_buffer_bytes=8*(2*(model.n+len(model.port_nodes))+len(model.pre)),
            field_buffer_scope='q, capacitance and conductance buffers, excludes temporary port vectors and Python/RNG overhead')
        print(json.dumps(dict(scheme=scheme,**{k:v for k,v in results[scheme].items() if k not in ('frozen_resolution','actual_reactions')})),flush=True)
    assert digest(args.archive)==original_hash
    assert hashes=={str(path.relative_to(ROOT)):digest(path) for path in code_paths}
    atomic_json(args.out/'report.json',dict(passed=True,baseline_git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        original_archive_sha256=original_hash,code_sha256=hashes,arms=results,
        continuous_time_field_reference='same frozen generator, bounded diagnostic, not production dense execution',
        legacy_original_128_prearrival_scores_exact=True,original_archive_unchanged=True,
        claim='Numerical field accuracy and full-state continuation; not improved predictive learning or complete biological dynamics'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
