"""Instrument first-field sensory evidence on the recorded real byte future."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.cgb.exchange_checkpoint import load_exchange, save_exchange
from information_boltzmann.cgb.sensory_channel import (
    field_channel, byte_posterior, uniform_information_bound)
from scripts.ib.generative.compare_support_transport import digest, same
from scripts.ib.generative.native_learning_stream import advance_byte, atomic_json


def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    model, metadata = load_exchange(args.archive)
    sham, sham_meta = load_exchange(args.archive)
    session, sham_session = copy.deepcopy(metadata['session']), copy.deepcopy(sham_meta['session'])
    sources = metadata['sources']
    if digest(args.run/'stream_blocks.npz') != sources['selected_stream_sha256']:
        raise ValueError('Recorded future changed')
    blocks = np.load(args.run/'stream_blocks.npz', allow_pickle=False)
    block = blocks[session['phase']]
    cursor = session['next_phase_byte']
    if cursor+args.bytes > len(block):
        raise ValueError('Instrument budget must stay inside recorded phase')
    prior = (np.asarray(session['byte_counts'])+.5)/(session['history_count']+128.)
    original_field = model._field
    records, captured = [], []
    for index in range(cursor, cursor+args.bytes):
        target = int(block[index])
        captured.clear()

        def instrument(dt, voltage):
            # First motion/reaction stage has not used the new byte. Full
            # pre-field state is conditioned on, including hidden world charge.
            if not captured:
                channel = field_channel(model, dt)
                original_field(dt, voltage)
                measured = model.charge[channel['observed_nodes']]
                uniform = byte_posterior(channel, measured)
                learned_prior = byte_posterior(channel, measured, prior=prior)
                captured.append(dict(time_ms=model.t, field_ms=dt,
                    posterior_nll_uniform=-float(uniform[target]),
                    posterior_nll_fixed_prior=-float(learned_prior[target]),
                    realized_log_evidence_ratio=float(learned_prior[target]-np.log(prior[target])),
                    uniform_input_information_bound_nats=uniform_information_bound(channel),
                    gain_frobenius=float(np.linalg.norm(channel['gain'])),
                    noise_covariance_trace=float(np.trace(channel['covariance'])),
                    workspace_numeric_bytes=channel['workspace_numeric_bytes'],
                    workspace_scope=channel['workspace_scope']))
            else:
                original_field(dt, voltage)

        model._field = instrument
        row = advance_byte(model, session, target, session['phase'])
        other = advance_byte(sham, sham_session, target, sham_session['phase'])
        assert row == other
        same(model.state_dict(), sham.state_dict())
        assert session == sham_session
        del model._field
        records.append(dict(phase=session['phase'], phase_byte=index, target=target,
                            prearrival_scores=row['scores'], **captured[0]))
    save_exchange(args.out/'complete.npz', model, dict(session=session, sources=sources,
                  audit='read-only first-field sensory instrument, sham exact'))
    result = dict(passed=True, source_archive=str(args.archive), source_sha256=digest(args.archive),
        bytes=args.bytes, first_phase_byte=cursor, final_ms=model.t,
        all_physical_state_three_rngs_and_prearrival_scores_sham_exact=True,
        scope='post-arrival first-field evidence conditional on full pre-field physical state; not agent posterior or prequential prediction',
        natural_text_information_gain='finite realized log evidence ratios; no uniform-input capacity claim for this nonuniform sample',
        mean={key:float(np.mean([r[key] for r in records])) for key in (
            'posterior_nll_uniform','posterior_nll_fixed_prior','realized_log_evidence_ratio',
            'uniform_input_information_bound_nats','gain_frobenius','noise_covariance_trace')},
        records=records)
    atomic_json(args.out/'report.json', result)
    for path in (Path(__file__), ROOT/'information_boltzmann/cgb/sensory_channel.py'):
        (args.out/path.name).write_bytes(path.read_bytes())
    print(json.dumps({k:v for k,v in result.items() if k != 'records'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bytes', type=int, default=16)
    args = parser.parse_args()
    if args.bytes <= 0:
        parser.error('Positive observation budget required')
    main(args)
