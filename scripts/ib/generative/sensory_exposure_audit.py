"""Complete actual source exposure and explicitly frozen body-channel audit."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.cgb.exchange_checkpoint import load_exchange, save_exchange
from information_boltzmann.cgb.sensory_channel import (
    frozen_field_channel, uniform_information_bound, source_innovation,
    source_path_posterior, uniform_binary_information)
from scripts.ib.generative.compare_support_transport import digest, same
from scripts.ib.generative.native_learning_stream import advance_byte, atomic_json


def rooted_nodes(model):
    adjacency = [[] for _ in range(model.n)]
    for i, j, g in zip(model.pre, model.post, model.bridge_conductance()):
        if g > 0:
            adjacency[i].append(int(j))
            adjacency[j].append(int(i))
    found = set(int(i) for i in model.port_nodes)
    queue = list(found)
    for i in queue:
        for j in adjacency[i]:
            if j not in found:
                found.add(j)
                queue.append(j)
    return np.array(sorted(found), dtype=np.int64)


def channel_record(model, nodes, world=(), durations=(2., 2.)):
    channel = frozen_field_channel(model, durations, nodes, world)
    record = dict(observed_brain_nodes=channel['observed_nodes'].tolist(),
                  observed_world_ports=channel['observed_world_ports'].tolist(),
                  exposure_ms=sum(durations), field_steps=len(durations),
                  gain_norm=float(np.linalg.norm(channel['gain'])),
                  noise_covariance_trace=float(np.trace(channel['covariance'])),
                  operator_buffer_bytes=channel['workspace_numeric_bytes'],
                  scope=channel['scope'])
    try:
        record['uniform_input_gaussian_bound_nats'] = uniform_information_bound(channel)
        record['noise_condition_number'] = float(np.linalg.cond(channel['covariance']))
    except np.linalg.LinAlgError:
        record['uniform_input_gaussian_bound_nats'] = None
        record['error'] = 'Degenerate observation covariance; no noise floor or pseudo-information inserted'
    return record


def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    model, metadata = load_exchange(args.archive)
    sham, sham_meta = load_exchange(args.archive)
    state, sham_state = copy.deepcopy(metadata['session']), copy.deepcopy(sham_meta['session'])
    source_hash = digest(args.archive)
    if digest(args.run/'stream_blocks.npz') != metadata['sources']['selected_stream_sha256']:
        raise ValueError('Recorded input changed')
    blocks = np.load(args.run/'stream_blocks.npz', allow_pickle=False)
    phases = ('fresh_A', 'fresh_B', 'revisit_A')
    first_phase = phases.index(state['phase'])
    original = [json.loads(line) for line in (args.run/'native_scores.jsonl').read_text().splitlines()]
    reference_rows = {(row['phase'], row['phase_byte']): row for row in original}
    prior = (np.array(state['byte_counts'])+.5)/(state['history_count']+128.)
    records, future_count = [], 0
    snapshot = args.out/'source_snapshot'
    snapshot.mkdir()
    sources = [Path(__file__), ROOT/'information_boltzmann/cgb/mobile_exchange.py',
               ROOT/'information_boltzmann/cgb/sensory_channel.py',
               ROOT/'docs/information_boltzmann/CGB_SENSORY_EXPOSURE_PREREG_20261022.md']
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in sources}
    for path in sources:
        (snapshot/path.name).write_bytes(path.read_bytes())
    for phase in phases[first_phase:]:
        cursor = state['next_phase_byte'] if state['phase'] == phase else 0
        state['phase'] = phase
        state['next_phase_byte'] = cursor
        if future_count < args.bytes:
            sham_state['phase'] = phase
            sham_state['next_phase_byte'] = cursor
        for index in range(cursor, len(blocks[phase])):
            target = int(blocks[phase][index])
            innovations, source_events, channels = [], [], {}
            if future_count < args.bytes:
                original_field = model._field

                def source_read(event):
                    innovations.append(source_innovation(model, event))
                    source_events.append({key:(value.tolist() if isinstance(value, np.ndarray) else value)
                                          for key, value in event.items()})

                def field_read(dt, voltage):
                    if not channels:
                        nodes = rooted_nodes(model)
                        channels['port_first_2ms'] = channel_record(model, model.port_nodes, durations=(dt,))
                        channels['ports_4ms'] = channel_record(model, model.port_nodes)
                        channels['rooted_brain_4ms'] = channel_record(model, nodes)
                        channels['brain_world_4ms'] = channel_record(model, nodes, np.arange(8))
                    original_field(dt, voltage)

                model.source_observer = source_read
                model._field = field_read
            row = advance_byte(model, state, target, phase)
            row.update(phase=phase, phase_byte=index)
            assert row == reference_rows[phase, index]
            if future_count < args.bytes:
                other = advance_byte(sham, sham_state, target, phase)
                other.update(phase=phase, phase_byte=index)
                assert row == other
                same(model.state_dict(), sham.state_dict())
                assert state == sham_state
                del model._field
                model.source_observer = None
                assert len(innovations) == 2
                posterior = source_path_posterior(innovations, prior=prior)
                snr = posterior['snr']
                records.append(dict(phase=phase, phase_byte=index, target=target,
                    actual_ms=row['after_ms']-row['before_ms'],
                    source_posterior_nll=-float(posterior['logposterior'][target]),
                    source_realized_log_evidence=float(posterior['logposterior'][target]-np.log(prior[target])),
                    source_uniform_binary_information_nats=float(uniform_binary_information(snr).sum()),
                    source_uniform_gaussian_bound_nats=min(8*np.log(2.), .5*float(np.log1p(snr).sum())),
                    source_work=sum(event['source_work'] for event in source_events),
                    source_heat=sum(event['source_heat'] for event in source_events),
                    source_events=source_events, frozen_channels=channels))
            future_count += 1
            if future_count % 128 == 0:
                print(json.dumps(dict(future_verified=future_count, time_ms=model.t)), flush=True)
    reference, final_meta = load_exchange(args.run/'native_revisit_A_complete.npz')
    same(model.state_dict(), reference.state_dict())
    assert state == final_meta['session']
    assert source_hash == digest(args.archive)
    assert hashes == {str(path.relative_to(ROOT)): digest(path) for path in sources}
    save_exchange(args.out/'complete.npz', model, dict(session=state, sources=metadata['sources'],
                  audit='full actual source path and frozen-field counterfactuals', source_archive_sha256=source_hash))
    means = {key:float(np.mean([r[key] for r in records])) for key in (
        'source_posterior_nll', 'source_realized_log_evidence', 'source_uniform_binary_information_nats',
        'source_uniform_gaussian_bound_nats', 'source_work', 'source_heat')}
    channel_means = {}
    for name in records[0]['frozen_channels']:
        values = [r['frozen_channels'][name]['uniform_input_gaussian_bound_nats'] for r in records]
        channel_means[name] = None if any(value is None for value in values) else float(np.mean(values))
    report = dict(passed=True, baseline_git=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        code_sha256=hashes, archive_sha256=source_hash, records=records, mean=means,
        frozen_channel_mean_bounds_nats=channel_means, actual_source_instrument_bytes=len(records),
        all_original_future_predictions_verified=future_count, final_time_ms=model.t,
        original_final_full_state_session_and_three_rngs_exact=True,
        instrumented_prefix_sham_exact=True, source_archive_unchanged=True,
        interpretation='Source-contact path evidence is not brain knowledge; frozen field endpoints are not adaptive-sequence mutual information; no predictive utility claim.')
    atomic_json(args.out/'report.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','code_sha256')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bytes', type=int, default=16)
    args = parser.parse_args()
    if not 0 < args.bytes <= 512:
        parser.error('Instrument between 1 and 512 future bytes')
    main(args)
