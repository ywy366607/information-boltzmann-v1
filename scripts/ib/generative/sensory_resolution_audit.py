"""Exact small-field reference and fixed-duration resolution, no retuning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.cgb.exchange_checkpoint import load_exchange
from information_boltzmann.cgb.sensory_channel import (
    frozen_field_channel, exact_field_reference, uniform_information_bound)
from scripts.ib.generative.sensory_exposure_audit import rooted_nodes
from scripts.ib.generative.native_learning_stream import advance_byte, atomic_json
from scripts.ib.generative.compare_support_transport import digest


def bounds(channel, port_rows, brain_count):
    result = {}
    for name, rows in (('ports', port_rows), ('rooted_brain', np.arange(brain_count)),
                       ('brain_world', np.arange(len(channel['offset'])))):
        part = dict(gain=channel['gain'][rows], covariance=channel['covariance'][np.ix_(rows, rows)])
        result[name] = uniform_information_bound(part)
    return result


def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    model, metadata = load_exchange(args.archive)
    session = metadata['session']
    if digest(args.run/'stream_blocks.npz') != metadata['sources']['selected_stream_sha256']:
        raise ValueError('Original external input changed')
    blocks = np.load(args.run/'stream_blocks.npz', allow_pickle=False)
    original = [json.loads(line) for line in (args.run/'native_scores.jsonl').read_text().splitlines()]
    rows = {(row['phase'],row['phase_byte']):row for row in original}
    records = []
    cursor = session['next_phase_byte']
    for index in range(cursor, cursor+16):
        captured = []
        original_field = model._field

        def observe(dt, voltage):
            if not captured:
                nodes = rooted_nodes(model)
                port_rows = np.array([int(np.flatnonzero(nodes==p)[0]) for p in model.port_nodes])
                exact = exact_field_reference(model, 4., nodes, np.arange(8))
                grid = []
                for k in (2,4,8,16,32,64,128):
                    current = frozen_field_channel(model, [4./k]*k, nodes, np.arange(8))
                    grid.append(dict(k=k, substep_ms=4./k, exposure_ms=4.,
                        gain_relative_error=float(np.linalg.norm(current['gain']-exact['gain'])/np.linalg.norm(exact['gain'])),
                        covariance_relative_error=float(np.linalg.norm(current['covariance']-exact['covariance'])/np.linalg.norm(exact['covariance'])),
                        bounds_nats=bounds(current, port_rows, len(nodes))))
                captured.append(dict(phase_byte=index, time_ms=model.t, rooted_brain_nodes=len(nodes),
                    exact_bounds_nats=bounds(exact, port_rows, len(nodes)), resolution=grid))
            original_field(dt, voltage)

        model._field = observe
        row = advance_byte(model, session, int(blocks[session['phase']][index]), session['phase'])
        row.update(phase=session['phase'],phase_byte=index)
        assert row == rows[session['phase'],index]
        del model._field
        records.append(captured[0])
    summary = dict(exact_bounds_nats={key:float(np.mean([row['exact_bounds_nats'][key] for row in records]))
                                     for key in ('ports','rooted_brain','brain_world')}, resolution=[])
    for i in range(7):
        values = [row['resolution'][i] for row in records]
        summary['resolution'].append(dict(k=values[0]['k'],substep_ms=values[0]['substep_ms'],
            gain_relative_error=float(np.mean([v['gain_relative_error'] for v in values])),
            covariance_relative_error=float(np.mean([v['covariance_relative_error'] for v in values])),
            bounds_nats={key:float(np.mean([v['bounds_nats'][key] for v in values]))
                         for key in ('ports','rooted_brain','brain_world')}))
    atomic_json(args.out/'report.json',dict(passed=True,source_archive_sha256=digest(args.archive),
        actual_bytes=16,actual_clock_preserved_ms=64.,original_prearrival_scores_exact=True,
        scope='bounded frozen-field diagnostic; K resolution differs, T4ms and actual adaptive dt2ms unchanged',
        summary=summary,records=records))
    paths=[Path(__file__), ROOT/'information_boltzmann/cgb/sensory_channel.py',
           ROOT/'docs/information_boltzmann/CGB_SENSORY_EXPOSURE_PREREG_20261022.md']
    for path in paths:
        (args.out/path.name).write_bytes(path.read_bytes())
    atomic_json(args.out/'sources.json',{str(path.relative_to(ROOT)):digest(path) for path in paths})
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args())
