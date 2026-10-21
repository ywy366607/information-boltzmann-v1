"""Resume a recorded branch cursor, then verify its entire recorded future."""
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
from scripts.ib.generative.compare_support_transport import digest, same
from scripts.ib.generative.native_learning_stream import advance_byte, atomic_json


def main(args):
    args.out.mkdir(parents=True, exist_ok=False)
    model, metadata = load_exchange(args.archive)
    session = copy.deepcopy(metadata['session'])
    sources = metadata['sources']
    block_path = args.run/'stream_blocks.npz'
    if digest(block_path) != sources['selected_stream_sha256']:
        raise ValueError('Selected external experience has changed')
    report = json.loads((args.run/'report.json').read_text(encoding='utf-8'))
    # Match the retained implementation, not just the current filename.
    for path, expected in report['code_sha256'].items():
        if digest(ROOT/path) != expected:
            raise ValueError(f'Use the retained implementation for resume: {path}')
    phases = ('fresh_A', 'fresh_B', 'revisit_A')
    phase_index = phases.index(session['phase'])
    if not session['fixed_frequency_frozen']:
        raise ValueError('This command resumes branch checkpoints, not the shared warm transition')
    blocks = np.load(block_path, allow_pickle=False)
    original_rows = [json.loads(line) for line in
                     (args.run/(args.arm+'_scores.jsonl')).read_text(encoding='utf-8').splitlines()]
    row_map = {(row['phase'], row['phase_byte']): row for row in original_rows}
    start_time = model.t
    count = 0
    with (args.out/'scores.jsonl').open('w', encoding='utf-8') as output:
        for phase in phases[phase_index:]:
            cursor = session['next_phase_byte'] if session['phase'] == phase else 0
            session['phase'] = phase
            session['next_phase_byte'] = cursor
            for index in range(cursor, len(blocks[phase])):
                row = advance_byte(model, session, int(blocks[phase][index]), phase)
                row.update(phase=phase, phase_byte=index)
                assert row == row_map[phase, index]
                output.write(json.dumps(row, separators=(',', ':'))+'\n')
                count += 1
    reference_path = args.run/(args.arm+'_revisit_A_complete.npz')
    reference, final_metadata = load_exchange(reference_path)
    same(model.state_dict(), reference.state_dict())
    assert session == final_metadata['session']
    save_exchange(args.out/'complete.npz', model, dict(session=session, sources=sources,
                  resumed_from=str(args.archive), byte_ms=4., dt_ms=2.))
    atomic_json(args.out/'report.json', dict(
        passed=True, start_ms=start_time, end_ms=model.t, future_bytes=count,
        source_archive_sha256=digest(args.archive), reference_sha256=digest(reference_path),
        physical_state_and_three_rngs_bitwise=True, external_session_exact=True,
        all_future_prearrival_predictions_exact=True,
        selected_stream_sha256=digest(block_path)))
    print(json.dumps(dict(passed=True, future_bytes=count, end_ms=model.t)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arm', choices=('native', 'no_trace_capture', 'frozen_adaptation'), default='native')
    main(parser.parse_args())
