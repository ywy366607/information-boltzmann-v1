"""Arm-preserving fault recovery for existing native-S research branches.

Extends the raw-only validation adapter without changing any learning code.
Uncommitted windows are excluded and each arm must restore its own mode.
"""
import json
from pathlib import Path

from scripts.ib_local.native_s_resume import read, digest


ARMS = ('raw', 'cut', 'mismatch')


def resume_prefix(directory, current_report):
    directory = Path(directory).resolve()
    prior = read(directory/'progress.json')
    arm = current_report['arm']
    if arm not in ARMS or prior['arm'] != arm:
        raise ValueError('resume arm must match the interrupted research branch')
    for key in ('source_sha256', 'historical_runtime_files', 'runtime_manifest_sha256', 'data_sources'):
        if prior[key] != current_report[key]:
            raise ValueError(f'resume provenance differs: {key}')
    allowed = {'disable_joint_archive', 'collect_window_garbage', 'max_owned_archive_bytes'}
    for key in set(prior['configuration']) | set(current_report['configuration']):
        if key not in allowed and prior['configuration'].get(key) != current_report['configuration'].get(key):
            raise ValueError(f'resume learning configuration differs: {key}')
    for path, value in prior['execution_sources'].items():
        if value != current_report['execution_sources'].get(path):
            raise ValueError(f'resume execution source changed: {path}')
    latest = read(directory/'latest.json')
    path = (directory/latest['current']).resolve()
    if path.parent != directory or not path.name.startswith('checkpoint_'):
        raise ValueError('resume checkpoint escaped original run')
    formal = int(path.stem.split('_')[-1])
    if formal not in (32, 64) or prior['formal_updates'] < formal:
        raise ValueError('resume requires a committed nonterminal phase checkpoint')
    if path.stat().st_size != latest['bytes'] or digest(path) != latest['sha256']:
        raise ValueError('resume disk checkpoint differs from committed manifest')
    windows = [json.loads(s) for s in (directory/'windows.jsonl').read_text().splitlines()][:formal+1]
    arrivals = [json.loads(s) for s in (directory/'arrivals.jsonl').read_text().splitlines()][:formal+1]
    if [w['formal_update'] for w in windows] != list(range(formal+1)) or len(arrivals) != formal+1:
        raise ValueError('committed prefix journals missing or out of order')
    cursor = prior['bridge_tokens']+1+formal*128
    if (arrivals[-1]['stop'] != cursor or
            arrivals[-1]['events'] != prior['source_events']+cursor or
            windows[-1]['stream_events'] != prior['source_events']+cursor-1):
        raise ValueError('resume prefix cursor differs')
    return dict(path=path, sha256=latest['sha256'], formal=formal, cursor=cursor,
                prior=prior, windows=windows, arrivals=arrivals, arm=arm)


def validate_resume_state(state, prefix):
    prior, formal, cursor, arm = (prefix[k] for k in ('prior', 'formal', 'cursor', 'arm'))
    if state['branch_driver'] != dict(next_arrival=cursor, formal_updates=formal, arm=arm):
        raise ValueError('resume checkpoint driver cursor or arm differs')
    learner, stream = state['learner'], state['stream']
    if (learner['updates'] != prior['source_updates']+formal+1 or
            learner['cache']['seen_tokens'] != prior['source_seen']+cursor or
            stream['events'] != prior['source_events']+cursor or
            stream['chain'] != prefix['arrivals'][-1]['history_chain'] or
            len(stream['inputs']) != 1 or stream['targets'] or
            not learner.get('native_s_sidecar') or learner['native_s_sidecar']['mode'] != arm):
        raise ValueError('resume learner/stream/qualification does not match committed prefix')
