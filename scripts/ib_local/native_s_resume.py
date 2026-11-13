"""Validate a committed research prefix; never splice unsaved later updates."""
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(65536), b''):
            value.update(block)
    return value.hexdigest()


def resume_prefix(directory, current_report):
    directory = Path(directory).resolve()
    prior = read(directory/'progress.json')
    if prior['arm'] != 'raw' or current_report['arm'] != 'raw':
        raise ValueError('only the interrupted raw research branch may resume')
    for key in ('source_sha256', 'historical_runtime_files', 'runtime_manifest_sha256', 'data_sources'):
        if prior[key] != current_report[key]:
            raise ValueError(f'resume provenance differs: {key}')
    allowed = {'disable_joint_archive', 'collect_window_garbage', 'max_owned_archive_bytes'}
    for key, value in prior['configuration'].items():
        if key not in allowed and value != current_report['configuration'].get(key):
            raise ValueError(f'resume learning configuration differs: {key}')
    for path, value in prior['execution_sources'].items():
        if Path(path).name != 'run_native_s_credit_retention.py' and value != current_report['execution_sources'].get(path):
            raise ValueError(f'non-storage execution source changed: {path}')
    latest = read(directory/'latest.json')
    path = (directory/latest['current']).resolve()
    if path.parent != directory or not path.name.startswith('checkpoint_'):
        raise ValueError('resume checkpoint escaped original run')
    formal = int(path.stem.split('_')[-1])
    if formal not in (32, 64) or prior['formal_updates'] < formal:
        raise ValueError('resume requires a committed nonterminal phase checkpoint')
    if path.stat().st_size != latest['bytes'] or digest(path) != latest['sha256']:
        raise ValueError('resume disk checkpoint differs from committed manifest')
    windows = [json.loads(line) for line in (directory/'windows.jsonl').read_text().splitlines()][:formal+1]
    arrivals = [json.loads(line) for line in (directory/'arrivals.jsonl').read_text().splitlines()][:formal+1]
    if [w['formal_update'] for w in windows] != list(range(formal+1)) or len(arrivals) != formal+1:
        raise ValueError('committed prefix journals missing or out of order')
    cursor = prior['bridge_tokens']+1+formal*128
    # The window settles BEFORE its observed next token becomes the first input
    # of the following window. Disk checkpoints are AFTER that real arrival.
    # Their two history hashes therefore must not be equated.
    if (arrivals[-1]['stop'] != cursor or
        arrivals[-1]['events'] != prior['source_events']+cursor or
        windows[-1]['stream_events'] != prior['source_events']+cursor-1):
        raise ValueError('resume prefix cursor or history chain differs')
    return dict(path=path, sha256=latest['sha256'], formal=formal, cursor=cursor,
                prior=prior, windows=windows, arrivals=arrivals)


def validate_resume_state(state, prefix):
    prior, formal, cursor = prefix['prior'], prefix['formal'], prefix['cursor']
    saved = state['branch_driver']
    if saved != dict(next_arrival=cursor, formal_updates=formal, arm='raw'):
        raise ValueError('resume checkpoint driver cursor differs')
    learner, stream = state['learner'], state['stream']
    if (learner['updates'] != prior['source_updates']+formal+1 or
        learner['cache']['seen_tokens'] != prior['source_seen']+cursor or
        stream['events'] != prior['source_events']+cursor or
        stream['chain'] != prefix['arrivals'][-1]['history_chain'] or
        len(stream['inputs']) != 1 or stream['targets'] or
        not learner.get('native_s_sidecar') or learner['native_s_sidecar']['mode'] != 'raw'):
        raise ValueError('resume learner/stream/qualification does not match committed prefix')
