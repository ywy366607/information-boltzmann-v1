import hashlib
import json

import pytest

from scripts.ib_local.native_s_resume import resume_prefix, validate_resume_state


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def fixture(tmp_path):
    checkpoint = tmp_path/'checkpoint_000032.pt'
    checkpoint.write_bytes(b'committed checkpoint fixture, not a learner')
    write(tmp_path/'latest.json', dict(current=checkpoint.name, bytes=checkpoint.stat().st_size,
          sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
    prior = dict(arm='raw', formal_updates=58, bridge_tokens=59, source_updates=524,
                 source_seen=450757, source_events=47941, source_sha256='source',
                 historical_runtime_files={'runtime':'hash'}, runtime_manifest_sha256='manifest',
                 data_sources={'stream':'data'}, configuration=dict(window=128, rank=2),
                 execution_sources={'native_s_learner.py':'learner',
                                    'run_native_s_credit_retention.py':'old_driver'})
    current = dict(prior, configuration=dict(window=128, rank=2, disable_joint_archive=True),
                   execution_sources={'native_s_learner.py':'learner',
                                      'run_native_s_credit_retention.py':'new_storage_driver'})
    write(tmp_path/'progress.json', prior)
    windows = [dict(formal_update=i, stream_events=48000+i*128,
                    history_chain=f'before_next_input{i}') for i in range(59)]
    arrivals = [dict(stop=60+i*128, events=48001+i*128,
                     history_chain=f'after_next_input{i}') for i in range(59)]
    for name, rows in [('windows',windows), ('arrivals',arrivals)]:
        (tmp_path/f'{name}.jsonl').write_text('\n'.join(json.dumps(x) for x in rows))
    return current


def test_resume_uses_only_the_manifested_complete_prefix(tmp_path):
    current = fixture(tmp_path)
    prefix = resume_prefix(tmp_path, current)
    assert prefix['formal'] == 32 and prefix['cursor'] == 4156
    assert len(prefix['windows']) == len(prefix['arrivals']) == 33
    assert prefix['windows'][-1]['history_chain'] == 'before_next_input32'
    state = dict(branch_driver=dict(next_arrival=4156,formal_updates=32,arm='raw'),
                 learner=dict(updates=557,cache=dict(seen_tokens=454913),
                              native_s_sidecar=dict(mode='raw')),
                 stream=dict(events=52097,chain='after_next_input32',inputs=[1],targets=[]))
    validate_resume_state(state,prefix)
    state['learner']['updates'] += 1
    with pytest.raises(ValueError,match='does not match'):
        validate_resume_state(state,prefix)


def test_resume_rejects_changed_learning_or_checkpoint(tmp_path):
    current = fixture(tmp_path)
    current['configuration']['rank'] = 3
    with pytest.raises(ValueError,match='learning configuration'):
        resume_prefix(tmp_path,current)
    current['configuration']['rank'] = 2
    current['execution_sources']['native_s_learner.py'] = 'changed'
    with pytest.raises(ValueError,match='non-storage execution'):
        resume_prefix(tmp_path,current)
    current['execution_sources']['native_s_learner.py'] = 'learner'
    (tmp_path/'checkpoint_000032.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='disk checkpoint'):
        resume_prefix(tmp_path,current)
