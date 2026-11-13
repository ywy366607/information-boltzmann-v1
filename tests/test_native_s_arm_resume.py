import json

import pytest

from test_native_s_resume import fixture, write
from scripts.ib_local.native_s_arm_resume import resume_prefix, validate_resume_state


@pytest.mark.parametrize('arm', ['raw', 'cut', 'mismatch'])
def test_complete_prefix_preserves_each_arm(tmp_path, arm):
    current = fixture(tmp_path)
    current['arm'] = arm
    current['execution_sources']['run_native_s_credit_retention.py'] = 'old_driver'
    prior = json.loads((tmp_path/'progress.json').read_text())
    prior['arm'] = arm
    write(tmp_path/'progress.json', prior)
    prefix = resume_prefix(tmp_path, current)
    assert prefix['formal'] == 32
    state = dict(branch_driver=dict(next_arrival=4156, formal_updates=32, arm=arm),
                 learner=dict(updates=557, cache=dict(seen_tokens=454913), native_s_sidecar=dict(mode=arm)),
                 stream=dict(events=52097, chain='after_next_input32', inputs=[1], targets=[]))
    validate_resume_state(state, prefix)
    state['learner']['native_s_sidecar']['mode'] = 'invalid'
    with pytest.raises(ValueError, match='qualification'):
        validate_resume_state(state, prefix)


def test_resume_rejects_other_arm_and_new_execution(tmp_path):
    current = fixture(tmp_path)
    current['arm'] = 'mismatch'
    with pytest.raises(ValueError, match='arm'):
        resume_prefix(tmp_path, current)
    current['arm'] = 'raw'
    with pytest.raises(ValueError, match='execution source'):
        resume_prefix(tmp_path, current)
