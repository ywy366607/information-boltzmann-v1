from types import SimpleNamespace

import torch

from scripts.ib_local.checkpoint_digest import state_digest
from scripts.ib_local.native_s_storage import learning_state_view, stop_future_joint_archives
from scripts.ib_local.pretrained_gdn_experience import JointExperience, LocalExperience


def test_storage_policy_only_disables_future_archive():
    capsule = object()
    learner = SimpleNamespace(joint_capture_enabled=True, memories={1:[capsule]},
                              eligibility=object(), optimizer=object())
    before = dict(learner.__dict__)
    result = stop_future_joint_archives(learner)
    assert result['previous_joint_capture_enabled'] is True
    assert learner.joint_capture_enabled is False
    for name in ('memories', 'eligibility', 'optimizer'):
        assert learner.__dict__[name] is before[name]
    assert learner.memories[1][0] is capsule


def test_learning_equivalence_excludes_only_passive_fields():
    local = torch.arange(6).reshape(1, 3, 2)
    joint = JointExperience(torch.ones(1,128,2), dict(states=[dict(S=torch.ones(2))]), 7)
    a = dict(joint_capture_enabled=True, native=dict(w=torch.ones(2)),
             memories={1:[LocalExperience(local, dict(S=torch.ones(2)), 4, joint=joint)]},
             native_s_sidecar=dict(clock=8, U=torch.ones(2)))
    b = dict(a, joint_capture_enabled=False,
             memories={1:[LocalExperience(local, dict(S=torch.ones(2)), 4, joint=None)]})
    stream = dict(events=9, chain='abc')
    digest = lambda x,s=stream:state_digest(learning_state_view(x,s))['sha256']
    assert digest(a) == digest(b)
    b['memories'][1][0].inputs = local+1
    assert digest(a) != digest(b)
    b['memories'][1][0].inputs = local
    b['native_s_sidecar'] = dict(clock=9, U=torch.ones(2))
    assert digest(a) != digest(b)
    assert digest(a,dict(events=10,chain='abc')) != digest(a)
