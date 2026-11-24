from types import SimpleNamespace

import torch

from scripts.ib_local.action_value_pair_audit import first_action


def test_conditional_actor_forwards_selected_draft_audit_flag():
    calls = []
    def original(targets, own, record, *, reuse_audit=False):
        calls.append(reuse_audit)
        return torch.tensor(1.), dict(reuse_audit=reuse_audit)
    e = SimpleNamespace(_utter=lambda: None, _choice_policy_term=original)
    with first_action(e, 0, lambda _: None, keep_actor_learning=True):
        value, replay = e._choice_policy_term(None, 0, {}, reuse_audit=True)
    assert calls == [True] and value == 1 and replay['reuse_audit']
    assert e._choice_policy_term is original
