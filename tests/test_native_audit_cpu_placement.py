"""Explicit diagnostic placement changes execution device, not restore steps."""
from types import SimpleNamespace

import pytest

from scripts.ib.fly_interface.native_causal_audit import trainer_restore_program


SOURCE = '''
def main():
    parser = make_parser()
    global marker
    if is_scratch_start:
        pass
    else:
        print('Loading archived weights')
        model = original.cuda()
        state = original.to('cuda')
        learner.load_interaction_state('retained')
        if args.dan_plastic_lr > 0:
            pass
'''


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_native_restore_cpu_selection_keeps_interaction_call_and_device_explicit(device):
    calls = []
    class Placement:
        def cuda(self):
            calls.append('cuda')
            return self
        def to(self, target):
            calls.append(target)
            return self
    _, code = trainer_restore_program(SOURCE, device=device)
    namespace = {'original': Placement(), 'learner': SimpleNamespace(
        load_interaction_state=lambda state: calls.append(state))}
    exec(code, namespace)
    assert calls == [device, device, 'retained']


def test_native_restore_refuses_unknown_placement_or_unreviewed_cuda_overload():
    with pytest.raises(ValueError, match='must be'):
        trainer_restore_program(SOURCE, device='other')
    with pytest.raises(ValueError, match='argument-bearing'):
        trainer_restore_program(SOURCE.replace('.cuda()', '.cuda(1)'), device='cpu')
