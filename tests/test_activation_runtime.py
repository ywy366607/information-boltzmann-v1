"""Two complete events preserve delayed learning across storage tiers."""
import gc

import numpy as np
import pytest
import torch

from scripts.native_policy_activation_storage import configure_runtime_storage, event_storage
from test_joint_world_cycle import complete_interface_engine


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA handoff storage')
def test_handoff_resumes_under_current_event_storage(tmp_path):
    """The rhythm yields at arrival and after backward; every resume gets a live store."""
    from types import SimpleNamespace
    from greenlet import greenlet, getcurrent
    model = torch.nn.Linear(32, 32).cuda()
    model.objective = lambda x: model(x).square().sum()
    engine = SimpleNamespace(model=model, params=list(model.parameters()),
                             _policy_grad=lambda: None)
    x = torch.randn(32, 32, device='cuda')
    expected = torch.autograd.grad(model.objective(x), engine.params)
    configure_runtime_storage(engine, tmp_path, mode='disk', threshold_bytes=1)
    parent = getcurrent()

    def transaction():
        parent.switch('world_arrived')
        for _ in range(2):
            gradients = torch.autograd.grad(model.objective(x), engine.params)
            for a, b in zip(gradients, expected):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            parent.switch('updated')

    task = greenlet(transaction)
    for phase in ('world_arrived', 'updated', 'updated'):
        with event_storage(engine):
            assert task.switch() == phase
        assert engine._activation_hooks is None
        if phase == 'updated':
            assert engine._world_activation_receipt['disk_written_bytes'] > 0
    with event_storage(engine):
        task.switch()
    assert task.dead and not list(tmp_path.rglob('*.bin'))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA storage integration')
def test_complete_events_match_native_with_host_and_disk_spill(tmp_path):
    from scripts.ib_local.language_navigation_turns_v3 import DecisionGraphV3
    from scripts.ib_local.run_causal_water_reply import to_device

    def run(mode):
        torch.set_num_threads(1)
        e = complete_interface_engine()
        torch.cuda.manual_seed_all(89)
        e.model.to('cuda')
        for name, value in list(vars(e).items()):
            if name not in {'model', 'params', 'optimizer', 'groups', 'decision', 'world', 'table'}:
                setattr(e, name, to_device(value, 'cuda'))
        e.device = torch.device('cuda')
        e.decision = DecisionGraphV3(e.model, capture=False)
        configure_runtime_storage(e, tmp_path/mode, mode=mode, threshold_bytes=1,
                                  gpu_budget_bytes=0, host_budget_bytes=16384)
        rows = []
        for _ in range(2):
            with event_storage(e):
                rows.append(e.step(torch.randn(8, 4, device='cuda')))
        from scripts.ib_local.checkpoint_digest import state_digest
        weights = state_digest(to_device(e.model.state_dict(), 'cpu'))
        recurrent = e.recurrent.cpu().clone()
        table = e.table.state_dict()
        receipt = e._world_activation_receipt
        # Runtime closures/streams must never leak into a durable checkpoint.
        import gzip
        from scripts.ib_local.navigation_checkpoint import save
        (tmp_path/mode).mkdir(parents=True, exist_ok=True)
        save(e, tmp_path/mode, 2, {})
        with gzip.open(tmp_path/mode/'last.pt.gz', 'rb') as f:
            saved = torch.load(f, map_location='cpu', weights_only=False)
        assert not any(k.startswith('_activation_') or k.endswith('_activation_receipt') or
                       k == '_policy_grad' for k in saved['engine'])
        del saved
        # The wrapper closure deliberately refers to its owner; collect it.
        del e
        gc.collect()
        torch.cuda.empty_cache()
        return rows, weights, recurrent, table, receipt

    native = run('device')
    for mode in ('disk', 'tiered'):
        other = run(mode)
        for a, b in zip(native[0], other[0]):
            for key in ('choice', 'utterance_hex', 'external_bytes', 'updates',
                        'choice_probability', 'metal_g', 'policy_replay_max_error'):
                assert a[key] == b[key], key
        assert native[1] == other[1]
        torch.testing.assert_close(native[2], other[2], atol=0, rtol=0)
        for k in native[3]['arrays']:
            assert np.array_equal(native[3]['arrays'][k], other[3]['arrays'][k])
        assert other[4]['disk_written_bytes'] > 0
    assert not list(tmp_path.rglob('*.bin'))
