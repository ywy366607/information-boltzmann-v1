import torch

from scripts.ib_local import autonomous_cycle, latent_action_planner as planner
from scripts.ib_local import persistent_rhythm as rhythm
from scripts.root_option_calibration_shadow import observe
from test_persistent_rhythm import body


def external(engine):
    for _ in range(96):
        row = autonomous_cycle.dispatch(engine)
        if row['autonomous_event']['kind'] == 'external':
            return row
    raise AssertionError('external transaction must complete')


def test_persistent_read_uses_scalar_bank_without_action_q(monkeypatch, tmp_path):
    torch.set_num_threads(1)
    e, archive = body(tmp_path)
    try:
        from scripts.ib_local.prospective_decision import preview_bind
        source = e._read_source()
        state, _ = preview_bind(e.model, source)
        def unexpected(*args, **kwargs):
            raise AssertionError('persistent scalar read must not invoke action q predictor')
        monkeypatch.setattr(planner, 'predict', unexpected)
        values, _, _ = rhythm.read_plans(e, [state], [source], carried=state)
        assert values.shape == (1,)
    finally:
        archive.close()


def test_root_shadow_preserves_complete_learning_and_random_path(tmp_path):
    torch.set_num_threads(1)
    results = []
    original_value, original_read = planner.value, rhythm.read_plans
    for name in ('reference', 'observed'):
        torch.manual_seed(201120)
        e, archive = body(tmp_path/name, root_mode='observe')
        try:
            external(e)
            if name == 'observed':
                with observe(e) as records:
                    row = external(e)
                assert len(records) == 1 and records[0]['calls']
                assert records[0]['extra_teacher_calls'] == 0
            else:
                row = external(e)
            results.append(dict(choice=row['choice'], words=row['utterance_hex'], gain=row['metal_g'],
                state=e.recurrent.clone(), parameters=[p.detach().clone() for p in e.model.parameters()],
                rng=torch.get_rng_state().clone()))
        finally:
            archive.close()
    a, b = results
    for key in ('choice', 'words', 'gain'):
        assert a[key] == b[key]
    torch.testing.assert_close(a['state'], b['state'], atol=0, rtol=0)
    assert all(torch.equal(x, y) for x, y in zip(a['parameters'], b['parameters']))
    assert torch.equal(a['rng'], b['rng'])
    assert planner.value is original_value and rhythm.read_plans is original_read
