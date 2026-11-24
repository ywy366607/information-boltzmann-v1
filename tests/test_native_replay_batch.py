"""Independent actual-operator trajectories under versioned read masks."""
import pytest
import torch

from scripts.ib_local.native_replay_batch import replay_scope, for_model
from scripts.ib_local.native_retention_feedback import without_identity, measure, language_loss
from scripts.ib_local.question_language_port import observation_scope
from test_native_retention_feedback import body


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@torch.no_grad()
def test_private_batches_preserve_each_masked_trajectory_and_question_scope(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA batch operator contract')
    torch.manual_seed(23091)
    e = body()
    m = e.model.to(device)
    if device == 'cuda':
        from scripts.ib_local import gdn_fused
        gdn_fused.enable()
    e.recurrent = e.recurrent.to(device).normal_(0, .015)
    m._question_context = torch.randn(m.encoder.out_features, device=device)
    ids = torch.tensor([[11, 12, 13, 14, 15, 16]], device=device)
    lp = torch.full((1, 6, 256), -5.54517744448, device=device)
    field = m.body.coevolution
    candidates = [(-1, -1)] + [(int(s), int(v)) for s, v in zip(field.slots_id[:3], field.versions[:3])]
    candidates += [(candidates[1][0], candidates[1][1]+1)]
    valid = field.valid
    before = {k:v.clone() for k,v in m.state_dict().items() if torch.is_tensor(v)}
    rng = torch.get_rng_state()
    with observation_scope(m, 2, 6):
        original_mask = m._question_input_mask
        expected, expected_state, expected_slots = [], [], []
        for slot, version in candidates:
            with without_identity(m, slot, version):
                out = m.predict_observed(ids, e.recurrent, lp)
            expected.append(out[0]); expected_state.append(out[1]); expected_slots.append(out[4].slots)
        for start in range(0, len(candidates), 3):
            group = candidates[start:start+3]
            b = len(group)
            with replay_scope(m, group, len(ids[0])):
                actual = m.predict_observed(ids.expand(b,-1), e.recurrent.expand(-1,b,-1).contiguous(), lp.expand(b,-1,-1))
            torch.testing.assert_close(actual[0], torch.cat(expected[start:start+b]), atol=3e-5, rtol=2e-4)
            torch.testing.assert_close(actual[1], torch.cat(expected_state[start:start+b],1), atol=3e-5, rtol=2e-4)
            assert torch.equal(actual[4].slots, torch.cat(expected_slots[start:start+b]))
        assert m._question_input_mask is original_mask
    assert field.valid is valid and for_model(m) is None
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in before.items():
        assert torch.equal(value, m.state_dict()[name]), name


@pytest.mark.parametrize('reuse_actual', [False, True])
def test_batch_measurement_keeps_original_live_gradient_and_read_credit(reuse_actual):
    torch.manual_seed(5223)
    e = body()
    m = e.model
    ids = torch.tensor([[11, 12, 13, 14, 15, 16]])
    lp = torch.full((1, 6, 256), -5.54517744448)
    m.memory.begin_read_credit()
    live = m.predict(ids, e.recurrent, lp)[0]
    original_receipts = len(m.memory._read_participation)
    original = dict(logits=live)
    measure(e, ids, ids+1, e.recurrent, lp, original)
    e.native_replay_batch_size = 3
    e.native_replay_reuse_actual = reuse_actual
    batched = dict(logits=live)
    measure(e, ids, ids+1, e.recurrent, lp, batched)
    a, b = (v['retention_feedback']['observations'] for v in (original, batched))
    assert [(r['slot'],r['version'],r['age_bytes']) for r in a] == [(r['slot'],r['version'],r['age_bytes']) for r in b]
    torch.testing.assert_close(torch.tensor([r['benefit'] for r in a]), torch.tensor([r['benefit'] for r in b]), atol=2e-5, rtol=2e-4)
    assert len(m.memory._read_participation) == original_receipts
    if reuse_actual:
        assert batched['retention_feedback']['replay_logit_max_error'] is None
        assert batched['retention_feedback']['factual_source'].startswith('actual objective')
    language_loss(live, ids+1).backward()
    assert m.body.coevolution.bind.weight.grad.norm() > 0
    assert all(layer.q_proj.weight.grad.norm() > 0 for layer in m.base.layers)


def test_runtime_settings_restore_private_width_and_original_factual_source():
    from scripts.ib_local.native_replay_batch import configure
    e = body()
    config = {}
    configure(e, config, batch_size=8, reuse_actual=True)
    other = body()
    configure(other, config)
    assert other.native_replay_batch_size == 8 and other.native_replay_reuse_actual
    assert for_model(other.model) is None


def test_private_scope_rejects_grad_and_restores_after_exception():
    e = body()
    with pytest.raises(ValueError):
        with replay_scope(e.model, [(-1,-1)], 2):
            pass
    with torch.no_grad(), observation_scope(e.model, 0, 2):
        previous = e.model._question_input_mask
        with pytest.raises(RuntimeError):
            with replay_scope(e.model, [(-1,-1),(-1,-1)], 2):
                raise RuntimeError('cancelled private replay')
        assert for_model(e.model) is None
        assert e.model._question_input_mask is previous
