"""Actual conditional input, learning boundary and finite correction checks."""
import torch

from test_relation_retention_use import configured
from scripts.ib_local import epistemic_value_reference as ev
from scripts.ib_local import native_amortized_decision as native


def test_native_sparse_correction_finite_expectation():
    weights = torch.tensor([.7, .2, .1], dtype=torch.float64)
    prediction = torch.tensor([1., -2., 4.], dtype=torch.float64)
    reference = torch.tensor([-.5, 3., 2.], dtype=torch.float64)
    estimates = (weights*prediction).sum()+reference-prediction
    torch.testing.assert_close((weights*estimates).sum(), (weights*reference).sum())
    torch.testing.assert_close((weights*prediction).sum()+(weights*(reference-prediction)).sum(),
                               (weights*reference).sum())


def test_production_polynomial_kernel_has_explicit_positive_feature_map():
    x = torch.tensor([[1., -.2, .3], [-.5, .7, .1], [.2, .3, -.9]], dtype=torch.float64)
    feature = torch.cat((torch.ones(len(x), 1, dtype=x.dtype), 2**.5*x,
                         (x[:, :, None]*x[:, None, :]).flatten(1)), -1)
    torch.testing.assert_close(native._kernel(x, x), feature@feature.T)
    assert torch.linalg.eigvalsh(native._kernel(x, x)+.1*torch.eye(len(x))).min() > 0


def test_same_body_conditioning_predictor_and_actual_world_labels(tmp_path):
    e, archive = configured(tmp_path)
    try:
        e.gain_units_schema = 1
        ev.configure(e, schema=5, word_execution='native_anchor')
        e.native_word_runtime = 'batch'
        e.model._preview_batch_enabled = True
        native.configure(e, dict(schema=1, seed=201120))
        first = e.step(torch.randn(8, 4))
        bank = e.native_value_bank
        assert bank['revision'] == 1 and not bank['pending']
        assert first['causal_use']['native_amortization']['new_world_label']
        assert any(r['kind']=='world' for r in bank['records'])
        assert any(r['kind']=='native' for r in bank['records'])
        assert len(bank['records']) <= e.model.causal_relations.evidence_budget**2
        fitted, base, _ = native.predict(e, [r['packet'] for r in bank['records']])
        target = fitted.new_tensor([r['target'] for r in bank['records']])
        count = {kind: sum(r['kind']==kind for r in bank['records']) for kind in ('native','world')}
        weight = fitted.new_tensor([1/count[r['kind']] for r in bank['records']])
        assert float((weight*(fitted-target).square()).sum()) <= float((weight*(base-target).square()).sum())+1e-6
        packet = bank['records'][0]['packet']
        before = native.features(e, [packet])
        changed = dict(packet, raw=packet['raw'].clone())
        changed['raw'][1] += .7  # SAME raw boundary, different known P port.
        assert not torch.equal(before, native.features(e, [changed]))
        changed = dict(packet, scalars=list(packet['scalars']))
        changed['scalars'][6] += 1  # typed question role is an explicit input
        assert not torch.equal(before, native.features(e, [changed]))
        clock, revision = int(e.model.memory.clock), bank['revision']
        rng = torch.get_rng_state().clone()
        native.predict(e, [packet])
        assert int(e.model.memory.clock) == clock and bank['revision'] == revision
        assert torch.equal(rng, torch.get_rng_state())
        second = e.step(torch.randn(8, 4))
        assert second['causal_use']['finite_U']['verified']
        assert second['causal_program']['successor_learning']['target_excluded']
        assert e.successor_distribution_last['schema'] == 3
        assert e.successor_distribution_last['native_count'] <= e.successor_distribution_last['budget']
        assert e.native_value_bank['revision'] == 2
    finally:
        archive.close()
