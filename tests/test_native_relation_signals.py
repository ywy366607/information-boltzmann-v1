"""Production-function observation/gradient contracts, no learning experiment."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.ib_local import causal_relation_program as cp
from scripts.ib_local import joint_relation_update as jr
from scripts.native_relation_signals import install_relation_signals
from scripts.native_topology_observer import TensorStore, _json, load_tensor


class Recorder:
    def __init__(self, directory):
        self.records, self.undo = [], []
        self.store = TensorStore(directory/'tensors.bin')

    def record(self, kind, data):
        self.records.append((kind, _json(data, self.store)))

    def _replace(self, owner, name, replacement):
        self.undo.append((owner, name, getattr(owner, name)))
        setattr(owner, name, replacement)

    def close(self):
        for owner, name, original in reversed(self.undo):
            setattr(owner, name, original)
        self.store.close()


def fixture_state():
    source = torch.linspace(-.3, .4, 256).reshape(1, -1)
    facts = tuple(cp.Evidence(source+i*.02, source.flip(1)-i*.01, source*.7+i*.03,
        cp.Port.MOTHER, True, fact_id=f'episode:{i+1}') for i in range(2))
    state = cp.RelationState(9, 3, 7, 0, torch.tensor([0, 1]), torch.tensor([2, 3]),
        torch.ones(2, dtype=torch.bool), release=0, constraints=facts,
        evidence_bound=True, program_weights=torch.tensor([.6, .4]))
    encoder = torch.nn.Linear(256, 128, bias=False)
    prior = torch.nn.Linear(256, 128, bias=False)
    mother = torch.nn.Linear(128, 128, bias=False)
    with torch.no_grad():
        encoder.weight.copy_(torch.arange(32768).reshape(128, 256).remainder(17)*.0001)
        prior.weight.fill_(.0002)
        mother.weight.fill_(.0003)
    model = SimpleNamespace(encoder=encoder, long_predictor=prior,
        mother_from_boundary=mother, joint_relation_update_schema=1,
        joint_relation_update_revision=4, memory=SimpleNamespace(
            source=torch.empty(0, 256), action=torch.empty(0, 256),
            mother_next=torch.empty(0, 256), internal_next=torch.empty(0, 256)))
    return model, state, source


def test_actual_mechanism_values_gradients_rng_and_alias_restoration(tmp_path):
    model, state, source = fixture_state()
    original, design_original = cp.mechanism, cp._bound_evidence_design
    expected = original(model, state, source, source.flip(1))
    grad_expected = torch.autograd.grad(expected['mean'].square().sum(), model.encoder.weight)[0]
    recorder = Recorder(tmp_path)
    old_rng = torch.random.get_rng_state().clone()
    try:
        install_relation_signals(recorder)
        # Keyword arguments exercise the signature-preserving wrapper too.
        actual = cp.mechanism(model=model, state=state, source=source, action=source.flip(1))
        grad_actual = torch.autograd.grad(actual['mean'].square().sum(), model.encoder.weight)[0]
        assert torch.equal(expected['mean'], actual['mean'])
        assert torch.equal(grad_expected, grad_actual)
        assert torch.equal(old_rng, torch.random.get_rng_state())
        records = {r['operation']: r for _, r in recorder.records}
        mechanism, design = records['mechanism'], records['_bound_evidence_design']
        assert design['parent_id'] == mechanism['call_id']
        assert design['evidence_row_counts'] == [1, 1]
        assert records['posterior']['parent_id'] == mechanism['call_id']
        assert 'base' not in records['posterior']['prediction']
        assert mechanism['state']['constraints'][0]['fact_id'] == 'episode:1'
        assert 'coefficient' in mechanism['prediction']
        assert mechanism['prediction']['x']['tensor_ref']['shape'][0] == 2
    finally:
        recorder.close()
    assert cp.mechanism is original and cp._bound_evidence_design is design_original
    prediction = records['mechanism']['prediction']
    x = load_tensor(tmp_path/'tensors.bin', prediction['x']).astype(float)
    q = load_tensor(tmp_path/'tensors.bin', prediction['query']).astype(float)
    coefficient = load_tensor(tmp_path/'tensors.bin', prediction['coefficient']).astype(float)
    # Actual evidence-space message contributions recover the native correction.
    messages = (q@x.T)[:, :, None]*coefficient[None, :, :]
    mean = load_tensor(tmp_path/'tensors.bin', prediction['mean'])
    base = load_tensor(tmp_path/'tensors.bin', prediction['base'])
    np.testing.assert_allclose(messages.sum(1), mean-base, atol=2e-8, rtol=2e-5)


def test_functional_revision_with_unchanged_version_is_distinguished(tmp_path):
    _, state, source = fixture_state()
    hypothetical = cp.Evidence(source, source, source*.2, cp.Port.MOTHER, False,
                                fact_id='imagined:unit-interface')
    recorder = Recorder(tmp_path)
    try:
        install_relation_signals(recorder)
        revised = cp.revise(state=state, evidence=hypothetical)
        row = recorder.records[-1][1]
        assert revised.version == state.version
        assert row['state']['extra'] == []
        assert len(row['revised_state']['extra']) == 1
        assert row['evidence']['observed'] is False
        assert row['state']['valid'] != row['revised_state']['valid']
        assert state.extra == ()
    finally:
        recorder.close()


def test_actual_consensus_return_identity_and_gradients(tmp_path):
    means = torch.arange(12, dtype=torch.float32).reshape(2, 1, 6).requires_grad_()
    distributions = tuple(dict(mean=means[i], scale2=torch.ones(1, 6)*(i+1), df=5.)
                          for i in range(2))
    weights = torch.tensor([.25, .75], requires_grad=True)
    original = jr.overlap_consensus
    expected = original(distributions, weights)
    gradients = torch.autograd.grad(expected['mean'].sum(), (means, weights))
    recorder = Recorder(tmp_path)
    try:
        install_relation_signals(recorder)
        actual = jr.overlap_consensus(distributions=distributions, weights=weights)
        current = torch.autograd.grad(actual['mean'].sum(), (means, weights))
        for left, right in zip(gradients, current):
            assert torch.equal(left, right)
        assert torch.equal(actual['mean'], expected['mean'])
        assert recorder.records[-1][1]['parent_id'] is None
        with pytest.raises(ValueError):
            jr.overlap_consensus(distributions, -weights)
        jr.overlap_consensus(distributions, weights)
        assert recorder.records[-1][1]['parent_id'] is None
    finally:
        recorder.close()
    assert jr.overlap_consensus is original


def test_actual_score_keeps_target_and_does_not_label_hypotheses_as_observations(tmp_path):
    mean = torch.linspace(-.2, .3, 128).reshape(1, -1).requires_grad_()
    target = mean.detach().flip(1)
    prediction = dict(mean=mean, scale2=torch.ones_like(mean), df=5.)
    expected = cp.log_score(prediction, target)
    grad = torch.autograd.grad(expected.sum(), mean)[0]
    recorder = Recorder(tmp_path)
    try:
        install_relation_signals(recorder)
        actual = cp.log_score(distribution=prediction, target=target)
        assert torch.equal(actual, expected)
        assert torch.equal(torch.autograd.grad(actual.sum(), mean)[0], grad)
        row = recorder.records[-1][1]
        assert row['operation'] == 'log_score'
        assert 'target' in row and 'score' in row
        assert 'observed' not in row
        assert row['caller']['function'].startswith('test_actual_score')
    finally:
        recorder.close()
