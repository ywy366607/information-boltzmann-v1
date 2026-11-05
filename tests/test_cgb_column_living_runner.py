"""Numerical/interface checks only; these streams are not capability tasks."""
import copy
import io

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from information_boltzmann.core.cgb_column_body import CGBColumnLanguageBody
from scripts.run_cgb_column_language_admission import (
    BYTE_OFFSET, CudaGraphValueCredit, _eager_value_credit_phase,
)


STEPS = dict(head_step=0.01, value_step=0.002, route_step=0.003,
             contact_step=0.004, s_contact_step=0.005)


def _body(dtype=torch.float32):
    torch.manual_seed(31)
    return CGBColumnLanguageBody(4, (2, 3), active_columns=2, contact_rank=2).to(dtype=dtype).eval()


def _runner(body=None, **overrides):
    body = _body() if body is None else body
    profile = dict(STEPS, block_size=2, record_s_return=True, collect_diagnostics=True)
    profile.update(overrides)
    return CudaGraphValueCredit(body, body.initial_state(1), **profile)


def _assert_same(left, right, *, rtol=0.0, atol=0.0):
    assert left.mutable_tensor_map().keys() == right.mutable_tensor_map().keys()
    for name, value in left.mutable_tensor_map().items():
        torch.testing.assert_close(value.cpu(), right.mutable_tensor_map()[name].cpu(),
                                   rtol=rtol, atol=atol, msg=name)
    for name, value in left.body.state_dict().items():
        torch.testing.assert_close(value.cpu(), right.body.state_dict()[name].cpu(),
                                   rtol=rtol, atol=atol, msg=name)
    assert left.scored_lifetime == right.scored_lifetime
    assert left.special_lifetime == right.special_lifetime
    assert left.update_calls == right.update_calls
    assert left.advanced_elapsed == right.advanced_elapsed


def _roundtrip(payload):
    stream = io.BytesIO()
    torch.save(payload, stream)
    stream.seek(0)
    return torch.load(stream, weights_only=True)


def test_three_factor_receipt_is_pre_then_post_then_next_arrival(monkeypatch):
    """A boundary score consumes the preceding local contact, never its own write.

    Event ``a`` is assimilated locally and issues a receipt containing its
    pre-participation and H response gate.  On event ``b``, the old state
    predicts ``b``; the resulting innovation is the third factor that may
    change only the receipt issued by ``a``.  This checks causal roles rather
    than a multi-event credit horizon.
    """
    runner = _runner()
    first = torch.tensor([35], dtype=torch.long)
    second = torch.tensor([77], dtype=torch.long)
    runner._step_event(first)
    issued = {
        "index": runner.previous_contact_index.clone(),
        "pre": runner.previous_contact_pre.clone(),
        "gate": runner.previous_contact_gate.clone(),
        "unique": runner.previous_contact_unique.clone(),
        "flux": runner.previous_s_flux.clone(),
        "weight": runner.previous_s_return_weight.clone(),
    }

    # `_step_tensors` exposes the old-state feature by contract, so this is
    # exactly the prediction which must be scored by the arrival of `second`.
    prepared = runner.body._step_tensors(
        runner.activity, runner.natural, runner.precision, runner.causal_token,
        runner.participation, second, runner.elapsed_zero, runner.external_source,
    )
    expected_logits = runner.body.readout(prepared[5]).detach().clone()
    original = runner.body._s_contact_return_from_tensors
    original_h = runner.body._update_reciprocal_h_from_tensors
    seen = {}

    def observe(logits, read_route, index, gate, unique, flux, weight, target, has_previous):
        seen.update(logits=logits.detach().clone(), index=index.detach().clone(),
                    gate=gate.detach().clone(), unique=unique.detach().clone(),
                    flux=flux.detach().clone(), weight=weight.detach().clone(),
                    target=target.detach().clone(), has_previous=has_previous.detach().clone())
        return original(logits, read_route, index, gate, unique, flux, weight, target, has_previous)

    def observe_h(logits, read_route, index, pre, gate, unique, target, step_size, has_previous):
        seen.update(h_logits=logits.detach().clone(), h_index=index.detach().clone(),
                    h_pre=pre.detach().clone(), h_gate=gate.detach().clone(),
                    h_unique=unique.detach().clone(), h_target=target.detach().clone(),
                    h_has_previous=has_previous.detach().clone())
        return original_h(logits, read_route, index, pre, gate, unique, target, step_size, has_previous)

    monkeypatch.setattr(runner.body, "_s_contact_return_from_tensors", observe)
    monkeypatch.setattr(runner.body, "_update_reciprocal_h_from_tensors", observe_h)
    runner._step_event(second)

    torch.testing.assert_close(seen["logits"], expected_logits, rtol=0.0, atol=0.0)
    torch.testing.assert_close(seen["target"], second, rtol=0.0, atol=0.0)
    assert bool(seen["has_previous"].all())
    for name in ("index", "gate", "unique", "flux", "weight"):
        torch.testing.assert_close(seen[name], issued[name], rtol=0.0, atol=0.0)
    # The exact H-contact three-factor update additionally consumes the
    # issuing pre-vector.  Thus its sole write direction is local pre × local
    # post-gate × the following boundary innovation.
    for name, issued_name in (("h_logits", None), ("h_index", "index"),
                              ("h_pre", "pre"), ("h_gate", "gate"),
                              ("h_unique", "unique")):
        torch.testing.assert_close(seen[name], expected_logits if issued_name is None else issued[issued_name],
                                   rtol=0.0, atol=0.0)
    torch.testing.assert_close(seen["h_target"], second, rtol=0.0, atol=0.0)
    assert bool(seen["h_has_previous"].all())
    assert not torch.equal(runner.previous_contact_pre, issued["pre"])


def test_original_constructor_defaults_roundtrip_without_optional_buffers():
    body = _body()
    runner = CudaGraphValueCredit(body, body.initial_state(1), 0.0, 0.0, 0.0, 0.0, 2)
    assert not runner.record_s_return and not runner.collect_diagnostics
    assert not hasattr(runner, "previous_s_flux")
    assert not hasattr(runner, "previous_s_return_weight")
    assert runner.diagnostic_totals is None
    result = runner.phase(np.asarray([35, 77, 0], dtype=np.int64), 100)
    assert "diagnostic_totals" not in result and result["progress"] == []
    assert all(value == 0 for value in runner.update_calls.values())
    payload = _roundtrip(runner.checkpoint_state())
    resumed = CudaGraphValueCredit.from_checkpoint(payload)
    _assert_same(runner, resumed)
    assert resumed.checkpoint_state()["profile"] == payload["profile"]
    for engine in (runner, resumed):
        engine.phase(np.asarray([18, 111], dtype=np.int64), 100)
    _assert_same(runner, resumed)


@pytest.mark.parametrize("s_step", [0.0, STEPS["s_contact_step"]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_event_order_matches_legacy_eager_with_special_tokens(s_step, dtype, monkeypatch):
    def no_cuda(*args, **kwargs):
        raise AssertionError("CPU runner must not invoke torch.cuda")

    for name in ("synchronize", "CUDAGraph", "graph", "is_available"):
        monkeypatch.setattr(torch.cuda, name, no_cuda)
    legacy = _body(dtype)
    runner = _runner(copy.deepcopy(legacy), s_contact_step=s_step)
    events = np.asarray([35, 77, 0, 18, 111, 259, 1, 91], dtype=np.int64)
    expected = _eager_value_credit_phase(
        legacy, legacy.initial_state(1), events,
        STEPS["head_step"], STEPS["value_step"], STEPS["route_step"],
        STEPS["contact_step"], 3, s_contact_step=s_step,
    )
    actual = runner.phase(events, 3)
    # Every lexical arrival is scored from the prior state and then written;
    # special markers advance time but never splice language runs together.
    assert actual["scored_bytes"] == expected["scored_bytes"] == 6
    assert actual["special_tokens_elapsed_without_lexical_write"] == 2
    assert actual["mean_nll_nats"] == pytest.approx(expected["mean_nll_nats"], abs=1e-6)
    assert [item["scored_bytes"] for item in actual["progress"]] == [3, 6]
    for name in expected["_state"].__dataclass_fields__:
        torch.testing.assert_close(getattr(runner.state(), name), getattr(expected["_state"], name))
    for name, value in legacy.state_dict().items():
        torch.testing.assert_close(value, runner.body.state_dict()[name])
    previous = expected["_previous"]
    receipt = legacy._pack_local_receipt(previous.contact_pre, previous.contact_response_gate,
                                         previous.route)
    for name, value in zip(("previous_contact_index", "previous_contact_pre",
                            "previous_contact_gate", "previous_contact_unique"), receipt):
        torch.testing.assert_close(getattr(runner, name), value)
    if s_step > 0:
        torch.testing.assert_close(runner.previous_s_flux, previous.edge_flux)
        torch.testing.assert_close(runner.previous_s_return_weight, previous.contact_return_weight)
    assert runner.scored_lifetime == 6 and runner.special_lifetime == 2
    assert runner.advanced_elapsed == 2.0
    assert runner.update_calls == dict(head=6, value=6, route=6, contact=6,
                                       s_contact=6 if s_step > 0 else 0)
    assert runner.graphs == {}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_weights_only_checkpoint_continues_all_buffers_without_phantom_events(dtype):
    reference = _runner(_body(dtype))
    split = _runner(copy.deepcopy(reference.body))
    prefix = np.asarray([31, 73, 19], dtype=np.int64)
    suffix = np.asarray([255, 8, 45], dtype=np.int64)
    for runner in (reference, split):
        runner._replay(prefix)
        runner.advance_empty_time(0.375)
    payload = _roundtrip(split.checkpoint_state())
    before = {name: value.clone() for name, value in payload["buffers"].items()}
    resumed = CudaGraphValueCredit.from_checkpoint(payload)
    _assert_same(split, resumed)
    assert payload["model_config"]["dtype"] == str(dtype).removeprefix("torch.")
    assert payload["profile"]["record_s_return"] is True
    assert all(not name.startswith("body.") for name in payload["buffers"])
    assert payload["lifetime"]["scored_lifetime"] == 3
    for runner in (reference, resumed):
        runner.advance_empty_time(1.25)
        runner._replay(suffix)
    _assert_same(reference, resumed)
    assert resumed.advanced_elapsed == 1.625
    for name in before:
        torch.testing.assert_close(payload["buffers"][name], before[name], rtol=0, atol=0)
    assert resumed.body.reciprocal_contact.grad is None


def test_checkpoint_forks_share_no_tensors_and_override_freezes_only_s_carrier_transfer():
    original = _runner()
    original._replay(np.asarray([31, 73, 19], dtype=np.int64))
    payload = original.checkpoint_state()
    learning = CudaGraphValueCredit.from_checkpoint(payload)
    fixed_s = CudaGraphValueCredit.from_checkpoint(payload, s_contact_step=0.0)
    _assert_same(learning, fixed_s)
    assert learning.record_s_return and fixed_s.record_s_return
    for name, tensor in learning.mutable_tensor_map().items():
        assert tensor.data_ptr() != fixed_s.mutable_tensor_map()[name].data_ptr()
    for name, tensor in learning.body.state_dict().items():
        assert tensor.data_ptr() != fixed_s.body.state_dict()[name].data_ptr()
    prior_s = fixed_s.participation.clone()
    prior_return = fixed_s.previous_s_return_weight.clone()
    # The delayed receipt is installed only in the existing S carrier after
    # the issuing event; no common dynamics parameter is rewritten.
    event = np.asarray([43], dtype=np.int64)
    learning._replay(event)
    fixed_s._replay(event)
    for (name, parameter), (other_name, other_parameter) in zip(
        learning.body.named_parameters(), fixed_s.body.named_parameters(),
    ):
        assert name == other_name
        torch.testing.assert_close(parameter, other_parameter, rtol=0, atol=0)
    for name in ("activity", "natural", "precision", "causal_token"):
        torch.testing.assert_close(getattr(learning, name), getattr(fixed_s, name), rtol=0, atol=0)
    assert not torch.equal(learning.participation, fixed_s.participation)
    assert not torch.equal(fixed_s.participation, prior_s)
    assert not torch.equal(fixed_s.previous_s_return_weight, prior_return)
    assert fixed_s.participation.sum() > 0
    assert payload["profile"]["s_contact_step"] == STEPS["s_contact_step"]
    assert fixed_s.update_calls["s_contact"] == original.update_calls["s_contact"]


@pytest.mark.parametrize("elapsed", [0.0, 0.125, 2.5])
def test_empty_elapsed_is_exact_prior_and_preserves_h_s_and_receipts(elapsed):
    runner = _runner()
    runner._replay(np.asarray([31, 73], dtype=np.int64))
    before = {name: value.detach().clone() for name, value in runner.mutable_tensor_map().items()}
    mean = runner.natural / runner.precision
    q = F.softplus(runner.body.process_variance_raw).detach()
    expected_precision = runner.precision / (1.0 + elapsed * q * runner.precision)
    runner.advance_empty_time(elapsed)
    torch.testing.assert_close(runner.precision, expected_precision, rtol=0, atol=0)
    if elapsed:
        torch.testing.assert_close(runner.natural, expected_precision * mean, rtol=0, atol=0)
    for name, value in runner.mutable_tensor_map().items():
        if elapsed == 0 or name not in ("natural", "precision"):
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert runner.advanced_elapsed == elapsed
    assert runner.scored_lifetime == 2


@pytest.mark.parametrize("elapsed", [-1.0, float("inf"), float("-inf"), float("nan")])
def test_empty_elapsed_rejects_negative_and_nonfinite_values(elapsed):
    runner = _runner(collect_diagnostics=False)
    before = runner.checkpoint_state()
    with pytest.raises(ValueError, match="finite and nonnegative"):
        runner.advance_empty_time(elapsed)
    for name, value in before["buffers"].items():
        torch.testing.assert_close(getattr(runner, name), value, rtol=0, atol=0)
    assert runner.advanced_elapsed == 0.0


def test_diagnostics_are_observational_count_forward_once_and_reset_per_phase(monkeypatch):
    body = _body()
    plain = _runner(copy.deepcopy(body), collect_diagnostics=False)
    observed = _runner(copy.deepcopy(body))
    no_s_update = _runner(copy.deepcopy(body), s_contact_step=0.0)
    # First event has no return receipt: its S direction is zero with or without
    # the VJP. Thus the next forward has identical weights for this comparison.
    first = np.asarray([31 + BYTE_OFFSET], dtype=np.int64)
    observed.phase(first, 100)
    no_s_update.phase(first, 100)
    torch.testing.assert_close(observed.diagnostic_totals[:4], no_s_update.diagnostic_totals[:4],
                               rtol=0, atol=0)
    assert observed.diagnostic_totals[4:].count_nonzero() == 0
    plain.phase(first, 100)
    observed_prefix = observed.checkpoint_state()
    forward_calls = []
    actual_step = observed.body._assimilate_tensors

    def count_actual_forward(*args, **kwargs):
        forward_calls.append(kwargs.get("diagnostic_totals"))
        return actual_step(*args, **kwargs)

    monkeypatch.setattr(observed.body, "_assimilate_tensors", count_actual_forward)
    rest = np.asarray([73 + BYTE_OFFSET, 0, 19 + BYTE_OFFSET, 255 + BYTE_OFFSET], dtype=np.int64)
    plain_result = plain.phase(rest, 100)
    result = observed.phase(rest, 100)
    assert "diagnostic_totals" not in plain_result
    assert result["progress"] == [] and not hasattr(observed, "progress")
    assert len(forward_calls) == 3
    assert all(value is observed.diagnostic_totals for value in forward_calls)
    assert observed.diagnostic_totals[4] > 0 and observed.diagnostic_totals[5] > 0
    for name, value in plain.mutable_tensor_map().items():
        torch.testing.assert_close(value, observed.mutable_tensor_map()[name], rtol=0, atol=0)
    assert result["mean_nll_nats"] == plain_result["mean_nll_nats"]
    # The three calls score and then assimilate the three lexical arrivals;
    # the marker advances time separately.  Thus S credit does not secretly
    # unroll an extra forward pass.
    empty = observed.phase(np.asarray([], dtype=np.int64), 100)
    assert empty["diagnostic_totals"] == [0.0] * 6
    assert observed.scored_lifetime == 4 and observed.special_lifetime == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("s_step", [0.0, STEPS["s_contact_step"]])
def test_cuda_capture_resume_matches_cpu_without_capture_events(s_step):
    body = _body()
    cpu = _runner(copy.deepcopy(body), s_contact_step=s_step)
    cuda_body = copy.deepcopy(body).cuda()
    initial = cuda_body.initial_state(1)
    cuda = _runner(cuda_body, s_contact_step=s_step)
    before = cpu.checkpoint_state()
    for name, value in before["buffers"].items():
        torch.testing.assert_close(cuda.mutable_tensor_map()[name].cpu(), value, rtol=0, atol=0)
    for name, value in body.state_dict().items():
        torch.testing.assert_close(cuda.body.state_dict()[name].cpu(), value, rtol=0, atol=0)
    assert cuda.scored_lifetime == cuda.special_lifetime == 0
    assert all(value == 0 for value in cuda.update_calls.values())
    assert cuda.advanced_elapsed == 0.0
    assert cuda.body.persistent_state_bytes(initial) == body.persistent_state_bytes(cpu.state())
    events = np.asarray([35, 77, 0, 18, 111, 259], dtype=np.int64)
    cpu_result = cpu.phase(events, 100)
    cuda_result = cuda.phase(events, 100)
    assert cuda_result["mean_nll_nats"] == pytest.approx(cpu_result["mean_nll_nats"], abs=2e-5)
    _assert_same(cpu, cuda, rtol=3e-5, atol=3e-6)
    saved = _roundtrip(cuda.checkpoint_state())
    resumed = CudaGraphValueCredit.from_checkpoint(saved, device="cuda")
    _assert_same(cuda, resumed)
    suffix = np.asarray([45, 3], dtype=np.int64)
    for runner in (cpu, cuda, resumed):
        runner.advance_empty_time(0.625)
        runner._replay(suffix)
    _assert_same(cuda, resumed)
    _assert_same(cpu, cuda, rtol=5e-5, atol=4e-6)
