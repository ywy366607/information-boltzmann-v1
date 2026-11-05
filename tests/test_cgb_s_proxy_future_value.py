"""Deterministic interface checks for the real-continuation S direction audit."""

import numpy as np
import torch

from information_boltzmann.core.cgb_column_body import CGBColumnLanguageBody
from scripts.audit_cgb_column_s_proxy_future_value import (
    apply_relative_s_delta,
    proxy_direction_for_byte,
)
from scripts.run_cgb_column_language_admission import CudaGraphValueCredit


def _engine():
    torch.manual_seed(41)
    body = CGBColumnLanguageBody(4, (2, 3), active_columns=2, contact_rank=2)
    return CudaGraphValueCredit(body, body.initial_state(1), 0.01, 0.001, 0.001, 0.001,
                                block_size=1, s_contact_step=0.0, record_s_return=True)


def test_proxy_direction_is_read_only_and_signed_s_perturbations_are_matched():
    engine = _engine()
    engine.phase(np.asarray([17, 83], dtype=np.int64), report_every=3)
    before = engine.checkpoint_state()
    direction = proxy_direction_for_byte(engine, 61)
    after = engine.checkpoint_state()
    assert direction.shape == engine.body.reciprocal_contact[engine.body.d_model + 2:].shape
    assert torch.isfinite(direction).all()
    # Extracting an audit direction is not a hidden learning transition.
    for name, tensor in before["buffers"].items():
        torch.testing.assert_close(tensor, after["buffers"][name], rtol=0.0, atol=0.0)
    for name, tensor in before["body_state_dict"].items():
        torch.testing.assert_close(tensor, after["body_state_dict"][name], rtol=0.0, atol=0.0)

    before_parameters = {name: value.detach().clone() for name, value in engine.body.named_parameters()}
    state = engine.checkpoint_state()
    plus = CudaGraphValueCredit.from_checkpoint(state, device="cpu", s_contact_step=0.0)
    minus = CudaGraphValueCredit.from_checkpoint(state, device="cpu", s_contact_step=0.0)
    before_s = engine.body.reciprocal_contact[engine.body.d_model + 2:].detach().clone()
    plus_info = apply_relative_s_delta(plus, direction, 1e-4)
    minus_info = apply_relative_s_delta(minus, direction, -1e-4)
    plus_delta = plus.body.reciprocal_contact[plus.body.d_model + 2:] - before_s
    minus_delta = minus.body.reciprocal_contact[minus.body.d_model + 2:] - before_s
    torch.testing.assert_close(plus_delta, -minus_delta, rtol=1e-6, atol=1e-8)
    assert plus_info["delta_l2"] == minus_info["delta_l2"]
    for name, parameter in plus.body.named_parameters():
        if name == "reciprocal_contact":
            torch.testing.assert_close(parameter[:plus.body.d_model + 2],
                                       before_parameters[name][:plus.body.d_model + 2],
                                       rtol=0.0, atol=0.0)
        else:
            torch.testing.assert_close(parameter, before_parameters[name], rtol=0.0, atol=0.0)
