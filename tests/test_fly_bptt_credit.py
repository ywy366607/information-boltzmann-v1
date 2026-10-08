"""The credit inspector must compose the same full BPTT as native autograd."""
import numpy as np
import torch
import torch.nn.functional as F

from test_fly_bptt_learning import make_model, physical
from scripts.ib.inspect_fly_bptt_credit import (
    flatten_state, move_state, one_event, reverse_credit, edge_credit,
)


def test_checkpointed_credit_matches_whole_window_autograd(tmp_path):
    torch.manual_seed(31)
    model = make_model(tmp_path)
    model.requires_grad_(False)
    names = [name for name, _ in model.named_parameters() if name.startswith('log_')]
    for name in names:
        dict(model.named_parameters())[name].requires_grad_(True)
    for name in ('edge_weight_e', 'edge_weight_i'):
        value = getattr(model, name)
        del model._buffers[name]
        model.register_parameter(name, torch.nn.Parameter(value))
    initial = physical(model)
    initial.h.copy_(torch.linspace(.02, .12, 6)[None])
    initial.ring = tuple(torch.ones_like(initial.h)*.3 for _ in range(4))
    start = move_state(initial, 'cpu', leaf=True)
    state, history, latents = start, [move_state(start, 'cpu')], []
    inputs = [0, 1, 2, 3, 4, 5, 6, 7]
    for token in inputs:
        state, _ = one_event(model, state, token, 'cpu')
        history.append(move_state(state, 'cpu'))
        latents.append(model.output_read(state.h[:, model.read_indices]))
    targets = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    loss = F.cross_entropy(model.decoder(model.read_norm(torch.cat(latents))), targets)
    named = dict(model.named_parameters())
    native = torch.autograd.grad(loss, tuple(flatten_state(start).values())+
        tuple(named[name] for name in names)+tuple(named[name] for name in ('edge_weight_e', 'edge_weight_i')),
        allow_unused=True)
    motor = torch.cat([item.h[:, model.read_indices] for item in history[1:]]).requires_grad_()
    direct_loss = F.cross_entropy(model.decoder(model.read_norm(model.output_read(motor))), targets)
    direct = torch.autograd.grad(direct_loss, motor)[0]
    rows, _, currents, params, error = reverse_credit(model, history, inputs, direct, 8,
        'cpu', names, lambda event, adjoint: {key: value.clone() for key, value in adjoint.items()},
        window_objective=True)
    assert error == 0
    for (key, value), expected in zip(flatten_state(start).items(), native[:11]):
        torch.testing.assert_close(rows[-1][key], torch.zeros_like(value) if expected is None else expected,
                                   atol=2e-7, rtol=2e-5)
    for name, expected in zip(names, native[11:11+len(names)]):
        torch.testing.assert_close(params[name], torch.zeros_like(params[name]) if expected is None else expected,
                                   atol=2e-7, rtol=2e-5)
    pulses = np.stack([[slot.detach().numpy().reshape(-1) for slot in item.ring] for item in history[:-1]])
    for k, kind in enumerate(('e', 'i')):
        actual, eligible = edge_credit(getattr(model, f'edge_pre_{kind}').numpy(),
            getattr(model, f'edge_post_{kind}').numpy(), getattr(model, f'splits_{kind}'),
            np.stack([item[k] for item in currents]), pulses)
        torch.testing.assert_close(torch.from_numpy(actual), native[-2+k], atol=2e-7, rtol=2e-5)
        assert np.all(actual[~eligible] == 0)
