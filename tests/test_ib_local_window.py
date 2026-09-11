import torch
import copy
import numpy as np

from scripts.ib_bpe_window import clock_inputs
from scripts.ib_local.window import LocalWindow
from scripts.ib_local.sampling import sample_window


def test_birth_inverse_and_density_are_finite():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    model = LocalWindow(vocab=32, hidden=16, particles=4, steps=1).double()
    density = model.core.initial.condition(torch.tensor([1, 2, 3]))
    z = torch.randn(5, 8, dtype=torch.float64)
    y, ld = model.core.initial.transform(z, density.context)
    zz, ild = model.core.initial.transform(y, density.context, inverse=True)
    torch.testing.assert_close(z, zz, atol=1e-10, rtol=1e-8)
    torch.testing.assert_close(ld, -ild, atol=1e-10, rtol=1e-8)
    assert torch.isfinite(density.log_prob(*y.chunk(2, -1))).all()


def test_window_birth_gradients_and_causal_prefix():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    model = LocalWindow(vocab=32, hidden=16, particles=4, steps=1)
    state = model.core.initialize(torch.tensor([1]))
    ids, targets = torch.tensor([1, 2]), torch.tensor([2, 3])
    clocks = clock_inputs(0, 2, 1, 'cpu')
    noise = torch.randn(8, 4, 4)
    first = model(state.x, state.v, ids[:1], targets[:1], clocks[:1], noise[:4], [[]])
    second = model(first[2], first[3], ids[1:], targets[1:], clocks[1:], noise[4:], [[]])
    full = model(state.x, state.v, ids, targets, clocks, noise, [[], []])
    torch.testing.assert_close(full[1], (first[1] + second[1]) / 2)
    torch.testing.assert_close(full[2], second[2])
    full[0].backward()
    grad = model.core.initial.layers[-1].net[-1].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_update_boundary_resume_preserves_state_rng_and_optimizer():
    torch.set_num_threads(1)
    torch.manual_seed(29)
    model = LocalWindow(vocab=32, hidden=16, particles=8, steps=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(11)
    noise_rng = torch.Generator().manual_seed(11)
    state = model.core.initialize(torch.tensor([1]), noise_rng)

    def update(model, optimizer, rng, noise_rng, x, v, cursor):
        tables, _ = sample_window(rng, 2, 1, 8, 'cpu')
        noise = torch.randn(8, 8, 4, generator=noise_rng)
        result = model(x, v, torch.tensor([1, 2]), torch.tensor([2, 3]),
                       clock_inputs(cursor, 2, 1, 'cpu'), noise, tables)
        optimizer.zero_grad(set_to_none=True)
        result[0].backward()
        optimizer.step()
        return result[1].detach(), result[2].detach(), result[3].detach()

    _, x, v = update(model, optimizer, rng, noise_rng, state.x, state.v, 0)
    saved = copy.deepcopy((model.state_dict(), optimizer.state_dict(), rng.bit_generator.state,
                           noise_rng.get_state(), x, v))
    expected = update(model, optimizer, rng, noise_rng, x, v, 2)
    resumed = LocalWindow(vocab=32, hidden=16, particles=8, steps=1)
    resumed.load_state_dict(saved[0])
    opt2 = torch.optim.AdamW(resumed.parameters(), lr=3e-4)
    opt2.load_state_dict(saved[1])
    rng2 = np.random.default_rng()
    rng2.bit_generator.state = saved[2]
    noise2 = torch.Generator()
    noise2.set_state(saved[3])
    actual = update(resumed, opt2, rng2, noise2, saved[4], saved[5], 2)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for a, b in zip(model.parameters(), resumed.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_recompute_toggle_preserves_parameter_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(41)
    a = LocalWindow(vocab=32, hidden=16, particles=8, steps=1)
    b = copy.deepcopy(a)
    b.recompute = False
    x, v = torch.randn(8, 4) * .1, torch.randn(8, 4)
    tables, _ = sample_window(np.random.default_rng(13), 2, 1, 8, 'cpu')
    args = (x, v, torch.tensor([1, 2]), torch.tensor([2, 3]),
            clock_inputs(0, 2, 1, 'cpu'), torch.randn(8, 8, 4), tables)
    aa, bb = a(*args), b(*args)
    torch.testing.assert_close(aa[0], bb[0], atol=0, rtol=0)
    aa[0].backward()
    bb[0].backward()
    for p, q in zip(a.parameters(), b.parameters()):
        if p.grad is None:
            assert q.grad is None
        else:
            torch.testing.assert_close(p.grad, q.grad, atol=1e-7, rtol=1e-6)
