"""Packed readback preserves the exact original conditional RNG sequence."""
import pytest
import torch
from scripts.ib_local.benchmark_runtime_components import draw_byte, draw_handover


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('stop_value', [-10., 10.])
def test_byte_and_handover_pack_preserve_draws_and_rng(device, stop_value):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA RNG/readback comparison')
    with torch.random.fork_rng(devices=[0] if device == 'cuda' else []):
        torch.manual_seed(1123)
        logits = torch.linspace(-3, 3, 256, device=device)
        stop = torch.tensor(stop_value, device=device)
        get = torch.cuda.get_rng_state if device == 'cuda' else torch.get_rng_state
        put = torch.cuda.set_rng_state if device == 'cuda' else torch.set_rng_state
        before = get()
        expected_ended = bool(torch.rand((), device=device) < stop.sigmoid())
        expected_logit = float(stop)
        expected_byte = expected_value = None
        if not expected_ended:
            logp = logits.log_softmax(-1)
            expected_byte = int(torch.multinomial(logp.exp(), 1))
            expected_value = float(logp[expected_byte])
        expected_rng = get()
        put(before)
        value, ended = draw_handover(stop)
        assert value == expected_logit and ended == expected_ended
        if not ended:
            byte, value = draw_byte(logits)
            assert byte == expected_byte and value == expected_value
        assert torch.equal(get(), expected_rng)
