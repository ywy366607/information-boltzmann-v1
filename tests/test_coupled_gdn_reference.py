import torch
from torch.autograd import forward_ad

from scripts.ib_local.coupled_gdn_reference import recurrent_reference


def test_forward_tangent_matches_centered_difference():
    generator = torch.Generator().manual_seed(37)
    q, k, v = [torch.randn(1, 3, 2, 4, generator=generator) for _ in range(3)]
    state = torch.randn(1, 2, 4, 4, generator=generator)
    beta = torch.rand(1, 3, 2, generator=generator)
    g = -torch.rand(1, 3, 2, generator=generator)
    direction = torch.randn(state.shape, generator=generator)
    def evaluate(s):
        return recurrent_reference(q,k,v,g,beta,initial_state=s,output_final_state=True,
                                   use_qk_l2norm_in_kernel=True)
    with forward_ad.dual_level():
        out, final = evaluate(forward_ad.make_dual(state,direction))
        _, out_jvp = forward_ad.unpack_dual(out)
        _, state_jvp = forward_ad.unpack_dual(final)
    epsilon = .001
    plus, plus_s = evaluate(state+epsilon*direction)
    minus, minus_s = evaluate(state-epsilon*direction)
    torch.testing.assert_close(out_jvp,(plus-minus)/(2*epsilon),atol=1e-4,rtol=2e-3)
    torch.testing.assert_close(state_jvp,(plus_s-minus_s)/(2*epsilon),atol=1e-4,rtol=2e-3)
