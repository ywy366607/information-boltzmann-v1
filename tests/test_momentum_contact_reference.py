"""Algebra/numerical contracts, not language or memory capability experiments."""
import torch
from torch.nn import functional as F

from scripts.ib_local.momentum_contact_reference import contact_write


def material():
    generator = torch.Generator().manual_seed(1711)
    rand = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.float64)
    s, p = rand(32, 8, 32, 32), rand(32, 8, 32, 32)
    key, value = F.normalize(rand(32, 8, 32), dim=-1), rand(32, 8, 32)
    alpha, beta, z = rand(32, 8).sigmoid(), rand(32, 8).sigmoid(), rand(32, 8).exp()
    return s, p, key, value, alpha, beta, z


def test_delta_state_preserved_and_contact_residual_contracts():
    s, p, k, v, a, b, z = material()
    sn, pn, report = contact_write(s, p, k, v, a, b, z)
    r = report['residual']
    torch.testing.assert_close(sn, a[..., None, None]*s+b[..., None, None]*k[..., :, None]*r[..., None, :])
    after = torch.einsum('...kv,...k->...v', pn/z[..., None, None], k)
    torch.testing.assert_close(r-after, (1-b[..., None])*(r-report['predicted']))
    change = (pn-p)/z[..., None, None]
    projection = k[..., :, None]*torch.einsum('...kv,...k->...v', change, k)[..., None, :]
    torch.testing.assert_close(change, projection)


def test_two_histories_keep_orthogonal_difference_and_energy_identity():
    s, p, k, v, a, b, z = material()
    delta = s-k[..., :, None]*torch.einsum('...kv,...k->...v', s, k)[..., None, :]
    _, pn, report = contact_write(s, p, k, v, a, b, z)
    _, other, _ = contact_write(s, p+z[..., None, None]*delta, k, v, a, b, z)
    torch.testing.assert_close(other-pn, z[..., None, None]*delta)
    m = p/z[..., None, None]
    measured = (pn/z[..., None, None]).square().sum((-2,-1))-m.square().sum((-2,-1))
    u, r = report['predicted'], report['residual']
    expected = 2*b*(u*(r-u)).sum(-1)+b.square()*(r-u).square().sum(-1)
    torch.testing.assert_close(measured, expected)


def test_write_gate_gradient_and_s_independence():
    s,p,k,v,a,b,z = [x[:1,:1] for x in material()]
    raw = torch.tensor(.3, dtype=s.dtype, requires_grad=True)
    def output(x):
        return contact_write(s,p,k,v,a,b,z,gain=x.sigmoid().expand_as(b))[1]
    assert torch.autograd.gradcheck(output, (raw,))
    low = contact_write(s,p,k,v,a,b,z,gain=torch.zeros_like(b))
    high = contact_write(s,p,k,v,a,b,z,gain=torch.ones_like(b))
    torch.testing.assert_close(low[0], high[0], rtol=0, atol=0)
    torch.testing.assert_close(low[1], p)
