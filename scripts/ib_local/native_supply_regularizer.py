"""Opt-in SADL supply at the native body's pre-encoder feature port.

The existing SIGReg is the demand regularizer. H is current raw body output
plus valid stored raw history; Z is its shared linear encoding, stopped for
the supply term. Historical H remains detached, current H keeps its original
body graph. No historical autograd graph or extra body forward is retained.
Reference: arxiv:2611.21596v1 equations 2--10 and Appendix B.
"""
from contextlib import contextmanager
import math

import torch


def excitation(u, *, gamma=9., epsilon=1e-8):
    n, d = u.shape
    if n < 2:
        raise ValueError('supply requires two aligned samples')
    centered = u-u.mean(0, keepdim=True)
    covariance = centered.T@centered/n
    energy = u.square().mean().clamp_min(epsilon)
    m = min(d, n-1)
    eye = torch.eye(d, dtype=u.dtype, device=u.device)
    return torch.linalg.slogdet(eye+gamma*covariance/energy)[1]/(
        m*math.log1p(gamma*d/m))


def transmission(h, z, *, gamma=9., ridge=1e-3, epsilon=1e-8):
    n = len(h)
    if len(z) != n or n < 2:
        raise ValueError('H and Z must contain the same aligned samples')
    h, z = h-h.mean(0, keepdim=True), z-z.mean(0, keepdim=True)
    ch, cz, cross = h.T@h/n, z.T@z/n, h.T@z/n
    # Keeping the H-side ridge scale live preserves the scale derivative.
    sh = ch.diag().mean().clamp_min(epsilon)
    sz = cz.diag().mean().clamp_min(epsilon)
    lh = torch.linalg.cholesky(ch+ridge*sh*torch.eye(ch.shape[0], device=h.device, dtype=h.dtype))
    lz = torch.linalg.cholesky(cz+ridge*sz*torch.eye(cz.shape[0], device=z.device, dtype=z.dtype))
    whitened = torch.linalg.solve_triangular(lh, cross, upper=False)
    whitened = torch.linalg.solve_triangular(lz, whitened.T, upper=False).T
    gram = whitened.T@whitened if z.shape[1] <= h.shape[1] else whitened@whitened.T
    m = min(h.shape[1], z.shape[1], n-1)
    eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
    return torch.linalg.slogdet(eye+gamma*gram)[1]/(m*math.log1p(gamma))


def sigreg_zero(count, like):
    """Exact all-zero reference for the current live_sigreg Epps--Pulley rule."""
    t = torch.linspace(0, 3, 17, device=like.device, dtype=like.dtype)
    phi = torch.exp(-t.square()/2)
    quad = torch.ones_like(t)*(6/16)
    quad[0] /= 2
    quad[-1] /= 2
    return ((1-phi).square()@(quad*phi))*count


def supply(h, z, demand_loss, zero_loss, *, gamma=9., ridge=1e-3, epsilon=1e-8):
    """SADL with detached endpoint/demand and live H energy/ridge scales."""
    with torch.autocast(device_type=h.device.type, enabled=False):
        # Float64 is kept for derivative contract checks; training uses FP32.
        dtype = torch.float64 if h.dtype == torch.float64 else torch.float32
        h, z = h.to(dtype), z.detach().to(dtype)
        if h.ndim != 2 or z.ndim != 2 or len(h) != len(z):
            raise ValueError('two aligned sample matrices required')
        q_reg = (1-demand_loss.detach().to(dtype)/zero_loss.detach().to(dtype)).clamp(0, 1)
        q_exc = excitation(z, gamma=gamma, epsilon=epsilon).clamp(0, 1)
        q = torch.exp(.5*(torch.log(q_reg+epsilon)+torch.log(q_exc+epsilon)))
        k_exc = excitation(h, gamma=gamma, epsilon=epsilon)
        k_tr = transmission(h, z, gamma=gamma, ridge=ridge, epsilon=epsilon)
        k = torch.minimum(k_exc, k_tr)
        deficit = (torch.log(q+epsilon)-torch.log(k+epsilon)).clamp_min(0)
        return deficit.square(), dict(q=q, k=k, q_reg=q_reg, q_exc=q_exc,
            k_exc=k_exc, k_tr=k_tr, samples=len(h))


def body_supply(model, current, demand_loss, **kwargs):
    history = model.memory.history[model.memory.history_valid.bool()].detach()
    h = torch.cat((current, history), 0)
    z = model.encoder(h).detach()
    return supply(h, z, demand_loss, sigreg_zero(len(h), z), **kwargs)


@contextmanager
def objective_supply(*, coefficient=.01, gamma=9., ridge=1e-3, epsilon=1e-8):
    """Private-run wrapper; default production modules remain unpatched.

    The caller records this explicit runtime choice in its protocol. Scope
    serially around one original engine/worker; restoring the same candidate
    requires restoring this wrapper as well.
    """
    from scripts.ib_local.language_navigation_kanli_v4l import LiveSigregBody
    original = LiveSigregBody.objective
    receipts = []

    def objective(model, *args, **kwargs):
        loss, state, h, diag = original(model, *args, **kwargs)
        if coefficient == 0:
            return loss, state, h, diag
        extra, info = body_supply(model, h, diag['sigreg'], gamma=gamma,
                                  ridge=ridge, epsilon=epsilon)
        receipt = {key:float(value.detach()) if torch.is_tensor(value) else value
                   for key, value in info.items()}
        receipt.update(loss=float(extra.detach()), coefficient=coefficient,
                       current_graph_live=h.requires_grad,
                       grad_enabled=torch.is_grad_enabled())
        if (not any('first_current_supply_gradient_norm' in r for r in receipts)
                and torch.is_grad_enabled() and h.requires_grad):
            local, = torch.autograd.grad(extra, h, retain_graph=True)
            receipt['first_current_supply_gradient_norm'] = float(local.detach().norm())
        receipts.append(receipt)
        return loss+coefficient*extra, state, h, dict(diag, supply_regularizer=receipt)

    LiveSigregBody.objective = objective
    try:
        yield receipts
    finally:
        LiveSigregBody.objective = original
