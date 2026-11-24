"""Candidate observation boundary only; not installed in the live organism.

S keeps the production delta write. In impedance-normalized coordinates M=P/Z,
the observation corrects the velocity prediction along its actual key. Shapes:
S/P [..., K,V], key [...,K], value [...,V], alpha/beta/gain [...].
This vectorized reference specifies the future fused kernel's numerical contract.
"""
import torch

from scripts.ib_local.gated_deltanet import admit_association_content


def contact_write(s, p, key, value, alpha, beta, impedance, *, gain=None, budget=0.):
    """Return S+, P+, and quantities used for identity/energy audits.

    Preconditions: unit key, 0<=gain<=1, impedance>0. Actual production keys
    are normalized; zero keys also leave P untouched. gain is the existing
    beta at migration, later optionally modulated by local write parameters.
    """
    gain = beta if gain is None else gain
    prior = alpha[..., None, None]*s
    residual = value-torch.einsum('...kv,...k->...v', prior, key)
    innovation = beta[..., None, None]*key[..., :, None]*residual[..., None, :]
    proposed = prior+innovation
    next_s = admit_association_content(proposed, budget) if budget else proposed
    normalized_p = p/impedance[..., None, None]
    predicted = torch.einsum('...kv,...k->...v', normalized_p, key)
    correction = gain[..., None, None]*key[..., :, None]*(residual-predicted)[..., None, :]
    next_p = impedance[..., None, None]*(normalized_p+correction)
    return next_s, next_p, dict(residual=residual, predicted=predicted,
                               correction=correction, admission=next_s-proposed)
