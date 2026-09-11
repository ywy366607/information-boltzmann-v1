"""Birth-only attentive conditional flow with unsaturated translations."""
import torch
from torch import nn

from fine_grain.information_boltzmann.density import InitialDensity, PhaseDensity


class BirthCoupling(nn.Module):
    def __init__(self, dim, hidden, parity):
        super().__init__()
        self.register_buffer('mask', ((torch.arange(dim) + parity) % 2).float())
        self.net = nn.Sequential(nn.Linear(dim + hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2 * dim))
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z, context, inverse=False):
        context = context.expand(*z.shape[:-1], -1)
        raw, shift = self.net(torch.cat((z * self.mask, context), -1)).chunk(2, -1)
        # Algebraic scale bound has polynomially decaying derivative.
        scale = .6 * raw / torch.sqrt(1 + raw.square()) * (1 - self.mask)
        shift = shift * (1 - self.mask)
        if inverse:
            return (z - shift) * (-scale).exp(), -scale.sum(-1)
        return z * scale.exp() + shift, scale.sum(-1)


class BirthDensity(InitialDensity):
    def __init__(self, vocab, phase_dim, hidden, layers=4):
        super().__init__(vocab, phase_dim, hidden, layers)
        self.encoder = nn.TransformerEncoderLayer(hidden, 4, 4 * hidden, dropout=0.,
                                                 activation='gelu', batch_first=True, norm_first=True)
        self.layers = nn.ModuleList(BirthCoupling(2 * phase_dim, hidden, i % 2) for i in range(layers))

    def condition(self, tokens, mask=None):
        if tokens.ndim != 1 or tokens.numel() == 0:
            raise ValueError('A nonempty one-dimensional birth sequence is required')
        h = self.embedding(tokens)
        weights = torch.ones(len(tokens), device=h.device, dtype=h.dtype) if mask is None else mask.to(h)
        if weights.shape != tokens.shape or not bool((weights > 0).any()):
            raise ValueError('Birth sequence needs a valid token mask')
        pos = torch.arange(len(tokens), device=h.device, dtype=h.dtype)[:, None]
        freq = torch.exp(-torch.arange(h.shape[-1], device=h.device, dtype=h.dtype) / h.shape[-1] * 8)
        h = self.encoder(h + torch.sin(pos * freq), src_key_padding_mask=weights <= 0)
        return PhaseDensity(self, (h * weights[:, None]).sum(0) / weights.sum())
