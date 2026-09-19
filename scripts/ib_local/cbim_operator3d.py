"""Resolution-independent 3D truncation of a continuous CBIM operator.

The state is a signed feature field h(x,t) on a periodic internal memory domain.
Grid sizes select a Fourier/Galerkin truncation; every learned parameter is
independent of that selection, so one state_dict can be evaluated on another
resolution. This is a feature kinetic model, not a non-negative gas density.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def _grid(shape):
    axes = [torch.arange(n, dtype=torch.float32) / n for n in shape]
    return torch.stack(torch.meshgrid(*axes, indexing='ij'), -1)


def _position_features(x):
    phase = 2 * math.pi * x
    return torch.cat((phase.sin(), phase.cos()), -1)


class ContinuousCayleyTransport3D(nn.Module):
    """Learn a continuous odd dispersion symbol and sample it on any 3D grid."""

    def __init__(self, shape=(4, 4, 4), d=128, hidden=64, dt=1.0):
        super().__init__()
        self.shape = tuple(shape)
        self.d = d
        self.dt = dt
        self.symbol = nn.Sequential(
            nn.Linear(12, hidden), nn.SiLU(), nn.Linear(hidden, d))
        # Near-identity transport at initialization without killing gradients.
        nn.init.normal_(self.symbol[-1].weight, std=1e-3)
        nn.init.zeros_(self.symbol[-1].bias)
        axes = [torch.fft.fftfreq(n) * 2 * math.pi for n in self.shape]
        wave = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1)
        positive = torch.cat((wave.sin(), wave.cos(),
                              (2 * wave).sin(), (2 * wave).cos()), -1)
        negative_wave = -wave
        negative = torch.cat((negative_wave.sin(), negative_wave.cos(),
                              (2 * negative_wave).sin(),
                              (2 * negative_wave).cos()), -1)
        self.register_buffer('symbol_input', positive, persistent=False)
        self.register_buffer('negative_symbol_input', negative, persistent=False)

    def dispersion(self):
        # Explicit antisymmetrization guarantees omega(-k) = -omega(k).
        return .5 * (self.symbol(self.symbol_input) -
                     self.symbol(self.negative_symbol_input))

    def multiplier(self):
        omega = self.dispersion()
        mu = .5 * self.dt * omega
        denom = 1 + mu.square()
        multiplier = torch.complex((1 - mu.square()) / denom,
                                   -2 * mu / denom)[None]
        return multiplier, omega

    def apply_multiplier(self, h, multiplier):
        transformed = torch.fft.fftn(h, dim=(1, 2, 3), norm='ortho')
        return torch.fft.ifftn(transformed * multiplier,
                              dim=(1, 2, 3), norm='ortho').real

    def forward(self, h):
        multiplier, omega = self.multiplier()
        output = self.apply_multiplier(h, multiplier)
        return output, {
            'dispersion_abs_mean': omega.detach().abs().mean(),
            'dispersion_abs_max': omega.detach().abs().max(),
        }


class LocalSourceOutflow3D(nn.Module):
    """State-aware localized source and outflow with a resolution-uniform bound."""

    def __init__(self, vocab_size=50257, shape=(4, 4, 4), d=128,
                 cell_radius=1.25, max_write_rate=.25,
                 max_outflow_rate=.05):
        super().__init__()
        self.shape = tuple(shape)
        self.d = d
        self.cell_radius = float(cell_radius)
        self.max_write_rate = float(max_write_rate)
        self.max_outflow_rate = float(max_outflow_rate)
        self.embedding = nn.Embedding(vocab_size, d)
        self.content = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.address = nn.Linear(d, 3)
        self.width = nn.Linear(d, 3)
        self.write_rate = nn.Linear(d, 1)
        self.context = nn.Sequential(
            nn.Linear(3 * d, d), nn.SiLU(), nn.Linear(d, d), nn.SiLU())
        self.content_correction = nn.Linear(d, d)
        self.context_write_rate = nn.Linear(d, 1)
        self.outflow_rate = nn.Linear(d, 1)
        for layer in (self.content_correction, self.context_write_rate,
                      self.outflow_rate):
            nn.init.normal_(layer.weight, std=1e-3)
            nn.init.zeros_(layer.bias)
        nn.init.constant_(self.write_rate.bias, -2.)
        nn.init.constant_(self.outflow_rate.bias, -4.)
        self.register_buffer('coordinates', _grid(self.shape)[None, ..., None, :],
                             persistent=False)

    def forward(self, h, token_ids):
        emb = self.embedding(token_ids)
        batch = h.shape[0]
        center = torch.sigmoid(self.address(emb)).view(batch, 1, 1, 1, 1, 3)
        # Width is expressed in physical torus units and therefore does not
        # change when the Galerkin grid is refined.
        sigma = (.04 + .21 * torch.sigmoid(self.width(emb))).view(
            batch, 1, 1, 1, 1, 3)
        delta = (self.coordinates - center).abs()
        distance = torch.minimum(delta, 1 - delta)
        envelope = torch.exp(-.5 * (distance / sigma).square().sum(-1))

        neighborhood = h
        for axis in (1, 2, 3):
            neighborhood = neighborhood + torch.roll(h, 1, axis) + torch.roll(h, -1, axis)
        neighborhood = neighborhood / 7
        expanded = emb[:, None, None, None].expand(-1, *self.shape, -1)
        context = self.context(torch.cat((expanded, h, neighborhood), -1))
        outflow = self.max_outflow_rate * torch.sigmoid(self.outflow_rate(context))
        retained = (1 - outflow) * h
        rate = envelope * self.max_write_rate * torch.sigmoid(
            self.write_rate(emb)[:, None, None, None] +
            self.context_write_rate(context))
        proposal = (1 - rate) * retained + rate * (
            self.content(emb)[:, None, None, None] +
            self.content_correction(context))
        norm = proposal.norm(dim=-1, keepdim=True)
        scale = torch.clamp(self.cell_radius / norm.clamp_min(1e-8), max=1.)
        result = proposal * scale
        return result, {
            'write_rate_mean': rate.detach().mean(),
            'outflow_rate_mean': outflow.detach().mean(),
            'local_projection_rate': (scale.detach() < 1).float().mean(),
        }


class ConservativeScattering3D(nn.Module):
    """Shared nonlinear pair scattering on all three axes."""

    def __init__(self, shape=(4, 4, 4), d=128, sweeps=1):
        super().__init__()
        if d % 2 or any(n % 2 for n in shape):
            raise ValueError('Even channel and grid dimensions are required')
        self.shape = tuple(shape)
        self.d = d
        self.sweeps = sweeps
        self.register_buffer('layers_applied', torch.tensor(sweeps * 6, dtype=torch.int64), persistent=False)
        self.angle = nn.Sequential(
            nn.Linear(2 * d + 3, d), nn.SiLU(), nn.Linear(d, d // 2))
        nn.init.normal_(self.angle[-1].weight, std=1e-3)
        nn.init.zeros_(self.angle[-1].bias)

    def _pairs(self, h, axis, parity, channel_shift):
        moved = h.movedim(axis + 1, 1)
        moved = torch.roll(moved, -channel_shift, -1)
        if parity == 0:
            p, q = moved[:, 0::2], moved[:, 1::2]
        else:
            p = moved[:, 1::2]
            q = torch.roll(moved[:, 0::2], -1, 1)
        direction = torch.zeros(*p.shape[:-1], 3, device=h.device, dtype=h.dtype)
        direction[..., axis] = 1
        theta = self.angle(torch.cat((p, q, direction), -1))
        mean = (p + q) / math.sqrt(2)
        difference = (p - q).reshape(*p.shape[:-1], self.d // 2, 2) / math.sqrt(2)
        cosine, sine = theta.cos(), theta.sin()
        rotated = torch.stack((cosine * difference[..., 0] - sine * difference[..., 1],
                               sine * difference[..., 0] + cosine * difference[..., 1]), -1)
        rotated = rotated.flatten(-2)
        p_new = (mean + rotated) / math.sqrt(2)
        q_new = (mean - rotated) / math.sqrt(2)
        result = torch.empty_like(moved)
        if parity == 0:
            result[:, 0::2], result[:, 1::2] = p_new, q_new
        else:
            result[:, 1::2] = p_new
            result[:, 0::2] = torch.roll(q_new, 1, 1)
        result = torch.roll(result, channel_shift, -1)
        return result.movedim(1, axis + 1)

    def forward(self, h):
        result = h
        layer = 0
        for _ in range(self.sweeps):
            for axis in range(3):
                for parity in range(2):
                    result = self._pairs(result, axis, parity, layer % self.d)
                    layer += 1
        return result, {'scattering_layers_applied': self.layers_applied}


class OperatorReadout3D(nn.Module):
    """State-only quadrature attention with continuous positional features."""

    def __init__(self, shape=(4, 4, 4), d=128, queries=4, heads=4):
        super().__init__()
        if d % heads:
            raise ValueError('d must be divisible by heads')
        self.shape, self.d, self.queries, self.heads = tuple(shape), d, queries, heads
        self.query = nn.Parameter(torch.randn(1, queries, d) * .02)
        self.position = nn.Sequential(nn.Linear(6, d), nn.SiLU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)
        self.key, self.value = nn.Linear(d, d), nn.Linear(d, d)
        self.merge = nn.Linear(queries * d, d)
        self.output = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.register_buffer('position_features',
                             _position_features(_grid(self.shape)).reshape(1, -1, 6),
                             persistent=False)

    def position_encoding(self):
        return self.position(self.position_features)

    def forward(self, h, position_encoding=None):
        batch, d, heads = h.shape[0], self.d, self.heads
        state = self.norm(h.reshape(batch, -1, d))
        q = self.query.expand(batch, -1, -1)
        position_encoding = (self.position_encoding() if position_encoding is None
                             else position_encoding)
        k = self.key(state + position_encoding)
        v = self.value(state)
        split = lambda x: x.reshape(batch, -1, heads, d // heads).transpose(1, 2)
        read = F.scaled_dot_product_attention(split(q), split(k), split(v), dropout_p=0.)
        read = read.transpose(1, 2).reshape(batch, self.queries * d)
        return self.output(self.merge(read))


class CBIMOperator3D(nn.Module):
    """One finite truncation of the same continuous 3D memory operator."""

    def __init__(self, vocab_size=50257, shape=(4, 4, 4), d=128,
                 cell_radius=1.25, scatter_sweeps=1, queries=4, heads=4):
        super().__init__()
        self.vocab_size, self.shape, self.d = vocab_size, tuple(shape), d
        self.L = math.prod(shape)
        self.state_shape = (*self.shape, d)
        self.source = LocalSourceOutflow3D(vocab_size, shape, d, cell_radius)
        self.transport = ContinuousCayleyTransport3D(shape, d)
        self.scattering = ConservativeScattering3D(shape, d, scatter_sweeps)
        self.readout = OperatorReadout3D(shape, d, queries, heads)
        self.decoder = nn.Linear(d, vocab_size)
        self.decoder.weight = self.source.embedding.weight

    def step(self, h, token_ids, *, disable_scattering=False):
        h, source_diag = self.source(h, token_ids)
        h, transport_diag = self.transport(h)
        if disable_scattering:
            scatter_diag = {'scattering_layers_applied': torch.zeros((), device=h.device)}
        else:
            h, scatter_diag = self.scattering(h)
        logits = self.decoder(self.readout(h))
        energy = .5 * h.detach().square().sum(-1).mean(dim=(1, 2, 3)).mean()
        return logits, h, {'source': source_diag, 'transport': transport_diag,
                           'scattering': scatter_diag, 'energy': energy}

    def forward(self, input_ids, targets, initial_h=None, *, disable_scattering=False):
        batch, length = input_ids.shape
        h = (torch.zeros(batch, *self.state_shape, device=input_ids.device)
             if initial_h is None else initial_h)
        initial_energy = .5 * h.detach().square().sum(-1).mean()
        features, energies, projections = [], [], []
        multiplier, _ = self.transport.multiplier()
        position_encoding = self.readout.position_encoding()
        for t in range(length):
            h, source_diag = self.source(h, input_ids[:, t])
            h = self.transport.apply_multiplier(h, multiplier)
            if not disable_scattering:
                h, _ = self.scattering(h)
            features.append(self.readout(h, position_encoding))
            energies.append(.5 * h.detach().square().sum(-1).mean())
            projections.append(source_diag['local_projection_rate'])
        logits = self.decoder(torch.stack(features, 1))
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        return loss, h, {
            'initial_energy': initial_energy,
            'final_energy': energies[-1],
            'mean_energy': torch.stack(energies).mean(),
            'local_projection_rate': torch.stack(projections).mean(),
        }
