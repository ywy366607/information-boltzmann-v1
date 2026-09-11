"""Causal BPE windows using strict local elastic collision events."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from fine_grain.information_boltzmann import InformationBoltzmann
from fine_grain.information_boltzmann.collision import CollisionKernel
from scripts.ib_local.density import BirthDensity
from scripts.ib_local.device_collision import collision_device, collision_layer


class LocalWindow(nn.Module):
    def __init__(self, vocab=50257, hidden=128, particles=512, steps=4, collision_hidden=32):
        super().__init__()
        self.core = InformationBoltzmann(vocab_size=vocab, phase_dim=4, particles=particles,
                                         hidden_dim=hidden, steps=steps, gamma=.3,
                                         gamma_mode='local', temperature=.1)
        self.core.initial = BirthDensity(vocab, 4, hidden)
        self.core.initial.embedding = self.core.force.embedding
        self.core.decoder.weight = self.core.force.embedding.weight
        self.core.collision = CollisionKernel(4, collision_hidden)
        self.steps = steps
        self.recompute = True
        self.layer_fn = collision_layer
        self.block_engine = None

    def kick(self, x, v, shared, noise, h):
        force = self.core.force
        drive = .5 * force.net[2](force.net[1](F.linear(x, force.net[0].weight[:, :4]) + shared)).tanh()
        gamma = force.damping(x)
        if x.is_cuda:
            from scripts.ib_fused_ou import FusedOU
            return FusedOU.apply(v, -x + drive, gamma, noise, h, .1)
        decay = torch.exp(-gamma * h)
        return decay * v - torch.expm1(-gamma * h) / gamma * (-x + drive) + torch.sqrt(-.1 * torch.expm1(-2 * gamma * h)) * noise

    def evolve(self, x, v, shared, clocks, noise, layers):
        dt = 1. / self.steps
        for half in range(2):
            drive_input = shared + clocks[half]
            v = self.kick(x, v, drive_input, noise[half * 2], dt / 4)
            x = x + dt / 2 * v
            v = self.kick(x, v, drive_input, noise[half * 2 + 1], dt / 4)
            if half == 0:
                v, lp, accepted = collision_device(x, v, self.core.collision, layers,
                                                   layer_fn=self.layer_fn)
        return x, v, lp, accepted

    def forward(self, x, v, input_ids, targets, clocks, noise, tables):
        force = self.core.force
        layer = force.net[0]
        shared = F.linear(force.embedding(input_ids), layer.weight[:, 4:-2], layer.bias)
        clock_proj = F.linear(clocks, layer.weight[:, -2:])
        if self.block_engine is not None:
            return self.forward_blocks(x, v, shared, targets, clock_proj, noise, tables)
        features, prefixes = [], []
        lp, accepted = v.sum() * 0., v.new_zeros(())
        for t in range(len(input_ids)):
            for s in range(self.steps):
                index = t * self.steps + s
                args = (x, v, shared[t], clock_proj[t, s], noise[index * 4:index * 4 + 4], tables[index])
                if torch.is_grad_enabled() and self.recompute:
                    x, v, change, count = checkpoint(self.evolve, *args, use_reentrant=False, preserve_rng_state=False)
                else:
                    x, v, change, count = self.evolve(*args)
                lp = lp + change
                accepted = accepted + count
            z = torch.cat((x, v), -1)
            def readout(z):
                return self.core.features(z).mean(0)
            feature = checkpoint(readout, z, use_reentrant=False, preserve_rng_state=False) if torch.is_grad_enabled() else readout(z)
            features.append(feature)
            prefixes.append(lp)
        logits = F.linear(torch.stack(features), self.core.decoder.weight, self.core.decoder.bias)
        ce = F.cross_entropy(logits, targets, reduction='none')
        # Constant baseline; causal score prefixes, no duplicated probability gradient.
        score = ((ce.detach() - math.log(self.core.vocab_size)) * torch.stack(prefixes)).mean()
        surrogate = ce.mean() + score - score.detach()
        return surrogate, ce.mean(), x, v, accepted

    def feature_block(self, x, v, shared, clocks, noise, tables, eager=False):
        features, prefixes = [], []
        lp, accepted = v.sum() * 0., v.new_zeros(())
        evolve = LocalWindow.evolve.__get__(self, LocalWindow) if eager else self.evolve
        for t in range(len(shared)):
            for s in range(self.steps):
                index = t * self.steps + s
                x, v, change, count = evolve(x, v, shared[t], clocks[t, s],
                                                  noise[index * 4:index * 4 + 4], tables[index])
                lp = lp + change
                accepted = accepted + count
            features.append(self.core.features(torch.cat((x, v), -1)).mean(0))
            prefixes.append(lp)
        return x, v, torch.stack(features), torch.stack(prefixes), accepted

    def forward_blocks(self, x, v, shared, targets, clocks, noise, tables):
        engine = self.block_engine
        packed = engine.packed(tables)
        features, prefixes = [], []
        prior, accepted = v.sum() * 0., v.new_zeros(())
        for start in range(0, len(shared), engine.block_tokens):
            end = min(start + engine.block_tokens, len(shared))
            a, b = start * self.steps, end * self.steps
            x, v, f, lp, count = engine.run(x, v, shared[start:end], clocks[start:end],
                                           noise[a * 4:b * 4], tables[a:b], packed)
            features.append(f)
            prefixes.append(prior + lp)
            prior = prior + lp[-1]
            accepted = accepted + count
        logits = F.linear(torch.cat(features), self.core.decoder.weight, self.core.decoder.bias)
        ce = F.cross_entropy(logits, targets, reduction='none')
        score = ((ce.detach() - math.log(self.core.vocab_size)) * torch.cat(prefixes)).mean()
        return ce.mean() + score - score.detach(), ce.mean(), x, v, accepted
