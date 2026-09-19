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
    def __init__(self, vocab=50257, hidden=128, particles=512, steps=4, collision_hidden=32, use_write_operator=False, coupling_mode='none', score_scale=1.0):
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
        self.use_write_operator = use_write_operator
        self.coupling_mode = coupling_mode
        self.score_scale = float(score_scale)
        self.write_operator = None
        self.adaptive_force = None
        self.message_coupling = None
        if coupling_mode == 'adaptive_force':
            from .adaptive_force import UnifiedAdaptiveForce
            self.adaptive_force = UnifiedAdaptiveForce(vocab=vocab, dim=4, hidden_dim=hidden, n_particles=particles)
        elif coupling_mode == 'message_coupling':
            from .message_coupling import ParticleMessageCoupling
            self.message_coupling = ParticleMessageCoupling(dim=4, hidden_dim=hidden, n_particles=particles)
        elif use_write_operator:
            from .write_operator import WriteOperator
            self.write_operator = WriteOperator(dim=4, hidden_dim=hidden)

    def kick(self, x, v, shared, noise, h, gamma=None):
        force = self.core.force
        if shared.shape[-1] == 4:
            drive = shared
        else:
            drive = .5 * force.net[2](force.net[1](F.linear(x, force.net[0].weight[:, :4]) + shared)).tanh()
        if gamma is None:
            gamma = force.damping(x)
        if x.is_cuda:
            from scripts.ib_fused_ou import FusedOU
            return FusedOU.apply(v, -x + drive, gamma, noise, h, .1)
        decay = torch.exp(-gamma * h)
        return decay * v - torch.expm1(-gamma * h) / gamma * (-x + drive) + torch.sqrt(-.1 * torch.expm1(-2 * gamma * h)) * noise

    def evolve(self, x, v, shared, clocks, noise, layers, gamma=None):
        dt = 1. / self.steps
        for half in range(2):
            if shared.shape[-1] == 4:
                clock_half = clocks[half, :4] if clocks.shape[-1] >= 4 else clocks[half]
                drive_input = shared + clock_half
            else:
                drive_input = shared + clocks[half]
            v = self.kick(x, v, drive_input, noise[half * 2], dt / 4, gamma=gamma)
            x = x + dt / 2 * v
            v = self.kick(x, v, drive_input, noise[half * 2 + 1], dt / 4, gamma=gamma)
            if half == 0:
                v, lp, accepted = collision_device(x, v, self.core.collision, layers,
                                                   layer_fn=self.layer_fn)
        return x, v, lp, accepted

    def forward(self, x, v, input_ids, targets, clocks, noise, tables):
        force = self.core.force
        layer = force.net[0]
        tok_embs = force.embedding(input_ids)
        shared = F.linear(tok_embs, layer.weight[:, 4:-2], layer.bias)
        clock_proj = F.linear(clocks, layer.weight[:, -2:])
        if self.block_engine is not None:
            return self.forward_blocks(x, v, shared, targets, clock_proj, noise, tables, tok_embs=tok_embs, input_ids=input_ids)
        features, prefixes = [], []
        lp, accepted = v.sum() * 0., v.new_zeros(())
        for t in range(len(input_ids)):
            gamma_t = None
            drive_t = shared[t]
            if self.coupling_mode == 'message_coupling':
                x, v, gamma_t, _ = self.message_coupling(x, v, tok_embs[t])
                drive_t = torch.zeros_like(v)
            elif self.coupling_mode == 'adaptive_force':
                drive_t, gamma_t, _ = self.adaptive_force(x, v, tok_embs[t])
            elif self.write_operator is not None:
                x, v, _ = self.write_operator(x, v, tok_embs[t])
            for s in range(self.steps):
                index = t * self.steps + s
                args = (x, v, drive_t, clock_proj[t, s], noise[index * 4:index * 4 + 4], tables[index], gamma_t)
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
        surrogate = ce.mean() + self.score_scale * (score - score.detach())
        return surrogate, ce.mean(), x, v, accepted

    def feature_block(self, x, v, shared, clocks, noise, tables, tok_embs=None, eager=False):
        features, prefixes = [], []
        lp, accepted = v.sum() * 0., v.new_zeros(())
        evolve = LocalWindow.evolve.__get__(self, LocalWindow) if eager else self.evolve
        for t in range(len(shared)):
            gamma_t = None
            drive_t = shared[t]
            if self.coupling_mode == 'message_coupling' and tok_embs is not None:
                x, v, gamma_t, _ = self.message_coupling(x, v, tok_embs[t])
                drive_t = torch.zeros(self.core.particles, 4, device=x.device, dtype=x.dtype)
            elif self.coupling_mode == 'adaptive_force' and tok_embs is not None:
                F_in, gamma_t, _ = self.adaptive_force(x, v, tok_embs[t])
                drive_t = F_in
            elif self.write_operator is not None and tok_embs is not None:
                x, v, _ = self.write_operator(x, v, tok_embs[t])

            for s in range(self.steps):
                index = t * self.steps + s
                x, v, change, count = evolve(x, v, drive_t, clocks[t, s],
                                             noise[index * 4:index * 4 + 4], tables[index], gamma=gamma_t)
                lp = lp + change
                accepted = accepted + count
            features.append(self.core.features(torch.cat((x, v), -1)).mean(0))
            prefixes.append(lp)
        return x, v, torch.stack(features), torch.stack(prefixes), accepted

    def forward_blocks(self, x, v, shared, targets, clocks, noise, tables, tok_embs=None, input_ids=None):
        engine = self.block_engine
        packed = engine.packed(tables)
        features, prefixes = [], []
        prior, accepted = v.sum() * 0., v.new_zeros(())
        for start in range(0, len(shared), engine.block_tokens):
            end = min(start + engine.block_tokens, len(shared))
            a, b = start * self.steps, end * self.steps
            t_ids = input_ids[start:end] if input_ids is not None else None
            x, v, f, lp, count = engine.run(x, v, shared[start:end], clocks[start:end],
                                           noise[a * 4:b * 4], tables[a:b], packed, token_ids=t_ids)
            features.append(f)
            prefixes.append(prior + lp)
            prior = prior + lp[-1]
            accepted = accepted + count
        logits = F.linear(torch.cat(features), self.core.decoder.weight, self.core.decoder.bias)
        ce = F.cross_entropy(logits, targets, reduction='none')
        score = ((ce.detach() - math.log(self.core.vocab_size)) * torch.cat(prefixes)).mean()
        return ce.mean() + self.score_scale * (score - score.detach()), ce.mean(), x, v, accepted

    def forward_path(self, x, v, input_ids, targets, clocks, noise, tables):
        force = self.core.force
        layer = force.net[0]
        tok_embs = force.embedding(input_ids)
        shared = F.linear(tok_embs, layer.weight[:, 4:-2], layer.bias)
        clock_proj = F.linear(clocks, layer.weight[:, -2:])
        curr_x, curr_v = x.detach().clone().requires_grad_(), v.detach().clone().requires_grad_()

        if self.block_engine is None:
            features = []
            for t in range(len(input_ids)):
                gamma_t = None
                drive_t = shared[t]
                if self.coupling_mode == 'message_coupling':
                    curr_x, curr_v, gamma_t, _ = self.message_coupling(curr_x, curr_v, tok_embs[t])
                    drive_t = torch.zeros(self.core.particles, 4, device=x.device, dtype=x.dtype)
                elif self.coupling_mode == 'adaptive_force':
                    F_in, gamma_t, _ = self.adaptive_force(curr_x, curr_v, tok_embs[t])
                    drive_t = F_in

                for s in range(self.steps):
                    idx = t * self.steps + s
                    curr_x, curr_v, change, count = self.evolve(curr_x, curr_v, drive_t, clock_proj[t, s],
                                                                noise[idx * 4:idx * 4 + 4], tables[idx], gamma=gamma_t)
                features.append(self.core.features(torch.cat((curr_x, curr_v), -1)).mean(0))
            logits = F.linear(torch.stack(features), self.core.decoder.weight, self.core.decoder.bias)
            ce = F.cross_entropy(logits, targets, reduction='none')
            return ce.mean(), curr_x.detach(), curr_v.detach(), ce.detach()

        engine = self.block_engine
        packed = engine.packed(tables)
        features = []
        for start in range(0, len(shared), engine.block_tokens):
            end = min(start + engine.block_tokens, len(shared))
            a, b = start * self.steps, end * self.steps
            t_embs = tok_embs[start:end] if tok_embs is not None else None
            curr_x, curr_v, f, _, _ = engine.run(curr_x, curr_v, shared[start:end], clock_proj[start:end],
                                                 noise[a * 4:b * 4], tables[a:b], packed, token_ids=input_ids[start:end])
            features.append(f)
        logits = F.linear(torch.cat(features), self.core.decoder.weight, self.core.decoder.bias)
        ce = F.cross_entropy(logits, targets, reduction='none')
        return ce.mean(), curr_x.detach(), curr_v.detach(), ce.detach()

    def forward_score(self, x, v, input_ids, targets, clocks, noise, tables, ce_detach):
        force = self.core.force
        layer = force.net[0]
        tok_embs = force.embedding(input_ids)
        shared = F.linear(tok_embs, layer.weight[:, 4:-2], layer.bias)
        clock_proj = F.linear(clocks, layer.weight[:, -2:])
        curr_x, curr_v = x.detach().clone().requires_grad_(), v.detach().clone().requires_grad_()

        if self.block_engine is None:
            prefixes = []
            prior, accepted = v.sum() * 0., v.new_zeros(())
            for t in range(len(input_ids)):
                gamma_t = None
                drive_t = shared[t]
                if self.coupling_mode == 'message_coupling':
                    curr_x, curr_v, gamma_t, _ = self.message_coupling(curr_x, curr_v, tok_embs[t])
                    drive_t = torch.zeros(self.core.particles, 4, device=x.device, dtype=x.dtype)
                elif self.coupling_mode == 'adaptive_force':
                    F_in, gamma_t, _ = self.adaptive_force(curr_x, curr_v, tok_embs[t])
                    drive_t = F_in

                for s in range(self.steps):
                    idx = t * self.steps + s
                    curr_x, curr_v, change, count = self.evolve(curr_x, curr_v, drive_t, clock_proj[t, s],
                                                                noise[idx * 4:idx * 4 + 4], tables[idx], gamma=gamma_t)
                    prior = prior + change
                    accepted = accepted + count
                prefixes.append(prior)
            baseline = math.log(self.core.vocab_size)
            score = ((ce_detach - baseline) * torch.stack(prefixes)).mean()
            return score, curr_x.detach(), curr_v.detach(), accepted

        engine = self.block_engine
        packed = engine.packed(tables)
        prefixes = []
        prior, accepted = v.sum() * 0., v.new_zeros(())
        for start in range(0, len(shared), engine.block_tokens):
            end = min(start + engine.block_tokens, len(shared))
            a, b = start * self.steps, end * self.steps
            t_embs = tok_embs[start:end] if tok_embs is not None else None
            curr_x, curr_v, _, lp, count = engine.run(curr_x, curr_v, shared[start:end], clock_proj[start:end],
                                                      noise[a * 4:b * 4], tables[a:b], packed, token_ids=input_ids[start:end])
            prefixes.append(prior + lp)
            prior = prior + lp[-1]
            accepted = accepted + count
        baseline = math.log(self.core.vocab_size)
        score = ((ce_detach - baseline) * torch.cat(prefixes)).mean()
        return score, curr_x.detach(), curr_v.detach(), accepted
