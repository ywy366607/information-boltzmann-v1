"""32-event truncated BPTT for one continuing sensory/motor COBA individual.

Physical values persist between windows. Only autograd history is detached.
The spike backward is the model's existing ATan surrogate, not a derivative
of the discontinuous hard threshold. Static edge topology remains fixed.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
import torch
from torch import nn
import torch.nn.functional as F

STP_PARAMETER_NAMES = ('logit_u0', 'log_tau_fac', 'log_tau_rec')


@dataclass
class FlyPhysicalState:
    h: torch.Tensor
    ring: tuple[torch.Tensor, ...]
    ge: torch.Tensor
    gi: torch.Tensor
    b: torch.Tensor
    x: torch.Tensor
    u: torch.Tensor
    baseline: torch.Tensor
    # Slow running mean of the membrane state (the read pathway's background
    # estimate).  Empty until the first event initializes it from h.
    h_mean: torch.Tensor = field(default_factory=lambda: torch.empty(0))

    def detached(self):
        return FlyPhysicalState(**{
            item.name: tuple(t.detach() for t in self.ring) if item.name == 'ring'
            else getattr(self, item.name).detach() for item in fields(self)})

    def state_dict(self):
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_online(cls, state, device):
        syn = state['syn']
        h = state['h'].to(device)
        h_mean = state.get('h_mean')
        h_mean = (h_mean.to(device) if h_mean is not None and h_mean.numel() == h.numel()
                  else torch.zeros_like(h))
        return cls(h, tuple(t.to(device) for t in state['ring']),
                   *(syn[key].to(device) for key in ('ge', 'gi', 'b', 'x', 'u')),
                   state['writer_baseline'].to(device), h_mean)


def advance_fly_input_event(model, state, token, *, settle_ticks=0,
                            writer_baseline_clock='input',
                            base_rates=None, thresholds=None,
                            conductance_gains=None, alif_params=None,
                            stp_params=None, h_mean_decay=0.99):
    """One token pulse followed by quiet physical ticks, before next-token read.

    Tick duration and the existing four-slot delays are unchanged. A quiet tick
    advances every physical state but neither embeds nor injects another token.
    The packet prediction baseline updates only at observed input events.
    ``writer_baseline_clock='physical'`` reproduces the archived quiet decay.
    ``settle_ticks=0`` exactly retains the archived one-tick event ordering.
    This is the shared forward interface for learning, evaluation and generation.
    """
    if not isinstance(settle_ticks, int) or settle_ticks < 0:
        raise ValueError('settle_ticks must be a nonnegative integer')
    if writer_baseline_clock not in ('input', 'physical'):
        raise ValueError('writer_baseline_clock must be input or physical')
    drive, baseline = model.topographic_writer.forward_with_state(
        model.embedding(token), state.h, state.baseline)
    options = dict(base_rates=base_rates, thresholds=thresholds,
                   conductance_gains=conductance_gains,
                   alif_params=alif_params, stp_params=stp_params)

    def tick(current, source, next_baseline):
        h, _, ring, ge, gi, b, x, u = model.step(
            current.h, token, spike_ring=current.ring, ge=current.ge,
            gi=current.gi, b=current.b, x=current.x, u=current.u,
            sensory_drive=source, **options)
        h_mean = current.h_mean
        if h_mean.numel() != h.numel():
            h_mean = torch.zeros_like(h)
        h_mean = h_mean_decay * h_mean + (1.0 - h_mean_decay) * h
        return FlyPhysicalState(h, ring, ge, gi, b, x, u, next_baseline, h_mean)

    state = tick(state, drive, baseline)
    if settle_ticks:
        quiet_source = torch.zeros_like(drive)
        for _ in range(settle_ticks):
            baseline = (model.topographic_writer.lambda_adapt * state.baseline
                        if writer_baseline_clock == 'physical' else state.baseline)
            state = tick(state, quiet_source, baseline)
    return state


def predict_fly_next(model, state, token, *, settle_ticks, writer_baseline_clock):
    """Label-free generation/read interface with the same event order as BPTT.

    The caller must supply the checkpoint's timing explicitly. Sampling or
    teacher-forcing the next token happens only after this function returns.
    """
    state = advance_fly_input_event(model, state, token, settle_ticks=settle_ticks,
                                    writer_baseline_clock=writer_baseline_clock)
    read_input = (state.h - state.h_mean
                  if getattr(model, 'read_centering', False)
                  and state.h_mean.numel() == state.h.numel() else state.h)
    return model.read(read_input), state


class FlyBPTTLearner:
    def __init__(self, model, state, *, lr=2e-4, lr_synapse=None,
                 lr_sensory=None, max_grad_norm=1.0, adam_names=None,
                 plasticity_optimizer='adamw', lr_decoder=None, settle_ticks=0,
                 writer_baseline_clock='input', learn_stp=False):
        if plasticity_optimizer not in ('adamw', 'sgd'):
            raise ValueError('plasticity_optimizer must be adamw or sgd')
        if not isinstance(settle_ticks, int) or settle_ticks < 0:
            raise ValueError('settle_ticks must be a nonnegative integer')
        self.settle_ticks = settle_ticks
        if writer_baseline_clock not in ('input', 'physical'):
            raise ValueError('writer_baseline_clock must be input or physical')
        self.writer_baseline_clock = writer_baseline_clock
        lr_synapse = lr if lr_synapse is None else lr_synapse
        lr_sensory = lr if lr_sensory is None else lr_sensory
        self.model, self.state = model, state.detached()
        model.requires_grad_(False)
        # Promote existing learned edge buffers without altering topology/values.
        for name in ('edge_weight_e', 'edge_weight_i'):
            value = getattr(model, name)
            if name in model._buffers:
                del model._buffers[name]
                model.register_parameter(name, nn.Parameter(value))
        named = dict(model.named_parameters())
        self.adam_names = list(adam_names or (
            'output_read.weight', 'read_norm.weight', 'decoder.weight', 'decoder.bias',
            'log_threshold', 'log_tau_m', 'log_beta', 'log_tau_a',
            'log_tau_s_e', 'log_tau_s_i', 'log_g_e', 'log_g_i',
            'topographic_writer.gate_linear.weight', 'topographic_writer.gate_linear.bias'))
        self.adam_names = [name for name in self.adam_names if name in named]
        if learn_stp:
            if not getattr(model, 'use_stp', False):
                raise ValueError('STP learning requires an STP-enabled model')
            for name in STP_PARAMETER_NAMES:
                if name not in named:
                    raise ValueError(f'Missing STP parameter: {name}')
                if name not in self.adam_names:
                    self.adam_names.append(name)
        self.learn_stp = all(name in self.adam_names for name in STP_PARAMETER_NAMES)
        decayed = [named[name] for name in self.adam_names
                   if name in ('output_read.weight', 'decoder.weight')]
        undecayed = [named[name] for name in self.adam_names
                     if name not in ('output_read.weight', 'decoder.weight')]
        for name in self.adam_names:
            named[name].requires_grad_(True)
        if lr_decoder is None:
            adam_groups = [{'params': decayed, 'weight_decay': 1e-4},
                           {'params': undecayed, 'weight_decay': 0.0}]
        else:
            adam_groups = [
                {'params': [p for p in decayed if p is not model.decoder.weight],
                 'weight_decay': 1e-4, 'lr': lr},
                {'params': [model.decoder.weight], 'weight_decay': 1e-4, 'lr': lr_decoder},
                {'params': undecayed, 'weight_decay': 0.0, 'lr': lr}]
        parameter_names = {id(p): n for n, p in named.items()}
        for group in adam_groups:
            group['parameter_names'] = [parameter_names[id(p)] for p in group['params']]
        self.optimizer = torch.optim.AdamW(adam_groups, lr=lr, fused=state.h.is_cuda)
        self.projections = [getattr(model.topographic_writer, name).weight
                            for name in ('proj_vis', 'proj_chemo', 'proj_mech')]
        self.edges = [model.edge_weight_e, model.edge_weight_i]
        for parameter in self.projections + self.edges:
            parameter.requires_grad_(True)
        plasticity_groups = [
            {'params': self.projections, 'lr': lr_sensory, 'weight_decay': 1e-4},
            {'params': self.edges, 'lr': lr_synapse, 'weight_decay': 0.0}]
        self.plasticity_optimizer_kind = plasticity_optimizer
        # Retain the legacy attribute/checkpoint key for old continuation tools.
        # New default gives sensory projections and edges adaptive scaling too.
        if plasticity_optimizer == 'adamw':
            self.sgd = torch.optim.AdamW(plasticity_groups, fused=state.h.is_cuda)
        else:
            self.sgd = torch.optim.SGD(plasticity_groups, foreach=False)
        self.trainable = [p for p in model.parameters() if p.requires_grad]
        self.max_grad_norm = max_grad_norm
        self.events = self.updates = 0
        self.physical_ticks = 0
        self.previous_token = None
        self.ema = 0.0
        self.latent_window = torch.zeros(128, model.embedding.embedding_dim, device=state.h.device)
        self.runner = None

    def load_adam_state(self, saved, *, newly_trainable=()):
        """Retain per-parameter Adam history across a decoder group split.

        Old two-group checkpoints have no name metadata. Their ordering is
        determined by adam_names and the original matrix weight-decay policy.
        New checkpoints record explicit names, making future regrouping exact.
        Requested learning rates remain those of the destination optimizer.
        """
        newly_trainable = set(newly_trainable)
        if not newly_trainable.issubset(STP_PARAMETER_NAMES):
            raise ValueError('Only explicitly activated STP parameters may start new moments')
        groups = saved['param_groups']
        if all('parameter_names' in group for group in groups):
            name_lists = [group['parameter_names'] for group in groups]
        elif len(groups) == 2:
            name_lists = [[n for n in self.adam_names if n not in newly_trainable
                           if n in ('output_read.weight', 'decoder.weight')],
                          [n for n in self.adam_names if n not in newly_trainable
                           if n not in ('output_read.weight', 'decoder.weight')]]
        else:
            raise ValueError('Optimizer checkpoint lacks parameter-name metadata')
        old_by_name, old_options = {}, {}
        for group, names in zip(groups, name_lists):
            if len(names) != len(group['params']):
                raise ValueError('Optimizer parameter ordering mismatch')
            for name, index in zip(names, group['params']):
                if name in old_by_name:
                    raise ValueError(f'Duplicate optimizer parameter: {name}')
                old_by_name[name] = saved['state'].get(index)
                old_options[name] = group
        if set(old_by_name) != set(self.adam_names) - newly_trainable:
            raise ValueError('Optimizer parameter coverage changed')
        destination = self.optimizer.state_dict()
        migrated = {}
        for group in destination['param_groups']:
            for index, name in zip(group['params'], group['parameter_names']):
                entry = old_by_name.get(name)
                if entry is not None:
                    migrated[index] = entry
            retained = [name for name in group['parameter_names'] if name in old_options]
            if retained:
                options = old_options[retained[0]]
                for key in ('betas', 'eps', 'amsgrad', 'maximize'):
                    if key in options:
                        group[key] = options[key]
            group['fused'] = self.state.h.is_cuda
        self.optimizer.load_state_dict({'state': migrated, 'param_groups': destination['param_groups']})
        if self.state.h.is_cuda:
            for entry in self.optimizer.state.values():
                if isinstance(entry.get('step'), torch.Tensor):
                    entry['step'] = entry['step'].cuda()

    def forward_window(self, input_ids, targets):
        """Every score precedes the single parameter update for this window."""
        m, state = self.model, self.state
        rates, thresholds = m.get_decay_rates(), m.get_thresholds()
        gains, alif, stp = m.get_conductance_gains(), m.get_alif_params(), m.get_stp_params()
        latents = []
        for token in input_ids.unbind(1):
            state = advance_fly_input_event(m, state, token,
                settle_ticks=self.settle_ticks, writer_baseline_clock=self.writer_baseline_clock,
                base_rates=rates,
                thresholds=thresholds, conductance_gains=gains,
                alif_params=alif, stp_params=stp)
            read_in = (state.h - state.h_mean
                       if getattr(m, 'read_centering', False)
                       and state.h_mean.numel() == state.h.numel() else state.h)
            latents.append(m.output_read(read_in[:, m.read_indices]))
        features = torch.cat(latents, dim=0)
        logits = m.decoder(m.read_norm(features))
        scores = F.cross_entropy(logits, targets.flatten(), reduction='none')
        return scores, state, features

    def observe(self, target_tokens):
        tokens = torch.as_tensor(target_tokens, device=self.state.h.device, dtype=torch.long).flatten()
        if self.previous_token is None:
            raise ValueError('A continuing previous token is required')
        ids = torch.cat((tokens.new_tensor([self.previous_token]), tokens[:-1]))[None]
        if self.runner is None:
            self.optimizer.zero_grad(set_to_none=True)
            self.sgd.zero_grad(set_to_none=True)
            scores, next_state, features = self.forward_window(ids, tokens[None])
        else:
            scores, next_state, features = self.runner.replay(ids, tokens[None])
        loss = scores.mean()
        if not torch.isfinite(loss).item():
            raise FloatingPointError('Non-finite BPTT training loss')
        if self.runner is None:
            loss.backward()
        group_norms = {
            'synapse_grad_norm': float(torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in self.edges]))),
            'writer_grad_norm': float(torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in self.projections]))),
        }
        # Keep the historical writer projection norm, but distinguish it from
        # writer gates, output projection, normalization gain and vocabulary.
        gate_parameters = list(self.model.topographic_writer.gate_linear.parameters())
        group_norms.update(
            writer_gate_grad_norm=float(torch.linalg.vector_norm(torch.stack(
                [p.grad.norm() for p in gate_parameters]))),
            read_grad_norm=float(self.model.output_read.weight.grad.norm()),
            decoder_grad_norm=float(self.model.decoder.weight.grad.norm()),
            read_norm_grad_norm=float(self.model.read_norm.weight.grad.norm()))
        group_norms['writer_total_grad_norm'] = (
            group_norms['writer_grad_norm']**2 + group_norms['writer_gate_grad_norm']**2)**.5
        group_norms['writer_gradient_rms'] = group_norms['writer_total_grad_norm'] / (
            sum(p.numel() for p in self.projections + gate_parameters)**.5)
        group_norms['synapse_gradient_rms'] = group_norms['synapse_grad_norm'] / (
            sum(p.numel() for p in self.edges)**.5)
        group_norms['read_gradient_rms'] = group_norms['read_grad_norm'] / (
            self.model.output_read.weight.numel()**.5)
        group_norms['decoder_gradient_rms'] = group_norms['decoder_grad_norm'] / (
            self.model.decoder.weight.numel()**.5)
        if self.learn_stp:
            for name in STP_PARAMETER_NAMES:
                group_norms[f'{name}_grad_norm'] = float(getattr(self.model, name).grad.norm())
        grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable, self.max_grad_norm,
                                                   error_if_nonfinite=True)
        self.optimizer.step()
        self.sgd.step()
        with torch.no_grad():
            for weight in self.edges:
                weight.clamp_(0.0, 5.0)  # Same sign/bounds as the archived online learner.
            self.model.topographic_writer.a_adapt.copy_(next_state.baseline)
            for i, feature in enumerate(features):
                self.latent_window[(self.events + i) % 128].copy_(feature)
        self.state = next_state.detached()
        values = scores.detach().cpu().tolist()
        for value in values:
            self.ema = value if self.events == 0 else .99 * self.ema + .01 * value
            self.events += 1
        self.previous_token = int(tokens[-1].item())
        self.physical_ticks += len(values) * (1 + self.settle_ticks)
        self.updates += 1
        self.optimizer.zero_grad(set_to_none=True)
        self.sgd.zero_grad(set_to_none=True)
        return values, {'grad_norm_before_clip': float(grad_norm), **group_norms}

    def state_dict(self):
        return {'physical': self.state.state_dict(), 'optimizer': self.optimizer.state_dict(),
                'sgd': self.sgd.state_dict(), 'adam_names': self.adam_names,
                'plasticity_optimizer_kind': self.plasticity_optimizer_kind,
                'events': self.events, 'updates': self.updates,
                'physical_ticks': self.physical_ticks, 'settle_ticks': self.settle_ticks,
                'writer_baseline_clock': self.writer_baseline_clock,
                'learn_stp': self.learn_stp,
                'previous_token': self.previous_token, 'ema': self.ema,
                'latent_window': self.latent_window}


class FlyBPTTGraph:
    """Capture forward/backward only; optimizer executes once per real window.

    Warm-up never updates weights or the continuing state. Inputs and all
    physical states are copied into fixed buffers before each actual replay.
    """
    def __init__(self, learner, window=32):
        if not learner.state.h.is_cuda:
            raise ValueError('CUDA Graph requires CUDA')
        self.learner, self.window = learner, window
        self.ids = torch.zeros(1, window, dtype=torch.long, device=learner.state.h.device)
        self.targets = torch.zeros_like(self.ids)
        original = learner.state
        self.initial = FlyPhysicalState(**{
            key: tuple(t.clone() for t in value) if key == 'ring' else value.clone()
            for key, value in original.state_dict().items()})

        def capture_step():
            learner.state = self.initial
            try:
                result = learner.forward_window(self.ids, self.targets)
                result[0].mean().backward()
                return result
            finally:
                learner.state = original

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                learner.model.zero_grad(set_to_none=True)
                warm_result = capture_step()
                del warm_result
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        learner.model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = capture_step()
        self.gradients = [(parameter, parameter.grad) for parameter in learner.trainable]
        if any(gradient is None for _, gradient in self.gradients):
            raise RuntimeError('A trainable parameter has no captured gradient')

    def replay(self, ids, targets):
        if ids.shape != self.ids.shape:
            raise ValueError('Captured BPTT windows must have the registered length')
        with torch.no_grad():
            for parameter, gradient in self.gradients:
                parameter.grad = gradient
                gradient.zero_()
            self.ids.copy_(ids)
            self.targets.copy_(targets)
            for key, value in self.learner.state.state_dict().items():
                destination = getattr(self.initial, key)
                if key == 'ring':
                    for dst, src in zip(destination, value):
                        dst.copy_(src)
                else:
                    destination.copy_(value)
        self.graph.replay()
        return self.output
