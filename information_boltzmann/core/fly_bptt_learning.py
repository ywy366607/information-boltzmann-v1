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
    if model.topographic_writer is not None:
        drive, baseline = model.topographic_writer.forward_with_state(
            model.embedding(token), state.h, state.baseline)
    else:
        # Non-topographic injection modes drive inside model.step through
        # their own input projection; the writer baseline does not apply.
        drive, baseline = None, state.baseline
    options = dict(base_rates=base_rates, thresholds=thresholds,
                   conductance_gains=conductance_gains,
                   alif_params=alif_params, stp_params=stp_params)

    def tick(current, source, next_baseline):
        ret = model.step(
            current.h, token, spike_ring=current.ring, ge=current.ge,
            gi=current.gi, b=current.b, x=current.x, u=current.u,
            sensory_drive=source, **options)
        # Flexible arity: [h, spike, ring, ge, gi] plus optional ALIF/STP
        # states depending on the model's flags.
        h, ring, ge, gi = ret[0], ret[2], ret[3], ret[4]
        b = ret[5] if len(ret) > 5 and ret[5] is not None else current.b
        x = ret[6] if len(ret) > 6 and ret[6] is not None else current.x
        u = ret[7] if len(ret) > 7 and ret[7] is not None else current.u
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
        # Promote existing learned edge buffers without altering topology or
        # values.  Layout-generic: COBA models carry edge_weight_e/i, other
        # models a single signed edge_weight.
        for name in tuple(model._buffers):
            if 'edge_weight' in name and model._buffers[name] is not None:
                value = model._buffers[name]
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
        # The topographic writer exists only for topographic injection; other
        # injection modes have no sensory projection group to train.
        self.projections = ([getattr(model.topographic_writer, name).weight
                             for name in ('proj_vis', 'proj_chemo', 'proj_mech')]
                            if model.topographic_writer is not None else [])
        # Layout-generic edge collection: COBA models split excitatory and
        # inhibitory synapses into separate tensors, other models keep a
        # single signed tensor; the signed-aware clamp handles both.
        self.edges = [parameter for name, parameter in model.named_parameters()
                      if 'edge_weight' in name]
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
        # Level-2: dopamine-gated three-factor local plasticity on the
        # incoming edges of DAN-innervated targets.  The gate is the DAN
        # drive computed from the terminal membrane each window; the rule is
        # applied under no_grad outside the BPTT graph, like the clamp.
        self.dan_plastic_lr = float(getattr(model, 'dan_plastic_lr', 0.0))
        self._dan_setup()
        self.dan_gate = None

    def _dan_setup(self):
        import numpy as np
        model = self.model
        if not getattr(model, 'has_dopamine', False) or self.dan_plastic_lr <= 0.0:
            self.dan_eligible = None
            return
        dan_post = model.dan_edge_post.detach().cpu().numpy()
        gated_targets = np.unique(dan_post)
        gated_set = np.zeros(model.n_neurons, dtype=bool)
        gated_set[gated_targets] = True
        self.dan_gate_target_index = torch.as_tensor(gated_targets, dtype=torch.long,
                                                     device=self.state.h.device)
        edge_sets = []
        if hasattr(model, 'edge_post_e'):
            edge_sets.append(('edge_weight_e', model.edge_post_e, model.edge_pre_e, (0.0, 5.0)))
        if hasattr(model, 'edge_post_i'):
            edge_sets.append(('edge_weight_i', model.edge_post_i, model.edge_pre_i, (-5.0, 0.0)))
        if hasattr(model, 'edge_post'):
            edge_sets.append(('edge_weight', model.edge_post, model.edge_pre, (-5.0, 5.0)))
        eligible = {}
        for weight_name, post_tensor, pre_tensor, clamp in edge_sets:
            post_np = post_tensor.detach().cpu().numpy()
            mask = gated_set[post_np]
            idx = torch.as_tensor(np.flatnonzero(mask), dtype=torch.long,
                                  device=self.state.h.device)
            if idx.numel():
                eligible[weight_name] = (idx, pre_tensor, post_tensor, clamp)
        self.dan_eligible = eligible or None

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
                # Sign-preserving bounds: COBA split tensors are single-signed,
                # a single signed tensor keeps both signs per element.
                positive = weight >= 0
                weight.copy_(torch.where(positive, weight.clamp(0.0, 5.0),
                                         weight.clamp(-5.0, 0.0)))
            self.model.topographic_writer.a_adapt.copy_(next_state.baseline)
            self._dan_update(next_state)
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

    def _dan_update(self, terminal_state):
        """Dopamine-gated three-factor local plasticity, once per window.

        gate_i = the DAN drive received by target i (a sparse matvec of the
        terminal membrane through the DAN edges); the eligible synapses'
        update is eta * gate * pre_membrane * post_membrane, clamped to the
        sign class of their tensor.  No gradient flows through this path.
        """
        if self.dan_eligible is None:
            return {}
        h = terminal_state.h[0]
        dan_drive = torch.zeros_like(h)
        dan_drive.index_add_(0, self.model.dan_edge_post[0],
                             self.model.dan_edge_weight[0] * h[self.model.dan_edge_pre[0]])
        applied = {}
        for name, (idx, pre_tensor, post_tensor, clamp) in self.dan_eligible.items():
            weight = getattr(self.model, name)
            gate = dan_drive[post_tensor[idx]]
            pre_act = h[0][pre_tensor[idx]]
            post_act = h[0][post_tensor[idx]]
            delta = self.dan_plastic_lr * gate * pre_act * post_act
            with torch.no_grad():
                updated = (weight[idx] + delta).clamp(*clamp)
                weight.index_put_((idx,), updated)
            applied[name] = float(delta.abs().sum())
        self.dan_gate = dan_drive.detach()
        return applied

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
