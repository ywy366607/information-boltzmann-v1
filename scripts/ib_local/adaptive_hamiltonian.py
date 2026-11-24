"""Batched invariant head potential and bounded, exact quiet-parameter traces.

Only the H+4 local dynamics coordinates have forward sensitivities. This is
not a Jacobian of the language body's weights or of its lifetime history.
Context is fixed between actual observation boundaries. Attention varies with
S through orthogonal-basis invariants; force includes that dependence.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def invariants(s):
    norm = s.square().sum((-2, -1))
    flat = s.flatten(-2)
    gram = flat @ flat.transpose(-1, -2)
    correlation = gram/torch.sqrt((1+norm.unsqueeze(-1))*(1+norm.unsqueeze(-2)))
    magnitude = torch.log1p(norm)
    return correlation, magnitude


def coefficients(theta, context, heads, frequency_scale=None):
    omega = F.softplus(theta[:, None, :heads, None, None]+
                      theta[:, None, heads:heads+1, None, None]*context)
    if frequency_scale is not None:
        omega = omega*frequency_scale
    strength = F.softplus(theta[:, heads+1])
    return omega, strength


def potential(s, theta, heads):
    corr, mag = invariants(s)
    logits = (theta[:, None, None, None, heads+2]*corr+
              theta[:, None, None, None, heads+3]*(mag.unsqueeze(-1)+mag.unsqueeze(-2)))
    weights = 2*logits.sigmoid()
    si, sj = s.unsqueeze(-3), s.unsqueeze(-4)
    a = si @ sj.transpose(-1, -2)-sj @ si.transpose(-1, -2)
    b = si.transpose(-1, -2) @ sj-sj.transpose(-1, -2) @ si
    pair = a.square().sum((-2, -1))+b.square().sum((-2, -1))
    strength = F.softplus(theta[:, heads+1])
    return strength[:, None]*(weights*pair).sum((-2, -1))/16


def force_analytic(s, theta, heads):
    """Exact analytical gradient d(potential.sum())/ds over the H(H-1)/2 unordered pairs.

    Matches `torch.func.grad(lambda x: potential(x, theta, heads).sum())` to FP32
    machine precision (~1e-7 relative error) while avoiding the H×H broadcast and
    nested autograd tape, and remains JVP/vmap-differentiable.
    """
    idx_i, idx_j = torch.triu_indices(heads, heads, offset=1, device=s.device)
    si = s[:, :, idx_i]
    sj = s[:, :, idx_j]
    ni = si.square().sum((-2, -1))
    nj = sj.square().sum((-2, -1))
    inv_d = torch.rsqrt(1.0 + ni) * torch.rsqrt(1.0 + nj)
    corr = (si * sj).sum((-2, -1)) * inv_d
    t2 = theta[:, None, None, heads + 2]
    t3 = theta[:, None, None, heads + 3]
    sig = (t2 * corr + t3 * (ni.log1p() + nj.log1p())).sigmoid()
    w = 2.0 * sig
    a = si @ sj.transpose(-1, -2) - sj @ si.transpose(-1, -2)
    b = si.transpose(-1, -2) @ sj - sj.transpose(-1, -2) @ si
    pair = a.square().sum((-2, -1)) + b.square().sum((-2, -1))
    strength = F.softplus(theta[:, heads + 1])[:, None, None]
    c_pair = (strength * (0.5 * w))[..., None, None]
    gi = c_pair * (a @ sj - sj @ b)
    gj = c_pair * (si @ b - a @ si)
    c_log = strength * (0.25 * sig * (1.0 - sig) * pair)
    c_cross = (c_log * t2 * inv_d)[..., None, None]
    c_si = (c_log * (2.0 * t3 - t2 * corr) / (1.0 + ni))[..., None, None]
    c_sj = (c_log * (2.0 * t3 - t2 * corr) / (1.0 + nj))[..., None, None]
    gi = gi + c_cross * sj + c_si * si
    gj = gj + c_cross * si + c_sj * sj
    grad = torch.zeros_like(s)
    grad = grad.index_add(2, idx_i, gi)
    grad = grad.index_add(2, idx_j, gj)
    return grad


def quiet(s, p, theta, context, dt, frequency_scale=None, impedance=None):
    heads = s.shape[-3]
    half = p-dt/2*force_analytic(s, theta, heads)
    omega, _ = coefficients(theta, context, heads, frequency_scale)
    angle = omega*dt
    c, sine = angle.cos(), angle.sin()
    if impedance is None:
        next_s = c*s+dt*torch.sinc(angle/math.pi)*half
        next_p = c*half-omega*sine*s
    else:
        # H_h = (omega / Z * P² + omega * Z * S²)/2.
        # Z fixes the phase ellipse; omega changes its traversal speed.
        next_s = c*s+sine/impedance*half
        next_p = c*half-impedance*sine*s
    return next_s, next_p-dt/2*force_analytic(next_s, theta, heads)


class StateBridge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, theta, sensitivity, heads, dk, dv):
        ctx.save_for_backward(sensitivity)
        ctx.shape = (heads, dk, dv)
        return state.clone()

    @staticmethod
    def backward(ctx, grad):
        sensitivity, = ctx.saved_tensors
        h, k, v = ctx.shape
        cotangent = grad[..., :h*k*v].reshape(grad.shape[0], grad.shape[1], h, k, v)
        parameter = torch.einsum('lbhkv,plbhkv->lp', cotangent, sensitivity)
        return grad, parameter, None, None, None, None


class MomentumReadBridge(torch.autograd.Function):
    """An actual historical P-read eligibility, paid by its later true error."""
    @staticmethod
    def forward(ctx, value, theta, sensitivity, module):
        ctx.save_for_backward(sensitivity)
        ctx.module = module
        return value.clone()

    @staticmethod
    def backward(ctx, grad):
        sensitivity, = ctx.saved_tensors  # [local parameter, layer, read width]
        cotangent = grad.reshape(-1, sensitivity.shape[1], sensitivity.shape[2]).sum(0)
        parameter = torch.einsum('ld,pld->lp', cotangent, sensitivity)
        with torch.no_grad():
            ctx.module.last_momentum_gradient.copy_(parameter)
            ctx.module.momentum_credit_count.add_(1)
        return grad, parameter, None, None


class AdaptiveHamiltonian(nn.Module):
    def __init__(self, layers, heads, dk, dv, *, frequency=.8, coupling=.005,
                 device=None, split_theta=False):
        super().__init__()
        self.layers, self.heads, self.dk, self.dv = layers, heads, dk, dv
        self._split_theta = bool(split_theta)
        width = heads+4
        raw = torch.zeros(layers, width, device=device)
        raw[:, :heads] = math.log(math.expm1(frequency))
        raw[:, heads+1] = math.log(math.expm1(coupling))
        if self._split_theta:
            self.theta_freq = nn.Parameter(raw[:, :heads].clone())
            self.theta_context = nn.Parameter(raw[:, heads:heads+1].clone())
            self.theta_coupling = nn.Parameter(raw[:, heads+1:heads+2].clone())
            self.theta_pair = nn.Parameter(raw[:, heads+2:heads+4].clone())
        else:
            self.theta = nn.Parameter(raw)
        self.register_buffer('context', torch.zeros(layers, 1, heads, 1, 1, device=device))
        self.register_buffer('context_valid', torch.tensor(False, device=device))
        self.register_buffer('trace_steps', torch.zeros((), dtype=torch.long, device=device))
        self.register_buffer('last_gradient', torch.zeros_like(raw))
        self.register_buffer('last_consumed_steps', torch.zeros((), dtype=torch.long, device=device))
        self.register_buffer('frequency_scale', torch.ones(layers, 1, heads, 1, 1, device=device))
        self.register_buffer('impedance', torch.ones(layers, 1, heads, 1, 1, device=device))
        self.register_buffer('spectrum_revision', torch.zeros((), dtype=torch.long, device=device))
        self.register_buffer('solver_phase_limit', torch.zeros((), device=device))
        self.register_buffer('last_solver_substeps', torch.ones((), dtype=torch.long, device=device))
        self.register_buffer('last_momentum_gradient', torch.zeros_like(raw))
        self.register_buffer('momentum_credit_count', torch.zeros((), dtype=torch.long, device=device))
        self.spectrum_enabled = False
        self.phase_limit = math.inf
        shape = (width, layers, 1, heads, dk, dv)
        self.register_buffer('sensitivity_s', torch.zeros(shape, device=device))
        self.register_buffer('sensitivity_p', torch.zeros(shape, device=device))

    def __getattr__(self, name):
        if name == 'theta' and self.__dict__.get('_split_theta', False):
            return torch.cat((self.theta_freq, self.theta_context,
                              self.theta_coupling, self.theta_pair), dim=-1)
        return super().__getattr__(name)

    def theta_parts(self):
        if self._split_theta:
            return (self.theta_freq, self.theta_context, self.theta_coupling, self.theta_pair)
        return (self.theta,)

    def theta_part_names(self):
        if self._split_theta:
            return ('freq', 'context', 'coupling', 'pair')
        return ('theta',)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Old complete checkpoints keep their original dynamics and every
        # pending participation; only newly introduced buffers get defaults.
        for name in ('frequency_scale', 'impedance', 'spectrum_revision', 'solver_phase_limit', 'last_solver_substeps',
                     'last_momentum_gradient', 'momentum_credit_count'):
            state_dict.setdefault(prefix+name, getattr(self, name).detach().clone())
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self.spectrum_enabled = bool(int(self.spectrum_revision))
        self.phase_limit = float(self.solver_phase_limit) if self.spectrum_enabled else math.inf

    def quiet(self, base, recurrent, momenta, *, dt, frequency=None, steps=1, capture=False):
        first = base.layers[0]
        if len(base.layers) != self.layers or any((l.heads, l.dk, l.dv) !=
                (self.heads, self.dk, self.dv) for l in base.layers):
            raise ValueError('adaptive quiet requires the declared uniform layer layout')
        s = recurrent[..., :first.matrix_size].reshape(self.layers, 1, self.heads, self.dk, self.dv)
        p = torch.stack(momenta)
        if not bool(self.context_valid):
            corr, _ = invariants(s.detach())
            self.context.copy_(corr.mean(-1)[..., None, None])
            self.context_valid.fill_(True)
        theta, context = self.theta.detach(), self.context.detach()
        omega, _ = coefficients(theta, context, self.heads, self.frequency_scale)
        subdivisions = max(1, math.ceil(abs(dt)*float(omega.max())/self.phase_limit))
        if subdivisions > 1024:
            raise FloatingPointError('learned frequency exceeds the finite solver budget')
        self.last_solver_substeps.fill_(subdivisions)
        sub_dt = dt/subdivisions
        for _ in range(steps*subdivisions):
            contact = getattr(self, 'contact', None)
            if capture and contact is not None:
                contact_fn = lambda x, y: quiet(x, y, theta, context, sub_dt,
                    self.frequency_scale, self.impedance if self.spectrum_enabled else None)
                _, contact_tangents = torch.vmap(lambda ds, dp: torch.func.jvp(
                    contact_fn, (s, p), (ds, dp)))(contact.sensitivity_s, contact.sensitivity_p)
                contact.sensitivity_s.copy_(contact_tangents[0].detach())
                contact.sensitivity_p.copy_(contact_tangents[1].detach())
            if capture:
                basis = torch.eye(theta.shape[-1], device=theta.device, dtype=theta.dtype)
                directions = basis[:, None, :].expand(-1, self.layers, -1)
                fn = lambda x, y, t: quiet(x, y, t, context, sub_dt, self.frequency_scale,
                    self.impedance if self.spectrum_enabled else None)
                def one(ds, dp, dtheta):
                    return torch.func.jvp(fn, (s, p, theta), (ds, dp, dtheta))
                next_s = next_p = None
                # Block independent local parameter directions, never layers,
                # heads or observed tokens. Same derivatives, bounded scratch.
                for start in range(0, theta.shape[-1], 4):
                    end = start+4
                    values, tangents = torch.vmap(one)(self.sensitivity_s[start:end],
                        self.sensitivity_p[start:end], directions[start:end])
                    if next_s is None:
                        next_s, next_p = values[0][0].detach().clone(), values[1][0].detach().clone()
                    self.sensitivity_s[start:end].copy_(tangents[0].detach())
                    self.sensitivity_p[start:end].copy_(tangents[1].detach())
                    del values, tangents
                s, p = next_s, next_p
            else:
                s, p = quiet(s, p, theta, context, sub_dt, self.frequency_scale,
                    self.impedance if self.spectrum_enabled else None)
                s, p = s.detach(), p.detach()
        if capture:
            self.trace_steps.add_(steps)
        result = recurrent.clone()
        result[..., :first.matrix_size] = s.flatten(2)
        return result, list(p.unbind(0))

    def bridge(self, state):
        if not torch.is_grad_enabled():
            return state
        if int(self.trace_steps):
            state = StateBridge.apply(state, self.theta, self.sensitivity_s,
                                      self.heads, self.dk, self.dv)
        contact = getattr(self, 'contact', None)
        if contact is not None:
            # Observation capture mutates the live trace before backward.
            state = StateBridge.apply(state, contact.phi, contact.sensitivity_s.clone(),
                                      self.heads, self.dk, self.dv)
        return state

    @torch.no_grad()
    def capture_momentum_read(self, queries):
        contact = getattr(self, 'contact', None)
        if not int(self.trace_steps) and contact is None:
            return None
        # Actual decision-time queries, held fixed in this local historical
        # partial derivative. No copy of the full state or language weights.
        query = torch.stack(queries)  # [layer, batch=1, head, key]
        if query.shape[1] != 1:
            raise ValueError('one continuous individual is required')
        original = (torch.einsum('plbhkv,lbhk->plhv', self.sensitivity_p, query).flatten(2).detach().clone()
                    if int(self.trace_steps) else None)
        if contact is None:
            return original
        gate = torch.einsum('plbhkv,lbhk->plhv', contact.sensitivity_p, query).flatten(2).detach().clone()
        return dict(quiet=original, contact=gate, schema=1)

    def bridge_momentum_read(self, value, sensitivity):
        if sensitivity is None or not torch.is_grad_enabled():
            return value
        if isinstance(sensitivity, dict):
            if sensitivity['schema'] != 1:
                raise ValueError('unknown contact read eligibility schema')
            if sensitivity['quiet'] is not None:
                value = MomentumReadBridge.apply(value, self.theta, sensitivity['quiet'], self)
            contact = self.contact
            return MomentumReadBridge.apply(value, contact.phi, sensitivity['contact'], contact)
        return MomentumReadBridge.apply(value, self.theta, sensitivity, self)

    @torch.no_grad()
    def consume(self):
        self.last_consumed_steps.copy_(self.trace_steps)
        parts = self.theta_parts()
        if any(p.grad is not None for p in parts):
            grad = (parts[0].grad if len(parts) == 1 else
                    torch.cat([torch.zeros_like(p) if p.grad is None else p.grad for p in parts], dim=-1))
            self.last_gradient.copy_(grad)
        self.trace_steps.zero_()
        self.sensitivity_s.zero_()
        self.sensitivity_p.zero_()
        self.context_valid.fill_(False)

    @torch.no_grad()
    def diagnostics(self):
        omega, strength = coefficients(self.theta, self.context, self.heads, self.frequency_scale)
        return dict(contact=(self.contact.diagnostics() if hasattr(self, 'contact') else None),
            frequency=omega.flatten(1).cpu().tolist(),
            coupling=strength.cpu().tolist(), context_gain=self.theta[:, self.heads].cpu().tolist(),
            content_coefficients=self.theta[:, -2:].cpu().tolist(),
            per_layer_gradient_norm=self.last_gradient.norm(dim=-1).cpu().tolist(),
            momentum_credit_count=int(self.momentum_credit_count),
            per_layer_momentum_credit_norm=self.last_momentum_gradient.norm(dim=-1).cpu().tolist(),
            spectrum_revision=int(self.spectrum_revision),
            solver_substeps=int(self.last_solver_substeps),
            last_consumed_steps=int(self.last_consumed_steps),
            trace_steps=int(self.trace_steps),
            trace_bytes=(self.sensitivity_s.numel()+self.sensitivity_p.numel())*self.theta.element_size())


def bridge_state(model, state):
    module = getattr(model, 'head_coupling', None)
    return module.bridge(state) if isinstance(module, AdaptiveHamiltonian) else state


def configure(engine, *, initialize=True, split_theta=False):
    if hasattr(engine.model, 'head_coupling'):
        if not isinstance(engine.model.head_coupling, AdaptiveHamiltonian):
            raise ValueError('do not silently replace an existing learned head potential')
        return
    base = engine.model.body.base
    layer = base.layers[0]
    old_params = engine.params
    module = AdaptiveHamiltonian(len(base.layers), layer.heads, layer.dk, layer.dv,
        frequency=engine.bfss_frequency, coupling=engine.bfss_coupling,
        device=engine.recurrent.device, split_theta=split_theta)
    engine.model.head_coupling = module
    # A non-registered runtime reference avoids a duplicate state_dict subtree.
    object.__setattr__(base, '_adaptive_quiet', module)
    added = list(module.theta_parts())
    engine.optimizer.add_param_group(dict(params=added,
        lr=engine.optimizer.param_groups[0]['lr'], weight_decay=0.))
    engine.params = [*old_params, *added]
    engine.groups = (*engine.groups, added)
    if initialize:
        def extend(value):
            if isinstance(value, (list, tuple)) and len(value) == len(old_params) and all(
                    v is None or torch.is_tensor(v) for v in value):
                return type(value)([*value, *([None] * len(added))])
            if isinstance(value, dict):
                return {k: extend(v) for k, v in value.items()}
            return value
        for name in ('prev', 'internal_pending', 'fire_grad'):
            if hasattr(engine, name):
                setattr(engine, name, extend(getattr(engine, name)))
        if hasattr(engine, 'head_mask'):
            engine.head_mask = [*engine.head_mask, *([True] * len(added))]
    engine.adaptive_dynamics_schema = 1


def migrate_split_theta(engine):
    """Split a packed AdaptiveHamiltonian.theta (L, H+4) into 4 role-specific parameters.

    Preserves exact weights, Adam exp_avg/exp_avg_sq slices, and any pending
    parameter-aligned eligibility lists while giving each role its own local
    coherence observer in BudgetedAdamW.
    """
    module = getattr(engine.model, 'head_coupling', None)
    if not isinstance(module, AdaptiveHamiltonian):
        raise ValueError('split_theta migration requires AdaptiveHamiltonian')
    if module._split_theta:
        return
    h = module.heads
    slices = ((0, h), (h, h + 1), (h + 1, h + 2), (h + 2, h + 4))
    old_theta = module._parameters.pop('theta')
    raw = old_theta.detach()
    module._split_theta = True
    module.theta_freq = nn.Parameter(raw[:, 0:h].clone().contiguous())
    module.theta_context = nn.Parameter(raw[:, h:h + 1].clone().contiguous())
    module.theta_coupling = nn.Parameter(raw[:, h + 1:h + 2].clone().contiguous())
    module.theta_pair = nn.Parameter(raw[:, h + 2:h + 4].clone().contiguous())
    new_parts = list(module.theta_parts())
    # Replace in optimizer param_groups in-place at the exact group/position.
    replaced = False
    for group in engine.optimizer.param_groups:
        for i, p in enumerate(group['params']):
            if p is old_theta:
                group['params'][i:i + 1] = new_parts
                replaced = True
                break
        if replaced:
            break
    if not replaced:
        raise RuntimeError('AdaptiveHamiltonian.theta not found in optimizer.param_groups')
    # Migrate Adam m/v slices; leave budget observer unset so BudgetedAdamW
    # adopts each sliced exp_avg with its own fresh local coherence observer.
    old_state = engine.optimizer.state.pop(old_theta, None)
    if hasattr(engine.optimizer, '_host_steps'):
        engine.optimizer._host_steps.pop(old_theta, None)
    if old_state is not None:
        for part, (lo, hi) in zip(new_parts, slices):
            engine.optimizer.state[part] = dict(
                step=old_state['step'].clone(),
                exp_avg=old_state['exp_avg'][:, lo:hi].clone().contiguous(),
                exp_avg_sq=old_state['exp_avg_sq'][:, lo:hi].clone().contiguous(),
            )
    # Splice engine.params, engine.groups, head_mask, and pending gradient lists.
    old_count = len(engine.params)
    theta_idx = next((i for i, p in enumerate(engine.params) if p is old_theta), None)
    if theta_idx is None:
        raise RuntimeError('AdaptiveHamiltonian.theta not found in engine.params')
    engine.params = [*engine.params[:theta_idx], *new_parts, *engine.params[theta_idx + 1:]]
    new_groups = []
    for grp in engine.groups:
        if any(p is old_theta for p in grp):
            out = []
            for p in grp:
                if p is old_theta:
                    out.extend(new_parts)
                else:
                    out.append(p)
            new_groups.append(out)
        else:
            new_groups.append(grp)
    engine.groups = tuple(new_groups)

    def splice(value):
        if isinstance(value, (list, tuple)) and len(value) == old_count and all(
                v is None or torch.is_tensor(v) for v in value):
            item = value[theta_idx]
            if item is None:
                mid = [None] * len(new_parts)
            elif torch.is_tensor(item) and item.shape == raw.shape:
                mid = [item[:, lo:hi].clone().contiguous() for lo, hi in slices]
            else:
                raise ValueError(f'unexpected pending tensor shape for theta: {getattr(item, "shape", type(item))}')
            return type(value)([*value[:theta_idx], *mid, *value[theta_idx + 1:]])
        if isinstance(value, dict):
            return {k: splice(v) for k, v in value.items()}
        return value

    for name in ('prev', 'internal_pending', 'fire_grad'):
        if hasattr(engine, name):
            setattr(engine, name, splice(getattr(engine, name)))
    if hasattr(engine, 'head_mask') and len(engine.head_mask) == old_count:
        flag = engine.head_mask[theta_idx]
        engine.head_mask = [*engine.head_mask[:theta_idx], *([flag] * len(new_parts)),
                            *engine.head_mask[theta_idx + 1:]]
