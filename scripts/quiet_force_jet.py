"""Bounded force-jet reuse for the existing split Hamiltonian quiet step.

The potential is scalar. Its state Hessian is symmetric, so a reverse-mode
product supplies H_ss ds; a reverse product of dV/dtheta supplies H_sθ dθ.
Both reuse the same potential graph. No dense state Jacobian is formed.
"""
from __future__ import annotations

import math

import torch

from scripts.ib_local.adaptive_hamiltonian import coefficients, force_analytic, potential


def independent_pair_potential(s, theta, heads):
    """Original scalar energy, summed once per unordered head pair.

    Both commutators change sign on exchange; their squared norms and the
    content gate are symmetric. Diagonal commutators vanish. Thus 1/16 times
    the ordered sum is 1/8 times this upper-triangle sum, including derivatives.
    """
    i, j = torch.triu_indices(heads, heads, offset=1, device=s.device)
    si, sj = s[..., i, :, :], s[..., j, :, :]
    ni, nj = si.square().sum((-2, -1)), sj.square().sum((-2, -1))
    corr = (si*sj).sum((-2, -1))/torch.sqrt((1+ni)*(1+nj))
    logits = (theta[:, None, None, heads+2]*corr +
              theta[:, None, None, heads+3]*(ni.log1p()+nj.log1p()))
    a = si @ sj.transpose(-1, -2)-sj @ si.transpose(-1, -2)
    b = si.transpose(-1, -2) @ sj-sj.transpose(-1, -2) @ si
    pair = a.square().sum((-2, -1))+b.square().sum((-2, -1))
    return torch.nn.functional.softplus(theta[:, heads+1, None])*(
        2*logits.sigmoid()*pair).sum(-1)/8


class ForceJet:
    def __init__(self, s, theta, heads):
        with torch.enable_grad():
            self.s = s.detach().requires_grad_(True)
            self.theta = theta.detach().requires_grad_(True)
            value = independent_pair_potential(self.s, self.theta, heads).sum()
            self.force, self.theta_gradient = torch.autograd.grad(
                value, (self.s, self.theta), create_graph=True)

    def apply(self, ds, dtheta):
        with torch.enable_grad():
            result, = torch.autograd.grad((self.force, self.theta_gradient), self.s,
                (ds, dtheta), is_grads_batched=True, retain_graph=True)
        return result.detach()


class SplitJet:
    def __init__(self, s, p, theta, context, dt, frequency_scale=None, impedance=None):
        self.dt, self.impedance = dt, impedance
        heads = s.shape[-3]
        self.s, self.p = s.detach(), p.detach()
        self.theta, self.context = theta.detach(), context.detach()
        # Preview stores the states needed to build the differential later.
        # Rejected previews never construct or retain a Hessian graph.
        self.first = self.second = None
        self.half = p-dt/2*force_analytic(s, theta, heads).detach()
        self.omega, _ = coefficients(theta, context, heads, frequency_scale)
        self.frequency_scale = frequency_scale
        angle = self.omega*dt
        self.c, self.sine = angle.cos(), angle.sin()
        if impedance is None:
            self.f = dt*torch.sinc(angle/math.pi)
            # Stable derivative of sin(omega*dt)/omega near zero.
            safe = torch.where(angle.abs() < 1e-3, torch.ones_like(angle), angle)
            regular = dt**2*(safe*safe.cos()-safe.sin())/safe.square()
            series = dt**2*(-angle/3+angle.pow(3)/30-angle.pow(5)/840)
            self.df = torch.where(angle.abs() < 1e-3, series, regular)
            next_s = self.c*s+self.f*self.half
            self.rotated_p = self.c*self.half-self.omega*self.sine*s
        else:
            next_s = self.c*s+self.sine/impedance*self.half
            self.rotated_p = self.c*self.half-impedance*self.sine*s
        self.next_s = next_s.detach()
        self.next_p = (self.rotated_p-dt/2*force_analytic(next_s, theta, heads).detach()).detach()

    def apply(self, ds, dp, dtheta, block=4):
        heads = self.s.shape[-3]
        if self.first is None:
            self.first = ForceJet(self.s, self.theta, heads)
            self.second = ForceJet(self.next_s, self.theta, heads)
        outputs_s, outputs_p = [], []
        for start in range(0, len(ds), block):
            a, b, t = ds[start:start+block], dp[start:start+block], dtheta[start:start+block]
            half = b-self.dt/2*self.first.apply(a, t)
            z = self.theta[:, None, :heads, None, None]+self.theta[:, None, heads:heads+1, None, None]*self.context
            dz = t[:, :, None, :heads, None, None]+t[:, :, None, heads:heads+1, None, None]*self.context
            dw = z.sigmoid()*dz
            if self.frequency_scale is not None:
                dw = dw*self.frequency_scale
            c, sine, dt = self.c, self.sine, self.dt
            if self.impedance is None:
                next_a = c*a+self.f*half+dw*(-dt*sine*self.s+self.df*self.half)
                next_b = c*half-self.omega*sine*a+dw*(-dt*sine*self.half-(sine+self.omega*dt*c)*self.s)
            else:
                zimp = self.impedance
                next_a = c*a+sine/zimp*half+dw*dt*(-sine*self.s+c/zimp*self.half)
                next_b = c*half-zimp*sine*a+dw*dt*(-sine*self.half-zimp*c*self.s)
            next_b = next_b-dt/2*self.second.apply(next_a, t)
            outputs_s.append(next_a.detach()); outputs_p.append(next_b.detach())
        return torch.cat(outputs_s), torch.cat(outputs_p)

    def close(self):
        self.first = self.second = None


def _tensor_key(tensor):
    return (id(tensor), tensor.data_ptr(), tensor._version, tuple(tensor.shape),
            tensor.dtype, tensor.device)


class PreparedQuiet:
    """One-use physical proposal and its matching, lazy eligibility transport."""
    def __init__(self, module, recurrent, momenta, jet, result):
        self.module, self.recurrent, self.momenta = module, recurrent, tuple(momenta)
        self.jet, self.result = jet, result
        self._next_p = jet.next_p
        self.used = False
        self.tensors = (recurrent, *momenta, *module.theta_parts(), module.context,
            module.context_valid, module.frequency_scale, module.impedance,
            module.sensitivity_s, module.sensitivity_p, module.trace_steps)
        contact = getattr(module, 'contact', None)
        if contact is not None:
            self.tensors += (contact.phi, contact.sensitivity_s, contact.sensitivity_p)
        self.keys = tuple(_tensor_key(t) for t in self.tensors)
        # Tiny parameter/config copies also detect fused writes that bypass
        # Tensor._version. State/eligibility are protected by the same-call scope.
        self.parameters = (*module.theta_parts(), module.context, module.frequency_scale, module.impedance)
        if contact is not None:
            self.parameters += (contact.phi,)
        self.values = tuple(t.detach().clone() for t in self.parameters)

    def valid(self, recurrent, momenta):
        m = self.module
        current = (recurrent, *momenta, *m.theta_parts(), m.context, m.context_valid,
                   m.frequency_scale, m.impedance, m.sensitivity_s,
                   m.sensitivity_p, m.trace_steps)
        if getattr(m, 'contact', None) is not None:
            current += (m.contact.phi, m.contact.sensitivity_s, m.contact.sensitivity_p)
        return (not self.used and recurrent is self.recurrent and
                len(momenta) == len(self.momenta) and
                all(a is b for a, b in zip(momenta, self.momenta)) and
                self.keys == tuple(_tensor_key(t) for t in current) and
                all(torch.equal(a, b) for a, b in zip(self.parameters, self.values)))

    @torch.no_grad()
    def commit(self, recurrent, momenta):
        if not self.valid(recurrent, momenta):
            raise RuntimeError('quiet proposal no longer matches its state/parameter/eligibility scope')
        m = self.module
        ds, dp = m.sensitivity_s, m.sensitivity_p
        n = len(ds)
        contact = getattr(m, 'contact', None)
        if contact is not None:
            ds = torch.cat((ds, contact.sensitivity_s))
            dp = torch.cat((dp, contact.sensitivity_p))
        directions = ds.new_zeros(len(ds), *m.theta.shape)
        directions[:n] = torch.eye(n, device=ds.device, dtype=ds.dtype)[:, None]
        es, ep = self.jet.apply(ds, dp, directions)
        m.sensitivity_s.copy_(es[:n]); m.sensitivity_p.copy_(ep[:n])
        if contact is not None:
            contact.sensitivity_s.copy_(es[n:]); contact.sensitivity_p.copy_(ep[n:])
        m.trace_steps.add_(1)
        self.used = True
        result = self.result
        self.close()
        return result, list(self.next_p.unbind(0))

    @property
    def next_p(self):
        return self._next_p

    def close(self):
        self.used = True
        if self.jet is not None:
            self._next_p = self.jet.next_p
            self.jet.close()
            self.jet = None


@torch.no_grad()
def prepare(module, base, recurrent, momenta, *, dt):
    """Prepare a single solver substep; longer solver paths use the reference.

    The graph lifetime is one immediate internal-choice call. The scalar
    frequency/context and pure state advance are the original split map.
    """
    first = base.layers[0]
    if len(base.layers) != module.layers or any((l.heads, l.dk, l.dv) !=
            (module.heads, module.dk, module.dv) for l in base.layers):
        return None
    s = recurrent[..., :first.matrix_size].reshape(module.layers, 1,
        module.heads, module.dk, module.dv)
    p = torch.stack(momenta)
    if not bool(module.context_valid):
        from scripts.ib_local.adaptive_hamiltonian import invariants
        corr, _ = invariants(s)
        module.context.copy_(corr.mean(-1)[..., None, None])
        module.context_valid.fill_(True)
    theta, context = module.theta.detach(), module.context.detach()
    omega, _ = coefficients(theta, context, module.heads, module.frequency_scale)
    subdivisions = max(1, math.ceil(abs(dt)*float(omega.max())/module.phase_limit))
    if subdivisions != 1:
        return None
    module.last_solver_substeps.fill_(1)
    jet = SplitJet(s, p, theta, context, dt, module.frequency_scale,
                   module.impedance if module.spectrum_enabled else None)
    result = recurrent.clone()
    result[..., :first.matrix_size] = jet.next_s.flatten(2)
    return PreparedQuiet(module, recurrent, momenta, jet, result)


def prepare_engine(engine):
    """One immediate selector scope; GPU default, CPU opt-in for verification."""
    from scripts.ib_local.adaptive_hamiltonian import AdaptiveHamiltonian
    stale = getattr(engine, '_quiet_proposal', None)
    if stale is not None:
        stale.close()
        engine._quiet_proposal = None
    module = getattr(engine.model, 'head_coupling', None)
    if not isinstance(module, AdaptiveHamiltonian):
        return None
    if not getattr(module, '_prepared_quiet_enabled', engine.recurrent.is_cuda):
        return None
    from scripts.ib_local import run_navigation_residual_wood as legacy
    legacy._DERIVED.add('_quiet_proposal')
    return prepare(module, engine.model.body.base, engine.recurrent,
                   engine.bfss_momenta, dt=engine.bfss_dt)


def finish_choice(engine, proposal, selected):
    if proposal is None:
        return
    if selected:
        engine._quiet_proposal = proposal
    else:
        proposal.close()


def commit_engine(engine, ticks):
    proposal = getattr(engine, '_quiet_proposal', None)
    if proposal is None:
        return None
    engine._quiet_proposal = None
    if (ticks != 1 or proposal.module is not engine.model.head_coupling or
            proposal.jet.dt != engine.bfss_dt or
            not proposal.valid(engine.recurrent, engine.bfss_momenta)):
        proposal.close()
        return None
    return proposal.commit(engine.recurrent, engine.bfss_momenta)

