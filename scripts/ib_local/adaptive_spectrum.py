"""Calibrate a log-spaced spectrum on the actual carried phase-space state.

No world events, optimizer updates, random draws, or replacement of S/P occur.
The sole operating resolution is an initial eight-sample period envelope;
step doubling admits it, then the actual spectrum leaves a factor-two margin.
"""
from __future__ import annotations

import math
import torch

from scripts.ib_local.adaptive_hamiltonian import coefficients, invariants, potential, quiet
from scripts.ib_local.bfss_gdn_reference import rotation_charge


def energy(s, p, theta, context, scale, impedance):
    omega, _ = coefficients(theta, context, s.shape[-3], scale)
    return ((omega/impedance*p.square()+omega*impedance*s.square()).sum((-3,-2,-1))/2
            + potential(s, theta, s.shape[-3]))


@torch.no_grad()
def admit(engine, *, tolerance=1e-4, probe_steps=8, max_halvings=8):
    module = engine.model.head_coupling
    if int(module.spectrum_revision):
        raise ValueError('spectrum already installed; do not reinitialize learned clocks')
    if tolerance <= 0 or probe_steps < 1:
        raise ValueError('positive solver tolerance and numerical probe length required')
    first = engine.model.body.base.layers[0]
    # Work on copies of the actual body, with the exact production potential.
    s = engine.recurrent[..., :first.matrix_size].reshape(
        module.layers, 1, module.heads, module.dk, module.dv).detach().cpu().double()
    p = torch.stack(engine.bfss_momenta).detach().cpu().double()
    theta = module.theta.detach().cpu().double()
    context = (module.context.detach().cpu().double() if bool(module.context_valid)
               else invariants(s)[0].mean(-1)[...,None,None])
    base_omega, _ = coefficients(theta, context, module.heads)
    impedance = base_omega*module.frequency_scale.detach().cpu().double()
    # Every layer covers the same octaves; permute their head positions across
    # depth instead of assigning the same head number a permanent fast role.
    order = ((torch.arange(module.heads)[None]+torch.arange(module.layers)[:,None])
             % module.heads).double()[:,None,:,None,None]
    dt = engine.bfss_dt
    capacity = min(layer.content_budget for layer in engine.model.body.base.layers)
    attempts = []
    for halving in range(max_halvings):
        envelope = math.pi/4/(2**halving)
        envelope_omega = envelope/dt*torch.exp2(-order)
        scale = envelope_omega/base_omega
        h0 = energy(s,p,theta,context,scale,impedance)
        full_s,full_p,fine_s,fine_p = s,p,s,p
        max_capacity = 0.
        for _ in range(probe_steps):
            full_s,full_p = quiet(full_s,full_p,theta,context,dt,scale,impedance)
            for _ in range(2):
                fine_s,fine_p = quiet(fine_s,fine_p,theta,context,dt/2,scale,impedance)
            max_capacity = max(max_capacity, float(full_s.abs().sum(-1).max()),
                               float(full_s.abs().sum(-2).max()))
        norm = lambda x: x.flatten(1).norm(dim=1)
        state_error = norm(full_s-fine_s)/norm(fine_s).clamp_min(1e-12)
        momentum_error = norm(full_p-fine_p)/norm(fine_p).clamp_min(1e-12)
        energy_error = ((energy(full_s,full_p,theta,context,scale,impedance)-h0).abs()
                        /h0.abs().clamp_min(1e-12))
        charge_error = max(float((a-b).abs().max()) for a,b in zip(
            rotation_charge(s,p), rotation_charge(full_s,full_p)))
        worst = max(float(state_error.max()), float(momentum_error.max()), float(energy_error.max()))
        attempt = dict(envelope_angle=envelope, state_error=float(state_error.max()),
            momentum_error=float(momentum_error.max()), energy_error=float(energy_error.max()),
            charge_error=charge_error, capacity_max=max_capacity)
        attempts.append(attempt)
        if not math.isfinite(worst) or worst > tolerance or charge_error > 1e-9 or (capacity and max_capacity > capacity):
            continue
        # Actual operation starts at half the admitted envelope, without
        # amplifying P/omega. Existing coordinates and optimizer moments stay.
        module.frequency_scale.copy_((scale/2).to(module.frequency_scale))
        module.impedance.copy_(impedance.to(module.impedance))
        module.spectrum_revision.fill_(1)
        module.solver_phase_limit.fill_(envelope)
        module.phase_limit = envelope
        module.spectrum_enabled = True
        return dict(schema=1, retained='same S/P, parameters, Adam, traces, world, evidence and pending credit',
            geometry='positive head Hamiltonian: omega/(2Z) P² + omega Z/2 S² + same coupled potential',
            initialization='dyadic angular-frequency grid per layer, head order rotated with depth; no random draws',
            frequency=(envelope_omega/2).flatten(1).tolist(),
            impedance=impedance.flatten(1).tolist(), solver_phase_limit=envelope,
            probe_steps=probe_steps, tolerance=tolerance, attempts=attempts,
            momentum_receipt_bytes=(module.theta.shape[-1]*module.layers*module.heads*module.dv*module.theta.element_size()),
            solver_scope='phase-resolution subdivision; not an unconditional nonlinear stability theorem')
    raise RuntimeError(f'no spectrum passed actual-state numerical admission: {attempts}')
