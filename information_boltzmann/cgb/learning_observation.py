"""Read existing local learning and port evidence without modifying dynamics."""
from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np


def port_bit_logits(model, amplitude=3., voltage=None):
    """Canonical two-voltage observation convention, evaluated before arrival.

    This defines a diagnostic likelihood, not a theorem about text sources.
    No trained head or target enters its construction.
    """
    if len(model.port_nodes) != 8 or not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError('Eight byte ports and positive finite source amplitude required')
    brain_c = model.capacitance[model.port_nodes]
    effective_c = 1/(1/brain_c+1/model.world_capacitance)
    v = model.charge[model.port_nodes]/brain_c if voltage is None else np.asarray(voltage, float)
    if v.shape != (8,) or not np.isfinite(v).all():
        raise ValueError('One finite pre-arrival voltage per bit required')
    return 2*effective_c*amplitude*v


def byte_nll(logits, byte):
    logits = np.asarray(logits, float)
    if logits.shape != (8,) or not np.isfinite(logits).all() or not isinstance(byte, (int, np.integer)) or not 0 <= byte < 256:
        raise ValueError('Finite bit logits and one byte required')
    bits = (int(byte) >> np.arange(8))&1
    return float(np.sum(np.logaddexp(0., logits)-bits*logits))


def native_learning_drift(model):
    """Exact reaction-generator drift; observation only, no reward or update."""
    degree = model.degrees()
    events = model.events(degree=degree)
    groups = {}
    totals = {name: dict(positive=0., negative=0., net=0.)
              for name in ('support', 'trace', 'bridges', 'conductance', 'charged')}
    heat_rate = 0.
    for event in events:
        rate = model.event_rate(event, degree)
        sign, kind = event.sign, event.kind
        dc = sign if kind == 'binding' else -sign if kind == 'transport' else 0
        dr = sign if kind in ('write','thermal') else 0
        dn = sign if kind == 'assembly' else 0
        dg = 0.
        if dn or dr:
            dg = model.contact_conductance_increment(event)
        dp = sum(change for name, node, change in model._pools(event) if name=='charged')
        deltas = dict(support=dc, trace=dr, bridges=dn, conductance=dg, charged=dp)
        bucket = groups.setdefault(kind, dict(rate=0., support_drift=0., trace_drift=0., conductance_drift=0.))
        bucket['rate'] += rate
        bucket['support_drift'] += rate*dc
        bucket['trace_drift'] += rate*dr
        bucket['conductance_drift'] += rate*dg
        for name, delta in deltas.items():
            contribution = rate*delta
            totals[name]['positive' if contribution >= 0 else 'negative'] += contribution
            totals[name]['net'] += contribution
        heat_rate -= rate*model.event_delta(event,degree)[0]
    result = dict(time_ms=model.t, totals=totals, by_reaction=groups,
                  conditional_reaction_heat_out_rate=heat_rate,
                  scope='Original bridge/pool reaction subgenerator conditional on full current state; not predictive utility')
    if hasattr(model, 'domain_generator_rates'):
        result['domains'] = model.domain_generator_rates()
        result['scope'] += '; domain birth has a continuous spatial mark: reported separately as exact recovery rates and birth bounds'
    return result


def parameter_quench(model, **changes):
    """Explicit energetic/ideal-barrier diagnostic, never a hidden state reset.

    Only parameters whose U/G dependence is already explicit are allowed.
    Kinetic exclusions are ideal measurement interventions, not new laws of
    self-maintenance or claims that their controller has been modeled.
    """
    energetic = {'capture_coupling','trace_energy','charged_energy','structure_energy',
                 'carried_energy','bound_energy','cohesion_energy','elastic_stiffness','rest_length'}
    kinetic = {'assembly_rate','binding_rate','transport_rate','release_rate','write_rate',
               'recovery_rate','conversion_rate','active_rate','mobility',
               'material_diffusion','support_diffusion','drive_diffusion',
               'gate_low','gate_high'}
    if not set(changes) <= energetic|kinetic:
        raise ValueError('Unaccounted parameter intervention')
    old = model.params
    u0,g0 = model.energy(),model.free_energy()
    candidate = replace(old,**changes)
    model.params = candidate
    try:
        model._validate()
        du,dg = model.energy()-u0,model.free_energy()-g0
        if abs(du-dg) > 1e-9:
            raise ValueError('Intervention changes unaccounted configurational entropy')
    except Exception:
        model.params = old
        raise
    model._add_ledger('work_in', du)
    model._add_ledger('entropy_balance', du-dg)
    return dict(time_ms=model.t,changes=changes,previous={k:asdict(old)[k] for k in changes},
                external_work=du,free_energy_change=dg,state_erased=False,
                kinetic_controller='ideal diagnostic barrier; not part of the living model')
