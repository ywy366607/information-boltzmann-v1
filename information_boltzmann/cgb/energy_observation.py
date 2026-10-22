"""Exact accounting instruments for the existing material energy function."""
from __future__ import annotations

import numpy as np


def material_energy_components(model):
    """Separate storage and interactions without changing any physical law.

    Units are the same declared kBT=1 reference as energy(). The residual
    detects unsupported new energy terms instead of silently omitting them.
    Configurational free-energy correction is not a measured Shannon entropy.
    """
    p = model.params
    port_n = getattr(model, 'port_installed', np.zeros(0, dtype=np.int64))
    port_c = getattr(model, 'port_bound', np.zeros(0, dtype=np.int64))
    port_r = getattr(model, 'port_trace', np.zeros(0, dtype=np.int64))
    degree = model.degrees()
    terms = dict(
        brain_electrical=.5*float(np.sum(model.charge**2/model.capacitance)),
        world_electrical=.5*float(np.sum(model.world_charge**2/model.world_capacitance)),
        finite_drive=p.charged_energy*float(model.charged.sum()),
        carried=p.carried_energy*float(model.carried.sum()),
        domains=getattr(model, 'domain_formation_energy', 0.)*model.n,
        bridges=p.structure_energy*float(len(model.pre)+port_n.sum()),
        elastic=.5*p.elastic_stiffness*float(np.sum((model.lengths()-p.rest_length)**2)),
        cohesion=-.5*p.cohesion_energy*float(np.dot(degree, degree-1)),
        bound_support=p.bound_energy*float(model.bound.sum()+port_c.sum()),
        trace=p.trace_energy*float(model.trace.sum()+port_r.sum()),
        capture=-p.capture_coupling*float(np.dot(model.bound, model.trace)+np.dot(port_c, port_r)),
        identity=(model.recognition_energy_fraction*float(np.sum(model.recognition_free_energy(model.pre, model.post)))
                  if hasattr(model, 'recognition_free_energy') else 0.))
    total = sum(terms.values())
    residual = float(model.energy()-total)
    if not np.isfinite(total) or abs(residual) > 1e-9:
        raise ValueError('Energy decomposition does not cover the current material law')
    return dict(terms=terms, total=float(total), decomposition_residual=residual,
                configurational_free_correction=float(model.free_energy()-model.energy()),
                scope='Existing point-state energy and standard configurational correction; not a predictive score or Shannon entropy measurement')
