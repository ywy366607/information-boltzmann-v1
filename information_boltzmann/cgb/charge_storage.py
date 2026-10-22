"""A physical charge partition for the finite polarized storage carrier.

The existing charge arrays store net nodal charge. In this material, r=1
separates one bound charge pair across the existing directed terminals.
Q_net = Q_mobile + Q_bound; no extra state arrays or event history are needed.
Both fueled and thermal changes of this same configuration move its bound
charge. Active transport still moves mobile charge. The old scalar carrier
is retained as a distinct material constitution, not silently reinterpreted.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np


def storage_charge_partition(model):
    brain=np.zeros(model.n)
    world=np.zeros(len(model.port_nodes))
    if model.params.polarized_storage:
        z=model.params.charge_per_event
        np.add.at(brain,model.pre,-z*model.trace)
        np.add.at(brain,model.post,z*model.trace)
        if hasattr(model,'port_trace'):
            world-=z*model.port_trace
            np.add.at(brain,model.port_nodes,z*model.port_trace)
    return dict(bound_brain=brain,bound_world=world,
        mobile_brain=model.charge-brain,mobile_world=model.world_charge-world,
        scope='Net capacitor charge already includes this finite bound-charge configuration; no double energy term')


def select_storage_constitution(model, polarized):
    """Explicit counterfactual material choice, never an autonomous update.

    All saved net charges, inventories, memories and RNGs are preserved. The
    microscopic mobile/bound interpretation changes. This is a material-law
    branch; its manufacturing/controller cost is not a modeled living action.
    """
    if not isinstance(polarized,bool):
        raise ValueError('An explicit material constitution is required')
    previous=model.params.polarized_storage
    before=model.energy(),model.free_energy()
    model.params=replace(model.params,polarized_storage=polarized)
    try:
        model._validate()
        du,dg=model.energy()-before[0],model.free_energy()-before[1]
        if du != 0. or dg != 0.:
            raise ValueError('This counterfactual assumes unchanged current effective-state energy')
    except Exception:
        model.params=replace(model.params,polarized_storage=previous)
        raise
    return dict(time_ms=model.t,previous_polarized_storage=previous,polarized_storage=polarized,
        net_state_energy_change=du,net_state_free_energy_change=dg,net_physical_state_erased=False,
        interpretation='Distinct physical material law with identical effective starting state; not a biological upgrade or modeled autonomous manufacturing')
