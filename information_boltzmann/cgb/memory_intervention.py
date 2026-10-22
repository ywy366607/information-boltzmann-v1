"""Stoichiometry-matched conditional memory interventions for observation.

These forced *allowed* jumps condition an experiment, never the living rule.
They do not estimate the work cost of an external experiment controller.
The original reaction energy/heat is accounted, with no naked bit reset.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .mobile_exchange import MaterialEvent


def contact_key(model, event, node_ids):
    if event.interface >= 0:
        return 'p:'+str(event.interface)
    return 'i:'+str(node_ids[event.i])+':'+str(node_ids[event.j])


def matched_trace_condition(model, node_ids, carrier_ids, *, erase):
    """Condition write- vs active-: identical q/P/D, different stored r.

    Traverse the entire declared cohort in stable carrier-ID order. Local
    D may limit admissibility; report every rejection, never search for a
    future-successful carrier. Both branches draw no random variates and
    keep exactly the same resource and charge changes. C, geometry, types,
    topology and all other stored state remain matched.
    """
    if len(node_ids) != model.n or len(set(node_ids)) != model.n:
        raise ValueError('Complete distinct material identities required')
    if not isinstance(erase, bool):
        raise ValueError('An explicit conditional branch is required')
    all_traced = []
    for e, (i, j) in enumerate(zip(model.pre, model.post)):
        if model.trace[e]:
            all_traced.append(MaterialEvent('write', int(i), int(j), e, -1))
    for port, node in enumerate(model.port_nodes):
        if model.port_trace[port]:
            all_traced.append(MaterialEvent('write', int(node), int(node), port, -1, interface=port))
    all_traced.sort(key=lambda event: carrier_ids[contact_key(model,event,node_ids)])
    before = model.energy(), model.free_energy(), model.heat_out
    receipts = []
    for write in all_traced:
        active = replace(write, kind='active')
        available = model.events()
        key = contact_key(model,write,node_ids)
        if write not in available or active not in available:
            receipts.append(dict(key=key, carrier_id=carrier_ids[key], applied=False,
                                 reason='Native inverse channel or local depleted composition unavailable'))
            continue
        chosen = write if erase else active
        du,dg = model.event_delta(chosen)
        bound,_,index = model._contact_arrays(write)
        receipts.append(dict(key=key,carrier_id=carrier_ids[key],applied=True,
            source_node_id=int(node_ids[write.i]),bound_before=int(bound[index]),
            kind=chosen.kind,sign=-1,delta_u=du,delta_g=dg,
            trace_energy_if_kept=float(model.params.trace_energy-model.params.capture_coupling*bound[index]),
            electrode_charge_delta=0.))
        model.apply_event(chosen)
    model._validate()
    if max(abs(float(v)) for v in model.balances().values())>1e-8:
        raise ValueError('Conditional memory jumps failed the original material/energy ledger')
    return dict(mode='write_inverse' if erase else 'active_inverse',
        inspected_traced_carriers=len(all_traced),applied=sum(row['applied'] for row in receipts),
        receipts=receipts,delta_u=model.energy()-before[0],delta_g=model.free_energy()-before[1],
        reaction_heat_out=model.heat_out-before[2],external_protocol_work=0.,
        scope='Allowed-jump conditional comparison; experiment-controller selection cost is not modeled; never part of autonomous learning')


def assert_matched_trace_states(kept, erased):
    """Exact conditional-state check; includes all three original RNGs."""
    if type(kept) is not type(erased) or kept.params!=erased.params:
        raise ValueError('The same material constitution is required')
    for name in ('pre','post','capacity','port_nodes',*kept._arrays):
        if name not in ('trace','port_trace'):
            np.testing.assert_array_equal(getattr(kept,name),getattr(erased,name))
    for name in ('rng_reaction','rng_field','rng_motion'):
        assert getattr(kept,name).bit_generator.state==getattr(erased,name).bit_generator.state
    for name in ('domain_attempts','domain_births','domain_deaths','motion_attempts','motion_accepted','t',
                 'work_in','source_charge','electrode_charge','motion_heat','source_heat','conduction_heat'):
        assert getattr(kept,name)==getattr(erased,name)
    difference=kept.energy()-erased.energy()
    assert abs(difference-(kept.free_energy()-erased.free_energy()))<1e-9
    assert abs(difference+kept.heat_out-erased.heat_out)<1e-8
    return dict(physical_states_match_except_memory_and_its_heat=True,
                stored_memory_energy_difference=float(difference))
