"""Read native eligibility candidates, carrier retention and resource budgets.

No reward, learned objective, additional update, or stateful fatigue is
introduced. Observations expose the existing physical generator's roles.
"""
from __future__ import annotations

from collections import defaultdict

from .mobile_exchange import MaterialEvent


def resource_bandwidth(model):
    """Conditional P-use/recovery intensities, from actual admitted events.

    Activity at a node and assembly onto it draw on the same local P. A
    positive P change is recovery regardless of the reaction's name. These
    are instantaneous rates, not a finite-time throughput guarantee.
    """
    rows = [dict(node=i, available=int(model.charged[i]), spent=int(model.depleted[i]),
                 use_rate=0., recovery_rate=0., by_reaction={}) for i in range(model.n)]
    degree = model.degrees()
    for event in model.events(degree=degree):
        rate = model.event_rate(event, degree)
        for name, node, change in model._pools(event):
            if name != 'charged':
                continue
            key = 'recovery_rate' if change > 0 else 'use_rate'
            value = abs(change)*rate
            rows[node][key] += value
            group = rows[node]['by_reaction'].setdefault(event.kind, dict(use_rate=0., recovery_rate=0.))
            group[key] += value
    return dict(time_ms=model.t, nodes=rows,
                scope='Conditional native bridge/pool intensities; domain birth bounds are separate')


def contact_roles(model, edge):
    """Separate fast state, support capture, paid carrier and conductance."""
    if not 0 <= edge < len(model.pre):
        raise ValueError('An existing internal carrier is required')
    i, j = int(model.pre[edge]), int(model.post[edge])
    events = model._installed_contact_events(i, j, edge)
    admitted = set(model.events())
    rates = defaultdict(float)
    for event in events:
        if event in admitted:
            rates[event.kind+('+' if event.sign > 0 else '-')] += model.event_rate(event)
    current = float(model.bridge_conductance()[edge])
    # This is a derived structural observable, not a state reset or a new G.
    basal = model.unit_conductance*model._contact_transport_factor(
        MaterialEvent('active', i, j, edge))*model.params.gate_low
    return dict(pre=i, post=j, installed=True,
                activity_configuration=int(model.trace[edge]),
                bound_support=int(model.bound[edge]), free_support_at_post=int(model.free[j]),
                actual_conductance=current, basal_installed_conductance=float(basal),
                configuration_conductance=current-float(basal),
                capture_rate=float(rates['binding+']), release_rate=float(rates['binding-']),
                dismantle_rate=float(rates['assembly-']),
                trace_write_rate=float(rates['write+']), trace_recovery_rate=float(rates['thermal-']),
                activity_rate=float(rates['active+']),
                role='r has short-response/capture roles; carrier identity and paid structure are separate')


def support_return_paths(model):
    """Admitted downstream C->H->F routes to an existing upstream contact.

    This is reachability in the present state, not a statement that returned
    support represents useful prediction. Paths are selected without scores.
    """
    admitted = set(model.events())
    rows = []
    outgoing = defaultdict(list)
    for downstream, (a, b) in enumerate(zip(model.pre, model.post)):
        outgoing[int(a)].append((downstream, int(b)))
    for upstream, (i, j) in enumerate(zip(model.pre, model.post)):
        if model.bound[upstream]:
            continue
        for downstream, b in outgoing[int(j)]:
            if b == i:
                continue
            event = MaterialEvent('transport', int(j), b, downstream)
            if event in admitted:
                rows.append(dict(upstream_edge=upstream, downstream_edge=downstream,
                                 pre=int(i), middle=int(j), downstream=int(b),
                                 actual_transport_rate=model.event_rate(event)))
    return rows
