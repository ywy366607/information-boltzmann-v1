"""Long-term efficacy on actual KC release positions and MBON partners.

Polyadic partners share a presynaptic packet/resource pool but retain distinct
contact efficacies. The existing DAN/MBON compartment teacher is retained;
this does not reconstruct spatial dopamine release or molecular receptors.
"""
from dataclasses import replace

import torch


KC_CONTACT_VERSION = 1
KC_CONTACT_ID_NAMES = ('kc_contact_ids',)
LEGACY_FIELDS = ('comp_memory', 'comp_trace', 'comp_da_trace', 'comp_p_stats',
                 'comp_camp', 'credit_comp', 'credit_comp_trace', 'credit_comp_camp')
KC_CONTACT_STATE_FIELDS = ('kc_contact_memory', 'kc_contact_trace',
    'kc_contact_da_trace', 'kc_contact_p_stats', 'kc_contact_camp',
    'kc_contact_credit', 'kc_contact_trace_credit', 'kc_contact_camp_credit')


def attach_kc_contact_memory(model, *, saved=None):
    if not hasattr(model, 'kct_index'):
        raise ValueError('Contact learning requires native release-site geometry')
    sites, edges, delays, fractions = [], [], [], []
    offset = 0
    for lag in range(1, 5):
        site = getattr(model, f'kct_comp_{lag}_site')
        edge = getattr(model, f'kct_comp_{lag}_edge')
        sites.append(site)
        edges.append(edge)
        delays.append(torch.full_like(site, lag))
        fractions.append(getattr(model, f'kct_comp_{lag}_fraction'))
        model.register_buffer(f'kcm_slots_{lag}', torch.arange(
            offset, offset + site.numel(), device=site.device), persistent=False)
        offset += site.numel()
    site, edge, delay = torch.cat(sites), torch.cat(edges), torch.cat(delays)
    identity = torch.stack((site, edge, delay), 1)
    if saved is not None:
        if 'kc_contact_ids' not in saved or not torch.equal(saved['kc_contact_ids'].cpu(), identity.cpu()):
            raise ValueError('KC plastic contact identity mismatch')
    for name, value, persistent in [('kc_contact_ids', identity, True),
        ('kcm_site', site, False), ('kcm_edge', edge, False),
        ('kcm_fraction', torch.cat(fractions), False)]:
        model.register_buffer(name, value, persistent=persistent)
    for name, value in [('kcm_pre', model.comp_pre[edge]), ('kcm_post', model.comp_post[edge]),
                        ('kcm_sequence', torch.arange(edge.numel(), device=edge.device))]:
        model.register_buffer(name, value, persistent=False)
    model.kc_contact_memory_enabled = True
    model.kc_contact_write_enabled = True
    model.kc_contact_read_enabled = True


def contact_state(model, current):
    values = [getattr(current, name) for name in KC_CONTACT_STATE_FIELDS]
    if not getattr(model, 'kc_contact_memory_enabled', False):
        if any(v.numel() for v in values):
            raise ValueError('Contact state requires its actual contact identities')
        return values
    count = model.kcm_site.numel()
    if not values[0].numel():
        if any(v.numel() for v in values):
            raise ValueError('Incomplete native contact history')
        edge = model.kcm_edge
        # Extend the represented history, never synthesize new experience.
        from .fly_kc_terminals import terminal_state
        terminal = terminal_state(model, current)
        source = [current.comp_memory, terminal[4],
            current.comp_da_trace, current.comp_p_stats, current.comp_camp,
            current.credit_comp, terminal[5],
            current.credit_comp_camp]
        defaults = (1, 1, 1, 2, 1, 0, 1, 0)
        values = [v.index_select(1, edge) if v.numel() else current.h.new_zeros(rows, count)
                  for v, rows in zip(source, defaults)]
    for name, value in zip(KC_CONTACT_STATE_FIELDS, values):
        if value.ndim != 2 or value.shape[1] != count:
            raise ValueError('Malformed nonempty native contact history: ' + name)
    return values


class ContactLearningView:
    """Only learning coordinates change; native physiology/parameters are shared."""
    def __init__(self, model):
        self.original = model
        self.comp_pre = model.kcm_pre
        self.comp_post = model.kcm_post
        self.comp_edge_kc = model.kcm_sequence

    def __getattr__(self, name):
        return getattr(self.original, name)


def advance_contact_learning(model, current, h, spikes, ring, dopamine,
                             credit_dopamine, dpm_trace, kc_values, learning_rule,
                             spatial_dopamine_values=None, local_inputs=None):
    if getattr(model, '_comp_rate_probe', None) is not None:
        raise ValueError('Pooled-edge probe is not a native-contact probe')
    values = contact_state(model, current)
    local = replace(current, **dict(zip(LEGACY_FIELDS, values)))
    working = list(kc_values)
    working[4], working[5] = values[1], values[6]
    activity = working[3][0:1].index_select(1, model.kcm_site)
    options = {}
    if spatial_dopamine_values is not None and model.kc_dopamine_teacher_enabled:
        from .fly_kc_dopamine import contact_teacher
        options['local_dopamine'], options['local_dopamine_credit'] = contact_teacher(model,spatial_dopamine_values)
    if getattr(model, 'kc_lateral_enabled', False):
        from .fly_kc_lateral import effector_gate
        options['local_camp_gate'] = effector_gate(model, kc_values[9], model.kcm_site, 'camp')
    if local_inputs and 'kc_camp_gate' in local_inputs:
        from .fly_native_runtime import local_input
        gate = local_input(local_inputs, 'kc_camp_gate', working[3][0:1]).index_select(1, model.kcm_site)
        options['local_camp_gate'] = options['local_camp_gate'] * gate if 'local_camp_gate' in options else gate
    learned, working = learning_rule(ContactLearningView(model), local, h, spikes,
        ring, dopamine, credit_dopamine, dpm_trace, working, local_activity=activity, **options)
    result = [learned[name] for name in LEGACY_FIELDS]
    if not model.kc_contact_write_enabled:
        from .fly_carried_credit import split_comp_rows, COMP_PARAMS
        previous_phi, _ = split_comp_rows(values[5], len(COMP_PARAMS))
        _, updated_dopamine_trace = split_comp_rows(result[5], len(COMP_PARAMS))
        result[0] = values[0]
        result[5] = (previous_phi if updated_dopamine_trace is None else
                     torch.cat((previous_phi, updated_dopamine_trace), 0))
    return result


def contact_arrival(model, current):
    from .fly_kc_terminals import kc_arrival, _sum
    if not model.kc_contact_read_enabled:
        # Matched reference: retained previous connection memory, not erased brain.
        return kc_arrival(model, current, 'comp', weights=model.comp_weight * (1 + current.comp_memory[0]))
    from .fly_kc_terminals import terminal_state
    packets = terminal_state(model, current)[3]
    memory = contact_state(model, current)[0][0]
    out = current.h.new_zeros(model.comp_pre.numel())
    for lag in range(1, 5):
        prefix = f'kct_comp_{lag}'
        edge = getattr(model, prefix + '_edge')
        site = getattr(model, prefix + '_site')
        phi = memory.index_select(0, getattr(model, f'kcm_slots_{lag}'))
        value = packets[lag - 1].index_select(0, site)
        value = value * getattr(model, prefix + '_fraction') * model.comp_weight[edge] * (1 + phi)
        out = out + _sum(model, prefix, value, model.comp_pre.numel())
    return out.unsqueeze(0)
