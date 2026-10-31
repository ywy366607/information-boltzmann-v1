"""Declarative native mechanism assembly, state ownership and phase execution.

A descriptor is an executable-interface contract, not proof of full biological
function. Each module keeps its physical law and every local tensor identity.
The registry never infers missing receptor expression from an ROI name.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from pathlib import Path
import hashlib
import json
import math

import torch


@dataclass(frozen=True)
class NativeAttachment:
    function: str
    config_flag: str
    arguments: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    gate: str | None = None
    additional_flags: tuple[tuple[str, str], ...] = ()
    accepts_saved: bool = True


@dataclass(frozen=True)
class NativeMechanism:
    name: str
    module: str
    states: tuple[str, ...] = ()
    parameters: str | tuple[str, ...] = ()
    identities: str | tuple[str, ...] = ()
    flag: str | None = None
    phase: str | None = None
    function: str | None = None
    attachment: NativeAttachment | None = None

    def names(self, field):
        value = getattr(self, field)
        if isinstance(value, str):
            return tuple(getattr(_module(self.module), value))
        return value


@lru_cache(None)
def _module(name):
    return import_module('.' + name, __package__)


# Single owners for each physical field. Parent-terminal outputs are split by
# the adapter, never duplicated as a second STP or lateral physical history.
MECHANISMS = (
    NativeMechanism('amines', 'fly_amine_modulation',
        ('amine_concentration', 'amine_receptor'), 'AMINE_PARAMETER_NAMES',
        flag='amine_enabled', phase='threshold', function='step_amine_modulation'),
    NativeMechanism('cx', 'fly_cx_state', ('cx_trace',), 'CX_PARAMETER_NAMES',
        flag='cx_enabled', phase='threshold', function='step_cx_state'),
    NativeMechanism('mb_regulators', 'fly_mb_consolidation', ('dpm_trace', 'apl_trace'),
        'MB_CONSOLIDATION_PARAMETER_NAMES', flag='mb_consolidation_enabled',
        phase='threshold', function='step_mb_consolidation'),
    NativeMechanism('al', 'fly_olfactory_relay', ('al_local_trace', 'al_pn_drive'),
        'AL_PARAMETER_NAMES', flag='olfactory_relay_enabled', phase='threshold',
        function='step_olfactory_relay'),
    NativeMechanism('endocrine', 'fly_endocrine_exchange',
        ('endocrine_calcium', 'endocrine_lk', 'endocrine_receptor'),
        'ENDOCRINE_PARAMETER_NAMES', flag='endocrine_enabled', phase='threshold',
        function='step_endocrine_exchange'),
    NativeMechanism('mb_terminal', 'fly_mb_terminal',
        ('mb_terminal_gaba', 'mb_dpm_concentration', 'mb_dpm_receptor'),
        'MB_TERMINAL_PARAMETER_NAMES', 'MB_TERMINAL_ID_NAMES', 'mb_terminal_enabled',
        attachment=NativeAttachment('attach_mb_terminal', 'mb_terminal_projection',
            ('graph', 'annotations', 'dendritic_shares'), gate='mb_terminal_enabled')),
    NativeMechanism('apl_cable', 'fly_apl_cable',
        ('apl_voltage', 'apl_ge', 'apl_gi', 'apl_packets'),
        'APL_CABLE_PARAMETER_NAMES', 'APL_CABLE_ID_NAMES', 'apl_output_enabled',
        attachment=NativeAttachment('attach_apl_cable', 'apl_cable', ('apl_geometry',),
                                    gate='apl_output_enabled')),
    NativeMechanism('apl_release', 'fly_apl_local_release',
        ('apl_release_gate', 'apl_release_calcium', 'apl_release_resources'),
        'APL_LOCAL_RELEASE_PARAMETER_NAMES', flag='apl_local_release_enabled',
        attachment=NativeAttachment('attach_apl_local_release', 'apl_local_release',
                                    dependencies=('apl_cable',))),
    NativeMechanism('kc_terminal', 'fly_kc_terminals',
        ('kc_terminal_gaba', 'kc_terminal_5ht', 'kc_terminal_receptor',
         'kc_terminal_packets', 'kc_terminal_eligibility', 'kc_terminal_eligibility_credit'),
        'KC_TERMINAL_PARAMETER_NAMES', 'KC_TERMINAL_ID_NAMES', 'kc_terminal_enabled',
        attachment=NativeAttachment('attach_kc_terminals', 'kc_terminal', ('kc_terminal_geometry',),
            ('mb_terminal', 'apl_cable'), 'kc_terminal_gating_enabled')),
    NativeMechanism('kc_site_stp', 'fly_kc_site_stp',
        ('kc_site_resources', 'kc_site_utilization'), 'KC_SITE_STP_PARAMETER_NAMES',
        flag='kc_site_stp_enabled', attachment=NativeAttachment('attach_kc_site_stp', 'kc_site_stp',
            dependencies=('kc_terminal',), gate='kc_site_stp_enabled')),
    NativeMechanism('kc_lateral', 'fly_kc_lateral',
        ('kc_lateral_receptor', 'kc_lateral_signal'), 'KC_LATERAL_PARAMETER_NAMES',
        'KC_LATERAL_ID_NAMES', 'kc_lateral_enabled',
        attachment=NativeAttachment('attach_kc_lateral', 'kc_lateral', ('kc_lateral_geometry',),
                                    ('kc_contact',))),
    NativeMechanism('kc_contact', 'fly_kc_contact_memory',
        # Read from the module below: contact identities do not share memory.
        parameters=(), identities='KC_CONTACT_ID_NAMES', flag='kc_contact_memory_enabled',
        attachment=NativeAttachment('attach_kc_contact_memory', 'kc_contact_memory',
                                    dependencies=('kc_terminal',))),
    NativeMechanism('kc_dopamine', 'fly_kc_dopamine',
        ('kc_dopamine_signal', 'kc_dopamine_credit'), 'KC_DOPAMINE_PARAMETER_NAMES',
        'KC_DOPAMINE_ID_NAMES', 'kc_dopamine_enabled',
        attachment=NativeAttachment('attach_kc_dopamine', 'kc_dopamine', ('kc_dopamine_geometry',),
                                    ('kc_terminal',))),
    NativeMechanism('dan_release', 'fly_dan_release',
        ('dan_local_receptor', 'dan_local_calcium', 'dan_local_resources', 'dan_local_packets'),
        'DAN_RELEASE_PARAMETER_NAMES', 'DAN_RELEASE_ID_NAMES', 'dan_local_enabled',
        attachment=NativeAttachment('attach_dan_release', 'dan_local_release', ('dan_release_geometry',),
            ('kc_terminal',), additional_flags=(('dan_carried_credit_enabled', 'dan_carried_credit'),))),
    NativeMechanism('dan_credit', 'fly_dan_credit',
        ('dan_credit_receptor', 'dan_credit_calcium', 'dan_credit_resources', 'dan_credit_packets'),
        flag='dan_carried_credit_enabled'),
    NativeMechanism('dan_receiver_credit', 'fly_dan_receiver_credit',
        ('dan_credit_dopamine',), flag='dan_receiver_credit_enabled',
        attachment=NativeAttachment('attach_receiver_credit', 'dan_receiver_credit',
                                    dependencies=('dan_release',), accepts_saved=False)),
    NativeMechanism('sensory_terminal', 'fly_sensory_terminals',
        ('sensory_terminal_signal', 'sensory_terminal_packets'), 'SENSORY_TERMINAL_PARAMETER_NAMES',
        'SENSORY_TERMINAL_ID_NAMES', 'sensory_terminal_enabled',
        attachment=NativeAttachment('attach_sensory_terminals', 'sensory_terminal',
                                    ('sensory_terminal_geometry',), gate='sensory_terminal_enabled')),
    NativeMechanism('body_route', 'fly_body_loop',
        ('body_action_trace', 'body_feedback_trace'), 'BODY_LOOP_PARAMETER_NAMES', flag='body_loop_enabled'),
    NativeMechanism('historical_body_actuator', 'fly_body_loop',
        ('body_muscle', 'body_reafference'), 'BODY_ACTUATOR_PARAMETER_NAMES', flag='body_proxy_compatibility'),
    NativeMechanism('motor_port', 'fly_native_motor_port', parameters='MOTOR_PORT_PARAMETER_NAMES',
        identities=('native_motor_body_ids',), flag='native_motor_port_enabled'),
    NativeMechanism('visual', 'fly_visual_physiology', parameters='VISUAL_PARAMETER_NAMES',
        flag='visual_graded_enabled'),
    NativeMechanism('generated', 'fly_native_runtime', ('native_state',),
        identities=('native_program_digest',), flag='native_program_enabled'),
)


OPERATIONS = {
    ('apl_cable', 'advance'): 'advance_apl_cable',
    ('apl_cable', 'readout'): 'apply_apl_readout',
    ('mb_terminal', 'advance'): 'step_mb_terminal',
    ('mb_terminal', 'release'): 'apply_mb_terminal_release',
    ('kc_terminal', 'advance'): 'advance_kc_terminals',
    ('kc_terminal', 'arrival'): 'kc_arrival',
    ('kc_terminal', 'state'): 'terminal_state',
    ('kc_contact', 'advance'): 'advance_contact_learning',
    ('kc_contact', 'arrival'): 'contact_arrival',
    ('kc_dopamine', 'advance'): 'advance_kc_dopamine',
    ('dan_release', 'advance'): 'advance_dan_release',
    ('dan_release', 'arrival'): 'dan_arrival',
    ('dan_release', 'state'): 'dan_state',
    ('dan_credit', 'advance'): 'advance_credit',
    ('dan_receiver_credit', 'advance'): 'advance_receiver_credit',
    ('sensory_terminal', 'advance'): 'advance_sensory_terminals',
    ('sensory_terminal', 'delivery'): 'sensory_delivery',
    ('sensory_terminal', 'state'): 'terminal_values',
    ('body_route', 'advance'): 'step_body_loop',
    ('historical_body_actuator', 'advance'): 'step_body_actuator',
}


def attachment_order():
    """Topological order generated from the same state/operator registry."""
    specs = {spec.name: spec for spec in MECHANISMS if spec.attachment is not None}
    active, complete, ordered = set(), set(), []
    def visit(name):
        if name in active:
            raise ValueError('Cyclic native mechanism dependency: ' + name)
        if name in complete:
            return
        if name not in specs:
            raise ValueError('Unregistered native anatomical dependency: ' + name)
        active.add(name)
        for dependency in specs[name].attachment.dependencies:
            visit(dependency)
        active.remove(name)
        complete.add(name)
        ordered.append(specs[name])
    for name in specs:
        visit(name)
    return tuple(ordered)


@lru_cache(None)
def operator(component, operation='advance'):
    spec = next((s for s in MECHANISMS if s.name == component), None)
    if spec is None or (component, operation) not in OPERATIONS:
        raise ValueError('Unregistered native operation: ' + component + ':' + operation)
    return getattr(_module(spec.module), OPERATIONS[component, operation])


class NativeStateLayout:
    """One persistent tensor for any number of generated local state fields.

    Offsets are storage metadata; no compartments or synapses are pooled.
    Shape/default declarations are part of the immutable program identity.
    """
    def __init__(self, fields):
        self.fields = tuple((name, tuple(shape), float(initial)) for name, shape, initial in fields)
        if len({name for name, _, _ in self.fields}) != len(self.fields):
            raise ValueError('Duplicate generated state identity')
        self.entries = {}
        offset = 0
        for name, shape, initial in self.fields:
            if not name or any(not isinstance(n, int) or n < 1 for n in shape) or not math.isfinite(initial):
                raise ValueError('Invalid generated state declaration')
            count = math.prod(shape)
            self.entries[name] = (offset, count, shape)
            offset += count
        self.size = offset

    def initialize(self, *, device=None, dtype=torch.float32):
        out = torch.empty(self.size, device=device, dtype=dtype)
        for name, _, value in self.fields:
            start, count, _ = self.entries[name]
            out[start:start + count].fill_(value)
        return out

    def unpack(self, values):
        if values.shape != (self.size,):
            raise ValueError('Partial or incompatible generated physical history')
        return {name: values.narrow(0, start, count).view(shape)
                for name, (start, count, shape) in self.entries.items()}

    def replace(self, values, updates):
        current = self.unpack(values)
        if set(updates) - set(current):
            raise ValueError('Unowned generated physical state write')
        fields = []
        for name, old in current.items():
            new = updates.get(name, old)
            if new.shape != old.shape or new.dtype != old.dtype or new.device != old.device:
                raise ValueError('Generated state shape/dtype/device changed: ' + name)
            fields.append(new.reshape(-1))
        return torch.cat(fields) if fields else values


@dataclass(frozen=True)
class NativeInstruction:
    name: str
    kind: str
    inputs: tuple[tuple[str, str], ...]
    outputs: tuple[str, ...]
    phase: str = 'pre_membrane'
    attributes: tuple[tuple[str, object], ...] = ()


def _binding_kernel(**values):
    from .fly_biophysical_operators import ligand_binding
    return (ligand_binding(**values),)


def _resource_kernel(available, probability, retained_fraction):
    from .fly_biophysical_operators import resource_recovery
    released = available * probability
    return resource_recovery(available, released, retained_fraction), released


def _transport_kernel(value, log_transport, geometry):
    from .fly_apl_cable import diffuse
    return (diffuse(value, log_transport, geometry),)


def _contact_kernel(packets, source, delay, target, weight, destination_size):
    # Explicit old delay slots, actual source sites and actual target nodes.
    # Coalescing addition does not replace any persistent local state.
    released = packets[delay - 1, source] * weight
    value = released.new_zeros(int(destination_size))
    return (value.index_add(0, target, released).unsqueeze(0),)


def _gate_kernel(gate, voltage, half_voltage, slope, tau):
    target = torch.sigmoid((voltage - half_voltage) / slope)
    return (gate + (-torch.expm1(-1. / tau)) * (target - gate),)


def _membrane_kernel(voltage, capacitance, conductance, reversal, current):
    # First dimension indexes ionic species; all spatial nodes remain in the
    # remaining dimensions. Positive capacitance/conductance come from bound
    # material parameterizations, not a biological channel-identity guess.
    total = conductance.sum(0)
    drive = (conductance * reversal).sum(0) + current
    rate = total / capacitance
    equilibrium = drive / total
    return (voltage + (-torch.expm1(-rate)) * (equilibrium - voltage),)


def _eligibility_kernel(trace, activity, retained_fraction):
    """Persistent local coactivity history, not a derivative of old BPTT."""
    return (retained_fraction * trace + (1. - retained_fraction) * activity,)


def _plasticity_kernel(memory, eligibility, teacher, learning_rate, forget_rate, window):
    """Existing three-factor balance; the binder identifies real learning sites.

    A declarative operand graph supplies the locally delivered teacher and
    eligibility. No world reward, decoder score or speaking quota is created.
    """
    weight = window * eligibility
    return (memory + weight * (learning_rate * teacher - forget_rate * memory),)


def _site_packet_kernel(packets, site, edge, weights, fraction, order, count, target,
                        *, size, lag, weight_first=False):
    value = packets[lag - 1].index_select(0, site)
    if weight_first:
        value = value * weights.index_select(0, edge) * fraction
    else:
        value = value * fraction * weights.index_select(0, edge)
    if order.numel():
        sums = torch.segment_reduce(value.index_select(0, order), 'sum', lengths=count)
        result = value.new_zeros(size).index_copy(0, target, sums)
    else:
        result = value.new_zeros(size)
    return (result.unsqueeze(0),)


def _sum_kernel(**values):
    result = torch.zeros_like(next(iter(values.values())))
    for value in values.values():
        result = result + value
    return (result,)


def _multiply_kernel(value, factor):
    return (value * factor,)


def _zero_at_kernel(value, index):
    return (value.index_fill(1, index if index.dtype == torch.long else index.long(), 0.),)


GENERATED_KERNELS = {
    'ligand_binding': _binding_kernel,
    'resource_balance': _resource_kernel,
    'conservative_transport': _transport_kernel,
    'contact_delivery': _contact_kernel,
    'voltage_gate': _gate_kernel,
    'conductance_membrane': _membrane_kernel,
    'local_eligibility': _eligibility_kernel,
    'three_factor_plasticity': _plasticity_kernel,
    'site_packet_delivery': _site_packet_kernel,
    'ordered_sum': _sum_kernel,
    'multiply': _multiply_kernel,
    'zero_at': _zero_at_kernel,
}


# Local ports name actual physical coordinates, not a population statistic.
# (phase, anatomical index-buffer, gate range). Global conductance ports remain
# available for the existing point-cell receivers; axonal receptors use these
# local surfaces instead of being silently projected to a soma.
LOCAL_PORTS = {
    'apl_ge': ('pre_membrane', 'aplc_owner', False, ('apl_cable',)),
    'apl_gi': ('pre_membrane', 'aplc_owner', False, ('apl_cable',)),
    'kc_calcium_gate': ('pre_membrane', 'kct_site_node', True, ('kc_terminal',)),
    'dan_receptor_input': ('post_release', 'danl_owner', False, ('dan_release', 'dan_credit')),
    'kc_camp_gate': ('learning', 'kct_site_node', True, ('kc_contact',)),
}

ROUTE_REPLACEMENTS = {
    'kc_fast': ('extra_ge', ('kct_index',)),
    'kc_apl': ('apl_ge', ('kct_index', 'apl_cable_index')),
    'kc_dan': ('dan_receptor_input', ('kct_index', 'danl_index')),
}


def owns_route(model, name):
    return (getattr(model, 'native_program_enabled', False)
            and name in getattr(model, 'native_program_routes', ()))


def validate_program_bindings(model):
    if not getattr(model, 'native_program_enabled', False):
        return
    document = model.native_program_document
    available = set(dict(model.named_buffers())) | set(dict(model.named_parameters()))
    for record in document['bindings'].values():
        if record['kind'] in ('buffer', 'tensor') and record['name'] not in available:
            raise ValueError('Unbound native anatomical/material operand: ' + record['name'])
        if record['kind'] == 'flag' and not isinstance(getattr(model, record['name'], None), bool):
            raise ValueError('Native route control is not an existing boolean interface')
    ports = {kind for _, kind, _ in model.native_execution_program.effectors}
    for name in document.get('owned_routes', ()):
        if name not in ROUTE_REPLACEMENTS:
            raise ValueError('Unregistered native route ownership: ' + name)
        port, anatomy = ROUTE_REPLACEMENTS[name]
        if port not in ports or any(not hasattr(model, key) for key in anatomy):
            raise ValueError('Native route replacement has no complete physical endpoint: ' + name)
    identities = document.get('identity_bindings', {})
    if document.get('owned_routes') and not identities:
        raise ValueError('Native route replacements require actual anatomical content identities')
    for name, expected in identities.items():
        if not hasattr(model, name) or getattr(model, name).detach().cpu().numpy().tobytes().hex() != expected:
            raise ValueError('Generated route anatomy changed: ' + name)


def component_inputs(values, component):
    result = {name: value for name, value in values.items()
              if name in LOCAL_PORTS and component in LOCAL_PORTS[name][3]}
    return result or None


def local_input(inputs, name, reference):
    """Validate a local physical operand without implicit spatial broadcast."""
    value = (inputs or {}).get(name)
    if value is None:
        return None
    if value.shape != reference.shape or value.dtype != reference.dtype or value.device != reference.device:
        raise ValueError('Native local input has incompatible location/dtype/device: ' + name)
    for block in value.reshape(-1).split(65536):
        if not bool(torch.isfinite(block).all()) or bool((block < 0).any()):
            raise ValueError('Invalid local native physical input: ' + name)
        if LOCAL_PORTS[name][2] and bool((block > 1).any()):
            raise ValueError('Native local gate exceeds resource bounds: ' + name)
    return value


class NativeExecutionProgram:
    """Validated local dataflow for generated mechanisms, with explicit clocks.

    old:/new: operands distinguish pre-tick history from current intermediate
    values. external: operands must be supplied by a registered brain adapter;
    missing receptor/effectors cannot silently turn into a soma injection.
    """
    PHASES = ('pre_membrane', 'post_release', 'learning')

    def __init__(self, layout, instructions, external_names, *, anatomy_digest, evidence,
                 effectors=()):
        self.layout = layout
        self.instructions = tuple(instructions)
        self.external_names = tuple(external_names)
        self.evidence = dict(evidence)
        self.anatomy_digest = anatomy_digest
        self.effectors = tuple(effectors)
        if not anatomy_digest or not self.evidence:
            raise ValueError('Generated native program needs anatomy and mechanism provenance')
        names = set(layout.entries)
        available = set()
        seen_instructions = set()
        previous_phase = -1
        for item in self.instructions:
            if item.name in seen_instructions or item.kind not in GENERATED_KERNELS:
                raise ValueError('Duplicate instruction or unknown biological operator')
            seen_instructions.add(item.name)
            argument_names = [name for name, _ in item.inputs + item.attributes]
            if len(set(argument_names)) != len(argument_names):
                raise ValueError('Duplicate generated kernel argument')
            if any(not isinstance(value, (bool, int, float, str)) for _, value in item.attributes):
                raise ValueError('Generated instruction attributes must be scalar data')
            if item.phase not in self.PHASES:
                raise ValueError('Unregistered native clock phase')
            phase = self.PHASES.index(item.phase)
            if phase < previous_phase:
                raise ValueError('Generated clock phases are out of causal order')
            previous_phase = phase
            for _, reference in item.inputs:
                namespace, separator, key = reference.partition(':')
                known = names if namespace == 'old' else (available if namespace == 'new' else set(self.external_names))
                if not separator or namespace not in ('old', 'new', 'external') or key not in known:
                    raise ValueError('Unbound native mechanism operand: ' + reference)
            if not item.outputs or len(set(item.outputs)) != len(item.outputs) or available.intersection(item.outputs):
                raise ValueError('Duplicate native output writer')
            available.update(item.outputs)
        if not names.issubset(available):
            raise ValueError('Every generated persistent field needs one explicit tick update')
        for phase, kind, output in self.effectors:
            global_kinds = ('threshold_multiplier', 'extra_ge', 'extra_gi')
            if phase not in self.PHASES or kind not in global_kinds + tuple(LOCAL_PORTS):
                raise ValueError('Undeclared native effector/clock interface')
            producers = [item for item in self.instructions if output in item.outputs]
            if len(producers) != 1 or self.PHASES.index(producers[0].phase) > self.PHASES.index(phase):
                raise ValueError('Native effector reads missing or future physical output')
            if kind in LOCAL_PORTS and phase != LOCAL_PORTS[kind][0]:
                raise ValueError('Local native effector is bound to the wrong physical clock')
            if kind in global_kinds and phase != 'pre_membrane':
                raise ValueError('Conductance/threshold delivery belongs before the membrane step')
        document = {'fields': layout.fields,
                    'instructions': [vars(item) for item in self.instructions],
                    'external_names': self.external_names, 'anatomy_digest': anatomy_digest,
                    'mechanism_evidence': self.evidence, 'effectors': self.effectors}
        self.digest = hashlib.sha256(json.dumps(document, sort_keys=True).encode()).digest()

    def state_dict(self):
        return {'version': 1, 'fields': self.layout.fields,
                'instructions': [vars(item) for item in self.instructions],
                'external_names': self.external_names, 'anatomy_digest': self.anatomy_digest,
                'evidence': self.evidence, 'effectors': self.effectors}

    @classmethod
    def from_state_dict(cls, data):
        if data['version'] != 1:
            raise ValueError('Unsupported generated physical program version')
        code = [NativeInstruction(item['name'], item['kind'],
                                  tuple(tuple(v) for v in item['inputs']), tuple(item['outputs']),
                                  item.get('phase', 'pre_membrane'),
                                  tuple(tuple(v) for v in item.get('attributes', ()))) for item in data['instructions']]
        return cls(NativeStateLayout(data['fields']), code, data['external_names'],
                   anatomy_digest=data['anatomy_digest'], evidence=data['evidence'],
                   effectors=tuple(tuple(e) for e in data['effectors']))

    def begin(self, values):
        return {'old': self.layout.unpack(values), 'new': {}, 'phases_done': []}

    def execute_phase(self, phase, context, external):
        number = self.PHASES.index(phase)
        if context['phases_done'] != list(self.PHASES[:number]):
            raise ValueError('Native phase repeated, skipped or reordered')
        for item in self.instructions:
            if item.phase != phase:
                continue
            arguments = {}
            for name, reference in item.inputs:
                namespace, _, key = reference.partition(':')
                source = external if namespace == 'external' else context[namespace]
                if key not in source:
                    raise ValueError('Native external operand not bound: ' + key)
                arguments[name] = source[key]
            arguments.update(item.attributes)
            values = GENERATED_KERNELS[item.kind](**arguments)
            if len(values) != len(item.outputs):
                raise ValueError('Native kernel output contract changed')
            context['new'].update(zip(item.outputs, values))
        context['phases_done'].append(phase)

    def finish(self, values, context):
        if context['phases_done'] != list(self.PHASES):
            raise ValueError('Incomplete physical tick cannot be saved as completed')
        updates = {name: context['new'][name] for name in self.layout.entries}
        return self.layout.replace(values, updates)


def attach_execution_program(model, program, *, saved=None):
    if not program.effectors:
        raise ValueError('Generated component lacks an actual neural effector; do not count as connected')
    device = next(model.parameters()).device
    digest = torch.tensor(list(program.digest), dtype=torch.uint8, device=device)
    if saved is not None:
        previous = saved.get('native_program_digest')
        if previous is None or not torch.equal(previous.cpu(), digest.cpu()):
            raise ValueError('Native program changed; explicit complete-state migration required')
    model.register_buffer('native_program_digest', digest, persistent=True)
    model.native_execution_program = program
    model.native_program_enabled = True


def generated_effectors(model, context, phase):
    if context is None:
        return {}
    result = {}
    for clock, kind, output in model.native_execution_program.effectors:
        if clock != phase:
            continue
        if clock not in context['phases_done']:
            raise ValueError('Effector used before its physical phase')
        value = context['new'][output]
        if kind in LOCAL_PORTS:
            _, surface, bounded_gate, _ = LOCAL_PORTS[kind]
            if not hasattr(model, surface):
                raise ValueError('Native local port anatomy is absent: ' + surface)
            expected_size = getattr(model, surface).numel()
        else:
            expected_size, bounded_gate = model.n_neurons, False
        if value.shape != (1, expected_size):
            raise ValueError('Native effector has incompatible anatomical surface: ' + output)
        anchor = next(model.parameters())
        if value.dtype != anchor.dtype:
            raise ValueError('Native effector dtype differs from the continuing brain')
        value = value.to(anchor.device)
        # These surfaces have N entries, never a dense source-by-cell product.
        # Keep diagnostics bounded even when callers use a strided view.
        for block in value.reshape(-1).split(65536):
            if not bool(torch.isfinite(block).all()) or bool((block < 0).any()):
                raise ValueError('Invalid native conductance/threshold effector: ' + output)
            if bounded_gate and bool((block > 1).any()):
                raise ValueError('Native release/learning gate exceeds physical resource bounds')
        if kind == 'threshold_multiplier' or bounded_gate:
            result[kind] = value if kind not in result else result[kind] * value
        else:
            result[kind] = value if kind not in result else result[kind] + value
    return result


def generated_parameter_names(model):
    return tuple(name for name, _ in model.named_parameters() if name.startswith('native_materials.'))


def attach_program_document(model, document, *, saved=None):
    """Bind a data-only generated program; no Python expressions in archives.

    Unknown kinetics require explicit material declarations. Constants/arrays,
    parameter transforms and actual physical/frame fields are the only legal
    operands. This enables batch construction without a new Python module or
    dataclass field for each mechanism/cell type.
    """
    import numpy as np
    from torch import nn
    document = json.loads(json.dumps(document))
    if document.get('state_device', 'brain') not in ('brain', 'cpu'):
        raise ValueError('Generated physical state placement must be brain or cpu')
    program = NativeExecutionProgram.from_state_dict(document['program'])
    program.digest = hashlib.sha256(json.dumps(document, sort_keys=True).encode()).digest()
    anchor = next(model.parameters())
    device = anchor.device
    declarations = document.get('materials', {})
    if hasattr(model, 'native_materials'):
        raise ValueError('Refuse to replace an attached native material history')
    model.native_materials = nn.ParameterDict()
    for name, record in declarations.items():
        if '.' in name or not name:
            raise ValueError('Invalid native material identity')
        value = torch.full(tuple(record['shape']), float(record['initial']), device=device, dtype=anchor.dtype)
        model.native_materials[name] = nn.Parameter(value)
    bindings = {}
    arrays = {}
    loaded_files = set()

    def load_array(record):
        path = Path(record['path'])
        if not record.get('sha256'):
            raise ValueError('Native anatomical operand lacks content identity')
        identity = (str(path.resolve()), record['sha256'])
        if identity not in loaded_files:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
                    digest.update(block)
            if digest.hexdigest() != record['sha256']:
                raise ValueError('Native anatomical operand identity changed')
            loaded_files.add(identity)
        key = identity + (record['key'],)
        if key not in arrays:
            with np.load(path, allow_pickle=False) as archive:
                arrays[key] = torch.from_numpy(archive[record['key']].copy())
        return arrays[key]

    def binding(record):
        kind = record['kind']
        if kind == 'material':
            name = record['name']
            if name not in model.native_materials:
                raise ValueError('Undeclared native material: ' + name)
            transform = record.get('transform', 'identity')
            if transform not in ('identity', 'exp', 'sigmoid'):
                raise ValueError('Unsupported native parameter transform')
            def material(model, current, frame):
                value = model.native_materials[name]
                value = value if transform == 'identity' else getattr(torch, transform)(value)
                return value.to(current.native_state.device)
            return material
        if kind in ('buffer', 'tensor'):
            name = record['name']
            # Spatial buffers are rebuilt after base parameters on restore.
            # The assembler validates them before any actual tick can run.
            return lambda model, current, frame: (model.get_parameter(name)
                if name in dict(model.named_parameters()) else model.get_buffer(name)).to(current.native_state.device)
        if kind == 'flag':
            name = record['name']
            return lambda model, current, frame: current.h.new_tensor(float(getattr(model, name))).to(current.native_state.device)
        if kind == 'component_state':
            component, slot = record['component'], record['slot']
            accessor = operator(component, 'state')
            return lambda model, current, frame: accessor(model, current)[slot].to(current.native_state.device)
        if kind in ('physical', 'frame'):
            name, slot = record['name'], record.get('slot')
            def physical(model, current, frame):
                value = getattr(current, name) if kind == 'physical' else frame[name]
                if slot is not None:
                    value = value[slot]
                if isinstance(value, tuple):
                    value = torch.cat(value, dim=0)
                return value.to(current.native_state.device) if isinstance(value, torch.Tensor) else value
            return physical
        if kind == 'geometry':
            from .fly_kc_terminals import CellwiseGeometry
            keys = record.get('keys', {'area': 'volume', 'first': 'u', 'second': 'v',
                                       'conductance': 'g', 'owner': 'owner'})
            parts = {name: load_array({**record, 'key': key}).numpy() for name, key in keys.items()}
            value = CellwiseGeometry(**parts)
            return lambda model, current, frame: value
        if kind == 'array':
            value = load_array(record)
        elif kind == 'constant':
            allowed = {'float32': torch.float32, 'float64': torch.float64,
                       'int64': torch.int64, 'int32': torch.int32}
            value = torch.tensor(record['value'], dtype=allowed[record.get('dtype', 'float32')])
        else:
            raise ValueError('Unregistered native operand kind: ' + kind)
        return lambda model, current, frame: value.to(current.native_state.device)
    if set(document['bindings']) != set(program.external_names):
        raise ValueError('Native adapter operand coverage mismatch')
    for name, record in document['bindings'].items():
        bindings[name] = binding(record)
    frames = {'pre_membrane': {'source', 'token', 'options', 'base_rates'},
              'post_release': {'ring', 'spikes', 'h', 'kc_values'},
              'learning': {'ring', 'h', 'dopamine', 'kc_values', 'dan_values'}}
    for instruction in program.instructions:
        for _, reference in instruction.inputs:
            if reference.startswith('external:'):
                record = document['bindings'][reference.partition(':')[2]]
                if record['kind'] == 'frame' and record['name'] not in frames[instruction.phase]:
                    raise ValueError('Native instruction reads an unavailable future frame')
    if saved is not None:
        expected = set(generated_parameter_names(model))
        found = {name for name in saved if name.startswith('native_materials.')}
        if expected != found:
            raise ValueError('Partial native material checkpoint')
        model.load_state_dict({name: saved[name] for name in expected}, strict=False)
    attach_execution_program(model, program, saved=saved)
    model.native_external_bindings = bindings
    model.native_program_document = document
    model.native_program_state_device = document.get('state_device', 'brain')
    model.native_program_routes = tuple(document.get('owned_routes', ()))
    return program


def archive_program(model):
    if not getattr(model, 'native_program_enabled', False):
        return None
    document = getattr(model, 'native_program_document', None)
    if document is None:
        raise ValueError('Inline native adapters are not archiveable; declare complete operand provenance')
    return document


def generated_phase(model, current, phase, context=None, **frame):
    """Advance only explicitly attached programs; old individuals stay exact."""
    if not getattr(model, 'native_program_enabled', False):
        return context
    program = model.native_execution_program
    if context is None:
        if phase != program.PHASES[0]:
            raise ValueError('Native generated tick was not started')
        context = program.begin(current.native_state)
    bindings = getattr(model, 'native_external_bindings', {})
    required = {ref.partition(':')[2] for item in program.instructions if item.phase == phase
                for _, ref in item.inputs if ref.startswith('external:')}
    external = {}
    for key in required:
        if key not in bindings:
            raise ValueError('Native executable adapter is unbound: ' + key)
        value = bindings[key]
        external[key] = value(model, current, frame) if callable(value) else value
    program.execute_phase(phase, context, external)
    return context


def mechanism_states(spec):
    if spec.name == 'kc_contact':
        return tuple(_module(spec.module).KC_CONTACT_STATE_FIELDS)
    return spec.states


def state_owners():
    owners = {}
    for spec in MECHANISMS:
        for name in mechanism_states(spec):
            if name in owners:
                raise ValueError('Duplicate native state writer: ' + name)
            owners[name] = spec.name
    return owners


def identity_names():
    return tuple(dict.fromkeys(name for spec in MECHANISMS for name in spec.names('identities')))


def parameter_names():
    return tuple(dict.fromkeys(name for spec in MECHANISMS for name in spec.names('parameters')))


def pack_component_updates(current, updates):
    """Map exact component outputs to owned fields; inactive fields persist."""
    known = {spec.name: spec for spec in MECHANISMS}
    unexpected = set(updates) - set(known)
    if unexpected:
        raise ValueError('Unknown native state components: ' + ','.join(sorted(unexpected)))
    result = {}
    for spec in MECHANISMS:
        names = mechanism_states(spec)
        values = updates.get(spec.name)
        if values is None:
            values = tuple(getattr(current, name) for name in names)
        if len(values) != len(names):
            raise ValueError(f'{spec.name}: {len(values)} outputs for {len(names)} owned states')
        result.update(zip(names, values))
    return result


def run_threshold_phase(model, current, thresholds):
    """Execute registered native threshold operators once, in declared order."""
    updates = {}
    for spec in MECHANISMS:
        if spec.phase != 'threshold':
            continue
        if spec.flag and not getattr(model, spec.flag, False):
            updates[spec.name] = tuple(getattr(current, name) for name in mechanism_states(spec))
            continue
        values = getattr(_module(spec.module), spec.function)(model, current, thresholds)
        thresholds, state = values[0], values[1:]
        if len(state) != len(mechanism_states(spec)):
            raise ValueError('Invalid threshold component output: ' + spec.name)
        updates[spec.name] = state
    return thresholds, updates


def runtime_inventory(model, current=None):
    """Report executable bindings separately from biological completeness."""
    parameters = dict(model.named_parameters())
    rows = []
    for spec in MECHANISMS:
        names = mechanism_states(spec)
        present = [name for name in spec.names('parameters') if name in parameters]
        enabled = bool(getattr(model, spec.flag, False)) if spec.flag else False
        configured_enabled = enabled
        excluded = None
        if spec.name == 'mb_regulators' and getattr(model, 'mb_pooled_quarantined', False):
            enabled, excluded = False, 'Historical pooled MB traces are quarantined'
        if spec.name == 'body_route' and not getattr(model, 'body_proxy_compatibility', False):
            enabled, excluded = False, 'Historical synthetic body feedback is quarantined'
        values = {} if current is None else {name: getattr(current, name) for name in names}
        rows.append({'component': spec.name, 'enabled': enabled, 'configured_enabled': configured_enabled,
                     'excluded_reason': excluded, 'phase': spec.phase,
                     'registered_operations': [operation for component, operation in OPERATIONS
                                               if component == spec.name],
                     'parameters': present, 'state_fields': list(names),
                     'state_bytes': sum(value.numel() * value.element_size() for value in values.values()),
                     'anatomical_identities': [n for n in spec.names('identities') if hasattr(model, n)],
                     'biological_completeness_verified': False})
    return rows


# index_copy needs int64 indices: buffers that feed it stay int64 (they are the smaller, unique target lists)
LONG_INDEX_PATTERNS = ('_target', 'injection_index', 'local_slots', 'apl_cable_index', 'mb_terminal_kc_index',
                       'kct_index', 'st_cell')


def compact_index_buffers(model, *, keep=()):
    """Store every int64 buffer whose values fit in int32 as int32 (execution memory only, 2026-10-19).

    Index values are unchanged, so every gather/scatter reads the same elements. Persistent identities (anatomical
    ids, digests) and buffers named in keep stay as they are. Returns (count, bytes saved).
    """
    import torch
    persistent = set(model.state_dict().keys())
    count = saved = 0
    for module_name, module in model.named_modules():
        for name, value in list(module._buffers.items()):
            full = f'{module_name}.{name}' if module_name else name
            if (value is None or value.dtype != torch.int64 or full in persistent or full in keep
                    or name.endswith('_ids') or 'digest' in name or not value.numel()
                    or any(pattern in name for pattern in LONG_INDEX_PATTERNS)):
                continue
            if int(value.max()) > torch.iinfo(torch.int32).max or int(value.min()) < torch.iinfo(torch.int32).min:
                continue
            module._buffers[name] = value.to(torch.int32)
            count += 1
            saved += value.numel() * 4
    return count, saved
