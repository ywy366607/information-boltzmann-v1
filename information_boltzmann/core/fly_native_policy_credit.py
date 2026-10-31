"""Delayed policy score credit to existing fly physiology and KC->MBON synapses.

The differentiable trajectory is the real window's BPTT replay. Only its score
derivatives are archived; rewards and statistical consequence beliefs are not
differentiated. The replay uses the same spike surrogate as ordinary fly BPTT.
Local DAN opening-state VJPs include their carried conditional chemistry
history. Recurrent spike-mediated history is not full RTRL.
"""
import math
from contextlib import contextmanager
from numbers import Integral

import torch
import torch.nn.functional as F


FORMAT = 'native-policy-score-v1'


def prepare_native_policy_backward(model):
    """Keep compiled tick buffers alive for score VJPs plus ordinary backward.

    PyTorch AOT buffer donation assumes a single backward. Native eligibility
    needs retained-graph VJPs. Keep compilation enabled, but recompile existing
    tick wrappers once with donation disabled before this path's first forward.
    This is process-local execution configuration, not physiological state.
    """
    if getattr(model, '_native_policy_retained_graph', False):
        return
    import torch._functorch.config as config
    if hasattr(config, 'donated_buffer'):
        config.donated_buffer = False
    for name in ('_compiled_coefficients', '_compiled_finish'):
        if hasattr(model, name):
            delattr(model, name)
    model._native_policy_retained_graph = True


def native_parameter_names(model):
    """Existing native parameters only; no port, decoder, edge list or homeostat."""
    from .fly_bptt_learning import (
        CLASS_PARAMETER_NAMES, COMPARTMENT_PARAMETER_NAMES,
        DOPAMINE_PARAMETER_NAMES, STP_PARAMETER_NAMES,
        AMINE_PARAMETER_NAMES,
        CX_PARAMETER_NAMES,
        MB_CONSOLIDATION_PARAMETER_NAMES,
        MB_TERMINAL_PARAMETER_NAMES,
        APL_CABLE_PARAMETER_NAMES,
        APL_LOCAL_RELEASE_PARAMETER_NAMES,
        KC_TERMINAL_PARAMETER_NAMES, KC_SITE_STP_PARAMETER_NAMES, KC_DOPAMINE_PARAMETER_NAMES, KC_LATERAL_PARAMETER_NAMES, DAN_RELEASE_PARAMETER_NAMES, SENSORY_TERMINAL_PARAMETER_NAMES,
        BODY_LOOP_PARAMETER_NAMES,
        AL_PARAMETER_NAMES,
        VISUAL_PARAMETER_NAMES,
        BODY_ACTUATOR_PARAMETER_NAMES, ENDOCRINE_PARAMETER_NAMES,
    )
    from .fly_carried_credit import COMP_PARAMS
    allowed = (set(CLASS_PARAMETER_NAMES) | set(COMPARTMENT_PARAMETER_NAMES)
               | set(COMP_PARAMS) | set(DOPAMINE_PARAMETER_NAMES)
               | set(AMINE_PARAMETER_NAMES)
               | set(CX_PARAMETER_NAMES)
               | set(MB_CONSOLIDATION_PARAMETER_NAMES)
               | set(MB_TERMINAL_PARAMETER_NAMES)
               | set(APL_CABLE_PARAMETER_NAMES)
               | set(APL_LOCAL_RELEASE_PARAMETER_NAMES)
               | set(KC_TERMINAL_PARAMETER_NAMES) | set(KC_SITE_STP_PARAMETER_NAMES) | set(KC_DOPAMINE_PARAMETER_NAMES) | set(KC_LATERAL_PARAMETER_NAMES) | set(DAN_RELEASE_PARAMETER_NAMES) | set(SENSORY_TERMINAL_PARAMETER_NAMES)
               | set(BODY_LOOP_PARAMETER_NAMES)
               | set(AL_PARAMETER_NAMES)
               | set(VISUAL_PARAMETER_NAMES)
               | set(BODY_ACTUATOR_PARAMETER_NAMES) | set(ENDOCRINE_PARAMETER_NAMES)
               | set(STP_PARAMETER_NAMES)
               | {'mbon_kappa', 'log_g_e', 'log_g_i', 'log_tau_s',
                  'log_graded_gain', 'log_kc_release'})
    if not getattr(model, 'body_proxy_compatibility', False):
        allowed -= set(BODY_LOOP_PARAMETER_NAMES) | set(BODY_ACTUATOR_PARAMETER_NAMES)
    if getattr(model, 'mb_pooled_quarantined', False):
        allowed -= set(MB_CONSOLIDATION_PARAMETER_NAMES)
    from .fly_native_runtime import generated_parameter_names
    allowed.update(generated_parameter_names(model))
    return tuple(name for name, parameter in model.named_parameters()
                 if name in allowed and parameter.requires_grad)


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{label} must be an integer >= {minimum}')
    return int(value)


def _finite(tensor, label):
    # isfinite creates full-size temporaries on either device. Split views,
    # including noncontiguous heads, without flattening/copying a whole score.
    blocks = [tensor.detach()]
    while blocks:
        block = blocks.pop()
        if block.numel() > 1 << 20:
            dim = max(range(block.ndim), key=lambda i: block.shape[i])
            width = max(1, (1 << 20) // (block.numel() // block.shape[dim]))
            blocks.extend(reversed(block.split(width, dim=dim)))
            continue
        if not bool(torch.isfinite(block).all()):
            raise ValueError(f'Non-finite {label} in native policy credit')


@contextmanager
def _auxiliary_backward(model):
    """Recomputation must not count extra life or leak edge gradients into CE."""
    from .triton_synapse import direct_edge_grad
    missing = object()
    attributes = {}
    for name in ('homeo_spikes', 'homeo_ticks', 'step_grad_clips'):
        value = getattr(model, name, missing)
        attributes[name] = (value, value.detach().clone() if torch.is_tensor(value) else value)
    # The fused large-edge kernel writes straight into leaf.grad in this mode.
    # Give it scratch .grad slots, keeping the pre-existing CE slots by reference.
    edges = [(p, p.grad) for name, p in model.named_parameters()
             if name in ('edge_weight', 'edge_weight_e', 'edge_weight_i')]
    for parameter, _ in edges:
        parameter.grad = None
    try:
        with direct_edge_grad():
            yield
    finally:
        with torch.no_grad():
            for parameter, gradient in edges:
                parameter.grad = gradient
            for name, (original, saved) in attributes.items():
                if original is missing:
                    if hasattr(model, name):
                        delattr(model, name)
                elif torch.is_tensor(original):
                    original.copy_(saved)
                    setattr(model, name, original)
                else:
                    setattr(model, name, original)


def _native_path(score, parameter_ids):
    """Graph connectivity diagnostic without another reverse-mode traversal."""
    pending, seen = [score.grad_fn], set()
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        variable = getattr(node, 'variable', None)
        if variable is not None and id(variable) in parameter_ids:
            return True
        pending.extend(parent for parent, _ in node.next_functions)
    return False


def native_policy_credit(learner, events, hazards, reads, intervals, *, raw_logits=False,
                         include_policy_heads=False, motor_availability=None):
    """Return ``(J, diagnostics)`` for the actual window, without applying it.

    ``hazards[T]`` and ``reads[T, D]`` retain their real replay graph. Events are
    chronological and tick offsets are local to this window. A ``clock`` event
    advances all physical ticks, including self-hearing and quiet gaps. Timing
    and content events follow that advance; ``tail_ticks`` discounts their score
    from its actual commit to the end of the already advanced interval.

    Timing: start, length, first (relative first commit, or None for survival).
    Content: tick, word, temperature, optional detached logit_bias in nats ADDED
    AFTER temperature scaling, exactly as in the real language sampling policy.
    response_begin snapshots E; response_end adds advantage * snapshot once.

    Persistent E and a pending response snapshot live in consequence_state;
    this call's G is returned as a linear surrogate and is never left pending.
    The caller divides J by its own window normalization, exactly once.
    """
    model = learner.model
    if not torch.is_tensor(hazards) or hazards.ndim != 1:
        raise ValueError('Native policy hazards must have shape [T]')
    if not torch.is_tensor(reads) or reads.ndim != 2:
        raise ValueError('Native policy reads must have shape [T, D]')
    lengths = [_integer(n, 'interval', 1) for n in intervals]
    total_ticks = sum(lengths)
    if len(hazards) != total_ticks or len(reads) != total_ticks:
        raise ValueError('Native policy trajectory and actual intervals disagree')
    _finite(hazards, 'hazards')
    _finite(reads, 'reads')
    if getattr(model, 'native_motor_port_enabled', False):
        if motor_availability is None or motor_availability.shape != (total_ticks,):
            raise ValueError('Native peripheral score needs its actual terminal packet clock')
        if not bool(((motor_availability == 0) | (motor_availability == 1)).all()):
            raise ValueError('Native terminal availability must be an actual event tape')
    from .fly_native_motor_port import action_logits
    if not raw_logits and bool(((hazards < 0) | (hazards > 1)).any()):
        raise ValueError('Native policy hazards must lie in [0, 1]')
    events = list(events)
    clock_ticks = sum(_integer(e.get('ticks'), 'clock ticks')
                      for e in events if e.get('kind') == 'clock')
    if clock_ticks != total_ticks:
        raise ValueError('Native policy events must account for every physical tick')

    all_parameters = dict(model.named_parameters())
    names = native_parameter_names(model)
    if include_policy_heads:
        names += tuple(name for name, parameter in model.named_parameters()
                       if parameter.requires_grad and name not in names and
                       (name in ('ai_w', 'ai_b') or name.startswith(('decoder.', 'read_norm.', 'motor_port_'))))
    parameters = {name: all_parameters[name] for name in names}
    archive_cpu = bool(getattr(learner, 'native_credit_cpu', False))
    raw_clock = getattr(model, 'comp_logit_trace', None)
    if not torch.is_tensor(raw_clock) or raw_clock.numel() != 1:
        raise ValueError('Native credit requires the scalar KC eligibility clock')
    _finite(raw_clock.detach(), 'KC eligibility clock')
    rho = float(torch.sigmoid(raw_clock.detach()))

    consequence = getattr(learner, 'consequence_state', None)
    if consequence is not None and not isinstance(consequence, dict):
        raise ValueError('Consequence state must be a dictionary')
    previous = (consequence or {}).get('native_credit')
    if previous is not None and not isinstance(previous, dict):
        raise ValueError('Native credit state must be a dictionary')
    previous = previous or {}
    if previous.get('format', FORMAT) != FORMAT:
        raise ValueError('Native policy credit schema needs explicit migration')

    def clone_trace(trace):
        if not isinstance(trace, dict):
            raise ValueError('Native score trace must be a parameter dictionary')
        result = {}
        for name, value in trace.items():
            if name not in parameters:
                raise ValueError(f'Archived native parameter is unavailable or frozen: {name}')
            parameter = parameters[name]
            if not torch.is_tensor(value) or value.shape != parameter.shape:
                raise ValueError(f'Native score shape changed: {name}')
            _finite(value, f'archived score {name}')
            result[name] = value.detach().to(device='cpu' if archive_cpu else parameter.device,
                                             dtype=parameter.dtype).clone()
        return result

    # Work transactionally: malformed events/non-finite derivatives cannot partly
    # mutate an individual's saved credit or unrelated consequence state.
    eligibility = clone_trace(previous.get('E', {}))
    response = previous.get('response_score')
    response = clone_trace(response) if response is not None else None
    deferred = {key: clone_trace(trace) for key, trace in previous.get('deferred_response_scores', {}).items()}
    deferred_expr = {}
    accumulated = {}
    # Linear differentiation lets every event share at most three reverse passes:
    # final historical E, an unfinished response archive, and this window's G.
    # This avoids one checkpoint recomputation per sampled policy event.
    eligibility_expr = response_expr = credit_expr = None
    diag = {'native_parameter_names': list(names), 'native_parameters': len(names),
            'native_parameter_elements': sum(p.numel() for p in parameters.values()),
            'native_clock_ticks': clock_ticks, 'native_score_events': 0,
            'native_score_connected_events': 0, 'native_score_no_graph_events': 0,
            'native_score_unused_events': 0, 'native_score_gradients': 0,
            'native_response_settlements': 0, 'native_unmatched_response_ends': 0,
            'native_reverse_passes': 0,
            'native_timing_connected': 0, 'native_content_connected': 0,
            'native_auxiliary_homeostasis_preserved': True,
            'native_auxiliary_edge_gradients_discarded': True}
    parameter_ids = {id(p) for p in parameters.values()}
    # The action's score also depends on past material effects in the opening
    # slow states. Contract their VJPs here; a later CE backward cannot recover
    # the derivatives already archived as native action eligibility.
    from .fly_dan_release import DAN_RELEASE_STATE_FIELDS
    from .fly_dan_credit import carried_gradients
    opening = getattr(learner, '_credit', None)
    history_states = set(DAN_RELEASE_STATE_FIELDS)
    if getattr(model, 'dan_receiver_credit_enabled', False):
        history_states.add('dopamine')
    opening_leaves = ({n: t for n, t in opening[0].items() if n in history_states}
                      if opening and getattr(model, 'dan_carried_credit_enabled', False) else {})
    parameter_ids.update(id(t) for t in opening_leaves.values())

    def add_score(score, tail_ticks, kind):
        nonlocal eligibility_expr
        diag['native_score_events'] += 1
        if not score.requires_grad or not parameters:
            diag['native_score_no_graph_events'] += 1
            return
        if not _native_path(score, parameter_ids):
            diag['native_score_unused_events'] += 1
            return
        diag['native_score_connected_events'] += 1
        diag[f'native_{kind}_connected'] += 1
        term = score * (rho ** tail_ticks)
        eligibility_expr = term if eligibility_expr is None else eligibility_expr + term

    elapsed = 0

    for event in events:
        kind = event.get('kind')
        if kind == 'clock':
            ticks = _integer(event.get('ticks'), 'clock ticks')
            elapsed += ticks
            if eligibility_expr is not None:
                eligibility_expr = eligibility_expr * (rho ** ticks)
            with torch.no_grad():
                for trace in eligibility.values():
                    trace.mul_(rho ** ticks)
        elif kind in ('timing', 'joint_timing'):
            start = _integer(event.get('start'), 'timing start')
            length = _integer(event.get('length'), 'timing length', 1)
            if start + length > total_ticks:
                raise ValueError('Timing decision lies outside the real window')
            if start + length > elapsed:
                raise ValueError('Timing score precedes its actual physical clock')
            first = event.get('first')
            if first is not None:
                first = _integer(first, 'first commit')
                if first >= length:
                    raise ValueError('First commit lies outside its decision interval')
            tail = _integer(event.get('tail_ticks', 0), 'timing tail_ticks')
            # Float64 log probabilities keep finite numerical values at saturated
            # float32 hazards without inventing extra sampling probability.
            if kind == 'joint_timing':
                if not raw_logits:
                    raise ValueError('Joint timing requires raw native motor logits')
                stop = length if first is None else first + 1
                value = event['word_value'].detach().to(reads)
                if value.shape != (model.decoder.out_features,):
                    raise ValueError('Joint word values must retain the complete vocabulary measure')
                temperature = float(event['temperature'])
                if not math.isfinite(temperature) or temperature <= 0:
                    raise ValueError('Joint temperature must be positive')
                lp = F.log_softmax(action_logits(model, reads[start:start + stop]).float()
                                   / temperature, -1)
                partition = torch.logsumexp(lp + value, -1) - float(event['wait_value'])
                z = hazards[start:start + stop].double() + partition.double()
                available = (motor_availability[start:start + stop].bool() if motor_availability is not None
                             else torch.ones_like(z, dtype=torch.bool))
                if first is not None and not bool(available[-1]):
                    raise ValueError('A native commit cannot precede actual terminal release')
                score = (F.logsigmoid(-z[available]).sum() if first is None else
                         F.logsigmoid(-z[:-1][available[:-1]]).sum() + F.logsigmoid(z[-1]))
            else:
                if raw_logits:
                    raise ValueError('Raw logits cannot be replayed as old hazard scores')
                h = hazards[start:start + length].double()
                h = h.clamp(torch.finfo(h.dtype).eps, 1 - torch.finfo(h.dtype).eps)
                score = (torch.log1p(-h).sum() if first is None else
                         torch.log1p(-h[:first]).sum() + torch.log(h[first]))
            add_score(score, tail, 'timing')
        elif kind == 'content':
            tick = _integer(event.get('tick'), 'content tick')
            if tick >= total_ticks:
                raise ValueError('Content commit lies outside the real window')
            if tick >= elapsed:
                raise ValueError('Content score precedes its actual physical clock')
            if motor_availability is not None and not bool(motor_availability[tick]):
                raise ValueError('A word cannot be executed without a native motor packet')
            word = _integer(event.get('word'), 'content word')
            temperature = float(event.get('temperature', 1.))
            if not math.isfinite(temperature):
                raise ValueError('Non-finite content temperature')
            logits = action_logits(model, reads[tick:tick + 1]).float()
            if word >= logits.shape[-1]:
                raise ValueError('Sampled word lies outside the decoder vocabulary')
            logits = logits / max(temperature, 1e-6)
            bias = event.get('logit_bias')
            if bias is not None:
                if not torch.is_tensor(bias) or bias.shape not in (logits.shape, logits.shape[1:]):
                    raise ValueError('Content logit bias must match the vocabulary')
                _finite(bias.detach(), 'content logit bias')
                logits = logits + bias.detach().to(logits)
            _finite(logits, 'content logits')
            score = F.log_softmax(logits, -1)[0, word]
            add_score(score, _integer(event.get('tail_ticks', 0), 'content tail_ticks'), 'content')
        elif kind == 'response_begin':
            if response is not None:
                raise ValueError('A native response score is already pending')
            response = {name: trace.clone() for name, trace in eligibility.items()}
            response_expr = eligibility_expr
        elif kind in ('response_end', 'deferred_response_end'):
            advantage = float(event.get('advantage'))
            if not math.isfinite(advantage):
                raise ValueError('Non-finite native response advantage')
            if kind == 'deferred_response_end':
                key = str(event['key'])
                if key not in deferred:
                    raise ValueError('Missing quarantined response score')
                settled, settled_expr = deferred.pop(key), deferred_expr.pop(key, None)
            else:
                settled, settled_expr = response, response_expr
                key = event.get('defer_key')
                if key is not None:
                    key = str(key)
                    if key in deferred:
                        raise ValueError('Duplicate quarantined response score')
                    deferred[key] = {n: t.clone() for n, t in (response or {}).items()}
                    deferred_expr[key] = response_expr
                response = response_expr = None
            if settled is None:
                # A legacy response can finish after entering native credit. Its
                # unavailable past derivatives cannot be fabricated or replayed.
                diag['native_unmatched_response_ends'] += 1
                continue
            with torch.no_grad():
                for name, trace in settled.items():
                    accumulated.setdefault(name, torch.zeros_like(trace)).add_(trace, alpha=advantage)
                    _finite(accumulated[name], f'window credit {name}')
            if settled_expr is not None:
                term = settled_expr * advantage
                credit_expr = term if credit_expr is None else credit_expr + term
            diag['native_response_settlements'] += 1
        elif kind == 'trial_end':
            if response is not None:
                raise ValueError('Settle the actual response before ending its causal trial')
            eligibility = {}
            eligibility_expr = None
        else:
            raise ValueError(f'Unknown native policy credit event: {kind!r}')

    def differentiate(expression):
        if expression is None:
            return {}
        with _auxiliary_backward(model):
            gradients = torch.autograd.grad(expression, tuple(parameters.values()) + tuple(opening_leaves.values()),
                                            retain_graph=True, allow_unused=True)
        diag['native_reverse_passes'] += 1
        result = {}
        for name, gradient in zip(parameters, gradients):
            if gradient is not None:
                _finite(gradient, f'score derivative {name}')
                result[name] = gradient.detach().to('cpu') if archive_cpu else gradient.detach()
                diag['native_score_gradients'] += 1
        if opening_leaves:
            carried = carried_gradients(model, dict(zip(opening_leaves, gradients[len(parameters):])), opening[1])
            for name, gradient in carried.items():
                if name not in parameters:
                    continue
                _finite(gradient, f'carried action score {name}')
                value = gradient.detach().to('cpu') if archive_cpu else gradient.detach()
                result[name] = result.get(name, torch.zeros_like(value)) + value
        return result

    def merge(destination, gradient):
        with torch.no_grad():
            for name, value in gradient.items():
                if name in destination:
                    destination[name].add_(value)
                else:
                    destination[name] = value.clone()
                _finite(destination[name], f'accumulated native score {name}')

    merge(eligibility, differentiate(eligibility_expr))
    if response is not None:
        merge(response, differentiate(response_expr))
    for key, expression in deferred_expr.items():
        merge(deferred[key], differentiate(expression))
    merge(accumulated, differentiate(credit_expr))
    surrogate = sum(((gradient.to(parameters[name]) * parameters[name]).sum()
                     for name, gradient in accumulated.items()), hazards.new_zeros(()))
    _finite(surrogate.detach(), 'linear surrogate')
    updated = {**previous, 'format': FORMAT, 'E': eligibility,
               'deferred_response_scores': deferred,
               'clock_ticks': int(previous.get('clock_ticks', 0)) + clock_ticks}
    if response is None:
        updated.pop('response_score', None)
    else:
        updated['response_score'] = response
    if consequence is None:
        learner.consequence_state = {}
    learner.consequence_state['native_credit'] = updated
    diag.update(native_response_pending=response is not None,
                native_eligibility_bytes=sum(t.numel() * t.element_size() for t in eligibility.values()),
                native_eligibility_device='cpu' if archive_cpu else str(hazards.device),
                native_eligibility_gpu_bytes=sum(t.numel() * t.element_size() for t in eligibility.values()
                                                 if t.is_cuda),
                native_window_credit_norm=math.sqrt(sum(float(g.double().square().sum())
                                                       for g in accumulated.values())),
                native_total_clock_ticks=updated['clock_ticks'])
    return surrogate, diag
