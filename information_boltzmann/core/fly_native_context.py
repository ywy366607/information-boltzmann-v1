"""Continuous context inference and amortization through existing DAN/phi dynamics.

The latent is external context CODED by compartment log efficacy. Saved physical
phi itself is exactly known. Gaussian directional filtering and local additive
quiet drift are explicit approximations, not a complete social world model.
"""
import copy

import torch
import torch.nn.functional as F

from .fly_context_filter import DirectionalContextFilter, categorical_information
from .fly_native_generative import _clone_state, _preserve_rollout
from .fly_native_generative_control import NativeGenerativeControl, _cpu
from .fly_native_policy_credit import native_parameter_names, prepare_native_policy_backward


FORMAT = 'native-continuous-context-control-v1'


def compartment_chart(model, physical):
    posts, groups = torch.unique(model.comp_post.long(), sorted=True, return_inverse=True)
    counts = torch.bincount(groups, minlength=len(posts)).to(physical.h).clamp_min(1)
    if getattr(model, 'kc_contact_memory_enabled', False) and getattr(model, 'kc_contact_read_enabled', True):
        from .fly_kc_contact_memory import contact_state
        phi = contact_state(model, physical)[0][0]
        groups = groups.index_select(0, model.kcm_edge)
        weight = model.kcm_fraction
    else:
        phi, weight = physical.comp_memory[0], 1.
    efficacy = (1. + phi).clamp_min(torch.finfo(physical.h.dtype).eps)
    field = efficacy.log().new_zeros(len(posts)).index_add(0, groups, efficacy.log()*weight) / counts
    return field, groups, posts


def apply_context(model, physical, field):
    center, groups, _ = compartment_chart(model, physical)
    offset = torch.as_tensor(field, device=center.device, dtype=center.dtype) - center
    # An invertible positive-efficacy chart; no clipping of quadrature nodes.
    extension = {}
    if getattr(model, 'kc_contact_memory_enabled', False) and getattr(model, 'kc_contact_read_enabled', True):
        from .fly_kc_contact_memory import contact_state, KC_CONTACT_STATE_FIELDS
        values = contact_state(model, physical)
        extension = dict(zip(KC_CONTACT_STATE_FIELDS, values))
        name, memory = 'kc_contact_memory', values[0]
    else:
        name, memory = 'comp_memory', physical.comp_memory
    efficacy = (1. + memory).clamp_min(torch.finfo(center.dtype).eps)
    phi = efficacy * offset[groups].exp()[None, :] - 1.
    if not bool(torch.isfinite(phi).all()):
        raise ValueError('Non-finite native context efficacy')
    return type(physical)(**{**physical.state_dict(), **extension, name: phi})


@torch.no_grad()
def context_rollout(learner, physical, token, ticks, *, source='external', eyes=None,
                    eye_gain=1., voice=False, reward=False, quiet=(), sensory_on=True):
    """One native counterfactual, preserving actual parameters/state/RNG/counters."""
    from . import fly_bptt_learning as fl
    from .fly_active_inference import motor_rest
    from .fly_mother_world import sensory_segments
    if ticks < 1 or (eyes is not None and len(eyes) != ticks):
        raise ValueError('Context rollout must follow the complete actual interval')
    model = learner.model
    with _preserve_rollout(model):
        current = _clone_state(physical)
        current = fl.prepare_sensory_expectation(model, current, int(token), source)
        options = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(),
                       conductance_gains=model.get_conductance_gains(), alif_params=model.get_alif_params(),
                       stp_params=model.get_stp_params())
        if getattr(model, 'dopamine_sensitivity', None) is not None:
            options['dopamine_s_class'] = fl.dopamine_class_sensitivity(model, current.h)
        muscles = motor_rest(model, current.h)
        command = torch.tensor([int(token)], device=current.h.device)
        segment = int(getattr(learner, 'stream_segment_ticks', 16) or 16)
        for start, length, on in sensory_segments(ticks, segment, quiet):
            forced = eyes[start:start + length] if eyes is not None else eye_gain
            out = fl.advance_fly_token_stream(
                model, current, command, [length], ticks=length,
                hold_input=learner.hold_input, writer_baseline_clock=learner.writer_baseline_clock,
                fresh=start == 0, sensory_on=bool(sensory_on and on), voice=voice,
                executed_action=bool(source == 'self_hearing' and sensory_on and on),
                reward_ticks=fl.reward_ticks_left(model, reward, start), ai_g=muscles,
                ai_tape=[False] * length, eye_forced=forced, update_expectation=False, **options)
            current, latent, muscles = out[0], out[1], out[-1]
        lp = F.log_softmax(model.decoder(model.read_norm(latent)).float(), -1)[0]
    return current, lp


class NativeContextWorld:
    def __init__(self, learner, control):
        self.learner, self.model, self.control = learner, learner.model, control
        field, _, posts = compartment_chart(self.model, learner.state)
        state = getattr(learner, 'native_context_state', None)
        if state is None:
            f = DirectionalContextFilter(dimension=len(posts),
                initial_variance=float(getattr(learner, 'context_initial_variance', 1.)),
                process_variance=float(getattr(learner, 'context_process_variance', 1e-4)))
            f.state['mean'] = field.detach().cpu().double()
            state = {'format': FORMAT, 'posts': posts.cpu(), 'filter': f.state,
                     'published_covariance': f.covariance.clone(),
                     'published_process_variance': f.state['process_variance'],
                     'published_clock': control['clock_ticks'], 'pending_teaching': None,
                     'direction_rng': torch.get_rng_state().clone(), 'self_excluded': 0,
                     'teaching_jobs': 0}
            learner.native_context_state = state
        if state.get('format') != FORMAT or not torch.equal(state['posts'], posts.cpu()):
            raise ValueError('Context topology/schema changed; migrate explicitly')
        self.state = state
        self.filter = DirectionalContextFilter(state['filter'], dimension=len(posts))
        self.width = len(posts)

    def direction(self):
        generator = torch.Generator(device='cpu')
        generator.set_state(self.state['direction_rng'])
        value = torch.randn(self.width, generator=generator, dtype=torch.float64)
        self.state['direction_rng'] = generator.get_state()
        return value / value.norm()

    def actor_nodes(self, physical, ticks, direction):
        # Ideal teacher updates never narrow the actor before real sensation.
        state = self.filter.state_dict()
        state.update(mean=compartment_chart(self.model, physical)[0].detach().cpu().double(),
                     covariance=self.state['published_covariance'].clone(), pending=None,
                     process_variance=self.state['published_process_variance'])
        age = self.control['clock_ticks'] - self.state['published_clock']
        preview = DirectionalContextFilter(state, dimension=self.width)
        return preview.begin_transition(state['mean'], int(ticks + age), direction)

    def likelihood(self, before, nodes, token, ticks, **kwargs):
        outputs = []
        for node in nodes:
            _, lp = context_rollout(self.learner, apply_context(self.model, before, node), token, ticks, **kwargs)
            outputs.append(lp)
        return torch.stack(outputs)

    def command_values(self, physical, candidate_ids, ticks, eye_gain=1.):
        direction = self.control['_planning_direction']
        nodes = self.actor_nodes(physical, ticks, direction)
        wait = self.likelihood(physical, nodes, 0, ticks, sensory_on=False, eye_gain=eye_gain)
        words = [self.likelihood(physical, nodes, word, ticks, source='self_hearing', eye_gain=eye_gain)
                 for word in candidate_ids]
        return {'wait': categorical_information(wait)['expected_information'],
                'words': torch.tensor([categorical_information(lp)['expected_information'] for lp in words])}


class NativeContextControl(NativeGenerativeControl):
    def __init__(self, learner):
        self.learner, self.model = learner, learner.model
        if not learner.hold_input:
            raise ValueError('Native context currently requires held input for its sensory-exposure contract')
        if learner.mother_action_clock != 'commit' or getattr(learner, 'eye_binary', False):
            raise ValueError('Native context needs commit clock and continuous senses')
        if getattr(learner, 'pending_speak_credit', None) is not None:
            raise ValueError('Drain pending legacy head credit before context migration')
        prepare_native_policy_backward(self.model)
        old = getattr(learner, 'consequence_state', {})
        if old.get('format') != FORMAT:
            if old.get('response') is not None or old.get('native_credit', {}).get('response_score') is not None:
                raise ValueError('Drain the actual pending response before context migration')
            clock = old.get('native_credit', {}).get('clock_ticks', 0)
            learner.consequence_state = {'format': FORMAT, 'native_credit': {
                'format': 'native-policy-score-v1', 'E': {}, 'clock_ticks': clock},
                'legacy': _cpu(old)}
        consequence = learner.consequence_state
        self.state = consequence.setdefault('control', {'source': old.get('source', 'unknown'), 'in_child_turn': False,
            'trial_active': False, 'commands': 0, 'response': None,
            'clock_ticks': consequence['native_credit']['clock_ticks'], 'settlements': 0, 'cancelled': 0})
        self.world_model = NativeContextWorld(learner, self.state)
        self.native_events, self.timing_biases, self.decisions = [], [], []
        self.reports, self.plans, self.teaching = [], [], []
        self.decision_bounds, self.word_bias, self.wait_value = (0, 0), None, 0.
        self.temperature, self.enabled, self.pending_interval = 1., False, None

    def begin_interval(self, physical, token, voice, reward):
        if self.pending_interval is not None:
            raise ValueError('A native context interval is already pending')
        self.state['_planning_direction'] = self.world_model.direction()
        self.pending_interval = (_clone_state(physical), int(token), bool(voice), bool(reward),
                                 self.state['source'], self.state['_planning_direction'].clone())

    def end_response(self):
        if self.state['response'] is not None:
            gain = self.state['response']['gain']
            event = {'kind': 'response_end', 'advantage': gain}
            job = self.world_model.state['pending_teaching']
            if job is not None and job.get('response_id') == self.state['response']['id']:
                # The last arrival has not yet been sensed. Quarantine its
                # credit across the optimizer/window boundary until it is.
                event.update(advantage=gain - job['information'], defer_key=job['id'])
                job['deferred_credit'] = True
            self.native_events.append(event)
            self.state['settlements'] += 1
        if self.state['trial_active']:
            self.native_events.append({'kind': 'trial_end'})
        self.state.update(response=None, trial_active=False, commands=0)

    def arrive(self, physical, word, source, ticks, quiet_ranges, eyes, end):
        before, incoming, voice, reward, held_source, direction = self.pending_interval
        world, f = self.world_model, self.world_model.filter
        field_before = compartment_chart(self.model, before)[0].detach().cpu().double()
        quiet_state, _ = context_rollout(self.learner, before, incoming, ticks,
                                        sensory_on=False, voice=False, reward=False)
        quiet_field = compartment_chart(self.model, quiet_state)[0].detach().cpu().double()
        drift = quiet_field - field_before
        job = world.state['pending_teaching']
        teaching = None
        if job is not None:
            if incoming != job['token'] or held_source != job['source']:
                raise ValueError('Native recognition job does not match the actually held input')
            from .fly_mother_world import sensory_segments
            eye = torch.as_tensor(eyes).reshape(-1)
            sensed = any(on and bool((eye[start:start + length] > 0).any())
                         for start, length, on in sensory_segments(ticks, ticks, quiet_ranges))
            if job.get('deferred_credit'):
                self.native_events.append({'kind': 'deferred_response_end', 'key': job['id'],
                                           'advantage': job['information'] if sensed else 0.})
            elif not sensed and job.get('response_id') is not None:
                if self.state['response'] is None or self.state['response']['id'] != job['response_id']:
                    raise ValueError('Unsensed evidence lost its causal response owner')
                self.state['response']['gain'] -= job['information']
            # Hidden input cannot narrow future uncertainty, even indirectly
            # through a later observed word. Undo the ideal teacher conditioning.
            if not sensed:
                f = DirectionalContextFilter(copy.deepcopy(job['unconditioned_filter']), dimension=world.width)
                f.discard_unobserved()
                world.filter = f
                world.state['filter'] = f.state
                world.state['unsensed_jobs'] = world.state.get('unsensed_jobs', 0) + 1
            # Publish uncertainty only after the corresponding input was sensed.
            delay = self.state['clock_ticks'] - job['clock']
            transported_covariance = (job['covariance'] + torch.eye(world.width,
                dtype=torch.float64) * (job['process_variance'] * delay))
            if sensed:
                teaching = {'target': job['mean'] + drift, 'covariance': transported_covariance,
                            'source': held_source, 'token': incoming}
                world.state['last_recognition_gap'] = float((compartment_chart(self.model, physical)[0]
                    .detach().cpu().double() - teaching['target']).norm())
                world.state['published_covariance'] = transported_covariance.clone()
                world.state['published_process_variance'] = job['process_variance']
                world.state['published_clock'] = self.state['clock_ticks']
                world.state['teaching_jobs'] += 1
            world.state['pending_teaching'] = None
        self.teaching.append(teaching)
        # Known native quiet relaxation is additive chart transport, not Q evidence.
        prior_mean = f.mean.clone() + drift
        nodes = f.begin_transition(prior_mean, int(ticks), direction)
        if source == 'self_hearing':
            f.discard_unobserved()
            world.state['self_excluded'] += 1
            report = {'updated': False, 'source': source, 'realized_information': 0.}
        else:
            predictions = world.likelihood(before, nodes - drift[None, :], incoming, ticks,
                source=held_source, eyes=eyes, voice=voice, reward=reward, quiet=quiet_ranges)
            forecast = f.forecast(predictions)
            causal = self.state['trial_active'] and (
                (source == 'mother_reply' and self.state['commands'] > 0) or
                (source == 'mother_call' and self.state['commands'] == 0))
            if self.state['trial_active'] and not causal:
                self.end_response()
                self.state['cancelled'] += 1
            if causal and self.state['response'] is None:
                self.state['response'] = {'gain': 0., 'source': source, 'id': self.state['clock_ticks']}
                self.native_events.append({'kind': 'response_begin'})
            unconditioned = f.state_dict()
            report = f.observe(int(word))
            report = {k: v for k, v in report.items() if k not in ('mean', 'covariance', 'posterior_weights')}
            report.update(source=source, updated=True, expected_information=forecast['expected_information'],
                          credited_to_trial=causal,
                          pre_assimilation_gap=float((compartment_chart(self.model, physical)[0].detach().cpu().double()
                                                          - f.mean).norm()))
            world.state['pending_teaching'] = {'token': int(word), 'source': source,
                'mean': f.mean.clone(), 'covariance': f.covariance.clone(),
                'process_variance': f.state['process_variance'], 'clock': self.state['clock_ticks'],
                'id': str(self.state['clock_ticks']), 'information': report['realized_information'],
                'response_id': self.state['response']['id'] if causal else None,
                'unconditioned_filter': unconditioned}
            if causal:
                self.state['response']['gain'] += report['realized_information']
                if end:
                    self.end_response()
        self.reports.append(report)
        self.pending_interval = None

    def finish(self):
        if self.pending_interval is not None:
            raise ValueError('Incomplete context interval cannot be saved as complete')
        f = self.world_model.filter
        self.learner.last_epistemic = {'source': 'native_continuous_context', 'policy_actuated': True,
            'external_updates': f.state['updates'], 'decisions': len(self.decisions),
            'commits': sum(self.decisions), 'settlements': self.state['settlements'],
            'plans': self.plans, 'evidence': self.reports, 'count_eye_reward': False,
            'fixed_speech_kl': False, 'context_variance_trace': float(f.covariance.trace()),
            'context_process_variance': f.state['process_variance'],
            'native_teacher_jobs': self.world_model.state['teaching_jobs'],
            'last_recognition_gap': self.world_model.state.get('last_recognition_gap'),
            'unsensed_jobs': self.world_model.state.get('unsensed_jobs', 0),
            'filter_approximation': 'directional_gaussian_local_additive_quiet_drift',
            'teacher_and_native_posterior_distinct': True}


def native_context_teaching(model, states, jobs):
    """Teach existing native dynamics to encode the detached Bayesian context.

    No new DAN current/score is injected. Only native parameters get this VJP;
    decoder/ports cannot shrink the fixed log-efficacy chart to satisfy it.
    """
    from .fly_native_policy_credit import _auxiliary_backward
    if not states or len(states) != len(jobs):
        raise ValueError('Native teaching must match every actual sensory interval')
    losses = []
    for state, job in zip(states, jobs):
        if job is None:
            continue
        field = compartment_chart(model, state)[0]
        target = job['target'].to(field)
        covariance = job['covariance'].to(device=field.device, dtype=torch.float64)
        delta = field - target
        eigenvalues, vectors = torch.linalg.eigh(covariance)
        positive = eigenvalues > torch.finfo(torch.float64).eps * max(1., float(eigenvalues.max()))
        if bool((~positive).any()) and float((vectors[:, ~positive].T @ delta.detach().double()).norm()) > 1e-7:
            raise ValueError('Native mean lies outside the support of the singular teacher covariance')
        reciprocal = torch.where(positive, eigenvalues.clamp_min(torch.finfo(torch.float64).tiny).reciprocal(), 0.)
        inverse = ((vectors * reciprocal[None, :]) @ vectors.T).to(field)
        losses.append(.5 * delta @ inverse @ delta)
    zero = states[-1].h.new_zeros(())
    if not losses:
        return zero, {'context_teacher_jobs': 0, 'context_teacher_kl': 0.}
    loss = torch.stack(losses).sum() / max(1, len(states))
    params = {n: p for n, p in model.named_parameters()
              if n in native_parameter_names(model) and p.requires_grad}
    if not loss.requires_grad or not params:
        return zero, {'context_teacher_jobs': len(losses), 'context_teacher_kl': float(loss.detach()),
                      'context_teacher_gradient_norms': {}, 'context_teacher_frozen': True}
    with _auxiliary_backward(model):
        gradients = torch.autograd.grad(loss, tuple(params.values()), retain_graph=True, allow_unused=True)
    surrogate = zero
    norms = {}
    for (name, parameter), gradient in zip(params.items(), gradients):
        if gradient is not None:
            if not bool(torch.isfinite(gradient).all()):
                raise ValueError('Non-finite native posterior teaching gradient')
            surrogate = surrogate + (gradient.detach() * parameter).sum()
            norms[name] = float(gradient.detach().double().norm())
    return surrogate, {'context_teacher_jobs': len(losses), 'context_teacher_kl': float(loss.detach()),
                       'context_teacher_gradient_norms': norms,
                       'context_teacher_ports_and_decoder': False}
