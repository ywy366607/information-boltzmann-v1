"""Opt-in action-consequence uncertainty, grounded in the existing mushroom body.

This is an explicit statistical adapter, not a claim that phi is Bayesian precision.
The physical brain, its DAN dynamics and its KC->MBON memory remain the organism.
Only the real live sensory pass updates this adapter; BPTT/replay only consumes credit.
"""
import math

import torch
import torch.nn.functional as F


class GaussianConsequences:
    """Local linear-Gaussian successor belief with shared, past-calibrated noise.

    P is covariance of each output's coefficient vector; W predicts KC responses.
    IG = .5 log(1 + x'Px/R) per output, before incorporating the observation.
    R is estimated from past innovations, with parameter variance subtracted.
    There is no finite memory horizon or forgetting imposed on the posterior.
    The stationary linear/noise assumptions limit what this uncertainty represents.
    """

    def __init__(self, state):
        self.state = state

    def initialize(self, x, y):
        x, y = x.double(), y.double()
        if 'P' not in self.state:
            self.state.update(P=torch.eye(x.numel(), device=x.device, dtype=x.dtype),
                              W=x.new_zeros(x.numel(), y.numel()), n=0,
                              noise_sum=1.0, noise_count=1)
        if self.state['P'].shape != (x.numel(), x.numel()) or self.state['W'].shape[1] != y.numel():
            raise ValueError('Consequence feature schema changed; migrate explicitly')

    def ratio(self, x):
        if 'P' not in self.state:
            return float(x.dot(x))
        noise = self.state['noise_sum'] / self.state['noise_count']
        x = x.to(self.state['P'])
        return float((x @ self.state['P'] @ x).clamp_min(0)) / max(noise, torch.finfo(x.dtype).eps)

    @torch.no_grad()
    def observe(self, x, y, exposure=1.0):
        if not 0.0 < exposure <= 1.0:
            raise ValueError('A consequence needs actual sensory exposure in (0, 1]')
        # This small statistical state needs more precision than neural activity.
        x, y = x.detach().double(), y.detach().double()
        self.initialize(x, y)
        p, w = self.state['P'], self.state['W']
        noise = max(self.state['noise_sum'] / self.state['noise_count'], torch.finfo(x.dtype).eps)
        px = p @ x
        variance = x.dot(px).clamp_min(0)
        r = noise / exposure
        denominator = r + variance
        gain = 0.5 * torch.log1p(variance / r)
        residual = y - x @ w
        # Use the OLD noise to value and update this observation. It cannot set its own reward scale.
        w.add_(torch.outer(px / denominator, residual))
        p.sub_(torch.outer(px, px) / denominator)
        p.copy_(0.5 * (p + p.T))
        innovation_noise = max(0.0, float(residual.square().mean() - variance)) * exposure
        self.state['noise_sum'] += innovation_noise
        self.state['noise_count'] += 1
        self.state['n'] += 1
        return {'gain': float(gain), 'prediction_mse': float(residual.square().mean()),
                'noise_before': noise, 'parameter_variance_before': float(variance),
                'updates': self.state['n']}


def first_commit_log_prob(model, frames, first_commit):
    """Likelihood of the actual first-commit time, or survival of the whole interval."""
    from .fly_active_inference import motor_logits
    logits = motor_logits(model, frames)[:, 1]
    if first_commit is None:
        return F.logsigmoid(-logits).sum()
    if not 0 <= first_commit < len(logits):
        raise ValueError('First commit must belong to its decision interval')
    return F.logsigmoid(-logits[:first_commit]).sum() + F.logsigmoid(logits[first_commit])


class NativeConsequenceCredit:
    """One live window's credit, using saved O(1) policy eligibility across windows.

    Topology pools KC activity onto its real MBON destinations. Context also contains
    MBON state, phi, and DAN concentrations. Outcomes are KC responses, not DAN reward.
    A sampled word uses the existing decoder, its exact commit read and temperature.
    No vocabulary policy head, fixed exploration rate or forced speaking target is added.
    """

    def __init__(self, learner):
        self.learner, self.model = learner, learner.model
        m = self.model
        if getattr(m, 'comp_pre', None) is None or not m.comp_pre.numel():
            raise ValueError('Consequence credit requires the native KC->MBON compartment memory')
        if getattr(learner, 'eye_binary', False):
            raise ValueError('Consequence credit currently requires continuous sensory precision')
        if getattr(m, 'ai_mode', None) != 'motor':
            raise ValueError('Consequence credit requires native motor acts')
        if not hasattr(learner, 'consequence_state'):
            learner.consequence_state = {'format': 'kc-action-consequence-v1', 'belief': {},
                                         'eligibility': {}, 'source': 'unknown', 'reward_n': 0, 'reward_sum': 0.0}
        self.state = learner.consequence_state
        if self.state.get('format') != 'kc-action-consequence-v1':
            raise ValueError('Unsupported consequence state')
        self.belief = GaussianConsequences(self.state['belief'])
        self.posts, self.groups = torch.unique(m.comp_post.long(), sorted=True, return_inverse=True)
        self.counts = torch.bincount(self.groups).to(learner.state.h).clamp_min(1)
        self.thresholds = m.get_thresholds().detach().expand_as(learner.state.h).clamp_min(1e-6)
        if self.state.get('posts') is None:
            self.state['posts'] = self.posts.clone()
        if not torch.equal(self.state['posts'], self.posts):
            raise ValueError('KC->MBON topology changed; migrate consequence state explicitly')
        self.params = {k: p for k, p in m.named_parameters() if p.requires_grad and
                       (k in ('ai_w', 'ai_b') or k.startswith(('decoder.', 'read_norm.')))}
        self.gradients = {}
        self.decisions = []
        self.gains = []
        self.diagnostics = []
        # Eligibility lifetime belongs to the existing, learnable KC trace, not a new tuned timer.
        tau_ticks = -1.0 / torch.log(torch.sigmoid(m.comp_logit_trace.detach()).clamp(1e-7, 1 - 1e-7))
        self.decay = math.exp(-1.0 / max(float(tau_ticks) / max(1., learner.stream_pace_mean), 1e-7))

    @torch.no_grad()
    def features(self, physical):
        m = self.model
        h = torch.tanh(physical.h / self.thresholds)[0]
        kc = h.new_zeros(len(self.posts)).scatter_add_(0, self.groups, h[m.comp_pre.long()]) / self.counts
        if getattr(m, 'kc_contact_memory_enabled', False) and getattr(m, 'kc_contact_read_enabled', True):
            from .fly_kc_contact_memory import contact_state
            phi = contact_state(m, physical)[0]
            groups = self.groups.index_select(0, m.kcm_edge)
            weight = m.kcm_fraction
        else:
            phi, groups, weight = physical.comp_memory, self.groups, 1.
            if phi.numel() != m.comp_pre.numel():
                raise ValueError('Native compartment state is missing')
        familiar = h.new_zeros(len(self.posts)).scatter_add_(0, groups, phi[0].tanh()*weight) / self.counts
        da = physical.dopamine
        if da.ndim != 2 or da.shape[0] != 2 or da.shape[1] != h.numel():
            raise ValueError('Native DAN state is missing')
        context = torch.cat((kc, h[self.posts], familiar, da[:, self.posts].tanh().flatten()))
        context = context / math.sqrt(context.numel())
        return context.detach(), kc.detach()

    def input_ratio(self, source):
        x = (self.state.get('response') or {}).get('action', self.state.get('action'))
        return self.belief.ratio(x) if source == 'mother_reply' and x is not None else 0.0

    @torch.no_grad()
    def observe(self, physical, source, exposure, end=True):
        x = self.state.get('action')
        if source != 'mother_reply' or x is None or exposure <= 0:
            if source == 'mother_reply':
                self._advance_trace()
            if source == 'mother_reply' and end:
                self.end_response()
            return
        _, y = self.features(physical)
        response = self.state.get('response')
        if response is None:
            # Adjacent words are one correlated consequence, not independent evidence.
            # Archive the score at the beginning of the causal response. Its complete
            # outcome has an undiscounted, per-response reward, irrespective of word
            # count. This archive is statistical delayed credit, not frozen physiology.
            response = {'action': x.clone(), 'sum': torch.zeros_like(y), 'n': 0, 'exposure': 0.,
                        'score': {k: v.clone() for k, v in self.state['eligibility'].items()}}
            self.state['response'] = response
        response['sum'].add_(y)
        response['n'] += 1
        response['exposure'] += min(1.0, float(exposure))
        self._advance_trace()
        if end:
            self.end_response()

    @torch.no_grad()
    def _advance_trace(self):
        # Historical eligibility still advances every external word, across windows.
        # Aggregating a response must not silently change this clock into "replies".
        for eligibility in self.state['eligibility'].values():
            eligibility.mul_(self.decay)
        self.state['clock_words'] = self.state.get('clock_words', 0) + 1

    @torch.no_grad()
    def end_response(self):
        response = self.state.pop('response', None)
        if response is None or response['n'] == 0:
            return
        result = self.belief.observe(response['action'], response['sum'] / response['n'],
                                     response['exposure'] / response['n'])
        gain = result['gain']
        baseline = self.state['reward_sum'] / max(1, self.state['reward_n'])
        for name, eligibility in self.state['eligibility'].items():
            if name not in self.params:
                raise ValueError(f'Consequence policy parameter missing: {name}')
            responsible = response['score'].get(name, torch.zeros_like(eligibility))
            self.gradients.setdefault(name, torch.zeros_like(eligibility)).add_(responsible, alpha=gain - baseline)
        self.state['reward_sum'] += gain
        self.state['reward_n'] += 1
        self.gains.append(gain)
        self.diagnostics.append(result)

    def _add_score(self, score, names, weight=1.0):
        params = [self.params[k] for k in names]
        grads = torch.autograd.grad(score, params, allow_unused=True)
        with torch.no_grad():
            for name, grad in zip(names, grads):
                if grad is not None:
                    old = self.state['eligibility'].setdefault(name, torch.zeros_like(grad))
                    old.add_(grad.detach(), alpha=weight)

    def choose(self, context, frames, first_commit, mother_turn=False):
        """Record a real timing choice. The unknown target is not an argument."""
        with torch.enable_grad():
            names = [k for k in self.params if k in ('ai_w', 'ai_b')]
            if not names:
                raise ValueError('Native motor policy is frozen')
            self._add_score(first_commit_log_prob(self.model, frames.detach(), first_commit), names)
        said = first_commit is not None
        self.decisions.append(said)
        # Interactions allow different consequences of speech in the same context.
        content = self.state.get('utterance', context.new_zeros(self.model.decoder.in_features))
        content = content / max(1, self.state.get('utterance_words', 0))
        self.state['action'] = torch.cat((context.new_ones(1), context, context * float(said),
                                          context.new_tensor([float(said), float(mother_turn)]), content))

    def say(self, read, word, temperature):
        """Credit the exact sampled language action, reusing the existing decoder."""
        if self.state.get('response') is not None:
            raise RuntimeError('Finish the heard response before recording a new spoken action')
        with torch.enable_grad():
            z = self.model.read_norm(read.detach())
            logits = self.model.decoder(z).float() / max(float(temperature), 1e-6)
            score = F.log_softmax(logits, -1)[0, int(word)]
            self._add_score(score, [k for k in self.params if k.startswith(('decoder.', 'read_norm.'))])
        with torch.no_grad():
            content = F.normalize(self.model.decoder.weight[int(word)].detach(), dim=0)
            self.state.setdefault('utterance', torch.zeros_like(content)).add_(content)
            self.state['utterance_words'] = self.state.get('utterance_words', 0) + 1
            self.state['action'][-content.numel():] = self.state['utterance'] / self.state['utterance_words']

    def begin_child_turn(self):
        self.state['utterance'] = self.learner.state.h.new_zeros(self.model.decoder.in_features)
        self.state['utterance_words'] = 0
        self.state['in_child_turn'] = True

    def finish(self):
        previous = getattr(self.learner, 'pending_speak_credit', None)
        if previous is not None:
            # A continuing individual may enter this mode with one unapplied legacy window.
            # Preserve that credit once; never silently erase it at the mechanism boundary.
            if isinstance(previous, dict):
                for name, gradient in previous['trace'].items():
                    self.gradients.setdefault(name, torch.zeros_like(gradient)).add_(gradient)
            else:
                from .fly_active_inference import delayed_speaking_value
                with torch.enable_grad():
                    score, _ = delayed_speaking_value(self.model, *previous, n_words=1)
                    names = [k for k in self.params if k in ('ai_w', 'ai_b')]
                    grads = torch.autograd.grad(score, [self.params[k] for k in names], allow_unused=True)
                for name, gradient in zip(names, grads):
                    if gradient is not None:
                        self.gradients.setdefault(name, torch.zeros_like(gradient)).add_(gradient.detach())
        self.learner.pending_speak_credit = {'trace': self.gradients, 'decisions': list(self.decisions)}
        self.learner.last_epistemic = {
            'source': 'native_action_consequence_parameter_ig',
            'consequence_observations': len(self.gains),
            'consequence_gain_mean': sum(self.gains) / max(1, len(self.gains)),
            'consequence_updates': self.state['belief'].get('n', 0),
            'consequence_noise': self.state['belief'].get('noise_sum', 1.) / self.state['belief'].get('noise_count', 1),
            'consequence_prediction_mse': sum(d['prediction_mse'] for d in self.diagnostics) / max(1, len(self.diagnostics)),
            'consequence_parameter_variance': sum(d['parameter_variance_before'] for d in self.diagnostics) / max(1, len(self.diagnostics)),
            'content_decisions': sum(self.decisions), 'timing_decisions': len(self.decisions),
            'eligibility_bytes': sum(v.numel() * v.element_size() for v in self.state['eligibility'].values()),
        }


class NativeCircuitConsequenceCredit(NativeConsequenceCredit):
    """Route the existing v1 consequence credit into native policy derivatives.

    Head and native score traces use the same physical KC clock, including
    silence and noncontingent input. Archived response scores remain fixed until
    settlement. The posterior, sampled actions and sensory stream are unchanged.
    """

    def __init__(self, learner):
        super().__init__(learner)
        if self.state.get('action_schema') is not None:
            raise ValueError('Native credit alone cannot downgrade an ordered action posterior')
        self.native_events = []
        self.decision_bounds = (0, 0)
        self.content_tick = None
        self.content_tail = 0

    def advance_ticks(self, ticks):
        ticks = int(ticks)
        if ticks < 0:
            raise ValueError('Consequence clock cannot run backwards')
        rho = float(torch.sigmoid(self.model.comp_logit_trace.detach()))
        with torch.no_grad():
            for eligibility in self.state['eligibility'].values():
                eligibility.mul_(rho ** ticks)
        # Preserve an old individual's accumulated trace; this records the
        # corrected clock from the mechanism boundary, without inventing its past.
        if 'head_clock_ticks' not in self.state:
            self.state['head_clock_ticks'] = int(self.state.get('native_credit', {}).get('clock_ticks', 0))
        self.state['head_clock_ticks'] += ticks
        self.native_events.append({'kind': 'clock', 'ticks': ticks})

    def _advance_trace(self):
        # observe() counts heard response words but must not discount a second
        # time: advance_ticks already accounted for every actual physical tick.
        self.state['clock_words'] = self.state.get('clock_words', 0) + 1

    def _add_score(self, score, names, weight=1.0):
        rho = float(torch.sigmoid(self.model.comp_logit_trace.detach()))
        # Both live decisions are recorded after their interval. Transport the
        # new score from its actual commit to that interval's end, just as the
        # native derivative tape does. Do not discount the existing trace again.
        return super()._add_score(score, names, weight=weight * rho ** self.content_tail)

    def choose(self, context, frames, first_commit, mother_turn=False):
        start, length = self.decision_bounds
        tail = length - first_commit - 1 if first_commit is not None else 0
        self.native_events.append({'kind': 'timing', 'start': start, 'length': length,
                                   'first': first_commit, 'tail_ticks': tail})
        self.content_tick = start + first_commit if first_commit is not None else None
        self.content_tail = tail
        return super().choose(context, frames, first_commit, mother_turn)

    def say(self, read, word, temperature):
        if self.content_tick is None:
            raise ValueError('A spoken word needs its actual first-commit time')
        self.native_events.append({'kind': 'content', 'tick': self.content_tick,
                                   'word': int(word), 'temperature': float(temperature),
                                   'tail_ticks': self.content_tail})
        self.content_tick = None
        return super().say(read, word, temperature)

    def observe(self, physical, source, exposure, end=True):
        if (source == 'mother_reply' and exposure > 0 and self.state.get('action') is not None
                and self.state.get('response') is None):
            self.native_events.append({'kind': 'response_begin'})
        return super().observe(physical, source, exposure, end)

    def end_response(self):
        count = len(self.gains)
        baseline = self.state['reward_sum'] / max(1, self.state['reward_n'])
        super().end_response()
        if len(self.gains) != count:
            self.native_events.append({'kind': 'response_end', 'advantage': self.gains[-1] - baseline})

    def finish(self):
        super().finish()
        self.learner.last_epistemic.update(native_credit_enabled=True,
                                          native_credit_mode='existing-consequence-only')
