"""Shadow next-event likelihood on fixed word marks and native fly context.

This stationary local statistical proxy observes the real physical clock. It
does not change sensing, motor actions, rewards, native credit or predictions.
Its Gaussian mark coordinates are fixed embedding projections, not categorical
token probabilities. Native phi/DAN enter as measured context, never precision.
"""
import math
from numbers import Integral

import torch
import torch.nn.functional as F

from .fly_action_model import cosine_projection
from .fly_consequence_belief import NativeConsequenceCredit
from .fly_event_belief import ProbitEventBelief, affine_candidate_moments


FORMAT = 'fly-native-generative-shadow-v1'
EXTERNAL_SOURCES = frozenset(('mother_opening', 'mother_call', 'mother_reply', 'external'))
SOURCES = EXTERNAL_SOURCES | {'self_hearing', 'quiet'}


def _ticks(value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'ticks must be an integer >= {minimum}')
    return int(value)


def _eye_mean(value):
    value = torch.as_tensor(value).detach().double()
    if not value.numel() or not bool(torch.isfinite(value).all()):
        raise ValueError('eye gain must contain finite actual sensory gains')
    if bool(((value < 0) | (value > 1)).any()):
        raise ValueError('eye gain must lie in [0,1]')
    return float(value.mean())


class NativeGenerativeObserver:
    """Durable passive observer, separate from the existing consequence adapter.

    Call begin_interval before its target is available. Eye gain is a measured
    conditioning variable, not an observation-noise rescaling. Only the initial
    current gain enters features; future realized effort never replaces it.
    The fully observed fixed word mark is a supervised world-likelihood target,
    not the organism's subjective sensory observation. This adapter supplies no
    attention information until an actual perception kernel has been modeled.
    """

    def __init__(self, learner):
        self.learner, self.model = learner, learner.model
        self.diagnostics = []
        m = self.model
        if getattr(m, 'comp_pre', None) is None or not m.comp_pre.numel():
            raise ValueError('generative observer requires native KC->MBON topology')
        if m.embedding.weight.requires_grad:
            raise ValueError('fixed event marks require a frozen input embedding')
        self.posts, self.groups = torch.unique(m.comp_post.long(), sorted=True, return_inverse=True)
        self.width = len(self.posts)
        self.counts = torch.bincount(self.groups).to(learner.state.h).clamp_min(1)
        self.thresholds = m.get_thresholds().detach().expand_as(learner.state.h).clamp_min(1e-6)
        self.feature_dim = 13 * self.width + 3
        self.projection = cosine_projection(m.embedding.weight.shape[1], self.width,
                                            m.embedding.weight).detach()
        state = getattr(learner, 'generative_event_state', None)
        if state is None:
            state = {}
            learner.generative_event_state = state
        if not isinstance(state, dict):
            raise ValueError('generative event state must be a dictionary')
        if not state:
            z = torch.zeros(self.width, dtype=torch.float64)
            state.update(format=FORMAT, posts=self.posts.detach().cpu().clone(),
                         feature_dim=self.feature_dim, mark_dim=self.width, belief={},
                         prefix={'n': 0, 'sum': z.clone(), 'weighted': z.clone(), 'last': z.clone()},
                         pending=None, clock_ticks=0, intervals=0, child_turns=0, in_child_turn=False,
                         source_counts={}, log_likelihood_sum=0., realized_information_sum=0.)
        if (state.get('format') != FORMAT or state.get('feature_dim') != self.feature_dim
                or state.get('mark_dim') != self.width
                or not torch.equal(state.get('posts'), self.posts.detach().cpu())):
            raise ValueError('generative event schema/topology changed; migrate explicitly')
        self.state = state
        self.belief = ProbitEventBelief(state['belief'], feature_dim=self.feature_dim,
                                       mark_dim=self.width)
        if (state['pending'] is None) != (state['belief']['trial'] is None):
            raise ValueError('generative pending interval and belief trial disagree')

    @torch.no_grad()
    def features(self, physical):
        """Reuse native context extraction without constructing old credit state."""
        self.thresholds = self.model.get_thresholds().detach().expand_as(physical.h).clamp_min(1e-6)
        context, _ = NativeConsequenceCredit.features(self, physical)
        return context.detach().cpu().double()

    @torch.no_grad()
    def word_coordinates(self, token):
        if isinstance(token, bool) or not isinstance(token, Integral):
            raise ValueError('word token must be an integer')
        if not 0 <= token < self.model.embedding.weight.shape[0]:
            raise ValueError('word token is outside the frozen vocabulary')
        emb = F.normalize(self.model.embedding.weight[int(token)].detach(), dim=0)
        return F.normalize(self.projection @ emb, dim=0).cpu().double()

    @torch.no_grad()
    def action_features(self, context, eye_gain, candidate=None):
        """Affine candidate coordinates; candidate=None means the current prefix."""
        context = torch.as_tensor(context).detach().cpu().double()
        if context.shape != (5 * self.width,) or not bool(torch.isfinite(context).all()):
            raise ValueError('native context shape or values are invalid')
        q = self.state['prefix']
        n = q['n'] + int(candidate is not None)
        s, weighted, last = q['sum'], q['weighted'], q['last']
        if candidate is not None:
            candidate = torch.as_tensor(candidate).detach().cpu().double()
            if candidate.shape != (self.width,) or not bool(torch.isfinite(candidate).all()):
                raise ValueError('candidate coordinates must match the fixed word projection')
            s, weighted, last = s + candidate, weighted + n * candidate, candidate
        mean = s / max(1, n)
        order = ((2 * weighted - (n + 1) * s) / (n * (n - 1))
                 if n > 1 else torch.zeros_like(s))
        joint = (context.reshape(5, self.width)
                 * ((mean + order + last) / math.sqrt(3.))).flatten()
        return torch.cat((context.new_ones(1), context, mean, order, last,
                          context.new_tensor([math.log1p(n)]), joint,
                          context.new_tensor([_eye_mean(eye_gain)])))

    def begin_child_turn(self):
        if self.state['pending'] is not None:
            raise ValueError('finish the current interval before changing child turn')
        q = self.state['prefix']
        q['n'] = 0
        for key in ('sum', 'weighted', 'last'):
            q[key].zero_()
        self.state['child_turns'] += 1
        self.state['in_child_turn'] = True

    @torch.no_grad()
    def candidate_values(self, context, eye_gain, coordinates, horizon, batch_size=1024):
        """Same posterior values for waiting and candidate command words.

        Coordinates must be from this observer's fixed input dictionary. The
        action summary is affine in the next word, so only small quadratics are
        materialized, never vocabulary x full features. H is a decision lookahead,
        not a memory lifetime. This is the world-likelihood approximation; it is
        NOT a subjective information value for changing sensory gain.
        """
        if self.state['pending'] is not None:
            raise ValueError('prospective choices require an explicit event boundary')
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
            raise ValueError('candidate batch_size must be a positive integer')
        coordinates = torch.as_tensor(coordinates).detach().cpu().double()
        if coordinates.ndim != 2 or coordinates.shape[1] != self.width:
            raise ValueError('word coordinates must match the fixed input dictionary')
        wait = self.action_features(context, eye_gain)
        zero = torch.zeros(self.width, dtype=torch.float64)
        base = self.action_features(context, eye_gain, zero)
        basis = torch.stack([self.action_features(context, eye_gain, e) - base
                             for e in torch.eye(self.width, dtype=torch.float64)], -1)
        prior = self.belief.state
        mu, variance = affine_candidate_moments(prior['mean'], prior['covariance'],
                                               base, basis, coordinates)
        mark = self.belief.mark.state
        _, leverage = affine_candidate_moments(mark['mean'], mark['covariance'],
                                               base, basis, coordinates)
        values = []
        for start in range(0, len(coordinates), batch_size):
            stop = start + batch_size
            pred = self.belief.predict_from_moments(mu[start:stop], variance[start:stop], horizon,
                                                  mark_leverage=leverage[start:stop])
            values.append(pred['total_information'])
        return torch.cat(values), self.belief.predict(wait, horizon)['total_information']

    def prospective_policy(self, context, eye_gains, sense_log_mass, output_logits,
                           word_logits, coordinates, horizon):
        """Connect the shared event belief to the single joint policy measure.

        Pure epistemic, optional analysis only: no automatic actuation, no reward
        and no mutation of physiology/credit. Sensory candidate values condition
        the world proxy on gain; they do not replace a still-missing perception
        likelihood. Caller must report that boundary when interpreting this plan.
        Native policy gradients remain available; inference values are detached.
        """
        from .fly_generative_policy import joint_event_policy
        gains = torch.as_tensor(eye_gains).detach().cpu().double()
        if gains.ndim != 1 or not gains.numel():
            raise ValueError('eye_gains must be a nonempty sensory measure')
        values, waits = [], []
        for gain in gains:
            u, wait = self.candidate_values(context, gain, coordinates, horizon)
            values.append(u)
            waits.append(wait)
        return joint_event_policy(sense_log_mass, output_logits, word_logits,
                                  torch.stack(waits), torch.stack(values))

    def begin_interval(self, physical, eye_gain):
        if self.state['pending'] is not None:
            raise ValueError('finish the pending generative interval first')
        context, gain = self.features(physical), _eye_mean(eye_gain)
        x = self.action_features(context, gain)
        # Archive the prospective question before either elapsed time or the
        # target is available. Later evidence must not rewrite this forecast.
        prediction = self.belief.predict(x, horizon=1)
        self.belief.begin(x)
        self.state['pending'] = {'context': context.clone(), 'x': x.clone(), 'eye_gain': gain,
                                 'expected_information_before': float(prediction['total_information'])}
        return x.clone()

    @torch.no_grad()
    def end_interval(self, ticks, token, source, *, actual_eye_gain=None):
        """Score before updating; a self command has no stochastic external mark.

        External arrivals: ticks-1 quiet ticks then an event. Known self echoes
        and explicit quiet boundaries: all ticks are external survival. A quiet
        boundary uses token=None and never adds a synthetic vocabulary item.
        """
        ticks = _ticks(ticks, 1)
        if source not in SOURCES:
            raise ValueError(f'unknown generative event source: {source!r}')
        if source == 'quiet':
            if token is not None:
                raise ValueError('quiet boundaries have no token')
            mark = None
        else:
            mark = self.word_coordinates(token)
        pending = self.state['pending']
        if pending is None:
            raise ValueError('begin the generative interval before observing its outcome')
        gain = pending['eye_gain']
        actual_gain = None if actual_eye_gain is None else _eye_mean(actual_eye_gain)
        x = pending['x'].clone()
        # A fixed one-tick prospective horizon does not use the realized future
        # waiting time to choose the reported expected-information question.
        horizon = 1
        external = source in EXTERNAL_SOURCES
        result = self.belief.resolve(external, mark=mark if external else None,
                                     quiet_ticks=ticks - int(external))
        mark_result = result['mark']
        mark_log_likelihood = 0.
        if mark_result is not None:
            distribution = torch.distributions.StudentT(
                mark_result['prediction_df_before'], mark_result['prediction_mean_before'],
                mark_result['prediction_scale2_before'].sqrt())
            mark_log_likelihood = float(distribution.log_prob(mark).sum())
        information = result['realized_information']
        if mark_result is not None:
            information += mark_result['realized_information']
        log_likelihood = result['log_evidence'] + mark_log_likelihood
        report = dict(source=source, token=None if token is None else int(token), ticks=ticks,
                      external_event=external, eye_gain=gain, actual_eye_gain=actual_gain,
                      timing_log_likelihood=result['log_evidence'],
                      mark_log_likelihood=mark_log_likelihood, log_likelihood=log_likelihood,
                      expected_information_before=pending['expected_information_before'],
                      prediction_horizon=horizon, realized_information=information,
                      clock_ticks=self.state['clock_ticks'] + ticks,
                      likelihood_kind='stationary_native_context_fixed_coordinate_proxy',
                      policy_changed=False)
        self.state.update(pending=None, clock_ticks=report['clock_ticks'],
                          intervals=self.state['intervals'] + 1, last_report=report,
                          log_likelihood_sum=self.state['log_likelihood_sum'] + log_likelihood,
                          realized_information_sum=self.state['realized_information_sum'] + information)
        counts = self.state['source_counts']
        counts[source] = counts.get(source, 0) + 1
        # The command is new knowledge only after the interval was scored.
        if source == 'self_hearing':
            q = self.state['prefix']
            q['n'] += 1
            q['sum'].add_(mark)
            q['weighted'].add_(mark, alpha=q['n'])
            q['last'].copy_(mark)
        self.diagnostics.append(report)
        return report


# Short public spelling for callers that already supply a native fly learner.
GenerativeEventObserver = NativeGenerativeObserver
