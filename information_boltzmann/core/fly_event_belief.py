"""Stationary Bayesian next-event belief, independent of policy and reward.

A fixed native feature vector x gives a latent z=x'beta and a per-physical-tick
arrival probability Phi(z). beta is Gaussian. Conditional on beta, ticks are
independent; marginally their shared unknown hazard makes them dependent.
The next-event outcome is first arrival at ticks 1..H or survival through H.
An optional independent NIG linear model describes a fixed-coordinate mark at
arrival. Its likelihood and posterior are exact; arrival posterior storage uses
Gaussian assumed-density filtering (ADF), with one-dimensional quadrature.

These are stationary local approximations, not a hidden-state world model or a
preference specification. No reward, speaking prior, drift, sensory-noise gate,
or neural/checkpoint loading is part of this module. All information is in nats.
"""
import math
from functools import lru_cache
from numbers import Integral

import numpy as np
import torch


FORMAT = 'fly-probit-marked-event-v1'
MARK_FORMAT = 'fly-nig-event-mark-v1'
MAX_HORIZON = 4096


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def _finite(value, name):
    if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name} must be a finite tensor')


def _required(state, names, name):
    missing = set(names).difference(state)
    if missing:
        raise ValueError(f'{name} is missing required fields: {sorted(missing)}')


def _vector(value, size, like, name):
    value = torch.as_tensor(value, device=like.device, dtype=torch.float64).detach()
    if value.shape != (size,):
        raise ValueError(f'{name} must have shape [{size}]')
    _finite(value, name)
    return value


def _features(value, size, like):
    value = torch.as_tensor(value, device=like.device, dtype=torch.float64).detach()
    if value.ndim not in (1, 2) or value.shape[-1] != size:
        raise ValueError(f'features must have shape [D] or [N,D], D={size}')
    _finite(value, 'features')
    return value


def _covariance(value, size, name):
    _finite(value, name)
    if value.dtype != torch.float64 or value.shape != (size, size):
        raise ValueError(f'{name} must be float64 with shape [{size},{size}]')
    if not torch.allclose(value, value.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f'{name} must be symmetric')
    if int(torch.linalg.cholesky_ex(value).info) != 0:
        raise ValueError(f'{name} must be positive definite')


@lru_cache(maxsize=16)
def _hermite_numpy(points):
    points = _integer(points, 'quadrature_points', 8)
    if points > 256:
        raise ValueError('quadrature_points must be <= 256')
    nodes, weights = np.polynomial.hermite.hermgauss(points)
    return nodes * math.sqrt(2.), weights / math.sqrt(math.pi)


def _hermite(points, like):
    nodes, weights = _hermite_numpy(points)
    return (torch.as_tensor(nodes, device=like.device, dtype=torch.float64),
            torch.as_tensor(weights, device=like.device, dtype=torch.float64))


def _nonnegative(value, name):
    _finite(value, name)
    if bool((value < -1e-9).any()):
        raise ArithmeticError(f'{name} became negative beyond rounding error')
    return value.clamp_min(0.)


@torch.no_grad()
def affine_candidate_moments(mean, covariance, base, basis, candidates):
    """Compute moments for x=base+basis@candidate without allocating [V,D].

    mean [D] (or [D,M]), covariance [D,D], basis [D,C], candidates [V,C].
    The full feature covariance is retained; only its C-dimensional candidate
    quadratic form is materialized. Returns means [V] / [V,M], variances [V].
    """
    _finite(mean, 'mean')
    _finite(covariance, 'covariance')
    mean = mean.detach().double()
    covariance = covariance.detach().to(mean)
    if mean.ndim not in (1, 2):
        raise ValueError('mean must have shape [D] or [D,M]')
    d = mean.shape[0]
    base = _vector(base, d, mean, 'base')
    basis = torch.as_tensor(basis, device=mean.device, dtype=torch.float64).detach()
    candidates = torch.as_tensor(candidates, device=mean.device, dtype=torch.float64).detach()
    if covariance.shape != (d, d) or basis.ndim != 2 or basis.shape[0] != d:
        raise ValueError('covariance or basis feature dimension differs')
    if candidates.ndim != 2 or candidates.shape[1] != basis.shape[1]:
        raise ValueError('candidates must have shape [V,C] matching basis [D,C]')
    _finite(basis, 'basis')
    _finite(candidates, 'candidates')
    pb = covariance @ base
    quadratic = basis.T @ covariance @ basis
    variances = ((candidates @ quadratic) * candidates).sum(-1)
    variances = variances + 2 * (candidates @ (basis.T @ pb)) + base @ pb
    means = base @ mean + candidates @ (basis.T @ mean)
    return means, _nonnegative(variances, 'candidate variance')


@torch.no_grad()
def probit_next_event(mean, variance, horizon, quadrature_points=64):
    """Pre-observation timing probabilities and mutual information.

    Last probability column is no event by H. Conditional hazard is constant
    through H: E[(1-p)^t p], not (1-E[p])^t E[p]. Quadrature integrates one latent
    Gaussian, and therefore distinguishes irreducible Bernoulli noise from
    uncertainty about the hazard. H=0 is a deterministic empty observation.
    """
    horizon = _integer(horizon, 'horizon')
    if horizon > MAX_HORIZON:
        raise ValueError(f'horizon exceeds numerical allocation limit {MAX_HORIZON}')
    mean = torch.as_tensor(mean, dtype=torch.float64).detach()
    variance = torch.as_tensor(variance, device=mean.device, dtype=torch.float64).detach()
    mean, variance = torch.broadcast_tensors(mean, variance)
    _finite(mean, 'mean')
    _finite(variance, 'variance')
    if bool((variance < 0).any()):
        raise ValueError('variance must be nonnegative')
    nodes, weights = _hermite(quadrature_points, mean)
    if horizon == 0:
        zero = torch.zeros_like(mean)
        return {'probabilities': torch.ones_like(mean)[..., None],
                'event_probability': zero, 'timing_information': zero,
                'expected_ticks': zero, 'horizon': 0}
    z = mean[..., None] + variance.sqrt()[..., None] * nodes
    log_yes, log_no = torch.special.log_ndtr(z), torch.special.log_ndtr(-z)
    log_survival = torch.zeros_like(z)
    conditional_entropy = torch.zeros_like(z)
    probabilities = []
    for _ in range(horizon):
        log_mass = log_survival + log_yes
        mass = log_mass.exp()
        probabilities.append((mass * weights).sum(-1))
        conditional_entropy -= torch.special.xlogy(mass, mass)
        log_survival = log_survival + log_no
    survival = log_survival.exp()
    probabilities.append((survival * weights).sum(-1))
    conditional_entropy -= torch.special.xlogy(survival, survival)
    probabilities = torch.stack(probabilities, -1)
    # xlogy gives the correct 0 log 0 limit at saturated hazards.
    entropy = -torch.special.xlogy(probabilities, probabilities).sum(-1)
    information = _nonnegative(entropy - (weights * conditional_entropy).sum(-1),
                               'timing mutual information')
    event_probability = probabilities[..., :-1].sum(-1)
    ticks = torch.arange(1, horizon + 1, device=mean.device, dtype=torch.float64)
    expected_ticks = (probabilities[..., :-1] * ticks).sum(-1) + horizon * probabilities[..., -1]
    return {'probabilities': probabilities, 'event_probability': event_probability,
            'timing_information': information, 'expected_ticks': expected_ticks,
            'horizon': horizon}


def _tilted_probit(mean, variance, quiet_ticks, event, points):
    """Scalar exact likelihood moments and KL by posterior-centered quadrature."""
    quiet_ticks = _integer(quiet_ticks, 'quiet_ticks')
    if not isinstance(event, bool):
        raise ValueError('event must be boolean')
    if variance < 0:
        raise ArithmeticError('arrival projection variance is negative')
    if quiet_ticks == 0 and not event:
        return mean, variance, mean.new_zeros(()), mean.new_zeros(())
    if variance == 0:
        log_likelihood = quiet_ticks * torch.special.log_ndtr(-mean)
        if event:
            log_likelihood += torch.special.log_ndtr(mean)
        return mean, variance, log_likelihood, mean.new_zeros(())

    # The posterior log density is concave. A Laplace-centered proposal avoids
    # prior-grid collapse when a long quiet interval moves its mass into a tail.
    mode = mean.clone()
    precision = 1. / variance
    for _ in range(64):
        log_phi = -.5 * mode.square() - .5 * math.log(2 * math.pi)
        yes = (log_phi - torch.special.log_ndtr(mode)).exp()
        no = (log_phi - torch.special.log_ndtr(-mode)).exp()
        gradient = -(mode - mean) / variance - quiet_ticks * no + int(event) * yes
        precision = (1. / variance + quiet_ticks * no * (no - mode)
                     + int(event) * yes * (yes + mode))
        step = gradient / precision
        mode = mode + step
        if abs(float(step)) < 1e-12 * (1 + abs(float(mode))):
            break
    else:
        raise ArithmeticError('probit posterior mode did not converge')
    if not bool(torch.isfinite(precision)) or precision <= 0:
        raise ArithmeticError('probit posterior curvature is invalid')
    scale = precision.rsqrt()
    nodes, weights = _hermite(points, mean)
    z = mode + scale * nodes
    log_likelihood = quiet_ticks * torch.special.log_ndtr(-z)
    if event:
        log_likelihood += torch.special.log_ndtr(z)
    log_ratio = -.5 * (z - mean).square() / variance + .5 * nodes.square()
    log_ratio += torch.log(scale) - .5 * torch.log(variance)
    log_weight = weights.log() + log_likelihood + log_ratio
    log_evidence = torch.logsumexp(log_weight, 0)
    posterior_weights = (log_weight - log_evidence).exp()
    new_mean = (posterior_weights * z).sum()
    new_variance = (posterior_weights * (z - new_mean).square()).sum()
    if not bool(torch.isfinite(new_variance)) or new_variance <= 0:
        raise ArithmeticError('probit quadrature posterior collapsed; increase quadrature_points')
    # This is KL of the unprojected Bayesian posterior to its Gaussian prior.
    information = _nonnegative((posterior_weights * log_likelihood).sum() - log_evidence,
                               'realized arrival information')
    return new_mean, new_variance, log_evidence, information


def _arrival_update(mean, covariance, x, quiet_ticks, event, points):
    px = covariance @ x
    mu = x @ mean
    variance = _nonnegative(x @ px, 'arrival variance')
    new_mu, new_variance, evidence, information = _tilted_probit(
        mu, variance, quiet_ticks, event, points)
    if variance == 0:
        return mean.clone(), covariance.clone(), {
            'log_evidence': float(evidence), 'realized_information': float(information),
            'adf_information': 0., 'prior_mean': float(mu), 'prior_variance': 0.,
            'posterior_mean': float(new_mu), 'posterior_variance': 0.}
    delta = new_mu - mu
    updated_mean = mean + px * (delta / variance)
    updated_covariance = covariance + torch.outer(px, px) * ((new_variance - variance) / variance.square())
    updated_covariance = .5 * (updated_covariance + updated_covariance.T)
    ratio = new_variance / variance
    adf_kl = .5 * (ratio - 1 - ratio.log() + delta.square() / variance)
    _finite(updated_mean, 'updated arrival mean')
    _finite(updated_covariance, 'updated arrival covariance')
    return updated_mean, updated_covariance, {
        'log_evidence': float(evidence), 'realized_information': float(information),
        'adf_information': float(_nonnegative(adf_kl, 'ADF information')),
        'prior_mean': float(mu), 'prior_variance': float(variance),
        'posterior_mean': float(new_mu), 'posterior_variance': float(new_variance)}


class NormalInverseGammaMarks:
    """Independent scalar-output NIG regressions with shared feature covariance.

    sigma_j^2 ~ IG(alpha,beta_j); w_j|sigma_j^2 ~ N(mean_j,sigma_j^2 P).
    Conditional coordinates and independent output noises are fixed assumptions.
    The mark posterior is independent of arrival parameters. beta rescales both
    coefficient and observation uncertainty, so beta alone does not alter EIG.
    """

    def __init__(self, state, feature_dim=None, mark_dim=None, prior_variance=1.,
                 alpha=2., beta=1.):
        if not isinstance(state, dict):
            raise ValueError('mark state must be a dictionary')
        self.state = state
        if not state:
            d = _integer(feature_dim, 'feature_dim', 1)
            m = _integer(mark_dim, 'mark_dim', 1)
            state.update(format=MARK_FORMAT, feature_dim=d, mark_dim=m,
                         mean=torch.zeros(d, m, dtype=torch.float64),
                         covariance=torch.eye(d, dtype=torch.float64) * _positive(prior_variance, 'prior_variance'),
                         alpha=_positive(alpha, 'alpha'),
                         beta=torch.full((m,), _positive(beta, 'beta'), dtype=torch.float64), updates=0)
        if state.get('format') != MARK_FORMAT:
            raise ValueError('mark schema requires explicit migration')
        _required(state, ('feature_dim', 'mark_dim', 'mean', 'covariance', 'alpha', 'beta', 'updates'),
                  'mark state')
        d = _integer(state.get('feature_dim'), 'feature_dim', 1)
        m = _integer(state.get('mark_dim'), 'mark_dim', 1)
        if ((feature_dim is not None and feature_dim != d)
                or (mark_dim is not None and mark_dim != m)):
            raise ValueError('mark dimensions changed; migrate explicitly')
        _finite(state.get('mean'), 'mark mean')
        _finite(state.get('beta'), 'mark beta')
        if state['mean'].dtype != torch.float64 or state['mean'].shape != (d, m):
            raise ValueError('mark mean shape/dtype differs from schema')
        if (state['beta'].dtype != torch.float64 or state['beta'].shape != (m,)
                or bool((state['beta'] <= 0).any())):
            raise ValueError('mark beta shape or scale is invalid')
        _covariance(state.get('covariance'), d, 'mark covariance')
        if not (state['mean'].device == state['covariance'].device == state['beta'].device):
            raise ValueError('mark posterior tensors must share a device')
        _positive(state.get('alpha'), 'alpha')
        _integer(state.get('updates'), 'updates')

    @torch.no_grad()
    def expected_information_from_leverage(self, leverage):
        """Exact E_y KL(p(w,sigma2|y)||p(w,sigma2)); sums independent outputs."""
        leverage = torch.as_tensor(leverage, device=self.state['mean'].device, dtype=torch.float64)
        _finite(leverage, 'mark leverage')
        if bool((leverage < 0).any()):
            raise ValueError('mark leverage must be nonnegative')
        a = leverage.new_tensor(self.state['alpha'])
        delta_psi = torch.digamma(a + .5) - torch.digamma(a)
        base = (.5 * math.log(2.) + torch.lgamma(a) + math.lgamma(.5) - torch.lgamma(a + .5)
                + (a + .5) * delta_psi - .5 * math.log(2 * math.pi * math.e)
                + .5 * torch.digamma(a))
        return _nonnegative(self.state['mark_dim'] * (base + .5 * torch.log1p(leverage)),
                            'expected mark information')

    @torch.no_grad()
    def expected_information(self, x):
        x = _features(x, self.state['feature_dim'], self.state['mean'])
        leverage = _nonnegative(((x @ self.state['covariance']) * x).sum(-1), 'mark leverage')
        return self.expected_information_from_leverage(leverage)

    @torch.no_grad()
    def observe(self, x, y):
        d, m = self.state['feature_dim'], self.state['mark_dim']
        x = _vector(x, d, self.state['mean'], 'mark features')
        y = _vector(y, m, self.state['mean'], 'mark outcome')
        p, mean, a, beta = (self.state[k] for k in ('covariance', 'mean', 'alpha', 'beta'))
        px = p @ x
        v = _nonnegative(x @ px, 'mark leverage')
        denominator = 1 + v
        prediction = x @ mean
        residual = y - prediction
        new_a = a + .5
        new_beta = beta + .5 * residual.square() / denominator
        new_mean = mean + torch.outer(px / denominator, residual)
        new_p = p - torch.outer(px, px) / denominator
        new_p = .5 * (new_p + new_p.T)
        # KL inverse-gamma(new||old), plus expected conditional Gaussian KL.
        at = beta.new_tensor(a); ant = beta.new_tensor(new_a)
        ig_kl = (a * (new_beta.log() - beta.log()) - torch.lgamma(ant) + torch.lgamma(at)
                 + .5 * torch.digamma(ant) + new_a * (beta / new_beta - 1))
        conditional_kl = .5 * (-v / denominator + torch.log1p(v)
                               + (new_a / new_beta) * residual.square() * v / denominator.square())
        information = _nonnegative((ig_kl + conditional_kl).sum(), 'realized mark information')
        result = {'expected_information_before': float(self.expected_information_from_leverage(v)),
                  'realized_information': float(information),
                  'prediction_mean_before': prediction.clone(),
                  'prediction_scale2_before': (beta / a * denominator).clone(),
                  'prediction_df_before': 2 * a}
        _finite(new_mean, 'updated mark mean')
        _finite(new_beta, 'updated mark beta')
        self.state.update(mean=new_mean, covariance=new_p, alpha=new_a, beta=new_beta,
                          updates=self.state['updates'] + 1)
        return result


class ProbitEventBelief:
    """Durable stationary arrival belief with an optional independent event mark.

    begin/advance/resolve preserves a trial's original Gaussian prior. Aggregated
    survival is recomputed from that prior, making physical-tick batching and
    save/resume invariant despite ADF storage. No marks are learned without an
    event. A zero-tick quiet resolution provides no arrival evidence.
    """

    def __init__(self, state, feature_dim=None, prior_variance=1., quadrature_points=64,
                 mark_dim=0, mark_alpha=2., mark_beta=1.):
        if not isinstance(state, dict):
            raise ValueError('event state must be a dictionary')
        self.state = state
        if not state:
            d = _integer(feature_dim, 'feature_dim', 1)
            _hermite_numpy(quadrature_points)
            m = _integer(mark_dim, 'mark_dim')
            mark = {}
            if m:
                NormalInverseGammaMarks(mark, d, m, prior_variance, mark_alpha, mark_beta)
            state.update(format=FORMAT, feature_dim=d, quadrature_points=int(quadrature_points),
                         mean=torch.zeros(d, dtype=torch.float64),
                         covariance=torch.eye(d, dtype=torch.float64) * _positive(prior_variance, 'prior_variance'),
                         mark=mark if m else None, trial=None, updates=0,
                         quiet_ticks=0, events=0)
        if state.get('format') != FORMAT:
            raise ValueError('event schema requires explicit migration')
        _required(state, ('feature_dim', 'quadrature_points', 'mean', 'covariance', 'mark',
                          'trial', 'updates', 'quiet_ticks', 'events'), 'event state')
        d = _integer(state.get('feature_dim'), 'feature_dim', 1)
        if feature_dim is not None and feature_dim != d:
            raise ValueError('event feature dimension changed; migrate explicitly')
        _finite(state.get('mean'), 'arrival mean')
        if state['mean'].dtype != torch.float64 or state['mean'].shape != (d,):
            raise ValueError('arrival mean shape/dtype differs from schema')
        _covariance(state.get('covariance'), d, 'arrival covariance')
        if state['mean'].device != state['covariance'].device:
            raise ValueError('arrival posterior tensors must share a device')
        _hermite_numpy(state.get('quadrature_points'))
        for name in ('updates', 'quiet_ticks', 'events'):
            _integer(state.get(name), name)
        self.mark = NormalInverseGammaMarks(state['mark'], feature_dim=d) if state.get('mark') is not None else None
        if mark_dim and (self.mark is None or self.mark.state['mark_dim'] != mark_dim):
            raise ValueError('event mark dimension changed; migrate explicitly')
        trial = state.get('trial')
        if trial is not None:
            if not isinstance(trial, dict):
                raise ValueError('pending trial must be a dictionary')
            _required(trial, ('x', 'mean', 'covariance', 'quiet_ticks', 'log_evidence'), 'pending trial')
            _vector(trial.get('x'), d, state['mean'], 'pending trial features')
            _vector(trial.get('mean'), d, state['mean'], 'pending trial prior mean')
            _covariance(trial.get('covariance'), d, 'pending trial covariance')
            _integer(trial.get('quiet_ticks'), 'pending quiet_ticks')
            if not math.isfinite(float(trial.get('log_evidence'))):
                raise ValueError('pending log evidence is invalid')

    @torch.no_grad()
    def predict_from_moments(self, mean, variance, horizon, mark_leverage=None):
        if self.state['trial'] is not None and self.state['trial']['quiet_ticks']:
            raise ValueError('Resolve the pending survival trial before Gaussian prospective prediction; '
                             'its aggregate posterior is not exactly Gaussian')
        result = probit_next_event(mean, variance, horizon, self.state['quadrature_points'])
        if self.mark is None:
            if mark_leverage is not None:
                raise ValueError('mark leverage supplied to an arrival-only belief')
            mark_information = torch.zeros_like(result['timing_information'])
        else:
            if mark_leverage is None:
                raise ValueError('mark-enabled belief needs mark leverage')
            mark_information = self.mark.expected_information_from_leverage(mark_leverage)
            if mark_information.shape != result['timing_information'].shape:
                raise ValueError('mark leverage shape must match arrival moments')
        result['mark_information_if_event'] = mark_information
        result['expected_mark_information'] = result['event_probability'] * mark_information
        result['total_information'] = result['timing_information'] + result['expected_mark_information']
        return result

    @torch.no_grad()
    def predict(self, x, horizon, mark_x=None):
        x = _features(x, self.state['feature_dim'], self.state['mean'])
        variance = _nonnegative(((x @ self.state['covariance']) * x).sum(-1), 'arrival variance')
        leverage = None
        if self.mark is not None:
            mark_x = x if mark_x is None else _features(mark_x, self.state['feature_dim'], self.state['mean'])
            if mark_x.shape != x.shape:
                raise ValueError('mark features must match arrival feature batch')
            leverage = _nonnegative(((mark_x @ self.mark.state['covariance']) * mark_x).sum(-1), 'mark leverage')
        elif mark_x is not None:
            raise ValueError('mark features supplied to an arrival-only belief')
        return self.predict_from_moments(x @ self.state['mean'], variance, horizon, leverage)

    @torch.no_grad()
    def begin(self, x):
        if self.state['trial'] is not None:
            raise ValueError('resolve the pending trial before beginning another')
        x = _vector(x, self.state['feature_dim'], self.state['mean'], 'trial features')
        self.state['trial'] = {'x': x.clone(), 'mean': self.state['mean'].clone(),
                               'covariance': self.state['covariance'].clone(),
                               'quiet_ticks': 0, 'log_evidence': 0.}

    @torch.no_grad()
    def advance(self, quiet_ticks):
        quiet_ticks = _integer(quiet_ticks, 'quiet_ticks')
        trial = self.state['trial']
        if trial is None:
            raise ValueError('begin a trial before advancing survival')
        total = trial['quiet_ticks'] + quiet_ticks
        mean, covariance, result = _arrival_update(trial['mean'], trial['covariance'], trial['x'],
                                                  total, False, self.state['quadrature_points'])
        result['incremental_log_evidence'] = result['log_evidence'] - trial['log_evidence']
        self.state.update(mean=mean, covariance=covariance,
                          quiet_ticks=self.state['quiet_ticks'] + quiet_ticks)
        trial.update(quiet_ticks=total, log_evidence=result['log_evidence'])
        return result

    @torch.no_grad()
    def resolve(self, event, mark=None, mark_x=None, quiet_ticks=0):
        if not isinstance(event, bool):
            raise ValueError('event must be boolean')
        quiet_ticks = _integer(quiet_ticks, 'quiet_ticks')
        trial = self.state['trial']
        if trial is None:
            raise ValueError('begin a trial before resolving it')
        if mark is not None and (not event or self.mark is None):
            raise ValueError('an observed mark requires an event and a mark model')
        if mark_x is not None and mark is None:
            raise ValueError('mark features require an observed mark')
        if mark is not None:
            mark_x = trial['x'] if mark_x is None else _vector(
                mark_x, self.state['feature_dim'], self.state['mean'], 'mark features')
            mark = _vector(mark, self.mark.state['mark_dim'], self.state['mean'], 'mark outcome')
        mean, covariance, result = _arrival_update(trial['mean'], trial['covariance'], trial['x'],
                                                  trial['quiet_ticks'] + quiet_ticks, event,
                                                  self.state['quadrature_points'])
        result['incremental_log_evidence'] = result['log_evidence'] - trial['log_evidence']
        # Validate every input and compute arrival before the exact mark mutation.
        result['mark'] = self.mark.observe(mark_x, mark) if mark is not None else None
        self.state.update(mean=mean, covariance=covariance, trial=None,
                          quiet_ticks=self.state['quiet_ticks'] + quiet_ticks,
                          events=self.state['events'] + int(event), updates=self.state['updates'] + 1)
        return result

    @torch.no_grad()
    def observe(self, x, quiet_ticks, event, mark=None, mark_x=None):
        """One aggregate outcome; equivalent to begin + resolve.

        Call predict before acting; realized_information is outcome-dependent
        posterior KL, a diagnostic distinct from pre-observation expected value.
        """
        if self.state['trial'] is not None:
            raise ValueError('use resolve for the already pending trial')
        # Transactional for invalid inputs: begin itself must not leave a trial.
        self.begin(x)
        try:
            return self.resolve(event, mark, mark_x, quiet_ticks)
        except Exception:
            self.state['trial'] = None
            raise
