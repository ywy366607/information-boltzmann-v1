"""Directional assumed-density filtering of a persistent latent world context.

Native compartment fields may encode the recognition mean of this context.
Its covariance is uncertainty about that modeled context, never uncertainty in
the exactly saved physical phi state. Three Gaussian quadrature nodes provide
a directional approximation; this is not full multivariate Bayes inference.
"""
import copy
import math
from numbers import Integral

import torch


FORMAT = 'native-directional-context-v1'
NODES = torch.tensor([-math.sqrt(3.), 0., math.sqrt(3.)], dtype=torch.float64)
WEIGHTS = torch.tensor([1. / 6., 2. / 3., 1. / 6.], dtype=torch.float64)


def _tensor(value, shape, name):
    value = torch.as_tensor(value).detach().to(device='cpu', dtype=torch.float64).clone()
    if value.shape != shape or not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name} must be finite with shape {shape}')
    return value


def _nonnegative(value, name):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return value


def _ticks(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError('ticks must be a nonnegative integer')
    return int(value)


def _covariance(value, dimension):
    value = _tensor(value, (dimension, dimension), 'covariance')
    if not torch.allclose(value, value.T, rtol=1e-10, atol=1e-12):
        raise ValueError('covariance must be symmetric')
    value = (value + value.T) * .5
    eigenvalues = torch.linalg.eigvalsh(value)
    tolerance = 1e-10 * max(1., float(value.abs().max()))
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError('covariance must be positive semidefinite')
    return value


def _distribution(log_probs):
    lp = torch.as_tensor(log_probs).detach().to(device='cpu', dtype=torch.float64).clone()
    if lp.ndim != 2 or lp.shape[0] != 3 or lp.shape[1] < 1:
        raise ValueError('categorical forecasts must have shape [3, vocabulary]')
    if bool((torch.isnan(lp) | torch.isposinf(lp)).any()):
        raise ValueError('categorical forecasts contain invalid log probabilities')
    if not torch.allclose(torch.logsumexp(lp, -1), torch.zeros(3, dtype=lp.dtype),
                          rtol=0, atol=1e-5):
        raise ValueError('categorical forecasts must be normalized')
    return lp


def categorical_information(log_probs):
    """Quadrature I(context node; external category), distinct from noise entropy."""
    lp = _distribution(log_probs)
    log_mix = torch.logsumexp(lp + WEIGHTS.log()[:, None], 0)
    ratio = torch.where(torch.isfinite(lp), lp - log_mix, torch.zeros_like(lp))
    information = (WEIGHTS[:, None] * lp.exp() * ratio).sum().clamp_min(0.)
    return {'log_predictive': log_mix, 'expected_information': float(information)}


class DirectionalContextFilter:
    """CPU-double Gaussian context filter with a learned Brownian transition.

    Initial variance and process variance are explicit dimensionless modeling
    choices, not biological measurements or prescribed memory lifetimes.
    ``begin_transition`` commits real elapsed ticks once; ``forecast`` archives
    pre-observation probabilities; ``observe`` consumes exactly one real label.
    """

    def __init__(self, state=None, *, dimension, initial_variance=1., process_variance=1e-4):
        if isinstance(dimension, bool) or not isinstance(dimension, Integral) or dimension < 1:
            raise ValueError('dimension must be a positive integer')
        self.dimension = int(dimension)
        if state is None:
            variance = _nonnegative(initial_variance, 'initial_variance')
            q = _nonnegative(process_variance, 'process_variance')
            state = {'format': FORMAT, 'dimension': self.dimension,
                     'mean': torch.zeros(self.dimension, dtype=torch.float64),
                     'covariance': torch.eye(self.dimension, dtype=torch.float64) * variance,
                     'process_variance': q, 'process_em_sum': 0., 'process_em_count': 0,
                     'clock_ticks': 0, 'updates': 0, 'pending': None}
        if state.get('format') != FORMAT or state.get('dimension') != self.dimension:
            raise ValueError('context filter schema/dimension changed; migrate explicitly')
        self.state = state
        state['mean'] = _tensor(state['mean'], (self.dimension,), 'mean')
        state['covariance'] = _covariance(state['covariance'], self.dimension)
        state['process_variance'] = _nonnegative(state['process_variance'], 'process_variance')
        state['process_em_sum'] = _nonnegative(state['process_em_sum'], 'process_em_sum')
        for name in ('process_em_count', 'clock_ticks', 'updates'):
            state[name] = _ticks(state[name])
        pending = state.get('pending')
        if pending is not None:
            pending['mean'] = _tensor(pending['mean'], (self.dimension,), 'pending mean')
            pending['P0'] = _covariance(pending['P0'], self.dimension)
            pending['C'] = _covariance(pending['C'], self.dimension)
            pending['v'] = _tensor(pending['v'], (self.dimension,), 'pending direction')
            pending['ticks'] = _ticks(pending['ticks'])
            pending['q'] = _nonnegative(pending['q'], 'pending process variance')
            if pending.get('log_probs') is not None:
                pending['log_probs'] = _distribution(pending['log_probs'])

    @property
    def mean(self):
        return self.state['mean']

    @property
    def covariance(self):
        return self.state['covariance']

    def state_dict(self):
        return copy.deepcopy(self.state)

    def begin_transition(self, mean, ticks, direction):
        """Use an actual native recognition mean and a label-independent direction.

        The supplied direction is in whitened coordinates. A symmetric PSD
        square root supports exactly zero variance without adding fake noise.
        """
        if self.state['pending'] is not None:
            raise ValueError('finish the pending transition before advancing again')
        mean = _tensor(mean, (self.dimension,), 'recognition mean')
        ticks = _ticks(ticks)
        u = _tensor(direction, (self.dimension,), 'direction')
        norm = float(u.norm())
        if norm == 0:
            raise ValueError('direction must be nonzero')
        u /= norm
        P0 = self.covariance.clone()
        q = self.state['process_variance']
        C = P0 + torch.eye(self.dimension, dtype=torch.float64) * (q * ticks)
        eigenvalues, vectors = torch.linalg.eigh(C)
        root = (vectors * eigenvalues.clamp_min(0).sqrt()[None, :]) @ vectors.T
        v = root @ u
        self.state.update(mean=mean.clone(), covariance=C.clone(),
                          clock_ticks=self.state['clock_ticks'] + ticks,
                          pending={'mean': mean, 'P0': P0, 'C': C, 'q': q,
                                   'ticks': ticks, 'v': v, 'log_probs': None})
        return mean[None, :] + NODES[:, None] * v[None, :]

    def forecast(self, log_probs):
        pending = self.state['pending']
        if pending is None:
            raise ValueError('begin a real transition before forecasting')
        lp = _distribution(log_probs)
        if pending['log_probs'] is not None and not torch.equal(lp, pending['log_probs']):
            raise ValueError('cannot replace a committed pre-observation forecast')
        pending['log_probs'] = lp
        return categorical_information(lp)

    def discard_unobserved(self):
        """End a known-self/censored trial without conditioning on a fake label.

        Real elapsed time and its propagated prior remain committed. No Bayes
        update, process-variance EM sample or observation count is manufactured.
        Speculative callers use this on a state clone, never the actual filter.
        """
        if self.state['pending'] is None:
            raise ValueError('no pending transition to discard')
        self.state['pending'] = None

    def cancel_pending(self):
        self.discard_unobserved()

    def observe(self, target):
        """Directional Bayes moments and a conditional Brownian-increment EM step."""
        p = self.state['pending']
        if p is None or p['log_probs'] is None:
            raise ValueError('a committed pre-observation forecast is required')
        lp = p['log_probs']
        if isinstance(target, bool) or not isinstance(target, Integral) or not 0 <= target < lp.shape[1]:
            raise ValueError('target must be an integer in the categorical vocabulary')
        joint = WEIGHTS.log() + lp[:, int(target)]
        logp = torch.logsumexp(joint, 0)
        if not bool(torch.isfinite(logp)):
            raise ValueError('all context nodes assign zero probability to the observation')
        posterior_log = joint - logp
        posterior = posterior_log.exp()
        xi_mean = (posterior * NODES).sum()
        xi_variance = (posterior * (NODES - xi_mean).square()).sum()
        m = p['mean'] + p['v'] * xi_mean
        P = p['C'] + torch.outer(p['v'], p['v']) * (xi_variance - 1.)
        P = _covariance(P, self.dimension)
        # E[ww^T|Y] for x=x0+w, x0~N(m0,P0), w~N(0,q*dt I).
        # This is a state-space EM moment, not a raw-surprisal novelty injection.
        q_step = p['q'] * p['ticks']
        em_q = None
        if p['ticks'] > 0:
            if q_step == 0:
                em_q = 0.
            else:
                inverse = torch.linalg.pinv(p['C'], hermitian=True)
                dm = m - p['mean']
                second = P + torch.outer(dm, dm)
                K = q_step * inverse
                ww = (torch.eye(self.dimension, dtype=torch.float64) * q_step
                      - q_step * q_step * inverse + K @ second @ K.T)
                em_q = max(0., float(torch.trace(ww)) / self.dimension / p['ticks'])
            self.state['process_em_sum'] += em_q
            self.state['process_em_count'] += 1
            self.state['process_variance'] = (self.state['process_em_sum']
                                              / self.state['process_em_count'])
        realized = (posterior * (posterior_log - WEIGHTS.log())).sum().clamp_min(0.)
        self.state.update(mean=m, covariance=P, pending=None, updates=self.state['updates'] + 1)
        return {'prequential_logp': float(logp), 'realized_information': float(realized),
                'posterior_weights': posterior, 'mean': m.clone(), 'covariance': P.clone(),
                'process_variance_em': em_q, 'process_variance': self.state['process_variance']}
