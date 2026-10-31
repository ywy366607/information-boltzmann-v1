"""Bayesian availability of one normal mother reply per ended child turn.

The caller supplies the world's decoded ``has_spoken`` flag, not a token count.
Quiet ticks, network polls and words inside a reply are not separate trials.
All state is JSON-safe; no Torch import or additional brain dynamics is needed.
"""
import copy
import math
from functools import lru_cache
from numbers import Integral


FORMAT = 'normal-reply-availability-beta-v1'
_EULER_GAMMA = 0.577215664901532860606512090082402431


def _count(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


@lru_cache(maxsize=256)
def _harmonic_remainder(n):
    """H_n - log(n) - gamma without cancellation at large integer counts."""
    if n <= 64:
        return math.fsum([*(1. / k for k in range(1, n + 1)),
                          -math.log(n), -_EULER_GAMMA])
    inverse = 1 / n
    square = inverse * inverse
    # Euler--Maclaurin remainder; the first omitted term is below floating-point
    # resolution here. This approximation affects arithmetic, not memory lifetime.
    return .5 * inverse + square * (-1 / 12 + square * (
        1 / 120 + square * (-1 / 252 + square * (1 / 240 - square / 132))))


def beta_occurrence_information(alpha, beta):
    """Exact Beta--Bernoulli expected parameter IG, in nats up to float precision.

    I = H_Bern(mu) - psi(s+1) + mu psi(alpha+1) + (1-mu) psi(beta+1).
    Integer harmonic remainders cancel the log/gamma terms analytically, avoiding
    subtraction of O(log(s)) terms to recover an O(1/s) information value.
    """
    alpha = _count(alpha, 'alpha', 1)
    beta = _count(beta, 'beta', 1)
    total = alpha + beta
    mean, other = alpha / total, beta / total
    value = math.fsum((mean * _harmonic_remainder(alpha),
                       other * _harmonic_remainder(beta),
                       -_harmonic_remainder(total)))
    return mean, max(0., value)


class ReplyAvailability:
    """Two Beta(1,1) beliefs and a single durable, idempotent trial ledger.

    ``begin`` happens when a child's turn ends, after decoded speech is known.
    ``settle(True)`` means the associated normal reply actually starts.
    ``settle(False)`` means an explicit world wait/call resolved that opportunity.
    ``settle(None)`` censors infrastructure failures; it does not teach rejection.

    Requests still pending must remain pending. The caller decides when a world
    outcome is observable; this helper never creates an artificial timeout.
    """

    def __init__(self, state):
        if not isinstance(state, dict):
            raise ValueError('Reply availability state must be a dictionary')
        if state.get('format', FORMAT) != FORMAT:
            raise ValueError('Reply availability schema requires explicit migration')
        buckets = state.get('buckets', {
            'quiet': {'alpha': 1, 'beta': 1},
            'spoken': {'alpha': 1, 'beta': 1},
        })
        if not isinstance(buckets, dict) or set(buckets) != {'quiet', 'spoken'}:
            raise ValueError('Reply availability needs quiet/spoken buckets')
        for name, bucket in buckets.items():
            if not isinstance(bucket, dict):
                raise ValueError(f'Invalid availability bucket: {name}')
            _count(bucket.get('alpha'), f'{name}.alpha', 1)
            _count(bucket.get('beta'), f'{name}.beta', 1)
        counter = _count(state.get('trial_counter', 0), 'trial_counter')
        current = state.get('current')
        if current is not None:
            if not isinstance(current, dict):
                raise ValueError('Current availability trial must be a dictionary')
            trial_id = _count(current.get('id'), 'current.id', 1)
            if trial_id > counter:
                raise ValueError('Current trial exceeds the cumulative trial counter')
            if not isinstance(current.get('has_spoken'), bool):
                raise ValueError('Current trial needs its decoded has_spoken flag')
            if current.get('status') not in ('pending', 'settled', 'censored'):
                raise ValueError('Unknown availability trial status')
            for key in ('p_before', 'eig_before'):
                value = current.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f'Current trial has invalid {key}')
            if not 0 <= current['p_before'] <= 1 or current['eig_before'] < 0:
                raise ValueError('Current availability prediction is outside its domain')
        for key in ('settled_trials', 'censored_trials'):
            _count(state.get(key, 0), key)
        self.state = state
        self.state.setdefault('format', FORMAT)
        self.state.setdefault('buckets', buckets)
        self.state.setdefault('trial_counter', counter)
        self.state.setdefault('current', current)
        self.state.setdefault('settled_trials', 0)
        self.state.setdefault('censored_trials', 0)

    @staticmethod
    def _bucket_name(has_spoken):
        if not isinstance(has_spoken, bool):
            raise ValueError('has_spoken must be the caller\'s decoded boolean flag')
        return 'spoken' if has_spoken else 'quiet'

    @property
    def pending(self):
        """The pending trial, or None; the settled last result remains in state."""
        current = self.state['current']
        return current if current is not None and current['status'] == 'pending' else None

    def expected(self, has_spoken):
        bucket = self.state['buckets'][self._bucket_name(has_spoken)]
        return beta_occurrence_information(bucket['alpha'], bucket['beta'])

    def begin(self, has_spoken):
        bucket_name = self._bucket_name(has_spoken)
        if self.pending is not None:
            raise ValueError('Do not overwrite an unresolved reply opportunity')
        probability, information = self.expected(has_spoken)
        trial_id = self.state['trial_counter'] + 1
        self.state['trial_counter'] = trial_id
        self.state['current'] = {
            'id': trial_id, 'has_spoken': has_spoken, 'bucket': bucket_name,
            'status': 'pending', 'p_before': probability, 'eig_before': information,
            'replied': None, 'information_awarded': 0.,
        }
        return trial_id

    def settle(self, replied, trial_id=None):
        """Return pre-observation expected occurrence IG once, or zero.

        A stale/duplicate/unknown trial ID cannot settle a newer opportunity.
        Censoring closes the opportunity without changing either Beta posterior.
        The return is EXPECTED parameter IG, not a claim of realized state entropy
        reduction or a semantic reward for speaking.
        """
        if replied is not None and not isinstance(replied, bool):
            raise ValueError('replied must be True, False or None for censoring')
        if trial_id is not None:
            trial_id = _count(trial_id, 'trial_id', 1)
        trial = self.state['current']
        if trial is None or trial['status'] != 'pending':
            return 0.
        if trial_id is not None and trial_id != trial['id']:
            return 0.
        if replied is None:
            trial.update(status='censored', replied=None, information_awarded=0.)
            self.state['censored_trials'] += 1
            return 0.
        information = trial['eig_before']
        bucket = self.state['buckets'][self._bucket_name(trial['has_spoken'])]
        bucket['alpha' if replied else 'beta'] += 1
        trial.update(status='settled', replied=replied, information_awarded=information,
                     p_after=self.expected(trial['has_spoken'])[0])
        self.state['settled_trials'] += 1
        return information

    def diagnostics(self):
        """Inspection copy, so a dashboard cannot alter the live trial ledger."""
        return {
            'format': FORMAT, 'trial_counter': self.state['trial_counter'],
            'settled_trials': self.state['settled_trials'],
            'censored_trials': self.state['censored_trials'],
            'current': copy.deepcopy(self.state['current']),
            'buckets': {name: {**bucket,
                               'p_reply': self.expected(name == 'spoken')[0],
                               'occurrence_eig': self.expected(name == 'spoken')[1]}
                        for name, bucket in self.state['buckets'].items()},
        }
