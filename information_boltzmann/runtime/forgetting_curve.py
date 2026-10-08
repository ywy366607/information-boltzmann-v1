"""First-revisit measurements on independent episodes of one active stream.

Each episode is revisited once. Storage is bounded by the largest configured
interval and cohort cadence, independent of lifetime; raw records live on disk.
All scores are pre-target-update, but later tokens in a revisit also benefit
from learning earlier tokens. Opening and whole-episode risk are kept distinct.
"""
from __future__ import annotations
import copy
import math


class FirstRevisitCurve:
    def __init__(self, lags, *, episode_tokens=128, cohort_every=5000, train_cursor=0):
        self.lags = tuple(sorted(set(int(x) for x in lags)))
        if not self.lags or min(self.lags) < 1 or episode_tokens < 3 or cohort_every < episode_tokens * len(self.lags):
            raise ValueError('Positive lags and nonoverlapping source cohorts required')
        self.episode_tokens, self.cohort_every = int(episode_tokens), int(cohort_every)
        self.next_cohort_cursor = (train_cursor // cohort_every + 1) * cohort_every
        self.cohort = 0
        self.capture = None
        self.pending = []
        self.excluded_interrupted_sources = 0
        self.corrected_boundary_starts = 0
        self.summary = {str(lag): {'episodes': 0, 'gap_sum': 0., 'gap_sq_sum': 0.,
            'opening_gap_sum': 0., 'initial_sum': 0., 'revisit_sum': 0.,
            'actual_interval_sum': 0, 'actual_interval_min': None, 'actual_interval_max': None}
            for lag in self.lags}

    def fresh_observation(self, token, score, *, event, train_cursor, updates):
        if self.capture is None and train_cursor > self.next_cohort_cursor:
            self.capture = {'cohort': self.cohort, 'slot': 0, 'tokens': [], 'scores': [],
                            'source_start_train_cursor': train_cursor, 'source_start_event': event,
                            'source_start_updates': updates}
            self.next_cohort_cursor += self.cohort_every
            self.cohort += 1
        if self.capture is None:
            return
        c = self.capture
        if not c['tokens']:
            # A due revisit may have run after the previous slot finished.
            # Start at the actual first observation, not its anticipated event.
            c.update(source_start_train_cursor=train_cursor,source_start_event=event,
                     source_start_updates=updates)
        c['tokens'].append(int(token)); c['scores'].append(float(score))
        if len(c['tokens']) < self.episode_tokens:
            return
        lag = self.lags[(c['slot'] + c['cohort']) % len(self.lags)]
        episode = copy.deepcopy(c)
        episode.update(target_lag=lag, source_end_event=event, due_event=event + lag,
                       source_end_updates=updates, episode_id=f"c{c['cohort']}-s{c['slot']}")
        self.pending.append(episode)
        if c['slot'] + 1 == len(self.lags):
            self.capture = None
        else:
            self.capture = {'cohort': c['cohort'], 'slot': c['slot']+1, 'tokens': [], 'scores': [],
                'source_start_train_cursor': train_cursor+1, 'source_start_event': event+1,
                'source_start_updates': updates}

    def pop_due(self, event):
        due = [episode for episode in self.pending if episode['due_event'] <= event]
        if not due:
            return None
        episode = min(due, key=lambda x: x['due_event'])
        self.pending.remove(episode)
        return episode

    def record_revisit(self, episode, scores, *, event_start, event_end, updates_start, updates_end):
        if len(scores) != len(episode['scores']):
            raise ValueError('First and revisit exposures must have matching targets')
        # First target is a different physical context bridge on each encounter;
        # record it explicitly, and compare identical within-episode transitions.
        first, revisit = episode['scores'][1:], list(scores)[1:]
        initial = sum(first) / len(first); recalled = sum(revisit) / len(revisit)
        opening = min(16, len(first))
        opening_gap = (sum(revisit[:opening])-sum(first[:opening])) / opening
        interval = event_start-episode['source_end_event']
        row = {'episode_id': episode['episode_id'], 'target_lag_events': episode['target_lag'],
            'actual_intervening_events': interval, 'source_event_interval': [episode['source_start_event'],episode['source_end_event']],
            'source_start_train_cursor': episode['source_start_train_cursor'],
            'revisit_event_interval': [event_start+1,event_end],
            'optimizer_updates_during_interval': updates_start-episode['source_end_updates'],
            'optimizer_updates_during_revisit': updates_end-updates_start,
            'initial_nll': initial, 'first_revisit_nll': recalled, 'forgetting_nll_gap': recalled-initial,
            'opening_16_forgetting_gap': opening_gap,
            'first_matched_prediction_initial_nll': first[0], 'first_matched_prediction_revisit_nll': revisit[0],
            'initial_bridge_nll': episode['scores'][0], 'revisit_bridge_nll': scores[0],
            'initial_curve': list(episode['scores']), 'first_revisit_curve': list(scores),
            'protocol': 'one active individual; independent source episode per lag; one revisit; all scores pre-target-update',
            'interpretation': 'active first-revisit retention/relearning curve; ongoing learning within an exposure remains active'}
        row['source_contiguous'] = (episode['source_end_event'] - episode['source_start_event'] + 1 == len(scores))
        if not row['source_contiguous']:
            # Another measured exposure can interrupt collection of fresh text.
            # Preserve its raw record, but do not pool mismatched transitions.
            self.excluded_interrupted_sources += 1
            row['exclusion_reason'] = 'initial source interrupted by intervening learning events'
            return row
        self.aggregate_record(row)
        return row

    def aggregate_record(self, row):
        stats = self.summary[str(row['target_lag_events'])]
        stats['episodes'] += 1
        gap = row['forgetting_nll_gap']
        stats['gap_sum'] += gap; stats['gap_sq_sum'] += gap*gap
        stats['opening_gap_sum'] += row['opening_16_forgetting_gap']
        stats['initial_sum'] += row['initial_nll']; stats['revisit_sum'] += row['first_revisit_nll']
        interval = row['actual_intervening_events']
        stats['actual_interval_sum'] += interval
        stats['actual_interval_min'] = interval if stats['actual_interval_min'] is None else min(interval,stats['actual_interval_min'])
        stats['actual_interval_max'] = interval if stats['actual_interval_max'] is None else max(interval,stats['actual_interval_max'])

    def rebuild_summary(self, records, *, through_event):
        """Reconstruct only completed measurements in the restored lifetime."""
        for stats in self.summary.values():
            for key in stats:
                stats[key] = None if key in ('actual_interval_min', 'actual_interval_max') else 0
        self.excluded_interrupted_sources = 0
        self.corrected_boundary_starts = 0
        recent_revisits = []
        horizon = 2 * max(self.lags) + self.episode_tokens * (len(self.lags) + 2)
        def actual_start(start):
            for left, right in recent_revisits:
                if left <= start <= right:
                    start = right + 1
            return start
        for row in records:
            if row['revisit_event_interval'][1] > through_event:
                continue
            row = dict(row)
            first, last = row['source_event_interval']
            corrected = actual_start(first)
            if corrected != first:
                self.corrected_boundary_starts += 1
                first = corrected
                row['source_event_interval'] = [first, last]
            if last - first + 1 != len(row['initial_curve']):
                self.excluded_interrupted_sources += 1
            else:
                self.aggregate_record(row)
            recent_revisits.append(tuple(row['revisit_event_interval']))
            recent_revisits = [pair for pair in recent_revisits
                               if pair[1] >= row['revisit_event_interval'][1] - horizon]
        # Early checkpoints anticipated a slot's start before dispatching an
        # already due revisit. Recover the actual start from executed traffic.
        for episode in self.pending:
            episode['source_start_event'] = actual_start(episode['source_start_event'])

    def curve(self):
        points = []
        for lag in self.lags:
            stats = self.summary[str(lag)]; n = stats['episodes']
            if n == 0:
                points.append({'target_lag_events': lag, 'episodes': 0})
                continue
            mean = stats['gap_sum']/n
            variance = max(0.,(stats['gap_sq_sum']-n*mean*mean)/(n-1)) if n > 1 else None
            points.append({'target_lag_events': lag, 'episodes': n,
                'actual_interval_mean': stats['actual_interval_sum']/n,
                'actual_interval_min': stats['actual_interval_min'], 'actual_interval_max': stats['actual_interval_max'],
                'initial_nll': stats['initial_sum']/n, 'first_revisit_nll': stats['revisit_sum']/n,
                'forgetting_nll_gap': mean, 'opening_16_forgetting_gap': stats['opening_gap_sum']/n,
                'episode_standard_error': math.sqrt(variance/n) if variance is not None else None})
        return {'points': points, 'pending_episodes': len(self.pending),
                'excluded_interrupted_sources': self.excluded_interrupted_sources,
                'corrected_boundary_starts': self.corrected_boundary_starts,
                'positive_gap_means': 'worse on first revisit; negative means improvement',
                'measurement': 'pre-target-update online scores, separated bridge; independent episodes, rotated lag assignment',
                'scope': 'includes natural context re-entry and active relearning; no forced monotonic/exponential fit'}

    def state_dict(self):
        return copy.deepcopy(self.__dict__)

    def load_state_dict(self, state):
        if tuple(state['lags']) != self.lags or state['episode_tokens'] != self.episode_tokens or state['cohort_every'] != self.cohort_every:
            raise ValueError('Keep the configured measurement protocol on resume')
        self.__dict__.update(copy.deepcopy(state))
