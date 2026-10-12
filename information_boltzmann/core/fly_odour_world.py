"""A closed-loop odour world that runs alongside the text stream (design: FLY_CONSEQUENCE_WORLD_DESIGN_20261012.html,
section "odour").  Text never pauses; odours enter through the olfactory receptor types; the fly acts through real
descending neurons; acts change what it smells next; contact delivers taste and the unconditioned dopamine.

Life segments of SEGMENT ticks: two odours, each a pair of receptor types drawn from POOL (types without a strong
innate meaning), each sweet or bitter at random; in REVERSE_P of the segments both valences flip at the midpoint.  So a
fixed odour -> act wiring earns nothing on average: only memory formed within the segment can.

Encounter: after a geometric gap (mean GAP), one of the two odours arrives as a plume: every tick a whiff with
probability p (0.3-0.7, per encounter: the evidence level) at intensity c / max(d, D_MIN) rheobase units on its
receptor cells, d the distance to the source (1 at arrival).
  approach (the approach pool wins the race): d falls to 0 over APPROACH ticks; contact: the taste cells of the
           odour's valence are driven for 4 ticks, the dopamine cells of that valence 3 ticks later for 4 ticks; the
           odour stays on CONTACT_ODOUR ticks, then the encounter ends.
  avoid    (the avoid pool wins): the odour fades (factor exp(-1/4) per tick) and the encounter ends, no outcome.
  no act:  every tick the odour leaves with hazard LEAVE or the fly drifts into contact with hazard DRIFT.
Race: an act happens when a pool's spike count over the last WINDOW ticks reaches the bound and exceeds the other's.

The world is stepped from the realized (hard) spikes of the act pools; everything it injects is recorded on a tape,
so a checkpoint recompute replays exactly the same drive.  The fitness of an encounter (sweet contact +1, bitter -1.5)
is seen only by BPTT (evolution); inside the brain there are only odour, taste and dopamine.
"""
import math
import numpy as np
import torch

POOL = ('DM3', 'DM4', 'DM6', 'DL5', 'VA3', 'VA4', 'VA5', 'VM2', 'VM3', 'DC1', 'DC2', 'DC4')
SWEET_TASTE, BITTER_TASTE = ('LgAG3', 'LB1b', 'PhG13'), ('LgAG1', 'PhG8', 'dorsal_tpGRN')
SWEET_DOPAMINE, BITTER_DOPAMINE = ('PAM01', 'PAM02', 'PAM11'), ('PPL101', 'PPL103', 'PPL106')
APPROACH_POOL, AVOID_POOL = ('DNa02', 'DNa03', 'DNp09'), ('MDN', 'DNp42')
SEGMENT, REVERSE_P, GAP = 3000, 0.25, 64
APPROACH, D_MIN, CONTACT_ODOUR, FADE = 8, 0.25, 6, math.exp(-1 / 4)
LEAVE, DRIFT, WINDOW = 1 / 48, 1 / 96, 8
TASTE_TICKS, DOPAMINE_DELAY, DOPAMINE_TICKS = 4, 3, 4
REWARD = {'sweet': 1.0, 'bitter': -1.5}


def cells_of(types, ann_type):
    return np.flatnonzero(np.isin(ann_type, list(types)))


class OdourWorld:
    def __init__(self, model, graph_npz, annotations, *, seed=0, odour_drive=4.0, outcome_drive=8.0, bound=3):
        import pandas as pd
        graph = np.load(graph_npz, allow_pickle=True)
        typ = pd.read_feather(annotations).set_index('bodyId').reindex(graph['neuron_body_ids'])['type'].astype(str).values
        self.device = model.read_indices.device
        self.n = len(typ)
        self.receptor = {g: np.flatnonzero(typ == f'ORN_{g}') for g in POOL}
        self.taste = {'sweet': cells_of(SWEET_TASTE, typ), 'bitter': cells_of(BITTER_TASTE, typ)}
        self.dopamine = {'sweet': cells_of(SWEET_DOPAMINE, typ), 'bitter': cells_of(BITTER_DOPAMINE, typ)}
        self.pools = (cells_of(APPROACH_POOL, typ), cells_of(AVOID_POOL, typ))
        for name, cells in [*self.receptor.items(), *self.taste.items(), *self.dopamine.items()]:
            assert len(cells), name
        assert all(len(p) for p in self.pools)
        index = np.concatenate(self.pools)
        model.register_buffer('act_pool_index', torch.as_tensor(index, device=self.device), persistent=False)
        self.split = len(self.pools[0])
        with torch.no_grad():
            rates0 = model.get_decay_rates()[0].expand(1, model.n_neurons)
            self.rheobase = (model.get_thresholds().expand(1, model.n_neurons)
                             * -torch.log(rates0.clamp(1e-5, 1 - 1e-7)))[0].float()
        self.odour_drive, self.outcome_drive, self.bound = float(odour_drive), float(outcome_drive), int(bound)
        self.rng = np.random.default_rng(seed)
        self.tick = 0
        self.history = [np.zeros(2, np.int64) for _ in range(WINDOW)]
        self.encounter_id = 0
        self.log = []                       # one row per finished encounter
        self._new_segment()
        self._new_gap()

    # ---- world state -------------------------------------------------------------------------------------------
    def _new_segment(self):
        self.segment_start = self.tick
        types = self.rng.choice(POOL, 4, replace=False)
        self.odours = [tuple(types[:2]), tuple(types[2:])]
        self.valence = [str(self.rng.choice(['sweet', 'bitter'])) for _ in range(2)]
        self.reverse = bool(self.rng.random() < REVERSE_P)
        self.reversed = False
        self.outcomes = [0, 0]               # outcomes experienced per odour in this segment (for revisit analysis)

    def _new_gap(self):
        self.phase, self.countdown = 'gap', int(self.rng.geometric(1 / GAP))
        self.pending = []                    # (tick, kind, valence)

    def _new_encounter(self):
        self.encounter_id += 1
        self.odour = int(self.rng.integers(2))
        self.whiff_p = float(self.rng.uniform(0.3, 0.7))
        self.distance, self.amplitude, self.phase = 1.0, 1.0, 'present'
        self.enc = {'id': self.encounter_id, 'segment_start': self.segment_start, 'odour': self.odour,
                    'valence': self.valence[self.odour], 'whiff_p': self.whiff_p, 'start': self.tick,
                    'prior_outcomes': self.outcomes[self.odour], 'reversed': self.reversed, 'act': None,
                    'latency': None, 'end': None, 'outcome': None}

    def value(self):
        """Fitness of contact with the present odour, as evolution sees it."""
        return REWARD[self.valence[self.odour]]

    # ---- one tick ------------------------------------------------------------------------------------------------
    def drive(self):
        """The external current for the coming tick, and what the coming tick is for the act objective: live (an act
        is possible and would count), the fitness of contact and the encounter id."""
        current = None
        cells, amounts = [], []
        if self.phase in ('present', 'approach', 'contact', 'avoid') and self.amplitude > 0:
            if self.rng.random() < self.whiff_p or self.phase == 'contact':
                level = self.odour_drive * self.amplitude / max(self.distance, D_MIN)
                for g in self.odours[self.odour]:
                    cells.append(self.receptor[g]); amounts.append(np.full(len(self.receptor[g]), level))
        for when, kind, valence in self.pending:
            if when == self.tick:
                source = self.taste[valence] if kind == 'taste' else self.dopamine[valence]
                cells.append(source); amounts.append(np.full(len(source), self.outcome_drive))
        if cells:
            idx = torch.as_tensor(np.concatenate(cells), device=self.device)
            amt = torch.as_tensor(np.concatenate(amounts), dtype=torch.float32, device=self.device)
            current = torch.zeros(self.n, device=self.device)
            current.index_add_(0, idx, amt * self.rheobase[idx])
            current = current.unsqueeze(0)
        live = self.phase == 'present'
        return current, {'live': live, 'value': self.value() if live else 0.0,
                         'encounter': self.encounter_id if live else 0}

    def observe(self, counts):
        """After the tick: the realized spike counts of the two pools; advance the world."""
        self.history = self.history[1:] + [np.asarray(counts, np.int64)]
        window = np.sum(self.history, 0)
        act = None
        if window[0] >= self.bound and window[0] > window[1]:
            act = 'approach'
        elif window[1] >= self.bound and window[1] > window[0]:
            act = 'avoid'
        self.pending = [p for p in self.pending if p[0] > self.tick]
        self.tick += 1
        if self.reverse and not self.reversed and self.tick - self.segment_start >= SEGMENT // 2:
            self.valence = ['bitter' if v == 'sweet' else 'sweet' for v in self.valence]
            self.reversed = True
        if self.phase == 'gap':
            self.countdown -= 1
            if self.tick - self.segment_start >= SEGMENT:
                self._new_segment()
            if self.countdown <= 0:
                self._new_encounter()
        elif self.phase == 'present':
            if act is not None:
                self.enc['act'], self.enc['latency'] = act, self.tick - self.enc['start']
                self.phase = 'approach' if act == 'approach' else 'avoid'
                self.steps = 0
            else:
                u = self.rng.random()
                if u < DRIFT:
                    self.enc['act'], self.enc['latency'] = 'drift', self.tick - self.enc['start']
                    self._contact()
                elif u < DRIFT + LEAVE:
                    self.enc['act'], self.enc['latency'] = 'none', self.tick - self.enc['start']
                    self._end(None)
        elif self.phase == 'approach':
            self.steps += 1
            self.distance = max(0.0, 1.0 - self.steps / APPROACH)
            if self.steps >= APPROACH:
                self._contact()
        elif self.phase == 'avoid':
            self.amplitude *= FADE
            if self.amplitude < 0.05:
                self._end(None)
        elif self.phase == 'contact':
            self.steps += 1
            if self.steps >= CONTACT_ODOUR:
                self._end(self.valence[self.odour])

    def _contact(self):
        valence = self.valence[self.odour]
        self.phase, self.steps, self.distance = 'contact', 0, 0.0
        self.pending += [(self.tick + k, 'taste', valence) for k in range(TASTE_TICKS)]
        self.pending += [(self.tick + DOPAMINE_DELAY + k, 'dopamine', valence) for k in range(DOPAMINE_TICKS)]
        self.outcomes[self.odour] += 1

    def _end(self, outcome):
        self.enc['end'], self.enc['outcome'] = self.tick, outcome
        self.log.append(self.enc)
        pending = [p for p in self.pending if p[0] >= self.tick]      # outcome pulses still due run out in the gap
        self._new_gap()
        self.pending = pending


def act_hazards(pool_ticks, history, bound, scale=1.0):
    """Act hazards from the pools' per-tick spike sums (0/1 spikes with surrogate gradients) [T, 2], with the
    WINDOW - 1 detached sums before the first tick.  Forward: the realized crossing (exactly the world's rule, 0 or 1);
    backward: through the smooth race h_app = sig(n_app - B + 1/2) sig(n_app - n_av) (straight through)."""
    full = torch.cat([history.to(pool_ticks), pool_ticks], 0)
    window = full.unfold(0, WINDOW, 1).sum(-1)[-pool_ticks.size(0):]          # [T, 2]
    a, v = window[:, 0], window[:, 1]
    soft_app = torch.sigmoid((a - bound + 0.5) / scale) * torch.sigmoid((a - v) / scale)
    soft_av = torch.sigmoid((v - bound + 0.5) / scale) * torch.sigmoid((v - a) / scale)
    with torch.no_grad():
        hard_app = ((a >= bound) & (a > v)).to(a.dtype)
        hard_av = ((v >= bound) & (v > a)).to(a.dtype)
    return hard_app + soft_app - soft_app.detach(), hard_av + soft_av - soft_av.detach()


def act_return(h_app, h_av, records, carry):
    """Expected fitness of the live ticks, encounter by encounter:  at a live tick, with S the probability of no act
    yet,  J += S (h_app r + (1 - h_app - h_av) DRIFT r),  S <- S (1 - h_app - h_av)(1 - DRIFT - LEAVE);  the ticks after
    the window (or after the realized act) are valued as never acting, S r DRIFT / (DRIFT + LEAVE).  carry: the
    encounter's S at the window start (detached).  Returns J (differentiable) and the carry for the next window."""
    J = h_app.new_zeros(())
    S, current, value = None, 0, 0.0
    tail = DRIFT / (DRIFT + LEAVE)
    for t, rec in enumerate(records):
        if not rec['live']:
            if current:                      # the encounter's live time ended (act, contact or leave)
                J = J + S * value * tail
                current = 0
            continue
        if rec['encounter'] != current:
            if current:
                J = J + S * value * tail
            current, value = rec['encounter'], rec['value']
            S = h_app.new_tensor(carry.get(current, 1.0))
        J = J + S * (h_app[t] * value + (1 - h_app[t] - h_av[t]) * DRIFT * value)
        S = S * (1 - h_app[t] - h_av[t]) * (1 - DRIFT - LEAVE)
    carry = {}
    if current:
        J = J + S * value * tail
        carry[current] = float(S.detach())
    return J, carry
