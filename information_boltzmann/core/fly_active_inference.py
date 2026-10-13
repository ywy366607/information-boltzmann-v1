"""Unified perception and action (GOAL: docs/information_boltzmann/FLY_ACTIVE_INFERENCE_GOAL_20261013.md).

One act head, two acts, read from the membranes of all output cells (the cells the language read uses; the model
chooses which cells act):
  channel 0  EYES   -- the precision (gain) of the text input, e = sigmoid(G_eye), decided from the previous tick's
                       integrator before the input is written.  Closed eyes: less interference, but a word arriving
                       meanwhile is written with gain e ~ 0 -- it is missed (the world never waits; the word is still
                       scored by the language objective).
  channel 1  OUTPUT -- commit the expectation standing on the output surface: a leaky integrator racing to a bound
                       (forward: the realized crossing, taped for checkpoint recompute; backward: straight through
                       with the soft crossing's slope at the bound); reset after a commit.
Objective terms (evolution's fitness for BPTT, nothing injected into the brain):
  perception VFE = the survival-weighted language CE (accuracy) + SIGReg on the read latents (complexity);
  epistemic value of looking = mean over word onsets of the information gain of seeing the word with gain e,
  0.5 * log(1 + e / (1 + n_seen(word))) (e pseudo-counts added to a count of 1 + n; concave in e, fading as the word
  becomes familiar), n_seen counting only what was actually seen (incremented by e);
  pragmatic value of output: the accuracy of committed predictions (M2).
"""
import math
import numpy as np
import torch

BOUND, SOFT = 1.0, 0.25
AI_PARAMETER_NAMES = ('ai_w', 'ai_b', 'ai_logit_leak')
AI_STATE_NAMES = AI_PARAMETER_NAMES + ('ai_mu', 'ai_sigma', 'ai_w_scale', 'ai_da_ref', 'ai_habit')
MOTOR_CONSTANTS = ('ai_mu', 'ai_sigma', 'ai_da_ref')
AI_RAW_SD = 0.02            # raw W of sd 0.02: an Adam step (~2e-4) is ~1% of the read


def attach_ai_head(model, init, eye_open=0.95):
    """init: dict with act_mu, act_sigma, act_w [2, n_read], act_b [2] (scripts/ib/fly_interface/act_calibrate.py:
    membrane mean/sd on text, a random read scaled so the integrator has sd ~0.15, the bias giving ~1/300 crossings).
    The eye channel's bias is set so that the eyes start open (e = eye_open at the integrator's steady state)."""
    device = model.read_indices.device
    model.register_buffer('ai_mu', init['act_mu'].to(device).float(), persistent=True)
    model.register_buffer('ai_sigma', init['act_sigma'].to(device).float(), persistent=True)
    scale = float(init['act_w'].std()) / AI_RAW_SD
    model.register_buffer('ai_w_scale', torch.tensor(scale, device=device), persistent=True)
    model.ai_w = torch.nn.Parameter((init['act_w'] / scale).to(device).float().clone())
    lam = math.exp(-1 / 8)
    b = init['act_b'].to(device).float().clone()
    b[0] = (1 - lam) * math.log(eye_open / (1 - eye_open))
    model.ai_b = torch.nn.Parameter(b)
    model.ai_logit_leak = torch.nn.Parameter(torch.full((2,), math.log(lam / (1 - lam)), device=device))
    return model


def attach_motor_acts(model, tau_muscle=8.0, eye_open=0.9, habit=0.2, brain_gradient=False, dopamine_precision=False):
    """Acts as the brain's own output cells (diversity first; GOAL section 'Behaviour diversity first').  The output
    cells' transmitted spikes drive a muscle activation m = lambda m + (1 - lambda) s (the body's low-pass, tau_muscle
    ticks, no threshold, no learning); the act logits are z = W m + b (W learned: which cells drive which act; starts
    at zero, so the acts start at their habits); the acts are SAMPLED, look ~ Bernoulli(sigmoid(z_eye)) and commit ~
    Bernoulli(sigmoid(z_out)) -- no integrator, no bound, no straight-through.  The commit's learning signal is the
    exact expectation over commit times along the one trajectory (each tick: 'if I answered now', as CTM scores every
    tick).  brain_gradient=False: the policy reads the muscles detached (the readout learns first).  ai_g carries m.
    dopamine_precision: the policy precision gamma of active inference, q(pi) ~ E exp(-gamma G), is encoded by the
    brain's own dopamine cells (Friston et al. 2014): z = ln E + gamma (W m + b), ln E the habit (frozen, not
    scaled), W m + b the learned value; gamma = (d + eps) / (d_ref + eps), d the mean transmitted spike of the 392
    dopamine cells low-passed with the brain's own slow dopamine clock (sigmoid(dopamine_logit_slow), ~556 ticks),
    d_ref its calibrated mean on text.  The population tone, not the dopamine delivered at the output cells: those
    receive 0.5% of it.  gamma is read, never trained (detached: it is a gain redundant with |W|; a gradient into the
    dopamine cells would only learn to burst into determinism and open the plasticity windows brain-wide).  No new
    cell, no new parameter; ai_g carries [m, d]."""
    device = model.read_indices.device
    n = model.read_indices.numel()
    model.ai_mode = 'motor'
    model.ai_brain_gradient = bool(brain_gradient)
    model.ai_precision = bool(dopamine_precision)
    if model.ai_precision and (getattr(model, 'dan_indices', None) is None
                               or getattr(model, 'dopamine_logit_slow', None) is None):
        raise ValueError('dopamine-encoded precision needs the dopamine cells and their slow clock (--dopamine-modulation)')
    model.register_buffer('ai_lambda', torch.tensor(math.exp(-1.0 / tau_muscle), device=device), persistent=True)
    model.ai_w = torch.nn.Parameter(torch.zeros(2, n, device=device))
    prior = torch.tensor([math.log(eye_open / (1 - eye_open)), math.log(habit / (1 - habit))], device=device)
    if model.ai_precision:
        model.register_buffer('ai_habit', prior, persistent=True)        # ln E, frozen
        model.ai_b = torch.nn.Parameter(torch.zeros(2, device=device))  # the value's bias, scaled with W m
    else:
        model.ai_b = torch.nn.Parameter(prior)
    return model


def motor_logits(model, m):
    precision = getattr(model, 'ai_precision', False)
    if precision:
        m, d = m[:, :-1], m[:, -1:]
    if getattr(model, 'ai_mu', None) is not None:
        m = (m - model.ai_mu) / model.ai_sigma                # each muscle's own gain (calibrate_muscles)
    value = m @ model.ai_w.T + model.ai_b
    if not precision:
        return value
    return model.ai_habit + gamma_of(model, d).detach() * value


DA_EPS = 1.0 / (392 * 1000)     # one dopamine spike per 1000 ticks over the population: a silent tone reads gamma > 0


def gamma_of(model, d):
    """The precision encoded by the dopamine tone d (the population's slow mean spike)."""
    return (d + DA_EPS) / (model.ai_da_ref + DA_EPS)


def motor_rest(model, like):
    """ai_g at rest: muscles 0; with dopamine precision, the tone at its mean (gamma = 1)."""
    g = like.new_zeros(1, model.read_indices.numel())
    if getattr(model, 'ai_precision', False):
        g = torch.cat([g, model.ai_da_ref.detach().reshape(1, 1).to(g)], 1)
    return g


def precision_of(model, g):
    """gamma from ai_g (None without dopamine precision)."""
    if not getattr(model, 'ai_precision', False) or g is None:
        return None
    return float(gamma_of(model, g[0, -1]))


def dopamine_rate(model, state):
    """Mean transmitted spike of the dopamine cells at this tick."""
    return state.ring[0].index_select(1, model.dan_indices).mean(1, keepdim=True)


def set_motor_constants(model, saved):
    """The calibrated constants (frozen) from a saved state dict."""
    device = model.read_indices.device
    for name in MOTOR_CONSTANTS:
        if name in saved:
            setattr(model, name, torch.nn.Parameter(saved[name].to(device).float(), requires_grad=False))


def calibrate_muscles(model, state, tokens, ticks=8):
    """Each muscle's resting mean and spread on text (fixed weights, no gradient): the readout sees standardized
    muscle activations, as the language read sees normalized membranes.  Stored as frozen parameters (saved with the
    model).  The spread is floored at its median (silent cells must not blow up)."""
    import importlib
    fl = importlib.import_module('information_boltzmann.core.fly_bptt_learning')
    opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(),
                conductance_gains=model.get_conductance_gains(), alif_params=model.get_alif_params(),
                stp_params=model.get_stp_params())
    writer = model.topographic_writer
    quiet = torch.zeros_like(state.h)
    m = state.h.new_zeros(1, model.read_indices.numel())
    rows, da, s = [], [], state
    precision = getattr(model, 'ai_precision', False)
    with torch.no_grad():
        for w in tokens:
            token = torch.tensor([int(w)], device=state.h.device)
            for t in range(ticks):
                if t == 0:
                    drive, baseline = fl.write_input(model, s, token)
                    s = fl.step_fly_physical_tick(model, s, token, drive, baseline, opts, base_rates=opts['base_rates'])
                else:
                    s = fl.step_fly_physical_tick(model, s, None, quiet, writer.lambda_adapt * s.baseline, opts,
                                                  base_rates=opts['base_rates'])
                m = model.ai_lambda * m + (1.0 - model.ai_lambda) * s.ring[0].index_select(1, model.read_indices)
                rows.append(m)
                if precision:
                    da.append(dopamine_rate(model, s))
        M = torch.cat(rows, 0)
        mu, sd = M.mean(0, keepdim=True), M.std(0, keepdim=True)
        active = sd[sd > 0]
        sd = torch.maximum(sd, active.median() if active.numel() else sd.new_tensor(1.0))   # silent cells read 0
    model.ai_mu = torch.nn.Parameter(mu.clone(), requires_grad=False)
    model.ai_sigma = torch.nn.Parameter(sd.clone(), requires_grad=False)
    if precision:
        ref = torch.cat(da).mean()
        if float(ref) < 10 * DA_EPS:
            raise ValueError(f'the dopamine cells are nearly silent on text (tone {float(ref):.2e}): no precision to read')
        model.ai_da_ref = torch.nn.Parameter(ref.reshape(()).clone(), requires_grad=False)
    return float(active.median()) if active.numel() else 0.0


def ai_eye(model, g, forced=None):
    """The eye gain for the coming tick, from the integrator (or the muscles) as it stands (before the input is
    written)."""
    if forced is not None:
        return g.new_full((1,), float(forced))
    if getattr(model, 'ai_mode', None) == 'motor':
        return torch.sigmoid(motor_logits(model, g)[0, 0:1])
    return torch.sigmoid(g[0, 0:1])


def ai_step(model, state, g, taped=None):
    """After a tick.  Motor mode: the muscles integrate the output cells' transmitted spikes; the commit hazard is
    h = sigmoid(z_out) (returned soft: the objective takes the exact expectation over commit times), the realized
    commit is a sample (taped for the recompute).  Integrator mode (M0-M3): both channels integrate the output cells'
    membranes; the output commits at the bound (straight-through) and resets.
    Returns (output hazard [1], new g, committed: bool)."""
    if getattr(model, 'ai_mode', None) == 'motor':
        s = state.ring[0].index_select(1, model.read_indices)
        if not model.ai_brain_gradient:
            s = s.detach()
        if getattr(model, 'ai_precision', False):
            # the dopamine tone on the brain's own slow dopamine clock
            da = dopamine_rate(model, state).detach()             # read, never trained (attach_motor_acts)
            rho = torch.sigmoid(model.dopamine_logit_slow).detach()
            g = torch.cat([model.ai_lambda * g[:, :-1] + (1.0 - model.ai_lambda) * s,
                           rho * g[:, -1:] + (1.0 - rho) * da], 1)
        else:
            g = model.ai_lambda * g + (1.0 - model.ai_lambda) * s
        hazard = torch.sigmoid(motor_logits(model, g)[0, 1:2])
        committed = bool(torch.rand(()) < float(hazard.detach())) if taped is None else bool(taped)
        return hazard, g, committed
    h = state.h
    x = (h.index_select(1, model.read_indices) - model.ai_mu) / model.ai_sigma
    g = torch.sigmoid(model.ai_logit_leak) * g + x @ (model.ai_w * model.ai_w_scale).T + model.ai_b
    committed = (float(g[0, 1].detach()) >= BOUND) if taped is None else bool(taped)
    # straight-through with the slope the soft crossing has AT the bound, everywhere: a sigmoid's slope vanishes far
    # from the bound, which makes any habit absorbing (an integrator that crosses every tick could never learn to wait)
    u = (g[0, 1:2] - BOUND) / (4.0 * SOFT)
    hazard = float(committed) + u - u.detach()
    if committed:
        g = torch.cat([g[:, :1], g[:, 1:] * 0.0], 1)
    return hazard, g, committed


class Familiarity:
    """Count-based novelty of what was actually seen: novelty(w) = 1 / sqrt(1 + n(w)), n(w) += eye gain at each onset
    of w.  The adult core already encodes long-term familiarity (word rarity); this is its within-life counterpart."""

    def __init__(self, vocab=50257):
        self.n = np.zeros(vocab, np.float64)

    def novelty(self, tokens):
        return 1.0 / np.sqrt(1.0 + self.n[np.asarray(tokens)])

    def see(self, tokens, eyes):
        np.add.at(self.n, np.asarray(tokens), np.asarray(eyes, dtype=np.float64))




def eye_act(model, g, forced=None, taped=None, sample=False):
    """The eyes as a decision (M3): look at the coming word or not.  Forward exactly 0 / 1 (closed eyes: the word is not
    written at all, so a question kept open cannot be answered by peeking); backward through p = sigmoid(G_eye),
    straight through, like the output commit.  sample=True: the look is drawn from Bernoulli(p) (active inference
    samples its policies; it also gives real closures with their real outcomes); else the look is p >= 0.5.
    Returns (gain [1], decision: bool)."""
    if forced is not None:
        return g.new_full((1,), float(forced)), bool(forced >= 0.5)
    soft = torch.sigmoid(motor_logits(model, g)[0, 0:1]) if getattr(model, 'ai_mode', None) == 'motor' \
        else torch.sigmoid(g[0, 0:1])
    if taped is not None:
        look = bool(taped)
    elif sample:
        look = bool(torch.rand(()) < float(soft.detach()))
    else:
        look = float(soft.detach()) >= 0.5
    u = 0.25 * g[0, 0:1]                                  # the slope of p at p = 1/2, everywhere (see ai_step)
    return float(look) + u - u.detach(), look


def pragmatic_value(hz, eyes, writes, logp, intervals, answers, asks, miss_value, depth=0):
    """Pragmatic value of output (EFE).  The main speaker asks a question with every word it says ("what comes
    next?"); background words (the noisy TV) ask nothing and answer nothing.  Question i (asks[i]) opens with word i;
    its answer is the main speaker's next word (answers[i], a token id).  It closes the first time the brain RECEIVES
    the next main word -- a tick that writes that word with the eyes open (pulse input: its onset; sustained input:
    any of its ticks) -- or once `depth` further main words have passed; while the eyes are closed it stays open and
    the brain keeps thinking.  Within a tick the eye acts before the write and the commit after the read, so a question
    closed at a tick cannot be answered at that tick.  A commit answers the oldest open question, valued -CE of the read
    then against that question's answer (logp [T, V] of every tick's read); a question closing unanswered is worth
    miss_value (-log V, or per question: minus the listener's prior CE of its answer -- an answer is worth the
    information it adds beyond what the listener already expects).  With soft hazards (motor acts) this is the exact
    expectation over commit times; with 0/1 straight-through hazards, the value along the realized path.

    hz [T] output hazards; eyes [T] eye gain at every tick; writes [T] (bool) the tick writes its word; intervals: the
    n interval lengths.  With pulse input and depth 0 (or the eyes always open) this is the per-word value
    sum_i [sum_t S h (-CE_t) + S_K miss].
    Returns (J summed over questions, answer latency in ticks per question (-1: missed; None: not a question), number
    of main words passed while the question stayed open and unanswered)."""
    starts, total = [], 0
    for k in intervals:
        starts.append(total); total += k
    n = len(intervals)
    misses = [float(x) for x in miss_value] if hasattr(miss_value, '__len__') else [float(miss_value)] * n
    w = torch.as_tensor(writes, dtype=hz.dtype, device=hz.device)
    told = eyes * w                                            # the tick delivers the word to the brain
    busy = [hz.new_zeros(k) for k in intervals]                # P(the commit at this tick was taken by an older question)
    J = hz.new_zeros(())
    first, extended = [], []
    for i in range(n):
        if not asks[i]:
            first.append(None); extended.append(0)
            continue
        alive, S = hz.new_ones(()), hz.new_ones(())            # P(still open), P(not yet answered)
        answer_tick, ext, elapsed, passed = -1, 0, 0, 0
        miss_value = misses[i]
        for j in range(i, n):
            sl = slice(starts[j], starts[j] + intervals[j])
            if j > i and asks[j]:                              # a main word: hearing it closes the question
                passed += 1
                if passed > depth:
                    J = J + alive * S * miss_value             # kept open as long as allowed
                    break
                if float(alive.detach()) > 0.5 and float(S.detach()) > 0.5 and float(told[starts[j]].detach()) < 0.5:
                    ext += 1
                keep = torch.cumprod(1.0 - told[sl], 0)
                a = alive * keep                               # open after the eye at each tick
                a_before = torch.cat([alive.reshape(1), a[:-1]])
            else:
                a = alive.expand(intervals[j]); a_before = a
            h = hz[sl] * (1.0 - busy[j])
            S_before = S * torch.cumprod(torch.cat([h.new_ones(1), 1.0 - h[:-1]]), 0)
            J = J + ((a_before - a) * S_before).sum() * miss_value   # told before answering: missed
            take = a * S_before * h
            J = J + (take * logp[sl, int(answers[i])]).sum()  # value -CE of the read against the answer
            busy[j] = busy[j] + take
            if answer_tick < 0:
                hits = torch.nonzero(take.detach() > 0.5)
                if len(hits):
                    answer_tick = elapsed + int(hits[0]) + 1
            S = S * torch.prod(1.0 - h)
            alive = a[-1]
            elapsed += intervals[j]
        else:
            J = J + alive * S * miss_value                     # window end: closes
        first.append(answer_tick); extended.append(ext)
    return J, first, extended


def habit_kl(h, habit):
    """KL(Bernoulli(h) || Bernoulli(habit)) per decision: the policy prior of active inference (q(pi) ~ E exp(-gamma G)
    minimises E_q[gamma G] + KL(q || E)); keeps the acts sampled instead of collapsing to 0 / 1."""
    h = h.clamp(1e-6, 1 - 1e-6)
    return h * torch.log(h / habit) + (1 - h) * torch.log((1 - h) / (1 - habit))
