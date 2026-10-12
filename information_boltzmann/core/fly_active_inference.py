"""Unified perception and action (GOAL: docs/information_boltzmann/FLY_ACTIVE_INFERENCE_GOAL_20261013.md).

One act head, two acts, read from the membranes of all output cells (the cells the language read uses; the model
chooses which cells act):
  channel 0  EYES   -- the precision (gain) of the text input, e = sigmoid(G_eye), decided from the previous tick's
                       integrator before the input is written.  Closed eyes: less interference, but a word arriving
                       meanwhile is written with gain e ~ 0 -- it is missed (the world never waits; the word is still
                       scored by the language objective).
  channel 1  OUTPUT -- commit the expectation standing on the output surface: a leaky integrator racing to a bound
                       (forward: the realized crossing, taped for checkpoint recompute; backward: a sigmoid,
                       straight through); reset after a commit.
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
AI_STATE_NAMES = AI_PARAMETER_NAMES + ('ai_mu', 'ai_sigma', 'ai_w_scale')
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


def ai_eye(model, g, forced=None):
    """The eye gain for the coming tick, from the integrator as it stands (before the input is written)."""
    if forced is not None:
        return g.new_full((1,), float(forced))
    return torch.sigmoid(g[0, 0:1])


def ai_step(model, h, g, taped=None):
    """After a tick: integrate both channels from the output cells' membranes; the output channel commits when it
    reaches the bound (forward: the realized crossing or the taped one; backward: straight through) and resets.
    Returns (output hazard [1], new g [1, 2], committed: bool)."""
    x = (h.index_select(1, model.read_indices) - model.ai_mu) / model.ai_sigma
    g = torch.sigmoid(model.ai_logit_leak) * g + x @ (model.ai_w * model.ai_w_scale).T + model.ai_b
    committed = (float(g[0, 1].detach()) >= BOUND) if taped is None else bool(taped)
    soft = torch.sigmoid((g[0, 1:2] - BOUND) / SOFT)
    hazard = float(committed) + soft - soft.detach()
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
    soft = torch.sigmoid(g[0, 0:1])
    if taped is not None:
        look = bool(taped)
    elif sample:
        look = bool(torch.rand(()) < float(soft.detach()))
    else:
        look = float(soft.detach()) >= 0.5
    return float(look) + soft - soft.detach(), look


def pragmatic_value(hz, eyes, logp, intervals, answers, asks, main_onset, miss_value, depth=0):
    """Pragmatic value of output (EFE).  The main speaker asks a question with every word it says ("what comes
    next?"); background words (the noisy TV) ask nothing and answer nothing.  Question i (asks[i]) opens at onset i; its
    answer is the main speaker's next word (answers[i], a token id).  It closes at the first MAIN onset where the eyes
    are open (the brain is told), or once `depth` + 1 main onsets have passed; while the eyes are closed it stays open
    and the brain keeps thinking.  A commit answers the oldest open question, valued -CE of the read then against that
    question's answer (logp [T, V] of every tick's read); a question closing unanswered is worth miss_value (-log V).
    Expected value along the realized path, differentiable in the hazards and the eyes (straight-through).

    hz [T] output hazards; eyes [n] eye gain at each word's onset; intervals: n interval lengths; main_onset[j]: onset j
    is a main word (j = 1..n-1; the onset after the window is treated as a closing main onset).  Without background
    words and with depth 0 (or the eyes always open) this is sum_i [sum_t S h (-CE_t) + S_K miss].
    Returns (J summed over questions, answer latency in ticks per question (-1: missed; None: not a question), number
    of main onsets passed with the eyes closed while the question was open and unanswered)."""
    starts, total = [], 0
    for k in intervals:
        starts.append(total); total += k
    n = len(intervals)
    busy = [hz.new_zeros(k) for k in intervals]              # P(the commit at this tick was taken by an older question)
    J = hz.new_zeros(())
    first, extended = [], []
    for i in range(n):
        if not asks[i]:
            first.append(None); extended.append(0)
            continue
        alive, S = hz.new_ones(()), hz.new_ones(())           # P(still open), P(not yet answered)
        answer_tick, ext, elapsed, passed = -1, 0, 0, 0
        for j in range(i, n):
            sl = slice(starts[j], starts[j] + intervals[j])
            h = hz[sl] * (1.0 - busy[j])
            surv = S * torch.cumprod(torch.cat([h.new_ones(1), 1.0 - h[:-1]]), 0)
            take = alive * surv * h
            J = J + (take * logp[sl, int(answers[i])]).sum()      # value -CE of the read against the answer
            busy[j] = busy[j] + take
            if answer_tick < 0 and float(alive.detach()) > 0.5:
                hits = torch.nonzero(take.detach() > 0.5)
                if len(hits):
                    answer_tick = elapsed + int(hits[0]) + 1
            S = S * torch.prod(1.0 - h)
            elapsed += intervals[j]
            if j + 1 >= n:
                J = J + alive * S * miss_value                # window end: closes
                break
            if not main_onset[j + 1]:
                continue                                      # a background word: the question stays open
            e = eyes[j + 1]
            J = J + alive * S * e * miss_value                # the brain looks at the main speaker: closes
            if float(alive.detach()) > 0.5 and float(S.detach()) > 0.5 and float(e.detach()) < 0.5:
                ext += 1
            alive = alive * (1.0 - e)
            passed += 1
            if passed > depth:
                J = J + alive * S * miss_value                # kept open as long as allowed
                break
        first.append(answer_tick); extended.append(ext)
    return J, first, extended
