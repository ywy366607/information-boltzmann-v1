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
