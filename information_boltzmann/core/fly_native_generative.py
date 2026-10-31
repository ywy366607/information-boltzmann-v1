"""Finite native-brain hypotheses for prospective categorical consequences.

These are two prequential experts, not a posterior reconstructed from optimizer
moments. The current organism and an explicitly retained ancestor share the
current sensory ports, decoder, physical starting state and event clock. Their
disagreement measures information about this finite hypothesis variable only.
"""
import copy
import math
import random
from contextlib import contextmanager

import numpy as np
import torch

from .fly_native_policy_credit import native_parameter_names


FORMAT = 'native-categorical-world-v1'


def _clone_state(state):
    return type(state)(**{
        key: tuple(t.detach().clone() for t in value) if isinstance(value, tuple)
        else value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in state.state_dict().items()
    })


@contextmanager
def _preserve_rollout(model):
    """Counterfactual ticks must not count as experience or consume randomness."""
    rng = (torch.get_rng_state(), random.getstate(), np.random.get_state(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
    missing = object()
    names = ('homeo_spikes', 'homeo_ticks', 'step_grad_clips', 'external_drive',
             '_superclass_onehot', '_comp_rate_probe')
    saved = {name: (getattr(model, name, missing),
                    copy.deepcopy(getattr(model, name, missing))) for name in names}
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, training in modes:
            module.training = training
        with torch.no_grad():
            for name, (original, value) in saved.items():
                if original is missing:
                    if hasattr(model, name):
                        delattr(model, name)
                elif torch.is_tensor(original):
                    original.copy_(value)
                    setattr(model, name, original)
                else:
                    setattr(model, name, value)
        torch.set_rng_state(rng[0])
        random.setstate(rng[1])
        np.random.set_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])


class NativeCategoricalWorld:
    """Two native physical predictors, updated only by actual external words.

    ``reference_parameters`` must be explicit ancestral native tensors. Passing
    current tensors is useful as a zero-disagreement control; it creates no
    epistemic drive. No new neural cells or arbitrary perturbations are added.
    The action forecast is a local next-external-word approximation. It does
    not model whether/when Mama replies or assign preferences to her replies.
    """

    def __init__(self, learner, reference_parameters=None, provenance=None):
        self.learner = learner
        model = learner.model
        self.names = native_parameter_names(model)
        if not self.names:
            raise ValueError('Native hypothesis needs existing trainable native parameters')
        parameters = dict(model.named_parameters())
        state = getattr(learner, 'native_generative_state', None)
        if state is None:
            if reference_parameters is None:
                raise ValueError('An explicit ancestral native reference is required')
            if set(reference_parameters) != set(self.names):
                raise ValueError('Native reference parameter names differ from the organism')
            reference = {}
            for name in self.names:
                value = reference_parameters[name]
                if value.shape != parameters[name].shape or not torch.isfinite(value).all():
                    raise ValueError(f'Invalid ancestral native parameter: {name}')
                reference[name] = value.detach().cpu().clone()
            state = {'format': FORMAT, 'reference_parameters': reference,
                     'provenance': copy.deepcopy(provenance),
                     'hypothesis_assumption': 'adaptive-current versus retained-native-ancestor; shared-current-decoder',
                     'log_weights': torch.full((2,), -math.log(2), dtype=torch.float64),
                     'external_updates': 0, 'self_excluded': 0}
            learner.native_generative_state = state
        if state.get('format') != FORMAT:
            raise ValueError('Unknown native generative state schema')
        if set(state['reference_parameters']) != set(self.names):
            raise ValueError('Saved native reference differs from current native parameter names')
        for name in self.names:
            if state['reference_parameters'][name].shape != parameters[name].shape:
                raise ValueError(f'Saved native reference shape differs: {name}')
        self.state = state

    @property
    def weights(self):
        return torch.softmax(self.state['log_weights'].detach().double().cpu(), 0)

    def _distribution(self, log_probs):
        values = torch.as_tensor(log_probs).detach().double()
        if values.ndim != 2 or values.shape[0] != 2 or values.shape[1] < 1:
            raise ValueError('Native forecasts must have shape [2, vocabulary]')
        if torch.isnan(values).any() or torch.isposinf(values).any():
            raise ValueError('Native forecasts contain invalid log probabilities')
        if not torch.allclose(torch.logsumexp(values, -1), torch.zeros(2, device=values.device,
                              dtype=values.dtype), atol=1e-5, rtol=0):
            raise ValueError('Native forecasts must be normalized log probabilities')
        return values

    def information(self, log_probs):
        """Exact categorical I(hypothesis; next external word), in nats."""
        lp = self._distribution(log_probs)
        weights = self.weights.to(lp.device)
        log_weights = self.state['log_weights'].to(lp)
        log_weights = log_weights - torch.logsumexp(log_weights, 0)
        log_mix = torch.logsumexp(lp + log_weights[:, None], 0)
        ratio = torch.where(torch.isfinite(lp), lp - log_mix, torch.zeros_like(lp))
        value = (weights[:, None] * lp.exp() * ratio).sum().clamp_min(0)
        return value

    def update(self, log_probs, target, source):
        """Score before update; a known self command supplies no external evidence."""
        if source == 'self_hearing':
            self.state['self_excluded'] += 1
            return {'updated': False, 'source': source, 'realized_information': 0.0}
        if not (str(source).startswith('mother_') or source == 'external'):
            return {'updated': False, 'source': source, 'realized_information': 0.0}
        lp = self._distribution(log_probs).cpu()
        target = int(target)
        if not 0 <= target < lp.shape[1]:
            raise ValueError('External target lies outside the vocabulary')
        old_log = self.state['log_weights'].double().cpu()
        old_log = old_log - torch.logsumexp(old_log, 0)
        old = old_log.exp()
        evidence = lp[:, target]
        log_joint = old_log + evidence
        logp = torch.logsumexp(log_joint, 0)
        if not torch.isfinite(logp):
            raise ValueError('All native hypotheses assign zero probability to the actual word')
        new_log = log_joint - logp
        new = new_log.exp()
        kl = torch.where(new > 0, new * (new_log - old_log), torch.zeros_like(new)).sum()
        self.state['log_weights'] = new_log
        self.state['external_updates'] += 1
        return {'updated': True, 'source': source, 'prequential_logp': float(logp),
                'realized_information': float(kl), 'weights_before': old.tolist(),
                'weights_after': new.tolist()}

    def predict(self, state, token, ticks, *, sensory_on=True, eye_gain=1.0,
                voice=False, reward=False, quiet_ranges=(), eye_tape=None):
        """Matched serial native rollouts; no actual state, parameter or RNG change."""
        from . import fly_bptt_learning as fl
        from .fly_active_inference import motor_rest
        from .fly_mother_world import sensory_segments
        if isinstance(ticks, bool) or not isinstance(ticks, int) or ticks < 1:
            raise ValueError('Forecast duration must be a positive physical tick count')
        if eye_tape is not None and len(eye_tape) != ticks:
            raise ValueError('Eye evidence must match the actual physical duration')
        model = self.learner.model
        parameters = dict(model.named_parameters())
        actual = {name: parameters[name].detach().clone() for name in self.names}
        segment_ticks = int(getattr(self.learner, 'stream_segment_ticks', 16) or 16)
        outputs = []
        with _preserve_rollout(model), torch.no_grad():
            rng = torch.get_rng_state()
            python_rng, numpy_rng = random.getstate(), np.random.get_state()
            cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
            external_drive = getattr(model, 'external_drive', None)
            external_copy = (external_drive.detach().clone() if torch.is_tensor(external_drive)
                             else copy.deepcopy(external_drive))
            try:
                for hypothesis in range(2):
                    torch.set_rng_state(rng)
                    random.setstate(python_rng)
                    np.random.set_state(numpy_rng)
                    if cuda_rng is not None:
                        torch.cuda.set_rng_state_all(cuda_rng)
                    model.external_drive = (external_copy.clone() if torch.is_tensor(external_copy)
                                            else copy.deepcopy(external_copy))
                    values = actual if hypothesis == 0 else self.state['reference_parameters']
                    for name in self.names:
                        parameters[name].copy_(values[name].to(parameters[name]))
                    current = _clone_state(state)
                    options = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds(),
                                   conductance_gains=model.get_conductance_gains(),
                                   alif_params=model.get_alif_params(), stp_params=model.get_stp_params())
                    if getattr(model, 'dopamine_sensitivity', None) is not None:
                        options['dopamine_s_class'] = fl.dopamine_class_sensitivity(model, current.h)
                    command = torch.tensor([int(token)], device=current.h.device)
                    # Frozen realized sensing is a physical intervention, not a
                    # claim that the eye gain is inverse observation variance.
                    ai_g = motor_rest(model, current.h)
                    for start, length, on in sensory_segments(ticks, segment_ticks, quiet_ranges):
                        forced = (eye_tape[start:start + length] if eye_tape is not None else eye_gain)
                        result = fl.advance_fly_token_stream(
                            model, current, command, [length], ticks=length,
                            sensory_on=bool(sensory_on and on), hold_input=getattr(self.learner, 'hold_input', True),
                            writer_baseline_clock=getattr(self.learner, 'writer_baseline_clock', 'physical'),
                            fresh=start == 0, voice=bool(voice),
                            reward_ticks=fl.reward_ticks_left(model, bool(reward), start),
                            ai_g=ai_g, ai_tape=[False] * length, eye_forced=forced, **options)
                        current, latent, ai_g = result[0], result[1], result[-1]
                    outputs.append(torch.log_softmax(model.decoder(model.read_norm(latent)).float(), -1)[0])
            finally:
                for name in self.names:
                    parameters[name].copy_(actual[name])
        return torch.stack(outputs)

    def command_values(self, state, candidate_ids, ticks, eye_gain=1.0):
        """Compare next-external-word uncertainty after quiet or a known word.

        The command is held on the real sensory surface; its identity is known
        and is never used as the external target for epistemic credit.
        """
        ids = [int(value) for value in candidate_ids]
        wait_lp = self.predict(state, 0, ticks, sensory_on=False, eye_gain=eye_gain)
        word_lp = [self.predict(state, word, ticks, eye_gain=eye_gain) for word in ids]
        return {'wait': self.information(wait_lp),
                'words': torch.stack([self.information(lp) for lp in word_lp]) if word_lp
                else torch.empty(0, device=wait_lp.device),
                'candidate_ids': ids, 'log_probs_wait': wait_lp,
                'log_probs_words': torch.stack(word_lp) if word_lp else wait_lp.new_empty((0, *wait_lp.shape))}
