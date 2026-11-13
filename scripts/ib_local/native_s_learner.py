"""Optional real-learner adapter for APPROXIMATE local native-S eligibility.

Original learner/backend remain unchanged. This does not implement full joint
24-layer/record eligibility. Raw credit is not gated by oracle error/cosine.
"""
from contextlib import contextmanager
import math

import numpy as np
import torch
from torch.nn import functional as F

from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner, language_loss
from scripts.ib_local.irreversible_credit_stream import PretrainedStreamBackend
from scripts.ib_local.native_s_eligibility import NativeSEligibility
from scripts.ib_local.bounded_causal_tag_blocked import workspace_budget


def mismatch_credit(raw):
    """Fixed within-layer/name-family permutation, preserving family norms."""
    return {name: np.roll(value, 1, axis=-1).copy() if value.size > 1 else value.copy()
            for name, value in raw.items()}


def select_native_credit(raw, mode, strict_old=None):
    """Change ONLY known-old additions; unknown/recent/rounding stay common."""
    if mode == 'raw':
        return raw
    if mode == 'cut':
        return {}
    if mode == 'mismatch':
        return mismatch_credit(raw)
    if mode not in ('cut_old', 'mismatch_old') or strict_old is None or set(strict_old) != set(raw):
        raise ValueError('strict old-source controls require age-resolved matching owners')
    if any(value.shape != strict_old[name].shape or not np.isfinite(strict_old[name]).all()
           for name, value in raw.items()):
        raise ValueError('invalid old-source shape/value')
    wrong = mismatch_credit(strict_old) if mode == 'mismatch_old' else None
    return {name: (value - strict_old[name] + (wrong[name] if wrong is not None else 0)).astype(np.float32)
            for name, value in raw.items()}


def apply_master_credit(learner, extra=None):
    """Same one Adam step; historical addition occurs in FP32 before clipping."""
    extra = {} if extra is None else extra
    names = {name for name, _ in learner.native}
    if set(extra) - names:
        raise ValueError('historical credit names do not belong to native masters')
    norm2 = 0.
    for master, (name, param) in zip(learner.masters, learner.native):
        if param.grad is not None and not bool(torch.isfinite(param.grad).all()):
            raise FloatingPointError('nonfinite native gradient')
        master.grad = None if param.grad is None else param.grad.detach().float().clone()
        if name in extra:
            value = np.asarray(extra[name], dtype=np.float32)
            if value.shape != tuple(master.shape) or not np.isfinite(value).all():
                raise ValueError('invalid historical credit shape/value')
            norm2 += float(np.sum(value.astype(np.float64) ** 2))
            # Preserve Adam's None semantics when the empty trace adds nothing.
            if np.count_nonzero(value):
                addition = torch.from_numpy(value.copy()).to(master.device)
                if master.grad is None:
                    master.grad = addition
                else:
                    master.grad.add_(addition)
    before_clip = torch.nn.utils.clip_grad_norm_(learner.masters, 1.)
    for param in learner.extras + learner.null_parameters:
        if param.grad is not None and not bool(torch.isfinite(param.grad).all()):
            raise FloatingPointError('nonfinite recall gradient')
    learner.optimizer.step()
    with torch.no_grad():
        for master, (_, param) in zip(learner.masters, learner.native):
            param.copy_(master.to(param.dtype))
    return dict(injected_gradient_norm=math.sqrt(norm2), combined_preclip_norm=float(before_clip),
                optimizer_steps=1)


class NativeSEligibilityLearner(PretrainedExperienceLearner):
    """Opt-in adapter; missing legacy sidecar never creates fictional history."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eligibility = None
        self.native_s_mode = None
        self._native_actual_count = 0
        self._native_capture_started = False
        self._native_window_version = None
        self._native_extra = None
        self._last_native_s = self._empty_report()

    @staticmethod
    def _empty_report():
        return dict(scope='local_native_s_leg_v1', has_factual_trace=False,
                    teaching_available=False, trace_empty_before=True,
                    old_credit_norm=0., teaching_norm=0., injected_gradient_norm=0.,
                    persistent_bytes=0, estimated_temporary_bytes=0)

    def enable_native_s_eligibility(self, mode='raw', rank=2, max_bytes=64*2**20, block_size=65536,
                                    source_ages=False):
        if self.eligibility is not None:
            raise RuntimeError('eligibility already initialized; restore rather than reset')
        if mode not in ('raw', 'cut', 'mismatch', 'cut_old', 'mismatch_old'):
            raise ValueError('unknown eligibility arm')
        if mode in ('cut_old', 'mismatch_old') and not source_ages:
            raise ValueError('old-only control requires explicit source ages')
        specs = {}
        for layer in self.layers:
            attn = self.model.model.layers[layer].attn
            if not attn.use_short_conv or attn.num_v_heads != attn.num_heads:
                raise ValueError('this candidate requires the actual equal-head short-convolution configuration')
            specs[layer] = dict(heads=attn.num_heads, key_dim=attn.head_k_dim,
                                value_dim=attn.head_v_dim, width=attn.a_proj.in_features)
        producer = NativeSEligibility
        if source_ages:
            from scripts.ib_local.age_resolved_native_s import AgeResolvedNativeSEligibility
            producer = AgeResolvedNativeSEligibility
        self.eligibility = producer(specs, rank=rank, block_size=block_size, max_bytes=max_bytes)
        native = dict(self.native)
        for name, shape in self.eligibility.parameter_shapes.items():
            if name not in native or tuple(native[name].shape) != tuple(shape):
                raise ValueError(f'gate owner mapping mismatch: {name}')
        self.native_s_mode = mode

    def _temporary_bytes(self):
        sidecar = self.eligibility
        if sidecar is None:
            return 0
        return sidecar.estimated_temporary_bytes

    def _check_budget(self):
        total = self.eligibility.nbytes + self._temporary_bytes()
        if total > self.eligibility.max_bytes:
            raise MemoryError('explicit local-S eligibility state/workspace exceeds its budget')
        return total

    def _initial_states(self, cache):
        states = {}
        for layer, spec in self.eligibility.specs.items():
            state = cache.states[layer].get('recurrent_state') if len(cache.states) > layer else None
            if state is None:
                states[layer] = np.zeros((1, spec['heads'], spec['key_dim'], spec['value_dim']), np.float32)
            else:
                states[layer] = state.detach().float().cpu().numpy().copy()
        return states

    @contextmanager
    def factual_capture(self, clock_offset):
        sidecar = self.eligibility
        if sidecar is None:
            yield
            return
        if clock_offset == 0:
            sidecar.open_window(self.cache._seen_tokens, self.updates, self._initial_states(self.cache))
            self._native_actual_count = 0
            self._native_window_version = self.updates
            self._native_capture_started = True
        elif sidecar.pending is None and not self._native_capture_started:
            # An inherited partial window has no captured factual participants.
            yield
            return
        if sidecar.pending is None or clock_offset != self._native_actual_count:
            raise RuntimeError('actual native capture/window position diverged')
        if self.updates != self._native_window_version:
            raise RuntimeError('parameters changed inside factual capture window')
        handles, packets = [], {}
        def active(layer):
            return not self.memories[layer]._replaying
        for layer in self.layers:
            attn = self.model.model.layers[layer].attn
            def before(module, args, kw, layer=layer):
                if not active(layer):
                    return
                h = args[0] if args else kw['hidden_states']
                if h.shape[0] != 1 or h.shape[1] + clock_offset > 128:
                    raise ValueError('bounded single-individual factual capture only')
                mode = 'fused_recurrent' if h.shape[1] <= 64 else module.mode
                if mode not in ('fused_recurrent', 'chunk'):
                    raise ValueError('unregistered native recurrence branch')
                packets[layer] = dict(branch='recurrent' if mode == 'fused_recurrent' else 'chunk')
            def conv(module, args, kw, output, layer=layer, name='k'):
                if active(layer):
                    value = output[0]
                    spec = sidecar.specs[layer]
                    dim = spec['key_dim'] if name == 'k' else spec['value_dim']
                    packets[layer][name] = value.detach().float().cpu().numpy()[0].reshape(-1, spec['heads'], dim).copy()
            def a_proj(module, args, output, layer=layer, attn=attn):
                if not active(layer):
                    return
                z = output.detach().float() + attn.dt_bias.detach()
                packets[layer]['x'] = args[0].detach().float().cpu().numpy()[0].copy()
                packets[layer]['g'] = (-attn.A_log.detach().float().exp() * F.softplus(z)).cpu().numpy()[0].copy()
                slope = torch.where(z > 20, torch.ones_like(z), z.sigmoid())
                packets[layer]['a_slope'] = slope.cpu().numpy()[0].copy()
                packets[layer]['Aexp'] = attn.A_log.detach().float().exp().cpu().numpy().copy()
            def b_proj(module, args, output, layer=layer):
                if active(layer):
                    packets[layer]['beta'] = output.detach().sigmoid().float().cpu().numpy()[0].copy()
            def after(module, args, kw, output, layer=layer):
                if active(layer):
                    sidecar.add_chunk(layer, clock_offset, packets.pop(layer))
            handles += [attn.register_forward_pre_hook(before, with_kwargs=True),
                        attn.k_conv1d.register_forward_hook(lambda m,a,k,o,l=layer: conv(m,a,k,o,layer=l,name='k'), with_kwargs=True),
                        attn.v_conv1d.register_forward_hook(lambda m,a,k,o,l=layer: conv(m,a,k,o,layer=l,name='v'), with_kwargs=True),
                        attn.a_proj.register_forward_hook(a_proj), attn.b_proj.register_forward_hook(b_proj),
                        attn.register_forward_hook(after, with_kwargs=True)]
        try:
            yield
            self._check_budget()
        finally:
            for h in handles:
                h.remove()

    def _release_unused_cuda(self, tensor):
        if tensor.is_cuda:
            before = dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved())
            torch.cuda.empty_cache()  # Free only unused allocator blocks for external Triton module loading.
            self._last_native_s['cuda_allocator_before_backward'] = dict(
                before=before, after=dict(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved()))

    def _regularizer(self):
        regularizer = super()._regularizer()
        self._release_unused_cuda(self.masters[0])
        return regularizer

    @contextmanager
    def _teaching_context(self, target, loss_mask):
        sidecar = self.eligibility
        leaves, handles = {}, []
        phase = {'forward': True}
        def initial(layer):
            def hook(module, args, kw):
                if not phase['forward'] or self.memories[layer]._replaying:
                    return
                cache = kw.get('past_key_values')
                if cache is None or len(cache.states) <= layer:
                    raise RuntimeError('selected incoming native S is unavailable')
                state = cache.states[layer].get('recurrent_state')
                if state is None:
                    raise RuntimeError('selected incoming native S is unavailable')
                leaf = state.detach().clone().requires_grad_(True)
                cache.states[layer]['recurrent_state'] = leaf
                leaves[layer] = leaf
            return hook
        def arrived(module, args, output):
            if not phase['forward']:
                return
            phase['forward'] = False
            if set(leaves) != set(self.layers):
                raise RuntimeError('loss did not use all selected incoming S leaves')
            _, ce = language_loss(output.logits, target, loss_mask)
            self._release_unused_cuda(output.logits)
            teaching = torch.autograd.grad(10 * ce, [leaves[l] for l in self.layers],
                                           retain_graph=True, allow_unused=True)
            if any(g is None for g in teaching):
                raise RuntimeError('current CE cannot reach selected incoming S')
            if any(not bool(torch.isfinite(g).all()) for g in teaching):
                raise FloatingPointError('nonfinite arrived native-S teaching')
            raw = sidecar.raw_credit({l: g.detach().float().cpu().numpy() for l, g in zip(self.layers, teaching)})
            empty = not (np.any(sidecar.code.U) and np.any(sidecar.code.V))
            self._native_extra = select_native_credit(raw, self.native_s_mode,
                getattr(sidecar, 'strict_old_gradients', None))
            self._last_native_s.update(has_factual_trace=True, teaching_available=True,
                trace_empty_before=bool(empty),
                old_credit_norm=math.sqrt(sum(float(np.sum(v.astype(np.float64)**2)) for v in raw.values())),
                teaching_norm=math.sqrt(sum(float(g.detach().float().square().sum()) for g in teaching)),
                source_age_resolved_credit=bool(getattr(sidecar, 'source_ages', None) is not None),
                strict_old_credit_norm=(None if not hasattr(sidecar, 'strict_old_gradients') else
                    math.sqrt(sum(float(np.sum(v.astype(np.float64)**2))
                                  for v in sidecar.strict_old_gradients.values()))))
        for layer in self.layers:
            handles.append(self.model.model.layers[layer].attn.register_forward_pre_hook(initial(layer), with_kwargs=True))
        handles.append(self.model.register_forward_hook(arrived))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
            leaves.clear()

    def step(self, x, target, **kwargs):
        self._native_extra = None
        self._last_native_s = self._empty_report()
        sidecar = self.eligibility
        if sidecar is not None and sidecar.pending is not None:
            if not sidecar.is_complete() or self.updates != self._native_window_version:
                raise RuntimeError('teaching/captured factual window mismatch')
            # Ensure the leaf hook sees a private execution cache, never live S.
            if kwargs.get('cache_override') is None:
                kwargs['cache_override'] = self.cache
            with self._teaching_context(target, kwargs.get('loss_mask')):
                result = super().step(x, target, **kwargs)
        else:
            if sidecar is not None and self._native_capture_started:
                raise RuntimeError('missing factual capture for enabled native-S learner')
            result = super().step(x, target, **kwargs)
        self._native_extra = None
        result['native_s'] = dict(self._last_native_s)
        return result

    def apply_gradients(self):
        report = apply_master_credit(self, self._native_extra)
        self._last_native_s.update(report)

    def state_dict(self):
        state = super().state_dict()
        state['native_s_sidecar'] = None if self.eligibility is None else dict(
            schema=1, mode=self.native_s_mode, state=self.eligibility.state_dict(),
            actual_count=self._native_actual_count, capture_started=self._native_capture_started,
            window_version=self._native_window_version)
        return state

    def load_state_dict(self, state, **kwargs):
        super().load_state_dict(state, **kwargs)
        stored = state.get('native_s_sidecar')
        self.eligibility = None
        self._native_extra = None
        self._last_native_s = self._empty_report()
        self._native_actual_count = 0
        self._native_capture_started = False
        self._native_window_version = None
        if stored is not None:
            if stored['schema'] != 1:
                raise ValueError('unknown native-S sidecar schema')
            s = stored['state']
            self.enable_native_s_eligibility(mode=stored['mode'], rank=s['rank'],
                max_bytes=s['max_bytes'], block_size=s['block_size'],
                source_ages=s.get('source_ages') is not None)
            self.eligibility.load_state_dict(s)
            self._native_actual_count = stored['actual_count']
            self._native_capture_started = stored['capture_started']
            self._native_window_version = stored['window_version']


class NativeSEligibilityBackend(PretrainedStreamBackend):
    def advance(self, tokens, *, clock_offset):
        learner = self.learner
        with learner.factual_capture(clock_offset):
            result = super().advance(tokens, clock_offset=clock_offset)
        if learner.eligibility is not None and learner.eligibility.pending is not None:
            learner._native_actual_count += len(tokens)
            if learner._native_actual_count == 128:
                learner.eligibility.set_end_s(learner._initial_states(learner.cache))
        return result

    def commit(self, inputs, targets, mask, boundary):
        learner = self.learner
        learner._last_native_s = learner._empty_report()
        result = super().commit(inputs, targets, mask, boundary)
        report = dict(result.get('native_s', learner._last_native_s))
        sidecar = learner.eligibility
        if sidecar is not None:
            learner._check_budget()
            if sidecar.pending is not None:
                if not sidecar.is_complete():
                    raise RuntimeError('cannot advance incomplete native-S participation')
                report['advancement'] = sidecar.advance_window()
                learner._native_actual_count = 0
            else:
                report['advancement'] = dict(skipped_inherited_partial_window=True)
            report.update(persistent_bytes=sidecar.nbytes,
                          estimated_temporary_bytes=learner._temporary_bytes(), arm=learner.native_s_mode)
            learner._check_budget()
        result['native_s'] = report
        return result
