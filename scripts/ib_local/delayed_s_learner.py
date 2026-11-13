"""Same native-S update with a measured, actually arrived feedback mask.

Ordinary CE/BPTT, replay, SIGReg, Adam and qualification advance are unchanged.
Only the delayed external correction supplies the extra third factor here.
"""
from contextlib import contextmanager
import math

import numpy as np
import torch

from scripts.ib_local.irreversible_credit_stream import IrreversibleCreditStream
from scripts.ib_local.native_s_learner import NativeSEligibilityLearner, NativeSEligibilityBackend


def live_read_receipts(memories):
    """The read that actually produced the next prediction, not a later reread."""
    result={}
    for layer,wrapper in memories.items():
        read=wrapper.last_read
        offset=int(wrapper.query_clock_offsets[-1])
        selected=[]
        for slot,weight in zip(read.slots[-1].detach().cpu().tolist(),read.weights[-1].detach().cpu().tolist()):
            if slot<0:continue
            selected.append(dict(slot=slot,version=int(wrapper.memory.read_credit_version[slot]),weight=weight,
                                 age_tokens=int(wrapper.memory.clock)+offset-int(wrapper.memory.birth[slot])))
        result[str(layer+1)]=selected
    return result


class _FeedbackReadout:
    """Mask contribution with the SAME denominator as the ordinary CE loss."""
    def __init__(self, actual, fraction, before_credit=None):
        self.actual, self.fraction = actual, fraction
        self.before_credit = before_credit

    def __getattr__(self, name):
        return getattr(self.actual, name)

    def raw_credit(self, teaching):
        if self.before_credit is not None:
            self.before_credit()
        # In this answer-only SFT protocol the fraction is exactly one. Do not
        # duplicate both incoming-S arrays while the backward graph is live.
        if self.fraction == 1.:
            return self.actual.raw_credit(teaching)
        return self.actual.raw_credit({l:g*self.fraction for l,g in teaching.items()})


class DelayedFeedbackLearner(NativeSEligibilityLearner):
    feedback_mask = None

    def _release_before_cpu_credit(self):
        parameter=next(self.model.parameters(),None)
        if parameter is not None and parameter.is_cuda:
            before=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())
            torch.cuda.empty_cache()
            self._last_native_s['cuda_allocator_before_cpu_credit']=dict(before=before,
                after=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved()))

    def _temporary_bytes(self):
        extra = 0 if self.eligibility is None else self.eligibility.layout.size*4+128
        return super()._temporary_bytes()+extra

    @contextmanager
    def _teaching_context(self, target, loss_mask):
        mask = self.feedback_mask
        if mask is None or len(mask) != target.numel():
            raise ValueError('explicit aligned feedback mask required')
        if not any(mask):
            yield  # Ordinary learning still runs; no invented delayed outcome.
        else:
            actual = torch.tensor([mask], dtype=torch.bool, device=target.device)
            if loss_mask is not None and bool((actual & ~loss_mask).any()):
                raise ValueError('feedback targets must also be actual external teaching')
            denominator = target.numel() if loss_mask is None else int(loss_mask.sum())
            fraction = int(actual.sum())/denominator
            sidecar = self.eligibility
            self.eligibility = _FeedbackReadout(sidecar, fraction, self._release_before_cpu_credit)
            try:
                with super()._teaching_context(target, actual):
                    yield
                self._last_native_s['teaching_norm'] *= fraction
                self._last_native_s.update(feedback_targets=int(actual.sum()),
                                           total_teaching_targets=denominator,
                                           feedback_fraction=fraction)
            finally:
                self.eligibility = sidecar


class DelayedFeedbackBackend(NativeSEligibilityBackend):
    phase = None
    gap_log_retention = None

    def begin_gap(self):
        self.gap_log_retention = {l:np.zeros(s['heads'], dtype=np.float64)
                                  for l,s in self.learner.eligibility.specs.items()}

    def advance(self, tokens, *, clock_offset):
        def observed(module,args,output):
            self.last_live_reads=live_read_receipts(self.learner.memories)
        handle=self.learner.model.register_forward_hook(observed)
        try:result = super().advance(tokens, clock_offset=clock_offset)
        finally:handle.remove()
        if self.phase == 'intervening' and self.gap_log_retention is not None:
            pending = self.learner.eligibility.pending
            if pending is None:
                raise RuntimeError('gap began before native qualification capture became available')
            for layer in self.learner.layers:
                packet = pending['chunks'][layer][-1]
                if len(packet['g']) != len(tokens):
                    raise RuntimeError('retention diagnostic is not the actual consumed packet')
                self.gap_log_retention[layer] += packet['g'].astype(np.float64).sum(axis=0)
        return result

    def commit(self, inputs, targets, mask, boundary):
        before={l:w.memory.read_credit_version.detach().clone() for l,w in self.learner.memories.items()}
        result=super().commit(inputs,targets,mask,boundary)
        writes={}
        for layer,wrapper in self.learner.memories.items():
            slots=(wrapper.memory.read_credit_version!=before[layer]).nonzero().flatten().tolist()
            writes[str(layer+1)]=[dict(slot=i,version=int(wrapper.memory.read_credit_version[i]),
                birth_clock=int(wrapper.memory.birth[i]),replay_input_tokens=wrapper.capsules[i].inputs.shape[1])
                for i in slots if wrapper.capsules[i] is not None]
        result['reservoir_writes']=writes
        return result

    def gap_survival(self):
        return {str(l):dict(log_upper_bounds=v.tolist(),
                            upper_bounds=[math.exp(float(x)) for x in v])
                for l,v in self.gap_log_retention.items()}


class DelayedFeedbackStream(IrreversibleCreditStream):
    """Tag teaching positions, including targets crossing a128-input boundary."""
    def __init__(self, backend, **kwargs):
        super().__init__(backend, **kwargs)
        self.feedback_positions = []
        self.arrival_phase = None

    def receive(self, tokens, *, phase, teach=False):
        self.arrival_phase = self.backend.phase = phase
        return super().receive(tokens, phase=phase, teach=teach)

    def _target(self, token, learn):
        self.feedback_positions.append(self.arrival_phase == 'feedback_answer' and bool(learn))
        return super()._target(token, learn)

    def _commit(self):
        if len(self.feedback_positions) != len(self.targets):
            raise RuntimeError('feedback tags not aligned with actual target positions')
        self.backend.learner.feedback_mask = list(self.feedback_positions)
        try:
            super()._commit()
        finally:
            self.backend.learner.feedback_mask = None
        self.feedback_positions = []

    def state_dict(self):
        result = super().state_dict()
        result['delayed_feedback_positions'] = list(self.feedback_positions)
        return result

    def restore_latest(self, state):
        super().restore_latest(state)
        positions = state.get('delayed_feedback_positions', [False]*len(self.targets))
        if len(positions) != len(self.targets) or any(type(v) is not bool for v in positions):
            raise ValueError('invalid pending feedback tags')
        self.feedback_positions = list(positions)
