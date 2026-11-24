"""CUDA decoding with live queries and draft-scoped, exact fact projections."""
from contextlib import contextmanager

import numpy as np
import torch
from torch.nn import functional as F

from scripts.ib_local.language_navigation_turns_v3 import DecisionGraphV3
from scripts.ib_local.train_gdn_fast_table import FLOOR


def configure_fused_step(engine, config):
    """Install the same runtime on first use and checkpoint restoration."""
    previous = engine.decision
    engine.decision = ScopedDecisionGraph(engine.model, capture=previous.capture,
        fused_step=True, capture_handover=hasattr(previous, '_capture_handover_budget'))
    config['decode_runtime'] = dict(schema=1, fused_step=True, precision='FP32',
        scope='original one-byte inference; original sequence learning and gradients')


class ScopedDecisionGraph(DecisionGraphV3):
    def __init__(self, model, capture=True, *, optimize=True, pack_inputs=True,
                 capture_handover=False, fused_step=False):
        w=model.encoder.weight
        self.model = model
        self.optimize = optimize
        self.fact_cache = None
        self.field_cache = None
        self.step_bundles = []
        if fused_step and optimize and capture and w.is_cuda:
            from scripts.ib_local.gdn_decode_step import make_bundle
            self.step_bundles = [(layer, make_bundle(layer)) for layer in model.base.layers]
            self.step_bundles = [(layer, bundle) for layer, bundle in self.step_bundles if bundle is not None]
        memory = model.memory
        field = getattr(model.body, 'coevolution', None)
        with torch.no_grad():
            if optimize and hasattr(memory, 'action'):
                shape = (memory.capacity, model.encoder.out_features)
                n_fact = 4 if hasattr(model, 'phase_boltzmann') else 3
                self.fact_cache = tuple(w.new_empty(shape) for _ in range(n_fact))
            if optimize and field is not None:
                self.field_cache = tuple(w.new_empty(field.slots, field.rank) for _ in range(2))
            self._refresh_projections()
        self.question=torch.zeros(model.encoder.out_features,device=w.device,dtype=w.dtype)
        self.budget=torch.zeros(1,1,1,device=w.device,dtype=w.dtype)
        self._input_host = None
        if optimize and pack_inputs and capture and w.is_cuda and w.dtype == torch.float32:
            # One 1036-byte transfer replaces byte fill, a temporary uploaded
            # probability tensor, dtype conversion and budget fill. Views are
            # aligned and keep the CUDA graph's original integer/FP32 types.
            self._input_host = torch.empty(1036, dtype=torch.uint8, pin_memory=True)
            self._input_device = torch.zeros(1036, dtype=torch.uint8, device=w.device)
            self._packed_decode_inputs = (self._input_device[:8].view(torch.int64).reshape(1,1),
                self._input_device[8:1032].view(torch.float32).reshape(1,1,256))
            self.budget = self._input_device[1032:].view(torch.float32).reshape(1,1,1)
            self._host_byte = self._input_host[:8].view(torch.int64)
            self._host_logp = self._input_host[8:1032].view(torch.float32).numpy()
            self._host_budget = self._input_host[1032:].view(torch.float32)
        previous=getattr(model,'_question_context',None)
        previous_budget=getattr(model,'_expression_remaining',None)
        if optimize and capture_handover and hasattr(model, 'handover'):
            self._capture_handover_budget = self.budget.reshape(1, 1)
        try:
            if capture and w.is_cuda:
                model._question_context=self.question
                model._expression_remaining=self.budget
            with self._decode_context():
                super().__init__(model,capture=capture)
        finally:
            model._question_context=previous
            model._expression_remaining=previous_budget

    @torch.no_grad()
    def _refresh_projections(self):
        from scripts.ib_local.preview_runtime import _scope
        active = _scope.get()
        encoder, memory = self.model.encoder, self.model.memory
        field = getattr(self.model.body, 'coevolution', None)
        scope_key = None
        if active is not None and active[0] is self.model:
            scope_key = (
                id(active[1]),
                tuple(p._version for p in encoder.parameters()),
                memory.source._version if hasattr(memory, 'source') else None,
                memory.action._version if hasattr(memory, 'action') else None,
                memory.long_successor._version if hasattr(memory, 'long_successor') else None,
                None if field is None else field.anchors._version,
            )
            if getattr(self, '_last_refresh_scope_key', None) == scope_key:
                return
        if self.step_bundles:
            from scripts.ib_local.gdn_decode_step import refresh_many
            refresh_many(self.step_bundles)
        if self.fact_cache is not None:
            enc_src = encoder(memory.source)
            enc_long = encoder(memory.long_successor)
            self.fact_cache[0].copy_(F.normalize(enc_src, dim=-1))
            self.fact_cache[1].copy_(encoder(memory.action))
            self.fact_cache[2].copy_(enc_long)
            if len(self.fact_cache) >= 4:
                self.fact_cache[3].copy_(F.normalize(enc_long - enc_src, dim=-1))
        if self.field_cache is not None:
            encoded = encoder(field.anchors)
            self.field_cache[0].copy_(F.normalize(field.key(encoded[0]), dim=-1))
            self.field_cache[1].copy_(field.bind(encoded.permute(1, 0, 2).flatten(1)).tanh())
        self._last_refresh_scope_key = scope_key

    @contextmanager
    def _decode_context(self):
        """No cache/flag escapes into learning, checkpointing, or other readers."""
        if not self.optimize:
            yield
            return
        changes = [(self.model.body, '_decode_no_diagnostics', True)]
        changes += [(layer, '_decode_no_diagnostics', True) for layer in self.model.base.layers]
        changes += [(layer, '_decode_step_bundle', bundle) for layer, bundle in self.step_bundles]
        if self.fact_cache is not None:
            changes.append((self.model.memory, '_decode_projection_cache', self.fact_cache))
        field = getattr(self.model.body, 'coevolution', None)
        if field is not None:
            changes += [(field, '_decode_no_diagnostics', True),
                        (field, '_decode_fused_exchange', True),
                        (field, '_decode_projection_cache', self.field_cache)]
        missing = object()
        saved = [(obj, name, getattr(obj, name, missing)) for obj, name, _ in changes]
        try:
            for obj, name, value in changes:
                setattr(obj, name, value)
            yield
        finally:
            for obj, name, value in reversed(saved):
                if value is missing:
                    delattr(obj, name)
                else:
                    setattr(obj, name, value)

    @torch.no_grad()
    def start(self, state):
        # Re-encode after every parameter/fact/version update, into stable
        # buffers already referenced by the CUDA graph. No lifetime cache.
        self._refresh_projections()
        return super().start(state)

    @torch.no_grad()
    def step(self, byte, probability):
        if self.capture:
            q=getattr(self.model,'_question_context',None)
            self.question.zero_() if q is None else self.question.copy_(q.reshape_as(self.question))
            remaining=getattr(self.model,'_expression_remaining',None)
            if self._input_host is not None:
                self._host_byte.fill_(byte)
                np.copyto(self._host_logp, np.log(np.maximum(probability, FLOOR)), casting='unsafe')
                self._host_budget.fill_(0. if remaining is None else float(remaining))
                # Synchronous completion of this tiny upload makes host-buffer
                # reuse safe even in audit loops without per-byte CPU readback.
                self._input_device.copy_(self._input_host, non_blocking=False)
                self._replay_captured()
                return self.logits[0,0].clone(), self.h[0].clone()
            self.budget.zero_() if remaining is None else self.budget.fill_(float(remaining))
            # The captured kernels already reference the scoped buffers. Do
            # not reassign 32 layers' Python attributes for every replay.
            return super().step(byte,probability)
        with self._decode_context():
            return super().step(byte,probability)
