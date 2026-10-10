"""Bounded, forward-only CUDA graphs for recomputed physical microsteps.

Every replay returns owned tensors. Event checkpointing keeps the ordinary
autograd recomputation, so graph buffers never become the BPTT history.
"""
from __future__ import annotations

import torch

from .state_checkpoint import _flatten, _restore


class NoGradSegmentGraph:
    def __init__(self, operation, *, max_variants=2, parameters=()):
        self.operation = operation
        self.max_variants = max_variants
        self.parameters = tuple(parameters)
        self.records = {}
        self.replays = 0

    def __call__(self, *args, **kwargs):
        if torch.is_grad_enabled():
            return self.operation(*args, **kwargs)
        leaves = []
        schema = _flatten((args, kwargs), leaves)
        if not leaves or any(not value.is_cuda for value in leaves):
            return self.operation(*args, **kwargs)
        if torch.cuda.is_current_stream_capturing():
            return self.operation(*args, **kwargs)
        key = (schema, tuple((value.shape, value.stride(), value.dtype,
                              value.device) for value in leaves),
               tuple((p.data_ptr(), p.dtype, p.device, p.requires_grad) for p in self.parameters))
        record = self.records.get(key)
        if record is None:
            if len(self.records) >= self.max_variants:
                return self.operation(*args, **kwargs)
            static = [value.detach().clone(memory_format=torch.preserve_format)
                      for value in leaves]
            call_args, call_kwargs = _restore(schema, static)
            # Compilation and library initialization must finish outside capture.
            stream = torch.cuda.Stream(device=leaves[0].device)
            stream.wait_stream(torch.cuda.current_stream(leaves[0].device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.operation(*call_args, **call_kwargs)
            torch.cuda.current_stream(leaves[0].device).wait_stream(stream)
            torch.cuda.synchronize(leaves[0].device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs = self.operation(*call_args, **call_kwargs)
            output_leaves = []
            output_schema = _flatten(outputs, output_leaves)
            record = (static, graph, output_schema, output_leaves)
            self.records[key] = record
        static, graph, output_schema, outputs = record
        for buffer, value in zip(static, leaves):
            buffer.copy_(value)
        graph.replay()
        self.replays += 1
        # Returning private graph storage would overwrite older token states.
        return _restore(output_schema, [value.clone() for value in outputs])
