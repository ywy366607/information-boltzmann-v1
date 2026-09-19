"""Reusable block graphs with recomputation and an unbroken window gradient.

Each replay owns static buffers. Clone its outputs, and use a reentrant
checkpoint so backward recomputes that block immediately before its backward
graph. Without both operations, repeated forward replays would overwrite
earlier states/activations and produce incorrect BPTT gradients.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .sampling import padded_capacities


class _CaptureBlock(nn.Module):
    def __init__(self, owner, capacities):
        super().__init__()
        self.owner = owner
        self.weights = nn.ParameterList(list(owner.parameters()))
        self.call_block = owner.feature_block
        self.capacities = capacities
        self.has_token_input = (owner.write_operator is not None) or (getattr(owner, 'coupling_mode', 'none') in ('adaptive_force', 'message_coupling'))

    def forward(self, x, v, shared, clocks, noise, indices, fields, token_ids=None):
        tables = []
        for step in range(shared.shape[0] * clocks.shape[1]):
            layers, cursor = [], 0
            for capacity in self.capacities:
                rows = slice(cursor, cursor + capacity)
                idx, data = indices[step, rows], fields[step, rows]
                layers.append((idx[:, 0], idx[:, 1], data[:, :4], data[:, 4], data[:, 5] > 0))
                cursor += capacity
            tables.append(layers)
        if self.has_token_input and token_ids is not None:
            tok_embs = self.owner.core.force.embedding(token_ids)
            return self.call_block(x, v, shared, clocks, noise, tables, tok_embs=tok_embs)
        return self.call_block(x, v, shared, clocks, noise, tables)


class BlockGraph:
    def __init__(self, owner, block_tokens=8):
        self.block_tokens = block_tokens
        self.capacities = padded_capacities(owner.core.particles)
        self.slots = sum(self.capacities)
        n, h, steps = owner.core.particles, owner.core.force.embedding.embedding_dim, owner.steps
        device = next(owner.parameters()).device
        x = torch.zeros(n, 4, device=device, requires_grad=True)
        v = torch.zeros_like(x, requires_grad=True)
        shared = torch.zeros(block_tokens, h, device=device, requires_grad=True)
        clocks = torch.zeros(block_tokens, steps, 2, h, device=device, requires_grad=True)
        noise = torch.zeros(block_tokens * steps * 4, n, 4, device=device)
        indices = torch.zeros(block_tokens * steps, self.slots, 2, device=device, dtype=torch.long)
        indices[..., 1] = 1
        fields = torch.zeros(block_tokens * steps, self.slots, 6, device=device)
        fields[..., 0] = 1
        fields[..., 4] = .5
        wrapper = _CaptureBlock(owner, self.capacities)
        self.has_token_input = wrapper.has_token_input
        if self.has_token_input:
            token_ids = torch.zeros(block_tokens, device=device, dtype=torch.long)
            self.graphed = torch.cuda.make_graphed_callables(
                wrapper, (x, v, shared, clocks, noise, indices, fields, token_ids), allow_unused_input=True)
        else:
            self.graphed = torch.cuda.make_graphed_callables(
                wrapper, (x, v, shared, clocks, noise, indices, fields), allow_unused_input=True)
        self.original = owner.feature_block

    def packed(self, tables):
        a = tables[0][0]
        indices = torch.stack((a[0]._base, a[1]._base), -1)
        fields = torch.cat((a[2]._base, a[3]._base[:, None], a[4]._base[:, None].float()), -1)
        return indices, fields

    def run(self, x, v, shared, clocks, noise, tables, packed, token_ids=None):
        regular = len(shared) == self.block_tokens and all(
            tuple(len(layer[0]) for layer in step) == self.capacities for step in tables)
        if regular:
            start = tables[0][0][0].storage_offset()
            end = start + len(tables) * self.slots
            indices = packed[0][start:end].view(len(tables), self.slots, 2)
            fields = packed[1][start:end].view(len(tables), self.slots, 6)

            def execute(*args):
                outputs = self.graphed(*args)
                if torch.is_grad_enabled():
                    # make_graphed_callables also returns reusable *gradient*
                    # buffers. Upstream blocks/parameter accumulation need owned
                    # copies before the next replay overwrites those buffers.
                    def own_gradients(inputs, outputs):
                        return tuple(g.clone() if g is not None else None for g in inputs)
                    outputs[0].grad_fn.register_hook(own_gradients)
                return tuple(t.clone() for t in outputs)

            if self.has_token_input:
                args = (x, v, shared, clocks, noise, indices, fields, token_ids)
            else:
                args = (x, v, shared, clocks, noise, indices, fields)
        else:
            # Bind the complete table now; no late-bound loop closure in backward.
            def execute(*args, table=tables):
                if self.has_token_input and token_ids is not None:
                    tok_embs = self.original.__self__.core.force.embedding(token_ids)
                    return self.original(*args, table, tok_embs=tok_embs, eager=True)
                return self.original(*args, table, eager=True)

            args = (x, v, shared, clocks, noise)
        if torch.is_grad_enabled():
            return checkpoint(execute, *args, use_reentrant=True, preserve_rng_state=False)
        return execute(*args)
