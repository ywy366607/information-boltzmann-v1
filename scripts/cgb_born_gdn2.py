"""Birth weights written by hand into the GDN-2 body (2026-11-03).

The one-tick learner of `gdn2_batched` cannot learn how to write its memory,
because a write pays only when a later tick reads it.  `gdn2_tbptt` learns it
with temporal backpropagation and ends 0.27 nats lower.  Here the write
circuit is written into the birth weights instead, the way a genome supplies
circuits whose credit lies beyond the individual's own learning.

The program, in residual coordinates of width 64:

* dim 0 is a constant; dims `A` hold a fixed random code of the current byte.
* layer 1, heads 1 and 2: a fading sum of past byte codes with decay 0.2 and
  0.6, written into blocks `P1` and `P2`.  Key and query are one constant
  direction, so the head state is that running sum.
* layer 2, heads 2 and 3: subtract the current byte's code from `A`.
* layer 2, heads 0 and 1: a delta-rule memory.  The key is the context without
  the current byte (block `P` minus the current code), the value is the
  current byte's code, the query is the context with the current byte.  A
  query therefore finds the byte that followed the same context last time, and
  the output adds that byte's code to `A`, which the tied decoder reads.

Every other weight keeps its default random initial value, so the control
`gdn2_batched` differs only in the written rows.  Learning stays on for every
weight in `gdn2_born`; `gdn2_born_fixed` keeps the written rows at birth value.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from cgb_language_candidates import CANDIDATES, GDN2Batched

ONE = 0
CODE = 24


def blocks(d: int, dk: int) -> dict:
    a = slice(1, 1 + CODE)
    p1 = slice(a.stop, a.stop + dk)
    p2 = slice(p1.stop, p1.stop + dk)
    if p2.stop > d:
        raise ValueError("width too small for the written program")
    return {"A": a, "P1": p1, "P2": p2}


def orthonormal_rows(rows: int, cols: int, generator: torch.Generator) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(cols, rows, generator=generator))
    return q[:, :rows].T.contiguous()


@torch.no_grad()
def write_birth(model, *, code_norm: float = 4.0, one: float = 2.0, decays=(0.2, 0.6),
                memory_decay: float = 0.98, gate: float = 4.0, memory_gain: float = 0.1,
                context_scale: float = 0.025, r2: float = 0.715,
                seed: int = 11) -> dict:
    """Write the program; return the masks of the written entries per parameter."""
    d, heads = model.d, model.heads
    dk = d // heads
    at = blocks(d, dk)
    generator = torch.Generator().manual_seed(seed)
    masks: dict[str, torch.Tensor] = {}

    def mark(name: str, param: torch.Tensor, index) -> None:
        mask = masks.setdefault(name, torch.zeros_like(param, dtype=torch.bool))
        mask[index] = True

    # Embedding, tied to the decoder: constant, then a random code of unit norm times code_norm.
    embedding = model.source.embedding.weight
    codes = torch.randn(256, CODE, generator=generator)
    codes = code_norm * codes / codes.norm(dim=1, keepdim=True)
    embedding[:, ONE] = one
    embedding[:, at["A"]] = codes
    embedding[:, at["P1"]] = 0.0
    embedding[:, at["P2"]] = 0.0
    for block in (slice(ONE, ONE + 1), at["A"], at["P1"], at["P2"]):
        mark("source.embedding.weight", embedding, (slice(None), block))
    rest = [i for i in range(d) if not (i == ONE or at["A"].start <= i < at["P2"].stop)]
    rms1 = math.sqrt((one ** 2 + code_norm ** 2 + float(embedding[:, rest].square().sum(1).mean())) / d)

    def head_rows(h: int) -> slice:
        return slice(h * dk, (h + 1) * dk)

    def logit(p: float) -> float:
        return math.log(p / (1.0 - p))

    def clear(layer, prefix: str, h: int) -> slice:
        rows = head_rows(h)
        for name in ("q_proj", "k_proj", "v_proj", "g_proj", "decay_proj", "erase_proj", "write_proj"):
            lin = getattr(layer, name)
            lin.weight[rows] = 0.0
            mark(f"{prefix}.{name}.weight", lin.weight, (rows,))
            if lin.bias is not None:
                mark(f"{prefix}.{name}.bias", lin.bias, (rows,))
        layer.o_proj.weight[:, rows] = 0.0
        mark(f"{prefix}.o_proj.weight", layer.o_proj.weight, (slice(None), rows))
        return rows

    # Layer 1: fading context sums.
    first = model.layers[0]
    silu = gate * one / rms1 * torch.sigmoid(torch.tensor(gate * one / rms1)).item()
    reads = {}
    for h, (decay, block) in zip((1, 2), zip(decays, ("P1", "P2"))):
        rows = clear(first, "layers.0", h)
        first.q_proj.weight[rows.start, ONE] = 1.0
        first.k_proj.weight[rows.start, ONE] = 1.0
        read = orthonormal_rows(dk, CODE, generator)
        first.v_proj.weight[rows, at["A"]] = read
        first.g_proj.weight[rows, ONE] = gate
        first.decay_proj.bias[rows] = logit(decay)
        first.erase_proj.bias[rows] = -8.0
        first.write_proj.bias[rows] = 8.0
        first.o_proj.weight[at[block], rows] = context_scale * torch.eye(dk)
        mark("layers.0.o_proj.weight", first.o_proj.weight, (at[block], slice(None)))
        reads[block] = read
    # The layer-1 output in a P block is factor * sum_j D^j read @ code_{t-j}, with this factor
    # relative to the code entering the same block of the next layer.
    factor = context_scale * silu / rms1

    # Layer 2: delta-rule memories keyed by the context before the current byte.
    second = model.layers[1]
    for h, block in zip((0, 1), ("P1", "P2")):
        rows = clear(second, "layers.1", h)
        read = reads[block]
        second.q_proj.weight[rows, at[block]] = torch.eye(dk) / factor
        second.k_proj.weight[rows, at[block]] = torch.eye(dk) / factor
        second.k_proj.weight[rows, at["A"]] = -read
        value = orthonormal_rows(dk, CODE, generator)
        second.v_proj.weight[rows, at["A"]] = value
        second.g_proj.weight[rows, ONE] = gate
        second.decay_proj.bias[rows] = logit(memory_decay)
        second.erase_proj.bias[rows] = 8.0
        second.write_proj.bias[rows] = 8.0
        second.o_proj.weight[at["A"], rows] = memory_gain * value.T
        mark("layers.1.o_proj.weight", second.o_proj.weight, (at["A"], slice(None)))
    # Layer 2, heads 2 and 3: remove the current byte's code from the decoder's view, so the
    # body does not predict the byte it just read.  The output scales as gate * one / r2 ** 2.
    basis = orthonormal_rows(CODE, CODE, generator)
    cancel = r2 ** 2 / (gate * one)
    for h, part in zip((2, 3), (slice(0, dk), slice(dk, CODE))):
        rows = clear(second, "layers.1", h)
        width = part.stop - part.start
        used = slice(rows.start, rows.start + width)
        second.q_proj.weight[rows.start, ONE] = 1.0
        second.k_proj.weight[rows.start, ONE] = 1.0
        second.v_proj.weight[used, at["A"]] = basis[part]
        second.g_proj.weight[rows, ONE] = gate
        second.erase_proj.bias[rows] = 8.0
        second.write_proj.bias[rows] = 8.0
        second.o_proj.weight[at["A"], used] = -cancel * basis[part].T
    return {"masks": masks, "factor": factor, "rms1": rms1, "silu": silu}


class GDN2Born(GDN2Batched):
    """`gdn2_batched` with the memory-write circuit written into its birth weights."""

    name = "gdn2_born"
    fixed_birth = False
    birth: dict = {}

    def __init__(self, budget_kib: int, seed: int = 3) -> None:
        super().__init__(budget_kib, seed)
        written = write_birth(self.model, **self.birth)
        if self.fixed_birth:
            named = dict(self.model.named_parameters())
            for name, mask in written["masks"].items():
                keep = (~mask).to(named[name].dtype)
                named[name].register_hook(lambda grad, keep=keep: grad * keep)
        self.state = self.model.initial_state(1)
        self.last = 32
        self.read()


class GDN2BornFixed(GDN2Born):
    """The same birth, with the written entries kept at their birth value for life."""

    name = "gdn2_born_fixed"
    fixed_birth = True


CANDIDATES.update({cls.name: cls for cls in (GDN2Born, GDN2BornFixed)})
