"""Finite, ordered eliminated-memory responses with time/credit transport.

Column convention: h[t+1] = D[t] h[t] + v[t], y[t] = B[t] h[t].
Each block is a clock_credit_jet.Jet. Initial history, each insertion and
the final unreturned state keep separate identities in the explicit audit.
The recursive implementation uses one bounded hidden state.
"""
from dataclasses import dataclass

import torch

from scripts.clock_credit_jet import Jet, add, compose


def lift(a):
    """Act on column (mass, clock, credit, mixed), with rectangular blocks."""
    m, t, e, h = a.mass, a.clock, a.credit, a.mixed
    z = torch.zeros_like(m)
    return torch.cat((torch.cat((m, z, z, z), 1),
                      torch.cat((t, m, z, z), 1),
                      torch.cat((e, z, m, z), 1),
                      torch.cat((h, e, t, m), 1)), 0)


def pack(a):
    return torch.cat((a.mass, a.clock, a.credit, a.mixed), 0)


def memory_responses(transports, returns, sources, initial):
    """Bounded recurrence; return the complete still-pending jet as well."""
    if not len(transports) == len(returns) == len(sources):
        raise ValueError('one transport, return and end-of-stage source per stage')
    state, outputs = initial, []
    for d, b, v in zip(transports, returns, sources):
        outputs.append(compose(b, state))
        state = add(compose(d, state), v)
    return outputs, state


def explicit_memory_responses(transports, returns, sources, initial):
    """Delay-resolved finite MZ expansion: B[t] D[t-1] ... D[j+1] v[j]."""
    states, outputs = [initial], []
    for d, b, v in zip(transports, returns, sources):
        outputs.append([compose(b, h) for h in states])
        states = [compose(d, h) for h in states] + [v]
    return outputs, states


def future_read_matrix(transports, returns):
    """Stack B0, B1 D0, B2 D1 D0, ... in the lifted joint coordinates.

    This is the declared finite future target. Its row space determines
    which hidden differences affect these delayed outputs, times or credit.
    """
    if not transports or len(transports) != len(returns):
        raise ValueError('nonempty equal-length transports and returns required')
    first = lift(transports[0])
    carried = torch.eye(first.shape[1], dtype=first.dtype, device=first.device)
    rows = []
    for d, b in zip(transports, returns):
        rows.append(lift(b) @ carried)
        carried = lift(d) @ carried
    return torch.cat(rows, 0)


def target_code(reads, *, atol=0.0, rtol=None):
    """Orthogonal row-space code and the directly measured discarded read.

    Exact quotient in real arithmetic when all positive singular modes are
    kept. Float cutoffs and the actual residual are returned explicitly.
    """
    _, singular, vh = torch.linalg.svd(reads, full_matrices=False)
    rtol = max(reads.shape)*torch.finfo(reads.dtype).eps if rtol is None else rtol
    threshold = max(atol, rtol*float(singular.max())) if len(singular) else atol
    keep = singular > threshold
    code = vh[keep]
    read_back = reads @ code.T
    error = reads - read_back @ code
    return dict(code=code, read_back=read_back, singular_values=singular,
                threshold=threshold, rank=int(keep.sum()),
                residual_operator_norm=float(torch.linalg.matrix_norm(error, ord=2)))


@dataclass
class DatedEligibility:
    """Eligibility E and source-age moment A, retaining parameter ownership.

    source enters at the END of a stage. With common transport L and elapsed
    dt: E' = L E + source; A' = L A + dt L E. Future time-aware linear reads
    can use A + future_dt E, regardless of the number of historical sources.
    Parameter/owner axes remain columns; they are never summed together.
    """
    eligibility: torch.Tensor
    age_moment: torch.Tensor

    def advance(self, elapsed, transport=None, source=None):
        if elapsed < 0:
            raise ValueError('elapsed time must be nonnegative')
        e, a = self.eligibility, self.age_moment
        if transport is not None:
            e, a = transport(e), transport(a)
        self.eligibility = e if source is None else e + source
        self.age_moment = a + elapsed*e

    def read(self, cotangent, future_wait=0.0):
        return (cotangent*self.eligibility).sum(), (
            cotangent*(self.age_moment+future_wait*self.eligibility)).sum()
