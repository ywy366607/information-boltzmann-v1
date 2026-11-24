import torch

from scripts.clock_credit_jet import Jet, add, compose, mark
from scripts.delayed_return_jet import (
    DatedEligibility, explicit_memory_responses, future_read_matrix, lift,
    memory_responses, pack, target_code,
)


def draw(g, n, m):
    return Jet(*(torch.randn(n, m, generator=g, dtype=torch.double)*.2 for _ in range(4)))


def test_rectangular_lift_preserves_ordered_clock_credit_composition():
    g = torch.Generator().manual_seed(782)
    a, b = draw(g, 3, 5), draw(g, 5, 2)
    torch.testing.assert_close(lift(compose(a, b)), lift(a)@lift(b))
    torch.testing.assert_close(pack(compose(a, b)), lift(a)@pack(b))


def test_time_varying_delayed_responses_keep_every_source_and_pending_tail():
    g = torch.Generator().manual_seed(51)
    d = [draw(g, 4, 4) for _ in range(5)]
    b = [draw(g, 2, 4) for _ in d]
    v = [draw(g, 4, 3) for _ in d]
    initial = draw(g, 4, 3)
    got, tail = memory_responses(d, b, v, initial)
    separated, tails = explicit_memory_responses(d, b, v, initial)
    for actual, parts in zip([*got, tail], [*separated, tails]):
        combined = parts[0]
        for part in parts[1:]:
            combined = add(combined, part)
        torch.testing.assert_close(pack(actual), pack(combined))


def test_joint_future_code_preserves_declared_delay_reads_with_rank_bound():
    # Matrix unit check, independent of any model/capability experiment.
    z = torch.zeros(2, 4, dtype=torch.double)
    b = [Jet(torch.tensor([[1., 0, 0, 0], [0, 1., 0, 0]]), z, z, z)]*3
    b = [Jet(*(v.double() for v in vars(j).values())) for j in b]
    d = [mark(torch.eye(4, dtype=torch.double), t) for t in (.1, .4, .2)]
    reads = future_read_matrix(d, b)
    code = target_code(reads)
    assert code['rank'] == 8
    assert code['residual_operator_norm'] < 1e-12
    g = torch.Generator().manual_seed(6)
    x = torch.randn(16, 7, generator=g, dtype=torch.double)
    torch.testing.assert_close(reads@x, code['read_back']@(code['code']@x))


def test_dated_credit_matches_each_insertion_and_keeps_owner_axis():
    g = torch.Generator().manual_seed(8)
    state = DatedEligibility(torch.zeros(3, 2, dtype=torch.double),
                             torch.zeros(3, 2, dtype=torch.double))
    explicit = []
    for dt in (.3, .8, .1, 1.2):
        a = torch.randn(3, 3, generator=g, dtype=torch.double)*.2
        source = torch.randn(3, 2, generator=g, dtype=torch.double)
        explicit = [(a@e, age+dt) for e, age in explicit]+[(source, 0.)]
        state.advance(dt, transport=lambda x: a@x, source=source)
        torch.testing.assert_close(state.eligibility, sum(e for e, _ in explicit))
        torch.testing.assert_close(state.age_moment, sum(age*e for e, age in explicit))
    read = torch.randn(3, 2, generator=g, dtype=torch.double)
    e, age = state.read(read, future_wait=.7)
    torch.testing.assert_close(e, sum((read*x).sum() for x, _ in explicit))
    torch.testing.assert_close(age, sum((read*x*(t+.7)).sum() for x, t in explicit))


def test_native_projection_separates_old_source_from_last_timestamp():
    from scripts.audit_native_delayed_return import project

    sources = [dict(event=i, time=float(i), score=s, kind=kind,
                    within_learning=True, internal_relation=None)
               for i, s, kind in ((2, .5, 'internal'), (4, -.5, 'release'))]
    pending = project(sources, (5, 5.), 1.)
    returned = project(sources, (9, 9.), -.25)
    assert pending['eligibility'] == returned['eligibility'] == 0.
    assert pending['joint_rank'] == returned['joint_rank'] == 2
    assert pending['last_stamp_error'] == pending['age_moment'] == 1.
    assert returned['last_stamp_error'] == -returned['age_moment'] == .25
