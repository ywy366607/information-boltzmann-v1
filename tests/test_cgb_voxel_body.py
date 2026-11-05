import numpy as np

from information_boltzmann.cgb import voxel_body as vb


def small_body(seed, period=2):
    body = vb.Body(L=6, D=4, seed=seed, max_seg=6000, max_syn=20000)
    body.genome.period = period
    return body


def test_tables_are_paired_and_balance_weight():
    tables = [(vb.seg_rows(), vb.SEG_WEIGHTS), (vb.slow_seg_rows(), vb.SEG_WEIGHTS), (vb.hop_rows(), vb.HOP_WEIGHTS),
              (vb.slow_syn_rows(), vb.SYN_WEIGHTS)]
    for sign in (1, -1):
        tables += [(vb.syn_rows(sign), vb.SYN_WEIGHTS), (vb.site_rows(sign), vb.SYN_WEIGHTS)]
    for rows, weights in tables:
        assert vb.rows_paired(rows) and vb.row_weights_balanced(rows, weights)


def test_the_closed_body_conserves_content_while_it_lives():
    body = small_body(1)
    start = body.quanta()
    for value in np.random.default_rng(0).integers(0, 36, 600).tolist():
        body.free()
        body.arrive(value)
    assert body.quanta() == start
    assert int(body.state.counters[1]) > 0


def test_structure_changes_for_life():
    body = small_body(3)
    body.genome.grace = 20
    state = body.state
    for value in np.random.default_rng(1).integers(0, 36, 900).tolist():
        body.free()
        body.arrive(value)
    assert state.life[0] + state.life[1] > 0  # something was removed
    n = int(state.counters[0])
    alive = state.seg_cell[:n] >= 0
    assert (state.seg[:n][alive].sum(1) == body.genome.n_ch).all()
    assert (state.seg[:n][~alive] == 0).all()


def test_only_the_stream_moves_the_particle():
    body = small_body(4)
    body.arrive(7)
    assert int(body.state.counters[2]) == 7
    occupancy = body.free()
    assert occupancy.sum() > 0
