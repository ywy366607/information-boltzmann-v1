"""E3 body: one two-in two-out table on a voxel lattice, grown and taught by the language stream alone.

Design: docs/information_boltzmann/CGB_E3_DESIGN_V1_20261102.md. Lean law layer:
`CGB.VoxelChem.axioms`, `CGB.VoxelState.row_conserves`.

The code only interprets rows `a + c -> b + d`. Rows are data and come in reverse pairs. Species
have indices, not roles; the comments name them for the reader only. Every count is an integer and
the whole body is closed: nothing is imported or exported, the byte only moves the particle.

Nothing outside the table reads or teaches the body. The sheet carries one conserved particle
(R1 `one_unit_one_label`), so a sheet site is a label. Undriven it hops under the site energies the
interior sets, much faster than the interior moves (the separation G13 needs), and its occupancy is
the prediction (G13, `sector_law_is_softmax`). At an arrival the world pins it. A cell is sensory
because its membrane sits where the particle is, pinned or free, and motor because its terminal sets
a site energy.

Credit has one form at both depths (G14a `two_phase_drift`, G14c `phases_must_balance`). Pinned, the
particle turns material into a signal species; free, into its twin, at the same rate. The two spread
inward, so their local balance is arrival minus prediction, and it reaches a contact only through
local contact (G9). A contact where both sides took part votes by that balance, the reciprocal
product of `GenerationReciprocalCredit`. Sheet contacts read the particle itself.

Growth never stops. Tips sprout from contacts that hold, contacts that lose every slot are removed,
and neurites left without contacts retract and return their material.
"""
from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass

import numpy as np
from numba import njit

# Segment scope: R C E X, then the voxel's holes.
R, C, E, X, H = range(5)
# Synapse scope: Y0 Y1 Y2 Zp Zm Z0, pre R C E X, post R C E X, post voxel A G N holes, particle free and pinned.
Y0, Y1, Y2, ZP, ZM, Z0 = range(6)
PRE, POST, VOX = 6, 10, 14
VA, VG, VN, VH = VOX, VOX + 1, VOX + 2, VOX + 3
PFREE, PPIN = 18, 19
NORM_CH, NORM_VOX = 0, 1
SEG_WEIGHTS, SYN_WEIGHTS, HOP_WEIGHTS = [1] * 5, [1] * 20, [1] * 8


def pair(a, c, b, d, k, drop=1e-3, norm=NORM_CH):
    """A row and its reverse; `drop` is the Boltzmann factor of the reverse."""
    return [(a, c, b, d, k, norm), (b, d, a, c, k * drop, norm)]


def seg_rows(ignite=.8, regen=1., prime=.3, leak=.08, spent=.25):
    return np.array(
        pair(C, C, C, E, ignite) + pair(C, E, E, E, regen) + pair(E, R, E, C, prime)
        + pair(C, H, R, H, leak, 1e-5, NORM_VOX) + pair(E, H, X, H, spent, 1e-6, NORM_VOX), float)


def slow_seg_rows(recover=.24):
    """A random quarter of segments every 16 ticks: a channel comes back after about 300 ticks."""
    return np.array(pair(X, H, R, H, recover, 1e-6, NORM_VOX), float)


def hop_rows(cable=3., cable_c=.15):
    """Between adjacent segments of one cell: u species 0-3, v species 4-7."""
    return np.array(pair(E, R + 4, E, E + 4, cable) + pair(C, R + 4, R, C + 4, cable_c, 1.)
                    + pair(C + 4, R, R + 4, C, cable_c, 1.), float)


def _retention(grow, cut, fade):
    return (pair(ZP, VA, Y0, Y0, grow, 1e-3, NORM_VOX) + pair(ZP, VH, Y0, VH, fade, 1e-3, NORM_VOX)
            + pair(ZM, VH, VA, VH, cut, 1e-3, NORM_VOX) + pair(ZM, VH, Y0, VH, fade, 1e-3, NORM_VOX))


def _transmission(sign, transmit):
    return pair(Y1, POST + R, Y2, POST + C, transmit) if sign > 0 else pair(Y1, POST + C, Y1, POST + R, transmit)


def syn_rows(sign=1, release=.9, transmit=.5, stay=.04, tag=.3, vote=.2, grow=.2, cut=.02, fade=.01):
    """Interior contacts: joint activity leaves a neutral tag, and the local balance of the two signals decides it."""
    up, down = (VG, VN) if sign > 0 else (VN, VG)
    return np.array(pair(PRE + E, Y0, PRE + E, Y1, release) + _transmission(sign, transmit)
                    + pair(Y1, POST + E, Z0, POST + E, tag) + pair(Y1, VH, Y2, VH, stay, 1e-3, NORM_VOX)
                    + pair(Z0, up, ZP, up, vote) + pair(Z0, down, ZM, down, vote)
                    + pair(Z0, VH, Y0, VH, fade, 1e-3, NORM_VOX) + _retention(grow, cut, fade), float)


def site_rows(sign, release=.9, transmit=.05, stay=.04, tag=.3, grow=.2, cut=.02, fade=.01):
    """Contacts on the sheet: a released site waits, and the particle's two forms move it in opposite directions."""
    up, down = (PPIN, PFREE) if sign > 0 else (PFREE, PPIN)
    return np.array(pair(PRE + E, Y0, PRE + E, Y1, release) + _transmission(sign, transmit)
                    + pair(Y1, up, ZP, up, tag) + pair(Y1, down, ZM, down, tag)
                    + pair(Y1, VH, Y2, VH, stay, 1e-3, NORM_VOX) + _retention(grow, cut, fade), float)


def slow_syn_rows(reload=1., turnover=.0008):
    """A random eighth of contacts every 16 ticks: release sites refill, and unused slots slowly return to the pool."""
    return np.array(pair(Y2, VH, Y0, VH, reload, 1e-3, NORM_VOX) + pair(Y0, VH, VA, VH, turnover, 1e-6, NORM_VOX), float)


def row_weights_balanced(rows, weights):
    """The check of `VoxelState.row_conserves`, on the actual table."""
    return all(weights[int(a)] + weights[int(c)] == weights[int(b)] + weights[int(d)]
               for a, c, b, d, _, _ in rows)


def rows_paired(rows):
    """Every row has a reverse."""
    keys = {(int(a), int(c), int(b), int(d)) for a, c, b, d, _, _ in rows}
    return all((b, d, a, c) in keys for a, c, b, d in keys)


@njit(cache=True)
def _fire(rows, loc, n_ch, n_vox):
    """Run a table once on local counts; a row runs at most what its inputs allow."""
    for r in range(rows.shape[0]):
        a, c, b, d = int(rows[r, 0]), int(rows[r, 1]), int(rows[r, 2]), int(rows[r, 3])
        k = rows[r, 4]
        if k <= 0.:
            continue
        ca, cc = loc[a], loc[c]
        if a == c:
            cc -= 1
        limit = min(ca, cc)
        if limit <= 0:
            continue
        norm = n_ch if rows[r, 5] == 0 else n_vox
        p = 1. - np.exp(-k * max(ca, cc) / norm)
        events = 0
        if limit * p < .05:
            events = 1 if np.random.random() < limit * p else 0
        else:
            for _ in range(limit):
                if np.random.random() < p:
                    events += 1
        if events:
            loc[a] -= events
            loc[c] -= events
            loc[b] += events
            loc[d] += events


State = namedtuple("State", "seg seg_v seg_cell seg_parent seg_kind seg_out seg_in seg_first seg_next seg_born cell_sign "
                   "cell_n fld syn syn_pre syn_post syn_next syn_in_next active syn_active counters cell_spiked site_z "
                   "site_next occupancy free_seg free_syn life")


@njit(cache=True)
def _holes(s, v, cap):
    return cap - s.fld[v, 0] - s.fld[v, 1] - s.fld[v, 2]


@njit(cache=True)
def _load_syn(syn, s, e, v, cap):
    for q in range(6):
        syn[q] = s.syn[e, q]
    for q in range(3):
        syn[VOX + q] = s.fld[v, q]
    syn[VH] = _holes(s, v, cap)


@njit(cache=True)
def _spread(s, cap, lattice, q, decay):
    """One signal species: it hops to a random neighbour voxel and decays back into material."""
    L, D = lattice
    for v in range(s.fld.shape[0]):
        count = s.fld[v, q]
        if count <= 0:
            continue
        back = np.random.binomial(count, decay)
        s.fld[v, q] -= back
        s.fld[v, 0] += back
        step = np.random.randint(0, 6)
        x, y, z = v % L, (v // L) % L, v // (L * L)
        x, y, z = x + (step == 0) - (step == 1), y + (step == 2) - (step == 3), z + (step == 4) - (step == 5)
        if x < 0 or x >= L or y < 0 or y >= L or z < 0 or z >= D:
            continue
        t = x + L * y + L * L * z
        moves = min(_holes(s, t, cap), (s.fld[v, q] + 1) // 2)
        s.fld[v, q] -= moves
        s.fld[t, q] += moves


@njit(cache=True)
def run_ticks(s, seg_table, hop_table, exc_table, inh_table, site_exc, site_inh, seg_slow, syn_slow, pin, ticks, cap, n_ch,
              k_sense, emit, unit, hops, record_from, diffuse_material, lattice, sheet):
    """`pin` < 0 runs free, otherwise the world holds the particle on site `pin`. `sheet` lists membrane on the sheet."""
    L, D = lattice
    n_seg, n_syn = s.counters[0], s.counters[1]
    sites = L * L
    held = pin >= 0
    loc = np.zeros(5, np.int64)
    hop = np.zeros(8, np.int64)
    syn = np.zeros(20, np.int64)
    if held:
        s.counters[2] = pin
    for tick in range(ticks):
        at = s.counters[2]
        if not held:
            x0, y0 = at % L, at // L
            for _ in range(hops):
                step = np.random.randint(0, 4)
                x1, y1 = x0 + (step == 0) - (step == 1), y0 + (step == 2) - (step == 3)
                if 0 <= x1 < L and 0 <= y1 < L:
                    gap = unit * (s.site_z[x1 + L * y1] - s.site_z[x0 + L * y0])
                    if gap >= 0 or np.random.random() < np.exp(gap):
                        x0, y0 = x1, y1
                if tick >= record_from:
                    s.occupancy[x0 + L * y0] += 1
            at = x0 + L * y0
            s.counters[2] = at
        # The particle turns material into one of two twin signals, at the same rate in both phases.
        made = min(emit, s.fld[at, 0])
        s.fld[at, 0] -= made
        s.fld[at, 1 if held else 2] += made
        # Whoever sits where the particle is feels it, pinned or free: the free half dreams through the same membrane.
        for i in range(sheet.size):
            u = sheet[i]
            if s.seg_v[u] == at and s.seg[u, R] > 0:
                for _ in range(s.seg[u, R]):
                    if np.random.random() < k_sense:
                        s.seg[u, R] -= 1
                        s.seg[u, C] += 1
                        s.active[u] = 1
        for u in np.flatnonzero(s.active[:n_seg]):
            v = s.seg_v[u]
            for q in range(4):
                loc[q] = s.seg[u, q]
            loc[H] = _holes(s, v, cap)
            _fire(seg_table, loc, n_ch, cap)
            for q in range(4):
                s.seg[u, q] = loc[q]
            if s.seg_parent[u] < 0 and s.seg[u, E] > 0:
                s.cell_spiked[s.seg_cell[u]] = 1
            # Conduction to the parent and to the children.
            w = s.seg_parent[u]
            while True:
                if w >= 0:
                    for q in range(4):
                        hop[q] = s.seg[u, q]
                        hop[q + 4] = s.seg[w, q]
                    _fire(hop_table, hop, n_ch, n_ch)
                    for q in range(4):
                        s.seg[u, q] = hop[q]
                        s.seg[w, q] = hop[q + 4]
                    if hop[5] + hop[6] > 0:
                        s.active[w] = 1
                w = s.seg_first[u] if w == s.seg_parent[u] else s.seg_next[w]
                if w < 0:
                    break
            # Release onto outgoing contacts.
            e = s.seg_out[u]
            while e >= 0:
                if s.seg[u, E] > 0 and s.syn[e, Y0] > 0:
                    s.syn_active[e] = 1
                e = s.syn_next[e]
            if s.seg[u, C] + s.seg[u, E] == 0:
                s.active[u] = 0
        s.site_next[:] = 0
        for e in np.flatnonzero(s.syn_active[:n_syn]):
            a, b = s.syn_pre[e], s.syn_post[e]
            v = s.seg_v[b]
            _load_syn(syn, s, e, v, cap)
            for q in range(4):
                syn[PRE + q] = s.seg[a, q]
                syn[POST + q] = s.seg[b, q]
            sign = s.cell_sign[s.seg_cell[a]]
            if v < sites:
                syn[PFREE] = 1 if not held and at == v else 0
                syn[PPIN] = 1 if held and at == v else 0
                _fire(site_exc if sign > 0 else site_inh, syn, n_ch, cap)
                s.site_next[v] += sign * syn[Y1]
            else:
                syn[PFREE] = syn[PPIN] = 0
                _fire(exc_table if sign > 0 else inh_table, syn, n_ch, cap)
            for q in range(6):
                s.syn[e, q] = syn[q]
            for q in range(4):
                s.seg[a, q] = syn[PRE + q]
                s.seg[b, q] = syn[POST + q]
            for q in range(3):
                s.fld[v, q] = syn[VOX + q]
            s.active[b] = 1
            if syn[Y1] + syn[ZP] + syn[ZM] + syn[Z0] == 0:
                s.syn_active[e] = 0
        s.site_z[:] = s.site_next
        if tick % 2 == 0:
            _spread(s, cap, lattice, 1, .1)
            _spread(s, cap, lattice, 2, .1)
        if tick % 16 == 0:
            for _ in range(n_seg // 4):
                u = np.random.randint(0, n_seg)
                if s.seg[u, X] > 0:
                    v = s.seg_v[u]
                    for q in range(4):
                        loc[q] = s.seg[u, q]
                    loc[H] = _holes(s, v, cap)
                    _fire(seg_slow, loc, n_ch, cap)
                    for q in range(4):
                        s.seg[u, q] = loc[q]
            for _ in range(n_syn // 8):
                e = np.random.randint(0, n_syn)
                if s.syn_pre[e] < 0:
                    continue
                v = s.seg_v[s.syn_post[e]]
                _load_syn(syn, s, e, v, cap)
                syn[PFREE] = syn[PPIN] = 0
                _fire(syn_slow, syn, n_ch, cap)
                for q in range(6):
                    s.syn[e, q] = syn[q]
                s.fld[v, 0] = syn[VA]
        # Material hops between voxels, trading places with holes.
        if tick % 8 == 0:
            for v in range(s.fld.shape[0]):
                step = np.random.randint(0, 6)
                x, y, z = v % L, (v // L) % L, v // (L * L)
                x, y, z = x + (step == 0) - (step == 1), y + (step == 2) - (step == 3), z + (step == 4) - (step == 5)
                if x < 0 or x >= L or y < 0 or y >= L or z < 0 or z >= D:
                    continue
                t = x + L * y + L * L * z
                moves = min(_holes(s, t, cap), int(diffuse_material * s.fld[v, 0] + np.random.random()))
                s.fld[v, 0] -= moves
                s.fld[t, 0] += moves


@njit(cache=True)
def add_segment(s, cell, parent, voxel, kind, n_ch):
    """A tip advances: `n_ch` material quanta in the voxel become channels. -1 if short."""
    if s.fld[voxel, 0] < n_ch:
        return -1
    if s.life[0] > 0:
        s.life[0] -= 1
        u = s.free_seg[s.life[0]]
    elif s.counters[0] < s.seg.shape[0]:
        u = s.counters[0]
        s.counters[0] += 1
    else:
        return -1
    s.fld[voxel, 0] -= n_ch
    s.seg[u, R], s.seg[u, C], s.seg[u, E], s.seg[u, X] = n_ch, 0, 0, 0
    s.seg_v[u] = voxel
    s.seg_cell[u] = cell
    s.seg_parent[u] = parent
    s.seg_kind[u] = kind
    s.seg_out[u] = s.seg_in[u] = s.seg_first[u] = s.seg_next[u] = -1
    s.seg_born[u] = s.life[3]
    if parent >= 0:
        s.seg_next[u] = s.seg_first[parent]
        s.seg_first[parent] = u
    s.life[2] += 1
    return u


@njit(cache=True)
def add_synapse(s, pre, post, slots):
    """Contact in one voxel becomes a synapse; material pays for the slots."""
    v = s.seg_v[post]
    if s.fld[v, 0] < slots:
        return -1
    if s.life[1] > 0:
        s.life[1] -= 1
        e = s.free_syn[s.life[1]]
    elif s.counters[1] < s.syn.shape[0]:
        e = s.counters[1]
        s.counters[1] += 1
    else:
        return -1
    s.fld[v, 0] -= slots
    s.syn[e] = 0
    s.syn[e, Y0] = slots
    s.syn_pre[e] = pre
    s.syn_post[e] = post
    s.syn_next[e] = s.seg_out[pre]
    s.seg_out[pre] = e
    s.syn_in_next[e] = s.seg_in[post]
    s.seg_in[post] = e
    return e


@njit(cache=True)
def _unlink(head, nexts, index, item):
    """Remove `item` from the singly linked list that starts at `head[index]`."""
    if head[index] == item:
        head[index] = nexts[item]
        return
    x = head[index]
    while x >= 0:
        if nexts[x] == item:
            nexts[x] = nexts[item]
            return
        x = nexts[x]


@njit(cache=True)
def delete_synapse(s, e):
    _unlink(s.seg_out, s.syn_next, s.syn_pre[e], e)
    _unlink(s.seg_in, s.syn_in_next, s.syn_post[e], e)
    s.syn_pre[e] = s.syn_post[e] = -1
    s.syn_active[e] = 0
    s.free_syn[s.life[1]] = e
    s.life[1] += 1


@njit(cache=True)
def retract_segment(s, u, heads, links, n_ch):
    """A contactless end returns its channels to the voxel as material."""
    v, parent = s.seg_v[u], s.seg_parent[u]
    if parent >= 0:
        _unlink(s.seg_first, s.seg_next, parent, u)
    list_head = heads[v, 1 if s.seg_kind[u] in (1, 3) else 0]
    if list_head == u:
        heads[v, 1 if s.seg_kind[u] in (1, 3) else 0] = links[u]
    else:
        x = list_head
        while x >= 0:
            if links[x] == u:
                links[x] = links[u]
                break
            x = links[x]
    s.fld[v, 0] += s.seg[u, R] + s.seg[u, C] + s.seg[u, E] + s.seg[u, X]
    s.seg[u] = 0
    s.seg_cell[u] = -1
    s.seg_parent[u] = -1
    s.free_seg[s.life[0]] = u
    s.life[0] += 1
    s.life[2] += 1


@njit(cache=True)
def life_cycle(s, tips, count, heads, links, n_ch, visits, grace, sprout_slots, sprout):
    """One byte of structural life: empty contacts go, idle ends retract, strong contacts sprout."""
    n, m = s.counters[0], s.counters[1]
    s.life[3] += 1
    for _ in range(visits if m > 0 else 0):
        e = np.random.randint(0, m)
        if s.syn_pre[e] < 0:
            continue
        held = s.syn[e, Y0] + s.syn[e, Y1] + s.syn[e, Y2] + s.syn[e, ZP] + s.syn[e, ZM] + s.syn[e, Z0]
        if held == 0:
            delete_synapse(s, e)
        elif held >= sprout_slots and np.random.random() < sprout and count + 2 < tips.shape[0]:
            pre, post = s.syn_pre[e], s.syn_post[e]
            tips[count, 0], tips[count, 1], tips[count, 2], tips[count, 3], tips[count, 4], tips[count, 5] = \
                pre, s.seg_cell[pre], 1, 3, 0, np.random.randint(0, 4)
            count += 1
            tips[count, 0], tips[count, 1], tips[count, 2], tips[count, 3], tips[count, 4], tips[count, 5] = \
                post, s.seg_cell[post], 0, 2, 0, np.random.randint(0, 4)
            count += 1
    for _ in range(visits if n > 0 else 0):
        u = np.random.randint(0, n)
        if (s.seg_cell[u] >= 0 and s.seg_first[u] < 0 and s.seg_out[u] < 0 and s.seg_in[u] < 0 and s.active[u] == 0
                and s.seg[u, C] + s.seg[u, E] == 0 and s.life[3] - s.seg_born[u] > grace):
            retract_segment(s, u, heads, links, n_ch)
    return count


@njit(cache=True)
def _match(s, pre, post, match, match_ee, match_si, sheet):
    """The genome's class affinity between pre and post cell classes."""
    a, b = s.seg_cell[pre], s.seg_cell[post]
    if s.seg_kind[pre] == 3:
        return match if s.cell_sign[b] > 0 else match_si
    return match_ee if s.cell_sign[a] > 0 and s.cell_sign[b] > 0 and s.seg_v[post] >= sheet else match


@njit(cache=True)
def _keep(tips, k, u, cell, axon, left, side, heading):
    tips[k, 0], tips[k, 1], tips[k, 2], tips[k, 3], tips[k, 4], tips[k, 5] = u, cell, axon, left, side, heading


@njit(cache=True)
def grow_tips(s, tips, count, heads, links, L, D, n_ch, match, match_ee, match_si, slots, sense_slots, reserve):
    """Each tip takes one step. Kept tips stay in `tips`. `heads[v, 0]` lists dendrites, `[v, 1]` axons."""
    kept = 0
    for i in range(count):
        u, cell, axon, left, side, heading = tips[i, 0], tips[i, 1], tips[i, 2], tips[i, 3], tips[i, 4], tips[i, 5]
        if s.seg_cell[u] != cell:
            continue
        v = s.seg_v[u]
        x, y, z = v % L, (v // L) % L, v // (L * L)
        draw = np.random.random()
        step = np.random.randint(0, 4)
        nx, ny, nz = x, y, z
        lateral = False
        if axon:
            # Boundary cells' axons descend. Others run laterally, some after climbing to the sheet.
            if side == 2:
                if draw < .85:
                    nz = z + 1
                else:
                    lateral = True
            elif side == 1 and z > 0:
                if draw < .85:
                    nz = z - 1
                else:
                    lateral = True
            else:
                lateral = True
                step = heading if draw < .9 else step
        else:
            if draw < .4:
                nz = z + (1 if step % 2 == 0 else -1)
            else:
                lateral = True
        if lateral:
            if step == 0:
                nx = x + 1
            elif step == 1:
                nx = x - 1
            elif step == 2:
                ny = y + 1
            else:
                ny = y - 1
        if nx < 0 or nx >= L or ny < 0 or ny >= L or nz < 0 or nz >= D:
            if axon and lateral:
                heading = heading ^ 1
            left -= 1
            if left > 0:
                _keep(tips, kept, u, cell, axon, left, side, heading)
                kept += 1
            continue
        t = nx + L * ny + L * L * nz
        # A tip advances only into material to spare: what remains stays free for learning.
        if s.fld[t, 0] < reserve:
            continue
        w = add_segment(s, cell, u, t, (3 if side == 2 else 1) if axon else 0, n_ch)
        if w < 0:
            continue
        mine = 1 if axon else 0
        links[w] = heads[t, mine]
        heads[t, mine] = w
        seen = 0
        o = heads[t, 1 - mine]
        while o >= 0 and seen < 32:
            seen += 1
            pre, post = (w, o) if axon else (o, w)
            if s.seg_cell[o] != cell and np.random.random() < _match(s, pre, post, match, match_ee, match_si, L * L):
                add_synapse(s, pre, post, sense_slots if s.seg_kind[pre] == 3 else slots)
            o = links[o]
        left -= 1
        if left > 0:
            _keep(tips, kept, w, cell, axon, left, side, heading)
            kept += 1
    return kept


@dataclass
class Genome:
    """Per stage and hemilineage: sign, target depth change, axon and dendrite length, spread."""
    stages: int = 4
    n_ch: int = 8
    period: int = 400
    axon: tuple = (12, 14)
    dendrite: int = 4
    local_axon: int = 6
    match: float = .35
    match_ee: float = .05
    match_si: float = .35
    slots: int = 1
    sense_slots: int = 2
    divide_above: int = 150
    reserve: int = 120
    grace: int = 4000
    sprout_slots: int = 3
    sprout: float = .02


class Body:
    def __init__(self, L=16, D=8, cap=1024, material=300, genome=None, seed=1, max_seg=200_000, max_syn=500_000):
        self.L, self.D, self.cap = L, D, cap
        self.genome = genome or Genome()
        self.rng = np.random.default_rng(seed)
        V = L * L * D
        self.state = State(
            seg=np.zeros((max_seg, 4), np.int16), seg_v=np.zeros(max_seg, np.int32), seg_cell=np.full(max_seg, -1, np.int32),
            seg_parent=np.full(max_seg, -1, np.int32), seg_kind=np.zeros(max_seg, np.int8),
            seg_out=np.full(max_seg, -1, np.int32), seg_in=np.full(max_seg, -1, np.int32),
            seg_first=np.full(max_seg, -1, np.int32), seg_next=np.full(max_seg, -1, np.int32),
            seg_born=np.zeros(max_seg, np.int64), cell_sign=np.zeros(max_seg // 4, np.int8),
            cell_n=np.zeros(1, np.int64), fld=np.zeros((V, 3), np.int32), syn=np.zeros((max_syn, 6), np.int16),
            syn_pre=np.full(max_syn, -1, np.int32), syn_post=np.full(max_syn, -1, np.int32),
            syn_next=np.full(max_syn, -1, np.int32), syn_in_next=np.full(max_syn, -1, np.int32),
            active=np.zeros(max_seg, np.int8), syn_active=np.zeros(max_syn, np.int8), counters=np.zeros(4, np.int64),
            cell_spiked=np.zeros(max_seg // 4, np.int8), site_z=np.zeros(L * L, np.int32), site_next=np.zeros(L * L, np.int32),
            occupancy=np.zeros(L * L, np.int32), free_seg=np.zeros(max_seg, np.int64), free_syn=np.zeros(max_syn, np.int64),
            life=np.zeros(4, np.int64))
        self.state.fld[:, 0] = material
        self.state.counters[2] = self.rng.integers(L * L)
        self.seg_table, self.hop_table = seg_rows(), hop_rows()
        self.seg_slow, self.syn_slow = slow_seg_rows(), slow_syn_rows()
        self.exc_table, self.inh_table = syn_rows(1), syn_rows(-1, stay=.05)
        self.site_exc, self.site_inh = site_rows(1), site_rows(-1)
        self._tips = np.zeros((max_seg // 2, 6), np.int32)
        self._tip_count = 0
        self._heads = np.full((V, 2), -1, np.int32)
        self._links = np.full(max_seg, -1, np.int32)
        self.neuroblasts = [dict(x=x, y=y, stage=0, clock=0) for x in range(L) for y in range(L)]
        self.params = dict(k_sense=.25, emit=4, unit=.25, hops=256, diffuse_material=.2, visits=256)
        self.ticks_per_byte = 24
        self.byte_count = 0
        self._sheet, self._sheet_version = np.zeros(0, np.int64), -1

    def voxel(self, x, y, z):
        return int(x + self.L * y + self.L * self.L * z)

    def _birth(self, nb):
        """A division while the local material allows. Hemilineage A is excitatory, B inhibitory; stages keep cycling."""
        g = self.genome
        s = self.state
        stage = nb["stage"] % g.stages
        for side in (0, 1):
            depth = 0 if stage == 0 and side == 0 else max(1, min(self.D - 1, 1 + stage * 2 - side))
            jitter = (0, 1) if depth == 0 else (-1, 2)
            x = int(np.clip(nb["x"] + self.rng.integers(*jitter), 0, self.L - 1))
            y = int(np.clip(nb["y"] + self.rng.integers(*jitter), 0, self.L - 1))
            v = self.voxel(x, y, depth)
            cell = int(s.cell_n[0])
            if s.fld[v, 0] < g.divide_above or cell >= s.cell_sign.size:
                continue
            kind = 2 if depth == 0 else 0
            u = add_segment(s, cell, -1, v, kind, g.n_ch)
            if u < 0:
                continue
            s.cell_n[0] += 1
            s.cell_sign[cell] = 1 if side == 0 else -1
            self._links[u] = self._heads[v, 0]
            self._heads[v, 0] = u
            kind_side = 2 if kind == 2 else side
            tips = [(1, g.axon[side], kind_side), (0, g.dendrite, kind_side), (0, g.dendrite, kind_side)]
            if side == 1:
                tips.append((1, g.local_axon, 0))  # a second, local arbor: side 0 keeps it in the cell's own layer
            elif depth > 0:
                tips.append((1, g.axon[1], 1))  # excitatory cells also climb to the sheet
            for axon, left, tip_side in tips:
                self._tips[self._tip_count] = (u, cell, axon, left, tip_side, self.rng.integers(4))
                self._tip_count += 1

    def develop(self):
        """Growth and learning are one process: this runs at every byte for the whole life."""
        g = self.genome
        for nb in self.neuroblasts:
            nb["clock"] += 1
            if nb["clock"] >= g.period:
                nb["clock"] = 0
                self._birth(nb)
                nb["stage"] += 1
        self._tip_count = life_cycle(self.state, self._tips, self._tip_count, self._heads, self._links, g.n_ch,
                                     self.params["visits"], g.grace, g.sprout_slots, g.sprout)
        if self._tip_count:
            self._tip_count = grow_tips(self.state, self._tips, self._tip_count, self._heads, self._links, self.L, self.D,
                                        g.n_ch, g.match, g.match_ee, g.match_si, g.slots, g.sense_slots, g.reserve)

    def sheet(self):
        """Membrane sitting on the boundary sheet: whoever is here hears the particle land."""
        s = self.state
        if s.life[2] != self._sheet_version:
            n = int(s.counters[0])
            self._sheet = np.flatnonzero((s.seg_v[:n] < self.L * self.L) & (s.seg_cell[:n] >= 0))
            self._sheet_version = int(s.life[2])
        return self._sheet

    def _run(self, pin, record_from):
        p = self.params
        run_ticks(self.state, self.seg_table, self.hop_table, self.exc_table, self.inh_table, self.site_exc, self.site_inh,
                  self.seg_slow, self.syn_slow, pin, self.ticks_per_byte // 2, self.cap, self.genome.n_ch, p["k_sense"],
                  p["emit"], p["unit"], p["hops"], record_from, p["diffuse_material"], (self.L, self.D), self.sheet())

    def free(self):
        """The undriven half. Returns where the particle stayed, the body's own expression of what comes next."""
        self.state.occupancy[:] = 0
        self._run(-1, self.ticks_per_byte // 4)
        return self.state.occupancy.copy()

    def arrive(self, value):
        """The world pins the particle on `value`; the body lives on."""
        self._run(int(value) % (self.L * self.L), 0)
        self.byte_count += 1
        self.develop()

    def quanta(self):
        """Total content of the closed body."""
        s = self.state
        n, m = int(s.counters[0]), int(s.counters[1])
        return int(s.seg[:n].astype(np.int64).sum() + s.fld.astype(np.int64).sum() + s.syn[:m].astype(np.int64).sum())
