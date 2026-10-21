"""CGB medium: the core equations of docs/information_boltzmann/CGB_CORE_EQUATIONS_20261020.md (V2, CPU, small scale).

Every state variable is a conserved pool or a bounded occupancy; every operation is done by the physics named in the
document (charge conservation, driving force, pools with finite replenishment, mass action, saturation). Nothing here
fixes a firing rate, clips a weight or normalises a vector. Units: time in ms, potentials normalised (rest 0,
threshold 1). Plain Euler steps of dt; the pools' exponential decays are applied exactly per step.

Equations (numbering of the document):
  E1  C du/dt = -gL (u - EL) + sum_j g_ij o_ij (E_s(j) - u) + I
  E2  spike when u > theta / r; then u <- u_reset, r <- (1 - kappa_r) r;  dr/dt = (1 - r) / tau_r
  E3  on a presynaptic spike: R = U v, v <- v - R;  dv/dt = (1 - v) / tau_v
  E4  o <- o + R (1 - o); do/dt = -o / tau_o;  dc_ij/dt = kN o B(u_post) + kb s_post - c / tau_c
  E5  dK/dt = aK hK(c)(1 - K) - K / tauK;  dP/dt = aP hP(c)(1 - P) - P / tauP;  dz/dt = (K - P) - m z - z / tau_z
  E6  dg/dt = eta m [z]+ (1 - g/gmax) - eta m [z]- g/gmax - lambda g
  E8  inhibitory synapses (Vogels 2011): dg/dt = eta_i (x_pre * s_post + s_pre * (x_post - 2 rho0 tau_x)) with traces x
  E9  trophic: release Q_i = q T_i [o_bar_i - eps]+ [cs_i - cf_i]+ ; uptake by active terminals; lambda = lambda0/(1+T/T*)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy

import numpy as np
import scipy.sparse as sp

from .structural_budget import StructuralBudget
from .support_transport import SupportTransport, SupportTransportParams


@dataclass
class CellClass:
    C: float = 1.0
    gL: float = 0.05            # 1/ms: membrane time constant C/gL = 20 ms
    EL: float = 0.0
    theta: float = 1.0
    u_reset: float = 0.0
    kappa_r: float = 0.95       # share of the sodium pool inactivated by one spike (an action potential inactivates most)
    tau_r: float = 5.0          # recovery of the sodium pool (ms)
    E_out: float = 5.0          # reversal of the transmitter this cell releases (excitatory 5, shunting ~ -0.3)
    U: float = 0.4              # release probability (share of the vesicle pool per spike)
    tau_v: float = 300.0        # vesicle replenishment (ms)
    inhibitory_plastic: bool = False


@dataclass
class SynapseParams:
    tau_o: float = 5.0          # transmitter occupancy (fast receptors)
    tau_nmda: float = 50.0      # NMDA occupancy (slow; the coincidence detector)
    kN: float = 1.0             # calcium per NMDA occupancy at full unblock
    kb: float = 0.6             # calcium per back-propagating spike
    u_half: float = 0.6         # magnesium unblock: B(u) = 1 / (1 + exp(-(u - u_half) / u_slope))
    u_slope: float = 0.15
    bap_amp: float = 3.0        # dendritic depolarisation of a back-propagating spike (the spike itself, seen by NMDA)
    tau_bap: float = 3.0
    tau_c: float = 20.0
    aK: float = 0.05; thK: float = 1.2; nK: float = 4.0; tauK: float = 200.0     # kinase: high threshold, steep
    aP: float = 0.05; thP: float = 0.5; nP: float = 2.0; tauP: float = 200.0     # phosphatase: lower threshold
    tau_z: float = 2000.0
    slide_A0: float = 0.0       # BCM sliding threshold: thK x (1 + A/A0), A = the cell's slow mean rate (Hz); 0 = off
    slide_tau: float = 5000.0   # the slow pool behind it (ms)
    m0: float = 1e-3            # tonic third factor (receptors always see some modulator)
    eta: float = 0.02
    gmax: float = 2.0
    lambda0: float = 0.0        # structural turnover without trophic support (1/ms)


@dataclass
class TrophicParams:
    """E9: release by a cell whose activity just fell while its input persists (residual explained), uptake by the
    terminals active at that moment (mass action), consumption by structure, turnover slowed by support."""
    rule: str = 'residual'      # release: 'residual' (E9 first draft), 'flux' (follows firing), 'constitutive' (q0)
    capture: str = 'activity'   # uptake by: 'activity' (presynaptic trace) or 'tag' (positive plasticity tag z, E5)
    q0: float = 1.0             # constitutive release (classic neurotrophic theory: a limited, steady supply)
    q: float = 1.0              # release per unit of (input present x activity fall)
    q_flux: float = 1.0
    eps: float = 0.0            # input occupancy below which nothing is released (stimulus gone)
    tau_cf: float = 50.0; tau_cs: float = 500.0          # fast / slow somatic calcium (ms)
    tau_N: float = 200.0        # trophic factor clearance near the cell
    kappa: float = 0.05         # uptake per unit terminal activity per unit factor (1/ms)
    tau_T: float = 20000.0      # terminal support decay
    phi: float = 0.0            # support consumed per unit structure
    lambda0: float = 1.0 / 20000.0                      # turnover without support (1/ms)
    T_star: float = 1.0


@dataclass
class Medium:
    """N cells with class indices, E edges (pre, post) with structure g; all pools as arrays."""
    classes: list
    cell_class: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    g: np.ndarray
    syn: SynapseParams = field(default_factory=SynapseParams)
    dt: float = 0.25
    seed: int = 0

    def __post_init__(self):
        # Own all mutable state; constructing a fork must not mutate its parent.
        self.cell_class = np.asarray(self.cell_class, dtype=np.int64).copy()
        self.pre = np.asarray(self.pre, dtype=np.int64).copy()
        self.post = np.asarray(self.post, dtype=np.int64).copy()
        self.g = np.asarray(self.g, dtype=np.float64).copy()
        N, E = len(self.cell_class), len(self.pre)
        self.N, self.E = N, E
        cc = [self.classes[k] for k in self.cell_class]
        col = lambda name: np.array([getattr(c, name) for c in cc], dtype=np.float64)
        for name in ('C', 'gL', 'EL', 'theta', 'u_reset', 'kappa_r', 'tau_r', 'E_out', 'U', 'tau_v'):
            setattr(self, name, col(name))
        self.inh_plastic_cell = np.array([c.inhibitory_plastic for c in cc])
        self.E_edge = self.E_out[self.pre]
        self.U_edge = self.U[self.pre]
        self.tau_v_edge = self.tau_v[self.pre]
        self.plastic_exc = ~self.inh_plastic_cell[self.pre]
        self.u = np.zeros(N); self.r = np.ones(N); self.spike = np.zeros(N, bool)
        self.v = np.ones(E); self.o = np.zeros(E); self.on = np.zeros(E); self.c = np.zeros(E)
        self.K = np.zeros(E); self.P = np.zeros(E); self.z = np.zeros(E)
        self.x_pre = np.zeros(E); self.x_post = np.zeros(N)       # E8 traces
        self.bap = np.zeros(N)                                    # dendritic depolarisation left by the spike
        self.A = np.zeros(N)                                      # slow mean rate (Hz), the sliding threshold's pool
        self.m = np.full(E, self.syn.m0)
        self.lam = np.full(E, self.syn.lambda0)
        self.to_post = sp.csr_matrix((np.ones(E), (self.post, np.arange(E))), shape=(N, E))
        self.t = 0.0
        self.rng = np.random.default_rng(self.seed)
        self.inh_eta = 0.0; self.inh_rho0 = 0.005; self.inh_tau = 20.0
        self.plastic = True
        self.trophic = None                     # TrophicParams to enable E9
        self.trophic_edges = np.zeros(E, bool)  # the terminals whose structure depends on support
        self.cf = np.zeros(N); self.cs = np.zeros(N); self.Nf = np.zeros(N); self.T = np.zeros(E)
        self.released = 0.0
        self.material = None                    # opt-in finite structural substrate
        self.support_transport = None           # opt-in conserved E9 reverse chain
        indeg = np.bincount(self.post, minlength=N).astype(float)
        self.indeg = np.maximum(indeg, 1.0)

    # -- one step -------------------------------------------------------------------------------------------------
    def step(self, I_ext=None, forced=None):
        s, dt = self.syn, self.dt
        if self.material is not None:
            self.material.advance(self.g, dt)
        # E1: charge conservation with driving force (sum, subtraction and shunting division for free)
        cond = self.g * self.o
        I_syn = self.to_post @ (cond * (self.E_edge - self.u[self.post]))
        I = I_syn - self.gL * (self.u - self.EL) + (0.0 if I_ext is None else I_ext)
        self.u = self.u + dt * I / self.C
        # E2: the sodium pool sets the effective threshold
        spk = self.u > self.theta / np.maximum(self.r, 1e-6)
        if forced is not None:
            spk = spk | forced
        self.u[spk] = self.u_reset[spk]
        self.r[spk] *= (1 - self.kappa_r[spk])
        self.r = 1 - (1 - self.r) * np.exp(-dt / self.tau_r)
        self.spike = spk
        # E3: vesicle pools (release on presynaptic spikes), then E4 occupancy jumps
        pre_spk = spk[self.pre]
        R = np.where(pre_spk, self.U_edge * self.v, 0.0)
        self.v = self.v - R
        self.v = 1 - (1 - self.v) * np.exp(-dt / self.tau_v_edge)
        self.o = (self.o + R * (1 - self.o)) * np.exp(-dt / s.tau_o)
        self.on = (self.on + R * (1 - self.on)) * np.exp(-dt / s.tau_nmda)
        if self.plastic:
            self._plasticity(R, pre_spk, spk)
        if self.trophic is not None:
            self._trophic(spk)
        self.t += dt
        return spk

    def _trophic(self, spk):
        p, dt = self.trophic, self.dt
        self.cf = self.cf * np.exp(-dt / p.tau_cf) + spk
        self.cs = self.cs * np.exp(-dt / p.tau_cs) + spk * (p.tau_cf / p.tau_cs)   # same mean as cf at steady rate
        if p.rule == 'constitutive':
            Q = np.full(self.N, p.q0)
        elif p.rule == 'residual':
            obar = (self.to_post @ self.o) / self.indeg                               # input still present
            Q = p.q * np.maximum(obar - p.eps, 0) * np.maximum(self.cs - self.cf, 0)   # activity just fell
        else:
            Q = p.q_flux * self.cf                                                    # follows recent firing
        if self.support_transport is not None:
            if p.rule != 'constitutive':
                raise ValueError('Conserved transport currently requires explicit constitutive boundary supply')
            carrier = np.maximum(self.z, 0.0) if p.capture == 'tag' else self.x_pre
            self.support_transport.advance(self.Nf, self.T, carrier, self.g,
                                           self.trophic_edges, dt, p)
            self.released += float(self.support_transport.source.sum()) * dt
            m = self.trophic_edges
            requested = self.g.copy()
            requested[m] *= np.exp(-dt * p.lambda0 / (1.0 + self.T[m] / p.T_star))
            self._set_structure(requested)
            return
        self.released += float(Q.sum()) * dt
        self.Nf = self.Nf * np.exp(-dt / p.tau_N) + dt * Q
        m = self.trophic_edges
        up = np.zeros(self.E)
        carrier = np.maximum(self.z, 0.0) if p.capture == 'tag' else self.x_pre      # tag capture or raw activity
        up[m] = p.kappa * carrier[m] * self.Nf[self.post[m]]                          # mass action uptake
        demand = np.bincount(self.post, up * dt, self.N)
        scale = np.minimum(1.0, self.Nf / np.maximum(demand, 1e-30))                   # cannot take more than is there
        up = up * scale[self.post]
        self.Nf = self.Nf - np.bincount(self.post, up * dt, self.N)
        self.T[m] = np.maximum(self.T[m] + dt * (up[m] - self.T[m] / p.tau_T - p.phi * self.g[m]), 0.0)
        lam = p.lambda0 / (1.0 + self.T[m] / p.T_star)
        requested = self.g.copy()
        requested[m] *= np.exp(-dt * lam)
        self._set_structure(requested)

    def _plasticity(self, R, pre_spk, spk):
        s, dt = self.syn, self.dt
        # the point model resets u at the spike; the dendrite still sees the spike's depolarisation for a few ms
        self.bap = self.bap * np.exp(-dt / s.tau_bap) + s.bap_amp * spk
        post_u = self.u[self.post] + self.bap[self.post]
        B = 1.0 / (1.0 + np.exp(-(post_u - s.u_half) / s.u_slope))
        # E4: calcium at the synapse, by mass action of NMDA occupancy and depolarisation, plus the back-propagating spike
        self.c = self.c * np.exp(-dt / s.tau_c) + dt * s.kN * self.on * B + s.kb * spk[self.post]
        # E5: two enzyme pools, bounded occupancies driven by calcium; the tag is their difference
        thK = s.thK
        if s.slide_A0 > 0:
            # a slow conserved pool of the cell's own activity raises its potentiation threshold (BCM)
            self.A = self.A * np.exp(-dt / s.slide_tau) + spk * (1000.0 / s.slide_tau)
            thK = s.thK * (1.0 + self.A[self.post] / s.slide_A0)
        hK = self.c ** s.nK / (self.c ** s.nK + thK ** s.nK)
        hP = self.c ** s.nP / (self.c ** s.nP + s.thP ** s.nP)
        self.K += dt * (s.aK * hK * (1 - self.K) - self.K / s.tauK)
        self.P += dt * (s.aP * hP * (1 - self.P) - self.P / s.tauP)
        e = self.plastic_exc
        self.z[e] += dt * ((self.K - self.P) - self.m * self.z - self.z / s.tau_z)[e]
        # E6: conversion of the tag into structure, bounded by the occupancy of finite slots; turnover
        zp, zm = np.maximum(self.z, 0), np.maximum(-self.z, 0)
        dg = s.eta * self.m * (zp * (1 - self.g / s.gmax) - zm * self.g / s.gmax) - self.lam * self.g
        requested = self.g.copy()
        if self.material is None:
            requested[e] = np.maximum(self.g[e] + dt * dg[e], 0.0)
        else:
            # Exact finite-slot update with frozen rates over this step. Large
            # tags cannot cause an Euler overshoot and then be hidden by clipping.
            if (self.m < 0).any():
                raise ValueError('Finite-slot mode currently requires nonnegative modulator occupancy')
            growth = s.eta * self.m * zp
            rate = (growth + s.eta * self.m * zm) / s.gmax + self.lam
            fraction = np.zeros(self.E)
            np.divide(-np.expm1(-dt * rate), rate, out=fraction, where=rate > 0)
            requested[e] = (self.g * np.exp(-dt * rate) + growth * fraction)[e]
        # E8: inhibitory plasticity balances excitation (Vogels 2011); its fixed point cancels the predictable input
        self.x_pre = self.x_pre * np.exp(-dt / self.inh_tau) + pre_spk
        self.x_post = self.x_post * np.exp(-dt / self.inh_tau) + spk
        if self.inh_eta:
            i = ~e
            alpha = 2 * self.inh_rho0 * self.inh_tau
            dgi = self.inh_eta * (self.x_pre * spk[self.post] + pre_spk * (self.x_post[self.post] - alpha))
            requested[i] = np.maximum(self.g[i] + dgi[i], 0.0)
        self._set_structure(requested)

    def _set_structure(self, requested):
        self.g = (requested if self.material is None else
                  self.material.transfer(self.g, requested))

    def enable_material(self, params=None, initial_free=0.0):
        if self.material is not None:
            raise ValueError('Material is already enabled; do not reset a living ledger')
        self.material = StructuralBudget(self.N, self.post, self.g, params, initial_free)

    def enable_support_transport(self, params=None, source=None):
        """Opt in without resetting captured support or the material ledger.

        `source` is one explicit boundary supply per cell, not per module.
        Omit it to inherit q0. Zero transport is the matched local control.
        Existing saved individuals remain on the historical path by default.
        """
        if self.support_transport is not None:
            raise ValueError('Support transport already enabled; do not reset a living ledger')
        if self.trophic is None or self.trophic.rule != 'constitutive':
            raise ValueError('Configure constitutive trophic supply before enabling transport')
        if self.trophic.capture not in ('tag', 'activity'):
            raise ValueError('Unknown trophic capture carrier')
        if (not np.isfinite(self.trophic.T_star) or self.trophic.T_star <= 0
                or not np.isfinite(self.trophic.lambda0) or self.trophic.lambda0 < 0):
            raise ValueError('Positive support scale and nonnegative turnover required')
        source = self.trophic.q0 if source is None else source
        self.support_transport = SupportTransport(
            self.N, self.pre, self.post, self.Nf, self.T, source,
            params or SupportTransportParams())

    _state_arrays = ('u', 'r', 'spike', 'v', 'o', 'on', 'c', 'K', 'P', 'z',
                     'x_pre', 'x_post', 'bap', 'A', 'm', 'lam', 'trophic_edges',
                     'cf', 'cs', 'Nf', 'T')
    _physiology_arrays = ('C', 'gL', 'EL', 'theta', 'u_reset', 'kappa_r', 'tau_r',
                          'E_out', 'U', 'tau_v', 'inh_plastic_cell', 'E_edge',
                          'U_edge', 'tau_v_edge', 'plastic_exc', 'indeg')

    def state_dict(self):
        """Complete persistent medium state, including slow pools and local RNG."""
        return {'schema': 1, 'classes': [asdict(c) for c in self.classes],
                'cell_class': self.cell_class.copy(), 'pre': self.pre.copy(),
                'post': self.post.copy(), 'g': self.g.copy(), 'syn': asdict(self.syn),
                'dt': self.dt, 'seed': self.seed, 't': self.t,
                'arrays': {name: getattr(self, name).copy()
                           for name in self._state_arrays + self._physiology_arrays},
                'inh_eta': self.inh_eta, 'inh_rho0': self.inh_rho0,
                'inh_tau': self.inh_tau, 'plastic': self.plastic, 'released': self.released,
                'trophic': None if self.trophic is None else asdict(self.trophic),
                'material': None if self.material is None else self.material.state_dict(),
                'support_transport': (None if self.support_transport is None else
                                      self.support_transport.state_dict()),
                'rng': copy.deepcopy(self.rng.bit_generator.state)}

    @classmethod
    def from_state_dict(cls, state):
        if state['schema'] != 1:
            raise ValueError('Unsupported medium state schema')
        obj = cls([CellClass(**c) for c in state['classes']], state['cell_class'],
                  state['pre'], state['post'], state['g'], SynapseParams(**state['syn']),
                  state['dt'], state['seed'])
        for name in cls._state_arrays + cls._physiology_arrays:
            if name in state['arrays']:
                setattr(obj, name, state['arrays'][name].copy())
        for name in ('t', 'inh_eta', 'inh_rho0', 'inh_tau', 'plastic', 'released'):
            setattr(obj, name, state[name])
        obj.trophic = None if state['trophic'] is None else TrophicParams(**state['trophic'])
        obj.material = (None if state['material'] is None else
                        StructuralBudget.from_state_dict(state['material']))
        transport = state.get('support_transport')
        if transport is not None:
            obj.support_transport = SupportTransport.from_state_dict(transport, obj.Nf, obj.T)
            if (not np.array_equal(obj.pre, obj.support_transport.pre)
                    or not np.array_equal(obj.post, obj.support_transport.post)):
                raise ValueError('Transport endpoints differ from medium wiring')
        obj.rng.bit_generator.state = copy.deepcopy(state['rng'])
        return obj
