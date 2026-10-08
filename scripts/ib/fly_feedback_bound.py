"""Sparse, conservative block certificate for one observed COBA trajectory.

This does not change physical dynamics. Blocks are h/ge/gi/b/x/u plus four
delay slots. Writer/observation states are outside this fixed-drive certificate.
"""

import numpy as np
import torch

from information_boltzmann.core.triton_synapse import execute_delayed_synaptic_transmission
from scripts.ib.fly_checkpoint_audit import CheckpointSpikeAudit


BLOCKS = ('h', 'ge', 'gi', 'b', 'x', 'u', 'p0', 'p1', 'p2', 'p3')


@torch.no_grad()
def transmission_rows(model, reference):
    """Absolute incoming row sums for each actual delay, without dense edges."""
    zeros, ones = torch.zeros_like(reference), torch.ones_like(reference)
    rows = []
    for kind in ('e', 'i'):
        by_delay = []
        for delay in range(4):
            ring = tuple(ones if slot == delay else zeros for slot in range(4))
            by_delay.append(execute_delayed_synaptic_transmission(
                ring, getattr(model, 'edge_pre_' + kind), getattr(model, 'edge_post_' + kind),
                getattr(model, 'edge_weight_' + kind).abs(), getattr(model, 'splits_' + kind)))
        rows.append(tuple(by_delay))
    return tuple(rows)


@torch.no_grad()
def tick_envelope(model, context, drive, outputs, rows):
    """Upper bounds of block induced infinity norms of A, B and C.

    Use the actual clamp derivatives, including their inclusive boundaries,
    and the detached reset convention. All coefficient/state units are retained.
    """
    if not model.detach_reset or not model.use_alif or not model.use_stp:
        raise ValueError('Certificate requires detached reset, COBA, ALIF and STP')
    h, k = context['h'], context['g_total']
    alpha, beta = context['alpha'], context['beta_int']
    le, li = context['leak_se'], context['leak_si']
    ge_gain, gi_gain = context['g_e'], context['g_i']
    rho_a, beta_a = context['alif_params']
    u0, rho_f, rho_r, norm = context['stp_params']
    x, u, s = context['x'], context['u'], outputs[1]
    v = alpha * h + beta * (context['base_current'] + drive)
    margin = v - context['eff_threshold']
    if model.surrogate_mode == 'threshold':
        margin = margin / context['thresholds'].detach()
    psi_max = (1 / (1 + (torch.pi * margin)**2)).max()
    da = -alpha * ((k >= 1e-5) & (k <= 20)).to(k.dtype)
    denominator = k.clamp_min(1e-5)
    db = -da / denominator - (1 - alpha) / denominator.square() * (k >= 1e-5)
    common = da * h + db * (context['base_current'] + drive)
    ve = (common + beta * model.E_E) * ge_gain
    vi = (common + beta * model.E_I) * gi_gain
    u_active = u + u0 * (1 - u) * s
    du_ds = u0 * (1 - u)
    du_du = 1 - u0 * s
    pulse_raw = (u_active * x / norm) * s
    unclipped = (pulse_raw <= 3).to(pulse_raw.dtype)
    zero = h.new_zeros(())
    a = [[zero for _ in BLOCKS] for _ in BLOCKS]
    b, c = [zero for _ in BLOCKS], [zero for _ in BLOCKS]
    maximum = lambda value: value.abs().max()
    c[0], c[1], c[2], c[3] = maximum(alpha), maximum(ve * le), maximum(vi * li), maximum(beta_a)
    a[0][0], a[0][1], a[0][2] = maximum((1-s)*alpha), maximum((1-s)*ve*le), maximum((1-s)*vi*li)
    a[1][1], a[2][2], a[3][3] = maximum(le), maximum(li), maximum(rho_a)
    a[4][4], a[4][5] = maximum(rho_r * (1-u_active*s)), maximum(rho_r*x*s*du_du)
    a[5][5] = maximum(rho_f * du_du)
    a[6][4], a[6][5] = maximum(unclipped*u_active*s/norm), maximum(unclipped*x*s*du_du/norm)
    b[3] = maximum(1-rho_a)
    b[4] = maximum(rho_r*x*(u_active+s*du_ds))
    b[5] = maximum(rho_f*du_ds)
    b[6] = maximum(unclipped*x/norm*(u_active+s*du_ds))
    for delay, (we, wi) in enumerate(zip(*rows)):
        incoming = ve.abs()*(1-le)*we + vi.abs()*(1-li)*wi
        c[6+delay] = maximum(incoming)
        a[0][6+delay] = maximum((1-s)*incoming)
        a[1][6+delay], a[2][6+delay] = maximum((1-le)*we), maximum((1-li)*wi)
    for delay in range(1, 4):
        a[6+delay][5+delay] = h.new_ones(())
    # One host transfer; never retain N by N Jacobians or extra GPU history.
    packed = torch.stack([v for row in a for v in row] + b + c + [psi_max]).double().cpu().numpy()
    return packed[:100].reshape(10, 10), packed[100:110], packed[110:120], float(packed[120])


def common_certificate(a, b, c, psi):
    """Common metric of the observed passive cascade, then maximum scalar chi.

    A fixed chi is a proposed surrogate-backward coefficient. It is not a
    physical damping term and does not supply a lifetime certificate.
    """
    result = {'passive_envelope': a.tolist(), 'event_envelope': b.tolist(),
              'margin_envelope': c.tolist(), 'max_surrogate_derivative': float(psi)}
    radius = float(np.max(np.abs(np.linalg.eigvals(a))))
    result['passive_envelope_spectral_radius'] = radius
    if radius >= 1:
        return {**result, 'feasible': False, 'reason': 'Passive block envelope is not strictly stable'}
    p = np.linalg.solve(np.eye(10) - a, np.ones(10))
    if not np.isfinite(p).all() or (p <= 0).any():
        return {**result, 'feasible': False, 'reason': 'Common metric is not positive and finite'}
    passive = (a @ p) / p
    feedback = b * float(psi) * float(c @ p) / p
    support = feedback > 0
    if (passive > 1).any() or ((passive >= 1) & support).any():
        return {**result, 'feasible': False, 'reason': 'Passive rows leave no positive event margin'}
    chi = min(1., float(np.min((1-passive[support])/feedback[support]))) if support.any() else 1.
    bound = passive + chi * feedback
    return {**result, 'feasible': True, 'inverse_metric_weights': p.tolist(),
            'metric_condition_number': float(p.max()/p.min()),
            'passive_weighted_row_sums': passive.tolist(),
            'event_weighted_row_sums': feedback.tolist(),
            'surrogate_coefficient': chi,
            'weighted_state_transition_bound': float(bound.max()),
            'scope': 'Observed fixed-drive 10N trajectory only; additional forcing/observation contracts required'}


class FlyFeedbackBound(CheckpointSpikeAudit):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.a, self.b, self.c = np.zeros((10, 10)), np.zeros(10), np.zeros(10)
        self.psi = 0.
        self.ticks = 0

    def record(self, voltage):
        if self.active is not None:
            self.active[2] += 1

    def __enter__(self):
        super().__enter__()
        self._finish_tick = self.model.finish_coba_tick
        self._rows = None

        def finish_tick(context, drive, *, return_biophysics=False):
            outputs = self._finish_tick(context, drive, return_biophysics=return_biophysics)
            state = outputs[0] if return_biophysics else outputs
            if self.active is not None and self.active[1] == 'original':
                if self._rows is None:
                    self._rows = transmission_rows(self.model, context['h'])
                a, b, c, psi = tick_envelope(self.model, context, drive, state, self._rows)
                self.a, self.b, self.c = np.maximum(self.a, a), np.maximum(self.b, b), np.maximum(self.c, c)
                self.psi = max(self.psi, psi)
                self.ticks += 1
            return outputs

        self.model.finish_coba_tick = finish_tick
        return self

    def __exit__(self, *unused):
        self.model.finish_coba_tick = self._finish_tick
        self._rows = None
        return super().__exit__(*unused)

    def summary(self):
        return {'original_ticks': self.ticks, 'blocks': BLOCKS,
                'certificate': common_certificate(self.a, self.b, self.c, self.psi)}
