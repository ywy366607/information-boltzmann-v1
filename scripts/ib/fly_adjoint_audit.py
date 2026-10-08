"""Observe full state adjoints without changing tensors or derivatives."""

import numpy as np

from scripts.ib.fly_checkpoint_audit import CheckpointSpikeAudit


class FlyAdjointAudit(CheckpointSpikeAudit):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.ticks = {}
        self.cpu_activity_bytes = 0

    def __enter__(self):
        # Reuse segment contexts without retaining voltage snapshots.
        super().__enter__()
        self._finish_tick = self.model.finish_coba_tick
        def finish_tick(context, drive, *, return_biophysics=False):
            outputs, bio = self._finish_tick(context, drive, return_biophysics=True)
            if self.active is not None and self.active[1] == 'original':
                segment, _, next_tick = self.active
                key = (segment, next_tick - 1)
                active = outputs[1].detach().cpu().numpy().astype(bool)
                self.cpu_activity_bytes += active.nbytes
                v = bio['v_pre'].detach().cpu().numpy()
                threshold = bio['eff_threshold'].detach().cpu().numpy()
                alpha = bio['alpha_eff'].detach().cpu().numpy()
                margin = v - threshold
                if self.model.surrogate_mode == 'threshold':
                    margin = margin / context['thresholds'].detach().cpu().numpy()
                psi = 1 / (1 + (np.pi * margin)**2)
                self.ticks[key] = {
                    'segment': segment, 'tick': next_tick - 1,
                    'spiking_fraction': float(active.mean()),
                    'surrogate_derivative_mean': float(psi.mean()),
                    'surrogate_derivative_on_silent_mean': float(psi[~active].mean()) if (~active).any() else None,
                    'membrane_alpha_max': float(alpha.max()),
                    'membrane_alpha_mean': float(alpha.mean()),
                    'adjoints': {},
                }
                channels = {'h': outputs[0], 'spike': outputs[1],
                            'pulse': outputs[2][0], 'ge': outputs[3], 'gi': outputs[4],
                            'b': outputs[5], 'x': outputs[6], 'u': outputs[7],
                            'v_pre': bio['v_pre']}
                for name, value in channels.items():
                    if value.requires_grad:
                        value.register_hook(self._gradient_hook(key, name, active))
            return (outputs, bio) if return_biophysics else outputs
        self.model.finish_coba_tick = finish_tick
        return self

    def record(self, voltage):
        if self.active is not None:
            self.active[2] += 1

    def _gradient_hook(self, key, name, active):
        def hook(gradient):
            values = gradient.detach().cpu().numpy().astype(np.float64)
            power = values * values
            total = float(power.sum())
            self.ticks[key]['adjoints'][name] = {
                'norm': total**.5,
                'max_abs': float(np.abs(values).max()),
                'nonzero_fraction': float(np.count_nonzero(values) / values.size),
                'power_on_silent_fraction': float(power[~active].sum()/total) if total else 0.,
                'finite': bool(np.isfinite(values).all()),
            }
            return gradient
        return hook

    def __exit__(self, *unused):
        self.model.finish_coba_tick = self._finish_tick
        return super().__exit__(*unused)

    def summary(self):
        rows = [self.ticks[key] for key in sorted(self.ticks)]
        return {
            'scope': 'Whole-window state adjoints from original forward hooks; no path pruned',
            'ticks': rows,
            'original_ticks': len(rows),
            'ticks_with_any_adjoint': sum(bool(row['adjoints']) for row in rows),
            'cpu_activity_bytes': self.cpu_activity_bytes,
            'extra_gpu_history_bytes': 0,
            'max_norm_by_channel': {
                name: max((row['adjoints'].get(name, {}).get('norm', 0) for row in rows), default=0)
                for name in ('h', 'ge', 'gi', 'b', 'x', 'u', 'spike', 'pulse', 'v_pre')},
            'segments': [{'segment': index, 'original_calls': segment['original_calls'],
                          'recompute_calls': segment['recompute_calls']}
                         for index, segment in enumerate(self.segments)],
        }
