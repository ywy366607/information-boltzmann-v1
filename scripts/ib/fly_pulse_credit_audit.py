"""Diagnostic backward-path intervention; never used by production training.

The two masks partition transmitted nonzero pulses, not the full recurrent
gradient: mixed active/silent paths are removed by both interventions.
"""

import numpy as np
import torch

from scripts.ib.fly_checkpoint_audit import CheckpointSpikeAudit


class PulseCreditAudit(CheckpointSpikeAudit):
    def __init__(self, model, path):
        super().__init__()
        if path not in ('active_only', 'silent_only'):
            raise ValueError('Choose active_only or silent_only')
        self.model = model
        self.path = path
        self.ticks = {}
        self.cpu_mask_bytes = 0

    def record(self, voltage):
        if self.active is not None:
            self.active[2] += 1

    def __enter__(self):
        super().__enter__()
        self._finish_tick = self.model.finish_coba_tick

        def finish_tick(context, drive, *, return_biophysics=False):
            outputs = self._finish_tick(context, drive, return_biophysics=return_biophysics)
            state = outputs[0] if return_biophysics else outputs
            if self.active is not None and self.active[1] == 'original':
                segment, _, next_tick = self.active
                key = (segment, next_tick - 1)
                pulse, spike = state[2][0], state[1]
                active = (pulse.detach() != 0).cpu().numpy()
                firing = (spike.detach() != 0).cpu().numpy()
                self.cpu_mask_bytes += active.nbytes
                self.ticks[key] = {
                    'segment': segment, 'tick': next_tick - 1,
                    'nonzero_pulse_fraction': float(active.mean()),
                    'spike_pulse_mask_disagreements': int(np.count_nonzero(active != firing)),
                    'pulse_requires_grad': pulse.requires_grad,
                    'hook_calls': 0,
                }
                if pulse.requires_grad:
                    pulse.register_hook(self._mask_hook(key, active))
                for name, value in (('h', state[0]), ('ge', state[3])):
                    if value.requires_grad:
                        value.register_hook(self._adjoint_hook(key, name))
            return outputs

        self.model.finish_coba_tick = finish_tick
        return self

    def _mask_hook(self, key, active):
        def hook(gradient):
            mask = torch.from_numpy(active).to(device=gradient.device)
            if self.path == 'silent_only':
                mask = ~mask
            self.ticks[key]['hook_calls'] += 1
            return gradient * mask
        return hook

    def _adjoint_hook(self, key, name):
        def hook(gradient):
            values = gradient.detach().cpu().numpy().astype(np.float64)
            self.ticks[key][name + '_adjoint_norm'] = float(np.sqrt(np.square(values).sum()))
            return gradient
        return hook

    def __exit__(self, *unused):
        self.model.finish_coba_tick = self._finish_tick
        return super().__exit__(*unused)

    def summary(self):
        rows = [self.ticks[key] for key in sorted(self.ticks)]
        return {
            'scope': 'Diagnostic pulse-route backward intervention; physical forward unchanged',
            'path': self.path,
            'initial_four_ring_slots': 'Preserved source state; not masked by future emission hooks',
            'nonadditivity': 'Both arms delete mixed active/silent recurrent paths; their gradients do not sum to all',
            'original_ticks': len(rows),
            'ticks_with_hook': sum(row['hook_calls'] > 0 for row in rows),
            'ticks_without_hook': sum(row['hook_calls'] == 0 for row in rows),
            'spike_pulse_mask_disagreements': sum(row['spike_pulse_mask_disagreements'] for row in rows),
            'cpu_mask_bytes': self.cpu_mask_bytes,
            'extra_gpu_history_bytes': 0,
            'ticks': rows,
        }
