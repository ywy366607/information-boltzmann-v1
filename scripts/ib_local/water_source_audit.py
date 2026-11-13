"""Two fixed factual A_log directions versus the existing shared K2 code.

The reference is exact ONLY for the held-input local S leg. It is a diagnostic,
not a learning rule, all-network derivative, or independent synaptic trace.
"""
import math

import numpy as np

from scripts.ib_local.native_s_eligibility import EPS, NativeSEligibility


def stable_norm(values):
    """Avoid squared-norm underflow for genuinely tiny transported sources."""
    scale = float(np.max(np.abs(values)))
    return 0. if scale == 0 else scale * float(np.linalg.norm(values / scale))


class WaterSourceAudit(NativeSEligibility):
    def __init__(self, *args, selected=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.audit_start = None
        self.selected = ([(layer, 0) for layer in self.specs] if selected is None else list(selected))
        if not self.selected or any(key not in self.layout.head_fields for key in self.selected):
            raise ValueError('selected directions must be actual layer/head pairs')
        self.owner_indices = [self.parameter_slices[f'layer{layer}.A_log'].start + head
                              for layer, head in self.selected]
        self.first_coefficients = np.zeros((self.rank, len(self.selected)), np.float64)
        self.exact = [np.zeros(self.layout.size, np.float64) for _ in self.selected]
        self.first_norms = None
        self.log_alpha = {key: 0. for key in self.layout.head_fields}
        self.drops = 0
        self.snapshots = []

    def open_window(self, start_seen, parameter_version, initial_s):
        super().open_window(start_seen, parameter_version, initial_s)
        if self.audit_start is None:
            self.audit_start = start_seen

    def _insert_source(self, source, owner):
        receipt = super()._insert_source(source, owner)
        active = self.pending['start_seen'] == self.audit_start
        selected_owner = owner[self.owner_indices] if active else np.zeros(len(self.selected))
        for row, coefficient in enumerate(receipt['coefficients']):
            if row == receipt['drop']:
                self.first_coefficients[row] = selected_owner
            else:
                self.first_coefficients[row] += coefficient * selected_owner
        self.drops += receipt['drop'] is not None
        return receipt

    def _advance_head(self, layer, head, packet, tick, primal, owner):
        key = (layer, head)
        field = self.layout.head_fields[key]
        sl = self.layout.slices[field]
        shape = self.layout.shapes[field]
        raw = packet['k'][tick, head].astype(np.float64)
        denominator = (np.linalg.norm(raw) + EPS if packet['branch'] == 'recurrent'
                       else np.sqrt(float(raw @ raw) + EPS))
        k = raw / denominator
        beta, g = float(packet['beta'][tick, head]), float(packet['g'][tick, head])
        alpha = math.exp(g)
        # The tag sees all physical transports, including those of other heads.
        for direction, source_field in zip(self.exact, self.selected):
            # Held-input direct transport never changes the source's head.
            # Skip zero fields without changing the conditional reference.
            if source_field != key:
                continue
            d = direction[sl].reshape(shape)
            direction[sl] = (alpha * (d - beta * np.outer(k, k @ d))).reshape(-1)
        if key in self.selected and self.pending['start_seen'] == self.audit_start:
            selected = self.selected.index(key)
            s = primal[layer][head]
            r = s - beta * np.outer(k, k @ s)
            self.exact[selected][sl] += (alpha * g * r).reshape(-1)
        self.log_alpha[key] += g
        return super()._advance_head(layer, head, packet, tick, primal, owner)

    def advance_window(self):
        report = super().advance_window()
        age = self.clock['seen'] - self.audit_start
        if self.first_norms is None:
            self.first_norms = [stable_norm(d) for d in self.exact]
        directions = []
        for i, ((layer, head), exact) in enumerate(zip(self.selected, self.exact)):
            approximate = self.first_coefficients[:, i] @ self.code.U.astype(np.float64)
            norm = stable_norm(exact)
            approx_norm = stable_norm(approximate)
            delta = stable_norm(exact - approximate)
            denominator = norm * approx_norm
            directions.append(dict(layer=layer + 1, head=head, parameter='A_log',
                source_window_start=self.audit_start, source_window_stop_exclusive=self.audit_start + 128,
                exact_norm=norm, compressed_norm=approx_norm,
                retention_from_128=norm / self.first_norms[i] if self.first_norms[i] else None,
                relative_error=delta / norm if norm else None,
                cosine=float(exact @ approximate / denominator) if denominator else None,
                first_source_coefficient_norm=float(np.linalg.norm(self.first_coefficients[:, i]))))
        snapshot = dict(tokens=age, strict_old_all_sources=age >= 256,
            oldest_source_age=age, youngest_source_age=age - 127,
            directions=directions, basis_drops=self.drops,
            cumulative_log_alpha={f'L{l+1}H{h}': v for (l, h), v in self.log_alpha.items()},
            reconstruction_delta=report['reconstruction_delta'],
            scope='held-input local S only; two first-window directions; no optimizer update')
        self.snapshots.append(snapshot)
        return snapshot

    def contract_first_window(self, teaching):
        """Current real CE cotangent; no norm-only inference about usefulness."""
        flat = np.zeros(self.layout.size, np.float64)
        for (layer, head), name in self.layout.head_fields.items():
            flat[self.layout.slices[name]] = teaching[layer][head].reshape(-1)
        basis = self.code.U.astype(np.float64) @ flat
        compressed = basis @ self.first_coefficients
        exact = [float(flat @ d) for d in self.exact]
        return dict(teaching_norm=stable_norm(flat),
                    exact_first_window_credit=exact,
                    compressed_first_window_credit=compressed.tolist(),
                    cauchy_bounds=[stable_norm(flat) * stable_norm(d) for d in self.exact])
