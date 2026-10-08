"""Optional causal local-history filters; no physical evolution or implicit reset.

For held probe input r, integrate dz/dt=-(alpha+i*omega)z+alpha*r exactly.
State is explicit and differentiable. Fixed-size storage is constant in stream
length; credit still follows the caller's learning/gradient-boundary policy.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class TemporalProbeState:
    value: torch.Tensor  # complex [batch, probes, modes, channels]
    elapsed: torch.Tensor  # real [batch]

    def detach(self) -> "TemporalProbeState":
        return TemporalProbeState(self.value.detach(), self.elapsed.detach())

    def state_dict(self) -> dict:
        return {"value": self.value, "elapsed": self.elapsed}

    @classmethod
    def from_state_dict(cls, values: dict) -> "TemporalProbeState":
        return cls(values["value"], values["elapsed"])


class CausalProbeFilterBank(nn.Module):
    """Damped oscillatory measurements, sharing rates across probe channels.

    Rates/frequencies are explicit model-time quantities, not magic token counts.
    Persistent states describe previously executed rates/positions. Updating
    parameters never retroactively reinterprets the past. The production adapter
    follows sensor identities along their trajectories; a fixed-location history
    after sensor relocation would instead need an explicit transport contract.
    Constructor frequencies use inverse model time; the learned coordinate is
    omega*time_reference. A unit reference preserves historical parameters.
    """

    def __init__(self, probes: int, channels: int, rates, frequencies, *,
                 time_reference: float = 1.0):
        super().__init__()
        if min(probes, channels) < 1:
            raise ValueError("Positive probe and channel counts required")
        alpha = torch.as_tensor(rates, dtype=torch.get_default_dtype()).flatten()
        omega = torch.as_tensor(frequencies, dtype=alpha.dtype).flatten()
        if alpha.numel() < 1 or alpha.shape != omega.shape:
            raise ValueError("One finite frequency per positive rate required")
        if not bool((torch.isfinite(alpha) & (alpha > 0)).all()) or not bool(torch.isfinite(omega).all()):
            raise ValueError("Finite positive rates and finite frequencies required")
        if not math.isfinite(time_reference) or time_reference <= 0:
            raise ValueError("Finite positive frequency time reference required")
        reference = alpha.new_tensor(time_reference)
        if not bool(torch.isfinite(reference) & (reference > 0)):
            raise ValueError("Frequency time reference must be representable")
        self.probes, self.channels, self.modes = probes, channels, alpha.numel()
        self.log_rate = nn.Parameter(alpha.log())
        self.register_buffer("frequency_time_reference", reference)
        # Keep the parameter name/order for legacy unit-reference optimizers.
        # The learned coordinate is phase per reference interval, omega*t_ref.
        self.frequency = nn.Parameter(omega * reference)

    @property
    def physical_frequency(self) -> torch.Tensor:
        """Angular frequency in inverse model time; `frequency` is dimensionless."""
        return self.frequency / self.frequency_time_reference

    @torch.no_grad()
    def reparameterize_frequency(self, time_reference: float, *, optimizer=None) -> dict:
        """Explicit branch migration; preserve omega, history and Adam evidence.

        Call after loading the model, optimizer and any pending gradients in
        their saved coordinates. For theta_new=c*theta_old, moments of its
        gradient become m/c, v/c^2 and pending gradients become g/c. Adam step
        counts, learning rates and the other parameters are retained. Keeping
        the same learning rate intentionally changes future physical steps;
        this is a recorded coordinate change, not an exact legacy resume.
        """
        if not math.isfinite(time_reference) or time_reference <= 0:
            raise ValueError("Finite positive frequency time reference required")
        reference = self.frequency_time_reference.new_tensor(time_reference)
        ratio = reference / self.frequency_time_reference
        if not bool(torch.isfinite(reference) & (reference > 0)
                    & torch.isfinite(ratio) & (ratio > 0)):
            raise ValueError("Frequency coordinate ratio must be finite and representable")
        if optimizer is not None:
            if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
                raise TypeError("Frequency coordinate migration supports Adam/AdamW only")
            if not any(parameter is self.frequency for group in optimizer.param_groups
                       for parameter in group["params"]):
                raise ValueError("Optimizer does not own this frequency parameter")
        coordinate = self.frequency * ratio
        gradient = None if self.frequency.grad is None else self.frequency.grad / ratio
        replacements = {}
        state = {} if optimizer is None else optimizer.state.get(self.frequency, {})
        for key, power in (("exp_avg", 1), ("exp_avg_sq", 2), ("max_exp_avg_sq", 2)):
            if key in state:
                if state[key].shape != self.frequency.shape:
                    raise ValueError(f"Frequency optimizer state shape mismatch: {key}")
                value = state[key]
                for _ in range(power):
                    value = value / ratio.to(value)
                replacements[key] = value
        values = [coordinate, *replacements.values()]
        if gradient is not None:
            values.append(gradient)
        if any(not bool(torch.isfinite(value).all()) for value in values):
            raise ValueError("Frequency coordinate migration would create nonfinite values")
        previous = float(self.frequency_time_reference)
        self.frequency.copy_(coordinate)
        self.frequency_time_reference.copy_(reference)
        if gradient is not None:
            self.frequency.grad.copy_(gradient)
        for key, value in replacements.items():
            state[key].copy_(value)
        return {"migration": "explicit_frequency_coordinate_branch",
                "previous_time_reference": previous,
                "time_reference": float(reference),
                "coordinate_scale": float(ratio),
                "optimizer_moments_transformed": sorted(replacements),
                "pending_gradient_transformed": gradient is not None}

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Non-unit migration changes optimizer coordinates; callers must migrate
        # their optimizer state explicitly instead of reusing its old moments.
        reference_key, frequency_key = (prefix + "frequency_time_reference",
                                        prefix + "frequency")
        if reference_key not in state_dict:
            # Old checkpoints store omega itself. Convert only this coordinate;
            # unit-reference continuation also preserves the old Adam moments.
            reference = self.frequency_time_reference.detach().clone()
            if frequency_key in state_dict:
                state_dict[frequency_key] = (state_dict[frequency_key]
                                             * reference.to(state_dict[frequency_key]))
            state_dict[reference_key] = reference
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def initial_state(self, batch: int = 1) -> TemporalProbeState:
        if batch < 1:
            raise ValueError("Positive batch required")
        real = self.log_rate
        dtype = torch.complex128 if real.dtype == torch.float64 else torch.complex64
        return TemporalProbeState(torch.zeros(batch, self.probes, self.modes, self.channels,
                                             dtype=dtype, device=real.device),
                                  torch.zeros(batch, dtype=torch.float64, device=real.device))

    def forward(self, signal: torch.Tensor, state: TemporalProbeState,
                duration: float | torch.Tensor) -> TemporalProbeState:
        if signal.ndim != 3 or signal.shape[1:] != (self.probes, self.channels):
            raise ValueError("Probe signal must have shape [batch, probes, channels]")
        if signal.dtype != self.log_rate.dtype or signal.device != self.log_rate.device:
            raise ValueError("Signal and filter parameters must share real dtype/device")
        expected = (signal.shape[0], self.probes, self.modes, self.channels)
        expected_dtype = torch.complex128 if signal.dtype == torch.float64 else torch.complex64
        if state.value.shape != expected or state.value.dtype != expected_dtype or state.value.device != signal.device:
            raise ValueError("Filter state shape/dtype/device mismatch")
        if state.elapsed.shape != (signal.shape[0],) or state.elapsed.dtype != torch.float64 or state.elapsed.device != signal.device:
            raise ValueError("Filter elapsed shape/dtype/device mismatch")
        torch._assert_async(torch.isfinite(signal).all(), "Finite probe signal required")
        torch._assert_async(torch.isfinite(state.value).all(), "Finite filter history required")
        torch._assert_async((torch.isfinite(state.elapsed) & (state.elapsed >= 0)).all(),
                            "Finite nonnegative elapsed time required")
        dt = torch.as_tensor(duration, dtype=signal.dtype, device=signal.device)
        clock_dt = torch.as_tensor(duration, dtype=torch.float64, device=signal.device)
        if dt.numel() not in (1, signal.shape[0]):
            raise ValueError("Duration must be scalar or one per batch")
        # Scalar/tensor validation is kept out of runtime scalar transfers.
        if isinstance(duration, (int, float)) and (not math.isfinite(duration) or duration < 0):
            raise ValueError("Finite nonnegative duration required")
        if isinstance(duration, torch.Tensor):
            torch._assert_async((torch.isfinite(dt) & (dt >= 0)).all(),
                                "Finite nonnegative duration required")
        dt = dt.reshape(-1, 1, 1, 1)
        alpha = self.log_rate.exp()[None, None, :, None]
        torch._assert_async((torch.isfinite(alpha) & (alpha > 0)).all(),
                            "Finite positive representable rate required")
        reference = self.frequency_time_reference
        torch._assert_async(torch.isfinite(reference) & (reference > 0),
                            "Finite positive frequency time reference required")
        omega = self.physical_frequency
        torch._assert_async(torch.isfinite(omega).all(), "Finite frequency required")
        lam = torch.complex(alpha, omega[None, None, :, None])
        exponent = -lam * dt
        torch._assert_async(torch.isfinite(exponent).all(), "Representable rate-duration product required")
        decay = exponent.exp()
        # expm1 avoids subtraction cancellation at short elapsed intervals.
        gain = -alpha * torch.expm1(exponent) / lam
        value = decay * state.value + gain * signal[:, :, None, :]
        return TemporalProbeState(value, state.elapsed + clock_dt.flatten())

    @staticmethod
    def features(state: TemporalProbeState) -> torch.Tensor:
        """[batch, probes, modes, channels, signed real/imag] — keep phase."""
        return torch.view_as_real(state.value)


def sample_compact_probes(reader, field: torch.Tensor) -> torch.Tensor:
    """Signed measurements before head mixing; same existing finite footprints.

    Instantaneous dependence is on this aperture only. Propagating medium state
    and prior probe history can carry information that originated elsewhere.
    """
    if reader.aperture_type != "compact_probes":
        raise ValueError("Temporal measurements require compact spatial probes")
    flat = field.reshape(field.shape[0], reader.nodes, reader.d)
    # Only this linear signed measurement commutes with the fixed channel
    # basis. Pointwise second moments/normalization in the reader stay local.
    measured = torch.einsum("pn,bnd->bpd", reader.weights().to(flat), flat)
    return reader.physical_coordinates(measured)


class TemporalProbeReadout(nn.Module):
    """Learned signed history mixing over existing compact sensor identities.

    A history follows its probe as its coordinate changes: it is a trajectory
    measurement, never reinterpreted as history at the probe's new location.
    Frequencies and rates learn jointly with the medium, in model-time units.
    """

    def __init__(self, probes: int, channels: int, rates, frequencies, *,
                 time_reference: float = 1.0):
        super().__init__()
        self.bank = CausalProbeFilterBank(probes, 2 * channels, rates, frequencies,
                                         time_reference=time_reference)
        self.mode_mix = nn.Parameter(torch.randn(probes, self.bank.modes, 2)
                                     / math.sqrt(2 * probes * self.bank.modes))
        self.projection = nn.Linear(2 * channels, channels, bias=False)

    def forward(self, state: TemporalProbeState) -> torch.Tensor:
        features = self.bank.features(state)
        mixed = (features * self.mode_mix[None, :, :, None, :]).sum((1, 2, 4))
        return self.projection(mixed)
