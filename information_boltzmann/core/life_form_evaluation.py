"""Life-Form Tri-Pillar Biological Evaluation Suite & Thermodynamic Entropy Auditor.

Provides a unified, biologically grounded evaluation framework for open,
continuous-time living connectome networks:

Pillar 1: Non-Repeating Continuous Stream Prequential Tracker
  - Measures one-step-ahead surprise S(t) = -log P(x_{t+1} | h_t) on an infinite,
    non-repeating physical text stream.
  - Replaces traditional offline/frozen eval NLL; serves as the primary lifelong metric.

Pillar 2: Unified Plastic Environmental Shock & Re-adaptation Half-Life (t_{1/2})
  - Evaluates homeostatic resilience and cognitive elasticity when the continuum
    is abruptly severed (teleported into a novel domain).
  - CRITICAL: Invokes the EXACT same running StreamingConnectomeLearner instance.
    Synaptic plasticity and online adaptation REMAIN FULLY ACTIVE.
  - Measures relaxation trajectory and computes the token half-life t_{1/2}
    required to halve the sudden environmental shock.

Pillar 3: Unbroken Continuous Lifecycle Ebbinghaus Savings (A -> B -> A)
  - Re-experiences a previously learned sequence A after an unbroken, intervening
    physical lifetime stream B of N_intervene tokens.
  - Zero state resets, zero artificial zero-context simulations.
  - Compares initial learning curve L_A,1(t) vs review curve L_A,2(t).
  - Measures retention index and Savings Ratio (Ebbinghaus 1885):
      Savings = (NLL_initial - NLL_relearn) / NLL_initial
    quantifying latent engram reactivation vs catastrophic forgetting.

Thermodynamic Entropy & Noise Expulsion Auditor:
  - Measures entropy inflow S_in (observation uncertainty / Shannon surprise).
  - Measures physical dissipation S_diss (membrane conductance leaks, shunting
    inhibition phase-space contraction, and synaptic homeostasis).
  - Tracks net entropy accumulation Delta S = S_in - S_diss and representation
    spectral entropy H_spectral.
  - Quantifies high-frequency temporal noise power P_noise, coherent signal power
    P_signal, and the Signal-to-Noise Ratio (SNR) shift Delta SNR_dB, determining
    whether the living system accumulates noise or actively expels it.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PrequentialStreamTracker:
    """Tracks continuous one-step-ahead predictive surprise on non-repeating streams."""
    window_size: int = 500
    ema_alpha: float = 0.01

    total_tokens: int = 0
    cumulative_nll: float = 0.0
    ema_nll: float = 7.5
    recent_losses: List[float] = field(default_factory=list)

    def update(self, loss_val: float) -> None:
        """Records the surprise of predicting the next token before it is absorbed."""
        self.total_tokens += 1
        self.cumulative_nll += loss_val
        self.ema_nll = (1.0 - self.ema_alpha) * self.ema_nll + self.ema_alpha * loss_val

        self.recent_losses.append(loss_val)
        if len(self.recent_losses) > self.window_size:
            self.recent_losses.pop(0)

    @property
    def mean_cumulative_nll(self) -> float:
        return self.cumulative_nll / max(self.total_tokens, 1)

    @property
    def window_mean_nll(self) -> float:
        if not self.recent_losses:
            return self.ema_nll
        return float(np.mean(self.recent_losses))

    @property
    def window_std_nll(self) -> float:
        if len(self.recent_losses) < 2:
            return 0.0
        return float(np.std(self.recent_losses))

    def summary(self) -> Dict[str, float]:
        return {
            "total_tokens": float(self.total_tokens),
            "ema_nll": float(self.ema_nll),
            "mean_cumulative_nll": float(self.mean_cumulative_nll),
            "window_mean_nll": float(self.window_mean_nll),
            "window_std_nll": float(self.window_std_nll),
        }


@dataclass
class InformationLedger:
    """Rigorous information-theoretic ledger for streaming representation and surprise."""
    total_tokens: int
    cumulative_surprise_nats: float       # Sum of prequential NLL (nats)
    mean_surprise_nats: float              # Mean NLL (nats/token)
    effective_rank: float                  # Roy & Vetterli (2007) exp(H_spectral)
    spectral_entropy: float                # H_spectral = -sum p_i ln p_i
    max_spectral_entropy: float            # ln(d_model)
    spectral_entropy_ratio: float          # H_spectral / H_max
    participation_ratio: float             # (sum lambda_i)^2 / sum lambda_i^2
    effective_dimension_ratio: float       # R_eff / d_model
    spatial_variance: float                # E[||z - mean(z)||^2]
    temporal_roughness: float              # 0.5 * E[||z_{t+1} - z_t||^2] (true state velocity / roughness)
    state_autocorrelation_proxy: float     # 1 - temporal_roughness / max(spatial_variance, 1e-9)
    # Layer-by-layer de-meaned effective rank auditing & Net Contextual Gain
    rank_motor_eff: float = 0.0            # R_eff(h_motor) before readout projection
    rank_w_read_eff: float = 0.0           # R_eff(z_raw) after W_read projection
    rank_rmsnorm_eff: float = 0.0          # R_eff(z_norm) after RMSNorm
    pr_motor: float = 0.0                  # Participation Ratio of h_motor
    pr_w_read: float = 0.0                 # Participation Ratio of z_raw
    pr_rmsnorm: float = 0.0                # Participation Ratio of z_norm
    unigram_baseline_nll: float = 7.2723   # Static unigram frequency baseline
    net_contextual_gain: float = 0.0       # unigram_baseline_nll - mean_surprise


@dataclass
class EnergyLedger:
    """Biophysical metabolic and conductance dissipation ledger."""
    total_tokens: int
    cumulative_spikes: int                 # Total action potentials across network
    mean_spikes_per_token: float           # Action potentials per token
    mean_firing_rate_density: float        # Mean spikes / (N * token)
    cumulative_conductance: float          # Sum of mean(g_total) across time
    mean_conductance_per_token: float      # Mean total membrane conductance (Liouville phase contraction proxy)
    cumulative_adaptation_load: float      # Sum of mean(b) threshold shift
    mean_adaptation_offset: float          # Mean ALIF threshold offset
    information_efficiency_nats_per_100k_spikes: float # NLL surprise absorbed per 100k action potentials


@dataclass
class ThermodynamicSummary:
    """Quantitative summary of entropy balance and noise expulsion dynamics."""
    total_steps: int
    cumulative_entropy_inflow: float       # S_in = sum NLL (uncertainty entering system)
    cumulative_entropy_dissipated: float   # S_diss = sum S_dot_diss (phase contraction + leak)
    net_entropy_accumulated: float         # Delta S = S_in - S_diss
    entropy_balance_ratio: float           # S_diss / max(S_in, 1e-6)
    mean_inflow_rate: float                # Mean NLL per token
    mean_dissipation_rate: float           # Mean dissipation per token
    spectral_entropy: float                # Shannon entropy of representation singular spectrum
    max_possible_spectral_entropy: float   # ln(d_model) isotropic noise limit
    spectral_entropy_ratio: float          # H_spectral / H_max
    signal_power: float                    # Coherent representation variance P_signal
    noise_power: float                     # High-frequency stochastic fluctuation P_noise (temporal roughness)
    snr_db: float                          # 10 * log10(P_signal / P_noise)
    initial_snr_db: float                  # SNR during initial calibration window
    delta_snr_db: float                    # snr_db - initial_snr_db
    information: Optional[InformationLedger] = None
    energy: Optional[EnergyLedger] = None
    effective_rank: float = 0.0
    participation_ratio: float = 0.0
    temporal_roughness: float = 0.0
    spatial_variance: float = 0.0


class ThermodynamicEntropyAuditor:
    """Measures physical entropy inflow, dissipation, spectral entropy, and SNR shifts.

    Grounds open-system non-equilibrium thermodynamics in living neural dynamics:
      1. Information Ledger: surprise inflow (NLL), effective rank, spectral entropy, temporal roughness.
      2. Energy Ledger: spike counts, membrane conductance dissipation, adaptation load.
      3. True SNR & roughness: separates spatial manifold variance from temporal difference velocity.
    """

    def __init__(self, window_size: int = 300, d_model: int = 128, n_neurons: int = 165122):
        self.window_size = window_size
        self.d_model = d_model
        self.n_neurons = n_neurons
        self.total_steps = 0

        self.cumulative_inflow = 0.0
        self.cumulative_dissipated = 0.0
        self.cumulative_spikes = 0
        self.cumulative_conductance = 0.0
        self.cumulative_adaptation = 0.0

        self.recent_inflows: List[float] = []
        self.recent_dissipations: List[float] = []
        self.recent_z: List[torch.Tensor] = []  # Readout vectors [d_model]

        self.initial_snr_db: Optional[float] = None

    def record_step(
        self,
        nll: float,
        biophysics: Dict[str, Any],
        z_vector: torch.Tensor,
        weight_norm_sq: float = 0.0,
        alif_b: Optional[torch.Tensor] = None,
        weight_decay_lambda: float = 1e-4,
        spikes: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Audits entropy inflow, physical dissipation, and representation state."""
        self.total_steps += 1
        s_dot_in = float(nll)
        self.cumulative_inflow += s_dot_in

        # Phase-space volume contraction rate: sum_j g_total,j / tau_m
        g_total = biophysics.get("g_total", None)
        if isinstance(g_total, torch.Tensor):
            phase_diss = float(g_total.mean().item())
        else:
            phase_diss = 1.0
        self.cumulative_conductance += phase_diss

        # Synaptic homeostasis & weight regularization dissipation
        syn_diss = float(weight_decay_lambda * weight_norm_sq)
        if alif_b is not None and isinstance(alif_b, torch.Tensor):
            alif_diss = float(alif_b.mean().item())
        else:
            alif_diss = 0.0
        self.cumulative_adaptation += alif_diss

        if spikes is not None:
            if isinstance(spikes, torch.Tensor):
                self.cumulative_spikes += int(spikes.sum().item())
            elif isinstance(spikes, (int, float)):
                self.cumulative_spikes += int(spikes)

        s_dot_diss = phase_diss + syn_diss + alif_diss
        self.cumulative_dissipated += s_dot_diss

        self.recent_inflows.append(s_dot_in)
        self.recent_dissipations.append(s_dot_diss)
        if len(self.recent_inflows) > self.window_size:
            self.recent_inflows.pop(0)
            self.recent_dissipations.pop(0)

        # Store detached readout representation for SNR & spectral entropy
        z_detached = z_vector.detach().cpu().squeeze()
        if z_detached.dim() == 1:
            self.recent_z.append(z_detached)
            if len(self.recent_z) > self.window_size:
                self.recent_z.pop(0)

        # Calibrate initial SNR after first 30 steps
        if self.initial_snr_db is None and len(self.recent_z) >= min(30, self.window_size // 5):
            current_snr = self.compute_snr()
            self.initial_snr_db = current_snr["snr_db"]

        return {
            "s_dot_in": s_dot_in,
            "s_dot_diss": s_dot_diss,
            "delta_s_step": s_dot_in - s_dot_diss,
        }

    def compute_spectral_entropy(self) -> float:
        """Computes the Shannon spectral entropy of representation singular values."""
        h_spec, _, _ = self.compute_spectral_and_rank()
        return h_spec

    def compute_spectral_and_rank(self) -> tuple[float, float, float]:
        """Computes spectral entropy, effective rank, and participation ratio."""
        if len(self.recent_z) < 10:
            return 0.0, 1.0, 1.0
        Z = torch.stack(self.recent_z, dim=0)  # [W, d]
        Z_c = Z - Z.mean(dim=0, keepdim=True)
        try:
            svd_vals = torch.linalg.svdvals(Z_c)
            var_norm = (svd_vals ** 2)
            total_var = var_norm.sum().clamp_min(1e-12)
            p = (var_norm / total_var).clamp_min(1e-12)
            h_spec = float(-(p * torch.log(p)).sum().item())
            r_eff = float(math.exp(h_spec))
            pr = float(((total_var ** 2) / (var_norm ** 2).sum().clamp_min(1e-12)).item())
            return h_spec, r_eff, pr
        except Exception:
            return 0.0, 1.0, 1.0

    def compute_snr(self) -> Dict[str, float]:
        """Calculates signal power, noise power (temporal roughness), and SNR."""
        if len(self.recent_z) < 10:
            return {
                "signal_power": 0.0,
                "noise_power": 1e-6,
                "snr_db": 0.0,
                "temporal_roughness": 1e-6,
                "spatial_variance": 0.0
            }

        Z = torch.stack(self.recent_z, dim=0)  # [W, d]

        # Total representation energy (centered variance)
        z_mean = Z.mean(dim=0, keepdim=True)
        p_total = float(((Z - z_mean) ** 2).sum(dim=1).mean().item())

        # High-frequency stochastic temporal noise: von Neumann difference estimator
        # E[||z_{t+1} - z_t||^2] / 2: measures temporal state roughness / velocity
        diff = Z[1:] - Z[:-1]
        p_noise = float(0.5 * (diff ** 2).sum(dim=1).mean().item())
        p_noise = max(p_noise, 1e-9)

        p_signal = max(1e-9, p_total - p_noise)
        snr_linear = p_signal / p_noise
        snr_db = float(10.0 * math.log10(max(snr_linear, 1e-9)))

        return {
            "signal_power": p_signal,
            "noise_power": p_noise,
            "snr_db": snr_db,
            "temporal_roughness": p_noise,
            "spatial_variance": p_total,
        }

    def summary(self) -> ThermodynamicSummary:
        """Produces full thermodynamic report with separated information and energy ledgers."""
        snr_data = self.compute_snr()
        current_snr_db = snr_data["snr_db"]
        init_snr = self.initial_snr_db if self.initial_snr_db is not None else current_snr_db
        delta_snr = current_snr_db - init_snr

        h_spec, r_eff, pr = self.compute_spectral_and_rank()
        d = self.d_model
        h_max = math.log(d) if d > 0 else 1.0

        net_ent = self.cumulative_inflow - self.cumulative_dissipated
        balance_ratio = self.cumulative_dissipated / max(self.cumulative_inflow, 1e-6)

        steps = max(self.total_steps, 1)
        mean_surprise = self.cumulative_inflow / steps

        info_ledger = InformationLedger(
            total_tokens=self.total_steps,
            cumulative_surprise_nats=self.cumulative_inflow,
            mean_surprise_nats=mean_surprise,
            effective_rank=r_eff,
            spectral_entropy=h_spec,
            max_spectral_entropy=h_max,
            spectral_entropy_ratio=h_spec / max(h_max, 1e-6),
            participation_ratio=pr,
            effective_dimension_ratio=r_eff / max(d, 1),
            spatial_variance=snr_data["spatial_variance"],
            temporal_roughness=snr_data["temporal_roughness"],
            state_autocorrelation_proxy=1.0 - snr_data["temporal_roughness"] / max(snr_data["spatial_variance"], 1e-9),
        )

        mean_spikes = self.cumulative_spikes / steps
        density = mean_spikes / max(self.n_neurons, 1)
        mean_cond = self.cumulative_conductance / steps
        mean_adapt = self.cumulative_adaptation / steps
        nats_per_100k = (mean_surprise * 100000.0) / max(mean_spikes, 1.0)

        energy_ledger = EnergyLedger(
            total_tokens=self.total_steps,
            cumulative_spikes=self.cumulative_spikes,
            mean_spikes_per_token=mean_spikes,
            mean_firing_rate_density=density,
            cumulative_conductance=self.cumulative_conductance,
            mean_conductance_per_token=mean_cond,
            cumulative_adaptation_load=self.cumulative_adaptation,
            mean_adaptation_offset=mean_adapt,
            information_efficiency_nats_per_100k_spikes=nats_per_100k,
        )

        return ThermodynamicSummary(
            total_steps=self.total_steps,
            cumulative_entropy_inflow=self.cumulative_inflow,
            cumulative_entropy_dissipated=self.cumulative_dissipated,
            net_entropy_accumulated=net_ent,
            entropy_balance_ratio=balance_ratio,
            mean_inflow_rate=mean_surprise,
            mean_dissipation_rate=self.cumulative_dissipated / steps,
            spectral_entropy=h_spec,
            max_possible_spectral_entropy=h_max,
            spectral_entropy_ratio=h_spec / max(h_max, 1e-6),
            signal_power=snr_data["signal_power"],
            noise_power=snr_data["noise_power"],
            snr_db=current_snr_db,
            initial_snr_db=init_snr,
            delta_snr_db=delta_snr,
            information=info_ledger,
            energy=energy_ledger,
            effective_rank=r_eff,
            participation_ratio=pr,
            temporal_roughness=snr_data["temporal_roughness"],
            spatial_variance=snr_data["spatial_variance"],
        )


class StreamingConnectomeLearner:
    """Unified online streaming learner for open-ended connectome dynamics.

    Maintains active physical states (membrane voltage, 4-tier delay ring buffers,
    COBA conductances, ALIF adaptive thresholds, and STP depression/facilitation)
    and executes exact analytical O(1) streaming parameter adaptation.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        grad_accum_tokens: int = 4,
        train_decoder_weight: Optional[bool] = None,
        create_optimizer: bool = True,
    ):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_accum_tokens = grad_accum_tokens
        self.device = next(model.parameters()).device

        self.model.requires_grad_(False)

        # Trainable parameter list matching the true online organism
        if train_decoder_weight is not None:
            self.train_decoder_weight = train_decoder_weight
        else:
            self.train_decoder_weight = (model.decoder.weight is not model.embedding.weight)
        self.trainable_params = [
            model.output_read.weight,
        ]
        if hasattr(model, "read_norm"):
            self.trainable_params.append(model.read_norm.weight)
        if hasattr(model.decoder, "bias") and model.decoder.bias is not None:
            self.trainable_params.append(model.decoder.bias)
        self.trainable_params.extend([
            model.log_tau_m,
            model.log_threshold,
            model.log_beta,
            model.log_tau_a,
            model.logit_u0,
            model.log_tau_fac,
            model.log_tau_rec,
            model.log_g_e,
            model.log_g_i,
        ])
        if self.train_decoder_weight:
            self.trainable_params.append(model.decoder.weight)

        if optimizer is not None:
            self.optimizer = optimizer
        elif create_optimizer:
            self.optimizer = torch.optim.AdamW(
                self.trainable_params, lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            self.optimizer = None

        # Physical living states
        n_neurons = model.n_neurons
        self.h = torch.zeros(1, n_neurons, device=self.device)
        self.ring = tuple(torch.zeros(1, n_neurons, device=self.device) for _ in range(4))
        self.syn_state: Dict[str, torch.Tensor] = {
            "ge": torch.zeros(1, n_neurons, device=self.device),
            "gi": torch.zeros(1, n_neurons, device=self.device),
            "b": torch.zeros(1, n_neurons, device=self.device),
            "x": torch.ones(1, n_neurons, device=self.device),
            "u": model.get_stp_params()[0].clone().expand(1, n_neurons).contiguous(),
        }
        self.h_prev = torch.zeros_like(self.h)

        # Structural buffers & sliced readout mode
        self.use_sliced_readout = hasattr(model, "read_indices") and (getattr(model, "read_surface", "all") != "all")
        self.read_mask_1n = model.read_mask.unsqueeze(0).to(self.device)
        self.superclass_ids = model.superclass_id.to(self.device)
        self.pi_const = math.pi

        # Analytical gradient accumulation buffers
        self.grad_w_read_acc = torch.zeros_like(model.output_read.weight)
        self.grad_norm_acc = torch.zeros_like(model.read_norm.weight) if hasattr(model, "read_norm") else None
        self.grad_b_dec_acc = torch.zeros_like(model.decoder.bias) if hasattr(model.decoder, "bias") and model.decoder.bias is not None else None
        self.grad_w_dec_acc = torch.zeros_like(model.decoder.weight) if self.train_decoder_weight else None
        self.grad_thresh_acc = torch.zeros_like(model.log_threshold)
        self.grad_tau_m_acc = torch.zeros_like(model.log_tau_m)
        self.grad_beta_acc = torch.zeros_like(model.log_beta)
        self.accum_count = 0

        # Caching
        self.update_cached_params()

        self.tracker = PrequentialStreamTracker()
        d_model = getattr(model, "d_model", model.output_read.weight.shape[0])
        self.auditor = ThermodynamicEntropyAuditor(d_model=d_model, n_neurons=n_neurons)

    def update_cached_params(self) -> None:
        self.thresholds = self.model.get_thresholds()
        self.beta_a = self.model.get_alif_params()[1]
        self.theta_0 = torch.exp(self.model.log_threshold)
        self.beta_val = torch.exp(self.model.log_beta)

    def step(self, tok_in: torch.Tensor, tok_tgt: torch.Tensor, learn: bool = True) -> Dict[str, Any]:
        """Executes a single causal physical step and online adaptation."""
        # 1. Forward physical step with biophysics
        self.h_prev.copy_(self.h)
        step_res, biophysics = self.model.step(
            self.h, tok_in, spike_ring=self.ring, return_biophysics=True, **self.syn_state
        )
        self.h = step_res[0]
        spikes_next = step_res[1]
        self.ring = step_res[2]

        if self.model.synapse_model == "coba":
            self.syn_state["ge"] = step_res[3]
            self.syn_state["gi"] = step_res[4]
            idx = 5
        else:
            self.syn_state["i_syn"] = step_res[3]
            idx = 4
        if getattr(self.model, "use_alif", False):
            self.syn_state["b"] = step_res[idx]; idx += 1
        if getattr(self.model, "use_stp", False):
            self.syn_state["x"] = step_res[idx]
            self.syn_state["u"] = step_res[idx + 1]

        # 2. Prequential prediction & surprise
        if self.use_sliced_readout:
            readout_vec = self.h[:, self.model.read_indices]
        else:
            readout_vec = self.h * self.read_mask_1n
        z_raw = self.model.output_read(readout_vec)  # [1, d_model]
        if hasattr(self.model, "read_norm"):
            z_latent = self.model.read_norm(z_raw)
        else:
            z_latent = z_raw
        logits = self.model.decoder(z_latent)
        probs = F.softmax(logits, dim=-1)

        tgt_id = tok_tgt[0].item()
        p_target = probs[0, tgt_id].clamp_min(1e-9)
        loss_val = float(-torch.log(p_target).item())

        self.tracker.update(loss_val)

        # 3. Thermodynamic entropy auditing with true action potential spikes
        w_sq = float((self.model.output_read.weight ** 2).sum().item())
        step_entropy = self.auditor.record_step(
            nll=loss_val,
            biophysics=biophysics,
            z_vector=z_latent,
            weight_norm_sq=w_sq,
            alif_b=self.syn_state.get("b", None),
            weight_decay_lambda=self.weight_decay,
            spikes=spikes_next,
        )

        # 4. Online learning via analytical e-prop (Plasticity)
        if learn:
            probs_grad = probs.clone()
            probs_grad[0, tgt_id] -= 1.0  # e_logits
            error_norm = torch.matmul(probs_grad, self.model.decoder.weight)

            if hasattr(self.model, "read_norm"):
                rms = torch.sqrt(torch.mean(z_raw ** 2, dim=-1, keepdim=True) + 1e-6)
                gamma = self.model.read_norm.weight
                u_norm = error_norm * gamma
                z_hat = z_raw / rms
                proj = (u_norm * z_hat).sum(dim=-1, keepdim=True) / z_raw.shape[-1]
                error_read = (u_norm - z_hat * proj) / rms
                if self.grad_norm_acc is not None:
                    self.grad_norm_acc.add_((error_norm * z_hat).squeeze(0))
            else:
                error_read = error_norm

            if self.use_sliced_readout:
                self.grad_w_read_acc.addr_(error_read.squeeze(0), readout_vec.squeeze(0))
            else:
                self.grad_w_read_acc.add_(torch.matmul(error_read.t(), readout_vec))
            if self.grad_b_dec_acc is not None:
                self.grad_b_dec_acc.add_(probs_grad.squeeze(0))
            if self.grad_w_dec_acc is not None:
                self.grad_w_dec_acc.addr_(probs_grad.squeeze(0), z_latent.squeeze(0))

            v_pre = biophysics["v_pre"]
            eff_thresh = biophysics["eff_threshold"]
            v_diff = v_pre - eff_thresh
            pi_x = self.pi_const * v_diff
            psi = 1.0 / (1.0 + pi_x * pi_x)

            if self.use_sliced_readout:
                L_motor = torch.matmul(error_read, self.model.output_read.weight).squeeze(0)
                motor_indices = self.model.read_indices
                motor_superclasses = self.superclass_ids[motor_indices]
                L_motor_psi = L_motor * psi[0, motor_indices]
                self.grad_thresh_acc.index_add_(0, motor_superclasses, -L_motor_psi * self.theta_0[motor_superclasses])
                self.grad_tau_m_acc.index_add_(0, motor_superclasses, L_motor_psi * self.h_prev[0, motor_indices])
                self.grad_beta_acc.index_add_(
                    0, motor_superclasses,
                    -L_motor_psi * self.syn_state.get("b", torch.zeros_like(self.h))[0, motor_indices] * self.beta_val[motor_superclasses]
                )
            else:
                L_read = torch.matmul(error_read, self.model.output_read.weight) * self.read_mask_1n
                L_psi = (L_read * psi).squeeze(0)
                self.grad_thresh_acc.index_add_(0, self.superclass_ids, -L_psi * self.theta_0[self.superclass_ids])
                self.grad_tau_m_acc.index_add_(0, self.superclass_ids, L_psi * self.h_prev.squeeze(0))
                self.grad_beta_acc.index_add_(
                    0, self.superclass_ids,
                    -L_psi * self.syn_state.get("b", torch.zeros_like(self.h)).squeeze(0) * self.beta_val[self.superclass_ids]
                )

            self.accum_count += 1
            if self.accum_count >= self.grad_accum_tokens:
                if self.optimizer is not None:
                    scale = 1.0 / self.accum_count
                    self.model.output_read.weight.grad = self.grad_w_read_acc * scale
                    if self.grad_norm_acc is not None:
                        self.model.read_norm.weight.grad = self.grad_norm_acc * scale
                    if self.grad_b_dec_acc is not None:
                        self.model.decoder.bias.grad = self.grad_b_dec_acc * scale
                    if self.grad_w_dec_acc is not None:
                        self.model.decoder.weight.grad = self.grad_w_dec_acc * scale
                    self.model.log_threshold.grad = self.grad_thresh_acc.clamp(-5.0, 5.0) * scale
                    self.model.log_tau_m.grad = self.grad_tau_m_acc.clamp(-5.0, 5.0) * scale
                    self.model.log_beta.grad = self.grad_beta_acc.clamp(-5.0, 5.0) * scale

                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.update_cached_params()

                self.grad_w_read_acc.zero_()
                if self.grad_norm_acc is not None:
                    self.grad_norm_acc.zero_()
                if self.grad_b_dec_acc is not None:
                    self.grad_b_dec_acc.zero_()
                if self.grad_w_dec_acc is not None:
                    self.grad_w_dec_acc.zero_()
                self.grad_thresh_acc.zero_()
                self.grad_tau_m_acc.zero_()
                self.grad_beta_acc.zero_()
                self.accum_count = 0

        spike_cnt = float(spikes_next.sum().item())
        return {
            "loss": loss_val,
            "step_entropy": step_entropy,
            "biophysics": biophysics,
            "z_latent": z_latent,
            "spikes": spikes_next,
            "spike_count": spike_cnt,
        }

    def fork(
        self,
        train_decoder_weight: bool = False,
        create_optimizer: bool = True,
    ) -> StreamingConnectomeLearner:
        """Deep clones learner states for isolated live branching without VRAM bloat."""
        forked = StreamingConnectomeLearner(
            model=self.model,
            optimizer=None,
            lr=self.lr,
            weight_decay=self.weight_decay,
            grad_accum_tokens=self.grad_accum_tokens,
            train_decoder_weight=train_decoder_weight,
            create_optimizer=create_optimizer,
        )
        # Copy physical states
        forked.h = self.h.clone()
        forked.ring = tuple(r.clone() for r in self.ring)
        forked.syn_state = {k: v.clone() for k, v in self.syn_state.items()}
        forked.h_prev = self.h_prev.clone()
        forked.accum_count = self.accum_count
        forked.grad_w_read_acc = self.grad_w_read_acc.clone()
        forked.grad_norm_acc = self.grad_norm_acc.clone() if self.grad_norm_acc is not None else None
        forked.grad_b_dec_acc = self.grad_b_dec_acc.clone() if self.grad_b_dec_acc is not None else None
        forked.grad_w_dec_acc = self.grad_w_dec_acc.clone() if (self.grad_w_dec_acc is not None and train_decoder_weight) else None
        forked.grad_thresh_acc = self.grad_thresh_acc.clone()
        forked.grad_tau_m_acc = self.grad_tau_m_acc.clone()
        forked.grad_beta_acc = self.grad_beta_acc.clone()
        forked.update_cached_params()
        # Copy auditor snapshot
        forked.auditor = copy.deepcopy(self.auditor)
        return forked


@dataclass
class ReAdaptationResult:
    """Metrics capturing homeostatic recovery after an environmental shock."""
    n_tokens: int
    shock_surprise: float            # Initial surprise upon entering new stream (first K tokens)
    plateau_surprise: float          # Asymptotic surprise after adaptation (last K tokens)
    delta_shock: float               # Magnitude of shock absorbed (shock - plateau)
    half_life_tokens: float          # Tokens required to recover 50% of the shock (t_{1/2})
    elasticity_score: float          # delta_shock / (half_life_tokens + 1)
    trajectory: List[float]          # Token-by-token prequential surprise curve
    thermodynamics: ThermodynamicSummary
    generalization: Dict[str, Any] = field(default_factory=dict)


class EnvironmentalShockEvaluator:
    """Measures physical re-adaptation dynamics under abrupt environmental shifts.

    Maintains active synaptic plasticity and parameter learning by calling
    the exact same running StreamingConnectomeLearner instance.
    """

    @staticmethod
    def evaluate_shock(
        learner: StreamingConnectomeLearner,
        tokens: torch.Tensor,
    ) -> ReAdaptationResult:
        """Executes plastic transfer into a novel stream using the unified learner."""
        T = tokens.shape[0] - 1
        if T < 10:
            raise ValueError("Evaluation sequence must have at least 10 tokens")

        surprises: List[float] = []
        for t in range(T):
            tok_in = tokens[t:t + 1]
            tok_tgt = tokens[t + 1:t + 2]
            res = learner.step(tok_in, tok_tgt, learn=True)
            surprises.append(res["loss"])

        k_init = min(20, T // 5)
        k_end = min(50, T // 4)
        shock_surprise = float(np.mean(surprises[:k_init]))
        plateau_surprise = float(np.mean(surprises[-k_end:]))
        delta_shock = max(0.0, shock_surprise - plateau_surprise)

        # Half-life calculation: first token where surprise drops below 50% of the shock
        target_mid = shock_surprise - 0.5 * delta_shock
        half_life_tokens = float(T)
        for idx_step, s in enumerate(surprises):
            if s <= target_mid:
                half_life_tokens = float(idx_step)
                break

        elasticity_score = delta_shock / (half_life_tokens + 1.0)
        thermo_summary = learner.auditor.summary()
        from ..runtime.lifelong_evaluation import adaptation_generalization_summary
        generalization = adaptation_generalization_summary(
            surprises, block_tokens=max(1, k_init), hold_blocks=2)

        return ReAdaptationResult(
            n_tokens=T,
            shock_surprise=shock_surprise,
            plateau_surprise=plateau_surprise,
            delta_shock=delta_shock,
            half_life_tokens=half_life_tokens,
            elasticity_score=elasticity_score,
            trajectory=surprises,
            thermodynamics=thermo_summary,
            generalization=generalization,
        )

    # Legacy static method for backward compatibility
    @staticmethod
    def evaluate_plastic_readaptation(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        tokens: torch.Tensor,
        h_init: torch.Tensor,
        ring_init: Tuple[torch.Tensor, ...],
        syn_state_init: Dict[str, torch.Tensor],
        grad_accum_tokens: int = 4,
        lr: float = 3e-4,
    ) -> ReAdaptationResult:
        learner = StreamingConnectomeLearner(
            model=model, optimizer=optimizer, lr=lr, grad_accum_tokens=grad_accum_tokens
        )
        learner.h = h_init.clone()
        learner.ring = tuple(r.clone() for r in ring_init)
        learner.syn_state = {k: v.clone() for k, v in syn_state_init.items()}
        return EnvironmentalShockEvaluator.evaluate_shock(learner, tokens)


@dataclass
class SavingsResult:
    """Metrics capturing Ebbinghaus savings and relearning acceleration across genuine lifetime."""
    n_tokens: int
    intervening_tokens: int          # Actual physical tokens elapsed between initial and review
    nll_initial_mean: float          # Average surprise during initial encounter A_1
    nll_relearn_mean: float          # Average surprise during review encounter A_2
    initial_opening_nll: float       # Surprise on token 0 during first encounter
    relearn_opening_nll: float       # Surprise on token 0 during review (immediate recall)
    retention_immediate_drop: float  # initial_opening - relearn_opening
    savings_ratio: float             # (nll_initial - nll_relearn) / nll_initial
    acceleration_factor: float       # Speedup in reaching criterion
    initial_trajectory: List[float]
    relearn_trajectory: List[float]
    thermodynamic_shift: Dict[str, float]  # SNR and entropy changes across lifecycle


class ContinuousLifecycleSavingsEvaluator:
    """Evaluates memory retention and relearning acceleration across genuine physical lifetime.

    Executes true unbroken trajectory: Sequence A -> Intervening Stream B -> Sequence A.
    Zero state resets, zero artificial zero-context simulations.
    """

    @staticmethod
    def evaluate_unbroken_lifecycle(
        learner: StreamingConnectomeLearner,
        tokens_a: torch.Tensor,              # Sequence A [L_A + 1]
        tokens_b: torch.Tensor,              # Intervening stream B [N_intervene + 1]
    ) -> SavingsResult:
        """Executes the unbroken A -> B -> A experience with active plasticity throughout."""
        L_A = tokens_a.shape[0] - 1
        N_B = tokens_b.shape[0] - 1
        if L_A < 5:
            raise ValueError("Sequence A must have at least 5 tokens")

        init_thermo = learner.auditor.summary()

        # Phase 1: Initial encounter with Sequence A
        initial_trajectory: List[float] = []
        for t in range(L_A):
            tok_in = tokens_a[t:t + 1]
            tok_tgt = tokens_a[t + 1:t + 2]
            res = learner.step(tok_in, tok_tgt, learn=True)
            initial_trajectory.append(res["loss"])

        # Phase 2: Intervening physical experience stream B (unbroken continuity!)
        for t in range(N_B):
            tok_in = tokens_b[t:t + 1]
            tok_tgt = tokens_b[t + 1:t + 2]
            learner.step(tok_in, tok_tgt, learn=True)

        # Phase 3: Review / Re-encounter with Sequence A (unbroken continuity!)
        relearn_trajectory: List[float] = []
        for t in range(L_A):
            tok_in = tokens_a[t:t + 1]
            tok_tgt = tokens_a[t + 1:t + 2]
            res = learner.step(tok_in, tok_tgt, learn=True)
            relearn_trajectory.append(res["loss"])

        post_thermo = learner.auditor.summary()

        # Analysis
        init_arr = np.array(initial_trajectory, dtype=np.float64)
        relearn_arr = np.array(relearn_trajectory, dtype=np.float64)

        nll_init_mean = float(np.mean(init_arr))
        nll_relearn_mean = float(np.mean(relearn_arr))

        init_opening = float(np.mean(init_arr[:min(5, L_A)]))
        relearn_opening = float(np.mean(relearn_arr[:min(5, L_A)]))
        retention_drop = init_opening - relearn_opening

        # Ebbinghaus Savings formula
        savings_ratio = (nll_init_mean - nll_relearn_mean) / max(nll_init_mean, 1e-4)

        # Acceleration factor
        target_init = init_opening - 0.25 * max(0.0, init_opening - np.min(init_arr))
        target_relearn = relearn_opening - 0.25 * max(0.0, relearn_opening - np.min(relearn_arr))

        t_init = next((i for i, v in enumerate(init_arr) if v <= target_init), L_A)
        t_relearn = next((i for i, v in enumerate(relearn_arr) if v <= target_relearn), L_A)
        acceleration = float(t_init + 1) / float(t_relearn + 1)

        thermo_shift = {
            "initial_snr_db": init_thermo.snr_db,
            "final_snr_db": post_thermo.snr_db,
            "delta_snr_db": post_thermo.snr_db - init_thermo.snr_db,
            "initial_spectral_entropy": init_thermo.spectral_entropy,
            "final_spectral_entropy": post_thermo.spectral_entropy,
            "net_entropy_accumulated": post_thermo.net_entropy_accumulated,
            "entropy_balance_ratio": post_thermo.entropy_balance_ratio,
        }

        return SavingsResult(
            n_tokens=L_A,
            intervening_tokens=N_B,
            nll_initial_mean=nll_init_mean,
            nll_relearn_mean=nll_relearn_mean,
            initial_opening_nll=init_opening,
            relearn_opening_nll=relearn_opening,
            retention_immediate_drop=retention_drop,
            savings_ratio=savings_ratio,
            acceleration_factor=acceleration,
            initial_trajectory=initial_trajectory,
            relearn_trajectory=relearn_trajectory,
            thermodynamic_shift=thermo_shift,
        )


# Backward-compatible alias
EbbinghausSavingsEvaluator = ContinuousLifecycleSavingsEvaluator


@dataclass
class CriticalityReport:
    """Subcritical reverberation and dynamical regime metrics (Wilting & Priesemann 2018)."""
    finite_time_lyapunov_exponent: float  # lambda via Benettin method on full physical states
    ftle_std: float
    branching_ratio: float                # sigma autoregressive branching ratio on true action potentials
    criticality_deficit: float            # |sigma - 1.0|
    population_susceptibility: float       # chi = N * Var(activity / N)
    is_subcritical_reverberating: bool    # 0.75 <= sigma <= 0.98 and lambda <= 0.05
    regime: str                           # Subcritical Reverberating / Supercritical / Damped


@dataclass
class ConvergenceReport:
    """Rigorous empirical convergence metrics for continuous streaming learners."""
    prequential_loss_mean: float
    trend_slope_beta: float               # nats / token
    error_autocorrelation_rho1: float     # Martingale lag-1 autocorrelation
    drift_to_noise_ratio_dnr: float       # ||E[g]||^2 / Var(g)
    is_readout_stationary: bool          # DNR < 0.05 and |rho1| < 0.25
    is_task_converged: bool              # is_readout_stationary and mean_loss < 5.0
    status_description: str


@dataclass
class FourPillarEvaluationResult:
    """Complete, unified Four-Pillar scientific evaluation report."""
    pillar_1_prequential: Dict[str, float]
    pillar_2_shock: ReAdaptationResult
    pillar_3_ebbinghaus: SavingsResult
    energy_ledger: EnergyLedger
    information_ledger: InformationLedger
    criticality: CriticalityReport
    convergence: ConvergenceReport

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pillar_1_prequential_tracker": self.pillar_1_prequential,
            "pillar_2_environmental_shock": {
                "n_tokens": self.pillar_2_shock.n_tokens,
                "shock_surprise_initial": self.pillar_2_shock.shock_surprise,
                "plateau_surprise": self.pillar_2_shock.plateau_surprise,
                "delta_shock": self.pillar_2_shock.delta_shock,
                "half_life_tokens": self.pillar_2_shock.half_life_tokens,
                "elasticity_score": self.pillar_2_shock.elasticity_score,
            },
            "pillar_3_unbroken_lifecycle_savings": {
                "n_tokens": self.pillar_3_ebbinghaus.n_tokens,
                "actual_intervening_tokens": self.pillar_3_ebbinghaus.intervening_tokens,
                "nll_initial_mean": self.pillar_3_ebbinghaus.nll_initial_mean,
                "nll_relearn_mean": self.pillar_3_ebbinghaus.nll_relearn_mean,
                "immediate_recall_drop": self.pillar_3_ebbinghaus.retention_immediate_drop,
                "savings_ratio": self.pillar_3_ebbinghaus.savings_ratio,
                "acceleration_factor": self.pillar_3_ebbinghaus.acceleration_factor,
            },
            "energy_ledger": {
                "cumulative_action_potentials": self.energy_ledger.cumulative_spikes,
                "mean_spikes_per_token": self.energy_ledger.mean_spikes_per_token,
                "mean_firing_rate_density": self.energy_ledger.mean_firing_rate_density,
                "cumulative_conductance": self.energy_ledger.cumulative_conductance,
                "mean_conductance_per_token": self.energy_ledger.mean_conductance_per_token,
                "cumulative_adaptation_load": self.energy_ledger.cumulative_adaptation_load,
                "information_efficiency_nats_per_100k_spikes": self.energy_ledger.information_efficiency_nats_per_100k_spikes,
            },
            "information_ledger": {
                "effective_rank": self.information_ledger.effective_rank,
                "participation_ratio": self.information_ledger.participation_ratio,
                "spectral_entropy": self.information_ledger.spectral_entropy,
                "max_spectral_entropy": self.information_ledger.max_spectral_entropy,
                "spectral_entropy_ratio": self.information_ledger.spectral_entropy_ratio,
                "spatial_variance": self.information_ledger.spatial_variance,
                "temporal_roughness": self.information_ledger.temporal_roughness,
                "state_autocorrelation_proxy": self.information_ledger.state_autocorrelation_proxy,
            },
            "criticality_reverberation": {
                "finite_time_lyapunov_exponent": self.criticality.finite_time_lyapunov_exponent,
                "ftle_std": self.criticality.ftle_std,
                "branching_ratio_sigma": self.criticality.branching_ratio,
                "criticality_deficit": self.criticality.criticality_deficit,
                "population_susceptibility": self.criticality.population_susceptibility,
                "is_subcritical_reverberating": self.criticality.is_subcritical_reverberating,
                "regime": self.criticality.regime,
            },
            "convergence_audit": {
                "prequential_loss_mean": self.convergence.prequential_loss_mean,
                "trend_slope_beta": self.convergence.trend_slope_beta,
                "error_autocorrelation_rho1": self.convergence.error_autocorrelation_rho1,
                "drift_to_noise_ratio_dnr": self.convergence.drift_to_noise_ratio_dnr,
                "is_readout_stationary": self.convergence.is_readout_stationary,
                "is_task_converged": self.convergence.is_task_converged,
                "status_description": self.convergence.status_description,
            },
        }


class FourPillarEvaluator:
    """High-efficiency unified evaluator for continuous living connectomes.
    
    Executes the complete Four-Pillar scientific contract in an ultra-fast, streamlined
    budget (~350-450 tokens total, under ~6-8 seconds on GPU):
      1. Pillar 1: Non-repeating prequential stream surprise & trends.
      2. Pillar 2: Plastic Environmental Shock & Re-adaptation (t_1/2, Elasticity).
      3. Pillar 3: Continuous unbroken Ebbinghaus savings (A -> B -> A).
      4. Pillar 4:
         - Physical Energy Ledger (True spikes, conductance dissipation, nats/100k spikes).
         - Information Ledger (Effective rank R_eff, participation ratio D_PR, temporal roughness Δz², Var(z)).
         - Criticality & Reverberation (Branching ratio σ on true action potentials, full-state FTLE λ).
         - Convergence & Stationarity (DNR, trend slope β, autocorrelation ρ_1).
    """

    @staticmethod
    def compute_demeaned_rank_and_pr(X: torch.Tensor) -> Tuple[float, float]:
        """Computes de-meaned Effective Rank (Roy & Vetterli 2007) and Participation Ratio."""
        if X.shape[0] < 10:
            return 1.0, 1.0
        X_c = X - X.mean(dim=0, keepdim=True)
        try:
            svd_vals = torch.linalg.svdvals(X_c)
            var_norm = svd_vals ** 2
            total_var = var_norm.sum().clamp_min(1e-12)
            p = (var_norm / total_var).clamp_min(1e-12)
            h_spec = float(-(p * torch.log(p)).sum().item())
            r_eff = float(math.exp(h_spec))
            pr = float(((total_var ** 2) / (var_norm ** 2).sum().clamp_min(1e-12)).item())
            return r_eff, pr
        except Exception:
            return 1.0, 1.0

    @staticmethod
    def evaluate(
        learner: StreamingConnectomeLearner,
        val_tokens: torch.Tensor,
        cursor: int = 0,
        shock_tokens: int = 150,
        ebbinghaus_seq_a_tokens: int = 40,
        ebbinghaus_intervene_tokens: int = 80,
        ftle_steps: int = 50,
    ) -> FourPillarEvaluationResult:
        device = learner.device
        model = learner.model

        # Save model parameter state to ensure the main organism is 100% unmutated by branch adaptations
        # Exclude static connectome graph buffers to save >300 MB of VRAM!
        saved_weights = {
            k: v.clone() for k, v in model.state_dict().items()
            if not k.startswith("edge_") and not k.startswith("dan_") and k != "lambda_0" and not k.startswith("embedding.")
        }

        def get_slice(start: int, length: int) -> torch.Tensor:
            s = val_tokens[start:start + length + 1]
            if not isinstance(s, torch.Tensor):
                s = torch.from_numpy(np.array(s, dtype=np.int64))
            return s.to(device)

        try:
            # -----------------------------------------------------------------
            # PHASE 1: PLASTIC ENVIRONMENTAL SHOCK + ENERGY/INFO LEDGERS + BRANCHING + DNR
            # -----------------------------------------------------------------
            shock_branch = learner.fork(train_decoder_weight=False, create_optimizer=True)
            tokens_shock = get_slice(cursor, shock_tokens)

            shock_surprises: List[float] = []
            activities: List[float] = []
            h_motor_list: List[torch.Tensor] = []
            g_sum: Optional[torch.Tensor] = None
            g_sq_sum = 0.0
            n_grad_samples = 0

            for t in range(shock_tokens):
                in_t = tokens_shock[t:t + 1]
                tgt_t = tokens_shock[t + 1:t + 2]
                res = shock_branch.step(in_t, tgt_t, learn=True)
                loss_val = res["loss"]
                shock_surprises.append(loss_val)

                # True action potentials (Binary spikes, NOT membrane voltages!)
                act = float(res.get("spike_count", res["spikes"].sum().item()))
                activities.append(act)

                # Gather motor representation states (CPU stored) for static projection rank audit
                with torch.no_grad():
                    if shock_branch.use_sliced_readout:
                        h_m = shock_branch.h[:, shock_branch.model.read_indices].detach()
                    else:
                        h_m = (shock_branch.h * shock_branch.read_mask_1n).detach()
                    h_motor_list.append(h_m.cpu())

                # Online O(1) gradient sampling for DNR
                if t % 2 == 0:
                    h_now = shock_branch.h
                    if shock_branch.use_sliced_readout:
                        readout_vec = h_now[:, shock_branch.model.read_indices]
                    else:
                        readout_vec = h_now * shock_branch.read_mask_1n
                    z_raw = model.output_read(readout_vec)
                    if hasattr(model, "read_norm"):
                        z_latent = model.read_norm(z_raw)
                    else:
                        z_latent = z_raw
                    logits = model.decoder(z_latent)
                    probs = F.softmax(logits, dim=-1)
                    probs[0, tgt_t[0]] -= 1.0
                    error_norm = torch.matmul(probs, model.decoder.weight)
                    if hasattr(model, "read_norm"):
                        rms = torch.sqrt(torch.mean(z_raw ** 2, dim=-1, keepdim=True) + 1e-6)
                        gamma = model.read_norm.weight
                        u_norm = error_norm * gamma
                        z_hat = z_raw / rms
                        proj = (u_norm * z_hat).sum(dim=-1, keepdim=True) / z_raw.shape[-1]
                        error_read = (u_norm - z_hat * proj) / rms
                    else:
                        error_read = error_norm
                    step_grad = torch.matmul(error_read.t(), readout_vec).detach().cpu().flatten()
                    if g_sum is None:
                        g_sum = step_grad.clone()
                    else:
                        g_sum.add_(step_grad)
                    g_sq_sum += float((step_grad ** 2).sum().item())
                    n_grad_samples += 1

            # Shock metrics
            k_init = min(20, shock_tokens // 5)
            k_end = min(40, shock_tokens // 4)
            shock_surprise = float(np.mean(shock_surprises[:k_init]))
            plateau_surprise = float(np.mean(shock_surprises[-k_end:]))
            delta_shock = max(0.0, shock_surprise - plateau_surprise)
            target_mid = shock_surprise - 0.5 * delta_shock
            half_life_tokens = float(shock_tokens)
            for idx_step, s in enumerate(shock_surprises):
                if s <= target_mid:
                    half_life_tokens = float(idx_step)
                    break
            elasticity_score = delta_shock / (half_life_tokens + 1.0)
            thermo_summary = shock_branch.auditor.summary()

            # Layer-by-layer de-meaned effective rank auditing & net contextual gain
            # Pure geometric projection under frozen snapshot weights (strictly zero parameter drift!)
            if h_motor_list:
                H_m = torch.cat(h_motor_list, dim=0)  # [shock_tokens, N_motor] on CPU
                with torch.no_grad():
                    w_read_static = saved_weights["output_read.weight"].to(device)
                    Z_r_static = torch.matmul(H_m.to(device), w_read_static.t())
                    if "read_norm.weight" in saved_weights:
                        gamma_static = saved_weights["read_norm.weight"].to(device)
                        rms_static = torch.sqrt(torch.mean(Z_r_static ** 2, dim=-1, keepdim=True) + 1e-6)
                        Z_n_static = (Z_r_static / rms_static) * gamma_static
                    else:
                        Z_n_static = Z_r_static

                r_eff_m, pr_m = FourPillarEvaluator.compute_demeaned_rank_and_pr(H_m)
                r_eff_r, pr_r = FourPillarEvaluator.compute_demeaned_rank_and_pr(Z_r_static.cpu())
                r_eff_n, pr_n = FourPillarEvaluator.compute_demeaned_rank_and_pr(Z_n_static.cpu())

                unigram_base = float(getattr(learner, "unigram_baseline_nll", 7.6076))
                if thermo_summary.information is not None:
                    thermo_summary.information.rank_motor_eff = r_eff_m
                    thermo_summary.information.rank_w_read_eff = r_eff_r
                    thermo_summary.information.rank_rmsnorm_eff = r_eff_n
                    thermo_summary.information.pr_motor = pr_m
                    thermo_summary.information.pr_w_read = pr_r
                    thermo_summary.information.pr_rmsnorm = pr_n
                    thermo_summary.information.unigram_baseline_nll = unigram_base
                    thermo_summary.information.net_contextual_gain = unigram_base - thermo_summary.information.mean_surprise_nats

            pillar_2_shock = ReAdaptationResult(
                n_tokens=shock_tokens,
                shock_surprise=shock_surprise,
                plateau_surprise=plateau_surprise,
                delta_shock=delta_shock,
                half_life_tokens=half_life_tokens,
                elasticity_score=elasticity_score,
                trajectory=shock_surprises,
                thermodynamics=thermo_summary,
            )

            # Branching ratio sigma on true action potentials (Wilting & Priesemann 2018)
            act_t = np.array(activities[:-1])
            act_t1 = np.array(activities[1:])
            cov_matrix = np.cov(act_t, act_t1)
            var_act_t = float(cov_matrix[0, 0])
            branching_ratio = float(cov_matrix[0, 1] / max(var_act_t, 1e-9)) if var_act_t > 0 else 1.0
            n_neurons = model.n_neurons
            var_density = np.var(np.array(activities) / n_neurons)
            susceptibility = float(n_neurons * var_density)

            # Convergence & DNR
            loss_arr = np.array(shock_surprises)
            poly_fit = np.polyfit(np.arange(len(loss_arr)), loss_arr, 1)
            beta_trend = float(poly_fit[0])
            mean_loss = float(np.mean(loss_arr))
            loss_var = float(np.var(loss_arr))
            loss_cent = loss_arr - mean_loss
            rho_1 = float(np.mean(loss_cent[:-1] * loss_cent[1:]) / max(loss_var, 1e-9))

            if n_grad_samples > 0 and g_sum is not None:
                g_mean = g_sum / n_grad_samples
                drift_power = float((g_mean ** 2).sum().item())
                mean_sq = g_sq_sum / n_grad_samples
                noise_power = max(1e-12, mean_sq - drift_power)
                dnr = drift_power / max(noise_power, 1e-12)
            else:
                dnr = 0.0

            is_readout_stationary = (dnr < 0.05) and (abs(rho_1) < 0.25)
            is_task_converged = is_readout_stationary and (mean_loss < 5.0)
            if is_task_converged:
                status_desc = "Task Converged"
            elif is_readout_stationary:
                status_desc = "Readout Stationary at Representation Plateau"
            else:
                status_desc = "Actively Adapting"

            convergence_report = ConvergenceReport(
                prequential_loss_mean=mean_loss,
                trend_slope_beta=beta_trend,
                error_autocorrelation_rho1=rho_1,
                drift_to_noise_ratio_dnr=dnr,
                is_readout_stationary=is_readout_stationary,
                is_task_converged=is_task_converged,
                status_description=status_desc,
            )

            # -----------------------------------------------------------------
            # PHASE 2: FTLE (BENETTIN SHADOW PERTURBATION METHOD ON FULL STATE)
            # -----------------------------------------------------------------
            eps_pert = 1e-4
            primary_fork = learner.fork(train_decoder_weight=False, create_optimizer=False)
            shadow_fork = learner.fork(train_decoder_weight=False, create_optimizer=False)

            pert_h = torch.randn_like(shadow_fork.h)
            pert_ge = torch.randn_like(shadow_fork.syn_state.get("ge", shadow_fork.h))
            pert_gi = torch.randn_like(shadow_fork.syn_state.get("gi", shadow_fork.h))
            pert_b = torch.randn_like(shadow_fork.syn_state.get("b", shadow_fork.h))
            total_pert_norm = torch.sqrt(
                (pert_h ** 2).sum() + (pert_ge ** 2).sum() + (pert_gi ** 2).sum() + (pert_b ** 2).sum()
            ).clamp_min(1e-12)

            shadow_fork.h.add_((pert_h / total_pert_norm) * eps_pert)
            if "ge" in shadow_fork.syn_state:
                shadow_fork.syn_state["ge"].add_((pert_ge / total_pert_norm) * eps_pert)
            if "gi" in shadow_fork.syn_state:
                shadow_fork.syn_state["gi"].add_((pert_gi / total_pert_norm) * eps_pert)
            if "b" in shadow_fork.syn_state:
                shadow_fork.syn_state["b"].add_((pert_b / total_pert_norm) * eps_pert)

            cursor_ftle = cursor + shock_tokens
            ftle_tokens = get_slice(cursor_ftle, ftle_steps)
            lyapunov_steps: List[float] = []

            for t in range(ftle_steps):
                tok_in = ftle_tokens[t:t + 1]
                tok_tgt = ftle_tokens[t + 1:t + 2]
                primary_fork.step(tok_in, tok_tgt, learn=False)
                shadow_fork.step(tok_in, tok_tgt, learn=False)

                d_h = shadow_fork.h - primary_fork.h
                d_ge = shadow_fork.syn_state.get("ge", torch.zeros_like(d_h)) - primary_fork.syn_state.get("ge", torch.zeros_like(d_h))
                d_gi = shadow_fork.syn_state.get("gi", torch.zeros_like(d_h)) - primary_fork.syn_state.get("gi", torch.zeros_like(d_h))
                d_b = shadow_fork.syn_state.get("b", torch.zeros_like(d_h)) - primary_fork.syn_state.get("b", torch.zeros_like(d_h))
                delta_norm = float(torch.sqrt((d_h ** 2).sum() + (d_ge ** 2).sum() + (d_gi ** 2).sum() + (d_b ** 2).sum()).item())
                ftle_step = math.log(max(delta_norm, 1e-12) / eps_pert)
                lyapunov_steps.append(ftle_step)

                scale = eps_pert / max(delta_norm, 1e-12)
                shadow_fork.h.copy_(primary_fork.h + d_h * scale)
                if "ge" in shadow_fork.syn_state:
                    shadow_fork.syn_state["ge"].copy_(primary_fork.syn_state["ge"] + d_ge * scale)
                if "gi" in shadow_fork.syn_state:
                    shadow_fork.syn_state["gi"].copy_(primary_fork.syn_state["gi"] + d_gi * scale)
                if "b" in shadow_fork.syn_state:
                    shadow_fork.syn_state["b"].copy_(primary_fork.syn_state["b"] + d_b * scale)

            mean_ftle = float(np.mean(lyapunov_steps))
            std_ftle = float(np.std(lyapunov_steps))

            is_subcritical = (0.75 <= branching_ratio <= 0.98) and (mean_ftle <= 0.05)
            if 0.98 < branching_ratio <= 1.05 and abs(mean_ftle) <= 0.05:
                regime = "Exact Critical Boundary (Edge-of-Chaos)"
            elif is_subcritical:
                regime = "Subcritical Reverberating (Biologically Optimal Echo, Wilting & Priesemann 2018)"
            elif branching_ratio < 0.75:
                regime = "Strongly Damped / Over-Dissipative"
            else:
                regime = "Supercritical / Epileptiform Runaway"

            criticality_report = CriticalityReport(
                finite_time_lyapunov_exponent=mean_ftle,
                ftle_std=std_ftle,
                branching_ratio=branching_ratio,
                criticality_deficit=abs(branching_ratio - 1.0),
                population_susceptibility=susceptibility,
                is_subcritical_reverberating=is_subcritical,
                regime=regime,
            )

            # -----------------------------------------------------------------
            # PHASE 3: UNBROKEN CONTINUOUS LIFECYCLE EBBINGHAUS SAVINGS (A -> B -> A)
            # -----------------------------------------------------------------
            ebb_branch = learner.fork(train_decoder_weight=False, create_optimizer=True)
            cursor_ebb = cursor_ftle + ftle_steps
            tokens_a = get_slice(cursor_ebb, ebbinghaus_seq_a_tokens)
            cursor_b = cursor_ebb + ebbinghaus_seq_a_tokens
            tokens_b = get_slice(cursor_b, ebbinghaus_intervene_tokens)

            pillar_3_ebbinghaus = ContinuousLifecycleSavingsEvaluator.evaluate_unbroken_lifecycle(
                learner=ebb_branch,
                tokens_a=tokens_a,
                tokens_b=tokens_b,
            )

            return FourPillarEvaluationResult(
                pillar_1_prequential=learner.tracker.summary(),
                pillar_2_shock=pillar_2_shock,
                pillar_3_ebbinghaus=pillar_3_ebbinghaus,
                energy_ledger=thermo_summary.energy,
                information_ledger=thermo_summary.information,
                criticality=criticality_report,
                convergence=convergence_report,
            )

        finally:
            # Restore model parameter weights to guarantee zero contamination of the running learner
            model.load_state_dict(saved_weights, strict=False)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

