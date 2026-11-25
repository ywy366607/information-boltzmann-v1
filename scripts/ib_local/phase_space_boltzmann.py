"""Full Phase-Space Boltzmann Coupling (Pieces 1 + 2 + 3) for MengLi's S/P Body.

Implements the formally verified statistical-mechanical upgrade proved in
`lean/CGBAxioms/reviews/PhaseSpaceBoltzmann.lean`:

1. **Phase-Space Hamiltonian Memory Readout (`H(S, P; i) = U(S, S_i) + K(P, P_i) - μ_i`)**:
   Adds live 32-layer BFSS momentum alignment `α_P · (q_P · k_{P,i})` to memory
   retrieval logits, where `q_P` is the normalized latent momentum readout via
   MengLi's existing `decision_momentum_read + momentum_to_boundary` and
   `k_{P,i} = normalize(enc(long_successor_i) - enc(source_i))` is each stored
   episode's observed phase-space trajectory direction (zero archive schema break!).

2. **Equipartition Kinetic Inverse Temperature (`β(P) = β₀ · exp(-κ · tanh(z_kin))`)**:
   Couples 32-layer kinetic energy `E_kin = mean(P^2)` relative to its causal
   EMA baseline to local inverse temperatures on:
     - Byte language readout (preserving Fast Table reference measure `g(x)`):
       `logit(x) = g(x) + β_lang(P) · (z(x) - g(x))`
     - Memory retrieval readout (`logits *= β_mem_scale(P)`)
     - High-level action selection (`wood_choice_logits *= β_act_scale(P)`)
   All 4 coupling scalars `(α_P, κ_lang, κ_mem, κ_act)` are bounded in the Lean
   safety envelope and trained end-to-end by MengLi's existing `BudgetedAdamW + Wiener`.

3. **Partition-Function Free-Energy Nosé–Hoover Thermostat (`F = -T log Z -> P`)**:
   Computes exact Boltzmann entropy `H = log Z - E_p[logit]` after each observed
   update and applies the Lean-proved forward-invariant NESS thermostat to `P`:
     `P_l <- exp(0.5 * (η_F · f_t - γ_diss · max(0, log(E_kin / E_ema)))) · P_l`
   High free-energy/entropy pressure amplifies `P`; Nosé–Hoover dissipation
   strictly bounds `E_kin` inside the NESS attractor.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def _inv_sigmoid(y: float, hi: float) -> float:
    frac = min(1.0 - 1e-5, max(1e-5, float(y) / float(hi)))
    return math.log(frac / (1.0 - frac))


def _inv_tanh(y: float, hi: float) -> float:
    frac = min(1.0 - 1e-5, max(-1.0 + 1e-5, float(y) / float(hi)))
    return 0.5 * math.log((1.0 + frac) / (1.0 - frac))


class PhaseSpaceBoltzmann(nn.Module):
    """Coupled phase-space Hamiltonian readout, kinetic temperature & NESS thermostat."""

    def __init__(self, latent: int, *,
                 alpha_max: float = 0.50, alpha_init: float = 0.08,
                 kappa_max: float = 0.35, kappa_lang_init: float = 0.04,
                 kappa_mem_init: float = 0.04, kappa_act_init: float = 0.04,
                 eta_free_energy: float = 0.12, gamma_diss: float = 0.16,
                 max_log_scale: float = 0.18, ema_decay: float = 0.85,
                 enable_momentum_read: bool = True,
                 enable_kinetic_temp: bool = True,
                 enable_thermostat: bool = True,
                 device=None, dtype=None):
        super().__init__()
        self.latent = int(latent)
        self.alpha_max = float(alpha_max)
        self.kappa_max = float(kappa_max)
        self.eta_free_energy = float(eta_free_energy)
        self.gamma_diss = float(gamma_diss)
        self.max_log_scale = float(max_log_scale)
        self.ema_decay = float(ema_decay)
        self.enable_momentum_read = bool(enable_momentum_read)
        self.enable_kinetic_temp = bool(enable_kinetic_temp)
        self.enable_thermostat = bool(enable_thermostat)

        init_vals = [
            _inv_sigmoid(alpha_init, self.alpha_max),
            _inv_tanh(kappa_lang_init, self.kappa_max),
            _inv_tanh(kappa_mem_init, self.kappa_max),
            _inv_tanh(kappa_act_init, self.kappa_max),
        ]
        self.theta = nn.Parameter(torch.tensor(init_vals, device=device, dtype=dtype or torch.float32))

        # Fixed-address buffers (100% CUDA-graph safe & checkpoint-restorable).
        dt = dtype or torch.float32
        self.register_buffer('live_momentum_dir', torch.zeros(self.latent, device=device, dtype=dt))
        self.register_buffer('live_z_kin', torch.zeros((), device=device, dtype=dt))
        self.register_buffer('kinetic_ema', torch.zeros((), device=device, dtype=dt))
        self.register_buffer('kinetic_count', torch.zeros((), device=device, dtype=torch.long))
        self.register_buffer('entropy_ema', torch.zeros((), device=device, dtype=dt))
        self.register_buffer('entropy_var', torch.full((), 0.04, device=device, dtype=dt))
        self.register_buffer('entropy_count', torch.zeros((), device=device, dtype=torch.long))
        self.register_buffer('thermostat_steps', torch.zeros((), device=device, dtype=torch.long))
        self.register_buffer('last_thermostat_scale', torch.ones((), device=device, dtype=dt))
        self.register_buffer('last_free_energy', torch.zeros((), device=device, dtype=dt))
        self.register_buffer('last_entropy', torch.zeros((), device=device, dtype=dt))
        self.register_buffer('last_kinetic_energy', torch.zeros((), device=device, dtype=dt))
        self.register_buffer('last_pressure', torch.zeros((), device=device, dtype=dt))

        # Purely observational telemetry (kept off state_dict buffers so legacy 4-param checkpoints load strictly).
        self._prev_post_log_e_kin: float | None = None
        self._prev_post_log_e_ema: float | None = None
        self._last_centered_free_energy: float | None = None
        self._last_kl_to_uniform: float | None = None
        self._last_energy_ledger: dict[str, float | str | None] = dict(
            source=None,
            log_e_kin_before=None,
            log_e_kin_after=None,
            log_e_ema_before=None,
            log_e_ema_after=None,
            delta_log_e_body=0.0,
            delta_log_e_thermostat=0.0,
            delta_log_e_ema=0.0,
            delta_log_e_total=0.0,
            u_before=None,
            u_after=None,
            delta_u=0.0,
        )

    # --- Differentiable bounded physical couplings (Theorems 1, 2, 3) ---

    def alpha_p(self) -> torch.Tensor:
        if not self.enable_momentum_read:
            return self.theta.new_zeros(())
        return self.alpha_max * torch.sigmoid(self.theta[0])

    def kappa_lang(self) -> torch.Tensor:
        if not self.enable_kinetic_temp:
            return self.theta.new_zeros(())
        return self.kappa_max * torch.tanh(self.theta[1])

    def kappa_mem(self) -> torch.Tensor:
        if not self.enable_kinetic_temp:
            return self.theta.new_zeros(())
        return self.kappa_max * torch.tanh(self.theta[2])

    def kappa_act(self) -> torch.Tensor:
        if not self.enable_kinetic_temp:
            return self.theta.new_zeros(())
        return self.kappa_max * torch.tanh(self.theta[3])

    def lang_inv_temp_scale(self) -> torch.Tensor:
        if not self.enable_kinetic_temp:
            return self.theta.new_ones(())
        return torch.exp(-self.kappa_lang() * self.live_z_kin)

    def mem_inv_temp_scale(self) -> torch.Tensor:
        if not self.enable_kinetic_temp:
            return self.theta.new_ones(())
        return torch.exp(-self.kappa_mem() * self.live_z_kin)

    def act_inv_temp_scale(self) -> torch.Tensor:
        if not self.enable_kinetic_temp:
            return self.theta.new_ones(())
        return torch.exp(-self.kappa_act() * self.live_z_kin)

    def memory_velocity_bonus(self, vel_keys: torch.Tensor) -> torch.Tensor:
        """Piece 1: Phase-space momentum alignment bonus `[1, capacity]`."""
        if not self.enable_momentum_read:
            return vel_keys.new_zeros((1, vel_keys.shape[0]))
        return (self.alpha_p() * (vel_keys @ self.live_momentum_dir)).unsqueeze(0)

    def modulate_language_logits(self, logits: torch.Tensor, table_logp: torch.Tensor) -> torch.Tensor:
        """Piece 2: Reference-measure-preserving Boltzmann language readout (Theorem 2)."""
        if not self.enable_kinetic_temp:
            return logits
        beta = self.lang_inv_temp_scale()
        return table_logp + beta * (logits - table_logp)

    # --- Live state refresh & Piece 3 NESS thermostat ---

    @staticmethod
    def _kinetic_energy(momenta) -> float:
        if not momenta:
            return 0.0
        total = 0.0
        count = 0
        for p in momenta:
            if torch.is_tensor(p) and p.numel() > 0:
                total += float(p.detach().float().square().mean().item())
                count += 1
        return total / max(1, count)

    @torch.no_grad()
    def refresh_live_phase_state(self, engine, query_raw: torch.Tensor | None = None,
                                 *, update_ema: bool = True,
                                 _from_thermostat: bool = False) -> dict:
        """Update fixed-address `(live_momentum_dir, live_z_kin)` at a real boundary.

        NEVER called mid-draft so CUDA graph drafting and batched proposal replay
        observe identical phase-space parameters.
        """
        momenta = getattr(engine, 'bfss_momenta', None)
        e_kin = self._kinetic_energy(momenta)
        self.last_kinetic_energy.fill_(e_kin)

        cnt = int(self.kinetic_count.item())
        eps = 1e-12
        e_ema_before = float(self.kinetic_ema.item()) if cnt > 0 else e_kin
        if cnt == 0:
            z_kin = 0.0
            if update_ema and e_kin > 0.0:
                self.kinetic_ema.fill_(e_kin)
                self.kinetic_count.add_(1)
        else:
            e_ema = float(self.kinetic_ema.item())
            u_raw = math.log((e_kin + eps) / (e_ema + eps))
            z_kin = math.tanh(u_raw)
            if update_ema and e_kin > 0.0:
                self.kinetic_ema.mul_(self.ema_decay).add_(e_kin * (1.0 - self.ema_decay))
                self.kinetic_count.add_(1)
        self.live_z_kin.fill_(z_kin)

        if update_ema and not _from_thermostat and math.isfinite(e_kin) and e_kin > 0.0:
            e_ema_after = float(self.kinetic_ema.item())
            log_e_after = math.log(e_kin + eps)
            log_ema_after = math.log(max(e_ema_after, 0.0) + eps)
            if self._prev_post_log_e_kin is not None and self._prev_post_log_e_ema is not None:
                log_e_before = self._prev_post_log_e_kin
                log_ema_before = self._prev_post_log_e_ema
            else:
                log_e_before = log_e_after
                log_ema_before = math.log(max(e_ema_before, 0.0) + eps) if cnt > 0 else log_ema_after
            delta_body = log_e_after - log_e_before
            delta_ema = log_ema_after - log_ema_before
            u_before = log_e_before - log_ema_before
            u_after = log_e_after - log_ema_after
            self._last_energy_ledger = dict(
                source='boundary_refresh',
                log_e_kin_before=log_e_before,
                log_e_kin_after=log_e_after,
                log_e_ema_before=log_ema_before,
                log_e_ema_after=log_ema_after,
                delta_log_e_body=delta_body,
                delta_log_e_thermostat=0.0,
                delta_log_e_ema=delta_ema,
                delta_log_e_total=delta_body,
                u_before=u_before,
                u_after=u_after,
                delta_u=u_after - u_before,
            )
            self._prev_post_log_e_kin = log_e_after
            self._prev_post_log_e_ema = log_ema_after

        # Read live 32-layer momentum direction into latent trajectory space.
        model = engine.model
        if self.enable_momentum_read and momenta and hasattr(model, 'momentum_to_boundary'):
            q_raw = query_raw
            if q_raw is None:
                m = getattr(model, 'memory', None)
                if m is not None and getattr(m, 'pending_count', 0) > 0:
                    q_raw = m.pending[-1:]
                elif getattr(engine, '_internal_boundary', None) is not None:
                    q_raw = engine._internal_boundary.reshape(1, -1)
            if q_raw is not None:
                from scripts.ib_local.navigation_momentum_wood import decision_momentum_read
                q2 = q_raw.detach().reshape(1, -1)
                p_read = decision_momentum_read(model, momenta, q2)
                v_p = model.momentum_to_boundary(p_read)[0]
                v_norm = float(v_p.norm().item())
                if math.isfinite(v_norm) and v_norm > 1e-8:
                    self.live_momentum_dir.copy_(v_p / v_norm)
                else:
                    self.live_momentum_dir.zero_()
            else:
                self.live_momentum_dir.zero_()
        else:
            self.live_momentum_dir.zero_()

        return dict(
            e_kin=e_kin,
            z_kin=z_kin,
            momentum_dir_norm=float(self.live_momentum_dir.norm().item()),
        )

    @torch.no_grad()
    def apply_free_energy_thermostat(self, engine, logits: torch.Tensor,
                                     query_raw: torch.Tensor | None = None) -> dict:
        """Piece 3: Exact Boltzmann entropy / free-energy Nosé–Hoover thermostat on P."""
        if logits is None or logits.numel() == 0:
            return dict(applied=False)
        lg = logits.detach().float()
        log_z = torch.logsumexp(lg, dim=-1)
        probs = torch.softmax(lg, dim=-1)
        entropy = float((log_z - (probs * lg).sum(dim=-1)).mean().item())
        beta_lang = max(1e-6, float(self.lang_inv_temp_scale().detach().item()))
        free_energy = float((-log_z.mean().item()) / beta_lang)
        # Shift-invariant gauge-fixed free energy (zero-mean logits) & KL-to-uniform telemetry.
        lg_centered = lg - lg.mean(dim=-1, keepdim=True)
        log_z_centered = torch.logsumexp(lg_centered, dim=-1)
        centered_free_energy = float((-log_z_centered.mean().item()) / beta_lang)
        kl_to_uniform = float(math.log(max(1, int(lg.shape[-1]))) - entropy)
        if not (math.isfinite(entropy) and math.isfinite(free_energy) and math.isfinite(centered_free_energy)):
            return dict(applied=False, non_finite=True)
        self.last_entropy.fill_(entropy)
        self.last_free_energy.fill_(free_energy)
        self._last_centered_free_energy = centered_free_energy
        self._last_kl_to_uniform = kl_to_uniform

        cnt = int(self.entropy_count.item())
        if cnt == 0:
            pressure = 0.0
            self.entropy_ema.fill_(entropy)
            self.entropy_count.add_(1)
        else:
            h_ema = float(self.entropy_ema.item())
            h_var = float(self.entropy_var.item())
            if not (math.isfinite(h_ema) and math.isfinite(h_var)):
                self.entropy_ema.fill_(entropy)
                self.entropy_var.zero_()
                pressure = 0.0
            else:
                diff = entropy - h_ema
                std = math.sqrt(max(h_var, 1e-4))
                pressure = math.tanh(diff / std)
                self.entropy_ema.mul_(self.ema_decay).add_(entropy * (1.0 - self.ema_decay))
                self.entropy_var.mul_(self.ema_decay).add_((diff * diff) * (1.0 - self.ema_decay))
            self.entropy_count.add_(1)
        self.last_pressure.fill_(pressure)

        eps = 1e-12
        scale = 1.0
        momenta = getattr(engine, 'bfss_momenta', None)
        e_kin_before = self._kinetic_energy(momenta) if momenta else 0.0
        had_prev_ema = int(self.kinetic_count.item()) > 0 and float(self.kinetic_ema.item()) > 0.0
        e_ema_before = float(self.kinetic_ema.item()) if had_prev_ema else e_kin_before
        if self.enable_thermostat and momenta:
            e_kin = e_kin_before
            e_ema = float(self.kinetic_ema.item())
            if math.isfinite(e_kin):
                if int(self.kinetic_count.item()) > 0 and math.isfinite(e_ema) and e_ema > 0.0:
                    u_pos = max(0.0, math.log((e_kin + eps) / (e_ema + eps)))
                else:
                    u_pos = 0.0
                # EMA-relative damping above kinetic_ema; positive free-energy pressure amplifies P.
                log_scale = 0.5 * (self.eta_free_energy * pressure - self.gamma_diss * u_pos)
                if math.isfinite(log_scale):
                    log_scale = max(-self.max_log_scale, min(self.max_log_scale, log_scale))
                    scale = math.exp(log_scale)
                    for p in momenta:
                        if torch.is_tensor(p):
                            p.mul_(scale)
                    engine.model._momentum_current = momenta
                    self.thermostat_steps.add_(1)
        self.last_thermostat_scale.fill_(scale)

        phase_info = self.refresh_live_phase_state(
            engine, query_raw=query_raw, update_ema=True, _from_thermostat=True,
        )
        e_kin_after = float(phase_info['e_kin'])
        e_ema_after = float(self.kinetic_ema.item())
        if math.isfinite(e_kin_before) and math.isfinite(e_kin_after) and (e_kin_before > 0.0 or e_kin_after > 0.0):
            log_e_pre = math.log(max(e_kin_before, 0.0) + eps)
            log_e_after = math.log(max(e_kin_after, 0.0) + eps)
            log_ema_after = math.log(max(e_ema_after, 0.0) + eps)
            if self._prev_post_log_e_kin is not None and self._prev_post_log_e_ema is not None:
                log_e_prev = self._prev_post_log_e_kin
                log_ema_prev = self._prev_post_log_e_ema
            else:
                log_e_prev = log_e_pre
                log_ema_prev = math.log(max(e_ema_before, 0.0) + eps) if had_prev_ema else log_ema_after
            delta_body = log_e_pre - log_e_prev
            delta_thermo = 2.0 * math.log(max(scale, eps))
            delta_total = log_e_after - log_e_prev
            delta_ema = log_ema_after - log_ema_prev
            u_before = log_e_prev - log_ema_prev
            u_after = log_e_after - log_ema_after
            self._last_energy_ledger = dict(
                source='thermostat',
                log_e_kin_before=log_e_prev,
                log_e_kin_pre_thermostat=log_e_pre,
                log_e_kin_after=log_e_after,
                log_e_ema_before=log_ema_prev,
                log_e_ema_after=log_ema_after,
                delta_log_e_body=delta_body,
                delta_log_e_thermostat=delta_thermo,
                delta_log_e_ema=delta_ema,
                delta_log_e_total=delta_total,
                u_before=u_before,
                u_after=u_after,
                delta_u=u_after - u_before,
            )
            self._prev_post_log_e_kin = log_e_after
            self._prev_post_log_e_ema = log_ema_after
        return dict(
            applied=True,
            entropy=entropy,
            free_energy=free_energy,
            centered_free_energy=centered_free_energy,
            kl_to_uniform=kl_to_uniform,
            pressure=pressure,
            thermostat_scale=scale,
            energy_ledger=dict(self._last_energy_ledger),
            **phase_info,
        )

    def diagnostics(self, engine=None) -> dict:
        wiener_gate = None
        if engine is not None and hasattr(engine, 'optimizer'):
            st = engine.optimizer.state.get(self.theta)
            if isinstance(st, dict) and torch.is_tensor(st.get('budget_gate')):
                wiener_gate = float(st['budget_gate'].detach().float().mean().item())
        return dict(
            schema=1,
            alpha_p=float(self.alpha_p().detach().item()),
            kappa_lang=float(self.kappa_lang().detach().item()),
            kappa_mem=float(self.kappa_mem().detach().item()),
            kappa_act=float(self.kappa_act().detach().item()),
            lang_inv_temp_scale=float(self.lang_inv_temp_scale().detach().item()),
            mem_inv_temp_scale=float(self.mem_inv_temp_scale().detach().item()),
            act_inv_temp_scale=float(self.act_inv_temp_scale().detach().item()),
            live_z_kin=float(self.live_z_kin.item()),
            kinetic_ema=float(self.kinetic_ema.item()),
            kinetic_count=int(self.kinetic_count.item()),
            entropy_ema=float(self.entropy_ema.item()),
            last_entropy=float(self.last_entropy.item()),
            last_free_energy=float(self.last_free_energy.item()),
            last_centered_free_energy=self._last_centered_free_energy,
            last_kl_to_uniform=self._last_kl_to_uniform,
            last_kinetic_energy=float(self.last_kinetic_energy.item()),
            last_pressure=float(self.last_pressure.item()),
            last_thermostat_scale=float(self.last_thermostat_scale.item()),
            thermostat_steps=int(self.thermostat_steps.item()),
            momentum_dir_norm=float(self.live_momentum_dir.norm().item()),
            energy_ledger=dict(self._last_energy_ledger),
            wiener_mean_gate=wiener_gate,
            enable_momentum_read=self.enable_momentum_read,
            enable_kinetic_temp=self.enable_kinetic_temp,
            enable_thermostat=self.enable_thermostat,
        )


def enabled(engine) -> bool:
    return (getattr(engine, 'phase_space_boltzmann_schema', None) == 1 and
            hasattr(engine.model, 'phase_boltzmann'))


def configure(engine, config=None, *, lr: float = 0.01, initialize: bool = True,
              recapture_decision: bool = True, **kwargs) -> PhaseSpaceBoltzmann:
    """Attach Full Phase-Space Boltzmann Coupling (1+2+3) cleanly to an existing engine."""
    if hasattr(engine.model, 'phase_boltzmann') and isinstance(engine.model.phase_boltzmann, PhaseSpaceBoltzmann):
        engine.phase_space_boltzmann_schema = 1
        object.__setattr__(engine.model.memory, '_phase_boltzmann', engine.model.phase_boltzmann)
        return engine.model.phase_boltzmann

    model = engine.model
    ref = model.encoder.weight
    latent = int(model.encoder.out_features)
    module = PhaseSpaceBoltzmann(latent, device=ref.device, dtype=ref.dtype, **kwargs)
    model.phase_boltzmann = module
    # Non-module private link on memory so state_dict never duplicates parameters.
    object.__setattr__(model.memory, '_phase_boltzmann', module)
    engine.phase_space_boltzmann_schema = 1

    if hasattr(engine, 'optimizer') and engine.optimizer is not None:
        old_count = len(getattr(engine, 'params', ()))
        engine.optimizer.add_param_group(dict(params=[module.theta], lr=float(lr), weight_decay=0.0))
        if hasattr(engine, 'params'):
            engine.params = [*engine.params, module.theta]
        if hasattr(engine, 'groups'):
            engine.groups = (*engine.groups, [module.theta])
        if initialize and old_count > 0:
            def extend(value):
                if isinstance(value, (list, tuple)) and len(value) == old_count and all(
                        v is None or torch.is_tensor(v) for v in value):
                    return type(value)([*value, None])
                if isinstance(value, dict):
                    return {k: extend(v) for k, v in value.items()}
                return value
            for name in ('prev', 'internal_pending', 'fire_grad'):
                if hasattr(engine, name):
                    setattr(engine, name, extend(getattr(engine, name)))
            if hasattr(engine, 'head_mask'):
                engine.head_mask = [*engine.head_mask, True]

    if initialize:
        module.refresh_live_phase_state(engine, update_ema=True)

    # Re-capture ScopedDecisionGraph so the 1-byte CUDA graph reads phase_boltzmann's
    # fixed-address live buffers and includes velocity keys in fact_cache!
    if recapture_decision and hasattr(engine, 'decision') and engine.decision is not None:
        from scripts.ib_local.scoped_decision_graph import ScopedDecisionGraph
        prev_dec = engine.decision
        fused = bool(getattr(prev_dec, 'step_bundles', ()))
        cap_hand = hasattr(prev_dec, '_capture_handover_budget')
        engine.decision = ScopedDecisionGraph(
            model, capture=bool(getattr(prev_dec, 'capture', ref.is_cuda)),
            fused_step=fused, capture_handover=cap_hand)

    if config is not None:
        config['phase_space_boltzmann'] = dict(schema=1, lr=float(lr), **kwargs)
    return module


def diagnostics(engine) -> dict | None:
    if not enabled(engine):
        return None
    return engine.model.phase_boltzmann.diagnostics(engine)
