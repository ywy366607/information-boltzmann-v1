"""Deterministic numerical and invariant tests for O(1) e-prop and Three-Factor Plasticity."""
import pytest
import torch
import numpy as np

from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core.eprop_credit_assignment import (
    EPropEligibilityState,
    EPropCreditAssignment,
    StreamingConnectomeTrainer,
)


@pytest.fixture
def tiny_fly_model(tmp_path):
    n = 20
    rng = np.random.default_rng(42)
    pre = rng.integers(0, n, 40).astype(np.int32)
    post = rng.integers(0, n, 40).astype(np.int32)
    weight = np.abs(rng.standard_normal(40)).astype(np.float32)
    nt_sign = np.array([1] * 20 + [-1] * 20, dtype=np.int8)
    superclass_id = np.array([0] * 7 + [1] * 7 + [2] * 6, dtype=np.int8)

    path = tmp_path / "tiny_eprop_coba.npz"
    np.savez_compressed(
        path,
        edge_pre=pre, edge_post=post, edge_weight=weight,
        edge_delay=np.ones_like(pre, dtype=np.int32),
        neuron_body_ids=np.arange(n, dtype=np.int64),
        nt_sign=nt_sign,
        superclass_id=superclass_id,
        superclass_names=np.array(["sensory", "interneuron", "output"]),
        meta=np.array("{}"),
    )

    model = FlyReservoirLM(
        path, vocab_size=32, d_model=16,
        learnable_time_constants=True,
        learnable_thresholds=True,
        learnable_conductance_gains=True,
        use_alif=True,
        use_stp=True,
        synapse_model="coba",
        threshold=0.08,
    )
    return model


def test_eprop_o1_memory_invariance(tiny_fly_model):
    """Verifies that eligibility state memory is strictly O(1) regardless of step count."""
    model = tiny_fly_model
    trainer = StreamingConnectomeTrainer(model, lr_readout=1e-3, lr_synapse=1e-4)

    h = torch.zeros(1, model.n_neurons)
    ring = tuple(torch.zeros(1, model.n_neurons) for _ in range(4))
    syn_state = {
        "ge": torch.zeros(1, model.n_neurons),
        "gi": torch.zeros(1, model.n_neurons),
        "b": torch.zeros(1, model.n_neurons),
        "x": torch.ones(1, model.n_neurons),
        "u": model.get_stp_params()[0].clone(),
    }

    # Step through 50 continuous streaming tokens
    losses = []
    for t in range(50):
        in_tok = torch.tensor([t % 32])
        tgt_tok = torch.tensor([(t + 1) % 32])
        loss, h, ring, syn_state = trainer.train_streaming_step(
            in_tok, tgt_tok, h, ring, syn_state, update_synapses=True
        )
        losses.append(loss)

    # State sizes must remain strictly invariant
    assert trainer.eligibility.z_bar.shape == (1, 4, model.n_neurons)
    assert trainer.eligibility.zeta_b.shape == (1, model.n_neurons)
    assert torch.isfinite(h).all()
    assert torch.isfinite(trainer.eligibility.z_bar).all()
    assert torch.isfinite(trainer.eligibility.zeta_b).all()


def test_catastrophic_forgetting_immunity_on_silent_circuits(tiny_fly_model):
    """Verifies that silent/inactive neural circuits receive EXACTLY ZERO weight change,

    providing mathematical immunity to catastrophic forgetting under three-factor rules.
    """
    model = tiny_fly_model
    d_model = getattr(model, "d_model", model.embedding.embedding_dim)
    eprop = EPropCreditAssignment(
        n_neurons=model.n_neurons, d_model=d_model, vocab_size=32
    )

    # Construct an eligibility state where neurons 0..9 are completely silent (z_bar = 0)
    z_bar = torch.zeros(1, 4, model.n_neurons)
    z_bar[0, :, 10:] = 1.0  # only neurons 10..19 are active
    phi = torch.ones(1, model.n_neurons)

    # Massive top-down learning signal broadcast across ALL neurons
    L = torch.full((1, model.n_neurons), 10.0)

    # Synthetic edges: some from silent pre-neurons, some from active pre-neurons
    edge_pre = torch.tensor([0, 1, 2, 10, 11, 12], dtype=torch.long)
    edge_post = torch.tensor([5, 6, 7, 15, 16, 17], dtype=torch.long)
    delay_splits = (0, 3, 4, 5, 6)

    delta_W = eprop.compute_synaptic_updates_sparse(
        L=L, phi=phi, z_bar=z_bar,
        edge_pre=edge_pre, edge_post=edge_post,
        delay_splits=delay_splits, scale=1.0
    )

    # Edges originating from silent neurons 0, 1, 2 MUST have delta_W == 0 strictly!
    assert (delta_W[:3] == 0.0).all(), "Silent circuits must receive zero weight update!"
    # Active circuits receive nonzero updates
    assert (delta_W[3:] != 0.0).all(), "Active circuits must receive nonzero credit assignment!"


def test_dales_law_and_bounded_plasticity(tiny_fly_model):
    """Verifies that online whole-brain updates strictly preserve Dale's law."""
    model = tiny_fly_model
    trainer = StreamingConnectomeTrainer(model, lr_readout=1e-2, lr_synapse=1e-2)

    h = torch.zeros(1, model.n_neurons)
    ring = tuple(torch.zeros(1, model.n_neurons) for _ in range(4))
    syn_state = {
        "ge": torch.zeros(1, model.n_neurons),
        "gi": torch.zeros(1, model.n_neurons),
        "b": torch.zeros(1, model.n_neurons),
        "x": torch.ones(1, model.n_neurons),
        "u": model.get_stp_params()[0].clone(),
    }

    # Run multiple aggressive update steps
    for t in range(20):
        in_tok = torch.tensor([t % 32])
        tgt_tok = torch.tensor([(t + 1) % 32])
        _, h, ring, syn_state = trainer.train_streaming_step(
            in_tok, tgt_tok, h, ring, syn_state, update_synapses=True
        )

    # Dale's law invariants: Exc >= 0, Inh >= 0 (magnitudes)
    assert (model.edge_weight_e >= 0.0).all()
    assert (model.edge_weight_i >= 0.0).all()
    assert torch.isfinite(model.edge_weight_e).all()
    assert torch.isfinite(model.edge_weight_i).all()
    assert torch.isfinite(model.output_read.weight).all()


def test_internal_neurons_receive_dopamine_and_feedback_signal(tiny_fly_model):
    """Verifies that internal neurons (read_mask == 0) receive non-zero third factors."""
    model = tiny_fly_model
    d_model = getattr(model, "d_model", model.embedding.embedding_dim)
    eprop = EPropCreditAssignment(
        n_neurons=model.n_neurons, d_model=d_model, vocab_size=32,
        dopamine_coupling=0.2, fa_coupling=0.2
    )

    logits = torch.randn(1, 32)
    target = torch.tensor([5])
    # Assume only neuron 19 is a readout neuron
    read_mask = torch.zeros(model.n_neurons)
    read_mask[19] = 1.0

    # Dopamine edges: from DAN neuron 0 to internal neurons 1..5
    dan_pre = torch.tensor([0, 0, 0, 0, 0], dtype=torch.long)
    dan_post = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    dan_weight = torch.ones(5, dtype=torch.float32)

    L_total, error_read, loss = eprop.compute_learning_signal_whole_brain(
        logits=logits,
        target_ids=target,
        w_read=model.output_read.weight,
        w_decoder=model.decoder.weight,
        read_mask=read_mask,
        dan_edge_pre=dan_pre,
        dan_edge_post=dan_post,
        dan_edge_weight=dan_weight,
    )

    # 1. Readout neuron 19 has nonzero signal
    assert L_total[0, 19] != 0.0
    # 2. Internal neurons (where read_mask == 0) MUST have nonzero signal via DFA + Dopamine!
    assert (L_total[0, :19] != 0.0).any(), "Internal circuits must receive non-zero learning signals!"
    # Target neurons of dopamine (1..5) must receive modulatory signal
    assert (L_total[0, 1:6].abs() > 0.0).all()


def test_coba_driving_force_sensitivity():
    """Verifies that COBA sensitivity kappa_E > 0 and kappa_I < 0 flip signs correctly."""
    eprop = EPropCreditAssignment(n_neurons=10, d_model=4, vocab_size=8, E_E=0.0, E_I=-0.2)
    state = EPropEligibilityState.init_zero(batch_size=1, n_neurons=10, device="cpu")

    # Hyperpolarized voltage V = -0.1 (between E_I and E_E)
    post_voltage = torch.full((1, 10), -0.1)
    effective_thresh = torch.full((1, 10), 0.1)
    beta_a = torch.full((1, 10), 0.05)
    alpha_m = torch.full((1, 10), 0.95)
    rho_a = torch.full((1, 10), 0.98)
    pulses = torch.ones(1, 10)

    phi_e, phi_i, _ = eprop.update_eligibility_coba(
        state=state,
        presynaptic_pulses=pulses,
        post_voltage=post_voltage,
        effective_threshold=effective_thresh,
        beta_adaptation=beta_a,
        alpha_membrane=alpha_m,
        rho_a=rho_a,
        g_e_gain=1.0,
        g_i_gain=1.0,
    )

    # For V = -0.1:
    # E_E - V = 0.0 - (-0.1) = +0.1 > 0  => phi_e > 0
    # E_I - V = -0.2 - (-0.1) = -0.1 < 0 => phi_i < 0
    assert (phi_e > 0.0).all(), "Excitatory sensitivity must be positive!"
    assert (phi_i < 0.0).all(), "Inhibitory sensitivity must be negative (Ohmic reversal driving force)!"


def test_delayed_pulses_tier_independence_no_cascade():
    """Verifies that each delay tier is updated by its authentic delayed pulse without filter cascading."""
    eprop = EPropCreditAssignment(n_neurons=8, d_model=4, vocab_size=8)
    state = EPropEligibilityState.init_zero(batch_size=1, n_neurons=8, device="cpu")

    # Pulse ONLY arrives at delay tier 2 (delay = 3 steps)
    p0 = torch.zeros(1, 8)
    p1 = torch.zeros(1, 8)
    p2 = torch.full((1, 8), 2.5)  # amplitude 2.5
    p3 = torch.zeros(1, 8)
    delayed_pulses = (p0, p1, p2, p3)

    v_pre = torch.zeros(1, 8)
    theta = torch.full((1, 8), 0.1)

    phi_e, phi_i, z_bar = eprop.update_eligibility_coba(
        state=state,
        delayed_pulses=delayed_pulses,
        v_pre=v_pre,
        effective_threshold=theta,
        trace_decay=0.9,
    )

    # Slot 2 MUST have exact amplitude 2.5
    assert torch.allclose(state.z_bar[:, 2], p2), "Slot 2 must receive its authentic pulse amplitude!"
    # Slots 0, 1, 3 MUST remain strictly zero (no cascade from/to other tiers!)
    assert (state.z_bar[:, 0] == 0.0).all(), "Slot 0 must not receive leakage from other tiers!"
    assert (state.z_bar[:, 1] == 0.0).all(), "Slot 1 must not receive leakage from other tiers!"
    assert (state.z_bar[:, 3] == 0.0).all(), "Slot 3 must not receive leakage from other tiers!"


def test_prereset_voltage_sensitivity_vs_postreset():
    """Verifies that pre-spike voltage V_pre captures firing sensitivity psi ~ 1.0,

    whereas post-reset zero voltage would falsely suppress firing neuron gradients.
    """
    eprop = EPropCreditAssignment(n_neurons=4, d_model=4, vocab_size=8)
    state = EPropEligibilityState.init_zero(batch_size=1, n_neurons=4, device="cpu")

    # Threshold is 0.15. Firing neuron has pre-spike voltage 0.155 (right at/above threshold)
    theta = torch.full((1, 4), 0.15)
    v_pre_firing = torch.full((1, 4), 0.155)
    v_post_reset = torch.zeros(1, 4)  # after reset: h = 0

    psi_prereset = eprop.compute_surrogate_derivative(v_pre_firing, theta)
    psi_postreset = eprop.compute_surrogate_derivative(v_post_reset, theta)

    # Pre-reset sensitivity must be near peak (approx 1.0)
    assert (psi_prereset > 0.99).all(), f"Pre-reset sensitivity must be near 1.0, got {psi_prereset}"
    # Post-reset sensitivity is significantly degraded
    assert (psi_postreset < 0.85).all(), f"Post-reset sensitivity should be suppressed, got {psi_postreset}"
    assert (psi_prereset > psi_postreset).all(), "Pre-reset voltage must provide strictly higher sensitivity than reset zero!"


def test_dynamic_coba_conductance_decay():
    """Verifies that high synaptic conductance accelerates membrane leak decay alpha_eff = exp(-g_total)."""
    eprop = EPropCreditAssignment(n_neurons=4, d_model=4, vocab_size=8)
    state_quiescent = EPropEligibilityState.init_zero(batch_size=1, n_neurons=4, device="cpu")
    state_barrage = EPropEligibilityState.init_zero(batch_size=1, n_neurons=4, device="cpu")

    state_quiescent.z_bar.fill_(1.0)
    state_barrage.z_bar.fill_(1.0)

    # Quiescent state: low conductance -> high retention alpha_eff = 0.95
    # High conductance barrage: g_total = 2.0 -> alpha_eff = exp(-2.0) = 0.135
    alpha_quiescent = torch.full((1, 4), 0.95)
    alpha_barrage = torch.full((1, 4), 0.135)

    zero_pulses = (torch.zeros(1, 4), torch.zeros(1, 4), torch.zeros(1, 4), torch.zeros(1, 4))
    v = torch.zeros(1, 4)
    th = torch.full((1, 4), 0.1)

    eprop.update_eligibility_coba(
        state=state_quiescent, delayed_pulses=zero_pulses, v_pre=v,
        effective_threshold=th, alpha_eff=alpha_quiescent,
    )
    eprop.update_eligibility_coba(
        state=state_barrage, delayed_pulses=zero_pulses, v_pre=v,
        effective_threshold=th, alpha_eff=alpha_barrage,
    )

    # Barrage state must decay much faster (shunting effect)
    assert (state_barrage.z_bar < state_quiescent.z_bar).all()
    assert torch.allclose(state_quiescent.z_bar, torch.tensor(0.95))
    assert torch.allclose(state_barrage.z_bar, torch.tensor(0.135))


def test_dopamine_receptor_occupancy_bounded():
    """Verifies that dopamine receptor occupancy q_j in [0, 1] and effective gate tilde_q in [q0, 1]."""
    from information_boltzmann.core.eprop_credit_assignment import DopamineReceptorState

    n = 20
    dop = DopamineReceptorState.init_zero(n_neurons=n, device="cpu", k_on=0.5, k_off=0.1, q_0=0.05)

    dan_indices = torch.tensor([0, 1], dtype=torch.long)
    dan_pre = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    dan_post = torch.tensor([5, 6, 7, 8], dtype=torch.long)
    dan_weight = torch.tensor([2.0, 1.5, 3.0, 0.5], dtype=torch.float32)

    # 1. Quiescent period: no DAN spikes, no activity
    spikes = torch.zeros(1, n)
    h = torch.zeros(1, n)
    tilde_q = dop.update_from_dan_activity(spikes, h, dan_indices, dan_pre, dan_post, dan_weight)

    assert (dop.q >= 0.0).all() and (dop.q <= 1.0).all()
    assert torch.allclose(tilde_q, torch.tensor(0.05))

    # 2. Burst event: DAN 0 and 1 fire action potentials
    spikes[0, 0] = 1.0
    spikes[0, 1] = 1.0
    tilde_q = dop.update_from_dan_activity(spikes, h, dan_indices, dan_pre, dan_post, dan_weight)

    # Target neurons 5, 6, 7, 8 must have elevated receptor occupancy
    assert (dop.q[5:9] > 0.0).all()
    assert (tilde_q[5:9] > 0.05).all()
    # Non-target neurons remain at floor q0
    assert torch.allclose(tilde_q[:5], torch.tensor(0.05))
    assert (tilde_q >= 0.05).all() and (tilde_q <= 1.0).all()


def test_sensory_writer_eligibility_and_plasticity():
    """Verifies forward O(1) sensitivity recursion and three-factor parameter updates."""
    from information_boltzmann.core.eprop_credit_assignment import SensoryWriterEligibilityState
    from torch import nn

    d_model = 16
    n_vis, n_chemo, n_mech = 4, 5, 6
    n_sens = n_vis + n_chemo + n_mech

    state = SensoryWriterEligibilityState.init_zero(
        d_model=d_model, device="cpu", n_vis=n_vis, n_chemo=n_chemo, n_mech=n_mech
    )

    class MockWriter(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj_vis = nn.Linear(d_model, n_vis, bias=False)
            self.proj_chemo = nn.Linear(d_model, n_chemo, bias=False)
            self.proj_mech = nn.Linear(d_model, n_mech, bias=False)

    writer = MockWriter()
    init_vis = writer.proj_vis.weight.data.clone()

    token_emb = torch.randn(1, d_model)
    gates = torch.tensor([[1.0, 1.0, 1.0]])
    alpha_sens = torch.full((n_sens,), 0.9)
    beta_int_sens = torch.full((n_sens,), 0.1)

    # Update input eligibility
    e_proj = state.update_input_eligibility(token_emb, gates, alpha_sens, beta_int_sens)
    assert e_proj.shape == (n_sens, d_model)
    assert torch.isfinite(e_proj).all()
    assert (e_proj != 0.0).any()

    # Apply Three-Factor sensory update
    L_sens = torch.full((n_sens,), 2.0)
    phi_sens = torch.ones(n_sens)
    q_sens = torch.full((n_sens,), 0.5)

    state.apply_sensory_updates_inplace(
        writer, L_sens, phi_sens, q_sens, scale=1e-3, weight_decay=0.0
    )

    # Visual projection weights must have moved in direction -scale * (q * L * phi) * E
    diff_vis = writer.proj_vis.weight.data - init_vis
    assert (diff_vis != 0.0).any()
    assert torch.isfinite(writer.proj_vis.weight.data).all()



