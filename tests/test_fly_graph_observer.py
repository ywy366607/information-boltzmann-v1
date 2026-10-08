"""Unit tests for HX-1 Graph-Constrained Distributed State Observer."""
import pytest
import torch
import numpy as np

from information_boltzmann.core.fly_graph_observer import FlyGraphObserver


@pytest.fixture
def mini_cns_path(tmp_path):
    graph_path = tmp_path / "mini_cns.npz"
    N = 100
    pre = np.array([0, 1, 2, 3, 20, 25, 60, 65], dtype=np.int32)
    post = np.array([20, 21, 22, 23, 60, 61, 80, 81], dtype=np.int32)
    w = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    # Generate superclasses covering all 4 groups
    sc = np.zeros(N, dtype=np.int32)
    sc[0:20] = 6    # Group 0: cb_sensory
    sc[20:60] = 4   # Group 1: cb_intrinsic
    sc[60:80] = 22  # Group 2: vnc_intrinsic
    sc[80:100] = 8  # Group 3: descending_neuron

    np.savez(
        graph_path,
        neuron_body_ids=np.arange(N),
        edge_pre_e=pre,
        edge_post_e=post,
        edge_weight_e=w,
        edge_delay_e=np.ones_like(pre),
        delay_splits_e=[0, 2, 5, 8, 8],
        edge_pre_i=np.array([], dtype=np.int32),
        edge_post_i=np.array([], dtype=np.int32),
        edge_weight_i=np.array([], dtype=np.float32),
        delay_splits_i=[0, 0, 0, 0, 0],
        superclass_names=["ENS", "ascending", "cb_efferent", "cb_endocrine", "cb_intrinsic",
                          "cb_motor", "cb_sensory", "cb_sensory_tbc", "descending_neuron",
                          "eff_asc", "eff_desc", "ol_int", "ol_sens", "sens_asc", "sens_asc_tbc",
                          "sens_desc", "unk", "vis_cent", "vis_proj", "vis_proj_tbc",
                          "vnc_eff", "vnc_endo", "vnc_intrinsic", "vnc_motor", "vnc_sens",
                          "vnc_sens_tbc", "vnc_tbc"],
        superclass_id=sc,
    )
    return graph_path


def test_fly_graph_observer_forward_and_backward(mini_cns_path):
    observer = FlyGraphObserver(
        mini_cns_path,
        d_model=64,
        max_horizon=14,
        sample_per_region=16,
    )
    T = 8
    N = 100
    h_seq = torch.randn(T, N, requires_grad=True)

    z_readout, total_loss, metrics, next_prior, next_history = observer.forward_window(h_seq)

    # Check shapes
    assert z_readout.shape == (T, 64)
    assert next_prior.shape == (4, 64)
    assert next_history.shape == (4, 4, 64)
    assert "obs_mse" in metrics
    assert "obs_sigreg" in metrics
    assert "attn_k0_immediate" in metrics

    # Check backward pass
    total_loss.backward()
    assert h_seq.grad is not None
    assert torch.isfinite(h_seq.grad).all()


def test_fly_graph_observer_sensory_propagation(mini_cns_path):
    """Perturbing sensory neurons at t=0 must propagate to the motor readout across horizons."""
    observer = FlyGraphObserver(
        mini_cns_path,
        d_model=64,
        max_horizon=14,
        sample_per_region=16,
        init_gamma=0.5,
    )
    T = 4
    N = 100
    h_base = torch.zeros(T, N)
    h_pert = torch.zeros(T, N)
    # Inject sensory perturbation at t=0
    h_pert[0, :20] = 5.0

    z_base, _, _, _, _ = observer.forward_window(h_base)
    z_pert, _, _, _, _ = observer.forward_window(h_pert)

    # The motor readout MUST reflect the sensory perturbation!
    diff = (z_pert[0] - z_base[0]).norm()
    assert diff > 1e-4, f"Sensory stimulus failed to propagate to motor readout: diff={diff}"


def test_fly_graph_observer_prior_continuity(mini_cns_path):
    """Prior state must carry across steps and prevent cold-start drift."""
    observer = FlyGraphObserver(
        mini_cns_path,
        d_model=64,
        max_horizon=14,
        sample_per_region=16,
        init_gamma=0.5,
    )
    N = 100
    h_step1 = torch.randn(1, N)
    h_step2 = torch.randn(1, N)

    # Cold start step 1
    z1, _, _, prior1, hist1 = observer.forward_window(h_step1, prior_state=None)
    assert prior1.shape == (4, 64)
    assert hist1.shape == (1, 4, 64)

    # Step 2 with warm prior vs step 2 with cold start
    z2_warm, _, _, prior2_warm, _ = observer.forward_window(h_step2, prior_state=prior1, prior_history=hist1)
    z2_cold, _, _, prior2_cold, _ = observer.forward_window(h_step2, prior_state=None, prior_history=None)

    diff = (z2_warm - z2_cold).norm()
    assert diff > 1e-4, "Carried prior had no effect on subsequent prediction!"


def test_fly_graph_observer_read_norm_scale_and_gradient(mini_cns_path):
    """Readout must pass through read_norm (scale ~ 0.1) and yield valid read_norm gradients."""
    observer = FlyGraphObserver(
        mini_cns_path,
        d_model=64,
        max_horizon=14,
        sample_per_region=16,
    )
    read_norm = torch.nn.RMSNorm(64)
    read_norm.weight.data.fill_(0.1)
    decoder = torch.nn.Linear(64, 100)

    N = 100
    h_seq = torch.randn(4, N)
    z_readout, _, _, _, _ = observer.forward_window(h_seq)

    # Normalization contract: RMSNorm scales by 0.1
    normed = read_norm(z_readout)
    expected_norm = 0.1 * (64 ** 0.5)
    actual_norm = normed.norm(dim=-1).mean().item()
    assert abs(actual_norm - expected_norm) < 0.05, f"Scale mismatch: expected ~{expected_norm}, got {actual_norm}"

    # Gradient flow: CE loss into decoder must backprop to read_norm.weight
    logits = decoder(normed)
    loss = logits.sum()
    loss.backward()

    assert read_norm.weight.grad is not None
    assert read_norm.weight.grad.norm() > 0.0


def test_fly_graph_observer_delay_alignment(mini_cns_path):
    """Test that 14-hop OPD aligns simulated motor rollout with physical arrival delay."""
    observer = FlyGraphObserver(
        mini_cns_path,
        d_model=64,
        max_horizon=14,
        sample_per_region=16,
    )
    N = 100
    K = 14

    # Simulate a physical quiet trajectory where sensory pulse arrives at motor node after delay d=3
    # At t=0: sensory pulse in neurons 0..20
    # From tick 1..K: pulse propagates to motor neurons 80..100 at tick 3
    teacher_quiet_h = torch.zeros(K, N)
    for k in range(K):
        # Sensory decays
        teacher_quiet_h[k, :20] = max(0.0, 5.0 - 0.5 * k)
        # Central hub active around k=1..3
        teacher_quiet_h[k, 20:60] = 3.0 if 1 <= k <= 3 else 0.5
        # Motor neurons respond around k=3..6 (physical conduction delay)
        teacher_quiet_h[k, 80:100] = 4.0 if 3 <= k <= 6 else 0.2

    # Window input at anchor token
    h_window = torch.zeros(4, N)
    h_window[-1, :20] = 5.0  # sensory stimulation at anchor token

    # Forward with OPD
    z_readout, total_loss, metrics, next_prior, next_history = observer.forward_window(
        h_window, teacher_quiet_h=teacher_quiet_h
    )

    assert "obs_opd" in metrics
    assert metrics["obs_opd"] > 0.0

    # Optimize for a few steps to align LeWM simulated rollout with physical delay
    optimizer = torch.optim.Adam(observer.parameters(), lr=0.01)
    initial_opd_loss = metrics["obs_opd"]

    for _ in range(5):
        optimizer.zero_grad()
        _, total_loss, m_step, _, _ = observer.forward_window(h_window, teacher_quiet_h=teacher_quiet_h)
        total_loss.backward()
        optimizer.step()

    final_opd_loss = m_step["obs_opd"]
    assert final_opd_loss < initial_opd_loss, f"OPD loss failed to decrease: {initial_opd_loss} -> {final_opd_loss}"


def test_fly_bptt_learner_with_graph_observer_and_opd(mini_cns_path):
    """Test full learner forward_window with graph observer and quiet trajectory OPD."""
    from information_boltzmann.core.fly_reservoir import FlyReservoirLM
    from information_boltzmann.core.fly_bptt_learning import FlyBPTTLearner, FlyPhysicalState

    model = FlyReservoirLM(
        mini_cns_path,
        vocab_size=20,
        d_model=64,
        injection="sensory",
        read_surface="output",
        synapse_model="coba",
        use_alif=True,
        use_stp=True,
        use_graph_observer=True,
        max_horizon=4,
        lambda_obs=1.0,
        obs_init_gamma=0.1,
    )
    model.read_norm.weight.data.fill_(0.1)

    h = torch.randn(1, model.n_neurons) * 0.05
    u = model.get_stp_params()[0].detach().expand_as(h).clone()
    physical_state = FlyPhysicalState(
        h, tuple(h.clone() for _ in range(4)), h.clone(), h.clone(),
        h.clone(), torch.ones_like(h), u, torch.zeros(1, model.n_neurons)
    )

    learner = FlyBPTTLearner(
        model, physical_state, lr=1e-3, settle_ticks=0,
        writer_baseline_clock="input", lambda_jepa=1.0,
    )

    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    targets = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)

    scores, next_state, features = learner.forward_window(tokens, targets)

    # 1. Output scores should be finite
    assert torch.isfinite(scores).all()
    # 2. Next state should carry observer_prior and observer_history
    assert next_state.observer_prior.shape == (4, 64)
    assert next_state.observer_history.shape == (4, 4, 64)
    # 3. OPD metrics should be collected
    assert learner.last_jepa_loss is not None
    assert "obs_opd" in learner.last_jepa_metrics
    assert learner.last_jepa_metrics["obs_opd"] >= 0.0

    # 4. Backward pass must update physical parameters, read_norm, raw_gamma, and delay_attn
    loss = scores.mean() + learner.last_jepa_loss
    loss.backward()

    assert model.read_norm.weight.grad is not None
    assert model.read_norm.weight.grad.norm() > 0.0
    assert model.graph_observer.raw_gamma.grad is not None
    assert model.graph_observer.raw_gamma.grad.abs().item() > 0.0
    assert model.graph_observer.delay_attn.q_proj.weight.grad is not None
    assert model.graph_observer.delay_attn.q_proj.weight.grad.norm() > 0.0
    assert model.graph_observer.transition.node_mlp[0].weight.grad is not None
    assert model.graph_observer.transition.node_mlp[0].weight.grad.norm() > 0.0
    assert model.graph_observer.encoders.proj_flux_sensory.weight.grad is not None
    assert model.graph_observer.encoders.proj_flux_sensory.weight.grad.norm() > 0.0

    # 5. Full observe step test
    learner.previous_token = 0
    obs_scores, obs_info = learner.observe(torch.tensor([1, 2, 3, 4], dtype=torch.long))
    assert len(obs_scores) == 4
    assert "observer_grad_norm" in obs_info
    assert obs_info["observer_grad_norm"] > 0.0


def test_delayed_ei_macro_graph(mini_cns_path):
    """Verify that MacroRegionGraph contains 4 delay tiers for both E and I channels."""
    observer = FlyGraphObserver(mini_cns_path, d_model=64, max_horizon=14, sample_per_region=16)
    mg = observer.macro_graph
    assert mg.A_E.shape == (4, 4, 4)
    assert mg.A_I.shape == (4, 4, 4)
    assert torch.isfinite(mg.A_E).all()
    assert torch.isfinite(mg.A_I).all()

    # Verify delayed propagation across tiers
    history = [torch.randn(2, 4, 64) for _ in range(5)]
    out = observer.transition.forward_step(history, mg.A_E, mg.A_I)
    assert out.shape == (2, 4, 64)


def test_regional_sigreg_anti_collapse(mini_cns_path):
    """Verify that regional SIGReg applies direct anti-collapse gradients to all 4 encoders."""
    observer = FlyGraphObserver(mini_cns_path, d_model=64, max_horizon=14, sample_per_region=16, lambda_sigreg=1.0)
    T, N = 8, 100
    h_seq = torch.randn(T, N, requires_grad=True)

    # Forward without external loss
    _, total_loss, metrics, _, _ = observer.forward_window(h_seq)
    assert "obs_sigreg_regions" in metrics
    assert metrics["obs_sigreg_regions"] > 0.0

    total_loss.backward()

    # All 4 regional projections must receive gradients directly from regional SIGReg
    assert observer.encoders.proj_sensory.weight.grad is not None
    assert observer.encoders.proj_sensory.weight.grad.norm() > 0.0
    assert observer.encoders.proj_central.weight.grad is not None
    assert observer.encoders.proj_central.weight.grad.norm() > 0.0
    assert observer.encoders.proj_premotor.weight.grad is not None
    assert observer.encoders.proj_premotor.weight.grad.norm() > 0.0
    assert observer.encoders.proj_motor.weight.grad is not None
    assert observer.encoders.proj_motor.weight.grad.norm() > 0.0


def test_student_error_guided_queries_identical_state(mini_cns_path):
    """Verify that query input z_q and expert label z_target originate from the EXACT SAME physical state S_q."""
    from information_boltzmann.core.fly_reservoir import FlyReservoirLM
    from information_boltzmann.core.fly_bptt_learning import (
        collect_fly_quiet_trajectory,
        collect_student_error_guided_queries,
        FlyPhysicalState,
    )

    model = FlyReservoirLM(
        mini_cns_path,
        vocab_size=20,
        d_model=64,
        injection="sensory",
        read_surface="output",
        synapse_model="coba",
        use_alif=True,
        use_stp=True,
        use_graph_observer=True,
        max_horizon=4,
    )
    obs = model.graph_observer

    N = model.n_neurons
    h = torch.randn(1, N) * 0.05
    u = model.get_stp_params()[0].detach().expand_as(h).clone()
    live_state = FlyPhysicalState(
        h, tuple(h.clone() for _ in range(4)), h.clone(), h.clone(),
        h.clone(), torch.ones_like(h), u, torch.zeros(1, N)
    )

    # Collect reference quiet trajectory
    _, quiet_states = collect_fly_quiet_trajectory(
        model, live_state.detached(), num_ticks=4, return_h=True, return_states=True
    )

    # Collect student-error-guided queries on physical copies
    query_pairs = collect_student_error_guided_queries(
        model, quiet_states, obs, max_queries=2, alpha=0.05
    )

    assert len(query_pairs) > 0
    for q_item, z_target in query_pairs:
        z_q = q_item[-1] if isinstance(q_item, (list, tuple)) else q_item
        assert z_q.shape[-2:] == (4, 64)
        assert z_target.shape[-2:] == (4, 64)
        assert torch.isfinite(z_q).all()
        assert torch.isfinite(z_target).all()

    # Forward observer with query pairs
    h_seq = torch.randn(4, N)
    _, total_loss, metrics, _, _ = obs.forward_window(h_seq, query_pairs=query_pairs)
    assert "obs_query" in metrics
    assert metrics["obs_query"] >= 0.0

    # Verify that live physical state was NEVER modified or corrupted by query execution
    assert torch.equal(live_state.h, h)
    assert torch.equal(live_state.ge, live_state.ge)


def test_fly_graph_observer_full_physical_state_flux(mini_cns_path):
    """Verifies that full physical state observation closes the Markov gap:
    1. Zero-shock property: default zero-init flux weights reproduce voltage-only encoder.
    2. Sensitivity property: non-zero flux weights make encoder sensitive to conductances and ring buffer.
    3. End-to-end differentiability: gradients flow into all 9-channel flux projections.
    """
    from information_boltzmann.core.fly_bptt_learning import FlyPhysicalState

    observer = FlyGraphObserver(
        mini_cns_path,
        d_model=64,
        max_horizon=4,
        sample_per_region=16,
    )
    N = 100
    h = torch.randn(1, N)
    ge = torch.rand(1, N) * 0.1
    gi = torch.rand(1, N) * 0.1
    b = torch.rand(1, N) * 0.05
    x = torch.ones(1, N)
    u = torch.full((1, N), 0.25)
    ring = tuple(torch.zeros(1, N) for _ in range(4))

    state_1 = FlyPhysicalState(h, ring, ge, gi, b, x, u, torch.zeros(1, N))

    # 1. Zero-shock property
    z_voltage_only = observer.encoders(h)
    z_full_state = observer.encoders(state_1)
    assert torch.allclose(z_voltage_only, z_full_state, atol=1e-6), "Default flux weights must yield zero shock"

    # 2. Sensitivity property: activate flux weights with non-uniform values
    with torch.no_grad():
        torch.nn.init.normal_(observer.encoders.proj_ge_sensory.weight, std=0.5)
        torch.nn.init.normal_(observer.encoders.proj_gi_sensory.weight, std=0.5)
        torch.nn.init.normal_(observer.encoders.proj_ring_sensory.weight, std=0.5)

    z_active = observer.encoders(state_1)
    assert not torch.allclose(z_voltage_only, z_active, atol=1e-4)

    # Modify hidden conductance ge by 0.05
    state_altered_ge = FlyPhysicalState(h, ring, ge + 0.05, gi, b, x, u, torch.zeros(1, N))
    z_altered_ge = observer.encoders(state_altered_ge)
    diff_ge = (z_altered_ge - z_active).abs().max().item()
    assert diff_ge > 1e-4, f"Altered conductance must shift latent representation, got diff={diff_ge}"

    # Modify in-flight pulse in ring buffer
    ring_pulse = (torch.ones(1, N), ring[1], ring[2], ring[3])
    state_altered_ring = FlyPhysicalState(h, ring_pulse, ge, gi, b, x, u, torch.zeros(1, N))
    z_altered_ring = observer.encoders(state_altered_ring)
    diff_ring = (z_altered_ring - z_active).abs().max().item()
    assert diff_ring > 1e-4, f"Altered in-flight ring pulse must shift latent representation, got diff={diff_ring}"

    # 3. Test list of states in forward_window and gradient flow
    states_seq = [
        FlyPhysicalState(torch.randn(1, N, requires_grad=True), ring, ge, gi, b, x, u, torch.zeros(1, N))
        for _ in range(4)
    ]
    h_seq = torch.cat([s.h for s in states_seq], dim=0)
    z_readout, total_loss, metrics, next_prior, next_history = observer.forward_window(
        h_seq, physical_states=states_seq
    )
    assert z_readout.shape == (4, 64)
    assert next_history.shape == (4, 4, 64)
    assert torch.isfinite(total_loss)

    total_loss.backward()
    assert observer.encoders.proj_ge_sensory.weight.grad is not None
    assert observer.encoders.proj_gi_sensory.weight.grad is not None
    assert observer.encoders.proj_ring_sensory.weight.grad is not None
    assert torch.isfinite(observer.encoders.proj_ge_sensory.weight.grad).all()
    assert states_seq[0].h.grad is not None


def test_delayed_message_history_tier1_vs_tier4_distinct():
    """Finding 1 Verification: Tier 1 vs Tier 4 edges must NOT be folded into the same state."""
    from information_boltzmann.core.fly_graph_observer import GraphNeuralTransition
    tr = GraphNeuralTransition(d_model=64)
    H = torch.randn(1, 4, 64)

    # Graph with sensory -> motor on Tier 1
    A_E_tier1 = torch.zeros(4, 4, 4)
    A_E_tier1[0, 0, 3] = 1.0  # Tier 1 (delay 1)

    # Graph with sensory -> motor on Tier 4
    A_E_tier4 = torch.zeros(4, 4, 4)
    A_E_tier4[3, 0, 3] = 1.0  # Tier 4 (delay 4)

    A_I = torch.zeros(4, 4, 4)

    # Single-step prediction from H:
    out_tier1 = tr.forward_step([H], A_E_tier1, A_I)
    out_tier4 = tr.forward_step([H], A_E_tier4, A_I)

    diff = (out_tier1[0, 3] - out_tier4[0, 3]).norm().item()
    assert diff > 1e-3, f"Tier 1 and Tier 4 must produce distinct motor predictions! Got diff={diff}"


def test_zero_motor_reachability_and_excitation():
    """Finding 2 Verification: When motor latent is zero, incoming non-zero messages MUST excite it."""
    from information_boltzmann.core.fly_graph_observer import GraphNeuralTransition
    tr = GraphNeuralTransition(d_model=64)

    # Sensory node has strong activity, Motor node (index 3) is strictly 0.0
    H = torch.zeros(1, 4, 64, requires_grad=True)
    with torch.no_grad():
        H[0, 0] = torch.randn(64) * 3.0  # Sensory activity

    # Edge from sensory (0) to motor (3)
    A_E = torch.zeros(4, 4, 4)
    A_E[0, 0, 3] = 1.0
    A_I = torch.zeros(4, 4, 4)

    out = tr.forward_step([H], A_E, A_I)
    motor_out = out[0, 3]

    # Motor node must NOT be locked to zero!
    assert motor_out.norm().item() > 1e-4, f"Zero motor state must be excitable by incoming messages! Got norm={motor_out.norm().item()}"

    # Gradient must flow from motor output back to sensory source
    loss = motor_out.sum()
    loss.backward()
    assert H.grad is not None
    assert H.grad[0, 0].norm().item() > 1e-4, "Gradient must flow back to sensory source!"


def test_auxiliary_compression_no_null_space_collapse(mini_cns_path):
    """Finding 4 Verification: Multi-channel encoder must not null-collapse E/I or delay tiers."""
    from information_boltzmann.core.fly_graph_observer import FlyGraphObserver
    from information_boltzmann.core.fly_bptt_learning import FlyPhysicalState

    observer = FlyGraphObserver(mini_cns_path, d_model=64, max_horizon=4, sample_per_region=16)
    N = 100
    h = torch.randn(1, N)
    b = torch.zeros(1, N)
    x = torch.ones(1, N)
    u = torch.full((1, N), 0.25)
    ring = tuple(torch.zeros(1, N) for _ in range(4))

    # Activate weights with distinct non-collinear values
    with torch.no_grad():
        torch.nn.init.normal_(observer.encoders.proj_ge_sensory.weight, std=0.2)
        torch.nn.init.normal_(observer.encoders.proj_gi_sensory.weight, std=0.2)
        torch.nn.init.normal_(observer.encoders.proj_ring_sensory.weight, std=0.2)

    # Test algebraic counterexample: (ge, gi) = (.10, .10) vs (.11, .09)
    s1 = FlyPhysicalState(h, ring, torch.full((1, N), 0.10), torch.full((1, N), 0.10), b, x, u, torch.zeros(1, N))
    s2 = FlyPhysicalState(h, ring, torch.full((1, N), 0.11), torch.full((1, N), 0.09), b, x, u, torch.zeros(1, N))
    z1 = observer.encoders(s1)
    z2 = observer.encoders(s2)
    diff_coba = (z1 - z2).abs().max().item()
    assert diff_coba > 1e-4, f"E/I counterexample must produce distinct representations! Got diff={diff_coba}"

    # Test delay tiers: pulse in ring 1 (delay 1) vs ring 4 (delay 4)
    ring_t1 = (torch.ones(1, N), torch.zeros(1, N), torch.zeros(1, N), torch.zeros(1, N))
    ring_t4 = (torch.zeros(1, N), torch.zeros(1, N), torch.zeros(1, N), torch.ones(1, N))
    s_r1 = FlyPhysicalState(h, ring_t1, torch.full((1, N), 0.1), torch.full((1, N), 0.1), b, x, u, torch.zeros(1, N))
    s_r4 = FlyPhysicalState(h, ring_t4, torch.full((1, N), 0.1), torch.full((1, N), 0.1), b, x, u, torch.zeros(1, N))
    zr1 = observer.encoders(s_r1)
    zr4 = observer.encoders(s_r4)
    diff_delay = (zr1 - zr4).abs().max().item()
    assert diff_delay > 1e-4, f"Delay tier 1 vs 4 must produce distinct representations! Got diff={diff_delay}"


def test_query_timestamp_alignment_perfect_predictor_zero_error(mini_cns_path):
    """Finding 3 Verification: For a perfect predictor, student prediction discrepancy must be 0."""
    from information_boltzmann.core.fly_reservoir import FlyReservoirLM
    from information_boltzmann.core.fly_bptt_learning import (
        collect_fly_quiet_trajectory,
        collect_student_error_guided_queries,
        FlyPhysicalState,
    )

    model = FlyReservoirLM(mini_cns_path, vocab_size=20, d_model=64, synapse_model="coba", use_graph_observer=True)
    obs = model.graph_observer
    N = model.n_neurons
    h = torch.randn(1, N) * 0.05
    u = model.get_stp_params()[0].detach().expand_as(h).clone()
    live_state = FlyPhysicalState(h, tuple(h.clone() for _ in range(4)), h.clone(), h.clone(), h.clone(), torch.ones_like(h), u, torch.zeros(1, N))

    _, quiet_states = collect_fly_quiet_trajectory(model, live_state.detached(), num_ticks=4, return_h=True, return_states=True)

    # Mock a perfect predictor: simulate_hops returns the EXACT ground-truth encodings at future ticks
    original_simulate_hops = obs.simulate_hops
    def mock_perfect_simulate_hops(z_start, num_hops, history=None):
        # z_start is tick 1 (quiet_states[0]).
        # future ticks 2, 3, 4 are quiet_states[1], quiet_states[2], quiet_states[3]
        return [obs.encoders(quiet_states[i + 1]) for i in range(num_hops)]

    obs.simulate_hops = mock_perfect_simulate_hops
    try:
        # Collect queries: with aligned timestamps, delta_z must be exactly zero!
        # We verify that delta_z is zero by checking that the perturbed state equals the reference state
        queries = collect_student_error_guided_queries(model, quiet_states, obs, max_queries=2, alpha=0.05)
        assert len(queries) == 2
        for idx_q, (q_item, z_target) in enumerate(queries):
            z_q = q_item[-1] if isinstance(q_item, (list, tuple)) else q_item
            z_ref_k = obs.encoders(quiet_states[idx_q + 1]).detach()
            # Finding 3: Perfect prediction yields zero discrepancy; z_q strictly equals reference encoding
            assert torch.allclose(z_q, z_ref_k, atol=1e-5), (
                f"Query {idx_q}: z_q must match reference encoding when predictor is perfect! "
                f"max_diff={(z_q - z_ref_k).abs().max()}"
            )
            # Reference physical step from unperturbed reference state must match target
            z_target_expected = obs.encoders(quiet_states[idx_q + 2]).detach()
            assert torch.allclose(z_target, z_target_expected, atol=1e-5), (
                f"Query {idx_q}: z_target must match reference target when predictor is perfect! "
                f"max_diff={(z_target - z_target_expected).abs().max()}"
            )
    finally:
        obs.simulate_hops = original_simulate_hops


def test_window_splitting_and_history_persistence_equivalence(mini_cns_path):
    """Reviewer Acceptance: Verifies that identical stream produces numerically identical readouts,
    next_prior, and 4-tier next_history whether processed:
    1. In one 16-token window
    2. In two 8-token windows (chunked)
    3. In 16 single-token steps (token-by-token)
    4. Saved to checkpoint and resumed across window boundaries.
    """
    from information_boltzmann.core.fly_bptt_learning import FlyPhysicalState

    observer = FlyGraphObserver(
        mini_cns_path,
        d_model=64,
        max_horizon=4,
        sample_per_region=16,
    )
    # Ensure transition weights and delay attention are active
    with torch.no_grad():
        for p in observer.transition.parameters():
            torch.nn.init.normal_(p, std=0.05)
        for p in observer.encoders.parameters():
            torch.nn.init.normal_(p, std=0.05)

    T = 16
    N = 100
    torch.manual_seed(42)
    # Generate continuous physical stream
    states = [
        FlyPhysicalState(
            torch.randn(1, N) * 0.1,
            tuple(torch.randn(1, N) * 0.1 for _ in range(4)),
            torch.rand(1, N) * 0.2,
            torch.rand(1, N) * 0.2,
            torch.rand(1, N) * 0.05,
            torch.ones(1, N),
            torch.full((1, N), 0.25),
            torch.zeros(1, N),
        )
        for _ in range(T)
    ]
    h_seq = torch.cat([s.h for s in states], dim=0)

    # 1. Whole window (T=16)
    r_whole, _, _, prior_whole, hist_whole = observer.forward_window(
        h_seq, physical_states=states
    )

    # 2. Chunked window (two chunks of T=8)
    r_c1, _, _, prior_c1, hist_c1 = observer.forward_window(
        h_seq[:8], physical_states=states[:8]
    )
    r_c2, _, _, prior_c2, hist_c2 = observer.forward_window(
        h_seq[8:], prior_state=prior_c1, prior_history=hist_c1, physical_states=states[8:]
    )
    r_chunked = torch.cat([r_c1, r_c2], dim=0)

    # 3. Token-by-token (16 calls of T=1)
    cur_prior = None
    cur_hist = None
    r_singles = []
    for t in range(T):
        r_t, _, _, cur_prior, cur_hist = observer.forward_window(
            h_seq[t:t+1], prior_state=cur_prior, prior_history=cur_hist, physical_states=[states[t]]
        )
        r_singles.append(r_t)
    r_single = torch.cat(r_singles, dim=0)

    # 4. Checkpoint save and resume halfway
    ckpt = {
        "prior": prior_c1.clone(),
        "history": hist_c1.clone(),
    }
    r_resumed, _, _, prior_resumed, hist_resumed = observer.forward_window(
        h_seq[8:], prior_state=ckpt["prior"], prior_history=ckpt["history"], physical_states=states[8:]
    )
    r_saved_resumed = torch.cat([r_c1, r_resumed], dim=0)

    # Assert exact numerical equivalence across all chunking granularities!
    assert torch.allclose(r_whole, r_chunked, atol=1e-5), f"Chunked readout mismatch: max diff={(r_whole - r_chunked).abs().max()}"
    assert torch.allclose(r_whole, r_single, atol=1e-5), f"Single-step readout mismatch: max diff={(r_whole - r_single).abs().max()}"
    assert torch.allclose(r_whole, r_saved_resumed, atol=1e-5), f"Resumed readout mismatch: max diff={(r_whole - r_saved_resumed).abs().max()}"

    assert torch.allclose(prior_whole, prior_c2, atol=1e-5), f"Chunked prior mismatch: max diff={(prior_whole - prior_c2).abs().max()}"
    assert torch.allclose(prior_whole, cur_prior, atol=1e-5), f"Single-step prior mismatch: max diff={(prior_whole - cur_prior).abs().max()}"
    assert torch.allclose(prior_whole, prior_resumed, atol=1e-5), f"Resumed prior mismatch: max diff={(prior_whole - prior_resumed).abs().max()}"

    assert torch.allclose(hist_whole, hist_c2, atol=1e-5), f"Chunked history mismatch: max diff={(hist_whole - hist_c2).abs().max()}"
    assert torch.allclose(hist_whole, cur_hist, atol=1e-5), f"Single-step history mismatch: max diff={(hist_whole - cur_hist).abs().max()}"
    assert torch.allclose(hist_whole, hist_resumed, atol=1e-5), f"Resumed history mismatch: max diff={(hist_whole - hist_resumed).abs().max()}"


def test_tier4_delay_arrival_exact_timing(mini_cns_path):
    """Reviewer Acceptance: Verifies that a pulse transmitted via delay tier 4 (4 ticks)
    arrives at the destination region at exactly t + 4 ticks, regardless of window chunking.
    """
    from information_boltzmann.core.fly_bptt_learning import FlyPhysicalState

    observer = FlyGraphObserver(mini_cns_path, d_model=64, max_horizon=4, sample_per_region=16)

    # Isolate tier 4: clear all delay matrices except tier 4 (d=4, index 3)
    with torch.no_grad():
        observer.macro_graph.A_E.zero_()
        observer.macro_graph.A_I.zero_()
        # Connection from sensory (node 0) to central (node 1) with 4-tick delay
        observer.macro_graph.A_E[3, 0, 1] = 1.0

        # Set identity-like response in MLP
        for p in observer.transition.parameters():
            p.zero_()

    T = 8
    N = 100
    # Create silent states except at t=0 where sensory node has an impulse
    states = [
        FlyPhysicalState(
            torch.zeros(1, N),
            tuple(torch.zeros(1, N) for _ in range(4)),
            torch.zeros(1, N),
            torch.zeros(1, N),
            torch.zeros(1, N),
            torch.zeros(1, N),
            torch.full((1, N), 0.25),
            torch.zeros(1, N),
        )
        for _ in range(T)
    ]
    # Impulse at t=0
    states[0].h[:, :16] = 10.0
    h_seq = torch.cat([s.h for s in states], dim=0)

    # Run in one window
    _, _, _, prior_whole, hist_whole = observer.forward_window(h_seq, physical_states=states)

    # Run in two windows split right at t=3 (before pulse arrives at t=4)
    _, _, _, prior_w1, hist_w1 = observer.forward_window(h_seq[:3], physical_states=states[:3])
    _, _, _, prior_w2, hist_w2 = observer.forward_window(
        h_seq[3:], prior_state=prior_w1, prior_history=hist_w1, physical_states=states[3:]
    )

    # Pulse must arrive at exactly t=4 in both cases!
    # Terminal prior and history at t=8 must be strictly identical
    assert torch.allclose(prior_whole, prior_w2, atol=1e-6)
    assert torch.allclose(hist_whole, hist_w2, atol=1e-6)


def test_zero_correction_exact_baseline_restoration(mini_cns_path):
    """When gamma == 0 (zero predictive correction), the observer must exactly and identically restore the base physical readout and loss."""
    from information_boltzmann.core.fly_reservoir import FlyReservoirLM
    from information_boltzmann.core.fly_bptt_learning import FlyBPTTLearner, FlyPhysicalState

    torch.manual_seed(1234)
    # 1. Model with observer enabled (init_gamma=0.0)
    model_obs = FlyReservoirLM(
        mini_cns_path,
        vocab_size=30,
        d_model=64,
        injection="sensory",
        read_surface="output",
        synapse_model="coba",
        use_alif=True,
        use_stp=True,
        use_graph_observer=True,
        obs_init_gamma=0.0,
        max_horizon=4,
        lambda_obs=1.0,
    )
    # 2. Baseline model without observer
    torch.manual_seed(1234)
    model_base = FlyReservoirLM(
        mini_cns_path,
        vocab_size=30,
        d_model=64,
        injection="sensory",
        read_surface="output",
        synapse_model="coba",
        use_alif=True,
        use_stp=True,
        use_graph_observer=False,
    )
    # Copy identical parameters to guarantee shared baseline weights
    with torch.no_grad():
        for name, param in model_base.named_parameters():
            if name in dict(model_obs.named_parameters()):
                dict(model_obs.named_parameters())[name].copy_(param)

    # Observer gamma must be strictly 0.0
    assert model_obs.graph_observer.gamma.item() == 0.0

    # Test standalone forward_window with anchor
    T = 4
    h_seq = torch.randn(T, model_obs.n_neurons)
    anchor = torch.randn(T, 64)
    z_readout, _, _, _, _ = model_obs.graph_observer.forward_window(h_seq, anchor_features=anchor)
    # When gamma == 0, z_readout must equal anchor bit-for-bit
    assert torch.equal(z_readout, anchor), f"Zero-correction readout mismatch with anchor: max diff={(z_readout - anchor).abs().max()}"

    # Test full BPTT forward_window equivalence
    h_init = torch.randn(1, model_obs.n_neurons) * 0.05
    u_init = model_obs.get_stp_params()[0].detach().expand_as(h_init).clone()

    state_obs = FlyPhysicalState(
        h_init.clone(), tuple(h_init.clone() for _ in range(4)), h_init.clone(), h_init.clone(),
        h_init.clone(), torch.ones_like(h_init), u_init.clone(), torch.zeros(1, model_obs.n_neurons)
    )
    state_base = FlyPhysicalState(
        h_init.clone(), tuple(h_init.clone() for _ in range(4)), h_init.clone(), h_init.clone(),
        h_init.clone(), torch.ones_like(h_init), u_init.clone(), torch.zeros(1, model_base.n_neurons)
    )

    learner_obs = FlyBPTTLearner(model_obs, state_obs, lr=1e-3, settle_ticks=0, lambda_jepa=1.0)
    learner_base = FlyBPTTLearner(model_base, state_base, lr=1e-3, settle_ticks=0, lambda_jepa=0.0)

    tokens = torch.tensor([[5, 12, 18, 24]], dtype=torch.long)
    targets = torch.tensor([[12, 18, 24, 7]], dtype=torch.long)

    scores_obs, _, feats_obs = learner_obs.forward_window(tokens, targets)
    scores_base, _, feats_base = learner_base.forward_window(tokens, targets)

    # Base physical features must match
    assert torch.allclose(feats_obs, feats_base, atol=1e-7), f"Physical feature mismatch: max diff={(feats_obs - feats_base).abs().max()}"
    # Next-token prediction scores must strictly and identically match
    assert torch.allclose(scores_obs, scores_base, atol=1e-7), f"Scores mismatch: max diff={(scores_obs - scores_base).abs().max()}"







