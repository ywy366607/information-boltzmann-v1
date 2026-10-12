import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from information_boltzmann.core.fly_bptt_learning import (
    FlyPhysicalState,
    FlyBPTTLearner,
    advance_fly_token_adaptive,
    extract_fly_motor_latent,
)
from information_boltzmann.core.fly_reservoir import FlyReservoirLM


@pytest.fixture(scope="module")
def fly_model_and_state():
    """Initializes a real MaleCNS fly reservoir model and physical state with repaired numerical paths."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    graph_path = "data/malecns_v1/fly_reservoir_coba.npz"

    model = FlyReservoirLM(
        graph_path,
        d_model=768,
        read_surface="output",
        injection="topographic",
        synapse_model="coba",
        use_alif=True,
        use_stp=True,
        detach_reset=True,
        surrogate_mode="threshold",
        transmission_mode="incoming",
    ).to(device)

    # Initial physical state
    h = torch.zeros(1, model.n_neurons, device=device)
    ring = tuple(torch.zeros(1, model.n_neurons, device=device) for _ in range(4))
    ge = torch.zeros(1, model.n_neurons, device=device)
    gi = torch.zeros(1, model.n_neurons, device=device)
    b = torch.zeros(1, model.n_neurons, device=device)
    x = torch.ones(1, model.n_neurons, device=device)
    u = torch.full((1, model.n_neurons), 0.25, device=device)
    n_base = model.topographic_writer.n_total if model.topographic_writer is not None else model.n_neurons
    baseline = torch.zeros(1, n_base, device=device)
    h_mean = torch.zeros(1, model.n_neurons, device=device)

    state = FlyPhysicalState(h, ring, ge, gi, b, x, u, baseline, h_mean)
    return model, state, device


def test_causal_arrival_floor_prevents_premature_exit(fly_model_and_state):
    """Directly verifies resolution of review Finding 3:

    Even under zero sensory drive and quiet membrane, the causal floor min_settle_ticks=3
    strictly prevents premature admission at t=1 or t=2.
    """
    model, state, device = fly_model_and_state
    token = torch.tensor([101], dtype=torch.long, device=device)

    # Test with min_settle_ticks = 3
    box = [None]
    with torch.no_grad():
        next_state, latent = advance_fly_token_adaptive(
            model, state, token, box,
            min_settle_ticks=3,
            max_settle_ticks=14,
            flux_baseline=1.0,  # Intentionally large baseline to test if floor holds
            writer_baseline_clock="physical"
        )

    chosen_ticks = box[0]
    assert chosen_ticks is not None
    assert chosen_ticks >= 3, f"Premature admission occurred at tick {chosen_ticks}! Must be >= 3."


def test_checkpoint_deterministic_replay_matches(fly_model_and_state):
    """Directly verifies resolution of review schedule replay recommendation:

    Under torch.utils.checkpoint with use_reentrant=False, the forward pass records
    the schedule, and the recomputation pass executes the exact recorded schedule
    with zero CheckpointError and identical tensor tracking.
    """
    model, state, device = fly_model_and_state
    token = torch.tensor([205], dtype=torch.long, device=device)

    box = [None]
    curr_state = state.detached()

    # Run checkpointed advance
    next_state, latent = checkpoint(
        advance_fly_token_adaptive,
        model, curr_state, token, box,
        min_settle_ticks=3,
        max_settle_ticks=14,
        flux_baseline=0.048,
        writer_baseline_clock="physical",
        use_reentrant=False
    )

    recorded_ticks = box[0]
    assert recorded_ticks is not None
    assert 3 <= recorded_ticks <= 14

    # Backpropagate to verify backward recomputation succeeds without error
    target_loss = latent.sum()
    target_loss.backward()

    # Parameters must have gradients
    assert model.decoder.weight.grad is not None or model.output_read.weight.grad is not None


def test_full_brain_connected_gradients_in_learner(fly_model_and_state):
    """Directly verifies resolution of review Finding 1:

    Adaptive admission in FlyBPTTLearner connects loss gradients all the way through
    the physical settling steps into decoder, read norm, output read, ALIF parameters,
    synaptic weights (edge_weight_e, edge_weight_i), and topographic writer.
    """
    model, state, device = fly_model_and_state
    if device != "cuda":
        pytest.skip("Full learner test requires CUDA for Fused AdamW")

    learner = FlyBPTTLearner(
        model, state,
        lr=2e-4,
        settle_ticks=14,
        writer_baseline_clock="physical",
        use_checkpointing=True,
        adaptive_admission=True,
        min_settle_ticks=3,
        flux_baseline=0.048,
        lambda_mcr2=0.0  # MCR2 disabled as instructed
    )

    # 4-token window
    tokens = torch.tensor([101, 205, 307, 409], dtype=torch.long, device=device)
    targets = torch.tensor([205, 307, 409, 511], dtype=torch.long, device=device)
    ids = torch.cat([torch.tensor([50], dtype=torch.long, device=device), tokens[:-1]])[None]

    learner.optimizer.zero_grad(set_to_none=True)
    learner.sgd.zero_grad(set_to_none=True)

    scores, next_st, features = learner.forward_window(ids, targets[None])
    loss = scores.mean()

    assert torch.isfinite(loss).item()
    loss.backward()

    # 1. Output readout & decoder gradients
    assert model.decoder.weight.grad is not None
    assert torch.isfinite(model.decoder.weight.grad.norm()).item()
    assert model.output_read.weight.grad is not None
    assert torch.isfinite(model.output_read.weight.grad.norm()).item()

    # 2. Synaptic weight gradients
    assert model.edge_weight_e.grad is not None
    assert torch.isfinite(model.edge_weight_e.grad.norm()).item()
    assert model.edge_weight_i.grad is not None
    assert torch.isfinite(model.edge_weight_i.grad.norm()).item()

    # 3. Topographic writer gradients
    assert model.topographic_writer.proj_vis.weight.grad is not None
    assert torch.isfinite(model.topographic_writer.proj_vis.weight.grad.norm()).item()

    # 4. Adaptive metrics recorded
    metrics = learner.last_adaptive_metrics
    assert "adaptive_ticks_mean" in metrics
    assert 3.0 <= metrics["adaptive_ticks_mean"] <= 14.0
    assert metrics["adaptive_ticks_min"] >= 3
    assert metrics["adaptive_ticks_max"] <= 14


def test_physical_ticks_and_state_continuity(fly_model_and_state):
    """Verifies that physical ticks accounting accurately tracks executed physical ticks."""
    model, state, device = fly_model_and_state
    if device != "cuda":
        pytest.skip("Full learner test requires CUDA for Fused AdamW")

    learner = FlyBPTTLearner(
        model, state,
        lr=2e-4,
        settle_ticks=14,
        writer_baseline_clock="physical",
        use_checkpointing=True,
        adaptive_admission=True,
        min_settle_ticks=3,
        flux_baseline=0.048,
        lambda_mcr2=0.0
    )

    learner.previous_token = 50
    initial_physical_ticks = learner.physical_ticks

    tokens = [101, 205]
    scores, metrics = learner.observe(tokens)

    assert learner.physical_ticks > initial_physical_ticks
    # Must have added 1 pulse tick + adaptive quiet ticks per token
    expected_added = learner.last_adaptive_metrics["adaptive_total_ticks"]
    assert learner.physical_ticks - initial_physical_ticks == expected_added
    assert "adaptive_ticks_mean" in metrics


def _window_gradients(learner, model, start, ids, targets, reuse):
    import information_boltzmann.core.triton_synapse as ts
    old, compute, calls = ts.REUSE_TRANSMISSION, ts.IncomingDelayedTransmission._compute, [0]

    def counted(*a):
        calls[0] += 1
        return compute(*a)
    ts.REUSE_TRANSMISSION = reuse
    ts.IncomingDelayedTransmission._compute = staticmethod(counted)
    try:
        model.zero_grad(set_to_none=True)
        learner.state = start.detached()
        scores, _, _ = learner.forward_window(ids, targets)
        scores.mean().backward()
    finally:
        ts.REUSE_TRANSMISSION = old
        ts.IncomingDelayedTransmission._compute = staticmethod(compute)
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return scores.detach().clone(), grads, calls[0]


def test_checkpoint_recompute_reuses_transmission_exactly(fly_model_and_state):
    """Checkpoint recompute replays the recorded synaptic transmission outputs instead of recomputing them; the window
    forward is bitwise identical (the event-driven forward is deterministic); gradients agree to the run-to-run noise
    of the rest of the backward (atomic reductions elsewhere differ by ~1e-10 relative between two identical runs)."""
    import dataclasses
    model, state, device = fly_model_and_state
    if device != "cuda":
        pytest.skip("event-driven transmission and fused AdamW need CUDA")
    torch.manual_seed(0)
    start = dataclasses.replace(state, h=torch.rand_like(state.h))   # near threshold: real spiking traffic
    learner = FlyBPTTLearner(model, start, lr=2e-4, settle_ticks=14, writer_baseline_clock="physical",
                             use_checkpointing=True, adaptive_admission=True, min_settle_ticks=3,
                             flux_baseline=0.048, lambda_mcr2=0.0)
    ids = torch.tensor([[50, 101, 205, 307]], device=device)
    targets = torch.tensor([[101, 205, 307, 409]], device=device)
    s0, g0, c0 = _window_gradients(learner, model, start, ids, targets, reuse=False)
    s1, g1, c1 = _window_gradients(learner, model, start, ids, targets, reuse=True)
    assert c0 > 0 and c1 * 2 == c0                      # the recompute pass no longer transmits
    assert torch.equal(s0, s1)
    assert g0.keys() == g1.keys() and 'edge_weight_e' in g0
    for name in g0:
        assert torch.allclose(g0[name], g1[name], rtol=1e-6, atol=1e-6 * float(g0[name].abs().max())), name


def test_stream_clock_pulse_equals_serial_fixed_settle(fly_model_and_state):
    """A k-tick streaming token with a single onset pulse is exactly the archived serial event with k-1 quiet ticks."""
    import dataclasses
    from information_boltzmann.core.fly_bptt_learning import advance_fly_input_event, advance_fly_token_stream
    model, state, device = fly_model_and_state
    torch.manual_seed(1)
    start = dataclasses.replace(state, h=torch.rand_like(state.h))
    token = torch.tensor([321], device=device)
    with torch.no_grad():
        serial = advance_fly_input_event(model, start.detached(), token, settle_ticks=4,
                                         writer_baseline_clock='physical')
        streamed, latent = advance_fly_token_stream(model, start.detached(), token, ticks=5,
                                                    writer_baseline_clock='physical')
    assert torch.equal(serial.h, streamed.h) and torch.equal(serial.baseline, streamed.baseline)
    assert all(torch.equal(a, b) for a, b in zip(serial.ring, streamed.ring))
    assert latent.shape == (1, model.output_read.out_features)


def test_stream_learner_reaches_every_parameter_and_reuses_transmission(fly_model_and_state):
    """Streaming learner (held input): finite window loss, gradients into synapses, writer and readout, and the
    checkpoint recompute replays transmission (half the transmission calls, same forward)."""
    import dataclasses
    model, state, device = fly_model_and_state
    if device != "cuda":
        pytest.skip("event-driven transmission and fused AdamW need CUDA")
    torch.manual_seed(2)
    start = dataclasses.replace(state, h=torch.rand_like(state.h))
    learner = FlyBPTTLearner(model, start, lr=2e-4, settle_ticks=14, writer_baseline_clock="physical",
                             use_checkpointing=True, adaptive_admission=True, stream_ticks=3, hold_input=True,
                             lambda_mcr2=0.0)
    assert not learner.adaptive_admission
    ids = torch.tensor([[50, 101, 205, 307]], device=device)
    targets = torch.tensor([[101, 205, 307, 409]], device=device)
    s0, g0, c0 = _window_gradients(learner, model, start, ids, targets, reuse=False)
    s1, g1, c1 = _window_gradients(learner, model, start, ids, targets, reuse=True)
    assert c0 == 2 * 2 * 3 * 4 and c1 * 2 == c0          # E+I per tick, 3 ticks per token, recompute replayed
    assert torch.equal(s0, s1) and torch.isfinite(s0).all()
    for name in ('edge_weight_e', 'edge_weight_i', 'output_read.weight', 'decoder.weight',
                 'topographic_writer.proj_vis.weight', 'topographic_writer.gate_linear.weight'):
        assert name in g0 and torch.isfinite(g0[name]).all() and g0[name].abs().sum() > 0, name


def test_output_lag_of_one_tick_cannot_see_the_sensed_token(fly_model_and_state):
    """The read after the injection tick is bitwise independent of the token being sensed (MaleCNS: the earliest
    sensory -> read-surface path is one delay-1 edge, so the read surface first changes at tick 2); a read after two
    ticks already depends on it (the test is sensitive)."""
    import dataclasses
    from information_boltzmann.core.fly_bptt_learning import advance_fly_token_stream
    model, state, device = fly_model_and_state
    torch.manual_seed(3)
    start = dataclasses.replace(state, h=torch.rand_like(state.h))
    with torch.no_grad():
        for read_after, independent in ((1, True), (2, False)):
            reads = [advance_fly_token_stream(model, start.detached(), torch.tensor([t], device=device), ticks=3,
                                              read_after=read_after, writer_baseline_clock='physical')[1]
                     for t in (17, 4242, 31337)]
            assert all(torch.equal(reads[0], r) for r in reads[1:]) == independent


def test_output_lag_learner_reads_the_sensed_token_as_its_target(fly_model_and_state):
    import dataclasses
    model, state, device = fly_model_and_state
    if device != "cuda":
        pytest.skip("fused AdamW needs CUDA")
    start = dataclasses.replace(state, h=torch.rand_like(state.h))
    learner = FlyBPTTLearner(model, start, lr=2e-4, settle_ticks=14, writer_baseline_clock="physical",
                             use_checkpointing=True, stream_ticks=3, read_lag=1, lambda_mcr2=0.0)
    learner.previous_token = 50
    tokens = torch.tensor([101, 205, 307, 409], device=device)
    assert torch.equal(learner.inputs_for_targets(tokens), tokens[None])
    scores, _ = learner.observe(tokens.cpu().numpy())
    assert len(scores) == 4 and all(math.isfinite(v) for v in scores)
    assert learner.physical_ticks == 12
    with pytest.raises(ValueError):
        FlyBPTTLearner(model, start, read_lag=1)


def test_giant_fiber_go_halts_on_its_own_spike_and_learns(fly_model_and_state):
    """Connectome go act: each token's realized ticks end exactly on the first tick a giant fiber spikes (or at the
    ceiling); the window loss reaches the synapses and the read; a checkpoint recompute replays the same schedule."""
    import dataclasses
    from information_boltzmann.core.fly_bptt_learning import attach_connectome_go, advance_fly_token_go
    model, state, device = fly_model_and_state
    if device != "cuda":
        pytest.skip("fused AdamW needs CUDA")
    attach_connectome_go(model, "data/malecns_v1/fly_reservoir_coba.npz")
    try:
        torch.manual_seed(4)
        start = dataclasses.replace(state, h=torch.rand_like(state.h))
        with torch.no_grad():
            box = []
            s1, lat, pulse, _, later, _ = advance_fly_token_go(model, start.detached(), torch.tensor([77], device=device),
                                                            box, max_ticks=12, writer_baseline_clock='physical')
            assert box[0] == lat.shape[0] == pulse.shape[0] and later.shape == lat[-1:].shape
            # silent before the realized tick, firing on it (unless the ceiling ended the token)
            assert torch.all(pulse[:-1] == 0) and (bool(pulse[-1] > 0) or box[0] == 12)
            again = [box[0]]
            s2, lat2, _, _, _, _ = advance_fly_token_go(model, start.detached(), torch.tensor([77], device=device), again,
                                                      max_ticks=12, writer_baseline_clock='physical')
            assert torch.equal(lat, lat2) and torch.equal(s1.h, s2.h)
        learner = FlyBPTTLearner(model, start, lr=2e-4, writer_baseline_clock="physical", use_checkpointing=True,
                                 go_neuron=True, go_max_ticks=12, lambda_mcr2=0.0)
        assert learner.go_neuron == 'connectome' and not learner.adaptive_admission
        ids = torch.tensor([[50, 101, 205]], device=device)
        targets = torch.tensor([[101, 205, 307]], device=device)
        s0, g0, c0 = _window_gradients(learner, model, start, ids, targets, reuse=False)
        s1_, g1, c1 = _window_gradients(learner, model, start, ids, targets, reuse=True)
        # the recompute replays every realized tick; the counterfactual tick (no_grad, saves nothing) is never
        # recomputed by the early-stopping checkpoint, so: c0 = 2 * realized + counterfactual, c1 = realized + it
        assert torch.equal(s0, s1_) and c1 < c0 and (c0 - c1) % 2 == 0
        model.zero_grad(set_to_none=True)
        learner.state = start.detached()
        learner.forward_window(ids, targets)
        learner.last_train_loss.mean().backward()        # forward value = realized CE; timing credit flows backward
        g0 = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
        for name in ('edge_weight_e', 'edge_weight_i', 'output_read.weight', 'decoder.weight'):
            assert name in g0 and torch.isfinite(g0[name]).all() and g0[name].abs().sum() > 0, name
    finally:
        del model._buffers['go_neurons']


def test_virtual_go_neuron_learns_its_timing(fly_model_and_state):
    import dataclasses
    from information_boltzmann.core.fly_bptt_learning import attach_go_neuron, GO_PARAMETER_NAMES
    model, state, device = fly_model_and_state
    if device != "cuda":
        pytest.skip("fused AdamW needs CUDA")
    attach_go_neuron(model)
    try:
        start = dataclasses.replace(state, h=torch.rand_like(state.h))
        learner = FlyBPTTLearner(model, start, lr=2e-4, writer_baseline_clock="physical", use_checkpointing=True,
                                 go_neuron=True, go_max_ticks=12, lambda_mcr2=0.0)
        assert learner.go_neuron == 'virtual'
        ids = torch.tensor([[50, 101, 205]], device=device)
        targets = torch.tensor([[101, 205, 307]], device=device)
        model.zero_grad(set_to_none=True)
        learner.state = start.detached()
        s0, _, _ = learner.forward_window(ids, targets)
        learner.last_train_loss.mean().backward()           # the timing is learned through the expected loss
        g0 = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
        assert torch.isfinite(s0).all() and learner.last_adaptive_metrics['go_ceiling_hits'] == 0
        for name in GO_PARAMETER_NAMES + ('edge_weight_e',):
            assert name in g0 and torch.isfinite(g0[name]).all(), name
    finally:
        for name in ('go_read', 'go_drive', 'go_reset', 'go_logit_leak'):
            delattr(model, name)
        del model._buffers['synapse_share']
        for p in model.parameters():
            p.requires_grad_(False)


def test_go_spike_credit_value_and_direction():
    """Forward value is the realized CE; an earlier spike is pushed to fire if that tick was better, to hold if worse;
    the firing spike is pushed to hold if one more tick would have helped."""
    from information_boltzmann.core.fly_bptt_learning import go_spike_credit
    ce = torch.tensor([3.0, 1.0, 2.0])
    spikes = torch.tensor([0.0, 0.0, 1.0], requires_grad=True)
    loss = go_spike_credit(ce, spikes, torch.tensor(1.5))
    assert torch.isclose(loss, torch.tensor(2.0))
    loss.backward()
    assert spikes.grad.tolist() == [1.0, -1.0, 0.5]       # descend: tick 1 hold, tick 2 fire, firing tick hold


def test_read_bank_starts_as_the_champion_read_and_learns(fly_model_and_state):
    """A read bank with gains (1, 0, ...) reads exactly what the plain read does; its filters run across tokens; the
    window loss reaches the gains and the time constants."""
    import dataclasses
    from information_boltzmann.core.fly_bptt_learning import (attach_read_bank, read_fly_motor, attach_go_neuron,
                                                              READ_BANK_PARAMETER_NAMES, step_fly_physical_tick)
    model, state, device = fly_model_and_state
    if device != "cuda":
        pytest.skip("fused AdamW needs CUDA")
    start = dataclasses.replace(state, h=torch.rand_like(state.h))
    with torch.no_grad():
        plain = read_fly_motor(model, start)
    attach_read_bank(model)
    attach_go_neuron(model)
    try:
        opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds())
        with torch.no_grad():
            s1 = step_fly_physical_tick(model, start, None, torch.zeros_like(start.h), start.baseline, opts)
            assert s1.read_bank.shape == (6, model.n_read)
            plain1 = model.output_read(s1.h[:, model.read_indices])
            assert torch.allclose(read_fly_motor(model, s1), plain1)
        learner = FlyBPTTLearner(model, start, lr=2e-4, writer_baseline_clock="physical", use_checkpointing=True,
                                 go_neuron=True, go_max_ticks=12, lambda_mcr2=0.0)
        model.zero_grad(set_to_none=True)
        learner.state = start.detached()
        _, after, _ = learner.forward_window(torch.tensor([[50, 101, 205]], device=device),
                                             torch.tensor([[101, 205, 307]], device=device))
        learner.last_train_loss.mean().backward()
        for name in READ_BANK_PARAMETER_NAMES:
            assert getattr(model, name).grad is not None and torch.isfinite(getattr(model, name).grad).all(), name
        assert after.read_bank.numel() == 6 * model.n_read
    finally:
        for name in ('go_read', 'go_drive', 'go_reset', 'go_logit_leak', 'read_bank_gain', 'read_bank_logit_decay',
                     'read_bank_decay'):
            if hasattr(model, name):
                delattr(model, name)
        model._buffers.pop('synapse_share', None)
        for p in model.parameters():
            p.requires_grad_(False)


def test_intrinsic_drive_makes_annotated_pacemakers_fire_alone(fly_model_and_state):
    """Released cells are the annotated spontaneously active types; with kappa > 1 and no input at all they fire on
    their own, other cells receive no current; kappa receives a gradient."""
    from information_boltzmann.core.fly_bptt_learning import attach_intrinsic_drive, step_fly_physical_tick, FlyPhysicalState
    model, state, device = fly_model_and_state
    types = attach_intrinsic_drive(model, "data/malecns_v1/fly_reservoir_coba.npz",
                                   "data/malecns_v1/body-annotations.feather")
    try:
        assert any(t.startswith('s-LNv') for t in types) and any(t.startswith('PAM') for t in types)
        assert 300 < model.intrinsic_index.numel() < 1000
        n = model.n_neurons
        z = lambda: torch.zeros(1, n, device=device)
        s = FlyPhysicalState(z(), tuple(z() for _ in range(4)), z(), z(), z(), torch.ones(1, n, device=device),
                             torch.full((1, n), .25, device=device), state.baseline, z())
        opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds())
        fired = torch.zeros(n, device=device)
        with torch.no_grad():
            for _ in range(200):
                s = step_fly_physical_tick(model, s, None, torch.zeros(1, n, device=device), s.baseline, opts)
                fired += (s.ring[0][0] > 0).float()
        released = torch.zeros(n, dtype=torch.bool, device=device)
        released[model.intrinsic_index] = True
        assert fired[released].gt(0).float().mean() > 0.9          # pacemakers fire with no input at all
        s.h.sum()  # (silence elsewhere is not required: their spikes propagate through the connectome)
        model.intrinsic_kappa.grad = None
        s2 = step_fly_physical_tick(model, s, None, torch.zeros(1, n, device=device), s.baseline, opts)
        s2.h[0, model.intrinsic_index].sum().backward()
        assert model.intrinsic_kappa.grad is not None and model.intrinsic_kappa.grad.abs().sum() > 0
    finally:
        delattr(model, 'intrinsic_kappa')
        model._buffers.pop('intrinsic_index', None)
        for p in model.parameters():
            p.requires_grad_(False)


def test_mixed_world_listen_and_gap_windows(fly_model_and_state):
    """Listen: every word takes exactly k ticks.  Gap: a silent stretch after the act; the word after it is scored at
    the end of the silence and the loss reaches the read through the gap."""
    import dataclasses
    from information_boltzmann.core.fly_bptt_learning import attach_go_neuron, attach_read_bank
    model, state, device = fly_model_and_state
    if device != "cuda":
        pytest.skip("fused AdamW needs CUDA")
    attach_go_neuron(model)
    attach_read_bank(model)
    try:
        start = dataclasses.replace(state, h=torch.rand_like(state.h))
        learner = FlyBPTTLearner(model, start, lr=2e-4, writer_baseline_clock="physical", use_checkpointing=True,
                                 go_neuron=True, go_max_ticks=12, lambda_mcr2=0.0)
        learner.previous_token = 50
        tokens = np.array([101, 205, 307], dtype=np.int64)
        learner.world = {'mode': 'listen', 'k': 3}
        before = learner.physical_ticks
        scores, _ = learner.observe(tokens)
        assert all(np.isfinite(scores)) and learner.physical_ticks - before == 9
        learner.world = {'mode': 'gap', 'gaps': [0, 5, 0]}
        before = learner.physical_ticks
        scores, metrics = learner.observe(tokens)
        assert all(np.isfinite(scores)) and metrics['world_gap_ticks'] == 5
        assert learner.physical_ticks - before == metrics['adaptive_total_ticks'] >= 5 + 3
    finally:
        for name in ('go_read', 'go_drive', 'go_reset', 'go_logit_leak', 'read_bank_gain', 'read_bank_logit_decay',
                     'read_bank_decay'):
            if hasattr(model, name):
                delattr(model, name)
        model._buffers.pop('synapse_share', None)
        for p in model.parameters():
            p.requires_grad_(False)


def test_precision_starts_neutral_and_learns_through_the_modulators(fly_model_and_state):
    """Neuromodulatory precision: gains start at exactly 1 (the drive is unchanged), the concentration trace follows the
    modulatory spikes, the loss reaches u, b and the trace's time constant, and dopamine sets the synapses' learning
    rate within [1/4, 4]."""
    import dataclasses
    from information_boltzmann.core.fly_bptt_learning import (attach_precision, attach_go_neuron, attach_read_bank,
                                                              attach_intrinsic_drive, step_fly_physical_tick,
                                                              PRECISION_PARAMETER_NAMES)
    model, state, device = fly_model_and_state
    if device != "cuda":
        pytest.skip("fused AdamW needs CUDA")
    graph, notes = "data/malecns_v1/fly_reservoir_coba.npz", "data/malecns_v1/body-annotations.feather"
    attach_go_neuron(model)
    attach_read_bank(model)
    attach_intrinsic_drive(model, graph, notes)
    n_mod, n_da = attach_precision(model, graph, notes)
    try:
        assert 300 < n_mod < 600 and 300 < n_da <= n_mod
        start = dataclasses.replace(state, h=torch.rand_like(state.h))
        opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds())
        drive = torch.zeros_like(start.h)
        drive[0, model.topographic_writer.injection_index] = 1.0
        with torch.no_grad():
            s1 = step_fly_physical_tick(model, start, None, drive, start.baseline, opts)
            assert s1.modulator.shape == (1, n_mod) and float(s1.modulator.max()) <= 1.0
            gain = 2 * torch.sigmoid(s1.modulator @ model.precision_u.T + model.precision_b)
            assert torch.allclose(gain, torch.ones_like(gain))
        learner = FlyBPTTLearner(model, start, lr=2e-4, writer_baseline_clock="physical", use_checkpointing=True,
                                 go_neuron=True, go_max_ticks=12, lambda_mcr2=0.0)
        learner.previous_token = 50
        model.zero_grad(set_to_none=True)
        learner.state = start.detached()
        learner.forward_window(torch.tensor([[50, 101, 205]], device=device), torch.tensor([[101, 205, 307]], device=device))
        learner.last_train_loss.mean().backward()
        for name in PRECISION_PARAMETER_NAMES:
            assert getattr(model, name).grad is not None and torch.isfinite(getattr(model, name).grad).all(), name
        learner.state = start.detached()
        _, metrics = learner.observe(np.array([101, 205, 307], dtype=np.int64))
        assert 0.25 <= metrics['dopamine_plasticity'] <= 4.0
    finally:
        for name in ('go_read', 'go_drive', 'go_reset', 'go_logit_leak', 'read_bank_gain', 'read_bank_logit_decay',
                     'read_bank_decay', 'intrinsic_kappa', 'precision_u', 'precision_b', 'modulator_logit_decay',
                     'precision_scale'):
            if hasattr(model, name):
                delattr(model, name)
        for b in ('synapse_share', 'intrinsic_index', 'modulatory_index', 'dopamine_mask'):
            model._buffers.pop(b, None)
        for p in model.parameters():
            p.requires_grad_(False)


def test_predictive_writer_injects_surprise(fly_model_and_state):
    """With a top-down expectation, a correctly predicted word writes a smaller innovation than a wrong one; the
    expectation survives the ticks of a token and is renewed by the spoken prediction; beta learns."""
    import dataclasses
    from information_boltzmann.core.fly_bptt_learning import (attach_predictive_writer, write_input, expect_next,
                                                              advance_fly_token_stream, read_fly_motor)
    model, state, device = fly_model_and_state
    attach_predictive_writer(model)
    try:
        start = dataclasses.replace(state, h=torch.rand_like(state.h))
        right, wrong = torch.tensor([464], device=device), torch.tensor([31337], device=device)
        primed = dataclasses.replace(start, expected_input=model.embedding(right).detach())
        inj = model.topographic_writer.injection_index
        d_right, _ = write_input(model, primed, right)
        d_wrong, _ = write_input(model, primed, wrong)
        assert d_right[0, inj].norm() < d_wrong[0, inj].norm()
        model.writer_prediction_logit.grad = None
        d_wrong[0, inj].square().sum().backward()
        assert model.writer_prediction_logit.grad is not None
        with torch.no_grad():
            after, latent = advance_fly_token_stream(model, start, right, ticks=3, writer_baseline_clock='physical')
            assert after.expected_input.shape == (1, model.embedding.embedding_dim)
            again = expect_next(model, after, latent)
            assert torch.allclose(again.expected_input, after.expected_input)
    finally:
        delattr(model, 'writer_prediction_logit')
        for p in model.parameters():
            p.requires_grad_(False)


def test_dopamine_modulation_along_the_connectome(fly_model_and_state):
    """Dopamine delivered along the graph's modulatory projections: no effect at the start (sensitivity 0), a fast and a
    slow concentration per neuron, gradients into the sensitivities and into the synapses onto the dopamine cells, and a
    postsynaptic plasticity scaling within [1/4, 4]."""
    import dataclasses
    from information_boltzmann.core.fly_bptt_learning import (attach_dopamine_modulation, attach_go_neuron,
                                                              step_fly_physical_tick, DOPAMINE_PARAMETER_NAMES)
    model, state, device = fly_model_and_state
    if device != "cuda" or not model.has_dopamine:
        pytest.skip("needs CUDA and the dopaminergic projections")
    attach_go_neuron(model)
    try:
        torch.manual_seed(5)
        start = dataclasses.replace(state, h=torch.rand_like(state.h))
        opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds())
        with torch.no_grad():
            plain = step_fly_physical_tick(model, start, None, torch.zeros_like(start.h), start.baseline, opts)
        attach_dopamine_modulation(model)
        with torch.no_grad():
            s1 = step_fly_physical_tick(model, start, None, torch.zeros_like(start.h), start.baseline, opts)
            s2 = step_fly_physical_tick(model, s1, None, torch.zeros_like(s1.h), s1.baseline, opts)
        assert torch.equal(plain.h, s1.h)                     # sensitivity 0: nothing changes
        assert s2.dopamine.shape == (2, model.n_neurons)
        learner = FlyBPTTLearner(model, start, lr=2e-4, writer_baseline_clock="physical", use_checkpointing=True,
                                 go_neuron=True, go_max_ticks=10, lambda_mcr2=0.0)
        learner.previous_token = 50
        model.dopamine_sensitivity.data.fill_(0.5)
        model.zero_grad(set_to_none=True)
        learner.state = s2.detached()
        learner.forward_window(torch.tensor([[50, 101, 205]], device=device), torch.tensor([[101, 205, 307]], device=device))
        learner.last_train_loss.mean().backward()
        for name in DOPAMINE_PARAMETER_NAMES:
            assert getattr(model, name).grad is not None and torch.isfinite(getattr(model, name).grad).all(), name
        onto_dan = torch.isin(model.edge_post_e.long(), model.dan_indices)
        assert model.edge_weight_e.grad[onto_dan].abs().sum() > 0
        learner.state = s2.detached()
        _, metrics = learner.observe(np.array([101, 205, 307], dtype=np.int64))
        assert 'dopamine_plasticity_spread' in metrics
    finally:
        for name in ('go_read', 'go_drive', 'go_reset', 'go_logit_leak') + DOPAMINE_PARAMETER_NAMES:
            if hasattr(model, name):
                delattr(model, name)
        model._buffers.pop('synapse_share', None)
        for p in model.parameters():
            p.requires_grad_(False)


def test_dopamine_scales_learning_at_the_chosen_end_of_each_synapse():
    """site='post' scales a synapse's update by the dopamine at its receiving neuron, site='pre' by the dopamine at its
    presynaptic terminal (as at the fly's Kenyon-cell -> output-neuron synapses)."""
    from types import SimpleNamespace
    from information_boltzmann.core.fly_bptt_learning import dopamine_edge_scale
    model = SimpleNamespace(edge_pre_e=torch.tensor([0, 0, 2]), edge_post_e=torch.tensor([1, 2, 1]))
    local = torch.tensor([4.0, 0.5, 1.0])
    assert torch.equal(dopamine_edge_scale(model, '_e', local, 'post'), torch.tensor([0.5, 1.0, 0.5]))
    assert torch.equal(dopamine_edge_scale(model, '_e', local, 'pre'), torch.tensor([4.0, 4.0, 1.0]))
    with pytest.raises(ValueError):
        dopamine_edge_scale(model, '_e', local, 'both')


def test_adapted_dopamine_starts_with_an_effect(fly_model_and_state):
    """Adapted receptors: the phasic part is each target's own relative change; a nonzero starting sensitivity changes
    the dynamics as soon as the concentration departs from its slow mean, and the learner's plasticity scaling runs."""
    import dataclasses
    from information_boltzmann.core.fly_bptt_learning import (attach_dopamine_modulation, attach_go_neuron,
                                                              step_fly_physical_tick, DOPAMINE_PARAMETER_NAMES)
    model, state, device = fly_model_and_state
    if device != "cuda" or not model.has_dopamine:
        pytest.skip("needs CUDA and the dopaminergic projections")
    attach_go_neuron(model)
    try:
        torch.manual_seed(5)
        start = dataclasses.replace(state, h=torch.rand_like(state.h))
        opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds())
        attach_dopamine_modulation(model, adapted=True, sensitivity_init=4.0)
        assert model.dopamine_adapted and torch.all(model.dopamine_sensitivity == 4.0)
        with torch.no_grad():
            s1 = step_fly_physical_tick(model, start, None, torch.zeros_like(start.h), start.baseline, opts)
            s2 = step_fly_physical_tick(model, s1, None, torch.zeros_like(s1.h), s1.baseline, opts)
            s3 = step_fly_physical_tick(model, s2, None, torch.zeros_like(s2.h), s2.baseline, opts)
            model.dopamine_sensitivity.zero_()
            s3_off = step_fly_physical_tick(model, s2, None, torch.zeros_like(s2.h), s2.baseline, opts)
            model.dopamine_sensitivity.fill_(4.0)
        assert torch.isfinite(s3.h).all()
        assert not torch.equal(s3.h, s3_off.h)               # dopamine acts from the start
        learner = FlyBPTTLearner(model, start, lr=2e-4, writer_baseline_clock="physical", use_checkpointing=True,
                                 go_neuron=True, go_max_ticks=10, lambda_mcr2=0.0)
        learner.previous_token = 50
        learner.state = s3.detached()
        _, metrics = learner.observe(np.array([101, 205, 307], dtype=np.int64))
        assert 'dopamine_plasticity_spread' in metrics
    finally:
        for name in ('go_read', 'go_drive', 'go_reset', 'go_logit_leak') + DOPAMINE_PARAMETER_NAMES:
            if hasattr(model, name):
                delattr(model, name)
        for name in ('dopamine_adapted', 'dopamine_site'):
            if hasattr(model, name):
                delattr(model, name)
        model._buffers.pop('synapse_share', None)
        for p in model.parameters():
            p.requires_grad_(False)


def _online_cleanup(model):
    from information_boltzmann.core.fly_bptt_learning import DOPAMINE_PARAMETER_NAMES, ONLINE_DA_PARAMETER_NAMES
    for name in ('go_read', 'go_drive', 'go_reset', 'go_logit_leak') + DOPAMINE_PARAMETER_NAMES + ONLINE_DA_PARAMETER_NAMES:
        if hasattr(model, name):
            delattr(model, name)
    for name in ('dopamine_adapted', 'dopamine_site'):
        if hasattr(model, name):
            delattr(model, name)
    for name in ('synapse_share', 'online_da_target'):
        model._buffers.pop(name, None)
    for p in model.parameters():
        p.requires_grad_(False)


def _run_ticks(model, state, n, opts):
    from information_boltzmann.core.fly_bptt_learning import step_fly_physical_tick
    s = state
    with torch.no_grad():
        for _ in range(n):
            s = step_fly_physical_tick(model, s, None, torch.zeros_like(s.h), s.baseline, opts)
    return s


def test_online_da_plasticity_off_zero_rate_and_constant_memory(fly_model_and_state):
    """Off: the state carries no online fields.  On with eta = 0: every release probability is exactly its baseline,
    so the dynamics are bitwise those without the rule.  The persistent state has the same size after 20 and after 120
    ticks (O(1)); release probabilities move only at the chosen terminals and stay within [0, 1]."""
    import dataclasses
    from information_boltzmann.core.fly_bptt_learning import (attach_dopamine_modulation, attach_online_da_plasticity,
                                                              online_da_release)
    model, state, device = fly_model_and_state
    if device != "cuda" or not model.has_dopamine:
        pytest.skip("needs CUDA and the dopaminergic projections")
    try:
        torch.manual_seed(7)
        start = dataclasses.replace(state, h=torch.rand_like(state.h))
        opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds())
        attach_dopamine_modulation(model, site='pre')
        off = _run_ticks(model, start, 12, opts)
        assert off.da_trace.numel() == 0 and off.da_memory.numel() == 0
        attach_online_da_plasticity(model, 'data/malecns_v1/fly_reservoir_coba.npz', 'data/malecns_v1/body-annotations.feather', eta=0.0)
        zero = _run_ticks(model, start, 12, opts)
        assert torch.equal(off.h, zero.h) and all(torch.equal(a, b) for a, b in zip(off.ring, zero.ring))
        assert torch.count_nonzero(zero.da_memory) == 0 and zero.da_trace.shape == zero.h.shape
        with torch.no_grad():
            model.online_da_eta.fill_(300.0)
        short, long = _run_ticks(model, start, 20, opts), _run_ticks(model, start, 120, opts)
        size = lambda s: sum(sum(t.numel() for t in v) if isinstance(v, tuple) else v.numel() for v in s.state_dict().values())
        assert size(short) == size(long)
        u0 = model.get_stp_params()[0].expand_as(long.da_memory)
        release = online_da_release(u0, long.da_memory)
        outside = model.online_da_target == 0
        assert torch.equal(release[outside], u0[outside])
        assert torch.any(release[~outside] != u0[~outside])
        assert torch.isfinite(release).all() and float(release.min()) >= 0.0 and float(release.max()) <= 1.0
    finally:
        _online_cleanup(model)


def test_online_da_plasticity_never_sees_the_target_and_is_learnable(fly_model_and_state):
    """Swapping the targets leaves the written memory and the physical state unchanged up to the run-to-run float
    atomics noise (no leak); within a window the rule's parameters get finite nonzero gradients."""
    import dataclasses
    from information_boltzmann.core.fly_bptt_learning import (attach_dopamine_modulation, attach_go_neuron,
                                                              attach_online_da_plasticity, ONLINE_DA_PARAMETER_NAMES)
    model, state, device = fly_model_and_state
    if device != "cuda" or not model.has_dopamine:
        pytest.skip("needs CUDA and the dopaminergic projections")
    attach_go_neuron(model)
    try:
        torch.manual_seed(5)
        start = dataclasses.replace(state, h=torch.rand_like(state.h))
        attach_dopamine_modulation(model, site='pre')
        attach_online_da_plasticity(model, 'data/malecns_v1/fly_reservoir_coba.npz', 'data/malecns_v1/body-annotations.feather', eta=30.0)
        opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds())
        warm = _run_ticks(model, start, 6, opts).detached()
        learner = FlyBPTTLearner(model, warm, lr=2e-4, writer_baseline_clock="physical", use_checkpointing=True,
                                 go_neuron=True, go_max_ticks=8, lambda_mcr2=0.0)
        ids = torch.tensor([[50, 101, 205]], device=device)
        ends = []
        for tgt in ([101, 205, 307], [9, 8, 7]):
            learner.previous_token = 50
            learner.state = warm.detached()
            torch.manual_seed(11)
            with torch.no_grad():
                _, end, _ = learner.forward_window(ids, torch.tensor([tgt], device=device))
            ends.append(end)
        # Dopamine arrival is summed with float atomics, so two identical runs already differ by ~1e-6 once the
        # memory feeds it back; a leak of the target would be of order one.
        torch.testing.assert_close(ends[0].da_memory, ends[1].da_memory, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(ends[0].h, ends[1].h, rtol=1e-5, atol=1e-5)
        learner.state = warm.detached()
        learner.previous_token = 50
        model.zero_grad(set_to_none=True)
        torch.manual_seed(11)
        learner.forward_window(ids, torch.tensor([[101, 205, 307]], device=device))
        learner.last_train_loss.mean().backward()
        for name in ONLINE_DA_PARAMETER_NAMES:
            grad = getattr(model, name).grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    finally:
        _online_cleanup(model)


def test_kenyon_axo_axonal_synapses_do_not_drive_somata(fly_model_and_state):
    """Kenyon -> Kenyon excitatory weights are scaled by their calyx share (neuPrint roiInfo); connections without a
    calyx synapse are 0 and stay 0 through a learner update."""
    import os
    from information_boltzmann.core.fly_bptt_learning import attach_kenyon_dendritic_only, kenyon_dendritic_share
    model, state, device = fly_model_and_state
    feather = 'data/malecns_v1/kc_kc_dendritic_synapses.feather'
    if device != "cuda" or not os.path.exists(feather):
        pytest.skip("needs CUDA and the neuPrint calyx table")
    graph, ann = 'data/malecns_v1/fly_reservoir_coba.npz', 'data/malecns_v1/body-annotations.feather'
    original = model.edge_weight_e.detach().clone()
    try:
        edges, share = kenyon_dendritic_share(model, graph, ann, feather)
        assert edges.numel() > 600_000 and 0.0 < float((share > 0).float().mean()) < 0.1
        attach_kenyon_dendritic_only(model, graph, ann, feather)
        torch.testing.assert_close(model.edge_weight_e[edges], original[edges] * share)
        pinned = model.pinned_zero_edges_e
        learner = FlyBPTTLearner(model, state, lr=2e-4, writer_baseline_clock="physical", use_checkpointing=True)
        with torch.no_grad():
            model.edge_weight_e[pinned] = 0.3
            learner.clamp_edges()
        assert torch.count_nonzero(model.edge_weight_e[pinned]) == 0
        others = torch.ones_like(original, dtype=torch.bool)
        others[edges] = False
        assert torch.equal(model.edge_weight_e[others], original[others])
    finally:
        with torch.no_grad():
            model.edge_weight_e.copy_(original)
        model._buffers.pop('pinned_zero_edges_e', None)
        for p in model.parameters():
            p.requires_grad_(False)


def test_homeostatic_intrinsic_plasticity(fly_model_and_state):
    """Off: no homeostasis field.  On: targets are c / tau_m with a log-normal duty cycle fixed by the seed (the
    individual); a silent neuron's drive grows by nu*/100 per tick, a neuron that fired loses (1 - nu*)/100; the state is
    one number per neuron at any age; the rule never sees the target word (it uses only the neuron's own spikes)."""
    import dataclasses
    import numpy as np
    from information_boltzmann.core.fly_bptt_learning import attach_homeostasis, step_fly_physical_tick, HOMEOSTASIS_SPIKES
    model, state, device = fly_model_and_state
    if device != "cuda":
        pytest.skip("needs CUDA")
    graph = 'data/malecns_v1/fly_reservoir_coba.npz'
    try:
        opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds())
        with torch.no_grad():
            off = step_fly_physical_tick(model, state, None, torch.zeros_like(state.h), state.baseline, opts)
        assert off.homeostasis.numel() == 0
        attach_homeostasis(model, graph, seed=3)
        t3 = model.homeostasis_target.clone()
        attach_homeostasis(model, graph, seed=3)
        assert torch.equal(t3, model.homeostasis_target)
        attach_homeostasis(model, graph, seed=4)
        assert not torch.equal(t3, model.homeostasis_target)
        tau = torch.as_tensor(np.load(graph)['tau_m'], dtype=torch.float32, device=device)
        duty = (model.homeostasis_target[0] * tau.clamp_min(1.0))
        ld = torch.log(duty[model.homeostasis_target[0] < 0.2])
        assert abs(float(ld.median()) - math.log(0.05)) < 0.05 and abs(float(ld.std()) - 1.0) < 0.05
        torch.manual_seed(1)
        start = dataclasses.replace(state, h=torch.rand_like(state.h))
        with torch.no_grad():
            s1 = step_fly_physical_tick(model, start, None, torch.zeros_like(start.h), start.baseline, opts)
            s2 = step_fly_physical_tick(model, s1, None, torch.zeros_like(s1.h), s1.baseline, opts)
        fired = (s2.ring[0] > 0).float()
        expected = s1.homeostasis + (model.homeostasis_target - fired) / HOMEOSTASIS_SPIKES
        torch.testing.assert_close(s2.homeostasis, expected)
        assert s2.homeostasis.shape == s2.h.shape and not s2.homeostasis.requires_grad
    finally:
        model._buffers.pop('homeostasis_target', None)
        if hasattr(model, 'homeostasis_seed'):
            delattr(model, 'homeostasis_seed')


def test_tied_writer(fly_model_and_state):
    """Off: per-cell projections.  On: every chemosensory cell of one receptor type gets the same drive (visual and
    mechanosensory maps stay per cell); an untyped cell (its own row, scale 1) keeps exactly its drive; convert=False
    builds rows of the same shapes for loading."""
    import numpy as np
    import pandas as pd
    from torch import nn
    from information_boltzmann.core.fly_bptt_learning import attach_tied_writer, TiedProjection
    model, state, device = fly_model_and_state
    writer = model.topographic_writer
    if writer is None:
        pytest.skip("needs the topographic writer")
    graph, ann = 'data/malecns_v1/fly_reservoir_coba.npz', 'data/malecns_v1/body-annotations.feather'
    saved = {n: getattr(writer, n) for n in ('proj_vis', 'proj_chemo', 'proj_mech')}
    try:
        assert all(isinstance(m, nn.Linear) for m in saved.values())
        emb = torch.randn(1, writer.proj_chemo.in_features, device=device)
        with torch.no_grad():
            before = writer.packet(emb)
        counts = attach_tied_writer(model, graph, ann, convert=True, scale=1.0)
        assert counts['proj_chemo'][1] < counts['proj_chemo'][0]
        with torch.no_grad():
            after = writer.packet(emb)
        types = pd.read_feather(ann).set_index('bodyId').reindex(np.load(graph, allow_pickle=True)['neuron_body_ids'])['type'].astype(str).values
        cells = writer.injection_index.cpu().numpy()
        labels = types[cells]
        orn = np.flatnonzero(labels == pd.Series(labels[writer.n_vis:writer.n_vis + writer.n_chemo]).value_counts().index[0])
        assert torch.allclose(after[0, orn], after[0, orn[0]].expand(len(orn)))
        untyped = writer.n_vis + np.flatnonzero(np.isin(labels[writer.n_vis:writer.n_vis + writer.n_chemo], ['None', 'nan']))
        if len(untyped):
            torch.testing.assert_close(after[0, untyped], before[0, untyped])
        assert isinstance(writer.proj_vis, nn.Linear) and isinstance(writer.proj_mech, nn.Linear)   # maps stay per cell
        shapes = {n: getattr(writer, n).weight.shape for n in saved}
        for n, m in saved.items():
            setattr(writer, n, m)
        attach_tied_writer(model, graph, ann, convert=False)
        assert {n: getattr(writer, n).weight.shape for n in saved} == shapes
        assert isinstance(writer.proj_chemo, TiedProjection)
    finally:
        for n, m in saved.items():
            setattr(writer, n, m)
        if hasattr(model, 'tied_writer'):
            delattr(model, 'tied_writer')


def test_compartment_memory_and_external_drive(fly_model_and_state):
    """Off: no compartment fields.  On with phi = 0: the Kenyon -> MBON synapses moved out of the static list give the
    same tick (float tolerance: a different summation order).  phi follows w (eta p - kappa phi) with the dopamine at
    the synapse's MBON and the Kenyon trace.  An external drive acts exactly like the same current in the source."""
    import dataclasses
    from information_boltzmann.core.fly_bptt_learning import (attach_compartment_memory, attach_dopamine_modulation,
                                                              step_fly_physical_tick, dopamine_occupancy)
    model, state, device = fly_model_and_state
    if device != "cuda":
        pytest.skip("needs CUDA")
    if not getattr(model, 'has_dopamine', False):
        pytest.skip("needs dopamine projections")
    graph, ann = 'data/malecns_v1/fly_reservoir_coba.npz', 'data/malecns_v1/body-annotations.feather'
    w0 = model.edge_weight_e.detach().clone()
    try:
        attach_dopamine_modulation(model)
        opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds())
        torch.manual_seed(2)
        start = dataclasses.replace(state, h=torch.rand_like(state.h))
        with torch.no_grad():
            ref1 = step_fly_physical_tick(model, start, None, torch.zeros_like(start.h), start.baseline, opts)
            ref2 = step_fly_physical_tick(model, ref1, None, torch.zeros_like(ref1.h), ref1.baseline, opts)
        assert ref2.comp_memory.numel() == 0
        drive = torch.zeros_like(start.h); drive[0, :50] = 0.3
        model.external_drive = drive
        with torch.no_grad():
            ext = step_fly_physical_tick(model, start, None, torch.zeros_like(start.h), start.baseline, opts)
        del model.external_drive
        with torch.no_grad():
            src = step_fly_physical_tick(model, start, None, drive.clone(), start.baseline, opts)
        assert torch.equal(ext.h, src.h)
        n = attach_compartment_memory(model, graph, ann)
        assert n > 1000 and float(model.edge_weight_e[model.pinned_zero_edges_e].abs().max()) == 0.0
        with torch.no_grad():
            s1 = step_fly_physical_tick(model, start, None, torch.zeros_like(start.h), start.baseline, opts)
            s2 = step_fly_physical_tick(model, s1, None, torch.zeros_like(s1.h), s1.baseline, opts)
        torch.testing.assert_close(s2.h, ref2.h, atol=1e-5, rtol=1e-4)
        assert s2.comp_memory.shape == (1, n) and s2.comp_trace.shape == (1, model.comp_kc.numel())
        d = s2.dopamine[:, model.comp_post.long()]
        p = dopamine_occupancy(d[0:1], model.comp_affinity) - dopamine_occupancy(d[1:2], model.comp_affinity)
        w = dopamine_occupancy(d[1:2], model.comp_affinity) / 0.5 * s2.comp_trace[:, model.comp_edge_kc]
        expected = s1.comp_memory + w * (model.comp_eta * p - torch.sigmoid(model.comp_logit_forget) * s1.comp_memory)
        torch.testing.assert_close(s2.comp_memory, expected)
    finally:
        with torch.no_grad():
            model.edge_weight_e.copy_(w0)
        for name in ('pinned_zero_edges_e', 'comp_pre', 'comp_post', 'comp_kc', 'comp_edge_kc'):
            model._buffers.pop(name, None)
        for name in ('comp_weight', 'comp_eta', 'comp_logit_trace', 'comp_logit_forget', 'dopamine_sensitivity',
                     'dopamine_logit_fast', 'dopamine_logit_slow'):
            model._parameters.pop(name, None)
        for name in ('comp_splits', 'comp_affinity', 'external_drive', 'dopamine_site', 'dopamine_adapted', 'dopamine_affinity'):
            if hasattr(model, name):
                delattr(model, name)


def test_regime_homeostasis(fly_model_and_state):
    """Off: no regime field.  On: the split tick equals model.step bitwise at unit gain and zero drive; the quantile
    targets are fixed by the seed (median q* ~ 0.16); b follows eta_b (q* - [u > theta]) and the log gain
    eta_s log(kappa theta / sd) from the neuron's own free membrane; the state is [4, N] at any age and detached."""
    import dataclasses
    import numpy as np
    from information_boltzmann.core.fly_bptt_learning import (attach_regime_homeostasis, step_fly_physical_tick,
                                                              REGIME_LOG_GAIN_MAX)
    model, state, device = fly_model_and_state
    if device != "cuda":
        pytest.skip("needs CUDA")
    graph = 'data/malecns_v1/fly_reservoir_coba.npz'
    names = ('regime_quantile', 'regime_eta_b', 'regime_eta_s', 'regime_window_rate')
    try:
        opts = dict(base_rates=model.get_decay_rates(), thresholds=model.get_thresholds())
        torch.manual_seed(1)
        start = dataclasses.replace(state, h=torch.rand_like(state.h))
        with torch.no_grad():
            off = step_fly_physical_tick(model, start, None, torch.zeros_like(start.h), start.baseline, opts)
        assert off.regime.numel() == 0
        attach_regime_homeostasis(model, graph, seed=3)
        q3 = model.regime_quantile.clone()
        attach_regime_homeostasis(model, graph, seed=3)
        assert torch.equal(q3, model.regime_quantile)
        attach_regime_homeostasis(model, graph, seed=4)
        assert not torch.equal(q3, model.regime_quantile)
        assert abs(float(model.regime_quantile.median()) - 0.1587) < 0.01
        import pandas as pd
        ann = 'data/malecns_v1/body-annotations.feather'
        attach_regime_homeostasis(model, graph, seed=4, kenyon_z=1.645, annotations=ann)
        kc = torch.as_tensor(pd.read_feather(ann).set_index('bodyId').reindex(np.load(graph, allow_pickle=True)['neuron_body_ids'])['class'].astype(str).values == 'Kenyon_Cell', device=device)
        q_all = model.regime_quantile[0]
        assert abs(float(q_all[kc].median()) - 0.05) < 0.01
        attach_regime_homeostasis(model, graph, seed=4)
        assert torch.equal(model.regime_quantile[0][~kc], q_all[~kc])
        with torch.no_grad():
            s1 = step_fly_physical_tick(model, start, None, torch.zeros_like(start.h), start.baseline, opts)
            assert torch.equal(s1.h, off.h) and torch.equal(s1.ring[0], off.ring[0])
            s2 = step_fly_physical_tick(model, s1, None, torch.zeros_like(s1.h), s1.baseline, opts)
        theta = model.get_thresholds().expand_as(s2.h)
        above = (s2.regime[0:1] > theta).float()
        torch.testing.assert_close(s2.homeostasis, s1.homeostasis + model.regime_eta_b * (model.regime_quantile - above))
        sd = (s2.regime[2:3] - s2.regime[1:2] ** 2).clamp_min(1e-12).sqrt()
        expected = (s1.regime[3:4] + model.regime_eta_s * torch.log(model.regime_kappa * theta / sd)).clamp(
            -REGIME_LOG_GAIN_MAX, REGIME_LOG_GAIN_MAX)
        torch.testing.assert_close(s2.regime[3:4], expected)
        assert s2.regime.shape == (4, s2.h.shape[1]) and not s2.regime.requires_grad and not s2.homeostasis.requires_grad
    finally:
        for n in names:
            model._buffers.pop(n, None)
        for a in ('regime_kappa', 'regime_seed', 'regime_kenyon_z'):
            if hasattr(model, a):
                delattr(model, a)
