"""Unit tests for FlyReservoirLM with Scheme B biological leak and dopamine modulation."""
from pathlib import Path
import numpy as np
import pytest
import torch

from information_boltzmann.core.fly_reservoir import FlyReservoirLM

DATA_PATH = Path("data/malecns_v1/fly_reservoir_biological.npz")

@pytest.mark.skipif(not DATA_PATH.exists(), reason="Biological graph NPZ not present")
def test_biological_reservoir_initialization():
    model = FlyReservoirLM(DATA_PATH, injection="sensory", read_surface="output")
    assert model.has_delays is True
    assert model.has_biological_leak is True
    assert model.has_dopamine is True
    assert model.lambda_0.shape == (165122,)
    assert (model.lambda_0 > 0.0).all() and (model.lambda_0 < 1.0).all()
    assert model.dan_edge_pre.shape[0] == 241701
    assert model.dan_edge_post.shape[0] == 241701
    assert len(model.dan_delay_splits) == 5

@pytest.mark.skipif(not DATA_PATH.exists() or not torch.cuda.is_available(), reason="CUDA and graph required")
def test_biological_reservoir_step_and_dopamine_gating():
    device = "cuda"
    model = FlyReservoirLM(DATA_PATH, injection="sensory", read_surface="output").to(device)
    
    h = torch.zeros(1, model.n_neurons, device=device)
    ring = tuple(torch.zeros(1, model.n_neurons, device=device) for _ in range(4))
    token = torch.tensor([42], device=device)

    # 1. Step without dopamine firing
    h_next, spikes, next_ring = model.step(h, token, ring)
    assert h_next.shape == (1, model.n_neurons)
    assert spikes.shape == (1, model.n_neurons)
    assert len(next_ring) == 4
    assert torch.isfinite(h_next).all()

    # 2. Trigger DAN spikes in the delay ring
    # Find DAN neurons in graph
    dan_pre_sample = model.dan_edge_pre[:10]
    ring_with_dan = list(ring)
    ring_with_dan[0] = ring[0].clone()
    ring_with_dan[0][0, dan_pre_sample] = 1.0 # Force DAN spikes

    h_with_dopamine, _, _ = model.step(h, token, ring_with_dan)
    assert torch.isfinite(h_with_dopamine).all()

    # Targets of these DANs should have higher residual potential due to closed leak
    target_sample = model.dan_edge_post[:10]
    # Check that execution was strictly passive and numerically stable
    assert (h_with_dopamine[0, target_sample] >= h_next[0, target_sample]).all()

@pytest.mark.skipif(not DATA_PATH.exists() or not torch.cuda.is_available(), reason="CUDA and graph required")
def test_biological_reservoir_backward():
    device = "cuda"
    model = FlyReservoirLM(DATA_PATH, injection="sensory", read_surface="output").to(device)
    
    h = torch.zeros(1, model.n_neurons, device=device)
    ring = tuple(torch.zeros(1, model.n_neurons, device=device) for _ in range(4))
    
    input_ids = torch.tensor([[10, 20, 30]], device=device)
    targets = torch.tensor([[20, 30, 40]], device=device)

    loss, h_out, diag, ring_out = model.forward_chunk(input_ids, targets, h, ring)
    assert torch.isfinite(loss)
    loss.backward()

    # Check gradients on trainable components
    assert model.input_proj.weight.grad is not None
    assert torch.isfinite(model.input_proj.weight.grad).all()
    assert model.output_read.weight.grad is not None
    assert torch.isfinite(model.output_read.weight.grad).all()


@pytest.mark.skipif(not DATA_PATH.exists() or not torch.cuda.is_available(), reason="CUDA and graph required")
def test_continuous_synaptic_conductance_decay():
    device = "cuda"
    model = FlyReservoirLM(DATA_PATH, injection="sensory", read_surface="output").to(device)
    
    h = torch.zeros(1, model.n_neurons, device=device)
    ring = tuple(torch.zeros(1, model.n_neurons, device=device) for _ in range(4))
    i_syn = torch.zeros(1, model.n_neurons, device=device)
    token = torch.tensor([42], device=device)

    # 1. Provide an impulse in delay 1
    sample_pre = model.edge_pre[:50]
    sample_post = model.edge_post[:50]
    ring_impulse = list(ring)
    ring_impulse[0] = ring[0].clone()
    ring_impulse[0][0, sample_pre] = 1.0
    
    # Step 1: impulse arrives
    h1, s1, ring1, i_syn1 = model.step(h, token, ring_impulse, i_syn)
    active_posts = sample_post[i_syn1[0, sample_post].abs() > 0]
    assert len(active_posts) > 0
    
    # Step 2: No new spikes in ring
    empty_ring = tuple(torch.zeros(1, model.n_neurons, device=device) for _ in range(4))
    h2, s2, ring2, i_syn2 = model.step(h1, token, empty_ring, i_syn1)
    
    # Check physical continuity: i_syn2 must NOT drop to zero; it must decay continuously
    assert (i_syn2[0, active_posts].abs() > 0).all()
    assert (i_syn2[0, active_posts].abs() <= i_syn1[0, active_posts].abs() + 1e-6).all()

