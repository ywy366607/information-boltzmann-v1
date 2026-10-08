"""Numerical contracts for trainable torus apertures; no capability claims."""
import math

import pytest
import torch

from information_boltzmann.core.torus3d import CBIMTorus3D
from scripts.ib.evaluate_continuous_owt import _model_from_checkpoint


def make_model(shape=(4, 4, 4), aperture="learned_probes"):
    return CBIMTorus3D(
        vocab_size=19, shape=shape, velocities=8, content_dim=2,
        write_type="w4_predictive_agent", readout_type="belief_agent",
        readout_aperture=aperture, event_duration=3, micro_steps=2,
    )


def test_probe_positions_scales_and_every_query_receive_ce_gradients():
    torch.manual_seed(72)
    model = make_model().double()
    reader = model.readout
    field = torch.randn(2, 4, 4, 4, 16, dtype=torch.float64, requires_grad=True)
    feature, diag = reader(field, torch.ones(2, 16, dtype=torch.float64), True)
    loss = torch.nn.functional.cross_entropy(model.decoder(feature), torch.tensor([3, 9]))
    loss.backward()
    assert reader.probe_coords.grad.isfinite().all()
    assert (reader.probe_coords.grad.norm(dim=-1) > 0).all()
    assert reader.head_log_scale.grad.isfinite().all()
    assert (reader.head_log_scale.grad.abs() > 0).all()
    assert reader.probe_log_scale.grad.isfinite().all()
    assert (reader.probe_log_scale.grad.abs() > 0).all()
    attention = diag["read_attention_weights"]
    assert (attention > 0).all()
    torch.testing.assert_close(attention.sum(-1), torch.ones(2, 4, 4, dtype=torch.float64))
    torch.testing.assert_close(diag["read_head_scales"], torch.full((4,), math.sqrt(4), dtype=torch.float64))


def test_aperture_is_periodic_and_varies_when_location_changes():
    torch.manual_seed(11)
    reader = make_model().double().readout
    field = torch.randn(1, 4, 4, 4, 16, dtype=torch.float64)
    precision = torch.ones(1, 16, dtype=torch.float64)
    feature, diag = reader(field, precision, True)
    with torch.no_grad():
        reader.probe_coords.add_(torch.tensor([1., -2., 3.]))
    periodic, periodic_diag = reader(field, precision, True)
    torch.testing.assert_close(periodic, feature, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(periodic_diag["read_attention_weights"], diag["read_attention_weights"], atol=1e-12, rtol=1e-12)
    with torch.no_grad():
        reader.probe_coords[0, 0, 0].add_(0.125)
    _, moved = reader(field, precision, True)
    assert not torch.allclose(moved["read_attention_weights"][:, 0, 0], diag["read_attention_weights"][:, 0, 0])


@pytest.mark.parametrize("aperture", ["atlas", "learned_probes"])
def test_checkpoint_evaluation_and_cross_grid_loading(aperture):
    model = make_model(aperture=aperture)
    cfg = dict(architecture=model.architecture, vocab_size=19, shape=[4, 4, 4],
               velocities=8, content_dim=2, collision_layers=2,
               readout_type="belief_agent", write_type="w4_predictive_agent",
               physical_time_policy="fixed_event_duration", event_duration=3,
               micro_steps=2)
    # Historical configurations omit this key entirely.
    if aperture == "learned_probes":
        cfg["readout_aperture"] = aperture
    recovered, _, _ = _model_from_checkpoint({"config": cfg})
    recovered.load_state_dict(model.state_dict(), strict=True)
    assert recovered.readout.aperture_type == aperture
    finer = make_model(shape=(8, 4, 4), aperture=aperture)
    finer.load_state_dict(model.state_dict(), strict=True)
    if aperture == "learned_probes":
        torch.testing.assert_close(finer.readout.probe_coords, model.readout.probe_coords)


def test_unmatched_aperture_checkpoint_is_rejected():
    legacy = make_model(aperture="atlas")
    new = make_model()
    with pytest.raises(RuntimeError, match="Missing key"):
        new.load_state_dict(legacy.state_dict(), strict=True)


def test_reader_change_preserves_all_other_initial_parameters():
    torch.manual_seed(11)
    legacy = make_model(aperture="atlas")
    torch.manual_seed(11)
    new = make_model()
    old_params = dict(legacy.named_parameters())
    for name, parameter in new.named_parameters():
        if not name.startswith("readout."):
            torch.testing.assert_close(parameter, old_params[name], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph numerical check")
def test_learned_probes_compiled_graph_preserves_feature_and_gradients(monkeypatch):
    monkeypatch.setenv("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER", "0")
    from torch._inductor import config as inductor_config
    monkeypatch.setattr(inductor_config, "use_static_cuda_launcher", False)
    torch.manual_seed(19)
    model = make_model().cuda()
    field = torch.randn(1, 4, 4, 4, 16, device="cuda", requires_grad=True)
    precision = torch.ones(1, 16, device="cuda")

    def loss_fn():
        feature, _ = model.readout(field, precision)
        return model.decoder(feature).square().mean()

    reference = make_model().cuda()
    reference.load_state_dict(model.state_dict())
    reference_field = field.detach().clone().requires_grad_(True)
    reference_feature, _ = reference.readout(reference_field, precision)
    eager_loss = reference.decoder(reference_feature).square().mean()
    eager_loss.backward()
    eager_grad = reference.readout.probe_coords.grad.clone()
    model.readout.forward = torch.compile(model.readout.forward, dynamic=False)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            model.zero_grad(set_to_none=False)
            if field.grad is not None:
                field.grad.zero_()
            loss_fn().backward()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        model.zero_grad(set_to_none=False)
        field.grad.zero_()
        captured_loss = loss_fn()
        captured_loss.backward()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured_loss, eager_loss, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(model.readout.probe_coords.grad, eager_grad, rtol=2e-4, atol=2e-6)
