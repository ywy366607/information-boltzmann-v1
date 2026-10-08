"""A transient Windows file lock must not kill a valid learning update."""
import json

from scripts.ib import train_medium_active_stream as entry


def test_spatial_snapshot_matches_sites_and_preserves_state():
    import torch
    from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
    from information_boltzmann.runtime.medium_health import spatial_medium_snapshot
    from information_boltzmann.runtime.training import belief_tensors
    torch.set_num_threads(1)
    torch.manual_seed(53)
    model = PlasticMediumPorts3D(vocab_size=17, shape=(2, 2, 2), channels=8,
        hidden=8, bath_type='conductance', activity_adaptation=True,
        short_term_plasticity=True, anisotropic_transport=True,
        read_mode='temporal', temporal_rates=[2., 5.], temporal_frequencies=[0., 3.])
    belief = model.initial_belief()
    before = [x.clone() for x in belief_tensors(belief)]
    rng = torch.get_rng_state().clone()
    report = spatial_medium_snapshot(model, belief)
    assert report['shape'] == [2, 2, 2]
    assert len(report['coordinates']) == len(report['field_energy']) == 8
    assert len(report['speed']) == len(report['material']) == 8
    assert all(v == [1., 1., 1.] for v in report['speed'])
    assert len(report['read_coords']) == 16
    assert report['read_heads'] * report['read_queries'] == len(report['read_coords'])
    torch.testing.assert_close(torch.tensor(report['read_footprint_by_probe']),
                               model.readout.weights().detach())
    json.dumps(report, allow_nan=False)
    assert torch.equal(rng, torch.get_rng_state())
    for x, y in zip(belief_tensors(belief), before):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert all(p.grad is None for p in model.parameters())
    assert max(report['write_port_displacement']) < 1e-6
    assert max(report['read_port_displacement']) < 1e-6
    assert len(report['transport_energy_current']) == 8
    assert len(report['effective_transport_factor']) == 8
    with torch.no_grad():
        model.write_agent.local_ports.centers[0, 0] += .02
        model.readout.probe_coords[0, 0, 1] += .03
    moved = spatial_medium_snapshot(model, belief)
    assert moved['write_port_displacement'][0] > .019
    assert moved['read_port_displacement'][0] > .029
    assert moved['read_coords'][0][1] != report['read_coords'][0][1]
    assert moved['read_footprint_by_probe'][0] != report['read_footprint_by_probe'][0]


def test_atomic_progress_replace_retries_transient_reader_lock(tmp_path, monkeypatch):
    replace = entry.os.replace
    attempts = []
    def locked_once(source, destination):
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError('Transient reader lock')
        return replace(source, destination)
    monkeypatch.setattr(entry.os, 'replace', locked_once)
    monkeypatch.setattr(entry.time, 'sleep', lambda duration: None)
    path = tmp_path / 'progress.json'
    entry.atomic_json(path, {'step': 3, 'status': 'training'})
    assert json.loads(path.read_text()) == {'step': 3, 'status': 'training'}
    assert len(attempts) == 3


def test_direct_pretrained_word_table_preserves_input_and_decoder_geometry(tmp_path):
    import torch
    from safetensors.torch import save_file
    from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
    model = PlasticMediumPorts3D(vocab_size=17, shape=(2, 2, 2), channels=4, hidden=8)
    table = torch.arange(68, dtype=torch.float32).reshape(17, 4) / 100
    source = tmp_path / 'words.safetensors'
    save_file({'wte.weight': table}, str(source))
    metadata = entry.initialize_pretrained_vocabulary(model, source)
    torch.testing.assert_close(model.source.embedding.weight, table, atol=0, rtol=0)
    torch.testing.assert_close(model.decoder.weight, table * metadata['decoder_global_scale'])
    assert metadata['method'] == 'direct word table'
    assert metadata['retained_squared_norm_fraction'] == 1
    assert not model.source.embedding.weight.requires_grad
    assert model.decoder.weight.requires_grad
