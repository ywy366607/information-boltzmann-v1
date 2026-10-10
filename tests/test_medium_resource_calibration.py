"""A resource measurement must retain the source learner's actual controls."""
import hashlib
from types import SimpleNamespace

import pytest

from scripts.ib.audit_medium_segment_graph import restore_calibration_runtime


def saved_stream(tmp_path):
    payload = {'train.npy': b'train bytes', 'validation.npy': b'validation bytes'}
    for name, value in payload.items():
        (tmp_path / name).write_bytes(value)
    manifest = b'{"tokenizer": "pinned"}'
    (tmp_path / 'manifest.json').write_bytes(manifest)
    return {
        'data_sha256': {name: hashlib.sha256(value).hexdigest() for name, value in payload.items()},
        'manifest_sha256': hashlib.sha256(manifest).hexdigest(),
        'structural_learning': {'dual_learning_rate': .0625},
    }


def test_resource_calibration_restores_source_dual_rate_and_validates_bytes(tmp_path):
    config = saved_stream(tmp_path)
    model = SimpleNamespace()
    provenance = restore_calibration_runtime(model, config, tmp_path)
    assert model.structure_dual_learning_rate == .0625
    assert provenance['verified_data_sha256'] == config['data_sha256']


@pytest.mark.parametrize('name', ['train.npy', 'validation.npy', 'manifest.json'])
def test_resource_calibration_rejects_data_change_before_runtime_mutation(tmp_path, name):
    config = saved_stream(tmp_path)
    model = SimpleNamespace(structure_dual_learning_rate=123.)
    (tmp_path / name).write_bytes(b'changed')
    with pytest.raises(ValueError, match='differs'):
        restore_calibration_runtime(model, config, tmp_path)
    assert model.structure_dual_learning_rate == 123.
