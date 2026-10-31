"""Routing/identity contracts, not learning or biological capability claims."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'scripts' / 'ib' / 'fly_interface'


def load(name):
    spec = importlib.util.spec_from_file_location(name, SOURCE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


compiler = load('compile_native_connectome_program')
linker = load('link_native_sensory_geometry')


def test_every_pair_keeps_identity_and_sensory_never_falls_back_to_soma():
    ids = np.array([90, 3, 44, 12, 87, 21, 18], dtype=np.int64)
    ann = pd.DataFrame({'bodyId': ids, 'class': ['Kenyon_Cell', 'MBON', '', '', '', '', ''],
                        'type': ['', '', '', 'APL', 'DPM', '', ''],
                        'superclass': ['central'] * 6 + ['vnc_sensory']})
    nt = pd.DataFrame({'body': ids, 'consensus_nt': ['acetylcholine', 'glutamate',
                       'dopamine', 'gaba', 'dopamine', 'octopamine', 'acetylcholine']})
    pre = np.repeat(np.arange(len(ids)), len(ids))
    post = np.tile(np.arange(len(ids)), len(ids))
    bits, counts, _, _ = compiler.compile_routes(ids, pre, post, ann, nt, ids[[2]])
    flags = compiler.ROUTES
    assert bits.shape == pre.shape and bits.dtype == np.uint16
    assert np.all(bits != 0)
    assert counts['sensory_local_return_required'] == len(ids)
    assert not np.any(bits[post == 6] & flags['fast_cell_baseline'])
    assert np.all(bits[pre == 4] & flags['DPM_local_cotransmission'])
    assert not np.any(bits[pre == 4] & flags['dopamine_delivery'])
    assert np.all(bits[(pre == 0) & (post == 1)] & flags['KC_MBON_contact_learning'])
    assert np.all(bits[(pre == 0) & (post == 2)] & flags['KC_DAN_local_feedback'])


def test_missing_sorted_identity_never_aliases_a_valid_endpoint():
    np.testing.assert_array_equal(linker.positions(np.array([2, 7, 19]), np.array([-1, 2, 3, 19, 20])),
                                  [-1, 0, -1, 2, -1])
    np.testing.assert_array_equal(linker.positions(np.array([], dtype=int), np.array([2, 3])), [-1, -1])


def test_native_geometry_keeps_disconnected_roots_and_capacity(tmp_path):
    path = tmp_path / 'roots.swc'
    np.savetxt(path, [[91, 0, 0, 0, 0, 2, -1], [17, 0, 10, 0, 0, 3, 91],
                      [53, 0, 0, 40, 0, 4, -1]])
    ids, xyz, radius, volume, child, parent, axial = linker.geometry(path)
    np.testing.assert_array_equal(ids, [91, 17, 53])
    np.testing.assert_array_equal(child, [1])
    np.testing.assert_array_equal(parent, [0])
    expected = np.pi * .08 * ((2 * .008) ** 2 + 2 * 3 * .008 ** 2 + (3 * .008) ** 2) / 3
    np.testing.assert_allclose(volume.sum(), expected + 4 * np.pi * (.032 ** 3) / 3)
    assert np.all(volume > 0) and np.all(axial > 0)


@pytest.mark.parametrize('rows', [
    [[1, 0, 0, 0, 0, 1, 2], [2, 0, 1, 0, 0, 1, 1]],
    [[1, 0, 0, 0, 0, 1, -1], [2, 0, 0, 0, 0, 1, 1]],
    [[1, 0, 0, 0, 0, 1, 8]],
])
def test_invalid_geometry_is_not_silently_repaired(tmp_path, rows):
    path = tmp_path / 'bad.swc'
    np.savetxt(path, rows)
    with pytest.raises(ValueError):
        linker.geometry(path)
