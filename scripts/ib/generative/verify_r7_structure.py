"""Independent spectral observation check of locked R7 structure results."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import eigh
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.cgb.exchange_checkpoint import load_exchange
from scripts.ib.generative.native_learning_stream import atomic_json
from scripts.ib.generative.r7_structure_constraints import structure_generator


def spectral_observation(generator, rho, clock, frames=60):
    """Reversible similarity transform; no expm/block integral from fitting."""
    t = generator[1:, 1:]
    weight = np.sqrt(rho[1:])
    symmetric = weight[:, None]*t/weight[None, :]
    np.testing.assert_allclose(symmetric, symmetric.T, atol=1e-15)
    eigenvalues, vectors = eigh(symmetric)
    scaled = eigenvalues*clock
    assert np.all(scaled < 0)
    integral_modes = np.expm1(scaled)/scaled
    birth_modes = vectors[0]/weight[0]
    first_mass = ((birth_modes*integral_modes) @ vectors.T)*weight
    visible = float(first_mass.sum())
    entry = first_mass/visible
    survival_modes = ((entry/weight) @ vectors)*(vectors.T @ weight)
    exponents = np.exp(np.arange(frames)[:, None]*scaled)
    survival = exponents @ survival_modes
    events = (exponents*(-np.expm1(scaled))) @ survival_modes
    return survival, events, entry, visible, np.sort(-scaled)


def main(args):
    base = args.base
    observations_path = base/'cgb_r7_observation_audit_20261022/observations.json'
    observations = json.loads(observations_path.read_text())
    control = sorted([w for w in observations['workbooks']
                      if w['kind'] == 'filopodium' and w['genotype'] == 'Control'], key=lambda w: w['file'])
    groups = {'train': control[:2], 'test': control[2:]}
    checked, diagnostics = [], []
    for folder in ('cgb_r7_structure_constraints_20261022',
                   'cgb_r7_structure_calibration_20261022',
                   'cgb_r7_joint_observables_20261022'):
        report = json.loads((base/folder/'report.json').read_text())
        lock = json.loads((base/folder/'training_locked.json').read_text())
        source = report['source']
        archive = Path(source.get('source_archive', source.get('parent_archive', '')))
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == source['parent_sha256']
        assert hashlib.sha256(observations_path.read_bytes()).hexdigest() == source['observations_sha256']
        parent, _ = load_exchange(archive)
        for row, locked in zip(report['rows'], lock['rows']):
            fit = row['fit']
            energy = row.get('structure_energy', fit.get('structure_energy', parent.params.structure_energy))
            q, rho, _ = structure_generator(replace(parent.params, structure_energy=energy), row['total_support'])
            saved_q = row.get('generator', fit.get('generator'))
            np.testing.assert_allclose(q, saved_q, rtol=1e-13, atol=1e-16)
            np.testing.assert_allclose(rho, row.get('stationary_distribution', fit.get('stationary_distribution')),
                                       rtol=1e-13, atol=1e-16)
            assert fit == locked['fit']
            clock = fit['clock_reference_ms_per_minute']
            survival, events, entry, visible, decays = spectral_observation(q, rho, clock)
            np.testing.assert_allclose(survival, row['survival_by_frame'], rtol=2e-7, atol=1e-10)
            np.testing.assert_allclose(entry, row['first_visible_state_distribution'], rtol=2e-7, atol=1e-10)
            np.testing.assert_allclose(visible, row['first_frame_visibility_probability'], rtol=2e-7)
            for split, workbooks in groups.items():
                values = []
                for workbook in workbooks:
                    for track in workbook['records']:
                        s, e = track['first_frame'], track['last_frame']
                        if s == 0:
                            continue
                        probability = events[e-s] if e < 59 else survival[e-s]
                        values.append(-np.log(probability))
                np.testing.assert_allclose(values, row[split+'_score']['per_track_nll'], rtol=2e-7, atol=1e-9)
                closure = row['count_closure']
                counts = np.concatenate([w['counts'] for w in workbooks])
                # In the high-energy calibration p_on can be ~1e-9. Summing
                # installed weights avoids cancellation in 1-rho_off.
                value = -float(binom.logpmf(counts, closure['candidate_locations'], float(rho[1:].sum())).mean())
                np.testing.assert_allclose(value, closure[split+'_mean_composite_nll'], rtol=1e-10)
            diagnostics.append(dict(run=folder, total_support=row['total_support'],
                                    decay_rates_per_minute=decays.tolist(),
                                    first_visible_fast_mode_to_slow_ratio=float(decays[1]/decays[0])))
        checked.append(folder)
    result = dict(passed=True, runs=checked, native_energy_and_rates_reproduced=True,
                  spectral_first_visibility_and_scores_reproduced=True,
                  locked_parameters_unchanged=True, observation_and_parent_hashes_verified=True,
                  biological_validation_complete=False, physical_individual_advanced=False,
                  spectral_diagnostics=diagnostics)
    atomic_json(base/'cgb_r7_joint_observables_20261022/independent_verification.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'spectral_diagnostics'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--base', type=Path, required=True)
    main(parser.parse_args())
