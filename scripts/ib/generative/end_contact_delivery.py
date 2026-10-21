"""Matched archived-state intervention on newly formed contact conduction."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.cgb.exchange_checkpoint import load_exchange, save_exchange
from scripts.ib.generative.compare_support_transport import digest, same


def main(args):
    archive = args.run/'coupled_complete.npz'
    before_hash = digest(archive)
    base, _ = load_exchange(archive)
    contact = np.linalg.norm(base.coordinates[base.pre]-base.coordinates[base.post], axis=1) == 0.
    # Hold slow composition fixed in all arms for this acute delivery check.
    # This is a new diagnostic copy, not an update/reset of the main archive.
    frozen = replace(base.params, binding_rate=0., transport_rate=0., release_rate=0.,
        assembly_rate=0., conversion_rate=0., active_rate=0., write_rate=0., recovery_rate=0.,
        material_diffusion=0., support_diffusion=0., drive_diffusion=0.)
    state = base.state_dict()
    arms = {}
    for label in ('intact', 'sham', 'contact_barrier'):
        model = type(base).from_state_dict(state)
        model.params = frozen
        if label == 'contact_barrier':
            model.unit_conductance[contact] = 0.
        # All arms begin with exactly the same q, reservoirs, world, U and G.
        assert model.energy() == base.energy() and model.free_energy() == base.free_energy()
        for _ in range(4):
            model.step(.125, world_voltage=None)
        assert model.t == base.t+.5
        for name in ('capacity', 'bound', 'material', 'trace', 'end_capacity'):
            np.testing.assert_array_equal(getattr(model, name), getattr(base, name))
        for v in model.balances().values():
            assert np.max(np.abs(v), initial=0.) < 1e-8
        arms[label] = model
    same(arms['intact'].state_dict(), arms['sham'].state_dict())
    delta = arms['intact'].charge-arms['contact_barrier'].charge
    assert np.max(np.abs(delta)) > 1e-8
    report = dict(source_archive=str(archive), source_sha256=before_hash,
        source_time_ms=base.t, final_time_ms=arms['intact'].t, dt_ms=.125, steps=4,
        probe_elapsed_ms=.5, sham_bitwise=True,
        initial_energy_equal=True, initial_free_energy_equal=True,
        intervention='zero unit conductance only on colocated newly formed bridges',
        common_freeze='all material reactions/diffusion frozen; finite world ports remain coupled',
        max_charge_difference=float(np.max(np.abs(delta))),
        l2_charge_difference=float(np.linalg.norm(delta)),
        material_trace_and_end_states_unchanged=True,
        maximum_balance_error=max(float(np.max(np.abs(v), initial=0.))
            for m in arms.values() for v in m.balances().values()),
        proof_scope='acute causal delivery by contact bridges; not learning benefit or full exploration',
        script_sha256=digest(Path(__file__)), main_archive_unchanged=before_hash == digest(archive))
    for label, model in arms.items():
        path = args.run/('delivery_'+label+'.npz')
        save_exchange(path, model, report)
        restored, _ = load_exchange(path)
        same(model.state_dict(), restored.state_dict())
    (args.run/'contact_delivery.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (args.run/'source_snapshot'/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args())
