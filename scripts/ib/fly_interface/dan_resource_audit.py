"""Read-only release-resource accounting during matched native-clock replay.

Observes both native spike events and the actual transmitted amplitude. Does
not alter kinetics, reset pools, train, or call the mother. Threshold counts
describe the distribution; they are not health or intervention thresholds.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.core import fly_bptt_learning as fl
from information_boltzmann.core.fly_reservoir import FlyReservoirLM
from information_boltzmann.core.fly_dan_release import dan_state
from scripts.ib.fly_interface.native_system_matrix import audit


def distribution(value):
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    return dict(zip(('min', 'p01', 'p10', 'median', 'p90', 'p99', 'max'),
                    np.quantile(values, (0, .01, .1, .5, .9, .99, 1)).tolist()))


@contextmanager
def observe_release():
    finish = FlyReservoirLM._finish_coba_tick
    advance = fl.advance_dan_release
    rows, material = [], {}
    latest_spikes = None

    def observe_finish(model, *args, **kwargs):
        nonlocal latest_spikes
        result = finish(model, *args, **kwargs)
        latest_spikes = result[1].detach()
        return result

    def observe_advance(model, current, ring, kc_values):
        before = dan_state(model, current)[2]
        result = advance(model, current, ring, kc_values)
        owner = model.danl_site_owner
        ap = ring[0].index_select(1, model.danl_index)
        events = latest_spikes.index_select(1, model.danl_index)
        rate = model.dan_local_log_release_rate.exp()[owner].unsqueeze(0)
        rho = model.dan_local_logit_resource_recovery.sigmoid()[owner].unsqueeze(0)
        gain = model.log_kc_release.exp()[model.superclass_id[model.danl_index]]
        local = gain[owner].unsqueeze(0) * result[1][:, model.danl_site_node]
        probability = -torch.expm1(-rate * (ap[:, owner] + local))
        used = before * probability
        recovered = (1 - before + used) * (1 - rho)
        residual = result[2] - (before - used + recovered)
        resources = result[2].detach().cpu().numpy().reshape(-1)
        owner_np = owner.cpu().numpy()
        if not material:
            material.update(
                use_stp=bool(model.use_stp),
                graded_DAN=int(model.graded_mask[:, model.danl_index].sum())
                    if hasattr(model, 'graded_mask') else 0,
                recovery_rho=distribution(rho.detach().cpu().numpy()),
                recovery_tau_ticks=distribution((-1 / rho.log()).detach().cpu().numpy()),
                calcium_rho=distribution(model.dan_local_logit_calcium_decay.sigmoid().detach().cpu().numpy()),
                release_rate=distribution(rate.detach().cpu().numpy()),
                inherited_DA_slow_rho=float(model.dopamine_logit_slow.sigmoid()),
                DAN_graph_index=model.danl_index.cpu().tolist(),
                site_count_by_DAN=np.bincount(owner_np, minlength=model.danl_index.numel()).tolist())
        low = resources < .01
        rows.append(dict(
            feedback_enabled=bool(model.dan_local_feedback_enabled),
            native_spike_cells=int((events > 0).sum()),
            transmitted_active_cells=int((ap > 0).sum()),
            transmission_amplitude=distribution(ap.detach().cpu().numpy()),
            packet_vs_spike_absmax=float((ap-events).abs().max()),
            local_release_drive=distribution(local.detach().cpu().numpy()),
            release_probability=distribution(probability.detach().cpu().numpy()),
            resources=distribution(resources),
            sites_below_001=int(low.sum()),
            site_fraction_below_001=float(low.mean()),
            cells_with_sites_below_001=int(np.unique(owner_np[low]).size),
            resource_min_by_DAN=[float(resources[owner_np == i].min())
                                 for i in range(model.danl_index.numel())],
            released_total=float(used.sum()), recovered_total=float(recovered.sum()),
            accounting_absmax=float(residual.abs().max())))
        return result

    FlyReservoirLM._finish_coba_tick = observe_finish
    fl.advance_dan_release = observe_advance
    try:
        yield material, rows
    finally:
        FlyReservoirLM._finish_coba_tick = finish
        fl.advance_dan_release = advance


def run(checkpoint, output):
    with observe_release() as (material, rows):
        matched = audit(checkpoint, output, targets=1, systems=['dan_local_feedback'])
    ticks = matched['ticks']
    names = ['sensory_recording', *matched['arms']]
    if len(rows) != len(names) * ticks:
        raise RuntimeError('Observed physical ticks disagree with matched replay')
    report = dict(checkpoint=str(checkpoint), sha256=matched['sha256'],
        scope='Frozen real queued input; no new exposure, updates or world calls',
        controls_passed=matched['controls_passed'],
        source_and_rng_preserved=matched['source_and_rng_preserved'],
        intervals=matched['intervals'], material=material,
        arms={name: rows[i*ticks:(i+1)*ticks] for i, name in enumerate(names)})
    (output / 'resources.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'resource_trace': str(output / 'resources.json'),
                      'intact_final': report['arms']['intact'][-1]['resources']}), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.checkpoint.parent.resolve():
        parser.error('Use a separate diagnostic directory')
    run(args.checkpoint, args.output)
