"""Summarize measured credit without treating support as a pruning guarantee."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, default=Path('results/fly_bptt32_credit_structure/report.json'))
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding='utf-8'))
    windows = report['measurements']
    median = lambda values: float(np.median(values))
    out = {key: report[key] for key in ('checkpoint_bptt_tokens', 'checkpoint_train_cursor',
                                       'checkpoint_updates', 'completed_windows', 'scope')}
    out['measured_real_tokens'] = len(windows)*report['window']
    out['largest_replay_state_error'] = max(w['replay_state_max_abs_error'] for w in windows)
    out['edges'] = {kind: {key: median([w['edge_gradients'][kind][key] for w in windows])
        for key in ('norm', 'exact_zero_fraction', 'fraction_with_any_arriving_pulse',
                    'fraction_for_90pct_power', 'fraction_for_99pct_power',
                    'power_share_top_1pct', 'inactive_edge_max_abs_gradient')}
        for kind in ('e', 'i')}
    endpoints = [w['endpoint_credit'][-1] for w in windows]
    lookup = [{row['lag']: row['states'] for row in item['states_by_lag']} for item in endpoints]
    out['endpoint_state_credit_by_lag'] = {}
    for lag in (0, 1, 2, 4, 8, 16, 24, 32):
        out['endpoint_state_credit_by_lag'][lag] = {
            state: {key: median([rows[lag][state][key] for rows in lookup])
                for key in ('norm', 'exact_zero_fraction', 'relative_state_perturbation_rms_response')}
            for state in lookup[0][lag]}
    out['spatial_support_lag16'] = {state: {key: median([rows[16][state][key] for rows in lookup])
        for key in ('fraction_for_90pct_power', 'fraction_for_99pct_power',
                    'credit_power_on_currently_spiking_neurons', 'spiking_fraction')}
        for state in ('h', 'ge', 'gi', 'b', 'x', 'u', 'ring0')}
    out['h_credit_norm_ratio_to_lag1'] = {lag: {
        'median': median([rows[lag]['h']['norm']/max(rows[1]['h']['norm'], 1e-30) for rows in lookup]),
        'min': min(rows[lag]['h']['norm']/max(rows[1]['h']['norm'], 1e-30) for rows in lookup),
        'max': max(rows[lag]['h']['norm']/max(rows[1]['h']['norm'], 1e-30) for rows in lookup)}
        for lag in (2, 4, 8, 16, 24, 32)}
    if report.get('drive_gradient_support', '').startswith('full'):
        out['drive_credit_scope'] = 'excluded: raw full-field drive adjoint includes physically forbidden injection coordinates'
    else:
        out['drive_credit_scope'] = 'actual sensory injection indices only'
        out['drive_credit_norm_by_lag'] = {lag: median([
            next(row['norm'] for row in endpoint['input_drive_by_lag'] if row['lag']==lag)
            for endpoint in endpoints]) for lag in (0, 1, 2, 4, 8, 16, 24, 31)}
    out['superclass_h_credit_share_lag16'] = {
        name: median([rows[16]['h']['superclasses'][name]['power_share'] for rows in lookup])
        for name in lookup[0][16]['h']['superclasses']}
    out['physical_parameter_grad_norms'] = {name: median([
        w['physical_and_gate_parameter_gradients'][name]['norm'] for w in windows])
        for name in windows[0]['physical_and_gate_parameter_gradients']}
    output = args.report.with_name('summary.json')
    output.write_text(json.dumps(out, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
