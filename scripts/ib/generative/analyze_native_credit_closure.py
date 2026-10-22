"""Summarize all preregistered conditional trials, including failed returns."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.ib.generative.native_learning_stream import atomic_json


def numeric(row, metric):
    phase, key = metric.split('/')
    return float(row[phase].get(key, 0.))


def main(folder):
    report = json.loads((folder/'report.json').read_text())
    rows = report['rows']
    delays = report['source']['delays_ms']
    assert len(rows) == 8*len(delays)*4
    groups = []
    metrics = ['after_return/capture_rate', 'final/bound_support',
               'final/original_carrier_alive', 'after_configuration_washout/basal_installed_conductance',
               'after_configuration_washout/dismantle_rate']
    for delay in delays:
        selected = [row for row in rows if row['delay_ms'] == delay]
        contrasts = {}
        for metric in metrics:
            values = []
            for group in range(8):
                panel = {(r['marked'], r['requested_return']): numeric(r, metric)
                         for r in selected if r['group'] == group}
                assert len(panel) == 4
                values.append(panel[True, True]-panel[True, False]
                              -panel[False, True]+panel[False, False])
            contrasts[metric] = dict(values=values, paired_mean=float(np.mean(values)),
                                     minimum=float(min(values)), maximum=float(max(values)),
                                     scope='Eight future-noise realizations of one selected native path')
        conditions = []
        for marked in (False, True):
            for returned in (False, True):
                case = [r for r in selected if r['marked'] == marked and r['requested_return'] == returned]
                conditions.append(dict(marked=marked, requested_return=returned, n=len(case),
                    actual_returns=sum(r['return_applied'] for r in case),
                    target_binding_events=sum(r['target_events']['binding'] for r in case),
                    final_bound_fraction=float(np.mean([r['final']['bound_support'] for r in case])),
                    original_retained_fraction=float(np.mean([r['final']['original_carrier_alive'] for r in case])),
                    rebuilt_fraction=float(np.mean([r['final']['rebuilt'] for r in case])),
                    capture_rate_mean=float(np.mean([r['after_return']['capture_rate'] for r in case])),
                    all_activity_configurations_washed=all(r['after_configuration_washout']['activity_configuration'] == 0 for r in case)))
        groups.append(dict(delay_ms=delay, conditions=conditions, interaction_contrasts=contrasts))
    result = dict(source=report['source'], groups=groups,
                  trials=len(rows), exact_resume_trials=sum(r['resumed_future_exact'] for r in rows),
                  actual_bytes=sum(r['actual_bytes'] for r in rows),
                  actual_reference_ms=sum(r['actual_ms'] for r in rows),
                  max_balance_error=max(r['maximum_error'] for r in rows),
                  resource_usage_bound_passed=all(r['resource_budget']['used'] <=
                      r['resource_budget']['start_charged']+r['resource_budget']['recovered'] for r in rows),
                  all_failed_returns_retained=True, all_trial_source_states_retained_in_provenance=True,
                  prediction_credit_proven=False, autonomous_policy_proven=False,
                  score_scope='Conditional causal delivery and retention; not independent biological samples or long-term learning')
    atomic_json(folder/'analysis.json', result)
    print(json.dumps([{k: v for k, v in row.items() if k != 'conditions'} for row in groups], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run)
