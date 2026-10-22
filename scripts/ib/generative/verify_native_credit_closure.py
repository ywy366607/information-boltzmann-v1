"""Verify full-state sham, checkpoint continuation and native credit receipts."""
from __future__ import annotations

import argparse
import copy
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.cgb.credit_observation import contact_roles, resource_bandwidth, support_return_paths
from information_boltzmann.cgb.exchange_checkpoint import load_exchange
from information_boltzmann.cgb.material_fork import fork_material
from scripts.ib.generative.compare_support_transport import same
from scripts.ib.generative.domain_rate_audit import CarrierObservation
from scripts.ib.generative.native_credit_closure import digest
from scripts.ib.generative.native_learning_stream import advance_byte, atomic_json


def main(folder):
    report = json.loads((folder/'report.json').read_text())
    source = report['source']
    assert digest(source['parent_archive']) == source['parent_sha256']
    block = np.load(folder/'actual_input.npy', allow_pickle=False)
    for file, expected in source['code_sha256'].items():
        assert digest(folder/'source_snapshot'/file) == expected
    budgets = {r['arm']: dict(used=0, recovered=0) for r in report['rows']}
    event_count = 0
    with gzip.open(folder/'events.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            item = json.loads(line)
            row, budget = item['event'], budgets[item['arm']]
            event_count += 1
            if row['kind'] in ('domain_birth', 'domain_death'):
                if row['accepted']:
                    budget['used' if row['kind'] == 'domain_birth' else 'recovered'] += 1
            else:
                for name, node, change in row['composition_changes']:
                    if name == 'charged':
                        budget['recovered' if change > 0 else 'used'] += abs(change)
    for row in report['rows']:
        assert row['actual_ms'] == row['delay_ms']+32
        assert row['actual_bytes']*4 == row['actual_ms']
        assert row['resource_budget']['used'] == budgets[row['arm']]['used']
        assert row['resource_budget']['recovered'] == budgets[row['arm']]['recovered']
    restored = []
    for row in report['rows']:
        if not row['resumed_future_exact']:
            continue
        label = row['arm']
        model, meta = load_exchange(folder/(label+'_mid.npz'))
        session = meta['session']
        observer = CarrierObservation(model, meta['growth_observer'], meta['carrier_observer'])
        model.event_observer, model.domain_observer = observer.native, observer.domain
        for byte in block[4:8]:
            advance_byte(model, session, int(byte), 'conditional_credit')
        final, fmeta = load_exchange(folder/(label+'_complete.npz'))
        same(model.state_dict(), final.state_dict())
        assert session == fmeta['session']
        assert observer.state == fmeta['growth_observer'] and observer.carriers == fmeta['carrier_observer']
        restored.append(label)
    # Real native-state observer sham: all diagnostic reads leave future
    # physical fields, reactions, three RNGs and scoring session unchanged.
    model, meta = load_exchange(folder/'g0_d0_m1_r1_start.npz')
    other = fork_material(model)
    original = model.state_dict()
    resource_bandwidth(model)
    paths = support_return_paths(model)
    for edge in range(len(model.pre)):
        contact_roles(model, edge)
    same(original, model.state_dict())
    session, other_session = copy.deepcopy(meta['session']), copy.deepcopy(meta['session'])
    for byte in block[:2]:
        assert advance_byte(model, session, int(byte), 'conditional_credit') == advance_byte(other, other_session, int(byte), 'conditional_credit')
    same(model.state_dict(), other.state_dict())
    result = dict(passed=True, source_hashes_verified=True, source_archive_unchanged=True,
                  conditional_trials_verified=len(report['rows']), receipts_verified=event_count,
                  full_checkpoint_future_reproduced=restored,
                  observer_sham_same_future_physical_state_three_rngs_and_session=True,
                  all_resource_receipts_and_actual_clocks_reproduced=True,
                  predictive_or_biological_credit_validated=False)
    atomic_json(folder/'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run)
