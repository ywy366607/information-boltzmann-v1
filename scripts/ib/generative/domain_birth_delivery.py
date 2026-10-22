"""Matched-clock delivery of actually grown domains; never replace the source."""
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
from information_boltzmann.cgb.exchange_checkpoint import load_exchange
from information_boltzmann.cgb.growing_material import GrowingIdentityMobileExchange
from scripts.ib.generative.compare_support_transport import digest, same
from scripts.ib.generative.domain_birth_audit import Observation
from scripts.ib.generative.native_learning_stream import advance_byte, atomic_json


def main(args):
    source = args.run/'growing_complete.npz'
    before_hash = digest(source)
    model, metadata = load_exchange(source)
    # Explicitly verify every forecast logit, not only the realized-target NLL.
    resume = {}
    block = np.load(args.run/'fresh_train_1280_1408.npy', allow_pickle=False)
    for label in ('growing', 'closed_domain_barrier'):
        m, meta = load_exchange(args.run/(label+'_mid.npz'))
        session = copy.deepcopy(meta['session'])
        observer = Observation(m, meta['growth_observer'])
        m.event_observer, m.domain_observer = observer.native, observer.domain
        expected = [json.loads(line) for line in (args.run/(label+'_scores.jsonl')).read_text().splitlines()]
        for index in range(64, 128):
            row = advance_byte(m, session, int(block[index]), 'fresh_C')
            assert all(value == expected[index][key] for key, value in row.items())
        final, final_meta = load_exchange(args.run/(label+'_complete.npz'))
        same(m.state_dict(), final.state_dict())
        assert session == final_meta['session'] and observer.state == final_meta['growth_observer']
        resume[label] = dict(future_bytes=64, all_logits_scores_and_full_state_exact=True)
    ids = metadata['growth_observer']['ids']
    initial_n = metadata['growth_observer']['initial_n']
    selected = np.array([uid >= initial_n for uid in ids])
    assert selected.sum() > 0
    with gzip.open(args.run/'growing_domains.jsonl.gz', 'rt', encoding='utf-8') as handle:
        domains = [json.loads(line) for line in handle]
    with gzip.open(args.run/'growing_events.jsonl.gz', 'rt', encoding='utf-8') as handle:
        events = [json.loads(line) for line in handle]
    births = [row for row in domains if row['accepted'] and row['kind'] == 'domain_birth']
    birth_pairs = {(row['observed_parent_id'], row['observed_child_id']) for row in births}
    contacts = [row for row in events if row['kind'] == 'assembly' and row['sign'] > 0
                and max(row['observed_pre_id'], row['observed_post_id']) >= initial_n]
    nonbirth_pairs = sorted({(row['observed_pre_id'], row['observed_post_id']) for row in contacts}-birth_pairs)
    next_raw = int(np.load(args.data/'train.npy', mmap_mode='r')[1408])
    if not 4 <= next_raw < 260:
        raise ValueError('Declare document boundary before this acute continuation')
    target = next_raw-4
    voltage = np.array([3. if (target >> bit) & 1 else -3. for bit in range(8)])
    branches = {name:GrowingIdentityMobileExchange.from_state_dict(model.state_dict())
                for name in ('intact', 'sham', 'cut_new_domain_conduction')}
    for branch in branches.values():
        # A stated acute diagnostic condition, shared by all branches. All
        # original slow learning, source reactions and real 4ms clock run.
        branch.domain_reactions_enabled = False
    cut = branches['cut_new_domain_conduction']
    original = cut.bridge_conductance
    cut.bridge_conductance = lambda: original() * ~(selected[cut.pre] | selected[cut.post])
    for branch in branches.values():
        for _ in range(2):
            branch.step(2., voltage)
    intact, sham = branches['intact'], branches['sham']
    same(intact.state_dict(), sham.state_dict())
    brain_difference = float(np.linalg.norm(intact.charge-cut.charge))
    assert brain_difference > 1e-12
    assert digest(source) == before_hash
    report = dict(source=str(source), source_sha256=before_hash, source_unchanged=True,
        resume=resume, actual_births=len(births),
        births_from_already_grown_parent=sum(row['observed_parent_id'] >= initial_n for row in births),
        native_assembly_events_touching_new_domains=len(contacts),
        native_contact_pairs_other_than_birth=nonbirth_pairs,
        actual_death_attempts=sum(row['kind'] == 'domain_death' for row in domains),
        actual_accepted_deaths=sum(row['accepted'] and row['kind'] == 'domain_death' for row in domains),
        selected_new_domain_ids=[uid for uid in ids if uid >= initial_n],
        interval_ms=[model.t, intact.t], next_actual_train_byte=1408, byte_value=target,
        same_initial_state_and_three_rngs=True, sham_full_state_exact=True,
        brain_charge_l2_difference=brain_difference,
        world_charge_l2_difference=float(np.linalg.norm(intact.world_charge-cut.world_charge)),
        branch_maximum_balance_errors={name:max(abs(float(v)) for v in branch.balances().values())
                                       for name,branch in branches.items()},
        scope='Acute diagnostic after actual paid growth, same real 4ms, all original learning active. New-domain birth held closed in all branches; ideal kinetic barrier changes only passive conduction on existing edges incident to the four new domains. Selection/barrier controller is not a natural mechanism. Cut is reconstructible from source+code+input; it is not saved as an uninstrumented full checkpoint. No learning benefit claim.')
    atomic_json(args.run/'delivery.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--run', type=Path, default=Path('F:/fly_stream_overflow/cgb_domain_birth_20261022'))
    p.add_argument('--data', type=Path, default=ROOT/'data/ib_owt_10000')
    main(p.parse_args())
