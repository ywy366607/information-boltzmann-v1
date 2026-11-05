"""Read-only whole-loop report for the authorized symbol-world diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.ib_local.symbol_navigation_world import SymbolNavigationWorld, validation_stream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root
    rows = [json.loads(line) for line in (root/'metrics.jsonl').read_text().splitlines()]
    result = json.loads((root/'summary.json').read_text())
    late = rows[len(rows)//2:]
    world = SymbolNavigationWorld(11)
    digest = hashlib.sha256()
    for row in rows:
        response, heldout, info = world.take(bytes.fromhex(row['utterance_hex']),256,256)
        assert info == row['world']
        digest.update(response.tobytes()); digest.update(heldout.tobytes())
    periodic, random = (result['final_validation'][k] for k in ('periodic','random'))
    errors = result['late']['nll32_errors']
    report = dict(
        world_transition_replay=True, reconstructed_external_sha256=digest.hexdigest(),
        synthetic_validation_sha256=hashlib.sha256(validation_stream().tobytes()).hexdigest(),
        theoretical_random_entropy_nats=float(np.log(2)),
        final_periodic_nll=periodic, final_random_nll=random,
        interpretation=dict(
            both_rooms_visited=all(result['room_visits'].get(k,0)>0 for k in ('periodic','random')),
            periodic_is_more_predictable=periodic<random,
            raw_F_beats_unconditional_nll32=errors['raw']<errors['blind'],
            used_F_beats_unconditional_nll32=errors['used']<errors['blind'],
            used_F_beats_action_base_nll32=errors['used']<errors['base'],
            removing_action_features_hurts_F=errors['no_action']>errors['raw']),
        late_nonzero_wood_to_fire=sum(r['wood_policy_tv']>0 for r in late),
        late_nonzero_residual_to_fire=sum(r['residual_policy_tv']>0 for r in late),
        late_spoken_endings={chr(c):sum(bool(r['utterance_hex']) and
            bytes.fromhex(r['utterance_hex'])[-1]==c for r in late) for c in (97,98)},
        note='No semantic labels or room reward enter learning. Feature removal is a prediction diagnostic, not a counterfactual environment intervention.')
    (root/'loop_audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    config = json.loads((root/'config.json').read_text())
    curated = dict(summary=result, loop_audit=report,
                   parameter_count=config['parameter_count'],
                   protocol='docs/information_boltzmann/CGB_SYMBOL_WORLD_20261105.md',
                   source_snapshot_sha256=config['source_sha256'],
                   source_hashes=config['source_hashes'])
    Path('results/published/cgb_symbol_navigation_20261105.json').write_text(
        json.dumps(curated,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
