"""Match frozen prior reads, current arrived labels, and recorded release choices."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
from statistics import mean

import torch

from scripts.ib_local import relation_retention_use as use


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    out = dict(scope='same recorded current state; replay local-only release identity and compare prior sign with arrived MOTHER marginal score',
               alternative_release='identity only; subsequent alternative world outcomes are not generated', arms={})
    for name in ('off', 'relation'):
        work = a.work/name
        db_path = next((work/'body').glob('causal_archive_*.sqlite'))
        with sqlite3.connect(db_path.resolve().as_uri()+'?mode=ro', uri=True) as db:
            units = json.loads(db.execute('SELECT payload FROM unknown_retention LIMIT 1').fetchone()[0])
        rows = [json.loads(s) for s in (work/'events.jsonl').read_text(encoding='utf8').splitlines()]
        rows = [r for r in rows if r['autonomous_event']['kind'] == 'external']
        details, pairs = [], []
        for r in rows:
            fresh = r['causal_use']['finite_U']['fresh_use']
            selection = r['shared_unknown']['revision']['retention_use']['trace']['selection']
            own = use.UseSnapshot(fresh['scope'], mode='apply', betas=tuple(units['betas']), eps=units['eps'],
                                  moments=tuple(tuple(v) for v in fresh['consumed']))
            joined = replace(own, prior_moments=tuple(tuple(v) for v in fresh.get('prior_consumed', [])))
            ids = selection['identities']
            h = torch.tensor(selection['geometric'], dtype=torch.float64)
            actual = h*use.biases(joined, ids, h).exp()
            torch.testing.assert_close(actual, h.new_tensor(selection['priority']), rtol=1e-5, atol=1e-6)
            local = h*use.biases(own, ids, h).exp()
            order = sorted(range(len(ids)), key=lambda j: ids[j])
            original = ids[order[int(actual[order].argmin())]]
            assert original == selection['released']
            other = ids[order[int(local[order].argmin())]]
            observed = {v[0] for v in own.moments if v[3] > 0}
            active = [v for v in joined.prior_moments if v[3] > 0 and v[0] not in observed]
            for identity, first, second, count in active:
                if identity in fresh['benefits']:
                    now = fresh['benefits'][identity]
                    pairs.append(dict(cycle=r['cycle'], identity=identity, count=count,
                                      prior_first=first, current_benefit=now, sign_matches=first*now > 0))
            details.append(dict(cycle=r['cycle'], active_prior=len(active),
                                prior_release=original, local_only_release=other, changed=other != original,
                                released_fresh_benefit=fresh['benefits'].get(original),
                                alternative_fresh_benefit=fresh['benefits'].get(other),
                                use_transport=r['use_transport']['mean']))
        out['arms'][name] = dict(selection=details, prior_current_pairs=pairs,
                                sign_matches=sum(v['sign_matches'] for v in pairs), compared=len(pairs),
                                changed_release=sum(v['changed'] for v in details),
                                blocks_U=[mean(r['metal_g'] for r in rows[i:i+4]) for i in range(0, len(rows), 4)])
    a.output.write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps({n:{k:v for k,v in d.items() if k not in ('selection', 'prior_current_pairs')}
                      for n,d in out['arms'].items()}))


if __name__ == '__main__':
    main()
