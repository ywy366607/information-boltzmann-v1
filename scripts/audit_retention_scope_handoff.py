"""Replay recorded retention priorities with identity-bound parent reads."""
import argparse
import json
from pathlib import Path
import sqlite3
from statistics import mean
from types import SimpleNamespace

import torch

from scripts.ib_local import relation_retention_use as use


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = dict(scope='recorded pre-response priorities and identity-bound past moments; alternative victim only', arms={})
    torch.set_num_threads(1)
    for name in ('observe', 'apply'):
        work = a.work/name
        db_path = next((work/'body').glob('causal_archive_*.sqlite'))
        with sqlite3.connect(db_path.resolve().as_uri()+'?mode=ro', uri=True) as db:
            data = json.loads(db.execute('SELECT payload FROM unknown_retention LIMIT 1').fetchone()[0])
        kwargs = dict(mode='apply', betas=tuple(data['betas']), eps=data['eps'])
        rows = [json.loads(s) for s in (work/'events.jsonl').read_text(encoding='utf8').splitlines()]
        rows = [r for r in rows if r['autonomous_event']['kind'] == 'external']
        cases = []
        for r in rows:
            finite = r['causal_use']['finite_U']
            local = finite['fresh_use']; parent = finite['fresh_organization']
            trace = r['shared_unknown']['revision']['retention_use']['trace']
            selection = trace['selection']; ids = selection['identities']
            old = use.UseSnapshot(local['scope'], moments=tuple(tuple(t) for t in local['consumed']), **kwargs)
            shared = use.UseSnapshot(parent['scope'], moments=tuple(tuple(t) for t in parent['consumed']), **kwargs)
            h = torch.tensor(selection['geometric'], dtype=torch.float64)
            baseline = h*use.biases(old, ids, h).exp()
            torch.testing.assert_close(baseline, h.new_tensor(selection['priority']), rtol=1e-5, atol=1e-6)
            joined = use.with_prior(SimpleNamespace(retention_scope_prior_mode='relation'), old, shared, ids)
            priority = h*use.biases(joined, ids, h).exp()
            order = sorted(range(len(ids)), key=lambda i: ids[i])
            victim = ids[order[int(priority[order].argmin())]]
            cases.append(dict(cycle=r['cycle'], unknown=local['scope'], parent=parent['scope'],
                cold=not any(t[3] for t in old.moments),
                applicable_parent_ids=[t[0] for t in joined.prior_moments if t[3] > 0
                    and t[0] not in {x[0] for x in old.moments if x[3] > 0}],
                original_release=selection['released'], prior_release=victim,
                release_changes=selection['released'] != victim,
                U=r['metal_g'], **r['use_transport']['mean']))
        report['arms'][name] = dict(cases=cases,
            cold_cycles=sum(c['cold'] for c in cases),
            cold_with_parent=sum(c['cold'] and bool(c['applicable_parent_ids']) for c in cases),
            changed_release=sum(c['release_changes'] for c in cases),
            last4={k: mean(c[k] for c in cases[-4:]) for k in ('append_gain', 'release_effect', 'actual_U_gain')})
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps({n:{k:v for k,v in m.items() if k!='cases'} for n,m in report['arms'].items()}))


if __name__ == '__main__':
    main()
