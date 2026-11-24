"""Actual pre-outcome gain data: context aliasing and chronological estimators.

Inputs are the frozen production support intervention and completed 32-event
traces. No body reset, world call, new label or optimizer update is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def moments(rows, name):
    n = len(rows)
    error = [(r[name]-r['target'])**2 for r in rows]
    return dict(n=n, mse=sum(error)/n)


def chronological(rows, field, decay=.95):
    # The production bounded slope rule: only previously arrived labels
    # determine a forecast's mixture coefficient. One set per actual option.
    cross, second, seen = [0.]*3, [0.]*3, [0]*3
    out = []
    for r in rows:
        option = r['choice']
        c, v = cross[option], second[option]
        alpha = max(0., min(1., c/v)) if seen[option] > 1 and v > 0 else 0.
        d, t = r[field]-r['base'], r['target']-r['base']
        p = r['base']+alpha*d
        out.append(dict(cycle=r['cycle'], alpha=alpha, prediction=p,
                        squared_error=(p-r['target'])**2))
        cross[option] = decay*c+(1-decay)*d*t
        second[option] = decay*v+(1-decay)*d*d
        seen[option] += 1
    return dict(mse=sum(r['squared_error'] for r in out)/len(out), rows=out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comparison', type=Path, default=Path('E:/cgb_preview_nvme_comparison_20261118'))
    p.add_argument('--support-report', type=Path,
                   default=Path('results/published/cgb_use_context_identity_fixed_2437_20261118.json'))
    p.add_argument('--output', type=Path,
                   default=Path('results/published/cgb_epistemic_readout_20261118.json'))
    args = p.parse_args()
    support = json.loads(args.support_report.read_text(encoding='utf8'))
    paired = support['observed_replay']['rows']
    full = paired[0]
    alternatives = [r for r in paired if r['state'].startswith('without:')]
    other = max(alternatives, key=lambda r: abs(r['expected'][0]-full['expected'][0]))
    assert full['raw']['input_hash'] == other['raw']['input_hash']
    assert full['raw']['prediction'] == other['raw']['prediction']
    mu0, mu1 = [10*r['expected'][0] for r in (full, other)]
    y0, y1 = [10*r['observed']['net_gain'] for r in (full, other)]
    out = dict(schema=1, contract='gain units = 10*finite-U nats/latent-coordinate',
        support_source=str(args.support_report),
        support_sha256=hashlib.sha256(args.support_report.read_bytes()).hexdigest(),
        alias=dict(states=[full['state'], other['state']],
            fixed_input_hash=full['raw']['input_hash'],
            query_ids=full['query_ids'],
            predicted_means=[mu0, mu1], observed_same_outcome_returns=[y0, y1],
            mean_difference=mu1-mu0,
            best_common_prediction_mse=(mu0-mu1)**2/4,
            proof_target='production predicted means under fixed Q and changed R',
            observed_scope='same actual outcome replay; not an estimate of conditional mean'), arms={})
    for arm in ('legacy', 'fixed'):
        path = args.comparison/arm/'events.jsonl'
        events = [json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]
        events = [r for r in events if 'pre_outcome_gain' in r]
        rows = [dict(cycle=2438+i, **r['pre_outcome_gain']) for i,r in enumerate(events)]
        assert len(rows) == 32
        assert all(abs(r['target']-10*e['metal_g']) < 1e-6 for r,e in zip(rows,events))
        stats = []
        for option in range(3):
            selected = [r for r in rows if r['choice'] == option]
            n = len(selected)
            c = sum((r['raw']-r['base'])*(r['target']-r['base']) for r in selected)/n
            v = sum((r['raw']-r['base'])**2 for r in selected)/n
            # Retrospective best bounded amplitude, explicitly descriptive.
            alpha = max(0., min(1., c/v)) if v > 0 else 0.
            stats.append(dict(option=option, n=n, cross=c, second=v,
                best_amplitude=alpha, base_mse=moments(selected,'base')['mse'],
                optimal_amplitude_mse=sum((r['base']+alpha*(r['raw']-r['base'])-r['target'])**2
                                          for r in selected)/n))
        out['arms'][arm] = dict(events=32, source=str(path),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            same_event_errors={name:moments(rows,name) for name in ('base','raw','expected')},
            raw_amplitude=stats,
            chronological_raw_mix=chronological(rows,'raw'),
            chronological_program_mix=chronological(rows,'expected'),
            chronological_halves=[dict(events=[start+1,start+16],
                **{name:moments(rows[start:start+16],name) for name in ('base','raw','expected')})
                for start in (0,16)],
            rows=rows)
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(alias=out['alias'], arms={k:dict(
        errors=v['same_event_errors'], amplitude=v['raw_amplitude'],
        chronological_program_mse=v['chronological_program_mix']['mse'])
        for k,v in out['arms'].items()}),ensure_ascii=False))


if __name__ == '__main__':
    main()
