"""Publish actual finite-U accounting after the complete private run exits."""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--wait', action='store_true')
    p.add_argument('--arm', choices=('C', 'C2'), default='C')
    args = p.parse_args()
    while True:
        status = json.loads((args.work/'status.json').read_text(encoding='utf8'))
        if status['status'] == 'completed':
            break
        if status['status'] == 'failed' or not args.wait:
            raise RuntimeError(status)
        time.sleep(10)
    result_path = args.work/args.arm/'result.json'
    result = json.loads(result_path.read_text(encoding='utf8'))
    baseline = json.loads(Path('F:/mengli-contact-momentum-2401-20261118/A/result.json').read_text(encoding='utf8'))
    rows = [r for r in result['receipts'] if r.get('use_transport', {}).get('queries')]
    records = []
    lines = ['import reviews.NativeUseTransport', 'open CGB.NativeUseTransport',
             '/- Every tuple is an exact rational mean of the real, held-out query scores.',
             'Source: '+str(result_path)+' -/']
    def literal(x):
        return f'({x.numerator} / {x.denominator} : ℝ)'
    for i, r in enumerate(rows):
        t = r['use_transport']; q = t['queries']
        keys = ('before_action', 'old_support_now', 'appended_now', 'committed_now')
        values = [sum((Fraction(x[k]) for x in q), Fraction(0))/len(q) for k in keys]
        a, b, c, d = values
        lines.append(f'noncomputable def event{i} : QueryScores := ⟨'+', '.join(map(literal, values))+'⟩')
        for name, value in [('appendGain', c-b), ('actualGain', d-b), ('releaseEffect', d-c)]:
            op = '>' if value > 0 else '<' if value < 0 else '='
            lines.append(f'example : {name} event{i} {op} 0 := by norm_num [{name}, event{i}]')
        records.append(dict(event=r['causal_use']['clock'], paid=r['causal_use']['observed_use'],
                            expected=r['causal_use']['expected_use'],
                            removed=t['removed'], mean=t['mean'], queries=q))
    pairs = [y['score']-x['score'] for x, y in zip(baseline['evaluations'][-1][1]['queries'],
                                                   result['evaluations'][-1][1]['queries'])]
    for x, y in zip(baseline['evaluations'][-1][1]['queries'], result['evaluations'][-1][1]['queries']):
        assert (x['relation'], x['query']) == (y['relation'], y['query'])
    assert baseline['before'] == result['before']
    summary = dict(status='completed', events=len(rows), source=str(args.work),
        arm=args.arm,
        fixed_query_score_C_minus_A=sum(pairs)/len(pairs), improved=sum(v > 0 for v in pairs),
        fixed_queries=len(pairs),
        paid_vs_actual_sign_reversals=sum(r['paid']*r['mean']['actual_U_gain'] < 0 for r in records),
        append_vs_actual_sign_reversals=sum(r['mean']['append_gain']*r['mean']['actual_U_gain'] < 0 for r in records),
        means={k: sum(r['mean'][k] for r in records)/len(records)
               for k in ('append_gain', 'release_effect', 'actual_U_gain', 'representation_shift')},
        records=records, curves=dict(A=baseline['evaluations'], C=result['evaluations']))
    if args.arm == 'C2':
        summary['paid_actual_max_error'] = max(abs(r['paid']-r['mean']['actual_U_gain']) for r in records)
        summary['verified_commits'] = sum(r['causal_use']['finite_U'].get('verified', False)
                                          for r in result['receipts'])
        old = json.loads(Path('F:/mengli-sp-c-use-transport-2401-20261118/C/result.json').read_text(encoding='utf8'))
        assert old['before'] == result['before']
        assert [(q['relation'], q['query']) for q in old['evaluations'][-1][1]['queries']] == [
            (q['relation'], q['query']) for q in result['evaluations'][-1][1]['queries']]
        summary['fixed_query_score_C2_minus_C'] = (result['evaluations'][-1][1]['mean_score']
                                                  -old['evaluations'][-1][1]['mean_score'])
        summary['curves'] = dict(A=baseline['evaluations'], C=old['evaluations'], C2=result['evaluations'])
        summary['choice_changed_vs_C'] = sum(a['choice'] != b['choice']
            for a,b in zip(result['receipts'], old['receipts']))
        summary['expected_actual_rmse'] = (sum((r['expected']-r['mean']['actual_U_gain'])**2
            for r in records)/len(records))**.5
        summary['seconds'] = result['seconds']
    suffix = 'finite_C2' if args.arm == 'C2' else 'transport'
    output = ROOT/f'results/published/cgb_actual_U_{suffix}_complete_20261118.json'
    write(output, summary)
    lean = ROOT/'lean/CGBAxioms'
    proof = lean/('reviews/NativeUseFiniteC2Actual2401.lean' if args.arm == 'C2' else
                 'reviews/NativeUseTransportActual2401.lean')
    proof.write_text('\n'.join(lines)+'\n', encoding='utf8')
    env = os.environ.copy()
    env['ELAN_HOME'] = 'E:/lean/elan'
    env['LEAN_PATH'] = str(lean)
    commands = [['lake', 'env', 'lean', '-o', 'reviews/NativeUseTransport.olean', 'reviews/NativeUseTransport.lean'],
                ['lake', 'env', 'lean', str(proof.relative_to(lean))]]
    logs = []
    for cmd in commands:
        done = subprocess.run(cmd, cwd=lean, env=env, capture_output=True, text=True, timeout=240)
        logs.append(dict(command=cmd, code=done.returncode, output=done.stdout+done.stderr))
        if done.returncode:
            break
    summary['lean'] = logs
    summary['lean_passed'] = len(logs) == 2 and all(x['code'] == 0 for x in logs)
    write(output, summary)
    write(args.work/'analysis_status.json', dict(status='completed', result=str(output),
                                               lean_passed=summary['lean_passed']))
    print(json.dumps({k: v for k,v in summary.items() if k not in ('records','curves','lean')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
