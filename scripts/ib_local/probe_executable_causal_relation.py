"""Fit and kernel-check one bounded relation on existing real-body receipts.

Offline research diagnostic only: case 0/1 fit, case 2 held out. It does not
change the body, does not use ground-truth needle labels, and does not claim
the model wrote this candidate or discovered a general causal law.
"""
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from scripts.ib_local.executable_causal_relation import Instruction, execute, linear_program

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/cgb_executable_relation_20261109'
LEAN_PROJECT = ROOT/'lean/CGBAxioms'
SCALE = 1_000_000


def examples(case):
    path=ROOT/f'results/cgb_reactivated_causal_credit_20261109/removal_checked_{case}/report.json'
    result=json.loads(path.read_text(encoding='utf-8'))
    rows=[]
    for record in result['researcher_finite_removals']:
        layer=record['layer']
        teacher=result['layers'][str(layer)]['teacher_norms'][str(record['slot'])]
        outcome=record['delta_ce']
        if not math.isfinite(teacher) or not math.isfinite(outcome):
            raise ValueError('nonfinite original receipt')
        rows.append(dict(case=case,layer=layer,slot=record['slot'],
                         x=[round(teacher*SCALE),int(layer==13)],y=round(outcome*SCALE)))
    return rows


def lean_source(program, rows):
    def instruction(i):
        if i.op in ('input','const'):
            return f'Op.{i.op} ({i.arg})'
        return f'Op.{i.op}'
    body=', '.join(instruction(i) for i in program)
    lines=['import Std', '', 'inductive Op where', '  | input : Nat → Op',
           '  | const : Int → Op', '  | add : Op', '  | mul : Op', '',
           'def run (xs : List Int) : List Op → List Int → Option Int',
           '  | [], [v] => some v', '  | [], _ => none',
           '  | .input i :: p, s => match xs[i]? with',
           '    | some x => run xs p (x :: s)', '    | none => none',
           '  | .const x :: p, s => run xs p (x :: s)',
           '  | .add :: p, a :: b :: s =>',
           '    let v := b + a',
           '    if v.natAbs ≤ 1000000000000000 then run xs p (v :: s) else none',
           '  | .mul :: p, a :: b :: s =>',
           '    let v := b * a',
           '    if v.natAbs ≤ 1000000000000000 then run xs p (v :: s) else none',
           '  | _, _ => none', '', f'def candidate : List Op := [{body}]',
           'example : candidate.length ≤ 16 := by decide']
    for row in rows:
        output=execute(program,row['x'])
        lines.append(f"example : run [{row['x'][0]}, {row['x'][1]}] candidate [] = some ({output}) := by decide")
    return '\n'.join(lines)+'\n'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    train=examples(0)+examples(1)
    held=examples(2)
    x=np.array([[1,*r['x']] for r in train],dtype=np.float64)
    y=np.array([r['y'] for r in train],dtype=np.float64)
    coefficients=np.linalg.lstsq(x,y,rcond=None)[0]
    bias,slope,layer=[int(round(float(v))) for v in coefficients]
    program=linear_program(bias,slope,layer)
    for row in train+held:
        row['predicted']=execute(program,row['x'])
    text=lean_source(program,train+held)
    lean_file=OUT/'CandidateExecution.lean'
    lean_file.write_text(text,encoding='utf-8')
    result=subprocess.run(['lake','env','lean',str(lean_file)],cwd=LEAN_PROJECT,
                          capture_output=True,text=True,timeout=120)
    (OUT/'lean.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    if result.returncode != 0:
        raise RuntimeError('Lean candidate execution check failed: '+result.stdout+result.stderr)
    baseline=int(round(float(y.mean())))
    def stats(rows):
        return dict(count=len(rows),mae_nats=sum(abs(r['predicted']-r['y']) for r in rows)/(len(rows)*SCALE),
                    mean_baseline_mae_nats=sum(abs(baseline-r['y']) for r in rows)/(len(rows)*SCALE))
    report=dict(status='complete',scope='offline bounded relation on actual finite-removal receipts',
        candidate_origin='researcher regression, not model-generated', optimizer_updates=0,body_updates=0,
        empirical_claim='observed removal effect prediction; not established general causal law',
        inputs=['current_teacher_norm_micro','human_layer_13_indicator'],output='removal_delta_ce_micro_nats',
        program=[dict(op=i.op,arg=i.arg) for i in program],program_operations=len(program),
        train_cases=[0,1],held_case=2,train=stats(train),held=stats(held),receipts=train+held,
        lean=dict(returncode=result.returncode,checked_executions=len(train+held),file=str(lean_file),
                  scope='arithmetic execution on listed inputs and program length; not world truth'))
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
