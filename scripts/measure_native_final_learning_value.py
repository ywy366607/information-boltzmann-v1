"""Final longitudinal score accounting on the completed whole-body branches.

Old observations use the body's actual leave-one-fact-out predictions; future
observations use its original frozen forecasts. All matching keeps real fact
and world identities. No training or synthetic evaluation task is introduced.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from scipy.special import gammaln
from scripts.measure_native_inner_loop import ROOT, load, sha


def fingerprint(ref):
    return ref['tensor_ref']['sha256']


def dist_key(pred):
    return (fingerprint(pred['mean']), fingerprint(pred['scale2']))


def summary(values):
    return dict(count=len(values),mean=float(np.mean(values)) if values else None,
        minimum=float(min(values)) if values else None,maximum=float(max(values)) if values else None,
        positive=sum(v>0 for v in values),negative=sum(v<0 for v in values),zero=sum(v==0 for v in values))


def read_arm(directory):
    obs=directory/'observations'
    rows=[json.loads(x) for x in gzip.open(obs/'operations.jsonl.gz','rt',encoding='utf8')]
    calls={r['data']['call_id']:r for r in rows if r['kind']=='relation_signal'}
    external=[r for r in rows if r['kind']=='event' and r['data']['autonomous_event']['kind']=='external']
    event_round={r['event']:i+1 for i,r in enumerate(external)}
    world={r['event']:r['data'] for r in rows if r['kind']=='world'}
    scores=defaultdict(list)
    for r in calls.values():
        d=r['data']
        if d['operation']=='log_score' and d['caller']['function']=='_outcome_loss':
            scores[(r['event'],r['update_clock'],dist_key(d['prediction']))].append(r)
    old=[]; certificates=[]; score_errors=[]
    with (obs/'tensors.bin').open('rb') as stream:
        for r in calls.values():
            d=r['data']
            caller=d.get('caller',{})
            recall_caller=caller.get('function')=='_outcome_loss' or (
                caller.get('function')=='mechanism' and caller.get('module') in (
                    'scripts.native_transport_curl_intervention','scripts.native_pending_mode_intervention'))
            if not (d['operation']=='mechanism' and d['kind']=='mother' and d['exclude_fact']
                    and recall_caller):continue
            excluded=d['exclude_fact']
            fact=next(e for e in d['state']['constraints'] if e['kind']=='mother' and e['fact_id']==excluded)
            assert fact['observed'] and not fact['encoded']
            assert fingerprint(d['source'])==fingerprint(fact['source'])
            assert fingerprint(d['action'])==fingerprint(fact['action'])
            design=[v for v in calls.values() if v['data']['operation']=='_bound_evidence_design'
                    and v['data']['parent_id']==d['call_id']]
            assert len(design)==1
            support=[e['fact_id'] for e in design[0]['data']['facts']]
            assert excluded not in support
            hits=[v for v in scores[(r['event'],r['update_clock'],dist_key(d['prediction']))]
                  if v['data']['call_id']>d['call_id']]
            assert len(hits)==1, (directory,d['call_id'],len(hits))
            scored=hits[0];sd=scored['data']
            mean,var,target=[load(stream,v) for v in (sd['prediction']['mean'],sd['prediction']['scale2'],sd['target'])]
            df=sd['prediction']['df'];df=load(stream,df) if isinstance(df,dict) else df
            var=np.maximum(var,np.finfo(np.float32).eps)
            calculated=(gammaln((df+1)/2)-gammaln(df/2)-.5*np.log(df*math.pi*var)
                        -(df+1)/2*np.log1p((target-mean)**2/var/df)).mean()
            actual=float(load(stream,sd['score']).mean())
            error=abs(calculated-actual);score_errors.append(error)
            assert error<2e-5,(directory,error)
            fid=int(excluded.split(':')[1])
            certificates.append([r['event'],scored['event'],r['update_clock'],scored['update_clock'],
                d['call_id'],sd['call_id'],fid,fid,int(excluded not in support),int(fact['observed'])])
            old.append(dict(round=event_round[r['event']],event=r['event'],update_clock=r['update_clock'],
                relation=[d['state']['identity'],d['state']['generation']],version=d['state']['version'],
                fact=excluded,raw_identity=[fingerprint(fact[k]) for k in ('source','action','outcome')],
                support=support,mechanism_call=d['call_id'],score_call=sd['call_id'],
                log_score=actual,mse=float(np.square(target-mean).mean()),
                encoded_target_norm=float(np.linalg.norm(target)),score_reconstruction_error=error))
    future=[]
    for index,r in enumerate(external,1):
        d=r['data'];w=world[r['event']]
        def raw_digest(key):return hashlib.sha256(bytes.fromhex(w[key])).hexdigest()
        future.append(dict(round=index,event=r['event'],update_clock=r['update_clock'],
            response_sha=raw_digest('response_hex'),holdout_sha=raw_digest('holdout_hex'),
            utterance_sha=hashlib.sha256(bytes.fromhex(d['utterance_hex'])).hexdigest(),
            external_bytes=d['external_bytes'],anchor=d['world']['anchor'],choice=d['choice'],
            heldout_pre_nll=d['heldout_pre_nll'],heldout_post_nll=d['heldout_post_nll'],
            settlement=d['settlement'],
            witness={k:d.get(k) for k in ('witness_pre','witness_post','witness_current_pre','witness_current_post')},
            mother=d['causal_program']['mother_prequential'],
            decision=d['causal_program']['decision_prequential']))
    histories=defaultdict(list)
    for row in old:histories[row['fact']].append(row)
    reused={fid:dict(rounds=[x['round'] for x in items],relations=[x['relation'] for x in items],
        first_score=items[0]['log_score'],last_score=items[-1]['log_score'],
        first_to_last_delta=items[-1]['log_score']-items[0]['log_score'],
        raw_identity_persistent=all(x['raw_identity']==items[0]['raw_identity'] for x in items))
        for fid,items in histories.items() if len(items)>1}
    assert all(x['raw_identity_persistent'] for x in reused.values())
    return dict(old_facts=old,reused_facts=reused,future=future,
        max_score_reconstruction_error=max(score_errors,default=0),
        source=dict(path=str(obs/'operations.jsonl.gz'),sha256=sha(obs/'operations.jsonl.gz'))),certificates


def paired(a,b):
    def key(r):return (r['round'],r['fact'],tuple(r['raw_identity']))
    aa={key(r):r for r in a['old_facts']};bb={key(r):r for r in b['old_facts']}
    assert len(aa)==len(a['old_facts']) and len(bb)==len(b['old_facts'])
    olds=[]
    for k in sorted(aa.keys()&bb.keys()):
        x,y=aa[k],bb[k]
        olds.append(dict(round=k[0],fact=k[1],relation_a=x['relation'],relation_b=y['relation'],
            same_relation=x['relation']==y['relation'],same_support=x['support']==y['support'],
            log_score_delta=y['log_score']-x['log_score'],mse_delta=y['mse']-x['mse'],
            encoded_target_norm_delta=y['encoded_target_norm']-x['encoded_target_norm']))
    future=[]
    for x,y in zip(a['future'],b['future']):
        assert x['round']==y['round']
        same_world=x['response_sha']==y['response_sha'] and x['holdout_sha']==y['holdout_sha']
        def forecast_delta(port, metric):
            return y[port][metric]-x[port][metric] if x[port] is not None and y[port] is not None else None
        future.append(dict(round=x['round'],same_world=same_world,
            same_utterance=x['utterance_sha']==y['utterance_sha'],
            heldout_nll_delta=y['heldout_pre_nll']-x['heldout_pre_nll'],
            mother_log_score_delta=forecast_delta('mother','log_score'),
            decision_log_score_delta=forecast_delta('decision','log_score'),
            mother_mse_delta=forecast_delta('mother','mse'),
            decision_mse_delta=forecast_delta('decision','mse'),
            settlement_delta=y['settlement']-x['settlement']))
    # Later rounds isolate propagation beyond the first intervention event.
    later=[x for x in future if x['round']>1 and x['same_world'] and x['same_utterance']]
    common_length=min(len(a['future']),len(b['future']))
    af=a['future'][:common_length];bf=b['future'][:common_length]
    return dict(old_pairs=olds,future_pairs=future,
        matched_old_score=summary([r['log_score_delta'] for r in olds if r['round']>1]),
        matched_old_mse=summary([r['mse_delta'] for r in olds if r['round']>1]),
        later_identical_world_and_utterance={key:summary([r[key] for r in later if r[key] is not None]) for key in
            ('heldout_nll_delta','mother_log_score_delta','decision_log_score_delta','settlement_delta')},
        own_route=dict(meaning='each branch actual chosen stream over equal external-round prefix',rounds=common_length,
            mean_pre_nll_a=float(np.mean([x['heldout_pre_nll'] for x in af])),
            mean_pre_nll_b=float(np.mean([x['heldout_pre_nll'] for x in bf])),
            cumulative_settlement_a=sum(x['settlement'] for x in af),
            cumulative_settlement_b=sum(x['settlement'] for x in bf)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    setups={
        'inner':('native_inner_loop_use_2401_20261117',('baseline','bridge_cut','matched_cut','order_only')),
        'curl':('native_transport_curl_use_2401_20261117',('baseline','bridge_cut','matched_cut')),
        'structure':('native_identity_preview_bound_2401_20261117',('baseline','bridge_cut','matched_cut','order_only')),
        'pending':('native_relation_mode_use_2401_20261117',('baseline','pending_mode_cut','pending_matched'))}
    reports={};certificates=[]
    for label,(folder,names) in setups.items():
        arms={}
        for name in names:
            arms[name],cert=read_arm(ROOT/'scratch'/folder/name);certificates.extend(cert)
            print(f'{label}/{name}: {len(arms[name]["old_facts"])} actual old scores',flush=True)
        reports[label]=dict(arms=arms,comparisons={name:paired(arms['baseline'],arms[name]) for name in names[1:]})
    proof=ROOT/'lean/CGBAxioms/reviews/NativeFinalValueActual2401.lean'
    proof.write_text('''import Mathlib.Tactic
namespace CGB.NativeFinalValueActual2401
-- Actual prediction -> score pairs, own fact excluded from evidence.
def rows : List (List Nat) := DATA
def valid (r : List Nat) : Bool :=
  r.length == 10 && r[0]! == r[1]! && r[2]! == r[3]! &&
  r[4]! < r[5]! && r[6]! == r[7]! && r[8]! == 1 && r[9]! == 1
example : rows.all valid = true := by native_decide
example : rows.length = COUNT := by native_decide
end CGB.NativeFinalValueActual2401
'''.replace('DATA',str(certificates)).replace('COUNT',str(len(certificates))),encoding='utf8')
    done=subprocess.run(['lake','env','lean',str(proof.relative_to(ROOT/'lean/CGBAxioms'))],
        cwd=ROOT/'lean/CGBAxioms',capture_output=True,text=True,encoding='utf8')
    args.output.with_suffix('.lean.log').write_text(done.stdout+done.stderr,encoding='utf8')
    if done.returncode:raise RuntimeError(done.stdout+done.stderr)
    result=dict(status='completed',groups=reports,actual_old_score_pairs=len(certificates),
        lean=dict(path=str(proof),sha256=sha(proof),exit_code=0),main_cycle=2401,
        new_training_updates=0,new_world_calls=0,
        definitions=dict(old='actual own-fact-excluded recall score in then-current encoder coordinates',
            future='original frozen forecast and its frozen encoder; realized later target',
            paired='match raw fact identity, or actual world bytes plus utterance; keep endogenous routes separately'))
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({g:{a:{k:v for k,v in c.items() if k not in ('old_pairs','future_pairs')}
        for a,c in d['comparisons'].items()} for g,d in reports.items()},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
