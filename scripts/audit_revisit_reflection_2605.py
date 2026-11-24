"""2594 reuse -> 2605 revisit: private local-value intervention; 2677 is evaluation only."""
from dataclasses import replace
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from scripts.audit_revisit_exchange import natural, observe, row, fact
from scripts.ib_local.action_value_pair_audit import first_action
from scripts.ib_local.action_value_cross_score import SettlementScorePort, cpu_state
from scripts.ib_local import autonomous_cycle, finite_use_value as finite
from scripts.ib_local.causal_relation_program import Evidence, Port, mechanism, revise, log_score
from scripts.ib_local.causal_use_value import calibrated, queries
from scripts.run_arrived_language_actor import main as runner

ROOT = Path.home()/'mengli-cf-revisit-2604-base'
NAMES = ('silence','generate','reuse')

def folder(i): return ROOT.with_name('mengli-cf-revisit-2605-'+NAMES[i])

@torch.no_grad()
def decision(engine, receipt, work):
    m=engine.model; st=finite.score_state(engine._program_question_state)
    old=row(natural,2594); ref=row(observe,2605)
    assert engine._shared_question_receipt['signature']==old['shared_unknown']['signature']==ref['shared_unknown']['signature']
    assert receipt['proposals']==ref['joint_language_search']['candidates'] and receipt['selected_proposal']==ref['joint_language_search']['selected']
    p=torch.tensor(receipt['nominal_probability'],dtype=torch.float64)
    torch.testing.assert_close(p,p.new_tensor(ref['choice_probability']),rtol=1e-5,atol=1e-6)
    arc=torch.load('scratch/cgb_counterfactual_2594_frozen.pt',weights_only=False,map_location='cpu')['forecast']
    past=torch.load('scratch/cgb_counterfactual_2594_port.pt',weights_only=False,map_location='cpu')
    a,b=2,1  # observed reuse, unexecuted generate on the factual history
    real=next(f for f in st.constraints if f.fact_id=='episode:2604' and f.kind==Port.MOTHER)
    torch.testing.assert_close(real.outcome.cpu(),fact(natural,'episode:2604').outcome)
    # Copy preserves the raw factual response. Lift only the old action contrast
    # by its minimum-norm raw representative, then re-encode in the current chart.
    w=arc['encoder_weight'].double(); delta=(arc['mean'][b]-arc['mean'][a]).double()
    lift=torch.linalg.pinv(w)@delta; torch.testing.assert_close(w@lift,delta)
    s=engine._internal_source.reshape(1,-1).expand_as(engine._internal_action); acts=engine._internal_action
    qs=queries(engine,engine._program_question_state); d=engine._program_forecasts['path']['mother']; mom=engine._momentum_decision_read[None].expand(len(acts),-1)
    def local(dist): return 10*calibrated(engine,finite.expected(engine,st,s,acts,dist,qs))
    original=local(d); torch.testing.assert_close(original,original.new_tensor(receipt['local']),atol=1e-5,rtol=1e-4)
    policy={'original':p.tolist()}; values={'original':original.tolist()}
    for name,raw in [('copy',real.outcome),('residual',real.outcome+lift.to(real.outcome))]:
        hyp=Evidence(past['source'][b:b+1].to(s),past['actions'][b:b+1].to(s),m.encoder(raw),Port.MOTHER,False,encoded=True)
        revised=revise(st,hyp,model=m)
        pred=mechanism(m,revised,s,acts,ridge=engine.relation_ridge,momentum=mom)
        val=local(pred); values[name]=val.tolist()
        policy[name]=(p.log()+10*engine.precision()*(val-original).double().cpu()).softmax(0).tolist()
    (work/'policy.json').write_text(json.dumps(dict(probability=policy,local=values,signature=engine._shared_question_receipt['signature'],endpoints=receipt['endpoint_sha256'],source=receipt['source_sha256'],proposals=receipt['proposals'],old_contrast_lift_error=float((w@lift-delta).abs().max())),indent=2))
    torch.save(dict(model=SettlementScorePort(m),state=cpu_state(st),ridge=engine.relation_ridge),work/'evaluation_port.pt')


def run(i):
    work=folder(i); dispatch=autonomous_cycle.dispatch
    def step(engine):
        with first_action(engine,i,lambda r:decision(engine,r,work),keep_actor_learning=True):
            return dispatch(engine)
    flags='--native-decision --native-word-execution --batch-native-words --batch-preview-scores --successor-distribution --joint-successor --amortized-decision --latent-action-planner --root-native-correction observe --persistent-rhythm coupled --retention-scope-prior off --activation-host-mib 512'.split()
    argv=[sys.argv[0],'--arrived-mode','observe','--checkpoint',str(ROOT/'body/last.pt.gz'),'--work',str(work),'--cycles','1',*flags]
    with patch.object(autonomous_cycle,'dispatch',step),patch.object(sys,'argv',argv): runner()


@torch.no_grad()
def score():
    for cycle in (2602,2603,2604):
        replay, original = row(ROOT,cycle), row(observe,cycle)
        assert all(replay[k]==original[k] for k in ('choice_probability','utterance_hex','world','shared_unknown'))
    future=Path('G:/cgb_speech_growth_2613_20261121/2741'); later=row(future,2677)
    target=fact(future,later['causal_program']['committed_episode_identity']); ref=row(observe,2605)
    assert later['shared_unknown']['signature']==ref['shared_unknown']['signature']
    policy=json.loads((folder(0)/'policy.json').read_text()); assert policy==json.loads((folder(2)/'policy.json').read_text())
    port=torch.load(folder(0)/'evaluation_port.pt',weights_only=False,map_location='cpu'); m,st,ridge=port.values()
    def value(state): return float(log_score(mechanism(m,state,target.source,target.action,ridge=ridge),m.encoder(target.outcome)).mean())
    base=value(st); gains=[]; contacts=[]
    for i,arm in enumerate((folder(0),observe,folder(2))):
        r=row(arm,2605); assert r['choice']==NAMES[i] and r['shared_unknown']['signature']==policy['signature']
        assert r['joint_language_search']['candidates']==policy['proposals']
        e=fact(arm,r['causal_program']['committed_episode_identity']); updated=revise(st,e,model=m)
        gains.append(value(updated)-base); contacts.append(dict(name=NAMES[i],start=r['world']['start'],match=r['world']['match']))
    rates={k:sum(x*y for x,y in zip(p,gains)) for k,p in policy['probability'].items()}
    report=dict(acquisition=2594,decision=2605,target=2677,controls=policy,contacts=contacts,evidence_gains=gains,expected_gain=rates,delta_vs_original={k:v-rates['original'] for k,v in rates.items()},eligible=rates['residual']>max(rates['copy'],rates['original']),scope='one common archived future target from the conditioned silence lineage; frozen pre-decision scoring chart and same eight original supports; exact finite-menu expected score, not three separate policy learning trajectories')
    path=Path('results/published/cgb_revisit_reflection_2605_20261123.json'); path.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report))

if __name__=='__main__':
    torch.set_num_threads(1)
    if sys.argv[1]=='score': score()
    else: run(int(sys.argv[1]))
