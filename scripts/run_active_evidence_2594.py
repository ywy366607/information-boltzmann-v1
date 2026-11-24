"""Replay the saved 2588 boundary, then change only the 2594 actor leaf."""
import json
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch
import torch

from scripts.ib_local import autonomous_cycle, run_mother_continuation
from scripts.ib_local.action_value_pair_audit import first_action
from scripts.ib_local.action_value_cross_score import SettlementScorePort,cpu_state,cpu_fact,fact_digest
from scripts.ib_local.active_evidence_quality import acquisition_quality
from scripts.ib_local.causal_relation_program import Port,Evidence
from scripts.ib_local.finite_use_value import score_state
from scripts.ib_local.shared_unknown_bindings import _read_evidence
from scripts.audit_wood_policy_restore import inherited_projection
from information_boltzmann import budgeted_adamw_fused as kernel
from scripts import run_native_preview_restore_comparison as runner


def load(p): return json.loads(p.read_text(encoding='utf8'))
def external(p): return [r for line in p.open(encoding='utf8') if (r:=json.loads(line)).get('cycle')]


def main():
    campaign=Path('scratch/cgb_wood_policy_use_16rounds_20261120')
    original={r['cycle']:r for r in external(campaign/'natural/run/events.jsonl')}
    target_run=Path('G:/cgb_arrived_actor_2601_20261120/observe')
    root=original[2594]; phase=root['shared_unknown']; cycle=2588
    work=Path(sys.argv[1]); publication=Path(sys.argv[2]); packet={}; intervention=None
    base_dispatch=autonomous_cycle.dispatch
    def dispatch(e):
        nonlocal cycle,intervention
        if cycle==2593 and intervention is None:
            def capture(receipt):
                assert receipt['sampled_option']==2 and receipt['option']==1
                assert receipt['proposals']==root['joint_language_search']['candidates']
                assert receipt['selected_proposal']==root['joint_language_search']['selected']
                assert max(abs(p-q) for p,q in zip(receipt['nominal_probability'],root['choice_probability']))<1e-6
                state=score_state(e._program_question_state)
                packet.update(model=SettlementScorePort(e.model),state=cpu_state(state),receipt=receipt)
            intervention=first_action(e,1,capture,keep_actor_learning=True);intervention.__enter__()
        row=base_dispatch(e)
        if row['autonomous_event']['kind']!='external':return row
        cycle+=1
        if cycle<2594:
            runner.assert_replayed_event(row,original[cycle],cycle)
        else:
            intervention.__exit__(None,None,None); intervention=None
            later=next(r for r in external(target_run/'events.jsonl') if r['cycle']==2605)
            arc=e.model._causal_archive
            assert row['choice']=='generate' and row['shared_unknown']['signature']==phase['signature']==later['shared_unknown']['signature']
            assert row['utterance_hex']!=root['utterance_hex']
            assert later['shared_unknown']['recalled']
            def fact(r): return r['causal_program']['committed_episode_identity']
            alternate=cpu_fact(_read_evidence(arc,int(fact(row).split(':')[1]))[0])
            snapshot=Path(load(target_run/'body/last.metadata.json')['config']['causal_archive']['snapshot'])
            with sqlite3.connect(snapshot.resolve().as_uri()+'?mode=ro',uri=True) as db:
                def actual(r):
                    key=fact(r);data=arc._unpack(db.execute('SELECT payload FROM episodes WHERE id=?',(int(key.split(':')[1]),)).fetchone()[0]);assert int(data['count'])==1 and bool(data['mother_valid'])
                    s,a,o=[torch.tensor(data[k]).reshape(1,-1) for k in ('source','action','mother_next')];return Evidence(s,a,o,Port.MOTHER,True,stamp=int(data['last_write']),fact_id=key)
                factual,target=actual(root),actual(later)
            assert fact_digest(factual)!=fact_digest(alternate)
            fresh=later['causal_use']['finite_U']['fresh_use']; ids=set(fresh['benefits'])-{fact(root)}
            assert set(root['causal_use']['finite_U']['supports'])-{fact(root)}==set(row['causal_use']['finite_U']['supports'])-{fact(row)}==ids
            common=tuple(f for f in packet['state'].constraints if f.kind==Port.MOTHER and f.fact_id in ids)
            assert len(common)==len(ids)==7
            scores={k:acquisition_quality(packet['model'],packet['state'],common,x,target,e.relation_ridge) for k,x in [('reuse',factual),('generate',alternate)]}
            report=dict(acquisition=2594,revisit=2605,question=phase['identity'],signature=phase['signature'],background=sorted(ids),later=fact(later),prior_coordinates='same pre-world root model for both actions',later_from='original reuse trajectory',recorded_later_chart_reuse=fresh['benefits'][fact(root)],actions={k:dict(actual_receipt=fact_digest(x),**scores[k]) for k,x in [('reuse',factual),('generate',alternate)]},reuse_minus_generate=scores['reuse']['benefit']-scores['generate']['benefit'],pre_action_probability=root['choice_probability'],source_positions=[root['world']['start'],row['world']['start']],replayed_external=5,alternative_external=1)
            publication.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps(report),flush=True)
        return row
    projection=torch.load(campaign/'original_live_projection.pt',map_location='cuda',weights_only=False)['weight']
    sys.argv=[sys.argv[0],'--work',str(work),'--checkpoint',str(campaign/'boundary04/last.pt.gz'),'--cycles','6','--world-activations','D:/cgb_arrived_actor_2601_hot/world','--policy-activations',str(work/'policy'),'--native-decision','--native-word-execution','--successor-distribution','--joint-successor','--batch-native-words','--batch-preview-scores','--amortized-decision','--latent-action-planner','--persistent-rhythm','coupled','--root-native-correction','observe','--retention-scope-prior','off']
    with patch.object(kernel,'fused_step',kernel._launch),inherited_projection(projection),patch.object(autonomous_cycle,'dispatch',dispatch):runner.main()


if __name__=='__main__':main()
