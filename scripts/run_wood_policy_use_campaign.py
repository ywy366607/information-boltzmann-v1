"""Fixed 16-round natural line and three prespecified two-step policy-use probes."""
import argparse
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.run_root_native_correction_comparison import digest,write


def load(path):return json.loads(Path(path).read_text(encoding='utf8'))


def external(root):
    return [r for line in (root/'run/events.jsonl').read_text(encoding='utf8').splitlines()
            if (r:=json.loads(line))['autonomous_event']['kind']=='external']


def report_boundary(boundary,observation,outcomes,baseline,epsilon):
    import numpy as np
    from scripts.ib_local.wood_policy_measurement import contribution
    action=observation['action'];legal=action['legal']
    pi=[action['production'][a] for a in legal]
    blind=[action['without_final_wood_bias'][a] for a in legal]
    gains=[outcomes[str(a)]['G2'] for a in legal]
    g=contribution(pi,blind,gains)
    segment_gains=np.asarray([outcomes[str(a)]['segment_G2'] for a in legal])
    segment_c=(np.asarray(pi)-np.asarray(blind))@segment_gains
    loo=[float(np.mean(np.delete(segment_c,k))) for k in range(len(segment_c))]
    g.update(boundary=boundary,actions={str(a):dict(probability=pi[i],blind_probability=blind[i],
        wood_two_step_raw_U=action['total'][a]/10,**outcomes[str(a)]) for i,a in enumerate(legal)},
        segment_C=segment_c.tolist(),leave_one_segment_out_C=loo,
        material_sensitivity_only='four segments are one decision readout, not four decisions',
        robust_positive=g['C']>epsilon and min(loo)>0,
        robust_negative=g['C'] < -epsilon and max(loo)<0,
        blind_baseline=baseline,question=action['unknown'])
    best=legal[int(np.argmax(gains))]
    chains=[]
    for i,a in enumerate(legal):
        if pi[i] <= blind[i] or gains[i] >= g['blind_expected_G']-epsilon:continue
        wa=action['total'][a]/10;wb=action['total'][best]/10
        pa=outcomes[str(a)]['P2'];pb=outcomes[str(best)]['P2']
        chains.append(dict(raised_action=a,comparison_action=best,
            delta_probability=pi[i]-blind[i],actual_G2=gains[i],better_G2=max(gains),
            wood_difference=wa-wb,realized_payment_difference=pa-pb,
            realized_chain=('Wood ranks the harmful candidate higher and realized local payment does too'
                if wa>wb and pa>pb else
                'Wood ranks the harmful candidate higher while realized local payment ranks it lower'
                if wa>wb and pa<pb else 'probability redistribution across the complete candidate set')))
    g['raised_probability_failure_chains']=chains
    return g


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--temporary',type=Path,required=True)
    args=p.parse_args()
    assert not args.work.exists()
    source=Path('scratch/cgb_retention_scope_prior_2569_20261120/off/body/last.pt.gz').resolve()
    source_hash=digest(source)
    assert source_hash=='266f8866194c4e60887018cc914a6ebf1fb90b05a75087005e9df15f1caeff45'
    args.work.mkdir(parents=True)
    args.temporary.mkdir(parents=True,exist_ok=False)
    e_root=Path('E:/mengli-wood-policy-prefixes-20261120');e_root.mkdir(exist_ok=False)
    f_root=Path('F:/mengli-wood-policy-use-20261120');f_root.mkdir(exist_ok=False)
    prefixes={'4':str((args.work/'boundary04').resolve()),
              '10':str(e_root/'boundary10'),'16':str(e_root/'boundary16')}
    validation=Path('data/cgb_natural_text_20261115/validation.bin').resolve()
    assert digest(validation)=='b09c3879960cab5b5bbe3d3282059d472d7ee140925c58d7550b7fedebfbee93'
    starts=[32768,65536,131072,229376];old=[0,95629,191258,286887]
    assert all(s+2049<=validation.stat().st_size for s in starts)
    assert all(s+2049<=o or o+2049<=s for s in starts for o in old)
    protocol=dict(source=str(source),source_sha256=source_hash,source_cycle=2585,
        natural_rounds=16,observation_rounds=[4,10,16],prefix_paths=prefixes,
        horizon=2,max_private_rounds=18,natural_first_actor_learning='unchanged',
        branch_first_actor_learning='original nominal policy; only choice draw conditioned',
        probability='actual final logits softmax vs same logits minus final Wood offset; candidates retained',
        validation=dict(path=str(validation),sha256=digest(validation),starts=starts,
                        bytes_per_segment=2048,total_scored_bytes=8192,
                        prior_score_starts=old,used_in_training=False),
        epsilon=1e-5,distinguishability='C > 1e-5 nats/byte and same sign after leaving each material segment out; segments never counted as decisions',
        stopping='exactly three prespecified boundaries; no extra trials to select winners',
        exit_rules=dict(grow='all three robust positive',mixed='positive and negative directions: report scope and retain mechanisms',
                        negative='at least two negative and no positive: trace raised-probability harmful candidates',
                        near_zero='use same table, outcome span and probability moved, not another experiment'))
    protocol_path=args.work/'protocol.json';write(protocol_path,protocol)
    paths=[*Path('scripts/ib_local').glob('*.py'),Path(__file__),
           Path('scripts/run_wood_policy_use_worker.py'),Path('scripts/run_native_preview_restore_comparison.py')]
    hashes={str(path.resolve()):digest(path) for path in paths}
    write(args.work/'source_hashes.json',hashes)
    # Admit the complete disk footprint before any additional learning.
    assert shutil.disk_usage(args.work).free>1.9*1024**3,'natural/prefix/atomic-save space'
    assert shutil.disk_usage(e_root).free>1.1*1024**3,'two retained boundary sources'
    assert shutil.disk_usage(f_root).free>1.35*1024**3,'one serial private branch and atomic save'
    assert shutil.disk_usage(args.temporary).free>(3.51+.5+.125)*1024**3,'backward plus two fast-table worksets'
    state=dict(status='running',pid=os.getpid(),started=datetime.now(timezone.utc).isoformat(),
               source=str(source),source_sha256=source_hash,natural_rounds=0,private_rounds=0,
               completed_boundaries=[],stage='natural')
    def update(**kw):state.update(kw);write(args.work/'pipeline_status.json',state)
    def unchanged():
        assert digest(source)==source_hash
        assert all(digest(p)==h for p,h in hashes.items()),'runtime changed'
    def run(mode,checkpoint,work,temp,option=None):
        temp.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-m','scripts.run_wood_policy_use_worker','--mode',mode,
            '--checkpoint',str(checkpoint),'--work',str(work),'--temporary',str(temp),
            '--protocol',str(protocol_path)]
        if option is not None:command+=['--option',str(option)]
        with (args.work/(work.name+'.log')).open('w',encoding='utf8') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            update(child_pid=child.pid)
            assert child.wait()==0,f'{mode} worker failed: {work}'
        unchanged()
        return load(work/'run/result.json')
    try:
        natural=args.work/'natural'
        result=run('natural',source,natural,args.temporary/'natural')
        assert result['cycles']==16 and result['updates']==64 and result['provider_calls']==0
        update(natural_rounds=16,stage='branches')
        observations=load(natural/'observations.json')
        readings=load(natural/'independent_scores.json')['readings']
        rows=external(natural)
        reports=[]
        for boundary in protocol['observation_rounds']:
            obs=observations[str(boundary)];action=obs['action']
            baseline=readings[f'before_{boundary}'];outcomes={}
            checkpoint=Path(obs['source']);assert digest(checkpoint)==obs['source_sha256']
            for option in action['legal']:
                update(boundary=boundary,option=option)
                if option==action['option'] and boundary+1<=16:
                    final=readings[str(boundary+1)]
                    outcomes[str(option)]=dict(G2=baseline['mean_nll']-final['mean_nll'],
                        segment_G2=[a-b for a,b in zip(baseline['segment_nll'],final['segment_nll'])],
                        P2=rows[boundary-1]['metal_g']+action['gamma']*rows[boundary]['metal_g'],
                        reuse='actual natural trajectory; same first choice and original nominal learning')
                    continue
                assert state['private_rounds']+2<=18
                branch=f_root/f'b{boundary:02d}_a{option}'
                temp=args.temporary/'branches'/branch.name
                branch_result=run('branch',checkpoint,branch,temp,option)
                assert branch_result['cycles']==2 and branch_result['updates']==8
                assert branch_result['provider_calls']==0
                assert branch_result['initial_source_sha256']==obs['source_sha256']
                receipt=load(branch/'pre_outcome_action.json')
                scores=load(branch/'independent_scores.json')['readings']
                assert receipt['behavior']['categorical_score_gradient']==1
                assert receipt['sampled_option']==action['sampled_option']
                assert load(branch/'independent_scores.json')['material']==protocol['validation']
                for key in ('source_sha256','endpoint_sha256','recurrent_sha256','unknown','questions',
                            'proposals','selected_proposal','local','future','total','gamma','habit','offset'):
                    assert receipt[key]==action[key],(boundary,option,key)
                assert max(abs(p-q) for p,q in zip(receipt['nominal_probability'],action['production']))<1e-7
                assert abs(scores['0']['mean_nll']-baseline['mean_nll'])<1e-7
                assert max(abs(p-q) for p,q in zip(scores['0']['segment_nll'],baseline['segment_nll']))<1e-7
                events=external(branch)
                assert all(r['causal_use']['finite_U']['verified'] for r in events)
                outcomes[str(option)]=dict(G2=baseline['mean_nll']-scores['2']['mean_nll'],
                    segment_G2=[a-b for a,b in zip(baseline['segment_nll'],scores['2']['segment_nll'])],
                    P2=events[0]['metal_g']+action['gamma']*events[1]['metal_g'],
                    reuse=False,branch=str(branch),checkpoint_sha256=branch_result['checkpoint_sha256'])
                update(private_rounds=state['private_rounds']+2)
                write(args.work/f'boundary{boundary:02d}_outcomes.json',outcomes)
                # Reversible archive only; active sources and natural terminal stay local.
                subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',
                    'scripts/maintenance/Archive-PolicyMeasurementBranch.ps1',
                    '-Root',str(branch),'-Temporary',str(temp)],check=True,
                    stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
            report=report_boundary(boundary,obs,outcomes,baseline,protocol['epsilon'])
            reports.append(report)
            write(args.work/f'boundary{boundary:02d}_report.json',report)
            update(completed_boundaries=[r['boundary'] for r in reports])
        import numpy as np
        cs=[r['C'] for r in reports];epsilon=protocol['epsilon']
        positive=sum(c>epsilon for c in cs);negative=sum(c < -epsilon for c in cs)
        decision=('return_to_longer_natural_growth' if all(r['robust_positive'] for r in reports) else
                  'mixed_scope_keep_mechanisms' if positive and negative else
                  'trace_specific_failed_choices' if negative>=2 and positive==0 else
                  'near_zero_read_existing_probability_outcome_table' if max(abs(c) for c in cs)<=epsilon else
                  'material_or_boundary_dependent_keep_mechanisms')
        final=dict(completed=True,protocol=protocol,boundaries=reports,mean_C=float(np.mean(cs)),
            decision=decision,natural_learning=dict(initial=readings['0'],final=readings['16'],
                G16=readings['0']['mean_nll']-readings['16']['mean_nll'],checkpoint=result['checkpoint']),
            budget=dict(natural_rounds=16,private_rounds=state['private_rounds'],max_private_rounds=18),
            unit='nats/byte, final probability redistribution conditional on the actual candidates',
            decision_count=3,material_segments_per_readout=4,production_changed=False)
        output=Path('results/published/cgb_wood_policy_use_16rounds_20261120.json')
        write(output,final);update(status='completed',stage='completed',report=str(output),decision=decision,mean_C=final['mean_C'])
    except Exception as error:
        update(status='failed',error=str(error));raise


if __name__=='__main__':main()
