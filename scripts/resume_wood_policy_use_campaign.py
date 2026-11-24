"""Finish the same fixed boundaries/budget after binding the original live cache."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.run_wood_policy_use_campaign import load,external,report_boundary
from scripts.run_root_native_correction_comparison import digest,write


def main():
    work=Path(sys.argv[1]).resolve()
    state=load(work/'pipeline_status.json')
    assert state['status']=='failed' and (state['error']=="(4, 1, 'source_sha256')" or
                                         'Archive-PolicyMeasurementBranch.ps1' in state['error'])
    if not (work/'initial_failure_status.json').exists():write(work/'initial_failure_status.json',state)
    else:write(work/'archive_failure_status.json',state)
    charged_rounds=max(2,state['private_rounds'])
    protocol=load(work/'protocol.json')
    assert load(work/'restore_projection_audit.json')['passed']
    source=Path(protocol['source']);assert digest(source)==protocol['source_sha256']
    natural=work/'natural';natural_result=load(natural/'run/result.json')
    assert natural_result['cycles']==16 and natural_result['updates']==64
    projection=work/'original_live_projection.pt'
    original_hashes=load(work/'source_hashes.json')
    assert all(digest(p)==h for p,h in original_hashes.items())
    files=[Path(__file__),Path('scripts/run_wood_policy_recovery_worker.py'),
           Path('scripts/audit_wood_policy_restore.py'),projection]
    recovery_hashes={str(p.resolve()):digest(p) for p in files}
    if (work/'recovery_source_hashes.json').exists():
        write(work/'recovery_hashes_before_archive_amendment.json',load(work/'recovery_source_hashes.json'))
    write(work/'recovery_source_hashes.json',recovery_hashes)
    failed=Path('F:/mengli-wood-policy-use-20261120/b04_a1')
    failed_result=load(failed/'run/result.json')
    assert failed_result['cycles']==2 and failed_result['updates']==8
    temporary=Path('C:/Users/85750/AppData/Local/Temp/mengli-wood-policy-use-20261120')
    f_root=failed.parent
    assert shutil.disk_usage(f_root).free>1.35*1024**3
    assert shutil.disk_usage(temporary).free>(3.51+.5+.125)*1024**3
    amendment=dict(reason='Triton parameter writes leave tensor versions unchanged; live packed P-query cache retains initial source projections',
        verification='restoring initial off2585 folded projections exactly matches all 14 decision conditions at observation 4',
        production_algorithm_changed=False,natural_rerun_rounds=0,read_only_audit_rounds=0,
        failed_private_rounds=2,projection=str(projection),projection_sha256=digest(projection),
        treatment='carry original live cache, preserving original cache invalidation and all nominal learning',
        original_budget=protocol['max_private_rounds'])
    write(work/'restore_amendment.json',amendment)
    state.update(status='running',stage='branches',pid=os.getpid(),private_rounds=charged_rounds,
                 resumed=datetime.now(timezone.utc).isoformat(),error=None,completed_boundaries=[])
    def update(**kw):state.update(kw);write(work/'pipeline_status.json',state)
    def unchanged():
        assert digest(source)==protocol['source_sha256']
        assert all(digest(p)==h for p,h in original_hashes.items())
        assert all(digest(p)==h for p,h in recovery_hashes.items())
    observations=load(natural/'observations.json')
    readings=load(natural/'independent_scores.json')['readings']
    rows=external(natural);reports=[]
    try:
        update()
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
                        reuse='actual uninterrupted natural trajectory; original nominal learning')
                    continue
                branch=f_root/f'b{boundary:02d}_a{option}_recovered'
                temp=temporary/'branches'/branch.name
                command=[sys.executable,'-m','scripts.run_wood_policy_recovery_worker',str(projection),
                    '--mode','branch','--option',str(option),'--checkpoint',str(checkpoint),
                    '--work',str(branch),'--temporary',str(temp),'--protocol',str(work/'protocol.json')]
                # Charge the full attempt before launch, including a failed
                # attempt, so a later failure cannot silently expand the budget.
                if not (branch/'run/result.json').exists():
                    assert state['private_rounds']+2<=18
                    temp.mkdir(parents=True,exist_ok=False)
                    update(private_rounds=state['private_rounds']+2)
                    with (work/(branch.name+'.log')).open('w',encoding='utf8') as log:
                        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                        update(child_pid=child.pid)
                        assert child.wait()==0,f'worker failed: {branch}'
                unchanged()
                result=load(branch/'run/result.json')
                assert result['cycles']==2 and result['updates']==8 and result['provider_calls']==0
                assert result['initial_source_sha256']==obs['source_sha256']
                receipt=load(branch/'pre_outcome_action.json')
                scores=load(branch/'independent_scores.json')['readings']
                checks={key:receipt[key]==action[key] for key in
                    ('source_sha256','endpoint_sha256','recurrent_sha256','unknown','questions',
                     'proposals','selected_proposal','local','future','total','gamma','habit','offset')}
                checks.update(sample=receipt['sampled_option']==action['sampled_option'],
                    probability=max(abs(p-q) for p,q in zip(receipt['nominal_probability'],action['production']))<1e-7,
                    learning=receipt['behavior']['categorical_score_gradient']==1,
                    material=load(branch/'independent_scores.json')['material']==protocol['validation'],
                    initial_score=abs(scores['0']['mean_nll']-baseline['mean_nll'])<1e-7 and
                        max(abs(p-q) for p,q in zip(scores['0']['segment_nll'],baseline['segment_nll']))<1e-7)
                write(branch/'matching_checks.json',checks);assert all(checks.values()),checks
                events=external(branch)
                assert all(r['causal_use']['finite_U']['verified'] for r in events)
                outcomes[str(option)]=dict(G2=baseline['mean_nll']-scores['2']['mean_nll'],
                    segment_G2=[a-b for a,b in zip(baseline['segment_nll'],scores['2']['segment_nll'])],
                    P2=events[0]['metal_g']+action['gamma']*events[1]['metal_g'],
                    reuse=False,branch=str(branch),checkpoint_sha256=result['checkpoint_sha256'])
                write(work/f'boundary{boundary:02d}_outcomes.json',outcomes)
                if not (branch/'retired_artifacts.json').exists():
                    with (work/(branch.name+'.archive.log')).open('w',encoding='utf8') as log:
                        archive=subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',
                            'scripts/maintenance/Archive-PolicyMeasurementBranch.ps1',
                            '-Root',branch.as_posix(),'-Temporary',temp.as_posix()],
                            stdout=log,stderr=subprocess.STDOUT,text=True)
                    if archive.returncode:
                        raise RuntimeError('Archive-PolicyMeasurementBranch.ps1 failed; see '+str(work/(branch.name+'.archive.log')))
            report=report_boundary(boundary,obs,outcomes,baseline,protocol['epsilon'])
            reports.append(report);write(work/f'boundary{boundary:02d}_report.json',report)
            update(completed_boundaries=[r['boundary'] for r in reports])
        import numpy as np
        cs=[r['C'] for r in reports];epsilon=protocol['epsilon']
        positive=sum(c>epsilon for c in cs);negative=sum(c < -epsilon for c in cs)
        decision=('return_to_longer_natural_growth' if all(r['robust_positive'] for r in reports) else
            'mixed_scope_keep_mechanisms' if positive and negative else
            'trace_specific_failed_choices' if negative>=2 and positive==0 else
            'near_zero_read_existing_probability_outcome_table' if max(abs(c) for c in cs)<=epsilon else
            'material_or_boundary_dependent_keep_mechanisms')
        final=dict(completed=True,protocol=protocol,restore_amendment=amendment,boundaries=reports,
            mean_C=float(np.mean(cs)),decision=decision,
            natural_learning=dict(initial=readings['0'],final=readings['16'],
                G16=readings['0']['mean_nll']-readings['16']['mean_nll'],checkpoint=natural_result['checkpoint']),
            budget=dict(natural_rounds=16,private_rounds=state['private_rounds'],
                failed_unmatched_rounds=2,valid_private_rounds=state['private_rounds']-2,max_private_rounds=18),
            decision_count=3,material_segments_per_readout=4,production_changed=False)
        output=Path('results/published/cgb_wood_policy_use_16rounds_20261120.json')
        write(output,final);update(status='completed',stage='completed',report=str(output),decision=decision,mean_C=final['mean_C'])
    except Exception as error:
        update(status='failed',error=str(error));raise


if __name__=='__main__':main()
