"""One matched two-action/two-transition experiment with a frozen main readout."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.run_root_native_correction_comparison import digest,write


def summarize(work,source_hash,report_path=None,capture_cross_score=False):
    arms = {}
    layout=json.loads((work/'run_layout.json').read_text(encoding='utf8'))
    for name in ('silence','generate'):
        root = Path(layout[name])
        result = json.loads((root/'run/result.json').read_text(encoding='utf8'))
        rows = [json.loads(s) for s in (root/'run/events.jsonl').read_text(encoding='utf8').splitlines()]
        rows = [r for r in rows if r['autonomous_event']['kind'] == 'external']
        scores = json.loads((root/'independent_scores.json').read_text(encoding='utf8'))
        action = json.loads((root/'pre_outcome_action.json').read_text(encoding='utf8'))
        assert result['updates'] == 8 and result['initial_source_sha256'] == source_hash
        assert all(r['causal_use']['finite_U']['verified'] for r in rows)
        assert result['provider_calls'] == 0
        gamma = action['gamma']; option = action['option']
        arms[name] = dict(action=action,rows=rows,score_material=scores['material'],
                         score_curve=[dict(event=s['event'],mean_nll=s['mean_nll'],segment_nll=s['segment_nll'])
                                      for s in scores['scores']],
                         P1=rows[0]['metal_g'],P2=rows[0]['metal_g']+gamma*rows[1]['metal_g'],
                         G1=scores['scores'][0]['mean_nll']-scores['scores'][1]['mean_nll'],
                         G2=scores['scores'][0]['mean_nll']-scores['scores'][2]['mean_nll'],
                         initial_wood_two_step_raw_U=action['total'][option]/10,
                         seconds=result['invocation']['seconds'],checkpoint=result['checkpoint'])
    a,b = arms['silence'],arms['generate']
    checks = dict(same_initial_scores=a['score_curve'][0]==b['score_curve'][0],
                  same_score_material=a['score_material']==b['score_material'],
                  same_pre_action_source=a['action']['source_sha256']==b['action']['source_sha256'],
                  same_pre_action_endpoints=a['action']['endpoint_sha256']==b['action']['endpoint_sha256'],
                  same_question=a['action']['unknown']==b['action']['unknown'],
                  same_proposals=a['action']['proposals']==b['action']['proposals'],
                  same_pre_outcome_values=a['action']['total']==b['action']['total'],
                  same_nominal_probability=a['action']['nominal_probability']==b['action']['nominal_probability'],
                  distinct_actual_words=a['action']['utterance_hex']!=b['action']['utterance_hex'])
    differences={k:b[k]-a[k] for k in ('P1','P2','G1','G2','initial_wood_two_step_raw_U')}
    out = dict(valid=all(checks.values()),checks=checks,source_sha256=source_hash,
               units=dict(P='raw finite-U; P2=U1+gamma*U2',G='held-out external nats/byte improvement'),
               comparison='paired ordering only; no unit conversion or numerical G-P residual',
               differences_generate_minus_silence=differences,arms=arms)
    if capture_cross_score:
        import torch
        from scripts.ib_local.action_value_cross_score import four_cells
        capsules = {name: torch.load(Path(layout[name])/'second_settlement.pt',
                                    map_location='cpu',weights_only=False)
                    for name in ('silence','generate')}
        out['cross_score'] = four_cells(capsules['silence'],capsules['generate'],
                                       differences['P1'],a['action']['gamma'])
        for name, capsule in capsules.items():
            assert abs(capsule['actual_U']-arms[name]['rows'][1]['metal_g']) < 1e-8
        ds=out['cross_score']['two_step_generate_minus_silence']
        sign=lambda v: (v>0)-(v<0)
        out['cross_score']['predeclared_prediction'] = (
            'Both common-query columns have the same ordering as independent G2; '
            'a reversed or mixed column falsifies the common-measure explanation at this boundary.')
        out['cross_score']['prediction_passed'] = (None if any(v is None for v in ds.values()) else
            all(sign(v)==sign(differences['G2']) for v in ds.values()))
    write(report_path or Path('results/published/cgb_action_value_pair_2585_20261120.json'),out)
    assert out['valid'],checks
    return differences


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--temporary',type=Path,required=True)
    p.add_argument('--silence-work',type=Path,required=True)
    p.add_argument('--generate-work',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,
        default=Path('scratch/cgb_retention_scope_prior_2569_20261120/off/body/last.pt.gz'))
    p.add_argument('--report',type=Path,
        default=Path('results/published/cgb_action_value_pair_2585_20261120.json'))
    p.add_argument('--capture-cross-score',action='store_true')
    a=p.parse_args()
    source=a.checkpoint.resolve()
    assert shutil.disk_usage(a.work.parent).free >= 256*1024**2,'metadata/report headroom'
    layout=dict(silence=str(a.silence_work.resolve()),generate=str(a.generate_work.resolve()))
    for folder in layout.values():
        target=Path(folder)
        assert not target.exists(), 'private branch must start in a new directory'
        assert shutil.disk_usage(target.parent).free >= 1.3*1024**3, 'one private body plus atomic save headroom'
    assert shutil.disk_usage(a.temporary.parent).free >= 5*1024**3,'measured 3.51GiB backward peak plus table/headroom'
    a.work.mkdir(exist_ok=False)
    write(a.work/'run_layout.json',layout)
    a.temporary.mkdir(exist_ok=True)
    source_hash=digest(source)
    paths=[*Path('scripts/ib_local').glob('*.py'),Path(__file__),
           Path('scripts/run_action_value_pair_worker.py'),Path('scripts/run_native_preview_restore_comparison.py')]
    hashes={str(p.resolve()):digest(p) for p in paths}
    write(a.work/'source_hashes.json',hashes)
    state=dict(status='running',pid=os.getpid(),started=datetime.now(timezone.utc).isoformat(),
               source=str(source),source_sha256=source_hash,horizon=2,world_calls='natural text only',
               protocol='two first actions, then original policy; P2 discounted two actual settlements; G2 fixed external validation')
    def update(**kw):
        state.update(kw);write(a.work/'pipeline_status.json',state)
    try:
        update(stage='starting')
        for name,option in (('silence',0),('generate',1)):
            with (a.work/(name+'.log')).open('w',encoding='utf8') as log:
                command=[sys.executable,'-m','scripts.run_action_value_pair_worker',
                    '--checkpoint',str(source),'--work',layout[name],
                    '--temporary',str(a.temporary),'--option',str(option)]
                if a.capture_cross_score:
                    command.append('--capture-cross-score')
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                update(stage=name,child_pid=child.pid)
                code=child.wait()
            assert code==0,f'{name} failed ({code}); inspect {name}.log'
            assert digest(source)==source_hash
            assert all(digest(p)==h for p,h in hashes.items()),'runtime changed'
        differences=summarize(a.work,source_hash,a.report,a.capture_cross_score)
        update(status='completed',stage='completed',differences=differences)
    except Exception as error:
        update(status='failed',error=str(error));raise


if __name__=='__main__':
    main()
