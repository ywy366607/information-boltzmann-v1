"""Complete the missing preidentified Wood-first action, reusing its control."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.run_root_native_correction_comparison import digest, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--branch', type=Path, required=True)
    parser.add_argument('--temporary', type=Path, required=True)
    args = parser.parse_args()
    source = Path('E:/mengli-action-pair-2585-20261120-silence/run/body/last.pt.gz')
    baseline_path = Path('results/published/cgb_action_cross_score_2587_20261120.json')
    earlier_path = Path('results/published/cgb_action_value_pair_2585_20261120.json')
    baseline = json.loads(baseline_path.read_text(encoding='utf8'))
    control = baseline['arms']['silence']
    initial = control['action']
    available = [i for i, p in enumerate(initial['nominal_probability']) if p > 0]
    wood = max(available, key=lambda i: initial['total'][i])
    habitual = max(available, key=lambda i: initial['habit'][i])
    assert (wood, habitual) == (2, 0), 'predeclared missing reuse-vs-silence comparison'
    source_hash = digest(source)
    assert source_hash == baseline['source_sha256']
    assert not args.work.exists() and not args.branch.exists()
    archive = Path(json.loads((source.parent/'config.json').read_text(encoding='utf8'))
                   ['causal_archive']['snapshot'])
    # Measured rolling save retains old body plus working/snapshot archives,
    # then writes the next body and next snapshot before atomic publication.
    checkpoint_headroom = 2*source.stat().st_size+3*archive.stat().st_size+64*1024**2
    assert shutil.disk_usage(args.branch.parent).free >= checkpoint_headroom
    assert shutil.disk_usage(args.temporary.parent).free >= (3.51+.25+.125)*1024**3
    args.work.mkdir()
    args.temporary.mkdir(exist_ok=True)
    code = [*Path('scripts/ib_local').glob('*.py'), Path(__file__),
            Path('scripts/run_action_value_pair_worker.py'),
            Path('scripts/run_native_preview_restore_comparison.py')]
    hashes = {str(p.resolve()): digest(p) for p in code}
    write(args.work/'source_hashes.json', hashes)
    protocol = dict(source=str(source), source_sha256=source_hash,
        control_report=str(baseline_path), control_report_sha256=digest(baseline_path),
        selected_before_outcome=dict(wood_option=wood,habit_option=habitual,
                                     all_wood_values=initial['total'],habit_logits=initial['habit']),
        question='At this complete boundary, first execute Wood argmax rather than habit argmax; then original policy and complete learning for H=2.',
        contrast='one forced first-action contrast, not a permanent Wood removal or a policy expectation',
        budget=dict(new_branches=1,external_transitions=2,optimizer_updates=8),
        prediction='Wood-first G2 exceeds habit-first G2 on the same four validation segments; either sign is retained.',
        independent_validation='same read-only FineWeb 8192 bytes; no reward feedback',
        checkpoint_peak_budget_bytes=checkpoint_headroom)
    write(args.work/'preregistration.json',protocol)
    state = dict(status='running',pid=os.getpid(),started=datetime.now(timezone.utc).isoformat(),
                 source=str(source),source_sha256=source_hash,stage='reuse',new_branches=1)
    def update(**kw):
        state.update(kw)
        write(args.work/'pipeline_status.json',state)
    try:
        with (args.work/'reuse.log').open('w',encoding='utf8') as log:
            child=subprocess.Popen([sys.executable,'-m','scripts.run_action_value_pair_worker',
                '--checkpoint',str(source),'--work',str(args.branch),
                '--temporary',str(args.temporary),'--option',str(wood)],stdout=log,stderr=subprocess.STDOUT)
            update(child_pid=child.pid)
            assert child.wait()==0, 'reuse worker failed; preserve its log'
        result=json.loads((args.branch/'run/result.json').read_text(encoding='utf8'))
        action=json.loads((args.branch/'pre_outcome_action.json').read_text(encoding='utf8'))
        scores=json.loads((args.branch/'independent_scores.json').read_text(encoding='utf8'))
        rows=[json.loads(line) for line in (args.branch/'run/events.jsonl').read_text(encoding='utf8').splitlines()]
        rows=[r for r in rows if r['autonomous_event']['kind']=='external']
        curve=[{k:s[k] for k in ('event','mean_nll','segment_nll')} for s in scores['scores']]
        checks={key:action[key]==initial[key] for key in (
            'source_sha256','endpoint_sha256','recurrent_sha256','unknown','questions',
            'proposals','selected_proposal','local','future','total','gamma','critic','habit','offset','nominal_probability')}
        checks.update(same_initial_score=curve[0]==control['score_curve'][0],
            same_validation=scores['material']==control['score_material'],
            wood_reuse=action['option']==action['wood_argmax']==wood,
            habitual_silence=action['habit_argmax']==habitual,
            actual_reuse=rows[0]['choice']=='reuse' and action['reused_slot'] is not None,
            full_learning=result['updates']==8 and len(rows)==2,
            verified_payments=all(r['causal_use']['finite_U']['verified'] for r in rows),
            paid_calls_zero=result['provider_calls']==0,
            restored_common_source=result['initial_source_sha256']==source_hash,
            source_unchanged=digest(source)==source_hash,
            runtime_unchanged=all(digest(p)==h for p,h in hashes.items()),
            control_unchanged=digest(baseline_path)==protocol['control_report_sha256'])
        assert all(checks.values()), checks
        g1=curve[0]['mean_nll']-curve[1]['mean_nll']
        g2=curve[0]['mean_nll']-curve[2]['mean_nll']
        segment_delta=[a-b for a,b in zip(control['score_curve'][2]['segment_nll'],curve[2]['segment_nll'])]
        previous=json.loads(earlier_path.read_text(encoding='utf8'))
        pa=previous['arms']['silence']['action']
        assert max(range(3),key=lambda i:pa['total'][i])==1
        assert max(range(3),key=lambda i:pa['habit'][i])==0
        out=dict(valid=True,checks=checks,protocol=protocol,
            arms=dict(habit_silence={k:control[k] for k in ('G1','G2','P1','P2','score_curve','checkpoint')},
                wood_reuse=dict(G1=g1,G2=g2,P1=rows[0]['metal_g'],
                    P2=rows[0]['metal_g']+action['gamma']*rows[1]['metal_g'],
                    score_curve=curve,checkpoint=result['checkpoint'],action=action,
                    rows=[{k:r[k] for k in ('cycle','choice','metal_g','external_bytes','self_bytes','world')} for r in rows],
                    seconds=result['invocation']['seconds'])),
            wood_minus_habit=dict(G1=g1-control['G1'],G2=g2-control['G2'],segment_G2=segment_delta),
            earlier_2585=dict(wood='generate',habit='silence',G2_difference=previous['differences_generate_minus_silence']['G2']),
            scope='two specific matched first-action contrasts at predeclared complete boundaries; all later learning and choices endogenous')
        report=Path('results/published/cgb_wood_choice_use_2587_20261120.json')
        write(report,out)
        update(status='completed',stage='completed',report=str(report),
               wood_minus_habit_G2=out['wood_minus_habit']['G2'])
    except Exception as error:
        update(status='failed',error=str(error))
        raise


if __name__=='__main__':
    main()
