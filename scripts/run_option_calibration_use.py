"""One fixed complete boundary: root calibration probability displacement x G2."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--temporary', type=Path, required=True)
    p.add_argument('--recover-replay-forwarding', action='store_true')
    args = p.parse_args()
    args.work.mkdir(parents=True, exist_ok=args.recover_replay_forwarding)
    args.temporary.mkdir(parents=True, exist_ok=args.recover_replay_forwarding)
    source = Path('scratch/cgb_wood_policy_use_16rounds_20261120/natural/run/body/last.pt.gz').resolve()
    expected_sha = 'd673ffaa3168b153fcd853ad6af76f9567fa73336dc44ed129ebdd780e31a729'
    assert digest(source) == expected_sha
    validation = Path('data/cgb_natural_text_20261115/validation.bin').resolve()
    validation_sha = 'b09c3879960cab5b5bbe3d3282059d472d7ee140925c58d7550b7fedebfbee93'
    assert digest(validation) == validation_sha
    starts = [16384, 81920, 163840, 262144]
    assert all(s+2049 <= validation.stat().st_size for s in starts)
    previous = [0, 95629, 191258, 286887, 32768, 65536, 131072, 229376]
    assert all(s+2049 <= old or old+2049 <= s for s in starts for old in previous)
    protocol = dict(schema=1, source=str(source), source_sha256=expected_sha, source_cycle=2601,
        boundary='first completed real decision after immutable 2601 source', horizon=2,
        observation_rounds=[], prefix_paths={}, max_private_rounds=6, optimizer_updates_max=24,
        probability='original actor and root-only persistent residual lift, same actual menu',
        lift='persistent value + calibrated action-plan expectation - original action-plan expectation',
        direct_q_patch='zero final-choice effect in persistent route; separately recorded',
        learning='original nominal actor and complete original learning in every conditional branch',
        continuation='original policy after first conditional action',
        validation=dict(path=str(validation), sha256=validation_sha, starts=starts,
            bytes_per_segment=2048, total_scored_bytes=8192, used_in_training=False),
        target='delta C = sum((pi_lift-pi_original)*G2)', epsilon=1e-5,
        stop='one prespecified boundary; exactly two rounds per legal first action',
        exits=dict(positive='retain connected candidate for next prespecified independent choice',
            negative='retain original mechanism; trace paired plan deltas and root value contraction',
            near_zero='use same probability and outcome-span decomposition to locate small factor'))
    if args.recover_replay_forwarding:
        old_state = load(args.work/'pipeline_status.json')
        assert old_state['status']=='failed' and old_state['charged_rounds']==2
        old_protocol = load(args.work/'protocol.json')
        assert old_protocol == protocol
        old_log = (args.work/'action_0.log').read_text(encoding='utf8')
        assert "unexpected keyword argument 'reuse_audit'" in old_log
        events = [json.loads(s) for s in (args.work/'action_0/run/events.jsonl').read_text().splitlines()]
        assert events and all(r['updates']==0 for r in events)
        write(args.work/'protocol.before_recovery.json', old_protocol)
        write(args.work/'failure.before_recovery.json', old_state)
        write(args.work/'source_hashes.before_recovery.json', load(args.work/'source_hashes.json'))
        amendment = dict(reason='first_action wrapper forwards the new reuse_audit keyword',
            failed_attempt_charged_rounds=2, failed_optimizer_updates=0,
            failed_arrived_world_bytes=sum(r.get('rhythm_event',{}).get('received_bytes',0) for r in events),
            original_completed_round_budget=6, completed_round_budget=6,
            attempted_round_budget=8, source_and_menu='same fixed 2601 source and already recorded first menu',
            changed_code=['scripts/ib_local/action_value_pair_audit.py', 'scripts/run_option_calibration_use.py'])
        protocol['max_private_rounds']=8
        protocol['recovery']=amendment
        write(args.work/'recovery_amendment.json', amendment)
    write(args.work/'protocol.json', protocol)
    files = [*Path('scripts/ib_local').glob('*.py'), Path(__file__),
        Path('scripts/root_option_calibration_shadow.py'), Path('scripts/option_basis_calibration.py'),
        Path('scripts/run_wood_policy_use_worker.py'), Path('scripts/run_native_preview_restore_comparison.py')]
    hashes = {str(f.resolve()): digest(f) for f in files}
    write(args.work/'source_hashes.json', hashes)
    assert shutil.disk_usage(args.work).free > 2.3*1024**3
    assert shutil.disk_usage(args.temporary).free > 1.5*1024**3
    state = dict(status='running', pid=__import__('os').getpid(),
        started=datetime.now(timezone.utc).isoformat(), source_sha256=expected_sha,
        completed_options=[], charged_rounds=2 if args.recover_replay_forwarding else 0,
        stage='first_branch', child_pid=None)
    def update(**changes):
        state.update(changes); write(args.work/'pipeline_status.json', state)
    outcomes, reference, shadow, legal = {}, None, None, [0]
    if args.recover_replay_forwarding:
        reference=load(args.work/'action_0/pre_outcome_action.json')
        shadow=load(args.work/'action_0/root_calibration_shadow.json')
        legal=shadow['legal']
        baseline=load(args.work/'action_0/independent_scores.json')['readings']['0']
        write(args.work/'pre_outcome_menu.json', dict(action=reference, shadow=shadow))
    try:
        index = 0
        while index < len(legal):
            option = legal[index]
            assert state['charged_rounds']+2 <= protocol['max_private_rounds']
            update(stage='conditional_learning', option=option, charged_rounds=state['charged_rounds']+2)
            name=f'action_{option}_recovered' if args.recover_replay_forwarding and option==0 else f'action_{option}'
            work = args.work/name
            command = [sys.executable, '-m', 'scripts.run_wood_policy_use_worker', '--mode', 'branch',
                '--option', str(option), '--checkpoint', str(source), '--work', str(work),
                '--temporary', str(args.temporary/name),
                '--protocol', str(args.work/'protocol.json'), '--root-calibration-shadow']
            with (args.work/f'{name}.log').open('w', encoding='utf8') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                update(child_pid=child.pid)
                code = child.wait()
            assert code == 0, f'action {option} worker failed; see log'
            result = load(work/'run/result.json')
            receipt = load(work/'pre_outcome_action.json')
            current_shadow = load(work/'root_calibration_shadow.json')
            scores = load(work/'independent_scores.json')['readings']
            assert result['cycles'] == 2 and result['updates'] == 8 and result['provider_calls'] == 0
            assert result['initial_source_sha256'] == expected_sha
            if reference is None:
                reference = receipt; shadow = current_shadow
                legal = current_shadow['legal']
                assert legal[0] == 0
                baseline = scores['0']
                write(args.work/'pre_outcome_menu.json', dict(action=receipt, shadow=shadow))
            checks = {k: receipt[k] == reference[k] for k in
                ('source_sha256', 'endpoint_sha256', 'recurrent_sha256', 'unknown', 'questions',
                 'proposals', 'selected_proposal', 'local', 'future', 'total', 'gamma', 'habit', 'offset',
                 'nominal_probability', 'sampled_option')}
            checks['shadow_same'] = current_shadow == shadow
            checks['initial_score'] = scores['0'] == baseline
            checks['original_learning'] = receipt['behavior']['categorical_score_gradient'] == 1
            write(work/'matching_checks.json', checks)
            assert all(checks.values()), checks
            outcomes[str(option)] = dict(G2=baseline['mean_nll']-scores['2']['mean_nll'],
                segment_G2=[a-b for a, b in zip(baseline['segment_nll'], scores['2']['segment_nll'])],
                world_rounds=2, optimizer_updates=8, checkpoint_sha256=result['checkpoint_sha256'])
            write(args.work/'outcomes.json', outcomes)
            assert digest(source) == expected_sha
            assert all(digest(f) == h for f, h in hashes.items()), 'execution code changed'
            update(completed_options=[*state['completed_options'], option], child_pid=None)
            index += 1
        import numpy as np
        from scripts.ib_local.wood_policy_measurement import contribution
        old, new, habit = [[shadow[name][i] for i in legal] for name in
                          ('original_probability', 'alternative_probability', 'habit_probability')]
        gains = [outcomes[str(i)]['G2'] for i in legal]
        delta = contribution(new, old, gains)
        old_c, new_c = contribution(old, habit, gains), contribution(new, habit, gains)
        np.testing.assert_allclose(delta['C'], new_c['C']-old_c['C'], atol=1e-14)
        segment_g = np.array([outcomes[str(i)]['segment_G2'] for i in legal])
        segment_c = (np.array(new)-np.array(old))@segment_g
        loo = [float(np.mean(np.delete(segment_c, i))) for i in range(4)]
        report = dict(schema=1, protocol=protocol, legal=legal, outcomes=outcomes,
            original_C=old_c, connected_C=new_c, delta_C=delta,
            segment_delta_C=segment_c.tolist(), leave_one_material_segment_out_delta_C=loo,
            direct_q_patch_delta_C=0, root_shadow=shadow,
            positive=delta['C'] > protocol['epsilon'] and min(loo)>0,
            negative=delta['C'] < -protocol['epsilon'] and max(loo)<0,
            world_rounds=2*len(legal), charged_rounds=state['charged_rounds'],
            optimizer_updates=8*len(legal), provider_calls=0,
            source_unchanged=digest(source)==expected_sha,
            interpretation='one actual first-choice decision; four material segments are one score')
        write(args.work/'report.json', report)
        write(Path('results/published/cgb_option_calibration_use_20261122.json'), report)
        update(status='completed', stage='report', delta_C=delta['C'], original_C=old_c['C'],
               connected_C=new_c['C'], finished=datetime.now(timezone.utc).isoformat())
        print(json.dumps(state), flush=True)
    except BaseException as exc:
        update(status='failed', error=repr(exc), child_pid=None)
        raise


if __name__ == '__main__':
    main()
