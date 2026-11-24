"""Wait for the current owner to finish, then observe the SAME body's next events."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from scripts.run_guarded_mother_continuation import digest, resources, stop_reason


def summary(rows):
    def group(items):
        if not items:
            return dict(events=0)
        mean = lambda key: statistics.mean(x[key] for x in items)
        result=dict(events=len(items), original_margin_mean=mean('original_margin'),
            relation_margin_mean=mean('relation_margin'),
            original_margin_median=statistics.median(x['original_margin'] for x in items),
            relation_margin_median=statistics.median(x['relation_margin'] for x in items),
            original_positive=sum(x['original_margin'] > 0 for x in items),
            relation_positive=sum(x['relation_margin'] > 0 for x in items),
            relation_log_margin_mean=mean('relation_log_margin'),
            comparisons=sum(len(x['different_alternatives']) for x in items),
            original_pair_wins=sum(x['original_wins'] for x in items),
            relation_pair_wins=sum(x['relation_wins'] for x in items),
            original_chosen_mse=statistics.mean(x['original_mse'][x['chosen']] for x in items),
            relation_chosen_mse=statistics.mean(x['relation_mse'][x['chosen']] for x in items))
        if all('legacy_relation_mse' in x for x in items):
            result.update(
                legacy_chosen_mse=statistics.mean(x['legacy_relation_mse'][x['chosen']] for x in items),
                relation_chosen_log_score=statistics.mean(x['relation_log_score'][x['chosen']] for x in items),
                legacy_chosen_log_score=statistics.mean(x['legacy_relation_log_score'][x['chosen']] for x in items),
                predicted_variance=statistics.mean(x['predicted_variance'][x['chosen']] for x in items),
                action_rank_given_condition=statistics.mean(x['shared_basis_geometry']['action_rank_given_condition'] for x in items))
        return result
    available = [r for r in rows if r.get('self_action_observer', {}).get('available')]
    options = [r['self_action_observer']['options'] for r in available]
    drafts = [r['self_action_observer']['generated_drafts'] for r in available
              if r['self_action_observer']['generated_drafts'] is not None]
    options = [x for x in options if x['different_alternatives']]
    drafts = [x for x in drafts if x['different_alternatives']]
    return dict(events=len(rows), cycles=[rows[0]['cycle'], rows[-1]['cycle']] if rows else [],
        actual_updates=sum(r.get('updates', 0) for r in rows),
        source_scope='real prospective mother replies; predictions frozen before each response',
        inference_scope='factual action-likelihood discrimination, not full counterfactual world identification',
        options=group(options), generated_drafts=group(drafts),
        reproduction_error_max=max((r['self_action_observer']['actual_score_reproduction_error']
                                    for r in available), default=0.),
        wood_policy_tv_mean=statistics.mean(r['wood_policy_tv'] for r in rows) if rows else None,
        draft_uniform_tv_mean=statistics.mean(sum(abs(v-.25) for v in
            r['joint_language_search']['distribution'])/2 for r in rows) if rows else None,
        fixed_validation=[dict(cycle=r['cycle'], **r['fixed_validation'])
                          for r in rows if 'fixed_validation' in r])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-status', required=True, type=Path)
    parser.add_argument('--expected-cycle', type=int, required=True)
    parser.add_argument('--checkpoint', type=Path,
                        help='explicit saved boundary, including recovery from an observer initialization')
    parser.add_argument('--cycles', type=int, default=32)
    parser.add_argument('--validate-every',type=int,
                        help='fixed sidecar interval, defaults to the full observation budget')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--observer-binding', type=Path)
    parser.add_argument('--minimum-start-commit-mib',type=int,default=3328,
                        help='explicit host admission reserve; critical event-boundary stopping is unchanged')
    args = parser.parse_args()
    if args.cycles < 1 or args.minimum_start_commit_mib < 1:
        raise ValueError('positive observation budget required')
    args.output.mkdir(parents=True, exist_ok=False)
    status = dict(status='waiting_prior_owner', pid=os.getpid(), planned_cycles=args.cycles,
        expected_source_cycle=args.expected_cycle, started_at=time.time(),
        minimum_start_commit_mib=args.minimum_start_commit_mib, minimum_system_disk_mib=1536,
        change='passive instrumentation; no change to architecture, reward or optimizer')

    def write():
        path = args.output/'status.json'
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(status, indent=2)+'\n', encoding='utf-8')
        temporary.replace(path)

    stop = args.output/'STOP'
    while True:
        prior = json.loads(args.wait_status.read_text(encoding='utf-8'))
        status['prior_status'] = prior['status']
        status['resources'] = resources(args.output)
        if stop.exists():
            status['status'] = 'stopped_before_start'; write(); return
        if prior['status'] not in ('running', 'waiting_resources'):
            if prior['status'] != 'complete' or prior.get('final_cycle') != args.expected_cycle:
                status['status'] = 'prior_did_not_complete_expected_boundary'; write(); return
            status['status'] = 'waiting_resources'
            r = status['resources']
            if (r['available_commit_mib'] >= args.minimum_start_commit_mib and r['system_disk_free_mib'] >= 1536
                    and r['output_disk_free_mib'] >= 512):
                break
        write(); time.sleep(5)
    checkpoint = args.checkpoint or args.wait_status.parent/'body/last.pt.gz'
    status.update(source_checkpoint=str(checkpoint.resolve()), source_sha256=digest(checkpoint))
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1',
               CUDA_MODULE_LOADING='LAZY', TRITON_CACHE_DIR=str(root/'scratch/triton_cache'))
    command = [sys.executable, '-u', '-m', 'scripts.ib_local.run_self_action_observation',
        '--checkpoint', str(checkpoint.resolve()), '--output', str((args.output/'body').resolve()),
        '--cycles', str(args.cycles), '--expected-cycle', str(args.expected_cycle),
        '--validate-every', str(args.validate_every or args.cycles), '--validation-bytes', '8192',
        '--mother-harness-url', 'http://127.0.0.1:8902', '--stop-file', str(stop.resolve())]
    if args.observer_binding is not None:
        command.extend(['--observer-binding', str(args.observer_binding.resolve())])
    with (args.output/'body.log').open('w', encoding='utf-8') as log:
        child = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        status.update(status='running', child_pid=child.pid); write()
        while child.poll() is None:
            status['resources'] = resources(args.output)
            reason = stop_reason(status['resources'])
            if reason and not stop.exists():
                stop.write_text(reason+': finish and save current event.\n', encoding='utf-8')
                status.update(guard_stop_reason=reason)
            write(); time.sleep(2)
    status.update(status='failed', exit_code=child.returncode, ended_at=time.time(),
                  source_unchanged=digest(checkpoint) == status['source_sha256'])
    completed = args.output/'body/completed.json'
    if completed.exists():
        done = json.loads(completed.read_text(encoding='utf-8'))
        status.update(status=done['status'], final_cycle=done['final_cycle'])
    metrics = args.output/'body/metrics.jsonl'
    if metrics.exists():
        data = metrics.read_bytes()
        rows = [json.loads(s) for s in data[:data.rfind(b'\n')+1].splitlines() if s]
        (args.output/'report.json').write_text(json.dumps(summary(rows), ensure_ascii=False, indent=2)+'\n',
                                              encoding='utf-8')
    write()


if __name__ == '__main__':
    main()
