"""Three complete learner branches for the measured internal transport mode."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.run_native_relation_mode_use import ROOT, prepare_private, sha, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--private-storage-root', type=Path, required=True)
    args = p.parse_args()
    args.previous = args.previous.resolve(); args.work = args.work.resolve()
    args.work.mkdir(parents=True, exist_ok=True)
    old = json.loads((args.previous/'plan.json').read_text(encoding='utf8'))
    args.task = Path(old['checkpoint']['path']).parent.parent
    lift_path = ROOT/'results/published/cgb_native_transport_curl_lift_2401_20261117.json'
    lift = json.loads(lift_path.read_text(encoding='utf8'))
    assert lift['proof']['exit_code'] == 0
    base = next(a for a in old['arms'] if a['arm'] == 'baseline')
    modes = {'baseline': 'baseline', 'bridge_cut': 'target', 'matched_cut': 'matched'}
    plan = copy.deepcopy(old)
    plan.update(experiment='single native motion-curl input intervention; norm and posterior-mean matched',
                transport_lift=lift, private_storage_root=str(args.private_storage_root.resolve()),
                release_observer_workspace=True, release_policy_workspace=True,
                disk_policy_activations=True, arms=[])
    for name, family in modes.items():
        row = copy.deepcopy(base)
        row.update(arm=name, inner_loop=dict(hold_preview=False, hold_feedback=False), transport_curl=family)
        plan['arms'].append(row)
    plan['runtime_sources'] = {name: sha(ROOT/name) for name in (
        'scripts/native_transport_curl_intervention.py', 'scripts/run_native_relation_mode_use.py',
        'scripts/native_inner_loop_intervention.py', 'scripts/native_observation_memory.py',
        'scripts/native_policy_activation_storage.py')}
    plan['matching'] = 'equal residual Frobenius norm AND equal full 128-coordinate posterior mean shift; kernel control preserves curl'
    if (args.work/'plan.json').exists():
        assert json.loads((args.work/'plan.json').read_text(encoding='utf8')) == plan
    else:
        write(args.work/'plan.json', plan)
    if not (args.work/'anchor.pt.gz').exists():
        os.link(args.previous/'anchor.pt.gz', args.work/'anchor.pt.gz')
    baseline = json.loads((args.previous/'baseline/result.json').read_text(encoding='utf8'))
    results = {}
    env = os.environ.copy()
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        env[key] = '1'
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'backend:native,garbage_collection_threshold:0.4'
    for name, family in modes.items():
        directory, _ = prepare_private(args, name)
        if not (directory/'result.json').exists():
            with (directory/'stdout.log').open('w', encoding='utf8') as out, (directory/'stderr.log').open('w', encoding='utf8') as err:
                child = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.run_native_relation_mode_use',
                    '--task', str(args.task), '--work', str(args.work), '--worker', '--arm', name, '--cycles', '8'],
                    cwd=ROOT, env=env, stdout=out, stderr=err, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                write(args.work/'status.json', dict(status='running', arm=name, family=family, worker_pid=child.pid, completed=list(results)))
                code = child.wait()
            if code:
                write(args.work/'status.json', dict(status='failed', arm=name, exit_code=code))
                raise RuntimeError(f'{name}: inspect {directory / "stderr.log"}')
        result = json.loads((directory/'result.json').read_text(encoding='utf8'))
        assert result['before'] == baseline['before']
        assert result['actual_optimizer_calls'] == 32 and result['provider_calls'] == 0
        assert len(result['transport_curl_interventions']) == 1
        if name == 'baseline':
            assert result['after'] == baseline['after']
        results[name] = result
        print(json.dumps(dict(arm=name, seconds=result['seconds'], intervention=result['transport_curl_interventions'][0]['relative_motion_curl_remaining'])), flush=True)
    for key in ('checkpoint', 'archive'):
        assert sha(plan[key]['path']) == plan[key]['sha256']
    write(args.work/'summary.json', dict(status='completed', main_source_hashes_preserved=True,
          baseline_carried_state_equal=True, arms={n:dict(seconds=r['seconds'], updates=r['actual_optimizer_calls'],
          receipt=r['transport_curl_interventions'], after=r['after']) for n, r in results.items()}))
    write(args.work/'status.json', dict(status='completed'))


if __name__ == '__main__':
    main()
