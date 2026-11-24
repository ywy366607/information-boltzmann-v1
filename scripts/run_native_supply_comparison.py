"""Bounded serial native-body comparison of original versus added SADL.

Copies a small source-only runtime first so a concurrent maintainer can keep
working. Complete bodies and temporary arrays stay on D/F, not the system
drive or removable disk. All original learning and selection remain active.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--candidate-work', type=Path, required=True)
    p.add_argument('--temporary', type=Path, required=True)
    p.add_argument('--rounds', type=int, default=8)
    p.add_argument('--recovery-from', type=Path)
    p.add_argument('--table-temporary', type=Path)
    a = p.parse_args()
    assert a.rounds > 0
    project = Path.cwd()
    a.work.mkdir(parents=True, exist_ok=False)
    assert not a.candidate_work.exists()
    a.temporary.mkdir(parents=True, exist_ok=False)
    if a.table_temporary:
        a.table_temporary.mkdir(parents=True, exist_ok=False)
    frozen = a.work/'runtime'
    hashes = {}
    for file in (project/'scripts').rglob('*'):
        if file.is_file() and file.suffix in ('.py', '.cu', '.cuh', '.cpp', '.h'):
            relative = file.relative_to(project)
            target = frozen/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, target)
            hashes[str(relative)] = digest(target)
    write(a.work/'runtime_hashes.json', hashes)
    source = Path('D:/cgb_reference_cap_2741_20261121/8/body/last.pt.gz')
    metadata = json.loads(source.with_name('last.metadata.json').read_text(encoding='utf8'))
    assert digest(source) == metadata['checkpoint_sha256']
    validation = project/'data/cgb_natural_text_20261115/validation.bin'
    protocol = dict(schema=1, source=str(source), source_sha256=metadata['checkpoint_sha256'],
        source_cycle=metadata['cycle'], natural_rounds=a.rounds,
        observation_rounds=[], prefix_paths={}, score_rounds=[a.rounds//2,a.rounds],
        record_actions=True, record_dispatch_timing=True,
        old_fact_material='D:/cgb_reference_cap_2741_20261121/8/material.pt',
        validation=dict(path=str(validation), sha256=digest(validation),
            starts=[40960,114688,196608,245760], bytes_per_segment=2048,
            total_scored_bytes=8192, used_in_training=False),
        arms=dict(original=0., supply=.01), max_world_rounds=2*a.rounds,
        max_optimizer_updates=8*a.rounds,
        candidate=dict(gamma=9., ridge=1e-3, epsilon=1e-8,
            port='raw current/valid history -> shared encoder',
            original_SIGReg='retained; supply endpoint and demand detached'),
        functions=['fixed real old-fact prediction','arrived-world outcome prediction',
                   'actual action/timing/experience support','language prediction','compute/storage'],
        runtime='frozen scripts only; original project cwd and data; serial GPU processes',
        storage=dict(initial_copy='deferred; immutable source retained; complete final save required',
            activation_host_mib=512, table_temporary=str(a.table_temporary)),
        stopping='fixed rounds; preserve complete checkpoints; no coefficient tuning during comparison')
    if a.recovery_from:
        previous = json.loads((a.recovery_from/'pipeline_status.json').read_text(encoding='utf8'))
        assert previous['status'] == 'failed' and previous['stage'] == 'original'
        protocol['resource_recovery'] = dict(attempt=str(a.recovery_from),
            error=previous['error'],
            failed_attempt_world_arrivals=2, failed_attempt_completed_updates=4,
            recovery='defer redundant initial body copy; 512 MiB bounded host cache; separate table disk',
            failed_artifacts='logs/metadata/SQL kept; two rebuildable copies reclaimed with hashes',
            runtime_change='lossless storage placement only; same objective, source and random state')
    write(a.work/'protocol.json', protocol)
    status = dict(status='running', stage='original', completed=[], child_pid=None,
        pid=os.getpid(), started=datetime.now(timezone.utc).isoformat(), protocol=str(a.work/'protocol.json'))
    write(a.work/'pipeline_status.json', status)
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONIOENCODING='utf-8')
    env['PYTHONPATH'] = str(frozen)+os.pathsep+str(project)
    try:
        for arm, coefficient in protocol['arms'].items():
            destination = a.work/arm if arm == 'original' else a.candidate_work
            temporary = (a.table_temporary or a.temporary)/arm
            destination.parent.mkdir(parents=True, exist_ok=True)
            assert shutil.disk_usage(destination.parent).free > 650*1024**2
            assert shutil.disk_usage(a.temporary).free > 400*1024**2
            code = ('import sys;sys.path.insert(0,'+repr(str(frozen))+');'
                    'from scripts.run_native_supply_life import main;main()')
            command = [sys.executable, '-c', code, '--supply-coefficient', str(coefficient),
                '--defer-initial-checkpoint', '--activation-host-mib', '512',
                '--activation-directory', str(a.temporary/arm),
                '--mode', 'natural', '--checkpoint', str(source), '--work', str(destination),
                '--temporary', str(temporary), '--protocol', str(a.work/'protocol.json')]
            status.update(stage=arm, destination=str(destination))
            with (a.work/(arm+'.log')).open('w',encoding='utf8') as log:
                child = subprocess.Popen(command, cwd=project, env=env,
                    stdout=log, stderr=subprocess.STDOUT)
                status['child_pid'] = child.pid
                write(a.work/'pipeline_status.json', status)
                assert child.wait() == 0, f'{arm} failed; inspect {arm}.log'
            assert digest(source) == protocol['source_sha256']
            assert all(digest(frozen/file) == sha for file,sha in hashes.items())
            result = json.loads((destination/'run/result.json').read_text(encoding='utf8'))
            assert result['status'] == 'completed' and result['cycles'] == a.rounds
            assert result['updates'] == 4*a.rounds and result['provider_calls'] == 0
            status['completed'].append(arm)
            status.update(child_pid=None)
            write(a.work/'pipeline_status.json', status)
        status.update(status='completed', stage='finished',
                      finished=datetime.now(timezone.utc).isoformat())
        write(a.work/'pipeline_status.json', status)
    except BaseException as error:
        status.update(status='failed', error=repr(error), child_pid=None)
        write(a.work/'pipeline_status.json', status)
        raise


if __name__ == '__main__':
    main()
