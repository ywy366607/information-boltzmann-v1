"""One resumable command for whole-graph requirements and local anatomy.

Only independent IO acquisitions run concurrently. Dependent binding and
verification are serial. This never migrates a checkpoint or starts training.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from compile_native_connectome_program import checksum
from link_native_sensory_geometry import atomic_json


SCRIPTS = Path(__file__).resolve().parent


def run(name, arguments, log_dir):
    path = log_dir / (name + '.log')
    with path.open('a', encoding='utf-8') as log:
        result = subprocess.run([sys.executable, '-u', str(SCRIPTS / (name + '.py')),
                                 *map(str, arguments)], stdout=log, stderr=subprocess.STDOUT,
                                cwd=SCRIPTS.parents[2],
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        raise RuntimeError(f'{name} failed ({result.returncode}); inspect {path}')


def main(args):
    for name in ('root', 'program', 'pairs', 'incoming', 'outgoing', 'morphology', 'geometry'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    args.root.mkdir(parents=True, exist_ok=True)
    logs = args.root / 'logs'
    logs.mkdir(exist_ok=True)
    program = args.program or args.root / 'program'
    pairs = args.pairs or args.root / 'return_pairs' / 'sensory_return_edges.feather'
    incoming = args.incoming or args.root / 'incoming'
    outgoing = args.outgoing or args.root / 'outgoing'
    morphology = args.morphology or args.root / 'morphology'
    geometry = args.geometry or args.root / 'geometry'
    state_path = args.root / 'batch_status.json'
    status = {'started_at': datetime.now(timezone.utc).isoformat(),
              'runtime_binding_complete': False, 'training_started': False,
              'checkpoint_changed': False, 'stages': {}}

    def stage(name, action):
        status['stages'][name] = 'running'
        atomic_json(state_path, status)
        try:
            action()
        except Exception:
            status['stages'][name] = 'failed'
            atomic_json(state_path, status)
            raise
        status['stages'][name] = 'passed'
        atomic_json(state_path, status)

    def program_stage():
        if not (program / 'program.json').exists():
            run('compile_native_connectome_program', ['--output', program], logs)
        record = json.loads((program / 'program.json').read_text(encoding='utf-8'))
        if (record['master_sha256'] != checksum(Path(record['master_graph'])) or
                record['current_sha256'] != checksum(Path(record['current_graph'])) or
                record['program_sha256'] != checksum(program / 'program.npz')):
            raise ValueError('Previously compiled anatomical program changed')
    stage('whole_graph_program', program_stage)

    def pair_stage():
        if not pairs.exists():
            run('catalog_sensory_return_edges', ['--output', pairs.parent], logs)
    stage('omitted_pair_catalog', pair_stage)

    def acquisitions():
        commands = [
            ('fetch_native_sensory_returns', ['--pairs', pairs, '--output', incoming, '--direction', 'incoming']),
            ('fetch_native_sensory_returns', ['--pairs', pairs, '--output', outgoing, '--direction', 'outgoing']),
            ('fetch_native_morphology_batch', ['--table', pairs, '--output', morphology])]
        # Separate log names for the two contact directions.
        def acquire(index, command):
            script, arguments = command
            folder = logs / str(index)
            folder.mkdir(exist_ok=True)
            run(script, arguments, folder)
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(acquire, i, command) for i, command in enumerate(commands)]
            errors = []
            for future in futures:
                try:
                    future.result()
                except Exception as error:
                    errors.append(str(error))
            if errors:
                raise RuntimeError('; '.join(errors))
    stage('contact_and_morphology_acquisition', acquisitions)
    stage('local_anatomical_link', lambda: run('link_native_sensory_geometry',
          ['--pairs', pairs, '--incoming', incoming, '--outgoing', outgoing,
           '--morphology', morphology, '--output', geometry], logs))
    stage('cross_block_release_binding', lambda: run('finalize_native_sensory_program',
          ['--geometry', geometry, '--output', args.root / 'binding_report.json'], logs))
    stage('whole_batch_geometry_verification', lambda: run('verify_native_contact_geometry',
          ['--geometry', geometry, '--output', args.root / 'transport_verification.json'], logs))
    status['finished_at'] = datetime.now(timezone.utc).isoformat()
    status['anatomical_batch_ready'] = True
    status['scope'] = ('Whole-graph requirements and native anatomical operands verified. '
                       'Receptor/effector execution binding, full-state migration and functional '
                       'causal acceptance remain separate; this status never authorizes training.')
    atomic_json(state_path, status)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--root', type=Path, required=True)
    for name in ('program', 'pairs', 'incoming', 'outgoing', 'morphology', 'geometry'):
        p.add_argument('--' + name, type=Path)
    main(p.parse_args())
