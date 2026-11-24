"""Continue only the pending supply arm of a completed frozen baseline."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.run_native_supply_comparison import digest, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--activations', type=Path, required=True)
    a = p.parse_args()
    status = json.loads((a.work/'pipeline_status.json').read_text(encoding='utf8'))
    assert status['status'] == 'failed' and status['completed'] == ['original']
    protocol = json.loads((a.work/'protocol.json').read_text(encoding='utf8'))
    hashes = json.loads((a.work/'runtime_hashes.json').read_text(encoding='utf8'))
    frozen = a.work/'runtime'
    assert all(digest(frozen/file) == sha for file,sha in hashes.items())
    source = Path(protocol['source'])
    assert digest(source) == protocol['source_sha256']
    a.candidate.parent.mkdir(parents=True, exist_ok=True)
    assert not a.candidate.exists()
    table = Path(protocol['storage']['table_temporary'])/'supply'
    code = ('import sys;sys.path.insert(0,'+repr(str(frozen))+');'
            'from scripts.run_native_supply_life import main;main()')
    command = [sys.executable, '-c', code, '--supply-coefficient', str(protocol['arms']['supply']),
        '--defer-initial-checkpoint', '--activation-host-mib', str(protocol['storage']['activation_host_mib']),
        '--activation-directory', str(a.activations), '--mode', 'natural',
        '--checkpoint', str(source), '--work', str(a.candidate), '--temporary', str(table),
        '--protocol', str(a.work/'protocol.json')]
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONIOENCODING='utf-8')
    env['PYTHONPATH'] = str(frozen)+os.pathsep+str(Path.cwd())
    status.update(status='running', stage='supply', error=None, pid=os.getpid(),
        destination=str(a.candidate),
        recovery='baseline complete; create missing candidate parent; reuse identical frozen runtime')
    try:
        with (a.work/'supply.log').open('w', encoding='utf8') as log:
            child = subprocess.Popen(command, cwd=Path.cwd(), env=env, stdout=log, stderr=subprocess.STDOUT)
            status['child_pid'] = child.pid
            write(a.work/'pipeline_status.json', status)
            assert child.wait() == 0, 'inspect supply.log'
        result = json.loads((a.candidate/'run/result.json').read_text(encoding='utf8'))
        assert result['status'] == 'completed' and result['cycles'] == protocol['natural_rounds']
        assert result['updates'] == 4*protocol['natural_rounds'] and result['provider_calls'] == 0
        assert digest(source) == protocol['source_sha256']
        assert all(digest(frozen/file) == sha for file,sha in hashes.items())
        status.update(status='completed', stage='finished', child_pid=None,
            completed=['original', 'supply'], finished=datetime.now(timezone.utc).isoformat())
        write(a.work/'pipeline_status.json', status)
    except BaseException as error:
        status.update(status='failed', child_pid=None, error=repr(error))
        write(a.work/'pipeline_status.json', status)
        raise


if __name__ == '__main__':
    main()
