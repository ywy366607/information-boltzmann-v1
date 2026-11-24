"""Resume the saved comparison after the legacy-payload audit correction."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from scripts.run_retention_scope_prior_comparison import summarize
from scripts.run_root_native_correction_comparison import digest, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--activation-storage', type=Path, required=True)
    a = p.parse_args()
    work, temporary = a.work.resolve(), a.activation_storage.resolve()
    status = work/'pipeline_status.json'
    state = json.loads(status.read_text(encoding='utf8'))
    assert state['status'] == 'failed', 'resume only a stopped, failed comparison'
    source = Path(state['source'])
    hashes = json.loads((work/'source_hashes.json').read_text(encoding='utf8'))
    audit = Path('scripts/audit_native_relation_use_restore.py').resolve()
    amendment = work/'restore_audit_amendment.json'
    change = dict(path=str(audit), original_sha256=hashes[str(audit)],
                  corrected_sha256=digest(audit), recovery_sha256=digest(__file__),
                  reason='legacy retention payloads omit three empty prior defaults; preserve exact comparison of every saved value',
                  regression='11 focused audit checks', changed_training_files=[])
    if amendment.exists():
        assert json.loads(amendment.read_text(encoding='utf8')) == change
    else:
        write(amendment, change)

    def verify():
        assert digest(source) == state['source_sha256'], 'source changed'
        assert all(digest(f) == (change['corrected_sha256'] if Path(f) == audit else h)
                   for f, h in hashes.items()), 'unapproved runtime change'
        assert digest(__file__) == change['recovery_sha256']

    def update(**fields):
        state.update(fields)
        write(status, state)

    def run(stage, module, arguments):
        verify()
        with (work/(stage+'.resume.log')).open('w', encoding='utf8') as log:
            child = subprocess.Popen([sys.executable, '-m', module, *arguments],
                                     stdout=log, stderr=subprocess.STDOUT)
            update(stage=stage, child_pid=child.pid)
            code = child.wait()
        assert code == 0, f'{stage} failed ({code}); inspect {stage}.resume.log'
        verify()

    verify()
    update(status='running', pid=os.getpid(), previous_error=state.pop('error', None),
           resumed=datetime.now(timezone.utc).isoformat())
    try:
        for name in ('off', 'relation'):
            result = work/name/'result.json'
            if result.exists():
                saved = json.loads(result.read_text(encoding='utf8'))
                assert (saved['status'], saved['cycle'], saved['cycles'], saved['updates'],
                        saved['initial_source_sha256']) == (
                            'completed', 2585, 16, 64, state['source_sha256'])
            else:
                assert not (work/name).exists(), 'preserve incomplete arm for explicit recovery'
                assert shutil.disk_usage(work).free >= 2*1024**3, 'checkpoint headroom'
                assert shutil.disk_usage(temporary).free >= 6*1024**3, 'backward headroom'
                run(name, 'scripts.run_native_preview_restore_comparison', [
                    '--checkpoint', str(source), '--work', str(work/name), '--cycles', '16',
                    '--native-decision', '--native-word-execution', '--successor-distribution', '--joint-successor',
                    '--batch-native-words', '--batch-preview-scores', '--amortized-decision', '--latent-action-planner',
                    '--persistent-rhythm', 'coupled', '--root-native-correction', 'observe',
                    '--retention-scope-prior', name, '--table-storage', str(temporary/'table'),
                    '--world-activations', str(temporary/'world'), '--policy-activations', str(temporary/'policy'),
                    '--profile-runtime'])
            restored = work/f'{name}_restore.json'
            if not restored.exists():
                run(name+'_restore', 'scripts.audit_native_relation_use_restore', [
                    '--work', str(work/name), '--output', str(restored),
                    '--table-storage', str(temporary/'table')])
            assert json.loads(restored.read_text(encoding='utf8'))['valid']
            update(completed_arm=name, cycle=2585)
        summarize(work)
        update(status='completed', stage='completed',
               report='results/published/cgb_retention_scope_prior_2569_20261120.json')
    except Exception as error:
        update(status='failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
