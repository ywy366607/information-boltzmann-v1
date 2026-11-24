"""Guarded serial full-life continuation, restore, and same-source report."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else _digest(f)


def _digest(f):
    h = hashlib.sha256()
    for block in iter(lambda: f.read(4*1024*1024), b''):
        h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--route', choices=('joint_response','word_batch','word_batch_scores','residual','amortized','action_planner'), default='joint_response')
    p.add_argument('--report-suffix', default='', choices=('', '_calibrated'))
    a = p.parse_args(); work = a.work.resolve()
    source = Path('scratch/cgb_momentum_value_2489_20261119/carry/body/last.pt.gz').resolve()
    expected = '7cccc744d4549eca2662727cba6b4c900609393243b9fb557dee6a2f17498107'
    assert digest(source) == expected, 'common source changed'
    # Completed matching four-round runs use about 1.1 GiB, including two
    # activation files. Keep six GiB for source copies and peak replacement.
    required = (6 if a.route == 'action_planner' else 8)*1024**3
    assert shutil.disk_usage(work.parent).free >= required, 'workspace lacks temporary headroom'
    work.mkdir(exist_ok=False)
    files = [Path('scripts/ib_local')/name for name in (
        'joint_response.py','joint_successor.py','epistemic_value_reference.py','finite_use_value.py',
        'causal_relation_program.py','navigation_causal_program.py','prospective_decision.py',
        'successor_distribution.py','generative_relations.py','causal_use_value.py','run_mother_continuation.py',
        'preview_profile.py','native_word_batch.py','prospective_self_execution.py','latent_navigation.py',
        'preview_runtime.py','native_residual_quadrature.py','navigation_internal_rollout.py',
        'causal_value_learning.py','native_amortized_decision.py','navigation_generative_cycle.py',
        'latent_action_planner.py','navigation_causal_program.py')]+[Path('scripts')/name for name in (
        'run_native_preview_restore_comparison.py','summarize_joint_response.py','audit_native_relation_use_restore.py',
        'summarize_native_word_batch.py','summarize_native_residual_loop.py','summarize_native_amortized_loop.py',
        'summarize_latent_action_loop.py')]
    hashes = {str(path):digest(path) for path in files}
    (work/'source_hashes.json').write_text(json.dumps(hashes, indent=2), encoding='utf8')
    state = dict(status='running', pid=os.getpid(), schema=6 if a.route=='joint_response' else 5, route=a.route,
                 started=datetime.now(timezone.utc).isoformat(), source_sha256=expected)
    def write(**fields):
        state.update(fields)
        temporary = work/'pipeline_status.tmp'
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf8')
        temporary.replace(work/'pipeline_status.json')
    def run(stage, module, arguments):
        with (work/(stage+'.log')).open('w', encoding='utf8') as log:
            child = subprocess.Popen([sys.executable, '-m', module, *arguments], stdout=log, stderr=subprocess.STDOUT)
            write(stage=stage, child_pid=child.pid)
            code = child.wait()
        if code:
            raise RuntimeError(f'{stage} failed with code {code}; see {stage}.log')
        for path, old in hashes.items():
            assert digest(Path(path)) == old, f'runtime source changed during {stage}: {path}'
    try:
        write(stage='starting')
        run('joint', 'scripts.run_native_preview_restore_comparison', [
            '--checkpoint',str(source),'--work',str(work/'joint'),'--cycles','4',
            '--native-decision','--native-word-execution','--successor-distribution',
            '--joint-successor','--profile-runtime',
            *(['--joint-response'] if a.route=='joint_response' else ['--batch-native-words']),
            *(['--batch-preview-scores'] if a.route in ('word_batch_scores','residual','amortized','action_planner') else []),
            *(['--residual-successor'] if a.route=='residual' else []),
            *(['--amortized-decision'] if a.route in ('amortized','action_planner') else []),
            *(['--latent-action-planner'] if a.route=='action_planner' else [])])
        run('restore', 'scripts.audit_native_relation_use_restore', ['--work',str(work/'joint'),
            '--output',str(work/'restore.json')])
        name = ('joint_response' if a.route=='joint_response' else
                'latent_action' if a.route=='action_planner' else
                'native_amortized' if a.route=='amortized' else
                'native_residual' if a.route=='residual' else
                'native_word_batch_scores' if a.route=='word_batch_scores' else 'native_word_batch')
        date = '20261120' if a.route in ('amortized','action_planner') else '20261119'
        output = Path(f'results/published/cgb_{name}_loop_2493_{date}{a.report_suffix}.json')
        module = ('summarize_joint_response' if a.route=='joint_response' else
                  'summarize_latent_action_loop' if a.route=='action_planner' else
                  'summarize_native_amortized_loop' if a.route=='amortized' else
                  'summarize_native_residual_loop' if a.route=='residual' else 'summarize_native_word_batch')
        run('summary', f'scripts.{module}', ['--work',str(work),'--output',str(output)])
        assert digest(source) == expected
        write(status='completed', stage='completed', cycles=4, updates=16, external_bytes=3840,
              report=str(output), source_unchanged=True)
    except Exception as error:
        write(status='failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
