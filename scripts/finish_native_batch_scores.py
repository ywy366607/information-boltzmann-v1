"""After the active source-frozen audit, admit equivalent schema5 score batches.

One-shot continuation of the authorized acceleration work. Wait for the exact
previous full-loop/restore/report to pass, then patch the runtime gate, test it,
and run the same schema5 four rounds. Live experiment files stay frozen until
its coordinator has published completion. No model or paid-world change.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    a=p.parse_args();a.work.mkdir(exist_ok=False)
    status=dict(status='waiting',stage='previous complete-loop acceptance',pid=os.getpid())
    def write(**kw):
        status.update(kw)
        temp=a.work/'status.tmp'
        temp.write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf8')
        temp.replace(a.work/'status.json')
    def change(file,old,new):
        path=Path(file);text=path.read_text(encoding='utf8')
        if text.count(old)!=1:
            raise RuntimeError(f'expected original runtime gate once: {file}')
        backup=a.work/'before'/file
        backup.parent.mkdir(parents=True,exist_ok=True)
        if not backup.exists():backup.write_bytes(path.read_bytes())
        path.write_text(text.replace(old,new),encoding='utf8')
    def command(stage,args):
        write(stage=stage)
        with (a.work/(stage+'.log')).open('w',encoding='utf8') as log:
            result=subprocess.run([sys.executable,*args],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{stage} failed; see {stage}.log')
    try:
        write()
        while True:
            file=a.previous/'pipeline_status.json'
            state=json.loads(file.read_text(encoding='utf-8-sig')) if file.exists() else {}
            if state.get('status')=='failed':
                raise RuntimeError('previous audit failed; retain evidence before further runtime changes')
            if state.get('status')=='completed':break
            time.sleep(20)
        hashes=json.loads((a.previous/'source_hashes.json').read_text(encoding='utf8'))
        assert all(digest(Path(f))==h for f,h in hashes.items()), 'previous frozen source changed'
        previous_report=json.loads(Path('results/published/cgb_native_word_batch_loop_2493_20261119.json').read_text(encoding='utf8'))
        assert previous_report['valid']
        write(status='running',stage='apply equivalent score batching')
        change('scripts/ib_local/epistemic_value_reference.py',
            "batch = enabled(model) and getattr(model, '_preview_batch_enabled', True)",
            "batch = getattr(model, '_preview_batch_enabled', enabled(model))")
        change('scripts/ib_local/finite_use_value.py',
            "batch = joint_response.enabled(engine.model) and getattr(engine.model, '_preview_batch_enabled', True)",
            "batch = getattr(engine.model, '_preview_batch_enabled', joint_response.enabled(engine.model))")
        change('scripts/run_native_preview_restore_comparison.py',
            "    p.add_argument('--successor-distribution',action='store_true',",
            "    p.add_argument('--batch-preview-scores', action='store_true',\n"
            "                   help='runtime revision2: batch original schema5 finite-U scores')\n"
            "    p.add_argument('--successor-distribution',action='store_true',")
        change('scripts/run_native_preview_restore_comparison.py',
            "    if args.successor_distribution and not (args.native_decision and args.native_word_execution):",
            "    if args.batch_preview_scores and not args.batch_native_words:\n"
            "        p.error('--batch-preview-scores requires --batch-native-words')\n"
            "    if args.successor_distribution and not (args.native_decision and args.native_word_execution):")
        change('scripts/run_native_preview_restore_comparison.py',
            "        if args.resume_work and loaded.config.get('native_word_runtime', {}).get('mode', 'serial') != runtime:",
            "        runtime_schema = 2 if args.batch_preview_scores else 1\n"
            "        if args.resume_work and (loaded.config.get('native_word_runtime', {}).get('mode', 'serial'),\n"
            "                loaded.config.get('native_word_runtime', {}).get('schema', 1)) != (runtime, runtime_schema):")
        change('scripts/run_native_preview_restore_comparison.py',
            "        loaded.config['native_word_runtime'] = dict(schema=1, mode=runtime, batch_size=4)",
            "        loaded.config['native_word_runtime'] = dict(schema=runtime_schema, mode=runtime, batch_size=4)\n"
            "        model._preview_batch_enabled = args.batch_preview_scores or getattr(engine, 'epistemic_value_schema', 0)>=6")
        change('scripts/ib_local/run_mother_continuation.py',
            "    engine.native_word_runtime = config.get('native_word_runtime', {}).get('mode', 'serial')",
            "    engine.native_word_runtime = config.get('native_word_runtime', {}).get('mode', 'serial')\n"
            "    engine.model._preview_batch_enabled = (config.get('native_word_runtime', {}).get('schema', 1)>=2\n"
            "        or getattr(engine, 'epistemic_value_schema', 0)>=6)")
        change('tests/test_relation_retention_use.py',
            "    e.native_word_runtime = runtime",
            "    e.native_word_runtime = runtime\n"
            "    e.model._preview_batch_enabled = runtime=='batch' or epischema>=6")
        change('tests/test_relation_retention_use.py',
            "native_word_runtime=dict(schema=1,mode=runtime,batch_size=4)",
            "native_word_runtime=dict(schema=2 if runtime=='batch' else 1,mode=runtime,batch_size=4)")
        change('tests/test_relation_retention_use.py',
            "        other.native_word_runtime = stored['config']['native_word_runtime']['mode']",
            "        other.native_word_runtime = stored['config']['native_word_runtime']['mode']\n"
            "        other.model._preview_batch_enabled = stored['config']['native_word_runtime']['schema']>=2 or epischema>=6")
        command('tests',['-m','pytest','tests/test_joint_response.py','tests/test_relation_retention_use.py',
                         '-k','joint_response or 5_batch','-q'])
        loop=Path('scratch/cgb_native_batch_scores_loop_2493_20261119')
        command('full_loop',['-m','scripts.run_joint_response_pipeline','--route','word_batch_scores','--work',str(loop)])
        write(status='completed',stage='completed',loop=str(loop))
    except Exception as error:
        write(status='failed',error=str(error))
        raise


if __name__=='__main__':main()
