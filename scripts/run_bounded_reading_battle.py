"""One immutable source, K=1 equivalence smoke and equal-time K=4 battle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts import run_equal_wallclock_rhythm_10min as audit

ROOT = Path('E:/mengli-bounded-reading-20261123')
PROJECT = Path('D:/information-boltzmann-v1')
SOURCE = Path('E:/mengli-wallclock-10min-20261123/strict_10min/body/last.pt.gz')
SOURCE_SHA = '1a848983905e07768dfe0e24f3720959a877b472a15aa765955f1f48213f11b3'
REPORT = PROJECT/'results/published/cgb_bounded_reading_battle_20261123.json'


def env():
    return dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        PYTHONIOENCODING='utf-8', PYTHONPATH=str(PROJECT),
        CGB_CHECKPOINT_SPOOL='E:/mengli-runtime-staging-20261123')


def run_arm(name, *, blocks=None, cycles=500, seconds=600, excess=.5):
    work = ROOT/name
    if (work/'result.json').exists():
        return json.loads((work/'result.json').read_text(encoding='utf8'))
    if work.exists():
        raise RuntimeError(f'existing incomplete arm requires inspection: {work}')
    tmp = ROOT/'temporary'/name
    args = [sys.executable, '-m', 'scripts.run_native_throughput_worker',
        '--profile-core', '--checkpoint', str(SOURCE), '--work', str(work),
        '--cycles', str(cycles), '--max-life-seconds', str(seconds),
        '--evaluation-material', 'E:/mengli-ridge-lr-acceptance-20261123/fresh_old_facts_64.pt',
        '--evaluate-every', '500', '--native-decision', '--native-word-execution',
        '--successor-distribution', '--joint-successor', '--batch-native-words',
        '--batch-preview-scores', '--amortized-decision', '--latent-action-planner',
        '--persistent-rhythm', 'coupled', '--root-native-correction', 'observe',
        '--retention-scope-prior', 'off', '--profile-runtime',
        '--activation-mode', 'tiered', '--activation-gpu-mib', '1536',
        '--activation-host-mib', '1024', '--direct-typed-question',
        '--factorized-endpoints', '--shared-utter-preview', '--fused-decode-step',
        '--native-replay-batch-size', '8', '--policy-proposal-batch-size', '4',
        '--reuse-objective-logits', '--warm-runtime', '--checkpoint-compressor', 'isal',
        '--checkpoint-compression-workers', '2', '--wiener-gate',
        '--split-hamiltonian-theta', '--learnable-precision-scales',
        '--table-storage', str(tmp/'table'), '--world-activations', str(tmp/'world'),
        '--policy-activations', str(tmp/'policy')]
    if blocks is not None:
        args += ['--reading-option-blocks', str(blocks), '--reading-option-surprise-excess', str(excess)]
    audit.write_json(ROOT/'status.json', dict(state='running', arm=name, command=args))
    with (ROOT/f'{name}.log').open('w', encoding='utf8') as log:
        p = subprocess.run(args, cwd=PROJECT, env=env(), stdout=log, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError(f'{name} failed ({p.returncode}); inspect log')
    digest = audit.reclaim_temp(name)
    result = json.loads((work/'result.json').read_text(encoding='utf8'))
    print(name, result['external_bytes'], result['life_seconds'], digest[:12], flush=True)
    return result


def rows(name):
    return [r for line in (ROOT/name/'events.jsonl').read_text(encoding='utf8').splitlines()
        if (r := json.loads(line)).get('autonomous_event', {}).get('kind') == 'external']


def setup():
    ROOT.mkdir(parents=True, exist_ok=True)
    assert audit.sha256_file(SOURCE) == SOURCE_SHA
    audit.OUT_ROOT = ROOT
    audit.SOURCE = SOURCE
    audit.FRESH_FACTS = ROOT/'strict_smoke/material.pt'
    # Both sets were fixed before observing any candidate endpoint.
    audit.FRESH_TEXT_SPEC = dict(audit.FRESH_TEXT_SPEC,
        starts=[4096, 98304, 155648, 221184])
    audit.CLASSIC_TEXT_SPEC = dict(audit.CLASSIC_TEXT_SPEC)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('smoke', 'battle'), required=True)
    args = parser.parse_args()
    setup()
    if args.stage == 'smoke':
        run_arm('strict_smoke', cycles=1, seconds=600)
        run_arm('k1_smoke', blocks=1, cycles=1, seconds=600)
        a, b = rows('strict_smoke')[0], rows('k1_smoke')[0]
        keys = ('choice', 'utterance_hex', 'external_bytes', 'response_nll',
                'metal_g', 'joint_pre_nll', 'joint_post_nll', 'gradient_norms')
        equality = {k: a[k] == b[k] for k in keys}
        assert all(equality.values()), equality
        run_arm('k4_goal_smoke', blocks=4, cycles=2, seconds=600)
        r = rows('k4_goal_smoke')
        assert all(x['bounded_reading_option']['unknown_id'] is not None for x in r)
        assert all(x['bounded_reading_option']['proposals'] == 4 for x in r)
        assert all(x['world_learning']['training_pieces']*480 == x['external_bytes'] for x in r)
        assert any(x['bounded_reading_option']['blocks'] > 1 for x in r), 'no actual multiblock exercised'
        run_arm('interrupt_smoke', blocks=4, excess=0., cycles=1, seconds=600)
        interrupt = rows('interrupt_smoke')[0]['bounded_reading_option']
        assert interrupt['stop'] == 'surprise' and interrupt['blocks'] < 4
        audit.write_json(ROOT/'smoke.json', dict(k1_equal=equality,
            k4=[x['bounded_reading_option'] for x in r], interrupt=interrupt, source_sha256=SOURCE_SHA))
        audit.write_json(ROOT/'status.json', dict(state='smoke_passed'))
        return
    assert (ROOT/'smoke.json').exists()
    # Freeze code, proof and test inputs for both arms.
    files = ['scripts/ib_local/bounded_reading_option.py', 'scripts/ib_local/world_learning_cycle.py',
        'scripts/ib_local/navigation_gain_units.py', 'scripts/ib_local/run_causal_water_reply.py',
        'scripts/ib_local/run_mother_continuation.py', 'scripts/run_native_preview_restore_comparison.py',
        'scripts/ib_local/mother_checkpoint.py',
        'scripts/run_native_throughput_worker.py', 'scripts/run_bounded_reading_battle.py',
        'scripts/run_equal_wallclock_rhythm_10min.py',
        'lean/CGBAxioms/reviews/BoundedReadingOption.lean']
    manifest = {f: audit.sha256_file(PROJECT/f) for f in files}
    audit.write_json(ROOT/'code_manifest.json', manifest)
    lean = subprocess.run(['C:/Users/85750/.elan/toolchains/leanprover--lean4---v4.36.0-rc1/bin/lean.exe',
        'lean/CGBAxioms/reviews/BoundedReadingOption.lean'], cwd=PROJECT, env=env(),
        capture_output=True, text=True)
    assert lean.returncode == 0, lean.stdout+lean.stderr
    started = time.perf_counter()
    origin = audit.eval_ckpt('origin_2897', SOURCE, ROOT)
    run_arm('strict_600s')
    run_arm('option4_600s', blocks=4)
    endpoints = {}
    for name in ('strict_600s', 'option4_600s'):
        ev = audit.eval_ckpt(name, ROOT/name/'body/last.pt.gz', ROOT/name)
        sm = audit.summarize_arm(name)
        life = rows(name)
        if name == 'option4_600s':
            state = ev['reading_option_state']
            assert state['options'] == len(life)
            assert state['blocks'] == sum(x['bounded_reading_option']['blocks'] for x in life)
            assert ev['pending_duration_blocks'] == life[-1]['bounded_reading_option']['blocks']
            assert ev['active_reading_option'] is False
        endpoints[name] = dict(summary=sm, audit=ev,
            fresh_nll_improvement=origin['fresh_8192_nll']['mean_nll']-ev['fresh_8192_nll']['mean_nll'],
            classic_nll_improvement=origin['classic_8192_nll']['mean_nll']-ev['classic_8192_nll']['mean_nll'],
            old_fact_score_improvement=ev['fresh_64_facts']['mean_score']-origin['fresh_64_facts']['mean_score'],
            option_receipts=[x['bounded_reading_option'] for x in life if 'bounded_reading_option' in x])
    assert audit.sha256_file(SOURCE) == SOURCE_SHA
    assert manifest == {f: audit.sha256_file(PROJECT/f) for f in files}
    result = dict(schema=1, source=str(SOURCE), source_sha256=SOURCE_SHA,
        source_cycle=2897, budget_seconds=600, fresh_text_spec=audit.FRESH_TEXT_SPEC,
        classic_text_spec=audit.CLASSIC_TEXT_SPEC, code_manifest=manifest,
        origin=origin, arms=endpoints, total_seconds=time.perf_counter()-started)
    audit.write_json(REPORT, result)
    audit.write_json(ROOT/'status.json', dict(state='completed', report=str(REPORT)))
    print(json.dumps({k:dict(bytes_per_second=v['summary']['bytes_per_second'],
        fresh_gain=v['fresh_nll_improvement'], classic_gain=v['classic_nll_improvement'],
        old_fact_gain=v['old_fact_score_improvement']) for k,v in endpoints.items()}, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        ROOT.mkdir(parents=True, exist_ok=True)
        audit.write_json(ROOT/'status.json', dict(state='failed', error=repr(error)))
        raise
