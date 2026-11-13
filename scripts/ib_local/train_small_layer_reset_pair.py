"""Real OWT paired learning: four layers carry vs layers1/3 reset S.

Both train and evaluate with the same128-byte boundary function. Start from
the original complete1792-cycle language individual, never a digit endpoint.
Only selected S matrices reset; convolution/history/experience remain carried.
Single GPU, two serial arms, existing objective and optimizer, no new modules.
"""
import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from scripts.ib_local import gdn_fused
from scripts.ib_local.native_state_policy import apply_native_boundary, configure_native_policy
from scripts.ib_local.navigation_checkpoint import save
from scripts.ib_local.run_causal_water_reply import build
from scripts.ib_local.small_gdn_ruler import fingerprint, request_branch, sha256
from scripts.ib_local.small_gdn_stream_eval import CarriedStream
from scripts.ib_local.train_small_native_credit import config_of, continuation_digest


def real_bytes(path, offset, start, count):
    raw = np.load(path, mmap_mode='r')
    stop = min(start+2*count, len(raw))
    section = np.asarray(raw[start:stop])
    selected = (section[section >= offset]-offset)[:count].astype(np.uint8)
    if len(selected) != count:
        raise ValueError('insufficient real bytes in requested source region')
    import hashlib
    return selected, dict(path=str(path), raw_interval=[start, stop], bytes=count,
                          sha256=hashlib.sha256(selected.tobytes()).hexdigest())


def training_step(engine, observed):
    raw = np.concatenate((np.array([engine.pending], dtype=np.uint8), observed))
    _, probability, _ = engine.table.run_credit(raw, max_window=128)
    ids, targets, logp = engine._inputs(raw, probability)
    incoming = apply_native_boundary(engine, engine.recurrent)
    loss, state, features, diag = engine.model.objective(
        ids, targets, incoming, logp, torch.randn(64, 32, device=engine.device))
    norm = engine._advance(loss, features, diag, state)
    engine.pending = int(raw[-1])
    engine.native_policy_training_updates += 1
    return dict(ce=float(diag['ce']), loss=float(loss.detach()), gradient_norm=norm,
                update=engine.native_policy_training_updates, observed=len(observed))


@torch.no_grad()
def evaluate(engine, observed, *, reference=False, keep_logits=False):
    before = continuation_digest(engine)
    policy = dict(engine.native_state_policy)
    with request_branch(engine, reset_history=False) as table:
        reader = CarriedStream(engine, table)
        score = reader.observe(observed, training_forward=reference)
        score['observed_memory_table_sha256'] = fingerprint(engine.model, table)
    assert continuation_digest(engine) == before
    assert engine.native_state_policy == policy
    score['policy'] = policy
    score['state_preserved'] = True
    score['nll_after_first1024'] = (float(np.mean(score['byte_nll'][1024:]))
                                   if len(observed) > 1024 else None)
    if not keep_logits:
        score = {k:v for k,v in score.items() if k not in ('logits', 'targets', 'final_state', 'byte_nll')}
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=4096)
    args = parser.parse_args()
    if args.steps < 1024 or args.steps % 4:
        parser.error('four equal phases and at least1024 updates required')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    gdn_fused.enable()
    data = Path('data/ib_owt_10000')
    offset = json.loads((data/'tokenizer.json').read_text())['offset']
    phase_steps, phase_bytes = args.steps//4, args.steps//4*128
    blocks, provenance = {}, {}
    for name, start in [('A', 4_000_000), ('B', 5_000_000), ('C', 6_000_000)]:
        blocks[name], provenance[name] = real_bytes(data/'train.npy', offset, start, phase_bytes)
    valid, provenance['validation'] = real_bytes(data/'validation.npy', offset, 262144, 8192)
    stream = np.concatenate((blocks['A'], blocks['B'], blocks['A'], blocks['C']))
    phases = ['A_first_in_run', 'B_first_in_run', 'A_revisit', 'C_first_in_run']
    report = dict(status='running', protocol=__doc__, source_checkpoint=str(args.checkpoint),
        source_sha256=sha256(args.checkpoint), seed_policy='same saved RNG/Adam/weights for both arms',
        interpretation='four layers alternate, not alternating whole-engine time steps',
        boundary_bytes=128, reset_scope='S_only_keep_short_convolution',
        steps_per_arm=args.steps, observed_bytes_per_arm=len(stream), phases=phases,
        phase_bytes=phase_bytes, source_data=provenance, arms={}, admissions={}, new_lean_coverage=0,
        capability_scope='fixed real-language budget; not convergence or infinite-lifetime proof')
    files = {Path(m.__file__) for name,m in sys.modules.copy().items()
             if name.startswith('scripts.ib_local.') and getattr(m, '__file__', None)} | {Path(__file__)}
    report['source_files'] = {str(p):sha256(p) for p in sorted(files)}
    sources = args.output/'executed_sources'
    sources.mkdir()
    for p in files:
        shutil.copy2(p, sources/p.name)
    destination = args.output/'report.json'
    started = time.time()
    def write():
        report['seconds'] = time.time()-started
        temporary = destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
        temporary.replace(destination)
    write()
    cfg = config_of(args.checkpoint)
    try:
        # Independent admission branches; calibration updates are discarded.
        for arm, reset_layers in [('all_carry', ()), ('alternating_S_reset', (0, 2))]:
            engine, _, _, source = build(args.checkpoint)
            policy = configure_native_policy(engine, reset_layers)
            engine.native_policy_training_updates = 0
            admission = dict(policy=policy)
            for i in range(8):
                training_step(engine, stream[128*i:128*(i+1)])
            reference = evaluate(engine, valid[:256], reference=True, keep_logits=True)
            inference = evaluate(engine, valid[:256], keep_logits=True)
            maximum = float((reference['logits']-inference['logits']).abs().max())
            assert maximum == 0
            assert torch.equal(reference['final_state'], inference['final_state'])
            assert reference['observed_memory_table_sha256'] == inference['observed_memory_table_sha256']
            admission['training_inference_logits_max_error'] = maximum
            admission['training_inference_state_and_memory_exact'] = True
            folder = args.output/('admission_'+arm)
            folder.mkdir()
            local_cfg = dict(cfg, native_state_policy=policy)
            save(engine, folder, source['cycle'], local_cfg)
            a = training_step(engine, stream[8*128:9*128])
            before = continuation_digest(engine)
            del engine, reference, inference
            gc.collect()
            torch.cuda.empty_cache()
            engine, _, _, _ = build(folder/'last.pt.gz')
            assert engine.native_state_policy == policy
            b = training_step(engine, stream[8*128:9*128])
            assert a == b and continuation_digest(engine) == before
            admission.update(disk_next_update_exact=True, saved_policy_restored=True, passed=True)
            report['admissions'][arm] = admission
            write()
            print('POLICY ADMISSION', arm, 'PASSED', flush=True)
            del engine
            gc.collect()
            torch.cuda.empty_cache()
        for arm, reset_layers in [('all_carry', ()), ('alternating_S_reset', (0, 2))]:
            engine, _, _, source = build(args.checkpoint)
            policy = configure_native_policy(engine, reset_layers)
            engine.native_policy_training_updates = 0
            result = dict(policy=policy, steps=[], evaluations=[dict(update=0, validation=evaluate(engine, valid))])
            report['arms'][arm] = result
            folder = args.output/arm
            folder.mkdir()
            local_cfg = dict(cfg, native_state_policy=policy)
            for i in range(args.steps):
                if (args.output/'STOP').exists():
                    raise InterruptedError('user STOP requested')
                step = training_step(engine, stream[i*128:(i+1)*128])
                step['phase'] = phases[i//phase_steps]
                result['steps'].append(step)
                update = i+1
                if update % 256 == 0:
                    print('POLICY', arm, update, step['phase'], step['ce'], flush=True)
                if update % 512 == 0 or update == args.steps or update % phase_steps == 0:
                    result['evaluations'].append(dict(update=update, validation=evaluate(engine, valid)))
                    local_cfg['layer_reset_experiment'] = dict(arm=arm, updates=update, phases=phases,
                                                               window=128, phase_steps=phase_steps)
                    save(engine, folder, source['cycle'], local_cfg)
                    print('POLICY EVAL', arm, update, result['evaluations'][-1]['validation']['nll'], flush=True)
                if update % 64 == 0 or update == args.steps:
                    write()
            result['phase_summaries'] = {phase:dict(
                mean_prequential_nll=float(np.mean([s['ce'] for s in result['steps'] if s['phase']==phase])),
                last256_mean_prequential_nll=float(np.mean([s['ce'] for s in result['steps'] if s['phase']==phase][-256:])))
                for phase in phases}
            result['checkpoint'] = str(folder/'last.pt.gz')
            del engine
            gc.collect()
            torch.cuda.empty_cache()
        report.update(status='complete', source_unchanged=sha256(args.checkpoint)==report['source_sha256'],
            executed_sources_unchanged=all(sha256(p)==h for p,h in report['source_files'].items()),
            peak_cuda_bytes=torch.cuda.max_memory_allocated())
        assert report['source_unchanged'] and report['executed_sources_unchanged']
    except Exception as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        write()


if __name__ == '__main__':
    main()
