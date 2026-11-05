"""Single continuous OWT reply loop for v4p; no population or paired training."""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from scripts.ib_local import gdn_fused, language_navigation_kanli_v4i
from scripts.ib_local import run_language_navigation_kanli_v4 as runner
from scripts.ib_local.language_action import digest, peek
from scripts.ib_local.language_navigation_kanli_v4m import WoodObservationBody
from scripts.ib_local.language_navigation_kanli_v4p import ResidualWoodEngine
from scripts.ib_local.language_navigation_kanli_v4q import SharedContextWoodEngine
from scripts.ib_local.run_language_navigation_v2 import validate


# Transient per-step tensors are overwritten before use. All other engine
# attributes, including pending TD/actor credit and witness RNG, are saved.
_DERIVED = {'model', 'optimizer', 'decision', 'world', 'table', 'params', 'groups',
    '_forecast', '_wood_replay', '_batch_baseline', '_pending_record', '_draft_actions',
    '_wood_source', '_wood_action', '_wood_predictions', 'draft_diagnostics'}


def snapshot(engine):
    return copy.deepcopy(dict(model=engine.model.state_dict(), optimizer=engine.optimizer.state_dict(),
        engine={k:v for k,v in vars(engine).items() if k not in _DERIVED},
        table=engine.table.state_dict(), world=engine.world.state_dict(),
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
        numpy_rng=np.random.get_state(), python_rng=random.getstate()))


def restore(engine, state):
    engine.model.load_state_dict(state['model'])
    engine.optimizer.load_state_dict(state['optimizer'])
    world = {k:(v.cpu().numpy() if torch.is_tensor(v) else v) for k,v in state['world'].items()}
    engine.world.load_state_dict(copy.deepcopy(world))
    engine.table = type(engine.table).from_state_dict(state['table'])
    for k,v in copy.deepcopy(state['engine']).items():
        setattr(engine, k, v)
    torch.set_rng_state(state['torch_rng'])
    torch.cuda.set_rng_state_all(state['cuda_rng'])
    np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])


def save_checkpoint(engine, output, cycle, config):
    target = output/'last.pt.gz'
    tmp = output/'last.tmp.gz'
    # Training is paused here. Serialize tensor-backed views without making
    # another 235 MB table copy, or pickling a giant NumPy bytes object.
    table = engine.table
    table_state = dict(arrays={k:torch.from_numpy(getattr(table, k)) for k in
        ('rows','count','tag','err','epoch','last_write')}, history=int(table.history),
        events=table.events, config={k:getattr(table,k) for k in
        ('orders','heads','slots_log2','gain_floor','err_floor','prior_weight')})
    world_state = engine.world.state_dict()
    for k,v in world_state.items():
        if isinstance(v,np.ndarray):
            world_state[k] = torch.from_numpy(v)
    state = dict(model=engine.model.state_dict(), optimizer=engine.optimizer.state_dict(),
        engine={k:v for k,v in vars(engine).items() if k not in _DERIVED},
        table=table_state, world=world_state,
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
        numpy_rng=np.random.get_state(), python_rng=random.getstate(),cycle=cycle,config=config)
    with gzip.open(tmp, 'wb', compresslevel=1) as stream:
        torch.save(state, stream)
    tmp.replace(target)


def resume_audit(engine):
    """Repeat one real transition; leave the live individual at its start."""
    state = snapshot(engine)
    first = engine.step(torch.randn(64, 32, device='cuda'))
    weights = {k:v.detach().clone() for k,v in engine.model.state_dict().items() if torch.is_tensor(v)}
    first_world = engine.world.state_dict()
    first_table = engine.table.state_dict()
    restore(engine, state)
    second = engine.step(torch.randn(64, 32, device='cuda'))
    errors = {k:float((v-weights[k]).abs().max()) for k,v in engine.model.state_dict().items()
              if torch.is_tensor(v) and v.is_floating_point() and v.numel()}
    same_table = all(np.array_equal(v, first_table['arrays'][k])
                     for k,v in engine.table.state_dict()['arrays'].items())
    fields = ('choice', 'wood_target', 'wood_prediction', 'metal_g', 'self_bytes')
    same_metrics = all(first[k] == second[k] for k in fields)
    same_world = np.array_equal(first_world['seen'], engine.world.state_dict()['seen'])
    report = dict(max_parameter_buffer_error=max(errors.values()),
                  same_table=same_table, same_metrics=same_metrics, same_world_seen=same_world,
                  saved_engine_fields=sorted(state['engine']))
    restore(engine, state)
    assert report['max_parameter_buffer_error'] == 0 and same_table and same_metrics and same_world, report
    return report


def wiring(engine):
    """One derivative check on the actual checkpoint's current state."""
    engine.decision.start(engine.recurrent)
    with torch.no_grad():
        _, h = engine.decision.step(engine.pending, peek(engine.table, engine.pending))
        h = h.detach().reshape(1, -1)
    ids = torch.arange(3, device=engine.device)
    residual = engine._residual(h.expand(3, -1), h.expand(3, -1), ids)
    grad = torch.autograd.grad(residual.sum(), engine.model.kanli.outcome[-1].weight)[0]
    z = engine.model.encoder(h).detach().expand(3, -1)
    width = z.shape[-1]
    inputs = torch.cat((z, torch.zeros_like(z), engine.intero().expand(3, -1),
                        torch.eye(3, device=z.device), z.new_zeros(3, 2)), -1).requires_grad_()
    critic_grad = torch.autograd.grad(engine.model.kanli.critic(inputs).sum(), inputs)[0][:, width:width+4]
    report = dict(zero_residual_max=float(residual.detach().abs().max()),
                  residual_head_gradient_norm=float(grad.norm()),
                  outcome_to_value_gradient_norm=float(critic_grad.norm()),
                  initial_evidence_count=int(engine.model.wood_evidence.seen.sum()),
                  critic_precision_initial=engine.precision(),
                  note='Existing critic warmup suppresses policy offsets initially; measured live after warmup.')
    assert report['zero_residual_max'] == 0
    assert report['residual_head_gradient_norm'] > 0 and report['outcome_to_value_gradient_norm'] > 0
    assert report['initial_evidence_count'] == 0
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cycles', type=int, default=128)
    p.add_argument('--validation-bytes', type=int, default=4096)
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--variant', choices=('p', 'q'), default='p')
    p.add_argument('--validate-every', type=int, default=64)
    p.add_argument('--checkpoint-every', type=int, default=256)
    p.add_argument('--resume-audit', action='store_true')
    p.add_argument('--resume-checkpoint', type=Path)
    p.add_argument('--verify-replay', type=Path, help='Prior metrics prefix to reconstruct exactly after an infrastructure failure')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    fused = gdn_fused.enable()
    old_body, old_engine = runner.LatentNavigationV3, language_navigation_kanli_v4i.FilmEngine
    try:
        runner.LatentNavigationV3 = WoodObservationBody
        language_navigation_kanli_v4i.FilmEngine = SharedContextWoodEngine if args.variant == 'q' else ResidualWoodEngine
        engine, table, valid, cfg = runner.build(equal=True, reply=True, film=True, capture=False,
                                                validation_bytes=args.validation_bytes, seed=args.seed)
    finally:
        runner.LatentNavigationV3, language_navigation_kanli_v4i.FilmEngine = old_body, old_engine
    del table  # resume audits replace engine.table; do not retain the old 235 MB copy
    files = [Path(__file__), Path('scripts/ib_local/language_navigation_kanli_v4p.py')]
    if args.variant == 'q':
        files.append(Path('scripts/ib_local/language_navigation_kanli_v4q.py'))
    files += [Path(f'scripts/ib_local/language_navigation_kanli_v4{s}.py') for s in ('k', 'l', 'm')]
    cfg.update(stage=f'v4{args.variant} shared loop with calibrated latent residual', fused=fused,
               cycles=args.cycles, validation_bytes=len(valid)-1, language_weight=10., sigreg_weight=.001,
               outcome_weight=engine.outcome_weight, replay_batch=engine.replay_batch,
               optimizer_lrs=[g['lr'] for g in engine.optimizer.param_groups],
               stop='fixed 128-turn diagnostic budget, not convergence' if args.cycles == 128 else 'fixed diagnostic budget',
               parameter_count=sum(p.numel() for p in engine.model.parameters()),
               sources={**cfg['sources'], **{str(f): digest(f) for f in files}})
    (args.output/'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    gate = wiring(engine)
    (args.output/'wiring.json').write_text(json.dumps(gate, indent=2), encoding='utf-8')
    print('WIRING', json.dumps(gate), flush=True)
    start_cycle = 0
    if args.resume_checkpoint:
        with gzip.open(args.resume_checkpoint, 'rb') as stream:
            saved = torch.load(stream, weights_only=False)
        if saved['config']['stage'] != cfg['stage']:
            raise ValueError('checkpoint variant mismatch')
        restore(engine, saved)
        start_cycle = saved['cycle']
        cfg['resume_checkpoint'] = str(args.resume_checkpoint)
        cfg['resume_cycle'] = start_cycle
        (args.output/'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
        del saved
    replay_prefix = [] if args.verify_replay is None else [json.loads(line) for line in args.verify_replay.read_text().splitlines()]
    if args.resume_audit:
        audit = resume_audit(engine)
        (args.output/'resume_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
        print('RESUME', json.dumps(audit), flush=True)
    initial = validate(engine.model, engine.table, valid)
    (args.output/'initial.json').write_text(json.dumps(dict(validation_nll=initial)), encoding='utf-8')
    step_count = 0
    step = engine.optimizer.step
    def counted_step(*a, **kw):
        nonlocal step_count
        result = step(*a, **kw)
        step_count += 1
        return result
    engine.optimizer.step = counted_step
    rows, start = [], time.perf_counter()
    with (args.output/'metrics.jsonl').open('w', encoding='utf-8') as log:
        for i in range(args.cycles):
            before = step_count
            row = engine.step(torch.randn(64, 32, device='cuda'))
            row.update(cycle=start_cycle+i+1, optimizer_steps=step_count-before, cumulative_optimizer_steps=step_count)
            if i < len(replay_prefix):
                old = replay_prefix[i]
                assert all(row[k] == old[k] for k in old if k != 'validation_nll'), f'reconstruction differs at cycle {i+1}'
                if i+1 == len(replay_prefix):
                    (args.output/'reconstruction.json').write_text(json.dumps(dict(exact_prefix_cycles=len(replay_prefix),
                        source=str(args.verify_replay))),encoding='utf-8')
            if (i+1) % args.validate_every == 0 or i+1 == args.cycles:
                row['validation_nll'] = validate(engine.model, engine.table, valid)
            rows.append(row)
            log.write(json.dumps(row)+'\n'); log.flush()
            if (i+1) % args.checkpoint_every == 0 or i+1 == args.cycles:
                save_checkpoint(engine, args.output, start_cycle+i+1, cfg)
            if (i+1) % 16 == 0:
                print(json.dumps(dict(cycle=i+1, elapsed=round(time.perf_counter()-start, 1),
                    trust=row['wood_confidence_next'], policy_tv=row['wood_policy_tv'],
                    residual_tv=row['residual_policy_tv'])), flush=True)
    late = rows[len(rows)//2:]
    summary = dict(scope='live OWT reply loop, diagnostic budget; no claim of convergence or paired causal gain',
        cycles=len(rows), initial_nll=initial, final_nll=rows[-1]['validation_nll'],
        elapsed_seconds=time.perf_counter()-start, external_bytes=sum(r['external_bytes'] for r in rows),
        self_bytes=sum(r['self_bytes'] for r in rows), optimizer_steps=step_count,
        replay_samples=sum(r['wood_replay_samples'] for r in rows),
        peak_cuda_allocated=torch.cuda.max_memory_allocated(),
        late_errors={name:np.mean([r['wood_'+name+'_error'] for r in late], axis=0).tolist()
                     for name in ('blind', 'base', 'raw', 'used')},
        late_trust_mean=np.mean([r['wood_confidence_used'] for r in late], axis=0).tolist(),
        late_policy_tv_mean=float(np.mean([r['wood_policy_tv'] for r in late])),
        late_residual_tv_mean=float(np.mean([r['residual_policy_tv'] for r in late])),
        expected_common_uniform_divergences=sum(r['wood_policy_tv'] for r in rows),
        choices={c:sum(r['choice']==c for r in rows) for c in ('silence','generate','reuse')},
        max_replay_error=max(r['policy_replay_max_error'] for r in rows))
    # TV is maximal-coupling disagreement, not the probability for an arbitrary
    # shared multinomial sampler. Keep the naming explicit in saved output.
    summary['sum_policy_total_variation'] = summary.pop('expected_common_uniform_divergences')
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('RESULT', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
