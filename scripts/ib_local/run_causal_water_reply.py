"""One continuing OWT body/world loop with explicit retrieved-record credit.

Scale diagnostics precede a single-seed comparison. All arms restore the same
complete individual; frozen-policy read interventions do not act on the world.
"""
import argparse
import copy
import gzip
import gc
import hashlib
import json
import uuid
import time
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.ib_local import gdn_fused
from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
from scripts.ib_local.train_gdn_fast_table import TableGDN
from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable
from scripts.ib_local.language_navigation_kanli_v4b import attach_imagination
from scripts.ib_local.language_navigation_kanli_v4h import ReplyTopicWorld
from scripts.ib_local.language_navigation_kanli_v4m import WoodObservationBody
from scripts.ib_local.language_navigation_kanli_v4t import TypedWoodEngine
from scripts.ib_local.navigation_checkpoint import restore, save, snapshot
from scripts.ib_local.navigation_settlement import configure_settlement, configure_reward_memory
from scripts.ib_local.navigation_read_credit import configure_read_credit
from scripts.ib_local.world_source import decoded_world_bytes
from scripts.ib_local.run_language_navigation_v2 import validate


class WaterReplyEngine(TypedWoodEngine):
    def plan(self, h):
        result = super().plan(h)
        if not torch.is_grad_enabled():
            enabled = self.model.memory.read_credit_enabled
            try:
                self.model.memory.read_credit_enabled = False
                blind = super().plan(h)[0]
            finally:
                self.model.memory.read_credit_enabled = enabled
            self._water_plan = dict(habit=result[0][0].detach().clone(),
                                    blind=blind[0].detach().clone())
        return result

    def step(self, *args, **kwargs):
        out = super().step(*args, **kwargs)
        record = self._pending_record
        logits = self._water_plan['habit'].new_tensor(record['choice_logits'])
        blind = logits-self._water_plan['habit']+self._water_plan['blind']
        out['water_direct_choice_tv'] = float((logits.softmax(-1)-blind.softmax(-1)).abs().sum()/2)
        out['water_preference_abs_max'] = float(self.model.memory.read_credit_mean.abs().max())
        out['water_credit_records'] = int((self.model.memory.read_credit_count > 0).sum())
        return out


def to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, nn.Module):
        return value.to(device)
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{field.name: to_device(getattr(value, field.name), device)
                                 for field in fields(value) if field.init})
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(to_device(v, device) for v in value)
    if isinstance(value, list):
        return [to_device(v, device) for v in value]
    return value


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build(checkpoint, device='cuda', *, table_storage=None, allow_external_archive=False):
    with gzip.open(checkpoint, 'rb') as stream:
        state = torch.load(stream, map_location='cpu', weights_only=False)
    cfg = state['config']
    if cfg.get('causal_archive') is not None and not allow_external_archive:
        raise ValueError('use the mother continuation loader to restore the bound external causal archive')
    root = Path('data/ib_owt_10000')
    offset = json.loads((root/'tokenizer.json').read_text())['offset']
    if 'source_raw_bounds' in cfg:
        bounds = cfg['source_raw_bounds']
        window = cfg['topic_window']
    elif cfg.get('stage') == 'causal water in the complete real OWT reply loop':
        bounds = cfg['source']['source_raw_bounds']
        # The first saved reply-run schema used the fixed 4096-byte world
        # but only stored its size in the ancestor configuration. Admit that
        # known schema without requiring the ancestor checkpoint to exist.
        window = cfg['source'].get('topic_window', 4096)
    else:
        raise ValueError('unknown real-language checkpoint world contract')
    train = np.load(root/'train.npy', mmap_mode='r')
    raw = np.asarray(train[bounds[0]:min(bounds[1], len(train))])
    source = decoded_world_bytes(raw, offset)
    del train, raw
    world = ReplyTopicWorld(source, window=window, seed=cfg['seed'])
    base = TableGDN(GatedDeltaNetLM(256, 128, 4, 4, 'decay_then_delta', 256, 4, 32.,
                                   decay_init='log_timescale'), True).to(device)
    optimizer = torch.optim.AdamW(base.parameters(), lr=3e-4, foreach=True)
    model = attach_imagination(WoodObservationBody(base))
    optimizer.add_param_group(dict(params=[p for n,p in model.named_parameters()
                                          if not n.startswith('body.')], lr=3e-4))
    engine = WaterReplyEngine(model, optimizer, state['engine']['recurrent'].to(device),
        RetrievedCreditTable(orders=(0,), heads=1, slots_log2=1), world,
        state['engine']['pending'], np.random.default_rng(cfg['seed']), 'active',
        32, state['engine']['response_bytes'], state['engine']['holdout_bytes'], capture=False)
    if cfg.get('wood_memory_features') == 14:
        # Reconstruct the known optional F input before strict state/optimizer
        # restore. Legacy checkpoints take the unchanged branch above.
        if cfg.get('wood_evidence_schema') != 2:
            raise ValueError('experimental F input lacks carried evidence; restore the declared predecessor')
        from scripts.ib_local.navigation_causal_wood import configure_causal_wood
        configure_causal_wood(engine, cfg['mode'])
        if cfg.get('wood_gain_mode') == 'direct_occurrence':
            from scripts.ib_local.navigation_direct_read import configure_direct_read
            configure_direct_read(engine, cfg['mode'])
    if cfg.get('internal_rollout') is not None:
        spec = cfg['internal_rollout']
        if spec['schema'] != 1:
            raise ValueError('unknown internal rollout schema')
        from scripts.ib_local.navigation_internal_rollout import configure_internal_rollout
        configure_internal_rollout(engine, spec['mode'])
    if cfg.get('bfss_dynamics') is not None:
        spec = cfg['bfss_dynamics']
        if spec['schema'] != 1:
            raise ValueError('unknown matrix momentum schema')
        from scripts.ib_local.navigation_bfss_dynamics import configure_momentum
        configure_momentum(engine, enabled=spec['enabled'], dt=spec['dt'],
                           frequency=spec['frequency'], coupling=spec['coupling'],
                           steps=spec['steps'])
    if cfg.get('momentum_wood') is not None:
        spec = cfg['momentum_wood']
        if spec['schema'] != 1:
            raise ValueError('unknown momentum-to-wood schema')
        from scripts.ib_local.navigation_momentum_wood import configure_momentum_wood
        configure_momentum_wood(engine)
    if cfg.get('relation_wood') is not None:
        spec = cfg['relation_wood']
        if spec['schema'] != 1:
            raise ValueError('unknown local relation schema')
        from scripts.ib_local.navigation_relation_wood import configure_relation_wood
        configure_relation_wood(engine, ridge=spec['ridge'], scale=spec['scale'])
    if cfg.get('question_context') is not None:
        if cfg['question_context']['schema'] != 1:
            raise ValueError('unknown question context schema')
        from scripts.ib_local.navigation_question_context import configure_question_context
        configure_question_context(engine)
    if cfg.get('shadow_preview') is not None:
        spec = cfg['shadow_preview']
        if spec['schema'] != 1:
            raise ValueError('unknown shadow preview schema')
        from scripts.ib_local.navigation_shadow_preview import configure_shadow_preview
        configure_shadow_preview(engine, samples=spec['samples'], mode=spec.get('mode', 'observe'),
                                 candidate_kind=spec.get('candidate_kind', 'raw'))
    if cfg.get('mother_boundary') is not None:
        if cfg['mother_boundary']['schema'] != 1:
            raise ValueError('unknown mother boundary schema')
        from scripts.ib_local.navigation_mother_boundary import configure_mother_boundary
        configure_mother_boundary(engine)
    if cfg.get('persistent_question') is not None:
        if cfg['persistent_question']['schema'] != 1:
            raise ValueError('unknown persistent question schema')
        from scripts.ib_local.navigation_persistent_question import configure_persistent_question
        configure_persistent_question(engine)
    if cfg.get('bound_question_shadow') is not None:
        if cfg['bound_question_shadow']['schema'] != 1 or cfg.get('persistent_question') is None:
            raise ValueError('invalid bound-question shadow schema')
        engine.bound_question_shadow = True
    if cfg.get('joint_question_update') is not None:
        if cfg['joint_question_update']['schema'] != 1 or cfg.get('bound_question_shadow') is None:
            raise ValueError('invalid joint-question update schema')
        engine.joint_question_update = True
    if cfg.get('align_question_boundary') is not None:
        if cfg['align_question_boundary']['schema'] != 1 or cfg.get('joint_question_update') is None:
            raise ValueError('invalid question boundary alignment schema')
        engine.align_question_boundary = True
    if cfg.get('mother_shadow') is not None:
        if cfg['mother_shadow']['schema'] != 1 or cfg.get('joint_question_update') is None:
            raise ValueError('invalid mother shadow schema')
        engine.mother_shadow_enabled = True
    if cfg.get('question_parent_carry') is not None:
        if cfg['question_parent_carry']['schema'] != 1 or not getattr(engine, 'mother_shadow_enabled', False):
            raise ValueError('question parent carry requires mother-shadow evidence')
        engine.question_parent_carry = True
    if cfg.get('causal_program') is not None:
        if cfg['causal_program']['schema'] != 1:
            raise ValueError('unknown shared causal program schema')
        from scripts.ib_local.navigation_causal_program import configure_causal_program
        configure_causal_program(engine)
    if cfg.get('gain_clipping') is not None:
        if cfg['gain_clipping']['schema'] != 1:
            raise ValueError('unknown gain clipping schema')
        from scripts.ib_local.navigation_gain_clipping import prepare_gain_clipping
        prepare_gain_clipping(engine)
    if cfg.get('small_native_replay') is not None:
        from scripts.ib_local.small_gdn_experience import configure_small_replay
        configure_small_replay(engine, cfg['small_native_replay'])
    if cfg.get('learned_head_coupling') is not None:
        if cfg['learned_head_coupling'].get('schema') != 1:
            raise ValueError('unknown learned head coupling schema')
        from scripts.ib_local.navigation_learned_coupling import configure_learned_coupling
        configure_learned_coupling(engine, cfg['learned_head_coupling']['kind'])
    if cfg.get('joint_world_cycle') is not None:
        spec = cfg['joint_world_cycle']
        if spec['schema'] != 1:
            raise ValueError('unknown joint world-cycle schema')
        from scripts.ib_local.navigation_joint_cycle import configure_joint_cycle
        configure_joint_cycle(engine, spec['proposal_budget'])
    if cfg.get('generative_relations') is not None:
        if cfg['generative_relations']['schema'] != 1:
            raise ValueError('unknown generative relation schema')
        from scripts.ib_local.navigation_generative_cycle import configure_generative_cycle
        configure_generative_cycle(engine)
    state['engine'] = to_device(state['engine'], device)
    # This state came from a private torch.load and is discarded below; the
    # table may safely own its loaded arrays, avoiding a full duplicate.
    table_dir = None if table_storage is None else Path(table_storage) / uuid.uuid4().hex
    restore(engine, state, take_table_ownership=True, table_storage=table_dir)
    engine.device = next(model.parameters()).device
    source_meta = dict(checkpoint=str(checkpoint), sha256=sha256_file(checkpoint),
        cycle=state['cycle'], world_source_bytes=len(source),
        topic_window=window,
        world_source_sha256=hashlib.sha256(source).hexdigest(), source_raw_bounds=bounds,
        world='same restored OWT ReplyTopicWorld, including cursor, seen bitmap and RNG',
        optimizer_states=len(optimizer.state), pending_actor=engine.prev is not None)
    del state
    valid_raw = np.load(root/'validation.npy', mmap_mode='r')
    return engine, valid_raw, offset, source_meta


def resume_check(engine):
    """Two executions of one complete transition, then restore its start."""
    initial = snapshot(engine)
    first = engine.step(torch.randn(64, 32, device=engine.device))
    tensors = {k:v.detach().clone() for k,v in engine.model.state_dict().items() if torch.is_tensor(v)}
    table = engine.table.state_dict()
    world = engine.world.state_dict()
    restore(engine, initial)
    second = engine.step(torch.randn(64, 32, device=engine.device))
    max_error = max(float((v-tensors[k]).abs().max()) for k,v in engine.model.state_dict().items()
                    if torch.is_tensor(v) and v.is_floating_point() and v.numel())
    same_table = all(np.array_equal(v,table['arrays'][k]) for k,v in engine.table.state_dict()['arrays'].items())
    same_world = np.array_equal(world['seen'],engine.world.seen)
    same_metrics = all(first[k] == second[k] for k in ('choice','utterance_hex','metal_g','read_credit'))
    restore(engine, initial)
    assert max_error == 0 and same_table and same_world and same_metrics
    del initial, tensors, table, world
    gc.collect()
    return dict(max_model_error=max_error, same_table=same_table, same_world=same_world,
                same_metrics=same_metrics, restored_to_start=True)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    fused = gdn_fused.enable()
    engine, raw, offset, source = build(args.checkpoint)
    if args.credit is None:
        args.credit = (engine.model.memory.read_credit_rule
                       if engine.model.memory.read_credit_enabled else 'off')
    # Common to every arm: the new reward contract is not the experimental variable.
    configure_settlement(engine, 'continuation_joint')
    configure_reward_memory(engine, 'observed')
    configure_read_credit(engine.model.memory, args.credit, engine.optimizer.param_groups[0])
    engine.model.memory.read_credit_diagnostics = True
    valid = np.asarray(raw[:2*(args.validation_offset+args.validation_bytes+1)])
    valid = (valid[valid >= offset]-offset)[args.validation_offset:
        args.validation_offset+args.validation_bytes+1].astype(np.uint8)
    del raw
    paths = [Path(__file__)] + [Path('scripts/ib_local')/(name+'.py') for name in (
        'navigation_read_credit', 'navigation_retrieval', 'navigation_checkpoint',
        'navigation_settlement', 'latent_navigation_v3', 'latent_navigation_v2',
        'language_navigation_v2', 'language_navigation_kanli_v4',
        'language_navigation_kanli_v4b', 'language_navigation_kanli_v4c',
        'language_navigation_kanli_v4l', 'language_navigation_kanli_v4m',
        'language_navigation_kanli_v4p', 'language_navigation_kanli_v4q',
        'language_navigation_kanli_v4t', 'language_navigation_kanli_v4i',
        'language_navigation_kanli_v4k', 'run_language_navigation_v2', 'table_branch')]
    cfg = dict(stage='causal water in the complete real OWT reply loop', source=source,
        credit=args.credit, cycles=args.cycles, phase=args.phase, seed=11, fused=fused,
        topic_window=engine.world.window, source_raw_bounds=source['source_raw_bounds'],
        read_preference_rule=engine.model.memory.read_credit_rule,
        read_preference_optimizer=dict(lr=engine.model.memory.read_credit_lr,
            betas=engine.model.memory.read_credit_betas, eps=engine.model.memory.read_credit_eps),
        common_settlement='continuation_joint', common_reward_memory='observed',
        objective_migration=engine.settlement_migration,
        gain_migration=engine.reward_memory_migration,
        validation_bytes=len(valid)-1, validation_offset=args.validation_offset,
        validation_sha256=hashlib.sha256(valid.tobytes()).hexdigest(),
        validation='independent validation file, cold-state adaptive table/memory sidecar; live state unchanged',
        parameter_count=sum(p.numel() for p in engine.model.parameters()),
        sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        decision='scale phase measures delivery, not capability; train phase fixed budget, not claimed convergence')
    (args.output/'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    if args.resume_check:
        checked = resume_check(engine)
        (args.output/'resume_check.json').write_text(json.dumps(checked,indent=2),encoding='utf-8')
        print('RESUME', json.dumps(checked), flush=True)
    initial = validate(engine.model, engine.table, valid)
    (args.output/'initial.json').write_text(json.dumps(dict(validation_nll=initial)), encoding='utf-8')
    print('START', args.phase, args.credit, initial, flush=True)
    updates, step = [0], engine.optimizer.step
    def counted(*a, **kw):
        updates[0] += 1
        return step(*a, **kw)
    engine.optimizer.step = counted
    rows, start = [], time.perf_counter()
    prefix = [] if args.verify_prefix is None else [json.loads(line) for line in
                                                   args.verify_prefix.read_text().splitlines()]
    with (args.output/'metrics.jsonl').open('w', encoding='utf-8') as log:
        for cycle in range(args.cycles):
            row = engine.step(torch.randn(64, 32, device=engine.device))
            row.update(cycle=source['cycle']+cycle+1, new_cycle=cycle+1,
                       cumulative_optimizer_updates=updates[0], seconds=time.perf_counter()-start)
            if cycle < len(prefix):
                differences = [k for k,v in prefix[cycle].items() if k != 'seconds' and row.get(k) != v]
                if differences:
                    raise AssertionError(f'prior execution differs at {cycle+1}: {differences}')
            if (cycle+1) % args.validate_every == 0 or cycle+1 == args.cycles:
                row['validation_nll'] = validate(engine.model, engine.table, valid)
                if args.phase == 'train':
                    enabled = engine.model.memory.read_credit_enabled
                    try:
                        engine.model.memory.read_credit_enabled = False
                        row['validation_no_read_credit_nll'] = validate(engine.model, engine.table, valid)
                    finally:
                        engine.model.memory.read_credit_enabled = enabled
            log.write(json.dumps(row)+'\n'); log.flush(); rows.append(row)
            if cycle < 4 or (cycle+1) % 32 == 0:
                print('STEP', cycle+1, row['choice'], row['water_preference_abs_max'],
                      row['water_direct_choice_tv'], row.get('validation_nll'), flush=True)
    summary = dict(phase=args.phase, credit=args.credit, cycles=len(rows),
        verified_prior_prefix_cycles=min(len(prefix), len(rows)),
        optimizer_updates=updates[0], external_bytes=sum(r['external_bytes'] for r in rows),
        self_bytes=sum(r['self_bytes'] for r in rows), validation_initial=initial,
        validation_final=rows[-1]['validation_nll'],
        direct_choice_tv_max=max(r['water_direct_choice_tv'] for r in rows),
        direct_choice_tv_sum=sum(r['water_direct_choice_tv'] for r in rows),
        preference_abs_max=max(r['water_preference_abs_max'] for r in rows),
        max_policy_replay_error=max(r['policy_replay_max_error'] for r in rows),
        mean_joint_progress=float(np.mean([r['joint_progress'] for r in rows])),
        choices={kind:sum(r['choice']==kind for r in rows) for kind in ('silence','generate','reuse')},
        seconds=time.perf_counter()-start, peak_cuda_bytes=torch.cuda.max_memory_allocated())
    if args.phase == 'train':
        save(engine, args.output, source['cycle']+args.cycles, cfg)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('RESULT', json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--phase', choices=('scale','train'), default='scale')
    p.add_argument('--credit', choices=('off','mean','adam'),
                   help='Default: preserve the restored individual\'s credit rule')
    p.add_argument('--cycles', type=int, default=8)
    p.add_argument('--validation-bytes', type=int, default=4096)
    p.add_argument('--validation-offset', type=int, default=0)
    p.add_argument('--validate-every', type=int, default=64)
    p.add_argument('--resume-check', action='store_true')
    p.add_argument('--verify-prefix', type=Path)
    run(p.parse_args())


if __name__ == '__main__':
    main()
