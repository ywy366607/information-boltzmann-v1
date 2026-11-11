"""Continue the 6 Nov navigation individual in the existing mother world.

The caller chooses an output directory; the source is never overwritten.
Saving follows every committed interaction. Failures propagate without
turning a failed external API request into a fabricated silent observation.
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.ib_local import gdn_fused
from scripts.ib_local.mother_reply_world import MotherReplyWorld, StrictToddlerMother
from scripts.ib_local.mother_checkpoint import save
from scripts.ib_local.run_causal_water_reply import build, sha256_file

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('E:/cgb_causal_wood_20261106/direct_pair_replay_fixed/train_real_causal/last.pt.gz')


def internal_mode_config(config, mode, cycle, allow_change=False):
    """Record an explicit intervention; preserve all learned calibration/state."""
    old = config.get('internal_rollout', {})
    previous = old.get('mode', mode)
    if previous != mode and not allow_change:
        raise ValueError('internal rollout mode change requires --allow-internal-mode-change')
    result = dict(old, schema=1, mode=mode, target='next actual decision boundary',
                  depth=2, transition_weight=.1,
                  migrated_at_cycle=old.get('migrated_at_cycle', cycle))
    if previous != mode:
        result['mode_interventions'] = list(old.get('mode_interventions', [])) + [
            dict(from_mode=previous, to_mode=mode, at_cycle=cycle,
                 scope='suppress new continuation-to-choice offset only')]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=SOURCE)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cycles', type=int, required=True)
    p.add_argument('--mode', choices=('real','blocked'), default='real')
    p.add_argument('--ambient-policy', choices=('reply','quiet'),
                   help='explicit event-boundary migration of inherited ambient navigation')
    p.add_argument('--mother-policy', choices=('legacy', 'responsive', 'responsive_v2', 'responsive_v3'),
                   help='explicit event-boundary caregiver strategy migration')
    p.add_argument('--shared-cache', type=Path, help='bounded shared actual-reply audit cache')
    p.add_argument('--stop-file', type=Path, help='finish the current event then stop if this file exists')
    p.add_argument('--internal-rollout', choices=('observe', 'real'),
                   help='explicit long-head migration to actual next-decision boundaries')
    p.add_argument('--allow-internal-mode-change', action='store_true',
                   help='record a research control; keep inherited body/world/calibration')
    p.add_argument('--bfss-dynamics', action='store_true',
                   help='explicitly migrate existing internal rollout to matrix momentum')
    p.add_argument('--bfss-quiet-mode', choices=('enabled', 'disabled'),
                   help='record a decision-boundary intervention; real P writes continue')
    p.add_argument('--momentum-wood', action='store_true',
                   help='add trainable action-time P read to existing wood transition')
    p.add_argument('--relation-wood', action='store_true',
                   help='construct local transition relations from actual boundary evidence')
    p.add_argument('--question-context', action='store_true',
                   help='condition utterance on a pre-draft local relation query')
    p.add_argument('--shadow-preview', action='store_true',
                   help='observe bounded imagined relation updates; no policy delivery')
    p.add_argument('--shadow-mode', choices=('observe', 'real'),
                   help='explicitly select calibrated shadow contribution to choice')
    p.add_argument('--shadow-candidates', choices=('raw', 'centered'),
                   help='explicit candidate generation migration at a decision boundary')
    p.add_argument('--mother-boundary', action='store_true',
                   help='learn the observed end-of-mother-reply boundary in the same body')
    p.add_argument('--persistent-question', action='store_true',
                   help='retain bounded versioned pre-draft relation receipts')
    p.add_argument('--bound-question-shadow', action='store_true',
                   help='use the retained relation receipt for shadow updates')
    p.add_argument('--joint-question-update', action='store_true',
                   help='include the actual retained next boundary in both real and shadow relation solves')
    p.add_argument('--align-question-boundary', action='store_true',
                   help='capture actual next boundary before both draft and shadow')
    p.add_argument('--mother-shadow', action='store_true',
                   help='use versioned mother-end evidence in the shared shadow relation')
    p.add_argument('--gain-clipping', action='store_true',
                   help='bound only gain forecast innovations using prior observed scale')
    args = p.parse_args()
    if args.cycles < 1 or args.output.exists():
        raise ValueError('positive cycles and new output directory required')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    fused = gdn_fused.enable()
    source_sha = sha256_file(args.checkpoint)
    with gzip.open(args.checkpoint, 'rb') as f:
        saved = torch.load(f, map_location='cpu', weights_only=False)
    config, world_state, start_cycle = saved['config'], saved['world'], int(saved['cycle'])
    config = dict(config)
    is_mother = 'mother_transcript' in world_state
    if is_mother and config.get('mother_mode') != args.mode:
        raise ValueError('cannot silently change an existing mother individual mode')
    provider_state = {k:(v.cpu().numpy() if torch.is_tensor(v) else v) for k,v in world_state.items()}
    del saved, world_state
    engine, valid, offset, source = build(args.checkpoint)
    del valid, offset
    if engine.__class__.__name__ not in ('DirectReadGainEngine', 'InternalRolloutEngine', 'MomentumEngine', 'MomentumWoodEngine', 'RelationWoodEngine', 'QuestionContextEngine', 'ShadowPreviewEngine', 'MotherBoundaryEngine', 'PersistentQuestionEngine'):
        raise RuntimeError('expected inherited direct-read navigation individual')
    if args.internal_rollout:
        from scripts.ib_local.navigation_internal_rollout import configure_internal_rollout
        internal_config = internal_mode_config(config, args.internal_rollout, start_cycle,
                                              args.allow_internal_mode_change)
        if engine.__class__.__name__ in ('MomentumEngine', 'MomentumWoodEngine', 'RelationWoodEngine', 'QuestionContextEngine', 'ShadowPreviewEngine'):
            if args.internal_rollout != engine.internal_mode:
                raise ValueError('change internal mode before migrating S/P dynamics')
        else:
            configure_internal_rollout(engine, args.internal_rollout)
        config['internal_rollout'] = internal_config
    if args.bfss_dynamics and config.get('bfss_dynamics') is None:
        from scripts.ib_local.navigation_bfss_dynamics import configure_momentum
        configure_momentum(engine)
        config['bfss_dynamics'] = dict(schema=1, enabled=True, dt=engine.bfss_dt,
            frequency=engine.bfss_frequency, coupling=engine.bfss_coupling,
            steps=engine.bfss_steps, migrated_at_cycle=start_cycle,
            initialization='zero P; all existing S and history preserved',
            coordinate_assumption='common key/value basis within each layer')
    if args.bfss_quiet_mode is not None:
        if config.get('bfss_dynamics') is None:
            raise ValueError('quiet-mode intervention requires a migrated S/P individual')
        old = engine.bfss_enabled
        engine.bfss_enabled = args.bfss_quiet_mode == 'enabled'
        spec = config['bfss_dynamics'] = dict(config['bfss_dynamics'])
        spec['enabled'] = engine.bfss_enabled
        if old != engine.bfss_enabled:
            spec['interventions'] = list(spec.get('interventions', [])) + [dict(
                at_cycle=start_cycle, from_enabled=old, to_enabled=engine.bfss_enabled,
                scope='quiet S/P motion only; continue actual S/P event writes')]
    if args.momentum_wood and config.get('momentum_wood') is None:
        from scripts.ib_local.navigation_momentum_wood import configure_momentum_wood
        configure_momentum_wood(engine)
        config['momentum_wood'] = dict(schema=1, migrated_at_cycle=start_cycle,
            initialization='zero output preserves prior decision transition',
            query='each layer q_proj applied to the shared decision feature',
            receipt='decision-time P read, qualified by actual episode version',
            label='actual next decision boundary')
    if args.relation_wood and config.get('relation_wood') is None:
        from scripts.ib_local.navigation_relation_wood import configure_relation_wood
        configure_relation_wood(engine)
        config['relation_wood'] = dict(schema=1, ridge=engine.relation_ridge,
            scale=engine.relation_scale, migrated_at_cycle=start_cycle,
            evidence='unmerged version-valid actual decision-boundary pairs',
            operator='per-query local dual ridge residual, current encoder',
            target_exclusion='exclude own episode from real boundary loss')
    if args.question_context and config.get('question_context') is None:
        from scripts.ib_local.navigation_question_context import configure_question_context
        configure_question_context(engine)
        config['question_context'] = dict(schema=1, migrated_at_cycle=start_cycle,
            query='local relation residual before draft, source=action=current decision feature',
            conditioning='same additive learned byte-logit bias during decision and action replay',
            receipt='pre-draft slots, versions, valid mask, and context',
            initialization='zero output; original policy preserved at migration')
    if args.shadow_preview and config.get('shadow_preview') is None:
        from scripts.ib_local.navigation_shadow_preview import configure_shadow_preview
        configure_shadow_preview(engine)
        config['shadow_preview'] = dict(schema=1, samples=engine.shadow_samples, mode='observe',
            migrated_at_cycle=start_cycle, scope='diagnostic only, no choice offset',
            outcomes='historical action-conditioned residual candidates, uncalibrated')
    if args.shadow_mode is not None:
        if config.get('shadow_preview') is None:
            raise ValueError('shadow mode requires shadow preview')
        old_mode = engine.shadow_mode
        engine.shadow_mode = args.shadow_mode
        spec = config['shadow_preview'] = dict(config['shadow_preview'], mode=args.shadow_mode)
        if old_mode != args.shadow_mode:
            spec['interventions'] = list(spec.get('interventions', [])) + [dict(
                at_cycle=start_cycle, from_mode=old_mode, to_mode=args.shadow_mode,
                scope='calibrated shadow continuation contribution to choice')]
    if args.shadow_candidates is not None:
        if config.get('shadow_preview') is None:
            raise ValueError('candidate migration requires shadow preview')
        from scripts.ib_local.navigation_shadow_preview import migrate_shadow_candidates
        old_kind = engine.shadow_candidate_kind
        migrate_shadow_candidates(engine, args.shadow_candidates)
        spec = config['shadow_preview'] = dict(config['shadow_preview'], candidate_kind=args.shadow_candidates)
        if old_kind != args.shadow_candidates:
            spec['interventions'] = list(spec.get('interventions', [])) + [dict(
                at_cycle=start_cycle, from_candidates=old_kind, to_candidates=args.shadow_candidates,
                scope='center historical candidate residuals; preserve old calibration separately')]
    if args.mother_boundary and config.get('mother_boundary') is None:
        from scripts.ib_local.navigation_mother_boundary import configure_mother_boundary
        configure_mother_boundary(engine)
        config['mother_boundary'] = dict(schema=1, migrated_at_cycle=start_cycle,
            target='actual mother-reply end before ambient continuation',
            evidence='bounded versioned real episode, shared transition and relation',
            q='mother residual applied before drafting', initialization='zero output')
    if args.persistent_question and config.get('persistent_question') is None:
        from scripts.ib_local.navigation_persistent_question import configure_persistent_question
        configure_persistent_question(engine)
        config['persistent_question'] = dict(schema=1, migrated_at_cycle=start_cycle,
            role='observed mother outcome', identity='episode slot and write version',
            evidence='pre-draft relation receipt with version validation',
            initialization='zero contribution; original state and navigation preserved')
    if args.bound_question_shadow and config.get('bound_question_shadow') is None:
        if engine.__class__.__name__ != 'PersistentQuestionEngine':
            raise ValueError('bound shadow requires the persistent-question body')
        from scripts.ib_local.navigation_direct_read import gain_state
        engine.bound_question_shadow = True
        # The candidate changed; do not reuse calibration learned for the
        # previous unrestricted local relation.
        if not hasattr(engine, 'shadow_gain_archive'):
            engine.shadow_gain_archive = []
        engine.shadow_gain_archive.append(dict(kind='before_bound_question', state=engine.shadow_gain_state))
        engine.shadow_gain_state = gain_state(engine.wood_decay)
        if engine.internal_pending is not None and 'shadow_forecast' in engine.internal_pending:
            engine.internal_pending['pre_bound_shadow_forecast'] = engine.internal_pending.pop('shadow_forecast')
        config['bound_question_shadow'] = dict(schema=1, migrated_at_cycle=start_cycle,
            receipt='current pre-draft retained relation, exact versions',
            calibration='new; preceding shadow calibration archived',
            scope='same local ridge solve with imagined boundary evidence; not full Adam')
    if args.joint_question_update and config.get('joint_question_update') is None:
        if engine.__class__.__name__ != 'PersistentQuestionEngine' or not getattr(engine, 'bound_question_shadow', False):
            raise ValueError('joint relation update requires the bound-question shadow')
        from scripts.ib_local.navigation_direct_read import gain_state
        engine.joint_question_update = True
        if not hasattr(engine, 'shadow_gain_archive'):
            engine.shadow_gain_archive = []
        engine.shadow_gain_archive.append(dict(kind='before_joint_question', state=engine.shadow_gain_state))
        engine.shadow_gain_state = gain_state(engine.wood_decay)
        if engine.internal_pending is not None and 'shadow_forecast' in engine.internal_pending:
            engine.internal_pending['pre_joint_shadow_forecast'] = engine.internal_pending.pop('shadow_forecast')
        config['joint_question_update'] = dict(schema=1, migrated_at_cycle=start_cycle,
            actual='version-valid observed next decision boundary, including mother and ambient',
            imagined='additional candidate next boundary, never committed',
            operation='same bounded local ridge solve; not an Adam shadow',
            calibration='new; earlier calibration and pending forecast archived')
    if args.align_question_boundary and config.get('align_question_boundary') is None:
        if engine.__class__.__name__ != 'PersistentQuestionEngine' or not getattr(engine, 'joint_question_update', False):
            raise ValueError('boundary alignment requires joint-question update')
        from scripts.ib_local.navigation_direct_read import gain_state
        engine.align_question_boundary = True
        engine.shadow_gain_archive.append(dict(kind='before_boundary_alignment', state=engine.shadow_gain_state))
        engine.shadow_gain_state = gain_state(engine.wood_decay)
        if engine.internal_pending is not None and 'shadow_forecast' in engine.internal_pending:
            engine.internal_pending['pre_alignment_shadow_forecast'] = engine.internal_pending.pop('shadow_forecast')
        config['align_question_boundary'] = dict(schema=1, migrated_at_cycle=start_cycle,
            scope='capture once before pre-draft relation and shadow preview',
            calibration='new; earlier calibration and pending forecast archived')
    if args.mother_shadow and config.get('mother_shadow') is None:
        from scripts.ib_local.navigation_shadow_preview import enable_mother_shadow
        enable_mother_shadow(engine)
        config['mother_shadow'] = dict(schema=1, migrated_at_cycle=start_cycle,
            evidence='versioned mother-end observation before ambient input',
            operation='same local ridge solve for real and imagined mother evidence',
            calibration='new; earlier calibration and pending forecast archived',
            continuation='mother-end latent through existing continuation; approximation')
    if args.gain_clipping and config.get('gain_clipping') is None:
        from scripts.ib_local.navigation_gain_clipping import configure_gain_clipping
        configure_gain_clipping(engine)
        config['gain_clipping'] = dict(schema=1, migrated_at_cycle=start_cycle,
            limit='three times prior clipped absolute innovation EMA',
            initial_scale='median absolute historical prequential innovation, last 32 unmerged records',
            scope='wood evidence gain base update only; raw gains and other channels remain unchanged',
            initial_value=float(engine.model.wood_evidence.gain_innovation_scale))
    if args.shared_cache:
        args.shared_cache.parent.mkdir(parents=True, exist_ok=True)
    inherited_policy = provider_state.get('mother_ambient_policy','quiet') if is_mother else 'reply'
    inherited_mother_policy = config.get('mother_policy', 'legacy')
    provider = StrictToddlerMother(shared_cache=args.shared_cache, policy=inherited_mother_policy)
    engine.world = MotherReplyWorld(engine.world, provider,
                                    ambient_policy=inherited_policy)
    if is_mother:
        engine.world.load_state_dict(provider_state)
    chosen_mother_policy = args.mother_policy or inherited_mother_policy
    if chosen_mother_policy != inherited_mother_policy:
        provider.set_policy(chosen_mother_policy)
        config['mother_policy_migration'] = dict(from_policy=inherited_mother_policy,
            to_policy=chosen_mother_policy, at_cycle=start_cycle,
            scope='caregiver instructions and prompt-keyed cache; preserve transcript and individual')
    chosen_policy = args.ambient_policy or inherited_policy
    engine.world.ambient_policy = chosen_policy
    engine.episodic_wood_mode = args.mode
    config.update(mother_mode=args.mode, mother_history=engine.world.history,
                  mother_policy=chosen_mother_policy,
                  mother_source='DeepSeek contingent reply plus inherited ambient continuation',
                  mother_holdout='ambient continuation, not a held-out mother reply',
                  mother_ambient_policy=chosen_policy,
                  mother_ambient_migration=dict(from_policy=inherited_policy,to_policy=chosen_policy,
                                                at_cycle=start_cycle),
                  mother_shared_cache=None if args.shared_cache is None else str(args.shared_cache),
                  mother_origin_sha256=source_sha, mother_origin_cycle=start_cycle,
                  mother_comparison='blocked suppresses the qualified experience-to-choice contribution only',
                  mother_fused=fused)
    learned = {id(p) for group in engine.optimizer.param_groups for p in group['params']}
    named = [(name,p) for name,p in engine.model.named_parameters()]
    config['mother_parameter_audit'] = dict(total=sum(p.numel() for _,p in named),
        trainable=sum(p.numel() for _,p in named if p.requires_grad),
        trainable_in_optimizer=sum(p.numel() for _,p in named if p.requires_grad and id(p) in learned),
        trainable_missing_from_optimizer=[name for name,p in named if p.requires_grad and id(p) not in learned],
        frozen=[name for name,p in named if not p.requires_grad])
    (args.output/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    save(engine,args.output,start_cycle,config)
    start = time.perf_counter()
    completed = 0
    with (args.output/'metrics.jsonl').open('w',encoding='utf-8') as log:
        for i in range(args.cycles):
            if args.stop_file is not None and args.stop_file.exists():
                break
            row = engine.step(torch.randn(64,32,device=engine.device))
            cycle = start_cycle+i+1
            # Commit the exact complete individual before exposing a result.
            save(engine,args.output,cycle,config)
            row.update(cycle=cycle, mode=args.mode, elapsed_seconds=time.perf_counter()-start)
            log.write(json.dumps(row,ensure_ascii=False)+'\n')
            log.flush()
            completed += 1
            print(json.dumps({k:row.get(k) for k in
                ('cycle','choice','self_bytes','metal_g','wood_target','water_credit_records','world')},
                ensure_ascii=True),flush=True)
    (args.output/'completed.json').write_text(json.dumps(dict(status='complete' if completed==args.cycles else 'stopped',
        cycles=completed,requested_cycles=args.cycles,
        source_sha256=source_sha, final_cycle=start_cycle+completed,source=source),
        ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
