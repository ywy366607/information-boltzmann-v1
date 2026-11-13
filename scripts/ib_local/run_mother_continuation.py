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
from scripts.ib_local.life_card import identity_for, branch_id_for, card_config_for, export_live
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


def continuation_parser():
    """Shared CLI defaults; the owner uses exactly the existing restore options."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=SOURCE)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cycles', type=int, required=True)
    p.add_argument('--mode', choices=('real','blocked'), default='real')
    p.add_argument('--device', choices=('cuda','cpu'), default='cuda',
                   help='explicit execution backend; CPU can validate the same complete model')
    p.add_argument('--table-storage', type=Path,
                   help='directory for a private disk-backed fast-table workspace')
    p.add_argument('--causal-storage', type=Path,
                   help='enable a unified SSD episode/relation archive with a private work file')
    p.add_argument('--causal-episode-capacity', type=int, default=65536)
    p.add_argument('--causal-relation-capacity', type=int, default=65536)
    p.add_argument('--generative-relations', action='store_true',
                   help='explicit whole-loop typed binding/program/evidence/internal-work migration')
    p.add_argument('--ambient-policy', choices=('reply','quiet'),
                   help='explicit event-boundary migration of inherited ambient navigation')
    p.add_argument('--mother-policy', choices=('legacy', 'responsive', 'responsive_v2', 'responsive_v3'),
                   help='explicit event-boundary caregiver strategy migration')
    p.add_argument('--shared-cache', type=Path, help='bounded shared actual-reply audit cache')
    p.add_argument('--mother-context', choices=('segmented', 'dsh'),
                   help='explicit caregiver-context migration; otherwise inherit the checkpoint')
    p.add_argument('--mother-harness-url', help='loopback URL of the official DSH mother service')
    p.add_argument('--mother-session', help='independent DSH branch identity; otherwise inherit it')
    p.add_argument('--life-card', type=Path,
                   help='explicit saved-snapshot card path under data/harness/life; DSH defaults to a branch card')
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
    p.add_argument('--question-parent-carry', action='store_true',
                   help='carry the reactivated real mother evidence into the next fixed-size question receipt')
    p.add_argument('--causal-program', action='store_true',
                   help='execute the shared typed relation, posterior revision, and mother/ambient planning graph')
    p.add_argument('--head-coupling', choices=('weighted','free'),
                   help='learn per-layer head interaction through actual language replay')
    p.add_argument('--joint-world-cycle', action='store_true',
                   help='joint source-aware learning, concrete language search and internal continuation')
    p.add_argument('--proposal-budget', type=int, default=4,
                   help='finite concrete language proposals for an explicit joint-cycle migration')
    p.add_argument('--validate-every', type=int, default=0,
                   help='optional fixed independent sidecars; zero preserves legacy runner')
    p.add_argument('--validation-bytes', type=int, default=8192)
    p.add_argument('--mother-validation', type=Path,
                   default=ROOT/'results/cgb_mother_continuation_20261110/validation/mother_validation.bin')
    p.add_argument('--gain-clipping', action='store_true',
                   help='bound only gain forecast innovations using prior observed scale')
    return p


def prepare_continuation(args, *, provider_factory=None, resume_output=False):
    """Restore the complete existing individual once, without running a cycle.

    CLI callers retain the new-directory requirement. The persistent owner may
    resume its own latest output, never replace an unrelated existing branch.
    """
    from types import SimpleNamespace
    resuming = args.output.exists()
    if args.cycles < 1 or (resuming and not resume_output):
        raise ValueError('positive cycles and new output directory required')
    if resuming and args.checkpoint.resolve() != (args.output/'last.pt.gz').resolve():
        raise ValueError('existing output must resume its own latest checkpoint')
    args.output.mkdir(parents=True, exist_ok=resuming)
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
    table_directory = getattr(args, 'table_storage', None)
    if table_directory is None and (config.get('causal_archive') is not None or
                                   getattr(args, 'causal_storage', None) is not None):
        table_directory = args.output/'fast_table_work'
    if config.get('causal_archive') is not None:
        engine, valid, offset, source = build(args.checkpoint, device=getattr(args, 'device', 'cuda'),
            table_storage=table_directory, allow_external_archive=True)
    elif table_directory is not None:
        engine, valid, offset, source = build(args.checkpoint, device=getattr(args, 'device', 'cuda'),
                                             table_storage=table_directory)
    elif getattr(args, 'device', 'cuda') == 'cuda':
        engine, valid, offset, source = build(args.checkpoint)
    else:
        engine, valid, offset, source = build(args.checkpoint, device='cpu')
    del valid, offset
    if args.question_parent_carry and config.get('question_parent_carry') is None:
        if not (getattr(engine, 'mother_shadow_enabled', False) and
                getattr(engine, 'joint_question_update', False)):
            raise ValueError('question parent carry requires the existing joint mother-shadow body')
        engine.question_parent_carry = True
        config['question_parent_carry'] = dict(schema=1, migrated_at_cycle=start_cycle,
            scope='retain the actual versioned parent mother observation in the next bounded receipt')
    if engine.__class__.__name__ not in ('DirectReadGainEngine', 'InternalRolloutEngine', 'MomentumEngine', 'MomentumWoodEngine', 'RelationWoodEngine', 'QuestionContextEngine', 'ShadowPreviewEngine', 'MotherBoundaryEngine', 'PersistentQuestionEngine', 'CausalProgramEngine', 'LearnedCouplingEngine', 'JointCycleEngine', 'JointLearnedCycleEngine', 'GenerativeCycleEngine', 'GenerativeLearnedCycleEngine'):
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
    if args.causal_program and config.get('causal_program') is None:
        from scripts.ib_local.navigation_causal_program import migrate_causal_program
        migrate_causal_program(engine)
        config['causal_program'] = dict(schema=1, migrated_at_cycle=start_cycle,
            relation_capacity=engine.model.causal_relations.capacity,
            evidence_budget=engine.model.causal_relations.evidence_budget,
            roles=['condition', 'subject=continuing body', 'action', 'mother', 'decision'],
            program=['body transition', 'mother response', 'shared evidence U', 'ambient transport', 'continuation'],
            uncertainty='normal-inverse-gamma residual operator in current body coordinates',
            calibration='new internal and shadow scores; previous history and pending labels retained',
            information='conditional operator quadrature diagnostic, no extra question reward')
    if args.head_coupling is not None:
        if config.get('learned_head_coupling') is not None:
            if config['learned_head_coupling']['kind'] != args.head_coupling:
                raise ValueError('cannot reinterpret an existing learned interaction')
        else:
            from scripts.ib_local.navigation_learned_coupling import configure_learned_coupling
            configure_learned_coupling(engine,args.head_coupling,calibrate=True)
            config['learned_head_coupling']=dict(schema=1,kind=args.head_coupling,
                migrated_at_cycle=start_cycle,
                learning='same real response loss and policy replay, one differentiable quiet prefix',
                retained='S/P, original weights and Adam, causal relations, world, pending credit',
                initialization='exact reference potential' if args.head_coupling=='weighted' else
                               'positive attention energy; initial force norm matched on inherited S',
                charge='original Q not assumed conserved for learned potentials')
    if (getattr(args, 'joint_world_cycle', False) and config.get('joint_world_cycle') is not None
            and config['joint_world_cycle']['proposal_budget'] != args.proposal_budget):
        raise ValueError('cannot silently change the saved joint proposal budget')
    if getattr(args, 'joint_world_cycle', False) and config.get('joint_world_cycle') is None:
        from scripts.ib_local.navigation_joint_cycle import migrate_joint_cycle
        migrate_joint_cycle(engine, args.proposal_budget)
        config['joint_world_cycle'] = dict(schema=1,proposal_budget=args.proposal_budget,
            migrated_at_cycle=start_cycle,settlement='world_stream',
            observations='same actual caregiver prefix and ambient stream, every byte learned once',
            utility='mean source-role joint predictive improvement on still-unlearned suffixes',
            planning='multiple concrete utterances through the same evidence U and continuation',
            internal='input-free S/P continuation; no prescribed goal or cadence',
            retained='all original parameters, Adam, S/P, episode evidence and actual pending credit')
        config['common_settlement'] = 'world_stream'
    if getattr(args, 'generative_relations', False) and config.get('generative_relations') is None:
        from scripts.ib_local.navigation_generative_cycle import configure_generative_cycle
        configure_generative_cycle(engine)
        config['generative_relations'] = dict(schema=1, migrated_at_cycle=start_cycle,
            grammar='shared response/transport/body-transition/continuation programs',
            binding='condition, action or response open role; same learned selector for internal work',
            revision='same finite evidence U, functional scope sharing/refutation',
            actor='all binding and internal choices join the original delayed policy trace',
            preserved='original parameters, Adam, S/P, world, facts and pending credit')
    if args.shared_cache:
        args.shared_cache.parent.mkdir(parents=True, exist_ok=True)
    inherited_policy = provider_state.get('mother_ambient_policy','quiet') if is_mother else 'reply'
    inherited_mother_policy = config.get('mother_policy', 'legacy')
    context=config.get('mother_context_harness')
    if args.mother_context == 'dsh':
        from scripts.ib_local.mengli_harness.mengli_world_client import (
            DEFAULT_ENDPOINT, DEFAULT_SETTINGS, branch_for_output)
        prior=context or {}
        context=dict(prior) if prior.get('kind')=='dsh' else dict(
            schema=2,kind='dsh',settings=dict(DEFAULT_SETTINGS),
            migrated_at_cycle=start_cycle,previous_harness=prior,
            scope='official persistent caregiver session and token-pressure memory compaction',
            world_change='mother retains summaries plus recent raw dialogue; summaries never feed the child')
        context['endpoint_url']=args.mother_harness_url or context.get('endpoint_url',DEFAULT_ENDPOINT)
        context['branch_id']=args.mother_session or context.get('branch_id') or branch_for_output(args.output)
        config['mother_context_harness']=context
    elif args.mother_context == 'segmented':
        if context is not None and context.get('kind')=='dsh':
            raise ValueError('a committed DSH world cannot silently drop its caregiver memory')
        if context is None:
            context=config['mother_context_harness']=dict(schema=1,kind='segmented',
                segment_turns=16,anchor_turns=4,max_context_chars=12000,
                migrated_at_cycle=start_cycle,
                scope='fixed leading caregiver instructions; append-only actual dialogue segments; provider KV telemetry',
                world_change='mother sees up to four anchor turns plus sixteen appended turns, not sliding six')
    if args.mother_harness_url or args.mother_session:
        if context is None or context.get('kind')!='dsh':
            raise ValueError('DSH endpoint/session options require a DSH caregiver world')
        context=dict(context)
        context['endpoint_url']=args.mother_harness_url or context['endpoint_url']
        context['branch_id']=args.mother_session or context['branch_id']
        config['mother_context_harness']=context
    world_type=MotherReplyWorld
    if context is not None:
        from scripts.ib_local.mother_context_harness import ContextCachedMother,ContextMotherWorld
        world_type=ContextMotherWorld
        if context.get('schema')==2 and context.get('kind')=='dsh':
            from scripts.ib_local.mengli_harness.mengli_world_client import DshMother
            provider=DshMother(endpoint_url=context['endpoint_url'],branch_id=context['branch_id'],
                settings=context['settings'],policy=inherited_mother_policy,
                provenance=dict(source_checkpoint=str(args.checkpoint.resolve()),
                                source_sha256=source_sha,cycle=start_cycle))
        elif context.get('schema')==1 and context.get('kind')=='segmented':
            provider=ContextCachedMother(shared_cache=args.shared_cache,policy=inherited_mother_policy,
                **{k:context[k] for k in ('segment_turns','anchor_turns','max_context_chars')})
        else:
            raise ValueError('unknown mother context harness')
    else:
        provider = StrictToddlerMother(shared_cache=args.shared_cache, policy=inherited_mother_policy)
    family_state = provider_state.get('mother_provider_state', {}).get('family_caregiver')
    if provider_factory is not None or family_state is not None:
        if provider_factory is None:
            from scripts.ib_local.mengli_harness.family_provider import FamilyCaregiver
            provider_factory = FamilyCaregiver
        provider = provider_factory(provider)
    engine.world = world_type(engine.world, provider,
        history=provider_state.get('mother_history',6),ambient_policy=inherited_policy)
    if is_mother:
        engine.world.load_state_dict(provider_state)
    if context is not None and context.get('kind')=='dsh':
        provider.inherit_world_history(engine.world.transcript)
    chosen_mother_policy = args.mother_policy or inherited_mother_policy
    if chosen_mother_policy != inherited_mother_policy:
        provider.set_policy(chosen_mother_policy)
        config['mother_policy_migration'] = dict(from_policy=inherited_mother_policy,
            to_policy=chosen_mother_policy, at_cycle=start_cycle,
            scope='caregiver instructions and prompt-keyed cache; preserve transcript and individual')
    if context is not None and context.get('kind')=='dsh':
        context['settings']=dict(provider.settings)
        context['cache_scope']='service-owned exact Chat Completions wire cache under data/harness'
        config['mother_harness_origin_ref']=(None if provider.checkpoint_ref is None
                                             else dict(provider.checkpoint_ref))
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
                  mother_shared_cache=('DSH service exact-wire cache under data/harness'
                    if context is not None and context.get('kind')=='dsh' else
                    None if args.shared_cache is None else str(args.shared_cache)),
                  mother_legacy_cache_argument=(str(args.shared_cache) if args.shared_cache is not None
                    and context is not None and context.get('kind')=='dsh' else None),
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
    config['life_identity'] = identity_for(config, source_sha)
    life_branch = branch_id_for(config, args.output/'last.pt.gz')
    life_card = None
    if args.life_card is not None or config.get('life_card') or (context or {}).get('kind') == 'dsh':
        life_card = config['life_card'] = card_config_for(config, life_branch, args.life_card)
    archive_manifest = config.get('causal_archive')
    archive_directory = getattr(args, 'causal_storage', None)
    if archive_manifest is not None or archive_directory is not None:
        from scripts.ib_local.causal_archive import attach_archive
        budgets = archive_manifest['budgets'] if archive_manifest is not None else dict(
            episodes=getattr(args, 'causal_episode_capacity', 65536),
            relations=getattr(args, 'causal_relation_capacity', 65536), max_bytes=512*1024*1024)
        attach_archive(engine, archive_directory or args.output/'causal_work', manifest=archive_manifest,
                       episode_capacity=budgets['episodes'], relation_capacity=budgets['relations'],
                       max_bytes=budgets['max_bytes'])
    card = None
    if not resuming:
        save(engine,args.output,start_cycle,config)
        (args.output/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        if life_card is not None:
            card = export_live(engine, checkpoint=args.output/'last.pt.gz', config=config,
                        cycle=start_cycle, output=life_card['path'], branch_id=life_branch)
    return SimpleNamespace(engine=engine, config=config, cycle=start_cycle,
        output=args.output, life_card=life_card, life_branch=life_branch,
        life_card_data=card, source=source, source_sha=source_sha)


def main():
    args = continuation_parser().parse_args()
    loaded = prepare_continuation(args)
    engine, config, start_cycle = loaded.engine, loaded.config, loaded.cycle
    life_card, life_branch = loaded.life_card, loaded.life_branch
    source, source_sha = loaded.source, loaded.source_sha
    evaluation=None
    if args.validate_every:
        if args.validate_every<1 or args.validation_bytes<512:
            raise ValueError('invalid independent validation budget')
        from scripts.ib_local.run_language_navigation_v2 import validate
        raw=np.load(ROOT/'data/ib_owt_10000/validation.npy',mmap_mode='r')
        token_offset=json.loads((ROOT/'data/ib_owt_10000/tokenizer.json').read_text())['offset']
        raw=np.asarray(raw[:2*(262144+args.validation_bytes+1)])
        owt=(raw[raw>=token_offset]-token_offset)[262144:262144+args.validation_bytes+1].astype(np.uint8)
        mom=np.frombuffer(args.mother_validation.read_bytes(),dtype=np.uint8)[:args.validation_bytes+1]
        if len(owt)!=args.validation_bytes+1 or len(mom)!=args.validation_bytes+1:
            raise ValueError('fixed validation data too short')
        def evaluation():
            return dict(owt_nll=validate(engine.model,engine.table,owt),
                        mother_nll=validate(engine.model,engine.table,mom),
                        bytes_each=args.validation_bytes,
                        scope='fixed cold sidecars; no gradient or live-state reset; no navigation during sidecar')
        (args.output/'initial_validation.json').write_text(json.dumps(evaluation(),indent=2),encoding='utf-8')
    start = time.perf_counter()
    completed = 0
    actual_updates=[0]
    original_step=engine.optimizer.step
    def counted_step(*a,**kw):
        result=original_step(*a,**kw)
        actual_updates[0]+=1
        return result
    engine.optimizer.step=counted_step
    with (args.output/'metrics.jsonl').open('w',encoding='utf-8') as log:
        for i in range(args.cycles):
            if args.stop_file is not None and args.stop_file.exists():
                break
            before_updates=actual_updates[0]
            row = engine.step(torch.randn(64,32,device=engine.device))
            row['updates']=actual_updates[0]-before_updates
            row['cumulative_optimizer_updates']=actual_updates[0]
            cycle = start_cycle+i+1
            # Commit the exact complete individual before exposing a result.
            save(engine,args.output,cycle,config)
            if life_card is not None:
                export_live(engine, checkpoint=args.output/'last.pt.gz', config=config,
                            cycle=cycle, row=row, output=life_card['path'], branch_id=life_branch)
            if evaluation is not None and ((i+1)%args.validate_every==0 or i+1==args.cycles):
                row['fixed_validation']=evaluation()
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
