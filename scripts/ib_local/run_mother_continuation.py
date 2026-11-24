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
from scripts.native_policy_activation_storage import (
    add_storage_arguments, configure_runtime_storage, event_storage, storage_options)

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
    add_storage_arguments(p)
    p.add_argument('--checkpoint', type=Path, default=SOURCE)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cycles', type=int, required=True)
    p.add_argument('--mode', choices=('real','blocked'), default='real')
    p.add_argument('--speech-only', action='store_true',
                   help='keep episodic reads, but expose only silence and generated speech')
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
    p.add_argument('--shared-unknowns', action='store_true',
                   help='persist typed questions and actual constraints across relation identities')
    p.add_argument('--content-question', action='store_true',
                   help='explicit recurrent question conditioning and speech/listening scope repair')
    p.add_argument('--grow-body', type=Path,
                   help='explicit function-preserving deep/width growth configuration')
    p.add_argument('--variable-expression',action='store_true',
                   help='learn content/budget-conditioned handover, up to 128 bytes')
    p.add_argument('--past-present-coevolution', type=Path,
                   help='explicit all-layer causal retrieved-field migration specification')
    p.add_argument('--admit-only',action='store_true',
                   help='save the complete migrated individual without calling the world or updating it')
    p.add_argument('--autonomous-events', action='store_true',
                   help='explicit independent internal/release/world event scheduling; same learned selector')
    p.add_argument('--adaptive-dynamics', action='store_true',
                   help='learn per-head timescales and invariant content coupling with bounded quiet sensitivities')
    p.add_argument('--adaptive-spectrum', action='store_true',
                   help='explicit actual-state admission of log-spaced clocks and fixed phase-ellipse scale')
    p.add_argument('--resume-output',action='store_true',
                   help='resume this output directory from its own last complete checkpoint')
    p.add_argument('--profile-first-event',action='store_true',
                   help='record actual end-to-end Python dispatch/wait costs for one real event')
    p.add_argument('--learning-audit-cycles',type=int,default=0,
                   help='observe this many original events, then remove all learning audit hooks')
    p.add_argument('--learning-audit-output',type=Path,
                   help='new JSON report for bounded per-layer learning observation')
    p.add_argument('--checkpoint-compression-level',type=int,choices=range(0,10),
                   help='explicit lossless compression policy; 0 favors commit speed')
    p.add_argument('--checkpoint-compressor', choices=('gzip', 'isal'),
                   help='saved lossless execution policy; isal uses bounded parallel compression')
    p.add_argument('--checkpoint-compression-workers', type=int, choices=(1,2,4),
                   help='explicit isal worker count; inherited on restore')
    p.add_argument('--checkpoint-every', type=int, default=1,
                   help='complete snapshot cadence; text-only durable event recovery between snapshots')
    p.add_argument('--optimizer-policy', choices=('adamw','budgeted'),
                   help='explicit boundary adoption; otherwise inherit the saved optimizer')
    p.add_argument('--repair-gain-units', action='store_true',
                   help='explicit repair of raw-gain/wood/critic units; preserve inherited weights and moments')
    p.add_argument('--ambient-policy', choices=('reply','quiet'),
                   help='explicit event-boundary migration of inherited ambient navigation')
    p.add_argument('--world-source', choices=('mother', 'text'),
                   help='explicit external boundary migration; text disables every caregiver API call')
    p.add_argument('--text-navigation', choices=('legacy', 'stable_context_v1'),
                   help='explicit stable world-address migration; otherwise inherit the saved rule')
    p.add_argument('--natural-text-manifest', type=Path,
                   help='append-only natural-text supply; preserve all previous byte addresses')
    p.add_argument('--mother-policy', choices=('legacy', 'responsive', 'responsive_v2', 'responsive_v3'),
                   help='explicit event-boundary caregiver strategy migration')
    p.add_argument('--shared-cache', type=Path, help='bounded shared actual-reply audit cache')
    p.add_argument('--mother-context', choices=('segmented', 'dsh'),
                   help='explicit caregiver-context migration; otherwise inherit the checkpoint')
    p.add_argument('--mother-harness-url', help='loopback URL of the official DSH mother service')
    p.add_argument('--mother-session', help='independent DSH branch identity; otherwise inherit it')
    p.add_argument('--life-card', type=Path,
                   help='explicit saved-snapshot card path under data/harness/life; DSH defaults to a branch card')
    p.add_argument('--fused-decode-step', action='store_true',
                   help='FP32 five-kernel original one-byte GDN execution; inherit on restore')
    p.add_argument('--native-replay-batch-size', type=int, choices=(1,2,4,8),
                   help='bounded private read-only trajectories; preserve each versioned mask')
    p.add_argument('--reuse-objective-logits', action='store_true',
                   help='reuse actual objective logits for arrived-language factual use')
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
    p.add_argument('--shared-relation-basis',action='store_true',
                   help='explicit shared four/five-coordinate causal mechanism migration')
    p.add_argument('--joint-relation-update',action='store_true',
                   help='joint scope U, persistent variable dependencies and legal output reconciliation')
    p.add_argument('--relation-reference-write-cap', type=int, choices=(6, 7, 8),
                   help='private quota for real relation reference writes only')
    p.add_argument('--reference-use', action='store_true',
                   help='retain shared references by their later predictive use')
    p.add_argument('--unified-relation-evidence',action='store_true',
                   help='explicit revision 3: same stable facts for prediction, organization and commit')
    p.add_argument('--qualified-delayed-evidence',action='store_true',
                   help='explicit revision 4: delayed target exclusion by committed fact identity')
    return p


def prepare_world(engine, config, args, provider_state, is_mother, start_cycle, source_sha, fused, provider_factory=None):
    """Restore the selected boundary; text mode never constructs an API provider."""
    world_source = getattr(args, 'world_source', None) or config.get('world_source', 'mother')
    if world_source == 'text':
        from scripts.ib_local.natural_text_world import NaturalTextWorld, append_source
        manifest_path = getattr(args, 'natural_text_manifest', None)
        if manifest_path is not None and config.get('natural_text_corpus') is not None:
            supplied = json.loads(manifest_path.read_text(encoding='utf-8'))
            if supplied['train_sha256'] != config['natural_text_corpus']['train_sha256']:
                raise ValueError('cannot replace an inherited natural-text source')
        elif manifest_path is not None:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            manifest['prefix_bytes'] = len(engine.world.data)
            engine.world = append_source(engine.world, manifest)
            provider_state = {**provider_state, **engine.world.checkpoint_state_dict()}
            config['natural_text_corpus'] = manifest
        old_navigation = config.get('text_navigation_policy',
            provider_state.get('text_navigation_policy', 'legacy'))
        navigation = getattr(args, 'text_navigation', None) or old_navigation
        if navigation != old_navigation:
            config.setdefault('text_navigation_migrations', []).append(dict(
                from_policy=old_navigation, to_policy=navigation, at_cycle=start_cycle,
                source_sha256=source_sha,
                retained='same body, optimizer, memories, pending participation and world cursor',
                meaning='world law changes at this observed boundary; past observations are not rewritten'))
        config['text_navigation_policy'] = navigation
        engine.world = NaturalTextWorld(engine.world, provider_state,
                                       config.get('natural_text_corpus'), navigation_policy=navigation)
        engine.episodic_wood_mode = args.mode
        previous = config.get('world_source', 'mother')
        config.update(world_source='text', mother_fused=fused)
        if previous != 'text':
            config['world_source_migration'] = dict(from_source=previous, to_source='text',
                at_cycle=start_cycle, source_sha256=source_sha,
                retained='same parameters, Adam, S/P, memory, causal archive, pending credit and mother history',
                boundary='real suffix-navigation text response; no API, no synthetic caregiver or SFT')
        return config.get('mother_context_harness')
    if world_source != 'mother':
        raise ValueError('unknown world source')
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
    config['world_source'] = 'mother'
    return context


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
    metadata_path=args.checkpoint.parent/'last.metadata.json'
    metadata=json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else None
    use_metadata=(metadata is not None and metadata.get('schema')==1
                  and metadata.get('checkpoint_sha256')==source_sha)
    if use_metadata:
        config,world_state,start_cycle=metadata['config'],metadata['mother_world'],int(metadata['cycle'])
        saved=None
    else:
        from scripts.ib_local.compressed_checkpoint import checkpoint_stream
        with checkpoint_stream(args.checkpoint) as f:
            saved = torch.load(f, map_location='cpu', weights_only=False)
        config, world_state, start_cycle = saved['config'], saved['world'], int(saved['cycle'])
    config = dict(config)
    if getattr(args,'checkpoint_compression_level',None) is not None:
        config['checkpoint_compression_level']=args.checkpoint_compression_level
    if (getattr(args, 'checkpoint_compressor', None) is not None or
            getattr(args, 'checkpoint_compression_workers', None) is not None or
            'checkpoint_compressor' in config):
        from scripts.ib_local.mother_checkpoint import configure_compression
        configure_compression(config, backend=getattr(args, 'checkpoint_compressor', None),
                              workers=getattr(args, 'checkpoint_compression_workers', None))
    is_mother = 'mother_transcript' in world_state
    if is_mother and config.get('mother_mode') != args.mode:
        raise ValueError('cannot silently change an existing mother individual mode')
    provider_state = {k:(v.cpu().numpy() if torch.is_tensor(v) else v) for k,v in world_state.items()}
    if not use_metadata and (config.get('body_growth') is not None or config.get('native_body') is not None):
        cached=dict(schema=1,checkpoint_sha256=source_sha,cycle=start_cycle,
            config=saved['config'],mother_world={k:v for k,v in provider_state.items() if k.startswith('mother_')})
        temporary=metadata_path.with_suffix('.tmp.json')
        temporary.write_text(json.dumps(cached,ensure_ascii=False)+'\n',encoding='utf-8')
        temporary.replace(metadata_path)
    del saved, world_state
    import gc
    gc.collect()
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
    if use_metadata:
        provider_state={**engine.world.checkpoint_state_dict(),**provider_state}
    if args.question_parent_carry and config.get('question_parent_carry') is None:
        if not (getattr(engine, 'mother_shadow_enabled', False) and
                getattr(engine, 'joint_question_update', False)):
            raise ValueError('question parent carry requires the existing joint mother-shadow body')
        engine.question_parent_carry = True
        config['question_parent_carry'] = dict(schema=1, migrated_at_cycle=start_cycle,
            scope='retain the actual versioned parent mother observation in the next bounded receipt')
    if engine.__class__.__name__ not in ('DirectReadGainEngine', 'InternalRolloutEngine', 'MomentumEngine', 'MomentumWoodEngine', 'RelationWoodEngine', 'QuestionContextEngine', 'ShadowPreviewEngine', 'MotherBoundaryEngine', 'PersistentQuestionEngine', 'CausalProgramEngine', 'LearnedCouplingEngine', 'JointCycleEngine', 'JointLearnedCycleEngine', 'GenerativeCycleEngine', 'GenerativeLearnedCycleEngine', 'PastPresentEngine'):
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
    if getattr(args, 'repair_gain_units', False) and config.get('gain_units') is None:
        from scripts.ib_local.navigation_gain_units import configure_gain_units
        configure_gain_units(engine)
        config['gain_units'] = dict(schema=1, repaired_at_cycle=start_cycle,
            critic_scale=100., wood_scale=10., raw_scale=1.,
            repair='TD bootstrap divided by critic scale; wood logits converted before critic precision',
            retained='all old weights, optimizer, calibration, real gains, history and pending participation',
            previous='legacy bootstrap used /10 with a 100*g critic; effective discount was 10*gamma')
    if args.shared_cache:
        args.shared_cache.parent.mkdir(parents=True, exist_ok=True)
    context = prepare_world(engine, config, args, provider_state, is_mother,
                            start_cycle, source_sha, fused, provider_factory)
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
    if config.get('shared_unknowns') is not None or getattr(args,'shared_unknowns',False):
        from scripts.ib_local.shared_unknown_bindings import configure_shared_unknowns
        configure_shared_unknowns(engine)
        if config.get('shared_unknowns') is None:
            config['shared_unknowns'] = dict(schema=1,migrated_at_cycle=start_cycle,
                capacity=engine.model._causal_archive.budgets['relations'],
                evidence_budget=engine.model.causal_relations.evidence_budget,
                identity='typed open port and exact known premises; explicit recall preserves the question',
                storage='same causal SQLite; raw premises and versioned real episode references',
                learning='same current encoder, typed mechanism, U and delayed actor; no new reward')
    if getattr(args,'shared_relation_basis',False) and config.get('shared_relation_basis') is None:
        from scripts.ib_local.shared_relation_basis import configure_shared_relation_basis
        receipt=configure_shared_relation_basis(engine,initialize=True)
        config['shared_relation_basis']=dict(schema=1,features=[4,5],
            migrated_at_cycle=start_cycle,initialization=receipt,
            semantics='learned common role coordinates; same bounded posterior/U throughout navigation',
            noise='learned pooled real-evidence prior, no local unit reset',
            retained='all original parameters and Adam, S/P, actual facts, world and pending participation')
        # Include the just-added shared parameters in the ordinary full-body
        # audit; existing parameter groups and optimizer moments are intact.
        named=list(engine.model.named_parameters())
        config['mother_parameter_audit'].update(
            total=sum(p.numel() for _,p in named),
            trainable=sum(p.numel() for _,p in named if p.requires_grad),
            trainable_in_optimizer=sum(p.numel() for group in engine.optimizer.param_groups for p in group['params'] if p.requires_grad),
            frozen=[name for name,p in named if not p.requires_grad])
    if (config.get('joint_relation_update') is not None or getattr(args,'joint_relation_update',False)
            or getattr(args,'unified_relation_evidence',False) or getattr(args,'qualified_delayed_evidence',False)):
        from scripts.ib_local.joint_relation_update import configure_joint_relation_update
        spec=config.get('joint_relation_update')
        inherited_revision=(spec or {}).get('implementation_revision',1)
        revision=(4 if getattr(args,'qualified_delayed_evidence',False) else
                  max(3,inherited_revision) if getattr(args,'unified_relation_evidence',False) else
                  max(2,inherited_revision) if getattr(args,'joint_relation_update',False) else inherited_revision)
        configure_joint_relation_update(engine,revision=revision)
        if config.get('joint_relation_update') is None:
            from scripts.ib_local.joint_relation_update import seed_actual_uses
            journal=args.checkpoint.parent/'metrics.jsonl'
            rows=(json.loads(s) for s in journal.read_text(encoding='utf-8').splitlines() if s.strip())
            dependencies=seed_actual_uses(engine,rows)
            config['joint_relation_update']=dict(schema=1,migrated_at_cycle=start_cycle,
                dependencies=dependencies,
                variables='versioned shared unknown -> bounded dependent relation uses',
                organization='same shared coordinates and noise prior as prediction',
                overlap='normalized precision quadratic solve at the same bound decision port')
        old_revision=config['joint_relation_update'].get('implementation_revision',1)
        if old_revision!=revision:
            config['joint_relation_update']['implementation_migration']=dict(
                source_cycle=start_cycle,previous=old_revision,current=revision,
                change=('delayed target exclusion follows committed fact identity; recycled addresses keep old participants'
                        if revision>=4 else 'canonical fact IDs for prediction, organization, replay and persistent commit'
                        if revision>=3 else 'accepted sharing propagates variable uses; D-optimal common role retention'))
        config['joint_relation_update']['implementation_revision']=revision
    reference_write_cap = getattr(args, 'relation_reference_write_cap', None)
    if reference_write_cap is not None or 'relation_reference_write_cap' in config:
        from scripts.ib_local.joint_relation_update import configure_reference_write_cap
        configure_reference_write_cap(engine, config, reference_write_cap)
    if getattr(args, 'content_question', False) and config.get('content_question') is None:
        from scripts.ib_local.question_language_port import configure_content_question
        configure_content_question(engine)
        config['content_question'] = dict(schema=1,migrated_at_cycle=start_cycle,
            transition='question-conditioned feature modulation before the existing GDN injection layer',
            scope='pending preparation and self-hearing; direct byte bias only on speech outputs',
            learning='same real response CE, latent losses and actual delayed draft policy credit',
            retained='all inherited weights, Adam, continuous S/P, facts, world and pending credit')
        named=list(engine.model.named_parameters())
        config['mother_parameter_audit'].update(
            total=sum(p.numel() for _,p in named),
            trainable=sum(p.numel() for _,p in named if p.requires_grad),
            trainable_in_optimizer=sum(p.numel() for group in engine.optimizer.param_groups for p in group['params'] if p.requires_grad))
    if getattr(args, 'grow_body', None) is not None:
        if config.get('body_growth') is not None:
            raise ValueError('this individual already has the declared growth')
        from scripts.ib_local.navigation_body_growth import configure_body_growth
        growth = json.loads(args.grow_body.read_text(encoding='utf-8'))
        spec = dict(schema=1, backbone=growth['backbone'], seed=111435,
                    source_cycle=start_cycle, feature_width=engine.model.memory.width,
                    mapping='duplicate heads/channels, split weights, identity depth, learned stable fact bridges')
        # Read-only numerical admission precedes the irreversible learning path.
        from scripts.ib_local.navigation_growth_admission import grow_with_checks
        receipt = grow_with_checks(engine, spec)
        config['body_growth'] = spec
        config['bfss_dynamics']['coupling'] = engine.bfss_coupling
        config['neural_parameters'] = receipt['parameters']
        config['mother_parameter_audit'] = dict(total=receipt['parameters'],
            trainable=receipt['parameters'], trainable_in_optimizer=sum(
                p.numel() for g in engine.optimizer.param_groups for p in g['params']),
            frozen=[], trainable_missing_from_optimizer=[])
        (args.output/'growth_admission.json').write_text(
            json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if getattr(args,'variable_expression',False):
        if config.get('variable_expression') is not None:
            raise ValueError('variable expression already inherited')
        from scripts.ib_local.variable_expression import configure_variable_expression
        configure_variable_expression(engine)
        archive=getattr(engine.model,'_causal_archive',None)
        attachment=None
        if archive is not None:
            from scripts.ib_local.causal_archive import grow_utterance_attachment
            attachment=grow_utterance_attachment(archive)
        config['variable_expression']=dict(schema=1,cap=128,
            migrated_at_cycle=start_cycle,stopping='learned content and remaining-budget handover',
            reward='inherited actual source learning progress; no length reward',attachment=attachment)
        total=sum(p.numel() for p in engine.model.parameters())
        config['mother_parameter_audit'].update(total=total,trainable=total,
            trainable_in_optimizer=sum(p.numel() for g in engine.optimizer.param_groups for p in g['params']))
    if getattr(args, 'past_present_coevolution', None) is not None:
        if config.get('past_present_coevolution') is not None:
            raise ValueError('co-evolution already inherited; do not migrate twice')
        from scripts.ib_local.past_present_coevolution import configure_coevolution
        spec = json.loads(args.past_present_coevolution.read_text(encoding='utf-8'))
        receipt = configure_coevolution(engine, spec)
        config['past_present_coevolution'] = dict(spec, migrated_at_cycle=start_cycle,
            source_checkpoint=str(args.checkpoint),
            semantics='versioned actual anchors; causal contextual field through all depth/time cells',
            learning='same actual language, latent, outcome and delayed policy objectives',
            retained='physical S/P, trained parameters, Adam, evidence, world and pending participation')
        config['neural_parameters'] = receipt['total_parameters']
        config['mother_parameter_audit'].update(total=receipt['total_parameters'],
            trainable=receipt['total_parameters'], trainable_in_optimizer=sum(
                p.numel() for g in engine.optimizer.param_groups for p in g['params']))
        (args.output/'coevolution_admission.json').write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    from scripts.ib_local.optimizer_policy import configure as configure_optimizer
    configure_optimizer(engine, config, getattr(args, 'optimizer_policy', None), start_cycle)
    if getattr(args, 'adaptive_dynamics', False) and config.get('adaptive_dynamics') is None:
        from scripts.ib_local.adaptive_hamiltonian import configure as configure_adaptive
        configure_adaptive(engine)
        config['adaptive_dynamics'] = dict(schema=1, adopted_at_cycle=start_cycle,
            features='per-head positive frequency, boundary-context gain, positive strength, invariant pair attention',
            credit='exact local quiet sensitivities to first true response and actual action-score replay',
            scope='no lifetime weight Jacobian; fixed numerical dt; same real losses')
        config['neural_parameters'] = sum(p.numel() for p in engine.model.parameters())
        config['mother_parameter_audit'].update(total=config['neural_parameters'],
            trainable=config['neural_parameters'], trainable_in_optimizer=sum(
                p.numel() for g in engine.optimizer.param_groups for p in g['params']))
    spectrum_changed = False
    if getattr(args, 'adaptive_spectrum', False):
        if config.get('adaptive_dynamics') is None or config.get('adaptive_frequency_spectrum') is not None:
            raise ValueError('spectrum requires existing adaptive dynamics and a single explicit migration')
        journal_path = args.output/'events.autonomous.pending.jsonl'
        if journal_path.exists():
            from scripts.ib_local.event_journal import complete_jsonl
            records = complete_jsonl(journal_path)
            if len(records) != 1 or records[0]['base_cycle'] != engine.autonomous_event_clock:
                raise ValueError('recover and save all pending events before changing the spectrum')
        from scripts.ib_local.adaptive_spectrum import admit
        receipt = admit(engine)
        config['adaptive_frequency_spectrum'] = dict(receipt, adopted_at_cycle=start_cycle,
            source_sha256=source_sha)
        (args.output/'spectrum_admission.json').write_text(json.dumps(
            config['adaptive_frequency_spectrum'], indent=2)+'\n', encoding='utf-8')
        spectrum_changed = True
    if getattr(args, 'autonomous_events', False) or config.get('autonomous_events') is not None:
        from scripts.ib_local.autonomous_cycle import configure as configure_autonomous
        configure_autonomous(engine)
        config.setdefault('autonomous_events', dict(schema=1, adopted_at_cycle=start_cycle,
            scheduling='independent internal, release, actual boundary events',
            credit='bounded existing-selector score; same real delayed actor settlement',
            no_input='no new bytes, facts, external clock, gain or optimizer update',
            forced_world_cadence=False))
    if 'identity_preview_revision' in config:
        engine.model.identity_preview_revision = int(config['identity_preview_revision'])
    if config.get('causal_use_value') is not None:
        from scripts.ib_local.causal_use_value import configure as configure_causal_use
        configure_causal_use(engine, config['causal_use_value']['arm'])
    if config.get('epistemic_value', {}).get('schema') in (1,2,3,4,5,6) or getattr(engine,'epistemic_value_schema',0) in (1,2,3,4,5,6):
        from scripts.ib_local.epistemic_value_reference import configure as configure_epistemic_value
        configure_epistemic_value(engine,schema=config.get('epistemic_value',{}).get('schema',
            getattr(engine,'epistemic_value_schema',1)),
            word_execution=config.get('epistemic_value',{}).get('word_execution','historical'))
    engine.native_word_runtime = config.get('native_word_runtime', {}).get('mode', 'serial')
    engine.model._preview_batch_enabled = (config.get('native_word_runtime', {}).get('schema', 1)>=2
        or getattr(engine, 'epistemic_value_schema', 0)>=6)
    if engine.native_word_runtime not in ('serial', 'batch'):
        raise ValueError('unknown native word execution runtime')
    if config.get('successor_quadrature') is not None:
        from scripts.ib_local.native_residual_quadrature import configure as configure_quadrature
        configure_quadrature(engine, config['successor_quadrature'])
    if config.get('native_amortization') is not None:
        from scripts.ib_local.native_amortized_decision import configure as configure_amortization
        configure_amortization(engine, config['native_amortization'])
    if config.get('bounded_reading_option') is not None:
        from scripts.ib_local.bounded_reading_option import configure as configure_reading
        spec = config['bounded_reading_option']
        configure_reading(engine, max_blocks=spec['max_blocks'], surprise_excess=spec['surprise_excess'],
                          hippocampal_anchor_scale=spec.get('hippocampal_anchor_scale', 0.0))
    if config.get('persistent_rhythm') is not None:
        from scripts.ib_local.persistent_rhythm import configure as configure_rhythm
        configure_rhythm(engine, interleave=config['persistent_rhythm']['interleave'])
    if config.get('event_credit') is not None:
        from scripts.ib_local.identified_event_credit import configure as configure_event_credit
        configure_event_credit(engine, config['event_credit']['mode'])
    if config.get('native_retention_feedback') is not None:
        from scripts.ib_local.native_retention_feedback import configure as configure_retention_feedback
        configure_retention_feedback(engine, config['native_retention_feedback']['mode'])
    if config.get('relation_retention_use') is not None:
        from scripts.ib_local.relation_retention_use import configure as configure_relation_use
        configure_relation_use(engine,config['relation_retention_use']['mode'],
                               schema=int(config['relation_retention_use'].get('schema',1)))
    if getattr(args, 'reference_use', False) or config.get('reference_use') == 1:
        from scripts.ib_local.reference_use import configure as configure_reference_use
        configure_reference_use(engine, config)
    if config.get('retention_scope_prior') is not None:
        from scripts.ib_local.retention_scope_prior import configure as configure_scope_prior
        configure_scope_prior(engine, config['retention_scope_prior']['mode'])
    if getattr(args, 'speech_only', False):
        config['speech_only'] = True
    if config.get('speech_only'):
        if getattr(engine, 'variable_expression_schema', None) != 1:
            raise ValueError('speech-only requires the unified variable expression path')
        engine.model._expression_speech_only = True
    from scripts.ib_local.continuation_schedule import configure_validation_schedule
    configure_validation_schedule(config, start_cycle, args.validate_every, args.validation_bytes)
    if (getattr(args, 'native_replay_batch_size', None) is not None or
            getattr(args, 'reuse_objective_logits', False) or 'native_replay_runtime' in config):
        from scripts.ib_local.native_replay_batch import configure as configure_replay
        configure_replay(engine, config, batch_size=getattr(args, 'native_replay_batch_size', None),
            reuse_actual=True if getattr(args, 'reuse_objective_logits', False) else None)
    if getattr(args, 'fused_decode_step', False) or config.get('decode_runtime', {}).get('fused_step', False):
        from scripts.ib_local.scoped_decision_graph import configure_fused_step
        configure_fused_step(engine, config)
    card = None
    checkpoint_sha = source_sha if resuming else None
    if not resuming or spectrum_changed:
        checkpoint_sha = save(engine,args.output,start_cycle,config)
        (args.output/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        if spectrum_changed:
            from scripts.ib_local.event_journal import EventJournal, execution_digest
            # Migration admitted only a complete empty journal above. Replace
            # its anchor after the new complete checkpoint, never drop events.
            (args.output/'events.autonomous.pending.jsonl').unlink(missing_ok=True)
            EventJournal(args.output, cycle=engine.autonomous_event_clock,
                checkpoint_sha=checkpoint_sha, code_sha=execution_digest(ROOT), config=config,
                filename='events.autonomous.pending.jsonl', config_key='autonomous_event_journal')
        if life_card is not None:
            card = export_live(engine, checkpoint=args.output/'last.pt.gz', config=config,
                        cycle=start_cycle, output=life_card['path'], branch_id=life_branch,
                        committed_sha=checkpoint_sha)
    return SimpleNamespace(engine=engine, config=config, cycle=start_cycle,
        output=args.output, life_card=life_card, life_branch=life_branch,
        life_card_data=card, source=source, source_sha=source_sha,
        checkpoint_sha=checkpoint_sha)


def main():
    from scripts.ib_local.continuation_schedule import validation_due
    args = continuation_parser().parse_args()
    loaded = prepare_continuation(args,resume_output=args.resume_output)
    engine, config, start_cycle = loaded.engine, loaded.config, loaded.cycle
    if torch.device(engine.device).type == 'cuda':
        configure_runtime_storage(engine, args.output/'activations', **storage_options(args))
    life_card, life_branch = loaded.life_card, loaded.life_branch
    source, source_sha = loaded.source, loaded.source_sha
    if args.admit_only:
        print(json.dumps(dict(status='admitted_without_training',cycle=start_cycle,
            parameters=sum(p.numel() for p in engine.model.parameters()),
            checkpoint=str(args.output/'last.pt.gz'),world_calls=0,optimizer_updates=0)),flush=True)
        return
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
        text_manifest=config.get('natural_text_corpus')
        fineweb=None
        if text_manifest is not None:
            import hashlib
            raw_text=Path(text_manifest['validation_path']).read_bytes()
            if hashlib.sha256(raw_text).hexdigest()!=text_manifest['validation_sha256']:
                raise ValueError('natural text validation source changed')
            fineweb=np.frombuffer(raw_text,dtype=np.uint8)[:args.validation_bytes+1]
            if len(fineweb)!=args.validation_bytes+1:
                raise ValueError('natural text validation too short')
        def evaluation():
            contribution = {}
            if config.get('world_source') == 'text':
                from scripts.audit_language_contribution import language_contribution
                contribution['owt'] = language_contribution(engine.model,engine.table,owt)
            scores = dict(owt_nll=(contribution['owt']['full_nll'] if contribution else
                                  validate(engine.model,engine.table,owt)),
                        mother_nll=validate(engine.model,engine.table,mom),
                        bytes_each=args.validation_bytes,
                        scope='fixed cold sidecars; no gradient or live-state reset; no navigation during sidecar')
            if fineweb is not None:
                if contribution:
                    contribution['fineweb'] = language_contribution(engine.model,engine.table,fineweb)
                    scores['fineweb_nll'] = contribution['fineweb']['full_nll']
                else:
                    scores['fineweb_nll']=validate(engine.model,engine.table,fineweb)
            if contribution: scores['language_contribution'] = contribution
            return scores
        initial_name=(f'resume_validation_{start_cycle}.json' if args.resume_output
                      else 'initial_validation.json')
        (args.output/initial_name).write_text(json.dumps(evaluation(),indent=2),encoding='utf-8')
    if config.get('autonomous_events') is not None:
        from scripts.ib_local.autonomous_runtime import run as run_autonomous
        return run_autonomous(args, loaded, evaluation, ROOT)
    start = time.perf_counter()
    completed = 0
    actual_updates=[0]
    original_step=engine.optimizer.step
    def counted_step(*a,**kw):
        result=original_step(*a,**kw)
        actual_updates[0]+=1
        return result
    engine.optimizer.step=counted_step
    learning_audit = None
    if args.learning_audit_cycles:
        if args.learning_audit_cycles < 1 or args.learning_audit_output is None:
            raise ValueError('positive learning audit budget and explicit output required')
        from scripts.ib_local.learning_dynamics_audit import LearningDynamicsAudit
        learning_audit = LearningDynamicsAudit(engine.model,engine.optimizer,
            args.learning_audit_output,args.learning_audit_cycles)
    if config.get('body_growth') is not None and not hasattr(engine,'growth_optimizer_updates'):
        prior_rows=[]
        prior_path=args.output/'metrics.jsonl'
        if args.resume_output and prior_path.exists():
            prior_rows=[json.loads(line) for line in prior_path.read_text(encoding='utf-8').splitlines() if line]
        engine.growth_optimizer_updates=sum(r.get('updates',0) for r in prior_rows if r.get('cycle',0)<=start_cycle)
        engine.growth_external_bytes=sum(r.get('external_bytes',0) for r in prior_rows if r.get('cycle',0)<=start_cycle)
    if config.get('native_body') is not None:
        from scripts.ib_local.native_navigation_body import restore_birth_counters, record_birth_event
        restore_birth_counters(engine,start_cycle,args.output/'metrics.jsonl')
    if args.checkpoint_every < 1:
        raise ValueError('checkpoint cadence must be positive')
    journal = None
    if args.checkpoint_every > 1 or (args.output/'events.pending.jsonl').exists():
        if config.get('world_source') != 'text':
            raise ValueError('event journal recovery requires the deterministic natural-text world')
        from scripts.ib_local.event_journal import EventJournal, execution_digest, rng_digest, complete_jsonl
        # A new output has already published its OWN complete snapshot. Its
        # config/branch/optimizer metadata can differ from the ancestor file.
        # Bind recovery to that published file, not the ancestor's SHA.
        anchor_sha = getattr(loaded,'checkpoint_sha',source_sha)
        if anchor_sha is None: anchor_sha = sha256_file(args.output/'last.pt.gz')
        journal = EventJournal(args.output, cycle=start_cycle, checkpoint_sha=anchor_sha,
                               code_sha=execution_digest(ROOT), config=config)
        if len(journal.pending) > args.cycles:
            raise ValueError('requested continuation ends before recorded pending events')
        # Recovery is execution of the already recorded observations, before
        # any new world action. Never silently skip a failed receipt.
        metrics = complete_jsonl(args.output/'metrics.jsonl')
        last_logged = max((r.get('cycle', 0) for r in metrics), default=start_cycle)
        for item in list(journal.pending):
            before = rng_digest()
            updates = actual_updates[0]
            fallback_before=getattr(engine, '_batched_replay_fallbacks', 0)
            with event_storage(engine):
                row = engine.step(torch.randn(engine.model.encoder.out_features,32,device=engine.device))
            from scripts.ib_local.variable_expression import release_fallback_cache
            replay_cache_release=release_fallback_cache(engine,fallback_before)
            row['updates'] = actual_updates[0]-updates
            if config.get('native_body') is not None:
                record_birth_event(engine, row)
            if config.get('body_growth') is not None:
                engine.growth_optimizer_updates += row['updates']
                engine.growth_external_bytes += row['external_bytes']
            journal.verify(item, row, before, rng_digest())
            if item['cycle'] > last_logged:
                restored_row = dict(item['row'], cycle=item['cycle'], recovered_from_event_journal=True)
                with (args.output/'metrics.jsonl').open('a', encoding='utf-8') as history:
                    history.write(json.dumps(restored_row,ensure_ascii=False)+'\n')
                    history.flush()
            completed += 1
        if completed:
            journal.bind(config, start_cycle+completed)
            committed_sha = save(engine, args.output, start_cycle+completed, config)
            journal.reset(start_cycle+completed, committed_sha)
            if life_card is not None:
                export_live(engine, checkpoint=args.output/'last.pt.gz', config=config,
                    cycle=start_cycle+completed, row=row, output=life_card['path'],
                    branch_id=life_branch, committed_sha=committed_sha)
    with (args.output/'metrics.jsonl').open('a' if args.resume_output else 'w',encoding='utf-8') as log:
        last_saved_cycle = start_cycle+completed
        last_row = None
        for i in range(completed, args.cycles):
            if args.stop_file is not None and args.stop_file.exists():
                break
            before_updates=actual_updates[0]
            event_start=time.perf_counter()
            rng_before = rng_digest() if journal is not None else None
            profiler=None
            if args.profile_first_event and i==0:
                import cProfile
                profiler=cProfile.Profile();profiler.enable()
            fallback_before=getattr(engine, '_batched_replay_fallbacks', 0)
            with event_storage(engine):
                row = engine.step(torch.randn(engine.model.encoder.out_features,32,device=engine.device))
            from scripts.ib_local.variable_expression import release_fallback_cache
            replay_cache_release=release_fallback_cache(engine,fallback_before)
            if profiler is not None:
                import pstats
                profiler.disable()
                profiler.dump_stats(str(args.output/'first_event.prof'))
                with (args.output/'first_event_profile.txt').open('w',encoding='utf-8') as profile_log:
                    pstats.Stats(profiler,stream=profile_log).sort_stats('cumulative').print_stats(65)
            event_seconds=time.perf_counter()-event_start
            row['updates']=actual_updates[0]-before_updates
            row['cumulative_optimizer_updates']=actual_updates[0]
            if config.get('native_body') is not None:
                record_birth_event(engine,row)
            if config.get('body_growth') is not None:
                engine.growth_optimizer_updates+=row['updates']
                engine.growth_external_bytes+=row['external_bytes']
                row['since_growth_updates']=engine.growth_optimizer_updates
                row['since_growth_external_bytes']=engine.growth_external_bytes
            cycle = start_cycle+i+1
            if journal is not None:
                journal.append(cycle, row, rng_before, rng_digest())
            # Every result has a durable receipt. Complete checkpoints and
            # hash-bound life cards share one publication boundary.
            save_start=time.perf_counter()
            due = (cycle-last_saved_cycle >= args.checkpoint_every or i+1 == args.cycles or
                   (args.stop_file is not None and args.stop_file.exists()) or
                   (evaluation is not None and validation_due(cycle, config['validation_schedule'])))
            committed_sha = None
            if due:
                if journal is not None: journal.bind(config, cycle)
                committed_sha = save(engine,args.output,cycle,config)
                last_saved_cycle = cycle
                if journal is not None: journal.reset(cycle, committed_sha)
            row['runtime']=dict(event_seconds=event_seconds,
                checkpoint_seconds=time.perf_counter()-save_start,
                complete_checkpoint_cycle=last_saved_cycle, durable_event_cycle=cycle,
                external_bytes_per_event_second=row.get('external_bytes',0)/event_seconds,
                replay_fallback_cache_release=replay_cache_release,
                cuda_allocated_mib=torch.cuda.memory_allocated()/2**20 if engine.device.type=='cuda' else None,
                cuda_reserved_mib=torch.cuda.memory_reserved()/2**20 if engine.device.type=='cuda' else None,
                cuda_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if engine.device.type=='cuda' else None,
                cuda_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if engine.device.type=='cuda' else None)
            if due:
                from scripts.ib_local.optimizer_policy import diagnostics
                row['optimizer_plasticity'] = diagnostics(engine)
            if life_card is not None and due:
                export_live(engine, checkpoint=args.output/'last.pt.gz', config=config,
                            cycle=cycle, row=row, output=life_card['path'], branch_id=life_branch,
                            committed_sha=committed_sha)
            if evaluation is not None and validation_due(cycle, config['validation_schedule'],
                                                         final=i+1==args.cycles):
                row['fixed_validation']=evaluation()
            row.update(cycle=cycle, mode=args.mode, elapsed_seconds=time.perf_counter()-start)
            log.write(json.dumps(row,ensure_ascii=False)+'\n')
            log.flush()
            if learning_audit is not None and learning_audit.active:
                learning_audit.finish_event(cycle,row)
            completed += 1
            last_row = row
            print(json.dumps({k:row.get(k) for k in
                ('cycle','choice','self_bytes','metal_g','wood_target','water_credit_records','world')},
                ensure_ascii=True),flush=True)
        # A STOP arriving between events still publishes the latest complete
        # organism; not just its most recent periodic snapshot.
        if last_saved_cycle < start_cycle+completed:
            cycle = start_cycle+completed
            if journal is not None: journal.bind(config, cycle)
            committed_sha = save(engine,args.output,cycle,config)
            if journal is not None: journal.reset(cycle, committed_sha)
            if life_card is not None:
                export_live(engine, checkpoint=args.output/'last.pt.gz', config=config,
                    cycle=cycle, row=last_row, output=life_card['path'], branch_id=life_branch,
                    committed_sha=committed_sha)
    if learning_audit is not None:
        learning_audit.close()
    (args.output/'completed.json').write_text(json.dumps(dict(status='complete' if completed==args.cycles else 'stopped',
        cycles=completed,requested_cycles=args.cycles,
        source_sha256=source_sha, final_cycle=start_cycle+completed,source=source),
        ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
