"""Isolated complete continuations comparing normal and simulated broken restore.

--cycles counts ADDITIONAL external rounds per invocation. Legacy mode only
simulates the old missing runtime flag; it is never a main-body candidate.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                    encoding='utf8')


def assert_replayed_event(row, old, index):
    """Compare the serialized event contract, including nested tuple records.

    Live retention moments are tuples; JSON restores those sequences as lists.
    Normalize representation only, keeping every scalar and identity exact.
    """
    for key in ('autonomous_event', 'choice', 'utterance_hex', 'external_bytes',
                'metal_g', 'shared_unknown'):
        actual = json.loads(json.dumps(row.get(key), ensure_ascii=False, allow_nan=False))
        assert actual == old.get(key), (index, key)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    from scripts.native_policy_activation_storage import add_storage_arguments, storage_options
    add_storage_arguments(p)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--cycles', type=int, default=32)
    p.add_argument('--max-life-seconds', type=float, default=None,
                   help='optional wall-clock budget; stops cleanly at the first external cycle reaching this duration')
    p.add_argument('--speech-only', action='store_true')
    p.add_argument('--evaluate-every', type=int, default=8)
    p.add_argument('--resume-work', action='store_true')
    p.add_argument('--legacy-restore', action='store_true')
    p.add_argument('--epistemic-value', action='store_true',
                   help='experimental continuation carries revised R and next Q')
    p.add_argument('--native-decision',action='store_true',
                   help='schema3 predicted full boundary, native question and actor selectors')
    p.add_argument('--native-word-execution',action='store_true',
                   help='schema3 native intention/word executor at the current physical anchor')
    p.add_argument('--batch-native-words', action='store_true',
                   help='same native word endpoints in bounded original-body trajectory batches')
    p.add_argument('--batch-preview-scores', action='store_true',
                   help='runtime revision2: batch original schema5 finite-U scores')
    p.add_argument('--successor-distribution',action='store_true',
                   help='schema4 coordinated successor distribution through native addressing and learning')
    p.add_argument('--joint-successor',action='store_true',
                   help='schema5 coherent output scatter and native address-cell integration')
    p.add_argument('--joint-response', action='store_true',
                   help='schema6 same joint response law at every preview depth and in actual learning')
    p.add_argument('--residual-successor', action='store_true',
                   help='schema5 original posterior, learned all-point baseline and native residuals')
    p.add_argument('--amortized-decision', action='store_true',
                   help='learn the conditional native decision port with arrived-world calibration')
    p.add_argument('--latent-action-planner', action='store_true',
                   help='batch learned action endpoints and latent planning with native importance correction')
    p.add_argument('--internal-step-review', action='store_true',
                   help='only expand native references after a self-selected internal step')
    p.add_argument('--teacher-selection', choices=('novelty_diversity', 'competing_actions'), default='novelty_diversity')
    p.add_argument('--root-native-correction', choices=('observe', 'apply'),
                   help='record or replace already evaluated root successor contributions before actor choice')
    p.add_argument('--persistent-rhythm', choices=('port', 'coupled'),
                   help='persistent conditional programs, optionally interleaved with real learning')
    p.add_argument('--evaluation-material', type=Path,
                   help='fixed real old queries shared across a comparison')
    p.add_argument('--reading-option-blocks', type=int, choices=(1,2,4,8))
    p.add_argument('--reading-option-surprise-excess', type=float, default=0.5)
    p.add_argument('--phase-coupled-rhythm', action='store_true',
                   help='event-driven brainwave phase-coupled multi-rate Gamma/Theta/SWR rhythm')
    p.add_argument('--wiener-gate', action='store_true',
                   help='opt-in momentum Wiener cooling gate (S - w*E) / ((1 - w) * S) in BudgetedAdamW')
    p.add_argument('--split-hamiltonian-theta', action='store_true',
                   help='opt-in split AdaptiveHamiltonian.theta into 4 role-specific parameters')
    p.add_argument('--learnable-precision-scales', action='store_true',
                   help='opt-in learnable log-space relation_ridge and memory.temperature')
    p.add_argument('--ridge-lr', type=float, default=None,
                   help='optional explicit scalar learning rate for log_relation_ridge (weight_decay=0)')
    p.add_argument('--table-storage', type=Path, help='explicit local disk for rebuildable fast-table work copies')
    p.add_argument('--profile-runtime', action='store_true',
                   help='record actual preview call counts and inclusive/exclusive host wall times')
    p.add_argument('--epistemic-momentum',action='store_true',
                   help='same epistemic U with the known P condition carried into continuation')
    p.add_argument('--event-credit', choices=('compressed','identified'),
                   help='version-matched actual-event/compressed-prototype comparison')
    p.add_argument('--native-retention-feedback', choices=('off', 'observe', 'apply'))
    p.add_argument('--relation-retention-use',choices=('observe','apply'))
    p.add_argument('--retention-scope-prior', choices=('off', 'relation'))
    p.add_argument('--counterfactual-response-shadow', action='store_true', default=None,
                   help='log residual-transport predictions without affecting learning or choice')
    p.add_argument('--counterfactual-response-capsules', type=Path,
                   help='save same-chart observation capsules outside the live archive')
    p.add_argument('--world-activations', type=Path)
    p.add_argument('--policy-activations', type=Path,
                   help='temporary lossless backward storage; defaults to the isolated work directory')
    p.add_argument('--replay-prefix', type=Path)
    args = p.parse_args()
    activation_options = storage_options(args)
    if args.native_word_execution and not args.native_decision:
        p.error('--native-word-execution requires --native-decision')
    if args.batch_native_words and not args.native_word_execution:
        p.error('--batch-native-words requires --native-word-execution')
    if args.batch_preview_scores and not args.batch_native_words:
        p.error('--batch-preview-scores requires --batch-native-words')
    if args.successor_distribution and not (args.native_decision and args.native_word_execution):
        p.error('--successor-distribution requires native decision and word execution')
    if args.joint_successor and not args.successor_distribution:
        p.error('--joint-successor requires --successor-distribution')
    if args.joint_response and not args.joint_successor:
        p.error('--joint-response requires --joint-successor')
    if args.residual_successor and (not args.joint_successor or args.joint_response):
        p.error('--residual-successor requires original schema5 --joint-successor')
    if args.amortized_decision and (not args.joint_successor or args.joint_response or args.residual_successor):
        p.error('--amortized-decision requires original schema5 without residual-grid integration')
    if args.latent_action_planner and not args.amortized_decision:
        p.error('--latent-action-planner requires --amortized-decision')
    if args.root_native_correction and not (args.persistent_rhythm and args.latent_action_planner):
        p.error('--root-native-correction requires persistent conditional planning')
    if args.internal_step_review and not (args.persistent_rhythm == 'coupled' and args.latent_action_planner):
        p.error('--internal-step-review requires coupled persistent rhythm and latent action planning')
    if args.cycles < 1 or args.evaluate_every < 1:
        p.error('--cycles and --evaluate-every must be positive')
    args.work, args.checkpoint = args.work.resolve(), args.checkpoint.resolve()
    body = args.work/'body'
    checkpoint = body/'last.pt.gz' if args.resume_work else args.checkpoint
    if not args.resume_work and args.checkpoint.is_relative_to(args.work):
        p.error('new work must be isolated from the source checkpoint')
    args.work.mkdir(parents=True, exist_ok=args.resume_work)
    if args.legacy_restore and (args.epistemic_value or args.epistemic_momentum or args.native_decision):
        p.error('epistemic-value requires the identity-correct restored body')
    mode = ('native_decision_context' if args.native_decision else
            'epistemic_momentum_context' if args.epistemic_momentum else
            'relation_retention_'+args.relation_retention_use if args.relation_retention_use else
            'native_retention_'+args.native_retention_feedback if args.native_retention_feedback else
            'event_credit_'+args.event_credit if args.event_credit else
            'epistemic_value_context' if args.epistemic_value else
            'legacy_restore_simulation' if args.legacy_restore else 'normal_repaired_restore')
    result_path, status_path = args.work/'result.json', args.work/'status.json'
    previous = (json.loads(result_path.read_text(encoding='utf8'))
                if args.resume_work and result_path.exists() else {})
    if previous and previous['mode'] != mode:
        p.error('resume must retain the comparison mode')
    write(status_path, dict(status='loading', mode=mode, pid=os.getpid()))
    replay = ([] if args.replay_prefix is None else
              [json.loads(line) for line in args.replay_prefix.read_text(encoding='utf8').splitlines()])
    replayed = 0
    world_activations = args.world_activations or args.work/'world_activations'

    import torch
    from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation
    from scripts.ib_local.autonomous_cycle import dispatch
    from scripts.ib_local.mother_checkpoint import save
    from scripts.ib_local import navigation_causal_program as navigation
    from scripts.ib_local import causal_use_value as use_value
    from scripts.ib_local.causal_relation_program import Port, log_score
    from scripts.ib_local.relation_evidence import mechanism_evidence, evidence_key
    from scripts.ib_local.shared_unknown_bindings import _evidence
    from scripts.ib_local.native_navigation_body import record_birth_event
    from scripts.native_policy_activation_storage import activation_storage, policy_storage
    from scripts.run_native_causal_use_value import fixed_queries, evaluate
    torch.set_num_threads(1)
    opts = continuation_parser().parse_args(['--checkpoint', str(checkpoint),
        '--output', str(body), '--cycles', str(args.cycles), '--world-source', 'text',
        '--autonomous-events', '--optimizer-policy', 'budgeted',
        '--table-storage', str(args.table_storage or args.work/'table'), '--causal-storage', str(args.work/'causal')]
        + (['--resume-output'] if args.resume_work else [])
        + (['--speech-only'] if args.speech_only else []))
    loaded = prepare_continuation(opts, resume_output=args.resume_work)
    engine, model = loaded.engine, loaded.engine.model
    if args.counterfactual_response_shadow is not None:
        engine.counterfactual_response_shadow = args.counterfactual_response_shadow
    if args.counterfactual_response_capsules is not None:
        if not getattr(engine, 'counterfactual_response_shadow', False):
            p.error('counterfactual capsules require the response observer')
        engine.counterfactual_response_capsules = args.counterfactual_response_capsules.resolve()
    try:
        prepared_revision = getattr(model, 'identity_preview_revision', 0)
        if args.legacy_restore:
            # Diagnostic counterfactual ONLY: keep inherited config semantics.
            model.identity_preview_revision = 0
        else:
            assert prepared_revision == 1, 'normal prepare must restore identity preview revision 1'
        if args.epistemic_value or args.epistemic_momentum or args.native_decision:
            from scripts.ib_local.epistemic_value_reference import configure as configure_epistemic_value
            epi_schema=6 if args.joint_response else 5 if args.joint_successor else 4 if args.successor_distribution else 3 if args.native_decision else 2 if args.epistemic_momentum else 1
            word_execution='native_anchor' if args.native_word_execution else 'historical'
            if args.resume_work:
                saved=loaded.config.get('epistemic_value',{})
                if (saved.get('schema',1),saved.get('word_execution','historical'))!=(epi_schema,word_execution):
                    raise ValueError('resume must preserve successor law and native execution semantics')
            configure_epistemic_value(engine,schema=epi_schema,word_execution=word_execution)
            loaded.config['epistemic_value'] = dict(schema=epi_schema, adopted_at_cycle=loaded.cycle,
                word_execution=word_execution,
                state='predicted body boundary plus branch-updated R and next Q',
                evaluator='same finite-U, posterior and query score in encoded coordinates',
                momentum=('known P tensor with predicted-boundary read' if epi_schema>=3 else
                          'known current projected P carried for one prospective step' if epi_schema==2 else 'legacy omitted'))
        runtime = 'batch' if args.batch_native_words else 'serial'
        runtime_schema = 2 if args.batch_preview_scores else 1
        if args.resume_work and (loaded.config.get('native_word_runtime', {}).get('mode', 'serial'),
                loaded.config.get('native_word_runtime', {}).get('schema', 1)) != (runtime, runtime_schema):
            raise ValueError('resume must retain the original word execution runtime')
        engine.native_word_runtime = runtime
        loaded.config['native_word_runtime'] = dict(schema=runtime_schema, mode=runtime, batch_size=4)
        model._preview_batch_enabled = args.batch_preview_scores or getattr(engine, 'epistemic_value_schema', 0)>=6
        if args.residual_successor:
            from scripts.ib_local.native_residual_quadrature import configure as configure_quadrature
            integration = dict(schema=1, mode='residual', seed=191120)
            if args.resume_work and loaded.config.get('successor_quadrature') != integration:
                raise ValueError('resume must preserve residual integration and private seed')
            configure_quadrature(engine, integration)
            loaded.config['successor_quadrature'] = integration
        elif loaded.config.get('successor_quadrature', {}).get('mode') == 'residual':
            raise ValueError('residual integration body requires --residual-successor')
        if args.amortized_decision:
            from scripts.ib_local.native_amortized_decision import configure as configure_amortization
            specification = dict(schema=1, seed=201120)
            if args.latent_action_planner:
                specification['planner'] = 'action_batch'
                specification['correction'] = 'decision_budget'
                if args.internal_step_review:
                    specification['review_cadence'] = 'internal_step'
                if args.teacher_selection != 'novelty_diversity':
                    specification['teacher_selection'] = args.teacher_selection
                if args.root_native_correction:
                    specification['root_native_correction'] = args.root_native_correction
            if args.resume_work and loaded.config.get('native_amortization') != specification:
                raise ValueError('resume must preserve conditional native amortization')
            configure_amortization(engine, specification)
            loaded.config['native_amortization'] = specification
        elif loaded.config.get('native_amortization') is not None:
            raise ValueError('amortized native body requires --amortized-decision')
        if args.persistent_rhythm:
            from scripts.ib_local.persistent_rhythm import configure as configure_rhythm
            spec = dict(schema=1, interleave=args.persistent_rhythm == 'coupled')
            if args.resume_work and loaded.config.get('persistent_rhythm') != spec:
                raise ValueError('resume must retain persistent rhythm and interleaving semantics')
            configure_rhythm(engine, interleave=spec['interleave'])
            loaded.config['persistent_rhythm'] = spec
        elif loaded.config.get('persistent_rhythm') is not None:
            raise ValueError('persistent body requires --persistent-rhythm')
        if args.reading_option_blocks:
            from scripts.ib_local.bounded_reading_option import configure as configure_reading
            spec = dict(schema=1, max_blocks=args.reading_option_blocks,
                        surprise_excess=args.reading_option_surprise_excess)
            if args.resume_work and loaded.config.get('bounded_reading_option') != spec:
                raise ValueError('resume must preserve bounded reading semantics')
            configure_reading(engine, loaded.config, max_blocks=spec['max_blocks'],
                              surprise_excess=spec['surprise_excess'])
        if args.phase_coupled_rhythm:
            from scripts.ib_local.phase_coupled_rhythm import configure as configure_phase_rhythm, DEFAULT_PARAMS
            expected = dict(schema=1, **DEFAULT_PARAMS)
            if args.resume_work and loaded.config.get('phase_coupled_rhythm') != expected:
                raise ValueError('resume must preserve phase-coupled rhythm semantics')
            configure_phase_rhythm(engine, loaded.config)
        elif loaded.config.get('phase_coupled_rhythm') is not None:
            raise ValueError('phase-coupled body requires --phase-coupled-rhythm')
        if args.wiener_gate:
            from scripts.ib_local.optimizer_policy import configure_wiener_gate
            if args.resume_work and not loaded.config.get('optimizer_plasticity', {}).get('wiener_gate', False):
                raise ValueError('resume must preserve wiener_gate setting')
            configure_wiener_gate(engine, loaded.config, True)
        elif loaded.config.get('optimizer_plasticity', {}).get('wiener_gate', False):
            raise ValueError('wiener_gate body requires --wiener-gate')
        if args.split_hamiltonian_theta:
            from scripts.ib_local.adaptive_hamiltonian import migrate_split_theta
            if args.resume_work and loaded.config.get('adaptive_dynamics_split_theta') is None:
                raise ValueError('resume must preserve split_hamiltonian_theta setting')
            migrate_split_theta(engine)
            loaded.config.setdefault('adaptive_dynamics_split_theta', dict(
                schema=1, adopted_at_cycle=loaded.cycle,
                roles=['freq', 'context', 'coupling', 'pair']))
        elif loaded.config.get('adaptive_dynamics_split_theta') is not None:
            raise ValueError('split-theta body requires --split-hamiltonian-theta')
        if args.learnable_precision_scales:
            from scripts.ib_local.learnable_precision_scales import (
                configure as configure_scales, set_scale_learning_rates,
            )
            spec = dict(schema=1, ridge=True, temperature=True)
            if args.ridge_lr is not None:
                spec['ridge_lr'] = float(args.ridge_lr)
            if args.resume_work and loaded.config.get('learnable_precision_scales') != spec:
                raise ValueError('resume must preserve learnable_precision_scales setting')
            configure_scales(engine, spec, initialize=not args.resume_work)
            loaded.config['learnable_precision_scales'] = dict(
                loaded.config.get('learnable_precision_scales') or {}, schema=1, ridge=True, temperature=True)
            if args.ridge_lr is not None:
                set_scale_learning_rates(engine, loaded.config, ridge_lr=args.ridge_lr)
            elif loaded.config.get('learnable_precision_scales', {}).get('ridge_lr') is not None:
                raise ValueError('ridge-lr body requires --ridge-lr')
        elif loaded.config.get('learnable_precision_scales') is not None:
            raise ValueError('learnable-precision-scales body requires --learnable-precision-scales')
        elif args.ridge_lr is not None:
            raise ValueError('--ridge-lr requires --learnable-precision-scales')
        if args.event_credit:
            from scripts.ib_local.identified_event_credit import configure as configure_event_credit
            configure_event_credit(engine, args.event_credit)
            loaded.config['event_credit'] = dict(mode=args.event_credit, schema=1,
                adopted_at_cycle=loaded.cycle,
                reward='finite_event_targeted_use_v3',
                invariant='same event for finite U, payment, shared revision and delayed boundary')
        if args.native_retention_feedback:
            from scripts.ib_local.native_retention_feedback import configure as configure_retention_feedback
            old_mode = loaded.config.get('native_retention_feedback', {}).get('mode')
            if args.resume_work and old_mode != args.native_retention_feedback:
                raise ValueError('resume must retain the native retention feedback mode')
            configure_retention_feedback(engine, args.native_retention_feedback)
            loaded.config.setdefault('native_retention_feedback', dict(schema=1,
                mode=args.native_retention_feedback, adopted_at_cycle=loaded.cycle,
                signal='finite removal from coevolving workset and deep read on arrived external bytes',
                units='10*nats/byte', actor_reward_unchanged=True))
        if args.relation_retention_use:
            from scripts.ib_local.relation_retention_use import configure as configure_relation_use
            previous_use=loaded.config.get('relation_retention_use',{})
            if args.resume_work and previous_use.get('schema',1)<2:
                raise ValueError('legacy relation-use results retain schema 1; use a new work directory for schema 2')
            configure_relation_use(engine,args.relation_retention_use)
            loaded.config['relation_retention_use']=dict(schema=2,
                mode=args.relation_retention_use,adopted_at_cycle=(previous_use.get('adopted_at_cycle',loaded.cycle)
                    if previous_use.get('schema',1)>=2 else loaded.cycle),
                signal='later MOTHER predictive finite removal per scope and episode',
                reward='finite_relation_use_v5',selection='whole U selects with old moments; new label updates successor')
        if args.retention_scope_prior:
            from scripts.ib_local.retention_scope_prior import configure as configure_scope_prior
            old_scope = loaded.config.get('retention_scope_prior', {}).get('mode')
            if args.resume_work and old_scope != args.retention_scope_prior:
                raise ValueError('resume must retain scoped retention prior semantics')
            configure_scope_prior(engine, args.retention_scope_prior)
            loaded.config['retention_scope_prior'] = dict(schema=1, mode=args.retention_scope_prior)
        assert getattr(model, 'joint_relation_update_revision', 0) >= 3
        if previous:
            assert loaded.cycle == previous['cycle'], 'resume requires last complete comparison boundary'
        material_path = args.work/'material.pt'
        if material_path.exists():
            material = torch.load(material_path, map_location=engine.device, weights_only=False)
        else:
            material = (torch.load(args.evaluation_material, map_location=engine.device, weights_only=False)
                        if args.evaluation_material is not None else fixed_queries(engine))
            assert len(material) == 64, 'require the fixed 64 real old queries'
            torch.save(material, material_path)
        evaluations = list(previous.get('evaluations', []))
        origin = previous.get('origin_cycle', loaded.cycle)
        total_external = previous.get('cycles', 0)
        total_bytes = previous.get('external_bytes', 0)
        total_self_bytes = previous.get('self_bytes', 0)
        updates = [previous.get('updates', 0)]
        initial_updates, initial_external = updates[0], total_external
        initial_bytes, initial_self_bytes = total_bytes, total_self_bytes
        lifetime_updates = getattr(engine, 'native_optimizer_updates', None)
        if not evaluations:
            evaluations.append(dict(external=total_external, cycle=loaded.cycle,
                                    **evaluate(engine, material)))
        write(args.work/'evaluations.json', evaluations)
        original_policy, original_advance = engine._policy_grad, engine._advance
        original_objective, original_step = model.objective, engine.optimizer.step
        original_mechanism, original_begin = navigation.mechanism, use_value.begin
        hooks, read_use, pre_outcome = [None], [], {}

        def begin(engine, record):
            original_begin(engine, record)
            receipt = engine.causal_use_pending
            pre_outcome.update(base=float(receipt['gain_base'][3]),
                raw=float(receipt['gain_raw'][3]),
                expected=10*receipt['finite']['expected'],
                choice=record['choice'], units='10*finite_U_nats_per_latent_coordinate')

        def _release_if_tight():
            if not torch.cuda.is_available():
                return
            # Avoid 0.8s+ cyclic GC + allocator thrashing when GPU memory has ample headroom.
            if (activation_options.get('gpu_budget_bytes', 0) >= 1536 * 1024**2 and
                    torch.cuda.memory_reserved() < 3200 * 1024**2):
                return
            gc.collect(); torch.cuda.empty_cache()

        def policy(*a, **kw):
            _release_if_tight()
            with policy_storage(engine.params, args.policy_activations or args.work/'policy_activations',
                                **activation_options):
                return original_policy(*a, **kw)

        def advance(*a, **kw):
            _release_if_tight()
            return original_advance(*a, **kw)

        def objective(*a, **kw):
            with hooks[0]():
                return original_objective(*a, **kw)

        def optimizer_step(*a, **kw):
            value = original_step(*a, **kw)
            updates[0] += 1
            return value

        def mechanism(model, state, source, action, **kw):
            distribution = original_mechanism(model, state, source, action, **kw)
            target_id = kw.get('exclude_fact', '')
            if not (torch.is_grad_enabled() and target_id and
                    kw.get('kind', Port.MOTHER) == Port.MOTHER):
                return distribution
            with torch.no_grad():
                supports = mechanism_evidence(model, state, Port.MOTHER,
                    exclude=kw.get('exclude', -1), exclude_fact=target_id)
                record = dict(target_fact_id=target_id,
                    support_ids=[evidence_key(e) for e in supports],
                    relation=dict(identity=state.identity, generation=state.generation,
                                  version=state.version),
                    comparison_update_clock=updates[0], memory_clock=int(model.memory.clock),
                    lifetime_update_clock=(None if lifetime_updates is None else
                        lifetime_updates+updates[0]-initial_updates))
                # Resolve the label by identity, never by a reused hot slot.
                candidates = list(state.constraints+state.extra)
                if target_id.startswith('episode:'):
                    try:
                        candidates = _evidence(model._causal_archive, int(target_id.split(':')[1]))
                    except ValueError:
                        candidates = []
                target = next((e for e in candidates if e.fact_id == target_id and
                    e.kind == Port.MOTHER and e.observed and not e.encoded and
                    torch.equal(e.source, source) and torch.equal(e.action, action)), None)
                if target is not None:
                    record['log_score'] = float(log_score(distribution, model.encoder(target.outcome)).mean())
                read_use.append(record)
            return distribution

        engine._policy_grad, engine._advance = policy, advance
        model.objective, engine.optimizer.step = objective, optimizer_step
        navigation.mechanism, use_value.begin = mechanism, begin
        from scripts.ib_local.preview_profile import instrument
        profiling = instrument(engine, enabled=args.profile_runtime)
        profile = profiling.__enter__()
        started = time.perf_counter()
        exhausted = True
        pending_updates = 0
        pending_read_use = []
        pending_activation_peak = 0
        try:
            with (args.work/'events.jsonl').open('a' if args.resume_work else 'w', encoding='utf8') as out:
                # Same exceptional cap as the audit runner; never force release
                # or prescribe an internal/external ratio. Save if it expires.
                event_count = 0
                while (event_count < max(512, args.cycles*64)
                       or getattr(engine, '_rhythm_runtime', None) is not None):
                    event_count += 1
                    before_updates = updates[0]
                    read_use.clear()
                    with activation_storage(engine.params, world_activations,
                                            **activation_options) as (storage, context):
                        hooks[0] = context
                        row = dispatch(engine)
                    hooks[0] = None
                    changed = updates[0]-before_updates
                    row['updates'] = changed
                    row['world_activation_bytes'] = storage.get('file_high_water_bytes', 0)
                    row['world_activation_storage'] = storage
                    if profile is not None:
                        write(args.work/'runtime_profile.json', profile.snapshot())
                    if replayed < len(replay):
                        old = replay[replayed]
                        assert_replayed_event(row, old, replayed)
                        replayed += 1
                    external = row['autonomous_event']['kind'] == 'external'
                    if external:
                        row['updates'] += pending_updates
                        pending_updates = 0
                        total_external += 1
                        total_bytes += int(row['external_bytes'])
                        total_self_bytes += int(row['self_bytes'])
                        row['read_use'] = pending_read_use+list(read_use)
                        pending_read_use = []
                        row['world_activation_bytes'] = max(row['world_activation_bytes'], pending_activation_peak)
                        pending_activation_peak = 0
                        row['pre_outcome_gain'] = dict(pre_outcome, target=10*row['metal_g'])
                        row['cycle'] = loaded.cycle+total_external-initial_external
                        if getattr(engine, 'learnable_precision_scales_schema', None) == 1:
                            from scripts.ib_local.learnable_precision_scales import diagnostics as scale_diag
                            row['learnable_precision_scales'] = scale_diag(engine)
                        from scripts.ib_local.optimizer_policy import diagnostics as opt_diag
                        row['optimizer_plasticity'] = opt_diag(engine)
                        if hasattr(engine, 'native_optimizer_updates'):
                            record_birth_event(engine, row)
                        if hasattr(engine, 'growth_optimizer_updates'):
                            engine.growth_optimizer_updates += row['updates']
                            engine.growth_external_bytes += row['external_bytes']
                        assert row.get('world', {}).get('provider_calls', 0) == 0
                    else:
                        pending_updates += changed
                        pending_read_use.extend(read_use)
                        pending_activation_peak = max(pending_activation_peak, row['world_activation_bytes'])
                    out.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'); out.flush()
                    if not external:
                        continue
                    time_up = (args.max_life_seconds is not None and
                               (time.perf_counter()-started) >= args.max_life_seconds)
                    done_now = (total_external-initial_external == args.cycles) or time_up
                    if total_external % args.evaluate_every == 0 or done_now:
                        evaluations.append(dict(external=total_external, cycle=row['cycle'],
                                                **evaluate(engine, material)))
                        write(args.work/'evaluations.json', evaluations)
                    write(status_path, dict(status='running', mode=mode, cycles=total_external,
                        cycle=row['cycle'], updates=updates[0], external_bytes=total_bytes,
                        invocation_cycles=total_external-initial_external,
                        seconds=time.perf_counter()-started))
                    if done_now:
                        exhausted = False
                        break
        finally:
            profiling.__exit__(None, None, None)
            if profile is not None:
                write(args.work/'runtime_profile.json', profile.snapshot())
            navigation.mechanism, use_value.begin = original_mechanism, original_begin
            del engine._policy_grad, engine._advance, model.objective
            engine.optimizer.step = original_step
        cycle = loaded.cycle+total_external-initial_external
        life_seconds = time.perf_counter()-started
        completed_profile = None if profile is None else profile.snapshot()
        commit_started = time.perf_counter()
        digest = save(engine, body, cycle, loaded.config)
        write(body/'config.json', loaded.config)
        checkpoint_commit_seconds = time.perf_counter()-commit_started
        result = dict(status='stopped_poll_budget' if exhausted else 'completed', mode=mode,
            diagnostic_only=args.legacy_restore, main_body_candidate=False,
            scope=('simulated original broken restore: runtime revision forced to 0 after prepare; '
                   'inherited source config unchanged' if args.legacy_restore else
                   'normal prepare restores runtime revision 1 without runner repair'),
            origin_cycle=origin, cycle=cycle, cycles=total_external, updates=updates[0],
            external_bytes=total_bytes, self_bytes=total_self_bytes,
            provider_calls=0, replayed_prefix_events=replayed,
            world_activations=str(world_activations), prepared_revision=prepared_revision,
            runtime_revision=model.identity_preview_revision,
            source_checkpoint=previous.get('source_checkpoint', str(checkpoint)),
            initial_source_sha256=previous.get('initial_source_sha256', loaded.source_sha),
            source_is_own_output=checkpoint == body/'last.pt.gz',
            checkpoint=str(body/'last.pt.gz'), checkpoint_sha256=digest,
            material=str(material_path), evaluations=evaluations,
            runtime_profile=completed_profile,
            life_seconds=life_seconds, checkpoint_commit_seconds=checkpoint_commit_seconds,
            invocation=dict(start_cycle=loaded.cycle, cycles=total_external-initial_external,
                updates=updates[0]-initial_updates, external_bytes=total_bytes-initial_bytes,
                self_bytes=total_self_bytes-initial_self_bytes, seconds=time.perf_counter()-started),
            poll_cap_rule='max(512, additional_cycles*64); stop/save, never force external action')
        write(result_path, result)
        write(status_path, {k: result[k] for k in ('status', 'mode', 'cycle', 'cycles', 'updates', 'external_bytes')})
        print(json.dumps(dict(status=result['status'], result=str(result_path), checkpoint=result['checkpoint'])))
        if exhausted:
            raise RuntimeError('audit poll cap reached; complete terminal checkpoint retained')
    finally:
        model._causal_archive.close()


if __name__ == '__main__':
    main()
