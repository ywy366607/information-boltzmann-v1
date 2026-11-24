"""Read-only paired support/use diagnostic on one complete C2 checkpoint.

Run serially with training. Only prepare_continuation restores/saves a private
working copy; measurement runs with query-only SQL and no world/optimizer step.
Support interventions use production score_state, not a whole-organization U:
current program weights/scope stay fixed, linked organization/uses are removed.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from scripts.ib_local.causal_relation_program import Port, compose, mechanism, revise, release_choice
from scripts.ib_local.causal_use_value import calibrated, utility
from scripts.ib_local.causal_value_learning import prediction
from scripts.ib_local.finite_use_value import expected, score_scope, score_state
from scripts.ib_local.joint_relation_update import relation_variables
from scripts.ib_local.relation_evidence import mechanism_evidence


def tensor_hash(*tensors):
    h = hashlib.sha256()
    for tensor in tensors:
        t = tensor.detach().cpu().contiguous()
        h.update(str((str(t.dtype), tuple(t.shape))).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def distribution(values):
    x = torch.as_tensor(values, dtype=torch.float64).flatten()
    return dict(count=len(x), min=float(x.min()), median=float(x.quantile(.5)),
                mean=float(x.mean()), max=float(x.max())) if len(x) else dict(count=0)


def fact_record(fact):
    return dict(id=fact.fact_id, port=fact.kind.value, slot=fact.slot, stamp=fact.stamp,
                observed=fact.observed, encoded=fact.encoded,
                input_hash=tensor_hash(fact.source, fact.action),
                outcome_hash=tensor_hash(fact.outcome))


def real_facts(model, state):
    return tuple(f for f in mechanism_evidence(model, state, Port.MOTHER)
                 if f.observed and not f.encoded and f.fact_id)


def support_states(model, state, *, exclude=()):
    """Full/reordered/single withdrawal; remove an episode's BOTH typed ports.

    score_state is the production local C2 scoring projection. No stale linked
    organization or variable use can reintroduce a removed constraint. This is
    not a counterfactual reorganization of the full shared SQL graph.
    """
    excluded = set(exclude)
    facts = tuple(f for kind in (Port.MOTHER, Port.DECISION)
                  for f in mechanism_evidence(model, state, kind)
                  if f.fact_id not in excluded)
    base = replace(score_state(state), constraints=facts, extra=(), evidence_bound=True,
                   revision_plan=None)
    ids = [f.fact_id for f in real_facts(model, base)]
    variants = [('full', base), ('reordered', replace(base, constraints=facts[::-1]))]
    variants.extend((f'without:{identity}', replace(base, constraints=tuple(
        f for f in facts if f.fact_id != identity))) for identity in ids)
    return variants


@torch.no_grad()
def raw_options(model, sources, actions):
    """The actual shared function, all three option inputs, no R/Q/outcome."""
    s, a = model.encoder(sources), model.encoder(actions)
    z = s[:, None].expand(-1, 3, -1)
    delta = (a-s)[:, None].expand_as(z)
    options = torch.arange(3, device=s.device).expand(len(s), -1)
    return prediction(model, z, delta, options)


def raw_record(model, sources, actions):
    raw = raw_options(model, sources, actions)
    s, a = model.encoder(sources), model.encoder(actions)
    options = torch.arange(3, device=s.device).expand(len(s), -1)
    return dict(scope='current action gain head at fixed source/action/option, not the complete R-dependent valuation',
                prediction=raw.tolist(), gain=raw[..., 3].tolist(),
                gain_distribution=distribution(raw[..., 3]),
                input_hash=tensor_hash(s[:, None].expand(-1, 3, -1),
                    (a-s)[:, None].expand(-1, 3, -1), options))


def path_record(model, path):
    packet = path['continuation']
    # Unlike the current action's gain_raw, continuation's source is the
    # R-dependent predicted/coordinated decision mean. Reuse the exact shared
    # function on its actual recalled displacement/option inputs, before the
    # production confidence mixture already retained in packet.prediction.
    m, slots = model.memory, packet['slots']
    s, a = model.encoder(m.source[slots]), model.encoder(m.action[slots])
    z = path['decision']['mean'][:, None].expand_as(s)
    options = m.wood_option[slots].clamp(0, 2)
    raw = prediction(model, z, a-s, options)
    return dict(value=path['value'].tolist(),
                value_distribution=distribution(path['value']),
                mother_mean_hash=tensor_hash(path['mother']['mean']),
                decision_mean_hash=tensor_hash(path['decision']['mean']),
                program_weights=(None if 'program_weights' not in path else path['program_weights'].tolist()),
                continuation=dict(slots=packet['slots'].tolist(), weights=packet['weights'].tolist(),
                    support=packet['support'].tolist(), prediction=packet['prediction'].tolist(),
                    raw_prediction=raw.tolist(), raw_gain_distribution=distribution(raw[..., 3]),
                    input_hash=tensor_hash(z, a-s, options),
                    scope='production confidence-mixed prediction and shared raw at R-dependent decision mean'))


@torch.no_grad()
def compare_supports(engine, state, sources, actions, query_facts, *, exclude=(),
                     incoming=None, momentum=None):
    """Core paired comparison shared by the CLI and CPU fixtures.

    Q and source/action/option inputs never change between states. For an
    observed replay, incoming is absent from EVERY pre-state. Its outcome is
    used only by the subsequent functional revise, never by a forecast call.
    No assertion requires a support change to change or improve utility.
    """
    query_facts = tuple(query_facts)
    excluded = set(exclude)
    if incoming is not None:
        if not incoming.observed or incoming.encoded or not incoming.fact_id:
            raise ValueError('incoming must be an identified real raw Evidence')
        excluded.add(incoming.fact_id)
        if any(f.fact_id == incoming.fact_id for f in query_facts):
            raise ValueError('incoming cannot be its own old query')
        if not torch.equal(sources, incoming.source) or not torch.equal(actions, incoming.action):
            raise ValueError('observed replay must use the actual incoming source/action')
    if not query_facts:
        raise ValueError('a nonempty fixed old query scope is required')
    rows = []
    for name, current in support_states(engine.model, state, exclude=excluded):
        d = mechanism(engine.model, current, sources, actions, ridge=engine.relation_ridge,
                      momentum=momentum)
        use = expected(engine, current, sources, actions, d, query_facts)
        path = compose(engine.model, current, sources, actions, ridge=engine.relation_ridge,
                       reward_valid=engine.reward_label_valid(wood=True), momentum=momentum)
        # Same functional utility formula, but intentionally FIXED Q. Calling
        # utility() here would silently change Q after a support withdrawal.
        total = 10*calibrated(engine, use)+engine.gamma*path['value']
        before = score_scope(engine, score_state(current), query_facts)
        fit_ids = []
        for q in query_facts:
            fit_ids.append([f.fact_id for f in mechanism_evidence(
                engine.model, current, Port.MOTHER, exclude_fact=q.fact_id)])
        row = dict(state=name, supports=[f.fact_id for f in real_facts(engine.model, current)],
            query_ids=[f.fact_id for f in query_facts], query_fit_ids=fit_ids,
            raw=raw_record(engine.model, sources, actions), expected=use.tolist(),
            expected_distribution=distribution(use), score_before=before.tolist(),
            total=total.tolist(), total_distribution=distribution(total), path=path_record(engine.model, path))
        if incoming is not None:
            updated = revise(score_state(current), incoming, model=engine.model)
            after = score_scope(engine, updated, query_facts)
            gains = after-before
            row['observed'] = dict(score_after=after.tolist(), gains=gains.tolist(),
                gain_distribution=distribution(gains), net_gain=float(gains.mean()),
                expected=float(use[0]), realized_residual=float(gains.mean()-use[0]),
                supports_after=[f.fact_id for f in real_facts(engine.model, updated)],
                query_fit_ids_after=[[f.fact_id for f in mechanism_evidence(
                    engine.model, updated, Port.MOTHER, exclude_fact=q.fact_id)] for q in query_facts],
                raw_after=raw_record(engine.model, sources, actions))
        rows.append(row)
    reference = rows[0]
    contrasts = []
    for row in rows[1:]:
        delta = torch.tensor(row['expected'])-torch.tensor(reference['expected'])
        contrast = dict(state=row['state'], prospective_expected_delta=delta.tolist(),
            prospective_expected_delta_distribution=distribution(delta),
            raw_gain_delta=(torch.tensor(row['raw']['gain'])-
                            torch.tensor(reference['raw']['gain'])).tolist())
        if incoming is not None:
            contrast['fixed_observed_outcome_gain_delta'] = (
                row['observed']['net_gain']-reference['observed']['net_gain'])
        contrasts.append(contrast)
    return dict(query_ids=[f.fact_id for f in query_facts], queries=[fact_record(f) for f in query_facts],
        source_action_hash=tensor_hash(sources, actions), rows=rows, contrasts_to_full=contrasts,
        interpretation='fixed-outcome target differences and prospective expected differences are separate; current gain_raw has fixed source/action, while C2 immediate reads R and continuation reads the R-dependent decision mean; a shared head may learn a marginal mean, not every random return',
        scope='production score_state local finite-U; frozen program weights/scope; not full organization intervention',
        query_fit='score_scope excludes each query identity from its own fit before and after U',
        expected_role='prospective quadrature estimate, never a real training label',
        total_formula='10*calibrated(expected_on_fixed_Q)+gamma*production_compose.value')


@torch.no_grad()
def current_binding(engine, source):
    """Read the existing applicable bind result, without bind's allocation/write.

    If the next production bind would create a relation, there is no existing
    current support to diagnose. Do not substitute a different old relation.
    """
    model, bank, m = engine.model, engine.model.causal_relations, engine.model.memory
    live = bank.identity > 0
    if hasattr(bank, 'applicable'):
        live = live & bank.applicable(source)
    similarity = (F.normalize(model.encoder(source), dim=-1) @
                  F.normalize(model.encoder(bank.condition), dim=-1).T)[0]
    value, index = similarity.masked_fill(~live, -2).max(0)
    if not bool(live.any()) or float(value) < m.threshold:
        raise ValueError('current source would allocate a new relation; no existing binding to audit')
    i = int(index)
    if hasattr(bank, 'program_owner') and int(bank.program_owner[i]) != int(bank.identity[i]):
        raise ValueError('current relation needs program initialization; refusing a body mutation')
    state = bank.state(i)
    valid = state.valid & m.occupied[state.slots] & (m.count[state.slots] == 1)
    valid &= (m.last_write[state.slots] == state.stamps)
    valid &= m.mother_valid[state.slots] & (m.mother_stamp[state.slots] == state.stamps)
    state = replace(state, valid=valid)
    state = replace(state, release=release_choice(model, state))
    return relation_variables(model, state)


@torch.no_grad()
def historical_errors(engine):
    """Current same-version individually labelled hot episodes, without fitting.

    Descriptive in-memory replay errors, NOT independent generalization or a
    recomputation of the accumulated pre-outcome ActionEvidence confidence.
    """
    m, model = engine.model.memory, engine.model
    valid = (m.occupied & (m.count == 1) & (m.wood_target_count == 1) &
             (m.wood_option >= 0) & engine.reward_label_valid(wood=True))
    slots = torch.nonzero(valid).flatten()
    records = []
    # Bound transfer/JSON size; aggregate every eligible row, not just topk.
    for chunk in slots.split(64):
        if not len(chunk):
            continue
        options = m.wood_option[chunk]
        s, a = model.encoder(m.source[chunk]), model.encoder(m.action[chunk])
        raw = prediction(model, s, a-s, options)[:, 3]
        base = engine.causal_use_value_evidence.bases()[options, 3]
        target = m.wood_target[chunk, 3]
        for j, slot in enumerate(chunk.tolist()):
            mapping = model._causal_archive.hot.get(str(slot))
            identity = (f"episode:{mapping['id']}" if mapping is not None and
                        model._causal_archive._signature(slot) == mapping['signature'] else None)
            records.append(dict(slot=slot, id=identity, stamp=int(m.last_write[slot]),
                option=int(options[j]), label=float(target[j]), base=float(base[j]), raw=float(raw[j]),
                base_error=float(base[j]-target[j]), raw_error=float(raw[j]-target[j])))
    def summarize(rows):
        return dict(samples=len(rows),
            base_error=distribution([r['base_error'] for r in rows]),
            raw_error=distribution([r['raw_error'] for r in rows]),
            base_squared_error=distribution([r['base_error']**2 for r in rows]),
            raw_squared_error=distribution([r['raw_error']**2 for r in rows]),
            raw_minus_base_squared_error=distribution([
                r['raw_error']**2-r['base_error']**2 for r in rows]))
    evidence = engine.causal_use_value_evidence
    return dict(scope='all eligible hot same-version unmerged old labels; current read-only errors, not held-out',
        reward_version=engine.settlement_kind, units='10*finite_U_nats_per_latent_coordinate',
        aggregate=summarize(records), by_option=[dict(option=o, **summarize([
            r for r in records if r['option'] == o])) for o in range(3)],
        sample_rows=records[:8], all_rows_sha256=hashlib.sha256(json.dumps(
            records, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        confidence=evidence.confidence().tolist(),
        confidence_evidence={k:getattr(evidence, k).tolist() for k in
            ('seen', 'mean', 'blind', 'base_error', 'raw_error')},
        confidence_scope='accumulated real pre-outcome calibration; structural pairs do not explain or override it')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume-work', action='store_true')
    args = parser.parse_args()
    from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation
    from scripts.ib_local.navigation_momentum_wood import decision_momentum_read
    from scripts.run_native_relation_mode_use import sha
    torch.set_num_threads(1)
    started = time.perf_counter()
    checkpoint = args.checkpoint.resolve()
    original_sha = sha(checkpoint)
    opts = continuation_parser().parse_args(['--checkpoint', str(checkpoint),
        '--output', str(args.work/'body'), '--cycles', '1', '--world-source', 'text',
        '--autonomous-events', '--optimizer-policy', 'budgeted',
        '--table-storage', str(args.work/'table'), '--causal-storage', str(args.work/'causal')])
    loaded = prepare_continuation(opts, resume_output=args.resume_work)
    engine, model = loaded.engine, loaded.engine.model
    archive = model._causal_archive
    try:
        if engine.causal_use_arm != 'C2':
            raise ValueError('requires a complete C2 checkpoint')
        archive.db.execute('PRAGMA query_only=ON')
        sql_changes = archive.db.total_changes
        with torch.no_grad():
            source = engine._read_source()
            state = current_binding(engine, source)
            facts = real_facts(model, state)
            if len(facts) < 2:
                raise ValueError('current actual binding needs at least two real supports for incoming/old-Q pairing')
            topk = min(8, model.memory.topk)
            action_facts = facts[:topk-1]
            if getattr(model, 'identity_preview_revision', 0) >= 1:
                from scripts.ib_local.identity_preview import ranked_action_facts
                action_facts = ranked_action_facts(model, source, facts, topk-1)
            actions = torch.cat((source, *[source+f.action-f.source for f in action_facts]))
            sources = source.expand(len(actions), -1)
            momentum = decision_momentum_read(model, engine.bfss_momenta, source).expand(len(actions), -1)
            native = compose(model, state, sources, actions, ridge=engine.relation_ridge,
                reward_valid=engine.reward_label_valid(wood=True), momentum=momentum)
            native_total = utility(engine, state, sources, actions, native)
            queries = tuple(sorted(facts, key=lambda f:f.fact_id)[:topk])
            current = compare_supports(engine, state, sources, actions, queries, momentum=momentum)
            # Selection uses real event identity/time only, never measured gains.
            incoming = max(facts, key=lambda f:(f.stamp, f.fact_id))
            old_queries = tuple(f for f in sorted(facts, key=lambda f:f.fact_id)
                                if f.fact_id != incoming.fact_id)[:topk]
            observed = compare_supports(engine, state, incoming.source, incoming.action, old_queries,
                                        incoming=incoming)
            payload = archive.db.execute('SELECT payload FROM episodes WHERE id=?',
                (int(incoming.fact_id.split(':')[1]),)).fetchone()
            episode = archive._unpack(payload[0])
            incoming_option = int(episode['wood_option'])
            history = historical_errors(engine)
            result = dict(schema=1, checkpoint=str(checkpoint), checkpoint_sha256=original_sha,
                restoration=dict(private_checkpoint=str((loaded.output/'last.pt.gz').resolve()),
                    private_checkpoint_sha256=loaded.checkpoint_sha,
                    archive_manifest=loaded.config.get('causal_archive'),
                    scope='standard complete prepare_continuation restore/save precedes query-only measurement'),
                code_sha256={name:sha(ROOT/'scripts/ib_local'/name) for name in
                    ('causal_value_learning.py', 'finite_use_value.py', 'causal_relation_program.py', 'causal_use_value.py')},
                cycle=loaded.cycle, reward_version=engine.settlement_kind,
                identity_preview=dict(declared=loaded.config.get('identity_preview_revision'),
                    runtime=getattr(model, 'identity_preview_revision', 0)),
                relation=[state.identity, state.generation, state.version],
                variable_uses=[dict(id=v.identity, port=v.port.value, version=version, seen_version=seen)
                               for v, version, seen in state.variable_uses],
                coordinate_hash=tensor_hash(model.encoder.weight, *([] if model.encoder.bias is None else [model.encoder.bias])),
                source_action_hash=tensor_hash(sources, actions), topk=topk,
                facts=[fact_record(f) for f in facts],
                candidates=[dict(candidate=i, origin='current' if i == 0 else action_facts[i-1].fact_id,
                    input_hash=tensor_hash(sources[i:i+1], actions[i:i+1])) for i in range(len(actions))],
                native=dict(raw=raw_record(model, sources, actions), path=path_record(model, native),
                    utility=native_total.tolist(), utility_distribution=distribution(native_total)),
                paired=current, incoming=dict(**fact_record(incoming), selection='latest actual bound support by stamp/id',
                    option=incoming_option,
                    stored_reward_version=(int(episode['$wood_settlement_versions'])
                        if '$wood_settlement_versions' in episode else None),
                    current_reward_version=engine.settlement_kind,
                    momentum_scope='no reconstructed historical P; observed replay uses actual source/action and current coordinates',
                    outcome_role='real observed label, used only after pre-revision predictions'),
                observed_replay=observed, historical_errors=history,
                units=dict(raw_channels='first three inherited outcome channels; fourth=10*finite_U_nats_per_latent_coordinate',
                    expected_and_observed='finite_U_nats_per_latent_coordinate',
                    score='Student-t nats_per_latent_coordinate', continuation_and_total='10*finite_U_nats_per_latent_coordinate'),
                calibration=dict(engine.causal_use_calibration), gamma=engine.gamma,
                optimizer_updates=0, world_calls=0, measurement_sql_changes=archive.db.total_changes-sql_changes,
                source_unchanged=sha(checkpoint) == original_sha,
                seconds=time.perf_counter()-started)
        if result['measurement_sql_changes'] or not result['source_unchanged']:
            raise RuntimeError('read-only diagnostic changed its source or SQL')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
        print(json.dumps(dict(cycle=loaded.cycle, output=str(args.output.resolve()),
            candidates=len(actions), supports=len(facts), old_queries=len(old_queries),
            optimizer_updates=0, world_calls=0)))
    finally:
        archive.close()


if __name__ == '__main__':
    main()
