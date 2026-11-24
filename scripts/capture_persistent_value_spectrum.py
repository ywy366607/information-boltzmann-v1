"""Capture the active scalar bank and actual root contractions, before world arrival.

Restore the original complete 2601 boundary and its normal internal prelude.
At the first actual actor, publish the same-menu receipt and stop. Root reads
return original values; a two-fit ridge candidate is evaluated in shadow.
"""
import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
import types

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import torch

from scripts.native_value_residual_filter import coefficients


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')


class Captured(BaseException):
    pass


@contextmanager
def observe(engine, destination):
    from scripts.ib_local import native_amortized_decision as amort
    from scripts.ib_local import persistent_rhythm as rhythm
    from scripts.ib_local import latent_action_planner as planner
    from scripts.ib_local.competing_action_teachers import action_key
    original_read, original_value = rhythm.read_plans, planner.value
    original_relation = engine._relation_value
    own, had = vars(engine).get('_relation_value'), '_relation_value' in vars(engine)
    active, calls, captures, results = [], [], [], []

    @torch.no_grad()
    def read(e, states, sources, *, carried=None):
        baseline, cases, count = original_read(e, states, sources, carried=carried)
        if e is not engine or not active:
            return baseline, cases, count
        packets = [amort.recipe(e, carried or st, source, selected=st)
                   for st, source in zip(states, sources)]
        cache = e._native_fit_cache
        assert cache['x'] is not None
        query = amort.features(e, packets)
        prior = amort._prior(e, packets)
        cross = amort._kernel(query, cache['x'])*cache['scale'][None]
        original = prior+(cross@cache['coefficient'])[:, 0]
        torch.testing.assert_close(original, baseline, atol=2e-5, rtol=2e-5)
        factor = cache['factor'].double()
        alpha = cache['coefficient'].double()
        delta_alpha = e.relation_ridge*torch.cholesky_solve(alpha, factor)
        delta = (cross.double()@delta_alpha)[:, 0].to(baseline)
        if not captures:
            bank = e.native_value_bank['records']
            bank_packets = [r['packet'] for r in bank]
            captures.append(dict(x=cache['x'].cpu(), factor=cache['factor'].cpu(),
                coefficient=cache['coefficient'].cpu(), scale=cache['scale'].cpu(),
                target=torch.tensor([r['target'] for r in bank]),
                prior=amort._prior(e, bank_packets).cpu(), ridge=e.relation_ridge,
                records=[{k:r[k] for k in ('kind', 'key', 'revision', 'target', 'predicted', 'prior')}
                         for r in bank], revision=e.native_value_bank['revision']))
        active[-1]['read'] = dict(baseline=baseline, delta=delta,
            cross=cross.cpu(), prior=prior.cpu())
        return baseline, cases, count

    @torch.no_grad()
    def value(e, state, source, action, mother, distribution, momentum):
        roots = getattr(e, '_competing_action_roots', None)
        key = action_key(source, action) if e is engine and roots else None
        if not roots or key not in roots or results:
            return original_value(e, state, source, action, mother, distribution, momentum)
        row = dict(key=key)
        active.append(row)
        try:
            answer = original_value(e, state, source, action, mother, distribution, momentum)
            got = row['read']
            w = answer.new_tensor(e.successor_distribution_last['weights'])
            torch.testing.assert_close(answer[0], (w*got['baseline']).sum(), atol=2e-5, rtol=2e-5)
            calls.append(dict(key=key, original=float(answer[0]),
                delta=float((w*got['delta']).sum()), weights=w.cpu(),
                cross=got['cross'], prior=got['prior'], baseline=got['baseline'].cpu()))
            return answer
        finally:
            active.pop()

    @torch.no_grad()
    def relation(e, *args, **kwargs):
        if results:
            return original_relation(*args, **kwargs)
        calls.clear()
        rhythm.read_plans, planner.value = read, value
        try:
            result = original_relation(*args, **kwargs)
        finally:
            rhythm.read_plans, planner.value = original_read, original_value
        critic, _ = result
        future = critic.new_tensor(e.epistemic_value_last['contextual_future'])
        delta = torch.zeros_like(critic)
        for key, options in e._competing_action_roots.items():
            rows = [r for r in calls if r['key'] == key]
            assert rows
            mean = sum(r['original'] for r in rows)/len(rows)
            for option in options:
                assert abs(float(future[option])-mean) < 2e-5
                delta[option] = sum(r['delta'] for r in rows)/len(rows)
        results.append(dict(critic_original=critic.tolist(),
            critic_alternative=(critic+10*e.gamma*delta).tolist(),
            future_delta=delta.tolist(), root_options=e._competing_action_roots,
            extra_teacher_calls=0, extra_world_calls=0))
        torch.save(dict(bank=captures[0], calls=calls, root_options=e._competing_action_roots),
                   destination/'capsule.pt')
        write(destination/'teacher_checks.json', e._latent_teacher_checks)
        return result

    engine._relation_value = types.MethodType(relation, engine)
    try:
        yield results
    finally:
        rhythm.read_plans, planner.value = original_read, original_value
        if had:
            engine._relation_value = own
        else:
            del engine._relation_value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--temporary', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--inherited-projection', type=Path,
                   help='original live cache capsule already verified by the historical campaign')
    a = p.parse_args()
    a.work.mkdir(parents=True, exist_ok=False)
    a.temporary.mkdir(parents=True, exist_ok=False)
    from scripts.ib_local import run_mother_continuation as continuation
    from scripts.ib_local.action_value_pair_audit import first_action
    from scripts.root_option_calibration_shadow import choice_receipt
    from scripts import run_native_preview_restore_comparison as runner
    from scripts.run_option_calibration_use import digest
    torch.set_num_threads(1)
    expected = json.loads(a.reference.read_text(encoding='utf8'))
    expected = expected.get('action', expected)
    source_sha = digest(a.checkpoint)
    write(a.work/'protocol.json', dict(source=str(a.checkpoint), source_sha256=source_sha,
        candidate='exactly two residual ridge fits, same active scalar bank',
        ridge='original', features='original', sources='original balanced weights',
        return_mode='original values throughout', world_calls=0, optimizer_updates=0,
        inherited_projection=None if a.inherited_projection is None else dict(
            path=str(a.inherited_projection), sha256=digest(a.inherited_projection)),
        acceptance='all-revision holdout plus same-menu existing G2 contribution'))
    prepare = continuation.prepare_continuation
    opened = []
    def prepared(*args, **kwargs):
        loaded = prepare(*args, **kwargs)
        e = loaded.engine
        def forbidden(*args, **kwargs):
            raise AssertionError('read-only capture reached world or optimizer')
        e.world.step = forbidden
        e.optimizer.step = forbidden
        observer = observe(e, a.work)
        rows = observer.__enter__(); opened.append(observer)
        def publish(receipt):
            fields = ('source_sha256', 'endpoint_sha256', 'recurrent_sha256', 'unknown',
                      'proposals', 'selected_proposal', 'local', 'future', 'total',
                      'habit', 'nominal_probability')
            checks = {k:receipt[k] == expected[k] for k in fields}
            write(a.work/'matching_checks.json', checks)
            assert all(checks.values()), checks
            assert len(rows) == 1
            # choice_receipt needs the base field only as a description.
            write(a.work/'root_shadow.json', choice_receipt(e, receipt, rows[0]))
            write(a.work/'action.json', receipt)
            raise Captured()
        context = first_action(e, 0, publish, keep_actor_learning=True)
        context.__enter__(); opened.append(context)
        return loaded
    continuation.prepare_continuation = prepared
    argv = sys.argv
    sys.argv = [argv[0], '--checkpoint', str(a.checkpoint), '--work', str(a.work/'runtime'),
        '--cycles', '1', '--native-decision', '--native-word-execution', '--successor-distribution',
        '--joint-successor', '--batch-native-words', '--batch-preview-scores',
        '--amortized-decision', '--latent-action-planner', '--persistent-rhythm', 'coupled',
        '--root-native-correction', 'observe', '--retention-scope-prior', 'off',
        '--table-storage', str(a.temporary/'table')]
    try:
        projection_context = nullcontext()
        if a.inherited_projection is not None:
            from scripts.audit_wood_policy_restore import inherited_projection
            projection_context = inherited_projection(torch.load(
                a.inherited_projection, map_location='cpu', weights_only=False)['weight'])
        try:
            with projection_context:
                runner.main()
        except Captured:
            pass
        assert (a.work/'root_shadow.json').exists()
        assert digest(a.checkpoint) == source_sha
        write(a.work/'status.json', dict(status='completed', world_calls=0, optimizer_updates=0,
            source_unchanged=True, capsule_bytes=(a.work/'capsule.pt').stat().st_size))
        print((a.work/'status.json').read_text(), flush=True)
    finally:
        while opened:
            opened.pop().__exit__(None, None, None)
        continuation.prepare_continuation = prepare
        sys.argv = argv


if __name__ == '__main__':
    main()
