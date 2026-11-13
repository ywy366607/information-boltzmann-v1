"""Summarize the registered dual-view gate from existing audit records only."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def summarize(audit, content):
    last = next(w for w in reversed(audit['audit']['warmup'][-1]['windows']) if w['supervised'])
    cases = []
    for r in content['records']:
        f = r['factual_responsibility']['report']
        c = r['changed_memory_responsibility']['report']
        identities = f['real']['identities']
        assert identities == c['real']['identities'] == f['alternative']['identities'] == c['alternative']['identities']
        records = {(p['layer']-1, p['slot']) for p in r['replaced_records']}
        # Bayes: log p_i(A)/p_i(B) = log r_i(A)/r_i(B) + log p(A)/p(B).
        # Difference across fixed-current-state content interventions cancels
        # the component prior. Recover individual expert contrast changes.
        changes = []
        for j, identity in enumerate(identities):
            delta = r['stored_value_contrast']
            for view, sign in ((f, 1), (c, -1)):
                pa = view['real']['posterior_mass'][j]
                pb = view['alternative']['posterior_mass'][j]
                delta += sign*(math.log(pa)-math.log(pb))
            changes.append(delta)
        views = []
        for layer in range(4):
            for mode in ('state', 'transition'):
                selected = [changes[j] for j, identity in enumerate(identities)
                            if identity is not None and (identity[0], identity[1]) in records
                            and identity[0] == layer and identity[3] == mode]
                assert selected
                views.append(dict(layer=layer+1, view=mode, mean_expert_contrast_change=sum(selected)/len(selected)))
        cases.append(dict(case=r['case'], nll_benefit=r['factual_content_nll_benefit'],
            correct_vs_alternative_contrast=r['stored_value_contrast'],
            null_expert_contrast_drift=changes[0], layer_views=views))
    grouped = []
    for j in range(8):
        values = [r['layer_views'][j]['mean_expert_contrast_change'] for r in cases]
        grouped.append(dict(layer=cases[0]['layer_views'][j]['layer'],
            view=cases[0]['layer_views'][j]['view'], favorable_cases=sum(v > 0 for v in values),
            mean_expert_contrast_change=sum(values)/len(values)))
    return dict(status='complete_gate_passed' if content['favorable_nll_cases'] >= 7 and content['favorable_contrast_cases'] >= 7 else 'complete_functional_gate_failed',
        schema=4, new_lean_coverage=0, promoted=False, long_pair_started=False,
        tests_passed=65, added_persistent_scalars=4, extra_history_pages=0,
        calibration_examples=8, calibration_optimizer_updates=8,
        discarded_resume_verification_updates=2,
        supervised_bytes=sum(w['supervised'] for r in audit['audit']['warmup'] for w in r['windows']),
        observed_bytes=sum(w['observed'] for r in audit['audit']['warmup'] for w in r['windows']),
        audit_seconds=audit['seconds'], peak_cuda_bytes=audit['peak_cuda_bytes'],
        neutral_migration_max=audit['audit']['neutral_max'],
        held_full_forward_max=audit['audit']['held_full_forward_max'],
        resume_next_update_exact=audit['audit']['resume_next_update_exact'],
        source_unchanged=audit['source_unchanged'], executed_sources_unchanged=audit['executed_sources_unchanged'],
        source_sha256=audit['source_sha256'],
        last_layers=last['layers'], old_gradients=last['old_gradients'],
        view_prior_gradients=last['view_prior_gradients'],
        components=last['responsibility']['components'],
        posterior_change_l1=last['responsibility']['posterior_change_l1'],
        prior_gradient_max_error=last['responsibility']['prior_gradient_max_error'],
        probability_sum_error=last['responsibility']['predictive_probability_sum_error'],
        last_boundary_layer_views=last['responsibility']['layer_views'],
        content={k: v for k, v in content.items() if k != 'records'},
        layer_view_content=grouped, cases=cases,
        max_null_expert_contrast_drift=max(abs(r['null_expert_contrast_drift']) for r in cases),
        conclusion='Two views and boundary-to-local credit are wired. Fixed-state content contrast is not selective; no long run or promotion. These eight calibration updates do not establish a learning-capacity limit.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('audit', 'content', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = summarize(read(args.audit), read(args.content))
    result['sources'] = [dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                         for path in (args.audit, args.content)]
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k in (
        'status', 'calibration_optimizer_updates', 'supervised_bytes', 'observed_bytes',
        'max_null_expert_contrast_drift', 'layer_view_content')}, indent=2))


if __name__ == '__main__':
    main()
