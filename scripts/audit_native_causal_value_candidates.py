"""Compare value candidates on recorded full-body events and actual relations.

The evidence experiment removes one supporting fact from an actual
leave-target-out prediction. Encoder, query, prior and target stay fixed.
This measures present cross-fact predictive contribution, not the unobserved
return of taking a different action. No body mutation or new training.
"""
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.audit_native_action_long_cause import ROOT, read_events
from scripts.measure_native_inner_loop import load
from scripts.measure_native_final_learning_value import dist_key
from scripts.ib_local.causal_relation_program import posterior, log_score


def summary(values):
    a = np.asarray(values, dtype=float)
    return dict(count=len(a), mean=float(a.mean()) if len(a) else None,
                minimum=float(a.min()) if len(a) else None,
                maximum=float(a.max()) if len(a) else None,
                positive=int((a > 1e-6).sum()), negative=int((a < -1e-6).sum()),
                within_1e_6=int((np.abs(a) <= 1e-6).sum()))


@torch.no_grad()
def audit_arm(folder):
    events, _ = read_events(folder)
    ranking = []
    for e in events:
        p = e['causal_program']['pre_draft']
        info = np.asarray(p['conditional_information'])
        value = np.asarray(p['expected_after_value'])
        assert p['selected'] == int(value.argmax())
        ranking.append(dict(event=e['event'], candidates=p['candidate_count'],
                            selected=int(value.argmax()), information_best=int(info.argmax()),
                            information=info.tolist(), value=value.tolist(),
                            realized_text_progress=e['settlement']))

    obs = folder / 'observations'
    rows = [json.loads(x) for x in gzip.open(obs/'operations.jsonl.gz', 'rt', encoding='utf8')]
    calls = {r['data']['call_id']: r for r in rows if r['kind'] == 'relation_signal'}
    children = defaultdict(list)
    scores = defaultdict(list)
    for r in calls.values():
        d = r['data']
        children[d['parent_id']].append(d)
        if d['operation'] == 'log_score' and d['caller']['function'] == '_outcome_loss':
            scores[(r['event'], r['update_clock'], dist_key(d['prediction']))].append(d)
    contributions, reconstruction = [], []
    with (obs/'tensors.bin').open('rb') as stream:
        def tensor(ref):
            return torch.from_numpy(np.array(load(stream, ref), copy=True)).float()

        for r in calls.values():
            d = r['data']
            if not (d['operation'] == 'mechanism' and d.get('kind') == 'mother'
                    and d.get('exclude_fact') and d['caller']['function'] == '_outcome_loss'):
                continue
            kids = children[d['call_id']]
            post = [k for k in kids if k['operation'] == 'posterior']
            design = [k for k in kids if k['operation'] == '_bound_evidence_design']
            assert len(post) == len(design) == 1
            p, b = post[0], design[0]
            facts = [f['fact_id'] for f in b['facts']]
            assert d['exclude_fact'] not in facts
            assert len(set(facts)) == len(facts)
            assert all(f['observed'] and not f['encoded'] for f in b['facts'])
            assert all(n == 1 for n in b['evidence_row_counts'])
            hits = scores[(r['event'], r['update_clock'], dist_key(d['prediction']))]
            assert len(hits) == 1
            score = hits[0]
            x, y, q = [tensor(p[k]) for k in ('design', 'residual', 'query')]
            prior = tensor(p['noise_prior']) if p['noise_prior'] is not None else None
            base, target = tensor(d['prediction']['base']), tensor(score['target'])
            full = posterior(x, y, q, p['ridge'], noise_prior=prior)
            full['mean'] = full['mean'] + base
            full_score = float(log_score(full, target).mean())
            original = float(tensor(score['score']).mean())
            error = abs(full_score-original)
            assert error < 2e-5, (folder, d['call_id'], error)
            reconstruction.append(error)
            for i, fact in enumerate(facts):
                keep = torch.arange(len(x)) != i
                reduced = posterior(x[keep], y[keep], q, p['ridge'], noise_prior=prior)
                reduced['mean'] = reduced['mean'] + base
                reduced_score = float(log_score(reduced, target).mean())
                # The body's own conditional Gaussian coefficient information
                # formula, now evaluated for this actual supporting fact.
                candidate = posterior(x[keep], y[keep], x[i:i+1], p['ridge'], noise_prior=prior)
                information = float(.5*y.shape[-1]*torch.log1p(candidate['leverage']).mean())
                # Targeted predictive information in the SAME conditional
                # Gaussian model: shared coefficient uncertainty couples the
                # candidate response to another query's future response.
                # It uses roles/queries only, not either response label.
                xx, qq, xi = x[keep].double(), q.double(), x[i:i+1].double()
                precision = p['ridge']*torch.eye(x.shape[1], dtype=torch.float64) + xx.T@xx
                aq = torch.linalg.solve(precision, qq.T)
                ai = torch.linalg.solve(precision, xi.T)
                rho2 = (qq@ai).square().flatten() / (
                    (1+(qq*aq.T).sum(-1))*(1+(xi*ai.T).sum(-1)))
                assert bool(((rho2 >= 0) & (rho2 < 1)).all())
                predictive_information = float(-.5*y.shape[-1]*torch.log1p(-rho2).mean())
                assert predictive_information <= information + 1e-4
                contributions.append(dict(event=r['event'], update_clock=r['update_clock'],
                    mechanism_call=d['call_id'], relation=[d['state']['identity'], d['state']['generation']],
                    target_fact=d['exclude_fact'], supporting_fact=fact,
                    retained_support=[f for j, f in enumerate(facts) if j != i],
                    conditional_information=information,
                    targeted_predictive_information=predictive_information,
                    full_score=full_score, removed_score=reduced_score,
                    transfer_log_score_gain=full_score-reduced_score,
                    transfer_mse_gain=float(((target-reduced['mean'])**2).mean()
                                            - ((target-full['mean'])**2).mean())))
    query_groups = defaultdict(list)
    for item in contributions:
        query_groups[item['mechanism_call']].append(item)
    selection = {}
    for metric in ('conditional_information', 'targeted_predictive_information'):
        chosen = [max(group, key=lambda item: item[metric]) for group in query_groups.values()]
        selection[metric] = summary([item['transfer_log_score_gain'] for item in chosen])
    return dict(pre_draft_rankings=ranking, evidence_contributions=contributions,
                summary=dict(events=len(events),
                    information_value_argmax_disagreements=sum(r['selected'] != r['information_best'] for r in ranking),
                    factual_queries=len(reconstruction), max_score_reconstruction_error=max(reconstruction),
                    contribution=summary([r['transfer_log_score_gain'] for r in contributions]),
                    mse_contribution=summary([r['transfer_mse_gain'] for r in contributions]),
                    information=summary([r['conditional_information'] for r in contributions]),
                    targeted_information=summary([r['targeted_predictive_information'] for r in contributions]),
                    highest_information_supporter_contribution=selection,
                    targeted_vs_coefficient_argmax_disagreements=sum(
                        max(group, key=lambda z:z['targeted_predictive_information'])['supporting_fact'] !=
                        max(group, key=lambda z:z['conditional_information'])['supporting_fact']
                        for group in query_groups.values())),
                source_sha256=hashlib.sha256((obs/'operations.jsonl.gz').read_bytes()).hexdigest())


def main():
    torch.set_num_threads(1)
    base = ROOT/'scratch/native_identity_preview_bound_2401_20261117'
    result = dict(arms={}, new_training_updates=0, new_world_calls=0,
        definitions=dict(information='conditional coefficient information; same production posterior and fixed noise prior',
            targeted_information='conditional Gaussian response-response mutual information on an actual excluded query; no response labels used',
            contribution='log score with supporting fact minus score without it on another actual excluded fact; fixed coordinates',
            ranking='recorded pre-draft binding candidates, not decoded utterance candidates',
            scope='current relation evidence contribution; proposal for valuation, not a deployed reward or causal-identification certificate'))
    for arm in ('baseline', 'bridge_cut', 'matched_cut'):
        result['arms'][arm] = audit_arm(base/arm)
        print(json.dumps({arm: result['arms'][arm]['summary']}), flush=True)
    result['source_code_sha256'] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
        'scripts/ib_local/causal_relation_program.py', 'scripts/ib_local/navigation_causal_program.py',
        'scripts/audit_native_causal_value_candidates.py')}
    output = ROOT/'results/published/cgb_native_causal_value_candidates_2401_20261117.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
