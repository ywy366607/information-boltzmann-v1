"""Compare the existing retention choice with its own fixed-query purpose.

Uses a real committed event and current common coordinates, not new world
observations. This is a same-boundary diagnostic, not a future-use oracle or a
proposal to greedily optimize old questions.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import torch
    from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation
    from scripts.ib_local.run_causal_water_reply import sha256_file
    from scripts.ib_local.causal_use_value import queries
    from scripts.ib_local.finite_use_value import score_scope, score_state
    from scripts.ib_local.causal_relation_program import Port, revise
    from scripts.ib_local.relation_evidence import mechanism_evidence, evidence_key
    from scripts.ib_local.shared_unknown_bindings import _read_evidence
    torch.set_num_threads(1)
    checkpoint = args.work/'body/last.pt.gz'
    before_hash = sha256_file(checkpoint)
    opts = continuation_parser().parse_args(['--checkpoint', str(checkpoint),
        '--output', str(checkpoint.parent), '--resume-output', '--cycles', '1',
        '--world-source', 'text', '--autonomous-events', '--optimizer-policy', 'budgeted',
        '--table-storage', str(args.work/'water_audit_table'),
        '--causal-storage', str(args.work/'water_audit_causal')])
    loaded = prepare_continuation(opts, resume_output=True)
    e, model = loaded.engine, loaded.engine.model
    try:
        with torch.no_grad():
            state = score_state(e.internal_pending['causal_program']['state'])
            fact_id = e.internal_pending['causal_program']['episode_identity']
            incoming = next(f for f in _read_evidence(model._causal_archive,
                int(fact_id.split(':')[1])) if f.kind == Port.MOTHER)
            qs = queries(e, state)
            old = tuple(mechanism_evidence(model, state, Port.MOTHER))
            assert len({evidence_key(f) for f in old}) == len(old)
            before = score_scope(e, state, qs)
            default = revise(state, incoming, model=model)
            default_ids = sorted(evidence_key(f) for f in mechanism_evidence(model, default, Port.MOTHER))
            default_scores = score_scope(e, default, qs)
            records = []
            # Incoming stays, matching production's admissible retention choices.
            for victim in old:
                kept = tuple(f for f in old if evidence_key(f) != evidence_key(victim))
                alternative = replace(state, constraints=kept+(incoming,), extra=())
                scores = score_scope(e, alternative, qs)
                ids = sorted(evidence_key(f) for f in mechanism_evidence(model, alternative, Port.MOTHER))
                records.append(dict(released=evidence_key(victim), supports=ids,
                    mean_score=float(scores.mean()), mean_gain=float((scores-before).mean()),
                    per_query_gain=(scores-before).tolist(), selected_by_production=ids == default_ids))
            selected = [r for r in records if r['selected_by_production']]
            assert len(selected) == 1, 'the actual finite U must match one admissible release'
            records.sort(key=lambda r:r['mean_score'], reverse=True)
            out = dict(cycle=loaded.cycle, event=fact_id, source_checkpoint=str(checkpoint),
                source_sha256=before_hash, source_unchanged=before_hash == sha256_file(checkpoint),
                world_calls=0, optimizer_updates=0, event_credit_mode=e.event_credit_mode,
                score_scope='same pre-event local Q and actual raw event, all re-encoded in current coordinates',
                queries=[evidence_key(q) for q in qs],
                old_supports=[evidence_key(f) for f in old],
                default=selected[0], best_on_same_Q=records[0], candidates=records,
                best_minus_default=records[0]['mean_score']-selected[0]['mean_score'],
                interpretation='Tests alignment of geometric release and the current local score. '
                    'Not proof of a long-term retention objective, and not a change to live U.',
                memory_retention=dict(consolidation_enabled=model.memory.consolidation_enabled,
                    consolidation_rule=getattr(model.memory,'consolidation_rule',None),
                    read_credit_enabled=model.memory.read_credit_enabled,
                    read_credit_rule=model.memory.read_credit_rule))
        args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
        print(json.dumps(dict(cycle=loaded.cycle, event=fact_id, selected=selected[0]['released'],
            best=records[0]['released'], gap=out['best_minus_default'],
            retention=out['memory_retention'], output=str(args.output))))
    finally:
        model._causal_archive.close()


if __name__ == '__main__':
    main()
