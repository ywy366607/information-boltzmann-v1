"""Read-only system summary; preserve channel units and actual source exposure."""
import argparse
import json
import math
import statistics
from pathlib import Path


def mean(values):
    values = [v for v in values if isinstance(v,(int,float)) and math.isfinite(v)]
    return statistics.fmean(values) if values else None


def paired(rows, first, second):
    values = [(first(r),second(r)) for r in rows]
    values = [(a,b) for a,b in values if isinstance(a,(int,float)) and isinstance(b,(int,float))
              and math.isfinite(a) and math.isfinite(b)]
    return dict(events=len(values), real_error=mean(a for a,b in values),
                control_error=mean(b for a,b in values),
                better=sum(a<b for a,b in values), worse=sum(a>b for a,b in values))


def channels(rows, key):
    values = [r[key] for r in rows if isinstance(r.get(key),list) and len(r[key]) == 4]
    return [mean(v[i] for v in values) for i in range(4)] if values else None


def unknown_lineage(rows):
    """Audit ordered real-use receipts, not pending historical snapshots.

    A gap is between observed SELECTED uses; dependencies may use the same
    unknown in between. A bounded monitoring window has no claim about earlier
    history. Apply an event's revision only after checking that event's reads.
    ``seen_version`` records edge registration, not the version actually read.
    """
    known, selected, dependent_relations = {}, {}, {}
    regressions, revisits, crossings = [], [], 0
    for row in rows:
        cycle = row['cycle']
        exposure = row.get('since_birth_external_bytes')
        question = row.get('shared_unknown') or {}
        identity = question.get('identity')
        relation = question.get('relation_identity')
        if identity is not None:
            version = question['version']
            if identity in known and version < known[identity]:
                regressions.append(dict(cycle=cycle, identity=identity, port='selected',
                                        actual_version=version, prior_revision=known[identity]))
            if identity in selected:
                old = selected[identity]
                gap = (exposure-old['exposure'] if exposure is not None and
                       old['exposure'] is not None else None)
                revisits.append(dict(identity=identity, cycle=cycle,
                    previous_selected_cycle=old['cycle'], selected_gap_external_bytes=gap,
                    relation_before=old['relation'], relation_now=relation,
                    actual_version=version, prior_revision=known.get(identity)))
            selected[identity] = dict(cycle=cycle, exposure=exposure, relation=relation)
        for use in (row.get('joint_relation_update') or {}).get('variable_uses', []):
            uid, version = use['identity'], use['version']
            if uid in known and version < known[uid]:
                regressions.append(dict(cycle=cycle, identity=uid, port='dependent',
                                        actual_version=version, prior_revision=known[uid]))
            relations = dependent_relations.setdefault(uid, set())
            if relation is not None:
                crossings += bool(relations and relation not in relations)
                relations.add(relation)
        revision = question.get('revision') or {}
        if revision.get('committed'):
            known[identity] = revision['version_after']
    longest = sorted((x for x in revisits if x['selected_gap_external_bytes'] is not None),
                     key=lambda x: x['selected_gap_external_bytes'], reverse=True)[:3]
    return dict(scope='current logged window; real selected/dependent reads versus prior real revisions',
        capability_claim=False, observed_selected_reactivations=len(revisits),
        observed_cross_relation_dependents=crossings, version_regressions=regressions,
        longest_selected_gaps=longest,
        interpretation='selected-use gaps are not inactivity durations; version consistency is not predictive benefit')


def summary(rows):
    credits = [v for r in rows for v in r.get('read_credit',{}).values() if isinstance(v,dict)]
    def internal(r): return r.get('internal_rollout',{})
    def settled(r): return internal(r).get('settled_two_turn') or {}
    result = dict(events=len(rows), shared_unknown_lineage=unknown_lineage(rows),
        source_bytes={k:sum(r['world']['text_source_bytes'][k] for r in rows)
        for k in ('inherited_prefix','natural_append')},
        choices={k:sum(r['choice']==k for r in rows) for k in ('silence','generate','reuse')},
        real_next_boundary_vs_unchanged=paired(rows,
            lambda r:internal(r).get('next_boundary_prequential_mse'),
            lambda r:internal(r).get('unchanged_state_prequential_mse')),
        real_next_boundary_vs_other_action=paired(rows,
            lambda r:internal(r).get('next_boundary_prequential_mse'),
            lambda r:internal(r).get('other_action_prequential_mse')),
        next_return_vs_base=paired([r for r in rows if settled(r).get('eligible')],
            lambda r:(settled(r)['prediction']-settled(r)['target'])**2,
            lambda r:(settled(r)['baseline']-settled(r)['target'])**2),
        channel_units=['head4_predictability','head32_predictability','joined','learning_progress'],
        wood_used_error=channels(rows,'wood_used_error'), wood_base_error=channels(rows,'wood_base_error'),
        episodic_real_error=channels(rows,'episodic_real_error'),
        episodic_shuffled_error=channels(rows,'episodic_shuffled_error'),
        internal_choice_tv=mean(internal(r).get('choice_probability_tv') for r in rows),
        reused_unknown_events=sum(bool(r.get('shared_unknown',{}).get('recalled')) for r in rows),
        unknown_real_revisions=sum(bool(r.get('shared_unknown',{}).get('revision',{}).get('committed')) for r in rows),
        actual_read_credit_records=sum(v.get('records',0) for v in credits),
        actual_read_credit_participations=sum(v.get('participations',0) for v in credits),
        stale_read_credits=sum(v.get('stale',0) for v in credits),
        question_transition_steps=mean(r.get('question_context',{}).get('conditional_transition',{}).get('parameter_step') for r in rows),
        delayed_actor_events=sum(bool(r.get('delayed_actor_applied')) for r in rows),
        expression_mean_bytes=mean(r.get('self_bytes') for r in rows),
        budget_stop_events=sum(r.get('expression',{}).get('reason')=='budget' for r in rows),
        coevolution_valid=mean(r['past_present']['valid'] for r in rows),
        coevolution_retained=mean(r['past_present']['retained'] for r in rows),
        coevolution_rms=mean(r['past_present']['workspace_rms'] for r in rows))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a = p.parse_args()
    state = json.loads((a.directory/'status.json').read_text())
    result = dict(schema=1, pipeline_status=state['status'], main_written=False,
        scope='prequential predictions and executed loop paths; no scalar pooling across incompatible outcome channels', arms={})
    for arm in ('budgeted','adamw','global_anneal'):
        path = a.directory/arm/'metrics.jsonl'
        if not path.exists(): continue
        rows = []
        for line in path.read_text().splitlines():
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: break
        phases = {phase:summary([r for r in rows if r['phase']==phase])
                  for phase in dict.fromkeys(r['phase'] for r in rows)}
        result['arms'][arm] = dict(events=len(rows), phases=phases,
            raw_report=str(a.directory/arm/'report.json'))
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:dict(events=v['events'],phases=list(v['phases'])) for k,v in result['arms'].items()}))


if __name__ == '__main__': main()
