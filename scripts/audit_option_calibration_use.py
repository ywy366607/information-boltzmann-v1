"""Read-only accounting of completed root-lift G2 and native teacher anchors."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    def load(path): return json.loads(Path(path).read_text(encoding='utf8'))
    report = load(a.work/'report.json')
    assert load(a.work/'pipeline_status.json')['status'] == 'completed'
    source = Path(report['protocol']['source'])
    h = hashlib.sha256()
    with source.open('rb') as stream:
        for b in iter(lambda: stream.read(8*1024*1024), b''): h.update(b)
    assert h.hexdigest() == report['protocol']['source_sha256']
    root = report['root_shadow']
    rows, first = [], None
    for option in report['legal']:
        folder = a.work/('action_0_recovered' if option==0 and (a.work/'action_0_recovered').exists() else f'action_{option}')
        matching = load(folder/'matching_checks.json'); assert all(matching.values())
        events = [json.loads(s) for s in (folder/'run/events.jsonl').read_text(encoding='utf8').splitlines()]
        external = [r for r in events if r['autonomous_event']['kind']=='external']
        # External rows summarize their transaction's four updates. Handoff
        # rows describe constituent updates and are not added a second time.
        assert len(external)==2 and sum(r['updates'] for r in external)==8
        assert load(folder/'run/result.json')['updates']==8
        assert all(r['causal_use']['finite_U']['verified'] for r in external)
        if first is None:
            first=external[0]
            receipt=load(folder/'pre_outcome_action.json')
        target = report['outcomes'][str(option)]
        dp = root['alternative_probability'][option]-root['original_probability'][option]
        rows.append(dict(option=option, name=['silence','generate','reuse'][option],
            original_probability=root['original_probability'][option],
            alternative_probability=root['alternative_probability'][option],
            delta_probability=dp, G2=target['G2'], delta_C_contribution=dp*target['G2'],
            local_payment_P2=external[0]['metal_g']+report['protocol'].get('gamma', receipt['gamma'])*external[1]['metal_g'],
            self_bytes=sum(r['self_bytes'] for r in external),
            external_bytes=sum(r['external_bytes'] for r in external),
            world_starts=[r['world']['start'] for r in external],
            suffix_matches=[r['world']['match'] for r in external],
            actions=[r['choice'] for r in external], matching_checks=len(matching)))
    assert abs(sum(r['delta_C_contribution'] for r in rows)-report['delta_C']['C']) < 1e-14
    checks = first['causal_use']['native_amortization']['action_planner']['teacher_checks']
    alignment = []
    for c in checks:
        calls = [r for r in root['calls'] if r['key']==c['root_key']]
        mean = sum(r['original'] for r in calls)/len(calls)
        options = [i for i,v in enumerate(receipt['future']) if abs(v-mean)<2e-6]
        matches=[]
        for i,r in enumerate(root['calls']):
            if r['key']!=c['root_key']: continue
            for j,v in enumerate(r['persistent']):
                if abs(v-c['predicted']) < 1e-7:
                    d=r['action_plan_calibrated'][j]-r['action_plan_old'][j]
                    e=v-c['native']
                    matches.append(dict(call=i, point=j, persistent=v,
                        action_plan_old=r['action_plan_old'][j], action_plan_calibrated=r['action_plan_calibrated'][j],
                        delta=d, persistent_error=e, lifted_error=e+d,
                        teacher_mse_improvement=e*e-(e+d)**2))
        assert len(matches)==1, 'exact scalar anchor must identify a unique captured point'
        alignment.append(dict(options=options, teacher=c, match=matches[0]))
    result = dict(schema=1, report=str(a.work/'report.json'), choices=rows,
        native_teacher_alignment=alignment,
        original_C=report['original_C']['C'], connected_C=report['connected_C']['C'],
        delta_C=report['delta_C']['C'],
        original_C_retained_fraction=report['connected_C']['C']/report['original_C']['C'],
        direct_q_patch_delta_C=0, optimizer_updates=sum(8 for _ in rows),
        complete_external_rounds=2*len(rows), actual_complete_external_bytes=sum(r['external_bytes'] for r in rows),
        provider_calls=0, source_sha256=h.hexdigest(),
        decision='retain original persistent route; keep q-basis and root lift as measured research variants')
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps(result))


if __name__=='__main__': main()
