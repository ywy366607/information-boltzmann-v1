"""Action-dependent contact gains versus the corresponding sequential contact control."""
import json
from pathlib import Path
from statistics import mean

from scripts.run_inquiry_navigation_trial import ROOT, OLD
from scripts.run_native_supply_comparison import digest


def read(p): return json.loads(p.read_text(encoding='utf8'))

def arm(folder):
    res=read(folder/'run/result.json'); ev=read(folder/'independent_scores.json'); old=read(folder/'run/evaluations.json')
    rows=[r for x in (folder/'run/events.jsonl').read_text(encoding='utf8').splitlines() if (r:=json.loads(x)).get('cycle')]
    assert (res['status'],res['cycle'],res['updates'],res['provider_calls'],len(rows))==('completed',2813,32,0,8)
    assert all(r['causal_use']['finite_U']['verified'] for r in rows)
    control=sum(r['world'].get('control_bytes',0) for r in rows)
    nll=ev['readings']; first=read(folder/'observations.json')['1']['action']
    return dict(root=str(folder), material=ev['material'], before=nll['0'], after=nll['8'], midway=nll['4']['mean_nll'],
        improvement=nll['0']['mean_nll']-nll['8']['mean_nll'],asks=sum(bool(r['utterance_hex']) for r in rows),
        hits=sum(r['world']['match']>0 for r in rows),moves=sum(r['world']['start']!=(r['world_before']['anchor']+r['world_before']['offset']) for r in rows),
        unique_fresh_bytes=sum(sum(r['world']['text_source_bytes'].values())-r['world']['holdout_previously_seen_bytes']-r['world']['response_previously_seen_bytes'] for r in rows),
        corpus_bytes=sum(sum(r['world']['text_source_bytes'].values()) for r in rows),control_bytes=control,
        old_fact_gain=old[-1]['mean_score']-old[0]['mean_score'], finite_U=mean(r['metal_g'] for r in rows),
        first={k:first[k] for k in ('utterance_hex','option','source_sha256','endpoint_sha256','production')},
        actions=[dict(ask=bool(r['utterance_hex']),start=r['world']['start'],term=r['world'].get('inquiry_term'),match=r['world']['match']) for r in rows],
        checkpoint_sha256=res['checkpoint_sha256'])


def main():
    rows={'old':arm(OLD/'original')}
    for k in ('inquire_terms_v1','inquire_read_control','suffix_read_control'): rows[k]=arm(ROOT/k)
    for r in rows.values():
        assert r['material']==rows['old']['material'] and r['before']==rows['old']['before']
        assert r['first']['utterance_hex']==rows['old']['first']['utterance_hex']
    def diff(active,control): return [b-a for a,b in zip(rows[active]['after']['segment_nll'],rows[control]['after']['segment_nll'])]
    contrasts={'old':diff('old','suffix_read_control'),'inquiry':diff('inquire_terms_v1','inquire_read_control')}
    did=[b-a for a,b in zip(contrasts['old'],contrasts['inquiry'])]
    out=dict(source=read(OLD/'protocol.json')['source_sha256'],arms=rows,active_advantage=contrasts,mean_advantage={k:mean(v) for k,v in contrasts.items()},difference_in_differences=did,mean_difference=mean(did))
    p=Path('results/published/cgb_inquiry_navigation_2805_2813_20261123.json'); p.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:{key:r[key] for key in ('improvement','asks','hits','moves','corpus_bytes','unique_fresh_bytes','old_fact_gain','midway')} for k,r in rows.items()})); print(contrasts, did, mean(did))

if __name__=='__main__': main()
