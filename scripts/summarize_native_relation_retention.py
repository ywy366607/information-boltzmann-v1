"""Whole-life trace: scope -> delayed use -> release -> later real response."""
import argparse
import json
from pathlib import Path
from statistics import mean,median
import sqlite3


def dist(xs):
    return dict(min=min(xs),median=median(xs),mean=mean(xs),max=max(xs)) if xs else None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    arms,checks,actual,initial={}, {}, {}, {}
    for name in ('observe','apply'):
        folder=args.work/name
        r=json.loads((folder/'result.json').read_text(encoding='utf8'))
        meta=json.loads((folder/'body/last.metadata.json').read_text(encoding='utf8'))
        events=[json.loads(x) for x in (folder/'events.jsonl').read_text(encoding='utf8').splitlines()]
        rows=[x for x in events if x['autonomous_event']['kind']=='external']
        actual[name]=rows;initial[name]=r['evaluations'][0]
        checks[name+'.completed']=(r['status'],r['origin_cycle'],r['cycle'],r['cycles'],r['updates'],r['external_bytes'])==('completed',2481,2489,8,32,7680)
        checks[name+'.same_source']=r['initial_source_sha256']=='b574e457ba0cca18f7aed6d7a291468f8b7eb443e285cf105226878f9210916d'
        checks[name+'.mode']=meta['config']['relation_retention_use']['mode']==name
        version='finite_relation_use_v5' if meta['config']['relation_retention_use'].get('schema',1)>=2 else 'finite_relation_use_v4'
        checks[name+'.version']=all(x['settlement_kind']==version for x in rows)
        checks[name+'.paid_U']=all(x['causal_use']['finite_U'].get('verified',False) for x in rows if x['causal_use']['finite_U']['committed'])
        checks[name+'.no_paid_calls']=r['provider_calls']==0
        scopes=[];benefits=[];fresh=[]
        for x in rows:
            q=x['shared_unknown']['revision'].get('retention_use')
            org=x['causal_program'].get('retention_organization')
            for kind,item in (('unknown',q),('organization',org)):
                if item:
                    trace=item['trace']
                    if trace is None:continue
                    selection=trace.get('selection')
                    scopes.append(dict(cycle=x['cycle'],kind=kind,scope=trace['scope'],
                        consumed_revision=trace['revision'],
                        consumed_nonzero=sum(v[3]>0 for v in trace['consumed']),
                        supports=trace['retained'],released=trace['released'],
                        selection=selection,benefits=trace['benefits']))
                    benefits.extend(trace['benefits'].values())
            for kind,key in (('unknown','fresh_use'),('organization','fresh_organization')):
                fu=x['causal_use']['finite_U'].get(key)
                if fu:
                    fresh.append(dict(cycle=x['cycle'],kind=kind,**fu))
        with sqlite3.connect(Path(meta['config']['causal_archive']['snapshot']).as_uri()+'?mode=ro',uri=True) as db:
            fk=db.execute('PRAGMA foreign_key_check').fetchall()
            persisted={}
            for table in ('relation_retention','unknown_retention'):
                data=[json.loads(x[0]) for x in db.execute('SELECT payload FROM '+table)]
                checks[name+'.budget.'+table]=all(len(x['moments'])<=8 for x in data)
                persisted[table]=dict(scopes=len(data),nonzero=sum(any(v[3]>0 for v in x['moments']) for x in data),
                    moments=sum(len(x['moments']) for x in data))
            checks[name+'.foreign_keys']=not fk
        changed=[x for x in scopes if x['selection'] and
                 x['selection']['released']!=x['selection']['geometric_released']]
        later=[x for x in fresh if any(v[3]>0 for v in x['consumed'])]
        evolution=[]
        for x in scopes:
            if not x['selection']:continue
            prior=x['selection']['biases']
            evolution.append(dict(cycle=x['cycle'],scope=x['scope'],
                consumed_nonzero=x['consumed_nonzero'],released=x['released'],
                geometric=x['selection']['geometric_released'],biases=prior))
        arms[name]=dict(cycle=r['cycle'],checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],
            transition_schema=meta['config']['relation_retention_use'].get('schema',1),
            updates=r['updates'],external_bytes=r['external_bytes'],
            internal_events=sum(x['autonomous_event']['kind']=='internal' for x in events),
            scope_events=scopes,consumed_scope_events=sum(x['consumed_nonzero']>0 for x in scopes),
            changed_release_events=changed,competition=evolution,
            observed_contributions=dist(benefits),positive=sum(x>0 for x in benefits),negative=sum(x<0 for x in benefits),
            pre_response_frozen=fresh,later_frozen_on_used_scope=later,
            fresh_scores=dist([x['score'] for x in fresh]),later_used_scope_scores=dist([x['score'] for x in later]),
            fresh_by_kind={kind:dict(count=sum(x.get('kind')==kind for x in fresh),
                scores=dist([x['score'] for x in fresh if x.get('kind')==kind]),
                later_count=sum(x.get('kind')==kind for x in later),
                later_scores=dist([x['score'] for x in later if x.get('kind')==kind]))
                for kind in ('unknown','organization')},
            actual_U=dist([x['metal_g'] for x in rows]),persisted=persisted,
            old_queries={label:{metric:dist([q[metric] for q in ev['queries']]) for metric in ('score','mse')}
                for label,ev in (('before',r['evaluations'][0]),('after',r['evaluations'][-1]))},
            seconds=r['invocation']['seconds'])
    checks['same_initial_old_queries']=initial['observe']==initial['apply']
    checks['same_transition_schema']=arms['observe']['transition_schema']==arms['apply']['transition_schema']
    out=dict(valid=all(checks.values()),checks=checks,arms=arms,
        transition_schema=arms['apply']['transition_schema'],
        same_outputs={k:all(x[k]==y[k] for x,y in zip(actual['observe'],actual['apply']))
            for k in ('choice','choice_probability','utterance_hex','world','metal_g')},
        objective='same finite-U held-out-query score; real MOMENT evidence is scope local and delayed',
        fresh_measurement='full prediction and encoder frozen before new response, distinct from update benefits',
        source_budget='eight external events per arm, four updates per event, four-event cross-process resume')
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    assert out['valid'],[k for k,v in checks.items() if not v]
    print(json.dumps(dict(valid=True,checks=len(checks),arms={k:dict(
        consumed=v['consumed_scope_events'],changed=len(v['changed_release_events']),
        later=len(v['later_frozen_on_used_scope']),old=v['old_queries']['after']) for k,v in arms.items()}),ensure_ascii=False))


if __name__=='__main__':main()
