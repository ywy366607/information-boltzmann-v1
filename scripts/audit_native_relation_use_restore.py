"""Verify persistent scope snapshots and real/imagined U after full restore."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')


def assert_restored_payload(payload, restored):
    """Compare all persisted values, supplying only the three new empty defaults."""
    expected = dict(prior_scope='', prior_revision=0, prior_moments=[])
    expected.update(payload)
    assert json.dumps(restored, sort_keys=True) == json.dumps(expected, sort_keys=True), payload['scope']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--table-storage',type=Path)
    a=p.parse_args()
    import torch
    from scripts.ib_local.run_mother_continuation import continuation_parser,prepare_continuation
    from scripts.ib_local.run_causal_water_reply import sha256_file
    from scripts.ib_local import relation_retention_use as use
    path=a.work/'body/last.pt.gz'
    before=sha256_file(path)
    opts=continuation_parser().parse_args(['--checkpoint',str(path),'--output',str(path.parent),
        '--resume-output','--cycles','1','--world-source','text','--autonomous-events',
        '--optimizer-policy','budgeted','--table-storage',str(a.table_storage or a.work/'table'),
        '--causal-storage',str(a.work/'restore_causal')])
    loaded=prepare_continuation(opts,resume_output=True)
    e=loaded.engine
    try:
        schema=loaded.config['relation_retention_use'].get('schema',1)
        assert e.settlement_kind==('finite_relation_use_v5' if schema>=2 else 'finite_relation_use_v4')
        epi=loaded.config.get('epistemic_value',{}).get('schema',1)
        assert e.epistemic_value_schema==epi
        assert e.epistemic_word_execution==loaded.config.get('epistemic_value',{}).get('word_execution','historical')
        assert e.native_word_runtime==loaded.config.get('native_word_runtime',{}).get('mode','serial')
        assert getattr(e, 'successor_quadrature', None)==loaded.config.get('successor_quadrature')
        assert getattr(e, 'native_amortization', None)==loaded.config.get('native_amortization')
        rhythm = loaded.config.get('persistent_rhythm')
        if rhythm is not None:
            assert e.persistent_rhythm_schema == rhythm['schema'] == 1
            assert e.persistent_rhythm_interleave == rhythm['interleave']
            assert e._rhythm_runtime is None
            assert len(e.rhythm_state['plans']) <= e.model.causal_relations.evidence_budget
            assert all(p['packet']['raw'].device.type == 'cpu' for p in e.rhythm_state['plans'].values())
        assert e.relation_retention_use_mode==e.model.relation_retention_use_mode==loaded.config['relation_retention_use']['mode']
        scope_prior = loaded.config.get('retention_scope_prior')
        if scope_prior is not None:
            assert e.retention_scope_prior_mode == e.model.retention_scope_prior_mode == scope_prior['mode']
        stats={}
        for table in ('relation_retention','unknown_retention'):
            payloads=[json.loads(r[0]) for r in e.model._causal_archive.db.execute('SELECT payload FROM '+table)]
            assert all(len(d['moments'])<=e.model.causal_relations.evidence_budget for d in payloads)
            for d in payloads:
                restored=asdict(use._load(e.model,d['scope']))
                assert_restored_payload(d, restored)
            stats[table]=dict(scopes=len(payloads),nonzero=sum(any(x[3]>0 for x in d['moments']) for d in payloads),
                legacy_empty_prior=sum('prior_scope' not in d for d in payloads))
        seen=e.causal_use_value_evidence.seen.clone()
        use.configure(e,e.relation_retention_use_mode,schema=schema)
        assert torch.equal(seen,e.causal_use_value_evidence.seen)
        assert e.model._epistemic_value_engine is e
        out=dict(valid=True,cycle=loaded.cycle,source_sha256=before,source_unchanged=before==sha256_file(path),
            epistemic_schema=epi,
            word_execution=e.epistemic_word_execution,
            native_word_runtime=e.native_word_runtime,
            successor_quadrature=getattr(e, 'successor_quadrature', None),
            native_amortization=getattr(e, 'native_amortization', None),
            native_value_records=len(getattr(e, 'native_value_bank', {}).get('records', [])),
            latent_action_records=len(getattr(e, 'latent_action_bank', {}).get('records', [])),
            mode=e.relation_retention_use_mode,persisted=stats,action_evidence_seen=seen.tolist(),
            updates=0,world_calls=0)
        if epi>=5:
            forecast=e.internal_pending['causal_program']['decision_forecast']
            scale=forecast['scale_matrix']
            torch.linalg.cholesky(scale)
            assert scale.shape==(1,e.model.encoder.out_features,e.model.encoder.out_features)
            snapshot=e.internal_pending['causal_program']['decision_learning_state']
            assert snapshot.organization is None
            out['joint_successor']=dict(forecast_shape=list(scale.shape),
                finite_evidence=len(snapshot.constraints)+len(snapshot.extra),
                last_strata=getattr(e,'successor_distribution_last',{}).get('joint'))
        assert out['source_unchanged']
        a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps(out,ensure_ascii=False))
    finally:e.model._causal_archive.close()


if __name__=='__main__':main()
