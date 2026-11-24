"""Actual complete restore reattaches the experimental value route by schema."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import torch
    from scripts.ib_local.run_mother_continuation import continuation_parser,prepare_continuation
    from scripts.ib_local.epistemic_value_reference import branch_value
    from scripts.run_native_relation_mode_use import sha
    path=(a.work/'body/last.pt.gz').resolve()
    before=sha(path)
    opts=continuation_parser().parse_args(['--checkpoint',str(path),'--output',str(path.parent),
        '--cycles','1','--resume-output','--world-source','text','--autonomous-events',
        '--optimizer-policy','budgeted','--table-storage',str(a.work/'table'),
        '--causal-storage',str(a.work/'restore_causal')])
    loaded=prepare_continuation(opts,resume_output=True)
    e=loaded.engine
    try:
        assert e.epistemic_value_schema==1
        assert e.model._epistemic_value_engine is e
        assert loaded.config['epistemic_value']['schema']==1
        assert e.model.identity_preview_revision==1
        seen=e.causal_use_value_evidence.seen.clone()
        with torch.no_grad():
            z=e.model.encoder(e._internal_source.reshape(1,-1))
            value=branch_value(e.model,e._program_question_state,z)
        assert torch.isfinite(value).all()
        assert torch.equal(seen,e.causal_use_value_evidence.seen)
        event_mode = loaded.config.get('event_credit', {}).get('mode')
        if event_mode is not None:
            assert e.event_credit_mode == event_mode
            assert e.settlement_kind == 'finite_event_targeted_use_v3'
        result=dict(passed=True,cycle=loaded.cycle,source=str(path),source_sha256=before,
            event_credit_mode=event_mode, settlement_kind=e.settlement_kind,
            unchanged=before==sha(path),engine_schema=e.epistemic_value_schema,
            config_schema=loaded.config['epistemic_value']['schema'],
            runtime_engine_identity=True,identity_preview=e.model.identity_preview_revision,
            action_evidence_seen=seen.tolist(),restored_value=value.tolist(),
            optimizer_updates=0,world_calls=0)
        assert result['unchanged']
        a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps(result,ensure_ascii=False))
    finally:
        e.model._causal_archive.close()


if __name__=='__main__':
    main()
