"""Numerical admission of shared joint capture; no main-life continuation."""
import argparse
from pathlib import Path
import time

import torch

from scripts.ib_local.pretrained_gdn import enable_streaming_convolution
from scripts.ib_local.pretrained_gdn_experience import copy_tree,tree_nbytes
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner
from scripts.ib_local.pretrained_gdn_stream_load import load_streamed_pure_gdn
from scripts.ib_local.irreversible_credit_stream import IrreversibleCreditStream,PretrainedStreamBackend
from scripts.ib_local.run_irreversible_credit import read_cases
from scripts.ib_local.evaluate_pretrained_gdn_ruler import encode_case
from scripts.ib_local.prepare_pretrained_gdn_ruler import sha256
from scripts.ib_local.run_pretrained_gdn_cross_window import write_json,assert_same


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','checkpoint','manifest','output'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    started=time.time()
    report=dict(status='loading',retained_updates=0,source_sha256=sha256(args.checkpoint),
        source_files={f:sha256(f) for f in (__file__,'scripts/ib_local/pretrained_gdn_experience.py',
                      'scripts/ib_local/pretrained_gdn_learning.py','scripts/ib_local/irreversible_credit_stream.py')})
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    if any(payload['stream']['learn_mask']):
        raise ValueError('this numerical fixture must not settle real outstanding teaching')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    _,cases=read_cases(args.manifest,[2048],payload['next_case']+1)
    prefix,_=encode_case(tokenizer,cases[payload['next_case']][1])
    model,_=load_streamed_pure_gdn(args.model)
    enable_streaming_convolution(model)
    state=payload['learner']
    learner=PretrainedExperienceLearner(model,layers=state['layers'],capacity=state['capacity'])
    reference=None;backend=None
    try:
        for capture in (False,True):
            learner.load_state_dict(state)
            if capture:
                learner.enable_joint_participation()
            backend=PretrainedStreamBackend(learner)
            windows=[]
            stream=IrreversibleCreditStream(backend,on_window=windows.append)
            stream.restore_latest(payload['stream'])
            scores=stream.receive(prefix[:129],phase='world',teach=False)
            backend.close();backend=None
            if learner.updates!=state['updates']:
                raise AssertionError('capture admission must not update slow parameters')
            observed=dict(cache=copy_tree(learner.cache.states),stream=stream.state_dict(),scores=scores,
                memory_buffers={i:{k:copy_tree(v) for k,v in m.state_dict().items()
                                   if isinstance(v,torch.Tensor)} for i,m in learner.memories.items()})
            if not capture:
                reference=observed
                continue
            assert_same(reference,observed,'capture_forward_and_lived_state')
            capsules=[c for m in learner.memories.values() for c in m.capsules
                      if c is not None and getattr(c,'joint',None) is not None]
            if len(capsules)!=2 or capsules[0].joint is not capsules[1].joint:
                raise AssertionError('two layer entries must share one actual window archive')
            joint=capsules[0].joint
            fixture=args.output/'discarded_capsule_fixture.pt'
            torch.save(capsules,fixture)
            a,b=torch.load(fixture,map_location='cpu',weights_only=False,mmap=True)
            if a.joint is not b.joint:
                raise AssertionError('disk restore duplicated the shared archive')
            saved=learner.state_dict()
            if not saved['joint_capture_enabled']:
                raise AssertionError('capture policy not serialized')
            backend=PretrainedStreamBackend(learner)
            next_token=prefix[129:130]
            original=stream.receive(next_token,phase='world',teach=False)
            expected_cache=copy_tree(learner.cache.states)
            expected_stream=stream.state_dict()
            backend.close();backend=None
            learner.load_state_dict(saved)
            if not learner.joint_capture_enabled:
                raise AssertionError('capture policy not restored')
            backend=PretrainedStreamBackend(learner)
            restored=IrreversibleCreditStream(backend)
            restored.restore_latest(observed['stream'])
            again=restored.receive(next_token,phase='world',teach=False)
            assert_same(original,again,'next_prediction')
            assert_same(expected_cache,copy_tree(learner.cache.states),'next_native_state')
            assert_same(expected_stream,restored.state_dict(),'next_stream')
            backend.close();backend=None
            report.update(status='complete',capture_forward_and_state_exact=True,
                next_prediction_and_state_restored=True,disk_shared_archive_preserved=True,
                joint_boundary_bytes=tree_nbytes(joint.boundary),incoming_vector_bytes=tree_nbytes(joint.incoming_features),
                joint_native_layers=len(joint.boundary['states']),
                two_entries_one_archive=True,experience_cpu_bytes=learner.experience_storage_nbytes,
                source_unchanged=sha256(args.checkpoint)==report['source_sha256'],
                source_files_unchanged=all(sha256(f)==h for f,h in report['source_files'].items()),
                fixed_capacity_worst_joint_bytes=state['capacity']*tree_nbytes(joint),
                new_lean_coverage=0)
    except BaseException as exc:
        report.update(status='failed',error=repr(exc))
        raise
    finally:
        if backend is not None:
            backend.close()
        report['seconds']=time.time()-started
        write_json(args.output/'report.json',report)


if __name__=='__main__':
    main()
