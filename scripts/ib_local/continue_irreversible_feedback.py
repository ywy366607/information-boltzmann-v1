"""Hear the real correction once, then learn its current language likelihood.

Continue the saved actual self-heard answer. The old-answer capsule is only a
research diagnostic: no positional reference/EOS labels are assigned to it.
Current external NLL can teach old computations through the existing read path.
"""
import argparse
import json
from pathlib import Path
import time

import torch

from scripts.ib_local.irreversible_credit_stream import IrreversibleCreditStream, PretrainedStreamBackend
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner
from scripts.ib_local.pretrained_gdn_stream_load import load_streamed_pure_gdn
from scripts.ib_local.pretrained_gdn import enable_streaming_convolution
from scripts.ib_local.prepare_pretrained_gdn_ruler import sha256
from scripts.ib_local.run_irreversible_credit import read_cases
from scripts.ib_local.run_pretrained_gdn_cross_window import write_json


def correction_language(row):
    # Use the world's actual question identifier, rather than an ambiguous
    # repeated "the correct answer is..." unrelated to its source question.
    return '\nCorrection: '+row['answer_prefix'].strip()+' '+', '.join(row['outputs'])+'.\n\n'


def hear_feedback(stream, tokens, *, boundary_token=None):
    """Hear correction, then an optional world document boundary, without reset.

    The marker is an observed input, not a retroactive EOS label on an answer.
    It is not a teaching target and does not force a partial window to commit.
    """
    scores = stream.receive(tokens, phase='feedback', teach=True)
    if boundary_token is not None:
        stream.receive([boundary_token], phase='world_marker', teach=False)
    return scores


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','checkpoint','manifest','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--document-boundary', action='store_true',
                        help='Hear tokenizer EOS after the correction; preserve all state.')
    parser.add_argument('--capture-joint',action='store_true',
                        help='Archive future real windows as shared vector/joint-boundary programs.')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve previous life evidence')
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    start=time.time()
    report=dict(status='loading',source=str(args.checkpoint),source_sha256=sha256(args.checkpoint),
        protocol=__doc__,windows=[],new_lean_coverage=0)
    write_json(args.output/'report.json',report)
    source=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    pending=source.get('pending')
    if pending is None or pending.consumed or pending.answer_end_seen!=source['learner']['cache']['seen_tokens']:
        raise ValueError('expected the actual self-heard answer before external feedback')
    _,cases=read_cases(args.manifest,[2048],source['next_case']+1)
    row=cases[source['next_case']][1]
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    boundary_token = tokenizer.eos_token_id if args.document_boundary else None
    if args.document_boundary and boundary_token is None:
        raise ValueError('tokenizer has no document boundary token')
    text=correction_language(row)
    tokens=tokenizer.encode(text,add_special_tokens=False)
    model,_=load_streamed_pure_gdn(args.model)
    enable_streaming_convolution(model)
    state=source['learner']
    learner=PretrainedExperienceLearner(model,layers=state['layers'],capacity=state['capacity'])
    learner.load_state_dict(state)
    if args.capture_joint:
        learner.enable_joint_participation()
    backend=PretrainedStreamBackend(learner)
    stream=IrreversibleCreditStream(backend,on_window=report['windows'].append)
    stream.restore_latest(source['stream'])
    before_seen,before_updates,before_events=learner.cache._seen_tokens,learner.updates,stream.events
    report.update(status='running',case=source['next_case'],feedback_text=text,
                  incoming_seen=before_seen,incoming_updates=before_updates,
                  document_boundary_token=boundary_token, state_resets=0)
    write_json(args.output/'report.json',report)
    try:
        scores=hear_feedback(stream,tokens,boundary_token=boundary_token)
        backend.close()
        heard_count = len(tokens) + int(boundary_token is not None)
        if learner.cache._seen_tokens-before_seen!=heard_count or stream.events-before_events!=heard_count:
            raise AssertionError('feedback must be heard exactly once')
        torch.cuda.empty_cache()
        target=args.output/'latest.pt'
        temporary=target.with_suffix('.tmp')
        torch.save(dict(learner=learner.state_dict(),stream=stream.state_dict(),
            next_case=source['next_case']+1,pending=None,
            feedback_contract='current external language NLL',
            world_document_boundary=boundary_token),temporary)
        temporary.replace(target)
        report.update(status='complete',outgoing_seen=learner.cache._seen_tokens,
            outgoing_updates=learner.updates,retained_new_updates=learner.updates-before_updates,
            feedback_tokens=len(tokens),current_feedback_nll=sum(scores)/len(scores),
            pending_inputs=len(stream.inputs),next_case=source['next_case']+1,
            old_answer_positional_labels_used=False,source_unchanged=sha256(args.checkpoint)==report['source_sha256'])
    except BaseException as exc:
        report.update(status='failed',error=repr(exc))
        raise
    finally:
        backend.close()
        report['seconds']=time.time()-start
        write_json(args.output/'report.json',report)


if __name__=='__main__':
    main()
