"""Matched-exposure diagnostic: replay actual actions, omit policy gradient.

This is an externally forced replay control, not an autonomous policy. Exact
response hashes and terminal table equality isolate the neural learning update
from the corpus exposure/self-hearing confound of coupled vs randomized.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.ib_local.language_action import ActionTrainer, TextChoiceWorld, ACTION_BYTES, joint_loss, digest
from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable
from scripts.ib_local.train_gdn_fast_table import TableGDN


class ReplayTrainer(ActionTrainer):
    def _backward(self):
        logits, state, _ = self.model.predict(self.ids,self.state,self.table)
        _,loss,log_policy = joint_loss(logits,self.targets,self.action,self.baseline)
        loss.backward()  # Only direct language credit; the executed actions are exogenous.
        return loss,state,log_policy


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from scripts.ib_local import gdn_fused
    from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
    from scripts.ib_local.run_ngram_fast_table import protocol_stream
    from scripts.ib_local.train_retrieved_credit import validate_online_sidecar
    config=json.loads((args.reference/'config.json').read_text())
    if config['mode']!='coupled':raise ValueError('reference must be coupled')
    for p,h in config['sources'].items():
        if digest(p)!=h:raise ValueError(f'executed reference source changed: {p}')
    if digest(config['checkpoint'])!=config['checkpoint_sha256']:
        raise ValueError('initial checkpoint changed')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(config['seed'])
    if gdn_fused.enable()!=config['fused']:raise ValueError('fused recurrence changed')
    root=Path('data/ib_owt_10000')
    if digest(root/'train.npy')!=config['corpus_sha256']:raise ValueError('corpus changed')
    offset=json.loads((root/'tokenizer.json').read_text())['offset']
    corpus=np.load(root/'train.npy',mmap_mode='r')
    world=TextChoiceWorld(corpus,offset,config['bounds'])
    valid=np.load(root/'validation.npy',mmap_mode='r')
    buf=np.asarray(valid[:2*(config['validation_bytes']+1)])
    validation=(buf[buf>=offset]-offset)[:config['validation_bytes']+1].astype(np.uint8)
    if hashlib.sha256(validation.tobytes()).hexdigest()!=config['validation_sha256']:
        raise ValueError('validation changed')
    saved=torch.load(config['checkpoint'],map_location='cpu',weights_only=False)
    base=GatedDeltaNetLM(256,128,4,4,'decay_then_delta',256,4,32.,decay_init='log_timescale').cuda()
    model=TableGDN(base,True).cuda();model.load_state_dict(saved['model'])
    runner=ReplayTrainer(model,config['tokens'])
    runner.optimizer.load_state_dict(saved['optimizer'])
    with torch.no_grad():runner.state.copy_(saved['recurrent'])
    table=RetrievedCreditTable.from_state_dict(saved['table'])
    old=saved['config'];old_stream,_=protocol_stream(old['steps'],old['tokens'])
    if hashlib.sha256(old_stream.tobytes()).hexdigest()!=old['data_sha256']:
        raise ValueError('initial pending byte provenance changed')
    pending=int(old_stream[-1]);del old_stream
    torch.set_rng_state(saved['torch_rng'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']]);del saved
    actual=dict(config,mode='replay_no_policy',reference=str(args.reference),
                replay_source_sha256=digest(__file__),
                action_log_sha256=digest(args.reference/'metrics.jsonl'))
    (args.output/'config.json').write_text(json.dumps(actual,indent=2))
    initial=validate_online_sidecar(model,table,validation,config['tokens'])
    expected=json.loads((args.reference/'initial.json').read_text())['validation_nll']
    if initial!=expected:raise AssertionError('initial body/sidecar does not match')
    started=time.perf_counter();matched=0;loss_sum=0.;baseline=0.
    with (args.reference/'metrics.jsonl').open() as stream,(args.output/'metrics.jsonl').open('w') as log:
        for line in stream:
            row=json.loads(line);step=row['step'];tick=time.perf_counter()
            if step!=matched+1:raise ValueError('noncontiguous replay')
            action=row['action']
            if action!=row['executed']:raise ValueError('not a coupled trajectory')
            response=world.take(action,config['tokens']-1)
            if hashlib.sha256(response.tobytes()).hexdigest()!=row['response_sha256']:
                raise AssertionError('exposure differs from reference')
            raw=np.concatenate((np.array([pending,ACTION_BYTES[action]],np.uint8),response))
            _,probs,receipt=table.run_credit(raw);del receipt
            loss,policy,grad=runner.step(raw,probs,action,baseline)
            pending=int(response[-1]);baseline=.95*baseline+.05*loss
            result=dict(step=step,nll=loss,action=action,executed=action,
                        response_sha256=row['response_sha256'],policy=policy.tolist(),
                        seconds=time.perf_counter()-tick,gradient_norm=grad,
                        external_bytes=step*(config['tokens']-1),self_bytes=step)
            if step%config['validate_every']==0 or step==config['steps']:
                result['validation_nll']=validate_online_sidecar(model,table,validation,config['tokens'])
                archive=dict(model=model.state_dict(),optimizer=runner.optimizer.state_dict(),
                    recurrent=runner.state.detach().cpu(),table=table.state_dict(),
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                    step=step,config=actual,pending=pending,baseline=baseline,cursors=list(world.cursors))
                temp=args.output/'last.pending.pt';torch.save(archive,temp,pickle_protocol=5)
                temp.replace(args.output/'last.pt');del archive
            log.write(json.dumps(result)+'\n');log.flush()
            matched+=1;loss_sum+=loss
            if step%500==0:
                print(json.dumps(result),flush=True)
                (args.output/'progress.json').write_text(json.dumps(result,indent=2))
    if matched!=config['steps']:raise AssertionError('incomplete reference')
    # Verify all table contents, metadata, and world/pending cursors, not just totals.
    ref=torch.load(args.reference/'last.pt',map_location='cpu',weights_only=False)
    for name,expected_array in ref['table']['arrays'].items():
        array=getattr(table,name)
        for start in range(0,len(array),1024):
            np.testing.assert_array_equal(array[start:start+1024],expected_array[start:start+1024])
    assert int(table.history)==ref['table']['history'] and table.events==ref['table']['events']
    assert world.cursors==ref['cursors'] and pending==ref['pending']
    summary=dict(status='complete',updates=matched,initial_validation_nll=initial,
        final_validation_nll=result['validation_nll'],mean_selected_stream_nll=loss_sum/matched,
        exposure_hashes_matched=matched,terminal_table_exactly_equal=True,
        external_bytes=matched*(config['tokens']-1),self_bytes=matched,
        gpu_peak_bytes=torch.cuda.max_memory_allocated(),table_bytes=table.nbytes,
        seconds=time.perf_counter()-started,autonomous=False,organism_admitted=False)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
