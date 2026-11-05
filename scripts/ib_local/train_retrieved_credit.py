"""Controlled extension of the existing TableGDN, not a new language body.

Run as python -m scripts.ib_local.train_retrieved_credit. Off/aligned/shuffled
share the same table representation and read-credit workspaces. Full snapshots
include occupancy/owners, neural memory, optimizer and RNG. No auto-deletion.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.ib_local import gdn_fused
from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable
from scripts.ib_local.run_ngram_fast_table import protocol_stream
from scripts.ib_local.train_gdn_fast_table import TableGDN, TableGraphTrainer, FLOOR


class CreditGraphTrainer(TableGraphTrainer):
    """Same loss/optimizer as the existing trainer, plus the actual table-input VJP."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.graph
        self.table.requires_grad_(True)
        self.table.grad = torch.zeros_like(self.table)
        saved = self.state.clone()

        def backward():
            self.table.grad.zero_()
            loss, state, diag = self.model(self.ids, self.targets, self.state, self.table)
            (loss / (self.tokens // self.chunk_tokens)).backward()
            return loss, state, diag

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.optimizer.zero_grad(set_to_none=False)
                backward()
        torch.cuda.current_stream().wait_stream(stream)
        self.optimizer.zero_grad(set_to_none=False)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.loss, self.next_state, self.diagnostics = backward()
        with torch.no_grad():
            self.state.copy_(saved)
        self.optimizer.zero_grad(set_to_none=False)

    def step_credit(self, ids, targets, logp):
        self.optimizer.zero_grad(set_to_none=False)
        losses, derivatives = [], []
        for start in range(0, self.tokens, self.chunk_tokens):
            end = start + self.chunk_tokens
            with torch.no_grad():
                self.ids.copy_(ids[:, start:end])
                self.targets.copy_(targets[:, start:end])
                self.table.copy_(logp[:, start:end])
            self.graph.replay()
            losses.append(self.loss.detach().clone())
            # backward uses mean loss across the complete optimizer window.
            derivatives.append(self.table.grad.detach().clone() * self.tokens)
            with torch.no_grad():
                self.state.copy_(self.next_state.detach())
        norms = torch.stack([torch.nn.utils.clip_grad_norm_(g, 1., foreach=True)
                             for g in self.groups])
        if not torch.isfinite(norms).all():
            raise FloatingPointError('nonfinite neural gradient')
        self.optimizer.step()
        return torch.stack(losses).mean(), torch.cat(derivatives, dim=1)


def save_snapshot(path, runner, table, rng, step, config):
    state = dict(model=runner.model.state_dict(), optimizer=runner.optimizer.state_dict(),
                 recurrent=runner.state.detach().cpu(), table=table.state_dict(),
                 numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all(), step=step, config=config)
    temporary = path.with_suffix('.pending.pt')
    # Protocol 5 writes NumPy buffers without protocol-2's large byte/string copies.
    torch.save(state, temporary, pickle_protocol=5)
    temporary.replace(path)


def restore_snapshot(snapshot, runner, rng):
    runner.model.load_state_dict(snapshot['model'])
    runner.optimizer.load_state_dict(snapshot['optimizer'])
    with torch.no_grad():
        runner.state.copy_(snapshot['recurrent'])
    rng.bit_generator.state = snapshot['numpy_rng']
    torch.set_rng_state(snapshot['torch_rng'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in snapshot['cuda_rng']])
    return RetrievedCreditTable.from_state_dict(snapshot['table'])


@torch.no_grad()
def validate_online_sidecar(model, table, data, tokens):
    """Weights fixed; cloned table assimilates validation bytes AFTER scoring.

    No count-credit updates here. Explicitly a common sidecar metric, not an
    estimate of the full actively learning individual's future performance.
    Does not alter training table, neural state, optimizer, or training cursor.
    """
    clone = RetrievedCreditTable.from_state_dict(table.state_dict())
    clone.history = np.uint64(0)  # independent document boundary for table contexts
    s = model.initial_state(1, device='cuda')
    total = 0.
    for i in range(0, len(data)-1, tokens):
        raw = data[i:min(i+tokens+1,len(data))]
        _, probs, _ = clone.run_credit(raw, max_window=tokens)
        ids = torch.tensor(raw[:-1].astype(np.int64),device='cuda')[None]
        targets = torch.tensor(raw[1:].astype(np.int64),device='cuda')[None]
        logp = torch.tensor(np.log(np.maximum(probs,FLOOR)),device='cuda')[None]
        loss,s,_ = model(ids,targets,s,logp)
        total += float(loss)*(len(raw)-1)
    return total/(len(data)-1)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['off','aligned','shuffled'],required=True)
    parser.add_argument('--steps',type=int,default=5000)
    parser.add_argument('--tokens',type=int,default=512)
    parser.add_argument('--chunk-tokens',type=int,default=512)
    parser.add_argument('--slots-log2',type=int,default=14)
    parser.add_argument('--seed',type=int,default=11)
    parser.add_argument('--validate-every',type=int,default=500)
    parser.add_argument('--validation-bytes',type=int,default=32768)
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    if not 0<args.tokens<=512:raise ValueError('declared receipt budget is 1 to 512 events')
    if min(args.steps,args.chunk_tokens,args.validate_every,args.validation_bytes)<=0:
        raise ValueError('step, chunk and validation budgets must be positive')
    args.output.mkdir(parents=True,exist_ok=args.resume is not None)
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed+1000)
    fused_hash=gdn_fused.enable()
    data,phases=protocol_stream(args.steps,args.tokens)
    root=Path('data/ib_owt_10000')
    offset=json.loads((root/'tokenizer.json').read_text())['offset']
    validraw=np.load(root/'validation.npy',mmap_mode='r')
    raw=np.asarray(validraw[:2*(args.validation_bytes+1)])
    validation=(raw[raw>=offset]-offset)[:args.validation_bytes+1].astype(np.uint8)
    if len(validation)!=args.validation_bytes+1:raise ValueError('insufficient validation bytes')
    config=dict(seed=args.seed,mode=args.mode,steps=args.steps,tokens=args.tokens,
        chunk_tokens=args.chunk_tokens,slots_log2=args.slots_log2,phases=phases,
        data_sha256=hashlib.sha256(data.tobytes()).hexdigest(),
        validation_sha256=hashlib.sha256(validation.tobytes()).hexdigest(),
        fused_sha256=fused_hash,
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                 [Path(__file__),Path('scripts/ib_local/retrieved_table_credit.py'),
                  Path('scripts/ib_local/train_gdn_fast_table.py'),
                  Path('scripts/ib_local/gated_deltanet.py')]},
        claim='local delayed retrieved-content credit; not original-write gradient',
        validation='cold neural/context sidecar with trained table clone, online assimilation, frozen weights and no count credit')
    base=GatedDeltaNetLM(256,128,4,4,'decay_then_delta',256,4,32.,decay_init='log_timescale').cuda()
    model=TableGDN(base,True).cuda()
    runner=CreditGraphTrainer(model,args.tokens,args.chunk_tokens)
    table=RetrievedCreditTable(slots_log2=args.slots_log2)
    start_step=0
    if args.resume:
        snapshot=torch.load(args.resume,map_location='cpu',weights_only=False)
        for k in ('seed','mode','steps','tokens','chunk_tokens','slots_log2','data_sha256','validation_sha256','sources'):
            if snapshot['config'][k]!=config[k]:raise ValueError(f'resume mismatch: {k}')
        table=restore_snapshot(snapshot,runner,rng);start_step=snapshot['step']
        del snapshot
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    warm=RetrievedCreditTable(slots_log2=4)
    warm.run_credit(data[:33]);del warm
    inputs=torch.tensor(data.astype(np.int64),device='cuda')[None]
    began=time.perf_counter()
    logpath=args.output/'metrics.jsonl'
    if args.resume:
        retained=[json.loads(line) for line in logpath.read_text().splitlines()
                  if line.strip()]
        retained=[row for row in retained if row['step']<=start_step]
        if [row['step'] for row in retained]!=list(range(1,start_step+1)):
            raise ValueError('resume metrics do not cover checkpoint cursor')
        # A crash may leave log rows newer than the last atomic checkpoint.
        logpath.write_text(''.join(json.dumps(row)+'\n' for row in retained))
    snapshot_seconds=0.
    with logpath.open('a' if args.resume else 'w',encoding='utf-8') as log:
        for step in range(start_step,args.steps):
            i=step*args.tokens
            tick=time.perf_counter()
            _,probs,packet=table.run_credit(data[i:i+args.tokens+1])
            read_done=time.perf_counter()
            logp=torch.tensor(np.log(np.maximum(probs,FLOOR)),device='cuda')[None]
            loss,gradient=runner.step_credit(inputs[:,i:i+args.tokens],inputs[:,i+1:i+args.tokens+1],logp)
            third=gradient[0].cpu().numpy()
            third[probs<=FLOOR]=0  # derivative of the actual clamp-to-log interface
            neural_done=time.perf_counter()
            receipt=table.settle(packet,third,mode=args.mode,rng=rng)
            phase=next(p['name'] for p in phases if p['start_update']<=step<p['stop_update'])
            row=dict(step=step+1,phase=phase,nll=float(loss),**receipt,
                     table_seconds=read_done-tick,
                     neural_seconds=neural_done-read_done,
                     settlement_seconds=time.perf_counter()-neural_done)
            checkpoint_due=(step+1)%args.validate_every==0 or step+1==args.steps
            if checkpoint_due:
                check_start=time.perf_counter()
                row['validation_nll']=validate_online_sidecar(model,table,validation,args.tokens)
                row['validation_seconds']=time.perf_counter()-check_start
            log.write(json.dumps(row)+'\n');log.flush()
            if checkpoint_due:
                check_start=time.perf_counter()
                save_snapshot(args.output/'last.pt',runner,table,rng,step+1,config)
                snapshot_seconds+=time.perf_counter()-check_start
            if (step+1)%50==0 or step==start_step:
                status=dict(status='running',**row,table_bytes=table.nbytes,
                            receipt_bytes=packet.nbytes,
                            bytes_per_second=(step+1-start_step)*args.tokens/(time.perf_counter()-began),
                            gpu_peak_bytes=torch.cuda.max_memory_allocated())
                (args.output/'progress.json').write_text(json.dumps(status,indent=2))
    (args.output/'progress.json').write_text(json.dumps(dict(status='complete',step=args.steps)))
    rows=[json.loads(x) for x in logpath.read_text().splitlines()]
    revisit=next(p for p in phases if p['name']=='A_revisit')
    original=next(p for p in phases if p['name']=='A_first')
    first_start=original['start_update']+(revisit['source_byte_start']-original['source_byte_start'])//args.tokens
    # Drop the last whole update in each interval: its final target is a different
    # phase bridge. The remaining next-byte pairs are exactly the same source bytes.
    count=revisit['stop_update']-revisit['start_update']-1
    aligned_revisit=None
    if count>0:
        first=np.mean([r['nll'] for r in rows[first_start:first_start+count]])
        later=np.mean([r['nll'] for r in rows[revisit['start_update']:revisit['start_update']+count]])
        aligned_revisit=dict(first=float(first),revisit=float(later),savings=float(first-later),
                             matched_bytes=count*args.tokens,excluded_boundary_updates=1)
    summary=dict(config=config,phase_nll={p['name']:float(np.mean([r['nll'] for r in rows if r['phase']==p['name']])) for p in phases},
                 final_validation=rows[-1].get('validation_nll'),table_bytes=table.nbytes,
                 aligned_revisit=aligned_revisit,snapshot_seconds_this_session=snapshot_seconds,
                 receipt_bytes=packet.nbytes,preloaded_input_bytes=inputs.numel()*inputs.element_size(),
                 parameters=sum(p.numel() for p in model.parameters()),
                 gpu_peak_bytes=torch.cuda.max_memory_allocated(),
                 gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                 measured_seconds=time.perf_counter()-began,
                 timing_seconds={k:sum(r.get(k,0.) for r in rows) for k in
                     ('table_seconds','neural_seconds','settlement_seconds','validation_seconds')},
                 old_reads=sum(r['reads_older_than_window'] for r in rows),
                 gross_exchange=sum(r['gross_count_exchange'] for r in rows),
                 max_conservation_error=max(r['pool_total_error'] for r in rows),
                 numerical_or_capability='budget_limited_real_language_run',organism_admitted=False)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
