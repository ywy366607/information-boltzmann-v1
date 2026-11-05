"""Same-body language action bridge; no new neural parameters or latch.

Two existing decoder bytes select two disjoint real-text streams. A self-produced
command is heard, but is excluded from external-language scoring. The joint loss
retains both the direct language derivative and the sampled policy-score term.
All derivatives condition on the detached body at this bounded episode boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numba
import numpy as np
import torch
import torch.nn.functional as F

from scripts.ib_local.ngram_fast_table import MASK64, _mix
from scripts.ib_local.retrieved_table_credit import RetrievedCreditTable
from scripts.ib_local.train_gdn_fast_table import TableGDN, FLOOR, LEXICAL

ACTION_BYTES = (48, 49)  # An explicit two-command environment alphabet, not semantic labels.


@numba.njit(cache=True)
def _peek(byte, history, rows, count, tag, err, orders, heads, slots,
          prior_weight, err_floor):
    history = ((history << np.uint64(8)) | np.uint64(byte)) & MASK64
    p = np.full(256, prior_weight / 256, dtype=np.float64)
    total = prior_weight
    for o in range(len(orders)):
        n = orders[o]
        key = history & ((np.uint64(1) << np.uint64(8*n))-np.uint64(1)) if n < 8 else history
        for h in range(heads):
            code = _mix(key ^ _mix(np.uint64(o*131+h*7919+1)))
            slot = (o*heads+h)*slots + np.int64(code & np.uint64(slots-1))
            fingerprint = np.uint16(((code >> np.uint64(40)) & np.uint64(0xFFFF)) | np.uint64(1))
            if tag[slot] == fingerprint and count[slot] > 0:
                c = count[slot]
                w = c/(c+1)/(err[slot]+err_floor)
                total += w
                for j in range(256):
                    p[j] += w*rows[slot,j]
    return (p/total).astype(np.float32)


def peek(table, byte):
    """Read the pending boundary byte without writing a fabricated target."""
    return _peek(byte, np.uint64(table.history), table.rows, table.count, table.tag,
                 table.err, np.asarray(table.orders, np.int64), table.heads,
                 1 << table.slots_log2, table.prior_weight, table.err_floor)


class TextChoiceWorld:
    """Two disk-backed streams, bounded reads and two cursors; no future previews.

    Exhaustion raises rather than silently recycling text as fresh experience.
    Raw offsets include tokenizer special tokens, which are skipped like the
    existing byte training protocol. These are corpus regions, not topic labels.
    """
    def __init__(self, raw, offset, bounds, cursors=None):
        self.raw, self.offset = raw, int(offset)
        self.bounds = tuple(tuple(map(int, b)) for b in bounds)
        if len(self.bounds) != 2:
            raise ValueError('exactly two action ports required')
        for lo, hi in self.bounds:
            if not 0 <= lo < hi <= len(raw):
                raise ValueError('invalid corpus bounds')
        if max(self.bounds[0][0], self.bounds[1][0]) < min(self.bounds[0][1], self.bounds[1][1]):
            raise ValueError('streams must be disjoint')
        self.cursors = list(cursors if cursors is not None else [b[0] for b in self.bounds])
        if any(not lo <= c <= hi for c, (lo, hi) in zip(self.cursors, self.bounds)):
            raise ValueError('invalid world cursor')

    def take(self, action, n):
        if action not in (0, 1) or n <= 0:
            raise ValueError('invalid action or exposure budget')
        cursor, end = self.cursors[action], self.bounds[action][1]
        result = np.empty(n, np.uint8)
        filled = 0
        while filled < n:
            if cursor == end:
                raise EOFError('real-text source exhausted')
            block = self.raw[cursor:min(end, cursor + 2*(n-filled)+32)]
            valid = np.flatnonzero(block >= self.offset)
            take = min(n-filled, len(valid))
            if take:
                values = block[valid[:take]] - self.offset
                if np.any(values > 255):
                    raise ValueError('not a byte corpus')
                result[filled:filled+take] = values
                cursor += int(valid[take-1])+1
                filled += take
            else:
                cursor += len(block)
        self.cursors[action] = cursor
        return result


def joint_loss(logits, targets, action, baseline):
    """Unbiased sampled derivative of conditional expected response NLL.

    Self-hearing is input position 1; target position 0 is our own command and
    receives no external CE. This is not a derivative through previous episodes
    or through future parameter updates. The baseline predates the action.
    """
    nll = F.cross_entropy(logits[:,1:].reshape(-1,256), targets[:,1:].reshape(-1))
    log_policy = logits[:,0,ACTION_BYTES[0]:ACTION_BYTES[-1]+1].log_softmax(-1)
    selected = log_policy.gather(1, action.reshape(1,1)).mean()
    return nll + (nll.detach()-baseline.detach())*selected, nll, log_policy


class ActionTrainer:
    """Two CUDA graphs: read-only decision, then the bounded joint update."""
    def __init__(self, model, tokens=512, capture=True):
        if tokens < 2 or tokens > 512:
            raise ValueError('2..512 boundary events per episode')
        self.model, self.tokens = model, tokens
        device = next(model.parameters()).device
        self.state = model.initial_state(1, device=device)
        self.ids = torch.zeros((1,tokens), dtype=torch.long, device=device)
        self.targets = torch.zeros_like(self.ids)
        self.table = torch.full((1,tokens,256), np.log(1/256), dtype=torch.float32, device=device)
        self.action = torch.zeros((), dtype=torch.long, device=device)
        self.baseline = torch.zeros((), device=device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, foreach=True)
        named = list(model.named_parameters())
        lexical = lambda n: n.removeprefix('base.') in LEXICAL
        self.groups = ([p for n,p in named if lexical(n)], [p for n,p in named if not lexical(n)])
        self.capture = capture
        if capture:
            if device.type != 'cuda':
                raise ValueError('CUDA graph requires CUDA')
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.optimizer.zero_grad(set_to_none=False)
                    self._backward()
                    self._decision()
            torch.cuda.current_stream().wait_stream(stream)
            self.optimizer.zero_grad(set_to_none=False)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.loss, self.next_state, self.log_policy = self._backward()
            self.decision_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.decision_graph):
                self.decision_policy = self._decision()
            self.optimizer.zero_grad(set_to_none=False)

    @torch.no_grad()
    def _decision(self):
        logits, _, _ = self.model.predict(self.ids[:,:1], self.state, self.table[:,:1])
        return logits[0,0,ACTION_BYTES[0]:ACTION_BYTES[-1]+1].softmax(-1)

    def _backward(self):
        logits, state, _ = self.model.predict(self.ids, self.state, self.table)
        objective, loss, log_policy = joint_loss(logits, self.targets, self.action, self.baseline)
        objective.backward()
        return loss, state, log_policy

    def decide(self, pending, table_prob):
        with torch.no_grad():
            self.ids[0,0] = pending
            self.table[:,:1].copy_(torch.as_tensor(np.log(np.maximum(table_prob,FLOOR)),
                                                 device=self.table.device)[None,None])
        if self.capture:
            self.decision_graph.replay()
            return self.decision_policy.detach().cpu().numpy().astype(np.float64)
        return self._decision().cpu().numpy().astype(np.float64)

    def step(self, raw, probs, action, baseline):
        with torch.no_grad():
            self.ids.copy_(torch.as_tensor(raw[:-1].astype(np.int64),device=self.ids.device)[None])
            self.targets.copy_(torch.as_tensor(raw[1:].astype(np.int64),device=self.ids.device)[None])
            self.table.copy_(torch.as_tensor(np.log(np.maximum(probs,FLOOR)),device=self.ids.device)[None])
            self.action.fill_(action)
            self.baseline.fill_(baseline)
        self.optimizer.zero_grad(set_to_none=False)
        if self.capture:
            self.graph.replay()
            loss, state, policy = self.loss, self.next_state, self.log_policy
        else:
            loss, state, policy = self._backward()
        norms = torch.stack([torch.nn.utils.clip_grad_norm_(g,1.,foreach=True) for g in self.groups])
        if not torch.isfinite(norms).all() or not torch.isfinite(loss):
            raise FloatingPointError('nonfinite update')
        self.optimizer.step()
        with torch.no_grad():
            self.state.copy_(state.detach())
        return float(loss.detach()), policy.detach().exp().cpu().numpy()[0], float(norms.norm())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('coupled','randomized'), required=True)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--tokens', type=int, default=512)
    parser.add_argument('--validate-every', type=int, default=500)
    parser.add_argument('--validation-bytes', type=int, default=32768)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.steps,args.validate_every,args.validation_bytes) <= 0:
        raise ValueError('positive budgets required')
    from scripts.ib_local import gdn_fused
    from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
    from scripts.ib_local.run_ngram_fast_table import protocol_stream
    from scripts.ib_local.train_retrieved_credit import validate_online_sidecar
    args.output.mkdir(parents=True, exist_ok=args.resume)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    fused = gdn_fused.enable()
    root = Path('data/ib_owt_10000')
    offset = json.loads((root/'tokenizer.json').read_text())['offset']
    corpus = np.load(root/'train.npy', mmap_mode='r')
    world = TextChoiceWorld(corpus, offset, ((8_000_000,24_000_000),(24_000_000,len(corpus))))
    valid = np.load(root/'validation.npy', mmap_mode='r')
    buf = np.asarray(valid[:2*(args.validation_bytes+1)])
    validation = (buf[buf>=offset]-offset)[:args.validation_bytes+1].astype(np.uint8)
    if len(validation) != args.validation_bytes+1:
        raise ValueError('insufficient independent validation')
    sources = [Path(__file__), Path('scripts/ib_local/gated_deltanet.py'),
               Path('scripts/ib_local/train_gdn_fast_table.py'),
               Path('scripts/ib_local/retrieved_table_credit.py'),
               Path('scripts/ib_local/train_retrieved_credit.py'),
               Path('scripts/ib_local/ngram_fast_table.py')]
    config = dict(mode=args.mode, seed=args.seed, steps=args.steps,tokens=args.tokens,
                  validate_every=args.validate_every,validation_bytes=args.validation_bytes,
                  checkpoint=str(args.checkpoint),checkpoint_sha256=digest(args.checkpoint),
                  corpus_sha256=digest(root/'train.npy'),bounds=world.bounds,
                  validation_sha256=hashlib.sha256(validation.tobytes()).hexdigest(),
                  sources={str(p):digest(p) for p in sources},fused=fused,
                  score='mean external response NLL; own command excluded',baseline_decay=.95,
                  claim='conditional finite-window action bridge; not survival or long-horizon credit')
    path = args.output/'last.pt' if args.resume else args.checkpoint
    saved = torch.load(path,map_location='cpu',weights_only=False)
    base = GatedDeltaNetLM(256,128,4,4,'decay_then_delta',256,4,32.,decay_init='log_timescale').cuda()
    model = TableGDN(base,True).cuda()
    model.load_state_dict(saved['model'])
    runner = ActionTrainer(model,args.tokens)
    runner.optimizer.load_state_dict(saved['optimizer'])
    with torch.no_grad():
        runner.state.copy_(saved['recurrent'])
    table = RetrievedCreditTable.from_state_dict(saved['table'])
    rng = np.random.default_rng(args.seed+2000)
    start, baseline = 0, 0.
    if args.resume:
        # JSON-normalize tuples for identical contracts on disk and in memory.
        if json.loads(json.dumps(config)) != json.loads(json.dumps(saved['config'])):
            raise ValueError('resume contract mismatch')
        start, pending, baseline = saved['step'],saved['pending'],saved['baseline']
        world.cursors[:] = saved['cursors']
        rng.bit_generator.state = saved['numpy_rng']
    else:
        old = saved['config']
        if old['mode'] != 'off' or saved['step'] != old['steps']:
            raise ValueError('requires completed no-feedback baseline')
        old_stream, _ = protocol_stream(old['steps'],old['tokens'])
        if hashlib.sha256(old_stream.tobytes()).hexdigest() != old['data_sha256']:
            raise ValueError('cannot recover pending baseline byte')
        pending = int(old_stream[-1])
        del old_stream
    torch.set_rng_state(saved['torch_rng'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
    del saved
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    if not args.resume:
        initial = dict(validation_nll=validate_online_sidecar(model,table,validation,args.tokens),
                       parameters=sum(p.numel() for p in model.parameters()),
                       recurrent_bytes=runner.state.numel()*runner.state.element_size(),
                       table_bytes=table.nbytes,checkpoint_step=old['steps'])
        (args.output/'initial.json').write_text(json.dumps(initial,indent=2))
    logpath = args.output/'metrics.jsonl'
    if args.resume:
        rows = [json.loads(s) for s in logpath.read_text().splitlines() if s.strip()]
        rows = [r for r in rows if r['step'] <= start]
        if [r['step'] for r in rows] != list(range(1,start+1)):
            raise ValueError('checkpoint/log cursor mismatch')
        logpath.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    began = time.perf_counter()
    with logpath.open('a' if args.resume else 'w') as log:
        for step in range(start,args.steps):
            tick = time.perf_counter()
            prediction = peek(table,pending)
            policy = runner.decide(pending,prediction)
            policy /= policy.sum()
            if not np.isfinite(policy).all() or np.any(policy <= 0):
                raise FloatingPointError('policy lost full support at machine precision')
            action = int(rng.choice(2,p=policy))
            independent_action = int(rng.integers(2))  # consume equally in both arms
            executed = action if args.mode == 'coupled' else independent_action
            response = world.take(executed,args.tokens-1)
            raw = np.concatenate((np.array([pending,ACTION_BYTES[action]],np.uint8),response))
            _, probs, packet = table.run_credit(raw)
            np.testing.assert_allclose(probs[0],prediction,rtol=0,atol=1e-7)
            del packet  # no retrieved-count feedback in this experiment
            loss, actual_policy, grad = runner.step(raw,probs,action,baseline)
            discrepancy = float(np.max(np.abs(policy-actual_policy)))
            if discrepancy > 2e-5:
                raise AssertionError('action was not sampled from the trained causal prefix')
            baseline = .95*baseline + .05*loss
            pending = int(response[-1])
            row = dict(step=step+1,nll=loss,action=action,executed=executed,
                       policy=policy.tolist(),policy_replay_error=discrepancy,
                       gradient_norm=grad,cursors=list(world.cursors),
                       external_bytes=(step+1)*(args.tokens-1),self_bytes=step+1,
                       response_sha256=hashlib.sha256(response.tobytes()).hexdigest(),
                       seconds=time.perf_counter()-tick)
            if (step+1)%args.validate_every == 0 or step+1 == args.steps:
                row['validation_nll'] = validate_online_sidecar(model,table,validation,args.tokens)
            log.write(json.dumps(row)+'\n'); log.flush()
            if 'validation_nll' in row:
                archive = dict(model=model.state_dict(),optimizer=runner.optimizer.state_dict(),
                    recurrent=runner.state.detach().cpu(),table=table.state_dict(),
                    numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all(),step=step+1,config=config,
                    pending=pending,baseline=baseline,cursors=list(world.cursors))
                temporary = args.output/'last.pending.pt'
                torch.save(archive,temporary,pickle_protocol=5)
                temporary.replace(args.output/'last.pt')
                del archive
            if (step+1)%50 == 0 or step == start or step+1 == args.steps:
                progress = dict(status='complete' if step+1==args.steps else 'running',**row,
                    table_bytes=table.nbytes,gpu_peak_bytes=torch.cuda.max_memory_allocated(),
                    seconds_this_session=time.perf_counter()-began)
                (args.output/'progress.json').write_text(json.dumps(progress,indent=2))
                print(json.dumps(progress),flush=True)
    rows = [json.loads(line) for line in logpath.read_text().splitlines()]
    summary = dict(status='complete',mode=args.mode,updates=len(rows),
        external_bytes=len(rows)*(args.tokens-1),self_bytes=len(rows),
        mean_selected_stream_nll=float(np.mean([r['nll'] for r in rows])),
        final_validation_nll=rows[-1]['validation_nll'],
        selected_counts=[sum(r['action']==a for r in rows) for a in range(2)],
        executed_counts=[sum(r['executed']==a for r in rows) for a in range(2)],
        mean_nll_by_source=[float(np.mean([r['nll'] for r in rows if r['executed']==a]))
                            if any(r['executed']==a for r in rows) else None for a in range(2)],
        max_policy_replay_error=max(r['policy_replay_error'] for r in rows),
        update_seconds=sum(r['seconds'] for r in rows),
        table_bytes=table.nbytes,gpu_peak_bytes=torch.cuda.max_memory_allocated(),
        organism_admitted=False,convergence_established=False)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
