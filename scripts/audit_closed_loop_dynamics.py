"""Paired whole-organism interventions on real natural-text events.

Private restored workspaces only; never save to the authoritative checkpoint.
2 x 2 S perturbations, crossed with endogenous / yoked / base-GDN streams.
This identifies transient causal response, not a trained capability advantage.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation
from scripts.audit_spectrum_runtime import digest, carried


def pulse(state, first, mask):
    """Swap two value coordinates; preserve Frobenius and row/column budgets."""
    receipts = []
    for bit, (layer, head) in enumerate(((7, 0), (23, 1))):
        if mask & (1 << bit):
            matrix = state[layer, ..., :first.matrix_size].reshape(1, first.heads, first.dk, first.dv)
            before = matrix[:, head].clone()
            matrix[:, head, :, [0, 1]] = matrix[:, head, :, [1, 0]]
            receipts.append(dict(layer=layer+1, head=head, norm=float(before.norm()),
                delta_norm=float((matrix[:, head]-before).norm()),
                after_norm=float(matrix[:, head].norm())))
    return receipts


def world_boundary(world):
    base = world.base
    return dict(cursor=int(base.cursor), anchor=int(base.anchor), offset=int(base.offset),
                moves=int(base.moves), rng=copy.deepcopy(base.rng.bit_generator.state))


def replay_take(world, record, utterance, response_bytes, holdout_bytes):
    """Replay the exact reference observation; synchronize only world state."""
    response = np.frombuffer(bytes.fromhex(record['response']), np.uint8).copy()
    holdout = np.frombuffer(bytes.fromhex(record['holdout']), np.uint8).copy()
    if (len(response), len(holdout)) != (response_bytes, holdout_bytes):
        raise ValueError('reference observation boundary mismatch')
    boundary = record['world_after']
    for name in ('cursor', 'anchor', 'offset', 'moves'):
        setattr(world.base, name, boundary[name])
    world.base.rng.bit_generator.state = copy.deepcopy(boundary['rng'])
    start = record['info']['start']
    world.base.seen[start:start+response_bytes+holdout_bytes] = True
    info = copy.deepcopy(record['info'])
    info['child_text'] = utterance.decode('utf-8', 'replace')
    return response, holdout, info


def sql_summary(engine):
    archive = engine.model._causal_archive
    result = {}
    for name in ('episodes', 'relations', 'unknowns', 'evidence', 'variable_uses'):
        if archive.db.execute('SELECT 1 FROM sqlite_master WHERE type="table" AND name=?', (name,)).fetchone():
            result[name] = archive.db.execute(f'SELECT COUNT(*) FROM {name}').fetchone()[0]
    # Logical rows rather than SQLite page layout/path or UUID.
    result['logical_hash'] = digest({name:archive.db.execute(f'SELECT * FROM {name} ORDER BY 1').fetchall()
                                   for name in result})
    return result


@torch.no_grad()
def snapshot(engine, directory, index, *, state=None):
    first = engine.model.body.base.layers[0]
    state = engine.recurrent if state is None else state
    arrays = dict(S=state[..., :first.matrix_size].reshape(32, -1).detach().cpu().numpy())
    if state.shape[-1] == engine.recurrent.shape[-1]:
        arrays['M_or_aux'] = state[..., first.state_size:].reshape(32, -1).detach().cpu().numpy()
    arrays['P'] = torch.stack(engine.bfss_momenta).reshape(32,-1).cpu().numpy()
    names, samples = [], []
    for name, parameter in engine.model.named_parameters():
        names.append(name)
        values = parameter.detach().flatten()
        samples.append(values[::max(1,values.numel()//32)][:32].cpu().numpy())
    arrays['parameters'] = np.concatenate(samples)
    # Persist per-parameter sampling boundaries; no random projection consumes RNG.
    np.savez_compressed(directory/f'state_{index:02}.npz', **arrays)
    return dict(state_hash=digest(state), parameter_sample_hash=digest(arrays['parameters']),
                parameter_names=names if index == 0 else None, archive=sql_summary(engine))


def run_arm(args):
    suffix = f'fire{args.fire_intervention}' if args.fire_intervention is not None else str(args.mask)
    directory = args.work/f'{args.condition}_{suffix}'
    directory.mkdir(parents=True, exist_ok=False)
    # Exact deterministic trace certificate, not an executed experimental arm.
    # If natural take already returns precisely the yoke's observations, info
    # and world state at EVERY step, induction on the identical full event
    # transition makes a second GPU execution redundant. 00 replay is a gate.
    if args.condition=='yoked' and args.mask and args.fire_intervention is None:
        baseline=json.loads((args.work/'closed_0/report.json').read_text(encoding='utf-8'))
        closed=json.loads((args.work/f'closed_{args.mask}/report.json').read_text(encoding='utf-8'))
        gate=json.loads((args.work/'yoked_0/replay_checks.json').read_text(encoding='utf-8'))
        checks=[]
        for actual,reference in zip(closed['observations'],baseline['observations']):
            expected=copy.deepcopy(reference['info'])
            expected['child_text']=actual['info']['child_text']
            checks.append(all(actual[k]==reference[k] for k in ('response','holdout','world_after'))
                          and actual['info']==expected)
        if all(gate.values()) and len(checks)==args.cycles and all(checks):
            (directory/'certificate.json').write_text(json.dumps(dict(
                kind='deterministic trace equivalence, no restored model or optimizer steps executed',
                equivalent_to=f'closed_{args.mask}',world_receipt_checks=checks,
                source_sha256=closed['source_sha256'],baseline_replay_gate=gate,
                premise='same initial complete state/RNG; take is the only observation intervention; actual take and replay return identical observations/info/world state at every reference step'),indent=2)+'\n',encoding='utf-8')
            print(json.dumps(dict(condition=args.condition,mask=args.mask,certified_equal_to=f'closed_{args.mask}',gpu_events_executed=0)),flush=True)
            return
    # Existing production benchmark contract: output points at source for a
    # restore-only prepare, private table/archive are explicitly elsewhere.
    # No main(), run(), checkpoint save or life-card export is called.
    options = ['--checkpoint',str(args.body/'last.pt.gz'), '--output',str(args.body),
        '--cycles',str(args.cycles), '--resume-output', '--world-source','text',
        '--table-storage',str(directory/'table'), '--causal-storage',str(directory/'causal'),
        '--optimizer-policy','budgeted', '--autonomous-events']
    source_files = {name:hashlib.sha256((args.body/name).read_bytes()).hexdigest()
                    for name in ('last.metadata.json', 'config.json')}
    loaded = prepare_continuation(continuation_parser().parse_args(options), resume_output=True)
    engine = loaded.engine
    before = carried(engine)
    result = dict(schema=1, condition=args.condition, mask=args.mask, cycles=args.cycles,
        source_cycle=loaded.cycle, source_sha256=loaded.source_sha, initial=before,
        fire_intervention=args.fire_intervention,
        private_workspace=str(directory), observations=[], events=[], checkpoints=[],
        scope='transient intervention; inherited weights; no independent vanilla pretraining or capability claim')
    first = engine.model.body.base.layers[0]
    state = engine.recurrent
    if args.condition == 'gdn':
        state = state[..., :first.state_size].detach().clone()
    result['pulse'] = pulse(state, first, args.mask)
    result['checkpoints'].append(snapshot(engine,directory,0,state=state))
    reference = None
    if args.condition != 'closed' or args.mask or args.fire_intervention is not None:
        reference = json.loads((args.work/'closed_0'/'report.json').read_text(encoding='utf-8'))
        if reference['source_sha256'] != loaded.source_sha:
            raise ValueError('authoritative source changed during paired audit')
    original_take = engine.world.take
    def take(utterance, rb, hb):
        index = len(result['observations'])
        if args.condition == 'yoked':
            response, holdout, info = replay_take(engine.world, reference['observations'][index], utterance, rb, hb)
        else:
            response, holdout, info = original_take(utterance,rb,hb)
        result['observations'].append(dict(response=bytes(response).hex(),holdout=bytes(holdout).hex(),
            utterance=utterance.hex(), info=info, world_after=world_boundary(engine.world)))
        return response,holdout,info
    engine.world.take = take
    updates = [0]
    old_step = engine.optimizer.step
    def step(*a,**kw):
        out = old_step(*a,**kw); updates[0]+=1; return out
    engine.optimizer.step = step
    original_multinomial = torch.multinomial
    fire_changed = [False]
    if args.fire_intervention is not None:
        def intervention(probabilities, *a, **kw):
            chosen = original_multinomial(probabilities, *a, **kw)
            if probabilities.numel()==3 and not fire_changed[0]:
                if not probabilities.flatten()[args.fire_intervention]>0:
                    raise RuntimeError('intervened action has no policy support')
                chosen = torch.full_like(chosen,args.fire_intervention)
                fire_changed[0] = True
            return chosen
        torch.multinomial = intervention
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    if args.condition == 'gdn':
        # Matched base weights, S, and optimizer state. Ordinary recurrent GDN
        # reads the same real stream; no external-memory, P/M or actor injection.
        base = engine.model.body.base
        engine.model._question_context = None
        pending = int(engine.pending)
        for i, record in enumerate(reference['observations'][:args.cycles]):
            external = np.frombuffer(bytes.fromhex(record['response']+record['holdout']),np.uint8)
            own = np.frombuffer(bytes.fromhex(record['utterance']),np.uint8)
            losses = []
            for j, chunk in enumerate(np.array_split(external,2)):
                raw = np.concatenate((np.array([pending],np.uint8),own if j==0 else np.empty(0,np.uint8),chunk))
                ids = torch.as_tensor(raw[:-1].astype(np.int64),device=engine.device)[None]
                target = torch.as_tensor(raw[1:].astype(np.int64),device=engine.device)[None]
                engine.optimizer.zero_grad(set_to_none=True)
                logits,next_state,_ = base.predict_logits(ids,state)
                ce = F.cross_entropy(logits.flatten(0,1),target.flatten(),reduction='none')
                loss = ce[len(own):].mean() if j==0 else ce.mean()
                (10*loss).backward()
                for group in engine.groups:
                    torch.nn.utils.clip_grad_norm_(group,1.,foreach=True)
                engine.optimizer.step()
                state = next_state.detach()
                pending = int(chunk[-1]); losses.append(float(loss.detach()))
            row = dict(cycle=loaded.cycle+i+1,response_nll=float(np.mean(losses)),updates=2,
                autonomous_event=dict(kind='external'),world=record['info'],forced_reference_stream=True)
            result['events'].append(row)
            result['checkpoints'].append(snapshot(engine,directory,i+1,state=state))
            (directory/'progress.json').write_text(json.dumps(dict(completed=i+1,condition=args.condition,mask=args.mask)),encoding='utf-8')
            print(json.dumps(dict(condition=args.condition,mask=args.mask,external=i+1,seconds=time.perf_counter()-started)),flush=True)
    else:
        from scripts.ib_local.autonomous_cycle import dispatch
        count = 0
        while count < args.cycles:
            row = dispatch(engine)
            result['events'].append(row)
            if row['autonomous_event']['kind'] == 'external':
                count += 1
                result['checkpoints'].append(snapshot(engine,directory,count))
                (directory/'progress.json').write_text(json.dumps(dict(completed=count,condition=args.condition,mask=args.mask)),encoding='utf-8')
                print(json.dumps(dict(condition=args.condition,mask=args.mask,external=count,
                    choice=row['choice'],g=row['metal_g'],seconds=time.perf_counter()-started)),flush=True)
            if len(result['events']) > args.cycles*200:
                raise RuntimeError('autonomous quiet interval exceeded audit budget; not forcing release')
    torch.multinomial = original_multinomial
    if args.fire_intervention is not None and not fire_changed[0]:
        raise RuntimeError('declared first-action intervention was never applied')
    result.update(seconds=time.perf_counter()-started,optimizer_updates=updates[0],
        final=carried(engine),peak_cuda_mib=torch.cuda.max_memory_allocated()/2**20)
    result['audit_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['clip_contract']='engine.groups, norm 1, foreach=True'
    if args.condition=='gdn':
        result['final']['unused_full_body_recurrent']=result['final'].pop('recurrent')
        result['final']['actual_gdn_recurrent']=digest(state)
    for name, original in source_files.items():
        if hashlib.sha256((args.body/name).read_bytes()).hexdigest()!=original:
            raise RuntimeError(f'authoritative file changed: {name}')
    result['authoritative_checkpoint_unchanged'] = hashlib.sha256((args.body/'last.pt.gz').read_bytes()).hexdigest()==loaded.source_sha
    if not result['authoritative_checkpoint_unchanged']: raise RuntimeError('source checkpoint changed')
    (directory/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if args.condition == 'yoked' and args.mask == 0 and args.fire_intervention is None:
        checks = {k:reference['final'][k]==result['final'][k] for k in result['final']}
        checks['archive'] = reference['checkpoints'][-1]['archive']==result['checkpoints'][-1]['archive']
        (directory/'replay_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
        if not all(checks.values()): raise RuntimeError(f'world-yoke baseline was not exact: {checks}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--body',type=Path,required=True)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--condition',choices=('closed','yoked','gdn'),required=True)
    parser.add_argument('--mask',type=int,choices=range(4),required=True)
    parser.add_argument('--cycles',type=int,default=8)
    parser.add_argument('--fire-intervention',type=int,choices=range(3))
    run_arm(parser.parse_args())


if __name__=='__main__': main()
