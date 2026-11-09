"""Read-only rank/use audit of surviving positive-credit rescued records.

Compare intact, a vacant existing weighted contribution, and that same weight
carrying the protected record. Only the designated compartment is intervened;
downstream computation responds normally. No updates, commits, or policy edits.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import types

import torch
from torch.nn import functional as F

from scripts.ib_local.pretrained_gdn import load_pure_gdn, enable_streaming_convolution, clone_cache
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner, language_loss
from scripts.ib_local.run_pretrained_gdn_cross_window import assert_same, write_json


@contextmanager
def retained_value_probe(wrapper, slot, version, mode):
    """Keep natural slots/weights; replace the weakest selected contribution."""
    if mode not in ('inspect','drop','swap'):
        raise ValueError('unknown read-only intervention')
    if 'retrieve' in wrapper.__dict__:
        raise RuntimeError('do not nest retrieval interventions')
    m = wrapper.memory
    if not bool(m.occupied[slot]) or int(m.read_credit_version[slot]) != version:
        raise ValueError('protected identity was replaced')
    original, receipts = wrapper.retrieve, []

    def retrieve(self, query, replay):
        value = original(query,replay)
        read = self.last_read
        if read.slots.shape[0] != 1:
            raise ValueError('legacy equal-weight swap probe needs a single query; use whole-record removal for multi-query reads')
        chosen = read.slots[0].tolist()
        if not bool(m.occupied[read.slots].all()):
            raise ValueError('full real memory required')
        q = F.normalize(self.encoder(query.float().cpu()),dim=-1)
        keys = self.sealed_addresses()
        scores = (q @ keys.T).masked_fill(~m.occupied,-torch.inf)
        selected = slot in chosen
        victim_index = chosen.index(slot) if selected else int(read.weights[0].argmin())
        victim_slot = chosen[victim_index]
        receipt = dict(candidate_slot=slot,candidate_version=version,
            candidate_age_tokens=int(m.clock-m.birth[slot]),
            candidate_rank=int((scores[0] > scores[0,slot]).sum())+1,
            candidate_similarity=float(scores[0,slot]),natural_slots=chosen,
            natural_weights=read.weights.detach().tolist(),already_selected=selected,
            victim_index=victim_index,victim_slot=victim_slot,
            victim_similarity=float(scores[0,victim_slot]),
            substituted_weight=float(read.weights[0,victim_index]))
        receipts.append(receipt)
        if mode == 'inspect' or (mode == 'swap' and selected):
            return value
        old = self.value_map(self.replayed[victim_index].float().cpu())
        weight = read.weights[:,victim_index,None]
        if mode == 'drop':
            return value-weight*old
        capsule = self.capsules[slot]
        if capsule is None or capsule.version != version:
            raise ValueError('protected capsule and record version diverged')
        new = self.value_map(replay(capsule)[:,-1].float().cpu())
        return value+weight*(new-old)

    wrapper.retrieve = types.MethodType(retrieve,wrapper)
    try:
        yield receipts
    finally:
        del wrapper.retrieve


def summarize(rows):
    n = len(rows)
    if not n:
        return dict(cases=0)
    tolerance = 1e-6
    mean = lambda key:sum(r[key] for r in rows)/n
    return dict(cases=n,selected=sum(r['already_selected'] for r in rows),
        candidate_ranks=[r['candidate_rank'] for r in rows],
        mean_swap_minus_intact=mean('swap_minus_intact'),
        mean_swap_minus_drop=mean('swap_minus_drop'),
        mean_drop_minus_intact=mean('drop_minus_intact'),
        improves_both=sum(r['swap_minus_intact'] < -tolerance and
                          r['swap_minus_drop'] < -tolerance for r in rows),
        worse_than_both=sum(r['swap_minus_intact'] > tolerance and
                           r['swap_minus_drop'] > tolerance for r in rows),
        scope='fixed candidate opportunity check; equal-weight swaps are diagnostic, not a trained retrieval rule')


def audit_empty_reads(learner,curriculum,offset,tokens):
    """Same whole-body incoming state, zero all episodic values, preserve S."""
    rows = []
    for wrapper in learner.memories.values():
        m = wrapper.memory
        if not bool(((m.clock-m.birth[m.occupied]) > 0).all()):
            raise ValueError('age-zero observations would escape the empty-read control')
    with torch.no_grad():
        for source,ids in curriculum['heldouts'].items():
            cache = clone_cache(learner.cache)
            for start in range(offset,offset+tokens,128):
                x,y = ids[start:start+128][None].cuda(),ids[start+1:start+129][None].cuda()
                intact,next_cache = learner.predict(x,cache)
                empty,_ = learner.predict(x,cache,suppress_older_than=0)
                _,base_loss = language_loss(intact,y)
                _,empty_loss = language_loss(empty,y)
                rows.append(dict(source=source,start=start,intact_nll=float(base_loss),
                    empty_nll=float(empty_loss),empty_minus_intact=float(empty_loss-base_loss)))
                cache = next_cache
                print('EMPTY',source,start,rows[-1]['empty_minus_intact'],flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--curriculum',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--published',type=Path)
    parser.add_argument('--offset',type=int,default=512)
    parser.add_argument('--tokens',type=int,default=512)
    parser.add_argument('--only-empty',action='store_true',
                        help='Whole episodic-read cut only, preserving native S and the same incoming state')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve previous evidence')
    if args.offset < 0 or args.tokens < 128 or args.tokens % 128:
        parser.error('nonnegative offset and positive128-aligned budget required')
    torch.set_num_threads(1)
    payload = torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    state = payload['learner']
    protected = [(key,meta) for key,meta in payload['protected'].items() if meta['evidence'] > 0]
    if not protected:
        raise ValueError('no surviving positive rescued identities to diagnose')
    curriculum = torch.load(args.curriculum,map_location='cpu',weights_only=False)
    if any(args.offset+args.tokens+1 > len(ids) for ids in curriculum['heldouts'].values()):
        raise ValueError('insufficient genuine heldout continuation')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report = dict(status='running',checkpoint=str(args.checkpoint),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        source_updates=state['updates'],dataset_revision=curriculum['dataset_revision'],
        protected={k:v for k,v in protected},optimizer_updates=0,commits=0,
        protocol=dict(offset=args.offset,tokens_per_source=args.tokens,window=128,
            contexts='latter half of the saved independent validation/A-unseen/B-unseen, not used in the prior endpoint probes',
            cache='each source begins from the saved body cache; advance only the intact frozen trajectory',
            intervention='same incoming full state and actual natural mixture weight, one protected record at a time',
            vacancy='remove the same victim value, keep all remaining weights unchanged',
            scope='read-only current usefulness/ranking; neither training benefit nor all-future usefulness'),records=[])
    write_json(args.output,report)
    model,_ = load_pure_gdn(args.model)
    enable_streaming_convolution(model)
    learner = PretrainedExperienceLearner(model,layers=state['layers'],capacity=state['capacity'])
    learner.load_state_dict(state)
    before = learner.state_dict()
    try:
        if args.only_empty:
            report['protocol']['intervention'] = 'zero all episodic values at both attached layers, preserve S and all model parameters'
            report['records'] = audit_empty_reads(learner,curriculum,args.offset,args.tokens)
            assert_same(before,learner.state_dict(),'complete_persistent_state_unchanged')
            report.update(status='complete',persistent_state_unchanged=True,
                source_checkpoint_unchanged=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()==report['checkpoint_sha256'],
                mean_empty_minus_intact=sum(r['empty_minus_intact'] for r in report['records'])/len(report['records']),
                empty_better=sum(r['empty_minus_intact'] < -1e-6 for r in report['records']),
                by_source={name:sum(r['empty_minus_intact'] for r in report['records'] if r['source']==name)/
                           sum(r['source']==name for r in report['records']) for name in curriculum['heldouts']})
            write_json(args.output,report)
            if args.published:
                args.published.parent.mkdir(parents=True,exist_ok=True)
                write_json(args.published,report)
            print('COMPLETE EMPTY',report['mean_empty_minus_intact'],flush=True)
            return
        with torch.no_grad():
            for source,ids in curriculum['heldouts'].items():
                cache = clone_cache(learner.cache)
                for start in range(args.offset,args.offset+args.tokens,128):
                    x,y = ids[start:start+128][None].cuda(),ids[start+1:start+129][None].cuda()
                    logits,next_cache = learner.predict(x,cache)
                    _,base = language_loss(logits,y)
                    for key,metadata in protected:
                        layer,slot,version = map(int,key.split(':'))
                        variants,receipts = {},{}
                        for mode in ('inspect','drop','swap'):
                            with retained_value_probe(learner.memories[layer],slot,version,mode) as trace:
                                varied,_ = learner.predict(x,cache)
                            if len(trace) != 1:
                                raise AssertionError('target read was not observed exactly once')
                            receipts[mode] = trace[0]
                            _,loss = language_loss(varied,y)
                            variants[mode] = float(loss)
                            if mode == 'inspect' and not torch.equal(varied,logits):
                                raise AssertionError('inspection changed original prediction')
                            if mode == 'swap' and trace[0]['already_selected'] and not torch.equal(varied,logits):
                                raise AssertionError('same-value control changed original prediction')
                        assert_same(receipts['inspect'],receipts['drop'],'same_target_query_and_weights')
                        assert_same(receipts['inspect'],receipts['swap'],'same_target_query_and_weights')
                        row = dict(source=source,start=start,identity=key,layer=layer,
                            original_rescue=metadata,intact_nll=float(base),drop_nll=variants['drop'],
                            swap_nll=variants['swap'],swap_minus_intact=variants['swap']-float(base),
                            swap_minus_drop=variants['swap']-variants['drop'],
                            drop_minus_intact=variants['drop']-float(base),**receipts['inspect'])
                        report['records'].append(row)
                    cache = next_cache
                    write_json(args.output,report)
                    print(source,start,[(r['identity'],r['candidate_rank'],round(r['swap_minus_intact'],6),
                                           round(r['swap_minus_drop'],6)) for r in report['records'][-len(protected):]],flush=True)
        assert_same(before,learner.state_dict(),'complete_persistent_state_unchanged')
        rows = report['records']
        report.update(status='complete',persistent_state_unchanged=True,
            source_checkpoint_unchanged=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()==report['checkpoint_sha256'],
            overall=summarize(rows),unselected=summarize([r for r in rows if not r['already_selected']]),
            by_identity={key:summarize([r for r in rows if r['identity']==key]) for key,_ in protected},
            by_source={key:summarize([r for r in rows if r['source']==key]) for key in curriculum['heldouts']})
        write_json(args.output,report)
        if args.published:
            args.published.parent.mkdir(parents=True,exist_ok=True)
            write_json(args.published,report)
        print('COMPLETE',report['unselected'],flush=True)
    except Exception as error:
        report.update(status='failed',error=repr(error))
        write_json(args.output,report)
        raise


if __name__ == '__main__':
    main()
