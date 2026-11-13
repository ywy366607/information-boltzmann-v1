"""Real QASPER excerpt SFT: matched bounded consolidation on/off.

Use annotated extractive answers present in a natural paper prefix. Questions
precede >=1024 tokens of text. No generated distractors or length padding is
used to create the long dependency. EOS compute padding is reported separately.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from scripts.ib_local.experience_consolidation import bounded_preference, replacement_slot
from scripts.ib_local.pretrained_gdn import load_pure_gdn, enable_streaming_convolution
from scripts.ib_local.pretrained_gdn_learning import PretrainedExperienceLearner, language_loss
from scripts.ib_local.probe_pretrained_gdn_instruction import prompt, encode_answer
from scripts.ib_local.probe_pretrained_gdn_sft_credit import windows


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def assert_first_forward_equal(full, held, reference):
    """Reject changed predictions/reads before attributing an update to credit."""
    fields = ('ce','nll_sum','sigreg','reads','supervised_tokens','supervised_nll_sum')
    for field in fields:
        if full[field] != held[field]:
            raise AssertionError(f'held intervention changed first forward: {field}')
        if full[field] != reference[field]:
            raise AssertionError(f'saved full reference cannot be reproduced: {field}')


def select_papers(payload, tokenizer, count, *, minimum=1024, maximum=2048):
    """Keep original paragraph boundaries and require the answer in the prefix."""
    keys = list(payload)
    random.Random(11).shuffle(keys)
    selected = []
    for key in keys:
        paper = payload[key]
        paragraphs = [paper['title'], paper['abstract']]
        for section in paper['full_text']:
            paragraphs.extend([section['section_name'], *section['paragraphs']])
        text = ''
        for paragraph in paragraphs:
            if not paragraph:
                continue
            trial = text + ('\n\n' if text else '') + paragraph
            if len(tokenizer.encode(trial, add_special_tokens=False)) > maximum:
                break
            text = trial
        size = len(tokenizer.encode(text, add_special_tokens=False))
        if size < minimum:
            continue
        choices = []
        for qa in paper['qas']:
            if len(tokenizer.encode(qa['question'], add_special_tokens=False)) > 24:
                continue
            for annotation in qa['answers']:
                answer = annotation['answer']
                if answer['unanswerable'] or len(answer['extractive_spans']) != 1:
                    continue
                span = answer['extractive_spans'][0].strip()
                if not span or span not in text:
                    continue
                try:
                    seq, boundary = encode_answer(tokenizer, prompt(text, qa['question'], 'early'), span)
                except ValueError:
                    continue
                if not 1 <= len(seq)-boundary <= 32:
                    continue
                choices.append(dict(id=qa['question_id'], question=qa['question'], answer=span,
                    answer_start=text.index(span), question_to_answer_tokens=boundary-
                    len(tokenizer.encode(prompt('', qa['question'], 'early').split('Passage:')[0],
                                         add_special_tokens=False))))
                break
        pair = next(([a,b] for i,a in enumerate(choices) for b in choices[i+1:]
                     if a['id'] != b['id'] and a['answer'].lower() not in b['answer'].lower()
                     and b['answer'].lower() not in a['answer'].lower()), None)
        if pair:
            selected.append(dict(paper_id=key, title=paper['title'], context=text,
                                  context_tokens=size, qa=pair))
        if len(selected) == count:
            return selected
    raise ValueError(f'only {len(selected)} eligible paper prefixes; requested {count}')


def prepare(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    source = json.loads((args.data/'manifest.json').read_text(encoding='utf-8'))
    raw = {split: json.loads((args.data/f'qasper-{name}-v0.3.json').read_text(encoding='utf-8'))
           for split,name in [('train','train'), ('validation','dev')]}
    task = dict(kind='real_qasper_prefix_extractive_pairs', source=source,
        selection='first <=2048 original tokens, whole paragraphs; actual annotated answer present; question <=24 tokens; answer1..32',
        scope='restricted excerpt subset, not official full-document QASPER benchmark',
        train=select_papers(raw['train'],tokenizer,args.train_papers),
        validation=select_papers(raw['validation'],tokenizer,args.eval_papers))
    if {p['paper_id'] for p in task['train']} & {p['paper_id'] for p in task['validation']}:
        raise AssertionError('train/dev paper overlap')
    write_json(args.output, task)
    print(json.dumps({split:dict(papers=len(task[split]),questions=2*len(task[split]),
        min_context_tokens=min(p['context_tokens'] for p in task[split]),
        max_context_tokens=max(p['context_tokens'] for p in task[split])) for split in raw}), flush=True)


@contextmanager
def admission_stats(learner):
    """Diagnostic counters outside the learner; no new retained body state."""
    stats = dict(full_admissions=0, actual_not_oldest=0, credit_not_oldest=0,
                 evicted_positive=0, evicted_negative=0, max_evicted_age=0)
    saved = []
    for wrapper in learner.memories.values():
        memory = wrapper.memory
        original = memory.commit_episode

        def commit(*a, _m=memory, _original=original, **kw):
            full = bool(_m.occupied.all())
            if full:
                oldest = int(_m.last_write.argmin())
                candidate = replacement_slot(_m)
                preferences = bounded_preference(_m).clone()
                births = _m.birth.clone()
            result = _original(*a, **kw)
            if full:
                slot = result[0]
                stats['full_admissions'] += 1
                stats['actual_not_oldest'] += slot != oldest
                stats['credit_not_oldest'] += candidate != oldest
                stats['evicted_positive'] += int(preferences[slot] > 0)
                stats['evicted_negative'] += int(preferences[slot] < 0)
                stats['max_evicted_age'] = max(stats['max_evicted_age'],int(_m.clock-births[slot]))
            return result

        memory.commit_episode = commit
        saved.append((memory,original))
    try:
        yield stats
    finally:
        for memory, original in saved:
            memory.commit_episode = original


def train(args):
    from transformers import AutoTokenizer
    task = json.loads(args.task.read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    torch.set_num_threads(1)
    model, _ = load_pure_gdn(args.model)
    enable_streaming_convolution(model)
    learner = PretrainedExperienceLearner(model,layers=source['layers'],capacity=source['capacity'])
    learner.load_state_dict(source,detach_replay=False)
    examples = [(p,q) for p in task['train'] for q in p['qa']]
    if 4+args.examples > len(examples):
        raise ValueError('insufficient actual distinct training questions')
    held_stage = args.stage == 'held'
    paired_credit = args.stage == 'credit-pair'
    credit_stage = held_stage or paired_credit
    reference = None
    if credit_stage:
        reference = json.loads(args.reference_report.read_text(encoding='utf-8'))
        if reference['status'] != 'complete' or reference['task_sha256'] != hashlib.sha256(args.task.read_bytes()).hexdigest():
            raise ValueError('completed full reference and exact same task required')
        if (reference['distinct_train_examples_per_arm'],reference['epochs'],reference['eval_papers']) != (
                args.examples,args.epochs,args.eval_papers):
            raise ValueError('held budget must match the completed reference')
        if args.checkpoint.resolve() != Path(reference['shared_checkpoint']).resolve():
            raise ValueError('held must start at the saved shared warm checkpoint')
    kind = ('real_long_sft_native_credit_pair' if paired_credit else
            'real_long_sft_held_old_credit' if held_stage else 'real_long_sft_consolidation_pair')
    report = dict(kind=kind,status='running',
        source_checkpoint=str(args.checkpoint),source_updates=source['updates'],
        task_sha256=hashlib.sha256(args.task.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        window=128,capacity=learner.capacity,layers=list(learner.layers),warm_examples=4,
        distinct_train_examples_per_arm=args.examples,epochs=args.epochs,
        train_examples_per_arm=args.examples*args.epochs,eval_papers=args.eval_papers,
        comparison='same aligned read and native replay gradient; joint retrieval/retention consolidation only on/off',
        primary='independent answer-token NLL and question-swap gap, with real admission/retention changes',
        note='one restored RNG seed; fixed-budget learning curves, not convergence or official QASPER capability',
        arms={})
    if credit_stage:
        report.update(warm_examples=0,shared_checkpoint=str(args.checkpoint),
            reference_report=str(args.reference_report),
            reference_report_sha256=hashlib.sha256(args.reference_report.read_bytes()).hexdigest(),
            reference_arm='off',
            comparison='same recall/reexecution/S/current-window learning, consolidation off; detach only old native response before live encoder',
            primary='full-minus-held independent answer NLL and revisit learning under the matched budget')

    def save():
        write_json(args.output,report)

    def stop_requested(location):
        if not (args.output.parent/'STOP').exists():
            return False
        checkpoint = args.output.parent/'stopped_last.pt'
        temp = checkpoint.with_suffix('.tmp')
        torch.save(learner.state_dict(),temp)
        temp.replace(checkpoint)
        report.update(status='stopped',stop_location=location,stop_checkpoint=str(checkpoint))
        save()
        return True

    def apply_policy(enabled):
        for wrapper in learner.memories.values():
            wrapper.memory.consolidation_enabled = enabled
            wrapper.memory.consolidation_rule = 'evidence'

    def train_example(paragraph, qa, *, diagnose=False):
        ids,mask,accounting = windows(tokenizer,paragraph['context'],qa['question'],qa['answer'])
        updates, question_records = [], {}
        for start in range(0,len(mask),128):
            x,y = ids[start:start+128][None].cuda(),ids[start+1:start+129][None].cuda()
            active = mask[start:start+128][None].cuda()
            if bool(active.any()):
                position = int(active[0].nonzero()[0])
                present_before = {layer: bool(learner.memories[layer].memory.occupied[slot]) and
                    int(learner.memories[layer].memory.read_credit_version[slot]) == version
                    for layer,(slot,version) in question_records.items()}
                row = learner.step(x,y,loss_mask=active,read_position=position,
                                   diagnose=diagnose and not updates)
                evidence = {}
                for layer,wrapper in learner.memories.items():
                    slot,version = question_records[layer]
                    m = wrapper.memory
                    present = bool(m.occupied[slot]) and int(m.read_credit_version[slot]) == version
                    evidence[str(layer)] = dict(present_after_update=present,
                        retrieved=present_before[layer] and slot in row['reads'][str(layer)]['slots'])
                row['question_record'] = evidence
                updates.append(row)
            else:
                slots = learner.observe(x)
                if start == 0:
                    question_records = {layer:(slot,int(learner.memories[layer].memory.read_credit_version[slot]))
                                        for layer,slot in slots.items()}
        return dict(id=qa['id'],paper_id=paragraph['paper_id'],accounting=accounting,
            answer_nll=sum(r['supervised_nll_sum'] for r in updates)/accounting['answer_tokens'],
            updates=updates)

    @torch.no_grad()
    def evaluate():
        state = learner.state_dict()
        rows = []
        try:
            for paragraph in task['validation'][:args.eval_papers]:
                matrix = []
                for question in paragraph['qa']:
                    scores = []
                    for answer in paragraph['qa']:
                        learner.load_state_dict(state)
                        ids,mask,acc = windows(tokenizer,paragraph['context'],question['question'],answer['answer'])
                        total,count = 0.,0
                        for start in range(0,len(mask),128):
                            active = mask[start:start+128][None].cuda()
                            position = int(active[0].nonzero()[0]) if bool(active.any()) else 0
                            _,logits = learner.observe(ids[start:start+128][None].cuda(),
                                                       return_logits=True,read_position=position)
                            if bool(active.any()):
                                _,ce = language_loss(logits,ids[start+1:start+129][None].cuda(),active)
                                total += float(ce)*int(active.sum())
                                count += int(active.sum())
                        scores.append(dict(mean=total/count,nll=total,tokens=count))
                    matrix.append(scores)
                gap = sum(matrix[1-i][i]['mean']-matrix[i][i]['mean'] for i in range(2))/2
                rows.append(dict(paper_id=paragraph['paper_id'],matrix=matrix,swap_gap=gap))
        finally:
            learner.load_state_dict(state)
        return dict(mean_answer_nll=sum(r['matrix'][i][i]['nll'] for r in rows for i in range(2))/
                    sum(r['matrix'][i][i]['tokens'] for r in rows for i in range(2)),
                    mean_question_swap_gap=sum(r['swap_gap'] for r in rows)/len(rows),
                    matching_correct=sum(r['swap_gap']>0 for r in rows),papers=len(rows),rows=rows)

    save()
    torch.cuda.reset_peak_memory_stats()
    try:
        apply_policy(False)
        if credit_stage:
            before = learner.state_dict()
            paragraph,qa = examples[4]
            full = train_example(paragraph,qa,diagnose=True)
            full_native = {n:p.detach().cpu().clone() for n,p in learner.native}
            full_masters = [p.detach().cpu().clone() for p in learner.masters]
            learner.load_state_dict(before,detach_replay=True)
            held = train_example(paragraph,qa)
            forward_fields = ('ce','nll_sum','sigreg','reads','supervised_tokens','supervised_nll_sum')
            report['credit_gate_attempt'] = dict(question_id=qa['id'],
                full={k:full['updates'][0][k] for k in forward_fields},
                held={k:held['updates'][0][k] for k in forward_fields},
                saved_reference={k:reference['arms']['off']['records'][0]['updates'][0][k]
                                 for k in forward_fields})
            save()
            first_full,first_held = full['updates'][0],held['updates'][0]
            assert_first_forward_equal(first_full,first_held,
                first_full if paired_credit else reference['arms']['off']['records'][0]['updates'][0])
            native_difference = {n:float((full_native[n].float()-p.detach().float().cpu()).norm())
                                 for n,p in learner.native}
            master_difference = max(float((a-p.detach().cpu()).abs().max())
                                    for a,p in zip(full_masters,learner.masters))
            if master_difference == 0:
                raise AssertionError('old-native credit did not change the actual optimizer update')
            report['credit_gate'] = dict(first_forward_full_held_exact=True,
                saved_reference_exact=all(first_full[k] == report['credit_gate_attempt']['saved_reference'][k]
                                          for k in forward_fields),
                saved_reference_ce_delta=first_full['ce']-report['credit_gate_attempt']['saved_reference']['ce'],
                first_ce=full['updates'][0]['ce'],first_reads=full['updates'][0]['reads'],
                full_diagnostics=full['updates'][0]['diagnostics'],
                native_parameter_difference_norms=native_difference,
                fp32_master_difference_max=master_difference,
                discarded_probe_optimizer_updates=len(full['updates'])+len(held['updates']))
            learner.load_state_dict(before,detach_replay=True)
            shared = before
            del full_native,full_masters
            report['initial_validation'] = (evaluate() if paired_credit else
                                           reference['arms']['off']['curve'][0]['validation'])
            report['initial_validation_provenance'] = ('fresh same-process frozen validation of the shared state'
                if paired_credit else 'reused full frozen reference; detach changes no forward value')
            save()
            print('CREDIT_GATE',json.dumps(report['credit_gate']),flush=True)
        else:
            report['warm'] = []
            for i,(p,q) in enumerate(examples[:4]):
                tick = time.perf_counter()
                report['warm'].append(train_example(p,q,diagnose=i==0))
                save()
                print('WARM',i+1,'seconds',time.perf_counter()-tick,flush=True)
                if stop_requested(f'warm:{i+1}'):
                    return
            audit = report['warm'][0]['updates'][0]['diagnostics']
            report['connection_gate'] = dict(symmetric_logit_max=audit['symmetric_logit_max'],
                old_read_cut_nll_delta=audit['old_read_cut_nll_delta'],
                old_native_gradient_nonzero={key:any(value is not None and value>0 for value in values.values())
                    for key,values in audit.items() if key.endswith('_old_native_grad_norms')})
            if not report['connection_gate']['old_native_gradient_nonzero'] or not all(
                    report['connection_gate']['old_native_gradient_nonzero'].values()):
                raise RuntimeError('actual old-native derivative not connected; do not run long pair')
            shared = learner.state_dict()
            shared_path = args.output.parent/'shared.pt'
            torch.save(shared,shared_path.with_suffix('.tmp'))
            shared_path.with_suffix('.tmp').replace(shared_path)
            report['shared_checkpoint'] = str(shared_path)
            report['initial_validation'] = evaluate()
        save()
        print('INITIAL',report['initial_validation']['mean_answer_nll'],flush=True)
        arm_config = ([('full',False),('held',False)] if paired_credit else
                      [('held',False)] if held_stage else [('off',False),('on',True)])
        for arm,enabled in arm_config:
            learner.load_state_dict(shared,detach_replay=(arm == 'held'))
            apply_policy(enabled)
            initial_updates = learner.updates
            report['arms'][arm] = dict(records=[],curve=[])
            initial = evaluate() if enabled else report['initial_validation']
            report['arms'][arm]['curve'].append(dict(examples=0,updates=0,validation=initial))
            with admission_stats(learner) as stats:
                for index,(p,q) in enumerate(examples[4:4+args.examples]*args.epochs,1):
                    tick = time.perf_counter()
                    row = train_example(p,q)
                    row['epoch'] = (index-1)//args.examples+1
                    row['seconds'] = time.perf_counter()-tick
                    report['arms'][arm]['records'].append(row)
                    if paired_credit and arm == 'held' and index == 1:
                        save()
                        assert_first_forward_equal(report['arms']['full']['records'][0]['updates'][0],
                                                   row['updates'][0],row['updates'][0])
                        report['credit_gate']['actual_training_first_forward_exact'] = True
                    if index % args.eval_every == 0 or index == args.examples*args.epochs:
                        # Validation writes must not count as real admissions.
                        snapshot = dict(stats)
                        validation = evaluate()
                        stats.clear(); stats.update(snapshot)
                        report['arms'][arm]['curve'].append(dict(examples=index,
                            updates=learner.updates-initial_updates,validation=validation,
                            admissions=dict(stats)))
                        checkpoint = args.output.parent/f'{arm}_last.pt'
                        temp = checkpoint.with_suffix('.tmp')
                        torch.save(learner.state_dict(),temp)
                        temp.replace(checkpoint)
                        report['arms'][arm].update(checkpoint=str(checkpoint),
                                                  checkpoint_examples=index)
                    save()
                    print(arm,index,round(row['answer_nll'],5),dict(stats),flush=True)
                    if stop_requested(f'{arm}:{index}'):
                        return
                report['arms'][arm]['actual_updates'] = learner.updates-initial_updates
                report['arms'][arm]['admissions'] = dict(stats)
            # Keep one actual final state per arm for an honest continuation.
            checkpoint = args.output.parent/f'{arm}_last.pt'
            temp = checkpoint.with_suffix('.tmp')
            torch.save(learner.state_dict(),temp)
            temp.replace(checkpoint)
            report['arms'][arm]['checkpoint'] = str(checkpoint)
            save()
        if credit_stage:
            full = (report['arms']['full']['curve'][-1]['validation'] if paired_credit else
                    reference['arms']['off']['curve'][-1]['validation'])
            held = report['arms']['held']['curve'][-1]['validation']
            report.update(answer_nll_full_minus_held=full['mean_answer_nll']-held['mean_answer_nll'],
                swap_gap_full_minus_held=full['mean_question_swap_gap']-held['mean_question_swap_gap'])
        else:
            off,on = [report['arms'][arm]['curve'][-1]['validation'] for arm in ('off','on')]
            report.update(answer_nll_on_minus_off=on['mean_answer_nll']-off['mean_answer_nll'],
                swap_gap_on_minus_off=on['mean_question_swap_gap']-off['mean_question_swap_gap'])
        report.update(status='complete',peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      source_navigation_individual_unchanged=True)
        save()
        print('COMPLETE',report.get('answer_nll_full_minus_held',report.get('answer_nll_on_minus_off')),flush=True)
    except Exception as error:
        report.update(status='failed',error=repr(error))
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prepare','train','held','credit-pair'),required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--data',type=Path)
    parser.add_argument('--task',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--reference-report',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--train-papers',type=int,default=44)
    parser.add_argument('--eval-papers',type=int,default=4)
    parser.add_argument('--examples',type=int,default=84)
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--eval-every',type=int,default=42)
    args = parser.parse_args()
    if args.examples < 1 or args.eval_every < 1 or args.eval_papers < 1 or args.epochs < 1:
        parser.error('positive question, evaluation interval and paper counts required')
    if args.stage in ('held','credit-pair') and args.reference_report is None:
        parser.error('held stage requires the completed full reference report')
    if args.output.exists():
        raise FileExistsError('preserve existing evidence')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.stage == 'prepare':
        prepare(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
