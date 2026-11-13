"""Read-only real GDN audit of old episode participation and current CE VJPs."""
import argparse
from contextlib import ExitStack
from functools import partial
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--paired-answer', action='store_true',
                   help='compare matched/cut/mismatched old VJPs on actual answer loss')
    p.add_argument('--case', type=int, default=0, help='index in the existing fixed training curriculum')
    p.add_argument('--write-prompt', action='store_true',
                   help='commit full observed prompt windows to the research copy')
    p.add_argument('--needle-removal', action='store_true',
                   help='researcher-only finite removal of selected episodes')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    for key, suffix in [('HF_HOME','huggingface'), ('TRANSFORMERS_CACHE','transformers'),
                        ('TRITON_CACHE_DIR','triton'), ('TORCHINDUCTOR_CACHE_DIR','inductor'),
                        ('TEMP','tmp'), ('TMP','tmp')]:
        os.environ[key] = 'D:/ml_cache/' + suffix
    os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    import torch
    from scripts.ib_local import run_native_s_credit_retention as old
    from scripts.ib_local import pretrained_gdn_stream_load as loading
    from scripts.ib_local.frozen_mlp_int8 import load_streamed_int8_frozen
    from scripts.ib_local.readonly_checkpoint import load_readonly_checkpoint
    from scripts.ib_local.scalar_credit_trial import ScalarCreditLearner
    from scripts.ib_local.pretrained_gdn import clone_cache
    from scripts.ib_local.pretrained_gdn_learning import language_loss, PretrainedExperienceLearner
    from scripts.ib_local.reactivated_credit_audit import audit_reactivated_credit, paired_reactivated_vjp

    source = ROOT/'scratch/joint_next_pending_20261108/pending_recovery.pt'
    source_sha = old.sha256(source)
    if source_sha != '514a45cbe1753c8f056ff75cae2b84121f07c2b5b2a70a6e9e24b9aa9eaca1b5':
        raise ValueError('protected source differs')
    runtime = argparse.Namespace(vendor=Path('E:/cgb_gdn340_probe/vendor/flash-linear-attention-0.2.2'),
                                 model=Path('E:/cgb_gdn340_probe/model'), bounded_autotune=True)
    loader = loading.load_streamed_pure_gdn
    loading.load_streamed_pure_gdn = partial(load_streamed_int8_frozen,
        mapped_directory=ROOT/'scratch/frozen_weight_cache_20261109')
    try:
        _, _, _, model, _ = old.load_runtime(runtime)
    finally:
        loading.load_streamed_pure_gdn = loader
    owner = load_readonly_checkpoint(source)
    state = owner['learner']
    learner = ScalarCreditLearner(model, layers=state['layers'], capacity=state['capacity'])
    learner.load_state_dict(state)
    learner.enable_complete_reactivation()
    data = json.loads((ROOT/'results/cgb_water_last_attempt_20261109/scalar_curriculum.json').read_text())
    if not 0 <= args.case < len(data['training']):
        raise ValueError('case index out of range')
    row = data['training'][args.case]
    ids = row['prefix']
    if len(ids) < 129:
        raise ValueError('actual prompt too short for real 128-token audit')
    cache = clone_cache(learner.cache)
    mask = None
    prompt_writes = []
    if args.paired_answer:
        # Run the real earlier prompt through the same body; no state is reset.
        for start in range(0, len(ids)-128, 128):
            stop = min(start+128, len(ids)-128)
            tokens = torch.tensor(ids[start:stop], device='cuda')[None]
            if args.write_prompt and stop-start == 128:
                slots = PretrainedExperienceLearner.observe(learner,tokens,cache_override=cache)
                cache = clone_cache(learner.cache)
                prompt_writes.append(dict(start=start,stop=stop,
                    slots={str(layer):dict(slot=int(slot),version=int(learner.memories[layer].memory.read_credit_version[slot]))
                           for layer,slot in slots.items()}))
            else:
                _, cache = learner.predict(tokens, cache)
        gold = row['gold']
        if not gold or len(row['tail']) < 128:
            raise ValueError('actual answer and 128-token continuation required')
        inputs = ids[-128:] + gold[:-1]
        targets = ids[-127:] + gold
        x = torch.tensor(inputs, device='cuda')[None]
        target = torch.tensor(targets, device='cuda')[None]
        mask = torch.zeros_like(target, dtype=torch.bool)
        mask[:, -len(gold):] = True
    else:
        x = torch.tensor(ids[:128], device='cuda')[None]
        target = torch.tensor(ids[1:129], device='cuda')[None]
    for memory in learner.memories.values():
        memory.begin(learner.optimizer.param_groups[1], query_limit=x.shape[1])
    current_boundary = clone_cache(cache) if args.needle_removal else None
    with ExitStack() as stack:
        for layer, memory in learner.memories.items():
            stack.enter_context(memory.attach(model, layer=layer))
        output = model(x, past_key_values=cache, use_cache=True)
    _, ce = language_loss(output.logits, target, mask)
    report = dict(status='complete', source_sha256=source_sha, optimizer_updates=0,
                  actual_life_updates=0, frozen_sidecar=True, case=args.case,
                  prompt_writes=prompt_writes,
                  target=('actual RULER answer CE' if args.paired_answer else 'prompt next-token CE'),
                  losses=float(ce.detach()), layers={})
    if args.write_prompt:
        gold=row['gold']
        # Researcher annotation only: never passed to memory or the model.
        positions=[i for i in range(len(ids)-len(gold)+1) if ids[i:i+len(gold)]==gold]
        report['researcher_answer_positions']=positions
        report['researcher_needle_writes']=[w for w in prompt_writes
            if any(w['start'] <= i and i+len(gold) <= w['stop'] for i in positions)]
    if args.paired_answer:
        # This checks only three gate-parameter families per layer; common
        # gradients and the forward path are the same for all three controls.
        suffixes = ('.A_log', '.a_proj.weight', '.b_proj.weight')
        selected = [(name, param) for name,param in learner.native if name.endswith(suffixes)]
        names, params = zip(*selected)
        actual = torch.autograd.grad(ce, params, retain_graph=True, allow_unused=True)
        base = {n:(torch.zeros_like(p) if g is None else g.detach()).cpu()
                for n,p,g in zip(names,params,actual)}
        matched, mismatched = {}, {}
        for layer, memory in learner.memories.items():
            layer_selected = [(n,p) for n,p in selected if n.startswith(f'layer{layer}.')]
            result = paired_reactivated_vjp(ce, memory, layer_selected)
            if result['reason']:
                raise RuntimeError(result['reason'])
            matched.update({n:g.cpu() for n,g in result['matched'].items()})
            mismatched.update({n:g.cpu() for n,g in result['mismatched'].items()})
            report['layers'][str(layer+1)] = dict(strict_old_records=len(result['identities']),
                                                  mismatched_records=result['rotated'],
                                                  identities=result['identities'], teacher_norms=result['teacher_norms'],
                selected_needle_records=[w['slots'][str(layer)]['slot'] for w in report.get('researcher_needle_writes',[])
                    if w['slots'][str(layer)]['slot'] in result['identities'] and
                    int(memory.memory.read_credit_version[w['slots'][str(layer)]['slot']]) == w['slots'][str(layer)]['version']])
        if args.needle_removal:
            if not args.write_prompt:
                raise ValueError('needle removal requires --write-prompt')
            removals = []
            with torch.no_grad():
                with learner.observation(), ExitStack() as stack:
                    for other,wrapper in learner.memories.items():
                        stack.enter_context(wrapper.attach(model,layer=other,capture=False))
                    repeated = model(x,past_key_values=clone_cache(current_boundary),use_cache=True)
                _, repeated_ce = language_loss(repeated.logits,target,mask)
                repeat_diff = float((repeated_ce-ce.detach()).abs())
                if repeat_diff > 1e-6:
                    raise AssertionError('unmodified answer sidecar does not reproduce original CE')
                report['researcher_same_forward_ce_difference'] = repeat_diff
                for layer, memory in learner.memories.items():
                    details = report['layers'][str(layer+1)]
                    needles = details['selected_needle_records']
                    if not needles:
                        continue
                    needle = needles[0]
                    alternatives = [int(s) for s in details['teacher_norms'] if int(s) not in needles]
                    distractor = max(alternatives, key=lambda s: details['teacher_norms'][str(s)]) if alternatives else None
                    for label,slot in [('needle',needle),('highest_teacher_other',distractor)]:
                        if slot is None:
                            continue
                        version = int(memory.memory.read_credit_version[slot])
                        with learner.observation(suppress_record=(layer,slot,version)), ExitStack() as stack:
                            for other,wrapper in learner.memories.items():
                                stack.enter_context(wrapper.attach(model,layer=other,capture=False))
                            modified = model(x,past_key_values=clone_cache(current_boundary),use_cache=True)
                        _, modified_ce = language_loss(modified.logits,target,mask)
                        removals.append(dict(layer=layer+1,kind=label,slot=slot,version=version,
                                             answer_ce=float(modified_ce), delta_ce=float(modified_ce-ce.detach())))
            report['researcher_finite_removals'] = removals
        gradient = {'matched':base, 'cut_old':{n:base[n]-matched[n] for n in names},
                    'mismatch_old':{n:base[n]-matched[n]+mismatched[n] for n in names}}
        next_cache = clone_cache(output.past_key_values)
        del output, ce, actual
        for memory in learner.memories.values():
            memory.current, memory.replayed, memory.replayed_by_slot, memory.last_read = None, [], {}, None
            memory.memory._read_participation = []
            memory.memory._read_credit_active = False
            memory.memory._read_credit_entries = 0
        # Frozen next-state sidecar: a first-order directional probe, not an
        # optimizer step or an estimate of a trained-model outcome.
        tail = row['tail']
        future_x = torch.tensor([row['gold'][-1]]+tail[:127], device='cuda')[None]
        future_y = torch.tensor(tail[:128], device='cuda')[None]
        with learner.observation(), ExitStack() as stack:
            for layer,memory in learner.memories.items():
                stack.enter_context(memory.attach(model,layer=layer,capture=False))
            future = model(future_x,past_key_values=next_cache,use_cache=True)
        _, future_ce = language_loss(future.logits,future_y)
        future_grad = torch.autograd.grad(future_ce,params,allow_unused=True)
        future_cpu = {n:(torch.zeros_like(p) if g is None else g.detach()).cpu()
                      for n,p,g in zip(names,params,future_grad)}
        def norm(vec):
            return sum(float(v.double().square().sum()) for v in vec.values())**.5
        report['paired'] = dict(parameter_names=names, future_ce=float(future_ce.detach()),
            matched_old_norm=norm(matched), mismatched_old_norm=norm(mismatched),
            common_forward=True, future_scope='128 actual subsequent OWT tokens, frozen no commit',
            controls={key:dict(gradient_norm=norm(value),
                predicted_change_per_unit_step=-sum(float((value[n].double()*future_cpu[n].double()).sum())
                                                    for n in names)) for key,value in gradient.items()})
    else:
        for layer, memory in learner.memories.items():
            selected = [(name, p) for name,p in learner.native if name.startswith(f'layer{layer}.')]
            report['layers'][str(layer+1)] = audit_reactivated_credit(ce, memory, selected)
    report['source_unchanged'] = old.sha256(source) == source_sha
    report['cuda_peak_bytes'] = torch.cuda.max_memory_allocated()
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False), encoding='utf-8')
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
