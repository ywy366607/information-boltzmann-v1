"""Pre-registered frozen factual-source audit: local transport versus shared K2.

Direct native model forward, inherited S, no extra recall hooks, no optimizer.
"""
import argparse
from functools import partial
import gc
import json
import os
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--all-heads', action='store_true', help='P03: all eight A_log directions, discovery-stage coverage')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True)
    for key, suffix in [('HF_HOME','huggingface'),('TRANSFORMERS_CACHE','transformers'),
                        ('TRITON_CACHE_DIR','triton'),('TORCHINDUCTOR_CACHE_DIR','inductor'),
                        ('TEMP','tmp'),('TMP','tmp')]:
        os.environ[key] = 'D:/ml_cache/' + suffix
    os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                      HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from scripts.ib_local import run_native_s_credit_retention as old
    from scripts.ib_local import pretrained_gdn_stream_load as loading
    from scripts.ib_local.frozen_mlp_int8 import load_streamed_int8_frozen
    from scripts.ib_local.readonly_checkpoint import load_readonly_checkpoint
    from scripts.ib_local.native_s_learner import NativeSEligibilityLearner
    from scripts.ib_local.pretrained_gdn import clone_cache
    from scripts.ib_local.run_pretrained_gdn_cross_window import read_text
    from scripts.ib_local.water_source_audit import WaterSourceAudit
    from scripts.ib_local.native_s_storage import stop_future_joint_archives
    source = root/'scratch/joint_next_pending_20261108/pending_recovery.pt'
    protocol = root/'docs/information_boltzmann/CGB_WATER_FINAL_DIAGNOSIS_20261109.md'
    files = [Path(__file__), root/'scripts/ib_local/water_source_audit.py',
             root/'scripts/ib_local/native_s_eligibility.py', root/'scripts/ib_local/bounded_causal_tag_blocked.py']
    report = dict(status='loading', protocol_sha256=old.sha256(protocol),
        execution_sources={str(f):old.sha256(f) for f in files},
        source_sha256=old.sha256(source), source_updates=524,
        optimizer_updates=0, main_life_updates=0, added_retrieval_hooks=False,
        inherited_pending='preserved in source; no learning or synthetic settlement in frozen branch',
        claim_scope=('all eight A_log conditional held-input directions' if args.all_heads else
                     'two conditional held-input local directions, not all-network credit'),
        records=[])
    start = time.monotonic()
    def save():
        report['seconds'] = time.monotonic()-start
        old.write_json(args.output/'report.json',report)
    save()
    runtime = argparse.Namespace(vendor=Path('E:/cgb_gdn340_probe/vendor/flash-linear-attention-0.2.2'),
        model=Path('E:/cgb_gdn340_probe/model'), bounded_autotune=True)
    original = loading.load_streamed_pure_gdn
    loading.load_streamed_pure_gdn = partial(load_streamed_int8_frozen,
        mapped_directory=root/'scratch/frozen_weight_cache_20261109')
    try:
        _,_,_,model,weight_loading = old.load_runtime(runtime)
    finally:
        loading.load_streamed_pure_gdn = original
    report['weight_loading'] = weight_loading
    owner = load_readonly_checkpoint(source)
    state = owner['learner']
    learner = NativeSEligibilityLearner(model,layers=state['layers'],capacity=state['capacity'])
    tokenizer = AutoTokenizer.from_pretrained(runtime.model,local_files_only=True)
    data = root/'data/ib_owt_10000'
    offset = old.read_json(data/'tokenizer.json')['offset']
    try:
        for raw_start in (12_000_000,14_000_000):
            if (args.output/'STOP').exists():
                raise RuntimeError('explicit diagnostic STOP')
            learner.load_state_dict(state)
            stop_future_joint_archives(learner)
            learner.enable_native_s_eligibility()
            specs = learner.eligibility.specs
            selected = ([(l,h) for l,s in specs.items() for h in range(s['heads'])]
                        if args.all_heads else None)
            audit = WaterSourceAudit(specs,rank=2,block_size=65536,max_bytes=64*2**20,selected=selected)
            learner.eligibility = audit
            versions = [x._version for x in model.parameters()]
            text,digest = read_text(data/'train.npy',offset,raw_start,length=196608)
            ids = tokenizer.encode(text,add_special_tokens=False)
            if len(ids)<769:
                raise ValueError('registered source too short')
            record = dict(raw_start=raw_start,raw_units=196608,text_sha256=digest,
                incoming_seen=learner.cache._seen_tokens, snapshots=[],optimizer_updates=0,
                uncompressed_reference_bytes=sum(d.nbytes for d in audit.exact),
                reference_storage_scope='extra researcher reference, not production eligibility budget')
            report['records'].append(record)
            report['status'] = 'running'
            save()
            for begin in range(0,640,128):
                if (args.output/'STOP').exists():
                    raise RuntimeError('explicit diagnostic STOP')
                x = torch.tensor([ids[begin:begin+128]],device='cuda')
                with learner.factual_capture(0),torch.no_grad():
                    output = model(x,past_key_values=learner.cache,use_cache=True)
                learner.cache = clone_cache(output.past_key_values)
                del output,x
                audit.set_end_s(learner._initial_states(learner.cache))
                snapshot = audit.advance_window()
                learner._native_actual_count = 0
                # A separate short current CE branch supplies teaching without
                # updating the lived cache, parameters, code, or RNG schedule.
                if begin+128 >= 256:
                    cache = clone_cache(learner.cache)
                    leaves = []
                    for layer in specs:
                        leaf = cache.states[layer]['recurrent_state'].detach().clone().requires_grad_(True)
                        cache.states[layer]['recurrent_state'] = leaf
                        leaves.append(leaf)
                    with torch.enable_grad():
                        y = torch.tensor([ids[begin+129:begin+161]],device='cuda')
                        # Official <=64 fused-recurrent kernel has no backward.
                        # A causal128 chunk supports VJP; score only first32.
                        inputs = torch.tensor([ids[begin+128:begin+256]],device='cuda')
                        prediction = model(inputs,past_key_values=cache,use_cache=True)
                        loss = 10*torch.nn.functional.cross_entropy(prediction.logits[:,:32].float().flatten(0,1),y.flatten())
                        gradients = torch.autograd.grad(loss,leaves)
                    teaching = {l:g.detach().float().cpu().numpy()[0] for l,g in zip(specs,gradients)}
                    snapshot['current_ce'] = float(loss.detach()/10)
                    snapshot['teaching'] = audit.contract_first_window(teaching)
                    del cache,leaves,leaf,loss,gradients,prediction,inputs,y,teaching
                record['snapshots'].append(snapshot)
                print(json.dumps(dict(raw_start=raw_start,tokens=begin+128,
                    directions=snapshot['directions'],teaching=snapshot.get('teaching'))),flush=True)
                save()
                gc.collect()
            record['parameters_unchanged'] = versions == [x._version for x in model.parameters()]
            if not record['parameters_unchanged'] or learner.updates != 524:
                raise AssertionError('frozen diagnostic modified slow parameters')
            report['peak_cuda_allocated'] = torch.cuda.max_memory_allocated()
        report.update(status='complete',source_unchanged=old.sha256(source)==report['source_sha256'],
            execution_sources_unchanged=all(old.sha256(f)==h for f,h in report['execution_sources'].items()),
            protocol_unchanged=old.sha256(protocol)==report['protocol_sha256'])
    except BaseException as exc:
        report.update(status='failed',error=repr(exc))
        raise
    finally:
        save()


if __name__=='__main__':
    main()
