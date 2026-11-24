"""Frozen real-checkpoint layer activity and memory-format audit; no training."""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve prior audit')
    torch.set_num_threads(1)
    with gzip.open(args.checkpoint, 'rb') as stream:
        saved = torch.load(stream, map_location='cpu', weights_only=False)
    model, engine, config = saved['model'], saved['engine'], saved['config']
    report = dict(cycle=int(saved['cycle']), checkpoint=str(args.checkpoint),
        scope='frozen production TableGDN language path on actual OWT bytes; no planner or training',
        layers=[], memory={}, original_world_cursor=int(saved['world']['cursor']))
    for i in range(32):
        root = f'body.base.layers.{i}.'
        item = dict(layer=i+1)
        for label, key in (('attention_output', 'o_proj.weight'), ('mlp_down', 'mlp.down.weight')):
            w = model[root+key]
            item[label] = dict(rms=float(w.square().mean().sqrt()),
                nonzero_fraction=float((w != 0).float().mean()), norm=float(w.norm()))
        report['layers'].append(item)
    for name in ('memory.source', 'memory.action', 'memory.successor', 'memory.long_successor',
                 'memory.internal_next', 'memory.retrieval.words', 'momentum_to_boundary.weight'):
        if name in model:
            report['memory'][name] = dict(shape=list(model[name].shape), dtype=str(model[name].dtype))
    report['memory']['table_parameters'] = {k:v for k,v in saved['table'].items() if k != 'arrays'}
    report['memory']['engine_windows'] = {k:engine.get(k) for k in ('response_bytes','holdout_bytes')}
    report['memory']['memory_extra'] = model.get('memory._extra_state')
    report['memory']['retrieval_buffers'] = {
        k:list(v.shape) for k,v in model.items() if torch.is_tensor(v)
        and ('retriev' in k or 'utterance' in k)}
    table = SimpleNamespace(**saved['table']['config'],
        **{k:v.numpy() for k,v in saved['table']['arrays'].items() if k in ('rows','count','tag','err')})
    state = engine['recurrent'].clone()
    weights = {k[len('body.'):]:v for k,v in model.items() if k.startswith('body.')}
    backbone = config['body_growth']['backbone']
    # Retain only the small inference backbone. Tables, optimizer and traces
    # are released before allocating GPU data; the live individual is untouched.
    del model, engine, config, saved
    gc.collect()
    from scripts.ib_local import gdn_fused
    from scripts.ib_local.gated_deltanet import GatedDeltaNetLM
    from scripts.ib_local.train_gdn_fast_table import TableGDN
    from scripts.ib_local.navigation_live_witness import frozen_table_logp
    gdn_fused.enable()
    lm = TableGDN(GatedDeltaNetLM(**backbone),True,1).cuda().eval()
    lm.load_state_dict(weights, strict=True)
    del weights
    state = state.cuda()
    root = Path(__file__).resolve().parents[2]
    offset = json.loads((root/'data/ib_owt_10000/tokenizer.json').read_text())['offset']
    raw = np.load(root/'data/ib_owt_10000/validation.npy', mmap_mode='r')
    raw = np.asarray(raw[:2*(262144+1025)])
    text = (raw[raw >= offset]-offset)[262144:262144+1025].astype(np.int64)
    ids = torch.as_tensor(text[:-1], device='cuda')[None]
    targets = torch.as_tensor(text[1:], device='cuda')[None]
    table_logp = torch.as_tensor(frozen_table_logp(table,text[:-1].astype(np.uint8)),device='cuda')[None]
    del table
    measurements, bypass, handles = {}, set(), []
    for i, layer in enumerate(lm.base.layers):
        def capture(module, inputs, output, index=i):
            h = inputs[0]
            if not bypass:
                measurements[index] = torch.stack((h.square().mean().sqrt(),
                    (output[0]-h).square().mean().sqrt()))
            if index in bypass:
                return h, output[1], output[2]
        handles.append(layer.register_forward_hook(capture))
    with torch.no_grad():
        logits, _, _ = lm.predict(ids, state.clone(), table_logp)
        report['backbone_nll'] = float(F.cross_entropy(logits.flatten(0,1),targets.flatten()))
        measured = torch.stack([measurements[i] for i in range(32)]).cpu().tolist()
        for item, (before, delta) in zip(report['layers'], measured):
            item.update(input_rms=before, residual_rms=delta, residual_input_ratio=delta/max(before,1e-12))
        bypass.update(range(4,32))
        logits_without, _, _ = lm.predict(ids, state.clone(), table_logp)
        report['without_new_residuals_backbone_nll'] = float(F.cross_entropy(
            logits_without.flatten(0,1), targets.flatten()))
        report['logit_rms_without_new_residuals'] = float((logits-logits_without).square().mean().sqrt())
    for handle in handles:
        handle.remove()
    report['validation_bytes'] = len(text)-1
    report['state_scope'] = 'same saved recurrent state; only new residual outputs bypassed; no parameter/state write'
    digest = hashlib.sha256()
    with args.checkpoint.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            digest.update(block)
    report['checkpoint_sha256'] = digest.hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('layers','memory')},ensure_ascii=False))
    print(json.dumps(report['layers'],ensure_ascii=False))
    print(json.dumps(report['memory'],ensure_ascii=False))


if __name__ == '__main__':
    main()
