"""Read-only-checkpoint information/causal audit of the real OWT organism loop.

Standard learning runs only in disposable diagnostic branches. We never save a
new body or modify the source checkpoint. Neural state, both memories, optimizer,
pending credit, world and RNG all start from the same complete saved individual.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from scripts.ib_local import gdn_fused
from scripts.ib_local.closed_loop_information import analyze_trace
from scripts.ib_local.navigation_checkpoint import snapshot, restore
from scripts.ib_local.run_causal_water_reply import build


GROUPS = dict(water=[0, 1], wood=[2, 3], fire=[4, 5], earth=[6, 7],
              metal=[8, 9], world=[10, 11])
CHANNELS = ['recalled_progress', 'recall_entropy', 'forecast_match', 'forecast_gain',
            'p_generate', 'p_reuse', 'response_nll', 'short_latent_loss',
            'settled_gain', 'critic_rms', 'world_match_fraction', 'world_seen_fraction']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


@contextmanager
def patch_method(obj, name, replacement):
    existed, old = name in vars(obj), getattr(obj, name)
    setattr(obj, name, replacement)
    try:
        yield old
    finally:
        if existed:
            setattr(obj, name, old)
        else:
            delattr(obj, name)


@contextmanager
def cut_online_recall(engine):
    """Cut current causal-record delivery; preserve storage, byte table and learning.

    This ablation is not IIT's marginalizing causal cut. Explicit wood replay
    remains, so the diagnosis is of current online recall, not all memory use.
    """
    memory = engine.model.memory
    original_read = memory.read

    def blocked_read(features, encoder):
        result = original_read(features, encoder)
        return type(result)(result.value*0, result.long_value*0,
                            result.slots, result.weights*0)

    def blocked_outcomes(z, *args, **kwargs):
        return z.new_zeros(3, 2)

    with patch_method(memory, 'read', blocked_read), patch_method(
            engine, 'recall_outcomes', blocked_outcomes):
        yield


@contextmanager
def observe_decision(engine):
    """Record actual first decision-time recall without extra reads/RNG draws."""
    old_plan, old_utter = engine.plan, engine._utter
    capture = {}

    def plan(h):
        result = old_plan(h)
        if capture.get('recording') and 'water' not in capture:
            item = result[1]
            weights = item['weights'][0].detach()
            entropy = -(weights*weights.clamp_min(1e-30).log()).sum()
            capture.update(water=[float(item['progress'][0]), float(entropy)],
                           slots=item['slots'][0].detach().cpu().tolist(),
                           weights=weights.cpu().tolist())
        return result

    def utter():
        capture.clear()
        capture['recording'] = True
        try:
            return old_utter()
        finally:
            capture['recording'] = False

    with patch_method(engine, 'plan', plan), patch_method(engine, '_utter', utter):
        yield capture


def observables(row, captured, engine):
    choice = ('silence', 'generate', 'reuse').index(row['choice'])
    forecast = row['wood_all_predictions'][choice]
    result = (captured['water'] + [forecast[2], forecast[3]/10] +
              row['choice_probability'][1:] + [row['response_nll'], row['short_loss'],
              row['metal_g'], row['critic_rms'] or 0., row['world']['match']/32,
              row['world']['response_previously_seen_bytes']/engine.response_bytes])
    if len(result) != len(CHANNELS) or not np.isfinite(result).all():
        raise ValueError('nonfinite or malformed closed-loop projection')
    return result


def compare_traces(full, other):
    n = min(len(full), len(other))
    a, b = full[:n], other[:n]
    def first(field):
        return next((i+1 for i,(x,y) in enumerate(zip(a,b)) if x[field] != y[field]), None)
    scale = np.std([r['observables'] for r in a], axis=0)
    scale = np.maximum(scale, 1e-4)
    distances = np.abs((np.array([r['observables'] for r in b])-
                        np.array([r['observables'] for r in a]))/scale)
    return dict(turns=n, first_utterance_divergence=first('utterance_hex'),
                first_action_divergence=first('choice'),
                first_world_position_divergence=first('world_start'),
                different_utterances=sum(x['utterance_hex'] != y['utterance_hex'] for x,y in zip(a,b)),
                different_world_positions=sum(x['world_start'] != y['world_start'] for x,y in zip(a,b)),
                max_action_distribution_tv=max(.5*sum(abs(u-v) for u,v in zip(
                    x['choice_probability'],y['choice_probability'])) for x,y in zip(a,b)),
                mean_observable_distance=distances.mean(0).tolist(),
                mean_response_nll_difference=float(np.mean([y['response_nll']-x['response_nll'] for x,y in zip(a,b)])),
                note='After world divergence the NLLs score different inputs, not matched capability.')


def run_arm(engine, name, cycles, output):
    rows, start = [], time.perf_counter()
    with observe_decision(engine) as captured, (output/(name+'.jsonl')).open('w', encoding='utf-8') as stream:
        for i in range(cycles):
            row = engine.step(torch.randn(64, 32, device=engine.device))
            row.update(diagnostic_turn=i+1, observables=observables(row,captured,engine),
                       recall_slots=captured['slots'], recall_weights=captured['weights'],
                       world_start=row['world']['start'])
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
            stream.flush()
            if i < 2 or (i+1) % 16 == 0:
                print('ARM',name,i+1,round(time.perf_counter()-start,1),row['choice'],flush=True)
    return rows


def analyze(rows):
    trace = np.asarray([r['observables'] for r in rows])
    result = {str(lag):analyze_trace(trace,GROUPS,lag=lag) for lag in (1,2,4)}
    # Body-only projection: external input is then unobserved, explicitly labelled.
    result['body_only_lag1'] = analyze_trace(trace[:,:10], {k:v for k,v in GROUPS.items() if k!='world'})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cycles',type=int,default=256)
    parser.add_argument('--cut-cycles',type=int,default=128)
    parser.add_argument('--pulse-cycles',type=int,default=64)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    if min(args.cycles,args.cut_cycles,args.pulse_cycles) < 1:
        raise ValueError('positive horizons required')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    gdn_fused.enable()
    before = sha(args.checkpoint)
    engine, raw, offset, source = build(args.checkpoint)
    del raw
    # Keep all saved settlement/credit contracts; no objective migration.
    initial = snapshot(engine)
    sources = {str(Path(module.__file__)) for name,module in list(sys.modules.items())
               if name.startswith('scripts.ib_local.') and getattr(module,'__file__',None)}
    config = dict(checkpoint=source,source_sha256=before,cycles=args.cycles,
                  cut_cycles=args.cut_cycles,pulse_cycles=args.pulse_cycles,
                  groups=GROUPS,channels=CHANNELS,
                  primary_metric='lag1 projected Gaussian Phi_G, all 31 raw bipartitions',
                  primary_validation='chronological train50%, cut selection25%, test25%',
                  regularization=1e-4,pulse='one actually recalled slot preference += 0.1 once',
                  optimizer='original complete branch updates; disposable only; source never saved',
                  read_cut='online memory.read values/weights and recall_outcomes zero; explicit replay preserved',
                  caveat='Phi_G is a projected Gaussian measure, not exact IIT3/4 or consciousness',
                  sources={p:sha(p) for p in sorted(sources)})
    frozen=args.output/'executed_sources'; frozen.mkdir()
    for p in sources:
        shutil.copyfile(p,frozen/Path(p).name)
    write_json(args.output/'config.json',config)
    full = run_arm(engine,'full',args.cycles,args.output)
    # Repeat the first actual complete turn and compare every recorded scalar.
    restore(engine,initial)
    repeat = run_arm(engine,'repeat',1,args.output)
    identical = full[0] == repeat[0]
    if not identical:
        raise AssertionError('complete start restore failed: '+str([k for k in full[0] if full[0][k]!=repeat[0].get(k)]))
    restore(engine,initial)
    with cut_online_recall(engine):
        cut = run_arm(engine,'cut_online_water',args.cut_cycles,args.output)
    restore(engine,initial)
    slots,weights=full[0]['recall_slots'],full[0]['recall_weights']
    slot=int(slots[int(np.argmax(weights))])
    with torch.no_grad():
        engine.model.memory.read_credit_mean[slot].add_(.1)
    pulse = run_arm(engine,'pulse_water',args.pulse_cycles,args.output)
    del engine,initial
    gc.collect(); torch.cuda.empty_cache()
    result=dict(scope='disposable full real OWT body/world diagnostic branches',
                start_repeat_exact=identical,source_checkpoint_unchanged=sha(args.checkpoint)==before,
                cut=compare_traces(full,cut),pulse=compare_traces(full,pulse),pulse_slot=slot,
                neural_training='only standard isolated branch updates; main checkpoint unchanged',
                collected_turns=dict(full=len(full),cut=len(cut),pulse=len(pulse)),
                metrics_definition='Geometric Phi_G, SI, and held-out lagged predictive cut penalty are distinct.')
    if not args.smoke:
        result['information']=analyze(full)
        if len(cut)>=96:
            result['cut_information']=analyze(cut)
    write_json(args.output/'result.json',result)
    assert result['source_checkpoint_unchanged']
    print('COMPLETE',str(args.output/'result.json'),flush=True)


if __name__=='__main__':
    main()
