"""Bounded real-OWT checkpoint profiling; never writes the source individual."""
import argparse
from pathlib import Path
import sys
import json
import time
import functools
from contextlib import contextmanager
from collections import defaultdict

TIMINGS = defaultdict(list)
MEASURING = False
USE_CUDA = True

@contextmanager
def section(label):
    if not MEASURING:
        yield
        return
    begin, end = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if USE_CUDA else (None, None)
    if begin is not None:
        begin.record()
    started = time.perf_counter()
    try:
        yield
    finally:
        if end is not None:
            end.record()
        TIMINGS[label].append((begin, end, time.perf_counter()-started))

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fine_grain.information_boltzmann import InformationBoltzmann
from fine_grain.information_boltzmann.streaming import StreamRunner
from scripts.train_information_boltzmann_lifetime import TensorBudget, event_observables


def annotate(obj, name, label):
    original = getattr(obj, name)
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        with section(label):
            return original(*args, **kwargs)
    setattr(obj, name, wrapped)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    p.add_argument('--no-accounting', action='store_true', help='Measure optional energy-accounting overhead; does not change dynamics')
    p.add_argument('--shared-projections', action='store_true')
    p.add_argument('--tokens', type=int, default=16)
    p.add_argument('--save-comparison', action='store_true')
    p.add_argument('--fused-ou', action='store_true')
    p.add_argument('--slice-channels',type=int,default=0)
    p.add_argument('--particles',type=int,default=32)
    p.add_argument('--checkpoint-drive',action='store_true')
    args = p.parse_args()
    global USE_CUDA
    USE_CUDA = args.device == 'cuda'
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.time()
    def save(value):
        (args.output/'status.json').write_text(json.dumps(value, indent=2), encoding='utf-8')
    save({'start_wall_time': start, 'status': 'running', 'concurrent_with_baseline': False})
    try:
        torch.set_num_threads(1)
        if USE_CUDA:
            torch.cuda.set_per_process_memory_fraction(.60)
        cfg = json.loads(args.config.read_text(encoding='utf-8'))
        torch.manual_seed(cfg['seed'])
        manifest = json.loads((args.data/'manifest.json').read_text(encoding='utf-8'))
        model = InformationBoltzmann.from_config(cfg).to(args.device)
        tr = cfg['train']
        opt = torch.optim.AdamW(model.parameters(), lr=tr['learning_rate'], weight_decay=tr['weight_decay'], foreach=USE_CUDA)
        runner = StreamRunner(model, manifest['bos_token'], cfg['seed'], opt, tr['update_every_real_tokens'], tr['gradient_clip'])
        if USE_CUDA:
            runner.load(args.checkpoint)
        else:
            # Cross-device runtime comparison: CUDA RNG state is not CPU compatible.
            # Preserve learned parameters and physical state, use a declared CPU seed.
            from fine_grain.information_boltzmann.state import PhaseState
            saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
            model.load_state_dict(saved['model'])
            opt.load_state_dict(saved['optimizer'])
            runner.state = PhaseState(**saved['state'])
            for name in ('current_token','events','updates','total_nll','candidates','accepted','cross_moment_abs_sum'):
                setattr(runner, name, saved[name])
            runner.generator.manual_seed(cfg['seed'])
            del saved
        if args.particles != runner.state.x.shape[0]:
            from fine_grain.information_boltzmann.state import PhaseState
            n=runner.state.x.shape[0]
            if args.particles%n:raise ValueError('Runtime expansion must be a multiple of saved particles')
            runner.state=PhaseState(runner.state.x.repeat(args.particles//n,1),runner.state.v.repeat(args.particles//n,1),runner.state.time)
        if args.slice_channels:
            from scripts.ib_slice_collision import install_slice_collision
            old=set(model.collision.parameters())
            install_slice_collision(model,args.slice_channels)
            for group in opt.param_groups:group['params']=[p for p in group['params'] if p not in old]
            for p in old:opt.state.pop(p,None)
            opt.add_param_group({'params':list(model.collision.parameters())})
        if args.shared_projections:
            from scripts.ib_shared_projections import install_shared_projections
            install_shared_projections(model,collision_context=not bool(args.slice_channels),checkpoint_drive=args.checkpoint_drive)
        if args.fused_ou:
            from scripts.ib_fused_ou import install_fused_ou
            install_fused_ou(model)
        structural_check=None
        if args.slice_channels:
            with torch.no_grad():
                out,_,_=model.collision(runner.state,model.event_interval/model.steps)
                structural_check={'initial_relative_velocity_change':float((out.v-runner.state.v).norm()/runner.state.v.norm()),'initial_momentum_max_error':float((out.v.sum(0)-runner.state.v.sum(0)).abs().max()),'initial_relative_energy_error':float((out.v.square().sum()-runner.state.v.square().sum()).abs()/runner.state.v.square().sum())}
        initial = runner.events
        for obj, name, label in [(model.force,'transport','stage/transport'), (model.force,'drive','stage/drive'), (model.collision,'forward','stage/collision'), (model.collision,'rate','stage/collision_rate'), (model,'decode','stage/decode'), (runner,'flush','stage/backward_and_update'), (opt,'step','stage/optimizer')]:
            if hasattr(obj,name):annotate(obj,name,label)
        tokens = np.load(args.data/'train.npy', mmap_mode='r')
        def event():
            with section('stage/predict'):
                runner.predict(None if args.no_accounting else TensorBudget())
            with section('stage/observe'):
                runner.observe(int(tokens[runner.events]))
            with section('stage/observables'):
                obs=event_observables(model, runner.state)
                torch.stack(list(obs.values())).detach().cpu()
        for _ in range(16):
            event()
        if USE_CUDA:
            torch.cuda.synchronize()
        t=time.perf_counter()
        global MEASURING
        MEASURING = True
        for _ in range(args.tokens):
            event()
        if USE_CUDA:
            torch.cuda.synchronize()
        elapsed=time.perf_counter()-t
        rows=[{'name':key,'calls':len(values),'cpu_wall_ms':sum(v[2] for v in values)*1000,'cuda_stream_elapsed_ms':(sum(v[0].elapsed_time(v[1]) for v in values) if USE_CUDA else None)} for key, values in TIMINGS.items()]
        (args.output/'stages.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        if args.save_comparison:
            torch.save({'model':model.state_dict(),'x':runner.state.x,'v':runner.state.v,'rng':runner.generator.get_state(),'nll':runner.total_nll,'accepted':runner.accepted},args.output/'comparison.pt')
        if structural_check is not None:
            structural_check['final_collision_gradient_norm']=float(torch.sqrt(sum(p.grad.square().sum() for p in model.collision.parameters() if p.grad is not None)))
        save({'status':'complete','start_wall_time':start,'end_wall_time':time.time(),'checkpoint':str(args.checkpoint),'initial_event':initial,'warmup_tokens':16,'profiled_tokens':args.tokens,'structural_check':structural_check,'particles':args.particles,'slice_channels':args.slice_channels,'shared_projections':args.shared_projections,'fused_ou':args.fused_ou,'checkpoint_drive':args.checkpoint_drive,'profiled_seconds':elapsed,'device':args.device,'accounting':not args.no_accounting,'peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20 if USE_CUDA else 0,'concurrent_with_baseline':False,'limitation':'Nested CPU-wall/CUDA-event intervals include submission gaps; not exclusive kernel time or capability evidence. CPU and CUDA random paths differ. Excludes file logging, live UI and periodic diagnostics.'})
    except Exception as exc:
        save({'status':'failed','start_wall_time':start,'end_wall_time':time.time(),'error':repr(exc),'concurrent_with_baseline':False})
        raise

if __name__=='__main__':
    main()
