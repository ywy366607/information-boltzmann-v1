"""Real OWT continuous-state CBIM training with whole-update CUDA replay."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from scripts.ib_local.cbim_field import CBIMFieldModel
from scripts.ib_local.cbim_cuda_graph import CBIMGraphTrainer


def atomic_json(path, data, *, required=False):
    """Best-effort monitor writes cannot terminate a training run on Windows.

    Local HTTP viewers can briefly retain a file handle while refreshes race an
    atomic rename. Checkpoints remain the source of recovery; progress/UI files
    retry and then fail open.
    """
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    payload = json.dumps(data, allow_nan=False)
    for attempt in range(60):
        try:
            tmp.write_text(payload, encoding='utf-8')
            os.replace(tmp, path)
            return True
        except PermissionError:
            time.sleep(.05 * min(attempt + 1, 4))
    if required:
        raise PermissionError(f'Unable to update required file: {path}')
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('data/ib_owt_gpt2'))
    p.add_argument('--output', type=Path, default=Path('results/cbim_v2_3000'))
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--tokens', type=int, default=128)
    p.add_argument('--validate-every', type=int, default=250)
    p.add_argument('--validation-tokens', type=int, default=4096)
    p.add_argument('--resume', type=Path)
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(11)
    torch.cuda.set_per_process_memory_fraction(.48)
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output/'last.pt').exists() and not a.resume:
        raise ValueError('Existing run requires --resume or a new output directory')
    train = np.load(a.data/'train.npy', mmap_mode='r')
    valid = np.load(a.data/'validation.npy', mmap_mode='r')
    assert a.steps*a.tokens+1 <= len(train)
    assert a.validation_tokens % a.tokens == 0
    config = {k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()}
    config.update(architecture='CBIM-v2', grid=64, channels=128, queries=4,
                  precision='FP32', seed=11, lr=3e-4,
                  manifest=json.loads((a.data/'manifest.json').read_text()),
                  stopping='fixed update budget; convergence not established')
    config['source_sha256'] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in ['scripts/ib_local/cbim_field.py','scripts/ib_local/cbim_cuda_graph.py',__file__]}
    atomic_json(a.output/'config.json', config, required=True)
    atomic_json(a.output/'progress.json', {'status':'capturing','step':0,'target_steps':a.steps})
    model = CBIMFieldModel().cuda()
    runner = CBIMGraphTrainer(model, tokens=a.tokens)
    step, best = 0, float('inf')
    if a.resume:
        saved = torch.load(a.resume, map_location='cuda', weights_only=False)
        model.load_state_dict(saved['model'])
        # Copy into graph-owned optimizer storage rather than replacing tensors.
        current = runner.optimizer.state_dict()
        for key, state in saved['optimizer']['state'].items():
            for name, value in state.items():
                current['state'][key][name].copy_(value)
        runner.state.copy_(saved['state'])
        step, best = saved['step'], saved['best_validation_nll']

    def batch(data, offset):
        return tuple(torch.as_tensor(np.array(data[offset+i:offset+i+a.tokens]),
                      dtype=torch.long, device='cuda')[None] for i in (0,1))

    def save(name):
        temp = a.output/(name+'.tmp')
        torch.save(dict(model=model.state_dict(), optimizer=runner.optimizer.state_dict(),
            state=runner.state.detach(), step=step, events=step*a.tokens,
            best_validation_nll=best, config=config), temp)
        os.replace(temp, a.output/name)

    def log(row):
        with (a.output/'metrics.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row),flush=True)

    @torch.no_grad()
    def validate():
        h = torch.zeros_like(runner.state)
        # Separate validation state; first window is burn-in, no training-state reset.
        total = 0.
        for offset in range(0,a.validation_tokens+a.tokens,a.tokens):
            ids, targets = batch(valid,offset)
            loss,h,_ = model(ids,targets,h)
            if offset:
                total += float(loss)*a.tokens
        return total/a.validation_tokens

    try:
        if not a.resume:
            best=validate();log(dict(kind='validation',step=0,validation_nll=best))
            save('BBest.pt');save('last.pt')
        while step < a.steps:
            ids,targets=batch(train,step*a.tokens)
            torch.cuda.synchronize(); started=time.perf_counter()
            loss, state, diagnostics=runner.step(ids,targets)
            torch.cuda.synchronize(); elapsed=time.perf_counter()-started
            step+=1
            value=float(loss)
            if not math.isfinite(value):
                raise FloatingPointError('Nonfinite training loss')
            if step==1 or step%10==0 or step==a.steps:
                h=state.detach()
                row=dict(kind='train',step=step,events=step*a.tokens,nll=value,seconds=elapsed,
                    energy=float(diagnostics['final_energy']),
                    clamp_trigger_rate=float(diagnostics['clamp_trigger_rate']),
                    grad_norm=float(runner.grad_norm),
                    spatial_variance=float(h.var(1,unbiased=False).mean()),
                    allocated_mib=torch.cuda.memory_allocated()/2**20,
                    reserved_mib=torch.cuda.memory_reserved()/2**20)
                log(row)
                atomic_json(a.output/'progress.json',dict(status='running',target_steps=a.steps,**row))
                atomic_json(a.output/'live_state.json',dict(**row,field=h[0].cpu().tolist()))
            if step%a.validate_every==0 or step==a.steps:
                score=validate();improved=score<best
                if improved:best=score
                log(dict(kind='validation',step=step,validation_nll=score,best_validation_nll=best))
                if improved:save('BBest.pt')
                save('last.pt');save(f'age_{step:06d}.pt')
        atomic_json(a.output/'progress.json',dict(status='complete',step=step,target_steps=a.steps,best_validation_nll=best))
    except BaseException as exc:
        atomic_json(a.output/'progress.json',dict(status='failed',step=step,error=str(exc)))
        raise


if __name__=='__main__':
    main()
