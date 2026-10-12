"""Where does a training window's time go?  A fresh MaleCNS model (as in the tests), one 32-word window at the world's
pace (mean 8 ticks/word, read every tick), forward + backward, under four configurations:
  base | + dopamine modulation (site pre) | + compartment memory | + odour world and act head
Timed after one warm-up window; also the torch.compile recompilation counters, and a torch.profiler table of the
top CUDA/CPU ops for the slowest configuration.
Usage: python profile_window.py
"""
import sys, time, dataclasses, torch
sys.path.insert(0, 'D:/information-boltzmann-v1'); sys.path.insert(0, 'D:/information-boltzmann-v1/tests')
import test_fly_adaptive_admission as T
import information_boltzmann.core.fly_bptt_learning as fl
from information_boltzmann.core import fly_odour_world as ow

G, A = 'data/malecns_v1/fly_reservoir_coba.npz', 'data/malecns_v1/body-annotations.feather'
ids = torch.randint(0, 50000, (1, 32), device='cuda', generator=torch.Generator('cuda').manual_seed(0))
tg = torch.randint(0, 50000, (1, 32), device='cuda', generator=torch.Generator('cuda').manual_seed(1))


def build(level):
    model, state, device = T.fly_model_and_state.__wrapped__()
    if level >= 1:
        fl.attach_dopamine_modulation(model, site='pre')
    if level >= 2:
        fl.attach_compartment_memory(model, G, A, eta=-2.0, forget=0.999)
    world = None
    if level >= 3:
        init = torch.load('scratch/fly_stream/act_init.pt', map_location='cpu')
        ow.attach_act_head(model, init)
        world = ow.OdourWorld(model, G, A, seed=3)
    start = dataclasses.replace(state, h=torch.rand_like(state.h) * 0.05)
    L = fl.FlyBPTTLearner(model, start, lr=2e-4, settle_ticks=14, writer_baseline_clock='physical',
                          use_checkpointing=True, stream_ticks=1, lambda_mcr2=0.0)
    L.stream_pace_mean, L.pace_seed = 8.0, 1
    if world is not None:
        L.odour_world, L.act_lambda = world, 3e-4
    model.train()
    return model, L


def window(model, L):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    L.forward_window(ids, tg)
    loss = L.last_train_loss.mean() + (L.last_act_loss if getattr(L, 'last_act_loss', None) is not None else 0)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    loss.backward(); torch.cuda.synchronize(); t2 = time.perf_counter()
    model.zero_grad(set_to_none=True)
    return t1 - t0, t2 - t1, L.last_adaptive_metrics.get('adaptive_total_ticks')


names = ['base', '+dopamine', '+compartment', '+world']
slowest = None
for level, name in enumerate(names):
    model, L = build(level)
    window(model, L)                                   # warm-up (compilation)
    from torch._dynamo.utils import counters
    before = dict(counters['stats'])
    f, b, ticks = window(model, L)
    after = dict(counters['stats'])
    print(f'{name:14s} ticks {ticks}  forward {f:.2f}s  backward {b:.2f}s  per tick {(f + b) / ticks * 1000:.1f} ms  '
          f'dynamo stats delta {({k: after.get(k, 0) - before.get(k, 0) for k in after if after.get(k, 0) != before.get(k, 0)})}',
          flush=True)
    if level == 2:
        slowest = (model, L)
    else:
        del model, L
    torch.cuda.empty_cache()
model, L = slowest
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    window(model, L)
print(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=18))
print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=12))
