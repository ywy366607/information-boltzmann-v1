"""Real OWT numerical audit of stochastic credit against exact two-event AD.

This audits derivative compression and fixed live storage, not task capability.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

from information_boltzmann.core.plastic_ports import PlasticMediumPorts3D
from information_boltzmann.runtime.online_credit import (
    OnlinePlasticTrainer, credit_tensors, replace_credit, squared_norm)
from information_boltzmann.runtime.training import belief_tensors, quiet_training_chunk
from scripts.ib.train_plastic_conductance import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('data/ib_owt_gpt2'))
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--events', type=int, default=64)
    parser.add_argument('--event-duration', type=float, default=.005)
    args = parser.parse_args()
    if min(args.samples,args.events) < 2:
        parser.error('At least two samples/events required')
    torch.set_num_threads(1)
    torch.manual_seed(449)
    torch.cuda.set_per_process_memory_fraction(.6)
    model = PlasticMediumPorts3D(bath_type='conductance',activity_adaptation=True,
                                 short_term_plasticity=True).cuda()
    learner = OnlinePlasticTrainer(model,event_duration=args.event_duration)
    corpus = np.load(args.data_dir/'train.npy',mmap_mode='r')
    sequence = torch.from_numpy(np.array(corpus[:args.events+1],dtype=np.int64)).cuda()
    parameters = learner.parameters
    initial = learner.belief
    loss1, state1, _ = quiet_training_chunk(model,sequence[:1].reshape(1,1),sequence[1:2].reshape(1,1),
                                           initial,event_duration=args.event_duration)
    state1_tensors = credit_tensors(state1)
    leaves = tuple(x.detach().requires_grad_() for x in state1_tensors)
    loss2, _, _ = quiet_training_chunk(model,sequence[1:2].reshape(1,1),sequence[2:3].reshape(1,1),
                                       replace_credit(state1,leaves),event_duration=args.event_duration)
    c_values = torch.autograd.grad(loss2,leaves,allow_unused=True)
    c = tuple(torch.zeros_like(x) if g is None else g.detach() for x,g in zip(leaves,c_values))
    active = [(i,x) for i,x in enumerate(state1_tensors) if x.requires_grad]
    def vjp(probe):
        values = torch.autograd.grad(tuple(x for _,x in active),parameters,
            grad_outputs=tuple(probe[i] for i,_ in active),allow_unused=True,retain_graph=True)
        return tuple(torch.zeros_like(p) if g is None else g.detach() for p,g in zip(parameters,values))
    exact = vjp(c)
    mean = [torch.zeros_like(p) for p in parameters]
    sample_norms, errors, cosines = [],[],[]
    exact_norm = squared_norm(exact).sqrt()
    for index in range(args.samples):
        signs = tuple(torch.empty_like(x).bernoulli_(.5,generator=learner.generator).mul_(2).sub_(1)
                      for x in leaves)
        projected = vjp(signs)
        credit = sum((x*y).sum() for x,y in zip(c,signs))
        sample = tuple(x*credit for x in projected)
        sample_norm = squared_norm(sample).sqrt()
        error = squared_norm(tuple(a-b for a,b in zip(sample,exact))).sqrt()
        cosine = sum((a*b).sum() for a,b in zip(sample,exact))/(sample_norm*exact_norm)
        sample_norms.append(float(sample_norm)); errors.append(float(error/exact_norm))
        cosines.append(float(cosine))
        for average,value in zip(mean,sample):
            average.add_(value/args.samples)
    mean_error = float(squared_norm(tuple(a-b for a,b in zip(mean,exact))).sqrt()/exact_norm)
    derivative = {'exact_two_event_history_norm':float(exact_norm),
                  'samples':args.samples,'rank_one_norm_mean':sum(sample_norms)/len(sample_norms),
                  'rank_one_relative_error_mean':sum(errors)/len(errors),
                  'rank_one_cosine_mean':sum(cosines)/len(cosines),
                  'sample_mean_relative_error':mean_error,
                  'claim':'Finite-sample derivative variance only; not a convergence/capability verdict.'}
    del loss1,loss2,state1,state1_tensors,leaves,active,c,exact,mean,projected,sample
    torch.cuda.empty_cache()
    sizes, live_memory, graph_free, rows = [],[],[],[]
    torch.cuda.synchronize()
    started = time.perf_counter()
    # Frozen parameters isolate eligibility evolution from optimizer changes.
    for index in range(args.events):
        model.zero_grad(set_to_none=True)
        result = learner.backward_event(sequence[index:index+1],sequence[index+1:index+2])
        sizes.append(learner.eligibility_bytes())
        tensors = list(belief_tensors(learner.belief))
        tensors += [t for group in (learner.state_factors,learner.parameter_factors)
                    for factor in group for t in factor]
        graph_free.append(all(t.grad_fn is None and not t.requires_grad for t in tensors))
        model.zero_grad(set_to_none=True)
        if (index+1) % 8 == 0:
            torch.cuda.synchronize()
            live_memory.append({'event':index+1,'allocated_mib':torch.cuda.memory_allocated()/2**20})
            rows.append({key:float(value) for key,value in result.items()})
            print(json.dumps({'event':index+1,'live':live_memory[-1],**rows[-1]}),flush=True)
    report = {'schema':1,'protocol':'Numerical derivatives and live-memory audit on real OWT; frozen weights',
              'parameters':sum(p.numel() for p in parameters),'shape':[8,8,4],'channels':128,
              'events':args.events,'event_duration':args.event_duration,'rank':learner.rank,
              'derivative_compression':derivative,'eligibility_bytes_min':min(sizes),
              'eligibility_bytes_max':max(sizes),'all_persisted_tensors_graph_free':all(graph_free),
              'live_memory':live_memory,'event_metrics':rows,
              'seconds_per_event':(time.perf_counter()-started)/args.events,
              'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20}
    atomic_json(args.output,report)
    print(json.dumps(report,allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
