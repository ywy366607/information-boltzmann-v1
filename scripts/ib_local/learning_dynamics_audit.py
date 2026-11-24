"""Bounded, read-only observation of REAL optimizer updates and layer traffic.

No model/optimizer/RNG state is changed by the observer. Actual parameter
displacements are sampled at deterministic coordinates; gradient/weight norms
use the full tensors. Hooks are removed after the declared event budget.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import statistics

import torch


def _mean(values):
    return statistics.fmean(values) if values else None


class LearningDynamicsAudit:
    def __init__(self, model, optimizer, output, cycles, *, samples=128):
        self.output, self.budget = Path(output), int(cycles)
        if self.budget < 1 or samples < 1 or self.output.exists():
            raise ValueError('positive bounded audit and a new output required')
        self.model, self.optimizer = model, optimizer
        self.original_step, self.original_objective = optimizer.step, model.objective
        self.active, self.in_objective = True, False
        self.pending, self.events, self.gradient_probes = [], [], {}
        self.counters, self.handles = [], []
        self.entries = []
        groups = {id(p): group for group in optimizer.param_groups for p in group['params']}
        for name, p in model.named_parameters():
            if not p.requires_grad or not p.numel():
                continue
            match = re.match(r'body\.base\.layers\.(\d+)\.(.+)', name)
            layer, family = (int(match[1])+1, match[2]) if match else (0, name)
            n = min(samples, p.numel())
            index = torch.arange(n, device=p.device) * max(p.numel()-1, 0) // max(n-1, 1)
            self.entries.append((name, p, layer, family, index, groups[id(p)]))
        optimizer.step = self._step

        def objective(*args, **kwargs):
            previous = self.in_objective
            self.in_objective = True
            try:
                return self.original_objective(*args, **kwargs)
            finally:
                self.in_objective = previous
        model.objective = objective
        for index, layer in enumerate(model.body.base.layers):
            self.handles.append(layer.register_forward_hook(self._layer_hook(index+1)))

    def _layer_hook(self, layer):
        def capture(module, inputs, output):
            if not self.in_objective or not torch.is_grad_enabled():
                return
            h, following = inputs[0], output[0]
            with torch.no_grad():
                values = torch.stack((h.float().square().mean().sqrt(),
                    following.float().square().mean().sqrt(),
                    (following-h).float().square().mean().sqrt())).detach()
            record = dict(layer=layer, values=values, backward=[])
            self.counters.append(record)
            if following.requires_grad:
                def backward(gradient):
                    record['backward'].append(gradient.detach().float().square().mean().sqrt())
                following.register_hook(backward)
        return capture

    def _step(self, *args, **kwargs):
        before, statistics_before, identities = [], [], []
        probes = []
        with torch.no_grad():
            for name, p, layer, family, index, group in self.entries:
                g = p.grad
                if g is None:
                    continue
                state = self.optimizer.state.get(p, {})
                m, v = state.get('exp_avg'), state.get('exp_avg_sq')
                norm = g.float().norm()
                mnorm = m.float().norm() if m is not None else norm.new_zeros(())
                dot = (g.float()*m.float()).sum() if m is not None else norm.new_zeros(())
                cosine = dot / (norm*mnorm).clamp_min(1e-30)
                age = state.get('step', 0)
                age = (age.detach().to(device=norm.device,dtype=torch.float32)
                    if torch.is_tensor(age) else norm.new_tensor(float(age)))
                values = torch.stack((p.float().square().mean().sqrt(),
                    norm/math.sqrt(p.numel()), mnorm/math.sqrt(p.numel()),
                    v.float().mean().sqrt() if v is not None else norm.new_zeros(()),
                    cosine, (g!=0).float().mean(), age))
                before.append(p.detach().flatten().index_select(0,index).clone())
                statistics_before.append(values)
                identities.append((name,p,layer,family,index,float(group['lr'])))
                probes.append(g.detach().flatten().index_select(0,index).clone())
        result = self.original_step(*args, **kwargs)
        if not identities:
            return result
        with torch.no_grad():
            displacements = [p.detach().flatten().index_select(0,index)-old
                for (_,p,_,_,index,_),old in zip(identities,before)]
            all_stats = torch.stack(statistics_before).cpu().tolist()
            probe_values = torch.stack([d.square().mean().sqrt() for d in displacements]).cpu().tolist()
            nonzero = torch.stack([(d!=0).float().mean() for d in displacements]).cpu().tolist()
            probe_gradients = [g.cpu().tolist() for g in probes]
        rows = []
        for item, values, delta, fraction, gradient in zip(
                identities,all_stats,probe_values,nonzero,probe_gradients):
            name,p,layer,family,index,lr = item
            weight,grad,moment,second,alignment,support,age = values
            rows.append(dict(name=name,layer=layer,family=family,numel=p.numel(),
                weight_rms=weight,gradient_rms=grad,moment_rms=moment,
                second_moment_rms=second,gradient_old_moment_cosine=alignment,
                gradient_nonzero_fraction=support,adam_age_before=age,lr=lr,
                sampled_delta_rms=delta,sampled_delta_nonzero_fraction=fraction,
                sampled_relative_delta=delta/max(weight,1e-30)))
            self.gradient_probes.setdefault(name,[]).append(gradient)
        self.pending.append(rows)
        return result

    def finish_event(self, cycle, row):
        if not self.active:
            return
        traffic = []
        for record in self.counters:
            a,b,c = record['values'].cpu().tolist()
            traffic.append(dict(layer=record['layer'],input_rms=a,output_rms=b,
                residual_input_ratio=c/max(a,1e-30),
                output_gradient_rms=[float(v.cpu()) for v in record['backward']]))
        self.events.append(dict(cycle=cycle,updates=self.pending,traffic=traffic,
            choice=row.get('choice'),external_bytes=row.get('external_bytes'),
            outcome_error=row.get('outcome_error'),response_nll=row.get('response_nll')))
        self.pending, self.counters = [], []
        if len(self.events) >= self.budget:
            self.close()
        self.write()

    def close(self):
        if not self.active:
            return
        self.optimizer.step = self.original_step
        self.model.objective = self.original_objective
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.active = False

    def write(self):
        summaries = []
        all_rows = [r for event in self.events for update in event['updates'] for r in update]
        for layer in range(1,len(self.model.body.base.layers)+1):
            rows = [r for r in all_rows if r['layer']==layer]
            traffic = [r for event in self.events for r in event['traffic'] if r['layer']==layer]
            families = {}
            for name in sorted({r['family'] for r in rows}):
                selected = [r for r in rows if r['family']==name]
                key = selected[0]['name']
                gradients = self.gradient_probes[key]
                # Small deterministic coordinate probes; no full model gradient history.
                n = len(gradients)
                mean_square = statistics.fmean(v*v for g in gradients for v in g)
                mean_direction_square = statistics.fmean(
                    statistics.fmean(g[j] for g in gradients)**2 for j in range(len(gradients[0])))
                coherence = math.sqrt(mean_direction_square/max(mean_square,1e-60))
                fields = ('weight_rms','gradient_rms','moment_rms','second_moment_rms',
                    'gradient_old_moment_cosine','gradient_nonzero_fraction','adam_age_before',
                    'sampled_delta_rms','sampled_relative_delta','sampled_delta_nonzero_fraction','lr')
                families[name] = {field:_mean([r[field] for r in selected]) for field in fields}
                families[name].update(observed_updates=len(selected),
                    sampled_gradient_coherence=coherence,numel=selected[0]['numel'])
            summaries.append(dict(layer=layer,families=families,
                residual_input_ratio=_mean([r['residual_input_ratio'] for r in traffic]),
                output_gradient_rms=_mean([g for r in traffic for g in r['output_gradient_rms']])))
        report=dict(schema=1,status='observing' if self.active else 'complete',
            scope='actual original full-loop updates; no additional losses, updates or random draws',
            budget_events=self.budget,observed_events=len(self.events),
            first_cycle=self.events[0]['cycle'],last_cycle=self.events[-1]['cycle'],
            parameter_delta_scope='deterministic at most 128 coordinates per tensor; norms use full tensors',
            activation_scope='original observed-input objective; not speculative drafts or validation',
            layer_summary=summaries,events=self.events)
        self.output.parent.mkdir(parents=True,exist_ok=True)
        temporary=self.output.with_suffix(self.output.suffix+'.tmp')
        temporary.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        from scripts.ib_local.checkpoint_commit import publish_checkpoint
        publish_checkpoint(temporary,self.output)
