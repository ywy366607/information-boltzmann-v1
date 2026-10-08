"""One continuing COBA learner shared by training and active evaluation.

Local input/parameter sensitivities include reset and ALIF exactly conditional
on arriving recurrent pulses. Recurrent e-prop and low-rank direct feedback are
approximations, not the exact whole-connectome adjoint. Memory is constant in
stream duration. No autograd tape is retained.
"""
from __future__ import annotations

from dataclasses import asdict
import math
import torch
from . import triton_online_credit as fused
from .coba_local_credit import (LocalCobaEdgeCredit, compensated_add_,
    voltage_conductance_derivative, update_local_edges_pytorch)
from .eprop_credit_assignment import (
    EPropCreditAssignment, EPropEligibilityState, DopamineReceptorState,
    SensoryWriterEligibilityState,
)


def rmsnorm_vjp(z, error, norm):
    eps = norm.eps if norm.eps is not None else torch.finfo(z.dtype).eps
    inv = torch.rsqrt(z.square().mean(-1, keepdim=True) + eps)
    zhat = z * inv
    weighted = error * norm.weight
    grad = inv * (weighted - zhat * (weighted * zhat).mean(-1, keepdim=True))
    return grad, (error * zhat).sum(0)


def advance_local_jacobian(eh, eb, direct_v, direct_threshold, direct_b,
                           alpha, voltage, spikes, psi, beta, rho):
    """Local forward-mode derivative; trailing dimensions enumerate parameters."""
    ev = alpha[:, None] * eh + direct_v
    ds = psi[:, None] * (ev - beta[:, None] * eb - direct_threshold)
    eh.copy_((1 - spikes)[:, None] * ev - voltage[:, None] * ds)
    eb.mul_(rho[:, None]).add_((1 - rho)[:, None] * ds + direct_b)


class FlyOnlineLearner:
    def __init__(self, model, *, lr=3e-4, lr_synapse=1e-5, lr_sensory=1e-4,
                 grad_accum_tokens=4, synapse_update_interval=2,
                 train_decoder=True, train_synapses=True, train_sensory=True,
                 dopamine_k_on=.2, dopamine_k_off=.05, dopamine_q0=.05):
        self.model = model
        model.requires_grad_(False)
        self.device = model.output_read.weight.device
        self.d_model = model.embedding.embedding_dim
        self.accumulation = int(grad_accum_tokens)
        self.synapse_interval = int(synapse_update_interval)
        if min(self.accumulation, self.synapse_interval) < 1:
            raise ValueError("Update intervals must be positive")
        self.lr_synapse, self.lr_sensory = lr_synapse, lr_sensory
        self.train_synapses, self.train_sensory = train_synapses, train_sensory
        self.parameters = {'output_read.weight': model.output_read.weight,
                           'read_norm.weight': model.read_norm.weight}
        if train_decoder:
            self.parameters['decoder.weight'] = model.decoder.weight
        if model.decoder.bias is not None:
            self.parameters['decoder.bias'] = model.decoder.bias
        self.physical_names = ('log_threshold', 'log_tau_m', 'log_beta', 'log_tau_a',
                               'log_tau_s_e', 'log_tau_s_i', 'log_g_e', 'log_g_i')
        for name in self.physical_names:
            self.parameters[name] = getattr(model, name)
        writer = model.topographic_writer
        if train_sensory:
            self.parameters['topographic_writer.gate_linear.weight'] = writer.gate_linear.weight
            self.parameters['topographic_writer.gate_linear.bias'] = writer.gate_linear.bias
        decayed, undecayed = [], []
        for name, param in self.parameters.items():
            (decayed if name in ('output_read.weight', 'decoder.weight') else undecayed).append(param)
        self.optimizer = torch.optim.AdamW([
            {'params': decayed, 'weight_decay': 1e-4},
            {'params': undecayed, 'weight_decay': 0.0}], lr=lr, fused=self.device.type=='cuda')
        self.grads = {name: torch.zeros_like(p) for name, p in self.parameters.items()}
        shape = (1, model.n_neurons)
        self.h = torch.zeros(shape, device=self.device)
        self.ring = tuple(torch.zeros_like(self.h) for _ in range(4))
        self.syn = {name: torch.zeros_like(self.h) for name in ('ge', 'gi', 'b')}
        self.syn['x'] = torch.ones_like(self.h)
        self.syn['u'] = model.get_stp_params()[0].clone().expand_as(self.h).contiguous()
        self.eprop = EPropCreditAssignment(model.n_neurons, self.d_model,
                                          model.embedding.num_embeddings,
                                          E_E=model.E_E, E_I=model.E_I).to(self.device)
        self.eligibility = EPropEligibilityState.init_zero(1, model.n_neurons, self.device)
        self.dopamine = DopamineReceptorState.init_zero(
            model.n_neurons, self.device, dopamine_k_on, dopamine_k_off, dopamine_q0)
        self.sensory = SensoryWriterEligibilityState.init_zero(
            self.d_model, self.device, writer.n_vis, writer.n_chemo, writer.n_mech) if train_sensory else None
        # Gates have their own historical derivative, including writer baseline.
        # [sensory neuron, 3*(d+1)] covers gate weights and bias.
        if train_sensory:
            self.modality = torch.cat([torch.full((count,), j, dtype=torch.long, device=self.device)
                                       for j, count in enumerate((writer.n_vis, writer.n_chemo, writer.n_mech))])
            gs = (writer.n_total, 3 * (self.d_model + 1))
            self.gate_h = torch.zeros(gs, device=self.device)
            self.gate_b = torch.zeros_like(self.gate_h)
            self.gate_baseline = torch.zeros_like(self.gate_h)
        else:
            self.gate_h = self.gate_b = self.gate_baseline = None
        self.phys_h = torch.zeros(model.n_neurons, 8, device=self.device)
        self.phys_b = torch.zeros_like(self.phys_h)
        self.phys_grad = torch.zeros_like(self.phys_h)
        self.ge_derivative = torch.zeros_like(self.h)
        self.gi_derivative = torch.zeros_like(self.h)
        self.pending_edges = [torch.zeros_like(model.edge_weight_e),
                              torch.zeros_like(model.edge_weight_i)] if train_synapses else []
        self.edge_credit = [LocalCobaEdgeCredit.zeros_like(weight)
                            for weight in (model.edge_weight_e, model.edge_weight_i)] if train_synapses else []
        self.projection_residuals = [torch.zeros_like(projection.weight)
            for projection in (writer.proj_vis, writer.proj_chemo, writer.proj_mech)] if train_sensory else []
        self.credit_migration = None
        self.decoder_errors = torch.zeros(self.accumulation, model.embedding.num_embeddings, device=self.device)
        self.decoder_latents = torch.zeros(self.accumulation, self.d_model, device=self.device)
        self.events = self.updates = self.synapse_updates = self.pending = self.edge_pending = 0
        self.ema = self.sum_loss = 0.0
        self.previous_token = None
        self.latent_window = torch.zeros(128,self.d_model,device=self.device)
        self.energy_totals = torch.zeros(5,device=self.device)
        self.local_log_gain = torch.zeros((),device=self.device)
        self.refresh()

    def refresh(self):
        m = self.model
        self.rates = m.get_decay_rates()
        self.threshold = m.get_thresholds()
        self.alif = m.get_alif_params()
        self.stp = m.get_stp_params()
        self.gains = m.get_conductance_gains()
        limits=((.01,2),(1,250),(1e-4,2),(2,500),(.5,100),(.5,100),(.01,100),(.01,100))
        masks=[]
        for name,(lower,upper) in zip(self.physical_names,limits):
            value=getattr(m,name).exp()
            active=((value>lower)&(value<upper)).to(value.dtype)
            masks.append(active.expand(m.n_neurons) if active.ndim==0 else active[m.superclass_id])
        self.physical_masks=torch.stack(masks,-1)

    @torch.no_grad()
    def step(self, input_token, target):
        m, writer = self.model, self.model.topographic_writer
        h_prev, b_prev = self.h, self.syn['b']
        ge_prev, gi_prev = self.syn['ge'], self.syn['gi']
        res, bio = m.step(self.h, input_token, spike_ring=self.ring, **self.syn,
                          base_rates=self.rates, thresholds=self.threshold,
                          conductance_gains=self.gains, alif_params=self.alif,
                          stp_params=self.stp, return_biophysics=True)
        self.h, spikes, self.ring = res[:3]
        self.syn = dict(zip(('ge', 'gi', 'b', 'x', 'u'), res[3:]))
        read_h = self.h[:, m.read_indices]
        z = m.output_read(read_h)
        latent = m.read_norm(z)
        logits = m.decoder(latent)
        logp = torch.log_softmax(logits, -1)
        loss = -logp[0, target.reshape(-1)[0]]
        error_logits = logp.exp()
        error_logits[0, target.reshape(-1)[0]] -= 1
        error_latent = error_logits @ m.decoder.weight
        error_read, gain_grad = rmsnorm_vjp(z, error_latent, m.read_norm)
        self.grads['read_norm.weight'].add_(gain_grad)
        self.grads['output_read.weight'].addmm_(error_read.T, read_h)
        if 'decoder.weight' in self.grads:
            self.decoder_errors[self.pending].copy_(error_logits[0])
            self.decoder_latents[self.pending].copy_(latent[0])
        if 'decoder.bias' in self.grads:
            self.grads['decoder.bias'].add_(error_logits[0])
        L, _, _ = self.eprop.compute_learning_signal_whole_brain(
            error_read=error_read, w_read=m.output_read.weight,
            read_indices=m.read_indices, read_mask=m.read_mask)
        q = self.dopamine.update_from_dan_activity(
            spikes, self.h, dan_edge_pre=getattr(m,"dan_edge_pre",None), dan_edge_post=getattr(m,"dan_edge_post",None),
            dan_edge_weight=getattr(m,"dan_edge_weight",None), dan_scale=getattr(m,"dan_scale",1.0),
            delayed_pulses=bio['delayed_pulses'], delay_splits=getattr(m,"dan_delay_splits",None))
        v, a, integ = (bio[name] for name in ('v_pre', 'alpha_eff', 'beta_int'))
        psi = 1 / (1 + (math.pi * (v - bio['eff_threshold'])).square())
        rho, beta_a = self.alif
        syn_term = integ*(self.gains[0]*self.syn['ge']*m.E_E+self.gains[1]*self.syn['gi']*m.E_I)
        old_term = a*h_prev
        input_term = v-old_term-syn_term
        passive = ((1-a.square())*h_prev.square()).sum()
        recurrent_work = (2*old_term*syn_term+syn_term.square()).sum()
        input_work = (2*(old_term+syn_term)*input_term+input_term.square()).sum()
        reset_loss = (v.square()*spikes).sum()
        change = (self.h.square()-h_prev.square()).sum()
        self.energy_totals.add_(torch.stack((input_work,recurrent_work,passive,reset_loss,
                                           change-input_work-recurrent_work+passive+reset_loss)))
        self.latent_window[self.events%128].copy_(latent[0])
        jhh=((1-spikes)-v*psi)*a; jhb=v*psi*beta_a
        jbh=(1-rho)*psi*a; jbb=rho-(1-rho)*psi*beta_a
        trace=jhh.square()+jhb.square()+jbh.square()+jbb.square()
        det=(jhh*jbb-jhb*jbh).square()
        norm_sq=.5*(trace+torch.sqrt((trace.square()-4*det).clamp_min(0)))
        self.local_log_gain.add_(.5*norm_sq.clamp_min(1e-30).log().mean())
        self._physical_gradients(L, bio, h_prev, b_prev, ge_prev, gi_prev, spikes, psi)
        if self.sensory is not None:
            idx = writer.injection_index
            if fused.supported(self.h):
                fused.update_projection_traces(self.sensory,writer.last_gates[0],self.modality,
                    writer.last_token_emb[0],a[0,idx],integ[0,idx],v[0,idx],spikes[0,idx],
                    psi[0,idx],beta_a[0,idx],rho[0,idx],writer.lambda_adapt)
            else:
                self.sensory.update_input_eligibility(
                    writer.last_token_emb, writer.last_gates, a[0, idx], integ[0, idx],
                    lambda_adapt=writer.lambda_adapt, v_pre=v[0, idx], spikes=spikes[0, idx],
                    psi=psi[0, idx], beta_adaptation=beta_a[0, idx], rho_adaptation=rho[0, idx])
            # This eligibility already differentiates voltage, reset and ALIF;
            # no excitatory driving-force multiplier belongs on an input current.
            if fused.supported(self.h):
                offset=0
                factor=L[0,idx]*q[idx]
                for projection,residual in zip((writer.proj_vis,writer.proj_chemo,writer.proj_mech),self.projection_residuals):
                    count=projection.weight.shape[0]
                    fused.update_projection_weights(projection.weight,self.sensory.e_proj[offset:offset+count],
                                                    factor[offset:offset+count],self.lr_sensory,1e-4,residual)
                    offset+=count
            else:
                offset=0
                factor=L[0,idx]*q[idx]
                for projection,residual in zip((writer.proj_vis,writer.proj_chemo,writer.proj_mech),self.projection_residuals):
                    count=projection.weight.shape[0]
                    gradient=factor[offset:offset+count,None]*self.sensory.e_proj[offset:offset+count]
                    compensated_add_(projection.weight,residual,-self.lr_sensory*(gradient+1e-4*projection.weight))
                    offset+=count
            self._gate_gradients(L[0, idx] * q[idx], a[0, idx], integ[0, idx],
                                 v[0, idx], spikes[0, idx], psi[0, idx], beta_a[0, idx], rho[0, idx])
        if self.train_synapses:
            # Exact local derivative conditional on actual arriving delayed
            # pulses. Destination-specific conductance/voltage/ALIF history
            # cannot be replaced by a single presynaptic filtered trace.
            dv_dG=voltage_conductance_derivative(bio,h_prev)
            signal=(L*q[None]).contiguous()
            pulses=torch.stack(bio['delayed_pulses']).contiguous()
            apply = self.edge_pending + 1 == self.synapse_interval
            for state,weight,pending,gain,reversal,decay,pre,post,splits in zip(
                    self.edge_credit,(m.edge_weight_e,m.edge_weight_i),self.pending_edges,self.gains,
                    (m.E_E,m.E_I),self.rates[1:3],
                    (m.edge_pre_e,m.edge_pre_i),(m.edge_post_e,m.edge_post_i),(m.splits_e,m.splits_i)):
                direct=(gain*(dv_dG+integ*reversal)).contiguous()
                if fused.supported(weight):
                    if weight.numel():
                        fused.update_local_coba_edges(state,weight,pending,pre,post,pulses,signal,bio,direct,
                            decay,spikes,psi,beta_a,rho,splits,self.lr_synapse,apply)
                else:
                    update_local_edges_pytorch(state,weight,pending,pre,post,splits,bio['delayed_pulses'],
                        signal,bio,direct,decay,spikes,psi,beta_a,rho,self.lr_synapse,apply)
            self.edge_pending = 0 if apply else self.edge_pending+1
            if apply: self.synapse_updates += 1
        self.pending += 1
        if self.pending == self.accumulation:
            if 'decoder.weight' in self.grads:
                self.grads['decoder.weight'].addmm_(self.decoder_errors.T, self.decoder_latents)
            for name, param in self.parameters.items():
                param.grad = self.grads[name] / self.pending
            torch.nn.utils.clip_grad_norm_(list(self.parameters.values()), 1.0, error_if_nonfinite=True)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            for grad in self.grads.values(): grad.zero_()
            self.pending = 0
            self.updates += 1
            self.refresh()
        score = float(loss.item())
        if not math.isfinite(score): raise FloatingPointError('Non-finite prequential NLL')
        self.events += 1
        self.sum_loss += score
        self.ema = score if self.events == 1 else .99*self.ema + .01*score
        return score

    def _physical_gradients(self, L, bio, h_prev, b_prev, ge_prev, gi_prev, spikes, psi):
        m = self.model
        v, a, integ, G = (bio[n] for n in ('v_pre', 'alpha_eff', 'beta_int', 'g_total'))
        rho, beta = self.alif
        if fused.supported(self.h):
            fused.update_physical_traces(self.phys_h,self.phys_b,self.ge_derivative,self.gi_derivative,self.phys_grad,
                L,bio,h_prev,b_prev,ge_prev,gi_prev,self.syn['ge'],self.syn['gi'],spikes,psi,rho,beta,
                self.rates,self.threshold,self.physical_masks,self.gains,m.E_E,m.E_I)
            for index,name in enumerate(self.physical_names):
                grad=self.phys_grad[:,index]
                if self.parameters[name].numel()==1: self.grads[name].add_(grad.sum().reshape_as(self.grads[name]))
                else: self.grads[name].index_add_(0,m.superclass_id,grad)
            return
        # Derivative of beta_int=(1-exp(-G))/G, including existing clamps.
        exp_derivative = -a * ((G >= 1e-5) & (G <= 20)).to(G.dtype)
        beta_derivative = (-exp_derivative*G.clamp_min(1e-5)
                           -(1-a)*(G >= 1e-5)) / G.clamp_min(1e-5).square()
        force = (v-a*h_prev) / integ
        dv_dG = exp_derivative*h_prev + beta_derivative*force
        def expanded_derivative(name, lower, upper):
            value = getattr(m, name).exp()
            active = ((value > lower) & (value < upper)).to(value.dtype)
            return active if value.ndim == 0 else active[m.superclass_id][None]
        dt_m = (-torch.log(self.rates[0])) * expanded_derivative('log_tau_m',1,250)
        d_v = torch.zeros_like(self.phys_h)
        d_th = torch.zeros_like(d_v)
        d_b = torch.zeros_like(d_v)
        d_v[:,1] = (-dt_m*dv_dG)[0]
        theta = self.threshold*expanded_derivative('log_threshold',.01,2)
        d_th[:,0] = theta[0]
        d_th[:,2] = (beta*b_prev*expanded_derivative('log_beta',1e-4,2))[0]
        d_b[:,3] = (rho*(-rho.log())*(b_prev-spikes)*expanded_derivative('log_tau_a',2,500))[0]
        for index, old, new, derivative, leak, gain, reversal, name in (
                (4,ge_prev,self.syn['ge'],self.ge_derivative,self.rates[1],self.gains[0],m.E_E,'log_tau_s_e'),
                (5,gi_prev,self.syn['gi'],self.gi_derivative,self.rates[2],self.gains[1],m.E_I,'log_tau_s_i')):
            arrival = (new-leak*old)/(1-leak)
            derivative.mul_(leak).add_((old-arrival)*leak*(-leak.log())*expanded_derivative(name,.5,100))
            d_v[:,index] = (gain*derivative*(dv_dG+integ*reversal))[0]
        d_v[:,6] = (self.gains[0]*self.syn['ge']*(dv_dG+integ*m.E_E)*expanded_derivative('log_g_e',.01,100))[0]
        d_v[:,7] = (self.gains[1]*self.syn['gi']*(dv_dG+integ*m.E_I)*expanded_derivative('log_g_i',.01,100))[0]
        advance_local_jacobian(self.phys_h,self.phys_b,d_v,d_th,d_b,a[0],v[0],spikes[0],psi[0],beta[0],rho[0])
        for index,name in enumerate(self.physical_names):
            grad = L[0]*self.phys_h[:,index]
            if self.parameters[name].numel() == 1:
                self.grads[name].add_(grad.sum().reshape_as(self.grads[name]))
            else:
                self.grads[name].index_add_(0,m.superclass_id,grad)

    def _gate_gradients(self, learning_signal, a, integ, v, spikes, psi, beta, rho):
        w = self.model.topographic_writer
        g = w.last_gates[0]
        p = g/3
        jac = torch.diag(g)-g[:,None]*p[None,:]
        token = torch.cat((w.last_token_emb[0], torch.ones(1,device=self.device)))
        packet = torch.cat((w.last_p_vis[0],w.last_p_chemo[0],w.last_p_mech[0]))
        modality = self.modality
        if fused.supported(self.h):
            fused.update_gate_traces(self.gate_h,self.gate_b,self.gate_baseline,packet,jac,modality,
                                    token[:-1],a,integ,v,spikes,psi,beta,rho,w.lambda_adapt)
        else:
            raw = ((packet[:,None]*jac[modality])[:,:,None]*token[None,None,:]).flatten(1)
            direct = integ[:,None]*(raw-self.gate_baseline)
            self.gate_baseline.mul_(w.lambda_adapt).add_(raw,alpha=1-w.lambda_adapt)
            advance_local_jacobian(self.gate_h,self.gate_b,direct,0,0,a,v,spikes,psi,beta,rho)
        grad = learning_signal @ self.gate_h
        grad = grad.reshape(3,-1)
        self.grads['topographic_writer.gate_linear.weight'].add_(grad[:,:-1])
        self.grads['topographic_writer.gate_linear.bias'].add_(grad[:,-1])

    def state_dict(self):
        return {'h':self.h,'ring':self.ring,'syn':self.syn,
                'latent_window':self.latent_window,'energy_totals':self.energy_totals,'local_log_gain':self.local_log_gain,
                'eligibility':asdict(self.eligibility),'dopamine':asdict(self.dopamine),
                'sensory':asdict(self.sensory) if self.sensory else None,
                'gate_h':self.gate_h,'gate_b':self.gate_b,'gate_baseline':self.gate_baseline,
                'phys_h':self.phys_h,'phys_b':self.phys_b,'ge_derivative':self.ge_derivative,
                'gi_derivative':self.gi_derivative,'pending_edges':self.pending_edges,
                'edge_credit':[asdict(state) for state in self.edge_credit],
                'projection_residuals':self.projection_residuals,
                'credit_revision':'destination-coba-compensated-v1','credit_migration':self.credit_migration,
                'writer_baseline':self.model.topographic_writer.a_adapt,
                'grads':self.grads,'decoder_errors':self.decoder_errors,'decoder_latents':self.decoder_latents,
                'optimizer':self.optimizer.state_dict(),'feedback':{name:value for name,value in self.eprop.named_buffers() if value is not None},
                'counters':{key:getattr(self,key) for key in ('events','updates','synapse_updates','pending',
                                                             'edge_pending','ema','sum_loss','previous_token')}}

    def load_state_dict(self, state):
        for key in ('latent_window','energy_totals','local_log_gain','h','phys_h','phys_b','ge_derivative','gi_derivative','decoder_errors','decoder_latents'):
            getattr(self,key).copy_(state[key])
        for dst,src in zip(self.ring,state['ring']): dst.copy_(src)
        for key in self.syn: self.syn[key].copy_(state['syn'][key])
        for name in ('eligibility','dopamine','sensory'):
            obj = getattr(self,name)
            if obj is not None:
                for key,value in state[name].items():
                    if isinstance(value,torch.Tensor): getattr(obj,key).copy_(value)
        for name in ('gate_h','gate_b','gate_baseline'):
            if getattr(self,name) is not None: getattr(self,name).copy_(state[name])
        for dst,src in zip(self.pending_edges,state['pending_edges']): dst.copy_(src)
        if 'edge_credit' in state:
            for obj,source in zip(self.edge_credit,state['edge_credit']):
                for name,value in source.items(): getattr(obj,name).copy_(value)
            for dst,src in zip(self.projection_residuals,state['projection_residuals']): dst.copy_(src)
            self.credit_migration=state.get('credit_migration')
        else:
            # Existing physical/optimizer/pending state is preserved. The old
            # factorized traces cannot reconstruct the newly defined per-edge
            # derivatives; these additional variables start at this revision.
            self.credit_migration={'from':'factorized-source-trace',
                'at_existing_events':state['counters']['events'],
                'new_variables':'per-edge conditional derivatives and rounding residuals initialized to zero',
                'preserved':'physical state, weights, existing pending updates, optimizer, other local traces and feedback'}
        self.model.topographic_writer.a_adapt.copy_(state['writer_baseline'])
        for name,value in state['grads'].items(): self.grads[name].copy_(value)
        self.optimizer.load_state_dict(state['optimizer'])
        for group in self.optimizer.param_groups:
            group['fused'] = self.device.type == 'cuda'
        # Older unfused checkpoints keep Adam's counter on CPU. Fused AdamW
        # requires the counter beside its moments; preserve the saved value.
        if self.device.type == 'cuda':
            for parameter, adam_state in self.optimizer.state.items():
                if isinstance(adam_state.get('step'), torch.Tensor):
                    adam_state['step'] = adam_state['step'].to(parameter.device)
        for name,value in state['feedback'].items(): getattr(self.eprop,name).copy_(value)
        for key,value in state['counters'].items(): setattr(self,key,value)
        self.refresh()
