"""Critical-dissipation Information Boltzmann field on the MaleCNS graph."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from scripts.ib_local.cbim_malecns import load_malecns_graph
from scripts.ib_local.cbim_malecns_v3 import (
    ContinuousGraphTransport, FullStateKineticCollision, GraphStateReadout)


class CriticalMaxwellBoundary(nn.Module):
    """Bounded local accommodation; one operator writes and replaces."""

    def __init__(self, graph, vocab_size=50257, d=64, packet_radius=1.25,
                 max_rate=.25):
        super().__init__()
        nf = graph["node_features"].float()
        self.packet_radius, self.max_rate = float(packet_radius), float(max_rate)
        self.embedding = nn.Embedding(vocab_size, d)
        self.address, self.width = nn.Linear(d, 3), nn.Linear(d, 3)
        self.token, self.state = nn.Linear(d, d), nn.Linear(d, d)
        self.neighbor, self.position = nn.Linear(d, d), nn.Linear(nf.shape[-1], d)
        self.proposal = nn.Sequential(
            nn.Linear(d, 2*d), nn.SiLU(), nn.Linear(2*d, d))
        self.rate = nn.Linear(d, 1)
        nn.init.normal_(self.proposal[-1].weight, std=1e-3)
        nn.init.zeros_(self.proposal[-1].bias)
        nn.init.normal_(self.rate.weight, std=1e-3); nn.init.zeros_(self.rate.bias)
        self.register_buffer("coordinates", graph["coordinates"].float()[None], persistent=False)
        self.register_buffer("node_features", nf[None], persistent=False)
        self.register_buffer("neighbor_indices", graph["neighbor_indices"].long(), persistent=False)
        self.register_buffer("neighbor_weights", graph["neighbor_weights"].float(), persistent=False)

    def forward(self, field, token_ids, conductance):
        emb = self.embedding(token_ids)
        center = torch.sigmoid(self.address(emb))[:, None]
        width = (.04 + .21*torch.sigmoid(self.width(emb)))[:, None]
        envelope = torch.exp(-.5*((self.coordinates-center)/width).square().sum(-1))
        neighbors = field[:, self.neighbor_indices]
        neighborhood = (neighbors*self.neighbor_weights[None,...,None]).sum(2)
        context = F.silu(self.token(emb)[:,None] + self.state(field)
                         + self.neighbor(neighborhood) + self.position(self.node_features))
        proposal = self.packet_radius*F.normalize(self.proposal(context), dim=-1)
        eta = self.max_rate*envelope[...,None]*torch.sigmoid(
            self.rate(context) + conductance[...,None])
        output = (1-eta)*field + eta*proposal
        work = .5*(output.square()-field.square()).sum(-1).mean()
        return output, {"boundary_work": work.detach(),
                        "accommodation_mean": eta.detach().mean(),
                        "accommodation_max": eta.detach().amax()}


class CBIMMaleCNSCritical(nn.Module):
    architecture = "CBIM-MaleCNS-critical-Maxwell-v1"

    def __init__(self, graph_path, vocab_size=50257, velocities=8,
                 content_dim=8, queries=4, heads=4, checkpoint_tokens=8,
                 probe_epsilon=1e-3, controller_ema=.01, controller_lr=.01):
        super().__init__(); graph = load_malecns_graph(graph_path)
        self.graph_path, self.velocities, self.content_dim = str(graph_path), velocities, content_dim
        self.d, self.L = velocities*content_dim, graph["coordinates"].shape[0]
        self.state_shape = (self.L, 2*self.d+2)
        self.checkpoint_tokens, self.probe_epsilon = int(checkpoint_tokens), float(probe_epsilon)
        self.controller_ema, self.controller_lr = float(controller_ema), float(controller_lr)
        self.boundary = CriticalMaxwellBoundary(graph, vocab_size, self.d)
        self.collision = FullStateKineticCollision(graph["node_features"], velocities, content_dim)
        self.transport = ContinuousGraphTransport(
            graph["laplacian_basis"], graph["laplacian_eigenvalues"], velocities, content_dim)
        self.readout = GraphStateReadout(graph["node_features"], self.d, queries, heads)
        self.decoder = nn.Linear(self.d, vocab_size); self.decoder.weight = self.boundary.embedding.weight
        probe = torch.sin(torch.arange(self.L*self.d, dtype=torch.float32)*1.618).reshape(self.L,self.d)
        probe = probe/probe.norm().clamp_min(1e-12)
        self.register_buffer("initial_probe", probe, persistent=False)

    def initial_state(self, batch_size, device=None, dtype=None):
        p=next(self.parameters()); device=p.device if device is None else device; dtype=p.dtype if dtype is None else dtype
        field=torch.zeros(batch_size,self.L,self.d,device=device,dtype=dtype)
        delta=self.initial_probe.to(device=device,dtype=dtype)[None].expand(batch_size,-1,-1)
        zeros=torch.zeros(batch_size,self.L,1,device=device,dtype=dtype)
        return torch.cat((field,delta,zeros,zeros),-1)

    def unpack(self, state):
        return state[...,:self.d], state[...,self.d:2*self.d], state[...,2*self.d], state[...,2*self.d+1]

    def kinetic(self, field, token_ids, conductance, disable_collision=False, disable_transport=False):
        field,boundary=self.boundary(field,token_ids,conductance)
        if disable_collision: collision={"collision_angle_abs_mean":field.new_zeros(())}
        else: field,collision=self.collision(field)
        if disable_transport: transport={"transport_angle_abs_mean":field.new_zeros(())}
        else: field,transport=self.transport(field)
        return field,boundary,collision,transport

    def evolve(self, state, token_ids, disable_collision=False, disable_transport=False):
        field,delta,conductance,lambda_ema=self.unpack(state)
        output,boundary,collision,transport=self.kinetic(
            field,token_ids,conductance,disable_collision,disable_transport)
        with torch.no_grad():
            perturbed=field.detach()+self.probe_epsilon*delta
            shadow,_,_,_=self.kinetic(
                perturbed,token_ids,conductance.detach(),disable_collision,disable_transport)
            tangent=(shadow-output.detach())/self.probe_epsilon
            before_norm=delta.flatten(1).norm(dim=-1).clamp_min(1e-7)
            after_square=tangent.square().sum((-1,-2)).clamp_min(1e-14)
            global_lambda=torch.log(after_square.sqrt()/before_norm).clamp(-1.,1.)
            # Spatial transport redistributes the probe but cannot change its
            # global norm.  Attribute genuine global growth to the nodes where
            # the transported perturbation arrives, rather than interpreting
            # every local arrival as newly created instability.
            local_weight=(self.L*tangent.square().sum(-1)
                          / after_square[:,None]).clamp_max(self.L)
            local_signal=global_lambda[:,None]*local_weight
            lambda_ema=(1-self.controller_ema)*lambda_ema+self.controller_ema*local_signal
            conductance=torch.relu(conductance+self.controller_lr*lambda_ema)
            delta=tangent/after_square.sqrt()[:,None,None]
        packed=torch.cat((output,delta,conductance[...,None],lambda_ema[...,None]),-1)
        return packed,{**boundary,
                       "field_energy":(.5*output.square().sum(-1).mean()).detach(),
                       "conductance_mean":conductance.mean(),"conductance_max":conductance.max(),
                       "lambda_mean":lambda_ema.mean(),"lambda_max":lambda_ema.max(),
                       "global_lambda":global_lambda.mean(),
                       "collision_angle_abs_mean":collision["collision_angle_abs_mean"],
                       "transport_angle_abs_mean":transport["transport_angle_abs_mean"]}

    def forward(self,input_ids,targets,state=None,disable_collision=False,disable_transport=False):
        batch,tokens=input_ids.shape
        state=self.initial_state(batch,input_ids.device) if state is None else state
        position=self.readout.position_encoding(); features=[]; sums=[]
        def segment(s,ids,pos):
            fs=[]; ds=[]
            for i in range(ids.shape[1]):
                s,d=self.evolve(s,ids[:,i],disable_collision,disable_transport)
                field,_,_,_=self.unpack(s); fs.append(self.readout(field,pos))
                ds.append(torch.stack((d["field_energy"],d["boundary_work"],d["accommodation_mean"],
                    d["accommodation_max"],d["conductance_mean"],d["conductance_max"],
                    d["lambda_mean"],d["lambda_max"],d["collision_angle_abs_mean"],
                    d["transport_angle_abs_mean"],d["global_lambda"])))
            return s,torch.stack(fs,1),torch.stack(ds).sum(0)
        for start in range(0,tokens,self.checkpoint_tokens):
            ids=input_ids[:,start:start+self.checkpoint_tokens]
            if self.training and torch.is_grad_enabled():
                state,fs,ds=checkpoint(segment,state,ids,position,use_reentrant=False,preserve_rng_state=False)
            else: state,fs,ds=segment(state,ids,position)
            features.append(fs);sums.append(ds)
        logits=self.decoder(torch.cat(features,1));loss=F.cross_entropy(logits.flatten(0,1),targets.flatten())
        d=torch.stack(sums).sum(0)/tokens; field,_,conductance,lam=self.unpack(state)
        names=("energy","boundary_work","accommodation_mean","accommodation_max","conductance_mean",
               "conductance_max","lambda_mean","lambda_max","collision_angle_abs_mean",
               "transport_angle_abs_mean","global_lambda")
        diagnostics={name:d[i] for i,name in enumerate(names)}
        diagnostics["final_energy"]=.5*field.square().sum(-1).mean().detach()
        diagnostics["node_conductance"]=conductance.detach();diagnostics["node_lambda"]=lam.detach()
        return loss,state,diagnostics
