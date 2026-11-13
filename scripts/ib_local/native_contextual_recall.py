"""A current query reads complete old transitions in native value coordinates.

Candidate delivery program, not an additional network: Q/K/V, short filters,
output gate/norm/projection are the existing GDN layer's parameters. Source
position j addresses the observed successor activity j+1. Historical programs
are reexecuted with held inputs/boundaries, never reference-answer labels.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from scripts.ib_local.pretrained_gdn_experience import LocalExperience, copy_tree


def conditional_head_read(query, keys, values, prior, temperature, *, return_weights=False,topk=None):
    """Batched bounded cross-attention; zero prior gives exactly zero read."""
    if query.ndim!=3 or keys.ndim!=3 or values.ndim!=3:
        raise ValueError('query[N,H,K], key[S,H,K], value[S,H,V] required')
    if prior.shape!=(query.shape[0],keys.shape[0]) or temperature<=0:
        raise ValueError('one finite token prior per query/key required')
    q=F.normalize(query.float(),dim=-1)
    k=F.normalize(keys.float(),dim=-1)
    score=torch.einsum('nhk,shk->nhs',q,k)/temperature
    # Normalized cosines keep the existing temperature's exponent bounded.
    weight=score.exp()*prior[:,None].float()
    if topk is not None:
        if type(topk) is not int or topk<1:
            raise ValueError('positive finite read fan-in required')
        ranking=score+prior[:,None].float().clamp_min(1e-30).log()
        selected=ranking.topk(min(topk,keys.shape[0]),dim=-1).indices
        mask=torch.zeros_like(weight,dtype=torch.bool).scatter_(-1,selected,True)
        weight=weight*mask
    weight=weight/weight.sum(-1,keepdim=True).clamp_min(1e-30)
    result=torch.einsum('nhs,shv->nhv',weight,values.float())
    return (result,weight) if return_weights else result


def native_successor_residual(wrapper,block,query,replay,*,query_boundary=None,
                              forced_slot=None,forced_value_position=None,frozen_cache=None):
    """Full-width residual for the existing pre-block read port.

    `forced_slot` is exclusively a researcher intervention. `frozen_cache` is
    valid only inside one no-update, unchanged-table diagnostic/decode scope.
    This candidate is not enabled in the ongoing individual by importing it.
    """
    if frozen_cache is not None and torch.is_grad_enabled():
        raise RuntimeError('a frozen read cache cannot carry learning graphs')
    if torch.is_grad_enabled() and wrapper.detach_replay_older_than is not None:
        raise RuntimeError('age-selective native successor credit is not admitted yet')
    attention=block.attn
    if attention.num_heads!=attention.num_v_heads:
        raise ValueError('this candidate currently requires the source GDN shared head count')
    read=wrapper.read(query)
    if forced_slot is not None:
        from dataclasses import replace
        read=replace(read,slots=torch.full((len(query),1),forced_slot,dtype=torch.long),
                     weights=query.new_ones(len(query),1).float().cpu())
    wrapper.last_read=read
    normalized=block.attn_norm(query[None])
    # Raw read-query inputs and memory-injected body inputs are different
    # streams. Keep their fixed-width delay buffers distinct in the same cache;
    # aliasing the body's Q prefix would violate packed/sequential equivalence.
    query_boundary={} if query_boundary is None else query_boundary
    projected,prefix=attention.q_conv1d(attention.q_proj(normalized),
        cache=query_boundary.get('recall_query_conv_state'),output_final_state=True)
    query_boundary['recall_query_conv_state']=prefix
    q=projected[0].reshape(len(query),attention.num_heads,attention.head_k_dim)
    keys,values,priors,token_slots,token_positions=[],[],[],[],[]
    for slot in dict.fromkeys(read.slots.flatten().tolist()):
        if not bool(wrapper.memory.occupied[slot]):
            continue
        capsule=wrapper.capsules[slot]
        if capsule is None or capsule.version!=int(wrapper.memory.read_credit_version[slot]):
            raise RuntimeError('old program identity changed')
        identity=id(capsule)
        packed=None if frozen_cache is None else frozen_cache.get(identity)
        if packed is None:
            complete=capsule.complete_inputs
            if not 65<=complete.shape[1]<=128:
                raise ValueError('bounded observed local programme required')
            response=replay(LocalExperience(complete,capsule.boundary,capsule.version))
            if wrapper.detach_replay:
                response=response.detach()
            wrapper.replayed.append(response)
            wrapper.replayed_by_slot[slot]=response[:, -1]
            old_norm=block.attn_norm(complete.to(query.device))
            old_prefix=(None if capsule.boundary is None else
                        copy_tree(capsule.boundary['conv_state'][1],query.device))
            key,_=attention.k_conv1d(attention.k_proj(old_norm),cache=old_prefix,
                                    output_final_state=False)
            key=key[0].reshape(-1,attention.num_heads,attention.head_k_dim)
            # This is a read-time decoder. The old F response retains the native
            # write/erase/forget derivatives; V itself is reused at read time.
            # Spatial pre/post pairing at the SAME event. Pretrained GDN
            # writes K_t/V_t together; a forced one-token lag is not native.
            value=F.silu(attention.v_proj(block.attn_norm(response)))
            value=value[0].reshape(-1,attention.num_heads,attention.head_v_dim)
            packed=key,value
            if frozen_cache is not None:
                frozen_cache[identity]=packed
        key,value=packed
        keys.append(key); values.append(value)
        token_slots.extend([slot]*len(key)); token_positions.extend(range(len(key)))
        mass=(read.weights*(read.slots==slot)).sum(-1).to(query.device)
        prior=mass[:,None].expand(-1,len(key))/len(key)
        if forced_value_position is not None:
            if forced_slot is None or not 0<=forced_value_position<len(key):
                raise ValueError('oracle position must name an observed successor in its specified record')
            mask=torch.arange(len(key),device=query.device)==forced_value_position
            prior=prior*mask[None]
        priors.append(prior)
    if not keys:
        return torch.zeros_like(query)
    conditional,weights=conditional_head_read(q,torch.cat(keys),torch.cat(values),
        torch.cat(priors,dim=1),wrapper.memory.temperature,return_weights=True,topk=wrapper.memory.topk)
    wrapper._native_last_alignment=dict(slots=token_slots,value_positions=token_positions,
                                         head_weights=weights[-1].detach().cpu())
    gate=attention.g_proj(normalized).reshape(1,len(query),attention.num_heads,attention.head_v_dim)
    normalized_read=attention.o_norm(conditional[None].to(query.dtype),gate)
    decoded=attention.o_proj(normalized_read.reshape(1,len(query),-1))[0]
    # Apply read/null mass AFTER normalization, so a tiny read isn't amplified
    # back to full strength by RMSNorm and a null read remains exactly neutral.
    return decoded*read.weights.sum(-1).to(query.device,decoded.dtype)[:,None]
