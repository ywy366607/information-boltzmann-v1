"""Exact native address cells in the current encoder's lifted coordinates.

The packet is a read-only, finite sufficient snapshot for binding_address():
live identities, encoded centers, raw halfspaces, threshold and affine lift.
It retains the body's raw scope tests and cosine rule. No branch value or
future label participates in the cell definition.
"""
import torch
from torch.nn import functional as F


@torch.no_grad()
def snapshot(model, raw_mean, latent_mean):
    bank=model.causal_relations
    w=model.encoder.weight
    from scripts.ib_local.preview_runtime import read,tensor_key
    right=read(model,'decision_right_inverse',tensor_key(w),
        lambda:torch.linalg.solve(w.double()@w.double().T,w.double()).T.to(w),anchors=(w,))
    packet=dict(identity=bank.identity.detach().clone(),
        centers=model.encoder(bank.condition).detach().clone(),
        axes=bank.scope_axes.detach().clone(),bounds=bank.scope_bounds.detach().clone(),
        signs=bank.scope_signs.detach().clone(),valid=bank.scope_valid.detach().clone(),
        right=right,raw_mean=raw_mean.detach().clone(),mean=latent_mean.detach().clone(),
        threshold=float(model.memory.threshold),encoder_weight=w.detach().clone(),
        encoder_bias=None if model.encoder.bias is None else model.encoder.bias.detach().clone())
    return packet


@torch.no_grad()
def addresses(packet, points, *, chunk=1024):
    """Vectorized native rule. Outputs identity, max similarity and acceptance."""
    if points.ndim!=2 or points.shape[-1]!=packet['mean'].shape[-1]:
        raise ValueError('one latent row per successor point required')
    axes=packet['axes']
    projected_axes=axes@packet['right']
    base=(axes@packet['raw_mean'][0])
    centers=F.normalize(packet['centers'],dim=-1)
    live=packet['identity']>0
    identities=[];scores=[];scope_counts=[]
    for z in points.split(chunk):
        projection=base[None]+torch.einsum('bd,ked->bke',z-packet['mean'],projected_axes)
        tolerance=torch.finfo(z.dtype).eps*(1+projection.abs()+packet['bounds'].abs())*8
        scope=((packet['signs']*(projection-packet['bounds'])<=tolerance)|~packet['valid']).all(-1)
        eligible=scope&live
        cosine=F.normalize(z,dim=-1)@centers.T
        score,index=cosine.masked_fill(~eligible,-2).max(-1)
        matched=eligible.any(-1)&(score>=packet['threshold'])
        identities.append(torch.where(matched,packet['identity'][index],-1))
        scores.append(score);scope_counts.append(eligible.sum(-1))
    return dict(identity=torch.cat(identities),similarity=torch.cat(scores),
                eligible_scopes=torch.cat(scope_counts))


@torch.no_grad()
def native_check(model,packet,points):
    """Check actual raw lift/halfspace/cosine operations, including fresh cells."""
    got=addresses(packet,points)
    expected=[];scores=[]
    for z in points:
        raw=packet['raw_mean']+(z-packet['mean'])@packet['right'].T
        matched,index,score=model.causal_relations.binding_address(model,raw)
        expected.append(int(model.causal_relations.identity[index]) if matched else -1)
        scores.append(float(score))
    return dict(identities_equal=got['identity'].tolist()==expected,
        max_similarity_error=float((got['similarity']-got['similarity'].new_tensor(scores)).abs().max()),
        max_encoder_residual=float((model.encoder(packet['raw_mean']+
            (points-packet['mean'])@packet['right'].T)-points).abs().max()))
