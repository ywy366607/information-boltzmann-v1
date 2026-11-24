"""Read-only old/new checks at the actual saved boundary, before world action."""
import time

import numpy as np
import torch

from scripts.ib_local.language_action import peek
from scripts.ib_local.navigation_body_growth import configure_body_growth
from scripts.ib_local.navigation_momentum_wood import decision_momentum_read


@torch.no_grad()
def grow_with_checks(engine, spec):
    model = engine.model
    # Use already available real source text; this does not advance the world.
    raw = np.asarray(engine.world.data[:17],dtype=np.int64)
    ids = torch.as_tensor(raw[None],device=engine.device)
    probability = peek(engine.table,engine.pending)
    logp = torch.as_tensor(np.log(np.maximum(probability,1e-6)),device=engine.device)[None,None].expand(1,len(raw),-1)
    start = time.perf_counter()
    old = model.predict_observed(ids,engine.recurrent,logp)
    old_logits, old_features = old[0].clone(),old[3].clone()
    old_latent = model.encoder(old_features).clone()
    old_p = model.momentum_to_boundary(decision_momentum_read(model,engine.bfss_momenta,old_features[:1])).clone()
    # These checks have no sampling. Explicitly preserve global random streams.
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if engine.device.type=='cuda' else None
    receipt = configure_body_growth(engine,spec)
    torch.set_rng_state(cpu_rng)
    if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
    new = model.predict_observed(ids,engine.recurrent,logp)
    new_p = model.momentum_to_boundary(decision_momentum_read(model,engine.bfss_momenta,new[3][:1]))
    receipt.update(logit_max_error=float((new[0]-old_logits).abs().max()),
                   feature_max_error=float((new[3]-old_features).abs().max()),
                   latent_max_error=float((model.encoder(new[3])-old_latent).abs().max()),
                   momentum_prediction_max_error=float((new_p-old_p).abs().max()),
                   seconds=time.perf_counter()-start, world_calls=0, optimizer_updates=0)
    if max(receipt[k] for k in ('logit_max_error','feature_max_error','latent_max_error',
                                'momentum_prediction_max_error')) > 5e-4:
        raise RuntimeError(f'growth changed inherited function: {receipt}')
    learned = {id(p) for g in engine.optimizer.param_groups for p in g['params']}
    if learned != {id(p) for p in model.parameters()}:
        raise RuntimeError('growth optimizer membership differs')
    return receipt
