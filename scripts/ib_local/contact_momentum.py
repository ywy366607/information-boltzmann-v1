"""Content-addressed momentum correction and persistent local eligibility."""
import torch
from torch import nn
from torch.nn import functional as F

from scripts.ib_local.momentum_contact_reference import contact_write


def transition(s, p, key, value, alpha, beta, impedance, phi, budget):
    residual = value-torch.einsum('bhkv,bhk->bhv', alpha[..., None, None]*s, key)
    trend = torch.einsum('bhkv,bhk->bhv', p/impedance[..., None, None], key)
    cosine = F.cosine_similarity(residual, trend, dim=-1, eps=1e-8)
    # Production beta is sigmoid; clamps only cover rounded endpoint values.
    logit = torch.logit(beta.clamp(1e-7, 1-1e-7))
    gain = (logit+phi[0]+phi[1]*cosine).sigmoid()
    sn, pn, _ = contact_write(s, p, key, value, alpha, beta, impedance,
                             gain=gain, budget=budget)
    return sn, pn


class ContactMomentum(nn.Module):
    def __init__(self, layers, heads, dk, dv, *, device=None):
        super().__init__()
        self.phi = nn.Parameter(torch.zeros(layers, 2, device=device))
        shape = (2, layers, 1, heads, dk, dv)
        self.register_buffer('sensitivity_s', torch.zeros(shape, device=device))
        self.register_buffer('sensitivity_p', torch.zeros(shape, device=device))
        self.register_buffer('last_momentum_gradient', torch.zeros_like(self.phi))
        self.register_buffer('momentum_credit_count', torch.zeros((), dtype=torch.long, device=device))
        self.register_buffer('observations', torch.zeros(layers, dtype=torch.long, device=device))

    @torch.no_grad()
    def observe(self, index, k, v, alpha, beta, s, p, impedance, budget):
        if s.is_cuda and s.dtype == torch.float32:
            from scripts.ib_local.contact_momentum_fused import capture
            sn, pn, es, ep, correction = capture(k, v, alpha, beta, s, p,
                self.sensitivity_s[:, index], self.sensitivity_p[:, index],
                impedance, self.phi[index].detach(), budget)
        else:
            es, ep = self.sensitivity_s[:, index], self.sensitivity_p[:, index]
            directions = torch.eye(2, device=s.device, dtype=s.dtype)
            correction = s.new_zeros(())
            for t in range(k.shape[1]):
                residual = v[:, t]-torch.einsum('bhkv,bhk->bhv', alpha[:, t, :, None, None]*s, k[:, t])
                proposed = alpha[:, t, :, None, None]*s+beta[:, t, :, None, None]*k[:, t, :, :, None]*residual[..., None, :]
                fn = lambda x, y, param: transition(x, y, k[:, t], v[:, t],
                    alpha[:, t], beta[:, t], impedance, param, budget)
                values, tangents = torch.vmap(lambda ds, dp, dphi: torch.func.jvp(
                    fn, (s, p, self.phi[index].detach()), (ds, dp, dphi)))(es, ep, directions)
                s, p = values[0][0], values[1][0]
                correction += (s-proposed).square().sum()
                es, ep = tangents
            correction = correction.sqrt()
            sn, pn = s, p
        self.sensitivity_s[:, index].copy_(es)
        self.sensitivity_p[:, index].copy_(ep)
        self.observations[index].add_(k.shape[1])
        return sn, pn, correction

    def diagnostics(self):
        return dict(schema=1, parameters=self.phi.detach().cpu().tolist(),
            observation_tokens=self.observations.cpu().tolist(),
            credit_calls=int(self.momentum_credit_count),
            eligibility_norm=float(self.sensitivity_s.norm()+self.sensitivity_p.norm()),
            read_gradient_per_layer=self.last_momentum_gradient.norm(dim=-1).cpu().tolist())


def configure(engine, *, initialize=True):
    from scripts.ib_local.adaptive_hamiltonian import AdaptiveHamiltonian
    adaptive = engine.model.head_coupling
    if not isinstance(adaptive, AdaptiveHamiltonian):
        raise ValueError('contact momentum requires the admitted adaptive S/P body')
    if hasattr(adaptive, 'contact'):
        return
    old_count = len(engine.params)
    module = ContactMomentum(adaptive.layers, adaptive.heads, adaptive.dk, adaptive.dv,
                             device=adaptive.theta.device)
    adaptive.contact = module
    engine.optimizer.add_param_group(dict(params=[module.phi],
        lr=engine.optimizer.param_groups[0]['lr'], weight_decay=0.))
    engine.params = [*engine.params, module.phi]
    engine.groups = (*engine.groups, [module.phi])
    if initialize:
        def extend(value):
            if isinstance(value, (list, tuple)) and len(value) == old_count and all(
                    v is None or torch.is_tensor(v) for v in value):
                return type(value)([*value, None])
            if isinstance(value, dict):
                return {k: extend(v) for k, v in value.items()}
            return value
        for name in ('prev', 'internal_pending', 'fire_grad'):
            if hasattr(engine, name):
                setattr(engine, name, extend(getattr(engine, name)))
        if hasattr(engine, 'head_mask'):
            engine.head_mask = [*engine.head_mask, True]
    engine.contact_momentum_schema = 1
