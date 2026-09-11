"""Opt-in algebraic acceleration; original checkpoint science sources stay intact."""
import math
from types import MethodType
import torch
from torch.nn import functional as F
from fine_grain.information_boltzmann.collision import reflect, spatial_kernel


def install_shared_projections(model, collision_context=True, checkpoint_drive=False):
    """Reuse input and frozen collision-context projections within their valid scope.

    Floating-point reduction order changes; this is not bitwise continuation.
    Caches retain autograd and are cleared before optimizer updates and on errors.
    """
    force, collision = model.force, model.collision
    force._shared_token_projection = None
    collision._shared_context = None

    def drive(self, x, token, time):
        dim = x.shape[-1]
        layer = self.net[0]
        shared = self._shared_token_projection
        if shared is None:
            shared = F.linear(self.embedding.weight[token], layer.weight[:, dim:-2], layer.bias)
        clock = x.new_tensor([math.sin(time), math.cos(time)])
        def project(position, shared_input, clock_input):
            z = F.linear(position, layer.weight[:, :dim]) + shared_input + F.linear(clock_input, layer.weight[:, -2:])
            return self.amplitude / math.sqrt(dim) * self.net[2](self.net[1](z)).tanh()
        if checkpoint_drive and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            return checkpoint(project,x,shared,clock,use_reentrant=False)
        return project(x,shared,clock)

    def advance_shared(self, state, observed_token, generator=None, budget=None):
        force = self.force
        dim = state.x.shape[-1]
        layer = force.net[0]
        force._shared_token_projection = F.linear(force.embedding.weight[observed_token], layer.weight[:, dim:-2], layer.bias)
        try:
            return type(self)._advance_steps(self, state, observed_token, generator, budget)
        finally:
            force._shared_token_projection = None

    def collision_shared(self, state, duration, generator=None):
        if self.max_rate == 0 or duration == 0:
            return type(self).forward(self,state,duration,generator)
        # Context is frozen by the original collision implementation for this substep.
        features = torch.cat((state.x, state.v), -1)
        self._shared_context = (self.key(features), self.value(features))
        try:
            return type(self).forward(self, state, duration, generator)
        finally:
            self._shared_context = None

    def rate(self, x, v, w, normal, context_x, context_v):
        if self._shared_context is None:
            features = torch.cat((context_x, context_v), -1)
            keys, values = self.key(features), self.value(features)
        else:
            keys, values = self._shared_context
        dim = x.shape[-1]
        vp, wp = reflect(v, w, normal)
        orbit = torch.stack([torch.cat((a,b,sign*normal)) for a,b in ((v,w),(w,v),(vp,wp),(wp,vp)) for sign in (1,-1)])
        q = self.query(orbit)
        # Translation of every key adds the same scalar to a query's logits,
        # which cancels in softmax. Values require the explicit translation.
        weights = spatial_kernel(context_x-x, self.width)
        scores = q @ keys.T / math.sqrt(q.shape[-1]) + weights.clamp_min(1e-12).log()
        attended = scores.softmax(-1) @ values - F.linear(x, self.value.weight[:,:dim])
        score = self.output(torch.tanh(attended+q)).mean()
        return self.max_rate*torch.sigmoid(score.clamp(-12,12))

    force.drive = MethodType(drive, force)
    model._advance_steps = MethodType(advance_shared, model)
    if not collision_context:
        return model
    collision.forward = MethodType(collision_shared, collision)
    collision.rate = MethodType(rate, collision)
    return model
