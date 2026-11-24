"""Identity-coupled quadrature and content-ranked finite action proposals.

Revision 1 is enabled explicitly on the model for paired whole-body runs.
Fact identity fixes a noise draw across permutations, subsets and restarts.
Actual event time remains in Evidence; presentation order is only a solver index.
"""
from functools import lru_cache
import hashlib

import torch
from torch.nn import functional as F


def fact_identity(evidence):
    from scripts.ib_local.relation_evidence import evidence_key
    return f'{evidence.kind.value}:{int(evidence.observed)}:{evidence_key(evidence)}'


@lru_cache(maxsize=512)
def _cpu_direction(identity, hypotheses, width):
    seed = int.from_bytes(hashlib.sha256(
        ('mengli-fact-quadrature-v1:' + identity).encode('utf8')).digest()[:8], 'little')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    half = (hypotheses + 1) // 2
    positive = torch.randn(half, width, generator=generator, dtype=torch.float64)
    return torch.cat((positive, -positive), 0)[:hypotheses]


def observation_noise(identities, template):
    """Return H x facts x latent, preserving each fact's antithetic sample."""
    h, _, d = template.shape
    if not identities:
        return template.new_empty((h, 0, d))
    return torch.stack([_cpu_direction(key, h, d) for key in identities], 1).to(template)


def ranked_action_facts(model, source, facts, budget):
    """Rank remembered actions by current learned condition similarity.

    Stable identity resolves equal scores; a fact's reply is never consulted.
    The existing continuation then evaluates the admitted proposed actions.
    """
    facts = tuple(sorted(facts, key=fact_identity))
    if not facts or budget <= 0:
        return ()
    encoded = model.encoder(torch.cat([e.source for e in facts]))
    current = model.encoder(source.reshape(1, -1))
    similarity = (F.normalize(encoded, dim=-1) * F.normalize(current, dim=-1)).sum(-1)
    indices = torch.argsort(similarity, descending=True, stable=True)[:budget].tolist()
    return tuple(facts[i] for i in indices)
