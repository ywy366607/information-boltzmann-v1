"""Bounded episodic material for the existing byte-by-byte speech path."""
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class Recall:
    slot: int
    version: int
    words: bytes
    weight: float
    feeling_distance: float


@torch.no_grad()
def retrieve(model, boundary, feeling, *, felt):
    m = model.memory
    table = model.retrieval
    valid = m.occupied & (table.word_count > 0)
    slots = valid.nonzero().flatten()
    if not len(slots):
        return ()
    keys = F.normalize(model.encoder(m.source[slots]), dim=-1)
    similarity = (keys @ F.normalize(model.encoder(boundary.reshape(1, -1)), dim=-1).T).flatten()
    distance = (m.feeling[slots]-feeling.reshape(1, -1)).square().mean(-1).sqrt()
    scores = similarity-distance if felt else similarity
    selected = scores.topk(min(len(slots), m.topk)).indices
    weights = (scores[selected]/m.temperature).softmax(0).tolist()
    return tuple(Recall(int(s), int(m.read_credit_version[s]),
                        bytes(table.words[s, :int(table.word_count[s])].cpu().tolist()),
                        w, float(d))
                 for s, w, d in zip(slots[selected], weights, distance[selected]))


def byte_log_probability(logits, prefix, recalled):
    """Offer a remembered continuation, never an entire emitted string.

    At a fresh start, each episode supplies its first byte. Later it must
    match the longest currently available suffix (up to four bytes). The
    ordinary decoder weights the candidates and remains free to emit any byte.
    """
    base = logits.log_softmax(-1)
    if not recalled:
        return base
    candidates = {}
    for n in range(min(4, len(prefix)), -1, -1):
        if n == 0 and prefix:
            break
        for r in recalled:
            if not r.words:
                continue
            indices = [j+n for j in range(len(r.words)-n)
                       if r.words[j:j+n] == prefix[-n:]] if n else [0]
            for i in indices:
                b = r.words[i]
                candidates[b] = candidates.get(b, 0.) + r.weight/len(indices)
        if candidates:
            break
    if not candidates:
        return base
    support = torch.zeros_like(base)
    for b, w in candidates.items():
        support[..., b] = w
    # One bounded evidence contribution: at most double native odds, no forced copy.
    return (base + support.log1p()).log_softmax(-1)
