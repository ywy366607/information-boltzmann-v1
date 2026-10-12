"""The curious shelf, version 1 (design: docs/information_boltzmann/FLY_CURIOUS_SHELF_DESIGN_20261012.html).

Books built only from the training split (validation never enters):
  learning zone   topic books: training documents clustered by their rare words (TF-IDF, k-means); a book is its
                  topic's documents in a fixed shuffled order
  mastered zone   one 256-token passage repeated without end
  noise zone      uniform random tokens; random tokens drawn with the unigram frequencies (same word statistics as text)
  hidden pair     one long document split into two books (first half / second half)
Every book keeps a fixed held-out probe (32 tokens) that is never read (for the noise books a fixed draw).
After every segment the book just read is probed with fixed weights from a copy of the brain's state (the state the
brain lives in is not touched); learning progress = probe NLL at the previous visit - probe NLL now.  Selectors:
progress (softmax of running progress, scale-free temperature, 10% exploration, untried books first), random,
surprise (softmax of the latest probe NLL), anti (softmax of minus progress).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

EOS = 50256
PROBE = 32


class Bookshelf:
    def __init__(self, train, selector='progress', seed=0, topics=8, segment=64, log_path=None,
                 reference_counts=None, explore=0.1):
        if selector not in ('progress', 'random', 'surprise', 'anti'):
            raise ValueError('selector must be progress, random, surprise or anti')
        self.rng = np.random.default_rng(seed)
        self.selector, self.segment, self.explore = selector, int(segment), float(explore)
        tokens = np.asarray(train)
        ends = np.flatnonzero(tokens == EOS)
        starts = np.r_[0, ends[:-1] + 1]
        docs = [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= 256]
        self.books, self.zone, self.probes, self.positions = [], [], [], []
        # hidden pair: the longest document, split in two
        longest = max(docs, key=lambda d: d[1] - d[0])
        docs.remove(longest)
        a, b = longest
        mid = (a + b) // 2
        for lo, hi in ((a, mid), (mid, b)):
            self._add_book(np.asarray(tokens[lo:hi - PROBE], dtype=np.int64), np.asarray(tokens[hi - PROBE:hi], dtype=np.int64), 'hidden pair')
        # learning zone: topic clusters by rare words
        groups = self._topics(tokens, docs, topics)
        for members in groups:
            order = self.rng.permutation(len(members))
            probe_doc = members[order[0]]
            body = np.concatenate([np.asarray(tokens[members[k][0]:members[k][1] + 1], dtype=np.int64) for k in order[1:]])
            self._add_book(body, np.asarray(tokens[probe_doc[0]:probe_doc[0] + PROBE], dtype=np.int64), 'learning')
        # mastered zone: one passage repeated
        d = docs[int(self.rng.integers(len(docs)))]
        passage = np.asarray(tokens[d[0]:d[0] + 256], dtype=np.int64)
        self._add_book(passage, passage[:PROBE].copy(), 'mastered', loop=True)
        # noise zone
        counts = reference_counts if reference_counts is not None else np.bincount(tokens.astype(np.int64), minlength=50257) + 1.0
        self.unigram = counts / counts.sum()
        # noise books keep a fixed held-out probe too: progress can only be measured on the same passage
        self._add_book(None, self.rng.integers(0, 50257, PROBE).astype(np.int64), 'noise uniform')
        self._add_book(None, self.rng.choice(50257, PROBE, p=self.unigram).astype(np.int64), 'noise unigram')
        n = len(self.books)
        self.progress = np.full(n, np.nan)
        self.last_probe = np.full(n, np.nan)
        self.visits = np.zeros(n, dtype=np.int64)
        self.current = int(self.rng.integers(n))
        self.read_in_segment = 0
        self.log = open(log_path, 'a') if log_path else None
        self.segments = 0

    def _add_book(self, body, probe, zone, loop=False):
        self.books.append((body, loop))
        self.probes.append(probe)
        self.zone.append(zone)
        self.positions.append(0)

    def _topics(self, tokens, docs, k):
        from sklearn.feature_extraction.text import TfidfTransformer
        from sklearn.cluster import MiniBatchKMeans
        from scipy.sparse import csr_matrix
        rows, cols, vals = [], [], []
        for i, (a, b) in enumerate(docs):
            ids, cnt = np.unique(np.asarray(tokens[a:b], dtype=np.int64), return_counts=True)
            rows.append(np.full(len(ids), i)); cols.append(ids); vals.append(cnt)
        X = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(docs), 50257))
        X = TfidfTransformer(sublinear_tf=True).fit_transform(X)
        labels = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=3, batch_size=2048).fit_predict(X)
        return [[docs[i] for i in np.flatnonzero(labels == c)] for c in range(k) if (labels == c).sum() >= 4]

    # ---- the stream ----
    def _draw(self, book, n):
        body, loop = self.books[book]
        zone = self.zone[book]
        if zone == 'noise uniform':
            return self.rng.integers(0, 50257, n).astype(np.int64)
        if zone == 'noise unigram':
            return self.rng.choice(50257, n, p=self.unigram).astype(np.int64)
        out, p = [], self.positions[book]
        while sum(len(x) for x in out) < n:
            take = body[p:p + n - sum(len(x) for x in out)]
            out.append(take)
            p += len(take)
            if p >= len(body):
                p = 0
        self.positions[book] = p
        return np.concatenate(out)

    def next_window(self, n):
        values = self._draw(self.current, n)
        self.read_in_segment += n
        return values

    def probe_tokens(self, book):
        return self.probes[book]

    def after_window(self, probe_fn):
        """Call after each training window; at a segment's end probes the book just read and chooses the next."""
        if self.read_in_segment < self.segment:
            return None
        self.read_in_segment = 0
        b = self.current
        nll = float(probe_fn(self.probe_tokens(b)))
        if not math.isnan(self.last_probe[b]):
            gain = self.last_probe[b] - nll
            self.progress[b] = gain if math.isnan(self.progress[b]) else 0.7 * self.progress[b] + 0.3 * gain
        self.last_probe[b] = nll
        self.visits[b] += 1
        self.current = self._choose()
        self.segments += 1
        row = {'segment': self.segments, 'read': b, 'zone': self.zone[b], 'probe_nll': nll,
               'progress': None if math.isnan(self.progress[b]) else float(self.progress[b]), 'next': self.current,
               'next_zone': self.zone[self.current]}
        if self.log:
            self.log.write(json.dumps(row) + '\n'); self.log.flush()
        return row

    def _softmax_pick(self, score):
        finite = np.isfinite(score)
        if not finite.any():
            return int(self.rng.integers(len(score)))
        s = np.where(finite, score, -np.inf)
        scale = np.median(np.abs(s[finite] - np.median(s[finite]))) + 1e-6   # scale-free temperature
        p = np.exp((s - s[finite].max()) / scale)
        p = p / p.sum()
        return int(self.rng.choice(len(score), p=p))

    def _choose(self):
        n = len(self.books)
        if self.selector == 'random' or self.rng.random() < self.explore:
            return int(self.rng.integers(n))
        # every selector tries each book first: progress needs two visits, surprise one
        untried = np.flatnonzero(self.visits < (2 if self.selector in ('progress', 'anti') else 1))
        if len(untried):
            return int(self.rng.choice(untried))
        if self.selector == 'progress':
            return self._softmax_pick(self.progress)
        if self.selector == 'anti':
            return self._softmax_pick(-self.progress)
        return self._softmax_pick(self.last_probe)        # surprise


def make_probe_fn(fl, model, learner, max_ticks, opts):
    """Mean NLL of a probe passage read from a copy of the brain's current state (the living state is untouched)."""
    def probe(tokens):
        s = learner.state.detached()
        ce = []
        with torch.no_grad():
            for i in range(len(tokens) - 1):
                s, lat, _, _, _, _ = fl.advance_fly_token_go(model, s, torch.tensor([int(tokens[i])], device='cuda'), [],
                                                             max_ticks=max_ticks, writer_baseline_clock='physical', **opts)
                ce.append(float(F.cross_entropy(model.decoder(model.read_norm(lat[-1:])).float(),
                                                torch.tensor([int(tokens[i + 1])], device='cuda'))))
        return float(np.mean(ce))
    return probe
