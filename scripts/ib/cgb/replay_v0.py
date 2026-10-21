"""CGB V0 on recorded mother speech (no API): does the minimal brain learn the stream it lives in?

The mother's turns of earlier runs are replayed in order, word by word, with one beat of silence between turns.
Online baselines on the same stream (predict, then count): unigram and interpolated bigram, both add-beta to uniform.
Prints block means [min | median | mean | max] of the per-word surprisal (nats) and writes every word to words.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import tiktoken

from information_boltzmann.cgb.v0 import Brain, VOCAB

SOURCES = ['E:/fly_stream_overflow/newborn/mother.jsonl', 'E:/fly_stream_overflow/mengli/mother.jsonl',
           'E:/fly_stream_overflow/newborn_reading/mother.jsonl']


class Ngram:
    def __init__(self, beta=1.0, gamma=2.0):
        self.uni, self.n, self.bi, self.ctx = defaultdict(int), 0, defaultdict(int), defaultdict(int)
        self.tri, self.ctx2 = defaultdict(int), defaultdict(int)
        self.beta, self.gamma = beta, gamma

    def p_uni(self, w):
        return (self.uni[w] + self.beta / VOCAB) / (self.n + self.beta)

    def surprisal(self, prev2, prev, w):
        pu = self.p_uni(w)
        pb = (self.bi[prev, w] + self.gamma * pu) / (self.ctx[prev] + self.gamma)
        pt = (self.tri[prev2, prev, w] + self.gamma * pb) / (self.ctx2[prev2, prev] + self.gamma)
        return -math.log(pu), -math.log(pb), -math.log(pt)

    def update(self, prev2, prev, w):
        self.uni[w] += 1; self.n += 1; self.bi[prev, w] += 1; self.ctx[prev] += 1
        self.tri[prev2, prev, w] += 1; self.ctx2[prev2, prev] += 1


def dist(x):
    x = np.asarray(x)
    return f'[{x.min():.2f} | {np.median(x):.2f} | {x.mean():.2f} | {x.max():.2f}]'


def main(a):
    enc = tiktoken.get_encoding('gpt2')
    turns = []
    for src in a.sources:
        for line in Path(src).read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row.get('who') == 'mother' and row.get('text', '').strip():
                turns.append(enc.encode(' ' + row['text'].strip() + '\n'))
    print(f'{len(turns)} mother turns, {sum(map(len, turns))} words', flush=True)
    brain = Brain(dict({'seed': a.seed, 'trace_gain': a.trace_gain}, **json.loads(a.genome)))
    ng = Ngram()
    a.out.mkdir(parents=True, exist_ok=True)
    log = (a.out / 'words.jsonl').open('w', encoding='utf-8')
    block, prev, words, t0 = [], enc.encode('\n')[0], 0, time.time()
    prev2 = prev
    for turn in turns:
        for w in turn:
            su, sb, st = ng.surprisal(prev2, prev, w)
            s = brain.heard_from_world(w)
            ng.update(prev2, prev, w)
            prev2, prev = prev, w
            block.append((s, su, sb, st))
            log.write(json.dumps({'w': words, 'tok': w, 's': round(s, 4), 'uni': round(su, 4), 'bi': round(sb, 4), 'tri': round(st, 4)}) + '\n')
            words += 1
            if words % a.block == 0:
                b = np.array(block)
                print(f'{words:6d} words {time.time() - t0:6.1f}s | brain {dist(b[:, 0])} | unigram {b[:, 1].mean():.2f} '
                      f'| bigram {b[:, 2].mean():.2f} | trigram {b[:, 3].mean():.2f} | x dead {brain.stats()["x_rate_dead"]:.3f}',
                      flush=True)
                block = []
            if words >= a.words:
                break
        if words >= a.words:
            break
        brain.silence(8)
    log.close()
    np.savez(a.out / 'brain.npz', **brain.state_dict())


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources', nargs='+', default=SOURCES)
    p.add_argument('--out', type=Path, default=Path('F:/fly_stream_overflow/cgb_v0/replay'))
    p.add_argument('--words', type=int, default=10 ** 9)
    p.add_argument('--block', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--trace-gain', type=float, default=1.0)
    p.add_argument('--genome', default='{}', help='JSON overrides of GENOME')
    main(p.parse_args())
