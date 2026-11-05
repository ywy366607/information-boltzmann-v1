"""What limits one unit field?  The same unit budget U under four codes, given an ideal address.

Contexts are the 12000 most frequent three-byte suffixes of the first half of the text, the rest pooled.
No recursion here: this bounds what a state could hold if it reached the context exactly.
  hard    one field, one active unit
  soft    one field, a mixture of units, the read of the kinetic equations
  fields  M fields of U / M units, one active in each, read as a product of experts
  signed  U signed real numbers, the truncated SVD read
Scored on the last quarter, each model mixed with 2% unigram.
"""
import json
from pathlib import Path

import numpy as np

N, ORDER, SMOOTH = 2_000_000, 3, 0.02
values = np.asarray(np.load("D:/information-boltzmann-v1/data/ib_owt_10000/train.npy", mmap_mode="r")
                    [20_000_000:20_000_000 + int(N * 1.02)])
text = (values[values >= 4].astype(np.int64) - 4)[:N]
rng = np.random.default_rng(1)

key = np.zeros(N, dtype=np.int64)
for lag in range(1, ORDER + 1):
    key = key * 256 + np.roll(text, lag)
half = N // 2
names, counts = np.unique(key[ORDER:half], return_counts=True)
keep = np.sort(names[np.argsort(-counts)[:12000]])
place = np.searchsorted(keep, key[ORDER:]).clip(0, len(keep) - 1)
ids = np.where(keep[place] == key[ORDER:], place, len(keep))
y = text[ORDER:]
rows = len(keep) + 1


def table(x, target):
    joint = np.zeros((rows, 256))
    np.add.at(joint, (x, target), 1.0)
    return joint


train = table(ids[:half - ORDER], y[:half - ORDER])
test = table(ids[3 * N // 4 - ORDER:], y[3 * N // 4 - ORDER:])
test = test / test.sum()
mass = train.sum(1, keepdims=True)
pc = np.maximum(mass, 1e-9) / mass.sum()
py = train.sum(0, keepdims=True) / train.sum()
empirical = train / np.maximum(mass, 1e-9)


def score(predicted):
    predicted = (1.0 - SMOOTH) * predicted + SMOOTH * py
    return round(float(-(test * np.log(np.maximum(predicted, 1e-12))).sum()), 4)


def hard(units, passes=40):
    assign = rng.integers(0, units, rows)
    for _ in range(passes):
        V = np.zeros((units, 256))
        np.add.at(V, assign, train)
        V = (V + py + 1e-6) / (V + py + 1e-6).sum(1, keepdims=True)
        fresh = (train @ np.log(V).T).argmax(1)
        if (fresh == assign).all():
            break
        assign = fresh
    return V[assign]


def soft(units, passes=120):
    a = rng.dirichlet(np.ones(units) * 0.3, rows)
    V = rng.dirichlet(np.ones(256) * 0.3, units)
    for _ in range(passes):
        model = np.maximum(a @ V, 1e-12)
        ratio = train / model
        new_a = a * (ratio @ V.T)
        new_V = V * (a.T @ ratio)
        a = (new_a + 1e-6) / (new_a + 1e-6).sum(1, keepdims=True)
        V = (new_V + 1e-6 * py) / (new_V + 1e-6 * py).sum(1, keepdims=True)
    return a @ V


def fields(units, count, passes=10, steps=4):
    size = units // count
    base = np.log(np.maximum(py, 1e-9))
    assign = rng.integers(0, size, (count, rows))
    W = np.zeros((count, size, 256))

    def logits(skip=None):
        total = np.repeat(base, rows, 0)
        for m in range(count):
            if m != skip:
                total = total + W[m][assign[m]]
        return total

    for _ in range(passes):
        for m in range(count):
            rest = logits(m)
            for _ in range(steps):
                full = rest + W[m][assign[m]]
                p = np.exp(full - full.max(1, keepdims=True))
                p = p / p.sum(1, keepdims=True)
                gradient = train - mass * p
                total = np.zeros((size, 256))
                np.add.at(total, assign[m], gradient)
                curve = np.zeros((size, 256))
                np.add.at(curve, assign[m], mass * p * (1.0 - p) + 0.5)
                W[m] += np.clip(total / curve, -2.0, 2.0)
            # Reassign each context to the unit that best explains it given the other fields.
            best, choice = -np.full(rows, np.inf), assign[m].copy()
            for unit in range(size):
                candidate = rest + W[m][unit]
                candidate = candidate - np.log(np.exp(candidate - candidate.max(1, keepdims=True)).sum(1, keepdims=True)) \
                    - candidate.max(1, keepdims=True)
                value = (train * candidate).sum(1)
                better = value > best
                best[better], choice[better] = value[better], unit
            assign[m] = choice
    full = logits()
    p = np.exp(full - full.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


def signed(units):
    px = pc
    Q = (train / train.sum()) / np.sqrt(px * np.maximum(py, 1e-12))
    left, values_, right = np.linalg.svd(Q, full_matrices=False)
    ratio = ((left[:, :units] * values_[:units]) @ right[:units]) * np.sqrt(np.maximum(py, 1e-12) / px)
    predicted = np.maximum(ratio, 0.0) * py
    predicted = predicted + 1e-9 * py
    return predicted / predicted.sum(1, keepdims=True)


result = {"unigram": score(np.repeat(py, rows, 0)), "table": score(empirical + (mass == 0) * py)}
print(result, flush=True)
for units in (64, 256):
    row = dict(units=units, hard=score(hard(units)), soft=score(soft(units)), signed=score(signed(units)))
    print(row, flush=True)
    for count in (2, 4, 8):
        row[f"fields{count}"] = score(fields(units, count))
        print(units, count, row[f"fields{count}"], flush=True)
    result[str(units)] = row
Path("results/cgb_kinetic_filter_20261102/capacity.json").write_text(json.dumps(result), encoding="utf-8")
