"""The four kinetic operators as one recursion on a unit field, online on real bytes.

    read       p(y) = sum_j z(j) V(j, y)
    collision  z+(i) is proportional to z(i) V(i, y)^beta
    transport  z'(j) = sum_i z+(i) A(i, j)

Diagnostic, not a body.  No temporal backward pass in any online arm; `yard` is batch EM with a full
backward pass, kept as a yardstick for what K units can hold.
  eye      A = I, no transport
  perm     a fixed permutation with a small leak
  permy    a random permutation for each byte, no leak, so states never merge
  filter   A learned from the filtered pair posterior, no trace
  trace    A learned by the forward statistic recursion (online EM), K^3 numbers, no history
  bytewise as filter, with one transport matrix for each arriving byte
  bath     as filter, with the global normalization replaced by the medium's local quadratic bath
           du/dt = -kappa u^3 and a small uniform source; only the read divides by the total mass
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

SEGMENT = 50_000
ALPHA = 8.0
HORIZON = 100_000.0
BURN = 2_000


def stream(total):
    values = np.asarray(np.load("D:/information-boltzmann-v1/data/ib_owt_10000/train.npy", mmap_mode="r")
                        [20_000_000:20_000_000 + int(total * 1.02)])
    return (values[values >= 4].astype(np.int64) - 4)[:total]


def floors(text):
    """Online order 0, 1, 2 byte tables, paired to the same segments."""
    rows = []
    c0, c1, c2 = np.full(256, 0.05), np.full((256, 256), 0.05), {}
    t1, t2 = np.full(256, 12.8), {}
    a, b = 32, 32
    totals = np.zeros(3)
    for position, y in enumerate(text.tolist(), 1):
        q = c0[y] / c0.sum()
        p1 = c1[b, y] / t1[b]
        key = a * 256 + b
        row = c2.get(key)
        p2 = ((row.get(y, 0.0) if row else 0.0) + 8.0 * p1) / (t2.get(key, 0.0) + 8.0)
        totals -= np.log([q, p1, p2])
        c0[y] += 1.0
        c1[b, y] += 1.0
        t1[b] += 1.0
        c2.setdefault(key, {})
        c2[key][y] = c2[key].get(y, 0.0) + 1.0
        t2[key] = t2.get(key, 0.0) + 1.0
        a, b = b, y
        if position % SEGMENT == 0:
            rows.append((totals / SEGMENT).round(4).tolist())
            totals = np.zeros(3)
    return rows


def information(unit, byte, before):
    """Nats the winning unit holds about the arrived byte, and about the byte before it given that byte."""
    def entropy(*columns):
        _, counts = np.unique(np.stack(columns), axis=1, return_counts=True)
        p = counts / counts.sum()
        return float(-(p * np.log(p)).sum())

    return dict(of_byte=round(entropy(unit) + entropy(byte) - entropy(unit, byte), 3), byte=round(entropy(byte), 3),
                of_before=round(entropy(unit, byte) + entropy(byte, before) - entropy(byte) - entropy(unit, byte, before), 3),
                before=round(entropy(byte, before) - entropy(byte), 3))


def online(text, arm, K, beta, seed, kappa=0.0, source=0.0):
    rng = np.random.default_rng(seed)
    q = np.ones(256)
    Cv = rng.gamma(0.05, 12.0 / (0.05 * 256), (K, 256))
    Tv = Cv.sum(1)
    A = np.eye(K)
    jump = None
    if arm == "perm":
        A = 0.95 * np.eye(K)[rng.permutation(K)] + 0.05 / K
    elif arm == "permy":
        jump = np.stack([rng.permutation(K) for _ in range(256)])
    elif arm in ("filter", "trace", "bath"):
        Ca = 1.0 + 0.2 * rng.random((K, K))
        A = Ca / Ca.sum(1, keepdims=True)
    if arm == "bytewise":
        Cb = 1.0 + 0.2 * rng.random((256, K, K))
        Ab = Cb / Cb.sum(2, keepdims=True)
        A = Ab[32]
    last = 32
    S = None
    if arm == "trace":
        S = np.zeros((K, K * K))
        for k in range(K):
            S[k] = (A / K).ravel()
    z = np.full(K, 1.0 / K)
    post = z.copy()
    field = np.ones(K)
    weight, total, rows, floor = 1.0, 0.0, [], 1e-300
    entropy, masses, losses, winners = 0.0, [], [], []
    for position, y in enumerate(text.tolist(), 1):
        e = (Cv[:, y] + ALPHA * q[y] / q.sum()) / (Tv + ALPHA)
        loss = -np.log(max(float(z @ e), floor))
        total += loss
        previous = post
        post = z * e ** beta
        post = post / post.sum()
        if position > len(text) - SEGMENT:
            entropy -= float(z @ np.log(np.maximum(z, 1e-30)))
            losses.append(loss)
            masses.append(float(field.sum()))
            winners.append(int(post.argmax()))
        # Content credit: the unit that explained the arrival, relaxed by mass.
        Cv[:, y] += weight * post
        Tv += weight * post
        q[y] += 1.0
        weight *= 1.0 + 1.0 / HORIZON
        if weight > 1e6:
            Cv /= weight
            Tv /= weight
            if arm in ("filter", "bath"):
                Ca /= weight
            if arm == "bytewise":
                Cb /= weight
            weight = 1.0
        if arm in ("filter", "bath") and position > BURN:
            pair = previous[:, None] * A * (e ** beta)[None, :]
            Ca += weight * pair / pair.sum()
            A = Ca / Ca.sum(1, keepdims=True)
        elif arm == "bytewise":
            if position > BURN:
                pair = previous[:, None] * A * (e ** beta)[None, :]
                Cb[last] += weight * pair / pair.sum()
                Ab[last] = Cb[last] / Cb[last].sum(1, keepdims=True)
            A = Ab[y]
            last = y
        elif arm == "trace":
            back = (previous[:, None] * A) / np.maximum(z, 1e-30)[None, :]
            rate = max(position ** -0.6, 1.0 / HORIZON)
            S = (1.0 - rate) * (back.T @ S)
            # One new pair (i, k) for each unit k: S[k, i * K + k] gains rate * back[i, k].
            S[np.arange(K)[:, None], np.arange(K)[None, :] * K + np.arange(K)[:, None]] += rate * back.T
            if position > BURN:
                pair = (post @ S).reshape(K, K)
                A = (pair + 1e-9) / (pair + 1e-9).sum(1, keepdims=True)
        if arm == "bath":
            driven = field * (e * q.sum() / q[y]) ** beta
            field = (driven / np.sqrt(1.0 + 2.0 * kappa * driven * driven)) @ A + source
            z = field / field.sum()
        elif jump is not None:
            z = np.zeros(K)
            np.add.at(z, jump[y], post)
        else:
            z = post @ A
        if position % SEGMENT == 0:
            rows.append(round(total / SEGMENT, 4))
            total = 0.0
    used = (A.max(1) > 2.0 / K).mean() if jump is None else 1.0
    mass, lost = np.log(masses), np.array(losses)
    held = information(np.array(winners), text[-len(winners):], text[-len(winners) - 1:-1])
    return dict(segments=rows, held=held, states=round(float(np.exp(entropy / SEGMENT)), 2), mass=round(float(np.mean(masses)), 3),
                mass_surprise=round(float(np.corrcoef(mass[:-1], lost[1:])[0, 1]), 4) if mass.std() > 0 else 0.0,
                unit_mass=np.sort(Tv / Tv.sum())[::-1][:4].round(3).tolist(), peaked=float(used))


def batch(text, K, seed, passes=30):
    """Baum-Welch on all but the last segment, scored forward on the last segment."""
    rng = np.random.default_rng(seed)
    train, test = text[:-SEGMENT], text[-SEGMENT:]
    A = 1.0 + 0.2 * rng.random((K, K))
    A /= A.sum(1, keepdims=True)
    V = 1.0 + rng.gamma(1.0, 1.0, (K, 256))
    V /= V.sum(1, keepdims=True)
    n = len(train)
    alpha, scale = np.zeros((n, K)), np.zeros(n)
    history = []
    for _ in range(passes):
        z = np.full(K, 1.0 / K)
        for t, y in enumerate(train):
            a = z * V[:, y]
            scale[t] = a.sum()
            alpha[t] = a / scale[t]
            z = alpha[t] @ A
        beta = np.ones(K)
        Ca, Cv = np.full((K, K), 1e-3), np.full((K, 256), 1e-3)
        Cv[np.arange(K)[:, None], train[-1]] += alpha[-1][:, None]
        for t in range(n - 2, -1, -1):
            y = train[t + 1]
            emitted = V[:, y] * beta
            Ca += alpha[t][:, None] * A * emitted[None, :] / scale[t + 1]
            beta = A @ emitted / scale[t + 1]
            Cv[:, train[t]] += alpha[t] * beta
        history.append(round(float(-np.log(scale).mean()), 4))
        A, V = Ca / Ca.sum(1, keepdims=True), Cv / Cv.sum(1, keepdims=True)
        print("  pass", len(history), history[-1], flush=True)
    z = np.full(K, 1.0 / K)
    total = 0.0
    for y in train[-5000:]:
        a = z * V[:, y]
        z = (a / a.sum()) @ A
    for y in test:
        a = z * V[:, y]
        total -= np.log(a.sum())
        z = (a / a.sum()) @ A
    return dict(train=history, held_out=round(total / len(test), 4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arm")
    parser.add_argument("--units", type=int, default=32)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--kappa", type=float, default=0.0)
    parser.add_argument("--source", type=float, default=0.01)
    parser.add_argument("--bytes", type=int, default=300_000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    text = stream(args.bytes)
    began = time.time()
    if args.arm == "floors":
        result = dict(order012=floors(text))
    elif args.arm == "yard":
        result = batch(text, args.units, args.seed)
    else:
        result = online(text, args.arm, args.units, args.beta, args.seed, args.kappa, args.source)
    result.update(vars(args) | dict(out=str(args.out), seconds=round(time.time() - began)))
    args.out.write_text(json.dumps(result), encoding="utf-8")
    print(json.dumps(result)[:400], flush=True)


if __name__ == "__main__":
    main()
