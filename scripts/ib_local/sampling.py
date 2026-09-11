"""CPU proposals generated in bulk; four device transfers per optimizer window."""
import numpy as np
import torch


def padded_capacities(particles):
    if particles > 256:
        return (128, 32, 16, 8, 4, 2)
    if particles > 128:
        return (64, 32, 8, 4, 2, 1)
    return (32, 16, 8, 4, 2, 1)


def sample_window(rng, length, steps, particles, device, dtype=torch.float32, padding=False):
    counts = rng.poisson((particles - 1) / (2 * steps), size=length * steps)
    total = int(counts.sum())
    i = rng.integers(particles, size=total)
    j = rng.integers(particles - 1, size=total)
    j += j >= i
    normal = rng.normal(size=(total, 4))
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    uniform = rng.random(total)
    order, boundaries, cursor = [], [], 0
    for count in counts:
        levels, last = [], [-1] * particles
        for index in range(cursor, cursor + int(count)):
            level = 1 + max(last[i[index]], last[j[index]])
            last[i[index]] = last[j[index]] = level
            while len(levels) <= level:
                levels.append([])
            levels[level].append(index)
        spans = []
        capacities = padded_capacities(particles) if padding else ()
        while len(levels) < len(capacities):
            levels.append([])
        for depth, level in enumerate(levels):
            start = len(order)
            order.extend(level)
            if padding:
                capacity = capacities[depth] if depth < len(capacities) else 1
                while capacity < len(level):
                    capacity *= 2
                order.extend([total] * (capacity - len(level)))
            spans.append((start, len(order)))
        boundaries.append(spans)
        cursor += int(count)
    order = np.asarray(order, dtype=np.int64)
    if padding:
        i, j = np.append(i, 0), np.append(j, 1)
        normal = np.concatenate((normal, [[1., 0., 0., 0.]]))
        uniform = np.append(uniform, .5)
    arrays = [torch.as_tensor(a[order], device=device, dtype=kind) for a, kind in
              ((i, torch.int64), (j, torch.int64), (normal, dtype), (uniform, dtype))]
    if padding:
        arrays.append(torch.as_tensor(order != total, device=device))
    tables = [[tuple(a[start:end] for a in arrays) for start, end in spans] for spans in boundaries]
    return tables, total
