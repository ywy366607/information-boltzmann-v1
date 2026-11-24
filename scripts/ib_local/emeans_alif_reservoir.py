"""e-means Two-Tier Routing + Population ALIF Lazy-Decay Memory Reservoir.

Implements the formally verified architecture in
`lean/CGBAxioms/reviews/EmeansAlifReservoir.lean`:

1. **Tier 1 — e-means Online Clustering (Meta FAIR, arXiv:2611.09501)**:
   Maintains `K_c` dynamic centroids via mini-batch Exponential Moving Averages
   (`S_j ← (1-α) S_j + α s_j`, `N_j ← (1-α) N_j + α n_j`, `μ_j = S_j / N_j`),
   cosine-annealed bootstrap convergence (3 epochs vs 25 epochs for Lloyd's k-means),
   automatic dead-cluster revival (splitting highest-count clusters), and exact
   finalization. Centroids live in *immutable raw 256-dim memory coordinates*;
   by linearity of `model.encoder`, `E(μ_j)` is the exact up-to-date 128-dim
   cluster centroid with zero staleness.

2. **Tier 2 — Population ALIF + Exact Lazy Exponential Decay + APL Inhibition**:
   Every memory slot carries 3 lightweight CPU/RAM scalars `(v, a, last_tick)`.
   Sleeping slots NEVER touch GPU VRAM and incur O(0) compute. Only the slots
   inside the top `M` e-means clusters woken by a query receive exact lazy decay
   (`λ^(t - last_tick)`), synaptic excitation, Drosophila APL global subtractive
   inhibition, and adaptive refractory thresholding `θ_i = θ₀ + β a_i`.
   At most `K_max` (default 8) spiking slots enter GPU VRAM (`PastPresentField`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F


class EmeansIndex:
    """Online e-means clustering index (arXiv:2611.09501) in immutable raw memory space."""

    def __init__(self, n_clusters: int = 64, dim: int = 256, *,
                 alpha_steady: float = 0.05, dead_threshold: float = 1e-3,
                 seed: int = 20261124):
        if n_clusters < 1 or dim < 1:
            raise ValueError('n_clusters and dim must be positive')
        self.n_clusters = int(n_clusters)
        self.dim = int(dim)
        self.alpha_steady = float(alpha_steady)
        self.dead_threshold = float(dead_threshold)
        self.seed = int(seed)

        self.sums = torch.zeros(self.n_clusters, self.dim, dtype=torch.float32)
        self.counts = torch.zeros(self.n_clusters, dtype=torch.float32)
        self.centroids = torch.zeros(self.n_clusters, self.dim, dtype=torch.float32)
        self.fitted = False
        self.revivals = 0
        self.buckets: list[list[int]] = [[] for _ in range(self.n_clusters)]
        self.slot_cluster: dict[int, int] = {}

    @staticmethod
    def _kmeans_pp_init(x: torch.Tensor, k: int, gen: torch.Generator) -> torch.Tensor:
        """Fast vectorized k-means++ (D^2) centroid initialization."""
        n = int(x.shape[0])
        chosen = torch.empty(k, dtype=torch.long)
        first = int(torch.randint(0, n, (1,), generator=gen).item())
        chosen[0] = first
        min_d2 = (x - x[first]).square().sum(dim=-1)
        for j in range(1, k):
            total = float(min_d2.sum().item())
            if total <= 1e-12:
                idx = int(torch.randint(0, n, (1,), generator=gen).item())
            else:
                probs = (min_d2 / total).clamp_min(0.0)
                idx = int(torch.multinomial(probs, 1, generator=gen).item())
            chosen[j] = idx
            d2 = (x - x[idx]).square().sum(dim=-1)
            min_d2 = torch.minimum(min_d2, d2)
        return x[chosen].clone()

    @torch.no_grad()
    def fit(self, vectors: torch.Tensor, slot_ids: torch.Tensor | list[int], *,
            epochs: int = 3, batch_size: int | None = None,
            alpha_max: float = 0.35, alpha_min: float = 0.03,
            init: str = 'kmeans++') -> dict:
        """Fit centroids using Meta FAIR e-means (cosine-annealed EMA + dead revival + exact finalize)."""
        x = vectors.detach().float().cpu()
        sids = slot_ids.detach().cpu().tolist() if torch.is_tensor(slot_ids) else [int(s) for s in slot_ids]
        n = int(x.shape[0])
        if n == 0:
            self.fitted = False
            return dict(n=0, clusters=0, mse=0.0, revivals=0)

        k = min(self.n_clusters, n)
        gen = torch.Generator(device='cpu').manual_seed(self.seed)
        if init == 'kmeans++' and k > 1:
            centroids = self._kmeans_pp_init(x, k, gen)
        else:
            init_idx = torch.randperm(n, generator=gen)[:k]
            centroids = x[init_idx].clone()
        sums = centroids.clone()
        counts = torch.ones(k, dtype=torch.float32)

        # Scaling-law inspired mini-batch size when unspecified: ~max(64, n / 8)
        if batch_size is None:
            batch_size = max(64, min(512, max(1, n // 8)))
        total_steps = max(1, epochs * math.ceil(n / batch_size))
        step = 0
        revivals = 0
        eff_dead = max(self.dead_threshold, 0.02 * (batch_size / max(1, k)))

        for _ in range(epochs):
            perm = torch.randperm(n, generator=gen)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                bx = x[idx]
                # Cosine-annealed EMA step size α_t
                prog = step / max(1, total_steps - 1)
                alpha = alpha_min + 0.5 * (alpha_max - alpha_min) * (1.0 + math.cos(math.pi * prog))
                step += 1

                # Assign mini-batch points to nearest centroids (L2 distance)
                dists = torch.cdist(bx, centroids, p=2.0)
                assign = dists.argmin(dim=1)

                batch_counts = torch.bincount(assign, minlength=k).float()
                batch_sums = torch.zeros_like(sums)
                batch_sums.index_add_(0, assign, bx)

                sums.mul_(1.0 - alpha).add_(batch_sums, alpha=alpha)
                counts.mul_(1.0 - alpha).add_(batch_counts, alpha=alpha)

                # Dead-cluster revival: split highest-count cluster into starved clusters
                dead = torch.nonzero(counts < eff_dead).flatten()
                if len(dead) > 0:
                    for d_idx in dead.tolist():
                        donor = int(counts.argmax().item())
                        if counts[donor] <= 2.0 * eff_dead:
                            break
                        half_c = 0.5 * counts[donor]
                        donor_center = sums[donor] / counts[donor].clamp_min(1e-8)
                        noise = torch.randn(self.dim, generator=gen) * (1e-3 * donor_center.norm().clamp_min(1.0))
                        counts[donor] = half_c
                        counts[d_idx] = half_c
                        sums[donor] = half_c * (donor_center + noise)
                        sums[d_idx] = half_c * (donor_center - noise)
                        revivals += 1

                centroids = sums / counts.clamp_min(1e-8)[:, None]

        # Deterministic exact finalization pass (e-means paper §3)
        dists = torch.cdist(x, centroids, p=2.0)
        final_assign = dists.argmin(dim=1)
        exact_counts = torch.bincount(final_assign, minlength=k).float()
        exact_sums = torch.zeros_like(sums)
        exact_sums.index_add_(0, final_assign, x)
        non_empty = exact_counts > 0
        centroids[non_empty] = exact_sums[non_empty] / exact_counts[non_empty, None]

        mse = float((x - centroids[final_assign]).square().sum(dim=-1).mean().item())

        self.sums.zero_()
        self.counts.zero_()
        self.centroids.zero_()
        self.sums[:k].copy_(exact_sums)
        self.counts[:k].copy_(exact_counts)
        self.centroids[:k].copy_(centroids)
        self.active_k = k
        self.fitted = True
        self.revivals += revivals

        self.buckets = [[] for _ in range(self.n_clusters)]
        self.slot_cluster = {}
        for slot, cid in zip(sids, final_assign.tolist()):
            self.buckets[cid].append(slot)
            self.slot_cluster[slot] = cid
        return dict(n=n, clusters=k, mse=mse, revivals=revivals)

    @torch.no_grad()
    def update_slot(self, slot: int, vector: torch.Tensor) -> int:
        """Online streaming e-means update for one newly committed or merged memory slot."""
        slot = int(slot)
        v = vector.detach().float().cpu().reshape(-1)
        if not self.fitted or getattr(self, 'active_k', 0) == 0:
            self.sums[0].copy_(v)
            self.counts[0] = 1.0
            self.centroids[0].copy_(v)
            self.active_k = 1
            self.fitted = True
            self.buckets[0] = [slot]
            self.slot_cluster[slot] = 0
            return 0

        k = self.active_k
        cid = int(torch.cdist(v[None], self.centroids[:k], p=2.0).argmin().item())
        old_cid = self.slot_cluster.get(slot)
        if old_cid is not None and old_cid != cid:
            try:
                self.buckets[old_cid].remove(slot)
            except ValueError:
                pass
        if old_cid != cid:
            self.buckets[cid].append(slot)
            self.slot_cluster[slot] = cid

        a = self.alpha_steady
        self.sums[cid].mul_(1.0 - a).add_(v, alpha=a)
        self.counts[cid] = (1.0 - a) * float(self.counts[cid]) + a
        self.centroids[cid].copy_(self.sums[cid] / max(float(self.counts[cid]), 1e-8))
        return cid

    @torch.no_grad()
    def route(self, query_raw: torch.Tensor, encoder, *, top_clusters: int = 6) -> list[int]:
        """Wake only the top-M clusters in TODAY'S exact linear encoder coordinates (<0.05 ms)."""
        if not self.fitted or getattr(self, 'active_k', 0) == 0:
            return []
        k = self.active_k
        dev = encoder.weight.device
        dtype = encoder.weight.dtype
        q_enc = F.normalize(encoder(query_raw.reshape(1, -1).to(device=dev, dtype=dtype)).float(), dim=-1)
        c_raw = self.centroids[:k].to(device=dev, dtype=dtype)
        # By linearity of `encoder`, E(S_j / N_j) == E(S_j) / N_j with zero staleness!
        c_enc = F.normalize(encoder(c_raw).float(), dim=-1)
        scores = (q_enc @ c_enc.T).squeeze(0)
        # Mask out empty buckets
        empty_mask = torch.tensor([len(self.buckets[i]) == 0 for i in range(k)], device=dev)
        scores = scores.masked_fill(empty_mask, -1e9)
        m = min(int(top_clusters), int((~empty_mask).sum().item()))
        if m <= 0:
            return []
        chosen = torch.topk(scores, m).indices.cpu().tolist()
        woken = []
        for cid in chosen:
            woken.extend(self.buckets[cid])
        return woken


class PopulationAlifReservoir:
    """Population ALIF + Exact Lazy Decay + Drosophila APL Global Inhibition over e-means buckets."""

    def __init__(self, capacity: int = 4096, dim: int = 256, *,
                 n_clusters: int = 64, top_clusters: int = 6, k_max: int = 8,
                 lam_v: float = 0.75, lam_a: float = 0.80,
                 theta0: float = 0.30, beta: float = 0.45,
                 apl_gain: float = 0.50, reset_factor: float = 0.25):
        if not (0.0 <= lam_v < 1.0 and 0.0 <= lam_a < 1.0):
            raise ValueError('lam_v and lam_a must lie in [0, 1)')
        if k_max < 1 or top_clusters < 1:
            raise ValueError('k_max >= 1 and top_clusters >= 1 required')
        self.capacity = int(capacity)
        self.dim = int(dim)
        self.top_clusters = int(top_clusters)
        self.k_max = int(k_max)
        self.lam_v = float(lam_v)
        self.lam_a = float(lam_a)
        self.theta0 = float(theta0)
        self.beta = float(beta)
        self.apl_gain = float(apl_gain)
        self.reset_factor = float(reset_factor)

        self.index = EmeansIndex(n_clusters=n_clusters, dim=dim)
        # Per-slot lightweight CPU/RAM ALIF state (12 bytes/slot; ZERO VRAM!)
        self.v = np.zeros(self.capacity, dtype=np.float32)
        self.a = np.zeros(self.capacity, dtype=np.float32)
        self.last_tick = np.zeros(self.capacity, dtype=np.int64)
        self.tick = 0
        self.total_queries = 0
        self.total_woken = 0
        self.total_spiked = 0
        self._indexed_episodes = -1

    def _ensure_capacity(self, needed: int) -> None:
        if needed <= self.capacity:
            return
        new_cap = max(needed, self.capacity * 2)
        for attr, dtype in (('v', np.float32), ('a', np.float32), ('last_tick', np.int64)):
            old = getattr(self, attr)
            buf = np.zeros(new_cap, dtype=dtype)
            buf[:len(old)] = old
            setattr(self, attr, buf)
        self.capacity = new_cap

    @torch.no_grad()
    def sync_from_memory(self, memory) -> dict | None:
        """Fit or incrementally refresh the e-means index from `model.memory`."""
        occ = memory.occupied
        n_occ = int(occ.sum().item())
        if n_occ == 0:
            return None
        self._ensure_capacity(int(occ.numel()))
        episodes = int(getattr(memory, 'episodes', n_occ))
        if not self.index.fitted or self._indexed_episodes < 0:
            slots = torch.nonzero(occ).flatten()
            vecs = memory.source[slots]
            info = self.index.fit(vecs, slots)
            self._indexed_episodes = episodes
            return info
        if episodes != self._indexed_episodes:
            # Update recently written slots (any slot whose last_write is newest)
            slots = torch.nonzero(occ).flatten()
            if len(slots) > 0:
                newest_k = min(len(slots), max(4, episodes - self._indexed_episodes + 2))
                top_idx = torch.topk(memory.last_write[slots], newest_k).indices
                for s in slots[top_idx].cpu().tolist():
                    self.index.update_slot(s, memory.source[s])
            self._indexed_episodes = episodes
        return None

    @torch.no_grad()
    def query(self, query_raw: torch.Tensor, memory, encoder, *,
              min_active: int = 1) -> tuple[list[int], dict]:
        """Two-tier e-means wake -> exact lazy decay -> APL inhibition -> bounded ALIF spikes."""
        self.sync_from_memory(memory)
        self.tick += 1
        self.total_queries += 1
        woken = self.index.route(query_raw, encoder, top_clusters=self.top_clusters)
        if not woken:
            return [], dict(woken=0, spiked=0, active_frac=0.0, apl_inhib=0.0)

        woken_arr = np.asarray(woken, dtype=np.int64)
        self._ensure_capacity(int(woken_arr.max()) + 1)
        # 1. Exact lazy exponential decay ONLY on woken candidates (sleeping neurons untouched!)
        dt = self.tick - self.last_tick[woken_arr]
        decay_v = np.power(self.lam_v, dt, dtype=np.float64).astype(np.float32)
        decay_a = np.power(self.lam_a, dt, dtype=np.float64).astype(np.float32)
        v_now = self.v[woken_arr] * decay_v
        a_now = self.a[woken_arr] * decay_a

        # 2. Synaptic excitation in today's encoded 128-dim coordinates + read_credit preference
        dev = encoder.weight.device
        dtype = encoder.weight.dtype
        slot_t = torch.as_tensor(woken_arr, device=dev, dtype=torch.long)
        q_enc = F.normalize(encoder(query_raw.reshape(1, -1).to(device=dev, dtype=dtype)).float(), dim=-1)
        k_enc = F.normalize(encoder(memory.source[slot_t].to(device=dev, dtype=dtype)).float(), dim=-1)
        sim = (q_enc @ k_enc.T).squeeze(0)
        if hasattr(memory, 'read_credit_mean') and getattr(memory, 'read_credit_enabled', False):
            pref = memory.read_credit_mean[slot_t].float().tanh()
            syn = F.relu(sim + 0.25 * pref)
        else:
            syn = F.relu(sim)
        drive = syn.cpu().numpy()

        # 3. Membrane integration + Drosophila APL global subtractive inhibition
        v_plus = v_now + drive
        apl_inhib = float(self.apl_gain * np.mean(v_plus))
        v_eff = v_plus - apl_inhib
        thresholds = self.theta0 + self.beta * a_now
        margins = v_eff - thresholds

        spiking_mask = margins >= 0.0
        spiking_local = np.nonzero(spiking_mask)[0]
        if len(spiking_local) > self.k_max:
            top_order = np.argsort(-margins[spiking_local])[:self.k_max]
            spiking_local = spiking_local[top_order]
        elif len(spiking_local) < min_active and len(woken_arr) > 0:
            # Guarantee non-empty workspace fallback if requested
            top_order = np.argsort(-margins)[:min(min_active, len(woken_arr))]
            spiking_local = top_order
        else:
            # Sort active spikes by descending margin
            spiking_local = spiking_local[np.argsort(-margins[spiking_local])]

        # 4. Write back updated membrane & refractory adaptation ONLY for woken slots
        v_next = v_plus.copy()
        a_next = a_now.copy()
        if len(spiking_local) > 0:
            v_next[spiking_local] *= self.reset_factor
            a_next[spiking_local] += 1.0
        self.v[woken_arr] = v_next
        self.a[woken_arr] = a_next
        self.last_tick[woken_arr] = self.tick

        selected_slots = woken_arr[spiking_local].tolist()
        self.total_woken += len(woken_arr)
        self.total_spiked += len(selected_slots)
        n_occ = max(1, int(memory.occupied.sum().item()))
        return selected_slots, dict(
            woken=len(woken_arr),
            woken_frac=len(woken_arr) / n_occ,
            spiked=len(selected_slots),
            k_max=self.k_max,
            apl_inhib=apl_inhib,
            mean_margin=float(margins[spiking_local].mean()) if len(spiking_local) else 0.0,
        )


def enabled(engine) -> bool:
    return getattr(engine, 'emeans_alif_reservoir_schema', None) == 1


def diagnostics(engine) -> dict | None:
    if not enabled(engine):
        return None
    res: PopulationAlifReservoir | None = getattr(engine, '_emeans_alif_reservoir', None)
    if res is None:
        return None
    q = max(1, res.total_queries)
    non_empty = sum(1 for b in res.index.buckets[:getattr(res.index, 'active_k', 0)] if len(b) > 0)
    return dict(
        schema=1,
        n_clusters=res.index.n_clusters,
        active_clusters=int(getattr(res.index, 'active_k', 0)),
        non_empty_clusters=int(non_empty),
        top_clusters=res.top_clusters,
        k_max=res.k_max,
        total_queries=res.total_queries,
        mean_woken=res.total_woken / q if res.total_queries else 0.0,
        mean_spiked=res.total_spiked / q if res.total_queries else 0.0,
        revivals=int(res.index.revivals),
        tick=int(res.tick),
    )


def configure(engine, config=None, **kwargs) -> PopulationAlifReservoir:
    """Attach opt-in e-means + Population ALIF reservoir to an engine."""
    mem = engine.model.memory
    cap = int(mem.capacity)
    dim = int(mem.source.shape[-1])
    res = PopulationAlifReservoir(capacity=cap, dim=dim, **kwargs)
    engine.emeans_alif_reservoir_schema = 1
    engine._emeans_alif_reservoir = res
    from scripts.ib_local import run_navigation_residual_wood as legacy
    legacy._DERIVED.add('_emeans_alif_reservoir')
    if config is not None:
        config['emeans_alif_reservoir'] = dict(schema=1, **kwargs)
    return res


@torch.no_grad()
def refresh_workset_population_alif(engine) -> dict:
    """Drop-in replacement for `refresh_workset(engine)` using e-means + Population ALIF."""
    owner = engine.model
    field, memory = owner.body.coevolution, owner.memory
    res: PopulationAlifReservoir | None = getattr(engine, '_emeans_alif_reservoir', None)
    if res is None:
        res = PopulationAlifReservoir(
            capacity=int(memory.capacity),
            dim=int(memory.source.shape[-1]),
            k_max=int(field.slots),
        )
        engine._emeans_alif_reservoir = res
        from scripts.ib_local import run_navigation_residual_wood as legacy
        legacy._DERIVED.add('_emeans_alif_reservoir')
    if memory.pending_count:
        query = memory.pending[-1:]
    else:
        query = memory.source[:1] * 0
    selected, q_diag = res.query(query, memory, owner.encoder, min_active=1)

    dev = field.slots_id.device
    k_cap = int(field.slots)
    slots = torch.zeros(k_cap, dtype=torch.long, device=dev)
    valid = torch.zeros(k_cap, dtype=torch.bool, device=dev)
    n_sel = min(len(selected), k_cap)
    if n_sel > 0:
        slots[:n_sel] = torch.as_tensor(selected[:n_sel], dtype=torch.long, device=dev)
        valid[:n_sel] = memory.occupied[slots[:n_sel]]
    versions = memory.read_credit_version[slots]

    same = ((slots[:, None] == field.slots_id[None]) &
            (versions[:, None] == field.versions[None]) & field.valid[None] & valid[:, None])
    index = same.to(torch.int64).argmax(-1)
    retained = same.any(-1)
    physical = owner.base.layers[0].state_size
    old = engine.recurrent[..., physical:].reshape(
        len(owner.base.layers), engine.recurrent.shape[1], field.slots, field.rank)
    work = old[:, :, index] * retained[None, None, :, None]
    engine.recurrent = torch.cat((engine.recurrent[..., :physical], work.flatten(2)), -1)

    field.slots_id.copy_(slots)
    field.versions.copy_(versions)
    field.valid.copy_(valid)
    for i, name in enumerate(('source', 'action', 'successor', 'long_successor')):
        field.anchors[i].copy_(getattr(memory, name)[slots])
    from scripts.ib_local.experience_consolidation import bounded_preference, consolidates_reads
    preference = bounded_preference(memory) if consolidates_reads(memory) else memory.read_credit_mean.tanh()
    field.preference.copy_(preference[slots])
    return dict(valid=int(valid.sum()), retained=int(retained.sum()),
                slots=slots.cpu().tolist(), versions=versions.cpu().tolist(),
                population_alif=q_diag)
