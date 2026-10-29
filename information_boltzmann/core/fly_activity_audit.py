"""Bounded per-cell activity diagnostics, separate from learning and capability claims."""
import torch


@torch.no_grad()
def instantaneous_release_metrics(release, graded_mask=None):
    """Keep continuous release distinct from transmitted spike events.

    Rates are fractions of cells at the current tick, not Hz. The legacy
    firing_rate denominator stays the whole population; a second field gives
    the active fraction within spiking cells only.
    """
    graded = (torch.zeros_like(release, dtype=torch.bool) if graded_mask is None
              else graded_mask.to(device=release.device, dtype=torch.bool).expand_as(release))
    active = release > 0
    spikes = active & ~graded
    spiking_count = int((~graded).sum())
    return {'firing_rate': float(spikes.float().mean()),
            'transmission_active_fraction': float(active.float().mean()),
            'spiking_cell_fraction_active': float(spikes.sum()) / max(1, spiking_count),
            'graded_cell_fraction': float(graded.float().mean()),
            'activity_metric_version': 2}


class CellActivityAudit:
    def __init__(self, like, graded_mask=None, refractory_ticks=0):
        n = like.shape[-1]
        self.graded = (torch.zeros(n, dtype=torch.bool, device=like.device) if graded_mask is None
                       else graded_mask.reshape(-1).to(device=like.device, dtype=torch.bool))
        self.count = torch.zeros(n, dtype=torch.int32, device=like.device)
        self.run = torch.zeros_like(self.count)
        self.longest_run = torch.zeros_like(self.count)
        self.release_sum = like.new_zeros(n)
        self.release_square_sum = like.new_zeros(n)
        self.ticks = 0
        self.rate_ceiling = 1.0 / (1 + int(refractory_ticks))

    @torch.no_grad()
    def observe(self, release):
        value = release.reshape(-1).detach()
        firing = (value > 0) & ~self.graded
        self.count.add_(firing)
        self.run = torch.where(firing, self.run + 1, 0)
        self.longest_run = torch.maximum(self.longest_run, self.run)
        self.release_sum.add_(value)
        self.release_square_sum.add_(value.square())
        self.ticks += 1

    def report(self, groups):
        """Always-spiking is an observation over this duration, not proof of a pathological attractor."""
        n = max(1, self.ticks)
        mean = self.release_sum / n
        sd = (self.release_square_sum / n - mean.square()).clamp_min(0).sqrt()
        report = {'ticks': self.ticks, 'unit': 'transmitted release; graded cells excluded from spike counts',
                  'spike_rate_ceiling_per_tick': self.rate_ceiling,
                  'near_ceiling_threshold_fraction': 0.9,
                  'groups': {}}
        for name, indices in groups.items():
            index = torch.as_tensor(indices, device=mean.device, dtype=torch.long)
            if not index.numel():
                continue
            spiking = index[~self.graded[index]]
            count = self.count[spiking].float()
            runs = self.longest_run[spiking].float()
            graded = index[self.graded[index]]
            report['groups'][name] = {
                'cells': int(index.numel()), 'spiking_cells': int(spiking.numel()), 'graded_cells': int(graded.numel()),
                'always_spiking_fraction': float((count == self.ticks).float().mean()) if spiking.numel() else None,
                'never_spiking_fraction': float((count == 0).float().mean()) if spiking.numel() else None,
                'near_ceiling_fraction': float((count / n >= 0.9 * self.rate_ceiling).float().mean())
                    if spiking.numel() else None,
                'spike_rate_mean_per_tick': float((count / n).mean()) if spiking.numel() else None,
                'spike_rate_max_per_tick': float((count / n).max()) if spiking.numel() else None,
                'longest_spike_run_max': int(runs.max()) if spiking.numel() else None,
                'longest_spike_run_median': float(runs.median()) if spiking.numel() else None,
                'release_mean': float(mean[index].mean()), 'release_temporal_sd_mean': float(sd[index].mean()),
                'graded_release_temporal_sd_mean': float(sd[graded].mean()) if graded.numel() else None,
            }
        return report
