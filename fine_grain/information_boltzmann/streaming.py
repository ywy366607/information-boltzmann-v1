"""One causal predict/observe interface for online training and deployment."""
import math
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from .model import InformationBoltzmann
from .state import PhaseState


class StreamRunner:
    def __init__(self, model: InformationBoltzmann, bos_token: int, seed: int = 11,
                 optimizer: torch.optim.Optimizer | None = None,
                 update_every: int = 16, gradient_clip: float = 1.0,
                 boot_tokens: Tensor | None = None):
        if update_every < 1:
            raise ValueError("update_every must be positive")
        self.model, self.optimizer = model, optimizer
        self.update_every, self.gradient_clip = update_every, gradient_clip
        device = next(model.parameters()).device
        self.generator = torch.Generator(device=device).manual_seed(seed)
        if boot_tokens is None:
            boot_tokens = torch.tensor([bos_token], device=device)
        else:
            boot_tokens = boot_tokens.to(device=device, dtype=torch.long)
        if boot_tokens.ndim != 1 or boot_tokens.numel() == 0:
            raise ValueError("boot_tokens must be a nonempty one-dimensional input context")
        if not bool(((0 <= boot_tokens) & (boot_tokens < model.vocab_size)).all()):
            raise ValueError("boot_tokens contain a token outside the vocabulary")
        # F_theta conditions f_0 on the supplied stream-start context exactly once.
        # Later observations only enter the continuous dynamics; they never reset f.
        self.state = model.initialize(boot_tokens, self.generator)
        if optimizer is None:
            self.state = self.state.detach()
        self.current_token = int(boot_tokens[-1])
        self.pending: Tensor | None = None
        self.prefix_log_prob = self.state.x.new_zeros(())
        self.loss_terms: list[Tensor] = []
        self.events, self.updates, self.total_nll = 0, 0, 0.0
        self.candidates, self.accepted = 0, 0
        self.cross_moment_abs_sum = 0.0
        self.last_gradient_norm = 0.0

    def predict(self, budget: dict | None = None) -> Tensor:
        if self.pending is not None:
            raise RuntimeError("Observe the pending prediction before predicting again")
        with torch.set_grad_enabled(self.optimizer is not None):
            if budget is None:
                self.state, log_prob, stats = self.model.advance(self.state, self.current_token, self.generator)
            else:
                if self.model.adaptive_gamma:
                    raise RuntimeError("Unvalidated criticality controller disabled")
                self.state, log_prob, stats = self.model._advance_steps(
                    self.state, self.current_token, self.generator, budget=budget)
            self.prefix_log_prob = self.prefix_log_prob + log_prob
            self.pending = self.model.decode(self.state)
        self.candidates += stats["candidates"]
        self.accepted += stats["accepted"]
        self.cross_moment_abs_sum += abs(stats["cross_moment_change"])
        self.last_collision_pairs = stats.get("pairs", [])
        return self.pending.detach().clone()  # callers cannot mutate the training graph

    def observe(self, token: int, *, external: bool = True) -> float:
        if self.pending is None:
            raise RuntimeError("Must predict before observing a target")
        if not 0 <= token < self.model.vocab_size:
            raise ValueError("Token outside vocabulary")
        if self.optimizer is not None and not external:
            raise ValueError("Self-generated tokens are not online supervision")
        target = torch.tensor([token], device=self.pending.device)
        ce = F.cross_entropy(self.pending[None], target)
        value = float(ce.detach())
        if not math.isfinite(value):
            raise FloatingPointError("Nonfinite prequential loss")
        if self.optimizer is not None:
            # Baseline is constant and independent of the sampled collision path.
            # Prefix score includes exactly those events causally preceding this CE.
            score_term = (ce.detach() - math.log(self.model.vocab_size)) * self.prefix_log_prob
            self.loss_terms.append(ce + score_term)
        self.total_nll += value
        self.events += 1
        self.current_token = token
        self.pending = None
        if len(self.loss_terms) == self.update_every:
            self.flush()
        return value

    def flush(self) -> None:
        """Finalize a partial learning window at an explicit stream stop only."""
        if self.pending is not None:
            raise RuntimeError("Cannot update with an unobserved pending prediction")
        if self.loss_terms:
            self.optimizer.zero_grad(set_to_none=True)
            torch.stack(self.loss_terms).mean().backward()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip,
                                                 error_if_nonfinite=True)
            self.last_gradient_norm = float(norm)
            self.optimizer.step()
            self.updates += 1
        self.loss_terms = []
        self.state = self.state.detach()  # preserves phase values, never initializes again
        self.prefix_log_prob = self.state.x.new_zeros(())

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        if self.pending is not None or self.loss_terms:
            raise RuntimeError("Checkpoint at an observed update boundary; do not silently reset the window")
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": None if self.optimizer is None else self.optimizer.state_dict(),
            "state": {"x": self.state.x, "v": self.state.v, "time": self.state.time},
            "force_gamma": getattr(self.model.force, "gamma", None),
            "force_kappa": getattr(self.model.force, "kappa", None),
            "generator": self.generator.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "current_token": self.current_token, "events": self.events, "updates": self.updates,
            "total_nll": self.total_nll, "candidates": self.candidates, "accepted": self.accepted,
            "cross_moment_abs_sum": self.cross_moment_abs_sum,
            "update_every": self.update_every, "gradient_clip": self.gradient_clip,
            "metadata": metadata or {},
        }, path)

    def load(self, path: str | Path) -> dict:
        if self.pending is not None or self.loss_terms:
            raise RuntimeError("Cannot load over a partially observed window")
        device = next(self.model.parameters()).device
        saved = torch.load(path, map_location=device, weights_only=True)
        if saved["update_every"] != self.update_every or saved["gradient_clip"] != self.gradient_clip:
            raise ValueError("Checkpoint online update settings differ")
        if (saved["optimizer"] is None) != (self.optimizer is None):
            raise ValueError("Checkpoint learning mode differs")
        self.model.load_state_dict(saved["model"])
        if saved.get("force_gamma") is not None and hasattr(self.model.force, "gamma"):
            self.model.force.gamma = float(saved["force_gamma"])
        if saved.get("force_kappa") is not None and hasattr(self.model.force, "kappa"):
            self.model.force.kappa = float(saved["force_kappa"])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(saved["optimizer"])
        self.state = PhaseState(**saved["state"])
        self.generator.set_state(saved["generator"].cpu())
        torch.set_rng_state(saved["torch_rng"].cpu())
        if saved["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
        for name in ("current_token", "events", "updates", "total_nll", "candidates", "accepted", "cross_moment_abs_sum"):
            setattr(self, name, saved[name])
        self.prefix_log_prob = self.state.x.new_zeros(())
        return saved["metadata"]
