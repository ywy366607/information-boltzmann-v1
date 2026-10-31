"""Automated experimental harness for small CGB lifeforms on real byte streams.

Provides:
1. Empirical Unigram and Bigram baselines on real data;
2. Strict [Min | Median | Mean | Max] statistical distributions;
3. Four Paired Controls:
   - Nominal (Active Learning) vs Frozen Learning (Plasticity check)
   - History Disrupted (Causal state retention check)
   - Information Cut (Active query / rental consequence check)
   - Resource Cut (Starvation and absorbing failure check)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from typing import Any, Callable

import numpy as np

try:
    from .cgb_low_resource_life import Observation, PrefixReplayWorld
    from .cgb_prequential_bridge import CGBPrequentialBridge
    from .cgb_information_ledger import PredictiveInformationLedger
except ImportError:
    from cgb_low_resource_life import Observation, PrefixReplayWorld
    from cgb_prequential_bridge import CGBPrequentialBridge
    from cgb_information_ledger import PredictiveInformationLedger

LN2 = math.log(2.0)


def format_stats(values: list[float] | tuple[float, ...]) -> dict[str, Any]:
    """Format numeric sequence as exact statistics with [Min | Median | Mean | Max]."""
    if not values:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0, "display": "[0.0000 | 0.0000 | 0.0000 | 0.0000]"}
    vals = [float(v) for v in values]
    mn, mx = min(vals), max(vals)
    med = statistics.median(vals)
    avg = statistics.mean(vals)
    display = f"[{mn:.4f} | {med:.4f} | {avg:.4f} | {mx:.4f}]"
    return {"min": mn, "median": med, "mean": avg, "max": mx, "display": display}


class EmpiricalBaselines:
    """Precomputed train-only Unigram and Bigram baselines for native byte streams."""

    def __init__(self, train_bytes: bytes | np.ndarray, vocab_size: int = 256, alpha: float = 0.5):
        if vocab_size != 256:
            raise ValueError("native byte alphabet size must be 256")
        if alpha <= 0.0 or not math.isfinite(alpha):
            raise ValueError("smoothing alpha must be finite and positive")
        data = np.frombuffer(train_bytes, dtype=np.uint8) if isinstance(train_bytes, (bytes, bytearray)) else train_bytes
        self.vocab_size = vocab_size
        self.alpha = float(alpha)
        # Unigram counts
        unigram_counts = np.bincount(data, minlength=vocab_size).astype(np.float64)
        total_tokens = float(len(data))
        self.unigram_probs = (unigram_counts + alpha) / (total_tokens + alpha * vocab_size)
        self.unigram_nlls = -np.log(self.unigram_probs)

        # Bigram transition counts: C[w_{t-1}, w_t]
        if len(data) > 1:
            prev = data[:-1].astype(np.int64)
            curr = data[1:].astype(np.int64)
            flat_indices = prev * vocab_size + curr
            bigram_counts = np.bincount(flat_indices, minlength=vocab_size * vocab_size).reshape(vocab_size, vocab_size).astype(np.float64)
            row_sums = bigram_counts.sum(axis=1, keepdims=True)
            self.bigram_probs = (bigram_counts + alpha) / (row_sums + alpha * vocab_size)
            self.bigram_nlls = -np.log(self.bigram_probs)
        else:
            self.bigram_nlls = np.tile(self.unigram_nlls, (vocab_size, 1))

    def unigram_nll(self, token: int) -> float:
        return float(self.unigram_nlls[token])

    def unigram_bits(self, token: int) -> float:
        return float(self.unigram_nlls[token]) / LN2

    def bigram_nll(self, prev: int | None, token: int) -> float:
        if prev is None:
            return self.unigram_nll(token)
        return float(self.bigram_nlls[prev, token])

    def bigram_bits(self, prev: int | None, token: int) -> float:
        return self.bigram_nll(prev, token) / LN2


@dataclass(frozen=True)
class StepLog:
    tick: int
    target: int
    model_nll_nats: float
    model_bits: float
    unigram_bits: float
    bigram_bits: float
    gain_over_unigram_bits: float
    gain_over_bigram_bits: float
    dissipation_units: int
    active: bool


def run_trial(
    world_factory: Callable[[], PrefixReplayWorld],
    learner_factory: Callable[[], Any],
    baselines: EmpiricalBaselines,
    events: int,
    *,
    control_mode: str = "nominal",
    request_interval: int = 16,
    request_lag: int = 0,
    request_length: int = 4,
    disrupt_interval: int = 8,
) -> dict[str, Any]:
    """Execute a single evaluation trial under one specific experimental condition.

    Supported control_mode:
    - 'nominal': plastic learning enabled, normal world and receipt
    - 'frozen': plasticity disabled (learning=False)
    - 'history_disrupted': internal history/carrier reset periodically
    - 'info_cut': receipts returned from world are masked with zeros
    - 'resource_cut': world supply is cut to zero, leading to starvation
    """
    valid_modes = {"nominal", "frozen", "history_disrupted", "info_cut", "resource_cut"}
    if control_mode not in valid_modes:
        raise ValueError(f"unknown control mode: {control_mode}")

    world = world_factory()
    learner = learner_factory()
    if control_mode == "frozen":
        if hasattr(learner, "learning"):
            learner.learning = False

    ledger = PredictiveInformationLedger(reference_name="train_unigram")
    bridge = CGBPrequentialBridge(learner, ledger, baselines.unigram_nll)

    step_logs: list[StepLog] = []
    prev_token: int | None = None
    fallback_uniform_nll = math.log(256)
    fallback_uniform_bits = fallback_uniform_nll / LN2

    for tick in range(events):
        before_counts = world.state_dict()["state"]["counts"]
        frame = world.begin()

        # Handle history disruption
        if control_mode == "history_disrupted" and (tick % disrupt_interval == 0):
            if hasattr(learner, "reset_history"):
                learner.reset_history()
            elif hasattr(learner, "context"):
                # Neutralize context to neutral zeros without corrupting parameter shapes
                learner.context = tuple(0 for _ in learner.context)

        # Handle auxiliary observation
        auxiliary = tuple(frame.receipt or ())
        if control_mode == "info_cut" and auxiliary:
            auxiliary = tuple(0 for _ in auxiliary)

        if frame.active:
            bridge.prepare(auxiliary=auxiliary)

        # Publish the event
        event = world.publish()
        token = event.payload

        # Action: Request external prefix replay if active
        if frame.active and (tick % request_interval == 3):
            world.request(request_lag, request_length)

        after_counts = world.state_dict()["state"]["counts"]
        paid = sum(after_counts[k] - before_counts[k] for k in world.COSTS)

        # Baseline predictions
        u_bits = baselines.unigram_bits(token)
        b_bits = baselines.bigram_bits(prev_token, token)

        if frame.active:
            res = bridge.step(event, write_units=0, dissipation_units=paid, auxiliary=auxiliary)
            m_nll = res.model_nll_nats
            m_bits = m_nll / LN2
        else:
            # Absorbing failure fallback
            m_nll = fallback_uniform_nll
            m_bits = fallback_uniform_bits
            ledger.record(tick=tick, model_nll_nats=m_nll, reference_nll_nats=baselines.unigram_nll(token),
                          write_units=0, dissipation_units=paid)

        log_entry = StepLog(
            tick=tick,
            target=token,
            model_nll_nats=m_nll,
            model_bits=m_bits,
            unigram_bits=u_bits,
            bigram_bits=b_bits,
            gain_over_unigram_bits=u_bits - m_bits,
            gain_over_bigram_bits=b_bits - m_bits,
            dissipation_units=paid,
            active=frame.active,
        )
        step_logs.append(log_entry)
        prev_token = token

    # Aggregate distribution statistics
    m_bits_seq = [s.model_bits for s in step_logs]
    u_gain_seq = [s.gain_over_unigram_bits for s in step_logs]
    b_gain_seq = [s.gain_over_bigram_bits for s in step_logs]
    dissipation_total = sum(s.dissipation_units for s in step_logs)
    active_ticks = sum(1 for s in step_logs if s.active)

    total_useful_bits = sum(max(0.0, g) for g in u_gain_seq)
    dissipation_per_useful_bit = (dissipation_total / total_useful_bits) if total_useful_bits > 0 else None

    return {
        "control_mode": control_mode,
        "total_ticks": events,
        "active_ticks": active_ticks,
        "survival_rate": active_ticks / float(events),
        "dissipation_units_total": dissipation_total,
        "dissipation_per_useful_bit": dissipation_per_useful_bit,
        "model_bits_per_byte": format_stats(m_bits_seq),
        "gain_over_unigram_bits": format_stats(u_gain_seq),
        "gain_over_bigram_bits": format_stats(b_gain_seq),
        "final_reserve": world.state_dict()["state"]["reserve"],
    }


def run_four_paired_controls(
    learner_factory: Callable[[], Any],
    source_bytes: bytes,
    train_reference_bytes: bytes,
    events: int = 256,
    *,
    reserve: int = 32,
    capacity: int = 64,
    nominal_supply: int = 2,
    delay: int = 3,
    max_block: int = 4,
    request_fee: int = 2,
    holding_fee: int = 1,
    read_per_byte: int = 1,
    maintenance: int = 1,
) -> dict[str, Any]:
    """Execute all four paired controls on identical stream and return contrasting metrics."""
    baselines = EmpiricalBaselines(train_reference_bytes)

    def make_world(supply: int):
        return PrefixReplayWorld(
            source_bytes,
            source_id="harness-stream",
            reserve=reserve,
            capacity=capacity,
            maintenance=maintenance,
            request_fee=request_fee,
            holding_fee=holding_fee,
            read_per_byte=read_per_byte,
            delay=delay,
            max_block=max_block,
            supply_per_tick=supply,
        )

    nominal_res = run_trial(lambda: make_world(nominal_supply), learner_factory, baselines, events, control_mode="nominal")
    frozen_res = run_trial(lambda: make_world(nominal_supply), learner_factory, baselines, events, control_mode="frozen")
    history_res = run_trial(lambda: make_world(nominal_supply), learner_factory, baselines, events, control_mode="history_disrupted")
    info_cut_res = run_trial(lambda: make_world(nominal_supply), learner_factory, baselines, events, control_mode="info_cut")
    resource_cut_res = run_trial(lambda: make_world(0), learner_factory, baselines, events, control_mode="resource_cut")

    # Compute pairwise net differences (Positive = Advantage for nominal)
    nom_mean_bits = nominal_res["model_bits_per_byte"]["mean"]
    froz_mean_bits = frozen_res["model_bits_per_byte"]["mean"]
    hist_mean_bits = history_res["model_bits_per_byte"]["mean"]
    info_mean_bits = info_cut_res["model_bits_per_byte"]["mean"]
    res_mean_bits = resource_cut_res["model_bits_per_byte"]["mean"]

    pairwise_contrasts = {
        "plasticity_net_gain_bits_per_byte": froz_mean_bits - nom_mean_bits,
        "history_retention_net_gain_bits_per_byte": hist_mean_bits - nom_mean_bits,
        "action_information_net_gain_bits_per_byte": info_mean_bits - nom_mean_bits,
        "resource_cut_starvation_ticks": resource_cut_res["active_ticks"],
        "starvation_penalty_bits_per_byte": res_mean_bits - nom_mean_bits,
    }

    return {
        "events": events,
        "nominal": nominal_res,
        "frozen_learning": frozen_res,
        "history_disrupted": history_res,
        "information_cut": info_cut_res,
        "resource_cut": resource_cut_res,
        "pairwise_contrasts": pairwise_contrasts,
    }
