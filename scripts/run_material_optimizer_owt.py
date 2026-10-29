"""Registered GPU optimizer experiment on cached OpenWebText bytes.

The ordinary causal language model is a controlled testbed, not the CGB
generated-brain architecture. Preparation reads real documents but executes
no model. Training requires a source-bound, scoped Lean/interface receipt.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_REGISTRATION = ROOT / "results/published/optimizer_continuous_prereg_20261028.json"
RUNTIME = "information_boltzmann/conservative_cycle_optimizer.py"
REVIEW = "lean/CGBAxioms/reviews/optimizer_conservative_cycle.lean"
RUNNER = "scripts/run_material_optimizer_owt.py"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(8 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def document_bounds(tokens):
    """The cache stores text,EOS,BOS for each document (including the last)."""
    ends = np.flatnonzero(tokens == 2)
    starts = np.r_[0, ends[:-1] + 2]
    if not len(ends) or ends[-1] + 2 != len(tokens) or not np.all(tokens[ends + 1] == 1):
        raise ValueError("unexpected cached document delimiters")
    return list(zip(starts.tolist(), ends.tolist()))


def prepare(registration, resource=False, feedback_control=False, matrix_comparison=False, auto_step_comparison=False, adam_cycle_comparison=False, matched_rate_comparison=False, latch_demo=False):
    """Reuse the verified real-data streams; register only the interface repair."""
    from copy import deepcopy
    previous = json.loads((ROOT / "results/published/optimizer_owt_prereg_20261028.json").read_text())
    record = deepcopy(previous)
    record.update(version=2, scope="continuous-parameter optimizer comparison", parameter_representation="continuous_fp32",
        question="Does the existing compact cycle optimizer learn on unchanged FP32 parameters at acceptable cost?",
        status="registered_pending_interface_preflight")
    paths = (RUNTIME, "information_boltzmann/conservative_cycle_cuda.py", REVIEW, RUNNER)
    record["sources"] = {p: sha(ROOT / p) for p in paths}
    record["lean_scope"] = dict(full_life_admitted=False,
        reference_receipt="results/published/optimizer_cycle_integer_reference_20261027.json",
        contracts=["unrestricted_parameter_initialization", "integer_joint_optimizer_step", "finite_block_optimizer_step", "functional_recovery_tracking"],
        applicability="same existing phase/resource law and encoded transition; independent disjoint coordinate blocks",
        error_scope="integer state exact; FP32 cue/write arithmetic remains finite precision, checked against reference",
        conditional_boundary="smooth/PL and gradient error premises are conditional, not asserted for arbitrary language losses",
        experimental_scope="fixed-budget learning/cost comparison only; useful lifetime retention and universal superiority remain open")
    exp = record["experiment"]
    exp.update(arms=["cycle", "adam"], initialization="same scratch Gaussian FP32 values; no projection, clipping, quantization or weight rewrite at optimizer construction")
    exp.pop("material")
    exp["cycle"] = dict(lr=0.1, fused=True, law="existing integerCycleStep; two int16 bytes per four coordinates", tuning="fixed initial rate; not claimed optimal, no hyperparameter sweep")
    record["measurements"]["interpretation"] = "same unconstrained FP32 parameter family and exact same initialization; only optimizer changes; rates fixed, not assumed equally tuned"
    record["decision_rules"] = dict(positive="learning and independent validation improvement at recorded cost; compare package quality/cost on identical parameter representation",
        negative="failure to improve or excess cost rejects the present algorithm/rate for this benchmark",
        ambiguous="one seed, fixed untuned rates and nonconvergence do not settle general optimizer superiority",
        do_not_claim=["healthy lifetime locking from a language curve", "weight self-reconstruction at zero gradient", "global functional diversity", "1B capability", "optimal hyperparameters"])
    if resource:
        record.update(version=3, scope="resource-observer continuous optimizer comparison",
                      question="Does the Lean-refined online direction/scale/phase update improve the earlier raw-gradient cycle at matched real-data exposure?",
                      interface_receipt="lean/CGBAxioms/.lake/resource_optimizer_tests.check.json")
        record["experiment"]["arms"] = ["resource", "adam"]
        record["experiment"].pop("cycle")
        record["experiment"]["resource"] = dict(lr=.001, observer_rate=.125, chunk_compartments=1024)
        record["lean_scope"].update(
            reference_receipt="results/published/optimizer_raw_observation_bridge_20261028.json",
            contracts=["integer_preconditioned_step", "resource_filtered_tracking", "raw_stream_forces_release",
                       "relative_cue_shared_scale", "integer_ignition_and_retention"],
            applicability="same online observer and integer transition; compartment-shared RMS; tail uses a smaller scalar readout rate",
            conditional_boundary="persistent constant contrary direction, exact arithmetic and smooth-model/error premises are scoped; functional retention and calibrated curvature are not assumed for OWT")
        record["construction_H"] = dict(
            observation_clock="rate times old available resource, normalized exposure startup",
            direction_and_scale="same-gain mean and empirical RMS; neither Bayes calibration nor Hessian estimation",
            compartment="64 contiguous slots with implicit cycles; zero-padded tail has RMS dilution corrected in readout",
            nominal_rates="resource lr .001, observer .125 fixed before results; Adam .0003 inherited; no tuning or optimality claim",
            missing_gradient="no observation; explicit zero gradient advances state")
    if feedback_control:
        record.update(version=4, scope="matched live-state feedback intervention",
                      question="What is the effect of regenerative coupling when state layout, observer, precision, data and nominal rates are held fixed?",
                      interface_receipt="lean/CGBAxioms/.lake/feedback_control_tests.check.json")
        exp["arms"] = ["feedback_on", "feedback_off"]
        record["lean_scope"].update(
            reference_receipt="results/published/optimizer_feedback_control_interface_20261028.json",
            contracts=["controlled_cycle_contract", "passive_quiet_quantization_band", "controlled_final_readout",
                       "resource_filtered_tracking", "relative_cue_shared_scale"],
            applicability="both settings retain identical live state and the same conditional final-readout certificate; active-specific regeneration properties are not assigned to the passive control")
        record["intervention"] = dict(
            changed="phase numerator: 8*K+2*R*K+R*C versus 8*K+R*C, same +8 rounding and denominator16",
            held_fixed=["FP32 network parameters and exact initialization", "FP32 mean/RMS/exposure and INT16 phase/reserve",
                        "64-slot grouping", "observer, resource ledger and readout formulas", "nominal rates", "data, exposure and update budget"],
            buffers="all retained state is used in both arms; no dummy arrays",
            indirect_effects="phase changes future resource use, observation gain and gradients through the SAME closed-loop formulas; this estimates their total feedback-coupling effect",
            quantization="passive quiet phase reaches codes0..1 in six steps; ties may retain code1, so exact zero forgetting is not claimed")
        record["decision_rules"] = dict(
            primary="Report paired prequential BPB and final A/B validation differences, feedback_on minus feedback_off",
            positive="Lower prequential BPB and lower final mean A/B validation support this coupling on this stream/budget; not universal superiority or isolated proof of useful latching",
            negative="If feedback worsens both measures, withhold its usefulness claim and keep the shared observer as the reusable baseline; do not tune away this result",
            ambiguous="Opposite signs, small differences, one seed or unconverged curves imply limited or mixed evidence",
            stop="exactly5000 updates per arm, no early-success stopping or post-result rate adjustment",
            do_not_claim=["Adam state-only explanation", "all forms of latching help or fail", "global diversity", "convergence", "1B capability"])
    if matrix_comparison:
        record.update(version=5, scope="matrix resource cycle versus Adam and Muon",
                      question="Does real-matrix budget competition train the same byte model at comparable quality and cost?",
                      skip_lean_by_user_request=True,
                      interface_receipt="lean/CGBAxioms/.lake/matrix_resource_tests.check.json")
        exp['arms'] = ['resource', 'adam', 'muon']
        exp['resource'] = dict(lr=.001, observer_rate=.125, sinkhorn_steps=8)
        exp['muon'] = dict(lr=.0003, weight_decay=0, momentum=.95, nesterov=True,
                           ns_steps=5, adjust_lr_fn='match_rms_adamw')
        record['lean_scope'] = dict(full_life_admitted=False, skipped='Explicit user request: run numerical/interface checks without additional Lean gate')
        record['construction_H'] = dict(
            groups='each actual matrix; non-matrices viewed as one-row matrices',
            observer='per-parameter mean and RMS, matrix-level observation mass',
            allocation='8 Sinkhorn row/column scaling iterations; measure actual residual',
            resource='floating per-row and per-column use/recovery; matrix-valued phase',
            muon='torch.optim.Muon on all 2D tensors including embeddings and head, no weight decay',
            hyperparameters='fixed before runs, same data budget; no tuning or claim of equally tuned rates')
        record['decision_rules'] = dict(
            primary='report prequential BPB, independent A/B validation, budget residual, full-state bytes, peak GPU memory and wall time',
            interpretation='one seed, fixed rates and 5000 steps are a budget-limited comparison; no claim of convergence or universal superiority',
            no_claim='do not infer healthy functional latching or non-monopoly without controlled feedback ablation',
            stop='5000 updates per arm unless nonfinite values terminate a run')
    if auto_step_comparison:
        record.update(scope='matched ResourceCycle automatic-step intervention',
                      question='Does directional feedback improve continuous OWT learning at the same initialization and state budget?',
                      interface_receipt='lean/CGBAxioms/.lake/auto_step_tests.check.json')
        exp['arms'] = ['step_fixed', 'step_adaptive']
        exp['resource'] = dict(lr=.001, observer_rate=.125, sinkhorn_steps=8, auto_step=True)
        record['intervention'] = dict(changed='adaptive multipliers 1.05/.95 versus both fixed at 1',
            held_fixed='model, initialization, data, state layout, previous-update reconstruction and 5000 updates',
            step_feedback='fresh gradient alignment with previous allocated direction; no extra forward pass, not measured loss decrease',
            rate_sensitivity='single initial lr; no claim of reduced tuning from this comparison')
        record['decision_rules'] = dict(primary='paired prequential BPB and independent A/B validation, plus rho trajectory and full cost',
            interpretation='5000 updates, one seed and one initial rate; neither convergence nor reduced tuning is established',
            no_claim='no general healthy-latch, lifelong-learning or benefit attribution without independent controls',
            stop='5000 updates per arm unless nonfinite values terminate a run')
    if adam_cycle_comparison:
        record.update(scope='matched Adam-base ResourceCycle competition intervention',
                      question='Does ResourceCycle allocation improve an Adam-base update on continuous OWT?',
                      interface_receipt='lean/CGBAxioms/.lake/adam_cycle_tests.check.json')
        exp['arms'] = ['adam_cycle_control', 'adam_cycle']
        exp['resource'] = dict(lr=.0003, adam_base=True, betas=[.9, .999], eps=1e-8, sinkhorn_steps=8)
        record['intervention'] = dict(changed='competition_strength 0 versus 1 in the final Adam update',
            held_fixed='FP32 model, Adam moments, initialized state layout, data, no weight decay, 5000 updates',
            interpretation='allocation budgets apply to allocation not parameter updates; no independent proof of functional latch',
            tuning='same fixed Adam learning rate in both arms; no tuning claim')
        record['decision_rules'] = dict(primary='paired prequential BPB and independent A/B validation, complete state and runtime',
            interpretation='one seed and one learning rate; fixed budget, not evidence of convergence',
            no_claim='no proof of healthy functional latch or conserved parameter updates',
            stop='5000 updates per arm unless nonfinite values terminate a run')
    if matched_rate_comparison:
        record.update(scope='same nominal learning rate matrix ResourceCycle versus Adam-base ResourceCycle',
                      question='How do the two existing update rules compare at nominal lr 0.001?',
                      interface_receipt='lean/CGBAxioms/.lake/matched_rate_tests.check.json')
        exp['arms'] = ['resource', 'adam_cycle']
        exp['resource'] = dict(lr=.001, observer_rate=.125, sinkhorn_steps=8)
        exp['adam_cycle'] = dict(lr=.001, adam_base=True, betas=[.9, .999], eps=1e-8, sinkhorn_steps=8)
        exp['record_update_stats'] = True
        record['intervention'] = dict(held_fixed='FP32 initialization, model, OWT data, 5000 updates, nominal lr 0.001, weight decay 0, allocation iterations',
            changed='resource observer versus Adam bias-corrected moments and allocation cue',
            update_diagnostics='actual update RMS and relative L2 norm, sampled every 250 updates',
            caveat='same nominal lr does not imply same effective update or identical state dynamics')
        record['decision_rules'] = dict(primary='paired prequential BPB and independent A/B validation, sampled update sizes and full cost',
            interpretation='one seed, one rate and 5000 steps; no claim of optimality or convergence',
            stop='5000 updates per arm unless nonfinite values terminate a run')
    if latch_demo:
        record.update(scope='ResourceCycle active-restoration demo versus matched-state control',
                      question='Does the experimental restorative path help or impede a continuous OWT stream?',
                      interface_receipt='lean/CGBAxioms/.lake/latch_demo_tests.check.json')
        exp['arms'] = ['latch_control', 'latch_demo']
        exp['resource'] = dict(lr=.001, adam_base=True, betas=[.9, .999], eps=1e-8,
                               sinkhorn_steps=8, latch_state=True, latch_strength=1.0, anchor_rate=.05)
        exp['record_update_stats'] = True
        record['intervention'] = dict(changed='restorative strength 0 versus 1',
            held_fixed='model, Adam moments, competition, state layout including moving reference, OWT data and 5000 updates',
            interpretation='phase and resource modulate restoration; the reference follows gradient-driven displacement',
            limitation='gradient magnitude is not proof of usefulness, and restoration is not a closed physical resource ledger')
        record['decision_rules'] = dict(primary='paired prequential BPB and independent A/B validation, sampled physical state, time and memory',
            interpretation='one seed and fixed budget; numerical chain demo only, not proof of useful locking or convergence',
            stop='5000 updates per arm unless nonfinite values terminate a run')
    write_json(registration, record)
    print(json.dumps(dict(registration=str(registration), arms=exp["arms"], updates=exp["updates"], parameter_representation=record["parameter_representation"])), flush=True)


def load_registration(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("parameter_representation") != "continuous_fp32":
        raise ValueError("this repaired launcher requires unchanged continuous FP32 parameters")
    for source, digest in record["sources"].items():
        if sha(ROOT / source) != digest:
            raise ValueError("registered source changed: " + source)
    if not record.get('skip_lean_by_user_request', False):
        receipt = json.loads((ROOT / record["lean_scope"]["reference_receipt"]).read_text())
        proof_hash = receipt.get("proof_sha256", receipt.get("sources", {}).get(REVIEW))
        names = {name.rsplit(".", 1)[-1] for name in receipt.get("named_lemmas_audited", receipt.get("audited_declarations", []))}
        if (receipt["lean_exit_code"] != 0 or proof_hash != sha(ROOT / REVIEW)
                or not set(record["lean_scope"]["contracts"]).issubset(names)):
            raise ValueError("current applicable Lean contracts required")
    tests = json.loads((ROOT / record.get("interface_receipt", "lean/CGBAxioms/.lake/continuous_cycle_tests.check.json")).read_text())
    if tests["returncode"] != 0 or tests["passed"] < 15:
        raise ValueError("continuous-parameter CPU/GPU interface tests required")
    for source, digest in tests["sources"].items():
        if sha(ROOT / source) != digest:
            raise ValueError("current CUDA interface pass required")
    for stream in record["data"]["streams"].values():
        if sha(ROOT / stream["path"]) != stream["sha256"]:
            raise ValueError("registered exposure stream changed")
    return record


def torch_setup(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return torch


def make_model(record):
    import torch
    from torch import nn
    from torch.nn import functional as F
    spec = record["model"]
    width, heads = spec["width"], spec["heads"]

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(width, 3 * width, bias=False)
            self.out = nn.Linear(width, width, bias=False)
            self.ff1 = nn.Linear(width, spec["ff_width"], bias=False)
            self.ff2 = nn.Linear(spec["ff_width"], width, bias=False)

        def forward(self, x):
            b, length, _ = x.shape
            q, k, v = self.qkv(F.layer_norm(x, (width,))).chunk(3, dim=-1)
            q, k, v = [t.view(b, length, heads, width // heads).transpose(1, 2) for t in (q, k, v)]
            # Explicit deterministic causal attention for this control model.
            score = (q @ k.transpose(-1, -2)) / math.sqrt(width // heads)
            mask = torch.ones((length, length), device=x.device, dtype=torch.bool).triu(1)
            attention = score.masked_fill(mask, -torch.inf).softmax(-1) @ v
            x = x + self.out(attention.transpose(1, 2).reshape(b, length, width))
            return x + self.ff2(F.gelu(self.ff1(F.layer_norm(x, (width,)))))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(spec["vocabulary"], width)
            self.blocks = nn.ModuleList([Block() for _ in range(spec["layers"])])
            self.head = nn.Linear(width, spec["vocabulary"], bias=False)
            position = torch.arange(spec["context"])[:, None]
            frequency = torch.exp(torch.arange(0, width, 2) * (-math.log(10000) / width))
            code = torch.empty(spec["context"], width)
            code[:, 0::2], code[:, 1::2] = torch.sin(position * frequency), torch.cos(position * frequency)
            self.register_buffer("position", code)
            for parameter in self.parameters():
                nn.init.normal_(parameter, std=width ** -0.5)

        def forward(self, x):
            hidden = self.embedding(x) + self.position[:x.shape[1]]
            for block in self.blocks:
                hidden = block(hidden)
            return self.head(F.layer_norm(hidden, (width,)))

    model = Model().cuda()
    return model


def make_optimizer(model, record, arm):
    import torch
    spec = record["experiment"]
    if arm == "adam":
        return torch.optim.Adam(model.parameters(), **spec["adam"], foreach=False)
    if arm == "muon":
        parameters = list(model.parameters())
        if any(p.ndim != 2 for p in parameters):
            raise ValueError('this Muon control requires every parameter to be a matrix')
        return torch.optim.Muon(parameters, **spec['muon'])
    if arm in ("resource", "feedback_on", "feedback_off", "step_fixed", "step_adaptive", "adam_cycle_control", "adam_cycle", "latch_control", "latch_demo", "causal_allocation_off", "causal_phase_off", "fatigue_on", "fatigue_off", "outcome_on", "outcome_off"):
        from information_boltzmann.conservative_cycle_optimizer import ResourceCycle
        options = dict(spec.get('adam_cycle', spec['resource']) if arm in ('adam_cycle', 'adam_cycle_control') else spec['resource'])
        if arm in ("feedback_on", "feedback_off"):
            options["feedback"] = arm == "feedback_on"
        if arm == 'step_fixed':
            options.update(gamma_up=1.0, gamma_down=1.0)
        if arm == 'adam_cycle_control':
            options['competition_strength'] = 0.0
        if arm == 'latch_control':
            options['latch_strength'] = 0.0
        if arm == 'causal_allocation_off':
            options['competition_strength'] = 0.0
        if arm == 'causal_phase_off':
            options['feedback'] = False
        if arm == 'fatigue_off':
            options['resource_fatigue'] = False
        if arm == 'fatigue_on':
            options['resource_fatigue'] = True
        if arm in ('outcome_on', 'outcome_off'):
            options['outcome_modulation'] = arm == 'outcome_on'
        return ResourceCycle(model.parameters(), **options)
    if arm != "cycle":
        raise ValueError("unknown continuous optimizer")
    from information_boltzmann.conservative_cycle_optimizer import ConservativeCycle
    return ConservativeCycle(model.parameters(), lr=spec["cycle"]["lr"], fused=spec["cycle"]["fused"])


def batch_from(stream, cursor, batch, context):
    import torch
    size = batch * context
    if cursor + size >= len(stream):
        raise ValueError("fresh exposure exhausted; refusing silent wrap/replay")
    x = torch.from_numpy(np.array(stream[cursor:cursor + size], dtype=np.int64)).view(batch, context).cuda()
    y = torch.from_numpy(np.array(stream[cursor + 1:cursor + size + 1], dtype=np.int64)).view(batch, context).cuda()
    return x, y


def batch_mean_loss(model, x, y):
    import torch
    with torch.no_grad():
        return float(torch.nn.functional.cross_entropy(
            model(x).flatten(0, 1), y.flatten(), reduction='mean').item())


def step(model, optimizer, x, y, record_update=False):
    import torch
    optimizer.zero_grad(set_to_none=False)
    losses = torch.nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten(), reduction="none")
    if not torch.isfinite(losses).all().item():
        raise FloatingPointError("nonfinite predictive loss")
    bytemask = y.flatten() >= 4
    stats = dict(nll=float(losses.sum().item()), targets=y.numel(),
        byte_nll=float(losses[bytemask].sum().item()), bytes=int(bytemask.sum().item()))
    losses.mean().backward()
    before = [p.detach().clone() for p in model.parameters()] if record_update else None
    optimizer.step()
    if hasattr(optimizer, 'param_groups'):
        predicted = sum(float(group.get('last_pred_decrease', 0.0))
                        for group in optimizer.param_groups)
        stats['predicted_decrease'] = predicted
        outcome_groups = [group for group in optimizer.param_groups
                          if group.get('outcome_modulation', False)]
        if outcome_groups:
            stats['outcome_rho'] = float(outcome_groups[0].get('rho', 1.0))
    if before is not None:
        with torch.no_grad():
            parameters = list(model.parameters())
            update_sq = sum(((p.detach() - old).double().square().sum() for p, old in zip(parameters, before)))
            weight_sq = sum((old.double().square().sum() for old in before))
            n = sum(p.numel() for p in parameters)
            magnitude = float(update_sq.sqrt())
            stats['update_rms'] = magnitude / math.sqrt(n)
            stats['relative_update_l2'] = magnitude / max(float(weight_sq.sqrt()), 1e-30)
            if getattr(optimizer, 'record_causal_diagnostics', False):
                selected = {}
                for (name, parameter), old in zip(model.named_parameters(), before):
                    if not name.endswith(('ff1.weight', 'ff2.weight')):
                        continue
                    delta = (parameter.detach() - old).float().flatten(1)
                    row_energy = delta.square().sum(dim=1)
                    total = row_energy.sum()
                    if float(total) > 0:
                        share = row_energy / total
                        selected[name] = {'max_row_share':float(share.max()),
                                          'effective_row_fraction':float(1 / (share.square().sum() * share.numel()))}
                stats['ff_update_distribution'] = selected
    return stats


def equal_nested(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_nested(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal_nested(x, y) for x, y in zip(a, b))
    return a == b


def snapshot(model, optimizer, progress, registration_hash):
    import torch
    return dict(registration_sha256=registration_hash, model=model.state_dict(), optimizer=optimizer.state_dict(),
        gradients=[None if p.grad is None else p.grad.detach().clone() for p in model.parameters()],
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), progress=progress)


def restore(model, optimizer, saved, registration_hash):
    import torch
    if saved["registration_sha256"] != registration_hash:
        raise ValueError("checkpoint registration differs")
    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    for parameter, gradient in zip(model.parameters(), saved["gradients"]):
        parameter.grad = None if gradient is None else gradient.clone()
    torch.set_rng_state(saved["torch_rng"])
    torch.cuda.set_rng_state_all(saved["cuda_rng"])
    return saved["progress"]


def physical_audit(optimizer):
    import torch
    state_bytes = 0
    code_blocks = 0
    phase_positive = 0
    phase_count = 0
    anchor_gap = 0.0
    anchor_count = 0
    resource_sum = 0.0
    resource_pool_sum = 0.0
    resource_count = 0
    resource_minimum = None
    resource_maximum = None
    for parameter, state in optimizer.state.items():
        for name in ('row_reserve', 'col_reserve'):
            if name in state:
                value = state[name]
                resource_sum += float(value.sum())
                resource_count += value.numel()
                low, high = float(value.min()), float(value.max())
                resource_minimum = low if resource_minimum is None else min(resource_minimum, low)
                resource_maximum = high if resource_maximum is None else max(resource_maximum, high)
        if 'resource_pool' in state:
            resource_pool_sum += float(state['resource_pool'])
        if 'phase' in state:
            phase_positive += int((state['phase'] > 0).sum())
            phase_count += state['phase'].numel()
        if 'anchor' in state:
            anchor_count += state['anchor'].numel()
            anchor_gap += float((parameter.detach().reshape_as(state['anchor']) - state['anchor']).abs().sum())
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state_bytes += value.numel() * value.element_size()
                if key == "code":
                    low, high = torch.aminmax(value)
                    if value.dtype != torch.int16 or int(low) < 0 or int(high) >= 585:
                        raise AssertionError("illegal encoded cycle state")
                    code_blocks += value.numel()
    residual = getattr(optimizer, 'last_budget_residual', None)
    return dict(state_bytes=state_bytes, cycle_blocks=code_blocks,
                max_allocation_budget_residual=None if residual is None else float(residual),
                phase_positive_fraction=phase_positive / phase_count if phase_count else None,
                mean_anchor_gap=anchor_gap / anchor_count if anchor_count else None,
                resource_mean=resource_sum / resource_count if resource_count else None,
                resource_pool=resource_pool_sum,
                resource_total=resource_sum + resource_pool_sum,
                resource_min=resource_minimum, resource_max=resource_maximum)


def preflight(record, registration, output):
    import torch
    from copy import deepcopy
    torch_setup(record["experiment"]["seed"])
    stream = np.load(ROOT / record["data"]["streams"]["train_A"]["path"], mmap_mode="r")
    x, y = batch_from(stream, 0, 8, 128)
    results = {}
    initial = None
    for arm in record["experiment"]["arms"]:
        torch_setup(record["experiment"]["seed"])
        model = make_model(record)
        if initial is None:
            initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        assert equal_nested(initial, {k: v.detach().cpu() for k, v in model.state_dict().items()})
        optimizer = make_optimizer(model, record, arm)
        assert equal_nested(initial, {k: v.detach().cpu() for k, v in model.state_dict().items()}), "optimizer construction changed weights"
        with torch.no_grad():
            a = model(x)
            changed = x.clone()
            changed[:, 64:] = (changed[:, 64:] + 1) % 260
            assert torch.equal(a[:, :64], model(changed)[:, :64]), "future input affects earlier logits"
        before = {k: v.clone() for k, v in model.state_dict().items()}
        torch.cuda.synchronize()
        started = time.perf_counter()
        first = step(model, optimizer, x, y)
        torch.cuda.synchronize()
        first_seconds = time.perf_counter() - started
        if not any(torch.count_nonzero(p.grad).item() for p in model.parameters()):
            raise AssertionError("all gradients zero")
        saved = deepcopy(snapshot(model, optimizer, dict(update=1), sha(registration)))
        disk = Path(output).parent / ("preflight_" + arm + ".pt")
        disk.parent.mkdir(parents=True, exist_ok=True)
        torch.save(saved, disk)
        started = time.perf_counter()
        second = step(model, optimizer, x, y)
        torch.cuda.synchronize()
        steady_seconds = time.perf_counter() - started
        continued = deepcopy(snapshot(model, optimizer, dict(update=2), sha(registration)))
        other_model = make_model(record)
        other_optimizer = make_optimizer(other_model, record, arm)
        restore(other_model, other_optimizer, torch.load(disk, weights_only=False), sha(registration))
        repeated = step(other_model, other_optimizer, x, y)
        assert second == repeated
        assert equal_nested(continued, snapshot(other_model, other_optimizer, dict(update=2), sha(registration)))
        results[arm] = dict(initial_loss=first["nll"] / first["targets"], exact_full_continuation=True,
            initialization_unchanged=True,
            initialization_sha256=hashlib.sha256(b"".join(k.encode()+v.numpy().tobytes() for k,v in initial.items())).hexdigest(),
            causal_logits=True, first_step_seconds=first_seconds, steady_step_seconds=steady_seconds,
            changed_parameters=sum(int(torch.count_nonzero(p - before[n]).item()) for n, p in model.state_dict().items()),
            parameters=sum(p.numel() for p in model.parameters()), physical=physical_audit(optimizer))
        del model, optimizer, other_model, other_optimizer, before, saved, continued
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({arm: results[arm]}), flush=True)
    receipt = dict(scope="real-data numerical/gradient/causality/full-resume interface; not capability evidence",
        registration_sha256=sha(registration), sources=record["sources"], arms=results,
        status="passed", full_life_admitted=False,
        training_scope="registered controlled optimizer comparison only; empirical quality remains open")
    write_json(output, receipt)


def evaluate(model, record):
    import torch
    model.eval()
    results = {}
    with torch.no_grad():
        for domain in ("A", "B"):
            stream = np.load(ROOT / record["data"]["streams"]["validation_" + domain]["path"] , mmap_mode="r")
            totals = Counter()
            for cursor in range(0, record["experiment"]["validation_targets_per_domain"], 1024):
                x, y = batch_from(stream, cursor, 8, 128)
                losses = torch.nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten(), reduction="none")
                mask = y.flatten() >= 4
                totals.update(nll=losses.sum().item(), targets=y.numel(),
                    byte_nll=losses[mask].sum().item(), bytes=int(mask.sum().item()))
            results[domain] = dict(totals, bpb=totals["byte_nll"] / totals["bytes"] / math.log(2))
    model.train()
    return results


def train(record, registration, arm, directory, preflight_path):
    import torch
    passed = json.loads(Path(preflight_path).read_text())
    if passed["status"] != "passed" or passed["registration_sha256"] != sha(registration):
        raise ValueError("matching full-resume real-data preflight required")
    torch_setup(record["experiment"]["seed"])
    directory = Path(directory) / arm
    directory.mkdir(parents=True, exist_ok=True)
    model = make_model(record)
    optimizer = make_optimizer(model, record, arm)
    optimizer.record_causal_diagnostics = record['experiment'].get('record_causal_diagnostics', False)
    streams = {d: np.load(ROOT / record["data"]["streams"]["train_" + d]["path"], mmap_mode="r") for d in ("A", "B")}
    progress = dict(update=0, cursors={"A": 0, "B": 0}, totals={}, wall_seconds=0.0, validation=[])
    checkpoint = directory / "last.pt"
    if checkpoint.exists():
        progress = restore(model, optimizer, torch.load(checkpoint, weights_only=False), sha(registration))
    elif (directory / "curve.jsonl").exists():
        raise ValueError("curve exists without a complete checkpoint; refusing ambiguous restart")
    # A terminated process may have appended a curve tail after its last save.
    # Preserve it as uncommitted exposure before resuming the exact checkpoint.
    curve = directory / "curve.jsonl"
    if curve.exists():
        lines = curve.read_text().splitlines()
        keep, tail = [], []
        for line in lines:
            row = json.loads(line)
            (keep if row["update"] <= progress["update"] else tail).append(line)
        if tail:
            (directory / ("uncommitted_tail_" + str(time.time_ns()) + ".jsonl")).write_text("\n".join(tail) + "\n")
            curve.write_text("\n".join(keep) + ("\n" if keep else ""))
    phases = [d for d, count in record["experiment"]["phases"] for _ in range(count)]
    torch.cuda.reset_peak_memory_stats()
    total = record["experiment"]["updates"]
    started = time.perf_counter()
    base_seconds = progress["wall_seconds"]
    with curve.open("a", encoding="utf-8", buffering=1) as log:
        if progress["update"] == 0:
            progress["validation"].append(dict(update=0, domains=evaluate(model, record)))
        for update in range(progress["update"], total):
            domain = phases[update]
            cursor = progress["cursors"][domain]
            x, y = batch_from(streams[domain], cursor, 8, 128)
            bridge = None
            if update + 1 < len(phases) and phases[update + 1] != domain:
                next_domain = phases[update + 1]
                target = int(streams[next_domain][progress["cursors"][next_domain]])
                y[-1, -1] = target
                bridge = dict(from_domain=domain, to_domain=next_domain, target=target)
            record_update = record['experiment'].get('record_update_stats', False) and (update + 1) % 250 == 0
            outcome_every = int(record['experiment'].get('outcome_every', 0))
            outcome_due = outcome_every > 0 and (update + 1) % outcome_every == 0 and hasattr(optimizer, 'observe_outcome')
            fresh = None
            if outcome_due:
                try:
                    fresh = batch_from(streams[domain], cursor + 1024, 8, 128)
                except ValueError:
                    fresh = None
            fresh_before = None if fresh is None else batch_mean_loss(model, *fresh)
            stats = step(model, optimizer, x, y, record_update=record_update)
            if fresh is not None:
                fresh_after = batch_mean_loss(model, *fresh)
                fresh_decrease = fresh_before - fresh_after
                optimizer.observe_outcome(fresh_decrease)
                stats.update(fresh_before=fresh_before, fresh_after=fresh_after,
                             fresh_decrease=fresh_decrease,
                             outcome_evidence=fresh_decrease - stats.get('predicted_decrease', 0.0),
                             outcome_rho=float(optimizer.param_groups[0].get('rho', 1.0)))
            progress["cursors"][domain] += 1024
            progress["update"] = update + 1
            for key, value in stats.items():
                if isinstance(value, (float, int)):
                    progress["totals"][key] = progress["totals"].get(key, 0) + value
            row = dict(update=update + 1, domain=domain, fresh_cursor=cursor,
                **stats, bpb=stats["byte_nll"] / stats["bytes"] / math.log(2))
            if arm in ('step_fixed', 'step_adaptive'):
                row['rho'] = optimizer.param_groups[0]['rho']
                row['effective_base_lr'] = optimizer.param_groups[0]['lr'] * row['rho']
            if bridge:
                row["scored_bridge"] = bridge
            if (update + 1) % record["experiment"]["validation_every"] == 0:
                row["validation"] = evaluate(model, record)
                row["physical"] = physical_audit(optimizer)
                progress["validation"].append(dict(update=update + 1, domains=row["validation"]))
            log.write(json.dumps(row) + "\n")
            if (update + 1) % 50 == 0:
                torch.cuda.synchronize()
                progress["wall_seconds"] = base_seconds + time.perf_counter() - started
                write_json(directory / "progress.json", dict(status="running", arm=arm,
                    update=update + 1, total=total, phase=domain, last_bpb=row["bpb"],
                    wall_seconds=progress["wall_seconds"], registration_sha256=sha(registration)))
                print(json.dumps(dict(arm=arm, update=update + 1, domain=domain, bpb=row["bpb"])), flush=True)
            if (update + 1) % record["experiment"]["checkpoint_every"] == 0 or update + 1 == total:
                torch.cuda.synchronize()
                progress["wall_seconds"] = base_seconds + time.perf_counter() - started
                temporary = directory / "next.pt"
                torch.save(snapshot(model, optimizer, progress, sha(registration)), temporary)
                temporary.replace(checkpoint)
    result = dict(status="completed_budget", arm=arm, updates=progress["update"],
        cursors=progress["cursors"], scored_targets=progress["totals"]["targets"],
        prequential_bpb=progress["totals"]["byte_nll"] / progress["totals"]["bytes"] / math.log(2),
        validation=progress["validation"], wall_seconds=progress["wall_seconds"],
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), resident_cuda_bytes=torch.cuda.memory_allocated(),
        parameters=sum(p.numel() for p in model.parameters()), physical=physical_audit(optimizer),
        checkpoint_sha256=sha(checkpoint), registration_sha256=sha(registration),
        convergence="fixed budget exhausted; convergence/superiority not inferred automatically")
    write_json(directory / "result.json", result)
    write_json(directory / "progress.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--resource", action="store_true", help="prepare the new online-observer registration at the explicit registration path")
    parser.add_argument("--feedback-control", action="store_true", help="prepare the same-state on/off coupling comparison")
    parser.add_argument("--matrix-comparison", action="store_true", help="prepare the user-requested matrix/Adam/Muon comparison")
    parser.add_argument("--auto-step-comparison", action="store_true", help="compare fixed and adaptive step on identical ResourceCycle state")
    parser.add_argument("--adam-cycle-comparison", action="store_true", help="compare Adam with and without ResourceCycle competition, matched state")
    parser.add_argument("--matched-rate-comparison", action="store_true", help="compare ResourceCycle and Adam-base at the same nominal rate")
    parser.add_argument("--latch-demo", action="store_true", help="compare the active restoration demo with a matched-state control")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--arm", choices=("cycle", "resource", "feedback_on", "feedback_off", "step_fixed", "step_adaptive", "adam_cycle_control", "adam_cycle", "latch_control", "latch_demo", "causal_allocation_off", "causal_phase_off", "fatigue_on", "fatigue_off", "outcome_on", "outcome_off", "adam", "muon", "all"))
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--output", type=Path, default=ROOT / "results/continuous_optimizer_owt_20261028")
    args = parser.parse_args()
    if args.prepare:
        if (args.resource or args.feedback_control or args.matrix_comparison or args.auto_step_comparison or args.adam_cycle_comparison or args.matched_rate_comparison or args.latch_demo) and args.registration == DEFAULT_REGISTRATION:
            parser.error("use a new --registration path to preserve the historical cycle experiment")
        prepare(args.registration, args.resource or args.feedback_control or args.matrix_comparison or args.auto_step_comparison or args.adam_cycle_comparison or args.matched_rate_comparison or args.latch_demo, args.feedback_control, args.matrix_comparison or args.auto_step_comparison or args.adam_cycle_comparison or args.matched_rate_comparison or args.latch_demo, args.auto_step_comparison, args.adam_cycle_comparison, args.matched_rate_comparison, args.latch_demo)
        return
    record = load_registration(args.registration)
    preflight_path = args.output / "preflight.json"
    if args.preflight:
        preflight(record, args.registration, preflight_path)
    elif args.arm == "all":
        for arm in record["experiment"]["arms"]:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), "--arm", arm,
                "--registration", str(args.registration), "--output", str(args.output)], check=True)
    elif args.arm:
        train(record, args.registration, args.arm, args.output, preflight_path)
    else:
        parser.error("choose --prepare, --preflight or --arm")


if __name__ == "__main__":
    main()
