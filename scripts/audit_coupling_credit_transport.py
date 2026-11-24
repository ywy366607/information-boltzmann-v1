"""Read saved coupling credit and optimizer transport without loading the body.

The local torch ZIP pickle is inspected as tensor descriptors. Only explicitly
selected small storages are materialized on CPU; large tables remain on disk.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import zipfile

import torch

from scripts.ib_local.compressed_checkpoint import checkpoint_stream


@dataclass
class StorageRef:
    kind: object
    key: str
    location: str
    count: int


@dataclass
class TensorRef:
    storage: StorageRef
    offset: int
    size: tuple
    stride: tuple
    dtype: object = None


def rebuild_tensor(storage, offset, size, stride, *unused):
    return TensorRef(storage, offset, tuple(size), tuple(stride))


def rebuild_tensor_v3(storage, offset, size, stride, requires_grad, hooks, dtype, *unused):
    return TensorRef(storage, offset, tuple(size), tuple(stride), dtype)


class DescriptorUnpickler(pickle.Unpickler):
    def persistent_load(self, identity):
        if identity[0] != "storage" or len(identity) != 5:
            raise ValueError(f"unsupported checkpoint persistent reference: {identity[0]}")
        return StorageRef(*identity[1:])

    def find_class(self, module, name):
        if module == "torch._utils" and name in ("_rebuild_tensor", "_rebuild_tensor_v2"):
            return rebuild_tensor
        if module == "torch._utils" and name == "_rebuild_tensor_v3":
            return rebuild_tensor_v3
        if module == "torch._utils" and name in ("_rebuild_parameter", "_rebuild_parameter_with_state"):
            return lambda value, *unused: value
        return super().find_class(module, name)


class TensorArchive:
    def __init__(self, stream):
        self.archive = zipfile.ZipFile(stream)
        entries = [n for n in self.archive.namelist() if n.endswith("/data.pkl")]
        if len(entries) != 1:
            raise ValueError("expected one native torch ZIP data.pkl")
        self.prefix = entries[0][:-len("data.pkl")]
        self.state = DescriptorUnpickler(io.BytesIO(self.archive.read(entries[0]))).load()
        self.loaded_bytes = 0
        self.cache = {}

    def tensor(self, ref):
        if not isinstance(ref, TensorRef):
            raise TypeError("expected a tensor descriptor")
        key = (ref.storage.key, ref.dtype)
        if key not in self.cache:
            name = ref.storage.kind.__name__
            dtype = ref.dtype or {"FloatStorage": torch.float32, "DoubleStorage": torch.float64,
                     "LongStorage": torch.int64, "IntStorage": torch.int32,
                     "BoolStorage": torch.bool, "ByteStorage": torch.uint8,
                     "HalfStorage": torch.float16, "BFloat16Storage": torch.bfloat16}[name]
            member = self.prefix+"data/"+ref.storage.key
            length = self.archive.getinfo(member).file_size
            if length > 32*1024**2:
                raise ValueError("audit refuses large storage materialization")
            raw = bytearray(self.archive.read(member))
            value = torch.frombuffer(raw, dtype=dtype).clone()
            expected_count = value.nbytes if ref.dtype is not None else value.numel()
            if expected_count != ref.storage.count:
                raise ValueError("checkpoint storage length mismatch")
            self.loaded_bytes += len(raw)
            self.cache[key] = value
        return self.cache[key].as_strided(ref.size, ref.stride, ref.offset)


@contextmanager
def open_archive(path, spool):
    with checkpoint_stream(path, spool_directory=spool) as stream:
        reader = TensorArchive(stream)
        try:
            yield reader
        finally:
            reader.archive.close()


def verify_reader():
    base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    buffer = io.BytesIO()
    small_unsigned = torch.arange(5).to(torch.uint16)
    expected_tensors = dict(base=base, view=base[:, 1::2], scalar=torch.tensor(3, dtype=torch.int64), unsigned=small_unsigned)
    torch.save(dict(expected_tensors, unrequested=torch.zeros(1000)), buffer)
    buffer.seek(0)
    reader = TensorArchive(buffer)
    assert reader.loaded_bytes == 0
    for name, expected in expected_tensors.items():
        assert torch.equal(reader.tensor(reader.state[name]), expected)
    assert reader.loaded_bytes == base.numel()*base.element_size()+8+small_unsigned.nbytes
    reader.archive.close()


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(8*1024**2):
            digest.update(data)
    return digest.hexdigest()


def stats(value):
    value = value.detach().double()
    return dict(min=float(value.min()), max=float(value.max()), rms=float(value.square().mean().sqrt()),
                norm=float(value.norm()), nonzero=int(value.count_nonzero()))


@torch.no_grad()
def replay_receipts(reader, group, parameter_id, parameter, receipts):
    from information_boltzmann.budgeted_adamw import BudgetedAdamW
    rows = [json.loads(x) for x in receipts.read_text(encoding="utf8").splitlines()]
    updates = [r for r in rows if r["kind"] == "optimizer_execution"]
    saved = reader.state["optimizer"]["state"][parameter_id]
    initial_state = {k: reader.tensor(v).clone() if isinstance(v, TensorRef) else copy.deepcopy(v)
                     for k, v in saved.items()}
    outputs = {}
    for mode in ("production", "lr_1e2", "epsilon_zero", "gradient_history_x100", "gate_bypass"):
        p = torch.nn.Parameter(parameter.clone())
        metadata = {k: copy.deepcopy(v) for k, v in group.items() if k != "params"}
        optimizer = BudgetedAdamW([dict(metadata, params=[p])])
        optimizer.state[p] = copy.deepcopy(initial_state)
        opts = optimizer.param_groups[0]
        opts.update(metadata)
        if mode == "lr_1e2":
            opts["lr"] = .01
        elif mode == "epsilon_zero":
            opts["eps"] = 0.
        elif mode == "gate_bypass":
            opts["cooling"] = False
        elif mode == "gradient_history_x100":
            st = optimizer.state[p]
            st["exp_avg"] *= 100
            st["exp_avg_sq"] *= 100**2
            st["budget_energy"] *= 100**2
            st["budget_prior_rms"] *= 100
        snapshots = []
        for row in updates:
            if row["gradient"] is None:
                continue
            p.grad = torch.tensor(row["gradient"], dtype=p.dtype).reshape_as(p)
            if mode == "gradient_history_x100":
                p.grad *= 100
            optimizer.step()
            snapshots.append(dict(call=row["optimizer_call"],
                                  gate=float(optimizer.state[p]["budget_gate"])
                                  if mode != "gate_bypass" else 1.,
                                  max_displacement=float((p-parameter).abs().max())))
        reference = torch.tensor(updates[-1]["after"], dtype=p.dtype).reshape_as(p)
        error = float((p-reference).abs().max())
        if mode == "production":
            assert error < 1e-6
        outputs[mode] = dict(snapshots=snapshots, max_raw_displacement=float((p-parameter).abs().max()),
                             max_softplus_relative_change=float((torch.nn.functional.softplus(p)/
                                 torch.nn.functional.softplus(parameter)-1).abs().max()),
                             saved_actual_final_max_error=error,
                             final_parameter=p.detach().tolist())
    credits = [r for r in rows if r["kind"].endswith("credit_return")]
    return dict(source=str(receipts), actual_optimizer_calls=len(updates),
                actual_coupling_updates=sum(r["gradient"] is not None for r in updates),
                state_credit_returns=sum(r["kind"] == "state_credit_return" for r in credits),
                momentum_credit_returns=sum(r["kind"] == "momentum_credit_return" for r in credits),
                actual_nonzero_gradients=[sum(g != 0 for g in r["gradient"])
                    for r in updates if r["gradient"] is not None],
                actual_max_step_reconstruction_error=max(r.get("reconstructed_parameter_max_error", 0) for r in updates),
                premise="identical saved actual gradient sequence and initial optimizer state; local update replay, no new world trajectory",
                scenarios=outputs)


def audit(path, spool, receipts=None):
    before = sha(path)
    with open_archive(path, spool) as reader:
        state = reader.state
        sd = state["model"]
        prefix = next(k[:-len("theta_freq")] for k in sd if k.endswith("theta_freq"))
        roles = ("freq", "context", "coupling", "pair")
        refs = [sd[prefix+"theta_"+role] for role in roles]
        # Role tensors are consecutive in the actual model parameter order;
        # match the unique optimizer group with their four declared shapes.
        optimizer = state["optimizer"]
        matches = []
        for group in optimizer["param_groups"]:
            ids = group["params"]
            if len(ids) == 4 and all(
                    optimizer["state"].get(i, {}).get("exp_avg", None) is not None and
                    optimizer["state"][i]["exp_avg"].size == ref.size for i, ref in zip(ids, refs)):
                matches.append(group)
        if len(matches) != 1:
            raise ValueError("role optimizer group must be uniquely identified")
        group = matches[0]
        theta = torch.cat([reader.tensor(ref) for ref in refs], dim=-1)
        heads = refs[0].size[-1]
        latest = reader.tensor(sd[prefix+"last_gradient"])
        momentum_credit = reader.tensor(sd[prefix+"last_momentum_gradient"])
        slices = ((0, heads), (heads, heads+1), (heads+1, heads+2), (heads+2, heads+4))
        roles_out = {}
        for role, parameter_id, ref, (lo, hi) in zip(roles, group["params"], refs, slices):
            opt = optimizer["state"][parameter_id]
            count = float(reader.tensor(opt["step"]))
            m = reader.tensor(opt["exp_avg"])
            v = reader.tensor(opt["exp_avg_sq"])
            beta1, beta2 = group["betas"]
            corrected_m = m/(1-beta1**count)
            corrected_rms = (v/(1-beta2**count)).sqrt()
            direction = corrected_m/(corrected_rms+group["eps"])
            gate = reader.tensor(opt["budget_gate"])
            nominal = -group["lr"]*direction
            executed = nominal*gate
            energy = reader.tensor(opt["budget_energy"])
            from information_boltzmann.budgeted_adamw import noise_weight, coherence_gate
            observer_count = opt["budget_count"]
            prior = reader.tensor(opt["budget_prior_rms"])
            corrected_energy = energy/(1-beta1**observer_count)
            upper = (m.square().mean().sqrt()+beta1**observer_count*prior)/(1-beta1**observer_count)
            signal = upper.square()
            noise = noise_weight(beta1, observer_count)
            calculated_gate = coherence_gate(signal, corrected_energy, noise,
                                              wiener=group.get("wiener_gate", False))
            assert torch.allclose(calculated_gate, gate, atol=2e-6, rtol=1e-5)
            roles_out[role] = dict(
                parameter_id=parameter_id, step=count, observer_count=observer_count,
                latest_consumed_credit=stats(latest[:, lo:hi]),
                latest_momentum_credit=stats(momentum_credit[:, lo:hi]),
                adam_m=stats(m), adam_corrected_rms=stats(corrected_rms),
                epsilon_fraction=stats(group["eps"]/(corrected_rms+group["eps"])),
                adam_direction=stats(direction), gate=float(gate), reconstructed_gate=float(calculated_gate),
                signal=float(signal), energy=float(corrected_energy), noise_weight=noise,
                signal_to_noise_threshold=float(signal/(noise*corrected_energy).clamp_min(1e-38)),
                ungated_raw_step=stats(nominal), terminal_direction_times_gate=stats(executed),
                parameter=reader.tensor(ref).tolist(), direction_values=direction.tolist(),
            )
        # Saved S/P and production geometry; every tensor stays on CPU.
        spec = state["config"]["native_body"]["backbone"]
        dim = spec["d"]//heads
        carried = reader.tensor(state["engine"]["recurrent"])
        s = carried[..., :heads*dim*dim].reshape(spec["layers"], -1, heads, dim, dim)
        p = torch.stack([reader.tensor(ref) for ref in state["engine"]["bfss_momenta"]]).reshape_as(s)
        context, scale, impedance = [reader.tensor(sd[prefix+k]) for k in ("context", "frequency_scale", "impedance")]
        from scripts.ib_local.adaptive_hamiltonian import coefficients, force_analytic, quiet
        omega, strength = coefficients(theta, context, heads, scale)
        force = force_analytic(s, theta, heads)
        self_force = omega*impedance*s
        norm = lambda x: x.flatten(-2).norm(dim=-1)
        ratios = norm(force)/norm(self_force).clamp_min(1e-30)
        # Counterfactual execution of terminal Adam direction, frozen boundary.
        role = roles_out["coupling"]
        candidate = theta.clone()
        direction = torch.tensor(role["direction_values"], dtype=theta.dtype)
        candidate[:, heads+1:heads+2] -= group["lr"]*direction
        candidate_force = force_analytic(s, candidate, heads)
        next_strength = coefficients(candidate, context, heads, scale)[1]
        force_change = norm(candidate_force-force)/norm(force).clamp_min(1e-30)
        without = theta.clone()
        without[:, heads+1] = -torch.inf
        actual_next = quiet(s, p, theta, context, .01, scale, impedance)
        without_next = quiet(s, p, without, context, .01, scale, impedance)
        ds = norm(actual_next[0]-without_next[0])/norm(actual_next[0]).clamp_min(1e-30)
        dp = norm(actual_next[1]-without_next[1])/norm(actual_next[1]).clamp_min(1e-30)
        flat_omega = omega.reshape(spec["layers"], heads)
        flat_ratios = ratios.reshape(spec["layers"], heads)
        force_norms = norm(force).reshape(spec["layers"], heads)
        self_norms = norm(self_force).reshape(spec["layers"], heads)
        order = flat_ratios.flatten().argsort(descending=True)[:12]
        strongest = []
        for index in order.tolist():
            layer, head = divmod(index, heads)
            strongest.append(dict(layer=layer+1, head=head+1,
                omega=float(flat_omega[layer, head]), force_ratio=float(flat_ratios[layer, head]),
                force_norm=float(force_norms[layer, head]), self_force_norm=float(self_norms[layer, head]),
                one_step_state_relative=float(ds.reshape_as(flat_ratios)[layer, head]),
                one_step_momentum_relative=float(dp.reshape_as(flat_ratios)[layer, head])))
        result = dict(
            checkpoint=str(path), source_sha256=before, cycle=state.get("cycle", None),
            root_keys=list(state), role_group=dict(lr=group["lr"], eps=group["eps"],
                weight_decay=group["weight_decay"], betas=group["betas"], wiener=group.get("wiener_gate", False)),
            roles=roles_out,
            carried_credit=dict(trace_steps=int(reader.tensor(sd[prefix+"trace_steps"])),
                trace_s=stats(reader.tensor(sd[prefix+"sensitivity_s"])),
                trace_p=stats(reader.tensor(sd[prefix+"sensitivity_p"])),
                momentum_credit_count=int(reader.tensor(sd[prefix+"momentum_credit_count"]))),
            physical=dict(coupling=strength.tolist(), omega_min=float(omega.min()), omega_max=float(omega.max()),
                head_force_ratio=ratios.flatten(1).tolist(), ratio_median=float(ratios.median()),
                layer_force_ratio_median=float((force.flatten(1).norm(dim=-1)/
                    self_force.flatten(1).norm(dim=-1).clamp_min(1e-30)).median()),
                ratio_max=float(ratios.max()), state_rms=float(s.square().mean().sqrt()),
                momentum_rms=float(p.square().mean().sqrt()),
                head_frequency=flat_omega.tolist(), strongest_heads=strongest,
                heads_above_one_percent=int((ratios>.01).sum()),
                heads_above_ten_percent=int((ratios>.1).sum()),
                one_step_removed_state_relative_max=float(ds.max()),
                one_step_removed_momentum_relative_max=float(dp.max())),
            terminal_direction_counterfactual=dict(
                meaning="same saved state and final Adam direction; hypothetical ungated step, not a reconstructed prior update",
                relative_coupling_change=(next_strength/strength-1).tolist(),
                relative_force_change=stats(force_change),
                raw_step_max=float((group["lr"]*direction).abs().max())),
            materialized_storage_bytes=reader.loaded_bytes,
            format="read-only tensor-descriptor inspection with exact CPU storage materialization",
        )
        if receipts is not None:
            result["actual_gradient_replay"] = replay_receipts(
                reader, group, group["params"][2], reader.tensor(refs[2]), receipts)
    result["source_unchanged"] = sha(path) == before
    assert result["source_unchanged"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--spool", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--receipts", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    verify_reader()
    report = dict(schema=1, optimizer_updates=0, world_calls=0, reader_storage_and_view_check=True, bodies=[])
    for path in args.checkpoint:
        body = audit(path, args.spool, args.receipts)
        report["bodies"].append(body)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
        print(json.dumps(dict(checkpoint=str(path), bytes=body["materialized_storage_bytes"],
                             coupling=body["roles"]["coupling"], ratio_median=body["physical"]["ratio_median"])), flush=True)


if __name__ == "__main__":
    main()
