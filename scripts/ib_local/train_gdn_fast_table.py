"""GDN body with the byte n-gram fast table as a second memory (2026-11-03).

Protocol and body match results/cgb_feedback_birth_20261103 (control arm): OpenWebText bytes,
5000 updates of 512 bytes, A 70% / new B / revisit A / new C, decay_then_delta with official
log-timescale retention, 512-byte backpropagation window, AdamW 3e-4, grouped clipping,
fused recurrence.  Only the prequential stream metrics are computed (the primary metrics of
that run); no frozen validation or context evaluation.

The table (`ngram_fast_table.py`) runs on the CPU, one window ahead of nothing: for the
window's bytes it returns, at each position, its read distribution over the next byte, then
writes the window.  Its arrays live in RAM or in np.memmap files.  The body sees log p_table
in two places, both learned by the language loss:

* input of layer 1, Engram-style contextual gate:
  h += sigmoid(<rms(h), rms(W_k f)> / sqrt(d)) * W_v f, with f = centred log p_table;
* output, geometric mixture: logits += softplus(w . h + b) * log p_table.

With --no-table the same script trains the plain control (no extra parameters); its numbers
must reproduce the control phases of the latch run.
"""

from __future__ import annotations

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0]) == script_dir:
    sys.path.pop(0)  # ib_local/types.py would shadow the standard library
ROOT_DIR = os.path.dirname(os.path.dirname(script_dir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

from scripts.ib_local import gdn_fused  # noqa: E402
from scripts.ib_local.gated_deltanet import GatedDeltaNetLM  # noqa: E402
from scripts.ib_local.ngram_fast_table import ByteNgramFastTable  # noqa: E402
from scripts.ib_local.run_ngram_fast_table import protocol_stream  # noqa: E402

LEXICAL = ("source.embedding.weight", "decoder.weight", "decoder.bias")
FLOOR = 1e-6


class TableGDN(nn.Module):
    """GatedDeltaNetLM plus table injection; the base is built first so its weights match."""

    def __init__(self, base: GatedDeltaNetLM, use_table: bool, inject_layer: int = 1):
        super().__init__()
        self.base, self.use_table, self.inject_layer = base, use_table, inject_layer
        if use_table:
            d = base.d
            self.table_key = nn.Linear(256, d, bias=False)
            self.table_value = nn.Linear(256, d, bias=False)
            nn.init.normal_(self.table_value.weight, std=0.02)
            self.key_norm, self.query_norm = nn.RMSNorm(d), nn.RMSNorm(d)
            self.mix = nn.Linear(d, 1)
            nn.init.zeros_(self.mix.weight)
            nn.init.constant_(self.mix.bias, math.log(math.expm1(0.5)))

    def initial_state(self, batch, device=None):
        state = self.base.initial_state(batch, device=device)
        if hasattr(self, 'coevolution'):
            state = torch.cat((state, state.new_zeros(*state.shape[:-1],
                               self.coevolution.state_size)), -1)
        return state

    def forward(self, ids, targets, state, table_logp):
        logits, next_state, diagnostics = self.predict(ids, state, table_logp)
        loss = F.cross_entropy(logits.reshape(-1, self.base.vocab_size), targets.reshape(-1))
        return loss, next_state, diagnostics

    def _run_layer_slice(self, start, end, h, state, table_logp, field, incoming,
                         field_keys, field_bias, diagnostics, skip_diagnostics):
        base = self.base
        states = []
        for index in range(start, end):
            layer = base.layers[index]
            if self.use_table and index == self.inject_layer:
                f = table_logp - table_logp.mean(-1, keepdim=True)
                key = self.key_norm(self.table_key(f))
                gate = torch.sigmoid((self.query_norm(h) * key).sum(-1, keepdim=True)
                                     / math.sqrt(base.d))
                h = h + gate * self.table_value(f)
                if not skip_diagnostics:
                    diagnostics["table_gate_mean"] = gate.detach().mean()
            packed = state[index]
            if field is not None:
                h, incoming, work_diag = field.exchange(layer, h, incoming,
                    packed[:, layer.state_size:], field_keys, field_bias)
                packed = packed[:, :layer.state_size]
            h, next_state, layer_diag = layer(h, packed)
            if field is not None:
                next_state = torch.cat((next_state, incoming[:, -1].flatten(1)), -1)
                layer_diag.update(work_diag)
            states.append(next_state)
            if not skip_diagnostics:
                for key_name, value in layer_diag.items():
                    diagnostics[f"l{index}_{key_name}"] = value
        return h, states, incoming

    def predict_hnet_enc(self, ids, state, table_logp, prev_k, has_prev):
        """H-Net §2.4 stage 1: outer encoder [0..enc_layers) + causal 1-byte router."""
        base = self.base
        hnet = self.hnet_chunker
        skip_diagnostics = getattr(self, '_decode_no_diagnostics', False) and not torch.is_grad_enabled()
        h = base.source.embedding(ids)
        field = getattr(self, 'coevolution', None)
        incoming = field_keys = field_bias = None
        if field is not None:
            incoming, field_keys, field_bias = field.prepare(self._coev_owner(), *ids.shape)
        if getattr(base, "self_oracle", False):
            from scripts.ib_local.self_state import marker_letter_mask
            mask = marker_letter_mask(ids)
            for layer in base.layers:
                layer.write_mask = mask
        diag = {}
        h_enc, enc_states, inc_enc = self._run_layer_slice(
            0, hnet.enc_layers, h, state, table_logp, field, incoming,
            field_keys, field_bias, diag, skip_diagnostics)
        p_t, b_t, k_t = hnet.route_step(h_enc, prev_k, has_prev)
        return h_enc, p_t, b_t, k_t, torch.stack(enc_states, 0), inc_enc, field_keys, field_bias

    def predict_hnet_core(self, h_enc, p_t, state, prev_z_bar, inc_enc, field_keys, field_bias):
        """H-Net §2.4 stage 2 (boundary-only): 24 deep core layers + EMA dechunk update."""
        base = self.base
        hnet = self.hnet_chunker
        core_start = hnet.enc_layers
        core_end = base.num_layers - hnet.dec_layers
        skip_diagnostics = getattr(self, '_decode_no_diagnostics', False) and not torch.is_grad_enabled()
        field = getattr(self, 'coevolution', None)
        diag = {}
        z_t, core_states, inc_core = self._run_layer_slice(
            core_start, core_end, h_enc, state, None, field, inc_enc,
            field_keys, field_bias, diag, skip_diagnostics)
        z_bar = hnet.dechunk_update(z_t, p_t, prev_z_bar)
        return z_bar, torch.stack(core_states, 0), inc_core

    def predict_hnet_dec(self, h_enc, z_bar, state, table_logp, inc_bar, field_keys, field_bias, *, decode=True):
        """H-Net §2.4 stage 3: skip residual + outer decoder [core_end..num_layers) + logits."""
        base = self.base
        hnet = self.hnet_chunker
        core_end = base.num_layers - hnet.dec_layers
        skip_diagnostics = getattr(self, '_decode_no_diagnostics', False) and not torch.is_grad_enabled()
        field = getattr(self, 'coevolution', None)
        diag = {}
        h = hnet.combine_skip(z_bar, h_enc)
        h, dec_states, _ = self._run_layer_slice(
            core_end, base.num_layers, h, state, table_logp, field, inc_bar,
            field_keys, field_bias, diag, skip_diagnostics)
        dec_stack = torch.stack(dec_states, 0)
        if not decode:
            return None, dec_stack, diag
        h = base.final_norm(h)
        logits = base.decoder(h)
        if self.use_table:
            weight = F.softplus(self.mix(h))
            logits = logits + weight * table_logp
            if not skip_diagnostics:
                diag["table_weight_mean"] = weight.detach().mean()
        return logits, dec_stack, diag

    def predict(self, ids, state, table_logp, *, decode=True):
        if not decode and torch.is_grad_enabled():
            raise ValueError('state-only execution is for known no-gradient payloads')
        base = self.base
        hnet = getattr(self, 'hnet_chunker', None)
        step_st = getattr(self, '_hnet_step_state', None)
        # Unscoped 1-byte calls outside `hnet_step_scope` cleanly bypass H-Net instead of
        # forcing a boundary + skip residual on every byte.
        hnet_active = (
            hnet is not None
            and hnet.active_for(ids.shape[1])
            and (ids.shape[1] > 1 or step_st is not None)
        )
        if hnet_active and step_st is not None and ids.shape[1] == 1:
            hnet._pending_ratio_loss = None
            core_start = hnet.enc_layers
            core_end = base.num_layers - hnet.dec_layers
            h_enc, p_t, b_t, k_t, enc_st, inc_enc, f_keys, f_bias = self.predict_hnet_enc(
                ids, state, table_logp, step_st.prev_k, step_st.has_prev)
            step_st.prev_k = k_t
            step_st.has_prev = True
            step_st.steps += 1
            if bool(b_t[0].item()):
                step_st.boundaries += 1
                z_bar, core_st, inc_core = self.predict_hnet_core(
                    h_enc, p_t, state, step_st.prev_z_bar, inc_enc, f_keys, f_bias)
                step_st.prev_z_bar = z_bar
                step_st.prev_inc = inc_core
            else:
                core_st = state[core_start:core_end]
                z_bar = step_st.prev_z_bar
                inc_core = step_st.prev_inc
            logits, dec_st, diagnostics = self.predict_hnet_dec(
                h_enc, z_bar, state, table_logp, inc_core, f_keys, f_bias, decode=decode)
            next_state = torch.cat((enc_st, core_st, dec_st), dim=0)
            return logits, next_state, diagnostics
        skip_diagnostics = getattr(self, '_decode_no_diagnostics', False) and not torch.is_grad_enabled()
        h = base.source.embedding(ids)
        diagnostics, states = {}, []
        field = getattr(self, 'coevolution', None)
        if field is not None:
            incoming, field_keys, field_bias = field.prepare(self._coev_owner(), *ids.shape)
        if getattr(base, "self_oracle", False):
            from scripts.ib_local.self_state import marker_letter_mask
            mask = marker_letter_mask(ids)
            for layer in base.layers:
                layer.write_mask = mask
        hnet_pack = None
        core_start = hnet.enc_layers if hnet_active else -1
        core_end = (base.num_layers - hnet.dec_layers) if hnet_active else -1
        if hnet is not None and not hnet_active:
            hnet._pending_ratio_loss = None
        for index, layer in enumerate(base.layers):
            if hnet_active and index == core_start:
                p_b, b_mask, r_loss = hnet.route(h)
                hnet._pending_ratio_loss = r_loss if torch.is_grad_enabled() else None
                hnet_pack = hnet.chunk(h, p_b, b_mask, incoming=(incoming if field is not None else None))
                h = hnet_pack['x_chunk']
                if field is not None:
                    incoming = hnet_pack['incoming_chunk']
                if not skip_diagnostics:
                    diagnostics["hnet_boundary_rate"] = b_mask.detach().mean()
                    diagnostics["hnet_ratio_loss"] = r_loss.detach()
            elif hnet_active and index == core_end:
                h, inc_up = hnet.dechunk(h, hnet_pack, incoming_chunk=(incoming if field is not None else None))
                if field is not None:
                    incoming = inc_up
                hnet_pack = None
            if self.use_table and index == self.inject_layer:
                f = table_logp - table_logp.mean(-1, keepdim=True)
                key = self.key_norm(self.table_key(f))
                gate = torch.sigmoid((self.query_norm(h) * key).sum(-1, keepdim=True)
                                     / math.sqrt(base.d))
                h = h + gate * self.table_value(f)
                if not skip_diagnostics:
                    diagnostics["table_gate_mean"] = gate.detach().mean()
            packed = state[index]
            if field is not None:
                h, incoming, work_diag = field.exchange(layer, h, incoming,
                    packed[:, layer.state_size:], field_keys, field_bias)
                packed = packed[:, :layer.state_size]
            if (hnet_pack is not None and h.shape[0] > 1
                    and int(hnet_pack['counts'].min().item()) < int(hnet_pack['M_max'])):
                # Exact B>1 padded deep-core state: step each row strictly up to its own boundary
                # count `m` so shorter rows' recurrent state is never over-advanced through padding.
                B_cur = h.shape[0]
                counts_l = hnet_pack['counts'].tolist()
                h_rows, st_rows = [], []
                layer_diag = {}
                for bi in range(B_cur):
                    m_i = max(1, int(counts_l[bi]))
                    h_i, st_i, diag_i = layer(h[bi:bi + 1, :m_i], packed[bi:bi + 1])
                    if m_i < h.shape[1]:
                        h_i = F.pad(h_i, (0, 0, 0, h.shape[1] - m_i))
                    h_rows.append(h_i)
                    st_rows.append(st_i)
                    layer_diag = diag_i
                h = torch.cat(h_rows, dim=0)
                next_state = torch.cat(st_rows, dim=0)
            else:
                h, next_state, layer_diag = layer(h, packed)
            if field is not None:
                if (hnet_pack is not None and h.shape[0] > 1
                        and int(hnet_pack['counts'].min().item()) < int(hnet_pack['M_max'])):
                    last_inc = torch.stack([
                        incoming[bi, max(0, int(hnet_pack['counts'][bi].item()) - 1)].flatten()
                        for bi in range(h.shape[0])
                    ], dim=0)
                    next_state = torch.cat((next_state, last_inc), -1)
                else:
                    next_state = torch.cat((next_state, incoming[:, -1].flatten(1)), -1)
                layer_diag.update(work_diag)
            states.append(next_state)
            for key_name, value in layer_diag.items():
                diagnostics[f"l{index}_{key_name}"] = value
        if not decode:
            return None, torch.stack(states, 0), diagnostics
        h = base.final_norm(h)
        logits = base.decoder(h)
        if self.use_table:
            weight = F.softplus(self.mix(h))
            logits = logits + weight * table_logp
            if not skip_diagnostics:
                diagnostics["table_weight_mean"] = weight.detach().mean()
        if not skip_diagnostics:
            diagnostics["alpha_mean"] = torch.stack(
                [diagnostics[f"l{i}_alpha_mean"] for i in range(base.num_layers)]).mean()
        return logits, torch.stack(states, 0), diagnostics


class TableGraphTrainer:
    """TruncatedInternalTimeGraphTrainer with one more static input (the table read)."""

    def __init__(self, model, tokens, chunk_tokens, lr=3e-4):
        if tokens % chunk_tokens:
            raise ValueError("tokens must be divisible by chunk_tokens")
        self.model, self.tokens, self.chunk_tokens = model, tokens, chunk_tokens
        self.ids = torch.zeros(1, chunk_tokens, dtype=torch.long, device="cuda")
        self.targets = torch.zeros_like(self.ids)
        self.table = torch.full((1, chunk_tokens, 256), math.log(1 / 256), device="cuda")
        self.state = model.initial_state(1, device="cuda")
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, foreach=True)
        named = list(model.named_parameters())
        lexical = lambda n: n.removeprefix("base.") in LEXICAL
        self.groups = ([p for n, p in named if lexical(n)], [p for n, p in named if not lexical(n)])
        originals = [p.detach().clone() for p in model.parameters()]
        initial_state = self.state.detach().clone()

        def backward_chunk():
            loss, next_state, diagnostics = model(self.ids, self.targets, self.state, self.table)
            (loss / (tokens // chunk_tokens)).backward()
            return loss, next_state, diagnostics
        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.optimizer.zero_grad(set_to_none=True); backward_chunk()
        torch.cuda.current_stream().wait_stream(stream); torch.cuda.empty_cache()
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(parameter)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.loss, self.next_state, self.diagnostics = backward_chunk()
        with torch.no_grad():
            for parameter, original in zip(model.parameters(), originals):
                parameter.copy_(original)
            self.state.copy_(initial_state)
        self.optimizer.zero_grad(set_to_none=False)

    def step(self, ids, targets, table_logp):
        self.optimizer.zero_grad(set_to_none=False)
        loss_sum, diagnostics = 0.0, None
        for start in range(0, self.tokens, self.chunk_tokens):
            stop = start + self.chunk_tokens
            self.ids.copy_(ids[:, start:stop])
            self.targets.copy_(targets[:, start:stop])
            self.table.copy_(table_logp[:, start:stop])
            self.graph.replay(); loss_sum += self.loss.detach().clone()
            diagnostics = {k: v.detach().clone() for k, v in self.diagnostics.items()}
            with torch.no_grad():
                self.state.copy_(self.next_state.detach())
        norms = torch.stack(tuple(torch.nn.utils.clip_grad_norm_(g, 1.0, foreach=True)
                                  for g in self.groups))
        if not torch.isfinite(norms.square().sum()):
            raise FloatingPointError("non-finite gradient norm")
        self.optimizer.step()
        return loss_sum / (self.tokens // self.chunk_tokens), diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--no-table", action="store_true")
    parser.add_argument("--slots-log2", type=int, default=16)
    parser.add_argument("--storage", type=Path, help="np.memmap directory; default RAM")
    parser.add_argument("--stream", choices=["protocol", "recall", "mode"], default="protocol",
                        help="recall: OWT with key-value records (recall_task.py); "
                             "mode: one held bit with unshown answers (mode_task.py)")
    parser.add_argument("--recurrence", choices=["decay_then_delta", "feedback_delta"],
                        default="decay_then_delta")
    parser.add_argument("--feedback-init", choices=["constant", "threshold_spread"],
                        default="threshold_spread")
    parser.add_argument("--save-table", action="store_true", help="keep the table arrays in --output")
    parser.add_argument("--recall-spacer", type=int, default=128)
    parser.add_argument("--self-state", choices=["off", "learned", "oracle"], default="off",
                        help="add a non-decaying, gate-written second S per layer (self_state.py); "
                             "oracle: written only at the mode marker letter")
    parser.add_argument("--mode-easy-steps", type=int, default=0,
                        help="mode stream: updates of the dense bootstrap schedule before the long one")
    args = parser.parse_args()
    if (args.output / "summary.json").exists():
        raise FileExistsError("refusing to overwrite a finished run")
    args.output.mkdir(parents=True, exist_ok=True)
    fused_sha = gdn_fused.enable()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    if args.stream == "recall":
        from scripts.ib_local.recall_task import train_stream
        stream, _ = train_stream(args.steps * args.tokens, seed=args.seed, spacer=args.recall_spacer)
        phases = [dict(name="recall_train", start_update=0, stop_update=args.steps,
                       source_byte_start=0, source_byte_stop=args.steps * args.tokens)]
    elif args.stream == "mode":
        from scripts.ib_local.mode_task import train_stream as mode_stream
        stream, target_bytes, _ = mode_stream(args.steps * args.tokens + 1, seed=args.seed,
                                              easy=args.mode_easy_steps * args.tokens)
        target_bytes = target_bytes[:args.steps * args.tokens]
        phases = [dict(name="mode_train", start_update=0, stop_update=args.steps,
                       source_byte_start=0, source_byte_stop=args.steps * args.tokens)]
    else:
        stream, phases = protocol_stream(args.steps, args.tokens)
    if args.stream != "mode":
        target_bytes = stream[1:args.steps * args.tokens + 1]
    base = GatedDeltaNetLM(256, 128, 4, 4, args.recurrence, 256, 4, 32.0,
                           decay_init="log_timescale",
                           feedback_init=args.feedback_init if args.recurrence == "feedback_delta"
                           else "constant").cuda()
    if args.self_state != "off":
        from scripts.ib_local.self_state import add_self_state
        add_self_state(base, oracle=args.self_state == "oracle").cuda()  # after the base: shared weights match the control
    model = TableGDN(base, not args.no_table).cuda()
    table = None if args.no_table else ByteNgramFastTable(slots_log2=args.slots_log2,
                                                          storage=args.storage)
    if table is not None:
        ByteNgramFastTable(slots_log2=4).run(stream[:64])  # compile outside the loop
    runner = TableGraphTrainer(model, args.tokens, args.chunk_tokens)
    data = torch.from_numpy(stream.astype(np.int64)).cuda()[None]
    target_data = torch.from_numpy(target_bytes.astype(np.int64)).cuda()[None]
    phase_of = np.empty(args.steps, dtype=object)
    for p in phases:
        phase_of[p["start_update"]:p["stop_update"]] = p["name"]
    body = {p["name"]: 0.0 for p in phases}
    alone = {p["name"]: 0.0 for p in phases}
    counts = {p["name"]: 0 for p in phases}
    per_update, table_per_update = np.empty(args.steps), np.empty(args.steps)
    uniform = torch.full((1, args.tokens, 256), math.log(1 / 256), device="cuda")
    started = time.perf_counter()
    table_seconds = 0.0
    log = (args.output / "metrics.jsonl").open("w", encoding="utf-8")
    for step in range(args.steps):
        lo = step * args.tokens
        window = stream[lo:lo + args.tokens + 1]
        if table is not None:
            t0 = time.perf_counter()
            nll_table, probs = table.run(window, keep_probs=True)
            table_seconds += time.perf_counter() - t0
            logp = torch.from_numpy(np.log(np.maximum(probs, FLOOR))).cuda()[None]
            table_per_update[step] = nll_table.mean()
        else:
            logp, table_per_update[step] = uniform, math.log(256)
        loss, diagnostics = runner.step(data[:, lo:lo + args.tokens],
                                        target_data[:, lo:lo + args.tokens], logp)
        per_update[step] = float(loss)
        name = phase_of[step]
        body[name] += per_update[step]; alone[name] += table_per_update[step]; counts[name] += 1
        if (step + 1) % 50 == 0 or step == 0:
            row = dict(step=step + 1, phase=name, window_nll=float(per_update[max(0, step - 49):step + 1].mean()),
                       table_alone_nll=float(table_per_update[max(0, step - 49):step + 1].mean()),
                       bytes_per_second=(step + 1) * args.tokens / (time.perf_counter() - started),
                       **{k: float(v) for k, v in diagnostics.items()
                          if k.startswith("table") or k == "alpha_mean" or "self_beta" in k})
            log.write(json.dumps(row) + "\n"); log.flush()
    log.close()
    seconds = time.perf_counter() - started
    revisit = next((p for p in phases if p["name"] == "A_revisit"), None)
    if revisit is not None:
        first_lo = revisit["source_byte_start"] // args.tokens
        first_hi = revisit["source_byte_stop"] // args.tokens
        matched_first = float(per_update[first_lo:first_hi].mean())
    else:
        matched_first = float("nan")
    summary = dict(
        seed=args.seed, table=not args.no_table, slots_log2=None if args.no_table else args.slots_log2,
        table_mib=None if table is None else table.nbytes / 2**20,
        storage=str(args.storage) if args.storage else "ram",
        phases={k: body[k] / counts[k] for k in body},
        table_alone_phases=None if table is None else {k: alone[k] / counts[k] for k in alone},
        matched_first=matched_first,
        revisit_saving=(matched_first - body["A_revisit"] / counts["A_revisit"]
                        if revisit is not None else None),
        stream=args.stream, recurrence=args.recurrence, model_config=base.config_dict(),
        recall_spacer=args.recall_spacer if args.stream == "recall" else None,
        mode_easy_steps=args.mode_easy_steps if args.stream == "mode" else None,
        self_state=args.self_state,
        bytes_per_second=args.steps * args.tokens / seconds, table_seconds=table_seconds,
        parameters=sum(p.numel() for p in model.parameters()),
        state_bytes=model.initial_state(1).numel() * 4,
        fused_sha256=fused_sha, phases_spec=phases)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1))
    torch.save(dict(model=model.state_dict(), summary=summary, state=runner.state.detach().cpu()),
               args.output / "last.pt")
    if table is not None and args.save_table:
        np.savez(args.output / "table.npz", rows=table.rows, count=table.count, tag=table.tag,
                 err=table.err, history=np.uint64(table.history), slots_log2=args.slots_log2)
    if args.storage:
        del table
        shutil.rmtree(args.storage, ignore_errors=True)
    print(json.dumps({k: summary[k] for k in ("phases", "table_alone_phases", "revisit_saving",
                                                "bytes_per_second")}))


if __name__ == "__main__":
    main()
