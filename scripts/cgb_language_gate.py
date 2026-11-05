"""Default-deny admission gate for a language body.

`cgb_gate.py` checks only what a manifest chooses to claim.  This gate reads
the fixed list in `CGB_HARD_OBLIGATIONS.json` instead: every obligation must
run on the named body and pass.  A check that did not run counts as not met,
and system-4 obligations are reported as deferred, never as met.

The checks read behaviour through the `Candidate` interface, so they do not
favour one carrier.  Lean obligations are looked up from the body's manifest
and are expected to stay open until a finalist exists.

Usage:
    python scripts/cgb_language_gate.py --reference
    python scripts/cgb_language_gate.py fading_cells --budget-kib 1024 --json out.json
    python scripts/cgb_language_gate.py column --checkpoint latest.pt --summary summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from cgb_language_candidates import CANDIDATES, Candidate, ContextTable  # noqa: E402

OBLIGATIONS = ROOT / "docs" / "information_boltzmann" / "CGB_HARD_OBLIGATIONS.json"
ARCHITECTURES = ROOT / "information_boltzmann" / "architectures"
LEAN = ROOT / "lean" / "CGBAxioms"
INVENTORY = LEAN / ".lake" / "audit_inventory.json"
DATA = ROOT / "data" / "ib_owt_10000"
OTHER_SOURCE = ROOT / "data" / "material_owt_domains_20261028" / "train_B.npy"
REFERENCE = ROOT / "results" / "cgb_language_bakeoff_20261101" / "reference.npz"

PASS, FAIL, NOT_RUN, DEFERRED = "pass", "fail", "not_run", "deferred"
# A theorem whose premise the body does not meet, with the reason recorded in its manifest.
# Never reported as passed; it does not block admission.
NOT_APPLICABLE = "not_applicable"

# The same raw intervals as run_cgb_column_language_living.py.
SCHEDULE = (
    ("warmup", "train", 0, 100_000),
    ("A_firstpass", "train", 100_000, 200_000),
    ("B_firstpass", "validation", 0, 64_000),
    ("A_revisit", "train", 100_000, 164_000),
    ("A_fresh", "train", 200_000, 264_000),
)
SEEN_REPEATEDLY = (270_000, 286_000)
SEEN_ONCE = (286_000, 302_000)
FILLER = (302_000, 366_000)
UNSEEN = (366_000, 382_000)
SWITCH_EXPOSURE = (1_000_000, 1_048_000)
SWITCH_HELD_OUT = (1_048_000, 1_064_000)


def stream(split: str, start: int, stop: int) -> list[int]:
    """Real bytes of one raw interval; metadata tokens carry no lexical event."""
    values = np.asarray(np.load(DATA / f"{split}.npy", mmap_mode="r")[start:stop])
    return (values[values >= 4].astype(np.int64) - 4).tolist()


def other_source(start: int, stop: int) -> list[int]:
    """Real bytes from a different OpenWebText domain."""
    values = np.asarray(np.load(OTHER_SOURCE, mmap_mode="r")[start:stop])
    return (values[values >= 4].astype(np.int64) - 4).tolist()


def load_obligations() -> dict:
    return json.loads(OBLIGATIONS.read_text(encoding="utf-8"))


def load_inventory() -> tuple[set[str], dict]:
    """Return declaration names and whether they describe the current sources."""
    data = json.loads(INVENTORY.read_text(encoding="utf-8"))
    sources = sorted((LEAN / "CGBAxioms").rglob("*.lean")) + [LEAN / "CGBAxioms.lean"]
    current = {path.relative_to(LEAN).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sources}
    return {row["name"] for row in data["declarations"]}, {
        "fresh": current == data.get("source_hashes"),
        "trusted": bool(data.get("proof_trust_passed")) and bool(data.get("module_coverage_passed")),
    }


def load_manifest(body: str) -> dict | None:
    path = ARCHITECTURES / f"language_{body}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def unknown_theorems(obligations: dict, names: set[str]) -> list[str]:
    """Obligation citations that no longer exist in the audited inventory."""
    short = {name.rsplit(".", 1)[-1] for name in names}
    return sorted({theorem for row in obligations["obligations"]
                   for theorem in row["theorems"] if theorem not in short})


def declared_check(manifest: dict | None, row: dict, names: set[str]) -> tuple[str, str]:
    """Per cited theorem: a Lean declaration for this body, or why its premise does not hold."""
    entries = (manifest or {}).get("declared", {}).get(row["check"]["id"], {})
    proved, excused = [], []
    for theorem in row["theorems"]:
        entry = entries.get(theorem)
        if not entry:
            return FAIL, f"{theorem}: nothing declared"
        if entry.get("lean"):
            if entry["lean"] not in names:
                return FAIL, f"{theorem}: {entry['lean']} is not in the audit inventory"
            proved.append(theorem)
        elif entry.get("not_applicable"):
            excused.append(f"{theorem}: {entry['not_applicable']}")
        else:
            return FAIL, f"{theorem}: neither a declaration nor a reason"
    if excused:
        return NOT_APPLICABLE, "; ".join(excused) + (f"; proved {', '.join(proved)}" if proved else "")
    return PASS, ", ".join(proved)


def lean_check(manifest: dict | None, check_id: str, names: set[str]) -> tuple[str, str]:
    declared = (manifest or {}).get("lean", {}).get(check_id)
    if not declared:
        return FAIL, "no Lean declaration is named for this body"
    if declared not in names:
        return FAIL, f"{declared} is not in the audit inventory"
    return PASS, declared


class FrozenColumn(Candidate):
    """`CGBColumnLanguageBody` from a checkpoint, slow parameters held fixed.

    Its language scores come from the recorded live run.  Held fixed, it can
    only answer questions about its state, so one-shot learning is not asked.
    """

    name = "column"

    def __init__(self, checkpoint: Path) -> None:
        from information_boltzmann.core.cgb_column_body import (
            CGBColumnLanguageBody, ColumnBodyState,
        )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        engine = payload["engines"]["s_on"]
        config = engine["model_config"]
        self.body = CGBColumnLanguageBody(
            config["d_model"], tuple(config["grid_shape"]),
            active_columns=config["active_columns"], contact_rank=config["contact_rank"],
            byte_vocab=config["vocab_size"],
        )
        self.body.initial_precision = config["initial_precision"]
        self.body.load_state_dict(engine["body_state_dict"])
        self.body.eval()
        buffers = engine["buffers"]
        self.state = ColumnBodyState(*(buffers[name].clone() for name in (
            "activity", "natural", "precision", "causal_token", "participation",
        )))

    @torch.no_grad()
    def predict(self) -> torch.Tensor:
        state = self.state
        features = self.body._predict_tensors(state.activity, state.natural, state.precision,
                                              state.causal_token, state.participation)[0]
        return torch.log_softmax(self.body.readout(features)[0], dim=0)

    @torch.no_grad()
    def observe(self, byte: int) -> None:
        self.state = self.body.step(self.state, torch.tensor([byte]), elapsed=0.0).state

    def advance(self, elapsed: float) -> None:
        self.state = self.body.advance_time(self.state, elapsed)

    def state_vector(self) -> torch.Tensor:
        state = self.state
        return torch.cat((state.activity.flatten(), (state.natural / state.precision).flatten(),
                          state.precision.flatten(), state.participation.flatten()))

    def state_bytes(self) -> int:
        return (self.body.persistent_state_bytes(self.state)["total"]
                + 4 * sum(value.numel() for value in self.body.parameters()))

    def clone(self) -> dict:
        return {"state": self.body.clone_state(self.state)}

    def restore(self, saved: dict) -> None:
        self.state = self.body.clone_state(saved["state"])


def build_reference(budget_kib: int = 1024) -> dict:
    """Online order-0 and order-1 code lengths and the context-table yardstick."""
    counts0, counts1, totals1 = np.full(256, 0.05), np.full((256, 256), 0.05), np.full(256, 12.8)
    table, order = ContextTable(budget_kib), 4
    seen: set[tuple[int, ...]] = set()
    history: list[int] = []
    phases: dict[str, dict[str, float]] = {}
    novel, table_nll = [], []
    for name, split, start, stop in SCHEDULE:
        nll0, nll1, count = 0.0, 0.0, 0
        for byte in stream(split, start, stop):
            nll0 -= np.log(counts0[byte] / counts0.sum())
            previous = history[-1] if history else None
            nll1 -= (np.log(counts1[previous, byte] / totals1[previous])
                     if previous is not None else np.log(1 / 256))
            key = tuple(history[-order:])
            cost = -table.log_prob(byte)
            if name == "A_fresh":
                novel.append(key not in seen)
                table_nll.append(cost)
            table.observe(byte)
            seen.add(key)
            counts0[byte] += 1
            if previous is not None:
                counts1[previous, byte] += 1
                totals1[previous] += 1
            history = (history + [byte])[-order:]
            count += 1
        phases[name] = {"order0": nll0 / count, "order1": nll1 / count}
    return {"order0": phases["A_fresh"]["order0"], "order1": phases["A_fresh"]["order1"],
            "novel": np.array(novel), "table_nll": np.array(table_nll, dtype=np.float32)}


def load_reference() -> dict:
    if REFERENCE.exists():
        data = np.load(REFERENCE)
        return {key: (data[key].item() if data[key].ndim == 0 else data[key]) for key in data.files}
    reference = build_reference()
    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(REFERENCE, **reference)
    return reference


def score(candidate: Candidate, bytes_: list[int], *, keep: bool = False) -> tuple[float, np.ndarray | None]:
    values = np.zeros(len(bytes_), dtype=np.float32)
    for index, byte in enumerate(bytes_):
        values[index] = -candidate.log_prob(byte)
        candidate.observe(byte)
    return float(values.mean()), values if keep else None


def run_stream(candidate: Candidate) -> dict:
    """Run the common schedule with learning on, scoring before each write."""
    phases, windows, fresh = {}, {}, None
    began = time.time()
    total = 0
    for name, split, start, stop in SCHEDULE:
        bytes_ = stream(split, start, stop)
        total += len(bytes_)
        phases[name], values = score(candidate, bytes_, keep=True)
        windows[name] = [float(values[i:i + 8000].mean()) for i in range(0, len(values), 8000)]
        if name == "A_fresh":
            fresh = values
    return {"phases": phases, "windows": windows, "fresh_nll": fresh,
            "microseconds_per_byte": 1e6 * (time.time() - began) / total,
            "state_bytes": max(candidate.state_bytes(), candidate.capacity_bytes())}


def run_selective(candidate: Candidate) -> dict:
    """Text seen three times against text seen once, after saturating filler."""
    repeated = stream("train", *SEEN_REPEATEDLY)
    single = stream("train", *SEEN_ONCE)
    for segment in (repeated, single, repeated, repeated, stream("train", *FILLER)):
        score(candidate, segment)
    unseen = score(candidate, stream("train", *UNSEEN))[0]
    single_score = score(candidate, single)[0]
    repeated_score = score(candidate, repeated)[0]
    return {"retained_repeated": unseen - repeated_score, "retained_single": unseen - single_score}


def run_switch(candidate: Candidate) -> dict:
    """The same unseen bytes of another source, before and after reading that source."""
    held_out = other_source(*SWITCH_HELD_OUT)
    saved = candidate.clone()
    before = score(candidate, held_out)[0]
    candidate.restore(saved)
    score(candidate, other_source(*SWITCH_EXPOSURE))
    return {"before": before, "after": score(candidate, held_out)[0]}


def held_out_accuracy(features: torch.Tensor, text: torch.Tensor, lag: int) -> float:
    """Ridge decode of x(t-lag) on the first 80%, scored on the last 20%."""
    count = len(features)
    split = int(count * 0.8)
    mean, scale = features[:split].mean(0), features[:split].std(0) + 1e-5
    inputs = (features - mean) / scale
    targets = torch.roll(text[:count], lag)
    train_inputs, train_targets = inputs[lag:split], targets[lag:split]
    gram = (train_inputs.T @ train_inputs).double()
    gram.diagonal().add_(300.0)
    onehot = F.one_hot(train_targets, 256).float()
    weight = torch.linalg.solve(gram, (train_inputs.T @ (onehot - onehot.mean(0))).double()).float()
    prediction = (inputs[split:] @ weight).argmax(dim=1)
    return float((prediction == targets[split:]).float().mean())


def run_decode(candidate: Candidate, lags: list[int], *, events: int = 16_000) -> dict:
    tokens = torch.as_tensor(stream("train", 400_000, 440_000)[:events])
    width = candidate.state_vector().numel()
    generator = torch.Generator().manual_seed(2)
    sketch = torch.randn(width, 4096, generator=generator) if width > 4096 else None
    features = torch.zeros(events, min(width, 4096))
    for index in range(events):
        candidate.observe(int(tokens[index]))
        vector = candidate.state_vector().float()
        features[index] = vector if sketch is None else vector @ sketch
    size = 2048
    recurrent = torch.randn(size, size, generator=generator) / size ** 0.5 * 0.95
    encoder = torch.randn(256, size, generator=generator) * 0.6
    reservoir, trace = torch.zeros(size), torch.zeros(events, size)
    current = torch.zeros(events, 256)
    current[torch.arange(events), tokens] = 1.0
    for index in range(events):
        reservoir = 0.5 * reservoir + 0.5 * torch.tanh(recurrent @ reservoir + encoder[tokens[index]])
        trace[index] = reservoir
    return {str(lag): {"body": held_out_accuracy(features, tokens, lag),
                       "reservoir": held_out_accuracy(trace, tokens, lag),
                       "current_byte": held_out_accuracy(current, tokens, lag)} for lag in lags}


def jensen_shannon(p: torch.Tensor, q: torch.Tensor) -> float:
    mixture = ((p.exp() + q.exp()) / 2).clamp_min(1e-12).log()
    return float(0.5 * ((p.exp() * (p - mixture)).sum() + (q.exp() * (q - mixture)).sum()))


def run_repeatable(candidate: Candidate, *, events: int = 12_000) -> dict:
    """Do the same recent bytes reach the same prediction, whatever came before?"""
    text = np.array(stream("train", 440_000, 480_000)[:events + 1], dtype=np.int64)
    predictions = torch.zeros(events, 256)
    for index in range(events):
        candidate.observe(int(text[index]))
        predictions[index] = candidate.predict()
    groups: dict[bytes, list[int]] = {}
    last: dict[int, list[int]] = {}
    for index in range(500, events):
        groups.setdefault(text[index - 7:index + 1].tobytes(), []).append(index)
        last.setdefault(int(text[index]), []).append(index)
    rng = np.random.default_rng(0)
    same = [jensen_shannon(predictions[v[0]], predictions[v[1]]) for v in groups.values() if len(v) > 1]
    different = []
    for values in last.values():
        if len(values) > 1:
            for a, b in rng.integers(0, len(values), (16, 2)):
                x, y = values[int(a)], values[int(b)]
                if text[x - 1] != text[y - 1]:
                    different.append(jensen_shannon(predictions[x], predictions[y]))
    return {"same_eight": float(np.mean(same)) if same else None, "pairs": len(same),
            "last_byte_only": float(np.mean(different)) if different else None}


def run_one_shot(candidate: Candidate, *, pairs: int = 40) -> dict:
    """How far does one exposure move a rare context against a frequent one?"""
    seen: dict[tuple[int, ...], int] = {}
    for name, split, start, stop in SCHEDULE:
        if name != "A_revisit":
            bytes_ = stream(split, start, stop)
            for index in range(2, len(bytes_)):
                key = tuple(bytes_[index - 2:index + 1])
                seen[key] = seen.get(key, 0) + 1
    text = stream("train", 480_000, 520_000)
    rare, frequent = [], []
    for index in range(16, len(text) - 1):
        count = seen.get(tuple(text[index - 2:index + 1]), 0)
        if count <= 2 and len(rare) < pairs:
            rare.append(index)
        elif count >= 200 and index % 7 == 0 and len(frequent) < pairs:
            frequent.append(index)

    def shift(index: int) -> float:
        saved = candidate.clone()
        context, target = text[index - 8:index + 1], text[index + 1]
        for byte in context:
            candidate.observe(byte)
        before = candidate.log_prob(target)
        candidate.observe(target)
        for byte in context:
            candidate.observe(byte)
        after = candidate.log_prob(target)
        candidate.restore(saved)
        return after - before

    return {"rare": float(np.median([shift(i) for i in rare])),
            "frequent": float(np.median([shift(i) for i in frequent]))}


def run_interaction(candidate: Candidate, probes: int) -> dict:
    """Interaction contrast of the next byte's log-probability over the two previous bytes.

    From one saved state the body reads the same 14 bytes, then each of the four
    combinations of (byte two back, byte one back) in {original, swapped}, and
    scores the true next byte.  Learning stays on; the state is restored after
    each combination.
    """
    text = stream("train", 540_000, 600_000)
    rng = np.random.default_rng(5)
    positions = list(range(64, len(text) - 1, (len(text) - 65) // probes))[:probes]
    contrasts, singles = [], []
    for t in positions:
        a, b, target = text[t - 2], text[t - 1], text[t]
        a_alt, b_alt = a, b
        while a_alt == a:
            a_alt = text[int(rng.integers(0, len(text)))]
        while b_alt == b:
            b_alt = text[int(rng.integers(0, len(text)))]
        saved = candidate.clone()
        values = {}
        for first in (a, a_alt):
            for second in (b, b_alt):
                for byte in (*text[t - 16:t - 2], first, second):
                    candidate.observe(byte)
                values[first, second] = candidate.log_prob(target)
                candidate.restore(saved)
        contrasts.append(values[a, b] - values[a, b_alt] - values[a_alt, b] + values[a_alt, b_alt])
        singles.extend((values[a, b] - values[a_alt, b], values[a, b] - values[a, b_alt]))
    return {"interaction": float(np.mean(np.abs(contrasts))), "single": float(np.mean(np.abs(singles))),
            "probes": len(positions)}


def check_causal(candidate: Candidate) -> tuple[str, str]:
    before = candidate.state_vector().clone()
    first, second = candidate.predict(), candidate.predict()
    ok = torch.equal(first, second) and torch.equal(before, candidate.state_vector())
    return (PASS if ok else FAIL), "reading a prediction changes nothing" if ok else "prediction moved state"


def check_resume(candidate: Candidate, text: list[int]) -> tuple[str, str]:
    saved = candidate.clone()
    first = []
    for byte in text[:64]:
        first.append(candidate.predict())
        candidate.observe(byte)
    candidate.restore(saved)
    equal = True
    for index, byte in enumerate(text[:64]):
        equal = equal and torch.equal(first[index], candidate.predict())
        candidate.observe(byte)
    return (PASS if equal else FAIL), "replay is exact" if equal else "replay diverged"


def check_elapsed(candidate: Candidate) -> tuple[str, str]:
    saved = candidate.clone()
    before = candidate.predict()
    candidate.advance(0.0)
    still = torch.equal(before, candidate.predict())
    candidate.advance(500.0)
    moved = not torch.equal(before, candidate.predict())
    candidate.restore(saved)
    return (PASS if still and moved else FAIL), f"zero elapsed no-op={still}, elapsed acts={moved}"


def judge(check_id: str, evidence: dict, thresholds: dict, reference: dict | None,
          budget_kib: int | None) -> tuple[str, str]:
    """Turn recorded evidence into a status.  Missing evidence is never a pass."""
    phases = evidence.get("phases") or {}
    if check_id == "finite_state":
        size = evidence.get("state_bytes")
        if size is None or budget_kib is None:
            return NOT_RUN, "no size against a declared budget"
        return (PASS if size <= budget_kib * 1024 * 1.25 else FAIL), f"{size} bytes at worst for {budget_kib} KiB"
    if check_id == "beats_order1":
        if "A_fresh" not in phases or reference is None:
            return NOT_RUN, "no live A_fresh score"
        floor = reference["order1"] - thresholds["floor_margin_nats"]
        return (PASS if phases["A_fresh"] < floor else FAIL), f"{phases['A_fresh']:.4f} vs order-1 {reference['order1']:.4f}"
    if check_id == "language_pays":
        if "A_fresh" not in phases or reference is None:
            return NOT_RUN, "no live A_fresh score"
        saved = reference["order0"] - phases["A_fresh"]
        return (PASS if saved >= thresholds["pays_margin_nats"] else FAIL), f"saves {saved:.4f} against order-0"
    if check_id == "revisit_drop":
        if "A_firstpass" not in phases or "A_revisit" not in phases:
            return NOT_RUN, "no revisit score"
        drop = phases["A_firstpass"] - phases["A_revisit"]
        return (PASS if drop >= thresholds["revisit_drop_nats"] else FAIL), f"drop {drop:.4f}"
    if check_id == "drift_recovery":
        row = evidence.get("switch")
        if row is None or "A_revisit" not in phases:
            return NOT_RUN, "no source switch"
        gain = row["before"] - row["after"]
        keeps = phases["A_revisit"] <= phases["A_firstpass"]
        ok = gain >= thresholds["adapt_margin_nats"] and keeps
        return (PASS if ok else FAIL), f"held-out gain {gain:.4f}, return not worse={keeps}"
    if check_id == "generalizes":
        values = evidence.get("fresh_nll")
        if values is None or reference is None:
            return NOT_RUN, "no per-byte scores"
        mask = reference["novel"]
        body, table = float(values[mask].mean()), float(reference["table_nll"][mask].mean())
        return (PASS if body < table else FAIL), f"{body:.4f} vs table {table:.4f} on {int(mask.sum())} novel-context bytes"
    if check_id == "capacity_curve":
        if evidence.get("small_revisit") is None or "A_revisit" not in phases:
            return NOT_RUN, "no smaller-budget run"
        ok = phases["A_revisit"] < evidence["small_revisit"]
        return (PASS if ok else FAIL), f"revisit {phases['A_revisit']:.4f} vs {evidence['small_revisit']:.4f} at a quarter budget"
    if check_id == "selective_retention":
        row = evidence.get("selective")
        if row is None:
            return NOT_RUN, "selective protocol did not run"
        margin = row["retained_repeated"] - row["retained_single"]
        ok = margin >= thresholds["selective_margin"] and row["retained_repeated"] > thresholds["selective_margin"]
        return (PASS if ok else FAIL), f"repeated {row['retained_repeated']:.4f}, single {row['retained_single']:.4f}"
    if check_id == "no_dominance":
        shares = evidence.get("shares")
        if shares is None:
            return NOT_RUN, "this body reports no regions"
        positive = shares[shares > 0]
        effective = float(torch.exp(-(positive * positive.log()).sum()))
        fraction = effective / shares.numel()
        ok = (float(shares.max()) <= thresholds["dominance_max_share"]
              and fraction >= thresholds["dominance_min_effective_fraction"])
        return (PASS if ok else FAIL), f"largest share {float(shares.max()):.3f}, effective fraction {fraction:.3f}"
    if check_id == "one_shot_gain":
        row = evidence.get("one_shot")
        if row is None:
            return NOT_RUN, "no learning run"
        ok = (row["rare"] >= thresholds["one_shot_min"]
              and row["rare"] >= thresholds["one_shot_ratio"] * max(row["frequent"], 0.0))
        return (PASS if ok else FAIL), f"rare {row['rare']:.4f}, frequent {row['frequent']:.4f}"
    if check_id == "state_repeatable":
        row = evidence.get("repeatable")
        if row is None or row["same_eight"] is None or row["last_byte_only"] is None:
            return NOT_RUN, "too few repeated contexts"
        ok = (row["same_eight"] <= thresholds["repeat_ratio"] * row["last_byte_only"]
              and row["last_byte_only"] >= thresholds["separation_js"])
        return (PASS if ok else FAIL), f"same 8 bytes {row['same_eight']:.4f}, last byte only {row['last_byte_only']:.4f}"
    if check_id == "decode_horizon":
        rows = evidence.get("decode")
        if rows is None:
            return NOT_RUN, "no decoding run"
        ok = all(rows[str(lag)]["body"] >= rows[str(lag)]["reservoir"] - thresholds["decode_margin_accuracy"]
                 for lag in thresholds["decode_lags"])
        text = "; ".join(f"t-{lag} {rows[str(lag)]['body']:.3f}/{rows[str(lag)]['reservoir']:.3f}"
                         for lag in thresholds["decode_lags"])
        return (PASS if ok else FAIL), text
    if check_id == "credit_has_carrier":
        rows = evidence.get("decode")
        if rows is None:
            return NOT_RUN, "no decoding run"
        row = rows[str(thresholds["carrier_lag"])]
        ok = row["body"] >= row["current_byte"] + thresholds["carrier_margin"]
        return (PASS if ok else FAIL), f"{row['body']:.3f} vs current byte {row['current_byte']:.3f}"
    if check_id == "interaction":
        row = evidence.get("interaction")
        if row is None:
            return NOT_RUN, "no interaction probes"
        ok = (row["interaction"] >= thresholds["interaction_min_nats"]
              and row["interaction"] >= thresholds["interaction_ratio"] * row["single"])
        return (PASS if ok else FAIL), (f"mean |contrast| {row['interaction']:.4f}, mean |single swap| "
                                        f"{row['single']:.4f} over {row['probes']} probes")
    return NOT_RUN, "this check is not implemented"


def collect_live(name: str, budget_kib: int) -> tuple[dict, Candidate]:
    thresholds = load_obligations()["language_thresholds"]
    builder = CANDIDATES[name]
    evidence = {"small_revisit": run_stream(builder(max(16, budget_kib // 4)))["phases"]["A_revisit"]}
    candidate = builder(budget_kib)
    evidence.update(run_stream(candidate))
    evidence["shares"] = candidate.load_shares()
    evidence["selective"] = run_selective(candidate)
    lags = sorted({*thresholds["decode_lags"], thresholds["carrier_lag"]})
    evidence["decode"] = run_decode(candidate, lags)
    evidence["repeatable"] = run_repeatable(candidate)
    evidence["one_shot"] = run_one_shot(candidate)
    evidence["switch"] = run_switch(candidate)
    evidence["interaction"] = run_interaction(candidate, thresholds["interaction_probes"])
    return evidence, candidate


def collect_column(checkpoint: Path, summary_path: Path | None) -> tuple[dict, Candidate]:
    thresholds = load_obligations()["language_thresholds"]
    candidate = FrozenColumn(checkpoint)
    evidence: dict = {"state_bytes": candidate.state_bytes()}
    if summary_path is not None:
        rows = json.loads(summary_path.read_text(encoding="utf-8"))["phases"]
        for phase in ("A_firstpass", "A_revisit", "A_fresh"):
            row = rows.get(f"{phase}/s_on")
            if row:
                evidence.setdefault("phases", {})[phase] = row["total_nll_nats"] / row["scored_bytes"]
    lags = sorted({*thresholds["decode_lags"], thresholds["carrier_lag"]})
    evidence["decode"] = run_decode(candidate, lags)
    evidence["repeatable"] = run_repeatable(candidate)
    return evidence, candidate


def evaluate(body: str, evidence: dict, candidate: Candidate | None, budget_kib: int | None,
             reference: dict | None, names: set[str], inventory: dict) -> dict:
    obligations = load_obligations()
    manifest = load_manifest(body)
    thresholds = obligations["language_thresholds"]
    text = stream("train", 530_000, 531_000)
    rows = []
    for row in obligations["obligations"]:
        kind, check_id = row["check"]["kind"], row["check"]["id"]
        if kind == "deferred":
            status, detail = DEFERRED, "applies after system 4 is connected"
        elif kind == "lean":
            status, detail = lean_check(manifest, check_id, names)
        elif kind == "declared":
            status, detail = declared_check(manifest, row, names)
        elif check_id == "causal_prediction":
            status, detail = check_causal(candidate) if candidate else (NOT_RUN, "no body")
        elif check_id == "exact_resume":
            status, detail = check_resume(candidate, text) if candidate else (NOT_RUN, "no body")
        elif check_id == "elapsed_time":
            status, detail = check_elapsed(candidate) if candidate else (NOT_RUN, "no body")
        else:
            status, detail = judge(check_id, evidence, thresholds, reference, budget_kib)
        rows.append(dict(id=row["id"], system=row["system"], title=row["title"],
                         check=check_id, status=status, detail=detail))
    open_rows = [row["id"] for row in rows if row["status"] in (FAIL, NOT_RUN)]
    unknown = unknown_theorems(obligations, names)
    reference_only = bool(candidate is not None and candidate.reference_only)
    admitted = (not open_rows and not unknown and inventory["fresh"] and inventory["trusted"]
                and manifest is not None and not reference_only)
    return {
        "body": body, "admitted": admitted, "reference_only": reference_only,
        "manifest_present": manifest is not None, "inventory": inventory,
        "unknown_theorems": unknown, "budget_kib": budget_kib,
        "state_bytes": evidence.get("state_bytes"),
        "microseconds_per_byte": evidence.get("microseconds_per_byte"),
        "phases": evidence.get("phases"),
        "counts": {name: sum(row["status"] == name for row in rows)
                   for name in (PASS, FAIL, NOT_RUN, DEFERRED, NOT_APPLICABLE)},
        "open_obligations": open_rows, "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("body", nargs="?")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--budget-kib", type=int, default=1024)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    reference = load_reference()
    if args.reference:
        print({key: value for key, value in reference.items() if not isinstance(value, np.ndarray)},
              "novel", int(reference["novel"].sum()), "of", len(reference["novel"]))
        return
    names, inventory = load_inventory()
    if args.body == "column":
        evidence, candidate = collect_column(args.checkpoint, args.summary)
        budget = None
    else:
        evidence, candidate = collect_live(args.body, args.budget_kib)
        budget = args.budget_kib
    report = evaluate(args.body, evidence, candidate, budget, reference, names, inventory)
    for row in report["rows"]:
        print(f"{row['id']} [{row['status']:8s}] {row['title']}: {row['detail']}")
    print("ADMITTED" if report["admitted"] else "REJECTED", report["counts"], flush=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    raise SystemExit(0 if report["admitted"] else 1)


if __name__ == "__main__":
    main()
