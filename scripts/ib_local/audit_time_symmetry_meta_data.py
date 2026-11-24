"""Read-only corpus accounting and algebra checks; no body or world execution.

The 2x2 identities check the native decay-then-delta formula, not a synthetic
capability task. Corpus payload counts exclude tokenizer boundary markers.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import tiktoken


def audit(root: Path) -> dict:
    encoding = tiktoken.get_encoding("gpt2")
    byte_lengths = np.zeros(encoding.n_vocab, dtype=np.int64)
    for token in range(encoding.n_vocab):
        if token != encoding.eot_token:
            byte_lengths[token] = len(encoding.decode_single_token_bytes(token))
    corpora, manifests = {}, {}
    for name in ("ib_owt_10000", "ib_owt_gpt2", "ib_owt_gpt2_31m"):
        folder = root / "data" / name
        manifest_path = folder / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifests[name] = manifest
        docs = manifest["documents"]
        splits = {}
        for split in ("train", "validation", "test"):
            path = folder / f"{split}.npy"
            tokens = np.load(path, mmap_mode="r")
            if manifest["tokenizer"] == "byte":
                payload = int(np.count_nonzero(tokens >= 4))
            else:
                frequencies = np.bincount(tokens, minlength=encoding.n_vocab)
                payload = int(frequencies @ byte_lengths)
            splits[split] = dict(entries=len(tokens), dtype=str(tokens.dtype),
                                 disk_bytes=path.stat().st_size,
                                 text_payload_bytes=payload)
        corpora[name] = dict(
            manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            tokenizer=manifest["tokenizer"], documents=len(docs),
            unique_documents=len({d["sha256"] for d in docs}),
            document_splits=dict(Counter(d["split"] for d in docs)), splits=splits)
    original = {d["sha256"] for d in manifests["ib_owt_10000"]["documents"]}
    extended = {d["sha256"] for d in manifests["ib_owt_gpt2_31m"]["documents"]}
    extended_train = {d["sha256"] for d in manifests["ib_owt_gpt2_31m"]["documents"]
                      if d["split"] == "train"}
    old_train = np.load(root / "data/ib_owt_gpt2/train.npy", mmap_mode="r")
    new_train = np.load(root / "data/ib_owt_gpt2_31m/train.npy", mmap_mode="r")
    prefix_equal = bool(np.array_equal(old_train, new_train[:len(old_train)]))
    raw = np.load(root / "data/ib_owt_10000/train.npy", mmap_mode="r")
    world_payload = int(np.count_nonzero(raw[8_000_000:56_000_000] >= 4))
    path = root / "results/cgb_content_question_20261114/learning/body/metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    prepared_repeated = sum(r["world"]["response_previously_seen_bytes"] +
                            r["world"]["holdout_previously_seen_bytes"] for r in rows)
    prepared_total = sum(r["external_bytes"] for r in rows)
    mother_bytes = sum(r["world"]["mother_bytes"] for r in rows)
    # The mother replaces an ambient prefix, and take() marks that prefix seen
    # even though it was not received. Aggregate logs cannot undo that history.
    repeat = dict(cycles=len(rows), first_cycle=rows[0]["cycle"],
                  last_cycle=rows[-1]["cycle"], prepared_ambient_bytes=prepared_total,
                  prepared_ambient_previously_marked_bytes=prepared_repeated,
                  prepared_marked_repeat_fraction=prepared_repeated/prepared_total,
                  mother_bytes=mother_bytes,
                  actually_received_owt_bytes=prepared_total-mother_bytes,
                  note="Prepared-position repeat is not actual consumed-text repeat: "
                       "the base marks positions before mother prefix replacement. "
                       "The logs do not identify all earlier overwritten positions.")
    # Native unprojected transition A=alpha*(I-beta*k*k.T).
    a, b = np.array([1., 0.]), np.array([1., 1.])/np.sqrt(2)
    alpha_a, alpha_b, beta_a, beta_b = .9, .8, .6, .7
    pa, pb = np.outer(a, a), np.outer(b, b)
    aa = alpha_a*(np.eye(2)-beta_a*pa)
    ab = alpha_b*(np.eye(2)-beta_b*pb)
    lhs = ab@aa-aa@ab
    rhs = alpha_a*alpha_b*beta_a*beta_b*(pb@pa-pa@pb)
    checks = dict(commutator_norm=float(np.linalg.norm(lhs)),
                  identity_max_error=float(np.abs(lhs-rhs).max()),
                  operator_norms=[float(np.linalg.norm(x, 2)) for x in (aa, ab)],
                  scope="Fixed gate/key native matrix step before capacity admission; "
                        "not the coupled S/P, memory, learning or whole organism map.")
    assert checks["identity_max_error"] < 1e-14
    assert checks["commutator_norm"] > 0
    assert checks["operator_norms"][0] <= alpha_a+1e-14
    assert checks["operator_norms"][1] <= alpha_b+1e-14
    return dict(schema=1, mode="read_only_no_body_no_training", corpora=corpora,
                overlap=dict(common_documents=len(original & extended),
                             extended_only_documents=len(extended-original),
                             extended_only_training_documents=len(extended_train-original),
                             original_gpt2_prefix_preserved=prefix_equal,
                             additional_training_payload_bytes=(
                                 corpora["ib_owt_gpt2_31m"]["splits"]["train"]["text_payload_bytes"]-
                                 corpora["ib_owt_gpt2"]["splits"]["train"]["text_payload_bytes"])),
                current_world=dict(requested_raw_bounds=[8_000_000, 56_000_000],
                                   actual_raw_end=len(raw), payload_bytes=world_payload),
                recent_exposure=repeat, native_algebra=checks,
                training_updates=0, world_calls=0, checkpoint_loaded=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path,
                        default=Path("results/cgb_time_symmetry_meta_data_20261114/report.json"))
    args = parser.parse_args()
    report = audit(args.root)
    output = args.output if args.output.is_absolute() else args.root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
