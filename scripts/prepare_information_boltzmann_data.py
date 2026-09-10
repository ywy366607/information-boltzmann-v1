"""Prepare checksum-pinned ordered token streams, splitting documents first."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import unicodedata

import numpy as np


SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["text", "openwebtext"], required=True)
    parser.add_argument("--text", type=Path, help="UTF-8 text: each nonempty line is a document")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-documents", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=260)
    parser.add_argument("--tokenizer", choices=["byte", "bpe"], default="byte")
    parser.add_argument("--revision", help="Dataset revision, resolved to immutable SHA")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use a new output directory to preserve prepared data provenance")
    revision, data_files = None, []
    if args.source == "text":
        if args.text is None:
            parser.error("--text is required for --source text")
        rows = ({"text": line} for line in args.text.read_text(encoding="utf-8").splitlines() if line.strip())
    else:
        # Some Windows stores contain malformed certificates. Keep verification
        # enabled, using certifi's CA bundle only if the system context fails.
        import ssl
        try:
            ssl.create_default_context()
        except ssl.SSLError:
            import certifi
            from functools import partial
            ssl.create_default_context = partial(ssl.create_default_context, cafile=certifi.where())
        from datasets import load_dataset
        from huggingface_hub import HfApi
        api = HfApi()
        repo = "Skylion007/openwebtext"
        # Converted Parquet is data-only; never load legacy remote Python scripts.
        revision = api.dataset_info(repo, revision=args.revision or "refs/convert/parquet").sha
        names = api.list_repo_files(repo, repo_type="dataset", revision=revision)
        data_files = sorted(name for name in names if name.endswith(".parquet") and "/train/" in name)
        if not data_files:
            raise RuntimeError("No train Parquet shards at the selected revision")
        urls = [f"hf://datasets/{repo}@{revision}/{name}" for name in data_files]
        def iter_shards():
            # Resolve only shards actually consumed. Resolving the entire corpus
            # up front defeats a four-document smoke and creates excess requests.
            for url in urls:
                yield from load_dataset("parquet", data_files={"train": [url]}, split="train", streaming=True)
        rows = iter_shards()
    splits = {"train": [], "validation": [], "test": []}
    seen, records = set(), []
    for index, row in enumerate(rows):
        if index >= args.max_documents:
            break
        text = unicodedata.normalize("NFC", row["text"]).replace("\r\n", "\n").strip()
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if not text or digest in seen:
            continue
        seen.add(digest)
        bucket = int(digest, 16) % 100
        split = "train" if bucket < 98 else "validation" if bucket == 98 else "test"
        splits[split].append(text)
        records.append({"sha256": digest, "split": split, "source_index": index})
    if not splits["train"]:
        raise ValueError("No training documents selected")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.tokenizer == "bpe":
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.train_from_iterator(splits["train"], trainers.BpeTrainer(
            vocab_size=args.vocab_size, special_tokens=SPECIAL,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        tokenizer.save(str(args.output / "tokenizer.json"))
        encode = lambda s: tokenizer.encode(s).ids
        vocab = tokenizer.get_vocab_size()
    else:
        if args.vocab_size != 260:
            raise ValueError("Byte smoke tokenizer has exactly 256+4 tokens")
        (args.output / "tokenizer.json").write_text(json.dumps({"type": "utf8_bytes", "offset": 4,
                                                               "special_tokens": SPECIAL}), encoding="utf-8")
        encode = lambda s: [b + 4 for b in s.encode("utf-8")]
        vocab = 260
    files, sizes = {}, {}
    for split, documents in splits.items():
        ids = []
        for doc in documents:
            ids.extend(encode(doc))
            ids.extend([2, 1])  # EOS/BOS mark document boundaries without resetting phase state
        path = args.output / f"{split}.npy"
        np.save(path, np.asarray(ids, dtype=np.int64))
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        sizes[split] = len(ids)
    files["tokenizer.json"] = hashlib.sha256((args.output / "tokenizer.json").read_bytes()).hexdigest()
    manifest = {"source": args.source, "revision": revision, "parquet_shards": data_files,
                "tokenizer": args.tokenizer, "vocab_size": vocab, "bos_token": 1,
                "split_rule": "NFC_document_sha256_mod100_98_1_1",
                "documents": records, "token_counts": sizes, "files": files}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"revision": revision, "token_counts": sizes, "vocab_size": vocab}))


if __name__ == "__main__":
    main()
