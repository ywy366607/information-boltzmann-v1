"""Real ShareGPT-4o data adapters for the North-Star multimodal graph.

The adapters describe capabilities as boundary conditions.  They do not add
task-private encoders or decoders: every sample becomes a persistent image
field, an observed/missing language prefix, and terminal image/token
likelihoods consumed by :class:`fine_grain.omni_model.DualStreamOmni`.
"""
from __future__ import annotations

import json
import re
import struct
import tarfile
import zlib
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image


FREEDOM_REPO = "FreedomIntelligence/ShareGPT-4o-Image"
OPENGV_REPO = "OpenGVLab/ShareGPT-4o"
REAL_TASKS = ("t2i", "it2i", "i2t", "it2t")


def _read_json(path: str | Path) -> list[dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"expected a JSON list in {path}")
    return [dict(row) for row in value]


def load_freedom_manifest(
    path: str | Path,
    task: str,
    *,
    max_prompt_chars: int | None = None,
    same_resolution: bool = False,
) -> list[dict[str, Any]]:
    """Normalize the official T2I or single-source IT2I manifest."""
    task = str(task).lower()
    if task not in ("t2i", "it2i"):
        raise ValueError(f"Freedom manifest does not provide task={task!r}")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(_read_json(path)):
        prompt = str(raw.get("input_prompt", "")).strip()
        if not prompt:
            continue
        if max_prompt_chars is not None and len(prompt) > int(max_prompt_chars):
            continue
        output = str(raw.get("output_image", "")).replace("\\", "/")
        out_res = tuple(int(x) for x in raw.get("output_image_resolution", ()))
        if not output or len(out_res) != 2:
            continue
        source: list[str] = []
        in_res: tuple[int, ...] = ()
        if task == "it2i":
            source = [
                str(x).replace("\\", "/")
                for x in (raw.get("input_image") or [])
            ]
            in_res = tuple(int(x) for x in raw.get("input_image_resolution", ()))
            if len(source) != 1 or len(in_res) != 2:
                continue
            if same_resolution and in_res != out_res:
                continue
        rows.append({
            "id": f"freedom-{task}-{index}",
            "dataset": FREEDOM_REPO,
            "task": task,
            "prompt": prompt,
            "answer": "",
            "source_members": source,
            "target_member": output,
            "source_resolution": list(in_res),
            "target_resolution": list(out_res),
        })
    return rows


def _conversation_role(turn: dict[str, Any]) -> str:
    return str(turn.get("from", turn.get("role", ""))).lower()


def _conversation_text(turn: dict[str, Any]) -> str:
    value = turn.get("value", turn.get("content", ""))
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts).strip()
    return str(value).strip()


def _strip_image_marker(text: str) -> str:
    text = re.sub(r"<image(?:_\d+)?>", " ", text, flags=re.IGNORECASE)
    return " ".join(text.split()).strip()


def _is_caption_request(prompt: str) -> bool:
    value = prompt.lower().strip(" .?!")
    markers = (
        "describe the image", "describe this image", "caption the image",
        "provide a caption", "what is in the image", "what do you see",
        "describe what this image", "describe the situation in the picture",
        "describe the picture", "describe what is happening",
        "what scene is mainly depicted",
    )
    if any(marker in value for marker in markers):
        return True
    # OpenGVLab contains many paraphrases of an unconditional caption request.
    # These words are task scaffolding, not semantic evidence about the target;
    # keeping them as IT2T lets a prompt template masquerade as a second
    # observed modality. Match only broad whole-image requests. Questions about
    # a named attribute/object (color, count, OCR, relation, etc.) remain IT2T.
    generic_patterns = (
        r"\bdescribe\b.*\b(image|picture|scene)\b",
        r"\bdescription\b.*\b(image|picture|scene)\b",
        r"\bdescribe\b.*\b(objects?|people|characters?|elements?|details?)\b",
        r"\b(description|describe)\b.*\b(main elements?|everything)\b",
        r"\b(explain|elaborate)\b.*\b(scene|image|picture)\b",
        r"\b(in[- ]depth analysis|analy[sz]e)\b.*\b(scene|image|picture)\b",
        r"\bwhat\b.*\b(objects?|items?|people|characters?)\b.*\b(picture|image)\b",
        r"\bwhat (scene|details?|elements?)\b.*\b(picture|image)\b",
        r"\bwhat is (compelling|striking|prominent)\b.*\b(image|picture)\b",
        r"\bwhat are (the )?(compelling|striking|prominent)\b.*\b(image|picture)\b",
        r"\bwhat is (shown|depicted|happening)\b.*\b(image|picture)\b",
        r"\beverything in (the|this) (image|picture)\b",
        r"\bcontent of (a|the|this|given) (image|picture)\b",
        r"\blist all\b.*\b(objects?|items?|people|characters?)\b.*\b(image|picture)\b",
    )
    return any(re.search(pattern, value) for pattern in generic_patterns)


def iter_opengv_conversations(path: str | Path) -> Iterable[dict[str, Any]]:
    """Yield normalized first-turn image conversations from the gated JSONL."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            raw = json.loads(line)
            images = raw.get("image", raw.get("images", []))
            if isinstance(images, str):
                images = [images]
            images = [str(x).replace("\\", "/") for x in (images or [])]
            if len(images) != 1:
                continue
            turns = raw.get("conversations", raw.get("messages", []))
            if not isinstance(turns, list):
                continue
            pair = None
            for left, right in zip(turns, turns[1:]):
                if not isinstance(left, dict) or not isinstance(right, dict):
                    continue
                if _conversation_role(left) in ("human", "user") and _conversation_role(
                    right
                ) in ("gpt", "assistant"):
                    pair = (left, right)
                    break
            if pair is None:
                continue
            prompt = _strip_image_marker(_conversation_text(pair[0]))
            answer = _conversation_text(pair[1])
            if not answer:
                continue
            image_only = _is_caption_request(prompt)
            yield {
                "id": f"opengv-{index}",
                "dataset": OPENGV_REPO,
                "task": "i2t" if image_only else "it2t",
                "prompt": "" if image_only else prompt,
                "source_members": images,
                "target_member": "",
                "answer": answer,
            }


def load_opengv_manifest(
    path: str | Path,
    *,
    max_rows: int | None = None,
    max_answer_chars: int | None = None,
    max_per_task: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts = {"i2t": 0, "it2t": 0}
    for row in iter_opengv_conversations(path):
        if max_answer_chars is not None and len(row["answer"]) > int(max_answer_chars):
            continue
        task = str(row["task"])
        if max_per_task is not None and counts[task] >= int(max_per_task):
            continue
        rows.append(row)
        counts[task] += 1
        if max_per_task is not None and all(
            count >= int(max_per_task) for count in counts.values()
        ):
            break
        if max_rows is not None and len(rows) >= int(max_rows):
            break
    return rows


def _safe_member_name(name: str) -> str:
    value = name.replace("\\", "/").lstrip("./")
    if not value or value.startswith("/") or ".." in value.split("/"):
        raise ValueError(f"unsafe archive member {name!r}")
    return value


def extract_referenced_tar(
    archive: str | Path,
    records: Sequence[dict[str, Any]],
    output_dir: str | Path,
    *,
    max_complete: int,
) -> list[dict[str, Any]]:
    """Extract the first complete referenced samples from a full/partial tar.

    Uncompressed HF tar files can be downloaded with an HTTP Range prefix.
    Streaming mode accepts a prefix ending mid-member and retains every
    complete example encountered before that boundary.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    member_to_rows: dict[str, set[int]] = {}
    needed: list[set[str]] = []
    for row_index, row in enumerate(records):
        names = set(row.get("source_members", ()))
        target = str(row.get("target_member", ""))
        if target:
            names.add(target)
        clean = {_safe_member_name(x) for x in names if x}
        needed.append(clean)
        for name in clean:
            member_to_rows.setdefault(name, set()).add(row_index)

    found: list[set[str]] = [set() for _ in records]
    local: list[dict[str, str]] = [dict() for _ in records]
    complete: list[int] = []
    try:
        with tarfile.open(archive, mode="r|*") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                name = _safe_member_name(member.name)
                row_ids = member_to_rows.get(name)
                if not row_ids:
                    continue
                source = tar.extractfile(member)
                if source is None:
                    continue
                payload = source.read()
                suffix = Path(name).suffix.lower() or ".bin"
                for row_index in row_ids:
                    destination = output / f"{records[row_index]['id']}-{len(local[row_index])}{suffix}"
                    destination.write_bytes(payload)
                    local[row_index][name] = str(destination.resolve())
                    found[row_index].add(name)
                    if needed[row_index] == found[row_index] and row_index not in complete:
                        complete.append(row_index)
                if len(complete) >= int(max_complete):
                    break
    except (tarfile.ReadError, EOFError):
        # Expected when ``archive`` is an HTTP Range prefix.
        pass

    result: list[dict[str, Any]] = []
    for row_index in complete[: int(max_complete)]:
        row = deepcopy(records[row_index])
        mapping = local[row_index]
        row["source_files"] = [mapping[x] for x in row.get("source_members", ())]
        target = str(row.get("target_member", ""))
        row["target_file"] = mapping.get(target, "")
        result.append(row)
    return result


def read_zip_directory(fetch, size: int) -> dict[str, dict[str, int]]:
    """Read a normal or ZIP64 central directory through byte-range ``fetch``.

    ``fetch(start, stop)`` must return the half-open byte range.  Keeping this
    transport-agnostic lets the gated 6.5 GB OpenGVLab archive be sampled
    without downloading it in full.
    """
    size = int(size)
    tail_start = max(0, size - 131072)
    tail = fetch(tail_start, size)
    eocd_at = tail.rfind(b"PK\x05\x06")
    if eocd_at < 0:
        raise ValueError("ZIP end-of-central-directory record not found")
    eocd = tail[eocd_at : eocd_at + 22]
    if len(eocd) < 22:
        raise ValueError("truncated ZIP end record")
    _, _, _, _, entries, directory_size, directory_offset, _ = struct.unpack(
        "<4s4H2LH", eocd,
    )
    if (
        entries == 0xFFFF
        or directory_size == 0xFFFFFFFF
        or directory_offset == 0xFFFFFFFF
    ):
        locator_at = tail.rfind(b"PK\x06\x07", 0, eocd_at)
        if locator_at < 0 or len(tail) < locator_at + 20:
            raise ValueError("ZIP64 locator not found")
        _, _, zip64_offset, _ = struct.unpack(
            "<4sLQL", tail[locator_at : locator_at + 20],
        )
        record = fetch(int(zip64_offset), int(zip64_offset) + 56)
        if len(record) < 56 or record[:4] != b"PK\x06\x06":
            raise ValueError("invalid ZIP64 end record")
        fields = struct.unpack("<4sQ2H2L4Q", record[:56])
        entries = int(fields[7])
        directory_size = int(fields[8])
        directory_offset = int(fields[9])
    directory = fetch(
        int(directory_offset), int(directory_offset) + int(directory_size),
    )
    offset = 0
    result: dict[str, dict[str, int]] = {}
    for _ in range(int(entries)):
        if directory[offset : offset + 4] != b"PK\x01\x02":
            raise ValueError(f"invalid central-directory entry at {offset}")
        fixed = directory[offset : offset + 46]
        values = struct.unpack("<4s6H3L5H2L", fixed)
        method = int(values[4])
        crc = int(values[7])
        compressed = int(values[8])
        uncompressed = int(values[9])
        name_len, extra_len, comment_len = map(int, values[10:13])
        local_offset = int(values[16])
        start = offset + 46
        raw_name = directory[start : start + name_len]
        extra = directory[start + name_len : start + name_len + extra_len]
        name = _safe_member_name(raw_name.decode("utf-8", errors="replace"))
        if any(value == 0xFFFFFFFF for value in (compressed, uncompressed, local_offset)):
            extra_offset = 0
            zip64_values: list[int] = []
            while extra_offset + 4 <= len(extra):
                kind, length = struct.unpack_from("<HH", extra, extra_offset)
                payload = extra[extra_offset + 4 : extra_offset + 4 + length]
                if kind == 0x0001:
                    zip64_values = [
                        struct.unpack_from("<Q", payload, at)[0]
                        for at in range(0, len(payload) - 7, 8)
                    ]
                    break
                extra_offset += 4 + length
            cursor = 0
            if uncompressed == 0xFFFFFFFF:
                uncompressed = int(zip64_values[cursor]); cursor += 1
            if compressed == 0xFFFFFFFF:
                compressed = int(zip64_values[cursor]); cursor += 1
            if local_offset == 0xFFFFFFFF:
                local_offset = int(zip64_values[cursor]); cursor += 1
        result[name] = {
            "method": method,
            "crc": crc,
            "compressed": compressed,
            "uncompressed": uncompressed,
            "local_offset": local_offset,
        }
        offset = start + name_len + extra_len + comment_len
    return result


def fetch_zip_member(fetch, entry: dict[str, int]) -> bytes:
    """Fetch and verify one stored/deflated member from a remote ZIP."""
    local_offset = int(entry["local_offset"])
    header = fetch(local_offset, local_offset + 30)
    if len(header) < 30 or header[:4] != b"PK\x03\x04":
        raise ValueError("invalid local ZIP header")
    values = struct.unpack("<4s5H3L2H", header)
    name_len, extra_len = int(values[-2]), int(values[-1])
    start = local_offset + 30 + name_len + extra_len
    payload = fetch(start, start + int(entry["compressed"]))
    method = int(entry["method"])
    if method == 0:
        value = payload
    elif method == 8:
        value = zlib.decompress(payload, -15)
    else:
        raise ValueError(f"unsupported ZIP compression method {method}")
    if len(value) != int(entry["uncompressed"]):
        raise ValueError("ZIP member length mismatch")
    if (zlib.crc32(value) & 0xFFFFFFFF) != int(entry["crc"]):
        raise ValueError("ZIP member CRC mismatch")
    return value


def letterbox_tensor(path: str | Path, resolution: int) -> torch.Tensor:
    """Preserve aspect ratio and place the complete image on a square field."""
    resolution = int(resolution)
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    with Image.open(path) as source:
        image = source.convert("RGB")
        scale = min(resolution / image.width, resolution / image.height)
        size = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        image = image.resize(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (resolution, resolution), (0, 0, 0))
        offset = ((resolution - size[0]) // 2, (resolution - size[1]) // 2)
        canvas.paste(image, offset)
        array = np.asarray(canvas, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def materialize_record(record: dict[str, Any], resolution: int) -> dict[str, Any]:
    """Load one normalized record into the model's boundary/likelihood form."""
    task = str(record["task"])
    if task not in REAL_TASKS:
        raise ValueError(f"unknown real task {task!r}")
    source_files = list(record.get("source_files", ()))
    target_file = str(record.get("target_file", ""))
    if task == "t2i":
        image = torch.zeros(3, resolution, resolution)
        target = letterbox_tensor(target_file, resolution)
        image_precision = 0.0
    elif task == "it2i":
        if len(source_files) != 1 or not target_file:
            raise ValueError("IT2I requires one source file and one target file")
        image = letterbox_tensor(source_files[0], resolution)
        target = letterbox_tensor(target_file, resolution)
        image_precision = 1.0
    else:
        if len(source_files) != 1:
            raise ValueError(f"{task.upper()} requires one source image")
        image = letterbox_tensor(source_files[0], resolution)
        target = image.clone()
        image_precision = 1.0
    return {
        **record,
        "image": image,
        "target_rgb": target,
        "image_precision": image_precision,
        "text_missing": task == "i2t",
        "need_pix": task in ("t2i", "it2i"),
        "need_text": task in ("i2t", "it2t"),
        "target_image_precision": 1.0 if task in ("t2i", "it2i") else 0.0,
        "target_text_precision": 1.0 if task in ("i2t", "it2t") else 0.0,
    }


def _token_ids(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _null_token_id(tokenizer) -> int:
    token_id = tokenizer.bos_token_id
    if token_id is None:
        token_id = tokenizer.eos_token_id
    if token_id is None:
        raise ValueError("tokenizer needs a BOS or EOS token")
    return int(token_id)


def collate_real_multimodal(
    tokenizer,
    samples: Sequence[dict[str, Any]],
    *,
    max_text_tokens: int = 256,
    min_answer_tokens: int = 64,
    max_answer_tokens: int | None = None,
    append_eos: bool = False,
) -> dict[str, Any]:
    """Collate mixed real tasks without exposing answer tokens to vision."""
    if not samples:
        raise ValueError("cannot collate an empty real batch")
    encoded = []
    for sample in samples:
        answer = str(sample.get("answer", "")).strip()
        answer_ids = _token_ids(tokenizer, " " + answer) if answer else []
        if max_answer_tokens is not None:
            content_limit = max(0, int(max_answer_tokens) - int(bool(append_eos)))
            answer_ids = answer_ids[:content_limit]
        if append_eos and answer_ids:
            eos = tokenizer.eos_token_id
            if eos is None:
                raise ValueError("append_eos requires tokenizer.eos_token_id")
            answer_ids.append(int(eos))
        reserve = min(
            len(answer_ids), max(0, int(min_answer_tokens)),
        )
        if sample.get("text_missing", False):
            prefix = [_null_token_id(tokenizer)]
            prefix_precision = [0.0]
            visual_prefix = [False]
        else:
            prefix = _token_ids(tokenizer, str(sample.get("prompt", "")).strip())
            if not prefix:
                raise ValueError("observed prompt tokenized to an empty span")
            prefix_budget = max(1, int(max_text_tokens) - reserve)
            prefix = prefix[:prefix_budget]
            prefix_precision = [1.0] * len(prefix)
            visual_prefix = [True] * len(prefix)
        available = max(0, int(max_text_tokens) - len(prefix))
        answer_ids = answer_ids[:available]
        ids = prefix + answer_ids
        encoded.append({
            "ids": ids,
            "labels": [-100] * len(prefix) + answer_ids,
            "prompt_mask": visual_prefix + [False] * len(answer_ids),
            "precision": prefix_precision + [1.0] * len(answer_ids),
        })
    width = max(len(row["ids"]) for row in encoded)
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    if pad is None:
        raise ValueError("tokenizer needs a pad or EOS token")

    def padded(values: list, fill: Any) -> list:
        return values + [fill] * (width - len(values))

    return {
        "image": torch.stack([sample["image"] for sample in samples]),
        "target_rgb": torch.stack([sample["target_rgb"] for sample in samples]),
        "input_ids": torch.tensor(
            [padded(row["ids"], int(pad)) for row in encoded], dtype=torch.long,
        ),
        "labels": torch.tensor(
            [padded(row["labels"], -100) for row in encoded], dtype=torch.long,
        ),
        "visual_prompt_mask": torch.tensor(
            [padded(row["prompt_mask"], False) for row in encoded], dtype=torch.bool,
        ),
        "text_precision": torch.tensor(
            [padded(row["precision"], 0.0) for row in encoded], dtype=torch.float32,
        ),
        "attention_mask": torch.tensor(
            [padded([1] * len(row["ids"]), 0) for row in encoded], dtype=torch.long,
        ),
        "image_precision": torch.tensor(
            [float(sample["image_precision"]) for sample in samples], dtype=torch.float32,
        ),
        "target_image_precision": torch.tensor(
            [float(sample["target_image_precision"]) for sample in samples],
        ),
        "target_text_precision": torch.tensor(
            [float(sample["target_text_precision"]) for sample in samples],
        ),
        "need_pix": [bool(sample["need_pix"]) for sample in samples],
        "need_text": [bool(sample["need_text"]) for sample in samples],
        "need_seg": [False] * len(samples),
        "answer": [str(sample.get("answer", "")) for sample in samples],
        "task": [str(sample["task"]) for sample in samples],
        "id": [str(sample["id"]) for sample in samples],
    }


def counterfactual_real_samples(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Shuffle the causal input while preserving each row's terminal target."""
    if len(samples) < 2:
        raise ValueError("counterfactual shuffle requires at least two samples")
    controls = [deepcopy(sample) for sample in samples]
    shifted = list(samples[1:]) + [samples[0]]
    for control, other in zip(controls, shifted):
        task = str(control["task"])
        if task == "t2i":
            control["prompt"] = str(other["prompt"])
        else:
            control["image"] = other["image"].clone()
    return controls
