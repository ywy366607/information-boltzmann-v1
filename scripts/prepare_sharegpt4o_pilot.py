#!/usr/bin/env python3
"""Prepare a small real T2I/IT2I pilot without downloading 262 GB.

The official image archives are uncompressed tar files.  This script fetches
only a byte prefix of the first archive and retains complete referenced pairs
encountered in that prefix.  Raw data remains under ML_CACHE_ROOT and is never
written to the repository.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fine_grain.sharegpt4o_data import (
    FREEDOM_REPO,
    OPENGV_REPO,
    extract_referenced_tar,
    fetch_zip_member,
    load_freedom_manifest,
    load_opengv_manifest,
    read_zip_directory,
)


def resolve_hf_file(repo: str, filename: str) -> tuple[str, int | None]:
    """Resolve a gated HF file with curl and report its range size."""
    from huggingface_hub import get_token

    curl = shutil.which("curl") or shutil.which("curl.exe")
    if curl is None:
        raise RuntimeError("curl is required for Hugging Face range access")
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{filename}"
    command = [
        curl, "-L", "--fail", "--retry", "8", "--retry-all-errors",
        "--silent", "--show-error", "--range", "0-0", "--max-filesize", "1024",
        "--dump-header", "-", "--output", os.devnull,
        "--write-out", "\nHF_EFFECTIVE:%{url_effective}\n", url,
    ]
    if sys.platform == "win32":
        command.insert(1, "--ssl-no-revoke")
    token = get_token()
    if token:
        command[1:1] = ["-H", f"Authorization: Bearer {token}"]
    result = subprocess.run(
        command, check=True, stdout=subprocess.PIPE, text=True,
        encoding="utf-8", errors="replace",
    )
    effective = re.findall(r"HF_EFFECTIVE:(.+)", result.stdout)
    if not effective:
        raise RuntimeError("curl did not report an effective Hugging Face URL")
    ranges = re.findall(r"(?i)content-range:\s*bytes\s+\d+-\d+/(\d+)", result.stdout)
    size = int(ranges[-1]) if ranges else None
    return effective[-1].strip(), size


def download_prefix(
    repo: str,
    filename: str,
    destination: Path,
    size_bytes: int,
    *,
    segment_bytes: int,
) -> None:
    """Download an HTTP Range prefix with a strict size ceiling."""
    if destination.is_file() and destination.stat().st_size >= int(size_bytes):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    url, _ = resolve_hf_file(repo, filename)
    segment = max(1024 * 1024, int(segment_bytes))
    ranges = [
        (start, min(int(size_bytes), start + segment))
        for start in range(0, int(size_bytes), segment)
    ]
    segment_dir = destination.with_name(destination.name + ".segments")
    segment_dir.mkdir(parents=True, exist_ok=True)

    def fetch_segment(item):
        index, bounds = item
        part = segment_dir / f"{index:05d}.part"
        expected = bounds[1] - bounds[0]
        if part.is_file() and part.stat().st_size == expected:
            return part
        payload = curl_fetch_range(url, bounds[0], bounds[1])
        part.write_bytes(payload)
        return part

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(ranges))) as pool:
        parts = list(pool.map(fetch_segment, enumerate(ranges)))
    with destination.open("wb") as handle:
        for part in parts:
            handle.write(part.read_bytes())
    if destination.stat().st_size > int(size_bytes) + 1024:
        raise RuntimeError("server ignored the Range request; refusing a full archive")


def curl_fetch_range(url: str, start: int, stop: int) -> bytes:
    """Return a half-open byte range without materializing the remote file."""
    curl = shutil.which("curl") or shutil.which("curl.exe")
    if curl is None:
        raise RuntimeError("curl is required for remote ZIP range reads")
    expected = int(stop) - int(start)
    command = [
        curl, "-L", "--fail",
        "--silent", "--show-error", "--range", f"{int(start)}-{int(stop) - 1}",
        "--max-filesize", str(expected + 1024), url,
    ]
    if sys.platform == "win32":
        command.insert(1, "--ssl-no-revoke")
    last = ""
    for attempt in range(12):
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if result.returncode == 0 and len(result.stdout) == expected:
            return result.stdout
        last = result.stderr.decode("utf-8", errors="replace").strip()
        time.sleep(min(8.0, 0.5 * (attempt + 1)))
    raise RuntimeError(
        f"range length mismatch after retries: wanted {expected}; {last}"
    )


def materialize_opengv_zip(
    records: list[dict], output_dir: Path, max_complete_each: int,
) -> list[dict]:
    """Selectively extract gated conversation images from the 6.5 GB ZIP."""
    signed_url, size = resolve_hf_file(OPENGV_REPO, "images.zip")
    if size is None:
        raise RuntimeError("OpenGVLab images.zip did not report a size")
    fetch = lambda start, stop: curl_fetch_range(signed_url, start, stop)
    directory = read_zip_directory(fetch, int(size))
    by_basename = {
        Path(name).name: name for name in directory if Path(name).name
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    selected_counts = {"i2t": 0, "it2t": 0}
    for row in records:
        task = str(row.get("task", ""))
        if task not in selected_counts:
            continue
        if selected_counts[task] >= int(max_complete_each):
            continue
        member = str(row["source_members"][0]).lstrip("./")
        candidates = (member, f"images/{member}", f"image/{member}")
        archive_name = next((name for name in candidates if name in directory), None)
        if archive_name is None:
            archive_name = by_basename.get(Path(member).name)
        if archive_name is None:
            continue
        destination = output_dir / f"{row['id']}{Path(member).suffix or '.jpg'}"
        selected.append((row, archive_name, destination))
        selected_counts[task] += 1
        if all(value >= int(max_complete_each) for value in selected_counts.values()):
            break

    def materialize(item):
        row, archive_name, destination = item
        if not destination.is_file() or destination.stat().st_size == 0:
            payload = fetch_zip_member(fetch, directory[archive_name])
            destination.write_bytes(payload)
        result = dict(row)
        result["source_files"] = [str(destination.resolve())]
        result["target_file"] = ""
        return result

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(8, max(1, len(selected)))
    ) as pool:
        result = list(pool.map(materialize, selected))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=r"D:\ml_cache\sharegpt4o")
    parser.add_argument("--prefix-mb", type=int, default=256)
    parser.add_argument(
        "--t2i-prefix-mb", type=int, default=None,
        help="Override the T2I tar prefix size (defaults to --prefix-mb).",
    )
    parser.add_argument(
        "--it2i-prefix-mb", type=int, default=None,
        help="Override the IT2I tar prefix size (defaults to --prefix-mb).",
    )
    parser.add_argument("--segment-mb", type=int, default=4)
    parser.add_argument("--max-each", type=int, default=8)
    parser.add_argument("--max-prompt-chars", type=int, default=2048)
    parser.add_argument("--max-answer-chars", type=int, default=4096)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--opengv", action=argparse.BooleanOptionalAction, default=True,
        help="Selectively range-extract gated OpenGVLab images.zip members.",
    )
    args = parser.parse_args()

    cache = Path(args.cache)
    t2i_json = cache / "text_to_image.json"
    edit_json = cache / "text_and_image_to_image.json"
    if not t2i_json.is_file() or not edit_json.is_file():
        raise FileNotFoundError(
            "download the two FreedomIntelligence JSON manifests into "
            f"{cache} before preparing images"
        )
    t2i = load_freedom_manifest(
        t2i_json, "t2i", max_prompt_chars=args.max_prompt_chars,
    )
    edit = load_freedom_manifest(
        edit_json, "it2i", max_prompt_chars=args.max_prompt_chars,
        same_resolution=True,
    )
    t2i_prefix_mb = int(args.t2i_prefix_mb or args.prefix_mb)
    it2i_prefix_mb = int(args.it2i_prefix_mb or args.prefix_mb)
    segment_bytes = int(args.segment_mb) * 1024 * 1024
    archives = {
        "t2i": cache / "text_to_image_part_0.prefix.tar",
        "it2i": cache / "text_and_image_to_image_part_0.prefix.tar",
    }
    if not args.skip_download:
        download_prefix(
            FREEDOM_REPO, "text_to_image_part_0.tar", archives["t2i"],
            t2i_prefix_mb * 1024 * 1024, segment_bytes=segment_bytes,
        )
        download_prefix(
            FREEDOM_REPO, "text_and_image_to_image_part_0.tar",
            archives["it2i"], it2i_prefix_mb * 1024 * 1024,
            segment_bytes=segment_bytes,
        )

    records = []
    records.extend(extract_referenced_tar(
        archives["t2i"], t2i, cache / "pilot_images" / "t2i",
        max_complete=args.max_each,
    ))
    records.extend(extract_referenced_tar(
        archives["it2i"], edit, cache / "pilot_images" / "it2i",
        max_complete=args.max_each,
    ))

    opengv_path = cache / "gpt-4o.jsonl"
    opengv = []
    opengv_records = []
    if opengv_path.is_file():
        opengv = load_opengv_manifest(
            opengv_path, max_answer_chars=args.max_answer_chars,
            max_per_task=args.max_each,
        )
        if args.opengv:
            opengv_records = materialize_opengv_zip(
                opengv, cache / "pilot_images" / "opengv", args.max_each,
            )
            records.extend(opengv_records)
    record = {
        "schema": "sharegpt4o-real-pilot-v1",
        "cache_only": True,
        "archive_prefix_mb": {
            "t2i": t2i_prefix_mb,
            "it2i": it2i_prefix_mb,
        },
        "freedom": {
            "t2i_manifest_rows": len(t2i),
            "it2i_same_resolution_rows": len(edit),
            "materialized_t2i": sum(row["task"] == "t2i" for row in records),
            "materialized_it2i": sum(row["task"] == "it2i" for row in records),
        },
        "opengv": {
            "conversation_rows_parsed": len(opengv),
            "images_materialized": len(opengv_records),
            "materialized_i2t": sum(row["task"] == "i2t" for row in opengv_records),
            "materialized_it2t": sum(row["task"] == "it2t" for row in opengv_records),
            "status": "selective range extraction complete" if opengv_records else (
                "metadata_ready; no matching ZIP members materialized"
            ),
        },
        "records": records,
    }
    output = cache / "pilot_manifest.json"
    output.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "freedom": record["freedom"],
        "opengv": record["opengv"],
    }, indent=2, ensure_ascii=False))
    if not records:
        raise SystemExit(
            "No complete samples found. Increase --prefix-mb; the script will not "
            "download a complete archive implicitly."
        )


if __name__ == "__main__":
    main()
