"""Fetch a few genuine DAVIS frames/masks by verified HTTP byte ranges."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fine_grain.sharegpt4o_data import read_zip_directory, fetch_zip_member

URL = "https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+", default=["blackswan", "camel"])
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 1, 2, 3])
    args = parser.parse_args()
    if any(Path(s).name != s or s in (".", "..") for s in args.sequences):
        parser.error("sequences must be simple names")
    if any(i < 0 for i in args.frames):
        parser.error("frame indices must be nonnegative")
    session = requests.Session()
    transferred = 0

    def fetch(start, stop):
        nonlocal transferred
        response = session.get(URL, headers={"Range": f"bytes={start}-{stop - 1}"},
                               timeout=40, stream=True)
        try:
            if response.status_code != 206:
                raise RuntimeError("server did not honor range; refusing full-archive download")
            expected = f"bytes {start}-{stop - 1}/"
            if not response.headers.get("Content-Range", "").startswith(expected):
                raise RuntimeError("mismatched Content-Range")
            payload = response.content
            if len(payload) != stop - start:
                raise RuntimeError("truncated range")
            transferred += len(payload)
            return payload
        finally:
            response.close()

    probe = session.get(URL, headers={"Range": "bytes=0-0"}, stream=True, timeout=40)
    try:
        if probe.status_code != 206:
            raise RuntimeError("byte-range access unavailable")
        size = int(probe.headers["Content-Range"].split("/")[-1])
    finally:
        probe.close()
    directory = read_zip_directory(fetch, size)
    args.cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for sequence in args.sequences:
        for frame in args.frames:
            row = {"sequence": sequence, "frame": frame, "dataset": "DAVIS-2017-trainval"}
            for kind, extension, key in (("JPEGImages", "jpg", "image"), ("Annotations", "png", "mask")):
                member = f"DAVIS/{kind}/480p/{sequence}/{frame:05d}.{extension}"
                if member not in directory:
                    raise KeyError(member)
                payload = fetch_zip_member(fetch, directory[member])
                destination = args.cache / f"{sequence}-{frame:05d}-{key}.{extension}"
                if destination.exists() and destination.read_bytes() != payload:
                    raise FileExistsError(f"cached file differs: {destination}")
                if not destination.exists():
                    destination.write_bytes(payload)
                row[key] = str(destination.resolve())
                row[f"{key}_member"] = member
                row[f"{key}_sha256"] = hashlib.sha256(payload).hexdigest()
            rows.append(row)
            print(f"cached {sequence} frame {frame}", flush=True)
    result = {"source": URL, "source_page": "https://davischallenge.org/davis2017/code.html",
              "bytes_transferred": transferred, "records": rows}
    (args.cache / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"frames": len(rows), "bytes_transferred": transferred}), flush=True)


if __name__ == "__main__":
    main()
