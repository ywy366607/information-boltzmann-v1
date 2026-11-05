"""Forward checks of the written birth on practice text, learning off.

Practice bytes come from train at 10 M, away from the evaluation stream at
20 M.  Reports the score with the written memory heads on and off, and how
often the memory's top byte equals the next byte, against an explicit cache
that predicts what followed the current byte last time.

Usage: python scripts/cgb_born_practice.py --bytes 20000 [--set key=value ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import cgb_language_gate as gate  # noqa: E402
import cgb_born_gdn2 as born  # noqa: E402


def score(body, text, heads_on: bool) -> tuple[float, float]:
    second = body.model.layers[1]
    dk = body.model.d // body.model.heads
    saved = second.o_proj.weight[:, :2 * dk].clone()
    if not heads_on:
        with torch.no_grad():
            second.o_proj.weight[:, :2 * dk] = 0.0
    body.state = body.model.initial_state(1)
    body.last = 32
    body.read()
    nll, hits = 0.0, 0
    with torch.no_grad():
        for byte in text:
            logp = body.predict()
            nll -= float(logp[byte])
            hits += int(int(logp.argmax()) == byte)
            body.state = body.pending[1].detach()
            body.last = byte
            body.read()
        second.o_proj.weight[:, :2 * dk] = saved
    return nll / len(text), hits / len(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bytes", type=int, default=20_000)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args()
    torch.set_num_threads(1)
    birth = {}
    for item in args.set:
        key, value = item.split("=")
        birth[key] = tuple(float(v) for v in value.split(",")) if "," in value else float(value)
    born.GDN2Born.birth = birth
    body = born.GDN2Born(1024)
    text = gate.stream("train", 10_000_000, 10_000_000 + int(args.bytes * 1.02))[:args.bytes]
    on = score(body, text, True)
    off = score(body, text, False)
    last, cache_hits, previous = {}, 0, None
    for byte in text:
        if previous is not None and last.get(previous) == byte:
            cache_hits += 1
        if previous is not None:
            last[previous] = byte
        previous = byte
    print(json.dumps({"birth": birth, "nll_on": round(on[0], 4), "nll_off": round(off[0], 4),
                      "top1_on": round(on[1], 4), "top1_off": round(off[1], 4),
                      "cache_top1": round(cache_hits / len(text), 4)}))


if __name__ == "__main__":
    main()
