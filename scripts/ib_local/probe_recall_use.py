"""Why did the GDN not recall in T1?  Counterfactual probes on saved checkpoints (2026-11-03).

For each probe: warm-up text, fact "[#KEY = A]", G bytes of text, query "[#KEY = ".  The
model reads it from its trained final state with frozen weights.  Three versions differ in
one place only:
  - fact value A (reference), fact value B (different word, different first letter),
  - query key changed to an unseen key (fact value A).
Reported at the position that predicts the first value byte:
  value_effect = log p(A1 | fact A) - log p(A1 | fact B)   (does the fact reach the query?)
  key_effect   = log p(A1 | fact A) - log p(A1 | query key changed)   (is the key matched?)
  p_A1 under the reference, and the share of probes where A1 is the argmax.
Also: the share of training pairs whose fact and query fall in one 512-byte window.
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

import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.ib_local import gdn_fused  # noqa: E402
from scripts.ib_local.eval_recall import fresh_table, load  # noqa: E402
from scripts.ib_local.recall_task import Builder, owt_bytes, record, train_stream, value_words  # noqa: E402
from scripts.ib_local.train_gdn_fast_table import FLOOR  # noqa: E402

ROOT = Path(ROOT_DIR)
RUN = ROOT / "results/cgb_recall_t1_20261103"
GAPS = (16, 64, 256, 1024)
PROBES = 48


@torch.no_grad()
def first_value_logp(model, state, table, sequence: bytes):
    """log-probabilities for the byte after the last byte of `sequence`."""
    stream = np.frombuffer(sequence, np.uint8)
    n = len(stream)
    ids = torch.from_numpy(stream.astype(np.int64)).cuda()[None]
    if table is not None:
        _, probs = table.run(np.concatenate([stream, stream[-1:]]), keep_probs=True)
        tab = torch.from_numpy(np.log(np.maximum(probs, FLOOR))).cuda()[None]
    else:
        tab = torch.full((1, n, 256), math.log(1 / 256), device="cuda")
    logits, _, _ = model.predict(ids, state.clone(), tab)
    return torch.log_softmax(logits[0, -1], -1).cpu().numpy()


def main():
    gdn_fused.enable()
    source = owt_bytes("train", 4_400_000)
    words = value_words(source[:4_000_000])
    rows = {}
    for arm in ("control_s11", "latch_s11", "table_s11"):
        model, state0, summary = load(RUN / arm)
        rows[arm] = {}
        for gap in GAPS:
            builder = Builder(source[3_000_000:], words, seed=7 + gap)
            value_eff, key_eff, p_ref, hit = [], [], [], []
            for _ in range(PROBES):
                a = words[builder.rng.integers(len(words))]
                b = a
                while b[:1] == a[:1]:
                    b = words[builder.rng.integers(len(words))]
                key, other = builder.key(), builder.key()
                warm, gap_text = builder.text(512), builder.text(gap)
                head = lambda k: b"\n[#" + k + b" = "
                ref = warm + record(key, a)[0] + gap_text + head(key)
                swap = warm + record(key, b)[0] + gap_text + head(key)
                rekey = warm + record(key, a)[0] + gap_text + head(other)
                outs = []
                for seq in (ref, swap, rekey):
                    table = fresh_table(RUN / arm) if summary["table"] else None
                    outs.append(first_value_logp(model, state0, table, seq))
                t = a[0]
                value_eff.append(outs[0][t] - outs[1][t]); key_eff.append(outs[0][t] - outs[2][t])
                p_ref.append(math.exp(outs[0][t])); hit.append(int(outs[0].argmax() == t))
            rows[arm][str(gap)] = dict(value_effect=float(np.mean(value_eff)),
                                       key_effect=float(np.mean(key_eff)),
                                       p_first=float(np.mean(p_ref)), argmax=float(np.mean(hit)))
            print(arm, gap, {k: round(v, 3) for k, v in rows[arm][str(gap)].items()}, flush=True)
    _, spans = train_stream(5000 * 512, seed=11)
    same = []
    for lo, hi, gap in spans:
        fact_start = lo - gap - len(b"\n[#Kq7 = ") - 10  # approximate start of the fact value
        same.append(fact_start // 512 == lo // 512)
    rows["share_pairs_in_one_window"] = float(np.mean(same))
    print("share of training pairs with fact and query in one 512-byte window:",
          round(rows["share_pairs_in_one_window"], 3))
    (RUN / "probe_use.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
