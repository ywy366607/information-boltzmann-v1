"""Run the random-sampling control without injecting a new corpus query.

The language body samples its own utterance exactly as in the active arm, but
the surrounding ``random`` mode disables the policy-credit update. This is an
unreinforced, model-conditioned control, not an independent random topic.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from scripts.ib_local.language_navigation_v2 import DialogueEngine
from scripts.ib_local import run_language_navigation_v2


def unreinforced_utter(engine, original):
    if engine.mode != 'random':
        raise ValueError('unreinforced wrapper requires random mode')
    engine.mode = 'active'
    try:
        return original(engine)
    finally:
        engine.mode = 'random'


def main():
    args = sys.argv[1:]
    if '--mode' not in args or args[args.index('--mode') + 1] != 'random':
        raise ValueError('requires --mode random')
    output = Path(args[args.index('--output') + 1])
    wrapper = Path(__file__)
    provenance = dict(control='model-sampled language; no policy-credit update',
                      wrapper_sha256=hashlib.sha256(wrapper.read_bytes()).hexdigest())
    marker = output.parent / (output.name + '.wrapper.json')
    if marker.exists() and json.loads(marker.read_text()) != provenance:
        raise ValueError('wrapper provenance changed')
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(provenance, indent=2))
    original = DialogueEngine._utter
    DialogueEngine._utter = lambda self: unreinforced_utter(self, original)
    run_language_navigation_v2.main()


if __name__ == '__main__':
    main()
