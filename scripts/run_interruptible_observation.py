"""Opt-in natural-text observation: one small contact, then decide again.

Uses the existing autonomous selector, silence/speech policy and learning
transaction. Silence continues the text cursor; no unread suffix is acquired
or held by the body. An observation can continue across any number of contacts.
"""
import json
from pathlib import Path
import sys
from unittest.mock import patch

from scripts.run_arrived_language_actor import main as run_actor

GROWTH_STATUS = Path('G:/cgb_speech_growth_2613_20261121/growth_status.json')
CONTRACT = dict(schema=1, response_bytes=240, continuation_bytes=240)


def configure(loaded):
    engine, config = loaded.engine, loaded.config
    if not (config.get('world_source') == 'text' and config.get('speech_only')
            and getattr(engine, 'autonomous_events_schema', None) == 1):
        raise ValueError('requires the autonomous two-action natural-text body')
    inherited = config.get('interruptible_observation')
    if inherited is not None and inherited != CONTRACT:
        raise ValueError('observation contact contract differs from the saved body')
    if inherited is not None and (engine.response_bytes, engine.holdout_bytes) != (
            CONTRACT['response_bytes'], CONTRACT['continuation_bytes']):
        raise ValueError('saved observation contact sizes do not match their contract')
    engine.response_bytes = CONTRACT['response_bytes']
    engine.holdout_bytes = CONTRACT['continuation_bytes']
    config['interruptible_observation'] = CONTRACT.copy()
    return loaded


def main():
    # Do not run a private learning branch while the approved growth run is live.
    if json.loads(GROWTH_STATUS.read_text(encoding='utf8'))['status'] != 'completed':
        raise SystemExit('Wait for the 512-round growth result before running this branch.')
    from scripts.ib_local import run_mother_continuation as continuation
    prepare = continuation.prepare_continuation

    def prepared(*args, **kwargs):
        return configure(prepare(*args, **kwargs))

    with patch.object(continuation, 'prepare_continuation', prepared):
        run_actor()


if __name__ == '__main__':
    main()
