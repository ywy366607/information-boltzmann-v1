"""Read-only replay of actual changing-domain frames, with stable display IDs."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.cgb.exchange_checkpoint import load_exchange
from scripts.ib.generative.native_learning_stream import atomic_json


def main(run):
    arms = []
    for label in ('growing', 'closed_domain_barrier'):
        frames = json.loads((run/(label+'_frames.json')).read_text(encoding='utf-8'))
        for frame in frames:
            n = len(frame['coordinates'])
            degree = np.bincount(frame['pre']+frame['post'], minlength=n)
            frame.update(ms=frame['time_ms'], bound=frame['support'],
                morphology=dict(bridges=len(frame['pre']), branch_nodes=int(np.sum(degree >= 3)),
                                seed_component_nodes=0))
            frame['material'] = sum(frame['material'])
        arms.append(dict(id=label, frames=frames))
    model, _ = load_exchange(run/'growing_complete.npz')
    atomic_json(run/'mobile_scene.json', dict(box_size=model.params.box_size, default='growing',
        arms=arms, source=str(run), read_only=True,
        initial_domain_count=24, total_conserved_material=122))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--run', type=Path, default=Path('F:/fly_stream_overflow/cgb_domain_birth_20261022'))
    main(p.parse_args().run)
