"""Atomic, non-pickle archives for the isolated finite exchange subsystem."""
import json
import os
from pathlib import Path

import numpy as np

from .finite_exchange import FiniteExchange


def save_exchange(path, model, provenance=None):
    path = Path(path)
    state = model.state_dict()
    arrays = {name:state.pop(name) for name in ('pre', 'post', 'capacity')}
    arrays.update({'state_'+k:v for k, v in state.pop('arrays').items()})
    arrays['metadata'] = np.array(json.dumps(dict(state=state, provenance=provenance or {}), allow_nan=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    try:
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        load_exchange(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_exchange(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        state = metadata['state']
        model_class = FiniteExchange
        if state.get('kind') == 'structural_exchange':
            from .structural_exchange import StructuralExchange
            model_class = StructuralExchange
        elif state.get('kind') == 'flow_exchange':
            from .flow_exchange import FlowExchange
            model_class = FlowExchange
        elif 'kind' in state:
            raise ValueError('Unknown exchange archive kind')
        for name in ('pre', 'post', 'capacity'):
            state[name] = archive[name].copy()
        state['arrays'] = {name:archive['state_'+name].copy() for name in model_class._arrays}
    return model_class.from_state_dict(state), metadata['provenance']
