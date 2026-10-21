"""Compare bounded-work execution to its retained, actually executed source."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.cgb.exchange_checkpoint import load_exchange
from information_boltzmann.cgb.mobile_exchange import MobileExchange
from scripts.ib.generative.driven_ports_audit import input_voltage


def main(args):
    source = args.reference/'source_snapshot/mobile_exchange.py'
    report = json.loads((args.reference/'report.json').read_text(encoding='utf-8'))
    source_key = next(k for k in report['code_sha256'] if k.replace('\\', '/').endswith('/mobile_exchange.py'))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == report['code_sha256'][source_key]
    name = 'information_boltzmann.cgb._retained_mobile_reference'
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    results = {}
    for label in report['arms']:
        archive = args.reference/(label+'_half.npz')
        with np.load(archive, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata']))
            state = meta['state']
            state.update({k: data[k].copy() for k in ('pre', 'post', 'capacity')})
            state['arrays'] = {k: data['state_'+k].copy() for k in MobileExchange._arrays}
        old = module.MobileExchange.from_state_dict(state)
        new = MobileExchange.from_state_dict(state)
        data = bytes.fromhex(meta['provenance']['byte_stream_hex'])
        times = []
        for model in (old, new):
            started = time.perf_counter()
            for step in range(64, 128):
                model.step(2., input_voltage(data, step, 2.))
            times.append(time.perf_counter()-started)
        for k in MobileExchange._arrays+('pre', 'post', 'capacity'):
            np.testing.assert_array_equal(getattr(old, k), getattr(new, k))
        for rng in ('rng_motion', 'rng_reaction', 'rng_field'):
            assert getattr(old, rng).bit_generator.state == getattr(new, rng).bit_generator.state
        assert old.reaction_counts == new.reaction_counts
        ledger_error = max(abs(getattr(old, k)-getattr(new, k)) for k in MobileExchange._ledgers)
        assert ledger_error < 1e-9
        results[label] = dict(physical_arrays_and_rng_bitwise=True,
            maximum_ledger_roundoff=ledger_error, reference_seconds=times[0], bounded_work_seconds=times[1])
    result = dict(reference=str(args.reference), reference_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        current_source_sha256=hashlib.sha256((ROOT/'information_boltzmann/cgb/mobile_exchange.py').read_bytes()).hexdigest(),
        same_physical_clock='128->256ms, dt2ms, same retained real byte stream', arms=results, passed=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args())
