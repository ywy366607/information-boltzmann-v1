"""Check real-scaffold contact totals at the exact zero-exposure spatial prolongation."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.ib.fly_interface.native_causal_audit import restore_native, ensure_gpu_idle, sha256
from information_boltzmann.core.fly_apl_cable import cable_state, _arrival, apl_mb_arrival


def verify(checkpoint, output):
    ensure_gpu_idle([])
    output.mkdir(parents=True, exist_ok=True)
    learner, saved, _, changed = restore_native(checkpoint, output, frozen=True)
    if changed:
        raise ValueError('Initial routing audit must not migrate the source')
    m, s = learner.model, learner.state
    _, _, _, packets = cable_state(m, s)
    rng_cpu, rng_cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
    results = {}

    def check(name, actual, expected):
        results[name] = {'absmax': float((actual - expected).abs().max()),
            'expected_absmax': float(expected.abs().max()),
            'close': torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)}

    def mean(value):
        return torch.segment_reduce((value * m.aplc_area_fraction)[0], 'sum', lengths=m.aplc_body_counts).unsqueeze(0)

    with torch.no_grad():
        for name, value in zip(['voltage', 'ge', 'gi'], cable_state(m, s)[:3]):
            old = getattr(s, 'h' if name == 'voltage' else name).index_select(1, m.apl_cable_index)
            check(name + '_prolongation', mean(value), old)
        for channel in ['e', 'i']:
            local = _arrival(m, s, packets, 'in_' + channel, destination_nodes=True)
            expected = m.transmit(s.ring, channel).index_select(1, m.apl_cable_index)
            check(channel + '_input_total', mean(local), expected)
        only_apl = tuple(torch.zeros_like(p).index_copy(1, m.apl_cable_index, p.index_select(1, m.apl_cable_index)) for p in s.ring)
        check('fast_output_total', _arrival(m, s, packets, 'out_i'), m.transmit(only_apl, 'i'))
        expected = s.h.new_zeros(m.mb_terminal_source_index.numel() * m.mb_terminal_kc_index.numel())
        for lag in range(1, 5):
            pre = getattr(m, f'mbt_pre_{lag}')
            value = only_apl[lag - 1][0, pre] * getattr(m, f'mbt_gaba_weight_{lag}')
            expected.index_add_(0, getattr(m, f'mbt_post_{lag}'), value)
        check('axonal_output_total', apl_mb_arrival(m, s), expected)
    fingerprint = sha256(checkpoint)
    preserved = torch.equal(rng_cpu, torch.get_rng_state()) and torch.equal(rng_cuda, torch.cuda.get_rng_state())
    report = {'checkpoint': str(checkpoint), 'sha256': fingerprint, 'trained': saved['bptt_train_tokens'],
        'checks': results, 'rng_preserved': preserved,
        'scope': 'initial uniform prolongation only; exact anatomical routing totals, not future trajectory equivalence or learning benefit',
        'status': 'initial_delivery_passed' if preserved and all(r['close'] for r in results.values()) else 'not_accepted'}
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    print(json.dumps(report), flush=True)
    if report['status'] != 'initial_delivery_passed':
        raise RuntimeError('APL local route changed initial physical contact totals')


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    verify(a.checkpoint, a.output)
