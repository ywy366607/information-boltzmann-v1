"""Admit the same complete CUDA organism, including delayed P-read credit.

Uses only its original actual natural-text events. The separate restore pass
compares carried tensors/optimizer/world/RNG with the just-committed boundary.
"""
from __future__ import annotations

import hashlib
import dataclasses
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

from scripts.ib_local.run_mother_continuation import continuation_parser, prepare_continuation, ROOT


def digest(value):
    h = hashlib.sha256()

    def add(item):
        if torch.is_tensor(item):
            a = item.detach().cpu().contiguous().numpy()
            h.update(str((a.dtype, a.shape)).encode()); h.update(a.tobytes())
        elif isinstance(item, np.ndarray):
            h.update(str((item.dtype, item.shape)).encode()); h.update(item.tobytes())
        elif isinstance(item, dict):
            for k in sorted(item, key=repr):
                add(k); add(item[k])
        elif isinstance(item, (tuple, list)):
            h.update(type(item).__name__.encode())
            for v in item: add(v)
        elif dataclasses.is_dataclass(item):
            h.update(type(item).__qualname__.encode())
            for field in dataclasses.fields(item):
                add(field.name); add(getattr(item, field.name))
        else:
            h.update(pickle.dumps(item, protocol=4))
    add(value)
    return h.hexdigest()


def carried(engine):
    from scripts.ib_local.event_journal import rng_digest
    return dict(model=digest(engine.model.state_dict()),
        optimizer=digest(engine.optimizer.state_dict()), recurrent=digest(engine.recurrent),
        momenta=digest(engine.bfss_momenta), pending=digest(engine.internal_pending),
        world=digest(engine.world.checkpoint_state_dict()), rng=rng_digest())


@torch.no_grad()
def phase_geometry(engine):
    module = engine.model.head_coupling
    first = engine.model.body.base.layers[0]
    s = engine.recurrent[..., :first.matrix_size].reshape(
        module.layers, 1, module.heads, module.dk, module.dv).cpu().double()
    p = torch.stack(engine.bfss_momenta).cpu().double()
    z = (module.impedance.cpu().double()*s+1j*p).flatten(-2).squeeze(1)
    norm = z.abs().square().sum(-1).sqrt()
    gram = z@z.conj().transpose(-1,-2)
    coherence = gram/(norm[...,None]*norm[:,None,:]).clamp_min(1e-30)
    pairs = torch.triu_indices(module.heads,module.heads,1)
    pair = coherence[:,pairs[0],pairs[1]]
    return dict(definition='normalized complex overlap of Z*S+i*P; one matrix head need not have a single scalar phase',
        per_layer_mean_coherence=pair.abs().mean(-1).tolist(),
        per_layer_mean_signed_alignment=pair.real.mean(-1).tolist(),
        coherent_pair_fraction=float((pair.abs()>.8).double().mean()),
        # Angles of weak/orthogonal pairs have no phase-lock interpretation.
        strongly_coherent_angles=pair.angle()[pair.abs()>.8].tolist())


def main():
    restore_only = '--restore-only' in sys.argv
    arguments = [a for a in sys.argv[1:] if a != '--restore-only']
    args = continuation_parser().parse_args(arguments)
    report_path = ROOT/'results/published/cgb_adaptive_spectrum_cuda_20261116.json'
    loaded = prepare_continuation(args,resume_output=args.resume_output)
    engine = loaded.engine
    if restore_only:
        report = json.loads(report_path.read_text(encoding='utf-8'))
        restored = carried(engine)
        report['restore'] = {k:restored[k]==v for k,v in report['carried_after'].items()}
        report_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        if not all(report['restore'].values()):
            raise RuntimeError(f"complete restore differs: {report['restore']}")
        print(json.dumps(dict(status='restored',checks=report['restore'])),flush=True)
        return
    report = dict(schema=1,start_cycle=loaded.cycle,source_sha256=loaded.source_sha,
        scope='actual original natural-text complete-body admission; no mother API, forced choice or synthetic task',
        geometry_before=phase_geometry(engine),carried_before=carried(engine))
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    from scripts.ib_local.autonomous_runtime import run
    run(args,loaded,None,ROOT)
    report.update(elapsed_seconds=time.perf_counter()-started,
        peak_cuda_mib=torch.cuda.max_memory_allocated()/1048576,
        geometry_after=phase_geometry(engine),carried_after=carried(engine),
        completed=json.loads((args.output/'completed.json').read_text(encoding='utf-8')))
    report_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    rows=[json.loads(line) for line in (args.output/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    report['events']=[{k:r[k] for k in ('cycle','updates','response_nll','metal_g',
        'adaptive_dynamics','momentum_current_credit_samples','momentum_delayed_credit_samples') if k in r}
        for r in rows if r['cycle']>loaded.cycle]
    report_path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='actual_events_committed',report=str(report_path))),flush=True)


if __name__=='__main__':main()
