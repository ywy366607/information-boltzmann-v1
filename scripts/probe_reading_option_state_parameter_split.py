"""Read-only counterfactual decomposition of the complete battle endpoints."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import torch

from scripts.audit_coupling_credit_transport import open_archive, TensorRef
from scripts.ib_local import run_mother_continuation as continuation
from scripts.ib_local.wood_policy_measurement import segments, score
from scripts.run_native_causal_use_value import evaluate

ROOT = Path('E:/mengli-bounded-reading-20261123')
REPORT = Path('D:/information-boltzmann-v1/results/published/cgb_bounded_reading_state_parameter_split_20261124.json')


def main():
    torch.set_num_threads(1)
    battle = json.loads(Path('results/published/cgb_bounded_reading_battle_20261123.json').read_text(encoding='utf8'))
    work = ROOT/'readonly_split'
    opts = continuation.continuation_parser().parse_args([
        '--checkpoint', str(ROOT/'option4_600s/body/last.pt.gz'), '--output', str(work/'body'),
        '--cycles', '1', '--world-source', 'text', '--autonomous-events', '--optimizer-policy', 'budgeted',
        '--table-storage', str(work/'table'), '--causal-storage', str(work/'causal')])
    continuation.save = lambda *a, **kw: None
    continuation.export_live = lambda *a, **kw: None
    loaded = continuation.prepare_continuation(opts)
    engine = loaded.engine
    material = segments(battle['fresh_text_spec'])[:2]
    facts = torch.load(ROOT/'strict_smoke/material.pt', map_location=engine.device, weights_only=False)
    params = dict(engine.model.named_parameters())
    current = {k:p.detach().cpu().clone() for k,p in params.items()}
    original_state = engine.recurrent.clone()
    rows, deltas = {}, {}

    def measure(label):
        rows[label] = dict(text=score(engine, material), facts=evaluate(engine, facts))
        rows[label]['facts'].pop('queries')
        REPORT.write_text(json.dumps(dict(rows=rows, parameter_delta=deltas), indent=2)+'\n', encoding='utf8')
        print(label, rows[label]['text']['mean_nll'], rows[label]['facts']['mean_score'], flush=True)

    measure('candidate_weights_candidate_state')
    # The same raw state coordinates are shared by all endpoints. Workspace
    # identities belong to the candidate's current field, so only physical S
    # is transplanted; the small workspace is cleared on both comparison arms.
    with open_archive(Path(battle['source']), Path('E:/mengli-runtime-staging-20261123')) as reader:
        source_s = reader.tensor(reader.state['engine']['recurrent']).to(engine.device)
        source_weights = {k:reader.tensor(reader.state['model'][k]).clone() for k in params}
        print('selected source bytes', reader.loaded_bytes, flush=True)
    physical = engine.model.body.base.layers[0].state_size
    engine.recurrent = original_state.clone()
    engine.recurrent[..., physical:] = 0
    measure('candidate_weights_candidate_physical_zero_workspace')
    engine.recurrent[..., :physical] = source_s[..., :physical]
    measure('candidate_weights_source_physical_zero_workspace')
    engine.recurrent = original_state.clone()
    with torch.no_grad():
        for name, p in params.items():
            delta = current[name]-source_weights[name]
            deltas[name] = dict(norm=float(delta.norm()), source_norm=float(source_weights[name].norm()))
            p.copy_(source_weights[name])
    measure('source_weights_candidate_state')
    engine.recurrent[..., physical:] = 0
    engine.recurrent[..., :physical] = source_s[..., :physical]
    measure('source_weights_source_physical_zero_workspace')
    # Identify which trained parameter family transports the language change.
    engine.recurrent = original_state.clone()
    families = {
        'language_core': lambda n:n.startswith('body.base.'),
        'encoder': lambda n:n.startswith('encoder.'),
        'working_field': lambda n:'coevolution' in n,
        'dynamics': lambda n:n.startswith('head_coupling.'),
    }
    for family, predicate in families.items():
        with torch.no_grad():
            for name,p in params.items():
                p.copy_(source_weights[name] if predicate(name) else current[name])
        measure('candidate_with_source_'+family)
    engine.model._causal_archive.db.close()
    shutil.rmtree(work)


if __name__ == '__main__':
    main()
