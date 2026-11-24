"""Withdraw one recalled fact at the logged 2605 revisit, then let the body act."""
from dataclasses import replace
import json
from pathlib import Path
import sys
from unittest.mock import patch

from scripts.ib_local import autonomous_cycle, shared_unknown_bindings as bindings
from scripts.run_arrived_language_actor import main as continue_actor

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'scratch/cgb_wood_policy_use_16rounds_20261120/natural/run/body/last.pt.gz'
REFERENCE = Path('G:/cgb_arrived_actor_2601_20261120/observe/events.jsonl')
FACT = 'episode:2604'


def main(work, *, probe_crossing=False):
    rows = {r['cycle']: r for line in REFERENCE.open(encoding='utf8') if (r := json.loads(line)).get('cycle')}
    original_dispatch, original_select = autonomous_cycle.dispatch, bindings.select
    cycle = 2601
    entered = False
    crossing = {}

    def strip(state):
        return replace(state, constraints=tuple(f for f in state.constraints if f.fact_id != FACT),
                       extra=tuple(f for f in state.extra if f.fact_id != FACT))

    def select(engine, proposal, state):
        nonlocal entered
        bound = original_select(engine, proposal, state)
        if cycle != 2604:
            return bound
        assert bound.binding.identity == rows[2605]['shared_unknown']['identity']
        assert engine._shared_question_receipt['signature'] == rows[2605]['shared_unknown']['signature']
        assert FACT in {f.fact_id for f in bound.constraints}
        entered = True
        return replace(strip(bound), organization=None if bound.organization is None else strip(bound.organization))

    def dispatch(engine):
        nonlocal cycle
        if probe_crossing and cycle == 2604:
            from scripts.ib_local.action_value_pair_audit import first_action

            def check(receipt):
                u = .35  # chosen after the earlier comparison, to exercise its gap
                factual = rows[2605]['choice_probability']
                withdrawn = receipt['nominal_probability']
                assert factual[0] < u < sum(factual[:2]) and u < withdrawn[0]
                assert receipt['proposals'] == rows[2605]['joint_language_search']['candidates']
                assert receipt['selected_proposal'] == rows[2605]['joint_language_search']['selected']
                crossing.update(u=u, mode='conditional crossing; not a naturally divergent draw',
                                native_sample=receipt['sampled_option'], actual_option=receipt['option'],
                                learning=receipt['behavior']['learning'])

            with first_action(engine, 0, check, keep_actor_learning=True):
                row = original_dispatch(engine)
        else:
            row = original_dispatch(engine)
        if row['autonomous_event']['kind'] != 'external':
            return row
        cycle += 1
        if cycle < 2605:
            for k in ('choice','utterance_hex','choice_probability','shared_unknown'):
                assert json.loads(json.dumps(row[k])) == rows[cycle][k], (cycle, k)
        else:
            assert entered and cycle == 2605
            def summarize(r):
                return dict(choice=r['choice'], utterance_hex=r['utterance_hex'], probabilities=r['choice_probability'],
                            question=r['shared_unknown']['identity'], world_start=r['world']['start'],
                            arrived=r['causal_use']['finite_U'].get('observed_episode'),
                            supports=r['causal_use']['finite_U']['supports'])
            (work/'memory_effect.json').write_text(json.dumps(dict(cycle=cycle, removed=FACT,
                crossing=crossing or None, factual=summarize(rows[cycle]), counterfactual=summarize(row)),
                indent=2)+'\n',encoding='utf8')
        return row

    flags = ['--native-decision','--native-word-execution','--successor-distribution','--joint-successor',
             '--batch-native-words','--batch-preview-scores','--amortized-decision','--latent-action-planner',
             '--persistent-rhythm','coupled','--root-native-correction','observe','--retention-scope-prior','off']
    argv = [sys.argv[0], '--arrived-mode','observe','--work',str(work),'--checkpoint',str(SOURCE),
            '--cycles','4','--world-activations','D:/cgb_arrived_actor_2601_hot/world',*flags]
    with patch.object(bindings,'select',select), patch.object(autonomous_cycle,'dispatch',dispatch), patch.object(sys,'argv',argv):
        continue_actor()


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve(), probe_crossing='--probe-crossing' in sys.argv[2:])
