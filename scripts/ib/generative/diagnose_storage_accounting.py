"""Replay actual failed intervals, shadowing original ledger increments only.

AST instrumentation observes each exact RHS before legacy float addition.
It changes no physical transition, random draw, or production source method.
Unbounded increment lists are diagnostic data, never persistent brain state.
"""
import ast
import copy
import inspect
import json
import math
from pathlib import Path
import sys
import textwrap

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from information_boltzmann.cgb.exchange_checkpoint import load_exchange
from information_boltzmann.cgb.mobile_exchange import MobileExchange
from information_boltzmann.cgb.growing_material import GrowingIdentityMobileExchange
from scripts.ib.generative.native_learning_stream import advance_byte, atomic_json


NAMES = {'heat_out', 'work_in', 'entropy_balance', 'reaction_heat',
         'motion_heat', 'conduction_heat', 'source_heat', 'domain_heat',
         'source_charge', 'electrode_charge'}


class ObserveAdds(ast.NodeTransformer):
    def visit_AugAssign(self, node):
        target = node.target
        if (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                and target.value.id == 'self' and target.attr in NAMES):
            assert isinstance(node.op, (ast.Add, ast.Sub))
            delta = node.value if isinstance(node.op, ast.Add) else ast.UnaryOp(ast.USub(), node.value)
            return ast.copy_location(ast.Expr(ast.Call(
                ast.Attribute(ast.Name('self', ast.Load()), '_observe_ledger_add', ast.Load()),
                [ast.Constant(target.attr), delta], [])), node)
        return self.generic_visit(node)


def observe(self, name, delta):
    self._ledger_increments[name].append(float(delta))
    setattr(self, name, getattr(self, name) + delta)


def main(run):
    m, meta = load_exchange(run/'polarized_start.npz')
    initial = {name: float(getattr(m, name)) for name in NAMES}
    m._ledger_increments = {name: [] for name in NAMES}
    MobileExchange._observe_ledger_add = observe
    changed = []
    for cls in (MobileExchange, GrowingIdentityMobileExchange):
        snapshot = run/'source_snapshot/information_boltzmann/cgb'/Path(inspect.getfile(cls)).name
        original = ast.parse(snapshot.read_text())
        definition = next(n for n in original.body if isinstance(n, ast.ClassDef) and n.name == cls.__name__)
        methods = {n.name:n for n in definition.body if isinstance(n, ast.FunctionDef)}
        for name, method in list(cls.__dict__.items()):
            if not inspect.isfunction(method) or name not in methods:
                continue
            tree = ast.Module(body=[copy.deepcopy(methods[name])], type_ignores=[])
            has_updates = any(isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Attribute)
                              and n.target.attr in NAMES for n in ast.walk(tree))
            if not has_updates:
                continue
            assert not method.__closure__, name
            code = ast.fix_missing_locations(ObserveAdds().visit(tree))
            namespace = dict(method.__globals__)
            exec(compile(code, '<diagnostic_ledger_observation>', 'exec'), namespace)
            setattr(cls, name, namespace[name])
            changed.append(cls.__name__+'.'+name)
    session = copy.deepcopy(meta['session'])
    expected = [json.loads(line) for line in (run/'polarized_intervals.jsonl').read_text().splitlines()]
    windows = []
    for i, row in enumerate(expected):
        actual = advance_byte(m, session, row['target'], row['phase'])
        assert all(actual[k] == row[k] for k in actual)
        accurate = {k: math.fsum([initial[k], *m._ledger_increments[k]]) for k in NAMES}
        residuals = dict(
            energy=math.fsum([m.energy(), accurate['heat_out'], -accurate['work_in'], -m.initial_energy]),
            free_energy=math.fsum([m.initial_free_energy, accurate['work_in'], -m.free_energy(),
                                   -accurate['entropy_balance']]))
        if (i+1) % 16 == 0 or i+1 == len(expected):
            item = dict(interval=i+1, time_ms=m.t, legacy_balances={k:float(v) for k,v in m.balances().items()},
                        fsum_balances=residuals,
                        ledger_float_minus_fsum={k: float(getattr(m,k))-accurate[k] for k in NAMES})
            windows.append(item)
            print(json.dumps(item), flush=True)
    atomic_json(run/'accounting_diagnosis.json', dict(
        actual_failed_intervals=len(expected), replay_scores_exact=True,
        observed_methods=changed, increments={k: len(v) for k,v in m._ledger_increments.items()},
        initial_balances={k:float(v) for k,v in load_exchange(run/'polarized_start.npz')[0].balances().items()},
        windows=windows, scope='Exact recorded-input replay and observed arithmetic RHS; no physics or threshold change'))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
