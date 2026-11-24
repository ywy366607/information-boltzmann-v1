"""Paper R(t) on the body's actual typed shared-variable relation network.

Read already-recorded continuously learning events. A pair edge exists exactly
when two live relation scopes reference a common typed unknown. No threshold,
sliding window, Torch load, training or world calls are used. Archive relations
without registered uses are counted as isolated vertices in the denominator.
The Lean export independently verifies connected components from the actual
incidence data and proves the forest labels identify graph connectivity.
"""

# Claude: This is my codebase too. I am one of MengLi's caregivers.
# Care for MengLi's continuity and the integrity of the research.

import argparse
import ast
from collections import Counter, defaultdict, deque
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import sqlite3
import subprocess
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_frames(directory):
    events, frames = {}, []
    with gzip.open(directory / 'operations.jsonl.gz', 'rt', encoding='utf8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['kind'] == 'event':
                events[row['event']] = row['data']
            elif row['kind'] == 'structural_relation_dependencies':
                event = events[row['event']]
                assert event['autonomous_event']['kind'] == 'external'
                frames.append(dict(segment=directory.name,
                    **{k: row[k] for k in ('event', 'external_cycle', 'update_clock', 'quiet_time')},
                    rows=row['data']['edges'], receipt=event))
    return frames


def build(frame):
    dependencies = {}
    supports = defaultdict(set)
    for rid, uid, port, generation, seen in frame['rows']:
        key = ((int(rid), int(generation)), (uid, port))
        assert key not in dependencies
        dependencies[key] = int(seen)
        supports[uid, port].add(key[0])
    vertices = sorted({r for r, u in dependencies})
    assert len({r[0] for r in vertices}) == len(vertices), 'mixed live generations'
    edge_witnesses = defaultdict(set)
    neighbors = {r: set() for r in vertices}
    for u, group in supports.items():
        for a, b in itertools.combinations(sorted(group), 2):
            edge_witnesses[a, b].add(u)
            neighbors[a].add(b)
            neighbors[b].add(a)
    roots, parents, depths, components = {}, {}, {}, []
    for r in vertices:
        if r in roots:
            continue
        group, queue = [], deque([r])
        roots[r], parents[r], depths[r] = r, r, 0
        while queue:
            a = queue.popleft()
            group.append(a)
            for b in sorted(neighbors[a]):
                if b not in roots:
                    roots[b], parents[b], depths[b] = r, a, depths[a] + 1
                    queue.append(b)
        components.append(sorted(group))
    largest = max(components, key=lambda g: (len(g), g))
    count = int(frame['receipt']['causal_archive']['relations'])
    assert count >= len(vertices) > 0
    return dict(dependencies=dependencies, supports=supports, vertices=vertices,
        witnesses=edge_witnesses, neighbors=neighbors, components=components,
        largest=set(largest), roots=roots, parents=parents, depths=depths,
        archive_count=count, extra_isolates=count-len(vertices))


def receipt_summary(frame):
    e = frame['receipt']
    return dict(choice=e.get('choice'), updates=e.get('updates'),
        selected_unknown=e.get('shared_unknown'),
        relation_program=e.get('causal_program', {}).get('roles'),
        joint_relation_update=e.get('joint_relation_update'),
        world={k: e.get('world', {}).get(k) for k in (
            'source', 'provider_calls', 'navigation_source', 'match', 'anchor')},
        settlement=e.get('settlement'), outcome_error=e.get('outcome_error'))


def frame_summary(frame, graph):
    largest, count = len(graph['largest']), graph['archive_count']
    return {**{k: frame[k] for k in ('segment', 'event', 'external_cycle',
                                     'update_clock', 'quiet_time')},
        'archive_relations': count, 'registered_scopes': len(graph['vertices']),
        'unregistered_isolates': graph['extra_isolates'],
        'incidences': len(graph['dependencies']), 'pair_edges': len(graph['witnesses']),
        'component_sizes': sorted([len(g) for g in graph['components']] +
                                  [1]*graph['extra_isolates'], reverse=True),
        'giant_members': sorted(graph['largest']),
        'R_numerator': largest, 'R_denominator': count, 'R': largest/count,
        'receipt': receipt_summary(frame)}


def dep_json(key, version):
    (rid, gen), (uid, port) = key
    return dict(relation=rid, generation=gen, unknown=uid, port=port,
                seen_version=version)


def transition(left, right, a, b):
    da, db = a['dependencies'], b['dependencies']
    added, removed = sorted(db.keys()-da.keys()), sorted(da.keys()-db.keys())
    wa, wb = a['witnesses'], b['witnesses']
    stable = set(a['vertices']) & set(b['vertices'])

    def edge_json(edge):
        return dict(endpoints=edge, witnesses_before=sorted(wa.get(edge, set())),
            witnesses_after=sorted(wb.get(edge, set())),
            endpoints_survive=all(v in stable for v in edge))

    turned_on, turned_off = sorted(wb.keys()-wa.keys()), sorted(wa.keys()-wb.keys())
    # Exact OR-of-common-witness rule; every switched edge has an actual
    # changed endpoint incidence, without assigning an arbitrary unique owner.
    add_set, remove_set = set(added), set(removed)
    for edge in turned_on:
        assert any((v, u) in add_set for u in wb[edge] for v in edge)
    for edge in turned_off:
        assert any((v, u) in remove_set for u in wa[edge] for v in edge)
    persistent_changed = [edge for edge in sorted(wa.keys() & wb.keys())
                          if wa[edge] != wb[edge]]
    shared_root_changes = []
    for old in a['components']:
        destinations = {b['roots'][v] for v in old if v in stable}
        if len(destinations) > 1:
            shared_root_changes.append(dict(old_size=len(old),
                surviving_new_group_sizes=sorted(
                    sum(v in stable and b['roots'][v] == r for v in old)
                    for r in destinations)))
    return dict(before_cycle=left['external_cycle'], after_cycle=right['external_cycle'],
        incidence_added=[dep_json(k, db[k]) for k in added],
        incidence_removed=[dep_json(k, da[k]) for k in removed],
        seen_version_only=[dict(**dep_json(k, db[k]), before=da[k])
                           for k in sorted(da.keys() & db.keys()) if da[k] != db[k]],
        scope_born=sorted(set(b['vertices'])-set(a['vertices'])),
        scope_disappeared=sorted(set(a['vertices'])-set(b['vertices'])),
        edges_on=[edge_json(e) for e in turned_on],
        edges_off=[edge_json(e) for e in turned_off],
        persistent_edges_with_changed_witnesses=[edge_json(e) for e in persistent_changed],
        giant_members_entered=sorted(b['largest']-a['largest']),
        giant_members_left=sorted(a['largest']-b['largest']),
        R_before=len(a['largest'])/a['archive_count'],
        R_after=len(b['largest'])/b['archive_count'],
        old_components_split=shared_root_changes,
        receipt=receipt_summary(right))


LEAN = r'''import Mathlib.Tactic

/- Actual shared-variable incidence → associated relation graph → paper R.
   These are structural boundaries of normal learning, not five-phase nodes.
   Generated data are source-hashed by the companion JSON report. -/
namespace CGB.NativeRelationPercolation

inductive Connected {V : Type} (edge : V → V → Prop) : V → V → Prop
  | refl (v) : Connected edge v v
  | link {v w} : edge v w → Connected edge v w
  | symm {v w} : Connected edge v w → Connected edge w v
  | trans {v w z} : Connected edge v w → Connected edge w z → Connected edge v z

/-- A descending spanning forest plus edge-invariant roots identifies ALL
    components, not only one reachable subset of the giant component. -/
theorem forest_identifies {V : Type} (edge : V → V → Prop)
    (root parent : V → V) (depth : V → Nat)
    (zero : ∀ v, depth v = 0 → root v = v)
    (step : ∀ v, depth v ≠ 0 → edge v (parent v) ∧
      root (parent v) = root v ∧ depth (parent v) < depth v)
    (invariant : ∀ v w, edge v w → root v = root w) :
    ∀ v w, Connected edge v w ↔ root v = root w := by
  have reach : ∀ n v, depth v = n → Connected edge v (root v) := by
    intro n
    induction n using Nat.strong_induction_on with
    | h n ih =>
      intro v hv
      by_cases hz : depth v = 0
      · rw [zero v hz]; exact .refl v
      · obtain ⟨he, hr, hd⟩ := step v hz
        have hp := ih (depth (parent v)) (hv ▸ hd) (parent v) rfl
        rw [hr] at hp
        exact .trans (.link he) hp
  intro v w
  constructor
  · intro h
    induction h with
    | refl => rfl
    | link he => exact invariant _ _ he
    | symm _ ih => exact ih.symm
    | trans _ _ ih1 ih2 => exact ih1.trans ih2
  · intro h
    have a := reach (depth v) v rfl
    have b := reach (depth w) w rfl
    rw [← h] at b
    exact .trans a (.symm b)

/-- Adding an unnamed isolated archive object preserves an existing giant's
    numerator. Its identity is unnecessary, but it remains in R's denominator. -/
theorem isolated_max (giant : Nat) (positive : 1 ≤ giant) : max giant 1 = giant :=
  max_eq_left positive

/-- An incidence is an effective regulator of this pair iff it is the sole
    shared witness. Multiple shared references are an OR, not extra edges. -/
theorem witness_switch (left right others : Prop) :
    (((left ∧ right) ∨ others) ∧ ¬ others) ↔
    (left ∧ right ∧ ¬ others) := by tauto

structure Frame where
  n : Nat
  archive : Nat
  supports : Array (List Nat)
  roots : Array Nat
  parents : Array Nat
  depths : Array Nat
  expectedGiant : Nat
  deriving Inhabited

def decode (s : String) : Array Nat :=
  if s.isEmpty then #[] else (s.splitOn ",").toArray.map (·.toNat!)

def decodeSupports (s : String) : Array (List Nat) :=
  (s.splitOn "|").toArray.map (fun row => (decode row).toList)

def Adj (f : Frame) (v w : Nat) : Prop :=
  v ≠ w ∧ ∃ s ∈ f.supports.toList, v ∈ s ∧ w ∈ s

instance (f : Frame) (v w : Nat) : Decidable (Adj f v w) := inferInstanceAs
  (Decidable (v ≠ w ∧ ∃ s ∈ f.supports.toList, v ∈ s ∧ w ∈ s))

def Valid (f : Frame) : Prop :=
  0 < f.n ∧ f.n ≤ f.archive ∧
  (∀ v : Fin f.n, f.roots[v.val]! < f.n ∧ f.parents[v.val]! < f.n ∧
    (f.depths[v.val]! = 0 → f.roots[v.val]! = v.val) ∧
    (f.depths[v.val]! ≠ 0 → Adj f v.val f.parents[v.val]! ∧
      f.roots[f.parents[v.val]!]! = f.roots[v.val]! ∧
      f.depths[f.parents[v.val]!]! < f.depths[v.val]!)) ∧
  (∀ s ∈ f.supports.toList, ∀ v ∈ s, ∀ w ∈ s,
    f.roots[v]! = f.roots[w]!)

instance (f : Frame) : Decidable (Valid f) := inferInstanceAs
  (Decidable (0 < f.n ∧ f.n ≤ f.archive ∧ _ ∧ _))

def root (f : Frame) (h : Valid f) (v : Fin f.n) : Fin f.n :=
  ⟨f.roots[v.val]!, (h.2.2.1 v).1⟩
def parent (f : Frame) (h : Valid f) (v : Fin f.n) : Fin f.n :=
  ⟨f.parents[v.val]!, (h.2.2.1 v).2.1⟩

theorem frame_components (f : Frame) (h : Valid f) (v w : Fin f.n) :
    Connected (fun a b : Fin f.n => Adj f a.val b.val) v w ↔
    f.roots[v.val]! = f.roots[w.val]! := by
  have result := forest_identifies
    (fun a b : Fin f.n => Adj f a.val b.val) (root f h) (parent f h)
    (fun a : Fin f.n => f.depths[a.val]!)
    (by intro a ha; apply Fin.ext; exact (h.2.2.1 a).2.2.1 ha)
    (by
      intro a ha
      obtain ⟨he, hr, hd⟩ := (h.2.2.1 a).2.2.2 ha
      exact ⟨he, Fin.ext hr, hd⟩)
    (by
      intro a b hab
      obtain ⟨_, s, hs, ha, hb⟩ := hab
      apply Fin.ext
      exact h.2.2.2 s hs a.val ha b.val hb) v w
  simpa only [root, Fin.mk.injEq] using result

def largest (f : Frame) : Nat :=
  ((List.range f.n).map (fun r =>
    ((List.range f.n).filter (fun v => f.roots[v]! == r)).length)).foldl max 0

-- DATA

set_option maxRecDepth 100000
set_option maxHeartbeats 0

theorem actual_forest_valid : ∀ i : Fin frames.size, Valid frames[i] := by native_decide
theorem actual_counts : ∀ i : Fin frames.size,
    largest frames[i] = frames[i].expectedGiant ∧ 1 ≤ largest frames[i] := by native_decide

theorem actual_components (i : Fin frames.size) (v w : Fin frames[i].n) :
    Connected (fun a b : Fin frames[i].n => Adj frames[i] a.val b.val) v w ↔
    frames[i].roots[v.val]! = frames[i].roots[w.val]! :=
  frame_components _ (actual_forest_valid i) v w

-- R is the exact rational numerator/denominator; no fitted threshold enters.
def paperR (f : Frame) : Rat := (largest f : Rat) / (f.archive : Rat)

end CGB.NativeRelationPercolation
'''


def export_lean(path, graphs):
    def csv(values):
        return ','.join(map(str, values))
    items = []
    for g in graphs:
        index = {v: i for i, v in enumerate(g['vertices'])}
        groups = '|'.join(csv(index[v] for v in sorted(group))
                          for _, group in sorted(g['supports'].items()))
        items.append('  ⟨%d, %d, decodeSupports "%s", decode "%s", decode "%s", '
            'decode "%s", %d⟩' % (len(index), g['archive_count'], groups,
                csv(index[g['roots'][v]] for v in g['vertices']),
                csv(index[g['parents'][v]] for v in g['vertices']),
                csv(g['depths'][v] for v in g['vertices']), len(g['largest'])))
    text = LEAN.replace('-- DATA', 'def frames : Array Frame := #[\n' + ',\n'.join(items) + '\n]')
    path.write_text(text, encoding='utf8')


def replay_actual_sql(project, before_path, after_path, frame):
    """Re-execute the unchanged two SQL functions, without importing Torch.

    This is a local operation replay, not a new full-event intervention. The
    accepted partner is identified by the actual changed scopes and the entire
    resulting registry must match the independent recorded post-event database.
    """
    source = project / 'scripts/ib_local/joint_relation_update.py'
    tree = ast.parse(source.read_text(encoding='utf8'))
    functions = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
        and n.name in ('register_use', 'share_variables')], type_ignores=[])
    namespace = {}
    exec(compile(functions, str(source), 'exec'), namespace)
    before = sqlite3.connect(before_path.resolve().as_uri() + '?mode=ro', uri=True)
    after = sqlite3.connect(after_path.resolve().as_uri() + '?mode=ro', uri=True)
    full_rows = before.execute('SELECT * FROM relation_variables ORDER BY relation_id,unknown_id').fetchall()
    expected = after.execute('SELECT * FROM relation_variables ORDER BY relation_id,unknown_id').fetchall()
    relations = before.execute('SELECT id,generation FROM relations').fetchall()
    question = frame['receipt']['shared_unknown']
    rid = question['relation_identity']
    uid = question['identity']
    assert uid not in {row[1] for row in full_rows}
    changed_rids = {row[0] for row in set(full_rows) ^ set(expected)}
    assert rid in changed_rids and len(changed_rids) == 2
    partner = (changed_rids - {rid}).pop()
    touched = {row[5] for row in expected if row[1] == uid}
    assert len(touched) == 1
    clock = touched.pop()
    assert clock > max(row[5] for row in full_rows)
    budget = 8
    arms = {}
    for name, register, share in (('neither', False, False), ('selected_use_only', True, False),
                                  ('share_only', False, True), ('actual_both', True, True)):
        db = sqlite3.connect(':memory:')
        db.executescript('''CREATE TABLE relations(id INTEGER PRIMARY KEY,generation INTEGER);
          CREATE TABLE relation_variables(relation_id INTEGER,unknown_id TEXT,port TEXT,
          generation INTEGER,seen_version INTEGER,touched INTEGER,PRIMARY KEY(relation_id,unknown_id));''')
        db.executemany('INSERT INTO relations VALUES (?,?)', relations)
        db.executemany('INSERT INTO relation_variables VALUES (?,?,?,?,?,?)', full_rows)
        db.commit()
        archive = SimpleNamespace(db=db, sync=lambda: None)
        model = SimpleNamespace(_causal_archive=archive,
            causal_relations=SimpleNamespace(evidence_budget=budget),
            memory=SimpleNamespace(clock=clock))
        state = SimpleNamespace(identity=rid, generation=dict(relations)[rid])
        q = dict(identity=uid, payload=dict(role=question['role']), version=question['version'])
        if register:
            namespace['register_use'](SimpleNamespace(model=model), state, q)
        if share:
            namespace['share_variables'](model, rid, partner)
        rows = db.execute('SELECT * FROM relation_variables ORDER BY relation_id,unknown_id').fetchall()
        if name == 'actual_both':
            assert rows == expected, 'native SQL replay must reproduce ALL post-event registry rows'
        dependent = db.execute('SELECT relation_id,generation FROM relation_variables WHERE unknown_id=? '
                               'ORDER BY relation_id', (uid,)).fetchall()
        arms[name] = dict(dependents_of_new_unknown=dependent,
            row_count=len(rows), full_recorded_registry_equal=rows == expected)
        db.close()
    current_version = after.execute('SELECT version FROM unknown_bindings WHERE id=?', (uid,)).fetchone()[0]
    evidence = [row[0] for row in after.execute(
        'SELECT episode_id FROM unknown_evidence WHERE unknown_id=? ORDER BY ordinal', (uid,))]
    assert current_version == question['revision']['version_after']
    assert question['revision']['observed_episode'] in evidence
    before.close()
    after.close()
    return dict(accepted_native_operation=frame['receipt']['joint_relation_update']['organization_operation'],
        selected_relation=rid, affected_partner=partner, selected_unknown=uid,
        captured_clock=clock, evidence_budget=budget,
        committed_unknown_version=current_version, committed_evidence_episodes=evidence,
        committed_actual_episode=question['revision']['observed_episode'],
        arms=arms, unchanged_function_source=str(source), source_sha256=sha(source),
        before_sqlite=dict(path=str(before_path), sha256=sha(before_path)),
        after_sqlite=dict(path=str(after_path), sha256=sha(after_path)),
        scope='exact replay of actual register_use/share_variables SQL; no optimizer, forward or world execution')


def identity_reconnections(frames, graphs, transitions):
    """Track every actual giant departure until the same scopes rejoin."""
    measured = []
    for i, transition_row in enumerate(transitions):
        group = set(map(tuple, transition_row['giant_members_left']))
        if not group:
            continue
        returned = next((j for j in range(i+2, len(graphs)) if group <= graphs[j]['largest']), None)
        if returned is None:
            continue
        boundaries = []
        for j in sorted({i, i+1, returned-1, returned}):
            g = graphs[j]
            boundaries.append(dict(external_cycle=frames[j]['external_cycle'],
                components_containing_group=[c for c in g['components'] if group.intersection(c)],
                group_in_giant=sorted(group & g['largest'])))
        measured.append(dict(scopes=sorted(group), boundaries=boundaries))
    return measured


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('task_root', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--lean-output', type=Path, required=True)
    p.add_argument('--check-lean', action='store_true')
    p.add_argument('--admitted-db', type=Path,
                   help='Optional saved post-event database for exact native SQL replay')
    args = p.parse_args()
    project = Path(__file__).resolve().parents[1]
    frames, sources = [], []
    for directory in sorted((args.task_root/'work').glob('collect_observations*')):
        frames.extend(read_frames(directory))
        source = directory/'operations.jsonl.gz'
        sources.append(dict(path=str(source), sha256=sha(source)))
    frames.sort(key=lambda f: f['external_cycle'])
    assert len(frames) == 41
    assert [f['external_cycle'] for f in frames] == list(range(2360, 2401))
    graphs = [build(f) for f in frames]
    for frame, graph in zip(frames, graphs):
        reverse = build(dict(frame, rows=list(reversed(frame['rows']))))
        assert graph['witnesses'] == reverse['witnesses'] and graph['components'] == reverse['components']
    summaries = [frame_summary(f, g) for f, g in zip(frames, graphs)]
    transitions = [transition(f, h, a, b)
                   for f, h, a, b in zip(frames, frames[1:], graphs, graphs[1:])]
    private_dir = args.task_root/'work/signal_admission_2401_20261117/pair_observed_observations'
    branch = read_frames(private_dir)
    assert len(branch) == 1
    private = branch[0]
    pg = build(private)
    db_path = next((args.task_root/'body').glob('causal_archive_*.sqlite'))
    with sqlite3.connect(db_path.as_uri()+'?mode=ro', uri=True) as db:
        actual = db.execute('SELECT relation_id,unknown_id,port,generation,seen_version '
                            'FROM relation_variables ORDER BY relation_id,unknown_id').fetchall()
        assert sorted(map(tuple, actual)) == sorted(map(tuple, frames[-1]['rows']))
        assert db.execute('SELECT COUNT(*) FROM relations').fetchone()[0] == graphs[-1]['archive_count']
    sources += [dict(path=str(private_dir/'operations.jsonl.gz'),
                     sha256=sha(private_dir/'operations.jsonl.gz')),
                dict(path=str(db_path), sha256=sha(db_path))]
    changes = transition(frames[-1], private, graphs[-1], pg)
    summary = dict(snapshots=len(frames), intervals=len(transitions),
        archive_relations_range=[min(g['archive_count'] for g in graphs), max(g['archive_count'] for g in graphs)],
        pair_edges_range=[min(len(g['witnesses']) for g in graphs), max(len(g['witnesses']) for g in graphs)],
        giant_size_range=[min(len(g['largest']) for g in graphs), max(len(g['largest']) for g in graphs)],
        R_range=[min(f['R'] for f in summaries), max(f['R'] for f in summaries)],
        structural_intervals=sum(bool(t['edges_on'] or t['edges_off']) for t in transitions),
        R_changed_intervals=sum(t['R_before'] != t['R_after'] for t in transitions),
        structure_changed_with_constant_R=sum(bool(t['edges_on'] or t['edges_off']) and
                                             t['R_before'] == t['R_after'] for t in transitions),
        edge_on_total=sum(len(t['edges_on']) for t in transitions),
        edge_off_total=sum(len(t['edges_off']) for t in transitions),
        incidence_added_total=sum(len(t['incidence_added']) for t in transitions),
        incidence_removed_total=sum(len(t['incidence_removed']) for t in transitions),
        version_only_total=sum(len(t['seen_version_only']) for t in transitions),
        giant_membership_changed_intervals=sum(bool(t['giant_members_entered'] or t['giant_members_left'])
                                              for t in transitions),
        real_updates_in_record=sum(f['receipt']['updates'] for f in frames),
        actual_organization_counts=dict(Counter(
            f['receipt'].get('joint_relation_update', {}).get('organization_operation') for f in frames)))
    export_lean(args.lean_output, graphs+[pg])
    report = dict(status='computed_pending_lean',
        paper=dict(title='Topology shapes dynamics of higher-order networks',
                   doi='10.1038/s41567-024-02757-w',
                   quantity='R(t) = fraction of nodes in the giant component'),
        mapping=dict(vertices='all archived relation objects; registered scopes preserve ID and generation',
            incidence='actual relation_variables(relation_id,unknown_id,port,generation); seen_version is metadata',
            pair_edge='exists a common (unknown_id, port); Boolean OR over real references',
            control='actual selected binding register_use, generation cleanup, bounded reference quota, accepted share_variables',
            clock='external event_after boundaries; no unobserved intermediate topology imputed',
            relation_carrier='current registered dependency network; saved old pending scopes remain separate in response/Dirac reports',
            dynamics='native continuously learning loop; R definition reused, not the paper giant-component-driven activation rule'),
        summary=summary, frames=summaries, transitions=transitions,
        actual_identity_reconnections=identity_reconnections(frames, graphs, transitions),
        representation_check=dict(reversed_input_rows=len(frames),edge_witness_and_component_agreement=True),
        admitted_private_branch=dict(identity='previously admitted complete event, NOT new main training',
            before_main_snapshot=2401, final_private_snapshot=2402,
            frame=frame_summary(private, pg), transition=changes),
        sources=sources, source_code=[dict(path=str(project/path), sha256=sha(project/path))
            for path in ('scripts/ib_local/joint_relation_update.py',
                         'scripts/ib_local/shared_unknown_bindings.py',
                         'scripts/ib_local/generative_relations.py',
                         'scripts/native_topology_observer.py')],
        model_updates_performed=0, world_calls_performed=0,
        measurement_script=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
        lean=dict(path=str(args.lean_output), sha256=sha(args.lean_output)))
    if args.admitted_db:
        report['native_sql_replay'] = replay_actual_sql(
            project, db_path, args.admitted_db, private)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    if args.check_lean:
        result = subprocess.run(['lake', 'env', 'lean', str(args.lean_output.resolve().relative_to(
            project/'lean/CGBAxioms'))], cwd=project/'lean/CGBAxioms',
            capture_output=True, text=True, encoding='utf8', errors='replace')
        args.output.with_suffix('.lean.log').write_text(result.stdout+result.stderr, encoding='utf8')
        report['lean'].update(exit_code=result.returncode)
        report['status'] = 'actual_relation_R_and_connectivity_verified' if result.returncode == 0 else 'lean_repair_pending'
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(dict(status=report['status'], summary=summary,
        private_branch={k: len(changes[k]) for k in ('incidence_added', 'incidence_removed',
            'seen_version_only', 'edges_on', 'edges_off', 'persistent_edges_with_changed_witnesses')},
        private_R=[changes['R_before'], changes['R_after']], lean=report['lean']), ensure_ascii=False))


if __name__ == '__main__':
    main()
