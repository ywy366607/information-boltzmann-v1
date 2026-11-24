"""Certify the actual retained fact/unknown map from a complete living event."""
import argparse
import itertools
import json
from pathlib import Path


def determinant3(m):
    a, b, c = m
    return a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('lean', type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding='utf8'))
    data = report['support_maps'][0]
    facts, columns = data['retained_fact_ids'], data['support_columns']
    n, m = len(facts), len(columns)
    signatures = [tuple(f in column for f in facts) for column in columns]
    reps = [signatures.index(s) for s in signatures]
    unique = sorted(set(reps))
    if data['exact_membership_rank_after'] != 3 or len(unique) != 3:
        raise ValueError('this certificate requires the actual observed three-class map')
    for rows in itertools.combinations(range(n), 3):
        matrix = [[int(facts[r] in columns[c]) for c in unique] for r in rows]
        det = determinant3(matrix)
        if det:
            break
    else:
        raise AssertionError('rank report has no nonzero three-row minor')
    array = lambda xs: '#['+', '.join(json.dumps(x) for x in xs)+']'
    vector = lambda xs: '!['+', '.join(map(str, xs))+']'
    text = f'''import Mathlib.Tactic

/-! Actual normal event 2937, relation 213 generation 4: post-retention
fact-membership observability. This is not a rank theorem about the whole
body or its semantics. Data originate from the complete admitted event pair.
Source operations SHA256: {report['sources'][2]['sha256']}
-/
namespace CGB.NativeFactMembershipActual2401

def unknowns : Array String := {array(data['unknowns'])}
def retainedFacts : Array String := {array(facts)}
def supports : Array (Array String) := #[{', '.join(array(c) for c in columns)}]

def membership (fact : Fin {n}) (unknown : Fin {m}) : Bool :=
  supports[unknown.val]!.contains retainedFacts[fact.val]!

def representative : Fin {m} → Fin {m} := {vector(reps)}

theorem actual_retained_columns_factor :
    ∀ unknown : Fin {m}, ∀ fact : Fin {n},
      membership fact unknown = membership fact (representative unknown) := by native_decide

theorem actual_representatives_are_three :
    ∀ unknown : Fin {m}, {' ∨ '.join(f'representative unknown = {u}' for u in unique)} := by decide

def minorRows : Fin 3 → Fin {n} := {vector(rows)}
def minorColumns : Fin 3 → Fin {m} := {vector(unique)}
def observedMinor : Matrix (Fin 3) (Fin 3) ℚ := fun i j =>
  if membership (minorRows i) (minorColumns j) then 1 else 0

theorem observed_minor_entries : observedMinor = !![{'; '.join(', '.join(map(str, row)) for row in matrix)}] := by native_decide

theorem actual_three_by_three_minor : observedMinor.det = {det} := by
  rw [observed_minor_entries]
  norm_num [Matrix.det_fin_three]

/-- Any linear grouping of these retained fact messages has the same exact
aliases. The weights may be any actual latent coordinate of row messages. -/
def groupedMessage (weights : Fin {n} → ℝ) (unknown : Fin {m}) : ℝ :=
  ∑ fact, weights fact * (if membership fact unknown then 1 else 0)

theorem actual_grouped_messages_factor (weights : Fin {n} → ℝ) (unknown : Fin {m}) :
    groupedMessage weights unknown = groupedMessage weights (representative unknown) := by
  unfold groupedMessage
  apply Finset.sum_congr rfl
  intro fact _
  rw [actual_retained_columns_factor unknown fact]

end CGB.NativeFactMembershipActual2401
'''
    args.lean.write_text(text, encoding='utf8')
    report['fact_membership_certificate'] = dict(path=str(args.lean),
        status='generated_lean_check_pending', selected_relation=data['relation'],
        dimensions=[n, m], representatives=unique, representative_of_each_unknown=reps,
        minor_rows=list(rows), minor_columns=unique, minor_determinant=det)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(report['fact_membership_certificate'], ensure_ascii=False))


if __name__ == '__main__':
    main()
