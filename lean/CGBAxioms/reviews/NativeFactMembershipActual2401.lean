import Mathlib.Tactic

/-! Actual normal event 2937, relation 213 generation 4: post-retention
fact-membership observability. This is not a rank theorem about the whole
body or its semantics. Data originate from the complete admitted event pair.
Source operations SHA256: 11badca1cfe2c9b9e942b444a913434259b9b0d18e6460a718d6d39aa111fdd0
-/
namespace CGB.NativeFactMembershipActual2401

def unknowns : Array String := #["unknown:1643", "unknown:1677", "unknown:1673", "unknown:1293", "unknown:1648", "unknown:1640", "unknown:1634", "unknown:1629"]
def retainedFacts : Array String := #["episode:1133", "episode:1616", "episode:1919", "episode:1977", "episode:203", "episode:2216", "episode:267", "episode:283"]
def supports : Array (Array String) := #[#["episode:1133", "episode:1616", "episode:1919", "episode:1977", "episode:203", "episode:2216", "episode:2339", "episode:267"], #["episode:1133", "episode:1616", "episode:1919", "episode:1977", "episode:203", "episode:2216", "episode:267", "episode:283"], #["episode:1133", "episode:1616", "episode:1919", "episode:1977", "episode:203", "episode:2216", "episode:267", "episode:283"], #["episode:104", "episode:1321", "episode:203", "episode:2366", "episode:263", "episode:267", "episode:283", "episode:929"], #["episode:1133", "episode:1616", "episode:1919", "episode:1977", "episode:203", "episode:2216", "episode:2349", "episode:267"], #["episode:1133", "episode:1616", "episode:1919", "episode:1977", "episode:203", "episode:2216", "episode:2336", "episode:267"], #["episode:1133", "episode:1616", "episode:1919", "episode:1977", "episode:203", "episode:2216", "episode:2326", "episode:267"], #["episode:1133", "episode:1616", "episode:1919", "episode:1977", "episode:203", "episode:2216", "episode:2319", "episode:267"]]

def membership (fact : Fin 8) (unknown : Fin 8) : Bool :=
  supports[unknown.val]!.contains retainedFacts[fact.val]!

def representative : Fin 8 → Fin 8 := ![0, 1, 1, 3, 0, 0, 0, 0]

theorem actual_retained_columns_factor :
    ∀ unknown : Fin 8, ∀ fact : Fin 8,
      membership fact unknown = membership fact (representative unknown) := by native_decide

theorem actual_representatives_are_three :
    ∀ unknown : Fin 8, representative unknown = 0 ∨ representative unknown = 1 ∨ representative unknown = 3 := by decide

def minorRows : Fin 3 → Fin 8 := ![0, 4, 7]
def minorColumns : Fin 3 → Fin 8 := ![0, 1, 3]
def observedMinor : Matrix (Fin 3) (Fin 3) ℚ := fun i j =>
  if membership (minorRows i) (minorColumns j) then 1 else 0

theorem observed_minor_entries : observedMinor = !![1, 1, 0; 1, 1, 1; 0, 1, 1] := by native_decide

theorem actual_three_by_three_minor : observedMinor.det = -1 := by
  rw [observed_minor_entries]
  norm_num [Matrix.det_fin_three]

/-- Any linear grouping of these retained fact messages has the same exact
aliases. The weights may be any actual latent coordinate of row messages. -/
def groupedMessage (weights : Fin 8 → ℝ) (unknown : Fin 8) : ℝ :=
  ∑ fact, weights fact * (if membership fact unknown then 1 else 0)

theorem actual_grouped_messages_factor (weights : Fin 8 → ℝ) (unknown : Fin 8) :
    groupedMessage weights unknown = groupedMessage weights (representative unknown) := by
  unfold groupedMessage
  apply Finset.sum_congr rfl
  intro fact _
  rw [actual_retained_columns_factor unknown fact]

end CGB.NativeFactMembershipActual2401
