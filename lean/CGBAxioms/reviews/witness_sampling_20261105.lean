import CGBAxioms.GenerationFutureValue

/-! Local sampling admission. These do not assert that past observations have
the distribution of future language or that a data-dependent trained model has
an unbiased generalization score. They verify the reservoir marginal step and
the corresponding finite linear average for fixed losses. -/

-- After n observations with capacity k, an old retained item survives with
-- probability 1-1/(n+1); its marginal inclusion remains k/(n+1).
example (k n : ℝ) (hn : n ≠ 0) (hn1 : n+1 ≠ 0) :
    k / n * (1 - 1 / (n+1)) = k / (n+1) := by
  field_simp
  ring

-- Equal marginal inclusion gives the fixed finite population's mean loss.
example {ι : Type*} (s : Finset ι) (loss : ι → ℝ) (k n : ℝ)
    (hk : k ≠ 0) (hn : n ≠ 0) :
    (∑ i ∈ s, (k / n) * loss i / k) = ∑ i ∈ s, loss i / n := by
  apply Finset.sum_congr rfl
  intro i hi
  field_simp
