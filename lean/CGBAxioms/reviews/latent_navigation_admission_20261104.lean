import CGBAxioms.GenerationSelfProduction

/-! Admission examples, reusing existing declarations rather than adding new
theorem counts. These are real-arithmetic local memory laws, not a proof of
Python execution, semantic sufficiency, anti-collapse or autonomous survival.
The runtime's FIFO assignment is the beta=1 one-hot endpoint. Its pending
queue contains only bounded observed features; target data is unavailable to
the read until the next commit. Soft addressing does not imply exact readout.
-/

open CGB.Generation

example {n : Nat} (slot : Fin n) (row : Fin n → ℝ) (value : ℝ) :
    dot (fun j => if j = slot then 1 else 0)
      (tiedWrite (fun j => if j = slot then 1 else 0) 1 value row) = value := by
  have hk : dot (fun j : Fin n => if j = slot then 1 else 0)
      (fun j => if j = slot then 1 else 0) = 1 := by simp [dot]
  have h := tied_write_error_shrinks (fun j => if j = slot then 1 else 0) row hk 1 value
  linarith

example {n : Nat} (slot : Fin n) (row query : Fin n → ℝ) (value : ℝ) :
    dot query (tiedWrite (fun j => if j = slot then 1 else 0) 1 value row) -
      dot query row = (value - row slot) * query slot := by
  simpa [dot] using tied_write_cross_read
    (fun j => if j = slot then 1 else 0) query row 1 value

example {n : Nat} (slot other : Fin n) (row : Fin n → ℝ) (value : ℝ)
    (h : other ≠ slot) :
    tiedWrite (fun j => if j = slot then 1 else 0) 1 value row other = row other := by
  simp [tiedWrite, h]

-- A linear encoder commutes with the frozen-weight mixture read. Encoding
-- stored features with today's projector avoids stale projector coordinates;
-- it does NOT reconstruct historical GDN features under today's GDN weights.
example {n d z : Nat} (w : Fin n → ℝ) (v : Fin n → Fin d → ℝ)
    (E : Fin z → Fin d → ℝ) (o : Fin z) :
    (∑ j, E o j * (∑ i, w i * v i j)) =
      ∑ i, w i * (∑ j, E o j * v i j) := by
  simp_rw [Finset.mul_sum]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  ring
