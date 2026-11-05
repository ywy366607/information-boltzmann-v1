import CGBAxioms.GenerationHoldCorrect

/-!
# Latch slot: renewal that does not ride on the input gate (generation chain, 2026-11-04)

Motivation (`results/cgb_rich_world_20261104/RESULT.md`, `results/cgb_mode_task_20261103/`):
the trained latch heads are erased by ordinary text (E2), and in the rich world the latching
heads were taken by the frequent medium state while the long state stayed in slowly decaying
heads. The user's question: biological latches are actively firing; is the passive store
drowned by the world's writes, and which is larger in a brain, the world's input or its own
state? In cortex most synapses are recurrent and the input is gated before it reaches the
circuits that hold state. In `feedback_delta` the renewal and the input are not separated.

In `feedback_delta` (`GenerationHoldCorrect`, Part 2), read along the event's key with value 0,
one event is `fb a c g` with `a = α (1 - β)` and `c = β η`, where `β` is the event's write gate
(`sigmoid (W x)`, the same gate that admits text). Hence:

* `coupled_closed_gate_decays`: closing the gate (`β = 0`) leaves pure decay `α r`.
* `coupled_small_gate_subcritical`: any gate with `β η g ≤ 1 - α` is subcritical (κ ≤ 1), so
  by `subcritical_forgets` the content shrinks every event. A head cannot close its gate to
  text without switching off its own renewal.
* `coupled_latch_gate_floor`, `coupled_disturbance_floor`: conversely, while the head latches
  (κ > 1), `β > (1-α)/(η g - α)`, so the text it admits writes at least
  `(1-α)(1-η)|v| / (η g - α)` per event: a floor that gating cannot remove. Only `α → 1`
  removes it, and then the head is a pure hold (`hold_accumulates`), not a corrector.

The slot separates the two. Each latch head has a fixed learned direction `u` renewed at every
event with its own rate `ρ`, independent of the input (event `(u, value 0, α = 1, β = ρ)`), after
the ordinary write, which still decays the whole state by the event's `α`:

* `slot_composite`: decay then renewal is again the family: `fb a c g (α r) = fb (a α) c (g α) r`,
  here `a = 1 - ρ`, `c = ρ η`.
* `slot_supercritical_iff`: the slot latches iff `α (1 + ρ (η g - 1)) > 1`; at `α = 1`
  (`slot_supercritical_no_decay`) iff `η g > 1`. Neither involves the write gate `β`, so the gate
  can close on text and keep the latch (`latch_converges`, `latch_corrects` then apply to the
  slot map with the parameters of `slot_composite`).

What this does not say: that training will open the gate on the cue and close it on text, or
that a slot is used for the state that needs it. Those are the predictions of the v6 runs
(PREREG amendment 3).
-/

namespace CGB.Generation.LatchSlot

open CGB.Generation.HoldCorrect

/-- Closed gate in the coupled rule: no renewal is left, only the decay `α`. -/
theorem coupled_closed_gate_decays (α η g r : ℝ) :
    fb (α * (1 - 0)) (0 * η) g r = α * r := by
  simp [fb]

/-- **A small gate is subcritical** in the coupled rule: with `0 ≤ α`, `0 ≤ β` and
`β η g ≤ 1 - α`, the renewal satisfies `c g ≤ 1 - a` (κ ≤ 1). -/
theorem coupled_small_gate_subcritical {α β η g : ℝ} (hα0 : 0 ≤ α) (hβ0 : 0 ≤ β)
    (hsmall : β * (η * g) ≤ 1 - α) :
    (β * η) * g ≤ 1 - α * (1 - β) := by
  have : 0 ≤ α * β := mul_nonneg hα0 hβ0
  nlinarith

/-- With a small gate the content is forgotten every event (`subcritical_forgets`). -/
theorem coupled_small_gate_forgets {α β η g x : ℝ} (hα0 : 0 ≤ α) (hβ0 : 0 < β)
    (hβ1 : β ≤ 1) (hη : 0 < η) (hg : 0 < g) (hsmall : β * (η * g) ≤ 1 - α) (hx : 0 < x) :
    fb (α * (1 - β)) (β * η) g x < x := by
  have ha0 : 0 ≤ α * (1 - β) := mul_nonneg hα0 (by linarith)
  exact (subcritical_forgets ha0 (mul_pos hβ0 hη) hg
    (coupled_small_gate_subcritical hα0 hβ0.le hsmall) hx).2.2

/-- **The gate floor of a coupled latch**: while it latches (κ > 1), `1 - α < β (η g - α)`. -/
theorem coupled_latch_gate_floor {α β η g : ℝ}
    (hlatch : 1 - α * (1 - β) < (β * η) * g) :
    1 - α < β * (η * g - α) := by
  have : β * (η * g - α) = β * η * g - α * β := by ring
  nlinarith

/-- **The disturbance floor**: a latching coupled head admits from a value of size `V` at least
`(1-α)(1-η)V/(η g - α)` per event (`D = β (1 - η) V`). -/
theorem coupled_disturbance_floor {α β η g V : ℝ} (hη1 : η < 1) (hV : 0 < V)
    (hκ : α < η * g) (hlatch : 1 - α * (1 - β) < (β * η) * g) :
    (1 - α) * ((1 - η) * V) / (η * g - α) < β * ((1 - η) * V) := by
  have hb := coupled_latch_gate_floor hlatch
  have hd : 0 < (1 - η) * V := mul_pos (by linarith) hV
  rw [div_lt_iff₀ (sub_pos.mpr hκ)]
  have := mul_lt_mul_of_pos_right hb hd
  linarith [show β * ((1 - η) * V) * (η * g - α) = β * (η * g - α) * ((1 - η) * V) by ring]

/-- Decay by `α` followed by the slot renewal is the same family with rescaled parameters. -/
theorem slot_composite (a c g α r : ℝ) : fb a c g (α * r) = fb (a * α) c (g * α) r := by
  simp only [fb]
  rw [show g * (α * r) = g * α * r by ring]
  ring

/-- **When the slot latches**: with `a = 1 - ρ`, `c = ρ η` after decay `α`, the composite map is
supercritical (`1 - a α < c g α`) iff `α (1 + ρ (η g - 1)) > 1`. The write gate does not enter. -/
theorem slot_supercritical_iff (ρ η g α : ℝ) :
    1 - (1 - ρ) * α < ρ * η * (g * α) ↔ 1 < α * (1 + ρ * (η * g - 1)) := by
  have h : α * (1 + ρ * (η * g - 1)) = (1 - ρ) * α + ρ * η * (g * α) := by ring
  constructor <;> intro hh <;> linarith

/-- Without decay the slot latches iff `η g > 1`, for any renewal rate `ρ > 0`. -/
theorem slot_supercritical_no_decay {ρ η g : ℝ} (hρ : 0 < ρ) :
    1 - (1 - ρ) * 1 < ρ * η * (g * 1) ↔ 1 < η * g := by
  rw [slot_supercritical_iff]
  constructor
  · intro h; nlinarith
  · intro h; nlinarith

end CGB.Generation.LatchSlot
