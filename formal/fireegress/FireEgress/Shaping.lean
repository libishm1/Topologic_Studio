import Mathlib

/-!
# Potential-based reward shaping (Studio's RL)

Studio's Q-learning shapes each reward with `γ Φ(s') - Φ(s)`, where `Φ` is minus
the remaining distance to the exit. The literature review's finding W5 is that
Ng, Harada & Russell's (1999) invariance needs `Φ = 0` at terminal states, and
Studio's burn states had a non-zero potential. This file proves the identity
behind the theorem, the invariance under `Φ(terminal) = 0`, and that without it
the order of outcomes can flip. Trajectories are indexed by time: `r t` is the
reward and `Φ t` the potential of the state at step `t`.
-/

namespace FireEgress.Shaping

/-- Discounted return of the first `T` rewards. -/
def ret (γ : ℝ) (r : ℕ → ℝ) (T : ℕ) : ℝ := ∑ t ∈ Finset.range T, γ ^ t * r t

/-- Shaped discounted return. -/
def shaped (γ : ℝ) (r Φ : ℕ → ℝ) (T : ℕ) : ℝ :=
  ∑ t ∈ Finset.range T, γ ^ t * (r t + γ * Φ (t + 1) - Φ t)

/-- L12. Shaping telescopes along any trajectory:
`shaped = ret + γ^T Φ(s_T) - Φ(s_0)`. -/
theorem shaped_eq (γ : ℝ) (r Φ : ℕ → ℝ) (T : ℕ) :
    shaped γ r Φ T = ret γ r T + γ ^ T * Φ T - Φ 0 := by
  induction T with
  | zero => simp [shaped, ret]
  | succ n ih =>
    simp only [shaped, ret, Finset.sum_range_succ] at ih ⊢
    rw [ih, pow_succ]
    ring

/-- L13 (deterministic, finite horizon). With zero potential at both terminal
states and a common start, shaping shifts every return by the same constant
`-Φ(s_0)`, so it cannot change which of two trajectories is better. -/
theorem shaping_preserves_order (γ : ℝ) (r₁ r₂ Φ₁ Φ₂ : ℕ → ℝ) (T₁ T₂ : ℕ)
    (h0 : Φ₁ 0 = Φ₂ 0) (hT₁ : Φ₁ T₁ = 0) (hT₂ : Φ₂ T₂ = 0) :
    shaped γ r₁ Φ₁ T₁ ≤ shaped γ r₂ Φ₂ T₂ ↔ ret γ r₁ T₁ ≤ ret γ r₂ T₂ := by
  rw [shaped_eq, shaped_eq, hT₁, hT₂, h0]
  constructor <;> intro h <;> linarith

/-- L14. Without `Φ(terminal) = 0` the order can flip. From a start of
potential 0, outcome B (return -9) is truly better than A (return -10); but if
A's terminal state has potential 5 and B's 0, the shaped values rank A first. -/
theorem nonzero_terminal_flips_order :
    ret 1 (fun _ => -10) 1 < ret 1 (fun _ => -9) 1 ∧
      shaped 1 (fun _ => -9) (fun t => if t = 0 then 0 else 0) 1 <
        shaped 1 (fun _ => -10) (fun t => if t = 0 then 0 else 5) 1 := by
  norm_num [ret, shaped]

end FireEgress.Shaping
