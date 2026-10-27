import Mathlib

/-!
# The straight-line A* heuristic (Studio's router)

Studio's A* (`app/graphs/pathfinding.py`, `src/core/route.js`) uses
`h(u) = dist(u, goal)`. It closes a node once popped, which is only exact when
`h` is consistent. These results cover that, for any metric space and any edge
weights at least as long as the edges - which Studio's hazard weighting keeps.
-/

namespace FireEgress.Heuristic

/-- L8. The straight-line heuristic is consistent whenever an edge's weight is
at least its length: `h(u) ≤ w(u, v) + h(v)`. -/
theorem consistent {P : Type*} [MetricSpace P] (g u v : P) (w : ℝ)
    (hw : dist u v ≤ w) : dist u g ≤ w + dist v g := by
  have := dist_triangle u v g
  linarith

/-- Consistency plus `h(goal) = 0` gives admissibility along any path: summing
`h(xᵢ) ≤ wᵢ + h(xᵢ₊₁)` from the start to the goal, `h(start) ≤ path cost`. -/
theorem admissible {P : Type*} [MetricSpace P] (g : P) (x : ℕ → P) (w : ℕ → ℝ) (n : ℕ)
    (hw : ∀ i < n, dist (x i) (x (i + 1)) ≤ w i) (hend : x n = g) :
    dist (x 0) g ≤ ∑ i ∈ Finset.range n, w i := by
  have key : ∀ k ≤ n, dist (x 0) g ≤ (∑ i ∈ Finset.range k, w i) + dist (x k) g := by
    intro k hk
    induction k with
    | zero => simp
    | succ k ih =>
      have h1 := ih (Nat.le_of_succ_le hk)
      have h2 := consistent g (x k) (x (k + 1)) (w k) (hw k (Nat.lt_of_succ_le hk))
      rw [Finset.sum_range_succ]
      linarith
  have := key n le_rfl
  rw [hend, dist_self] at this
  linarith

/-- Hazard weighting `length * (1 + α h)` with `α, h ≥ 0` never drops below the
length, so the heuristic stays consistent while the fire reweights routes. -/
theorem hazard_weight_ge_length (len α h : ℝ) (hl : 0 ≤ len) (hα : 0 ≤ α) (hh : 0 ≤ h) :
    len ≤ len * (1 + α * h) := by
  have : 0 ≤ len * (α * h) := mul_nonneg hl (mul_nonneg hα hh)
  nlinarith

end FireEgress.Heuristic
