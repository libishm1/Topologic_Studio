/-
The 16-neighbour lattice's bound (topologicpy-web-frontend/src/core/floorRaster.js
SIXTEEN_BOUND; wiki/roadmap/evacuation-map-and-accessibility-audit-plan-2026-11-11.md).

In the open, a lattice walk to a point between two adjacent move directions
u and v (unit vectors, u · v = c) goes a along u and b along v, and the
straight walk is |a u + b v|, whose square is a^2 + b^2 + 2 a b c. The lattice
walk a + b is at most the straight walk over cos(theta / 2), since

  (a + b)^2 (1 + c) <= 2 (a^2 + b^2 + 2 a b c),  the difference being (1 - c)(a - b)^2.

The 8-neighbour lattice's worst pair is 45 degrees apart (c = sqrt 2 / 2, the
8.24 % of Eikonal.lean); the 16-neighbour lattice adds the knight's moves,
and its worst pair is (1, 0) and (2, 1), atan (1/2) apart, c = 2 / sqrt 5:
the walk is less than 1.0558^(1/2), under 1.0276, times the straight walk.
-/
import Mathlib

namespace FireEgress.Sixteen

/-- Two directions' walk against the straight one, for any angle between them up to 180 degrees. -/
theorem two_directions (a b c : ℝ) (hc : c ≤ 1) :
    (a + b) ^ 2 * (1 + c) ≤ 2 * (a ^ 2 + b ^ 2 + 2 * a * b * c) := by
  nlinarith [mul_nonneg (sub_nonneg.mpr hc) (sq_nonneg (a - b))]

/-- The equality: the bound is tight where the walk splits evenly (a = b). -/
theorem two_directions_eq (a c : ℝ) :
    (a + a) ^ 2 * (1 + c) = 2 * (a ^ 2 + a ^ 2 + 2 * a * a * c) := by
  ring

theorem sqrt_five_lt : Real.sqrt 5 < 2.2361 := by
  rw [Real.sqrt_lt' (by norm_num)]
  norm_num

theorem sqrt_five_pos : 0 < Real.sqrt 5 := Real.sqrt_pos.mpr (by norm_num)

/-- The knight pair's cosine, 2 / sqrt 5, is more than 0.8944. -/
theorem knight_cos_gt : (0.8944 : ℝ) < 2 / Real.sqrt 5 := by
  have h := sqrt_five_lt
  have hp := sqrt_five_pos
  rw [lt_div_iff₀ hp]
  nlinarith

/-- The 16-neighbour bound: the squared lattice walk is less than 1.0558 times the squared straight walk. -/
theorem sixteen_bound (a b : ℝ) :
    (a + b) ^ 2 ≤ 1.0558 * (a ^ 2 + b ^ 2 + 2 * a * b * (2 / Real.sqrt 5)) := by
  set c := 2 / Real.sqrt 5 with hcdef
  have hc1 : c ≤ 1 := by
    rw [hcdef, div_le_one sqrt_five_pos]
    have : (2 : ℝ) ^ 2 ≤ 5 := by norm_num
    calc (2 : ℝ) = Real.sqrt (2 ^ 2) := by rw [Real.sqrt_sq (by norm_num)]
      _ ≤ Real.sqrt 5 := Real.sqrt_le_sqrt this
  have hc := knight_cos_gt
  have key := two_directions a b c hc1
  have hpos : 0 ≤ (a + b) ^ 2 := sq_nonneg _
  -- (a + b)^2 (1 + c) <= 2 S, and 1.0558 (1 + c) >= 2, so (a + b)^2 * 2 <= 1.0558 (1 + c) (a + b)^2 <= 1.0558 * 2 S.
  nlinarith [mul_nonneg hpos (sub_nonneg.mpr hc.le)]

/-- And 1.0558 is under 1.0276 squared: the walk is under 2.76 % long. -/
theorem sixteen_percent : (1.0558 : ℝ) < 1.0276 ^ 2 := by norm_num

end FireEgress.Sixteen
