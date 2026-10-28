import Mathlib
import FireEgress.Segments
import FireEgress.AStar

/-!
# Floating-point error in Studio's geometry and routing

Lean and Mathlib have no IEEE-754 semantics, so this file works in the **standard rounding
model** (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., §2.2): every basic
operation returns `fl(x op y) = (x op y)(1 + δ)` with `|δ| ≤ u`, where `x op y` is the exact
result on the already-rounded operands. `Rnd u x y` says that `y` is such a rounding of `x`
(`rnd_iff`); a computation is modelled by one `Rnd` hypothesis per operation, in the order
the JavaScript evaluates it. Unit roundoffs: `uDouble = 2⁻⁵³` (JS numbers, `Float64Array`)
and `uSingle = 2⁻²⁴` (`Float32Array`, `Math.fround`), both round to nearest.

**What the model covers, and what it does not.** IEEE round-to-nearest satisfies `Rnd u x y`
for every operation whose exact result is `0` or has magnitude in the normal range (doubles
`[2⁻¹⁰²², 2¹⁰²⁴)`, floats `[2⁻¹²⁶, 2¹²⁸)`). Bit-level behaviour outside that range -
subnormal results (gradual underflow, whose error is absolute, not relative), overflow to
`±Infinity`, `NaN`, signed zeros - is outside the model, and no theorem here says anything
about it. For Studio's code:
* a difference of two doubles that underflows is exact (Hauser's theorem), so the
  subtractions in `segmentsCross` always satisfy the model; its products need coordinate
  differences above about `1e-154` m, or exactly zero (a zero product is exact);
* `segmentsCross` compares signs (`opposite`) instead of multiplying `d1 * d2`, so the
  product that could underflow or overflow is never formed: that guard is why `jsOpposite`
  below is a literal transcription and no product of determinants appears;
* JavaScript never contracts `a * b - c * d` into a fused multiply-add, so every operation
  is separately rounded, as modelled;
* `Math.hypot` is "implementation-approximated" in ECMAScript; its relative accuracy is a
  parameter `ε` here (V8 is within a couple of ulps; not verified).

## Results

1. **Orientation filter** (`lattice.js` `segmentsCross`). For
   `d = (a - b)(c - e) - (f - g)(h - k)` evaluated with all seven operations rounded,
   `|f̃ - d| ≤ ((1 + u)⁴ - 1)(|ab'| + |cd'|)` in exact terms (`flDet_err_exact`) and
   `|f̃ - d| ≤ errA u (|l| + |r|)` in the computed products `l`, `r`
   (`flDet_err`, `errA u ≤ (4 + 7u)u`, `errA_double`: `≤ 4.4408920985006271e-16` for doubles).
   So `|f̃| > errA u (|l| + |r|)` certifies the sign (`flDet_sign_certain`), a
   Shewchuk-style `|f̃| > (3 + 14u)u(|l| + |r|)` does too (`flDet_sign_certain_shewchuk`),
   and the bound can itself be evaluated in floating point with the double constant
   `(4 + 16u)u = 2⁻⁵¹ + 2⁻¹⁰²` (`flDet_sign_certain_code`). When all four determinants
   clear their bounds, the browser's `segmentsCross` equals the exact test
   (`segmentsCross_float_exact`) and hence the geometric characterisation
   `Segments.crossTest_iff` (`segmentsCross_float_geometric`).
2. **Summation** (route lengths in `route.js`, `straighten.js`, and A*'s `dist[end]`). The
   left-to-right floating sum of `n + 1` non-negative terms is within
   `((1 + u)ⁿ - 1) ≤ γₙ = nu/(1 - nu)` of the exact sum, relatively (`flSum_err`,
   `route_sum_rel_err`); a 10,000-link route in doubles: `γ₉₉₉₉ ≤ 1.1102e-12`
   (`gamma_10000_links`). With per-link length errors `ε` as well: `route_length_err`.
3. **Routing under perturbed weights.** Relative link errors `ε < 1` cost at most the factor
   `(1 + ε)/(1 - ε)` (`perturbed_weights_near_optimal`), absolute ones `α` at most
   `α (k₀ + k)` (`perturbed_weights_near_optimal_abs`). For **closed-set A* whose heuristic is
   consistent only up to `δ`** (reduced weights `≥ -δ`, as the `-4.44e-16` cases show): the
   label of every vertex A* pops is the cost of a real walk and at most `c + kδ` for every
   walk of cost `c` with `k` links (`astar_delta_near_optimal`), sharply at most `c` plus the
   sum of the walk's own inconsistencies (`astar_le_cost_add_inconsistency`). So the error is
   bounded by `δ` times the links of an *optimal* route. The naive bounds "at most `δ`" and
   "at most `δ` times the links of the returned route" are false
   (`astar_delta_counterexample`: error `1.5 δ` on a one-link answer). Where `δ` comes from in
   floating point: `float_reduced_ge`.
4. **Float32 coordinates.** Rounding points at distance at most `M` from the origin moves a
   link length by at most `2uM` (`length_err_rounded_points`, coordinatewise rounding:
   `round_coords_norm`); Studio (`M = 100` m, `u = 2⁻²⁴`): at most `1.2e-5` m
   (`studio_float32_length_err`); with `Math.hypot`'s own error: `computed_length_err`.
5. **Which links can be misclassified.** If the float `segmentsCross` disagrees with the
   exact test, some endpoint of one segment is within `B/|other segment|` of the other
   segment's line (`|orient| ≤ B`), and within `κ ≈ 4u` times its distance to that
   segment's first vertex (`misclassified_near`, `kappa_double`: `κ ≤ 4.441e-16`): links
   through (or ending within rounding of) an outline vertex or wall line, or collinear with
   an outline edge, and nothing else.
-/

namespace FireEgress.FloatError

open Finset

/-! ## The rounding model -/

/-- `Rnd u x y`: `y` is the exact result `x` of one operation after rounding with unit
roundoff `u`, i.e. `|y - x| ≤ u |x|`. Every arithmetic step of the modelled Studio code
contributes one such hypothesis. -/
def Rnd (u x y : ℝ) : Prop := |y - x| ≤ u * |x|

/-- Unit roundoff of IEEE binary64, round to nearest: JS numbers and `Float64Array`. -/
noncomputable def uDouble : ℝ := 1 / 2 ^ 53

/-- Unit roundoff of IEEE binary32, round to nearest: `Float32Array` and `Math.fround`. -/
noncomputable def uSingle : ℝ := 1 / 2 ^ 24

/-- The model in Higham's form: `Rnd u x y` iff `y = x (1 + δ)` for some `|δ| ≤ u`. So the
hypotheses used below are exactly the standard model `fl(x op y) = (x op y)(1 + δ)`. -/
theorem rnd_iff {u x y : ℝ} (hu : 0 ≤ u) : Rnd u x y ↔ ∃ δ, |δ| ≤ u ∧ y = x * (1 + δ) := by
  constructor
  · intro h
    rcases eq_or_ne x 0 with rfl | hx
    · have hy : y = 0 := by
        have : |y - 0| ≤ u * |(0 : ℝ)| := h
        rw [abs_zero, mul_zero, sub_zero] at this
        exact abs_nonpos_iff.1 this
      exact ⟨0, by rw [abs_zero]; exact hu, by rw [hy]; ring⟩
    · refine ⟨(y - x) / x, ?_, ?_⟩
      · rw [abs_div, div_le_iff₀ (abs_pos.2 hx)]
        exact h
      · have e : x * ((y - x) / x) = y - x := by field_simp
        rw [mul_add, mul_one, e]
        ring
  · rintro ⟨δ, hδ, rfl⟩
    unfold Rnd
    rw [show x * (1 + δ) - x = x * δ by ring, abs_mul, mul_comm u]
    exact mul_le_mul_of_nonneg_left hδ (abs_nonneg x)

/-- `|a - b| ≤ |a| + |b|`. -/
theorem fe_abs_sub_le (a b : ℝ) : |a - b| ≤ |a| + |b| := abs_sub a b

/-- Relative errors compose: `y` within `α|x|` of `x` and `z` within `β|y|` of `y` put `z`
within `(α + β + αβ)|x|` of `x`. This is how rounding errors accumulate along a chain of
operations. -/
theorem rel_trans {x y z α β : ℝ} (hβ : 0 ≤ β)
    (h₁ : |y - x| ≤ α * |x|) (h₂ : |z - y| ≤ β * |y|) : |z - x| ≤ (α + β + α * β) * |x| := by
  have hy : |y| ≤ (1 + α) * |x| := by
    have := abs_sub_abs_le_abs_sub y x
    linarith
  calc |z - x| = |(z - y) + (y - x)| := by rw [sub_add_sub_cancel]
    _ ≤ |z - y| + |y - x| := abs_add_le _ _
    _ ≤ β * ((1 + α) * |x|) + α * |x| :=
        add_le_add (h₂.trans (mul_le_mul_of_nonneg_left hy hβ)) h₁
    _ = (α + β + α * β) * |x| := by ring

/-- Relative errors of factors add up in a product: `|x'y' - xy| ≤ (α + β + αβ)|xy|`. -/
theorem rel_mul {x x' y y' α β : ℝ} (hα : 0 ≤ α)
    (hx : |x' - x| ≤ α * |x|) (hy : |y' - y| ≤ β * |y|) :
    |x' * y' - x * y| ≤ (α + β + α * β) * |x * y| := by
  have hy' : |y'| ≤ (1 + β) * |y| := by
    have := abs_sub_abs_le_abs_sub y' y
    linarith
  calc |x' * y' - x * y| = |(x' - x) * y' + x * (y' - y)| := by congr 1; ring
    _ ≤ |(x' - x) * y'| + |x * (y' - y)| := abs_add_le _ _
    _ = |x' - x| * |y'| + |x| * |y' - y| := by rw [abs_mul, abs_mul]
    _ ≤ (α * |x|) * ((1 + β) * |y|) + |x| * (β * |y|) :=
        add_le_add (mul_le_mul hx hy' (abs_nonneg _) (mul_nonneg hα (abs_nonneg _)))
          (mul_le_mul_of_nonneg_left hy (abs_nonneg _))
    _ = (α + β + α * β) * |x * y| := by rw [abs_mul]; ring

/-- The error of a rounding, measured against the *computed* value: `|x - y| ≤ u/(1-u) |y|`.
This is what lets code bound errors from the numbers it actually holds. -/
theorem rnd_back {u x y : ℝ} (hu : 0 ≤ u) (hu1 : u < 1) (h : Rnd u x y) :
    |x - y| ≤ u / (1 - u) * |y| := by
  unfold Rnd at h
  have h1 : 0 < 1 - u := by linarith
  have hx : (1 - u) * |x| ≤ |y| := by
    have := abs_sub_abs_le_abs_sub x y
    rw [abs_sub_comm] at this
    nlinarith
  rw [abs_sub_comm, div_mul_eq_mul_div, le_div_iff₀ h1]
  nlinarith [mul_le_mul_of_nonneg_right h h1.le, mul_le_mul_of_nonneg_left hx hu]

/-- A rounded product of two rounded differences, `l = fl(fl(a) · fl(b))`, is within
`((1 + u)³ - 1)|ab|` of the exact product `ab`. -/
theorem rnd_prod {u a b p q l : ℝ} (hu : 0 ≤ u) (hp : Rnd u a p) (hq : Rnd u b q)
    (hl : Rnd u (p * q) l) : |l - a * b| ≤ ((1 + u) ^ 3 - 1) * |a * b| := by
  have h1 := rel_mul hu hp hq
  have h2 := rel_trans hu h1 hl
  calc |l - a * b| ≤ (u + u + u * u + u + (u + u + u * u) * u) * |a * b| := h2
    _ = ((1 + u) ^ 3 - 1) * |a * b| := by ring

/-- The same product error measured against the computed product `l`:
`|ab - l| ≤ ((1 + u/(1-u))³ - 1)|l|`. -/
theorem rnd_prod_back {u a b p q l : ℝ} (hu : 0 ≤ u) (hu1 : u < 1) (hp : Rnd u a p)
    (hq : Rnd u b q) (hl : Rnd u (p * q) l) :
    |a * b - l| ≤ ((1 + u / (1 - u)) ^ 3 - 1) * |l| := by
  have hv0 : 0 ≤ u / (1 - u) := div_nonneg hu (by linarith)
  have h1 : |a * b - p * q| ≤ (u / (1 - u) + u / (1 - u) + u / (1 - u) * (u / (1 - u))) *
      |p * q| := rel_mul hv0 (rnd_back hu hu1 hp) (rnd_back hu hu1 hq)
  have h2 := rel_trans (add_nonneg (add_nonneg hv0 hv0) (mul_nonneg hv0 hv0))
    (rnd_back hu hu1 hl) h1
  calc |a * b - l| ≤ (u / (1 - u) + (u / (1 - u) + u / (1 - u) + u / (1 - u) * (u / (1 - u))) +
        u / (1 - u) * (u / (1 - u) + u / (1 - u) + u / (1 - u) * (u / (1 - u)))) * |l| := h2
    _ = ((1 + u / (1 - u)) ^ 3 - 1) * |l| := by ring

/-! ## (1) The orientation filter -/

/-- `(x₁ - x₂)(y₁ - y₂) - (z₁ - z₂)(w₁ - w₂)` in exact arithmetic: the shape of each of the
four determinants in `segmentsCross`. -/
def crossDet (x₁ x₂ y₁ y₂ z₁ z₂ w₁ w₂ : ℝ) : ℝ := (x₁ - x₂) * (y₁ - y₂) - (z₁ - z₂) * (w₁ - w₂)

/-- `FlDet u … l r f`: `f` is a value the browser can compute for `crossDet …` when it
evaluates `(x₁ - x₂) * (y₁ - y₂) - (z₁ - z₂) * (w₁ - w₂)` with each of its seven operations
rounded (four differences `p, q, p', q'`, two products `l, r`, one difference `f`). The two
computed products `l` and `r` are what the code can use to bound the error. -/
def FlDet (u x₁ x₂ y₁ y₂ z₁ z₂ w₁ w₂ l r f : ℝ) : Prop :=
  ∃ p q p' q' : ℝ, Rnd u (x₁ - x₂) p ∧ Rnd u (y₁ - y₂) q ∧ Rnd u (z₁ - z₂) p' ∧
    Rnd u (w₁ - w₂) q' ∧ Rnd u (p * q) l ∧ Rnd u (p' * q') r ∧ Rnd u (l - r) f

/-- The constant of the computed-value bound: `errA u = u + ((1 + u/(1-u))³ - 1) ≈ 4u`. -/
noncomputable def errA (u : ℝ) : ℝ := u + ((1 + u / (1 - u)) ^ 3 - 1)

/-- Core of the exact-terms bound: if the two products are within relative `θ` of exact
products `X`, `Y` and `f` rounds `l - r`, then `|f - (X - Y)| ≤ (θ + u(1 + θ))(|X| + |Y|)`. -/
theorem det_err_core {u X Y l r f θ : ℝ} (hu : 0 ≤ u)
    (hL : |l - X| ≤ θ * |X|) (hR : |r - Y| ≤ θ * |Y|) (hf : Rnd u (l - r) f) :
    |f - (X - Y)| ≤ (θ + u * (1 + θ)) * (|X| + |Y|) := by
  unfold Rnd at hf
  have hl' : |l| ≤ (1 + θ) * |X| := by
    have := abs_sub_abs_le_abs_sub l X
    linarith
  have hr' : |r| ≤ (1 + θ) * |Y| := by
    have := abs_sub_abs_le_abs_sub r Y
    linarith
  have e1 : |f - (X - Y)| ≤ |f - (l - r)| + |l - X| + |r - Y| := by
    have : f - (X - Y) = (f - (l - r)) + (l - X) - (r - Y) := by ring
    rw [this]
    exact (fe_abs_sub_le _ _).trans (add_le_add (abs_add_le _ _) le_rfl)
  nlinarith [mul_le_mul_of_nonneg_left (fe_abs_sub_le l r) hu,
    mul_le_mul_of_nonneg_left (add_le_add hl' hr') hu]

/-- Core of the computed-value bound: errors `η` relative to the computed products give
`|f - (X - Y)| ≤ (u + η)(|l| + |r|)`. -/
theorem det_err_core' {u X Y l r f η : ℝ} (hu : 0 ≤ u)
    (hL : |X - l| ≤ η * |l|) (hR : |Y - r| ≤ η * |r|) (hf : Rnd u (l - r) f) :
    |f - (X - Y)| ≤ (u + η) * (|l| + |r|) := by
  unfold Rnd at hf
  have e1 : |f - (X - Y)| ≤ |f - (l - r)| + |X - l| + |Y - r| := by
    have : f - (X - Y) = (f - (l - r)) - (X - l) + (Y - r) := by ring
    rw [this]
    exact (abs_add_le _ _).trans (add_le_add (fe_abs_sub_le _ _) le_rfl)
  nlinarith [mul_le_mul_of_nonneg_left (fe_abs_sub_le l r) hu]

/-- Core of the Shewchuk-style bound: the final subtraction's error is charged to `|f|`
itself, `|f - (X - Y)| ≤ u/(1-u)|f| + η(|l| + |r|)`. -/
theorem det_err_core_back {u X Y l r f η : ℝ} (hu : 0 ≤ u) (hu1 : u < 1)
    (hL : |X - l| ≤ η * |l|) (hR : |Y - r| ≤ η * |r|) (hf : Rnd u (l - r) f) :
    |f - (X - Y)| ≤ u / (1 - u) * |f| + η * (|l| + |r|) := by
  have hf' : |(l - r) - f| ≤ u / (1 - u) * |f| := rnd_back hu hu1 hf
  have e1 : |f - (X - Y)| ≤ |(l - r) - f| + |X - l| + |Y - r| := by
    have : f - (X - Y) = -((l - r) - f) - (X - l) + (Y - r) := by ring
    rw [this]
    calc |-((l - r) - f) - (X - l) + (Y - r)| ≤ |-((l - r) - f) - (X - l)| + |Y - r| :=
          abs_add_le _ _
      _ ≤ (|-((l - r) - f)| + |X - l|) + |Y - r| := add_le_add (fe_abs_sub_le _ _) le_rfl
      _ = |(l - r) - f| + |X - l| + |Y - r| := by rw [abs_neg]
  nlinarith

/-- **(1) Error bound in exact terms.** Every possible floating evaluation `f` of one of
`segmentsCross`'s determinants satisfies
`|f - d| ≤ ((1 + u)⁴ - 1)(|(x₁ - x₂)(y₁ - y₂)| + |(z₁ - z₂)(w₁ - w₂)|)`, about
`4u (|X| + |Y|)`. -/
theorem flDet_err_exact {u x₁ x₂ y₁ y₂ z₁ z₂ w₁ w₂ l r f : ℝ} (hu : 0 ≤ u)
    (h : FlDet u x₁ x₂ y₁ y₂ z₁ z₂ w₁ w₂ l r f) :
    |f - crossDet x₁ x₂ y₁ y₂ z₁ z₂ w₁ w₂| ≤
      ((1 + u) ^ 4 - 1) * (|(x₁ - x₂) * (y₁ - y₂)| + |(z₁ - z₂) * (w₁ - w₂)|) := by
  obtain ⟨p, q, p', q', hp, hq, hp', hq', hl, hr, hf⟩ := h
  have key := det_err_core hu (rnd_prod hu hp hq hl) (rnd_prod hu hp' hq' hr) hf
  have e : (1 + u) ^ 3 - 1 + u * (1 + ((1 + u) ^ 3 - 1)) = (1 + u) ^ 4 - 1 := by ring
  rw [e] at key
  exact key

/-- **(1) Error bound in computed terms (what code can evaluate).** With `l`, `r` the two
products the browser computed, `|f - d| ≤ errA u (|l| + |r|)`, `errA u ≈ 4u`. -/
theorem flDet_err {u x₁ x₂ y₁ y₂ z₁ z₂ w₁ w₂ l r f : ℝ} (hu : 0 ≤ u) (hu1 : u < 1)
    (h : FlDet u x₁ x₂ y₁ y₂ z₁ z₂ w₁ w₂ l r f) :
    |f - crossDet x₁ x₂ y₁ y₂ z₁ z₂ w₁ w₂| ≤ errA u * (|l| + |r|) := by
  obtain ⟨p, q, p', q', hp, hq, hp', hq', hl, hr, hf⟩ := h
  exact det_err_core' hu (rnd_prod_back hu hu1 hp hq hl) (rnd_prod_back hu hu1 hp' hq' hr) hf

/-- `errA u ≤ (4 + 7u)u` for `u ≤ 1/16`: the bound is `4u` up to a negligible term. -/
theorem errA_le {u : ℝ} (hu1 : u ≤ 1 / 16) : errA u ≤ (4 + 7 * u) * u := by
  have h1 : 0 < 1 - u := by linarith
  have h1' : 1 - u ≠ 0 := h1.ne'
  have e : 1 + u / (1 - u) = 1 / (1 - u) := by field_simp; ring
  unfold errA
  rw [e, div_pow, one_pow]
  have key : 1 ≤ (1 + 3 * u + 7 * u ^ 2) * (1 - u) ^ 3 := by
    have h16 : 0 ≤ 1 / 16 - u := by linarith
    have hpoly : 0 ≤ 1 - 13 * u + 18 * u ^ 2 - 7 * u ^ 3 := by
      nlinarith [sq_nonneg u, mul_nonneg (sq_nonneg u) h16, mul_nonneg h16 h16]
    nlinarith [mul_nonneg (sq_nonneg u) hpoly]
  have : 1 / (1 - u) ^ 3 ≤ 1 + 3 * u + 7 * u ^ 2 := by
    rw [div_le_iff₀ (pow_pos h1 3)]
    linarith
  nlinarith

/-- `errA` is non-negative. -/
theorem errA_nonneg {u : ℝ} (hu : 0 ≤ u) (hu1 : u < 1) : 0 ≤ errA u := by
  have hv : 0 ≤ u / (1 - u) := div_nonneg hu (by linarith)
  have : 1 ≤ (1 + u / (1 - u)) ^ 3 := one_le_pow₀ (by linarith)
  unfold errA
  linarith

/-- Two reals have the same sign (both positive, both negative, or both zero). -/
def SameSign (f d : ℝ) : Prop := (0 < f ↔ 0 < d) ∧ (f < 0 ↔ d < 0)

/-- If the error is smaller than the computed value, the computed sign is the exact sign. -/
theorem sameSign_of_lt {f d : ℝ} (h : |f - d| < |f|) : SameSign f d := by
  rcases lt_trichotomy f 0 with hf | rfl | hf
  · rw [abs_of_neg hf] at h
    obtain ⟨h1, h2⟩ := abs_lt.1 h
    exact ⟨⟨fun h' => by linarith, fun h' => by linarith⟩, ⟨fun _ => by linarith, fun _ => hf⟩⟩
  · rw [abs_zero] at h
    exact absurd h (not_lt.2 (abs_nonneg _))
  · rw [abs_of_pos hf] at h
    obtain ⟨h1, h2⟩ := abs_lt.1 h
    exact ⟨⟨fun _ => by linarith, fun _ => hf⟩, ⟨fun h' => by linarith, fun h' => by linarith⟩⟩

/-- A sign error needs an error at least as large as the exact value: if the signs differ,
`|d| ≤ |f - d|`. -/
theorem abs_le_of_not_sameSign {f d : ℝ} (h : ¬ SameSign f d) : |d| ≤ |f - d| := by
  by_contra hlt
  apply h
  have := sameSign_of_lt (f := d) (d := f) (by rw [abs_sub_comm]; exact not_le.1 hlt)
  exact ⟨this.1.symm, this.2.symm⟩

/-- **(1) Sign certificate.** If a computed determinant clears its bound,
`|f| > errA u (|l| + |r|)`, its sign is the exact sign. -/
theorem flDet_sign_certain {u x₁ x₂ y₁ y₂ z₁ z₂ w₁ w₂ l r f : ℝ} (hu : 0 ≤ u) (hu1 : u < 1)
    (h : FlDet u x₁ x₂ y₁ y₂ z₁ z₂ w₁ w₂ l r f) (hB : errA u * (|l| + |r|) < |f|) :
    SameSign f (crossDet x₁ x₂ y₁ y₂ z₁ z₂ w₁ w₂) :=
  sameSign_of_lt ((flDet_err hu hu1 h).trans_lt hB)

/-- A zero bound means an exact result: if both computed products are `0`, the computed
determinant equals the exact one (both are `0`). -/
theorem flDet_exact_of_zero {u x₁ x₂ y₁ y₂ z₁ z₂ w₁ w₂ l r f : ℝ} (hu : 0 ≤ u) (hu1 : u < 1)
    (h : FlDet u x₁ x₂ y₁ y₂ z₁ z₂ w₁ w₂ l r f) (h0 : |l| + |r| = 0) :
    f = crossDet x₁ x₂ y₁ y₂ z₁ z₂ w₁ w₂ := by
  have := flDet_err hu hu1 h
  rw [h0, mul_zero] at this
  exact sub_eq_zero.1 (abs_nonpos_iff.1 this)

/-- **(1) Shewchuk-style certificate** (sharper constant, about `3u`): charging the final
subtraction's error to `|f|`, `|f| > (3 + 14u)u(|l| + |r|)` already certifies the sign. -/
theorem flDet_sign_certain_shewchuk {u x₁ x₂ y₁ y₂ z₁ z₂ w₁ w₂ l r f : ℝ} (hu : 0 ≤ u)
    (hu1 : u ≤ 1 / 32) (h : FlDet u x₁ x₂ y₁ y₂ z₁ z₂ w₁ w₂ l r f)
    (hB : (3 + 14 * u) * u * (|l| + |r|) < |f|) :
    SameSign f (crossDet x₁ x₂ y₁ y₂ z₁ z₂ w₁ w₂) := by
  apply sameSign_of_lt
  have hu1' : u < 1 := by linarith
  obtain ⟨p, q, p', q', hp, hq, hp', hq', hl, hr, hf⟩ := h
  have key := det_err_core_back hu hu1' (rnd_prod_back hu hu1' hp hq hl)
    (rnd_prod_back hu hu1' hp' hq' hr) hf
  have hv : u / (1 - u) ≤ 2 * u := by
    rw [div_le_iff₀ (by linarith)]
    nlinarith
  have hη : (1 + u / (1 - u)) ^ 3 - 1 ≤ 3 * u + 7 * u ^ 2 := by
    have := errA_le (u := u) (by linarith)
    unfold errA at this
    nlinarith
  have hT : 0 ≤ |l| + |r| := by positivity
  have hf0 : 0 ≤ |f| := abs_nonneg f
  have c1 : (3 * u + 7 * u ^ 2) * (|l| + |r|) ≤
      (1 - 2 * u) * ((3 + 14 * u) * u * (|l| + |r|)) := by
    have h28 : 0 ≤ u * (1 - 28 * u) := mul_nonneg hu (by linarith)
    nlinarith [mul_nonneg (mul_nonneg hu h28) hT]
  have c2 : (1 - 2 * u) * ((3 + 14 * u) * u * (|l| + |r|)) < (1 - 2 * u) * |f| :=
    mul_lt_mul_of_pos_left hB (by linarith)
  calc |f - crossDet x₁ x₂ y₁ y₂ z₁ z₂ w₁ w₂|
      ≤ u / (1 - u) * |f| + ((1 + u / (1 - u)) ^ 3 - 1) * (|l| + |r|) := key
    _ ≤ 2 * u * |f| + (3 * u + 7 * u ^ 2) * (|l| + |r|) :=
        add_le_add (mul_le_mul_of_nonneg_right hv hf0) (mul_le_mul_of_nonneg_right hη hT)
    _ < |f| := by linarith

/-- The bound can be evaluated in floating point. If the code computes `t = fl(S)` (with
`S = |l| + |r|`, exact sums of non-negative doubles being rounded once) and `b = fl(c' t)`
with a constant padded so that `c ≤ c' (1 - u)²`, then `b ≥ c S`: comparing `|f|` with the
computed `b` is at least as strict as comparing with the real bound. -/
theorem bound_eval_safe {u c c' S t b : ℝ} (hu1 : u ≤ 1) (hS : 0 ≤ S)
    (hc' : 0 ≤ c') (hcc : c ≤ c' * (1 - u) ^ 2) (ht : Rnd u S t) (hb : Rnd u (c' * t) b) :
    c * S ≤ b := by
  unfold Rnd at ht hb
  rw [abs_of_nonneg hS] at ht
  have ht0 : (1 - u) * S ≤ t := by
    have := (abs_le.1 ht).1
    linarith
  have ht1 : 0 ≤ t := le_trans (mul_nonneg (by linarith) hS) ht0
  rw [abs_of_nonneg (mul_nonneg hc' ht1)] at hb
  have hb0 : (1 - u) * (c' * t) ≤ b := by
    have := (abs_le.1 hb).1
    linarith
  have h2 : c' * (1 - u) ^ 2 * S ≤ (1 - u) * (c' * t) := by
    have := mul_le_mul_of_nonneg_left ht0 (mul_nonneg hc' (show (0 : ℝ) ≤ 1 - u by linarith))
    nlinarith
  nlinarith [mul_le_mul_of_nonneg_right hcc hS]

/-- The double constant `(4 + 16u)u` is padded enough: `errA u ≤ (4 + 16u)u (1 - u)²`
for `0 ≤ u ≤ 1/32`. For `u = 2⁻⁵³` it is the double `2⁻⁵¹ + 2⁻¹⁰²` (`code_constant_double`),
written in JS as `(4 + 16 * 2 ** -53) * 2 ** -53`. -/
theorem code_constant_safe {u : ℝ} (hu : 0 ≤ u) (hu1 : u ≤ 1 / 32) :
    errA u ≤ (4 + 16 * u) * u * (1 - u) ^ 2 := by
  have h := errA_le (u := u) (by linarith)
  have hq : 0 ≤ 1 - 28 * u + 16 * u ^ 2 := by nlinarith [sq_nonneg u]
  nlinarith [mul_nonneg (mul_nonneg hu hu) hq]

/-- **(1) Sign certificate the code can evaluate.** With `t = fl(|l| + |r|)` and
`b = fl(((4 + 16u)u) · t)` computed in floating point, `|f| > b` certifies the sign of the
determinant. -/
theorem flDet_sign_certain_code {u x₁ x₂ y₁ y₂ z₁ z₂ w₁ w₂ l r f t b : ℝ} (hu : 0 ≤ u)
    (hu1 : u ≤ 1 / 32) (h : FlDet u x₁ x₂ y₁ y₂ z₁ z₂ w₁ w₂ l r f)
    (ht : Rnd u (|l| + |r|) t) (hb : Rnd u ((4 + 16 * u) * u * t) b) (hgt : b < |f|) :
    SameSign f (crossDet x₁ x₂ y₁ y₂ z₁ z₂ w₁ w₂) := by
  have hc := bound_eval_safe (by linarith) (by positivity)
    (mul_nonneg (by linarith) hu) (code_constant_safe hu hu1) ht hb
  exact flDet_sign_certain hu (by linarith) h (lt_of_le_of_lt hc hgt)

theorem uDouble_pos : 0 < uDouble := by unfold uDouble; positivity

theorem uDouble_le : uDouble ≤ 1 / 32 := by unfold uDouble; norm_num

/-- **(1) Numeric form for doubles.** `B = errA(2⁻⁵³)(|l| + |r|)` with
`errA(2⁻⁵³) ≤ 4.4408920985006271e-16` (that is `2⁻⁵¹ + 7·2⁻¹⁰⁶`, just above `4u = 2⁻⁵¹`). -/
theorem errA_double : errA uDouble ≤ 4.4408920985006271e-16 := by
  refine (errA_le (by linarith [uDouble_le])).trans ?_
  unfold uDouble
  norm_num

/-- The code constant in doubles, exactly: `(4 + 16u)u = 2⁻⁵¹ + 2⁻¹⁰²`
(`≈ 4.440892098500628e-16`), a representable double, so JS computes it without rounding. -/
theorem code_constant_double : (4 + 16 * uDouble) * uDouble = 1 / 2 ^ 51 + 1 / 2 ^ 102 := by
  unfold uDouble
  norm_num

/-! ### `segmentsCross` in floating point -/

/-- The floating evaluation of `Segments.orient O P Q` that `segmentsCross` performs: for
`d1 = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx)` it is `FlOrient u C D A`. -/
def FlOrient (u : ℝ) (O P Q : ℝ × ℝ) (l r f : ℝ) : Prop :=
  FlDet u P.1 O.1 Q.2 O.2 P.2 O.2 Q.1 O.1 l r f

/-- The determinant shape is the orientation of `Segments.lean`. -/
theorem crossDet_orient (O P Q : ℝ × ℝ) :
    crossDet P.1 O.1 Q.2 O.2 P.2 O.2 Q.1 O.1 = Segments.orient O P Q := rfl

/-- Literal transcription of the JS `opposite` in `segmentsCross`:
`(u > 0 && v < 0) || (u < 0 && v > 0)`. -/
def jsOpposite (x y : ℝ) : Prop := (0 < x ∧ y < 0) ∨ (x < 0 ∧ 0 < y)

/-- The browser's `segmentsCross` on its four computed determinants. -/
def floatCross (f₁ f₂ f₃ f₄ : ℝ) : Prop := jsOpposite f₁ f₂ ∧ jsOpposite f₃ f₄

/-- Comparing signs is the product test of `Segments.segmentsCross` (on reals). -/
theorem jsOpposite_iff (x y : ℝ) : jsOpposite x y ↔ x * y < 0 := mul_neg_iff.symm

/-- `jsOpposite` only looks at signs. -/
theorem jsOpposite_congr {x y x' y' : ℝ} (hx : SameSign x x') (hy : SameSign y y') :
    jsOpposite x y ↔ jsOpposite x' y' := by
  unfold jsOpposite
  rw [hx.1, hx.2, hy.1, hy.2]

/-- If the four computed determinants have the exact signs, the browser's `segmentsCross`
gives the exact answer. -/
theorem floatCross_iff_of_sameSign {A B C D : ℝ × ℝ} {f₁ f₂ f₃ f₄ : ℝ}
    (h₁ : SameSign f₁ (Segments.orient C D A)) (h₂ : SameSign f₂ (Segments.orient C D B))
    (h₃ : SameSign f₃ (Segments.orient A B C)) (h₄ : SameSign f₄ (Segments.orient A B D)) :
    floatCross f₁ f₂ f₃ f₄ ↔ Segments.crossTest A B C D := by
  rw [Segments.crossTest_iff_orient, ← jsOpposite_iff, ← jsOpposite_iff]
  exact and_congr (jsOpposite_congr h₁ h₂) (jsOpposite_congr h₃ h₄)

/-- **(1) `segmentsCross` is exact when its determinants clear the filter.** If each of the
four computed determinants satisfies `|fᵢ| > errA u (|lᵢ| + |rᵢ|)`, the browser's answer is
the exact-arithmetic test of `Segments.lean`. -/
theorem segmentsCross_float_exact {u : ℝ} (hu : 0 ≤ u) (hu1 : u < 1) {A B C D : ℝ × ℝ}
    {l₁ r₁ f₁ l₂ r₂ f₂ l₃ r₃ f₃ l₄ r₄ f₄ : ℝ}
    (h₁ : FlOrient u C D A l₁ r₁ f₁) (h₂ : FlOrient u C D B l₂ r₂ f₂)
    (h₃ : FlOrient u A B C l₃ r₃ f₃) (h₄ : FlOrient u A B D l₄ r₄ f₄)
    (c₁ : errA u * (|l₁| + |r₁|) < |f₁|) (c₂ : errA u * (|l₂| + |r₂|) < |f₂|)
    (c₃ : errA u * (|l₃| + |r₃|) < |f₃|) (c₄ : errA u * (|l₄| + |r₄|) < |f₄|) :
    floatCross f₁ f₂ f₃ f₄ ↔ Segments.crossTest A B C D :=
  floatCross_iff_of_sameSign (flDet_sign_certain hu hu1 h₁ c₁) (flDet_sign_certain hu hu1 h₂ c₂)
    (flDet_sign_certain hu hu1 h₃ c₃) (flDet_sign_certain hu hu1 h₄ c₄)

/-- **(1) …and then it is geometrically exact.** Under the same filter, the browser's
`segmentsCross` is true iff the closed segments meet and no endpoint of either lies on the
other (`Segments.crossTest_iff`). -/
theorem segmentsCross_float_geometric {u : ℝ} (hu : 0 ≤ u) (hu1 : u < 1) {A B C D : ℝ × ℝ}
    {l₁ r₁ f₁ l₂ r₂ f₂ l₃ r₃ f₃ l₄ r₄ f₄ : ℝ}
    (h₁ : FlOrient u C D A l₁ r₁ f₁) (h₂ : FlOrient u C D B l₂ r₂ f₂)
    (h₃ : FlOrient u A B C l₃ r₃ f₃) (h₄ : FlOrient u A B D l₄ r₄ f₄)
    (c₁ : errA u * (|l₁| + |r₁|) < |f₁|) (c₂ : errA u * (|l₂| + |r₂|) < |f₂|)
    (c₃ : errA u * (|l₃| + |r₃|) < |f₃|) (c₄ : errA u * (|l₄| + |r₄|) < |f₄|) :
    floatCross f₁ f₂ f₃ f₄ ↔ (segment ℝ A B ∩ segment ℝ C D).Nonempty ∧
      A ∉ segment ℝ C D ∧ B ∉ segment ℝ C D ∧ C ∉ segment ℝ A B ∧ D ∉ segment ℝ A B :=
  (segmentsCross_float_exact hu hu1 h₁ h₂ h₃ h₄ c₁ c₂ c₃ c₄).trans Segments.crossTest_iff

/-- **(1) The same, with bounds the code evaluates itself.** For each determinant the code
computes `tᵢ = fl(|lᵢ| + |rᵢ|)` and `bᵢ = fl(((4 + 16u)u) tᵢ)`; if every `|fᵢ| > bᵢ`, the
browser's `segmentsCross` is exactly right. This is the filter to put in `lattice.js`. -/
theorem segmentsCross_float_exact_code {u : ℝ} (hu : 0 ≤ u) (hu1 : u ≤ 1 / 32)
    {A B C D : ℝ × ℝ} {l₁ r₁ f₁ l₂ r₂ f₂ l₃ r₃ f₃ l₄ r₄ f₄ t₁ t₂ t₃ t₄ b₁ b₂ b₃ b₄ : ℝ}
    (h₁ : FlOrient u C D A l₁ r₁ f₁) (h₂ : FlOrient u C D B l₂ r₂ f₂)
    (h₃ : FlOrient u A B C l₃ r₃ f₃) (h₄ : FlOrient u A B D l₄ r₄ f₄)
    (t₁h : Rnd u (|l₁| + |r₁|) t₁) (t₂h : Rnd u (|l₂| + |r₂|) t₂)
    (t₃h : Rnd u (|l₃| + |r₃|) t₃) (t₄h : Rnd u (|l₄| + |r₄|) t₄)
    (b₁h : Rnd u ((4 + 16 * u) * u * t₁) b₁) (b₂h : Rnd u ((4 + 16 * u) * u * t₂) b₂)
    (b₃h : Rnd u ((4 + 16 * u) * u * t₃) b₃) (b₄h : Rnd u ((4 + 16 * u) * u * t₄) b₄)
    (g₁ : b₁ < |f₁|) (g₂ : b₂ < |f₂|) (g₃ : b₃ < |f₃|) (g₄ : b₄ < |f₄|) :
    floatCross f₁ f₂ f₃ f₄ ↔ Segments.crossTest A B C D :=
  floatCross_iff_of_sameSign (flDet_sign_certain_code hu hu1 h₁ t₁h b₁h g₁)
    (flDet_sign_certain_code hu hu1 h₂ t₂h b₂h g₂) (flDet_sign_certain_code hu hu1 h₃ t₃h b₃h g₃)
    (flDet_sign_certain_code hu hu1 h₄ t₄h b₄h g₄)

/-! ## (5) Which links can be misclassified -/

/-- Squared Euclidean distance in the plane (`ℝ × ℝ` carries the sup norm in Mathlib, so
distances are written out). -/
def sqd (X Y : ℝ × ℝ) : ℝ := (X.1 - Y.1) ^ 2 + (X.2 - Y.2) ^ 2

/-- The misclassification radius factor `κ = errA u (1 + u)³ ≈ 4u`. -/
noncomputable def kappa (u : ℝ) : ℝ := errA u * (1 + u) ^ 3

/-- Lagrange's identity, as an inequality: `orient(O, P, Q)² ≤ |OP|² |OQ|²`. -/
theorem orient_sq_le (O P Q : ℝ × ℝ) : Segments.orient O P Q ^ 2 ≤ sqd O P * sqd O Q := by
  simp only [Segments.orient, sqd]
  nlinarith [sq_nonneg ((P.1 - O.1) * (Q.1 - O.1) + (P.2 - O.2) * (Q.2 - O.2))]

/-- Cauchy-Schwarz for the two products of an orientation determinant:
`(|X| + |Y|)² ≤ |OP|² |OQ|²`. -/
theorem abs_add_abs_sq_le (O P Q : ℝ × ℝ) :
    (|(P.1 - O.1) * (Q.2 - O.2)| + |(P.2 - O.2) * (Q.1 - O.1)|) ^ 2 ≤ sqd O P * sqd O Q := by
  have e : sqd O P * sqd O Q = (|P.1 - O.1| ^ 2 + |P.2 - O.2| ^ 2) *
      (|Q.1 - O.1| ^ 2 + |Q.2 - O.2| ^ 2) := by
    simp only [sqd, sq_abs]
    ring
  rw [e, abs_mul, abs_mul]
  nlinarith [sq_nonneg (|P.1 - O.1| * |Q.1 - O.1| - |P.2 - O.2| * |Q.2 - O.2|)]

/-- The orientation determinant measures distance to a line: there is a point `Q'` on the
line `OP` with `|QQ'|² |OP|² = orient(O, P, Q)²`, i.e. `Q` is `|orient| / |OP|` from it. -/
theorem exists_on_line_sqd (O P Q : ℝ × ℝ) (hOP : sqd O P ≠ 0) :
    ∃ Q', Segments.orient O P Q' = 0 ∧ sqd Q Q' * sqd O P = Segments.orient O P Q ^ 2 := by
  refine ⟨(Q.1 + Segments.orient O P Q / sqd O P * (P.2 - O.2),
    Q.2 - Segments.orient O P Q / sqd O P * (P.1 - O.1)), ?_, ?_⟩
  · simp only [Segments.orient, sqd] at hOP ⊢
    field_simp
    ring
  · simp only [Segments.orient, sqd] at hOP ⊢
    field_simp
    ring

/-- **(5) One determinant.** If the computed sign of `orient(O, P, Q)` is wrong, then
`|orient(O, P, Q)| ≤ errA u (|l| + |r|)` (the point `Q` is within `B / |OP|` of the line
`OP`), and some point `Q'` of the line `OP` is within `κ |OQ|` of `Q`. -/
theorem orient_near_of_disagree {u : ℝ} (hu : 0 ≤ u) (hu1 : u < 1) {O P Q : ℝ × ℝ}
    {l r f : ℝ} (h : FlOrient u O P Q l r f) (hne : ¬ SameSign f (Segments.orient O P Q)) :
    |Segments.orient O P Q| ≤ errA u * (|l| + |r|) ∧
      ∃ Q', Segments.orient O P Q' = 0 ∧ sqd Q Q' ≤ kappa u ^ 2 * sqd O Q := by
  have herr : |f - Segments.orient O P Q| ≤ errA u * (|l| + |r|) := flDet_err hu hu1 h
  have hB := (abs_le_of_not_sameSign hne).trans herr
  refine ⟨hB, ?_⟩
  obtain ⟨p, q, p', q', hp, hq, hp', hq', hl, hr, hf⟩ := h
  have hL := rnd_prod hu hp hq hl
  have hR := rnd_prod hu hp' hq' hr
  have hl' : |l| ≤ (1 + u) ^ 3 * |(P.1 - O.1) * (Q.2 - O.2)| := by
    have := abs_sub_abs_le_abs_sub l ((P.1 - O.1) * (Q.2 - O.2))
    linarith
  have hr' : |r| ≤ (1 + u) ^ 3 * |(P.2 - O.2) * (Q.1 - O.1)| := by
    have := abs_sub_abs_le_abs_sub r ((P.2 - O.2) * (Q.1 - O.1))
    linarith
  have hE : 0 ≤ errA u := errA_nonneg hu hu1
  have h3 : 0 ≤ (1 + u) ^ 3 := pow_nonneg (by linarith) 3
  have habs : |Segments.orient O P Q| ≤ kappa u *
      (|(P.1 - O.1) * (Q.2 - O.2)| + |(P.2 - O.2) * (Q.1 - O.1)|) := by
    unfold kappa
    calc |Segments.orient O P Q| ≤ errA u * (|l| + |r|) := hB
      _ ≤ errA u * ((1 + u) ^ 3 *
          (|(P.1 - O.1) * (Q.2 - O.2)| + |(P.2 - O.2) * (Q.1 - O.1)|)) :=
          mul_le_mul_of_nonneg_left (by linarith) hE
      _ = errA u * (1 + u) ^ 3 *
          (|(P.1 - O.1) * (Q.2 - O.2)| + |(P.2 - O.2) * (Q.1 - O.1)|) := by ring
  have hsq : Segments.orient O P Q ^ 2 ≤ kappa u ^ 2 * (sqd O P * sqd O Q) := by
    have h1 : Segments.orient O P Q ^ 2 ≤ (kappa u *
        (|(P.1 - O.1) * (Q.2 - O.2)| + |(P.2 - O.2) * (Q.1 - O.1)|)) ^ 2 := by
      rw [← sq_abs]
      exact pow_le_pow_left₀ (abs_nonneg _) habs 2
    calc Segments.orient O P Q ^ 2 ≤ (kappa u *
          (|(P.1 - O.1) * (Q.2 - O.2)| + |(P.2 - O.2) * (Q.1 - O.1)|)) ^ 2 := h1
      _ = kappa u ^ 2 *
          (|(P.1 - O.1) * (Q.2 - O.2)| + |(P.2 - O.2) * (Q.1 - O.1)|) ^ 2 := by ring
      _ ≤ kappa u ^ 2 * (sqd O P * sqd O Q) :=
          mul_le_mul_of_nonneg_left (abs_add_abs_sq_le O P Q) (sq_nonneg _)
  by_cases hOP : sqd O P = 0
  · refine ⟨Q, ?_, ?_⟩
    · have := orient_sq_le O P Q
      rw [hOP, zero_mul] at this
      exact (pow_eq_zero_iff two_ne_zero).1 (le_antisymm this (sq_nonneg _))
    · have : sqd Q Q = 0 := by simp [sqd]
      rw [this]
      exact mul_nonneg (sq_nonneg _) (by unfold sqd; positivity)
  · obtain ⟨Q', hQ', heq⟩ := exists_on_line_sqd O P Q hOP
    refine ⟨Q', hQ', ?_⟩
    have hpos : 0 < sqd O P := lt_of_le_of_ne (by unfold sqd; positivity) (Ne.symm hOP)
    refine le_of_mul_le_mul_right ?_ hpos
    rw [heq]
    nlinarith [hsq]

/-- Being near a line, in the two forms of `orient_near_of_disagree`: `Q` is within
`errA u (|l| + |r|) / |OP|` of the line `OP`, and within `κ |OQ|` of a point of it. -/
def NearLine (u : ℝ) (O P Q : ℝ × ℝ) (l r : ℝ) : Prop :=
  |Segments.orient O P Q| ≤ errA u * (|l| + |r|) ∧
    ∃ Q', Segments.orient O P Q' = 0 ∧ sqd Q Q' ≤ kappa u ^ 2 * sqd O Q

/-- **(5) Which links can be misclassified.** If the browser's `segmentsCross` disagrees with
the exact test, then a link endpoint lies within rounding of the wall segment's line (`A` or
`B` near line `CD`), or an outline vertex lies within rounding of the link's line (`C` or `D`
near line `AB`): within `B / |segment|`, and within `κ ≈ 4u` times its distance to that
segment's first vertex. Links passing through or ending on an outline vertex or wall line,
or collinear with an outline edge, and only those. -/
theorem misclassified_near {u : ℝ} (hu : 0 ≤ u) (hu1 : u < 1) {A B C D : ℝ × ℝ}
    {l₁ r₁ f₁ l₂ r₂ f₂ l₃ r₃ f₃ l₄ r₄ f₄ : ℝ}
    (h₁ : FlOrient u C D A l₁ r₁ f₁) (h₂ : FlOrient u C D B l₂ r₂ f₂)
    (h₃ : FlOrient u A B C l₃ r₃ f₃) (h₄ : FlOrient u A B D l₄ r₄ f₄)
    (hne : ¬ (floatCross f₁ f₂ f₃ f₄ ↔ Segments.crossTest A B C D)) :
    NearLine u C D A l₁ r₁ ∨ NearLine u C D B l₂ r₂ ∨ NearLine u A B C l₃ r₃ ∨
      NearLine u A B D l₄ r₄ := by
  by_cases s₁ : SameSign f₁ (Segments.orient C D A)
  · by_cases s₂ : SameSign f₂ (Segments.orient C D B)
    · by_cases s₃ : SameSign f₃ (Segments.orient A B C)
      · by_cases s₄ : SameSign f₄ (Segments.orient A B D)
        · exact absurd (floatCross_iff_of_sameSign s₁ s₂ s₃ s₄) hne
        · exact Or.inr (Or.inr (Or.inr (orient_near_of_disagree hu hu1 h₄ s₄)))
      · exact Or.inr (Or.inr (Or.inl (orient_near_of_disagree hu hu1 h₃ s₃)))
    · exact Or.inr (Or.inl (orient_near_of_disagree hu hu1 h₂ s₂))
  · exact Or.inl (orient_near_of_disagree hu hu1 h₁ s₁)

/-- **(5) Numeric radius for doubles.** `κ(2⁻⁵³) ≤ 4.441e-16`: a misclassified link has an
endpoint or outline vertex within `4.441e-16 × (its distance to the other segment's first
vertex)` of the other segment's line - at building scale (distances below 300 m) within
`1.4e-13` m. -/
theorem kappa_double : kappa uDouble ≤ 4.441e-16 := by
  unfold kappa
  have h1 := errA_le (u := uDouble) (by linarith [uDouble_le])
  have h2 : 0 ≤ (1 + uDouble) ^ 3 := pow_nonneg (by linarith [uDouble_pos]) 3
  calc errA uDouble * (1 + uDouble) ^ 3 ≤ (4 + 7 * uDouble) * uDouble * (1 + uDouble) ^ 3 :=
        mul_le_mul_of_nonneg_right h1 h2
    _ ≤ 4.441e-16 := by unfold uDouble; norm_num

/-! ## (2) Summation -/

/-- The code's running sum (`let length = 0; length += Math.hypot(...)` in `findRoute` and
`straightenRoute`, and `dist[u] + w[e]` along A*'s route): `s 0 = x 0` because `0 + x₀` is
exact, and each later partial sum is the rounded sum of the previous one and the next term. -/
def FlSum (u : ℝ) (x s : ℕ → ℝ) : Prop := s 0 = x 0 ∧ ∀ k, Rnd u (s k + x (k + 1)) (s (k + 1))

/-- Higham's `γₘ = mu / (1 - mu)`. -/
noncomputable def gammaBound (m : ℕ) (u : ℝ) : ℝ := m * u / (1 - m * u)

/-- **(2) Left-to-right summation.** Summing `n + 1` non-negative terms with `n` rounded
additions gives a result within `((1 + u)ⁿ - 1)` of the exact sum, relatively. -/
theorem flSum_err {u : ℝ} (hu : 0 ≤ u) {x s : ℕ → ℝ} (hx : ∀ i, 0 ≤ x i) (hs : FlSum u x s)
    (n : ℕ) :
    |s n - ∑ i ∈ range (n + 1), x i| ≤ ((1 + u) ^ n - 1) * ∑ i ∈ range (n + 1), x i := by
  induction n with
  | zero => simp [hs.1]
  | succ n ih =>
    rw [sum_range_succ]
    have hS : 0 ≤ ∑ i ∈ range (n + 1), x i := sum_nonneg (fun i _ => hx i)
    have hxn := hx (n + 1)
    have hr : |s (n + 1) - (s n + x (n + 1))| ≤ u * |s n + x (n + 1)| := hs.2 n
    have hpow : 1 + u ≤ (1 + u) ^ (n + 1) := by
      have := one_add_mul_le_pow (show (-2 : ℝ) ≤ u by linarith) (n + 1)
      push_cast at this
      nlinarith
    have hpn : 1 ≤ (1 + u) ^ n := one_le_pow₀ (by linarith)
    have hsn : |s n| ≤ (1 + u) ^ n * ∑ i ∈ range (n + 1), x i := by
      have := abs_sub_abs_le_abs_sub (s n) (∑ i ∈ range (n + 1), x i)
      rw [abs_of_nonneg hS] at this
      nlinarith
    have h1 : |s n + x (n + 1)| ≤ (1 + u) ^ n * ∑ i ∈ range (n + 1), x i + x (n + 1) := by
      calc |s n + x (n + 1)| ≤ |s n| + |x (n + 1)| := abs_add_le _ _
        _ ≤ (1 + u) ^ n * ∑ i ∈ range (n + 1), x i + x (n + 1) := by
            rw [abs_of_nonneg hxn]
            linarith
    calc |s (n + 1) - (∑ i ∈ range (n + 1), x i + x (n + 1))|
        = |(s (n + 1) - (s n + x (n + 1))) + (s n - ∑ i ∈ range (n + 1), x i)| := by
          congr 1; ring
      _ ≤ |s (n + 1) - (s n + x (n + 1))| + |s n - ∑ i ∈ range (n + 1), x i| := abs_add_le _ _
      _ ≤ u * ((1 + u) ^ n * ∑ i ∈ range (n + 1), x i + x (n + 1)) +
          ((1 + u) ^ n - 1) * ∑ i ∈ range (n + 1), x i :=
          add_le_add (hr.trans (mul_le_mul_of_nonneg_left h1 hu)) ih
      _ ≤ ((1 + u) ^ (n + 1) - 1) * (∑ i ∈ range (n + 1), x i + x (n + 1)) := by
          have e2 : (1 + u) ^ (n + 1) * ∑ i ∈ range (n + 1), x i =
              (1 + u) ^ n * ∑ i ∈ range (n + 1), x i +
                u * ((1 + u) ^ n * ∑ i ∈ range (n + 1), x i) := by ring
          nlinarith [mul_le_mul_of_nonneg_right hpow hxn, e2]

/-- `(1 + u)ᵐ (1 - mu) ≤ 1` for every `m`. -/
theorem pow_mul_le_one {u : ℝ} (hu : 0 ≤ u) (m : ℕ) : (1 + u) ^ m * (1 - m * u) ≤ 1 := by
  induction m with
  | zero => simp
  | succ m ih =>
    have hp : 0 ≤ (1 + u) ^ m := pow_nonneg (by linarith) m
    have e : (1 + u) ^ (m + 1) * (1 - ((m + 1 : ℕ) : ℝ) * u) =
        (1 + u) ^ m * (1 - m * u) - (1 + u) ^ m * ((m + 1) * u ^ 2) := by
      push_cast
      ring
    rw [e]
    have : 0 ≤ (1 + u) ^ m * ((m + 1) * u ^ 2) := mul_nonneg hp (by positivity)
    linarith

/-- `(1 + u)ᵐ - 1 ≤ γₘ` whenever `mu < 1` (Higham's Lemma 3.1, one-sided). -/
theorem pow_sub_one_le_gamma {u : ℝ} (hu : 0 ≤ u) {m : ℕ} (h : m * u < 1) :
    (1 + u) ^ m - 1 ≤ gammaBound m u := by
  unfold gammaBound
  have h1 : 0 < 1 - m * u := by linarith
  rw [le_div_iff₀ h1]
  have := pow_mul_le_one hu m
  nlinarith

/-- **(2) A route of `n + 1` links** summed left to right has relative error at most `γₙ`:
`|s - S| ≤ γₙ S`. So a route length of `n` links is within `γₙ₋₁` of the exact sum of its
(computed) link lengths. -/
theorem route_sum_rel_err {u : ℝ} (hu : 0 ≤ u) {x s : ℕ → ℝ} (hx : ∀ i, 0 ≤ x i)
    (hs : FlSum u x s) (n : ℕ) (hn : n * u < 1) :
    |s n - ∑ i ∈ range (n + 1), x i| ≤ gammaBound n u * ∑ i ∈ range (n + 1), x i :=
  (flSum_err hu hx hs n).trans
    (mul_le_mul_of_nonneg_right (pow_sub_one_le_gamma hu hn) (sum_nonneg fun i _ => hx i))

/-- **(2) The number for a 10,000-link route in doubles:** `γ₉₉₉₉(2⁻⁵³) ≤ 1.1102e-12`, so the
reported length of a 5 km route is off by at most `5.6e-9` m from the exact sum of its link
lengths. -/
theorem gamma_10000_links : gammaBound 9999 uDouble ≤ 1.1102e-12 := by
  unfold gammaBound uDouble
  norm_num

/-- **(2) Route length with per-link errors.** If each computed link length is within
relative `ε ≤ 1` of the true length and the lengths are summed left to right, the total is
within `((1 + ε)(1 + u)ⁿ - 1)` of the true route length (`n + 1` links), relatively. -/
theorem route_length_err {u ε : ℝ} (hu : 0 ≤ u) (hε1 : ε ≤ 1) {x xt s : ℕ → ℝ}
    (hx : ∀ i, 0 ≤ x i) (hxt : ∀ i, |xt i - x i| ≤ ε * x i) (hs : FlSum u xt s) (n : ℕ) :
    |s n - ∑ i ∈ range (n + 1), x i| ≤
      ((1 + ε) * (1 + u) ^ n - 1) * ∑ i ∈ range (n + 1), x i := by
  have hxt0 : ∀ i, 0 ≤ xt i := by
    intro i
    have := (abs_le.1 (hxt i)).1
    nlinarith [hx i]
  have h1 := flSum_err hu hxt0 hs n
  have hS : 0 ≤ ∑ i ∈ range (n + 1), x i := sum_nonneg fun i _ => hx i
  have h2 : |∑ i ∈ range (n + 1), xt i - ∑ i ∈ range (n + 1), x i| ≤
      ε * ∑ i ∈ range (n + 1), x i := by
    rw [← sum_sub_distrib, mul_sum]
    exact (abs_sum_le_sum_abs _ _).trans (sum_le_sum fun i _ => hxt i)
  have h3 : ∑ i ∈ range (n + 1), xt i ≤ (1 + ε) * ∑ i ∈ range (n + 1), x i := by
    have := (abs_le.1 h2).2
    linarith
  have hpn : 1 ≤ (1 + u) ^ n := one_le_pow₀ (by linarith)
  calc |s n - ∑ i ∈ range (n + 1), x i|
      ≤ |s n - ∑ i ∈ range (n + 1), xt i| +
          |∑ i ∈ range (n + 1), xt i - ∑ i ∈ range (n + 1), x i| := abs_sub_le _ _ _
    _ ≤ ((1 + u) ^ n - 1) * ((1 + ε) * ∑ i ∈ range (n + 1), x i) +
          ε * ∑ i ∈ range (n + 1), x i :=
        add_le_add (h1.trans (mul_le_mul_of_nonneg_left h3 (by linarith))) h2
    _ = ((1 + ε) * (1 + u) ^ n - 1) * ∑ i ∈ range (n + 1), x i := by ring

/-! ## (3) Routing under perturbed weights -/

/-- **(3) Near-optimality from relative cost errors.** If every candidate route's computed
cost `c'` is within relative `ε < 1` of its true cost `c`, a route optimal for `c'` costs at
most `(1 + ε)/(1 - ε)` times the true optimum. `W` is any set of routes (for Studio: walks
from the start to the exit over open links); `AStar.astar_correct` supplies the optimality
for the computed weights. -/
theorem near_optimal_of_rel_err {ι : Type*} {W : Set ι} {c c' : ι → ℝ} {ε : ℝ} (hε : ε < 1)
    (herr : ∀ π ∈ W, |c' π - c π| ≤ ε * c π) {π₀ : ι} (h₀ : π₀ ∈ W)
    (hopt : ∀ π ∈ W, c' π₀ ≤ c' π) : ∀ π ∈ W, c π₀ ≤ (1 + ε) / (1 - ε) * c π := by
  intro π hπ
  have h1 : 0 < 1 - ε := by linarith
  have a := (abs_le.1 (herr π₀ h₀)).1
  have b := (abs_le.1 (herr π hπ)).2
  have := hopt π hπ
  rw [div_mul_eq_mul_div, le_div_iff₀ h1]
  nlinarith

/-- Summing relative link errors: `|Σ w' - Σ w| ≤ ε Σ w` along any route. -/
theorem sum_map_rel_err {L : Type*} {w w' : L → ℝ} {ε : ℝ}
    (herr : ∀ e, |w' e - w e| ≤ ε * w e) (π : List L) :
    |(π.map w').sum - (π.map w).sum| ≤ ε * (π.map w).sum := by
  induction π with
  | nil => simp
  | cons e π ih =>
    simp only [List.map_cons, List.sum_cons]
    calc |(w' e + (π.map w').sum) - (w e + (π.map w).sum)|
        = |(w' e - w e) + ((π.map w').sum - (π.map w).sum)| := by congr 1; ring
      _ ≤ |w' e - w e| + |(π.map w').sum - (π.map w).sum| := abs_add_le _ _
      _ ≤ ε * w e + ε * (π.map w).sum := add_le_add (herr e) ih
      _ = ε * (w e + (π.map w).sum) := by ring

/-- Summing absolute link errors: `|Σ w' - Σ w| ≤ α · (number of links)`. -/
theorem sum_map_abs_err {L : Type*} {w w' : L → ℝ} {α : ℝ}
    (herr : ∀ e, |w' e - w e| ≤ α) (π : List L) :
    |(π.map w').sum - (π.map w).sum| ≤ α * π.length := by
  induction π with
  | nil => simp
  | cons e π ih =>
    simp only [List.map_cons, List.sum_cons, List.length_cons]
    push_cast
    calc |(w' e + (π.map w').sum) - (w e + (π.map w).sum)|
        = |(w' e - w e) + ((π.map w').sum - (π.map w).sum)| := by congr 1; ring
      _ ≤ |w' e - w e| + |(π.map w').sum - (π.map w).sum| := abs_add_le _ _
      _ ≤ α + α * π.length := add_le_add (herr e) ih
      _ = α * (π.length + 1) := by ring

/-- **(3) Perturbed link weights, relative.** If every computed link weight `w'` is within
relative `ε < 1` of the true weight `w ≥ 0`, a route optimal for `w'` is within the factor
`(1 + ε)/(1 - ε)` of optimal for `w`. -/
theorem perturbed_weights_near_optimal {L : Type*} {W : Set (List L)} {w w' : L → ℝ} {ε : ℝ}
    (hε : ε < 1) (herr : ∀ e, |w' e - w e| ≤ ε * w e) {π₀ : List L} (h₀ : π₀ ∈ W)
    (hopt : ∀ π ∈ W, (π₀.map w').sum ≤ (π.map w').sum) :
    ∀ π ∈ W, (π₀.map w).sum ≤ (1 + ε) / (1 - ε) * (π.map w).sum :=
  near_optimal_of_rel_err (c := fun π => (π.map w).sum) (c' := fun π => (π.map w').sum) hε
    (fun π _ => sum_map_rel_err herr π) h₀ hopt

/-- **(3) Perturbed link weights, absolute** (the form float32 coordinates give: every link
length off by at most `α = 2uM`, `length_err_rounded_points`). A route `π₀` optimal for `w'`
costs at most `α (k₀ + k)` more than any route `π` under `w`, where `k₀`, `k` are their link
counts. -/
theorem perturbed_weights_near_optimal_abs {L : Type*} {W : Set (List L)} {w w' : L → ℝ}
    {α : ℝ} (herr : ∀ e, |w' e - w e| ≤ α) {π₀ : List L}
    (hopt : ∀ π ∈ W, (π₀.map w').sum ≤ (π.map w').sum) :
    ∀ π ∈ W, (π₀.map w).sum ≤ (π.map w).sum + α * (π₀.length + π.length) := by
  intro π hπ
  have a := (abs_le.1 (sum_map_abs_err herr π₀)).1
  have b := (abs_le.1 (sum_map_abs_err herr π)).2
  have := hopt π hπ
  nlinarith

/-- Where the `δ` of a float heuristic comes from. If the exact heuristic is consistent on a
link (`h₁ ≤ w + h₂`) and the computed weight and heuristic values are within relative `ε`,
the computed values are consistent up to `δ = ε (w + h₁ + h₂)`: for Studio,
`ε (link length + 2 × distance to the goal)`, about `2u × 600 m ≈ 1.3e-13` m at building
scale if `Math.hypot` is within 1 ulp; `-4.44e-16` on the unit-scale test graphs. -/
theorem float_reduced_ge {w h₁ h₂ w' h₁' h₂' ε : ℝ} (hc : h₁ ≤ w + h₂)
    (ew : |w' - w| ≤ ε * w) (e₁ : |h₁' - h₁| ≤ ε * h₁) (e₂ : |h₂' - h₂| ≤ ε * h₂) :
    h₁' ≤ w' + h₂' + ε * (w + h₁ + h₂) := by
  have := (abs_le.1 ew).1
  have := (abs_le.1 e₁).2
  have := (abs_le.1 e₂).1
  nlinarith

/-! ### Closed-set A* with a heuristic consistent up to `δ` -/

section AStarDelta

variable {V : Type*}

/-- A walk from `s` to `v` whose *clipped* cost - the sum of `max a 0` over its links - is
`p`. Negative links count as `0`. -/
inductive ClipWalk (E : V → V → ℝ → Prop) (s : V) : V → ℝ → Prop
  | nil : ClipWalk E s s 0
  | snoc {u v : V} {p a : ℝ} : ClipWalk E s u p → E u v a → ClipWalk E s v (p + max a 0)

/-- A walk from `s` to `v` of cost `c` with `k` links. -/
inductive LenWalk (E : V → V → ℝ → Prop) (s : V) : V → ℝ → ℕ → Prop
  | nil : LenWalk E s s 0 0
  | snoc {u v : V} {c a : ℝ} {k : ℕ} : LenWalk E s u c k → E u v a →
      LenWalk E s v (c + a) (k + 1)

/-- A walk from `s` to `v` of cost `c` whose links' heuristic inconsistencies
`max (h x - a - h y) 0` add up to `i`. -/
inductive IncWalk (E : V → V → ℝ → Prop) (h : V → ℝ) (s : V) : V → ℝ → ℝ → Prop
  | nil : IncWalk E h s s 0 0
  | snoc {u v : V} {c i a : ℝ} : IncWalk E h s u c i → E u v a →
      IncWalk E h s v (c + a) (i + max (h u - a - h v) 0)

/-- Dijkstra's invariant without a sign condition on the links: settled labels are at most
the clipped cost of every walk (`opt`); the rest is as in `AStar.Inv`. -/
structure InvClip (E : V → V → ℝ → Prop) (s : V) (σ : AStar.State V) : Prop where
  src : s ∈ σ.R ∧ σ.d s ≤ 0
  sub : σ.S ⊆ σ.R
  real : ∀ v ∈ σ.R, AStar.WalkCost E s v (σ.d v)
  opt : ∀ v ∈ σ.S, ∀ p, ClipWalk E s v p → σ.d v ≤ p
  relaxed : ∀ x ∈ σ.S, ∀ v a, E x v a → v ∉ σ.S → v ∈ σ.R ∧ σ.d v ≤ σ.d x + a

/-- The label-setting loop starts in a state satisfying the clipped invariant. -/
theorem init_invClip (E : V → V → ℝ → Prop) (s : V) (σ : AStar.State V) (h : AStar.Init s σ) :
    InvClip E s σ := by
  obtain ⟨hS, hR, hd⟩ := h
  refine ⟨⟨by simp [hR], by simp [hd]⟩, by simp [hS], ?_, by simp [hS], by simp [hS]⟩
  intro v hv
  rw [hR, mem_singleton] at hv
  subst hv
  exact AStar.WalkCost.nil.cast hd.symm

/-- With links of any sign, a vertex the label-setting loop may pop has label at most the
clipped cost of every walk from the source to any unsettled vertex. -/
theorem selectable_le_clip (E : V → V → ℝ → Prop) (s : V) (σ : AStar.State V)
    (hI : InvClip E s σ) (u : V) (hu : AStar.Selectable (fun _ => 0) σ u) :
    ∀ x p, ClipWalk E s x p → x ∉ σ.S → σ.d u ≤ p := by
  intro x p hw
  induction hw with
  | nil =>
    intro hs
    have := hu.2.2 s hI.src.1 hs
    linarith [hI.src.2]
  | @snoc y x p a hw' hyx ih =>
    intro hx
    by_cases hy : y ∈ σ.S
    · obtain ⟨hxR, hxd⟩ := hI.relaxed y hy x a hyx hx
      have h1 := hI.opt y hy p hw'
      have h2 := hu.2.2 x hxR hx
      have h3 := le_max_left a 0
      linarith
    · have := ih hy
      have h3 := le_max_right a 0
      linarith

/-- An original walk with `k` links, under reduced weights that are all `≥ -δ`, is a reduced
walk of clipped cost at most `c - h s + h v + kδ`. -/
theorem lenWalk_reduced_clip (E : V → V → ℝ → Prop) (h : V → ℝ) {δ : ℝ} (hδ0 : 0 ≤ δ)
    (hδ : ∀ u v a, E u v a → h u ≤ a + h v + δ) {s v : V} {c : ℝ} {k : ℕ}
    (hw : LenWalk E s v c k) :
    ∃ p, ClipWalk (AStar.reduced E h) s v p ∧ p ≤ c - h s + h v + k * δ := by
  induction hw with
  | nil => exact ⟨0, ClipWalk.nil, by simp⟩
  | @snoc u v c a k hw' hE ih =>
    obtain ⟨p, hp, hle⟩ := ih
    refine ⟨p + max (a - h u + h v) 0, hp.snoc ⟨a, hE, rfl⟩, ?_⟩
    have := hδ u v a hE
    have hmax : max (a - h u + h v) 0 ≤ (a - h u + h v) + δ := max_le (by linarith) (by linarith)
    push_cast
    nlinarith

/-- `max t 0 = t + max (-t) 0`. -/
theorem max_zero_eq (t : ℝ) : max t 0 = t + max (-t) 0 := by
  rcases le_total t 0 with ht | ht
  · rw [max_eq_right ht, max_eq_left (by linarith)]
    ring
  · rw [max_eq_left ht, max_eq_right (by linarith)]
    ring

/-- An original walk with inconsistency sum `i` is a reduced walk of clipped cost exactly
`c - h s + h v + i`. -/
theorem incWalk_reduced_clip (E : V → V → ℝ → Prop) (h : V → ℝ) {s v : V} {c i : ℝ}
    (hw : IncWalk E h s v c i) : ClipWalk (AStar.reduced E h) s v (c - h s + h v + i) := by
  induction hw with
  | nil =>
    have e : (0 : ℝ) - h s + h s + 0 = 0 := by ring
    rw [e]
    exact ClipWalk.nil
  | @snoc u v c i a hw' hE ih =>
    have e : c - h s + h u + i + max (a - h u + h v) 0 =
        c + a - h s + h v + (i + max (h u - a - h v) 0) := by
      rw [max_zero_eq (a - h u + h v)]
      ring_nf
    rw [← e]
    exact ih.snoc ⟨a, hE, rfl⟩

variable [DecidableEq V]

/-- A label-setting step preserves the clipped invariant, whatever the signs of the links. -/
theorem step_invClip (E : V → V → ℝ → Prop) (s : V) (σ σ' : AStar.State V)
    (hI : InvClip E s σ) (hst : AStar.Step E (fun _ => 0) σ σ') : InvClip E s σ' := by
  obtain ⟨u, hu, hS, hR, hfix, hdrop, hrelax, horig⟩ := hst
  have huopt := selectable_le_clip E s σ hI u hu
  refine ⟨⟨hR hI.src.1, le_trans (hdrop s hI.src.1) hI.src.2⟩, ?_, ?_, ?_, ?_⟩
  · intro v hv
    rw [hS, mem_insert] at hv
    rcases hv with rfl | hv
    · exact hR hu.1
    · exact hR (hI.sub hv)
  · intro v hv
    rcases horig v hv with ⟨hvR, hd⟩ | ⟨a, ha, hd⟩
    · rw [hd]; exact hI.real v hvR
    · rw [hd]; exact (hI.real u hu.1).snoc ha
  · intro v hv p hp
    rw [hfix v hv]
    rw [hS, mem_insert] at hv
    rcases hv with rfl | hv
    · exact huopt v p hp hu.2.1
    · exact hI.opt v hv p hp
  · intro x hx v a hxv hv
    have hx' := hx
    rw [hS, mem_insert] at hx'
    rcases hx' with rfl | hxS
    · obtain ⟨h1, h2⟩ := hrelax v a hxv hv
      exact ⟨h1, by rw [hfix x hx]; exact h2⟩
    · have hvS : v ∉ σ.S := fun h => hv (by rw [hS]; exact mem_insert_of_mem h)
      obtain ⟨hvR, hvd⟩ := hI.relaxed x hxS v a hxv hvS
      refine ⟨hR hvR, ?_⟩
      rw [hfix x hx]
      exact le_trans (hdrop v hvR) hvd

/-- Every state reachable from the start satisfies the clipped invariant. -/
theorem reachable_invClip (E : V → V → ℝ → Prop) (s : V) (σ₀ σ : AStar.State V)
    (h0 : AStar.Init s σ₀) (hrun : Relation.ReflTransGen (AStar.Step E (fun _ => 0)) σ₀ σ) :
    InvClip E s σ := by
  induction hrun with
  | refl => exact init_invClip E s σ₀ h0
  | tail _ hst ih => exact step_invClip E s _ _ ih hst

/-- **Dijkstra with links of any sign.** At any point of any run, the vertex popped next has
as label the cost of a real walk, at most the clipped cost (negative links counted as `0`)
of every walk from the source to it. With non-negative links this is
`AStar.dijkstra_correct`. -/
theorem dijkstra_clip (E : V → V → ℝ → Prop) (s : V) (σ₀ σ : AStar.State V)
    (h0 : AStar.Init s σ₀) (hrun : Relation.ReflTransGen (AStar.Step E (fun _ => 0)) σ₀ σ)
    (u : V) (hu : AStar.Selectable (fun _ => 0) σ u) :
    AStar.WalkCost E s u (σ.d u) ∧ ∀ p, ClipWalk E s u p → σ.d u ≤ p := by
  have hI := reachable_invClip E s σ₀ σ h0 hrun
  exact ⟨hI.real u hu.1, fun p hp => selectable_le_clip E s σ hI u hu u p hp hu.2.1⟩

/-- **(3) A* with a heuristic consistent up to `δ`** (every reduced weight
`a - h u + h v ≥ -δ`), closed set and no reopening, as `route.js` runs it. The label of the
vertex A* pops - in particular `dist[end]` when it pops the goal - is the cost of a real
walk, and for every walk of cost `c` with `k` links, `label ≤ c + kδ`. Taking an optimal
route: A*'s answer exceeds the optimum by at most `δ` times the number of links of an optimal
route. No admissibility is assumed. -/
theorem astar_delta_near_optimal (E : V → V → ℝ → Prop) (h : V → ℝ) {δ : ℝ} (hδ0 : 0 ≤ δ)
    (hδ : ∀ u v a, E u v a → h u ≤ a + h v + δ) (s : V) (σ₀ σ : AStar.State V)
    (h0 : AStar.Init s σ₀) (hrun : Relation.ReflTransGen (AStar.Step E h) σ₀ σ) (u : V)
    (hu : AStar.Selectable h σ u) :
    AStar.WalkCost E s u (σ.d u) ∧ ∀ c k, LenWalk E s u c k → σ.d u ≤ c + k * δ := by
  have hrun' := AStar.astar_run_is_dijkstra_run E h s σ₀ σ hrun
  have hd := dijkstra_clip (AStar.reduced E h) s _ _ (AStar.shift_init h s σ₀ h0) hrun' u
    (AStar.shift_selectable h s σ u hu)
  refine ⟨?_, fun c k hw => ?_⟩
  · rw [AStar.walkCost_reduced_iff E h]
    exact hd.1.cast (by simp only [AStar.shift]; ring)
  · obtain ⟨p, hp, hle⟩ := lenWalk_reduced_clip E h hδ0 hδ hw
    have := hd.2 p hp
    simp only [AStar.shift] at this
    linarith

/-- **(3) The sharp form.** For every walk of cost `c` whose links' inconsistencies
`max (h x - a - h y) 0` sum to `i`, the label A* pops is at most `c + i`. The excess over the
optimum is at most the inconsistency accumulated along an optimal route (the bound of
Valenzano, Sturtevant and Schaeffer for best-first search without re-expansions); the
counterexample below meets it up to the decoy slack. -/
theorem astar_le_cost_add_inconsistency (E : V → V → ℝ → Prop) (h : V → ℝ) (s : V)
    (σ₀ σ : AStar.State V) (h0 : AStar.Init s σ₀)
    (hrun : Relation.ReflTransGen (AStar.Step E h) σ₀ σ) (u : V)
    (hu : AStar.Selectable h σ u) : ∀ c i, IncWalk E h s u c i → σ.d u ≤ c + i := by
  have hrun' := AStar.astar_run_is_dijkstra_run E h s σ₀ σ hrun
  have hd := dijkstra_clip (AStar.reduced E h) s _ _ (AStar.shift_init h s σ₀ h0) hrun' u
    (AStar.shift_selectable h s σ u hu)
  intro c i hw
  have := hd.2 _ (incWalk_reduced_clip E h hw)
  simp only [AStar.shift] at this
  linarith

end AStarDelta

/-! ### The naive bounds fail: a counterexample -/

/-- Links of the counterexample, both directions; `0` is the start and `3` the goal.
`0 - 3` (43) is a one-link decoy; `0 - 1 - 2 - 3` (17 + 10 + 10 = 37) is optimal. -/
def cxE (u v : ℕ) (a : ℝ) : Prop :=
  (u = 0 ∧ v = 1 ∧ a = 17) ∨ (u = 1 ∧ v = 0 ∧ a = 17) ∨
  (u = 0 ∧ v = 2 ∧ a = 30) ∨ (u = 2 ∧ v = 0 ∧ a = 30) ∨
  (u = 0 ∧ v = 3 ∧ a = 43) ∨ (u = 3 ∧ v = 0 ∧ a = 43) ∨
  (u = 1 ∧ v = 2 ∧ a = 10) ∨ (u = 2 ∧ v = 1 ∧ a = 10) ∨
  (u = 2 ∧ v = 3 ∧ a = 10) ∨ (u = 3 ∧ v = 2 ∧ a = 10)

/-- The heuristic: `42, 28, 14, 0`. It drops by 14 along each 10-long chain link, so it is
consistent up to `δ = 4` there (and exactly consistent everywhere else). -/
noncomputable def cxH (v : ℕ) : ℝ :=
  if v = 0 then 42 else if v = 1 then 28 else if v = 2 then 14 else 0

/-- A*'s start state. -/
noncomputable def cxS0 : AStar.State ℕ := ⟨∅, {0}, fun _ => 0⟩

/-- A*'s state after popping the start: `g = 17, 30, 43` on `1, 2, 3`, so
`g + h = 45, 44, 43`. -/
noncomputable def cxS1 : AStar.State ℕ :=
  ⟨{0}, {0, 1, 2, 3}, fun v => if v = 1 then 17 else if v = 2 then 30 else if v = 3 then 43 else 0⟩

theorem cx_pos : ∀ u v a, cxE u v a → 0 < a := by
  intro u v a h
  rcases h with ⟨-, -, rfl⟩ | ⟨-, -, rfl⟩ | ⟨-, -, rfl⟩ | ⟨-, -, rfl⟩ | ⟨-, -, rfl⟩ |
    ⟨-, -, rfl⟩ | ⟨-, -, rfl⟩ | ⟨-, -, rfl⟩ | ⟨-, -, rfl⟩ | ⟨-, -, rfl⟩ <;> norm_num

theorem cx_consistent : ∀ u v a, cxE u v a → cxH u ≤ a + cxH v + 4 := by
  intro u v a h
  rcases h with ⟨rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl⟩ |
    ⟨rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl⟩ |
    ⟨rfl, rfl, rfl⟩ <;> norm_num [cxH]

theorem cx_step : AStar.Step cxE cxH cxS0 cxS1 := by
  refine ⟨0, ⟨by simp [cxS0], by simp [cxS0], ?_⟩, by simp [cxS0, cxS1], by simp [cxS0, cxS1],
    ?_, ?_, ?_, ?_⟩
  · intro v hv _
    simp only [cxS0, Finset.mem_singleton] at hv
    subst hv
    exact le_rfl
  · intro v hv
    simp only [cxS1, Finset.mem_singleton] at hv
    subst hv
    simp [cxS0, cxS1]
  · intro v hv
    simp only [cxS0, Finset.mem_singleton] at hv
    subst hv
    simp [cxS0, cxS1]
  · intro v a hE _
    rcases hE with ⟨-, rfl, rfl⟩ | ⟨h, -⟩ | ⟨-, rfl, rfl⟩ | ⟨h, -⟩ | ⟨-, rfl, rfl⟩ | ⟨h, -⟩ |
      ⟨h, -⟩ | ⟨h, -⟩ | ⟨h, -⟩ | ⟨h, -⟩
    all_goals first
      | exact absurd h (by norm_num)
      | exact ⟨by simp [cxS1], by norm_num [cxS0, cxS1]⟩
  · intro v hv
    simp only [cxS1, Finset.mem_insert, Finset.mem_singleton] at hv
    rcases hv with rfl | rfl | rfl | rfl
    · left
      simp [cxS0, cxS1]
    · right
      exact ⟨17, by norm_num [cxE], by norm_num [cxS0, cxS1]⟩
    · right
      exact ⟨30, by norm_num [cxE], by norm_num [cxS0, cxS1]⟩
    · right
      exact ⟨43, by norm_num [cxE], by norm_num [cxS0, cxS1]⟩

theorem cx_selectable : AStar.Selectable cxH cxS1 3 := by
  refine ⟨by simp [cxS1], by simp [cxS1], ?_⟩
  intro v hv hv0
  simp only [cxS1, Finset.mem_insert, Finset.mem_singleton] at hv hv0
  rcases hv with rfl | rfl | rfl | rfl
  · exact absurd rfl hv0
  all_goals norm_num [cxS1, cxH]

theorem cx_walk : LenWalk cxE 0 3 37 3 := by
  have h := ((LenWalk.nil.snoc (show cxE 0 1 17 by norm_num [cxE])).snoc
    (show cxE 1 2 10 by norm_num [cxE])).snoc (show cxE 2 3 10 by norm_num [cxE])
  norm_num at h
  exact h

/-- **(3) Counterexample: the naive bounds fail.** A graph with positive weights and a
heuristic consistent up to `δ = 4` (`h(goal) = 0`) on which closed-set A*, after one step,
pops the goal with label `43` - the cost of the one-link decoy - although the route
`0 - 1 - 2 - 3` costs `37`. The error `6 = 1.5 δ` exceeds `δ`, and exceeds `δ` times the
number of links of the route A* returns (one). It respects `astar_delta_near_optimal`
(`37 + 3 δ = 49`) and the sharp `astar_le_cost_add_inconsistency` (`37 + 2 δ = 45`). Longer
chains push the error to `(k - 1)(δ - η)` for decoy gaps `η → 0`; an admissible, undirected
variant (one more link after the chain) gives `2.25 δ` against a two-link answer. -/
theorem astar_delta_counterexample :
    (∀ u v a, cxE u v a → 0 < a) ∧ (∀ u v a, cxE u v a → cxH u ≤ a + cxH v + 4) ∧
      cxH 3 = 0 ∧ AStar.Init 0 cxS0 ∧ Relation.ReflTransGen (AStar.Step cxE cxH) cxS0 cxS1 ∧
      AStar.Selectable cxH cxS1 3 ∧ cxS1.d 3 = 43 ∧ AStar.WalkCost cxE 0 3 43 ∧
      LenWalk cxE 0 3 37 3 ∧ 37 + 1 * 4 < cxS1.d 3 ∧ cxS1.d 3 ≤ 37 + 3 * 4 := by
  refine ⟨cx_pos, cx_consistent, by norm_num [cxH], by simp [AStar.Init, cxS0],
    Relation.ReflTransGen.single cx_step, cx_selectable, by norm_num [cxS1], ?_, cx_walk,
    by norm_num [cxS1], by norm_num [cxS1]⟩
  exact (AStar.WalkCost.nil.snoc (show cxE 0 3 43 by norm_num [cxE])).cast (by norm_num)

/-! ## (4) Float32 coordinates -/

/-- **(4) Rounded points, exact lengths.** In any normed space, if each of two points moves by
at most `u` times its distance from the origin (`‖P' - P‖ ≤ u ‖P‖`), and both are within `M`
of the origin, the distance between them changes by at most `2uM`. -/
theorem length_err_rounded_points {E : Type*} [SeminormedAddCommGroup E] {u M : ℝ} {P Q P' Q' : E}
    (hu : 0 ≤ u) (hP : ‖P' - P‖ ≤ u * ‖P‖) (hQ : ‖Q' - Q‖ ≤ u * ‖Q‖) (hPM : ‖P‖ ≤ M)
    (hQM : ‖Q‖ ≤ M) : |‖P' - Q'‖ - ‖P - Q‖| ≤ 2 * u * M := by
  calc |‖P' - Q'‖ - ‖P - Q‖| ≤ ‖(P' - Q') - (P - Q)‖ := abs_norm_sub_norm_le _ _
    _ = ‖(P' - P) - (Q' - Q)‖ := by congr 1; abel
    _ ≤ ‖P' - P‖ + ‖Q' - Q‖ := norm_sub_le _ _
    _ ≤ u * ‖P‖ + u * ‖Q‖ := add_le_add hP hQ
    _ ≤ 2 * u * M := by
        nlinarith [mul_le_mul_of_nonneg_left hPM hu, mul_le_mul_of_nonneg_left hQM hu]

/-- **(4) Coordinatewise rounding** (`Float32Array` stores each coordinate rounded): if every
coordinate satisfies the model, the point moves by at most `u ‖P‖` in the Euclidean norm.
`‖P' - Q'‖` here is exactly what `Math.hypot` approximates. -/
theorem round_coords_norm {n : ℕ} {u : ℝ} (hu : 0 ≤ u) (P P' : EuclideanSpace ℝ (Fin n))
    (h : ∀ i, Rnd u (P i) (P' i)) : ‖P' - P‖ ≤ u * ‖P‖ := by
  rw [EuclideanSpace.norm_eq, EuclideanSpace.norm_eq]
  have key : ∑ i, ‖(P' - P).ofLp i‖ ^ 2 ≤ u ^ 2 * ∑ i, ‖P.ofLp i‖ ^ 2 := by
    rw [mul_sum]
    refine sum_le_sum fun i _ => ?_
    have hi : |P' i - P i| ≤ u * |P i| := h i
    rw [PiLp.sub_apply, Real.norm_eq_abs, Real.norm_eq_abs]
    calc |P'.ofLp i - P.ofLp i| ^ 2 ≤ (u * |P.ofLp i|) ^ 2 :=
          pow_le_pow_left₀ (abs_nonneg _) hi 2
      _ = u ^ 2 * |P.ofLp i| ^ 2 := by ring
  calc √(∑ i, ‖(P' - P).ofLp i‖ ^ 2) ≤ √(u ^ 2 * ∑ i, ‖P.ofLp i‖ ^ 2) := Real.sqrt_le_sqrt key
    _ = u * √(∑ i, ‖P.ofLp i‖ ^ 2) := by rw [Real.sqrt_mul (sq_nonneg u), Real.sqrt_sq hu]

/-- **(4) Studio's instance.** Web-ifc moves the model near the origin, so lattice nodes lie
within about `M = 100` m of it; storing them in a `Float32Array` (`u = 2⁻²⁴`) changes any link
length - and any distance the heuristic measures - by at most `2 · 2⁻²⁴ · 100 ≤ 1.2e-5` m. -/
theorem studio_float32_length_err {n : ℕ} (P Q P' Q' : EuclideanSpace ℝ (Fin n))
    (hP : ∀ i, Rnd uSingle (P i) (P' i)) (hQ : ∀ i, Rnd uSingle (Q i) (Q' i))
    (hPM : ‖P‖ ≤ 100) (hQM : ‖Q‖ ≤ 100) : |‖P' - Q'‖ - ‖P - Q‖| ≤ 1.2e-5 := by
  have hu : 0 ≤ uSingle := by unfold uSingle; positivity
  refine (length_err_rounded_points hu (round_coords_norm hu P P' hP)
    (round_coords_norm hu Q Q' hQ) hPM hQM).trans ?_
  unfold uSingle
  norm_num

/-- **(4) With `Math.hypot`'s own error.** If the computed length `ℓ` is within relative `ε`
of the exact length of the rounded points, it is within `2uM + ε (‖P - Q‖ + 2uM)` of the
exact length of the unrounded points. For doubles `ε` is a few `2⁻⁵³`, so the float32 term
dominates by about nine orders of magnitude. -/
theorem computed_length_err {E : Type*} [SeminormedAddCommGroup E] {u M ε ℓ : ℝ}
    {P Q P' Q' : E} (hu : 0 ≤ u) (hε : 0 ≤ ε) (hP : ‖P' - P‖ ≤ u * ‖P‖)
    (hQ : ‖Q' - Q‖ ≤ u * ‖Q‖) (hPM : ‖P‖ ≤ M) (hQM : ‖Q‖ ≤ M)
    (hℓ : |ℓ - ‖P' - Q'‖| ≤ ε * ‖P' - Q'‖) :
    |ℓ - ‖P - Q‖| ≤ 2 * u * M + ε * (‖P - Q‖ + 2 * u * M) := by
  have h1 := length_err_rounded_points hu hP hQ hPM hQM
  have h2 : ‖P' - Q'‖ ≤ ‖P - Q‖ + 2 * u * M := by
    have := (abs_le.1 h1).2
    linarith
  calc |ℓ - ‖P - Q‖| ≤ |ℓ - ‖P' - Q'‖| + |‖P' - Q'‖ - ‖P - Q‖| := abs_sub_le _ _ _
    _ ≤ ε * (‖P - Q‖ + 2 * u * M) + 2 * u * M :=
        add_le_add (hℓ.trans (mul_le_mul_of_nonneg_left h2 hε)) h1
    _ = 2 * u * M + ε * (‖P - Q‖ + 2 * u * M) := by ring

end FireEgress.FloatError

#print axioms FireEgress.FloatError.rnd_iff
#print axioms FireEgress.FloatError.fe_abs_sub_le
#print axioms FireEgress.FloatError.rel_trans
#print axioms FireEgress.FloatError.rel_mul
#print axioms FireEgress.FloatError.rnd_back
#print axioms FireEgress.FloatError.rnd_prod
#print axioms FireEgress.FloatError.rnd_prod_back
#print axioms FireEgress.FloatError.det_err_core
#print axioms FireEgress.FloatError.det_err_core'
#print axioms FireEgress.FloatError.det_err_core_back
#print axioms FireEgress.FloatError.flDet_err_exact
#print axioms FireEgress.FloatError.flDet_err
#print axioms FireEgress.FloatError.errA_le
#print axioms FireEgress.FloatError.errA_nonneg
#print axioms FireEgress.FloatError.sameSign_of_lt
#print axioms FireEgress.FloatError.abs_le_of_not_sameSign
#print axioms FireEgress.FloatError.flDet_sign_certain
#print axioms FireEgress.FloatError.flDet_exact_of_zero
#print axioms FireEgress.FloatError.flDet_sign_certain_shewchuk
#print axioms FireEgress.FloatError.bound_eval_safe
#print axioms FireEgress.FloatError.code_constant_safe
#print axioms FireEgress.FloatError.flDet_sign_certain_code
#print axioms FireEgress.FloatError.uDouble_pos
#print axioms FireEgress.FloatError.uDouble_le
#print axioms FireEgress.FloatError.errA_double
#print axioms FireEgress.FloatError.code_constant_double
#print axioms FireEgress.FloatError.crossDet_orient
#print axioms FireEgress.FloatError.jsOpposite_iff
#print axioms FireEgress.FloatError.jsOpposite_congr
#print axioms FireEgress.FloatError.floatCross_iff_of_sameSign
#print axioms FireEgress.FloatError.segmentsCross_float_exact
#print axioms FireEgress.FloatError.segmentsCross_float_geometric
#print axioms FireEgress.FloatError.segmentsCross_float_exact_code
#print axioms FireEgress.FloatError.orient_sq_le
#print axioms FireEgress.FloatError.abs_add_abs_sq_le
#print axioms FireEgress.FloatError.exists_on_line_sqd
#print axioms FireEgress.FloatError.orient_near_of_disagree
#print axioms FireEgress.FloatError.misclassified_near
#print axioms FireEgress.FloatError.kappa_double
#print axioms FireEgress.FloatError.flSum_err
#print axioms FireEgress.FloatError.pow_mul_le_one
#print axioms FireEgress.FloatError.pow_sub_one_le_gamma
#print axioms FireEgress.FloatError.route_sum_rel_err
#print axioms FireEgress.FloatError.gamma_10000_links
#print axioms FireEgress.FloatError.route_length_err
#print axioms FireEgress.FloatError.near_optimal_of_rel_err
#print axioms FireEgress.FloatError.sum_map_rel_err
#print axioms FireEgress.FloatError.sum_map_abs_err
#print axioms FireEgress.FloatError.perturbed_weights_near_optimal
#print axioms FireEgress.FloatError.perturbed_weights_near_optimal_abs
#print axioms FireEgress.FloatError.float_reduced_ge
#print axioms FireEgress.FloatError.init_invClip
#print axioms FireEgress.FloatError.selectable_le_clip
#print axioms FireEgress.FloatError.lenWalk_reduced_clip
#print axioms FireEgress.FloatError.max_zero_eq
#print axioms FireEgress.FloatError.incWalk_reduced_clip
#print axioms FireEgress.FloatError.step_invClip
#print axioms FireEgress.FloatError.reachable_invClip
#print axioms FireEgress.FloatError.dijkstra_clip
#print axioms FireEgress.FloatError.astar_delta_near_optimal
#print axioms FireEgress.FloatError.astar_le_cost_add_inconsistency
#print axioms FireEgress.FloatError.cx_pos
#print axioms FireEgress.FloatError.cx_consistent
#print axioms FireEgress.FloatError.cx_step
#print axioms FireEgress.FloatError.cx_selectable
#print axioms FireEgress.FloatError.cx_walk
#print axioms FireEgress.FloatError.astar_delta_counterexample
#print axioms FireEgress.FloatError.length_err_rounded_points
#print axioms FireEgress.FloatError.round_coords_norm
#print axioms FireEgress.FloatError.studio_float32_length_err
#print axioms FireEgress.FloatError.computed_length_err
