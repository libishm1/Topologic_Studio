import Mathlib

/-!
# The 8-connected lattice's geometry (Studio's navigation grid)

`buildLatticeGraph` (`src/core/lattice.js`) samples every walkable surface at
lattice positions `(i * s, j * s)` (in the frame turned to the building's
dominant wall direction) and links each floor node to its 4 axis neighbours and,
unless `rectilinear`, its 4 diagonal neighbours. A diagonal link is only made
when both axis neighbours it passes are walkable floor nodes too ("no corner
cutting", the `fullSteps` loop). Edge weights are Euclidean lengths, so an axis
link costs `s` and a diagonal `s * √2`. The route panel shows the grid route's
length ("On the grid") next to the straightened length.

Here the lattice is `ℤ × ℤ` with unit spacing (multiply every length by the
spacing `s`), a path is a list of steps, and

  `oct a b = max |a| |b| + (√2 - 1) * min |a| |b|`

is the octile distance. The file proves

1. some lattice path has length exactly `oct` (`octile_attained`);
2. no lattice path is shorter (`step_potential`, `octile_le_pathLen`);
3. `euclid ≤ oct ≤ √(4 - 2√2) * euclid ≈ 1.0824 * euclid`, the constant is
   attained in the 22.5° direction, and never at a lattice point
   (`euclid_le_oct`, `oct_le_K_mul_euclid`, `K_attained`,
   `oct_lt_K_mul_euclid_on_lattice`);
4. no corner cutting: a diagonal link on a walkable set `W` can always be
   replaced by two axis links inside `W`, so diagonals never connect what the
   4-connected grid on `W` does not, and every route on `W` is at least the
   octile distance (`diagonal_detour`, `axis_refinement`,
   `walls_never_shorter`).

Walls (a walkable set `W` smaller than the whole plane) can only remove links,
so they make the grid route longer than the octile distance, never shorter.
This is a statement about lattice links only: `buildLatticeGraph` also adds
off-lattice door links and wall-face nodes, which this model does not cover.
-/

namespace FireEgress.Lattice

open Real

/-! ## The octile distance -/

/-- The octile distance of a plan offset `(x, y)`, in lattice steps. -/
noncomputable def oct (x y : ℝ) : ℝ := max |x| |y| + (√2 - 1) * min |x| |y|

/-- The octile distance of an integer lattice offset. -/
noncomputable def octZ (v : ℤ × ℤ) : ℝ := oct (v.1 : ℝ) (v.2 : ℝ)

lemma sqrt2_sq : √2 ^ 2 = 2 := Real.sq_sqrt (by norm_num)

lemma one_lt_sqrt2 : 1 < √2 := by
  have h := sqrt2_sq
  have h0 : 0 ≤ √2 := Real.sqrt_nonneg _
  nlinarith

lemma sqrt2_lt_two : √2 < 2 := by
  have h := sqrt2_sq
  have h0 : 0 ≤ √2 := Real.sqrt_nonneg _
  nlinarith

/-- The octile distance is the larger of two weighted `ℓ¹` norms, because the
diagonal surcharge `√2 - 1` is below 1. -/
theorem oct_eq_max (x y : ℝ) :
    oct x y = max (|x| + (√2 - 1) * |y|) (|y| + (√2 - 1) * |x|) := by
  have h1 := one_lt_sqrt2
  have h2 := sqrt2_lt_two
  unfold oct
  rcases le_total |x| |y| with h | h
  · rw [max_eq_right h, min_eq_left h, max_eq_right]
    nlinarith [mul_nonneg (by linarith : (0 : ℝ) ≤ 2 - √2) (by linarith : 0 ≤ |y| - |x|)]
  · rw [max_eq_left h, min_eq_right h, max_eq_left]
    nlinarith [mul_nonneg (by linarith : (0 : ℝ) ≤ 2 - √2) (by linarith : 0 ≤ |x| - |y|)]

theorem oct_comm (x y : ℝ) : oct x y = oct y x := by
  unfold oct; rw [max_comm, min_comm]

theorem oct_nonneg (x y : ℝ) : 0 ≤ oct x y := by
  have h1 := one_lt_sqrt2
  unfold oct
  have : 0 ≤ min |x| |y| := le_min (abs_nonneg _) (abs_nonneg _)
  have : 0 ≤ max |x| |y| := le_trans (abs_nonneg x) (le_max_left _ _)
  nlinarith

/-- The octile distance is subadditive (a triangle inequality): the octile
distance of a sum of offsets is at most the sum of their octile distances. -/
theorem oct_add (x y x' y' : ℝ) : oct (x + x') (y + y') ≤ oct x y + oct x' y' := by
  have h1 := one_lt_sqrt2
  have hc : 0 ≤ √2 - 1 := by linarith
  rw [oct_eq_max, oct_eq_max x y, oct_eq_max x' y']
  have ax := abs_add_le x x'
  have ay := abs_add_le y y'
  apply max_le
  · have := mul_le_mul_of_nonneg_left ay hc
    have := le_max_left (|x| + (√2 - 1) * |y|) (|y| + (√2 - 1) * |x|)
    have := le_max_left (|x'| + (√2 - 1) * |y'|) (|y'| + (√2 - 1) * |x'|)
    linarith
  · have := mul_le_mul_of_nonneg_left ax hc
    have := le_max_right (|x| + (√2 - 1) * |y|) (|y| + (√2 - 1) * |x|)
    have := le_max_right (|x'| + (√2 - 1) * |y'|) (|y'| + (√2 - 1) * |x'|)
    linarith

/-! ## Lattice steps and paths -/

/-- One of the 8 lattice moves: each coordinate changes by at most one, and
something moves. -/
def IsStep (d : ℤ × ℤ) : Prop := |d.1| ≤ 1 ∧ |d.2| ≤ 1 ∧ d ≠ 0

/-- A diagonal move: both coordinates change. -/
def IsDiag (d : ℤ × ℤ) : Prop := d.1 ≠ 0 ∧ d.2 ≠ 0

/-- The length of a move, as `edgeLengths` measures it at unit spacing: 1 for
an axis move, `√2` for a diagonal. -/
noncomputable def stepLen (d : ℤ × ℤ) : ℝ := if d.1 = 0 ∨ d.2 = 0 then 1 else √2

/-- A path's length: the sum of its steps' lengths. -/
noncomputable def pathLen (p : List (ℤ × ℤ)) : ℝ := (p.map stepLen).sum

/-- Each move's length is its own octile distance (1 on an axis, `√2` on a
diagonal), so `oct` is the right yardstick for single links. -/
theorem stepLen_eq_octZ (d : ℤ × ℤ) (hd : IsStep d) : stepLen d = octZ d := by
  obtain ⟨x, y⟩ := d
  obtain ⟨hx, hy, hne⟩ := hd
  simp only at hx hy
  rw [abs_le] at hx hy
  have hx' : x = -1 ∨ x = 0 ∨ x = 1 := by omega
  have hy' : y = -1 ∨ y = 0 ∨ y = 1 := by omega
  have h1 := one_lt_sqrt2
  rcases hx' with rfl | rfl | rfl <;> rcases hy' with rfl | rfl | rfl <;>
    first
    | exact absurd rfl hne
    | (simp [stepLen, octZ, oct]; try ring_nf)

/-- (2) Potential argument. One step `d` from `u` lowers the octile
distance to the target `t` by at most the step's length. This is the
consistency condition that makes the octile distance a lower bound on any
route (cf. `FireEgress.Heuristic.consistent` for the Euclidean one). -/
theorem step_potential (t u d : ℤ × ℤ) (hd : IsStep d) :
    octZ (t - u) ≤ stepLen d + octZ (t - (u + d)) := by
  rw [stepLen_eq_octZ d hd]
  unfold octZ
  have := oct_add (d.1 : ℝ) (d.2 : ℝ) ((t - (u + d)).1 : ℝ) ((t - (u + d)).2 : ℝ)
  have e1 : ((t - u).1 : ℝ) = (d.1 : ℝ) + ((t - (u + d)).1 : ℝ) := by
    simp only [Prod.fst_sub, Prod.fst_add]; push_cast; ring
  have e2 : ((t - u).2 : ℝ) = (d.2 : ℝ) + ((t - (u + d)).2 : ℝ) := by
    simp only [Prod.snd_sub, Prod.snd_add]; push_cast; ring
  rw [e1, e2]
  exact this

/-- (2) No lattice path is shorter than the octile distance between its ends:
for any list of moves, the octile distance of its total displacement is at
most its length. -/
theorem octile_le_pathLen (p : List (ℤ × ℤ)) (hp : ∀ d ∈ p, IsStep d) :
    octZ p.sum ≤ pathLen p := by
  induction p with
  | nil => simp [octZ, oct, pathLen]
  | cons d p ih =>
    have hd := hp d (List.mem_cons_self ..)
    have ih' := ih (fun e he => hp e (List.mem_cons_of_mem _ he))
    have := step_potential (d + p.sum) 0 d hd
    simp only [zero_add, sub_zero, add_sub_cancel_left] at this
    simp only [List.sum_cons, pathLen, List.map_cons] at ih' ⊢
    linarith

/-- Swapping the two coordinates of every move. -/
lemma sum_map_swap (p : List (ℤ × ℤ)) : (p.map Prod.swap).sum = p.sum.swap := by
  induction p with
  | nil => rfl
  | cons d p ih => ext <;> simp [ih]

/-- The explicit shortest path when `|b| ≤ |a|`: `|b|` diagonal moves, then
`|a| - |b|` axis moves. -/
lemma build (a b : ℤ) (h : b.natAbs ≤ a.natAbs) :
    ∃ p : List (ℤ × ℤ), (∀ d ∈ p, IsStep d) ∧ p.sum = (a, b) ∧ pathLen p = octZ (a, b) := by
  refine ⟨List.replicate b.natAbs (a.sign, b.sign) ++
    List.replicate (a.natAbs - b.natAbs) (a.sign, 0), ?_, ?_, ?_⟩
  · intro d hd
    rcases List.mem_append.mp hd with hd | hd
    · obtain ⟨hn, rfl⟩ := List.mem_replicate.mp hd
      have hb : b ≠ 0 := by rintro rfl; simp at hn
      have ha : a ≠ 0 := by
        rintro rfl
        simp only [Int.natAbs_zero, nonpos_iff_eq_zero, Int.natAbs_eq_zero] at h
        exact hb h
      refine ⟨by rw [Int.abs_sign_of_ne_zero ha], by rw [Int.abs_sign_of_ne_zero hb], ?_⟩
      intro h0
      have := congrArg Prod.fst h0
      simp [Int.sign_eq_zero_iff_zero, ha] at this
    · obtain ⟨hn, rfl⟩ := List.mem_replicate.mp hd
      have ha : a ≠ 0 := by rintro rfl; simp at hn
      refine ⟨by rw [Int.abs_sign_of_ne_zero ha], by simp, ?_⟩
      intro h0
      have := congrArg Prod.fst h0
      simp [Int.sign_eq_zero_iff_zero, ha] at this
  · simp only [List.sum_append, List.sum_replicate, Prod.smul_mk, nsmul_eq_mul, mul_zero,
      Prod.mk_add_mk, add_zero]
    ext
    · simp only
      rw [Nat.cast_sub h]
      linear_combination Int.sign_mul_natAbs a
    · simp only
      linear_combination Int.sign_mul_natAbs b
  · have h1 := one_lt_sqrt2
    have hR : (b.natAbs : ℝ) ≤ a.natAbs := by exact_mod_cast h
    have ea : |(a : ℝ)| = (a.natAbs : ℝ) := by rw [Nat.cast_natAbs, Int.cast_abs]
    have eb : |(b : ℝ)| = (b.natAbs : ℝ) := by rw [Nat.cast_natAbs, Int.cast_abs]
    simp only [pathLen, List.map_append, List.map_replicate, List.sum_append,
      List.sum_replicate, nsmul_eq_mul, octZ, oct, ea, eb, max_eq_left hR, min_eq_right hR]
    rw [Nat.cast_sub h]
    have s0 : stepLen (a.sign, 0) = 1 := by simp [stepLen]
    rw [s0]
    by_cases hb : b = 0
    · subst hb; simp
    · have ha : a ≠ 0 := by
        rintro rfl
        simp only [Int.natAbs_zero, nonpos_iff_eq_zero, Int.natAbs_eq_zero] at h
        exact hb h
      have s1 : stepLen (a.sign, b.sign) = √2 := by
        simp [stepLen, Int.sign_eq_zero_iff_zero, ha, hb]
      rw [s1]
      ring

/-- (1) The octile distance is achieved: for every lattice offset `v` there is
a path of 8-connected moves from the origin to `v` whose length is exactly
`octZ v` (`min(|a|,|b|)` diagonals and `max - min` axis moves). -/
theorem octile_attained (v : ℤ × ℤ) :
    ∃ p : List (ℤ × ℤ), (∀ d ∈ p, IsStep d) ∧ p.sum = v ∧ pathLen p = octZ v := by
  obtain ⟨a, b⟩ := v
  rcases le_total b.natAbs a.natAbs with h | h
  · exact build a b h
  · obtain ⟨p, hs, hsum, hlen⟩ := build b a h
    refine ⟨p.map Prod.swap, ?_, ?_, ?_⟩
    · intro d hd
      obtain ⟨e, he, rfl⟩ := List.mem_map.mp hd
      obtain ⟨h1, h2, h3⟩ := hs e he
      refine ⟨h2, h1, ?_⟩
      intro h0; apply h3; ext
      · exact congrArg Prod.snd h0
      · exact congrArg Prod.fst h0
    · rw [sum_map_swap, hsum]; rfl
    · have : (p.map Prod.swap).map stepLen = p.map stepLen := by
        rw [List.map_map]; congr 1; funext d; simp [stepLen, or_comm]
      unfold pathLen at hlen ⊢
      rw [this, hlen]
      simp [octZ, oct_comm]

/-- (1)+(2) The shortest 8-connected lattice path to `v` has length exactly the
octile distance `octZ v`: it is the least length over all such paths. On the
open lattice of spacing `s`, Studio's grid route between two lattice nodes is
therefore `s * oct`. -/
theorem octile_is_shortest (v : ℤ × ℤ) :
    IsLeast {L : ℝ | ∃ p : List (ℤ × ℤ), (∀ d ∈ p, IsStep d) ∧ p.sum = v ∧ pathLen p = L}
      (octZ v) := by
  refine ⟨octile_attained v, ?_⟩
  rintro L ⟨p, hs, rfl, rfl⟩
  exact octile_le_pathLen p hs

/-! ## Octile against Euclidean -/

/-- The worst-case ratio of octile to Euclidean length, `√(4 - 2√2)`. -/
noncomputable def K : ℝ := √(4 - 2 * √2)

lemma sq_add_sq_max_min (x y : ℝ) :
    x ^ 2 + y ^ 2 = (max |x| |y|) ^ 2 + (min |x| |y|) ^ 2 := by
  rcases le_total |x| |y| with h | h
  · rw [max_eq_right h, min_eq_left h, sq_abs, sq_abs]; ring
  · rw [max_eq_left h, min_eq_right h, sq_abs, sq_abs]

/-- (3, lower) The octile distance is never shorter than the straight line:
`√(x² + y²) ≤ oct x y`. So the grid length is at least the Euclidean
distance between its ends. -/
theorem euclid_le_oct (x y : ℝ) : √(x ^ 2 + y ^ 2) ≤ oct x y := by
  have hs := sqrt2_sq
  have h1 := one_lt_sqrt2
  rw [sq_add_sq_max_min]
  set M := max |x| |y|
  set m := min |x| |y|
  have hm : 0 ≤ m := le_min (abs_nonneg _) (abs_nonneg _)
  have hmM : m ≤ M := min_le_max
  have key : M ^ 2 + m ^ 2 ≤ (oct x y) ^ 2 := by
    have e : (oct x y) ^ 2 - (M ^ 2 + m ^ 2) = 2 * (√2 - 1) * m * (M - m) := by
      unfold oct; linear_combination m ^ 2 * hs
    have : 0 ≤ 2 * (√2 - 1) * m * (M - m) :=
      mul_nonneg (mul_nonneg (by linarith) hm) (by linarith)
    linarith
  calc √(M ^ 2 + m ^ 2) ≤ √((oct x y) ^ 2) := Real.sqrt_le_sqrt key
    _ = oct x y := Real.sqrt_sq (oct_nonneg x y)

lemma K_sq_identity (x y : ℝ) :
    (4 - 2 * √2) * (x ^ 2 + y ^ 2) - (oct x y) ^ 2 =
      ((√2 - 1) * max |x| |y| - min |x| |y|) ^ 2 := by
  rw [sq_add_sq_max_min]
  unfold oct
  linear_combination (-(max |x| |y|) ^ 2 - (min |x| |y|) ^ 2) * sqrt2_sq

lemma four_sub_nonneg : 0 ≤ 4 - 2 * √2 := by linarith [sqrt2_lt_two]

/-- (3, upper) The octile distance is at most `K = √(4 - 2√2)` (about 1.0824)
times the straight line: `oct x y ≤ K * √(x² + y²)`. (Cauchy-Schwarz:
`K² = 1 + (√2 - 1)²`.) -/
theorem oct_le_K_mul_euclid (x y : ℝ) : oct x y ≤ K * √(x ^ 2 + y ^ 2) := by
  unfold K
  rw [← Real.sqrt_mul four_sub_nonneg]
  have h := K_sq_identity x y
  have : (oct x y) ^ 2 ≤ (4 - 2 * √2) * (x ^ 2 + y ^ 2) := by
    nlinarith [sq_nonneg ((√2 - 1) * max |x| |y| - min |x| |y|)]
  exact le_trans (le_abs_self _) (Real.abs_le_sqrt this)

/-- `tan 22.5° = √2 - 1`. -/
theorem tan_pi_div_eight : Real.tan (π / 8) = √2 - 1 := by
  have hs := sqrt2_sq
  have h1 := one_lt_sqrt2
  rw [Real.tan_eq_sin_div_cos, Real.sin_pi_div_eight, Real.cos_pi_div_eight]
  have e : (2 : ℝ) - √2 = (√2 - 1) ^ 2 * (2 + √2) := by
    linear_combination (-(√2)) * hs
  have hpos : 0 < √(2 + √2) := Real.sqrt_pos.mpr (by linarith)
  rw [e, Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq (by linarith)]
  field_simp

/-- (3, sharp) The constant `K` is attained in the 22.5° direction: for the
offset `(1, tan 22.5°)` the octile distance equals `K` times the Euclidean
length. So no smaller constant bounds the ratio for all directions. -/
theorem K_attained :
    oct 1 (Real.tan (π / 8)) = K * √(1 ^ 2 + (Real.tan (π / 8)) ^ 2) := by
  have hs := sqrt2_sq
  have h1 := one_lt_sqrt2
  have h2 := sqrt2_lt_two
  rw [tan_pi_div_eight]
  have e : (1 : ℝ) ^ 2 + (√2 - 1) ^ 2 = 4 - 2 * √2 := by linear_combination hs
  rw [e]
  unfold K
  rw [Real.mul_self_sqrt four_sub_nonneg]
  unfold oct
  rw [abs_one, abs_of_pos (by linarith : (0 : ℝ) < √2 - 1),
    max_eq_left (by linarith : √2 - 1 ≤ 1), min_eq_right (by linarith : √2 - 1 ≤ 1)]
  linear_combination hs

/-- The constant is about 1.0824: `1.0823 < K < 1.0824`. -/
theorem K_bounds : (1.0823 : ℝ) < K ∧ K < 1.0824 := by
  have hs := sqrt2_sq
  have h0 : 0 ≤ √2 := Real.sqrt_nonneg _
  have lo : (1.41421 : ℝ) < √2 := by nlinarith
  have hi : √2 < (1.41422 : ℝ) := by nlinarith
  unfold K
  constructor
  · rw [Real.lt_sqrt (by norm_num)]; nlinarith
  · rw [Real.sqrt_lt' (by norm_num)]; nlinarith

/-- (3) For any non-zero offset the ratio of octile to Euclidean length lies in
`[1, 1.0824)`: the range the property test checks for grid routes on an open
lattice. -/
theorem ratio_bounds (x y : ℝ) (h : (x, y) ≠ 0) :
    1 ≤ oct x y / √(x ^ 2 + y ^ 2) ∧ oct x y / √(x ^ 2 + y ^ 2) < 1.0824 := by
  have hpos : 0 < √(x ^ 2 + y ^ 2) := by
    apply Real.sqrt_pos.mpr
    by_contra hle
    have hx : x = 0 := by nlinarith [sq_nonneg x, sq_nonneg y]
    have hy : y = 0 := by nlinarith [sq_nonneg x, sq_nonneg y]
    exact h (by rw [hx, hy]; rfl)
  constructor
  · rw [le_div_iff₀ hpos, one_mul]; exact euclid_le_oct x y
  · rw [div_lt_iff₀ hpos]
    calc oct x y ≤ K * √(x ^ 2 + y ^ 2) := oct_le_K_mul_euclid x y
      _ < 1.0824 * √(x ^ 2 + y ^ 2) := mul_lt_mul_of_pos_right K_bounds.2 hpos

/-- `√2 - 1` is irrational, so it is never a ratio of integers. -/
lemma sqrt2_sub_one_mul_ne (m n : ℤ) (hn : n ≠ 0) : (√2 - 1) * n ≠ m := by
  intro h
  apply irrational_sqrt_two.ne_rat (((m + n : ℤ) : ℚ) / n)
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn
  push_cast
  field_simp
  linarith

/-- (3, lattice) At a lattice point the bound is strict: for every non-zero
integer offset `oct < K * euclid`. The 22.5° direction has irrational slope, so
grid routes between lattice nodes only approach the ratio 1.0824, never reach
it. -/
theorem oct_lt_K_mul_euclid_on_lattice (a b : ℤ) (h : (a, b) ≠ 0) :
    oct a b < K * √((a : ℝ) ^ 2 + (b : ℝ) ^ 2) := by
  unfold K
  rw [← Real.sqrt_mul four_sub_nonneg]
  have hid := K_sq_identity (a : ℝ) (b : ℝ)
  have hne : (√2 - 1) * max |(a : ℝ)| |(b : ℝ)| - min |(a : ℝ)| |(b : ℝ)| ≠ 0 := by
    have ea : |(a : ℝ)| = ((|a| : ℤ) : ℝ) := by push_cast; rfl
    have eb : |(b : ℝ)| = ((|b| : ℤ) : ℝ) := by push_cast; rfl
    rw [ea, eb]
    rcases le_total |a| |b| with hab | hab
    · have hR : ((|a| : ℤ) : ℝ) ≤ ((|b| : ℤ) : ℝ) := by exact_mod_cast hab
      rw [max_eq_right hR, min_eq_left hR, sub_ne_zero]
      have hb : |b| ≠ 0 := by
        intro hb0
        have hb' : b = 0 := abs_eq_zero.mp hb0
        have ha' : a = 0 := abs_eq_zero.mp (le_antisymm (hb0 ▸ hab) (abs_nonneg a))
        exact h (by rw [ha', hb']; rfl)
      exact sqrt2_sub_one_mul_ne _ _ hb
    · have hR : ((|b| : ℤ) : ℝ) ≤ ((|a| : ℤ) : ℝ) := by exact_mod_cast hab
      rw [max_eq_left hR, min_eq_right hR, sub_ne_zero]
      have ha : |a| ≠ 0 := by
        intro ha0
        have ha' : a = 0 := abs_eq_zero.mp ha0
        have hb' : b = 0 := abs_eq_zero.mp (le_antisymm (ha0 ▸ hab) (abs_nonneg b))
        exact h (by rw [ha', hb']; rfl)
      exact sqrt2_sub_one_mul_ne _ _ ha
  have hlt : (oct a b) ^ 2 < (4 - 2 * √2) * ((a : ℝ) ^ 2 + (b : ℝ) ^ 2) := by
    have := pow_pos (abs_pos.mpr hne) 2
    rw [sq_abs] at this
    linarith
  exact (Real.lt_sqrt (oct_nonneg _ _)).mpr hlt

/-! ## No corner cutting on a walkable set -/

/-- The floor-link rule of `buildLatticeGraph` on a set `W` of walkable lattice
cells: from `u` the move `d` is a link when `d` is one of the 8 moves, both
ends are in `W`, and, for a diagonal, both axis cells it passes
(`u + (d.1, 0)` and `u + (0, d.2)`) are in `W` too. -/
def Link (W : Set (ℤ × ℤ)) (u d : ℤ × ℤ) : Prop :=
  IsStep d ∧ u ∈ W ∧ u + d ∈ W ∧ (IsDiag d → u + (d.1, 0) ∈ W ∧ u + (0, d.2) ∈ W)

/-- A walk on `W` from `u`: a list of moves, each a link from where the walk
is. -/
def IsWalk (W : Set (ℤ × ℤ)) : ℤ × ℤ → List (ℤ × ℤ) → Prop
  | u, [] => u ∈ W
  | u, d :: p => Link W u d ∧ IsWalk W (u + d) p

/-- The cells a walk from `u` visits, in order. -/
def cells : ℤ × ℤ → List (ℤ × ℤ) → List (ℤ × ℤ)
  | u, [] => [u]
  | u, d :: p => u :: cells (u + d) p

/-- (4) A diagonal link sweeps a unit square all of whose four corners are
walkable. -/
theorem link_square_in_W (W : Set (ℤ × ℤ)) (u d : ℤ × ℤ) (h : Link W u d) (hd : IsDiag d) :
    u ∈ W ∧ u + (d.1, 0) ∈ W ∧ u + (0, d.2) ∈ W ∧ u + d ∈ W :=
  ⟨h.2.1, (h.2.2.2 hd).1, (h.2.2.2 hd).2, h.2.2.1⟩

lemma axis_link (W : Set (ℤ × ℤ)) (u d : ℤ × ℤ) (hs : |d.1| ≤ 1 ∧ |d.2| ≤ 1)
    (hz : d ≠ 0) (hnd : ¬ IsDiag d) (hu : u ∈ W) (hv : u + d ∈ W) : Link W u d :=
  ⟨⟨hs.1, hs.2, hz⟩, hu, hv, fun h => absurd h hnd⟩

/-- (4) No corner cutting. A diagonal link can be replaced by two axis links
inside `W`, in either order (`x` then `y`, or `y` then `x`), of total length 2. -/
theorem diagonal_detour (W : Set (ℤ × ℤ)) (u d : ℤ × ℤ) (h : Link W u d) (hd : IsDiag d) :
    Link W u (d.1, 0) ∧ Link W (u + (d.1, 0)) (0, d.2) ∧
    Link W u (0, d.2) ∧ Link W (u + (0, d.2)) (d.1, 0) ∧
    u + (d.1, 0) + (0, d.2) = u + d ∧ u + (0, d.2) + (d.1, 0) = u + d ∧
    stepLen (d.1, 0) + stepLen (0, d.2) = 2 := by
  obtain ⟨⟨hx, hy, _⟩, hu, hv, hc⟩ := h
  obtain ⟨hxW, hyW⟩ := hc hd
  have e1 : u + (d.1, 0) + (0, d.2) = u + d := by ext <;> simp
  have e2 : u + (0, d.2) + (d.1, 0) = u + d := by ext <;> simp
  have nz1 : ((d.1, 0) : ℤ × ℤ) ≠ 0 := fun h0 => hd.1 (congrArg Prod.fst h0)
  have nz2 : ((0, d.2) : ℤ × ℤ) ≠ 0 := fun h0 => hd.2 (congrArg Prod.snd h0)
  have nd1 : ¬ IsDiag ((d.1, 0) : ℤ × ℤ) := fun h0 => h0.2 rfl
  have nd2 : ¬ IsDiag ((0, d.2) : ℤ × ℤ) := fun h0 => h0.1 rfl
  refine ⟨axis_link W u _ ⟨hx, by simp⟩ nz1 nd1 hu hxW,
    axis_link W _ _ ⟨by simp, hy⟩ nz2 nd2 hxW (e1 ▸ hv),
    axis_link W u _ ⟨by simp, hy⟩ nz2 nd2 hu hyW,
    axis_link W _ _ ⟨hx, by simp⟩ nz1 nd1 hyW (e2 ▸ hv), e1, e2, ?_⟩
  simp [stepLen]; norm_num

/-- (4) Every cell a walk on `W` visits is in `W`. -/
theorem walk_cells_in_W (W : Set (ℤ × ℤ)) (u : ℤ × ℤ) (p : List (ℤ × ℤ))
    (h : IsWalk W u p) : ∀ c ∈ cells u p, c ∈ W := by
  induction p generalizing u with
  | nil => intro c hc; simp [cells] at hc; subst hc; exact h
  | cons d p ih =>
    intro c hc
    simp only [cells, List.mem_cons] at hc
    rcases hc with rfl | hc
    · exact h.1.2.1
    · exact ih (u + d) h.2 c hc

/-- (4) Diagonals never connect what the 4-connected grid on `W` does not:
every walk on `W` can be refined to a walk on `W` with axis moves only, with
the same displacement and at most `√2` times the length (each diagonal `√2`
becomes `2 = √2 · √2`). With `walk_cells_in_W`, the refined walk visits only
cells of `W`: a lattice route never needs a cell outside `W`. -/
theorem axis_refinement (W : Set (ℤ × ℤ)) (u : ℤ × ℤ) (p : List (ℤ × ℤ))
    (h : IsWalk W u p) :
    ∃ q : List (ℤ × ℤ), IsWalk W u q ∧ (∀ d ∈ q, ¬ IsDiag d) ∧ q.sum = p.sum ∧
      pathLen q ≤ √2 * pathLen p := by
  have h1 := one_lt_sqrt2
  have hs := sqrt2_sq
  induction p generalizing u with
  | nil => exact ⟨[], h, by simp, rfl, by simp [pathLen]⟩
  | cons d p ih =>
    obtain ⟨hl, hw⟩ := h
    obtain ⟨q, hq, hax, hsum, hlen⟩ := ih (u + d) hw
    by_cases hd : IsDiag d
    · obtain ⟨l1, l2, -, -, e1, -, hlen2⟩ := diagonal_detour W u d hl hd
      refine ⟨(d.1, 0) :: (0, d.2) :: q, ⟨l1, l2, e1 ▸ hq⟩, ?_, ?_, ?_⟩
      · intro e he
        simp only [List.mem_cons] at he
        rcases he with rfl | rfl | he
        · exact fun h0 => h0.2 rfl
        · exact fun h0 => h0.1 rfl
        · exact hax e he
      · simp only [List.sum_cons, hsum]
        rw [← add_assoc]
        congr 1
        ext <;> simp
      · have sd : stepLen d = √2 := by simp [stepLen, hd.1, hd.2]
        simp only [pathLen, List.map_cons, List.sum_cons] at hlen ⊢
        rw [sd]
        nlinarith
    · refine ⟨d :: q, ⟨hl, hq⟩, ?_, by simp [hsum], ?_⟩
      · intro e he
        simp only [List.mem_cons] at he
        rcases he with rfl | he
        · exact hd
        · exact hax e he
      · have sd : stepLen d = 1 := by
          unfold IsDiag at hd
          have hax' : d.1 = 0 ∨ d.2 = 0 := by tauto
          simp [stepLen, hax']
        simp only [pathLen, List.map_cons, List.sum_cons] at hlen ⊢
        rw [sd]
        nlinarith

/-- Walls make the grid route longer than the octile distance, never shorter:
every walk on any walkable set `W` (any lattice with cells removed) has length
at least the octile distance between its ends. Removing cells only removes
links. (Off-lattice links, such as `buildLatticeGraph`'s door links, are
outside this model.) -/
theorem walls_never_shorter (W : Set (ℤ × ℤ)) (u : ℤ × ℤ) (p : List (ℤ × ℤ))
    (h : IsWalk W u p) : octZ p.sum ≤ pathLen p := by
  apply octile_le_pathLen
  induction p generalizing u with
  | nil => simp
  | cons d p ih =>
    intro e he
    simp only [List.mem_cons] at he
    rcases he with rfl | he
    · exact h.1.1
    · exact ih (u + d) h.2 e he

/-- An L-shaped corner `W = {(0,0), (1,0), (1,1)}`: the diagonal from `(0,0)`
to `(1,1)` would cut the missing corner `(0,1)` and is refused, while the walk
round the corner (length 2, longer than the octile distance `√2`) is a walk on
`W`. -/
theorem corner_cut_refused :
    ¬ Link {(0, 0), (1, 0), (1, 1)} (0, 0) (1, 1) ∧
    IsWalk {(0, 0), (1, 0), (1, 1)} (0, 0) [(1, 0), (0, 1)] ∧
    pathLen [(1, 0), (0, 1)] = 2 ∧ octZ (1, 1) = √2 := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro h
    have := (h.2.2.2 ⟨one_ne_zero, one_ne_zero⟩).2
    simp at this
  · refine ⟨⟨⟨by simp, by simp, by simp⟩, by simp, by simp, fun h0 => absurd rfl h0.2⟩,
      ⟨⟨by simp, by simp, by simp⟩, by simp, by simp, fun h0 => absurd rfl h0.1⟩, ?_⟩
    simp [IsWalk]
  · simp [pathLen, stepLen]; norm_num
  · simp [octZ, oct]

end FireEgress.Lattice

#print axioms FireEgress.Lattice.oct_eq_max
#print axioms FireEgress.Lattice.oct_comm
#print axioms FireEgress.Lattice.oct_nonneg
#print axioms FireEgress.Lattice.oct_add
#print axioms FireEgress.Lattice.stepLen_eq_octZ
#print axioms FireEgress.Lattice.step_potential
#print axioms FireEgress.Lattice.octile_le_pathLen
#print axioms FireEgress.Lattice.octile_attained
#print axioms FireEgress.Lattice.octile_is_shortest
#print axioms FireEgress.Lattice.euclid_le_oct
#print axioms FireEgress.Lattice.oct_le_K_mul_euclid
#print axioms FireEgress.Lattice.tan_pi_div_eight
#print axioms FireEgress.Lattice.K_attained
#print axioms FireEgress.Lattice.K_bounds
#print axioms FireEgress.Lattice.ratio_bounds
#print axioms FireEgress.Lattice.oct_lt_K_mul_euclid_on_lattice
#print axioms FireEgress.Lattice.link_square_in_W
#print axioms FireEgress.Lattice.diagonal_detour
#print axioms FireEgress.Lattice.walk_cells_in_W
#print axioms FireEgress.Lattice.axis_refinement
#print axioms FireEgress.Lattice.walls_never_shorter
#print axioms FireEgress.Lattice.corner_cut_refused
