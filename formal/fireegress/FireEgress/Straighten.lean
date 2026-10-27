import Mathlib
import FireEgress.Heuristic

/-!
# Route straightening (`src/core/straighten.js`)

`straightenRoute` replaces a lattice route `x₀, …, xₙ` (node ids) by the nodes
at indices `0 = f 0 < f 1 < … < f m = n`: from each kept vertex it jumps to the
farthest later vertex in clear sight (`clear()`), or to the next vertex when
none is. The app claims two things:

* "the result is never longer than the lattice route", and
* shortcuts keep "a shoulder's width (0.25 m)" off the walls.

**Length.** The first claim is the generalised triangle inequality and holds in
any metric space (`subseq_length_le`, index form as the code builds it;
`pathLen_sublist_le`, list form). It reuses `FireEgress.Heuristic.admissible`
(the triangle inequality along a path) for each shortcut.

**Clearance.** `clear(a, b)` tests the wall distance only at the
`N = ⌈|ab| / (s/4)⌉` interior samples `lineMap a b (k/N)`, `0 < k < N`, so the
sample gap is `δ = |ab| / N ≤ s/4` (`sample_gap_le`). At sample `k` it requires
`min(c, ca + (cb - ca) k/N) - 1e-6`, where `ca`, `cb` are the endpoints' own
wall distances capped at `c`. What follows from that:

* in any metric space (distance to a set is 1-Lipschitz), every point of the
  shortcut keeps `m - δ/2`, `m = min(c, ca, cb) - 1e-6` (`shortcut_clearance`);
  with the code's numbers `c - s/8 = 0.1875 m` at `s = 0.5`;
* in the Euclidean plane (which the code works in) the sharper
  `√(m² - (δ/2)²)` holds for any obstacle set (`euclid_shortcut_clearance`),
  ≈ 0.2421 m at `s = 0.5` (`code_numbers_euclid`);
* the full `0.25 m` is *not* guaranteed: a wall corner midway between two
  samples that are exactly `0.25 m` off it is only `√(0.25² - 0.0625²) m` from
  the route (`full_clearance_not_guaranteed`, in `ℂ` as the plane).
* When the required clearance `m` is at most `δ/2` (the taper near a wall-face
  endpoint), the sampled test alone does not stop a wall thinner than the
  sample gap: an obstacle exactly between two samples `δ/2` away passes it and
  lies on the route (`thin_wall_not_excluded`). The code's exact `crosses()`
  test on the wall outline is what rules that out, not the sampling.
-/

namespace FireEgress.Straighten

open Finset

/-! ## Never longer than the lattice route -/

section Length

variable {P : Type*} [PseudoMetricSpace P]

/-- One shortcut is no longer than the stretch of route it replaces:
`|x_a x_b| ≤ ∑_{a ≤ j < b} |x_j x_{j+1}|`. This is
`FireEgress.Heuristic.admissible` applied to the route from `x_a` with edge
weights equal to the edge lengths. -/
theorem shortcut_le {Q : Type*} [MetricSpace Q] (x : ℕ → Q) {a b : ℕ} (hab : a ≤ b) :
    dist (x a) (x b) ≤ ∑ j ∈ Ico a b, dist (x j) (x (j + 1)) := by
  have h := FireEgress.Heuristic.admissible (x b) (fun i => x (a + i))
    (fun i => dist (x (a + i)) (x (a + i + 1))) (b - a) (fun i _ => le_rfl)
    (by rw [Nat.add_sub_cancel' hab])
  rw [Finset.sum_Ico_eq_sum_range]
  simpa [Nat.add_assoc] using h

/-- **Never longer (index form, as the code builds it).** If the kept vertices
are `x (f 0), …, x (f m)` with `f` monotone, `f 0 = 0` and `f m = n` (same
first and last node), the straightened polyline is no longer than the route
`x 0, …, x n`. Holds in any metric space; Studio's lengths are 3-D Euclidean
(`Math.hypot` over all three coordinates). -/
theorem subseq_length_le {Q : Type*} [MetricSpace Q] (x : ℕ → Q) (f : ℕ → ℕ) (m n : ℕ)
    (hf : Monotone f) (h0 : f 0 = 0) (hm : f m = n) :
    ∑ i ∈ range m, dist (x (f i)) (x (f (i + 1))) ≤ ∑ j ∈ range n, dist (x j) (x (j + 1)) := by
  have key : ∀ k, ∑ i ∈ range k, dist (x (f i)) (x (f (i + 1))) ≤
      ∑ j ∈ Ico (f 0) (f k), dist (x j) (x (j + 1)) := by
    intro k
    induction k with
    | zero => simp
    | succ k ih =>
      rw [Finset.sum_range_succ,
        ← Finset.sum_Ico_consecutive _ (hf (Nat.zero_le k)) (hf (Nat.le_succ k))]
      exact add_le_add ih (shortcut_le x (hf (Nat.le_succ k)))
  have := key m
  rwa [h0, hm, ← Finset.range_eq_Ico] at this

/-- Length of the polyline through a list of points. -/
noncomputable def pathLen : List P → ℝ
  | [] => 0
  | [_] => 0
  | a :: b :: t => dist a b + pathLen (b :: t)

theorem pathLen_nonneg : ∀ l : List P, 0 ≤ pathLen l
  | [] => le_rfl
  | [_] => le_rfl
  | _ :: b :: t => add_nonneg dist_nonneg (pathLen_nonneg (b :: t))

/-- Skipping one vertex `x` after `a` does not lengthen the path. -/
theorem pathLen_cons_le (a x : P) : ∀ l : List P, pathLen (a :: l) ≤ dist a x + pathLen (x :: l)
  | [] => by simp [pathLen]
  | b :: t => by
    simp only [pathLen]
    linarith [dist_triangle a x b]

/-- Helper: a sublist `l'` of `l` that (when nonempty) ends where `l` ends,
prefixed by the same vertex `a`, is no longer. -/
theorem pathLen_cons_sublist_le {l' l : List P} (h : l'.Sublist l) :
    ∀ a : P, (l' ≠ [] → l'.getLast? = l.getLast?) → pathLen (a :: l') ≤ pathLen (a :: l) := by
  induction h with
  | slnil => intro _ _; exact le_rfl
  | @cons l₁ l₂ x hs ih =>
    intro a hlast
    by_cases hne : l₁ = []
    · subst hne
      simp only [pathLen]
      exact add_nonneg dist_nonneg (pathLen_nonneg _)
    · have hl₂ : l₂ ≠ [] := by
        rintro rfl
        exact hne (List.sublist_nil.mp hs)
      have hl : l₁.getLast? = l₂.getLast? := by
        rw [hlast hne, List.getLast?_cons, List.getLast?_eq_some_getLast hl₂]
        rfl
      calc pathLen (a :: l₁) ≤ dist a x + pathLen (x :: l₁) := pathLen_cons_le a x l₁
        _ ≤ dist a x + pathLen (x :: l₂) := by linarith [ih x (fun _ => hl)]
        _ = pathLen (a :: x :: l₂) := rfl
  | @cons_cons l₁ l₂ x hs ih =>
    intro a hlast
    show dist a x + pathLen (x :: l₁) ≤ dist a x + pathLen (x :: l₂)
    by_cases hne : l₁ = []
    · subst hne
      simp only [pathLen]
      linarith [pathLen_nonneg (x :: l₂)]
    · have hl₂ : l₂ ≠ [] := by
        rintro rfl
        exact hne (List.sublist_nil.mp hs)
      have h1 := hlast (List.cons_ne_nil _ _)
      rw [List.getLast?_cons, List.getLast?_cons, List.getLast?_eq_some_getLast hne,
        List.getLast?_eq_some_getLast hl₂] at h1
      have hl : l₁.getLast? = l₂.getLast? := by
        rw [List.getLast?_eq_some_getLast hne, List.getLast?_eq_some_getLast hl₂]
        simpa using h1
      linarith [ih x (fun _ => hl)]

/-- **Never longer (list form).** In any (pseudo)metric space, the polyline
through a subsequence `s` of the points `l` with the same first and the same
last point is no longer than the polyline through `l`. -/
theorem pathLen_sublist_le {s l : List P} (h : s.Sublist l) (hhead : s.head? = l.head?)
    (hlast : s.getLast? = l.getLast?) : pathLen s ≤ pathLen l := by
  cases l with
  | nil => rw [List.sublist_nil.mp h]
  | cons a l₀ =>
    cases s with
    | nil => simp at hhead
    | cons b s₀ =>
      simp only [List.head?_cons, Option.some.injEq] at hhead
      subst hhead
      have hs : s₀.Sublist l₀ := List.cons_sublist_cons.mp h
      apply pathLen_cons_sublist_le hs
      intro hne
      have hl₀ : l₀ ≠ [] := by
        rintro rfl
        exact hne (List.sublist_nil.mp hs)
      rw [List.getLast?_cons, List.getLast?_cons, List.getLast?_eq_some_getLast hne,
        List.getLast?_eq_some_getLast hl₀] at hlast
      rw [List.getLast?_eq_some_getLast hne, List.getLast?_eq_some_getLast hl₀]
      simpa using hlast

end Length

/-! ## Clearance between samples -/

section Clearance

/-- **The honest clearance bound (any metric space).** If every point of a
segment `seg` lies within `δ/2` of some sample in `S`, and every sample keeps
at least `c` from the walls `O`, then every point of the segment keeps at least
`c - δ/2` (the distance to a set is 1-Lipschitz). -/
theorem clearance_of_samples {Q : Type*} [PseudoMetricSpace Q] (seg S O : Set Q) (c δ : ℝ)
    (hcover : ∀ x ∈ seg, ∃ p ∈ S, dist x p ≤ δ / 2)
    (hS : ∀ p ∈ S, c ≤ Metric.infDist p O) :
    ∀ x ∈ seg, c - δ / 2 ≤ Metric.infDist x O := by
  intro x hx
  obtain ⟨p, hp, hxp⟩ := hcover x hx
  have h1 := Metric.infDist_le_infDist_add_dist (s := O) (x := p) (y := x)
  have h2 := hS p hp
  rw [dist_comm] at hxp
  linarith

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- With samples `lineMap a b (k/N)`, `k = 0, …, N`, every point of the segment
`[a, b]` is within `|ab| / (2N)` of one (the nearest, `k = ⌊tN + 1/2⌋`). -/
theorem near_sample (a b : E) (N : ℕ) (hN : 0 < N) (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1) :
    ∃ k : ℕ, k ≤ N ∧
      dist (AffineMap.lineMap a b t) (AffineMap.lineMap a b ((k : ℝ) / N)) ≤ dist a b / (2 * N) := by
  have hNr : (0 : ℝ) < N := by exact_mod_cast hN
  set y : ℝ := t * N + 1 / 2 with hy
  have hy0 : 0 ≤ y := by positivity
  refine ⟨⌊y⌋₊, ?_, ?_⟩
  · have h3 : y < ((N + 1 : ℕ) : ℝ) := by
      have : t * N ≤ N := by nlinarith
      push_cast
      linarith
    have := (Nat.floor_lt hy0).mpr h3
    omega
  · rw [dist_lineMap_lineMap, Real.dist_eq]
    have hf1 := Nat.floor_le hy0
    have hf2 := Nat.lt_floor_add_one y
    have habs : |t - (⌊y⌋₊ : ℝ) / N| ≤ 1 / (2 * N) := by
      rw [abs_le]
      constructor
      · rw [show t - (⌊y⌋₊ : ℝ) / N = (t * N - ⌊y⌋₊) / N by field_simp]
        rw [show -(1 / (2 * (N : ℝ))) = (-(1 / 2)) / N by field_simp]
        apply div_le_div_of_nonneg_right _ hNr.le
        linarith
      · rw [show t - (⌊y⌋₊ : ℝ) / N = (t * N - ⌊y⌋₊) / N by field_simp]
        rw [show (1 / (2 * (N : ℝ))) = (1 / 2) / N by field_simp]
        apply div_le_div_of_nonneg_right _ hNr.le
        linarith
    calc |t - (⌊y⌋₊ : ℝ) / N| * dist a b ≤ 1 / (2 * N) * dist a b :=
          mul_le_mul_of_nonneg_right habs dist_nonneg
      _ = dist a b / (2 * N) := by ring

/-- The code's sample count `N = ⌈L / (s/4)⌉` gives a sample gap `L / N ≤ s/4`
(for `L > 0`, `s > 0`). -/
theorem sample_gap_le (L s : ℝ) (hL : 0 < L) (hs : 0 < s) :
    0 < ⌈L / (s / 4)⌉₊ ∧ L / ⌈L / (s / 4)⌉₊ ≤ s / 4 := by
  have hq : 0 < L / (s / 4) := by positivity
  have hc : L / (s / 4) ≤ ⌈L / (s / 4)⌉₊ := Nat.le_ceil _
  have hpos : (0 : ℝ) < ⌈L / (s / 4)⌉₊ := lt_of_lt_of_le hq hc
  refine ⟨by exact_mod_cast hpos, ?_⟩
  rw [div_le_iff₀ hpos]
  have := mul_le_mul_of_nonneg_left hc (by positivity : (0 : ℝ) ≤ s / 4)
  rw [mul_div_cancel₀ _ (by positivity : s / 4 ≠ 0)] at this
  linarith

/-- The clearance the code demands at a sample is at least `min(c, ca, cb) - ε`:
the taper `ca + (cb - ca) u`, `u ∈ [0, 1]`, never drops below `min ca cb`. -/
theorem need_ge (c ca cb ε u : ℝ) (hu0 : 0 ≤ u) (hu1 : u ≤ 1) :
    min c (min ca cb) - ε ≤ min c (ca + (cb - ca) * u) - ε := by
  have : min ca cb ≤ ca + (cb - ca) * u := by
    rcases le_total ca cb with h | h
    · rw [min_eq_left h]; nlinarith
    · rw [min_eq_right h]; nlinarith
  have := min_le_min_left c this
  linarith

/-- **What `clear()` guarantees, any metric (normed) space.** Suppose the
endpoints keep `ca`, `cb` off the walls `O` (the code's capped wall distances),
every interior sample `k = 1, …, N-1` keeps the code's required
`min(c, ca + (cb - ca) k/N) - ε` (`ε = 1e-6`), and the sample gap
`|ab| / N` is at most `δ`. Then every point of the shortcut keeps at least
`min(c, ca, cb) - ε - δ/2`. With both endpoints at least `c` off the walls this
is `c - ε - δ/2`. -/
theorem shortcut_clearance (a b : E) (O : Set E) (N : ℕ) (hN : 0 < N) (c ca cb ε δ : ℝ)
    (hε : 0 ≤ ε) (ha : ca ≤ Metric.infDist a O) (hb : cb ≤ Metric.infDist b O)
    (hk : ∀ k : ℕ, 0 < k → k < N →
      min c (ca + (cb - ca) * ((k : ℝ) / N)) - ε ≤
        Metric.infDist (AffineMap.lineMap a b ((k : ℝ) / N)) O)
    (hδ : dist a b / N ≤ δ) :
    ∀ t : ℝ, 0 ≤ t → t ≤ 1 →
      min c (min ca cb) - ε - δ / 2 ≤ Metric.infDist (AffineMap.lineMap a b t) O := by
  have hNr : (0 : ℝ) < N := by exact_mod_cast hN
  set S : Set E := {p | ∃ k : ℕ, k ≤ N ∧ p = AffineMap.lineMap a b ((k : ℝ) / N)}
  set seg : Set E := {x | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ x = AffineMap.lineMap a b t}
  have hcover : ∀ x ∈ seg, ∃ p ∈ S, dist x p ≤ δ / 2 := by
    rintro x ⟨t, ht0, ht1, rfl⟩
    obtain ⟨k, hkN, hd⟩ := near_sample a b N hN t ht0 ht1
    refine ⟨_, ⟨k, hkN, rfl⟩, hd.trans ?_⟩
    rw [show dist a b / (2 * N) = (dist a b / N) / 2 by ring]
    linarith
  have hS : ∀ p ∈ S, min c (min ca cb) - ε ≤ Metric.infDist p O := by
    rintro p ⟨k, hkN, rfl⟩
    rcases Nat.eq_zero_or_pos k with h0 | hpos
    · subst h0
      simp only [Nat.cast_zero, zero_div, AffineMap.lineMap_apply_zero]
      have : min c (min ca cb) ≤ ca := (min_le_right _ _).trans (min_le_left _ _)
      linarith
    · rcases lt_or_eq_of_le hkN with hlt | heq
      · have hu0 : 0 ≤ (k : ℝ) / N := by positivity
        have hu1 : (k : ℝ) / N ≤ 1 := by
          rw [div_le_one hNr]; exact_mod_cast hkN
        exact (need_ge c ca cb ε _ hu0 hu1).trans (hk k hpos hlt)
      · subst heq
        simp only [div_self hNr.ne', AffineMap.lineMap_apply_one]
        have : min c (min ca cb) ≤ cb := (min_le_right _ _).trans (min_le_right _ _)
        linarith
  intro t ht0 ht1
  exact clearance_of_samples seg S O _ δ hcover hS _ ⟨t, ht0, ht1, rfl⟩

/-- **The code's numbers (metric bound).** `δ = s/4`, so the guarantee is
`c - ε - s/8`; at `c = 0.25 m`, `s = 0.5 m`, `ε = 1e-6`: `0.1874990 m`, i.e.
`0.25 - 0.0625 = 0.1875 m` up to the `1e-6` slack. -/
theorem code_numbers_metric :
    (0.25 : ℝ) - 1e-6 - (0.5 / 4) / 2 = 0.187499 ∧ (0.25 : ℝ) - (0.5 / 4) / 2 = 0.1875 := by
  norm_num

end Clearance

/-! ## The sharper Euclidean bound -/

section Euclid

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- `|(1-u) v + u w|² = (1-u)|v|² + u|w|² - u(1-u)|v - w|²`. -/
theorem norm_sq_convex_comb (v w : E) (u : ℝ) :
    ‖(1 - u) • v + u • w‖ ^ 2 = (1 - u) * ‖v‖ ^ 2 + u * ‖w‖ ^ 2 - u * (1 - u) * ‖v - w‖ ^ 2 := by
  rw [@norm_add_sq_real, @norm_sub_sq_real, norm_smul, norm_smul, inner_smul_left,
    inner_smul_right]
  simp only [Real.norm_eq_abs, conj_trivial]
  rw [mul_pow, mul_pow, sq_abs, sq_abs]
  ring

/-- Parallelogram-type identity: for `x = (1-u) p + u q` and any `y`,
`|x - y|² = (1-u)|p - y|² + u|q - y|² - u(1-u)|p - q|²`. -/
theorem norm_sq_lineMap (p q y : E) (u : ℝ) :
    ‖AffineMap.lineMap p q u - y‖ ^ 2 =
      (1 - u) * ‖p - y‖ ^ 2 + u * ‖q - y‖ ^ 2 - u * (1 - u) * ‖p - q‖ ^ 2 := by
  rw [AffineMap.lineMap_apply_module]
  have e1 : (1 - u) • p + u • q - y = (1 - u) • (p - y) + u • (q - y) := by module
  have e2 : p - q = (p - y) - (q - y) := by abel
  rw [e1, e2, norm_sq_convex_comb]

/-- **Between two samples (Euclidean).** If two samples `p`, `q` at most `δ`
apart each keep at least `m ≥ 0` off an arbitrary obstacle set `O`, every point
between them keeps at least `√(m² - (δ/2)²)`. -/
theorem between_samples (p q : E) (O : Set E) (m δ u : ℝ) (hm : 0 ≤ m)
    (hpq : dist p q ≤ δ) (hu0 : 0 ≤ u) (hu1 : u ≤ 1)
    (hp : m ≤ Metric.infDist p O) (hq : m ≤ Metric.infDist q O) :
    Real.sqrt (m ^ 2 - (δ / 2) ^ 2) ≤ Metric.infDist (AffineMap.lineMap p q u) O := by
  rcases O.eq_empty_or_nonempty with hO | hO
  · subst hO
    simp only [Metric.infDist_empty] at hp ⊢
    have : m = 0 := le_antisymm hp hm
    subst this
    rw [Real.sqrt_eq_zero'.mpr (by nlinarith [sq_nonneg (δ / 2)])]
  rw [Metric.le_infDist hO]
  intro y hy
  rw [dist_eq_norm]
  apply Real.sqrt_le_iff.mpr
  refine ⟨norm_nonneg _, ?_⟩
  rw [norm_sq_lineMap]
  have hpy : m ≤ ‖p - y‖ := by
    rw [← dist_eq_norm]; exact hp.trans (Metric.infDist_le_dist_of_mem hy)
  have hqy : m ≤ ‖q - y‖ := by
    rw [← dist_eq_norm]; exact hq.trans (Metric.infDist_le_dist_of_mem hy)
  have hpq' : ‖p - q‖ ≤ δ := by rw [← dist_eq_norm]; exact hpq
  have hp2 : m ^ 2 ≤ ‖p - y‖ ^ 2 := pow_le_pow_left₀ hm hpy 2
  have hq2 : m ^ 2 ≤ ‖q - y‖ ^ 2 := pow_le_pow_left₀ hm hqy 2
  have hpq2 : ‖p - q‖ ^ 2 ≤ δ ^ 2 := by
    have := pow_le_pow_left₀ (norm_nonneg _) hpq' 2
    exact this
  have huu : 0 ≤ u * (1 - u) := mul_nonneg hu0 (by linarith)
  have huu' : u * (1 - u) ≤ 1 / 4 := by nlinarith [sq_nonneg (u - 1 / 2)]
  have hδ2 : 0 ≤ δ ^ 2 := sq_nonneg δ
  nlinarith [mul_le_mul_of_nonneg_left hpq2 huu, mul_le_mul_of_nonneg_left hp2 (by linarith : (0:ℝ) ≤ 1 - u),
    mul_le_mul_of_nonneg_left hq2 hu0, mul_le_mul_of_nonneg_right huu' hδ2]

/-- A point `lineMap a b t` lies between the consecutive samples `k/N` and
`(k+1)/N`, `k = min(⌊tN⌋, N-1)`. -/
theorem between_consecutive (a b : E) (N : ℕ) (hN : 0 < N) (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1) :
    ∃ k : ℕ, k < N ∧ ∃ u : ℝ, 0 ≤ u ∧ u ≤ 1 ∧
      AffineMap.lineMap a b t =
        AffineMap.lineMap (AffineMap.lineMap a b ((k : ℝ) / N))
          (AffineMap.lineMap a b (((k + 1 : ℕ) : ℝ) / N)) u := by
  have hNr : (0 : ℝ) < N := by exact_mod_cast hN
  have htN : 0 ≤ t * N := by positivity
  set k := min ⌊t * N⌋₊ (N - 1) with hk
  have hkN : k < N := by omega
  have hk1 : (k : ℝ) ≤ t * N := by
    have : (k : ℝ) ≤ ⌊t * N⌋₊ := by exact_mod_cast min_le_left _ _
    exact this.trans (Nat.floor_le htN)
  have hk2 : t * N ≤ k + 1 := by
    rcases le_total ⌊t * N⌋₊ (N - 1) with h | h
    · rw [hk, min_eq_left h]; exact (Nat.lt_floor_add_one _).le
    · rw [hk, min_eq_right h]
      have : ((N - 1 : ℕ) : ℝ) + 1 = N := by
        rw [Nat.cast_sub (by omega)]; simp
      rw [this]; nlinarith
  refine ⟨k, hkN, t * N - k, by linarith, by linarith, ?_⟩
  have hs : t = (t * N - k) * ((((k + 1 : ℕ) : ℝ)) / N - (k : ℝ) / N) + (k : ℝ) / N := by
    push_cast
    field_simp
    ring
  simp only [AffineMap.lineMap_apply_module']
  conv_lhs => rw [hs]
  module

/-- **What `clear()` guarantees in the plane (any inner product space, any
obstacle set).** Same hypotheses as `shortcut_clearance`, with `m = min(c, ca,
cb) - ε ≥ 0`: every point of the shortcut keeps at least `√(m² - (δ/2)²)`. -/
theorem euclid_shortcut_clearance (a b : E) (O : Set E) (N : ℕ) (hN : 0 < N) (c ca cb ε δ : ℝ)
    (hε : 0 ≤ ε) (hm : 0 ≤ min c (min ca cb) - ε)
    (ha : ca ≤ Metric.infDist a O) (hb : cb ≤ Metric.infDist b O)
    (hk : ∀ k : ℕ, 0 < k → k < N →
      min c (ca + (cb - ca) * ((k : ℝ) / N)) - ε ≤
        Metric.infDist (AffineMap.lineMap a b ((k : ℝ) / N)) O)
    (hδ : dist a b / N ≤ δ) :
    ∀ t : ℝ, 0 ≤ t → t ≤ 1 →
      Real.sqrt ((min c (min ca cb) - ε) ^ 2 - (δ / 2) ^ 2) ≤
        Metric.infDist (AffineMap.lineMap a b t) O := by
  have hNr : (0 : ℝ) < N := by exact_mod_cast hN
  set m := min c (min ca cb) - ε
  have hS : ∀ k : ℕ, k ≤ N → m ≤ Metric.infDist (AffineMap.lineMap a b ((k : ℝ) / N)) O := by
    intro k hkN
    rcases Nat.eq_zero_or_pos k with h0 | hpos
    · subst h0
      simp only [Nat.cast_zero, zero_div, AffineMap.lineMap_apply_zero]
      have : min c (min ca cb) ≤ ca := (min_le_right _ _).trans (min_le_left _ _)
      linarith
    · rcases lt_or_eq_of_le hkN with hlt | heq
      · have hu0 : 0 ≤ (k : ℝ) / N := by positivity
        have hu1 : (k : ℝ) / N ≤ 1 := by
          rw [div_le_one hNr]; exact_mod_cast hkN
        exact (need_ge c ca cb ε _ hu0 hu1).trans (hk k hpos hlt)
      · subst heq
        simp only [div_self hNr.ne', AffineMap.lineMap_apply_one]
        have : min c (min ca cb) ≤ cb := (min_le_right _ _).trans (min_le_right _ _)
        linarith
  intro t ht0 ht1
  obtain ⟨k, hkN, u, hu0, hu1, hx⟩ := between_consecutive a b N hN t ht0 ht1
  rw [hx]
  apply between_samples _ _ O m δ u hm _ hu0 hu1 (hS k hkN.le) (hS (k + 1) hkN)
  rw [dist_lineMap_lineMap, Real.dist_eq]
  have : |(k : ℝ) / N - ((k + 1 : ℕ) : ℝ) / N| = 1 / N := by
    push_cast
    rw [show (k : ℝ) / N - (k + 1) / N = -(1 / N) by field_simp; ring, abs_neg,
      abs_of_pos (by positivity)]
  rw [this, one_div_mul_eq_div]
  exact hδ

/-- **The code's numbers (Euclidean bound).** With `c = 0.25 m`, `ε = 1e-6`,
`δ = s/4 = 0.125 m`, the guaranteed clearance `√((c - ε)² - (δ/2)²)` lies
between `0.242 m` and `0.2421 m`: about 8 mm short of the claimed `0.25 m`. -/
theorem code_numbers_euclid :
    (0.242 : ℝ) < Real.sqrt ((0.25 - 1e-6) ^ 2 - (0.125 / 2) ^ 2) ∧
      Real.sqrt ((0.25 - 1e-6) ^ 2 - (0.125 / 2) ^ 2) < 0.2421 := by
  constructor
  · rw [Real.lt_sqrt (by norm_num)]; norm_num
  · rw [Real.sqrt_lt' (by norm_num)]; norm_num

end Euclid

/-! ## What the sampling does not guarantee -/

/-- **The full 0.25 m is not guaranteed.** In the plane (`ℂ`), two samples
`∓δ/2` (`δ = 0.125 m`, the code's gap at `s = 0.5`) each exactly `0.25 m` from
a wall corner `o = h i`, `h = √(0.25² - 0.0625²)`, pass the sampled test, yet
the midpoint of the route between them is only `h < 0.25 m` from that corner. -/
theorem full_clearance_not_guaranteed :
    let h : ℝ := Real.sqrt (0.25 ^ 2 - 0.0625 ^ 2)
    let o : ℂ := ⟨0, h⟩
    dist (⟨-0.0625, 0⟩ : ℂ) o = 0.25 ∧ dist (⟨0.0625, 0⟩ : ℂ) o = 0.25 ∧
      dist (AffineMap.lineMap (⟨-0.0625, 0⟩ : ℂ) ⟨0.0625, 0⟩ (1 / 2 : ℝ)) o < 0.25 := by
  intro h o
  have hh : h ^ 2 = 0.25 ^ 2 - 0.0625 ^ 2 := Real.sq_sqrt (by norm_num)
  have hh0 : 0 ≤ h := Real.sqrt_nonneg _
  have hmid : AffineMap.lineMap (⟨-0.0625, 0⟩ : ℂ) ⟨0.0625, 0⟩ (1 / 2 : ℝ) = 0 := by
    rw [AffineMap.lineMap_apply_module]
    apply Complex.ext <;> norm_num
  refine ⟨?_, ?_, ?_⟩
  · rw [Complex.dist_eq, Complex.norm_eq_sqrt_sq_add_sq]
    simp only [o, Complex.sub_re, Complex.sub_im]
    rw [Real.sqrt_eq_iff_mul_self_eq (by positivity) (by norm_num)]
    nlinarith
  · rw [Complex.dist_eq, Complex.norm_eq_sqrt_sq_add_sq]
    simp only [o, Complex.sub_re, Complex.sub_im]
    rw [Real.sqrt_eq_iff_mul_self_eq (by positivity) (by norm_num)]
    nlinarith
  · rw [hmid, Complex.dist_eq, Complex.norm_eq_sqrt_sq_add_sq]
    simp only [o, Complex.sub_re, Complex.sub_im, Complex.zero_re, Complex.zero_im]
    rw [Real.sqrt_lt' (by norm_num)]
    nlinarith

/-- **A wall thinner than the sample gap is not excluded by sampling.** When
the required clearance is `δ/2` or less (the taper near a wall-face endpoint),
an obstacle point exactly between two samples `δ` apart is `δ/2` from each,
so both samples pass, yet it lies on the route (distance `0`). Shown on the
real line with `δ = 0.125`. In Studio the exact `crosses()` test on the wall
outline, not the sampling, keeps a shortcut from passing through such a wall. -/
theorem thin_wall_not_excluded :
    Metric.infDist (-0.0625 : ℝ) {0} = 0.125 / 2 ∧ Metric.infDist (0.0625 : ℝ) {0} = 0.125 / 2 ∧
      Metric.infDist (AffineMap.lineMap (-0.0625 : ℝ) 0.0625 (1 / 2 : ℝ)) {0} = 0 := by
  simp only [Metric.infDist_singleton, AffineMap.lineMap_apply_module, Real.dist_eq, smul_eq_mul]
  norm_num

end FireEgress.Straighten

#print axioms FireEgress.Straighten.shortcut_le
#print axioms FireEgress.Straighten.subseq_length_le
#print axioms FireEgress.Straighten.pathLen_nonneg
#print axioms FireEgress.Straighten.pathLen_cons_le
#print axioms FireEgress.Straighten.pathLen_cons_sublist_le
#print axioms FireEgress.Straighten.pathLen_sublist_le
#print axioms FireEgress.Straighten.clearance_of_samples
#print axioms FireEgress.Straighten.near_sample
#print axioms FireEgress.Straighten.sample_gap_le
#print axioms FireEgress.Straighten.need_ge
#print axioms FireEgress.Straighten.shortcut_clearance
#print axioms FireEgress.Straighten.code_numbers_metric
#print axioms FireEgress.Straighten.norm_sq_convex_comb
#print axioms FireEgress.Straighten.norm_sq_lineMap
#print axioms FireEgress.Straighten.between_samples
#print axioms FireEgress.Straighten.between_consecutive
#print axioms FireEgress.Straighten.euclid_shortcut_clearance
#print axioms FireEgress.Straighten.code_numbers_euclid
#print axioms FireEgress.Straighten.full_clearance_not_guaranteed
#print axioms FireEgress.Straighten.thin_wall_not_excluded
