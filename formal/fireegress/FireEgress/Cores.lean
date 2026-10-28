import Mathlib

/-!
# Assumed stair lines in vertical cores, and exits per part of the graph

Two pieces of `topologicpy-web-frontend/src/core` that decide what an occupant can walk.

**`cores.js`, `walkingLine`.** A core stair with no modelled treads is walked along an
assumed dog-leg line from height `ha` to `hb`: `F ≥ 2` flights of riser `r` and going `g`
along the core's longer side, alternating between two lanes, each flight split into
`pieces ≥ 1` equal pieces; then the line is mirrored and turned into the lattice's axes, and
a point within `1e-6` m of the last one kept is dropped. The header says a route through the
core is "as long as the stair it stands for (about 1.9 times the rise, plus the landing)".
This file proves:

1. the flights' plan length is `(hb - ha) / r · g` (`flight_planLen`, `flight_span`,
   `linePts_planLen`), and their height variation is `hb - ha` (`linePts_riseLen`);
2. any polyline is at least `√(P² + V²)` long in 3D, `P` its plan length and `V` its total
   height variation (`len3_ge`, a Minkowski inequality, `hypot_add_le`), so at least
   `√(P² + (h_end - h_start)²)` (`len3_ge_ends`); monotone heights are not needed;
3. hence the code's line is at least `(hb - ha) √(g² + r²) / r` long (`walkingLine_length_ge`,
   via `stair_identity`), which for the defaults (175 mm, 280 mm) is between 1.88 and 1.89
   times the rise (`default_ratio`); the mirror and the axis swap keep every length
   (`planDist_outPt`, `dist3_outPt`, `polyLen_map`), and dropping a repeated point keeps it
   (`polyLen_destutter`);
4. heights along the line never decrease for `ha ≤ hb` (`flightPts_heights`,
   `linePts_heights`), also after the mirror, the swap and the de-duplication
   (`walkingLine_heights`).

**`ifcSemantics.js`, `exitsPerPart` and `exitDoors`.** Fallback doors are added as exits in
each part (connected component, walls on) of at least `max(50, 0.02 n)` nodes that holds no
flagged exit. Stated over a finite graph with a component labelling (`label u = label v`
iff `u` and `v` are joined by open links, as `componentLabels` computes): every part of at
least the threshold size that holds a fallback door ends with an exit in it, reachable from
every node of the part (`exitsPerPart_serves`, `exitNodes_serves`,
`large_part_reaches_exit`); a node is added only in a part of that size with no flagged
exit (`mem_exitsPerPart`).

## What is not covered

* Arithmetic is exact. In exact arithmetic consecutive points of the line are either a piece
  of a flight or the landing between the two lanes, so the code's `1e-6` test drops exactly
  the repeated points where the two lanes coincide (narrow cores), unless a flight's pieces
  are shorter than `1e-6` m; the lattice calls `walkingLine` only for rises of 1 to 8 m, but
  that no piece is that short is not proved, so the length bound is proved for the line
  before the de-duplication and for any de-duplication that drops only repeated points.
  The height property holds after any de-duplication.
* `flights`, `pieces`, the lanes and `start` are free parameters (any `F ≥ 1`,
  `pieces ≥ 1`, lanes and start); their formulas in the code are not used. The landing
  segments only add length; the lower bound ignores them.
* `componentLabels` is assumed correct (the hypothesis `hlabel`); its depth-first search is
  not verified. `exitDoors` is modelled by its node set; door records, names and
  de-duplication by node are not. With `graph = null` the code adds nothing, which the model
  does not cover.
-/

namespace FireEgress.Cores

open Real

/-! ## Polylines -/

/-- A point of the walking line in the core's frame: along the core, across it, height (m).
After `push`'s axis swap the first two are the lattice's `u, v`. -/
abbrev Pt := ℝ × ℝ × ℝ

/-- Plan distance (the first two coordinates). -/
noncomputable def planDist (p q : Pt) : ℝ := √((q.1 - p.1) ^ 2 + (q.2.1 - p.2.1) ^ 2)

/-- Height difference. -/
noncomputable def riseDist (p q : Pt) : ℝ := |q.2.2 - p.2.2|

/-- 3D distance (`Math.hypot` of the three differences, as a lattice link's length). -/
noncomputable def dist3 (p q : Pt) : ℝ :=
  √((q.1 - p.1) ^ 2 + (q.2.1 - p.2.1) ^ 2 + (q.2.2 - p.2.2) ^ 2)

/-- The length of a polyline under a segment length `f`: the sum over consecutive points. -/
noncomputable def polyLen (f : Pt → Pt → ℝ) : List Pt → ℝ
  | [] => 0
  | [_] => 0
  | p :: q :: rest => f p q + polyLen f (q :: rest)

theorem planDist_nonneg (p q : Pt) : 0 ≤ planDist p q := Real.sqrt_nonneg _

theorem riseDist_nonneg (p q : Pt) : 0 ≤ riseDist p q := abs_nonneg _

theorem dist3_eq (p q : Pt) : dist3 p q = √(planDist p q ^ 2 + riseDist p q ^ 2) := by
  unfold dist3 planDist riseDist
  rw [Real.sq_sqrt (by positivity), sq_abs]

/-- Minkowski's inequality in the plane: `‖u + w‖ ≤ ‖u‖ + ‖w‖` for `u = (a, b)`,
`w = (x, y)`. -/
theorem hypot_add_le (a b x y : ℝ) :
    √((a + x) ^ 2 + (b + y) ^ 2) ≤ √(a ^ 2 + b ^ 2) + √(x ^ 2 + y ^ 2) := by
  have hA := Real.sqrt_nonneg (a ^ 2 + b ^ 2)
  have hX := Real.sqrt_nonneg (x ^ 2 + y ^ 2)
  have hA2 := Real.sq_sqrt (by positivity : (0 : ℝ) ≤ a ^ 2 + b ^ 2)
  have hX2 := Real.sq_sqrt (by positivity : (0 : ℝ) ≤ x ^ 2 + y ^ 2)
  have hcs2 : (a * x + b * y) ^ 2 ≤ (√(a ^ 2 + b ^ 2) * √(x ^ 2 + y ^ 2)) ^ 2 := by
    rw [mul_pow, hA2, hX2]
    nlinarith [sq_nonneg (a * y - b * x)]
  have hcs : a * x + b * y ≤ √(a ^ 2 + b ^ 2) * √(x ^ 2 + y ^ 2) := by
    have h0 := mul_nonneg hA hX
    nlinarith
  rw [Real.sqrt_le_iff]
  refine ⟨add_nonneg hA hX, ?_⟩
  nlinarith

/-- **A polyline is at least `√(P² + V²)` long**, `P` its plan length and `V` its total height
variation. -/
theorem len3_ge : ∀ l : List Pt,
    √(polyLen planDist l ^ 2 + polyLen riseDist l ^ 2) ≤ polyLen dist3 l
  | [] => by simp [polyLen]
  | [_] => by simp [polyLen]
  | p :: q :: rest => by
    have ih := len3_ge (q :: rest)
    show √((planDist p q + polyLen planDist (q :: rest)) ^ 2 +
        (riseDist p q + polyLen riseDist (q :: rest)) ^ 2) ≤ dist3 p q + polyLen dist3 (q :: rest)
    calc √((planDist p q + polyLen planDist (q :: rest)) ^ 2 +
          (riseDist p q + polyLen riseDist (q :: rest)) ^ 2)
        ≤ √(planDist p q ^ 2 + riseDist p q ^ 2) +
            √(polyLen planDist (q :: rest) ^ 2 + polyLen riseDist (q :: rest) ^ 2) :=
          hypot_add_le _ _ _ _
      _ ≤ dist3 p q + polyLen dist3 (q :: rest) := by
          rw [dist3_eq]
          linarith

/-- The total height variation is at least the net rise from the first point to the last. -/
theorem riseDist_ge_ends : ∀ (p : Pt) (rest : List Pt),
    |((p :: rest).getLast (List.cons_ne_nil p rest)).2.2 - p.2.2| ≤ polyLen riseDist (p :: rest)
  | p, [] => by simp [polyLen]
  | p, q :: rest => by
    have ih := riseDist_ge_ends q rest
    rw [List.getLast_cons_cons]
    show _ ≤ riseDist p q + polyLen riseDist (q :: rest)
    calc |((q :: rest).getLast (List.cons_ne_nil q rest)).2.2 - p.2.2|
        ≤ |((q :: rest).getLast (List.cons_ne_nil q rest)).2.2 - q.2.2| + |q.2.2 - p.2.2| :=
          abs_sub_le _ _ _
      _ ≤ polyLen riseDist (q :: rest) + riseDist p q := by
          rw [show riseDist p q = |q.2.2 - p.2.2| from rfl]
          linarith
      _ = riseDist p q + polyLen riseDist (q :: rest) := add_comm _ _

/-- **A line rising by `Δh` over a plan length `P` is at least `√(P² + Δh²)` long** (in 3D),
whatever happens in between (monotone heights are not needed). -/
theorem len3_ge_ends (p : Pt) (rest : List Pt) :
    √(polyLen planDist (p :: rest) ^ 2 +
        (((p :: rest).getLast (List.cons_ne_nil p rest)).2.2 - p.2.2) ^ 2) ≤
      polyLen dist3 (p :: rest) := by
  refine le_trans (Real.sqrt_le_sqrt ?_) (len3_ge (p :: rest))
  have h := riseDist_ge_ends p rest
  rw [← sq_abs (((p :: rest).getLast (List.cons_ne_nil p rest)).2.2 - p.2.2)]
  have := pow_le_pow_left₀ (abs_nonneg _) h 2
  linarith

theorem polyLen_nonneg {f : Pt → Pt → ℝ} (hf : ∀ p q, 0 ≤ f p q) : ∀ l, 0 ≤ polyLen f l
  | [] => le_rfl
  | [_] => le_rfl
  | p :: q :: rest => add_nonneg (hf p q) (polyLen_nonneg hf (q :: rest))

/-- Joining two polylines adds a segment of non-negative length. -/
theorem polyLen_append_ge {f : Pt → Pt → ℝ} (hf : ∀ p q, 0 ≤ f p q) :
    ∀ l₁ l₂ : List Pt, polyLen f l₁ + polyLen f l₂ ≤ polyLen f (l₁ ++ l₂)
  | [], l₂ => by simp [polyLen]
  | [_], [] => by simp [polyLen]
  | [p], q :: l₂ => by
    show 0 + polyLen f (q :: l₂) ≤ f p q + polyLen f (q :: l₂)
    linarith [hf p q]
  | p :: q :: rest, l₂ => by
    have ih := polyLen_append_ge hf (q :: rest) l₂
    rw [List.cons_append] at ih
    show f p q + polyLen f (q :: rest) + polyLen f l₂ ≤ f p q + polyLen f (q :: (rest ++ l₂))
    linarith

/-- The length of concatenated pieces is at least the sum of their lengths. -/
theorem polyLen_flatMap_ge {f : Pt → Pt → ℝ} (hf : ∀ p q, 0 ≤ f p q) (g : ℕ → List Pt) :
    ∀ F : ℕ, ∑ k ∈ Finset.range F, polyLen f (g k) ≤ polyLen f ((List.range F).flatMap g)
  | 0 => by simp [polyLen]
  | F + 1 => by
    rw [Finset.sum_range_succ, List.range_succ, List.flatMap_append]
    have ih := polyLen_flatMap_ge hf g F
    have h2 := polyLen_append_ge hf ((List.range F).flatMap g) ([F].flatMap g)
    have h3 : [F].flatMap g = g F := by simp
    rw [h3] at h2 ⊢
    linarith

/-- A polyline of `n + 1` points `P 0, …, P n`. -/
theorem polyLen_map_range (f : Pt → Pt → ℝ) :
    ∀ (n : ℕ) (P : ℕ → Pt),
      polyLen f ((List.range (n + 1)).map P) = ∑ i ∈ Finset.range n, f (P i) (P (i + 1))
  | 0, P => by simp [polyLen]
  | n + 1, P => by
    have e : (List.range (n + 1 + 1)).map P = P 0 :: (List.range (n + 1)).map (fun i => P (i + 1)) := by
      rw [List.range_succ_eq_map, List.map_cons, List.map_map]; rfl
    have e2 : (List.range (n + 1)).map (fun i => P (i + 1)) =
        P 1 :: (List.range n).map (fun i => P (i + 1 + 1)) := by
      rw [List.range_succ_eq_map, List.map_cons, List.map_map]; rfl
    have ih := polyLen_map_range f n (fun i => P (i + 1))
    rw [e, e2]
    show f (P 0) (P 1) + polyLen f (P 1 :: (List.range n).map (fun i => P (i + 1 + 1))) = _
    rw [← e2, ih, Finset.sum_range_succ']
    simp only [zero_add]
    ring

/-- Lengths are kept by any map that keeps segment lengths. -/
theorem polyLen_map {f : Pt → Pt → ℝ} {T : Pt → Pt} (hT : ∀ p q, f (T p) (T q) = f p q) :
    ∀ l, polyLen f (l.map T) = polyLen f l
  | [] => rfl
  | [_] => rfl
  | p :: q :: rest => by
    show f (T p) (T q) + polyLen f ((q :: rest).map T) = f p q + polyLen f (q :: rest)
    rw [hT, polyLen_map hT (q :: rest)]

/-! ## Repeated points -/

theorem destutter'_eq_cons (R : Pt → Pt → Prop) [DecidableRel R] :
    ∀ (l : List Pt) (a : Pt), ∃ t, List.destutter' R a l = a :: t
  | [], a => ⟨[], by rw [List.destutter'_nil]⟩
  | b :: l, a => by
    rw [List.destutter'_cons]
    split_ifs
    · exact ⟨_, rfl⟩
    · exact destutter'_eq_cons R l a

theorem polyLen_destutter' {f : Pt → Pt → ℝ} (hf : ∀ p, f p p = 0) :
    ∀ (l : List Pt) (a : Pt), polyLen f (List.destutter' (· ≠ ·) a l) = polyLen f (a :: l)
  | [], a => by rw [List.destutter'_nil]
  | b :: l, a => by
    rw [List.destutter'_cons]
    split_ifs with hab
    · obtain ⟨t, ht⟩ := destutter'_eq_cons (· ≠ ·) l b
      have ih := polyLen_destutter' hf l b
      rw [ht] at ih ⊢
      show f a b + polyLen f (b :: t) = f a b + polyLen f (b :: l)
      rw [ih]
    · have hab' : a = b := not_not.1 hab
      subst hab'
      rw [polyLen_destutter' hf l a]
      show polyLen f (a :: l) = f a a + polyLen f (a :: l)
      rw [hf, zero_add]

/-- **Dropping repeated points keeps the length** (where the two lanes coincide, the first
point of a flight repeats the last of the one before). -/
theorem polyLen_destutter {f : Pt → Pt → ℝ} (hf : ∀ p, f p p = 0) (l : List Pt) :
    polyLen f (l.destutter (· ≠ ·)) = polyLen f l := by
  cases l with
  | nil => rfl
  | cons a l => exact polyLen_destutter' hf l a

/-! ## The walking line of `walkingLine` -/

/-- `total = (hb - ha) / riser · going`: the stair's going in total. -/
noncomputable def stairTotal (ha hb riser going : ℝ) : ℝ := (hb - ha) / riser * going

/-- Where flight `k` starts along the core: `start` for even `k`, `start + run` for odd. -/
noncomputable def flightFrom (start run : ℝ) (k : ℕ) : ℝ := if k % 2 = 0 then start else start + run

/-- Where flight `k` ends along the core. -/
noncomputable def flightTo (start run : ℝ) (k : ℕ) : ℝ := if k % 2 = 0 then start + run else start

/-- `hA` of flight `k`: `ha + (hb - ha) k / F` (its `hB` is `hAt (k + 1)`). -/
noncomputable def hAt (ha hb : ℝ) (F k : ℕ) : ℝ := ha + (hb - ha) * k / F

/-- The points `push` receives for one flight: `(from, lane, hA)` and then, for
`i = 1, …, N`, `(from + (to - from) i / N, lane, hA + (hB - hA) i / N)`. -/
noncomputable def flightPts (fr dst lane hA hB : ℝ) (N : ℕ) : List Pt :=
  (List.range (N + 1)).map fun i : ℕ =>
    (fr + (dst - fr) * (i : ℝ) / N, lane, hA + (hB - hA) * (i : ℝ) / N)

/-- All the points `walkingLine` pushes, in the core's frame (before the mirror, the axis
swap and the de-duplication): `F` flights of `run = total / F`, flight `k` on lane `lane k`
in `pieces k` pieces. -/
noncomputable def linePts (ha hb riser going start : ℝ) (lane : ℕ → ℝ) (F : ℕ)
    (pieces : ℕ → ℕ) : List Pt :=
  (List.range F).flatMap fun k =>
    flightPts (flightFrom start (stairTotal ha hb riser going / F) k)
      (flightTo start (stairTotal ha hb riser going / F) k) (lane k) (hAt ha hb F k)
      (hAt ha hb F (k + 1)) (pieces k)

theorem planDist_piece (fr dst lane hA hB : ℝ) {N : ℕ} (hN : 0 < N) (i : ℕ) :
    planDist (fr + (dst - fr) * i / N, lane, hA + (hB - hA) * i / N)
      (fr + (dst - fr) * ((i + 1 : ℕ) : ℝ) / N, lane, hA + (hB - hA) * ((i + 1 : ℕ) : ℝ) / N) =
      |dst - fr| / N := by
  have hN' : (0 : ℝ) < N := by exact_mod_cast hN
  show √((fr + (dst - fr) * ((i + 1 : ℕ) : ℝ) / N - (fr + (dst - fr) * i / N)) ^ 2 +
    (lane - lane) ^ 2) = |dst - fr| / N
  have h1 : fr + (dst - fr) * ((i + 1 : ℕ) : ℝ) / N - (fr + (dst - fr) * i / N) = (dst - fr) / N := by
    push_cast
    ring
  rw [h1, sub_self, show ((0 : ℝ)) ^ 2 = 0 by norm_num, add_zero, Real.sqrt_sq_eq_abs, abs_div,
    abs_of_pos hN']

theorem riseDist_piece (fr dst lane hA hB : ℝ) {N : ℕ} (hN : 0 < N) (i : ℕ) :
    riseDist (fr + (dst - fr) * i / N, lane, hA + (hB - hA) * i / N)
      (fr + (dst - fr) * ((i + 1 : ℕ) : ℝ) / N, lane, hA + (hB - hA) * ((i + 1 : ℕ) : ℝ) / N) =
      |hB - hA| / N := by
  have hN' : (0 : ℝ) < N := by exact_mod_cast hN
  show |hA + (hB - hA) * ((i + 1 : ℕ) : ℝ) / N - (hA + (hB - hA) * i / N)| = |hB - hA| / N
  have h1 : hA + (hB - hA) * ((i + 1 : ℕ) : ℝ) / N - (hA + (hB - hA) * i / N) = (hB - hA) / N := by
    push_cast
    ring
  rw [h1, abs_div, abs_of_pos hN']

/-- **A flight's plan length is its run**: its `N` pieces are collinear and equal. -/
theorem flight_planLen {N : ℕ} (hN : 0 < N) (fr dst lane hA hB : ℝ) :
    polyLen planDist (flightPts fr dst lane hA hB N) = |dst - fr| := by
  have hN' : (0 : ℝ) < N := by exact_mod_cast hN
  calc polyLen planDist (flightPts fr dst lane hA hB N)
      = ∑ i ∈ Finset.range N, |dst - fr| / N := by
        unfold flightPts
        rw [polyLen_map_range]
        exact Finset.sum_congr rfl fun i _ => planDist_piece fr dst lane hA hB hN i
    _ = |dst - fr| := by
        rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
        field_simp

/-- A flight rises by `hB - hA`. -/
theorem flight_riseLen {N : ℕ} (hN : 0 < N) (fr dst lane hA hB : ℝ) :
    polyLen riseDist (flightPts fr dst lane hA hB N) = |hB - hA| := by
  have hN' : (0 : ℝ) < N := by exact_mod_cast hN
  calc polyLen riseDist (flightPts fr dst lane hA hB N)
      = ∑ i ∈ Finset.range N, |hB - hA| / N := by
        unfold flightPts
        rw [polyLen_map_range]
        exact Finset.sum_congr rfl fun i _ => riseDist_piece fr dst lane hA hB hN i
    _ = |hB - hA| := by
        rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
        field_simp

/-- Every flight spans `run` along the core, whichever way it goes. -/
theorem flight_span (start run : ℝ) (k : ℕ) :
    |flightTo start run k - flightFrom start run k| = |run| := by
  unfold flightTo flightFrom
  split_ifs
  · congr 1
    ring
  · rw [show start - (start + run) = -run by ring, abs_neg]

theorem hAt_step (ha hb : ℝ) {F : ℕ} (hF : 0 < F) (k : ℕ) :
    hAt ha hb F (k + 1) - hAt ha hb F k = (hb - ha) / F := by
  have hF' : (0 : ℝ) < F := by exact_mod_cast hF
  unfold hAt
  push_cast
  ring

/-- **The flights' total plan length is `(hb - ha) / r · g`** (for `ha ≤ hb`, `r > 0`,
`g ≥ 0`, `|total| = total`): `F` flights of `total / F` each, and the landings only add. -/
theorem linePts_planLen (ha hb riser going start : ℝ) (lane : ℕ → ℝ) {F : ℕ} (hF : 0 < F)
    {pieces : ℕ → ℕ} (hp : ∀ k, 0 < pieces k) :
    |stairTotal ha hb riser going| ≤
      polyLen planDist (linePts ha hb riser going start lane F pieces) := by
  have hF' : (0 : ℝ) < F := by exact_mod_cast hF
  have h1 : ∀ k ∈ Finset.range F,
      polyLen planDist (flightPts (flightFrom start (stairTotal ha hb riser going / F) k)
        (flightTo start (stairTotal ha hb riser going / F) k) (lane k) (hAt ha hb F k)
        (hAt ha hb F (k + 1)) (pieces k)) = |stairTotal ha hb riser going / F| :=
    fun k _ => by rw [flight_planLen (hp k), flight_span]
  have h2 : ∑ _k ∈ Finset.range F, |stairTotal ha hb riser going / F| =
      |stairTotal ha hb riser going| := by
    rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul, abs_div, Nat.abs_cast]
    field_simp
  calc |stairTotal ha hb riser going|
      = ∑ k ∈ Finset.range F,
          polyLen planDist (flightPts (flightFrom start (stairTotal ha hb riser going / F) k)
            (flightTo start (stairTotal ha hb riser going / F) k) (lane k) (hAt ha hb F k)
            (hAt ha hb F (k + 1)) (pieces k)) := by rw [Finset.sum_congr rfl h1, h2]
    _ ≤ polyLen planDist (linePts ha hb riser going start lane F pieces) :=
        polyLen_flatMap_ge planDist_nonneg _ F

/-- The flights rise by `hb - ha` in total. -/
theorem linePts_riseLen (ha hb riser going start : ℝ) (lane : ℕ → ℝ) {F : ℕ} (hF : 0 < F)
    {pieces : ℕ → ℕ} (hp : ∀ k, 0 < pieces k) :
    |hb - ha| ≤ polyLen riseDist (linePts ha hb riser going start lane F pieces) := by
  have hF' : (0 : ℝ) < F := by exact_mod_cast hF
  have h1 : ∀ k ∈ Finset.range F,
      polyLen riseDist (flightPts (flightFrom start (stairTotal ha hb riser going / F) k)
        (flightTo start (stairTotal ha hb riser going / F) k) (lane k) (hAt ha hb F k)
        (hAt ha hb F (k + 1)) (pieces k)) = |(hb - ha) / F| :=
    fun k _ => by rw [flight_riseLen (hp k), hAt_step ha hb hF]
  have h2 : ∑ _k ∈ Finset.range F, |(hb - ha) / F| = |hb - ha| := by
    rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul, abs_div, Nat.abs_cast]
    field_simp
  calc |hb - ha|
      = ∑ k ∈ Finset.range F,
          polyLen riseDist (flightPts (flightFrom start (stairTotal ha hb riser going / F) k)
            (flightTo start (stairTotal ha hb riser going / F) k) (lane k) (hAt ha hb F k)
            (hAt ha hb F (k + 1)) (pieces k)) := by rw [Finset.sum_congr rfl h1, h2]
    _ ≤ polyLen riseDist (linePts ha hb riser going start lane F pieces) :=
        polyLen_flatMap_ge riseDist_nonneg _ F

/-- `√(total² + (hb - ha)²) = (hb - ha) √(g² + r²) / r`. -/
theorem stair_identity {ha hb r g : ℝ} (hab : ha ≤ hb) (hr : 0 < r) :
    √(stairTotal ha hb r g ^ 2 + (hb - ha) ^ 2) = (hb - ha) * √(g ^ 2 + r ^ 2) / r := by
  unfold stairTotal
  have e : ((hb - ha) / r * g) ^ 2 + (hb - ha) ^ 2 = ((hb - ha) / r) ^ 2 * (g ^ 2 + r ^ 2) := by
    field_simp
  rw [e, Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq (div_nonneg (by linarith) hr.le)]
  ring

/-- **The assumed stair is as long as the stair it stands for**: for `ha ≤ hb`, riser
`r > 0` and going `g ≥ 0`, the 3D length of the line `walkingLine` pushes is at least
`(hb - ha) √(g² + r²) / r`. -/
theorem walkingLine_length_ge {ha hb riser going : ℝ} (hab : ha ≤ hb) (hr : 0 < riser)
    (hg : 0 ≤ going) (start : ℝ) (lane : ℕ → ℝ) {F : ℕ} (hF : 0 < F) {pieces : ℕ → ℕ}
    (hp : ∀ k, 0 < pieces k) :
    (hb - ha) * √(going ^ 2 + riser ^ 2) / riser ≤
      polyLen dist3 (linePts ha hb riser going start lane F pieces) := by
  rw [← stair_identity hab hr]
  refine le_trans ?_ (len3_ge _)
  apply Real.sqrt_le_sqrt
  have hd : 0 ≤ hb - ha := sub_nonneg.2 hab
  have hT : 0 ≤ stairTotal ha hb riser going := by
    unfold stairTotal
    exact mul_nonneg (div_nonneg hd hr.le) hg
  have h1 := linePts_planLen ha hb riser going start lane hF hp
  have h2 := linePts_riseLen ha hb riser going start lane hF hp
  rw [abs_of_nonneg hT] at h1
  rw [abs_of_nonneg hd] at h2
  have h3 := pow_le_pow_left₀ hT h1 2
  have h4 := pow_le_pow_left₀ hd h2 2
  linarith

/-- With the code's defaults (riser 175 mm, going 280 mm) the factor `√(g² + r²) / r` lies
between 1.88 and 1.89: "about 1.9 times the rise". -/
theorem default_ratio :
    (1.88 : ℝ) < √(0.28 ^ 2 + 0.175 ^ 2) / 0.175 ∧ √((0.28 : ℝ) ^ 2 + 0.175 ^ 2) / 0.175 < 1.89 := by
  constructor
  · rw [lt_div_iff₀ (by norm_num), Real.lt_sqrt (by norm_num)]
    norm_num
  · rw [div_lt_iff₀ (by norm_num), Real.sqrt_lt' (by norm_num)]
    norm_num

/-! ### Heights never decrease -/

theorem flightPts_heights {hA hB : ℝ} (hAB : hA ≤ hB) {N : ℕ} (hN : 0 < N) (fr dst lane : ℝ) :
    (flightPts fr dst lane hA hB N).Pairwise (fun p q => p.2.2 ≤ q.2.2) ∧
      ∀ p ∈ flightPts fr dst lane hA hB N, hA ≤ p.2.2 ∧ p.2.2 ≤ hB := by
  have hN' : (0 : ℝ) < N := by exact_mod_cast hN
  constructor
  · unfold flightPts
    rw [List.pairwise_map]
    refine (List.pairwise_lt_range (n := N + 1)).imp fun {a b} hab => ?_
    show hA + (hB - hA) * a / N ≤ hA + (hB - hA) * b / N
    have h1 : (a : ℝ) ≤ b := by exact_mod_cast hab.le
    have h2 : (hB - hA) * a ≤ (hB - hA) * b := mul_le_mul_of_nonneg_left h1 (by linarith)
    have h3 := div_le_div_of_nonneg_right h2 hN'.le
    linarith
  · intro p hp
    unfold flightPts at hp
    obtain ⟨i, hi, rfl⟩ := List.mem_map.1 hp
    have hi' : (i : ℝ) ≤ N := by exact_mod_cast Nat.lt_succ_iff.1 (List.mem_range.1 hi)
    show hA ≤ hA + (hB - hA) * i / N ∧ hA + (hB - hA) * i / N ≤ hB
    have h0 : 0 ≤ (hB - hA) * i / N :=
      div_nonneg (mul_nonneg (by linarith) (Nat.cast_nonneg i)) hN'.le
    have h1 : (hB - hA) * i / N ≤ hB - hA := by
      rw [div_le_iff₀ hN']
      nlinarith
    constructor <;> linarith

theorem hAt_mono {ha hb : ℝ} (hab : ha ≤ hb) (F : ℕ) {k k' : ℕ} (hk : k ≤ k') :
    hAt ha hb F k ≤ hAt ha hb F k' := by
  unfold hAt
  have h1 : (hb - ha) * (k : ℝ) ≤ (hb - ha) * k' :=
    mul_le_mul_of_nonneg_left (by exact_mod_cast hk) (by linarith)
  have h2 := div_le_div_of_nonneg_right h1 (Nat.cast_nonneg (α := ℝ) F)
  linarith

/-- **Heights along the line never decrease** (for `ha ≤ hb`). -/
theorem linePts_heights {ha hb : ℝ} (hab : ha ≤ hb) (riser going start : ℝ) (lane : ℕ → ℝ)
    {F : ℕ} {pieces : ℕ → ℕ} (hp : ∀ k, 0 < pieces k) :
    (linePts ha hb riser going start lane F pieces).Pairwise (fun p q => p.2.2 ≤ q.2.2) := by
  unfold linePts
  rw [List.pairwise_flatMap]
  constructor
  · intro k _
    exact (flightPts_heights (hAt_mono hab F (Nat.le_succ k)) (hp k) _ _ _).1
  · refine (List.pairwise_lt_range (n := F)).imp fun {a b} hab' => ?_
    intro x hx y hy
    have hx' := ((flightPts_heights (hAt_mono hab F (Nat.le_succ a)) (hp a) _ _ _).2 x hx).2
    have hy' := ((flightPts_heights (hAt_mono hab F (Nat.le_succ b)) (hp b) _ _ _).2 y hy).1
    exact hx'.trans ((hAt_mono hab F (Nat.succ_le_of_lt hab')).trans hy')

/-- `push`'s mirrors and axis swap: `a ↦ a0 + a1 - a` if `mirrorAlong`, `c ↦ c0 + c1 - c` if
`mirrorAcross`, then `[a, c, h]` or `[c, a, h]`. -/
noncomputable def outPt (a0 a1 c0 c1 : ℝ) (mirrorAlong mirrorAcross alongU : Bool) (p : Pt) : Pt :=
  if alongU = true then
    (if mirrorAlong = true then a0 + a1 - p.1 else p.1,
      if mirrorAcross = true then c0 + c1 - p.2.1 else p.2.1, p.2.2)
  else
    (if mirrorAcross = true then c0 + c1 - p.2.1 else p.2.1,
      if mirrorAlong = true then a0 + a1 - p.1 else p.1, p.2.2)

theorem outPt_height (a0 a1 c0 c1 : ℝ) (mA mC aU : Bool) (p : Pt) :
    (outPt a0 a1 c0 c1 mA mC aU p).2.2 = p.2.2 := by
  unfold outPt
  split_ifs <;> rfl

theorem planDist_outPt (a0 a1 c0 c1 : ℝ) (mA mC aU : Bool) (p q : Pt) :
    planDist (outPt a0 a1 c0 c1 mA mC aU p) (outPt a0 a1 c0 c1 mA mC aU q) = planDist p q := by
  unfold outPt planDist
  cases mA <;> cases mC <;> cases aU <;> simp only [Bool.false_eq_true, ↓reduceIte] <;>
    congr 1 <;> ring

theorem dist3_outPt (a0 a1 c0 c1 : ℝ) (mA mC aU : Bool) (p q : Pt) :
    dist3 (outPt a0 a1 c0 c1 mA mC aU p) (outPt a0 a1 c0 c1 mA mC aU q) = dist3 p q := by
  unfold outPt dist3
  cases mA <;> cases mC <;> cases aU <;> simp only [Bool.false_eq_true, ↓reduceIte] <;>
    congr 1 <;> ring

/-- The line `walkingLine` returns: the pushed points mirrored and swapped, each dropped when
it is within `1e-6` of the last point kept. -/
noncomputable def walkingLineOut (ha hb riser going start : ℝ) (lane : ℕ → ℝ) (F : ℕ)
    (pieces : ℕ → ℕ) (a0 a1 c0 c1 : ℝ) (mA mC aU : Bool) : List Pt :=
  ((linePts ha hb riser going start lane F pieces).map (outPt a0 a1 c0 c1 mA mC aU)).destutter
    fun p q => ¬ dist3 p q < 1e-6

/-- **Heights along the returned line never decrease** (for `ha ≤ hb`), whatever the
mirrors, the axes and the points the de-duplication drops. -/
theorem walkingLine_heights {ha hb : ℝ} (hab : ha ≤ hb) (riser going start : ℝ)
    (lane : ℕ → ℝ) {F : ℕ} {pieces : ℕ → ℕ} (hp : ∀ k, 0 < pieces k) (a0 a1 c0 c1 : ℝ)
    (mA mC aU : Bool) :
    (walkingLineOut ha hb riser going start lane F pieces a0 a1 c0 c1 mA mC aU).Pairwise
      (fun p q => p.2.2 ≤ q.2.2) := by
  unfold walkingLineOut
  refine List.Pairwise.sublist (List.destutter_sublist _ _) ?_
  rw [List.pairwise_map]
  exact (linePts_heights hab riser going start lane hp).imp fun h => by
    rwa [outPt_height, outPt_height]

/-- The mirrored and swapped line has the length bound of `walkingLine_length_ge`, and so does
any de-duplication of it that drops only repeated points. -/
theorem walkingLine_out_length_ge {ha hb riser going : ℝ} (hab : ha ≤ hb) (hr : 0 < riser)
    (hg : 0 ≤ going) (start : ℝ) (lane : ℕ → ℝ) {F : ℕ} (hF : 0 < F) {pieces : ℕ → ℕ}
    (hp : ∀ k, 0 < pieces k) (a0 a1 c0 c1 : ℝ) (mA mC aU : Bool) :
    (hb - ha) * √(going ^ 2 + riser ^ 2) / riser ≤
      polyLen dist3 (((linePts ha hb riser going start lane F pieces).map
        (outPt a0 a1 c0 c1 mA mC aU)).destutter (· ≠ ·)) := by
  rw [polyLen_destutter (fun p => by simp [dist3]), polyLen_map (dist3_outPt a0 a1 c0 c1 mA mC aU)]
  exact walkingLine_length_ge hab hr hg start lane hF hp

/-! ## `exitsPerPart`: every large part with a fallback door gets an exit -/

section Exits

variable {V : Type*} [Fintype V] [DecidableEq V]

/-- `sizes[c]`: the number of nodes labelled `c`. -/
def partSize (label : V → ℕ) (c : ℕ) : ℕ := (Finset.univ.filter fun v => label v = c).card

/-- The threshold `max(50, 0.02 n)`. -/
noncomputable def minPart (n : ℕ) : ℝ := max 50 (0.02 * n)

/-- `exitsPerPart(graph, flagged, fallback)`: the fallback nodes whose part holds no flagged
exit and has at least `minPart n` nodes. -/
noncomputable def exitsPerPart (label : V → ℕ) (flagged fallback : List V) : List V :=
  fallback.filter fun n =>
    decide (label n ∉ flagged.map label) &&
      decide (minPart (Fintype.card V) ≤ (partSize label (label n) : ℝ))

/-- The exit nodes of `exitDoors` (given the graph): with no flagged exit, every fallback
door; otherwise the flagged exits and the fallback doors `exitsPerPart` adds. -/
noncomputable def exitNodes (label : V → ℕ) (flagged fallback : List V) : List V :=
  if flagged = [] then fallback else flagged ++ exitsPerPart label flagged fallback

/-- What `exitsPerPart` adds: exactly the fallback nodes in a part with no flagged exit and of
at least the threshold size. -/
theorem mem_exitsPerPart {label : V → ℕ} {flagged fallback : List V} {n : V} :
    n ∈ exitsPerPart label flagged fallback ↔
      n ∈ fallback ∧ (∀ x ∈ flagged, label x ≠ label n) ∧
        minPart (Fintype.card V) ≤ (partSize label (label n) : ℝ) := by
  unfold exitsPerPart
  simp only [List.mem_filter, Bool.and_eq_true, decide_eq_true_eq, List.mem_map, not_exists,
    not_and]

/-- **Every part of at least the threshold size that holds a fallback door has an exit**: a
flagged one, or the door itself is added. -/
theorem exitsPerPart_serves (label : V → ℕ) (flagged fallback : List V) {n : V}
    (hn : n ∈ fallback) (hsize : minPart (Fintype.card V) ≤ (partSize label (label n) : ℝ)) :
    ∃ e ∈ flagged ++ exitsPerPart label flagged fallback, label e = label n := by
  by_cases h : ∃ x ∈ flagged, label x = label n
  · obtain ⟨x, hx, hl⟩ := h
    exact ⟨x, List.mem_append_left _ hx, hl⟩
  · push Not at h
    exact ⟨n, List.mem_append_right _ (mem_exitsPerPart.2 ⟨hn, h, hsize⟩), rfl⟩

/-- The same for `exitDoors`' exit nodes (with no flagged exit, every fallback door is one). -/
theorem exitNodes_serves (label : V → ℕ) (flagged fallback : List V) {n : V}
    (hn : n ∈ fallback) (hsize : minPart (Fintype.card V) ≤ (partSize label (label n) : ℝ)) :
    ∃ e ∈ exitNodes label flagged fallback, label e = label n := by
  unfold exitNodes
  split_ifs
  · exact ⟨n, hn, rfl⟩
  · exact exitsPerPart_serves label flagged fallback hn hsize

/-- **Every node of a large part with a fallback door can reach an exit.** With `label` a
component labelling of the open-link graph `Adj` (`componentLabels`, walls on), every node
`v` in the part of a fallback door `n` whose part has at least `max(50, 0.02 n)` nodes is
joined by open links to some exit node. -/
theorem large_part_reaches_exit (Adj : V → V → Prop) (label : V → ℕ)
    (hlabel : ∀ u v, label u = label v ↔ Relation.ReflTransGen Adj u v)
    (flagged fallback : List V) {n : V} (hn : n ∈ fallback)
    (hsize : minPart (Fintype.card V) ≤ (partSize label (label n) : ℝ)) (v : V)
    (hv : label v = label n) :
    ∃ e ∈ exitNodes label flagged fallback, Relation.ReflTransGen Adj v e := by
  obtain ⟨e, he, hl⟩ := exitNodes_serves label flagged fallback hn hsize
  exact ⟨e, he, (hlabel v e).1 (hv.trans hl.symm)⟩

end Exits

end FireEgress.Cores

/-! ## Axioms -/

#print axioms FireEgress.Cores.planDist_nonneg
#print axioms FireEgress.Cores.riseDist_nonneg
#print axioms FireEgress.Cores.dist3_eq
#print axioms FireEgress.Cores.hypot_add_le
#print axioms FireEgress.Cores.len3_ge
#print axioms FireEgress.Cores.riseDist_ge_ends
#print axioms FireEgress.Cores.len3_ge_ends
#print axioms FireEgress.Cores.polyLen_nonneg
#print axioms FireEgress.Cores.polyLen_append_ge
#print axioms FireEgress.Cores.polyLen_flatMap_ge
#print axioms FireEgress.Cores.polyLen_map_range
#print axioms FireEgress.Cores.polyLen_map
#print axioms FireEgress.Cores.destutter'_eq_cons
#print axioms FireEgress.Cores.polyLen_destutter'
#print axioms FireEgress.Cores.polyLen_destutter
#print axioms FireEgress.Cores.planDist_piece
#print axioms FireEgress.Cores.riseDist_piece
#print axioms FireEgress.Cores.flight_planLen
#print axioms FireEgress.Cores.flight_riseLen
#print axioms FireEgress.Cores.flight_span
#print axioms FireEgress.Cores.hAt_step
#print axioms FireEgress.Cores.linePts_planLen
#print axioms FireEgress.Cores.linePts_riseLen
#print axioms FireEgress.Cores.stair_identity
#print axioms FireEgress.Cores.walkingLine_length_ge
#print axioms FireEgress.Cores.default_ratio
#print axioms FireEgress.Cores.flightPts_heights
#print axioms FireEgress.Cores.hAt_mono
#print axioms FireEgress.Cores.linePts_heights
#print axioms FireEgress.Cores.outPt_height
#print axioms FireEgress.Cores.planDist_outPt
#print axioms FireEgress.Cores.dist3_outPt
#print axioms FireEgress.Cores.walkingLine_heights
#print axioms FireEgress.Cores.walkingLine_out_length_ge
#print axioms FireEgress.Cores.mem_exitsPerPart
#print axioms FireEgress.Cores.exitsPerPart_serves
#print axioms FireEgress.Cores.exitNodes_serves
#print axioms FireEgress.Cores.large_part_reaches_exit
