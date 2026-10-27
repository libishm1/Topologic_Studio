import Mathlib

/-!
# Wall blocking's segment test (Studio's lattice)

`buildLatticeGraph` in `src/core/lattice.js` marks a lattice link `a-b` as blocked
when `SegmentIndex.crosses` finds a wall-outline segment `c-d` with
`segmentsCross(ax, ay, bx, by, cx, cy, dx, dy)`, i.e. `d1 * d2 < 0 && d3 * d4 < 0`
with the four orientation determinants

* `d1 = (dx - cx)(ay - cy) - (dy - cy)(ax - cx)`  (side of `a` w.r.t. line `cd`),
* `d2 = (dx - cx)(by - cy) - (dy - cy)(bx - cx)`  (side of `b` w.r.t. line `cd`),
* `d3 = (bx - ax)(cy - ay) - (by - ay)(cx - ax)`  (side of `c` w.r.t. line `ab`),
* `d4 = (bx - ax)(dy - ay) - (by - ay)(dx - ax)`  (side of `d` w.r.t. line `ab`).

Every blocked link that is not caught by the extra `inside` samples rests on this.
The results here are over exact real arithmetic in `ℝ × ℝ`; the browser runs the
same formula in IEEE doubles (see the property test `src/core/segments.property.test.mjs`).
Segments are Mathlib's closed `segment ℝ A B` and `openSegment ℝ A B`.

* `sound`: if the test is true, the closed segments meet in exactly one point, the
  lines are not parallel, and that point is strictly between the endpoints of both.
* `complete`: if the lines are not parallel and the segments meet at a point of
  both open segments, the test is true.
* `crossTest_iff`: exact characterisation - the test is true iff the closed segments
  meet and no endpoint of either lies on the other closed segment.
* `false_while_touching_iff`, `touching_nonparallel_at_endpoint`,
  `touching_parallel_collinear`: the cases where the closed segments touch but the
  test says no are exactly "an endpoint lies on the other segment"; with
  non-parallel lines the contact point is itself one of the four endpoints, with
  parallel lines all four points are collinear (collinear overlap or end-to-end
  contact). These are the cases the code's `inside` samples are meant to cover.
-/

namespace FireEgress.Segments

/-- Literal transcription of `segmentsCross` in `src/core/lattice.js` (the JS `by`
is written `by_`, a Lean keyword). -/
def segmentsCross (ax ay bx by_ cx cy dx dy : ℝ) : Prop :=
  ((dx - cx) * (ay - cy) - (dy - cy) * (ax - cx)) *
      ((dx - cx) * (by_ - cy) - (dy - cy) * (bx - cx)) < 0 ∧
    ((bx - ax) * (cy - ay) - (by_ - ay) * (cx - ax)) *
      ((bx - ax) * (dy - ay) - (by_ - ay) * (dx - ax)) < 0

/-- Orientation determinant `(P - O) × (Q - O)`: positive if `O, P, Q` turn
anticlockwise, zero iff they are collinear. -/
def orient (O P Q : ℝ × ℝ) : ℝ := (P.1 - O.1) * (Q.2 - O.2) - (P.2 - O.2) * (Q.1 - O.1)

/-- The code's test on points `A = a, B = b, C = c, D = d`. -/
def crossTest (A B C D : ℝ × ℝ) : Prop :=
  segmentsCross A.1 A.2 B.1 B.2 C.1 C.2 D.1 D.2

/-- `(B - A) × (D - C)`: zero iff the two segments are parallel (or one is a point). -/
def den (A B C D : ℝ × ℝ) : ℝ := (B.1 - A.1) * (D.2 - C.2) - (B.2 - A.2) * (D.1 - C.1)

/-- The code's `d1, d2, d3, d4` are `orient C D A`, `orient C D B`, `orient A B C`,
`orient A B D`. -/
theorem crossTest_iff_orient (A B C D : ℝ × ℝ) :
    crossTest A B C D ↔
      orient C D A * orient C D B < 0 ∧ orient A B C * orient A B D < 0 := Iff.rfl

/-! ## Helper lemmas -/

theorem mem_segment_iff' {P A B : ℝ × ℝ} :
    P ∈ segment ℝ A B ↔ ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = A + t • (B - A) := by
  rw [segment_eq_image']
  constructor
  · rintro ⟨t, ⟨h0, h1⟩, rfl⟩; exact ⟨t, h0, h1, rfl⟩
  · rintro ⟨t, h0, h1, rfl⟩; exact ⟨t, ⟨h0, h1⟩, rfl⟩

theorem mem_openSegment_iff' {P A B : ℝ × ℝ} :
    P ∈ openSegment ℝ A B ↔ ∃ t : ℝ, 0 < t ∧ t < 1 ∧ P = A + t • (B - A) := by
  rw [openSegment_eq_image']
  constructor
  · rintro ⟨t, ⟨h0, h1⟩, rfl⟩; exact ⟨t, h0, h1, rfl⟩
  · rintro ⟨t, h0, h1, rfl⟩; exact ⟨t, ⟨h0, h1⟩, rfl⟩

theorem orient_lerp (O Q A B : ℝ × ℝ) (t : ℝ) :
    orient O Q (A + t • (B - A)) = (1 - t) * orient O Q A + t * orient O Q B := by
  simp only [orient, Prod.fst_add, Prod.snd_add, Prod.smul_fst, Prod.smul_snd,
    Prod.fst_sub, Prod.snd_sub, smul_eq_mul]
  ring

theorem orient_on_own_line (C D : ℝ × ℝ) (u : ℝ) : orient C D (C + u • (D - C)) = 0 := by
  simp only [orient, Prod.fst_add, Prod.snd_add, Prod.smul_fst, Prod.smul_snd,
    Prod.fst_sub, Prod.snd_sub, smul_eq_mul]
  ring

/-- A point of the closed segment `CD` is on the line `CD`. -/
theorem orient_eq_zero_of_mem_segment {C D X : ℝ × ℝ} (h : X ∈ segment ℝ C D) :
    orient C D X = 0 := by
  obtain ⟨u, -, -, rfl⟩ := mem_segment_iff'.1 h
  exact orient_on_own_line C D u

theorem den_eq_left (A B C D : ℝ × ℝ) : den A B C D = orient C D A - orient C D B := by
  simp only [den, orient]; ring

theorem den_eq_right (A B C D : ℝ × ℝ) : den A B C D = orient A B D - orient A B C := by
  simp only [den, orient]; ring

/-- If an affine combination `(1 - t) a + t b` with `0 < t < 1` vanishes and `a ≠ b`,
then `a` and `b` have strictly opposite signs. -/
theorem neg_prod_of_affine_zero {a b t : ℝ} (ht0 : 0 < t) (ht1 : t < 1)
    (h : (1 - t) * a + t * b = 0) (hne : a ≠ b) : a * b < 0 := by
  have hb : b ≠ 0 := by
    intro hb
    subst hb
    have h1t : (1 - t) ≠ 0 := by linarith
    have : a = 0 := by
      have : (1 - t) * a = 0 := by linarith
      rcases mul_eq_zero.1 this with h' | h'
      · exact absurd h' h1t
      · exact h'
    exact hne this
  have hkey : (1 - t) * (a * b) = -t * (b * b) := by linear_combination b * h
  have hbb : 0 < b * b := mul_self_pos.2 hb
  by_contra hc
  push Not at hc
  have : 0 ≤ (1 - t) * (a * b) := mul_nonneg (by linarith) hc
  nlinarith

/-- If `a * b < 0` then `t = a / (a - b)` lies strictly between 0 and 1. -/
theorem ratio_mem_Ioo {a b : ℝ} (h : a * b < 0) :
    a - b ≠ 0 ∧ 0 < a / (a - b) ∧ a / (a - b) < 1 := by
  rcases mul_neg_iff.1 h with ⟨ha, hb⟩ | ⟨ha, hb⟩
  · have hpos : 0 < a - b := by linarith
    refine ⟨hpos.ne', div_pos ha hpos, ?_⟩
    rw [div_lt_one hpos]; linarith
  · have hneg : a - b < 0 := by linarith
    refine ⟨hneg.ne, div_pos_of_neg_of_neg ha hneg, ?_⟩
    rw [div_lt_one_of_neg hneg]; linarith

/-! ## Soundness -/

/-- **Soundness.** If the code's test is true, the two lines are not parallel and
there are `0 < t < 1`, `0 < u < 1` with `A + t (B - A) = C + u (D - C)`, and that
point is the *only* point the closed segments `AB` and `CD` have in common. So a
blocked link really crosses the wall outline once, strictly inside both segments. -/
theorem sound {A B C D : ℝ × ℝ} (h : crossTest A B C D) :
    den A B C D ≠ 0 ∧
      ∃ t u : ℝ, 0 < t ∧ t < 1 ∧ 0 < u ∧ u < 1 ∧
        A + t • (B - A) = C + u • (D - C) ∧
        segment ℝ A B ∩ segment ℝ C D = {A + t • (B - A)} := by
  rw [crossTest_iff_orient] at h
  obtain ⟨h12, h34⟩ := h
  obtain ⟨hne12, ht0, ht1⟩ := ratio_mem_Ioo h12
  obtain ⟨hne34, hu0, hu1⟩ := ratio_mem_Ioo h34
  have hden : den A B C D ≠ 0 := by rw [den_eq_left]; exact hne12
  refine ⟨hden, orient C D A / (orient C D A - orient C D B),
    orient A B C / (orient A B C - orient A B D), ht0, ht1, hu0, hu1, ?_, ?_⟩
  · have e1 := hne12
    have e2 := hne34
    simp only [orient] at e1 e2 ⊢
    ext
    · simp only [Prod.fst_add, Prod.smul_fst, Prod.fst_sub, smul_eq_mul]
      field_simp
      ring
    · simp only [Prod.snd_add, Prod.smul_snd, Prod.snd_sub, smul_eq_mul]
      field_simp
      ring
  · set t := orient C D A / (orient C D A - orient C D B) with ht
    set u := orient A B C / (orient A B C - orient A B D) with hu
    have hmeet : A + t • (B - A) = C + u • (D - C) := by
      have e1 := hne12
      have e2 := hne34
      rw [ht, hu]
      simp only [orient] at e1 e2 ⊢
      ext
      · simp only [Prod.fst_add, Prod.smul_fst, Prod.fst_sub, smul_eq_mul]
        field_simp
        ring
      · simp only [Prod.snd_add, Prod.smul_snd, Prod.snd_sub, smul_eq_mul]
        field_simp
        ring
    apply Set.eq_singleton_iff_unique_mem.2
    refine ⟨⟨mem_segment_iff'.2 ⟨t, ht0.le, ht1.le, rfl⟩,
      mem_segment_iff'.2 ⟨u, hu0.le, hu1.le, hmeet⟩⟩, ?_⟩
    rintro X ⟨hX1, hX2⟩
    obtain ⟨s, -, -, rfl⟩ := mem_segment_iff'.1 hX1
    have hz := orient_eq_zero_of_mem_segment hX2
    rw [orient_lerp] at hz
    have hs : s = t := by
      rw [ht, eq_div_iff hne12]
      linarith
    rw [hs]

/-- Soundness in Mathlib's vocabulary: the common point lies in both open segments,
is none of the four endpoints, and is the whole intersection of the closed segments. -/
theorem sound_openSegment {A B C D : ℝ × ℝ} (h : crossTest A B C D) :
    ∃ P, P ∈ openSegment ℝ A B ∧ P ∈ openSegment ℝ C D ∧
      P ≠ A ∧ P ≠ B ∧ P ≠ C ∧ P ≠ D ∧ segment ℝ A B ∩ segment ℝ C D = {P} := by
  obtain ⟨hden, t, u, ht0, ht1, hu0, hu1, hmeet, hset⟩ := sound h
  have hAB : A ≠ B := by
    rintro rfl; apply hden; simp [den]
  have hCD : C ≠ D := by
    rintro rfl; apply hden; simp [den]
  have hP1 : A + t • (B - A) ∈ openSegment ℝ A B := mem_openSegment_iff'.2 ⟨t, ht0, ht1, rfl⟩
  have hP2 : A + t • (B - A) ∈ openSegment ℝ C D :=
    mem_openSegment_iff'.2 ⟨u, hu0, hu1, hmeet⟩
  refine ⟨_, hP1, hP2, ?_, ?_, ?_, ?_, hset⟩
  · intro e; rw [e] at hP1; exact hAB (left_mem_openSegment_iff.1 hP1)
  · intro e; rw [e] at hP1; exact hAB (right_mem_openSegment_iff.1 hP1)
  · intro e; rw [e] at hP2; exact hCD (left_mem_openSegment_iff.1 hP2)
  · intro e; rw [e] at hP2; exact hCD (right_mem_openSegment_iff.1 hP2)

/-- A degenerate (zero-length) link or wall segment never counts as a crossing. -/
theorem degenerate_false {A B C D : ℝ × ℝ} (h : A = B ∨ C = D) : ¬ crossTest A B C D := by
  intro hc
  apply (sound hc).1
  rcases h with rfl | rfl <;> simp [den]

/-! ## Completeness -/

/-- **Completeness.** If the segments are not parallel (`den ≠ 0`) and meet at a point
strictly inside both, the code's test is true: a link that properly crosses a wall
outline segment is always blocked. -/
theorem complete {A B C D P : ℝ × ℝ} (hden : den A B C D ≠ 0)
    (h1 : P ∈ openSegment ℝ A B) (h2 : P ∈ openSegment ℝ C D) : crossTest A B C D := by
  rw [crossTest_iff_orient]
  obtain ⟨t, ht0, ht1, hPt⟩ := mem_openSegment_iff'.1 h1
  obtain ⟨u, hu0, hu1, hPu⟩ := mem_openSegment_iff'.1 h2
  constructor
  · have hz : orient C D P = 0 := by rw [hPu]; exact orient_on_own_line C D u
    rw [hPt, orient_lerp] at hz
    refine neg_prod_of_affine_zero ht0 ht1 hz ?_
    intro e; apply hden; rw [den_eq_left, e, sub_self]
  · have hz : orient A B P = 0 := by rw [hPt]; exact orient_on_own_line A B t
    rw [hPu, orient_lerp] at hz
    refine neg_prod_of_affine_zero hu0 hu1 hz ?_
    intro e; apply hden; rw [den_eq_right, e, sub_self]

/-! ## Exact characterisation -/

/-- Projection onto the line `CD`, as a parameter along `D - C`. -/
noncomputable def proj (C D Y : ℝ × ℝ) : ℝ :=
  ((Y.1 - C.1) * (D.1 - C.1) + (Y.2 - C.2) * (D.2 - C.2)) /
    ((D.1 - C.1) ^ 2 + (D.2 - C.2) ^ 2)

theorem normSq_pos {C D : ℝ × ℝ} (h : C ≠ D) : 0 < (D.1 - C.1) ^ 2 + (D.2 - C.2) ^ 2 := by
  by_contra hc
  push Not at hc
  have h1 : D.1 - C.1 = 0 := by nlinarith [sq_nonneg (D.1 - C.1), sq_nonneg (D.2 - C.2)]
  have h2 : D.2 - C.2 = 0 := by nlinarith [sq_nonneg (D.1 - C.1), sq_nonneg (D.2 - C.2)]
  exact h (Prod.ext (by linarith) (by linarith))

/-- A point on the line `CD` is `C + proj Y • (D - C)`. -/
theorem eq_of_on_line {C D Y : ℝ × ℝ} (hCD : C ≠ D) (hY : orient C D Y = 0) :
    Y = C + proj C D Y • (D - C) := by
  have hq := (normSq_pos hCD).ne'
  simp only [orient] at hY
  ext
  · simp only [proj, Prod.fst_add, Prod.smul_fst, Prod.fst_sub, smul_eq_mul]
    field_simp
    linear_combination (-(D.2 - C.2)) * hY
  · simp only [proj, Prod.snd_add, Prod.smul_snd, Prod.snd_sub, smul_eq_mul]
    field_simp
    linear_combination (D.1 - C.1) * hY

theorem proj_lerp (C D : ℝ × ℝ) (hCD : C ≠ D) (A B : ℝ × ℝ) (t : ℝ) :
    proj C D (A + t • (B - A)) = proj C D A + t * (proj C D B - proj C D A) := by
  have hq := (normSq_pos hCD).ne'
  simp only [proj, Prod.fst_add, Prod.snd_add, Prod.smul_fst, Prod.smul_snd,
    Prod.fst_sub, Prod.snd_sub, smul_eq_mul]
  field_simp
  ring

theorem proj_own (C D : ℝ × ℝ) (hCD : C ≠ D) (u : ℝ) : proj C D (C + u • (D - C)) = u := by
  have hq := (normSq_pos hCD).ne'
  simp only [proj, Prod.fst_add, Prod.snd_add, Prod.smul_fst, Prod.smul_snd,
    Prod.fst_sub, Prod.snd_sub, smul_eq_mul]
  field_simp
  ring

/-- A convex combination α + t (β - α), 0 ≤ t ≤ 1, lies between α and β. -/
theorem lerp_between (α β t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1) :
    min α β ≤ α + t * (β - α) ∧ α + t * (β - α) ≤ max α β := by
  rcases le_total α β with h | h
  · rw [min_eq_left h, max_eq_right h]
    constructor <;> nlinarith [mul_nonneg ht0 (sub_nonneg.2 h), mul_nonneg (sub_nonneg.2 ht1) (sub_nonneg.2 h)]
  · rw [min_eq_right h, max_eq_left h]
    constructor <;> nlinarith [mul_nonneg ht0 (sub_nonneg.2 h), mul_nonneg (sub_nonneg.2 ht1) (sub_nonneg.2 h)]

/-- Collinear segments that meet: some endpoint of one lies on the other
(the 1-D fact that two overlapping intervals share an endpoint). -/
theorem collinear_endpoint {A B C D X : ℝ × ℝ} (hCD : C ≠ D)
    (hA : orient C D A = 0) (hB : orient C D B = 0)
    (hX1 : X ∈ segment ℝ A B) (hX2 : X ∈ segment ℝ C D) :
    A ∈ segment ℝ C D ∨ B ∈ segment ℝ C D ∨ C ∈ segment ℝ A B ∨ D ∈ segment ℝ A B := by
  obtain ⟨t, ht0, ht1, hXt⟩ := mem_segment_iff'.1 hX1
  obtain ⟨u, hu0, hu1, hXu⟩ := mem_segment_iff'.1 hX2
  set α := proj C D A with hα
  set β := proj C D B with hβ
  have eA : A = C + α • (D - C) := eq_of_on_line hCD hA
  have eB : B = C + β • (D - C) := eq_of_on_line hCD hB
  have hu : u = α + t * (β - α) := by
    have := congrArg (proj C D) (hXt.symm.trans hXu)
    rw [proj_lerp C D hCD, proj_own C D hCD] at this
    linarith
  by_cases hα01 : 0 ≤ α ∧ α ≤ 1
  · exact Or.inl (mem_segment_iff'.2 ⟨α, hα01.1, hα01.2, eA⟩)
  by_cases hβ01 : 0 ≤ β ∧ β ≤ 1
  · exact Or.inr (Or.inl (mem_segment_iff'.2 ⟨β, hβ01.1, hβ01.2, eB⟩))
  -- `α` and `β` are both outside `[0, 1]` but `u ∈ [0, 1]` lies between them,
  -- so `0` lies between them and `C` is on `AB`.
  right; right; left
  have hside : (α < 0 ∧ 1 < β) ∨ (1 < α ∧ β < 0) := by
    rcases not_and_or.1 hα01 with h | h <;> rcases not_and_or.1 hβ01 with h' | h' <;>
      push Not at h h'
    · exfalso; have := (lerp_between α β t ht0 ht1).2; rw [← hu] at this
      have : max α β < 0 := max_lt h h'; linarith
    · left; exact ⟨h, h'⟩
    · right; exact ⟨h, h'⟩
    · exfalso; have := (lerp_between α β t ht0 ht1).1; rw [← hu] at this
      have : 1 < min α β := lt_min h h'; linarith
  have hne : α - β ≠ 0 := by rcases hside with ⟨h, h'⟩ | ⟨h, h'⟩ <;> linarith
  refine mem_segment_iff'.2 ⟨α / (α - β), ?_, ?_, ?_⟩
  · rcases hside with ⟨h, h'⟩ | ⟨h, h'⟩
    · exact (div_pos_of_neg_of_neg h (by linarith)).le
    · exact (div_pos (by linarith) (by linarith)).le
  · rcases hside with ⟨h, h'⟩ | ⟨h, h'⟩
    · rw [div_le_one_of_neg (by linarith)]; linarith
    · rw [div_le_one (by linarith)]; linarith
  · rw [eA, eB]
    ext
    · simp only [Prod.fst_add, Prod.smul_fst, Prod.fst_sub, smul_eq_mul]
      field_simp
      ring
    · simp only [Prod.snd_add, Prod.smul_snd, Prod.snd_sub, smul_eq_mul]
      field_simp
      ring

/-- **Exact characterisation.** The code's test is true if and only if the closed
segments meet and no endpoint of either lies on the other closed segment. -/
theorem crossTest_iff {A B C D : ℝ × ℝ} :
    crossTest A B C D ↔
      (segment ℝ A B ∩ segment ℝ C D).Nonempty ∧ A ∉ segment ℝ C D ∧
        B ∉ segment ℝ C D ∧ C ∉ segment ℝ A B ∧ D ∉ segment ℝ A B := by
  constructor
  · intro h
    have h' := h
    rw [crossTest_iff_orient] at h'
    obtain ⟨-, t, u, -, -, -, -, -, hset⟩ := sound h
    refine ⟨⟨A + t • (B - A), by rw [hset]; rfl⟩, ?_, ?_, ?_, ?_⟩ <;> intro hm <;>
      have hz := orient_eq_zero_of_mem_segment hm
    · rw [hz, zero_mul] at h'; exact lt_irrefl _ h'.1
    · rw [hz, mul_zero] at h'; exact lt_irrefl _ h'.1
    · rw [hz, zero_mul] at h'; exact lt_irrefl _ h'.2
    · rw [hz, mul_zero] at h'; exact lt_irrefl _ h'.2
  · rintro ⟨⟨X, hX1, hX2⟩, hA, hB, hC, hD⟩
    obtain ⟨t, ht0, ht1, hXt⟩ := mem_segment_iff'.1 hX1
    obtain ⟨u, hu0, hu1, hXu⟩ := mem_segment_iff'.1 hX2
    have ht0' : 0 < t := by
      rcases ht0.lt_or_eq with h | h
      · exact h
      · exfalso; apply hA; rw [hXt, ← h] at hX2; simpa using hX2
    have ht1' : t < 1 := by
      rcases ht1.lt_or_eq with h | h
      · exact h
      · exfalso; apply hB; rw [hXt, h] at hX2; simpa using hX2
    have hu0' : 0 < u := by
      rcases hu0.lt_or_eq with h | h
      · exact h
      · exfalso; apply hC; rw [hXu, ← h] at hX1; simpa using hX1
    have hu1' : u < 1 := by
      rcases hu1.lt_or_eq with h | h
      · exact h
      · exfalso; apply hD; rw [hXu, h] at hX1; simpa using hX1
    by_cases hden : den A B C D = 0
    · exfalso
      have hCD : C ≠ D := by
        rintro rfl
        apply hC
        have : X = C := by rw [hXu]; simp
        rw [← this]; exact hX1
      have hz : orient C D X = 0 := orient_eq_zero_of_mem_segment hX2
      rw [hXt, orient_lerp] at hz
      have heq : orient C D A = orient C D B := by
        have := den_eq_left A B C D; linarith
      have hA0 : orient C D A = 0 := by rw [← heq] at hz; linarith
      have hB0 : orient C D B = 0 := by linarith
      rcases collinear_endpoint hCD hA0 hB0 hX1 hX2 with h | h | h | h
      · exact hA h
      · exact hB h
      · exact hC h
      · exact hD h
    · exact complete hden (mem_openSegment_iff'.2 ⟨t, ht0', ht1', hXt⟩)
        (mem_openSegment_iff'.2 ⟨u, hu0', hu1', hXu⟩)

/-- **When the test says no although the closed segments touch.** Given that the
closed segments meet, the test is false exactly when an endpoint of one lies on the
other closed segment (T-junction, shared vertex, end-to-end contact or collinear
overlap). These are the only contacts the code's `inside` samples have to catch. -/
theorem false_while_touching_iff {A B C D : ℝ × ℝ}
    (hmeet : (segment ℝ A B ∩ segment ℝ C D).Nonempty) :
    ¬ crossTest A B C D ↔
      A ∈ segment ℝ C D ∨ B ∈ segment ℝ C D ∨ C ∈ segment ℝ A B ∨ D ∈ segment ℝ A B := by
  rw [crossTest_iff]
  tauto

/-- Non-parallel case: if the closed segments touch at `X` and the test is false,
`X` is one of the four endpoints (the link starts or ends on the wall line, or passes
exactly through an outline vertex). -/
theorem touching_nonparallel_at_endpoint {A B C D X : ℝ × ℝ} (hden : den A B C D ≠ 0)
    (hX1 : X ∈ segment ℝ A B) (hX2 : X ∈ segment ℝ C D) (hno : ¬ crossTest A B C D) :
    X = A ∨ X = B ∨ X = C ∨ X = D := by
  obtain ⟨t, ht0, ht1, hXt⟩ := mem_segment_iff'.1 hX1
  obtain ⟨u, hu0, hu1, hXu⟩ := mem_segment_iff'.1 hX2
  rcases ht0.lt_or_eq with ht0' | ht0'
  · rcases ht1.lt_or_eq with ht1' | ht1'
    · rcases hu0.lt_or_eq with hu0' | hu0'
      · rcases hu1.lt_or_eq with hu1' | hu1'
        · exact absurd (complete hden (mem_openSegment_iff'.2 ⟨t, ht0', ht1', hXt⟩)
            (mem_openSegment_iff'.2 ⟨u, hu0', hu1', hXu⟩)) hno
        · right; right; right; rw [hXu, hu1']; simp
      · right; right; left; rw [hXu, ← hu0']; simp
    · right; left; rw [hXt, ht1']; simp
  · left; rw [hXt, ← ht0']; simp

/-- Parallel case: closed segments that are parallel (`den = 0`) and touch are
collinear - all four determinants vanish - and the test is false. This is the
collinear-overlap (a link running along a wall face) and end-to-end case. -/
theorem touching_parallel_collinear {A B C D X : ℝ × ℝ} (hden : den A B C D = 0)
    (hX1 : X ∈ segment ℝ A B) (hX2 : X ∈ segment ℝ C D) :
    orient C D A = 0 ∧ orient C D B = 0 ∧ orient A B C = 0 ∧ orient A B D = 0 ∧
      ¬ crossTest A B C D := by
  obtain ⟨t, -, -, hXt⟩ := mem_segment_iff'.1 hX1
  obtain ⟨u, -, -, hXu⟩ := mem_segment_iff'.1 hX2
  have hz1 : orient C D X = 0 := by rw [hXu]; exact orient_on_own_line C D u
  have hz2 : orient A B X = 0 := by rw [hXt]; exact orient_on_own_line A B t
  rw [hXt, orient_lerp] at hz1
  rw [hXu, orient_lerp] at hz2
  have e1 := den_eq_left A B C D
  have e2 := den_eq_right A B C D
  have hab : orient C D A = orient C D B := by linarith
  have hcd : orient A B C = orient A B D := by linarith
  have hA : orient C D A = 0 := by linear_combination hz1 + t * hab
  have hB : orient C D B = 0 := by linarith
  have hC : orient A B C = 0 := by linear_combination hz2 + u * hcd
  have hD : orient A B D = 0 := by linarith
  refine ⟨hA, hB, hC, hD, ?_⟩
  rw [crossTest_iff_orient, hA, zero_mul]
  exact fun h => lt_irrefl _ h.1

end FireEgress.Segments

#print axioms FireEgress.Segments.crossTest_iff_orient
#print axioms FireEgress.Segments.mem_segment_iff'
#print axioms FireEgress.Segments.mem_openSegment_iff'
#print axioms FireEgress.Segments.orient_lerp
#print axioms FireEgress.Segments.orient_on_own_line
#print axioms FireEgress.Segments.orient_eq_zero_of_mem_segment
#print axioms FireEgress.Segments.den_eq_left
#print axioms FireEgress.Segments.den_eq_right
#print axioms FireEgress.Segments.neg_prod_of_affine_zero
#print axioms FireEgress.Segments.ratio_mem_Ioo
#print axioms FireEgress.Segments.sound
#print axioms FireEgress.Segments.sound_openSegment
#print axioms FireEgress.Segments.degenerate_false
#print axioms FireEgress.Segments.complete
#print axioms FireEgress.Segments.normSq_pos
#print axioms FireEgress.Segments.eq_of_on_line
#print axioms FireEgress.Segments.proj_lerp
#print axioms FireEgress.Segments.proj_own
#print axioms FireEgress.Segments.lerp_between
#print axioms FireEgress.Segments.collinear_endpoint
#print axioms FireEgress.Segments.crossTest_iff
#print axioms FireEgress.Segments.false_while_touching_iff
#print axioms FireEgress.Segments.touching_nonparallel_at_endpoint
#print axioms FireEgress.Segments.touching_parallel_collinear
