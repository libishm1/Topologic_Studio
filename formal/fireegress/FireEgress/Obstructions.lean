import Mathlib

/-!
# Obstructions: NFPA 13's beam rule and three times rule

Specifications for the obstruction check of `src/core/sprinklers.js`: `OBSTRUCTION_RULES`,
`beamRuleRow`, `governing`, `meets`, `clearance` and `checkObstructions`
(`wiki/research/sprinkler-codes-2026-11-07.md` section 3.4, rows 33 to 38). Real arithmetic.
The beam table is the code's: each row runs from its `from` up to, not including, its `below`,
both in feet times `FT = 0.3048`, and allows the deflector at most `max` metres above the
obstruction's bottom (`Row`, `row`, `beamRows`). `A` is the plan distance from the head to the
obstruction's footprint and `B` the deflector's height above its bottom; a head moved in plan keeps
its deflector, so `B` is fixed while `A` changes. Each limit sits beside the statement it
limits.

## Which theorem backs which part of the code

* `beamRuleRow` (`if (!(A >= 0)) return null; for (row of rows) if (A < row.below) return row`):
  `lookup` over `beamRows`, null for a negative `A` (`beamRuleRow`). The table's shape:
  `beamRows_before` (each row's `max` is at least every earlier row's, and it starts at or after
  every earlier row ends), `beamRows_contiguous` (each row starts where the one before ends),
  `beamRows_head`, `beamRows_hi_le`, `beamRows_lo_le`, `beamRows_lo_nonneg`. Then
  `beamRuleRow_eq` (a distance in a row's half-open interval gets that row: no interpolation, the
  checks of `sprinklerObstructions.test.mjs` at a row's start, inside it, just below its end and at
  its end), `beamRuleRow_row` (for `0 ≤ A < 7 ft 6 in` there is a row and `A` lies in it, so
  `governing`, which asks for a row only there, never reads `max` of null) and `beamRuleRow_none`
  (null below 0 and from 7 ft 6 in on, where the table sets no limit).
* Moving a head away never makes the beam rule stricter: `lookup_mono` (for any table whose `max`
  grows row by row, the first-match lookup's `max` is monotone in `A`), `beam_mono` (the code's
  table: the row at `A` allows no more than the row at any `A' ≥ A`) and `beam_pass_mono` (a head
  that meets the rule against an obstruction at `A`, `governing`'s margin `max - B ≥ -EPS`
  (`beamOK`), meets it at every `A' ≥ A`, including beyond 7 ft 6 in where no row applies).
* `clearance` (the plan distance `move` keeps from each obstruction): `need`, `need_iff`,
  `need_le`, `need_ge` (for a contiguous table with `max` growing, the first-match row allows `x`
  exactly when `A` is at least the start of the first row whose `max ≥ x`) and `clearance_beam`
  (the beam rule passes at `A` exactly when `A ≥` the `from` of the first row with
  `max ≥ B - EPS`, or 7 ft 6 in when there is none: the code's comment "the beam table's B grows
  row by row, so the beam rule's limit is the A of the first row that allows the head's B");
  `clearance_three` (the three times rule passes exactly when `A ≥ min(3 d, 0.61) - EPS`);
  `clearance_exact` (both rules pass exactly when `A ≥` the larger limit, which is `move`'s
  `blocked` test); `blocked_hull` (the points nearer a convex footprint than a distance form a
  convex set, so an obstruction that blocks all four corners of the search rectangle blocks all of
  it, the shortcut `move` takes).
* The three times rule, `required = min(3 d, 0.61)` with `d` the larger of depth and plan width
  (`threeRequired`, `threeOK`): `three_times_mono` (monotone in `d`), `three_times_le` (never more
  than 24 in), `three_times_nonneg`, `three_pass_mono` (a head meeting it at distance `x` meets it at
  every `x' ≥ x`), `three_pass_dim` (and against any smaller obstruction), `three_times_far` (from
  0.61 m on it always passes).
* `REACH = max(7 ft 6 in, 0.61)`: `reach_eq`, `reach_skip` (from `REACH` on both rules pass, so
  `governing`'s `if (A >= REACH) continue` and the box prefilter change no verdict).
* `governing` and `meets`: `least_margin_iff` (the least margin over the obstructions is at least
  `-EPS` exactly when every margin is, so keeping only the governing obstruction loses nothing) and
  `meets_mono` (a head moved so that it is no nearer any obstruction keeps meeting both rules: the
  sets the rules take depend on the deflector height and the obstruction, not on the plan point).

## Hypotheses, and what is not covered

* `A ≥ 0` wherever a row is read: the code's `A` is a distance (`distanceToConvex`, 0 inside).
* Moving away from one obstruction can bring a head nearer another; the statements are per
  obstruction, and `meets_mono` asks every distance not to fall.
* Not covered: the plan geometry (`distanceToConvex`, `obstructionFromMesh`'s hulls, the box index,
  the room scoping of `obstructionsOver`), the wide rule (`wideVerdicts`), the obstructed
  construction check, `move`'s search order and floating point (`EPS = 1e-6` enters as `eps`).
-/

namespace FireEgress.Obstructions

/-! ## The beam rule's table -/

/-- `FT`, one foot in metres. -/
noncomputable def FT : ℝ := 0.3048

/-- A row of the beam rule: it holds for `lo ≤ A < hi` (m) and allows the deflector at most `B`
(m) above the obstruction's bottom (the code's `from`, `below` and `max`). -/
structure Row where
  lo : ℝ
  hi : ℝ
  B : ℝ

/-- A row as `OBSTRUCTION_RULES.beam.rows` builds it: `from` and `below` in feet times `FT`. -/
noncomputable def row (fromFt belowFt maxB : ℝ) : Row := ⟨fromFt * FT, belowFt * FT, maxB⟩

/-- `OBSTRUCTION_RULES.beam.rows`, NFPA 13-2019 Table 10.2.7.1.2, as in the code. -/
noncomputable def beamRows : List Row :=
  [row 0 1 0, row 1 1.5 0.065, row 1.5 2 0.09, row 2 2.5 0.14, row 2.5 3 0.19, row 3 3.5 0.24,
    row 3.5 4 0.3, row 4 4.5 0.35, row 4.5 5 0.415, row 5 5.5 0.45, row 5.5 6 0.5,
    row 6 6.5 0.6, row 6.5 7 0.75, row 7 7.5 0.875]

/-- `BEAM.reach`, 7 ft 6 in (2.286 m). -/
noncomputable def beamReach : ℝ := 7.5 * FT

/-- The loop of `beamRuleRow`: the first row whose `below` lies beyond `A`. -/
noncomputable def lookup : List Row → ℝ → Option Row
  | [], _ => none
  | r :: rs, A => if A < r.hi then some r else lookup rs A

/-- `beamRuleRow(A)`: null for a negative `A`, else the first row with `A < below`. -/
noncomputable def beamRuleRow (A : ℝ) : Option Row := if 0 ≤ A then lookup beamRows A else none

/-- Rows in order: a later row allows at least the `max` of an earlier one and starts at or
after the earlier one's end. -/
def Before (r s : Row) : Prop := r.B ≤ s.B ∧ r.hi ≤ s.lo

/-- Each row starts where the one before it ends. -/
def Contiguous : List Row → Prop
  | [] => True
  | [_] => True
  | r :: s :: rest => r.hi = s.lo ∧ Contiguous (s :: rest)

theorem contiguous_cons {s : Row} {ss : List Row} (h : Contiguous (s :: ss)) :
    (∀ t ∈ ss.head?, s.hi = t.lo) ∧ Contiguous ss := by
  cases ss with
  | nil => simp [Contiguous]
  | cons t ts =>
    simp only [Contiguous] at h
    refine ⟨fun u hu => ?_, h.2⟩
    simp only [List.head?_cons, Option.mem_def, Option.some.injEq] at hu
    subst hu
    exact h.1

theorem beamRows_before : beamRows.Pairwise Before := by
  norm_num [beamRows, row, Before, FT, List.pairwise_cons]

theorem beamRows_contiguous : Contiguous beamRows := by
  simp [beamRows, row, Contiguous]

theorem beamRows_head : beamRows.head? = some (row 0 1 0) := rfl

theorem beamRows_hi_le : ∀ r ∈ beamRows, r.hi ≤ beamReach := by
  norm_num [beamRows, row, beamReach, FT]

theorem beamRows_lo_le : ∀ r ∈ beamRows, r.lo ≤ beamReach := by
  norm_num [beamRows, row, beamReach, FT]

theorem beamRows_lo_nonneg : ∀ r ∈ beamRows, 0 ≤ r.lo := by
  norm_num [beamRows, row, FT]

theorem beamRows_head_le {A : ℝ} (hA : 0 ≤ A) : ∀ r ∈ beamRows.head?, r.lo ≤ A := by
  intro r hr
  rw [beamRows_head, Option.mem_def, Option.some.injEq] at hr
  subst hr
  simpa [row] using hA

/-! ## The first-match lookup -/

theorem lookup_mem {rs : List Row} {A : ℝ} {r : Row} (h : lookup rs A = some r) :
    r ∈ rs ∧ A < r.hi := by
  induction rs with
  | nil => simp [lookup] at h
  | cons s ss ih =>
    simp only [lookup] at h
    by_cases hs : A < s.hi
    · rw [ite_eq_left hs] at h
      rw [← Option.some.inj h]
      exact ⟨List.mem_cons.2 (Or.inl rfl), hs⟩
    · rw [ite_eq_right hs] at h
      exact ⟨List.mem_cons.2 (Or.inr (ih h).1), (ih h).2⟩

theorem lookup_some {rs : List Row} {A : ℝ} (h : ∃ t ∈ rs, A < t.hi) :
    ∃ r, lookup rs A = some r := by
  induction rs with
  | nil => obtain ⟨t, ht, -⟩ := h; simp at ht
  | cons s ss ih =>
    simp only [lookup]
    by_cases hs : A < s.hi
    · exact ⟨s, ite_eq_left hs⟩
    · rw [ite_eq_right hs]
      obtain ⟨t, ht, hAt⟩ := h
      rcases List.mem_cons.1 ht with rfl | ht'
      · exact absurd hAt hs
      · exact ih ⟨t, ht', hAt⟩

theorem lookup_none {rs : List Row} {A : ℝ} (h : ∀ t ∈ rs, t.hi ≤ A) : lookup rs A = none := by
  induction rs with
  | nil => rfl
  | cons s ss ih =>
    simp only [lookup]
    rw [ite_eq_right (not_lt.2 (h s (List.mem_cons.2 (Or.inl rfl))))]
    exact ih fun t ht => h t (List.mem_cons.2 (Or.inr ht))

/-- In a contiguous table starting at or below `A`, the row found starts at or below `A`. -/
theorem lookup_lo {rs : List Row} {A : ℝ} (hC : Contiguous rs) (hA : ∀ r ∈ rs.head?, r.lo ≤ A)
    {r : Row} (h : lookup rs A = some r) : r.lo ≤ A := by
  induction rs with
  | nil => simp [lookup] at h
  | cons s ss ih =>
    have hC' := contiguous_cons hC
    simp only [lookup] at h
    by_cases hs : A < s.hi
    · rw [ite_eq_left hs] at h
      rw [← Option.some.inj h]
      exact hA s (by simp)
    · rw [ite_eq_right hs] at h
      refine ih hC'.2 (fun t ht => ?_) h
      rw [← hC'.1 t ht]
      exact not_lt.1 hs

/-- A distance in a row's half-open interval gets that row. -/
theorem lookup_eq_of_mem {rs : List Row} (hP : rs.Pairwise Before) {r : Row} (hr : r ∈ rs)
    {A : ℝ} (h0 : r.lo ≤ A) (h1 : A < r.hi) : lookup rs A = some r := by
  induction rs with
  | nil => simp at hr
  | cons s ss ih =>
    rw [List.pairwise_cons] at hP
    simp only [lookup]
    rcases List.mem_cons.1 hr with rfl | hr'
    · rw [ite_eq_left h1]
    · rw [ite_eq_right (not_lt.2 (le_trans (hP.1 r hr').2 h0))]
      exact ih hP.2 hr'

/-- The first-match lookup's `max` never falls as `A` grows, for any table whose `max` grows row
by row. -/
theorem lookup_mono {rs : List Row} (hs : rs.Pairwise Before) {A A' : ℝ} (hA : A ≤ A')
    {r' : Row} (h' : lookup rs A' = some r') : ∃ r, lookup rs A = some r ∧ r.B ≤ r'.B := by
  induction rs with
  | nil => simp [lookup] at h'
  | cons s ss ih =>
    rw [List.pairwise_cons] at hs
    simp only [lookup] at h' ⊢
    by_cases hs' : A' < s.hi
    · rw [ite_eq_left hs'] at h'
      rw [← Option.some.inj h']
      exact ⟨s, ite_eq_left (lt_of_le_of_lt hA hs'), le_rfl⟩
    · rw [ite_eq_right hs'] at h'
      by_cases hsA : A < s.hi
      · exact ⟨s, ite_eq_left hsA, (hs.1 r' (lookup_mem h').1).1⟩
      · rw [ite_eq_right hsA]
        exact ih hs.2 h'

/-! ## `beamRuleRow` -/

/-- No interpolation: a distance from a row's `from` up to, not including, its `below` gets that
row. -/
theorem beamRuleRow_eq {A : ℝ} {r : Row} (hr : r ∈ beamRows) (h0 : r.lo ≤ A) (h1 : A < r.hi) :
    beamRuleRow A = some r := by
  unfold beamRuleRow
  rw [ite_eq_left (le_trans (beamRows_lo_nonneg r hr) h0)]
  exact lookup_eq_of_mem beamRows_before hr h0 h1

/-- Within 7 ft 6 in a row exists and holds `A`: `governing` never reads `max` of null. -/
theorem beamRuleRow_row {A : ℝ} (hA : 0 ≤ A) (hR : A < beamReach) :
    ∃ r ∈ beamRows, beamRuleRow A = some r ∧ r.lo ≤ A ∧ A < r.hi := by
  have hsome : ∃ t ∈ beamRows, A < t.hi :=
    ⟨row 7 7.5 0.875, by simp [beamRows], by simpa [row, beamReach] using hR⟩
  obtain ⟨r, hr⟩ := lookup_some hsome
  have hm := lookup_mem hr
  refine ⟨r, hm.1, ?_, lookup_lo beamRows_contiguous (beamRows_head_le hA) hr, hm.2⟩
  rw [beamRuleRow, ite_eq_left hA]
  exact hr

/-- Null below 0 and from 7 ft 6 in on, where the table sets no limit. -/
theorem beamRuleRow_none {A : ℝ} (h : A < 0 ∨ beamReach ≤ A) : beamRuleRow A = none := by
  unfold beamRuleRow
  rcases h with h | h
  · rw [ite_eq_right (not_le.2 h)]
  · by_cases h0 : 0 ≤ A
    · rw [ite_eq_left h0]
      exact lookup_none fun t ht => le_trans (beamRows_hi_le t ht) h
    · rw [ite_eq_right h0]

/-- The beam rule's `B` is non-decreasing in `A`: the row at `A` allows no more than the row at
any `A' ≥ A`. -/
theorem beam_mono {A A' : ℝ} {r r' : Row} (hA : 0 ≤ A) (hAA : A ≤ A')
    (hr : beamRuleRow A = some r) (hr' : beamRuleRow A' = some r') : r.B ≤ r'.B := by
  unfold beamRuleRow at hr hr'
  rw [ite_eq_left hA] at hr
  rw [ite_eq_left (le_trans hA hAA)] at hr'
  obtain ⟨r₀, h₀, hB⟩ := lookup_mono beamRows_before hAA hr'
  rw [hr] at h₀
  rw [Option.some.inj h₀]
  exact hB

/-- The beam rule against one obstruction as `governing` and `meets` judge it: within the table
the row's margin `max - B` is at least `-ε`; where `beamRuleRow` is null there is no limit. -/
def beamOK (eps A B : ℝ) : Prop := ∀ r, beamRuleRow A = some r → -eps ≤ r.B - B

/-- Moving a head away from an obstruction never makes the beam rule stricter. -/
theorem beam_pass_mono {eps A A' B : ℝ} (hA : 0 ≤ A) (hAA : A ≤ A') (h : beamOK eps A B) :
    beamOK eps A' B := by
  intro r' hr'
  have hA' : 0 ≤ A' := le_trans hA hAA
  obtain ⟨r, hr, hrr⟩ : ∃ r, beamRuleRow A = some r ∧ r.B ≤ r'.B := by
    unfold beamRuleRow at hr' ⊢
    rw [ite_eq_left hA'] at hr'
    rw [ite_eq_left hA]
    exact lookup_mono beamRows_before hAA hr'
  have := h r hr
  linarith

/-! ## `clearance` -/

/-- The beam part of `clearance`: the `from` of the first row whose `max` is at least `x`, else
`R` (`BEAM.rows.find((r) => r.max >= B - EPS)`, `row ? row.from : BEAM.reach`). -/
noncomputable def need (R x : ℝ) : List Row → ℝ
  | [] => R
  | s :: ss => if x ≤ s.B then s.lo else need R x ss

theorem need_ge {R x c : ℝ} {rs : List Row} (hrs : ∀ t ∈ rs, c ≤ t.lo) (hR : c ≤ R) :
    c ≤ need R x rs := by
  induction rs with
  | nil => simpa [need] using hR
  | cons s ss ih =>
    simp only [need]
    split_ifs
    · exact hrs s (List.mem_cons.2 (Or.inl rfl))
    · exact ih fun t ht => hrs t (List.mem_cons.2 (Or.inr ht))

theorem need_le {R x : ℝ} {rs : List Row} (h : ∀ t ∈ rs, t.lo ≤ R) : need R x rs ≤ R := by
  induction rs with
  | nil => simp [need]
  | cons s ss ih =>
    simp only [need]
    split_ifs
    · exact h s (List.mem_cons.2 (Or.inl rfl))
    · exact ih fun t ht => h t (List.mem_cons.2 (Or.inr ht))

/-- For a contiguous table in order, ending at or before `R` and starting at or below `A`: the row
found at `A` allows `x` exactly when `A` is at least `need R x`. -/
theorem need_iff {R x A : ℝ} {rs : List Row} (hP : rs.Pairwise Before) (hC : Contiguous rs)
    (hR : ∀ r ∈ rs, r.hi ≤ R) (hA : ∀ r ∈ rs.head?, r.lo ≤ A) {r : Row}
    (h : lookup rs A = some r) : x ≤ r.B ↔ need R x rs ≤ A := by
  induction rs with
  | nil => simp [lookup] at h
  | cons s ss ih =>
    rw [List.pairwise_cons] at hP
    have hC' := contiguous_cons hC
    have hsA : s.lo ≤ A := hA s (by simp)
    simp only [lookup] at h
    simp only [need]
    by_cases hAs : A < s.hi
    · rw [ite_eq_left hAs] at h
      rw [← Option.some.inj h]
      by_cases hx : x ≤ s.B
      · rw [ite_eq_left hx]
        exact ⟨fun _ => hsA, fun _ => hx⟩
      · rw [ite_eq_right hx]
        have hgt : s.hi ≤ need R x ss :=
          need_ge (fun t ht => (hP.1 t ht).2) (hR s (List.mem_cons.2 (Or.inl rfl)))
        exact ⟨fun h => absurd h hx, fun h => absurd (lt_of_lt_of_le hAs hgt) (not_lt.2 h)⟩
    · rw [ite_eq_right hAs] at h
      have hr := (lookup_mem h).1
      by_cases hx : x ≤ s.B
      · rw [ite_eq_left hx]
        exact ⟨fun _ => hsA, fun _ => le_trans hx (hP.1 r hr).1⟩
      · rw [ite_eq_right hx]
        refine ih hP.2 hC'.2 (fun t ht => hR t (List.mem_cons.2 (Or.inr ht))) ?_ h
        intro t ht
        rw [← hC'.1 t ht]
        exact not_lt.1 hAs

/-- `clearance`'s beam distance for a deflector `B` above the obstruction's bottom. -/
noncomputable def beamNeed (eps B : ℝ) : ℝ := need beamReach (B - eps) beamRows

/-- The beam rule passes at `A` exactly when `A` is at least `clearance`'s beam distance. -/
theorem clearance_beam {eps A B : ℝ} (hA : 0 ≤ A) : beamOK eps A B ↔ beamNeed eps B ≤ A := by
  by_cases hR : A < beamReach
  · obtain ⟨r, -, hr, -, -⟩ := beamRuleRow_row hA hR
    have hr' : lookup beamRows A = some r := by
      rw [beamRuleRow, ite_eq_left hA] at hr
      exact hr
    have key := need_iff (x := B - eps) beamRows_before beamRows_contiguous beamRows_hi_le
      (beamRows_head_le hA) hr'
    unfold beamOK beamNeed
    constructor
    · intro h
      exact key.1 (by linarith [h r hr])
    · intro h r' hr''
      rw [hr] at hr''
      rw [← Option.some.inj hr'']
      linarith [key.2 h]
  · have hnone : beamRuleRow A = none := beamRuleRow_none (Or.inr (not_lt.1 hR))
    unfold beamOK beamNeed
    refine ⟨fun _ => le_trans (need_le beamRows_lo_le) (not_lt.1 hR), fun _ r hr => ?_⟩
    rw [hnone] at hr
    cases hr

/-! ## The three times rule -/

/-- `Math.min(THREE.factor * dimension, THREE.cap)`: three times the obstruction's largest
dimension, never more than 24 in (0.61 m). -/
noncomputable def threeRequired (d : ℝ) : ℝ := min (3 * d) 0.61

/-- The three times rule as `governing` and `meets` judge it: margin `A - required ≥ -ε`. -/
def threeOK (eps A d : ℝ) : Prop := -eps ≤ A - threeRequired d

theorem three_times_mono : Monotone threeRequired :=
  fun _ _ h => min_le_min (by linarith) le_rfl

theorem three_times_le (d : ℝ) : threeRequired d ≤ 0.61 := min_le_right _ _

theorem three_times_nonneg {d : ℝ} (hd : 0 ≤ d) : 0 ≤ threeRequired d :=
  le_min (by linarith) (by norm_num)

/-- A head meeting the three times rule at distance `x` meets it at every larger distance. -/
theorem three_pass_mono {eps x x' d : ℝ} (h : threeOK eps x d) (hx : x ≤ x') :
    threeOK eps x' d := by
  unfold threeOK at *
  linarith

/-- ... and against any smaller obstruction. -/
theorem three_pass_dim {eps x d d' : ℝ} (h : threeOK eps x d') (hd : d ≤ d') :
    threeOK eps x d := by
  unfold threeOK at *
  linarith [three_times_mono hd]

/-- From 0.61 m on the rule always passes. -/
theorem three_times_far {eps A d : ℝ} (heps : 0 ≤ eps) (hA : 0.61 ≤ A) : threeOK eps A d := by
  unfold threeOK
  linarith [three_times_le d]

/-- `clearance`'s three times distance, `min(3 d, 0.61) - EPS`, is exact. -/
theorem clearance_three {eps A d : ℝ} : threeOK eps A d ↔ threeRequired d - eps ≤ A := by
  unfold threeOK
  constructor <;> intro h <;> linarith

/-- Both rules: a head passes exactly when `A` is at least the larger limit (`move`'s
`blocked`). -/
theorem clearance_exact {eps A B d : ℝ} (hA : 0 ≤ A) :
    beamOK eps A B ∧ threeOK eps A d ↔ max (beamNeed eps B) (threeRequired d - eps) ≤ A := by
  rw [clearance_beam hA, clearance_three, max_le_iff]

/-- `move`'s shortcut: the points nearer a convex footprint than a distance form a convex set, so
an obstruction that blocks every corner of the search rectangle blocks the whole rectangle. -/
theorem blocked_hull {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] {K : Set E}
    (hK : Convex ℝ K) (need : ℝ) {P : Set E} (hP : P ⊆ Metric.thickening need K) :
    convexHull ℝ P ⊆ Metric.thickening need K :=
  convexHull_min hP (hK.thickening need)

/-! ## `REACH`, `governing` and `meets` -/

/-- `REACH = Math.max(BEAM.reach, THREE.cap)`. -/
noncomputable def REACH : ℝ := max beamReach 0.61

theorem reach_eq : REACH = beamReach := max_eq_left (by norm_num [beamReach, FT])

/-- From `REACH` on both rules pass: skipping such obstructions changes no verdict. -/
theorem reach_skip {eps A : ℝ} (heps : 0 ≤ eps) (hA : REACH ≤ A) (B d : ℝ) :
    beamOK eps A B ∧ threeOK eps A d := by
  rw [reach_eq] at hA
  have h61 : (0.61 : ℝ) ≤ beamReach := by norm_num [beamReach, FT]
  refine ⟨fun r hr => ?_, three_times_far heps (le_trans h61 hA)⟩
  rw [beamRuleRow_none (Or.inr hA)] at hr
  cases hr

/-- `governing` keeps the obstruction of least margin: it meets `-ε` exactly when every
obstruction's margin does. -/
theorem least_margin_iff {ι : Type*} {S : Finset ι} (hS : S.Nonempty) (margin : ι → ℝ)
    (eps : ℝ) : -eps ≤ S.inf' hS margin ↔ ∀ o ∈ S, -eps ≤ margin o := by
  rw [Finset.le_inf'_iff]

/-- `meets`: every obstruction the beam rule takes (`Sb`) and every one the three times rule takes
(`St`) passes, at plan distances `A o`, heights `B o` and largest dimensions `d o`. -/
def Meets {ι : Type*} (eps : ℝ) (Sb St : Finset ι) (A B d : ι → ℝ) : Prop :=
  (∀ o ∈ Sb, beamOK eps (A o) (B o)) ∧ ∀ o ∈ St, threeOK eps (A o) (d o)

/-- A head moved so that it is no nearer any obstruction keeps meeting both rules. -/
theorem meets_mono {ι : Type*} {eps : ℝ} {Sb St : Finset ι} {A A' B d : ι → ℝ}
    (hA : ∀ o, 0 ≤ A o) (hAA : ∀ o, A o ≤ A' o) (h : Meets eps Sb St A B d) :
    Meets eps Sb St A' B d :=
  ⟨fun o ho => beam_pass_mono (hA o) (hAA o) (h.1 o ho),
    fun o ho => three_pass_mono (h.2 o ho) (hAA o)⟩

end FireEgress.Obstructions
