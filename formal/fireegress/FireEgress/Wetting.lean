import Mathlib
import FireEgress.Suppression

/-!
# Wetting: the sprinkler water that soaks into the fuel

Specifications for the "wetting" mode of the spray, in `spray` of `src/core/flamescapes.js` and
the WGSL `spray` kernel of `src/core/flamescapesGpu/kernels.js` (the mode's description is in the
header of `src/core/suppression.js`). Down each column the water evaporates as in the spray mode
(Suppression.lean, item 8); where the column ends on a fuel cell's top surface, a share
`wet = WETTED_FRACTION` of the water left soaks into that cell's surface layer, `moisture += dw`
with `dw = wet m`, mixed in at the water's temperature:
`Ts' = (C Ts + dw c_w T_w) / (C + dw c_w)`, `C = solidCapacity` before the water, `c_w = 4186`,
`T_w = 293.15 K`. The rest runs off to the floor. Real arithmetic; it reuses Suppression.lean's
`column_budget` and `column_nonneg`.

## Which theorem backs which part of the code

* The mixing step (`wetTs`): `wet_energy` (sensible energy is conserved exactly,
  `(C + dw c_w) Ts' = C Ts + dw c_w T_w`), `wet_sub_water` and `wet_sub_surface` (the surface moves
  toward the water's temperature by the share `dw c_w / (C + dw c_w)` of the gap), `wet_between`
  (`Ts'` lies between `T_w` and `Ts`), `wet_cools` (for `Ts ≥ T_w` the surface never warms and never
  falls below the water's temperature), `wet_zero` (no water, no change).
* The cell's heat capacity (`solidCapacity`, `inertC + skin cp + moisture 4186`, the CPU's
  `solidCapacity` and the GPU's `capOf`): `wet_capacity` (adding `dw` to the moisture adds exactly
  `dw c_w`, the `C + dw c_w` of the mixing step), `wet_energy_cell` (so the layer, with its water,
  keeps its energy: capacity after times `Ts'` equals capacity before times `Ts` plus the water's
  `dw c_w T_w`).
* The water taken (`dw = wet m`): `wet_bounds` (`0 ≤ dw ≤ m` for `0 ≤ wet ≤ 1`),
  `wetted_fraction_mem` (the code's 0.52 lies in `[0, 1]`).
* The column budget with wetting (`stats.emitted`, `evaporated`, `wetted`, `floor`):
  `column_budget_wet` (one column: emitted = evaporated + wetted + at the floor, Suppression.lean's
  `column_budget` with one final wetting term, each part non-negative; a column that does not end on
  fuel is the case `wet = 0`) and `spray_budget_wet` (summed over the columns, as the code sums its
  statistics).

## Hypotheses, and what is not covered

* `C > 0` (a fuel cell's surface layer has heat capacity), `dw ≥ 0`, `c_w ≥ 0` (`> 0` is not
  needed); `0 ≤ wet ≤ 1`, which `createSuppression` does not check for a caller's own `wet`.
* While water remains after the solver's drying step the surface stays at or below boiling: that is
  Suppression.lean's `wet_skin_pinned`, which the wetted moisture enters unchanged.
* Not covered: floating point (the GPU's float32), the per-cell evaporation share (a parameter, as
  in Suppression.lean), the drag of the falling water on the gas, the calibration of
  `WETTED_FRACTION` against Evans's decay (a measurement, `tools/wetting-calibrate.mjs`), and which
  cell a column ends on (`sprinklerSetup`'s geometry).
-/

namespace FireEgress.Wetting

/-- `WATER_CP`, liquid water's specific heat (J/(kg K)). -/
noncomputable def WATER_CP : ℝ := 4186

/-- `WATER_TEMPERATURE`, the sprinkler water's temperature (K). -/
noncomputable def WATER_TEMPERATURE : ℝ := 293.15

/-- `WETTED_FRACTION`, the share of the water left that soaks in. -/
noncomputable def WETTED_FRACTION : ℝ := 0.52

/-! ## The mixing step -/

/-- The surface layer's temperature after `dw` of water at `Tw` soaks in:
`Ts' = (C Ts + dw c_w T_w) / (C + dw c_w)`. -/
noncomputable def wetTs (C Ts dw cw Tw : ℝ) : ℝ := (C * Ts + dw * cw * Tw) / (C + dw * cw)

/-- Sensible energy is conserved exactly. -/
theorem wet_energy {C Ts dw cw Tw : ℝ} (hC : 0 < C) (hdw : 0 ≤ dw) (hcw : 0 ≤ cw) :
    (C + dw * cw) * wetTs C Ts dw cw Tw = C * Ts + dw * cw * Tw := by
  have hne : C + dw * cw ≠ 0 := (by positivity : (0 : ℝ) < C + dw * cw).ne'
  unfold wetTs
  exact mul_div_cancel₀ _ hne

/-- The new temperature less the water's: the old gap scaled by `C / (C + dw c_w)`. -/
theorem wet_sub_water {C Ts dw cw Tw : ℝ} (hC : 0 < C) (hdw : 0 ≤ dw) (hcw : 0 ≤ cw) :
    wetTs C Ts dw cw Tw - Tw = C * (Ts - Tw) / (C + dw * cw) := by
  have hne : C + dw * cw ≠ 0 := (by positivity : (0 : ℝ) < C + dw * cw).ne'
  unfold wetTs
  field_simp
  ring

/-- The old temperature less the new: the gap scaled by `dw c_w / (C + dw c_w)`. -/
theorem wet_sub_surface {C Ts dw cw Tw : ℝ} (hC : 0 < C) (hdw : 0 ≤ dw) (hcw : 0 ≤ cw) :
    Ts - wetTs C Ts dw cw Tw = dw * cw * (Ts - Tw) / (C + dw * cw) := by
  have hne : C + dw * cw ≠ 0 := (by positivity : (0 : ℝ) < C + dw * cw).ne'
  unfold wetTs
  field_simp
  ring

/-- The new temperature lies between the water's and the old. -/
theorem wet_between {C Ts dw cw Tw : ℝ} (hC : 0 < C) (hdw : 0 ≤ dw) (hcw : 0 ≤ cw) :
    min Tw Ts ≤ wetTs C Ts dw cw Tw ∧ wetTs C Ts dw cw Tw ≤ max Tw Ts := by
  have hpos : 0 < C + dw * cw := by positivity
  have hw : 0 ≤ dw * cw := mul_nonneg hdw hcw
  have h1 := mul_le_mul_of_nonneg_left (min_le_right Tw Ts) hC.le
  have h2 := mul_le_mul_of_nonneg_left (min_le_left Tw Ts) hw
  have h3 := mul_le_mul_of_nonneg_left (le_max_right Tw Ts) hC.le
  have h4 := mul_le_mul_of_nonneg_left (le_max_left Tw Ts) hw
  unfold wetTs
  constructor
  · rw [le_div_iff₀ hpos]
    nlinarith
  · rw [div_le_iff₀ hpos]
    nlinarith

/-- Water no warmer than the surface never warms it, and never cools it below the water's own
temperature. -/
theorem wet_cools {C Ts dw cw Tw : ℝ} (hC : 0 < C) (hdw : 0 ≤ dw) (hcw : 0 ≤ cw) (hT : Tw ≤ Ts) :
    Tw ≤ wetTs C Ts dw cw Tw ∧ wetTs C Ts dw cw Tw ≤ Ts := by
  have hpos : 0 < C + dw * cw := by positivity
  have hg : 0 ≤ Ts - Tw := sub_nonneg.2 hT
  have h1 := wet_sub_water (Tw := Tw) (Ts := Ts) hC hdw hcw
  have h2 := wet_sub_surface (Tw := Tw) (Ts := Ts) hC hdw hcw
  have h3 : 0 ≤ C * (Ts - Tw) / (C + dw * cw) := div_nonneg (mul_nonneg hC.le hg) hpos.le
  have h4 : 0 ≤ dw * cw * (Ts - Tw) / (C + dw * cw) :=
    div_nonneg (mul_nonneg (mul_nonneg hdw hcw) hg) hpos.le
  constructor <;> linarith

/-- No water, no change. -/
theorem wet_zero {C Ts cw Tw : ℝ} (hC : C ≠ 0) : wetTs C Ts 0 cw Tw = Ts := by
  unfold wetTs
  simp only [zero_mul, add_zero]
  exact mul_div_cancel_left₀ Ts hC

/-! ## The cell's heat capacity -/

/-- `solidCapacity`: the inert part, the fuel's surface layer and its water (J/K). -/
noncomputable def solidCapacity (inertC skin cp moist : ℝ) : ℝ :=
  inertC + skin * cp + moist * WATER_CP

/-- Adding `dw` of moisture adds exactly `dw c_w`. -/
theorem wet_capacity (inertC skin cp moist dw : ℝ) :
    solidCapacity inertC skin cp (moist + dw) = solidCapacity inertC skin cp moist + dw * WATER_CP := by
  unfold solidCapacity
  ring

/-- The layer with its water keeps its energy: the capacity after the water times the new
temperature is the capacity before times the old plus the water's `dw c_w T_w`. -/
theorem wet_energy_cell {inertC skin cp moist dw Ts : ℝ}
    (hC : 0 < solidCapacity inertC skin cp moist) (hdw : 0 ≤ dw) :
    solidCapacity inertC skin cp (moist + dw) *
        wetTs (solidCapacity inertC skin cp moist) Ts dw WATER_CP WATER_TEMPERATURE =
      solidCapacity inertC skin cp moist * Ts + dw * WATER_CP * WATER_TEMPERATURE := by
  rw [wet_capacity]
  exact wet_energy hC hdw (by norm_num [WATER_CP])

/-! ## The water taken -/

/-- A share `w ∈ [0, 1]` of what is left: `0 ≤ dw ≤ m`. -/
theorem wet_bounds {w m : ℝ} (hw0 : 0 ≤ w) (hw1 : w ≤ 1) (hm : 0 ≤ m) : 0 ≤ w * m ∧ w * m ≤ m :=
  ⟨mul_nonneg hw0 hm, mul_le_of_le_one_left hm hw1⟩

theorem wetted_fraction_mem : 0 ≤ WETTED_FRACTION ∧ WETTED_FRACTION ≤ 1 := by
  norm_num [WETTED_FRACTION]

/-! ## The column budget with wetting -/

/-- One column: the water emitted at the top is the water evaporated down the column, plus the
water soaked into the fuel it ends on, plus the water left at the floor; each part is
non-negative. -/
theorem column_budget_wet {m e : ℕ → ℝ} {w : ℝ} (hstep : ∀ j, m (j + 1) = m j - e j) {n : ℕ}
    (he : ∀ j < n, 0 ≤ e j ∧ e j ≤ m j) (h0 : 0 ≤ m 0) (hw0 : 0 ≤ w) (hw1 : w ≤ 1) :
    m 0 = ∑ j ∈ Finset.range n, e j + w * m n + (m n - w * m n) ∧
      0 ≤ ∑ j ∈ Finset.range n, e j ∧ 0 ≤ w * m n ∧ 0 ≤ m n - w * m n := by
  have hb := Suppression.column_budget hstep n
  have hn := Suppression.column_nonneg hstep he h0 n le_rfl
  obtain ⟨h1, h2⟩ := wet_bounds hw0 hw1 hn
  exact ⟨by linarith, Finset.sum_nonneg fun j hj => (he j (Finset.mem_range.1 hj)).1, h1,
    by linarith⟩

/-- All columns, as the code sums its statistics: emitted = evaporated + wetted + at the
floor. -/
theorem spray_budget_wet {κ : Type*} {K : Finset κ} {m e : κ → ℕ → ℝ} {n : κ → ℕ} {w : κ → ℝ}
    (hstep : ∀ k ∈ K, ∀ j, m k (j + 1) = m k j - e k j)
    (he : ∀ k ∈ K, ∀ j < n k, 0 ≤ e k j ∧ e k j ≤ m k j) (h0 : ∀ k ∈ K, 0 ≤ m k 0)
    (hw0 : ∀ k ∈ K, 0 ≤ w k) (hw1 : ∀ k ∈ K, w k ≤ 1) :
    ∑ k ∈ K, m k 0 = ∑ k ∈ K, ∑ j ∈ Finset.range (n k), e k j + ∑ k ∈ K, w k * m k (n k) +
        ∑ k ∈ K, (m k (n k) - w k * m k (n k)) ∧
      0 ≤ ∑ k ∈ K, ∑ j ∈ Finset.range (n k), e k j ∧ 0 ≤ ∑ k ∈ K, w k * m k (n k) ∧
      0 ≤ ∑ k ∈ K, (m k (n k) - w k * m k (n k)) := by
  have one := fun k (hk : k ∈ K) =>
    column_budget_wet (hstep k hk) (he k hk) (h0 k hk) (hw0 k hk) (hw1 k hk)
  refine ⟨?_, Finset.sum_nonneg fun k hk => (one k hk).2.1,
    Finset.sum_nonneg fun k hk => (one k hk).2.2.1, Finset.sum_nonneg fun k hk => (one k hk).2.2.2⟩
  rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun k hk => (one k hk).1

end FireEgress.Wetting
