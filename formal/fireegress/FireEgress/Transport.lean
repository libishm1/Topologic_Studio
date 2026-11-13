import Mathlib

/-!
# The heat-conserving transport (flamescapes.js `transportGas`, proposal F)

The fire solver carried temperature `T` by back-tracing and took the density from it
by the gas law at constant pressure, `ρ T = k`. Gases mix by volume at constant
pressure, so it is the density that mixes linearly, not `T`
(wiki/research/fire-calibration-claim-protocol-2026-11-12.md, section 1). This
module proves the facts the change rests on:

* `mix_conserves_mass`, `mix_conserves_enthalpy`: mixing volumes `v i` of densities
  `ρ i` into one volume of their volume-weighted mean density keeps the mass and,
  with `ρ T = k`, the enthalpy `∑ c ρ T v`;
* `arith_T_ge_mixed_T`: the old transport's volume-weighted arithmetic mean of `T`
  is at least the mixed gas's temperature `k / ρ̄` (Cauchy–Schwarz, the weighted
  AM–HM inequality), so averaging `T` made heat, and never lost it;
* `flux_line_conserves`: what the faces of a line of cells carry sums to nothing
  when its two ends carry nothing (a telescoping sum): the flux form makes no mass;
* `giveback_uniform`: a cell of a uniform gas keeps its density whatever its
  faces' divergence, once it gives back its density times that divergence;
* `giveback_start_density` and `giveback_current_density_errs`: over the three
  axis sweeps, giving back the starting density times each sweep's divergence adds
  nothing when the flow is divergence-free, and the current density can;
* `expansion_volume`: the divergence `ln (T1 / T0) / dt` over a step expands a
  volume by exactly `T1 / T0`, the gas law's ratio at constant pressure;
* `substeps_courant`: cutting a step `dt` into `n ≥ s dt / (C h)` pieces moves gas
  at speed `s` at most `C` cells a piece (`substepsFor`);
* `calibrated_indicative`: a quantity in the calibrated band is in the indicative band;
* `burn_heat_le_oxygen`, `burner_pending_closes`: a burner cell releases no more heat
  than its oxygen allows, and what it does not burn waits, so none is lost.

Floating point is not covered: `flamescapesConservation.test.mjs` checks the
sealed box's heat and mass to 1 % and 0.5 %, and `heat-closure.mjs` to 0.1 %.
-/

namespace FireEgress.Transport

open Finset

variable {ι : Type*}

/-- The volume-weighted mean density of the volumes `v` of densities `ρ`. -/
noncomputable def mixedDensity (s : Finset ι) (v ρ : ι → ℝ) : ℝ :=
  (∑ i ∈ s, v i * ρ i) / ∑ i ∈ s, v i

/-- Mixing keeps the mass. -/
theorem mix_conserves_mass (s : Finset ι) (v ρ : ι → ℝ) (hv : ∑ i ∈ s, v i ≠ 0) :
    mixedDensity s v ρ * ∑ i ∈ s, v i = ∑ i ∈ s, v i * ρ i := by
  unfold mixedDensity
  field_simp

/-- At constant pressure (`ρ T = k` in every volume and in the mixture), mixing keeps
the enthalpy `∑ c ρ T v`: it is `c k` times the total volume before and after. -/
theorem mix_conserves_enthalpy (s : Finset ι) (v ρ T : ι → ℝ) (c k : ℝ)
    (hv : ∑ i ∈ s, v i ≠ 0) (hgas : ∀ i ∈ s, ρ i * T i = k)
    (hmix : mixedDensity s v ρ ≠ 0) :
    c * mixedDensity s v ρ * (k / mixedDensity s v ρ) * ∑ i ∈ s, v i
      = ∑ i ∈ s, c * ρ i * T i * v i := by
  have hl : c * mixedDensity s v ρ * (k / mixedDensity s v ρ) * ∑ i ∈ s, v i
      = c * k * ∑ i ∈ s, v i := by
    field_simp
  have hr : ∑ i ∈ s, c * ρ i * T i * v i = ∑ i ∈ s, c * k * v i := by
    refine Finset.sum_congr rfl (fun i hi => ?_)
    rw [mul_assoc c, hgas i hi]
  rw [hl, hr, ← Finset.mul_sum]

/-- Cauchy–Schwarz for positive volumes and densities: `(∑ v)² ≤ (∑ v / ρ) (∑ v ρ)`. -/
theorem sum_sq_le (s : Finset ι) (v ρ : ι → ℝ) (hv : ∀ i ∈ s, 0 ≤ v i)
    (hρ : ∀ i ∈ s, 0 < ρ i) :
    (∑ i ∈ s, v i) ^ 2 ≤ (∑ i ∈ s, v i / ρ i) * ∑ i ∈ s, v i * ρ i := by
  have key : ∀ i ∈ s, Real.sqrt (v i / ρ i) * Real.sqrt (v i * ρ i) = v i := by
    intro i hi
    have h1 : 0 ≤ v i / ρ i := div_nonneg (hv i hi) (hρ i hi).le
    rw [← Real.sqrt_mul h1, show v i / ρ i * (v i * ρ i) = v i ^ 2 by
      field_simp [(hρ i hi).ne']]
    exact Real.sqrt_sq (hv i hi)
  have cs := Finset.sum_mul_sq_le_sq_mul_sq s (fun i => Real.sqrt (v i / ρ i))
    (fun i => Real.sqrt (v i * ρ i))
  rw [Finset.sum_congr rfl key] at cs
  have e1 : ∀ i ∈ s, Real.sqrt (v i / ρ i) ^ 2 = v i / ρ i := fun i hi =>
    Real.sq_sqrt (div_nonneg (hv i hi) (hρ i hi).le)
  have e2 : ∀ i ∈ s, Real.sqrt (v i * ρ i) ^ 2 = v i * ρ i := fun i hi =>
    Real.sq_sqrt (mul_nonneg (hv i hi) (hρ i hi).le)
  rwa [Finset.sum_congr rfl e1, Finset.sum_congr rfl e2] at cs

/-- The old transport averaged `T` by volume: its mean is at least the temperature
`k / ρ̄` the mixed gas has, so every mixing step it made heat (and never lost it). -/
theorem arith_T_ge_mixed_T (s : Finset ι) (v ρ T : ι → ℝ) (k : ℝ) (hk : 0 ≤ k)
    (hv : ∀ i ∈ s, 0 ≤ v i) (hV : 0 < ∑ i ∈ s, v i) (hρ : ∀ i ∈ s, 0 < ρ i)
    (hgas : ∀ i ∈ s, T i = k / ρ i) :
    k / mixedDensity s v ρ ≤ (∑ i ∈ s, v i * T i) / ∑ i ∈ s, v i := by
  have hM : 0 < ∑ i ∈ s, v i * ρ i := by
    obtain ⟨j, hj, hvj⟩ : ∃ j ∈ s, 0 < v j := by
      by_contra h
      have h' : ∀ i ∈ s, v i ≤ 0 := fun i hi => not_lt.mp (fun hlt => h ⟨i, hi, hlt⟩)
      have : ∑ i ∈ s, v i ≤ 0 := Finset.sum_nonpos h'
      linarith
    exact lt_of_lt_of_le (mul_pos hvj (hρ j hj))
      (Finset.single_le_sum (fun i hi => mul_nonneg (hv i hi) (hρ i hi).le) hj)
  have hT : ∑ i ∈ s, v i * T i = k * ∑ i ∈ s, v i / ρ i := by
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl (fun i hi => by rw [hgas i hi]; ring)
  unfold mixedDensity
  rw [hT, div_div_eq_mul_div, div_le_div_iff₀ hM hV]
  have := sum_sq_le s v ρ hv hρ
  nlinarith [this, hk]

/-- A line of cells: face `i` carries `F i` into cell `i` from cell `i - 1`, so cell
`i` changes by `F i - F (i + 1)`. With nothing through the two ends, the line's
total does not change: the flux form makes no mass. -/
theorem flux_line_conserves (F : ℕ → ℝ) (n : ℕ) (h0 : F 0 = 0) (hn : F n = 0) :
    ∑ i ∈ range n, (F i - F (i + 1)) = 0 := by
  rw [Finset.sum_range_sub', h0, hn, sub_zero]

/-- A cell of a uniform gas, density `ρ`: whatever its faces carry in (`cin`) and out
(`cout`) as Courant numbers, the flux difference and the give-back of `ρ` times their
divergence cancel, so it keeps its density. -/
theorem giveback_uniform (ρ cin cout : ℝ) :
    ρ + (cin * ρ - cout * ρ) + ρ * (cout - cin) = ρ := by
  ring

/-- Over the three sweeps, giving back the starting density `ρ0` times each sweep's
divergence adds nothing when the flow is divergence-free. -/
theorem giveback_start_density (ρ0 dx dy dz : ℝ) (hdiv : dx + dy + dz = 0) :
    ρ0 * dx + ρ0 * dy + ρ0 * dz = 0 := by
  rw [← mul_add, ← mul_add, hdiv, mul_zero]

/-- Giving back each sweep's current density instead can add mass even when the flow
is divergence-free: a 5 % splitting error in the sealed box (2026-11-12). -/
theorem giveback_current_density_errs :
    ∃ ρ0 ρ1 ρ2 dx dy dz : ℝ, dx + dy + dz = 0 ∧ ρ0 * dx + ρ1 * dy + ρ2 * dz ≠ 0 :=
  ⟨1, 2, 2, 1, -1, 0, by norm_num, by norm_num⟩

/-- The divergence `ln (T1 / T0) / dt` held over a step `dt` expands a volume `V0` to
`V0 T1 / T0`, the gas law's volume at constant pressure. -/
theorem expansion_volume (V0 T0 T1 dt : ℝ) (h0 : 0 < T0) (h1 : 0 < T1) (hdt : dt ≠ 0) :
    V0 * Real.exp (Real.log (T1 / T0) / dt * dt) = V0 * (T1 / T0) := by
  rw [div_mul_cancel₀ _ hdt, Real.exp_log (div_pos h1 h0)]

/-- `substepsFor`: a step `dt` cut into `n` pieces, `n` at least `s dt / (C h)` and
positive, moves gas at speed `s` at most `C` cells of size `h` a piece. -/
theorem substeps_courant (s dt C h n : ℝ) (hC : 0 < C)
    (hh : 0 < h) (hn : 0 < n) (hcut : s * dt / (C * h) ≤ n) :
    s * (dt / n) / h ≤ C := by
  have hCh : 0 < C * h := mul_pos hC hh
  rw [div_le_iff₀ hCh] at hcut
  rw [div_le_iff₀ hh, mul_div_assoc', div_le_iff₀ hn]
  nlinarith [hcut]

/-- This project's bands (the 2026-11-08 plan, section 8, on the FDS Validation Guide's
statistics): a quantity within the calibrated band (bias 0.8 to 1.25, scatter at most
0.35) is within the indicative band (0.67 to 1.5, at most 0.5). -/
theorem calibrated_indicative (δ σ : ℝ) (h : 0.8 ≤ δ ∧ δ ≤ 1.25 ∧ σ ≤ 0.35) :
    0.67 ≤ δ ∧ δ ≤ 1.5 ∧ σ ≤ 0.5 := by
  obtain ⟨h1, h2, h3⟩ := h
  exact ⟨by linarith, by linarith, by linarith⟩

/-- A burner cell burns at most what its oxygen allows (flamescapes.js `burnInCell`,
2026-11-12): offered `f` kg of fuel that needs `o2` kg of oxygen a kg, with `M` kg of
oxygen in the cell, it burns `min f (M / o2)`; at Huggett's `q` joules a kg of oxygen
(heat of combustion `o2 * q`), the heat it releases is at most `M * q`. -/
theorem burn_heat_le_oxygen (f M o2 q : ℝ) (ho : 0 < o2) (hq : 0 ≤ q) :
    min f (M / o2) * (o2 * q) ≤ M * q := by
  have hoq : 0 ≤ o2 * q := mul_nonneg ho.le hq
  calc min f (M / o2) * (o2 * q) ≤ (M / o2) * (o2 * q) :=
        mul_le_mul_of_nonneg_right (min_le_right _ _) hoq
    _ = M * q := by field_simp

/-- What a burner step does not burn waits and is offered again (`pending`): with
`E` offered and `b` burnt, released `b` and pending `E - b` make up `E`. -/
theorem burner_pending_closes (E b : ℝ) : b + (E - b) = E := by ring

end FireEgress.Transport
