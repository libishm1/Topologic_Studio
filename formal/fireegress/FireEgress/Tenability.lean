import Mathlib
import FireEgress.Dim

/-!
# Tenability and egress movement (T1-T17, E1-E11)

The literature review (`wiki/research/fire-egress-methods-2026-10-27.md`, sections 4, 5.4,
6 and 8.4) collects the tenability model Studio would use (ISO 13571:2012 fractional
effective dose, ASET, RSET, the safety margin, the tenability window) and the movement
model (SFPE hydraulic speed-density law, specific flow, Weidmann's law). The ids (T-nn,
E-nn) are those of the adversarially checked extraction.

Two kinds of statement are proved.

* **Dimensions** (section `Units`, all by `decide`). Movement needs a person count, which
  `FireEgress.Dim` has no slot for, so `PD` below adds a person exponent (scaled by 60 like
  the others). Empirical constants carry the units their correlation implies, and each
  statement names that unit. Three things a dimension check cannot see are stated as
  findings rather than hidden: concentrations (ppm, %, uL/L) are all dimension one;
  minutes and seconds are both `time`; and kelvin and degree-Celsius intervals are both
  `temp`, so the kelvin trap of T3 is invisible to `Dim` and is proved numerically
  instead (`T3_kelvin_trap`).
* **Mathematics** (real analysis). FED accumulation (discrete and continuous), the
  convective and radiant tolerance laws, the hydraulic and Weidmann speed laws, the
  margin, and the FIFO tenability window.

Units in the real theorems: times to incapacitation in min, temperatures in degC, radiant
flux in kW m⁻², densities in persons m⁻², speeds in m s⁻¹, as in the sources.
-/

namespace FireEgress.Tenability

open Real Filter Topology MeasureTheory

/-! ## Units -/

/-- A dimension together with an exponent of "person" (a counted quantity), scaled by 60
like the exponents of `Dim`. -/
structure PD where
  d : Dim
  P : Int
  deriving DecidableEq

namespace PD

instance : Mul PD := ⟨fun a b => ⟨a.d * b.d, a.P + b.P⟩⟩
instance : Inv PD := ⟨fun a => ⟨a.d⁻¹, -a.P⟩⟩
instance : Div PD := ⟨fun a b => a * b⁻¹⟩

/-- Embed an SI dimension with no person exponent. -/
def of (d : Dim) : PD := ⟨d, 0⟩
def one : PD := of Dim.one
def person : PD := ⟨Dim.one, 60⟩
def length : PD := of Dim.length
def area : PD := of Dim.area
def time : PD := of Dim.time
def velocity : PD := of Dim.velocity
/-- Occupant density, persons m⁻². -/
def density : PD := person / area
/-- Specific flow, persons s⁻¹ m⁻¹. -/
def specFlow : PD := person / (time * length)
/-- Flow, persons s⁻¹. -/
def flow : PD := person / time

end PD

/-- A concentration (uL/L, ppm or volume %) is a ratio of volumes: dimension one. Hence
`Dim` cannot tell ppm from %, and the factor 10⁴ between them is invisible to it. -/
def conc : Dim := Dim.volume / Dim.volume

theorem units_conc_dimensionless : conc = Dim.one := by decide

/-- T1, T11. A FED increment `Δt / t_I` is dimensionless (min / min). -/
theorem units_T1_increment : Dim.time / Dim.time = Dim.one := by decide

/-- T1, ISO Eq (1). The concentration form `C_i Δt / (C t)_i` is dimensionless, with
`C_i` a concentration and `(C t)_i` in concentration × min. -/
theorem units_T1_iso_eq1 : conc * Dim.time / (conc * Dim.time) = Dim.one := by decide

/-- T3, T11. The constant 5 × 10⁷ of `t = 5e7 T^(-3.4)` carries min · degC^(17/5): with
that unit the law gives a time, and `T^(17/5)` is exactly typeable (3.4 = 17/5, and
60 · 17 / 5 = 204). In the note's normalised form `t/(1 min) = 5e7 (T / 1 degC)^(-3.4)`
every factor is dimension one. -/
theorem units_T3_constant :
    Dim.exactPow Dim.temp 17 5 ∧ Dim.exactPow Dim.temp (-17) 5 ∧
      (Dim.time * Dim.rpow Dim.temp 17 5) * Dim.rpow Dim.temp (-17) 5 = Dim.time ∧
      Dim.one * Dim.rpow Dim.one (-17) 5 = Dim.one := by decide

/-- T4, T5, T11, finding. The exponents 3.61 = 361/100 (clothed convective law) and
1.56 = 39/25 (ISO second-degree burn law) are *not* typeable in `Dim` (exponents scaled by
60): `temp^3.61` and `(kW m⁻²)^1.56` are not exact. These laws can only be checked on their
normalised (dimension-one) form, `t/(1 min) = c (x / 1 unit)^(-p)`. -/
theorem units_T4_T5_untypeable :
    ¬ Dim.exactPow Dim.temp 361 100 ∧ ¬ Dim.exactPow (Dim.power / Dim.area) 39 25 := by
  decide

/-- T5. Radiant flux (kW m⁻²) has dimension M T⁻³. Purser's `t = r / q^(4/3)` gives a time
when `r` carries (kW m⁻²)^(4/3) min, and ISO's pain law `t = 4.2 q^(-1.9)` gives a time when
4.2 carries min (kW m⁻²)^(19/10); both powers are exact. -/
theorem units_T5_radiant :
    Dim.power / Dim.area = ⟨60, 0, -180, 0⟩ ∧
    Dim.exactPow (Dim.power / Dim.area) 4 3 ∧ Dim.exactPow (Dim.power / Dim.area) 19 10 ∧
    (Dim.time * Dim.rpow (Dim.power / Dim.area) 4 3) / Dim.rpow (Dim.power / Dim.area) 4 3
      = Dim.time ∧
    (Dim.time * Dim.rpow (Dim.power / Dim.area) 19 10) *
      Dim.rpow (Dim.power / Dim.area) (-19) 10 = Dim.time := by decide

/-- T6. The combined heat FED `(1/t_rad + 1/t_conv) Δt` adds two rates of the same
dimension (min⁻¹) and is dimensionless. -/
theorem units_T6_heat : (Dim.one / Dim.time) * Dim.time = Dim.one := by decide

/-- T7. With concentrations of dimension one, 35 000 (uL/L · min) and 1.2 × 10⁶
((uL/L)^2.36 · min) both have dimension `time`, and both asphyxiant increments
`φ ν Δt / 35000` and `(φ ν)^2.36 Δt / 1.2e6` are dimensionless. -/
theorem units_T7_asphyxiant :
    conc * Dim.one * Dim.time / (conc * Dim.time) = Dim.one ∧
    Dim.exactPow (conc * Dim.one) 59 25 ∧
    Dim.rpow (conc * Dim.one) 59 25 * Dim.time / (Dim.rpow conc 59 25 * Dim.time)
      = Dim.one := by decide

/-- T8. In `ν = exp(φ_CO2 / 5)` both `φ` and the 5 are volume percentages, so the exponent
is dimensionless. -/
theorem units_T8_exponent : conc / conc = Dim.one := by decide

/-- T10. The route-dose integrand `dt / t_I` with `t` the travel clock (s) and `t_I` in min
is dimensionless as far as `Dim` can tell: the factor 1/60 is a scale, not a dimension,
so the s/min trap of T10 is outside what dimensional analysis checks. Edge traversal time
`τ = ℓ / S` is a time. -/
theorem units_T10 :
    Dim.time / Dim.time = Dim.one ∧ Dim.length / Dim.velocity = Dim.time := by decide

/-- E1. In `S = k (1 - a D)`: `a` (m² person⁻¹) times `D` (persons m⁻²) is dimension one,
so `1 - a D` is well formed, and `S` has the dimension of `k` (m s⁻¹). The jam density
`1/a` is a density. -/
theorem units_E1 :
    (PD.area / PD.person) * PD.density = PD.one ∧
    PD.velocity * PD.one = PD.velocity ∧
    PD.one / (PD.area / PD.person) = PD.density := by decide

/-- E4, E5. Specific flow `F_s = S D` is persons s⁻¹ m⁻¹; the optimum density
`D* = 1/(2a)` is a density and `F_s,max = k/(4a)` is a specific flow. -/
theorem units_E4_E5 :
    PD.velocity * PD.density = PD.specFlow ∧
    PD.velocity / (PD.area / PD.person) = PD.specFlow := by decide

/-- E6. Flow `F = F_s W_e` is persons s⁻¹. -/
theorem units_E6 : PD.specFlow * PD.length = PD.flow := by decide

/-- E10. Arc capacity `F_s,max W_e Δt` is a number of persons, node storage `A_v D_max`
is a number of persons, and transit `ℓ / (S Δt)` is a number of steps. -/
theorem units_E10 :
    PD.specFlow * PD.length * PD.time = PD.person ∧
    PD.area * PD.density = PD.person ∧
    PD.length / (PD.velocity * PD.time) = PD.one := by decide

/-- E8. In Weidmann's `v = v₀ (1 - exp(-γ (1/ρ - 1/ρ_max)))`, `γ` must be a density
(persons m⁻²) for the exponent to be dimension one; `v` has the dimension of `v₀`. -/
theorem units_E8 :
    PD.density * (PD.one / PD.density) = PD.one ∧ PD.velocity * PD.one = PD.velocity := by
  decide

/-- E9. In single file `d = 1/ρ_1D` with `ρ_1D` in persons m⁻¹, so `d` is m person⁻¹;
in `d = 0.36 + 1.06 v` the 0.36 carries m person⁻¹ and the 1.06 carries s person⁻¹. -/
theorem units_E9 :
    PD.one / (PD.person / PD.length) = PD.length / PD.person ∧
    (PD.time / PD.person) * PD.velocity = PD.length / PD.person := by decide

/-- E11, T13, T14. The A* heuristic `‖x‖ / v_max` is a time, as are RSET's components and
the margin `ASET - RSET`. -/
theorem units_E11 : PD.length / PD.velocity = PD.time := by decide

/-! ## T1. Fractional effective dose, discrete form -/

/-- T1, ISO Eq (1a). Discrete FED after `n` increments: `∑_{k<n} Δt_k / t_{I,k}`. -/
noncomputable def fedSum (dt tI : ℕ → ℝ) (n : ℕ) : ℝ := ∑ k ∈ Finset.range n, dt k / tI k

/-- T1. With non-negative increments and positive tolerance times, the discrete FED is
monotone in the number of increments (exposure time). -/
theorem T1_fedSum_monotone (dt tI : ℕ → ℝ) (hdt : ∀ k, 0 ≤ dt k) (htI : ∀ k, 0 < tI k) :
    Monotone (fedSum dt tI) := by
  apply monotone_nat_of_le_succ
  intro n
  simp only [fedSum, Finset.sum_range_succ]
  have := div_nonneg (hdt n) (htI n).le
  linarith

/-- T1. The discrete FED is additive over consecutive intervals: the dose over `m + n`
increments is the dose over the first `m` plus the dose over the next `n`. -/
theorem T1_fedSum_additive (dt tI : ℕ → ℝ) (m n : ℕ) :
    fedSum dt tI (m + n) =
      fedSum dt tI m + ∑ k ∈ Finset.range n, dt (m + k) / tI (m + k) := by
  unfold fedSum
  exact Finset.sum_range_add _ m n

/-- T1. Monotone in the hazard history: if `t_I = g(h)` with `g` positive and antitone
(a worse hazard shortens the tolerance time), then a pointwise worse hazard history
`h ≤ h'` gives at least as much dose. -/
theorem T1_fedSum_mono_hazard (dt : ℕ → ℝ) (g : ℝ → ℝ) (h h' : ℕ → ℝ) (n : ℕ)
    (hdt : ∀ k, 0 ≤ dt k) (hg : ∀ x, 0 < g x) (hanti : Antitone g)
    (hh : ∀ k, h k ≤ h' k) :
    fedSum dt (fun k => g (h k)) n ≤ fedSum dt (fun k => g (h' k)) n := by
  unfold fedSum
  apply Finset.sum_le_sum
  intro k _
  exact div_le_div_of_nonneg_left (hdt k) (hg _) (hanti (hh k))

/-- T1, T12. ASET as a first-passage time. This file fixes the criterion as
`X_crit ≤ X(n)` ("reaches"). For a monotone dose that reaches `X_crit` at some step, the
least such step `Nat.find` is well defined and the dose is at or above `X_crit` exactly
from that step on. -/
theorem T12_first_passage (X : ℕ → ℝ) (hX : Monotone X) (Xc : ℝ) (hex : ∃ n, Xc ≤ X n)
    (n : ℕ) : Xc ≤ X n ↔ Nat.find hex ≤ n := by
  constructor
  · intro h; exact Nat.find_min' hex h
  · intro h; exact le_trans (Nat.find_spec hex) (hX h)

/-- T1, finding. For the discrete sum, "reaches" (`≥`) and ISO's "exceeds" (`>`) can give
first-passage steps one apart: with `Δt = 1` and `t_I = 2`, the dose is exactly 1 after
two steps (reached, not exceeded) and exceeds 1 only after three. -/
theorem T1_reaches_vs_exceeds :
    fedSum (fun _ => 1) (fun _ => 2) 1 < 1 ∧
    fedSum (fun _ => 1) (fun _ => 2) 2 = 1 ∧
    ¬ (1 < fedSum (fun _ => 1) (fun _ => 2) 2) ∧
    1 < fedSum (fun _ => 1) (fun _ => 2) 3 := by
  simp only [fedSum, Finset.sum_range_succ, Finset.sum_range_zero]
  norm_num

/-! ## T1. Fractional effective dose, continuous form -/

/-- T1. Continuous FED `X(t) = ∫₀ᵗ dτ / t_I(τ)`, with `t_I(τ) = t_I(c(τ))` the tolerance
time under the conditions at time `τ` (min). -/
noncomputable def fed (tI : ℝ → ℝ) (t : ℝ) : ℝ := ∫ τ in (0 : ℝ)..t, 1 / tI τ

/-- T1. Additivity over intervals: `X(t) = X(s) + ∫ₛᵗ dτ / t_I(τ)`, for any `s, t`, when
`1 / t_I` is integrable on bounded intervals (e.g. piecewise continuous). -/
theorem T1_fed_additive (tI : ℝ → ℝ)
    (hint : ∀ a b : ℝ, IntervalIntegrable (fun τ => 1 / tI τ) volume a b) (s t : ℝ) :
    fed tI t = fed tI s + ∫ τ in s..t, 1 / tI τ := by
  unfold fed
  exact (intervalIntegral.integral_add_adjacent_intervals (hint 0 s) (hint s t)).symm

/-- T1. With positive tolerance times, the continuous FED is monotone in the exposure
time. -/
theorem T1_fed_monotone (tI : ℝ → ℝ) (hpos : ∀ τ, 0 < tI τ)
    (hint : ∀ a b : ℝ, IntervalIntegrable (fun τ => 1 / tI τ) volume a b) :
    Monotone (fed tI) := by
  intro s t hst
  rw [T1_fed_additive tI hint s t]
  have : 0 ≤ ∫ τ in s..t, 1 / tI τ :=
    intervalIntegral.integral_nonneg hst (fun u _ => (one_div_pos.mpr (hpos u)).le)
  linarith

/-- T1. Under a constant tolerance time `t_I`, the dose is `X(t) = t / t_I`. -/
theorem T1_fed_const (tI t : ℝ) : fed (fun _ => tI) t = t / tI := by
  unfold fed
  rw [intervalIntegral.integral_const, smul_eq_mul]
  ring

/-- T1. For constant tolerance time `t_I > 0`, the dose reaches 1 exactly at `t = t_I`,
and (T12) reaches the threshold `X_crit` exactly from `t = X_crit t_I` on: under constant
conditions ASET is `X_crit t_I`. -/
theorem T1_fed_const_threshold (tI Xc : ℝ) (htI : 0 < tI) :
    fed (fun _ => tI) tI = 1 ∧ ∀ t, (Xc ≤ fed (fun _ => tI) t ↔ Xc * tI ≤ t) := by
  refine ⟨by rw [T1_fed_const]; exact div_self htI.ne', fun t => ?_⟩
  rw [T1_fed_const, le_div_iff₀ htI]

/-- T10. Edge dose `∫_a^{a+τ} r(t) dt` along a time-indexed route. If the dose rate `r` is
non-decreasing in time (a monotone fire) the edge dose is non-decreasing in the entry
time `a`, for a fixed traversal time `τ ≥ 0`. (Monotone rates are integrable, so no
further hypothesis is needed.) -/
theorem T10_edge_dose_mono_entry (r : ℝ → ℝ) (hr : Monotone r) (τ : ℝ) (hτ : 0 ≤ τ)
    {a a' : ℝ} (ha : a ≤ a') :
    ∫ t in a..a + τ, r t ≤ ∫ t in a'..a' + τ, r t := by
  have e1 : ∫ s in (0 : ℝ)..τ, r (a + s) = ∫ t in a..a + τ, r t := by
    rw [intervalIntegral.integral_comp_add_left, add_zero]
  have e2 : ∫ s in (0 : ℝ)..τ, r (a' + s) = ∫ t in a'..a' + τ, r t := by
    rw [intervalIntegral.integral_comp_add_left, add_zero]
  rw [← e1, ← e2]
  have hm1 : Monotone (fun s => r (a + s)) := fun x y hxy => hr (by linarith)
  have hm2 : Monotone (fun s => r (a' + s)) := fun x y hxy => hr (by linarith)
  exact intervalIntegral.integral_mono_on hτ hm1.intervalIntegrable hm2.intervalIntegrable
    (fun s _ => hr (by linarith))

/-! ## T3, T4. Convective heat -/

/-- T3. Time to incapacitation by convected heat, unclothed or lightly clothed,
`t = 5 × 10⁷ T^(-3.4)` (min, `T` in degC). -/
noncomputable def tConv (T : ℝ) : ℝ := 5e7 * T ^ (-(3.4 : ℝ))

/-- T4. Fully clothed, `t = 4.1 × 10⁸ T^(-3.61)` (min, `T` in degC). -/
noncomputable def tConvClothed (T : ℝ) : ℝ := 4.1e8 * T ^ (-(3.61 : ℝ))

lemma tConv_eq (T : ℝ) (hT : 0 ≤ T) : tConv T = 5e7 / T ^ ((17 : ℝ) / 5) := by
  unfold tConv
  rw [show (3.4 : ℝ) = 17 / 5 by norm_num, Real.rpow_neg hT]
  ring

lemma rpow175_pow5 {T : ℝ} (hT : 0 ≤ T) : (T ^ ((17 : ℝ) / 5)) ^ 5 = T ^ 17 := by
  rw [← Real.rpow_natCast, ← Real.rpow_mul hT,
    show (17 : ℝ) / 5 * ((5 : ℕ) : ℝ) = ((17 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]

/-- A power law `c T^(-p)` with `c, p > 0` is strictly decreasing for `T > 0`. -/
theorem powerLaw_strictAntiOn (c p : ℝ) (hc : 0 < c) (hp : 0 < p) :
    StrictAntiOn (fun T : ℝ => c * T ^ (-p)) (Set.Ioi 0) := by
  intro x hx y _ hxy
  simp only [Set.mem_Ioi] at hx
  have := Real.rpow_lt_rpow_of_neg hx hxy (by linarith : -p < 0)
  exact mul_lt_mul_of_pos_left this hc

/-- T3. The lightly clothed tolerance time is strictly decreasing in temperature (degC,
`T > 0`). -/
theorem T3_strictAntiOn : StrictAntiOn tConv (Set.Ioi 0) :=
  powerLaw_strictAntiOn 5e7 3.4 (by norm_num) (by norm_num)

/-- T4. The clothed tolerance time is strictly decreasing in temperature (degC, `T > 0`). -/
theorem T4_strictAntiOn : StrictAntiOn tConvClothed (Set.Ioi 0) :=
  powerLaw_strictAntiOn 4.1e8 3.61 (by norm_num) (by norm_num)

/-- T3, T10. The convective dose rate `1/t = T^3.4 / 5e7` (min⁻¹) composed with a
non-negative, non-decreasing temperature history is non-decreasing in time: the
monotone-fire hypothesis of `T10_edge_dose_mono_entry`. -/
theorem T3_rate_monotone (T : ℝ → ℝ) (hT : Monotone T) (h0 : ∀ t, 0 ≤ T t) :
    Monotone (fun t => 1 / tConv (T t)) ∧
      ∀ t, 0 < T t → 1 / tConv (T t) = T t ^ (3.4 : ℝ) / 5e7 := by
  have key : ∀ t, 1 / tConv (T t) = T t ^ (3.4 : ℝ) / 5e7 := by
    intro t
    unfold tConv
    rw [Real.rpow_neg (h0 t)]
    by_cases hz : T t ^ (3.4 : ℝ) = 0
    · rw [hz]; simp
    · field_simp
  refine ⟨fun s t hst => ?_, fun t _ => key t⟩
  show 1 / tConv (T s) ≤ 1 / tConv (T t)
  rw [key s, key t]
  exact div_le_div_of_nonneg_right (Real.rpow_le_rpow (h0 s) (hT hst) (by norm_num))
    (by norm_num)

lemma tConv_bounds (T lo hi : ℝ) (hT : 0 < T) (hlo : 0 < lo) (hhi : 0 < hi)
    (h1 : (5e7 / hi) ^ 5 < T ^ 17) (h2 : T ^ 17 < (5e7 / lo) ^ 5) :
    lo < tConv T ∧ tConv T < hi := by
  have hy : 0 < T ^ ((17 : ℝ) / 5) := Real.rpow_pos_of_pos hT _
  have hy5 := rpow175_pow5 hT.le
  have a : 5e7 / hi < T ^ ((17 : ℝ) / 5) :=
    lt_of_pow_lt_pow_left₀ 5 hy.le (by rw [hy5]; exact h1)
  have b : T ^ ((17 : ℝ) / 5) < 5e7 / lo :=
    lt_of_pow_lt_pow_left₀ 5 (by positivity) (by rw [hy5]; exact h2)
  rw [tConv_eq T hT.le]
  rw [div_lt_iff₀ hhi] at a
  rw [lt_div_iff₀ hlo] at b
  constructor
  · rw [lt_div_iff₀ hy]; linarith
  · rw [div_lt_iff₀ hy]; linarith

/-- T3, worked values. `45.0 min` at 60 degC (`44.9 < t < 45.1`), and at ambient 20 degC
`1880 < t < 1890` min: the law is not zero at ambient (`1/t ≈ 1/1886` min⁻¹), so a "no
dose at ambient" property needs an explicit cut-off. -/
theorem T3_values :
    44.9 < tConv 60 ∧ tConv 60 < 45.1 ∧ 1880 < tConv 20 ∧ tConv 20 < 1890 := by
  obtain ⟨a1, a2⟩ := tConv_bounds 60 44.9 45.1 (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
  obtain ⟨b1, b2⟩ := tConv_bounds 20 1880 1890 (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
  exact ⟨a1, a2, b1, b2⟩

/-- T3, T10, finding (the kelvin trap). Feeding the degC law a kelvin temperature: 20 degC
ambient as 293.15 gives `0.2 < t < 0.21` min instead of about 1886 min, so at
`X_crit = 0.3` a person standing in ambient air for 4 s (1/15 min) is already past the
threshold, whereas with the correct degC input it takes more than 564 min. -/
theorem T3_kelvin_trap :
    0.2 < tConv 293.15 ∧ tConv 293.15 < 0.21 ∧
    (0.3 : ℝ) ≤ fed (fun _ => tConv 293.15) (1 / 15) ∧
    ∀ t, (0.3 : ℝ) ≤ fed (fun _ => tConv 20) t → 564 < t := by
  obtain ⟨a1, a2⟩ := tConv_bounds 293.15 0.2 0.21 (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
  obtain ⟨b1, _⟩ := tConv_bounds 20 1880 1890 (by norm_num) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num)
  refine ⟨a1, a2, ?_, ?_⟩
  · rw [(T1_fed_const_threshold _ 0.3 (by linarith)).2]; linarith
  · intro t ht
    rw [(T1_fed_const_threshold _ 0.3 (by linarith)).2] at ht
    linarith

/-! ## T5, T6. Radiant heat -/

/-- T5, T6. ISO 13571 Eq (8) radiant dose rate for pain, `1/t = q^1.9 / 4.2` (min⁻¹), set
to zero below the incident-flux threshold `q < 2.5` kW m⁻² (so `q = 2.5` counts). -/
noncomputable def radRate (q : ℝ) : ℝ := if q < 2.5 then 0 else q ^ (1.9 : ℝ) / 4.2

/-- T5. The thresholded radiant dose rate is zero below 2.5 kW m⁻², positive at and above
it, and monotone in the flux. -/
theorem T5_radRate :
    (∀ q, q < 2.5 → radRate q = 0) ∧ (∀ q, 2.5 ≤ q → 0 < radRate q) ∧ Monotone radRate := by
  refine ⟨fun q hq => by simp [radRate, hq], fun q hq => ?_, ?_⟩
  · have : ¬ q < 2.5 := not_lt.mpr hq
    simp only [radRate, this, ↓reduceIte]
    exact div_pos (Real.rpow_pos_of_pos (by linarith) _) (by norm_num)
  · intro x y hxy
    unfold radRate
    split_ifs with hx hy hy
    · exact le_rfl
    · exact div_nonneg (Real.rpow_nonneg (by linarith) _) (by norm_num)
    · linarith
    · exact div_le_div_of_nonneg_right (Real.rpow_le_rpow (by linarith) hxy (by norm_num))
        (by norm_num)

/-- T5, finding (CFAST grouping). CFAST computes the rate as `(4.2 q)^1.9`; for every
`q ≥ 0` that is exactly `4.2^2.9` times ISO's `q^1.9 / 4.2`, and `64 < 4.2^2.9 < 64.5`. -/
theorem T5_cfast_grouping :
    (∀ q : ℝ, 0 ≤ q → (4.2 * q) ^ (1.9 : ℝ) = (4.2 : ℝ) ^ (2.9 : ℝ) * (q ^ (1.9 : ℝ) / 4.2)) ∧
    64 < (4.2 : ℝ) ^ (2.9 : ℝ) ∧ (4.2 : ℝ) ^ (2.9 : ℝ) < 64.5 := by
  have hsplit : (4.2 : ℝ) ^ (2.9 : ℝ) = (4.2 : ℝ) ^ (1.9 : ℝ) * 4.2 := by
    rw [show (2.9 : ℝ) = 1.9 + 1 by norm_num, Real.rpow_add (by norm_num), Real.rpow_one]
  have h29 : ((4.2 : ℝ) ^ (2.9 : ℝ)) ^ 10 = (4.2 : ℝ) ^ 29 := by
    rw [← Real.rpow_natCast, ← Real.rpow_mul (by norm_num),
      show (2.9 : ℝ) * ((10 : ℕ) : ℝ) = ((29 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
  have hpos : 0 < (4.2 : ℝ) ^ (2.9 : ℝ) := Real.rpow_pos_of_pos (by norm_num) _
  refine ⟨fun q hq => ?_, ?_, ?_⟩
  · rw [Real.mul_rpow (by norm_num) hq, hsplit]; field_simp
  · exact lt_of_pow_lt_pow_left₀ 10 hpos.le (by rw [h29]; norm_num)
  · exact lt_of_pow_lt_pow_left₀ 10 (by norm_num) (by rw [h29]; norm_num)

/-! ## T8, T9. CO₂ hyperventilation and the FDS cyanide term -/

/-- T8. ISO 13571 CO₂ hyperventilation factor `ν = exp(φ_CO2 / 5)` (φ in volume %). -/
noncomputable def nuCO2 (φ : ℝ) : ℝ := Real.exp (φ / 5)

/-- T8. `ν(0) = 1`, `ν(5 %) = e`, `ν ≥ 1` for `φ ≥ 0`, and `ν` is monotone. -/
theorem T8_nu : nuCO2 0 = 1 ∧ nuCO2 5 = Real.exp 1 ∧ (∀ φ, 0 ≤ φ → 1 ≤ nuCO2 φ) ∧
    Monotone nuCO2 := by
  refine ⟨by simp [nuCO2], by norm_num [nuCO2], fun φ hφ => ?_, fun x y h => ?_⟩
  · unfold nuCO2; exact Real.one_le_exp (by linarith)
  · unfold nuCO2; exact Real.exp_le_exp.mpr (by linarith)

/-- T9, finding. FDS's `HV_CO2 = exp(0.1903 C + 2.0004) / 7.1` is above 1.04 at `C = 0`, so
switching it on "as soon as any CO₂ is present" makes a jump from 1 to more than 1.04. -/
theorem T9_hv_jump : 1.04 < Real.exp (0.1903 * 0 + 2.0004) / 7.1 := by
  have he := Real.exp_one_gt_d9
  have h2 : Real.exp 2 ≤ Real.exp (0.1903 * 0 + 2.0004) := Real.exp_le_exp.mpr (by norm_num)
  have h3 : Real.exp 2 = Real.exp 1 * Real.exp 1 := by rw [← Real.exp_add]; norm_num
  rw [lt_div_iff₀ (by norm_num)]
  nlinarith

/-- T9. FDS User's Guide cyanide integrand `exp(C/43)/220 - 0.0045` (per min). -/
noncomputable def cnGuide (C : ℝ) : ℝ := Real.exp (C / 43) / 220 - 0.0045

/-- T9. FDS code cyanide integrand: `(exp(C/43) - 1)/220` for `C > 0`, else 0. -/
noncomputable def cnCode (C : ℝ) : ℝ := if 0 < C then (Real.exp (C / 43) - 1) / 220 else 0

/-- T9, finding. As printed in the User's Guide the integrand is positive in clean air
(`C = 0`) and negative at `C = -1` ppm (possible since `C_CN = C_HCN - C_NOx`), so the dose
could decrease. The code's clamped form is non-negative and monotone, which is the form a
monotone-dose argument needs. -/
theorem T9_cyanide :
    0 < cnGuide 0 ∧ cnGuide (-1) < 0 ∧ (∀ C, 0 ≤ cnCode C) ∧ Monotone cnCode := by
  refine ⟨by norm_num [cnGuide], ?_, fun C => ?_, fun x y hxy => ?_⟩
  · unfold cnGuide
    have h1 : 1 + 1 / 43 ≤ Real.exp (1 / 43) := by
      have := Real.add_one_le_exp (1 / 43 : ℝ); linarith
    have h2 : Real.exp (-1 / 43) * Real.exp (1 / 43) = 1 := by
      rw [← Real.exp_add]; norm_num
    have h3 : 0 < Real.exp (-1 / 43) := Real.exp_pos _
    nlinarith
  · unfold cnCode
    split_ifs with h
    · have : 1 ≤ Real.exp (C / 43) := Real.one_le_exp (by positivity)
      linarith
    · exact le_rfl
  · unfold cnCode
    split_ifs with hx hy hy
    · have := Real.exp_le_exp.mpr (show x / 43 ≤ y / 43 by linarith); linarith
    · linarith
    · have : 1 ≤ Real.exp (y / 43) := Real.one_le_exp (by positivity); linarith
    · exact le_rfl

/-! ## T13, T14, T16. RSET, margin, tenability window -/

/-- T13. `RSET = t_det + t_alarm + t_pre + t_trav` (s). -/
def rset (tdet talarm tpre ttrav : ℝ) : ℝ := tdet + talarm + tpre + ttrav

/-- T14. Safety margin `t_marg = ASET - RSET` (s). -/
def margin (aset rset : ℝ) : ℝ := aset - rset

/-- T14. `RSET ≤ ASET` iff the margin is non-negative, and `RSET < ASET` iff it is
positive. -/
theorem T14_margin (aset r : ℝ) :
    (r ≤ aset ↔ 0 ≤ margin aset r) ∧ (r < aset ↔ 0 < margin aset r) := by
  unfold margin
  exact ⟨sub_nonneg.symm, sub_pos.symm⟩

/-- T14. The Italian code's `t_marg ≥ 30 s` implies `RSET < ASET`. -/
theorem T14_italian (aset r : ℝ) (h : 30 ≤ margin aset r) : r < aset := by
  unfold margin at h; linarith

/-- T13. RSET is monotone in each of its components, and the PD 7974-6 two-case value
`max(t_det + t_a + t_pre,99 + t_walk, t_det + t_a + t_pre,1 + t_walk + t_flow)` is at least
each case. -/
theorem T13_rset (d a p w f d' a' p' w' : ℝ) (hd : d ≤ d') (ha : a ≤ a') (hp : p ≤ p')
    (hw : w ≤ w') (p99 p1 : ℝ) :
    rset d a p w ≤ rset d' a' p' w' ∧
    d + a + p99 + w ≤ max (d + a + p99 + w) (d + a + p1 + w + f) ∧
    d + a + p1 + w + f ≤ max (d + a + p99 + w) (d + a + p1 + w + f) := by
  refine ⟨by unfold rset; linarith, le_max_left _ _, le_max_right _ _⟩

/-- T16. Arrival function with a hard tenability window: entering at `t`, the edge is
completed at `t + τ` if `t + τ < b` and is infeasible (`⊤`) otherwise. -/
noncomputable def arrival (τ b t : ℝ) : WithTop ℝ :=
  if t + τ < b then ((t + τ : ℝ) : WithTop ℝ) else ⊤

/-- T16. The windowed arrival function is FIFO (non-decreasing in the entry time), for a
fixed traversal time `τ` and window end `b`. -/
theorem T16_fifo (τ b : ℝ) : Monotone (arrival τ b) := by
  intro s t hst
  unfold arrival
  split_ifs with h1 h2 h2
  · exact WithTop.coe_le_coe.mpr (by linarith)
  · exact le_top
  · exfalso; exact h1 (by linarith)
  · exact le_rfl

/-! ## E1-E5. Hydraulic speed and specific flow -/

/-- E1. Hydraulic speed `S = k (1 - a D)` (m s⁻¹). -/
def speed (k a D : ℝ) : ℝ := k * (1 - a * D)

/-- E4. Specific flow `F_s = S D = k D (1 - a D)` (persons s⁻¹ m⁻¹). -/
def specFlow (k a D : ℝ) : ℝ := speed k a D * D

/-- E1. For `k, a ≥ 0` the hydraulic speed is non-increasing in density. -/
theorem E1_speed_antitone (k a : ℝ) (hk : 0 ≤ k) (ha : 0 ≤ a) : Antitone (speed k a) := by
  intro x y hxy
  unfold speed
  exact mul_le_mul_of_nonneg_left (by nlinarith) hk

/-- E1. The speed is zero at the jam density `D = 1/a`. -/
theorem E1_jam (k a : ℝ) (ha : a ≠ 0) : speed k a (1 / a) = 0 := by
  unfold speed; field_simp; ring

/-- E1, finding. The stated upper limit 3.8 persons m⁻² lies beyond the jam density
`1/0.266 = 500/133 < 3.8`: there the level-route speed is negative, both in the form
`1.4 (1 - 0.266 D)` (`-0.01512`) and in Gwynne's rounded `1.4 - 0.37 D` (`-0.006`). -/
theorem E1_negative_at_3_8 :
    (1 : ℝ) / 0.266 = 500 / 133 ∧ (500 : ℝ) / 133 < 3.8 ∧
    speed 1.4 0.266 3.8 = -0.01512 ∧ (1.4 : ℝ) - 0.37 * 3.8 = -0.006 := by
  refine ⟨by norm_num, by norm_num, by norm_num [speed], by norm_num⟩

/-- E5. For `k ≥ 0`, `a > 0`, the specific flow is maximised at `D* = 1/(2a)` with value
`k/(4a)`; for `k > 0` this maximiser is unique. -/
theorem E5_max_flow (k a : ℝ) (hk : 0 ≤ k) (ha : 0 < a) :
    specFlow k a (1 / (2 * a)) = k / (4 * a) ∧
    (∀ D, specFlow k a D ≤ k / (4 * a)) ∧
    (0 < k → ∀ D, specFlow k a D = k / (4 * a) → D = 1 / (2 * a)) := by
  have gap : ∀ D, k / (4 * a) - specFlow k a D = k * (2 * a * D - 1) ^ 2 / (4 * a) := by
    intro D; unfold specFlow speed; field_simp; ring
  refine ⟨by unfold specFlow speed; field_simp; ring, fun D => ?_, fun hk' D hD => ?_⟩
  · have := gap D
    have : 0 ≤ k * (2 * a * D - 1) ^ 2 / (4 * a) := by positivity
    linarith
  · have h := gap D
    rw [hD, sub_self] at h
    have h1 : k * (2 * a * D - 1) ^ 2 = 0 := by
      have h4 : (4 * a) ≠ 0 := by positivity
      field_simp at h; linarith
    have h2 : (2 * a * D - 1) ^ 2 = 0 := by
      rcases mul_eq_zero.mp h1 with h | h
      · exact absurd h hk'.ne'
      · exact h
    have h3 : 2 * a * D - 1 = 0 := pow_eq_zero_iff (by norm_num) |>.mp h2
    field_simp; linarith

/-- E5. With `a = 0.266` m² person⁻¹: `D* = 250/133` persons m⁻² and `F_s,max = 125 k/133`;
for the level route `k = 1.4`, `F_s,max = 25/19 ≈ 1.316` persons s⁻¹ m⁻¹. -/
theorem E5_values (k : ℝ) :
    1 / (2 * (0.266 : ℝ)) = 250 / 133 ∧ k / (4 * 0.266) = 125 / 133 * k ∧
    (1.4 : ℝ) / (4 * 0.266) = 25 / 19 := by
  refine ⟨by norm_num, by ring, by norm_num⟩

/-! ## E2, E11. The low-density switch and the A* speed bound -/

/-- E1 + E2. SFPE speed with the low-density switch at `sw`: `0.85 k` for `D < sw`,
`k (1 - 0.266 D)` otherwise. SFPE uses `sw = 0.54`. -/
noncomputable def sfpeSpeed (sw k D : ℝ) : ℝ := if D < sw then 0.85 * k else speed k 0.266 D

/-- E2, finding. With the switch at 0.54 the speed jumps *up* at the switch
(`0.85 k < 0.85636 k`), so for `k > 0` the piecewise SFPE speed is **not** non-increasing
in density. -/
theorem E2_not_antitone (k : ℝ) (hk : 0 < k) : ¬ Antitone (sfpeSpeed 0.54 k) := by
  intro h
  have := h (show (0.5 : ℝ) ≤ 0.54 by norm_num)
  simp only [sfpeSpeed, speed] at this
  norm_num at this
  linarith

/-- E2. Moving the switch to the continuity point `75/133 ≈ 0.564` (where
`k (1 - 0.266 D) = 0.85 k`) makes the piecewise speed non-increasing in density. -/
theorem E2_continuity_switch (k : ℝ) (hk : 0 ≤ k) :
    speed k 0.266 (75 / 133) = 0.85 * k ∧ Antitone (sfpeSpeed (75 / 133) k) := by
  refine ⟨by unfold speed; ring, fun x y hxy => ?_⟩
  unfold sfpeSpeed speed
  split_ifs with hy hx hx
  · exact le_rfl
  · linarith
  · nlinarith
  · exact mul_le_mul_of_nonneg_left (by nlinarith) hk

/-- E11, finding. On a level route (`k = 1.4`) with the 0.54 switch, every density gives a
speed at most `149863/125000 ≈ 1.1989` m s⁻¹, this bound is attained at `D = 0.54`, and it
exceeds the single-evacuee 1.19 m s⁻¹. -/
theorem E11_sup_speed :
    (∀ D, 0 ≤ D → sfpeSpeed 0.54 1.4 D ≤ 149863 / 125000) ∧
    sfpeSpeed 0.54 1.4 0.54 = 149863 / 125000 ∧ (1.19 : ℝ) < 149863 / 125000 := by
  refine ⟨fun D _ => ?_, by norm_num [sfpeSpeed, speed], by norm_num⟩
  unfold sfpeSpeed speed
  split_ifs with h
  · norm_num
  · have h' : (0.54 : ℝ) ≤ D := not_lt.mp h
    nlinarith

/-- E11, finding. Hence `h = ‖x‖ / 1.19` is not a lower bound on traversal time: an edge of
any length `ℓ > 0` at density 0.54 is traversed in `ℓ / S < ℓ / 1.19`. With
`v_max = 149863/125000` the bound `ℓ / v_max ≤ ℓ / S` holds for every density at which the
speed is positive. -/
theorem E11_heuristic (ℓ : ℝ) :
    (0 < ℓ → ℓ / sfpeSpeed 0.54 1.4 0.54 < ℓ / 1.19) ∧
    (0 ≤ ℓ → ∀ D, 0 ≤ D → 0 < sfpeSpeed 0.54 1.4 D →
      ℓ / (149863 / 125000) ≤ ℓ / sfpeSpeed 0.54 1.4 D) := by
  obtain ⟨hsup, hat, hlt⟩ := E11_sup_speed
  refine ⟨fun hl => ?_, fun hl D hD hpos => ?_⟩
  · rw [hat]; exact div_lt_div_of_pos_left hl (by norm_num) hlt
  · exact div_le_div_of_nonneg_left hl hpos (hsup D hD)

/-! ## E8. Weidmann's law -/

/-- E8. Weidmann (Kladek form) `v(ρ) = v₀ (1 - exp(-γ (1/ρ - 1/ρ_max)))`. -/
noncomputable def weidmann (v0 γ ρmax ρ : ℝ) : ℝ :=
  v0 * (1 - Real.exp (-γ * (1 / ρ - 1 / ρmax)))

/-- E8. The speed is zero at the maximum density. -/
theorem E8_zero_at_max (v0 γ ρmax : ℝ) : weidmann v0 γ ρmax ρmax = 0 := by
  simp [weidmann]

/-- E8. For `v₀, γ > 0` the speed is strictly decreasing in density on `ρ > 0`. -/
theorem E8_strictAntiOn (v0 γ ρmax : ℝ) (hv : 0 < v0) (hγ : 0 < γ) :
    StrictAntiOn (weidmann v0 γ ρmax) (Set.Ioi 0) := by
  intro x hx y _ hxy
  simp only [Set.mem_Ioi] at hx
  have h1 : 1 / y < 1 / x := one_div_lt_one_div_of_lt hx hxy
  have h2 : -γ * (1 / x - 1 / ρmax) < -γ * (1 / y - 1 / ρmax) := by nlinarith
  have h3 := Real.exp_lt_exp.mpr h2
  unfold weidmann
  exact mul_lt_mul_of_pos_left (by linarith) hv

/-- E8. On `0 < ρ ≤ ρ_max` the speed lies in `[0, v₀)`: the free speed `v₀` is never
attained at positive density. -/
theorem E8_range (v0 γ ρmax ρ : ℝ) (hv : 0 < v0) (hγ : 0 < γ) (hρ : 0 < ρ)
    (hle : ρ ≤ ρmax) : 0 ≤ weidmann v0 γ ρmax ρ ∧ weidmann v0 γ ρmax ρ < v0 := by
  have h1 : 1 / ρmax ≤ 1 / ρ := one_div_le_one_div_of_le hρ hle
  have h2 : -γ * (1 / ρ - 1 / ρmax) ≤ 0 := by nlinarith
  have h3 : Real.exp (-γ * (1 / ρ - 1 / ρmax)) ≤ 1 := Real.exp_le_one_iff.mpr h2
  have h4 := Real.exp_pos (-γ * (1 / ρ - 1 / ρmax))
  unfold weidmann
  constructor
  · exact mul_nonneg hv.le (by linarith)
  · nlinarith

/-- E8. As `ρ → 0⁺` the speed tends to the free speed `v₀` (for `γ > 0`). Together with
`E8_zero_at_max` and `E8_strictAntiOn`: the speed decreases from `v₀` (as a limit) to 0 at
the maximum density. -/
theorem E8_tendsto_free (v0 γ ρmax : ℝ) (hγ : 0 < γ) :
    Tendsto (weidmann v0 γ ρmax) (𝓝[>] 0) (𝓝 v0) := by
  have h1 : Tendsto (fun ρ : ℝ => ρ⁻¹) (𝓝[>] 0) atTop := tendsto_inv_nhdsGT_zero
  have h2 : Tendsto (fun ρ : ℝ => -γ * ρ⁻¹ + γ / ρmax) (𝓝[>] 0) atBot :=
    tendsto_atBot_add_const_right _ _ (h1.const_mul_atTop_of_neg (by linarith))
  have h3 := Real.tendsto_exp_atBot.comp h2
  have h4 : Tendsto (fun ρ : ℝ => v0 * (1 - Real.exp (-γ * ρ⁻¹ + γ / ρmax))) (𝓝[>] 0)
      (𝓝 (v0 * (1 - 0))) :=
    (tendsto_const_nhds.sub h3).const_mul v0
  rw [sub_zero, mul_one] at h4
  refine h4.congr (fun ρ => ?_)
  simp only [weidmann]
  rw [show -γ * (1 / ρ - 1 / ρmax) = -γ * ρ⁻¹ + γ / ρmax by ring]

end FireEgress.Tenability

#print axioms FireEgress.Tenability.units_conc_dimensionless
#print axioms FireEgress.Tenability.units_T1_increment
#print axioms FireEgress.Tenability.units_T1_iso_eq1
#print axioms FireEgress.Tenability.units_T3_constant
#print axioms FireEgress.Tenability.units_T4_T5_untypeable
#print axioms FireEgress.Tenability.units_T5_radiant
#print axioms FireEgress.Tenability.units_T6_heat
#print axioms FireEgress.Tenability.units_T7_asphyxiant
#print axioms FireEgress.Tenability.units_T8_exponent
#print axioms FireEgress.Tenability.units_T10
#print axioms FireEgress.Tenability.units_E1
#print axioms FireEgress.Tenability.units_E4_E5
#print axioms FireEgress.Tenability.units_E6
#print axioms FireEgress.Tenability.units_E10
#print axioms FireEgress.Tenability.units_E8
#print axioms FireEgress.Tenability.units_E9
#print axioms FireEgress.Tenability.units_E11
#print axioms FireEgress.Tenability.T1_fedSum_monotone
#print axioms FireEgress.Tenability.T1_fedSum_additive
#print axioms FireEgress.Tenability.T1_fedSum_mono_hazard
#print axioms FireEgress.Tenability.T12_first_passage
#print axioms FireEgress.Tenability.T1_reaches_vs_exceeds
#print axioms FireEgress.Tenability.T1_fed_additive
#print axioms FireEgress.Tenability.T1_fed_monotone
#print axioms FireEgress.Tenability.T1_fed_const
#print axioms FireEgress.Tenability.T1_fed_const_threshold
#print axioms FireEgress.Tenability.T10_edge_dose_mono_entry
#print axioms FireEgress.Tenability.powerLaw_strictAntiOn
#print axioms FireEgress.Tenability.T3_strictAntiOn
#print axioms FireEgress.Tenability.T4_strictAntiOn
#print axioms FireEgress.Tenability.T3_rate_monotone
#print axioms FireEgress.Tenability.T3_values
#print axioms FireEgress.Tenability.T3_kelvin_trap
#print axioms FireEgress.Tenability.T5_radRate
#print axioms FireEgress.Tenability.T5_cfast_grouping
#print axioms FireEgress.Tenability.T8_nu
#print axioms FireEgress.Tenability.T9_hv_jump
#print axioms FireEgress.Tenability.T9_cyanide
#print axioms FireEgress.Tenability.T14_margin
#print axioms FireEgress.Tenability.T14_italian
#print axioms FireEgress.Tenability.T13_rset
#print axioms FireEgress.Tenability.T16_fifo
#print axioms FireEgress.Tenability.E1_speed_antitone
#print axioms FireEgress.Tenability.E1_jam
#print axioms FireEgress.Tenability.E1_negative_at_3_8
#print axioms FireEgress.Tenability.E5_max_flow
#print axioms FireEgress.Tenability.E5_values
#print axioms FireEgress.Tenability.E2_not_antitone
#print axioms FireEgress.Tenability.E2_continuity_switch
#print axioms FireEgress.Tenability.E11_sup_speed
#print axioms FireEgress.Tenability.E11_heuristic
#print axioms FireEgress.Tenability.E8_zero_at_max
#print axioms FireEgress.Tenability.E8_strictAntiOn
#print axioms FireEgress.Tenability.E8_range
#print axioms FireEgress.Tenability.E8_tendsto_free
