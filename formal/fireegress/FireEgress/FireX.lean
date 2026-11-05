import Mathlib
import FireEgress.Dim

/-!
# Fire-X (Wrede et al. 2025): every numbered equation, and the transcriber's notes

Wrede, Wagner, Mahfuz, Pałubicki, Michels and Pirk (2025), *Fire-X: Extinguishing Fire with
Stoichiometric Heat Release*, ACM Transactions on Graphics 44(6) 268,
https://doi.org/10.1145/3763338 (CC BY 4.0). The equations are those of the transcription
`wiki/sources/papers/firex-2025.md` and its catalogue `firex-2025-equations.json`: X1 to X32,
the boxes of Fig. 3 (F3a to F3e), the legend of Fig. 17 and the inline definitions I1 to I5.
Each equation is a Lean definition of its right-hand side at a point; derivatives and
Laplacians enter as given values. Units are checked with `Dim` extended by an exponent of
amount of substance (`DN`), because the paper mixes per-mole and per-kilogram quantities.
Studio does not run Fire-X itself; the Flamescapes port uses its gas chemistry
(`Flamescapes.lean`, `FlamescapesPort.lean`).

## Which theorem covers which equation

* X1, momentum (F3b is X1). `X1_units`; `X1_buoy_up` (with gravity pointing down, gas hotter
  than ambient rises), `X1_buoy_lt_g` (never faster than `|g|`), `X1_boussinesq` (under the
  ideal gas law the bracket is the density deficit `(ρ_amb - ρ)/ρ_amb`).
* X2, energy. `X2_units`; `X2_radiative_units` and `X2_radiative_fix` (note 1);
  `X2_radiative_sign` (the term cools gas hotter than ambient); `X2_fig3a` (F3a is X2 with X6).
* X3, species. `X3_units`; `X3_fig3c` (F3c is X3 with X11).
* X4, heat of combustion. `X4_o2_balance` (the bracket `c + 0.25h - 0.5o` is the O₂ demand),
  `X4_per_mol_O2` (417 kJ per mole of O₂ for every fuel), `X4_per_kg_O2` (13.03 MJ per kg of
  O₂, Huggett's oxygen-consumption principle, within 0.53% of his 13.1 MJ kg⁻¹),
  `X4_table1` (note 3), `X4_units`.
* X5, Arrhenius rate (F3e is X5). `X5_pos`, `X5_strictMonoOn`, `X5_exponent_units`,
  `X5_A_units` (note 18).
* X6, heat release. `X6_units`, `X6_pos`, `X6_heat_per_O2`, `X6_table1` (note 4).
* X7, species sources. `X7_units`, `X7_sum`, `X7_table1_balanced` and
  `X7_configurable_breaks` (note 19).
* X8, equation of state. `X8_units` (note 5), `X8_eos`, `X8_strictAntiOn`,
  `X8_printed_molar_mass`.
* X9, evaporation (F3d is X9). `X9_units` and `X9_below_boiling` (note 6), `X9_at_ambient`,
  `X9_eq_rho_e` (X9 is the per-step evaporated density of Sec. 5.4).
* X10, droplet conduction. `X10_units` and `X10_energy` (note 7), `X10_sign`.
* X11, evaporation source. `X11_units`, `X11_vs_X9` (note 8).
* X12, X13, X18, I3, I4, kernels. `kern_mem`, `kern_antitone`, `X12_X13_bounds`,
  `X13_negative_beyond` (note 10).
* X14 to X17, SPH forces (I5 is the force-acceleration link). `X14_units`, `X14_sign`;
  `X15_units` and `X15_direction` (note 11); `X16_symm`, `X17_antisymm`, `X17_units` (note 12).
* X19 to X25, particle-grid transfer. `X19_X21_units` (note 13), `X19_total`, `X21_uniform`,
  `X21_not_conservative`; `X22_X24_units`, `X23_between`, `X24_between` (via `wmean_mem`);
  `X25_between`.
* X26 to X28, I1, I2, droplets. `X26_X27_units`, `X28_volume` (note 9), `I2_evaporated_mass`,
  `I2_le_liquid`.
* X29 to X32, mixing. `X29_mem`, `X29_eq_one_iff` (note 14), `X30_enthalpy`, `X30_between`,
  `X31_species_mass`, `X31_sum` (note 15), `X31_fix`, `X32_momentum`, `X30_X32_identity`,
  `X29_X32_units`.
* Table 1, `note17_cp` (note 17). Fig. 17, `F17_mccaffrey` (its exponents are those of
  `FireDynamics.FD12_from_FD11`). Constant: `R_pos`.

## The transcriber's notes

Confirmed by a proof: notes 1, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 17 (given the ideal-gas
bound `C_p ≥ 5R/2` per mole), 18 (that the units of `A` vary) and 19. Notes 8 and 10 are
substantiated beyond what they claim: Eq. 11 driven by `ρ_l` creates vapour at `T_l = T_amb`,
where Eq. 9 removes no liquid; and the printed cubic kernel of Eq. 13 is negative at any
distance beyond `h_s`. Note 13 holds with a qualification: as printed, Eqs. 19 and 21 agree
with each other (mass in, mass out), and Eq. 21 returns a density only under App. A.1.2's unit
for `ρ_l`. Not mathematical claims, and so not decided here: notes 2 and 16 (notation), the
missing activation energies and the Westbrook-Dryer recollection of note 18, and notes 20 to 24.
No note is refuted.

Found beyond the notes: the round trip of Eqs. 19 to 21 keeps a uniform mass but does not
conserve mass in general, although Sec. 5.1 calls the transfer conservative
(`X21_not_conservative`: masses 1 and 0 return as 2/3 and 2/3); and the printed Eq. 8 makes a
heavier gas lighter (`X8_printed_molar_mass`).

## What is not covered

* The partial differential equations are stated by their right-hand sides at a point.
  Existence, uniqueness and the numerical schemes (semi-Lagrangian advection, the pressure
  projection, vorticity confinement, the SPH integrator) are not formalised.
* SPH vectors live in `Fin 3 → ℝ`; neighbour search is not modelled. Eq. 17 is stated with
  `Δu` in its linear term, the reading of note 12.
* Arithmetic is real; the GPU's single precision is not modelled.
-/

namespace FireEgress.FireX

open Real

/-! ## Dimensions with an amount of substance -/

/-- A dimension of `Dim` together with an exponent of amount of substance (mol), scaled by 60
like the others. Fire-X mixes per-mole and per-kilogram quantities, which `Dim` alone cannot
tell apart. -/
structure DN where
  d : Dim
  N : Int
  deriving DecidableEq

namespace DN

instance : Mul DN := ⟨fun a b => ⟨a.d * b.d, a.N + b.N⟩⟩
instance : Inv DN := ⟨fun a => ⟨a.d⁻¹, -a.N⟩⟩
instance : Div DN := ⟨fun a b => a * b⁻¹⟩

/-- `a ^ (n / d)`, exact when every scaled exponent times `n` is divisible by `d`. -/
def rpow (a : DN) (n d : Int) : DN := ⟨Dim.rpow a.d n d, a.N * n / d⟩

/-- Embed an SI dimension with no amount exponent. -/
def of (d : Dim) : DN := ⟨d, 0⟩
def one : DN := of Dim.one
def mol : DN := ⟨Dim.one, 60⟩
def mass : DN := of Dim.mass
def length : DN := of Dim.length
def time : DN := of Dim.time
def temp : DN := of Dim.temp
def area : DN := of Dim.area
def volume : DN := of Dim.volume
def velocity : DN := of Dim.velocity
def accel : DN := of Dim.accel
def energy : DN := of Dim.energy
def power : DN := of Dim.power
def pressure : DN := of Dim.pressure
def density : DN := of Dim.density
def specificHeat : DN := of Dim.specificHeat
def conductivity : DN := of Dim.conductivity
def htc : DN := of Dim.htc
/-- Diffusivity and kinematic viscosity, m² s⁻¹. -/
def diffusivity : DN := area / time
/-- Molar mass, kg mol⁻¹. -/
def molarMass : DN := mass / mol
/-- Molar concentration, mol m⁻³. -/
def conc : DN := mol / volume
/-- Molar energy, J mol⁻¹. -/
def molarEnergy : DN := energy / mol
/-- Specific energy, J kg⁻¹ (= m² s⁻²). -/
def specificEnergy : DN := energy / mass
/-- The gas constant, J mol⁻¹ K⁻¹. -/
def gasConst : DN := energy / (mol * temp)
/-- A volumetric reaction rate, mol m⁻³ s⁻¹. -/
def rate : DN := conc / time
/-- The Stefan-Boltzmann constant, W m⁻² K⁻⁴. -/
def stefan : DN := power / (area * temp * temp * temp * temp)
/-- Power per volume, W m⁻³. -/
def volPower : DN := power / volume
/-- Heat flux, W m⁻². -/
def flux : DN := power / area
/-- A force, kg m s⁻². -/
def force : DN := mass * accel

end DN

open DN

/-! ## Constants -/

/-- `R` (J mol⁻¹ K⁻¹), App. A.1.1. -/
noncomputable def R : ℝ := 8.314
/-- The heat released per mole of O₂ consumed in Eq. 4 (J mol⁻¹). -/
noncomputable def heatPerMolO2 : ℝ := 417e3
/-- The molar mass of O₂ (kg mol⁻¹). -/
noncomputable def M_O2 : ℝ := 0.032

theorem R_pos : 0 < R := by norm_num [R]

/-! ## X1. Momentum, and its buoyancy term -/

/-- Eq. 1's right-hand side, vertical component: `ν ∇²u - (1/ρ) ∂p/∂z - g_z (1 - T_amb/T)`,
with `g_z` the signed vertical component of gravity (`-9.81` when `z` points up). -/
noncomputable def X1_rhs (ν lapu dpdz ρ gz Tamb T : ℝ) : ℝ :=
  ν * lapu - (1 / ρ) * dpdz - gz * (1 - Tamb / T)

/-- Eq. 1's buoyancy acceleration, vertical component. -/
noncomputable def X1_buoy (gz Tamb T : ℝ) : ℝ := -gz * (1 - Tamb / T)

/-- Units of Eq. 1: every term is an acceleration. -/
theorem X1_units :
    velocity / time = accel ∧ diffusivity * (velocity / area) = accel ∧
      (one / density) * (pressure / length) = accel ∧ temp / temp = one ∧ accel * one = accel := by
  decide

/-- With gravity pointing down (`g_z < 0`), gas hotter than ambient accelerates upward. -/
theorem X1_buoy_up {gz Tamb T : ℝ} (hg : gz < 0) (ha : 0 < Tamb) (hT : Tamb < T) :
    0 < X1_buoy gz Tamb T := by
  unfold X1_buoy
  have hT0 : 0 < T := ha.trans hT
  have : Tamb / T < 1 := (div_lt_one hT0).2 hT
  nlinarith

/-- And never by more than `|g|`. -/
theorem X1_buoy_lt_g {gz Tamb T : ℝ} (hg : gz < 0) (ha : 0 < Tamb) (hT : Tamb < T) :
    X1_buoy gz Tamb T < -gz := by
  unfold X1_buoy
  have hT0 : 0 < T := ha.trans hT
  have : 0 < Tamb / T := div_pos ha hT0
  nlinarith

/-- Under the ideal gas law at fixed pressure and composition (`ρ T = ρ_amb T_amb`), Eq. 1's
bracket is the Boussinesq density deficit `(ρ_amb - ρ) / ρ_amb`. -/
theorem X1_boussinesq {ρ ρa T Ta : ℝ} (hρa : 0 < ρa) (hT : 0 < T) (hgas : ρ * T = ρa * Ta) :
    1 - Ta / T = (ρa - ρ) / ρa := by
  field_simp
  linarith

/-! ## X2. Energy -/

/-- Eq. 2's right-hand side. -/
noncomputable def X2_rhs (k ρ Cp lapT ε σ Tamb T Jc : ℝ) : ℝ :=
  k / (ρ * Cp) * lapT + ε * σ / Cp * (Tamb ^ 4 - T ^ 4) + Jc / (ρ * Cp)

/-- Units of Eq. 2's conduction and source terms: K s⁻¹, as `DT/Dt`. -/
theorem X2_units :
    temp / time = conductivity / (density * specificHeat) * (temp / area) ∧
      temp / time = volPower / (density * specificHeat) := by
  decide

/-- Transcriber's note 1, confirmed: the radiative term `ε σ T⁴ / C_p` is not a temperature
rate; it has units kg K m⁻² s⁻¹. -/
theorem X2_radiative_units :
    one * stefan * (temp * temp * temp * temp) / specificHeat = mass * temp / (area * time) ∧
      one * stefan * (temp * temp * temp * temp) / specificHeat ≠ temp / time := by
  decide

/-- Note 1's repair, confirmed: a factor of (area per volume) / ρ makes it a rate. -/
theorem X2_radiative_fix :
    one * stefan * (temp * temp * temp * temp) / specificHeat * (area / volume) / density =
      temp / time := by
  decide

/-- The radiative term cools gas hotter than ambient and warms gas cooler than it. -/
theorem X2_radiative_sign {ε σ Cp Tamb T : ℝ} (hc : 0 < ε * σ / Cp) (ha : 0 ≤ Tamb) (hT : 0 ≤ T) :
    ε * σ / Cp * (Tamb ^ 4 - T ^ 4) < 0 ↔ Tamb < T := by
  constructor
  · intro h
    by_contra h'
    have : T ^ 4 ≤ Tamb ^ 4 := pow_le_pow_left₀ hT (not_lt.1 h') 4
    nlinarith
  · intro h
    have : Tamb ^ 4 < T ^ 4 := pow_lt_pow_left₀ h ha (by norm_num)
    nlinarith

/-- Fig. 3's thermal box is Eq. 2 with Eq. 6 substituted for `J_c`. -/
theorem X2_fig3a (k ρ Cp lapT ε σ Tamb T φ v ΔH : ℝ) :
    X2_rhs k ρ Cp lapT ε σ Tamb T (-φ * v * ΔH) =
      k / (ρ * Cp) * lapT + ε * σ / Cp * (Tamb ^ 4 - T ^ 4) - ΔH / (ρ * Cp) * φ * v := by
  unfold X2_rhs; ring

/-! ## X3. Species transport -/

/-- Eq. 3's right-hand side. -/
noncomputable def X3_rhs (D lapY v q M ρ S : ℝ) : ℝ := D * lapY + v * q * M / ρ + S

/-- Units of Eq. 3: every term is s⁻¹. -/
theorem X3_units :
    one / time = diffusivity * (one / area) ∧ one / time = rate * one * molarMass / density ∧
      one / time = (volume / (mass * time)) * density := by
  decide

/-- Fig. 3's species box is Eq. 3 with Eq. 11's `S_i = Γ_i ρ_l`. -/
theorem X3_fig3c (D lapY v q M ρ Γ ρl : ℝ) :
    X3_rhs D lapY v q M ρ (Γ * ρl) = D * lapY + Γ * ρl + q * M / ρ * v := by
  unfold X3_rhs; ring

/-! ## X4. Heat of combustion: Huggett's oxygen-consumption principle -/

/-- Eq. 4: `Δ_c H° ≈ -417 (c + 0.25 h - 0.5 o)` (kJ mol⁻¹ of fuel). -/
noncomputable def X4 (c h o : ℝ) : ℝ := -417 * (c + 0.25 * h - 0.5 * o)

/-- The moles of O₂ that burn one mole of `C_c H_h O_o` to CO₂ and H₂O. -/
noncomputable def o2Demand (c h o : ℝ) : ℝ := c + h / 4 - o / 2

/-- The bracket of Eq. 4 is that O₂ demand: oxygen balances in
`C_c H_h O_o + x O₂ → c CO₂ + (h/2) H₂O` exactly when `x = c + h/4 - o/2`. -/
theorem X4_o2_balance (c h o x : ℝ) : o + 2 * x = 2 * c + h / 2 ↔ x = o2Demand c h o := by
  unfold o2Demand; constructor <;> intro hx <;> linarith

/-- So Eq. 4 releases 417 kJ per mole of O₂ consumed, whatever the fuel. -/
theorem X4_per_mol_O2 (c h o : ℝ) : X4 c h o = -417 * o2Demand c h o := by
  unfold X4 o2Demand; ring

/-- 417 kJ per mole of O₂ is 13.03 MJ per kilogram of O₂: Huggett's (1980) principle, whose
constant 13.1 MJ kg⁻¹ it matches within 0.53%. -/
theorem X4_per_kg_O2 :
    heatPerMolO2 / M_O2 = 13031250 ∧ |heatPerMolO2 / M_O2 - 13.1e6| ≤ 0.01 * 13.1e6 ∧
      |heatPerMolO2 / M_O2 - 13.1e6| / 13.1e6 < 0.0053 := by
  norm_num [heatPerMolO2, M_O2, abs_le]

/-- Transcriber's note 3, confirmed: all six Table 1 values (J mol⁻¹) are Eq. 4 per mole of
fuel, times 1000 (the fuels have no oxygen). -/
theorem X4_table1 :
    1000 * X4 2 2 0 = -1042500 ∧ 1000 * X4 4 10 0 = -2710500 ∧ 1000 * X4 3 6 0 = -1876500 ∧
      1000 * X4 3 8 0 = -2085000 ∧ 1000 * X4 1 4 0 = -834000 ∧ 1000 * X4 2 4 0 = -1251000 := by
  norm_num [X4]

/-- Units: Eq. 4's 417 is a molar energy and its bracket is a mole ratio. -/
theorem X4_units : molarEnergy * (mol / mol) = molarEnergy := by decide

/-! ## X5. Arrhenius rate -/

/-- Eq. 5: `v_c = A T^n exp(-E_a / (R T)) c_f^a c_o^b`. -/
noncomputable def X5 (A n Ea T cf co a b : ℝ) : ℝ :=
  A * T ^ n * Real.exp (-Ea / (R * T)) * cf ^ a * co ^ b

theorem X5_pos {A n Ea T cf co a b : ℝ} (hA : 0 < A) (hT : 0 < T) (hf : 0 < cf) (ho : 0 < co) :
    0 < X5 A n Ea T cf co a b := by
  unfold X5
  have := Real.rpow_pos_of_pos hT n
  have := Real.rpow_pos_of_pos hf a
  have := Real.rpow_pos_of_pos ho b
  have := Real.exp_pos (-Ea / (R * T))
  positivity

/-- With `E_a > 0` and `n ≥ 0` (Table 4: `n = 0`) the rate rises strictly with temperature. -/
theorem X5_strictMonoOn {A n Ea cf co a b : ℝ} (hA : 0 < A) (hn : 0 ≤ n) (hE : 0 < Ea)
    (hf : 0 < cf) (ho : 0 < co) :
    StrictMonoOn (fun T => X5 A n Ea T cf co a b) (Set.Ioi 0) := by
  intro T₁ h₁ T₂ h₂ h12
  simp only [Set.mem_Ioi] at h₁ h₂
  unfold X5
  have hp1 : T₁ ^ n ≤ T₂ ^ n := Real.rpow_le_rpow h₁.le h12.le hn
  have hp0 : 0 < T₁ ^ n := Real.rpow_pos_of_pos h₁ n
  have he : Real.exp (-Ea / (R * T₁)) < Real.exp (-Ea / (R * T₂)) := by
    rw [Real.exp_lt_exp, neg_div, neg_div, neg_lt_neg_iff]
    exact div_lt_div_of_pos_left hE (mul_pos R_pos h₁) (mul_lt_mul_of_pos_left h12 R_pos)
  have hc : 0 < cf ^ a * co ^ b := mul_pos (Real.rpow_pos_of_pos hf a) (Real.rpow_pos_of_pos ho b)
  have hAe : A * T₁ ^ n * Real.exp (-Ea / (R * T₁)) < A * T₂ ^ n * Real.exp (-Ea / (R * T₂)) := by
    have h1 : A * T₁ ^ n * Real.exp (-Ea / (R * T₁)) < A * T₁ ^ n * Real.exp (-Ea / (R * T₂)) :=
      mul_lt_mul_of_pos_left he (mul_pos hA hp0)
    have h2 : A * T₁ ^ n * Real.exp (-Ea / (R * T₂)) ≤ A * T₂ ^ n * Real.exp (-Ea / (R * T₂)) :=
      mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hp1 hA.le) (Real.exp_pos _).le
    exact h1.trans_le h2
  have := mul_lt_mul_of_pos_right hAe hc
  calc A * T₁ ^ n * Real.exp (-Ea / (R * T₁)) * cf ^ a * co ^ b
      = A * T₁ ^ n * Real.exp (-Ea / (R * T₁)) * (cf ^ a * co ^ b) := by ring
    _ < A * T₂ ^ n * Real.exp (-Ea / (R * T₂)) * (cf ^ a * co ^ b) := this
    _ = A * T₂ ^ n * Real.exp (-Ea / (R * T₂)) * cf ^ a * co ^ b := by ring

/-- The exponent `E_a / (R T)` is dimensionless. -/
theorem X5_exponent_units : molarEnergy / (gasConst * temp) = one := by decide

/-- Transcriber's note 18, confirmed: the units of `A` vary with `a + b`. For methane
(`a + b = 1`) `A` is in s⁻¹; for acetylene (`a + b = 7/4`) in mol^(-3/4) m^(9/4) s⁻¹. -/
theorem X5_A_units :
    rate / DN.rpow conc 1 1 = one / time ∧
      rate / DN.rpow conc 7 4 ≠ one / time ∧
      rate / DN.rpow conc 7 4 = ⟨Dim.rpow Dim.length 9 4 / Dim.time, -45⟩ := by
  decide

/-! ## X6. Heat release -/

/-- Eq. 6: `J_c = -φ v_c Δ_c H⁰`. -/
noncomputable def X6 (φ v ΔH : ℝ) : ℝ := -φ * v * ΔH

/-- Units: with `Δ_c H` per mole (Eq. 4, Table 1) `J_c` is W m⁻³; read per kilogram, as App.
A.1.1 also allows, it is not. -/
theorem X6_units :
    one * rate * molarEnergy = volPower ∧ one * rate * specificEnergy ≠ volPower := by
  decide

theorem X6_pos {φ v ΔH : ℝ} (hφ : 0 < φ) (hv : 0 < v) (hH : ΔH < 0) : 0 < X6 φ v ΔH := by
  unfold X6; nlinarith [mul_pos hφ hv]

/-- With stoichiometric coefficients (`q_o = q_f D`, `D` the O₂ demand per mole of fuel),
the heat released per mole of O₂ consumed is `φ · 417 kJ / |q_f|`. -/
theorem X6_heat_per_O2 {φ v qf c h o : ℝ} (hv : v ≠ 0) (hqf : qf < 0) (hD : o2Demand c h o ≠ 0) :
    1000 * X6 φ v (X4 c h o) / (-(qf * o2Demand c h o) * v) = φ * heatPerMolO2 / (-qf) := by
  unfold X6 heatPerMolO2
  rw [X4_per_mol_O2]
  have : qf ≠ 0 := hqf.ne
  field_simp
  ring

/-- Transcriber's note 4, confirmed: for the Table 1 fuels with `q_f = -2` (acetylene, butane,
cyclopropane), Eq. 6 releases 208.5 kJ per mole of O₂ that Eq. 7 removes, half of Eq. 4's
417; for those with `q_f = -1` (propane, methane, ethylene), the full 417. -/
theorem X6_table1 :
    1000 * X6 1 1 (X4 2 2 0) / 5 = 208500 ∧ 1000 * X6 1 1 (X4 4 10 0) / 13 = 208500 ∧
      1000 * X6 1 1 (X4 3 6 0) / 9 = 208500 ∧ 1000 * X6 1 1 (X4 3 8 0) / 5 = 417000 ∧
      1000 * X6 1 1 (X4 1 4 0) / 2 = 417000 ∧ 1000 * X6 1 1 (X4 2 4 0) / 3 = 417000 := by
  norm_num [X6, X4]

/-! ## X7. Species sources of the reaction -/

/-- Eq. 7: `∂Y_i/∂t = v_c q_i M_i / ρ`. -/
noncomputable def X7 (v q M ρ : ℝ) : ℝ := v * q * M / ρ

theorem X7_units : one / time = rate * one * molarMass / density := by decide

/-- The reaction keeps `Σ Y_i` exactly when `Σ q_i M_i = 0`. -/
theorem X7_sum {ι : Type*} (s : Finset ι) (v ρ : ℝ) (q M : ι → ℝ) (hv : v ≠ 0) (hρ : ρ ≠ 0) :
    ∑ i ∈ s, X7 v (q i) (M i) ρ = 0 ↔ ∑ i ∈ s, q i * M i = 0 := by
  unfold X7
  have : ∑ i ∈ s, v * q i * M i / ρ = v / ρ * ∑ i ∈ s, q i * M i := by
    rw [Finset.mul_sum]; congr 1; ext i; ring
  rw [this]
  constructor
  · intro h
    rcases mul_eq_zero.1 h with h | h
    · exact absurd h (div_ne_zero hv hρ)
    · exact h
  · intro h; rw [h, mul_zero]

/-- Molar mass of `C_c H_h O_o` from atomic masses. -/
def mm (mC mH mO : ℝ) (c h o : ℝ) : ℝ := c * mC + h * mH + o * mO

/-- Transcriber's note 19, first half, confirmed: the six Table 1 reactions balance every
element, so `Σ q_i M_i = 0` for any atomic masses. -/
theorem X7_table1_balanced (mC mH mO : ℝ) :
    let M := mm mC mH mO
    (-2 * M 2 2 0 - 5 * M 0 0 2 + 4 * M 1 0 2 + 2 * M 0 2 1 = 0) ∧
      (-2 * M 4 10 0 - 13 * M 0 0 2 + 8 * M 1 0 2 + 10 * M 0 2 1 = 0) ∧
      (-2 * M 3 6 0 - 9 * M 0 0 2 + 6 * M 1 0 2 + 6 * M 0 2 1 = 0) ∧
      (-1 * M 3 8 0 - 5 * M 0 0 2 + 3 * M 1 0 2 + 4 * M 0 2 1 = 0) ∧
      (-1 * M 1 4 0 - 2 * M 0 0 2 + 1 * M 1 0 2 + 2 * M 0 2 1 = 0) ∧
      (-1 * M 2 4 0 - 3 * M 0 0 2 + 2 * M 1 0 2 + 2 * M 0 2 1 = 0) := by
  simp only [mm]
  refine ⟨by ring, by ring, by ring, by ring, by ring, by ring⟩

/-- Note 19, second half, confirmed: a configurable product coefficient breaks the balance;
methane with `q_CO₂ = 2` (inside Table 3's 0 to 10) leaves `Σ q_i M_i = M_CO₂ > 0`, so by
`X7_sum` the reaction creates mass. -/
theorem X7_configurable_breaks (mC mH mO : ℝ) (hC : 0 < mC) (hO : 0 < mO) :
    let M := mm mC mH mO
    (-1 * M 1 4 0 - 2 * M 0 0 2 + 2 * M 1 0 2 + 2 * M 0 2 1 = M 1 0 2) ∧ 0 < M 1 0 2 := by
  simp only [mm]
  constructor
  · ring
  · linarith

/-! ## X8. Equation of state -/

/-- Eq. 8 as printed, `ρ = p_amb (T R Σ Y_i M_i)⁻¹` (the comma read as nothing). -/
noncomputable def X8_printed {ι : Type*} [Fintype ι] (p T : ℝ) (Y M : ι → ℝ) : ℝ :=
  p / (T * R * ∑ i, Y i * M i)

/-- Eq. 8 with `Y_i / M_i`, the mixture ideal-gas law. -/
noncomputable def X8 {ι : Type*} [Fintype ι] (p T : ℝ) (Y M : ι → ℝ) : ℝ :=
  p / (R * T * ∑ i, Y i / M i)

/-- Transcriber's note 5, confirmed: as printed the right-hand side is not a density; with
`Y_i / M_i` it is. -/
theorem X8_units :
    pressure / (temp * gasConst * (one * molarMass)) ≠ density ∧
      pressure / (gasConst * temp * (one / molarMass)) = density := by
  decide

/-- The corrected Eq. 8 is the ideal gas law `p = ρ R T Σ Y_i / M_i`. -/
theorem X8_eos {ι : Type*} [Fintype ι] {p T : ℝ} {Y M : ι → ℝ} (hT : 0 < T)
    (hW : 0 < ∑ i, Y i / M i) : p = X8 p T Y M * R * T * ∑ i, Y i / M i := by
  unfold X8
  have := R_pos
  field_simp

/-- Hot gas is lighter. -/
theorem X8_strictAntiOn {ι : Type*} [Fintype ι] {p : ℝ} {Y M : ι → ℝ} (hp : 0 < p)
    (hW : 0 < ∑ i, Y i / M i) : StrictAntiOn (fun T => X8 p T Y M) (Set.Ioi 0) := by
  intro T₁ h₁ T₂ h₂ h12
  simp only [Set.mem_Ioi] at h₁ h₂
  unfold X8
  have := R_pos
  apply div_lt_div_of_pos_left hp (by positivity)
  exact mul_lt_mul_of_pos_right (mul_lt_mul_of_pos_left h12 R_pos) hW

/-- Note 5 beyond units: for a single species the printed form makes a heavier gas lighter,
`ρ ∝ 1/M`, where the ideal gas law gives `ρ ∝ M`. -/
theorem X8_printed_molar_mass {p T M₁ M₂ : ℝ} (hp : 0 < p) (hT : 0 < T) (h₁ : 0 < M₁)
    (h12 : M₁ < M₂) :
    X8_printed p T (fun _ : Unit => (1 : ℝ)) (fun _ => M₂) <
        X8_printed p T (fun _ : Unit => (1 : ℝ)) (fun _ => M₁) ∧
      X8 p T (fun _ : Unit => (1 : ℝ)) (fun _ => M₁) < X8 p T (fun _ : Unit => (1 : ℝ)) (fun _ => M₂) := by
  unfold X8_printed X8
  simp only [Finset.univ_unique, Finset.sum_singleton, one_mul]
  have := R_pos
  have h₂ : 0 < M₂ := h₁.trans h12
  constructor
  · exact div_lt_div_of_pos_left hp (by positivity) (mul_lt_mul_of_pos_left h12 (by positivity))
  · rw [div_lt_div_iff₀ (by positivity) (by positivity)]
    have : R * T * (1 / M₂) < R * T * (1 / M₁) :=
      mul_lt_mul_of_pos_left (one_div_lt_one_div_of_lt h₁ h12) (by positivity)
    nlinarith

/-! ## X9. Evaporation -/

/-- Eq. 9's right-hand side. -/
noncomputable def X9 (ρl Cp Tl Tamb TB ΔvH : ℝ) : ℝ :=
  -(ρl * Cp * (Tl - Tamb)) / (Cp * (TB - Tamb) + ΔvH)

/-- Transcriber's note 6, confirmed: the right-hand side is a density, not a density per
time. -/
theorem X9_units :
    specificHeat * temp = specificEnergy ∧ density * specificHeat * temp / specificEnergy = density ∧
      density ≠ density / time := by
  decide

/-- Note 6's second point, confirmed: any liquid warmer than ambient evaporates, below its
boiling point too. -/
theorem X9_below_boiling {ρl Cp Tl Tamb TB ΔvH : ℝ} (hρ : 0 < ρl) (hC : 0 < Cp) (hH : 0 < ΔvH)
    (ha : Tamb < Tl) (hb : Tl < TB) : X9 ρl Cp Tl Tamb TB ΔvH < 0 := by
  unfold X9
  have hd : 0 < Cp * (TB - Tamb) + ΔvH := by nlinarith
  apply div_neg_of_neg_of_pos _ hd
  have : 0 < ρl * Cp * (Tl - Tamb) := by have := sub_pos.2 ha; positivity
  linarith

theorem X9_at_ambient (ρl Cp Tamb TB ΔvH : ℝ) : X9 ρl Cp Tamb Tamb TB ΔvH = 0 := by
  simp [X9]

/-! ## X10. Droplet conduction -/

/-- Eq. 10's intermediate `T_dc = (ρ_l / ρ_d)(A_d / d_d)(T_l - T)`. -/
noncomputable def X10_Tdc (ρl ρd Ad dd Tl T : ℝ) : ℝ := ρl / ρd * (Ad / dd) * (Tl - T)

/-- Eq. 10's gas rate `k T_dc / (ρ C_p)`; the liquid's is its negative. -/
noncomputable def X10_rate (k Tdc ρ Cp : ℝ) : ℝ := k * Tdc / (ρ * Cp)

/-- Transcriber's note 7, confirmed: `T_dc` is in K m and the rate in K m³ s⁻¹, not K s⁻¹. -/
theorem X10_units :
    density / density * (area / length) * temp = temp * length ∧
      conductivity * (temp * length) / (density * specificHeat) = temp * volume / time ∧
      temp * volume / time ≠ temp / time := by
  decide

/-- Note 7's second point, confirmed: equal and opposite temperature rates conserve energy only
when gas and liquid have equal heat capacities per volume. -/
theorem X10_energy (ρ Cp ρl Cl r : ℝ) (hr : r ≠ 0) :
    ρ * Cp * r + ρl * Cl * (-r) = 0 ↔ ρ * Cp = ρl * Cl := by
  constructor
  · intro h
    have : (ρ * Cp - ρl * Cl) * r = 0 := by linarith
    rcases mul_eq_zero.1 this with h | h
    · linarith
    · exact absurd h hr
  · intro h; rw [h]; ring

/-- The gas warms when the liquid is hotter. -/
theorem X10_sign {k ρl ρd Ad dd Tl T ρ Cp : ℝ} (hk : 0 < k) (hl : 0 < ρl) (hd : 0 < ρd)
    (hA : 0 < Ad) (hdd : 0 < dd) (hρ : 0 < ρ) (hC : 0 < Cp) (hT : T < Tl) :
    0 < X10_rate k (X10_Tdc ρl ρd Ad dd Tl T) ρ Cp := by
  unfold X10_rate X10_Tdc
  have := sub_pos.2 hT
  positivity

/-! ## X11. Species source from evaporation -/

/-- Eq. 11: `∂Y_i/∂t = Γ_i ρ_l`. -/
noncomputable def X11 (Γ ρl : ℝ) : ℝ := Γ * ρl

theorem X11_units : one / time = (volume / (mass * time)) * density := by decide

/-- Transcriber's note 8, substantiated: at `T_l = T_amb` Eq. 9 removes no liquid while
Eq. 11, driven by `ρ_l`, still adds vapour, so the pair creates mass. The evaporated density
`ρ_e` of p. 7 vanishes there (`I2_evaporated_mass`, with `E_l = 0`). -/
theorem X11_vs_X9 {Γ ρl : ℝ} (hΓ : 0 < Γ) (hρ : 0 < ρl) (Cp Tamb TB ΔvH : ℝ) :
    X9 ρl Cp Tamb Tamb TB ΔvH = 0 ∧ 0 < X11 Γ ρl :=
  ⟨X9_at_ambient _ _ _ _ _, mul_pos hΓ hρ⟩

/-! ## X12 to X17. SPH -/

/-- The kernels of Eqs. 12, 13: `(1 - r/h)²` and `(1 - r/h)³`; Eq. 18's weight is `1 - r/h`. -/
noncomputable def kern (r h : ℝ) : ℝ := 1 - r / h

theorem kern_mem {r h : ℝ} (hh : 0 < h) (hr0 : 0 ≤ r) (hrh : r ≤ h) :
    0 ≤ kern r h ∧ kern r h ≤ 1 := by
  unfold kern
  have : r / h ≤ 1 := (div_le_one hh).2 hrh
  have : 0 ≤ r / h := div_nonneg hr0 hh.le
  constructor <;> linarith

/-- The kernels fall with distance. -/
theorem kern_antitone {h : ℝ} (hh : 0 < h) (n : ℕ) :
    AntitoneOn (fun r => kern r h ^ n) (Set.Icc 0 h) := by
  intro r₁ h₁ r₂ h₂ h12
  simp only
  apply pow_le_pow_left₀ (kern_mem hh h₂.1 h₂.2).1
  unfold kern
  have : r₁ / h ≤ r₂ / h := div_le_div_of_nonneg_right h12 hh.le
  linarith

/-- Eq. 12: the pseudo-density `Σ_j (1 - r_ij/h_s)²`. -/
noncomputable def X12 {ι : Type*} (s : Finset ι) (r : ι → ℝ) (h : ℝ) : ℝ :=
  ∑ j ∈ s, kern (r j) h ^ 2

/-- Eq. 13: the near-density `Σ_j (1 - r_ij/h_s)³`. -/
noncomputable def X13 {ι : Type*} (s : Finset ι) (r : ι → ℝ) (h : ℝ) : ℝ :=
  ∑ j ∈ s, kern (r j) h ^ 3

/-- Over the neighbourhood (`r ≤ h_s`, I3) both are non-negative and the near-density is at most
the density. -/
theorem X12_X13_bounds {ι : Type*} (s : Finset ι) {r : ι → ℝ} {h : ℝ} (hh : 0 < h)
    (hr : ∀ j ∈ s, 0 ≤ r j ∧ r j ≤ h) :
    0 ≤ X13 s r h ∧ X13 s r h ≤ X12 s r h := by
  unfold X12 X13
  constructor
  · exact Finset.sum_nonneg (fun j hj => pow_nonneg (kern_mem hh (hr j hj).1 (hr j hj).2).1 3)
  · apply Finset.sum_le_sum
    intro j hj
    obtain ⟨h0, h1⟩ := kern_mem hh (hr j hj).1 (hr j hj).2
    have : kern (r j) h ^ 3 = kern (r j) h ^ 2 * kern (r j) h := by ring
    rw [this]
    calc kern (r j) h ^ 2 * kern (r j) h ≤ kern (r j) h ^ 2 * 1 :=
          mul_le_mul_of_nonneg_left h1 (sq_nonneg _)
      _ = kern (r j) h ^ 2 := mul_one _

/-- Transcriber's note 10, substantiated: Eq. 13 needs `r ≤ h_s`, which `N_s(i)` guarantees for
`r_s_ij` only; at any distance beyond `h_s` the printed cubic kernel is negative. -/
theorem X13_negative_beyond {r h : ℝ} (hh : 0 < h) (hr : h < r) : kern r h ^ 3 < 0 := by
  unfold kern
  have : 1 < r / h := (one_lt_div hh).2 hr
  have : 1 - r / h < 0 := by linarith
  exact Odd.pow_neg (by decide) this

/-- Eq. 14: `p = k_s (ρ_s - ρ_s0)`, `p_near = k_s^near ρ_s^near`. -/
noncomputable def X14 (ks ρ ρ0 : ℝ) : ℝ := ks * (ρ - ρ0)

theorem X14_units :
    specificEnergy * one = specificEnergy ∧ specificEnergy = of (Dim.area / (Dim.time * Dim.time)) := by
  decide

/-- Below the rest density the pseudo-pressure is negative (attraction: Clavet's surface
tension). -/
theorem X14_sign {ks ρ ρ0 : ℝ} (hk : 0 < ks) : X14 ks ρ ρ0 < 0 ↔ ρ < ρ0 := by
  unfold X14
  constructor
  · intro h; by_contra h'; nlinarith [not_lt.1 h']
  · intro h; nlinarith

/-- Eq. 15 as printed for one neighbour: `-(p_j w + p_j^near w²)`. -/
noncomputable def X15_term (p pn w : ℝ) : ℝ := -(p * w + pn * w ^ 2)

/-- Transcriber's note 11, confirmed: the printed sum has units of a specific energy, neither a
force nor an acceleration. -/
theorem X15_units :
    specificEnergy * one ≠ force ∧ specificEnergy * one ≠ accel := by
  decide

/-- Note 11's missing direction, substantiated: as a scalar, two particles at the same pressure
push each other with equal, not opposite, terms, so momentum is not conserved; with the unit
vector `r̂_ij` (and `r̂_ji = -r̂_ij`) the pair cancels. -/
theorem X15_direction (p pn w : ℝ) (hp : 0 < p) (hpn : 0 ≤ pn) (hw : 0 < w) (rhat : ℝ) :
    X15_term p pn w + X15_term p pn w ≠ 0 ∧
      X15_term p pn w * rhat + X15_term p pn w * (-rhat) = 0 := by
  unfold X15_term
  constructor
  · have : 0 < p * w + pn * w ^ 2 := by positivity
    linarith
  · ring

/-- The dot product on `ℝ³`. -/
def dot (a b : Fin 3 → ℝ) : ℝ := ∑ k, a k * b k

/-- Eq. 16: `Δu = (u_i - u_j) · r̂_ij`. -/
def X16 (ui uj rhat : Fin 3 → ℝ) : ℝ := dot (ui - uj) rhat

/-- The radial velocity difference is the same seen from either particle. -/
theorem X16_symm (ui uj rhat : Fin 3 → ℝ) : X16 uj ui (-rhat) = X16 ui uj rhat := by
  unfold X16 dot
  congr 1; ext k
  simp only [Pi.sub_apply, Pi.neg_apply]
  ring

/-- Eq. 17 with `Δu` in the linear term: `(1 - r/h)(α Δu + β Δu²) r̂`. -/
noncomputable def X17 (r h α β Δu : ℝ) (rhat : Fin 3 → ℝ) : Fin 3 → ℝ :=
  fun k => kern r h * (α * Δu + β * Δu ^ 2) * rhat k

/-- Each pair's viscous terms are equal and opposite, so momentum is conserved. -/
theorem X17_antisymm (r h α β : ℝ) (ui uj rhat : Fin 3 → ℝ) :
    X17 r h α β (X16 uj ui (-rhat)) (-rhat) = -X17 r h α β (X16 ui uj rhat) rhat := by
  rw [X16_symm]
  unfold X17
  ext k
  simp only [Pi.neg_apply]
  ring

/-- Transcriber's note 12, confirmed: with App. A.1.2's units (`α` in s⁻¹, `β` in m⁻¹) both
`α Δu` and `β Δu²` are accelerations. -/
theorem X17_units :
    (one / time) * velocity = accel ∧ (one / length) * (velocity * velocity) = accel := by
  decide

/-! ## X18 to X25. Particle-grid transfer -/

section Transfer

variable {ι κ : Type*} [Fintype ι] [Fintype κ]

/-- Eq. 19: `ρ_lj = Σ_i m_i w_ij`. -/
noncomputable def X19 (m : ι → ℝ) (w : ι → κ → ℝ) (j : κ) : ℝ := ∑ i, m i * w i j

/-- Eq. 20: `ω_j = Σ_i w_ij`. -/
noncomputable def X20 (w : ι → κ → ℝ) (j : κ) : ℝ := ∑ i, w i j

/-- Eq. 21: `m_i = (Σ_j w_ij)⁻¹ Σ_j ρ_lj w_ij / ω_j`. -/
noncomputable def X21 (ρl : κ → ℝ) (w : ι → κ → ℝ) (i : ι) : ℝ :=
  (∑ j, w i j)⁻¹ * ∑ j, ρl j * w i j / X20 w j

/-- Transcriber's note 13, confirmed: Eq. 19 gives a mass (kg), where App. A.1.2 says kg m⁻³;
as printed Eqs. 19 and 21 agree (mass back to mass); with the appendix's unit Eq. 21 returns a
density. -/
theorem X19_X21_units :
    mass * one ≠ density ∧ mass * one = mass ∧ (one / one) * (mass * one / one) = mass ∧
      (one / one) * (density * one / one) ≠ mass := by
  decide

/-- The grid total is `Σ_i m_i W_i`, with `W_i = Σ_j w_ij`: particle-to-grid keeps the mass only
when every particle's weights sum to one. -/
theorem X19_total (m : ι → ℝ) (w : ι → κ → ℝ) :
    ∑ j, X19 m w j = ∑ i, m i * ∑ j, w i j := by
  unfold X19
  rw [Finset.sum_comm]
  congr 1; ext i
  rw [Finset.mul_sum]

/-- The round trip keeps a uniform mass: if every particle has mass `μ`, Eq. 21 returns `μ`. -/
theorem X21_uniform (μ : ℝ) (w : ι → κ → ℝ) (hω : ∀ j, X20 w j ≠ 0) (i : ι)
    (hW : ∑ j, w i j ≠ 0) : X21 (X19 (fun _ => μ) w) w i = μ := by
  unfold X21
  have : ∀ j, X19 (fun _ => μ) w j * w i j / X20 w j = μ * w i j := by
    intro j
    unfold X19
    rw [← Finset.mul_sum]
    have := hω j
    unfold X20 at this ⊢
    field_simp
  simp_rw [this]
  rw [← Finset.mul_sum]
  field_simp

/-- But it does not conserve mass in general (Sec. 5.1 says "conservative transfer"): two
particles with masses 1 and 0, one cell, weights 1 and 1/2, come back with masses 2/3 and 2/3. -/
theorem X21_not_conservative :
    let w : Fin 2 → Fin 1 → ℝ := fun i _ => if i = 0 then 1 else 1 / 2
    let m : Fin 2 → ℝ := fun i => if i = 0 then 1 else 0
    X21 (X19 m w) w 0 = 2 / 3 ∧ X21 (X19 m w) w 1 = 2 / 3 ∧ ∑ i, m i = 1 := by
  simp [X21, X19, X20, Fin.sum_univ_two]
  norm_num

/-- Eq. 22: `ω_j^T = Σ_i m_i w_ij`. -/
noncomputable def X22 (m : ι → ℝ) (w : ι → κ → ℝ) (j : κ) : ℝ := ∑ i, m i * w i j

/-- Eq. 23: `T_lj = Σ_i T_i m_i w_ij / ω_j^T`. -/
noncomputable def X23 (T m : ι → ℝ) (w : ι → κ → ℝ) (j : κ) : ℝ :=
  (∑ i, T i * m i * w i j) / X22 m w j

/-- Eq. 24: `T_pi = (Σ_j m_i w_ij / ω_j^T)⁻¹ Σ_j T_lj m_i w_ij / ω_j^T`. -/
noncomputable def X24 (Tl : κ → ℝ) (m : ι → ℝ) (w : ι → κ → ℝ) (i : ι) : ℝ :=
  (∑ j, m i * w i j / X22 m w j)⁻¹ * ∑ j, Tl j * (m i * w i j / X22 m w j)

theorem X22_X24_units :
    mass * one = mass ∧ temp * mass * one / mass = temp ∧
      (mass * one / mass) * (temp * (mass * one / mass)) = temp := by
  decide

/-- A weighted mean with non-negative weights and positive total lies in any interval holding
the values. -/
lemma wmean_mem {α : Type*} (s : Finset α) (a x : α → ℝ) {L U : ℝ} (ha : ∀ i ∈ s, 0 ≤ a i)
    (hpos : 0 < ∑ i ∈ s, a i) (hx : ∀ i ∈ s, a i ≠ 0 → L ≤ x i ∧ x i ≤ U) :
    L ≤ (∑ i ∈ s, x i * a i) / ∑ i ∈ s, a i ∧ (∑ i ∈ s, x i * a i) / ∑ i ∈ s, a i ≤ U := by
  constructor
  · rw [le_div_iff₀ hpos, Finset.mul_sum]
    apply Finset.sum_le_sum
    intro i hi
    by_cases h0 : a i = 0
    · simp [h0]
    · exact mul_le_mul_of_nonneg_right (hx i hi h0).1 (ha i hi)
  · rw [div_le_iff₀ hpos, Finset.mul_sum]
    apply Finset.sum_le_sum
    intro i hi
    by_cases h0 : a i = 0
    · simp [h0]
    · exact mul_le_mul_of_nonneg_right (hx i hi h0).2 (ha i hi)

omit [Fintype κ] in
/-- Eq. 23 is a weighted mean: the cell's liquid temperature lies between its particles'. -/
theorem X23_between (T m : ι → ℝ) (w : ι → κ → ℝ) (j : κ) {L U : ℝ}
    (hm : ∀ i, 0 ≤ m i) (hw : ∀ i, 0 ≤ w i j) (hpos : 0 < X22 m w j)
    (hT : ∀ i, L ≤ T i ∧ T i ≤ U) : L ≤ X23 T m w j ∧ X23 T m w j ≤ U := by
  have e : X23 T m w j = (∑ i, T i * (m i * w i j)) / ∑ i, m i * w i j := by
    unfold X23 X22; congr 1; congr 1; ext i; ring
  rw [e]
  exact wmean_mem Finset.univ (fun i => m i * w i j) T (fun i _ => mul_nonneg (hm i) (hw i))
    hpos (fun i _ _ => hT i)

/-- Eq. 24 is a weighted mean too: the particle's temperature lies between its cells'. -/
theorem X24_between (Tl : κ → ℝ) (m : ι → ℝ) (w : ι → κ → ℝ) (i : ι) {L U : ℝ}
    (hwt : ∀ j, 0 ≤ m i * w i j / X22 m w j) (hpos : 0 < ∑ j, m i * w i j / X22 m w j)
    (hT : ∀ j, L ≤ Tl j ∧ Tl j ≤ U) : L ≤ X24 Tl m w i ∧ X24 Tl m w i ≤ U := by
  have e : X24 Tl m w i = (∑ j, Tl j * (m i * w i j / X22 m w j)) /
      ∑ j, m i * w i j / X22 m w j := by
    unfold X24; rw [div_eq_inv_mul]
  rw [e]
  exact wmean_mem Finset.univ (fun j => m i * w i j / X22 m w j) Tl (fun j _ => hwt j) hpos
    (fun j _ _ => hT j)

/-- Eq. 25: `u_lj = (1/|N(j)|) Σ_i u_i`, a plain mean: it lies between the particles'. -/
theorem X25_between {n : ℕ} (hn : 0 < n) (u : Fin n → ℝ) {L U : ℝ} (hu : ∀ i, L ≤ u i ∧ u i ≤ U) :
    L ≤ (1 / n) * ∑ i, u i ∧ (1 / n) * ∑ i, u i ≤ U := by
  have h := wmean_mem Finset.univ (fun _ => (1 : ℝ)) u (fun _ _ => zero_le_one)
    (by simp; exact_mod_cast hn) (fun i _ _ => hu i)
  simp only [mul_one, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at h
  constructor
  · rw [one_div_mul_eq_div]; exact h.1
  · rw [one_div_mul_eq_div]; exact h.2

end Transfer

/-! ## X26 to X28, I1, I2. Droplet evaporation -/

/-- Eq. 26: `E_l = (T_l - T_amb) C_p ρ_l V_c`. -/
noncomputable def X26 (Tl Tamb Cp ρl Vc : ℝ) : ℝ := (Tl - Tamb) * Cp * ρl * Vc

/-- Eq. 27: `E_d = (C_p (T_B - T_amb) + Δ_v H⁰) ρ_d V_d`. -/
noncomputable def X27 (Cp TB Tamb ΔvH ρd Vd : ℝ) : ℝ := (Cp * (TB - Tamb) + ΔvH) * ρd * Vd

theorem X26_X27_units :
    temp * specificHeat * density * volume = energy ∧ specificHeat * temp = specificEnergy ∧
      specificEnergy * density * volume = energy ∧ energy * one = energy := by
  decide

/-- Transcriber's note 9, confirmed: if `ρ_d` is the liquid's own density (Eq. 27, App. A.1.1),
the cell's liquid mass is `n_d ρ_d V_d`, and Eq. 28's `ρ_l = ρ_d n_d` then forces `V_d = V_c`. -/
theorem X28_volume {ρl ρd nd Vd Vc : ℝ} (hmass : ρl * Vc = nd * ρd * Vd) (h28 : ρl = ρd * nd)
    (hn : nd * ρd ≠ 0) : Vd = Vc := by
  rw [h28] at hmass
  have : nd * ρd * (Vc - Vd) = 0 := by linarith
  rcases mul_eq_zero.1 this with h | h
  · exact absurd h hn
  · linarith

/-- I1 and I2: `n_e = E_l / E_d`, `ρ_e = n_e ρ_d V_d / V_c`. The evaporated mass `ρ_e V_c` is
the available energy over the energy per kilogram: energy is conserved. -/
theorem I2_evaporated_mass {Tl Tamb Cp ρl Vc TB ΔvH ρd Vd : ℝ} (hd : ρd * Vd ≠ 0)
    (hD : Cp * (TB - Tamb) + ΔvH ≠ 0) :
    X26 Tl Tamb Cp ρl Vc / X27 Cp TB Tamb ΔvH ρd Vd * ρd * Vd =
      X26 Tl Tamb Cp ρl Vc / (Cp * (TB - Tamb) + ΔvH) := by
  unfold X27
  have hρd : ρd ≠ 0 := left_ne_zero_of_mul hd
  have hVd : Vd ≠ 0 := right_ne_zero_of_mul hd
  field_simp

/-- Eq. 9's right-hand side is the per-step evaporated density of Sec. 5.4: `-X9 = ρ_e`, so a
factor of one per time step is implicit in Eq. 9 (note 6). -/
theorem X9_eq_rho_e {Tl Tamb Cp ρl Vc TB ΔvH ρd Vd : ℝ} (hd : ρd * Vd ≠ 0) (hV : Vc ≠ 0)
    (hD : Cp * (TB - Tamb) + ΔvH ≠ 0) :
    -X9 ρl Cp Tl Tamb TB ΔvH = X26 Tl Tamb Cp ρl Vc / X27 Cp TB Tamb ΔvH ρd Vd * ρd * Vd / Vc := by
  rw [I2_evaporated_mass hd hD]
  unfold X9 X26
  field_simp

/-- No more evaporates than there is, as long as `C_p (T_l - T_amb) ≤ C_p (T_B - T_amb) + Δ_v H`
(for water, `T_l` up to `T_B` plus about 540 K). -/
theorem I2_le_liquid {ρl Cp Tl Tamb TB ΔvH : ℝ} (hρ : 0 ≤ ρl) (hD : 0 < Cp * (TB - Tamb) + ΔvH)
    (h : Cp * (Tl - Tamb) ≤ Cp * (TB - Tamb) + ΔvH) : -X9 ρl Cp Tl Tamb TB ΔvH ≤ ρl := by
  unfold X9
  rw [neg_div, neg_neg, div_le_iff₀ hD]
  nlinarith

/-! ## X29 to X32. Mixing the vapour in -/

/-- Eq. 29: `ρ' = ρ / (ρ + Δρ)`. -/
noncomputable def X29 (ρ Δρ : ℝ) : ℝ := ρ / (ρ + Δρ)

theorem X29_mem {ρ Δρ : ℝ} (hρ : 0 < ρ) (hΔ : 0 ≤ Δρ) : 0 < X29 ρ Δρ ∧ X29 ρ Δρ ≤ 1 := by
  unfold X29
  constructor
  · positivity
  · rw [div_le_one (by linarith)]; linarith

/-- Transcriber's note 14, confirmed: `ρ'` is the remaining gas fraction, 1 exactly when nothing
evaporates (not the "evaporated density fraction"). -/
theorem X29_eq_one_iff {ρ Δρ : ℝ} (hρ : 0 < ρ) (hΔ : 0 ≤ Δρ) : X29 ρ Δρ = 1 ↔ Δρ = 0 := by
  unfold X29
  rw [div_eq_one_iff_eq (by linarith)]
  constructor <;> intro h <;> linarith

/-- Eq. 30: `T' = (1 - ρ') T_B + ρ' T`. -/
noncomputable def X30 (ρ Δρ TB T : ℝ) : ℝ := (1 - X29 ρ Δρ) * TB + X29 ρ Δρ * T

/-- It is the mixing of two masses at equal heat capacity: `(ρ + Δρ) T' = ρ T + Δρ T_B`. -/
theorem X30_enthalpy {ρ Δρ TB T : ℝ} (h : ρ + Δρ ≠ 0) :
    (ρ + Δρ) * X30 ρ Δρ TB T = ρ * T + Δρ * TB := by
  unfold X30 X29
  field_simp
  ring

theorem X30_between {ρ Δρ TB T : ℝ} (hρ : 0 < ρ) (hΔ : 0 ≤ Δρ) :
    min TB T ≤ X30 ρ Δρ TB T ∧ X30 ρ Δρ TB T ≤ max TB T := by
  obtain ⟨h0, h1⟩ := X29_mem hρ hΔ
  unfold X30
  have a := min_le_left TB T
  have b := min_le_right TB T
  have c := le_max_left TB T
  have d := le_max_right TB T
  constructor <;> nlinarith

/-- Eq. 31: `Y_i' = ρ' Y_i`. -/
noncomputable def X31 (ρ Δρ Y : ℝ) : ℝ := X29 ρ Δρ * Y

/-- Each existing species keeps its mass: `(ρ + Δρ) Y_i' = ρ Y_i`. -/
theorem X31_species_mass {ρ Δρ Y : ℝ} (h : ρ + Δρ ≠ 0) : (ρ + Δρ) * X31 ρ Δρ Y = ρ * Y := by
  unfold X31 X29; field_simp

/-- Transcriber's note 15, confirmed: with `Σ Y_i = 1` the rescaled fractions sum to `ρ' < 1`
whenever something evaporates. -/
theorem X31_sum {ι : Type*} (s : Finset ι) (Y : ι → ℝ) (hY : ∑ i ∈ s, Y i = 1) {ρ Δρ : ℝ}
    (hρ : 0 < ρ) (hΔ : 0 < Δρ) : ∑ i ∈ s, X31 ρ Δρ (Y i) = X29 ρ Δρ ∧ X29 ρ Δρ < 1 := by
  constructor
  · unfold X31; rw [← Finset.mul_sum, hY, mul_one]
  · unfold X29; rw [div_lt_one (by linarith)]; linarith

/-- Note 15's repair: adding `1 - ρ'` to the vapour species restores the sum to one. -/
theorem X31_fix {ι : Type*} (s : Finset ι) (Y : ι → ℝ) (hY : ∑ i ∈ s, Y i = 1) (ρ Δρ : ℝ) :
    ∑ i ∈ s, X31 ρ Δρ (Y i) + (1 - X29 ρ Δρ) = 1 := by
  unfold X31; rw [← Finset.mul_sum, hY]; ring

/-- Eq. 32: `u' = (1 - ρ') u_l + ρ' u`, momentum mixing: `(ρ + Δρ) u' = ρ u + Δρ u_l`. -/
theorem X32_momentum {ρ Δρ ul u : ℝ} (h : ρ + Δρ ≠ 0) :
    (ρ + Δρ) * ((1 - X29 ρ Δρ) * ul + X29 ρ Δρ * u) = ρ * u + Δρ * ul := by
  unfold X29; field_simp; ring

/-- With nothing evaporated, Eqs. 30 to 32 change nothing. -/
theorem X30_X32_identity {ρ : ℝ} (hρ : 0 < ρ) (TB T Y ul u : ℝ) :
    X30 ρ 0 TB T = T ∧ X31 ρ 0 Y = Y ∧ (1 - X29 ρ 0) * ul + X29 ρ 0 * u = u := by
  have : X29 ρ 0 = 1 := by unfold X29; rw [add_zero, div_self hρ.ne']
  unfold X30 X31
  rw [this]
  refine ⟨by ring, by ring, by ring⟩

theorem X29_X32_units : density / density = one ∧ one * temp = temp ∧ one * velocity = velocity := by
  decide

/-! ## Transcriber's note 17: Table 1's heat capacities -/

/-- An ideal gas has a molar heat capacity of at least `5R/2` (translation alone gives
`C_v = 3R/2`, and `C_p = C_v + R`). Read as J kg⁻¹ K⁻¹, Table 1's 63 (acetylene) and 114
(cyclopropane) fall below `5R/(2M)`; the other four lie above it. Note 17 confirmed, given that
physical bound. -/
theorem note17_cp :
    (63 : ℝ) < 5 * R / (2 * 0.026) ∧ (114 : ℝ) < 5 * R / (2 * 0.042) ∧
      5 * R / (2 * 0.05812) < 1720 ∧ 5 * R / (2 * 0.0441) < 1670 ∧
      5 * R / (2 * 0.01604) < 2230 ∧ 5 * R / (2 * 0.02805) < 1550 ∧
      5 * R / 2 < 63 ∧ 5 * R / 2 < 114 := by
  norm_num [R]

/-! ## Fig. 17: McCaffrey's temperature exponents -/

/-- Fig. 17's exponents `η = 0, -1, -5/3` of `ΔT ∝ (z/Q^(2/5))^η` are `2η' - 1` for McCaffrey's
velocity exponents `η' = 1/2, 0, -1/3` (`FireDynamics.FD12_from_FD11`). -/
theorem F17_mccaffrey : (2 * (1 / 2 : ℝ) - 1 = 0) ∧ (2 * (0 : ℝ) - 1 = -1) ∧
    (2 * (-1 / 3 : ℝ) - 1 = -5 / 3) := by
  norm_num

end FireEgress.FireX
