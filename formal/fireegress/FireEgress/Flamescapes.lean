import Mathlib
import FireEgress.Dim
import FireEgress.FireX

/-!
# Flamescapes (Wrede, Wagner et al. 2026): every numbered equation and the supplement

Wrede, Wagner, Pałubicki, Michels and Pirk (2026), *Flamescapes: Simulating the Wildlandfire
Urban Interface*, ACM Transactions on Graphics 45(6) 246, https://doi.org/10.1145/3842518
(CC BY 4.0). The equations are those of the transcription
`wiki/sources/papers/flamescapes-2026.md`, its supplement `flamescapes-2026-supplement.md` and
the catalogue `flamescapes-2026-equations.json`: F1 to F19, S1, the inline definitions I1 to I4,
line 2 of the main Algorithm 1 (A1-2), S-I1, S-I2 and the lines of supplement Algorithm 2.
Each equation is a Lean definition at a point; units use `FireX.DN`. The gas chemistry is
Fire-X's, and the theorems that say so reuse `FireX.lean`.

## Which theorem covers which equation

* F1, F2, volume fractions and moisture. `F1_units`, `F1_sum_le_one`, `F1_leaf_overflow`;
  `F2_bounds`.
* F3 to F5, I1, I2, interior fuel. `F3_F4_units`; `F5_expected_total` (draws with mean
  `M_b / N_b` have expected total `M_b`). I3, durability: `I3_permanent`.
* F6, momentum. `F6_units`; `note9_printed_pushes_hot_gas_down` (note 9);
  `F6_intended_eq_port`, `F6_vs_X1`, `F6_port_vs_X1`, `F6_minus_X1`, `F6_X1_at_1200K`,
  `F6_X1_printed_opposite` (the relation to Fire-X's Eq. 1, below).
* F7, equation of state. `note8_units` (note 8), `F7_with_pressure`, `F7_from_ideal_gas`,
  `F7_strictAntiOn`, `F7_table5`.
* F8 to F12, the gas. `F8_units` (note 7); `note3_radiation_heats` (note 3, with F17);
  `F9_eq_X6`, `F9_table3`, `F9_units`; `F10_eq_X5`, `F10_pos`, `F10_strictMonoOn`,
  `F10_table3_is_methane`, `F10_units`; `F11_units`; `F12_eq_X7`, `F12_units`.
* F13, solid mixtures. `F13_cp_between`, `F13_rho_between`, `F13_units`,
  `F13_heat_capacity_not_additive`.
* F14, F15, pyrolysis and oxidation. `note6_units` (note 6), `F14_F15_heat_units`,
  `note5_signs` (note 5).
* F16, F17, I4, radiation. `F16_sign`, `F16_units`; `note10_units` (note 10); `Pabs_mem`,
  `Pabs_compose` (`1 - e^{-a(l₁+l₂)} = 1 - (1 - P₁)(1 - P₂)`), `Pabs_monotone`.
* F18, F19, the solid and its interface. `F18_units`; `note12_units` (note 12), `F19_out`.
  A1-2: `A12_equilibrium`.
* Supplement Algorithm 1, radiation in expectation. `SA1_balance`, `SA1_pass_exp`.
* Supplement Algorithm 2, particles to grid. `notes_S3_S4_units` (supplement notes 3 and 4),
  `SA2_line7_fixed_energy`, `SA2_ember_balance`, `SA2_line10`, `SA2_line11`.
* S1, S-I1, S-I2, building rules. `S1_ext`, `Side.opp_opp`, `SI2_symm`.

## Buoyancy, exactly

As printed, Eq. 6's term is `-g (1 - ρ_amb/ρ_g)`; with Table 5's `g = -9.81` m s⁻² it pushes
gas lighter than ambient downward (note 9). Read as intended, `+g (1 - ρ_amb/ρ_g)` with `g`
pointing down is `|g| (ρ_amb/ρ_g - 1)` upward, which is the port's form. Under the ideal gas
law at fixed pressure and composition (`ρ T = ρ_amb T_amb`) it equals Fire-X's
`-g (1 - T_amb/T)` multiplied by `ρ_amb/ρ_g = T/T_amb`: the two have the same sign and agree to
first order in `T - T_amb`, differing by `|g| (T - T_amb)² / (T T_amb)`; at 1200 K over 300 K
Fire-X gives `0.75 |g|` and Flamescapes `3 |g|`. Fire-X's form is Boussinesq's (the density
deficit over `ρ_amb`) and stays below `|g|`; the Flamescapes form divides by `ρ_g`, the
acceleration of a parcel under Archimedes' force, and has no bound. Under one convention for
`g` the two printed terms point in opposite directions (`F6_X1_printed_opposite`).

## The transcriber's notes

Confirmed by a proof: main notes 3, 5, 6, 7, 8, 9 (with Table 5's `g = -9.81`), 10 and 12, and
supplement notes 3 and 4. Not decided here: main note 2 (a gradient where a divergence is
meant is a typing slip that units cannot see, both carry W m⁻³), notes 1, 4, 11 and 13 to 18
(notation, prose, figure references and table ranges), and supplement notes 1, 2 and 5 to 12;
for supplement note 5, `SA2_line11` shows that taking the maximum with either temperature can
only raise the gas reaction rate. No note is refuted.

Found beyond the notes: the leaf-thickness correction can push a volume fraction above one
(`F1_leaf_overflow`); Eq. 13's volume-weighted specific heat makes `ρ_s C_p,s` differ from the
constituents' summed heat capacities, 8% low for half wood and half moisture
(`F13_heat_capacity_not_additive`); Table 3's gas values are Fire-X's Table 1 methane; and
printed Eq. 7 with Table 5's numbers gives air at 300 K a density of 1.36e-5 kg m⁻³, while the
pressure form gives 1.171 kg m⁻³, within 0.2% of Table 5's `ρ_amb` (`F7_table5`).

## What is not covered

* The partial differential equations are stated pointwise; the authors' GPU schemes are not
  formalised. `FlamescapesPort.lean` covers Studio's schemes.
* The conversion of Eqs. 14 and 15 from a rate of solid volume fraction to a gas mass fraction
  (note 6's second point) is not given in the paper and is not formalised. The Arrhenius forms
  of `v_p,s` and `v_c,s` are not printed; if they share Eq. 10's form, `F10_pos` and
  `F10_strictMonoOn` apply to them.
* The fuel graph, the wave function collapse solver beyond S1 and the symmetry of S-I2, the
  motion of embers and flame-front particles, and the renderer.
* `F5_expected_total` is about the distribution; the code's draws (Box-Muller from a seeded
  generator) are not modelled.
-/

namespace FireEgress.Flamescapes

open Real MeasureTheory ProbabilityTheory FireEgress.FireX FireEgress.FireX.DN

/-! ## Constants (Tables 3 and 5) -/

/-- Table 5: gravitational acceleration, the signed vertical component (m s⁻²). -/
noncomputable def gTable : ℝ := -9.81
/-- Table 5: ambient gas density (kg m⁻³). -/
noncomputable def rhoAmbTable : ℝ := 1.1726
/-- Table 5: ambient temperature (K). -/
noncomputable def TambTable : ℝ := 300
/-- Table 3: solid heat of combustion (J mol⁻¹). -/
noncomputable def dHcs : ℝ := -2.80e6
/-- Table 3: solid heat of pyrolysis (J kg⁻¹). -/
noncomputable def dHps : ℝ := 2.50e5
/-- Table 3: gas heat of combustion (J mol⁻¹). -/
noncomputable def dHcg : ℝ := -8.34e5

/-! ## F1, F2. Volume fractions and moisture -/

/-- Eq. 1: `φ_i = Σ_r l_{i,r} / Σ_r d_r`. -/
noncomputable def F1 {ρ : Type*} (s : Finset ρ) (l d : ρ → ℝ) : ℝ := (∑ r ∈ s, l r) / ∑ r ∈ s, d r

theorem F1_units : length / length = one := by decide

/-- If no ray spends longer inside the materials than it travels across the voxel, the volume
fractions are non-negative and sum to at most one. -/
theorem F1_sum_le_one {ρ ι : Type*} (s : Finset ρ) (mats : Finset ι) (l : ι → ρ → ℝ)
    (d : ρ → ℝ) (hd : 0 < ∑ r ∈ s, d r) (hl : ∀ i ∈ mats, ∀ r ∈ s, 0 ≤ l i r)
    (hcap : ∀ r ∈ s, ∑ i ∈ mats, l i r ≤ d r) :
    (∀ i ∈ mats, 0 ≤ F1 s (l i) d) ∧ ∑ i ∈ mats, F1 s (l i) d ≤ 1 := by
  constructor
  · intro i hi
    exact div_nonneg (Finset.sum_nonneg (fun r hr => hl i hi r hr)) hd.le
  · unfold F1
    rw [← Finset.sum_div, div_le_one hd, Finset.sum_comm]
    exact Finset.sum_le_sum hcap

/-- The leaf-thickness correction can break that hypothesis: thirteen billboard hits of
`δ_l = 0.01` m (Table 5's largest) on one ray across a 0.125 m voxel give `φ = 1.04 > 1`. -/
theorem F1_leaf_overflow :
    F1 ({0} : Finset ℕ) (fun _ => 13 * 0.01) (fun _ => 0.125) = 1.04 := by
  norm_num [F1]

/-- Eq. 2: `φ_moisture = r_d φ_dense + r_s φ_sparse`. -/
noncomputable def F2 (rd rs φd φs : ℝ) : ℝ := rd * φd + rs * φs

/-- With ratios in `[0, 1]` the moisture fraction is at most the biomass fraction. -/
theorem F2_bounds {rd rs φd φs : ℝ} (hd0 : 0 ≤ rd) (hd1 : rd ≤ 1) (hs0 : 0 ≤ rs) (hs1 : rs ≤ 1)
    (hφd : 0 ≤ φd) (hφs : 0 ≤ φs) : 0 ≤ F2 rd rs φd φs ∧ F2 rd rs φd φs ≤ φd + φs := by
  unfold F2
  constructor
  · positivity
  · nlinarith

/-! ## F3 to F5, I1, I2. Interior fuel -/

/-- Eq. 3: `Q_b = q''_b A`. -/
noncomputable def F3 (q A : ℝ) : ℝ := q * A
/-- Eq. 4: `M_b = Q_b / Δ_c H^eff`. -/
noncomputable def F4 (Q ΔH : ℝ) : ℝ := Q / ΔH
/-- I2: `μ_b = M_b / N_b`. -/
noncomputable def I2 (M : ℝ) (N : ℕ) : ℝ := M / N

/-- Units: `Q_b` is an energy, and `M_b` a mass when `Δ_c H^eff` is per kilogram (the paper
gives no unit). -/
theorem F3_F4_units :
    energy / area * area = energy ∧ energy / specificEnergy = mass ∧
      energy / molarEnergy ≠ mass := by
  decide

/-- Eq. 5 with I2: per-voxel draws `m_{b,t} ~ N(M_b / N_b, σ²)` have expected total `M_b`
(no independence needed). -/
theorem F5_expected_total {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω) {N : ℕ} (hN : 0 < N)
    (M : ℝ) (v : NNReal) (m : Fin N → Ω → ℝ)
    (hlaw : ∀ t, HasLaw (m t) (gaussianReal (I2 M N) v) P) :
    ∫ ω, ∑ t, m t ω ∂P = M := by
  have hid : Integrable id (gaussianReal (I2 M N) v) :=
    (memLp_id_gaussianReal (μ := I2 M N) (v := v) 1).integrable le_rfl
  have hint : ∀ t, Integrable (m t) P := fun t => (hlaw t).integrable hid
  rw [integral_finsetSum _ (fun t _ => hint t)]
  have hmean : ∀ t, ∫ ω, m t ω ∂P = I2 M N := by
    intro t
    rw [(hlaw t).integral_eq, integral_id_gaussianReal]
  simp_rw [hmean]
  rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  unfold I2
  have : (N : ℝ) ≠ 0 := by exact_mod_cast hN.ne'
  field_simp

/-- I3: a module is destroyed once its durability falls below `τ D_max`. Since fuel only
burns away (durability never rises), destruction is permanent. -/
theorem I3_permanent {D : ℝ → ℝ} (hD : Antitone D) (τ Dmax : ℝ) {t t' : ℝ} (htt : t ≤ t')
    (h : D t < τ * Dmax) : D t' < τ * Dmax :=
  lt_of_le_of_lt (hD htt) h

/-! ## F6. Buoyancy (note 9) and Fire-X's form -/

/-- Eq. 6's buoyancy term as printed, vertical component: `-g_z (1 - ρ_amb / ρ_g)`. -/
noncomputable def F6_buoy (gz ρa ρ : ℝ) : ℝ := -gz * (1 - ρa / ρ)

/-- The term with the sign read as intended (note 9's Boussinesq-type form):
`+g_z (1 - ρ_amb / ρ_g)`. -/
noncomputable def F6_buoy_intended (gz ρa ρ : ℝ) : ℝ := gz * (1 - ρa / ρ)

theorem F6_units :
    pressure / length / density = accel ∧ accel * (density / density) = accel := by
  decide

/-- Transcriber's note 9, confirmed: with Table 5's `g = -9.81` m s⁻² (pointing down) the
printed term accelerates gas lighter than ambient downward. -/
theorem note9_printed_pushes_hot_gas_down {ρa ρ : ℝ} (hρ : 0 < ρ) (hl : ρ < ρa) :
    F6_buoy gTable ρa ρ < 0 := by
  unfold F6_buoy gTable
  have : 1 < ρa / ρ := (one_lt_div hρ).2 hl
  nlinarith

/-- Read as intended, with gravity `-|g|` pointing down, the term is the port's
`(ρ₀/ρ - 1) |g|` upward: positive exactly for gas lighter than ambient. -/
theorem F6_intended_eq_port {g ρa ρ : ℝ} (hρ : 0 < ρ) :
    F6_buoy_intended (-g) ρa ρ = g * (ρa / ρ - 1) ∧
      (0 < g → (0 < F6_buoy_intended (-g) ρa ρ ↔ ρ < ρa)) := by
  constructor
  · unfold F6_buoy_intended; ring
  · intro hg
    unfold F6_buoy_intended
    constructor
    · intro h
      by_contra h'
      have : ρa / ρ ≤ 1 := (div_le_one hρ).2 (not_lt.1 h')
      nlinarith
    · intro h
      have : 1 < ρa / ρ := (one_lt_div hρ).2 h
      nlinarith

/-- The exact relation to Fire-X's Eq. 1 under the ideal gas law at fixed pressure and
composition (`ρ T = ρ_amb T_amb`): the intended Flamescapes term is Fire-X's
`-g_z (1 - T_amb/T)` multiplied by `ρ_amb / ρ = T / T_amb`; same sign, larger by that factor
for gas hotter than ambient. -/
theorem F6_vs_X1 {gz ρa ρ T Ta : ℝ} (hρ : 0 < ρ) (hT : 0 < T) (hTa : 0 < Ta)
    (hgas : ρ * T = ρa * Ta) :
    F6_buoy_intended gz ρa ρ = (ρa / ρ) * X1_buoy gz Ta T ∧ ρa / ρ = T / Ta := by
  have hρa : ρa = ρ * T / Ta := by field_simp; linarith
  constructor
  · unfold F6_buoy_intended X1_buoy
    rw [hρa]
    field_simp
    ring
  · rw [hρa]; field_simp

/-- In upward components (gravity `-g`, `g > 0`): the port's `g (ρ₀/ρ - 1)` is
`(ρ₀/ρ)` times Fire-X's `g (1 - T_amb/T)`. -/
theorem F6_port_vs_X1 {g ρa ρ T Ta : ℝ} (hρ : 0 < ρ) (hT : 0 < T) (hTa : 0 < Ta)
    (hgas : ρ * T = ρa * Ta) :
    g * (ρa / ρ - 1) = (ρa / ρ) * X1_buoy (-g) Ta T := by
  have hρa : ρa = ρ * T / Ta := by field_simp; linarith
  unfold X1_buoy
  rw [hρa]
  field_simp

/-- They agree to first order in `T - T_amb`: the difference is `g (T - T_amb)² / (T T_amb)`. -/
theorem F6_minus_X1 {g T Ta : ℝ} (hT : 0 < T) (hTa : 0 < Ta) :
    g * (T / Ta - 1) - X1_buoy (-g) Ta T = g * (T - Ta) ^ 2 / (T * Ta) := by
  unfold X1_buoy
  field_simp

/-- But not at fire temperatures: at 1200 K over 300 K, Fire-X's buoyancy is `0.75 |g|` and the
intended Flamescapes term `3 |g|`. -/
theorem F6_X1_at_1200K (g : ℝ) :
    X1_buoy (-g) 300 1200 = 0.75 * g ∧ g * (1200 / 300 - 1) = 3 * g := by
  unfold X1_buoy; constructor <;> ring

/-- Under one convention for `g` the two printed terms point opposite ways: with the ideal gas
law and `ρ ≠ ρ_amb`, the product of Fire-X's and Flamescapes' printed terms is negative, so at
most one of the papers has the sign right for a given `g`. -/
theorem F6_X1_printed_opposite {gz ρa ρ T Ta : ℝ} (hg : gz ≠ 0) (hρ : 0 < ρ) (hT : 0 < T)
    (hTa : 0 < Ta) (hgas : ρ * T = ρa * Ta) (hne : ρ ≠ ρa) :
    F6_buoy gz ρa ρ * X1_buoy gz Ta T < 0 := by
  have hρa : ρa = ρ * T / Ta := by field_simp; linarith
  have hTne : T ≠ Ta := by
    intro h; apply hne; rw [hρa, h]; field_simp
  unfold F6_buoy X1_buoy
  rw [hρa]
  have e : -gz * (1 - ρ * T / Ta / ρ) * (-gz * (1 - Ta / T)) =
      -(gz ^ 2 * (T - Ta) ^ 2 / (Ta * T)) := by
    field_simp; ring
  rw [e]
  have : 0 < gz ^ 2 * (T - Ta) ^ 2 / (Ta * T) := by
    have h1 : 0 < gz ^ 2 := by positivity
    have h2 : 0 < (T - Ta) ^ 2 := by
      have : T - Ta ≠ 0 := sub_ne_zero.2 hTne
      positivity
    positivity
  linarith

/-! ## F7. Equation of state (note 8) -/

/-- Eq. 7 as printed: `ρ_g = ρ_amb (R T_g Σ Y_i / M_i)⁻¹`. -/
noncomputable def F7 {ι : Type*} [Fintype ι] (ρa T : ℝ) (Y M : ι → ℝ) : ℝ :=
  ρa / (FireX.R * T * ∑ i, Y i / M i)

/-- Transcriber's note 8, confirmed: as printed the right-hand side is not a density; with a
pressure in place of `ρ_amb` it is. -/
theorem note8_units :
    density / (gasConst * temp * (one / molarMass)) ≠ density ∧
      pressure / (gasConst * temp * (one / molarMass)) = density := by
  decide

/-- With the ambient pressure in place of `ρ_amb`, Eq. 7 is Fire-X's corrected Eq. 8, the ideal
gas law over the mixture. -/
theorem F7_with_pressure {ι : Type*} [Fintype ι] (p T : ℝ) (Y M : ι → ℝ) :
    F7 p T Y M = X8 p T Y M := rfl

/-- And it follows from that law: at the pressure of the ambient state the density is the
ambient density scaled by `T_amb W_amb / (T W)`, with `W = Σ Y_i / M_i`. -/
theorem F7_from_ideal_gas {ι : Type*} [Fintype ι] {p ρa Ta T : ℝ} {Y Ya M : ι → ℝ}
    (hTa : 0 < Ta) (hT : 0 < T) (hWa : 0 < ∑ i, Ya i / M i) (hW : 0 < ∑ i, Y i / M i)
    (hamb : p = ρa * FireX.R * Ta * ∑ i, Ya i / M i) :
    X8 p T Y M = ρa * (Ta * ∑ i, Ya i / M i) / (T * ∑ i, Y i / M i) := by
  unfold X8
  rw [hamb]
  have := FireX.R_pos
  field_simp

/-- Hot gas is lighter, in the printed form too. -/
theorem F7_strictAntiOn {ι : Type*} [Fintype ι] {ρa : ℝ} {Y M : ι → ℝ} (hρ : 0 < ρa)
    (hW : 0 < ∑ i, Y i / M i) : StrictAntiOn (fun T => F7 ρa T Y M) (Set.Ioi 0) :=
  X8_strictAntiOn hρ hW

/-- The species of Table 5's air: O₂ and N₂ by mass. -/
noncomputable def airY : Fin 2 → ℝ := ![0.232, 0.768]
noncomputable def airM : Fin 2 → ℝ := ![0.032, 0.028]

/-- With Table 5's numbers the printed form gives air at 300 K a density near 1.36e-5 kg m⁻³;
with 101 325 Pa in place of `ρ_amb` it gives 1.171 kg m⁻³, within 0.2% of Table 5's 1.1726. -/
theorem F7_table5 :
    1.35e-5 < F7 rhoAmbTable TambTable airY airM ∧ F7 rhoAmbTable TambTable airY airM < 1.36e-5 ∧
      1.170 < F7 101325 TambTable airY airM ∧ F7 101325 TambTable airY airM < 1.1726 := by
  norm_num [F7, rhoAmbTable, TambTable, airY, airM, FireX.R, Fin.sum_univ_two]

/-! ## F8, F9, F10, F11, F12. The gas -/

/-- Eq. 8's right-hand side, with `∇q_rad` the radiation term as printed. -/
noncomputable def F8_rhs (divkT gradq qc : ℝ) : ℝ := divkT - gradq + qc

/-- Units of Eq. 8 (with a divergence for the conduction term): every term is W m⁻³; A.4's
J s⁻¹ for the source terms is not (transcriber's note 7, confirmed). -/
theorem F8_units :
    density * specificHeat * temp / time = volPower ∧
      conductivity * temp / area = volPower ∧ power ≠ volPower := by
  decide

/-- Eq. 17: `-∇q_rad = 4 Y_s κ_s σ (T_g⁴ - T_amb⁴)`. -/
noncomputable def F17 (Y κ σ T Ta : ℝ) : ℝ := 4 * Y * κ * σ * (T ^ 4 - Ta ^ 4)

/-- Transcriber's note 3, confirmed: substituting Eq. 17 into Eq. 8, hot sooty gas gains
energy from its own radiation term. -/
theorem note3_radiation_heats {Y κ σ T Ta divkT qc : ℝ} (hY : 0 < Y) (hκ : 0 < κ) (hσ : 0 < σ)
    (hTa : 0 ≤ Ta) (hT : Ta < T) :
    F8_rhs divkT (-F17 Y κ σ T Ta) qc - (divkT + qc) = F17 Y κ σ T Ta ∧ 0 < F17 Y κ σ T Ta := by
  constructor
  · unfold F8_rhs; ring
  · unfold F17
    have : Ta ^ 4 < T ^ 4 := pow_lt_pow_left₀ hT hTa (by norm_num)
    have : 0 < T ^ 4 - Ta ^ 4 := by linarith
    positivity

/-- Eq. 9: `q̇_c,g = -Δ_c,g H⁰ v_c,g`, Fire-X's Eq. 6 with `φ = 1`. -/
noncomputable def F9 (ΔH v : ℝ) : ℝ := -ΔH * v

theorem F9_eq_X6 (ΔH v : ℝ) : F9 ΔH v = X6 1 v ΔH := by unfold F9 X6; ring

/-- With Table 3's `-8.34e5` J mol⁻¹ the gas releases heat; that is Fire-X's methane value. -/
theorem F9_table3 {v : ℝ} (hv : 0 < v) : 0 < F9 dHcg v ∧ dHcg = 1000 * X4 1 4 0 := by
  constructor
  · unfold F9 dHcg; nlinarith
  · norm_num [dHcg, X4]

theorem F9_units : molarEnergy * rate = volPower := by decide

/-- Eq. 10: `v_c,g = A exp(-E_a / (R T_g)) c_f^a c_o^b`, Fire-X's Eq. 5 with `n = 0`. -/
noncomputable def F10 (A Ea T cf co a b : ℝ) : ℝ := A * Real.exp (-Ea / (FireX.R * T)) * cf ^ a * co ^ b

theorem F10_eq_X5 (A Ea T cf co a b : ℝ) : F10 A Ea T cf co a b = X5 A 0 Ea T cf co a b := by
  unfold F10 X5; rw [Real.rpow_zero, mul_one]

/-- Arrhenius rates are positive. -/
theorem F10_pos {A Ea T cf co a b : ℝ} (hA : 0 < A) (hT : 0 < T) (hf : 0 < cf) (ho : 0 < co) :
    0 < F10 A Ea T cf co a b := by
  rw [F10_eq_X5]; exact X5_pos hA hT hf ho

/-- And they increase with temperature. -/
theorem F10_strictMonoOn {A Ea cf co a b : ℝ} (hA : 0 < A) (hE : 0 < Ea) (hf : 0 < cf)
    (ho : 0 < co) : StrictMonoOn (fun T => F10 A Ea T cf co a b) (Set.Ioi 0) := by
  have h := X5_strictMonoOn (n := 0) (a := a) (b := b) hA le_rfl hE hf ho
  intro T₁ h₁ T₂ h₂ h12
  simp only [F10_eq_X5]
  exact h h₁ h₂ h12

/-- Table 3's gas values (`A = 8.30e5`, `a = -0.30`, `b = 1.30`) are Fire-X's Table 1 methane. -/
theorem F10_table3_is_methane : (8.30e5 : ℝ) = 830000 ∧ (-0.30 : ℝ) = -0.3 ∧ (1.30 : ℝ) = 1.3 := by
  norm_num

theorem F10_units : molarEnergy / (gasConst * temp) = one := by decide

/-- Eq. 11's right-hand side for one species, with the advective term `u · ∇Y_i` moved to it:
`-u · ∇Y_i + D_i ∇²Y_i + ṁ_c,g + ṁ_e,s + ṁ_p,s + ṁ_c,s`. -/
noncomputable def F11_rhs (uGradY D lapY mcg mes mps mcs : ℝ) : ℝ :=
  -uGradY + D * lapY + mcg + mes + mps + mcs

/-- Eq. 11: every term is s⁻¹. -/
theorem F11_units :
    one / time = velocity * (one / length) ∧ one / time = diffusivity * (one / area) := by
  decide

/-- Eq. 12: `ṁ_c,g = v_c,g q_i M_i / ρ_g`, Fire-X's Eq. 7. -/
theorem F12_eq_X7 (v q M ρ : ℝ) : v * q * M / ρ = X7 v q M ρ := rfl

theorem F12_units : one / time = rate * one * molarMass / density := by decide

/-! ## F13. Solid mixtures -/

/-- Eq. 13: `ρ_s = Σ φ_i ρ_i`. -/
noncomputable def F13_rho {ι : Type*} (s : Finset ι) (φ ρ : ι → ℝ) : ℝ := ∑ i ∈ s, φ i * ρ i

/-- Eq. 13: `C_p,s = Σ φ_i C_p,i / Σ φ_i` (and conductivity alike). -/
noncomputable def F13_cp {ι : Type*} (s : Finset ι) (φ C : ι → ℝ) : ℝ :=
  (∑ i ∈ s, φ i * C i) / ∑ i ∈ s, φ i

/-- The mixture's specific heat (and conductivity) lies between the constituents'. -/
theorem F13_cp_between {ι : Type*} (s : Finset ι) (φ C : ι → ℝ) {L U : ℝ}
    (hφ : ∀ i ∈ s, 0 ≤ φ i) (hpos : 0 < ∑ i ∈ s, φ i) (hC : ∀ i ∈ s, L ≤ C i ∧ C i ≤ U) :
    L ≤ F13_cp s φ C ∧ F13_cp s φ C ≤ U := by
  have e : F13_cp s φ C = (∑ i ∈ s, C i * φ i) / ∑ i ∈ s, φ i := by
    unfold F13_cp; congr 1; congr 1; ext i; ring
  rw [e]
  exact wmean_mem s φ C hφ hpos (fun i hi _ => hC i hi)

/-- The density lies between `Σφ` times the least and the largest constituent density; between
them when the fractions sum to one. -/
theorem F13_rho_between {ι : Type*} (s : Finset ι) (φ ρ : ι → ℝ) {L U : ℝ}
    (hφ : ∀ i ∈ s, 0 ≤ φ i) (hρ : ∀ i ∈ s, L ≤ ρ i ∧ ρ i ≤ U) :
    (∑ i ∈ s, φ i) * L ≤ F13_rho s φ ρ ∧ F13_rho s φ ρ ≤ (∑ i ∈ s, φ i) * U := by
  unfold F13_rho
  rw [Finset.sum_mul, Finset.sum_mul]
  constructor
  · exact Finset.sum_le_sum (fun i hi => mul_le_mul_of_nonneg_left (hρ i hi).1 (hφ i hi))
  · exact Finset.sum_le_sum (fun i hi => mul_le_mul_of_nonneg_left (hρ i hi).2 (hφ i hi))

theorem F13_units : one * density = density ∧ one * specificHeat / one = specificHeat := by
  decide

/-- The volumetric heat capacity `ρ_s C_p,s` of Eq. 13 is not the sum `Σ φ_i ρ_i C_p,i` of the
constituents': half wood and half moisture (Table 4) gives 2 473 600 J m⁻³ K⁻¹ against
2 692 000, 8% low. (The port sums the constituents' heat capacities.) -/
theorem F13_heat_capacity_not_additive :
    let φ : Fin 2 → ℝ := ![0.5, 0.5]
    let ρ : Fin 2 → ℝ := ![600, 1000]
    let C : Fin 2 → ℝ := ![2000, 4184]
    F13_rho Finset.univ φ ρ * F13_cp Finset.univ φ C = 2473600 ∧
      ∑ i, φ i * ρ i * C i = 2692000 := by
  simp [F13_rho, F13_cp, Fin.sum_univ_two]
  norm_num

/-! ## F14, F15. Pyrolysis and oxidation (notes 5, 6) -/

/-- Eqs. 14 and 15: `ṁ = (M_i / ρ_i) v`, `q̇ = Δ H⁰ v`. -/
noncomputable def F14_m (M ρ v : ℝ) : ℝ := M / ρ * v
noncomputable def F14_q (ΔH v : ℝ) : ℝ := ΔH * v

/-- Transcriber's note 6, confirmed: with A.4's pyrolysis rate in kg m⁻³ s⁻¹, `ṁ_p,s` is in
kg mol⁻¹ s⁻¹, not s⁻¹; with the combustion rate in mol m⁻³ s⁻¹, `ṁ_c,s` is in s⁻¹. -/
theorem note6_units :
    molarMass / density * (density / time) = mass / (mol * time) ∧
      molarMass / density * (density / time) ≠ one / time ∧
      molarMass / density * rate = one / time := by
  decide

/-- The heat terms are W m⁻³ (pyrolysis per kilogram, oxidation per mole). -/
theorem F14_F15_heat_units :
    specificEnergy * (density / time) = volPower ∧ molarEnergy * rate = volPower := by
  decide

/-- Transcriber's note 5, confirmed: with Table 3's signs and positive rates, Eq. 15 makes solid
oxidation a heat sink and Eq. 14 makes pyrolysis a heat source, the reverse of the text; the gas
term of Eq. 9, which carries a minus sign, releases heat. -/
theorem note5_signs {v : ℝ} (hv : 0 < v) :
    F14_q dHcs v < 0 ∧ 0 < F14_q dHps v ∧ 0 < F9 dHcg v := by
  unfold F14_q F9 dHcs dHps dHcg
  refine ⟨by nlinarith, by nlinarith, by nlinarith⟩

/-! ## F16, F17, I4. Radiation (note 10) -/

/-- Eq. 16: `q_rad,s = ε σ (T_s⁴ - T_amb⁴)`. -/
noncomputable def F16 (ε σ Ts Ta : ℝ) : ℝ := ε * σ * (Ts ^ 4 - Ta ^ 4)

/-- Eq. 16's emission has the sign of `T_s - T_amb`. -/
theorem F16_sign {ε σ Ts Ta : ℝ} (hε : 0 < ε) (hσ : 0 < σ) (hs : 0 ≤ Ts) (ha : 0 ≤ Ta) :
    (0 < F16 ε σ Ts Ta ↔ Ta < Ts) ∧ (F16 ε σ Ts Ta = 0 ↔ Ts = Ta) ∧
      (F16 ε σ Ts Ta < 0 ↔ Ts < Ta) := by
  have hc : 0 < ε * σ := mul_pos hε hσ
  have key : ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → (x ^ 4 < y ^ 4 ↔ x < y) := by
    intro x y hx hy
    exact pow_lt_pow_iff_left₀ hx hy (by norm_num)
  unfold F16
  refine ⟨?_, ?_, ?_⟩
  · rw [← key Ta Ts ha hs]
    constructor
    · intro h; by_contra h'; nlinarith [not_lt.1 h']
    · intro h; nlinarith
  · constructor
    · intro h
      have : Ts ^ 4 - Ta ^ 4 = 0 := by
        rcases mul_eq_zero.1 h with h | h
        · exact absurd h hc.ne'
        · exact h
      have h4 : Ts ^ 4 = Ta ^ 4 := by linarith
      exact (pow_left_inj₀ hs ha (by norm_num : (4 : ℕ) ≠ 0)).1 h4
    · intro h; rw [h]; ring
  · rw [← key Ts Ta hs ha]
    constructor
    · intro h; by_contra h'; nlinarith [not_lt.1 h']
    · intro h; nlinarith

theorem F16_units : stefan * (temp * temp * temp * temp) = flux := by decide

/-- Transcriber's note 10, confirmed: inside `exp(-κ_s Y_s ℓ)`, and for Eq. 17 to be a power per
volume, `κ_s` must be in m⁻¹; A.4's dimensionless `κ` fails both. -/
theorem note10_units :
    (one / length) * one * length = one ∧ one * one * length ≠ one ∧
      one * (one / length) * stefan * (temp * temp * temp * temp) = volPower ∧
      one * one * stefan * (temp * temp * temp * temp) ≠ volPower := by
  decide

/-- I4: `P_abs = 1 - exp(-κ_s Y_s ℓ)`. -/
noncomputable def Pabs (κ Y l : ℝ) : ℝ := 1 - Real.exp (-(κ * Y * l))

theorem Pabs_mem {κ Y l : ℝ} (hκY : 0 ≤ κ * Y) (hl : 0 ≤ l) : 0 ≤ Pabs κ Y l ∧ Pabs κ Y l < 1 := by
  unfold Pabs
  have h1 : Real.exp (-(κ * Y * l)) ≤ 1 := by
    rw [Real.exp_le_one_iff]; nlinarith [mul_nonneg hκY hl]
  have h2 := Real.exp_pos (-(κ * Y * l))
  constructor <;> linarith

/-- Absorption probabilities compose along a ray: two segments absorb as their sum,
`1 - e^{-a(l₁ + l₂)} = 1 - (1 - P₁)(1 - P₂)`. -/
theorem Pabs_compose (κ Y l₁ l₂ : ℝ) :
    Pabs κ Y (l₁ + l₂) = 1 - (1 - Pabs κ Y l₁) * (1 - Pabs κ Y l₂) := by
  unfold Pabs
  rw [show -(κ * Y * (l₁ + l₂)) = -(κ * Y * l₁) + -(κ * Y * l₂) by ring, Real.exp_add]
  ring

/-- A longer path absorbs more. -/
theorem Pabs_monotone {κ Y : ℝ} (hκY : 0 ≤ κ * Y) : Monotone (Pabs κ Y) := by
  intro l₁ l₂ h
  unfold Pabs
  have : Real.exp (-(κ * Y * l₂)) ≤ Real.exp (-(κ * Y * l₁)) := by
    rw [Real.exp_le_exp]; nlinarith
  linarith

/-! ## F18, F19. Solid energy and the interface (notes 7, 12) -/

/-- Eq. 18's right-hand side: `∇ · (k_s ∇T_s) + q̇_e,s + q̇_c,s + q̇_p,s`. -/
noncomputable def F18_rhs (divkT qe qc qp : ℝ) : ℝ := divkT + qe + qc + qp

theorem F18_units :
    density * specificHeat * temp / time = volPower ∧ conductivity * temp / area = volPower ∧
      power ≠ volPower := by
  decide

/-- Eq. 19's right-hand side: `h (T_s - T_g) + ε σ (T_s⁴ - T_amb⁴)`. -/
noncomputable def F19 (h Ts Tg ε σ Ta : ℝ) : ℝ := h * (Ts - Tg) + F16 ε σ Ts Ta

/-- Transcriber's note 12, confirmed: `-k_s ∇T_s · n` is a heat flux, so `h` must be in
W m⁻² K⁻¹; a dimensionless `h` (A.4) makes `h (T_s - T_g)` a temperature. -/
theorem note12_units :
    conductivity * (temp / length) = flux ∧ htc * temp = flux ∧ one * temp ≠ flux := by
  decide

/-- With the solid hotter than both the gas and the ambient, the interface carries heat out of
the solid. -/
theorem F19_out {h Ts Tg ε σ Ta : ℝ} (hh : 0 ≤ h) (hε : 0 < ε) (hσ : 0 < σ) (ha : 0 ≤ Ta)
    (hg : Tg ≤ Ts) (hs : Ta < Ts) : 0 < F19 h Ts Tg ε σ Ta := by
  unfold F19
  have := ((F16_sign hε hσ (ha.trans hs.le) ha).1).2 hs
  nlinarith

/-- Algorithm 1, line 2: at the initial state (`T_s = T_g = T_amb`) the radiative and interface
terms vanish. -/
theorem A12_equilibrium (h ε σ Ta Y κ : ℝ) :
    F16 ε σ Ta Ta = 0 ∧ F17 Y κ σ Ta Ta = 0 ∧ F19 h Ta Ta ε σ Ta = 0 := by
  unfold F19 F16 F17; refine ⟨by ring, by ring, by ring⟩

/-! ## Supplement Algorithm 1: Monte Carlo radiation, in expectation -/

/-- The expected energy a ray of energy `E` leaves in its `k`-th absorbing segment, each with
absorption probability `P j`: `E Π_{j<k} (1 - P_j) P_k`. -/
noncomputable def SA1_gas (E : ℝ) (P : ℕ → ℝ) (k : ℕ) : ℝ :=
  E * (∏ j ∈ Finset.range k, (1 - P j)) * P k

/-- The expected energy that crosses `n` segments unabsorbed (to the first solid or out). -/
noncomputable def SA1_pass (E : ℝ) (P : ℕ → ℝ) (n : ℕ) : ℝ := E * ∏ j ∈ Finset.range n, (1 - P j)

/-- Energy is conserved in expectation: what the gas absorbs plus what passes is the ray's. -/
theorem SA1_balance (E : ℝ) (P : ℕ → ℝ) (n : ℕ) :
    ∑ k ∈ Finset.range n, SA1_gas E P k + SA1_pass E P n = E := by
  induction n with
  | zero => simp [SA1_pass]
  | succ n ih =>
    rw [Finset.sum_range_succ]
    unfold SA1_gas SA1_pass at *
    rw [Finset.prod_range_succ]
    linarith [ih]

/-- With I4's probabilities the expected energy passing `n` segments is `E exp(-Σ κ Y_j ℓ_j)`. -/
theorem SA1_pass_exp (E κ : ℝ) (Y l : ℕ → ℝ) (n : ℕ) :
    SA1_pass E (fun j => Pabs κ (Y j) (l j)) n =
      E * Real.exp (-∑ j ∈ Finset.range n, κ * Y j * l j) := by
  unfold SA1_pass Pabs
  simp only [sub_sub_cancel]
  rw [← Finset.sum_neg_distrib, Real.exp_sum]

/-! ## Supplement Algorithm 2: particles to grid (supplement notes 3, 4) -/

/-- Line 5: `q̇_e = m_e (T_e - T_s) C_e`. -/
noncomputable def SA2_line5 (me Te Ts Ce : ℝ) : ℝ := me * (Te - Ts) * Ce

/-- Line 7 as printed: `T_s ← T_s + q̇ ρ_s C_p,s`. -/
noncomputable def SA2_line7 (Ts q ρ C : ℝ) : ℝ := Ts + q * (ρ * C)

/-- Line 7 dividing by the cell's heat capacity `ρ_s C_p,s V`. -/
noncomputable def SA2_line7_fixed (Ts q ρ C V : ℝ) : ℝ := Ts + q / (ρ * C * V)

/-- Supplement note 3, confirmed: line 5 is an energy (J), not a rate, despite its dot.
Supplement note 4, confirmed: line 7 as printed adds J² m⁻³ K⁻¹ to a temperature; dividing by
`ρ_s C_p,s` and the cell volume gives kelvin. -/
theorem notes_S3_S4_units :
    mass * temp * specificHeat = energy ∧ mass * temp * specificHeat ≠ power ∧
      energy * (density * specificHeat) ≠ temp ∧
      energy / (density * specificHeat * volume) = temp := by
  decide

/-- With the division, the solid gains exactly the embers' excess energy. -/
theorem SA2_line7_fixed_energy {Ts q ρ C V : ℝ} (h : ρ * C * V ≠ 0) :
    ρ * C * V * (SA2_line7_fixed Ts q ρ C V - Ts) = q := by
  have hρ : ρ ≠ 0 := left_ne_zero_of_mul (left_ne_zero_of_mul h)
  have hC : C ≠ 0 := right_ne_zero_of_mul (left_ne_zero_of_mul h)
  have hV : V ≠ 0 := right_ne_zero_of_mul h
  unfold SA2_line7_fixed; field_simp; ring

/-- An ember brought to the solid's temperature gives up exactly line 5's energy. -/
theorem SA2_ember_balance (me Te Ts Ce : ℝ) :
    me * Ce * (Te - Ts) = SA2_line5 me Te Ts Ce := by
  unfold SA2_line5; ring

/-- Lines 8 to 10: the flame-front temperature is the running maximum, at least every
particle's. -/
theorem SA2_line10 (Tp : List ℝ) : ∀ x ∈ Tp, x ≤ Tp.foldl max 0 := by
  have key : ∀ (l : List ℝ) (a : ℝ), a ≤ l.foldl max a ∧ ∀ x ∈ l, x ≤ l.foldl max a := by
    intro l
    induction l with
    | nil => intro a; simp
    | cons y l ih =>
      intro a
      simp only [List.foldl_cons, List.mem_cons]
      obtain ⟨h1, h2⟩ := ih (max a y)
      refine ⟨le_trans (le_max_left a y) h1, ?_⟩
      rintro x (rfl | hx)
      · exact le_trans (le_max_right a x) h1
      · exact h2 x hx
  exact (key Tp 0).2

/-- Line 11: using `max(T_f, T)` never slows the gas reaction (Eq. 10 rises with temperature). -/
theorem SA2_line11 {A Ea cf co a b Tf T : ℝ} (hA : 0 < A) (hE : 0 < Ea) (hf : 0 < cf) (ho : 0 < co)
    (hT : 0 < T) : F10 A Ea T cf co a b ≤ F10 A Ea (max Tf T) cf co a b := by
  rcases eq_or_lt_of_le (le_max_right Tf T) with h | h
  · rw [← h]
  · exact ((F10_strictMonoOn hA hE hf ho) hT (hT.trans h) h).le

/-! ## Supplement S1, S-I1, S-I2: wave function collapse rules -/

/-- S-I1's four sides. -/
inductive Side where
  | left | right | up | down
  deriving DecidableEq

/-- The opposite side `d̄`. -/
def Side.opp : Side → Side
  | .left => .right
  | .right => .left
  | .up => .down
  | .down => .up

theorem Side.opp_opp (d : Side) : d.opp.opp = d := by cases d <;> rfl

/-- S1: a rule is a triple `(S, T, E)` of scope, target condition and effect. -/
structure Rule (Scope Target Effect : Type*) where
  S : Scope
  T : Target
  E : Effect

/-- Two rules are equal exactly when their scopes, targets and effects are. -/
theorem S1_ext {α β γ : Type*} (r r' : Rule α β γ) :
    r = r' ↔ r.S = r'.S ∧ r.T = r'.T ∧ r.E = r'.E := by
  cases r; cases r'; simp

/-- S-I2: states `t_i`, `t_j` are compatible in direction `d` when their facing labels match. -/
def compatible {State Label : Type*} (σ : State → Side → Label) (ti tj : State) (d : Side) : Prop :=
  σ ti d = σ tj d.opp

/-- Compatibility is symmetric: `t_i` fits `t_j` in direction `d` exactly when `t_j` fits `t_i`
in direction `d̄`. -/
theorem SI2_symm {State Label : Type*} (σ : State → Side → Label) (ti tj : State) (d : Side) :
    compatible σ ti tj d ↔ compatible σ tj ti d.opp := by
  unfold compatible
  rw [Side.opp_opp]
  exact eq_comm

end FireEgress.Flamescapes
