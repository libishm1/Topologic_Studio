import Mathlib
import FireEgress.Dim
import FireEgress.Tenability
import FireEgress.FireX
import FireEgress.Flamescapes

/-!
# The browser port of Flamescapes: exact steps, bookkeeping and egress quantities

Studio's port of Flamescapes (`topologicpy-web-frontend/src/core/flamescapes.js`,
`voxelize.js`, `flamescapesRun.js` and `fireClasses.js`, read on 2026-11-05 after commit 30c019a)
runs on a grid of 0.25 to 0.5 m with steps of 0.05 to 0.2 s. It therefore integrates every
reaction exactly over a step for its rate frozen at the step's start, and every two-body heat
exchange as the exact relaxation of the pair. This file proves, function by function and over
real arithmetic, the properties the code relies on, and ties each scheme to the paper equation it
replaces (`Flamescapes.lean`, `FireX.lean`). Where commit 30c019a corrected a defect these proofs
found, the version before it is kept under an `Old` name or an `_old` suffix, with the
counterexample that documents the defect.

## Which theorem covers which function

* Constants. `HEAT_PER_O2_eq`, `HEAT_PER_O2_pos`, `HEAT_PER_O2_eq_FireX` (417 kJ mol⁻¹ over
  0.032 kg mol⁻¹), `HEAT_PER_O2_near_huggett` (within 1% of 13.1 MJ kg⁻¹), `HEAT_PER_O2_gap`
  (68 750 J kg⁻¹, 0.525%), `R_GAS_pos`, `P_AMBIENT_pos`.
* `arrhenius`. `arrhenius_pos`, `arrhenius_nonneg`, `arrhenius_strictMonoOn`.
* `survives`. `survives_pos`, `survives_le_one`, `survives_of_nonneg`, `survives_zero`,
  `survives_remaining_mem` (the mass left lies in `[0, m]`), `survives_reacted_mem`,
  `survives_add` (two steps compose), `survives_hasDerivAt` and `survives_unique` (the exact
  solution of `dm/dt = -k m`), `survives_ge_euler` and `survives_sub_euler_le` (the paper's
  Euler update `m (1 - k dt)` is its first-order expansion, with error at most `m (k dt)²/2`),
  `survives_euler_negative` (Euler goes negative once `k dt > 1`, the exponential never),
  `survives_arrhenius_monotoneOn`; the helper `exp_neg_le_quadratic`.
* `exchange`. `mu_pos`, `mu_inv`, `exchange_of_pos`, `exchange_of_not`, `exchange_energy`
  (energy conserved), `exchange_diff`, `exchange_sign` and `exchange_mul_diff_nonneg` (the
  moved heat has the sign of `Ta - Tb`), `exchange_no_overshoot`, `exchange_swap`,
  `exchange_between` (neither temperature overshoots the other), `exchange_hasDerivAt`,
  `exchange_ode` (the closed form of the linear pair), `exchange_first_order` and
  `exchange_hasDerivAt_zero` (Eq. 19's explicit flux `G (Ta - Tb) dt` to first order),
  `exchange_vs_F19` (with `G = h A`), `exchange_explicit_overshoots` (the explicit flux over a
  whole step can reverse the two temperatures).
* `changeGas`. `changeGas_total`, `changeGas_species` (each species changes by its delta),
  `changeGas_mass`, `changeGas_nitrogen` (nitrogen is kept), `changeGas_clamp_creates_mass`.
* `stoichiometry`, `addProducts`, `fireClasses.js`. `sum_Sp`, `closes_accepted`,
  `stoichiometry_h2o_bounds`, `stoichiometry_refuses_old_counterexample`,
  `addProducts_mass_solid` and `addProducts_mass_vapour` (the mass balance, for every fuel),
  `addProducts_mass_solid_noAsh` (`+dm` from a solid), `addProducts_mass_vapour_noAsh` (0 from
  vapour), `addProducts_tracked_mass`, `addProducts_heat_per_O2` (`HEAT_PER_O2` per kg of O₂
  consumed), `stoichiometry_o2_vs_huggett`, `fireClasses_close` and `fireClasses_accepted`
  (classes A, B, C, D, K and L, with `co2FromCarbon`), `classD_heat_per_O2`,
  `addProducts_guard`, `addProducts_gas_mass`. Before 30c019a: `addProducts_clamp_counterexample`,
  `carbon_yield_exceeds` (for `stoichiometryOld`).
* `chemistry`. The liquid item: `chemistry_liquid_energy` (`C T + L · vapour` is conserved),
  `chemistry_liquid_above_boiling`, `chemistry_liquid_capped`, `chemistry_liquid_unclamped`, and
  the pool's burning-rate cap `poolCap_bounds`; before 30c019a,
  `chemistry_liquid_evaporation_energy_old`. Fresh fuel: `chemistry_mixFresh_enthalpy`,
  `chemistry_mixFresh_between`, `chemistry_fresh_refills_skin`. Ash: `chemistry_ash_capacity`,
  `chemistry_ash_capacity_mono`, `chemistry_ash_capacity_mono_cp`,
  `chemistry_ash_capacity_counterexample`.
* `march`. `march_taken_mem`, `march_taken_transmit`, `march_taken_transmit_two`,
  `march_leftFrom_eq`, `march_expected` (the deterministic attenuation is the expected value of
  supplement Algorithm 1's Monte Carlo absorption); `endDeposit_balance`, `marchLoop_balance` and
  `march_balance` (gas + solid + outside = emitted, nothing dropped), `marchLoop_nonneg`,
  `marchLoop_vs_old`, `march_reach_deposits`. Before 30c019a: `marchLoopOld_balance`,
  `marchOld_balance`, `marchLoopOld_nonneg`, `marchLoopOld_dropped_of_gt`,
  `marchOld_reach_drops_all`, `marchLoopOld_all_gas`.
* `radiation`. `radiation_emitter_cap`, `radiation_shares`. `particleToGrid`:
  `particleToGrid_energy` (supplement Algorithm 2's line 7 with the division its note 4 asks
  for).
* `gasDensity` (Eq. 7 with the pressure). `gasDensity_eos`, `gasDensity_strictAntiOn`,
  `gasDensity_ratio`. `buoyancy` (Eq. 6): `buoyancy_sign`, `buoyancy_eq_F6`, `buoyancy_vs_FireX`.
* `diffuse`. `diffuse_between`, `diffuse_heat_pair`, `diffuse_lamD_mem`, `diffuse_species_pair`
  (each species' mass conserved by a pair exchange below the cap), `diffuse_species_capped`.
  Before 30c019a: `diffuse_species_pair_old`, `diffuse_species_counterexample_old`.
* `flamescapesRun.js`. `visibility_formula`, `visibility_mem`, `visibility_antitone` (never
  rises with the soot density), `visibility_strictAntiOn`, `visibility_cap_continuous`,
  `visibility_units`; `coDoseRate_of_nonneg`, `coDoseRate_monotone`, `coDoseRate_integral`,
  `coDoseRate_reaches_one` (the dose reaches 1 at 35 000 ppm·min), `coDoseRate_const`,
  `coDoseRate_examples`.
* `voxelize.js`. `volumeFractions_nonneg`, `volumeFractions_sum_le_one`,
  `volumeFractions_ratio` (the final pass of Eq. 1); `interiorFuel_expected_sum`,
  `interiorFuel_gaussian_mean`, `interiorFuel_cut_raises_mean`, `interiorFuel_cut`,
  `interiorFuel_rescaled_sum` (the contents hold exactly `M`), `interiorFuel_rescaled_nonneg`,
  `interiorFuel_rescaled_ratio`.

## What the proofs found in the port, and what commit 30c019a changed

* `march` dropped what was left after `rayReach` cells whatever its size
  (`marchOld_reach_drops_all`). It now leaves it in the cell the ray ends in, so gas, solid and
  outside account for every joule (`march_balance`, `marchLoop_vs_old`).
* `diffuse` gave each cell its own coefficient, so species mass changed across a temperature
  jump (`diffuse_species_pair_old`). A face now takes the mean of its cells' `ρD`, and a pair
  exchange conserves each species while the cap of 1/7 does not bind on either side
  (`diffuse_species_pair`); where it binds it does not (`diffuse_species_capped`).
* The liquid branch set the surface to its boiling point even when the layer ran out first
  (`chemistry_liquid_evaporation_energy_old`). It now takes the latent heat of what evaporated,
  which conserves `C T + L dv` and never cools the surface below boiling.
* A metal's oxide vanished. It now adds `ash · dOx · 900` J K⁻¹ to the cell. That keeps the
  solid's heat capacity from falling while the bulk refills the surface layer, or whenever
  `fuelHeatCapacity ≤ ash · 900`; not otherwise: magnesium is given wood's 2000 J kg⁻¹ K⁻¹ and
  its oxide holds 1492 J K⁻¹ per kg burnt, so once the remaining metal lies within the layer the
  capacity falls (`chemistry_ash_capacity_counterexample`). The ash's mass is still not a field
  (`addProducts_tracked_mass`).
* Yields heavier than the fuel and its oxygen created mass through the clamp on water
  (`addProducts_clamp_counterexample`). They are now refused, water closes the balance exactly,
  and the mass balance holds for every fuel (`addProducts_mass_solid`); all six classes are
  accepted (`fireClasses_accepted`). The tolerance of 1e-9 admits water down to `-1e-9` kg per
  kg; the six classes close without it.
* Cutting the contents' draws at zero put more than `M` in the building in expectation
  (`interiorFuel_cut_raises_mean`). The masses are now rescaled to exactly `M`.
* Unchanged and by design: class D sets its own `o2`, so it releases about 37.6 MJ per kg of O₂,
  not `HEAT_PER_O2` (`classD_heat_per_O2`).
* The paper's forms that the port corrects: Eq. 7 with `p_amb` (`gasDensity`), Eq. 6 with
  the intended sign (`buoyancy`), supplement Algorithm 2's line 7 divided by the heat capacity
  (`particleToGrid`), Eq. 13's heat capacity summed over the constituents (`createSolver`), and
  Eq. 1's fractions renormalised to at most one (`volumeFractions`).

## What is not covered

* Floating point: the code keeps its fields in `Float32Array`.
* The ray geometry (the Amanatides-Woo traversal, directions, Lambert sampling) is abstracted to
  a path of cells with optical depths `τ = κ Y_soot l ≥ 0` and the kind of the cell at each step.
* Advection, the pressure projection, vorticity confinement, particle motion, the burner,
  boundaries, and the ember spot ignition, which sets a surface to its ignition temperature
  without an energy balance. In the liquid branch `C` is the capacity read before the moisture
  step, as the code reads it.
* That each call site meets the hypotheses is read from the code: `dOx ≤ o2 / st.o2` and
  `dF ≤ yO mass / st.o2` (`addProducts_guard`), `Cf ≤ C` (`chemistry_mixFresh_*`), and soot
  fractions that stay non-negative (`diffuse_between` with `diffuse_lamD_mem`; trilinear
  advection takes convex combinations).
-/

namespace FireEgress.FlamescapesPort

open Real Filter Topology MeasureTheory ProbabilityTheory

/-! ## Constants of `flamescapes.js` -/

/-- `R_GAS` (J mol⁻¹ K⁻¹). -/
noncomputable def R_GAS : ℝ := 8.314462618
/-- `P_AMBIENT` (Pa). -/
noncomputable def P_AMBIENT : ℝ := 101325
/-- `HEAT_PER_O2 = 417e3 / 0.032`, J per kg of O₂: Fire-X's 417 kJ per mol of O₂ (its Eq. 4)
over the molar mass of O₂. -/
noncomputable def HEAT_PER_O2 : ℝ := 417e3 / 0.032

theorem R_GAS_pos : 0 < R_GAS := by norm_num [R_GAS]

theorem P_AMBIENT_pos : 0 < P_AMBIENT := by norm_num [P_AMBIENT]

theorem HEAT_PER_O2_eq : HEAT_PER_O2 = 13031250 := by norm_num [HEAT_PER_O2]

theorem HEAT_PER_O2_pos : 0 < HEAT_PER_O2 := by norm_num [HEAT_PER_O2]

/-- It is Fire-X's Eq. 4 constant per kilogram of O₂ (`FireX.X4_per_kg_O2`). -/
theorem HEAT_PER_O2_eq_FireX : HEAT_PER_O2 = FireX.heatPerMolO2 / FireX.M_O2 := by
  norm_num [HEAT_PER_O2, FireX.heatPerMolO2, FireX.M_O2]

/-- `HEAT_PER_O2` lies within 1% of Huggett's (1980) 13.1 MJ per kg of O₂: the gap is
68 750 J kg⁻¹, 0.525% of 13.1 MJ kg⁻¹. -/
theorem HEAT_PER_O2_near_huggett : |HEAT_PER_O2 - 13.1e6| ≤ 0.01 * 13.1e6 := by
  rw [HEAT_PER_O2_eq, abs_le]; constructor <;> norm_num

/-- The gap exactly: `13.1e6 - HEAT_PER_O2 = 68750`, which is `0.525%` of Huggett's value. -/
theorem HEAT_PER_O2_gap :
    13.1e6 - HEAT_PER_O2 = 68750 ∧ (13.1e6 - HEAT_PER_O2) / 13.1e6 < 0.00525 := by
  rw [HEAT_PER_O2_eq]; constructor <;> norm_num

/-! ## `arrhenius`: the rate of Eqs. 10, 14 and 15 -/

/-- `arrhenius(A, E, T) = T > 0 ? A exp(-E / (R T)) : 0` (s⁻¹). -/
noncomputable def arrhenius (A E T : ℝ) : ℝ := if 0 < T then A * Real.exp (-E / (R_GAS * T)) else 0

theorem arrhenius_pos {A E T : ℝ} (hA : 0 < A) (hT : 0 < T) : 0 < arrhenius A E T := by
  unfold arrhenius; rw [ite_eq_left hT]; exact mul_pos hA (Real.exp_pos _)

theorem arrhenius_nonneg {A E : ℝ} (hA : 0 ≤ A) (T : ℝ) : 0 ≤ arrhenius A E T := by
  unfold arrhenius; split_ifs
  · exact mul_nonneg hA (Real.exp_pos _).le
  · exact le_rfl

/-- With a positive activation energy the rate rises strictly with temperature. -/
theorem arrhenius_strictMonoOn {A E : ℝ} (hA : 0 < A) (hE : 0 < E) :
    StrictMonoOn (arrhenius A E) (Set.Ioi 0) := by
  intro T₁ h₁ T₂ h₂ h12
  simp only [Set.mem_Ioi] at h₁ h₂
  unfold arrhenius; rw [ite_eq_left h₁, ite_eq_left h₂]
  apply mul_lt_mul_of_pos_left _ hA
  rw [Real.exp_lt_exp, neg_div, neg_div, neg_lt_neg_iff]
  apply div_lt_div_of_pos_left hE (mul_pos R_GAS_pos h₁)
  exact mul_lt_mul_of_pos_left h12 R_GAS_pos

/-! ## `survives`: first-order decay, solved exactly over a step -/

/-- `survives(k, dt) = Math.exp(-Math.max(0, k) * dt)`. -/
noncomputable def survives (k dt : ℝ) : ℝ := Real.exp (-(max 0 k) * dt)

theorem survives_pos (k dt : ℝ) : 0 < survives k dt := Real.exp_pos _

theorem survives_le_one (k : ℝ) {dt : ℝ} (hdt : 0 ≤ dt) : survives k dt ≤ 1 := by
  unfold survives
  rw [Real.exp_le_one_iff]
  have : 0 ≤ max 0 k := le_max_left _ _
  nlinarith

theorem survives_of_nonneg {k : ℝ} (hk : 0 ≤ k) (dt : ℝ) :
    survives k dt = Real.exp (-k * dt) := by
  unfold survives; rw [max_eq_right hk]

theorem survives_zero (k : ℝ) : survives k 0 = 1 := by simp [survives]

/-- The mass left after a step, `m · survives k dt`, lies in `[0, m]`, whatever the sign of
`k` (the code clamps `k` at 0). -/
theorem survives_remaining_mem {m : ℝ} (k : ℝ) {dt : ℝ} (hm : 0 ≤ m) (hdt : 0 ≤ dt) :
    0 ≤ m * survives k dt ∧ m * survives k dt ≤ m := by
  refine ⟨mul_nonneg hm (survives_pos k dt).le, ?_⟩
  calc m * survives k dt ≤ m * 1 := mul_le_mul_of_nonneg_left (survives_le_one k hdt) hm
    _ = m := mul_one m

/-- The mass reacted, `m (1 - survives k dt)` (the code's `dOx`, `dPy` and `dF` before their
oxygen limits), lies in `[0, m]`. -/
theorem survives_reacted_mem {m : ℝ} (k : ℝ) {dt : ℝ} (hm : 0 ≤ m) (hdt : 0 ≤ dt) :
    0 ≤ m * (1 - survives k dt) ∧ m * (1 - survives k dt) ≤ m := by
  have h1 := survives_le_one k hdt
  have h0 := survives_pos k dt
  constructor
  · exact mul_nonneg hm (by linarith)
  · nlinarith

/-- Two steps compose: decaying over `t₁` and then `t₂` is decaying over `t₁ + t₂`. -/
theorem survives_add (k t₁ t₂ : ℝ) : survives k (t₁ + t₂) = survives k t₁ * survives k t₂ := by
  unfold survives; rw [← Real.exp_add]; ring_nf

/-- `t ↦ m · survives k t` solves `dm/dt = -max(0, k) m`. -/
theorem survives_hasDerivAt (m k t : ℝ) :
    HasDerivAt (fun s => m * survives k s) (-(max 0 k) * (m * survives k t)) t := by
  unfold survives
  have h : HasDerivAt (fun s => -(max 0 k) * s) (-(max 0 k)) t := by
    simpa using (hasDerivAt_id t).const_mul (-(max 0 k))
  have := (h.exp).const_mul m
  convert this using 1
  ring

/-- It is the only solution: any `f` with `f' = -k f` and `f 0 = m` is `m · survives k`. -/
theorem survives_unique {k m : ℝ} (hk : 0 ≤ k) (f : ℝ → ℝ)
    (hf : ∀ t, HasDerivAt f (-k * f t) t) (h0 : f 0 = m) (t : ℝ) :
    f t = m * survives k t := by
  have hg : ∀ s, HasDerivAt (fun s => f s * Real.exp (k * s)) 0 s := by
    intro s
    have he : HasDerivAt (fun s => Real.exp (k * s)) (Real.exp (k * s) * k) s := by
      have := ((hasDerivAt_id s).const_mul k).exp
      simpa using this
    have := (hf s).mul he
    convert this using 1
    ring
  have hconst : f t * Real.exp (k * t) = f 0 * Real.exp (k * 0) :=
    is_const_of_deriv_eq_zero (fun s => (hg s).differentiableAt) (fun s => (hg s).deriv) t 0
  rw [h0, mul_zero, Real.exp_zero, mul_one] at hconst
  rw [survives_of_nonneg hk]
  have hE : Real.exp (k * t) * Real.exp (-k * t) = 1 := by
    rw [← Real.exp_add]; simp
  calc f t = f t * (Real.exp (k * t) * Real.exp (-k * t)) := by rw [hE, mul_one]
    _ = (f t * Real.exp (k * t)) * Real.exp (-k * t) := by ring
    _ = m * Real.exp (-k * t) := by rw [hconst]

/-- `exp (-x) ≤ 1 - x + x²/2` for `x ≥ 0`. -/
lemma exp_neg_le_quadratic {x : ℝ} (hx : 0 ≤ x) : Real.exp (-x) ≤ 1 - x + x ^ 2 / 2 := by
  have h1 := Real.quadratic_le_exp_of_nonneg hx
  have hp : 0 < 1 + x + x ^ 2 / 2 := by positivity
  have hE : Real.exp (-x) * Real.exp x = 1 := by rw [← Real.exp_add]; simp
  have hn : 0 < Real.exp (-x) := Real.exp_pos _
  have hA : Real.exp (-x) * (1 + x + x ^ 2 / 2) ≤ 1 := by
    calc Real.exp (-x) * (1 + x + x ^ 2 / 2) ≤ Real.exp (-x) * Real.exp x :=
          mul_le_mul_of_nonneg_left h1 hn.le
      _ = 1 := hE
  have hB : 1 ≤ (1 - x + x ^ 2 / 2) * (1 + x + x ^ 2 / 2) := by nlinarith [sq_nonneg (x ^ 2)]
  have := hA.trans hB
  exact le_of_mul_le_mul_right this hp

/-- The paper's explicit Euler update `m (1 - k dt)` never exceeds the exact remaining mass. -/
theorem survives_ge_euler {m k : ℝ} (dt : ℝ) (hm : 0 ≤ m) (hk : 0 ≤ k) :
    m * (1 - k * dt) ≤ m * survives k dt := by
  rw [survives_of_nonneg hk]
  apply mul_le_mul_of_nonneg_left _ hm
  have := Real.add_one_le_exp (-k * dt)
  linarith

/-- Euler's update is the first-order expansion of the exact one: they differ by at most
`m (k dt)² / 2`. -/
theorem survives_sub_euler_le {m k dt : ℝ} (hm : 0 ≤ m) (hk : 0 ≤ k) (hdt : 0 ≤ dt) :
    0 ≤ m * survives k dt - m * (1 - k * dt) ∧
      m * survives k dt - m * (1 - k * dt) ≤ m * (k * dt) ^ 2 / 2 := by
  refine ⟨by linarith [survives_ge_euler dt hm hk], ?_⟩
  rw [survives_of_nonneg hk]
  have hx : 0 ≤ k * dt := mul_nonneg hk hdt
  have h := exp_neg_le_quadratic hx
  have h' : Real.exp (-k * dt) = Real.exp (-(k * dt)) := by ring_nf
  rw [h']
  have := mul_le_mul_of_nonneg_left h hm
  nlinarith

/-- Euler's update goes negative once `k dt > 1`; the exact update never does. -/
theorem survives_euler_negative {m k dt : ℝ} (hm : 0 < m) (h : 1 < k * dt) :
    m * (1 - k * dt) < 0 ∧ 0 < m * survives k dt :=
  ⟨mul_neg_of_pos_of_neg hm (by linarith), mul_pos hm (survives_pos k dt)⟩

/-- At the paper's step the reaction removes more with a hotter solid: the reacted fraction
`1 - survives (arrhenius A E T) dt` rises with `T`. -/
theorem survives_arrhenius_monotoneOn {A E dt : ℝ} (hA : 0 < A) (hE : 0 < E) (hdt : 0 ≤ dt) :
    MonotoneOn (fun T => 1 - survives (arrhenius A E T) dt) (Set.Ioi 0) := by
  intro T₁ h₁ T₂ h₂ h12
  simp only
  have hk1 : 0 ≤ arrhenius A E T₁ := arrhenius_nonneg hA.le _
  have hk2 : 0 ≤ arrhenius A E T₂ := arrhenius_nonneg hA.le _
  have hmono : arrhenius A E T₁ ≤ arrhenius A E T₂ :=
    (arrhenius_strictMonoOn hA hE).monotoneOn h₁ h₂ h12
  rw [survives_of_nonneg hk1, survives_of_nonneg hk2]
  have : Real.exp (-arrhenius A E T₂ * dt) ≤ Real.exp (-arrhenius A E T₁ * dt) := by
    rw [Real.exp_le_exp]; nlinarith
  linarith

/-! ## `exchange`: two-body heat exchange, solved exactly -/

/-- The reduced heat capacity `μ = Ca Cb / (Ca + Cb)` of `exchange`. -/
noncomputable def mu (Ca Cb : ℝ) : ℝ := Ca * Cb / (Ca + Cb)

theorem mu_pos {Ca Cb : ℝ} (ha : 0 < Ca) (hb : 0 < Cb) : 0 < mu Ca Cb := by
  unfold mu; positivity

/-- `1/Ca + 1/Cb = 1/μ`. -/
theorem mu_inv {Ca Cb : ℝ} (ha : 0 < Ca) (hb : 0 < Cb) : 1 / Ca + 1 / Cb = 1 / mu Ca Cb := by
  unfold mu; field_simp; ring

/-- `exchange(Ta, Tb, Ca, Cb, G, dt)`: the heat moved from body a to body b (J). -/
noncomputable def exchange (Ta Tb Ca Cb G dt : ℝ) : ℝ :=
  if 0 < Ca ∧ 0 < Cb ∧ 0 < G then
    (Ta - Tb) * mu Ca Cb * (1 - Real.exp (-G * dt / mu Ca Cb))
  else 0

/-- Body a's temperature after the exchange, `Ta - q / Ca`, as `conduct` applies it. -/
noncomputable def afterA (Ta Tb Ca Cb G dt : ℝ) : ℝ := Ta - exchange Ta Tb Ca Cb G dt / Ca

/-- Body b's temperature after the exchange, `Tb + q / Cb`. -/
noncomputable def afterB (Ta Tb Ca Cb G dt : ℝ) : ℝ := Tb + exchange Ta Tb Ca Cb G dt / Cb

theorem exchange_of_pos {Ta Tb Ca Cb G : ℝ} (ha : 0 < Ca) (hb : 0 < Cb) (hG : 0 < G) (dt : ℝ) :
    exchange Ta Tb Ca Cb G dt = (Ta - Tb) * mu Ca Cb * (1 - Real.exp (-G * dt / mu Ca Cb)) := by
  unfold exchange; rw [ite_eq_left ⟨ha, hb, hG⟩]

theorem exchange_of_not {Ta Tb Ca Cb G : ℝ} (h : ¬ (0 < Ca ∧ 0 < Cb ∧ 0 < G)) (dt : ℝ) :
    exchange Ta Tb Ca Cb G dt = 0 := by
  unfold exchange; rw [ite_eq_right h]

/-- Energy is conserved: `Ca Ta' + Cb Tb' = Ca Ta + Cb Tb`, with no hypothesis (when a capacity
is not positive the code moves nothing). -/
theorem exchange_energy (Ta Tb Ca Cb G dt : ℝ) :
    Ca * afterA Ta Tb Ca Cb G dt + Cb * afterB Ta Tb Ca Cb G dt = Ca * Ta + Cb * Tb := by
  unfold afterA afterB
  by_cases h : 0 < Ca ∧ 0 < Cb ∧ 0 < G
  · obtain ⟨ha, hb, -⟩ := h
    field_simp
    ring
  · rw [exchange_of_not h]; simp

/-- After the exchange the temperature difference has decayed by `exp (-G dt / μ)`. -/
theorem exchange_diff {Ta Tb Ca Cb G : ℝ} (ha : 0 < Ca) (hb : 0 < Cb) (hG : 0 < G) (dt : ℝ) :
    afterA Ta Tb Ca Cb G dt - afterB Ta Tb Ca Cb G dt =
      (Ta - Tb) * Real.exp (-G * dt / mu Ca Cb) := by
  unfold afterA afterB
  rw [exchange_of_pos ha hb hG]
  have hmu := mu_inv ha hb
  have hm : 0 < mu Ca Cb := mu_pos ha hb
  have key : (Ta - Tb) * mu Ca Cb * (1 - Real.exp (-G * dt / mu Ca Cb)) / Ca +
      (Ta - Tb) * mu Ca Cb * (1 - Real.exp (-G * dt / mu Ca Cb)) / Cb =
      (Ta - Tb) * (1 - Real.exp (-G * dt / mu Ca Cb)) := by
    have : (Ta - Tb) * mu Ca Cb * (1 - Real.exp (-G * dt / mu Ca Cb)) / Ca +
        (Ta - Tb) * mu Ca Cb * (1 - Real.exp (-G * dt / mu Ca Cb)) / Cb =
        (Ta - Tb) * mu Ca Cb * (1 - Real.exp (-G * dt / mu Ca Cb)) * (1 / Ca + 1 / Cb) := by
      ring
    rw [this, hmu]
    field_simp
  linarith

/-- The moved heat is `c (Ta - Tb)` with `c ≥ 0` (and `c > 0` once `dt > 0` and the exchange is
active): it has the sign of `Ta - Tb`. -/
theorem exchange_sign (Ta Tb Ca Cb G : ℝ) {dt : ℝ} (hdt : 0 ≤ dt) :
    ∃ c, 0 ≤ c ∧ exchange Ta Tb Ca Cb G dt = c * (Ta - Tb) ∧
      (0 < dt → 0 < Ca ∧ 0 < Cb ∧ 0 < G → 0 < c) := by
  by_cases h : 0 < Ca ∧ 0 < Cb ∧ 0 < G
  · obtain ⟨ha, hb, hG⟩ := h
    have hm := mu_pos ha hb
    refine ⟨mu Ca Cb * (1 - Real.exp (-G * dt / mu Ca Cb)), ?_, ?_, ?_⟩
    · apply mul_nonneg hm.le
      have : Real.exp (-G * dt / mu Ca Cb) ≤ 1 := by
        rw [Real.exp_le_one_iff]
        apply div_nonpos_of_nonpos_of_nonneg _ hm.le
        nlinarith
      linarith
    · rw [exchange_of_pos ha hb hG]; ring
    · intro hdt' _
      apply mul_pos hm
      have : Real.exp (-G * dt / mu Ca Cb) < 1 := by
        rw [Real.exp_lt_one_iff]
        apply div_neg_of_neg_of_pos _ hm
        nlinarith
      linarith
  · exact ⟨0, le_rfl, by rw [exchange_of_not h]; ring, fun _ h' => absurd h' h⟩

/-- `0 ≤ q (Ta - Tb)`: heat never flows from the colder body to the hotter. -/
theorem exchange_mul_diff_nonneg (Ta Tb Ca Cb G : ℝ) {dt : ℝ} (hdt : 0 ≤ dt) :
    0 ≤ exchange Ta Tb Ca Cb G dt * (Ta - Tb) := by
  obtain ⟨c, hc, he, -⟩ := exchange_sign Ta Tb Ca Cb G hdt
  rw [he]
  nlinarith [sq_nonneg (Ta - Tb)]

/-- No overshoot, for `Tb ≤ Ta`: `Tb ≤ Tb' ≤ Ta' ≤ Ta`. -/
theorem exchange_no_overshoot {Ta Tb Ca Cb G dt : ℝ} (hdt : 0 ≤ dt) (hab : Tb ≤ Ta) :
    Tb ≤ afterB Ta Tb Ca Cb G dt ∧ afterB Ta Tb Ca Cb G dt ≤ afterA Ta Tb Ca Cb G dt ∧
      afterA Ta Tb Ca Cb G dt ≤ Ta := by
  by_cases h : 0 < Ca ∧ 0 < Cb ∧ 0 < G
  · obtain ⟨ha, hb, hG⟩ := h
    have hd := exchange_diff (Ta := Ta) (Tb := Tb) ha hb hG dt
    obtain ⟨c, hc, he, -⟩ := exchange_sign Ta Tb Ca Cb G hdt
    have hq : 0 ≤ exchange Ta Tb Ca Cb G dt := by rw [he]; nlinarith
    have hE : 0 ≤ (Ta - Tb) * Real.exp (-G * dt / mu Ca Cb) :=
      mul_nonneg (by linarith) (Real.exp_pos _).le
    refine ⟨?_, ?_, ?_⟩
    · unfold afterB; have := div_nonneg hq hb.le; linarith
    · linarith
    · unfold afterA; have := div_nonneg hq ha.le; linarith
  · unfold afterA afterB; rw [exchange_of_not h]; simp [hab]

/-- `exchange` is antisymmetric under swapping the bodies. -/
theorem exchange_swap (Ta Tb Ca Cb G dt : ℝ) :
    exchange Tb Ta Cb Ca G dt = -exchange Ta Tb Ca Cb G dt := by
  unfold exchange
  have hmu : mu Cb Ca = mu Ca Cb := by unfold mu; ring
  by_cases h : 0 < Ca ∧ 0 < Cb ∧ 0 < G
  · obtain ⟨ha, hb, hG⟩ := h
    rw [ite_eq_left ⟨hb, ha, hG⟩, ite_eq_left ⟨ha, hb, hG⟩, hmu]; ring
  · have h' : ¬ (0 < Cb ∧ 0 < Ca ∧ 0 < G) := fun ⟨hb, ha, hG⟩ => h ⟨ha, hb, hG⟩
    rw [ite_eq_right h, ite_eq_right h']; ring

/-- Neither temperature leaves the interval between the two, whichever is hotter. -/
theorem exchange_between {Ta Tb Ca Cb G dt : ℝ} (hdt : 0 ≤ dt) :
    min Ta Tb ≤ afterA Ta Tb Ca Cb G dt ∧ afterA Ta Tb Ca Cb G dt ≤ max Ta Tb ∧
      min Ta Tb ≤ afterB Ta Tb Ca Cb G dt ∧ afterB Ta Tb Ca Cb G dt ≤ max Ta Tb := by
  rcases le_total Tb Ta with hab | hba
  · obtain ⟨h1, h2, h3⟩ := exchange_no_overshoot (Ca := Ca) (Cb := Cb) (G := G) hdt hab
    rw [min_eq_right hab, max_eq_left hab]
    exact ⟨by linarith, h3, h1, by linarith⟩
  · obtain ⟨h1, h2, h3⟩ := exchange_no_overshoot (Ca := Cb) (Cb := Ca) (G := G) hdt hba
    have eA : afterA Ta Tb Ca Cb G dt = afterB Tb Ta Cb Ca G dt := by
      unfold afterA afterB; rw [exchange_swap]; ring
    have eB : afterB Ta Tb Ca Cb G dt = afterA Tb Ta Cb Ca G dt := by
      unfold afterA afterB; rw [exchange_swap]; ring
    rw [eA, eB, min_eq_left hba, max_eq_right hba]
    exact ⟨h1, by linarith, by linarith, h3⟩

/-- The heat moved is differentiable in the step, with derivative `G (Ta - Tb) exp(-G t/μ)`. -/
theorem exchange_hasDerivAt {Ta Tb Ca Cb G : ℝ} (ha : 0 < Ca) (hb : 0 < Cb) (hG : 0 < G)
    (t : ℝ) :
    HasDerivAt (fun s => exchange Ta Tb Ca Cb G s)
      (G * (Ta - Tb) * Real.exp (-G * t / mu Ca Cb)) t := by
  have hm := mu_pos ha hb
  have hfun : (fun s => exchange Ta Tb Ca Cb G s) =
      fun s => (Ta - Tb) * mu Ca Cb * (1 - Real.exp (-G * s / mu Ca Cb)) := by
    funext s; exact exchange_of_pos ha hb hG s
  rw [hfun]
  have hl : HasDerivAt (fun s => -G * s / mu Ca Cb) (-G / mu Ca Cb) t := by
    have := ((hasDerivAt_id t).const_mul (-G)).div_const (mu Ca Cb)
    simpa using this
  have := ((hl.exp).const_sub 1).const_mul ((Ta - Tb) * mu Ca Cb)
  convert this using 1
  field_simp

/-- It is the closed-form solution of the linear pair `Ca dTa/dt = G (Tb - Ta)`,
`Cb dTb/dt = G (Ta - Tb)`, from `(Ta, Tb)` at `t = 0`. -/
theorem exchange_ode {Ta Tb Ca Cb G : ℝ} (ha : 0 < Ca) (hb : 0 < Cb) (hG : 0 < G) (t : ℝ) :
    HasDerivAt (fun s => afterA Ta Tb Ca Cb G s)
        (G * (afterB Ta Tb Ca Cb G t - afterA Ta Tb Ca Cb G t) / Ca) t ∧
      HasDerivAt (fun s => afterB Ta Tb Ca Cb G s)
        (G * (afterA Ta Tb Ca Cb G t - afterB Ta Tb Ca Cb G t) / Cb) t ∧
      afterA Ta Tb Ca Cb G 0 = Ta ∧ afterB Ta Tb Ca Cb G 0 = Tb := by
  have hq := exchange_hasDerivAt (Ta := Ta) (Tb := Tb) ha hb hG t
  have hd := exchange_diff (Ta := Ta) (Tb := Tb) ha hb hG t
  have h0 : exchange Ta Tb Ca Cb G 0 = 0 := by
    rw [exchange_of_pos ha hb hG]; simp
  refine ⟨?_, ?_, ?_, ?_⟩
  · have := (hq.div_const Ca).const_sub Ta
    show HasDerivAt (fun s => Ta - exchange Ta Tb Ca Cb G s / Ca) _ t
    convert this using 1
    rw [show afterB Ta Tb Ca Cb G t - afterA Ta Tb Ca Cb G t =
      -((Ta - Tb) * Real.exp (-G * t / mu Ca Cb)) by linarith]
    ring
  · have := (hq.div_const Cb).const_add Tb
    show HasDerivAt (fun s => Tb + exchange Ta Tb Ca Cb G s / Cb) _ t
    convert this using 1
    rw [hd]
    ring
  · unfold afterA; rw [h0]; simp
  · unfold afterB; rw [h0]; simp

/-- First-order agreement with Eq. 19's explicit flux `G (Ta - Tb) dt`: the difference is at
most `|Ta - Tb| G² dt² / (2 μ)`. -/
theorem exchange_first_order {Ta Tb Ca Cb G dt : ℝ} (ha : 0 < Ca) (hb : 0 < Cb) (hG : 0 < G)
    (hdt : 0 ≤ dt) :
    |exchange Ta Tb Ca Cb G dt - G * (Ta - Tb) * dt| ≤
      |Ta - Tb| * (G ^ 2 * dt ^ 2 / (2 * mu Ca Cb)) := by
  have hm := mu_pos ha hb
  rw [exchange_of_pos ha hb hG]
  set x := G * dt / mu Ca Cb with hx
  have hx0 : 0 ≤ x := div_nonneg (mul_nonneg hG.le hdt) hm.le
  have hexp : Real.exp (-G * dt / mu Ca Cb) = Real.exp (-x) := by rw [hx]; ring_nf
  rw [hexp]
  have hGdt : G * dt = mu Ca Cb * x := by rw [hx]; field_simp
  have e1 := Real.one_sub_le_exp_neg x
  have e2 := exp_neg_le_quadratic hx0
  have hrewrite : (Ta - Tb) * mu Ca Cb * (1 - Real.exp (-x)) - G * (Ta - Tb) * dt =
      (Ta - Tb) * (mu Ca Cb * (1 - x - Real.exp (-x))) := by
    rw [show G * (Ta - Tb) * dt = (Ta - Tb) * (G * dt) by ring, hGdt]; ring
  rw [hrewrite, abs_mul]
  apply mul_le_mul_of_nonneg_left _ (abs_nonneg _)
  have hbound : |mu Ca Cb * (1 - x - Real.exp (-x))| ≤ mu Ca Cb * (x ^ 2 / 2) := by
    rw [abs_le]; constructor
    · nlinarith
    · nlinarith
  have hrhs : G ^ 2 * dt ^ 2 / (2 * mu Ca Cb) = mu Ca Cb * (x ^ 2 / 2) := by
    rw [hx]; field_simp
  rw [hrhs]; exact hbound

/-- As `dt → 0` the moved heat is `G (Ta - Tb) dt + o(dt)`: its derivative at `dt = 0` is
Eq. 19's flux. -/
theorem exchange_hasDerivAt_zero {Ta Tb Ca Cb G : ℝ} (ha : 0 < Ca) (hb : 0 < Cb) (hG : 0 < G) :
    HasDerivAt (fun s => exchange Ta Tb Ca Cb G s) (G * (Ta - Tb)) 0 := by
  have := exchange_hasDerivAt (Ta := Ta) (Tb := Tb) ha hb hG 0
  simpa using this

/-- With the interface conductance `G = h A`, the exact exchange starts at Eq. 19's convective
flux through the area `A`: `h (T_s - T_g) A` (the radiative part of Eq. 19 is carried by rays). -/
theorem exchange_vs_F19 {Ts Tg Cs Cg h A ε σ Ta : ℝ} (hs : 0 < Cs) (hg : 0 < Cg) (hh : 0 < h)
    (hA : 0 < A) :
    HasDerivAt (fun s => exchange Ts Tg Cs Cg (h * A) s)
      (A * (Flamescapes.F19 h Ts Tg ε σ Ta - Flamescapes.F16 ε σ Ts Ta)) 0 := by
  have := exchange_hasDerivAt_zero (Ta := Ts) (Tb := Tg) hs hg (mul_pos hh hA)
  convert this using 1
  unfold Flamescapes.F19
  ring

/-- Eq. 19's explicit flux, taken over a whole step, overshoots: with `Ca = Cb = G = 1`,
`dt = 3`, `Ta = 1`, `Tb = 0`, body a ends at `-2` and body b at `3`. The exact exchange keeps
both in `[0, 1]` (`exchange_between`). -/
theorem exchange_explicit_overshoots :
    let q : ℝ := 1 * (1 - 0) * 3
    (1 : ℝ) - q / 1 = -2 ∧ (0 : ℝ) + q / 1 = 3 := by
  norm_num

/-! ## `changeGas`: species masses change by the deltas, nitrogen is kept -/

section ChangeGas

variable {ι : Type*} [Fintype ι]

/-- `changeGas(sim, c, delta)` on a cell of volume `V`, density `rho` and tracked mass
fractions `Y` (nitrogen the rest): the new mass fractions and density. -/
noncomputable def changeGas (V rho : ℝ) (Y δ : ι → ℝ) : (ι → ℝ) × ℝ :=
  if 0 < ∑ q, max 0 (Y q * (rho * V) + δ q) + max 0 (rho * V - ∑ q, Y q * (rho * V)) then
    (fun q => max 0 (Y q * (rho * V) + δ q) /
        (∑ q, max 0 (Y q * (rho * V) + δ q) + max 0 (rho * V - ∑ q, Y q * (rho * V))),
      (∑ q, max 0 (Y q * (rho * V) + δ q) + max 0 (rho * V - ∑ q, Y q * (rho * V))) / V)
  else (Y, rho)

/-- Under the guards (no species driven negative, tracked fractions summing to at most one,
a positive result), the normaliser `total` is the old mass plus the deltas. -/
lemma changeGas_total {V rho : ℝ} {Y δ : ι → ℝ} (hδ : ∀ q, 0 ≤ Y q * (rho * V) + δ q)
    (hN : 0 ≤ rho * V - ∑ q, Y q * (rho * V)) :
    ∑ q, max 0 (Y q * (rho * V) + δ q) + max 0 (rho * V - ∑ q, Y q * (rho * V)) =
      rho * V + ∑ q, δ q := by
  rw [max_eq_right hN]
  have : ∀ q, max 0 (Y q * (rho * V) + δ q) = Y q * (rho * V) + δ q :=
    fun q => max_eq_right (hδ q)
  simp_rw [this, Finset.sum_add_distrib]
  ring

/-- Each tracked species' mass changes by exactly its delta. -/
theorem changeGas_species {V rho : ℝ} {Y δ : ι → ℝ} (hV : 0 < V)
    (hδ : ∀ q, 0 ≤ Y q * (rho * V) + δ q) (hN : 0 ≤ rho * V - ∑ q, Y q * (rho * V))
    (hpos : 0 < rho * V + ∑ q, δ q) (q : ι) :
    (changeGas V rho Y δ).1 q * ((changeGas V rho Y δ).2 * V) = Y q * (rho * V) + δ q := by
  have ht := changeGas_total hδ hN
  unfold changeGas
  rw [ht, ite_eq_left hpos]
  simp only
  rw [max_eq_right (hδ q)]
  field_simp

/-- The new mass is the old mass plus the deltas. -/
theorem changeGas_mass {V rho : ℝ} {Y δ : ι → ℝ} (hV : 0 < V)
    (hδ : ∀ q, 0 ≤ Y q * (rho * V) + δ q) (hN : 0 ≤ rho * V - ∑ q, Y q * (rho * V))
    (hpos : 0 < rho * V + ∑ q, δ q) :
    (changeGas V rho Y δ).2 * V = rho * V + ∑ q, δ q := by
  have ht := changeGas_total hδ hN
  unfold changeGas
  rw [ht, ite_eq_left hpos]
  field_simp

/-- Nitrogen, the untracked rest, is kept. -/
theorem changeGas_nitrogen {V rho : ℝ} {Y δ : ι → ℝ} (hV : 0 < V)
    (hδ : ∀ q, 0 ≤ Y q * (rho * V) + δ q) (hN : 0 ≤ rho * V - ∑ q, Y q * (rho * V))
    (hpos : 0 < rho * V + ∑ q, δ q) :
    (changeGas V rho Y δ).2 * V - ∑ q, (changeGas V rho Y δ).1 q * ((changeGas V rho Y δ).2 * V) =
      rho * V - ∑ q, Y q * (rho * V) := by
  simp_rw [changeGas_species hV hδ hN hpos]
  rw [changeGas_mass hV hδ hN hpos, Finset.sum_add_distrib]
  ring

/-- Without the guard the clamp creates mass: taking 2 kg of a species a cell holds 1 kg of
gives the cell 1 kg more than the deltas say. -/
theorem changeGas_clamp_creates_mass :
    let total : ℝ := max 0 (1 + (-2)) + max 0 (2 - 1)
    total ≠ 2 + (-2) := by
  norm_num

end ChangeGas

/-! ## `stoichiometry` and `addProducts`: mass and heat per oxygen -/

/-- The gas species of `flamescapes.js` (`SPECIES`); nitrogen is the rest. -/
inductive Sp where
  | fuel | o2 | co2 | h2o | soot | co
  deriving DecidableEq

instance : Fintype Sp :=
  ⟨{.fuel, .o2, .co2, .h2o, .soot, .co}, fun x => by cases x <;> decide⟩

/-- A fire class as `stoichiometry` reads it (`fireClasses.js`): heat of combustion
(J kg⁻¹), the yields of CO₂, CO and soot, an optional oxygen demand `o2` (kg per kg, for
metals) and the ash yield (`ash ?? 0`, kg of solid oxide per kg). -/
structure FireClass where
  heat : ℝ
  co2 : ℝ
  co : ℝ
  soot : ℝ
  o2 : Option ℝ
  ash : ℝ

/-- `fuel.o2 ?? fuel.heatOfCombustion / HEAT_PER_O2`. -/
noncomputable def FireClass.o2kg (f : FireClass) : ℝ := f.o2.getD (f.heat / HEAT_PER_O2)

/-- The products of burning 1 kg of fuel. -/
structure Stoich where
  o2 : ℝ
  co2 : ℝ
  co : ℝ
  soot : ℝ
  h2o : ℝ
  ash : ℝ
  heat : ℝ

/-- The `delta` that `addProducts(sim, c, st, dm, vapour)` passes to `changeGas` (the ash is
not a gas species). -/
noncomputable def productsDelta (st : Stoich) (dm : ℝ) (vapour : Bool) : Sp → ℝ
  | .fuel => if vapour then -dm else 0
  | .o2 => -st.o2 * dm
  | .co2 => st.co2 * dm
  | .h2o => st.h2o * dm
  | .soot => st.soot * dm
  | .co => st.co * dm

lemma sum_Sp (f : Sp → ℝ) :
    ∑ q, f q = f .fuel + f .o2 + f .co2 + f .h2o + f .soot + f .co := by
  show ∑ q ∈ ({.fuel, .o2, .co2, .h2o, .soot, .co} : Finset Sp), f q = _
  rw [Finset.sum_insert (by decide), Finset.sum_insert (by decide), Finset.sum_insert (by decide),
    Finset.sum_insert (by decide), Finset.sum_insert (by decide), Finset.sum_singleton]
  ring

/-- The yields leave room for water: `co2 + co + soot + ash ≤ 1 + o2`. -/
def FireClass.closes (f : FireClass) : Prop := f.co2 + f.co + f.soot + f.ash ≤ 1 + f.o2kg

/-! ### The version before commit 30c019a, which clamped water at zero -/

/-- `stoichiometry` before 30c019a: water closed the balance, clamped at 0. -/
noncomputable def stoichiometryOld (f : FireClass) : Stoich :=
  ⟨f.o2kg, f.co2, f.co, f.soot, max 0 (1 + f.o2kg - f.co2 - f.co - f.soot - f.ash), f.ash, f.heat⟩

/-- The defect it had: a fuel with `Δh = HEAT_PER_O2` (1 kg of O₂ per kg) and a CO₂ yield of 3
added 2 kg of gas per kg burnt from a solid, not 1. -/
theorem addProducts_clamp_counterexample :
    ∑ q, productsDelta (stoichiometryOld ⟨HEAT_PER_O2, 3, 0, 0, none, 0⟩) 1 false q = 2 := by
  rw [sum_Sp]
  simp only [productsDelta, stoichiometryOld, FireClass.o2kg, Option.getD_none,
    Bool.false_eq_true, ↓reduceIte]
  rw [div_self HEAT_PER_O2_pos.ne']
  norm_num

/-- Pure carbon burnt to CO₂ would have crossed that edge: its CO₂ yield 44/12 exceeds
`1 + 32.8e6 / HEAT_PER_O2`. -/
theorem carbon_yield_exceeds : 1 + 32.8e6 / HEAT_PER_O2 < (44 : ℝ) / 12 := by
  rw [HEAT_PER_O2_eq]; norm_num

/-! ### The current version: such yields are refused -/

/-- `stoichiometry` accepts a fuel unless `co2 + co + soot + ash > 1 + o2 + 1e-9`. -/
def FireClass.accepted (f : FireClass) : Prop := f.co2 + f.co + f.soot + f.ash ≤ 1 + f.o2kg + 1e-9

/-- `stoichiometry(fuel)`: water closes the balance exactly,
`h2o = 1 + o2 - co2 - co - soot - ash`. -/
noncomputable def stoichiometry (f : FireClass) : Stoich :=
  ⟨f.o2kg, f.co2, f.co, f.soot, 1 + f.o2kg - f.co2 - f.co - f.soot - f.ash, f.ash, f.heat⟩

theorem closes_accepted {f : FireClass} (h : f.closes) : f.accepted := by
  unfold FireClass.closes at h; unfold FireClass.accepted; linarith

/-- An accepted fuel's water is at least `-1e-9` kg per kg, and non-negative when it closes
without the tolerance. -/
theorem stoichiometry_h2o_bounds {f : FireClass} :
    (f.accepted → -1e-9 ≤ (stoichiometry f).h2o) ∧ (f.closes → 0 ≤ (stoichiometry f).h2o) := by
  unfold FireClass.accepted FireClass.closes
  simp only [stoichiometry]
  constructor <;> intro h <;> linarith

/-- The fuel that showed the old defect is now refused. -/
theorem stoichiometry_refuses_old_counterexample :
    ¬ (FireClass.accepted ⟨HEAT_PER_O2, 3, 0, 0, none, 0⟩) := by
  unfold FireClass.accepted FireClass.o2kg
  simp only [Option.getD_none]
  rw [div_self HEAT_PER_O2_pos.ne']
  norm_num

/-- Burning `dm` of solid fuel: the gas gains `(1 - ash) dm` and the ash takes the rest. With
water closing the balance exactly this holds for every fuel, accepted or not. -/
theorem addProducts_mass_solid (f : FireClass) (dm : ℝ) :
    ∑ q, productsDelta (stoichiometry f) dm false q + f.ash * dm = dm := by
  rw [sum_Sp]
  simp only [productsDelta, stoichiometry, Bool.false_eq_true, ↓reduceIte]
  ring

/-- Burning `dm` of fuel vapour: the gas loses only the ash (none for the classes that burn as
vapour). -/
theorem addProducts_mass_vapour (f : FireClass) (dm : ℝ) :
    ∑ q, productsDelta (stoichiometry f) dm true q + f.ash * dm = 0 := by
  rw [sum_Sp]
  simp only [productsDelta, stoichiometry, ↓reduceIte]
  ring

/-- Without ash (classes A, B, C, K, L): burning `dm` from a solid adds exactly `dm` to the
gas. -/
theorem addProducts_mass_solid_noAsh {f : FireClass} (h0 : f.ash = 0) (dm : ℝ) :
    ∑ q, productsDelta (stoichiometry f) dm false q = dm := by
  have := addProducts_mass_solid f dm
  rw [h0, zero_mul, add_zero] at this
  exact this

/-- Without ash, burning `dm` of vapour leaves the gas's mass unchanged. -/
theorem addProducts_mass_vapour_noAsh {f : FireClass} (h0 : f.ash = 0) (dm : ℝ) :
    ∑ q, productsDelta (stoichiometry f) dm true q = 0 := by
  have := addProducts_mass_vapour f dm
  rw [h0, zero_mul, add_zero] at this
  exact this

/-- The ash's mass is not a field of the solver: `chemistry` removes `dm` from the solid's fuel
and the gas gains `(1 - ash) dm`, so the tracked mass (fuel plus gas) falls by `ash · dm`. Since
30c019a the ash's heat capacity is kept (`chemistry_ash_capacity`). -/
theorem addProducts_tracked_mass (f : FireClass) (dm : ℝ) :
    -dm + ∑ q, productsDelta (stoichiometry f) dm false q = -(f.ash * dm) := by
  have := addProducts_mass_solid f dm
  linarith

/-- For a class without its own `o2`, the heat released, `dm Δh`, is `HEAT_PER_O2` per kg of
oxygen consumed. -/
theorem addProducts_heat_per_O2 {f : FireClass} (ho : f.o2 = none) (dm : ℝ) :
    dm * (stoichiometry f).heat = HEAT_PER_O2 * ((stoichiometry f).o2 * dm) := by
  simp only [stoichiometry, FireClass.o2kg, ho, Option.getD_none]
  field_simp [HEAT_PER_O2_pos.ne']

/-- With Huggett's 13.1 MJ kg⁻¹ in place of `HEAT_PER_O2` the oxygen consumed per joule would
differ by the same 0.525%: the ratio of the two oxygen masses is `13.1e6 / HEAT_PER_O2`. -/
theorem stoichiometry_o2_vs_huggett {f : FireClass} (ho : f.o2 = none) :
    (stoichiometry f).o2 = (f.heat / 13.1e6) * (13.1e6 / HEAT_PER_O2) := by
  simp only [stoichiometry, FireClass.o2kg, ho, Option.getD_none]
  field_simp [HEAT_PER_O2_pos.ne']

/-- `co2FromCarbon(carbon, co, soot) = 44/12 max(0, carbon - 12/28 co - soot)`. -/
noncomputable def co2FromCarbon (carbon co soot : ℝ) : ℝ := 44 / 12 * max 0 (carbon - 12 / 28 * co - soot)

/-- `fuel(heatMJ, carbon, co, soot)` of `fireClasses.js`. -/
noncomputable def carbonFuel (heatMJ carbon co soot : ℝ) : FireClass :=
  ⟨heatMJ * 1e6, co2FromCarbon carbon co soot, co, soot, none, 0⟩

/-- The fire classes of `fireClasses.js` as of 2026-11-05: A, B, C, K and L from their carbon
fractions, D (magnesium) with its own oxygen and ash. -/
noncomputable def classA : FireClass := carbonFuel 18.0 0.444 0.01 0.015
noncomputable def classB : FireClass := carbonFuel 44.1 0.839 0.022 0.072
noncomputable def classC : FireClass := carbonFuel 30.3 0.857 0.056 0.056
noncomputable def classD : FireClass := ⟨24.75e6, 0, 0, 0, some 0.658, 1.658⟩
noncomputable def classK : FireClass := carbonFuel 37.4 0.774 0.021 0.069
noncomputable def classL : FireClass := carbonFuel 30.3 0.857 0.056 0.056

/-- Every current class closes without the tolerance (class D exactly, `1 + 0.658 = 1.658`), so
`stoichiometry` accepts all six and their water is non-negative. -/
theorem fireClasses_close :
    classA.closes ∧ classB.closes ∧ classC.closes ∧ classD.closes ∧ classK.closes ∧
      classL.closes := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    norm_num [FireClass.closes, FireClass.o2kg, classA, classB, classC, classD, classK, classL,
      carbonFuel, co2FromCarbon, HEAT_PER_O2, max_def]

theorem fireClasses_accepted :
    classA.accepted ∧ classB.accepted ∧ classC.accepted ∧ classD.accepted ∧ classK.accepted ∧
      classL.accepted := by
  obtain ⟨a, b, c, d, k, l⟩ := fireClasses_close
  exact ⟨closes_accepted a, closes_accepted b, closes_accepted c, closes_accepted d,
    closes_accepted k, closes_accepted l⟩

/-- The heat-per-oxygen statement does not apply to class D, which sets its own `o2`: magnesium
releases about 37.6 MJ per kg of O₂, not `HEAT_PER_O2`. -/
theorem classD_heat_per_O2 :
    37.6e6 < (stoichiometry classD).heat / (stoichiometry classD).o2 ∧
      (stoichiometry classD).heat / (stoichiometry classD).o2 < 37.7e6 := by
  norm_num [stoichiometry, FireClass.o2kg, classD]

/-- The guards in `chemistry` keep `changeGas` from clamping: with `dm ≤ m_O₂ / o2` (and, for
vapour, `dm ≤ m_fuel`), non-negative yields, a class that closes and `dm ≥ 0`, no species is
driven negative. -/
theorem addProducts_guard {f : FireClass} {dm : ℝ} {m : Sp → ℝ} {vapour : Bool}
    (hm : ∀ q, 0 ≤ m q) (hdm : 0 ≤ dm) (ho2 : 0 < f.o2kg) (hclose : f.closes)
    (hco2 : 0 ≤ f.co2) (hco : 0 ≤ f.co) (hsoot : 0 ≤ f.soot)
    (hO : dm ≤ m .o2 / (stoichiometry f).o2) (hF : vapour = true → dm ≤ m .fuel) :
    ∀ q, 0 ≤ m q + productsDelta (stoichiometry f) dm vapour q := by
  have ho2' : 0 < (stoichiometry f).o2 := ho2
  have hw := stoichiometry_h2o_bounds.2 hclose
  intro q
  cases q with
  | fuel =>
    simp only [productsDelta]
    split_ifs with hv
    · linarith [hF hv]
    · linarith [hm .fuel]
  | o2 =>
    simp only [productsDelta]
    have := (le_div_iff₀ ho2').1 hO
    linarith
  | co2 => simp only [productsDelta, stoichiometry]; nlinarith [hm .co2]
  | h2o => simp only [productsDelta]; nlinarith [hm .h2o]
  | soot => simp only [productsDelta, stoichiometry]; nlinarith [hm .soot]
  | co => simp only [productsDelta, stoichiometry]; nlinarith [hm .co]

/-- Together: under the guards, `addProducts` changes each species by its delta, keeps the
nitrogen (`changeGas_species`, `changeGas_nitrogen`), and changes the gas's mass by
`(1 - ash) dm` from a solid and by `-ash · dm` from vapour. -/
theorem addProducts_gas_mass {V rho dm : ℝ} {Y : Sp → ℝ} {f : FireClass} {vapour : Bool}
    (hV : 0 < V) (hrho : 0 < rho) (hY : ∀ q, 0 ≤ Y q) (hN : ∑ q, Y q ≤ 1)
    (hdm : 0 ≤ dm) (ho2 : 0 < f.o2kg) (hclose : f.closes) (hco2 : 0 ≤ f.co2) (hco : 0 ≤ f.co)
    (hsoot : 0 ≤ f.soot)
    (hO : dm ≤ Y .o2 * (rho * V) / (stoichiometry f).o2)
    (hF : vapour = true → dm ≤ Y .fuel * (rho * V))
    (hpos : 0 < rho * V + ((if vapour then 0 else dm) - f.ash * dm)) :
    (changeGas V rho Y (productsDelta (stoichiometry f) dm vapour)).2 * V =
      rho * V + ((if vapour then 0 else dm) - f.ash * dm) := by
  have hmass : 0 < rho * V := mul_pos hrho hV
  have hg := addProducts_guard (m := fun q => Y q * (rho * V)) (vapour := vapour)
    (fun q => mul_nonneg (hY q) hmass.le) hdm ho2 hclose hco2 hco hsoot hO hF
  have hNn : 0 ≤ rho * V - ∑ q, Y q * (rho * V) := by
    rw [← Finset.sum_mul]; nlinarith
  have hsum : ∑ q, productsDelta (stoichiometry f) dm vapour q =
      (if vapour then 0 else dm) - f.ash * dm := by
    cases vapour
    · have := addProducts_mass_solid f dm
      simp only [Bool.false_eq_true, ↓reduceIte]
      linarith
    · have := addProducts_mass_vapour f dm
      simp only [↓reduceIte]
      linarith
  have hp : 0 < rho * V + ∑ q, productsDelta (stoichiometry f) dm vapour q := by
    rw [hsum]; exact hpos
  rw [changeGas_mass hV hg hNn hp, hsum]

/-! ### The liquid first item (classes B and K) -/

/-- Before 30c019a: `dv = min(skin, C (T - T_b) / L)` and then `T ← T_b`. Unclamped, the heat
taken was the latent heat carried off; when the skin ran out first, `C (T - T_b) - dv L > 0` was
lost. -/
theorem chemistry_liquid_evaporation_energy_old {C T Tb L skin : ℝ} (hC : 0 < C) (hL : 0 < L) :
    (C * (T - Tb) / L ≤ skin → C * (T - Tb) = min skin (C * (T - Tb) / L) * L) ∧
      (skin < C * (T - Tb) / L → 0 < C * (T - Tb) - min skin (C * (T - Tb) / L) * L) := by
  constructor
  · intro h; rw [min_eq_right h]; field_simp
  · intro h
    rw [min_eq_left h.le]
    have := (lt_div_iff₀ hL).1 h
    linarith

/-- The most a pool of exposed area `A` gives off in `dt`: `m''∞ (1 - exp(-kβ D)) A dt`, with
`D = √(4 A / π)` the diameter of a round pool of that area (Babrauskas 1983; `poolCap` in
flamescapes.js). -/
noncomputable def poolCap (mInf kBeta A dt : ℝ) : ℝ :=
  mInf * (1 - Real.exp (-(kBeta * Real.sqrt (4 * A / Real.pi)))) * A * dt

/-- The cap is never negative, and never more than the infinite pool's `m''∞ A dt`. -/
theorem poolCap_bounds {mInf kBeta A dt : ℝ} (hm : 0 ≤ mInf) (hk : 0 ≤ kBeta) (hA : 0 ≤ A)
    (hdt : 0 ≤ dt) : 0 ≤ poolCap mInf kBeta A dt ∧ poolCap mInf kBeta A dt ≤ mInf * A * dt := by
  have hx : 0 ≤ kBeta * Real.sqrt (4 * A / Real.pi) := mul_nonneg hk (Real.sqrt_nonneg _)
  have he1 : Real.exp (-(kBeta * Real.sqrt (4 * A / Real.pi))) ≤ 1 :=
    Real.exp_le_one_iff.2 (by linarith)
  have he0 : 0 < Real.exp (-(kBeta * Real.sqrt (4 * A / Real.pi))) := Real.exp_pos _
  have hf0 : 0 ≤ 1 - Real.exp (-(kBeta * Real.sqrt (4 * A / Real.pi))) := by linarith
  have hf1 : 1 - Real.exp (-(kBeta * Real.sqrt (4 * A / Real.pi))) ≤ 1 := by linarith
  have hmAdt : 0 ≤ mInf * A * dt := mul_nonneg (mul_nonneg hm hA) hdt
  unfold poolCap
  constructor
  · exact mul_nonneg (mul_nonneg (mul_nonneg hm hf0) hA) hdt
  · have : mInf * (1 - Real.exp (-(kBeta * Real.sqrt (4 * A / Real.pi)))) * A * dt
        = (1 - Real.exp (-(kBeta * Real.sqrt (4 * A / Real.pi)))) * (mInf * A * dt) := by ring
    rw [this]
    calc (1 - Real.exp (-(kBeta * Real.sqrt (4 * A / Real.pi)))) * (mInf * A * dt)
        ≤ 1 * (mInf * A * dt) := mul_le_mul_of_nonneg_right hf1 hmAdt
      _ = mInf * A * dt := one_mul _

/-- The vapour made: `dv = min(skin, C (T - T_b) / L, cap)` (since 30c019a without `cap`; the
pool's burning-rate cap `cap ≥ 0` since 2026-11-05: without it the pilot boiled a whole pool off
in a few steps into one cell, too rich to burn). An uncapped liquid takes `cap` large. -/
noncomputable def liquidVapour (C T Tb L skin cap : ℝ) : ℝ := min (min skin (C * (T - Tb) / L)) cap

/-- The surface temperature after it: `T - dv L / C`. -/
noncomputable def liquidT (C T Tb L skin cap : ℝ) : ℝ := T - liquidVapour C T Tb L skin cap * L / C

/-- The energy `C T + L · (vapour made)` is conserved, whatever the cap. -/
theorem chemistry_liquid_energy {C T Tb L skin cap : ℝ} (hC : 0 < C) :
    C * liquidT C T Tb L skin cap + L * liquidVapour C T Tb L skin cap = C * T := by
  unfold liquidT
  field_simp
  ring

/-- The surface never falls below its boiling point, and never warms (for `T > T_b`,
`skin ≥ 0`, `cap ≥ 0`). -/
theorem chemistry_liquid_above_boiling {C T Tb L skin cap : ℝ} (hC : 0 < C) (hL : 0 < L)
    (hT : Tb < T) (hs : 0 ≤ skin) (hcap : 0 ≤ cap) :
    Tb ≤ liquidT C T Tb L skin cap ∧ liquidT C T Tb L skin cap ≤ T ∧
      0 ≤ liquidVapour C T Tb L skin cap := by
  have hle : liquidVapour C T Tb L skin cap ≤ C * (T - Tb) / L :=
    (min_le_left _ _).trans (min_le_right _ _)
  have hnn : 0 ≤ liquidVapour C T Tb L skin cap :=
    le_min (le_min hs (div_nonneg (mul_nonneg hC.le (by linarith)) hL.le)) hcap
  have h1 : liquidVapour C T Tb L skin cap * L / C ≤ T - Tb := by
    rw [div_le_iff₀ hC]
    have := (le_div_iff₀ hL).1 hle
    linarith
  have h2 : 0 ≤ liquidVapour C T Tb L skin cap * L / C := by positivity
  unfold liquidT
  exact ⟨by linarith, by linarith, hnn⟩

/-- The pool never gives off more than its burning rate allows over the step. -/
theorem chemistry_liquid_capped {C T Tb L skin cap : ℝ} :
    liquidVapour C T Tb L skin cap ≤ cap :=
  min_le_right _ _

/-- Unclamped (enough liquid in the layer, the cap not binding), the surface ends exactly at its
boiling point. -/
theorem chemistry_liquid_unclamped {C T Tb L skin cap : ℝ} (hC : 0 < C) (hL : 0 < L)
    (h : C * (T - Tb) / L ≤ skin) (hc : C * (T - Tb) / L ≤ cap) :
    liquidT C T Tb L skin cap = Tb := by
  unfold liquidT liquidVapour
  rw [min_eq_right h, min_eq_left hc]
  field_simp
  ring

/-! ## Fresh fuel mixed into the surface layer -/

/-- The code's update `Ts ← (Ts max(0, C - Cf) + T_amb Cf) / max(C, 1e-9)`, with `C` the
layer's heat capacity after the fresh fuel joined and `Cf` the fresh fuel's (and its water's). -/
noncomputable def mixFresh (T Tamb C Cf : ℝ) : ℝ :=
  (T * max 0 (C - Cf) + Tamb * Cf) / max C 1e-9

/-- Enthalpy is conserved: `C T' = (C - Cf) T + Cf T_amb`. -/
theorem chemistry_mixFresh_enthalpy {T Tamb C Cf : ℝ} (hCf : Cf ≤ C) (hC : 1e-9 ≤ C) :
    C * mixFresh T Tamb C Cf = (C - Cf) * T + Cf * Tamb := by
  unfold mixFresh
  rw [max_eq_right (by linarith), max_eq_left hC]
  have : C ≠ 0 := by positivity
  field_simp

/-- The new temperature lies between `T_amb` and the old `T`. -/
theorem chemistry_mixFresh_between {T Tamb C Cf : ℝ} (hCf0 : 0 ≤ Cf) (hCf : Cf ≤ C) (hC : 1e-9 ≤ C) :
    min T Tamb ≤ mixFresh T Tamb C Cf ∧ mixFresh T Tamb C Cf ≤ max T Tamb := by
  have hCpos : 0 < C := lt_of_lt_of_le (by norm_num) hC
  have he := chemistry_mixFresh_enthalpy (T := T) (Tamb := Tamb) hCf hC
  have hm : mixFresh T Tamb C Cf = ((C - Cf) * T + Cf * Tamb) / C := by
    rw [← he]; field_simp
  rw [hm]
  constructor
  · rw [le_div_iff₀ hCpos]
    have h1 := min_le_left T Tamb
    have h2 := min_le_right T Tamb
    nlinarith
  · rw [div_le_iff₀ hCpos]
    have h1 := le_max_left T Tamb
    have h2 := le_max_right T Tamb
    nlinarith

/-- `fresh = min(d, max(0, F' - (skin - d)))` refills the layer exactly: with `skin = min F S`
before the reaction, `d ≤ skin` reacted and `F' = F - d`, the new layer `min F' S` is the old
layer's remainder plus `fresh`. So `C - Cf` is the heat capacity of what was at `Ts`. -/
theorem chemistry_fresh_refills_skin {F S d : ℝ} (hd0 : 0 ≤ d) :
    min (F - d) S = (min F S - d) + min d (max 0 ((F - d) - (min F S - d))) := by
  rcases le_total F S with hFS | hSF
  · rw [min_eq_left hFS]
    rw [min_eq_left (by linarith : F - d ≤ S)]
    have : (F - d) - (F - d) = 0 := by ring
    rw [this, max_self, min_eq_right hd0]
    ring
  · rw [min_eq_right hSF]
    rcases le_total (F - d) S with h | h
    · rw [min_eq_left h, max_eq_right (by linarith), min_eq_right (by linarith)]
      ring
    · rw [min_eq_right h, max_eq_right (by linarith), min_eq_left (by linarith)]
      ring

/-! ## Ash: the oxide of a metal stays as heat capacity -/

/-- `solidCapacity` before and after an oxidation step of `dOx`: the fuel mass falls from `F`
to `F - dOx`, the surface layer is `min(F, S)` at `c_f` J kg⁻¹ K⁻¹, and since 30c019a the ash
adds `ash · dOx · c_ash` to `inertC` (`c_ash = 900`, MgO). -/
noncomputable def capBefore (inert F S cf w : ℝ) : ℝ := inert + min F S * cf + w * 4186

noncomputable def capAfter (inert F S cf w ash dOx cAsh : ℝ) : ℝ :=
  (inert + ash * dOx * cAsh) + min (F - dOx) S * cf + w * 4186

/-- The change is the ash's capacity less the surface fuel burnt. -/
theorem chemistry_ash_capacity (inert F S cf w ash dOx cAsh : ℝ) :
    capAfter inert F S cf w ash dOx cAsh - capBefore inert F S cf w =
      ash * dOx * cAsh - cf * (min F S - min (F - dOx) S) := by
  unfold capAfter capBefore; ring

/-- While the bulk refills the surface layer (`F - dOx ≥ S`), oxidising a metal never lowers the
solid's heat capacity. -/
theorem chemistry_ash_capacity_mono {inert F S cf w ash dOx cAsh : ℝ} (hbulk : S ≤ F - dOx)
    (hd : 0 ≤ dOx) (ha : 0 ≤ ash) (hc : 0 ≤ cAsh) :
    capBefore inert F S cf w ≤ capAfter inert F S cf w ash dOx cAsh := by
  have h := chemistry_ash_capacity inert F S cf w ash dOx cAsh
  rw [min_eq_right hbulk, min_eq_right (by linarith : S ≤ F)] at h
  have : 0 ≤ ash * dOx * cAsh := by positivity
  linarith

/-- Nor at any stage when the oxide holds at least the heat of the fuel it came from,
`c_f ≤ ash · c_ash`. -/
theorem chemistry_ash_capacity_mono_cp {inert F S cf w ash dOx cAsh : ℝ} (hd : 0 ≤ dOx)
    (hcf : 0 ≤ cf) (hcp : cf ≤ ash * cAsh) :
    capBefore inert F S cf w ≤ capAfter inert F S cf w ash dOx cAsh := by
  have h := chemistry_ash_capacity inert F S cf w ash dOx cAsh
  have hm : min F S - min (F - dOx) S ≤ dOx := by
    rcases le_total F S with h1 | h1
    · rw [min_eq_left h1, min_eq_left (by linarith : F - dOx ≤ S)]; linarith
    · rw [min_eq_right h1]
      have := min_le_left (F - dOx) S
      have := min_le_right (F - dOx) S
      rcases le_total (F - dOx) S with h2 | h2
      · rw [min_eq_left h2]; linarith
      · rw [min_eq_right h2]; linarith
  have : cf * (min F S - min (F - dOx) S) ≤ cf * dOx := mul_le_mul_of_nonneg_left hm hcf
  nlinarith

/-- But not once the item's remaining fuel lies within its surface layer: magnesium is given
wood's `fuelHeatCapacity` of 2000 J kg⁻¹ K⁻¹, and its 1.658 kg of MgO per kg burnt hold only
1492 J K⁻¹. With 0.1 kg left in a 0.2 kg layer, burning 0.05 kg lowers the capacity from 200 to
174.6 J K⁻¹. -/
theorem chemistry_ash_capacity_counterexample :
    capAfter 0 0.1 0.2 2000 0 1.658 0.05 900 < capBefore 0 0.1 0.2 2000 0 := by
  norm_num [capAfter, capBefore, min_def]

/-! ## `march`: ray energy bookkeeping -/

/-- What lies past the cell a ray is in. -/
inductive Next where
  | gas | outside | solid

/-- One cell of a ray's path: its optical depth `κ Y_soot l` (0 in the emitter's own cell and
in solids) and what comes next. -/
structure Seg where
  tau : ℝ
  next : Next

/-- Where a ray's energy went. -/
structure Tally where
  gas : ℝ
  solid : ℝ
  out : ℝ
  dropped : ℝ

/-- The energy a gas cell takes: `left (1 - exp(-τ))` if `τ > 0`. -/
noncomputable def taken (left tau : ℝ) : ℝ := if 0 < tau then left * (1 - Real.exp (-tau)) else 0

/-! ### The version before commit 30c019a, which dropped the rest -/

/-- The loop of `march` before 30c019a, with `k` iterations of `rayReach` left, at path index `n` with `left`
joules; it stops when `left ≤ cut` (`cut = 1e-9 · energy`). -/
noncomputable def marchLoopOld (cut : ℝ) (path : ℕ → Seg) : ℕ → ℕ → ℝ → Tally
  | 0, _, left => ⟨0, 0, 0, left⟩
  | k + 1, n, left =>
    if left ≤ cut then ⟨0, 0, 0, left⟩ else
      match (path n).next with
      | .outside => ⟨taken left (path n).tau, 0, left - taken left (path n).tau, 0⟩
      | .solid => ⟨taken left (path n).tau, left - taken left (path n).tau, 0, 0⟩
      | .gas =>
        ⟨taken left (path n).tau +
            (marchLoopOld cut path k (n + 1) (left - taken left (path n).tau)).gas,
          (marchLoopOld cut path k (n + 1) (left - taken left (path n).tau)).solid,
          (marchLoopOld cut path k (n + 1) (left - taken left (path n).tau)).out,
          (marchLoopOld cut path k (n + 1) (left - taken left (path n).tau)).dropped⟩

/-- `march` before 30c019a, with `rayReach` iterations. -/
noncomputable def marchOld (reach : ℕ) (cut : ℝ) (path : ℕ → Seg) (energy : ℝ) : Tally :=
  marchLoopOld cut path reach 0 energy

theorem march_taken_mem {left : ℝ} (tau : ℝ) (hl : 0 ≤ left) :
    0 ≤ taken left tau ∧ taken left tau ≤ left := by
  unfold taken
  split_ifs with h
  · have h1 : Real.exp (-tau) < 1 := by rw [Real.exp_lt_one_iff]; linarith
    have h2 := Real.exp_pos (-tau)
    constructor <;> nlinarith
  · exact ⟨le_rfl, hl⟩

/-- What a cell passes on is `left exp(-τ)` for `τ ≥ 0`: the paper's absorption probability
`1 - exp(-κ Y l)` taken as an expected value. -/
theorem march_taken_transmit (left : ℝ) {tau : ℝ} (ht : 0 ≤ tau) :
    left - taken left tau = left * Real.exp (-tau) := by
  unfold taken
  split_ifs with h
  · ring
  · have : tau = 0 := le_antisymm (not_lt.1 h) ht
    rw [this]; simp

/-- Two cells in a row pass on `left exp(-(τ₁ + τ₂))`. -/
theorem march_taken_transmit_two (left : ℝ) {τ₁ τ₂ : ℝ} (h₁ : 0 ≤ τ₁) (h₂ : 0 ≤ τ₂) :
    (left - taken left τ₁) - taken (left - taken left τ₁) τ₂ = left * Real.exp (-(τ₁ + τ₂)) := by
  rw [march_taken_transmit _ h₂, march_taken_transmit _ h₁, neg_add, Real.exp_add]; ring

/-- Bookkeeping of the old loop: gas + solid + outside + dropped = what entered. -/
theorem marchLoopOld_balance (cut : ℝ) (path : ℕ → Seg) (k : ℕ) :
    ∀ n left, (marchLoopOld cut path k n left).gas + (marchLoopOld cut path k n left).solid +
      (marchLoopOld cut path k n left).out + (marchLoopOld cut path k n left).dropped = left := by
  induction k with
  | zero => intro n left; simp [marchLoopOld]
  | succ k ih =>
    intro n left
    simp only [marchLoopOld]
    split_ifs
    · simp
    · cases (path n).next with
      | outside => simp only; ring
      | solid => simp only; ring
      | gas =>
        simp only
        have := ih (n + 1) (left - taken left (path n).tau)
        linarith

/-- Ray bookkeeping before 30c019a: the energy left in gas cells, plus that deposited in the
first solid, plus that leaving the domain, plus what was dropped, equals the energy emitted. -/
theorem marchOld_balance (reach : ℕ) (cut : ℝ) (path : ℕ → Seg) (energy : ℝ) :
    (marchOld reach cut path energy).gas + (marchOld reach cut path energy).solid +
      (marchOld reach cut path energy).out + (marchOld reach cut path energy).dropped = energy :=
  marchLoopOld_balance cut path reach 0 energy

/-- Every part of the old tally is non-negative. -/
theorem marchLoopOld_nonneg (cut : ℝ) (path : ℕ → Seg) (k : ℕ) :
    ∀ n left, 0 ≤ left → 0 ≤ (marchLoopOld cut path k n left).gas ∧
      0 ≤ (marchLoopOld cut path k n left).solid ∧ 0 ≤ (marchLoopOld cut path k n left).out ∧
      0 ≤ (marchLoopOld cut path k n left).dropped := by
  induction k with
  | zero => intro n left hl; simp [marchLoopOld, hl]
  | succ k ih =>
    intro n left hl
    have ht := march_taken_mem (path n).tau hl
    simp only [marchLoopOld]
    split_ifs
    · simp [hl]
    · cases (path n).next with
      | outside => simp only; exact ⟨ht.1, le_rfl, by linarith, le_rfl⟩
      | solid => simp only; exact ⟨ht.1, by linarith, le_rfl, le_rfl⟩
      | gas =>
        simp only
        obtain ⟨a, b, c, d⟩ := ih (n + 1) (left - taken left (path n).tau) (by linarith)
        exact ⟨by linarith, b, c, d⟩

/-- Before 30c019a, energy above the cutoff was dropped only when the ray had used all its
`rayReach` steps in gas: then nothing reached a solid or left the domain. -/
theorem marchLoopOld_dropped_of_gt {cut : ℝ} (hcut : 0 ≤ cut) (path : ℕ → Seg) (k : ℕ) :
    ∀ n left, cut < (marchLoopOld cut path k n left).dropped →
      (marchLoopOld cut path k n left).solid = 0 ∧ (marchLoopOld cut path k n left).out = 0 ∧
        ∀ i < k, (path (n + i)).next = .gas := by
  induction k with
  | zero => intro n left _; simp [marchLoopOld]
  | succ k ih =>
    intro n left h
    by_cases hc : left ≤ cut
    · have e : marchLoopOld cut path (k + 1) n left = ⟨0, 0, 0, left⟩ := by
        simp only [marchLoopOld]; rw [ite_eq_left hc]
      rw [e] at h; simp only at h; linarith
    · cases hn : (path n).next with
      | outside =>
        have e : marchLoopOld cut path (k + 1) n left =
            ⟨taken left (path n).tau, 0, left - taken left (path n).tau, 0⟩ := by
          simp only [marchLoopOld]; rw [ite_eq_right hc, hn]
        rw [e] at h; simp only at h; linarith
      | solid =>
        have e : marchLoopOld cut path (k + 1) n left =
            ⟨taken left (path n).tau, left - taken left (path n).tau, 0, 0⟩ := by
          simp only [marchLoopOld]; rw [ite_eq_right hc, hn]
        rw [e] at h; simp only at h; linarith
      | gas =>
        have e : marchLoopOld cut path (k + 1) n left =
            ⟨taken left (path n).tau +
                (marchLoopOld cut path k (n + 1) (left - taken left (path n).tau)).gas,
              (marchLoopOld cut path k (n + 1) (left - taken left (path n).tau)).solid,
              (marchLoopOld cut path k (n + 1) (left - taken left (path n).tau)).out,
              (marchLoopOld cut path k (n + 1) (left - taken left (path n).tau)).dropped⟩ := by
          simp only [marchLoopOld]; rw [ite_eq_right hc, hn]
        rw [e] at h ⊢
        simp only at h ⊢
        obtain ⟨h1, h2, h3⟩ := ih (n + 1) _ h
        refine ⟨h1, h2, ?_⟩
        intro i hi
        rcases i with _ | i
        · simpa using hn
        · have := h3 i (by omega)
          rwa [show n + 1 + i = n + (i + 1) by ring] at this

/-- The defect the fix removed: through soot-free gas that does not end within `rayReach` cells,
the old `march` dropped all of the emitted energy, however far above the cutoff. -/
theorem marchOld_reach_drops_all (reach : ℕ) {cut energy : ℝ} (h : cut < energy) :
    marchOld reach cut (fun _ => ⟨0, .gas⟩) energy = ⟨0, 0, 0, energy⟩ := by
  unfold marchOld
  suffices ∀ k n, marchLoopOld cut (fun _ => ⟨0, .gas⟩) k n energy = ⟨0, 0, 0, energy⟩ from this _ _
  intro k
  induction k with
  | zero => intro n; simp [marchLoopOld]
  | succ k ih =>
    intro n
    have hn : ¬ energy ≤ cut := not_le.2 h
    simp only [marchLoopOld, hn, ↓reduceIte]
    have h0 : taken energy 0 = 0 := by simp [taken]
    simp only [h0, sub_zero, ih, zero_add]

/-- What a ray carries after `k` absorbing cells from index `n`, cell by cell as `march` takes it. -/
noncomputable def leftFrom (path : ℕ → Seg) : ℕ → ℕ → ℝ → ℝ
  | _, 0, x => x
  | n, k + 1, x => leftFrom path (n + 1) k (x - taken x (path n).tau)

/-- It is `x exp(-Σ τ)`. -/
theorem march_leftFrom_eq (path : ℕ → Seg) (hτ : ∀ j, 0 ≤ (path j).tau) (k : ℕ) :
    ∀ n x, leftFrom path n k x = x * Real.exp (-∑ j ∈ Finset.range k, (path (n + j)).tau) := by
  induction k with
  | zero => intro n x; simp [leftFrom]
  | succ k ih =>
    intro n x
    simp only [leftFrom]
    have hS : ∑ j ∈ Finset.range k, (path (n + 1 + j)).tau =
        ∑ j ∈ Finset.range k, (path (n + (j + 1))).tau :=
      Finset.sum_congr rfl (fun j _ => by rw [show n + 1 + j = n + (j + 1) by ring])
    rw [ih, march_taken_transmit _ (hτ n), Finset.sum_range_succ', hS, Nat.add_zero, mul_assoc,
      ← Real.exp_add]
    congr 2
    ring

/-- Through gas cells only, with no cutoff, the old loop leaves `leftFrom` at the end and puts
the rest in the gas; the current loop then deposits that remainder in the end cell
(`marchLoop_vs_old`). -/
theorem marchLoopOld_all_gas {cut : ℝ} (hcut : cut < 0) (path : ℕ → Seg)
    (hgas : ∀ j, (path j).next = .gas) (k : ℕ) :
    ∀ n left, 0 ≤ left → marchLoopOld cut path k n left =
      ⟨left - leftFrom path n k left, 0, 0, leftFrom path n k left⟩ := by
  induction k with
  | zero => intro n left _; simp [marchLoopOld, leftFrom]
  | succ k ih =>
    intro n left hl
    have hc : ¬ left ≤ cut := by linarith
    have ht := march_taken_mem (path n).tau hl
    have e : marchLoopOld cut path (k + 1) n left =
        ⟨taken left (path n).tau +
            (marchLoopOld cut path k (n + 1) (left - taken left (path n).tau)).gas,
          (marchLoopOld cut path k (n + 1) (left - taken left (path n).tau)).solid,
          (marchLoopOld cut path k (n + 1) (left - taken left (path n).tau)).out,
          (marchLoopOld cut path k (n + 1) (left - taken left (path n).tau)).dropped⟩ := by
      simp only [marchLoopOld]; rw [ite_eq_right hc, hgas n]
    rw [e, ih (n + 1) _ (by linarith)]
    simp only [leftFrom]
    congr 1
    ring

/-- The port's deterministic attenuation is the paper's Monte Carlo absorption in expectation:
with `τ_j = κ Y_j ℓ_j`, the energy a ray carries past `k` gas cells is supplement Algorithm 1's
expected pass-through `E Π (1 - P_abs,j)`, and the gas keeps the rest
(`Flamescapes.SA1_balance`). -/
theorem march_expected {κ : ℝ} {Y l : ℕ → ℝ} (path : ℕ → Seg)
    (hτ : ∀ j, (path j).tau = κ * Y j * l j) (hnn : ∀ j, 0 ≤ κ * Y j * l j) (E : ℝ) (k : ℕ) :
    leftFrom path 0 k E = Flamescapes.SA1_pass E (fun j => Flamescapes.Pabs κ (Y j) (l j)) k := by
  rw [Flamescapes.SA1_pass_exp, march_leftFrom_eq path (fun j => by rw [hτ]; exact hnn j)]
  simp only [zero_add, hτ]

/-! ### The current version: the rest goes into the cell the ray ends in -/

/-- Since 30c019a, what a ray still carries when its loop stops (at `rayReach` or below the
cut-off) goes into the cell it is in: its gas (`T`) or its solid (`Ts`). -/
noncomputable def endDeposit (gasEnd : Bool) (left : ℝ) : Tally :=
  if gasEnd then ⟨left, 0, 0, 0⟩ else ⟨0, left, 0, 0⟩

/-- The loop of `march`, with `isGas n` the kind of the cell at path index `n`. -/
noncomputable def marchLoop (cut : ℝ) (path : ℕ → Seg) (isGas : ℕ → Bool) : ℕ → ℕ → ℝ → Tally
  | 0, n, left => endDeposit (isGas n) left
  | k + 1, n, left =>
    if left ≤ cut then endDeposit (isGas n) left else
      match (path n).next with
      | .outside => ⟨taken left (path n).tau, 0, left - taken left (path n).tau, 0⟩
      | .solid => ⟨taken left (path n).tau, left - taken left (path n).tau, 0, 0⟩
      | .gas =>
        ⟨taken left (path n).tau +
            (marchLoop cut path isGas k (n + 1) (left - taken left (path n).tau)).gas,
          (marchLoop cut path isGas k (n + 1) (left - taken left (path n).tau)).solid,
          (marchLoop cut path isGas k (n + 1) (left - taken left (path n).tau)).out,
          (marchLoop cut path isGas k (n + 1) (left - taken left (path n).tau)).dropped⟩

/-- `march(sim, x, y, z, dir, energy, from, skipOwn)` with `rayReach` iterations. -/
noncomputable def march (reach : ℕ) (cut : ℝ) (path : ℕ → Seg) (isGas : ℕ → Bool)
    (energy : ℝ) : Tally :=
  marchLoop cut path isGas reach 0 energy

lemma endDeposit_balance (b : Bool) (left : ℝ) :
    (endDeposit b left).gas + (endDeposit b left).solid + (endDeposit b left).out = left ∧
      (endDeposit b left).dropped = 0 := by
  cases b <;> simp [endDeposit]

/-- Bookkeeping of the loop: gas + solid + outside = what entered, and nothing is dropped. -/
theorem marchLoop_balance (cut : ℝ) (path : ℕ → Seg) (isGas : ℕ → Bool) (k : ℕ) :
    ∀ n left, (marchLoop cut path isGas k n left).gas + (marchLoop cut path isGas k n left).solid +
      (marchLoop cut path isGas k n left).out = left ∧
      (marchLoop cut path isGas k n left).dropped = 0 := by
  induction k with
  | zero => intro n left; exact endDeposit_balance _ _
  | succ k ih =>
    intro n left
    simp only [marchLoop]
    split_ifs
    · exact endDeposit_balance _ _
    · cases (path n).next with
      | outside => simp only; constructor <;> first | ring | trivial
      | solid => simp only; constructor <;> first | ring | trivial
      | gas =>
        simp only
        obtain ⟨h1, h2⟩ := ih (n + 1) (left - taken left (path n).tau)
        exact ⟨by linarith, h2⟩

/-- Ray bookkeeping in `march`: the energy left in gas cells, plus that in solids, plus that
leaving the domain, equals the energy emitted, with nothing dropped. -/
theorem march_balance (reach : ℕ) (cut : ℝ) (path : ℕ → Seg) (isGas : ℕ → Bool) (energy : ℝ) :
    (march reach cut path isGas energy).gas + (march reach cut path isGas energy).solid +
      (march reach cut path isGas energy).out = energy ∧
      (march reach cut path isGas energy).dropped = 0 :=
  marchLoop_balance cut path isGas reach 0 energy

/-- Every part is non-negative. -/
theorem marchLoop_nonneg (cut : ℝ) (path : ℕ → Seg) (isGas : ℕ → Bool) (k : ℕ) :
    ∀ n left, 0 ≤ left → 0 ≤ (marchLoop cut path isGas k n left).gas ∧
      0 ≤ (marchLoop cut path isGas k n left).solid ∧
      0 ≤ (marchLoop cut path isGas k n left).out := by
  have hend : ∀ (b : Bool) (left : ℝ), 0 ≤ left →
      0 ≤ (endDeposit b left).gas ∧ 0 ≤ (endDeposit b left).solid ∧
        0 ≤ (endDeposit b left).out := by
    intro b left hl; cases b <;> simp [endDeposit, hl]
  induction k with
  | zero => intro n left hl; exact hend _ _ hl
  | succ k ih =>
    intro n left hl
    have ht := march_taken_mem (path n).tau hl
    simp only [marchLoop]
    split_ifs
    · exact hend _ _ hl
    · cases (path n).next with
      | outside => simp only; exact ⟨ht.1, le_rfl, by linarith⟩
      | solid => simp only; exact ⟨ht.1, by linarith, le_rfl⟩
      | gas =>
        simp only
        obtain ⟨a, b, c⟩ := ih (n + 1) (left - taken left (path n).tau) (by linarith)
        exact ⟨by linarith, b, c⟩

/-- The fix moves exactly what the old version dropped into the end cell: gas plus solid is the
old gas plus solid plus dropped, and what leaves the domain is unchanged. -/
theorem marchLoop_vs_old (cut : ℝ) (path : ℕ → Seg) (isGas : ℕ → Bool) (k : ℕ) :
    ∀ n left, (marchLoop cut path isGas k n left).gas + (marchLoop cut path isGas k n left).solid =
        (marchLoopOld cut path k n left).gas + (marchLoopOld cut path k n left).solid +
          (marchLoopOld cut path k n left).dropped ∧
      (marchLoop cut path isGas k n left).out = (marchLoopOld cut path k n left).out := by
  have hend : ∀ (b : Bool) (left : ℝ), (endDeposit b left).gas + (endDeposit b left).solid = left ∧
      (endDeposit b left).out = 0 := by
    intro b left; cases b <;> simp [endDeposit]
  induction k with
  | zero =>
    intro n left
    obtain ⟨h1, h2⟩ := hend (isGas n) left
    change (endDeposit (isGas n) left).gas + (endDeposit (isGas n) left).solid = 0 + 0 + left ∧
      (endDeposit (isGas n) left).out = 0
    exact ⟨by linarith, h2⟩
  | succ k ih =>
    intro n left
    simp only [marchLoop, marchLoopOld]
    split_ifs
    · obtain ⟨h1, h2⟩ := hend (isGas n) left
      exact ⟨by simp only; linarith, h2⟩
    · cases (path n).next with
      | outside => simp only; constructor <;> first | ring | trivial
      | solid => simp only; constructor <;> first | ring | trivial
      | gas =>
        simp only
        obtain ⟨h1, h2⟩ := ih (n + 1) (left - taken left (path n).tau)
        exact ⟨by linarith, h2⟩

/-- Through soot-free gas that does not end within `rayReach` cells, the energy that the old
version dropped now all goes into the gas cell the ray ends in. -/
theorem march_reach_deposits (reach : ℕ) {cut energy : ℝ} (h : cut < energy) :
    march reach cut (fun _ => ⟨0, .gas⟩) (fun _ => true) energy = ⟨energy, 0, 0, 0⟩ := by
  unfold march
  suffices ∀ k n, marchLoop cut (fun _ => ⟨0, .gas⟩) (fun _ => true) k n energy =
      ⟨energy, 0, 0, 0⟩ from this _ _
  intro k
  induction k with
  | zero => intro n; simp [marchLoop, endDeposit]
  | succ k ih =>
    intro n
    have hn : ¬ energy ≤ cut := not_le.2 h
    simp only [marchLoop, hn, ↓reduceIte]
    have h0 : taken energy 0 = 0 := by simp [taken]
    simp only [h0, sub_zero, ih, zero_add]

/-! ## `radiation`: the emitter never cools below the midpoint with ambient -/

/-- `radiation` caps a cell's emission at half its heat above ambient and subtracts it:
for `T > T_amb`, `C > 0` and any `e ≥ 0` the cell stays at or above `(T + T_amb) / 2`. -/
theorem radiation_emitter_cap {T Tamb C e : ℝ} (hC : 0 < C) (he : 0 ≤ e) (hT : Tamb < T) :
    (T + Tamb) / 2 ≤ T - min e (0.5 * C * (T - Tamb)) / C ∧
      0 ≤ min e (0.5 * C * (T - Tamb)) := by
  constructor
  · have h1 := min_le_right e (0.5 * C * (T - Tamb))
    have h2 : min e (0.5 * C * (T - Tamb)) / C ≤ 0.5 * (T - Tamb) := by
      rw [div_le_iff₀ hC]; linarith
    linarith
  · exact le_min he (by nlinarith)

/-- The rays carry equal shares that add up to the emitted energy. -/
theorem radiation_shares (emitted : ℝ) {rays : ℕ} (h : 0 < rays) :
    ∑ _r ∈ Finset.range rays, emitted / rays = emitted := by
  rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
  field_simp

/-- `particleToGrid` adds `(excess + own) / C` to the solid, `C` its heat capacity: supplement
Algorithm 2's line 7 with the division its note 4 asks for, so the solid gains exactly the
deposited energy (`Flamescapes.SA2_line7_fixed_energy`). -/
theorem particleToGrid_energy {Ts q C : ℝ} (hC : C ≠ 0) : C * ((Ts + q / C) - Ts) = q := by
  field_simp; ring

/-! ## `gasDensity` (Eq. 7) and `buoyancy` (Eq. 6) -/

section Density

variable {ι : Type*} [Fintype ι]

/-- The code's `Σ_k Y_k / M_k + max(0, 1 - Σ_k Y_k) / M_N₂` (mol kg⁻¹). -/
noncomputable def molPerKg (Y M : ι → ℝ) (MN2 : ℝ) : ℝ :=
  ∑ s, Y s / M s + max 0 (1 - ∑ s, Y s) / MN2

/-- `gasDensity(T, Y) = P_AMBIENT / (R_GAS T molPerKg)`. -/
noncomputable def gasDensity (T : ℝ) (Y M : ι → ℝ) (MN2 : ℝ) : ℝ :=
  P_AMBIENT / (R_GAS * T * molPerKg Y M MN2)

/-- The ideal gas law over the mixture: `p = ρ R T Σ Y/M`. -/
theorem gasDensity_eos {T : ℝ} {Y M : ι → ℝ} {MN2 : ℝ} (hT : 0 < T)
    (hn : 0 < molPerKg Y M MN2) :
    P_AMBIENT = gasDensity T Y M MN2 * R_GAS * T * molPerKg Y M MN2 := by
  unfold gasDensity
  have := R_GAS_pos
  field_simp

/-- Density falls strictly as temperature rises. -/
theorem gasDensity_strictAntiOn {Y M : ι → ℝ} {MN2 : ℝ} (hn : 0 < molPerKg Y M MN2) :
    StrictAntiOn (fun T => gasDensity T Y M MN2) (Set.Ioi 0) := by
  intro T₁ h₁ T₂ h₂ h12
  simp only [Set.mem_Ioi] at h₁ h₂
  unfold gasDensity
  apply div_lt_div_of_pos_left P_AMBIENT_pos (by have := R_GAS_pos; positivity)
  have := R_GAS_pos
  apply mul_lt_mul_of_pos_right _ hn
  exact mul_lt_mul_of_pos_left h12 this

/-- At a fixed composition the density ratio is the inverse temperature ratio. -/
theorem gasDensity_ratio {T₀ T : ℝ} {Y M : ι → ℝ} {MN2 : ℝ} (h₀ : 0 < T₀) (hT : 0 < T)
    (hn : 0 < molPerKg Y M MN2) :
    gasDensity T₀ Y M MN2 / gasDensity T Y M MN2 = T / T₀ := by
  unfold gasDensity
  have := R_GAS_pos
  have := P_AMBIENT_pos
  field_simp

end Density

/-- `buoyancy`: the face velocity gains `dt g (ρ₀/ρ - 1)`, positive (upward) exactly when the
gas is lighter than ambient. -/
theorem buoyancy_sign {g rho0 rho : ℝ} (hg : 0 < g) (hrho : 0 < rho) :
    0 < g * (rho0 / rho - 1) ↔ rho < rho0 := by
  rw [mul_pos_iff_of_pos_left hg, sub_pos, one_lt_div hrho]

/-- The port resolves note 9 of Flamescapes: `g (ρ₀/ρ - 1)` upward is Eq. 6's term with gravity
`-g` pointing down and the sign read as intended (`Flamescapes.F6_buoy_intended`). -/
theorem buoyancy_eq_F6 (g rho0 rho : ℝ) :
    g * (rho0 / rho - 1) = Flamescapes.F6_buoy_intended (-g) rho0 rho := by
  unfold Flamescapes.F6_buoy_intended; ring

/-- And under the ideal gas law at fixed composition (`ρ T = ρ₀ T₀`) it is Fire-X's
`-g_z (1 - T_amb/T)` (with `g_z = -g`) times `ρ₀ / ρ = T / T₀`. -/
theorem buoyancy_vs_FireX {g rho0 rho T T0 : ℝ} (hrho : 0 < rho) (hT : 0 < T) (hT0 : 0 < T0)
    (hgas : rho * T = rho0 * T0) :
    g * (rho0 / rho - 1) = (rho0 / rho) * FireX.X1_buoy (-g) T0 T ∧ rho0 / rho = T / T0 :=
  ⟨Flamescapes.F6_port_vs_X1 hrho hT hT0 hgas, (Flamescapes.F6_vs_X1 (gz := -g) hrho hT hT0 hgas).2⟩

/-! ## `diffuse`: bounds, and what is and is not conserved -/

/-- One cell's explicit update with `λ ≤ 1/7` over at most six open neighbours stays inside any
interval holding the cell and its neighbours (so mass fractions stay non-negative). -/
theorem diffuse_between {n : ℕ} (hn : n ≤ 6) {lam y L U : ℝ} {yn : Fin n → ℝ}
    (hl0 : 0 ≤ lam) (hl : lam ≤ 1 / 7) (hy : L ≤ y ∧ y ≤ U) (hyn : ∀ i, L ≤ yn i ∧ yn i ≤ U) :
    L ≤ y + ∑ i, lam * (yn i - y) ∧ y + ∑ i, lam * (yn i - y) ≤ U := by
  have hn' : (n : ℝ) ≤ 6 := by exact_mod_cast hn
  have hw : 0 ≤ 1 - n * lam := by nlinarith
  have hsum : y + ∑ i, lam * (yn i - y) = (1 - n * lam) * y + lam * ∑ i, yn i := by
    rw [← Finset.mul_sum, Finset.sum_sub_distrib]; simp; ring
  rw [hsum]
  have hlo : (n : ℝ) * L ≤ ∑ i, yn i := by
    have := Finset.sum_le_sum (fun i (_ : i ∈ Finset.univ) => (hyn i).1)
    simpa using this
  have hhi : ∑ i, yn i ≤ (n : ℝ) * U := by
    have := Finset.sum_le_sum (fun i (_ : i ∈ Finset.univ) => (hyn i).2)
    simpa using this
  constructor <;> nlinarith [hy.1, hy.2]

/-- Heat: uncapped, `ρ_c c_p λT_c = k dt / h²` for every cell, so the pair exchange
`ρ_a c_a λ_a (T_b - T_a) + ρ_b c_b λ_b (T_a - T_b)` cancels: heat diffusion conserves energy. -/
theorem diffuse_heat_pair {Ka Kb kdt Ta Tb : ℝ} (ha : Ka = kdt) (hb : Kb = kdt) :
    Ka * (Tb - Ta) + Kb * (Ta - Tb) = 0 := by
  rw [ha, hb]; ring

/-- Species before 30c019a, each cell with its own coefficient: the pair's mass change is
`V (ρ_a λ_a - ρ_b λ_b)(Y_b - Y_a)`, which vanishes only when `ρ_a λ_a = ρ_b λ_b`; with
`λ ∝ (T/T₀)^1.75` and `ρ ∝ 1/T`, `ρλ ∝ T^0.75` differed between a hot and a cold cell. -/
theorem diffuse_species_pair_old (V ρa ρb la lb Ya Yb : ℝ) :
    V * ρa * (la * (Yb - Ya)) + V * ρb * (lb * (Ya - Yb)) = V * (ρa * la - ρb * lb) * (Yb - Ya) := by
  ring

theorem diffuse_species_counterexample_old :
    (1 : ℝ) * 1 * ((1 / 7) * (1 - 0)) + 1 * 0.5 * ((1 / 14) * (0 - 1)) ≠ 0 := by
  norm_num

/-- Since 30c019a a face takes the mean of its cells' `ρD`:
`lamD = min(0.5 (ρD_c + ρD_n) dt / (ρ_c h²), 1/7)`. -/
noncomputable def lamD (rhoDc rhoDn rhoc dt h : ℝ) : ℝ :=
  min (0.5 * (rhoDc + rhoDn) * dt / (rhoc * h ^ 2)) (1 / 7)

/-- It lies in `[0, 1/7]`, so `diffuse_between` applies: mass fractions stay within their
neighbours' bounds. -/
theorem diffuse_lamD_mem {rhoDc rhoDn rhoc dt h : ℝ} (hc : 0 ≤ rhoDc) (hn : 0 ≤ rhoDn)
    (hr : 0 < rhoc) (hdt : 0 ≤ dt) :
    0 ≤ lamD rhoDc rhoDn rhoc dt h ∧ lamD rhoDc rhoDn rhoc dt h ≤ 1 / 7 := by
  unfold lamD
  refine ⟨le_min (by positivity) (by norm_num), min_le_right _ _⟩

/-- Each species' mass is conserved by a pair exchange while the cap is not binding on either
side: `V ρ_c lamD_cn (Y_n - Y_c) + V ρ_n lamD_nc (Y_c - Y_n) = 0`. -/
theorem diffuse_species_pair {V rhoDc rhoDn rhoc rhon dt h Yc Yn : ℝ} (hc : 0 < rhoc)
    (hn : 0 < rhon) (hh : h ≠ 0)
    (hcapc : 0.5 * (rhoDc + rhoDn) * dt / (rhoc * h ^ 2) ≤ 1 / 7)
    (hcapn : 0.5 * (rhoDn + rhoDc) * dt / (rhon * h ^ 2) ≤ 1 / 7) :
    V * rhoc * (lamD rhoDc rhoDn rhoc dt h * (Yn - Yc)) +
      V * rhon * (lamD rhoDn rhoDc rhon dt h * (Yc - Yn)) = 0 := by
  unfold lamD
  rw [min_eq_left hcapc, min_eq_left hcapn]
  field_simp
  ring

/-- When the cap binds on one side it is not: with `ρ_c = 1`, `ρ_n = 0.1` and a face flux that
gives cell c the coefficient 0.1, cell n's would be 1 and is capped at 1/7. -/
theorem diffuse_species_capped :
    (1 : ℝ) * 1 * (0.1 * (1 - 0)) + 1 * 0.1 * (min 1 (1 / 7) * (0 - 1)) ≠ 0 := by
  norm_num [min_def]

/-! ## Egress quantities (`flamescapesRun.js`): visibility and the CO dose -/

/-- `visibility(rhoSoot, C, sigma)`: `S = C / (σ_m ρ_soot)` (m), capped at `MAX_VISIBILITY`
= 100 m; `C = 3` for light-reflecting signs, `σ_m = 8700` m² kg⁻¹. The soot density is
`ρ Y_soot`. -/
noncomputable def visibility (C σm rhoSoot : ℝ) : ℝ :=
  if C / 100 < σm * rhoSoot then C / (σm * rhoSoot) else 100

/-- Where it is not capped, it is the formula `C / (σ_m ρ Y_soot)`. -/
theorem visibility_formula {C σm rho Ysoot : ℝ} (h : C / 100 < σm * (rho * Ysoot)) :
    visibility C σm (rho * Ysoot) = C / (σm * rho * Ysoot) := by
  unfold visibility; rw [ite_eq_left h, mul_assoc]

/-- It lies in `(0, 100]` m. -/
theorem visibility_mem {C σm s : ℝ} (hC : 0 < C) :
    0 < visibility C σm s ∧ visibility C σm s ≤ 100 := by
  unfold visibility
  split_ifs with h
  · have hK : 0 < σm * s := lt_trans (by positivity) h
    refine ⟨div_pos hC hK, ?_⟩
    rw [div_le_iff₀ hK]
    have := (div_lt_iff₀ (by norm_num : (0 : ℝ) < 100)).1 h
    linarith
  · norm_num

/-- Visibility never rises with the soot density. -/
theorem visibility_antitone {C σm : ℝ} (hC : 0 < C) (hσ : 0 < σm) :
    Antitone (visibility C σm) := by
  intro s₁ s₂ h12
  have hK : σm * s₁ ≤ σm * s₂ := mul_le_mul_of_nonneg_left h12 hσ.le
  unfold visibility
  split_ifs with h2 h1 h1
  · have hpos : 0 < σm * s₁ := lt_trans (by positivity) h1
    exact div_le_div_of_nonneg_left hC.le hpos hK
  · have hK2 : 0 < σm * s₂ := lt_trans (by positivity) h2
    rw [div_le_iff₀ hK2]
    have := (div_lt_iff₀ (by norm_num : (0 : ℝ) < 100)).1 h2
    linarith
  · exact absurd (lt_of_lt_of_le h1 hK) h2
  · exact le_rfl

/-- And it falls strictly with the soot density wherever it is below the cap. -/
theorem visibility_strictAntiOn {C σm : ℝ} (hC : 0 < C) (hσ : 0 < σm) :
    StrictAntiOn (visibility C σm) {s | C / 100 < σm * s} := by
  intro s₁ h₁ s₂ h₂ h12
  have h₁ : C / 100 < σm * s₁ := h₁
  have h₂ : C / 100 < σm * s₂ := h₂
  unfold visibility
  rw [ite_eq_left h₁, ite_eq_left h₂]
  have hpos : 0 < σm * s₁ := lt_trans (by positivity) h₁
  exact div_lt_div_of_pos_left hC hpos (mul_lt_mul_of_pos_left h12 hσ)

/-- The cap joins the formula continuously: at `σ_m ρ_soot = C / 100` the formula gives 100. -/
theorem visibility_cap_continuous {C : ℝ} (hC : 0 < C) : C / (C / 100) = 100 := by
  field_simp

/-- Units: `σ_m ρ Y_soot` is an extinction coefficient (m⁻¹), so `S` is a length. -/
theorem visibility_units :
    (Dim.area / Dim.mass) * Dim.density * Dim.one = Dim.one / Dim.length ∧
      Dim.one / ((Dim.area / Dim.mass) * Dim.density * Dim.one) = Dim.length := by
  decide

/-- `coDoseRate(ppm) = ppm > 0 ? ppm / (35000 · 60) : 0` per second (ISO 13571:
35 000 ppm·min to FED 1). -/
noncomputable def coDoseRate (ppm : ℝ) : ℝ := if 0 < ppm then ppm / (35000 * 60) else 0

theorem coDoseRate_of_nonneg {ppm : ℝ} (h : 0 ≤ ppm) : coDoseRate ppm = ppm / (35000 * 60) := by
  unfold coDoseRate
  split_ifs with hp
  · rfl
  · rw [le_antisymm (not_lt.1 hp) h]; simp

theorem coDoseRate_monotone : Monotone coDoseRate := by
  intro a b hab
  unfold coDoseRate
  split_ifs with ha hb hb
  · exact div_le_div_of_nonneg_right hab (by norm_num)
  · exact absurd (lt_of_lt_of_le ha hab) hb
  · exact div_nonneg hb.le (by norm_num)
  · exact le_rfl

/-- The CO dose accumulated over `[0, t]` (s). -/
noncomputable def coDose (C : ℝ → ℝ) (t : ℝ) : ℝ := ∫ τ in (0 : ℝ)..t, coDoseRate (C τ)

/-- For a non-negative concentration the dose is the exposure in ppm·min over 35 000. -/
theorem coDoseRate_integral {C : ℝ → ℝ} (hC : ∀ τ, 0 ≤ C τ) (t : ℝ) :
    coDose C t = ((∫ τ in (0 : ℝ)..t, C τ) / 60) / 35000 := by
  unfold coDose
  simp_rw [coDoseRate_of_nonneg (hC _)]
  rw [intervalIntegral.integral_div]
  ring

/-- It reaches 1 exactly when the exposure reaches 35 000 ppm·min. -/
theorem coDoseRate_reaches_one {C : ℝ → ℝ} (hC : ∀ τ, 0 ≤ C τ) (t : ℝ) :
    coDose C t = 1 ↔ (∫ τ in (0 : ℝ)..t, C τ) / 60 = 35000 := by
  rw [coDoseRate_integral hC]
  constructor
  · intro h; field_simp at h; linarith
  · intro h; rw [h]; norm_num

/-- At a constant `c > 0` ppm it is `c t / (35000 × 60)`, ISO 13571's FED with a time to
incapacitation of `35000 × 60 / c` seconds (`Tenability.fed`). -/
theorem coDoseRate_const {c : ℝ} (hc : 0 < c) (t : ℝ) :
    coDose (fun _ => c) t = c * t / (35000 * 60) ∧
      coDose (fun _ => c) t = Tenability.fed (fun _ => 35000 * 60 / c) t := by
  have h1 : coDose (fun _ => c) t = c * t / (35000 * 60) := by
    unfold coDose
    simp only [coDoseRate_of_nonneg hc.le, intervalIntegral.integral_const, smul_eq_mul, sub_zero]
    ring
  refine ⟨h1, ?_⟩
  rw [h1, Tenability.T1_fed_const]
  field_simp

/-- At 35 000 ppm the dose reaches 1 after one minute; at 1000 ppm after 35 minutes. -/
theorem coDoseRate_examples :
    coDose (fun _ => 35000) 60 = 1 ∧ coDose (fun _ => 1000) (35 * 60) = 1 := by
  constructor
  · rw [(coDoseRate_const (by norm_num : (0 : ℝ) < 35000) 60).1]; norm_num
  · rw [(coDoseRate_const (by norm_num : (0 : ℝ) < 1000) (35 * 60)).1]; norm_num

/-! ## `voxelize.js`: volume fractions (Eq. 1) and interior fuel (Eqs. 3 to 5) -/

section Voxel

variable {ι : Type*} [Fintype ι]

/-- `volumeFractions`' final pass: a cell whose fractions sum past 1 is rescaled to sum 1. -/
noncomputable def normalise (φ : ι → ℝ) : ι → ℝ :=
  if 1 < ∑ m, φ m then fun m => φ m / ∑ m, φ m else φ

theorem volumeFractions_nonneg {φ : ι → ℝ} (h : ∀ m, 0 ≤ φ m) (m : ι) : 0 ≤ normalise φ m := by
  unfold normalise
  split_ifs with h1
  · exact div_nonneg (h m) (by linarith)
  · exact h m

/-- After it, a cell's fractions sum to at most 1. -/
theorem volumeFractions_sum_le_one (φ : ι → ℝ) : ∑ m, normalise φ m ≤ 1 := by
  unfold normalise
  split_ifs with h1
  · rw [← Finset.sum_div, div_self (by linarith)]
  · exact not_lt.1 h1

/-- And the proportions between materials are kept. -/
theorem volumeFractions_ratio (φ : ι → ℝ) (m m' : ι) :
    normalise φ m * φ m' = normalise φ m' * φ m := by
  unfold normalise
  split_ifs <;> ring

end Voxel

/-- `interiorFuel`: `E = q A`, `M = E / Δh`, `μ = M / N`. -/
noncomputable def interiorMean (q A Δh : ℝ) (N : ℕ) : ℝ := q * A / Δh / N

/-- Draws with mean `μ = M / N`, one per interior cell, have expected sum `M` (no independence
needed). -/
theorem interiorFuel_expected_sum {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω)
    {N : ℕ} (hN : 0 < N) (q A Δh : ℝ) (m : Fin N → Ω → ℝ) (hint : ∀ i, Integrable (m i) P)
    (hmean : ∀ i, ∫ ω, m i ω ∂P = interiorMean q A Δh N) :
    ∫ ω, ∑ i, m i ω ∂P = q * A / Δh := by
  rw [integral_finsetSum _ (fun i _ => hint i)]
  simp_rw [hmean]
  rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  unfold interiorMean
  have : (N : ℝ) ≠ 0 := by exact_mod_cast hN.ne'
  field_simp

/-- For the paper's draw `N(μ, σ²)` the mean is `μ`. -/
theorem interiorFuel_gaussian_mean (μ : ℝ) (v : NNReal) : ∫ x, x ∂gaussianReal μ v = μ :=
  integral_id_gaussianReal

/-- The code cuts each draw at zero, `max(0, μ + σ Z)`; for `σ > 0` that raises its mean
strictly above `μ`, so the cut draws' expected total exceeds `M` (by `N E[max(0, -X)]`, under
1e-4 of `M` for the default `σ = 0.3 μ`, a bound not proved here). Before 30c019a these were the
contents' masses; since then they are rescaled to sum to `M` (`interiorFuel_rescaled_sum`). -/
theorem interiorFuel_cut_raises_mean (μ : ℝ) {v : NNReal} (hv : v ≠ 0) :
    μ < ∫ x, max 0 x ∂gaussianReal μ v := by
  have hint : Integrable (fun x : ℝ => x) (gaussianReal μ v) :=
    (memLp_id_gaussianReal (μ := μ) (v := v) 1).integrable le_rfl
  have hint2 : Integrable (fun x : ℝ => max 0 x) (gaussianReal μ v) := by
    refine hint.mono (by fun_prop) ?_
    exact Filter.Eventually.of_forall (fun x => by
      simp only [Real.norm_eq_abs]
      rw [abs_of_nonneg (le_max_left 0 x)]
      exact max_le (abs_nonneg x) (le_abs_self x))
  have hdiff : Integrable (fun x : ℝ => max 0 x - x) (gaussianReal μ v) := hint2.sub hint
  have hnn : 0 ≤ fun x : ℝ => max 0 x - x := fun x => by
    simp only [Pi.zero_apply]; linarith [le_max_right 0 x]
  have hsupp : Function.support (fun x : ℝ => max 0 x - x) = Set.Iio 0 := by
    ext x
    simp only [Function.mem_support, Set.mem_Iio, ne_eq]
    constructor
    · intro h
      by_contra hx
      exact h (by rw [max_eq_right (not_lt.1 hx)]; ring)
    · intro hx h
      rw [max_eq_left hx.le] at h
      linarith
  have hpos : 0 < ∫ x, (max 0 x - x) ∂gaussianReal μ v := by
    rw [integral_pos_iff_support_of_nonneg hnn hdiff, hsupp]
    have hac := gaussianReal_absolutelyContinuous' μ hv
    refine pos_iff_ne_zero.2 (fun h00 => ?_)
    have := hac h00
    simp at this
  rw [integral_sub hint2 hint, integral_id_gaussianReal] at hpos
  linarith

/-- Since 30c019a `interiorFuel` rescales the cut draws by `M / total` when `total > 0`. -/
noncomputable def rescale {N : ℕ} (m : Fin N → ℝ) (M : ℝ) : Fin N → ℝ :=
  if 0 < ∑ i, m i then fun i => m i * (M / ∑ i, m i) else m

/-- The contents then hold exactly the paper's budget `M`. -/
theorem interiorFuel_rescaled_sum {N : ℕ} (m : Fin N → ℝ) (M : ℝ) (h : 0 < ∑ i, m i) :
    ∑ i, rescale m M i = M := by
  unfold rescale
  rw [ite_eq_left h, ← Finset.sum_mul]
  field_simp

/-- Each cell keeps a non-negative mass. -/
theorem interiorFuel_rescaled_nonneg {N : ℕ} {m : Fin N → ℝ} {M : ℝ} (hm : ∀ i, 0 ≤ m i)
    (hM : 0 ≤ M) (i : Fin N) : 0 ≤ rescale m M i := by
  unfold rescale
  split_ifs with h
  · exact mul_nonneg (hm i) (div_nonneg hM h.le)
  · exact hm i

/-- And the proportions between cells are those of the draws. -/
theorem interiorFuel_rescaled_ratio {N : ℕ} (m : Fin N → ℝ) (M : ℝ) (i j : Fin N) :
    rescale m M i * m j = rescale m M j * m i := by
  unfold rescale
  split_ifs <;> ring

/-- The draws are cut at zero before the rescaling: `max(0, μ + σ Z)` lies in `[0, ∞)` and is
`μ + σ Z` whenever that is non-negative. -/
theorem interiorFuel_cut (x : ℝ) : 0 ≤ max 0 x ∧ (0 ≤ x → max 0 x = x) :=
  ⟨le_max_left _ _, fun h => max_eq_right h⟩

end FireEgress.FlamescapesPort
