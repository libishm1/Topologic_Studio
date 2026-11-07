import Mathlib
import FireEgress.FireX
import FireEgress.FlamescapesPort
import FireEgress.Sprinklers

/-!
# Suppression: the spray zone, its steam, and what water does to the burn

Specifications for step 3 of the sprinkler work, the Eulerian spray zone of
`wiki/research/sprinkler-physics-2026-11-07.md` (section 9, items 8 to 15), proved before
`src/core/flamescapes.js` and its GPU port implement it. Activation and Evans's envelope (items 1
to 6 and 14 (i), (ii), (iv)) are in `Sprinklers.lean`, whose `decay`, `decay_antitone` and
`evans_energy` this file reuses, with the port's `survives`, `arrhenius`, `changeGas`,
`molPerKg`, `gasDensity` and `liquidT` (`FlamescapesPort.lean`) and Fire-X's `X29` to `X31`
(`FireX.lean`). Real arithmetic, as elsewhere; the GPU's float32 is not modelled. Each
counterexample sits beside the theorem it limits.

## Which theorem backs which part of the code

* The spray zone (item 8). In each step a column under an open head receives
  `m₀ = flux · weight · dt`; cell `j` evaporates `e_j ∈ [0, m_j]` and passes `m_{j+1} = m_j - e_j`
  down; what reaches the bottom is dropped at the floor, not added to the fuel. One column:
  `column_budget` (`m₀ = Σ e_j + m_bottom`), `column_nonneg`, `column_antitone`, `spray_column`.
  One head over its footprint, weights `w ≥ 0` with `Σ w = 1`: `spray_head` (emitted = evaporated
  + dropped, both non-negative). All open heads: `spray_budget`, and `spray_budget_merged` if the
  code sums the heads over a shared column before the pass.
* Evaporation in a cell (item 9), with the reference temperature at the boiling point `T_B` and
  water arriving at `T_w`: `e = c_w (T_B - T_w) + L` (`evapEnergy`, `evapEnergy_pos`),
  `dv = min(η m_w, C_g (T - T_B) / e)` (`evapMass`) and
  `T' = (C_g T - dv (e - c_v T_B)) / (C_g + c_v dv)` (`evapT`). `evaporate_energy` (i), the
  identity `C_g T = (C_g + c_v dv) T' + dv (e - c_v T_B)`; `evaporate_bounds` (ii),
  `0 ≤ dv ≤ m_w`; `evaporate_between` (iii), `T_B ≤ T' ≤ T` for any `dv` within the energy limit;
  `evaporate_step`, the three for the code's `dv`; `evaporate_limit_pinned`, `evaporate_vs_X30`,
  `evaporate_vs_liquid`.
* The steam (item 10), `addGas(sim, c, SPECIES.H2O, dv)` (`addVapour`). `single_h2o_nonneg`,
  `sum_single_h2o`, `vapour_guards`; `vapour_changeGas` (other species kept, water and the cell
  up by `dv`, fractions valid); `vapour_X31` (the fractions are Fire-X's Eq. 31 plus the
  vapour's own share); `vapour_mixture`, `molPerKg_pos`, `vapour_moles` (`+dv / M_w`),
  `vapour_heatCapacity` (`+c_p,w dv`: the `C_g + c_v dv` of item 9 with `c_v = CP[H2O]`);
  `vapour_dilutes`, `vapour_dilutes_O2` (the O₂ mole fraction `n_O₂ / (n + dv / M_w)` never
  rises, and falls strictly when `dv > 0` and there is oxygen).
* Dilution (item 11), the gas reaction of `chemistry` (`gasOmega`, `gasBurn`, the constants
  `gasA`, `gasE`, `gasFuelExponent = -0.3`, `gasOxygenExponent = 1.3`). `port_exponents`
  (`(a - 1) + b = 0`), `dilution_omega`, `gasBurn_mul`, `gasBurn_nonneg`,
  `dilution_scales_burn` (the burn is exactly `λ` times the undiluted), `dilution_heat` (and the
  stoichiometric bound `λ m_O HEAT_PER_O2`), `dilution_lambda` (the equation of state turns the
  steam into `λ = n / (n + dv / M_w)`, `dilution`).
* Cooling (item 12). `cooling_factor_lt_one`, `cooling_first_order`, `gasE_over_R`,
  `cooling_first_order_port`. Beside them the two counterexamples: `cooling_raises_cap` and
  `cooling_cap_counterexample` (the cap rises by `T/T'`, 5/3 from 1000 K to 600 K);
  `gasBurn_fast`, `cooling_fast_chemistry` (in the fast-chemistry limit the exact scheme burns
  `T/T'` times more after cooling) and `cooling_oxygen_limited` (the same whenever the oxygen cap
  binds before and after).
* Extinction (item 13), FDS's `EXTINCTION 1` if adopted (`xLim`, `Extinct`). `xLim_antitone`,
  `extinct_monotone`.
* Evans per cell (item 14 (iii)) and the code's cap `r exp(-(t - t_act)/τ) dt` held for the step
  (`evansCap`). `survives_anti`, `survives_inv_eq_decay`, `evans_cells` (per-cell `τ_c ≤ τ_max`),
  `evans_cells_uniform`, `evansCap_antitone`, `one_sub_exp_neg_ge`, `evans_energy_le` (completes
  `Sprinklers.evans_energy` to `Q (τ + dt)`), `evansCap_sum`.
* The wet surface (item 15), the moisture branch, oxidation and pyrolysis of `chemistry`
  (`dryWater`, `dryTs`, `oxidised`, `pyroFactor`, `Y_O2_AIR`, `T_BOIL`, `oxidationA`,
  `oxidationE`). `wet_skin_pinned` (i); `arrhenius_le_boiling` and `wet_skin_heat` (ii),
  `oxidation_at_boiling` (2.23e-5 per second for the port's kinetics, between 2e-5 and 2.5e-5);
  beside them the counterexample `pyrolysis_dilution_counterexample` (iii) and
  `pyrolysis_rises_with_steam`.

## Hypotheses added, and where the report differs

* Item 9 (i) also needs `0 ≤ dv`, so that `C_g + c_v dv ≠ 0`; the code's `dv` meets it by (ii).
  (ii) holds with `0 ≤ C_g`.
* Item 9 and Fire-X. The report says the identity reduces to `X30_enthalpy` when `T' = T_B`. It
  does not: at `T' = T_B` it reads `C_g (T - T_B) = dv e`, the energy limit binding
  (`evaporate_limit_pinned`). Eq. 30 is the case `e = 0` with the vapour's specific heat equal to
  the gas's (`evaporate_vs_X30`): Fire-X takes the evaporation energy from the liquid's sensible
  heat (Eqs. 26 to 28) and only mixes the vapour in at `T_B`, while the spray takes it from the
  gas and lets the vapour carry its sensible heat above `T_B`.
* Item 10. The comparison of mole fractions needs every molar mass positive (`M_q > 0`,
  `M_N₂ > 0`), not only `M_w`, so that the cell's moles are positive (`molPerKg_pos`). The guards
  of `changeGas` follow from valid fractions, `ρ > 0`, `V > 0` and `dv ≥ 0` (`vapour_guards`).
* Item 11 holds with `0 ≤ c_F`, `0 ≤ c_O`.
* Item 12. For the port `E/R` is 15 154.3 K (`gasE_over_R`), so the first-order statement holds up
  to 15 154 K, a little short of the report's 15 155 K.
* Item 13 holds under the weaker `0 ≤ X_OI` and `T_fb ≤ T_OI`.
* Item 14 (iii) needs `q_c ≥ 0` and `τ_c > 0` for each cell.
* Item 15 (ii). The code multiplies the oxidation constant by `Y_O₂ / 0.232`, so the bound by
  `arrhenius(A, E, T_B)` needs `Y_O₂ ≤ 0.232` in the gas beside the fuel; and `T_B > 0`.

## What is not covered

* Floating point, the GPU's fixed-point `atomicAdd` and the order in which heads and cells are
  visited: the budgets are identities over the reals.
* The per-cell fraction `η`, a saturation limit (a Clausius-Clapeyron cap would need the
  intermediate value theorem for a unique saturated `dv`), the spray's drag on the `v` faces and
  wetting through `moisture`.
* That the code meets the hypotheses: `0 ≤ e_j ≤ m_j`, the weights, valid fractions, a fuel
  cell's gas at most at air's oxygen.
-/

namespace FireEgress.Suppression

open Real Filter Topology
open FireEgress.FlamescapesPort (survives arrhenius changeGas molPerKg gasDensity Sp R_GAS
  HEAT_PER_O2 liquidT liquidVapour)

/-! ## The spray zone: water down a column (item 8) -/

section Column

variable {m e : ℕ → ℝ}

/-- Down one column the water entering the top is the water evaporated in the first `n` cells
plus the water leaving the `n`-th: `m 0 = Σ_{j<n} e j + m n`. -/
theorem column_budget (hstep : ∀ j, m (j + 1) = m j - e j) (n : ℕ) :
    m 0 = ∑ j ∈ Finset.range n, e j + m n := by
  induction n with
  | zero => simp
  | succ n ih => rw [Finset.sum_range_succ, hstep n]; linarith

/-- The water passing each cell of the column is never negative. -/
theorem column_nonneg (hstep : ∀ j, m (j + 1) = m j - e j) {n : ℕ}
    (he : ∀ j < n, 0 ≤ e j ∧ e j ≤ m j) (h0 : 0 ≤ m 0) : ∀ j ≤ n, 0 ≤ m j
  | 0, _ => h0
  | j + 1, hj => by rw [hstep j]; linarith [(he j (by omega)).2]

/-- The flux never increases downward. -/
theorem column_antitone (hstep : ∀ j, m (j + 1) = m j - e j) {n : ℕ}
    (he : ∀ j < n, 0 ≤ e j ∧ e j ≤ m j) {i j : ℕ} (hij : i ≤ j) (hj : j ≤ n) : m j ≤ m i := by
  induction j, hij using Nat.le_induction with
  | base => exact le_rfl
  | succ k hik ih =>
    rw [hstep k]
    linarith [(he k (by omega)).1, ih (by omega)]

/-- One column in full: the budget, every flux non-negative and non-increasing downward, and
the evaporated water between 0 and what entered. -/
theorem spray_column (hstep : ∀ j, m (j + 1) = m j - e j) {n : ℕ}
    (he : ∀ j < n, 0 ≤ e j ∧ e j ≤ m j) (h0 : 0 ≤ m 0) :
    m 0 = ∑ j ∈ Finset.range n, e j + m n ∧ (∀ j ≤ n, 0 ≤ m j) ∧
      (∀ i j, i ≤ j → j ≤ n → m j ≤ m i) ∧
      0 ≤ ∑ j ∈ Finset.range n, e j ∧ ∑ j ∈ Finset.range n, e j ≤ m 0 := by
  have hb := column_budget hstep n
  have hn := column_nonneg hstep he h0
  refine ⟨hb, hn, fun i j hij hj => column_antitone hstep he hij hj, ?_, ?_⟩
  · exact Finset.sum_nonneg fun j hj => (he j (Finset.mem_range.1 hj)).1
  · linarith [hn n le_rfl]

end Column

/-- One head: its flux `F` (kg/s) spread over the columns `S` of its footprint with weights
`w ≥ 0`, `Σ w = 1`, each column receiving `F w dt` in the step. The water emitted in the step is
the water evaporated in the columns plus the water dropped at their bottoms, both non-negative. -/
theorem spray_head {κ : Type*} {S : Finset κ} {w : κ → ℝ} {m e : κ → ℕ → ℝ} {n : κ → ℕ}
    {F dt : ℝ} (hF : 0 ≤ F) (hdt : 0 ≤ dt) (hw0 : ∀ c ∈ S, 0 ≤ w c) (hw : ∑ c ∈ S, w c = 1)
    (htop : ∀ c ∈ S, m c 0 = F * w c * dt)
    (hstep : ∀ c ∈ S, ∀ j, m c (j + 1) = m c j - e c j)
    (he : ∀ c ∈ S, ∀ j < n c, 0 ≤ e c j ∧ e c j ≤ m c j) :
    F * dt = ∑ c ∈ S, ∑ j ∈ Finset.range (n c), e c j + ∑ c ∈ S, m c (n c) ∧
      0 ≤ ∑ c ∈ S, ∑ j ∈ Finset.range (n c), e c j ∧ 0 ≤ ∑ c ∈ S, m c (n c) := by
  have hm0 : ∀ c ∈ S, 0 ≤ m c 0 := fun c hc => by
    rw [htop c hc]; exact mul_nonneg (mul_nonneg hF (hw0 c hc)) hdt
  refine ⟨?_, ?_, ?_⟩
  · rw [← Finset.sum_add_distrib]
    calc F * dt = ∑ c ∈ S, F * w c * dt := by
          rw [← Finset.sum_mul, ← Finset.mul_sum, hw, mul_one]
      _ = ∑ c ∈ S, m c 0 := Finset.sum_congr rfl fun c hc => (htop c hc).symm
      _ = _ := Finset.sum_congr rfl fun c hc => column_budget (hstep c hc) (n c)
  · exact Finset.sum_nonneg fun c hc =>
      Finset.sum_nonneg fun j hj => (he c hc j (Finset.mem_range.1 hj)).1
  · exact Finset.sum_nonneg fun c hc => column_nonneg (hstep c hc) (he c hc) (hm0 c hc) (n c) le_rfl

/-- All open heads, each processed over its own footprint `S h`: the water the heads emit in a
step is the water evaporated plus the water dropped at the floor. -/
theorem spray_budget {ι κ : Type*} {H : Finset ι} {S : ι → Finset κ} {F : ι → ℝ}
    {w : ι → κ → ℝ} {m e : ι → κ → ℕ → ℝ} {n : ι → κ → ℕ} {dt : ℝ}
    (hF : ∀ h ∈ H, 0 ≤ F h) (hdt : 0 ≤ dt) (hw0 : ∀ h ∈ H, ∀ c ∈ S h, 0 ≤ w h c)
    (hw : ∀ h ∈ H, ∑ c ∈ S h, w h c = 1)
    (htop : ∀ h ∈ H, ∀ c ∈ S h, m h c 0 = F h * w h c * dt)
    (hstep : ∀ h ∈ H, ∀ c ∈ S h, ∀ j, m h c (j + 1) = m h c j - e h c j)
    (he : ∀ h ∈ H, ∀ c ∈ S h, ∀ j < n h c, 0 ≤ e h c j ∧ e h c j ≤ m h c j) :
    (∑ h ∈ H, F h) * dt = ∑ h ∈ H, ∑ c ∈ S h, ∑ j ∈ Finset.range (n h c), e h c j +
        ∑ h ∈ H, ∑ c ∈ S h, m h c (n h c) ∧
      0 ≤ ∑ h ∈ H, ∑ c ∈ S h, ∑ j ∈ Finset.range (n h c), e h c j ∧
      0 ≤ ∑ h ∈ H, ∑ c ∈ S h, m h c (n h c) := by
  have one := fun h (hh : h ∈ H) =>
    spray_head (hF h hh) hdt (hw0 h hh) (hw h hh) (htop h hh) (hstep h hh) (he h hh)
  refine ⟨?_, Finset.sum_nonneg fun h hh => (one h hh).2.1,
    Finset.sum_nonneg fun h hh => (one h hh).2.2⟩
  rw [Finset.sum_mul, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun h hh => (one h hh).1

/-- Heads sharing columns: if a column takes the water of every head over it in one pass,
`m c 0 = Σ_h F_h w_h c dt` with each head's weights summing to one over the columns `S`, the
budget still closes. -/
theorem spray_budget_merged {ι κ : Type*} {H : Finset ι} {S : Finset κ} {F : ι → ℝ}
    {w : ι → κ → ℝ} {m e : κ → ℕ → ℝ} {n : κ → ℕ} {dt : ℝ}
    (hF : ∀ h ∈ H, 0 ≤ F h) (hdt : 0 ≤ dt) (hw0 : ∀ h ∈ H, ∀ c ∈ S, 0 ≤ w h c)
    (hw : ∀ h ∈ H, ∑ c ∈ S, w h c = 1)
    (htop : ∀ c ∈ S, m c 0 = ∑ h ∈ H, F h * w h c * dt)
    (hstep : ∀ c ∈ S, ∀ j, m c (j + 1) = m c j - e c j)
    (he : ∀ c ∈ S, ∀ j < n c, 0 ≤ e c j ∧ e c j ≤ m c j) :
    (∑ h ∈ H, F h) * dt = ∑ c ∈ S, ∑ j ∈ Finset.range (n c), e c j + ∑ c ∈ S, m c (n c) ∧
      0 ≤ ∑ c ∈ S, ∑ j ∈ Finset.range (n c), e c j ∧ 0 ≤ ∑ c ∈ S, m c (n c) := by
  have hm0 : ∀ c ∈ S, 0 ≤ m c 0 := fun c hc => by
    rw [htop c hc]
    exact Finset.sum_nonneg fun h hh => mul_nonneg (mul_nonneg (hF h hh) (hw0 h hh c hc)) hdt
  refine ⟨?_, ?_, ?_⟩
  · rw [← Finset.sum_add_distrib]
    calc (∑ h ∈ H, F h) * dt = ∑ h ∈ H, ∑ c ∈ S, F h * w h c * dt := by
          rw [Finset.sum_mul]
          refine Finset.sum_congr rfl fun h hh => ?_
          rw [← Finset.sum_mul, ← Finset.mul_sum, hw h hh, mul_one]
      _ = ∑ c ∈ S, ∑ h ∈ H, F h * w h c * dt := Finset.sum_comm
      _ = ∑ c ∈ S, m c 0 := Finset.sum_congr rfl fun c hc => (htop c hc).symm
      _ = _ := Finset.sum_congr rfl fun c hc => column_budget (hstep c hc) (n c)
  · exact Finset.sum_nonneg fun c hc =>
      Finset.sum_nonneg fun j hj => (he c hc j (Finset.mem_range.1 hj)).1
  · exact Finset.sum_nonneg fun c hc => column_nonneg (hstep c hc) (he c hc) (hm0 c hc) (n c) le_rfl

/-! ## Evaporation in a cell (item 9) -/

/-- The energy to take 1 kg of water arriving at `Tw` to its boiling point `TB` and vaporise it,
`e = c_w (T_B - T_w) + L` (J/kg): the reference temperature is `T_B`. -/
noncomputable def evapEnergy (cw TB Tw L : ℝ) : ℝ := cw * (TB - Tw) + L

/-- It is positive for water arriving no hotter than its boiling point. -/
theorem evapEnergy_pos {cw TB Tw L : ℝ} (hcw : 0 ≤ cw) (hTw : Tw ≤ TB) (hL : 0 < L) :
    0 < evapEnergy cw TB Tw L := by
  unfold evapEnergy; nlinarith

/-- The water evaporated in a cell from the `m_w` passing it: a fraction `η` of it, at most what
the gas's heat above `T_B` pays for, `dv = min(η m_w, C_g (T - T_B) / e)`. -/
noncomputable def evapMass (η mw Cg T TB e : ℝ) : ℝ := min (η * mw) (Cg * (T - TB) / e)

/-- The gas temperature after it: the gas (heat capacity `C_g`, J/K) pays `dv e`, and the vapour
(specific heat `c_v`) joins it at the new temperature,
`T' = (C_g T - dv (e - c_v T_B)) / (C_g + c_v dv)`. -/
noncomputable def evapT (Cg cv TB e dv T : ℝ) : ℝ := (Cg * T - dv * (e - cv * TB)) / (Cg + cv * dv)

/-- (i) Energy relative to liquid water at `Tw` is conserved, the vapour carrying its sensible
heat above `T_B`: `C_g T = (C_g + c_v dv) T' + dv (e - c_v T_B)`. -/
theorem evaporate_energy {Cg cv TB e dv T : ℝ} (hCg : 0 < Cg) (hcv : 0 ≤ cv) (hdv : 0 ≤ dv) :
    Cg * T = (Cg + cv * dv) * evapT Cg cv TB e dv T + dv * (e - cv * TB) := by
  have hpos : 0 < Cg + cv * dv := by positivity
  unfold evapT
  field_simp
  ring

/-- (ii) The water evaporated lies between 0 and the water passing, and within the energy
limit. -/
theorem evaporate_bounds {η mw Cg T TB e : ℝ} (hη0 : 0 ≤ η) (hη1 : η ≤ 1) (hmw : 0 ≤ mw)
    (hCg : 0 ≤ Cg) (he : 0 < e) (hT : TB ≤ T) :
    0 ≤ evapMass η mw Cg T TB e ∧ evapMass η mw Cg T TB e ≤ mw ∧
      evapMass η mw Cg T TB e ≤ Cg * (T - TB) / e := by
  unfold evapMass
  refine ⟨le_min (mul_nonneg hη0 hmw) (div_nonneg (mul_nonneg hCg (by linarith)) he.le),
    (min_le_left _ _).trans (mul_le_of_le_one_left hmw hη1), min_le_right _ _⟩

/-- (iii) Within the energy limit the gas cools, but never below the boiling point. -/
theorem evaporate_between {Cg cv TB e dv T : ℝ} (hCg : 0 < Cg) (hcv : 0 ≤ cv) (he : 0 < e)
    (hT : TB ≤ T) (hdv0 : 0 ≤ dv) (hdv : dv ≤ Cg * (T - TB) / e) :
    TB ≤ evapT Cg cv TB e dv T ∧ evapT Cg cv TB e dv T ≤ T := by
  have hpos : 0 < Cg + cv * dv := by positivity
  have hde : dv * e ≤ Cg * (T - TB) := (le_div_iff₀ he).1 hdv
  unfold evapT
  constructor
  · rw [le_div_iff₀ hpos]; nlinarith
  · rw [div_le_iff₀ hpos]
    nlinarith [mul_nonneg (mul_nonneg hcv hdv0) (sub_nonneg.2 hT), mul_nonneg hdv0 he.le]

/-- The step as the code takes it, `dv = min(η m_w, C_g (T - T_B) / e)`: (i), (ii) and (iii)
together. -/
theorem evaporate_step {η mw Cg cv T TB e : ℝ} (hη0 : 0 ≤ η) (hη1 : η ≤ 1) (hmw : 0 ≤ mw)
    (hCg : 0 < Cg) (hcv : 0 ≤ cv) (he : 0 < e) (hT : TB ≤ T) :
    Cg * T = (Cg + cv * evapMass η mw Cg T TB e) * evapT Cg cv TB e (evapMass η mw Cg T TB e) T +
        evapMass η mw Cg T TB e * (e - cv * TB) ∧
      0 ≤ evapMass η mw Cg T TB e ∧ evapMass η mw Cg T TB e ≤ mw ∧
      TB ≤ evapT Cg cv TB e (evapMass η mw Cg T TB e) T ∧
      evapT Cg cv TB e (evapMass η mw Cg T TB e) T ≤ T := by
  obtain ⟨h0, h1, h2⟩ := evaporate_bounds hη0 hη1 hmw hCg.le he hT
  obtain ⟨h3, h4⟩ := evaporate_between hCg hcv he hT h0 h2
  exact ⟨evaporate_energy hCg hcv h0, h0, h1, h3, h4⟩

/-- When the energy limit binds, the gas ends exactly at the boiling point: (i) then reads
`C_g (T - T_B) = dv e`. -/
theorem evaporate_limit_pinned {Cg cv TB e T : ℝ} (hCg : 0 < Cg) (hcv : 0 ≤ cv) (he : 0 < e)
    (hT : TB ≤ T) : evapT Cg cv TB e (Cg * (T - TB) / e) T = TB := by
  have hdv : 0 ≤ Cg * (T - TB) / e := div_nonneg (mul_nonneg hCg.le (by linarith)) he.le
  have hpos : 0 < Cg + cv * (Cg * (T - TB) / e) := by positivity
  unfold evapT
  rw [div_eq_iff hpos.ne']
  field_simp
  ring

/-- Fire-X's Eq. 30 is the case `e = 0` with the vapour's specific heat equal to the gas's:
mixing `Δρ` of vapour at `T_B` into `ρ` of gas at `T` by mass. -/
theorem evaporate_vs_X30 {c ρ Δρ TB T : ℝ} (hc : 0 < c) (hρ : 0 < ρ) (hΔ : 0 ≤ Δρ) :
    evapT (c * ρ) c TB 0 Δρ T = FireX.X30 ρ Δρ TB T := by
  have h : ρ + Δρ ≠ 0 := by positivity
  unfold evapT FireX.X30 FireX.X29
  field_simp
  ring

/-- With `c_v = 0` the update is the liquid item's of `chemistry` with latent heat `e`
(`FlamescapesPort.liquidT`), whose bookkeeping `chemistry_liquid_energy` proves. -/
theorem evaporate_vs_liquid {C T TB e skin cap : ℝ} (hC : 0 < C) :
    evapT C 0 TB e (liquidVapour C T TB e skin cap) T = liquidT C T TB e skin cap ∧
      C * evapT C 0 TB e (liquidVapour C T TB e skin cap) T + e * liquidVapour C T TB e skin cap =
        C * T := by
  have hC' : C ≠ 0 := hC.ne'
  have h : evapT C 0 TB e (liquidVapour C T TB e skin cap) T = liquidT C T TB e skin cap := by
    unfold evapT liquidT
    simp only [zero_mul, add_zero, sub_zero]
    rw [sub_div, mul_div_cancel_left₀ T hC']
  exact ⟨h, by rw [h]; exact FlamescapesPort.chemistry_liquid_energy hC⟩

/-! ## The steam (item 10) -/

/-- `addGas(sim, c, SPECIES.H2O, dv)`: `changeGas` with `dv` of water and nothing else. -/
noncomputable def addVapour (V rho : ℝ) (Y : Sp → ℝ) (dv : ℝ) : (Sp → ℝ) × ℝ :=
  changeGas V rho Y (Pi.single Sp.h2o dv)

/-- The steam's `delta` is never negative. -/
lemma single_h2o_nonneg {dv : ℝ} (hdv : 0 ≤ dv) (q : Sp) :
    0 ≤ (Pi.single Sp.h2o dv : Sp → ℝ) q := by
  rw [Pi.single_apply]; split_ifs <;> simp [hdv]

/-- Weighted over the species, the steam's `delta` counts only water. -/
lemma sum_single_h2o (f : Sp → ℝ) (dv : ℝ) :
    ∑ q, (Pi.single Sp.h2o dv : Sp → ℝ) q * f q = dv * f Sp.h2o := by
  rw [FlamescapesPort.sum_Sp]; simp

/-- The guards of `changeGas` hold for steam added to a cell with valid fractions. -/
lemma vapour_guards {V rho dv : ℝ} {Y : Sp → ℝ} (hV : 0 < V) (hrho : 0 < rho)
    (hY : ∀ q, 0 ≤ Y q) (hN : ∑ q, Y q ≤ 1) (hdv : 0 ≤ dv) :
    (∀ q, 0 ≤ Y q * (rho * V) + (Pi.single Sp.h2o dv : Sp → ℝ) q) ∧
      0 ≤ rho * V - ∑ q, Y q * (rho * V) ∧
      0 < rho * V + ∑ q, (Pi.single Sp.h2o dv : Sp → ℝ) q := by
  have hm : 0 < rho * V := mul_pos hrho hV
  have hs : ∑ q, (Pi.single Sp.h2o dv : Sp → ℝ) q = dv := by
    simp
  refine ⟨fun q => add_nonneg (mul_nonneg (hY q) hm.le) (single_h2o_nonneg hdv q), ?_, ?_⟩
  · rw [← Finset.sum_mul]; nlinarith
  · rw [hs]; linarith

/-- Steam through `changeGas`: every other species keeps its mass, water gains `dv`, the cell
gains `dv`, and the mass fractions stay valid (each non-negative, the tracked ones summing to
at most one, so nitrogen stays non-negative). -/
theorem vapour_changeGas {V rho dv : ℝ} {Y : Sp → ℝ} (hV : 0 < V) (hrho : 0 < rho)
    (hY : ∀ q, 0 ≤ Y q) (hN : ∑ q, Y q ≤ 1) (hdv : 0 ≤ dv) :
    (∀ q, q ≠ Sp.h2o →
        (addVapour V rho Y dv).1 q * ((addVapour V rho Y dv).2 * V) = Y q * (rho * V)) ∧
      (addVapour V rho Y dv).1 Sp.h2o * ((addVapour V rho Y dv).2 * V) =
        Y Sp.h2o * (rho * V) + dv ∧
      (addVapour V rho Y dv).2 * V = rho * V + dv ∧
      (∀ q, 0 ≤ (addVapour V rho Y dv).1 q) ∧ ∑ q, (addVapour V rho Y dv).1 q ≤ 1 := by
  obtain ⟨hδ, hN', hpos⟩ := vapour_guards hV hrho hY hN hdv
  have hs := FlamescapesPort.changeGas_species hV hδ hN' hpos
  have hm := FlamescapesPort.changeGas_mass hV hδ hN' hpos
  have hn := FlamescapesPort.changeGas_nitrogen hV hδ hN' hpos
  have hsum : ∑ q, (Pi.single Sp.h2o dv : Sp → ℝ) q = dv := by
    simp
  rw [hsum] at hm
  unfold addVapour
  have hM : 0 < (changeGas V rho Y (Pi.single Sp.h2o dv)).2 * V := by
    rw [hm]; have := mul_pos hrho hV; linarith
  refine ⟨fun q hq => ?_, ?_, hm, fun q => ?_, ?_⟩
  · rw [hs q, Pi.single_eq_of_ne hq, add_zero]
  · rw [hs Sp.h2o, Pi.single_eq_same]
  · refine le_of_mul_le_mul_right ?_ hM
    rw [zero_mul, hs q]
    exact hδ q
  · have h1 : (∑ q, (changeGas V rho Y (Pi.single Sp.h2o dv)).1 q) *
        ((changeGas V rho Y (Pi.single Sp.h2o dv)).2 * V) ≤
        1 * ((changeGas V rho Y (Pi.single Sp.h2o dv)).2 * V) := by
      rw [Finset.sum_mul]; linarith
    exact le_of_mul_le_mul_right h1 hM

/-- The other species' fractions scale as Fire-X's Eq. 31 with the vapour density `dv / V`, and
the water's is Eq. 31's plus the vapour's own share `1 - X29`, which closes the sum that Eq. 31
alone leaves below one (`FireX.X31_fix`). -/
theorem vapour_X31 {V rho dv : ℝ} {Y : Sp → ℝ} (hV : 0 < V) (hrho : 0 < rho)
    (hY : ∀ q, 0 ≤ Y q) (hN : ∑ q, Y q ≤ 1) (hdv : 0 ≤ dv) :
    (∀ q, q ≠ Sp.h2o → (addVapour V rho Y dv).1 q = FireX.X31 rho (dv / V) (Y q)) ∧
      (addVapour V rho Y dv).1 Sp.h2o =
        FireX.X31 rho (dv / V) (Y Sp.h2o) + (1 - FireX.X29 rho (dv / V)) := by
  obtain ⟨ho, hw, hm, -, -⟩ := vapour_changeGas hV hrho hY hN hdv
  have hM : 0 < rho * V + dv := by have := mul_pos hrho hV; linarith
  rw [hm] at ho hw
  have hρ : rho + dv / V ≠ 0 := by have := div_nonneg hdv hV.le; linarith
  constructor
  · intro q hq
    have h : (addVapour V rho Y dv).1 q = Y q * (rho * V) / (rho * V + dv) := by
      rw [eq_div_iff hM.ne']; exact ho q hq
    rw [h]
    unfold FireX.X31 FireX.X29
    field_simp
  · have h : (addVapour V rho Y dv).1 Sp.h2o = (Y Sp.h2o * (rho * V) + dv) / (rho * V + dv) := by
      rw [eq_div_iff hM.ne']; exact hw
    rw [h]
    unfold FireX.X31 FireX.X29
    field_simp
    ring

/-! ### Moles, heat capacity and the oxygen mole fraction -/

/-- Any per-kilogram mixture property of the code's form, `Σ Y_k f_k + max(0, 1 - Σ Y_k) f_N₂`
(moles with `f = 1/M`, heat capacity with `f = c_p`), times the cell's mass, rises by exactly the
steam's `dv f_H₂O`. -/
theorem vapour_mixture {V rho dv fN2 : ℝ} {Y f : Sp → ℝ} (hV : 0 < V) (hrho : 0 < rho)
    (hY : ∀ q, 0 ≤ Y q) (hN : ∑ q, Y q ≤ 1) (hdv : 0 ≤ dv) :
    (∑ q, (addVapour V rho Y dv).1 q * f q + max 0 (1 - ∑ q, (addVapour V rho Y dv).1 q) * fN2) *
        ((addVapour V rho Y dv).2 * V) =
      (∑ q, Y q * f q + max 0 (1 - ∑ q, Y q) * fN2) * (rho * V) + dv * f Sp.h2o := by
  obtain ⟨-, -, -, -, hsum'⟩ := vapour_changeGas hV hrho hY hN hdv
  obtain ⟨hδ, hN', hpos⟩ := vapour_guards hV hrho hY hN hdv
  have hs := FlamescapesPort.changeGas_species hV hδ hN' hpos
  have hn := FlamescapesPort.changeGas_nitrogen hV hδ hN' hpos
  unfold addVapour at hsum' ⊢
  set r := changeGas V rho Y (Pi.single Sp.h2o dv)
  rw [max_eq_right (by linarith), max_eq_right (by linarith)]
  have e1 : (∑ q, r.1 q * f q + (1 - ∑ q, r.1 q) * fN2) * (r.2 * V) =
      ∑ q, (r.1 q * (r.2 * V)) * f q + (r.2 * V - ∑ q, r.1 q * (r.2 * V)) * fN2 := by
    rw [add_mul, Finset.sum_mul]
    congr 1
    · exact Finset.sum_congr rfl fun q _ => by ring
    · rw [← Finset.sum_mul]; ring
  rw [e1, hn]
  simp_rw [hs, add_mul, Finset.sum_add_distrib, sum_single_h2o]
  rw [Finset.sum_mul, ← Finset.sum_mul]
  have e2 : ∑ q, Y q * (rho * V) * f q = ∑ q, Y q * f q * (rho * V) :=
    Finset.sum_congr rfl fun q _ => by ring
  rw [e2]
  ring

/-- The code's moles per kilogram are positive for valid fractions and positive molar
masses. -/
theorem molPerKg_pos {Y M : Sp → ℝ} {MN2 : ℝ} (hM : ∀ q, 0 < M q) (hMN2 : 0 < MN2)
    (hY : ∀ q, 0 ≤ Y q) (hN : ∑ q, Y q ≤ 1) : 0 < molPerKg Y M MN2 := by
  unfold molPerKg
  rw [max_eq_right (by linarith)]
  have h1 : 0 ≤ ∑ q, Y q / M q := Finset.sum_nonneg fun q _ => div_nonneg (hY q) (hM q).le
  rcases lt_or_eq_of_le hN with hlt | heq
  · have : 0 < (1 - ∑ q, Y q) / MN2 := div_pos (by linarith) hMN2
    linarith
  · have hpos : 0 < ∑ q, Y q / M q := by
      apply Finset.sum_pos' (fun q _ => div_nonneg (hY q) (hM q).le)
      by_contra hcon
      push Not at hcon
      have hz : ∑ q, Y q = 0 := Finset.sum_eq_zero fun q hq => by
        have h := (div_le_iff₀ (hM q)).1 (hcon q hq)
        linarith [hY q]
      linarith
    have : 0 ≤ (1 - ∑ q, Y q) / MN2 := div_nonneg (by linarith) hMN2.le
    linarith

/-- The moles in the cell rise by exactly `dv / M_H₂O`. -/
theorem vapour_moles {V rho dv MN2 : ℝ} {Y M : Sp → ℝ} (hV : 0 < V) (hrho : 0 < rho)
    (hY : ∀ q, 0 ≤ Y q) (hN : ∑ q, Y q ≤ 1) (hdv : 0 ≤ dv) :
    molPerKg (addVapour V rho Y dv).1 M MN2 * ((addVapour V rho Y dv).2 * V) =
      molPerKg Y M MN2 * (rho * V) + dv / M Sp.h2o := by
  have h := vapour_mixture (f := fun q => 1 / M q) (fN2 := 1 / MN2) hV hrho hY hN hdv
  unfold molPerKg
  simp only [mul_one_div] at h
  exact h

/-- The cell's heat capacity after the steam is the old one plus `c_p,H₂O dv`: in item 9,
`C_g + c_v dv` with `c_v = CP[H2O]` (1864 J/(kg K)) is the cell's own `mixtureCp` times its
mass after `changeGas`. -/
theorem vapour_heatCapacity {V rho dv cpN2 : ℝ} {Y cp : Sp → ℝ} (hV : 0 < V) (hrho : 0 < rho)
    (hY : ∀ q, 0 ≤ Y q) (hN : ∑ q, Y q ≤ 1) (hdv : 0 ≤ dv) :
    (∑ q, (addVapour V rho Y dv).1 q * cp q + max 0 (1 - ∑ q, (addVapour V rho Y dv).1 q) * cpN2) *
        ((addVapour V rho Y dv).2 * V) =
      (∑ q, Y q * cp q + max 0 (1 - ∑ q, Y q) * cpN2) * (rho * V) + cp Sp.h2o * dv := by
  rw [vapour_mixture hV hrho hY hN hdv, mul_comm dv]

/-- The mole fraction of species `q`, `(Y_q / M_q) / molPerKg`. -/
noncomputable def moleFraction (Y M : Sp → ℝ) (MN2 : ℝ) (q : Sp) : ℝ :=
  (Y q / M q) / molPerKg Y M MN2

/-- After the steam, the mole fraction of every species other than water is `n_q / (n + dv/M_w)`
with `n_q` and `n` the moles before: it never rises, and falls strictly when `dv > 0` and the
species is present. -/
theorem vapour_dilutes {V rho dv MN2 : ℝ} {Y M : Sp → ℝ} (hM : ∀ q, 0 < M q) (hMN2 : 0 < MN2)
    (hV : 0 < V) (hrho : 0 < rho) (hY : ∀ q, 0 ≤ Y q) (hN : ∑ q, Y q ≤ 1) (hdv : 0 ≤ dv)
    {q : Sp} (hq : q ≠ Sp.h2o) :
    moleFraction (addVapour V rho Y dv).1 M MN2 q =
        (Y q * (rho * V) / M q) / (molPerKg Y M MN2 * (rho * V) + dv / M Sp.h2o) ∧
      moleFraction (addVapour V rho Y dv).1 M MN2 q ≤ moleFraction Y M MN2 q ∧
      (0 < dv → 0 < Y q →
        moleFraction (addVapour V rho Y dv).1 M MN2 q < moleFraction Y M MN2 q) := by
  obtain ⟨ho, -, hm, hY', hN''⟩ := vapour_changeGas hV hrho hY hN hdv
  have hmass : 0 < rho * V := mul_pos hrho hV
  have hM' : 0 < (addVapour V rho Y dv).2 * V := by rw [hm]; linarith
  have hn := molPerKg_pos hM hMN2 hY hN
  have hn' := molPerKg_pos hM hMN2 hY' hN''
  have hmol := vapour_moles (M := M) (MN2 := MN2) hV hrho hY hN hdv
  have heq : moleFraction (addVapour V rho Y dv).1 M MN2 q =
      (Y q * (rho * V) / M q) / (molPerKg Y M MN2 * (rho * V) + dv / M Sp.h2o) := by
    have h2 : (addVapour V rho Y dv).2 ≠ 0 := by
      intro h; rw [h, zero_mul] at hM'; exact lt_irrefl _ hM'
    have hMq := (hM q).ne'
    have hn'' := hn'.ne'
    have hV' := hV.ne'
    unfold moleFraction
    rw [← hmol, ← ho q hq]
    field_simp
  have hold : moleFraction Y M MN2 q =
      (Y q * (rho * V) / M q) / (molPerKg Y M MN2 * (rho * V)) := by
    have hMq := (hM q).ne'
    have hn'' := hn.ne'
    have hm' := hmass.ne'
    unfold moleFraction
    field_simp
  have hnum : 0 ≤ Y q * (rho * V) / M q := div_nonneg (mul_nonneg (hY q) hmass.le) (hM q).le
  have hd : 0 ≤ dv / M Sp.h2o := div_nonneg hdv (hM _).le
  refine ⟨heq, ?_, fun hdv' hYq => ?_⟩
  · rw [heq, hold]
    exact div_le_div_of_nonneg_left hnum (mul_pos hn hmass) (by linarith)
  · rw [heq, hold]
    have hd' : 0 < dv / M Sp.h2o := div_pos hdv' (hM _)
    exact div_lt_div_of_pos_left (div_pos (mul_pos hYq hmass) (hM q)) (mul_pos hn hmass)
      (by linarith)

/-- In particular oxygen: steam never raises the O₂ mole fraction, and lowers it strictly when
`dv > 0` and there is oxygen. -/
theorem vapour_dilutes_O2 {V rho dv MN2 : ℝ} {Y M : Sp → ℝ} (hM : ∀ q, 0 < M q)
    (hMN2 : 0 < MN2) (hV : 0 < V) (hrho : 0 < rho) (hY : ∀ q, 0 ≤ Y q) (hN : ∑ q, Y q ≤ 1)
    (hdv : 0 ≤ dv) :
    moleFraction (addVapour V rho Y dv).1 M MN2 Sp.o2 ≤ moleFraction Y M MN2 Sp.o2 ∧
      (0 < dv → 0 < Y Sp.o2 →
        moleFraction (addVapour V rho Y dv).1 M MN2 Sp.o2 < moleFraction Y M MN2 Sp.o2) :=
  (vapour_dilutes hM hMN2 hV hrho hY hN hdv (by decide)).2

/-! ## Dilution: the gas burn under steam (item 11) -/

/-- The gas reaction's rate constant in `chemistry`, `omega = k(T) c_F^(a-1) c_O^b` (s⁻¹), with
`c_F`, `c_O` the molar concentrations. -/
noncomputable def gasOmega (A E a b T cF cO : ℝ) : ℝ := arrhenius A E T * cF ^ (a - 1) * cO ^ b

/-- The fuel burnt in the step, `dF = min(m_F (1 - survives(omega, dt)), m_O / s)`, with `s` the
oxygen per kilogram of fuel (`st.o2`). -/
noncomputable def gasBurn (mF mO s ω dt : ℝ) : ℝ := min (mF * (1 - survives ω dt)) (mO / s)

/-- `DEFAULTS.gasA` (s⁻¹ with the concentration exponents). -/
noncomputable def gasA : ℝ := 8.3e5
/-- `DEFAULTS.gasE` (J/mol). -/
noncomputable def gasE : ℝ := 1.26e5
/-- `DEFAULTS.gasFuelExponent`, Westbrook and Dryer's. -/
noncomputable def gasFuelExponent : ℝ := -0.3
/-- `DEFAULTS.gasOxygenExponent`, Westbrook and Dryer's. -/
noncomputable def gasOxygenExponent : ℝ := 1.3

/-- The port's exponents satisfy `(a - 1) + b = 0`. -/
theorem port_exponents : (gasFuelExponent - 1) + gasOxygenExponent = 0 := by
  norm_num [gasFuelExponent, gasOxygenExponent]

/-- Scaling both concentrations by `λ > 0` leaves `omega` unchanged when `(a - 1) + b = 0`. -/
theorem dilution_omega {A E a b T cF cO lam : ℝ} (hl : 0 < lam) (hF : 0 ≤ cF) (hO : 0 ≤ cO)
    (hab : (a - 1) + b = 0) :
    gasOmega A E a b T (lam * cF) (lam * cO) = gasOmega A E a b T cF cO := by
  unfold gasOmega
  rw [Real.mul_rpow hl.le hF, Real.mul_rpow hl.le hO]
  have h : lam ^ (a - 1) * lam ^ b = 1 := by rw [← Real.rpow_add hl, hab, Real.rpow_zero]
  calc arrhenius A E T * (lam ^ (a - 1) * cF ^ (a - 1)) * (lam ^ b * cO ^ b)
      = arrhenius A E T * cF ^ (a - 1) * cO ^ b * (lam ^ (a - 1) * lam ^ b) := by ring
    _ = arrhenius A E T * cF ^ (a - 1) * cO ^ b := by rw [h, mul_one]

/-- The burn scales with the masses it is given. -/
theorem gasBurn_mul {mF mO s ω dt lam : ℝ} (hl : 0 ≤ lam) :
    gasBurn (lam * mF) (lam * mO) s ω dt = lam * gasBurn mF mO s ω dt := by
  unfold gasBurn
  rw [mul_min_of_nonneg _ _ hl]
  congr 1 <;> ring

/-- The burn is never negative. -/
theorem gasBurn_nonneg {mF mO s ω dt : ℝ} (hmF : 0 ≤ mF) (hmO : 0 ≤ mO) (hs : 0 ≤ s)
    (hdt : 0 ≤ dt) : 0 ≤ gasBurn mF mO s ω dt :=
  le_min (mul_nonneg hmF (by linarith [FlamescapesPort.survives_le_one ω hdt]))
    (div_nonneg hmO hs)

/-- Steam at fixed temperature and pressure scales the fuel's and the oxygen's masses and
concentrations in the cell by the same `λ ∈ (0, 1]`. With `(a - 1) + b = 0` the rate constant is
unchanged, so the fuel burnt is exactly `λ` times the undiluted, never more. -/
theorem dilution_scales_burn {A E a b T cF cO mF mO s dt lam : ℝ} (hl0 : 0 < lam)
    (hl1 : lam ≤ 1) (hF : 0 ≤ cF) (hO : 0 ≤ cO) (hab : (a - 1) + b = 0) (hmF : 0 ≤ mF)
    (hmO : 0 ≤ mO) (hs : 0 ≤ s) (hdt : 0 ≤ dt) :
    gasBurn (lam * mF) (lam * mO) s (gasOmega A E a b T (lam * cF) (lam * cO)) dt =
        lam * gasBurn mF mO s (gasOmega A E a b T cF cO) dt ∧
      gasBurn (lam * mF) (lam * mO) s (gasOmega A E a b T (lam * cF) (lam * cO)) dt ≤
        gasBurn mF mO s (gasOmega A E a b T cF cO) dt := by
  rw [dilution_omega hl0 hF hO hab, gasBurn_mul hl0.le]
  exact ⟨rfl, mul_le_of_le_one_left (gasBurn_nonneg hmF hmO hs hdt) hl1⟩

/-- The heat it releases, `dF · st.heat` with `st.heat = HEAT_PER_O2 · st.o2`
(`addProducts_heat_per_O2`), is `λ` times the undiluted and at most `λ m_O HEAT_PER_O2`: the
stoichiometric bound, scaled by `λ`. -/
theorem dilution_heat {A E a b T cF cO mF mO s dt lam : ℝ} (hl0 : 0 < lam) (hF : 0 ≤ cF)
    (hO : 0 ≤ cO) (hab : (a - 1) + b = 0) (hs : 0 < s) :
    gasBurn (lam * mF) (lam * mO) s (gasOmega A E a b T (lam * cF) (lam * cO)) dt *
          (HEAT_PER_O2 * s) =
        lam * (gasBurn mF mO s (gasOmega A E a b T cF cO) dt * (HEAT_PER_O2 * s)) ∧
      gasBurn (lam * mF) (lam * mO) s (gasOmega A E a b T (lam * cF) (lam * cO)) dt *
          (HEAT_PER_O2 * s) ≤ lam * (mO * HEAT_PER_O2) := by
  rw [dilution_omega hl0 hF hO hab, gasBurn_mul hl0.le]
  refine ⟨by ring, ?_⟩
  have hb : gasBurn mF mO s (gasOmega A E a b T cF cO) dt ≤ mO / s := min_le_right _ _
  have hH := FlamescapesPort.HEAT_PER_O2_pos
  have hs' := hs.ne'
  calc lam * gasBurn mF mO s (gasOmega A E a b T cF cO) dt * (HEAT_PER_O2 * s)
      ≤ lam * (mO / s) * (HEAT_PER_O2 * s) :=
        mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hb hl0.le) (mul_pos hH hs).le
    _ = lam * (mO * HEAT_PER_O2) := by field_simp

/-- The dilution factor of `dv` kg of steam in a cell holding `n` mol: `n / (n + dv / M_H₂O)`. -/
noncomputable def dilution (n dv Mw : ℝ) : ℝ := n / (n + dv / Mw)

/-- The equation of state turns the steam into `λ`: at a fixed temperature, after `changeGas` and
`gasDensity` every species but water has its molar concentration `ρ Y_q / M_q` multiplied by
`λ = n / (n + dv / M_H₂O) ∈ (0, 1]`, `n` the cell's moles before (`vapour_dilutes`). This is the
`λ` of `dilution_scales_burn`. -/
theorem dilution_lambda {V rho dv MN2 T : ℝ} {Y M : Sp → ℝ} (hM : ∀ q, 0 < M q)
    (hMN2 : 0 < MN2) (hV : 0 < V) (hrho : 0 < rho) (hY : ∀ q, 0 ≤ Y q) (hN : ∑ q, Y q ≤ 1)
    (hdv : 0 ≤ dv) (hT : 0 < T) {q : Sp} (hq : q ≠ Sp.h2o) :
    0 < dilution (molPerKg Y M MN2 * (rho * V)) dv (M Sp.h2o) ∧
      dilution (molPerKg Y M MN2 * (rho * V)) dv (M Sp.h2o) ≤ 1 ∧
      gasDensity T (addVapour V rho Y dv).1 M MN2 * (addVapour V rho Y dv).1 q / M q =
        dilution (molPerKg Y M MN2 * (rho * V)) dv (M Sp.h2o) *
          (gasDensity T Y M MN2 * Y q / M q) := by
  obtain ⟨-, -, -, hY', hN'⟩ := vapour_changeGas hV hrho hY hN hdv
  have hn := molPerKg_pos hM hMN2 hY hN
  have hn' := molPerKg_pos hM hMN2 hY' hN'
  have hmass : 0 < rho * V := mul_pos hrho hV
  have hN0 : 0 < molPerKg Y M MN2 * (rho * V) := mul_pos hn hmass
  have hd : 0 ≤ dv / M Sp.h2o := div_nonneg hdv (hM _).le
  have hfrac := (vapour_dilutes hM hMN2 hV hrho hY hN hdv hq).1
  have hR := FlamescapesPort.R_GAS_pos
  have hc : ∀ Z : Sp → ℝ, 0 < molPerKg Z M MN2 → gasDensity T Z M MN2 * Z q / M q =
      FlamescapesPort.P_AMBIENT / (R_GAS * T) * moleFraction Z M MN2 q := by
    intro Z hZ
    have := (hM q).ne'
    have := hZ.ne'
    have := hT.ne'
    have := hR.ne'
    unfold gasDensity moleFraction
    field_simp
  refine ⟨div_pos hN0 (by linarith), (div_le_one (by linarith)).2 (by linarith), ?_⟩
  rw [hc _ hn', hc _ hn, hfrac]
  have := (hM q).ne'
  have := hn.ne'
  have := hmass.ne'
  have := hT.ne'
  have := hR.ne'
  have hsum : molPerKg Y M MN2 * (rho * V) + dv / M Sp.h2o ≠ 0 := by linarith
  unfold dilution moleFraction
  field_simp

/-! ## Cooling (item 12) -/

/-- For `0 < T' < T ≤ E/R`: `(T/T') exp(-(E/R)(1/T' - 1/T)) < 1`. -/
theorem cooling_factor_lt_one {E R T T' : ℝ} (hE : 0 < E) (hR : 0 < R) (h0 : 0 < T')
    (h1 : T' < T) (hT : T ≤ E / R) : T / T' * exp (-(E / R * (1 / T' - 1 / T))) < 1 := by
  have hTpos : 0 < T := by linarith
  have hk : 1 ≤ E / R / T := by rw [le_div_iff₀ hTpos]; linarith
  have hx' : E / R * (1 / T' - 1 / T) = E / R / T * ((T - T') / T') := by
    have := hTpos.ne'; have := h0.ne'; have := hR.ne'
    field_simp
  have hq : 0 < (T - T') / T' := div_pos (by linarith) h0
  have hxge : (T - T') / T' ≤ E / R * (1 / T' - 1 / T) := by
    rw [hx']; exact le_mul_of_one_le_left hq.le hk
  have hxpos : 0 < E / R * (1 / T' - 1 / T) := lt_of_lt_of_le hq hxge
  have hexp := Real.add_one_lt_exp hxpos.ne'
  have hTT : T / T' = (T - T') / T' + 1 := by rw [div_add_one h0.ne', sub_add_cancel]
  have hlt : T / T' < exp (E / R * (1 / T' - 1 / T)) := by rw [hTT]; linarith
  calc T / T' * exp (-(E / R * (1 / T' - 1 / T)))
      < exp (E / R * (1 / T' - 1 / T)) * exp (-(E / R * (1 / T' - 1 / T))) :=
        mul_lt_mul_of_pos_right hlt (exp_pos _)
    _ = 1 := by rw [← Real.exp_add]; simp

/-- Cooling from `T` to `T'` at fixed pressure and mass fractions raises the cell's gas mass and
both concentrations by `T/T'` (`gasDensity_ratio`), yet for `0 < T' < T ≤ E/R` the first-order
burn rate `m_F omega` falls: the Arrhenius factor loses more than the mass gains. -/
theorem cooling_first_order {A E a b T T' mF cF cO : ℝ} (hA : 0 < A) (hE : 0 < E)
    (hab : (a - 1) + b = 0) (h0 : 0 < T') (h1 : T' < T) (hT : T ≤ E / R_GAS) (hmF : 0 < mF)
    (hcF : 0 < cF) (hcO : 0 < cO) :
    T / T' * mF * gasOmega A E a b T' (T / T' * cF) (T / T' * cO) <
      mF * gasOmega A E a b T cF cO := by
  have hTpos : 0 < T := by linarith
  have hk : 0 < T / T' := div_pos hTpos h0
  have hR := FlamescapesPort.R_GAS_pos
  rw [dilution_omega hk hcF.le hcO.le hab]
  unfold gasOmega arrhenius
  rw [ite_eq_left h0, ite_eq_left hTpos]
  have hsplit : exp (-E / (R_GAS * T')) =
      exp (-E / (R_GAS * T)) * exp (-(E / R_GAS * (1 / T' - 1 / T))) := by
    rw [← Real.exp_add]
    congr 1
    have := hTpos.ne'; have := h0.ne'; have := hR.ne'
    field_simp
    ring
  rw [hsplit]
  have hf := cooling_factor_lt_one hE hR h0 h1 hT
  have hP : 0 < mF * (A * exp (-E / (R_GAS * T)) * cF ^ (a - 1) * cO ^ b) := by positivity
  have key := mul_lt_mul_of_pos_left hf hP
  calc T / T' * mF * (A * (exp (-E / (R_GAS * T)) * exp (-(E / R_GAS * (1 / T' - 1 / T)))) *
          cF ^ (a - 1) * cO ^ b)
      = mF * (A * exp (-E / (R_GAS * T)) * cF ^ (a - 1) * cO ^ b) *
          (T / T' * exp (-(E / R_GAS * (1 / T' - 1 / T)))) := by ring
    _ < mF * (A * exp (-E / (R_GAS * T)) * cF ^ (a - 1) * cO ^ b) * 1 := key
    _ = mF * (A * exp (-E / (R_GAS * T)) * cF ^ (a - 1) * cO ^ b) := mul_one _

/-- For the port's `E = 1.26e5` J/mol, `E/R` lies between 15 154 and 15 155 K (15 154.3 K). -/
theorem gasE_over_R : 15154 < gasE / R_GAS ∧ gasE / R_GAS < 15155 := by
  have hR := FlamescapesPort.R_GAS_pos
  constructor
  · rw [lt_div_iff₀ hR]; norm_num [gasE, FlamescapesPort.R_GAS]
  · rw [div_lt_iff₀ hR]; norm_num [gasE, FlamescapesPort.R_GAS]

/-- With the port's kinetics, cooling never raises the first-order burn rate up to 15 154 K. -/
theorem cooling_first_order_port {T T' mF cF cO : ℝ} (h0 : 0 < T') (h1 : T' < T)
    (hT : T ≤ 15154) (hmF : 0 < mF) (hcF : 0 < cF) (hcO : 0 < cO) :
    T / T' * mF * gasOmega gasA gasE gasFuelExponent gasOxygenExponent T' (T / T' * cF)
        (T / T' * cO) <
      mF * gasOmega gasA gasE gasFuelExponent gasOxygenExponent T cF cO :=
  cooling_first_order (by norm_num [gasA]) (by norm_num [gasE]) port_exponents h0 h1
    (hT.trans gasE_over_R.1.le) hmF hcF hcO

/-- Counterexample to "cooling never helps the burn": the equation of state raises a cooled
cell's gas mass, and the stoichiometric cap `Y_O₂ m / s` with it, by `T/T'`. -/
theorem cooling_raises_cap {T T' V s MN2 : ℝ} {Y M : Sp → ℝ} (h0 : 0 < T') (h1 : T' < T)
    (hn : 0 < molPerKg Y M MN2) (hV : 0 < V) (hO : 0 < Y Sp.o2) (hs : 0 < s) :
    Y Sp.o2 * (gasDensity T' Y M MN2 * V) / s =
        T / T' * (Y Sp.o2 * (gasDensity T Y M MN2 * V) / s) ∧
      Y Sp.o2 * (gasDensity T Y M MN2 * V) / s < Y Sp.o2 * (gasDensity T' Y M MN2 * V) / s := by
  have hTpos : 0 < T := by linarith
  have hρ : 0 < gasDensity T Y M MN2 :=
    div_pos FlamescapesPort.P_AMBIENT_pos (mul_pos (mul_pos FlamescapesPort.R_GAS_pos hTpos) hn)
  have hratio := FlamescapesPort.gasDensity_ratio h0 hTpos hn
  have hρ' : gasDensity T' Y M MN2 = T / T' * gasDensity T Y M MN2 := by
    rw [← hratio, div_mul_cancel₀ _ hρ.ne']
  have hk : 1 < T / T' := (one_lt_div h0).2 h1
  have hcap : 0 < Y Sp.o2 * (gasDensity T Y M MN2 * V) / s :=
    div_pos (mul_pos hO (mul_pos hρ hV)) hs
  have heq : Y Sp.o2 * (gasDensity T' Y M MN2 * V) / s =
      T / T' * (Y Sp.o2 * (gasDensity T Y M MN2 * V) / s) := by rw [hρ']; ring
  exact ⟨heq, by rw [heq]; exact lt_mul_of_one_lt_left hcap hk⟩

/-- The report's numbers: from 1000 K to 600 K the cell holds 5/3 as much gas. -/
theorem cooling_cap_counterexample {Y M : Sp → ℝ} {MN2 : ℝ} (hn : 0 < molPerKg Y M MN2) :
    gasDensity 600 Y M MN2 = 5 / 3 * gasDensity 1000 Y M MN2 := by
  have h := FlamescapesPort.gasDensity_ratio (T₀ := 600) (T := 1000) (by norm_num) (by norm_num) hn
  have hρ : 0 < gasDensity 1000 Y M MN2 :=
    div_pos FlamescapesPort.P_AMBIENT_pos
      (mul_pos (mul_pos FlamescapesPort.R_GAS_pos (by norm_num)) hn)
  rw [div_eq_iff hρ.ne'] at h
  rw [h]; norm_num

/-- As `omega → ∞` (chemistry fast against the step) the burn tends to `min(m_F, m_O / s)`. -/
theorem gasBurn_fast {mF mO s dt : ℝ} (hdt : 0 < dt) :
    Tendsto (fun ω => gasBurn mF mO s ω dt) atTop (𝓝 (min mF (mO / s))) := by
  have h1 : Tendsto (fun ω : ℝ => max 0 ω * dt) atTop atTop :=
    (tendsto_atTop_mono (fun ω => le_max_right 0 ω) tendsto_id).atTop_mul_const hdt
  have h2 : Tendsto (fun ω : ℝ => survives ω dt) atTop (𝓝 0) := by
    refine Tendsto.congr (fun ω => ?_) (Real.tendsto_exp_atBot.comp (tendsto_neg_atTop_atBot.comp h1))
    simp only [Function.comp_apply, survives, neg_mul]
  have h3 : Tendsto (fun ω => mF * (1 - survives ω dt)) atTop (𝓝 (mF * (1 - 0))) :=
    (tendsto_const_nhds.sub h2).const_mul mF
  rw [sub_zero, mul_one] at h3
  unfold gasBurn
  exact h3.min tendsto_const_nhds

/-- Second counterexample: in the fast-chemistry limit the exact scheme burns `T/T'` times more
fuel after cooling, so `cooling_first_order` is a first-order (small `omega dt`) statement
only. -/
theorem cooling_fast_chemistry {T T' mF mO s dt : ℝ} (h0 : 0 < T') (h1 : T' < T) (hmF : 0 < mF)
    (hmO : 0 < mO) (hs : 0 < s) (hdt : 0 < dt) :
    Tendsto (fun ω => gasBurn mF mO s ω dt) atTop (𝓝 (min mF (mO / s))) ∧
      Tendsto (fun ω => gasBurn (T / T' * mF) (T / T' * mO) s ω dt) atTop
        (𝓝 (T / T' * min mF (mO / s))) ∧
      min mF (mO / s) < T / T' * min mF (mO / s) := by
  have hk : 1 < T / T' := (one_lt_div h0).2 h1
  refine ⟨gasBurn_fast hdt, ?_, lt_mul_of_one_lt_left (lt_min hmF (div_pos hmO hs)) hk⟩
  have h := gasBurn_fast (mF := T / T' * mF) (mO := T / T' * mO) (s := s) hdt
  rwa [mul_div_assoc, ← mul_min_of_nonneg _ _ (by linarith : (0 : ℝ) ≤ T / T')] at h

/-- The same away from the limit: whenever the oxygen cap binds before and after cooling, the
burn after is exactly `T/T'` times the burn before. -/
theorem cooling_oxygen_limited {T T' mF mO s ω ω' dt : ℝ} (h0 : 0 < T') (h1 : T' < T)
    (hmO : 0 < mO) (hs : 0 < s) (hcap : mO / s ≤ mF * (1 - survives ω dt))
    (hcap' : mO / s ≤ mF * (1 - survives ω' dt)) :
    gasBurn (T / T' * mF) (T / T' * mO) s ω' dt = T / T' * gasBurn mF mO s ω dt ∧
      gasBurn mF mO s ω dt < gasBurn (T / T' * mF) (T / T' * mO) s ω' dt := by
  have hk : 1 < T / T' := (one_lt_div h0).2 h1
  have e1 : gasBurn mF mO s ω dt = mO / s := min_eq_right hcap
  have e2 : gasBurn mF mO s ω' dt = mO / s := min_eq_right hcap'
  rw [gasBurn_mul (by linarith), e1, e2]
  exact ⟨rfl, lt_mul_of_one_lt_left (div_pos hmO hs) hk⟩

/-! ## Extinction, if adopted (item 13) -/

/-- FDS's `EXTINCTION 1` limiting oxygen fraction: `X_OI (T_OI - T)/(T_OI - T_∞)` below the
free-burn temperature `T_fb`, 0 from it on. -/
noncomputable def xLim (XOI TOI Tinf Tfb T : ℝ) : ℝ :=
  if T < Tfb then XOI * (TOI - T) / (TOI - Tinf) else 0

/-- A cell does not burn while its oxygen fraction is below the limit. -/
def Extinct (XOI TOI Tinf Tfb X T : ℝ) : Prop := X < xLim XOI TOI Tinf Tfb T

/-- The limit never rises with temperature. -/
theorem xLim_antitone {XOI TOI Tinf Tfb : ℝ} (hX : 0 ≤ XOI) (hinf : Tinf < TOI)
    (hfb : Tfb ≤ TOI) : Antitone (xLim XOI TOI Tinf Tfb) := by
  intro T₁ T₂ h
  have hd : 0 < TOI - Tinf := by linarith
  unfold xLim
  by_cases h2 : T₂ < Tfb
  · have h1 : T₁ < Tfb := lt_of_le_of_lt h h2
    rw [ite_eq_left h2, ite_eq_left h1]
    exact div_le_div_of_nonneg_right (by nlinarith) hd.le
  · rw [ite_eq_right h2]
    by_cases h1 : T₁ < Tfb
    · rw [ite_eq_left h1]
      exact div_nonneg (mul_nonneg hX (by linarith)) hd.le
    · rw [ite_eq_right h1]

/-- An extinct cell stays extinct when it is cooled or its oxygen diluted: water never re-lights
it. -/
theorem extinct_monotone {XOI TOI Tinf Tfb X T X' T' : ℝ} (hX : 0 ≤ XOI) (hinf : Tinf < TOI)
    (hfb : Tfb ≤ TOI) (h : Extinct XOI TOI Tinf Tfb X T) (hX' : X' ≤ X) (hT' : T' ≤ T) :
    Extinct XOI TOI Tinf Tfb X' T' :=
  lt_of_le_of_lt hX' (lt_of_lt_of_le h (xLim_antitone hX hinf hfb hT'))

/-! ## Evans per cell, and the code's cap (item 14 (iii)) -/

/-- `survives` falls as the rate rises. -/
theorem survives_anti {k k' dt : ℝ} (hdt : 0 ≤ dt) (h : k ≤ k') :
    survives k' dt ≤ survives k dt := by
  unfold survives
  apply Real.exp_le_exp.mpr
  have : max 0 k ≤ max 0 k' := max_le_max le_rfl h
  nlinarith

/-- With `k = 1/τ` it is Evans's decay: `Q survives(1/τ, t) = Sprinklers.decay Q 0 τ t`. -/
theorem survives_inv_eq_decay {Q τ : ℝ} (hτ : 0 < τ) (t : ℝ) :
    Q * survives (1 / τ) t = Sprinklers.decay Q 0 τ t := by
  unfold Sprinklers.decay
  rw [FlamescapesPort.survives_of_nonneg (by positivity)]
  congr 2
  ring

/-- Per cell with its own time constant: if cell `c`'s heat release `n` steps after activation is
at most `q_c survives(1/τ_c, n dt)`, with `Σ q_c = Q` and every `τ_c ≤ τ_max`, the total is at
most `Q survives(1/τ_max, n dt)`: the slowest cell's envelope bounds the room. -/
theorem evans_cells {κ : Type*} {S : Finset κ} {q τ : κ → ℝ} {hrr : κ → ℕ → ℝ} {Q τmax dt : ℝ}
    (hdt : 0 ≤ dt) (hq : ∀ c ∈ S, 0 ≤ q c) (hτ : ∀ c ∈ S, 0 < τ c)
    (hτmax : ∀ c ∈ S, τ c ≤ τmax) (hQ : ∑ c ∈ S, q c = Q)
    (hcell : ∀ c ∈ S, ∀ n : ℕ, hrr c n ≤ q c * survives (1 / τ c) (n * dt)) (n : ℕ) :
    ∑ c ∈ S, hrr c n ≤ Q * survives (1 / τmax) (n * dt) := by
  rw [← hQ, Finset.sum_mul]
  refine Finset.sum_le_sum fun c hc => (hcell c hc n).trans ?_
  exact mul_le_mul_of_nonneg_left
    (survives_anti (by positivity) (one_div_le_one_div_of_le (hτ c hc) (hτmax c hc))) (hq c hc)

/-- With one time constant for all cells the total follows Evans's envelope exactly. -/
theorem evans_cells_uniform {κ : Type*} {S : Finset κ} {q : κ → ℝ} {hrr : κ → ℕ → ℝ}
    {Q τ dt : ℝ} (hdt : 0 ≤ dt) (hq : ∀ c ∈ S, 0 ≤ q c) (hτ : 0 < τ) (hQ : ∑ c ∈ S, q c = Q)
    (hcell : ∀ c ∈ S, ∀ n : ℕ, hrr c n ≤ q c * survives (1 / τ) (n * dt)) (n : ℕ) :
    ∑ c ∈ S, hrr c n ≤ Sprinklers.decay Q 0 τ (n * dt) := by
  rw [← survives_inv_eq_decay hτ]
  exact evans_cells (τ := fun _ => τ) hdt hq (fun _ _ => hτ) (fun _ _ => le_rfl) hQ hcell n

/-- The code's cap on a covered cell's loss in the step that starts at `t`:
`r exp(-(t - t_act)/τ) dt`, held for the step. -/
noncomputable def evansCap (r tAct τ dt t : ℝ) : ℝ := Sprinklers.decay r tAct τ t * dt

/-- The cap never rises from one step to the next. -/
theorem evansCap_antitone {r tAct τ dt : ℝ} (hr : 0 ≤ r) (hτ : 0 < τ) (hdt : 0 ≤ dt) :
    Antitone (evansCap r tAct τ dt) := fun _ _ h =>
  mul_le_mul_of_nonneg_right (Sprinklers.decay_antitone hr hτ h) hdt

/-- `1 - exp(-x) ≥ x / (1 + x)` for `x ≥ 0`. -/
theorem one_sub_exp_neg_ge {x : ℝ} (hx : 0 ≤ x) : x / (1 + x) ≤ 1 - exp (-x) := by
  have hpos : 0 < 1 + x := by linarith
  have h1 : x + 1 ≤ exp x := Real.add_one_le_exp x
  have hE : exp (-x) * exp x = 1 := by rw [← Real.exp_add]; simp
  have h2 : exp (-x) ≤ 1 / (1 + x) := by
    rw [le_div_iff₀ hpos]; nlinarith [exp_pos (-x)]
  have h3 : x / (1 + x) = 1 - 1 / (1 + x) := by field_simp; ring
  linarith

/-- Evans's energy bound to the end: what `Sprinklers.evans_energy` bounds by
`Q dt / (1 - exp(-dt/τ))` is at most `Q (τ + dt)`. -/
theorem evans_energy_le {Q dt τ : ℝ} (hQ : 0 ≤ Q) (hdt : 0 < dt) (hτ : 0 < τ) (N : ℕ) :
    ∑ n ∈ Finset.range N, Sprinklers.decay Q 0 τ (n * dt) * dt ≤ Q * (τ + dt) := by
  refine (Sprinklers.evans_energy hQ hdt hτ N).trans ?_
  have hx : 0 < dt / τ := div_pos hdt hτ
  have h1 := one_sub_exp_neg_ge hx.le
  have hden : 0 < 1 - exp (-(dt / τ)) := lt_of_lt_of_le (div_pos hx (by linarith)) h1
  rw [div_le_iff₀ hden]
  have h2 : (τ + dt) * (dt / τ / (1 + dt / τ)) = dt := by
    have := hτ.ne'
    have : τ + dt ≠ 0 := by linarith
    field_simp
  have h3 : dt ≤ (τ + dt) * (1 - exp (-(dt / τ))) :=
    calc dt = (τ + dt) * (dt / τ / (1 + dt / τ)) := h2.symm
      _ ≤ (τ + dt) * (1 - exp (-(dt / τ))) :=
        mul_le_mul_of_nonneg_left h1 (by linarith : (0 : ℝ) ≤ τ + dt)
  nlinarith

/-- Summed over the steps from activation, `t = t_act + n dt`, the cap allows at most
`r (τ + dt)`, however long the run. -/
theorem evansCap_sum {r tAct τ dt : ℝ} (hr : 0 ≤ r) (hτ : 0 < τ) (hdt : 0 < dt) (N : ℕ) :
    ∑ n ∈ Finset.range N, evansCap r tAct τ dt (tAct + n * dt) ≤ r * (τ + dt) := by
  have h : ∀ n : ℕ, evansCap r tAct τ dt (tAct + n * dt) = Sprinklers.decay r 0 τ (n * dt) * dt := by
    intro n
    simp only [evansCap, Sprinklers.decay, add_sub_cancel_left, sub_zero]
  simp_rw [h]
  exact evans_energy_le hr hdt hτ N

/-! ## The wet surface (item 15) -/

/-- `Y_O2_AIR`, oxygen's mass fraction in air. -/
noncomputable def Y_O2_AIR : ℝ := 0.232

/-- The boiling point the moisture branch of `chemistry` uses, 373.15 K. -/
noncomputable def T_BOIL : ℝ := 373.15

/-- The water dried off a fuel cell's surface layer in `chemistry`: when `m_w > 0` and `T_s > T_B`,
`dw = min(m_w, C (T_s - T_B) / L)`, otherwise nothing. -/
noncomputable def dryWater (C L TB mw Ts : ℝ) : ℝ :=
  if 0 < mw ∧ TB < Ts then min mw (C * (Ts - TB) / L) else 0

/-- The surface temperature after it, `T_s - dw L / C`. -/
noncomputable def dryTs (C L TB mw Ts : ℝ) : ℝ := Ts - dryWater C L TB mw Ts * L / C

/-- (i) While water remains after the drying step the surface is no hotter than the boiling
point. -/
theorem wet_skin_pinned {C L TB mw Ts : ℝ} (hC : 0 < C) (hL : 0 < L)
    (hwet : 0 < mw - dryWater C L TB mw Ts) : dryTs C L TB mw Ts ≤ TB := by
  unfold dryTs
  unfold dryWater at hwet ⊢
  by_cases h : 0 < mw ∧ TB < Ts
  · rw [ite_eq_left h] at hwet ⊢
    have hm : min mw (C * (Ts - TB) / L) = C * (Ts - TB) / L := by
      rcases min_choice mw (C * (Ts - TB) / L) with h' | h'
      · rw [h'] at hwet; linarith
      · exact h'
    rw [hm]
    have := hC.ne'
    have := hL.ne'
    field_simp
    linarith
  · rw [ite_eq_right h, zero_mul, zero_div, sub_zero] at ⊢
    rw [ite_eq_right h] at hwet
    have : Ts ≤ TB := by
      by_contra hT
      exact h ⟨by linarith, by linarith⟩
    linarith

/-- (ii) Arrhenius at a surface no hotter than `T_B` is at most its value at `T_B`. -/
theorem arrhenius_le_boiling {A E T TB : ℝ} (hA : 0 < A) (hE : 0 < E) (hTB : 0 < TB)
    (hT : T ≤ TB) : arrhenius A E T ≤ arrhenius A E TB := by
  by_cases h : 0 < T
  · exact (FlamescapesPort.arrhenius_strictMonoOn hA hE).monotoneOn h hTB hT
  · have h0 : arrhenius A E T = 0 := by unfold arrhenius; exact ite_eq_right h
    rw [h0]; exact FlamescapesPort.arrhenius_nonneg hA.le TB

/-- The solid's oxidation in `chemistry`,
`dOx = min(skin (1 - survives(k(T_s) Y_O₂ / 0.232, dt)), m_O₂ / st.o2)`. -/
noncomputable def oxidised (A E Ts yO skin o2 so2 dt : ℝ) : ℝ :=
  min (skin * (1 - survives (arrhenius A E Ts * (yO / Y_O2_AIR)) dt)) (o2 / so2)

/-- (ii) While water remains on it, a cell's solid releases at most
`skin (1 - survives(k(T_B), dt)) heat`, and so at most `skin k(T_B) dt heat`. -/
theorem wet_skin_heat {A E C L TB mw Ts yO skin o2 so2 dt heat : ℝ} (hA : 0 < A) (hE : 0 < E)
    (hC : 0 < C) (hL : 0 < L) (hTB : 0 < TB) (hwet : 0 < mw - dryWater C L TB mw Ts)
    (hyO1 : yO ≤ Y_O2_AIR) (hskin : 0 ≤ skin) (hdt : 0 ≤ dt) (hheat : 0 ≤ heat) :
    oxidised A E (dryTs C L TB mw Ts) yO skin o2 so2 dt * heat ≤
        skin * (1 - survives (arrhenius A E TB) dt) * heat ∧
      skin * (1 - survives (arrhenius A E TB) dt) * heat ≤
        skin * (arrhenius A E TB * dt) * heat := by
  have hk := arrhenius_le_boiling hA hE hTB (wet_skin_pinned hC hL hwet)
  have hY : (0 : ℝ) < Y_O2_AIR := by norm_num [Y_O2_AIR]
  have hf1 : yO / Y_O2_AIR ≤ 1 := (div_le_one hY).2 hyO1
  have hk0 := FlamescapesPort.arrhenius_nonneg (E := E) hA.le (dryTs C L TB mw Ts)
  have hko : arrhenius A E (dryTs C L TB mw Ts) * (yO / Y_O2_AIR) ≤ arrhenius A E TB :=
    (mul_le_of_le_one_right hk0 hf1).trans hk
  have hs := survives_anti hdt hko
  constructor
  · apply mul_le_mul_of_nonneg_right _ hheat
    exact (min_le_left _ _).trans (mul_le_mul_of_nonneg_left (by linarith) hskin)
  · apply mul_le_mul_of_nonneg_right _ hheat
    apply mul_le_mul_of_nonneg_left _ hskin
    rw [FlamescapesPort.survives_of_nonneg (FlamescapesPort.arrhenius_nonneg hA.le TB)]
    have := Real.add_one_le_exp (-arrhenius A E TB * dt)
    linarith

/-- `DEFAULTS.oxidationA` (s⁻¹). -/
noncomputable def oxidationA : ℝ := 1.8e8
/-- `DEFAULTS.oxidationE` (J/mol). -/
noncomputable def oxidationE : ℝ := 9.22e4

/-- For the port's kinetics the oxidation constant at the boiling point is about 2e-5 per second
(2.23e-5): between 2e-5 and 2.5e-5. -/
theorem oxidation_at_boiling :
    2e-5 < arrhenius oxidationA oxidationE T_BOIL ∧
      arrhenius oxidationA oxidationE T_BOIL < 2.5e-5 := by
  have hT : (0 : ℝ) < T_BOIL := by norm_num [T_BOIL]
  have hR := FlamescapesPort.R_GAS_pos
  have hy1 : 29.7 < oxidationE / (FlamescapesPort.R_GAS * T_BOIL) := by
    rw [lt_div_iff₀ (mul_pos hR hT)]; norm_num [oxidationE, FlamescapesPort.R_GAS, T_BOIL]
  have hy2 : oxidationE / (FlamescapesPort.R_GAS * T_BOIL) < 29.72 := by
    rw [div_lt_iff₀ (mul_pos hR hT)]; norm_num [oxidationE, FlamescapesPort.R_GAS, T_BOIL]
  have he1 := Real.exp_one_gt_d9
  have he2 := Real.exp_one_lt_d9
  have hlow : 7.2e12 < exp 29.7 := by
    have e : exp 29.7 = exp 1 ^ 29 * exp 0.7 := by
      rw [← Real.exp_nat_mul, ← Real.exp_add]; norm_num
    have h07 : 1.945 ≤ exp 0.7 := by
      have := Real.quadratic_le_exp_of_nonneg (x := 0.7) (by norm_num)
      norm_num at this; linarith
    have hp : (2.7182818283 : ℝ) ^ 29 < exp 1 ^ 29 :=
      pow_lt_pow_left₀ he1 (by norm_num) (by norm_num)
    have hn : (7.2e12 : ℝ) < 2.7182818283 ^ 29 * 1.945 := by norm_num
    rw [e]
    nlinarith [exp_pos 0.7, pow_pos (exp_pos 1) 29]
  have hhigh : exp 29.72 < 9e12 := by
    have e : exp 29.72 = exp 1 ^ 29 * exp 0.09 ^ 8 := by
      rw [← Real.exp_nat_mul, ← Real.exp_nat_mul, ← Real.exp_add]; norm_num
    have h09 : exp 0.09 ≤ 1 / 0.91 := by
      have h := Real.add_one_le_exp (-0.09)
      have hE : exp 0.09 * exp (-0.09) = 1 := by rw [← Real.exp_add]; norm_num
      rw [le_div_iff₀ (by norm_num)]
      nlinarith [exp_pos 0.09]
    have hp : exp 1 ^ 29 < (2.7182818286 : ℝ) ^ 29 :=
      pow_lt_pow_left₀ he2 (exp_pos 1).le (by norm_num)
    have hp8 : exp 0.09 ^ 8 ≤ (1 / 0.91 : ℝ) ^ 8 := pow_le_pow_left₀ (exp_pos _).le h09 8
    have hn : (2.7182818286 : ℝ) ^ 29 * (1 / 0.91) ^ 8 < 9e12 := by norm_num
    rw [e]
    calc exp 1 ^ 29 * exp 0.09 ^ 8 ≤ exp 1 ^ 29 * (1 / 0.91 : ℝ) ^ 8 :=
          mul_le_mul_of_nonneg_left hp8 (pow_pos (exp_pos 1) 29).le
      _ < (2.7182818286 : ℝ) ^ 29 * (1 / 0.91) ^ 8 :=
          mul_lt_mul_of_pos_right hp (by norm_num)
      _ < 9e12 := hn
  have hE1 : exp 29.7 < exp (oxidationE / (FlamescapesPort.R_GAS * T_BOIL)) := exp_lt_exp.mpr hy1
  have hE2 : exp (oxidationE / (FlamescapesPort.R_GAS * T_BOIL)) < exp 29.72 := exp_lt_exp.mpr hy2
  unfold arrhenius
  rw [ite_eq_left hT, neg_div, exp_neg]
  generalize exp (oxidationE / (FlamescapesPort.R_GAS * T_BOIL)) = z at hE1 hE2 ⊢
  have hz : 0 < z := by linarith
  unfold oxidationA
  rw [← div_eq_mul_inv]
  constructor
  · rw [lt_div_iff₀ hz]; linarith
  · rw [div_lt_iff₀ hz]; linarith

/-- The pyrolysis multiplier of `chemistry`, `0.25 + 0.75 (1 - min(1, Y_O₂ / 0.232))`. -/
noncomputable def pyroFactor (yO : ℝ) : ℝ := 0.25 + 0.75 * (1 - min 1 (yO / Y_O2_AIR))

/-- (iii) Counterexample to "steam only slows the fire": the multiplier falls as oxygen rises,
strictly up to air's 0.232, from 1 to 0.25. Steam that dilutes the oxygen beside a hot, dry fuel
surface therefore raises its pyrolysis rate, up to four times. An interaction to test, not a
defect. -/
theorem pyrolysis_dilution_counterexample :
    Antitone pyroFactor ∧ StrictAntiOn pyroFactor (Set.Icc 0 Y_O2_AIR) ∧
      pyroFactor 0 = 4 * pyroFactor Y_O2_AIR ∧
      ∀ y y', 0 ≤ y' → pyroFactor y' ≤ 4 * pyroFactor y := by
  have hY : (0 : ℝ) < Y_O2_AIR := by norm_num [Y_O2_AIR]
  refine ⟨fun y y' h => ?_, fun y hy y' hy' h => ?_, ?_, fun y y' hy' => ?_⟩
  · unfold pyroFactor
    have : min 1 (y / Y_O2_AIR) ≤ min 1 (y' / Y_O2_AIR) :=
      min_le_min le_rfl (div_le_div_of_nonneg_right h hY.le)
    linarith
  · unfold pyroFactor
    rw [min_eq_right ((div_le_one hY).2 hy.2), min_eq_right ((div_le_one hY).2 hy'.2)]
    have : y / Y_O2_AIR < y' / Y_O2_AIR := div_lt_div_of_pos_right h hY
    linarith
  · unfold pyroFactor
    rw [zero_div, div_self hY.ne', min_self, min_eq_right zero_le_one]
    norm_num
  · unfold pyroFactor
    have h1 : min 1 (y / Y_O2_AIR) ≤ 1 := min_le_left _ _
    have h2 : 0 ≤ min 1 (y' / Y_O2_AIR) := le_min zero_le_one (div_nonneg hy' hY.le)
    linarith

/-- The counterexample through the port's own steps: steam added beside a fuel surface whose gas
holds at most air's oxygen raises the pyrolysis constant strictly. -/
theorem pyrolysis_rises_with_steam {A E Ts V rho dv : ℝ} {Y : Sp → ℝ} (hA : 0 < A)
    (hTs : 0 < Ts) (hV : 0 < V) (hrho : 0 < rho) (hY : ∀ q, 0 ≤ Y q) (hN : ∑ q, Y q ≤ 1)
    (hdv : 0 < dv) (hO : 0 < Y Sp.o2) (hOair : Y Sp.o2 ≤ Y_O2_AIR) :
    arrhenius A E Ts * pyroFactor (Y Sp.o2) <
      arrhenius A E Ts * pyroFactor ((addVapour V rho Y dv).1 Sp.o2) := by
  obtain ⟨ho, -, hm, hY', -⟩ := vapour_changeGas hV hrho hY hN hdv.le
  have hmass : 0 < rho * V := mul_pos hrho hV
  have hlt : (addVapour V rho Y dv).1 Sp.o2 < Y Sp.o2 := by
    have h := ho Sp.o2 (by decide)
    rw [hm] at h
    by_contra hcon
    push Not at hcon
    nlinarith [mul_le_mul_of_nonneg_right hcon (by linarith : (0 : ℝ) ≤ rho * V + dv),
      mul_pos hO hdv]
  apply mul_lt_mul_of_pos_left _ (FlamescapesPort.arrhenius_pos hA hTs)
  exact pyrolysis_dilution_counterexample.2.1 ⟨hY' _, hlt.le.trans hOair⟩ ⟨hO.le, hOair⟩ hlt

end FireEgress.Suppression
