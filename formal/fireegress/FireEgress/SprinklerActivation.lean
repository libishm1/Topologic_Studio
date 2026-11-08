import Mathlib
import FireEgress.FlamescapesPort
import FireEgress.Sprinklers

/-!
# Sprinkler activation: the wall coefficient, Alpert's estimate and the head's distance

Studio's sprinkler heads open when their link, stepped exactly against a gas reading
(`activationFrame` in `topologicpy-web-frontend/src/core/suppression.js`), reaches its activation
temperature. The fix of 2026-11-08 changed two inputs of that reading. In `conduct`
(`flamescapes.js`) an inert wall now exchanges heat with the gas beside it at its own coefficient,
lower than the fuel's, and in `sprinklerSetup` a head's distance from the fire is measured from the
head's exact position to the fire cell's centre rather than between two cell centres. This file
proves, over real arithmetic, what the fix relies on. It reuses `FlamescapesPort.exchange`,
`afterA` and `afterB` for the wall exchange and `Sprinklers.linkAfter` for the link.

## Which theorem covers what

* The wall exchange (`conduct`, with the wall as body a). `decay_le_of_coeff`;
  `wall_coeff_one_exchange`: a smaller coefficient never cools the gas more in one exchange, and
  the wall ends no hotter.
* A lumped ceiling-jet cell over a wall (`cellStep`, `cellRun`, `gasRun`). `cellRun_energy`,
  `cellRun_diff`, `cellRun_compare`, `wall_coeff_gas_hotter`: under the smaller coefficient the gas
  is at least as hot at every step, for any non-negative source history.
* The link (`activationFrame` with `linkStep`). `wall_coeff_link`,
  `wall_coeff_activates_no_later`, `wall_coeff_activates_exact`, `wall_coeff_first_activation`: a
  head opens no later under the smaller wall coefficient.
* `alpert`. `alpert_switch_cube`, `alpert_switch`, `alpertDT_nonneg`, `alpertDT_mono_Q`,
  `alpertDT_anti_H`, `alpertDT_H_switch`, `alpertDT_far_at_switch`, `alpertDT_anti_r`;
  `alpertU_far_at_switch`, `alpertU_jump`, `alpertU_not_antitone`.
* The estimate reading `max(Tg, layer + dT)`. `estimateGas_mono`, `estimate_activates_no_later`;
  the speed's part in the link: `linkRun`, `linkRun_mono`, `faster_link_can_open_later`.
* The head's distance (`sprinklerSetup`). `abs_sub_cellCentre_le`, `hypot_eq_dist`,
  `dist_snap_le`, `old_distance_error`, `new_distance_error`, `distance_example`.
* Helpers: `rpow_two_thirds_cube`, `rpow_five_sixths_pow`, `pos_of_far`, `hypot_self`.

## What the proofs found

* Alpert's speed jumps up by about 1.09 %, not by less than 1 %, where `r` passes `0.15 H`: the
  ratio `0.197 / (0.947 · 0.15^(5/6))` lies between 1.01 and 1.011 (`alpertU_jump`), at every `Q`
  and `H` (`alpertU_far_at_switch`). The speed is therefore not antitone in `r`
  (`alpertU_not_antitone`). The excess temperature falls, by about 0.14 %, where `r` passes
  `0.18 H` (`alpert_switch`), so it is antitone in `r` over the whole range (`alpertDT_anti_r`),
  but for the same reason it is not antitone in `H` across that switch (`alpertDT_H_switch`).
* "More heat never opens a head later" holds with the link's weights held fixed
  (`estimate_activates_no_later`). The estimate also raises the speed with `Q`
  (`Math.max(ug, jet.U)`), and a faster link cools faster whenever the gas is cooler than the
  link, so a run that is never cooler and never slower can open later
  (`faster_link_can_open_later`). It cannot while the slower run's link has not passed its gas,
  the heating phase (`linkRun_mono`).

## What is not covered

* Floating point (the solver keeps `Float32Array` fields; the link is stepped in doubles).
* The cell model is lumped: one gas cell over one wall cell, with a source history that is the
  same in both runs. In the solver the source (advection, buoyancy, radiation and the plume) itself
  depends on the gas temperatures, so the comparison of two whole simulations is not proved.
* The layer's mean over the heads in sight and the frame mean `0.5 (prevT + T)` are monotone
  operations; the theorems take the reading's history as given. The crossing time inside a frame
  is not modelled: activation is read at the frame's end, as `linkAfter` does.
-/

namespace FireEgress.SprinklerActivation

open Real

/-! ## One exchange between a wall and the gas beside it -/

/-- The exchange's decay factor `exp (-G dt / μ)` never falls when the coefficient falls: for
`G' ≤ G` and `dt ≥ 0`, `exp (-G dt / μ) ≤ exp (-G' dt / μ)`. A smaller coefficient leaves more of
the wall-to-gas difference in place after a step. -/
lemma decay_le_of_coeff {Cw Cg G G' dt : ℝ} (hw : 0 < Cw) (hg : 0 < Cg) (hdt : 0 ≤ dt)
    (hG : G' ≤ G) :
    exp (-G * dt / FlamescapesPort.mu Cw Cg) ≤ exp (-G' * dt / FlamescapesPort.mu Cw Cg) := by
  have hm := FlamescapesPort.mu_pos hw hg
  rw [exp_le_exp]
  apply div_le_div_of_nonneg_right _ hm.le
  nlinarith

/-- Item 1. A smaller wall coefficient never cools the gas more in one exchange. For a wall no
hotter than the gas, positive capacities, `dt ≥ 0` and `0 < G' ≤ G`, the gas after `conduct`'s
exchange (`afterB`, the wall as body a) is at least as hot under `G'` as under `G`, and the wall is
no hotter. In the app, lowering the wall's coefficient `P.wallConvection` below the fuel's can
only leave the gas at a ceiling hotter, never cooler, after a step. -/
theorem wall_coeff_one_exchange {Tw Tg Cw Cg G G' dt : ℝ} (hw : 0 < Cw) (hg : 0 < Cg)
    (hdt : 0 ≤ dt) (hG' : 0 < G') (hG : G' ≤ G) (hT : Tw ≤ Tg) :
    FlamescapesPort.afterB Tw Tg Cw Cg G dt ≤ FlamescapesPort.afterB Tw Tg Cw Cg G' dt ∧
      FlamescapesPort.afterA Tw Tg Cw Cg G' dt ≤ FlamescapesPort.afterA Tw Tg Cw Cg G dt := by
  have hG0 : 0 < G := lt_of_lt_of_le hG' hG
  have hm := FlamescapesPort.mu_pos hw hg
  have he := decay_le_of_coeff hw hg hdt hG
  unfold FlamescapesPort.afterA FlamescapesPort.afterB
  rw [FlamescapesPort.exchange_of_pos hw hg hG0, FlamescapesPort.exchange_of_pos hw hg hG']
  have hq : (Tw - Tg) * FlamescapesPort.mu Cw Cg * (1 - exp (-G * dt / FlamescapesPort.mu Cw Cg)) ≤
      (Tw - Tg) * FlamescapesPort.mu Cw Cg * (1 - exp (-G' * dt / FlamescapesPort.mu Cw Cg)) := by
    apply mul_le_mul_of_nonpos_left _ (by nlinarith)
    linarith
  constructor
  · have := div_le_div_of_nonneg_right hq hg.le; linarith
  · have := div_le_div_of_nonneg_right hq hw.le; linarith

/-! ## A lumped ceiling-jet cell over a wall -/

/-- One step of a lumped ceiling-jet cell over an inert wall, state `(Tw, Tg)`: the gas first
gains the source `s` (kelvin), then exchanges with the wall as `conduct` does. -/
noncomputable def cellStep (Cw Cg G dt s : ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  (FlamescapesPort.afterA p.1 (p.2 + s) Cw Cg G dt,
    FlamescapesPort.afterB p.1 (p.2 + s) Cw Cg G dt)

/-- The cell after `n` steps of the source history `s` from the state `p0`. -/
noncomputable def cellRun (Cw Cg G dt : ℝ) (s : ℕ → ℝ) (p0 : ℝ × ℝ) : ℕ → ℝ × ℝ
  | 0 => p0
  | n + 1 => cellStep Cw Cg G dt (s n) (cellRun Cw Cg G dt s p0 n)

/-- The cell's gas temperature after `n` steps. -/
noncomputable def gasRun (Cw Cg G dt : ℝ) (s : ℕ → ℝ) (p0 : ℝ × ℝ) (n : ℕ) : ℝ :=
  (cellRun Cw Cg G dt s p0 n).2

/-- The energy `Cw Tw + Cg Tg` after `n` steps is the initial energy plus the sources, whatever
the coefficient. Two runs that differ only in the wall coefficient hold the same energy at every
step; the coefficient only moves heat between the gas and the wall. -/
theorem cellRun_energy (Cw Cg G dt : ℝ) (s : ℕ → ℝ) (p0 : ℝ × ℝ) :
    ∀ n, Cw * (cellRun Cw Cg G dt s p0 n).1 + Cg * (cellRun Cw Cg G dt s p0 n).2 =
      Cw * p0.1 + Cg * p0.2 + Cg * ∑ i ∈ Finset.range n, s i
  | 0 => by simp [cellRun]
  | n + 1 => by
      have ih := cellRun_energy Cw Cg G dt s p0 n
      simp only [cellRun, cellStep]
      rw [FlamescapesPort.exchange_energy, Finset.sum_range_succ]
      linarith

/-- The gas-to-wall difference after a step is the difference after the source, decayed by
`exp (-G dt / μ)` (`exchange_diff`). -/
theorem cellRun_diff {Cw Cg G dt : ℝ} (hw : 0 < Cw) (hg : 0 < Cg) (hG : 0 < G) (s : ℕ → ℝ)
    (p0 : ℝ × ℝ) (n : ℕ) :
    (cellRun Cw Cg G dt s p0 (n + 1)).2 - (cellRun Cw Cg G dt s p0 (n + 1)).1 =
      ((cellRun Cw Cg G dt s p0 n).2 - (cellRun Cw Cg G dt s p0 n).1 + s n) *
        exp (-G * dt / FlamescapesPort.mu Cw Cg) := by
  simp only [cellRun, cellStep]
  have := FlamescapesPort.exchange_diff (Ta := (cellRun Cw Cg G dt s p0 n).1)
    (Tb := (cellRun Cw Cg G dt s p0 n).2 + s n) hw hg hG dt
  linear_combination (-1 : ℝ) * this

/-- Starting with the gas no colder than the wall, and with non-negative sources, the gas stays
no colder than the wall, and its lead over the wall under the smaller coefficient is at least the
lead under the larger one, at every step. -/
theorem cellRun_compare {Cw Cg G G' dt : ℝ} (hw : 0 < Cw) (hg : 0 < Cg) (hdt : 0 ≤ dt)
    (hG' : 0 < G') (hG : G' ≤ G) {s : ℕ → ℝ} (hs : ∀ n, 0 ≤ s n) {p0 : ℝ × ℝ}
    (h0 : p0.1 ≤ p0.2) :
    ∀ n, 0 ≤ (cellRun Cw Cg G dt s p0 n).2 - (cellRun Cw Cg G dt s p0 n).1 ∧
      (cellRun Cw Cg G dt s p0 n).2 - (cellRun Cw Cg G dt s p0 n).1 ≤
        (cellRun Cw Cg G' dt s p0 n).2 - (cellRun Cw Cg G' dt s p0 n).1
  | 0 => ⟨by show 0 ≤ p0.2 - p0.1; linarith, le_rfl⟩
  | n + 1 => by
      obtain ⟨ih0, ih1⟩ := cellRun_compare hw hg hdt hG' hG hs h0 n
      have hG0 : 0 < G := lt_of_lt_of_le hG' hG
      rw [cellRun_diff hw hg hG0, cellRun_diff hw hg hG']
      have he := decay_le_of_coeff hw hg hdt hG
      have hpos := exp_pos (-G * dt / FlamescapesPort.mu Cw Cg)
      have hsn := hs n
      exact ⟨mul_nonneg (by linarith) hpos.le, mul_le_mul (by linarith) he hpos.le (by linarith)⟩

/-- Item 2. In a lumped ceiling-jet cell over a wall, where each step the gas first gains a
source `s n ≥ 0` and then exchanges with the wall, two runs from the same state with the gas no
colder than the wall keep the gas under the smaller coefficient at least as hot, and the wall no
hotter, at every step, for any non-negative source history. The energy is the same in both runs
(`cellRun_energy`) and the gas temperature at fixed energy rises with the gas-to-wall difference,
which the smaller coefficient keeps larger (`cellRun_compare`). In the app, the gas at a sprinkler
under a ceiling is never cooler after the wall coefficient is lowered, for the same heat input. -/
theorem wall_coeff_gas_hotter {Cw Cg G G' dt : ℝ} (hw : 0 < Cw) (hg : 0 < Cg) (hdt : 0 ≤ dt)
    (hG' : 0 < G') (hG : G' ≤ G) {s : ℕ → ℝ} (hs : ∀ n, 0 ≤ s n) {p0 : ℝ × ℝ}
    (h0 : p0.1 ≤ p0.2) (n : ℕ) :
    gasRun Cw Cg G dt s p0 n ≤ gasRun Cw Cg G' dt s p0 n ∧
      (cellRun Cw Cg G' dt s p0 n).1 ≤ (cellRun Cw Cg G dt s p0 n).1 := by
  unfold gasRun
  obtain ⟨-, hD⟩ := cellRun_compare hw hg hdt hG' hG hs h0 n
  have hE := cellRun_energy Cw Cg G dt s p0 n
  have hE' := cellRun_energy Cw Cg G' dt s p0 n
  have hC : 0 < Cw + Cg := by linarith
  have hmul := mul_le_mul_of_nonneg_left hD hw.le
  have hmul' := mul_le_mul_of_nonneg_left hD hg.le
  constructor
  · apply le_of_mul_le_mul_left _ hC
    linear_combination hE - hE' + hmul
  · apply le_of_mul_le_mul_left _ hC
    linear_combination hE' - hE + hmul'

/-! ## The link under the two coefficients -/

/-- The link driven by the cell's gas, for any weights with `k ≥ 0` and `k + c ≤ 1`, is at least
as hot at every step under the smaller wall coefficient (item 2 with `linkAfter_mono`). -/
theorem wall_coeff_link {Cw Cg G G' dt k c Tm T0 : ℝ} (hw : 0 < Cw) (hg : 0 < Cg)
    (hdt : 0 ≤ dt) (hG' : 0 < G') (hG : G' ≤ G) {s : ℕ → ℝ} (hs : ∀ n, 0 ≤ s n)
    {p0 : ℝ × ℝ} (h0 : p0.1 ≤ p0.2) (hk : 0 ≤ k) (hkc : k + c ≤ 1) (n : ℕ) :
    Sprinklers.linkAfter k c Tm T0 (gasRun Cw Cg G dt s p0) n ≤
      Sprinklers.linkAfter k c Tm T0 (gasRun Cw Cg G' dt s p0) n :=
  Sprinklers.linkAfter_mono hk hkc (fun m => (wall_coeff_gas_hotter hw hg hdt hG' hG hs h0 m).1) n

/-- Item 3. A link driven by the cell's gas reaches its activation temperature `Ta` no later under
the smaller wall coefficient: whenever it has under `G`, it has under `G'` (item 2 composed with
`activates_no_later`). In the app, lowering the wall coefficient never delays a head. -/
theorem wall_coeff_activates_no_later {Cw Cg G G' dt k c Tm T0 Ta : ℝ} (hw : 0 < Cw)
    (hg : 0 < Cg) (hdt : 0 ≤ dt) (hG' : 0 < G') (hG : G' ≤ G) {s : ℕ → ℝ}
    (hs : ∀ n, 0 ≤ s n) {p0 : ℝ × ℝ} (h0 : p0.1 ≤ p0.2) (hk : 0 ≤ k) (hkc : k + c ≤ 1) {n : ℕ}
    (h : Ta ≤ Sprinklers.linkAfter k c Tm T0 (gasRun Cw Cg G dt s p0) n) :
    Ta ≤ Sprinklers.linkAfter k c Tm T0 (gasRun Cw Cg G' dt s p0) n :=
  Sprinklers.activates_no_later hk hkc
    (fun m => (wall_coeff_gas_hotter hw hg hdt hG' hG hs h0 m).1) h

/-- Item 3 with the exact steps `activationFrame` takes: with `a = √u / RTI ≥ 0`,
`b = C / RTI ≥ 0`, `a + b > 0` and a frame `τ ≥ 0`, each step is `exactStep`
(`exactStep_eq_linkStep`) and its weights meet `exactStep_weights`, so the head opens no later
under the smaller wall coefficient, for any frame length. -/
theorem wall_coeff_activates_exact {Cw Cg G G' dt a b τ Tm T0 Ta : ℝ} (hw : 0 < Cw)
    (hg : 0 < Cg) (hdt : 0 ≤ dt) (hG' : 0 < G') (hG : G' ≤ G) {s : ℕ → ℝ}
    (hs : ∀ n, 0 ≤ s n) {p0 : ℝ × ℝ} (h0 : p0.1 ≤ p0.2) (ha : 0 ≤ a) (hb : 0 ≤ b)
    (hab : 0 < a + b) (hτ : 0 ≤ τ) {n : ℕ}
    (h : Ta ≤ Sprinklers.linkAfter ((1 - exp (-((a + b) * τ))) * a / (a + b))
      ((1 - exp (-((a + b) * τ))) * b / (a + b)) Tm T0 (gasRun Cw Cg G dt s p0) n) :
    Ta ≤ Sprinklers.linkAfter ((1 - exp (-((a + b) * τ))) * a / (a + b))
      ((1 - exp (-((a + b) * τ))) * b / (a + b)) Tm T0 (gasRun Cw Cg G' dt s p0) n := by
  obtain ⟨hk, -, hkc⟩ := Sprinklers.exactStep_weights ha hb hab hτ
  exact wall_coeff_activates_no_later hw hg hdt hG' hG hs h0 hk hkc h

/-- Item 3 as a first opening step: if the head ever opens under `G`, it opens under `G'`, at a
step no later than the first under `G`. -/
theorem wall_coeff_first_activation {Cw Cg G G' dt k c Tm T0 Ta : ℝ} (hw : 0 < Cw)
    (hg : 0 < Cg) (hdt : 0 ≤ dt) (hG' : 0 < G') (hG : G' ≤ G) {s : ℕ → ℝ}
    (hs : ∀ n, 0 ≤ s n) {p0 : ℝ × ℝ} (h0 : p0.1 ≤ p0.2) (hk : 0 ≤ k) (hkc : k + c ≤ 1)
    (h : ∃ n, Ta ≤ Sprinklers.linkAfter k c Tm T0 (gasRun Cw Cg G dt s p0) n) :
    ∃ h' : (∃ n, Ta ≤ Sprinklers.linkAfter k c Tm T0 (gasRun Cw Cg G' dt s p0) n),
      Nat.find h' ≤ Nat.find h := by
  have hl := wall_coeff_activates_no_later hw hg hdt hG' hG hs h0 hk hkc (Nat.find_spec h)
  exact ⟨⟨Nat.find h, hl⟩, Nat.find_min' _ hl⟩

/-! ## Alpert's ceiling jet -/

/-- `alpert(Q, H, r).dT` of `suppression.js` (Q in kW, H and r in m): Alpert's excess temperature
of the ceiling jet, 0 unless `Q > 0` and `H > 0`. -/
noncomputable def alpertDT (Q H r : ℝ) : ℝ :=
  if 0 < Q ∧ 0 < H then
    (if r / H ≤ 0.18 then 16.9 * Q ^ ((2 : ℝ) / 3) / H ^ ((5 : ℝ) / 3)
      else 5.38 * (Q / r) ^ ((2 : ℝ) / 3) / H)
  else 0

/-- `alpert(Q, H, r).U`: Alpert's ceiling-jet speed (m/s), 0 unless `Q > 0` and `H > 0`. -/
noncomputable def alpertU (Q H r : ℝ) : ℝ :=
  if 0 < Q ∧ 0 < H then
    (if r / H ≤ 0.15 then 0.947 * (Q / H) ^ ((1 : ℝ) / 3)
      else 0.197 * Q ^ ((1 : ℝ) / 3) * H ^ (0.5 : ℝ) / r ^ ((5 : ℝ) / 6))
  else 0

lemma rpow_two_thirds_cube {x : ℝ} (hx : 0 ≤ x) : (x ^ ((2 : ℝ) / 3)) ^ 3 = x ^ 2 := by
  rw [← rpow_natCast, ← rpow_mul hx,
    show (2 : ℝ) / 3 * ((3 : ℕ) : ℝ) = ((2 : ℕ) : ℝ) by norm_num, rpow_natCast]

lemma rpow_five_sixths_pow {x : ℝ} (hx : 0 ≤ x) : (x ^ ((5 : ℝ) / 6)) ^ 6 = x ^ 5 := by
  rw [← rpow_natCast, ← rpow_mul hx,
    show (5 : ℝ) / 6 * ((6 : ℕ) : ℝ) = ((5 : ℕ) : ℝ) by norm_num, rpow_natCast]

/-- The constants of the two branches, cubed: `(5.38 / 16.9)³ ≤ 0.18²`. -/
theorem alpert_switch_cube : (5.38 / 16.9 : ℝ) ^ 3 ≤ 0.18 ^ 2 := by norm_num

/-- At `r = 0.18 H` the far branch gives `5.38 / 0.18^(2/3) ≤ 16.9` times `Q^(2/3) / H^(5/3)`,
the near branch's value: Alpert's excess falls, by about 0.14 %, as `r` passes the switch. This is
what keeps the estimate's temperature from rising with distance there. -/
theorem alpert_switch : 5.38 / (0.18 : ℝ) ^ ((2 : ℝ) / 3) ≤ 16.9 := by
  have hx : 0 < (0.18 : ℝ) ^ ((2 : ℝ) / 3) := rpow_pos_of_pos (by norm_num) _
  rw [div_le_iff₀ hx]
  by_contra hlt
  have h1 : (0.18 : ℝ) ^ ((2 : ℝ) / 3) < 5.38 / 16.9 := by
    rw [lt_div_iff₀ (by norm_num)]; linarith [not_le.mp hlt]
  have h2 := pow_lt_pow_left₀ h1 hx.le (by norm_num : (3 : ℕ) ≠ 0)
  rw [rpow_two_thirds_cube (by norm_num)] at h2
  linarith [alpert_switch_cube]

/-- Alpert's excess is never negative, so the estimate never reads below the layer. -/
theorem alpertDT_nonneg (Q H r : ℝ) : 0 ≤ alpertDT Q H r := by
  unfold alpertDT
  split_ifs with h1 h2
  · exact div_nonneg (mul_nonneg (by norm_num) (rpow_nonneg h1.1.le _))
      (rpow_nonneg h1.2.le _)
  · have hr : 0 < r := by
      have := mul_pos (show (0 : ℝ) < r / H by linarith [not_le.mp h2]) h1.2
      rwa [div_mul_cancel₀ r h1.2.ne'] at this
    exact div_nonneg (mul_nonneg (by norm_num) (rpow_nonneg (div_nonneg h1.1.le hr.le) _))
      h1.2.le
  · exact le_rfl

/-- In the far branch the head is beyond the plume: `b < r / H` with `b ≥ 0` and `H > 0` gives
`r > 0`. -/
lemma pos_of_far {r H b : ℝ} (hH : 0 < H) (hb : 0 ≤ b) (h : b < r / H) : 0 < r := by
  have := mul_pos (show (0 : ℝ) < r / H by linarith) hH
  rwa [div_mul_cancel₀ r hH.ne'] at this

/-- Item 4. Alpert's excess never falls when the heat release rises, at any ceiling height and
distance. In the app, a larger fire never lowers the estimated gas at a head. -/
theorem alpertDT_mono_Q (H r : ℝ) : Monotone (fun Q => alpertDT Q H r) := by
  intro Q Q' hQ
  simp only
  by_cases hq : 0 < Q ∧ 0 < H
  · have hq' : 0 < Q' ∧ 0 < H := ⟨lt_of_lt_of_le hq.1 hQ, hq.2⟩
    unfold alpertDT
    rw [ite_eq_left hq, ite_eq_left hq']
    split_ifs with hb
    · apply div_le_div_of_nonneg_right _ (rpow_nonneg hq.2.le _)
      exact mul_le_mul_of_nonneg_left (rpow_le_rpow hq.1.le hQ (by norm_num)) (by norm_num)
    · have hr : 0 < r := pos_of_far hq.2 (by norm_num) (not_le.mp hb)
      apply div_le_div_of_nonneg_right _ hq.2.le
      apply mul_le_mul_of_nonneg_left _ (by norm_num)
      exact rpow_le_rpow (div_nonneg hq.1.le hr.le) (div_le_div_of_nonneg_right hQ hr.le)
        (by norm_num)
  · have : alpertDT Q H r = 0 := by unfold alpertDT; rw [ite_eq_right hq]
    rw [this]; exact alpertDT_nonneg Q' H r

/-- Item 4. Alpert's excess never rises with the ceiling height `H` while the head stays in one
branch: `r / H ≤ 0.18` (near the plume, which then holds at `H'` too) or `0.18 < r / H'` (far from
it, which then holds at `H` too). -/
theorem alpertDT_anti_H {Q H H' r : ℝ} (hH : 0 < H) (hHH : H ≤ H')
    (hb : r / H ≤ 0.18 ∨ 0.18 < r / H') : alpertDT Q H' r ≤ alpertDT Q H r := by
  have hH' : 0 < H' := lt_of_lt_of_le hH hHH
  by_cases hQ : 0 < Q
  · unfold alpertDT
    have hq : 0 < Q ∧ 0 < H := ⟨hQ, hH⟩
    have hq' : 0 < Q ∧ 0 < H' := ⟨hQ, hH'⟩
    rw [ite_eq_left hq, ite_eq_left hq']
    rcases hb with hb | hb
    · have hb' : r / H' ≤ 0.18 := by
        rcases le_or_gt r 0 with hr | hr
        · have : r / H' ≤ 0 := div_nonpos_of_nonpos_of_nonneg hr hH'.le
          linarith
        · have : r / H' ≤ r / H := div_le_div_of_nonneg_left hr.le hH hHH
          linarith
      rw [ite_eq_left hb, ite_eq_left hb']
      exact div_le_div_of_nonneg_left (mul_nonneg (by norm_num) (rpow_nonneg hQ.le _))
        (rpow_pos_of_pos hH _) (rpow_le_rpow hH.le hHH (by norm_num))
    · have hr : 0 < r := pos_of_far hH' (by norm_num) hb
      have hb0 : 0.18 < r / H := lt_of_lt_of_le hb (div_le_div_of_nonneg_left hr.le hH hHH)
      rw [ite_eq_right (not_le.mpr hb0), ite_eq_right (not_le.mpr hb)]
      exact div_le_div_of_nonneg_left
        (mul_nonneg (by norm_num) (rpow_nonneg (div_nonneg hQ.le hr.le) _)) hH hHH
  · unfold alpertDT
    rw [ite_eq_right (show ¬ (0 < Q ∧ 0 < H) from fun h => hQ h.1),
      ite_eq_right (show ¬ (0 < Q ∧ 0 < H') from fun h => hQ h.1)]

/-- Across the switch Alpert's excess is not antitone in `H`: at `Q = 1` kW and `r = 0.18` m, a
ceiling 0.999 m above the fire gives a smaller excess than one 1 m above it, since the head moves
from the far branch into the near one. -/
theorem alpertDT_H_switch : alpertDT 1 0.999 0.18 < alpertDT 1 1 0.18 := by
  unfold alpertDT
  have hc1 : ¬ ((0.18 : ℝ) / 0.999 ≤ 0.18) := by norm_num
  have hc2 : (0.18 : ℝ) / 1 ≤ 0.18 := by norm_num
  have hp1 : (0 : ℝ) < 1 ∧ (0 : ℝ) < 0.999 := by norm_num
  have hp2 : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 := by norm_num
  rw [ite_eq_left hp1, ite_eq_left hp2, ite_eq_right hc1,
    ite_eq_left hc2]
  simp only [one_rpow]
  have hy0 : 0 ≤ ((1 : ℝ) / 0.18) ^ ((2 : ℝ) / 3) := rpow_nonneg (by norm_num) _
  have hy3 := rpow_two_thirds_cube (show (0 : ℝ) ≤ 1 / 0.18 by norm_num)
  have hy : ((1 : ℝ) / 0.18) ^ ((2 : ℝ) / 3) < 16.9 * 0.999 / 5.38 := by
    by_contra h
    have := pow_le_pow_left₀ (by norm_num) (not_lt.mp h) 3
    rw [hy3] at this
    norm_num at this
  rw [div_lt_iff₀ (by norm_num)]
  linarith

/-- At the switch the far branch is the near branch scaled by `5.38 / 0.18^(2/3) / 16.9`, at every
`Q` and `H`: `5.38 (Q / (0.18 H))^(2/3) / H = (5.38 / 0.18^(2/3)) Q^(2/3) / H^(5/3)`. -/
theorem alpertDT_far_at_switch {Q H : ℝ} (hQ : 0 ≤ Q) (hH : 0 < H) :
    5.38 * (Q / (0.18 * H)) ^ ((2 : ℝ) / 3) / H =
      5.38 / (0.18 : ℝ) ^ ((2 : ℝ) / 3) * (Q ^ ((2 : ℝ) / 3) / H ^ ((5 : ℝ) / 3)) := by
  have h53 : H ^ ((5 : ℝ) / 3) = H ^ ((2 : ℝ) / 3) * H := by
    rw [show (5 : ℝ) / 3 = 2 / 3 + 1 by norm_num, rpow_add_one hH.ne']
  rw [div_rpow hQ (by positivity), mul_rpow (by norm_num) hH.le, h53]
  have h1 : 0 < (0.18 : ℝ) ^ ((2 : ℝ) / 3) := rpow_pos_of_pos (by norm_num) _
  have h2 : 0 < H ^ ((2 : ℝ) / 3) := rpow_pos_of_pos hH _
  field_simp

/-- Item 4. Alpert's excess never rises with the distance `r`, over the whole range and across
the switch at `r = 0.18 H`, where it falls by 0.14 % (`alpert_switch`). In the app, a head
measured nearer the fire never gets a cooler estimate. -/
theorem alpertDT_anti_r (Q H : ℝ) : Antitone (alpertDT Q H) := by
  intro r r' hr
  by_cases h : 0 < Q ∧ 0 < H
  · obtain ⟨hQ, hH⟩ := h
    unfold alpertDT
    have hq : 0 < Q ∧ 0 < H := ⟨hQ, hH⟩
    rw [ite_eq_left hq, ite_eq_left hq]
    by_cases h1 : r' / H ≤ 0.18
    · have h0 : r / H ≤ 0.18 := le_trans (div_le_div_of_nonneg_right hr hH.le) h1
      rw [ite_eq_left h0, ite_eq_left h1]
    · rw [ite_eq_right h1]
      have hr' : 0 < r' := pos_of_far hH (by norm_num) (not_le.mp h1)
      have hfar : ∀ ρ, 0.18 * H ≤ ρ → ρ ≤ r' →
          5.38 * (Q / r') ^ ((2 : ℝ) / 3) / H ≤ 5.38 * (Q / ρ) ^ ((2 : ℝ) / 3) / H := by
        intro ρ hρ hρr
        have hρ0 : 0 < ρ := lt_of_lt_of_le (by positivity) hρ
        apply div_le_div_of_nonneg_right _ hH.le
        apply mul_le_mul_of_nonneg_left _ (by norm_num)
        exact rpow_le_rpow (div_nonneg hQ.le hr'.le) (div_le_div_of_nonneg_left hQ.le hρ0 hρr)
          (by norm_num)
      by_cases h0 : r / H ≤ 0.18
      · rw [ite_eq_left h0]
        have hsw : 0.18 * H ≤ r' := by
          have := not_le.mp h1
          rw [lt_div_iff₀ hH] at this; linarith
        refine le_trans (hfar (0.18 * H) le_rfl hsw) ?_
        rw [alpertDT_far_at_switch hQ.le hH, mul_div_assoc]
        exact mul_le_mul_of_nonneg_right alpert_switch
          (div_nonneg (rpow_nonneg hQ.le _) (rpow_nonneg hH.le _))
      · rw [ite_eq_right h0]
        have hrpos : 0 < r := pos_of_far hH (by norm_num) (not_le.mp h0)
        have hsw : 0.18 * H ≤ r := by
          have := not_le.mp h0
          rw [lt_div_iff₀ hH] at this; linarith
        exact hfar r hsw hr
  · unfold alpertDT; rw [ite_eq_right h, ite_eq_right h]

/-- At the switch `r = 0.15 H` the speed's far branch is the near branch scaled by
`0.197 / 0.15^(5/6) / 0.947`, at every `Q` and `H`. -/
theorem alpertU_far_at_switch {Q H : ℝ} (hQ : 0 ≤ Q) (hH : 0 < H) :
    0.197 * Q ^ ((1 : ℝ) / 3) * H ^ (0.5 : ℝ) / (0.15 * H) ^ ((5 : ℝ) / 6) =
      0.197 / (0.15 : ℝ) ^ ((5 : ℝ) / 6) * (Q / H) ^ ((1 : ℝ) / 3) := by
  have h56 : H ^ ((5 : ℝ) / 6) = H ^ (0.5 : ℝ) * H ^ ((1 : ℝ) / 3) := by
    rw [← rpow_add hH]; norm_num
  rw [mul_rpow (by norm_num) hH.le, div_rpow hQ hH.le, h56]
  have h1 : 0 < (0.15 : ℝ) ^ ((5 : ℝ) / 6) := rpow_pos_of_pos (by norm_num) _
  have h2 : 0 < H ^ ((1 : ℝ) / 3) := rpow_pos_of_pos hH _
  have h3 : 0 < H ^ (0.5 : ℝ) := rpow_pos_of_pos hH _
  field_simp

/-- Item 4, the speed. Where `r` passes `0.15 H` Alpert's speed jumps up by the factor
`0.197 / (0.947 · 0.15^(5/6))`, which lies between 1.01 and 1.011 (1.0109): the jump is 1.09 %,
so the claim that it lies within 1 % above 0.947 is false, and the bound that holds is 1.1 %.
In the app, the estimate's speed just beyond `0.15 H` is slightly higher than at `0.15 H`. -/
theorem alpertU_jump :
    0.947 * 1.01 < 0.197 / (0.15 : ℝ) ^ ((5 : ℝ) / 6) ∧
      0.197 / (0.15 : ℝ) ^ ((5 : ℝ) / 6) ≤ 0.947 * 1.011 := by
  have hy : 0 < (0.15 : ℝ) ^ ((5 : ℝ) / 6) := rpow_pos_of_pos (by norm_num) _
  have hy6 := rpow_five_sixths_pow (show (0 : ℝ) ≤ 0.15 by norm_num)
  constructor
  · rw [lt_div_iff₀ hy]
    by_contra h
    have h' : 0.197 / (0.947 * 1.01) ≤ (0.15 : ℝ) ^ ((5 : ℝ) / 6) := by
      rw [div_le_iff₀ (by norm_num)]; linarith [not_lt.mp h]
    have := pow_le_pow_left₀ (by norm_num) h' 6
    rw [hy6] at this
    norm_num at this
  · rw [div_le_iff₀ hy]
    by_contra h
    have h' : (0.15 : ℝ) ^ ((5 : ℝ) / 6) < 0.197 / (0.947 * 1.011) := by
      rw [lt_div_iff₀ (by norm_num)]; linarith [not_le.mp h]
    have := pow_lt_pow_left₀ h' hy.le (by norm_num : (6 : ℕ) ≠ 0)
    rw [hy6] at this
    norm_num at this

/-- Alpert's speed is not antitone in `r`: at `Q = 1` kW and `H = 1` m it is 0.947 m/s at
`r = 0.15` m and about 0.952 m/s at `r = 0.151` m. The link's weight therefore need not fall as a
head is placed farther from the fire, which is why `estimate_activates_no_later` holds the speed
fixed. -/
theorem alpertU_not_antitone : alpertU 1 1 0.15 < alpertU 1 1 0.151 := by
  unfold alpertU
  have hc1 : (0.15 : ℝ) / 1 ≤ 0.15 := by norm_num
  have hc2 : ¬ ((0.151 : ℝ) / 1 ≤ 0.15) := by norm_num
  have hp : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 := by norm_num
  rw [ite_eq_left hp, ite_eq_left hp, ite_eq_left hc1,
    ite_eq_right hc2, div_one]
  simp only [one_rpow]
  have hy : 0 < (0.151 : ℝ) ^ ((5 : ℝ) / 6) := rpow_pos_of_pos (by norm_num) _
  have hy6 := rpow_five_sixths_pow (show (0 : ℝ) ≤ 0.151 by norm_num)
  rw [lt_div_iff₀ hy]
  by_contra h
  have h' : 0.197 * 1 * 1 / 0.947 ≤ (0.151 : ℝ) ^ ((5 : ℝ) / 6) := by
    rw [div_le_iff₀ (by norm_num)]; linarith [not_lt.mp h]
  have := pow_le_pow_left₀ (by norm_num) h' 6
  rw [hy6] at this
  norm_num at this

/-! ## The estimate reading and the link -/

/-- The estimate's gas at a head in sight of the fire: `max(Tg, layer + dT)`. -/
noncomputable def estimateGas (Tcell layer Q H r : ℝ) : ℝ := max Tcell (layer + alpertDT Q H r)

/-- Item 4. The estimate's gas never falls when the cell's gas, the layer or the heat release
rises, or when the head is measured nearer the fire. -/
theorem estimateGas_mono {Tc Tc' L L' Q Q' H r r' : ℝ} (hT : Tc ≤ Tc') (hL : L ≤ L')
    (hQ : Q ≤ Q') (hr : r' ≤ r) : estimateGas Tc L Q H r ≤ estimateGas Tc' L' Q' H r' := by
  unfold estimateGas
  apply max_le_max hT
  have h1 := alpertDT_mono_Q H r hQ
  have h2 := alpertDT_anti_r Q' H hr
  simp only at h1
  linarith

/-- Item 4 with item 3's comparison. With the link's weights held fixed, a cell gas, layer and
heat release at least as high at every frame, and a head measured no farther from the fire, the
estimate opens the head no later. The cell gas may be item 2's, so a lower wall coefficient never
delays the estimate either. In the app this covers the temperature channel of the estimate; the
speed channel is `linkRun_mono`. -/
theorem estimate_activates_no_later {k c Tm T0 Ta H r r' : ℝ} (hk : 0 ≤ k) (hkc : k + c ≤ 1)
    {Tc Tc' L L' Q Q' : ℕ → ℝ} (hT : ∀ n, Tc n ≤ Tc' n) (hL : ∀ n, L n ≤ L' n)
    (hQ : ∀ n, Q n ≤ Q' n) (hr : r' ≤ r) {n : ℕ}
    (h : Ta ≤ Sprinklers.linkAfter k c Tm T0 (fun m => estimateGas (Tc m) (L m) (Q m) H r) n) :
    Ta ≤ Sprinklers.linkAfter k c Tm T0 (fun m => estimateGas (Tc' m) (L' m) (Q' m) H r') n :=
  Sprinklers.activates_no_later hk hkc (fun m => estimateGas_mono (hT m) (hL m) (hQ m) hr) h

/-- The link with a weight `k n` per frame, as `activationFrame` steps it when the speed changes
from frame to frame. -/
def linkRun (k : ℕ → ℝ) (c Tm T0 : ℝ) (g : ℕ → ℝ) : ℕ → ℝ
  | 0 => T0
  | n + 1 => Sprinklers.linkStep (k n) c Tm (linkRun k c Tm T0 g n) (g n)

/-- A faster link can open later under the same gas. With weights 1/2, 1/4, 1/2 against
1/2, 1/2, 1/2 (the second run never slower), the same gas 10, 0, 10 K above ambient and no
conduction, the slower link ends 6.875 K above ambient and the faster 6.25 K, so a head that
opens at 6.5 K opens in the slower run only. The faster link lost more heat in the frame when the
gas was cooler than the link. In the app, the estimate raises the speed with the heat release as
well as the gas, so "more heat never opens a head later" needs the heating-phase hypothesis of
`linkRun_mono`. -/
theorem faster_link_can_open_later :
    linkRun (fun n => if n = 1 then 1 / 2 else 1 / 2) 0 0 0 (fun n => if n = 1 then 0 else 10) 3
        < 6.5 ∧
      6.5 ≤ linkRun (fun n => if n = 1 then 1 / 4 else 1 / 2) 0 0 0
        (fun n => if n = 1 then 0 else 10) 3 := by
  norm_num [linkRun, Sprinklers.linkStep]

/-- The corrected comparison with the speed free. If the second run's weights are at least the
first's, its gas at least the first's, `k' + c ≤ 1`, and the first run's gas has been no cooler
than its link before step `n` (the heating phase, before a head opens), then the second link is
at least as hot at step `n`. In the app, a hotter and faster estimate opens a head no later while
the gas is still heating the link. -/
theorem linkRun_mono {k k' : ℕ → ℝ} {c Tm T0 : ℝ} {g g' : ℕ → ℝ} (hk : ∀ n, 0 ≤ k n)
    (hkk : ∀ n, k n ≤ k' n) (hkc : ∀ n, k' n + c ≤ 1) (hg : ∀ n, g n ≤ g' n) :
    ∀ n, (∀ m < n, linkRun k c Tm T0 g m ≤ g m) →
      linkRun k c Tm T0 g n ≤ linkRun k' c Tm T0 g' n
  | 0, _ => le_rfl
  | n + 1, hheat => by
      have ih := linkRun_mono hk hkk hkc hg n (fun m hm => hheat m (Nat.lt_succ_of_lt hm))
      have hh := hheat n (Nat.lt_succ_self n)
      simp only [linkRun, Sprinklers.linkStep]
      have hk' : 0 ≤ k' n := le_trans (hk n) (hkk n)
      have hw : 0 ≤ 1 - k' n - c := by linarith [hkc n]
      nlinarith [mul_le_mul_of_nonneg_left ih hw, mul_le_mul_of_nonneg_left (hg n) hk',
        mul_le_mul_of_nonneg_right (hkk n) (sub_nonneg.mpr hh)]

/-! ## The head's distance from the fire -/

/-- `Math.hypot(a, b)`. -/
noncomputable def hypot (a b : ℝ) : ℝ := √(a ^ 2 + b ^ 2)

/-- A point of the plan in cell units, `(x, z)`. -/
noncomputable def pt (x z : ℝ) : EuclideanSpace ℝ (Fin 2) := !₂[x, z]

/-- `hypot` of the coordinate differences is the Euclidean distance. -/
theorem hypot_eq_dist (x z x' z' : ℝ) : hypot (x - x') (z - z') = dist (pt x z) (pt x' z') := by
  rw [EuclideanSpace.dist_eq, Fin.sum_univ_two]
  simp [pt, hypot, Real.dist_eq, sq_abs]

/-- The centre of the cell that holds `x`, in cell units. -/
noncomputable def cellCentre (x : ℝ) : ℝ := ⌊x⌋ + 1 / 2

/-- Item 5. Every coordinate lies within half a cell of its cell's centre. -/
theorem abs_sub_cellCentre_le (x : ℝ) : |x - cellCentre x| ≤ 1 / 2 := by
  unfold cellCentre
  have h1 := Int.floor_le x
  have h2 := Int.lt_floor_add_one x
  rw [abs_le]; constructor <;> linarith

/-- Snapping a point to its cell's centre moves it at most `√2 / 2` cells. -/
theorem dist_snap_le (x z : ℝ) : dist (pt x z) (pt (cellCentre x) (cellCentre z)) ≤ √2 / 2 := by
  rw [← hypot_eq_dist]
  unfold hypot
  have hx := abs_le.mp (abs_sub_cellCentre_le x)
  have hz := abs_le.mp (abs_sub_cellCentre_le z)
  have h2 : (√2 / 2) ^ 2 = 1 / 2 := by rw [div_pow, sq_sqrt (by norm_num)]; norm_num
  rw [sqrt_le_left (by positivity), h2]
  nlinarith

/-- Item 5, before the fix. The head's distance was measured between the centres of the head's
cell `(⌊x⌋, ⌊z⌋)` and the fire cell `(⌊fx⌋, ⌊fz⌋)`, `h · hypot(i - pi, k - pk)`. It differs from
the true distance between the head at `(x, z)` and the fire at `(fx, fz)` by at most `h √2`, half
a cell's diagonal at each end. -/
theorem old_distance_error {h x z fx fz : ℝ} (hh : 0 ≤ h) :
    |h * hypot ((⌊x⌋ : ℝ) - ⌊fx⌋) ((⌊z⌋ : ℝ) - ⌊fz⌋) - h * hypot (x - fx) (z - fz)| ≤
      h * √2 := by
  have e1 : hypot ((⌊x⌋ : ℝ) - ⌊fx⌋) ((⌊z⌋ : ℝ) - ⌊fz⌋) =
      dist (pt (cellCentre x) (cellCentre z)) (pt (cellCentre fx) (cellCentre fz)) := by
    rw [← hypot_eq_dist]; unfold cellCentre; congr 1 <;> ring
  rw [e1, hypot_eq_dist, ← mul_sub, abs_mul, abs_of_nonneg hh]
  apply mul_le_mul_of_nonneg_left _ hh
  have s1 := dist_snap_le x z
  have s2 := dist_snap_le fx fz
  have a1 := dist_triangle4 (pt (cellCentre x) (cellCentre z)) (pt x z) (pt fx fz)
    (pt (cellCentre fx) (cellCentre fz))
  have a2 := dist_triangle4 (pt x z) (pt (cellCentre x) (cellCentre z))
    (pt (cellCentre fx) (cellCentre fz)) (pt fx fz)
  have c1 := dist_comm (pt x z) (pt (cellCentre x) (cellCentre z))
  have c2 := dist_comm (pt fx fz) (pt (cellCentre fx) (cellCentre fz))
  rw [abs_le]; constructor <;> linarith

/-- Item 5, after the fix. The distance is now measured from the head's exact position to the
fire cell's centre, `h · hypot(x - (pi + 1/2), z - (pk + 1/2))`. When the fire cell is the cell of
the true fire point, it differs from the true distance by at most `h √2 / 2`, half the old bound:
only the fire's end is snapped. -/
theorem new_distance_error {h x z fx fz : ℝ} (hh : 0 ≤ h) :
    |h * hypot (x - (⌊fx⌋ + 1 / 2)) (z - (⌊fz⌋ + 1 / 2)) - h * hypot (x - fx) (z - fz)| ≤
      h * √2 / 2 := by
  have e1 : hypot (x - (⌊fx⌋ + 1 / 2)) (z - (⌊fz⌋ + 1 / 2)) =
      dist (pt x z) (pt (cellCentre fx) (cellCentre fz)) := by
    rw [← hypot_eq_dist]; rfl
  rw [e1, hypot_eq_dist, ← mul_sub, abs_mul, abs_of_nonneg hh, mul_div_assoc]
  apply mul_le_mul_of_nonneg_left _ hh
  have s2 := dist_snap_le fx fz
  have a1 := dist_triangle (pt x z) (pt fx fz) (pt (cellCentre fx) (cellCentre fz))
  have a2 := dist_triangle (pt x z) (pt (cellCentre fx) (cellCentre fz)) (pt fx fz)
  have c2 := dist_comm (pt fx fz) (pt (cellCentre fx) (cellCentre fz))
  rw [abs_le]; constructor <;> linarith

lemma hypot_self (a : ℝ) : hypot a a = |a| * √2 := by
  unfold hypot
  rw [show a ^ 2 + a ^ 2 = a ^ 2 * 2 by ring, sqrt_mul' _ (by norm_num), sqrt_sq_eq_abs]

/-- Both bounds are close to the worst case. With the fire at a cell's corner `(0, 0)` and the
head at `(-0.1, -0.1)` in the next cell diagonally, the true distance is `0.1 √2` cells; the old
distance is `√2`, wrong by `0.9 √2`, and the new one is `0.6 √2`, wrong by exactly `√2 / 2`, the
new bound. In the app, the old measure could misplace a head by nearly a cell diagonal and the new
one by at most half of one. -/
theorem distance_example :
    hypot ((⌊(-0.1 : ℝ)⌋ : ℝ) - ⌊(0 : ℝ)⌋) ((⌊(-0.1 : ℝ)⌋ : ℝ) - ⌊(0 : ℝ)⌋) -
        hypot (-0.1 - 0) (-0.1 - 0) = 0.9 * √2 ∧
      hypot (-0.1 - (⌊(0 : ℝ)⌋ + 1 / 2)) (-0.1 - (⌊(0 : ℝ)⌋ + 1 / 2)) -
        hypot (-0.1 - 0) (-0.1 - 0) = √2 / 2 := by
  have hf : ⌊(-0.1 : ℝ)⌋ = -1 := by rw [Int.floor_eq_iff]; norm_num
  rw [hf, Int.floor_zero, hypot_self, hypot_self, hypot_self]
  constructor <;> norm_num [abs_of_neg] <;> ring

end FireEgress.SprinklerActivation
