import Mathlib
import FireEgress.Dim
import FireEgress.FireDynamics
import FireEgress.Forecast

/-!
# The two-zone room model is monotone (LG8, properties P1 to P5)

Studio's two-zone fire model (`topologicpy-web-frontend/src/core/zone.js`, specified in
`wiki/research/zone-model-spec-2026-10-28.md`) gives every IFC room an upper-layer
temperature and an interface height on a physical clock, and hands the routers a forecast in
the shape `buildForecast` returns. The routers are exact, and the deadline map conservative,
only for heat that never falls (`Forecast.lean`, `Deadline.lean`). The specification's
properties P1 to P4 say that this model meets that condition; this file proves them over
real arithmetic, function by function, and checks the units of P5 with `Dim`. The
equations reuse `FireDynamics`: `mqh` (FD-17, MQH), `hkEff` and `tPen` (FD-18, the lining's
conductance) and the unit theorems of FD-01, FD-08, FD-17 and FD-18.

## Which theorem covers which function

* P1, the fire room (`heatRelease`, `liningConductance`, `mqhRise`, `fireRoomHistory`).
  `heatRelease_monotoneOn` (F1: `Q = min(α t², Q_max)` never falls for `t ≥ 0`);
  `liningConductance_eq`, `liningConductance_antitoneOn` (F2: the code's `h_k` is
  `FireDynamics.hkEff / 1000`, so it never rises, across `t_p` included, by
  `FireDynamics.FD18_antitone_time`); `mqhRise_mono_Q`, `mqhRise_anti_hk` (the MQH rise
  rises with `Q` and falls with `h_k`, via `FD17_strictMono_Q` and `FD17_antitone_hk`);
  `rawRise_monotoneOn` (so the raw rise never falls, `t = 0` included, where `Q = 0` and
  the code never reads `h_k(0) = Infinity`); `latch_monotone` (F3: the flashover latch, 900 K
  from the first step at which the raw rise reaches 500 K, keeps it monotone);
  `fireRoomRise_monotone`, `fireRoomRise_nonneg`.
* P2, the layer step (`layerStep`). `layerStep_le_self` (a step never raises the interface),
  `layerStep_mem` (it stays in `[0, H]`), and `layer_antitone`: the interface sequence the
  code computes never rises, **whatever** the inflow, outflow and density inputs. The
  specification's hypothesis of non-decreasing inputs is not needed: the clamp of
  `ṁ_in - ṁ_out` at 0 does all the work, as its note on P2 says.
* P3, spread (`doorOutflow`, `fedRoomRise`, `simulateRooms`). `doorOutflow_mono` (S2 rises
  with the upstream temperature and falls with the upstream interface);
  `fedRoomRise_eq_spec` (the code's `dT / (1 + h_c A_T / (ṁ c_p))` is the specification's
  R3), `fedRoomRise_mono` (R3 rises with the inflow and its temperature); and
  `rooms_rise_monotone`, `zone_rooms_monotone`: by strong induction along the breadth-first
  order (every feeding room comes earlier), every room's rise never falls, is `≥ 0`, and its
  interface never rises.
* P4, nodes (`inUpperLayer`, `nodeTemperatureC`, `buildZoneForecast`). `inUpper_upset`,
  `nodeTemperature_monotone`, `untenable_upset`: a node's temperature never falls and its
  untenability (layer at head height, and either the smoke criterion or `T ≥ θ`) is an
  up-set in time. Discharging the forecast hypotheses: `nodeTemperature_runMax` (the stored
  running maximum is the temperature itself), `nodeRate_monotone` (its FED rate never falls,
  `Forecast.doseRate_monotone`), `zone_closed_iff` (via `Forecast.closed_iff`, the node is
  untenable at a step before the horizon exactly from its closing step on).
* P5, units (`Dim`). `S2_units`, `S3_units`, `R3_units`, `gasDensity_units`, and `P5_units`,
  which adds FD-01 (F1), FD-08 (S1), FD-17 (F2) and FD-18 (`h_k`, `t_p`) from
  `FireDynamics`.

## What is not covered

* Floating point. The code's header says each formula is written so that every double
  operation is monotone in the argument concerned; that rounding preserves monotonicity is
  checked by `zone.test.mjs`, not proved here. Arithmetic here is real.
* The physical hypotheses are taken as stated: positive vent area and height, lining
  properties and wall area (`rooms.js` keeps `A_T ≥ A_f > 0`), non-negative door widths and
  growth coefficient, `dt ≥ 0`. `Math.cbrt` is read as the real cube root `x^(1/3)` on
  `x ≥ 0`, which is all MQH evaluates.
* The plume (S1), the virtual origin and the fire diameter enter only the fire room's layer
  step, and P2 holds for any inflow, so their values are not used; FD-08's monotonicity in
  height is in `FireDynamics`.
* Simulation is modelled by its equations: `zone_rooms_monotone` holds for any functions
  that satisfy the recurrences of `simulateRooms` along a ranking in which every feeding room
  comes first (`roomOrder`'s `pos`); that `roomOrder` is a breadth-first order is read from
  the code. Rooms never reached stay at ambient in the code and are trivially monotone.
* The frame sampling of `buildZoneForecast` (`hottest`, the running maximum over the Euler
  steps of each frame) and of `roomTemperatureFrames` (rounded step index) is not modelled
  beyond `nodeTemperature_runMax`. P6 (convergence of the Euler step) is numerical and is not
  covered.
* Units: the dimension type does not see the kW/W and kJ/J prefixes the code mixes (`c_p` in
  kJ kg⁻¹ K⁻¹ with `h_c` in kW m⁻² K⁻¹ is consistent; so is `h_k / 1000` in kW).
-/

namespace FireEgress.Zone

open Real FireEgress.Forecast

/-! ## Constants of `zone.js` -/

/-- `T_AMBIENT` (K). -/
noncomputable def Ta : ℝ := 293.15
/-- `KELVIN`, the Celsius offset of N1. -/
noncomputable def kelvin : ℝ := 273.15
/-- `gasDensity(T) = 353 / T` (kg m⁻³). -/
noncomputable def gasDensity (T : ℝ) : ℝ := 353 / T
/-- `RHO_AMBIENT`. -/
noncomputable def rhoAmbient : ℝ := gasDensity Ta
/-- `FLASHOVER_RISE` (K, F3). -/
noncomputable def flashoverRise : ℝ := 500
/-- `POST_FLASHOVER_RISE` (K, F3). -/
noncomputable def postFlashoverRise : ℝ := 900
/-- `DOOR_CD` (S2). -/
noncomputable def doorCd : ℝ := 0.7
/-- `GRAVITY` (m s⁻², S2). -/
noncomputable def gravity : ℝ := 9.81
/-- `CP` (kJ kg⁻¹ K⁻¹, R3). -/
noncomputable def cp : ℝ := 1.0
/-- `WALL_LOSS`, `h_c` (kW m⁻² K⁻¹, R3). -/
noncomputable def wallLoss : ℝ := 0.015

theorem Ta_pos : 0 < Ta := by norm_num [Ta]

theorem rhoAmbient_pos : 0 < rhoAmbient := by
  unfold rhoAmbient gasDensity
  exact div_pos (by norm_num) Ta_pos

theorem flashover_le_post : flashoverRise ≤ postFlashoverRise := by
  norm_num [flashoverRise, postFlashoverRise]

/-! ## P1. The fire room's rise never falls -/

/-- (F1) `heatRelease(t, α, Q_max) = min(α t², Q_max)` (kW). -/
noncomputable def heatRelease (α Qmax t : ℝ) : ℝ := min (α * t * t) Qmax

/-- (F1) is FD-01's t-squared fire from `t₀ = 0`, capped. -/
theorem heatRelease_eq_tsq (α Qmax t : ℝ) :
    heatRelease α Qmax t = min (FireDynamics.tsq α 0 t) Qmax := by
  unfold heatRelease FireDynamics.tsq
  congr 1
  ring

/-- **P1 (F1).** For `α ≥ 0` the heat release never falls on `t ≥ 0`. -/
theorem heatRelease_monotoneOn {α : ℝ} (hα : 0 ≤ α) (Qmax : ℝ) :
    MonotoneOn (heatRelease α Qmax) (Set.Ici 0) := by
  intro s hs t _ hst
  simp only [Set.mem_Ici] at hs
  unfold heatRelease
  apply min_le_min_right
  have h1 : s * s ≤ t * t := mul_self_le_mul_self hs hst
  calc α * s * s = α * (s * s) := by ring
    _ ≤ α * (t * t) := mul_le_mul_of_nonneg_left h1 hα
    _ = α * t * t := by ring

theorem heatRelease_zero_le (α Qmax : ℝ) : heatRelease α Qmax 0 ≤ 0 := by
  unfold heatRelease
  exact (min_le_left _ _).trans (by norm_num)

/-- A positive heat release comes after ignition. -/
theorem pos_of_heatRelease_pos {α Qmax t : ℝ} (ht : 0 ≤ t) (hQ : 0 < heatRelease α Qmax t) :
    0 < t := by
  rcases ht.lt_or_eq with h | h
  · exact h
  · rw [← h] at hQ
    exact absurd (heatRelease_zero_le α Qmax) (not_le.2 hQ)

/-- (F2, FD-18) `liningConductance(t)` (kW m⁻² K⁻¹): `√(k ρ c / t) / 1000` before `t_p`,
`k / (1000 δ)` from `t_p` on. -/
noncomputable def liningConductance (k ρ c δ t : ℝ) : ℝ :=
  if t < FireDynamics.tPen k ρ c δ then √(k * ρ * c / t) / 1000 else k / (1000 * δ)

/-- The code's conductance is FD-18's `hkEff` in kW (the same branch convention, `t < t_p`
thick). -/
theorem liningConductance_eq (k ρ c δ t : ℝ) :
    liningConductance k ρ c δ t = FireDynamics.hkEff k ρ c δ t / 1000 := by
  unfold liningConductance FireDynamics.hkEff
  split_ifs
  · rfl
  · rw [div_div, mul_comm]

/-- **P1 (F2).** The lining's conductance never rises with time, across `t_p` included. -/
theorem liningConductance_antitoneOn {k ρ c δ : ℝ} (hk : 0 < k) (hρ : 0 < ρ) (hc : 0 < c)
    (hδ : 0 < δ) : AntitoneOn (liningConductance k ρ c δ) (Set.Ioi 0) := by
  intro a ha b hb hab
  rw [liningConductance_eq, liningConductance_eq]
  exact div_le_div_of_nonneg_right (FireDynamics.FD18_antitone_time k ρ c δ hk hρ hc hδ ha hb hab)
    (by norm_num)

theorem liningConductance_pos {k ρ c δ : ℝ} (hk : 0 < k) (hρ : 0 < ρ) (hc : 0 < c)
    (hδ : 0 < δ) {t : ℝ} (ht : 0 < t) : 0 < liningConductance k ρ c δ t := by
  unfold liningConductance
  split_ifs
  · exact div_pos (Real.sqrt_pos.mpr (by positivity)) (by norm_num)
  · positivity

/-- (F2, FD-17) `mqhRise`: `6.85 (Q² / (A_o √H_o h_k A_T))^(1/3)` for `Q > 0`, else 0. -/
noncomputable def mqhRise (Q Ao Ho hk AT : ℝ) : ℝ :=
  if 0 < Q then FireDynamics.mqh 6.85 Q Ao Ho hk AT else 0

theorem mqh_nonneg {Q Ao Ho hk AT : ℝ} (hA : 0 < Ao) (hH : 0 < Ho) (hk0 : 0 < hk)
    (hAT : 0 < AT) : 0 ≤ FireDynamics.mqh 6.85 Q Ao Ho hk AT := by
  unfold FireDynamics.mqh
  have h1 : 0 ≤ Q ^ 2 / (Ao * √Ho * hk * AT) := by positivity
  have h2 := Real.rpow_nonneg h1 ((1 : ℝ) / 3)
  positivity

/-- The MQH rise is `≥ 0` (a positive conductance is needed only when `Q > 0`). -/
theorem mqhRise_nonneg {Q Ao Ho hk AT : ℝ} (hA : 0 < Ao) (hH : 0 < Ho) (hAT : 0 < AT)
    (hk0 : 0 < Q → 0 < hk) : 0 ≤ mqhRise Q Ao Ho hk AT := by
  unfold mqhRise
  split_ifs with hQ
  · exact mqh_nonneg hA hH (hk0 hQ) hAT
  · exact le_rfl

/-- **P1.** The MQH rise never falls with `Q`. -/
theorem mqhRise_mono_Q {Q Q' Ao Ho hk AT : ℝ} (hA : 0 < Ao) (hH : 0 < Ho) (hk0 : 0 < hk)
    (hAT : 0 < AT) (hQ : Q ≤ Q') : mqhRise Q Ao Ho hk AT ≤ mqhRise Q' Ao Ho hk AT := by
  unfold mqhRise
  split_ifs with h1 h2 h2
  · exact (FireDynamics.FD17_strictMono_Q 6.85 Ao Ho hk AT (by norm_num) hA hH hk0 hAT).monotoneOn
      (Set.mem_Ici.2 h1.le) (Set.mem_Ici.2 (h1.le.trans hQ)) hQ
  · exact absurd (h1.trans_le hQ) h2
  · exact mqh_nonneg hA hH hk0 hAT
  · exact le_rfl

/-- **P1.** The MQH rise never rises with the conductance. -/
theorem mqhRise_anti_hk {Q Ao Ho AT hk hk' : ℝ} (hA : 0 < Ao) (hH : 0 < Ho) (hAT : 0 < AT)
    (hk0 : 0 < hk) (hkk : hk ≤ hk') : mqhRise Q Ao Ho hk' AT ≤ mqhRise Q Ao Ho hk AT := by
  unfold mqhRise
  split_ifs with h
  · exact FireDynamics.FD17_antitone_hk 6.85 Q Ao Ho AT (by norm_num) hA hH hAT hk0
      (hk0.trans_le hkk) hkk
  · exact le_rfl

/-- The fire room: growth coefficient `α`, ventilation-limited peak `Q_max`, vent `A_o`,
`H_o`, and internal surface `A_T` (`fireRoomHistory`'s arguments). -/
structure Fire where
  α : ℝ
  Qmax : ℝ
  Ao : ℝ
  Ho : ℝ
  AT : ℝ

/-- The lining: `k`, `ρ`, `c`, `δ` (`GYPSUM` by default). -/
structure Lining where
  k : ℝ
  ρ : ℝ
  c : ℝ
  δ : ℝ

/-- The physical hypotheses `fireRoomHistory` is called with (`rooms.js`, `fireVent`,
`GYPSUM`). -/
structure Fire.Good (F : Fire) (L : Lining) : Prop where
  hα : 0 ≤ F.α
  hAo : 0 < F.Ao
  hHo : 0 < F.Ho
  hAT : 0 < F.AT
  hk : 0 < L.k
  hρ : 0 < L.ρ
  hc : 0 < L.c
  hδ : 0 < L.δ

/-- The raw MQH rise at time `t`: `mqhRise(Q(t), A_o, H_o, h_k(t), A_T)`. -/
noncomputable def rawRise (F : Fire) (L : Lining) (t : ℝ) : ℝ :=
  mqhRise (heatRelease F.α F.Qmax t) F.Ao F.Ho (liningConductance L.k L.ρ L.c L.δ t) F.AT

theorem rawRise_nonneg {F : Fire} {L : Lining} (hG : F.Good L) {t : ℝ} (ht : 0 ≤ t) :
    0 ≤ rawRise F L t :=
  mqhRise_nonneg hG.hAo hG.hHo hG.hAT fun hQ =>
    liningConductance_pos hG.hk hG.hρ hG.hc hG.hδ (pos_of_heatRelease_pos ht hQ)

/-- **P1.** The raw rise never falls on `t ≥ 0`: `Q` never falls and `h_k` never rises. At
`t = 0`, `Q = 0` and the rise is 0 (the code returns before reading `h_k(0)`). -/
theorem rawRise_monotoneOn {F : Fire} {L : Lining} (hG : F.Good L) :
    MonotoneOn (rawRise F L) (Set.Ici 0) := by
  intro s hs t ht hst
  have hs' : 0 ≤ s := hs
  have ht' : 0 ≤ t := ht
  by_cases hQs : 0 < heatRelease F.α F.Qmax s
  · have hs0 : 0 < s := pos_of_heatRelease_pos hs' hQs
    have ht0 : 0 < t := hs0.trans_le hst
    have hks := liningConductance_pos hG.hk hG.hρ hG.hc hG.hδ hs0
    have hkt := liningConductance_pos hG.hk hG.hρ hG.hc hG.hδ ht0
    unfold rawRise
    calc mqhRise (heatRelease F.α F.Qmax s) F.Ao F.Ho (liningConductance L.k L.ρ L.c L.δ s) F.AT
        ≤ mqhRise (heatRelease F.α F.Qmax t) F.Ao F.Ho
            (liningConductance L.k L.ρ L.c L.δ s) F.AT :=
          mqhRise_mono_Q hG.hAo hG.hHo hks hG.hAT (heatRelease_monotoneOn hG.hα F.Qmax hs ht hst)
      _ ≤ mqhRise (heatRelease F.α F.Qmax t) F.Ao F.Ho
            (liningConductance L.k L.ρ L.c L.δ t) F.AT :=
          mqhRise_anti_hk hG.hAo hG.hHo hG.hAT hkt
            (liningConductance_antitoneOn hG.hk hG.hρ hG.hc hG.hδ hs0 ht0 hst)
  · have e : rawRise F L s = 0 := by
      unfold rawRise mqhRise
      exact ite_eq_right hQs
    rw [e]
    exact rawRise_nonneg hG ht'

/-- (F3) The flashover latch of `fireRoomHistory`: `900` from the first step at which the raw
rise reaches `500`, the raw rise before. -/
noncomputable def latch (raw : ℕ → ℝ) (i : ℕ) : ℝ :=
  open Classical in
  if ∃ j ≤ i, flashoverRise ≤ raw j then postFlashoverRise else raw i

/-- **P1 (F3).** The latch keeps a monotone rise monotone: before flashover the rise is below
500 K, after it 900 K. -/
theorem latch_monotone {raw : ℕ → ℝ} (hraw : Monotone raw) : Monotone (latch raw) := by
  intro i i' hii'
  unfold latch
  split_ifs with h1 h2 h2
  · exact le_rfl
  · obtain ⟨j, hj, hr⟩ := h1
    exact absurd ⟨j, hj.trans hii', hr⟩ h2
  · push Not at h1
    linarith [h1 i le_rfl, flashover_le_post]
  · exact hraw hii'

/-- `fireRoomHistory`'s `rise[i]`, at `t = i dt`. -/
noncomputable def fireRoomRise (F : Fire) (L : Lining) (dt : ℝ) (i : ℕ) : ℝ :=
  latch (fun i => rawRise F L (i * dt)) i

/-- **P1.** The fire room's rise never falls, across `t_p` and flashover. -/
theorem fireRoomRise_monotone {F : Fire} {L : Lining} (hG : F.Good L) {dt : ℝ} (hdt : 0 ≤ dt) :
    Monotone (fireRoomRise F L dt) :=
  latch_monotone fun i j hij =>
    rawRise_monotoneOn hG (Set.mem_Ici.2 (by positivity)) (Set.mem_Ici.2 (by positivity))
      (mul_le_mul_of_nonneg_right (Nat.cast_le.2 hij) hdt)

theorem fireRoomRise_nonneg {F : Fire} {L : Lining} (hG : F.Good L) {dt : ℝ} (hdt : 0 ≤ dt)
    (i : ℕ) : 0 ≤ fireRoomRise F L dt i := by
  unfold fireRoomRise latch
  split_ifs
  · norm_num [postFlashoverRise]
  · exact rawRise_nonneg hG (by positivity)

/-! ## P2. The layer step never raises the interface -/

/-- (S3) The fall of one Euler step, `dt max(0, ṁ_in - ṁ_out) / (ρ_U A_f)`. -/
noncomputable def layerFall (mIn mOut ρU Af dt : ℝ) : ℝ := dt * max 0 (mIn - mOut) / (ρU * Af)

/-- (S3) `layerStep`: `min(H, max(0, fall > 0 ? z - fall : z))`. -/
noncomputable def layerStep (z mIn mOut ρU Af H dt : ℝ) : ℝ :=
  min H (max 0 (if 0 < layerFall mIn mOut ρU Af dt then z - layerFall mIn mOut ρU Af dt else z))

/-- **P2.** A step never raises an interface that is above the floor, whatever its inputs
(the code subtracts only a positive fall). -/
theorem layerStep_le_self {z : ℝ} (hz : 0 ≤ z) (mIn mOut ρU Af H dt : ℝ) :
    layerStep z mIn mOut ρU Af H dt ≤ z := by
  unfold layerStep
  refine (min_le_right _ _).trans (max_le hz ?_)
  split_ifs with h
  · linarith
  · exact le_rfl

/-- **P2.** A step keeps the interface in `[0, H]`. -/
theorem layerStep_mem {H : ℝ} (hH : 0 ≤ H) (z mIn mOut ρU Af dt : ℝ) :
    0 ≤ layerStep z mIn mOut ρU Af H dt ∧ layerStep z mIn mOut ρU Af H dt ≤ H :=
  ⟨le_min hH (le_max_left _ _), min_le_left _ _⟩

/-- **P2.** The interface `simulateRooms` computes (`zj[0] = H`,
`zj[i + 1] = layerStep(zj[i], …)`) stays in `[0, H]` and never rises, for any inflow,
outflow and density at every step. -/
theorem layer_antitone {z : ℕ → ℝ} {H : ℝ} (hH : 0 ≤ H) (mIn mOut ρU : ℕ → ℝ) (Af dt : ℝ)
    (h0 : z 0 = H) (hstep : ∀ i, z (i + 1) = layerStep (z i) (mIn i) (mOut i) (ρU i) Af H dt) :
    (∀ i, 0 ≤ z i ∧ z i ≤ H) ∧ Antitone z := by
  have hb : ∀ i, 0 ≤ z i ∧ z i ≤ H := by
    intro i
    cases i with
    | zero => rw [h0]; exact ⟨hH, le_rfl⟩
    | succ i => rw [hstep]; exact layerStep_mem hH _ _ _ _ _ _
  exact ⟨hb, antitone_nat_of_succ_le fun i => by
    rw [hstep]; exact layerStep_le_self (hb i).1 _ _ _ _ _ _⟩

/-! ## P3. Spread: door outflow, the fed room's rise, and the breadth-first induction -/

/-- (S2) `doorOutflow(W_d, H_d, T_U, z)`: `(2/3) C_d W_d ρ_a √(2 g (1 - T_a / T_U))
(H_d - z)^1.5` when the layer is below the door's top and above ambient, else 0 (kg s⁻¹). -/
noncomputable def doorOutflow (Wd Hd TU z : ℝ) : ℝ :=
  if 0 < Hd - z ∧ Ta < TU then
    2 / 3 * doorCd * Wd * rhoAmbient * √(2 * gravity * (1 - Ta / TU)) * (Hd - z) ^ (1.5 : ℝ)
  else 0

theorem doorOutflow_nonneg {Wd : ℝ} (hW : 0 ≤ Wd) (Hd TU z : ℝ) : 0 ≤ doorOutflow Wd Hd TU z := by
  unfold doorOutflow
  split_ifs with h
  · have h1 := rhoAmbient_pos
    have h2 := Real.rpow_nonneg h.1.le (1.5 : ℝ)
    have h3 : (0 : ℝ) ≤ doorCd := by norm_num [doorCd]
    positivity
  · exact le_rfl

/-- **P3 (S2).** Door outflow never falls with the upstream layer temperature and never rises
with the upstream interface height. -/
theorem doorOutflow_mono {Wd Hd TU TU' z z' : ℝ} (hW : 0 ≤ Wd) (hT : TU ≤ TU') (hz : z' ≤ z) :
    doorOutflow Wd Hd TU z ≤ doorOutflow Wd Hd TU' z' := by
  by_cases h : 0 < Hd - z ∧ Ta < TU
  · have h' : 0 < Hd - z' ∧ Ta < TU' := ⟨by linarith [h.1], h.2.trans_le hT⟩
    unfold doorOutflow
    rw [ite_eq_left h, ite_eq_left h']
    have hD : 0 < Hd - z := h.1
    have hTU : 0 < TU := Ta_pos.trans h.2
    have h1 : Ta / TU' ≤ Ta / TU := div_le_div_of_nonneg_left Ta_pos.le hTU hT
    have hg : (0 : ℝ) ≤ 2 * gravity := by norm_num [gravity]
    have h2 : √(2 * gravity * (1 - Ta / TU)) ≤ √(2 * gravity * (1 - Ta / TU')) :=
      Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left (by linarith) hg)
    have h3 : (Hd - z) ^ (1.5 : ℝ) ≤ (Hd - z') ^ (1.5 : ℝ) :=
      Real.rpow_le_rpow hD.le (by linarith) (by norm_num)
    have hK : 0 ≤ 2 / 3 * doorCd * Wd * rhoAmbient := by
      have := rhoAmbient_pos
      have : (0 : ℝ) ≤ doorCd := by norm_num [doorCd]
      positivity
    calc 2 / 3 * doorCd * Wd * rhoAmbient * √(2 * gravity * (1 - Ta / TU)) * (Hd - z) ^ (1.5 : ℝ)
        ≤ 2 / 3 * doorCd * Wd * rhoAmbient * √(2 * gravity * (1 - Ta / TU')) *
            (Hd - z) ^ (1.5 : ℝ) :=
          mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left h2 hK) (Real.rpow_nonneg hD.le _)
      _ ≤ 2 / 3 * doorCd * Wd * rhoAmbient * √(2 * gravity * (1 - Ta / TU')) *
            (Hd - z') ^ (1.5 : ℝ) :=
          mul_le_mul_of_nonneg_left h3 (mul_nonneg hK (Real.sqrt_nonneg _))
  · have e : doorOutflow Wd Hd TU z = 0 := by
      unfold doorOutflow
      exact ite_eq_right h
    rw [e]
    exact doorOutflow_nonneg hW _ _ _

/-- (R3) `fedRoomRise(dT_in, ṁ_in, A_T)`: `dT_in / (1 + h_c A_T / (ṁ_in c_p))` for positive
inflow and rise, else 0. -/
noncomputable def fedRoomRise (dTin mIn AT : ℝ) : ℝ :=
  if 0 < mIn ∧ 0 < dTin then dTin / (1 + wallLoss * AT / (mIn * cp)) else 0

/-- The code's form of (R3) is the specification's `ΔT_in ṁ c_p / (ṁ c_p + h_c A_T)`. -/
theorem fedRoomRise_eq_spec {dTin mIn AT : ℝ} (hm : 0 < mIn) (hd : 0 < dTin) :
    fedRoomRise dTin mIn AT = dTin * (mIn * cp) / (mIn * cp + wallLoss * AT) := by
  unfold fedRoomRise
  rw [ite_eq_left ⟨hm, hd⟩]
  have hc : 0 < mIn * cp := by norm_num [cp]; exact hm
  have hY : mIn * cp ≠ 0 := hc.ne'
  rw [show 1 + wallLoss * AT / (mIn * cp) = (mIn * cp + wallLoss * AT) / (mIn * cp) by
    rw [add_div, div_self hY], div_div_eq_mul_div]

theorem fedRoomRise_nonneg {AT : ℝ} (hAT : 0 ≤ AT) (dTin mIn : ℝ) : 0 ≤ fedRoomRise dTin mIn AT := by
  unfold fedRoomRise
  split_ifs with h
  · have : 0 < mIn * cp := by norm_num [cp]; exact h.1
    have : 0 ≤ wallLoss * AT := mul_nonneg (by norm_num [wallLoss]) hAT
    have : 0 ≤ wallLoss * AT / (mIn * cp) := by positivity
    exact div_nonneg h.2.le (by linarith)
  · exact le_rfl

/-- **P3 (R3).** The fed room's rise never falls with its inflow or with the incoming
temperature rise. -/
theorem fedRoomRise_mono {AT : ℝ} (hAT : 0 ≤ AT) {d d' m m' : ℝ} (hd : d ≤ d') (hm : m ≤ m') :
    fedRoomRise d m AT ≤ fedRoomRise d' m' AT := by
  by_cases h : 0 < m ∧ 0 < d
  · have h' : 0 < m' ∧ 0 < d' := ⟨h.1.trans_le hm, h.2.trans_le hd⟩
    unfold fedRoomRise
    rw [ite_eq_left h, ite_eq_left h']
    have hmc : 0 < m * cp := by norm_num [cp]; exact h.1
    have hmc' : m * cp ≤ m' * cp := by norm_num [cp]; exact hm
    have hW : 0 ≤ wallLoss * AT := mul_nonneg (by norm_num [wallLoss]) hAT
    have hq : wallLoss * AT / (m' * cp) ≤ wallLoss * AT / (m * cp) :=
      div_le_div_of_nonneg_left hW hmc hmc'
    have hq0 : 0 ≤ wallLoss * AT / (m' * cp) := div_nonneg hW (hmc.trans_le hmc').le
    exact div_le_div₀ h'.2.le hd (by linarith) (by linarith)
  · have e : fedRoomRise d m AT = 0 := by
      unfold fedRoomRise
      exact ite_eq_right h
    rw [e]
    exact fedRoomRise_nonneg hAT _ _

/-- A door that feeds a room: its width and height (m) and the room on its other side. -/
structure Feed (R : Type*) where
  width : ℝ
  height : ℝ
  src : R

/-- (R1) The inflow into a fed room at step `i`: the sum over its feeding doors of the
outflow computed from the room behind each (`doorOutflow(d.width, d.height, T_a +
rise[from][i], z[from][i])`). -/
noncomputable def inflow {R : Type*} (feeds : List (Feed R)) (rise z : R → ℕ → ℝ) (i : ℕ) : ℝ :=
  (feeds.map fun f => doorOutflow f.width f.height (Ta + rise f.src i) (z f.src i)).sum

/-- (R2) The incoming rise: the largest of the feeding rooms' rises, from 0 (the code's
`dTin = 0; if (rise > dTin) dTin = rise`). -/
noncomputable def inRise {R : Type*} (feeds : List (Feed R)) (rise : R → ℕ → ℝ) (i : ℕ) : ℝ :=
  (feeds.map fun f => rise f.src i).foldl max 0

theorem foldl_max_monotone {ι : Type*} (L : List ι) (g : ι → ℕ → ℝ)
    (hg : ∀ x ∈ L, Monotone (g x)) :
    ∀ a : ℕ → ℝ, Monotone a → Monotone fun i => (L.map fun x => g x i).foldl max (a i) := by
  induction L with
  | nil =>
    intro a ha i j hij
    simp only [List.map_nil, List.foldl_nil]
    exact ha hij
  | cons x L ih =>
    intro a ha i j hij
    have hx : Monotone (g x) := hg x (by simp)
    have := ih (fun y hy => hg y (by simp [hy])) (fun i => max (a i) (g x i))
      (fun i j hij => max_le_max (ha hij) (hx hij)) hij
    simp only [List.map_cons, List.foldl_cons]
    exact this

theorem sum_map_monotone {ι : Type*} (L : List ι) (g : ι → ℕ → ℝ)
    (hg : ∀ x ∈ L, Monotone (g x)) : Monotone fun i => (L.map fun x => g x i).sum := by
  induction L with
  | nil => intro i j _; simp
  | cons x L ih =>
    intro i j hij
    simp only [List.map_cons, List.sum_cons]
    exact add_le_add (hg x (by simp) hij) (ih (fun y hy => hg y (by simp [hy])) hij)

/-- **P3, the breadth-first induction.** Rooms are ranked by `pos` (`roomOrder`), every door
that feeds a room comes from a room ranked earlier, door widths and wall areas are `≥ 0`, and
every interface never rises (P2). If the fire room's rise never falls and is `≥ 0` (P1) and
every other room's rise is (R1)-(R3) of the rooms before it, then every room's rise never
falls and is `≥ 0`. -/
theorem rooms_rise_monotone {R : Type*} (pos : R → ℕ) (fireRoom : R) (feeds : R → List (Feed R))
    (AT : R → ℝ) (rise z : R → ℕ → ℝ) (hfeed : ∀ j, ∀ f ∈ feeds j, pos f.src < pos j)
    (hW : ∀ j, ∀ f ∈ feeds j, 0 ≤ f.width) (hAT : ∀ j, 0 ≤ AT j) (hz : ∀ j, Antitone (z j))
    (hfire : Monotone (rise fireRoom)) (hfire0 : ∀ i, 0 ≤ rise fireRoom i)
    (hrise : ∀ j, j ≠ fireRoom → ∀ i,
      rise j i = fedRoomRise (inRise (feeds j) rise i) (inflow (feeds j) rise z i) (AT j)) :
    ∀ j, Monotone (rise j) ∧ ∀ i, 0 ≤ rise j i := by
  have key : ∀ n, ∀ j, pos j = n → Monotone (rise j) ∧ ∀ i, 0 ≤ rise j i := by
    intro n
    induction n using Nat.strong_induction_on with
    | _ n ih =>
      intro j hj
      by_cases hjf : j = fireRoom
      · rw [hjf]
        exact ⟨hfire, hfire0⟩
      · have hsrc : ∀ f ∈ feeds j, Monotone (rise f.src) := fun f hf =>
          (ih (pos f.src) (by rw [← hj]; exact hfeed j f hf) f.src rfl).1
        have hin : Monotone (inRise (feeds j) rise) :=
          foldl_max_monotone (feeds j) (fun f i => rise f.src i) hsrc (fun _ => 0) monotone_const
        have hfl : Monotone (inflow (feeds j) rise z) :=
          sum_map_monotone (feeds j)
            (fun f i => doorOutflow f.width f.height (Ta + rise f.src i) (z f.src i))
            (fun f hf i i' hii' => doorOutflow_mono (hW j f hf)
              (by linarith [hsrc f hf hii']) (hz f.src hii'))
        refine ⟨fun i i' hii' => ?_, fun i => ?_⟩
        · rw [hrise j hjf i, hrise j hjf i']
          exact fedRoomRise_mono (hAT j) (hin hii') (hfl hii')
        · rw [hrise j hjf i]
          exact fedRoomRise_nonneg (hAT j) _ _
  exact fun j => key (pos j) j rfl

/-- **P1 to P3 for `simulateRooms`.** The fire room's rise is `fireRoomHistory`'s (P1), every
other room's is (R1)-(R3) of the rooms before it in `pos`, and every room's interface follows
(S3) from its height `H j ≥ 0` with any inflow, outflow and density. Then every room's rise
never falls and is `≥ 0`, and every interface stays in `[0, H j]` and never rises. -/
theorem zone_rooms_monotone {R : Type*} (pos : R → ℕ) (fireRoom : R) (feeds : R → List (Feed R))
    (AT H Af : R → ℝ) (dt : ℝ) (F : Fire) (L : Lining) (hG : F.Good L) (hdt : 0 ≤ dt)
    (rise z mIn mOut ρU : R → ℕ → ℝ) (hfeed : ∀ j, ∀ f ∈ feeds j, pos f.src < pos j)
    (hW : ∀ j, ∀ f ∈ feeds j, 0 ≤ f.width) (hAT : ∀ j, 0 ≤ AT j) (hH : ∀ j, 0 ≤ H j)
    (hz0 : ∀ j, z j 0 = H j)
    (hzs : ∀ j i, z j (i + 1) = layerStep (z j i) (mIn j i) (mOut j i) (ρU j i) (Af j) (H j) dt)
    (hfire : ∀ i, rise fireRoom i = fireRoomRise F L dt i)
    (hrise : ∀ j, j ≠ fireRoom → ∀ i,
      rise j i = fedRoomRise (inRise (feeds j) rise i) (inflow (feeds j) rise z i) (AT j)) :
    ∀ j, Monotone (rise j) ∧ (∀ i, 0 ≤ rise j i) ∧ Antitone (z j) ∧
      ∀ i, 0 ≤ z j i ∧ z j i ≤ H j := by
  have hzl := fun j => layer_antitone (hH j) (mIn j) (mOut j) (ρU j) (Af j) dt (hz0 j) (hzs j)
  have hr := rooms_rise_monotone pos fireRoom feeds AT rise z hfeed hW hAT (fun j => (hzl j).2)
    (by rw [show rise fireRoom = fireRoomRise F L dt from funext hfire]
        exact fireRoomRise_monotone hG hdt)
    (fun i => by rw [hfire]; exact fireRoomRise_nonneg hG hdt i) hrise
  exact fun j => ⟨(hr j).1, (hr j).2, (hzl j).2, (hzl j).1⟩

/-! ## P4. Nodes: temperatures that never fall, untenability an up-set -/

/-- (Section 5) `inUpperLayer(z, H, zHead)`: the interface at or below head height and below
the ceiling. -/
def InUpper (z H zHead : ℝ) : Prop := z ≤ zHead ∧ z < H

/-- (N1) `nodeTemperatureC`: the layer's temperature in °C when the node is in it, ambient
otherwise. -/
noncomputable def nodeTemperatureC (z H rise zHead : ℝ) : ℝ :=
  (if z ≤ zHead ∧ z < H then Ta + rise else Ta) - kelvin

/-- Once in the upper layer, a node stays in it: the interface never rises. -/
theorem inUpper_upset {z : ℕ → ℝ} (hz : Antitone z) {H zHead : ℝ} {i j : ℕ} (hij : i ≤ j)
    (h : InUpper (z i) H zHead) : InUpper (z j) H zHead :=
  ⟨(hz hij).trans h.1, (hz hij).trans_lt h.2⟩

/-- **P4.** A node's temperature never falls, given its room's interface never rises and its
rise never falls and is `≥ 0` (P1 to P3). -/
theorem nodeTemperature_monotone {z rise : ℕ → ℝ} (hz : Antitone z) (hr : Monotone rise)
    (hr0 : ∀ i, 0 ≤ rise i) (H zHead : ℝ) :
    Monotone fun i => nodeTemperatureC (z i) H (rise i) zHead := by
  intro i j hij
  simp only [nodeTemperatureC]
  split_ifs with h1 h2 h2
  · linarith [hr hij]
  · exact absurd (inUpper_upset hz hij h1) h2
  · linarith [hr0 j]
  · exact le_rfl

/-- `buildZoneForecast`'s untenability test at a step: in the upper layer and either the smoke
criterion is on or the temperature has reached `θ` (`untenableC`). -/
def NodeUntenable (z H rise zHead θ : ℝ) (smoke : Bool) : Prop :=
  InUpper z H zHead ∧ (smoke = true ∨ θ ≤ nodeTemperatureC z H rise zHead)

/-- **P4.** Untenability is an up-set in time: a node untenable at step `i` is untenable at
every later step. -/
theorem untenable_upset {z rise : ℕ → ℝ} (hz : Antitone z) (hr : Monotone rise)
    (hr0 : ∀ i, 0 ≤ rise i) {H zHead θ : ℝ} {smoke : Bool} {i j : ℕ} (hij : i ≤ j)
    (h : NodeUntenable (z i) H (rise i) zHead θ smoke) :
    NodeUntenable (z j) H (rise j) zHead θ smoke := by
  refine ⟨inUpper_upset hz hij h.1, ?_⟩
  rcases h.2 with hs | hθ
  · exact Or.inl hs
  · exact Or.inr (hθ.trans (nodeTemperature_monotone hz hr hr0 H zHead hij))

/-- The stored running maximum (`hottest`) of a node's temperature is the temperature itself:
`Forecast.runMax_of_monotone`. -/
theorem nodeTemperature_runMax {z rise : ℕ → ℝ} (hz : Antitone z) (hr : Monotone rise)
    (hr0 : ∀ i, 0 ≤ rise i) (H zHead : ℝ) (k : ℕ) :
    runMax (fun i => nodeTemperatureC (z i) H (rise i) zHead) k =
      nodeTemperatureC (z k) H (rise k) zHead :=
  runMax_of_monotone (nodeTemperature_monotone hz hr hr0 H zHead) k

/-- **P4, the forecast hypothesis.** A node's FED rate never falls: the heat the routers of
`forecast.js` and `deadline.js` read never falls, which is what makes them exact and the
deadline map conservative. -/
theorem nodeRate_monotone {z rise : ℕ → ℝ} (hz : Antitone z) (hr : Monotone rise)
    (hr0 : ∀ i, 0 ≤ rise i) (H zHead : ℝ) :
    Monotone fun i => doseRate (nodeTemperatureC (z i) H (rise i) zHead) :=
  doseRate_monotone.comp (nodeTemperature_monotone hz hr hr0 H zHead)

/-- The indicator of an untenability predicate, as a stored heat with threshold 1. -/
noncomputable def untenableInd (U : ℕ → Prop) (i : ℕ) : ℝ :=
  open Classical in if U i then 1 else 0

theorem untenableInd_monotone {U : ℕ → Prop} (hU : ∀ i j, i ≤ j → U i → U j) :
    Monotone (untenableInd U) := by
  intro i j hij
  unfold untenableInd
  split_ifs with h1 h2 h2
  · exact le_rfl
  · exact absurd (hU i j hij h1) h2
  · norm_num
  · exact le_rfl

/-- **P4, closing times that never reopen.** For an untenability that is an up-set, the
closing step `closeStep` (the first untenable step before the horizon `K0`) is exact: before
the horizon the node is untenable at step `k` iff `k` is at or after it
(`Forecast.closed_iff`). -/
theorem zone_closed_iff {U : ℕ → Prop} (hU : ∀ i j, i ≤ j → U i → U j) {K0 k : ℕ}
    (hk : k < K0) : U k ↔ closeStep (untenableInd U) 1 K0 ≤ k := by
  rw [← closed_iff (untenableInd_monotone hU) 1 hk]
  by_cases h : U k <;> simp [untenableInd, h]

/-- The node untenability of `buildZoneForecast` satisfies `zone_closed_iff`. -/
theorem node_closed_iff {z rise : ℕ → ℝ} (hz : Antitone z) (hr : Monotone rise)
    (hr0 : ∀ i, 0 ≤ rise i) (H zHead θ : ℝ) (smoke : Bool) {K0 k : ℕ} (hk : k < K0) :
    NodeUntenable (z k) H (rise k) zHead θ smoke ↔
      closeStep (untenableInd fun i => NodeUntenable (z i) H (rise i) zHead θ smoke) 1 K0 ≤ k :=
  zone_closed_iff (fun _ _ hij h => untenable_upset hz hr hr0 hij h) hk

/-! ## P5. Units (`Dim`) -/

/-- (S2) `C_d W_d ρ_a √(2 g (1 - T_a/T_U)) (H_d - z)^(3/2)` is a mass flow (`C_d`
dimensionless, `T_a / T_U` dimensionless). -/
theorem S2_units :
    Dim.length * Dim.density * Dim.rpow (Dim.accel * (Dim.temp / Dim.temp)) 1 2 *
        Dim.rpow Dim.length 3 2 = Dim.massFlow ∧
      Dim.exactPow (Dim.accel * (Dim.temp / Dim.temp)) 1 2 ∧ Dim.exactPow Dim.length 3 2 := by
  decide

/-- (S3) `Δt ṁ / (ρ_U A_f)` is a length, so `z - fall` is. -/
theorem S3_units : Dim.time * Dim.massFlow / (Dim.density * Dim.area) = Dim.length := by
  decide

/-- (R3) `h_c A_T / (ṁ c_p)` is dimensionless, so `ΔT_in / (1 + h_c A_T / (ṁ c_p))` is a
temperature; both terms of the specification's `ṁ c_p + h_c A_T` are W K⁻¹. -/
theorem R3_units :
    Dim.htc * Dim.area / (Dim.massFlow * Dim.specificHeat) = Dim.one ∧
      Dim.massFlow * Dim.specificHeat = Dim.power / Dim.temp ∧
      Dim.htc * Dim.area = Dim.power / Dim.temp := by
  decide

/-- `353 / T` is a density when 353 carries kg m⁻³ K. -/
theorem gasDensity_units : Dim.density * Dim.temp / Dim.temp = Dim.density := by decide

/-- **P5.** The units of (S2), (S3), (R3) and the gas density close, and (F1), (S1), (F2) and
`h_k`, `t_p` are FD-01, FD-08, FD-17 and FD-18 of `FireDynamics`. -/
theorem P5_units :
    (Dim.length * Dim.density * Dim.rpow (Dim.accel * (Dim.temp / Dim.temp)) 1 2 *
        Dim.rpow Dim.length 3 2 = Dim.massFlow) ∧
      Dim.time * Dim.massFlow / (Dim.density * Dim.area) = Dim.length ∧
      Dim.htc * Dim.area / (Dim.massFlow * Dim.specificHeat) = Dim.one ∧
      (FireDynamics.dAlpha * Dim.time * Dim.time = Dim.power) ∧
      (FireDynamics.d0071 * Dim.rpow Dim.power 1 3 * Dim.rpow Dim.length 5 3 = Dim.massFlow) ∧
      (FireDynamics.dMQH * Dim.rpow (Dim.power * Dim.power /
        (Dim.area * Dim.rpow Dim.length 1 2 * Dim.htc * Dim.area)) 1 3 = Dim.temp) ∧
      (Dim.conductivity / Dim.length = Dim.htc) :=
  ⟨S2_units.1, S3_units, R3_units.1, FireDynamics.FD01_units.1, FireDynamics.FD08_units.1,
    FireDynamics.FD17_units.1, FireDynamics.FD18_units.2.2.1⟩

end FireEgress.Zone

/-! ## Axioms -/

#print axioms FireEgress.Zone.Ta_pos
#print axioms FireEgress.Zone.rhoAmbient_pos
#print axioms FireEgress.Zone.flashover_le_post
#print axioms FireEgress.Zone.heatRelease_eq_tsq
#print axioms FireEgress.Zone.heatRelease_monotoneOn
#print axioms FireEgress.Zone.heatRelease_zero_le
#print axioms FireEgress.Zone.pos_of_heatRelease_pos
#print axioms FireEgress.Zone.liningConductance_eq
#print axioms FireEgress.Zone.liningConductance_antitoneOn
#print axioms FireEgress.Zone.liningConductance_pos
#print axioms FireEgress.Zone.mqh_nonneg
#print axioms FireEgress.Zone.mqhRise_nonneg
#print axioms FireEgress.Zone.mqhRise_mono_Q
#print axioms FireEgress.Zone.mqhRise_anti_hk
#print axioms FireEgress.Zone.rawRise_nonneg
#print axioms FireEgress.Zone.rawRise_monotoneOn
#print axioms FireEgress.Zone.latch_monotone
#print axioms FireEgress.Zone.fireRoomRise_monotone
#print axioms FireEgress.Zone.fireRoomRise_nonneg
#print axioms FireEgress.Zone.layerStep_le_self
#print axioms FireEgress.Zone.layerStep_mem
#print axioms FireEgress.Zone.layer_antitone
#print axioms FireEgress.Zone.doorOutflow_nonneg
#print axioms FireEgress.Zone.doorOutflow_mono
#print axioms FireEgress.Zone.fedRoomRise_eq_spec
#print axioms FireEgress.Zone.fedRoomRise_nonneg
#print axioms FireEgress.Zone.fedRoomRise_mono
#print axioms FireEgress.Zone.foldl_max_monotone
#print axioms FireEgress.Zone.sum_map_monotone
#print axioms FireEgress.Zone.rooms_rise_monotone
#print axioms FireEgress.Zone.zone_rooms_monotone
#print axioms FireEgress.Zone.inUpper_upset
#print axioms FireEgress.Zone.nodeTemperature_monotone
#print axioms FireEgress.Zone.untenable_upset
#print axioms FireEgress.Zone.nodeTemperature_runMax
#print axioms FireEgress.Zone.nodeRate_monotone
#print axioms FireEgress.Zone.untenableInd_monotone
#print axioms FireEgress.Zone.zone_closed_iff
#print axioms FireEgress.Zone.node_closed_iff
#print axioms FireEgress.Zone.S2_units
#print axioms FireEgress.Zone.S3_units
#print axioms FireEgress.Zone.R3_units
#print axioms FireEgress.Zone.gasDensity_units
#print axioms FireEgress.Zone.P5_units
