import Mathlib
import FireEgress.Dim

/-!
# Fire dynamics correlations behind Studio's fire layer (FD-01 to FD-36)

The literature review (`wiki/research/fire-egress-methods-2026-10-27.md`, sections 3.1-3.4
and 8.4) collects the hand-calculation fire correlations Studio would use: the t-squared
design fire, three flashover estimates, Heskestad's plume, McCaffrey's plume regions,
Alpert's and CFAST's ceiling jets, MQH, and the CFAST two-zone equations. The equation
ids (FD-nn) are those of the adversarially checked extraction.

Two kinds of statement are proved.

* **Dimensions** (section `Units`, all by `decide` on `FireEgress.Dim`). Empirical
  constants are not dimensionless: each one carries the unit the correlation implies,
  and the statement names that unit. Where a dimensional constant is the reduction of a
  dimensionless form (CFAST's `Q*`, Evans's `ΔT*`, `U*`, MQH's `X₁`, `X₂`), we also
  prove that the ambient-property group it absorbed has exactly that unit. That second
  check is not circular: it compares the note's unit with an independent derivation.
* **Mathematics** (real analysis). Each correlation is a real function and the
  properties Studio relies on (monotonicity in `Q`, decrease with height or radius,
  convexity, the inverse that gives a flashover time, the zone-model energy balance)
  are proved for it, with the constants as printed. Numerical claims about constants
  (rounding, continuity jumps at the branch points) are proved as rational bounds.

Units in the real-number theorems: `Q` in kW, lengths in m, times in s, temperatures in K,
as in the sources. The real theorems hold for the numbers; the `Units` section is what
says those numbers mean the stated units.

Graph numerics (FD-27, FD-31 to FD-33) are in `GraphNumerics.lean`, and Classic's
overshoot (FD-34) is in `Temperature.lean`.
-/

namespace FireEgress.FireDynamics

open Real

/-! ## Helpers for numerical bounds on fractional powers -/

/-- If `lo ^ n < x < hi ^ n` with `x, hi ≥ 0`, then `lo < x ^ (1/n) < hi`. -/
lemma root_bounds {x lo hi : ℝ} (n : ℕ) (hn : n ≠ 0) (hx : 0 ≤ x) (hhi : 0 ≤ hi)
    (h1 : lo ^ n < x) (h2 : x < hi ^ n) :
    lo < x ^ ((1 : ℝ) / n) ∧ x ^ ((1 : ℝ) / n) < hi := by
  have hy : (x ^ ((1 : ℝ) / n)) ^ n = x := by
    rw [one_div]; exact Real.rpow_inv_natCast_pow hx hn
  have hy0 : 0 ≤ x ^ ((1 : ℝ) / n) := Real.rpow_nonneg hx _
  exact ⟨lt_of_pow_lt_pow_left₀ n hy0 (by rw [hy]; exact h1),
    lt_of_pow_lt_pow_left₀ n hhi (by rw [hy]; exact h2)⟩

/-- `x ^ (p/q) = (x ^ p) ^ (1/q)` for `x ≥ 0`. -/
lemma rpow_frac {x : ℝ} (hx : 0 ≤ x) (p q : ℕ) :
    x ^ ((p : ℝ) / q) = (x ^ p) ^ ((1 : ℝ) / q) := by
  rw [← Real.rpow_natCast x p, ← Real.rpow_mul hx]
  congr 1; ring

/-- `(x ^ y) ^ n = x ^ (y n)` for `x ≥ 0`. -/
lemma rpow_pow {x : ℝ} (hx : 0 ≤ x) (y : ℝ) (n : ℕ) : (x ^ y) ^ n = x ^ (y * n) :=
  (Real.rpow_mul_natCast hx y n).symm

/-! ## FD-01, FD-02. The t-squared design fire -/

/-- FD-01. t-squared heat release rate `Q(t) = α (t - t₀)²` (kW), as written in the note
(no clamp before `t₀`). -/
def tsq (α t0 t : ℝ) : ℝ := α * (t - t0) ^ 2

/-- FD-01 with zero heat release before ignition: `Q(t) = α (max (t - t₀) 0)²`. -/
noncomputable def tsqClamped (α t0 t : ℝ) : ℝ := α * (max (t - t0) 0) ^ 2

/-- FD-01. With `α = Q_ref / t_g²` the fire reaches `Q_ref` exactly `t_g` seconds after
`t₀` (this is what defines the growth time). -/
theorem FD01_reaches_ref (Qref tg t0 : ℝ) (htg : tg ≠ 0) :
    tsq (Qref / tg ^ 2) t0 (t0 + tg) = Qref := by
  unfold tsq; field_simp; ring

/-- FD-01. For `α > 0`, heat release is strictly increasing after `t₀`. -/
theorem FD01_strictMonoOn (α t0 : ℝ) (hα : 0 < α) :
    StrictMonoOn (tsq α t0) (Set.Ici t0) := by
  intro s hs t _ hst
  simp only [Set.mem_Ici] at hs
  unfold tsq
  have : (s - t0) ^ 2 < (t - t0) ^ 2 :=
    pow_lt_pow_left₀ (by linarith) (by linarith) (by norm_num)
  exact mul_lt_mul_of_pos_left this hα

/-- FD-01. For `α ≥ 0`, heat release is convex in `t` (on all of `ℝ`): the growth
accelerates. The gap is `α a b (x - y)²`. -/
theorem FD01_convexOn (α t0 : ℝ) (hα : 0 ≤ α) : ConvexOn ℝ Set.univ (tsq α t0) := by
  refine ⟨convex_univ, ?_⟩
  intro x _ y _ a b ha hb hab
  simp only [tsq, smul_eq_mul]
  have hb' : b = 1 - a := by linarith
  subst hb'
  have key : a * (α * (x - t0) ^ 2) + (1 - a) * (α * (y - t0) ^ 2)
      - α * (a * x + (1 - a) * y - t0) ^ 2 = α * (a * (1 - a) * (x - y) ^ 2) := by ring
  have : 0 ≤ α * (a * (1 - a) * (x - y) ^ 2) :=
    mul_nonneg hα (mul_nonneg (mul_nonneg ha hb) (sq_nonneg _))
  linarith

/-- FD-01, clamped form. For `α ≥ 0`, `α (max (t - t₀) 0)²` is monotone on all of `ℝ`. -/
theorem FD01_clamped_monotone (α t0 : ℝ) (hα : 0 ≤ α) : Monotone (tsqClamped α t0) := by
  intro s t hst
  unfold tsqClamped
  have h1 : max (s - t0) 0 ≤ max (t - t0) 0 := max_le_max (by linarith) le_rfl
  exact mul_le_mul_of_nonneg_left (pow_le_pow_left₀ (le_max_right _ _) h1 2) hα

/-- FD-01, clamped form. For `α ≥ 0`, `α (max (t - t₀) 0)²` is convex on all of `ℝ`, so
the "zero, then t-squared" fire has no kink that breaks convexity at `t₀`. -/
theorem FD01_clamped_convexOn (α t0 : ℝ) (hα : 0 ≤ α) :
    ConvexOn ℝ Set.univ (tsqClamped α t0) := by
  refine ⟨convex_univ, ?_⟩
  intro x _ y _ a b ha hb hab
  simp only [tsqClamped, smul_eq_mul]
  set u := max (x - t0) 0
  set v := max (y - t0) 0
  have hu : 0 ≤ u := le_max_right _ _
  have hv : 0 ≤ v := le_max_right _ _
  have hp0 : 0 ≤ max (a * x + b * y - t0) 0 := le_max_right _ _
  have hp : max (a * x + b * y - t0) 0 ≤ a * u + b * v := by
    apply max_le
    · have h1 : x - t0 ≤ u := le_max_left _ _
      have h2 : y - t0 ≤ v := le_max_left _ _
      have : a * x + b * y - t0 = a * (x - t0) + b * (y - t0) := by
        have : t0 = (a + b) * t0 := by rw [hab]; ring
        linarith [this]
      rw [this]
      exact add_le_add (mul_le_mul_of_nonneg_left h1 ha) (mul_le_mul_of_nonneg_left h2 hb)
    · positivity
  have hsq : (max (a * x + b * y - t0) 0) ^ 2 ≤ (a * u + b * v) ^ 2 :=
    pow_le_pow_left₀ hp0 hp 2
  have hb' : b = 1 - a := by linarith
  have hconv : (a * u + b * v) ^ 2 ≤ a * u ^ 2 + b * v ^ 2 := by
    rw [hb']
    have : a * u ^ 2 + (1 - a) * v ^ 2 - (a * u + (1 - a) * v) ^ 2 =
        a * (1 - a) * (u - v) ^ 2 := by ring
    have : 0 ≤ a * (1 - a) * (u - v) ^ 2 :=
      mul_nonneg (mul_nonneg ha (by linarith)) (sq_nonneg _)
    linarith
  calc α * (max (a * x + b * y - t0) 0) ^ 2 ≤ α * (a * u ^ 2 + b * v ^ 2) :=
        mul_le_mul_of_nonneg_left (le_trans hsq hconv) hα
    _ = a * (α * u ^ 2) + b * (α * v ^ 2) := by ring

/-- FD-02. The growth coefficients are `1055 / t_g²` with `t_g = 600, 300, 150, 75 s`,
i.e. exactly `1055/360000`, `1055/90000`, `1055/22500`, `1055/5625` kW s⁻², and the
note's 0.00293, 0.0117, 0.0469, 0.188 are correct 3-significant-figure roundings (each
within half a unit of its last digit). -/
theorem FD02_growth_classes :
    (1055 : ℝ) / 600 ^ 2 = 1055 / 360000 ∧ (1055 : ℝ) / 300 ^ 2 = 1055 / 90000 ∧
    (1055 : ℝ) / 150 ^ 2 = 1055 / 22500 ∧ (1055 : ℝ) / 75 ^ 2 = 1055 / 5625 ∧
    |(1055 : ℝ) / 360000 - 0.00293| ≤ 0.000005 ∧ |(1055 : ℝ) / 90000 - 0.0117| ≤ 0.00005 ∧
    |(1055 : ℝ) / 22500 - 0.0469| ≤ 0.00005 ∧ |(1055 : ℝ) / 5625 - 0.188| ≤ 0.0005 := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, ?_, ?_, ?_, ?_⟩ <;>
    rw [abs_le] <;> constructor <;> norm_num

/-! ## FD-03 to FD-05. Flashover estimates -/

/-- FD-03. The time to flashover `t_fo = t₀ + √(Q_fo / α)` is when the t-squared fire
reaches `Q_fo`. -/
theorem FD03_tfo_reaches (α t0 Qfo : ℝ) (hα : 0 < α) (hQ : 0 ≤ Qfo) :
    tsq α t0 (t0 + √(Qfo / α)) = Qfo := by
  unfold tsq
  rw [add_sub_cancel_left, Real.sq_sqrt (div_nonneg hQ hα.le)]
  field_simp

/-- FD-03. For `Q_fo ≥ 0`, `t_fo` is the only time at or after `t₀` with `Q(t) = Q_fo`
(for `Q_fo < 0` there is none, so the hypothesis is needed). -/
theorem FD03_tfo_unique (α t0 Qfo t : ℝ) (hα : 0 < α) (hQ : 0 ≤ Qfo) (ht : t0 ≤ t) :
    tsq α t0 t = Qfo ↔ t = t0 + √(Qfo / α) := by
  unfold tsq
  constructor
  · intro h
    have : (t - t0) ^ 2 = Qfo / α := by rw [← h]; field_simp
    rw [← this, Real.sqrt_sq (by linarith)]; ring
  · intro h
    rw [h, add_sub_cancel_left, Real.sq_sqrt (div_nonneg hQ hα.le)]
    field_simp

/-- FD-03. Babrauskas's minimum flashover heat release `Q_fo = 750 A_o √H_o` (kW; the
750 carries kW m⁻⁵ᐟ²). -/
noncomputable def qfoBabrauskas (Ao Ho : ℝ) : ℝ := 750 * Ao * √Ho

/-- FD-03. `Q_fo` grows with the door area and the door height. -/
theorem FD03_babrauskas_monotone (Ao Ao' Ho Ho' : ℝ) (hA : 0 ≤ Ao) (hAA : Ao ≤ Ao')
    (hH : Ho ≤ Ho') : qfoBabrauskas Ao Ho ≤ qfoBabrauskas Ao' Ho' := by
  unfold qfoBabrauskas
  have h1 : √Ho ≤ √Ho' := Real.sqrt_le_sqrt hH
  have h2 : 0 ≤ √Ho := Real.sqrt_nonneg _
  have := mul_le_mul hAA h1 h2 (by linarith)
  nlinarith

/-- FD-03, worked example. A 1.8 m² door 2 m high gives `1909 < Q_fo < 1910` kW, and a
fast fire (`α = 1055/150²`) reaches it between 201 and 202 s after `t₀`. -/
theorem FD03_worked_example :
    1909 < qfoBabrauskas 1.8 2 ∧ qfoBabrauskas 1.8 2 < 1910 ∧
    201 < √(qfoBabrauskas 1.8 2 / (1055 / 150 ^ 2)) ∧
    √(qfoBabrauskas 1.8 2 / (1055 / 150 ^ 2)) < 202 := by
  have hs1 : (1.41421 : ℝ) < √2 := by rw [Real.lt_sqrt (by norm_num)]; norm_num
  have hs2 : √2 < (1.41422 : ℝ) := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
  have hq1 : 1909 < qfoBabrauskas 1.8 2 := by unfold qfoBabrauskas; nlinarith
  have hq2 : qfoBabrauskas 1.8 2 < 1910 := by unfold qfoBabrauskas; nlinarith
  refine ⟨hq1, hq2, ?_, ?_⟩
  · rw [Real.lt_sqrt (by norm_num), lt_div_iff₀ (by norm_num)]; nlinarith
  · rw [Real.sqrt_lt' (by norm_num), div_lt_iff₀ (by norm_num)]; nlinarith

/-- FD-17. MQH upper-layer temperature rise
`ΔT_g = c (Q² / (A_o √H_o h_k A_T))^(1/3)` (K), with the constant `c` a parameter
(6.85 K^(2/3) kW^(-1/3) m^(5/6) in the note). -/
noncomputable def mqh (c Q Ao Ho hk AT : ℝ) : ℝ :=
  c * (Q ^ 2 / (Ao * √Ho * hk * AT)) ^ ((1 : ℝ) / 3)

/-- FD-04 / FD-17. Inverting MQH: with `c > 0` and positive geometry, the heat release
`Q = √((ΔT/c)³) √(A_o √H_o h_k A_T)` produces exactly the rise `ΔT ≥ 0`. This is the
form `Q_fo = K (h_k A_T A √h)^(1/2)` of Peacock's FD-04 with `K = (ΔT/c)^(3/2)`. -/
theorem FD04_mqh_inverse (c Ao Ho hk AT ΔT : ℝ) (hc : 0 < c) (hA : 0 < Ao) (hH : 0 < Ho)
    (hk0 : 0 < hk) (hAT : 0 < AT) (hT : 0 ≤ ΔT) :
    mqh c (√((ΔT / c) ^ 3) * √(Ao * √Ho * hk * AT)) Ao Ho hk AT = ΔT := by
  unfold mqh
  have hD : 0 < Ao * √Ho * hk * AT := by positivity
  have hx : 0 ≤ (ΔT / c) ^ 3 := by positivity
  rw [mul_pow, Real.sq_sqrt hx, Real.sq_sqrt hD.le, mul_div_assoc, div_self hD.ne', mul_one]
  rw [show ((1 : ℝ) / 3) = ((3 : ℕ) : ℝ)⁻¹ by norm_num,
    Real.pow_rpow_inv_natCast (div_nonneg hT hc.le) (by norm_num)]
  field_simp

/-- FD-04, finding. Inverting MQH with the note's 6.85 at Peacock's `ΔT = 575 K` gives a
coefficient `√((575/6.85)³)` between 769 and 770, not Peacock's 740. Conversely, any MQH
constant `c` whose inversion at 575 K is exactly 740 lies between 7.02 and 7.03, which is
neither 6.85 nor the `c_p = 1.0` value (about 6.98). So FD-04 and FD-17 as printed are not
the same correlation: they differ by about 4% in `Q_fo`. -/
theorem FD04_740_vs_685 :
    (769 : ℝ) < √((575 / 6.85) ^ 3) ∧ √((575 / 6.85 : ℝ) ^ 3) < 770 ∧
    ∀ c : ℝ, 0 < c → √((575 / c) ^ 3) = 740 → 7.02 < c ∧ c < 7.03 := by
  refine ⟨by rw [Real.lt_sqrt (by norm_num)]; norm_num,
    by rw [Real.sqrt_lt' (by norm_num)]; norm_num, ?_⟩
  intro c hc h
  have h2 : (575 / c) ^ 3 = 740 ^ 2 := by
    rw [← h, Real.sq_sqrt (by positivity)]
  have h3 : c ^ 3 = 575 ^ 3 / 740 ^ 2 := by
    rw [div_pow] at h2; field_simp at h2 ⊢; linarith
  constructor
  · by_contra hneg; push Not at hneg
    have : c ^ 3 ≤ (7.02 : ℝ) ^ 3 := pow_le_pow_left₀ hc.le hneg 3
    rw [h3] at this; norm_num at this
  · by_contra hneg; push Not at hneg
    have : (7.03 : ℝ) ^ 3 ≤ c ^ 3 := pow_le_pow_left₀ (by norm_num) hneg 3
    rw [h3] at this; norm_num at this

/-- FD-05. Thomas's flashover heat release `Q_fo = 7.8 A_T + 378 A √h` (kW; 7.8 in
kW m⁻², 378 in kW m⁻⁵ᐟ²). -/
noncomputable def qfoThomas (AT A h : ℝ) : ℝ := 7.8 * AT + 378 * A * √h

/-- FD-05. Thomas's `Q_fo` grows with the enclosure surface area and with the vent
factor `A √h`. -/
theorem FD05_thomas_monotone (AT AT' A A' h h' : ℝ) (hAT : AT ≤ AT') (hA : 0 ≤ A)
    (hAA : A ≤ A') (hh : h ≤ h') : qfoThomas AT A h ≤ qfoThomas AT' A' h' := by
  unfold qfoThomas
  have h1 : √h ≤ √h' := Real.sqrt_le_sqrt hh
  have h2 : 0 ≤ √h := Real.sqrt_nonneg _
  have := mul_le_mul hAA h1 h2 (by linarith)
  nlinarith

/-! ## FD-06 to FD-10. Heskestad's flame height and plume -/

/-- FD-06. Heskestad mean flame height `L = 0.235 Q^(2/5) - 1.02 D` (m). -/
noncomputable def flameHeight (Q D : ℝ) : ℝ := 0.235 * Q ^ ((2 : ℝ) / 5) - 1.02 * D

/-- FD-07. Heskestad virtual origin `z₀ = 0.083 Q^(2/5) - 1.02 D` (m). -/
noncomputable def virtualOrigin (Q D : ℝ) : ℝ := 0.083 * Q ^ ((2 : ℝ) / 5) - 1.02 * D

/-- FD-06/07. The diameter cancels: the mean flame tip is `0.152 Q^(2/5)` above the
virtual origin, so for `Q > 0` the virtual origin is always below the flame tip. -/
theorem FD07_origin_below_flame (Q D : ℝ) (hQ : 0 < Q) :
    flameHeight Q D - virtualOrigin Q D = 0.152 * Q ^ ((2 : ℝ) / 5) ∧
      virtualOrigin Q D < flameHeight Q D := by
  have h : flameHeight Q D - virtualOrigin Q D = 0.152 * Q ^ ((2 : ℝ) / 5) := by
    unfold flameHeight virtualOrigin; ring
  refine ⟨h, ?_⟩
  have : 0 < Q ^ ((2 : ℝ) / 5) := Real.rpow_pos_of_pos hQ _
  linarith

/-- FD-06. For a fixed fire diameter the flame height increases strictly with `Q ≥ 0`. -/
theorem FD06_flame_strictMono (D : ℝ) :
    StrictMonoOn (fun Q => flameHeight Q D) (Set.Ici 0) := by
  intro a ha b _ hab
  simp only [flameHeight]
  have := Real.rpow_lt_rpow (Set.mem_Ici.mp ha) hab (by norm_num : (0 : ℝ) < 2 / 5)
  linarith

/-- FD-08. Heskestad plume mass flow above the flame,
`ṁ_p = 0.071 Q_c^(1/3) s^(5/3) [1 + 0.026 Q_c^(2/3) s^(-5/3)]` with `s = z - z₀` (kg s⁻¹). -/
noncomputable def plumeMass (Qc s : ℝ) : ℝ :=
  0.071 * Qc ^ ((1 : ℝ) / 3) * s ^ ((5 : ℝ) / 3) *
    (1 + 0.026 * Qc ^ ((2 : ℝ) / 3) * s ^ (-(5 : ℝ) / 3))

/-- FD-08. Expanded, the correction term is independent of height:
`ṁ_p = 0.071 Q_c^(1/3) s^(5/3) + 0.001846 Q_c` for `s > 0`, `Q_c ≥ 0`. -/
theorem FD08_expand (Qc s : ℝ) (hQ : 0 ≤ Qc) (hs : 0 < s) :
    plumeMass Qc s = 0.071 * Qc ^ ((1 : ℝ) / 3) * s ^ ((5 : ℝ) / 3) + 0.001846 * Qc := by
  unfold plumeMass
  have h1 : Qc ^ ((1 : ℝ) / 3) * Qc ^ ((2 : ℝ) / 3) = Qc := by
    rw [← Real.rpow_add' hQ (by norm_num)]; norm_num
  have h2 : s ^ ((5 : ℝ) / 3) * s ^ (-(5 : ℝ) / 3) = 1 := by
    rw [← Real.rpow_add hs]; norm_num
  have : 0.071 * Qc ^ ((1 : ℝ) / 3) * s ^ ((5 : ℝ) / 3) *
      (1 + 0.026 * Qc ^ ((2 : ℝ) / 3) * s ^ (-(5 : ℝ) / 3)) =
      0.071 * Qc ^ ((1 : ℝ) / 3) * s ^ ((5 : ℝ) / 3) +
        0.071 * 0.026 * (Qc ^ ((1 : ℝ) / 3) * Qc ^ ((2 : ℝ) / 3)) *
          (s ^ ((5 : ℝ) / 3) * s ^ (-(5 : ℝ) / 3)) := by ring
  rw [this, h1, h2]; norm_num

/-- FD-08. For `Q_c > 0` the plume mass flow increases strictly with height above the
virtual origin. -/
theorem FD08_strictMono_height (Qc : ℝ) (hQ : 0 < Qc) :
    StrictMonoOn (plumeMass Qc) (Set.Ioi 0) := by
  intro a ha b hb hab
  simp only [Set.mem_Ioi] at ha hb
  rw [FD08_expand Qc a hQ.le ha, FD08_expand Qc b hQ.le hb]
  have h1 := Real.rpow_lt_rpow ha.le hab (by norm_num : (0 : ℝ) < 5 / 3)
  have h2 : 0 < 0.071 * Qc ^ ((1 : ℝ) / 3) := by
    have := Real.rpow_pos_of_pos hQ ((1 : ℝ) / 3); positivity
  nlinarith

/-- FD-09. Heskestad centreline excess temperature, capped at 900 K:
`ΔT₀ = min 900 (9.1 (T∞/(g c_p² ρ∞²))^(1/3) Q_c^(2/3) (z - z₀)^(-5/3))`. -/
noncomputable def plumeTemp (T g cp ρ Qc z0 z : ℝ) : ℝ :=
  min 900 (9.1 * (T / (g * cp ^ 2 * ρ ^ 2)) ^ ((1 : ℝ) / 3) * Qc ^ ((2 : ℝ) / 3) *
    (z - z0) ^ (-(5 : ℝ) / 3))

/-- FD-09. For fixed `Q_c ≥ 0`, the (capped) plume excess temperature does not increase
with height above the virtual origin. -/
theorem FD09_antitone_height (T g cp ρ Qc z0 : ℝ) (hQ : 0 ≤ Qc) (hT : 0 ≤ T) (hg : 0 ≤ g) :
    AntitoneOn (plumeTemp T g cp ρ Qc z0) (Set.Ioi z0) := by
  intro a ha b _ hab
  simp only [Set.mem_Ioi] at ha
  unfold plumeTemp
  apply min_le_min_left
  have h1 : (b - z0) ^ (-(5 : ℝ) / 3) ≤ (a - z0) ^ (-(5 : ℝ) / 3) :=
    Real.rpow_le_rpow_of_nonpos (by linarith) (by linarith) (by norm_num)
  have h0 : 0 ≤ 9.1 * (T / (g * cp ^ 2 * ρ ^ 2)) ^ ((1 : ℝ) / 3) * Qc ^ ((2 : ℝ) / 3) := by
    have : 0 ≤ T / (g * cp ^ 2 * ρ ^ 2) := div_nonneg hT (by positivity)
    have := Real.rpow_nonneg this ((1 : ℝ) / 3)
    have := Real.rpow_nonneg hQ ((2 : ℝ) / 3)
    positivity
  exact mul_le_mul_of_nonneg_left h1 h0

/-- FD-09. Below the 900 K cap the decrease is strict when `Q_c > 0` and the ambient
group is positive. -/
theorem FD09_strictAnti_uncapped (T g cp ρ Qc z0 : ℝ) (hQ : 0 < Qc) (hT : 0 < T)
    (hg : 0 < g) (hcp : 0 < cp) (hρ : 0 < ρ) :
    StrictAntiOn (fun z => 9.1 * (T / (g * cp ^ 2 * ρ ^ 2)) ^ ((1 : ℝ) / 3) *
      Qc ^ ((2 : ℝ) / 3) * (z - z0) ^ (-(5 : ℝ) / 3)) (Set.Ioi z0) := by
  intro a ha b _ hab
  simp only [Set.mem_Ioi] at ha
  have h1 : (b - z0) ^ (-(5 : ℝ) / 3) < (a - z0) ^ (-(5 : ℝ) / 3) :=
    Real.rpow_lt_rpow_of_neg (by linarith) (by linarith) (by norm_num)
  have h0 : 0 < 9.1 * (T / (g * cp ^ 2 * ρ ^ 2)) ^ ((1 : ℝ) / 3) * Qc ^ ((2 : ℝ) / 3) := by
    have := Real.rpow_pos_of_pos (show 0 < T / (g * cp ^ 2 * ρ ^ 2) by positivity)
      ((1 : ℝ) / 3)
    have := Real.rpow_pos_of_pos hQ ((2 : ℝ) / 3)
    positivity
  exact mul_lt_mul_of_pos_left h1 h0

/-- FD-09, interface. With the modified origin `z₀' = z_I - (T_u/T_l)^(3/5) (z_I - z₀)`
and an ideal-gas interface (`T_u ρ_u = T_l ρ_l`), the upper-layer formula at `z_I` equals
the lower-layer formula at `z_I`: the plume temperature is continuous across the
interface (CFAST eqs. 3.25-3.26). -/
theorem FD09_interface_continuous (Tu Tl ρu ρl g cp Qc zI z0 : ℝ) (hTu : 0 < Tu)
    (hTl : 0 < Tl) (hρl : 0 < ρl) (hg : 0 < g) (hcp : 0 < cp)
    (hz : z0 < zI) (hgas : Tu * ρu = Tl * ρl) :
    9.1 * (Tu / (g * cp ^ 2 * ρu ^ 2)) ^ ((1 : ℝ) / 3) * Qc ^ ((2 : ℝ) / 3) *
        (zI - (zI - (Tu / Tl) ^ ((3 : ℝ) / 5) * (zI - z0))) ^ (-(5 : ℝ) / 3) =
      9.1 * (Tl / (g * cp ^ 2 * ρl ^ 2)) ^ ((1 : ℝ) / 3) * Qc ^ ((2 : ℝ) / 3) *
        (zI - z0) ^ (-(5 : ℝ) / 3) := by
  set r := Tu / Tl with hr
  have hr0 : 0 < r := div_pos hTu hTl
  have hs : 0 < zI - z0 := by linarith
  have hX : 0 < Tl / (g * cp ^ 2 * ρl ^ 2) := by positivity
  have hρ : ρu = Tl * ρl / Tu := by field_simp; linarith
  have hA : Tu / (g * cp ^ 2 * ρu ^ 2) = r ^ 3 * (Tl / (g * cp ^ 2 * ρl ^ 2)) := by
    rw [hρ, hr]; field_simp
  have hA' : (Tu / (g * cp ^ 2 * ρu ^ 2)) ^ ((1 : ℝ) / 3) =
      r * (Tl / (g * cp ^ 2 * ρl ^ 2)) ^ ((1 : ℝ) / 3) := by
    rw [hA, Real.mul_rpow (by positivity) hX.le,
      show ((1 : ℝ) / 3) = ((3 : ℕ) : ℝ)⁻¹ by norm_num,
      Real.pow_rpow_inv_natCast hr0.le (by norm_num)]
  have hB : (zI - (zI - r ^ ((3 : ℝ) / 5) * (zI - z0))) ^ (-(5 : ℝ) / 3) =
      r⁻¹ * (zI - z0) ^ (-(5 : ℝ) / 3) := by
    rw [show zI - (zI - r ^ ((3 : ℝ) / 5) * (zI - z0)) = r ^ ((3 : ℝ) / 5) * (zI - z0) by
      ring, Real.mul_rpow (Real.rpow_nonneg hr0.le _) hs.le, ← Real.rpow_mul hr0.le]
    norm_num [Real.rpow_neg_one]
  rw [hA', hB]
  field_simp

/-- FD-10. Heskestad centreline velocity (uncapped branch)
`u₀ = 3.4 (g/(c_p ρ∞ T∞))^(1/3) Q_c^(1/3) (z - z₀)^(-1/3)` (m s⁻¹). -/
noncomputable def plumeVel (g cp ρ T Qc z0 z : ℝ) : ℝ :=
  3.4 * (g / (cp * ρ * T)) ^ ((1 : ℝ) / 3) * Qc ^ ((1 : ℝ) / 3) * (z - z0) ^ (-(1 : ℝ) / 3)

/-- FD-10. For fixed `Q_c ≥ 0` the centreline velocity does not increase with height. -/
theorem FD10_antitone_height (g cp ρ T Qc z0 : ℝ) (hQ : 0 ≤ Qc) (hg : 0 ≤ g)
    (hcp : 0 ≤ cp) (hρ : 0 ≤ ρ) (hT : 0 ≤ T) :
    AntitoneOn (plumeVel g cp ρ T Qc z0) (Set.Ioi z0) := by
  intro a ha b _ hab
  simp only [Set.mem_Ioi] at ha
  unfold plumeVel
  have h1 : (b - z0) ^ (-(1 : ℝ) / 3) ≤ (a - z0) ^ (-(1 : ℝ) / 3) :=
    Real.rpow_le_rpow_of_nonpos (by linarith) (by linarith) (by norm_num)
  have h0 : 0 ≤ 3.4 * (g / (cp * ρ * T)) ^ ((1 : ℝ) / 3) * Qc ^ ((1 : ℝ) / 3) := by
    have := Real.rpow_nonneg (show 0 ≤ g / (cp * ρ * T) by positivity) ((1 : ℝ) / 3)
    have := Real.rpow_nonneg hQ ((1 : ℝ) / 3)
    positivity
  exact mul_le_mul_of_nonneg_left h1 h0

/-- FD-10. The constant `3.4 · 9.1^(-1/5)` lies between 2.18 and 2.19, so CFAST's 2.2 is
this value rounded. (That the cap constant is `3.4 · 9.1^(-1/5)`, obtained by putting
`ΔT₀ = 650 K` into FD-09 and substituting into FD-10, is the review's hand derivation; only
the numerical value and the units, in `FD10_units`, are proved here.) -/
theorem FD10_umax_constant :
    2.18 < (3.4 : ℝ) / (9.1 : ℝ) ^ ((1 : ℝ) / 5) ∧
      (3.4 : ℝ) / (9.1 : ℝ) ^ ((1 : ℝ) / 5) < 2.19 := by
  obtain ⟨h1, h2⟩ := root_bounds (x := (9.1 : ℝ)) (lo := 1.553) (hi := 1.558) 5 (by norm_num)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  have h1' : (1.553 : ℝ) < (9.1 : ℝ) ^ ((1 : ℝ) / 5) := by exact_mod_cast h1
  have h2' : (9.1 : ℝ) ^ ((1 : ℝ) / 5) < 1.558 := by exact_mod_cast h2
  have hp : 0 < (9.1 : ℝ) ^ ((1 : ℝ) / 5) := by linarith
  constructor
  · rw [lt_div_iff₀ hp]; nlinarith
  · rw [div_lt_iff₀ hp]; nlinarith

/-! ## FD-11, FD-12. McCaffrey's plume regions -/

/-- FD-12 from FD-11. In any McCaffrey region `V = k Q^(1/5) x^η` with scaled height
`x = z / Q^(2/5)`, and with the buoyancy relation `V = C √(2 g z ΔT / T₀)`, the
temperature law is `2 g ΔT / T₀ = (k/C)² x^(2η - 1)`. This holds for every `η`, so the
printed temperature table follows from the velocity table and `C = 0.9`. -/
theorem FD12_from_FD11 (k C η Q z g ΔT T0 V : ℝ) (hQ : 0 < Q) (hz : 0 < z) (hC : C ≠ 0)
    (hT0 : T0 ≠ 0) (harg : 0 ≤ 2 * g * z * ΔT / T0)
    (hV : V = k * Q ^ ((1 : ℝ) / 5) * (z / Q ^ ((2 : ℝ) / 5)) ^ η)
    (hB : V = C * √(2 * g * z * ΔT / T0)) :
    2 * g * ΔT / T0 = (k / C) ^ 2 * (z / Q ^ ((2 : ℝ) / 5)) ^ (2 * η - 1) := by
  set q := Q ^ ((2 : ℝ) / 5) with hq
  have hq0 : 0 < q := Real.rpow_pos_of_pos hQ _
  set x := z / q with hx
  have hx0 : 0 < x := div_pos hz hq0
  have hQ15 : (Q ^ ((1 : ℝ) / 5)) ^ 2 = q := by
    rw [rpow_pow hQ.le]; norm_num [hq]
  have hxη : (x ^ η) ^ 2 = x ^ (2 * η) := by
    rw [rpow_pow hx0.le]; ring_nf
  have hx1 : x ^ (2 * η - 1) = x ^ (2 * η) / x := Real.rpow_sub_one hx0.ne' _
  have hV2 : V ^ 2 = C ^ 2 * (2 * g * z * ΔT / T0) := by
    rw [hB, mul_pow, Real.sq_sqrt harg]
  have hV2' : V ^ 2 = k ^ 2 * q * x ^ (2 * η) := by
    rw [hV, mul_pow, mul_pow, hQ15, hxη]
  rw [hx1]
  have hz' : z = x * q := by rw [hx]; field_simp
  have hC2 : C ^ 2 ≠ 0 := pow_ne_zero 2 hC
  have : C ^ 2 * (2 * g * z * ΔT / T0) = k ^ 2 * q * x ^ (2 * η) := by rw [← hV2, hV2']
  rw [hz'] at this
  field_simp
  field_simp at this
  linear_combination this

/-! ## FD-13, FD-14. Alpert's ceiling jet -/

/-- FD-13. Alpert's maximum ceiling-jet excess temperature (K):
`16.9 Q^(2/3) H^(-5/3)` for `r ≤ 0.18 H`, `5.38 (Q/r)^(2/3) / H` beyond. The boundary
`r/H = 0.18` is assigned to the near-axis branch. -/
noncomputable def alpertT (Q H r : ℝ) : ℝ :=
  if r ≤ 0.18 * H then 16.9 * Q ^ ((2 : ℝ) / 3) * H ^ (-(5 : ℝ) / 3)
  else 5.38 * (Q / r) ^ ((2 : ℝ) / 3) / H

/-- FD-13. The far branch evaluated at the break `r = 0.18 H` is the near branch with
constant `5.38 / 0.18^(2/3)` in place of 16.9. -/
theorem FD13_far_at_break (Q H : ℝ) (hQ : 0 ≤ Q) (hH : 0 < H) :
    5.38 * (Q / (0.18 * H)) ^ ((2 : ℝ) / 3) / H =
      (5.38 / (0.18 : ℝ) ^ ((2 : ℝ) / 3)) * Q ^ ((2 : ℝ) / 3) * H ^ (-(5 : ℝ) / 3) := by
  have h018 : 0 < (0.18 : ℝ) ^ ((2 : ℝ) / 3) := Real.rpow_pos_of_pos (by norm_num) _
  have hH23 : 0 < H ^ ((2 : ℝ) / 3) := Real.rpow_pos_of_pos hH _
  rw [Real.div_rpow hQ (by positivity), Real.mul_rpow (by norm_num) hH.le]
  have h53 : H ^ (-(5 : ℝ) / 3) = (H ^ ((2 : ℝ) / 3) * H)⁻¹ := by
    rw [show (-(5 : ℝ) / 3) = -((2 : ℝ) / 3 + 1) by ring, Real.rpow_neg hH.le,
      Real.rpow_add hH, Real.rpow_one]
  rw [h53]
  field_simp

/-- FD-13, jump bound. The equivalent near-axis constant of the far branch,
`5.38 / 0.18^(2/3)`, lies in `(16.87, 16.88)`: Alpert's two branches are not continuous
at `r/H = 0.18`; the far branch starts about 0.14% below the near branch. -/
theorem FD13_jump_bound :
    16.87 < (5.38 : ℝ) / (0.18 : ℝ) ^ ((2 : ℝ) / 3) ∧
      (5.38 : ℝ) / (0.18 : ℝ) ^ ((2 : ℝ) / 3) < 16.88 := by
  have he : (0.18 : ℝ) ^ ((2 : ℝ) / 3) = ((0.18 : ℝ) ^ 2) ^ ((1 : ℝ) / 3) := by
    have := rpow_frac (x := (0.18 : ℝ)) (by norm_num) 2 3
    norm_num at this ⊢; exact this
  obtain ⟨h1, h2⟩ := root_bounds (x := (0.18 : ℝ) ^ 2) (lo := 0.31873) (hi := 0.31890) 3
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  have h1' : (0.31873 : ℝ) < (0.18 : ℝ) ^ ((2 : ℝ) / 3) := by
    rw [he]; exact_mod_cast h1
  have h2' : (0.18 : ℝ) ^ ((2 : ℝ) / 3) < 0.31890 := by
    rw [he]; exact_mod_cast h2
  have hp : 0 < (0.18 : ℝ) ^ ((2 : ℝ) / 3) := by linarith
  constructor
  · rw [lt_div_iff₀ hp]; nlinarith
  · rw [div_lt_iff₀ hp]; nlinarith

/-- FD-13. For `Q ≥ 0` and `H > 0`, Alpert's ceiling-jet temperature is antitone in the
radius over all `r`: constant inside `0.18 H`, decreasing beyond, and the jump at the
break is downward. -/
theorem FD13_antitone_radius (Q H : ℝ) (hQ : 0 ≤ Q) (hH : 0 < H) : Antitone (alpertT Q H) := by
  have hfar : ∀ a b : ℝ, 0 < a → a ≤ b →
      5.38 * (Q / b) ^ ((2 : ℝ) / 3) / H ≤ 5.38 * (Q / a) ^ ((2 : ℝ) / 3) / H := by
    intro a b ha hab
    have : Q / b ≤ Q / a := div_le_div_of_nonneg_left hQ ha hab
    have := Real.rpow_le_rpow (div_nonneg hQ (by linarith)) this (by norm_num : (0 : ℝ) ≤ 2 / 3)
    have := mul_le_mul_of_nonneg_left this (by norm_num : (0 : ℝ) ≤ 5.38)
    exact div_le_div_of_nonneg_right this hH.le
  have hbreak : 5.38 * (Q / (0.18 * H)) ^ ((2 : ℝ) / 3) / H ≤
      16.9 * Q ^ ((2 : ℝ) / 3) * H ^ (-(5 : ℝ) / 3) := by
    rw [FD13_far_at_break Q H hQ hH]
    have hc := FD13_jump_bound.2
    have h0 : 0 ≤ Q ^ ((2 : ℝ) / 3) * H ^ (-(5 : ℝ) / 3) :=
      mul_nonneg (Real.rpow_nonneg hQ _) (Real.rpow_nonneg hH.le _)
    nlinarith
  intro a b hab
  unfold alpertT
  split_ifs with hb ha ha
  · exact le_rfl
  · exact absurd (le_trans hab hb) ha
  · push Not at hb
    exact le_trans (hfar (0.18 * H) b (by positivity) hb.le) hbreak
  · push Not at ha
    exact hfar a b (by linarith [show 0 < 0.18 * H by positivity]) hab

/-- FD-13. Alpert's ceiling-jet temperature increases with `Q` at every radius. -/
theorem FD13_monotone_Q (H r : ℝ) (hH : 0 < H) :
    MonotoneOn (fun Q => alpertT Q H r) (Set.Ici 0) := by
  intro a ha b _ hab
  simp only [Set.mem_Ici] at ha
  unfold alpertT
  split_ifs with h
  · have := Real.rpow_le_rpow ha hab (by norm_num : (0 : ℝ) ≤ 2 / 3)
    have h0 : 0 ≤ H ^ (-(5 : ℝ) / 3) := Real.rpow_nonneg hH.le _
    have := mul_le_mul_of_nonneg_right this h0
    nlinarith
  · push Not at h
    have hr0 : 0 < r := lt_of_le_of_lt (by positivity) h
    have := Real.rpow_le_rpow (div_nonneg ha hr0.le) (div_le_div_of_nonneg_right hab hr0.le)
      (by norm_num : (0 : ℝ) ≤ 2 / 3)
    have := mul_le_mul_of_nonneg_left this (by norm_num : (0 : ℝ) ≤ 5.38)
    exact div_le_div_of_nonneg_right this hH.le

/-- FD-14. Alpert's maximum ceiling-jet velocity (m s⁻¹): `0.96 (Q/H)^(1/3)` for
`r ≤ 0.15 H`, `0.195 Q^(1/3) H^(1/2) r^(-5/6)` beyond. -/
noncomputable def alpertU (Q H r : ℝ) : ℝ :=
  if r ≤ 0.15 * H then 0.96 * (Q / H) ^ ((1 : ℝ) / 3)
  else 0.195 * Q ^ ((1 : ℝ) / 3) * H ^ ((1 : ℝ) / 2) * r ^ (-(5 : ℝ) / 6)

/-- FD-14. The far branch at `r = 0.15 H` is the near branch with `0.195 · 0.15^(-5/6)`
in place of 0.96. -/
theorem FD14_far_at_break (Q H : ℝ) (hQ : 0 ≤ Q) (hH : 0 < H) :
    0.195 * Q ^ ((1 : ℝ) / 3) * H ^ ((1 : ℝ) / 2) * (0.15 * H) ^ (-(5 : ℝ) / 6) =
      (0.195 * (0.15 : ℝ) ^ (-(5 : ℝ) / 6)) * (Q / H) ^ ((1 : ℝ) / 3) := by
  rw [Real.mul_rpow (by norm_num) hH.le, Real.div_rpow hQ hH.le]
  have hH13 : 0 < H ^ ((1 : ℝ) / 3) := Real.rpow_pos_of_pos hH _
  have : H ^ ((1 : ℝ) / 2) * H ^ (-(5 : ℝ) / 6) = (H ^ ((1 : ℝ) / 3))⁻¹ := by
    rw [← Real.rpow_add hH, ← Real.rpow_neg hH.le]; norm_num
  calc 0.195 * Q ^ ((1 : ℝ) / 3) * H ^ ((1 : ℝ) / 2) *
        ((0.15 : ℝ) ^ (-(5 : ℝ) / 6) * H ^ (-(5 : ℝ) / 6))
      = 0.195 * (0.15 : ℝ) ^ (-(5 : ℝ) / 6) * Q ^ ((1 : ℝ) / 3) *
          (H ^ ((1 : ℝ) / 2) * H ^ (-(5 : ℝ) / 6)) := by ring
    _ = _ := by rw [this]; field_simp

/-- FD-14, jump bound. `0.195 · 0.15^(-5/6)` lies in `(0.947, 0.948)`, below 0.96: the
velocity branches are discontinuous at `r/H = 0.15` with a downward jump of about 1.3%. -/
theorem FD14_jump_bound :
    0.947 < (0.195 : ℝ) * (0.15 : ℝ) ^ (-(5 : ℝ) / 6) ∧
      (0.195 : ℝ) * (0.15 : ℝ) ^ (-(5 : ℝ) / 6) < 0.948 := by
  have he : (0.15 : ℝ) ^ ((5 : ℝ) / 6) = ((0.15 : ℝ) ^ 5) ^ ((1 : ℝ) / 6) := by
    have := rpow_frac (x := (0.15 : ℝ)) (by norm_num) 5 6
    norm_num at this ⊢; exact this
  have hneg : (0.15 : ℝ) ^ (-(5 : ℝ) / 6) = ((0.15 : ℝ) ^ ((5 : ℝ) / 6))⁻¹ := by
    rw [← Real.rpow_neg (by norm_num)]; norm_num
  obtain ⟨h1, h2⟩ := root_bounds (x := (0.15 : ℝ) ^ 5) (lo := 0.20570) (hi := 0.20591) 6
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  have h1' : (0.20570 : ℝ) < (0.15 : ℝ) ^ ((5 : ℝ) / 6) := by rw [he]; exact_mod_cast h1
  have h2' : (0.15 : ℝ) ^ ((5 : ℝ) / 6) < 0.20591 := by rw [he]; exact_mod_cast h2
  have hp : 0 < (0.15 : ℝ) ^ ((5 : ℝ) / 6) := by linarith
  rw [hneg, ← div_eq_mul_inv]
  constructor
  · rw [lt_div_iff₀ hp]; nlinarith
  · rw [div_lt_iff₀ hp]; nlinarith

/-- FD-14. For `Q ≥ 0` and `H > 0`, Alpert's ceiling-jet velocity is antitone in the
radius over all `r` (constant, then decreasing, with a downward jump at the break). -/
theorem FD14_antitone_radius (Q H : ℝ) (hQ : 0 ≤ Q) (hH : 0 < H) : Antitone (alpertU Q H) := by
  have hc0 : 0 ≤ 0.195 * Q ^ ((1 : ℝ) / 3) * H ^ ((1 : ℝ) / 2) :=
    mul_nonneg (mul_nonneg (by norm_num) (Real.rpow_nonneg hQ _)) (Real.rpow_nonneg hH.le _)
  have hfar : ∀ a b : ℝ, 0 < a → a ≤ b →
      0.195 * Q ^ ((1 : ℝ) / 3) * H ^ ((1 : ℝ) / 2) * b ^ (-(5 : ℝ) / 6) ≤
        0.195 * Q ^ ((1 : ℝ) / 3) * H ^ ((1 : ℝ) / 2) * a ^ (-(5 : ℝ) / 6) := by
    intro a b ha hab
    exact mul_le_mul_of_nonneg_left (Real.rpow_le_rpow_of_nonpos ha hab (by norm_num)) hc0
  have hbreak : 0.195 * Q ^ ((1 : ℝ) / 3) * H ^ ((1 : ℝ) / 2) * (0.15 * H) ^ (-(5 : ℝ) / 6) ≤
      0.96 * (Q / H) ^ ((1 : ℝ) / 3) := by
    rw [FD14_far_at_break Q H hQ hH]
    have hc := FD14_jump_bound.2
    have h0 : 0 ≤ (Q / H) ^ ((1 : ℝ) / 3) := Real.rpow_nonneg (div_nonneg hQ hH.le) _
    nlinarith
  intro a b hab
  unfold alpertU
  split_ifs with hb ha ha
  · exact le_rfl
  · exact absurd (le_trans hab hb) ha
  · push Not at hb
    exact le_trans (hfar (0.15 * H) b (by positivity) hb.le) hbreak
  · push Not at ha
    exact hfar a b (by linarith [show 0 < 0.15 * H by positivity]) hab

/-! ## FD-15. CFAST ceiling jet -/

/-- FD-15. CFAST ceiling-jet temperature rise as a function of radius (K), from the
plume rise `ΔT₀(H)` at the ceiling: `ΔT₀(H)` for `r/H ≤ 0.2`,
`0.182 ΔT₀(H) (0.225 + 0.27 r/H)^(-4/3)` beyond (stated range `r/H < 4`). -/
noncomputable def cfastCJ (dT0 H r : ℝ) : ℝ :=
  if r / H ≤ 0.2 then dT0 else 0.182 * dT0 * (0.225 + 0.27 * (r / H)) ^ (-(4 : ℝ) / 3)

/-- FD-15, jump bound. At `r/H = 0.2` the outer branch is `0.182 · 0.279^(-4/3)` times the
inner one, and this factor lies in `(0.997, 1)`: the constant 0.182 is
`0.279^(4/3) ≈ 0.18231` rounded down, so the branches meet to within 0.3%. -/
theorem FD15_jump_bound :
    0.997 < (0.182 : ℝ) * (0.279 : ℝ) ^ (-(4 : ℝ) / 3) ∧
      (0.182 : ℝ) * (0.279 : ℝ) ^ (-(4 : ℝ) / 3) < 1 := by
  have he : (0.279 : ℝ) ^ ((4 : ℝ) / 3) = ((0.279 : ℝ) ^ 4) ^ ((1 : ℝ) / 3) := by
    have := rpow_frac (x := (0.279 : ℝ)) (by norm_num) 4 3
    norm_num at this ⊢; exact this
  have hneg : (0.279 : ℝ) ^ (-(4 : ℝ) / 3) = ((0.279 : ℝ) ^ ((4 : ℝ) / 3))⁻¹ := by
    rw [← Real.rpow_neg (by norm_num)]; norm_num
  obtain ⟨h1, h2⟩ := root_bounds (x := (0.279 : ℝ) ^ 4) (lo := 0.1822) (hi := 0.1824) 3
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num)
  have h1' : (0.1822 : ℝ) < (0.279 : ℝ) ^ ((4 : ℝ) / 3) := by rw [he]; exact_mod_cast h1
  have h2' : (0.279 : ℝ) ^ ((4 : ℝ) / 3) < 0.1824 := by rw [he]; exact_mod_cast h2
  have hp : 0 < (0.279 : ℝ) ^ ((4 : ℝ) / 3) := by linarith
  rw [hneg, ← div_eq_mul_inv]
  constructor
  · rw [lt_div_iff₀ hp]; nlinarith
  · rw [div_lt_iff₀ hp]; nlinarith

/-- FD-15. For `ΔT₀(H) ≥ 0` and `H > 0`, CFAST's ceiling-jet temperature is antitone in
the radius on `r ≥ 0` (the small jump at `r/H = 0.2` is downward). -/
theorem FD15_antitone_radius (dT0 H : ℝ) (hT : 0 ≤ dT0) (hH : 0 < H) :
    AntitoneOn (cfastCJ dT0 H) (Set.Ici 0) := by
  have hc0 : 0 ≤ 0.182 * dT0 := by positivity
  have hfar : ∀ a b : ℝ, 0 ≤ a → a ≤ b →
      0.182 * dT0 * (0.225 + 0.27 * (b / H)) ^ (-(4 : ℝ) / 3) ≤
        0.182 * dT0 * (0.225 + 0.27 * (a / H)) ^ (-(4 : ℝ) / 3) := by
    intro a b ha hab
    have hab' : a / H ≤ b / H := div_le_div_of_nonneg_right hab hH.le
    have ha' : 0 ≤ a / H := div_nonneg ha hH.le
    exact mul_le_mul_of_nonneg_left
      (Real.rpow_le_rpow_of_nonpos (by positivity) (by linarith) (by norm_num)) hc0
  have hbreak : ∀ b : ℝ, 0.2 < b / H →
      0.182 * dT0 * (0.225 + 0.27 * (b / H)) ^ (-(4 : ℝ) / 3) ≤ dT0 := by
    intro b hb
    have h1 : (0.225 + 0.27 * (b / H)) ^ (-(4 : ℝ) / 3) ≤ (0.279 : ℝ) ^ (-(4 : ℝ) / 3) :=
      Real.rpow_le_rpow_of_nonpos (by norm_num) (by linarith) (by norm_num)
    have h2 := FD15_jump_bound.2
    have h3 : 0 ≤ (0.225 + 0.27 * (b / H)) ^ (-(4 : ℝ) / 3) :=
      Real.rpow_nonneg (by linarith) _
    nlinarith
  intro a ha b _ hab
  simp only [Set.mem_Ici] at ha
  unfold cfastCJ
  split_ifs with hb ha' ha'
  · exact le_rfl
  · exact absurd (le_trans (div_le_div_of_nonneg_right hab hH.le) hb) ha'
  · push Not at hb; exact hbreak b hb
  · exact hfar a b ha hab

/-! ## FD-16. CFAST weak-plume limit -/

/-- FD-16. For `ṁ_e > 0`, `c_p > 0` and `T_u > T_l`, the upper layer is cooler than the
plume, `T_u < (Q_c + ṁ_e c_p T_l)/(ṁ_e c_p)`, exactly when
`ṁ_e < Q_c / (c_p (T_u - T_l))`. The hypothesis `T_u > T_l` is what allows dividing by
`c_p (T_u - T_l)`. -/
theorem FD16_weak_plume (Qc me cp Tu Tl : ℝ) (hm : 0 < me) (hc : 0 < cp) (hT : Tl < Tu) :
    Tu < (Qc + me * cp * Tl) / (me * cp) ↔ me < Qc / (cp * (Tu - Tl)) := by
  have h1 : 0 < me * cp := mul_pos hm hc
  have h2 : 0 < cp * (Tu - Tl) := mul_pos hc (by linarith)
  rw [lt_div_iff₀ h1, lt_div_iff₀ h2]
  constructor <;> intro h <;> nlinarith

/-! ## FD-17, FD-18. MQH and the wall conductance -/

/-- FD-17. For `c > 0` and positive geometry, the MQH temperature rise is strictly
increasing in `Q ≥ 0`. -/
theorem FD17_strictMono_Q (c Ao Ho hk AT : ℝ) (hc : 0 < c) (hA : 0 < Ao) (hH : 0 < Ho)
    (hk0 : 0 < hk) (hAT : 0 < AT) :
    StrictMonoOn (fun Q => mqh c Q Ao Ho hk AT) (Set.Ici 0) := by
  intro a ha b _ hab
  simp only [Set.mem_Ici] at ha
  unfold mqh
  have hD : 0 < Ao * √Ho * hk * AT := by positivity
  have h1 : a ^ 2 / (Ao * √Ho * hk * AT) < b ^ 2 / (Ao * √Ho * hk * AT) :=
    div_lt_div_of_pos_right (pow_lt_pow_left₀ hab ha (by norm_num)) hD
  exact mul_lt_mul_of_pos_left
    (Real.rpow_lt_rpow (by positivity) h1 (by norm_num)) hc

/-- FD-17. For `c ≥ 0`, a larger effective conductance `h_k > 0` never raises the MQH
temperature. (`A_T` and `A_o` enter the same denominator, so the same argument applies to
them; only the `h_k` case is stated.) -/
theorem FD17_antitone_hk (c Q Ao Ho AT : ℝ) (hc : 0 ≤ c) (hA : 0 < Ao) (hH : 0 < Ho)
    (hAT : 0 < AT) : AntitoneOn (fun hk => mqh c Q Ao Ho hk AT) (Set.Ioi 0) := by
  intro a ha b _ hab
  simp only [Set.mem_Ioi] at ha
  have hb0 : 0 < b := lt_of_lt_of_le ha hab
  unfold mqh
  have hDa : 0 < Ao * √Ho * a * AT := by positivity
  have hle : Q ^ 2 / (Ao * √Ho * b * AT) ≤ Q ^ 2 / (Ao * √Ho * a * AT) := by
    apply div_le_div_of_nonneg_left (sq_nonneg Q) hDa
    have : 0 < Ao * √Ho := by positivity
    nlinarith
  exact mul_le_mul_of_nonneg_left
    (Real.rpow_le_rpow (div_nonneg (sq_nonneg Q) (by positivity)) hle (by norm_num)) hc

/-- FD-17. The dimensional MQH form is the dimensionless one
`ΔT / T₀ = 1.63 X₁^(2/3) X₂^(-1/3)`, `X₁ = Q/(√g ρ₀ c_p T₀ A_o √H_o)`,
`X₂ = h_k A_T/(√g ρ₀ c_p A_o √H_o)`, with constant
`1.63 T₀^(1/3) (√g ρ₀ c_p)^(-1/3)`. -/
theorem FD17_dimensionless_form (Q Ao Ho hk AT g ρ cp T0 : ℝ) (hQ : 0 ≤ Q) (hA : 0 < Ao)
    (hH : 0 < Ho) (hk0 : 0 < hk) (hAT : 0 < AT) (hg : 0 < g) (hρ : 0 < ρ) (hcp : 0 < cp)
    (hT0 : 0 < T0) :
    T0 * (1.63 * (Q / (√g * ρ * cp * T0 * (Ao * √Ho))) ^ ((2 : ℝ) / 3) *
        (hk * AT / (√g * ρ * cp * (Ao * √Ho))) ^ (-(1 : ℝ) / 3)) =
      mqh (1.63 * T0 ^ ((1 : ℝ) / 3) * (√g * ρ * cp) ^ (-(1 : ℝ) / 3)) Q Ao Ho hk AT := by
  unfold mqh
  have hs : 0 < √g * ρ * cp := by positivity
  have ha : 0 < Ao * √Ho := by positivity
  have hX1 : 0 ≤ Q / (√g * ρ * cp * T0 * (Ao * √Ho)) := by positivity
  have hX2 : 0 < hk * AT / (√g * ρ * cp * (Ao * √Ho)) := by positivity
  have hD : 0 < Ao * √Ho * hk * AT := by positivity
  have hL : 0 ≤ T0 * (1.63 * (Q / (√g * ρ * cp * T0 * (Ao * √Ho))) ^ ((2 : ℝ) / 3) *
      (hk * AT / (√g * ρ * cp * (Ao * √Ho))) ^ (-(1 : ℝ) / 3)) := by
    have := Real.rpow_nonneg hX1 ((2 : ℝ) / 3)
    have := Real.rpow_nonneg hX2.le (-(1 : ℝ) / 3)
    positivity
  have hR : 0 ≤ 1.63 * T0 ^ ((1 : ℝ) / 3) * (√g * ρ * cp) ^ (-(1 : ℝ) / 3) *
      (Q ^ 2 / (Ao * √Ho * hk * AT)) ^ ((1 : ℝ) / 3) := by
    have := Real.rpow_nonneg hT0.le ((1 : ℝ) / 3)
    have := Real.rpow_nonneg hs.le (-(1 : ℝ) / 3)
    have := Real.rpow_nonneg (show 0 ≤ Q ^ 2 / (Ao * √Ho * hk * AT) by positivity) ((1 : ℝ) / 3)
    positivity
  rw [← pow_left_inj₀ hL hR (by norm_num : (3 : ℕ) ≠ 0)]
  simp only [mul_pow]
  rw [rpow_pow hX1, rpow_pow hX2.le, rpow_pow hT0.le, rpow_pow hs.le,
    rpow_pow (by positivity : (0 : ℝ) ≤ Q ^ 2 / (Ao * √Ho * hk * AT))]
  norm_num [Real.rpow_neg_one, Real.rpow_two]
  field_simp

/-- FD-17. With Quintiere's property values (`g = 9.8`, `ρ₀ = 1.2`, `c_p = 1.05`,
`T₀ = 295 K`), the MQH constant `1.63 T₀^(1/3) (√g ρ₀ c_p)^(-1/3)` lies in
`(6.86, 6.88)`: the note's 6.85 is this reduction, rounded (the unrounded value is
about 6.868). -/
theorem FD17_constant_685 :
    6.86 < 1.63 * (295 : ℝ) ^ ((1 : ℝ) / 3) * (√9.8 * 1.2 * 1.05) ^ (-(1 : ℝ) / 3) ∧
      1.63 * (295 : ℝ) ^ ((1 : ℝ) / 3) * (√9.8 * 1.2 * 1.05) ^ (-(1 : ℝ) / 3) < 6.88 := by
  have hs1 : (3.1304 : ℝ) < √9.8 := by rw [Real.lt_sqrt (by norm_num)]; norm_num
  have hs2 : √9.8 < (3.1306 : ℝ) := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
  have hs : 0 < √9.8 * 1.2 * 1.05 := by positivity
  have hcomb : (295 : ℝ) ^ ((1 : ℝ) / 3) * (√9.8 * 1.2 * 1.05) ^ (-(1 : ℝ) / 3) =
      (295 / (√9.8 * 1.2 * 1.05)) ^ ((1 : ℝ) / 3) := by
    rw [show (-(1 : ℝ) / 3) = -((1 : ℝ) / 3) by ring, Real.rpow_neg hs.le,
      Real.div_rpow (by norm_num) hs.le]
    exact (div_eq_mul_inv _ _).symm
  have hx1 : (6.86 / 1.63 : ℝ) ^ 3 < 295 / (√9.8 * 1.2 * 1.05) := by
    rw [lt_div_iff₀ hs]; nlinarith
  have hx2 : 295 / (√9.8 * 1.2 * 1.05) < (6.88 / 1.63 : ℝ) ^ 3 := by
    rw [div_lt_iff₀ hs]; nlinarith
  obtain ⟨h1, h2⟩ := root_bounds (x := 295 / (√9.8 * 1.2 * 1.05)) (lo := 6.86 / 1.63)
    (hi := 6.88 / 1.63) 3 (by norm_num) (by positivity) (by norm_num) hx1 hx2
  have h1' : (6.86 / 1.63 : ℝ) < (295 / (√9.8 * 1.2 * 1.05)) ^ ((1 : ℝ) / 3) := by
    exact_mod_cast h1
  have h2' : (295 / (√9.8 * 1.2 * 1.05)) ^ ((1 : ℝ) / 3) < 6.88 / 1.63 := by
    exact_mod_cast h2
  rw [mul_assoc, hcomb]
  constructor
  · have := mul_lt_mul_of_pos_left h1' (by norm_num : (0 : ℝ) < 1.63)
    norm_num at this ⊢; linarith
  · have := mul_lt_mul_of_pos_left h2' (by norm_num : (0 : ℝ) < 1.63)
    norm_num at this ⊢; linarith

/-- FD-18. Thermal penetration time `t_p = (ρ c / k) (δ/2)²` (s). -/
noncomputable def tPen (k ρ c δ : ℝ) : ℝ := ρ * c / k * (δ / 2) ^ 2

/-- FD-18. Effective wall conductance with the convention fixed explicitly:
`√(k ρ c / t)` for `t < t_p` (thick wall), `k / δ` for `t ≥ t_p` (thin wall). -/
noncomputable def hkEff (k ρ c δ t : ℝ) : ℝ :=
  if t < tPen k ρ c δ then √(k * ρ * c / t) else k / δ

/-- FD-18, finding. At `t = t_p` the thick-wall branch equals `2k/δ`, twice the
thin-wall branch: Quintiere's source writes both branches as inclusive at `t_p`, which is
ambiguous by a factor of 2, so the convention must be fixed (here: `t < t_p` thick). -/
theorem FD18_branches_at_tp (k ρ c δ : ℝ) (hk : 0 < k) (hρ : 0 < ρ) (hc : 0 < c)
    (hδ : 0 < δ) : √(k * ρ * c / tPen k ρ c δ) = 2 * (k / δ) := by
  unfold tPen
  have : k * ρ * c / (ρ * c / k * (δ / 2) ^ 2) = (2 * (k / δ)) ^ 2 := by
    field_simp
  rw [this, Real.sqrt_sq (by positivity)]

/-- FD-18. With positive wall properties the effective conductance never increases with
exposure time (the switch at `t_p` is a drop from `2k/δ` to `k/δ`). -/
theorem FD18_antitone_time (k ρ c δ : ℝ) (hk : 0 < k) (hρ : 0 < ρ) (hc : 0 < c)
    (hδ : 0 < δ) : AntitoneOn (hkEff k ρ c δ) (Set.Ioi 0) := by
  intro a ha b _ hab
  simp only [Set.mem_Ioi] at ha
  have hsq : ∀ s t : ℝ, 0 < s → s ≤ t → √(k * ρ * c / t) ≤ √(k * ρ * c / s) := by
    intro s t hs hst
    exact Real.sqrt_le_sqrt (div_le_div_of_nonneg_left (by positivity) hs hst)
  unfold hkEff
  split_ifs with hb ha'
  · exact hsq a b ha hab
  · exact absurd (lt_of_le_of_lt hab hb) ha'
  · push Not at hb
    have htp : 0 < tPen k ρ c δ := by unfold tPen; positivity
    calc k / δ ≤ 2 * (k / δ) := by have : 0 < k / δ := by positivity
                                   linarith
      _ = √(k * ρ * c / tPen k ρ c δ) := (FD18_branches_at_tp k ρ c δ hk hρ hc hδ).symm
      _ ≤ √(k * ρ * c / a) := hsq a _ ha (by linarith)
  · exact le_rfl

/-- FD-17 with FD-18. At fixed `Q`, the MQH layer temperature never decreases with
exposure time, because the walls' effective conductance never increases. -/
theorem FD17_monotone_time (c Q Ao Ho AT k ρ cw δ : ℝ) (hc : 0 ≤ c) (hA : 0 < Ao)
    (hH : 0 < Ho) (hAT : 0 < AT) (hk : 0 < k) (hρ : 0 < ρ) (hcw : 0 < cw) (hδ : 0 < δ) :
    MonotoneOn (fun t => mqh c Q Ao Ho (hkEff k ρ cw δ t) AT) (Set.Ioi 0) := by
  intro a ha b hb hab
  have hpos : ∀ t, 0 < t → 0 < hkEff k ρ cw δ t := by
    intro t ht
    unfold hkEff
    split_ifs
    · exact Real.sqrt_pos.mpr (by positivity)
    · positivity
  exact FD17_antitone_hk c Q Ao Ho AT hc hA hH hAT (hpos b hb) (hpos a ha)
    (FD18_antitone_time k ρ cw δ hk hρ hcw hδ ha hb hab)

/-! ## FD-19 to FD-22. CFAST two-zone model -/

/-- FD-19. CFAST's gas constant: with `c_p = 1012 J kg⁻¹ K⁻¹` and `γ = 1.4`,
`R = (γ - 1)/γ · c_p = 2024/7 ≈ 289.143`, within 0.005 of the printed 289.14 (and not the
textbook 287.05 for air). -/
theorem FD19_gas_constant :
    ((1.4 : ℝ) - 1) / 1.4 * 1012 = 2024 / 7 ∧ |(2024 / 7 : ℝ) - 289.14| < 0.005 := by
  refine ⟨by norm_num, ?_⟩
  rw [abs_lt]; constructor <;> norm_num

/-- FD-19. With `γ = c_p / c_v`, `R = (γ - 1)/γ · c_p` is the same as `R = c_p - c_v`. -/
theorem FD19_R_eq (cp cv : ℝ) (hcp : cp ≠ 0) (hcv : cv ≠ 0) :
    (cp / cv - 1) / (cp / cv) * cp = cp - cv := by
  field_simp

/-- FD-19. For an ideal gas `P V = m R T` with `R = c_p - c_v > 0` and `γ = c_p/c_v`, the
layer's internal energy is `c_v m T = P V / (γ - 1)`. This is the identity that turns the
energy equation into FD-20 and FD-21. -/
theorem FD19_internal_energy (cp cv m T P V : ℝ) (hcv : 0 < cv) (hlt : cv < cp)
    (hgas : P * V = m * (cp - cv) * T) : cv * m * T = P * V / (cp / cv - 1) := by
  have h1 : cp / cv - 1 = (cp - cv) / cv := by field_simp
  have h2 : cp - cv ≠ 0 := by linarith
  rw [h1, hgas]
  field_simp

/-- FD-20, derivation. Take two layers in a room of fixed volume `V`
(`V_u(s) + V_l(s) = V` for all `s`), each with internal energy `P V_i / (γ - 1)` and the
energy balance `d/dt (P V_i/(γ - 1)) + P dV_i/dt = ḣ_i` (FD-19). Then
`dP/dt = (γ - 1)/V · (ḣ_l + ḣ_u)`. Derivatives are taken at one instant `t`. -/
theorem FD20_pressure_equation (P Vu Vl : ℝ → ℝ) (P' Vu' Vl' Eu' El' γ V hu hl t : ℝ)
    (hγ : γ ≠ 1) (hV : V ≠ 0)
    (hP : HasDerivAt P P' t) (hVu : HasDerivAt Vu Vu' t) (hVl : HasDerivAt Vl Vl' t)
    (hvol : ∀ s, Vu s + Vl s = V)
    (hEu : HasDerivAt (fun s => P s * Vu s / (γ - 1)) Eu' t)
    (hEl : HasDerivAt (fun s => P s * Vl s / (γ - 1)) El' t)
    (hbu : Eu' + P t * Vu' = hu) (hbl : El' + P t * Vl' = hl) :
    P' = (γ - 1) / V * (hl + hu) := by
  have hγ' : γ - 1 ≠ 0 := sub_ne_zero.mpr hγ
  have e1 : Eu' = (P' * Vu t + P t * Vu') / (γ - 1) :=
    hEu.unique ((hP.mul hVu).div_const (γ - 1))
  have e2 : El' = (P' * Vl t + P t * Vl') / (γ - 1) :=
    hEl.unique ((hP.mul hVl).div_const (γ - 1))
  have hsum : Vu' + Vl' = 0 := by
    have h2 : HasDerivAt (fun s => Vu s + Vl s) 0 t := by
      have : (fun s => Vu s + Vl s) = fun _ => V := funext hvol
      rw [this]; exact hasDerivAt_const t V
    exact (hVu.add hVl).unique h2
  have hbu' : P' * Vu t + P t * Vu' + (γ - 1) * (P t * Vu') = (γ - 1) * hu := by
    rw [← hbu, e1]; field_simp
  have hbl' : P' * Vl t + P t * Vl' + (γ - 1) * (P t * Vl') = (γ - 1) * hl := by
    rw [← hbl, e2]; field_simp
  rw [div_mul_eq_mul_div, eq_div_iff hV, ← hvol t]
  linear_combination hbu' + hbl' - γ * P t * hsum

/-- FD-20, energy conservation. Under the pressure equation the room's total internal
energy `P V/(γ - 1)` (fixed `V`) changes at exactly `ḣ_l + ḣ_u`: every watt of net
enthalpy flow into the two layers goes into internal energy, none is lost or created. -/
theorem FD20_energy_conserved (P : ℝ → ℝ) (P' γ V hu hl t : ℝ) (hγ : γ ≠ 1) (hV : V ≠ 0)
    (hP : HasDerivAt P P' t) (hpe : P' = (γ - 1) / V * (hl + hu)) :
    HasDerivAt (fun s => P s * V / (γ - 1)) (hl + hu) t := by
  have hγ' : γ - 1 ≠ 0 := sub_ne_zero.mpr hγ
  refine ((hP.mul_const V).div_const (γ - 1)).congr_deriv ?_
  rw [hpe]; field_simp

/-- FD-21. The upper-layer energy balance
`(P' V_u + P V_u')/(γ - 1) + P V_u' = ḣ_u` is equivalent to CFAST's volume equation
`dV_u/dt = ((γ - 1) ḣ_u - V_u dP/dt)/(P γ)` (for `γ ≠ 0, 1` and `P ≠ 0`). -/
theorem FD21_upper_volume (P P' Vu Vu' hu γ : ℝ) (hγ1 : γ ≠ 1) (hγ0 : γ ≠ 0) (hP : P ≠ 0) :
    (P' * Vu + P * Vu') / (γ - 1) + P * Vu' = hu ↔
      Vu' = ((γ - 1) * hu - Vu * P') / (P * γ) := by
  have hγ' : γ - 1 ≠ 0 := sub_ne_zero.mpr hγ1
  rw [div_add' _ _ _ hγ', div_eq_iff hγ', eq_div_iff (mul_ne_zero hP hγ0)]
  constructor <;> intro h <;> linear_combination h

/-- FD-21. Interface height `z_i = (V - V_u)/A_f` of a prismatic room lies in
`[0, V/A_f]` when `0 ≤ V_u ≤ V`, and falls as the upper layer grows. -/
theorem FD21_interface_height (V Vu Vu' Af : ℝ) (hAf : 0 < Af) (h0 : 0 ≤ Vu) (h1 : Vu ≤ V)
    (h2 : Vu ≤ Vu') :
    0 ≤ (V - Vu) / Af ∧ (V - Vu) / Af ≤ V / Af ∧ (V - Vu') / Af ≤ (V - Vu) / Af := by
  refine ⟨div_nonneg (by linarith) hAf.le, div_le_div_of_nonneg_right (by linarith) hAf.le,
    div_le_div_of_nonneg_right (by linarith) hAf.le⟩

/-- FD-22, derivation. For a layer obeying the ideal gas law `P V = m R T` at all times,
with `R = (γ - 1)/γ · c_p` and the volume equation of FD-21, the layer temperature obeys
`dT/dt = (ḣ - c_p ṁ T + V dP/dt)/(c_p m)` (CFAST eqs. 2.7-2.8). -/
theorem FD22_layer_temperature (P V m T : ℝ → ℝ) (P' V' m' T' R γ cp h t : ℝ)
    (hP : HasDerivAt P P' t) (hV : HasDerivAt V V' t) (hm : HasDerivAt m m' t)
    (hT : HasDerivAt T T' t) (hgas : ∀ s, P s * V s = m s * R * T s)
    (hR : R = (γ - 1) / γ * cp) (hγ1 : γ ≠ 1) (hγ0 : γ ≠ 0) (hPt : P t ≠ 0)
    (hmt : m t ≠ 0) (hcp : cp ≠ 0)
    (hvol : V' = ((γ - 1) * h - V t * P') / (P t * γ)) :
    T' = (h - cp * m' * T t + V t * P') / (cp * m t) := by
  have hγ' : γ - 1 ≠ 0 := sub_ne_zero.mpr hγ1
  have hd : P' * V t + P t * V' = m' * R * T t + m t * R * T' := by
    have h1 : HasDerivAt (fun s => P s * V s) (P' * V t + P t * V') t := hP.mul hV
    have h2 : HasDerivAt (fun s => m s * R * T s) (m' * R * T t + m t * R * T') t :=
      (hm.mul_const R).mul hT
    have hf : (fun s => P s * V s) = (fun s => m s * R * T s) := funext hgas
    rw [hf] at h1
    exact h1.unique h2
  have hPV : P t * V' = ((γ - 1) * h - V t * P') / γ := by
    rw [hvol]; field_simp
  have key : (γ - 1) / γ * (cp * m t * T') = (γ - 1) / γ * (h - cp * m' * T t + V t * P') := by
    have e : m t * R * T' = P' * V t + ((γ - 1) * h - V t * P') / γ - m' * R * T t := by
      linarith
    calc (γ - 1) / γ * (cp * m t * T') = m t * R * T' := by rw [hR]; ring
      _ = P' * V t + ((γ - 1) * h - V t * P') / γ - m' * R * T t := e
      _ = (γ - 1) / γ * (h - cp * m' * T t + V t * P') := by
        rw [hR]; field_simp; ring
  have hc : (γ - 1) / γ ≠ 0 := div_ne_zero hγ' hγ0
  have := mul_left_cancel₀ hc key
  rw [eq_div_iff (mul_ne_zero hcp hmt)]
  linarith

/-! ## FD-23. Vent mass flow through a vertical opening -/

/-- FD-23. Linear (hydrostatic) cross-vent pressure-difference magnitude over a segment
from `b` (value `P_b`) to `t` (value `P_t`). -/
noncomputable def dPlin (b t Pb Pt z : ℝ) : ℝ := Pb + (Pt - Pb) * ((z - b) / (t - b))

/-- FD-23. For a single-sign segment (`P_b, P_t ≥ 0`, `P_t ≠ P_b`) the orifice integral
`∫_b^t C √(2 ρ ΔP(z)) w dz` equals the note's closed form
`(2/3) C √(2ρ) w (t - b) (P_t^(3/2) - P_b^(3/2)) / (P_t - P_b)`. -/
theorem FD23_vent_integral (C ρ w b t Pb Pt : ℝ) (hbt : b < t) (hPb : 0 ≤ Pb) (hPt : 0 ≤ Pt)
    (hne : Pt ≠ Pb) (hρ : 0 ≤ ρ) :
    ∫ z in b..t, C * √(2 * ρ * dPlin b t Pb Pt z) * w =
      2 / 3 * C * √(2 * ρ) * w * (t - b) * (Pt ^ ((3 : ℝ) / 2) - Pb ^ ((3 : ℝ) / 2)) /
        (Pt - Pb) := by
  have htb : t - b ≠ 0 := by linarith
  have hPd : Pt - Pb ≠ 0 := sub_ne_zero.mpr hne
  have hnn : ∀ x ∈ Set.uIcc b t, 0 ≤ dPlin b t Pb Pt x := by
    intro x hx
    rw [Set.uIcc_of_le hbt.le] at hx
    obtain ⟨h1, h2⟩ := hx
    have hl0 : 0 ≤ (x - b) / (t - b) := div_nonneg (by linarith) (by linarith)
    have hl1 : (x - b) / (t - b) ≤ 1 := by rw [div_le_one (by linarith)]; linarith
    unfold dPlin
    nlinarith
  set F : ℝ → ℝ := fun z => 2 / 3 * C * √(2 * ρ) * w * (t - b) / (Pt - Pb) *
    dPlin b t Pb Pt z ^ ((3 : ℝ) / 2) with hF
  have hderiv : ∀ x ∈ Set.uIcc b t,
      HasDerivAt F (C * √(2 * ρ * dPlin b t Pb Pt x) * w) x := by
    intro x hx
    have hp : HasDerivAt (dPlin b t Pb Pt) ((Pt - Pb) * (1 / (t - b))) x := by
      have := ((((hasDerivAt_id x).sub_const b).div_const (t - b)).const_mul
        (Pt - Pb)).const_add Pb
      exact this
    have hr := (hp.rpow_const (p := (3 : ℝ) / 2) (Or.inr (by norm_num))).const_mul
      (2 / 3 * C * √(2 * ρ) * w * (t - b) / (Pt - Pb))
    refine hr.congr_deriv ?_
    rw [Real.sqrt_mul (show (0 : ℝ) ≤ 2 * ρ by positivity) (dPlin b t Pb Pt x),
      Real.sqrt_eq_rpow (dPlin b t Pb Pt x)]
    norm_num
    field_simp
  have hcont : Continuous (fun z => C * √(2 * ρ * dPlin b t Pb Pt z) * w) := by
    unfold dPlin; fun_prop
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt hderiv (hcont.intervalIntegrable _ _)]
  simp only [hF, dPlin, sub_self, zero_div, mul_zero, add_zero, div_self htb, mul_one]
  ring_nf

/-- FD-23. The note's second closed form: for `P_t, P_b ≥ 0`, `P_t ≠ P_b`,
`(P_t^(3/2) - P_b^(3/2))/(P_t - P_b) = (P_t + √(P_t P_b) + P_b)/(√P_t + √P_b)`. -/
theorem FD23_closed_form_identity (Pt Pb : ℝ) (hPt : 0 ≤ Pt) (hPb : 0 ≤ Pb) (hne : Pt ≠ Pb) :
    (Pt ^ ((3 : ℝ) / 2) - Pb ^ ((3 : ℝ) / 2)) / (Pt - Pb) =
      (Pt + √(Pt * Pb) + Pb) / (√Pt + √Pb) := by
  have h3 : ∀ x : ℝ, 0 ≤ x → x ^ ((3 : ℝ) / 2) = (√x) ^ 3 := by
    intro x hx
    rw [Real.sqrt_eq_rpow, rpow_pow hx]; norm_num
  rw [h3 Pt hPt, h3 Pb hPb, Real.sqrt_mul hPt]
  have ha := Real.sq_sqrt hPt
  have hc := Real.sq_sqrt hPb
  have ha0 := Real.sqrt_nonneg Pt
  have hc0 := Real.sqrt_nonneg Pb
  set a := √Pt
  set c := √Pb
  rw [← ha, ← hc]
  have hac : a ≠ c := by
    intro h; apply hne; rw [← ha, ← hc, h]
  have h1 : a ^ 2 - c ^ 2 ≠ 0 := by
    intro h
    have : (a - c) * (a + c) = 0 := by linear_combination h
    rcases mul_eq_zero.mp this with h' | h'
    · exact hac (by linarith)
    · have : a = 0 := by linarith
      have : c = 0 := by linarith
      exact hac (by linarith)
  have h2 : a + c ≠ 0 := by
    intro h
    have : a = 0 := by linarith
    have : c = 0 := by linarith
    exact hac (by linarith)
  rw [div_eq_div_iff h1 h2]
  ring

/-- FD-23. Equal-pressure limit: if `ΔP` is the same `P ≥ 0` at both ends of the segment,
the flow is `C √(2 ρ P) w (t - b)`. -/
theorem FD23_equal_pressure (C ρ w b t P : ℝ) :
    ∫ z in b..t, C * √(2 * ρ * dPlin b t P P z) * w = (t - b) * (C * √(2 * ρ * P) * w) := by
  simp [dPlin]
  ring

/-! ## FD-25, FD-26, FD-36. Fire-front geometry on the graph and lattice -/

/-- FD-25. With non-negative flashover and barrier delays, the arrival time
`T_j = min_i (T_i + t_fo,i + t_bar,ij)` is never earlier than the earliest arrival among
the source rooms `i`. -/
theorem FD25_arrival_after_sources {ι : Type*} (s : Finset ι) (hs : s.Nonempty)
    (T tfo tbar : ι → ℝ) (h1 : ∀ i ∈ s, 0 ≤ tfo i) (h2 : ∀ i ∈ s, 0 ≤ tbar i) :
    s.inf' hs T ≤ s.inf' hs (fun i => T i + tfo i + tbar i) := by
  apply Finset.le_inf'
  intro i hi
  have := Finset.inf'_le T hi
  linarith [h1 i hi, h2 i hi]

/-- FD-25. Longer delays never make the fire arrive earlier: the min-plus update is
monotone in the flashover and barrier delays. -/
theorem FD25_monotone_delays {ι : Type*} (s : Finset ι) (hs : s.Nonempty)
    (T d d' : ι → ℝ) (h : ∀ i ∈ s, d i ≤ d' i) :
    s.inf' hs (fun i => T i + d i) ≤ s.inf' hs (fun i => T i + d' i) := by
  apply Finset.le_inf'
  intro i hi
  have := Finset.inf'_le (fun i => T i + d i) hi
  linarith [h i hi]

/-- FD-26. Metrication: for consecutive stencil directions an angle `φ ∈ (0, π)` apart,
`(sin(φ - ψ) + sin ψ)/sin φ = cos(φ/2 - ψ)/cos(φ/2) ≤ 1/cos(φ/2)` for every `ψ`. -/
theorem FD26_metrication (φ ψ : ℝ) (hφ : 0 < φ) (hφπ : φ < π) :
    (Real.sin (φ - ψ) + Real.sin ψ) / Real.sin φ = Real.cos (φ / 2 - ψ) / Real.cos (φ / 2) ∧
      Real.cos (φ / 2 - ψ) / Real.cos (φ / 2) ≤ 1 / Real.cos (φ / 2) := by
  set a := φ / 2 with ha
  have hφa : φ = 2 * a := by rw [ha]; ring
  have hcos : 0 < Real.cos a :=
    Real.cos_pos_of_mem_Ioo ⟨by linarith [Real.pi_pos], by linarith⟩
  have hsin : 0 < Real.sin a := Real.sin_pos_of_pos_of_lt_pi (by linarith) (by linarith)
  have hnum : Real.sin (φ - ψ) + Real.sin ψ = 2 * Real.sin a * Real.cos (a - ψ) := by
    rw [hφa, Real.sin_sub, Real.cos_sub, Real.sin_two_mul, Real.cos_two_mul]
    linear_combination (-2 * Real.sin ψ) * Real.sin_sq_add_cos_sq a
  have hden : Real.sin φ = 2 * Real.sin a * Real.cos a := by rw [hφa, Real.sin_two_mul]
  refine ⟨?_, div_le_div_of_nonneg_right (Real.cos_le_one _) hcos.le⟩
  rw [hnum, hden]
  field_simp

/-- FD-26. For the 8-direction stencil (`φ = π/4`) the worst-case metrication factor
`1/cos(π/8)` lies in `(1.0823, 1.0824)`, so the bound of `FD26_metrication` at
`φ = π/4` is an overestimate of at most 8.24%. -/
theorem FD26_eight_directions :
    1.0823 < 1 / Real.cos (π / 8) ∧ 1 / Real.cos (π / 8) < 1.0824 := by
  rw [Real.cos_pi_div_eight]
  have hs1 : (1.41421 : ℝ) < √2 := by rw [Real.lt_sqrt (by norm_num)]; norm_num
  have hs2 : √2 < (1.41422 : ℝ) := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
  have h1 : (1.847746 : ℝ) < √(2 + √2) := by rw [Real.lt_sqrt (by norm_num)]; nlinarith
  have h2 : √(2 + √2) < (1.84777 : ℝ) := by rw [Real.sqrt_lt' (by norm_num)]; nlinarith
  have hp : 0 < √(2 + √2) / 2 := by linarith
  constructor
  · rw [lt_div_iff₀ hp]; linarith
  · rw [div_lt_iff₀ hp]; linarith

/-- FD-36. Hop-count anisotropy: one hop covers 2 cells along an axis but `√2` cells of
distance along the diagonal, so the per-hop distance ratio is `2/√2 = √2` and the
anisotropy `√2 - 1` lies in `(0.414, 0.4143)`. -/
theorem FD36_hop_anisotropy :
    2 / √2 - 1 = √2 - 1 ∧ 0.414 < √2 - 1 ∧ √2 - 1 < 0.4143 := by
  have h : √2 * √2 = 2 := Real.mul_self_sqrt (by norm_num)
  have hp : 0 < √2 := by positivity
  have hs1 : (1.41421 : ℝ) < √2 := by rw [Real.lt_sqrt (by norm_num)]; norm_num
  have hs2 : √2 < (1.41422 : ℝ) := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
  have e : 2 / √2 = √2 := by rw [div_eq_iff hp.ne']; linarith
  exact ⟨by rw [e], by linarith, by linarith⟩

/-! ## Units (FD-24 and the dimensions of every empirical constant)

kW and W have the same dimension (the factor 1000 is a number), so `Dim.power` stands for
both. Each `d…` below is the unit a constant carries; its docstring names the constant. -/

section Units

/-- Fire growth coefficient α: kW s⁻². -/
def dAlpha : Dim := Dim.power / (Dim.time * Dim.time)
/-- Babrauskas's 750 and Thomas's 378: kW m⁻⁵ᐟ². -/
def dVent : Dim := Dim.power / Dim.rpow Dim.length 5 2
/-- Thomas's 7.8: kW m⁻². -/
def dThomasA : Dim := Dim.power / Dim.area
/-- Peacock's 740: kW^(1/2) K^(1/2) m^(-5/4) (the extraction gives no unit for it). -/
def d740 : Dim := Dim.rpow Dim.power 1 2 * Dim.rpow Dim.temp 1 2 * Dim.rpow Dim.length (-5) 4
/-- Heskestad's 0.235 and 0.083 (and McCaffrey's scaled height `z/Q^(2/5)`): m kW^(-2/5). -/
def dHesk : Dim := Dim.length * Dim.rpow Dim.power (-2) 5
/-- Heskestad's 0.071 (and Zukoski's coefficient): kg s⁻¹ kW^(-1/3) m^(-5/3). -/
def d0071 : Dim := Dim.massFlow * Dim.rpow Dim.power (-1) 3 * Dim.rpow Dim.length (-5) 3
/-- Heskestad's 0.026: kW^(-2/3) m^(5/3). -/
def d0026 : Dim := Dim.rpow Dim.power (-2) 3 * Dim.rpow Dim.length 5 3
/-- Plume and ceiling-jet temperature coefficients (Heskestad's reduced 25,
Alpert's 16.9 and 5.38): K kW^(-2/3) m^(5/3). -/
def dTcoef : Dim := Dim.temp * Dim.rpow Dim.power (-2) 3 * Dim.rpow Dim.length 5 3
/-- Plume and ceiling-jet velocity coefficients (Heskestad's reduced 1.03, Alpert's 0.96
and 0.195, McCaffrey's plume-region 1.1): m^(4/3) s⁻¹ kW^(-1/3). -/
def dUcoef : Dim := Dim.rpow Dim.length 4 3 / Dim.time * Dim.rpow Dim.power (-1) 3
/-- MQH's 6.85: K^(2/3) kW^(-1/3) m^(5/6). -/
def dMQH : Dim := Dim.rpow Dim.temp 2 3 * Dim.rpow Dim.power (-1) 3 * Dim.rpow Dim.length 5 6
/-- McCaffrey's flame-region 6.8: m^(1/2) s⁻¹. -/
def kFlame : Dim := Dim.rpow Dim.length 1 2 / Dim.time
/-- McCaffrey's intermittent-region 1.9: m s⁻¹ kW^(-1/5). -/
def kInter : Dim := Dim.velocity * Dim.rpow Dim.power (-1) 5
/-- The ambient group `ρ∞ c_p T∞ √g` of CFAST's and Evans's dimensionless heat release
`Q* = Q / (ρ∞ c_p T∞ √g D^(5/2))`. -/
def heatGroup : Dim := Dim.density * Dim.specificHeat * Dim.temp * Dim.rpow Dim.accel 1 2

/-- FD-01, FD-24 row 9. `α` is in kW s⁻²: `α t²` is a power, and `Q_ref / t_g²` has the
unit of `α`. -/
theorem FD01_units :
    dAlpha * Dim.time * Dim.time = Dim.power ∧ Dim.power / (Dim.time * Dim.time) = dAlpha := by
  decide

/-- FD-03. `750 A_o √H_o` is a power when 750 is in kW m⁻⁵ᐟ², and
`t_fo - t₀ = √(Q_fo/α)` is a time. -/
theorem FD03_units :
    dVent * Dim.area * Dim.rpow Dim.length 1 2 = Dim.power ∧ Dim.exactPow Dim.length 1 2 ∧
      Dim.rpow (Dim.power / dAlpha) 1 2 = Dim.time ∧ Dim.exactPow (Dim.power / dAlpha) 1 2 := by
  decide

/-- FD-04. `740 (h_k A_T A √h)^(1/2)` is a power only if 740 carries
kW^(1/2) K^(1/2) m^(-5/4), and that unit is exactly `K^(3/2) [6.85]^(-3/2)`, the unit of
`(ΔT/6.85)^(3/2)`: FD-04 is dimensionally the inversion of MQH at a fixed `ΔT`. -/
theorem FD04_units :
    d740 * Dim.rpow (Dim.htc * Dim.area * Dim.area * Dim.rpow Dim.length 1 2) 1 2 = Dim.power ∧
      Dim.exactPow (Dim.htc * Dim.area * Dim.area * Dim.rpow Dim.length 1 2) 1 2 ∧
      Dim.exactPow Dim.power 1 2 ∧ Dim.exactPow Dim.temp 1 2 ∧
      Dim.exactPow Dim.length (-5) 4 ∧
      d740 = Dim.rpow Dim.temp 3 2 * Dim.rpow dMQH (-3) 2 ∧ Dim.exactPow dMQH (-3) 2 := by
  decide

/-- FD-05. `7.8 A_T + 378 A √h`: both terms are powers with 7.8 in kW m⁻² and 378 in
kW m⁻⁵ᐟ². -/
theorem FD05_units :
    dThomasA * Dim.area = Dim.power ∧ dVent * Dim.area * Dim.rpow Dim.length 1 2 = Dim.power := by
  decide

/-- FD-06, FD-07. `0.235 Q^(2/5)` and `0.083 Q^(2/5)` are lengths with the constants in
m kW^(-2/5). Independently, CFAST's `Q*` is dimensionless and its form
`L = D (3.7 Q*^(2/5) - 1.02)` hides the group `(ρ c_p T √g)^(-2/5)`, whose unit is exactly
m kW^(-2/5): the note's unit for 0.235 is the right one. -/
theorem FD06_units :
    dHesk * Dim.rpow Dim.power 2 5 = Dim.length ∧ Dim.exactPow Dim.power 2 5 ∧
      Dim.exactPow Dim.power (-2) 5 ∧
      Dim.power / (heatGroup * Dim.rpow Dim.length 5 2) = Dim.one ∧
      Dim.exactPow Dim.length 5 2 ∧ Dim.exactPow Dim.accel 1 2 ∧
      Dim.rpow heatGroup (-2) 5 = dHesk ∧ Dim.exactPow heatGroup (-2) 5 := by
  decide

/-- FD-08, FD-30. `0.071 Q_c^(1/3) (z - z₀)^(5/3)` is a mass flow and
`0.026 Q_c^(2/3) (z - z₀)^(-5/3)` is dimensionless with the stated units. Independently,
CFAST's groups `(g ρ²/(c_p T))^(1/3)` and `(√g c_p ρ T)^(-2/3)` have exactly the units of
0.071 and 0.026. -/
theorem FD08_units :
    d0071 * Dim.rpow Dim.power 1 3 * Dim.rpow Dim.length 5 3 = Dim.massFlow ∧
      d0026 * Dim.rpow Dim.power 2 3 * Dim.rpow Dim.length (-5) 3 = Dim.one ∧
      Dim.exactPow Dim.power 1 3 ∧ Dim.exactPow Dim.power 2 3 ∧ Dim.exactPow Dim.length 5 3 ∧
      Dim.rpow (Dim.accel * Dim.density * Dim.density / (Dim.specificHeat * Dim.temp)) 1 3 =
        d0071 ∧
      Dim.exactPow (Dim.accel * Dim.density * Dim.density / (Dim.specificHeat * Dim.temp)) 1 3 ∧
      Dim.rpow heatGroup (-2) 3 = d0026 ∧ Dim.exactPow heatGroup (-2) 3 := by
  decide

/-- FD-09. The coefficient `9.1 (T∞/(g c_p² ρ∞²))^(1/3)` has the unit K kW^(-2/3) m^(5/3)
of the reduced constant 25, and with it `ΔT₀` is a temperature. The modified origin
`z_I - (T_u/T_l)^(3/5)(z_I - z₀)` is a length. -/
theorem FD09_units :
    dTcoef * Dim.rpow Dim.power 2 3 * Dim.rpow Dim.length (-5) 3 = Dim.temp ∧
      Dim.rpow (Dim.temp / (Dim.accel * Dim.specificHeat * Dim.specificHeat * Dim.density *
        Dim.density)) 1 3 = dTcoef ∧
      Dim.exactPow (Dim.temp / (Dim.accel * Dim.specificHeat * Dim.specificHeat * Dim.density *
        Dim.density)) 1 3 ∧
      Dim.rpow (Dim.temp / Dim.temp) 3 5 * Dim.length = Dim.length := by
  decide

/-- FD-10. `3.4 (g/(c_p ρ T))^(1/3) Q_c^(1/3) (z - z₀)^(-1/3)` is a velocity, and the group
has the unit of the reduced 1.03. In
`u₀,max = 2.2 g^(2/5) T^(-2/5) (c_p ρ)^(-1/5) (650 Q_c)^(1/5)` the result is a velocity
only if 650 is 650 K; with a bare 650 it would carry an extra K^(-1/5). -/
theorem FD10_units :
    dUcoef * Dim.rpow Dim.power 1 3 * Dim.rpow Dim.length (-1) 3 = Dim.velocity ∧
      Dim.rpow (Dim.accel / (Dim.specificHeat * Dim.density * Dim.temp)) 1 3 = dUcoef ∧
      Dim.exactPow (Dim.accel / (Dim.specificHeat * Dim.density * Dim.temp)) 1 3 ∧
      Dim.rpow Dim.accel 2 5 * Dim.rpow Dim.temp (-2) 5 *
        Dim.rpow (Dim.specificHeat * Dim.density) (-1) 5 *
        Dim.rpow (Dim.temp * Dim.power) 1 5 = Dim.velocity ∧
      Dim.exactPow Dim.accel 2 5 ∧ Dim.exactPow Dim.temp (-2) 5 ∧
      Dim.exactPow (Dim.specificHeat * Dim.density) (-1) 5 ∧
      Dim.exactPow (Dim.temp * Dim.power) 1 5 ∧ Dim.exactPow Dim.power 1 5 ∧
      Dim.rpow Dim.accel 2 5 * Dim.rpow Dim.temp (-2) 5 *
        Dim.rpow (Dim.specificHeat * Dim.density) (-1) 5 * Dim.rpow Dim.power 1 5 =
          Dim.velocity * Dim.rpow Dim.temp (-1) 5 := by
  decide

/-- FD-11. In all three McCaffrey regions `V = k Q^(1/5) (z/Q^(2/5))^η` is a velocity,
with `k` in m^(1/2) s⁻¹ (η = 1/2), m s⁻¹ kW^(-1/5) (η = 0) and m^(4/3) s⁻¹ kW^(-1/3)
(η = -1/3), as McCaffrey's table prints them. -/
theorem FD11_units :
    kFlame * Dim.rpow Dim.power 1 5 * Dim.rpow dHesk 1 2 = Dim.velocity ∧
      kInter * Dim.rpow Dim.power 1 5 * Dim.rpow dHesk 0 1 = Dim.velocity ∧
      dUcoef * Dim.rpow Dim.power 1 5 * Dim.rpow dHesk (-1) 3 = Dim.velocity ∧
      Dim.exactPow dHesk 1 2 ∧ Dim.exactPow dHesk (-1) 3 ∧ Dim.exactPow Dim.power 1 5 ∧
      Dim.exactPow Dim.length 1 2 ∧ Dim.exactPow Dim.power (-1) 5 := by
  decide

/-- FD-12. `2 g ΔT / T₀` is an acceleration, and so is `(k/C)² (z/Q^(2/5))^(2η - 1)` in
each region (`C` dimensionless). -/
theorem FD12_units :
    Dim.accel * Dim.temp / Dim.temp = Dim.accel ∧
      kFlame * kFlame * Dim.rpow dHesk 0 1 = Dim.accel ∧
      kInter * kInter * Dim.rpow dHesk (-1) 1 = Dim.accel ∧
      dUcoef * dUcoef * Dim.rpow dHesk (-5) 3 = Dim.accel ∧ Dim.exactPow dHesk (-5) 3 := by
  decide

/-- FD-13. Both Alpert temperature branches are temperatures with 16.9 and 5.38 in
K kW^(-2/3) m^(5/3). Independently, Evans's `ΔT/T∞ = 6.18 Q_H*^(2/3)` hides the group
`T∞ (ρ c_p T∞ √g)^(-2/3)`, whose unit is exactly that. -/
theorem FD13_units :
    dTcoef * Dim.rpow Dim.power 2 3 * Dim.rpow Dim.length (-5) 3 = Dim.temp ∧
      dTcoef * Dim.rpow (Dim.power / Dim.length) 2 3 / Dim.length = Dim.temp ∧
      Dim.exactPow (Dim.power / Dim.length) 2 3 ∧
      Dim.temp * Dim.rpow heatGroup (-2) 3 = dTcoef := by
  decide

/-- FD-14. Both Alpert velocity branches are velocities with 0.96 and 0.195 in
m^(4/3) s⁻¹ kW^(-1/3). Independently, Evans's `U/√(gH) = 3.16 Q_H*^(1/3)` hides the group
`√g (ρ c_p T∞ √g)^(-1/3)`, whose unit is exactly that. -/
theorem FD14_units :
    dUcoef * Dim.rpow (Dim.power / Dim.length) 1 3 = Dim.velocity ∧
      dUcoef * Dim.rpow Dim.power 1 3 * Dim.rpow Dim.length 1 2 * Dim.rpow Dim.length (-5) 6 =
        Dim.velocity ∧
      Dim.exactPow (Dim.power / Dim.length) 1 3 ∧ Dim.exactPow Dim.length (-5) 6 ∧
      Dim.rpow Dim.accel 1 2 * Dim.rpow heatGroup (-1) 3 = dUcoef ∧
      Dim.exactPow heatGroup (-1) 3 := by
  decide

/-- FD-15. `Q_H* = Q/(ρ∞ c_p T∞ √g H^(5/2))` is dimensionless and `√(gH)` is a velocity,
so CFAST's ceiling-jet laws need no dimensional constants. -/
theorem FD15_units :
    Dim.power / (heatGroup * Dim.rpow Dim.length 5 2) = Dim.one ∧
      Dim.rpow (Dim.accel * Dim.length) 1 2 = Dim.velocity ∧
      Dim.exactPow (Dim.accel * Dim.length) 1 2 := by
  decide

/-- FD-16. `ṁ_e c_p (T_u - T_l)` is a power and `Q_c/(c_p (T_u - T_l))` a mass flow. -/
theorem FD16_units :
    Dim.massFlow * Dim.specificHeat * Dim.temp = Dim.power ∧
      Dim.power / (Dim.specificHeat * Dim.temp) = Dim.massFlow := by
  decide

/-- FD-17, FD-24 row 6. MQH's `ΔT` is a temperature with 6.85 in K^(2/3) kW^(-1/3) m^(5/6).
Independently, `X₁` and `X₂` are dimensionless and the absorbed group
`T₀^(1/3) (√g ρ₀ c_p)^(-1/3)` has exactly the unit of 6.85. -/
theorem FD17_units :
    dMQH * Dim.rpow (Dim.power * Dim.power /
        (Dim.area * Dim.rpow Dim.length 1 2 * Dim.htc * Dim.area)) 1 3 = Dim.temp ∧
      Dim.exactPow (Dim.power * Dim.power /
        (Dim.area * Dim.rpow Dim.length 1 2 * Dim.htc * Dim.area)) 1 3 ∧
      Dim.rpow Dim.temp 1 3 *
        Dim.rpow (Dim.rpow Dim.accel 1 2 * Dim.density * Dim.specificHeat) (-1) 3 = dMQH ∧
      Dim.exactPow (Dim.rpow Dim.accel 1 2 * Dim.density * Dim.specificHeat) (-1) 3 ∧
      Dim.exactPow Dim.temp 1 3 ∧ Dim.exactPow Dim.temp 2 3 ∧ Dim.exactPow Dim.power (-1) 3 ∧
      Dim.exactPow Dim.length 5 6 ∧
      Dim.power / (Dim.rpow Dim.accel 1 2 * Dim.density * Dim.specificHeat * Dim.temp *
        Dim.area * Dim.rpow Dim.length 1 2) = Dim.one ∧
      Dim.htc * Dim.area / (Dim.rpow Dim.accel 1 2 * Dim.density * Dim.specificHeat *
        Dim.area * Dim.rpow Dim.length 1 2) = Dim.one := by
  decide

/-- FD-18. Both branches of `h_k` are heat transfer coefficients (W m⁻² K⁻¹) and `t_p` is
a time. -/
theorem FD18_units :
    Dim.rpow (Dim.conductivity * Dim.density * Dim.specificHeat / Dim.time) 1 2 = Dim.htc ∧
      Dim.exactPow (Dim.conductivity * Dim.density * Dim.specificHeat / Dim.time) 1 2 ∧
      Dim.conductivity / Dim.length = Dim.htc ∧
      Dim.density * Dim.specificHeat / Dim.conductivity * Dim.area = Dim.time := by
  decide

/-- FD-19. `P V = m R T` balances with `R` a specific heat, and both terms of the energy
equation `d(c_v m T)/dt + P dV/dt` are powers. -/
theorem FD19_units :
    Dim.pressure * Dim.volume = Dim.mass * Dim.specificHeat * Dim.temp ∧
      Dim.specificHeat * Dim.mass * Dim.temp / Dim.time = Dim.power ∧
      Dim.pressure * Dim.volume / Dim.time = Dim.power := by
  decide

/-- FD-20 to FD-22, FD-24 rows 1-3. `[dP/dt] = W m⁻³ = Pa s⁻¹`,
`[dV_u/dt] = W Pa⁻¹ = m³ s⁻¹`, `[dT_u/dt] = W/(J K⁻¹) = K s⁻¹`; the terms
`c_p ṁ T` and `V dP/dt` in FD-22 are powers; `z_i = (V - V_u)/A_f` is a length. -/
theorem FD24_zone_units :
    Dim.power / Dim.volume = Dim.pressure / Dim.time ∧
      Dim.power / Dim.pressure = Dim.volume / Dim.time ∧
      Dim.power / (Dim.energy / Dim.temp) = Dim.temp / Dim.time ∧
      Dim.power / (Dim.specificHeat * Dim.mass) = Dim.temp / Dim.time ∧
      Dim.specificHeat * Dim.massFlow * Dim.temp = Dim.power ∧
      Dim.volume * (Dim.pressure / Dim.time) = Dim.power ∧
      Dim.volume / Dim.area = Dim.length := by
  decide

/-- FD-23. `C √(2 ρ |ΔP|) w dz` is a mass flow (`C` dimensionless). -/
theorem FD23_units :
    Dim.rpow (Dim.density * Dim.pressure) 1 2 * Dim.area = Dim.massFlow ∧
      Dim.exactPow (Dim.density * Dim.pressure) 1 2 := by
  decide

/-- FD-28, FD-29. The eikonal `F |∇T| = 1` is dimensionless, and each Bellman edge
weight `ℓ/F` is a time. -/
theorem FD28_FD29_units :
    Dim.velocity * (Dim.time / Dim.length) = Dim.one ∧ Dim.length / Dim.velocity = Dim.time := by
  decide

end Units

/-! ## Axioms -/

#print axioms root_bounds
#print axioms rpow_frac
#print axioms rpow_pow
#print axioms FD01_reaches_ref
#print axioms FD01_strictMonoOn
#print axioms FD01_convexOn
#print axioms FD01_clamped_monotone
#print axioms FD01_clamped_convexOn
#print axioms FD02_growth_classes
#print axioms FD03_tfo_reaches
#print axioms FD03_tfo_unique
#print axioms FD03_babrauskas_monotone
#print axioms FD03_worked_example
#print axioms FD04_mqh_inverse
#print axioms FD04_740_vs_685
#print axioms FD05_thomas_monotone
#print axioms FD07_origin_below_flame
#print axioms FD06_flame_strictMono
#print axioms FD08_expand
#print axioms FD08_strictMono_height
#print axioms FD09_antitone_height
#print axioms FD09_strictAnti_uncapped
#print axioms FD09_interface_continuous
#print axioms FD10_antitone_height
#print axioms FD10_umax_constant
#print axioms FD12_from_FD11
#print axioms FD13_far_at_break
#print axioms FD13_jump_bound
#print axioms FD13_antitone_radius
#print axioms FD13_monotone_Q
#print axioms FD14_far_at_break
#print axioms FD14_jump_bound
#print axioms FD14_antitone_radius
#print axioms FD15_jump_bound
#print axioms FD15_antitone_radius
#print axioms FD16_weak_plume
#print axioms FD17_strictMono_Q
#print axioms FD17_antitone_hk
#print axioms FD17_dimensionless_form
#print axioms FD17_constant_685
#print axioms FD18_branches_at_tp
#print axioms FD18_antitone_time
#print axioms FD17_monotone_time
#print axioms FD19_gas_constant
#print axioms FD19_R_eq
#print axioms FD19_internal_energy
#print axioms FD20_pressure_equation
#print axioms FD20_energy_conserved
#print axioms FD21_upper_volume
#print axioms FD21_interface_height
#print axioms FD22_layer_temperature
#print axioms FD23_vent_integral
#print axioms FD23_closed_form_identity
#print axioms FD23_equal_pressure
#print axioms FD25_arrival_after_sources
#print axioms FD25_monotone_delays
#print axioms FD26_metrication
#print axioms FD26_eight_directions
#print axioms FD36_hop_anisotropy
#print axioms FD01_units
#print axioms FD03_units
#print axioms FD04_units
#print axioms FD05_units
#print axioms FD06_units
#print axioms FD08_units
#print axioms FD09_units
#print axioms FD10_units
#print axioms FD11_units
#print axioms FD12_units
#print axioms FD13_units
#print axioms FD14_units
#print axioms FD15_units
#print axioms FD16_units
#print axioms FD17_units
#print axioms FD18_units
#print axioms FD19_units
#print axioms FD24_zone_units
#print axioms FD23_units
#print axioms FD28_FD29_units

end FireEgress.FireDynamics
