import Mathlib

/-!
# Pre-movement: the measured Weibull, the share that never responds, detection, scenarios

`src/core/performanceCheck.js` and `src/core/rsetPresets.js`: beside the design
log-normal of `Performance.lean`, a preset may give pre-movement as a Weibull
distribution with scale `lam` and shape `k`, fitted to measured times; a share
`q` of the occupants may never respond; detection is the first of several
ceiling detectors to alarm; and the report gathers several fire scenarios and
gives the worst margin among them.

* `weibull_cdf_quantile`: the Weibull's quantile function inverts its
  distribution function on `(0, 1)`.
* `weibull_mono`, `weibull_nonneg`, `weibull_le_one`: the distribution
  function is monotone and lies between 0 and 1 (its lower bound needs a scale
  of at least 0).
* `responding_le`, `responding_mono`: the share that has started when a share
  `q` never responds is at most `1 - q` and at most the responders' share, and
  it is monotone in time.
* `responding_pass`: a pass at the responders' percentile with share `s`
  means at least `(1 - q) s` of all occupants start in time.
* `detection_more_detectors`, `detection_le`: detection is the earliest alarm,
  so adding detectors never makes it later, and it is no later than any one
  detector's alarm.
* `report_pass_iff`: the report's worst margin is at least 0 exactly when
  every scenario's is.
-/

namespace FireEgress.PreMovement

open Real

/-- The Weibull distribution function with scale `lam` and shape `k`. -/
noncomputable def weibullCdf (lam k t : ℝ) : ℝ :=
  if t ≤ 0 then 0 else 1 - exp (-((t / lam) ^ k))

/-- Its quantile function: the time by which the share `p` has started. -/
noncomputable def weibullQuantile (lam k p : ℝ) : ℝ := lam * (-log (1 - p)) ^ (1 / k)

/-- The quantile inverts the distribution function on `(0, 1)`. -/
theorem weibull_cdf_quantile {lam k p : ℝ} (hlam : 0 < lam) (hk : 0 < k) (hp0 : 0 < p)
    (hp1 : p < 1) : weibullCdf lam k (weibullQuantile lam k p) = p := by
  have h1p : 0 < 1 - p := sub_pos.mpr hp1
  have hy : 0 < -log (1 - p) := by
    have := log_neg h1p (by linarith)
    linarith
  have hq : 0 < weibullQuantile lam k p := mul_pos hlam (rpow_pos_of_pos hy _)
  unfold weibullCdf
  rw [ite_eq_right (not_le.mpr hq)]
  have e : weibullQuantile lam k p / lam = (-log (1 - p)) ^ (1 / k) := by
    unfold weibullQuantile
    field_simp
  rw [e, ← rpow_mul hy.le, one_div_mul_cancel hk.ne', rpow_one, neg_neg, exp_log h1p]
  ring

/-- The distribution function never falls. -/
theorem weibull_mono {lam k : ℝ} (hlam : 0 < lam) (hk : 0 < k) : Monotone (weibullCdf lam k) := by
  intro t t' htt
  unfold weibullCdf
  by_cases h' : t' ≤ 0
  · rw [ite_eq_left (le_trans htt h'), ite_eq_left h']
  · rw [ite_eq_right h']
    have ht'0 : 0 < t' := not_le.mp h'
    by_cases h : t ≤ 0
    · rw [ite_eq_left h]
      have h0 : 0 ≤ (t' / lam) ^ k := rpow_nonneg (div_nonneg ht'0.le hlam.le) k
      have h1 := exp_le_exp.mpr (show -((t' / lam) ^ k) ≤ 0 by linarith)
      rw [exp_zero] at h1
      linarith
    · rw [ite_eq_right h]
      have ht0 : 0 < t := not_le.mp h
      have hpow : (t / lam) ^ k ≤ (t' / lam) ^ k :=
        rpow_le_rpow (div_nonneg ht0.le hlam.le) (div_le_div_of_nonneg_right htt hlam.le) hk.le
      have h1 := exp_le_exp.mpr (show -((t' / lam) ^ k) ≤ -((t / lam) ^ k) by linarith)
      linarith

/-- The distribution function is at least 0 for a scale of at least 0. -/
theorem weibull_nonneg {lam : ℝ} (k : ℝ) (hlam : 0 ≤ lam) (t : ℝ) : 0 ≤ weibullCdf lam k t := by
  unfold weibullCdf
  split_ifs with h
  · exact le_rfl
  · have h0 : 0 ≤ (t / lam) ^ k := rpow_nonneg (div_nonneg (not_le.mp h).le hlam) k
    have h1 := exp_le_exp.mpr (show -((t / lam) ^ k) ≤ 0 by linarith)
    rw [exp_zero] at h1
    linarith

/-- The distribution function is at most 1. -/
theorem weibull_le_one (lam k t : ℝ) : weibullCdf lam k t ≤ 1 := by
  unfold weibullCdf
  split_ifs
  · exact zero_le_one
  · linarith [exp_pos (-((t / lam) ^ k))]

/-- The share of all occupants that has started, when a share `q` never
responds and `F` is the responders' distribution value. -/
def responding (q F : ℝ) : ℝ := (1 - q) * F

/-- It is at most the share that responds at all, and at most the responders' share. -/
theorem responding_le {q F : ℝ} (hq0 : 0 ≤ q) (hq1 : q ≤ 1) (hF0 : 0 ≤ F) (hF1 : F ≤ 1) :
    responding q F ≤ 1 - q ∧ responding q F ≤ F := by
  unfold responding
  constructor
  · have := mul_le_mul_of_nonneg_left hF1 (sub_nonneg.mpr hq1)
    rwa [mul_one] at this
  · have e : (1 - q) * F = F - q * F := by ring
    rw [e]
    linarith [mul_nonneg hq0 hF0]

/-- It never falls in time. -/
theorem responding_mono {q : ℝ} {F : ℝ → ℝ} (hq : q ≤ 1) (hF : Monotone F) :
    Monotone (fun t => responding q (F t)) :=
  fun _ _ h => mul_le_mul_of_nonneg_left (hF h) (sub_nonneg.mpr hq)

/-- A pass at the responders' percentile, with share `s` at `q99`, means at
least `(1 - q) s` of all occupants start in time. -/
theorem responding_pass {F : ℝ → ℝ} (hF : Monotone F) {q s q99 cue L : ℝ} (hs : s ≤ F q99)
    (hq99 : q99 ≤ L - cue) (hq1 : q ≤ 1) : (1 - q) * s ≤ responding q (F (L - cue)) :=
  mul_le_mul_of_nonneg_left (le_trans hs (hF hq99)) (sub_nonneg.mpr hq1)

/-- Detection is the earliest alarm: adding detectors never makes it later. -/
theorem detection_more_detectors {ι : Type*} {D D' : Finset ι} (hD : D.Nonempty)
    (hD' : D'.Nonempty) (h : D ⊆ D') (t : ι → ℝ) : D'.inf' hD' t ≤ D.inf' hD t :=
  (Finset.le_inf'_iff hD t).mpr (fun _ hd => Finset.inf'_le t (h hd))

/-- Detection is no later than any one detector's alarm. -/
theorem detection_le {ι : Type*} {D : Finset ι} (hD : D.Nonempty) (t : ι → ℝ) {d : ι}
    (hd : d ∈ D) : D.inf' hD t ≤ t d :=
  Finset.inf'_le t hd

/-- The report's worst margin is at least 0 exactly when every scenario's is. -/
theorem report_pass_iff {ι : Type*} {S : Finset ι} (hS : S.Nonempty) (w : ι → ℝ) :
    0 ≤ S.inf' hS w ↔ ∀ s ∈ S, 0 ≤ w s :=
  Finset.le_inf'_iff hS w

end FireEgress.PreMovement
