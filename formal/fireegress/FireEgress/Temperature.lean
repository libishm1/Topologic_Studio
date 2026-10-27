import Mathlib

/-!
# Topologic Studio's temperature fire step

One node's update in `temperature_steps` (server `app/graphs/fire.py`, browser
`src/core/fire.js`): with current value `a` and neighbour mean `m`,

  `a' = a + c * (m - a)` when `m > a`, otherwise `a' = a`.

Classic (eCAADe 2019) used `c = 1.2 / degree`. The literature review's finding
W1 (`wiki/research/fire-egress-methods-2026-10-27.md`) is that this breaks the
discrete maximum principle at dead ends; Studio now uses `c = min 1 (1.2 / degree)`.
This file proves why, and that the fix is the minimal one.
-/

namespace FireEgress.Temperature

/-- One node's update: heat only flows into a cooler node. -/
noncomputable def step (c a m : ℝ) : ℝ := if a < m then a + c * (m - a) else a

/-- L3. Classic's coefficient overshoots: a dead end (degree 1, so `c = 1.2`) at
20 °C next to a 120 °C neighbour jumps to 140 °C, above its neighbour. -/
theorem classic_overshoots : step (1.2 / 1) 20 120 = 140 := by
  norm_num [step]

/-- L4. With `0 ≤ c ≤ 1` the update stays between the old value and the
neighbour mean: no overshoot. -/
theorem step_between (c a m : ℝ) (h0 : 0 ≤ c) (h1 : c ≤ 1) :
    min a m ≤ step c a m ∧ step c a m ≤ max a m := by
  unfold step
  split_ifs with h
  · have hpos : 0 ≤ c * (m - a) := mul_nonneg h0 (by linarith)
    have hle : c * (m - a) ≤ m - a := by nlinarith
    exact ⟨le_trans (min_le_left a m) (by linarith), le_trans (by linarith) (le_max_right a m)⟩
  · exact ⟨min_le_left a m, le_max_left a m⟩

/-- The capped coefficient lies in `[0, 1]` for any degree. -/
theorem capped_coefficient_unit (k : ℝ) (deg : ℕ) (hk : 0 ≤ k) :
    0 ≤ min 1 (k / deg) ∧ min 1 (k / deg) ≤ 1 :=
  ⟨le_min zero_le_one (div_nonneg hk (Nat.cast_nonneg _)), min_le_left _ _⟩

/-- From degree 2 up the cap is inactive (`1.2 / deg ≤ 1`), so the corrected
step equals Classic's everywhere except at dead ends: results stay comparable. -/
theorem cap_inactive_from_degree_two (deg : ℕ) (hd : 2 ≤ deg) :
    min 1 ((1.2 : ℝ) / deg) = 1.2 / deg := by
  have h2 : (2 : ℝ) ≤ deg := by exact_mod_cast hd
  apply min_eq_right
  rw [div_le_one (by linarith)]
  linarith

/-- Discrete maximum principle for one step: if a node and all its neighbours
lie in `[lo, hi]`, so does the node after the update. -/
theorem step_bounded {ι : Type*} (s : Finset ι) (hs : s.Nonempty) (T : ι → ℝ)
    (a c lo hi : ℝ) (h0 : 0 ≤ c) (h1 : c ≤ 1) (ha : lo ≤ a ∧ a ≤ hi)
    (hT : ∀ j ∈ s, lo ≤ T j ∧ T j ≤ hi) :
    lo ≤ step c a ((∑ j ∈ s, T j) / s.card) ∧
      step c a ((∑ j ∈ s, T j) / s.card) ≤ hi := by
  have hcard : (0 : ℝ) < s.card := by exact_mod_cast hs.card_pos
  have hsum_lo : (s.card : ℝ) * lo ≤ ∑ j ∈ s, T j := by
    have := Finset.sum_le_sum (fun j hj => (hT j hj).1)
    simpa [Finset.sum_const, nsmul_eq_mul] using this
  have hsum_hi : ∑ j ∈ s, T j ≤ (s.card : ℝ) * hi := by
    have := Finset.sum_le_sum (fun j hj => (hT j hj).2)
    simpa [Finset.sum_const, nsmul_eq_mul] using this
  have hm_lo : lo ≤ (∑ j ∈ s, T j) / s.card := by
    rw [le_div_iff₀ hcard]; linarith
  have hm_hi : (∑ j ∈ s, T j) / s.card ≤ hi := by
    rw [div_le_iff₀ hcard]; linarith
  obtain ⟨hl, hu⟩ := step_between c a ((∑ j ∈ s, T j) / s.card) h0 h1
  exact ⟨le_trans (le_min ha.1 hm_lo) hl, le_trans hu (max_le ha.2 hm_hi)⟩

end FireEgress.Temperature
