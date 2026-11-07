import FireEgress.Performance

/-!
# The queue at each door

`src/core/performanceCheck.js`: a door passes at most `c` persons per second.
The check counts the arrivals `a b` at a door in time bins of width `Δ`
seconds and runs a fluid first-in, first-out queue on the bins, Lindley's
recursion: the queue at the start of bin `b + 1` is the queue at the start of
bin `b`, plus the bin's arrivals, less what the door passes in a bin, `c Δ`,
and never below zero. The time a person spends queueing, `W`, is added to the
required start of `Performance.lean`.

* `queue_zero`, `queue_succ`: the recursion.
* `queue_nonneg`, `out_le`, `out_nonneg`: the queue is never negative, and the
  departures in a bin lie between 0 and the door's capacity over the bin,
  `c Δ`, when arrivals and capacity are at least 0.
* `conservation`: everyone who arrived in the first `B` bins has either left
  in them or is still in the queue at the start of bin `B`.
* `queue_mono`: more arrivals never shorten the queue.
* `queue_le_window`: over the bins `n, ..., n + m - 1` the queue grows by at
  most the arrivals and drains by `c Δ` a bin until it is empty.
* `clears`: when the queue at the end of bin `b` is at most `m` bins of
  capacity, everyone who arrived up to bin `b` has left by the end of bin
  `b + m`.
* `exit_mono`, `exit_clears`: the bin `exitBin b`, by whose start the
  arrivals of bin `b` have all left, never falls as `b` grows (first in,
  first out), and those arrivals have indeed all left by then.
* `requiredQ_iff`, `requiredQ_zero`, `requiredQ_mono`: with a wait `W` at the
  door, the check passes exactly when the margin and the ratio hold with RSET
  `base + τ + W` and ASET `L + τ`; with no wait it is `Performance.required`;
  a longer wait never asks less of a place while the factor is at least 0.
* `queued_share_le`, `safe_starts_downward`: queues never raise the share
  that starts in time, and with a monotone exit-adjusted start the starts that
  pass are closed downward.
-/

namespace FireEgress.DoorQueue

open Finset

/-- The queue at the start of bin `b`, in persons, for arrivals `a`, a door of
capacity `c` persons per second and bins `Δ` seconds wide (Lindley's recursion). -/
noncomputable def queue (a : ℕ → ℝ) (c Δ : ℝ) : ℕ → ℝ
  | 0 => 0
  | b + 1 => max 0 (queue a c Δ b + a b - c * Δ)

/-- The departures in bin `b`. -/
noncomputable def out (a : ℕ → ℝ) (c Δ : ℝ) (b : ℕ) : ℝ :=
  queue a c Δ b + a b - queue a c Δ (b + 1)

/-- The queue starts empty. -/
theorem queue_zero (a : ℕ → ℝ) (c Δ : ℝ) : queue a c Δ 0 = 0 := rfl

/-- One step of Lindley's recursion. -/
theorem queue_succ (a : ℕ → ℝ) (c Δ : ℝ) (b : ℕ) :
    queue a c Δ (b + 1) = max 0 (queue a c Δ b + a b - c * Δ) := rfl

/-- The queue is never negative. -/
theorem queue_nonneg (a : ℕ → ℝ) (c Δ : ℝ) (b : ℕ) : 0 ≤ queue a c Δ b := by
  cases b with
  | zero => exact le_of_eq (queue_zero a c Δ).symm
  | succ b => rw [queue_succ]; exact le_max_left _ _

/-- The departures in a bin never exceed the door's capacity over the bin. -/
theorem out_le (a : ℕ → ℝ) (c Δ : ℝ) (b : ℕ) : out a c Δ b ≤ c * Δ := by
  unfold out
  rw [queue_succ]
  have := le_max_right 0 (queue a c Δ b + a b - c * Δ)
  linarith

/-- With arrivals and capacity at least 0, the departures in a bin are at least 0. -/
theorem out_nonneg {a : ℕ → ℝ} {c Δ : ℝ} (ha : ∀ j, 0 ≤ a j) (hcΔ : 0 ≤ c * Δ) (b : ℕ) :
    0 ≤ out a c Δ b := by
  unfold out
  rw [queue_succ]
  have h0 := queue_nonneg a c Δ b
  have h1 := ha b
  have : max 0 (queue a c Δ b + a b - c * Δ) ≤ queue a c Δ b + a b :=
    max_le (by linarith) (by linarith)
  linarith

/-- Conservation: the departures over the first `B` bins are the arrivals less
the queue left at the start of bin `B`. -/
theorem conservation (a : ℕ → ℝ) (c Δ : ℝ) (B : ℕ) :
    ∑ j ∈ range B, out a c Δ j = ∑ j ∈ range B, a j - queue a c Δ B := by
  induction B with
  | zero => simp [queue_zero]
  | succ B ih =>
    rw [sum_range_succ, sum_range_succ, ih]
    unfold out
    ring

/-- More arrivals never shorten the queue. -/
theorem queue_mono {a a' : ℕ → ℝ} {c Δ : ℝ} (h : ∀ j, a j ≤ a' j) (b : ℕ) :
    queue a c Δ b ≤ queue a' c Δ b := by
  induction b with
  | zero => rw [queue_zero, queue_zero]
  | succ b ih =>
    rw [queue_succ, queue_succ]
    exact max_le_max le_rfl (by linarith [h b])

/-- Over the bins `n, ..., n + m - 1` the queue grows by at most their arrivals
and drains by `c Δ` a bin until it is empty. -/
theorem queue_le_window {a : ℕ → ℝ} {c Δ : ℝ} (ha : ∀ j, 0 ≤ a j) (hcΔ : 0 ≤ c * Δ)
    (n m : ℕ) :
    queue a c Δ (n + m) ≤ max 0 (queue a c Δ n - m * (c * Δ)) +
      (∑ j ∈ range (n + m), a j - ∑ j ∈ range n, a j) := by
  induction m with
  | zero =>
    simp only [Nat.cast_zero, zero_mul, sub_zero, add_zero, sub_self]
    exact le_max_right _ _
  | succ m ih =>
    rw [← add_assoc, queue_succ, sum_range_succ]
    push_cast
    have hS : ∑ j ∈ range n, a j ≤ ∑ j ∈ range (n + m), a j :=
      sum_le_sum_of_subset_of_nonneg
        (by intro x hx; simp only [mem_range] at hx ⊢; omega) (fun j _ _ => ha j)
    have e : queue a c Δ n - ((m : ℝ) + 1) * (c * Δ) =
        (queue a c Δ n - m * (c * Δ)) - c * Δ := by ring
    rw [e]
    have hX : max 0 (queue a c Δ n - m * (c * Δ)) - c * Δ ≤
        max 0 ((queue a c Δ n - m * (c * Δ)) - c * Δ) := by
      rcases le_total 0 (queue a c Δ n - m * (c * Δ)) with h | h
      · rw [max_eq_right h]; exact le_max_right _ _
      · rw [max_eq_left h]; linarith [le_max_left 0 ((queue a c Δ n - m * (c * Δ)) - c * Δ)]
    apply max_le
    · linarith [le_max_left 0 ((queue a c Δ n - m * (c * Δ)) - c * Δ), ha (n + m)]
    · linarith

/-- When the queue at the end of bin `b` is at most `m` bins of capacity,
everyone who arrived up to bin `b` has left by the end of bin `b + m`. -/
theorem clears {a : ℕ → ℝ} {c Δ : ℝ} (ha : ∀ j, 0 ≤ a j) (hcΔ : 0 ≤ c * Δ) {b m : ℕ}
    (hq : queue a c Δ (b + 1) ≤ m * (c * Δ)) :
    ∑ j ∈ range (b + 1), a j ≤ ∑ j ∈ range (b + 1 + m), out a c Δ j := by
  rw [conservation]
  have h := queue_le_window ha hcΔ (b + 1) m
  rw [max_eq_left (by linarith : queue a c Δ (b + 1) - m * (c * Δ) ≤ 0)] at h
  linarith

/-- The bin by whose start everyone who arrived up to bin `b` has left: the
queue at the end of bin `b` drains at `c Δ` persons a bin. -/
noncomputable def exitBin (a : ℕ → ℝ) (c Δ : ℝ) (b : ℕ) : ℤ :=
  b + 1 + ⌈queue a c Δ (b + 1) / (c * Δ)⌉

/-- First in, first out: a later arrival bin never leaves earlier. -/
theorem exit_mono {a : ℕ → ℝ} {c Δ : ℝ} (ha : ∀ j, 0 ≤ a j) (hc : 0 < c) (hΔ : 0 < Δ) :
    Monotone (exitBin a c Δ) := by
  have hcΔ : 0 < c * Δ := mul_pos hc hΔ
  apply monotone_nat_of_le_succ
  intro b
  unfold exitBin
  have hq : queue a c Δ (b + 1) - c * Δ ≤ queue a c Δ (b + 1 + 1) := by
    rw [queue_succ a c Δ (b + 1)]
    have := le_max_right 0 (queue a c Δ (b + 1) + a (b + 1) - c * Δ)
    linarith [ha (b + 1)]
  have hdiv : queue a c Δ (b + 1) / (c * Δ) - 1 ≤ queue a c Δ (b + 1 + 1) / (c * Δ) := by
    have e : queue a c Δ (b + 1) / (c * Δ) - 1 = (queue a c Δ (b + 1) - c * Δ) / (c * Δ) := by
      rw [sub_div, div_self hcΔ.ne']
    rw [e]
    exact div_le_div_of_nonneg_right hq hcΔ.le
  have hceil := Int.ceil_mono hdiv
  rw [Int.ceil_sub_one] at hceil
  push_cast
  omega

/-- The arrivals up to bin `b` have all left within the first `exitBin b` bins. -/
theorem exit_clears {a : ℕ → ℝ} {c Δ : ℝ} (ha : ∀ j, 0 ≤ a j) (hc : 0 < c) (hΔ : 0 < Δ)
    (b : ℕ) :
    ∑ j ∈ range (b + 1), a j ≤ ∑ j ∈ range (exitBin a c Δ b).toNat, out a c Δ j := by
  have hcΔ : 0 < c * Δ := mul_pos hc hΔ
  have hx0 : 0 ≤ queue a c Δ (b + 1) / (c * Δ) :=
    div_nonneg (queue_nonneg a c Δ (b + 1)) hcΔ.le
  have hnat : (exitBin a c Δ b).toNat = b + 1 + ⌈queue a c Δ (b + 1) / (c * Δ)⌉₊ := by
    unfold exitBin
    have h1 : (0 : ℤ) ≤ ⌈queue a c Δ (b + 1) / (c * Δ)⌉ := Int.ceil_nonneg hx0
    rw [← Int.ceil_toNat]
    omega
  rw [hnat]
  apply clears ha hcΔ.le
  calc queue a c Δ (b + 1) = queue a c Δ (b + 1) / (c * Δ) * (c * Δ) := by
        rw [div_mul_cancel₀ _ hcΔ.ne']
    _ ≤ (⌈queue a c Δ (b + 1) / (c * Δ)⌉₊ : ℝ) * (c * Δ) :=
        mul_le_mul_of_nonneg_right (Nat.le_ceil _) hcΔ.le

/-! ### The queue in the check

With `base = cue + q99`, `τ` the walk with no fire and `W` the time spent
queueing, RSET is `base + τ + W` and ASET is `L + τ`. -/

/-- The latest start the check asks of a place whose occupants wait `W` at a door. -/
noncomputable def requiredQ (base τ f m W : ℝ) : ℝ := Performance.required base (τ + W) f m + W

/-- A pass is the margin and the ratio with the wait counted in RSET. -/
theorem requiredQ_iff {base τ f m W L : ℝ} :
    requiredQ base τ f m W ≤ L ↔
      m ≤ (L + τ) - (base + τ + W) ∧ f * (base + τ + W) ≤ L + τ := by
  unfold requiredQ Performance.required
  rw [← le_sub_iff_add_le, max_le_iff]
  have e : f * (base + τ + W) = f * base + (f - 1) * (τ + W) + (τ + W) := by ring
  constructor
  · rintro ⟨h1, h2⟩
    exact ⟨by linarith, by linarith⟩
  · rintro ⟨h1, h2⟩
    exact ⟨by linarith, by linarith⟩

/-- With no wait it is the check's required start. -/
theorem requiredQ_zero {base τ f m : ℝ} :
    requiredQ base τ f m 0 = Performance.required base τ f m := by
  unfold requiredQ
  simp only [add_zero]

/-- A longer wait never asks less, while the factor is at least 0. -/
theorem requiredQ_mono {base τ f m W W' : ℝ} (hf : 0 ≤ f) (hW : W ≤ W') :
    requiredQ base τ f m W ≤ requiredQ base τ f m W' := by
  unfold requiredQ Performance.required
  rw [← le_sub_iff_add_le, max_le_iff]
  have e : f * base + (f - 1) * (τ + W') + W' - (f * base + (f - 1) * (τ + W) + W) =
      f * W' - f * W := by ring
  have hfW := mul_le_mul_of_nonneg_left hW hf
  constructor
  · linarith [le_max_left (base + m) (f * base + (f - 1) * (τ + W'))]
  · linarith [le_max_right (base + m) (f * base + (f - 1) * (τ + W'))]

/-- Queues never raise the share that starts in time: when the exit-adjusted
start `g s` is never earlier than `s` and meets `L`, the share that has
started by `s` is at most the share the check credits at `L`. -/
theorem queued_share_le {F g : ℝ → ℝ} (hF : Monotone F) (hg : ∀ s, s ≤ g s) {s cue L : ℝ}
    (h : g s ≤ L) : F (s - cue) ≤ F (L - cue) :=
  hF (by linarith [hg s])

/-- With a monotone exit-adjusted start, a start earlier than one that passes
passes too. -/
theorem safe_starts_downward {g : ℝ → ℝ} (hg : Monotone g) {s s' L : ℝ} (hs : s ≤ s')
    (h : g s' ≤ L) : g s ≤ L :=
  le_trans (hg hs) h

end FireEgress.DoorQueue
