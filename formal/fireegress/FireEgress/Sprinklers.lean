import Mathlib

/-!
# Sprinklers: layout on a ceiling, heating of the link, activation, decay

Specifications for the planned suppression feature
(`wiki/research/fire-suppression-sprinklers-2026-11-07.md`), proved before
any of it is implemented.

* **Layout.** Along a ceiling of length `L` with a largest spacing `S`,
  `n = ⌈L / S⌉` heads at the centres of `n` equal bays are `L / n ≤ S`
  apart and `L / (2n) ≤ S / 2` from each wall, and every point of the
  length is within `L / (2n)` of one. A bay count across chosen from the
  protection area `A` keeps the area each head covers within `A`.
* **The link.** The heat-responsive link of a sprinkler follows
  `dT/dt = (√u / RTI)(Tg - T) - (C / RTI)(T - Tm)` (Heskestad and Bill's
  conduction factor `C`, mount temperature `Tm`). Its explicit Euler step,
  with `k = Δt √u / RTI ≥ 0`, `c = Δt C / RTI ≥ 0` and `k + c ≤ 1`, is a
  convex combination of the link, the gas and the mount temperatures: it
  stays between them (a discrete maximum principle), and it is monotone in
  both the gas and the link temperature. So a hotter gas history never
  opens a head later.
* **Closed form.** For a constant gas temperature and no conduction, the
  exact solution and the activation time
  `t = (RTI / √u) ln ((Tg - T0) / (Tg - Ta))`, which shortens as the gas
  gets hotter.
* **After it opens.** A heat release decaying as `Q0 exp (-(t - t0) / τ)`
  (the form of Evans's sprinkler suppression correlation) never rises and
  never exceeds `Q0`.
-/

namespace FireEgress.Sprinklers

open Real

/-! ## Layout along one direction -/

/-- The number of equal bays along a length `L` with a largest spacing `S`. -/
noncomputable def bays (L S : ℝ) : ℕ := ⌈L / S⌉₊

theorem bays_pos {L S : ℝ} (hL : 0 < L) (hS : 0 < S) : 0 < bays L S :=
  Nat.ceil_pos.mpr (div_pos hL hS)

/-- Heads in adjacent bays are no more than `S` apart. -/
theorem spacing_le {L S : ℝ} (hL : 0 < L) (hS : 0 < S) : L / bays L S ≤ S := by
  have hn : (0 : ℝ) < bays L S := by exact_mod_cast bays_pos hL hS
  have h : L / S ≤ (bays L S : ℝ) := Nat.le_ceil (L / S)
  rw [div_le_iff₀ hS] at h
  rw [div_le_iff₀ hn]
  linarith

/-- The end heads are no more than `S / 2` from the walls. -/
theorem wall_distance_le {L S : ℝ} (hL : 0 < L) (hS : 0 < S) :
    L / (2 * bays L S) ≤ S / 2 := by
  have h := spacing_le hL hS
  have : L / (2 * bays L S) = (L / bays L S) / 2 := by ring
  rw [this]
  linarith

/-- The centre of bay `i` of `n` equal bays along `[0, L]`. -/
noncomputable def centre (L : ℝ) (n i : ℕ) : ℝ := (i + 1 / 2) * (L / n)

/-- Every point of the length is within half a bay of a head. -/
theorem covered {L : ℝ} {n : ℕ} (hL : 0 < L) (hn : 0 < n) {x : ℝ}
    (hx0 : 0 ≤ x) (hxL : x ≤ L) :
    ∃ i < n, |x - centre L n i| ≤ L / (2 * n) := by
  have hnR : (0 : ℝ) < n := by exact_mod_cast hn
  set h := L / n with hh
  have hpos : 0 < h := div_pos hL hnR
  -- The bay that holds x, the last one for x = L.
  let i := min ⌊x / h⌋₊ (n - 1)
  refine ⟨i, lt_of_le_of_lt (min_le_right _ _) (Nat.sub_lt hn one_pos), ?_⟩
  have hhalf : L / (2 * n) = h / 2 := by rw [hh]; field_simp
  rw [hhalf, centre, ← hh, abs_le]
  have hfl : (⌊x / h⌋₊ : ℝ) ≤ x / h := Nat.floor_le (div_nonneg hx0 hpos.le)
  have hfl' : x / h < ⌊x / h⌋₊ + 1 := Nat.lt_floor_add_one (x / h)
  have hxh : x / h ≤ n := by rw [div_le_iff₀ hpos, hh]; field_simp; linarith
  by_cases hc : ⌊x / h⌋₊ ≤ n - 1
  · have hi : (i : ℝ) = ⌊x / h⌋₊ := by simp [i, min_eq_left hc]
    rw [hi]
    have h1 : (⌊x / h⌋₊ : ℝ) * h ≤ x := by rwa [le_div_iff₀ hpos] at hfl
    have h2 : x < ((⌊x / h⌋₊ : ℝ) + 1) * h := by rwa [div_lt_iff₀ hpos] at hfl'
    constructor <;> nlinarith
  · -- x / h ≥ n, so x = L and the last bay holds it.
    push Not at hc
    have hi : i = n - 1 := min_eq_right hc.le
    have hge : (n : ℝ) ≤ x / h := by
      have : (n - 1 : ℕ) + 1 ≤ ⌊x / h⌋₊ := hc
      have hcast : ((n - 1 : ℕ) : ℝ) + 1 = n := by
        rw [Nat.cast_sub (by omega : 1 ≤ n)]; simp
      calc (n : ℝ) = ((n - 1 : ℕ) : ℝ) + 1 := hcast.symm
        _ ≤ (⌊x / h⌋₊ : ℝ) := by exact_mod_cast this
        _ ≤ x / h := hfl
    have hxeq : x / h = n := le_antisymm hxh hge
    have hx : x = n * h := by field_simp at hxeq ⊢; linarith
    have hcast : ((n - 1 : ℕ) : ℝ) = n - 1 := by rw [Nat.cast_sub (by omega : 1 ≤ n)]; simp
    rw [hi, hcast, hx]
    constructor <;> nlinarith

/-- Across: with `nx` bays along and at least `⌈(L / nx) W / A⌉` across, each
head covers at most `A`. -/
theorem area_le {L W A : ℝ} {nx ny : ℕ} (hL : 0 < L) (hW : 0 < W) (hA : 0 < A)
    (hnx : 0 < nx) (hny : ⌈(L / nx) * W / A⌉₊ ≤ ny) :
    (L / nx) * (W / ny) ≤ A := by
  have hnxR : (0 : ℝ) < nx := by exact_mod_cast hnx
  have hSx : 0 < L / nx := div_pos hL hnxR
  have hpos : 0 < (L / nx) * W / A := div_pos (mul_pos hSx hW) hA
  have hny0 : 0 < ny := lt_of_lt_of_le (Nat.ceil_pos.mpr hpos) hny
  have hnyR : (0 : ℝ) < ny := by exact_mod_cast hny0
  have hc : (L / nx) * W / A ≤ ny := le_trans (Nat.le_ceil _) (by exact_mod_cast hny)
  rw [div_le_iff₀ hA] at hc
  rw [← mul_div_assoc, div_le_iff₀ hnyR]
  linarith

/-! ## The link -/

/-- One explicit Euler step of the link: gas heating with `k = Δt √u / RTI`,
conduction to the mount at `Tm` with `c = Δt C / RTI`. -/
def linkStep (k c Tm T Tg : ℝ) : ℝ := T + k * (Tg - T) - c * (T - Tm)

/-- The step is a convex combination of the link, the gas and the mount. -/
theorem linkStep_eq (k c Tm T Tg : ℝ) :
    linkStep k c Tm T Tg = (1 - k - c) * T + k * Tg + c * Tm := by
  unfold linkStep; ring

/-- Discrete maximum principle: the link stays within the range of its own,
the gas and the mount temperatures. -/
theorem linkStep_bounded {k c Tm T Tg lo hi : ℝ} (hk : 0 ≤ k) (hc : 0 ≤ c)
    (hkc : k + c ≤ 1) (hT : lo ≤ T ∧ T ≤ hi) (hg : lo ≤ Tg ∧ Tg ≤ hi)
    (hm : lo ≤ Tm ∧ Tm ≤ hi) :
    lo ≤ linkStep k c Tm T Tg ∧ linkStep k c Tm T Tg ≤ hi := by
  rw [linkStep_eq]
  have hw : 0 ≤ 1 - k - c := by linarith
  constructor <;> nlinarith [mul_le_mul_of_nonneg_left hT.1 hw, mul_le_mul_of_nonneg_left hT.2 hw,
    mul_le_mul_of_nonneg_left hg.1 hk, mul_le_mul_of_nonneg_left hg.2 hk,
    mul_le_mul_of_nonneg_left hm.1 hc, mul_le_mul_of_nonneg_left hm.2 hc]

/-- Hotter gas, hotter link. -/
theorem linkStep_mono_gas {k c Tm T Tg Tg' : ℝ} (hk : 0 ≤ k) (h : Tg ≤ Tg') :
    linkStep k c Tm T Tg ≤ linkStep k c Tm T Tg' := by
  unfold linkStep; nlinarith

/-- A hotter link stays hotter, while `k + c ≤ 1`. -/
theorem linkStep_mono_link {k c Tm T T' Tg : ℝ} (hkc : k + c ≤ 1) (h : T ≤ T') :
    linkStep k c Tm T Tg ≤ linkStep k c Tm T' Tg := by
  rw [linkStep_eq, linkStep_eq]
  have hw : 0 ≤ 1 - k - c := by linarith
  nlinarith

/-- The link after `n` steps of a gas temperature history `g`. -/
def linkAfter (k c Tm T0 : ℝ) (g : ℕ → ℝ) : ℕ → ℝ
  | 0 => T0
  | n + 1 => linkStep k c Tm (linkAfter k c Tm T0 g n) (g n)

/-- Comparison: under a gas history at least as hot at every step, the link
is at least as hot at every step. -/
theorem linkAfter_mono {k c Tm T0 : ℝ} (hk : 0 ≤ k) (hkc : k + c ≤ 1)
    {g g' : ℕ → ℝ} (hg : ∀ n, g n ≤ g' n) :
    ∀ n, linkAfter k c Tm T0 g n ≤ linkAfter k c Tm T0 g' n
  | 0 => le_rfl
  | n + 1 => by
      simp only [linkAfter]
      exact le_trans (linkStep_mono_link hkc (linkAfter_mono hk hkc hg n))
        (linkStep_mono_gas hk (hg n))

/-- Hotter gas never opens a head later: whenever the link under the cooler
history has reached the activation temperature, so has the other. -/
theorem activates_no_later {k c Tm T0 Ta : ℝ} (hk : 0 ≤ k) (hkc : k + c ≤ 1)
    {g g' : ℕ → ℝ} (hg : ∀ n, g n ≤ g' n) {n : ℕ}
    (h : Ta ≤ linkAfter k c Tm T0 g n) : Ta ≤ linkAfter k c Tm T0 g' n :=
  le_trans h (linkAfter_mono hk hkc hg n)

/-- The link never passes the hottest of its start, the mount and the gas:
a head opens only if the gas reaches its activation temperature. -/
theorem linkAfter_le {k c Tm T0 hi : ℝ} (hk : 0 ≤ k) (hc : 0 ≤ c) (hkc : k + c ≤ 1)
    {g : ℕ → ℝ} (h0 : T0 ≤ hi) (hm : Tm ≤ hi) (hg : ∀ n, g n ≤ hi) :
    ∀ n, linkAfter k c Tm T0 g n ≤ hi
  | 0 => h0
  | n + 1 => by
      simp only [linkAfter]
      rw [linkStep_eq]
      have hw : 0 ≤ 1 - k - c := by linarith
      have ih := linkAfter_le hk hc hkc h0 hm hg n
      nlinarith [mul_le_mul_of_nonneg_left ih hw, mul_le_mul_of_nonneg_left (hg n) hk,
        mul_le_mul_of_nonneg_left hm hc]

/-- The exact step over `Δt` with the gas and mount held constant, as FDS
integrates the link: it relaxes toward `(a Tg + b Tm) / (a + b)` at the rate
`a + b`, with `a = √u / RTI` and `b = C / RTI`. -/
noncomputable def exactStep (a b dt Tm T Tg : ℝ) : ℝ :=
  (a * Tg + b * Tm) / (a + b) + (T - (a * Tg + b * Tm) / (a + b)) * exp (-((a + b) * dt))

/-- The exact step is the Euler step with `k = (1 - w) a / (a + b)` and
`c = (1 - w) b / (a + b)`, `w = exp (-(a + b) Δt)` ... -/
theorem exactStep_eq_linkStep {a b : ℝ} (hab : 0 < a + b) (dt Tm T Tg : ℝ) :
    exactStep a b dt Tm T Tg =
      linkStep ((1 - exp (-((a + b) * dt))) * a / (a + b))
        ((1 - exp (-((a + b) * dt))) * b / (a + b)) Tm T Tg := by
  unfold exactStep linkStep
  field_simp
  ring

/-- ... whose weights meet the hypotheses of every theorem above for any
`Δt ≥ 0`: the exact step needs no bound on the time step. -/
theorem exactStep_weights {a b dt : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : 0 < a + b)
    (hdt : 0 ≤ dt) :
    0 ≤ (1 - exp (-((a + b) * dt))) * a / (a + b) ∧
      0 ≤ (1 - exp (-((a + b) * dt))) * b / (a + b) ∧
      (1 - exp (-((a + b) * dt))) * a / (a + b) +
        (1 - exp (-((a + b) * dt))) * b / (a + b) ≤ 1 := by
  have hw : exp (-((a + b) * dt)) ≤ 1 := exp_le_one_iff.mpr (by nlinarith)
  have hw0 : 0 < exp (-((a + b) * dt)) := exp_pos _
  refine ⟨div_nonneg (mul_nonneg (by linarith) ha) hab.le,
    div_nonneg (mul_nonneg (by linarith) hb) hab.le, ?_⟩
  rw [← add_div, ← mul_add, mul_div_assoc, div_self hab.ne', mul_one]
  linarith

/-! ## Constant gas temperature: the exact link and the activation time -/

/-- The link's exact temperature under a constant gas temperature `Tg`, with
`a = √u / RTI` and no conduction. -/
noncomputable def linkExact (a T0 Tg t : ℝ) : ℝ := Tg - (Tg - T0) * exp (-(a * t))

theorem linkExact_zero (a T0 Tg : ℝ) : linkExact a T0 Tg 0 = T0 := by
  simp [linkExact]

/-- It solves `dT/dt = a (Tg - T)`. -/
theorem linkExact_deriv (a T0 Tg t : ℝ) :
    HasDerivAt (linkExact a T0 Tg) (a * (Tg - linkExact a T0 Tg t)) t := by
  have h1 : HasDerivAt (fun t => -(a * t)) (-a) t := by
    have h := ((hasDerivAt_id t).const_mul a).neg
    simp only [id_eq, mul_one] at h
    exact h
  have h2 := (h1.exp.const_mul (Tg - T0)).const_sub Tg
  have heq : a * (Tg - linkExact a T0 Tg t) = -((Tg - T0) * (exp (-(a * t)) * -a)) := by
    unfold linkExact; ring
  rw [heq]
  exact h2

/-- The activation time for `T0 < Ta < Tg`. -/
noncomputable def activation (a T0 Tg Ta : ℝ) : ℝ := (1 / a) * log ((Tg - T0) / (Tg - Ta))

/-- The link reaches `Ta` exactly at the activation time. -/
theorem linkExact_activation {a T0 Tg Ta : ℝ} (ha : 0 < a) (h0 : T0 < Ta) (h1 : Ta < Tg) :
    linkExact a T0 Tg (activation a T0 Tg Ta) = Ta := by
  have hpos : 0 < (Tg - T0) / (Tg - Ta) := div_pos (by linarith) (by linarith)
  have hexp : exp (-(a * activation a T0 Tg Ta)) = (Tg - Ta) / (Tg - T0) := by
    have : a * activation a T0 Tg Ta = log ((Tg - T0) / (Tg - Ta)) := by
      unfold activation; field_simp
    rw [this, ← log_inv, exp_log (inv_pos.mpr hpos), inv_div]
  have hne : Tg - T0 ≠ 0 := (by linarith : (0 : ℝ) < Tg - T0).ne'
  unfold linkExact
  rw [hexp, mul_div_cancel₀ (Tg - Ta) hne]
  ring

/-- Hotter gas, earlier activation. -/
theorem activation_antitone {a T0 Ta Tg Tg' : ℝ} (ha : 0 < a) (h0 : T0 < Ta) (h1 : Ta < Tg)
    (h : Tg ≤ Tg') : activation a T0 Tg' Ta ≤ activation a T0 Tg Ta := by
  unfold activation
  apply mul_le_mul_of_nonneg_left _ (le_of_lt (one_div_pos.mpr ha))
  apply log_le_log (div_pos (by linarith) (by linarith))
  rw [div_le_div_iff₀ (by linarith) (by linarith)]
  nlinarith

/-- The closed form is the only solution of `dT/dt = a (Tg - T)` from `T0`. -/
theorem linkExact_unique {a T0 Tg : ℝ} (f : ℝ → ℝ)
    (hf : ∀ t, HasDerivAt f (a * (Tg - f t)) t) (h0 : f 0 = T0) (t : ℝ) :
    f t = linkExact a T0 Tg t := by
  have hg : ∀ s, HasDerivAt (fun s => (f s - Tg) * exp (a * s)) 0 s := by
    intro s
    have he : HasDerivAt (fun s => exp (a * s)) (exp (a * s) * a) s := by
      have := ((hasDerivAt_id s).const_mul a).exp
      simpa using this
    have := ((hf s).sub_const Tg).mul he
    convert this using 1
    ring
  have hconst : (f t - Tg) * exp (a * t) = (f 0 - Tg) * exp (a * 0) :=
    is_const_of_deriv_eq_zero (fun s => (hg s).differentiableAt) (fun s => (hg s).deriv) t 0
  rw [h0, mul_zero, exp_zero, mul_one] at hconst
  have hE : exp (a * t) * exp (-(a * t)) = 1 := by rw [← exp_add]; simp
  have key : f t - Tg = (T0 - Tg) * exp (-(a * t)) := by
    calc f t - Tg = (f t - Tg) * (exp (a * t) * exp (-(a * t))) := by rw [hE, mul_one]
      _ = ((f t - Tg) * exp (a * t)) * exp (-(a * t)) := by ring
      _ = (T0 - Tg) * exp (-(a * t)) := by rw [hconst]
  unfold linkExact
  linear_combination key

/-- While the gas is hotter than the link, the link rises strictly. -/
theorem linkExact_strictMono {a T0 Tg : ℝ} (ha : 0 < a) (h : T0 < Tg) :
    StrictMono (linkExact a T0 Tg) := by
  intro s t hst
  unfold linkExact
  have : exp (-(a * t)) < exp (-(a * s)) := exp_lt_exp.mpr (by nlinarith)
  nlinarith

/-- Before the activation time the head is shut. -/
theorem below_before_activation {a T0 Tg Ta : ℝ} (ha : 0 < a) (h0 : T0 < Ta) (h1 : Ta < Tg)
    {t : ℝ} (ht : t < activation a T0 Tg Ta) : linkExact a T0 Tg t < Ta := by
  rw [← linkExact_activation ha h0 h1]
  exact linkExact_strictMono ha (by linarith) ht

/-- A gas no hotter than the activation temperature never opens the head. -/
theorem never_opens {a T0 Tg Ta : ℝ} (h0 : T0 < Tg) (h : Tg ≤ Ta) (t : ℝ) :
    linkExact a T0 Tg t < Ta := by
  unfold linkExact
  have : 0 < (Tg - T0) * exp (-(a * t)) := mul_pos (by linarith) (exp_pos _)
  linarith

/-- With `a = √u / RTI`, the activation time is proportional to the RTI. -/
theorem activation_rti (s RTI T0 Tg Ta : ℝ) :
    activation (s / RTI) T0 Tg Ta = RTI * activation s T0 Tg Ta := by
  unfold activation
  rw [one_div_div]
  ring

/-! ## After a head opens -/

/-- A heat release decaying from `Q0` at `t0` with time constant `τ`. -/
noncomputable def decay (Q0 t0 τ t : ℝ) : ℝ := Q0 * exp (-((t - t0) / τ))

theorem decay_le {Q0 t0 τ t : ℝ} (hQ : 0 ≤ Q0) (hτ : 0 < τ) (ht : t0 ≤ t) :
    decay Q0 t0 τ t ≤ Q0 := by
  unfold decay
  have : exp (-((t - t0) / τ)) ≤ 1 := by
    rw [exp_le_one_iff]; have := div_nonneg (sub_nonneg.mpr ht) hτ.le; linarith
  nlinarith [exp_pos (-((t - t0) / τ))]

theorem decay_antitone {Q0 t0 τ : ℝ} (hQ : 0 ≤ Q0) (hτ : 0 < τ) {t t' : ℝ} (h : t ≤ t') :
    decay Q0 t0 τ t' ≤ decay Q0 t0 τ t := by
  unfold decay
  apply mul_le_mul_of_nonneg_left _ hQ
  apply exp_le_exp.mpr
  have : (t - t0) / τ ≤ (t' - t0) / τ := div_le_div_of_nonneg_right (by linarith) hτ.le
  linarith

/-- Two exact steps under the same gas compose into one: the step is the
solution of the link's equation, whatever the step length. -/
theorem exactStep_add {a b : ℝ} (dt1 dt2 Tm T Tg : ℝ) :
    exactStep a b (dt1 + dt2) Tm T Tg = exactStep a b dt2 Tm (exactStep a b dt1 Tm T Tg) Tg := by
  unfold exactStep
  have : exp (-((a + b) * (dt1 + dt2))) = exp (-((a + b) * dt1)) * exp (-((a + b) * dt2)) := by
    rw [← exp_add]; congr 1; ring
  rw [this]
  ring

/-- Evans's time constant, `τ = 3.0 w^(-1.85)` s with `w` in mm/s. -/
noncomputable def evansTau (w : ℝ) : ℝ := 3 * w ^ (-(1.85 : ℝ))

/-- More water, faster decay. -/
theorem evansTau_antitone {w w' : ℝ} (hw : 0 < w) (h : w ≤ w') : evansTau w' ≤ evansTau w := by
  unfold evansTau
  rw [Real.rpow_neg hw.le, Real.rpow_neg (by linarith)]
  have h1 : w ^ (1.85 : ℝ) ≤ w' ^ (1.85 : ℝ) := Real.rpow_le_rpow hw.le h (by norm_num)
  have h2 : 0 < w ^ (1.85 : ℝ) := Real.rpow_pos_of_pos hw _
  have := inv_anti₀ h2 h1
  linarith

/-- The heat released after a head opens, stepped at `dt`, is bounded however
long the run: at most `Q dt / (1 - exp (-dt/τ))`, about `Q (τ + dt)`. -/
theorem evans_energy {Q dt τ : ℝ} (hQ : 0 ≤ Q) (hdt : 0 < dt) (hτ : 0 < τ) (N : ℕ) :
    ∑ n ∈ Finset.range N, decay Q 0 τ (n * dt) * dt ≤ Q * dt / (1 - exp (-(dt / τ))) := by
  set r := exp (-(dt / τ)) with hr
  have hr0 : 0 < r := exp_pos _
  have hr1 : r < 1 := by
    have hneg : -(dt / τ) < 0 := by have := div_pos hdt hτ; linarith
    have := exp_lt_exp.mpr hneg
    rwa [exp_zero] at this
  have hterm : ∀ n : ℕ, decay Q 0 τ (n * dt) * dt = Q * dt * r ^ n := by
    intro n
    unfold decay
    rw [hr, ← Real.exp_nat_mul]
    have : -(((n : ℝ) * dt - 0) / τ) = (n : ℝ) * -(dt / τ) := by ring
    rw [this]
    ring
  simp_rw [hterm]
  rw [← Finset.mul_sum]
  have h1r : 0 < 1 - r := by linarith
  have hS : (∑ n ∈ Finset.range N, r ^ n) * (1 - r) = 1 - r ^ N := geom_sum_mul_neg r N
  have hpow : 0 ≤ r ^ N := pow_nonneg hr0.le N
  have hSle : (∑ n ∈ Finset.range N, r ^ n) ≤ 1 / (1 - r) := by
    rw [le_div_iff₀ h1r]
    linarith
  calc Q * dt * ∑ n ∈ Finset.range N, r ^ n ≤ Q * dt * (1 / (1 - r)) :=
        mul_le_mul_of_nonneg_left hSle (mul_nonneg hQ hdt.le)
    _ = Q * dt / (1 - r) := by ring

end FireEgress.Sprinklers
