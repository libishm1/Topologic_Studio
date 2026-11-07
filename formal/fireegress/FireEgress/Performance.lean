import Mathlib

/-!
# The performance-based egress check

`src/core/performanceCheck.js`: a place passes when its escape deadline `L`
(from the forecast's deadline map) is at least the required start,
`cue + q99 + margin`, where `cue` is detection plus alarm and `q99` the
pre-movement time's 99th percentile. Pre-movement is a log-normal fitted to
its 1st and 99th percentiles.

* `fitted_upper`, `fitted_lower`, `sigma_pos`: the fitted log-normal passes
  through both percentiles exactly.
* `pass_share`: a pass means at least the percentile's share of occupants
  start in time, for any monotone distribution function.
* `share_mono`: a later deadline never lowers that share.
* `pass_on_map_sound`: the map is never later than the exact dose deadline
  (`Deadline.studio_map_le_exact` concludes `(j δ : EReal) ≤ doseDeadline`),
  so a pass on the map is a pass on the exact deadline.
* `pass_antitone`, `storey_pass_mono`: a pass under longer detection, alarm,
  pre-movement or margin is a pass under shorter ones, and a storey that
  passes keeps passing when its deadlines grow.
* `ratio_iff`, `ratio_margin`: with a factor `f` on RSET (the walk with no
  fire as its travel term), ASET / RSET ≥ f is `L ≥ f base + (f - 1) τ`,
  a margin of at least `f - 1` times RSET.
* `required_pass`, `required_one`, `required_pass_base`, `required_mono`:
  the check's required start meets both the margin and the ratio, is the
  plain start at `f = 1`, implies a pass at the percentile for `f ≥ 1`, and
  never falls as the factor grows.
-/

namespace FireEgress.Performance

open Real

/-- The log-normal's location fitted to the percentiles `p1` and `p99`. -/
noncomputable def mu (p1 p99 : ℝ) : ℝ := (log p1 + log p99) / 2

/-- Its scale, with `z` the standard normal's 99th percentile. -/
noncomputable def sigma (p1 p99 z : ℝ) : ℝ := (log p99 - log p1) / (2 * z)

theorem fitted_upper {p1 p99 z : ℝ} (h2 : 0 < p99) (hz : 0 < z) :
    exp (mu p1 p99 + sigma p1 p99 z * z) = p99 := by
  have : mu p1 p99 + sigma p1 p99 z * z = log p99 := by
    unfold mu sigma; field_simp; ring
  rw [this, exp_log h2]

theorem fitted_lower {p1 p99 z : ℝ} (h1 : 0 < p1) (hz : 0 < z) :
    exp (mu p1 p99 - sigma p1 p99 z * z) = p1 := by
  have : mu p1 p99 - sigma p1 p99 z * z = log p1 := by
    unfold mu sigma; field_simp; ring
  rw [this, exp_log h1]

theorem sigma_pos {p1 p99 z : ℝ} (h1 : 0 < p1) (h : p1 < p99) (hz : 0 < z) :
    0 < sigma p1 p99 z := by
  unfold sigma
  apply div_pos _ (by linarith)
  have := log_lt_log h1 h
  linarith

/-- The latest start the check accepts. -/
def start (cue q99 margin : ℝ) : ℝ := cue + q99 + margin

/-- A pass at the percentile `q` means at least its share `s` starts in time. -/
theorem pass_share {F : ℝ → ℝ} (hF : Monotone F) {q s cue L margin : ℝ} (hq : s ≤ F q)
    (hm : 0 ≤ margin) (h : start cue q margin ≤ L) : s ≤ F (L - cue) := by
  unfold start at h
  exact le_trans hq (hF (by linarith))

/-- A later deadline never lowers the share that starts in time. -/
theorem share_mono {F : ℝ → ℝ} (hF : Monotone F) {cue L L' : ℝ} (h : L ≤ L') :
    F (L - cue) ≤ F (L' - cue) :=
  hF (by linarith)

/-- Sound: with the map's time at most the exact dose deadline, a pass on the
map is a pass on the exact deadline. -/
theorem pass_on_map_sound {m s : ℝ} {D : EReal} (hmap : ((m : ℝ) : EReal) ≤ D) (hp : s ≤ m) :
    ((s : ℝ) : EReal) ≤ D :=
  le_trans (EReal.coe_le_coe_iff.mpr hp) hmap

/-- Robust: passing with longer cue, pre-movement or margin passes with shorter ones. -/
theorem pass_antitone {cue cue' q q' m m' L : ℝ} (h1 : cue' ≤ cue) (h2 : q' ≤ q) (h3 : m' ≤ m)
    (hp : start cue q m ≤ L) : start cue' q' m' ≤ L := by
  unfold start at *
  linarith

/-- A storey that passes keeps passing when its deadlines grow. -/
theorem storey_pass_mono {ι : Type*} {S : Finset ι} {L L' : ι → ℝ} (h : ∀ v ∈ S, L v ≤ L' v) {t : ℝ}
    (hp : ∀ v ∈ S, t ≤ L v) : ∀ v ∈ S, t ≤ L' v :=
  fun v hv => le_trans (hp v hv) (h v hv)

/-! ### A factor on RSET

With `base = cue + q99` and `τ` the walk with no fire, RSET is `base + τ`
and ASET the deadline plus the same walk, `L + τ`. -/

/-- The ratio criterion: ASET / RSET ≥ f is L ≥ f base + (f - 1) τ. -/
theorem ratio_iff {L base τ f : ℝ} (h : 0 < base + τ) :
    f ≤ (L + τ) / (base + τ) ↔ f * base + (f - 1) * τ ≤ L := by
  rw [le_div_iff₀ h]
  constructor <;> intro hh <;> linarith [mul_add f base τ, sub_mul f 1 τ]

/-- So the margin at the percentile, L - base, is at least f - 1 times RSET. -/
theorem ratio_margin {L base τ f : ℝ} :
    f * base + (f - 1) * τ ≤ L ↔ (f - 1) * (base + τ) ≤ L - base := by
  constructor <;> intro hh <;> linarith [mul_add (f - 1) base τ, sub_mul f 1 base]

/-- The latest start the check asks of a place: at least `m` seconds of
margin, and the ratio `f`. -/
noncomputable def required (base τ f m : ℝ) : ℝ := max (base + m) (f * base + (f - 1) * τ)

/-- A pass meets both. -/
theorem required_pass {L base τ f m : ℝ} (hp : required base τ f m ≤ L) :
    base + m ≤ L ∧ f * base + (f - 1) * τ ≤ L :=
  ⟨le_trans (le_max_left _ _) hp, le_trans (le_max_right _ _) hp⟩

/-- With f = 1 and a margin of at least 0 it is the plain required start. -/
theorem required_one {cue q99 τ m : ℝ} (hm : 0 ≤ m) :
    required (cue + q99) τ 1 m = start cue q99 m := by
  unfold required start
  rw [max_eq_left (by linarith)]

/-- A pass with f ≥ 1 is a pass at the percentile: the 99th percentile starts in time. -/
theorem required_pass_base {L base τ f m : ℝ} (hf : 1 ≤ f) (h : 0 ≤ base + τ)
    (hp : required base τ f m ≤ L) : base ≤ L := by
  have := (required_pass hp).2
  nlinarith [mul_nonneg (sub_nonneg.mpr hf) h]

/-- A larger factor never asks less, while RSET is at least 0. -/
theorem required_mono {base τ f f' m : ℝ} (hff : f ≤ f') (h : 0 ≤ base + τ) :
    required base τ f m ≤ required base τ f' m := by
  unfold required
  apply max_le_max le_rfl
  nlinarith [mul_le_mul_of_nonneg_right hff h]

end FireEgress.Performance
