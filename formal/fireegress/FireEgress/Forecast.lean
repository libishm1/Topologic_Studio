import Mathlib
import FireEgress.TimeDependent
import FireEgress.MinDose

/-!
# Planning against the fire forecast (LG1, LG2)

Studio's forecast routers (`topologicpy-web-frontend/src/core/forecast.js`, gap analysis
2026-10-27, gaps LG1 and LG2) run the fire model to its horizon first and then route on the
state each node will be in when the evacuee gets there. This file states what the code claims
and proves it, reusing `TimeDependent` (time-dependent Dijkstra), `GraphNumerics` (closure
windows), `Tenability` (the convective FED law and sliding-window doses) and `MinDose`
(holdover).

## The model, as in the code

* **Clock.** Fire step `k` covers `[k dt, (k + 1) dt)`; the horizon is `H = K0 dt`.
* **Forecast** (`buildForecast`). Per node the heat is stored as its running maximum
  (`runMax`); if memory requires, one frame is kept per `stride` steps, holding the state of
  the window's last step (`windowEnd`). Rates are `r(T) = T^3.4 / (5e7 · 60)` per second
  (`doseRate`, ISO 13571 convective FED with `t_I = 5e7 T^-3.4` min).
* **Closing.** `c v = min (u v) H`, `u v` the start of the first step at which the stored heat
  reaches the threshold (`closeStep`).
* **Network** (`Net`). Links `E u v` (both directions of every unblocked edge), walking time
  `tau u v = l / speed ≥ 0`, closing times `close v`, and `dose u v t`: the FED of the move
  `u → v` leaving at `t` (`linkDose(u, v, t, t + tau)`, the integral of the hotter end's
  piecewise-constant rate).
* **Safe walks** (`Safe`). A walk is safe if it arrives at every node strictly before that
  node closes, the start included (`t0 < close s`). Walks do not wait (`SafeW` allows it).

## Results

1. *Forecast construction.* `runMax_monotone`, `le_runMax`, `runMax_le` (least monotone
   majorant), `runMax_of_monotone` (Studio's models never cool, so it changes nothing);
   `coarse_conservative`, `fine_le_coarse`, `coarse_dose_ge` (holding a window's last value
   never lowers a rate or a dose); `closed_iff`, `closed_at_iff`, `closeStep_runMax`
   (closing times: a node is closed at exactly the steps from its first passage on, and the
   first passage of the stored heat is that of the model's own heat).
2. *FED rate.* `doseRate_eq_tConv` (the code's rate is `1 / (60 Tenability.tConv)`),
   `doseRate_monotone`, `doseRate_strictMonoOn` (via `Tenability.T3_strictAntiOn`),
   `stepRate_monotone`, `max_monotone`, `window_monotone` (via
   `Tenability.T10_edge_dose_mono_entry`), `linkDose_monotone`, `linkDose_nonneg`,
   `forecast_rate_monotone`.
3. *LG1.* `arr_fifo`, `arr_ge` (the closing link `GraphNumerics.closeArc` is FIFO and never
   early), `safe_iff_arr` (safe walks are exactly the finite-time walks of
   `TimeDependent.Arr`), and the router: `earliestArrival_exact` (the first exit popped
   carries the earliest safe arrival over all exits), `earliestArrival_none` (an exhausted
   queue means no safe route), `earliestArrival_settled` (every settled label, hence every
   time on the returned route, is that node's earliest safe arrival),
   `earliestArrival_sweep` (`TimeDependent.closure_window_exact`).
4. *LG2, dominance.* `dominance_preserved`; `waiting_never_helps` (a walk that may wait is
   matched by a no-wait walk arriving no later with no more dose); `dp_no_holdover` (on a
   whole-step clock the forecast's link doses satisfy `MinDose.no_holdover`'s exact
   hypothesis).
5. *LG2, the label loop of `doseFront`.* The loop is modelled as a relation `FStep`
   (mirroring `TimeDependent.Step`): pop a label of least time; discard it unless its dose is
   below `best[v]` and `bestExit`; accept an exit label onto the front; expand any other
   label, pushing each allowed move that passes the same test. The invariant `FInv` holds
   along every run (`finv_reachable`). At exhaustion (empty heap):
   (a) `front_complete`: every safe walk to an exit is matched or beaten on both counts by a
       front point; (b) `front_sound`: every front point is a safe walk's (arrival, dose),
       at every point of the run; (c) `front_head_earliest`: the first point's arrival is the
       earliest safe arrival; (d) `front_last_leastDose`: the last point's dose is the least
       dose of any safe walk; `front_sorted`: arrival non-decreasing, dose strictly falling.
   `front_minimal` says exactly when a front point can be weakly beaten: only by a *later*
   front point with the *same* arrival time. `front_pareto` (no two points tied in time ⇒
   every point is Pareto-optimal). `keepStrict_front`: the filter the code had before 2026-10-28 ("keep the
   strict front") returned the front unchanged, so it did not remove the tied point; the code
   now keeps only the last point of a tie, which `front_minimal` shows is not beaten.
   `tie_counterexample`: a four-node run to exhaustion whose front is
   `[(2 s, dose 2), (2 s, dose 0)]`; the first point (the plan's `fastest`) is beaten at the
   same arrival. The code returned this front on the same network before the tie filter was
   fixed (property test, now a regression test).
6. *Snapshot planning walks into the fire.* `snapshot_walks_into_fire`: on the five-node
   network of the test "a route planned on the fire at departure can walk into it", the
   shortest route on the departure-time fire reaches node 1 at 2 s, 0.5 s after it closes,
   while the earliest safe route arrives at 10 s. The snapshot plan is `earliestArrival` with
   no closing times on the graph without the links of nodes untenable at departure, that is
   static Dijkstra (`TimeDependent.static_hyps`) to the nearest exit; `snapshot_plan_unsafe`
   shows that whichever shortest route it returns (any tie-breaking), indeed every route
   shorter than 10 s, passes node 1 after it has closed.

## What is left between this file and the code

* Floating point (IEEE doubles, `Float32Array` rates) is not modelled; arithmetic is in `ℝ`.
* `linkDose`'s frame-by-frame sum is modelled by the integral of the step rate it sums
  (`linkDose`); the identity of the sum and the integral is checked by the property test,
  not proved. Times before 0 are outside the model (`Nat.floor` sends them to frame 0, as the
  code now does; before 2026-10-28 it read `rate` at a negative index and got `NaN`).
* The heaps are abstracted to "pop an entry of least key" (`Selectable`, `Least`); a
  binary heap with lazy deletion is not verified. For `earliestArrival`, the code never
  queues an `Infinity` label and relaxes into settled nodes, which is a no-op since their
  label is no later; a vertex the model labels `⊤` is one the code leaves at `Infinity`, and
  an empty code heap is the model's "every unsettled label is `⊤`".
* `doseFront`'s heap is a multiset; the model's is a set. A duplicate `(node, time, dose)`
  is discarded on its second pop in the code, so the outputs agree. Parent pointers are not
  modelled; `best[]` and `bestExit` are, with `Infinity` as `⊤`.
* The `maxLabels` truncation is reported by the code, not modelled; termination of the label
  loop is not proved (the code's label budget bounds it).
* Parallel links: the network has one link per ordered pair. A longer parallel link is
  dominated by the shorter (later arrival, and a window integral over a longer interval of a
  non-negative rate), so dropping it changes no front point.
-/

namespace FireEgress.Forecast

open MeasureTheory

/-! ## (1) Forecast construction -/

section Construction

/-- The running maximum of a heat history `T 0, T 1, …`: `buildForecast`'s `peak[v]`, updated
by `if (T > peak[v]) peak[v] = T` from `peak[v] = -Infinity`. -/
def runMax (T : ℕ → ℝ) : ℕ → ℝ
  | 0 => T 0
  | k + 1 => max (runMax T k) (T (k + 1))

/-- The stored heat never falls: the running maximum is monotone in the step. For Studio:
whatever the fire model does, the forecast the routers read is a fire that only grows. -/
theorem runMax_monotone (T : ℕ → ℝ) : Monotone (runMax T) :=
  monotone_nat_of_le_succ fun k => le_max_left (runMax T k) (T (k + 1))

/-- The stored heat is at least the model's heat at every step: never optimistic. -/
theorem le_runMax (T : ℕ → ℝ) : ∀ k, T k ≤ runMax T k
  | 0 => le_rfl
  | k + 1 => le_max_right (runMax T k) (T (k + 1))

/-- The running maximum is the least monotone sequence above the field: the smallest
correction that makes the forecast a fire that only grows. -/
theorem runMax_le {T g : ℕ → ℝ} (hg : Monotone g) (hle : ∀ k, T k ≤ g k) :
    ∀ k, runMax T k ≤ g k
  | 0 => hle 0
  | k + 1 => max_le ((runMax_le hg hle k).trans (hg (Nat.le_succ k))) (hle (k + 1))

/-- For a field that never cools (Studio's three models: diffusion only heats, a burning node
stays burning) the running maximum is the field itself: the header's "the maximum changes
nothing for them". -/
theorem runMax_of_monotone {T : ℕ → ℝ} (hT : Monotone T) (k : ℕ) : runMax T k = T k :=
  le_antisymm (runMax_le hT (fun _ => le_rfl) k) (le_runMax T k)

/-- The stored heat has reached `θ` at step `k` exactly when the model's heat reached `θ` at
some step up to `k`. -/
theorem le_runMax_iff (T : ℕ → ℝ) (θ : ℝ) : ∀ k, θ ≤ runMax T k ↔ ∃ i ≤ k, θ ≤ T i
  | 0 => ⟨fun h => ⟨0, le_rfl, h⟩, fun ⟨i, hi, h⟩ => by
      obtain rfl : i = 0 := by omega
      exact h⟩
  | k + 1 => by
    show θ ≤ max (runMax T k) (T (k + 1)) ↔ _
    rw [le_max_iff, le_runMax_iff T θ k]
    constructor
    · rintro (⟨i, hi, h⟩ | h)
      · exact ⟨i, by omega, h⟩
      · exact ⟨k + 1, le_rfl, h⟩
    · rintro ⟨i, hi, h⟩
      rcases Nat.lt_or_ge i (k + 1) with hlt | hge
      · exact Or.inl ⟨i, by omega, h⟩
      · obtain rfl : i = k + 1 := by omega
        exact Or.inr h

/-- Frame `j` of a forecast that keeps one frame per `stride` fire steps (of `K0`) holds the
state of the last step of its window, `min ((j + 1) stride - 1) (K0 - 1)`: `take` writes row
`⌊k0 / stride⌋` when `(k0 + 1) % stride = 0` or `k0 = K0 - 1`. -/
def windowEnd (stride K0 j : ℕ) : ℕ := min ((j + 1) * stride - 1) (K0 - 1)

/-- Every step `k` (clamped to the last step) lies in window `k / stride`, at or before the
step that window holds. -/
theorem min_le_windowEnd {stride : ℕ} (hs : 0 < stride) (K0 k : ℕ) :
    min k (K0 - 1) ≤ windowEnd stride K0 (k / stride) := by
  unfold windowEnd
  have h1 := Nat.lt_div_mul_add (a := k) hs
  have h2 : (k / stride + 1) * stride = k / stride * stride + stride := by ring
  have h3 : k ≤ (k / stride + 1) * stride - 1 := by omega
  exact min_le_min h3 le_rfl

/-- Later frames hold later steps. -/
theorem windowEnd_monotone (stride K0 : ℕ) : Monotone (windowEnd stride K0) := by
  intro i j hij
  unfold windowEnd
  exact min_le_min (Nat.sub_le_sub_right (Nat.mul_le_mul_right _ (by omega)) 1) le_rfl

/-- Coarsening is conservative: for a sequence that never falls (the stored heat, or its
rate), the value a frame holds is at least the value at every step of its window. For
Studio: a coarsened forecast never shows a node cooler than the model had it. -/
theorem coarse_conservative {g : ℕ → ℝ} (hg : Monotone g) {stride : ℕ} (hs : 0 < stride)
    {K0 k : ℕ} (hk : k < K0) : g k ≤ g (windowEnd stride K0 (k / stride)) := by
  have h := min_le_windowEnd hs K0 k
  rw [min_eq_left (by omega)] at h
  exact hg h

/-- The rate at time `s` read from the full forecast: step `⌊s / dt⌋`, clamped to the last
step `K0 - 1`. -/
noncomputable def fineRate (R : ℕ → ℝ) (dt : ℝ) (K0 : ℕ) (s : ℝ) : ℝ :=
  R (min ⌊s / dt⌋₊ (K0 - 1))

/-- The rate at time `s` read from the coarsened forecast: frame `⌊s / (stride dt)⌋`, which
holds the last step of its window. -/
noncomputable def coarseRate (R : ℕ → ℝ) (dt : ℝ) (stride K0 : ℕ) (s : ℝ) : ℝ :=
  R (windowEnd stride K0 ⌊s / (stride * dt)⌋₊)

/-- Pointwise in time, the coarsened forecast's rate is at least the full forecast's. -/
theorem fine_le_coarse {R : ℕ → ℝ} (hR : Monotone R) {stride : ℕ} (hs : 0 < stride) (dt : ℝ)
    (K0 : ℕ) (s : ℝ) : fineRate R dt K0 s ≤ coarseRate R dt stride K0 s := by
  unfold fineRate coarseRate
  rw [mul_comm (stride : ℝ) dt, ← div_div, Nat.floor_div_natCast]
  exact hR (min_le_windowEnd hs K0 _)

theorem fineRate_monotone {R : ℕ → ℝ} (hR : Monotone R) {dt : ℝ} (hdt : 0 < dt) (K0 : ℕ) :
    Monotone (fineRate R dt K0) := fun _ _ h =>
  hR (min_le_min (Nat.floor_le_floor (div_le_div_of_nonneg_right h hdt.le)) le_rfl)

theorem coarseRate_monotone {R : ℕ → ℝ} (hR : Monotone R) {dt : ℝ} (hdt : 0 < dt)
    {stride : ℕ} (hs : 0 < stride) (K0 : ℕ) : Monotone (coarseRate R dt stride K0) :=
  fun _ _ h => hR (windowEnd_monotone stride K0
    (Nat.floor_le_floor (div_le_div_of_nonneg_right h (by positivity))))

/-- Coarsening never lowers a dose: over any interval the integral of the coarsened rate is
at least the integral of the full one. For Studio: when `buildForecast` coarsens to fit in
memory, every link dose (and so every front point's dose) can only go up. -/
theorem coarse_dose_ge {R : ℕ → ℝ} (hR : Monotone R) {stride : ℕ} (hs : 0 < stride) {dt : ℝ}
    (hdt : 0 < dt) (K0 : ℕ) {a b : ℝ} (hab : a ≤ b) :
    ∫ s in a..b, fineRate R dt K0 s ≤ ∫ s in a..b, coarseRate R dt stride K0 s :=
  intervalIntegral.integral_mono_on hab (fineRate_monotone hR hdt K0).intervalIntegrable
    (coarseRate_monotone hR hdt hs K0).intervalIntegrable
    (fun s _ => fine_le_coarse hR hs dt K0 s)

open Classical in
/-- The step at which a node closes: the first step `k < K0` whose stored heat `P k` reaches
the threshold `θ`, or `K0` (the horizon) if none does. `closes[v] = closeStep · dt`, that is
`min(untenable[v], horizon)`. -/
noncomputable def closeStep (P : ℕ → ℝ) (θ : ℝ) (K0 : ℕ) : ℕ :=
  if h : ∃ k, k < K0 ∧ θ ≤ P k then Nat.find h else K0

/-- A node closes by the horizon. -/
theorem closeStep_le (P : ℕ → ℝ) (θ : ℝ) (K0 : ℕ) : closeStep P θ K0 ≤ K0 := by
  unfold closeStep
  split_ifs with h
  · exact (Nat.find_spec h).1.le
  · exact le_rfl

/-- Untenability from a monotone field. Before the horizon, a node is untenable at step `k`
(stored heat `≥ θ`) exactly when `k` is at or after its closing step: the untenable steps are
an up-set, so "closed at `k`" implies "closed at every later step". Reuses
`Tenability.T12_first_passage`. -/
theorem closed_iff {P : ℕ → ℝ} (hP : Monotone P) (θ : ℝ) {K0 k : ℕ} (hk : k < K0) :
    θ ≤ P k ↔ closeStep P θ K0 ≤ k := by
  unfold closeStep
  constructor
  · intro h
    have hex : ∃ k, k < K0 ∧ θ ≤ P k := ⟨k, hk, h⟩
    rw [dite_eq_left hex]
    exact Nat.find_min' hex ⟨hk, h⟩
  · intro h
    split_ifs at h with hex
    · have hex' : ∃ n, θ ≤ P n := ⟨_, (Nat.find_spec hex).2⟩
      have hfirst := (Tenability.T12_first_passage P hP θ hex' k).2
      exact hfirst ((Nat.find_min' hex' (Nat.find_spec hex).2).trans h)
    · omega

/-- Closing times in seconds. With `dt > 0`, for a time `t ≥ 0` before the horizon,
`t ≥ c = closeStep · dt` exactly when the stored heat of the step containing `t` has reached
`θ`. So `close v ≤ t` is the forecast's own "untenable at `t`", and it holds at every later
time. -/
theorem closed_at_iff {P : ℕ → ℝ} (hP : Monotone P) (θ : ℝ) {K0 : ℕ} {dt t : ℝ}
    (hdt : 0 < dt) (ht : 0 ≤ t) (hH : t < K0 * dt) :
    (closeStep P θ K0 : ℝ) * dt ≤ t ↔ θ ≤ P ⌊t / dt⌋₊ := by
  have htd : 0 ≤ t / dt := div_nonneg ht hdt.le
  have hk : ⌊t / dt⌋₊ < K0 := (Nat.floor_lt htd).2 ((div_lt_iff₀ hdt).2 hH)
  rw [closed_iff hP θ hk, Nat.le_floor_iff htd, le_div_iff₀ hdt]

/-- The closing step of the stored heat (running maximum) is the first step at which the
model's own heat reaches `θ`: storing the maximum does not move any closing time, it only
keeps a node closed if the model's heat falls back. -/
theorem closeStep_runMax (T : ℕ → ℝ) (θ : ℝ) (K0 : ℕ) :
    closeStep (runMax T) θ K0 = closeStep T θ K0 := by
  unfold closeStep
  by_cases h2 : ∃ k, k < K0 ∧ θ ≤ T k
  · have h1 : ∃ k, k < K0 ∧ θ ≤ runMax T k :=
      let ⟨k, hk, h⟩ := h2; ⟨k, hk, h.trans (le_runMax T k)⟩
    rw [dite_eq_left h1, dite_eq_left h2]
    apply le_antisymm
    · exact Nat.find_min' h1 ⟨(Nat.find_spec h2).1, (Nat.find_spec h2).2.trans (le_runMax T _)⟩
    · obtain ⟨i, hi, hTi⟩ := (le_runMax_iff T θ _).1 (Nat.find_spec h1).2
      exact (Nat.find_min' h2 ⟨by have := (Nat.find_spec h1).1; omega, hTi⟩).trans hi
  · have h1 : ¬ ∃ k, k < K0 ∧ θ ≤ runMax T k := by
      rintro ⟨k, hk, h⟩
      obtain ⟨i, hi, hTi⟩ := (le_runMax_iff T θ k).1 h
      exact h2 ⟨i, by omega, hTi⟩
    rw [dite_eq_right h1, dite_eq_right h2]

end Construction

/-! ## (2) The FED rate and link doses -/

section Rate

/-- `convectiveDoseRate`: FED per second at `T` degC, `T^3.4 / (5e7 · 60)` for `T > 0` and
`0` otherwise (ISO 13571 convective heat, `t_I = 5e7 T^-3.4` min). -/
noncomputable def doseRate (T : ℝ) : ℝ := if 0 < T then T ^ (3.4 : ℝ) / (5e7 * 60) else 0

/-- The code's rate is the reciprocal of `Tenability.tConv` (min), per second. Reuses
`Tenability.T3_rate_monotone`. -/
theorem doseRate_eq_tConv {T : ℝ} (hT : 0 < T) : doseRate T = 1 / Tenability.tConv T / 60 := by
  have h := (Tenability.T3_rate_monotone (fun _ => T) monotone_const (fun _ => hT.le)).2 0 hT
  rw [doseRate, ite_eq_left hT, h]
  ring

theorem doseRate_nonneg (T : ℝ) : 0 ≤ doseRate T := by
  unfold doseRate
  split_ifs with h
  · exact div_nonneg (Real.rpow_nonneg h.le _) (by norm_num)
  · exact le_rfl

/-- The FED rate never falls as the heat rises, for every real `T` (it is `0` up to `0`). So
the rate of a heat that never falls never falls. -/
theorem doseRate_monotone : Monotone doseRate := by
  intro a b hab
  unfold doseRate
  split_ifs with ha hb hb
  · exact div_le_div_of_nonneg_right (Real.rpow_le_rpow ha.le hab (by norm_num)) (by norm_num)
  · exact absurd (ha.trans_le hab) hb
  · exact div_nonneg (Real.rpow_nonneg hb.le _) (by norm_num)
  · exact le_rfl

/-- For `T > 0` the rate is strictly increasing. Reuses `Tenability.T3_strictAntiOn`. -/
theorem doseRate_strictMonoOn : StrictMonoOn doseRate (Set.Ioi 0) := by
  intro a ha b hb hab
  have ha' : (0 : ℝ) < a := ha
  have hb' : (0 : ℝ) < b := hb
  rw [doseRate_eq_tConv ha', doseRate_eq_tConv hb']
  have h := Tenability.T3_strictAntiOn ha hb hab
  have hpos : 0 < Tenability.tConv b := by
    unfold Tenability.tConv
    exact mul_pos (by norm_num) (Real.rpow_pos_of_pos hb' _)
  exact div_lt_div_of_pos_right (one_div_lt_one_div_of_lt hpos h) (by norm_num)

/-- A per-frame rate `R k` read at time `s`: frame `⌊s / f⌋` (`rate[k * n + v]` with
`k = Math.floor(s / frameSeconds)` in `linkDose`). -/
noncomputable def stepRate (R : ℕ → ℝ) (f s : ℝ) : ℝ := R ⌊s / f⌋₊

/-- A piecewise-constant rate whose frames never fall never falls in time. -/
theorem stepRate_monotone {R : ℕ → ℝ} (hR : Monotone R) {f : ℝ} (hf : 0 < f) :
    Monotone (stepRate R f) :=
  fun _ _ h => hR (Nat.floor_le_floor (div_le_div_of_nonneg_right h hf.le))

/-- A link takes the hotter of its two ends: the maximum of two rates that never fall never
falls. -/
theorem max_monotone {r r' : ℝ → ℝ} (h : Monotone r) (h' : Monotone r') :
    Monotone (fun s => max (r s) (r' s)) :=
  fun _ _ hab => max_le_max (h hab) (h' hab)

/-- The sliding-window integral `t ↦ ∫_t^{t+τ} r` of a rate that never falls is
non-decreasing in `t`. Reuses `Tenability.T10_edge_dose_mono_entry`. -/
theorem window_monotone {r : ℝ → ℝ} (hr : Monotone r) {τ : ℝ} (hτ : 0 ≤ τ) :
    Monotone (fun t => ∫ s in t..t + τ, r s) :=
  fun _ _ hab => Tenability.T10_edge_dose_mono_entry r hr τ hτ hab

/-- The sliding-window integral of a non-negative rate is non-negative. -/
theorem window_nonneg {r : ℝ → ℝ} (hr0 : ∀ s, 0 ≤ r s) {τ : ℝ} (hτ : 0 ≤ τ) (t : ℝ) :
    0 ≤ ∫ s in t..t + τ, r s :=
  intervalIntegral.integral_nonneg (by linarith) (fun s _ => hr0 s)

/-- `linkDose(v, w, t, t + τ)`: the integral over the walk's window of the hotter end's
piecewise-constant rate (`Rv`, `Rw` per frame, frames of `f` seconds). -/
noncomputable def linkDose (Rv Rw : ℕ → ℝ) (f t τ : ℝ) : ℝ :=
  ∫ s in t..t + τ, max (stepRate Rv f s) (stepRate Rw f s)

/-- LG2's hypothesis on links. With frames that never fall, a link's dose never falls with
the departure time: leaving later never costs less. -/
theorem linkDose_monotone {Rv Rw : ℕ → ℝ} (hv : Monotone Rv) (hw : Monotone Rw) {f : ℝ}
    (hf : 0 < f) {τ : ℝ} (hτ : 0 ≤ τ) : Monotone (fun t => linkDose Rv Rw f t τ) :=
  window_monotone (max_monotone (stepRate_monotone hv hf) (stepRate_monotone hw hf)) hτ

/-- With non-negative rates every link dose is non-negative. -/
theorem linkDose_nonneg {Rv Rw : ℕ → ℝ} (hv : ∀ k, 0 ≤ Rv k) (f : ℝ) {τ : ℝ} (hτ : 0 ≤ τ)
    (t : ℝ) : 0 ≤ linkDose Rv Rw f t τ :=
  window_nonneg (fun s => (hv _).trans (le_max_left _ (stepRate Rw f s))) hτ t

/-- The rates `buildForecast` stores, frame `j` of node `v`: the FED rate of the running
maximum of the model's heat at the step frame `j` holds. Whatever the model's field `T v`, they
are non-negative and non-decreasing in the frame, so every link dose built from them is
non-negative and non-decreasing in the departure time. -/
theorem forecast_rate_monotone {V : Type*} (T : V → ℕ → ℝ) (stride K0 : ℕ) (v : V) :
    Monotone (fun j => doseRate (runMax (T v) (windowEnd stride K0 j))) ∧
      ∀ j, 0 ≤ doseRate (runMax (T v) (windowEnd stride K0 j)) :=
  ⟨doseRate_monotone.comp ((runMax_monotone _).comp (windowEnd_monotone stride K0)),
    fun _ => doseRate_nonneg _⟩

/-- The link doses of the forecast `buildForecast` builds satisfy LG2's hypotheses: for
frames of `f > 0` seconds and a walking time `τ ≥ 0`, the dose of the move `u → w` is
non-negative and non-decreasing in the departure time. -/
theorem forecast_linkDose {V : Type*} (T : V → ℕ → ℝ) (stride K0 : ℕ) (u w : V) {f : ℝ}
    (hf : 0 < f) {τ : ℝ} (hτ : 0 ≤ τ) :
    let R := fun v j => doseRate (runMax (T v) (windowEnd stride K0 j))
    Monotone (fun t => linkDose (R u) (R w) f t τ) ∧ ∀ t, 0 ≤ linkDose (R u) (R w) f t τ :=
  ⟨linkDose_monotone (forecast_rate_monotone T stride K0 u).1
      (forecast_rate_monotone T stride K0 w).1 hf hτ,
    fun t => linkDose_nonneg (forecast_rate_monotone T stride K0 u).2 f hτ t⟩

end Rate

/-! ## (3) LG1: the earliest safe arrival -/

section Network

variable {V : Type*}

/-- A building under a forecast. -/
structure Net (V : Type*) where
  /-- `E u v`: an unblocked link from `u` to `v` (`csr` lists both directions). -/
  E : V → V → Prop
  /-- Walking time of the link, `l_e / speed` (s). -/
  tau : V → V → ℝ
  /-- Closing time `c v = min (u v) H` (s). -/
  close : V → ℝ
  /-- FED of the move `u → v` leaving at `t`: `linkDose(u, v, t, t + tau u v)`. -/
  dose : V → V → ℝ → ℝ

/-- What the proofs need of a network: links take non-negative time, and each link's dose is
non-negative and non-decreasing in the departure time (`forecast_linkDose`). -/
structure Net.Good (N : Net V) : Prop where
  tau_nonneg : ∀ u v, N.E u v → 0 ≤ N.tau u v
  dose_nonneg : ∀ u v, N.E u v → ∀ t, 0 ≤ N.dose u v t
  dose_mono : ∀ u v, N.E u v → Monotone (N.dose u v)

/-- `Safe N s t0 v t D`: some walk from `s`, leaving at `t0`, reaches `v` at time `t` with dose
`D`, arriving at every node (the start included) strictly before that node closes. -/
inductive Safe (N : Net V) (s : V) (t0 : ℝ) : V → ℝ → ℝ → Prop
  | start : t0 < N.close s → Safe N s t0 s t0 0
  | move {u v : V} {t D : ℝ} : Safe N s t0 u t D → N.E u v → t + N.tau u v < N.close v →
      Safe N s t0 v (t + N.tau u v) (D + N.dose u v t)

/-- Every safe walk starts at an open start: if the start has closed at departure there is no
safe walk at all (both routers return nothing). -/
theorem Safe.start_open {N : Net V} {s : V} {t0 : ℝ} {v : V} {t D : ℝ}
    (h : Safe N s t0 v t D) : t0 < N.close s := by
  induction h with
  | start h => exact h
  | move _ _ _ ih => exact ih

/-- A safe walk is at its end before that node closes. -/
theorem Safe.lt_close {N : Net V} {s : V} {t0 : ℝ} {v : V} {t D : ℝ}
    (h : Safe N s t0 v t D) : t < N.close v := by
  cases h with
  | start h => exact h
  | move _ _ hlt => exact hlt

/-- LG1: the arrival function of link `u → v`: arrive at `t + tau` if that is before `v`
closes, else never (`GraphNumerics.closeArc`). -/
noncomputable def Net.arr (N : Net V) (u v : V) : WithTop ℝ → WithTop ℝ :=
  GraphNumerics.closeArc (N.tau u v) (N.close v)

/-- LG1: every link's arrival function is FIFO (`GraphNumerics.R2_closeArc_fifo`). -/
theorem arr_fifo (N : Net V) (u v : V) : Monotone (N.arr u v) :=
  GraphNumerics.R2_closeArc_fifo _ _

/-- LG1: no link arrives before it is entered (`TimeDependent.closeArc_ge`). -/
theorem arr_ge {N : Net V} (hN : N.Good) {u v : V} (h : N.E u v) (x : WithTop ℝ) :
    x ≤ N.arr u v x :=
  TimeDependent.closeArc_ge _ _ (hN.tau_nonneg u v h) x

/-- Safe walks are exactly the walks of `TimeDependent.Arr` under the closing links that
arrive at a finite time. So the router's labels, which are `Arr` arrival times, are safe
arrival times, and a label `⊤` means "no safe walk". -/
theorem safe_iff_arr (N : Net V) {s : V} {t0 : ℝ} (h0 : t0 < N.close s) (v : V) (t : ℝ) :
    (∃ D, Safe N s t0 v t D) ↔
      TimeDependent.Arr N.E N.arr s (t0 : WithTop ℝ) v (t : WithTop ℝ) := by
  constructor
  · rintro ⟨D, h⟩
    induction h with
    | start _ => exact TimeDependent.Arr.nil
    | @move u v t D _ huv hlt ih =>
      refine (ih.snoc huv).cast ?_
      show GraphNumerics.closeArc _ _ (t : WithTop ℝ) = _
      rw [GraphNumerics.closeArc_coe, ite_eq_left hlt]
  · intro h
    suffices key : ∀ x y, TimeDependent.Arr N.E N.arr s (t0 : WithTop ℝ) x y →
        ∀ t : ℝ, y = t → ∃ D, Safe N s t0 x t D from key v t h t rfl
    intro x y hxy
    induction hxy with
    | nil =>
      intro t ht
      rw [← WithTop.coe_inj.mp ht]
      exact ⟨0, Safe.start h0⟩
    | @snoc u w y _ huw ih =>
      intro t ht
      induction y using WithTop.recTopCoe with
      | top => exact absurd ht (by rw [show N.arr u w ⊤ = ⊤ from rfl]; exact WithTop.top_ne_coe)
      | coe t' =>
        have ht' : GraphNumerics.closeArc (N.tau u w) (N.close w) (t' : WithTop ℝ) = t := ht
        rw [GraphNumerics.closeArc_coe] at ht'
        split_ifs at ht' with hlt
        · obtain ⟨D, hD⟩ := ih t' rfl
          rw [← WithTop.coe_inj.mp ht']
          exact ⟨_, hD.move huw hlt⟩
        · exact absurd ht' WithTop.top_ne_coe

/-- LG1, exactness of `earliestArrival`. Run the label-setting loop of `TimeDependent` with the
closing links from the start at the departure time. If the vertex popped next is an exit with
a finite label and no exit has been settled before (the code returns at the first exit it
pops), that label is the earliest safe arrival at any exit. Reuses
`TimeDependent.td_dijkstra_correct` and `TimeDependent.early_exit_lower_bound`. -/
theorem earliestArrival_exact [DecidableEq V] {N : Net V} (hN : N.Good) (isExit : V → Prop)
    {s : V} {t0 : ℝ} (h0 : t0 < N.close s) {σ₀ σ : TimeDependent.State V (WithTop ℝ)}
    (hinit : TimeDependent.Init s (t0 : WithTop ℝ) σ₀)
    (hrun : Relation.ReflTransGen (TimeDependent.Step N.E N.arr) σ₀ σ) {u : V}
    (hu : TimeDependent.Selectable σ u) (hx : isExit u) (hnone : ∀ x ∈ σ.S, ¬ isExit x)
    {t : ℝ} (ht : σ.d u = t) :
    IsLeast {a | ∃ x D, isExit x ∧ Safe N s t0 x a D} t := by
  have hfifo : ∀ u v, N.E u v → Monotone (N.arr u v) := fun u v _ => arr_fifo N u v
  have hge : ∀ u v, N.E u v → ∀ t, t ≤ N.arr u v t := fun _ _ h x => arr_ge hN h x
  have hopt := TimeDependent.td_dijkstra_correct N.E N.arr hfifo hge s _ σ₀ σ hinit hrun u hu
  refine ⟨?_, ?_⟩
  · obtain ⟨D, hD⟩ := (safe_iff_arr N h0 u t).2 (by rw [← ht]; exact hopt.1)
    exact ⟨u, D, hx, hD⟩
  · rintro a ⟨x, D, hxe, hsafe⟩
    have harr := (safe_iff_arr N h0 x a).1 ⟨D, hsafe⟩
    have hxS : x ∉ σ.S := fun h => hnone x h hxe
    have h := TimeDependent.early_exit_lower_bound N.E N.arr hfifo hge s _ σ₀ σ hinit hrun u hu
      x hxS harr
    rw [ht] at h
    exact WithTop.coe_le_coe.mp h

/-- LG1: `earliestArrival` returns `exit: -1` only when no safe route exists. If no exit has
been settled and every reached, unsettled vertex has label `⊤` (the code's heap is empty),
no safe walk reaches any exit. Reuses `TimeDependent.early_exit_lower_bound` and
`TimeDependent.td_dijkstra_exhausted`. -/
theorem earliestArrival_none [DecidableEq V] {N : Net V} (hN : N.Good) (isExit : V → Prop)
    {s : V} {t0 : ℝ} (h0 : t0 < N.close s) {σ₀ σ : TimeDependent.State V (WithTop ℝ)}
    (hinit : TimeDependent.Init s (t0 : WithTop ℝ) σ₀)
    (hrun : Relation.ReflTransGen (TimeDependent.Step N.E N.arr) σ₀ σ)
    (hnone : ∀ x ∈ σ.S, ¬ isExit x) (hdry : ∀ v ∈ σ.R, v ∉ σ.S → σ.d v = ⊤) :
    ¬ ∃ x a D, isExit x ∧ Safe N s t0 x a D := by
  have hfifo : ∀ u v, N.E u v → Monotone (N.arr u v) := fun u v _ => arr_fifo N u v
  have hge : ∀ u v, N.E u v → ∀ t, t ≤ N.arr u v t := fun _ _ h x => arr_ge hN h x
  rintro ⟨x, a, D, hx, hsafe⟩
  have harr := (safe_iff_arr N h0 x a).1 ⟨D, hsafe⟩
  have hxS : x ∉ σ.S := fun h => hnone x h hx
  by_cases hne : (σ.R \ σ.S).Nonempty
  · obtain ⟨u, hu, hmin⟩ := Finset.exists_min_image (σ.R \ σ.S) σ.d hne
    rw [Finset.mem_sdiff] at hu
    have hsel : TimeDependent.Selectable σ u :=
      ⟨hu.1, hu.2, fun v hv hvS => hmin v (Finset.mem_sdiff.2 ⟨hv, hvS⟩)⟩
    have h := TimeDependent.early_exit_lower_bound N.E N.arr hfifo hge s _ σ₀ σ hinit hrun u
      hsel x hxS harr
    rw [hdry u hu.1 hu.2] at h
    exact absurd h (not_le.2 (WithTop.coe_lt_top a))
  · have hsub : σ.R ⊆ σ.S := by
      intro v hv
      by_contra h
      exact hne ⟨v, Finset.mem_sdiff.2 ⟨hv, h⟩⟩
    exact hxS (((TimeDependent.td_dijkstra_exhausted N.E N.arr hfifo hge s _ σ₀ σ hinit hrun
      hsub).1 x).1 ⟨_, harr⟩)

/-- LG1: every settled label is exact. A settled vertex with finite label `t` has `t` as its
earliest safe arrival. For Studio: the `times` of the returned route (labels of settled
nodes on the parent chain) are each node's earliest safe arrival. Reuses
`TimeDependent.settled_optimal`. -/
theorem earliestArrival_settled [DecidableEq V] {N : Net V} (hN : N.Good) {s : V} {t0 : ℝ}
    (h0 : t0 < N.close s) {σ₀ σ : TimeDependent.State V (WithTop ℝ)}
    (hinit : TimeDependent.Init s (t0 : WithTop ℝ) σ₀)
    (hrun : Relation.ReflTransGen (TimeDependent.Step N.E N.arr) σ₀ σ) {v : V} (hv : v ∈ σ.S)
    {t : ℝ} (ht : σ.d v = t) : IsLeast {a | ∃ D, Safe N s t0 v a D} t := by
  have hfifo : ∀ u v, N.E u v → Monotone (N.arr u v) := fun u v _ => arr_fifo N u v
  have hge : ∀ u v, N.E u v → ∀ t, t ≤ N.arr u v t := fun _ _ h x => arr_ge hN h x
  have hopt := TimeDependent.settled_optimal N.E N.arr hfifo hge s _ σ₀ σ hinit hrun v hv
  rw [ht] at hopt
  refine ⟨(safe_iff_arr N h0 v t).2 hopt.1, fun a ha => ?_⟩
  exact WithTop.coe_le_coe.mp (hopt.2 ((safe_iff_arr N h0 v a).1 ha))

/-- LG1 as a full sweep: run to exhaustion, the router's output at every vertex is the
earliest arrival `EA` over all walks of the closing links (`⊤`: no safe walk). This is
`TimeDependent.closure_window_exact` with closing time `close v` on every link into `v`. -/
theorem earliestArrival_sweep [DecidableEq V] {N : Net V} (hN : N.Good) (s : V) (t0 : ℝ)
    {σ₀ σ : TimeDependent.State V (WithTop ℝ)} (hinit : TimeDependent.Init s (t0 : WithTop ℝ) σ₀)
    (hrun : Relation.ReflTransGen (TimeDependent.Step N.E N.arr) σ₀ σ) (hdry : σ.R ⊆ σ.S)
    (v : V) : TimeDependent.outLabel σ v = TimeDependent.EA N.E N.arr s (t0 : WithTop ℝ) v :=
  TimeDependent.closure_window_exact N.E N.tau (fun _ v => N.close v) hN.tau_nonneg s _ σ₀ σ
    hinit hrun hdry v

/-! ## (4) LG2: dominance and waiting -/

/-- LG2, the core lemma: dominance is preserved by extension. If at node `u` the pair
`(t1, D1)` arrives no later with no more dose than `(t2, D2)`, and the move `u → v` is allowed
from `(t2, D2)` (it arrives before `v` closes), then it is allowed from `(t1, D1)` and the
results stay ordered. For Studio: `doseFront` may discard any label beaten on both counts. -/
theorem dominance_preserved {N : Net V} (hN : N.Good) {u v : V} (huv : N.E u v)
    {t1 t2 D1 D2 : ℝ} (ht : t1 ≤ t2) (hD : D1 ≤ D2) (hok : t2 + N.tau u v < N.close v) :
    t1 + N.tau u v < N.close v ∧ t1 + N.tau u v ≤ t2 + N.tau u v ∧
      D1 + N.dose u v t1 ≤ D2 + N.dose u v t2 :=
  ⟨by linarith, by linarith, add_le_add hD (hN.dose_mono u v huv ht)⟩

/-- Safe walks that may also wait: at `v` from `t` to `t' ≥ t`, still before `v` closes,
accruing `wait v t t'` (for example `∫_t^{t'} r_v`). -/
inductive SafeW (N : Net V) (wait : V → ℝ → ℝ → ℝ) (s : V) (t0 : ℝ) : V → ℝ → ℝ → Prop
  | start : t0 < N.close s → SafeW N wait s t0 s t0 0
  | move {u v : V} {t D : ℝ} : SafeW N wait s t0 u t D → N.E u v → t + N.tau u v < N.close v →
      SafeW N wait s t0 v (t + N.tau u v) (D + N.dose u v t)
  | hold {v : V} {t t' D : ℝ} : SafeW N wait s t0 v t D → t ≤ t' → t' < N.close v →
      SafeW N wait s t0 v t' (D + wait v t t')

/-- LG2: waiting never helps. With link doses that never fall with the departure time and
non-negative waiting doses, every walk that may wait is matched by a walk that never waits,
ending at the same node no later and with no more dose. So `doseFront` may search no-wait
walks only. (The discrete, whole-step form is `MinDose.no_holdover`; see `dp_no_holdover`.) -/
theorem waiting_never_helps {N : Net V} (hN : N.Good) {wait : V → ℝ → ℝ → ℝ}
    (hw : ∀ v t t', t ≤ t' → 0 ≤ wait v t t') {s : V} {t0 : ℝ} {v : V} {t D : ℝ}
    (h : SafeW N wait s t0 v t D) : ∃ t' D', Safe N s t0 v t' D' ∧ t' ≤ t ∧ D' ≤ D := by
  induction h with
  | start h => exact ⟨_, _, Safe.start h, le_rfl, le_rfl⟩
  | @move u v t D _ huv hlt ih =>
    obtain ⟨t', D', h', ht, hD⟩ := ih
    obtain ⟨hok, ht', hD'⟩ := dominance_preserved hN huv ht hD hlt
    exact ⟨_, _, h'.move huv hok, ht', hD'⟩
  | @hold v t t'' D _ htt _ ih =>
    obtain ⟨t', D', h', ht, hD⟩ := ih
    exact ⟨t', D', h', ht.trans htt, by have := hw v t t'' htt; linarith⟩

/-- Every safe walk is a walk that may wait (and never does). -/
theorem safeW_of_safe {N : Net V} {wait : V → ℝ → ℝ → ℝ} {s : V} {t0 : ℝ} {v : V} {t D : ℝ}
    (h : Safe N s t0 v t D) : SafeW N wait s t0 v t D := by
  induction h with
  | start h => exact SafeW.start h
  | move _ huv hlt ih => exact ih.move huv hlt

end Network

/-- LG2 and `MinDose`. On a whole-step clock (step `Δ ≥ 0`, link `e` taking `tau e` steps), take
as move dose the forecast's link dose `ld e (k Δ)` while the move arrives before its head
closes and `⊤` after. If every `ld e` never falls with the departure time and every wait dose
is `≥ 0`, this is `MinDose`'s exact holdover hypothesis, so the DP without holdover arcs has
the same value as the DP with them (`MinDose.no_holdover`). -/
theorem dp_no_holdover {V ι : Type*} [Fintype ι] [DecidableEq V] (M : MinDose.Net V ι) {Δ : ℝ}
    (hΔ : 0 ≤ Δ) (close : V → ℝ) (ld : ι → ℝ → ℝ) (hld : ∀ e, Monotone (ld e))
    (hdose : ∀ e k, M.dose e k = if (k : ℝ) * Δ + (M.tau e : ℝ) * Δ < close (M.head e) then
      ((ld e ((k : ℝ) * Δ) : ℝ) : WithTop ℝ) else ⊤)
    (hw : ∀ v k, 0 ≤ M.wait v k) (H t : ℕ) (v : V) :
    MinDose.val M false H t v = MinDose.val M true H t v := by
  refine MinDose.no_holdover M (fun e k k' hk => ?_) hw H t v
  rw [hdose, hdose]
  have hkk : (k : ℝ) * Δ ≤ (k' : ℝ) * Δ := mul_le_mul_of_nonneg_right (Nat.cast_le.2 hk) hΔ
  split_ifs with h1 h2 h2
  · exact WithTop.coe_le_coe.2 (hld e hkk)
  · exact le_top
  · exact absurd (by linarith) h1
  · exact le_rfl

/-! ## (5) LG2: the label loop of `doseFront` -/

section Front

variable {V : Type*}

/-- A label of `doseFront`: node, arrival time, dose. The parent pointer is not modelled. -/
structure Lbl (V : Type*) where
  node : V
  time : ℝ
  dose : ℝ

/-- The label that one move along `L.node → w` makes from `L`. -/
def Lbl.next (N : Net V) (L : Lbl V) (w : V) : Lbl V :=
  ⟨w, L.time + N.tau L.node w, L.dose + N.dose L.node w L.time⟩

/-- The state of `doseFront`'s loop: the heap, the labels accepted so far (popped and kept),
`best[v]` and `bestExit` (`⊤` for `Infinity`), and the front (accepted exit labels in pop
order). -/
structure FState (V : Type*) where
  heap : Set (Lbl V)
  acc : Set (Lbl V)
  best : V → WithTop ℝ
  bestExit : WithTop ℝ
  front : List (Lbl V)

/-- The code's test, at a pop and at a push: `D < best[v] && D < bestExit`. -/
def Pass (σ : FState V) (L : Lbl V) : Prop :=
  (L.dose : WithTop ℝ) < σ.best L.node ∧ (L.dose : WithTop ℝ) < σ.bestExit

/-- The start of the loop: the start label alone in the heap, nothing accepted, `best` and
`bestExit` at `Infinity`, an empty front. (The code only enters the loop when
`depart < closes[start]`.) -/
def FInit (s : V) (t0 : ℝ) (σ : FState V) : Prop :=
  σ.heap = {(⟨s, t0, 0⟩ : Lbl V)} ∧ σ.acc = ∅ ∧ (∀ v, σ.best v = ⊤) ∧ σ.bestExit = ⊤ ∧
    σ.front = []

/-- `L` may be popped: it is in the heap with least time. Ties may be broken any way. -/
def Least (σ : FState V) (L : Lbl V) : Prop := L ∈ σ.heap ∧ ∀ L' ∈ σ.heap, L.time ≤ L'.time

/-- Pop `L` and keep it: `best[v] = D`. -/
def accept [DecidableEq V] (σ : FState V) (L : Lbl V) : FState V :=
  { σ with heap := σ.heap \ {L}, acc := insert L σ.acc,
           best := Function.update σ.best L.node (L.dose : WithTop ℝ) }

/-- The labels the expansion of `L` pushes: one per allowed move (`t2 < closes[w]`) that passes
the test against `best` and `bestExit` as they stand after `L` was kept (they do not change
during the expansion). -/
def pushed (N : Net V) (σ : FState V) (L : Lbl V) : Set (Lbl V) :=
  {L' | ∃ w, N.E L.node w ∧ L.time + N.tau L.node w < N.close w ∧ L' = L.next N w ∧ Pass σ L'}

/-- One iteration of `doseFront`'s loop: pop a label of least time; discard it if it fails the
test; keep an exit label as the next front point (`bestExit = D`; exits are absorbing); keep
any other label and push its allowed, passing moves. -/
inductive FStep [DecidableEq V] (N : Net V) (isExit : V → Prop) : FState V → FState V → Prop
  | discard {σ : FState V} {L : Lbl V} : Least σ L → ¬ Pass σ L →
      FStep N isExit σ { σ with heap := σ.heap \ {L} }
  | exit {σ : FState V} {L : Lbl V} : Least σ L → Pass σ L → isExit L.node →
      FStep N isExit σ
        { accept σ L with bestExit := (L.dose : WithTop ℝ), front := σ.front ++ [L] }
  | expand {σ : FState V} {L : Lbl V} : Least σ L → Pass σ L → ¬ isExit L.node →
      FStep N isExit σ { accept σ L with heap := (σ.heap \ {L}) ∪ pushed N (accept σ L) L }

/-- `L` is covered by the accepted labels: one of them, at `L`'s node or at an exit, arrives no
later than `L` with no more dose. -/
def Cov (isExit : V → Prop) (acc : Set (Lbl V)) (L : Lbl V) : Prop :=
  ∃ B ∈ acc, (B.node = L.node ∨ isExit B.node) ∧ B.time ≤ L.time ∧ B.dose ≤ L.dose

/-- The invariant of `doseFront`'s loop. Every label in the heap or accepted is a safe walk's;
`best[v]` is the least dose accepted at `v` and `bestExit` the least accepted at an exit
(attained, `⊤` if none); accepted labels are no later than heap labels; the front is the
accepted exit labels, in non-decreasing time and strictly falling dose; and every label the
loop owes (the start, and each allowed move out of an accepted non-exit label) is in the heap
or already covered. -/
structure FInv (N : Net V) (isExit : V → Prop) (s : V) (t0 : ℝ) (σ : FState V) : Prop where
  heap_safe : ∀ L ∈ σ.heap, Safe N s t0 L.node L.time L.dose
  acc_safe : ∀ L ∈ σ.acc, Safe N s t0 L.node L.time L.dose
  best_le : ∀ A ∈ σ.acc, σ.best A.node ≤ A.dose
  best_att : ∀ v (x : ℝ), σ.best v = x → ∃ A ∈ σ.acc, A.node = v ∧ A.dose = x
  exit_le : ∀ A ∈ σ.acc, isExit A.node → σ.bestExit ≤ A.dose
  exit_att : ∀ x : ℝ, σ.bestExit = x → ∃ A ∈ σ.acc, isExit A.node ∧ A.dose = x
  order : ∀ A ∈ σ.acc, ∀ L ∈ σ.heap, A.time ≤ L.time
  front_iff : ∀ A, A ∈ σ.front ↔ A ∈ σ.acc ∧ isExit A.node
  sorted : σ.front.Pairwise (fun p q => p.time ≤ q.time ∧ q.dose < p.dose)
  start : (⟨s, t0, 0⟩ : Lbl V) ∈ σ.heap ∨ Cov isExit σ.acc ⟨s, t0, 0⟩
  next : ∀ A ∈ σ.acc, ¬ isExit A.node → ∀ w, N.E A.node w →
    A.time + N.tau A.node w < N.close w → A.next N w ∈ σ.heap ∨ Cov isExit σ.acc (A.next N w)

variable {isExit : V → Prop}

theorem cov_mono {acc acc' : Set (Lbl V)} (h : acc ⊆ acc') {L : Lbl V}
    (hc : Cov isExit acc L) : Cov isExit acc' L :=
  let ⟨B, hB, h1, h2, h3⟩ := hc; ⟨B, h hB, h1, h2, h3⟩

theorem cov_self {acc : Set (Lbl V)} {L : Lbl V} (h : L ∈ acc) : Cov isExit acc L :=
  ⟨L, h, Or.inl rfl, le_rfl, le_rfl⟩

/-- A label that fails the test is covered, provided every accepted label is no later: the
code's comment "a new label is beaten on both counts exactly when its dose is not below
this". -/
theorem cov_of_not_pass {acc : Set (Lbl V)} {best : V → WithTop ℝ} {bestExit : WithTop ℝ}
    (hatt : ∀ v (x : ℝ), best v = x → ∃ A ∈ acc, A.node = v ∧ A.dose = x)
    (hxatt : ∀ x : ℝ, bestExit = x → ∃ A ∈ acc, isExit A.node ∧ A.dose = x)
    {L : Lbl V} (htime : ∀ A ∈ acc, A.time ≤ L.time)
    (h : ¬ ((L.dose : WithTop ℝ) < best L.node ∧ (L.dose : WithTop ℝ) < bestExit)) :
    Cov isExit acc L := by
  rw [not_and_or, not_lt, not_lt] at h
  rcases h with h | h
  · obtain ⟨x, hx⟩ := WithTop.ne_top_iff_exists.1 (ne_top_of_le_ne_top WithTop.coe_ne_top h)
    obtain ⟨A, hA, hAn, hAd⟩ := hatt L.node x hx.symm
    rw [← hx] at h
    exact ⟨A, hA, Or.inl hAn, htime A hA, by rw [hAd]; exact WithTop.coe_le_coe.1 h⟩
  · obtain ⟨x, hx⟩ := WithTop.ne_top_iff_exists.1 (ne_top_of_le_ne_top WithTop.coe_ne_top h)
    obtain ⟨A, hA, hAx, hAd⟩ := hxatt x hx.symm
    rw [← hx] at h
    exact ⟨A, hA, Or.inr hAx, htime A hA, by rw [hAd]; exact WithTop.coe_le_coe.1 h⟩

/-- An obligation in the heap or covered stays so when `L` is popped and kept. -/
theorem handled_accept {heap acc : Set (Lbl V)} {L X : Lbl V}
    (h : X ∈ heap ∨ Cov isExit acc X) : X ∈ heap \ {L} ∨ Cov isExit (insert L acc) X := by
  by_cases hX : X = L
  · subst hX
    exact Or.inr (cov_self (Set.mem_insert X acc))
  · rcases h with h | h
    · exact Or.inl ⟨h, hX⟩
    · exact Or.inr (cov_mono (Set.subset_insert L acc) h)

section Inv

variable {N : Net V} {s : V} {t0 : ℝ}

/-- An obligation in the heap or covered stays so when `L` is popped and discarded. -/
theorem handled_discard {σ : FState V} (hI : FInv N isExit s t0 σ) {L : Lbl V}
    (hL : Least σ L) (hp : ¬ Pass σ L) {X : Lbl V} (h : X ∈ σ.heap ∨ Cov isExit σ.acc X) :
    X ∈ σ.heap \ {L} ∨ Cov isExit σ.acc X := by
  by_cases hX : X = L
  · subst hX
    exact Or.inr (cov_of_not_pass hI.best_att hI.exit_att (fun A hA => hI.order A hA X hL.1) hp)
  · rcases h with h | h
    · exact Or.inl ⟨h, hX⟩
    · exact Or.inr h

/-- Keeping `L` keeps `best` right: the updated `best[v]` is still the least dose kept at `v`,
and attained. -/
theorem best_accept [DecidableEq V] {σ : FState V} (hI : FInv N isExit s t0 σ) {L : Lbl V}
    (hpass : Pass σ L) :
    (∀ A ∈ insert L σ.acc, Function.update σ.best L.node (L.dose : WithTop ℝ) A.node ≤ A.dose) ∧
      ∀ v (x : ℝ), Function.update σ.best L.node (L.dose : WithTop ℝ) v = x →
        ∃ A ∈ insert L σ.acc, A.node = v ∧ A.dose = x := by
  constructor
  · intro A hA
    by_cases hn : A.node = L.node
    · rw [hn, Function.update_self]
      rcases hA with rfl | hA
      · exact le_rfl
      · exact hpass.1.le.trans (hn ▸ hI.best_le A hA)
    · rw [Function.update_of_ne hn]
      rcases hA with rfl | hA
      · exact absurd rfl hn
      · exact hI.best_le A hA
  · intro v x hx
    by_cases hv : v = L.node
    · subst hv
      rw [Function.update_self] at hx
      exact ⟨L, Set.mem_insert L σ.acc, rfl, WithTop.coe_inj.mp hx⟩
    · rw [Function.update_of_ne hv] at hx
      obtain ⟨A, hA, h1, h2⟩ := hI.best_att v x hx
      exact ⟨A, Set.mem_insert_of_mem L hA, h1, h2⟩

/-- The loop starts in a state satisfying the invariant (when the start is open). -/
theorem finv_init (h0 : t0 < N.close s) {σ : FState V} (h : FInit s t0 σ) :
    FInv N isExit s t0 σ := by
  obtain ⟨hheap, hacc, hbest, hexit, hfront⟩ := h
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro L hL
    rw [hheap, Set.mem_singleton_iff] at hL
    subst hL
    exact Safe.start h0
  · intro L hL; rw [hacc] at hL; exact absurd hL (Set.notMem_empty L)
  · intro A hA; rw [hacc] at hA; exact absurd hA (Set.notMem_empty A)
  · intro v x hx; rw [hbest v] at hx; exact absurd hx WithTop.top_ne_coe
  · intro A hA; rw [hacc] at hA; exact absurd hA (Set.notMem_empty A)
  · intro x hx; rw [hexit] at hx; exact absurd hx WithTop.top_ne_coe
  · intro A hA; rw [hacc] at hA; exact absurd hA (Set.notMem_empty A)
  · intro A; rw [hfront, hacc]; simp
  · rw [hfront]; exact List.Pairwise.nil
  · left; rw [hheap]; exact Set.mem_singleton _
  · intro A hA; rw [hacc] at hA; exact absurd hA (Set.notMem_empty A)

/-- Discarding a label keeps the invariant. -/
theorem finv_discard {σ : FState V} (hI : FInv N isExit s t0 σ) {L : Lbl V} (hL : Least σ L)
    (hp : ¬ Pass σ L) : FInv N isExit s t0 { σ with heap := σ.heap \ {L} } where
  heap_safe L' h := hI.heap_safe L' h.1
  acc_safe := hI.acc_safe
  best_le := hI.best_le
  best_att := hI.best_att
  exit_le := hI.exit_le
  exit_att := hI.exit_att
  order A hA L' h := hI.order A hA L' h.1
  front_iff := hI.front_iff
  sorted := hI.sorted
  start := handled_discard hI hL hp hI.start
  next A hA hx w hw hlt := handled_discard hI hL hp (hI.next A hA hx w hw hlt)

/-- Keeping an exit label as the next front point keeps the invariant. -/
theorem finv_exit [DecidableEq V] {σ : FState V} (hI : FInv N isExit s t0 σ) {L : Lbl V}
    (hL : Least σ L) (hp : Pass σ L) (hx : isExit L.node) :
    FInv N isExit s t0
      { accept σ L with bestExit := (L.dose : WithTop ℝ), front := σ.front ++ [L] } := by
  obtain ⟨hb1, hb2⟩ := best_accept hI hp
  refine ⟨fun L' h => hI.heap_safe L' h.1, ?_, hb1, hb2, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro A hA
    rcases hA with rfl | hA
    · exact hI.heap_safe A hL.1
    · exact hI.acc_safe A hA
  · intro A hA hAx
    rcases hA with rfl | hA
    · exact le_rfl
    · exact hp.2.le.trans (hI.exit_le A hA hAx)
  · intro x hx'
    exact ⟨L, Set.mem_insert L σ.acc, hx, (WithTop.coe_inj.mp hx').symm ▸ rfl⟩
  · intro A hA L' hL'
    rcases hA with rfl | hA
    · exact hL.2 L' hL'.1
    · exact hI.order A hA L' hL'.1
  · intro A
    show A ∈ σ.front ++ [L] ↔ A ∈ insert L σ.acc ∧ isExit A.node
    rw [List.mem_append, List.mem_singleton, hI.front_iff A, Set.mem_insert_iff]
    constructor
    · rintro (⟨h1, h2⟩ | rfl)
      · exact ⟨Or.inr h1, h2⟩
      · exact ⟨Or.inl rfl, hx⟩
    · rintro ⟨rfl | h1, h2⟩
      · exact Or.inr rfl
      · exact Or.inl ⟨h1, h2⟩
  · show (σ.front ++ [L]).Pairwise _
    refine List.pairwise_append.2 ⟨hI.sorted, List.pairwise_singleton _ _, ?_⟩
    intro p hp' q hq
    rw [List.mem_singleton] at hq
    subst hq
    obtain ⟨hpa, hpx⟩ := (hI.front_iff p).1 hp'
    exact ⟨hI.order p hpa q hL.1,
      WithTop.coe_lt_coe.1 (lt_of_lt_of_le hp.2 (hI.exit_le p hpa hpx))⟩
  · exact handled_accept hI.start
  · intro A hA hAx w hw hlt
    rcases hA with rfl | hA
    · exact absurd hx hAx
    · exact handled_accept (hI.next A hA hAx w hw hlt)

/-- Expanding a non-exit label keeps the invariant. -/
theorem finv_expand [DecidableEq V] (hN : N.Good) {σ : FState V} (hI : FInv N isExit s t0 σ)
    {L : Lbl V} (hL : Least σ L) (hp : Pass σ L) (hx : ¬ isExit L.node) :
    FInv N isExit s t0
      { accept σ L with heap := (σ.heap \ {L}) ∪ pushed N (accept σ L) L } := by
  obtain ⟨hb1, hb2⟩ := best_accept hI hp
  have hpush : ∀ L' ∈ pushed N (accept σ L) L, ∃ w, N.E L.node w ∧
      L.time + N.tau L.node w < N.close w ∧ L' = L.next N w := fun L' ⟨w, h1, h2, h3, _⟩ =>
    ⟨w, h1, h2, h3⟩
  have hexit_att : ∀ x : ℝ, σ.bestExit = x → ∃ A ∈ insert L σ.acc, isExit A.node ∧ A.dose = x :=
    fun x hx' => let ⟨A, hA, h1, h2⟩ := hI.exit_att x hx'
      ⟨A, Set.mem_insert_of_mem L hA, h1, h2⟩
  have hLtime : ∀ A ∈ insert L σ.acc, A.time ≤ L.time := by
    intro A hA
    rcases hA with rfl | hA
    · exact le_rfl
    · exact hI.order A hA L hL.1
  refine ⟨?_, ?_, hb1, hb2, ?_, hexit_att, ?_, ?_, hI.sorted, ?_, ?_⟩
  · intro L' hL'
    rcases hL' with hL' | hL'
    · exact hI.heap_safe L' hL'.1
    · obtain ⟨w, hw, hlt, rfl⟩ := hpush L' hL'
      exact (hI.heap_safe L hL.1).move hw hlt
  · intro A hA
    rcases hA with rfl | hA
    · exact hI.heap_safe A hL.1
    · exact hI.acc_safe A hA
  · intro A hA hAx
    rcases hA with rfl | hA
    · exact absurd hAx hx
    · exact hI.exit_le A hA hAx
  · intro A hA L' hL'
    rcases hL' with hL' | hL'
    · rcases hA with rfl | hA
      · exact hL.2 L' hL'.1
      · exact hI.order A hA L' hL'.1
    · obtain ⟨w, hw, _, rfl⟩ := hpush L' hL'
      have := hN.tau_nonneg _ _ hw
      have := hLtime A hA
      simp only [Lbl.next]
      linarith
  · intro A
    show A ∈ σ.front ↔ A ∈ insert L σ.acc ∧ isExit A.node
    rw [hI.front_iff A, Set.mem_insert_iff]
    constructor
    · rintro ⟨h1, h2⟩
      exact ⟨Or.inr h1, h2⟩
    · rintro ⟨rfl | h1, h2⟩
      · exact absurd h2 hx
      · exact ⟨h1, h2⟩
  · rcases handled_accept (L := L) hI.start with h | h
    · exact Or.inl (Or.inl h)
    · exact Or.inr h
  · intro A hA hAx w hw hlt
    rcases hA with hAL | hA
    · rw [hAL] at hw hlt ⊢
      by_cases hq : Pass (accept σ L) (L.next N w)
      · exact Or.inl (Or.inr ⟨w, hw, hlt, rfl, hq⟩)
      · refine Or.inr (cov_of_not_pass hb2 hexit_att ?_ hq)
        intro B hB
        have := hN.tau_nonneg _ _ hw
        have := hLtime B hB
        simp only [Lbl.next]
        linarith
    · rcases handled_accept (L := L) (hI.next A hA hAx w hw hlt) with h | h
      · exact Or.inl (Or.inl h)
      · exact Or.inr h

/-- One iteration keeps the invariant. -/
theorem finv_step [DecidableEq V] (hN : N.Good) {σ σ' : FState V} (hI : FInv N isExit s t0 σ)
    (h : FStep N isExit σ σ') : FInv N isExit s t0 σ' := by
  cases h with
  | discard hL hp => exact finv_discard hI hL hp
  | exit hL hp hx => exact finv_exit hI hL hp hx
  | expand hL hp hx => exact finv_expand hN hI hL hp hx

/-- Every state the loop can reach from its start satisfies the invariant. -/
theorem finv_reachable [DecidableEq V] (hN : N.Good) (h0 : t0 < N.close s) {σ₀ σ : FState V}
    (hinit : FInit s t0 σ₀) (hrun : Relation.ReflTransGen (FStep N isExit) σ₀ σ) :
    FInv N isExit s t0 σ := by
  induction hrun with
  | refl => exact finv_init h0 hinit
  | tail _ hst ih => exact finv_step hN ih hst

/-- At exhaustion every safe walk is covered: some accepted label at its end, or at an exit,
arrives no later with no more dose. The induction runs along the walk, using
`dominance_preserved` at each move. -/
theorem cov_of_safe (hN : N.Good) {σ : FState V} (hI : FInv N isExit s t0 σ)
    (hdry : σ.heap = ∅) {v : V} {t D : ℝ} (h : Safe N s t0 v t D) :
    Cov isExit σ.acc ⟨v, t, D⟩ := by
  induction h with
  | start _ =>
    rcases hI.start with h | h
    · rw [hdry] at h; exact absurd h (Set.notMem_empty _)
    · exact h
  | @move u v t D _ huv hlt ih =>
    obtain ⟨B, hB, hBn, hBt, hBd⟩ := ih
    dsimp only at hBn hBt hBd
    have hτ := hN.tau_nonneg u v huv
    by_cases hx : isExit B.node
    · refine ⟨B, hB, Or.inr hx, ?_, ?_⟩
      · dsimp only; linarith
      · have := hN.dose_nonneg u v huv t
        dsimp only; linarith
    · have hBu : B.node = u := hBn.resolve_right hx
      have hE : N.E B.node v := by rw [hBu]; exact huv
      have hok : B.time + N.tau B.node v < N.close v := by rw [hBu]; linarith
      rcases hI.next B hB hx v hE hok with h | h
      · rw [hdry] at h; exact absurd h (Set.notMem_empty _)
      · obtain ⟨B', hB', hB'n, hB't, hB'd⟩ := h
        simp only [Lbl.next, hBu] at hB'n hB't hB'd
        have hmono := hN.dose_mono u v huv hBt
        exact ⟨B', hB', hB'n, by dsimp only; linarith, by dsimp only; linarith⟩

/-- LG2 (b), soundness, at every point of every run: every front point is an exit label whose
(arrival, dose) is that of a safe walk from the start. For Studio: `doseFront` never invents a
route, and the route it reports (the parent chain) walks to the reported numbers. -/
theorem front_sound {σ : FState V} (hI : FInv N isExit s t0 σ) {p : Lbl V}
    (hp : p ∈ σ.front) : isExit p.node ∧ Safe N s t0 p.node p.time p.dose :=
  let ⟨ha, hx⟩ := (hI.front_iff p).1 hp; ⟨hx, hI.acc_safe p ha⟩

/-- LG2: the front is in non-decreasing arrival and strictly falling dose, at every point of
every run (the order the code documents: "ordered by arrival (dose strictly falling)"). -/
theorem front_sorted {σ : FState V} (hI : FInv N isExit s t0 σ) :
    σ.front.Pairwise (fun p q => p.time ≤ q.time ∧ q.dose < p.dose) :=
  hI.sorted

/-- LG2 (a), completeness, at exhaustion: every safe walk from the start to an exit is matched
or beaten on both counts by a front point. -/
theorem front_complete (hN : N.Good) {σ : FState V} (hI : FInv N isExit s t0 σ)
    (hdry : σ.heap = ∅) {x : V} {t D : ℝ} (hx : isExit x) (h : Safe N s t0 x t D) :
    ∃ p ∈ σ.front, p.time ≤ t ∧ p.dose ≤ D := by
  obtain ⟨B, hB, hBn, hBt, hBd⟩ := cov_of_safe hN hI hdry h
  have hBx : isExit B.node := hBn.elim (fun h => by rw [h]; exact hx) id
  exact ⟨B, (hI.front_iff B).2 ⟨hB, hBx⟩, hBt, hBd⟩

/-- At exhaustion the front is empty exactly when no safe walk reaches an exit. -/
theorem front_empty_iff (hN : N.Good) {σ : FState V} (hI : FInv N isExit s t0 σ)
    (hdry : σ.heap = ∅) : σ.front = [] ↔ ¬ ∃ x a D, isExit x ∧ Safe N s t0 x a D := by
  constructor
  · rintro hf ⟨x, a, D, hx, h⟩
    obtain ⟨p, hp, -⟩ := front_complete hN hI hdry hx h
    rw [hf] at hp
    exact absurd hp List.not_mem_nil
  · intro hno
    by_contra hf
    obtain ⟨p, hp⟩ := List.exists_mem_of_ne_nil _ hf
    obtain ⟨hx, h⟩ := front_sound hI hp
    exact hno ⟨_, _, _, hx, h⟩

/-- LG2 (c): the first front point's arrival is the earliest safe arrival at any exit (the
plan's `fastest`). -/
theorem front_head_earliest (hN : N.Good) {σ : FState V} (hI : FInv N isExit s t0 σ)
    (hdry : σ.heap = ∅) {p : Lbl V} {rest : List (Lbl V)} (hf : σ.front = p :: rest) :
    IsLeast {a | ∃ x D, isExit x ∧ Safe N s t0 x a D} p.time := by
  have hp : p ∈ σ.front := by rw [hf]; exact List.mem_cons_self
  refine ⟨let ⟨hx, h⟩ := front_sound hI hp; ⟨_, _, hx, h⟩, ?_⟩
  rintro a ⟨x, D, hx, h⟩
  obtain ⟨q, hq, hqt, -⟩ := front_complete hN hI hdry hx h
  have hs := hI.sorted
  rw [hf, List.pairwise_cons] at hs
  rw [hf, List.mem_cons] at hq
  rcases hq with rfl | hq
  · exact hqt
  · exact (hs.1 q hq).1.trans hqt

/-- LG2 (d): the last front point's dose is the least dose of any safe walk to an exit (the
plan's `leastDose`). -/
theorem front_last_leastDose (hN : N.Good) {σ : FState V} (hI : FInv N isExit s t0 σ)
    (hdry : σ.heap = ∅) {p : Lbl V} {init : List (Lbl V)} (hf : σ.front = init ++ [p]) :
    IsLeast {D | ∃ x a, isExit x ∧ Safe N s t0 x a D} p.dose := by
  have hp : p ∈ σ.front := by rw [hf]; simp
  refine ⟨let ⟨hx, h⟩ := front_sound hI hp; ⟨_, _, hx, h⟩, ?_⟩
  rintro D ⟨x, a, hx, h⟩
  obtain ⟨q, hq, -, hqd⟩ := front_complete hN hI hdry hx h
  have hs := hI.sorted
  rw [hf, List.pairwise_append] at hs
  rw [hf, List.mem_append, List.mem_singleton] at hq
  rcases hq with hq | rfl
  · exact (hs.2.2 q hq p (List.mem_singleton_self p)).2.le.trans hqd
  · exact hqd

/-- LG2, how minimal the front is. If a safe walk to an exit arrives no later than a front
point `p` with no more dose, then either it has exactly `p`'s arrival and dose, or a *later*
front point has the *same* arrival as `p` and no more dose than the walk (so strictly less
than `p`). A front point can only be beaten by a point tied with it in arrival time, popped
after it. -/
theorem front_minimal (hN : N.Good) {σ : FState V} (hI : FInv N isExit s t0 σ)
    (hdry : σ.heap = ∅) {l1 l2 : List (Lbl V)} {p : Lbl V} (hf : σ.front = l1 ++ p :: l2)
    {x : V} {a D : ℝ} (hx : isExit x) (h : Safe N s t0 x a D) (ha : a ≤ p.time)
    (hD : D ≤ p.dose) :
    (a = p.time ∧ D = p.dose) ∨ ∃ q ∈ l2, q.time = p.time ∧ q.dose ≤ D ∧ q.dose < p.dose := by
  obtain ⟨q, hq, hqt, hqd⟩ := front_complete hN hI hdry hx h
  have hs := hI.sorted
  rw [hf, List.pairwise_append, List.pairwise_cons] at hs
  rw [hf, List.mem_append, List.mem_cons] at hq
  rcases hq with hq | rfl | hq
  · have h2 := (hs.2.2 q hq p List.mem_cons_self).2
    linarith
  · exact Or.inl ⟨le_antisymm ha hqt, le_antisymm hD hqd⟩
  · have h1 := hs.2.1.1 q hq
    exact Or.inr ⟨q, hq, le_antisymm (hqt.trans ha) h1.1, hqd, h1.2⟩

/-- LG2: when no two front points are tied in arrival time, every front point is
Pareto-optimal: no safe walk to an exit is at least as good on both counts unless it has the
same arrival and dose. -/
theorem front_pareto (hN : N.Good) {σ : FState V} (hI : FInv N isExit s t0 σ)
    (hdry : σ.heap = ∅) (hdistinct : σ.front.Pairwise (fun p q => p.time ≠ q.time))
    {p : Lbl V} (hp : p ∈ σ.front) {x : V} {a D : ℝ} (hx : isExit x) (h : Safe N s t0 x a D)
    (ha : a ≤ p.time) (hD : D ≤ p.dose) : a = p.time ∧ D = p.dose := by
  obtain ⟨l1, l2, hf⟩ := List.append_of_mem hp
  rcases front_minimal hN hI hdry hf hx h ha hD with h1 | ⟨q, hq, hqt, -⟩
  · exact h1
  · rw [hf, List.pairwise_append, List.pairwise_cons] at hdistinct
    exact absurd hqt.symm (hdistinct.2.1.1 q hq)

end Inv

/-- The code's last pass: keep a front label only if its dose is below the last label kept
("Labels tied in time can settle in either dose order; keep the strict front"). The argument
is the last kept dose (`none` before the first). -/
noncomputable def keepStrict : Option ℝ → List (Lbl V) → List (Lbl V)
  | _, [] => []
  | none, L :: rest => L :: keepStrict (some L.dose) rest
  | some d, L :: rest =>
    if L.dose < d then L :: keepStrict (some L.dose) rest else keepStrict (some d) rest

theorem keepStrict_some : ∀ (d : ℝ) (l : List (Lbl V)),
    l.Pairwise (fun p q => q.dose < p.dose) → (∀ L ∈ l, L.dose < d) → keepStrict (some d) l = l
  | _, [], _, _ => rfl
  | d, L :: rest, hl, hd => by
    rw [List.pairwise_cons] at hl
    rw [keepStrict, ite_eq_left (hd L List.mem_cons_self), keepStrict_some L.dose rest hl.2 hl.1]

/-- The filter the code had before 2026-10-28 is a no-op: the front's doses already fall
strictly (every kept exit label passed `D < bestExit`), so `kept` is the front itself and a
label tied in time with a later, lower-dose label is not removed. The code now drops the
earlier of two points tied in arrival. -/
theorem keepStrict_front {N : Net V} {s : V} {t0 : ℝ} {σ : FState V}
    (hI : FInv N isExit s t0 σ) : keepStrict none σ.front = σ.front := by
  have hs : σ.front.Pairwise (fun p q => q.dose < p.dose) := hI.sorted.imp (fun h => h.2)
  cases hf : σ.front with
  | nil => rfl
  | cons L rest =>
    rw [hf, List.pairwise_cons] at hs
    rw [keepStrict, keepStrict_some L.dose rest hs.2 hs.1]

/-- At exhaustion, the code's output is the front: complete, sound, first point the earliest
safe arrival, last point the least dose, and the final filter changes nothing. -/
theorem doseFront_correct [DecidableEq V] {N : Net V} (hN : N.Good) {s : V} {t0 : ℝ}
    (h0 : t0 < N.close s) {σ₀ σ : FState V} (hinit : FInit s t0 σ₀)
    (hrun : Relation.ReflTransGen (FStep N isExit) σ₀ σ) (hdry : σ.heap = ∅) :
    keepStrict none σ.front = σ.front ∧
    (∀ p ∈ σ.front, isExit p.node ∧ Safe N s t0 p.node p.time p.dose) ∧
    (∀ x t D, isExit x → Safe N s t0 x t D → ∃ p ∈ σ.front, p.time ≤ t ∧ p.dose ≤ D) ∧
    (∀ p rest, σ.front = p :: rest → IsLeast {a | ∃ x D, isExit x ∧ Safe N s t0 x a D} p.time) ∧
    (∀ init p, σ.front = init ++ [p] →
      IsLeast {D | ∃ x a, isExit x ∧ Safe N s t0 x a D} p.dose) := by
  have hI := finv_reachable hN h0 hinit hrun
  exact ⟨keepStrict_front hI, fun p hp => front_sound hI hp,
    fun x t D hx h => front_complete hN hI hdry hx h,
    fun p rest hf => front_head_earliest hN hI hdry hf,
    fun init p hf => front_last_leastDose hN hI hdry hf⟩

/-- The front only grows: a step appends at most one point and never removes one. -/
theorem front_append [DecidableEq V] {N : Net V} {σ σ' : FState V}
    (h : FStep N isExit σ σ') : ∃ l, σ'.front = σ.front ++ l := by
  cases h with
  | discard => exact ⟨[], (List.append_nil _).symm⟩
  | exit => exact ⟨_, rfl⟩
  | expand => exact ⟨[], (List.append_nil _).symm⟩

end Front

/-! ### A tie in arrival time leaves a beaten point on the front -/

section Tie

/-- The tie network: `0` the start, `1` a hot room, `2` a cool room, `3` the exit; links
`0-1`, `0-2`, `1-3`, `2-3`, both directions. -/
def cxAdj : Fin 4 → Fin 4 → Bool :=
  ![![false, true, true, false], ![true, false, false, true], ![true, false, false, true],
    ![false, true, true, false]]

/-- Every link takes 1 s, nothing closes before 100 s, and a move into or out of the hot room
costs dose 1 (any other move 0). -/
noncomputable def cxNet : Net (Fin 4) :=
  ⟨fun u v => cxAdj u v = true, fun _ _ => 1, fun _ => 100,
    fun u v _ => if u = 1 ∨ v = 1 then 1 else 0⟩

theorem cxNet_good : cxNet.Good where
  tau_nonneg _ _ _ := zero_le_one
  dose_nonneg u v _ t := by
    show (0 : ℝ) ≤ if u = 1 ∨ v = 1 then 1 else 0
    split_ifs <;> norm_num
  dose_mono _ _ _ := monotone_const

/-- The exit. -/
def cxExit (v : Fin 4) : Prop := v = 3

noncomputable def cxL0 : Lbl (Fin 4) := ⟨0, 0, 0⟩
noncomputable def cxL1 : Lbl (Fin 4) := ⟨1, 1, 1⟩
noncomputable def cxL2 : Lbl (Fin 4) := ⟨2, 1, 0⟩
/-- The exit label through the hot room: 2 s, dose 2. -/
noncomputable def cxL3 : Lbl (Fin 4) := ⟨3, 2, 2⟩
/-- The exit label through the cool room: 2 s, dose 0. -/
noncomputable def cxL4 : Lbl (Fin 4) := ⟨3, 2, 0⟩

/-- The run: expand the start, the hot room, the cool room; then pop the hot route's exit
label before the cool route's (they are tied at 2 s). -/
noncomputable def cx0 : FState (Fin 4) := ⟨{cxL0}, ∅, fun _ => ⊤, ⊤, []⟩
noncomputable def cx1 : FState (Fin 4) :=
  { accept cx0 cxL0 with heap := (cx0.heap \ {cxL0}) ∪ pushed cxNet (accept cx0 cxL0) cxL0 }
noncomputable def cx2 : FState (Fin 4) :=
  { accept cx1 cxL1 with heap := (cx1.heap \ {cxL1}) ∪ pushed cxNet (accept cx1 cxL1) cxL1 }
noncomputable def cx3 : FState (Fin 4) :=
  { accept cx2 cxL2 with heap := (cx2.heap \ {cxL2}) ∪ pushed cxNet (accept cx2 cxL2) cxL2 }
noncomputable def cx4 : FState (Fin 4) :=
  { accept cx3 cxL3 with bestExit := (cxL3.dose : WithTop ℝ), front := cx3.front ++ [cxL3] }
noncomputable def cx5 : FState (Fin 4) :=
  { accept cx4 cxL4 with bestExit := (cxL4.dose : WithTop ℝ), front := cx4.front ++ [cxL4] }

/-- A pushed label is one move from the popped one along a link. -/
theorem cx_pushed {σ : FState (Fin 4)} {L L' : Lbl (Fin 4)} (h : L' ∈ pushed cxNet σ L) :
    ∃ w, cxAdj L.node w = true ∧
      L' = ⟨w, L.time + 1, L.dose + if L.node = 1 ∨ w = 1 then 1 else 0⟩ ∧ Pass σ L' := by
  obtain ⟨w, hw, -, rfl, hp⟩ := h
  exact ⟨w, hw, rfl, hp⟩

theorem cx1_heap {L : Lbl (Fin 4)} (h : L ∈ cx1.heap) : L = cxL1 ∨ L = cxL2 := by
  rcases h with ⟨h1, h2⟩ | h
  · exact absurd h1 h2
  · obtain ⟨w, hw, rfl, -⟩ := cx_pushed h
    fin_cases w <;> simp [cxAdj, cxL0, cxL1, cxL2] at hw ⊢

/-- A test the label passes, from the values of `best` and `bestExit`. -/
theorem pass_of {V : Type*} {σ : FState V} {L : Lbl V} {b e : WithTop ℝ}
    (hb : σ.best L.node = b) (he : σ.bestExit = e) (h1 : (L.dose : WithTop ℝ) < b)
    (h2 : (L.dose : WithTop ℝ) < e) : Pass σ L :=
  ⟨hb ▸ h1, he ▸ h2⟩

/-- A test the label fails: `best` at its node is already no more than its dose. -/
theorem not_pass_of {V : Type*} {σ : FState V} {L : Lbl V} {x : ℝ}
    (hb : σ.best L.node = x) (hx : x ≤ L.dose) : ¬ Pass σ L := fun hp => by
  have h := hp.1
  rw [hb] at h
  exact absurd (WithTop.coe_lt_coe.1 h) (not_lt.2 hx)

theorem cx1_mem : cxL1 ∈ cx1.heap ∧ cxL2 ∈ cx1.heap := by
  refine ⟨Or.inr ⟨1, rfl, ?_, ?_, ?_⟩, Or.inr ⟨2, rfl, ?_, ?_, ?_⟩⟩
  · show (0 : ℝ) + 1 < 100; norm_num
  · simp [cxNet, cxL0, cxL1, Lbl.next]
  · exact pass_of rfl rfl (WithTop.coe_lt_top _) (WithTop.coe_lt_top _)
  · show (0 : ℝ) + 1 < 100; norm_num
  · simp [cxNet, cxL0, cxL2, Lbl.next]
  · exact pass_of rfl rfl (WithTop.coe_lt_top _) (WithTop.coe_lt_top _)

theorem cx2_heap {L : Lbl (Fin 4)} (h : L ∈ cx2.heap) : L = cxL2 ∨ L = cxL3 := by
  rcases h with ⟨h1, h2⟩ | h
  · rcases cx1_heap h1 with rfl | rfl
    · exact absurd rfl h2
    · exact Or.inl rfl
  · obtain ⟨w, hw, rfl, hp⟩ := cx_pushed h
    fin_cases w
    · exact absurd hp (not_pass_of (x := 0) rfl (by norm_num [cxL1]))
    · simp [cxAdj, cxL1] at hw
    · simp [cxAdj, cxL1] at hw
    · right; norm_num [cxL1, cxL3]

theorem cx2_mem : cxL2 ∈ cx2.heap ∧ cxL3 ∈ cx2.heap := by
  refine ⟨Or.inl ⟨cx1_mem.2, by simp [cxL1, cxL2]⟩, Or.inr ⟨3, rfl, ?_, ?_, ?_⟩⟩
  · show (1 : ℝ) + 1 < 100; norm_num
  · norm_num [cxNet, cxL1, cxL3, Lbl.next]
  · exact pass_of rfl rfl (WithTop.coe_lt_top _) (WithTop.coe_lt_top _)

theorem cx3_heap {L : Lbl (Fin 4)} (h : L ∈ cx3.heap) : L = cxL3 ∨ L = cxL4 := by
  rcases h with ⟨h1, h2⟩ | h
  · rcases cx2_heap h1 with rfl | rfl
    · exact absurd rfl h2
    · exact Or.inl rfl
  · obtain ⟨w, hw, rfl, hp⟩ := cx_pushed h
    fin_cases w
    · exact absurd hp (not_pass_of (x := 0) rfl (by norm_num [cxL2]))
    · simp [cxAdj, cxL2] at hw
    · simp [cxAdj, cxL2] at hw
    · right; norm_num [cxL2, cxL4]

theorem cx3_mem : cxL3 ∈ cx3.heap ∧ cxL4 ∈ cx3.heap := by
  refine ⟨Or.inl ⟨cx2_mem.2, by simp [cxL2, cxL3]⟩, Or.inr ⟨3, rfl, ?_, ?_, ?_⟩⟩
  · show (1 : ℝ) + 1 < 100; norm_num
  · norm_num [cxNet, cxL2, cxL4, Lbl.next]
  · exact pass_of rfl rfl (WithTop.coe_lt_top _) (WithTop.coe_lt_top _)

/-- The run, start to exhaustion, popping the hot route's exit label first. -/
theorem cx_run : Relation.ReflTransGen (FStep cxNet cxExit) cx0 cx5 := by
  have s1 : FStep cxNet cxExit cx0 cx1 := by
    refine FStep.expand ⟨rfl, fun L' h => ?_⟩
      (pass_of rfl rfl (WithTop.coe_lt_top _) (WithTop.coe_lt_top _)) (by simp [cxExit, cxL0])
    rw [Set.mem_singleton_iff.1 h]
  have s2 : FStep cxNet cxExit cx1 cx2 := by
    refine FStep.expand ⟨cx1_mem.1, fun L' h => ?_⟩
      (pass_of rfl rfl (WithTop.coe_lt_top _) (WithTop.coe_lt_top _)) (by simp [cxExit, cxL1])
    rcases cx1_heap h with rfl | rfl <;> simp [cxL1, cxL2]
  have s3 : FStep cxNet cxExit cx2 cx3 := by
    refine FStep.expand ⟨cx2_mem.1, fun L' h => ?_⟩
      (pass_of rfl rfl (WithTop.coe_lt_top _) (WithTop.coe_lt_top _)) (by simp [cxExit, cxL2])
    rcases cx2_heap h with rfl | rfl <;> norm_num [cxL2, cxL3]
  have s4 : FStep cxNet cxExit cx3 cx4 := by
    refine FStep.exit ⟨cx3_mem.1, fun L' h => ?_⟩
      (pass_of rfl rfl (WithTop.coe_lt_top _) (WithTop.coe_lt_top _)) (by simp [cxExit, cxL3])
    rcases cx3_heap h with rfl | rfl <;> simp [cxL3, cxL4]
  have s5 : FStep cxNet cxExit cx4 cx5 := by
    refine FStep.exit ⟨⟨cx3_mem.2, by simp [cxL3, cxL4]⟩, fun L' h => ?_⟩
      (pass_of (b := ((2 : ℝ) : WithTop ℝ)) (e := ((2 : ℝ) : WithTop ℝ)) rfl rfl
        (WithTop.coe_lt_coe.2 (by norm_num [cxL4])) (WithTop.coe_lt_coe.2 (by norm_num [cxL4])))
      (by simp [cxExit, cxL4])
    rcases cx3_heap h.1 with rfl | rfl <;> simp [cxL3, cxL4]
  exact ((((Relation.ReflTransGen.single s1).tail s2).tail s3).tail s4).tail s5

/-- **A tie in arrival time leaves a beaten point on the front.** On `cxNet` (two routes of
the same length to the exit, one through a hot room) a run of `doseFront`'s loop from the start
runs to exhaustion with front `[(2 s, dose 2), (2 s, dose 0)]`: the heap is keyed by time
alone, so the two exit labels tied at 2 s may pop in either order, and both pass
`D < bestExit`. The first point (the plan's `fastest`) is beaten by the second on dose at the
same arrival, and the filter the code had before 2026-10-28 kept both (`keepStrict_front`).
`front_minimal` shows this is the only way a front point can be beaten; the code now keeps
only the last point of a tie (`forecast.property.test.mjs` checks it on this network). -/
theorem tie_counterexample :
    FInit 0 0 cx0 ∧ Relation.ReflTransGen (FStep cxNet cxExit) cx0 cx5 ∧ cx5.heap = ∅ ∧
    cx5.front = [⟨3, 2, 2⟩, ⟨3, 2, 0⟩] ∧ keepStrict none cx5.front = cx5.front ∧
    Safe cxNet 0 0 3 2 2 ∧ Safe cxNet 0 0 3 2 0 := by
  have hinit : FInit 0 0 cx0 := ⟨rfl, rfl, fun _ => rfl, rfl, rfl⟩
  have hI := finv_reachable cxNet_good (by norm_num [cxNet] : (0 : ℝ) < cxNet.close 0) hinit
    cx_run
  have hf : cx5.front = [⟨3, 2, 2⟩, ⟨3, 2, 0⟩] := rfl
  refine ⟨hinit, cx_run, ?_, hf, keepStrict_front hI, ?_, ?_⟩
  · ext L
    simp only [Set.mem_empty_iff_false, iff_false]
    rintro ⟨⟨h1, h2⟩, h3⟩
    rcases cx3_heap h1 with rfl | rfl
    · exact h2 rfl
    · exact h3 rfl
  · exact (front_sound hI (p := ⟨3, 2, 2⟩) (by rw [hf]; simp)).2
  · exact (front_sound hI (p := ⟨3, 2, 0⟩) (by rw [hf]; simp)).2

end Tie

/-! ## (6) A route planned on the fire at departure walks into it -/

section Snapshot

/-- The network of the test "a route planned on the fire at departure can walk into it":
nodes `0 (0,0)`, `1 (2,0)`, `2 (4,0)`, `3 (0,3)`, `4 (4,3)`, links `0-1`, `1-2` (2 m),
`0-3`, `4-2` (3 m), `3-4` (4 m), both directions. -/
def snAdj : Fin 5 → Fin 5 → Bool :=
  ![![false, true, false, true, false], ![true, false, true, false, false],
    ![false, true, false, false, true], ![true, false, false, false, true],
    ![false, false, true, true, false]]

/-- Walking times at 1 m/s (s). -/
noncomputable def snTau : Fin 5 → Fin 5 → ℝ :=
  ![![0, 2, 0, 3, 0], ![2, 0, 2, 0, 0], ![0, 2, 0, 0, 3], ![3, 0, 0, 0, 4], ![0, 0, 3, 4, 0]]

/-- Node 1 is fine at departure but untenable from 1.5 s; the others never are, so they close
at the horizon (100 s). -/
noncomputable def snClose : Fin 5 → ℝ := ![100, 1.5, 100, 100, 100]

/-- The counterexample network (no dose: only safety matters here). -/
noncomputable def snNet : Net (Fin 5) := ⟨fun u v => snAdj u v = true, snTau, snClose, fun _ _ _ => 0⟩

/-- The departure-time fire closes nothing: every node is tenable at departure (`0`). -/
theorem sn_snapshot_open (v : Fin 5) : (0 : ℝ) < snClose v := by
  fin_cases v <;> norm_num [snClose]

/-- The static walk times of the snapshot plan: each link takes `snTau`. -/
noncomputable def snStatic (u v : Fin 5) (t : ℝ) : ℝ := t + snTau u v

/-- Shortest-path distances from node 0, a potential: no link shortens them. -/
noncomputable def snDist : Fin 5 → ℝ := ![0, 2, 4, 3, 7]

theorem sn_potential (u v : Fin 5) (h : snAdj u v = true) : snDist v ≤ snDist u + snTau u v := by
  fin_cases u <;> fin_cases v <;> simp [snAdj, snDist, snTau] at h ⊢ <;> norm_num

/-- Every walk arrives no earlier than the potential says. -/
theorem sn_lower : ∀ (w : List (Fin 5)) (u : Fin 5) (t : ℝ),
    GraphNumerics.IsWalk (fun a b => snAdj a b = true) u w →
      t + snDist (GraphNumerics.wEnd u w) ≤ GraphNumerics.wArr snStatic u t w + snDist u
  | [], u, t, _ => le_rfl
  | v :: vs, u, t, hw => by
    have h1 := sn_lower vs v (t + snTau u v) hw.2
    have h2 := sn_potential u v hw.1
    show t + snDist (GraphNumerics.wEnd v vs) ≤ GraphNumerics.wArr snStatic v (t + snTau u v) vs + snDist u
    linarith

/-- Every safe walk from node 0 leaving at 0 avoids node 1 (it would arrive at 2 s or later,
after 1.5 s) and reaches the exit (node 2) at 10 s or later. -/
theorem sn_safe_bounds {v : Fin 5} {t D : ℝ} (h : Safe snNet 0 0 v t D) :
    0 ≤ t ∧ v ≠ 1 ∧ (v = 2 → 10 ≤ t) ∧ (v = 3 → 3 ≤ t) ∧ (v = 4 → 7 ≤ t) := by
  induction h with
  | start _ => exact ⟨le_rfl, by decide, by decide, by decide, by decide⟩
  | @move u v t D _ huv hlt ih =>
    obtain ⟨h0, h1, h2, h3, h4⟩ := ih
    change snAdj u v = true at huv
    change t + snTau u v < snClose v at hlt
    show 0 ≤ t + snTau u v ∧ v ≠ 1 ∧ (v = 2 → 10 ≤ t + snTau u v) ∧
      (v = 3 → 3 ≤ t + snTau u v) ∧ (v = 4 → 7 ≤ t + snTau u v)
    fin_cases u <;> fin_cases v <;> simp [snAdj, snTau, snClose] at huv hlt h1 h2 h3 h4 ⊢ <;>
      first
      | (refine ⟨?_, ?_⟩ <;> linarith)
      | linarith

/-- The route the forecast router returns, `0 → 3 → 4 → 2`, is safe and reaches the exit at
10 s. -/
theorem sn_forecast_route : Safe snNet 0 0 2 10 0 := by
  have h0 : Safe snNet 0 0 0 0 0 := Safe.start (sn_snapshot_open 0)
  have h3 := h0.move (v := 3) (by decide : snAdj 0 3 = true)
    (by simp [snNet, snTau, snClose]; norm_num)
  have h4 := h3.move (v := 4) (by decide : snAdj 3 4 = true)
    (by simp [snNet, snTau, snClose]; norm_num)
  have h2 := h4.move (v := 2) (by decide : snAdj 4 2 = true)
    (by simp [snNet, snTau, snClose]; norm_num)
  convert h2 using 1 <;> norm_num [snNet, snTau]

/-- **Snapshot planning walks into the fire.** On this network, leaving node 0 at time 0 for
the exit 2: (i) the fire at departure closes nothing, so the snapshot plan is a shortest
route of the whole graph; `0 → 1 → 2` is one, of 4 s, and no walk is shorter; (ii) walking it
reaches node 1 at 2 s, not before it closes at 1.5 s: late by 0.5 s (the test's
`lateBy = 0.5`); (iii) yet a safe route exists, and the earliest safe arrival is 10 s
(`0 → 3 → 4 → 2`, the forecast router's route). For Studio: re-planning on the fire as it
stands is not safe; the forecast must be read at the time the evacuee arrives. -/
theorem snapshot_walks_into_fire :
    (∀ v, (0 : ℝ) < snClose v) ∧
    GraphNumerics.wArr snStatic 0 0 [1, 2] = 4 ∧
    (∀ w, GraphNumerics.IsWalk (fun a b => snAdj a b = true) 0 w →
      GraphNumerics.wEnd 0 w = 2 → 4 ≤ GraphNumerics.wArr snStatic 0 0 w) ∧
    ¬ (0 + snTau 0 1 < snClose 1) ∧ 0 + snTau 0 1 - snClose 1 = 0.5 ∧
    IsLeast {a | ∃ D, Safe snNet 0 0 2 a D} 10 := by
  refine ⟨sn_snapshot_open, ?_, ?_, ?_, ?_, ⟨⟨0, sn_forecast_route⟩, ?_⟩⟩
  · simp [GraphNumerics.wArr, snStatic, snTau]; norm_num
  · intro w hw he
    have := sn_lower w 0 0 hw
    rw [he] at this
    simpa [snDist] using this
  · simp [snTau, snClose]; norm_num
  · simp [snTau, snClose]; norm_num
  · rintro a ⟨D, h⟩
    exact (sn_safe_bounds h).2.2.1 rfl

/-- Distances from node 0 on the graph without node 1, a potential there. -/
noncomputable def snDist' : Fin 5 → ℝ := ![0, 0, 10, 3, 7]

theorem sn_potential' (u v : Fin 5) (h : snAdj u v = true) (hu : u ≠ 1) (hv : v ≠ 1) :
    snDist' v ≤ snDist' u + snTau u v := by
  fin_cases u <;> fin_cases v <;> simp [snAdj, snDist', snTau] at h hu hv ⊢ <;> norm_num

/-- A walk that never enters node 1 arrives no earlier than `snDist'` says. -/
theorem sn_lower' : ∀ (w : List (Fin 5)) (u : Fin 5) (t : ℝ),
    GraphNumerics.IsWalk (fun a b => snAdj a b = true) u w → u ≠ 1 → 1 ∉ w →
      t + snDist' (GraphNumerics.wEnd u w) ≤ GraphNumerics.wArr snStatic u t w + snDist' u
  | [], u, t, _, _, _ => le_rfl
  | v :: vs, u, t, hw, hu, h1 => by
    have hv : v ≠ 1 := fun h => h1 (by simp [h])
    have h1' : 1 ∉ vs := fun h => h1 (List.mem_cons_of_mem v h)
    have h2 := sn_lower' vs v (t + snTau u v) hw.2 hv h1'
    have h3 := sn_potential' u v hw.1 hu hv
    show t + snDist' (GraphNumerics.wEnd v vs) ≤
      GraphNumerics.wArr snStatic v (t + snTau u v) vs + snDist' u
    linarith

/-- **Whatever route the snapshot search returns, it walks into the fire.** Every walk from
node 0 to the exit shorter than 10 s (so every shortest route of the snapshot graph, 4 s,
whatever the search's tie-breaking) passes through node 1, and the part of it up to node 1
arrives there after node 1 has closed. -/
theorem snapshot_plan_unsafe (w : List (Fin 5))
    (hw : GraphNumerics.IsWalk (fun a b => snAdj a b = true) 0 w)
    (he : GraphNumerics.wEnd 0 w = 2) (hshort : GraphNumerics.wArr snStatic 0 0 w < 10) :
    ∃ p q, w = p ++ 1 :: q ∧ snClose 1 < GraphNumerics.wArr snStatic 0 0 (p ++ [1]) := by
  have h1 : (1 : Fin 5) ∈ w := by
    by_contra h
    have h' := sn_lower' w 0 0 hw (by decide) h
    rw [he] at h'
    simp [snDist'] at h'
    linarith
  obtain ⟨p, q, rfl⟩ := List.append_of_mem h1
  refine ⟨p, q, rfl, ?_⟩
  have hsplit := (GraphNumerics.IsWalk_append _ 0 p (1 :: q)).1 hw
  have hp : GraphNumerics.IsWalk (fun a b => snAdj a b = true) 0 (p ++ [1]) :=
    (GraphNumerics.IsWalk_append _ 0 p [1]).2 ⟨hsplit.1, hsplit.2.1, trivial⟩
  have h2 := sn_lower (p ++ [1]) 0 0 hp
  rw [GraphNumerics.wEnd_append] at h2
  have h3 : GraphNumerics.wEnd (GraphNumerics.wEnd 0 p) [1] = 1 := rfl
  rw [h3] at h2
  simp [snDist, snClose] at h2 ⊢
  linarith

end Snapshot

end FireEgress.Forecast

#print axioms FireEgress.Forecast.runMax_monotone
#print axioms FireEgress.Forecast.le_runMax
#print axioms FireEgress.Forecast.runMax_le
#print axioms FireEgress.Forecast.runMax_of_monotone
#print axioms FireEgress.Forecast.le_runMax_iff
#print axioms FireEgress.Forecast.min_le_windowEnd
#print axioms FireEgress.Forecast.windowEnd_monotone
#print axioms FireEgress.Forecast.coarse_conservative
#print axioms FireEgress.Forecast.fine_le_coarse
#print axioms FireEgress.Forecast.fineRate_monotone
#print axioms FireEgress.Forecast.coarseRate_monotone
#print axioms FireEgress.Forecast.coarse_dose_ge
#print axioms FireEgress.Forecast.closeStep_le
#print axioms FireEgress.Forecast.closed_iff
#print axioms FireEgress.Forecast.closed_at_iff
#print axioms FireEgress.Forecast.closeStep_runMax
#print axioms FireEgress.Forecast.doseRate_eq_tConv
#print axioms FireEgress.Forecast.doseRate_nonneg
#print axioms FireEgress.Forecast.doseRate_monotone
#print axioms FireEgress.Forecast.doseRate_strictMonoOn
#print axioms FireEgress.Forecast.stepRate_monotone
#print axioms FireEgress.Forecast.max_monotone
#print axioms FireEgress.Forecast.window_monotone
#print axioms FireEgress.Forecast.window_nonneg
#print axioms FireEgress.Forecast.linkDose_monotone
#print axioms FireEgress.Forecast.linkDose_nonneg
#print axioms FireEgress.Forecast.forecast_rate_monotone
#print axioms FireEgress.Forecast.forecast_linkDose
#print axioms FireEgress.Forecast.Safe.start_open
#print axioms FireEgress.Forecast.Safe.lt_close
#print axioms FireEgress.Forecast.arr_fifo
#print axioms FireEgress.Forecast.arr_ge
#print axioms FireEgress.Forecast.safe_iff_arr
#print axioms FireEgress.Forecast.earliestArrival_exact
#print axioms FireEgress.Forecast.earliestArrival_none
#print axioms FireEgress.Forecast.earliestArrival_settled
#print axioms FireEgress.Forecast.earliestArrival_sweep
#print axioms FireEgress.Forecast.dominance_preserved
#print axioms FireEgress.Forecast.waiting_never_helps
#print axioms FireEgress.Forecast.safeW_of_safe
#print axioms FireEgress.Forecast.dp_no_holdover
#print axioms FireEgress.Forecast.cov_mono
#print axioms FireEgress.Forecast.cov_self
#print axioms FireEgress.Forecast.cov_of_not_pass
#print axioms FireEgress.Forecast.handled_accept
#print axioms FireEgress.Forecast.handled_discard
#print axioms FireEgress.Forecast.best_accept
#print axioms FireEgress.Forecast.finv_init
#print axioms FireEgress.Forecast.finv_discard
#print axioms FireEgress.Forecast.finv_exit
#print axioms FireEgress.Forecast.finv_expand
#print axioms FireEgress.Forecast.finv_step
#print axioms FireEgress.Forecast.finv_reachable
#print axioms FireEgress.Forecast.cov_of_safe
#print axioms FireEgress.Forecast.front_sound
#print axioms FireEgress.Forecast.front_sorted
#print axioms FireEgress.Forecast.front_complete
#print axioms FireEgress.Forecast.front_empty_iff
#print axioms FireEgress.Forecast.front_head_earliest
#print axioms FireEgress.Forecast.front_last_leastDose
#print axioms FireEgress.Forecast.front_minimal
#print axioms FireEgress.Forecast.front_pareto
#print axioms FireEgress.Forecast.keepStrict_some
#print axioms FireEgress.Forecast.keepStrict_front
#print axioms FireEgress.Forecast.doseFront_correct
#print axioms FireEgress.Forecast.front_append
#print axioms FireEgress.Forecast.cxNet_good
#print axioms FireEgress.Forecast.cx_pushed
#print axioms FireEgress.Forecast.cx1_heap
#print axioms FireEgress.Forecast.pass_of
#print axioms FireEgress.Forecast.not_pass_of
#print axioms FireEgress.Forecast.cx1_mem
#print axioms FireEgress.Forecast.cx2_heap
#print axioms FireEgress.Forecast.cx2_mem
#print axioms FireEgress.Forecast.cx3_heap
#print axioms FireEgress.Forecast.cx3_mem
#print axioms FireEgress.Forecast.cx_run
#print axioms FireEgress.Forecast.tie_counterexample
#print axioms FireEgress.Forecast.sn_snapshot_open
#print axioms FireEgress.Forecast.sn_potential
#print axioms FireEgress.Forecast.sn_lower
#print axioms FireEgress.Forecast.sn_safe_bounds
#print axioms FireEgress.Forecast.sn_forecast_route
#print axioms FireEgress.Forecast.snapshot_walks_into_fire
#print axioms FireEgress.Forecast.sn_potential'
#print axioms FireEgress.Forecast.sn_lower'
#print axioms FireEgress.Forecast.snapshot_plan_unsafe
