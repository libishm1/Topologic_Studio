import Mathlib
import FireEgress.GraphNumerics
import FireEgress.TimeDependent
import FireEgress.MinDose

/-!
# Escape deadlines (the latest safe departure from every node)

Studio's escape-deadline map (`src/core/deadline.js`, with the forecast of
`src/core/forecast.js`) answers, for every node of the walkable graph, "how long can an
occupant standing here wait before setting off, and still reach an exit?". This file is
the specification it must meet, and proves that the code's four computations meet it.

## Model

* The graph is a relation `E v w` ("an open link from `v` to `w`", blocked links absent;
  `route.js` builds both directions of every edge). Walking `v → w` takes `τ v w ≥ 0`
  seconds (`len / speed`, constant, so links are FIFO). Exits are `X v = true`.
* Node `v` closes at `c v : EReal` (seconds; `⊤` = never). In `escapeDeadlines` and
  `departureSafe` it is the untenability time `u`; with a forecast it is
  `forecast.closes = min(u, H)`. A walk is **safe** from `(v, t)` if it reaches every
  node, the start included, strictly before that node closes (`SafeWalk`). Walks do not
  wait (under FIFO waiting never helps: `waiting_never_helps_safe`).
* A walk is recorded as its vertex list after the start, as in `GraphNumerics` (`wEnd`,
  `wArr`, `IsWalk`); it need not stop at the first exit.
* Times are real; deadlines live in `EReal` (`⊥` = no safe departure at all, `⊤` = never
  cut off), exactly the code's `-Infinity` / `Infinity`.
* Doses: `d v w t` is the dose of walking `v → w` leaving at `t` (Studio: the integral of
  the hotter end's FED rate over the link's time interval, `hotDose`); a walk's dose is
  the sum (`walkDose`). The dose limit `θ : EReal` (`⊤` = no limit).
* Arithmetic is exact (ℝ); nothing is said about IEEE rounding.

## Results

1. **Hard deadlines** (`escapeDeadlines`, `departureSafe`). The safe departures from `v`
   (`safeSet`) are downward closed (`safeSet_downward`; also as L10a,
   `safeSet_downward_of_fifo`) and open (`safeSet_open`); their supremum `deadline`
   (`L`) satisfies `t ∈ safeSet v ↔ t < L v` (`mem_safeSet_iff`, strict, as
   `departureSafe` tests), is the best margin over routes (`deadline_eq_sSup_margin`),
   satisfies the Bellman equation (`deadline_exit`, `deadline_bellman`,
   `deadline_bellman_finset`) and is its least supersolution (`deadline_le_of_super`;
   the equation alone does not determine it, `bellman_not_unique`), and is `⊥` exactly
   when no exit is reachable at all (`deadline_eq_bot_iff`). Suffixes of safe walks are
   safe (`safeWalk_append`, L11's shape) and waiting never helps
   (`waiting_never_helps_safe`, L10b).
   The max-first label-setting loop of `escapeDeadlines`, modelled as a run relation
   (`DInit`, `DStep`, `DInv`), is the dual of Dijkstra in the `(max, min)` order: every
   popped label is exact (`dselectable_exact`), at "heap empty" every label equals `L`
   (`escapeDeadlines_correct`), and on a finite graph some run gets there
   (`escapeDeadlines_total`). `departureSafe` is time-dependent Dijkstra on gate links
   (`gateArc`, FIFO), so `TimeDependent.td_dijkstra_glb` makes it return true exactly
   when `t < L` (`departureSafe_correct`, `departureSafe_early_true`).
2. **Dose deadlines** (`doseDeadline`). With link doses non-decreasing in the departure
   time and non-negative: the least dose is non-decreasing in the departure time
   (`leastDose_mono`); the good set is downward closed (`goodSet_downward`); its supremum
   `doseDeadline` is at most `L` (`doseDeadline_le_deadline`) and equals it with no limit
   (`doseDeadline_top`); the least dose is attained by a simple walk
   (`exists_simple_le`, `exists_least_walk`, `good_iff_leastDose`), so "some safe route has
   dose `≤ θ`" (the
   code's test) is "least dose finite and `≤ θ`". Bisection keeps "good is good, bad is
   bad" and halves the gap (`bisect_*`), so the returned `good` is a good departure within
   `tolerance` of `T*` (`bisect_correct`, `doseDeadline_bisection`).
3. **The map is conservative** (`doseDeadlineMap`). The code's backward pass `gridV` is
   MinDose's backward induction without holdover on the time-expanded grid
   (`gridV_eq_val`, `gridV_dp_exact`), it is non-decreasing in the grid step
   (`gridV_mono`), and whenever it is finite and `≤ θ` the real departure at that grid
   time is good (`grid_conservative`, `map_le_doseDeadline`), provided link times are
   rounded **up** (`ceil_steps_cover`). The gap `T* - map` is **not** bounded here.
   The code's `ceil(x - 1e-9)` can round down (`epsilon_rounds_down`), which breaks the
   hypothesis; Studio's doses meet the others (`hotDose_*`, `studio_map_le_exact`).
4. **The dose limit can bind** (`example_dose_limit_binds`): `T* = 4.5 < 9 = L`.

## Not formalised (stated so a reviewer can compare with the code)

* The heaps are abstracted, as in `AStar.lean` and `TimeDependent.lean`: a step pops a
  vertex of largest (`escapeDeadlines`) or least (`departureSafe`) label; stale heap entries
  and the `done` flags are not modelled. Parallel links are merged (max of the candidates,
  i.e. the shortest link time).
* `doseFront` is taken as exact (its own specification is `Forecast.lean`'s subject):
  `doseDeadline`'s `works(t)` is assumed to be membership in `goodSet`. Its `maxLabels`
  truncation is not modelled; `doseDeadline` ignores the `truncated` flag.
* `forecast.linkDose` is taken to be `hotDose`, the integral of the hotter end's rate for
  some non-decreasing, non-negative rate. That the code's piecewise-constant sum (and the
  single-frame shortcut in `doseDeadlineMap`) equals this integral, and that the forecast's
  rates never fall (running maximum, coarsening, `Float32`), is not proved here.
* The choice of the grid step (`typical / q`, `maxWork`) and `J = floor(H / δ)` are free
  parameters; any `δ > 0` and `J` are covered.
* The gap `T* - map deadline` is not bounded. No uniqueness of Bellman solutions is proved.
* All arithmetic is exact (`ℝ`, `EReal`); IEEE rounding (`t + len / speed` against
  `closes`, `j * δ`) is not covered.
-/

namespace FireEgress.Deadline

open Finset FireEgress.GraphNumerics

/-! ## Real numbers are dense in `EReal` -/

/-- Shifting by a real link time: arriving `t + a` before `m` is leaving `t` before
`m - a`, in `EReal` (where `⊤ - a = ⊤` and `⊥ - a = ⊥`). -/
theorem coe_add_lt_iff (t a : ℝ) (m : EReal) :
    ((t + a : ℝ) : EReal) < m ↔ (t : EReal) < m - (a : EReal) := by
  induction m using EReal.rec with
  | bot => simp [EReal.bot_sub]
  | coe m =>
    rw [← EReal.coe_sub, EReal.coe_lt_coe_iff, EReal.coe_lt_coe_iff]
    constructor <;> intro h <;> linarith
  | top =>
    rw [EReal.top_sub_coe]
    exact ⟨fun _ => EReal.coe_lt_top t, fun _ => EReal.coe_lt_top _⟩

/-- Two extended reals are equal if the same reals lie below them. -/
theorem eq_of_coe_lt_iff {a b : EReal} (h : ∀ t : ℝ, (t : EReal) < a ↔ (t : EReal) < b) :
    a = b := by
  rcases lt_trichotomy a b with hab | hab | hab
  · obtain ⟨t, h1, h2⟩ := EReal.lt_iff_exists_real_btwn.1 hab
    exact absurd ((h t).2 h2) (not_lt.2 h1.le)
  · exact hab
  · obtain ⟨t, h1, h2⟩ := EReal.lt_iff_exists_real_btwn.1 hab
    exact absurd ((h t).1 h2) (not_lt.2 h1.le)

/-- `a ≤ b` in `EReal` if every real below `a` is below `b`. -/
theorem le_of_coe_lt_imp {a b : EReal} (h : ∀ t : ℝ, (t : EReal) < a → (t : EReal) < b) :
    a ≤ b := by
  by_contra hab
  push Not at hab
  obtain ⟨t, h1, h2⟩ := EReal.lt_iff_exists_real_btwn.1 hab
  exact absurd (h t h2) (not_lt.2 h1.le)

/-- Subtracting a real keeps a value above `⊥`. -/
theorem bot_lt_sub_coe {m : EReal} (h : ⊥ < m) (a : ℝ) : ⊥ < m - (a : EReal) := by
  obtain ⟨s, -, hs⟩ := EReal.lt_iff_exists_real_btwn.1 h
  have : ((s - a : ℝ) : EReal) < m - (a : EReal) :=
    (coe_add_lt_iff (s - a) a m).1 (by simpa using hs)
  exact lt_of_le_of_lt bot_le this

/-! ## (1) Hard deadlines -/

section Hard

variable {V : Type*} (E : V → V → Prop) (τ : V → V → ℝ) (c : V → EReal) (X : V → Bool)

/-- `SafeWalk E τ c t v ws`: leaving `v` at time `t` and walking the vertices `ws` in turn
along links of `E` (link `v → w` takes `τ v w` seconds, no waiting), the walker reaches
every vertex, the start included, strictly before it closes (`c`). This is the safety
test of `escapeDeadlines`, `departureSafe`, `earliestArrival` and `doseFront`
(`t < untenable[w]`, `t2 < closes[w]`). -/
def SafeWalk : ℝ → V → List V → Prop
  | t, v, [] => (t : EReal) < c v
  | t, v, w :: ws => (t : EReal) < c v ∧ E v w ∧ SafeWalk (t + τ v w) w ws

/-- The safe departure times from `v`: leaving at `t`, some safe walk ends at an exit. -/
def safeSet (v : V) : Set ℝ :=
  {t | ∃ ws, SafeWalk E τ c t v ws ∧ X (wEnd v ws) = true}

/-- The escape deadline `L(v)`: the supremum of the safe departure times from `v`
(`⊥` if there is none, `⊤` if every departure is safe). `escapeDeadlines` returns it. -/
noncomputable def deadline (v : V) : EReal :=
  sSup ((fun t : ℝ => (t : EReal)) '' safeSet E τ c X v)

/-- The margin of a walk: the latest departure from `v` at which it is still safe,
`min` over its vertices of closing time minus the time it takes to get there. -/
noncomputable def margin : V → List V → EReal
  | v, [] => c v
  | v, w :: ws => min (c v) (margin w ws - (τ v w : EReal))

/-- A safe walk starts before its start closes. -/
theorem safeWalk_start {t : ℝ} {v : V} {ws : List V} (h : SafeWalk E τ c t v ws) :
    (t : EReal) < c v := by
  cases ws with
  | nil => exact h
  | cons w ws => exact h.1

/-- FIFO with constant link times: a walk that is safe leaving at `t'` is safe leaving at
any `t ≤ t'` (it reaches every vertex `t' - t` earlier). For Studio: leaving earlier is
never worse. -/
theorem safeWalk_anti : ∀ (ws : List V) (v : V) (t t' : ℝ), t ≤ t' →
    SafeWalk E τ c t' v ws → SafeWalk E τ c t v ws
  | [], _, _, _, htt, h => lt_of_le_of_lt (EReal.coe_le_coe_iff.2 htt) h
  | w :: ws, v, _, _, htt, ⟨h1, h2, h3⟩ =>
    ⟨lt_of_le_of_lt (EReal.coe_le_coe_iff.2 htt) h1, h2,
      safeWalk_anti ws w _ _ (by linarith) h3⟩

/-- A walk is safe from `t` iff it is a walk and `t` is below its margin. -/
theorem safeWalk_iff_margin : ∀ (ws : List V) (v : V) (t : ℝ),
    SafeWalk E τ c t v ws ↔ IsWalk E v ws ∧ (t : EReal) < margin τ c v ws
  | [], v, t => by simp [SafeWalk, IsWalk, margin]
  | w :: ws, v, t => by
    simp only [SafeWalk, IsWalk, margin, lt_min_iff, safeWalk_iff_margin ws w,
      ← coe_add_lt_iff]
    tauto

/-- `t` is a safe departure iff some walk to an exit has margin above `t`. -/
theorem mem_safeSet_iff_margin (v : V) (t : ℝ) :
    t ∈ safeSet E τ c X v ↔
      ∃ ws, IsWalk E v ws ∧ X (wEnd v ws) = true ∧ (t : EReal) < margin τ c v ws := by
  simp only [safeSet, Set.mem_ofPred_eq, safeWalk_iff_margin]
  constructor
  · rintro ⟨ws, ⟨h1, h2⟩, h3⟩
    exact ⟨ws, h1, h3, h2⟩
  · rintro ⟨ws, h1, h3, h2⟩
    exact ⟨ws, ⟨h1, h2⟩, h3⟩

/-- **Leaving earlier is never worse**: the safe departures form a downward closed set.
(The code's comment "the good departures form an interval from 0".) -/
theorem safeSet_downward {v : V} {t t' : ℝ} (htt : t ≤ t') (h : t' ∈ safeSet E τ c X v) :
    t ∈ safeSet E τ c X v := by
  obtain ⟨ws, hs, hx⟩ := h
  exact ⟨ws, safeWalk_anti E τ c ws v t t' htt hs, hx⟩

/-- The safe departures form an open set: every safe departure can be delayed a little.
All the tests are strict (`t < c`), finitely many per walk. -/
theorem safeSet_open {v : V} {t : ℝ} (h : t ∈ safeSet E τ c X v) :
    ∃ t', t < t' ∧ t' ∈ safeSet E τ c X v := by
  obtain ⟨ws, hw, hx, hm⟩ := (mem_safeSet_iff_margin E τ c X v t).1 h
  obtain ⟨t', h1, h2⟩ := EReal.lt_iff_exists_real_btwn.1 hm
  exact ⟨t', EReal.coe_lt_coe_iff.1 h1, (mem_safeSet_iff_margin E τ c X v t').2 ⟨ws, hw, hx, h2⟩⟩

/-- **Departing at `t` is safe iff `t < L(v)`**, strictly, as `departureSafe` and the
code's tests use it: leaving just before the deadline works, leaving at it does not. -/
theorem mem_safeSet_iff (v : V) (t : ℝ) :
    t ∈ safeSet E τ c X v ↔ (t : EReal) < deadline E τ c X v := by
  constructor
  · intro h
    obtain ⟨t', htt, h'⟩ := safeSet_open E τ c X h
    exact lt_of_lt_of_le (EReal.coe_lt_coe_iff.2 htt) (le_sSup ⟨t', h', rfl⟩)
  · intro h
    obtain ⟨_, ⟨t', h', rfl⟩, hlt⟩ := lt_sSup_iff.1 h
    exact safeSet_downward E τ c X (EReal.coe_lt_coe_iff.1 hlt).le h'

/-- The deadline is the best margin over all walks to an exit. -/
theorem deadline_eq_sSup_margin (v : V) :
    deadline E τ c X v =
      sSup {m | ∃ ws, IsWalk E v ws ∧ X (wEnd v ws) = true ∧ m = margin τ c v ws} := by
  apply eq_of_coe_lt_iff
  intro t
  rw [← mem_safeSet_iff, mem_safeSet_iff_margin, lt_sSup_iff]
  constructor
  · rintro ⟨ws, h1, h2, h3⟩
    exact ⟨_, ⟨ws, h1, h2, rfl⟩, h3⟩
  · rintro ⟨_, ⟨ws, h1, h2, rfl⟩, h3⟩
    exact ⟨ws, h1, h2, h3⟩

/-- Bellman equation at an exit: `L(exit) = u(exit)`. -/
theorem deadline_exit {v : V} (hx : X v = true) : deadline E τ c X v = c v := by
  apply eq_of_coe_lt_iff
  intro t
  rw [← mem_safeSet_iff]
  constructor
  · rintro ⟨ws, hs, -⟩
    exact safeWalk_start E τ c hs
  · intro h
    exact ⟨[], h, hx⟩

/-- One step of the recursion: from a non-exit `v`, leaving at `t` is safe iff `v` is
still open at `t` and some link `v → w` reaches `w` at a safe departure time from `w`.
(The suffix of a safe walk is safe from where and when it arrives: L11's shape.) -/
theorem mem_safeSet_cons {v : V} (hx : X v = false) (t : ℝ) :
    t ∈ safeSet E τ c X v ↔
      (t : EReal) < c v ∧ ∃ w, E v w ∧ t + τ v w ∈ safeSet E τ c X w := by
  constructor
  · rintro ⟨ws, hs, he⟩
    cases ws with
    | nil =>
      change X v = true at he
      rw [hx] at he
      exact absurd he Bool.false_ne_true
    | cons w ws => exact ⟨hs.1, w, hs.2.1, ws, hs.2.2, he⟩
  · rintro ⟨hc, w, hvw, ws, hs, he⟩
    exact ⟨w :: ws, ⟨hc, hvw, hs⟩, he⟩

/-- **Bellman equation** (the header of `deadline.js`): at a non-exit node,
`L(v) = max over open links (v, w) of min(u(v), L(w) - τ(v, w))`, and `⊥` (`-Infinity`)
when `v` has no open link. -/
theorem deadline_bellman {v : V} (hx : X v = false) :
    deadline E τ c X v = ⨆ (w : V) (_ : E v w), min (c v) (deadline E τ c X w - (τ v w : EReal)) := by
  apply eq_of_coe_lt_iff
  intro t
  rw [← mem_safeSet_iff, mem_safeSet_cons E τ c X hx]
  simp only [lt_iSup_iff, lt_min_iff, ← coe_add_lt_iff, ← mem_safeSet_iff]
  constructor
  · rintro ⟨h1, w, h2, h3⟩
    exact ⟨w, h2, h1, h3⟩
  · rintro ⟨w, h2, h1, h3⟩
    exact ⟨h1, w, h2, h3⟩

/-- The Bellman equation on a finite graph, as a finite maximum (`Finset.sup`, with
`⊥ = -Infinity` for a node without open links), as `escapeDeadlines` evaluates it. -/
theorem deadline_bellman_finset [Fintype V] [DecidableRel E] {v : V} (hx : X v = false) :
    deadline E τ c X v =
      (univ.filter (E v)).sup fun w => min (c v) (deadline E τ c X w - (τ v w : EReal)) := by
  apply eq_of_coe_lt_iff
  intro t
  rw [← mem_safeSet_iff, mem_safeSet_cons E τ c X hx, Finset.lt_sup_iff]
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, lt_min_iff, ← coe_add_lt_iff,
    ← mem_safeSet_iff]
  constructor
  · rintro ⟨h1, w, h2, h3⟩
    exact ⟨w, h2, h1, h3⟩
  · rintro ⟨w, h2, h1, h3⟩
    exact ⟨h1, w, h2, h3⟩

/-- The Bellman inequality along every link (exits included): `min(u(v), L(w) - τ) ≤ L(v)`.
This is what makes a relaxed label a lower bound on the deadline. -/
theorem deadline_link {v w : V} (hvw : E v w) :
    min (c v) (deadline E τ c X w - (τ v w : EReal)) ≤ deadline E τ c X v := by
  apply le_of_coe_lt_imp
  intro t ht
  rw [lt_min_iff, ← coe_add_lt_iff, ← mem_safeSet_iff] at ht
  obtain ⟨hc, ws, hs, hx⟩ := ht
  rw [← mem_safeSet_iff]
  exact ⟨w :: ws, ⟨hc, hvw, hs⟩, hx⟩

/-- **`L` is the least supersolution of the Bellman equation.** Any labelling `d` with
`d ≥ u` at the exits and `d(v) ≥ min(u(v), d(w) - τ)` on every link out of a non-exit is
`≥ L`. The Bellman equation alone does not determine `L` (on a cycle with no exit and
`u = ⊤`, the labelling `⊤` also solves it); this is why the label-setting loop builds `L`
up from `⊥`, the dual of `TimeDependent.bellman_subsolution_le`. -/
theorem deadline_le_of_super (d : V → EReal) (hexit : ∀ v, X v = true → c v ≤ d v)
    (hlink : ∀ v w, X v = false → E v w → min (c v) (d w - (τ v w : EReal)) ≤ d v) (v : V) :
    deadline E τ c X v ≤ d v := by
  have key : ∀ (ws : List V) (v : V) (t : ℝ), SafeWalk E τ c t v ws →
      X (wEnd v ws) = true → (t : EReal) < d v := by
    intro ws
    induction ws with
    | nil => intro v t hs hx; exact lt_of_lt_of_le hs (hexit v hx)
    | cons w ws ih =>
      intro v t hs hx
      cases hv : X v with
      | true => exact lt_of_lt_of_le hs.1 (hexit v hv)
      | false =>
        have h2 : ((t + τ v w : ℝ) : EReal) < d w := ih w _ hs.2.2 hx
        rw [coe_add_lt_iff] at h2
        exact lt_of_lt_of_le (lt_min hs.1 h2) (hlink v w hv hs.2.1)
  apply le_of_coe_lt_imp
  intro t ht
  obtain ⟨ws, hs, hx⟩ := (mem_safeSet_iff E τ c X v t).2 ht
  exact key ws v t hs hx

/-- With every closing time above `⊥` (Studio's are times `≥ 0` or `Infinity`), each walk
has a margin above `⊥`. -/
theorem margin_bot_lt (hc : ∀ v, ⊥ < c v) : ∀ (ws : List V) (v : V), ⊥ < margin τ c v ws
  | [], v => hc v
  | w :: ws, v => lt_min (hc v) (bot_lt_sub_coe (margin_bot_lt hc ws w) (τ v w))

/-- **`L(v) = -Infinity` exactly when no exit is reachable at all** (through open links,
whatever the fire), as the comment of `escapeDeadlines` says, provided no node is closed
from `-∞`. -/
theorem deadline_eq_bot_iff (hc : ∀ v, ⊥ < c v) (v : V) :
    deadline E τ c X v = ⊥ ↔ ¬ ∃ ws, IsWalk E v ws ∧ X (wEnd v ws) = true := by
  constructor
  · rintro hL ⟨ws, hw, hx⟩
    obtain ⟨t, -, ht⟩ := EReal.lt_iff_exists_real_btwn.1 (margin_bot_lt τ c hc ws v)
    have := (mem_safeSet_iff E τ c X v t).1 ((mem_safeSet_iff_margin E τ c X v t).2 ⟨ws, hw, hx, ht⟩)
    rw [hL] at this
    exact not_lt_bot this
  · intro h
    refine le_antisymm (sSup_le ?_) bot_le
    rintro _ ⟨t, ⟨ws, hs, hx⟩, rfl⟩
    exact absurd ⟨ws, ((safeWalk_iff_margin E τ c ws v t).1 hs).1, hx⟩ h

/-- The walking time of a walk. -/
def walkTime : V → List V → ℝ
  | _, [] => 0
  | v, w :: ws => τ v w + walkTime w ws

theorem walkTime_append : ∀ (p q : List V) (v : V),
    walkTime τ v (p ++ q) = walkTime τ v p + walkTime τ (wEnd v p) q
  | [], q, v => by simp [walkTime, wEnd]
  | w :: p, q, v => by
    simp only [List.cons_append, walkTime, wEnd, walkTime_append p q w]
    ring

theorem walkTime_nonneg (hτ : ∀ v w, E v w → 0 ≤ τ v w) :
    ∀ (ws : List V) (v : V), IsWalk E v ws → 0 ≤ walkTime τ v ws
  | [], _, _ => le_rfl
  | w :: ws, v, hw => add_nonneg (hτ v w hw.1) (walkTime_nonneg hτ ws w hw.2)

/-- Splitting a walk: `p ++ q` is safe from `t` iff `p` is, and `q` is safe from where and
when `p` ends. In particular a suffix of a safe walk is safe from where and when the walker
gets there (L11's shape, `GraphNumerics.L11_suffix_ea`; no FIFO needed): replanning on an
unchanged forecast never turns a safe plan unsafe. -/
theorem safeWalk_append : ∀ (p q : List V) (v : V) (t : ℝ),
    SafeWalk E τ c t v (p ++ q) ↔
      SafeWalk E τ c t v p ∧ SafeWalk E τ c (t + walkTime τ v p) (wEnd v p) q
  | [], q, v, t => by
    simp only [List.nil_append, walkTime, add_zero, wEnd]
    exact ⟨fun h => ⟨safeWalk_start E τ c h, h⟩, fun h => h.2⟩
  | w :: p, q, v, t => by
    simp only [List.cons_append, SafeWalk, walkTime, wEnd, safeWalk_append p q w, add_assoc]
    tauto

/-! ### The forward check: `departureSafe` is time-dependent Dijkstra on gate links -/

/-- A gate link (`departureSafe`, `earliestArrival`): entering at `t`, arrive at `t + a` if
that is strictly before the head closes (`b : EReal`, `⊤` = never), otherwise never (`⊤`).
`GraphNumerics.closeArc` with an extended-real closing time. -/
noncomputable def gateArc (a : ℝ) (b : EReal) (x : WithTop ℝ) : WithTop ℝ :=
  WithTop.recTopCoe ⊤ (fun t => if ((t + a : ℝ) : EReal) < b then ((t + a : ℝ) : WithTop ℝ) else ⊤) x

theorem gateArc_top (a : ℝ) (b : EReal) : gateArc a b ⊤ = ⊤ := rfl

theorem gateArc_coe (a : ℝ) (b : EReal) (t : ℝ) :
    gateArc a b t = if ((t + a : ℝ) : EReal) < b then ((t + a : ℝ) : WithTop ℝ) else ⊤ := rfl

/-- A gate link is FIFO (R2 for `gateArc`). -/
theorem gateArc_mono (a : ℝ) (b : EReal) : Monotone (gateArc a b) := by
  intro x y hxy
  induction y using WithTop.recTopCoe with
  | top => rw [gateArc_top]; exact le_top
  | coe t' =>
    induction x using WithTop.recTopCoe with
    | top => simp at hxy
    | coe t =>
      have htt : t ≤ t' := WithTop.coe_le_coe.mp hxy
      rw [gateArc_coe, gateArc_coe]
      split_ifs with h1 h2 h2
      · exact WithTop.coe_le_coe.mpr (by linarith)
      · exact le_top
      · exact absurd (lt_of_le_of_lt (EReal.coe_le_coe_iff.2 (by linarith)) h2) h1
      · exact le_top

/-- A gate link with `a ≥ 0` never arrives before it is entered. -/
theorem gateArc_ge (a : ℝ) (ha : 0 ≤ a) (b : EReal) (x : WithTop ℝ) : x ≤ gateArc a b x := by
  induction x using WithTop.recTopCoe with
  | top => exact le_of_eq rfl
  | coe t =>
    rw [gateArc_coe]
    split_ifs
    · exact WithTop.coe_le_coe.mpr (by linarith)
    · exact le_top

/-- Once closed, a walk on gate links stays closed. -/
theorem wArr_gate_top : ∀ (ws : List V) (v : V),
    wArr (fun v w => gateArc (τ v w) (c w)) v ⊤ ws = ⊤
  | [], _ => rfl
  | w :: ws, _ => wArr_gate_top ws w

/-- A safe walk is a walk whose start is open and whose arrival on gate links is finite:
the safety test of this file is the arrival function of `departureSafe`. -/
theorem safeWalk_iff_wArr : ∀ (ws : List V) (v : V) (t : ℝ),
    SafeWalk E τ c t v ws ↔ (t : EReal) < c v ∧ IsWalk E v ws ∧
      wArr (fun v w => gateArc (τ v w) (c w)) v (t : WithTop ℝ) ws ≠ ⊤
  | [], v, t => by simp [SafeWalk, IsWalk, wArr]
  | w :: ws, v, t => by
    simp only [SafeWalk, IsWalk, wArr]
    rw [gateArc_coe]
    split_ifs with h
    · rw [safeWalk_iff_wArr ws w]
      tauto
    · constructor
      · rintro ⟨-, -, hs⟩
        exact absurd (safeWalk_start E τ c hs) h
      · rintro ⟨-, -, hne⟩
        exact absurd (wArr_gate_top τ c ws w) hne

/-- L10a gives the downward closure too: on gate links (FIFO, `gateArc_mono`) a later
departure never arrives earlier (`TimeDependent.walk_fifo`), so a walk that is still open
when left at `t'` is open when left at `t ≤ t'`. -/
theorem safeSet_downward_of_fifo {v : V} {t t' : ℝ} (htt : t ≤ t')
    (h : t' ∈ safeSet E τ c X v) : t ∈ safeSet E τ c X v := by
  obtain ⟨ws, hs, hx⟩ := h
  obtain ⟨hc, hw, hne⟩ := (safeWalk_iff_wArr E τ c ws v t').1 hs
  refine ⟨ws, (safeWalk_iff_wArr E τ c ws v t).2 ⟨lt_of_le_of_lt (EReal.coe_le_coe_iff.2 htt) hc,
    hw, ne_top_of_le_ne_top hne ?_⟩, hx⟩
  exact TimeDependent.walk_fifo E _ (fun a b _ => gateArc_mono (τ a b) (c b)) v ws hw
    (WithTop.coe_le_coe.2 htt)

/-- Safe departures in the language of `TimeDependent`: leaving `v` at `t` is safe iff `v`
is open at `t` and some exit has a finite arrival on gate links. -/
theorem mem_safeSet_iff_arr (v : V) (t : ℝ) :
    t ∈ safeSet E τ c X v ↔ (t : EReal) < c v ∧ ∃ x, X x = true ∧
      ∃ T : ℝ, TimeDependent.Arr E (fun v w => gateArc (τ v w) (c w)) v (t : WithTop ℝ) x T := by
  constructor
  · rintro ⟨ws, hs, hx⟩
    obtain ⟨hc, hw, hne⟩ := (safeWalk_iff_wArr E τ c ws v t).1 hs
    obtain ⟨T, hT⟩ := WithTop.ne_top_iff_exists.1 hne
    exact ⟨hc, _, hx, T, (TimeDependent.arr_iff_walk E _ v _ _ _).2 ⟨ws, hw, rfl, hT.symm⟩⟩
  · rintro ⟨hc, x, hx, T, hT⟩
    obtain ⟨ws, hw, he, ha⟩ := (TimeDependent.arr_iff_walk E _ v _ _ _).1 hT
    refine ⟨ws, (safeWalk_iff_wArr E τ c ws v t).2 ⟨hc, hw, ?_⟩, he ▸ hx⟩
    rw [ha]
    exact WithTop.coe_ne_top

/-- L10b for safety: a walk that may wait at nodes (`TimeDependent.ArrW`, which does not
even check closures while waiting) reaches no exit that a walk without waiting does not
reach, at a finite time no later. So `departureSafe`, `doseFront` and the deadlines lose
nothing by not waiting. Reuses `TimeDependent.waiting_never_helps`. -/
theorem waiting_never_helps_safe (v x : V) (t T : ℝ)
    (h : TimeDependent.ArrW E (fun v w => gateArc (τ v w) (c w)) v (t : WithTop ℝ) x T) :
    ∃ T' : ℝ, T' ≤ T ∧ TimeDependent.Arr E (fun v w => gateArc (τ v w) (c w)) v (t : WithTop ℝ) x T' := by
  obtain ⟨τ', h', hle⟩ := TimeDependent.waiting_never_helps E _
    (fun a b _ => gateArc_mono (τ a b) (c b)) v _ h
  have hne : τ' ≠ ⊤ := ne_top_of_le_ne_top WithTop.coe_ne_top hle
  obtain ⟨T', rfl⟩ := WithTop.ne_top_iff_exists.1 hne
  exact ⟨T', WithTop.coe_le_coe.1 hle, h'⟩

/-- **`departureSafe` is exact.** Run time-dependent Dijkstra (`TimeDependent.Step`) on the
gate links from `v` at time `t` until the queue is dry. The code returns true iff `v` is
open at `t` and some exit gets a finite label: by `TimeDependent.td_dijkstra_glb` that is
iff `t < L(v)`. -/
theorem departureSafe_correct [DecidableEq V] (hτ : ∀ v w, E v w → 0 ≤ τ v w) (v : V) (t : ℝ)
    (σ₀ σ : TimeDependent.State V (WithTop ℝ)) (h0 : TimeDependent.Init v (t : WithTop ℝ) σ₀)
    (hrun : Relation.ReflTransGen (TimeDependent.Step E (fun v w => gateArc (τ v w) (c w))) σ₀ σ)
    (hdry : σ.R ⊆ σ.S) :
    ((t : EReal) < c v ∧ ∃ x, X x = true ∧ TimeDependent.outLabel σ x ≠ ⊤) ↔
      (t : EReal) < deadline E τ c X v := by
  rw [← mem_safeSet_iff, mem_safeSet_iff_arr]
  have hglb := fun x => TimeDependent.td_dijkstra_glb E (fun v w => gateArc (τ v w) (c w))
    (fun a b _ => gateArc_mono (τ a b) (c b)) (fun a b hab => gateArc_ge (τ a b) (hτ a b hab) (c b))
    v _ σ₀ σ h0 hrun hdry x
  have key : ∀ x, TimeDependent.outLabel σ x ≠ ⊤ ↔
      ∃ T : ℝ, TimeDependent.Arr E (fun v w => gateArc (τ v w) (c w)) v (t : WithTop ℝ) x T := by
    intro x
    constructor
    · intro hne
      by_contra hno
      push Not at hno
      apply hne
      refine top_le_iff.1 ((hglb x).2 fun a ha => ?_)
      by_contra hat
      obtain ⟨T, rfl⟩ := WithTop.ne_top_iff_exists.1 (fun h => hat (h ▸ le_rfl))
      exact hno T ha
    · rintro ⟨T, hT⟩
      exact ne_top_of_le_ne_top WithTop.coe_ne_top ((hglb x).1 hT)
  simp only [key]

/-- The early return of `departureSafe` is sound: if the vertex popped next is an exit
with a finite label, and the start was open, leaving at `t` is safe
(`TimeDependent.td_dijkstra_correct`: the popped label is a real arrival). -/
theorem departureSafe_early_true [DecidableEq V] (hτ : ∀ v w, E v w → 0 ≤ τ v w) (v : V)
    (t : ℝ) (σ₀ σ : TimeDependent.State V (WithTop ℝ))
    (h0 : TimeDependent.Init v (t : WithTop ℝ) σ₀)
    (hrun : Relation.ReflTransGen (TimeDependent.Step E (fun v w => gateArc (τ v w) (c w))) σ₀ σ)
    (x : V) (hsel : TimeDependent.Selectable σ x) (hx : X x = true) (hfin : σ.d x ≠ ⊤)
    (hc : (t : EReal) < c v) : (t : EReal) < deadline E τ c X v := by
  have h := TimeDependent.td_dijkstra_correct E (fun v w => gateArc (τ v w) (c w))
    (fun a b _ => gateArc_mono (τ a b) (c b)) (fun a b hab => gateArc_ge (τ a b) (hτ a b hab) (c b))
    v _ σ₀ σ h0 hrun x hsel
  obtain ⟨T, hT⟩ := WithTop.ne_top_iff_exists.1 hfin
  rw [← mem_safeSet_iff, mem_safeSet_iff_arr]
  exact ⟨hc, x, hx, T, hT ▸ h.1⟩

/-! ### The label-setting loop of `escapeDeadlines` -/

/-- The loop's state: settled vertices `S` (`done`) and labels `d` (`L`). -/
structure DState (V : Type*) where
  S : Finset V
  d : V → EReal

/-- The start: nothing settled, every exit labelled with its closing time, every other
vertex `⊥` (`-Infinity`). -/
def DInit (σ : DState V) : Prop :=
  σ.S = ∅ ∧ ∀ v, σ.d v = if X v = true then c v else ⊥

/-- `w` may be popped next: unsettled, labelled above `⊥` (the heap holds only such
labels), and of **largest** label among the unsettled vertices (ties any way). -/
def DSelectable (σ : DState V) (w : V) : Prop :=
  w ∉ σ.S ∧ ⊥ < σ.d w ∧ ∀ v ∉ σ.S, σ.d v ≤ σ.d w

/-- One iteration: pop a selectable `w`, settle it, and for every link `v → w` into it from
an unsettled `v` set `L[v] ← max(L[v], min(u(v), L[w] - τ(v, w)))`; nothing else
changes. -/
def DStep [DecidableEq V] (σ σ' : DState V) : Prop :=
  ∃ w, DSelectable σ w ∧ σ'.S = insert w σ.S ∧
    (∀ v, E v w → v ∉ σ'.S → σ'.d v = max (σ.d v) (min (c v) (σ.d w - (τ v w : EReal)))) ∧
    (∀ v, ¬ (E v w ∧ v ∉ σ'.S) → σ'.d v = σ.d v)

/-- The heap is empty: every unsettled vertex is still at `⊥`. -/
def DDone (σ : DState V) : Prop := ∀ v ∉ σ.S, σ.d v = ⊥

/-- The invariant (Dijkstra's, in the `(max, min)` order): every label is a lower bound on
the deadline, settled labels are exact, exits carry at least their closing time, and every
link into a settled vertex from an unsettled one has been relaxed. -/
structure DInv (σ : DState V) : Prop where
  le : ∀ v, σ.d v ≤ deadline E τ c X v
  opt : ∀ v ∈ σ.S, deadline E τ c X v ≤ σ.d v
  exits : ∀ x, X x = true → c x ≤ σ.d x
  relaxed : ∀ w ∈ σ.S, ∀ v, E v w → v ∉ σ.S → min (c v) (σ.d w - (τ v w : EReal)) ≤ σ.d v

/-- The loop starts in a state satisfying the invariant. -/
theorem dinit_inv (σ : DState V) (h : DInit c X σ) : DInv E τ c X σ := by
  obtain ⟨hS, hd⟩ := h
  refine ⟨fun v => ?_, by simp [hS], fun x hx => by rw [hd x, ite_eq_left hx], by simp [hS]⟩
  rw [hd v]
  split_ifs with hx
  · rw [deadline_exit E τ c X hx]
  · exact bot_le

/-- **The key lemma** (dual of `TimeDependent.selectable_le`). If every unsettled label is
at most `b`, every unsettled deadline is at most `b`. A safe walk from an unsettled vertex
either reaches an exit while still unsettled (its label is `≥ u > ` the time) or enters the
settled set through a relaxed link `v → x`: it reaches `x` before `L(x) = d(x)`, so it
left `v` before `min(u(v), d(x) - τ) ≤ d(v) ≤ b`; earlier links only add time (`τ ≥ 0`). -/
theorem unsettled_le (hτ : ∀ v w, E v w → 0 ≤ τ v w) {σ : DState V} (hI : DInv E τ c X σ)
    (b : EReal) (hb : ∀ v ∉ σ.S, σ.d v ≤ b) (v : V) (hv : v ∉ σ.S) :
    deadline E τ c X v ≤ b := by
  have key : ∀ (ws : List V) (v : V) (t : ℝ), v ∉ σ.S → SafeWalk E τ c t v ws →
      X (wEnd v ws) = true → (t : EReal) < b := by
    intro ws
    induction ws with
    | nil => intro v t hv hs hx; exact lt_of_lt_of_le hs ((hI.exits v hx).trans (hb v hv))
    | cons x ws ih =>
      intro v t hv hs hx
      by_cases hxS : x ∈ σ.S
      · have h1 : t + τ v x ∈ safeSet E τ c X x := ⟨ws, hs.2.2, hx⟩
        rw [mem_safeSet_iff] at h1
        have h2 : ((t + τ v x : ℝ) : EReal) < σ.d x := lt_of_lt_of_le h1 (hI.opt x hxS)
        rw [coe_add_lt_iff] at h2
        exact lt_of_lt_of_le (lt_min hs.1 h2) ((hI.relaxed x hxS v hs.2.1 hv).trans (hb v hv))
      · have h2 := ih x (t + τ v x) hxS hs.2.2 hx
        exact lt_of_le_of_lt (EReal.coe_le_coe_iff.2 (by linarith [hτ v x hs.2.1])) h2
  apply le_of_coe_lt_imp
  intro t ht
  obtain ⟨ws, hs, hx⟩ := (mem_safeSet_iff E τ c X v t).2 ht
  exact key ws v t hv hs hx

/-- **Every popped label is exact**: the vertex of largest label among the unsettled
carries its deadline. -/
theorem dselectable_exact (hτ : ∀ v w, E v w → 0 ≤ τ v w) {σ : DState V}
    (hI : DInv E τ c X σ) {w : V} (hw : DSelectable σ w) : σ.d w = deadline E τ c X w :=
  le_antisymm (hI.le w) (unsettled_le E τ c X hτ hI (σ.d w) hw.2.2 w hw.1)

/-- One step preserves the invariant. -/
theorem dstep_inv [DecidableEq V] (hτ : ∀ v w, E v w → 0 ≤ τ v w) {σ σ' : DState V}
    (hI : DInv E τ c X σ) (hst : DStep E τ c σ σ') : DInv E τ c X σ' := by
  obtain ⟨w, hw, hS, hrel, hkeep⟩ := hst
  have hwL := dselectable_exact E τ c X hτ hI hw
  have hge : ∀ v, σ.d v ≤ σ'.d v := by
    intro v
    by_cases h : E v w ∧ v ∉ σ'.S
    · rw [hrel v h.1 h.2]
      exact le_max_left _ _
    · rw [hkeep v h]
  have hfix : ∀ v ∈ σ'.S, σ'.d v = σ.d v := fun v hv => hkeep v (fun h => h.2 hv)
  refine ⟨fun v => ?_, fun v hv => ?_, fun x hx => (hI.exits x hx).trans (hge x), ?_⟩
  · by_cases h : E v w ∧ v ∉ σ'.S
    · rw [hrel v h.1 h.2]
      refine max_le (hI.le v) ?_
      rw [hwL]
      exact deadline_link E τ c X h.1
    · rw [hkeep v h]
      exact hI.le v
  · rw [hfix v hv]
    rw [hS, Finset.mem_insert] at hv
    rcases hv with rfl | hv
    · exact hwL.ge
    · exact hI.opt v hv
  · intro x hx v hvx hv
    have hxS' := hx
    rw [hS, Finset.mem_insert] at hx
    rw [hfix x hxS']
    rcases hx with rfl | hx
    · rw [hrel v hvx hv]
      exact le_max_right _ _
    · have hvS : v ∉ σ.S := fun h => hv (by rw [hS]; exact Finset.mem_insert_of_mem h)
      exact (hI.relaxed x hx v hvx hvS).trans (hge v)

/-- Every state reachable from the start satisfies the invariant. -/
theorem dreachable_inv [DecidableEq V] (hτ : ∀ v w, E v w → 0 ≤ τ v w) (σ₀ σ : DState V)
    (h0 : DInit c X σ₀) (hrun : Relation.ReflTransGen (DStep E τ c) σ₀ σ) : DInv E τ c X σ := by
  induction hrun with
  | refl => exact dinit_inv E τ c X σ₀ h0
  | tail _ hst ih => exact dstep_inv E τ c X hτ ih hst

/-- **`escapeDeadlines` is exact.** At any point of any run, a popped label is the
deadline; when the heap is empty (every unsettled label still `⊥`), every label is the
deadline: settled ones by the invariant, unsettled ones because their deadline is `⊥`
(no safe departure at all). -/
theorem escapeDeadlines_correct [DecidableEq V] (hτ : ∀ v w, E v w → 0 ≤ τ v w)
    (σ₀ σ : DState V) (h0 : DInit c X σ₀) (hrun : Relation.ReflTransGen (DStep E τ c) σ₀ σ)
    (hdone : DDone σ) (v : V) : σ.d v = deadline E τ c X v := by
  have hI := dreachable_inv E τ c X hτ σ₀ σ h0 hrun
  by_cases hv : v ∈ σ.S
  · exact le_antisymm (hI.le v) (hI.opt v hv)
  · have h1 := unsettled_le E τ c X hτ hI ⊥ (fun v hv => (hdone v hv).le) v hv
    rw [hdone v hv]
    exact (le_bot_iff.1 h1).symm

open Classical in
/-- One concrete iteration (the code's relaxation loop). -/
noncomputable def drelax [DecidableEq V] (σ : DState V) (w : V) : DState V where
  S := insert w σ.S
  d := fun v => if E v w ∧ v ∉ insert w σ.S then max (σ.d v) (min (c v) (σ.d w - (τ v w : EReal)))
    else σ.d v

/-- The concrete relaxation is a `DStep`. -/
theorem drelax_step [DecidableEq V] (σ : DState V) (w : V) (hw : DSelectable σ w) :
    DStep E τ c σ (drelax E τ c σ w) :=
  ⟨w, hw, rfl, fun v hvw hv => by simp only [drelax]; rw [ite_eq_left ⟨hvw, hv⟩],
    fun v h => by simp only [drelax] at h ⊢; rw [ite_eq_right h]⟩

/-- On a finite graph the loop always reaches "heap empty". -/
theorem dexists_done [DecidableEq V] [Fintype V] (n : ℕ) :
    ∀ σ : DState V, Fintype.card V - σ.S.card ≤ n →
      ∃ σ', Relation.ReflTransGen (DStep E τ c) σ σ' ∧ DDone σ' := by
  induction n with
  | zero =>
    intro σ hσ
    refine ⟨σ, Relation.ReflTransGen.refl, fun v hv => ?_⟩
    have hle := Finset.card_le_univ σ.S
    have huniv : σ.S = Finset.univ := Finset.eq_univ_of_card _ (by omega)
    exact absurd (huniv ▸ Finset.mem_univ v) hv
  | succ n ih =>
    intro σ hσ
    by_cases hdone : DDone σ
    · exact ⟨σ, Relation.ReflTransGen.refl, hdone⟩
    · unfold DDone at hdone
      push Not at hdone
      obtain ⟨v, hv, hvb⟩ := hdone
      obtain ⟨w, hwm, hmax⟩ := Finset.exists_max_image (univ.filter (· ∉ σ.S)) σ.d
        ⟨v, by simp [hv]⟩
      have hwS : w ∉ σ.S := (Finset.mem_filter.1 hwm).2
      have hsel : DSelectable σ w :=
        ⟨hwS, lt_of_lt_of_le (bot_lt_iff_ne_bot.2 hvb) (hmax v (by simp [hv])),
          fun u hu => hmax u (by simp [hu])⟩
      have hcard : (drelax E τ c σ w).S.card = σ.S.card + 1 := Finset.card_insert_of_notMem hwS
      have hle := Finset.card_le_univ (drelax E τ c σ w).S
      obtain ⟨σ', hrun, hd⟩ := ih (drelax E τ c σ w) (by omega)
      exact ⟨σ', Relation.ReflTransGen.head (drelax_step E τ c σ w hsel) hrun, hd⟩

/-- **Total correctness of `escapeDeadlines`** on a finite graph with non-negative link
times: from the start some run empties the heap, and its labels are the deadlines `L`
at every vertex (`-Infinity` where no exit is reachable). -/
theorem escapeDeadlines_total [DecidableEq V] [Fintype V] (hτ : ∀ v w, E v w → 0 ≤ τ v w)
    (σ₀ : DState V) (h0 : DInit c X σ₀) :
    ∃ σ, Relation.ReflTransGen (DStep E τ c) σ₀ σ ∧ DDone σ ∧ ∀ v, σ.d v = deadline E τ c X v := by
  obtain ⟨σ, hrun, hdone⟩ := dexists_done E τ c _ σ₀ le_rfl
  exact ⟨σ, hrun, hdone, escapeDeadlines_correct E τ c X hτ σ₀ σ h0 hrun hdone⟩

end Hard

/-! ## (2) Dose deadlines -/

section Dose

variable {V : Type*} (E : V → V → Prop) (τ : V → V → ℝ) (c : V → EReal) (X : V → Bool)
  (d : V → V → ℝ → ℝ)

/-- The dose of a walk left at `t`: the sum of its link doses, each read at the time the
walker enters the link (`doseFront`'s `D + linkDose(v, w, t, t2)`). -/
def walkDose : ℝ → V → List V → ℝ
  | _, _, [] => 0
  | t, v, w :: ws => d v w t + walkDose (t + τ v w) w ws

/-- The good departures from `v` under the dose limit `θ` (`⊤` = no limit): some safe walk
reaches an exit with dose at most `θ`. This is `doseDeadline`'s test `works(t)` (front
non-empty and its least dose `≤ limit`), with `doseFront` taken as exact. -/
def goodSet (θ : EReal) (v : V) : Set ℝ :=
  {t | ∃ ws, SafeWalk E τ c t v ws ∧ X (wEnd v ws) = true ∧
    ((walkDose τ d t v ws : ℝ) : EReal) ≤ θ}

/-- The dose deadline `T*(v)`: the supremum of the good departures (`⊥` if none). -/
noncomputable def doseDeadline (θ : EReal) (v : V) : EReal :=
  sSup ((fun t : ℝ => (t : EReal)) '' goodSet E τ c X d θ v)

/-- The least dose `D*(v, t)` of leaving `v` at `t` over safe walks to an exit (`⊤` if
there is none): the last point of `doseFront`'s front. -/
noncomputable def leastDose (v : V) (t : ℝ) : EReal :=
  sInf {x | ∃ ws, SafeWalk E τ c t v ws ∧ X (wEnd v ws) = true ∧
    x = ((walkDose τ d t v ws : ℝ) : EReal)}

/-- With link doses non-decreasing in the departure time, the same walk left earlier
accrues no more dose. -/
theorem walkDose_mono (hd : ∀ v w, E v w → Monotone (d v w)) :
    ∀ (ws : List V) (v : V) (t t' : ℝ), IsWalk E v ws → t ≤ t' →
      walkDose τ d t v ws ≤ walkDose τ d t' v ws
  | [], _, _, _, _, _ => le_rfl
  | w :: ws, v, t, t', hw, htt =>
    add_le_add (hd v w hw.1 htt) (walkDose_mono hd ws w _ _ hw.2 (by linarith))

theorem walkDose_nonneg (hd0 : ∀ v w, E v w → ∀ t, 0 ≤ d v w t) :
    ∀ (ws : List V) (v : V) (t : ℝ), IsWalk E v ws → 0 ≤ walkDose τ d t v ws
  | [], _, _, _ => le_rfl
  | w :: ws, v, t, hw => add_nonneg (hd0 v w hw.1 t) (walkDose_nonneg hd0 ws w _ hw.2)

theorem walkDose_append : ∀ (p q : List V) (v : V) (t : ℝ),
    walkDose τ d t v (p ++ q) = walkDose τ d t v p + walkDose τ d (t + walkTime τ v p) (wEnd v p) q
  | [], q, v, t => by simp [walkDose, walkTime, wEnd]
  | w :: p, q, v, t => by
    simp only [List.cons_append, walkDose, walkTime, wEnd, walkDose_append p q w, add_assoc]

/-- **Leaving earlier is never worse, in dose**: `D*(v, t)` is non-decreasing in `t`, when
every link dose is non-decreasing in its departure time (heat that never falls). The same
walk left earlier is still safe (`safeWalk_anti`) and no more costly (`walkDose_mono`). -/
theorem leastDose_mono (hd : ∀ v w, E v w → Monotone (d v w)) (v : V) :
    Monotone (leastDose E τ c X d v) := by
  intro t t' htt
  apply le_sInf
  rintro _ ⟨ws, hs, hx, rfl⟩
  have h1 : leastDose E τ c X d v t ≤ ((walkDose τ d t v ws : ℝ) : EReal) :=
    sInf_le ⟨ws, safeWalk_anti E τ c ws v t t' htt hs, hx, rfl⟩
  refine h1.trans ?_
  exact EReal.coe_le_coe_iff.2
    (walkDose_mono E τ d hd ws v t t' ((safeWalk_iff_margin E τ c ws v t').1 hs).1 htt)

/-- The good departures are downward closed, so they form an interval reaching back from
`T*` (the code's "the good departures form an interval from 0"). -/
theorem goodSet_downward (hd : ∀ v w, E v w → Monotone (d v w)) {θ : EReal} {v : V}
    {t t' : ℝ} (htt : t ≤ t') (h : t' ∈ goodSet E τ c X d θ v) :
    t ∈ goodSet E τ c X d θ v := by
  obtain ⟨ws, hs, hx, hθ⟩ := h
  exact ⟨ws, safeWalk_anti E τ c ws v t t' htt hs, hx, (EReal.coe_le_coe_iff.2
    (walkDose_mono E τ d hd ws v t t' ((safeWalk_iff_margin E τ c ws v t').1 hs).1 htt)).trans hθ⟩

/-- A good departure is a safe one. -/
theorem goodSet_subset (θ : EReal) (v : V) : goodSet E τ c X d θ v ⊆ safeSet E τ c X v :=
  fun _ ⟨ws, hs, hx, _⟩ => ⟨ws, hs, hx⟩

/-- **`T*(v) ≤ L(v)`**: a dose limit can only bring the deadline forward. -/
theorem doseDeadline_le_deadline (θ : EReal) (v : V) :
    doseDeadline E τ c X d θ v ≤ deadline E τ c X v :=
  sSup_le_sSup (Set.image_mono (goodSet_subset E τ c X d θ v))

/-- With no limit every safe departure is good. -/
theorem goodSet_top (v : V) : goodSet E τ c X d ⊤ v = safeSet E τ c X v := by
  ext t
  exact ⟨fun ⟨ws, hs, hx, _⟩ => ⟨ws, hs, hx⟩, fun ⟨ws, hs, hx⟩ => ⟨ws, hs, hx, le_top⟩⟩

/-- **No dose limit is the hard deadline**: `T*(v) = L(v)` when `θ = ⊤` (`limit: Infinity`;
then `doseDeadline`'s bracket must contain `L`). -/
theorem doseDeadline_top (v : V) : doseDeadline E τ c X d ⊤ v = deadline E τ c X v := by
  unfold doseDeadline deadline
  rw [goodSet_top]

/-- A good departure is at most `T*`. -/
theorem le_doseDeadline {θ : EReal} {v : V} {t : ℝ} (h : t ∈ goodSet E τ c X d θ v) :
    (t : EReal) ≤ doseDeadline E τ c X d θ v :=
  le_sSup ⟨t, h, rfl⟩

/-- A departure strictly before `T*` is good (the good set is downward closed). Whether
`T*` itself is good is not decided: a dose limit met with equality makes it good. -/
theorem mem_goodSet_of_lt (hd : ∀ v w, E v w → Monotone (d v w)) {θ : EReal} {v : V}
    {t : ℝ} (h : (t : EReal) < doseDeadline E τ c X d θ v) : t ∈ goodSet E τ c X d θ v := by
  obtain ⟨_, ⟨t', h', rfl⟩, hlt⟩ := lt_sSup_iff.1 h
  exact goodSet_downward E τ c X d hd (EReal.coe_lt_coe_iff.1 hlt).le h'

/-- A bad departure is an upper bound on `T*`. -/
theorem doseDeadline_le_of_not_mem (hd : ∀ v w, E v w → Monotone (d v w)) {θ : EReal}
    {v : V} {b : ℝ} (hb : b ∉ goodSet E τ c X d θ v) :
    doseDeadline E τ c X d θ v ≤ (b : EReal) := by
  apply sSup_le
  rintro _ ⟨t, ht, rfl⟩
  by_contra h
  push Not at h
  exact hb (goodSet_downward E τ c X d hd (EReal.coe_lt_coe_iff.1 h).le ht)

/-- A good departure has least dose at most the limit. -/
theorem leastDose_le_of_good {θ : EReal} {v : V} {t : ℝ} (h : t ∈ goodSet E τ c X d θ v) :
    leastDose E τ c X d v t ≤ θ := by
  obtain ⟨ws, hs, hx, hθ⟩ := h
  have h1 : leastDose E τ c X d v t ≤ ((walkDose τ d t v ws : ℝ) : EReal) :=
    sInf_le ⟨ws, hs, hx, rfl⟩
  exact h1.trans hθ

/-- Cutting a cycle out of a safe walk keeps it safe (the rest is walked earlier), ends it
at the same vertex, and does not raise its dose (the cycle's dose is `≥ 0`, the rest is
entered earlier). -/
theorem cut_cycle (hτ : ∀ v w, E v w → 0 ≤ τ v w) (hd : ∀ v w, E v w → Monotone (d v w))
    (hd0 : ∀ v w, E v w → ∀ t, 0 ≤ d v w t) (v : V) (t : ℝ) (p cy q : List V)
    (hs : SafeWalk E τ c t v (p ++ cy ++ q)) (hc : wEnd (wEnd v p) cy = wEnd v p) :
    SafeWalk E τ c t v (p ++ q) ∧ wEnd v (p ++ q) = wEnd v (p ++ cy ++ q) ∧
      walkDose τ d t v (p ++ q) ≤ walkDose τ d t v (p ++ cy ++ q) := by
  rw [safeWalk_append, safeWalk_append] at hs
  obtain ⟨⟨hp, hcy⟩, hq⟩ := hs
  rw [walkTime_append, wEnd_append, hc] at hq
  have hcyw : IsWalk E (wEnd v p) cy := ((safeWalk_iff_margin E τ c cy _ _).1 hcy).1
  have hT := walkTime_nonneg E τ hτ cy _ hcyw
  have hqw : IsWalk E (wEnd v p) q := ((safeWalk_iff_margin E τ c q _ _).1 hq).1
  refine ⟨(safeWalk_append E τ c p q v t).2 ⟨hp, safeWalk_anti E τ c q _ _ _ (by linarith) hq⟩,
    by simp only [wEnd_append, hc], ?_⟩
  rw [walkDose_append, walkDose_append, walkDose_append, walkTime_append, wEnd_append, hc]
  have h1 := walkDose_nonneg E τ d hd0 cy (wEnd v p) (t + walkTime τ v p) hcyw
  have h2 := walkDose_mono E τ d hd q (wEnd v p) (t + walkTime τ v p)
    (t + (walkTime τ v p + walkTime τ (wEnd v p) cy)) hqw (by linarith)
  linarith

/-- Simple walks suffice for the dose too: every safe walk is matched by a safe walk to the
same vertex that visits no vertex twice, with no more dose. -/
theorem exists_simple_le (hτ : ∀ v w, E v w → 0 ≤ τ v w) (hd : ∀ v w, E v w → Monotone (d v w))
    (hd0 : ∀ v w, E v w → ∀ t, 0 ≤ d v w t) (v : V) (t : ℝ) (n : ℕ) :
    ∀ ws : List V, ws.length ≤ n → SafeWalk E τ c t v ws →
      ∃ ws', SafeWalk E τ c t v ws' ∧ wEnd v ws' = wEnd v ws ∧ (v :: ws').Nodup ∧
        walkDose τ d t v ws' ≤ walkDose τ d t v ws := by
  induction n with
  | zero =>
    intro ws hl hs
    have : ws = [] := List.eq_nil_of_length_eq_zero (by omega)
    subst this
    exact ⟨[], hs, rfl, List.nodup_singleton v, le_rfl⟩
  | succ n ih =>
    intro ws hl hs
    by_cases hnd : (v :: ws).Nodup
    · exact ⟨ws, hs, rfl, hnd, le_rfl⟩
    · obtain ⟨p, cy, q, rfl, hcy, hcyc⟩ := TimeDependent.exists_cycle v ws hnd
      obtain ⟨hs', he, hle⟩ := cut_cycle E τ c d hτ hd hd0 v t p cy q hs hcyc
      have hlen : (p ++ q).length ≤ n := by
        have hc0 : cy.length ≠ 0 := fun h0 => hcy (List.eq_nil_of_length_eq_zero h0)
        simp only [List.length_append] at hl ⊢
        omega
      obtain ⟨w', h1, h2, h3, h4⟩ := ih (p ++ q) hlen hs'
      exact ⟨w', h1, h2.trans he, h3, h4.trans hle⟩

/-- On a finite graph the least dose is attained: if leaving `v` at `t` is safe at all, some
safe walk to an exit has the least dose. -/
theorem exists_least_walk [Fintype V] (hτ : ∀ v w, E v w → 0 ≤ τ v w)
    (hd : ∀ v w, E v w → Monotone (d v w)) (hd0 : ∀ v w, E v w → ∀ t, 0 ≤ d v w t) (v : V)
    (t : ℝ) (hex : t ∈ safeSet E τ c X v) :
    ∃ ws, SafeWalk E τ c t v ws ∧ X (wEnd v ws) = true ∧
      ∀ ws', SafeWalk E τ c t v ws' → X (wEnd v ws') = true →
        walkDose τ d t v ws ≤ walkDose τ d t v ws' := by
  have hfin : Set.Finite
      {ws : List V | SafeWalk E τ c t v ws ∧ X (wEnd v ws) = true ∧ (v :: ws).Nodup} :=
    (List.finite_length_le V (Fintype.card V)).subset (fun ws hws => by
      have := hws.2.2.length_le_card
      simp only [List.length_cons] at this
      show ws.length ≤ Fintype.card V
      omega)
  obtain ⟨ws0, hs0, hx0⟩ := hex
  obtain ⟨w1, hs1, he1, hnd1, -⟩ := exists_simple_le E τ c d hτ hd hd0 v t _ ws0 le_rfl hs0
  obtain ⟨ws, ⟨hs, hx, -⟩, hmin⟩ := Set.exists_min_image _ (walkDose τ d t v) hfin
    ⟨w1, hs1, by rw [he1]; exact hx0, hnd1⟩
  refine ⟨ws, hs, hx, fun ws' hs' hx' => ?_⟩
  obtain ⟨w'', hs'', he'', hnd'', hle⟩ := exists_simple_le E τ c d hτ hd hd0 v t _ ws' le_rfl hs'
  exact (hmin w'' ⟨hs'', by rw [he'']; exact hx', hnd''⟩).trans hle

/-- **The code's test is the least-dose test.** On a finite graph, "some safe route has dose
`≤ θ`" (`works(t)`: front non-empty and last dose `≤ limit`) holds iff the least dose
`D*(v, t)` is finite and `≤ θ` ("finite first", as `doseDeadlineMap` says: with `θ = ⊤`, "no
route" must not count as within the limit). With `leastDose_mono`, the good set
`{t : D*(v, t) < ⊤ ∧ D*(v, t) ≤ θ}` is downward closed and `T*` is its supremum. -/
theorem good_iff_leastDose [Fintype V] (hτ : ∀ v w, E v w → 0 ≤ τ v w)
    (hd : ∀ v w, E v w → Monotone (d v w)) (hd0 : ∀ v w, E v w → ∀ t, 0 ≤ d v w t)
    (θ : EReal) (v : V) (t : ℝ) :
    t ∈ goodSet E τ c X d θ v ↔ leastDose E τ c X d v t < ⊤ ∧ leastDose E τ c X d v t ≤ θ := by
  constructor
  · rintro ⟨ws, hs, hx, hθ⟩
    have h : leastDose E τ c X d v t ≤ ((walkDose τ d t v ws : ℝ) : EReal) :=
      sInf_le ⟨ws, hs, hx, rfl⟩
    exact ⟨lt_of_le_of_lt h (EReal.coe_lt_top _), h.trans hθ⟩
  · rintro ⟨hlt, hθ⟩
    have hex : t ∈ safeSet E τ c X v := by
      by_contra hno
      apply hlt.ne
      unfold leastDose
      apply sInf_eq_top.2
      rintro _ ⟨ws, hs, hx, rfl⟩
      exact absurd ⟨ws, hs, hx⟩ hno
    obtain ⟨ws, hs, hx, hmin⟩ := exists_least_walk E τ c X d hτ hd hd0 v t hex
    have hleast : leastDose E τ c X d v t = ((walkDose τ d t v ws : ℝ) : EReal) := by
      unfold leastDose
      refine IsGLB.sInf_eq (IsLeast.isGLB ⟨⟨ws, hs, hx, rfl⟩, ?_⟩)
      rintro _ ⟨ws', hs', hx', rfl⟩
      exact EReal.coe_le_coe_iff.2 (hmin ws' hs' hx')
    exact ⟨ws, hs, hx, by rw [← hleast]; exact hθ⟩

end Dose

/-! ### Bisection (`doseDeadline`) -/

section Bisection

variable (works : ℝ → Prop) [DecidablePred works] (tol : ℝ)

/-- One pass of `doseDeadline`'s loop on `(good, bad)`: while `bad - good > tolerance`,
test the midpoint and move the end it belongs to; otherwise stop (a fixed point). -/
noncomputable def bisectStep (p : ℝ × ℝ) : ℝ × ℝ :=
  if tol < p.2 - p.1 then
    (if works ((p.1 + p.2) / 2) then ((p.1 + p.2) / 2, p.2) else (p.1, (p.1 + p.2) / 2))
  else p

/-- The loop keeps its invariant: `good` is good and `bad` is bad. -/
theorem bisect_inv (G : Set ℝ) (hG : ∀ t, works t ↔ t ∈ G) (p : ℝ × ℝ) (h1 : p.1 ∈ G)
    (h2 : p.2 ∉ G) : (bisectStep works tol p).1 ∈ G ∧ (bisectStep works tol p).2 ∉ G := by
  unfold bisectStep
  split_ifs with hgap hmid
  · exact ⟨(hG _).1 hmid, h2⟩
  · exact ⟨h1, fun h => hmid ((hG _).2 h)⟩
  · exact ⟨h1, h2⟩

theorem bisect_iter_inv (G : Set ℝ) (hG : ∀ t, works t ↔ t ∈ G) (p : ℝ × ℝ) (h1 : p.1 ∈ G)
    (h2 : p.2 ∉ G) : ∀ n : ℕ,
      ((bisectStep works tol)^[n] p).1 ∈ G ∧ ((bisectStep works tol)^[n] p).2 ∉ G
  | 0 => ⟨h1, h2⟩
  | n + 1 => by
    rw [Function.iterate_succ_apply']
    obtain ⟨a, b⟩ := bisect_iter_inv G hG p h1 h2 n
    exact bisect_inv works tol G hG _ a b

/-- Each pass halves the gap, or stops. -/
theorem bisect_gap (p : ℝ × ℝ) :
    (bisectStep works tol p).2 - (bisectStep works tol p).1 =
      if tol < p.2 - p.1 then (p.2 - p.1) / 2 else p.2 - p.1 := by
  unfold bisectStep
  split_ifs <;> simp only <;> ring

/-- After `n` passes the gap is at most `max(tolerance, gap₀ / 2ⁿ)`. -/
theorem bisect_iter_gap (p : ℝ × ℝ) : ∀ n : ℕ,
    ((bisectStep works tol)^[n] p).2 - ((bisectStep works tol)^[n] p).1 ≤
      max tol ((p.2 - p.1) / 2 ^ n)
  | 0 => by simp
  | n + 1 => by
    rw [Function.iterate_succ_apply', bisect_gap]
    have ih := bisect_iter_gap p n
    split_ifs with h
    · have h' : ((bisectStep works tol)^[n] p).2 - ((bisectStep works tol)^[n] p).1 ≤
          (p.2 - p.1) / 2 ^ n := by
        rcases le_max_iff.1 ih with h' | h'
        · linarith
        · exact h'
      refine le_max_of_le_right ?_
      rw [pow_succ, div_mul_eq_div_div]
      linarith
    · exact le_max_of_le_left (not_lt.1 h)

/-- A good point is below the supremum of a downward closed set, a bad one above it. -/
theorem bracket_of_mem (G : Set ℝ) (hdown : ∀ t t', t ≤ t' → t' ∈ G → t ∈ G) {g b : ℝ}
    (hg : g ∈ G) (hb : b ∉ G) :
    (g : EReal) ≤ sSup ((fun t : ℝ => (t : EReal)) '' G) ∧
      sSup ((fun t : ℝ => (t : EReal)) '' G) ≤ (b : EReal) := by
  refine ⟨le_sSup ⟨g, hg, rfl⟩, sSup_le ?_⟩
  rintro _ ⟨t, ht, rfl⟩
  by_contra h
  push Not at h
  exact hb (hdown b t (EReal.coe_lt_coe_iff.1 h).le ht)

/-- **Bisection is correct.** For a downward closed good set `G` with supremum `T*`, start
from a good `good` and a bad `bad`, with `tolerance > 0`. The loop stops (after finitely many
passes) at a pair with `bad - good ≤ tolerance`, `good` still good and `bad` still bad, so
`good ≤ T* ≤ bad ≤ good + tolerance`: the returned `good` is a good departure within
`tolerance` of `T*`. -/
theorem bisect_correct (G : Set ℝ) (hG : ∀ t, works t ↔ t ∈ G)
    (hdown : ∀ t t', t ≤ t' → t' ∈ G → t ∈ G) (htol : 0 < tol) (p : ℝ × ℝ) (h1 : p.1 ∈ G)
    (h2 : p.2 ∉ G) :
    ∃ q : ℝ × ℝ, (∃ n : ℕ, (bisectStep works tol)^[n] p = q) ∧ bisectStep works tol q = q ∧
      q.2 - q.1 ≤ tol ∧ q.1 ∈ G ∧ q.2 ∉ G ∧
      (q.1 : EReal) ≤ sSup ((fun t : ℝ => (t : EReal)) '' G) ∧
      sSup ((fun t : ℝ => (t : EReal)) '' G) ≤ ((q.1 + tol : ℝ) : EReal) := by
  obtain ⟨n, hn⟩ := pow_unbounded_of_one_lt ((p.2 - p.1) / tol) (by norm_num : (1 : ℝ) < 2)
  have hpow : (0 : ℝ) < 2 ^ n := by positivity
  have hsmall : (p.2 - p.1) / 2 ^ n ≤ tol := by
    rw [div_lt_iff₀ htol] at hn
    rw [div_le_iff₀ hpow]
    linarith
  set q := (bisectStep works tol)^[n] p with hq
  have hgap : q.2 - q.1 ≤ tol := (bisect_iter_gap works tol p n).trans (max_le le_rfl hsmall)
  obtain ⟨hq1, hq2⟩ := bisect_iter_inv works tol G hG p h1 h2 n
  obtain ⟨hlo, hhi⟩ := bracket_of_mem G hdown hq1 hq2
  refine ⟨q, ⟨n, rfl⟩, ?_, hgap, hq1, hq2, hlo, hhi.trans (EReal.coe_le_coe_iff.2 (by linarith))⟩
  unfold bisectStep
  rw [ite_eq_right (not_lt.2 hgap)]

end Bisection

section DoseBisection

variable {V : Type*} (E : V → V → Prop) (τ : V → V → ℝ) (c : V → EReal) (X : V → Bool)
  (d : V → V → ℝ → ℝ)

/-- **`doseDeadline` brackets `T*`.** The code starts from `good = 0` (it checked `works(0)`)
and `bad = L(start)` from `escapeDeadlines` on the forecast's closing times. `bad` is bad:
the good set lies inside the safe set, which is `{t < L}` (`mem_safeSet_iff`). With an exact
`works` (`doseFront`), the loop stops with `good` good, `bad` bad and
`good ≤ T* ≤ bad ≤ good + tolerance`. -/
theorem doseDeadline_bisection (hd : ∀ v w, E v w → Monotone (d v w)) (θ : EReal) (v : V)
    (works : ℝ → Prop) [DecidablePred works]
    (hworks : ∀ t, works t ↔ t ∈ goodSet E τ c X d θ v) (tol : ℝ) (htol : 0 < tol)
    (h0 : (0 : ℝ) ∈ goodSet E τ c X d θ v) (b : ℝ) (hb : (b : EReal) = deadline E τ c X v) :
    ∃ q : ℝ × ℝ, (∃ n : ℕ, (bisectStep works tol)^[n] (0, b) = q) ∧
      bisectStep works tol q = q ∧ q.2 - q.1 ≤ tol ∧
      q.1 ∈ goodSet E τ c X d θ v ∧ q.2 ∉ goodSet E τ c X d θ v ∧
      (q.1 : EReal) ≤ doseDeadline E τ c X d θ v ∧
      doseDeadline E τ c X d θ v ≤ ((q.1 + tol : ℝ) : EReal) := by
  have hbad : b ∉ goodSet E τ c X d θ v := fun h => by
    have := (mem_safeSet_iff E τ c X v b).1 (goodSet_subset E τ c X d θ v h)
    rw [hb] at this
    exact lt_irrefl _ this
  exact bisect_correct works tol (goodSet E τ c X d θ v) hworks
    (fun t t' htt h => goodSet_downward E τ c X d hd htt h) htol (0, b) h0 hbad

end DoseBisection

/-! ## (3) The map is conservative (`doseDeadlineMap`) -/

section Grid

variable {V : Type*} [Fintype V] [DecidableEq V]

/-- The time grid of `doseDeadlineMap`: step `δ` seconds, layers `0..J`, link `v → w` takes
`steps v w ≥ 1` grid steps (the code: `max(1, ceil(len / speed / δ - 1e-9))`), and moving
along it from grid step `j` is charged `g v w j` (the code: `linkDose(v, w, jδ, (j + τ)δ)`). -/
structure Grid (V : Type*) where
  δ : ℝ
  J : ℕ
  steps : V → V → ℕ
  steps_pos : ∀ v w, 1 ≤ steps v w
  g : V → V → ℕ → ℝ

variable (E : V → V → Prop) [DecidableRel E] (c : V → EReal) (X : V → Bool) (G : Grid V)

/-- The code's backward pass, literally: `V_j(v) = ⊤` unless `j ≤ J` and `jδ < c(v)`;
then `V_j(exit) = 0`, and otherwise the least, over open links `v → w` with
`j + τ ≤ J` and `(j + τ)δ < c(w)`, of `g + V_{j+τ}(w)`. Links are indexed by pairs
`e = (v, w)` (the code scans `v`'s CSR row). -/
noncomputable def gridV (j : ℕ) (v : V) : WithTop ℝ :=
  if j ≤ G.J ∧ ((j * G.δ : ℝ) : EReal) < c v then
    (if X v = true then 0 else
      Finset.univ.inf fun e : V × V =>
        if _h : e.1 = v ∧ E e.1 e.2 ∧ j + G.steps e.1 e.2 ≤ G.J ∧
            ((((j + G.steps e.1 e.2 : ℕ) : ℝ) * G.δ : ℝ) : EReal) < c e.2 then
          ((G.g e.1 e.2 j : ℝ) : WithTop ℝ) + gridV (j + G.steps e.1 e.2) e.2
        else ⊤)
  else ⊤
termination_by G.J + 1 - j
decreasing_by have := G.steps_pos e.1 e.2; omega

/-- The defining equation of `gridV`. -/
theorem gridV_eq (j : ℕ) (v : V) :
    gridV E c X G j v =
      if j ≤ G.J ∧ ((j * G.δ : ℝ) : EReal) < c v then
        (if X v = true then 0 else
          Finset.univ.inf fun e : V × V =>
            if e.1 = v ∧ E e.1 e.2 ∧ j + G.steps e.1 e.2 ≤ G.J ∧
                ((((j + G.steps e.1 e.2 : ℕ) : ℝ) * G.δ : ℝ) : EReal) < c e.2 then
              ((G.g e.1 e.2 j : ℝ) : WithTop ℝ) + gridV E c X G (j + G.steps e.1 e.2) e.2
            else ⊤)
      else ⊤ := by
  rw [gridV]
  simp only [dite_eq_ite]

/-- The time-expanded grid as a `MinDose.Net`: links are pairs `(v, w)`, a move is charged
`g` when it is allowed (link open, `v` open at `jδ`, `w` open at `(j + τ)δ`) and `⊤`
otherwise; waiting is never allowed (`⊤`, and the DP is run with `hold = false`). -/
noncomputable def gridNet : MinDose.Net V (V × V) where
  tail := Prod.fst
  head := Prod.snd
  tau e := G.steps e.1 e.2
  tau_pos e := G.steps_pos e.1 e.2
  dose e j := if E e.1 e.2 ∧ ((j * G.δ : ℝ) : EReal) < c e.1 ∧
      ((((j + G.steps e.1 e.2 : ℕ) : ℝ) * G.δ : ℝ) : EReal) < c e.2 then
    ((G.g e.1 e.2 j : ℝ) : WithTop ℝ) else ⊤
  wait _ _ := ⊤
  isExit := X

/-- **The map's recursion is MinDose's backward induction without holdover**
(`MinDose.val false`) on `gridNet`, except at an exit that is already closed, where the code
says `⊤` and MinDose says `0`; no move ever reads such a value (a move into an exit needs it
open on arrival). -/
theorem gridV_eq_val : ∀ (k j : ℕ) (v : V), G.J + 1 - j = k →
    gridV E c X G j v = if X v = true ∧ ¬ ((j * G.δ : ℝ) : EReal) < c v then ⊤
      else MinDose.val (gridNet E c X G) false G.J j v := by
  intro k
  induction k using Nat.strong_induction_on with
  | _ k ih =>
  intro j v hk
  rw [gridV_eq]
  have hval := MinDose.val_eq (gridNet E c X G) false G.J j v
  have hex : (gridNet E c X G).isExit v = X v := rfl
  have hhold : ¬ (false = true ∧ j + 1 ≤ G.J) := fun h => Bool.false_ne_true h.1
  rw [hex, ite_eq_right hhold, min_top_left] at hval
  by_cases hj : j ≤ G.J
  · by_cases hc : ((j * G.δ : ℝ) : EReal) < c v
    · have hR : ¬ (X v = true ∧ ¬ ((j * G.δ : ℝ) : EReal) < c v) := fun h => h.2 hc
      rw [ite_eq_left (show j ≤ G.J ∧ ((j * G.δ : ℝ) : EReal) < c v from ⟨hj, hc⟩),
        ite_eq_right hR, hval]
      by_cases hx : X v = true
      · rw [ite_eq_left hx, ite_eq_left hx, ite_eq_left hj]
      · rw [ite_eq_right hx, ite_eq_right hx]
        refine Finset.inf_congr rfl (fun e _ => ?_)
        show _ = if e.1 = v ∧ j + G.steps e.1 e.2 ≤ G.J then
          (if E e.1 e.2 ∧ ((j * G.δ : ℝ) : EReal) < c e.1 ∧
              ((((j + G.steps e.1 e.2 : ℕ) : ℝ) * G.δ : ℝ) : EReal) < c e.2 then
            ((G.g e.1 e.2 j : ℝ) : WithTop ℝ) else ⊤) +
            MinDose.val (gridNet E c X G) false G.J (j + G.steps e.1 e.2) e.2 else ⊤
        by_cases h1 : e.1 = v ∧ j + G.steps e.1 e.2 ≤ G.J
        · by_cases h2 : E e.1 e.2 ∧
              ((((j + G.steps e.1 e.2 : ℕ) : ℝ) * G.δ : ℝ) : EReal) < c e.2
          · have hL : e.1 = v ∧ E e.1 e.2 ∧ j + G.steps e.1 e.2 ≤ G.J ∧
                ((((j + G.steps e.1 e.2 : ℕ) : ℝ) * G.δ : ℝ) : EReal) < c e.2 :=
              ⟨h1.1, h2.1, h1.2, h2.2⟩
            have hD : E e.1 e.2 ∧ ((j * G.δ : ℝ) : EReal) < c e.1 ∧
                ((((j + G.steps e.1 e.2 : ℕ) : ℝ) * G.δ : ℝ) : EReal) < c e.2 :=
              ⟨h2.1, by rw [h1.1]; exact hc, h2.2⟩
            have hN : ¬ (X e.2 = true ∧
                ¬ ((((j + G.steps e.1 e.2 : ℕ) : ℝ) * G.δ : ℝ) : EReal) < c e.2) :=
              fun h => h.2 h2.2
            have hlt : G.J + 1 - (j + G.steps e.1 e.2) < k := by
              have := h1.2
              have := G.steps_pos e.1 e.2
              omega
            rw [ite_eq_left hL, ite_eq_left h1, ite_eq_left hD,
              ih (G.J + 1 - (j + G.steps e.1 e.2)) hlt (j + G.steps e.1 e.2) e.2 rfl,
              ite_eq_right hN]
          · have hL : ¬ (e.1 = v ∧ E e.1 e.2 ∧ j + G.steps e.1 e.2 ≤ G.J ∧
                ((((j + G.steps e.1 e.2 : ℕ) : ℝ) * G.δ : ℝ) : EReal) < c e.2) :=
              fun h => h2 ⟨h.2.1, h.2.2.2⟩
            have hD : ¬ (E e.1 e.2 ∧ ((j * G.δ : ℝ) : EReal) < c e.1 ∧
                ((((j + G.steps e.1 e.2 : ℕ) : ℝ) * G.δ : ℝ) : EReal) < c e.2) :=
              fun h => h2 ⟨h.1, h.2.2⟩
            rw [ite_eq_right hL, ite_eq_left h1, ite_eq_right hD, WithTop.top_add]
        · have hL : ¬ (e.1 = v ∧ E e.1 e.2 ∧ j + G.steps e.1 e.2 ≤ G.J ∧
              ((((j + G.steps e.1 e.2 : ℕ) : ℝ) * G.δ : ℝ) : EReal) < c e.2) :=
            fun h => h1 ⟨h.1, h.2.2.1⟩
          rw [ite_eq_right hL, ite_eq_right h1]
    · rw [ite_eq_right (show ¬ (j ≤ G.J ∧ ((j * G.δ : ℝ) : EReal) < c v) from fun h => hc h.2)]
      by_cases hx : X v = true
      · rw [ite_eq_left (show X v = true ∧ ¬ ((j * G.δ : ℝ) : EReal) < c v from ⟨hx, hc⟩)]
      · rw [ite_eq_right (show ¬ (X v = true ∧ ¬ ((j * G.δ : ℝ) : EReal) < c v) from
          fun h => hx h.1), hval, ite_eq_right hx]
        symm
        refine (Finset.inf_eq_top_iff _ _).2 (fun e _ => ?_)
        show (if e.1 = v ∧ j + G.steps e.1 e.2 ≤ G.J then
          (if E e.1 e.2 ∧ ((j * G.δ : ℝ) : EReal) < c e.1 ∧
              ((((j + G.steps e.1 e.2 : ℕ) : ℝ) * G.δ : ℝ) : EReal) < c e.2 then
            ((G.g e.1 e.2 j : ℝ) : WithTop ℝ) else ⊤) +
            MinDose.val (gridNet E c X G) false G.J (j + G.steps e.1 e.2) e.2 else ⊤) = ⊤
        by_cases h1 : e.1 = v ∧ j + G.steps e.1 e.2 ≤ G.J
        · have hD : ¬ (E e.1 e.2 ∧ ((j * G.δ : ℝ) : EReal) < c e.1 ∧
              ((((j + G.steps e.1 e.2 : ℕ) : ℝ) * G.δ : ℝ) : EReal) < c e.2) :=
            fun h => hc (by rw [← h1.1]; exact h.2.1)
          rw [ite_eq_left h1, ite_eq_right hD, WithTop.top_add]
        · rw [ite_eq_right h1]
  · rw [ite_eq_right (show ¬ (j ≤ G.J ∧ ((j * G.δ : ℝ) : EReal) < c v) from fun h => hj h.1)]
    split_ifs
    · rfl
    · exact (MinDose.val_gt _ false v (by omega)).symm

/-- So MinDose's exactness transfers: wherever the code's value is not the closed-exit
special case, `V_j(v)` is the least grid dose of a no-wait schedule in the time-expanded grid
that reaches an exit by layer `J`, and it is attained (`MinDose.dp_exact`). -/
theorem gridV_dp_exact (j : ℕ) (v : V) (hv : ¬ (X v = true ∧ ¬ ((j * G.δ : ℝ) : EReal) < c v)) :
    (∀ s, MinDose.Valid (gridNet E c X G) false G.J j v s →
      gridV E c X G j v ≤ MinDose.cost (gridNet E c X G) j v s) ∧
    (gridV E c X G j v ≠ ⊤ → ∃ s, MinDose.Valid (gridNet E c X G) false G.J j v s ∧
      MinDose.cost (gridNet E c X G) j v s = gridV E c X G j v) := by
  rw [gridV_eq_val E c X G _ j v rfl, ite_eq_right hv]
  exact MinDose.dp_exact _ false G.J j v

omit [Fintype V] [DecidableEq V] in
/-- With grid doses non-decreasing in the grid step (heat that never falls) and `δ ≥ 0`, the
grid's move doses are non-decreasing in the departure step (closures never reopen). -/
theorem gridNet_dose_mono (hδ : 0 ≤ G.δ) (hg : ∀ v w, Monotone (G.g v w)) (e : V × V) :
    Monotone ((gridNet E c X G).dose e) := by
  intro j j' hjj
  show (if E e.1 e.2 ∧ ((j * G.δ : ℝ) : EReal) < c e.1 ∧
      ((((j + G.steps e.1 e.2 : ℕ) : ℝ) * G.δ : ℝ) : EReal) < c e.2 then
    ((G.g e.1 e.2 j : ℝ) : WithTop ℝ) else ⊤) ≤
    (if E e.1 e.2 ∧ ((j' * G.δ : ℝ) : EReal) < c e.1 ∧
      ((((j' + G.steps e.1 e.2 : ℕ) : ℝ) * G.δ : ℝ) : EReal) < c e.2 then
    ((G.g e.1 e.2 j' : ℝ) : WithTop ℝ) else ⊤)
  have h1 : ((j : ℝ) * G.δ) ≤ (j' : ℝ) * G.δ :=
    mul_le_mul_of_nonneg_right (Nat.cast_le.2 hjj) hδ
  have h2 : (((j + G.steps e.1 e.2 : ℕ) : ℝ) * G.δ) ≤ ((j' + G.steps e.1 e.2 : ℕ) : ℝ) * G.δ :=
    mul_le_mul_of_nonneg_right (Nat.cast_le.2 (by omega)) hδ
  split_ifs with ha hb hb'
  · exact WithTop.coe_le_coe.2 (hg _ _ hjj)
  · exact le_top
  · exact absurd ⟨hb'.1, lt_of_le_of_lt (EReal.coe_le_coe_iff.2 h1) hb'.2.1,
      lt_of_le_of_lt (EReal.coe_le_coe_iff.2 h2) hb'.2.2⟩ ha
  · exact le_rfl

/-- MinDose's value without holdover is non-decreasing in the start step, via
`MinDose.strip_le` (leaving earlier along the same moves costs no more). -/
theorem gridNet_val_mono (hδ : 0 ≤ G.δ) (hg : ∀ v w, Monotone (G.g v w)) (v : V) {j j' : ℕ}
    (hjj : j ≤ j') :
    MinDose.val (gridNet E c X G) false G.J j v ≤ MinDose.val (gridNet E c X G) false G.J j' v := by
  by_cases h : MinDose.val (gridNet E c X G) false G.J j' v = ⊤
  · rw [h]
    exact le_top
  · obtain ⟨s, hs, hcost⟩ := (MinDose.dp_exact _ false G.J j' v).2 h
    obtain ⟨h1, h2⟩ := MinDose.strip_le _ (gridNet_dose_mono E c X G hδ hg)
      (fun _ _ => le_top) false G.J s j' j v hs hjj
    rw [← hcost]
    exact (MinDose.val_le_cost _ false G.J (MinDose.strip s) j v h1).trans h2

/-- **`V_j(v)` is non-decreasing in `j`**, so the grid times at which `V` is finite and
`≤ limit` form an initial segment and "the largest good grid time" (the map's deadline, the
first hit of the code's backward loop) is well defined. -/
theorem gridV_mono (hδ : 0 ≤ G.δ) (hg : ∀ v w, Monotone (G.g v w)) (v : V) {j j' : ℕ}
    (hjj : j ≤ j') : gridV E c X G j v ≤ gridV E c X G j' v := by
  rw [gridV_eq_val E c X G _ j v rfl, gridV_eq_val E c X G _ j' v rfl]
  have hcl : ((j' * G.δ : ℝ) : EReal) < c v → ((j * G.δ : ℝ) : EReal) < c v := fun h =>
    lt_of_le_of_lt (EReal.coe_le_coe_iff.2 (mul_le_mul_of_nonneg_right (Nat.cast_le.2 hjj) hδ)) h
  split_ifs with h1 h2 h2
  · exact le_rfl
  · exact absurd ⟨h1.1, fun h => h1.2 (hcl h)⟩ h2
  · exact le_top
  · exact gridNet_val_mono E c X G hδ hg v hjj

/-- A finite infimum over a `Finset` that equals a number is attained. -/
theorem exists_eq_of_inf_eq_coe {ι : Type*} (s : Finset ι) (f : ι → WithTop ℝ) (x : ℝ)
    (h : s.inf f = x) : ∃ e ∈ s, f e = x := by
  have hne : s.Nonempty := by
    by_contra hemp
    rw [Finset.not_nonempty_iff_eq_empty] at hemp
    rw [hemp, Finset.inf_empty] at h
    exact WithTop.top_ne_coe h
  obtain ⟨e, he, hfe⟩ := Finset.exists_mem_eq_inf s hne f
  exact ⟨e, he, hfe ▸ h⟩

/-- **The map is conservative.** Suppose link times are rounded **up** to whole grid steps
(`τ ≤ steps · δ`) and the grid charges at least the real dose of any no-later departure along
the link (`d v w t ≤ g v w j` for `t ≤ jδ`; `hotDose_charge` proves it for Studio's doses).
If `V_j(v) = x` is finite, then leaving `v` at any real `t ≤ jδ` there is a real safe walk to
an exit with dose `≤ x`. The real walk takes the grid schedule's links without waiting: by
induction it reaches each node no later than the grid time (so before it closes), and each
link is entered no later and walked no longer than the grid assumed, so it accrues no more
dose than the grid charged. -/
theorem grid_conservative (τ : V → V → ℝ) (d : V → V → ℝ → ℝ)
    (hround : ∀ v w, E v w → τ v w ≤ G.steps v w * G.δ)
    (hcharge : ∀ v w, E v w → ∀ (t : ℝ) (j : ℕ), t ≤ j * G.δ → d v w t ≤ G.g v w j) :
    ∀ (k j : ℕ) (v : V), G.J + 1 - j = k → ∀ x : ℝ, gridV E c X G j v = (x : WithTop ℝ) →
      ∀ t : ℝ, t ≤ j * G.δ → ∃ ws, SafeWalk E τ c t v ws ∧ X (wEnd v ws) = true ∧
        walkDose τ d t v ws ≤ x := by
  intro k
  induction k using Nat.strong_induction_on with
  | _ k ih =>
  intro j v hk x hx t ht
  rw [gridV_eq] at hx
  by_cases hjc : j ≤ G.J ∧ ((j * G.δ : ℝ) : EReal) < c v
  · rw [ite_eq_left hjc] at hx
    have htc : (t : EReal) < c v := lt_of_le_of_lt (EReal.coe_le_coe_iff.2 ht) hjc.2
    by_cases hX : X v = true
    · rw [ite_eq_left hX] at hx
      have hx0 : x = 0 := by exact_mod_cast hx.symm
      refine ⟨[], htc, hX, ?_⟩
      show (0 : ℝ) ≤ x
      rw [hx0]
    · rw [ite_eq_right hX] at hx
      obtain ⟨⟨a, b⟩, -, hfe⟩ := exists_eq_of_inf_eq_coe _ _ x hx
      dsimp only at hfe
      split_ifs at hfe with hcond
      · obtain ⟨rfl, hE, hJ, hcb⟩ := hcond
        have hne : gridV E c X G (j + G.steps a b) b ≠ ⊤ := by
          intro htop
          rw [htop, WithTop.add_top] at hfe
          exact WithTop.top_ne_coe hfe
        obtain ⟨y, hy⟩ := WithTop.ne_top_iff_exists.1 hne
        rw [← hy] at hfe
        have hxy : G.g a b j + y = x := by exact_mod_cast hfe
        have hT : t + τ a b ≤ ((j + G.steps a b : ℕ) : ℝ) * G.δ := by
          push_cast
          rw [add_mul]
          linarith [hround a b hE]
        obtain ⟨ws, hs, hxw, hdose⟩ := ih (G.J + 1 - (j + G.steps a b))
          (by have := G.steps_pos a b; omega) (j + G.steps a b) b rfl y hy.symm (t + τ a b) hT
        refine ⟨b :: ws, ⟨htc, hE, hs⟩, hxw, ?_⟩
        show d a b t + walkDose τ d (t + τ a b) b ws ≤ x
        rw [← hxy]
        exact add_le_add (hcharge a b hE t j ht) hdose
      · exact absurd hfe WithTop.top_ne_coe
  · rw [ite_eq_right hjc] at hx
    exact absurd hx WithTop.top_ne_coe

/-- A grid time the map calls good is a good real departure. -/
theorem map_mem_goodSet (τ : V → V → ℝ) (d : V → V → ℝ → ℝ)
    (hround : ∀ v w, E v w → τ v w ≤ G.steps v w * G.δ)
    (hcharge : ∀ v w, E v w → ∀ (t : ℝ) (j : ℕ), t ≤ j * G.δ → d v w t ≤ G.g v w j)
    (θ : EReal) (j : ℕ) (v : V) (x : ℝ) (hx : gridV E c X G j v = x) (hθ : (x : EReal) ≤ θ) :
    ((j : ℝ) * G.δ) ∈ goodSet E τ c X d θ v := by
  obtain ⟨ws, hs, hxw, hd⟩ :=
    grid_conservative E c X G τ d hround hcharge _ j v rfl x hx _ le_rfl
  exact ⟨ws, hs, hxw, (EReal.coe_le_coe_iff.2 hd).trans hθ⟩

/-- **The map is never later than the exact deadline**: every grid time `jδ` with `V_j(v)`
finite and `≤ θ` (in particular the map's deadline, the largest such) is `≤ T*(v) ≤ L(v)`.
How far below `T*` the map can be (the rounding gap) is **not** bounded by these proofs. -/
theorem map_le_doseDeadline (τ : V → V → ℝ) (d : V → V → ℝ → ℝ)
    (hround : ∀ v w, E v w → τ v w ≤ G.steps v w * G.δ)
    (hcharge : ∀ v w, E v w → ∀ (t : ℝ) (j : ℕ), t ≤ j * G.δ → d v w t ≤ G.g v w j)
    (θ : EReal) (j : ℕ) (v : V) (x : ℝ) (hx : gridV E c X G j v = x) (hθ : (x : EReal) ≤ θ) :
    ((((j : ℝ) * G.δ : ℝ)) : EReal) ≤ doseDeadline E τ c X d θ v ∧
      doseDeadline E τ c X d θ v ≤ deadline E τ c X v :=
  ⟨le_doseDeadline E τ c X d (map_mem_goodSet E c X G τ d hround hcharge θ j v x hx hθ),
    doseDeadline_le_deadline E τ c X d θ v⟩

end Grid

/-! ### Studio's rounding and doses meet the hypotheses (and where they do not) -/

/-- Rounding a link time up to whole steps, at least one (`max(1, ceil(a / δ))`), covers it:
`a ≤ steps · δ`. This is the hypothesis `hround` of `grid_conservative`. -/
theorem ceil_steps_cover (a δ : ℝ) (hδ : 0 < δ) : a ≤ ((max 1 ⌈a / δ⌉₊ : ℕ) : ℝ) * δ := by
  have h1 : a / δ ≤ (⌈a / δ⌉₊ : ℝ) := Nat.le_ceil _
  have h2 : (⌈a / δ⌉₊ : ℝ) ≤ ((max 1 ⌈a / δ⌉₊ : ℕ) : ℝ) := by exact_mod_cast le_max_right _ _
  rw [div_le_iff₀ hδ] at h1
  nlinarith

/-- **The code's rounding can round down.** `doseDeadlineMap` uses
`max(1, ceil(a / δ - 1e-9))`. For a link of `a = 1 + 5e-10` s on a grid of `δ = 1` s this is
one step, shorter than the link; with the head closing at `1 + 3e-10` s the grid admits the
move from `j = 0` (`(0 + 1) δ < c`) while the real walker arrives at `a ≥ c`. So `hround`
fails and the map can be later than the exact deadline (by at most `1e-9 δ` per link). -/
theorem epsilon_rounds_down :
    ((max 1 ⌈(1 + 5e-10 : ℝ) / 1 - 1e-9⌉₊ : ℕ) : ℝ) * 1 < 1 + 5e-10 ∧
      (((((0 + max 1 ⌈(1 + 5e-10 : ℝ) / 1 - 1e-9⌉₊ : ℕ) : ℝ) * 1 : ℝ) : EReal) <
        ((1 + 3e-10 : ℝ) : EReal)) ∧
      ¬ (((0 + (1 + 5e-10) : ℝ) : EReal) < ((1 + 3e-10 : ℝ) : EReal)) := by
  have h : ⌈(1 + 5e-10 : ℝ) / 1 - 1e-9⌉₊ = 1 := by
    rw [Nat.ceil_eq_iff one_ne_zero]
    norm_num
  rw [h]
  refine ⟨by norm_num, EReal.coe_lt_coe_iff.2 (by norm_num), fun hlt => ?_⟩
  have := EReal.coe_lt_coe_iff.1 hlt
  norm_num at this

section StudioDoses

variable {V : Type*}

/-- Studio's link dose (`forecast.linkDose`): walking `v → w` for `a` seconds from `t`
accrues the integral of the hotter end's FED rate, `∫_t^{t+a} max(r_v, r_w)`. -/
noncomputable def hotDose (r : V → ℝ → ℝ) (v w : V) (a t : ℝ) : ℝ :=
  ∫ s in t..t + a, max (r v s) (r w s)

theorem hotRate_mono (r : V → ℝ → ℝ) (hr : ∀ v, Monotone (r v)) (v w : V) :
    Monotone (fun s => max (r v s) (r w s)) :=
  fun _ _ h => max_le_max (hr v h) (hr w h)

/-- Heat that never falls makes the link dose non-decreasing in the departure time. Reuses
`Tenability.T10_edge_dose_mono_entry`. -/
theorem hotDose_mono (r : V → ℝ → ℝ) (hr : ∀ v, Monotone (r v)) (v w : V) {a : ℝ}
    (ha : 0 ≤ a) : Monotone (hotDose r v w a) :=
  fun _ _ h => Tenability.T10_edge_dose_mono_entry _ (hotRate_mono r hr v w) a ha h

theorem hotDose_nonneg (r : V → ℝ → ℝ) (hr0 : ∀ v s, 0 ≤ r v s) (v w : V) {a : ℝ}
    (ha : 0 ≤ a) (t : ℝ) : 0 ≤ hotDose r v w a t :=
  intervalIntegral.integral_nonneg (by linarith) (fun s _ => le_max_of_le_left (hr0 v s))

/-- A longer walk along the same link from the same time accrues no less (rate `≥ 0`). -/
theorem hotDose_longer (r : V → ℝ → ℝ) (hr : ∀ v, Monotone (r v)) (hr0 : ∀ v s, 0 ≤ r v s)
    (v w : V) {a b : ℝ} (ha : 0 ≤ a) (hab : a ≤ b) (t : ℝ) :
    hotDose r v w a t ≤ hotDose r v w b t :=
  intervalIntegral.integral_mono_interval le_rfl (by linarith) (by linarith)
    (Filter.Eventually.of_forall fun s => le_max_of_le_left (hr0 v s))
    (hotRate_mono r hr v w).intervalIntegrable

/-- **The grid charges at least the real dose**: entering no later (`t ≤ t'`) and walking no
longer (`a ≤ b`) accrues no more. This is `hcharge` of `grid_conservative` with `t' = jδ`,
`a = τ`, `b = steps · δ`. -/
theorem hotDose_charge (r : V → ℝ → ℝ) (hr : ∀ v, Monotone (r v)) (hr0 : ∀ v s, 0 ≤ r v s)
    (v w : V) {a b t t' : ℝ} (ha : 0 ≤ a) (hab : a ≤ b) (htt : t ≤ t') :
    hotDose r v w a t ≤ hotDose r v w b t' :=
  (hotDose_mono r hr v w ha htt).trans (hotDose_longer r hr hr0 v w ha hab t')

end StudioDoses

section Studio

variable {V : Type*} [Fintype V] [DecidableEq V] (E : V → V → Prop) [DecidableRel E]
  (τ : V → V → ℝ) (c : V → EReal) (X : V → Bool) (G : Grid V) (r : V → ℝ → ℝ)

/-- **Studio's map, end to end.** Rates that never fall and are `≥ 0` (the forecast keeps the
running maximum of the heat), link times `τ ≥ 0`, a grid step `δ > 0`, link steps rounded up
(`max(1, ceil(τ / δ))`, without the code's `- 1e-9`), real link doses `hotDose r v w τ` and
grid doses `hotDose r v w (steps δ) (jδ)`. Then the map's `V` is non-decreasing in `j`, and
every grid time it calls good is `≤ T*(v) ≤ L(v)`. -/
theorem studio_map_le_exact (hτ : ∀ v w, 0 ≤ τ v w) (hδ : 0 < G.δ)
    (hr : ∀ v, Monotone (r v)) (hr0 : ∀ v s, 0 ≤ r v s)
    (hsteps : ∀ v w, G.steps v w = max 1 ⌈τ v w / G.δ⌉₊)
    (hgd : ∀ v w j, G.g v w j = hotDose r v w (G.steps v w * G.δ) (j * G.δ))
    (θ : EReal) (v : V) :
    (∀ j j' : ℕ, j ≤ j' → gridV E c X G j v ≤ gridV E c X G j' v) ∧
    ∀ (j : ℕ) (x : ℝ), gridV E c X G j v = x → (x : EReal) ≤ θ →
      ((((j : ℝ) * G.δ : ℝ)) : EReal) ≤ doseDeadline E τ c X (fun v w => hotDose r v w (τ v w)) θ v ∧
      doseDeadline E τ c X (fun v w => hotDose r v w (τ v w)) θ v ≤ deadline E τ c X v := by
  have hround : ∀ v w, E v w → τ v w ≤ G.steps v w * G.δ := fun v w _ => by
    rw [hsteps]
    exact ceil_steps_cover _ _ hδ
  have hcharge : ∀ v w, E v w → ∀ (t : ℝ) (j : ℕ), t ≤ j * G.δ →
      hotDose r v w (τ v w) t ≤ G.g v w j := fun v w hvw t j ht => by
    rw [hgd]
    exact hotDose_charge r hr hr0 v w (hτ v w) (hround v w hvw) ht
  have hg : ∀ v w, Monotone (G.g v w) := fun v w j j' hjj => by
    rw [hgd, hgd]
    exact hotDose_mono r hr v w (by positivity) (mul_le_mul_of_nonneg_right (Nat.cast_le.2 hjj) hδ.le)
  exact ⟨fun j j' hjj => gridV_mono E c X G hδ.le hg v hjj, fun j x hx hθ =>
    map_le_doseDeadline E c X G τ _ hround hcharge θ j v x hx hθ⟩

end Studio

/-! ## (4) The dose limit can bind -/

section Example

/-- A start `0` and an exit `1`, one link `0 → 1`. -/
def exE (v w : Fin 2) : Prop := v = 0 ∧ w = 1

/-- The link takes 1 s. -/
def exτ (_ _ : Fin 2) : ℝ := 1

/-- Both nodes close at 10 s. -/
def exc (_ : Fin 2) : EReal := ((10 : ℝ) : EReal)

/-- Node `1` is the exit. -/
def exX (v : Fin 2) : Bool := decide (v = 1)

/-- The link's dose leaving at `t`: `clamp(t - 4, 0, 1)`, the integral over `[t, t + 1]` of
a rate that steps from 0 to 1 per second at time 5 (taken here in closed form). It is
non-decreasing and non-negative. -/
noncomputable def exd (_ _ : Fin 2) (t : ℝ) : ℝ := max 0 (min 1 (t - 4))

/-- In the example the only walk from `0` to the exit is `[1]`, safe iff `t < 9`. -/
theorem ex_safeWalk_iff (t : ℝ) (ws : List (Fin 2)) :
    SafeWalk exE exτ exc t 0 ws ∧ exX (wEnd 0 ws) = true ↔ ws = [1] ∧ t < 9 := by
  constructor
  · rintro ⟨hs, hx⟩
    rcases ws with _ | ⟨w, _ | ⟨x, ws⟩⟩
    · exact absurd hx (by decide)
    · obtain ⟨-, ⟨-, rfl⟩, h3⟩ := hs
      refine ⟨rfl, ?_⟩
      have : ((t + 1 : ℝ) : EReal) < ((10 : ℝ) : EReal) := h3
      have := EReal.coe_lt_coe_iff.1 this
      linarith
    · obtain ⟨-, ⟨-, rfl⟩, -, ⟨h, -⟩, -⟩ := hs
      exact absurd h (by decide)
  · rintro ⟨rfl, ht⟩
    refine ⟨⟨?_, ⟨rfl, rfl⟩, ?_⟩, rfl⟩
    · show ((t : ℝ) : EReal) < ((10 : ℝ) : EReal)
      exact EReal.coe_lt_coe_iff.2 (by linarith)
    · show ((t + 1 : ℝ) : EReal) < ((10 : ℝ) : EReal)
      exact EReal.coe_lt_coe_iff.2 (by linarith)

/-- `L(0) = 9`: the exit closes at 10 and the walk takes 1 s. -/
theorem ex_deadline : deadline exE exτ exc exX 0 = ((9 : ℝ) : EReal) := by
  apply eq_of_coe_lt_iff
  intro t
  rw [← mem_safeSet_iff]
  constructor
  · rintro ⟨ws, h⟩
    exact EReal.coe_lt_coe_iff.2 ((ex_safeWalk_iff t ws).1 h).2
  · intro h
    exact ⟨[1], (ex_safeWalk_iff t [1]).2 ⟨rfl, EReal.coe_lt_coe_iff.1 h⟩⟩

/-- With the limit `θ = 1/2`, leaving at `t` is good iff `t ≤ 4.5`. -/
theorem ex_goodSet (t : ℝ) :
    t ∈ goodSet exE exτ exc exX exd ((1 / 2 : ℝ) : EReal) 0 ↔ t ≤ 9 / 2 := by
  constructor
  · rintro ⟨ws, hs, hx, hθ⟩
    obtain ⟨rfl, -⟩ := (ex_safeWalk_iff t ws).1 ⟨hs, hx⟩
    have h1 : walkDose exτ exd t 0 [1] ≤ 1 / 2 := EReal.coe_le_coe_iff.1 hθ
    simp only [walkDose, exd, add_zero] at h1
    by_contra h
    push Not at h
    have h2 : 1 / 2 < min 1 (t - 4) := lt_min (by norm_num) (by linarith)
    linarith [le_max_right 0 (min 1 (t - 4))]
  · intro ht
    refine ⟨[1], ((ex_safeWalk_iff t [1]).2 ⟨rfl, by linarith⟩).1, rfl, ?_⟩
    apply EReal.coe_le_coe_iff.2
    simp only [walkDose, exd, add_zero]
    exact max_le (by norm_num) ((min_le_right _ _).trans (by linarith))

/-- `T*(0) = 4.5`. -/
theorem ex_doseDeadline :
    doseDeadline exE exτ exc exX exd ((1 / 2 : ℝ) : EReal) 0 = ((9 / 2 : ℝ) : EReal) := by
  have hset : goodSet exE exτ exc exX exd ((1 / 2 : ℝ) : EReal) 0 = Set.Iic (9 / 2) :=
    Set.ext fun t => ex_goodSet t
  unfold doseDeadline
  rw [hset]
  refine IsLUB.sSup_eq (IsGreatest.isLUB ⟨⟨9 / 2, Set.mem_Iic.2 le_rfl, rfl⟩, ?_⟩)
  rintro _ ⟨t, ht, rfl⟩
  exact EReal.coe_le_coe_iff.2 ht

/-- **The dose limit binds**: on a one-link network whose link dose is non-decreasing and
non-negative, the hard deadline is `L(0) = 9` s but with the limit `1/2` the dose deadline is
`T*(0) = 4.5` s. Waiting until just before `L` would get the occupant out alive by the
untenability test, with twice the allowed dose. -/
theorem example_dose_limit_binds :
    (∀ v w, exE v w → Monotone (exd v w)) ∧ (∀ v w, exE v w → ∀ t, 0 ≤ exd v w t) ∧
      doseDeadline exE exτ exc exX exd ((1 / 2 : ℝ) : EReal) 0 = ((9 / 2 : ℝ) : EReal) ∧
      deadline exE exτ exc exX 0 = ((9 : ℝ) : EReal) ∧
      doseDeadline exE exτ exc exX exd ((1 / 2 : ℝ) : EReal) 0 < deadline exE exτ exc exX 0 := by
  refine ⟨fun v w _ t t' h => max_le_max le_rfl (min_le_min le_rfl (by linarith)),
    fun v w _ t => le_max_left _ _, ex_doseDeadline, ex_deadline, ?_⟩
  rw [ex_doseDeadline, ex_deadline]
  exact EReal.coe_lt_coe_iff.2 (by norm_num)

/-- Two rooms joined both ways, no exit. -/
def cyE (v w : Fin 2) : Prop := v ≠ w

/-- **The Bellman equation alone does not determine `L`.** On two rooms joined both ways
(1 s), with no exit and nothing ever closing, the labelling `⊤` everywhere satisfies
`d(v) = max over links of min(u(v), d(w) - τ)`, yet `L = ⊥` (no exit is reachable). So a
check that only tests the Bellman equation does not verify `escapeDeadlines`; `L` is the
least supersolution (`deadline_le_of_super`), which the label-setting loop builds from `⊥`.
(With every `τ > 0` and every closing time finite, as with `forecast.closes`, the argmax chain
of a finite solution strictly increases, so it cannot cycle; that uniqueness is not proved
here.) -/
theorem bellman_not_unique :
    (∀ v : Fin 2, (⊤ : EReal) =
      ⨆ (w : Fin 2) (_ : cyE v w), min (⊤ : EReal) ((⊤ : EReal) - ((1 : ℝ) : EReal))) ∧
    ∀ v : Fin 2, deadline cyE (fun _ _ => 1) (fun _ => ⊤) (fun _ => false) v = ⊥ := by
  refine ⟨fun v => ?_, fun v => ?_⟩
  · rw [EReal.top_sub_coe, min_self]
    refine le_antisymm ?_ le_top
    fin_cases v
    · exact le_iSup₂_of_le (f := fun (w : Fin 2) (_ : cyE 0 w) => (⊤ : EReal)) 1
        (show (0 : Fin 2) ≠ 1 by decide) le_rfl
    · exact le_iSup₂_of_le (f := fun (w : Fin 2) (_ : cyE 1 w) => (⊤ : EReal)) 0
        (show (1 : Fin 2) ≠ 0 by decide) le_rfl
  · refine (deadline_eq_bot_iff _ _ _ _ (fun _ => bot_lt_top) v).2 ?_
    rintro ⟨ws, -, hx⟩
    exact Bool.false_ne_true hx

end Example

end FireEgress.Deadline

#print axioms FireEgress.Deadline.coe_add_lt_iff
#print axioms FireEgress.Deadline.eq_of_coe_lt_iff
#print axioms FireEgress.Deadline.le_of_coe_lt_imp
#print axioms FireEgress.Deadline.bot_lt_sub_coe
#print axioms FireEgress.Deadline.safeWalk_start
#print axioms FireEgress.Deadline.safeWalk_anti
#print axioms FireEgress.Deadline.safeWalk_iff_margin
#print axioms FireEgress.Deadline.mem_safeSet_iff_margin
#print axioms FireEgress.Deadline.safeSet_downward
#print axioms FireEgress.Deadline.safeSet_open
#print axioms FireEgress.Deadline.mem_safeSet_iff
#print axioms FireEgress.Deadline.deadline_eq_sSup_margin
#print axioms FireEgress.Deadline.deadline_exit
#print axioms FireEgress.Deadline.mem_safeSet_cons
#print axioms FireEgress.Deadline.deadline_bellman
#print axioms FireEgress.Deadline.deadline_bellman_finset
#print axioms FireEgress.Deadline.deadline_link
#print axioms FireEgress.Deadline.deadline_le_of_super
#print axioms FireEgress.Deadline.margin_bot_lt
#print axioms FireEgress.Deadline.deadline_eq_bot_iff
#print axioms FireEgress.Deadline.walkTime_append
#print axioms FireEgress.Deadline.walkTime_nonneg
#print axioms FireEgress.Deadline.safeWalk_append
#print axioms FireEgress.Deadline.gateArc_top
#print axioms FireEgress.Deadline.gateArc_coe
#print axioms FireEgress.Deadline.gateArc_mono
#print axioms FireEgress.Deadline.gateArc_ge
#print axioms FireEgress.Deadline.wArr_gate_top
#print axioms FireEgress.Deadline.safeWalk_iff_wArr
#print axioms FireEgress.Deadline.safeSet_downward_of_fifo
#print axioms FireEgress.Deadline.mem_safeSet_iff_arr
#print axioms FireEgress.Deadline.waiting_never_helps_safe
#print axioms FireEgress.Deadline.departureSafe_correct
#print axioms FireEgress.Deadline.departureSafe_early_true
#print axioms FireEgress.Deadline.dinit_inv
#print axioms FireEgress.Deadline.unsettled_le
#print axioms FireEgress.Deadline.dselectable_exact
#print axioms FireEgress.Deadline.dstep_inv
#print axioms FireEgress.Deadline.dreachable_inv
#print axioms FireEgress.Deadline.escapeDeadlines_correct
#print axioms FireEgress.Deadline.drelax_step
#print axioms FireEgress.Deadline.dexists_done
#print axioms FireEgress.Deadline.escapeDeadlines_total
#print axioms FireEgress.Deadline.walkDose_mono
#print axioms FireEgress.Deadline.walkDose_nonneg
#print axioms FireEgress.Deadline.walkDose_append
#print axioms FireEgress.Deadline.leastDose_mono
#print axioms FireEgress.Deadline.goodSet_downward
#print axioms FireEgress.Deadline.goodSet_subset
#print axioms FireEgress.Deadline.doseDeadline_le_deadline
#print axioms FireEgress.Deadline.goodSet_top
#print axioms FireEgress.Deadline.doseDeadline_top
#print axioms FireEgress.Deadline.le_doseDeadline
#print axioms FireEgress.Deadline.mem_goodSet_of_lt
#print axioms FireEgress.Deadline.doseDeadline_le_of_not_mem
#print axioms FireEgress.Deadline.leastDose_le_of_good
#print axioms FireEgress.Deadline.cut_cycle
#print axioms FireEgress.Deadline.exists_simple_le
#print axioms FireEgress.Deadline.exists_least_walk
#print axioms FireEgress.Deadline.good_iff_leastDose
#print axioms FireEgress.Deadline.bisect_inv
#print axioms FireEgress.Deadline.bisect_iter_inv
#print axioms FireEgress.Deadline.bisect_gap
#print axioms FireEgress.Deadline.bisect_iter_gap
#print axioms FireEgress.Deadline.bracket_of_mem
#print axioms FireEgress.Deadline.bisect_correct
#print axioms FireEgress.Deadline.doseDeadline_bisection
#print axioms FireEgress.Deadline.gridV_eq
#print axioms FireEgress.Deadline.gridV_eq_val
#print axioms FireEgress.Deadline.gridV_dp_exact
#print axioms FireEgress.Deadline.gridNet_dose_mono
#print axioms FireEgress.Deadline.gridNet_val_mono
#print axioms FireEgress.Deadline.gridV_mono
#print axioms FireEgress.Deadline.exists_eq_of_inf_eq_coe
#print axioms FireEgress.Deadline.grid_conservative
#print axioms FireEgress.Deadline.map_mem_goodSet
#print axioms FireEgress.Deadline.map_le_doseDeadline
#print axioms FireEgress.Deadline.ceil_steps_cover
#print axioms FireEgress.Deadline.epsilon_rounds_down
#print axioms FireEgress.Deadline.hotRate_mono
#print axioms FireEgress.Deadline.hotDose_mono
#print axioms FireEgress.Deadline.hotDose_nonneg
#print axioms FireEgress.Deadline.hotDose_longer
#print axioms FireEgress.Deadline.hotDose_charge
#print axioms FireEgress.Deadline.studio_map_le_exact
#print axioms FireEgress.Deadline.ex_safeWalk_iff
#print axioms FireEgress.Deadline.ex_deadline
#print axioms FireEgress.Deadline.ex_goodSet
#print axioms FireEgress.Deadline.ex_doseDeadline
#print axioms FireEgress.Deadline.example_dose_limit_binds
#print axioms FireEgress.Deadline.bellman_not_unique
