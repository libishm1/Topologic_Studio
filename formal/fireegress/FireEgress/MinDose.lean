import Mathlib
import FireEgress.Tenability
import FireEgress.GraphNumerics

/-!
# Minimum-dose egress on the time-expanded graph (T17)

The literature review (`wiki/research/fire-egress-methods-2026-10-27.md`, sections 4.1-4.3
and 5.3; `wiki/research/fire-equations-verified-2026-10-27.json`, entry `T17-dose-dp`)
proposes routing an evacuee so as to minimise the fractional effective dose (FED) accrued on
the way, `X_FED = ∑ Δt / t_I`, by a dynamic programme on the time-expanded graph; a
dose-constrained variant (reach an exit while the dose stays below `X_crit`); and the claim
that waiting ("holdover") arcs are not needed under a monotone fire. Studio does not
implement this yet: the theorems below are the specification it must meet.

## Model

Time is counted in whole steps `0, 1, …, H` (step length `Δt`, a modelling choice of the
caller). A network `N : Net V E` has
* links `e : E` (a finite type) from `tail e` to `head e`, taking `tau e ≥ 1` steps;
* `dose e t`: the FED accrued by moving along `e` departing at step `t` (and arriving at
  `t + tau e`), a value in `WithTop ℝ`; `⊤` means the move is not allowed (link closed);
* `wait v t`: the FED accrued by waiting at `v` from step `t` to `t + 1` (holdover);
* `isExit v`: exits are absorbing, a schedule stops at the first exit it reaches and accrues
  nothing more.

A schedule from `(v, t)` is a list of actions `wait` / `move e`, that is a walk in the
time-expanded graph; it is `Valid` for the horizon `H` when it is a walk and ends at an exit
by step `H`. Its dose (`cost`) is the sum of the doses of its actions: the FED is additive
over consecutive intervals (`Tenability.T1_fedSum_additive`, ISO 13571 assumes no recovery).
The flag `hold` says whether waiting is allowed (`true`) or not (`false`, the DP without
holdover arcs). Doses are only assumed non-negative where a theorem says so.

## Results

1. *Exactness* (`dp_exact`, `dp_isLeast`, `val_eq_top_iff`). The backward induction `val`
   (`val_horizon`: `V_H(v) = 0` at an exit, `⊤` elsewhere; `val_bellman`:
   `V_t(v) = min (w_v(t) + V_{t+1}(v)) (min_e d_e(t) + V_{t+τ_e}(head e))`) is the least dose
   of a valid schedule, attained by the DP's own argmin schedule (`Greedy`); it is `⊤`
   exactly when every schedule that reaches an exit by `H` crosses a closed link.
2. *Dose constraint* (`safe_iff`, `safe_iff_le`, `greedy_safe`, `safe_prefix`,
   `earliest_safe_arrival`). A schedule with dose `< θ` exists iff `V_t(v) < θ`; the argmin
   schedule is one; every prefix of it stays below `θ` when doses are non-negative; and the
   earliest arrival under the dose limit is the least horizon whose DP value is below `θ`.
3. *Holdover* (`no_holdover`, `exists_optimal_nowait`, `optimal_never_waits`,
   `oneLink_no_holdover_iff`, `relay_counterexample`). The exact hypothesis is: every
   **move** dose `d_e(t)` is non-decreasing in the departure step `t` (with `⊤` above every
   number, so a link that has closed stays closed), and every wait dose is `≥ 0`. Nothing
   else is needed: wait doses need not be monotone, and no condition links them to move
   doses. Under it the DP without holdover arcs has the same value, some optimal schedule
   never waits, and if every wait dose is `> 0`, no optimal schedule waits. Monotonicity of
   the move doses cannot be weakened: on a single link with free waiting, "no holdover" holds
   iff the link's dose is monotone; and the local test "waiting at the tail of `e` and then
   taking `e` is never cheaper than taking `e` now" can hold for every link while holdover
   still strictly lowers the optimum, because a wait delays every later move.
   `no_holdover_monotone_fire` and `no_holdover_convective` discharge the hypothesis for
   edge doses `∫ r_e` with a non-decreasing rate, reusing `Tenability.T10_edge_dose_mono_entry`
   and `Tenability.T3_rate_monotone`.
4. *Shortest time* (`cost_eq_arrive`, `dp_earliest_arrival`, `earliest_escape_no_wait`,
   `dp_is_earliest_escape`, `bfs_layer_iff_dp`). With dose equal to time (`d_e(t) = τ_e` on
   open links, `w = 1`) the DP value is the earliest arrival at an exit minus the start step;
   with permanent closures (a closure window, or the earliest-escape burn check
   `t < ign(head)`) waiting does not help, so the DP without holdover arcs is an
   earliest-arrival search with general traversal times. With unit links and the burn check
   it returns exactly the layer of the exit in the earliest-escape BFS of
   `GraphNumerics.L7_bfs_layer_eq_earliest`.

Arithmetic is over `WithTop ℝ`; the theorems say nothing about floating-point rounding.
-/

namespace FireEgress.MinDose

open Finset

/-- A time-expanded egress network. Links `e` go from `tail e` to `head e` in `tau e ≥ 1`
steps; `dose e t` is the FED of moving along `e` departing at step `t` (`⊤`: not allowed);
`wait v t` is the FED of waiting at `v` from step `t` to `t + 1`; `isExit` marks the exits. -/
structure Net (V E : Type*) where
  tail : E → V
  head : E → V
  tau : E → ℕ
  tau_pos : ∀ e, 1 ≤ tau e
  dose : E → ℕ → WithTop ℝ
  wait : V → ℕ → WithTop ℝ
  isExit : V → Bool

/-- One action of a schedule: wait one step where you are, or move along a link. -/
inductive Act (E : Type*) where
  | wait : Act E
  | move : E → Act E

variable {V E : Type*} (N : Net V E)

/-- `Valid hold H t v s`: the schedule `s`, started at node `v` at step `t`, is a walk in the
time-expanded graph (waits only if `hold`; each move along a link out of the current node)
that never continues from an exit and ends at an exit by step `H`. -/
def Valid (hold : Bool) (H : ℕ) : ℕ → V → List (Act E) → Prop
  | t, v, [] => N.isExit v = true ∧ t ≤ H
  | t, v, Act.wait :: s => N.isExit v = false ∧ hold = true ∧ t + 1 ≤ H ∧ Valid hold H (t + 1) v s
  | t, v, Act.move e :: s => N.isExit v = false ∧ N.tail e = v ∧ t + N.tau e ≤ H ∧
      Valid hold H (t + N.tau e) (N.head e) s

/-- The total dose of a schedule started at `v` at step `t`: the sum of its wait and move
doses, each read at the step the action starts (section 5.3 of the review: hazard indexed by
the evacuee's own clock). -/
noncomputable def cost : ℕ → V → List (Act E) → WithTop ℝ
  | _, _, [] => 0
  | t, v, Act.wait :: s => N.wait v t + cost (t + 1) v s
  | t, _, Act.move e :: s => N.dose e t + cost (t + N.tau e) (N.head e) s

/-- The step at which a schedule started at `v` at step `t` ends. -/
def arrive : ℕ → V → List (Act E) → ℕ
  | t, _, [] => t
  | t, v, Act.wait :: s => arrive (t + 1) v s
  | t, _, Act.move e :: s => arrive (t + N.tau e) (N.head e) s

/-- Delete every wait from a schedule, keeping the moves in order (each then departs as soon
as the previous one arrives). -/
def strip : List (Act E) → List (Act E)
  | [] => []
  | Act.wait :: s => strip s
  | Act.move e :: s => Act.move e :: strip s

/-! ## Facts about schedules -/

/-- With non-negative wait and move doses every schedule has non-negative dose. -/
theorem cost_nonneg (hd : ∀ e t, 0 ≤ N.dose e t) (hw : ∀ v t, 0 ≤ N.wait v t) :
    ∀ (s : List (Act E)) (t : ℕ) (v : V), 0 ≤ cost N t v s
  | [], _, _ => by simp [cost]
  | Act.wait :: s, t, v => add_nonneg (hw v t) (cost_nonneg hd hw s (t + 1) v)
  | Act.move e :: s, t, _ => add_nonneg (hd e t) (cost_nonneg hd hw s (t + N.tau e) (N.head e))

/-- T17, "no Pareto labels". With non-negative doses the dose of every prefix of a schedule
(the FED accumulated so far, `Tenability.T1_fedSum_monotone`) is at most the dose of the whole
schedule. So for Studio a route whose final dose is below `X_crit` is below `X_crit` at every
step on the way, and a dose-constrained search may prune on the running dose. -/
theorem cost_take_le (hd : ∀ e t, 0 ≤ N.dose e t) (hw : ∀ v t, 0 ≤ N.wait v t) :
    ∀ (s : List (Act E)) (k t : ℕ) (v : V), cost N t v (s.take k) ≤ cost N t v s
  | [], _, _, _ => by simp
  | a :: s, 0, t, v => by
    rw [List.take_zero, show cost N t v [] = 0 from rfl]
    exact cost_nonneg N hd hw (a :: s) t v
  | Act.wait :: s, k + 1, t, v => by
    simp only [List.take_succ_cons, cost]
    exact add_le_add le_rfl (cost_take_le hd hw s k (t + 1) v)
  | Act.move e :: s, k + 1, t, _ => by
    simp only [List.take_succ_cons, cost]
    exact add_le_add le_rfl (cost_take_le hd hw s k (t + N.tau e) (N.head e))

/-- A schedule that reaches an exit by `H` also does so by any later horizon `H'`. -/
theorem valid_mono_horizon {hold : Bool} {H H' : ℕ} (hH : H ≤ H') :
    ∀ (s : List (Act E)) (t : ℕ) (v : V), Valid N hold H t v s → Valid N hold H' t v s
  | [], _, _, ⟨hx, ht⟩ => ⟨hx, by omega⟩
  | Act.wait :: s, t, v, ⟨hx, hh, ht, hs⟩ =>
    ⟨hx, hh, by omega, valid_mono_horizon hH s (t + 1) v hs⟩
  | Act.move e :: s, t, _, ⟨hx, he, ht, hs⟩ =>
    ⟨hx, he, by omega, valid_mono_horizon hH s (t + N.tau e) (N.head e) hs⟩

/-- A schedule never ends before it starts. -/
theorem le_arrive : ∀ (s : List (Act E)) (t : ℕ) (v : V), t ≤ arrive N t v s
  | [], _, _ => le_rfl
  | Act.wait :: s, t, v => (Nat.le_succ t).trans (le_arrive s (t + 1) v)
  | Act.move e :: s, t, _ => (Nat.le_add_right t (N.tau e)).trans (le_arrive s _ _)

/-- A valid schedule ends by the horizon. -/
theorem arrive_le {hold : Bool} {H : ℕ} :
    ∀ (s : List (Act E)) (t : ℕ) (v : V), Valid N hold H t v s → arrive N t v s ≤ H
  | [], _, _, ⟨_, ht⟩ => ht
  | Act.wait :: s, t, v, ⟨_, _, _, hs⟩ => arrive_le s (t + 1) v hs
  | Act.move e :: s, t, _, ⟨_, _, _, hs⟩ => arrive_le s (t + N.tau e) (N.head e) hs

/-- A valid schedule is valid with its own arrival step as the horizon. -/
theorem valid_arrive {hold : Bool} {H : ℕ} :
    ∀ (s : List (Act E)) (t : ℕ) (v : V), Valid N hold H t v s →
      Valid N hold (arrive N t v s) t v s
  | [], _, _, ⟨hx, _⟩ => ⟨hx, le_rfl⟩
  | Act.wait :: s, t, v, ⟨hx, hh, _, hs⟩ =>
    ⟨hx, hh, le_arrive N s (t + 1) v, valid_arrive s (t + 1) v hs⟩
  | Act.move e :: s, t, _, ⟨hx, he, _, hs⟩ =>
    ⟨hx, he, le_arrive N s (t + N.tau e) (N.head e), valid_arrive s _ _ hs⟩

/-- A schedule valid without holdover contains no wait. -/
theorem valid_false_nowait {H : ℕ} :
    ∀ (s : List (Act E)) (t : ℕ) (v : V), Valid N false H t v s → Act.wait ∉ s
  | [], _, _, _ => by simp
  | Act.wait :: _, _, _, ⟨_, hh, _⟩ => absurd hh Bool.false_ne_true
  | Act.move e :: s, t, _, ⟨_, _, _, hs⟩ => by
    simpa using valid_false_nowait s (t + N.tau e) (N.head e) hs

/-- A schedule valid without holdover is valid with holdover allowed. -/
theorem valid_of_false {hold : Bool} {H : ℕ} :
    ∀ (s : List (Act E)) (t : ℕ) (v : V), Valid N false H t v s → Valid N hold H t v s
  | [], _, _, h => h
  | Act.wait :: _, _, _, ⟨_, hh, _⟩ => absurd hh Bool.false_ne_true
  | Act.move e :: s, t, _, ⟨hx, he, ht, hs⟩ =>
    ⟨hx, he, ht, valid_of_false s (t + N.tau e) (N.head e) hs⟩

omit N in
/-- Stripping removes every wait. -/
theorem wait_not_mem_strip : ∀ s : List (Act E), Act.wait ∉ strip s
  | [] => by simp [strip]
  | Act.wait :: s => by simpa [strip] using wait_not_mem_strip s
  | Act.move e :: s => by simpa [strip] using wait_not_mem_strip s

/-- T17, no holdover (the exchange argument). If every move dose is non-decreasing in its
departure step and every wait dose is non-negative, deleting the waits from a valid schedule,
and even starting it earlier, gives a valid schedule without waits whose dose is no larger:
each move departs no later than before (so costs no more), and each deleted wait costs
`≥ 0`. For Studio: under a fire that only grows, an evacuee never lowers their dose by
pausing. -/
theorem strip_le (hmono : ∀ e, Monotone (N.dose e)) (hw : ∀ v t, 0 ≤ N.wait v t)
    (hold : Bool) (H : ℕ) :
    ∀ (s : List (Act E)) (t t' : ℕ) (v : V), Valid N hold H t v s → t' ≤ t →
      Valid N false H t' v (strip s) ∧ cost N t' v (strip s) ≤ cost N t v s
  | [], t, t', v, ⟨hx, ht⟩, htt => ⟨⟨hx, by omega⟩, le_rfl⟩
  | Act.wait :: s, t, t', v, ⟨_, _, _, hs⟩, htt => by
    obtain ⟨h1, h2⟩ := strip_le hmono hw hold H s (t + 1) t' v hs (by omega)
    refine ⟨h1, ?_⟩
    simp only [strip, cost]
    calc cost N t' v (strip s) ≤ cost N (t + 1) v s := h2
      _ = 0 + cost N (t + 1) v s := (zero_add _).symm
      _ ≤ N.wait v t + cost N (t + 1) v s := add_le_add (hw v t) le_rfl
  | Act.move e :: s, t, t', v, ⟨hx, he, ht, hs⟩, htt => by
    obtain ⟨h1, h2⟩ :=
      strip_le hmono hw hold H s (t + N.tau e) (t' + N.tau e) (N.head e) hs (by omega)
    refine ⟨⟨hx, he, by omega, h1⟩, ?_⟩
    simp only [strip, cost]
    exact add_le_add (hmono e htt) h2

/-- In `WithTop ℝ`, adding a positive amount to a finite total strictly increases it. -/
theorem lt_add_of_pos_left' {a b : WithTop ℝ} (ha : 0 < a) (hab : a + b ≠ ⊤) : b < a + b := by
  have ha' : a ≠ ⊤ := (WithTop.add_ne_top.1 hab).1
  have hb' : b ≠ ⊤ := (WithTop.add_ne_top.1 hab).2
  lift a to ℝ using ha'
  lift b to ℝ using hb'
  have h0 : (0 : ℝ) < a := by exact_mod_cast ha
  exact_mod_cast (by linarith : b < a + b)

/-- In `WithTop ℝ`, `a ≤ b` and `c < d` give `a + c < b + d` when `b + d` is finite. -/
theorem add_lt_add_of_le_of_lt' {a b c d : WithTop ℝ} (hab : a ≤ b) (hcd : c < d)
    (hbd : b + d ≠ ⊤) : a + c < b + d := by
  have hb : b ≠ ⊤ := (WithTop.add_ne_top.1 hbd).1
  have hd : d ≠ ⊤ := (WithTop.add_ne_top.1 hbd).2
  have ha : a ≠ ⊤ := ne_top_of_le_ne_top hb hab
  have hc : c ≠ ⊤ := hcd.ne_top
  lift a to ℝ using ha
  lift b to ℝ using hb
  lift c to ℝ using hc
  lift d to ℝ using hd
  have h1 : a ≤ b := by exact_mod_cast hab
  have h2 : c < d := by exact_mod_cast hcd
  exact_mod_cast (by linarith : a + c < b + d)

/-- T17, strict form. If every wait dose is positive (e.g. `Δt / t_I` with a finite tolerance
time), deleting the waits from a schedule of finite dose that does wait strictly lowers its
dose. -/
theorem strip_lt (hmono : ∀ e, Monotone (N.dose e)) (hw : ∀ v t, 0 < N.wait v t)
    (hold : Bool) (H : ℕ) :
    ∀ (s : List (Act E)) (t t' : ℕ) (v : V), Valid N hold H t v s → t' ≤ t →
      Act.wait ∈ s → cost N t v s ≠ ⊤ → cost N t' v (strip s) < cost N t v s
  | [], _, _, _, _, _, hmem, _ => absurd hmem (by simp)
  | Act.wait :: s, t, t', v, ⟨_, _, _, hs⟩, htt, _, hfin => by
    have h2 := (strip_le N hmono (fun v t => (hw v t).le) hold H s (t + 1) t' v hs
      (by omega)).2
    simp only [strip, cost] at hfin ⊢
    exact lt_of_le_of_lt h2 (lt_add_of_pos_left' (hw v t) hfin)
  | Act.move e :: s, t, t', v, ⟨_, _, _, hs⟩, htt, hmem, hfin => by
    have hmem' : Act.wait ∈ s := by simpa using hmem
    simp only [strip, cost] at hfin ⊢
    have ih := strip_lt hmono hw hold H s (t + N.tau e) (t' + N.tau e) (N.head e) hs
      (by omega) hmem' (WithTop.add_ne_top.1 hfin).2
    exact add_lt_add_of_le_of_lt' (hmono e htt) ih hfin

/-! ## Dose equal to time -/

/-- Dose equal to time: an allowed move costs its traversal time `τ_e`, an allowed wait costs
one step, and a forbidden action costs `⊤`. -/
def TimeCosts : Prop :=
  (∀ e t, N.dose e t = ((N.tau e : ℝ) : WithTop ℝ) ∨ N.dose e t = ⊤) ∧
  (∀ v t, N.wait v t = ((1 : ℝ) : WithTop ℝ) ∨ N.wait v t = ⊤)

/-- T17 and earliest arrival. When dose equals time, a schedule of finite dose has dose equal
to its travel time, arrival step minus start step. -/
theorem cost_eq_arrive (hT : TimeCosts N) :
    ∀ (s : List (Act E)) (t : ℕ) (v : V), cost N t v s ≠ ⊤ →
      cost N t v s = (((arrive N t v s : ℝ) - t : ℝ) : WithTop ℝ)
  | [], t, v, _ => by simp [cost, arrive]
  | Act.wait :: s, t, v, h => by
    simp only [cost] at h ⊢
    have ih := cost_eq_arrive hT s (t + 1) v (WithTop.add_ne_top.1 h).2
    rcases hT.2 v t with h1 | h1
    · rw [h1, ih, ← WithTop.coe_add]
      simp only [arrive]
      congr 1
      push_cast
      ring
    · exact absurd h1 (WithTop.add_ne_top.1 h).1
  | Act.move e :: s, t, v, h => by
    simp only [cost] at h ⊢
    have ih := cost_eq_arrive hT s (t + N.tau e) (N.head e) (WithTop.add_ne_top.1 h).2
    rcases hT.1 e t with h1 | h1
    · rw [h1, ih, ← WithTop.coe_add]
      simp only [arrive]
      congr 1
      push_cast
      ring
    · exact absurd h1 (WithTop.add_ne_top.1 h).1

/-- A move dose equal to `τ` while the link is open and `⊤` once it is closed is monotone in
the departure step exactly when a closed link stays closed (`ok` is antitone): a monotone
fire. -/
theorem closure_dose_monotone (τ : ℕ) (ok : ℕ → Prop) [DecidablePred ok]
    (hok : ∀ t t', t ≤ t' → ok t' → ok t) :
    Monotone (fun t => if ok t then ((τ : ℝ) : WithTop ℝ) else ⊤) := by
  intro t t' h
  dsimp only
  by_cases h' : ok t'
  · rw [ite_eq_left (hok t t' h h'), ite_eq_left h']
  · rw [ite_eq_right h']
    exact le_top

/-! ## Doses from a monotone fire (reusing `Tenability`) -/

/-- The dose of an action of `τ` steps of length `Δ` (minutes) starting at step `k`, under a
dose rate `r` (min⁻¹): `∫_{kΔ}^{kΔ + τΔ} r`, as in the review's `c(e, k)` (T10, T17). -/
noncomputable def intDose (r : ℝ → ℝ) (Δ : ℝ) (τ k : ℕ) : ℝ :=
  ∫ s in (k : ℝ) * Δ..(k : ℝ) * Δ + (τ : ℝ) * Δ, r s

/-- T10 to T17. A non-decreasing dose rate (a fire that only grows) gives an edge dose that is
non-decreasing in the departure step. Reuses `Tenability.T10_edge_dose_mono_entry`. -/
theorem intDose_monotone (r : ℝ → ℝ) (hr : Monotone r) (Δ : ℝ) (hΔ : 0 ≤ Δ) (τ : ℕ) :
    Monotone (intDose r Δ τ) := by
  intro k k' hk
  exact Tenability.T10_edge_dose_mono_entry r hr ((τ : ℝ) * Δ) (by positivity)
    (mul_le_mul_of_nonneg_right (Nat.cast_le.mpr hk) hΔ)

/-- A non-negative dose rate gives a non-negative dose over any step. -/
theorem intDose_nonneg (r : ℝ → ℝ) (hr : ∀ s, 0 ≤ r s) (Δ : ℝ) (hΔ : 0 ≤ Δ) (τ k : ℕ) :
    0 ≤ intDose r Δ τ k :=
  intervalIntegral.integral_nonneg (by nlinarith [(Nat.cast_nonneg τ : (0 : ℝ) ≤ τ)])
    (fun s _ => hr s)

/-! ## The dynamic programme -/

variable [Fintype E] [DecidableEq V]

/-- The backward induction on the time-expanded graph: the least dose with which an evacuee
at `v` at step `t` can reach an exit by step `H`. At an exit it is `0`; elsewhere the better
of waiting one step (if `hold`) and the best move out of `v`; a move that would end after
`H` is not allowed. -/
noncomputable def val (hold : Bool) (H t : ℕ) (v : V) : WithTop ℝ :=
  if N.isExit v = true then (if t ≤ H then 0 else ⊤) else
    min (if _h : hold = true ∧ t + 1 ≤ H then N.wait v t + val hold H (t + 1) v else ⊤)
      (Finset.univ.inf fun e =>
        if _h : N.tail e = v ∧ t + N.tau e ≤ H then
          N.dose e t + val hold H (t + N.tau e) (N.head e)
        else ⊤)
termination_by H - t
decreasing_by
  all_goals first | omega | (have := N.tau_pos e; omega)

/-- The defining equation of `val`. -/
theorem val_eq (hold : Bool) (H t : ℕ) (v : V) :
    val N hold H t v =
      if N.isExit v = true then (if t ≤ H then 0 else ⊤) else
        min (if hold = true ∧ t + 1 ≤ H then N.wait v t + val N hold H (t + 1) v else ⊤)
          (Finset.univ.inf fun e =>
            if N.tail e = v ∧ t + N.tau e ≤ H then
              N.dose e t + val N hold H (t + N.tau e) (N.head e)
            else ⊤) := by
  rw [val]
  simp only [dite_eq_ite]

/-- Past the horizon nothing is reachable in time: `V_t(v) = ⊤` for `t > H`. -/
theorem val_gt (hold : Bool) {H t : ℕ} (v : V) (ht : H < t) : val N hold H t v = ⊤ := by
  rw [val_eq]
  by_cases hx : N.isExit v = true
  · rw [ite_eq_left hx, ite_eq_right (by omega)]
  · rw [ite_eq_right hx, ite_eq_right (fun h => absurd h.2 (by omega))]
    have hX : (Finset.univ.inf fun e => if N.tail e = v ∧ t + N.tau e ≤ H then
        N.dose e t + val N hold H (t + N.tau e) (N.head e) else ⊤) = ⊤ :=
      (Finset.inf_eq_top_iff _ _).2 (fun e _ => ite_eq_right (fun h => absurd h.2 (by omega)))
    rw [hX, min_self]

/-- At an exit, by the horizon, the remaining dose is `0` (exits are absorbing). -/
theorem val_exit (hold : Bool) {H t : ℕ} {v : V} (hx : N.isExit v = true) (ht : t ≤ H) :
    val N hold H t v = 0 := by
  rw [val_eq, ite_eq_left hx, ite_eq_left ht]

/-- T17, the DP's terminal condition: `V_H(v) = 0` if `v` is an exit and `⊤` otherwise. -/
theorem val_horizon (hold : Bool) (H : ℕ) (v : V) :
    val N hold H H v = if N.isExit v = true then 0 else ⊤ := by
  by_cases hx : N.isExit v = true
  · rw [ite_eq_left hx, val_exit N hold hx le_rfl]
  · rw [ite_eq_right hx, val_eq, ite_eq_right hx, ite_eq_right (fun h => absurd h.2 (by omega))]
    have hX : (Finset.univ.inf fun e => if N.tail e = v ∧ H + N.tau e ≤ H then
        N.dose e H + val N hold H (H + N.tau e) (N.head e) else ⊤) = ⊤ :=
      (Finset.inf_eq_top_iff _ _).2 (fun e _ =>
        ite_eq_right (fun h => absurd h.2 (by have := N.tau_pos e; omega)))
    rw [hX, min_self]

/-- T17, the DP's recursion (Bellman equation) at a non-exit node, for every step `t`:
`V_t(v) = min (w_v(t) + V_{t+1}(v)) (min_{e : tail e = v} d_e(t) + V_{t+τ_e}(head e))`, the
first term only when waiting is allowed. Moves past the horizon need no special case, since
`V = ⊤` there (`val_gt`). This is the recursion Studio's DP must implement. -/
theorem val_bellman (hold : Bool) (H t : ℕ) (v : V) (hx : N.isExit v = false) :
    val N hold H t v =
      min (if hold = true then N.wait v t + val N hold H (t + 1) v else ⊤)
        (Finset.univ.inf fun e =>
          if N.tail e = v then N.dose e t + val N hold H (t + N.tau e) (N.head e) else ⊤) := by
  rw [val_eq, ite_eq_right (by simp [hx])]
  congr 1
  · by_cases hh : hold = true
    · by_cases ht : t + 1 ≤ H
      · rw [ite_eq_left ⟨hh, ht⟩, ite_eq_left hh]
      · rw [ite_eq_right (fun h => ht h.2), ite_eq_left hh, val_gt N hold v (by omega), WithTop.add_top]
    · rw [ite_eq_right (fun h => hh h.1), ite_eq_right hh]
  · refine Finset.inf_congr rfl (fun e _ => ?_)
    by_cases he : N.tail e = v
    · by_cases ht : t + N.tau e ≤ H
      · rw [ite_eq_left ⟨he, ht⟩, ite_eq_left he]
      · rw [ite_eq_right (fun h => ht h.2), ite_eq_left he, val_gt N hold _ (by omega), WithTop.add_top]
    · rw [ite_eq_right (fun h => he h.1), ite_eq_right he]

/-- T17, lower bound: the DP value is at most the dose of every valid schedule. -/
theorem val_le_cost (hold : Bool) (H : ℕ) :
    ∀ (s : List (Act E)) (t : ℕ) (v : V), Valid N hold H t v s → val N hold H t v ≤ cost N t v s
  | [], t, v, ⟨hx, ht⟩ => by
    rw [val_exit N hold hx ht]
    simp [cost]
  | Act.wait :: s, t, v, ⟨hx, hh, _, hs⟩ => by
    have ih := val_le_cost hold H s (t + 1) v hs
    rw [val_bellman N hold H t v hx]
    refine min_le_of_left_le ?_
    rw [ite_eq_left hh]
    simp only [cost]
    exact add_le_add le_rfl ih
  | Act.move e :: s, t, v, ⟨hx, he, _, hs⟩ => by
    have ih := val_le_cost hold H s (t + N.tau e) (N.head e) hs
    rw [val_bellman N hold H t v hx]
    refine min_le_of_right_le ((Finset.inf_le (Finset.mem_univ e)).trans ?_)
    rw [ite_eq_left he]
    simp only [cost]
    exact add_le_add le_rfl ih

/-- `Greedy hold H t v s`: `s` is an argmin schedule of the DP, i.e. valid, and every action it
takes attains the minimum in the Bellman equation at the node and step where it is taken.
Studio's DP recovers its route by following these argmins. -/
def Greedy (hold : Bool) (H : ℕ) : ℕ → V → List (Act E) → Prop
  | t, v, [] => N.isExit v = true ∧ t ≤ H
  | t, v, Act.wait :: s => N.isExit v = false ∧ hold = true ∧ t + 1 ≤ H ∧
      val N hold H t v = N.wait v t + val N hold H (t + 1) v ∧ Greedy hold H (t + 1) v s
  | t, v, Act.move e :: s => N.isExit v = false ∧ N.tail e = v ∧ t + N.tau e ≤ H ∧
      val N hold H t v = N.dose e t + val N hold H (t + N.tau e) (N.head e) ∧
      Greedy hold H (t + N.tau e) (N.head e) s

/-- T17. Following the DP's argmins gives a valid schedule whose dose is the DP value. -/
theorem greedy_valid (hold : Bool) (H : ℕ) :
    ∀ (s : List (Act E)) (t : ℕ) (v : V), Greedy N hold H t v s →
      Valid N hold H t v s ∧ cost N t v s = val N hold H t v
  | [], t, v, ⟨hx, ht⟩ => ⟨⟨hx, ht⟩, by rw [val_exit N hold hx ht]; simp [cost]⟩
  | Act.wait :: s, t, v, ⟨hx, hh, ht, hv, hs⟩ => by
    obtain ⟨h1, h2⟩ := greedy_valid hold H s (t + 1) v hs
    refine ⟨⟨hx, hh, ht, h1⟩, ?_⟩
    simp only [cost]
    rw [h2, hv]
  | Act.move e :: s, t, v, ⟨hx, he, ht, hv, hs⟩ => by
    obtain ⟨h1, h2⟩ := greedy_valid hold H s (t + N.tau e) (N.head e) hs
    refine ⟨⟨hx, he, ht, h1⟩, ?_⟩
    simp only [cost]
    rw [h2, hv]

/-- T17. Whenever the DP value is finite, an argmin schedule exists (by induction on the
remaining time `H - t`). -/
theorem exists_greedy (hold : Bool) (H : ℕ) :
    ∀ (k t : ℕ) (v : V), H - t = k → val N hold H t v ≠ ⊤ → ∃ s, Greedy N hold H t v s := by
  intro k
  induction k using Nat.strong_induction_on with
  | _ k ih =>
  intro t v hk hne
  by_cases hx : N.isExit v = true
  · refine ⟨[], hx, ?_⟩
    by_contra ht
    exact hne (val_gt N hold v (by omega))
  have hx' : N.isExit v = false := by simpa using hx
  have hb := val_bellman N hold H t v hx'
  rcases min_choice (if hold = true then N.wait v t + val N hold H (t + 1) v else ⊤)
      (Finset.univ.inf fun e => if N.tail e = v then
        N.dose e t + val N hold H (t + N.tau e) (N.head e) else ⊤) with hm | hm
  · rw [hm] at hb
    rw [hb] at hne
    by_cases hh : hold = true
    · rw [ite_eq_left hh] at hb
      rw [ite_eq_left hh] at hne
      have h1 : val N hold H (t + 1) v ≠ ⊤ := (WithTop.add_ne_top.1 hne).2
      have ht : t + 1 ≤ H := by
        by_contra ht
        exact h1 (val_gt N hold v (by omega))
      obtain ⟨s, hs⟩ := ih (H - (t + 1)) (by omega) (t + 1) v rfl h1
      exact ⟨Act.wait :: s, hx', hh, ht, hb, hs⟩
    · rw [ite_eq_right hh] at hne
      exact absurd rfl hne
  · rw [hm] at hb
    rw [hb] at hne
    rcases (Finset.univ : Finset E).eq_empty_or_nonempty with h0 | h0
    · rw [h0, Finset.inf_empty] at hne
      exact absurd rfl hne
    obtain ⟨e, -, he⟩ := Finset.exists_mem_eq_inf Finset.univ h0
      (fun e => if N.tail e = v then N.dose e t + val N hold H (t + N.tau e) (N.head e) else ⊤)
    rw [he] at hb
    rw [he] at hne
    by_cases hte : N.tail e = v
    · rw [ite_eq_left hte] at hb
      rw [ite_eq_left hte] at hne
      have h1 : val N hold H (t + N.tau e) (N.head e) ≠ ⊤ := (WithTop.add_ne_top.1 hne).2
      have ht : t + N.tau e ≤ H := by
        by_contra ht
        exact h1 (val_gt N hold _ (by omega))
      have := N.tau_pos e
      obtain ⟨s, hs⟩ := ih (H - (t + N.tau e)) (by omega) (t + N.tau e) (N.head e) rfl h1
      exact ⟨Act.move e :: s, hx', hte, ht, hb, hs⟩
    · rw [ite_eq_right hte] at hne
      exact absurd rfl hne

/-- T17, item 1: **exactness of the DP**. The backward induction value `V_t(v)` is at most the
dose of every schedule from `(v, t)` that reaches an exit by `H`, and, when finite, is the
dose of one of them. For Studio: the DP returns the minimum-FED route, not an
approximation. -/
theorem dp_exact (hold : Bool) (H t : ℕ) (v : V) :
    (∀ s, Valid N hold H t v s → val N hold H t v ≤ cost N t v s) ∧
    (val N hold H t v ≠ ⊤ → ∃ s, Valid N hold H t v s ∧ cost N t v s = val N hold H t v) :=
  ⟨fun s hs => val_le_cost N hold H s t v hs, fun hne => by
    obtain ⟨s, hs⟩ := exists_greedy N hold H _ t v rfl hne
    exact ⟨s, greedy_valid N hold H s t v hs⟩⟩

/-- T17, item 1, as a minimum: if any schedule from `(v, t)` reaches an exit by `H`, the DP
value is the least element of the set of their doses. -/
theorem dp_isLeast (hold : Bool) (H t : ℕ) (v : V) (hex : ∃ s, Valid N hold H t v s) :
    IsLeast {c | ∃ s, Valid N hold H t v s ∧ cost N t v s = c} (val N hold H t v) := by
  refine ⟨?_, fun c ⟨s, hs, hc⟩ => hc ▸ val_le_cost N hold H s t v hs⟩
  by_cases h : val N hold H t v = ⊤
  · obtain ⟨s, hs⟩ := hex
    have := val_le_cost N hold H s t v hs
    rw [h, top_le_iff] at this
    exact ⟨s, hs, by rw [this, h]⟩
  · exact (dp_exact N hold H t v).2 h

/-- T17. The DP value is `⊤` exactly when every schedule reaching an exit by `H` has dose
`⊤`, i.e. crosses a closed link: Studio reports "no route" exactly then. -/
theorem val_eq_top_iff (hold : Bool) (H t : ℕ) (v : V) :
    val N hold H t v = ⊤ ↔ ∀ s, Valid N hold H t v s → cost N t v s = ⊤ := by
  constructor
  · intro h s hs
    have := val_le_cost N hold H s t v hs
    rwa [h, top_le_iff] at this
  · intro h
    by_contra hne
    obtain ⟨s, hs, hc⟩ := (dp_exact N hold H t v).2 hne
    exact hne (hc ▸ h s hs)

/-- With non-negative doses the DP value is non-negative. -/
theorem val_nonneg (hd : ∀ e t, 0 ≤ N.dose e t) (hw : ∀ v t, 0 ≤ N.wait v t) (hold : Bool)
    (H t : ℕ) (v : V) : 0 ≤ val N hold H t v := by
  by_cases h : val N hold H t v = ⊤
  · rw [h]
    exact le_top
  · obtain ⟨s, -, hc⟩ := (dp_exact N hold H t v).2 h
    rw [← hc]
    exact cost_nonneg N hd hw s t v

/-! ## Dose-constrained routes -/

/-- T17, item 2: **dose constraint**. A schedule from `(v, t)` that reaches an exit by `H` with
dose below the threshold `θ` (e.g. `X_crit = 0.3` or `1`; below, because
`Tenability.T12_first_passage` counts reaching `X_crit` as incapacitation) exists iff
`V_t(v) < θ`. Studio can decide feasibility from one DP value. -/
theorem safe_iff (hold : Bool) (H t : ℕ) (v : V) (θ : ℝ) :
    (∃ s, Valid N hold H t v s ∧ cost N t v s < (θ : WithTop ℝ)) ↔
      val N hold H t v < (θ : WithTop ℝ) := by
  constructor
  · rintro ⟨s, hs, hc⟩
    exact lt_of_le_of_lt (val_le_cost N hold H s t v hs) hc
  · intro h
    obtain ⟨s, hs, hc⟩ := (dp_exact N hold H t v).2 (ne_top_of_lt h)
    exact ⟨s, hs, by rw [hc]; exact h⟩

/-- T17, item 2, with ISO's "exceeds" reading (dose `≤ X_crit` still tenable, the review's
`F*(exit, k) ≤ X_crit`): a schedule with dose `≤ θ` exists iff `V_t(v) ≤ θ`. -/
theorem safe_iff_le (hold : Bool) (H t : ℕ) (v : V) (θ : ℝ) :
    (∃ s, Valid N hold H t v s ∧ cost N t v s ≤ (θ : WithTop ℝ)) ↔
      val N hold H t v ≤ (θ : WithTop ℝ) := by
  constructor
  · rintro ⟨s, hs, hc⟩
    exact le_trans (val_le_cost N hold H s t v hs) hc
  · intro h
    obtain ⟨s, hs, hc⟩ := (dp_exact N hold H t v).2 (ne_top_of_le_ne_top WithTop.coe_ne_top h)
    exact ⟨s, hs, by rw [hc]; exact h⟩

/-- T17, item 2: **the DP's argmin schedule is safe** whenever any schedule is: if
`V_t(v) < θ`, every argmin schedule is valid with dose `V_t(v) < θ`. -/
theorem greedy_safe (hold : Bool) (H t : ℕ) (v : V) (θ : ℝ) {s : List (Act E)}
    (hs : Greedy N hold H t v s) (h : val N hold H t v < (θ : WithTop ℝ)) :
    Valid N hold H t v s ∧ cost N t v s < (θ : WithTop ℝ) := by
  obtain ⟨h1, h2⟩ := greedy_valid N hold H s t v hs
  exact ⟨h1, by rw [h2]; exact h⟩

/-- T17, item 2: if `V_t(v) < θ`, an argmin schedule exists, and it is a valid schedule with
dose below `θ`. -/
theorem exists_greedy_safe (hold : Bool) (H t : ℕ) (v : V) (θ : ℝ)
    (h : val N hold H t v < (θ : WithTop ℝ)) :
    ∃ s, Greedy N hold H t v s ∧ Valid N hold H t v s ∧ cost N t v s < (θ : WithTop ℝ) := by
  obtain ⟨s, hs⟩ := exists_greedy N hold H _ t v rfl (ne_top_of_lt h)
  exact ⟨s, hs, greedy_safe N hold H t v θ hs h⟩

/-- T17, item 2: along the argmin schedule of a feasible instance, the accumulated dose stays
below `θ` at every step (non-negative doses): the evacuee is never incapacitated on the way. -/
theorem safe_prefix (hd : ∀ e t, 0 ≤ N.dose e t) (hw : ∀ v t, 0 ≤ N.wait v t) (hold : Bool)
    (H t : ℕ) (v : V) (θ : ℝ) {s : List (Act E)} (hs : Greedy N hold H t v s)
    (h : val N hold H t v < (θ : WithTop ℝ)) (k : ℕ) :
    cost N t v (s.take k) < (θ : WithTop ℝ) :=
  lt_of_le_of_lt (cost_take_le N hd hw s k t v) (greedy_safe N hold H t v θ hs h).2

/-- A longer horizon never raises the least dose. -/
theorem val_antitone_horizon (hold : Bool) {H H' : ℕ} (hH : H ≤ H') (t : ℕ) (v : V) :
    val N hold H' t v ≤ val N hold H t v := by
  by_cases h : val N hold H t v = ⊤
  · rw [h]
    exact le_top
  · obtain ⟨s, hs, hc⟩ := (dp_exact N hold H t v).2 h
    rw [← hc]
    exact val_le_cost N hold H' s t v (valid_mono_horizon N hH s t v hs)

/-- T17, formulation 3 of the review (earliest arrival under a cumulative dose limit). If some
horizon allows a dose below `θ`, the least such horizon `K` is the earliest arrival step of a
schedule with dose below `θ`: one arrives exactly at `K`, and none arrives earlier. Studio can
compute it by scanning horizons, with no Pareto labels. -/
theorem earliest_safe_arrival (hold : Bool) (t : ℕ) (v : V) (θ : ℝ)
    (hex : ∃ H, val N hold H t v < (θ : WithTop ℝ)) :
    (∃ s, Valid N hold (Nat.find hex) t v s ∧ cost N t v s < (θ : WithTop ℝ) ∧
        arrive N t v s = Nat.find hex) ∧
      ∀ H s, Valid N hold H t v s → cost N t v s < (θ : WithTop ℝ) →
        Nat.find hex ≤ arrive N t v s := by
  have hlow : ∀ H s, Valid N hold H t v s → cost N t v s < (θ : WithTop ℝ) →
      Nat.find hex ≤ arrive N t v s := by
    intro H s hs hc
    exact Nat.find_min' hex ((safe_iff N hold _ t v θ).1 ⟨s, valid_arrive N s t v hs, hc⟩)
  refine ⟨?_, hlow⟩
  obtain ⟨s, hs, hc⟩ := (safe_iff N hold _ t v θ).2 (Nat.find_spec hex)
  exact ⟨s, hs, hc, le_antisymm (arrive_le N s t v hs) (hlow _ s hs hc)⟩

/-! ## Holdover -/

/-- Allowing waits never raises the least dose. -/
theorem val_le_val_false (hold : Bool) (H t : ℕ) (v : V) :
    val N hold H t v ≤ val N false H t v := by
  by_cases h : val N false H t v = ⊤
  · rw [h]
    exact le_top
  · obtain ⟨s, hs, hc⟩ := (dp_exact N false H t v).2 h
    rw [← hc]
    exact val_le_cost N hold H s t v (valid_of_false N s t v hs)

/-- T17, item 3: **no holdover under a monotone fire**. If every move dose `d_e(t)` is
non-decreasing in the departure step `t` and every wait dose is `≥ 0`, the DP without holdover
arcs has the same value as the DP with them, at every node, step and horizon. For Studio: the
holdover arcs (and the stay term of the recursion) can be dropped. -/
theorem no_holdover (hmono : ∀ e, Monotone (N.dose e)) (hw : ∀ v t, 0 ≤ N.wait v t)
    (H t : ℕ) (v : V) : val N false H t v = val N true H t v := by
  refine le_antisymm ?_ (val_le_val_false N true H t v)
  by_cases h : val N true H t v = ⊤
  · rw [h]
    exact le_top
  · obtain ⟨s, hs, hc⟩ := (dp_exact N true H t v).2 h
    obtain ⟨h1, h2⟩ := strip_le N hmono hw true H s t t v hs le_rfl
    rw [← hc]
    exact (val_le_cost N false H (strip s) t v h1).trans h2

/-- T17, item 3: under the same hypothesis, whenever a finite-dose route exists, **some optimal
schedule never waits** (it is the DP's argmin schedule without holdover arcs). -/
theorem exists_optimal_nowait (hmono : ∀ e, Monotone (N.dose e)) (hw : ∀ v t, 0 ≤ N.wait v t)
    (H t : ℕ) (v : V) (hne : val N true H t v ≠ ⊤) :
    ∃ s, Valid N true H t v s ∧ Act.wait ∉ s ∧ cost N t v s = val N true H t v := by
  rw [← no_holdover N hmono hw H t v] at hne ⊢
  obtain ⟨s, hs, hc⟩ := (dp_exact N false H t v).2 hne
  exact ⟨s, valid_of_false N s t v hs, valid_false_nowait N s t v hs, hc⟩

/-- T17, item 3, strict form. If moreover every wait dose is positive (holdover cost
`Δt / t_I` with a finite tolerance time `t_I`), then **no** optimal schedule of finite dose
waits. -/
theorem optimal_never_waits (hmono : ∀ e, Monotone (N.dose e)) (hw : ∀ v t, 0 < N.wait v t)
    {H t : ℕ} {v : V} {s : List (Act E)} (hs : Valid N true H t v s)
    (hc : cost N t v s = val N true H t v) (hne : val N true H t v ≠ ⊤) : Act.wait ∉ s := by
  intro hmem
  have hw' : ∀ v t, 0 ≤ N.wait v t := fun v t => (hw v t).le
  have hfin : cost N t v s ≠ ⊤ := by rw [hc]; exact hne
  have h1 := strip_lt N hmono hw true H s t t v hs le_rfl hmem hfin
  have h2 := (strip_le N hmono hw' true H s t t v hs le_rfl).1
  have h3 := val_le_cost N false H (strip s) t v h2
  rw [no_holdover N hmono hw' H t v, ← hc] at h3
  exact absurd (lt_of_le_of_lt h3 h1) (lt_irrefl _)

/-- T10 + T17 for Studio. If every link's dose is the integral of a non-decreasing dose rate
`r_e` (min⁻¹) over the traversal window, `d_e(k) = ∫_{kΔ}^{kΔ+τ_eΔ} r_e` with `Δ ≥ 0` the step
length in minutes, and every wait dose is non-negative, the DP needs no holdover arcs. This is
the review's "monotone fire" claim, with its hypothesis made exact. Reuses
`Tenability.T10_edge_dose_mono_entry`. -/
theorem no_holdover_monotone_fire (r : E → ℝ → ℝ) (hr : ∀ e, Monotone (r e)) (Δ : ℝ)
    (hΔ : 0 ≤ Δ) (hd : ∀ e k, N.dose e k = ((intDose (r e) Δ (N.tau e) k : ℝ) : WithTop ℝ))
    (hw : ∀ v k, 0 ≤ N.wait v k) (H t : ℕ) (v : V) :
    val N false H t v = val N true H t v := by
  refine no_holdover N (fun e k k' hk => ?_) hw H t v
  rw [hd, hd]
  exact WithTop.coe_le_coe.2 (intDose_monotone (r e) (hr e) Δ hΔ (N.tau e) hk)

/-- T3 + T10 + T17 for Studio: convective heat. If the gas temperature `T_e(s)` (degC, `≥ 0`)
on every link is non-decreasing in time, the link dose is the convective FED
`∫ ds / t_I,conv(T_e(s))` over the traversal window (ISO 13571, `t_I = 5e7 T^-3.4` min), and
waiting at a node accrues the same FED under its own non-negative temperature history, then
the DP needs no holdover arcs. The node temperatures need not be monotone. Reuses
`Tenability.T3_rate_monotone`. -/
theorem no_holdover_convective (T : E → ℝ → ℝ) (hT : ∀ e, Monotone (T e))
    (hT0 : ∀ e s, 0 ≤ T e s) (Tn : V → ℝ → ℝ) (hTn0 : ∀ v s, 0 ≤ Tn v s) (Δ : ℝ) (hΔ : 0 ≤ Δ)
    (hd : ∀ e k, N.dose e k =
      ((intDose (fun s => 1 / Tenability.tConv (T e s)) Δ (N.tau e) k : ℝ) : WithTop ℝ))
    (hw : ∀ v k, N.wait v k =
      ((intDose (fun s => 1 / Tenability.tConv (Tn v s)) Δ 1 k : ℝ) : WithTop ℝ))
    (H t : ℕ) (v : V) : val N false H t v = val N true H t v := by
  refine no_holdover_monotone_fire N _
    (fun e => (Tenability.T3_rate_monotone (T e) (hT e) (hT0 e)).1) Δ hΔ hd (fun v k => ?_) H t v
  rw [hw]
  refine WithTop.coe_nonneg.2 (intDose_nonneg _ (fun s => ?_) Δ hΔ 1 k)
  unfold Tenability.tConv
  exact one_div_nonneg.2 (mul_nonneg (by norm_num) (Real.rpow_nonneg (hTn0 v s) _))

/-! ## Dose equal to time: the DP is the earliest-arrival search -/

/-- T17, item 4: **relation to shortest time**. With dose equal to time (`TimeCosts`), when an
exit is reachable through open links by `H`, the DP value is `m - t` where `m` is the earliest
arrival step at an exit over all schedules through open links. So the min-dose DP generalises
the earliest-arrival (earliest-escape) search: the latter is the special dose `d_e(t) = τ_e`. -/
theorem dp_earliest_arrival (hT : TimeCosts N) (hold : Bool) (H t : ℕ) (v : V)
    (hne : val N hold H t v ≠ ⊤) :
    ∃ m : ℕ, val N hold H t v = (((m : ℝ) - t : ℝ) : WithTop ℝ) ∧
      IsLeast {a | ∃ s, Valid N hold H t v s ∧ cost N t v s ≠ ⊤ ∧ arrive N t v s = a} m := by
  obtain ⟨s, hs, hc⟩ := (dp_exact N hold H t v).2 hne
  have hcs : cost N t v s ≠ ⊤ := by rw [hc]; exact hne
  refine ⟨arrive N t v s, by rw [← hc, cost_eq_arrive N hT s t v hcs], ⟨s, hs, hcs, rfl⟩, ?_⟩
  rintro a ⟨s', hs', hc', rfl⟩
  have h := val_le_cost N hold H s' t v hs'
  rw [← hc, cost_eq_arrive N hT s t v hcs, cost_eq_arrive N hT s' t v hc',
    WithTop.coe_le_coe] at h
  have h' : (arrive N t v s : ℝ) ≤ arrive N t v s' := by linarith
  exact_mod_cast h'

/-- T16 + T17, item 4 with closures. Let a link be usable at step `t` only while `ok e t`
holds, with closures permanent (`ok` antitone: a closure window `t + τ_e < b_e`, or the
earliest-escape burn check `t < ign(head e)`), costing `τ_e` when usable, and let waiting cost
one step. Then waiting never yields an earlier arrival (the DP with and without holdover
agree), and the DP without holdover returns the earliest arrival step minus `t`. This is
Studio's earliest-escape search, generalised to multi-step links. -/
theorem earliest_escape_no_wait (ok : E → ℕ → Prop) [∀ e, DecidablePred (ok e)]
    (hok : ∀ e t t', t ≤ t' → ok e t' → ok e t)
    (hd : ∀ e t, N.dose e t = if ok e t then ((N.tau e : ℝ) : WithTop ℝ) else ⊤)
    (hw : ∀ v t, N.wait v t = ((1 : ℝ) : WithTop ℝ)) (H t : ℕ) (v : V) :
    val N false H t v = val N true H t v ∧
      (val N false H t v ≠ ⊤ → ∃ m : ℕ, val N false H t v = (((m : ℝ) - t : ℝ) : WithTop ℝ) ∧
        IsLeast {a | ∃ s, Valid N false H t v s ∧ cost N t v s ≠ ⊤ ∧ arrive N t v s = a} m) := by
  have hmono : ∀ e, Monotone (N.dose e) := by
    intro e
    have := closure_dose_monotone (N.tau e) (ok e) (hok e)
    intro a b hab
    rw [hd, hd]
    exact this hab
  have hT : TimeCosts N := ⟨fun e t => by
    rw [hd]
    split_ifs
    · exact Or.inl rfl
    · exact Or.inr rfl, fun v t => Or.inl (hw v t)⟩
  refine ⟨no_holdover N hmono (fun v t => by rw [hw]; exact WithTop.coe_nonneg.2 zero_le_one)
    H t v, dp_earliest_arrival N hT false H t v⟩

omit [Fintype E] [DecidableEq V] in
/-- Towards L7. A schedule without waits and of finite dose under the earliest-escape costs
(unit links, a move leaving at tick `t` allowed only if `t < ign(head)`), started where a safe
walk of `t` moves from `s₀` ends, extends that walk to a safe walk (`GraphNumerics.SafeReach`)
that ends at the exit at the schedule's arrival tick. -/
theorem safeReach_of_valid (ign : V → WithTop ℝ) (x s₀ : V) (hτ : ∀ e, N.tau e = 1)
    (hd : ∀ e t, N.dose e t =
      if ((t : ℝ) : WithTop ℝ) < ign (N.head e) then ((1 : ℝ) : WithTop ℝ) else ⊤)
    (hx : ∀ v, N.isExit v = true → v = x) {H : ℕ} :
    ∀ (s : List (Act E)) (t : ℕ) (v : V), Valid N false H t v s → cost N t v s ≠ ⊤ →
      GraphNumerics.SafeReach (fun u w => ∃ e, N.tail e = u ∧ N.head e = w) ign s₀ t v →
      GraphNumerics.SafeReach (fun u w => ∃ e, N.tail e = u ∧ N.head e = w) ign s₀
        (arrive N t v s) x
  | [], _, v, ⟨hv, _⟩, _, hr => by
    rw [hx v hv] at hr
    exact hr
  | Act.wait :: _, _, _, ⟨_, hh, _⟩, _, _ => absurd hh Bool.false_ne_true
  | Act.move e :: s, t, v, ⟨_, he, _, hs⟩, hc, hr => by
    simp only [cost] at hc
    have h1 : N.dose e t ≠ ⊤ := (WithTop.add_ne_top.1 hc).1
    have hign : ((t : ℝ) : WithTop ℝ) < ign (N.head e) := by
      by_contra hn
      rw [hd, ite_eq_right hn] at h1
      exact h1 rfl
    have hr' := (GraphNumerics.safeReach_succ (fun u w => ∃ e, N.tail e = u ∧ N.head e = w)
      ign s₀ t (N.head e)).2 ⟨v, hr, ⟨e, he, rfl⟩, hign⟩
    rw [← hτ e] at hr'
    exact safeReach_of_valid ign x s₀ hτ hd hx s (t + N.tau e) (N.head e) hs
      (WithTop.add_ne_top.1 hc).2 hr'

omit [Fintype E] [DecidableEq V] in
/-- Towards L7, converse. A safe walk `w` of `k ≤ H` moves that reaches the exit `x` only at its
end gives, from each of its points `(w j, j)`, a schedule without waits, of finite dose under
the earliest-escape costs, that follows the walk and arrives at tick `k`. -/
theorem valid_of_safeReach (ign : V → WithTop ℝ) (x : V) (hτ : ∀ e, N.tau e = 1)
    (hd : ∀ e t, N.dose e t =
      if ((t : ℝ) : WithTop ℝ) < ign (N.head e) then ((1 : ℝ) : WithTop ℝ) else ⊤)
    (hxe : N.isExit x = true) (hne : ∀ v, v ≠ x → N.isExit v = false)
    (w : ℕ → V) (k H : ℕ) (hk : k ≤ H) (hwk : w k = x)
    (hstep : ∀ i < k, (∃ e, N.tail e = w i ∧ N.head e = w (i + 1)) ∧
      ((i : ℝ) : WithTop ℝ) < ign (w (i + 1)))
    (hfirst : ∀ i < k, w i ≠ x) :
    ∀ (n j : ℕ), k - j = n → j ≤ k →
      ∃ s, Valid N false H j (w j) s ∧ cost N j (w j) s ≠ ⊤ ∧ arrive N j (w j) s = k
  | 0, j, hn, hj => by
    have hjk : j = k := by omega
    rw [hjk, hwk]
    exact ⟨[], ⟨hxe, hk⟩, by simp [cost], rfl⟩
  | n + 1, j, hn, hj => by
    have hjk : j < k := by omega
    obtain ⟨⟨e, he1, he2⟩, hign⟩ := hstep j hjk
    obtain ⟨s, hs, hc, ha⟩ :=
      valid_of_safeReach ign x hτ hd hxe hne w k H hk hwk hstep hfirst n (j + 1) (by omega)
        (by omega)
    have hτe : j + N.tau e = j + 1 := by rw [hτ e]
    refine ⟨Act.move e :: s, ⟨hne _ (hfirst j hjk), he1, by omega, ?_⟩, ?_, ?_⟩
    · rw [hτe, he2]
      exact hs
    · simp only [cost]
      rw [hτe, he2, hd, he2, ite_eq_left hign]
      exact WithTop.add_ne_top.2 ⟨WithTop.coe_ne_top, hc⟩
    · simp only [arrive]
      rw [hτe, he2]
      exact ha

open Classical in
/-- T17, item 4 and L7: **the DP is the earliest-escape search**. Take Studio's earliest-escape
rule (`rl.js`/`rl.py` `earliestEscape`, proved exact in `GraphNumerics.L7_bfs_layer_eq_earliest`):
unit links, a move leaving at tick `t` toward `v` allowed only if `t < ign v`, one exit `x`.
Give each allowed move dose 1 (dose = time), each wait dose 1. If a safe walk from `s₀` reaches
`x`, and the least number `K` of its moves is within the horizon, then the min-dose DP from
`(s₀, 0)`, with or without holdover arcs, returns exactly `K`. -/
theorem dp_is_earliest_escape (ign : V → WithTop ℝ) (x s₀ : V) (hτ : ∀ e, N.tau e = 1)
    (hd : ∀ e t, N.dose e t =
      if ((t : ℝ) : WithTop ℝ) < ign (N.head e) then ((1 : ℝ) : WithTop ℝ) else ⊤)
    (hw : ∀ v t, N.wait v t = ((1 : ℝ) : WithTop ℝ)) (hx : ∀ v, N.isExit v = decide (v = x))
    (H : ℕ)
    (hex : ∃ k, GraphNumerics.SafeReach (fun u w => ∃ e, N.tail e = u ∧ N.head e = w) ign s₀ k x)
    (hH : Nat.find hex ≤ H) :
    val N false H 0 s₀ = ((Nat.find hex : ℝ) : WithTop ℝ) ∧
      val N true H 0 s₀ = ((Nat.find hex : ℝ) : WithTop ℝ) := by
  have hd' : ∀ e t, N.dose e t = if ((t : ℝ) : WithTop ℝ) < ign (N.head e) then
      ((N.tau e : ℝ) : WithTop ℝ) else ⊤ := by
    intro e t
    rw [hd, hτ e, Nat.cast_one]
  have hok : ∀ (e : E) (t t' : ℕ), t ≤ t' → ((t' : ℝ) : WithTop ℝ) < ign (N.head e) →
      ((t : ℝ) : WithTop ℝ) < ign (N.head e) :=
    fun e t t' h h' => lt_of_le_of_lt (WithTop.coe_le_coe.2 (Nat.cast_le.2 h)) h'
  obtain ⟨hsame, hearliest⟩ := earliest_escape_no_wait N
    (fun e t => ((t : ℝ) : WithTop ℝ) < ign (N.head e)) hok hd' hw H 0 s₀
  have hxe : N.isExit x = true := by rw [hx]; exact decide_eq_true rfl
  have hne : ∀ v, v ≠ x → N.isExit v = false := fun v hv => by rw [hx]; exact decide_eq_false hv
  have hx' : ∀ v, N.isExit v = true → v = x := fun v hv => by
    rw [hx] at hv
    exact of_decide_eq_true hv
  -- The least safe walk to the exit visits the exit only at its end.
  obtain ⟨w, hw0, hwK, hstep⟩ := Nat.find_spec hex
  have hfirst : ∀ i < Nat.find hex, w i ≠ x := by
    intro i hi hwi
    exact Nat.find_min hex hi ⟨w, hw0, hwi, fun t ht => hstep t (by omega)⟩
  obtain ⟨s, hs, hc, ha⟩ := valid_of_safeReach N ign x hτ hd hxe hne w (Nat.find hex) H hH hwK
    hstep hfirst (Nat.find hex) 0 (by omega) (Nat.zero_le _)
  rw [hw0] at hs hc ha
  have hfin : val N false H 0 s₀ ≠ ⊤ :=
    ne_top_of_le_ne_top hc (val_le_cost N false H s 0 s₀ hs)
  obtain ⟨m, hm, hleast⟩ := hearliest hfin
  have hmK : m ≤ Nat.find hex := hleast.2 ⟨s, hs, hc, ha⟩
  have hKm : Nat.find hex ≤ m := by
    obtain ⟨s', hs', hc', ha'⟩ := hleast.1
    have h0 : GraphNumerics.SafeReach (fun u w => ∃ e, N.tail e = u ∧ N.head e = w) ign s₀ 0 s₀ :=
      (GraphNumerics.safeReach_zero _ ign s₀ s₀).2 rfl
    have hr := safeReach_of_valid N ign x s₀ hτ hd hx' s' 0 s₀ hs' hc' h0
    rw [ha'] at hr
    exact Nat.find_min' hex hr
  have hval : val N false H 0 s₀ = ((Nat.find hex : ℝ) : WithTop ℝ) := by
    rw [hm, le_antisymm hmK hKm, Nat.cast_zero, sub_zero]
  exact ⟨hval, hsame ▸ hval⟩

open Classical in
/-- T17, item 4 and L7, in BFS terms. Under the earliest-escape costs, for every layer `k`
within the horizon: the exit is in layer `k` of the earliest-escape BFS
(`GraphNumerics.bfs`) iff the min-dose DP from `(s₀, 0)` without holdover returns `k`. So the
time-expanded DP, specialised to dose = time and unit links, reproduces `earliestEscape`
layer for layer, and with general doses and traversal times it generalises it. -/
theorem bfs_layer_iff_dp (ign : V → WithTop ℝ) (x s₀ : V) (hτ : ∀ e, N.tau e = 1)
    (hd : ∀ e t, N.dose e t =
      if ((t : ℝ) : WithTop ℝ) < ign (N.head e) then ((1 : ℝ) : WithTop ℝ) else ⊤)
    (hw : ∀ v t, N.wait v t = ((1 : ℝ) : WithTop ℝ)) (hx : ∀ v, N.isExit v = decide (v = x))
    (H k : ℕ) (hk : k ≤ H) :
    x ∈ (GraphNumerics.bfs (fun u w => ∃ e, N.tail e = u ∧ N.head e = w) ign s₀ k).1 ↔
      val N false H 0 s₀ = ((k : ℝ) : WithTop ℝ) := by
  rw [GraphNumerics.L7_bfs_layer_eq_earliest]
  constructor
  · rintro ⟨hex, hfind⟩
    rw [(dp_is_earliest_escape N ign x s₀ hτ hd hw hx H hex (by omega)).1, hfind]
  · intro hval
    have hfin : val N false H 0 s₀ ≠ ⊤ := by rw [hval]; exact WithTop.coe_ne_top
    obtain ⟨s, hs, hc⟩ := (dp_exact N false H 0 s₀).2 hfin
    have hx' : ∀ v, N.isExit v = true → v = x := fun v hv => by
      rw [hx] at hv
      exact of_decide_eq_true hv
    have hcs : cost N 0 s₀ s ≠ ⊤ := by rw [hc]; exact hfin
    have hr := safeReach_of_valid N ign x s₀ hτ hd hx' s 0 s₀ hs hcs
      ((GraphNumerics.safeReach_zero _ ign s₀ s₀).2 rfl)
    have hex : ∃ k, GraphNumerics.SafeReach (fun u w => ∃ e, N.tail e = u ∧ N.head e = w)
        ign s₀ k x := ⟨_, hr⟩
    have hH : Nat.find hex ≤ H := (Nat.find_min' hex hr).trans (arrive_le N s 0 s₀ hs)
    refine ⟨hex, ?_⟩
    have h2 := (dp_is_earliest_escape N ign x s₀ hτ hd hw hx H hex hH).1
    rw [hval, WithTop.coe_eq_coe] at h2
    exact_mod_cast h2.symm

/-! ## The hypothesis cannot be weakened -/

/-- A single link from `false` (start) to `true` (exit) taking one step, with move dose `d`
and free waiting. -/
def oneLink (d : ℕ → WithTop ℝ) : Net Bool Unit where
  tail _ := false
  head _ := true
  tau _ := 1
  tau_pos _ := le_rfl
  dose _ := d
  wait _ _ := 0
  isExit v := v

/-- On the single link, without holdover, the least dose from the start at step `t` (when
the move fits in the horizon) is the link's dose at `t`. -/
theorem oneLink_val_nowait (d : ℕ → WithTop ℝ) (H t : ℕ) (ht : t + 1 ≤ H) :
    val (oneLink d) false H t false = d t := by
  have h1 : val (oneLink d) false H (t + 1) true = 0 := val_exit (oneLink d) false rfl ht
  rw [val_bellman (oneLink d) false H t false rfl, ite_eq_right Bool.false_ne_true, min_top_left,
    Finset.univ_unique, Finset.inf_singleton,
    ite_eq_left (show (oneLink d).tail default = false from rfl)]
  exact (congrArg (d t + ·) h1).trans (add_zero _)

/-- T17, item 3, **the hypothesis is exact on one link**. On a single link with free waiting,
"the DP without holdover arcs is exact at every step and horizon" holds iff the link's move
dose is non-decreasing in the departure step. So monotonicity of move doses is not just
sufficient: no weaker condition on a link's own dose works when waiting is free. For Studio:
in a decay phase (dose rate falling, sprinklers, venting), holdover arcs are needed. -/
theorem oneLink_no_holdover_iff (d : ℕ → WithTop ℝ) :
    (∀ H t, val (oneLink d) false H t false = val (oneLink d) true H t false) ↔ Monotone d := by
  constructor
  · intro h
    refine monotone_nat_of_le_succ (fun n => ?_)
    by_contra hlt
    rw [not_le] at hlt
    have h1 := oneLink_val_nowait d (n + 2) n (by omega)
    have hv : Valid (oneLink d) true (n + 2) n false [Act.wait, Act.move ()] :=
      ⟨rfl, rfl, by omega, rfl, rfl, by show n + 1 + 1 ≤ n + 2; omega, rfl,
        by show n + 1 + 1 ≤ n + 2; omega⟩
    have h2 := val_le_cost (oneLink d) true (n + 2) [Act.wait, Act.move ()] n false hv
    have hc : cost (oneLink d) n false [Act.wait, Act.move ()] = d (n + 1) := by
      show 0 + (d (n + 1) + 0) = d (n + 1)
      rw [zero_add, add_zero]
    rw [hc, ← h (n + 2) n, h1] at h2
    exact absurd (lt_of_le_of_lt h2 hlt) (lt_irrefl _)
  · intro hd H t
    exact no_holdover (oneLink d) (fun _ => hd) (fun _ _ => le_rfl) H t false

/-- The decay counterexample: a link whose dose falls from 1 to 0 after the first step. -/
theorem decay_counterexample :
    val (oneLink (fun t => if t = 0 then 1 else 0)) false 2 0 false = 1 ∧
      val (oneLink (fun t => if t = 0 then 1 else 0)) true 2 0 false = 0 := by
  have h1 := oneLink_val_nowait (fun t => if t = 0 then (1 : WithTop ℝ) else 0) 2 0 (by omega)
  refine ⟨h1, le_antisymm ?_ (val_nonneg _ (fun _ t => by
    show (0 : WithTop ℝ) ≤ if t = 0 then 1 else 0
    split_ifs
    · exact zero_le_one
    · exact le_rfl) (fun _ _ => le_rfl) true 2 0 false)⟩
  have hv : Valid (oneLink (fun t => if t = 0 then (1 : WithTop ℝ) else 0)) true 2 0 false
      [Act.wait, Act.move ()] :=
    ⟨rfl, rfl, by omega, rfl, rfl, by show 0 + 1 + 1 ≤ 2; omega, rfl,
      by show 0 + 1 + 1 ≤ 2; omega⟩
  refine (val_le_cost _ true 2 _ 0 false hv).trans (le_of_eq ?_)
  show (0 : WithTop ℝ) + ((if 0 + 1 = 0 then 1 else 0) + 0) = 0
  simp

/-- Two links in series: `0 → 1` (always dose 0) then `1 → 2` (dose 3 when departing at step
`≤ 1`, 0 afterwards); node 2 is the exit; waiting is free at node 0 and costs 5 at node 1. -/
noncomputable def relay : Net (Fin 3) Bool where
  tail e := if e then 1 else 0
  head e := if e then 2 else 1
  tau _ := 1
  tau_pos _ := le_rfl
  dose e t := ((if e = true ∧ t ≤ 1 then 3 else 0 : ℝ) : WithTop ℝ)
  wait v _ := ((if v = 0 then 0 else 5 : ℝ) : WithTop ℝ)
  isExit v := decide (v = 2)

/-- T17, item 3, **a local test is not enough**. In `relay` all doses are non-negative and,
for every link and step, waiting one step at the link's tail and then taking the link is
never cheaper than taking it now (`d_e(t) ≤ w_{tail e}(t) + d_e(t+1)`). Yet with holdover the
least dose from node 0 at step 0 (horizon 3) is 0, and without holdover it is 3: the cheap
wait happens upstream, at node 0, and pays off on the later link. So the no-holdover
hypothesis must constrain how every later move's dose changes with time (monotonicity), not
just compare waiting and moving link by link. -/
theorem relay_counterexample :
    (∀ e t, 0 ≤ relay.dose e t) ∧ (∀ v t, 0 ≤ relay.wait v t) ∧
      (∀ e t, relay.dose e t ≤ relay.wait (relay.tail e) t + relay.dose e (t + 1)) ∧
      val relay true 3 0 0 = 0 ∧ val relay false 3 0 0 = 3 := by
  have hd : ∀ e t, 0 ≤ relay.dose e t := by
    intro e t
    show (0 : WithTop ℝ) ≤ ((if e = true ∧ t ≤ 1 then 3 else 0 : ℝ) : WithTop ℝ)
    refine WithTop.coe_nonneg.2 ?_
    split_ifs <;> norm_num
  have hw : ∀ v t, 0 ≤ relay.wait v t := by
    intro v t
    show (0 : WithTop ℝ) ≤ ((if v = 0 then 0 else 5 : ℝ) : WithTop ℝ)
    refine WithTop.coe_nonneg.2 ?_
    split_ifs <;> norm_num
  refine ⟨hd, hw, ?_, ?_, ?_⟩
  · intro e t
    show ((if e = true ∧ t ≤ 1 then 3 else 0 : ℝ) : WithTop ℝ) ≤
      ((if (if e then (1 : Fin 3) else 0) = 0 then 0 else 5 : ℝ) : WithTop ℝ) +
        ((if e = true ∧ t + 1 ≤ 1 then 3 else 0 : ℝ) : WithTop ℝ)
    rw [← WithTop.coe_add, WithTop.coe_le_coe]
    cases e
    · simp
    · have h5 : ((1 : Fin 3) = 0) = False := by decide
      simp only [ite_true, h5, ite_false, true_and]
      split_ifs <;> norm_num
  · refine le_antisymm ?_ (val_nonneg relay hd hw true 3 0 0)
    have hv : Valid relay true 3 0 0 [Act.wait, Act.move false, Act.move true] :=
      ⟨rfl, rfl, by omega, rfl, rfl, by show 0 + 1 + 1 ≤ 3; omega, rfl, rfl,
        by show 0 + 1 + 1 + 1 ≤ 3; omega, rfl, by show 0 + 1 + 1 + 1 ≤ 3; omega⟩
    refine (val_le_cost relay true 3 _ 0 0 hv).trans (le_of_eq ?_)
    show ((if (0 : Fin 3) = 0 then 0 else 5 : ℝ) : WithTop ℝ) +
      (((if false = true ∧ 0 + 1 ≤ 1 then 3 else 0 : ℝ) : WithTop ℝ) +
        (((if true = true ∧ 0 + 1 + 1 ≤ 1 then 3 else 0 : ℝ) : WithTop ℝ) + 0)) = 0
    simp
  · have v22 : val relay false 3 2 2 = 0 := val_exit relay false rfl (by omega)
    have v11 : val relay false 3 1 1 = 3 := by
      rw [val_bellman relay false 3 1 1 rfl, ite_eq_right Bool.false_ne_true, min_top_left,
        Fintype.univ_bool, Finset.inf_insert, Finset.inf_singleton]
      have ht : relay.tail true = 1 := rfl
      have hf : relay.tail false ≠ 1 := by decide
      rw [ite_eq_left ht, ite_eq_right hf]
      show min (((if true = true ∧ 1 ≤ 1 then 3 else 0 : ℝ) : WithTop ℝ) +
        val relay false 3 (1 + 1) 2) ⊤ = 3
      rw [v22, min_top_right]
      norm_num
    rw [val_bellman relay false 3 0 0 rfl, ite_eq_right Bool.false_ne_true, min_top_left,
      Fintype.univ_bool, Finset.inf_insert, Finset.inf_singleton]
    have ht : relay.tail false = 0 := rfl
    have hf : relay.tail true ≠ 0 := by decide
    rw [ite_eq_left ht, ite_eq_right hf]
    show min ⊤ (((if false = true ∧ 0 ≤ 1 then 3 else 0 : ℝ) : WithTop ℝ) +
      val relay false 3 (0 + 1) 1) = 3
    rw [min_top_left, zero_add, v11]
    norm_num

end FireEgress.MinDose

#print axioms FireEgress.MinDose.cost_nonneg
#print axioms FireEgress.MinDose.cost_take_le
#print axioms FireEgress.MinDose.valid_mono_horizon
#print axioms FireEgress.MinDose.le_arrive
#print axioms FireEgress.MinDose.arrive_le
#print axioms FireEgress.MinDose.valid_arrive
#print axioms FireEgress.MinDose.valid_false_nowait
#print axioms FireEgress.MinDose.valid_of_false
#print axioms FireEgress.MinDose.wait_not_mem_strip
#print axioms FireEgress.MinDose.strip_le
#print axioms FireEgress.MinDose.lt_add_of_pos_left'
#print axioms FireEgress.MinDose.add_lt_add_of_le_of_lt'
#print axioms FireEgress.MinDose.strip_lt
#print axioms FireEgress.MinDose.cost_eq_arrive
#print axioms FireEgress.MinDose.closure_dose_monotone
#print axioms FireEgress.MinDose.intDose_monotone
#print axioms FireEgress.MinDose.intDose_nonneg
#print axioms FireEgress.MinDose.val_eq
#print axioms FireEgress.MinDose.val_gt
#print axioms FireEgress.MinDose.val_exit
#print axioms FireEgress.MinDose.val_horizon
#print axioms FireEgress.MinDose.val_bellman
#print axioms FireEgress.MinDose.val_le_cost
#print axioms FireEgress.MinDose.greedy_valid
#print axioms FireEgress.MinDose.exists_greedy
#print axioms FireEgress.MinDose.dp_exact
#print axioms FireEgress.MinDose.dp_isLeast
#print axioms FireEgress.MinDose.val_eq_top_iff
#print axioms FireEgress.MinDose.val_nonneg
#print axioms FireEgress.MinDose.safe_iff
#print axioms FireEgress.MinDose.safe_iff_le
#print axioms FireEgress.MinDose.greedy_safe
#print axioms FireEgress.MinDose.exists_greedy_safe
#print axioms FireEgress.MinDose.safe_prefix
#print axioms FireEgress.MinDose.val_antitone_horizon
#print axioms FireEgress.MinDose.earliest_safe_arrival
#print axioms FireEgress.MinDose.val_le_val_false
#print axioms FireEgress.MinDose.no_holdover
#print axioms FireEgress.MinDose.exists_optimal_nowait
#print axioms FireEgress.MinDose.optimal_never_waits
#print axioms FireEgress.MinDose.no_holdover_monotone_fire
#print axioms FireEgress.MinDose.no_holdover_convective
#print axioms FireEgress.MinDose.dp_earliest_arrival
#print axioms FireEgress.MinDose.earliest_escape_no_wait
#print axioms FireEgress.MinDose.safeReach_of_valid
#print axioms FireEgress.MinDose.valid_of_safeReach
#print axioms FireEgress.MinDose.dp_is_earliest_escape
#print axioms FireEgress.MinDose.bfs_layer_iff_dp
#print axioms FireEgress.MinDose.oneLink_val_nowait
#print axioms FireEgress.MinDose.oneLink_no_holdover_iff
#print axioms FireEgress.MinDose.decay_counterexample
#print axioms FireEgress.MinDose.relay_counterexample
