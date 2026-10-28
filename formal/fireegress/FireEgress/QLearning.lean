import Mathlib
import FireEgress.Shaping

/-!
# Tabular Q-learning as Studio runs it: convergence and shaping at the `Q*` level (L15)

Studio's RL escape (`topologicpy-web-frontend/src/core/rl.js`, `qLearningPath`) is tabular
Q-learning on the navigation graph. Read from the code:

* **Actions** are CSR slots `k`: the directed, unblocked links out of a node. A slot fixes its
  source node, so the table is indexed by the slot alone (no fire) or by `(slot, t)` (fire).
* **State.** With a fire (`timed`, a non-empty ignition map) the state is the pair `(node, t)`,
  `t = 0, …, maxSteps - 1`: the key is `k * (maxSteps + 1) + t`. The burn test
  `t ≥ ignite(next)` depends on the step count, so the node alone is *not* Markov; the
  time-indexed state is, and the code uses it. Without a fire the key is `k` alone and the
  state is the node.
* **Transitions** are deterministic: slot `k` at step `t` leads to `(neighbours[k], t + 1)`.
* **Rewards:** `-slotCost[k]` (`0.1` per median link length), `+10` on entering the exit,
  `-10` on entering a node with `t ≥ ignite(next)`; both end the episode (`done`).
* **Horizon:** at `t = maxSteps - 1` the target is not bootstrapped (`last`), in both modes.
* **Update:** `Q ← Q + α (target - Q)` with constant `α = 0.5`, `γ = 0.99`, target
  `shaped + γ max_j Q(j, t + 1)` (no bootstrap if `done` or `last`; the max over an empty slot
  list is `0`). `shaped = reward + γ Φ_next - Φ(state)`, `Φ(v) = -0.1 d(v, exit) / unit`, and
  `Φ_next = 0` when `done` (exit or burn), else `Φ(next)` - **including at the horizon**.
* **Exploration:** ε-greedy, `ε = 0.2`, ties broken at random; 200 episodes of at most 200 steps
  from a fixed start. The reported path is a greedy rollout that never revisits a node.

## Model

A `DetMDP S P` is a deterministic decision problem written on its state-action pairs `P`
(Studio: slots, or `(slot, t)`): each pair has a state `st p`, a reward `r p`, and the finite
set `succ p` of pairs available after taking it (Studio: `slots[next]`, at `t + 1` when timed);
`succ p = ∅` marks a terminal transition (exit, burn or horizon). The Bellman optimality
operator is `T Q p = r p + γ · best Q (succ p)` with `best Q ∅ = 0`, exactly the code's target.
Writing the problem on pairs covers state-dependent action sets (Studio's slot lists) and the
textbook `S × A` form alike. Arithmetic is over `ℝ`; the code's IEEE doubles are not modelled.

## Results

1. *Bellman contraction* (`bellman_contracting`, `qstar_isFixed`, `eq_qstar_of_fixed`,
   `value_iteration_error`, `value_iteration_tendsto`). For `0 ≤ γ < 1` on a finite set of
   pairs, `T` is a `γ`-contraction in the sup norm, so it has one fixed point `Q*` and value
   iteration is within `γⁿ · dist Q₀ Q*` of it after `n` sweeps.
2. *Asynchronous deterministic Q-learning with a constant step* (`update_dist_le`,
   `sweep_contracts`, `qlearning_tendsto`, `qlearning_tendsto_weighted`,
   `eventually_greedy_optimal`). One update `Q(p) ← (1-α) Q(p) + α (T Q)(p)` with
   `α ∈ [0, 1]` never moves `Q` away from `Q*` in the sup norm; any window of updates that
   touches every pair with steps `≥ α_min` shrinks the distance by `1 - α_min (1 - γ)`; so if
   every pair is updated infinitely often with steps in `[α_min, 1]`, `α_min > 0`, then
   `Q → Q*`, and eventually every greedy choice is `Q*`-optimal. No Robbins-Monro condition is
   needed: the targets carry no noise. The weighted form only needs the pairs of a set closed
   under `succ` (Studio: the pairs reachable from the start) to be updated infinitely often.
3. *Finite horizon, any `γ ≤ 1`* (`iterate_agree`, `dag_fixed_exists_unique`,
   `dag_value_iteration_exact`, `timeIndexed_level`, `qlearning_tendsto_dag`). When every
   successor is one level lower (time-indexed pairs, `level = H - t`, horizon terminal), `T`
   has exactly one fixed point for every `γ`, including `γ = 1`; `n` sweeps of value iteration
   are exact on the last `n` layers (backward induction); and Q-learning converges by the same
   sweep argument in the weighted norm `max_p 2^{-level p} |Q p - Q* p|`.
4. *Shaping at the `Q*` level* (`shaped_fixed_general`, `shaped_fixed`, `shaped_qstar_eq`,
   `shaped_greedy_iff`, `shaped_same_greedy`, `shaped_dag_eq`). With `ψ p` the potential the
   code gives the state after `p` (`Φ(next)` when not terminal), `Q*_shaped = Q* - Φ ∘ st`
   when `ψ = 0` on terminal pairs, so greedy actions coincide (Ng, Harada & Russell 1999,
   deterministic case). Without that hypothesis `Q*_shaped = Q*_bonus - Φ ∘ st`, where the
   bonus problem pays an extra `γ ψ p` on terminal pairs.
5. *Counterexamples* (`nonzero_terminal_flips_qstar`, `horizon_potential_prefers_fire`,
   `untimed_horizon_breaks_fixed_point`). L14 lifted to `Q*`: a non-zero terminal potential
   flips the greedy action. Studio still keeps `Φ(next) ≠ 0` at the horizon: at the last step,
   with a burning neighbour and a safe neighbour 150 median links from the exit, the shaped
   `Q*` prefers walking into the fire to surviving (the same arithmetic holds whenever the safe
   neighbour is more than 100 median links away, i.e. `Φ < -10`). Without fire the state has no
   clock, yet the last step is not bootstrapped, so the update is not one operator and `Q*` is
   not a fixed point of it.

## What is not covered

* Stochastic transitions or rewards: Watkins & Dayan (1992), Tsitsiklis (1994) and Jaakkola,
  Jordan & Singh (1994) need Robbins-Monro steps (`∑ α = ∞`, `∑ α² < ∞`) and martingale or
  stochastic-approximation arguments; nothing here is probabilistic. Studio's fire is
  precomputed, so its environment is deterministic and this is not needed for Studio.
* That ε-greedy exploration visits every relevant pair infinitely often (a probability-one
  statement about the random run). It is a hypothesis (`hinf`) here, not a theorem.
* Any statement about Studio's finite run: 200 episodes of at most 200 steps give at most
  40,000 updates, against (slots × 200) time-indexed pairs; the theorems are asymptotic and
  say nothing about the trained table, the late-training success rate, or the no-revisit
  rollout. That `Q*` (the fixed point) equals the best return over trajectories is the
  standard identity, not re-proved here; `MinDose.lean` proves the backward-induction
  optimality for its dose objective.
* Floating-point rounding, random tie-breaking, and the untrained prior initialisation (the
  results hold for every initial table).
-/

namespace FireEgress.QLearning

open Finset Filter Topology

/-! ## The model -/

/-- A deterministic decision problem on state-action pairs `P`: the pair `p` is taken at state
`st p`, pays `r p`, and is followed by a choice among the pairs `succ p` (`∅`: terminal). -/
structure DetMDP (S P : Type*) where
  /-- The state at which a pair is taken (Studio: the slot's source node, and `t` if timed). -/
  st : P → S
  /-- The reward of taking a pair. -/
  r : P → ℝ
  /-- The pairs available after taking a pair; `∅` for a terminal transition. -/
  succ : P → Finset P

variable {S P : Type*}

/-- The best value among a finite set of pairs, `0` for the empty set: the code's
`Number.isFinite(bestNext) ? bestNext : 0`, and no bootstrap after a terminal transition. -/
noncomputable def best (Q : P → ℝ) (s : Finset P) : ℝ :=
  if h : s.Nonempty then s.sup' h Q else 0

/-- The Bellman optimality operator: `T Q p = r p + γ · max_{q ∈ succ p} Q q`. -/
noncomputable def bellman (M : DetMDP S P) (γ : ℝ) (Q : P → ℝ) : P → ℝ :=
  fun p => M.r p + γ * best Q (M.succ p)

theorem best_empty (Q : P → ℝ) : best Q ∅ = 0 := by
  simp [best]

theorem best_of_nonempty {s : Finset P} (h : s.Nonempty) (Q : P → ℝ) :
    best Q s = s.sup' h Q := by
  simp [best, h]

theorem le_best {s : Finset P} {q : P} (hq : q ∈ s) (Q : P → ℝ) : Q q ≤ best Q s := by
  rw [best_of_nonempty ⟨q, hq⟩]
  exact Finset.le_sup' Q hq

theorem best_le {s : Finset P} (h : s.Nonempty) {Q : P → ℝ} {c : ℝ}
    (hc : ∀ q ∈ s, Q q ≤ c) : best Q s ≤ c := by
  rw [best_of_nonempty h]
  exact Finset.sup'_le h Q hc

theorem exists_best {s : Finset P} (h : s.Nonempty) (Q : P → ℝ) : ∃ q ∈ s, best Q s = Q q := by
  rw [best_of_nonempty h]
  exact Finset.exists_mem_eq_sup' h Q

/-- `best` only reads the values on its set. -/
theorem best_congr {s : Finset P} {Q Q' : P → ℝ} (h : ∀ q ∈ s, Q q = Q' q) :
    best Q s = best Q' s := by
  rcases s.eq_empty_or_nonempty with rfl | hs
  · rw [best_empty, best_empty]
  · apply le_antisymm
    · exact best_le hs fun q hq => (h q hq).symm ▸ le_best hq Q'
    · exact best_le hs fun q hq => (h q hq) ▸ le_best hq Q

/-- The maxima of two tables over one non-empty set differ by at most the difference of the two
tables at some pair of the set. -/
theorem abs_best_sub_le {s : Finset P} (h : s.Nonempty) (Q Q' : P → ℝ) :
    ∃ q ∈ s, |best Q s - best Q' s| ≤ |Q q - Q' q| := by
  obtain ⟨q₁, h₁, e₁⟩ := exists_best h Q
  obtain ⟨q₂, h₂, e₂⟩ := exists_best h Q'
  rcases le_total (best Q' s) (best Q s) with hle | hle
  · refine ⟨q₁, h₁, ?_⟩
    rw [abs_of_nonneg (sub_nonneg.2 hle)]
    have := le_best h₁ Q'
    calc best Q s - best Q' s ≤ Q q₁ - Q' q₁ := by rw [e₁]; linarith
      _ ≤ |Q q₁ - Q' q₁| := le_abs_self _
  · refine ⟨q₂, h₂, ?_⟩
    rw [abs_of_nonpos (sub_nonpos.2 hle)]
    have := le_best h₂ Q
    calc -(best Q s - best Q' s) ≤ -(Q q₂ - Q' q₂) := by rw [e₂]; linarith
      _ ≤ |Q q₂ - Q' q₂| := neg_le_abs _

/-- Subtracting a function that is constant on the set lowers the maximum by that constant. -/
theorem best_sub_const {s : Finset P} (h : s.Nonempty) (Q f : P → ℝ) (c : ℝ)
    (hf : ∀ q ∈ s, f q = c) : best (fun q => Q q - f q) s = best Q s - c := by
  apply le_antisymm
  · refine best_le h fun q hq => ?_
    have := le_best hq Q
    rw [hf q hq]
    linarith
  · have : best Q s ≤ best (fun q => Q q - f q) s + c := by
      refine best_le h fun q hq => ?_
      have := le_best hq (fun q => Q q - f q)
      rw [hf q hq] at this
      linarith
    linarith

/-! ## (1) The Bellman operator is a `γ`-contraction -/

section Contraction

variable [Fintype P]

/-- L15.1 (Lipschitz bound). For `γ ≥ 0`, `‖T Q - T Q'‖∞ ≤ γ ‖Q - Q'‖∞`: two tables that
differ by at most `d` everywhere give Bellman targets that differ by at most `γ d`. -/
theorem dist_bellman_le (M : DetMDP S P) {γ : ℝ} (hγ : 0 ≤ γ) (Q Q' : P → ℝ) :
    dist (bellman M γ Q) (bellman M γ Q') ≤ γ * dist Q Q' := by
  refine (dist_pi_le_iff (mul_nonneg hγ dist_nonneg)).2 fun p => ?_
  rw [Real.dist_eq]
  simp only [bellman]
  rw [show M.r p + γ * best Q (M.succ p) - (M.r p + γ * best Q' (M.succ p)) =
      γ * (best Q (M.succ p) - best Q' (M.succ p)) by ring, abs_mul, abs_of_nonneg hγ]
  apply mul_le_mul_of_nonneg_left _ hγ
  rcases (M.succ p).eq_empty_or_nonempty with h | h
  · rw [h, best_empty, best_empty, sub_self, abs_zero]
    exact dist_nonneg
  · obtain ⟨q, -, hq⟩ := abs_best_sub_le h Q Q'
    calc |best Q (M.succ p) - best Q' (M.succ p)| ≤ |Q q - Q' q| := hq
      _ = dist (Q q) (Q' q) := (Real.dist_eq _ _).symm
      _ ≤ dist Q Q' := dist_le_pi_dist Q Q' q

/-- L15.1. For `0 ≤ γ < 1` the Bellman optimality operator is a contraction with constant `γ`
on tables `P → ℝ` with the sup norm (Mathlib's `ContractingWith`). For Studio (`γ = 0.99`)
this holds with or without fire, whatever the graph. -/
theorem bellman_contracting (M : DetMDP S P) {γ : ℝ} (hγ0 : 0 ≤ γ) (hγ1 : γ < 1) :
    ContractingWith ⟨γ, hγ0⟩ (bellman M γ) :=
  ⟨NNReal.coe_lt_coe.1 (by rw [NNReal.coe_one]; exact hγ1),
    LipschitzWith.of_dist_le_mul fun Q Q' => dist_bellman_le M hγ0 Q Q'⟩

/-- `Q*`: the fixed point of the Bellman optimality operator (Banach). -/
noncomputable def qstar (M : DetMDP S P) (γ : ℝ) (hγ0 : 0 ≤ γ) (hγ1 : γ < 1) : P → ℝ :=
  (bellman_contracting M hγ0 hγ1).fixedPoint (bellman M γ)

/-- L15.1. `Q*` satisfies the Bellman optimality equation `T Q* = Q*`. -/
theorem qstar_isFixed (M : DetMDP S P) {γ : ℝ} (hγ0 : 0 ≤ γ) (hγ1 : γ < 1) :
    bellman M γ (qstar M γ hγ0 hγ1) = qstar M γ hγ0 hγ1 :=
  (bellman_contracting M hγ0 hγ1).fixedPoint_isFixedPt

/-- L15.1. `Q*` is the only solution of the Bellman optimality equation: any table that
reproduces its own targets is `Q*`. -/
theorem eq_qstar_of_fixed (M : DetMDP S P) {γ : ℝ} (hγ0 : 0 ≤ γ) (hγ1 : γ < 1) {Q : P → ℝ}
    (hQ : bellman M γ Q = Q) : Q = qstar M γ hγ0 hγ1 :=
  (bellman_contracting M hγ0 hγ1).fixedPoint_unique hQ

/-- L15.1 (value iteration). After `n` synchronous sweeps from any table `Q₀`, the error to any
fixed point is at most `γⁿ` times the initial error. -/
theorem value_iteration_error (M : DetMDP S P) {γ : ℝ} (hγ0 : 0 ≤ γ) {Qs : P → ℝ}
    (hQs : bellman M γ Qs = Qs) (Q₀ : P → ℝ) (n : ℕ) :
    dist ((bellman M γ)^[n] Q₀) Qs ≤ γ ^ n * dist Q₀ Qs := by
  induction n with
  | zero => simp
  | succ n ih =>
    calc dist ((bellman M γ)^[n + 1] Q₀) Qs
        = dist (bellman M γ ((bellman M γ)^[n] Q₀)) (bellman M γ Qs) := by
          rw [Function.iterate_succ_apply', hQs]
      _ ≤ γ * dist ((bellman M γ)^[n] Q₀) Qs := dist_bellman_le M hγ0 _ _
      _ ≤ γ * (γ ^ n * dist Q₀ Qs) := mul_le_mul_of_nonneg_left ih hγ0
      _ = γ ^ (n + 1) * dist Q₀ Qs := by ring

/-- L15.1 (value iteration converges). For `0 ≤ γ < 1`, `Tⁿ Q₀ → Q*` from every table. -/
theorem value_iteration_tendsto (M : DetMDP S P) {γ : ℝ} (hγ0 : 0 ≤ γ) (hγ1 : γ < 1)
    (Q₀ : P → ℝ) :
    Tendsto (fun n => (bellman M γ)^[n] Q₀) atTop (𝓝 (qstar M γ hγ0 hγ1)) :=
  (bellman_contracting M hγ0 hγ1).tendsto_iterate_fixedPoint Q₀

end Contraction

/-! ## (2) Asynchronous Q-learning with a constant step -/

section Async

variable [DecidableEq P]

/-- One Q-learning update of the pair `p` with step `α`:
`Q(p) ← (1 - α) Q(p) + α (r p + γ max_{q ∈ succ p} Q q)`; every other entry is unchanged.
This is `q.set(key, current + alpha * (target - current))` in `rl.js`. -/
noncomputable def qUpdate (M : DetMDP S P) (γ α : ℝ) (Q : P → ℝ) (p : P) : P → ℝ :=
  Function.update Q p ((1 - α) * Q p + α * bellman M γ Q p)

/-- A run of asynchronous Q-learning: at step `n` the pair `pn n` is updated with step `α n`.
Episode boundaries only move the agent, not the table, so Studio's training (in the timed
case) is such a run with `α n = 0.5`. -/
def IsRun (M : DetMDP S P) (γ : ℝ) (pn : ℕ → P) (α : ℕ → ℝ) (Q : ℕ → P → ℝ) : Prop :=
  ∀ n, Q (n + 1) = qUpdate M γ (α n) (Q n) (pn n)

/-- L15.2 (one update, weighted). Let `Q*` be a fixed point, `A` a set of pairs closed under
`succ`, and weights `w ≥ 0` with `γ w p ≤ κ w q` along every link `p → q` inside `A`. If the
weighted error is at most `D` on `A`, updating `p ∈ A` leaves its weighted error at most
`(1 - α) · (old weighted error at p) + α κ D`. -/
theorem update_err_le (M : DetMDP S P) {γ α κ D : ℝ} {w : P → ℝ} {A : Set P}
    {Qs Q : P → ℝ} (hQs : bellman M γ Qs = Qs) (hγ : 0 ≤ γ) (hα0 : 0 ≤ α) (hα1 : α ≤ 1)
    (hκ0 : 0 ≤ κ) (hw : ∀ p, 0 ≤ w p) (hA : ∀ p ∈ A, ∀ q ∈ M.succ p, q ∈ A)
    (hwκ : ∀ p ∈ A, ∀ q ∈ M.succ p, γ * w p ≤ κ * w q)
    (hD : ∀ q ∈ A, w q * |Q q - Qs q| ≤ D) {p : P} (hp : p ∈ A) :
    w p * |qUpdate M γ α Q p p - Qs p| ≤ (1 - α) * (w p * |Q p - Qs p|) + α * (κ * D) := by
  have hb : M.r p + γ * best Qs (M.succ p) = Qs p := congrFun hQs p
  have key : qUpdate M γ α Q p p - Qs p =
      (1 - α) * (Q p - Qs p) + α * (γ * (best Q (M.succ p) - best Qs (M.succ p))) := by
    simp only [qUpdate, Function.update_self, bellman]
    linear_combination α * hb
  have hbest : γ * (w p * |best Q (M.succ p) - best Qs (M.succ p)|) ≤ κ * D := by
    rcases (M.succ p).eq_empty_or_nonempty with h | h
    · rw [h, best_empty, best_empty, sub_self, abs_zero, mul_zero, mul_zero]
      exact mul_nonneg hκ0 (le_trans (mul_nonneg (hw p) (abs_nonneg _)) (hD p hp))
    · obtain ⟨q, hq, hle⟩ := abs_best_sub_le h Q Qs
      calc γ * (w p * |best Q (M.succ p) - best Qs (M.succ p)|)
          ≤ γ * (w p * |Q q - Qs q|) := by
            gcongr
            · exact hw p
      _ = (γ * w p) * |Q q - Qs q| := by ring
      _ ≤ (κ * w q) * |Q q - Qs q| :=
          mul_le_mul_of_nonneg_right (hwκ p hp q hq) (abs_nonneg _)
      _ = κ * (w q * |Q q - Qs q|) := by ring
      _ ≤ κ * D := by gcongr; exact hD q (hA p hp q hq)
  rw [key]
  calc w p * |(1 - α) * (Q p - Qs p) + α * (γ * (best Q (M.succ p) - best Qs (M.succ p)))|
      ≤ w p * (|(1 - α) * (Q p - Qs p)| +
          |α * (γ * (best Q (M.succ p) - best Qs (M.succ p)))|) := by
        gcongr
        · exact hw p
        · exact abs_add_le _ _
    _ = (1 - α) * (w p * |Q p - Qs p|) +
          α * (γ * (w p * |best Q (M.succ p) - best Qs (M.succ p)|)) := by
        rw [abs_mul, abs_mul, abs_mul, abs_of_nonneg (sub_nonneg.2 hα1), abs_of_nonneg hα0,
          abs_of_nonneg hγ]
        ring
    _ ≤ (1 - α) * (w p * |Q p - Qs p|) + α * (κ * D) := by gcongr

/-- L15.2 (window invariant, weighted). Along a run with steps in `[α_min, 1]`, from step `N`
on: the weighted error on `A` never exceeds its bound `D` at step `N`, and every pair of `A`
updated at some step in `[N, m)` has weighted error at most `(1 - α_min (1 - κ)) D` at
step `m`. -/
theorem run_window (M : DetMDP S P) {γ κ αmin D : ℝ} {w : P → ℝ} {A : Set P}
    {Qs : P → ℝ} {pn : ℕ → P} {α : ℕ → ℝ} {Q : ℕ → P → ℝ} {N : ℕ}
    (hQs : bellman M γ Qs = Qs) (hγ : 0 ≤ γ) (hκ0 : 0 ≤ κ) (hκ1 : κ ≤ 1)
    (hw : ∀ p, 0 ≤ w p) (hA : ∀ p ∈ A, ∀ q ∈ M.succ p, q ∈ A)
    (hwκ : ∀ p ∈ A, ∀ q ∈ M.succ p, γ * w p ≤ κ * w q) (hrun : IsRun M γ pn α Q)
    (hα : ∀ n, αmin ≤ α n ∧ α n ≤ 1) (hαmin : 0 ≤ αmin)
    (hD : ∀ q ∈ A, w q * |Q N q - Qs q| ≤ D) :
    ∀ m, N ≤ m → (∀ q ∈ A, w q * |Q m q - Qs q| ≤ D) ∧
      ∀ q ∈ A, (∃ n, N ≤ n ∧ n < m ∧ pn n = q) →
        w q * |Q m q - Qs q| ≤ (1 - αmin * (1 - κ)) * D := by
  intro m hm
  induction m, hm using Nat.le_induction with
  | base => exact ⟨hD, fun q _ ⟨n, h1, h2, _⟩ => absurd h2 (not_lt.2 h1)⟩
  | succ m hNm ih =>
    obtain ⟨ih1, ih2⟩ := ih
    rw [hrun m]
    have hfresh : ∀ q ∈ A, q = pn m →
        w q * |qUpdate M γ (α m) (Q m) (pn m) q - Qs q| ≤ (1 - αmin * (1 - κ)) * D := by
      intro q hq hqm
      subst hqm
      have h1 := update_err_le M hQs hγ (hαmin.trans (hα m).1) (hα m).2 hκ0 hw hA hwκ ih1 hq
      have h2 := ih1 _ hq
      have hD0 : 0 ≤ D := le_trans (mul_nonneg (hw _) (abs_nonneg _)) h2
      have h3 : 0 ≤ 1 - α m := sub_nonneg.2 (hα m).2
      nlinarith [mul_le_mul_of_nonneg_left h2 h3,
        mul_nonneg (mul_nonneg (sub_nonneg.2 (hα m).1) (sub_nonneg.2 hκ1)) hD0]
    refine ⟨fun q hq => ?_, fun q hq hupd => ?_⟩
    · by_cases hqm : q = pn m
      · have := hfresh q hq hqm
        have h2 := ih1 _ hq
        have hD0 : 0 ≤ D := le_trans (mul_nonneg (hw _) (abs_nonneg _)) h2
        have : 0 ≤ αmin * (1 - κ) * D :=
          mul_nonneg (mul_nonneg hαmin (sub_nonneg.2 hκ1)) hD0
        nlinarith
      · rw [qUpdate, Function.update_of_ne hqm]
        exact ih1 q hq
    · by_cases hqm : q = pn m
      · exact hfresh q hq hqm
      · rw [qUpdate, Function.update_of_ne hqm]
        obtain ⟨n, h1, h2, h3⟩ := hupd
        have hnm : n ≠ m := fun h => hqm (h ▸ h3.symm)
        exact ih2 q hq ⟨n, h1, by omega, h3⟩

variable [Fintype P]

/-- L15.2 (one update never hurts). If `Q*` is a fixed point, `0 ≤ γ ≤ 1` and `α ∈ [0, 1]`,
one Q-learning update does not increase the sup-norm distance to `Q*`. -/
theorem update_dist_le (M : DetMDP S P) {γ α : ℝ} {Qs : P → ℝ} (hQs : bellman M γ Qs = Qs)
    (hγ0 : 0 ≤ γ) (hγ1 : γ ≤ 1) (hα0 : 0 ≤ α) (hα1 : α ≤ 1) (Q : P → ℝ) (p : P) :
    dist (qUpdate M γ α Q p) Qs ≤ dist Q Qs := by
  refine (dist_pi_le_iff dist_nonneg).2 fun q => ?_
  by_cases hqp : q = p
  · subst hqp
    have hD : ∀ x ∈ (Set.univ : Set P), (1 : ℝ) * |Q x - Qs x| ≤ dist Q Qs := by
      intro x _
      rw [one_mul, ← Real.dist_eq]
      exact dist_le_pi_dist Q Qs x
    have h := update_err_le M (w := fun _ => 1) (A := Set.univ) (κ := γ) hQs hγ0 hα0 hα1 hγ0
      (fun _ => zero_le_one) (fun _ _ _ _ => trivial) (fun _ _ _ _ => by simp) hD
      (Set.mem_univ q)
    have hq := hD q (Set.mem_univ q)
    simp only [one_mul] at h hq
    rw [Real.dist_eq]
    have h3 : 0 ≤ 1 - α := sub_nonneg.2 hα1
    nlinarith [mul_le_mul_of_nonneg_left hq h3, dist_nonneg (x := Q) (y := Qs),
      mul_nonneg (mul_nonneg hα0 (sub_nonneg.2 hγ1)) (dist_nonneg (x := Q) (y := Qs))]
  · rw [qUpdate, Function.update_of_ne hqp]
    exact dist_le_pi_dist Q Qs q

/-- L15.2 (a covering window contracts). Along a run with steps in `[α_min, 1]` and
`0 ≤ γ ≤ 1`, if every pair is updated at least once between steps `N` and `M'`, the sup-norm
distance to `Q*` shrinks by at least the factor `1 - α_min (1 - γ)`. For Studio,
`α_min = 0.5`, `γ = 0.99`: a factor `0.995` per covering window. -/
theorem sweep_contracts (M : DetMDP S P) {γ αmin : ℝ} {Qs : P → ℝ} {pn : ℕ → P}
    {α : ℕ → ℝ} {Q : ℕ → P → ℝ} (hQs : bellman M γ Qs = Qs) (hγ0 : 0 ≤ γ) (hγ1 : γ ≤ 1)
    (hrun : IsRun M γ pn α Q) (hα : ∀ n, αmin ≤ α n ∧ α n ≤ 1) (hαmin : 0 ≤ αmin)
    {N M' : ℕ} (hNM : N ≤ M') (hcover : ∀ q, ∃ n, N ≤ n ∧ n < M' ∧ pn n = q) :
    dist (Q M') Qs ≤ (1 - αmin * (1 - γ)) * dist (Q N) Qs := by
  have hD : ∀ x ∈ (Set.univ : Set P), (1 : ℝ) * |Q N x - Qs x| ≤ dist (Q N) Qs := by
    intro x _
    rw [one_mul, ← Real.dist_eq]
    exact dist_le_pi_dist (Q N) Qs x
  have h := (run_window M (w := fun _ => 1) (A := Set.univ) (κ := γ) hQs hγ0 hγ0 hγ1
    (fun _ => zero_le_one) (fun _ _ _ _ => trivial) (fun _ _ _ _ => by simp) hrun hα hαmin
    hD M' hNM).2
  have hαmin1 : αmin ≤ 1 := (hα 0).1.trans (hα 0).2
  have hρ : 0 ≤ 1 - αmin * (1 - γ) := by nlinarith
  refine (dist_pi_le_iff (mul_nonneg hρ dist_nonneg)).2 fun q => ?_
  have := h q (Set.mem_univ q) (hcover q)
  rw [one_mul] at this
  rwa [Real.dist_eq]

/-- L15.2 (convergence, weighted form). Let `A` be closed under `succ` (Studio: the pairs
reachable from the start), weights `w > 0` with `γ w p ≤ κ w q` along links inside `A`,
`κ < 1`, and a run whose steps lie in `[α_min, 1]` with `α_min > 0`. If every pair of `A` is
updated infinitely often, then `Q n p → Q* p` for every `p ∈ A`, whatever happens elsewhere. -/
theorem qlearning_tendsto_weighted (M : DetMDP S P) {γ κ αmin : ℝ} {w : P → ℝ} {A : Set P}
    {Qs : P → ℝ} {pn : ℕ → P} {α : ℕ → ℝ} {Q : ℕ → P → ℝ}
    (hQs : bellman M γ Qs = Qs) (hγ : 0 ≤ γ) (hκ0 : 0 ≤ κ) (hκ1 : κ < 1)
    (hw : ∀ p, 0 < w p) (hA : ∀ p ∈ A, ∀ q ∈ M.succ p, q ∈ A)
    (hwκ : ∀ p ∈ A, ∀ q ∈ M.succ p, γ * w p ≤ κ * w q) (hrun : IsRun M γ pn α Q)
    (hα : ∀ n, αmin ≤ α n ∧ α n ≤ 1) (hαmin : 0 < αmin)
    (hinf : ∀ q ∈ A, ∀ N, ∃ n, N ≤ n ∧ pn n = q) :
    ∀ q ∈ A, Tendsto (fun n => Q n q) atTop (𝓝 (Qs q)) := by
  set ρ := 1 - αmin * (1 - κ) with hρdef
  set D₀ := ∑ q, w q * |Q 0 q - Qs q| with hD₀def
  have hw0 : ∀ p, 0 ≤ w p := fun p => (hw p).le
  have hαmin1 : αmin ≤ 1 := (hα 0).1.trans (hα 0).2
  have hρ0 : 0 ≤ ρ := by nlinarith
  have hρ1 : ρ < 1 := by
    have : 0 < αmin * (1 - κ) := mul_pos hαmin (by linarith)
    linarith
  have hD₀ : ∀ q ∈ A, w q * |Q 0 q - Qs q| ≤ D₀ := fun q _ =>
    Finset.single_le_sum (f := fun q => w q * |Q 0 q - Qs q|)
      (fun i _ => mul_nonneg (hw0 i) (abs_nonneg _)) (Finset.mem_univ q)
  have hD₀nn : 0 ≤ D₀ := Finset.sum_nonneg fun i _ => mul_nonneg (hw0 i) (abs_nonneg _)
  have cover : ∀ N, ∃ M', N ≤ M' ∧ ∀ q ∈ A, ∃ n, N ≤ n ∧ n < M' ∧ pn n = q := by
    intro N
    have h : ∀ q, ∃ n, q ∈ A → N ≤ n ∧ pn n = q := fun q => by
      by_cases hq : q ∈ A
      · obtain ⟨n, hn⟩ := hinf q hq N
        exact ⟨n, fun _ => hn⟩
      · exact ⟨0, fun h => absurd h hq⟩
    choose f hf using h
    refine ⟨N + Finset.univ.sup f + 1, by omega,
      fun q hq => ⟨f q, (hf q hq).1, ?_, (hf q hq).2⟩⟩
    have := Finset.le_sup (f := f) (Finset.mem_univ q)
    omega
  have window := fun (N : ℕ) (D : ℝ) (hD : ∀ q ∈ A, w q * |Q N q - Qs q| ≤ D) =>
    run_window M (N := N) (D := D) hQs hγ hκ0 hκ1.le hw0 hA hwκ hrun hα hαmin.le hD
  have stage : ∀ k : ℕ, ∃ N, ∀ m, N ≤ m → ∀ q ∈ A, w q * |Q m q - Qs q| ≤ ρ ^ k * D₀ := by
    intro k
    induction k with
    | zero =>
      refine ⟨0, fun m hm q hq => ?_⟩
      rw [pow_zero, one_mul]
      exact (window 0 D₀ hD₀ m hm).1 q hq
    | succ k ih =>
      obtain ⟨N, hN⟩ := ih
      obtain ⟨M', hNM, hcov⟩ := cover N
      have hM := (window N (ρ ^ k * D₀) (hN N le_rfl) M' hNM).2
      have hM' : ∀ q ∈ A, w q * |Q M' q - Qs q| ≤ ρ ^ (k + 1) * D₀ := fun q hq => by
        have := hM q hq (hcov q hq)
        calc w q * |Q M' q - Qs q| ≤ ρ * (ρ ^ k * D₀) := this
          _ = ρ ^ (k + 1) * D₀ := by ring
      exact ⟨M', fun m hm q hq => (window M' _ hM' m hm).1 q hq⟩
  intro q hq
  rw [Metric.tendsto_atTop]
  intro ε hε
  have hwq := hw q
  obtain ⟨k, hk⟩ := exists_pow_lt_of_lt_one (show 0 < ε * w q / (D₀ + 1) by positivity) hρ1
  obtain ⟨N, hN⟩ := stage k
  refine ⟨N, fun m hm => ?_⟩
  rw [Real.dist_eq]
  have h1 := hN m hm q hq
  have h2 : ρ ^ k * D₀ < ε * w q := by
    calc ρ ^ k * D₀ ≤ ρ ^ k * (D₀ + 1) :=
          mul_le_mul_of_nonneg_left (by linarith) (pow_nonneg hρ0 k)
      _ < ε * w q / (D₀ + 1) * (D₀ + 1) := mul_lt_mul_of_pos_right hk (by linarith)
      _ = ε * w q := by field_simp
  have : w q * |Q m q - Qs q| < w q * ε := by linarith
  exact lt_of_mul_lt_mul_left this hwq.le

/-- L15.2 (convergence). For `0 ≤ γ < 1` and a run with steps in `[α_min, 1]`, `α_min > 0`,
in which every pair is updated infinitely often, the table converges to `Q*`. In particular
a constant step such as Studio's `α = 0.5` suffices in a deterministic environment; no
Robbins-Monro condition is needed. -/
theorem qlearning_tendsto (M : DetMDP S P) {γ αmin : ℝ} {pn : ℕ → P} {α : ℕ → ℝ}
    {Q : ℕ → P → ℝ} (hγ0 : 0 ≤ γ) (hγ1 : γ < 1) (hrun : IsRun M γ pn α Q)
    (hα : ∀ n, αmin ≤ α n ∧ α n ≤ 1) (hαmin : 0 < αmin)
    (hinf : ∀ q N, ∃ n, N ≤ n ∧ pn n = q) :
    Tendsto Q atTop (𝓝 (qstar M γ hγ0 hγ1)) := by
  refine tendsto_pi_nhds.2 fun q => ?_
  exact qlearning_tendsto_weighted M (w := fun _ => 1) (A := Set.univ) (κ := γ)
    (qstar_isFixed M hγ0 hγ1) hγ0 hγ0 hγ1 (fun _ => zero_lt_one) (fun _ _ _ _ => trivial)
    (fun _ _ _ _ => by simp) hrun hα hαmin (fun q _ => hinf q) q (Set.mem_univ q)

end Async

/-- L15.2 (greedy choices become optimal). If the table converges to `Q*` on a set `A`, then
from some step on, every choice that is greedy for the table among pairs of `A` is optimal
for `Q*`: the policy Studio reads off the table is eventually a `Q*`-greedy policy. -/
theorem eventually_greedy_optimal [Finite P] {Q : ℕ → P → ℝ} {Qs : P → ℝ} (A : Set P)
    (hQ : ∀ q ∈ A, Tendsto (fun n => Q n q) atTop (𝓝 (Qs q))) :
    ∀ᶠ n in atTop, ∀ s : Finset P, (↑s : Set P) ⊆ A → ∀ p ∈ s,
      (∀ q ∈ s, Q n q ≤ Q n p) → ∀ q ∈ s, Qs q ≤ Qs p := by
  have h : ∀ᶠ n in atTop, ∀ p q : P, p ∈ A → q ∈ A → Qs p < Qs q → Q n p < Q n q := by
    rw [Filter.eventually_all]
    intro p
    rw [Filter.eventually_all]
    intro q
    by_cases hp : p ∈ A
    · by_cases hq : q ∈ A
      · by_cases hlt : Qs p < Qs q
        · exact ((hQ p hp).eventually_lt (hQ q hq) hlt).mono fun n hn _ _ _ => hn
        · exact Filter.Eventually.of_forall fun n _ _ h => absurd h hlt
      · exact Filter.Eventually.of_forall fun n _ h => absurd h hq
    · exact Filter.Eventually.of_forall fun n h => absurd h hp
  filter_upwards [h] with n hn s hs p hp hgreedy q hq
  by_contra hcon
  have := hn p q (hs hp) (hs hq) (lt_of_not_ge hcon)
  linarith [hgreedy q hq]

/-! ## (3) Finite horizon: time-indexed pairs, any `γ ≤ 1` -/

section Horizon

/-- L15.3 (backward induction). If every successor of a pair is one level lower
(`level q < level p`), then after `n` sweeps of value iteration the table no longer depends on
where it started on the pairs of level `< n`: the last `n` time layers are final. -/
theorem iterate_agree (M : DetMDP S P) (γ : ℝ) (lvl : P → ℕ)
    (hlvl : ∀ p, ∀ q ∈ M.succ p, lvl q < lvl p) (Q Q' : P → ℝ) :
    ∀ n p, lvl p < n → (bellman M γ)^[n] Q p = (bellman M γ)^[n] Q' p := by
  intro n
  induction n with
  | zero => intro p h; omega
  | succ n ih =>
    intro p hp
    rw [Function.iterate_succ_apply', Function.iterate_succ_apply']
    simp only [bellman]
    rw [best_congr fun q hq => ih q (by have := hlvl p q hq; omega)]

/-- L15.3. On a finite set of pairs whose successors are always one level lower (for Studio:
time-indexed pairs with the horizon terminal), the Bellman equation has exactly one solution
for every `γ`, including the undiscounted `γ = 1`. -/
theorem dag_fixed_exists_unique [Fintype P] (M : DetMDP S P) (γ : ℝ) (lvl : P → ℕ)
    (hlvl : ∀ p, ∀ q ∈ M.succ p, lvl q < lvl p) : ∃! Qs : P → ℝ, bellman M γ Qs = Qs := by
  set L := Finset.univ.sup lvl
  have hL : ∀ p, lvl p < L + 1 := fun p => by
    have := Finset.le_sup (f := lvl) (Finset.mem_univ p)
    omega
  refine ⟨(bellman M γ)^[L + 1] 0, ?_, ?_⟩
  · show bellman M γ ((bellman M γ)^[L + 1] 0) = (bellman M γ)^[L + 1] 0
    rw [← Function.iterate_succ_apply' (bellman M γ), Function.iterate_succ_apply]
    funext p
    exact iterate_agree M γ lvl hlvl _ _ (L + 1) p (hL p)
  · intro Q hQ
    funext p
    rw [← Function.iterate_fixed hQ (L + 1)]
    exact iterate_agree M γ lvl hlvl _ _ (L + 1) p (hL p)

/-- L15.3 (value iteration is exact after `H` sweeps). On such a finite set of pairs, once `n`
exceeds every level, `n` sweeps from any table give the fixed point exactly: backward
induction on the time-expanded graph is `H` sweeps of the same Bellman operator. -/
theorem dag_value_iteration_exact (M : DetMDP S P) (γ : ℝ) (lvl : P → ℕ)
    (hlvl : ∀ p, ∀ q ∈ M.succ p, lvl q < lvl p) {Qs : P → ℝ} (hQs : bellman M γ Qs = Qs)
    (Q₀ : P → ℝ) (n : ℕ) (hn : ∀ p, lvl p < n) : (bellman M γ)^[n] Q₀ = Qs := by
  funext p
  rw [iterate_agree M γ lvl hlvl Q₀ Qs n p (hn p), Function.iterate_fixed hQs]

/-- L15.3 (time-indexed pairs are levelled). If every successor of a pair is taken one step
later (`t + 1`) and every pair is taken before the horizon `H` (the code's last step is
`maxSteps - 1` and is terminal), then `level p = H - t p` decreases along every link. -/
theorem timeIndexed_level (M : DetMDP S P) (tm : P → ℕ) (H : ℕ) (htm : ∀ p, tm p < H)
    (hsucc : ∀ p, ∀ q ∈ M.succ p, tm q = tm p + 1) :
    ∀ p, ∀ q ∈ M.succ p, H - tm q < H - tm p := by
  intro p q hq
  have := htm p
  have := hsucc p q hq
  omega

variable [Fintype P] [DecidableEq P]

/-- L15.3 (Q-learning on a finite horizon, any `0 ≤ γ ≤ 1`). On levelled pairs, a run with
steps in `[α_min, 1]`, `α_min > 0`, that updates every pair infinitely often converges to the
fixed point, also for `γ = 1`. The proof is the sweep argument in the weighted norm
`max_p 2^{-level p} |Q p - Q* p|`, in which the update contracts by `γ / 2`. -/
theorem qlearning_tendsto_dag (M : DetMDP S P) {γ αmin : ℝ} (lvl : P → ℕ)
    (hlvl : ∀ p, ∀ q ∈ M.succ p, lvl q < lvl p) {Qs : P → ℝ} (hQs : bellman M γ Qs = Qs)
    {pn : ℕ → P} {α : ℕ → ℝ} {Q : ℕ → P → ℝ} (hγ0 : 0 ≤ γ) (hγ1 : γ ≤ 1)
    (hrun : IsRun M γ pn α Q) (hα : ∀ n, αmin ≤ α n ∧ α n ≤ 1) (hαmin : 0 < αmin)
    (hinf : ∀ q N, ∃ n, N ≤ n ∧ pn n = q) :
    Tendsto Q atTop (𝓝 Qs) := by
  refine tendsto_pi_nhds.2 fun q => ?_
  refine qlearning_tendsto_weighted M (w := fun p => (1 / 2 : ℝ) ^ lvl p) (A := Set.univ)
    (κ := γ / 2) hQs hγ0 (by positivity) (by linarith) (fun p => by positivity)
    (fun _ _ _ _ => trivial) ?_ hrun hα hαmin (fun q _ => hinf q) q (Set.mem_univ q)
  intro p _ q hq
  have h : (1 / 2 : ℝ) ^ lvl p ≤ (1 / 2) ^ (lvl q + 1) :=
    pow_le_pow_of_le_one (by norm_num) (by norm_num) (hlvl p q hq)
  rw [pow_succ] at h
  calc γ * (1 / 2 : ℝ) ^ lvl p ≤ γ * ((1 / 2) ^ lvl q * (1 / 2)) :=
        mul_le_mul_of_nonneg_left h hγ0
    _ = γ / 2 * (1 / 2) ^ lvl q := by ring

end Horizon

/-! ## (4) Potential-based shaping at the level of `Q*` (L15) -/

section ShapingQ

/-- The shaped problem: the pair `p` pays `r p + γ ψ p - Φ(st p)`, where `ψ p` is the
potential the code assigns to the state after `p` (`phiNext` in `rl.js`). -/
def shapedMDP (M : DetMDP S P) (γ : ℝ) (Φ : S → ℝ) (ψ : P → ℝ) : DetMDP S P where
  st := M.st
  r := fun p => M.r p + γ * ψ p - Φ (M.st p)
  succ := M.succ

/-- The bonus problem: the unshaped rewards plus `γ ψ p` on terminal pairs only. -/
noncomputable def bonusMDP (M : DetMDP S P) (γ : ℝ) (ψ : P → ℝ) : DetMDP S P where
  st := M.st
  r := fun p => M.r p + if (M.succ p).Nonempty then 0 else γ * ψ p
  succ := M.succ

/-- L15.4 (shaping in general). Let `ψ p` be the potential of the next state whenever `p` is
not terminal (`ψ p = Φ(st q)` for every successor `q`). If `Q_b` solves the Bellman equation
of the bonus problem, then `Q_b - Φ ∘ st` solves that of the shaped problem. So shaping with a
terminal potential `ψ ≠ 0` is the same as paying a bonus `γ ψ` on terminal transitions. -/
theorem shaped_fixed_general (M : DetMDP S P) (γ : ℝ) (Φ : S → ℝ) (ψ : P → ℝ)
    (hψ : ∀ p, ∀ q ∈ M.succ p, ψ p = Φ (M.st q)) {Qb : P → ℝ}
    (hQb : bellman (bonusMDP M γ ψ) γ Qb = Qb) :
    bellman (shapedMDP M γ Φ ψ) γ (fun p => Qb p - Φ (M.st p)) =
      fun p => Qb p - Φ (M.st p) := by
  funext p
  have hb := congrFun hQb p
  simp only [bellman, shapedMDP, bonusMDP] at hb ⊢
  rcases (M.succ p).eq_empty_or_nonempty with h | h
  · rw [h, best_empty] at hb ⊢
    simp only [Finset.not_nonempty_empty, ite_false] at hb
    linear_combination hb
  · rw [best_sub_const h Qb (fun q => Φ (M.st q)) (ψ p) (fun q hq => (hψ p q hq).symm)]
    simp only [h, ite_true] at hb
    linear_combination hb

/-- L15.4 (shaping with `Φ = 0` at terminals, Ng-Harada-Russell, deterministic). If `ψ` is the
potential of the next state on non-terminal pairs and `0` on terminal ones (the W5 fix), then
`Q* - Φ ∘ st` solves the shaped Bellman equation. -/
theorem shaped_fixed (M : DetMDP S P) (γ : ℝ) (Φ : S → ℝ) (ψ : P → ℝ)
    (hψ : ∀ p, ∀ q ∈ M.succ p, ψ p = Φ (M.st q)) (hterm : ∀ p, M.succ p = ∅ → ψ p = 0)
    {Qs : P → ℝ} (hQs : bellman M γ Qs = Qs) :
    bellman (shapedMDP M γ Φ ψ) γ (fun p => Qs p - Φ (M.st p)) =
      fun p => Qs p - Φ (M.st p) := by
  refine shaped_fixed_general M γ Φ ψ hψ ?_
  funext p
  have hb := congrFun hQs p
  simp only [bellman, bonusMDP] at hb ⊢
  rcases (M.succ p).eq_empty_or_nonempty with h | h
  · rw [hterm p h]
    simp only [h, Finset.not_nonempty_empty, ite_false, mul_zero, add_zero] at hb ⊢
    exact hb
  · simp only [h, ite_true, add_zero]
    exact hb

/-- L15.4 (`Q*` of the shaped problem). For `0 ≤ γ < 1` and zero potential at terminal
transitions, `Q*_shaped(p) = Q*(p) - Φ(st p)` for every pair. -/
theorem shaped_qstar_eq [Fintype P] (M : DetMDP S P) {γ : ℝ} (hγ0 : 0 ≤ γ) (hγ1 : γ < 1)
    (Φ : S → ℝ) (ψ : P → ℝ) (hψ : ∀ p, ∀ q ∈ M.succ p, ψ p = Φ (M.st q))
    (hterm : ∀ p, M.succ p = ∅ → ψ p = 0) :
    qstar (shapedMDP M γ Φ ψ) γ hγ0 hγ1 = fun p => qstar M γ hγ0 hγ1 p - Φ (M.st p) :=
  (eq_qstar_of_fixed _ hγ0 hγ1 (shaped_fixed M γ Φ ψ hψ hterm (qstar_isFixed M hγ0 hγ1))).symm

/-- L15.4 (greedy choices coincide). Under the same hypotheses, two pairs taken at the same
state compare the same way under `Q*_shaped` as under `Q*`. -/
theorem shaped_greedy_iff [Fintype P] (M : DetMDP S P) {γ : ℝ} (hγ0 : 0 ≤ γ) (hγ1 : γ < 1)
    (Φ : S → ℝ) (ψ : P → ℝ) (hψ : ∀ p, ∀ q ∈ M.succ p, ψ p = Φ (M.st q))
    (hterm : ∀ p, M.succ p = ∅ → ψ p = 0) {p q : P} (hpq : M.st p = M.st q) :
    qstar (shapedMDP M γ Φ ψ) γ hγ0 hγ1 p ≤ qstar (shapedMDP M γ Φ ψ) γ hγ0 hγ1 q ↔
      qstar M γ hγ0 hγ1 p ≤ qstar M γ hγ0 hγ1 q := by
  rw [shaped_qstar_eq M hγ0 hγ1 Φ ψ hψ hterm]
  dsimp only
  rw [hpq]
  constructor <;> intro h <;> linarith

/-- L15.4 (the greedy action set is unchanged). Among the actions `s` of one state, a pair is
greedy for `Q*_shaped` exactly when it is greedy for `Q*`. -/
theorem shaped_same_greedy [Fintype P] (M : DetMDP S P) {γ : ℝ} (hγ0 : 0 ≤ γ) (hγ1 : γ < 1)
    (Φ : S → ℝ) (ψ : P → ℝ) (hψ : ∀ p, ∀ q ∈ M.succ p, ψ p = Φ (M.st q))
    (hterm : ∀ p, M.succ p = ∅ → ψ p = 0) (s : Finset P) (x : S)
    (hs : ∀ q ∈ s, M.st q = x) {p : P} (hp : p ∈ s) :
    (∀ q ∈ s, qstar (shapedMDP M γ Φ ψ) γ hγ0 hγ1 q ≤ qstar (shapedMDP M γ Φ ψ) γ hγ0 hγ1 p) ↔
      ∀ q ∈ s, qstar M γ hγ0 hγ1 q ≤ qstar M γ hγ0 hγ1 p := by
  refine forall₂_congr fun q hq => ?_
  exact shaped_greedy_iff M hγ0 hγ1 Φ ψ hψ hterm ((hs q hq).trans (hs p hp).symm)

/-- L15.4 on a finite horizon (any `γ`, including `γ = 1`). On levelled pairs (time-indexed,
horizon terminal) with zero potential at terminal transitions, the solution of the shaped
Bellman equation is `Q* - Φ ∘ st`. -/
theorem shaped_dag_eq [Fintype P] (M : DetMDP S P) (γ : ℝ) (lvl : P → ℕ)
    (hlvl : ∀ p, ∀ q ∈ M.succ p, lvl q < lvl p) (Φ : S → ℝ) (ψ : P → ℝ)
    (hψ : ∀ p, ∀ q ∈ M.succ p, ψ p = Φ (M.st q)) (hterm : ∀ p, M.succ p = ∅ → ψ p = 0)
    {Qs Qsh : P → ℝ} (hQs : bellman M γ Qs = Qs)
    (hQsh : bellman (shapedMDP M γ Φ ψ) γ Qsh = Qsh) :
    Qsh = fun p => Qs p - Φ (M.st p) :=
  (dag_fixed_exists_unique (shapedMDP M γ Φ ψ) γ lvl hlvl).unique hQsh
    (shaped_fixed M γ Φ ψ hψ hterm hQs)

end ShapingQ

/-! ## (5) Counterexamples -/

section Counterexamples

/-- L14's two outcomes as one decision: from one state, action `false` ends the episode with
reward `-10` in a terminal state of potential `5`; action `true` ends it with reward `-9` in a
terminal state of potential `0`. -/
noncomputable def flipMDP : DetMDP Unit Bool where
  st := fun _ => ()
  r := fun b => if b then -9 else -10
  succ := fun _ => ∅

/-- The terminal potentials of `flipMDP`. -/
noncomputable def flipψ : Bool → ℝ := fun b => if b then 0 else 5

theorem bellman_terminal_only (M : DetMDP S P) (γ : ℝ) (h : ∀ p, M.succ p = ∅)
    (Q : P → ℝ) : bellman M γ Q = Q ↔ Q = M.r := by
  have : bellman M γ Q = M.r := by
    funext p
    simp [bellman, h p, best_empty]
  rw [this]
  exact eq_comm

/-- L15.5 (L14 at the level of `Q*`). With `Φ(terminal) ≠ 0` the greedy action changes:
in `flipMDP` (`γ = 1`, start potential `0`) the unique `Q*` prefers `true` (`-9 > -10`), but
the unique shaped `Q*` prefers `false` (`-5 > -9`). The values are L14's returns
(`Shaping.ret`, `Shaping.shaped`), and the flip is `Shaping.nonzero_terminal_flips_order`. -/
theorem nonzero_terminal_flips_qstar :
    (∀ Q : Bool → ℝ, bellman flipMDP 1 Q = Q ↔ Q = flipMDP.r) ∧
    (∀ Q : Bool → ℝ, bellman (shapedMDP flipMDP 1 (fun _ => 0) flipψ) 1 Q = Q ↔
      Q = fun b => flipMDP.r b + flipψ b) ∧
    flipMDP.r false < flipMDP.r true ∧
    flipMDP.r true + flipψ true < flipMDP.r false + flipψ false := by
  obtain ⟨h1, h2⟩ := FireEgress.Shaping.nonzero_terminal_flips_order
  refine ⟨bellman_terminal_only _ _ (fun _ => rfl), fun Q => ?_, ?_, ?_⟩
  · rw [bellman_terminal_only _ _ (fun _ => rfl)]
    have : (shapedMDP flipMDP 1 (fun _ => 0) flipψ).r = fun b => flipMDP.r b + flipψ b := by
      funext b
      simp [shapedMDP]
    rw [this]
  · have e1 : flipMDP.r false = FireEgress.Shaping.ret 1 (fun _ => -10) 1 := by
      simp [flipMDP, FireEgress.Shaping.ret]
    have e2 : flipMDP.r true = FireEgress.Shaping.ret 1 (fun _ => -9) 1 := by
      simp [flipMDP, FireEgress.Shaping.ret]
    rw [e1, e2]
    exact h1
  · have e1 : flipMDP.r true + flipψ true =
        FireEgress.Shaping.shaped 1 (fun _ => -9) (fun t => if t = 0 then 0 else 0) 1 := by
      simp [flipMDP, flipψ, FireEgress.Shaping.shaped]
    have e2 : flipMDP.r false + flipψ false =
        FireEgress.Shaping.shaped 1 (fun _ => -10) (fun t => if t = 0 then 0 else 5) 1 := by
      simp [flipMDP, flipψ, FireEgress.Shaping.shaped]
    rw [e1, e2]
    exact h2

/-- Studio's last step (`t = maxSteps - 1`): pair `false` enters a burning node (reward `-10`,
`done`, so `rl.js` uses potential `0`); pair `true` takes a median-length link (reward `-0.1`)
to a safe node 150 median links from the exit, where the horizon ends the episode but `rl.js`
keeps `Φ(next) = -0.1 · 150 = -15`. -/
noncomputable def horizonMDP : DetMDP Unit Bool where
  st := fun _ => ()
  r := fun b => if b then -1 / 10 else -10
  succ := fun _ => ∅

/-- The potentials `rl.js` gives the states after the two last-step pairs. -/
noncomputable def horizonψ : Bool → ℝ := fun b => if b then -15 else 0

/-- L15.5 (Studio's horizon potential changes the optimum). In `horizonMDP` with Studio's
`γ = 0.99` and any potential `Φ` of the current node, the unique `Q*` prefers surviving to the
horizon (`-0.1 > -10`), but the unique shaped `Q*` that `rl.js` learns prefers walking into the
fire (`-10 - Φ > -0.1 - 14.85 - Φ`). The same arithmetic flips the choice whenever the safe
neighbour is more than 100 median links from the exit (`Φ < -10`); the cause is that `rl.js`
zeroes `Φ_next` on `done` but not on `last`. -/
theorem horizon_potential_prefers_fire (Φ : Unit → ℝ) :
    (∀ Q : Bool → ℝ, bellman horizonMDP (99 / 100) Q = Q ↔ Q = horizonMDP.r) ∧
    (∀ Q : Bool → ℝ, bellman (shapedMDP horizonMDP (99 / 100) Φ horizonψ) (99 / 100) Q = Q ↔
      Q = fun b => horizonMDP.r b + 99 / 100 * horizonψ b - Φ ()) ∧
    horizonMDP.r false < horizonMDP.r true ∧
    horizonMDP.r true + 99 / 100 * horizonψ true - Φ () <
      horizonMDP.r false + 99 / 100 * horizonψ false - Φ () := by
  refine ⟨bellman_terminal_only _ _ (fun _ => rfl),
    bellman_terminal_only (shapedMDP horizonMDP (99 / 100) Φ horizonψ) _ (fun _ => rfl), ?_, ?_⟩
  · norm_num [horizonMDP]
  · norm_num [horizonMDP, horizonψ]

/-- One pair that leads back to itself (an agent that can keep moving), reward `-1`. -/
noncomputable def loopMDP : DetMDP Unit Unit where
  st := fun _ => ()
  r := fun _ => -1
  succ := fun _ => {()}

/-- L15.5 (no clock, but a terminal horizon). Without fire `rl.js` keys the table by the slot
alone, yet at `t = maxSteps - 1` it drops the bootstrap. In `loopMDP` with `γ = 1/2`,
`Q* = -2`, while the last-step target is the bare reward `-1`; so every last-step update with
`α > 0` moves the entry off `Q*`, and the run is not one operator iterated. -/
theorem untimed_horizon_breaks_fixed_point :
    bellman loopMDP (1 / 2) (fun _ => -2) = (fun _ => -2) ∧
    ∀ α : ℝ, 0 < α → (1 - α) * (-2) + α * loopMDP.r () ≠ -2 := by
  refine ⟨?_, fun α hα => ?_⟩
  · funext p
    simp only [bellman, loopMDP]
    rw [best_of_nonempty (Finset.singleton_nonempty ()), Finset.sup'_singleton]
    norm_num
  · simp only [loopMDP]
    intro h
    linarith

end Counterexamples

end FireEgress.QLearning

#print axioms FireEgress.QLearning.best_empty
#print axioms FireEgress.QLearning.best_of_nonempty
#print axioms FireEgress.QLearning.le_best
#print axioms FireEgress.QLearning.best_le
#print axioms FireEgress.QLearning.exists_best
#print axioms FireEgress.QLearning.best_congr
#print axioms FireEgress.QLearning.abs_best_sub_le
#print axioms FireEgress.QLearning.best_sub_const
#print axioms FireEgress.QLearning.dist_bellman_le
#print axioms FireEgress.QLearning.bellman_contracting
#print axioms FireEgress.QLearning.qstar_isFixed
#print axioms FireEgress.QLearning.eq_qstar_of_fixed
#print axioms FireEgress.QLearning.value_iteration_error
#print axioms FireEgress.QLearning.value_iteration_tendsto
#print axioms FireEgress.QLearning.update_err_le
#print axioms FireEgress.QLearning.run_window
#print axioms FireEgress.QLearning.update_dist_le
#print axioms FireEgress.QLearning.sweep_contracts
#print axioms FireEgress.QLearning.qlearning_tendsto_weighted
#print axioms FireEgress.QLearning.qlearning_tendsto
#print axioms FireEgress.QLearning.eventually_greedy_optimal
#print axioms FireEgress.QLearning.iterate_agree
#print axioms FireEgress.QLearning.dag_fixed_exists_unique
#print axioms FireEgress.QLearning.dag_value_iteration_exact
#print axioms FireEgress.QLearning.timeIndexed_level
#print axioms FireEgress.QLearning.qlearning_tendsto_dag
#print axioms FireEgress.QLearning.shaped_fixed_general
#print axioms FireEgress.QLearning.shaped_fixed
#print axioms FireEgress.QLearning.shaped_qstar_eq
#print axioms FireEgress.QLearning.shaped_greedy_iff
#print axioms FireEgress.QLearning.shaped_same_greedy
#print axioms FireEgress.QLearning.shaped_dag_eq
#print axioms FireEgress.QLearning.bellman_terminal_only
#print axioms FireEgress.QLearning.nonzero_terminal_flips_qstar
#print axioms FireEgress.QLearning.horizon_potential_prefers_fire
#print axioms FireEgress.QLearning.untimed_horizon_breaks_fixed_point
