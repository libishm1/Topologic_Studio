import Mathlib

/-!
# Shielded learning on the time-expanded fire graph (`rlSafe.js`)

Studio's learned escape (`topologicpy-web-frontend/src/core/rlSafe.js`) replaces the tabular
Q-learning of `rl.js` (after Jabi et al. 2019, doi:10.5151/proceedings-ecaadesigradi2019_671).
The study behind it is `wiki/research/rl-egress-study-2026-11-05.md`. This file proves what the
module's header claims, on the module's own model, read from the code:

* **The clock** is `rl.js`'s: one open link per step. A move made at step `t` into `w` is safe
  when it arrives at `t + 1 ≤ opn w`, with `opn w = ⌈ig w⌉` capped at the step limit `H`
  (`openUntil`; `ig` is the fire's hop front over the fire speed). An exit ends the walk. A
  `Net` holds the open links `nb`, the exits `ex`, `opn` and `H`; `Esc G t v k` says that from
  `v` at step `t` an exit is reached safely in exactly `k` moves by step `H`.
* **Value iteration** (`escapeValues`): `bi G n` is the table `n` sweeps back from the horizon,
  so the value at step `t` is `bi G (H - t)`. It is the earliest safe escape (`bi_spec`,
  `vi_le`, `vi_esc`, `vi_top_iff`, `vi_eq_earliest`), and never falls with a later step
  (`esc_mono`, `val_mono_time`): leaving later never escapes sooner.
* **The deadline map** (`hopDeadlines`): any `D` solving its recursion (`IsDeadline`: `D x = opn x`
  at exits, `D v = max (-1) (min (opn v) (max_w D w - 1))` elsewhere; `rlSafe.test.mjs` checks
  the code's output against it) satisfies `t ≤ D v ↔ t ≤ opn v ∧ ∃ k, Esc G t v k`
  (`deadline_iff`). The recursion has no other solution, so the code's map is the deadline.
* **The shield** (moves into `w` at step `t` only when `t + 1 ≤ D w`): an allowed move is safe on
  arrival and keeps an escape (`shield_safe`, `shield_keeps`); a move is masked only if no
  escape goes through it (`shield_complete`); some move is allowed until an exit is reached
  (`shield_nonempty`); and every walk inside the shield from a start with an escape reaches an
  exit by step `H`, never entering a node after it closes (`shield_reaches_exit`), whatever
  chooses its moves: a learner, exploration or chance.
* **The learner** (`learnSafeEscape`, learning real-time search on `(node, step)`): a table of
  lower bounds is `Admissible`; the fire-free or deadline-restricted hop distance is one
  (`consistent_admissible`); the update `max (V v t) (min over allowed w of 1 + V w (t + 1))` keeps
  the table admissible (`lrta_admissible`), and so does applying a bound to every later step, as
  `StepTable` does (`raise_admissible`). A trial no longer than `V start 0` is an earliest
  escape (`walk_esc`, `certificate_optimal`); a trial that changes no value passes that test
  (`trial_no_change`); and tables that only rise and stay below a bound change only finitely
  often (`bounded_raises`), so learning stops with the certificate after finitely many trials.
* **Shaping**: initialising the values with a potential `h` is the same update as shaping the
  move costs by `h w - h v` from zero (`potential_init_eq_shaping`; Wiewiora 2003 for
  Q-values). That shaping leaves the optimal policy unchanged on a time-expanded problem with
  `Φ = 0` at its terminals is `QLearning.shaped_dag_eq` with states `(node, step)`; under the
  shield every terminal is an exit, where `h = 0`.
* **Counterexamples**: no policy on the node alone escapes earliest from every start
  (`node_policy_fails`: the state of Jabi et al. 2019 and Classic is not enough); masking only the
  nodes untenable on arrival can trap the agent while an escape exists, and the deadline shield
  masks the trapping move (`arrival_mask_traps`).

## What is not covered

* The bucket label-setting of `hopDeadlines`, the breadth-first `earliestSafeEscape` (it is
  `GraphNumerics.L7_bfs_layer_eq_earliest` for one target) and the loops of the learner as
  programs; the tie-breaking by fire-free metres; floating point (all quantities here are
  whole steps, as in the code, except `ig`, whose ceiling the code takes).
* The number of trials the learner needs on a given building: `bounded_raises` gives a bound
  (the sum over states of the exact value minus the initial one), far above what the study
  measures.
* Any fire that is not known in advance: the shield is computed from the forecast, and is only
  as safe as the forecast is not optimistic.
-/

namespace FireEgress.SafeEscape

open Finset

/-- A fire on the hop clock (`rlSafe.js`): the open links `nb v`, the exits `ex`, the last step at
which a node may be entered `opn v` (`openUntil`), and the step limit `H`. -/
structure Net (V : Type*) where
  nb : V → Finset V
  ex : V → Bool
  opn : V → ℕ
  H : ℕ

variable {V : Type*}

/-- `Esc G t v k`: standing at `v` at step `t`, an exit is reached safely in exactly `k` moves by
step `H`. Each move follows an open link, arrives at `t + 1 ≤ opn w` and `t + 1 ≤ H`; an exit
ends the walk (no move leaves it). -/
inductive Esc (G : Net V) : ℕ → V → ℕ → Prop
  | here {t : ℕ} {v : V} : G.ex v = true → Esc G t v 0
  | step {t : ℕ} {v w : V} {k : ℕ} : G.ex v = false → w ∈ G.nb v → t + 1 ≤ G.opn w →
      t + 1 ≤ G.H → Esc G (t + 1) w k → Esc G t v (k + 1)

/-- Finite-horizon value iteration (`escapeValues`), `n` sweeps back from the horizon: the table
at step `H - n`. An exit is worth `0`; off the exits the horizon is worth `⊤` (no escape); a
sweep takes the least `1 + bi G n w` over open links whose arrival, at step `H - n`, is safe. -/
noncomputable def bi (G : Net V) : ℕ → V → ℕ∞
  | 0, v => if G.ex v then 0 else ⊤
  | n + 1, v => if G.ex v then 0 else
      (G.nb v).inf fun w => if G.H - n ≤ G.opn w then bi G n w + 1 else ⊤

variable (G : Net V)

/-! ## Escapes -/

theorem esc_zero {t : ℕ} {v : V} (h : Esc G t v 0) : G.ex v = true := by
  cases h with
  | here hx => exact hx

/-- An escape of `k ≥ 1` moves from step `t` ends by the step limit. -/
theorem esc_time {t : ℕ} {v : V} {k : ℕ} (h : Esc G t v k) : k = 0 ∨ t + k ≤ G.H := by
  induction h with
  | here _ => exact Or.inl rfl
  | @step t v w k _ _ _ hH _ ih =>
    right
    rcases ih with h | h <;> omega

theorem esc_inv {t : ℕ} {v : V} {k : ℕ} (h : Esc G t v k) (hv : G.ex v = false) :
    ∃ w ∈ G.nb v, ∃ k', k = k' + 1 ∧ t + 1 ≤ G.opn w ∧ t + 1 ≤ G.H ∧ Esc G (t + 1) w k' := by
  cases h with
  | here hx => rw [hx] at hv; exact absurd hv (by decide)
  | @step _ _ w k' _ hw ho hH hk => exact ⟨w, hw, k', rfl, ho, hH, hk⟩

/-- Leaving earlier never hurts: an escape from step `t` is an escape from any earlier step,
since every arrival comes earlier and a node is open up to its closing step. -/
theorem esc_mono {t : ℕ} {v : V} {k : ℕ} (h : Esc G t v k) : ∀ s ≤ t, Esc G s v k := by
  induction h with
  | here hx => intro s _; exact Esc.here hx
  | @step t v w k hv hw ho hH _ ih =>
    intro s hs
    exact Esc.step hv hw (by omega) (by omega) (ih (s + 1) (by omega))

/-! ## Value iteration is the earliest safe escape -/

theorem bi_exit {n : ℕ} {v : V} (hx : G.ex v = true) : bi G n v = 0 := by
  cases n <;> simp [bi, hx]

theorem bi_zero {v : V} (hv : G.ex v = false) : bi G 0 v = ⊤ := by
  simp [bi, hv]

theorem bi_succ {n : ℕ} {v : V} (hv : G.ex v = false) :
    bi G (n + 1) v = (G.nb v).inf fun w => if G.H - n ≤ G.opn w then bi G n w + 1 else ⊤ := by
  simp [bi, hv]

/-- After `n` sweeps the table at step `t = H - n` is a lower bound on every escape's length,
and each finite value is the length of an escape. -/
theorem bi_spec (n : ℕ) : ∀ t v, t + n = G.H →
    (∀ k, Esc G t v k → bi G n v ≤ k) ∧ (∀ k : ℕ, bi G n v = k → Esc G t v k) := by
  induction n with
  | zero =>
    intro t v ht
    refine ⟨fun k hk => ?_, fun k hk => ?_⟩
    · cases hk with
      | here hx => rw [bi_exit G hx]; exact zero_le
      | step _ _ _ hH _ => omega
    · cases hx : G.ex v
      · rw [bi_zero G hx] at hk
        exact absurd hk (by simp)
      · rw [bi_exit G hx] at hk
        have : k = 0 := by exact_mod_cast hk.symm
        subst this
        exact Esc.here hx
  | succ n ih =>
    intro t v ht
    refine ⟨fun k hk => ?_, fun k hk => ?_⟩
    · cases hk with
      | here hx => rw [bi_exit G hx]; exact zero_le
      | @step _ _ w k' hv hw ho hH hk' =>
        have h1 := (ih (t + 1) w (by omega)).1 k' hk'
        have ho' : G.H - n ≤ G.opn w := by omega
        rw [bi_succ G hv]
        calc (G.nb v).inf (fun w => if G.H - n ≤ G.opn w then bi G n w + 1 else ⊤)
            ≤ (if G.H - n ≤ G.opn w then bi G n w + 1 else ⊤) := Finset.inf_le hw
          _ = bi G n w + 1 := by simp only [ho', ↓reduceIte]
          _ ≤ (k' : ℕ∞) + 1 := add_le_add h1 le_rfl
          _ = ((k' + 1 : ℕ) : ℕ∞) := (Nat.cast_succ k').symm
    · cases hx : G.ex v
      · rw [bi_succ G hx] at hk
        rcases (G.nb v).eq_empty_or_nonempty with he | hne
        · rw [he, Finset.inf_empty] at hk
          exact absurd hk (by simp)
        · obtain ⟨w, hw, hwe⟩ := Finset.exists_mem_eq_inf (G.nb v) hne
            (fun w => if G.H - n ≤ G.opn w then bi G n w + 1 else ⊤)
          rw [hwe] at hk
          by_cases ho : G.H - n ≤ G.opn w
          · simp only [ho, ↓reduceIte] at hk
            by_cases htop : bi G n w = ⊤
            · rw [htop] at hk
              exact absurd hk (by simp)
            · obtain ⟨j, hj⟩ := ENat.ne_top_iff_exists.mp htop
              rw [← hj] at hk
              have hkj : k = j + 1 := by exact_mod_cast hk.symm
              subst hkj
              exact Esc.step hx hw (by omega) (by omega) ((ih (t + 1) w (by omega)).2 j hj.symm)
          · simp only [ho, ↓reduceIte] at hk
            exact absurd hk (by simp)
      · rw [bi_exit G hx] at hk
        have : k = 0 := by exact_mod_cast hk.symm
        subst this
        exact Esc.here hx

/-- Value iteration never exceeds an escape's length. -/
theorem vi_le {t : ℕ} {v : V} {k : ℕ} (ht : t ≤ G.H) (h : Esc G t v k) :
    bi G (G.H - t) v ≤ k :=
  (bi_spec G (G.H - t) t v (by omega)).1 k h

/-- Each finite value is the length of an escape. -/
theorem vi_esc {t : ℕ} {v : V} {k : ℕ} (ht : t ≤ G.H) (h : bi G (G.H - t) v = k) :
    Esc G t v k :=
  (bi_spec G (G.H - t) t v (by omega)).2 k h

/-- The value is `⊤` exactly when no escape is left (`escapeValues`' `NEVER`). -/
theorem vi_top_iff {t : ℕ} {v : V} (ht : t ≤ G.H) :
    bi G (G.H - t) v = ⊤ ↔ ¬ ∃ k, Esc G t v k := by
  constructor
  · rintro htop ⟨k, hk⟩
    have := vi_le G ht hk
    rw [htop] at this
    exact absurd this (by simp)
  · intro hno
    by_contra htop
    obtain ⟨j, hj⟩ := ENat.ne_top_iff_exists.mp htop
    exact hno ⟨j, vi_esc G ht hj.symm⟩

open Classical in
/-- Finite-horizon value iteration on the time-expanded graph returns the earliest safe escape:
the fewest moves of any safe walk to an exit (`escapeValues` equals `earliestSafeEscape`). -/
theorem vi_eq_earliest {t : ℕ} {v : V} (ht : t ≤ G.H) (h : ∃ k, Esc G t v k) :
    bi G (G.H - t) v = Nat.find h := by
  apply le_antisymm (vi_le G ht (Nat.find_spec h))
  have htop : bi G (G.H - t) v ≠ ⊤ := fun e => (vi_top_iff G ht).1 e h
  obtain ⟨j, hj⟩ := ENat.ne_top_iff_exists.mp htop
  rw [← hj]
  exact_mod_cast Nat.find_min' h (vi_esc G ht hj.symm)

/-- Leaving later never escapes sooner: the exact value never falls with the step. This is what
lets `StepTable` apply a bound learned at one step to every later one. -/
theorem val_mono_time {t s : ℕ} {v : V} (hts : t ≤ s) (hs : s ≤ G.H) :
    bi G (G.H - t) v ≤ bi G (G.H - s) v := by
  by_cases htop : bi G (G.H - s) v = ⊤
  · rw [htop]
    exact le_top
  · obtain ⟨j, hj⟩ := ENat.ne_top_iff_exists.mp htop
    rw [← hj]
    exact vi_le G (by omega) (esc_mono G (vi_esc G hs hj.symm) t hts)

/-! ## The deadline map -/

/-- Arrived at `w` at step `t` safely, with an escape left: what `t ≤ D w` means. -/
def AV (t : ℕ) (w : V) : Prop := t ≤ G.opn w ∧ ∃ k, Esc G t w k

/-- The recursion `hopDeadlines` solves, with `-1` for "never" and the maximum over no links
`-1`: `D x = opn x` at an exit, `D v = max (-1) (min (opn v) (max_w D w - 1))` elsewhere. -/
def IsDeadline (D : V → ℤ) : Prop :=
  (∀ x, G.ex x = true → D x = G.opn x) ∧
    ∀ v, G.ex v = false → D v = max (-1) (min (G.opn v : ℤ) ((G.nb v).fold max (-1) D - 1))

theorem deadline_le_H (hopn : ∀ v, G.opn v ≤ G.H) {D : V → ℤ} (hD : IsDeadline G D) (v : V) :
    D v ≤ G.H := by
  have h := hopn v
  cases hx : G.ex v
  · rw [hD.2 v hx]
    refine max_le (by omega) ((min_le_left _ _).trans (by exact_mod_cast h))
  · rw [hD.1 v hx]
    exact_mod_cast h

/-- The deadline map is exact: for every step `t`, `t ≤ D v` iff `v` can be entered at `t`
safely with an escape left. Any solution of the recursion is this map, so the code's output,
which the tests check against the recursion, is the deadline. -/
theorem deadline_iff (hopn : ∀ v, G.opn v ≤ G.H) {D : V → ℤ} (hD : IsDeadline G D) :
    ∀ (t : ℕ) (v : V), ((t : ℤ) ≤ D v ↔ AV G t v) := by
  have past : ∀ t : ℕ, G.H < t → ∀ v, ((t : ℤ) ≤ D v ↔ AV G t v) := by
    intro t ht v
    constructor
    · intro h
      have := deadline_le_H G hopn hD v
      omega
    · rintro ⟨h, -⟩
      have := hopn v
      omega
  have key : ∀ n t : ℕ, G.H < t + n → ∀ v, ((t : ℤ) ≤ D v ↔ AV G t v) := by
    intro n
    induction n with
    | zero => intro t ht v; exact past t (by omega) v
    | succ n ih =>
      intro t ht v
      by_cases htH : G.H < t
      · exact past t htH v
      cases hx : G.ex v
      · -- Off the exits: the level set of the max-min recursion.
        have ih' : ∀ w, (((t + 1 : ℕ) : ℤ) ≤ D w ↔ AV G (t + 1) w) := ih (t + 1) (by omega)
        rw [hD.2 v hx, le_max_iff, le_min_iff]
        have h2 : ((t : ℤ) ≤ (G.nb v).fold max (-1) D - 1) ↔
            ∃ w ∈ G.nb v, AV G (t + 1) w := by
          rw [show ((t : ℤ) ≤ (G.nb v).fold max (-1) D - 1) ↔
              (t : ℤ) + 1 ≤ (G.nb v).fold max (-1) D by omega, Finset.le_fold_max]
          constructor
          · rintro (h | ⟨w, hw, h⟩)
            · omega
            · exact ⟨w, hw, (ih' w).1 (by push_cast; exact h)⟩
          · rintro ⟨w, hw, h⟩
            exact Or.inr ⟨w, hw, by have := (ih' w).2 h; push_cast at this; exact this⟩
        rw [h2]
        constructor
        · rintro (h | ⟨h, w, hw, hav, k, hk⟩)
          · omega
          · exact ⟨by exact_mod_cast h, k + 1,
              Esc.step hx hw hav (le_trans hav (hopn w)) hk⟩
        · rintro ⟨h, k, hk⟩
          obtain ⟨w, hw, k', -, ho, -, hk'⟩ := esc_inv G hk hx
          exact Or.inr ⟨by exact_mod_cast h, w, hw, ho, k', hk'⟩
      · rw [hD.1 v hx]
        constructor
        · intro h
          exact ⟨by exact_mod_cast h, 0, Esc.here hx⟩
        · rintro ⟨h, -⟩
          exact_mod_cast h
  intro t v
  exact key (G.H + 1) t (by omega) v

/-! ## The shield -/

section Shield

variable (hopn : ∀ v, G.opn v ≤ G.H) {D : V → ℤ} (hD : IsDeadline G D)
include hopn hD

/-- The shield allows the move into `w` at step `t` exactly when `w` can be entered at `t + 1`
safely with an escape left. -/
theorem shield_iff (t : ℕ) (w : V) : ((t : ℤ) + 1 ≤ D w) ↔ AV G (t + 1) w := by
  have := deadline_iff G hopn hD (t + 1) w
  push_cast at this
  exact this

/-- An allowed move is safe on arrival. -/
theorem shield_safe {t : ℕ} {w : V} (h : (t : ℤ) + 1 ≤ D w) : t + 1 ≤ G.opn w :=
  ((shield_iff G hopn hD t w).1 h).1

/-- An allowed move keeps an escape. -/
theorem shield_keeps {t : ℕ} {w : V} (h : (t : ℤ) + 1 ≤ D w) : ∃ k, Esc G (t + 1) w k :=
  ((shield_iff G hopn hD t w).1 h).2

/-- The shield removes no escape: a move through which some escape goes is allowed. -/
theorem shield_complete {t : ℕ} {w : V} {k : ℕ} (ho : t + 1 ≤ G.opn w) (hk : Esc G (t + 1) w k) :
    (t : ℤ) + 1 ≤ D w :=
  (shield_iff G hopn hD t w).2 ⟨ho, k, hk⟩

/-- Until an exit is reached, some move is allowed. -/
theorem shield_nonempty {t : ℕ} {v : V} {k : ℕ} (hk : Esc G t v k) (hv : G.ex v = false) :
    ∃ w ∈ G.nb v, (t : ℤ) + 1 ≤ D w := by
  obtain ⟨w, hw, k', -, ho, -, hk'⟩ := esc_inv G hk hv
  exact ⟨w, hw, shield_complete G hopn hD ho hk'⟩

/-- Every walk inside the shield from a start with an escape reaches an exit by step `H`, and
enters every node before it closes, however its moves are chosen. -/
theorem shield_reaches_exit (p : ℕ → V) (hs : ∃ k, Esc G 0 (p 0) k)
    (hstep : ∀ i, G.ex (p i) = false → p (i + 1) ∈ G.nb (p i) ∧ (i : ℤ) + 1 ≤ D (p (i + 1))) :
    ∃ m ≤ G.H, G.ex (p m) = true ∧ (∀ j < m, G.ex (p j) = false) ∧
      ∀ i, 1 ≤ i → i ≤ m → i ≤ G.opn (p i) := by
  classical
  have viable : ∀ i, (∀ j < i, G.ex (p j) = false) → ∃ k, Esc G i (p i) k := by
    intro i
    induction i with
    | zero => intro _; exact hs
    | succ i ih =>
      intro hj
      exact shield_keeps G hopn hD (hstep i (hj i (by omega))).2
  have hexists : ∃ m, m ≤ G.H ∧ G.ex (p m) = true := by
    by_contra hno
    have hall : ∀ j, j ≤ G.H → G.ex (p j) = false := fun j hj => by
      cases h : G.ex (p j)
      · rfl
      · exact absurd ⟨j, hj, h⟩ hno
    obtain ⟨k, hk⟩ := viable G.H (fun j hj => hall j (by omega))
    have hH := hall G.H le_rfl
    rcases esc_time G hk with h0 | h1
    · subst h0
      rw [esc_zero G hk] at hH
      exact absurd hH (by decide)
    · have hk0 : k ≠ 0 := by
        rintro rfl
        rw [esc_zero G hk] at hH
        exact absurd hH (by decide)
      omega
  have hm := Nat.find_spec hexists
  have hmin : ∀ j < Nat.find hexists, G.ex (p j) = false := fun j hj => by
    have h := Nat.find_min hexists hj
    cases hx : G.ex (p j)
    · rfl
    · exact absurd ⟨by omega, hx⟩ h
  refine ⟨Nat.find hexists, hm.1, hm.2, hmin, fun i hi1 him => ?_⟩
  obtain ⟨i', rfl⟩ : ∃ i', i = i' + 1 := ⟨i - 1, by omega⟩
  exact shield_safe G hopn hD (hstep i' (hmin i' (by omega))).2

end Shield

/-! ## The learner: learning real-time search on `(node, step)` -/

/-- A table of lower bounds on the steps left, read at `(step, node)` (`learnSafeEscape`'s `get`:
the initial potential, raised by learning; `0` at exits) is admissible when it never exceeds the
exact value. -/
def Admissible (Q : ℕ → V → ℕ∞) : Prop := ∀ t ≤ G.H, ∀ v, Q t v ≤ bi G (G.H - t) v

/-- A potential that is `0` at the exits and falls by at most one along any move an escape can
make is admissible: the fire-free hop distance to the nearest exit, and the hop distance inside
the deadline map's viable nodes (`exitHops` with `deadline`), are such potentials. -/
theorem consistent_admissible (h : V → ℕ) (hx : ∀ x, G.ex x = true → h x = 0)
    (hc : ∀ t v w k, G.ex v = false → w ∈ G.nb v → t + 1 ≤ G.opn w → Esc G (t + 1) w k →
      h v ≤ h w + 1) :
    Admissible G (fun _ v => (h v : ℕ∞)) := by
  have lb : ∀ t v k, Esc G t v k → h v ≤ k := by
    intro t v k hk
    induction hk with
    | here hx' => simp [hx _ hx']
    | @step t v w k hv hw ho _ hk ih =>
      have := hc t v w k hv hw ho hk
      omega
  intro t ht v
  by_cases htop : bi G (G.H - t) v = ⊤
  · rw [htop]
    exact le_top
  · obtain ⟨j, hj⟩ := ENat.ne_top_iff_exists.mp htop
    rw [← hj]
    show (h v : ℕ∞) ≤ j
    exact_mod_cast lb t v j (vi_esc G ht hj.symm)

/-- The learner's update keeps the table admissible: any new value at most the least
`1 + V (t + 1) w` over the moves the shield allows is at most the exact value. -/
theorem lrta_admissible (hopn : ∀ v, G.opn v ≤ G.H) {D : V → ℤ} (hD : IsDeadline G D)
    {Q : ℕ → V → ℕ∞} (hQ : Admissible G Q) {t : ℕ} (ht : t < G.H) {v : V}
    (hv : G.ex v = false) {c : ℕ∞}
    (hc : c ≤ ((G.nb v).filter (fun w => (t : ℤ) + 1 ≤ D w)).inf (fun w => Q (t + 1) w + 1)) :
    c ≤ bi G (G.H - t) v := by
  by_cases htop : bi G (G.H - t) v = ⊤
  · rw [htop]
    exact le_top
  obtain ⟨j, hj⟩ := ENat.ne_top_iff_exists.mp htop
  obtain ⟨w, hw, k', hjk, ho, hH, hk⟩ := esc_inv G (vi_esc G ht.le hj.symm) hv
  have hallow : (t : ℤ) + 1 ≤ D w := shield_complete G hopn hD ho hk
  calc c ≤ _ := hc
    _ ≤ Q (t + 1) w + 1 := Finset.inf_le (Finset.mem_filter.2 ⟨hw, hallow⟩)
    _ ≤ bi G (G.H - (t + 1)) w + 1 := add_le_add (hQ (t + 1) (by omega) w) le_rfl
    _ ≤ (k' : ℕ∞) + 1 := add_le_add (vi_le G (by omega) hk) le_rfl
    _ = bi G (G.H - t) v := by rw [← hj, hjk, Nat.cast_succ]

/-- Raising the bound at `(t, v)` to `c` at step `t` and every later step (`StepTable.raise`). -/
noncomputable def raise [DecidableEq V] (Q : ℕ → V → ℕ∞) (t : ℕ) (v : V) (c : ℕ∞) : ℕ → V → ℕ∞ :=
  fun s u => if u = v ∧ t ≤ s then max (Q s u) c else Q s u

/-- Applying a bound learned at step `t` to every later step keeps the table admissible, because
the exact value never falls with the step (`val_mono_time`). -/
theorem raise_admissible [DecidableEq V] {Q : ℕ → V → ℕ∞} (hQ : Admissible G Q) {t : ℕ} {v : V}
    {c : ℕ∞} (hc : c ≤ bi G (G.H - t) v) : Admissible G (raise Q t v c) := by
  intro s hs u
  unfold raise
  split_ifs with h
  · obtain ⟨rfl, hts⟩ := h
    exact max_le (hQ s hs u) (hc.trans (val_mono_time G hts hs))
  · exact hQ s hs u

/-- A walk that follows open links, arrives everywhere before closing and by the step limit, and
ends at an exit, is an escape. -/
theorem walk_esc (p : ℕ → V) (m : ℕ) (hx : G.ex (p m) = true)
    (hw : ∀ i < m, G.ex (p i) = false ∧ p (i + 1) ∈ G.nb (p i) ∧ i + 1 ≤ G.opn (p (i + 1)) ∧
      i + 1 ≤ G.H) :
    Esc G 0 (p 0) m := by
  have key : ∀ d, d ≤ m → Esc G (m - d) (p (m - d)) d := by
    intro d
    induction d with
    | zero => intro _; rw [Nat.sub_zero]; exact Esc.here hx
    | succ d ih =>
      intro hd
      obtain ⟨h1, h2, h3, h4⟩ := hw (m - (d + 1)) (by omega)
      have e : m - (d + 1) + 1 = m - d := by omega
      rw [e] at h2 h3
      refine Esc.step (w := p (m - d)) h1 h2 (by omega) (by omega) ?_
      rw [e]
      exact ih (by omega)
  simpa using key m le_rfl

/-- The certificate: a safe escape of `m` moves with `m` at most an admissible value at the start
is an earliest escape. -/
theorem certificate_optimal {Q : ℕ → V → ℕ∞} (hQ : Admissible G Q) {s : V} {m : ℕ}
    (hm : Esc G 0 s m) (hle : (m : ℕ∞) ≤ Q 0 s) :
    bi G G.H s = m ∧ ∀ k, Esc G 0 s k → m ≤ k := by
  have h1 : bi G (G.H - 0) s ≤ m := vi_le G (Nat.zero_le _) hm
  have h2 : Q 0 s ≤ bi G (G.H - 0) s := hQ 0 (Nat.zero_le _) s
  rw [Nat.sub_zero] at h1 h2
  refine ⟨le_antisymm h1 (hle.trans h2), fun k hk => ?_⟩
  have h3 := vi_le G (Nat.zero_le _) hk
  rw [Nat.sub_zero] at h3
  exact_mod_cast hle.trans (h2.trans h3)

/-- A trial that changes no value (at each step the stored value is already at least one more
than the value at the next state) has the start's value at least its length, so it passes the
certificate's test. -/
theorem trial_no_change (Q : ℕ → V → ℕ∞) (p : ℕ → V) (m : ℕ)
    (hstep : ∀ i < m, Q (i + 1) (p (i + 1)) + 1 ≤ Q i (p i)) : (m : ℕ∞) ≤ Q 0 (p 0) := by
  have key : ∀ d, d ≤ m → (d : ℕ∞) ≤ Q (m - d) (p (m - d)) := by
    intro d
    induction d with
    | zero => intro _; simp
    | succ d ih =>
      intro hd
      have h := hstep (m - (d + 1)) (by omega)
      have e : m - (d + 1) + 1 = m - d := by omega
      rw [e] at h
      calc ((d + 1 : ℕ) : ℕ∞) = (d : ℕ∞) + 1 := Nat.cast_succ d
        _ ≤ Q (m - d) (p (m - d)) + 1 := add_le_add (ih (by omega)) le_rfl
        _ ≤ _ := h
  simpa using key m le_rfl

/-- Tables of whole numbers that only rise and stay below `B` change at most
`∑ (B - Q 0)` times. With `trial_no_change` (a trial that is not certified changes the table)
and the cap `H + 1` on every value, learning stops with the certificate after finitely many
trials. -/
theorem bounded_raises {S : Type*} [Fintype S] (Q : ℕ → S → ℕ) (B : S → ℕ)
    (hmono : ∀ j s, Q j s ≤ Q (j + 1) s) (hB : ∀ j s, Q j s ≤ B s) (N : ℕ) :
    ((Finset.range N).filter (fun j => Q (j + 1) ≠ Q j)).card ≤ ∑ s, (B s - Q 0 s) := by
  have key : ∀ N, ((Finset.range N).filter (fun j => Q (j + 1) ≠ Q j)).card + ∑ s, Q 0 s ≤
      ∑ s, Q N s := by
    intro N
    induction N with
    | zero => simp
    | succ N ih =>
      have hle : ∑ s, Q N s ≤ ∑ s, Q (N + 1) s := Finset.sum_le_sum fun s _ => hmono N s
      rw [Finset.range_add_one, Finset.filter_insert]
      split_ifs with hne
      · have hlt : ∑ s, Q N s < ∑ s, Q (N + 1) s := by
          obtain ⟨s, hs⟩ := Function.ne_iff.mp hne
          exact Finset.sum_lt_sum (fun s _ => hmono N s)
            ⟨s, Finset.mem_univ _, lt_of_le_of_ne (hmono N s) (Ne.symm hs)⟩
        have hcard := Finset.card_insert_le N
          ((Finset.range N).filter (fun j => Q (j + 1) ≠ Q j))
        omega
      · omega
  have hsum : ∑ s, (B s - Q 0 s) + ∑ s, Q 0 s = ∑ s, B s := by
    rw [← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun s _ => Nat.sub_add_cancel (hB 0 s)
  have hle : ∑ s, Q N s ≤ ∑ s, B s := Finset.sum_le_sum fun s _ => hB N s
  have := key N
  omega

/-- Initialising the values with a potential `h` is the same update as shaping the costs: with
values `q - h` and move cost `1 + h w - h v` the target is the unshaped target less `h v`
(Wiewiora 2003, here for the deterministic minimum-cost update). -/
theorem potential_init_eq_shaping {ι : Type*} (s : Finset ι) (hs : s.Nonempty) (q h : ι → ℝ)
    (hv : ℝ) :
    s.inf' hs (fun w => (q w - h w) + (1 + h w - hv)) = s.inf' hs (fun w => q w + 1) - hv := by
  have e : (fun w => (q w - h w) + (1 + h w - hv)) = fun w => (q w + 1) - hv := by
    funext w
    ring
  rw [e]
  apply le_antisymm
  · obtain ⟨w, hw, he⟩ := Finset.exists_mem_eq_inf' hs (fun w => q w + 1)
    rw [he]
    exact Finset.inf'_le (fun w => (q w + 1) - hv) hw
  · refine Finset.le_inf' hs _ fun w hw => ?_
    have := Finset.inf'_le (fun w => q w + 1) hw
    linarith

/-! ## Counterexamples -/

section NodeOnly

/-- Two starts that pass one node at different steps. `s1 - v`, `s2 - a - v`, `v - x1` (an exit
that closes at step 2), `v - b - x2` (an exit that stays open). -/
inductive N7
  | s1 | s2 | a | v | x1 | b | x2
  deriving DecidableEq

/-- The graph of `node_policy_fails` (`rlSafe.test.mjs` builds the same with a fire). -/
def G7 : Net N7 where
  nb := fun u => match u with
    | .s1 => {.v}
    | .s2 => {.a}
    | .a => {.s2, .v}
    | .v => {.s1, .a, .x1, .b}
    | .x1 => {.v}
    | .b => {.v, .x2}
    | .x2 => {.b}
  ex := fun u => match u with
    | .x1 => true
    | .x2 => true
    | _ => false
  opn := fun u => match u with
    | .x1 => 2
    | _ => 10
  H := 10

/-- The walk of a policy on the node alone (Jabi et al. 2019, Classic): `π` gives the next node. -/
def pw (π : N7 → N7) (s : N7) : ℕ → N7
  | 0 => s
  | i + 1 => π (pw π s i)

/-- `π` takes `s` to an exit in `m` moves, each arriving before its node closes. -/
def EscBy (π : N7 → N7) (s : N7) (m : ℕ) : Prop :=
  G7.ex (pw π s m) = true ∧ ∀ i < m, G7.ex (pw π s i) = false ∧ i + 1 ≤ G7.opn (pw π s (i + 1))

/-- No policy on the node alone is right from every start. From `s1` the earliest escape takes 2
moves, through `x1` at step 2; from `s2` an escape exists (4 moves). At `(v, 1)` the move to `x1`
escapes at once, at `(v, 2)` it is too late. So any node-only policy that escapes earliest from
`s1` sends `v` to `x1`, and then never escapes from `s2`; a table on `(node, step)` holds both. -/
theorem node_policy_fails :
    Esc G7 0 .s1 2 ∧ (∀ k, Esc G7 0 .s1 k → 2 ≤ k) ∧ Esc G7 0 .s2 4 ∧
      Esc G7 1 .v 1 ∧ ¬ Esc G7 2 .v 1 ∧
      ∀ π : N7 → N7, (∀ u, π u ∈ G7.nb u) → EscBy π .s1 2 → ∀ m, ¬ EscBy π .s2 m := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · exact Esc.step (w := .v) rfl (by decide) (by decide) (by decide)
      (Esc.step (w := .x1) rfl (by decide) (by decide) (by decide) (Esc.here rfl))
  · intro k hk
    cases hk with
    | here hx => exact absurd hx (by decide)
    | @step _ _ w k' _ hw _ _ hk' =>
      have hwv : w = .v := by simpa [G7] using hw
      subst hwv
      rcases Nat.eq_zero_or_pos k' with h0 | hpos
      · subst h0
        exact absurd (esc_zero G7 hk') (by decide)
      · omega
  · exact Esc.step (w := .a) rfl (by decide) (by decide) (by decide)
      (Esc.step (w := .v) rfl (by decide) (by decide) (by decide)
        (Esc.step (w := .b) rfl (by decide) (by decide) (by decide)
          (Esc.step (w := .x2) rfl (by decide) (by decide) (by decide) (Esc.here rfl))))
  · exact Esc.step (w := .x1) rfl (by decide) (by decide) (by decide) (Esc.here rfl)
  · intro h
    cases h with
    | @step _ _ w k' _ hw ho _ hk' =>
      have hx := esc_zero G7 hk'
      have key : ∀ u ∈ G7.nb .v, G7.ex u = true → ¬ (2 + 1 ≤ G7.opn u) := by decide
      exact key w hw hx ho
  · intro π hπ h1 m h2
    have hs1 : π .s1 = .v := by simpa [G7] using hπ .s1
    have hs2 : π .s2 = .a := by simpa [G7] using hπ .s2
    have hvx : π .v = .x1 := by
      have hx : G7.ex (π (π .s1)) = true := h1.1
      rw [hs1] at hx
      have key : ∀ u ∈ G7.nb .v, G7.ex u = true → u = .x1 := by decide
      exact key _ (hπ .v) hx
    have ha : π .a = .s2 ∨ π .a = .v := by
      have := hπ .a
      simpa [G7] using this
    rcases ha with ha | ha
    · -- The walk from s2 shuttles between s2 and a and never reaches an exit.
      have loop : ∀ i, pw π .s2 i = .s2 ∨ pw π .s2 i = .a := by
        intro i
        induction i with
        | zero => exact Or.inl rfl
        | succ i ih =>
          rcases ih with h | h
          · right
            show π (pw π .s2 i) = .a
            rw [h, hs2]
          · left
            show π (pw π .s2 i) = .s2
            rw [h, ha]
      rcases loop m with h | h
      · have := h2.1
        rw [h] at this
        exact absurd this (by decide)
      · have := h2.1
        rw [h] at this
        exact absurd this (by decide)
    · -- The walk from s2 reaches v at step 2 and x1 at step 3, after x1 closes.
      have p0 : pw π .s2 0 = .s2 := rfl
      have p1 : pw π .s2 1 = .a := by show π .s2 = .a; exact hs2
      have p2 : pw π .s2 2 = .v := by show π (π .s2) = .v; rw [hs2, ha]
      have p3 : pw π .s2 3 = .x1 := by show π (π (π .s2)) = .x1; rw [hs2, ha, hvx]
      by_cases hm : 3 < m
      · have := (h2.2 3 hm).1
        rw [p3] at this
        exact absurd this (by decide)
      · interval_cases m
        · have := h2.1
          rw [p0] at this
          exact absurd this (by decide)
        · have := h2.1
          rw [p1] at this
          exact absurd this (by decide)
        · have := h2.1
          rw [p2] at this
          exact absurd this (by decide)
        · have : 2 + 1 ≤ G7.opn (pw π .s2 3) := (h2.2 2 (by norm_num)).2
          rw [p3] at this
          exact absurd this (by decide)

end NodeOnly

section ArrivalMask

/-- `s - p`, `s - x` (the exit): the fire closes `s` after step 1 and `p` and `x` after step 2. -/
inductive N3
  | s | p | x
  deriving DecidableEq

/-- The graph of `arrival_mask_traps` (`rlSafe.test.mjs` builds the same with a fire origin
beside `s`). -/
def G3 : Net N3 where
  nb := fun u => match u with
    | .s => {.p, .x}
    | .p => {.s}
    | .x => {.s}
  ex := fun u => match u with
    | .x => true
    | _ => false
  opn := fun u => match u with
    | .s => 1
    | .p => 2
    | .x => 2
  H := 2

theorem G3_no_escape_from_p (k : ℕ) : ¬ Esc G3 1 .p k := by
  intro hk
  cases hk with
  | here hx => exact absurd hx (by decide)
  | @step _ _ w k' _ hw ho _ _ =>
    have key : ∀ u ∈ G3.nb .p, ¬ (1 + 1 ≤ G3.opn u) := by decide
    exact key w hw ho

/-- Masking only the nodes untenable on arrival can trap the agent while an escape exists:
from `s` the exit is one move away; the move into `p` is safe on arrival, but at `(p, 1)` every
move is masked, since `s` has closed. The deadline shield masks the move into `p`. -/
theorem arrival_mask_traps :
    Esc G3 0 .s 1 ∧ 0 + 1 ≤ G3.opn .p ∧ (∀ w ∈ G3.nb .p, ¬ (1 + 1 ≤ G3.opn w)) ∧
      (¬ ∃ k, Esc G3 1 .p k) ∧
      ∀ D : N3 → ℤ, IsDeadline G3 D → ¬ ((0 : ℤ) + 1 ≤ D .p) := by
  refine ⟨Esc.step (w := .x) rfl (by decide) (by decide) (by decide) (Esc.here rfl),
    by decide, by decide, fun ⟨k, hk⟩ => G3_no_escape_from_p k hk, ?_⟩
  intro D hD h
  have hopn : ∀ u, G3.opn u ≤ G3.H := by intro u; cases u <;> decide
  obtain ⟨-, k, hk⟩ := (shield_iff G3 hopn hD 0 .p).1 (by exact_mod_cast h)
  exact G3_no_escape_from_p k hk

end ArrivalMask

section LegacyReward

/-- `rl.js`'s reward can make staying inside optimal while an escape exists. With its discount
`γ = 0.99`, an escape of 122 moves along links of the median length pays `-0.1` for each of the
first 121 moves and `+10` for the last, a return of `-10 + 20 γ^121 ≈ -4.09`; walking back and
forth over a link a quarter of the median length until the 200-step limit pays `-0.025` per move,
`≈ -2.17`. So the exact optimum of the shipped reward does not escape, as the study measured on
one corpus model (an escape of 122 moves, an optimum of `-2.28` that stays inside). An escape
of `m` median links returns `-10 + 20 γ^(m-1)`, below zero beyond 70 moves. -/
theorem legacy_reward_prefers_staying :
    -(1 / 10 : ℝ) * ∑ i ∈ range 121, (99 / 100 : ℝ) ^ i + 10 * (99 / 100 : ℝ) ^ 121 <
      -(1 / 40 : ℝ) * ∑ i ∈ range 200, (99 / 100 : ℝ) ^ i := by
  have hg : ∀ n : ℕ, ∑ i ∈ range n, (99 / 100 : ℝ) ^ i = 100 * (1 - (99 / 100 : ℝ) ^ n) := by
    intro n
    rw [geom_sum_eq (by norm_num)]
    ring
  rw [hg, hg]
  have h121 : (99 / 100 : ℝ) ^ 121 < 3 / 10 := by norm_num
  have h200 : 0 ≤ (99 / 100 : ℝ) ^ 200 := by positivity
  linarith

end LegacyReward

end FireEgress.SafeEscape
