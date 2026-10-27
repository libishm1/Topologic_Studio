import Mathlib
import FireEgress.GraphNumerics

/-!
# Time-dependent Dijkstra is exact under FIFO (L10c)

Studio's planned metric-clock router (literature review
`wiki/research/fire-egress-methods-2026-10-27.md`, sections 5.1-5.4, candidate L10c)
routes an evacuee while the fire grows. A link `u → v` entered at time `t` is left at
time `A u v t`; a fire that slows a link makes `A u v t` larger, and a fire that closes
it makes `A u v t = ⊤` (in `WithTop ℝ`; see `GraphNumerics.closeArc`). The router is
Dreyfus' label-setting loop (1969): settle the unsettled vertex of least label `d u`,
then relax every link out of it with `d v ← min (d v) (A u v (d u))`. Kaufman and
Smith (1993) identified FIFO as the condition that makes it exact. Studio does not
implement the router yet, so this file is the specification it must meet.

Two hypotheses on every link carry the whole argument:
* **FIFO** (non-overtaking): `A u v` is monotone, `s ≤ t → A u v s ≤ A u v t`;
* **no time travel**: `t ≤ A u v t`.
The second replaces Dijkstra's "weights are non-negative"; FIFO is new. In the key
step (`selectable_le`) a walk leaves the settled set through a link `y → x` that was
relaxed at the settled label `d y`, and the walk itself reached `y` no earlier than
`d y`. FIFO turns `d y ≤ τ` into `A y x (d y) ≤ A y x τ`.

What is proved:
1. **Earliest arrival.** `Arr E A s t0 v τ`: some walk from `s`, leaving at `t0`,
   reaches `v` at `τ`. It agrees with the list walks of `GraphNumerics`
   (`arr_iff_walk`). The earliest arrival is the infimum (`IsGLB`) of
   `arrivals E A s t0 v`, and `EA` is its `sInf`. Under FIFO and `t ≤ A t`, cycles never
   help (`cycle_never_helps`), so simple walks suffice (`simple_walk_suffices`,
   `isGLB_arrivals_iff_simple`), and on a finite graph the infimum is a minimum
   attained by a simple walk (`ea_walk_exists`). Waiting at nodes never helps either
   (`waiting_never_helps`, the network form of `GraphNumerics.L10b_wait_never_helps`).
   In Bellman form the earliest arrival is the *greatest* labelling satisfying
   `d v ≤ A u v (d u)` (`bellman_subsolution_le`, `ea_bellman`); "least fixed point"
   in the time order is false (`bellman_least_fixed_point_false`).
2. **The loop invariant.** Mirroring `AStar.lean`: `init_inv`, `selectable_le`,
   `selectable_optimal`, `step_inv`, `reachable_inv`; every settled label is the
   earliest arrival (`settled_optimal`, `td_dijkstra_correct`).
3. **Exhaustion.** When the queue runs dry the settled vertices are exactly those with a
   walk, a vertex never reached has no walk, and the output (label, or `⊤` if
   unsettled) is the earliest arrival at every vertex (`td_dijkstra_exhausted`,
   `td_dijkstra_glb`, `never_reached_no_walk`, `td_dijkstra_EA_withTop`). Stopping
   when the least label is `⊤` is also exact (`td_dijkstra_stop_at_top`). On a
   finite graph some run always reaches exhaustion (`td_dijkstra_total`).
4. **FIFO is necessary.** A four-vertex network with one non-FIFO link (it gets faster
   later) on which the loop returns `10` while a walk arrives at `3`
   (`fifo_necessary`); waiting at a node would also reach the exit at `3`. Every run,
   whatever its tie-breaking, returns `10` (`fifo_necessary_every_run`). Without FIFO
   the loop is still sound and complete for reachability, only not earliest
   (`exhausted_without_fifo`).
5. **Replanning.** Re-solving from the evacuee's current node and time, under the
   unchanged forecast, returns the plan's arrival time (`replan_consistent`,
   `replan_same_label`), via `GraphNumerics.L11_suffix_ea`.

Modelling choices, stated so a reviewer can compare with a future implementation:
* The graph is a relation `E u v` with one arrival function `A u v` per ordered pair.
  Parallel links under FIFO can be merged by the pointwise minimum, which is again
  monotone and `≥ t`. A link closed for good can be left out of `E`; a link that
  closes at a time is `A u v t = ⊤` from then on (`closeArc`, `hopArc`).
* Times are any preorder for correctness (`ℝ`, `ℕ` ticks, `WithTop ℝ`, `ℝ≥0∞`); the
  existence results use a linear order, and the `⊤`-valued output needs `OrderTop`.
* No continuity of `A u v` is assumed: closure windows jump to `⊤`.
* The step is a relation, as in `AStar.lean`: it fixes what the code guarantees after
  popping `u` and scanning its links, not the scan order or the heap. Every link into
  an unsettled vertex is relaxed, including links whose arrival is `⊤`; an
  implementation that never queues `⊤` labels stops at the same output
  (`td_dijkstra_stop_at_top`).
* Walks do not wait. Under FIFO this loses nothing (`waiting_never_helps`).
-/

namespace FireEgress.TimeDependent

open Finset FireEgress.GraphNumerics

variable {V α : Type*}

/-! ## (1) Walks and earliest arrival -/

/-- `Arr E A s t0 v τ`: some walk from `s`, leaving at `t0`, along links of `E`, reaches
`v` at time `τ`. A link `u → v` entered at `τ` is left at `A u v τ`. No waiting. -/
inductive Arr (E : V → V → Prop) (A : V → V → α → α) (s : V) (t0 : α) : V → α → Prop
  | nil : Arr E A s t0 s t0
  | snoc {u v : V} {τ : α} : Arr E A s t0 u τ → E u v → Arr E A s t0 v (A u v τ)

/-- A walk's recorded arrival time can be rewritten by an equal expression. -/
theorem Arr.cast {E : V → V → Prop} {A : V → V → α → α} {s : V} {t0 : α} {v : V}
    {τ τ' : α} (h : Arr E A s t0 v τ) (hτ : τ = τ') : Arr E A s t0 v τ' := hτ ▸ h

/-- Extending a recorded walk by a list walk (`GraphNumerics.IsWalk`) gives a recorded
walk to the list walk's end at the list walk's arrival time. -/
theorem arr_of_walk (E : V → V → Prop) (A : V → V → α → α) (s : V) (t0 : α) :
    ∀ (w : List V) (u : V) (τ : α), Arr E A s t0 u τ → IsWalk E u w →
      Arr E A s t0 (wEnd u w) (wArr A u τ w)
  | [], _, _, h, _ => h
  | v :: vs, u, τ, h, hw => arr_of_walk E A s t0 vs v (A u v τ) (h.snoc hw.1) hw.2

/-- The recorded walks of this file are the list walks of `GraphNumerics`: some walk
reaches `v` at `τ` iff some vertex list `w` is a walk of `E` from `s` ending at `v` with
`wArr A s t0 w = τ`. So results stated with `IsWalk`, `wEnd`, `wArr` and `IsEA` (L10a,
L10b, L11, R3) apply to the router's labels unchanged. -/
theorem arr_iff_walk (E : V → V → Prop) (A : V → V → α → α) (s : V) (t0 : α) (v : V)
    (τ : α) :
    Arr E A s t0 v τ ↔ ∃ w, IsWalk E s w ∧ wEnd s w = v ∧ wArr A s t0 w = τ := by
  constructor
  · intro h
    induction h with
    | nil => exact ⟨[], trivial, rfl, rfl⟩
    | @snoc u v τ _ huv ih =>
      obtain ⟨w, hw, he, ha⟩ := ih
      refine ⟨w ++ [v], (IsWalk_append E s w [v]).2 ⟨hw, ?_⟩, ?_, ?_⟩
      · rw [he]; exact ⟨huv, trivial⟩
      · rw [wEnd_append, he]; rfl
      · rw [wArr_append, he, ha]; rfl
  · rintro ⟨w, hw, rfl, rfl⟩
    exact arr_of_walk E A s t0 w s t0 Arr.nil hw

/-- The times at which some walk from `s`, leaving at `t0`, reaches `v`. -/
def arrivals (E : V → V → Prop) (A : V → V → α → α) (s : V) (t0 : α) (v : V) : Set α :=
  {τ | Arr E A s t0 v τ}

/-- Earliest arrival at `v`: the infimum over all walks from `s` leaving at `t0` of the
composed arrival times. With `sInf ∅ = ⊤` (in `WithTop ℝ` or any complete lattice) it is
`⊤` exactly when no walk arrives at a finite time. -/
noncomputable def EA [InfSet α] (E : V → V → Prop) (A : V → V → α → α) (s : V) (t0 : α)
    (v : V) : α :=
  sInf (arrivals E A s t0 v)

/-- No walk arrives before it leaves: if every link takes non-negative time, every
arrival time from `s` is at least the departure time `t0`. -/
theorem arr_ge_start [Preorder α] (E : V → V → Prop) (A : V → V → α → α)
    (hge : ∀ u v, E u v → ∀ t, t ≤ A u v t) (s : V) (t0 : α) {v : V} {τ : α}
    (h : Arr E A s t0 v τ) : t0 ≤ τ := by
  induction h with
  | nil => exact le_rfl
  | @snoc u v τ _ huv ih => exact ih.trans (hge u v huv τ)

/-- An earliest-arrival walk in the sense of `GraphNumerics.IsEA` arrives at the least
element of the arrival set: its arrival time is the earliest arrival. -/
theorem isLeast_of_isEA [Preorder α] (E : V → V → Prop) (A : V → V → α → α) (s : V)
    (t0 : α) (d : V) (w : List V) (h : IsEA E A s t0 d w) :
    IsLeast (arrivals E A s t0 d) (wArr A s t0 w) := by
  obtain ⟨hw, he, hopt⟩ := h
  refine ⟨(arr_iff_walk E A s t0 d _).2 ⟨w, hw, he, rfl⟩, fun τ hτ => ?_⟩
  obtain ⟨w', hw', he', rfl⟩ := (arr_iff_walk E A s t0 d τ).1 hτ
  exact hopt w' hw' he'

/-- Conversely, a walk to `d` whose arrival time is the least arrival time is an
earliest-arrival walk (`GraphNumerics.IsEA`). This turns a router label back into a
route that `L11` can cut. -/
theorem isEA_of_isLeast [Preorder α] (E : V → V → Prop) (A : V → V → α → α) (s : V)
    (t0 : α) (d : V) (w : List V) (hw : IsWalk E s w) (he : wEnd s w = d)
    (h : IsLeast (arrivals E A s t0 d) (wArr A s t0 w)) : IsEA E A s t0 d w :=
  ⟨hw, he, fun w' hw' he' => h.2 ((arr_iff_walk E A s t0 d _).2 ⟨w', hw', he', rfl⟩)⟩

/-! ## Cycles and waiting never help under FIFO -/

/-- The arrival functions met along the walk from `u` visiting `w`, first link first.
Feeding them to `GraphNumerics.pathArr` gives the walk's arrival function. -/
def arcs (A : V → V → α → α) : V → List V → List (α → α)
  | _, [] => []
  | u, v :: vs => A u v :: arcs A v vs

/-- A walk's arrival time is the composition of its links' arrival functions
(`GraphNumerics.pathArr`), so L10a and R1 apply to walks. -/
theorem wArr_eq_pathArr (A : V → V → α → α) :
    ∀ (u : V) (t : α) (w : List V), wArr A u t w = pathArr (arcs A u w) t
  | _, _, [] => rfl
  | u, t, v :: vs => by
    show wArr A v (A u v t) vs = (pathArr (arcs A v vs) ∘ A u v) t
    rw [wArr_eq_pathArr A v (A u v t) vs]
    rfl

/-- Every arrival function met along a walk of `E` is the arrival function of a link of
`E`, so it inherits any property `P` that all links have (FIFO, `t ≤ A t`). -/
theorem arcs_forall (E : V → V → Prop) (A : V → V → α → α) (P : (α → α) → Prop)
    (hP : ∀ u v, E u v → P (A u v)) :
    ∀ (u : V) (w : List V), IsWalk E u w → ∀ B ∈ arcs A u w, P B
  | _, [], _, B, hB => by simp [arcs] at hB
  | u, v :: vs, hw, B, hB => by
    simp only [arcs, List.mem_cons] at hB
    rcases hB with rfl | hB
    · exact hP u v hw.1
    · exact arcs_forall E A P hP v vs hw.2 B hB

/-- L10a for walks: with FIFO links, a walk's arrival time is monotone in its departure
time. For Studio: leaving later along the same route never arrives earlier. Reuses
`GraphNumerics.L10a_fifo_compose`. -/
theorem walk_fifo [Preorder α] (E : V → V → Prop) (A : V → V → α → α)
    (hfifo : ∀ u v, E u v → Monotone (A u v)) (u : V) (w : List V) (hw : IsWalk E u w) :
    Monotone (fun t => wArr A u t w) := by
  intro a b hab
  simp only [wArr_eq_pathArr]
  exact L10a_fifo_compose (arcs A u w) (arcs_forall E A Monotone hfifo u w hw) hab

/-- Extending a walk never makes arrival earlier: if every link takes non-negative time,
a walk from `u` left at `t` arrives no earlier than `t`. This plays the role of
non-negative weights in Dijkstra. Reuses `GraphNumerics.pathArr_ge`. -/
theorem walk_ge [Preorder α] (E : V → V → Prop) (A : V → V → α → α)
    (hge : ∀ u v, E u v → ∀ t, t ≤ A u v t) (u : V) (w : List V) (hw : IsWalk E u w)
    (t : α) : t ≤ wArr A u t w := by
  rw [wArr_eq_pathArr]
  exact pathArr_ge (arcs A u w) (arcs_forall E A (fun B => ∀ t, t ≤ B t) hge u w hw) t

/-- Cycles never help. If the walk `p ++ c ++ q` from `s` goes round a closed walk `c`
at the vertex where `p` ends, cutting `c` out leaves a walk `p ++ q` to the same end that
arrives no later. Needs `t ≤ A t` on the cycle (the cycle is a delay) and FIFO after it
(a delay never helps the rest). For Studio: an evacuee never gains by looping back. -/
theorem cycle_never_helps [Preorder α] (E : V → V → Prop) (A : V → V → α → α)
    (hfifo : ∀ u v, E u v → Monotone (A u v)) (hge : ∀ u v, E u v → ∀ t, t ≤ A u v t)
    (s : V) (t0 : α) (p c q : List V) (hw : IsWalk E s (p ++ c ++ q))
    (hc : wEnd (wEnd s p) c = wEnd s p) :
    IsWalk E s (p ++ q) ∧ wEnd s (p ++ q) = wEnd s (p ++ c ++ q) ∧
      wArr A s t0 (p ++ q) ≤ wArr A s t0 (p ++ c ++ q) := by
  rw [IsWalk_append, IsWalk_append, wEnd_append, hc] at hw
  obtain ⟨⟨hp, hcw⟩, hq⟩ := hw
  refine ⟨(IsWalk_append E s p q).2 ⟨hp, hq⟩, by simp only [wEnd_append, hc], ?_⟩
  simp only [wArr_append, wEnd_append, hc]
  exact walk_fifo E A hfifo _ q hq (walk_ge E A hge _ c hcw _)

/-- A walk that repeats a vertex contains a cycle: if the vertex sequence `s :: w` has a
repeat, then `w = p ++ c ++ q` with `c` a non-empty closed walk at the end of `p`. -/
theorem exists_cycle (s : V) (w : List V) (h : ¬ (s :: w).Nodup) :
    ∃ p c q, w = p ++ c ++ q ∧ c ≠ [] ∧ wEnd (wEnd s p) c = wEnd s p := by
  induction w generalizing s with
  | nil => exact absurd (List.nodup_singleton s) h
  | cons v vs ih =>
    rw [List.nodup_cons, not_and_or, not_not] at h
    rcases h with h | h
    · obtain ⟨l1, l2, hl⟩ := List.append_of_mem h
      refine ⟨[], l1 ++ [s], l2, ?_, by simp, ?_⟩
      · rw [hl]; simp
      · show wEnd s (l1 ++ [s]) = s
        rw [wEnd_append]; rfl
    · obtain ⟨p, c, q, hpcq, hc, hcyc⟩ := ih v h
      exact ⟨v :: p, c, q, by rw [hpcq]; rfl, hc, hcyc⟩

/-- Simple walks suffice, by induction on length: every walk of length at most `n` is
matched by a walk to the same end that visits no vertex twice and arrives no later. -/
theorem exists_simple_walk_le [Preorder α] (E : V → V → Prop) (A : V → V → α → α)
    (hfifo : ∀ u v, E u v → Monotone (A u v)) (hge : ∀ u v, E u v → ∀ t, t ≤ A u v t)
    (s : V) (t0 : α) (n : ℕ) :
    ∀ w : List V, w.length ≤ n → IsWalk E s w →
      ∃ w', IsWalk E s w' ∧ wEnd s w' = wEnd s w ∧ (s :: w').Nodup ∧
        wArr A s t0 w' ≤ wArr A s t0 w := by
  induction n with
  | zero =>
    intro w hl hw
    have hnil : w = [] := List.eq_nil_of_length_eq_zero (by omega)
    subst hnil
    exact ⟨[], hw, rfl, List.nodup_singleton s, le_rfl⟩
  | succ n ih =>
    intro w hl hw
    by_cases hnd : (s :: w).Nodup
    · exact ⟨w, hw, rfl, hnd, le_rfl⟩
    · obtain ⟨p, c, q, rfl, hc, hcyc⟩ := exists_cycle s w hnd
      obtain ⟨hw', he, hle⟩ := cycle_never_helps E A hfifo hge s t0 p c q hw hcyc
      have hc0 : c.length ≠ 0 := fun h0 => hc (List.eq_nil_of_length_eq_zero h0)
      have hlen : (p ++ q).length ≤ n := by
        simp only [List.length_append] at hl ⊢
        omega
      obtain ⟨w', h1, h2, h3, h4⟩ := ih (p ++ q) hlen hw'
      exact ⟨w', h1, h2.trans he, h3, h4.trans hle⟩

/-- Simple walks suffice. Under FIFO and `t ≤ A t`, every walk from `s` is matched by a
walk to the same vertex that visits no vertex twice and arrives no later. For Studio: the
router may search simple routes only, and a brute-force check may enumerate simple
walks. -/
theorem simple_walk_suffices [Preorder α] (E : V → V → Prop) (A : V → V → α → α)
    (hfifo : ∀ u v, E u v → Monotone (A u v)) (hge : ∀ u v, E u v → ∀ t, t ≤ A u v t)
    (s : V) (t0 : α) (w : List V) (hw : IsWalk E s w) :
    ∃ w', IsWalk E s w' ∧ wEnd s w' = wEnd s w ∧ (s :: w').Nodup ∧
      wArr A s t0 w' ≤ wArr A s t0 w :=
  exists_simple_walk_le E A hfifo hge s t0 w.length w le_rfl hw

/-- The arrival times of simple walks (no vertex visited twice) from `s` to `v`. -/
def simpleArrivals (E : V → V → Prop) (A : V → V → α → α) (s : V) (t0 : α) (v : V) :
    Set α :=
  {τ | ∃ w, IsWalk E s w ∧ wEnd s w = v ∧ (s :: w).Nodup ∧ wArr A s t0 w = τ}

/-- Under FIFO and `t ≤ A t`, all walks and simple walks have the same lower bounds on
their arrival times. -/
theorem lowerBounds_simple [Preorder α] (E : V → V → Prop) (A : V → V → α → α)
    (hfifo : ∀ u v, E u v → Monotone (A u v)) (hge : ∀ u v, E u v → ∀ t, t ≤ A u v t)
    (s : V) (t0 : α) (v : V) :
    lowerBounds (arrivals E A s t0 v) = lowerBounds (simpleArrivals E A s t0 v) := by
  apply Set.Subset.antisymm
  · apply lowerBounds_mono_set
    rintro τ ⟨w, hw, he, -, rfl⟩
    exact (arr_iff_walk E A s t0 v _).2 ⟨w, hw, he, rfl⟩
  · intro x hx τ hτ
    obtain ⟨w, hw, rfl, rfl⟩ := (arr_iff_walk E A s t0 v τ).1 hτ
    obtain ⟨w', hw', he', hnd, hle⟩ := simple_walk_suffices E A hfifo hge s t0 w hw
    exact (hx ⟨w', hw', he', hnd, rfl⟩).trans hle

/-- Earliest arrival over all walks equals earliest arrival over simple walks: the two
infima are the same. For Studio: allowing loops in the search changes nothing. -/
theorem isGLB_arrivals_iff_simple [Preorder α] (E : V → V → Prop) (A : V → V → α → α)
    (hfifo : ∀ u v, E u v → Monotone (A u v)) (hge : ∀ u v, E u v → ∀ t, t ≤ A u v t)
    (s : V) (t0 : α) (v : V) (τ : α) :
    IsGLB (arrivals E A s t0 v) τ ↔ IsGLB (simpleArrivals E A s t0 v) τ := by
  unfold IsGLB
  rw [lowerBounds_simple E A hfifo hge s t0 v]

/-- On a finite graph the earliest arrival is a minimum, attained by a simple walk.
There are infinitely many walks (cycles), but under FIFO and `t ≤ A t` only the finitely
many simple ones matter. For Studio: whenever the exit is reachable at all, some route
achieves the earliest arrival, so "the earliest-arrival route" is well defined. -/
theorem ea_walk_exists [Fintype V] [LinearOrder α] (E : V → V → Prop) (A : V → V → α → α)
    (hfifo : ∀ u v, E u v → Monotone (A u v)) (hge : ∀ u v, E u v → ∀ t, t ≤ A u v t)
    (s : V) (t0 : α) (v : V) (w0 : List V) (hw0 : IsWalk E s w0) (he0 : wEnd s w0 = v) :
    ∃ w, IsEA E A s t0 v w ∧ (s :: w).Nodup := by
  have hfin : Set.Finite {w : List V | IsWalk E s w ∧ wEnd s w = v ∧ (s :: w).Nodup} :=
    (List.finite_length_le V (Fintype.card V)).subset (fun w hw => by
      have := hw.2.2.length_le_card
      simp only [List.length_cons] at this
      show w.length ≤ Fintype.card V
      omega)
  obtain ⟨w1, hw1, he1, hnd1, -⟩ := simple_walk_suffices E A hfifo hge s t0 w0 hw0
  obtain ⟨w, ⟨hw, he, hnd⟩, hmin⟩ := Set.exists_min_image
    {w : List V | IsWalk E s w ∧ wEnd s w = v ∧ (s :: w).Nodup} (wArr A s t0) hfin
    ⟨w1, hw1, he1.trans he0, hnd1⟩
  refine ⟨w, ⟨hw, he, fun w' hw' he' => ?_⟩, hnd⟩
  obtain ⟨w'', hw'', he'', hnd'', hle⟩ := simple_walk_suffices E A hfifo hge s t0 w' hw'
  exact (hmin w'' ⟨hw'', he''.trans he', hnd''⟩).trans hle

/-- On a finite graph, every vertex with a walk from `s` has an earliest arrival time
that some walk attains (the arrival set has a least element). -/
theorem earliest_exists [Fintype V] [LinearOrder α] (E : V → V → Prop)
    (A : V → V → α → α) (hfifo : ∀ u v, E u v → Monotone (A u v))
    (hge : ∀ u v, E u v → ∀ t, t ≤ A u v t) (s : V) (t0 : α) (v : V)
    (h : ∃ τ, Arr E A s t0 v τ) : ∃ τ, IsLeast (arrivals E A s t0 v) τ := by
  obtain ⟨τ, hτ⟩ := h
  obtain ⟨w0, hw0, he0, -⟩ := (arr_iff_walk E A s t0 v τ).1 hτ
  obtain ⟨w, hea, -⟩ := ea_walk_exists E A hfifo hge s t0 v w0 hw0 he0
  exact ⟨_, isLeast_of_isEA E A s t0 v w hea⟩

/-- Walks that may wait: as `Arr`, plus waiting at the current vertex from `τ` to any
later `τ'`. -/
inductive ArrW [LE α] (E : V → V → Prop) (A : V → V → α → α) (s : V) (t0 : α) :
    V → α → Prop
  | nil : ArrW E A s t0 s t0
  | snoc {u v : V} {τ : α} : ArrW E A s t0 u τ → E u v → ArrW E A s t0 v (A u v τ)
  | wait {v : V} {τ τ' : α} : ArrW E A s t0 v τ → τ ≤ τ' → ArrW E A s t0 v τ'

/-- A walk without waiting is a walk that may wait (and never does). -/
theorem arrW_of_arr [Preorder α] (E : V → V → Prop) (A : V → V → α → α) (s : V) (t0 : α)
    {v : V} {τ : α} (h : Arr E A s t0 v τ) : ArrW E A s t0 v τ := by
  induction h with
  | nil => exact ArrW.nil
  | snoc _ huv ih => exact ih.snoc huv

/-- L10b on the whole network: under FIFO, waiting never helps. Every walk that waits
is matched by a walk that does not wait and arrives no later. For Studio: the router
need not model "hold at a refuge until the corridor clears"; under FIFO it cannot win.
(`GraphNumerics.L10b_wait_never_helps` is the single-route form.) -/
theorem waiting_never_helps [Preorder α] (E : V → V → Prop) (A : V → V → α → α)
    (hfifo : ∀ u v, E u v → Monotone (A u v)) (s : V) (t0 : α) {v : V} {τ : α}
    (h : ArrW E A s t0 v τ) : ∃ τ', Arr E A s t0 v τ' ∧ τ' ≤ τ := by
  induction h with
  | nil => exact ⟨t0, Arr.nil, le_rfl⟩
  | snoc _ huv ih =>
    obtain ⟨τ', h', hle⟩ := ih
    exact ⟨_, h'.snoc huv, hfifo _ _ huv hle⟩
  | wait _ hle ih =>
    obtain ⟨τ', h', hle'⟩ := ih
    exact ⟨τ', h', hle'.trans hle⟩

/-- Under FIFO, the earliest arrival without waiting is also the earliest arrival when
waiting is allowed. -/
theorem isLeast_waiting [Preorder α] (E : V → V → Prop) (A : V → V → α → α)
    (hfifo : ∀ u v, E u v → Monotone (A u v)) (s : V) (t0 : α) (v : V) (τ : α)
    (h : IsLeast (arrivals E A s t0 v) τ) : IsLeast {τ | ArrW E A s t0 v τ} τ := by
  refine ⟨arrW_of_arr E A s t0 h.1, fun τ' hτ' => ?_⟩
  obtain ⟨τ'', h'', hle⟩ := waiting_never_helps E A hfifo s t0 hτ'
  exact (h.2 h'').trans hle

/-! ## The time-dependent Bellman inequalities -/

/-- L10b (Bellman form, corrected direction). Under FIFO, any labelling `d` with
`d s ≤ t0` that satisfies every link's Bellman inequality `d v ≤ A u v (d u)` lies below
every walk's arrival time, hence below the earliest arrival. With `ea_bellman`, the
earliest arrival is the *greatest* such labelling in the time order. -/
theorem bellman_subsolution_le [Preorder α] (E : V → V → Prop) (A : V → V → α → α)
    (hfifo : ∀ u v, E u v → Monotone (A u v)) (s : V) (t0 : α) (d : V → α)
    (hs : d s ≤ t0) (hd : ∀ u v, E u v → d v ≤ A u v (d u)) {v : V} {τ : α}
    (h : Arr E A s t0 v τ) : d v ≤ τ := by
  induction h with
  | nil => exact hs
  | @snoc u v τ _ huv ih => exact (hd u v huv).trans (hfifo u v huv ih)

/-- The earliest arrival satisfies the Bellman inequality on every link between vertices
whose earliest arrival is attained: `EA(v) ≤ A u v (EA(u))`. No FIFO needed. -/
theorem ea_bellman [Preorder α] (E : V → V → Prop) (A : V → V → α → α) (s : V) (t0 : α)
    (u v : V) (eu ev : α) (hu : IsLeast (arrivals E A s t0 u) eu)
    (hv : IsLeast (arrivals E A s t0 v) ev) (huv : E u v) : ev ≤ A u v eu :=
  hv.2 (Arr.snoc hu.1 huv)

/-- Two-vertex network for the Bellman counterexample: `0 → 1` (one tick) and a
zero-time self-loop `1 → 1` (a landing one can stand on). -/
def bmE (u v : Fin 2) : Prop := (u = 0 ∧ v = 1) ∨ (u = 1 ∧ v = 1)

instance : DecidableRel bmE := fun u v =>
  inferInstanceAs (Decidable ((u = 0 ∧ v = 1) ∨ (u = 1 ∧ v = 1)))

/-- Arrival functions of `bmE`: one tick on `0 → 1`, zero time on the loop at `1`. -/
def bmA (u _ : Fin 2) (t : ℕ) : ℕ := if u = 1 then t else t + 1

/-- Every walk from `0` at tick 0 reaches `1` at tick 1 or later. -/
theorem bm_lower {x : Fin 2} {τ : ℕ} (h : Arr bmE bmA 0 0 x τ) : x = 1 → 1 ≤ τ := by
  induction h with
  | nil => exact fun h => absurd h (by decide)
  | @snoc y z τ' _ hyz ih =>
    intro _
    rcases hyz with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
    · show 1 ≤ (if (0 : Fin 2) = 1 then τ' else τ' + 1)
      rw [ite_eq_right (by decide)]
      omega
    · show 1 ≤ (if (1 : Fin 2) = 1 then τ' else τ' + 1)
      rw [ite_eq_left rfl]
      exact ih rfl

/-- The literature review's L10b wording ("earliest arrival is the least fixed point of
the time-dependent Bellman equation") holds only in the reversed, information order.
In the time order it fails: on `bmE` (FIFO, no time travel) both `d = (0, 0)` and the
earliest arrival `(0, 1)` solve `d 1 = min (A 0 1 (d 0)) (A 1 1 (d 1))`, and the least
solution `(0, 0)` is not the earliest arrival. Zero-time cycles make Bellman solutions
non-unique; the earliest arrival is the greatest one (`bellman_subsolution_le`). For
Studio: a Bellman-Ford style solver must iterate down from `⊤`, not up from `t0`. -/
theorem bellman_least_fixed_point_false :
    (∀ u v, bmE u v → Monotone (bmA u v)) ∧ (∀ u v, bmE u v → ∀ t, t ≤ bmA u v t) ∧
      (0 : ℕ) = min (bmA 0 1 0) (bmA 1 1 0) ∧ (1 : ℕ) = min (bmA 0 1 0) (bmA 1 1 1) ∧
      IsLeast (arrivals bmE bmA 0 0 1) 1 := by
  refine ⟨fun u v _ a b hab => ?_, fun u v _ t => ?_, by decide, by decide, ?_, ?_⟩
  · unfold bmA; split_ifs <;> omega
  · unfold bmA; split_ifs <;> omega
  · exact ((Arr.nil : Arr bmE bmA 0 0 0 0).snoc (show bmE 0 1 by decide)).cast (by decide)
  · exact fun τ hτ => bm_lower hτ rfl

/-! ## (2) The label-setting loop -/

/-- The loop's state: settled vertices `S`, reached vertices `R` (those with a label)
and labels `d` (tentative arrival times). -/
structure State (V α : Type*) where
  S : Finset V
  R : Finset V
  d : V → α

/-- The start of the loop: nothing settled, only the source reached, at the departure
time `t0`. -/
def Init (s : V) (t0 : α) (σ : State V α) : Prop := σ.S = ∅ ∧ σ.R = {s} ∧ σ.d s = t0

/-- `u` may be popped next: reached, not settled, and of least label among the
reached, unsettled vertices. Ties may be broken any way. -/
def Selectable [LE α] (σ : State V α) (u : V) : Prop :=
  u ∈ σ.R ∧ u ∉ σ.S ∧ ∀ v ∈ σ.R, v ∉ σ.S → σ.d u ≤ σ.d v

/-- One iteration: pop a selectable `u`, settle it, and relax its links into unsettled
vertices with `d v ← min (d v) (A u v (d u))`. In order, the clauses say: `u` is
settled; nothing is un-reached; settled labels (including `u`'s) do not change; labels
only drop; every link `u → v` into an unsettled `v` leaves `v` reached with
`d v ≤ A u v (d u)`; and every label afterwards is an old label or `A u v (d u)` for a
link `u → v`. -/
def Step [DecidableEq V] [LE α] (E : V → V → Prop) (A : V → V → α → α)
    (σ σ' : State V α) : Prop :=
  ∃ u, Selectable σ u ∧ σ'.S = insert u σ.S ∧ σ.R ⊆ σ'.R ∧
    (∀ v ∈ σ'.S, σ'.d v = σ.d v) ∧
    (∀ v ∈ σ.R, σ'.d v ≤ σ.d v) ∧
    (∀ v, E u v → v ∉ σ'.S → v ∈ σ'.R ∧ σ'.d v ≤ A u v (σ.d u)) ∧
    (∀ v ∈ σ'.R, (v ∈ σ.R ∧ σ'.d v = σ.d v) ∨ (E u v ∧ σ'.d v = A u v (σ.d u)))

/-- The time-dependent Dijkstra invariant: the source is reached with label at most
`t0`, settled vertices are reached, every reached label is the arrival time of a real
walk, every settled label is at most every walk's arrival time, and every link out of a
settled vertex into an unsettled one has been relaxed. -/
structure Inv [LE α] (E : V → V → Prop) (A : V → V → α → α) (s : V) (t0 : α)
    (σ : State V α) : Prop where
  src : s ∈ σ.R ∧ σ.d s ≤ t0
  sub : σ.S ⊆ σ.R
  real : ∀ v ∈ σ.R, Arr E A s t0 v (σ.d v)
  opt : ∀ v ∈ σ.S, ∀ τ, Arr E A s t0 v τ → σ.d v ≤ τ
  relaxed : ∀ x ∈ σ.S, ∀ v, E x v → v ∉ σ.S → v ∈ σ.R ∧ σ.d v ≤ A x v (σ.d x)

/-- The loop starts in a state that satisfies the invariant. -/
theorem init_inv [Preorder α] (E : V → V → Prop) (A : V → V → α → α) (s : V) (t0 : α)
    (σ : State V α) (h : Init s t0 σ) : Inv E A s t0 σ := by
  obtain ⟨hS, hR, hd⟩ := h
  refine ⟨⟨by simp [hR], by simp [hd]⟩, by simp [hS], ?_, by simp [hS], by simp [hS]⟩
  intro v hv
  rw [hR, mem_singleton] at hv
  subst hv
  exact Arr.nil.cast hd.symm

/-- The core of the argument (L10c). With FIFO links that take non-negative time, every
walk from the source to an unsettled vertex arrives no earlier than the label of a
selectable vertex. The walk must leave the settled set through a relaxed link `y → x`;
it reached `y` no earlier than `d y` (settled labels are optimal), so by FIFO it reaches
`x` no earlier than `A y x (d y) ≥ d x ≥ d u`; later links only add time. -/
theorem selectable_le [Preorder α] (E : V → V → Prop) (A : V → V → α → α)
    (hfifo : ∀ u v, E u v → Monotone (A u v)) (hge : ∀ u v, E u v → ∀ t, t ≤ A u v t)
    (s : V) (t0 : α) (σ : State V α) (hI : Inv E A s t0 σ) (u : V) (hu : Selectable σ u) :
    ∀ x τ, Arr E A s t0 x τ → x ∉ σ.S → σ.d u ≤ τ := by
  intro x τ hw
  induction hw with
  | nil =>
    intro hs
    exact (hu.2.2 s hI.src.1 hs).trans hI.src.2
  | @snoc y x τ hw' hyx ih =>
    intro hx
    by_cases hy : y ∈ σ.S
    · obtain ⟨hxR, hxd⟩ := hI.relaxed y hy x hyx hx
      exact (hu.2.2 x hxR hx).trans (hxd.trans (hfifo y x hyx (hI.opt y hy τ hw')))
    · exact (ih hy).trans (hge y x hyx τ)

/-- L10c (key lemma). With FIFO links taking non-negative time, the vertex the loop may
settle next carries its earliest arrival time: its label is the arrival time of a walk
from the source, and no walk arrives earlier. -/
theorem selectable_optimal [Preorder α] (E : V → V → Prop) (A : V → V → α → α)
    (hfifo : ∀ u v, E u v → Monotone (A u v)) (hge : ∀ u v, E u v → ∀ t, t ≤ A u v t)
    (s : V) (t0 : α) (σ : State V α) (hI : Inv E A s t0 σ) (u : V) (hu : Selectable σ u) :
    IsLeast (arrivals E A s t0 u) (σ.d u) :=
  ⟨hI.real u hu.1, fun τ hτ => selectable_le E A hfifo hge s t0 σ hI u hu u τ hτ hu.2.1⟩

/-- L10c. One step of the loop on FIFO links taking non-negative time preserves the
invariant. -/
theorem step_inv [DecidableEq V] [Preorder α] (E : V → V → Prop) (A : V → V → α → α)
    (hfifo : ∀ u v, E u v → Monotone (A u v)) (hge : ∀ u v, E u v → ∀ t, t ≤ A u v t)
    (s : V) (t0 : α) (σ σ' : State V α) (hI : Inv E A s t0 σ) (hst : Step E A σ σ') :
    Inv E A s t0 σ' := by
  obtain ⟨u, hu, hS, hR, hfix, hdrop, hrelax, horig⟩ := hst
  have huopt := selectable_optimal E A hfifo hge s t0 σ hI u hu
  refine ⟨⟨hR hI.src.1, le_trans (hdrop s hI.src.1) hI.src.2⟩, ?_, ?_, ?_, ?_⟩
  · intro v hv
    rw [hS, mem_insert] at hv
    rcases hv with rfl | hv
    · exact hR hu.1
    · exact hR (hI.sub hv)
  · intro v hv
    rcases horig v hv with ⟨hvR, hd⟩ | ⟨ha, hd⟩
    · rw [hd]; exact hI.real v hvR
    · rw [hd]; exact (hI.real u hu.1).snoc ha
  · intro v hv τ hτ
    rw [hfix v hv]
    rw [hS, mem_insert] at hv
    rcases hv with rfl | hv
    · exact huopt.2 hτ
    · exact hI.opt v hv τ hτ
  · intro x hx v hxv hv
    have hx' := hx
    rw [hS, mem_insert] at hx'
    rcases hx' with rfl | hxS
    · obtain ⟨h1, h2⟩ := hrelax v hxv hv
      exact ⟨h1, by rw [hfix x hx]; exact h2⟩
    · have hvS : v ∉ σ.S := fun h => hv (by rw [hS]; exact mem_insert_of_mem h)
      obtain ⟨hvR, hvd⟩ := hI.relaxed x hxS v hxv hvS
      refine ⟨hR hvR, ?_⟩
      rw [hfix x hx]
      exact le_trans (hdrop v hvR) hvd

/-- Every state the loop can reach from its start satisfies the invariant. -/
theorem reachable_inv [DecidableEq V] [Preorder α] (E : V → V → Prop) (A : V → V → α → α)
    (hfifo : ∀ u v, E u v → Monotone (A u v)) (hge : ∀ u v, E u v → ∀ t, t ≤ A u v t)
    (s : V) (t0 : α) (σ₀ σ : State V α) (h0 : Init s t0 σ₀)
    (hrun : Relation.ReflTransGen (Step E A) σ₀ σ) : Inv E A s t0 σ := by
  induction hrun with
  | refl => exact init_inv E A s t0 σ₀ h0
  | tail _ hst ih => exact step_inv E A hfifo hge s t0 _ _ ih hst

/-- L10c, the invariant asked of the router: at any point of any run, every settled
vertex's label is its earliest arrival time from `s` leaving at `t0` (the arrival time of
some walk, and no walk arrives earlier). -/
theorem settled_optimal [DecidableEq V] [Preorder α] (E : V → V → Prop)
    (A : V → V → α → α) (hfifo : ∀ u v, E u v → Monotone (A u v))
    (hge : ∀ u v, E u v → ∀ t, t ≤ A u v t) (s : V) (t0 : α) (σ₀ σ : State V α)
    (h0 : Init s t0 σ₀) (hrun : Relation.ReflTransGen (Step E A) σ₀ σ) (v : V)
    (hv : v ∈ σ.S) : IsLeast (arrivals E A s t0 v) (σ.d v) := by
  have hI := reachable_inv E A hfifo hge s t0 σ₀ σ h0 hrun
  exact ⟨hI.real v (hI.sub hv), fun τ hτ => hI.opt v hv τ hτ⟩

/-- L10c. Time-dependent Dijkstra is exact under FIFO. At any point of any run from the
start, the vertex popped next has as label its earliest arrival time. In particular, a
router that stops when it pops the exit returns the exit's earliest arrival time. -/
theorem td_dijkstra_correct [DecidableEq V] [Preorder α] (E : V → V → Prop)
    (A : V → V → α → α) (hfifo : ∀ u v, E u v → Monotone (A u v))
    (hge : ∀ u v, E u v → ∀ t, t ≤ A u v t) (s : V) (t0 : α) (σ₀ σ : State V α)
    (h0 : Init s t0 σ₀) (hrun : Relation.ReflTransGen (Step E A) σ₀ σ) (u : V)
    (hu : Selectable σ u) : IsLeast (arrivals E A s t0 u) (σ.d u) :=
  selectable_optimal E A hfifo hge s t0 σ (reachable_inv E A hfifo hge s t0 σ₀ σ h0 hrun) u hu

/-- L10c at exhaustion. When the queue runs dry (every reached vertex is settled), the
settled vertices are exactly those with a walk from `s`, and each carries its earliest
arrival time. For Studio: a full sweep gives the earliest arrival at every node. -/
theorem td_dijkstra_exhausted [DecidableEq V] [Preorder α] (E : V → V → Prop)
    (A : V → V → α → α) (hfifo : ∀ u v, E u v → Monotone (A u v))
    (hge : ∀ u v, E u v → ∀ t, t ≤ A u v t) (s : V) (t0 : α) (σ₀ σ : State V α)
    (h0 : Init s t0 σ₀) (hrun : Relation.ReflTransGen (Step E A) σ₀ σ) (hdry : σ.R ⊆ σ.S) :
    (∀ v, (∃ τ, Arr E A s t0 v τ) ↔ v ∈ σ.S) ∧
      ∀ v ∈ σ.S, IsLeast (arrivals E A s t0 v) (σ.d v) := by
  have hI := reachable_inv E A hfifo hge s t0 σ₀ σ h0 hrun
  refine ⟨fun v => ⟨?_, fun hv => ⟨_, hI.real v (hI.sub hv)⟩⟩,
    fun v hv => ⟨hI.real v (hI.sub hv), fun τ hτ => hI.opt v hv τ hτ⟩⟩
  rintro ⟨τ, hw⟩
  induction hw with
  | nil => exact hdry hI.src.1
  | @snoc y x τ _ hyx ih =>
    by_contra hx
    exact hx (hdry (hI.relaxed y ih x hyx hx).1)

/-- L10c at exhaustion: a vertex the loop never reached has no walk from the source at
all. For Studio: "no route" is a true statement about the building, not an artefact of
the search. -/
theorem never_reached_no_walk [DecidableEq V] [Preorder α] (E : V → V → Prop)
    (A : V → V → α → α) (hfifo : ∀ u v, E u v → Monotone (A u v))
    (hge : ∀ u v, E u v → ∀ t, t ≤ A u v t) (s : V) (t0 : α) (σ₀ σ : State V α)
    (h0 : Init s t0 σ₀) (hrun : Relation.ReflTransGen (Step E A) σ₀ σ) (hdry : σ.R ⊆ σ.S)
    (v : V) (hv : v ∉ σ.R) :
    arrivals E A s t0 v = ∅ ∧ ∀ w, IsWalk E s w → wEnd s w ≠ v := by
  have hreach := (td_dijkstra_exhausted E A hfifo hge s t0 σ₀ σ h0 hrun hdry).1 v
  have hI := reachable_inv E A hfifo hge s t0 σ₀ σ h0 hrun
  have hnot : ¬ ∃ τ, Arr E A s t0 v τ := fun h => hv (hI.sub (hreach.1 h))
  refine ⟨?_, fun w hw he => hnot ⟨_, (arr_iff_walk E A s t0 v _).2 ⟨w, hw, he, rfl⟩⟩⟩
  ext τ
  simp only [Set.mem_empty_iff_false, iff_false]
  exact fun h => hnot ⟨τ, h⟩

/-- The router's output: the label of a settled vertex, `⊤` (never) for any other. -/
def outLabel [DecidableEq V] [Top α] (σ : State V α) (v : V) : α :=
  if v ∈ σ.S then σ.d v else ⊤

/-- L10c, output form. At exhaustion, the router's output at every vertex is the
infimum over all walks of the arrival time: the settled label where a walk exists, and
`⊤` where none does. This is "EA(v) = inf over walks", with `⊤` for unreachable. -/
theorem td_dijkstra_glb [DecidableEq V] [Preorder α] [OrderTop α] (E : V → V → Prop)
    (A : V → V → α → α) (hfifo : ∀ u v, E u v → Monotone (A u v))
    (hge : ∀ u v, E u v → ∀ t, t ≤ A u v t) (s : V) (t0 : α) (σ₀ σ : State V α)
    (h0 : Init s t0 σ₀) (hrun : Relation.ReflTransGen (Step E A) σ₀ σ) (hdry : σ.R ⊆ σ.S)
    (v : V) : IsGLB (arrivals E A s t0 v) (outLabel σ v) := by
  obtain ⟨hreach, hopt⟩ := td_dijkstra_exhausted E A hfifo hge s t0 σ₀ σ h0 hrun hdry
  unfold outLabel
  split_ifs with hv
  · exact (hopt v hv).isGLB
  · have hempty : arrivals E A s t0 v = ∅ := by
      ext τ
      simp only [Set.mem_empty_iff_false, iff_false]
      exact fun h => hv ((hreach v).1 ⟨τ, h⟩)
    rw [hempty]
    exact isGLB_empty

/-- L10c in Studio's time type `WithTop ℝ` (seconds, `⊤` = never; closed links arrive
at `⊤`). At exhaustion the router's output equals `EA`, the `sInf` over all walks of
the arrival time, at every vertex. -/
theorem td_dijkstra_EA_withTop [DecidableEq V] (E : V → V → Prop)
    (A : V → V → WithTop ℝ → WithTop ℝ) (hfifo : ∀ u v, E u v → Monotone (A u v))
    (hge : ∀ u v, E u v → ∀ t, t ≤ A u v t) (s : V) (t0 : WithTop ℝ)
    (σ₀ σ : State V (WithTop ℝ)) (h0 : Init s t0 σ₀)
    (hrun : Relation.ReflTransGen (Step E A) σ₀ σ) (hdry : σ.R ⊆ σ.S) (v : V) :
    outLabel σ v = EA E A s t0 v := by
  have h := td_dijkstra_glb E A hfifo hge s t0 σ₀ σ h0 hrun hdry v
  exact ((WithTop.isGLB_sInf' ⟨_, h.1⟩).unique h).symm

/-- L10c in any complete linear order of times (for example `ℝ≥0∞`, seconds since the
alarm with `⊤` = never): at exhaustion the router's output equals `EA` everywhere. -/
theorem td_dijkstra_EA [DecidableEq V] [CompleteLinearOrder α] (E : V → V → Prop)
    (A : V → V → α → α) (hfifo : ∀ u v, E u v → Monotone (A u v))
    (hge : ∀ u v, E u v → ∀ t, t ≤ A u v t) (s : V) (t0 : α) (σ₀ σ : State V α)
    (h0 : Init s t0 σ₀) (hrun : Relation.ReflTransGen (Step E A) σ₀ σ) (hdry : σ.R ⊆ σ.S)
    (v : V) : outLabel σ v = EA E A s t0 v :=
  (td_dijkstra_glb E A hfifo hge s t0 σ₀ σ h0 hrun hdry v).sInf_eq.symm

/-- Early exit is sound: at any point of any run, the label of the vertex popped next is
a lower bound on the arrival time of every walk to every unsettled vertex. So once the
exit is popped, or once the least label is `⊤`, nothing left in the queue can improve. -/
theorem early_exit_lower_bound [DecidableEq V] [Preorder α] (E : V → V → Prop)
    (A : V → V → α → α) (hfifo : ∀ u v, E u v → Monotone (A u v))
    (hge : ∀ u v, E u v → ∀ t, t ≤ A u v t) (s : V) (t0 : α) (σ₀ σ : State V α)
    (h0 : Init s t0 σ₀) (hrun : Relation.ReflTransGen (Step E A) σ₀ σ) (u : V)
    (hu : Selectable σ u) (x : V) (hx : x ∉ σ.S) :
    σ.d u ∈ lowerBounds (arrivals E A s t0 x) :=
  fun τ hτ => selectable_le E A hfifo hge s t0 σ (reachable_inv E A hfifo hge s t0 σ₀ σ h0 hrun)
    u hu x τ hτ hx

/-- Stopping when the least queued label is `⊤` is exact: the output (`⊤` for every
unsettled vertex) is then the earliest arrival everywhere. For Studio: a router that
never queues `Infinity` labels, or stops when the heap's minimum is `Infinity`, returns
the same answer as a full sweep. -/
theorem td_dijkstra_stop_at_top [DecidableEq V] [Preorder α] [OrderTop α]
    (E : V → V → Prop) (A : V → V → α → α) (hfifo : ∀ u v, E u v → Monotone (A u v))
    (hge : ∀ u v, E u v → ∀ t, t ≤ A u v t) (s : V) (t0 : α) (σ₀ σ : State V α)
    (h0 : Init s t0 σ₀) (hrun : Relation.ReflTransGen (Step E A) σ₀ σ) (u : V)
    (hu : Selectable σ u) (htop : σ.d u = ⊤) (v : V) :
    IsGLB (arrivals E A s t0 v) (outLabel σ v) := by
  unfold outLabel
  split_ifs with hv
  · exact (settled_optimal E A hfifo hge s t0 σ₀ σ h0 hrun v hv).isGLB
  · refine ⟨fun τ hτ => ?_, fun x _ => le_top⟩
    have := early_exit_lower_bound E A hfifo hge s t0 σ₀ σ h0 hrun u hu v hv hτ
    rwa [htop] at this

/-! ## Progress and termination on a finite graph -/

open Classical in
/-- One concrete iteration on a finite graph: settle `u`, then for every link `u → v`
into an unsettled `v` set `d v ← min (d v) (A u v (d u))` (or `A u v (d u)` if `v` had no
label). -/
noncomputable def relaxState [DecidableEq V] [Fintype V] [LinearOrder α]
    (E : V → V → Prop) (A : V → V → α → α) (σ : State V α) (u : V) : State V α where
  S := insert u σ.S
  R := σ.R ∪ Finset.univ.filter (fun v => E u v ∧ v ∉ insert u σ.S)
  d := fun v => if E u v ∧ v ∉ insert u σ.S then
      (if v ∈ σ.R then min (σ.d v) (A u v (σ.d u)) else A u v (σ.d u)) else σ.d v

/-- The concrete relaxation is a `Step`: the relational step is implementable, with
`min` updates as in Dreyfus' algorithm. -/
theorem relaxState_step [DecidableEq V] [Fintype V] [LinearOrder α] (E : V → V → Prop)
    (A : V → V → α → α) (σ : State V α) (u : V) (hu : Selectable σ u) :
    Step E A σ (relaxState E A σ u) := by
  classical
  refine ⟨u, hu, rfl, Finset.subset_union_left, ?_, ?_, ?_, ?_⟩
  · intro v hv
    simp only [relaxState] at hv ⊢
    exact ite_eq_right (fun h => h.2 hv)
  · intro v hv
    simp only [relaxState]
    split_ifs
    · exact min_le_left _ _
    · exact le_rfl
  · intro v huv hv
    simp only [relaxState] at hv ⊢
    refine ⟨Finset.mem_union_right _ (Finset.mem_filter.2 ⟨Finset.mem_univ _, huv, hv⟩), ?_⟩
    rw [ite_eq_left ⟨huv, hv⟩]
    split_ifs
    · exact min_le_right _ _
    · exact le_rfl
  · intro v hv
    simp only [relaxState] at hv ⊢
    by_cases h : E u v ∧ v ∉ insert u σ.S
    · rw [ite_eq_left h]
      split_ifs with hR
      · rcases min_choice (σ.d v) (A u v (σ.d u)) with hm | hm
        · exact Or.inl ⟨hR, hm⟩
        · exact Or.inr ⟨h.1, hm⟩
      · exact Or.inr ⟨h.1, rfl⟩
    · rw [ite_eq_right h]
      rcases Finset.mem_union.1 hv with hR | hF
      · exact Or.inl ⟨hR, rfl⟩
      · exact absurd (Finset.mem_filter.1 hF).2 h

/-- Every step settles a vertex that was not settled before, so no run has more steps
than the graph has vertices. -/
theorem step_settles_new [DecidableEq V] [LE α] (E : V → V → Prop) (A : V → V → α → α)
    (σ σ' : State V α) (hst : Step E A σ σ') : σ.S ⊂ σ'.S := by
  obtain ⟨u, hu, hS, -⟩ := hst
  rw [hS]
  exact Finset.ssubset_insert hu.2.1

/-- On a finite graph the loop can always run to exhaustion: from any state some run
reaches a state where every reached vertex is settled. -/
theorem exists_exhausted [DecidableEq V] [Fintype V] [LinearOrder α] (E : V → V → Prop)
    (A : V → V → α → α) (n : ℕ) :
    ∀ σ : State V α, Fintype.card V - σ.S.card ≤ n →
      ∃ σ', Relation.ReflTransGen (Step E A) σ σ' ∧ σ'.R ⊆ σ'.S := by
  induction n with
  | zero =>
    intro σ hσ
    refine ⟨σ, Relation.ReflTransGen.refl, ?_⟩
    have hle := Finset.card_le_univ σ.S
    have huniv : σ.S = Finset.univ := Finset.eq_univ_of_card _ (by omega)
    intro v _
    rw [huniv]
    exact Finset.mem_univ v
  | succ n ih =>
    intro σ hσ
    by_cases hdry : σ.R ⊆ σ.S
    · exact ⟨σ, Relation.ReflTransGen.refl, hdry⟩
    · have hne : (σ.R \ σ.S).Nonempty := Finset.sdiff_nonempty.2 hdry
      obtain ⟨u, hu, hmin⟩ := Finset.exists_min_image (σ.R \ σ.S) σ.d hne
      rw [Finset.mem_sdiff] at hu
      have hsel : Selectable σ u :=
        ⟨hu.1, hu.2, fun v hv hvS => hmin v (Finset.mem_sdiff.2 ⟨hv, hvS⟩)⟩
      have hcard : (relaxState E A σ u).S.card = σ.S.card + 1 :=
        Finset.card_insert_of_notMem hu.2
      have hle := Finset.card_le_univ (relaxState E A σ u).S
      obtain ⟨σ', hrun, hdry'⟩ := ih (relaxState E A σ u) (by omega)
      exact ⟨σ', Relation.ReflTransGen.head (relaxState_step E A σ u hsel) hrun, hdry'⟩

/-- L10c, total correctness on a finite graph. From the start, some run of the loop
reaches exhaustion, and its output is the earliest arrival (infimum over walks, `⊤` if
none) at every vertex. For Studio: the router always finishes and is always right,
provided every link is FIFO and takes non-negative time. -/
theorem td_dijkstra_total [DecidableEq V] [Fintype V] [LinearOrder α] [OrderTop α]
    (E : V → V → Prop) (A : V → V → α → α) (hfifo : ∀ u v, E u v → Monotone (A u v))
    (hge : ∀ u v, E u v → ∀ t, t ≤ A u v t) (s : V) (t0 : α) (σ₀ : State V α)
    (h0 : Init s t0 σ₀) :
    ∃ σ, Relation.ReflTransGen (Step E A) σ₀ σ ∧ σ.R ⊆ σ.S ∧
      ∀ v, IsGLB (arrivals E A s t0 v) (outLabel σ v) := by
  obtain ⟨σ, hrun, hdry⟩ := exists_exhausted E A _ σ₀ le_rfl
  exact ⟨σ, hrun, hdry, td_dijkstra_glb E A hfifo hge s t0 σ₀ σ h0 hrun hdry⟩

/-! ## Studio's arrival functions satisfy the hypotheses -/

/-- Static links: a fixed non-negative traversal time `w` gives `A t = t + w`, which is
FIFO and takes non-negative time. So static Dijkstra (`AStar.dijkstra_correct`) is the
special case of this file with constant weights. -/
theorem static_hyps (w : ℝ) (hw : 0 ≤ w) :
    Monotone (fun t : ℝ => t + w) ∧ ∀ t : ℝ, t ≤ t + w :=
  ⟨fun _ _ h => by linarith, fun _ => by linarith⟩

/-- A closure-window link (`GraphNumerics.closeArc`: walk `τ` seconds, closed from `b`
on) takes non-negative time when `τ ≥ 0`. -/
theorem closeArc_ge (τ b : ℝ) (hτ : 0 ≤ τ) (x : WithTop ℝ) : x ≤ closeArc τ b x := by
  induction x using WithTop.recTopCoe with
  | top => exact (closeArc_top τ b).ge
  | coe t =>
    rw [closeArc_coe]
    split_ifs
    · exact WithTop.coe_le_coe.mpr (by linarith)
    · exact le_top

/-- The tick rule of `rl.js`/`rl.py` (`GraphNumerics.hopArc`) takes non-negative time. -/
theorem hopArc_ge (ign x : WithTop ℝ) : x ≤ hopArc ign x := by
  induction x using WithTop.recTopCoe with
  | top => exact le_of_eq rfl
  | coe t =>
    rw [hopArc_coe]
    split_ifs
    · exact WithTop.coe_le_coe.mpr (by linarith)
    · exact le_top

/-- L10c for the fire as closure windows (literature review section 5.2). Every link
`u → v` takes `τ u v ≥ 0` seconds and is closed by fire from time `b u v` on. At
exhaustion the router's output equals the earliest arrival `EA` (in `WithTop ℝ`, `⊤` =
no route before the fire closes it) at every vertex. -/
theorem closure_window_exact [DecidableEq V] (E : V → V → Prop) (τ b : V → V → ℝ)
    (hτ : ∀ u v, E u v → 0 ≤ τ u v) (s : V) (t0 : WithTop ℝ)
    (σ₀ σ : State V (WithTop ℝ)) (h0 : Init s t0 σ₀)
    (hrun : Relation.ReflTransGen (Step E (fun u v => closeArc (τ u v) (b u v))) σ₀ σ)
    (hdry : σ.R ⊆ σ.S) (v : V) :
    outLabel σ v = EA E (fun u v => closeArc (τ u v) (b u v)) s t0 v :=
  td_dijkstra_EA_withTop E _ (fun u v _ => R2_closeArc_fifo (τ u v) (b u v))
    (fun u v huv => closeArc_ge (τ u v) (b u v) (hτ u v huv)) s t0 σ₀ σ h0 hrun hdry v

/-- L10c for the code's tick rule: with `hopArc` links (move in one tick if the target
has not ignited at departure), the router's output at exhaustion is the earliest
arrival at every vertex. -/
theorem hop_rule_exact [DecidableEq V] (E : V → V → Prop) (ign : V → WithTop ℝ) (s : V)
    (t0 : WithTop ℝ) (σ₀ σ : State V (WithTop ℝ)) (h0 : Init s t0 σ₀)
    (hrun : Relation.ReflTransGen (Step E (fun _ v => hopArc (ign v))) σ₀ σ)
    (hdry : σ.R ⊆ σ.S) (v : V) :
    outLabel σ v = EA E (fun _ v => hopArc (ign v)) s t0 v :=
  td_dijkstra_EA_withTop E _ (fun _ v _ => R2_hopArc_fifo (ign v))
    (fun _ v _ => hopArc_ge (ign v)) s t0 σ₀ σ h0 hrun hdry v

/-! ## What survives without FIFO -/

/-- The part of the invariant that needs neither FIFO nor `t ≤ A t`: the source is
reached, settled vertices are reached, every label is the arrival time of a real walk,
and every link out of a settled vertex leads to a reached vertex. -/
structure Sound (E : V → V → Prop) (A : V → V → α → α) (s : V) (t0 : α)
    (σ : State V α) : Prop where
  src : s ∈ σ.R
  sub : σ.S ⊆ σ.R
  real : ∀ v ∈ σ.R, Arr E A s t0 v (σ.d v)
  relaxed : ∀ x ∈ σ.S, ∀ v, E x v → v ∉ σ.S → v ∈ σ.R

/-- The loop starts sound, for any arrival functions. -/
theorem sound_init (E : V → V → Prop) (A : V → V → α → α) (s : V) (t0 : α)
    (σ : State V α) (h : Init s t0 σ) : Sound E A s t0 σ := by
  obtain ⟨hS, hR, hd⟩ := h
  refine ⟨by simp [hR], by simp [hS], ?_, by simp [hS]⟩
  intro v hv
  rw [hR, mem_singleton] at hv
  subst hv
  exact Arr.nil.cast hd.symm

/-- Any step keeps the loop sound, for any arrival functions: no FIFO, no `t ≤ A t`. -/
theorem sound_step [DecidableEq V] [LE α] (E : V → V → Prop) (A : V → V → α → α) (s : V)
    (t0 : α) (σ σ' : State V α) (hI : Sound E A s t0 σ) (hst : Step E A σ σ') :
    Sound E A s t0 σ' := by
  obtain ⟨u, hu, hS, hR, -, -, hrelax, horig⟩ := hst
  refine ⟨hR hI.src, ?_, ?_, ?_⟩
  · intro v hv
    rw [hS, mem_insert] at hv
    rcases hv with rfl | hv
    · exact hR hu.1
    · exact hR (hI.sub hv)
  · intro v hv
    rcases horig v hv with ⟨hvR, hd⟩ | ⟨ha, hd⟩
    · rw [hd]; exact hI.real v hvR
    · rw [hd]; exact (hI.real u hu.1).snoc ha
  · intro x hx v hxv hv
    rw [hS, mem_insert] at hx
    rcases hx with rfl | hxS
    · exact (hrelax v hxv hv).1
    · have hvS : v ∉ σ.S := fun h => hv (by rw [hS]; exact mem_insert_of_mem h)
      exact hR (hI.relaxed x hxS v hxv hvS)

/-- Every state the loop can reach is sound, for any arrival functions. -/
theorem sound_reachable [DecidableEq V] [LE α] (E : V → V → Prop) (A : V → V → α → α)
    (s : V) (t0 : α) (σ₀ σ : State V α) (h0 : Init s t0 σ₀)
    (hrun : Relation.ReflTransGen (Step E A) σ₀ σ) : Sound E A s t0 σ := by
  induction hrun with
  | refl => exact sound_init E A s t0 σ₀ h0
  | tail _ hst ih => exact sound_step E A s t0 _ _ ih hst

/-- What the router guarantees without FIFO. At exhaustion, for any arrival functions,
the settled vertices are exactly those with a walk from `s`, and each label is the
arrival time of a real walk. For Studio: with a non-FIFO hazard model the router still
never invents a route and never misses a reachable node; only "earliest" is lost. -/
theorem exhausted_without_fifo [DecidableEq V] [LE α] (E : V → V → Prop)
    (A : V → V → α → α) (s : V) (t0 : α) (σ₀ σ : State V α) (h0 : Init s t0 σ₀)
    (hrun : Relation.ReflTransGen (Step E A) σ₀ σ) (hdry : σ.R ⊆ σ.S) :
    (∀ v, (∃ τ, Arr E A s t0 v τ) ↔ v ∈ σ.S) ∧ ∀ v ∈ σ.S, Arr E A s t0 v (σ.d v) := by
  have hI := sound_reachable E A s t0 σ₀ σ h0 hrun
  refine ⟨fun v => ⟨?_, fun hv => ⟨_, hI.real v (hI.sub hv)⟩⟩,
    fun v hv => hI.real v (hI.sub hv)⟩
  rintro ⟨τ, hw⟩
  induction hw with
  | nil => exact hdry hI.src
  | @snoc y x τ _ hyx ih =>
    by_contra hx
    exact hx (hdry (hI.relaxed y ih x hyx hx))

/-! ## (4) FIFO is necessary: a non-FIFO counterexample -/

section NonFIFO

/-- Counterexample graph: `0` = start, `1` = corridor node, `2` = side room, `3` = exit;
links `0 → 1`, `0 → 2`, `2 → 1`, `1 → 3`. -/
def nfE (u v : Fin 4) : Prop :=
  (u = 0 ∧ v = 1) ∨ (u = 0 ∧ v = 2) ∨ (u = 2 ∧ v = 1) ∨ (u = 1 ∧ v = 3)

instance : DecidableRel nfE := fun u v =>
  inferInstanceAs (Decidable ((u = 0 ∧ v = 1) ∨ (u = 0 ∧ v = 2) ∨ (u = 2 ∧ v = 1) ∨
    (u = 1 ∧ v = 3)))

/-- Counterexample arrival functions, in ticks. Every link takes one tick, except
`1 → 3`, which is not FIFO: entered before tick 2 it takes until tick 10 (a door that
is slow to open), entered at tick 2 or later it takes one tick. -/
def nfA (u v : Fin 4) (t : ℕ) : ℕ :=
  if u = 1 ∧ v = 3 then (if t < 2 then 10 else t + 1) else t + 1

/-- The non-FIFO link `1 → 3`: tick 10 if entered before tick 2, else one tick. -/
theorem nfA_13 (t : ℕ) : nfA 1 3 t = if t < 2 then 10 else t + 1 := by simp [nfA]

/-- Every other link of the counterexample takes one tick. -/
theorem nfA_ne (u v : Fin 4) (t : ℕ) (h : ¬ (u = 1 ∧ v = 3)) : nfA u v t = t + 1 :=
  ite_eq_right h

/-- Start of the run: only `0` reached, at tick 0. -/
def nfState0 : State (Fin 4) ℕ := ⟨∅, {0}, ![0, 0, 0, 0]⟩
/-- After popping `0`: `1` and `2` reached at tick 1. -/
def nfState1 : State (Fin 4) ℕ := ⟨{0}, {0, 1, 2}, ![0, 1, 1, 0]⟩
/-- After popping `1` (label 1): the exit is labelled `nfA 1 3 1 = 10`. -/
def nfState2 : State (Fin 4) ℕ := ⟨{0, 1}, {0, 1, 2, 3}, ![0, 1, 1, 10]⟩
/-- After popping `2`: its link `2 → 1` leads to a settled vertex and is not relaxed. -/
def nfState3 : State (Fin 4) ℕ := ⟨{0, 1, 2}, {0, 1, 2, 3}, ![0, 1, 1, 10]⟩
/-- After popping the exit at label 10: the queue is dry. -/
def nfState4 : State (Fin 4) ℕ := ⟨{0, 1, 2, 3}, {0, 1, 2, 3}, ![0, 1, 1, 10]⟩

/-- The counterexample run starts as the router would: from `0` at tick 0. -/
theorem nf_init : Init 0 0 nfState0 := by unfold Init; decide

/-- Pop `0` (the only reached vertex) and relax `0 → 1`, `0 → 2`. -/
theorem nf_step01 : Step nfE nfA nfState0 nfState1 :=
  ⟨0, by unfold Selectable; decide, by decide, by decide, by decide, by decide, by decide,
    by decide⟩

/-- Pop `1` (label 1, tied with `2`) and relax `1 → 3` to `nfA 1 3 1 = 10`. -/
theorem nf_step12 : Step nfE nfA nfState1 nfState2 :=
  ⟨1, by unfold Selectable; decide, by decide, by decide, by decide, by decide, by decide,
    by decide⟩

/-- Pop `2`; its only link leads to the settled `1`. -/
theorem nf_step23 : Step nfE nfA nfState2 nfState3 :=
  ⟨2, by unfold Selectable; decide, by decide, by decide, by decide, by decide, by decide,
    by decide⟩

/-- Pop the exit at label 10. -/
theorem nf_step34 : Step nfE nfA nfState3 nfState4 :=
  ⟨3, by unfold Selectable; decide, by decide, by decide, by decide, by decide, by decide,
    by decide⟩

/-- The counterexample run, start to exhaustion. -/
theorem nf_run : Relation.ReflTransGen (Step nfE nfA) nfState0 nfState4 :=
  (((Relation.ReflTransGen.single nf_step01).tail nf_step12).tail nf_step23).tail nf_step34

/-- Every walk from `0` at tick 0 reaches a vertex other than `0` at tick 1 or later,
and reaches the exit at tick 3 or later. -/
theorem nf_lower {x : Fin 4} {τ : ℕ} (h : Arr nfE nfA 0 0 x τ) :
    (x ≠ 0 → 1 ≤ τ) ∧ (x = 3 → 3 ≤ τ) := by
  induction h with
  | nil => exact ⟨fun h => absurd rfl h, fun h => absurd h (by decide)⟩
  | @snoc y z τ' _ hyz ih =>
    rcases hyz with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
    · rw [nfA_ne 0 1 τ' (by decide)]
      exact ⟨fun _ => by omega, fun h => absurd h (by decide)⟩
    · rw [nfA_ne 0 2 τ' (by decide)]
      exact ⟨fun _ => by omega, fun h => absurd h (by decide)⟩
    · rw [nfA_ne 2 1 τ' (by decide)]
      exact ⟨fun _ => by omega, fun h => absurd h (by decide)⟩
    · have h1 := ih.1 (by decide)
      rw [nfA_13]
      constructor <;> intro _ <;> split_ifs <;> omega

/-- The detour `0 → 2 → 1 → 3` reaches the exit at tick 3. -/
theorem nf_fast : Arr nfE nfA 0 0 3 3 :=
  ((((Arr.nil : Arr nfE nfA 0 0 0 0).snoc (show nfE 0 2 by decide)).snoc
    (show nfE 2 1 by decide)).snoc (show nfE 1 3 by decide)).cast (by decide)

/-- The earliest arrival at the exit is tick 3. -/
theorem nf_isLeast : IsLeast (arrivals nfE nfA 0 0 3) 3 :=
  ⟨nf_fast, fun _ hτ => (nf_lower hτ).2 rfl⟩

/-- Waiting at `1` from tick 1 to tick 2 on the direct route `0 → 1 → 3` also reaches
the exit at tick 3. -/
theorem nf_wait : ArrW nfE nfA 0 0 3 3 :=
  ((((ArrW.nil : ArrW nfE nfA 0 0 0 0).snoc (show nfE 0 1 by decide)).wait
    (show nfA 0 1 0 ≤ 2 by decide)).snoc (show nfE 1 3 by decide)).wait
    (show nfA 1 3 2 ≤ 3 by decide)

/-- L10c, FIFO is necessary. On this network every link takes non-negative time and all
but `1 → 3` are FIFO; `1 → 3` gets faster if entered later. The label-setting loop runs
to exhaustion and labels the exit 10, but the earliest arrival is 3 (via the detour
`0 → 2 → 1`, which reaches `1` later and so catches the fast link; waiting at `1` does
the same). The loop keeps only the earliest arrival at `1`, which is the wrong one to
extend. For Studio: a hazard model whose links can get faster with time (smoke clearing,
a door opening on a timer) breaks the router unless it is FIFO-ised first. -/
theorem fifo_necessary :
    (∀ u v, nfE u v → ∀ t, t ≤ nfA u v t) ∧ ¬ Monotone (nfA 1 3) ∧
      Init 0 0 nfState0 ∧ Relation.ReflTransGen (Step nfE nfA) nfState0 nfState4 ∧
      nfState4.R ⊆ nfState4.S ∧ nfState4.d 3 = 10 ∧
      IsLeast (arrivals nfE nfA 0 0 3) 3 ∧ ¬ IsLeast (arrivals nfE nfA 0 0 3) (nfState4.d 3) ∧
      ArrW nfE nfA 0 0 3 3 := by
  refine ⟨?_, ?_, nf_init, nf_run, by decide, rfl, nf_isLeast, ?_, nf_wait⟩
  · intro u v _ t
    unfold nfA
    split_ifs <;> omega
  · intro h
    have h12 : nfA 1 3 1 ≤ nfA 1 3 2 := h (by norm_num)
    exact absurd h12 (by decide)
  · intro h
    have := h.unique nf_isLeast
    exact absurd this (by decide)

/-- The only link into `1` comes from `0` or `2`. -/
theorem nfE_into_1 (u : Fin 4) (h : nfE u 1) : u = 0 ∨ u = 2 := by revert u h; decide

/-- The only link into `2` comes from `0`. -/
theorem nfE_into_2 (u : Fin 4) (h : nfE u 2) : u = 0 := by revert u h; decide

/-- The only link into the exit comes from `1`. -/
theorem nfE_into_3 (u : Fin 4) (h : nfE u 3) : u = 1 := by revert u h; decide

/-- What every run of the loop on the counterexample keeps true: `0` at tick 0, `1` and
`2` at tick 1 once reached (and `1` is reached no later than `2`), the exit at tick 10
once reached. -/
def nfP (σ : State (Fin 4) ℕ) : Prop :=
  0 ∈ σ.R ∧ σ.d 0 = 0 ∧ σ.S ⊆ σ.R ∧ (2 ∈ σ.R → 1 ∈ σ.R) ∧ (1 ∈ σ.R → σ.d 1 = 1) ∧
    (2 ∈ σ.R → σ.d 2 = 1) ∧ (3 ∈ σ.R → σ.d 3 = 10)

/-- `nfP` holds at the start of any run from `0` at tick 0. -/
theorem nfP_init (σ : State (Fin 4) ℕ) (h : Init 0 0 σ) : nfP σ := by
  obtain ⟨hS, hR, hd⟩ := h
  have hnot : ∀ v : Fin 4, v ≠ 0 → v ∉ σ.R := by
    intro v hv hvR
    rw [hR, mem_singleton] at hvR
    exact hv hvR
  refine ⟨by rw [hR]; exact mem_singleton_self 0, hd, by rw [hS]; exact empty_subset _,
    fun h => absurd h (hnot 2 (by decide)), fun h => absurd h (hnot 1 (by decide)),
    fun h => absurd h (hnot 2 (by decide)), fun h => absurd h (hnot 3 (by decide))⟩

/-- Every step of the loop on the counterexample keeps `nfP`, whatever vertex it pops
among the tied ones. -/
theorem nfP_step (σ σ' : State (Fin 4) ℕ) (hP : nfP σ) (hst : Step nfE nfA σ σ') :
    nfP σ' := by
  obtain ⟨h0R, h0d, hSR, h21, hd1, hd2, hd3⟩ := hP
  obtain ⟨u, hu, hS, hR, -, hdrop, hrelax, horig⟩ := hst
  have hSR' : σ'.S ⊆ σ'.R := by
    intro v hv
    rw [hS, mem_insert] at hv
    rcases hv with rfl | hv
    · exact hR hu.1
    · exact hR (hSR hv)
  have hd0' : σ'.d 0 = 0 := by
    have := hdrop 0 h0R
    omega
  refine ⟨hR h0R, hd0', hSR', ?_, ?_, ?_, ?_⟩
  · intro h2
    rcases horig 2 h2 with ⟨h2R, -⟩ | ⟨hu2, -⟩
    · exact hR (h21 h2R)
    · obtain rfl := nfE_into_2 u hu2
      by_cases h1S : (1 : Fin 4) ∈ σ'.S
      · exact hSR' h1S
      · exact (hrelax 1 (by decide) h1S).1
  · intro h1
    rcases horig 1 h1 with ⟨h1R, hd⟩ | ⟨hu1, hd⟩
    · rw [hd]; exact hd1 h1R
    · rcases nfE_into_1 u hu1 with rfl | rfl
      · rw [nfA_ne 0 1 _ (by decide)] at hd
        omega
      · have h2R : (2 : Fin 4) ∈ σ.R := hu.1
        have h1R := h21 h2R
        have hle := hdrop 1 h1R
        rw [nfA_ne 2 1 _ (by decide), hd2 h2R] at hd
        have := hd1 h1R
        omega
  · intro h2
    rcases horig 2 h2 with ⟨h2R, hd⟩ | ⟨hu2, hd⟩
    · rw [hd]; exact hd2 h2R
    · obtain rfl := nfE_into_2 u hu2
      rw [nfA_ne 0 2 _ (by decide)] at hd
      omega
  · intro h3
    rcases horig 3 h3 with ⟨h3R, hd⟩ | ⟨hu3, hd⟩
    · rw [hd]; exact hd3 h3R
    · obtain rfl := nfE_into_3 u hu3
      rw [nfA_13, hd1 hu.1] at hd
      split_ifs at hd <;> omega

/-- `nfP` holds at every state of every run from `0` at tick 0. -/
theorem nfP_reachable (σ₀ σ : State (Fin 4) ℕ) (h0 : Init 0 0 σ₀)
    (hrun : Relation.ReflTransGen (Step nfE nfA) σ₀ σ) : nfP σ := by
  induction hrun with
  | refl => exact nfP_init σ₀ h0
  | tail _ hst ih => exact nfP_step _ _ ih hst

/-- L10c, FIFO is necessary, for every run. On the counterexample network, every run of
the label-setting loop from `0` at tick 0 that reaches exhaustion settles the exit at
label 10, whatever tie-breaking it uses, while the earliest arrival is 3
(`nf_isLeast`). The failure is the algorithm's, not an unlucky choice. -/
theorem fifo_necessary_every_run (σ₀ σ : State (Fin 4) ℕ) (h0 : Init 0 0 σ₀)
    (hrun : Relation.ReflTransGen (Step nfE nfA) σ₀ σ) (hdry : σ.R ⊆ σ.S) :
    3 ∈ σ.S ∧ σ.d 3 = 10 ∧ ¬ IsLeast (arrivals nfE nfA 0 0 3) (σ.d 3) := by
  have hP := nfP_reachable σ₀ σ h0 hrun
  have h3 : (3 : Fin 4) ∈ σ.S :=
    ((exhausted_without_fifo nfE nfA 0 0 σ₀ σ h0 hrun hdry).1 3).1 ⟨3, nf_fast⟩
  have hd3 : σ.d 3 = 10 := hP.2.2.2.2.2.2 (hP.2.2.1 h3)
  refine ⟨h3, hd3, fun h => ?_⟩
  have := h.unique nf_isLeast
  omega

end NonFIFO

/-! ## (5) Receding-horizon replanning -/

/-- L11 with the router. Suppose the plan `p ++ q` is an earliest-arrival walk from `s`
at `t0` to `d`, and the evacuee has followed `p`. Re-running the router from where `p`
ends, at the time `p` reaches it, under the unchanged forecast, settles `d` at the
plan's arrival time. For Studio: "live rerouting" with a fixed precomputed fire never
changes the arrival time; it adds information only when the forecast changes. -/
theorem replan_consistent [DecidableEq V] [PartialOrder α] (E : V → V → Prop)
    (A : V → V → α → α) (hfifo : ∀ u v, E u v → Monotone (A u v))
    (hge : ∀ u v, E u v → ∀ t, t ≤ A u v t) (s : V) (t0 : α) (d : V) (p q : List V)
    (hplan : IsEA E A s t0 d (p ++ q)) (σ₀ σ : State V α)
    (h0 : Init (wEnd s p) (wArr A s t0 p) σ₀) (hrun : Relation.ReflTransGen (Step E A) σ₀ σ)
    (hd : d ∈ σ.S) : σ.d d = wArr A s t0 (p ++ q) := by
  have h1 := settled_optimal E A hfifo hge _ _ σ₀ σ h0 hrun d hd
  have h2 := isLeast_of_isEA E A _ _ d q (L11_suffix_ea E A s t0 d p q hplan)
  rw [wArr_append]
  exact h1.unique h2

/-- Replanning, router to router. The first run from `s` at `t0` settles `d` at label
`σ.d d`, and `p ++ q` is a route to `d` arriving at that label (the planned route). If
the evacuee follows `p` and the router is re-run from where `p` ends, at the time it
gets there, the second run settles `d` at the same label, provided the forecast is
unchanged. -/
theorem replan_same_label [DecidableEq V] [PartialOrder α] (E : V → V → Prop)
    (A : V → V → α → α) (hfifo : ∀ u v, E u v → Monotone (A u v))
    (hge : ∀ u v, E u v → ∀ t, t ≤ A u v t) (s : V) (t0 : α) (d : V)
    (σ₀ σ : State V α) (h0 : Init s t0 σ₀) (hrun : Relation.ReflTransGen (Step E A) σ₀ σ)
    (hd : d ∈ σ.S) (p q : List V) (hw : IsWalk E s (p ++ q)) (he : wEnd s (p ++ q) = d)
    (harr : wArr A s t0 (p ++ q) = σ.d d) (σ₀' σ' : State V α)
    (h0' : Init (wEnd s p) (wArr A s t0 p) σ₀')
    (hrun' : Relation.ReflTransGen (Step E A) σ₀' σ') (hd' : d ∈ σ'.S) :
    σ'.d d = σ.d d := by
  have hplan : IsEA E A s t0 d (p ++ q) := isEA_of_isLeast E A s t0 d _ hw he
    (harr ▸ settled_optimal E A hfifo hge s t0 σ₀ σ h0 hrun d hd)
  rw [replan_consistent E A hfifo hge s t0 d p q hplan σ₀' σ' h0' hrun' hd', harr]

end FireEgress.TimeDependent

#print axioms FireEgress.TimeDependent.Arr.cast
#print axioms FireEgress.TimeDependent.arr_of_walk
#print axioms FireEgress.TimeDependent.arr_iff_walk
#print axioms FireEgress.TimeDependent.arr_ge_start
#print axioms FireEgress.TimeDependent.isLeast_of_isEA
#print axioms FireEgress.TimeDependent.isEA_of_isLeast
#print axioms FireEgress.TimeDependent.wArr_eq_pathArr
#print axioms FireEgress.TimeDependent.arcs_forall
#print axioms FireEgress.TimeDependent.walk_fifo
#print axioms FireEgress.TimeDependent.walk_ge
#print axioms FireEgress.TimeDependent.cycle_never_helps
#print axioms FireEgress.TimeDependent.exists_cycle
#print axioms FireEgress.TimeDependent.exists_simple_walk_le
#print axioms FireEgress.TimeDependent.simple_walk_suffices
#print axioms FireEgress.TimeDependent.lowerBounds_simple
#print axioms FireEgress.TimeDependent.isGLB_arrivals_iff_simple
#print axioms FireEgress.TimeDependent.ea_walk_exists
#print axioms FireEgress.TimeDependent.earliest_exists
#print axioms FireEgress.TimeDependent.arrW_of_arr
#print axioms FireEgress.TimeDependent.waiting_never_helps
#print axioms FireEgress.TimeDependent.isLeast_waiting
#print axioms FireEgress.TimeDependent.bellman_subsolution_le
#print axioms FireEgress.TimeDependent.ea_bellman
#print axioms FireEgress.TimeDependent.bm_lower
#print axioms FireEgress.TimeDependent.bellman_least_fixed_point_false
#print axioms FireEgress.TimeDependent.init_inv
#print axioms FireEgress.TimeDependent.selectable_le
#print axioms FireEgress.TimeDependent.selectable_optimal
#print axioms FireEgress.TimeDependent.step_inv
#print axioms FireEgress.TimeDependent.reachable_inv
#print axioms FireEgress.TimeDependent.settled_optimal
#print axioms FireEgress.TimeDependent.td_dijkstra_correct
#print axioms FireEgress.TimeDependent.td_dijkstra_exhausted
#print axioms FireEgress.TimeDependent.never_reached_no_walk
#print axioms FireEgress.TimeDependent.td_dijkstra_glb
#print axioms FireEgress.TimeDependent.td_dijkstra_EA_withTop
#print axioms FireEgress.TimeDependent.td_dijkstra_EA
#print axioms FireEgress.TimeDependent.early_exit_lower_bound
#print axioms FireEgress.TimeDependent.td_dijkstra_stop_at_top
#print axioms FireEgress.TimeDependent.relaxState_step
#print axioms FireEgress.TimeDependent.step_settles_new
#print axioms FireEgress.TimeDependent.exists_exhausted
#print axioms FireEgress.TimeDependent.td_dijkstra_total
#print axioms FireEgress.TimeDependent.static_hyps
#print axioms FireEgress.TimeDependent.closeArc_ge
#print axioms FireEgress.TimeDependent.hopArc_ge
#print axioms FireEgress.TimeDependent.closure_window_exact
#print axioms FireEgress.TimeDependent.hop_rule_exact
#print axioms FireEgress.TimeDependent.sound_init
#print axioms FireEgress.TimeDependent.sound_step
#print axioms FireEgress.TimeDependent.sound_reachable
#print axioms FireEgress.TimeDependent.exhausted_without_fifo
#print axioms FireEgress.TimeDependent.nfA_13
#print axioms FireEgress.TimeDependent.nfA_ne
#print axioms FireEgress.TimeDependent.nf_init
#print axioms FireEgress.TimeDependent.nf_step01
#print axioms FireEgress.TimeDependent.nf_step12
#print axioms FireEgress.TimeDependent.nf_step23
#print axioms FireEgress.TimeDependent.nf_step34
#print axioms FireEgress.TimeDependent.nf_run
#print axioms FireEgress.TimeDependent.nf_lower
#print axioms FireEgress.TimeDependent.nf_fast
#print axioms FireEgress.TimeDependent.nf_isLeast
#print axioms FireEgress.TimeDependent.nf_wait
#print axioms FireEgress.TimeDependent.fifo_necessary
#print axioms FireEgress.TimeDependent.nfE_into_1
#print axioms FireEgress.TimeDependent.nfE_into_2
#print axioms FireEgress.TimeDependent.nfE_into_3
#print axioms FireEgress.TimeDependent.nfP_init
#print axioms FireEgress.TimeDependent.nfP_step
#print axioms FireEgress.TimeDependent.nfP_reachable
#print axioms FireEgress.TimeDependent.fifo_necessary_every_run
#print axioms FireEgress.TimeDependent.replan_consistent
#print axioms FireEgress.TimeDependent.replan_same_label
