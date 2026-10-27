import Mathlib
import FireEgress.Heuristic

/-!
# A* returns the optimum (L9)

Studio's browser router `astar` (`src/core/route.js`) pops the open node of least
`g + h` from a binary heap, closes it, and relaxes its open links; it stops when it
pops the goal. `h(u)` is the straight-line distance to the goal, and every link
weight is at least the link's length (plain lengths, or the hazard weights
`length * (1 + α h)`, which only scale up: `Heuristic.hazard_weight_ge_length`).
`Heuristic.lean` proved that this `h` is consistent and admissible. Admissibility
alone does not make a closed-set A* exact; this file proves that it is, in three
steps:

1. **Reduced weights are non-negative.** With a consistent `h`,
   `w'(u, v) = w(u, v) - h(u) + h(v) ≥ 0`.
2. **A* is Dijkstra on `w'`.** Along any walk from `s` to `t` the reduced cost is
   `cost - h(s) + h(t)` (telescoping), so the walks that minimise `cost` and
   `cost'` are the same walks; and one A* step on `(w, h)` is, label for label,
   one Dijkstra step on `w'` with labels shifted by `h(v) - h(s)`.
3. **Dijkstra is exact.** For the label-setting loop (settled set, reached set,
   tentative labels, settle a reached unsettled vertex of least label, relax its
   links to unsettled vertices), every vertex the loop may settle carries its true
   shortest distance, and when the loop runs dry every vertex with a walk from the
   source is settled.

Modelling choices, stated so a reviewer can compare with the code:
* The graph is a relation `E u v a`: "there is a usable link from `u` to `v` of
  weight `a`". This covers parallel links, one-way links, and blocked links (simply
  absent from `E`); `route.js` builds both directions of every unblocked edge.
* A walk is recorded by its cost: `WalkCost E s v c` says some walk from `s` to `v`
  has total weight `c`. "Optimal" means the label is the cost of some walk and is
  `≤` the cost of every walk.
* The step is a relation, not a program: it fixes what the code guarantees after
  popping `u` and scanning its links (settled labels untouched, labels only drop,
  each open neighbour `v` ends with `d v ≤ d u + a`, and each new label is an old
  label or `d u + a` for a link `u → v`), not the order of the scan. The binary
  heap is abstracted to "the popped vertex has least `d + h` among reached,
  unsettled vertices" (`Selectable`); the heap itself is not verified.
* Arithmetic is over `ℝ`. `route.js` computes in IEEE doubles; the theorems say
  nothing about rounding.
* The vertex type need not be finite for correctness; finiteness is only needed for
  termination, which is not formalised here.
-/

namespace FireEgress.AStar

open Finset

variable {V : Type*}

/-! ## (1) Reduced weights -/

/-- L9.1. A consistent heuristic (`h u ≤ w + h v` along a link of weight `w`)
makes the reduced weight `w - h u + h v` non-negative. -/
theorem reduced_nonneg (h : V → ℝ) (u v : V) (w : ℝ) (hc : h u ≤ w + h v) :
    0 ≤ w - h u + h v := by
  linarith

/-- L9.1 for Studio's heuristic. If a link's weight is at least its length
(`dist u v ≤ w`), the reduced weight under the straight-line heuristic
`h x = dist x g` is non-negative. Reuses `Heuristic.consistent`. -/
theorem reduced_nonneg_straight_line {P : Type*} [MetricSpace P] (g u v : P) (w : ℝ)
    (hw : dist u v ≤ w) : 0 ≤ w - dist u g + dist v g := by
  have := FireEgress.Heuristic.consistent g u v w hw
  linarith

/-- L9.1 with fire weighting. A hazard weight `length * (1 + α hz)` with `α, hz ≥ 0`
on a link whose `length` is the straight-line distance between its ends keeps the
reduced weight non-negative. Reuses `Heuristic.hazard_weight_ge_length`. -/
theorem reduced_nonneg_hazard {P : Type*} [MetricSpace P] (g u v : P) (α hz : ℝ)
    (hα : 0 ≤ α) (hhz : 0 ≤ hz) :
    0 ≤ dist u v * (1 + α * hz) - dist u g + dist v g :=
  reduced_nonneg_straight_line g u v _
    (FireEgress.Heuristic.hazard_weight_ge_length _ α hz dist_nonneg hα hhz)

/-! ## (2) Reduced walk cost: A* is Dijkstra on `w'` -/

/-- L9.2. Telescoping: along a walk `x 0, …, x n` whose `i`-th link has weight
`w i`, the reduced cost is the cost minus `h(start)` plus `h(end)`:
`∑ (w i - h (x i) + h (x (i+1))) = ∑ w i - h (x 0) + h (x n)`. -/
theorem reduced_walk_cost (h : V → ℝ) (x : ℕ → V) (w : ℕ → ℝ) (n : ℕ) :
    ∑ i ∈ range n, (w i - h (x i) + h (x (i + 1))) =
      (∑ i ∈ range n, w i) - h (x 0) + h (x n) := by
  induction n with
  | zero => simp
  | succ n ih => rw [sum_range_succ, sum_range_succ, ih]; ring

/-- L9.2. Two walks with the same start and the same end compare the same way
under reduced and original weights. So the walks from `s` to `t` minimising the
reduced cost are exactly those minimising the cost: running Dijkstra on `w'` finds
the same routes as minimising `w`. -/
theorem reduced_le_iff (h : V → ℝ) (x y : ℕ → V) (wx wy : ℕ → ℝ) (n m : ℕ)
    (h0 : x 0 = y 0) (hn : x n = y m) :
    (∑ i ∈ range n, (wx i - h (x i) + h (x (i + 1)))) ≤
        ∑ i ∈ range m, (wy i - h (y i) + h (y (i + 1))) ↔
      ∑ i ∈ range n, wx i ≤ ∑ i ∈ range m, wy i := by
  rw [reduced_walk_cost, reduced_walk_cost, h0, hn]
  constructor <;> intro hle <;> linarith

/-- There is a walk from `s` to `v` over links of `E` with total weight `c`. -/
inductive WalkCost (E : V → V → ℝ → Prop) (s : V) : V → ℝ → Prop
  | nil : WalkCost E s s 0
  | snoc {u v : V} {c a : ℝ} : WalkCost E s u c → E u v a → WalkCost E s v (c + a)

/-- A walk's recorded cost can be rewritten by an equal expression. -/
theorem WalkCost.cast {E : V → V → ℝ → Prop} {s v : V} {c c' : ℝ}
    (hw : WalkCost E s v c) (hc : c = c') : WalkCost E s v c' := hc ▸ hw

/-- The reduced graph: a link `u → v` of weight `a` becomes a link of weight
`a - h u + h v`. -/
def reduced (E : V → V → ℝ → Prop) (h : V → ℝ) : V → V → ℝ → Prop :=
  fun u v a' => ∃ a, E u v a ∧ a' = a - h u + h v

/-- L9.1 on the reduced graph: a consistent `h` gives non-negative reduced links. -/
theorem reduced_links_nonneg (E : V → V → ℝ → Prop) (h : V → ℝ)
    (hc : ∀ u v a, E u v a → h u ≤ a + h v) :
    ∀ u v a', reduced E h u v a' → 0 ≤ a' := by
  rintro u v a' ⟨a, ha, rfl⟩
  exact reduced_nonneg h u v a (hc u v a ha)

/-- L9.2 for walks in the graph model: a walk from `s` to `v` of cost `c` exists
exactly when a walk of reduced cost `c - h s + h v` exists. Shortest distances in
the two graphs therefore differ by `h v - h s`, with the same minimising walks. -/
theorem walkCost_reduced_iff (E : V → V → ℝ → Prop) (h : V → ℝ) (s v : V) (c : ℝ) :
    WalkCost E s v c ↔ WalkCost (reduced E h) s v (c - h s + h v) := by
  constructor
  · intro hw
    induction hw with
    | nil => exact WalkCost.nil.cast (by ring)
    | @snoc u v c a _ hE ih =>
      exact (ih.snoc ⟨a, hE, rfl⟩).cast (by ring)
  · intro hw
    have key : ∀ x c', WalkCost (reduced E h) s x c' → WalkCost E s x (c' + h s - h x) := by
      intro x c' hw'
      induction hw' with
      | nil => exact WalkCost.nil.cast (by ring)
      | @snoc u v c' a' _ hE ih =>
        obtain ⟨a, ha, rfl⟩ := hE
        exact (ih.snoc ha).cast (by ring)
    exact (key v _ hw).cast (by ring)

/-! ## (3) The label-setting loop -/

/-- The loop's state: settled (closed) vertices `S`, reached vertices `R` (those
with a finite label; `route.js` keeps `Infinity` for the others) and labels `d`. -/
structure State (V : Type*) where
  S : Finset V
  R : Finset V
  d : V → ℝ

/-- The start of the loop: nothing settled, only the source reached, at label 0. -/
def Init (s : V) (σ : State V) : Prop := σ.S = ∅ ∧ σ.R = {s} ∧ σ.d s = 0

/-- `u` may be popped next: reached, not settled, and of least `d + key` among the
reached, unsettled vertices. Dijkstra uses `key = 0`; A* uses `key = h`. -/
def Selectable (key : V → ℝ) (σ : State V) (u : V) : Prop :=
  u ∈ σ.R ∧ u ∉ σ.S ∧ ∀ v ∈ σ.R, v ∉ σ.S → σ.d u + key u ≤ σ.d v + key v

variable [DecidableEq V]

/-- One iteration: pop a selectable `u`, settle it, and relax its links into
unsettled vertices. In order, the clauses say: `u` is settled; nothing is
un-reached; settled labels (including `u`'s) do not change; labels only drop;
every link `u → v` of weight `a` into an unsettled `v` leaves `v` reached with
`d v ≤ d u + a`; and every label afterwards is an old label or `d u + a` for a
link `u → v`. -/
def Step (E : V → V → ℝ → Prop) (key : V → ℝ) (σ σ' : State V) : Prop :=
  ∃ u, Selectable key σ u ∧ σ'.S = insert u σ.S ∧ σ.R ⊆ σ'.R ∧
    (∀ v ∈ σ'.S, σ'.d v = σ.d v) ∧
    (∀ v ∈ σ.R, σ'.d v ≤ σ.d v) ∧
    (∀ v a, E u v a → v ∉ σ'.S → v ∈ σ'.R ∧ σ'.d v ≤ σ.d u + a) ∧
    (∀ v ∈ σ'.R, (v ∈ σ.R ∧ σ'.d v = σ.d v) ∨ ∃ a, E u v a ∧ σ'.d v = σ.d u + a)

/-- Dijkstra's invariant: the source is reached with label at most 0, settled
vertices are reached, every reached label is the cost of a real walk, every settled
label is the shortest distance, and every link out of a settled vertex into an
unsettled one has been relaxed. -/
structure Inv (E : V → V → ℝ → Prop) (s : V) (σ : State V) : Prop where
  src : s ∈ σ.R ∧ σ.d s ≤ 0
  sub : σ.S ⊆ σ.R
  real : ∀ v ∈ σ.R, WalkCost E s v (σ.d v)
  opt : ∀ v ∈ σ.S, ∀ c, WalkCost E s v c → σ.d v ≤ c
  relaxed : ∀ x ∈ σ.S, ∀ v a, E x v a → v ∉ σ.S → v ∈ σ.R ∧ σ.d v ≤ σ.d x + a

omit [DecidableEq V] in
/-- The loop starts in a state that satisfies the invariant. -/
theorem init_inv (E : V → V → ℝ → Prop) (s : V) (σ : State V) (h : Init s σ) :
    Inv E s σ := by
  obtain ⟨hS, hR, hd⟩ := h
  refine ⟨⟨by simp [hR], by simp [hd]⟩, by simp [hS], ?_, by simp [hS], by simp [hS]⟩
  intro v hv
  rw [hR, mem_singleton] at hv
  subst hv
  exact WalkCost.nil.cast hd.symm

/-- The core of Dijkstra's argument: with non-negative links, every walk from the
source to an unsettled vertex costs at least the label of a selectable vertex
(the walk must leave the settled set through a relaxed link). -/
theorem selectable_le (E : V → V → ℝ → Prop) (hE : ∀ u v a, E u v a → 0 ≤ a) (s : V)
    (σ : State V) (hI : Inv E s σ) (u : V) (hu : Selectable (fun _ => 0) σ u) :
    ∀ x c, WalkCost E s x c → x ∉ σ.S → σ.d u ≤ c := by
  intro x c hw
  induction hw with
  | nil =>
    intro hs
    have := hu.2.2 s hI.src.1 hs
    linarith [hI.src.2]
  | @snoc y x c a hw' hyx ih =>
    intro hx
    by_cases hy : y ∈ σ.S
    · obtain ⟨hxR, hxd⟩ := hI.relaxed y hy x a hyx hx
      have h1 := hI.opt y hy c hw'
      have h2 := hu.2.2 x hxR hx
      linarith
    · have := ih hy
      have := hE _ _ _ hyx
      linarith

/-- L9.3 (key lemma). With non-negative links, a vertex Dijkstra may settle next
carries its true shortest distance: its label is the cost of a walk from the
source, and no walk from the source to it is cheaper. -/
theorem selectable_optimal (E : V → V → ℝ → Prop) (hE : ∀ u v a, E u v a → 0 ≤ a)
    (s : V) (σ : State V) (hI : Inv E s σ) (u : V) (hu : Selectable (fun _ => 0) σ u) :
    WalkCost E s u (σ.d u) ∧ ∀ c, WalkCost E s u c → σ.d u ≤ c :=
  ⟨hI.real u hu.1, fun c hc => selectable_le E hE s σ hI u hu u c hc hu.2.1⟩

/-- L9.3. A Dijkstra step on non-negative links preserves the invariant. -/
theorem step_inv (E : V → V → ℝ → Prop) (hE : ∀ u v a, E u v a → 0 ≤ a) (s : V)
    (σ σ' : State V) (hI : Inv E s σ) (hst : Step E (fun _ => 0) σ σ') : Inv E s σ' := by
  obtain ⟨u, hu, hS, hR, hfix, hdrop, hrelax, horig⟩ := hst
  have huopt := selectable_optimal E hE s σ hI u hu
  refine ⟨⟨hR hI.src.1, le_trans (hdrop s hI.src.1) hI.src.2⟩, ?_, ?_, ?_, ?_⟩
  · intro v hv
    rw [hS, mem_insert] at hv
    rcases hv with rfl | hv
    · exact hR hu.1
    · exact hR (hI.sub hv)
  · intro v hv
    rcases horig v hv with ⟨hvR, hd⟩ | ⟨a, ha, hd⟩
    · rw [hd]; exact hI.real v hvR
    · rw [hd]; exact (hI.real u hu.1).snoc ha
  · intro v hv c hc
    rw [hfix v hv]
    rw [hS, mem_insert] at hv
    rcases hv with rfl | hv
    · exact huopt.2 c hc
    · exact hI.opt v hv c hc
  · intro x hx v a hxv hv
    have hx' := hx
    rw [hS, mem_insert] at hx'
    rcases hx' with rfl | hxS
    · obtain ⟨h1, h2⟩ := hrelax v a hxv hv
      exact ⟨h1, by rw [hfix x hx]; exact h2⟩
    · have hvS : v ∉ σ.S := fun h => hv (by rw [hS]; exact mem_insert_of_mem h)
      obtain ⟨hvR, hvd⟩ := hI.relaxed x hxS v a hxv hvS
      refine ⟨hR hvR, ?_⟩
      rw [hfix x hx]
      exact le_trans (hdrop v hvR) hvd

/-- Every state Dijkstra can reach from its start satisfies the invariant. -/
theorem reachable_inv (E : V → V → ℝ → Prop) (hE : ∀ u v a, E u v a → 0 ≤ a) (s : V)
    (σ₀ σ : State V) (h0 : Init s σ₀) (hrun : Relation.ReflTransGen (Step E (fun _ => 0)) σ₀ σ) :
    Inv E s σ := by
  induction hrun with
  | refl => exact init_inv E s σ₀ h0
  | tail _ hst ih => exact step_inv E hE s _ _ ih hst

/-- L9.3. Dijkstra is exact. On links of non-negative weight, at any point of any
run from the start, the vertex popped next has as label its true shortest distance
from the source (the cost of some walk, and at most the cost of every walk). -/
theorem dijkstra_correct (E : V → V → ℝ → Prop) (hE : ∀ u v a, E u v a → 0 ≤ a) (s : V)
    (σ₀ σ : State V) (h0 : Init s σ₀) (hrun : Relation.ReflTransGen (Step E (fun _ => 0)) σ₀ σ)
    (u : V) (hu : Selectable (fun _ => 0) σ u) :
    WalkCost E s u (σ.d u) ∧ ∀ c, WalkCost E s u c → σ.d u ≤ c :=
  selectable_optimal E hE s σ (reachable_inv E hE s σ₀ σ h0 hrun) u hu

/-- L9.3. When Dijkstra's queue runs dry (every reached vertex is settled), the
settled vertices are exactly those with a walk from the source, and each carries
its shortest distance. This is `distancesFrom` in `route.js`. -/
theorem dijkstra_exhausted (E : V → V → ℝ → Prop) (hE : ∀ u v a, E u v a → 0 ≤ a) (s : V)
    (σ₀ σ : State V) (h0 : Init s σ₀) (hrun : Relation.ReflTransGen (Step E (fun _ => 0)) σ₀ σ)
    (hdry : σ.R ⊆ σ.S) :
    (∀ v, (∃ c, WalkCost E s v c) ↔ v ∈ σ.S) ∧
      ∀ v ∈ σ.S, WalkCost E s v (σ.d v) ∧ ∀ c, WalkCost E s v c → σ.d v ≤ c := by
  have hI := reachable_inv E hE s σ₀ σ h0 hrun
  refine ⟨fun v => ⟨?_, fun hv => ⟨_, hI.real v (hI.sub hv)⟩⟩,
    fun v hv => ⟨hI.real v (hI.sub hv), hI.opt v hv⟩⟩
  rintro ⟨c, hw⟩
  induction hw with
  | nil => exact hdry hI.src.1
  | @snoc y x c a _ hyx ih =>
    by_contra hx
    exact hx (hdry (hI.relaxed y ih x a hyx hx).1)

/-! ## A* is Dijkstra on the reduced weights -/

/-- A* labels seen as Dijkstra labels on the reduced graph: `d v + h v - h s`. -/
def shift (h : V → ℝ) (s : V) (σ : State V) : State V :=
  ⟨σ.S, σ.R, fun v => σ.d v + h v - h s⟩

omit [DecidableEq V] in
/-- A* and Dijkstra start from the same state: the source's label 0 shifted by
`h s - h s` is still 0. -/
theorem shift_init (h : V → ℝ) (s : V) (σ : State V) (h0 : Init s σ) :
    Init s (shift h s σ) := by
  obtain ⟨hS, hR, hd⟩ := h0
  exact ⟨hS, hR, by simp [shift, hd]⟩

omit [DecidableEq V] in
/-- The vertex A* pops (least `g + h`) is a vertex Dijkstra may pop on the shifted
labels (least `g + h - h s`). -/
theorem shift_selectable (h : V → ℝ) (s : V) (σ : State V) (u : V)
    (hu : Selectable h σ u) : Selectable (fun _ => 0) (shift h s σ) u := by
  obtain ⟨huR, huS, hmin⟩ := hu
  refine ⟨huR, huS, fun v hv hvS => ?_⟩
  have := hmin v hv hvS
  simp only [shift]
  linarith

/-- L9.2. One A* step (pop the least `g + h`, relax with `w`) is one Dijkstra step
(pop the least label, relax with `w' = w - h u + h v`) on the shifted labels. -/
theorem astar_step_is_dijkstra_step (E : V → V → ℝ → Prop) (h : V → ℝ) (s : V)
    (σ σ' : State V) (hst : Step E h σ σ') :
    Step (reduced E h) (fun _ => 0) (shift h s σ) (shift h s σ') := by
  obtain ⟨u, hu, hS, hR, hfix, hdrop, hrelax, horig⟩ := hst
  refine ⟨u, shift_selectable h s σ u hu, hS, hR, ?_, ?_, ?_, ?_⟩
  · intro v hv
    simp only [shift] at hv ⊢
    rw [hfix v hv]
  · intro v hv
    simp only [shift] at hv ⊢
    linarith [hdrop v hv]
  · rintro v a' ⟨a, ha, rfl⟩ hv
    simp only [shift] at hv ⊢
    obtain ⟨h1, h2⟩ := hrelax v a ha hv
    exact ⟨h1, by linarith⟩
  · intro v hv
    simp only [shift] at hv ⊢
    rcases horig v hv with ⟨hvR, hd⟩ | ⟨a, ha, hd⟩
    · exact Or.inl ⟨hvR, by rw [hd]⟩
    · exact Or.inr ⟨a - h u + h v, ⟨a, ha, rfl⟩, by rw [hd]; ring⟩

/-- L9.2. Every A* run is, step for step, a Dijkstra run on the reduced weights. -/
theorem astar_run_is_dijkstra_run (E : V → V → ℝ → Prop) (h : V → ℝ) (s : V)
    (σ₀ σ : State V) (hrun : Relation.ReflTransGen (Step E h) σ₀ σ) :
    Relation.ReflTransGen (Step (reduced E h) (fun _ => 0)) (shift h s σ₀) (shift h s σ) := by
  induction hrun with
  | refl => exact Relation.ReflTransGen.refl
  | tail _ hst ih => exact ih.tail (astar_step_is_dijkstra_step E h s _ _ hst)

/-- L9. A* with a consistent heuristic is exact. At any point of any run from the
start, the vertex A* pops next (least `g + h` among open vertices) has as label
`g` its true shortest distance from the source: the cost of some walk, and at
most the cost of every walk. In particular when the goal is popped - where
`route.js` stops - `dist[end]` is the optimum. -/
theorem astar_correct (E : V → V → ℝ → Prop) (h : V → ℝ)
    (hc : ∀ u v a, E u v a → h u ≤ a + h v) (s : V) (σ₀ σ : State V) (h0 : Init s σ₀)
    (hrun : Relation.ReflTransGen (Step E h) σ₀ σ) (u : V) (hu : Selectable h σ u) :
    WalkCost E s u (σ.d u) ∧ ∀ c, WalkCost E s u c → σ.d u ≤ c := by
  obtain ⟨hreal, hopt⟩ := dijkstra_correct (reduced E h) (reduced_links_nonneg E h hc) s
    (shift h s σ₀) (shift h s σ) (shift_init h s σ₀ h0) (astar_run_is_dijkstra_run E h s σ₀ σ hrun)
    u (shift_selectable h s σ u hu)
  refine ⟨?_, fun c hw => ?_⟩
  · rw [walkCost_reduced_iff E h]
    exact hreal.cast (by simp only [shift]; ring)
  · have := hopt _ ((walkCost_reduced_iff E h s u c).1 hw)
    simp only [shift] at this
    linarith

/-- L9. When A*'s heap runs dry without popping the goal, the vertices it settled
are exactly those reachable from the source; so an unsettled goal has no walk
from the source, and `route.js` is right to return `null`. -/
theorem astar_exhausted (E : V → V → ℝ → Prop) (h : V → ℝ)
    (hc : ∀ u v a, E u v a → h u ≤ a + h v) (s : V) (σ₀ σ : State V) (h0 : Init s σ₀)
    (hrun : Relation.ReflTransGen (Step E h) σ₀ σ) (hdry : σ.R ⊆ σ.S) (v : V) :
    (∃ c, WalkCost E s v c) ↔ v ∈ σ.S := by
  have := (dijkstra_exhausted (reduced E h) (reduced_links_nonneg E h hc) s
    (shift h s σ₀) (shift h s σ) (shift_init h s σ₀ h0)
    (astar_run_is_dijkstra_run E h s σ₀ σ hrun) hdry).1 v
  simp only [shift] at this
  rw [← this]
  constructor
  · rintro ⟨c, hw⟩; exact ⟨_, (walkCost_reduced_iff E h s v c).1 hw⟩
  · rintro ⟨c', hw⟩
    exact ⟨c' + h s - h v, (walkCost_reduced_iff E h s v _).2 (hw.cast (by ring))⟩

/-- L9 for Studio's router. Vertices sit at points `pos v` of a metric space, the
goal is `t`, the heuristic is the straight-line distance `dist (pos v) (pos t)`,
and every link weighs at least the straight-line distance between its ends (plain
lengths, or hazard weights - `reduced_nonneg_hazard`). Then the vertex A* pops
next, and in particular the goal when `route.js` stops, carries the least cost of
any walk from the source in the graph. -/
theorem astar_straight_line_correct {P : Type*} [MetricSpace P] (pos : V → P) (t : V)
    (E : V → V → ℝ → Prop) (hlen : ∀ u v a, E u v a → dist (pos u) (pos v) ≤ a)
    (s : V) (σ₀ σ : State V) (h0 : Init s σ₀)
    (hrun : Relation.ReflTransGen (Step E (fun v => dist (pos v) (pos t))) σ₀ σ) (u : V)
    (hu : Selectable (fun v => dist (pos v) (pos t)) σ u) :
    WalkCost E s u (σ.d u) ∧ ∀ c, WalkCost E s u c → σ.d u ≤ c :=
  astar_correct E _ (fun u v a ha => FireEgress.Heuristic.consistent _ _ _ a (hlen u v a ha))
    s σ₀ σ h0 hrun u hu

end FireEgress.AStar

#print axioms FireEgress.AStar.reduced_nonneg
#print axioms FireEgress.AStar.reduced_nonneg_straight_line
#print axioms FireEgress.AStar.reduced_nonneg_hazard
#print axioms FireEgress.AStar.reduced_walk_cost
#print axioms FireEgress.AStar.reduced_le_iff
#print axioms FireEgress.AStar.WalkCost.cast
#print axioms FireEgress.AStar.reduced_links_nonneg
#print axioms FireEgress.AStar.walkCost_reduced_iff
#print axioms FireEgress.AStar.init_inv
#print axioms FireEgress.AStar.selectable_le
#print axioms FireEgress.AStar.selectable_optimal
#print axioms FireEgress.AStar.step_inv
#print axioms FireEgress.AStar.reachable_inv
#print axioms FireEgress.AStar.dijkstra_correct
#print axioms FireEgress.AStar.dijkstra_exhausted
#print axioms FireEgress.AStar.shift_init
#print axioms FireEgress.AStar.shift_selectable
#print axioms FireEgress.AStar.astar_step_is_dijkstra_step
#print axioms FireEgress.AStar.astar_run_is_dijkstra_run
#print axioms FireEgress.AStar.astar_correct
#print axioms FireEgress.AStar.astar_exhausted
#print axioms FireEgress.AStar.astar_straight_line_correct
