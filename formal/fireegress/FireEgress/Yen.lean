import Mathlib
import FireEgress.AStar

/-!
# Alternative routes: Yen's k shortest loopless paths, and "distinct" (L12)

Studio's "Alternative routes" (`src/core/alternatives.js`, mirrored by the server's
`app/graphs/alternatives.py`) offers two methods.

* **Yen** (`method: "yen"`, function `yen`). The first route is the A* route. Then, while
  fewer than `k` routes are accepted: for every node `prev[i]` of the last accepted route
  `prev` except the exit (the *spur node* `v`; the *root* is `prev[0..i]`, and `pre` is the
  root without `v`), mask every link touching a node of `pre`, and every link between `v`
  and the next node of each accepted route that begins with the same root; run A* from `v`
  to the exit on what is left; if it finds a spur `S`, the candidate is `pre ++ S` (skipped
  if `seen`). Then move the cheapest candidate to the accepted list. A time budget may stop
  the loop between spur searches; it then returns the accepted list as it stands.
* **distinct** (the default, "Different space", function `distinct`): a penalty method, §4.

What is proved:

1. **Specification (§1).** Simple (loopless) `s`-`t` paths, and `KShortest`: a list of
   distinct simple paths whose costs are non-decreasing, with no simple path outside the
   list cheaper than its last one. Every prefix of a `KShortest` list is one
   (`KShortest.take`); a cheaper-than-everything-outside extension is one (`KShortest.snoc`).
2. **Yen's key lemma (§2).** If `Q` is a simple path not yet accepted, it *deviates* from
   an accepted path `P` at a spur node `v`: the same root up to `v`, then a spur avoiding
   the root's other nodes and the first link out of `v` of every accepted path with that
   root, and no later-accepted path has that root (`deviation`). So if each accepted path's
   spur searches were run, with the accepted list of that moment, some candidate costs at
   most `cost Q` (`yen_key`): the cheapest candidate is a cheapest path not yet accepted.
3. **Yen's output (§3).** With a round (all spur searches from the last accepted path, then
   accept the cheapest candidate) modelled as a relation, every accepted list the loop can
   reach - so also the one returned when the time budget runs out, and every prefix of it -
   is a `KShortest` list (`yen_correct`, `yen_kshortest`). If the candidates run out, every
   simple path has been returned (`yen_dry_complete`), and on a finite graph the loop
   accepts at most as many routes as there are simple paths (`yen_length_le`). The spur
   search's contract (`ShortestPath`) is derived from `AStar.astar_correct`
   (`shortestPath_of_astar`), and "A* found nothing" from `AStar.astar_exhausted`
   (`no_walk_of_astar_dry`).
4. **distinct (§4).** By construction every route after the first costs at most
   `maxStretch * best + 1e-9` and was found by one of the first `4k` penalised searches
   (`distinct_stretch`); each kept route's share near every route kept *before it, in
   discovery order*, is at most `maxOverlap` (`distinct_overlap`); the result depends on
   at most `4k` penalised searches and has at most `max 1 k` routes (`distinct_bounded`);
   and the first route is the A* route, so with the rest sorted by cost the list is
   non-decreasing (`distinct_sorted`). It is **not complete**: it can return fewer than `k`
   routes while `k` routes within the stretch and overlap limits exist, by stopping at the
   first over-stretch penalised route (`distinct_can_miss`) or by spending its `4k`
   attempts on near-duplicates (`distinct_can_miss_cycle`). And the overlap bound is in
   discovery order only: after the sort a route can run more than `maxOverlap` near a
   cheaper route found after it (`distinct_rank_overlap_gap`).

Modelling choices, stated so a reviewer can compare with the code:
* The graph is a link relation `G u v` (the usable links; `route.js` builds both directions
  of every edge, and a blocked edge is absent) and a weight `w u v` per ordered pair of
  nodes. The code works with node lists and, between two nodes, the cheapest parallel edge;
  Studio's parallel edges join the same two points and so have the same length, which is
  why one weight per pair loses nothing for `base = edgeLengths`.
* A route is its node list; its cost is `w` summed over consecutive pairs (`cost`), which is
  `pathCost` in the code, and `rootCost + spur.cost` for a candidate (`candidate_cost`).
* The spur search is specified, not re-implemented: it returns a simple path in the masked
  graph costing no more than any walk there (`ShortestPath`). `shortestPath_of_astar` gets
  this from `AStar.astar_correct`, given that the node list `route.js` rebuilds from `prev`
  is a simple path whose cost is A*'s label; that reconstruction is not formalised here
  (`astar.property.test.mjs` checks it).
* The candidate heap is a set; "accept the cheapest" is any candidate of least cost, so
  every tie-break is covered. `seen` is the union of the accepted list and the candidates.
* Arithmetic is over `ℝ`. The code adds doubles, so routes of equal or nearly equal cost may
  be ordered differently by rounding.
* Non-negative weights enter only through A* (a consistent heuristic); the argument of
  §2-§3 holds for any weights once each spur search meets its contract.
-/

namespace FireEgress.Yen

variable {V : Type*}

/-! ## (1) Specification: simple paths and k-shortest lists -/

/-- Consecutive nodes of the list are joined by links of `G`. -/
def Linked (G : V → V → Prop) : List V → Prop
  | [] => True
  | [_] => True
  | a :: b :: rest => G a b ∧ Linked G (b :: rest)

/-- The cost of a route given as a node list: `w` summed over consecutive pairs. This is
`pathCost` in `alternatives.js` and A*'s label for the route it returns. -/
def cost (w : V → V → ℝ) : List V → ℝ
  | [] => 0
  | [_] => 0
  | a :: b :: rest => w a b + cost w (b :: rest)

/-- A walk from `a` to `b` in `G`: a node list that starts at `a`, ends at `b`, and steps
along links of `G`. -/
structure IsPath (G : V → V → Prop) (a b : V) (p : List V) : Prop where
  head : p.head? = some a
  last : p.getLast? = some b
  linked : Linked G p

/-- A simple (loopless) `s`-`t` path: a walk from `s` to `t` that repeats no node. Yen's
routes are of this kind: `alternatives.test.mjs` checks `new Set(node_ids).size`. -/
structure Simple (G : V → V → Prop) (s t : V) (p : List V) : Prop where
  path : IsPath G s t p
  nodup : p.Nodup

/-- `L` is a list of the `L.length` shortest simple `s`-`t` paths: distinct simple paths,
costs non-decreasing along the list, and no simple path outside the list cheaper than the
last one in it. With ties, several lists qualify; Yen may return any of them. -/
structure KShortest (G : V → V → Prop) (w : V → V → ℝ) (s t : V) (L : List (List V)) :
    Prop where
  nodup : L.Nodup
  simple : ∀ P ∈ L, Simple G s t P
  sorted : L.Pairwise (fun P Q => cost w P ≤ cost w Q)
  last_le : ∀ Q, Simple G s t Q → Q ∉ L → ∀ P ∈ L.getLast?, cost w P ≤ cost w Q

/-- Splitting a node list at a node splits its links: the part up to `v` and the part from
`v` are each linked exactly when the whole is. -/
theorem linked_append (G : V → V → Prop) (v : V) (m : List V) :
    ∀ l : List V, Linked G (l ++ v :: m) ↔ Linked G (l ++ [v]) ∧ Linked G (v :: m)
  | [] => by simp [Linked]
  | [a] => by simp [Linked]
  | a :: b :: l => by
    have ih := linked_append G v m (b :: l)
    simp only [List.cons_append, Linked] at ih ⊢
    rw [ih, and_assoc]

/-- Splitting a route at a node splits its cost: `cost (l ++ v :: m) = cost (l ++ [v]) +
cost (v :: m)`. This is the code's `rootCost + spur.cost`. -/
theorem cost_append (w : V → V → ℝ) (v : V) (m : List V) :
    ∀ l : List V, cost w (l ++ v :: m) = cost w (l ++ [v]) + cost w (v :: m)
  | [] => by simp [cost]
  | [a] => by simp [cost]
  | a :: b :: l => by
    have ih := cost_append w v m (b :: l)
    simp only [List.cons_append, cost] at ih ⊢
    rw [ih]
    ring

/-- A list linked in `G` is linked in `H` if every `G`-link between two of its nodes is an
`H`-link. -/
theorem linked_mono_mem {G H : V → V → Prop} :
    ∀ p : List V, (∀ a ∈ p, ∀ b ∈ p, G a b → H a b) → Linked G p → Linked H p
  | [], _, _ => by simp [Linked]
  | [_], _, _ => by simp [Linked]
  | a :: b :: rest, hGH, hL => by
    simp only [Linked] at hL ⊢
    exact ⟨hGH a (by simp) b (by simp) hL.1,
      linked_mono_mem (b :: rest) (fun x hx y hy => hGH x (List.mem_cons_of_mem _ hx) y
        (List.mem_cons_of_mem _ hy)) hL.2⟩

/-- A walk in a subgraph is a walk in the graph. -/
theorem linked_mono {G H : V → V → Prop} (hGH : ∀ a b, G a b → H a b) (p : List V)
    (h : Linked G p) : Linked H p :=
  linked_mono_mem p (fun a _ b _ => hGH a b) h

/-- A walk in a subgraph is a walk in the graph (with its ends). -/
theorem IsPath.mono {G H : V → V → Prop} (hGH : ∀ a b, G a b → H a b) {a b : V}
    {p : List V} (h : IsPath G a b p) : IsPath H a b p :=
  ⟨h.head, h.last, linked_mono hGH p h.linked⟩

/-- Every node of a walk after its first is entered by a link. -/
theorem linked_mem_tail {H : V → V → Prop} :
    ∀ {a : V} {l : List V}, Linked H (a :: l) → ∀ b ∈ l, ∃ c, H c b
  | _, [], _, b, hb => by simp at hb
  | a, c :: l, h, b, hb => by
    simp only [Linked] at h
    rcases List.mem_cons.mp hb with rfl | hb
    · exact ⟨a, h.1⟩
    · exact linked_mem_tail h.2 b hb

/-- The last node of a list is in it. -/
theorem mem_of_getLast? {α : Type*} {l : List α} {a : α} (h : l.getLast? = some a) :
    a ∈ l := by
  obtain ⟨ys, rfl⟩ := List.getLast?_eq_some_iff.mp h
  simp

/-- Appending in front does not change the last node of a non-empty list. -/
theorem getLast?_append_cons {α : Type*} (r l : List α) (a : α) :
    (r ++ a :: l).getLast? = (a :: l).getLast? := by
  rw [List.getLast?_append]
  cases h : (a :: l).getLast? with
  | none => simp at h
  | some x => rfl

/-- A list is empty or ends in some last element. -/
theorem eq_nil_or_snoc {α : Type*} (l : List α) : l = [] ∨ ∃ l' b, l = l' ++ [b] := by
  rcases List.eq_nil_or_concat l with h | ⟨l', b, h⟩
  · exact Or.inl h
  · exact Or.inr ⟨l', b, by simpa using h⟩

/-- The last element of `L ++ [C]` is `C`. -/
theorem mem_getLast?_snoc {α : Type*} {L : List α} {C P : α}
    (h : P ∈ (L ++ [C]).getLast?) : P = C := by
  have h1 : (L ++ [C]).getLast? = some C := by rw [getLast?_append_cons]; rfl
  exact (Option.some.inj (h1.symm.trans (Option.mem_def.mp h))).symm

/-- In a `KShortest` list every route, not only the last, costs at most any simple path
outside the list. -/
theorem KShortest.le_outside {G : V → V → Prop} {w : V → V → ℝ} {s t : V}
    {L : List (List V)} (hL : KShortest G w s t L) :
    ∀ P ∈ L, ∀ Q, Simple G s t Q → Q ∉ L → cost w P ≤ cost w Q := by
  intro P hP Q hQ hQL
  rcases eq_nil_or_snoc L with rfl | ⟨L', z, rfl⟩
  · simp at hP
  · have hz : cost w z ≤ cost w Q := hL.last_le Q hQ hQL z (by simp)
    rcases List.mem_append.mp hP with hP' | hP'
    · exact le_trans ((List.pairwise_append.mp hL.sorted).2.2 P hP' z (by simp)) hz
    · rw [List.mem_singleton] at hP'
      rw [hP']
      exact hz

/-- Every prefix of a `KShortest` list is a `KShortest` list: the first `j` routes Studio
shows are the `j` shortest, for each `j`. -/
theorem KShortest.take {G : V → V → Prop} {w : V → V → ℝ} {s t : V} {L : List (List V)}
    (hL : KShortest G w s t L) (j : ℕ) : KShortest G w s t (L.take j) := by
  refine ⟨hL.nodup.sublist (List.take_sublist j L),
    fun P hP => hL.simple P (List.mem_of_mem_take hP),
    hL.sorted.sublist (List.take_sublist j L), ?_⟩
  intro Q hQ hQL P hP
  have hPt : P ∈ L.take j := mem_of_getLast? (Option.mem_def.mp hP)
  have hPL : P ∈ L := List.mem_of_mem_take hPt
  by_cases hQin : Q ∈ L
  · have hsplit := List.take_append_drop j L
    have hQd : Q ∈ L.drop j := by
      rw [← hsplit] at hQin
      rcases List.mem_append.mp hQin with h | h
      · exact absurd h hQL
      · exact h
    have hs := hL.sorted
    rw [← hsplit] at hs
    exact (List.pairwise_append.mp hs).2.2 P hPt Q hQd
  · exact hL.le_outside P hPL Q hQ hQin

/-- Extending a `KShortest` list by a simple path not in it that costs at most every simple
path outside it gives a `KShortest` list. This is the step "accept the cheapest
candidate", once the candidate is known to be such a path. -/
theorem KShortest.snoc {G : V → V → Prop} {w : V → V → ℝ} {s t : V} {L : List (List V)}
    {C : List V} (hL : KShortest G w s t L) (hC : Simple G s t C) (hCL : C ∉ L)
    (hmin : ∀ Q, Simple G s t Q → Q ∉ L → cost w C ≤ cost w Q) :
    KShortest G w s t (L ++ [C]) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [List.nodup_append]
    refine ⟨hL.nodup, List.nodup_singleton C, ?_⟩
    intro a ha b hb hab
    rw [List.mem_singleton] at hb
    rw [hab, hb] at ha
    exact hCL ha
  · intro P hP
    rcases List.mem_append.mp hP with h | h
    · exact hL.simple P h
    · rw [List.mem_singleton] at h
      rw [h]
      exact hC
  · rw [List.pairwise_append]
    refine ⟨hL.sorted, List.pairwise_singleton _ _, ?_⟩
    intro a ha b hb
    rw [List.mem_singleton] at hb
    rw [hb]
    exact hL.le_outside a ha C hC hCL
  · intro Q hQ hQL P hP
    rw [mem_getLast?_snoc hP]
    exact hmin Q hQ (fun h => hQL (List.mem_append_left _ h))

/-! ## (2) Yen's key lemma -/

/-- A link the spur search may use, for spur node `v` with root `pre ++ [v]`, given the
accepted routes `A`: a link of `G` that touches no node of `pre` (`yen` masks every edge at
a root node before `v`) and is not the edge between `v` and the next node of an accepted
route beginning with the root (`yen` masks those edges, so both directions). -/
structure SpurLink (G : V → V → Prop) (A : List (List V)) (pre : List V) (v a b : V) :
    Prop where
  link : G a b
  left : a ∉ pre
  right : b ∉ pre
  not_next : ¬ (a = v ∧ ∃ P ∈ A, ∃ rest, P = pre ++ v :: b :: rest)
  not_next_rev : ¬ (b = v ∧ ∃ P ∈ A, ∃ rest, P = pre ++ v :: a :: rest)

/-- Masking more accepted routes leaves fewer usable links. -/
theorem SpurLink.mono {G : V → V → Prop} {A A' : List (List V)} {pre : List V} {v a b : V}
    (h : SpurLink G A pre v a b) (hsub : ∀ P ∈ A', P ∈ A) : SpurLink G A' pre v a b :=
  ⟨h.link, h.left, h.right, fun ⟨hav, P, hP, rest, he⟩ => h.not_next ⟨hav, P, hsub P hP, rest, he⟩,
    fun ⟨hbv, P, hP, rest, he⟩ => h.not_next_rev ⟨hbv, P, hsub P hP, rest, he⟩⟩

/-- What one A* call returns, as a contract: a simple `a`-`b` path in `H` that costs no
more than any `a`-`b` walk in `H`. `shortestPath_of_astar` proves `route.js`'s A* meets it. -/
structure ShortestPath (H : V → V → Prop) (w : V → V → ℝ) (a b : V) (p : List V) : Prop where
  path : IsPath H a b p
  nodup : p.Nodup
  le : ∀ T, IsPath H a b T → cost w p ≤ cost w T

/-- The spur searches from the accepted route `P` have been run, `A` being the accepted list
at that moment (ending in `P`): for each spur node `v` of `P` (every node but the exit),
if the masked graph has a walk from `v` to the exit then a shortest spur `S` was found and
the candidate `pre ++ S` is in `pool` (the accepted routes and the candidates). -/
def Processed (G : V → V → Prop) (w : V → V → ℝ) (t : V) (A : List (List V)) (P : List V)
    (pool : Set (List V)) : Prop :=
  ∀ pre v x rest, P = pre ++ v :: x :: rest → (∃ T, IsPath (SpurLink G A pre v) v t T) →
    ∃ S, ShortestPath (SpurLink G A pre v) w v t S ∧ pre ++ S ∈ pool

/-- A larger pool keeps a round's record. -/
theorem Processed.mono {G : V → V → Prop} {w : V → V → ℝ} {t : V} {A : List (List V)}
    {P : List V} {pool pool' : Set (List V)} (hp : Processed G w t A P pool)
    (h : pool ⊆ pool') : Processed G w t A P pool' := by
  intro pre v x rest hP hT
  obtain ⟨S, hS, hm⟩ := hp pre v x rest hP hT
  exact ⟨S, hS, h hm⟩

/-- The candidate's cost is the root's cost plus the spur's: the code's
`rootCost + spur.cost` is the cost of the node list `pre ++ S` it stores. -/
theorem candidate_cost (w : V → V → ℝ) (pre S : List V) (v : V) (hS : S.head? = some v) :
    cost w (pre ++ S) = cost w (pre ++ [v]) + cost w S := by
  obtain ⟨S', rfl⟩ := List.head?_eq_some_iff.mp hS
  exact cost_append w v S' pre

/-- Every candidate is a simple `s`-`t` path: the root part of an accepted simple path,
then a simple spur that avoids the root. So `yen` never offers a route with a loop. -/
theorem spur_candidate_simple {G : V → V → Prop} {w : V → V → ℝ} {s t : V}
    {A : List (List V)} {pre : List V} {v x : V} {rest S : List V}
    (hprev : Simple G s t (pre ++ v :: x :: rest))
    (hS : ShortestPath (SpurLink G A pre v) w v t S) : Simple G s t (pre ++ S) := by
  obtain ⟨⟨hSh, hSl, hSL⟩, hSnd, -⟩ := hS
  obtain ⟨S', rfl⟩ := List.head?_eq_some_iff.mp hSh
  obtain ⟨⟨hph, -, hpL⟩, hpnd⟩ := hprev
  rw [List.nodup_append] at hpnd
  refine ⟨⟨?_, ?_, ?_⟩, ?_⟩
  · simpa [List.head?_append] using hph
  · rw [getLast?_append_cons]
    exact hSl
  · rw [linked_append]
    exact ⟨((linked_append G v (x :: rest) pre).1 hpL).1,
      linked_mono (fun a b h => h.link) _ hSL⟩
  · rw [List.nodup_append]
    refine ⟨hpnd.1, hSnd, ?_⟩
    intro a ha b hb hab
    rw [← hab] at hb
    rcases List.mem_cons.mp hb with h | hb'
    · rw [h] at ha
      exact hpnd.2.2 v ha v (by simp) rfl
    · obtain ⟨c, hc⟩ := linked_mem_tail hSL a hb'
      exact hc.right ha

/-- Length of the common prefix of two lists. -/
def cpl [DecidableEq V] : List V → List V → ℕ
  | a :: as, b :: bs => if a = b then cpl as bs + 1 else 0
  | _, _ => 0

/-- Two lists that begin with `r` share a prefix at least as long as `r`. -/
theorem le_cpl [DecidableEq V] (l₁ l₂ : List V) :
    ∀ r : List V, r.length ≤ cpl (r ++ l₁) (r ++ l₂)
  | [] => Nat.zero_le _
  | a :: r => by
    have ih := le_cpl l₁ l₂ r
    simp [cpl]
    omega

/-- Two lists split as a common prefix of length `cpl`, after which they differ (or one
ends). -/
theorem cpl_split [DecidableEq V] : ∀ l₁ l₂ : List V, ∃ r l₁' l₂', l₁ = r ++ l₁' ∧
    l₂ = r ++ l₂' ∧ r.length = cpl l₁ l₂ ∧ ∀ x y r₁ r₂, l₁' = x :: r₁ → l₂' = y :: r₂ → x ≠ y
  | [], l₂ => ⟨[], [], l₂, rfl, rfl, rfl, by simp⟩
  | a :: as, [] => ⟨[], a :: as, [], rfl, rfl, rfl, by simp⟩
  | a :: as, b :: bs => by
    by_cases hab : a = b
    · subst hab
      obtain ⟨r, l₁', l₂', h1, h2, h3, h4⟩ := cpl_split as bs
      exact ⟨a :: r, l₁', l₂', by simp [h1], by simp [h2], by simp [cpl, h3], h4⟩
    · refine ⟨[], a :: as, b :: bs, rfl, rfl, by simp [cpl, hab], ?_⟩
      intro x y r₁ r₂ h1 h2
      simp only [List.cons.injEq] at h1 h2
      rw [← h1.1, ← h2.1]
      exact hab

/-- If some element satisfies `p`, the list splits at the last such element. -/
theorem exists_last_split {α : Type*} (p : α → Prop) :
    ∀ l : List α, (∃ x ∈ l, p x) → ∃ l₁ x l₂, l = l₁ ++ x :: l₂ ∧ p x ∧ ∀ y ∈ l₂, ¬ p y
  | [], h => by simp at h
  | a :: l, h => by
    by_cases h' : ∃ x ∈ l, p x
    · obtain ⟨l₁, x, l₂, rfl, hx, hl₂⟩ := exists_last_split p l h'
      exact ⟨a :: l₁, x, l₂, rfl, hx, hl₂⟩
    · simp only [not_exists, not_and] at h'
      obtain ⟨x, hx, hpx⟩ := h
      rcases List.mem_cons.mp hx with rfl | hx
      · exact ⟨[], x, l, rfl, hpx, h'⟩
      · exact absurd hpx (h' x hx)

/-- Yen's deviation lemma. Let the accepted list `A` hold simple `s`-`t` paths and let `Q` be
a simple `s`-`t` path not in it. Then `A = A₁ ++ P :: A₂` where `Q` deviates from `P` at a
spur node `v`: `P = pre ++ v :: x :: _` and `Q = pre ++ v :: y :: _` with `x ≠ y`; `Q`'s spur
`v :: y :: _` is a walk using only links the spur search may use with the *whole* accepted
list masked (it avoids `pre` and the first link out of `v` of every accepted route with
root `pre ++ [v]`); and no route accepted after `P` has that root. (Take `P` sharing the
longest prefix with `Q`, the latest among ties.) For Studio: every route Yen has not yet
returned branches off some returned route at a point where Yen ran a spur search. -/
theorem deviation (G : V → V → Prop) (s t : V) (A : List (List V))
    (hA : ∀ P ∈ A, Simple G s t P) (hne : A ≠ []) (Q : List V) (hQ : Simple G s t Q)
    (hQA : Q ∉ A) :
    ∃ A₁ P A₂ pre v x y rP rQ, A = A₁ ++ P :: A₂ ∧ P = pre ++ v :: x :: rP ∧
      Q = pre ++ v :: y :: rQ ∧ x ≠ y ∧ IsPath (SpurLink G A pre v) v t (v :: y :: rQ) ∧
      ∀ P' ∈ A₂, ∀ r, P' ≠ pre ++ v :: r := by
  classical
  obtain ⟨P₀, hP₀, hmax⟩ := Finset.exists_max_image A.toFinset (fun P => cpl P Q)
    ((List.toFinset_nonempty_iff A).mpr hne)
  have hmax' : ∀ P' ∈ A, cpl P' Q ≤ cpl P₀ Q := fun P' h => hmax P' (List.mem_toFinset.mpr h)
  obtain ⟨A₁, P, A₂, hsplit, hPm, hA₂⟩ :=
    exists_last_split (fun P => cpl P Q = cpl P₀ Q) A ⟨P₀, List.mem_toFinset.mp hP₀, rfl⟩
  have hPm' : cpl P Q = cpl P₀ Q := hPm
  have hPA : P ∈ A := by rw [hsplit]; simp
  have hPQ : P ≠ Q := fun h => hQA (h ▸ hPA)
  obtain ⟨⟨hPh, hPl, -⟩, hPnd⟩ := hA P hPA
  obtain ⟨⟨hQh, hQl, hQL⟩, hQnd⟩ := hQ
  obtain ⟨r, l₁, l₂, h1, h2, hr, hdiff⟩ := cpl_split P Q
  -- `P` continues after the common prefix
  obtain ⟨x, rP, rfl⟩ : ∃ x rP, l₁ = x :: rP := by
    rcases l₁ with _ | ⟨x, rP⟩
    · exfalso
      rcases l₂ with _ | ⟨y, rQ⟩
      · exact hPQ (by rw [h1, h2])
      · have htr : t ∈ r := by
          rw [h1, List.append_nil] at hPl
          exact mem_of_getLast? hPl
        have htl : t ∈ y :: rQ := by
          rw [h2, getLast?_append_cons] at hQl
          exact mem_of_getLast? hQl
        rw [h2, List.nodup_append] at hQnd
        exact hQnd.2.2 t htr t htl rfl
    · exact ⟨x, rP, rfl⟩
  -- and so does `Q`
  obtain ⟨y, rQ, rfl⟩ : ∃ y rQ, l₂ = y :: rQ := by
    rcases l₂ with _ | ⟨y, rQ⟩
    · exfalso
      have htr : t ∈ r := by
        rw [h2, List.append_nil] at hQl
        exact mem_of_getLast? hQl
      have htl : t ∈ x :: rP := by
        rw [h1, getLast?_append_cons] at hPl
        exact mem_of_getLast? hPl
      rw [h1, List.nodup_append] at hPnd
      exact hPnd.2.2 t htr t htl rfl
    · exact ⟨y, rQ, rfl⟩
  have hxy : x ≠ y := hdiff x y rP rQ rfl rfl
  -- the common prefix is not empty: both start at `s`
  obtain ⟨pre, v, rfl⟩ : ∃ pre v, r = pre ++ [v] := by
    rcases eq_nil_or_snoc r with hr0 | h
    · exfalso
      rw [hr0] at h1 h2
      rw [h1] at hPh
      rw [h2] at hQh
      simp only [List.nil_append, List.head?_cons, Option.some.injEq] at hPh hQh
      exact hxy (hPh.trans hQh.symm)
    · exact h
  simp only [List.append_assoc, List.singleton_append] at h1 h2
  refine ⟨A₁, P, A₂, pre, v, x, y, rP, rQ, hsplit, h1, h2, hxy, ?_, ?_⟩
  · -- `Q`'s spur is a walk the spur search may use
    have hQnd' := hQnd
    rw [h2, List.nodup_append] at hQnd'
    obtain ⟨-, hsp, hdisj⟩ := hQnd'
    have hv : v ∉ y :: rQ := (List.nodup_cons.mp hsp).1
    have hQL' := hQL
    rw [h2, linked_append] at hQL'
    have hspL := hQL'.2
    simp only [Linked] at hspL
    refine ⟨rfl, ?_, ?_⟩
    · rw [h2, getLast?_append_cons] at hQl
      exact hQl
    · simp only [Linked]
      refine ⟨⟨hspL.1, fun h => hdisj v h v (by simp) rfl, fun h => hdisj y h y (by simp) rfl,
        ?_, ?_⟩, ?_⟩
      · rintro ⟨-, P', hP'A, rest, hP'⟩
        have hle := le_cpl rest rQ (pre ++ [v, y])
        simp only [List.append_assoc, List.cons_append, List.nil_append] at hle
        rw [← hP', ← h2] at hle
        have := hmax' P' hP'A
        simp only [List.length_append, List.length_cons, List.length_nil] at hle hr
        omega
      · rintro ⟨hyv, -⟩
        exact hv (hyv ▸ List.mem_cons_self ..)
      · exact linked_mono_mem _ (fun a ha b hb hab =>
          ⟨hab, fun h => hdisj a h a (List.mem_cons_of_mem _ ha) rfl,
            fun h => hdisj b h b (List.mem_cons_of_mem _ hb) rfl,
            fun ⟨hav, _⟩ => hv (hav ▸ ha), fun ⟨hbv, _⟩ => hv (hbv ▸ hb)⟩) hspL.2
  · -- no route accepted after `P` has the root `pre ++ [v]`
    intro P' hP' r' hP'eq
    have hP'A : P' ∈ A := by rw [hsplit]; simp [hP']
    have hle := le_cpl r' (y :: rQ) (pre ++ [v])
    simp only [List.append_assoc, List.singleton_append] at hle
    rw [← hP'eq, ← h2] at hle
    have := hmax' P' hP'A
    exact hA₂ P' hP' (show cpl P' Q = cpl P₀ Q by omega)

/-- Yen's key lemma (completeness of candidate generation). Suppose the accepted list `A`
holds simple `s`-`t` paths and, for every accepted route `P`, the spur searches from `P`
were run with the accepted list of that moment (`A₁ ++ [P]`), their candidates landing in
the accepted list or the candidate set `B`. Then for every simple path `Q` not yet
accepted, some candidate in `B` costs at most `cost Q`. For Studio: the cheapest
candidate is a cheapest route not yet returned, so accepting it never skips a route. -/
theorem yen_key (G : V → V → Prop) (w : V → V → ℝ) (s t : V) (A : List (List V))
    (B : Set (List V)) (hA : ∀ P ∈ A, Simple G s t P) (hne : A ≠ [])
    (hproc : ∀ A₁ P A₂, A = A₁ ++ P :: A₂ →
      Processed G w t (A₁ ++ [P]) P {p | p ∈ A ∨ p ∈ B})
    (Q : List V) (hQ : Simple G s t Q) (hQA : Q ∉ A) : ∃ C ∈ B, cost w C ≤ cost w Q := by
  obtain ⟨A₁, P, A₂, pre, v, x, y, rP, rQ, hsplit, hP, hQeq, -, hspur, hlatest⟩ :=
    deviation G s t A hA hne Q hQ hQA
  have hsub : ∀ P' ∈ A₁ ++ [P], P' ∈ A := by
    intro P' h
    rw [hsplit]
    simp only [List.mem_append, List.mem_cons] at h ⊢
    tauto
  have hT : IsPath (SpurLink G (A₁ ++ [P]) pre v) v t (v :: y :: rQ) :=
    hspur.mono (fun a b h => h.mono hsub)
  obtain ⟨S, hS, hmem⟩ := hproc A₁ P A₂ hsplit pre v x rP hP ⟨_, hT⟩
  obtain ⟨S', rfl⟩ := List.head?_eq_some_iff.mp hS.path.head
  have hcost : cost w (pre ++ v :: S') ≤ cost w Q := by
    rw [hQeq, cost_append w v S' pre, cost_append w v (y :: rQ) pre]
    have := hS.le _ hT
    linarith
  simp only [Set.mem_ofPred_eq] at hmem
  rcases hmem with hA' | hB
  · -- the candidate cannot be an accepted route
    exfalso
    have hPA : P ∈ A := hsub P (by simp)
    have hPnd := (hA P hPA).nodup
    have hPl := (hA P hPA).path.last
    rw [hP] at hPnd hPl
    rw [getLast?_append_cons, List.getLast?_cons_cons] at hPl
    have hvt : v ≠ t := by
      intro hvt
      rw [List.nodup_append] at hPnd
      exact (List.nodup_cons.mp hPnd.2.1).1 (hvt ▸ mem_of_getLast? hPl)
    obtain ⟨z, S'', rfl⟩ : ∃ z S'', S' = z :: S'' := by
      rcases S' with _ | ⟨z, S''⟩
      · exfalso
        have := hS.path.last
        simp only [List.getLast?_singleton, Option.some.injEq] at this
        exact hvt this
      · exact ⟨z, S'', rfl⟩
    have hlink := hS.path.linked
    simp only [Linked] at hlink
    rw [hsplit] at hA'
    simp only [List.mem_append, List.mem_cons] at hA'
    rcases hA' with h | h | h
    · exact hlink.1.not_next ⟨rfl, _, List.mem_append_left _ h, S'', rfl⟩
    · exact hlink.1.not_next ⟨rfl, P, by simp, S'', h.symm⟩
    · exact hlatest _ h (z :: S'') rfl
  · exact ⟨_, hB, hcost⟩

/-! ## (3) Yen's loop -/

/-- Yen's state between rounds: the accepted routes, in the order accepted, and the
candidates not yet accepted. -/
structure YState (V : Type*) where
  A : List (List V)
  B : Set (List V)

/-- Start of `yen`: the A* route alone is accepted, no candidates. -/
def YInit (G : V → V → Prop) (w : V → V → ℝ) (s t : V) (σ : YState V) : Prop :=
  ∃ P, ShortestPath G w s t P ∧ σ.A = [P] ∧ σ.B = ∅

/-- One pass of the `for (i …)` loop of `yen` over the last accepted route `prev`, taking
the candidates from `B` to `B'`: candidates are only added; each new one is not accepted
(`seen`) and is `pre ++ S` for a split `prev = pre ++ v :: x :: rest` and a shortest spur
`S` in the masked graph; and for every such split whose masked graph has a walk to the
exit, a shortest spur's candidate is accepted or a candidate (`seen` skips it otherwise). -/
def SpurRound (G : V → V → Prop) (w : V → V → ℝ) (t : V) (A : List (List V))
    (B B' : Set (List V)) : Prop :=
  ∃ prev, A.getLast? = some prev ∧ B ⊆ B' ∧
    (∀ C ∈ B', C ∉ B → C ∉ A ∧ ∃ pre v x rest S, prev = pre ++ v :: x :: rest ∧
      ShortestPath (SpurLink G A pre v) w v t S ∧ C = pre ++ S) ∧
    Processed G w t A prev {p | p ∈ A ∨ p ∈ B'}

/-- One round of `yen`'s `while` loop: the spur searches, then `accepted.push(
candidates.shift())` after sorting, i.e. a candidate of least cost moves to the accepted
list. -/
def YStep (G : V → V → Prop) (w : V → V → ℝ) (t : V) (σ σ' : YState V) : Prop :=
  ∃ B, SpurRound G w t σ.A σ.B B ∧ ∃ C ∈ B, (∀ D ∈ B, cost w C ≤ cost w D) ∧
    σ'.A = σ.A ++ [C] ∧ σ'.B = B \ {C}

/-- Yen's invariant: the accepted list is a `KShortest` list, every candidate is a simple
path not yet accepted, and the spur searches of every accepted route but the last have
been run. -/
structure YInv (G : V → V → Prop) (w : V → V → ℝ) (s t : V) (σ : YState V) : Prop where
  ne : σ.A ≠ []
  ks : KShortest G w s t σ.A
  cand : ∀ C ∈ σ.B, Simple G s t C ∧ C ∉ σ.A
  hist : ∀ A₁ P A₂, σ.A = A₁ ++ P :: A₂ → A₂ ≠ [] →
    Processed G w t (A₁ ++ [P]) P {p | p ∈ σ.A ∨ p ∈ σ.B}

/-- `yen` starts in a state satisfying the invariant: the A* route alone is the shortest
simple path. -/
theorem yinv_init {G : V → V → Prop} {w : V → V → ℝ} {s t : V} {σ : YState V}
    (h : YInit G w s t σ) : YInv G w s t σ := by
  obtain ⟨P, hP, hA, hB⟩ := h
  refine ⟨by simp [hA], ?_, by simp [hB], ?_⟩
  · rw [hA]
    refine ⟨List.nodup_singleton P, ?_, List.pairwise_singleton _ P, ?_⟩
    · intro P' h
      rw [List.mem_singleton] at h
      rw [h]
      exact ⟨hP.path, hP.nodup⟩
    · intro Q hQ _ P' hP'
      have h1 : P' = P := mem_getLast?_snoc (L := []) hP'
      rw [h1]
      exact hP.le Q hQ.path
  · intro A₁ P' A₂ hsplit hA₂
    exfalso
    have hl := congrArg List.length (hA.symm.trans hsplit)
    simp only [List.length_append, List.length_cons, List.length_nil] at hl
    exact hA₂ (List.eq_nil_of_length_eq_zero (by omega))

/-- What a spur round guarantees, given the invariant: every candidate is a simple path
not yet accepted, every accepted route's spur searches have been run, and (by `yen_key`)
every simple path not yet accepted costs at least some candidate. -/
theorem round_facts {G : V → V → Prop} {w : V → V → ℝ} {s t : V} {σ : YState V}
    {B : Set (List V)} (hI : YInv G w s t σ) (hR : SpurRound G w t σ.A σ.B B) :
    (∀ D ∈ B, Simple G s t D ∧ D ∉ σ.A) ∧
    (∀ A₁ P A₂, σ.A = A₁ ++ P :: A₂ →
      Processed G w t (A₁ ++ [P]) P {p | p ∈ σ.A ∨ p ∈ B}) ∧
    (∀ Q, Simple G s t Q → Q ∉ σ.A → ∃ D ∈ B, cost w D ≤ cost w Q) := by
  obtain ⟨prev, hlast, hBB, hnew, hproc⟩ := hR
  have hprevA : prev ∈ σ.A := mem_of_getLast? hlast
  have hvalid : ∀ D ∈ B, Simple G s t D ∧ D ∉ σ.A := by
    intro D hD
    by_cases hDB : D ∈ σ.B
    · exact hI.cand D hDB
    · obtain ⟨hDA, pre, v, x, rest, S, hprev, hS, hDeq⟩ := hnew D hD hDB
      refine ⟨?_, hDA⟩
      have hps := hI.ks.simple prev hprevA
      rw [hprev] at hps
      rw [hDeq]
      exact spur_candidate_simple hps hS
  have hhist : ∀ A₁ P A₂, σ.A = A₁ ++ P :: A₂ →
      Processed G w t (A₁ ++ [P]) P {p | p ∈ σ.A ∨ p ∈ B} := by
    intro A₁ P A₂ hsplit
    by_cases hA₂ : A₂ = []
    · rw [hA₂] at hsplit
      have hP : P = prev := by
        have : some P = some prev := by
          rw [← hlast, hsplit, getLast?_append_cons]
          rfl
        exact Option.some.inj this
      rw [hP] at hsplit
      rw [hP, ← hsplit]
      exact hproc
    · exact (hI.hist A₁ P A₂ hsplit hA₂).mono (fun p hp => by
        rcases hp with h | h
        · exact Or.inl h
        · exact Or.inr (hBB h))
  exact ⟨hvalid, hhist, yen_key G w s t σ.A B (fun P hP => hI.ks.simple P hP) hI.ne hhist⟩

/-- A round of `yen` preserves the invariant. The accepted candidate is a simple path, new,
and by `yen_key` costs at most every simple path not yet accepted. -/
theorem yinv_step {G : V → V → Prop} {w : V → V → ℝ} {s t : V} {σ σ' : YState V}
    (hI : YInv G w s t σ) (hst : YStep G w t σ σ') : YInv G w s t σ' := by
  obtain ⟨B, hR, C, hCB, hCmin, hA', hB'⟩ := hst
  obtain ⟨hvalid, hhist, hcover⟩ := round_facts hI hR
  have hCle : ∀ Q, Simple G s t Q → Q ∉ σ.A → cost w C ≤ cost w Q := by
    intro Q hQ hQA
    obtain ⟨D, hD, hDQ⟩ := hcover Q hQ hQA
    exact (hCmin D hD).trans hDQ
  have hpool : {p | p ∈ σ.A ∨ p ∈ B} ⊆ {p | p ∈ σ'.A ∨ p ∈ σ'.B} := by
    intro p hp
    simp only [Set.mem_ofPred_eq] at hp ⊢
    rw [hA', hB']
    by_cases hpC : p = C
    · left
      simp [hpC]
    · rcases hp with h | h
      · left
        simp [h]
      · right
        exact Set.mem_sdiff_of_mem h (by simpa using hpC)
  refine ⟨by rw [hA']; simp, ?_, ?_, ?_⟩
  · rw [hA']
    exact hI.ks.snoc (hvalid C hCB).1 (hvalid C hCB).2 hCle
  · intro D hD
    rw [hB'] at hD
    obtain ⟨hDB, hDC⟩ := hD
    rw [hA']
    refine ⟨(hvalid D hDB).1, ?_⟩
    simp only [List.mem_append, List.mem_singleton, not_or]
    exact ⟨(hvalid D hDB).2, by simpa using hDC⟩
  · intro A₁ P A₂ hsplit hA₂
    obtain ⟨A₂', c, rfl⟩ := (eq_nil_or_snoc A₂).resolve_left hA₂
    have h2 : σ.A ++ [C] = (A₁ ++ P :: A₂') ++ [c] := by
      rw [← hA', hsplit]
      simp
    have h3 := (List.append_inj' h2 rfl).1
    exact (hhist A₁ P A₂' h3).mono hpool

/-- L12 (Yen is correct). Every state `yen` can reach from its start satisfies the
invariant. -/
theorem yen_correct {G : V → V → Prop} {w : V → V → ℝ} {s t : V} {σ₀ σ : YState V}
    (h0 : YInit G w s t σ₀) (hrun : Relation.ReflTransGen (YStep G w t) σ₀ σ) :
    YInv G w s t σ := by
  induction hrun with
  | refl => exact yinv_init h0
  | tail _ hst ih => exact yinv_step ih hst

/-- L12 for Studio. Whatever `yen` returns - after `k` routes, when the candidates run out,
or when the time budget stops it between spur searches - is a list of the shortest simple
routes in order: distinct, loopless, costs non-decreasing, and no loopless route left out
is cheaper than the last one shown. The same holds for its first `j` routes, every `j`. -/
theorem yen_kshortest {G : V → V → Prop} {w : V → V → ℝ} {s t : V} {σ₀ σ : YState V}
    (h0 : YInit G w s t σ₀) (hrun : Relation.ReflTransGen (YStep G w t) σ₀ σ) :
    KShortest G w s t σ.A ∧ ∀ j, KShortest G w s t (σ.A.take j) :=
  ⟨(yen_correct h0 hrun).ks, fun j => (yen_correct h0 hrun).ks.take j⟩

/-- The cheapest candidate after a round is a cheapest simple path not yet accepted: it is
new, and costs at most every simple path not yet accepted. -/
theorem yen_accept_next {G : V → V → Prop} {w : V → V → ℝ} {s t : V} {σ : YState V}
    {B : Set (List V)} (hI : YInv G w s t σ) (hR : SpurRound G w t σ.A σ.B B)
    {C : List V} (hC : C ∈ B) (hmin : ∀ D ∈ B, cost w C ≤ cost w D) :
    Simple G s t C ∧ C ∉ σ.A ∧ ∀ Q, Simple G s t Q → Q ∉ σ.A → cost w C ≤ cost w Q := by
  obtain ⟨hvalid, -, hcover⟩ := round_facts hI hR
  refine ⟨(hvalid C hC).1, (hvalid C hC).2, fun Q hQ hQA => ?_⟩
  obtain ⟨D, hD, hDQ⟩ := hcover Q hQ hQA
  exact (hmin D hD).trans hDQ

/-- When a round leaves no candidates (`if (!candidates.length) break;`), every simple
`s`-`t` path has been accepted: fewer than `k` Yen routes, without the time-budget note,
means the building has no other loopless route. -/
theorem yen_dry_complete {G : V → V → Prop} {w : V → V → ℝ} {s t : V} {σ : YState V}
    (hI : YInv G w s t σ) (hR : SpurRound G w t σ.A σ.B ∅) :
    ∀ Q, Simple G s t Q → Q ∈ σ.A := by
  intro Q hQ
  by_contra hQA
  obtain ⟨D, hD, -⟩ := (round_facts hI hR).2.2 Q hQ hQA
  exact hD

/-- Termination bound: on a finite graph a `KShortest` list, so Yen's accepted list, is no
longer than the number of loopless node lists; each round adds a new route, so `yen`
stops after finitely many rounds even for a large `k`. -/
theorem yen_length_le [Fintype V] {G : V → V → Prop} {w : V → V → ℝ} {s t : V}
    {L : List (List V)} (hL : KShortest G w s t L) :
    L.length ≤ Fintype.card {p : List V // p.Nodup} := by
  have hlen : (L.pmap (fun p hp => (⟨p, hp⟩ : {p : List V // p.Nodup}))
      (fun p hp => (hL.simple p hp).nodup)).length = L.length := List.length_pmap
  rw [← hlen]
  exact List.Nodup.length_le_card (hL.nodup.pmap (fun a _ b _ h => congrArg Subtype.val h))

/-! ### The spur search is A* -/

/-- The links of `H` with their weights, in `AStar.lean`'s form `E u v a`. -/
def links (H : V → V → Prop) (w : V → V → ℝ) : V → V → ℝ → Prop :=
  fun a b c => H a b ∧ c = w a b

/-- A walk can be extended at its start by a link. -/
theorem walkCost_cons {E : V → V → ℝ → Prop} {a b u : V} {x c : ℝ} (hab : E a b x)
    (hw : AStar.WalkCost E b u c) : AStar.WalkCost E a u (x + c) := by
  induction hw with
  | nil => exact (AStar.WalkCost.nil.snoc hab).cast (by ring)
  | snoc _ hE ih => exact (ih.snoc hE).cast (by ring)

/-- A node list that is a walk from `v` to `t` in `H` is a walk of cost `cost w T` in
`AStar.lean`'s sense. -/
theorem walkCost_of_isPath (H : V → V → Prop) (w : V → V → ℝ) (t : V) :
    ∀ (T : List V) (v : V), IsPath H v t T → AStar.WalkCost (links H w) v t (cost w T)
  | [], v, h => by simpa using h.head
  | [a], v, h => by
    have h1 := h.head
    have h2 := h.last
    simp only [List.head?_cons, List.getLast?_singleton, Option.some.injEq] at h1 h2
    rw [← h1, ← h2]
    exact AStar.WalkCost.nil.cast (by simp [cost])
  | a :: b :: rest, v, h => by
    have h1 := h.head
    simp only [List.head?_cons, Option.some.injEq] at h1
    rw [← h1]
    have hL := h.linked
    simp only [Linked] at hL
    have ih := walkCost_of_isPath H w t (b :: rest) b
      ⟨rfl, by simpa using h.last, hL.2⟩
    exact (walkCost_cons (show links H w a b (w a b) from ⟨hL.1, rfl⟩) ih).cast
      (by simp [cost])

/-- The spur search meets its contract. If A* (`AStar.Step` with a consistent heuristic
`h`) on the masked graph `H` reaches the point where it pops the exit `t` - where
`route.js` stops - and the node list it rebuilds is a simple `v`-`t` path in `H` whose
cost is the label `dist[t]`, then that list is a `ShortestPath`. Reuses
`AStar.astar_correct`. -/
theorem shortestPath_of_astar [DecidableEq V] (H : V → V → Prop) (w : V → V → ℝ)
    (h : V → ℝ) (hc : ∀ a b, H a b → h a ≤ w a b + h b) (v t : V)
    (σ₀ σ : AStar.State V) (h0 : AStar.Init v σ₀)
    (hrun : Relation.ReflTransGen (AStar.Step (links H w) h) σ₀ σ)
    (ht : AStar.Selectable h σ t) (S : List V) (hS : IsPath H v t S) (hnd : S.Nodup)
    (hcost : cost w S = σ.d t) : ShortestPath H w v t S := by
  have hc' : ∀ a b c, links H w a b c → h a ≤ c + h b := by
    rintro a b c ⟨hab, rfl⟩
    exact hc a b hab
  refine ⟨hS, hnd, fun T hT => ?_⟩
  have := (AStar.astar_correct (links H w) h hc' v σ₀ σ h0 hrun t ht).2 _
    (walkCost_of_isPath H w t T v hT)
  linarith

/-- The spur search's contract for Studio's heuristic: the straight-line distance to the
exit, with every link at least as heavy as it is long (the masked graph keeps a subset of
the links, so this holds there too). Reuses `Heuristic.consistent`. -/
theorem shortestPath_of_astar_straight_line [DecidableEq V] {P : Type*} [MetricSpace P]
    (pos : V → P) (H : V → V → Prop) (w : V → V → ℝ)
    (hlen : ∀ a b, H a b → dist (pos a) (pos b) ≤ w a b) (v t : V)
    (σ₀ σ : AStar.State V) (h0 : AStar.Init v σ₀)
    (hrun : Relation.ReflTransGen (AStar.Step (links H w) (fun x => dist (pos x) (pos t))) σ₀ σ)
    (ht : AStar.Selectable (fun x => dist (pos x) (pos t)) σ t) (S : List V)
    (hS : IsPath H v t S) (hnd : S.Nodup) (hcost : cost w S = σ.d t) :
    ShortestPath H w v t S :=
  shortestPath_of_astar H w _ (fun a b hab => FireEgress.Heuristic.consistent _ _ _ _
    (hlen a b hab)) v t σ₀ σ h0 hrun ht S hS hnd hcost

/-- "A* found no spur" is right. If A*'s heap runs dry without settling the exit (`astar`
returns `null`), the masked graph has no walk from the spur node to the exit, so `yen`
loses no candidate by skipping that spur node. Reuses `AStar.astar_exhausted`. -/
theorem no_walk_of_astar_dry [DecidableEq V] (H : V → V → Prop) (w : V → V → ℝ)
    (h : V → ℝ) (hc : ∀ a b, H a b → h a ≤ w a b + h b) (v t : V)
    (σ₀ σ : AStar.State V) (h0 : AStar.Init v σ₀)
    (hrun : Relation.ReflTransGen (AStar.Step (links H w) h) σ₀ σ)
    (hdry : σ.R ⊆ σ.S) (ht : t ∉ σ.S) : ¬ ∃ T, IsPath H v t T := by
  have hc' : ∀ a b c, links H w a b c → h a ≤ c + h b := by
    rintro a b c ⟨hab, rfl⟩
    exact hc a b hab
  rintro ⟨T, hT⟩
  exact ht ((AStar.astar_exhausted (links H w) h hc' v σ₀ σ h0 hrun hdry t).1
    ⟨_, walkCost_of_isPath H w t T v hT⟩)

/-! ## (4) The "distinct" alternatives -/

/-- The parameters of `distinct`: `maxStretch` (default 1.5), `maxOverlap` (0.5), the
code's slack `eps` (`1e-9` in `cost > maxStretch * best + 1e-9`), and `k` routes asked. -/
structure DParams where
  maxStretch : ℝ
  maxOverlap : ℝ
  eps : ℝ
  k : ℕ

/-- The `for (attempt …)` loop of `distinct`, with `fuel` attempts left, at attempt `a`,
with the routes kept so far (the first is the A* route). `next a` is the route the
penalised A* returns at attempt `a`, or `none` when no link lies near the last route or
no route is found (the code breaks). `c` is the base cost, `share p q` the share of `p`'s
length within `separation` of `q` (`sharedShare`), `best` the first route's cost. Rules:
stop unless fewer than `k` routes are kept; stop at the first route costing more than
`maxStretch * best + eps`; keep a route if its share near every kept route is at most
`maxOverlap`. The penalties themselves are not modelled: the statements below hold for
every sequence of penalised routes. -/
noncomputable def dloop (P : DParams) (c : List V → ℝ) (share : List V → List V → ℝ)
    (best : ℝ) (next : ℕ → Option (List V)) : ℕ → ℕ → List (List V) → List (List V)
  | 0, _, routes => routes
  | fuel + 1, a, routes =>
    if routes.length < P.k then
      match next a with
      | none => routes
      | some p =>
        if c p ≤ P.maxStretch * best + P.eps then
          if ∀ q ∈ routes, share p q ≤ P.maxOverlap then
            dloop P c share best next fuel (a + 1) (routes ++ [p])
          else dloop P c share best next fuel (a + 1) routes
        else routes
    else routes

/-- What the loop adds: the kept routes stay, in order, followed by new ones; each new
route costs at most `maxStretch * best + eps` and was returned by one of the attempts
`a, …, a + fuel - 1`. -/
theorem dloop_spec (P : DParams) (c : List V → ℝ) (share : List V → List V → ℝ) (best : ℝ)
    (next : ℕ → Option (List V)) : ∀ fuel a (routes : List (List V)), ∃ added,
      dloop P c share best next fuel a routes = routes ++ added ∧
      ∀ p ∈ added, c p ≤ P.maxStretch * best + P.eps ∧
        ∃ i, a ≤ i ∧ i < a + fuel ∧ next i = some p := by
  intro fuel
  induction fuel with
  | zero => intro a routes; exact ⟨[], by simp [dloop], by simp⟩
  | succ fuel ih =>
    intro a routes
    by_cases hk : routes.length < P.k
    · cases hn : next a with
      | none => exact ⟨[], by simp [dloop, hk, hn], by simp⟩
      | some p =>
        by_cases hc : c p ≤ P.maxStretch * best + P.eps
        · by_cases ho : ∀ q ∈ routes, share p q ≤ P.maxOverlap
          · obtain ⟨added, h1, h2⟩ := ih (a + 1) (routes ++ [p])
            refine ⟨p :: added, ?_, ?_⟩
            · simp only [dloop, ite_eq_left hk, hn, ite_eq_left hc, ite_eq_left ho, h1, List.append_assoc,
                List.singleton_append]
            · intro q hq
              rcases List.mem_cons.mp hq with rfl | hq
              · exact ⟨hc, a, le_rfl, by omega, hn⟩
              · obtain ⟨h3, i, hi1, hi2, hi3⟩ := h2 q hq
                exact ⟨h3, i, by omega, by omega, hi3⟩
          · obtain ⟨added, h1, h2⟩ := ih (a + 1) routes
            refine ⟨added, ?_, ?_⟩
            · simp only [dloop, ite_eq_left hk, hn, ite_eq_left hc, ite_eq_right ho, h1]
            · intro q hq
              obtain ⟨h3, i, hi1, hi2, hi3⟩ := h2 q hq
              exact ⟨h3, i, by omega, by omega, hi3⟩
        · exact ⟨[], by simp only [dloop, ite_eq_left hk, hn, ite_eq_right hc, List.append_nil], by simp⟩
    · exact ⟨[], by simp only [dloop, ite_eq_right hk, List.append_nil], by simp⟩

/-- Overlap, by construction, in discovery order: if every kept route has share at most
`maxOverlap` near each route kept before it, the loop keeps it so. -/
theorem dloop_pairwise (P : DParams) (c : List V → ℝ) (share : List V → List V → ℝ)
    (best : ℝ) (next : ℕ → Option (List V)) : ∀ fuel a (routes : List (List V)),
    routes.Pairwise (fun q p => share p q ≤ P.maxOverlap) →
    (dloop P c share best next fuel a routes).Pairwise
      (fun q p => share p q ≤ P.maxOverlap) := by
  intro fuel
  induction fuel with
  | zero => intro a routes h; simpa [dloop] using h
  | succ fuel ih =>
    intro a routes h
    by_cases hk : routes.length < P.k
    · cases hn : next a with
      | none => simpa [dloop, hk, hn] using h
      | some p =>
        by_cases hc : c p ≤ P.maxStretch * best + P.eps
        · by_cases ho : ∀ q ∈ routes, share p q ≤ P.maxOverlap
          · simp only [dloop, ite_eq_left hk, hn, ite_eq_left hc, ite_eq_left ho]
            apply ih
            rw [List.pairwise_append]
            refine ⟨h, List.pairwise_singleton _ _, ?_⟩
            intro q hq p' hp'
            rw [List.mem_singleton] at hp'
            rw [hp']
            exact ho q hq
          · simp only [dloop, ite_eq_left hk, hn, ite_eq_left hc, ite_eq_right ho]
            exact ih _ _ h
        · simpa only [dloop, ite_eq_left hk, hn, ite_eq_right hc] using h
    · simpa only [dloop, ite_eq_right hk] using h

/-- The loop keeps at most `k` routes (or the ones it started with). -/
theorem dloop_length (P : DParams) (c : List V → ℝ) (share : List V → List V → ℝ)
    (best : ℝ) (next : ℕ → Option (List V)) : ∀ fuel a (routes : List (List V)),
    (dloop P c share best next fuel a routes).length ≤ max routes.length P.k := by
  intro fuel
  induction fuel with
  | zero => intro a routes; simp [dloop]
  | succ fuel ih =>
    intro a routes
    by_cases hk : routes.length < P.k
    · cases hn : next a with
      | none => simp [dloop, hk, hn]
      | some p =>
        by_cases hc : c p ≤ P.maxStretch * best + P.eps
        · by_cases ho : ∀ q ∈ routes, share p q ≤ P.maxOverlap
          · simp only [dloop, ite_eq_left hk, hn, ite_eq_left hc, ite_eq_left ho]
            have := ih (a + 1) (routes ++ [p])
            simp only [List.length_append, List.length_singleton] at this
            omega
          · simp only [dloop, ite_eq_left hk, hn, ite_eq_left hc, ite_eq_right ho]
            exact ih _ _
        · simp [dloop, hk, hn, hc]
    · simp [dloop, hk]

/-- The loop only looks at attempts `a, …, a + fuel - 1`: two penalised searches that agree
there give the same result. -/
theorem dloop_congr (P : DParams) (c : List V → ℝ) (share : List V → List V → ℝ)
    (best : ℝ) (next next' : ℕ → Option (List V)) : ∀ fuel a (routes : List (List V)),
    (∀ i, a ≤ i → i < a + fuel → next i = next' i) →
    dloop P c share best next fuel a routes = dloop P c share best next' fuel a routes := by
  intro fuel
  induction fuel with
  | zero => intro a routes _; simp [dloop]
  | succ fuel ih =>
    intro a routes h
    have ha : next a = next' a := h a le_rfl (by omega)
    have key : ∀ r, dloop P c share best next fuel (a + 1) r =
        dloop P c share best next' fuel (a + 1) r :=
      fun r => ih (a + 1) r (fun i h1 h2 => h i (by omega) (by omega))
    simp only [dloop, ha, key]

/-- One attempt that keeps its route. -/
theorem dloop_accept (P : DParams) (c : List V → ℝ) (share : List V → List V → ℝ)
    (best : ℝ) (next : ℕ → Option (List V)) (fuel a : ℕ) (routes : List (List V))
    (p : List V) (hk : routes.length < P.k) (hn : next a = some p)
    (hc : c p ≤ P.maxStretch * best + P.eps) (ho : ∀ q ∈ routes, share p q ≤ P.maxOverlap) :
    dloop P c share best next (fuel + 1) a routes =
      dloop P c share best next fuel (a + 1) (routes ++ [p]) := by
  simp only [dloop, ite_eq_left hk, hn, ite_eq_left hc, ite_eq_left ho]

/-- With `k` routes kept the loop stops (`routes.length < k` fails). -/
theorem dloop_full (P : DParams) (c : List V → ℝ) (share : List V → List V → ℝ)
    (best : ℝ) (next : ℕ → Option (List V)) (fuel a : ℕ) (routes : List (List V))
    (hk : P.k ≤ routes.length) : dloop P c share best next fuel a routes = routes := by
  cases fuel with
  | zero => simp [dloop]
  | succ fuel => simp only [dloop, ite_eq_right (not_lt.mpr hk)]

/-- If every route the penalised search proposes in the remaining attempts is rejected -
over the stretch limit, or sharing more than `maxOverlap` with a kept route - the loop
keeps nothing new, however many routes exist that it would accept. -/
theorem dloop_rejected (P : DParams) (c : List V → ℝ) (share : List V → List V → ℝ)
    (best : ℝ) (next : ℕ → Option (List V)) : ∀ fuel a (routes : List (List V)),
    (∀ i p, a ≤ i → i < a + fuel → next i = some p →
      P.maxStretch * best + P.eps < c p ∨ ∃ q ∈ routes, P.maxOverlap < share p q) →
    dloop P c share best next fuel a routes = routes := by
  intro fuel
  induction fuel with
  | zero => intro a routes _; simp [dloop]
  | succ fuel ih =>
    intro a routes h
    by_cases hk : routes.length < P.k
    · cases hn : next a with
      | none => simp [dloop, hk, hn]
      | some p =>
        by_cases hc : c p ≤ P.maxStretch * best + P.eps
        · have ho : ¬ ∀ q ∈ routes, share p q ≤ P.maxOverlap := by
            rcases h a p le_rfl (by omega) hn with h1 | ⟨q, hq, h2⟩
            · exact absurd hc (not_le.mpr h1)
            · exact fun hall => absurd (hall q hq) (not_le.mpr h2)
          simp only [dloop, ite_eq_left hk, hn, ite_eq_left hc, ite_eq_right ho]
          exact ih (a + 1) routes (fun i p' h1 h2 h3 => h i p' (by omega) (by omega) h3)
        · simp only [dloop, ite_eq_left hk, hn, ite_eq_right hc]
    · simp only [dloop, ite_eq_right hk]

/-- What `distinct` returns: the A* route, then the routes kept by `4k` attempts, sorted by
cost (`routes.splice(1, …, ...routes.slice(1).sort(…))`). -/
noncomputable def distinctRun (P : DParams) (c : List V → ℝ) (share : List V → List V → ℝ)
    (first : List V) (next : ℕ → Option (List V)) : List (List V) :=
  first :: ((dloop P c share (c first) next (4 * P.k) 0 [first]).tail.mergeSort
    (fun a b => decide (c a ≤ c b)))

/-- The routes `distinct` returns, in discovery order: the A* route, then the kept ones. -/
theorem distinct_found (P : DParams) (c : List V → ℝ) (share : List V → List V → ℝ)
    (first : List V) (next : ℕ → Option (List V)) :
    ∃ added, dloop P c share (c first) next (4 * P.k) 0 [first] = first :: added ∧
      distinctRun P c share first next =
        first :: added.mergeSort (fun a b => decide (c a ≤ c b)) ∧
      ∀ p ∈ added, c p ≤ P.maxStretch * c first + P.eps ∧
        ∃ i, i < 4 * P.k ∧ next i = some p := by
  obtain ⟨added, h1, h2⟩ := dloop_spec P c share (c first) next (4 * P.k) 0 [first]
  refine ⟨added, by simpa using h1, by simp [distinctRun, h1], fun p hp => ?_⟩
  obtain ⟨h3, i, -, hi, hi'⟩ := h2 p hp
  exact ⟨h3, i, by omega, hi'⟩

/-- L12.D1 (stretch, by construction). Every route `distinct` returns after the first
costs at most `maxStretch * best + eps` (`eps = 1e-9` in the code), `best` being the first
route's cost, and came from one of the first `4k` penalised searches. -/
theorem distinct_stretch (P : DParams) (c : List V → ℝ) (share : List V → List V → ℝ)
    (first : List V) (next : ℕ → Option (List V)) :
    ∀ p ∈ (distinctRun P c share first next).tail,
      c p ≤ P.maxStretch * c first + P.eps ∧ ∃ i, i < 4 * P.k ∧ next i = some p := by
  obtain ⟨added, -, h2, h3⟩ := distinct_found P c share first next
  intro p hp
  rw [h2] at hp
  simp only [List.tail_cons, List.mem_mergeSort] at hp
  exact h3 p hp

/-- L12.D2 (overlap, by construction, in discovery order). In the order the penalties
found them, each route `distinct` keeps has share at most `maxOverlap` near every route
kept before it (including the first). The returned list is re-sorted by cost, so this
does not bound a route's share near a cheaper route found *after* it. -/
theorem distinct_overlap (P : DParams) (c : List V → ℝ) (share : List V → List V → ℝ)
    (first : List V) (next : ℕ → Option (List V)) :
    (dloop P c share (c first) next (4 * P.k) 0 [first]).Pairwise
      (fun q p => share p q ≤ P.maxOverlap) :=
  dloop_pairwise P c share (c first) next _ _ _ (List.pairwise_singleton _ _)

/-- L12.D3 (termination and size). `distinct` makes at most `4k` penalised searches - its
result depends on the penalised routes of attempts `0, …, 4k - 1` only - and returns at
most `max 1 k` routes. -/
theorem distinct_bounded (P : DParams) (c : List V → ℝ) (share : List V → List V → ℝ)
    (first : List V) (next next' : ℕ → Option (List V))
    (h : ∀ i, i < 4 * P.k → next i = next' i) :
    distinctRun P c share first next = distinctRun P c share first next' ∧
      (distinctRun P c share first next).length ≤ max 1 P.k := by
  refine ⟨?_, ?_⟩
  · simp only [distinctRun]
    rw [dloop_congr P c share (c first) next next' (4 * P.k) 0 [first]
      (fun i _ hi => h i (by omega))]
  · obtain ⟨added, h1, h2, -⟩ := distinct_found P c share first next
    have hl := dloop_length P c share (c first) next (4 * P.k) 0 [first]
    rw [h1] at hl
    rw [h2]
    simp only [List.length_cons, List.length_mergeSort] at hl ⊢
    simpa using hl

/-- The list `distinct` returns is sorted by cost once the first route costs at most every
route the penalised search can return. -/
theorem distinct_sorted_of (P : DParams) (c : List V → ℝ) (share : List V → List V → ℝ)
    (first : List V) (next : ℕ → Option (List V))
    (h : ∀ i p, next i = some p → c first ≤ c p) :
    (distinctRun P c share first next).Pairwise (fun p q => c p ≤ c q) := by
  obtain ⟨added, -, h2, h3⟩ := distinct_found P c share first next
  rw [h2, List.pairwise_cons]
  refine ⟨fun p hp => ?_, ?_⟩
  · rw [List.mem_mergeSort] at hp
    obtain ⟨-, i, -, hi⟩ := h3 p hp
    exact h i p hi
  · have := List.pairwise_mergeSort (le := fun a b => decide (c a ≤ c b))
      (fun a b d hab hbd => by simp only [decide_eq_true_eq] at *; exact le_trans hab hbd)
      (fun a b => by simpa using le_total (c a) (c b)) added
    exact this.imp (fun h => by simpa using h)

/-- L12.D4 (the first route is the shortest). If the first route is A*'s (a
`ShortestPath`: `AStar.astar_correct` via `shortestPath_of_astar`) and every penalised
route is an `s`-`t` walk in the same graph (the penalties change weights, not links), then
the list `distinct` returns starts with a cheapest route and its costs are non-decreasing,
so every reported stretch is at least 1. -/
theorem distinct_sorted {G : V → V → Prop} {w : V → V → ℝ} {s t : V} (P : DParams)
    (share : List V → List V → ℝ) (first : List V) (next : ℕ → Option (List V))
    (hfirst : ShortestPath G w s t first) (hnext : ∀ i p, next i = some p → IsPath G s t p) :
    (distinctRun P (cost w) share first next).head? = some first ∧
      (distinctRun P (cost w) share first next).Pairwise (fun p q => cost w p ≤ cost w q) :=
  ⟨by simp [distinctRun], distinct_sorted_of P (cost w) share first next
    (fun i p hi => hfirst.le p (hnext i p hi))⟩

/-- L12.D5 (`distinct` is not complete, 1: the stretch `break`). The rules do not ensure `k`
routes even when `k` sufficiently different routes exist. Here the shortest route costs 10
and a second route costs 14 (stretch 1.4, within 1.5) and shares nothing, but the
penalised search proposes a route of cost 16 first - as it does when the second route runs
near the first for part of its length, so that its penalised cost (here 14 + 4 = 18)
exceeds the third route's 16 - and `distinct` stops at that over-stretch route
(`if (cost > maxStretch * best + 1e-9) break;`, on the comment's reasoning that "penalties
only rise"), returning one route where two were available. In a probe on random
4-neighbour lattices with holes (k = 2), 22 of the 28 misses found ended at this break. -/
theorem distinct_can_miss : ∃ (P : DParams) (w : ℕ → ℕ → ℝ) (share : List ℕ → List ℕ → ℝ)
    (first good : List ℕ) (next : ℕ → Option (List ℕ)),
    2 ≤ P.k ∧ good ≠ first ∧ cost w good ≤ P.maxStretch * cost w first ∧
      share good first ≤ P.maxOverlap ∧ distinctRun P (cost w) share first next = [first] := by
  refine ⟨⟨3 / 2, 1 / 2, 1 / 10 ^ 9, 2⟩,
    fun a b => if a = 0 ∧ b = 1 then 10 else if a = 3 ∨ b = 3 then 8 else 7,
    fun _ _ => 0, [0, 1], [0, 2, 1], fun _ => some [0, 3, 1], le_rfl, by decide, ?_, ?_, ?_⟩
  · simp only [cost]
    norm_num
  · norm_num
  · simp only [distinctRun]
    rw [show 4 * 2 = 7 + 1 from rfl, dloop]
    norm_num [cost]

/-- L12.D5 (`distinct` is not complete, 2: the attempts run out). Penalties go on the links
near the *last* route found, kept or not, so the search can alternate between two
near-duplicates of the first route until its `4k` attempts are spent (`dloop_rejected`).
Observed: on an open 5 x 6 unit lattice (4-neighbour links, no walls), start (0,0), exit
(3,5), `k = 2`, the first route costs 8; attempts 0, 2, 4, 6 return a route of cost 8 with
62.5% of its length within 1.5 m of the first, attempts 1, 3, 5, 7 one with 100%; all are
rejected and `distinct` returns one route with the note "Only 1 of 2 routes use
substantially different space ... within 1.5x the shortest cost", although the route
round the edge, (0,0)-(4,0)-(4,5)-(3,5), costs 10 (stretch 1.25) and has 40% of its length
within 1.5 m of the first (50% the other way). This is that run in the model. -/
theorem distinct_can_miss_cycle : ∃ (P : DParams) (c : List ℕ → ℝ)
    (share : List ℕ → List ℕ → ℝ) (first good : List ℕ) (next : ℕ → Option (List ℕ)),
    2 ≤ P.k ∧ good ≠ first ∧ c good ≤ P.maxStretch * c first ∧
      share good first ≤ P.maxOverlap ∧
      (∀ i p, next i = some p → c p ≤ P.maxStretch * c first) ∧
      distinctRun P c share first next = [first] := by
  refine ⟨⟨3 / 2, 1 / 2, 1 / 10 ^ 9, 2⟩, fun p => if p = [3] then 10 else 8,
    fun p _ => if p = [3] then 2 / 5 else if p = [1] then 5 / 8 else 1, [0], [3],
    fun i => if i % 2 = 0 then some [1] else some [2], le_rfl, by decide, by norm_num,
    by norm_num, ?_, ?_⟩
  · intro i p hp
    simp only at hp
    split_ifs at hp <;> (simp only [Option.some.injEq] at hp; subst hp; norm_num)
  · simp only [distinctRun]
    rw [dloop_rejected]
    · simp
    · intro i p _ _ hp
      right
      refine ⟨[0], by simp, ?_⟩
      split_ifs at hp <;> (simp only [Option.some.injEq] at hp; subst hp; norm_num)

/-- L12.D6 (the overlap bound holds in discovery order, not in rank order). The panel says
`distinct` "keeps a route only if at most half of it runs within 1.5 m of a better one".
The code checks a new route against the routes kept *before* it, then sorts by cost, so a
route found early can end up ranked below a cheaper route found later that it was never
checked against. Here route `X` (cost 14) is kept at attempt 0 and `Y` (cost 13) at
attempt 1; `Y`'s share near `X` (40%) was checked, `X`'s share near `Y` (60%) was not; in
the returned list `Y` ranks before `X`, and `X` runs more than half within the separation
of a better route. Observed in a probe: a 15 x 10 lattice with diagonals, `k = 5`, routes
found at costs 15.90, 17.07, 22.83, 21.66; the 22.83 route ranks 4th and its reported
`overlap` is 51.8%. -/
theorem distinct_rank_overlap_gap : ∃ (P : DParams) (c : List ℕ → ℝ)
    (share : List ℕ → List ℕ → ℝ) (first X Y : List ℕ) (next : ℕ → Option (List ℕ)),
    (dloop P c share (c first) next (4 * P.k) 0 [first]).Pairwise
      (fun q p => share p q ≤ P.maxOverlap) ∧
    X ∈ distinctRun P c share first next ∧ Y ∈ distinctRun P c share first next ∧
    (distinctRun P c share first next).Pairwise (fun p q => c p ≤ c q) ∧
    c Y < c X ∧ P.maxOverlap < share X Y := by
  let P : DParams := ⟨3 / 2, 1 / 2, 1 / 10 ^ 9, 3⟩
  let c : List ℕ → ℝ := fun p => if p = [1] then 14 else if p = [2] then 13 else 10
  let share : List ℕ → List ℕ → ℝ := fun p q =>
    if p = [1] ∧ q = [2] then 3 / 5 else if p = [2] ∧ q = [1] then 2 / 5 else 0
  let next : ℕ → Option (List ℕ) := fun i => if i = 0 then some [1] else some [2]
  have hrun : dloop P c share (c [0]) next (4 * P.k) 0 [[0]] = [[0], [1], [2]] := by
    have e1 := dloop_accept P c share (c [0]) next 11 0 [[0]] [1] (by norm_num [P]) rfl
      (by norm_num [c, P]) (by norm_num [share, P])
    have e2 := dloop_accept P c share (c [0]) next 10 1 [[0], [1]] [2] (by norm_num [P]) rfl
      (by norm_num [c, P]) (by norm_num [share, P])
    have e3 := dloop_full P c share (c [0]) next 10 2 [[0], [1], [2]] (by norm_num [P])
    exact e1.trans (e2.trans e3)
  have hmem : ∀ p ∈ [[1], [2]], p ∈ distinctRun P c share [0] next := by
    intro p hp
    simp only [distinctRun]
    rw [hrun]
    exact List.mem_cons_of_mem _ (List.mem_mergeSort.mpr hp)
  refine ⟨P, c, share, [0], [1], [2], next, distinct_overlap _ _ _ _ _, hmem _ (by simp),
    hmem _ (by simp), distinct_sorted_of _ _ _ _ _ ?_, by norm_num [c], by norm_num [share, P]⟩
  intro i p hp
  simp only [next] at hp
  split_ifs at hp <;> (simp only [Option.some.injEq] at hp; subst hp; norm_num [c])

end FireEgress.Yen

#print axioms FireEgress.Yen.linked_append
#print axioms FireEgress.Yen.cost_append
#print axioms FireEgress.Yen.linked_mono_mem
#print axioms FireEgress.Yen.linked_mono
#print axioms FireEgress.Yen.IsPath.mono
#print axioms FireEgress.Yen.linked_mem_tail
#print axioms FireEgress.Yen.mem_of_getLast?
#print axioms FireEgress.Yen.getLast?_append_cons
#print axioms FireEgress.Yen.eq_nil_or_snoc
#print axioms FireEgress.Yen.mem_getLast?_snoc
#print axioms FireEgress.Yen.KShortest.le_outside
#print axioms FireEgress.Yen.KShortest.take
#print axioms FireEgress.Yen.KShortest.snoc
#print axioms FireEgress.Yen.SpurLink.mono
#print axioms FireEgress.Yen.Processed.mono
#print axioms FireEgress.Yen.candidate_cost
#print axioms FireEgress.Yen.spur_candidate_simple
#print axioms FireEgress.Yen.le_cpl
#print axioms FireEgress.Yen.cpl_split
#print axioms FireEgress.Yen.exists_last_split
#print axioms FireEgress.Yen.deviation
#print axioms FireEgress.Yen.yen_key
#print axioms FireEgress.Yen.yinv_init
#print axioms FireEgress.Yen.round_facts
#print axioms FireEgress.Yen.yinv_step
#print axioms FireEgress.Yen.yen_correct
#print axioms FireEgress.Yen.yen_kshortest
#print axioms FireEgress.Yen.yen_accept_next
#print axioms FireEgress.Yen.yen_dry_complete
#print axioms FireEgress.Yen.yen_length_le
#print axioms FireEgress.Yen.walkCost_cons
#print axioms FireEgress.Yen.walkCost_of_isPath
#print axioms FireEgress.Yen.shortestPath_of_astar
#print axioms FireEgress.Yen.shortestPath_of_astar_straight_line
#print axioms FireEgress.Yen.no_walk_of_astar_dry
#print axioms FireEgress.Yen.dloop_spec
#print axioms FireEgress.Yen.dloop_pairwise
#print axioms FireEgress.Yen.dloop_length
#print axioms FireEgress.Yen.dloop_congr
#print axioms FireEgress.Yen.dloop_accept
#print axioms FireEgress.Yen.dloop_full
#print axioms FireEgress.Yen.dloop_rejected
#print axioms FireEgress.Yen.distinct_found
#print axioms FireEgress.Yen.distinct_stretch
#print axioms FireEgress.Yen.distinct_overlap
#print axioms FireEgress.Yen.distinct_bounded
#print axioms FireEgress.Yen.distinct_sorted_of
#print axioms FireEgress.Yen.distinct_sorted
#print axioms FireEgress.Yen.distinct_can_miss
#print axioms FireEgress.Yen.distinct_can_miss_cycle
#print axioms FireEgress.Yen.distinct_rank_overlap_gap
