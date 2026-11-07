import Mathlib
import FireEgress.Sprinklers
import FireEgress.Suppression

/-!
# Extinguishers: placement on the walking graph and one person's attempt

Specifications for `src/core/extinguishers.js`, items 1 to 11 of
`wiki/research/extinguishers-2026-11-07.md` section 6.5. Real arithmetic; the walking graph enters
through its distances (`δ`, `T`), not its links. It reuses Sprinklers.lean's `decay` and
Suppression.lean's `evansCap` and `evansCap_sum`. Each counterexample sits beside the theorem it
limits.

## Which theorem backs which part of the code

* Item 1, `placementCheck`'s multi-source walk (`walkFrom`, every site at 0, `Infinity` when none
  is reached): `travel` (the least over the sites, `⊤` for none), `travel_mono_sites` (adding a
  site never lengthens any node's walk), `travel_le_site`, `travel_pass_mono` (a passing layout
  stays passing when units are added).
* Item 2, a pass on the graph is a pass of the radius check, not the converse: `graph_ge_euclid`
  (a walk whose links are at least their straight lengths is at least the straight line between its
  ends, in any metric space), `graph_pass_radius`; beside them `radius_pass_not_graph` (a point 1 m
  away in a straight line, 3 m away by the walk).
* Item 3, the sampling margin `delta = s / √2` (`samplingMargin`): `sample_corner` (every point of
  an `s` by `s` cell lies within `s / √2` of one of its corners), `sample_margin` (a node within
  `D - delta` passes for every floor point within `delta` of it; with the code's 1e-9 as `tol`).
* Item 4, `requiredCount`, `max(1, ⌈area / min(n_A a_h, 1045) - 1e-9⌉)` for a positive area and
  rating: `count_rule_antitone` (never rises with a higher rating `n_A` or a larger area per unit
  `a_h`, that is a lighter hazard at the same rating), `count_rule_pos` (at least 1),
  `count_area_mono`. Beside them `count_default_rating_counterexample`: with each hazard's least
  rating (2-A ordinary, 4-A extra, the code's default when no rating is given), 1000 m² needs 4
  units at ordinary hazard and 3 at extra, so "never rises with a lighter hazard" holds for a given
  rating only.
* Item 5, `layoutExtinguishers`' greedy cover (`gain`, `best`, `greedy`): `gain_antitone` (a
  candidate's count only falls as places are covered, the lazy queue's premise), `cands_nonempty`,
  `best_spec`, `best_gain`, `card_sdiff_lt`, `greedy_covers` (if the candidates' balls cover the
  storey's places, greedy ends with every place covered), `greedy_card_le` (with at most as many
  units as places), `greedy_layout` (both, from nothing covered). Chvátal's `H(m)` bound is not
  proved.
* Item 6, `attempt`'s discharge time `t_s = t_go + T(start, ext) + t_take + T(ext, attack)`
  (`arrival`): `arrival_mono` (non-decreasing in `t_go`, so in the response time `t_r`, in `t_take`
  and in each walking time), `walkTime_mono`, `walkTime_anti` (walking times `d / v` rise with the
  distance and fall with the speed).
* Item 7, the three-valued size verdict of `judge` (failure above `high + 1e-9`, cannot tell above
  `low + 1e-9`, success otherwise; `Verdict`, `Verdict.rank`): `judge_success_iff`,
  `judge_failure_iff`, `judge_mono` (the verdict never improves with a bigger fire),
  `success_antitone` (if `Q` is non-decreasing on `[t, t']`, success at `t'` implies success at `t`),
  `failure_mono` (and failure at `t` implies failure at `t'`), `capacity_mono`,
  `success_mono_capacity` (success is monotone in `n_A`, `q_A` and `eta`), `capacity_2A` (a 2-A unit:
  success up to 80 kW, failure above 900 kW), `success_mono_reach` (the reach test, monotone in
  `r`), `success_iff_tsq` and `judge_tsq` (for `Q = alpha t²`: success iff
  `t ≤ √((low + tol) / alpha)`), `worked_example` (the note's t² example: 68 s succeeds, 368 s
  fails).
* Item 8, `timeLost = t_s + t_d + E(attack) - (t_go + E(start))`, `E` the walking time to the
  nearest exit: `exitTime_triangle` (the time to the nearest exit satisfies
  `E u ≤ T u v + E v` when `T` satisfies the triangle inequality), `time_lost_nonneg` (the time lost
  is at least `t_take + t_d ≥ 0`), `time_lost_nonneg_exits` (the same from the triangle inequality
  of `T` alone).
* Item 9, `capFor`, `tau_x = t_d / ln(1/epsilon)`: `log_inv_pos`, `capFor_pos`,
  `decay_at_discharge_end` (the cap at `t_s + t_d` is exactly `epsilon r_c(t_s)`), `tau_mono_tD`,
  `tau_mono_eps` (`tau_x` rises with `t_d` and with `epsilon`).
* Item 10, `extinguisherCap`: `reached_energy_le` (the reached cells release after `t_s` at most
  `Σ r_c (tau_x + dt)`, from `evansCap_sum`, with each cell's start at most its rate),
  `cap_never_rises` (where a cell is already under a slower sprinkler cap, the start
  `min(rate, envelope)` and the faster decay keep the new cap at or below the old at every later
  time).
* Item 11: `cap_not_global`, a two-frame toy fire. The cap scales the reached cell's burn by a
  factor at most 1, yet gas it leaves unburnt moves on and burns in a later frame, where the
  capped fire releases more heat than the free one; the total over the frames is the same. The
  provable statements are about the reached cells' fuel and energy, not a global peak.

## Hypotheses, and what is not covered

* Item 1 states the multi-source walk as a least over the sites; that `walkFrom`'s Dijkstra
  computes it is the label-setting result of AStar.lean, not repeated here. The storeys, stairs and
  doors of `placementCheck` are not modelled.
* Item 2 needs every link at least the straight line between its nodes (`edgeLengths` are those
  lengths). Item 3 assumes a full lattice cell around the point; near walls the node at the nearest
  corner can be missing, which the margin does not cover.
* Item 5 models the greedy choice by `Finset.exists_max_image`; ties by exit distance and the
  lazy heap are not modelled (any maximiser will do, and the proofs do not use maximality). The
  fallback pass's candidates are thinned to one per metre in plan, which need not keep them within
  walking reach of every uncovered place: the layout is checked again by `placementCheck`.
* Item 7 covers the size verdict for one fire; the overall outcome also takes the agent and the
  reach at `t_s` (`success_mono_reach` covers the reach test alone).
* Item 8 needs the walking times symmetric in the sense the code uses them (`toRing` walks from the
  attack ring to the extinguisher; the graph is undirected) and one speed throughout.
* Floating point is not modelled; the code's tolerances (1e-9) enter as `tol`.
-/

namespace FireEgress.Extinguishers

open Real

/-! ## Item 1: the multi-source walk -/

/-- `walkFrom`'s distance with every site at 0: the least walk `δ v e` from `v` to a site of `E`,
`⊤` (the code's `Infinity`) when no site is reached. -/
noncomputable def travel {V : Type*} (δ : V → V → WithTop ℝ) (E : Finset V) (v : V) :
    WithTop ℝ :=
  E.inf (δ v)

/-- Adding extinguishers never lengthens any node's walk to the nearest one. -/
theorem travel_mono_sites {V : Type*} (δ : V → V → WithTop ℝ) {E E' : Finset V} (h : E ⊆ E')
    (v : V) : travel δ E' v ≤ travel δ E v :=
  Finset.inf_mono h

theorem travel_le_site {V : Type*} (δ : V → V → WithTop ℝ) {E : Finset V} {e : V} (he : e ∈ E)
    (v : V) : travel δ E v ≤ δ v e :=
  Finset.inf_le he

/-- A layout that passes the travel rule still passes when units are added. -/
theorem travel_pass_mono {V : Type*} (δ : V → V → WithTop ℝ) {E E' P : Finset V} {D : ℝ}
    (h : E ⊆ E') (hp : ∀ v ∈ P, travel δ E v ≤ D) : ∀ v ∈ P, travel δ E' v ≤ D :=
  fun v hv => (travel_mono_sites δ h v).trans (hp v hv)

/-! ## Item 2: the walk against the straight line -/

/-- A walk whose links are at least their straight lengths is at least the straight line between
its ends. -/
theorem graph_ge_euclid {X : Type*} [PseudoMetricSpace X] (p : ℕ → X) (w : ℕ → ℝ)
    (hw : ∀ i, dist (p i) (p (i + 1)) ≤ w i) (n : ℕ) :
    dist (p 0) (p n) ≤ ∑ i ∈ Finset.range n, w i := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Finset.sum_range_succ]
    exact (dist_triangle _ _ _).trans (add_le_add ih (hw n))

/-- A pass on the graph within `D` is a pass of the radius check within `D`. -/
theorem graph_pass_radius {X : Type*} [PseudoMetricSpace X] (p : ℕ → X) (w : ℕ → ℝ)
    (hw : ∀ i, dist (p i) (p (i + 1)) ≤ w i) {n : ℕ} {D : ℝ}
    (hD : ∑ i ∈ Finset.range n, w i ≤ D) : dist (p 0) (p n) ≤ D :=
  (graph_ge_euclid p w hw n).trans hD

/-- Not the converse: a point 1 m away in a straight line can be 3 m away along the walk. -/
theorem radius_pass_not_graph :
    ∃ p : ℕ → ℝ, dist (p 0) (p 2) ≤ 1 ∧ 1 < ∑ i ∈ Finset.range 2, dist (p i) (p (i + 1)) := by
  refine ⟨fun i => if i = 1 then 2 else if i = 2 then 1 else 0, ?_, ?_⟩
  · norm_num [Real.dist_eq]
  · norm_num [Finset.sum_range_succ, Real.dist_eq]

/-! ## Item 3: the sampling margin -/

/-- `samplingMargin(s) = s / √2`. -/
noncomputable def samplingMargin (s : ℝ) : ℝ := s / Real.sqrt 2

/-- Every point of an `s` by `s` cell lies within `s / √2` of one of its corners. -/
theorem sample_corner {s a b : ℝ} (hs : 0 ≤ s) (ha0 : 0 ≤ a) (ha : a ≤ s) (hb0 : 0 ≤ b)
    (hb : b ≤ s) : ∃ i j : ℝ, (i = 0 ∨ i = s) ∧ (j = 0 ∨ j = s) ∧
      Real.sqrt ((a - i) ^ 2 + (b - j) ^ 2) ≤ samplingMargin s := by
  have key : ∀ x : ℝ, 0 ≤ x → x ≤ s → ∃ i : ℝ, (i = 0 ∨ i = s) ∧ (x - i) ^ 2 ≤ s ^ 2 / 4 := by
    intro x hx0 hx
    by_cases h : x ≤ s / 2
    · exact ⟨0, Or.inl rfl, by nlinarith⟩
    · have h' : s / 2 < x := not_le.1 h
      exact ⟨s, Or.inr rfl, by nlinarith⟩
  obtain ⟨i, hi, hai⟩ := key a ha0 ha
  obtain ⟨j, hj, hbj⟩ := key b hb0 hb
  refine ⟨i, j, hi, hj, ?_⟩
  have h2 : Real.sqrt (s ^ 2 / 2) = samplingMargin s := by
    rw [samplingMargin, Real.sqrt_div (sq_nonneg s), Real.sqrt_sq hs]
  rw [← h2]
  exact Real.sqrt_le_sqrt (by linarith)

/-- A node passing within `D - delta` passes for every floor point within `delta` of it: the point
walks straight to its node, then on. -/
theorem sample_margin {X : Type*} [PseudoMetricSpace X] {walk : X → ℝ} {x v : X}
    {delta D tol : ℝ} (hwalk : walk x ≤ dist x v + walk v) (hx : dist x v ≤ delta)
    (hv : walk v ≤ D - delta + tol) : walk x ≤ D + tol := by
  linarith

/-! ## Item 4: the count rule -/

/-- `requiredCount` for a positive area and rating: `max(1, ⌈area / min(n_A a_h, 1045) - tol⌉)`,
with the code's `tol = 1e-9`. -/
noncomputable def requiredCount (area aH nA tol : ℝ) : ℤ :=
  max 1 ⌈area / min (nA * aH) 1045 - tol⌉

/-- The required number never rises with a higher rating or a larger area per unit. -/
theorem count_rule_antitone {area aH aH' nA nA' tol : ℝ} (hA : 0 ≤ area) (haH : 0 < aH)
    (hnA : 0 < nA) (hh : aH ≤ aH') (hn : nA ≤ nA') :
    requiredCount area aH' nA' tol ≤ requiredCount area aH nA tol := by
  unfold requiredCount
  have hp : 0 < min (nA * aH) 1045 := lt_min (mul_pos hnA haH) (by norm_num)
  have hpp : min (nA * aH) 1045 ≤ min (nA' * aH') 1045 :=
    min_le_min (mul_le_mul hn hh haH.le (by linarith)) le_rfl
  have := div_le_div_of_nonneg_left hA hp hpp
  exact max_le_max le_rfl (Int.ceil_mono (by linarith))

/-- At least one unit on any floor. -/
theorem count_rule_pos (area aH nA tol : ℝ) : 1 ≤ requiredCount area aH nA tol :=
  le_max_left _ _

/-- More floor, no fewer units. -/
theorem count_area_mono {area area' aH nA tol : ℝ} (haH : 0 < aH) (hnA : 0 < nA)
    (h : area ≤ area') : requiredCount area aH nA tol ≤ requiredCount area' aH nA tol := by
  unfold requiredCount
  have hp : 0 < min (nA * aH) 1045 := lt_min (mul_pos hnA haH) (by norm_num)
  have := div_le_div_of_nonneg_right h hp.le
  exact max_le_max le_rfl (Int.ceil_mono (by linarith))

/-- With each hazard's least rating, the code's default (2-A at ordinary hazard, 139 m² per unit;
4-A at extra, 92.9 m²), 1000 m² needs 4 units at ordinary hazard and 3 at extra: the lighter
hazard needs more. -/
theorem count_default_rating_counterexample :
    requiredCount 1000 92.9 4 1e-9 = 3 ∧ requiredCount 1000 139 2 1e-9 = 4 := by
  constructor
  · unfold requiredCount
    rw [min_eq_left (by norm_num : (4 : ℝ) * 92.9 ≤ 1045)]
    have h : ⌈(1000 : ℝ) / (4 * 92.9) - 1e-9⌉ = 3 := by
      rw [Int.ceil_eq_iff]
      norm_num
    rw [h]
    exact max_eq_right (by norm_num)
  · unfold requiredCount
    rw [min_eq_left (by norm_num : (2 : ℝ) * 139 ≤ 1045)]
    have h : ⌈(1000 : ℝ) / (2 * 139) - 1e-9⌉ = 4 := by
      rw [Int.ceil_eq_iff]
      norm_num
    rw [h]
    exact max_eq_right (by norm_num)

/-! ## Item 5: the greedy cover -/

section Greedy

variable {α β : Type*} [DecidableEq α]

/-- The places candidate `c` would newly cover: the places of its ball not yet covered. -/
def gain (U cov : Finset α) (ball : β → Finset α) (c : β) : Finset α := (ball c ∩ U) \ cov

/-- A candidate's count only falls as places are covered: the premise of the lazy queue. -/
theorem gain_antitone {U cov cov' : Finset α} {ball : β → Finset α} (h : cov ⊆ cov') (c : β) :
    gain U cov' ball c ⊆ gain U cov ball c :=
  Finset.sdiff_subset_sdiff (Finset.Subset.refl _) h

theorem cands_nonempty {C : Finset β} {P : β → Prop} (h : ∃ c ∈ C, P c) : C.Nonempty := by
  obtain ⟨c, hc, -⟩ := h
  exact ⟨c, hc⟩

/-- A candidate covering the most uncovered places. -/
noncomputable def best (U cov : Finset α) (C : Finset β) (ball : β → Finset α)
    (hC : C.Nonempty) : β :=
  Classical.choose (C.exists_max_image (fun c => (gain U cov ball c).card) hC)

theorem best_spec (U cov : Finset α) (C : Finset β) (ball : β → Finset α) (hC : C.Nonempty) :
    best U cov C ball hC ∈ C ∧
      ∀ c ∈ C, (gain U cov ball c).card ≤ (gain U cov ball (best U cov C ball hC)).card :=
  Classical.choose_spec (C.exists_max_image (fun c => (gain U cov ball c).card) hC)

/-- Greedy set cover as `layoutExtinguishers` runs it: while some candidate covers an uncovered
place, take one covering the most; stop when none does (or after `k` units). -/
noncomputable def greedy (U : Finset α) (C : Finset β) (ball : β → Finset α) :
    ℕ → Finset α → List β
  | 0, _ => []
  | k + 1, cov =>
    if h : ∃ c ∈ C, (gain U cov ball c).Nonempty then
      best U cov C ball (cands_nonempty h) ::
        greedy U C ball k (cov ∪ (ball (best U cov C ball (cands_nonempty h)) ∩ U))
    else []

/-- The candidate taken covers at least one new place. -/
theorem best_gain {U cov : Finset α} {C : Finset β} {ball : β → Finset α}
    (h : ∃ c ∈ C, (gain U cov ball c).Nonempty) :
    (gain U cov ball (best U cov C ball (cands_nonempty h))).Nonempty := by
  obtain ⟨c, hc, hne⟩ := id h
  have hle := (best_spec U cov C ball (cands_nonempty h)).2 c hc
  rw [← Finset.card_pos] at hne ⊢
  exact lt_of_lt_of_le hne hle

/-- Each unit taken leaves fewer places uncovered. -/
theorem card_sdiff_lt {U cov s : Finset α} (h : ((s ∩ U) \ cov).Nonempty) :
    (U \ (cov ∪ (s ∩ U))).card < (U \ cov).card := by
  apply Finset.card_lt_card
  rw [Finset.ssubset_iff_of_subset
    (Finset.sdiff_subset_sdiff (Finset.Subset.refl _) Finset.subset_union_left)]
  obtain ⟨u, hu⟩ := h
  simp only [Finset.mem_sdiff, Finset.mem_inter] at hu
  exact ⟨u, Finset.mem_sdiff.2 ⟨hu.1.2, hu.2⟩,
    fun h' => (Finset.mem_sdiff.1 h').2 (Finset.mem_union_right _ (Finset.mem_inter.2 hu.1))⟩

/-- If the candidates' balls cover the places, greedy ends with every place covered. -/
theorem greedy_covers {U : Finset α} {C : Finset β} {ball : β → Finset α}
    (hcov : ∀ u ∈ U, ∃ c ∈ C, u ∈ ball c) :
    ∀ (k : ℕ) (cov : Finset α), (U \ cov).card ≤ k →
      ∀ u ∈ U, u ∈ cov ∨ ∃ c ∈ greedy U C ball k cov, u ∈ ball c := by
  intro k
  induction k with
  | zero =>
    intro cov hk u hu
    left
    by_contra hn
    have he : U \ cov = ∅ := Finset.card_eq_zero.1 (by omega)
    have hm : u ∈ U \ cov := Finset.mem_sdiff.2 ⟨hu, hn⟩
    rw [he] at hm
    simp at hm
  | succ k ih =>
    intro cov hk u hu
    simp only [greedy]
    split_ifs with h
    · have hlt := card_sdiff_lt (best_gain h)
      rcases ih _ (Nat.le_of_lt_succ (lt_of_lt_of_le hlt hk)) u hu with hc | ⟨d, hd, hud⟩
      · rcases Finset.mem_union.1 hc with hc | hc
        · exact Or.inl hc
        · exact Or.inr ⟨_, List.mem_cons.2 (Or.inl rfl), (Finset.mem_inter.1 hc).1⟩
      · exact Or.inr ⟨d, List.mem_cons.2 (Or.inr hd), hud⟩
    · left
      by_contra hn
      obtain ⟨c, hc, hu'⟩ := hcov u hu
      exact h ⟨c, hc, u, Finset.mem_sdiff.2 ⟨Finset.mem_inter.2 ⟨hu', hu⟩, hn⟩⟩

/-- Greedy takes at most as many units as there are uncovered places. -/
theorem greedy_card_le {U : Finset α} {C : Finset β} {ball : β → Finset α} :
    ∀ (k : ℕ) (cov : Finset α), (greedy U C ball k cov).length ≤ (U \ cov).card := by
  intro k
  induction k with
  | zero => intro cov; simp [greedy]
  | succ k ih =>
    intro cov
    simp only [greedy]
    split_ifs with h
    · have hlt := card_sdiff_lt (best_gain h)
      have hrest := ih (cov ∪ (ball (best U cov C ball (cands_nonempty h)) ∩ U))
      simp only [List.length_cons]
      omega
    · simp

/-- From nothing covered: a cover of every place with at most as many units as places. -/
theorem greedy_layout {U : Finset α} {C : Finset β} {ball : β → Finset α}
    (hcov : ∀ u ∈ U, ∃ c ∈ C, u ∈ ball c) :
    (∀ u ∈ U, ∃ c ∈ greedy U C ball U.card ∅, u ∈ ball c) ∧
      (greedy U C ball U.card ∅).length ≤ U.card := by
  refine ⟨fun u hu => ?_, by simpa using greedy_card_le (U := U) (C := C) (ball := ball) U.card ∅⟩
  rcases greedy_covers hcov U.card ∅ (by simp) u hu with h | h
  · simp at h
  · exact h

end Greedy

/-! ## Item 6: when discharge starts -/

/-- `t_s = t_go + T(start, ext) + t_take + T(ext, attack)` (`tLeave + toAttack`). -/
def arrival (tGo walk1 tTake walk2 : ℝ) : ℝ := tGo + walk1 + tTake + walk2

/-- `t_s` is non-decreasing in the start (so in the response time), the readying time and each
walking time. -/
theorem arrival_mono {tGo tGo' w1 w1' tk tk' w2 w2' : ℝ} (h0 : tGo ≤ tGo') (h1 : w1 ≤ w1')
    (h2 : tk ≤ tk') (h3 : w2 ≤ w2') : arrival tGo w1 tk w2 ≤ arrival tGo' w1' tk' w2' := by
  unfold arrival
  linarith

/-- A walking time `d / v` rises with the distance ... -/
theorem walkTime_mono {d d' v : ℝ} (hv : 0 < v) (h : d ≤ d') : d / v ≤ d' / v :=
  div_le_div_of_nonneg_right h hv.le

/-- ... and falls with the speed. -/
theorem walkTime_anti {d v v' : ℝ} (hd : 0 ≤ d) (hv : 0 < v) (h : v ≤ v') : d / v' ≤ d / v :=
  div_le_div_of_nonneg_left hd hv h

/-! ## Item 7: the size verdict -/

/-- The outcome of `judge`, ordered success, cannot tell, failure. -/
inductive Verdict
  | success
  | cannotTell
  | failure

/-- The order of the verdicts (the code's `outcome`: failure over cannot tell over success). -/
def Verdict.rank : Verdict → ℕ
  | .success => 0
  | .cannotTell => 1
  | .failure => 2

/-- `judge(value, low, high)`: failure above `high + tol`, cannot tell above `low + tol`, else
success (the code's `tol = 1e-9`). -/
noncomputable def judge (tol low high Q : ℝ) : Verdict :=
  if high + tol < Q then .failure else if low + tol < Q then .cannotTell else .success

theorem judge_success_iff {tol low high Q : ℝ} (hlh : low ≤ high) :
    judge tol low high Q = .success ↔ Q ≤ low + tol := by
  unfold judge
  by_cases h1 : high + tol < Q
  · rw [ite_eq_left h1]
    exact ⟨(fun h => nomatch h), fun h => absurd h (not_le.2 (by linarith))⟩
  · rw [ite_eq_right h1]
    by_cases h2 : low + tol < Q
    · rw [ite_eq_left h2]
      exact ⟨(fun h => nomatch h), fun h => absurd h (not_le.2 h2)⟩
    · rw [ite_eq_right h2]
      exact ⟨fun _ => not_lt.1 h2, fun _ => rfl⟩

theorem judge_failure_iff {tol low high Q : ℝ} :
    judge tol low high Q = .failure ↔ high + tol < Q := by
  unfold judge
  by_cases h1 : high + tol < Q
  · rw [ite_eq_left h1]
    exact ⟨fun _ => h1, fun _ => rfl⟩
  · rw [ite_eq_right h1]
    by_cases h2 : low + tol < Q
    · rw [ite_eq_left h2]
      exact ⟨(fun h => nomatch h), fun h => absurd h h1⟩
    · rw [ite_eq_right h2]
      exact ⟨(fun h => nomatch h), fun h => absurd h h1⟩

/-- A bigger fire never gets a better verdict. -/
theorem judge_mono {tol low high Q Q' : ℝ} (hQ : Q ≤ Q') :
    (judge tol low high Q).rank ≤ (judge tol low high Q').rank := by
  unfold judge
  by_cases h1 : high + tol < Q
  · rw [ite_eq_left h1, ite_eq_left (lt_of_lt_of_le h1 hQ)]
  · rw [ite_eq_right h1]
    by_cases h2 : low + tol < Q
    · rw [ite_eq_left h2]
      by_cases h3 : high + tol < Q'
      · rw [ite_eq_left h3]
        simp [Verdict.rank]
      · rw [ite_eq_right h3, ite_eq_left (lt_of_lt_of_le h2 hQ)]
    · rw [ite_eq_right h2]
      simp [Verdict.rank]

/-- While the fire grows (`Q` non-decreasing on `[t, t']`, the hypothesis is needed), success at
the later time implies success at the earlier: a later arrival never turns failure into
success. -/
theorem success_antitone {Q : ℝ → ℝ} {tol low high t t' : ℝ} (hlh : low ≤ high) (htt : t ≤ t')
    (hQ : MonotoneOn Q (Set.Icc t t')) (h : judge tol low high (Q t') = .success) :
    judge tol low high (Q t) = .success := by
  rw [judge_success_iff hlh] at h ⊢
  exact le_trans (hQ ⟨le_rfl, htt⟩ ⟨htt, le_rfl⟩ htt) h

/-- ... and failure at the earlier time implies failure at the later. -/
theorem failure_mono {Q : ℝ → ℝ} {tol low high t t' : ℝ} (htt : t ≤ t')
    (hQ : MonotoneOn Q (Set.Icc t t')) (h : judge tol low high (Q t) = .failure) :
    judge tol low high (Q t') = .failure := by
  rw [judge_failure_iff] at h ⊢
  exact lt_of_lt_of_le h (hQ ⟨le_rfl, htt⟩ ⟨htt, le_rfl⟩ htt)

/-- The class A capacity `eta q_A n_A` (kW). -/
def capacity (eta qA nA : ℝ) : ℝ := eta * qA * nA

theorem capacity_mono {eta qA nA eta' qA' nA' : ℝ} (he0 : 0 ≤ eta) (hq0 : 0 ≤ qA)
    (hn0 : 0 ≤ nA) (he : eta ≤ eta') (hq : qA ≤ qA') (hn : nA ≤ nA') :
    capacity eta qA nA ≤ capacity eta' qA' nA' := by
  unfold capacity
  exact mul_le_mul (mul_le_mul he hq hq0 (le_trans he0 he)) hn hn0
    (mul_nonneg (le_trans he0 he) (le_trans hq0 hq))

/-- Success is monotone in `n_A`, `q_A` and `eta`. -/
theorem success_mono_capacity {tol Q eta qA nA eta' qA' nA' high high' : ℝ} (he0 : 0 ≤ eta)
    (hq0 : 0 ≤ qA) (hn0 : 0 ≤ nA) (he : eta ≤ eta') (hq : qA ≤ qA') (hn : nA ≤ nA')
    (hlh : capacity eta qA nA ≤ high) (hlh' : capacity eta' qA' nA' ≤ high')
    (h : judge tol (capacity eta qA nA) high Q = .success) :
    judge tol (capacity eta' qA' nA') high' Q = .success := by
  rw [judge_success_iff hlh] at h
  rw [judge_success_iff hlh']
  linarith [capacity_mono he0 hq0 hn0 he hq hn]

/-- A 2-A unit at the low and high ends (`eta` 0.2 and 1.0, `q_A` 200 and 450 kW): success up to
80 kW, failure above 900 kW. -/
theorem capacity_2A : capacity 0.2 200 2 = 80 ∧ capacity 1.0 450 2 = 900 := by
  norm_num [capacity]

/-- The reach test at `t_s`, no burning point farther than `r + delta + tol`, passes for every
longer reach. -/
theorem success_mono_reach {tol r r' delta farthest : ℝ} (h : farthest ≤ r + delta + tol)
    (hr : r ≤ r') : farthest ≤ r' + delta + tol := by
  linarith

/-- For a t² fire, `alpha t² ≤ cap` exactly when `t ≤ √(cap / alpha)`. -/
theorem success_iff_tsq {alpha cap t : ℝ} (ha : 0 < alpha) (ht : 0 ≤ t) (hc : 0 ≤ cap) :
    alpha * t ^ 2 ≤ cap ↔ t ≤ Real.sqrt (cap / alpha) := by
  have hq : 0 ≤ cap / alpha := div_nonneg hc ha.le
  constructor
  · intro h
    have h' : t ^ 2 ≤ cap / alpha := by
      rw [le_div_iff₀ ha]
      linarith
    calc t = Real.sqrt (t ^ 2) := (Real.sqrt_sq ht).symm
      _ ≤ Real.sqrt (cap / alpha) := Real.sqrt_le_sqrt h'
  · intro h
    have h2 : t * t ≤ Real.sqrt (cap / alpha) * Real.sqrt (cap / alpha) :=
      mul_self_le_mul_self ht h
    rw [Real.mul_self_sqrt hq, le_div_iff₀ ha] at h2
    nlinarith

/-- The verdict for a t² fire: success exactly while `t ≤ √((low + tol) / alpha)`. -/
theorem judge_tsq {tol low high alpha t : ℝ} (hlh : low ≤ high) (ha : 0 < alpha) (ht : 0 ≤ t)
    (hc : 0 ≤ low + tol) :
    judge tol low high (alpha * t ^ 2) = .success ↔ t ≤ Real.sqrt ((low + tol) / alpha) := by
  rw [judge_success_iff hlh, success_iff_tsq ha ht hc]

/-- The note's worked example, a medium t² fire (alpha 0.0117 kW/s²) and a 2-A unit: discharge at
68 s (54 kW) succeeds even at the low ends; at 368 s (1.6 MW) it fails even at the high ends. -/
theorem worked_example :
    judge 1e-9 (capacity 0.2 200 2) (capacity 1.0 450 2) (0.0117 * 68 ^ 2) = .success ∧
      judge 1e-9 (capacity 0.2 200 2) (capacity 1.0 450 2) (0.0117 * 368 ^ 2) = .failure := by
  have hlh : capacity 0.2 200 2 ≤ capacity 1.0 450 2 := by norm_num [capacity]
  rw [judge_success_iff hlh, judge_failure_iff]
  norm_num [capacity]

/-! ## Item 8: the time lost -/

/-- The walking time to the nearest exit of `X`. -/
noncomputable def exitTime {V : Type*} (T : V → V → ℝ) (X : Finset V) (hX : X.Nonempty)
    (u : V) : ℝ :=
  X.inf' hX (T u)

/-- With `T` satisfying the triangle inequality, so does the time to the nearest exit. -/
theorem exitTime_triangle {V : Type*} {T : V → V → ℝ} {X : Finset V} (hX : X.Nonempty)
    (hT : ∀ u v w, T u w ≤ T u v + T v w) (u v : V) :
    exitTime T X hX u ≤ T u v + exitTime T X hX v := by
  obtain ⟨x, hx, hxv⟩ := Finset.exists_mem_eq_inf' hX (T v)
  unfold exitTime
  rw [hxv]
  exact (Finset.inf'_le _ hx).trans (hT u v x)

/-- `timeLost = t_s + t_d + E(attack) - (t_go + E(start))`. -/
def timeLost (tGo tS tD Eattack Estart : ℝ) : ℝ := tS + tD + Eattack - (tGo + Estart)

/-- The responder's evacuation with an attempt never starts earlier than without one: the time
lost is at least `t_take + t_d ≥ 0`. -/
theorem time_lost_nonneg {V : Type*} {T : V → V → ℝ} {E : V → ℝ}
    (hE : ∀ u v, E u ≤ T u v + E v) (start ext attack : V) {tGo tTake tD : ℝ}
    (htake : 0 ≤ tTake) (htD : 0 ≤ tD) :
    tTake + tD ≤ timeLost tGo (arrival tGo (T start ext) tTake (T ext attack)) tD (E attack)
        (E start) ∧
      0 ≤ timeLost tGo (arrival tGo (T start ext) tTake (T ext attack)) tD (E attack)
        (E start) := by
  have h1 := hE start ext
  have h2 := hE ext attack
  unfold timeLost arrival
  constructor <;> linarith

/-- The same from the triangle inequality of the walking times alone, with `E` the time to the
nearest exit. -/
theorem time_lost_nonneg_exits {V : Type*} {T : V → V → ℝ} {X : Finset V} (hX : X.Nonempty)
    (hT : ∀ u v w, T u w ≤ T u v + T v w) (start ext attack : V) {tGo tTake tD : ℝ}
    (htake : 0 ≤ tTake) (htD : 0 ≤ tD) :
    0 ≤ timeLost tGo (arrival tGo (T start ext) tTake (T ext attack)) tD
      (exitTime T X hX attack) (exitTime T X hX start) :=
  (time_lost_nonneg (exitTime_triangle hX hT) start ext attack htake htD).2

/-! ## Item 9: the decay constant -/

/-- `capFor`: `tau_x = t_d / ln(1/epsilon)`. -/
noncomputable def capFor (tD eps : ℝ) : ℝ := tD / Real.log (1 / eps)

theorem log_inv_pos {eps : ℝ} (h0 : 0 < eps) (h1 : eps < 1) : 0 < Real.log (1 / eps) :=
  Real.log_pos (one_lt_one_div h0 h1)

theorem capFor_pos {tD eps : ℝ} (h0 : 0 < eps) (h1 : eps < 1) (htD : 0 < tD) :
    0 < capFor tD eps :=
  div_pos htD (log_inv_pos h0 h1)

/-- The cap at the end of the discharge, `t_s + t_d`, is exactly `epsilon r_c(t_s)`. It holds for
every `epsilon > 0`; the code takes `0 < epsilon < 1`, where `tau_x > 0` (`capFor_pos`). -/
theorem decay_at_discharge_end {tD eps : ℝ} (h0 : 0 < eps) (htD : 0 < tD) (r tS : ℝ) :
    Sprinklers.decay r tS (capFor tD eps) (tS + tD) = eps * r := by
  unfold Sprinklers.decay capFor
  have hlog : (tS + tD - tS) / (tD / Real.log (1 / eps)) = Real.log (1 / eps) := by
    rw [show tS + tD - tS = tD by ring, div_div_eq_mul_div, mul_div_cancel_left₀ _ htD.ne']
  rw [hlog, Real.exp_neg, Real.exp_log (one_div_pos.2 h0), one_div, inv_inv]
  ring

/-- `tau_x` rises with the discharge time ... -/
theorem tau_mono_tD {tD tD' eps : ℝ} (h0 : 0 < eps) (h1 : eps < 1) (h : tD ≤ tD') :
    capFor tD eps ≤ capFor tD' eps :=
  div_le_div_of_nonneg_right h (log_inv_pos h0 h1).le

/-- ... and with `epsilon`. -/
theorem tau_mono_eps {tD eps eps' : ℝ} (htD : 0 ≤ tD) (h0 : 0 < eps) (h : eps ≤ eps')
    (h1 : eps' < 1) : capFor tD eps ≤ capFor tD eps' := by
  unfold capFor
  apply div_le_div_of_nonneg_left htD (log_inv_pos (lt_of_lt_of_le h0 h) h1)
  exact Real.log_le_log (one_div_pos.2 (lt_of_lt_of_le h0 h)) (one_div_le_one_div_of_le h0 h)

/-- Both at once: a longer discharge or a looser `epsilon` never shortens `tau_x`. -/
theorem tau_mono {tD tD' eps eps' : ℝ} (htD : 0 ≤ tD) (h0 : 0 < eps) (hD : tD ≤ tD')
    (he : eps ≤ eps') (h1 : eps' < 1) : capFor tD eps ≤ capFor tD' eps' :=
  (tau_mono_eps htD h0 he h1).trans (tau_mono_tD (lt_of_lt_of_le h0 he) h1 hD)

/-! ## Item 10: the reached cells -/

/-- Each capped cell's loss in step `n` after `t_s` is at most its cap
`s_c exp(-(t - t_s)/tau) dt`, with its start `s_c` at most its rate `r_c(t_s)`: in all, at most
`Σ r_c (tau + dt)` however long the run. -/
theorem reached_energy_le {κ : Type*} {S : Finset κ} {r s : κ → ℝ} {loss : κ → ℕ → ℝ}
    {tS τ dt : ℝ} (hτ : 0 < τ) (hdt : 0 < dt) (hs0 : ∀ c ∈ S, 0 ≤ s c)
    (hsr : ∀ c ∈ S, s c ≤ r c)
    (hloss : ∀ c ∈ S, ∀ n : ℕ, loss c n ≤ Suppression.evansCap (s c) tS τ dt (tS + n * dt))
    (N : ℕ) : ∑ c ∈ S, ∑ n ∈ Finset.range N, loss c n ≤ ∑ c ∈ S, r c * (τ + dt) := by
  apply Finset.sum_le_sum
  intro c hc
  calc ∑ n ∈ Finset.range N, loss c n
      ≤ ∑ n ∈ Finset.range N, Suppression.evansCap (s c) tS τ dt (tS + n * dt) :=
        Finset.sum_le_sum fun n _ => hloss c hc n
    _ ≤ s c * (τ + dt) := Suppression.evansCap_sum (hs0 c hc) hτ hdt N
    _ ≤ r c * (τ + dt) := mul_le_mul_of_nonneg_right (hsr c hc) (by linarith)

/-- A cell already under a sprinkler's cap with `tau_S > tau` (else the code keeps that cap):
the new cap starts at `min(rate, envelope at t_s)` and decays faster, so it never lies above the
old at any later time. -/
theorem cap_never_rises {rate rS tAct τS τ tS t : ℝ} (hτ : 0 < τ) (hτS : τ < τS)
    (hrS : 0 ≤ rS) (ht : tS ≤ t) :
    Sprinklers.decay (min rate (Sprinklers.decay rS tAct τS tS)) tS τ t ≤
      Sprinklers.decay rS tAct τS t := by
  unfold Sprinklers.decay
  have hE : 0 ≤ rS * Real.exp (-((tS - tAct) / τS)) := mul_nonneg hrS (Real.exp_pos _).le
  have h1 : min rate (rS * Real.exp (-((tS - tAct) / τS))) * Real.exp (-((t - tS) / τ)) ≤
      rS * Real.exp (-((tS - tAct) / τS)) * Real.exp (-((t - tS) / τ)) :=
    mul_le_mul_of_nonneg_right (min_le_right _ _) (Real.exp_pos _).le
  have h2 : Real.exp (-((t - tS) / τ)) ≤ Real.exp (-((t - tS) / τS)) := by
    apply Real.exp_le_exp.2
    have := div_le_div_of_nonneg_left (sub_nonneg.2 ht) hτ hτS.le
    linarith
  have h3 : rS * Real.exp (-((tS - tAct) / τS)) * Real.exp (-((t - tS) / τS)) =
      rS * Real.exp (-((t - tAct) / τS)) := by
    rw [mul_assoc, ← Real.exp_add]
    congr 2
    ring
  calc _ ≤ _ := h1
    _ ≤ rS * Real.exp (-((tS - tAct) / τS)) * Real.exp (-((t - tS) / τS)) :=
        mul_le_mul_of_nonneg_left h2 hE
    _ = _ := h3

/-! ## Item 11: the cap is per cell, not a bound on the fire's peak -/

/-- A toy two-frame fire: gas `m` in a reached cell at `t_s`. In frame 0 the cell burns a share
`φ` of it (1 free, at most 1 under the cap); what is left moves to an unreached cell and burns
there in frame 1. The heat released in each frame, `H` per kilogram. -/
def toyFrames (H m φ : ℝ) : ℝ × ℝ := (H * φ * m, H * (1 - φ) * m)

/-- The cap halves the reached cell's burn, yet the capped fire releases more heat than the free
one in the later frame; the total is unchanged. -/
theorem cap_not_global {H m : ℝ} (hH : 0 < H) (hm : 0 < m) :
    (toyFrames H m (1 / 2)).1 ≤ (toyFrames H m 1).1 ∧
      (toyFrames H m 1).2 < (toyFrames H m (1 / 2)).2 ∧
      (toyFrames H m (1 / 2)).1 + (toyFrames H m (1 / 2)).2 =
        (toyFrames H m 1).1 + (toyFrames H m 1).2 := by
  simp only [toyFrames]
  refine ⟨by nlinarith, by nlinarith, by ring⟩

end FireEgress.Extinguishers
