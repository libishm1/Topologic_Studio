import Mathlib

/-!
# Jones's multi-agent Physarum model: the core rules

Jeff Jones's slime mould model (Jones 2010, doi:10.1162/artl.2010.16.2.16202;
Jones 2011, doi:10.1007/s11047-010-9223-z), as his example code implements it
and as Studio ports it (browser `src/core/physarum.js`, pseudocode in
`wiki/research/physarum-jones-models-2026-11-11.md`). Agents stand one to a
cell, step forward and lay trail, read three sensors and turn, and the trail
diffuses and decays. This file checks the four rules the slime mould egress
leans on:

1. Trail diffusion. On the wrap-around grid (a torus) one step multiplies the
   total trail by exactly `1 - d`, keeps it non-negative and keeps every cell
   under `(1 - d)` times the old maximum. Without wrap-around, where the
   border and the walls absorb, the total can only fall faster.
2. The sensory turn, exactly as the code branches, including a finding: on a
   tie between the two side sensors, both above the front one, the code
   always turns left where the 2010 paper turns at random.
3. Movement keeps one agent per cell and none on a wall, in any update order,
   and so a window of `k` cells never holds more than `k` agents.
4. Erosion lowers the substrate by the erosion amount, never below zero.
-/

namespace FireEgress.Physarum

open Finset

/-! ## 1. Trail diffusion -/

/-- The nine offsets of the 3x3 window, `(-1, 0, 1) × (-1, 0, 1)`, indexed by
`Fin 3 × Fin 3`. -/
def offset (o : Fin 3 × Fin 3) : ℤ × ℤ := (((o.1 : ℕ) : ℤ) - 1, ((o.2 : ℕ) : ℤ) - 1)

/-! ### On the torus (the code's wrap-around grid) -/

/-- The wrap-around grid of `w × h` cells: indices wrap modulo `w` and `h`.
The theorems take `[NeZero w] [NeZero h]`, that is `w, h ≥ 1`. -/
abbrev Torus (w h : ℕ) := ZMod w × ZMod h

/-- An integer offset read on the torus. -/
def toTorus (w h : ℕ) (v : ℤ × ℤ) : Torus w h := ((v.1 : ZMod w), (v.2 : ZMod h))

/-- The mean of the 3x3 window round `x` on the torus, divisor fixed at 9 as in
the code. On a torus narrower than 3 cells the window wraps onto itself and
counts a cell more than once, again as the code does. -/
noncomputable def mean3 {w h : ℕ} (T : Torus w h → ℝ) (x : Torus w h) : ℝ :=
  (∑ o : Fin 3 × Fin 3, T (x + toTorus w h (offset o))) / 9

/-- One diffusion step on the torus: every cell becomes `(1 - d)` times the
mean of its 3x3 window. -/
noncomputable def diffuse {w h : ℕ} (d : ℝ) (T : Torus w h → ℝ) (x : Torus w h) : ℝ :=
  (1 - d) * mean3 T x

section Torus

variable {w h : ℕ} [NeZero w] [NeZero h]

/-- Mass is conserved up to damping: on the wrap-around grid one step leaves
exactly `(1 - d)` times the total trail. Each cell's value is read by exactly
nine windows, so nothing is created or lost by the averaging itself. Holds for
every `d`; no bound on `d` is needed. -/
theorem diffuse_total (d : ℝ) (T : Torus w h → ℝ) :
    ∑ x, diffuse d T x = (1 - d) * ∑ x, T x := by
  have hshift : ∀ o : Fin 3 × Fin 3,
      ∑ x : Torus w h, T (x + toTorus w h (offset o)) = ∑ x, T x :=
    fun o => Equiv.sum_comp (Equiv.addRight (toTorus w h (offset o))) T
  have hswap : ∑ x : Torus w h, ∑ o : Fin 3 × Fin 3, T (x + toTorus w h (offset o))
      = 9 * ∑ x, T x := by
    rw [Finset.sum_comm]
    simp only [hshift, Finset.sum_const, Finset.card_univ, Fintype.card_prod,
      Fintype.card_fin, nsmul_eq_mul]
    norm_num
  simp only [diffuse, mean3]
  rw [← Finset.mul_sum, ← Finset.sum_div, hswap]
  ring

end Torus

/-- Trail never goes negative: with `d ≤ 1`, a non-negative trail stays
non-negative after a step. -/
theorem diffuse_nonneg {w h : ℕ} {d : ℝ} (hd : d ≤ 1) {T : Torus w h → ℝ} (hT : ∀ x, 0 ≤ T x)
    (x : Torus w h) : 0 ≤ diffuse d T x :=
  mul_nonneg (by linarith) (div_nonneg (Finset.sum_nonneg fun o _ => hT _) (by norm_num))

/-- Discrete maximum principle: with `d ≤ 1`, if every cell is at most `M`
then every cell after a step is at most `(1 - d) M`, so the trail never
overshoots and decays geometrically. (`M ≥ 0` is not needed.) -/
theorem diffuse_le {w h : ℕ} {d : ℝ} (hd : d ≤ 1) {T : Torus w h → ℝ} {M : ℝ} (hT : ∀ x, T x ≤ M)
    (x : Torus w h) : diffuse d T x ≤ (1 - d) * M := by
  have hsum : ∑ o : Fin 3 × Fin 3, T (x + toTorus w h (offset o)) ≤ 9 * M := by
    calc ∑ o : Fin 3 × Fin 3, T (x + toTorus w h (offset o))
        ≤ ∑ _o : Fin 3 × Fin 3, M := Finset.sum_le_sum fun o _ => hT _
      _ = 9 * M := by
        simp only [Finset.sum_const, Finset.card_univ, Fintype.card_prod, Fintype.card_fin,
          nsmul_eq_mul]
        norm_num
  have hmean : mean3 T x ≤ M := by
    unfold mean3
    rw [div_le_iff₀ (by norm_num)]
    linarith
  exact mul_le_mul_of_nonneg_left hmean (by linarith)

/-! ### Without wrap-around: zero padding and absorbing cells -/

/-- Zero padding: the trail on the grid's cells `G`, and 0 past the grid. -/
def pad (G : Finset (ℤ × ℤ)) (T : ℤ × ℤ → ℝ) (y : ℤ × ℤ) : ℝ :=
  if y ∈ G then T y else 0

/-- The 3x3 mean with the cells past the grid counting as zero (divisor 9). -/
noncomputable def meanPad (G : Finset (ℤ × ℤ)) (T : ℤ × ℤ → ℝ) (x : ℤ × ℤ) : ℝ :=
  (∑ o : Fin 3 × Fin 3, pad G T (x + offset o)) / 9

/-- One diffusion step without wrap-around: an absorbing cell ends at 0, any
other cell becomes `(1 - d)` times its zero-padded 3x3 mean.

Modelling choice: the grid is any finite set `G` of integer cells (a rectangle,
or any floor shape) and `absorbing` is any predicate. Jones's code takes `G`
the `w × h` rectangle and `absorbing` the border cells or the walls; Studio's
`diffuse` in `src/core/physarum.js` takes `absorbing` the walls only and lets
the zero padding absorb at the border. One theorem covers both. -/
noncomputable def diffuseAbs (G : Finset (ℤ × ℤ)) (absorbing : ℤ × ℤ → Prop)
    [DecidablePred absorbing] (d : ℝ) (T : ℤ × ℤ → ℝ) (x : ℤ × ℤ) : ℝ :=
  if absorbing x then 0 else (1 - d) * meanPad G T x

/-- Shifting the window never reads more trail than the grid holds: the cells
`x + c` for `x` in the grid are distinct, and those past the grid read 0. -/
theorem sum_pad_shift_le (G : Finset (ℤ × ℤ)) (T : ℤ × ℤ → ℝ) (hT : ∀ y ∈ G, 0 ≤ T y)
    (c : ℤ × ℤ) : ∑ x ∈ G, pad G T (x + c) ≤ ∑ y ∈ G, T y := by
  have h1 : ∑ x ∈ G, pad G T (x + c) = ∑ x ∈ G.filter (fun x => x + c ∈ G), T (x + c) := by
    rw [Finset.sum_filter]
    rfl
  have h2 : ∑ x ∈ G.filter (fun x => x + c ∈ G), T (x + c)
      = ∑ y ∈ (G.filter (fun x => x + c ∈ G)).map (Equiv.addRight c).toEmbedding, T y := by
    rw [Finset.sum_map]
    rfl
  rw [h1, h2]
  apply Finset.sum_le_sum_of_subset_of_nonneg
  · intro y hy
    simp only [Finset.mem_map, Finset.mem_filter, Equiv.coe_toEmbedding,
      Equiv.coe_addRight] at hy
    obtain ⟨x, ⟨_, hx⟩, rfl⟩ := hy
    exact hx
  · intro y hy _
    exact hT y hy

/-- Without wrap-around the border and the walls only absorb: for a
non-negative trail and `d ≤ 1`, the total over the grid after one step is at
most `(1 - d)` times the total before. Trail can leave through the border or
into a wall but is never created, whatever the grid's shape. -/
theorem diffuse_absorbing_total_le (G : Finset (ℤ × ℤ)) (absorbing : ℤ × ℤ → Prop)
    [DecidablePred absorbing] {d : ℝ} (hd : d ≤ 1) (T : ℤ × ℤ → ℝ)
    (hT : ∀ y ∈ G, 0 ≤ T y) :
    ∑ x ∈ G, diffuseAbs G absorbing d T x ≤ (1 - d) * ∑ x ∈ G, T x := by
  have hpad : ∀ y, 0 ≤ pad G T y := by
    intro y
    unfold pad
    split_ifs with hy
    · exact hT y hy
    · exact le_refl 0
  have hm : ∀ x, 0 ≤ (1 - d) * meanPad G T x := fun x =>
    mul_nonneg (by linarith) (div_nonneg (Finset.sum_nonneg fun o _ => hpad _) (by norm_num))
  have step1 : ∑ x ∈ G, diffuseAbs G absorbing d T x ≤ ∑ x ∈ G, (1 - d) * meanPad G T x := by
    apply Finset.sum_le_sum
    intro x _
    unfold diffuseAbs
    split_ifs
    · exact hm x
    · exact le_refl _
  have step2 : ∑ x ∈ G, (1 - d) * meanPad G T x
      = (1 - d) / 9 * ∑ o : Fin 3 × Fin 3, ∑ x ∈ G, pad G T (x + offset o) := by
    have hx : ∀ x, (1 - d) * meanPad G T x
        = (1 - d) / 9 * ∑ o : Fin 3 × Fin 3, pad G T (x + offset o) := fun x => by
      unfold meanPad
      ring
    simp only [hx]
    rw [← Finset.mul_sum, Finset.sum_comm]
  have step3 : ∑ o : Fin 3 × Fin 3, ∑ x ∈ G, pad G T (x + offset o) ≤ 9 * ∑ x ∈ G, T x := by
    calc ∑ o : Fin 3 × Fin 3, ∑ x ∈ G, pad G T (x + offset o)
        ≤ ∑ _o : Fin 3 × Fin 3, ∑ x ∈ G, T x :=
          Finset.sum_le_sum fun o _ => sum_pad_shift_le G T hT (offset o)
      _ = 9 * ∑ x ∈ G, T x := by
        simp only [Finset.sum_const, Finset.card_univ, Fintype.card_prod, Fintype.card_fin,
          nsmul_eq_mul]
        norm_num
  have hk : 0 ≤ (1 - d) / 9 := div_nonneg (by linarith) (by norm_num)
  calc ∑ x ∈ G, diffuseAbs G absorbing d T x
      ≤ ∑ x ∈ G, (1 - d) * meanPad G T x := step1
    _ = (1 - d) / 9 * ∑ o : Fin 3 × Fin 3, ∑ x ∈ G, pad G T (x + offset o) := step2
    _ ≤ (1 - d) / 9 * (9 * ∑ x ∈ G, T x) := mul_le_mul_of_nonneg_left step3 hk
    _ = (1 - d) * ∑ x ∈ G, T x := by ring

/-- The `w × h` rectangle of cells `0 ≤ x < w`, `0 ≤ y < h`. -/
def rect (w h : ℕ) : Finset (ℤ × ℤ) := Finset.Ico 0 (w : ℤ) ×ˢ Finset.Ico 0 (h : ℤ)

/-- The rectangle's border cells, which Jones's non-wrap code sets to 0. -/
def onBorder (w h : ℕ) (x : ℤ × ℤ) : Prop :=
  x.1 = 0 ∨ x.1 = (w : ℤ) - 1 ∨ x.2 = 0 ∨ x.2 = (h : ℤ) - 1

instance (w h : ℕ) : DecidablePred (onBorder w h) := fun x => by
  unfold onBorder
  infer_instance

/-- Why the zero padding is faithful to Jones's code: in the rectangle, the
3x3 window of every cell off the border lies inside the grid, so the cells the
code actually averages never read past the border, and the border cells are
zeroed anyway. How out-of-grid reads are treated therefore does not change
Jones's step, which is `diffuseAbs (rect w h) (onBorder w h ∨ wall)`. -/
theorem interior_window_in_grid {w h : ℕ} {x : ℤ × ℤ} (hx : x ∈ rect w h)
    (hb : ¬ onBorder w h x) (o : Fin 3 × Fin 3) : x + offset o ∈ rect w h := by
  have h1 := o.1.isLt
  have h2 := o.2.isLt
  simp only [rect, onBorder, offset, Finset.mem_product, Finset.mem_Ico, Prod.fst_add,
    Prod.snd_add, not_or] at hx hb ⊢
  omega

/-! ## 2. The sensory turn -/

/-- The turn Jones's code takes from the three sensor readings `f` (front),
`fl` (front-left) and `fr` (front-right): 0 keeps the heading, `1` turns
towards `fr`, `-1` towards `fl`.

- front strictly highest: keep heading;
- front strictly lowest: turn towards the larger side, and left on a tie;
- otherwise: turn towards the strictly larger side, or keep heading on a tie. -/
noncomputable def turn (f fl fr : ℝ) : ℤ :=
  if f > fl ∧ f > fr then 0
  else if f < fl ∧ f < fr then (if fl < fr then 1 else -1)
  else if fl < fr then 1
  else if fr < fl then -1
  else 0

/-- An agent whose front sensor reads strictly more than both side sensors
keeps its heading. -/
theorem turn_keep_front {f fl fr : ℝ} (h : f > fl ∧ f > fr) : turn f fl fr = 0 := by
  simp [turn, h]

/-- An agent turns right (towards `fr`) exactly when the front is not
strictly highest and the right reading is strictly above the left. -/
theorem turn_pos_iff (f fl fr : ℝ) : turn f fl fr = 1 ↔ ¬(f > fl ∧ f > fr) ∧ fl < fr := by
  unfold turn
  split_ifs with h1 h2 h3 h4 h5 <;> simp_all

/-- The turn never goes away from a strictly larger side: a right turn means
the right reading is strictly larger, and a left turn means the left reading
is at least the right one. -/
theorem turn_toward_larger (f fl fr : ℝ) :
    (turn f fl fr = 1 → fl < fr) ∧ (turn f fl fr = -1 → fr ≤ fl) := by
  unfold turn
  split_ifs with h1 h2 h3 h4 h5 <;> simp_all

/-- Finding: on a tie of the two side sensors, both strictly above the front,
the code always turns left (`-1`). Jones's 2010 paper
(doi:10.1162/artl.2010.16.2.16202) describes a random choice of side there.
Studio's port keeps the same branch (`src/core/physarum.js`, the
`F < FL && F < FR` case). The bias needs an exact tie with both sides strictly
above the front, as a mirror-symmetric trail field gives an agent heading
along its axis into a dip; a blank field (`f = fl = fr`) keeps the heading. -/
theorem turn_tie_left {f fl fr : ℝ} (hf : f < fl) (htie : fl = fr) : turn f fl fr = -1 := by
  subst htie
  simp [turn, hf, hf.le]

/-- The turn is always one of `-1`, `0`, `1`. -/
theorem turn_range (f fl fr : ℝ) : turn f fl fr ∈ ({-1, 0, 1} : Set ℤ) := by
  unfold turn
  split_ifs <;> simp

/-! ## 3. Movement: one agent per cell, none on a wall -/

section Movement

variable {C : Type*} [DecidableEq C] {n : ℕ} (wall : C → Prop) [DecidablePred wall]

/-- A placement of the agents is valid when no two agents share a cell and no
agent stands on a wall. -/
def Valid (pos : Fin n → C) : Prop := Function.Injective pos ∧ ∀ i, ¬ wall (pos i)

/-- Agent `i` tries cell `c`: it moves iff `c` is not a wall and no other agent
stands on `c`; otherwise it stays (in the code it then picks a new random
heading, which does not change any position). -/
def moveOne (pos : Fin n → C) (i : Fin n) (c : C) : Fin n → C :=
  if ¬ wall c ∧ ∀ j, j ≠ i → pos j ≠ c then Function.update pos i c else pos

/-- A whole movement phase: the attempts in a list of `(agent, target)` pairs
applied one after another. Any list is allowed, so this covers the code's
shuffled update order, an agent missing a turn, or one trying twice. -/
def moveAll (pos : Fin n → C) : List (Fin n × C) → Fin n → C
  | [] => pos
  | m :: ms => moveAll (moveOne wall pos m.1 m.2) ms

/-- Respawn (teleport): agent `i` is put on cell `c`. -/
def respawn (pos : Fin n → C) (i : Fin n) (c : C) : Fin n → C := Function.update pos i c

variable {wall}

omit [DecidableEq C] [DecidablePred wall] in
/-- Putting agent `i` on a cell that is no wall and holds no other agent keeps
the placement valid. -/
theorem update_valid {pos : Fin n → C} (hv : Valid wall pos) {i : Fin n} {c : C}
    (hc : ¬ wall c) (hfree : ∀ j, j ≠ i → pos j ≠ c) :
    Valid wall (Function.update pos i c) := by
  obtain ⟨hinj, hw⟩ := hv
  refine ⟨?_, ?_⟩
  · intro a b hab
    by_cases ha : a = i <;> by_cases hb : b = i
    · rw [ha, hb]
    · subst ha
      rw [Function.update_self, Function.update_of_ne hb] at hab
      exact absurd hab.symm (hfree b hb)
    · subst hb
      rw [Function.update_self, Function.update_of_ne ha] at hab
      exact absurd hab (hfree a ha)
    · rw [Function.update_of_ne ha, Function.update_of_ne hb] at hab
      exact hinj hab
  · intro j
    by_cases hj : j = i
    · subst hj
      rw [Function.update_self]
      exact hc
    · rw [Function.update_of_ne hj]
      exact hw j

/-- One movement attempt keeps the placement valid, whether it succeeds or
not. -/
theorem moveOne_valid {pos : Fin n → C} (hv : Valid wall pos) (i : Fin n) (c : C) :
    Valid wall (moveOne wall pos i c) := by
  unfold moveOne
  split_ifs with h
  · exact update_valid hv h.1 h.2
  · exact hv

/-- A whole movement phase, in any order, keeps one agent per cell and none on
a wall: if the placement is valid before, it is valid after, for every list
of attempts. -/
theorem moveAll_injective {pos : Fin n → C} (hv : Valid wall pos) (ms : List (Fin n × C)) :
    Valid wall (moveAll wall pos ms) := by
  induction ms generalizing pos with
  | nil => exact hv
  | cons m ms ih => exact ih (moveOne_valid hv m.1 m.2)

omit [DecidableEq C] [DecidablePred wall] in
/-- Respawn onto a free cell that is no wall keeps the placement valid. ("Free"
need only exclude the other agents; the agent's own cell is allowed.) -/
theorem respawn_preserves {pos : Fin n → C} (hv : Valid wall pos) {i : Fin n} {c : C}
    (hc : ¬ wall c) (hfree : ∀ j, j ≠ i → pos j ≠ c) : Valid wall (respawn pos i c) :=
  update_valid hv hc hfree

end Movement

/-- With one agent per cell, any set `W` of cells holds at most `W.card`
agents. So in Jones's shrinkage test, whose window is 5x5 = 25 cells, a
survival limit of 23 agents can be exceeded only by a count of 24 or 25
(`shrink_limit_exceeded`); a limit of 25 or more could never be exceeded. -/
theorem window_count_le {C : Type*} [DecidableEq C] {n : ℕ} (pos : Fin n → C) (hinj : Function.Injective pos)
    (W : Finset C) : (Finset.univ.filter (fun i => pos i ∈ W)).card ≤ W.card := by
  apply Finset.card_le_card_of_injOn pos
  · intro i hi
    exact (Finset.mem_filter.1 (Finset.mem_coe.1 hi)).2
  · exact hinj.injOn

/-- Jones's shrinkage window: 25 cells, a limit of 23; the count exceeds the
limit only as 24 or 25. -/
theorem shrink_limit_exceeded {C : Type*} [DecidableEq C] {n : ℕ} (pos : Fin n → C)
    (hinj : Function.Injective pos) (W : Finset C) (hW : W.card = 25)
    (h : 23 < (Finset.univ.filter (fun i => pos i ∈ W)).card) :
    (Finset.univ.filter (fun i => pos i ∈ W)).card = 24 ∨
      (Finset.univ.filter (fun i => pos i ∈ W)).card = 25 := by
  have := window_count_le pos hinj W
  omega

/-! ## 4. Erosion -/

/-- The erosion variant: a successful step wears `e` off the substrate `s`
under the agent, never below zero. -/
noncomputable def erode (s e : ℝ) : ℝ := max 0 (s - e)

/-- Eroded substrate is never negative. -/
theorem erode_nonneg (s e : ℝ) : 0 ≤ erode s e := le_max_left _ _

/-- Erosion never raises the substrate: for `s ≥ 0` and `e ≥ 0`,
`erode s e ≤ s`. Both hypotheses are needed: `erode (-1) 0 = 0 > -1` and
`erode 1 (-1) = 2 > 1`. -/
theorem erode_le {s e : ℝ} (hs : 0 ≤ s) (he : 0 ≤ e) : erode s e ≤ s :=
  max_le hs (by linarith)

/-- Where there is at least `e` of substrate, a step wears off exactly `e`. -/
theorem erode_exact {s e : ℝ} (h : e ≤ s) : s - erode s e = e := by
  unfold erode
  rw [max_eq_right (by linarith)]
  ring

/-- More substrate before, at least as much after: erosion keeps the order of
cells (monotone in `s`). -/
theorem erode_mono (e : ℝ) : Monotone (fun s => erode s e) := fun a b hab =>
  max_le_max le_rfl (by linarith)

/-- A harder wear leaves no more substrate: erosion is antitone in `e`. -/
theorem erode_antitone (s : ℝ) : Antitone (fun e => erode s e) := fun a b hab =>
  max_le_max le_rfl (by linarith)

/-- Erosion is not antitone in the substrate `s`, for any wear `e`:
`erode e e = 0` but `erode (e + 1) e = 1`. (It is monotone, `erode_mono`.) -/
theorem erode_not_antitone_in_s (e : ℝ) : ¬ Antitone (fun s => erode s e) := by
  intro h
  have h1 := h (show e ≤ e + 1 by linarith)
  simp only [erode, sub_self, max_self, add_sub_cancel_left] at h1
  have h2 : (1 : ℝ) ≤ max 0 1 := le_max_right _ _
  linarith

end FireEgress.Physarum
