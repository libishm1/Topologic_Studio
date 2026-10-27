import Mathlib
import FireEgress.Dim

/-!
# Graph numerics behind Studio's fire and routing (L1, L2, L10a, L10b, L11, L7)

The literature review (`wiki/research/fire-egress-methods-2026-10-27.md`, section 3.5
and sections 5.1-5.3) proposes a physical graph heat equation to replace Classic's
"temperature" step, and time-dependent earliest-arrival routing to replace snapshot
rerouting. This file proves the parts of that argument Studio would rely on.

* **Graph heat equation (G1-G4).** `C θ' = -(L_g + H) θ + q` with `θ = T - T_a`.
  Explicit Euler with `q = 0` is `θⁿ⁺¹ = M θⁿ`, `M = I - Δt C⁻¹ (L_g + H)`.
  L1 (Lemma A): `M ≥ 0` iff `Δt (G_i + H_i) ≤ C_i` for every node, and the row sums
  are `1 - Δt H_i / C_i ≤ 1`. L2 (Theorem B): the discrete maximum principle for every
  `n`, including the corrected Dirichlet clause, and a Lean counterexample to the
  note's uncorrected Dirichlet clause.
  Units: `C` is in J K⁻¹, `g` and `H` in W K⁻¹, `Δt` in s. The real-number theorems
  are unit-free, so they hold in any consistent units; the section `Units` checks
  with `FireEgress.Dim` that `Δt (G + H) / C` is dimensionless, so the step condition
  is a statement about seconds and not a fitted constant.
* **Time-dependent routing (R1-R4, R6, R7).** Arrival functions `A_e : α → α` on an
  arbitrary preorder of times (in Studio `WithTop ℝ`, with `⊤` = edge closed by fire).
  FIFO means `A_e` is monotone. L10a: FIFO arrival functions compose to FIFO.
  L10b: inserting a wait never gives an earlier arrival. L11: a suffix of an
  earliest-arrival walk is earliest-arrival from where and when the walk reached it
  (no FIFO needed). L7: the layered BFS in `rl.js`/`rl.py` `earliestEscape` puts each
  node in the layer equal to the minimum length of a safe walk to it.
-/

namespace FireEgress.GraphNumerics

open Finset Matrix

/-! ## Explicit Euler on the graph heat equation -/

section Euler

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- Total conductance at node `i`: `G_i = ∑_j g_ij` (W K⁻¹). -/
def degW (g : ι → ι → ℝ) (i : ι) : ℝ := ∑ j, g i j

/-- Weighted graph Laplacian `L_g = diag(G) - [g_ij]` (W K⁻¹). Mathlib's `lapMatrix`
is unweighted, so it is defined here. -/
def lap (g : ι → ι → ℝ) : Matrix ι ι ℝ := Matrix.diagonal (degW g) - Matrix.of g

/-- G2. The explicit-Euler iteration matrix `M = I - Δt C⁻¹ (L_g + H)` of
`C θ' = -(L_g + H) θ` (with `q = 0`). `C i` is the heat capacity of node `i`
(J K⁻¹), `H i` its loss to ambient (W K⁻¹), `g i j` the conductance (W K⁻¹), `dt`
the time step (s). -/
noncomputable def eulerM (C H : ι → ℝ) (g : ι → ι → ℝ) (dt : ℝ) : Matrix ι ι ℝ :=
  1 - dt • (Matrix.diagonal (fun i => (C i)⁻¹) * (lap g + Matrix.diagonal H))

/-- G2, entrywise. With no self-loops (`g i i = 0`):
`M_ii = 1 - Δt (G_i + H_i) / C_i` and `M_ij = Δt g_ij / C_i` for `j ≠ i`.
Without `g i i = 0` the diagonal would be `1 - Δt (G_i - g_ii + H_i) / C_i`. -/
theorem eulerM_apply (C H : ι → ℝ) (g : ι → ι → ℝ) (dt : ℝ) (hgii : ∀ i, g i i = 0)
    (i j : ι) :
    eulerM C H g dt i j =
      if i = j then 1 - dt * (degW g i + H i) / C i else dt * g i j / C i := by
  simp only [eulerM, lap, Matrix.sub_apply, Matrix.one_apply, Matrix.smul_apply, smul_eq_mul,
    Matrix.diagonal_mul, Matrix.add_apply, Matrix.diagonal_apply, Matrix.of_apply]
  split_ifs with h
  · subst h; rw [hgii]; ring
  · ring

/-- G2. Row sums of the explicit-Euler matrix: `∑_j M_ij = 1 - Δt H_i / C_i`
(needs `g i i = 0`). -/
theorem eulerM_row_sum (C H : ι → ℝ) (g : ι → ι → ℝ) (dt : ℝ) (hgii : ∀ i, g i i = 0)
    (i : ι) : ∑ j, eulerM C H g dt i j = 1 - dt * H i / C i := by
  have hsplit : ∀ j, eulerM C H g dt i j =
      (if i = j then 1 - dt * (degW g i + H i) / C i else 0) + dt * g i j / C i := by
    intro j
    rw [eulerM_apply C H g dt hgii]
    split_ifs with h
    · subst h; rw [hgii]; ring
    · ring
  rw [Finset.sum_congr rfl (fun j _ => hsplit j), Finset.sum_add_distrib, Finset.sum_ite_eq]
  simp only [Finset.mem_univ, ite_true]
  rw [← Finset.sum_div, ← Finset.mul_sum]
  unfold degW
  ring

/-- L1 / Lemma A, the "iff". For `Δt ≥ 0`, positive capacities, non-negative
conductances and no self-loops, the explicit-Euler matrix is entrywise non-negative
exactly when `Δt (G_i + H_i) ≤ C_i` at every node. This is the division-free form of
`Δt ≤ min_i C_i / (G_i + H_i)` with `C_i / 0 = +∞`. Symmetry of `g` and `H ≥ 0` are
not used. -/
theorem L1_nonneg_iff (C H : ι → ℝ) (g : ι → ι → ℝ) (dt : ℝ) (hC : ∀ i, 0 < C i)
    (hg : ∀ i j, 0 ≤ g i j) (hgii : ∀ i, g i i = 0) (hdt : 0 ≤ dt) :
    (∀ i j, 0 ≤ eulerM C H g dt i j) ↔ ∀ i, dt * (degW g i + H i) ≤ C i := by
  constructor
  · intro h i
    have hii := h i i
    rw [eulerM_apply C H g dt hgii] at hii
    simp only [↓reduceIte] at hii
    have h2 : dt * (degW g i + H i) / C i ≤ 1 := by linarith
    rwa [div_le_one (hC i)] at h2
  · intro h i j
    rw [eulerM_apply C H g dt hgii]
    split_ifs with hij
    · have : dt * (degW g i + H i) / C i ≤ 1 := (div_le_one (hC i)).mpr (h i)
      linarith
    · exact div_nonneg (mul_nonneg hdt (hg i j)) (hC i).le

/-- L1 / Lemma A. Under the step condition `Δt (G_i + H_i) ≤ C_i` (with `Δt ≥ 0`,
`C > 0`, `g ≥ 0`, `g_ii = 0`, `H ≥ 0`) every entry of `M` is non-negative, every row
sums to `1 - Δt H_i / C_i`, and so every row sums to at most 1. The bound `≤ 1` is
where `H ≥ 0` is used. -/
theorem L1_lemmaA (C H : ι → ℝ) (g : ι → ι → ℝ) (dt : ℝ) (hC : ∀ i, 0 < C i)
    (hg : ∀ i j, 0 ≤ g i j) (hgii : ∀ i, g i i = 0) (hH : ∀ i, 0 ≤ H i) (hdt : 0 ≤ dt)
    (hA : ∀ i, dt * (degW g i + H i) ≤ C i) :
    (∀ i j, 0 ≤ eulerM C H g dt i j) ∧
      (∀ i, ∑ j, eulerM C H g dt i j = 1 - dt * H i / C i) ∧
      (∀ i, ∑ j, eulerM C H g dt i j ≤ 1) := by
  refine ⟨(L1_nonneg_iff C H g dt hC hg hgii hdt).2 hA, eulerM_row_sum C H g dt hgii,
    fun i => ?_⟩
  rw [eulerM_row_sum C H g dt hgii]
  have : 0 ≤ dt * H i / C i := div_nonneg (mul_nonneg hdt (hH i)) (hC i).le
  linarith

/-- The explicit iteration `θⁿ⁺¹ = M θⁿ` (source term `q = 0`). -/
noncomputable def iter (M : Matrix ι ι ℝ) (θ0 : ι → ℝ) : ℕ → ι → ℝ
  | 0 => θ0
  | n + 1 => M *ᵥ iter M θ0 n

/-- `iter` is the matrix power: `θⁿ = Mⁿ θ⁰`. -/
theorem iter_eq_pow (M : Matrix ι ι ℝ) (θ0 : ι → ℝ) (n : ℕ) :
    iter M θ0 n = (M ^ n) *ᵥ θ0 := by
  induction n with
  | zero => simp [iter]
  | succ n ih => rw [iter, ih, pow_succ', Matrix.mulVec_mulVec]

/-- The iteration with Dirichlet nodes: nodes in `D` are held at `θf` (Studio's
source reset), the others take one explicit step. -/
noncomputable def iterD (M : Matrix ι ι ℝ) (D : ι → Prop) [DecidablePred D] (θf : ℝ)
    (θ0 : ι → ℝ) : ℕ → ι → ℝ
  | 0 => θ0
  | n + 1 => fun i => if D i then θf else (M *ᵥ iterD M D θf θ0 n) i

omit [DecidableEq ι] in
/-- One step of a non-negative matrix maps values in `[lo, hi]` into
`[lo · rowsum, hi · rowsum]`. -/
theorem mulVec_bounds (M : Matrix ι ι ℝ) (hM : ∀ i j, 0 ≤ M i j) (v : ι → ℝ) (lo hi : ℝ)
    (hv : ∀ j, lo ≤ v j ∧ v j ≤ hi) (i : ι) :
    lo * ∑ j, M i j ≤ (M *ᵥ v) i ∧ (M *ᵥ v) i ≤ hi * ∑ j, M i j := by
  simp only [Matrix.mulVec, dotProduct]
  constructor
  · rw [Finset.mul_sum]
    exact Finset.sum_le_sum fun j _ => by
      nlinarith [mul_nonneg (hM i j) (sub_nonneg.2 (hv j).1)]
  · rw [Finset.mul_sum]
    exact Finset.sum_le_sum fun j _ => by
      nlinarith [mul_nonneg (hM i j) (sub_nonneg.2 (hv j).2)]

omit [DecidableEq ι] in
/-- One step of a non-negative, row-substochastic matrix keeps values in `[lo, hi]`
when `lo ≤ 0 ≤ hi`. -/
theorem substochastic_step (M : Matrix ι ι ℝ) (hM : ∀ i j, 0 ≤ M i j)
    (hrow : ∀ i, ∑ j, M i j ≤ 1) (v : ι → ℝ) (lo hi : ℝ) (hlo : lo ≤ 0) (hhi : 0 ≤ hi)
    (hv : ∀ j, lo ≤ v j ∧ v j ≤ hi) (i : ι) :
    lo ≤ (M *ᵥ v) i ∧ (M *ᵥ v) i ≤ hi := by
  obtain ⟨h1, h2⟩ := mulVec_bounds M hM v lo hi hv i
  have hr := hrow i
  have a1 : 0 ≤ (-lo) * (1 - ∑ j, M i j) := mul_nonneg (neg_nonneg.2 hlo) (sub_nonneg.2 hr)
  have a2 : 0 ≤ hi * (1 - ∑ j, M i j) := mul_nonneg hhi (sub_nonneg.2 hr)
  constructor <;> nlinarith

omit [DecidableEq ι] in
/-- L2, abstract form. If `M ≥ 0` entrywise with row sums `≤ 1`, and `lo ≤ 0 ≤ hi`
bound the initial data, then every iterate `Mⁿ θ⁰` stays in `[lo, hi]`, for all `n`. -/
theorem dmp_substochastic (M : Matrix ι ι ℝ) (hM : ∀ i j, 0 ≤ M i j)
    (hrow : ∀ i, ∑ j, M i j ≤ 1) (θ0 : ι → ℝ) (lo hi : ℝ) (hlo : lo ≤ 0) (hhi : 0 ≤ hi)
    (h0 : ∀ j, lo ≤ θ0 j ∧ θ0 j ≤ hi) (n : ℕ) :
    ∀ i, lo ≤ iter M θ0 n i ∧ iter M θ0 n i ≤ hi := by
  induction n with
  | zero => exact h0
  | succ n ih => exact substochastic_step M hM hrow _ lo hi hlo hhi ih

omit [DecidableEq ι] in
/-- L2, lossless abstract form. If `M ≥ 0` with row sums exactly 1, any bounds
`[lo, hi]` on the initial data hold for all iterates (no `0` needed). -/
theorem dmp_stochastic (M : Matrix ι ι ℝ) (hM : ∀ i j, 0 ≤ M i j)
    (hrow : ∀ i, ∑ j, M i j = 1) (θ0 : ι → ℝ) (lo hi : ℝ)
    (h0 : ∀ j, lo ≤ θ0 j ∧ θ0 j ≤ hi) (n : ℕ) :
    ∀ i, lo ≤ iter M θ0 n i ∧ iter M θ0 n i ≤ hi := by
  induction n with
  | zero => exact h0
  | succ n ih =>
    intro i
    obtain ⟨h1, h2⟩ := mulVec_bounds M hM _ lo hi ih i
    rw [hrow i, mul_one] at h1 h2
    exact ⟨h1, h2⟩

omit [DecidableEq ι] in
/-- L2 with Dirichlet nodes, abstract form: nodes in `D` held at `θf`. The bounds must
contain `0` *and* `θf`. -/
theorem dmp_dirichlet (M : Matrix ι ι ℝ) (hM : ∀ i j, 0 ≤ M i j)
    (hrow : ∀ i, ∑ j, M i j ≤ 1) (D : ι → Prop) [DecidablePred D] (θf : ℝ) (θ0 : ι → ℝ)
    (lo hi : ℝ) (hlo : lo ≤ 0) (hhi : 0 ≤ hi) (hflo : lo ≤ θf) (hfhi : θf ≤ hi)
    (h0 : ∀ j, lo ≤ θ0 j ∧ θ0 j ≤ hi) (n : ℕ) :
    ∀ i, lo ≤ iterD M D θf θ0 n i ∧ iterD M D θf θ0 n i ≤ hi := by
  induction n with
  | zero => exact h0
  | succ n ih =>
    intro i
    simp only [iterD]
    split_ifs
    · exact ⟨hflo, hfhi⟩
    · exact substochastic_step M hM hrow _ lo hi hlo hhi ih i

/-- L2 / Theorem B for the graph heat equation. Under Lemma A (`Δt ≥ 0`, `C > 0`,
`g ≥ 0`, `g_ii = 0`, `H ≥ 0`, `Δt (G_i + H_i) ≤ C_i`) and `q = 0`, for every step `n`
and node `i`:
`min(0, min_j θ⁰_j) ≤ θⁿ_i ≤ max(0, max_j θ⁰_j)`.
A physical graph diffusion stepped this way never overshoots. -/
theorem L2_dmp [Nonempty ι] (C H : ι → ℝ) (g : ι → ι → ℝ) (dt : ℝ) (hC : ∀ i, 0 < C i)
    (hg : ∀ i j, 0 ≤ g i j) (hgii : ∀ i, g i i = 0) (hH : ∀ i, 0 ≤ H i) (hdt : 0 ≤ dt)
    (hA : ∀ i, dt * (degW g i + H i) ≤ C i) (θ0 : ι → ℝ) (n : ℕ) (i : ι) :
    min 0 (Finset.univ.inf' Finset.univ_nonempty θ0) ≤ iter (eulerM C H g dt) θ0 n i ∧
      iter (eulerM C H g dt) θ0 n i ≤ max 0 (Finset.univ.sup' Finset.univ_nonempty θ0) := by
  obtain ⟨hM, -, hrow⟩ := L1_lemmaA C H g dt hC hg hgii hH hdt hA
  refine dmp_substochastic _ hM hrow θ0 _ _ (min_le_left _ _) (le_max_left _ _) ?_ n i
  intro j
  exact ⟨(min_le_right _ _).trans (Finset.inf'_le _ (Finset.mem_univ j)),
    (Finset.le_sup' _ (Finset.mem_univ j)).trans (le_max_right _ _)⟩

/-- L2 / Theorem B, lossless case (`H = 0`): the `0` drops out,
`min_j θ⁰_j ≤ θⁿ_i ≤ max_j θ⁰_j` for every `n`. -/
theorem L2_dmp_lossless [Nonempty ι] (C : ι → ℝ) (g : ι → ι → ℝ) (dt : ℝ)
    (hC : ∀ i, 0 < C i) (hg : ∀ i j, 0 ≤ g i j) (hgii : ∀ i, g i i = 0) (hdt : 0 ≤ dt)
    (hA : ∀ i, dt * degW g i ≤ C i) (θ0 : ι → ℝ) (n : ℕ) (i : ι) :
    Finset.univ.inf' Finset.univ_nonempty θ0 ≤ iter (eulerM C 0 g dt) θ0 n i ∧
      iter (eulerM C 0 g dt) θ0 n i ≤ Finset.univ.sup' Finset.univ_nonempty θ0 := by
  have hA' : ∀ i, dt * (degW g i + (0 : ι → ℝ) i) ≤ C i := by
    intro i; simpa using hA i
  have hM := (L1_nonneg_iff C 0 g dt hC hg hgii hdt).2 hA'
  have hrow : ∀ i, ∑ j, eulerM C 0 g dt i j = 1 := by
    intro i; rw [eulerM_row_sum C 0 g dt hgii]; simp
  refine dmp_stochastic _ hM hrow θ0 _ _ ?_ n i
  intro j
  exact ⟨Finset.inf'_le _ (Finset.mem_univ j), Finset.le_sup' _ (Finset.mem_univ j)⟩

/-- L2 / Theorem B with Dirichlet nodes (corrected clause). Nodes in `D` are held at
`θf` every step. Under Lemma A, for every `n` and `i`:
`min(0, θf, min_j θ⁰_j) ≤ θⁿ_i ≤ max(0, θf, max_j θ⁰_j)`.
The `0` cannot be dropped when `H ≠ 0` (see `L2_note_dirichlet_clause_false`); with a
fire source, `θf = T_f - T_a > 0`, so the upper bound is `max(θf, max_j θ⁰_j, 0)`. -/
theorem L2_dmp_dirichlet [Nonempty ι] (C H : ι → ℝ) (g : ι → ι → ℝ) (dt : ℝ)
    (hC : ∀ i, 0 < C i) (hg : ∀ i j, 0 ≤ g i j) (hgii : ∀ i, g i i = 0) (hH : ∀ i, 0 ≤ H i)
    (hdt : 0 ≤ dt) (hA : ∀ i, dt * (degW g i + H i) ≤ C i) (D : ι → Prop) [DecidablePred D]
    (θf : ℝ) (θ0 : ι → ℝ) (n : ℕ) (i : ι) :
    min (min 0 θf) (Finset.univ.inf' Finset.univ_nonempty θ0) ≤
        iterD (eulerM C H g dt) D θf θ0 n i ∧
      iterD (eulerM C H g dt) D θf θ0 n i ≤
        max (max 0 θf) (Finset.univ.sup' Finset.univ_nonempty θ0) := by
  obtain ⟨hM, -, hrow⟩ := L1_lemmaA C H g dt hC hg hgii hH hdt hA
  refine dmp_dirichlet _ hM hrow D θf θ0 _ _ ((min_le_left _ _).trans (min_le_left _ _))
    ((le_max_left _ _).trans (le_max_left _ _)) ((min_le_left _ _).trans (min_le_right _ _))
    ((le_max_right _ _).trans (le_max_left _ _)) ?_ n i
  intro j
  exact ⟨(min_le_right _ _).trans (Finset.inf'_le _ (Finset.mem_univ j)),
    (Finset.le_sup' _ (Finset.mem_univ j)).trans (le_max_right _ _)⟩

end Euler

section Counterexample

/-- Two nodes joined by one conductance `g₀₁ = g₁₀ = 1`, no self-loops. -/
noncomputable def gEx : Fin 2 → Fin 2 → ℝ := fun i j => if i = j then 0 else 1

/-- Loss to ambient: none at node 0, `H₁ = 1` at node 1. -/
noncomputable def HEx : Fin 2 → ℝ := ![0, 1]

/-- The note's uncorrected Dirichlet clause ("a node held at `θf` extends the upper
bound to `max(θf, max_j θ⁰_j)`") is false when `H ≠ 0` and `θf < 0`. Two nodes,
`g = 1`, `C = 1`, `H = (0, 1)`, `Δt = 0.3` (Lemma A holds), node 0 held at `θf = -5`,
`θ⁰ = (-5, -10)`. After two steps node 1 is at `-3.7`, above `max(θf, max θ⁰) = -5`.
The loss term pulls toward `0`, which is why `0` must stay in the bound. -/
theorem L2_note_dirichlet_clause_false :
    (∀ i, (0.3 : ℝ) * (degW gEx i + HEx i) ≤ 1) ∧
      iterD (eulerM (fun _ => 1) HEx gEx 0.3) (· = 0) (-5) ![-5, -10] 2 1 = -3.7 ∧
      max (-5 : ℝ) (Finset.univ.sup' Finset.univ_nonempty (![-5, -10] : Fin 2 → ℝ)) < -3.7 := by
  have hgii : ∀ i, gEx i i = 0 := by intro i; simp [gEx]
  have hdeg : ∀ i, degW gEx i = 1 := by
    intro i; fin_cases i <;> simp [degW, gEx, Fin.sum_univ_two]
  refine ⟨?_, ?_, ?_⟩
  · intro i; fin_cases i <;> simp [hdeg, HEx] <;> norm_num
  · have e10 : eulerM (fun _ => 1) HEx gEx 0.3 1 0 = 0.3 := by
      rw [eulerM_apply _ _ _ _ hgii]; norm_num [gEx]
    have e11 : eulerM (fun _ => 1) HEx gEx 0.3 1 1 = 0.4 := by
      rw [eulerM_apply _ _ _ _ hgii, hdeg]; norm_num [HEx]
    generalize eulerM (fun _ => 1) HEx gEx 0.3 = M at e10 e11 ⊢
    simp [iterD, Matrix.mulVec, dotProduct, Fin.sum_univ_two, e10, e11]
    norm_num
  · have hs : Finset.univ.sup' Finset.univ_nonempty (![-5, -10] : Fin 2 → ℝ) ≤ -5 :=
      Finset.sup'_le _ _ (fun j _ => by fin_cases j <;> norm_num)
    exact max_lt (by norm_num) (by linarith)

end Counterexample

/-! ## Time-dependent routing: arrival functions, FIFO, waiting, replanning -/

section TimeDependent

variable {α : Type*}

/-- R3. Arrival time at the end of a path given by its arcs' arrival functions,
first arc first: `a_p = A_{e_k} ∘ ⋯ ∘ A_{e_1}`. -/
def pathArr : List (α → α) → α → α
  | [] => id
  | A :: rest => pathArr rest ∘ A

/-- Concatenating paths composes their arrival functions. -/
theorem pathArr_append (p q : List (α → α)) : pathArr (p ++ q) = pathArr q ∘ pathArr p := by
  induction p with
  | nil => rfl
  | cons A p ih => simp only [List.cons_append, pathArr, ih, Function.comp_assoc]

/-- L10a. FIFO arrival functions compose to FIFO: if every arc's arrival function is
monotone (entering later never arrives earlier), so is the path's. Times are any
preorder; Studio uses `WithTop ℝ` (seconds) or ticks with `⊤` for "edge burnt". -/
theorem L10a_fifo_compose [Preorder α] :
    ∀ es : List (α → α), (∀ A ∈ es, Monotone A) → Monotone (pathArr es)
  | [], _ => monotone_id
  | A :: rest, h =>
    (L10a_fifo_compose rest (fun B hB => h B (List.mem_cons_of_mem A hB))).comp
      (h A (by simp))

/-- R1. If no arc arrives before it is entered (`t ≤ A_e t`), neither does a path. -/
theorem pathArr_ge [Preorder α] :
    ∀ es : List (α → α), (∀ A ∈ es, ∀ t, t ≤ A t) → ∀ t, t ≤ pathArr es t
  | [], _, _ => le_rfl
  | A :: rest, h, t =>
    le_trans (h A (by simp) t)
      (pathArr_ge rest (fun B hB => h B (List.mem_cons_of_mem A hB)) (A t))

/-- L10b. Under FIFO, departing the rest of a route later never arrives earlier:
`t' ≥ a(u) ⇒ a_q(t') ≥ a_q(a(u))`. -/
theorem L10b_later_departure [Preorder α] (q : List (α → α)) (hq : ∀ A ∈ q, Monotone A)
    {a t' : α} (h : a ≤ t') : pathArr q a ≤ pathArr q t' :=
  L10a_fifo_compose q hq h

/-- L10b. Waiting never helps: inserting a wait `W` (any map with `t ≤ W t`) at the
node between route parts `p` and `q` never gives an earlier final arrival, provided
the arcs after the wait are FIFO. No assumption on `p` is needed. -/
theorem L10b_wait_never_helps [Preorder α] (p q : List (α → α)) (W : α → α)
    (hq : ∀ A ∈ q, Monotone A) (hW : ∀ t, t ≤ W t) (t : α) :
    pathArr (p ++ q) t ≤ pathArr (p ++ W :: q) t := by
  rw [pathArr_append, pathArr_append]
  exact L10a_fifo_compose q hq (hW _)

/-- R2. Closure-window arc (note §5.2): entering at `t`, arrive at `t + τ` if that is
strictly before the closing time `b`, otherwise never (`⊤`). Times in seconds. -/
noncomputable def closeArc (τ b : ℝ) (x : WithTop ℝ) : WithTop ℝ :=
  WithTop.recTopCoe ⊤ (fun t => if t + τ < b then ((t + τ : ℝ) : WithTop ℝ) else ⊤) x

theorem closeArc_top (τ b : ℝ) : closeArc τ b ⊤ = ⊤ := rfl

theorem closeArc_coe (τ b t : ℝ) :
    closeArc τ b t = if t + τ < b then ((t + τ : ℝ) : WithTop ℝ) else ⊤ := rfl

/-- R2. A closure window is FIFO for any walking time `τ` and closing time `b`. -/
theorem R2_closeArc_fifo (τ b : ℝ) : Monotone (closeArc τ b) := by
  intro x y hxy
  induction y using WithTop.recTopCoe with
  | top => rw [closeArc_top]; exact le_top
  | coe t' =>
    induction x using WithTop.recTopCoe with
    | top => simp at hxy
    | coe t =>
      have htt : t ≤ t' := WithTop.coe_le_coe.mp hxy
      rw [closeArc_coe, closeArc_coe]
      split_ifs with h1 h2 h2
      · exact WithTop.coe_le_coe.mpr (by linarith)
      · exact le_top
      · exact absurd (by linarith : t + τ < b) h1
      · exact le_top

/-- R2, the rule the code uses (`rl.js` line 52, `rl.py` line 129): leaving at tick `t`
toward a node that ignites at `ign`, arrive at `t + 1` if `t < ign`, otherwise never.
It checks the departure tick, so arrival at `t + 1` may equal `ign`: one tick more
permissive than the closure window `closeArc 1 ign`. -/
noncomputable def hopArc (ign : WithTop ℝ) (x : WithTop ℝ) : WithTop ℝ :=
  WithTop.recTopCoe ⊤ (fun t => if (t : WithTop ℝ) < ign then ((t + 1 : ℝ) : WithTop ℝ) else ⊤) x

theorem hopArc_coe (ign : WithTop ℝ) (t : ℝ) :
    hopArc ign t = if (t : WithTop ℝ) < ign then ((t + 1 : ℝ) : WithTop ℝ) else ⊤ := rfl

/-- R2. The code's hop rule is FIFO for any ignition time, so L10a/L10b apply to it. -/
theorem R2_hopArc_fifo (ign : WithTop ℝ) : Monotone (hopArc ign) := by
  intro x y hxy
  induction y using WithTop.recTopCoe with
  | top => exact le_top
  | coe t' =>
    induction x using WithTop.recTopCoe with
    | top => simp at hxy
    | coe t =>
      have htt : t ≤ t' := WithTop.coe_le_coe.mp hxy
      rw [hopArc_coe, hopArc_coe]
      split_ifs with h1 h2 h2
      · exact WithTop.coe_le_coe.mpr (by linarith)
      · exact le_top
      · exact absurd (lt_of_le_of_lt hxy h2) h1
      · exact le_top

end TimeDependent

section Walks

variable {V α : Type*}

/-- End vertex of the walk that starts at `s` and visits `vs` in turn. -/
def wEnd : V → List V → V
  | s, [] => s
  | _, v :: vs => wEnd v vs

/-- Arrival time of the walk from `u`, leaving at `t`, along `vs`, with arc arrival
functions `A u v`. -/
def wArr (A : V → V → α → α) : V → α → List V → α
  | _, t, [] => t
  | u, t, v :: vs => wArr A v (A u v t) vs

/-- Every step of the walk is an arc of `E`. -/
def IsWalk (E : V → V → Prop) : V → List V → Prop
  | _, [] => True
  | u, v :: vs => E u v ∧ IsWalk E v vs

theorem wEnd_append (s : V) (p q : List V) : wEnd s (p ++ q) = wEnd (wEnd s p) q := by
  induction p generalizing s with
  | nil => rfl
  | cons v p ih => exact ih v

theorem wArr_append (A : V → V → α → α) (s : V) (t : α) (p q : List V) :
    wArr A s t (p ++ q) = wArr A (wEnd s p) (wArr A s t p) q := by
  induction p generalizing s t with
  | nil => rfl
  | cons v p ih => exact ih v (A s v t)

theorem IsWalk_append (E : V → V → Prop) (s : V) (p q : List V) :
    IsWalk E s (p ++ q) ↔ IsWalk E s p ∧ IsWalk E (wEnd s p) q := by
  induction p generalizing s with
  | nil => simp [IsWalk, wEnd]
  | cons v p ih => simp only [List.cons_append, IsWalk, wEnd, ih, and_assoc]

/-- R7. `w` is an earliest-arrival walk from `s`, leaving at `t`, to `d`: it is a walk
of `E` ending at `d`, and no walk of `E` from `s` to `d` arrives earlier. -/
def IsEA [LE α] (E : V → V → Prop) (A : V → V → α → α) (s : V) (t : α) (d : V)
    (w : List V) : Prop :=
  IsWalk E s w ∧ wEnd s w = d ∧
    ∀ w', IsWalk E s w' → wEnd s w' = d → wArr A s t w ≤ wArr A s t w'

/-- L11 (replanning consistency). If the walk `p ++ q` is earliest-arrival from `s`
at `t` to `d`, then its suffix `q` is earliest-arrival to `d` from the vertex `p`
ends at, leaving at the time `p` reached it, under the same arrival functions
(unchanged forecast). No FIFO assumption is needed: the order on times is any `LE`,
and optimality is over walks, which may be concatenated. -/
theorem L11_suffix_ea [LE α] (E : V → V → Prop) (A : V → V → α → α) (s : V) (t : α)
    (d : V) (p q : List V) (h : IsEA E A s t d (p ++ q)) :
    IsEA E A (wEnd s p) (wArr A s t p) d q := by
  obtain ⟨hw, hend, hopt⟩ := h
  rw [IsWalk_append] at hw
  rw [wEnd_append] at hend
  refine ⟨hw.2, hend, fun q' hq' hend' => ?_⟩
  have := hopt (p ++ q') ((IsWalk_append E s p q').2 ⟨hw.1, hq'⟩)
    (by rw [wEnd_append]; exact hend')
  rwa [wArr_append, wArr_append] at this

section PrefixCounterexample

/-- R3 counterexample graph: `s = 0`, `v = 1`, `x = 2`, `exit = 3`, arcs
`s → v`, `s → x`, `x → v`, `v → exit`. -/
def Ecx : Fin 4 → Fin 4 → Prop := fun u v =>
  (u = 0 ∧ v = 1) ∨ (u = 0 ∧ v = 2) ∨ (u = 2 ∧ v = 1) ∨ (u = 1 ∧ v = 3)

/-- R3 counterexample arrival functions (seconds): `v → exit` opens at 5,
`A(t) = max(t, 5)`; every other arc takes 1 s. -/
noncomputable def Acx : Fin 4 → Fin 4 → ℝ → ℝ := fun u v t =>
  if u = 1 ∧ v = 3 then max t 5 else t + 1

theorem cx_last : ∀ (w : List (Fin 4)) (u : Fin 4) (t : ℝ), IsWalk Ecx u w →
    wEnd u w = 3 → w ≠ [] → 5 ≤ wArr Acx u t w
  | [], _, _, _, _, h => absurd rfl h
  | [v], u, t, hw, he, _ => by
    change v = 3 at he
    subst he
    rcases hw.1 with ⟨-, h⟩ | ⟨-, h⟩ | ⟨-, h⟩ | ⟨rfl, -⟩
    · exact absurd h (by decide)
    · exact absurd h (by decide)
    · exact absurd h (by decide)
    · show 5 ≤ Acx 1 3 t
      simp [Acx]
  | v :: w :: ws, u, t, hw, he, _ => cx_last (w :: ws) v (Acx u v t) hw.2 he (by simp)

/-- R3 / L10a, the note's universal wording is false under plain FIFO. Every arc is
FIFO, the walk `s → x → v → exit` is earliest-arrival from `s` at time 0 (it arrives
at 5, and no walk can reach the exit before 5), yet its prefix `s → x → v` reaches `v`
at 2 while `s → v` reaches it at 1, so the prefix is not earliest-arrival. Dean's
Lemma 3.4 only gives *some* optimal path with optimal subpaths; the universal form
needs strict FIFO. (The suffix direction, L11, holds regardless.) -/
theorem R3_prefix_not_ea_under_fifo :
    (∀ u v, Monotone (Acx u v)) ∧ IsEA Ecx Acx 0 0 3 ([2, 1] ++ [3]) ∧
      ¬ IsEA Ecx Acx 0 0 1 [2, 1] := by
  refine ⟨?_, ⟨?_, rfl, ?_⟩, ?_⟩
  · intro u v a b h
    simp only [Acx]
    split_ifs
    · exact max_le_max h le_rfl
    · linarith
  · simp [IsWalk, Ecx]
  · intro w' hw' he'
    have h5 := cx_last w' 0 0 hw' he' (by rintro rfl; exact absurd he' (by decide))
    have : wArr Acx 0 0 ([2, 1] ++ [3]) = 5 := by
      norm_num [wArr, Acx]
    rw [this]
    exact h5
  · rintro ⟨-, -, h⟩
    have := h [1] (by simp [IsWalk, Ecx]) rfl
    norm_num [wArr, Acx] at this

end PrefixCounterexample

end Walks

/-! ## L7: the earliest-escape BFS computes the earliest safe arrival -/

section BFS

variable {V : Type*} (E : V → V → Prop) (ign : V → WithTop ℝ) (s : V)

/-- R6. `SafeReach k v`: some walk of exactly `k` moves goes from `s` to `v`, each
move along an open edge `E`, and each move leaving at tick `t` toward a node that has
not ignited yet (`t < ign(w (t+1))`, the code's burn check). The start node's own
ignition is not checked, as in the code. `ign` may be fractional (`step / fireSpeed`)
or `⊤` (never ignites). -/
def SafeReach (k : ℕ) (v : V) : Prop :=
  ∃ w : ℕ → V, w 0 = s ∧ w k = v ∧
    ∀ t < k, E (w t) (w (t + 1)) ∧ ((t : ℝ) : WithTop ℝ) < ign (w (t + 1))

theorem safeReach_zero (v : V) : SafeReach E ign s 0 v ↔ v = s := by
  constructor
  · rintro ⟨w, h0, hk, -⟩; rw [← hk, h0]
  · rintro rfl; exact ⟨fun _ => v, rfl, rfl, fun t ht => absurd ht (Nat.not_lt_zero t)⟩

theorem safeReach_succ (k : ℕ) (v : V) :
    SafeReach E ign s (k + 1) v ↔
      ∃ u, SafeReach E ign s k u ∧ E u v ∧ ((k : ℝ) : WithTop ℝ) < ign v := by
  constructor
  · rintro ⟨w, h0, hk, hw⟩
    refine ⟨w k, ⟨w, h0, rfl, fun t ht => hw t (by omega)⟩, ?_⟩
    have := hw k (by omega)
    rw [hk] at this
    exact this
  · rintro ⟨u, ⟨w, h0, hk, hw⟩, he, hign⟩
    refine ⟨fun t => if t ≤ k then w t else v, by simp [h0], by simp, fun t ht => ?_⟩
    by_cases htk : t < k
    · have h1 : t ≤ k := htk.le
      have h2 : t + 1 ≤ k := htk
      simp only [h1, h2, ite_true]
      exact hw t htk
    · have htk' : t = k := by omega
      subst htk'
      have h3 : ¬ (t + 1 ≤ t) := by omega
      simp only [le_refl, ite_true, h3, ite_false]
      rw [hk]
      exact ⟨he, hign⟩

/-- The layered BFS of `earliestEscape` (`rl.js` lines 41-62, `rl.py` lines 98-138),
as sets: `(frontier, seen)` after `k` layers. A node joins layer `k + 1` if it is not
yet seen, has an open edge from a node of layer `k`, and passes the burn check
`k < ign v` at departure tick `k`. A node that fails the burn check is not marked
seen. -/
def bfs : ℕ → Set V × Set V
  | 0 => ({s}, {s})
  | k + 1 =>
    ({v | v ∉ (bfs k).2 ∧ ∃ u ∈ (bfs k).1, E u v ∧ ((k : ℝ) : WithTop ℝ) < ign v},
     (bfs k).2 ∪ {v | v ∉ (bfs k).2 ∧ ∃ u ∈ (bfs k).1, E u v ∧ ((k : ℝ) : WithTop ℝ) < ign v})

/-- The BFS invariant: layer `k` is exactly the set of nodes whose earliest safe walk
has `k` moves, and `seen` after `k` layers is the set reachable safely in at most `k`
moves. The step that needs the fire to be monotone (fixed ignition times) is: a node
first reached at `j < k` can make the same move at `j` that was safe at `k`. -/
theorem bfs_spec (k : ℕ) :
    (bfs E ign s k).1 = {v | SafeReach E ign s k v ∧ ∀ j < k, ¬ SafeReach E ign s j v} ∧
      (bfs E ign s k).2 = {v | ∃ j ≤ k, SafeReach E ign s j v} := by
  induction k with
  | zero =>
    constructor
    · ext v; simp [bfs, safeReach_zero]
    · ext v; simp [bfs, safeReach_zero]
  | succ k ih =>
    obtain ⟨hF, hS⟩ := ih
    have key : ∀ v, (v ∉ (bfs E ign s k).2 ∧
        ∃ u ∈ (bfs E ign s k).1, E u v ∧ ((k : ℝ) : WithTop ℝ) < ign v) ↔
        (SafeReach E ign s (k + 1) v ∧ ∀ j < k + 1, ¬ SafeReach E ign s j v) := by
      intro v
      rw [hF, hS, safeReach_succ E ign s k v]
      simp only [Set.mem_ofPred_eq, not_exists, not_and]
      constructor
      · rintro ⟨hnot, u, ⟨hu, -⟩, he, hi⟩
        exact ⟨⟨u, hu, he, hi⟩, fun j hj => hnot j (by omega)⟩
      · rintro ⟨⟨u, hu, he, hi⟩, hfirst⟩
        refine ⟨fun j hj => hfirst j (by omega), u, ⟨hu, fun j hj hju => ?_⟩, he, hi⟩
        apply hfirst (j + 1) (by omega)
        rw [safeReach_succ]
        refine ⟨u, hju, he, lt_of_le_of_lt ?_ hi⟩
        exact WithTop.coe_le_coe.mpr (Nat.cast_le.mpr hj.le)
    constructor
    · ext v
      exact key v
    · ext v
      show v ∈ (bfs E ign s k).2 ∨ (v ∉ (bfs E ign s k).2 ∧
        ∃ u ∈ (bfs E ign s k).1, E u v ∧ ((k : ℝ) : WithTop ℝ) < ign v) ↔ _
      rw [key v, hS]
      simp only [Set.mem_ofPred_eq]
      constructor
      · rintro (⟨j, hj, h⟩ | ⟨h, -⟩)
        · exact ⟨j, by omega, h⟩
        · exact ⟨k + 1, le_rfl, h⟩
      · rintro ⟨j, hj, h⟩
        by_cases hex : ∃ j' ≤ k, SafeReach E ign s j' v
        · exact Or.inl hex
        · right
          have hj' : j = k + 1 := by
            by_contra hne
            exact hex ⟨j, by omega, h⟩
          subst hj'
          exact ⟨h, fun j' hj' hr => hex ⟨j', by omega, hr⟩⟩

open Classical in
/-- L7 (specification level). A node is in BFS layer `k` exactly when `k` is the
least number of moves of a safe walk to it: `layer(v) = e*(v) = min {k : SafeReach k v}`.
So `earliestEscape` returning layer `t + 1` for the exit returns the earliest safe
arrival tick; if it returns null within `maxSteps`, no safe walk of at most
`maxSteps` moves exists. -/
theorem L7_bfs_layer_eq_earliest (k : ℕ) (v : V) :
    v ∈ (bfs E ign s k).1 ↔ ∃ h : ∃ n, SafeReach E ign s n v, Nat.find h = k := by
  rw [(bfs_spec E ign s k).1]
  constructor
  · rintro ⟨hr, hmin⟩
    exact ⟨⟨k, hr⟩, (Nat.find_eq_iff _).2 ⟨hr, hmin⟩⟩
  · rintro ⟨h, hk⟩
    exact (Nat.find_eq_iff h).1 hk

end BFS

/-! ## Units of the step condition (G1-G3, G6), checked with `FireEgress.Dim` -/

section Units

/-- Heat capacity of a node, J K⁻¹. -/
def heatCapacity : Dim := Dim.energy / Dim.temp
/-- Conductance and loss coefficient, W K⁻¹. -/
def conductance : Dim := Dim.power / Dim.temp

/-- G1. Every term of `C_i Ṫ_i = ∑ g_ij (T_j - T_i) - H_i (T_i - T_a) + q_i` is a
power (W). -/
theorem G1_terms_are_power :
    heatCapacity * Dim.temp / Dim.time = Dim.power ∧ conductance * Dim.temp = Dim.power := by
  decide

/-- G3. The step-condition ratio `Δt (G_i + H_i) / C_i` is dimensionless, so
`Δt (G_i + H_i) ≤ C_i` compares joules per kelvin with joules per kelvin. -/
theorem G3_ratio_dimensionless : Dim.time * conductance / heatCapacity = Dim.one := by
  decide

/-- G3. The time-step bound `C_i / (G_i + H_i)` is a time (s). -/
theorem G3_bound_is_time : heatCapacity / conductance = Dim.time := by decide

/-- G6, `d = 3`: `g = k h` is a conductance (W K⁻¹) and `C = ρ c_p h³` a heat
capacity (J K⁻¹). -/
theorem G6_lattice_d3 :
    Dim.conductivity * Dim.length = conductance ∧
      Dim.density * Dim.specificHeat * Dim.volume = heatCapacity := by
  decide

/-- G6, `d = 2`: `g = k h⁰ = k` is not a conductance (it is W K⁻¹ per metre of depth),
but the ratio `C / G = ρ c_p h² / k` is still a time. -/
theorem G6_lattice_d2 :
    Dim.conductivity ≠ conductance ∧
      Dim.density * Dim.specificHeat * Dim.area / Dim.conductivity = Dim.time := by
  decide

/-- G6, `d = 1`: `g = k / h` is not a conductance (W K⁻¹ per m² of cross-section),
but the ratio `C / G = ρ c_p h / (k / h)` is still a time. -/
theorem G6_lattice_d1 :
    Dim.conductivity / Dim.length ≠ conductance ∧
      Dim.density * Dim.specificHeat * Dim.length / (Dim.conductivity / Dim.length) =
        Dim.time := by
  decide

/-- G6. The Fourier number `α Δt / h²` with `α = k / (ρ c_p)` is dimensionless. -/
theorem G6_fourier_dimensionless :
    Dim.conductivity / (Dim.density * Dim.specificHeat) * Dim.time / Dim.area = Dim.one := by
  decide

end Units

end FireEgress.GraphNumerics

#print axioms FireEgress.GraphNumerics.eulerM_apply
#print axioms FireEgress.GraphNumerics.eulerM_row_sum
#print axioms FireEgress.GraphNumerics.L1_nonneg_iff
#print axioms FireEgress.GraphNumerics.L1_lemmaA
#print axioms FireEgress.GraphNumerics.iter_eq_pow
#print axioms FireEgress.GraphNumerics.mulVec_bounds
#print axioms FireEgress.GraphNumerics.substochastic_step
#print axioms FireEgress.GraphNumerics.dmp_substochastic
#print axioms FireEgress.GraphNumerics.dmp_stochastic
#print axioms FireEgress.GraphNumerics.dmp_dirichlet
#print axioms FireEgress.GraphNumerics.L2_dmp
#print axioms FireEgress.GraphNumerics.L2_dmp_lossless
#print axioms FireEgress.GraphNumerics.L2_dmp_dirichlet
#print axioms FireEgress.GraphNumerics.L2_note_dirichlet_clause_false
#print axioms FireEgress.GraphNumerics.pathArr_append
#print axioms FireEgress.GraphNumerics.L10a_fifo_compose
#print axioms FireEgress.GraphNumerics.pathArr_ge
#print axioms FireEgress.GraphNumerics.L10b_later_departure
#print axioms FireEgress.GraphNumerics.L10b_wait_never_helps
#print axioms FireEgress.GraphNumerics.closeArc_top
#print axioms FireEgress.GraphNumerics.closeArc_coe
#print axioms FireEgress.GraphNumerics.R2_closeArc_fifo
#print axioms FireEgress.GraphNumerics.hopArc_coe
#print axioms FireEgress.GraphNumerics.R2_hopArc_fifo
#print axioms FireEgress.GraphNumerics.wEnd_append
#print axioms FireEgress.GraphNumerics.wArr_append
#print axioms FireEgress.GraphNumerics.IsWalk_append
#print axioms FireEgress.GraphNumerics.L11_suffix_ea
#print axioms FireEgress.GraphNumerics.cx_last
#print axioms FireEgress.GraphNumerics.R3_prefix_not_ea_under_fifo
#print axioms FireEgress.GraphNumerics.safeReach_zero
#print axioms FireEgress.GraphNumerics.safeReach_succ
#print axioms FireEgress.GraphNumerics.bfs_spec
#print axioms FireEgress.GraphNumerics.L7_bfs_layer_eq_earliest
#print axioms FireEgress.GraphNumerics.G1_terms_are_power
#print axioms FireEgress.GraphNumerics.G3_ratio_dimensionless
#print axioms FireEgress.GraphNumerics.G3_bound_is_time
#print axioms FireEgress.GraphNumerics.G6_lattice_d3
#print axioms FireEgress.GraphNumerics.G6_lattice_d2
#print axioms FireEgress.GraphNumerics.G6_lattice_d1
#print axioms FireEgress.GraphNumerics.G6_fourier_dimensionless
