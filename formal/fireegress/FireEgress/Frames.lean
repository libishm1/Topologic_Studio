import Mathlib

/-!
# Coordinate frames: IFC to Studio's viewer, and what that does to route lengths

Studio's viewer loads an IFC with web-ifc (`COORDINATE_TO_ORIGIN`, Y up). A point
`(x, y, z)` in IFC world coordinates (Z up) appears in the viewer as `(x, z, -y)`,
then moved by the model's coordination matrix (That Open's
`model.getCoordinationMatrix()`, read by `ViewerManager.getCoordinationMatrix`), and
Studio lifts it by the agent height. The navigation cross-check
(`app/graphs/navigation_check.py`, `Alignment`, `from_coordination`) composes exactly
this: `view = C (T p) + (t + lift e_y)`, with `T = TURNS["y"]`, `C` the upper-left
3 x 3 block of the coordination matrix and `t` its translation column, then reports
`_polyline_length` of routes in that frame.

What the coordination matrix may contain (read from `@thatopen/fragments` 3.4.7,
`CoordinatesManager.getCoordinationMatrix`): it is rebuilt from a stored position
`p`, an x direction `xd` and a y direction `yd` as the columns
`[xd | yd | xd ⨯₃ yd | p]`. `xd` and `yd` are the first two columns of web-ifc's own
coordination matrix, rounded to 1e-3 and 1e-5 respectively and stored as float32.
Nothing normalises them. So:

* if web-ifc's columns are orthonormal, `C` is a proper rotation (`fragM_mem_SO`);
* whatever they are, `det C = ‖xd ⨯₃ yd‖² ≥ 0`, so it is never a reflection
  (`fragM_det_nonneg`);
* if web-ifc's columns carried a unit scale `k` (`xd = k u`, `yd = k v`), `C` would
  stretch x and y by `k` but z by `k²`: not a uniform scale unless `k = 1`
  (`fragM_scaled_not_uniform`).

This file proves:

1. `(x, y, z) ↦ (x, z, -y)` is a linear isometry of `ℝ³` with determinant `+1`, and it
   commutes with the cross product (handedness is kept).
2. Any rigid motion (linear isometry plus translation) preserves distances and so
   polyline lengths; in particular Studio's whole frame change does when `C` is
   orthogonal.
3. A similarity `k • A + t` (`A` a linear isometry, `k ≥ 0`) multiplies every polyline
   length by exactly `k`, and lengths come out in metres exactly when `k` is the size
   of one source unit in metres.

Plus the bound behind the rounding of `xd`, `yd`: a map that stretches every distance
by a factor in `[1 - ε, 1 + ε]` changes polyline lengths by at most that factor.
-/

namespace FireEgress.Frames

open Matrix

/-- Three-dimensional Euclidean space, the frame of both IFC and the viewer. -/
abbrev E3 := EuclideanSpace ℝ (Fin 3)

/-! ## Polyline length -/

section Polyline

variable {P Q : Type*} [PseudoMetricSpace P] [PseudoMetricSpace Q]

/-- Length of a polyline: the sum of the distances between consecutive points. This is
`_polyline_length` in `navigation_check.py` (`norm(diff(points)).sum()`) and the
`length` loop in `findRoute` (`src/core/route.js`); fewer than two points give `0`. -/
def polyLen : List P → ℝ
  | a :: b :: rest => dist a b + polyLen (b :: rest)
  | _ => 0

/-- If a map multiplies every distance by `k`, it multiplies every polyline length by
`k`. -/
theorem polyLen_map_of_dist_eq (f : P → Q) (k : ℝ)
    (h : ∀ a b, dist (f a) (f b) = k * dist a b) :
    ∀ ps : List P, polyLen (ps.map f) = k * polyLen ps
  | [] => by simp [polyLen]
  | [_] => by simp [polyLen]
  | a :: b :: rest => by
      have ih := polyLen_map_of_dist_eq f k h (b :: rest)
      simp only [List.map_cons] at ih ⊢
      rw [polyLen, polyLen, ih, h]
      ring

/-- If a map stretches every distance by at most `c`, every polyline grows by at most
`c`. -/
theorem polyLen_map_le (f : P → Q) (c : ℝ)
    (h : ∀ a b, dist (f a) (f b) ≤ c * dist a b) :
    ∀ ps : List P, polyLen (ps.map f) ≤ c * polyLen ps
  | [] => by simp [polyLen]
  | [_] => by simp [polyLen]
  | a :: b :: rest => by
      have ih := polyLen_map_le f c h (b :: rest)
      simp only [List.map_cons] at ih ⊢
      rw [polyLen, polyLen, mul_add]
      exact add_le_add (h a b) ih

/-- If a map shrinks every distance by at most `c`, every polyline shrinks by at most
`c`. -/
theorem polyLen_map_ge (f : P → Q) (c : ℝ)
    (h : ∀ a b, c * dist a b ≤ dist (f a) (f b)) :
    ∀ ps : List P, c * polyLen ps ≤ polyLen (ps.map f)
  | [] => by simp [polyLen]
  | [_] => by simp [polyLen]
  | a :: b :: rest => by
      have ih := polyLen_map_ge f c h (b :: rest)
      simp only [List.map_cons] at ih ⊢
      rw [polyLen, polyLen, mul_add]
      exact add_le_add (h a b) ih

/-- An isometry (any distance-preserving map) preserves every polyline length. -/
theorem polyLen_map_isometry {f : P → Q} (hf : Isometry f) (ps : List P) :
    polyLen (ps.map f) = polyLen ps := by
  rw [polyLen_map_of_dist_eq f 1 (fun a b => by rw [hf.dist_eq, one_mul]) ps, one_mul]

end Polyline

/-! ## Orthogonal matrices act as linear isometries -/

section Orthogonal

variable {n : Type*} [Fintype n] [DecidableEq n]

omit [DecidableEq n] in
/-- The squared Euclidean norm is the dot product of the coordinates with themselves. -/
theorem norm_sq_eq_dot (w : EuclideanSpace ℝ n) :
    ‖w‖ ^ 2 = WithLp.ofLp w ⬝ᵥ WithLp.ofLp w := by
  rw [← real_inner_self_eq_norm_sq, EuclideanSpace.inner_eq_star_dotProduct]
  simp

/-- An orthogonal matrix (`Mᵀ M = 1`) preserves the Euclidean norm. -/
theorem norm_toEuclideanLin_of_orth (M : Matrix n n ℝ) (hM : Mᵀ * M = 1)
    (v : EuclideanSpace ℝ n) : ‖toEuclideanLin M v‖ = ‖v‖ := by
  have hsq : ‖toEuclideanLin M v‖ ^ 2 = ‖v‖ ^ 2 := by
    rw [norm_sq_eq_dot, norm_sq_eq_dot, toLpLin_apply, WithLp.ofLp_toLp,
      dotProduct_mulVec, ← mulVec_transpose, mulVec_mulVec, hM, one_mulVec,
      dotProduct_comm]
  exact (sq_eq_sq₀ (norm_nonneg _) (norm_nonneg _)).1 hsq

/-- An orthogonal matrix as a linear isometry of Euclidean space. -/
noncomputable def orthIso (M : Matrix n n ℝ) (hM : Mᵀ * M = 1) :
    EuclideanSpace ℝ n →ₗᵢ[ℝ] EuclideanSpace ℝ n where
  toLinearMap := toEuclideanLin M
  norm_map' := norm_toEuclideanLin_of_orth M hM

theorem orthIso_apply (M : Matrix n n ℝ) (hM : Mᵀ * M = 1) (v : EuclideanSpace ℝ n) :
    orthIso M hM v = WithLp.toLp 2 (M *ᵥ WithLp.ofLp v) :=
  toLpLin_apply 2 2 M v

/-- The determinant of the linear map is the determinant of its matrix. -/
theorem det_orthIso (M : Matrix n n ℝ) (hM : Mᵀ * M = 1) :
    LinearMap.det (orthIso M hM).toLinearMap = M.det := by
  change LinearMap.det (toEuclideanLin M) = M.det
  rw [toEuclideanLin_eq_toLin_orthonormal, LinearMap.det_toLin]

end Orthogonal

/-! ## 1. web-ifc's turn `(x, y, z) ↦ (x, z, -y)` -/

section Turn

/-- `TURNS["y"]` in `navigation_check.py`: web-ifc's turn from IFC's Z-up frame into
three.js's Y-up frame. -/
def turnY : Matrix (Fin 3) (Fin 3) ℝ := !![1, 0, 0; 0, 0, 1; 0, -1, 0]

/-- The turn sends `(x, y, z)` to `(x, z, -y)`. -/
theorem turnY_mulVec (x y z : ℝ) : turnY *ᵥ ![x, y, z] = ![x, z, -y] := by
  ext i; fin_cases i <;> simp [turnY, mulVec, dotProduct, Fin.sum_univ_three]

/-- The turn is orthogonal: `Tᵀ T = 1`. -/
theorem turnY_orth : turnYᵀ * turnY = 1 := by
  ext i j; fin_cases i <;> fin_cases j <;>
    simp [turnY, mul_apply, Fin.sum_univ_three, transpose_apply]

/-- The turn has determinant `+1`: it is a proper rotation (a quarter turn about x),
not a reflection. -/
theorem turnY_det : turnY.det = 1 := by
  simp [turnY, det_fin_three]

/-- The turn is in `SO(3)`. -/
theorem turnY_mem_SO : turnY ∈ specialOrthogonalGroup (Fin 3) ℝ :=
  mem_specialOrthogonalGroup_iff.2 ⟨(mem_orthogonalGroup_iff' (Fin 3) ℝ).2 turnY_orth, turnY_det⟩

/-- (1) The turn as a linear isometry of `ℝ³`. -/
noncomputable def turn : E3 →ₗᵢ[ℝ] E3 := orthIso turnY turnY_orth

/-- (1) On coordinates the isometry is exactly `(x, y, z) ↦ (x, z, -y)`. -/
theorem turn_apply (x y z : ℝ) :
    turn (WithLp.toLp 2 ![x, y, z]) = WithLp.toLp 2 ![x, z, -y] := by
  rw [turn, orthIso_apply, WithLp.ofLp_toLp, turnY_mulVec]

/-- (1) The isometry has determinant `+1`. -/
theorem turn_det : LinearMap.det turn.toLinearMap = 1 := by
  rw [turn, det_orthIso, turnY_det]

/-- (1) Handedness is kept: the turn commutes with the cross product,
`T (u ⨯₃ v) = (T u) ⨯₃ (T v)`, so a right-handed frame goes to a right-handed frame. -/
theorem turnY_cross (u v : Fin 3 → ℝ) :
    turnY *ᵥ (u ⨯₃ v) = (turnY *ᵥ u) ⨯₃ (turnY *ᵥ v) := by
  ext i; fin_cases i <;>
    simp [turnY, mulVec, dotProduct, Fin.sum_univ_three, cross_apply] <;> ring

/-- (1) Distances are preserved: `|T a - T b| = |a - b|`. -/
theorem turn_dist (a b : E3) : dist (turn a) (turn b) = dist a b :=
  turn.isometry.dist_eq a b

end Turn

/-! ## 2. Rigid motions preserve route lengths -/

section Rigid

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- A rigid motion: a linear isometry followed by a translation. -/
def rigid (A : V →ₗᵢ[ℝ] V) (t : V) (v : V) : V := A v + t

/-- (2) A rigid motion preserves distances. -/
theorem rigid_dist (A : V →ₗᵢ[ℝ] V) (t a b : V) :
    dist (rigid A t a) (rigid A t b) = dist a b := by
  simp only [rigid, dist_eq_norm, add_sub_add_right_eq_sub, ← map_sub, A.norm_map]

theorem rigid_isometry (A : V →ₗᵢ[ℝ] V) (t : V) : Isometry (rigid A t) :=
  Isometry.of_dist_eq (rigid_dist A t)

/-- (2) A rigid motion preserves every polyline length. -/
theorem rigid_polyLen (A : V →ₗᵢ[ℝ] V) (t : V) (ps : List V) :
    polyLen (ps.map (rigid A t)) = polyLen ps :=
  polyLen_map_isometry (rigid_isometry A t) ps

/-- The viewer's up axis, `e_y`. -/
noncomputable def eY : E3 := EuclideanSpace.single 1 1

/-- Studio's frame change from IFC to the viewer, as `from_coordination` builds it for
`up_axis = "y"`: turn, then the coordination matrix's 3 x 3 block `C`, then its
translation `t` plus the agent `lift` along `e_y`. -/
noncomputable def studioFrame (C : Matrix (Fin 3) (Fin 3) ℝ) (t : E3) (lift : ℝ)
    (p : E3) : E3 :=
  toEuclideanLin C (turn p) + (t + lift • eY)

/-- `from_coordination` multiplies the block by the turn once (`m[:3,:3] @ TURN`); that
is the same map as turning and then applying the block. -/
theorem studioFrame_eq_mul (C : Matrix (Fin 3) (Fin 3) ℝ) (t : E3) (lift : ℝ) (p : E3) :
    studioFrame C t lift p =
      WithLp.toLp 2 ((C * turnY) *ᵥ WithLp.ofLp p) + (t + lift • eY) := by
  simp only [studioFrame, turn, orthIso_apply, toLpLin_apply,
    mulVec_mulVec]

/-- (2) When the coordination block `C` is orthogonal (`Cᵀ C = 1`), the whole frame
change, translation and lift included, preserves every route length. -/
theorem studioFrame_polyLen (C : Matrix (Fin 3) (Fin 3) ℝ) (hC : Cᵀ * C = 1) (t : E3)
    (lift : ℝ) (ps : List E3) :
    polyLen (ps.map (studioFrame C t lift)) = polyLen ps := by
  have hf : studioFrame C t lift = rigid ((orthIso C hC).comp turn) (t + lift • eY) := by
    funext p; rfl
  rw [hf, rigid_polyLen]

/-- (2) When `C ∈ SO(3)` the composite linear part `C T` is in `SO(3)` too: the frame
change keeps handedness. -/
theorem studioFrame_linear_mem_SO (C : Matrix (Fin 3) (Fin 3) ℝ)
    (hC : C ∈ specialOrthogonalGroup (Fin 3) ℝ) :
    C * turnY ∈ specialOrthogonalGroup (Fin 3) ℝ :=
  Submonoid.mul_mem _ hC turnY_mem_SO

end Rigid

/-! ## 3. Uniform scale -/

section Scale

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- A similarity: a linear isometry scaled by `k`, followed by a translation. -/
def similar (k : ℝ) (A : V →ₗᵢ[ℝ] V) (t : V) (v : V) : V := k • A v + t

/-- (3) A similarity with `k ≥ 0` multiplies every distance by `k`. -/
theorem similar_dist (k : ℝ) (hk : 0 ≤ k) (A : V →ₗᵢ[ℝ] V) (t a b : V) :
    dist (similar k A t a) (similar k A t b) = k * dist a b := by
  simp only [similar, dist_eq_norm, add_sub_add_right_eq_sub, ← smul_sub, ← map_sub,
    norm_smul, A.norm_map, Real.norm_eq_abs, abs_of_nonneg hk]

/-- (3) A similarity with `k ≥ 0` multiplies every polyline length by exactly `k`. -/
theorem similar_polyLen (k : ℝ) (hk : 0 ≤ k) (A : V →ₗᵢ[ℝ] V) (t : V) (ps : List V) :
    polyLen (ps.map (similar k A t)) = k * polyLen ps :=
  polyLen_map_of_dist_eq _ k (similar_dist k hk A t) ps

/-- The unit segment from `0` to `e_x`, a polyline of length `1`. -/
theorem polyLen_unit : polyLen [(0 : E3), EuclideanSpace.single 0 1] = 1 := by
  simp [polyLen, dist_eq_norm]

/-- (3) When lengths are in metres. Suppose the source coordinates are in a unit worth
`u` metres, so a polyline of coordinate length `L` is `u L` metres long, and the
frame change is a similarity with factor `k ≥ 0`. The viewer reports `k L`. That is
the length in metres for every polyline exactly when `k = u`: `k = 1` for a metre
model, `k = 0.001` for a millimetre model. With `k = 1` and a millimetre model every
length is 1000 times too large. -/
theorem similar_metres_iff (k u : ℝ) (hk : 0 ≤ k) (A : E3 →ₗᵢ[ℝ] E3) (t : E3) :
    (∀ ps : List E3, polyLen (ps.map (similar k A t)) = u * polyLen ps) ↔ k = u := by
  constructor
  · intro h
    have := h [0, EuclideanSpace.single 0 1]
    rw [similar_polyLen k hk, polyLen_unit, mul_one, mul_one] at this
    exact this
  · rintro rfl ps
    exact similar_polyLen k hk A t ps

/-- Rounding bound. If a map stretches every distance by a factor between `1 - ε` and
`1 + ε` (as a nearly orthogonal block does, e.g. rotation columns rounded to 1e-3),
every polyline length changes by at most that factor. -/
theorem approx_polyLen {P : Type*} [PseudoMetricSpace P] (f : P → P) (ε : ℝ)
    (h : ∀ a b, (1 - ε) * dist a b ≤ dist (f a) (f b) ∧
      dist (f a) (f b) ≤ (1 + ε) * dist a b) (ps : List P) :
    (1 - ε) * polyLen ps ≤ polyLen (ps.map f) ∧
      polyLen (ps.map f) ≤ (1 + ε) * polyLen ps :=
  ⟨polyLen_map_ge f _ (fun a b => (h a b).1) ps, polyLen_map_le f _ (fun a b => (h a b).2) ps⟩

end Scale

/-! ## What That Open's coordination block can be -/

section Fragments

/-- `CoordinatesManager.getCoordinationMatrix` (`@thatopen/fragments` 3.4.7): the 3 x 3
block has columns `xd`, `yd` and `xd ⨯₃ yd`. -/
def fragM (xd yd : Fin 3 → ℝ) : Matrix (Fin 3) (Fin 3) ℝ := (Matrix.of ![xd, yd, xd ⨯₃ yd])ᵀ

/-- The block's determinant is `‖xd ⨯₃ yd‖²`. -/
theorem fragM_det (xd yd : Fin 3 → ℝ) :
    (fragM xd yd).det = (xd ⨯₃ yd) ⬝ᵥ (xd ⨯₃ yd) := by
  rw [fragM, det_transpose, triple_product_permutation, triple_product_eq_det]
  rfl

/-- Whatever `xd` and `yd` are, the block is never a reflection: `det ≥ 0`, and `0`
only when `xd`, `yd` are parallel. -/
theorem fragM_det_nonneg (xd yd : Fin 3 → ℝ) : 0 ≤ (fragM xd yd).det := by
  rw [fragM_det]
  exact Finset.sum_nonneg (fun i _ => mul_self_nonneg _)

/-- If `xd`, `yd` are orthonormal, the block is orthogonal. -/
theorem fragM_orth (xd yd : Fin 3 → ℝ) (hx : xd ⬝ᵥ xd = 1) (hy : yd ⬝ᵥ yd = 1)
    (hxy : xd ⬝ᵥ yd = 0) : (fragM xd yd)ᵀ * fragM xd yd = 1 := by
  have hcc : (xd ⨯₃ yd) ⬝ᵥ (xd ⨯₃ yd) = 1 := by
    rw [cross_dot_cross, hx, hy, hxy, dotProduct_comm yd xd, hxy]; ring
  have hyx : yd ⬝ᵥ xd = 0 := by rw [dotProduct_comm]; exact hxy
  have h1 : xd ⬝ᵥ (xd ⨯₃ yd) = 0 := dot_self_cross xd yd
  have h2 : yd ⬝ᵥ (xd ⨯₃ yd) = 0 := dot_cross_self xd yd
  have h1' : (xd ⨯₃ yd) ⬝ᵥ xd = 0 := by rw [dotProduct_comm]; exact h1
  have h2' : (xd ⨯₃ yd) ⬝ᵥ yd = 0 := by rw [dotProduct_comm]; exact h2
  have hrow : ∀ i, (Matrix.of ![xd, yd, xd ⨯₃ yd]) i = ![xd, yd, xd ⨯₃ yd] i := fun _ => rfl
  ext i j
  rw [fragM, transpose_transpose, mul_apply']
  fin_cases i <;> fin_cases j <;>
    simp [hrow, hx, hy, hxy, hyx, h1, h2, h1', h2', hcc]

/-- If `xd`, `yd` are orthonormal, the block is a proper rotation, in `SO(3)`. -/
theorem fragM_mem_SO (xd yd : Fin 3 → ℝ) (hx : xd ⬝ᵥ xd = 1) (hy : yd ⬝ᵥ yd = 1)
    (hxy : xd ⬝ᵥ yd = 0) : fragM xd yd ∈ specialOrthogonalGroup (Fin 3) ℝ := by
  refine mem_specialOrthogonalGroup_iff.2 ⟨(mem_orthogonalGroup_iff' (Fin 3) ℝ).2 (fragM_orth xd yd hx hy hxy), ?_⟩
  rw [fragM_det, cross_dot_cross, hx, hy, hxy, dotProduct_comm yd xd, hxy]; ring

/-- If web-ifc's directions carried a scale `k > 0`, `k ≠ 1` (`xd = k u`, `yd = k v`,
`u`, `v` orthonormal), the rebuilt block would not scale lengths uniformly: no single
factor `c` has `‖C w‖ = c ‖w‖` for every `w` (it stretches `e_x` by `k` and `e_z` by
`k²`). Studio's lengths would then not even be a fixed multiple of the IFC lengths. -/
theorem fragM_scaled_not_uniform (u v : Fin 3 → ℝ) (hu : u ⬝ᵥ u = 1) (hv : v ⬝ᵥ v = 1)
    (huv : u ⬝ᵥ v = 0) (k : ℝ) (hk : 0 < k) (hk1 : k ≠ 1) :
    ¬ ∃ c : ℝ, ∀ w : E3, ‖toEuclideanLin (fragM (k • u) (k • v)) w‖ = c * ‖w‖ := by
  rintro ⟨c, hc⟩
  have hcc : (u ⨯₃ v) ⬝ᵥ (u ⨯₃ v) = 1 := by
    rw [cross_dot_cross, hu, hv, huv, dotProduct_comm v u, huv]; ring
  -- squared norm of the image of a basis vector
  have key : ∀ j : Fin 3, ‖toEuclideanLin (fragM (k • u) (k • v)) (EuclideanSpace.single j 1)‖ ^ 2
      = (fun l => fragM (k • u) (k • v) l j) ⬝ᵥ (fun l => fragM (k • u) (k • v) l j) := by
    intro j
    rw [norm_sq_eq_dot, toLpLin_apply, WithLp.ofLp_toLp]
    congr 1 <;> (ext l; simp [mulVec, dotProduct])
  have h0 := key 0
  have h2 := key 2
  have hn : ∀ j : Fin 3, ‖(EuclideanSpace.single j (1 : ℝ) : E3)‖ = 1 := fun j => by simp
  rw [hc, hn, mul_one] at h0 h2
  have c0 : (fun l => fragM (k • u) (k • v) l 0) = k • u := by
    ext l; simp [fragM]
  have c2 : (fun l => fragM (k • u) (k • v) l 2) = (k * k) • (u ⨯₃ v) := by
    ext l; simp [fragM, LinearMap.smul_apply, smul_smul]
  rw [c0, dotProduct_smul, smul_dotProduct, hu] at h0
  rw [c2, dotProduct_smul, smul_dotProduct, hcc] at h2
  simp only [smul_eq_mul, mul_one] at h0 h2
  have hk2 : k * k = (k * k) * (k * k) := h0.symm.trans h2
  have : k * k * (k * k - 1) = 0 := by linarith [hk2]
  rcases mul_eq_zero.1 this with h | h
  · exact absurd h (by positivity)
  · have : (k - 1) * (k + 1) = 0 := by linarith [h]
    rcases mul_eq_zero.1 this with h' | h'
    · exact hk1 (by linarith)
    · linarith

/-- Rounding breaks exactness. A model turned 45° about the up axis has `xd = (cos, sin, 0)`;
fragments rounds it to 1e-3, `(0.707, 0.707, 0)`, so the rebuilt block stretches `e_x`
by `√0.999698 ≈ 0.99985`: lengths along it come out 0.015% short, and `from_coordination`
is not an exact frame change for such a model. -/
theorem fragM_rounded_45 :
    ((fragM ![0.707, 0.707, 0] ![-0.70711, 0.70711, 0])ᵀ *
      fragM ![0.707, 0.707, 0] ![-0.70711, 0.70711, 0]) 0 0 = 0.999698 := by
  rw [fragM, transpose_transpose, mul_apply']
  simp [dotProduct, Fin.sum_univ_three]
  norm_num

end Fragments

end FireEgress.Frames

#print axioms FireEgress.Frames.polyLen_map_of_dist_eq
#print axioms FireEgress.Frames.polyLen_map_le
#print axioms FireEgress.Frames.polyLen_map_ge
#print axioms FireEgress.Frames.polyLen_map_isometry
#print axioms FireEgress.Frames.norm_sq_eq_dot
#print axioms FireEgress.Frames.norm_toEuclideanLin_of_orth
#print axioms FireEgress.Frames.orthIso_apply
#print axioms FireEgress.Frames.det_orthIso
#print axioms FireEgress.Frames.turnY_mulVec
#print axioms FireEgress.Frames.turnY_orth
#print axioms FireEgress.Frames.turnY_det
#print axioms FireEgress.Frames.turnY_mem_SO
#print axioms FireEgress.Frames.turn_apply
#print axioms FireEgress.Frames.turn_det
#print axioms FireEgress.Frames.turnY_cross
#print axioms FireEgress.Frames.turn_dist
#print axioms FireEgress.Frames.rigid_dist
#print axioms FireEgress.Frames.rigid_isometry
#print axioms FireEgress.Frames.rigid_polyLen
#print axioms FireEgress.Frames.studioFrame_eq_mul
#print axioms FireEgress.Frames.studioFrame_polyLen
#print axioms FireEgress.Frames.studioFrame_linear_mem_SO
#print axioms FireEgress.Frames.similar_dist
#print axioms FireEgress.Frames.similar_polyLen
#print axioms FireEgress.Frames.polyLen_unit
#print axioms FireEgress.Frames.similar_metres_iff
#print axioms FireEgress.Frames.approx_polyLen
#print axioms FireEgress.Frames.fragM_det
#print axioms FireEgress.Frames.fragM_det_nonneg
#print axioms FireEgress.Frames.fragM_orth
#print axioms FireEgress.Frames.fragM_mem_SO
#print axioms FireEgress.Frames.fragM_scaled_not_uniform
#print axioms FireEgress.Frames.fragM_rounded_45
