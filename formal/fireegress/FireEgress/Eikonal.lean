import Mathlib
import FireEgress.Lattice
import FireEgress.AStar

/-!
# The eikonal fire front behind Studio's graph fire (FD-28, FD-29)

Studio's interim fire model (review `wiki/research/fire-egress-methods-2026-10-27.md`,
section 3.1 and section 10 step 3: "Dijkstra on `ℓ_e / F`") is `frontArrival` in
`src/core/deadline.js`: graph distance from the ignition node over open links, divided by a
speed `F` (m/s). The continuum model it stands for is the first-arrival time `T` of a front
moving with normal speed `F > 0`, the eikonal problem `F |∇T| = 1`, `T = 0` on the ignition
set (Osher and Sethian 1988; Sethian 1996, PNAS 93:1591, FD-28), understood in the viscosity
sense (Crandall and Lions 1983). On a graph the discrete analogue is the Bellman (min-plus)
equation `T v = min_u (T u + ℓ(u, v) / F)`, `T s = 0` (FD-29), solved exactly by Dijkstra
(Tsitsiklis 1995 for one-pass methods on discretised Hamilton-Jacobi equations).

What is proved here, in order:

1. **One ignition point, constant speed** (any real inner product space, so Studio's plan
   `EuclideanSpace ℝ (Fin 2)` and 3-D alike). `T x = ‖x - x₀‖ / F` is differentiable at every
   `x ≠ x₀` with gradient `(x - x₀) / (F ‖x - x₀‖)` of norm `1 / F`, so `T` is a classical
   solution of the eikonal equation away from the source; it is not differentiable at the
   source; it is `(1/F)`-Lipschitz; and it is the least travel time over polylines and over
   all curves (curve length is Mathlib's `eVariationOn`, so rectifiable or not).
2. **Several ignition points.** `T = minᵢ ‖x - xᵢ‖ / F` is `(1/F)`-Lipschitz; at every point of
   differentiability off the sources `|∇T| = 1/F`; it is differentiable wherever the nearest
   source is unique; and it is *not* differentiable where two distinct sources tie (the
   Voronoi edges where fronts meet). That last fact is why the eikonal problem needs viscosity
   solutions at all.
3. **Graph front against the continuum, open floor.** On the 8-connected lattice of spacing
   `h` the graph front `T_h = h · oct / F` (Lattice.lean's octile distance, the least
   8-connected path) satisfies `‖v - v₀‖/F ≤ T_h ≤ √(4 - 2√2) ‖v - v₀‖/F`, strictly below
   `1.0824 ‖v - v₀‖/F` off the source, and the offset `(12, 5)` already reaches ratio
   `> 1.0823`. Refining the lattice does not help: at the fixed point `(12c, 5c)` the lattice
   time is `(7 + 5√2) c / F` for every spacing `c / k`, against the true `13 c / F`. On the
   4-connected (`rectilinear`) lattice the factor is `√2`, attained on the diagonal. Every
   vertex Dijkstra pops on the open lattice carries exactly `T_h`.
4. **Walls.** With the free region `Ω` and the polyline geodesic distance `d_Ω` (`⊤` when
   unreachable), any graph whose links are straight segments inside `Ω` weighted by at least
   length / F (Studio's lattice links, wall-face links and door links alike, if their segments
   are free) has every walk cost, hence every Dijkstra label, `≥ d_Ω / F ≥ ‖·‖ / F`. On the
   lattice with walls, every walk is also `≥` the open-lattice octile time. Walls only delay.
5. **The discrete eikonal equation.** When Dijkstra's queue runs dry its labels satisfy the
   Bellman equation (`T s = 0`, `T v = min` over settled in-neighbours of `T u + a`, and no
   settled vertex links to an unsettled one); for edge times `ℓ / F` this reads
   `max_u (T v - T u) / ℓ(u, v) = 1 / F`, a one-sided `|∇T| = 1/F`. Conversely, on a finite
   graph with positive link times the Bellman equation has exactly one solution, the
   shortest-path arrival time (a discrete comparison principle).
6. **Speed.** A slower medium never makes the front arrive earlier: on graphs with per-link
   speeds, and for the continuum travel-time functional `∫ ‖γ'‖ / F(γ)` along any family of
   paths. The free-space radial model (`radialTimeline` in `fire.js`, ignition band
   `⌊‖x - x₀‖ / ℓ̃⌋`) is never later than any walled front with the same speed.

What is **not** covered, and why:
* Viscosity solutions: existence, the comparison principle and uniqueness for general `F(x)`
  or with obstacles are not in Mathlib; we do not prove that the geodesic arrival time is the
  viscosity solution in `Ω`, only that it bounds the graph front from below.
* Geodesics over rectifiable curves in `Ω`: `d_Ω` here is an infimum over polylines. For
  open `Ω` the two agree, but that is not proved.
* Fast marching (Sethian 1996) and its convergence order, and the level-set PDE
  `φ_t + F |∇φ| = 0`: not formalised. The lattice Dijkstra front is *not* a consistent
  discretisation of the Euclidean eikonal (item 3): it converges to the octile front.
* Anisotropic spread (wind, slope; ordered upwind methods), variable `F(x)` beyond
  monotonicity: no existence of optimal paths, no eikonal equation for variable `F`.
* The upper bound `T_h ≤ K ‖·‖/F` is proved on the unbounded lattice `ℤ²`; on a bounded
  rectangular floor it needs the octile path to stay in the bounding box, which is checked
  by the JavaScript property test, not proved. Off-lattice door links and wall-face nodes
  are covered only through the general graph theorem (item 4).
* Floating point: all arithmetic is real; `route.js` uses float32 coordinates and doubles.
* Studio's temperature model (`temperatureSteps`) is a graph relaxation (a diffusion-like
  smoother), not an eikonal front; nothing here applies to it.
-/

namespace FireEgress.Eikonal

open Real Set Filter Topology
open scoped ENNReal NNReal Gradient

/-! ## 1. One ignition point, constant speed -/

section PointSource

variable {E : Type*} [NormedAddCommGroup E]

/-- The arrival time of a front spreading from `x₀` at constant speed `F` in free space:
`T x = ‖x - x₀‖ / F` (`frontArrival` on a graph with straight links and no walls, and the
radial model before banding). -/
noncomputable def pointArrival (F : ℝ) (x0 x : E) : ℝ := ‖x - x0‖ / F

/-- The distance to `x₀` is differentiable at every `x ≠ x₀`, with derivative
`v ↦ ⟪(x - x₀)/‖x - x₀‖, v⟫`: the unit vector pointing away from the source. -/
lemma hasFDerivAt_norm_sub [InnerProductSpace ℝ E] {x0 x : E} (hx : x ≠ x0) :
    HasFDerivAt (fun y => ‖y - x0‖) (innerSL ℝ (‖x - x0‖⁻¹ • (x - x0))) x := by
  have hv : x - x0 ≠ 0 := sub_ne_zero.mpr hx
  have hn : 0 < ‖x - x0‖ := norm_pos_iff.mpr hv
  have h1 : HasFDerivAt (fun y => ‖y - x0‖ ^ 2)
      (2 • (innerSL ℝ (x - x0)).comp (ContinuousLinearMap.id ℝ E)) x :=
    ((hasFDerivAt_id x).sub_const x0).norm_sq
  have h2 := h1.sqrt (by positivity)
  have e : (fun y => √(‖y - x0‖ ^ 2)) = fun y => ‖y - x0‖ := by
    funext y; exact Real.sqrt_sq (norm_nonneg _)
  rw [e] at h2
  convert h2 using 1
  ext y
  simp [innerSL_apply_apply, Real.sqrt_sq (norm_nonneg _)]
  field_simp

/-- Away from the ignition point the free-space arrival time is differentiable, with
derivative `⟪(x - x₀)/(F ‖x - x₀‖), ·⟫`: the front moves radially outward. -/
theorem pointArrival_hasFDerivAt [InnerProductSpace ℝ E] {F : ℝ} {x0 x : E} (hx : x ≠ x0) :
    HasFDerivAt (pointArrival F x0) (innerSL ℝ ((F * ‖x - x0‖)⁻¹ • (x - x0))) x := by
  have h := (hasFDerivAt_norm_sub hx).mul_const F⁻¹
  have e : pointArrival F x0 = fun y => ‖y - x0‖ * F⁻¹ := by
    funext y; simp [pointArrival, div_eq_mul_inv]
  rw [e]
  convert h using 1
  ext y
  simp [innerSL_apply_apply]
  ring

/-- FD-28, point source. The free-space arrival time solves the eikonal equation
classically away from the source: the derivative has operator norm `1 / F`, i.e.
`F |∇T| = 1` at every `x ≠ x₀`. -/
theorem pointArrival_eikonal [InnerProductSpace ℝ E] {F : ℝ} (hF : 0 < F) {x0 x : E}
    (hx : x ≠ x0) : ‖fderiv ℝ (pointArrival F x0) x‖ = 1 / F := by
  have hn : 0 < ‖x - x0‖ := norm_pos_iff.mpr (sub_ne_zero.mpr hx)
  rw [(pointArrival_hasFDerivAt hx).fderiv, innerSL_apply_norm, norm_smul, norm_inv,
    norm_mul, Real.norm_of_nonneg hF.le, norm_norm]
  field_simp

/-- FD-28 with Mathlib's `gradient` (complete inner product spaces, e.g. the plan
`EuclideanSpace ℝ (Fin 2)` or 3-D space): `∇T x = (x - x₀)/(F ‖x - x₀‖)`, of length
exactly `1 / F`. -/
theorem pointArrival_gradient [InnerProductSpace ℝ E] [CompleteSpace E] {F : ℝ} (hF : 0 < F)
    {x0 x : E} (hx : x ≠ x0) :
    ∇ (pointArrival F x0) x = (F * ‖x - x0‖)⁻¹ • (x - x0) ∧
      ‖∇ (pointArrival F x0) x‖ = 1 / F := by
  have hn : 0 < ‖x - x0‖ := norm_pos_iff.mpr (sub_ne_zero.mpr hx)
  have h : HasGradientAt (pointArrival F x0) ((F * ‖x - x0‖)⁻¹ • (x - x0)) x := by
    rw [hasGradientAt_iff_hasFDerivAt]
    convert pointArrival_hasFDerivAt (F := F) hx using 1
    ext z
    rfl
  refine ⟨h.gradient, ?_⟩
  rw [h.gradient, norm_smul, norm_inv, norm_mul, Real.norm_of_nonneg hF.le, norm_norm]
  field_simp

/-- FD-28 on Studio's plan: in `EuclideanSpace ℝ (Fin 2)` the free-space front satisfies
`|∇T| = 1/F` at every point other than the ignition point. -/
theorem plane_eikonal {F : ℝ} (hF : 0 < F) {x0 x : EuclideanSpace ℝ (Fin 2)} (hx : x ≠ x0) :
    ‖∇ (pointArrival F x0) x‖ = 1 / F :=
  (pointArrival_gradient hF hx).2

/-- The eikonal equation holds only away from the source: the cone `‖x - x₀‖ / F` has no
derivative at its tip (in any non-trivial space). -/
theorem pointArrival_not_differentiableAt_source [InnerProductSpace ℝ E] [Nontrivial E]
    {F : ℝ} (hF : 0 < F) (x0 : E) : ¬ DifferentiableAt ℝ (pointArrival F x0) x0 := by
  intro h
  have hg : DifferentiableAt ℝ (pointArrival F x0) ((fun y : E => y + x0) 0) := by
    simpa using h
  have h2 : DifferentiableAt ℝ (fun y : E => F * pointArrival F x0 (y + x0)) 0 :=
    (hg.comp (0 : E) (differentiableAt_id.add_const x0)).const_mul F
  have e : (fun y : E => F * pointArrival F x0 (y + x0)) = fun y => ‖y‖ := by
    funext y; simp only [pointArrival, add_sub_cancel_right]; field_simp
  rw [e] at h2
  exact not_differentiableAt_norm_zero E h2

/-- The free-space front is `(1/F)`-Lipschitz: two points `d` metres apart ignite at most
`d / F` seconds apart. -/
theorem pointArrival_lipschitz {F : ℝ} (hF : 0 < F) (x0 x y : E) :
    |pointArrival F x0 x - pointArrival F x0 y| ≤ ‖x - y‖ / F := by
  unfold pointArrival
  rw [← sub_div, abs_div, abs_of_pos hF]
  apply div_le_div_of_nonneg_right _ hF.le
  have := abs_norm_sub_norm_le (x - x0) (y - x0)
  rwa [sub_sub_sub_cancel_right] at this

/-- The same, as Mathlib's `LipschitzWith` with constant `1/F`. -/
theorem pointArrival_lipschitzWith {F : ℝ} (hF : 0 < F) (x0 : E) :
    LipschitzWith ⟨F⁻¹, (inv_pos.mpr hF).le⟩ (pointArrival F x0) := by
  apply LipschitzWith.of_dist_le_mul
  intro x y
  show dist _ _ ≤ F⁻¹ * dist x y
  rw [Real.dist_eq, dist_eq_norm, ← div_eq_inv_mul]
  exact pointArrival_lipschitz hF x0 x y

/-- The length of the polyline from `p` through the waypoints `qs`. -/
noncomputable def plLength : E → List E → ℝ
  | _, [] => 0
  | p, q :: qs => ‖q - p‖ + plLength q qs

/-- The end point of the polyline from `p` through the waypoints `qs`. -/
def plEnd : E → List E → E
  | p, [] => p
  | _, q :: qs => plEnd q qs

/-- A polyline is never shorter than the straight line between its ends (the triangle
inequality, by induction on the waypoints). -/
theorem norm_le_plLength (p : E) (qs : List E) : ‖plEnd p qs - p‖ ≤ plLength p qs := by
  induction qs generalizing p with
  | nil => simp [plEnd, plLength]
  | cons q qs ih =>
    simp only [plEnd, plLength]
    calc ‖plEnd q qs - p‖ = ‖(plEnd q qs - q) + (q - p)‖ := by congr 1; abel
      _ ≤ ‖plEnd q qs - q‖ + ‖q - p‖ := norm_add_le _ _
      _ ≤ ‖q - p‖ + plLength q qs := by linarith [ih q]

/-- The free-space arrival time is the least travel time, at speed `F`, over all polylines
from the source to `x`; the straight segment attains it. So the radial formula is the
first-arrival time, not merely an estimate of it. -/
theorem pointArrival_isLeast_pl {F : ℝ} (hF : 0 < F) (x0 x : E) :
    IsLeast {τ | ∃ qs : List E, plEnd x0 qs = x ∧ τ = plLength x0 qs / F}
      (pointArrival F x0 x) := by
  refine ⟨⟨[x], by simp [plEnd], by simp [plLength, pointArrival]⟩, ?_⟩
  rintro τ ⟨qs, rfl, rfl⟩
  exact div_le_div_of_nonneg_right (norm_le_plLength x0 qs) hF.le

/-- The travel time at speed `F` along a curve `γ` on `[a, b]`: its length (Mathlib's total
variation `eVariationOn`, `⊤` for a non-rectifiable curve) divided by `F`. -/
noncomputable def curveTime (F : ℝ) (γ : ℝ → E) (a b : ℝ) : ℝ≥0∞ :=
  eVariationOn γ (Icc a b) / ENNReal.ofReal F

/-- No curve from the source reaches `x` faster than the free-space front: any curve,
rectifiable or not, is at least as long as the straight line between its ends. -/
theorem pointArrival_le_curveTime {F : ℝ} (hF : 0 < F) {x0 x : E} {γ : ℝ → E} {a b : ℝ}
    (hab : a ≤ b) (h0 : γ a = x0) (h1 : γ b = x) :
    ENNReal.ofReal (pointArrival F x0 x) ≤ curveTime F γ a b := by
  unfold pointArrival curveTime
  rw [ENNReal.ofReal_div_of_pos hF]
  gcongr
  have := eVariationOn.edist_le γ (left_mem_Icc.2 hab) (right_mem_Icc.2 hab)
  rwa [h0, h1, edist_dist, dist_comm, dist_eq_norm] at this

/-- The straight segment `t ↦ x₀ + t (x - x₀)`, `t ∈ [0, 1]`, takes exactly the free-space
arrival time. -/
theorem curveTime_segment [NormedSpace ℝ E] {F : ℝ} (hF : 0 < F) (x0 x : E) :
    curveTime F (fun t : ℝ => x0 + t • (x - x0)) 0 1 = ENNReal.ofReal (pointArrival F x0 x) := by
  apply le_antisymm
  · unfold curveTime pointArrival
    rw [ENNReal.ofReal_div_of_pos hF]
    gcongr
    have hL : LipschitzWith ‖x - x0‖₊ (fun t : ℝ => x0 + t • (x - x0)) := by
      apply LipschitzWith.of_dist_le_mul
      intro s t
      rw [dist_eq_norm, dist_eq_norm, add_sub_add_left_eq_sub, ← sub_smul, norm_smul,
        coe_nnnorm, mul_comm]
    have h1 := (hL.lipschitzOnWith (s := univ)).comp_eVariationOn_le (g := id)
      (s := Icc (0 : ℝ) 1) (mapsTo_univ _ _)
    have h2 : eVariationOn (id : ℝ → ℝ) (Icc 0 1) ≤ ENNReal.ofReal 1 := by
      have := (monotoneOn_id (s := Icc (0 : ℝ) 1)).eVariationOn_eq
        (left_mem_Icc.2 zero_le_one) (right_mem_Icc.2 zero_le_one)
      rw [inter_self] at this
      rw [this]
      simp
    calc eVariationOn (fun t : ℝ => x0 + t • (x - x0)) (Icc 0 1)
        ≤ ‖x - x0‖₊ * eVariationOn id (Icc (0 : ℝ) 1) := h1
      _ ≤ ‖x - x0‖₊ * ENNReal.ofReal 1 := by gcongr
      _ = ENNReal.ofReal ‖x - x0‖ := by
        rw [ENNReal.ofReal_one, mul_one, ofReal_norm, enorm_eq_nnnorm]
  · exact pointArrival_le_curveTime hF zero_le_one (by simp) (by simp)

/-- FD-28, minimal-time form. The free-space arrival time is the least travel time at speed
`F` over all curves from the source to `x` (with curve length as total variation), and the
straight line attains it. -/
theorem pointArrival_isLeast_curve [NormedSpace ℝ E] {F : ℝ} (hF : 0 < F) (x0 x : E) :
    IsLeast {τ | ∃ (γ : ℝ → E) (a b : ℝ), a ≤ b ∧ γ a = x0 ∧ γ b = x ∧ τ = curveTime F γ a b}
      (ENNReal.ofReal (pointArrival F x0 x)) := by
  refine ⟨⟨fun t : ℝ => x0 + t • (x - x0), 0, 1, zero_le_one, by simp, by simp,
    (curveTime_segment hF x0 x).symm⟩, ?_⟩
  rintro τ ⟨γ, a, b, hab, h0, h1, rfl⟩
  exact pointArrival_le_curveTime hF hab h0 h1

end PointSource

/-! ## 2. Several ignition points -/

section MultiSource

variable {E : Type*} [NormedAddCommGroup E] {ι : Type*}

/-- The arrival time of fronts started together at the ignition points `src i`, `i ∈ s`, all
at speed `F`: `T x = minᵢ ‖x - src i‖ / F`. -/
noncomputable def multiArrival (s : Finset ι) (hs : s.Nonempty) (F : ℝ) (src : ι → E) (x : E) :
    ℝ :=
  s.inf' hs (fun i => pointArrival F (src i) x)

/-- Each source's own front bounds the combined front from above. -/
theorem multiArrival_le (s : Finset ι) (hs : s.Nonempty) (F : ℝ) (src : ι → E) (x : E) {i : ι}
    (hi : i ∈ s) : multiArrival s hs F src x ≤ pointArrival F (src i) x :=
  Finset.inf'_le _ hi

/-- The multi-source front is `(1/F)`-Lipschitz: points `d` metres apart ignite at most
`d / F` seconds apart, however many fires there are. -/
theorem multiArrival_lipschitz {s : Finset ι} (hs : s.Nonempty) {F : ℝ} (hF : 0 < F)
    (src : ι → E) (x y : E) :
    |multiArrival s hs F src x - multiArrival s hs F src y| ≤ ‖x - y‖ / F := by
  have key : ∀ x y : E, multiArrival s hs F src x ≤ multiArrival s hs F src y + ‖x - y‖ / F := by
    intro x y
    obtain ⟨j, hj, hjy⟩ := Finset.exists_mem_eq_inf' hs (fun i => pointArrival F (src i) y)
    have h1 := multiArrival_le s hs F src x hj
    have h2 := pointArrival_lipschitz hF (src j) x y
    have h3 : multiArrival s hs F src y = pointArrival F (src j) y := hjy
    rw [abs_le] at h2
    linarith [h2.2]
  rw [abs_le]
  constructor
  · have := key y x
    rw [norm_sub_rev] at this
    linarith
  · linarith [key x y]

/-- The same, as Mathlib's `LipschitzWith` with constant `1/F`. -/
theorem multiArrival_lipschitzWith {s : Finset ι} (hs : s.Nonempty) {F : ℝ} (hF : 0 < F)
    (src : ι → E) : LipschitzWith ⟨F⁻¹, (inv_pos.mpr hF).le⟩ (multiArrival s hs F src) := by
  apply LipschitzWith.of_dist_le_mul
  intro x y
  show dist _ _ ≤ F⁻¹ * dist x y
  rw [Real.dist_eq, dist_eq_norm, ← div_eq_inv_mul]
  exact multiArrival_lipschitz hs hF src x y

/-- Fermat's rule for touching functions: if `g ≤ f` everywhere, they touch at `x`, and both
are differentiable there, their derivatives at `x` agree (`f - g` has a minimum at `x`). -/
lemma hasFDerivAt_eq_of_touch {G : Type*} [NormedAddCommGroup G] [NormedSpace ℝ G]
    {f g : G → ℝ} {x : G} {f' g' : G →L[ℝ] ℝ}
    (hf : HasFDerivAt f f' x) (hg : HasFDerivAt g g' x) (hle : ∀ y, g y ≤ f y)
    (heq : g x = f x) : f' = g' := by
  have hmin : IsLocalMin (fun y => f y - g y) x :=
    Filter.Eventually.of_forall (fun y => by simp only; linarith [hle y])
  exact sub_eq_zero.mp (hmin.hasFDerivAt_eq_zero (hf.sub hg))

/-- FD-28 for several ignition points. Wherever the multi-source front is differentiable
(and the point is not itself a source), `F |∇T| = 1`: the derivative is that of the nearest
source's own front. -/
theorem multiArrival_eikonal [InnerProductSpace ℝ E] {s : Finset ι} (hs : s.Nonempty) {F : ℝ}
    (hF : 0 < F) (src : ι → E) {x : E} (hx : ∀ i ∈ s, x ≠ src i)
    (hd : DifferentiableAt ℝ (multiArrival s hs F src) x) :
    ‖fderiv ℝ (multiArrival s hs F src) x‖ = 1 / F := by
  obtain ⟨j, hj, hjx⟩ := Finset.exists_mem_eq_inf' hs (fun i => pointArrival F (src i) x)
  have hfj := pointArrival_hasFDerivAt (F := F) (hx j hj)
  have e := hasFDerivAt_eq_of_touch hfj hd.hasFDerivAt
    (fun y => multiArrival_le s hs F src y hj) hjx
  rw [← e, ← hfj.fderiv]
  exact pointArrival_eikonal hF (hx j hj)

/-- Where one ignition point is strictly nearest (other indices may repeat the same point),
the multi-source front is differentiable, with the derivative of that source's front; so by
`pointArrival_eikonal` it solves `F |∇T| = 1` there. -/
theorem multiArrival_hasFDerivAt_of_nearest [InnerProductSpace ℝ E] {s : Finset ι}
    (hs : s.Nonempty) {F : ℝ} (hF : 0 < F) (src : ι → E) {x : E} {j : ι} (hj : j ∈ s)
    (hxj : x ≠ src j) (hnear : ∀ i ∈ s, src i = src j ∨ ‖x - src j‖ < ‖x - src i‖) :
    HasFDerivAt (multiArrival s hs F src) (innerSL ℝ ((F * ‖x - src j‖)⁻¹ • (x - src j))) x := by
  apply (pointArrival_hasFDerivAt hxj).congr_of_eventuallyEq
  have hev : ∀ᶠ y in 𝓝 x, ∀ i ∈ s, src i = src j ∨ ‖y - src j‖ < ‖y - src i‖ := by
    rw [Filter.eventually_all_finset]
    intro i hi
    rcases hnear i hi with h | h
    · exact Filter.Eventually.of_forall (fun _ => Or.inl h)
    · exact (ContinuousAt.eventually_lt (by fun_prop) (by fun_prop) h).mono
        (fun _ hy => Or.inr hy)
  filter_upwards [hev] with y hy
  apply le_antisymm (multiArrival_le s hs F src y hj)
  apply Finset.le_inf'
  intro i hi
  rcases hy i hi with h | h
  · simp [pointArrival, h]
  · exact div_le_div_of_nonneg_right h.le hF.le

/-- Where the fronts of two distinct ignition points meet (both attain the minimum), the
multi-source arrival time has no derivative. The classical eikonal equation fails on these
meeting lines; this is the reason the theory (Crandall and Lions) uses viscosity solutions. -/
theorem multiArrival_not_differentiableAt_tie [InnerProductSpace ℝ E] {s : Finset ι}
    (hs : s.Nonempty) {F : ℝ} (hF : 0 < F) (src : ι → E) {x : E} {j k : ι} (hj : j ∈ s)
    (hk : k ∈ s) (hjk : src j ≠ src k)
    (htj : multiArrival s hs F src x = pointArrival F (src j) x)
    (htk : multiArrival s hs F src x = pointArrival F (src k) x) :
    ¬ DifferentiableAt ℝ (multiArrival s hs F src) x := by
  intro hd
  have hjk' : ‖x - src j‖ = ‖x - src k‖ := (div_left_inj' hF.ne').mp (htj.symm.trans htk)
  have hxj : x ≠ src j := by
    intro hxj
    apply hjk
    have : ‖x - src k‖ = 0 := by rw [← hjk', hxj, sub_self, norm_zero]
    rw [norm_eq_zero, sub_eq_zero] at this
    rw [← hxj, this]
  have hxk : x ≠ src k := by
    intro hxk
    apply hjk
    have : ‖x - src j‖ = 0 := by rw [hjk', hxk, sub_self, norm_zero]
    rw [norm_eq_zero, sub_eq_zero] at this
    rw [← this, hxk]
  have ej := hasFDerivAt_eq_of_touch (pointArrival_hasFDerivAt (F := F) hxj) hd.hasFDerivAt
    (fun y => multiArrival_le s hs F src y hj) htj
  have ek := hasFDerivAt_eq_of_touch (pointArrival_hasFDerivAt (F := F) hxk) hd.hasFDerivAt
    (fun y => multiArrival_le s hs F src y hk) htk
  have e := ej.trans ek.symm
  have hu : (F * ‖x - src j‖)⁻¹ • (x - src j) = (F * ‖x - src k‖)⁻¹ • (x - src k) := by
    apply ext_inner_right ℝ
    intro v
    have := congrArg (fun L : E →L[ℝ] ℝ => L v) e
    simp only [innerSL_apply_apply] at this
    exact this
  rw [← hjk'] at hu
  have hn : 0 < ‖x - src j‖ := norm_pos_iff.mpr (sub_ne_zero.mpr hxj)
  have hc : (F * ‖x - src j‖)⁻¹ ≠ 0 := by positivity
  exact hjk (sub_right_injective (smul_right_injective E hc hu))

/-- The multi-source front is the least travel time at speed `F` over polylines starting at
any ignition point: the arrival time from the ignition set. -/
theorem multiArrival_isLeast_pl {s : Finset ι} (hs : s.Nonempty) {F : ℝ} (hF : 0 < F)
    (src : ι → E) (x : E) :
    IsLeast {τ | ∃ i ∈ s, ∃ qs : List E, plEnd (src i) qs = x ∧ τ = plLength (src i) qs / F}
      (multiArrival s hs F src x) := by
  obtain ⟨j, hj, hjx⟩ := Finset.exists_mem_eq_inf' hs (fun i => pointArrival F (src i) x)
  refine ⟨⟨j, hj, [x], by simp [plEnd], ?_⟩, ?_⟩
  · rw [show multiArrival s hs F src x = _ from hjx]
    simp [plLength, pointArrival]
  · rintro τ ⟨i, hi, qs, hqs, rfl⟩
    exact (multiArrival_le s hs F src x hi).trans
      ((pointArrival_isLeast_pl hF (src i) x).2 ⟨qs, hqs, rfl⟩)

end MultiSource

/-! ## 3. The lattice front against the continuum, open floor -/

section LatticeFront

open FireEgress.Lattice

/-- Studio's plan, with the Euclidean norm. -/
abbrev Plane := EuclideanSpace ℝ (Fin 2)

/-- The plan position of lattice node `v` at spacing `h` (in the frame turned to the
building's walls, as `buildLatticeGraph` places it). -/
noncomputable def latPos (h : ℝ) (v : ℤ × ℤ) : Plane := !₂[h * v.1, h * v.2]

/-- The Euclidean distance between two lattice nodes is `h √(a² + b²)` for the integer
offset `(a, b)`. -/
theorem norm_latPos_sub {h : ℝ} (hh : 0 ≤ h) (u v : ℤ × ℤ) :
    ‖latPos h v - latPos h u‖ = h * √(((v - u).1 : ℝ) ^ 2 + ((v - u).2 : ℝ) ^ 2) := by
  rw [EuclideanSpace.norm_eq, Fin.sum_univ_two]
  simp [latPos]
  have e : (h * v.1 - h * u.1) ^ 2 + (h * v.2 - h * u.2) ^ 2 =
      h ^ 2 * (((v.1 : ℝ) - u.1) ^ 2 + ((v.2 : ℝ) - u.2) ^ 2) := by ring
  rw [e, Real.sqrt_mul (sq_nonneg h), Real.sqrt_sq hh]

/-- The graph fire front on the open 8-connected lattice: `frontArrival` at node `v` for
ignition node `v₀`, i.e. (spacing × octile distance) / speed. -/
noncomputable def latArrival (h F : ℝ) (v0 v : ℤ × ℤ) : ℝ := h * octZ (v - v0) / F

/-- `latArrival` is the graph front: the least time, at speed `F`, over all 8-connected
lattice paths from `v₀` to `v` (Lattice.lean's `octile_is_shortest`, scaled). -/
theorem latArrival_isLeast {h F : ℝ} (hh : 0 < h) (hF : 0 < F) (v0 v : ℤ × ℤ) :
    IsLeast {τ | ∃ p : List (ℤ × ℤ), (∀ d ∈ p, IsStep d) ∧ v0 + p.sum = v ∧
      τ = h * pathLen p / F} (latArrival h F v0 v) := by
  obtain ⟨⟨p, hs, hsum, hlen⟩, hlow⟩ := octile_is_shortest (v - v0)
  refine ⟨⟨p, hs, by rw [hsum]; abel, by rw [hlen]; rfl⟩, ?_⟩
  rintro τ ⟨q, hq, hqs, rfl⟩
  have : octZ (v - v0) ≤ pathLen q := hlow ⟨q, hq, by rw [← hqs]; abel, rfl⟩
  unfold latArrival
  gcongr

/-- The graph fire front is never early: at every lattice node it arrives no sooner than
the continuum (eikonal) front `‖v - v₀‖ / F`. -/
theorem latArrival_never_early {h F : ℝ} (hh : 0 ≤ h) (hF : 0 < F) (v0 v : ℤ × ℤ) :
    pointArrival F (latPos h v0) (latPos h v) ≤ latArrival h F v0 v := by
  unfold pointArrival latArrival octZ
  rw [norm_latPos_sub hh]
  gcongr
  exact euclid_le_oct _ _

/-- The graph fire front is at most `K = √(4 - 2√2) ≈ 1.0824` times the continuum front:
at most about 8.2% late on open floor. -/
theorem latArrival_le_K {h F : ℝ} (hh : 0 ≤ h) (hF : 0 < F) (v0 v : ℤ × ℤ) :
    latArrival h F v0 v ≤ K * pointArrival F (latPos h v0) (latPos h v) := by
  unfold pointArrival latArrival octZ
  rw [norm_latPos_sub hh]
  calc h * oct ((v - v0).1 : ℝ) ((v - v0).2 : ℝ) / F
      ≤ h * (K * √(((v - v0).1 : ℝ) ^ 2 + ((v - v0).2 : ℝ) ^ 2)) / F := by
        gcongr; exact oct_le_K_mul_euclid _ _
    _ = K * (h * √(((v - v0).1 : ℝ) ^ 2 + ((v - v0).2 : ℝ) ^ 2) / F) := by ring

/-- Off the ignition node the bound is strict, and below `1.0824`: the graph front is less
than 8.24% late at every lattice node (the worst direction, 22.5°, has irrational slope). -/
theorem latArrival_lt {h F : ℝ} (hh : 0 < h) (hF : 0 < F) (v0 v : ℤ × ℤ) (hv : v ≠ v0) :
    latArrival h F v0 v < K * pointArrival F (latPos h v0) (latPos h v) ∧
      K * pointArrival F (latPos h v0) (latPos h v) <
        1.0824 * pointArrival F (latPos h v0) (latPos h v) := by
  have hne : ((v - v0).1, (v - v0).2) ≠ (0 : ℤ × ℤ) := by
    rw [Prod.mk.eta]; exact sub_ne_zero.mpr hv
  have hlt := oct_lt_K_mul_euclid_on_lattice _ _ hne
  have hpos : 0 < √(((v - v0).1 : ℝ) ^ 2 + ((v - v0).2 : ℝ) ^ 2) := by
    have h1 := oct_nonneg ((v - v0).1 : ℝ) ((v - v0).2 : ℝ)
    by_contra hle
    have h2 : √(((v - v0).1 : ℝ) ^ 2 + ((v - v0).2 : ℝ) ^ 2) = 0 :=
      le_antisymm (not_lt.mp hle) (Real.sqrt_nonneg _)
    rw [h2, mul_zero] at hlt
    linarith
  unfold pointArrival latArrival octZ
  rw [norm_latPos_sub hh.le]
  constructor
  · calc h * oct ((v - v0).1 : ℝ) ((v - v0).2 : ℝ) / F
        < h * (K * √(((v - v0).1 : ℝ) ^ 2 + ((v - v0).2 : ℝ) ^ 2)) / F := by gcongr
      _ = K * (h * √(((v - v0).1 : ℝ) ^ 2 + ((v - v0).2 : ℝ) ^ 2) / F) := by ring
  · have := K_bounds.2
    gcongr

/-- The 8.2% is reached on the lattice: the offset `(12, 5)` (a 5-12-13 triangle, 22.6°)
has octile length `7 + 5√2` against Euclidean length 13, a ratio above `1.0823`. -/
theorem oct_12_5 : oct 12 5 = 7 + 5 * √2 ∧ 1.0823 * 13 < oct 12 5 := by
  have hs := sqrt2_sq
  have h0 : 0 ≤ √2 := Real.sqrt_nonneg _
  have lo : (1.41421 : ℝ) < √2 := by nlinarith
  have e : oct 12 5 = 7 + 5 * √2 := by
    unfold oct
    rw [abs_of_pos (by norm_num : (0 : ℝ) < 12), abs_of_pos (by norm_num : (0 : ℝ) < 5),
      max_eq_left (by norm_num), min_eq_right (by norm_num)]
    ring
  refine ⟨e, ?_⟩
  rw [e]
  linarith

/-- Refining the lattice keeps the same physical point: node `(12k, 5k)` at spacing `c / k`
is the point `(12c, 5c)` for every `k`. -/
theorem latPos_refine (c : ℝ) (k : ℕ) (hk : 0 < k) :
    latPos (c / k) ((12 * k : ℤ), (5 * k : ℤ)) = latPos c (12, 5) := by
  have hk' : (k : ℝ) ≠ 0 := by exact_mod_cast hk.ne'
  unfold latPos
  push_cast
  congr 2 <;> field_simp

/-- No convergence under refinement. At the fixed point `(12c, 5c)` the 8-connected graph
front arrives at `(7 + 5√2) c / F` for every spacing `c / k`, while the continuum front
arrives at `13 c / F`: the 8.2% error does not shrink as the lattice is refined. The
lattice Dijkstra front converges to the octile front, not to the eikonal solution. -/
theorem latArrival_no_refinement {F c : ℝ} (hF : 0 < F) (hc : 0 < c) (k : ℕ) (hk : 0 < k) :
    latArrival (c / k) F 0 ((12 * k : ℤ), (5 * k : ℤ)) = (7 + 5 * √2) * c / F ∧
      pointArrival F (latPos (c / k) 0) (latPos (c / k) ((12 * k : ℤ), (5 * k : ℤ))) =
        13 * c / F := by
  have hk' : (0 : ℝ) < k := by exact_mod_cast hk
  constructor
  · unfold latArrival octZ oct
    simp only [sub_zero]
    push_cast
    rw [abs_of_pos (by positivity : (0 : ℝ) < 12 * k),
      abs_of_pos (by positivity : (0 : ℝ) < 5 * k),
      max_eq_left (by nlinarith), min_eq_right (by nlinarith)]
    field_simp
    ring
  · unfold pointArrival
    rw [norm_latPos_sub (by positivity)]
    simp only [sub_zero]
    push_cast
    have e : (12 * (k : ℝ)) ^ 2 + (5 * k) ^ 2 = (13 * k) ^ 2 := by ring
    rw [e, Real.sqrt_sq (by positivity)]
    field_simp

/-- An axis move of the 4-connected (`rectilinear`) lattice. -/
def IsAxisStep (d : ℤ × ℤ) : Prop := IsStep d ∧ ¬ IsDiag d

/-- The `ℓ¹` (taxicab) length of an integer offset. -/
noncomputable def l1Z (v : ℤ × ℤ) : ℝ := |(v.1 : ℝ)| + |(v.2 : ℝ)|

/-- An axis move has length 1 and taxicab length 1. -/
lemma axisStep_vals {d : ℤ × ℤ} (hd : IsAxisStep d) : stepLen d = 1 ∧ l1Z d = 1 := by
  obtain ⟨⟨hx, hy, hne⟩, hnd⟩ := hd
  obtain ⟨x, y⟩ := d
  simp only at hx hy
  rw [abs_le] at hx hy
  have hx' : x = -1 ∨ x = 0 ∨ x = 1 := by omega
  have hy' : y = -1 ∨ y = 0 ∨ y = 1 := by omega
  rcases hx' with rfl | rfl | rfl <;> rcases hy' with rfl | rfl | rfl <;>
    first
    | exact absurd rfl hne
    | (simp [IsDiag] at hnd; done)
    | simp [stepLen, l1Z]

/-- No 4-connected path is shorter than the taxicab distance between its ends. -/
theorem l1Z_le_pathLen (p : List (ℤ × ℤ)) (hp : ∀ d ∈ p, IsAxisStep d) :
    l1Z p.sum ≤ pathLen p := by
  induction p with
  | nil => simp [l1Z, pathLen]
  | cons d p ih =>
    have hd := axisStep_vals (hp d (List.mem_cons_self ..))
    have ih' := ih (fun e he => hp e (List.mem_cons_of_mem _ he))
    simp only [List.sum_cons, pathLen, List.map_cons] at ih' ⊢
    have tri : l1Z (d + p.sum) ≤ l1Z d + l1Z p.sum := by
      unfold l1Z
      simp only [Prod.fst_add, Prod.snd_add]
      push_cast
      linarith [abs_add_le (d.1 : ℝ) (p.sum.1 : ℝ), abs_add_le (d.2 : ℝ) (p.sum.2 : ℝ)]
    rw [hd.1]
    linarith [hd.2]

/-- Some 4-connected path has length exactly the taxicab distance (`|a|` moves along one
axis, `|b|` along the other). -/
theorem l1Z_attained (v : ℤ × ℤ) :
    ∃ p : List (ℤ × ℤ), (∀ d ∈ p, IsAxisStep d) ∧ p.sum = v ∧ pathLen p = l1Z v := by
  obtain ⟨a, b⟩ := v
  refine ⟨List.replicate a.natAbs (a.sign, 0) ++ List.replicate b.natAbs (0, b.sign), ?_, ?_, ?_⟩
  · intro d hd
    rcases List.mem_append.mp hd with hd | hd
    · obtain ⟨hn, rfl⟩ := List.mem_replicate.mp hd
      have ha : a ≠ 0 := by rintro rfl; simp at hn
      refine ⟨⟨by rw [Int.abs_sign_of_ne_zero ha], by simp, ?_⟩, fun h0 => h0.2 rfl⟩
      intro h0
      have := congrArg Prod.fst h0
      simp [Int.sign_eq_zero_iff_zero, ha] at this
    · obtain ⟨hn, rfl⟩ := List.mem_replicate.mp hd
      have hb : b ≠ 0 := by rintro rfl; simp at hn
      refine ⟨⟨by simp, by rw [Int.abs_sign_of_ne_zero hb], ?_⟩, fun h0 => h0.1 rfl⟩
      intro h0
      have := congrArg Prod.snd h0
      simp [Int.sign_eq_zero_iff_zero, hb] at this
  · simp only [List.sum_append, List.sum_replicate, Prod.smul_mk, nsmul_eq_mul, mul_zero,
      Prod.mk_add_mk, add_zero, zero_add]
    ext
    · simp only
      linear_combination Int.sign_mul_natAbs a
    · simp only
      linear_combination Int.sign_mul_natAbs b
  · have ea : |(a : ℝ)| = (a.natAbs : ℝ) := by rw [Nat.cast_natAbs, Int.cast_abs]
    have eb : |(b : ℝ)| = (b.natAbs : ℝ) := by rw [Nat.cast_natAbs, Int.cast_abs]
    simp only [pathLen, List.map_append, List.map_replicate, List.sum_append,
      List.sum_replicate, nsmul_eq_mul, l1Z, ea, eb]
    have s1 : stepLen (a.sign, 0) = 1 := by simp [stepLen]
    have s2 : stepLen (0, b.sign) = 1 := by simp [stepLen]
    rw [s1, s2]
    ring

/-- On the 4-connected lattice the shortest path between two nodes is the taxicab
distance. -/
theorem rect_isShortest (v : ℤ × ℤ) :
    IsLeast {L : ℝ | ∃ p : List (ℤ × ℤ), (∀ d ∈ p, IsAxisStep d) ∧ p.sum = v ∧ pathLen p = L}
      (l1Z v) := by
  refine ⟨l1Z_attained v, ?_⟩
  rintro L ⟨p, hs, rfl, rfl⟩
  exact l1Z_le_pathLen p hs

/-- The taxicab distance is never shorter than the straight line. -/
theorem euclid_le_l1 (x y : ℝ) : √(x ^ 2 + y ^ 2) ≤ |x| + |y| := by
  have hxy := mul_nonneg (abs_nonneg x) (abs_nonneg y)
  have key : x ^ 2 + y ^ 2 ≤ (|x| + |y|) ^ 2 := by
    nlinarith [sq_abs x, sq_abs y]
  calc √(x ^ 2 + y ^ 2) ≤ √((|x| + |y|) ^ 2) := Real.sqrt_le_sqrt key
    _ = |x| + |y| := Real.sqrt_sq (by positivity)

/-- The taxicab distance is at most `√2` times the straight line. -/
theorem l1_le_sqrt2_mul_euclid (x y : ℝ) : |x| + |y| ≤ √2 * √(x ^ 2 + y ^ 2) := by
  rw [← Real.sqrt_mul (by norm_num)]
  have : (|x| + |y|) ^ 2 ≤ 2 * (x ^ 2 + y ^ 2) := by
    nlinarith [sq_nonneg (|x| - |y|), sq_abs x, sq_abs y]
  exact le_trans (le_abs_self _) (Real.abs_le_sqrt this)

/-- The factor `√2` is attained, on the diagonal `(1, 1)`. -/
theorem l1_sqrt2_attained : |(1 : ℝ)| + |1| = √2 * √(1 ^ 2 + 1 ^ 2) := by
  norm_num

/-- The graph fire front on the open 4-connected (`rectilinear`) lattice. -/
noncomputable def rectArrival (h F : ℝ) (v0 v : ℤ × ℤ) : ℝ := h * l1Z (v - v0) / F

/-- `rectArrival` is the least time over 4-connected lattice paths. -/
theorem rectArrival_isLeast {h F : ℝ} (hh : 0 < h) (hF : 0 < F) (v0 v : ℤ × ℤ) :
    IsLeast {τ | ∃ p : List (ℤ × ℤ), (∀ d ∈ p, IsAxisStep d) ∧ v0 + p.sum = v ∧
      τ = h * pathLen p / F} (rectArrival h F v0 v) := by
  obtain ⟨⟨p, hs, hsum, hlen⟩, hlow⟩ := rect_isShortest (v - v0)
  refine ⟨⟨p, hs, by rw [hsum]; abel, by rw [hlen]; rfl⟩, ?_⟩
  rintro τ ⟨q, hq, hqs, rfl⟩
  have : l1Z (v - v0) ≤ pathLen q := hlow ⟨q, hq, by rw [← hqs]; abel, rfl⟩
  unfold rectArrival
  gcongr

/-- On the 4-connected lattice the graph front is never early and at most `√2` (41%) late. -/
theorem rectArrival_bounds {h F : ℝ} (hh : 0 ≤ h) (hF : 0 < F) (v0 v : ℤ × ℤ) :
    pointArrival F (latPos h v0) (latPos h v) ≤ rectArrival h F v0 v ∧
      rectArrival h F v0 v ≤ √2 * pointArrival F (latPos h v0) (latPos h v) := by
  unfold pointArrival rectArrival l1Z
  rw [norm_latPos_sub hh]
  constructor
  · gcongr
    exact euclid_le_l1 _ _
  · calc h * (|((v - v0).1 : ℝ)| + |((v - v0).2 : ℝ)|) / F
        ≤ h * (√2 * √(((v - v0).1 : ℝ) ^ 2 + ((v - v0).2 : ℝ) ^ 2)) / F := by
          gcongr; exact l1_le_sqrt2_mul_euclid _ _
      _ = √2 * (h * √(((v - v0).1 : ℝ) ^ 2 + ((v - v0).2 : ℝ) ^ 2) / F) := by ring

/-- The 41% is attained: at the diagonal neighbour the 4-connected front is exactly `√2`
times the continuum front. -/
theorem rectArrival_diagonal {h F : ℝ} (hh : 0 ≤ h) (v0 : ℤ × ℤ) :
    rectArrival h F v0 (v0 + (1, 1)) =
      √2 * pointArrival F (latPos h v0) (latPos h (v0 + (1, 1))) := by
  unfold pointArrival rectArrival l1Z
  rw [norm_latPos_sub hh, add_sub_cancel_left]
  have := l1_sqrt2_attained
  simp only [Int.cast_one] at this ⊢
  rw [this]
  ring

end LatticeFront

/-! ## 4. Walls: a free region and its geodesic distance -/

section Obstacles

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The polyline from `p` through `qs` stays in the free region `Ω`: every segment lies in
`Ω` (no segment passes through a wall). -/
def PLIn (Ω : Set E) : E → List E → Prop
  | _, [] => True
  | p, q :: qs => segment ℝ p q ⊆ Ω ∧ PLIn Ω q qs

/-- A polyline free in `Ω` is free in any larger region. -/
theorem PLIn.mono {Ω Ω' : Set E} (h : Ω ⊆ Ω') :
    ∀ {p : E} {qs : List E}, PLIn Ω p qs → PLIn Ω' p qs
  | _, [], _ => trivial
  | _, _ :: _, ⟨h1, h2⟩ => ⟨h1.trans h, PLIn.mono h h2⟩

/-- The geodesic (shortest-path) distance from `x` to `y` in the free region `Ω`, over
polylines in `Ω`; `⊤` when no polyline in `Ω` joins them. -/
noncomputable def geodDist (Ω : Set E) (x y : E) : ℝ≥0∞ :=
  ⨅ (qs : List E) (_ : PLIn Ω x qs ∧ plEnd x qs = y), ENNReal.ofReal (plLength x qs)

/-- The geodesic distance is never shorter than the straight line. -/
theorem ofReal_norm_le_geodDist (Ω : Set E) (x y : E) :
    ENNReal.ofReal ‖y - x‖ ≤ geodDist Ω x y := by
  unfold geodDist
  refine le_iInf₂ (fun qs hqs => ?_)
  rw [← hqs.2]
  exact ENNReal.ofReal_le_ofReal (norm_le_plLength x qs)

/-- Any free polyline bounds the geodesic distance from above. -/
theorem geodDist_le_plLength {Ω : Set E} {x : E} {qs : List E} (h : PLIn Ω x qs) :
    geodDist Ω x (plEnd x qs) ≤ ENNReal.ofReal (plLength x qs) :=
  iInf₂_le qs ⟨h, rfl⟩

/-- More walls (a smaller free region) can only lengthen the geodesic distance. -/
theorem geodDist_anti {Ω Ω' : Set E} (h : Ω ⊆ Ω') (x y : E) :
    geodDist Ω' x y ≤ geodDist Ω x y :=
  le_iInf₂ fun qs hqs => iInf₂_le qs ⟨hqs.1.mono h, hqs.2⟩

/-- Where `x` sees `y` (the segment is free) the geodesic distance is the straight line. -/
theorem geodDist_of_segment {Ω : Set E} {x y : E} (h : segment ℝ x y ⊆ Ω) :
    geodDist Ω x y = ENNReal.ofReal ‖y - x‖ := by
  refine le_antisymm ?_ (ofReal_norm_le_geodDist Ω x y)
  have := geodDist_le_plLength (Ω := Ω) (x := x) (qs := [y]) ⟨h, trivial⟩
  simpa [plLength, plEnd] using this

/-- On open floor (no walls) the geodesic distance is the straight line. -/
theorem geodDist_univ (x y : E) : geodDist univ x y = ENNReal.ofReal ‖y - x‖ :=
  geodDist_of_segment (subset_univ _)

/-- The continuum arrival time with walls: geodesic distance in the free region over the
speed (`⊤` where the fire cannot reach). -/
noncomputable def geodArrival (Ω : Set E) (F : ℝ) (x0 x : E) : ℝ≥0∞ :=
  geodDist Ω x0 x / ENNReal.ofReal F

/-- With walls the continuum front is never earlier than the free-space front. -/
theorem geodArrival_never_early (Ω : Set E) {F : ℝ} (hF : 0 < F) (x0 x : E) :
    ENNReal.ofReal (pointArrival F x0 x) ≤ geodArrival Ω F x0 x := by
  unfold pointArrival geodArrival
  rw [ENNReal.ofReal_div_of_pos hF]
  gcongr
  exact ofReal_norm_le_geodDist Ω x0 x

/-- Walls only delay: shrinking the free region never makes the continuum front earlier. -/
theorem geodArrival_walls_delay {Ω Ω' : Set E} (h : Ω ⊆ Ω') (F : ℝ) (x0 x : E) :
    geodArrival Ω' F x0 x ≤ geodArrival Ω F x0 x := by
  unfold geodArrival
  gcongr
  exact geodDist_anti h x0 x

/-- On open floor the walled arrival time is the free-space one. -/
theorem geodArrival_open_floor {F : ℝ} (hF : 0 < F) (x0 x : E) :
    geodArrival univ F x0 x = ENNReal.ofReal (pointArrival F x0 x) := by
  unfold geodArrival pointArrival
  rw [geodDist_univ, ENNReal.ofReal_div_of_pos hF]

omit [NormedAddCommGroup E] [NormedSpace ℝ E] in
/-- Appending a waypoint moves the polyline's end to it. -/
lemma plEnd_append (p : E) (qs : List E) (q : E) : plEnd p (qs ++ [q]) = q := by
  induction qs generalizing p with
  | nil => rfl
  | cons r qs ih => exact ih r

omit [NormedSpace ℝ E] in
/-- Appending a waypoint adds the last segment's length. -/
lemma plLength_append (p : E) (qs : List E) (q : E) :
    plLength p (qs ++ [q]) = plLength p qs + ‖q - plEnd p qs‖ := by
  induction qs generalizing p with
  | nil => simp [plLength, plEnd]
  | cons r qs ih =>
    simp only [List.cons_append, plLength, plEnd, ih r]
    ring

/-- Appending a free segment keeps a polyline free. -/
lemma PLIn_append {Ω : Set E} (p : E) (qs : List E) (q : E) (h : PLIn Ω p qs)
    (hq : segment ℝ (plEnd p qs) q ⊆ Ω) : PLIn Ω p (qs ++ [q]) := by
  induction qs generalizing p with
  | nil => exact ⟨hq, trivial⟩
  | cons r qs ih => exact ⟨h.1, ih r h.2 hq⟩

/-- A graph walk is a polyline. If every link `u → v` of the graph is the straight segment
between the nodes' positions, lies in the free region, and takes at least its length / F,
then every walk of cost `c` traces a free polyline of time at most `c`. -/
theorem walkCost_polyline {V : Type*} (pos : V → E) (Ω : Set E) {F : ℝ}
    (G : V → V → ℝ → Prop)
    (hG : ∀ u v a, G u v a → segment ℝ (pos u) (pos v) ⊆ Ω ∧ ‖pos v - pos u‖ / F ≤ a)
    {s v : V} {c : ℝ} (hw : AStar.WalkCost G s v c) :
    ∃ qs : List E, PLIn Ω (pos s) qs ∧ plEnd (pos s) qs = pos v ∧
      plLength (pos s) qs / F ≤ c := by
  induction hw with
  | nil => exact ⟨[], trivial, rfl, by simp [plLength]⟩
  | @snoc u w c a _ hE ih =>
    obtain ⟨qs, hin, hend, hlen⟩ := ih
    obtain ⟨hseg, ha⟩ := hG u w a hE
    refine ⟨qs ++ [pos w], PLIn_append _ _ _ hin (hend ▸ hseg), plEnd_append _ _ _, ?_⟩
    rw [plLength_append, hend, add_div]
    linarith

/-- Walls can only delay the graph front. For any navigation graph whose links are free
straight segments weighted by at least length / F (Studio's lattice links, wall-face links
and door links alike, when their segments avoid the walls), the cost of every walk from the
ignition node is at least the walled continuum time `d_Ω / F`, which is at least the
free-space time `‖·‖ / F`. -/
theorem graph_front_never_early {V : Type*} (pos : V → E) (Ω : Set E) {F : ℝ} (hF : 0 < F)
    (G : V → V → ℝ → Prop)
    (hG : ∀ u v a, G u v a → segment ℝ (pos u) (pos v) ⊆ Ω ∧ ‖pos v - pos u‖ / F ≤ a)
    {s v : V} {c : ℝ} (hw : AStar.WalkCost G s v c) :
    ENNReal.ofReal (pointArrival F (pos s) (pos v)) ≤ geodArrival Ω F (pos s) (pos v) ∧
      geodArrival Ω F (pos s) (pos v) ≤ ENNReal.ofReal c ∧
      pointArrival F (pos s) (pos v) ≤ c := by
  obtain ⟨qs, hin, hend, hlen⟩ := walkCost_polyline pos Ω G hG hw
  refine ⟨geodArrival_never_early Ω hF _ _, ?_, ?_⟩
  · unfold geodArrival
    calc geodDist Ω (pos s) (pos v) / ENNReal.ofReal F
        ≤ ENNReal.ofReal (plLength (pos s) qs) / ENNReal.ofReal F := by
          gcongr
          rw [← hend]
          exact geodDist_le_plLength hin
      _ = ENNReal.ofReal (plLength (pos s) qs / F) := (ENNReal.ofReal_div_of_pos hF).symm
      _ ≤ ENNReal.ofReal c := ENNReal.ofReal_le_ofReal hlen
  · have := norm_le_plLength (pos s) qs
    rw [hend] at this
    unfold pointArrival
    exact le_trans (div_le_div_of_nonneg_right this hF.le) hlen

/-- The same for the labels Dijkstra outputs (`distancesFrom`, hence `frontArrival`): once
the queue runs dry, every settled node's label is at least the walled continuum time, so
the graph fire front is never early. -/
theorem dijkstra_front_never_early {V' : Type*} [DecidableEq V'] (pos : V' → E) (Ω : Set E)
    {F : ℝ} (hF : 0 < F) (G : V' → V' → ℝ → Prop)
    (hG : ∀ u v a, G u v a → segment ℝ (pos u) (pos v) ⊆ Ω ∧ ‖pos v - pos u‖ / F ≤ a)
    (s : V') (σ₀ σ : AStar.State V') (h0 : AStar.Init s σ₀)
    (hrun : Relation.ReflTransGen (AStar.Step G (fun _ => 0)) σ₀ σ) (hdry : σ.R ⊆ σ.S)
    (v : V') (hv : v ∈ σ.S) :
    geodArrival Ω F (pos s) (pos v) ≤ ENNReal.ofReal (σ.d v) ∧
      pointArrival F (pos s) (pos v) ≤ σ.d v := by
  have hE : ∀ u v a, G u v a → 0 ≤ a := fun u v a h =>
    le_trans (div_nonneg (norm_nonneg _) hF.le) (hG u v a h).2
  have hw := ((AStar.dijkstra_exhausted G hE s σ₀ σ h0 hrun hdry).2 v hv).1
  obtain ⟨-, h2, h3⟩ := graph_front_never_early pos Ω hF G hG hw
  exact ⟨h2, h3⟩

end Obstacles

/-! ## 5. The discrete eikonal (Bellman) equation -/

section Bellman

open FireEgress.AStar

variable {V : Type*}

/-- With non-negative link times every walk costs at least 0. -/
lemma walkCost_nonneg {G : V → V → ℝ → Prop} (hG : ∀ u v a, G u v a → 0 ≤ a) {s v : V} {c : ℝ}
    (hw : WalkCost G s v c) : 0 ≤ c := by
  induction hw with
  | nil => exact le_rfl
  | snoc _ hE ih => linarith [hG _ _ _ hE]

/-- FD-29. Dijkstra's output satisfies the Bellman (min-plus) equation. When the queue runs
dry: the source is settled with `T s = 0`; every link `u → v` out of a settled `u` lands on a
settled `v` with `T v ≤ T u + a` (an unsettled node, `Infinity` in `route.js`, has no
settled in-neighbour); and every other settled node has an in-link attaining
`T v = T u + a`. -/
theorem dijkstra_bellman [DecidableEq V] (G : V → V → ℝ → Prop) (hG : ∀ u v a, G u v a → 0 ≤ a)
    (s : V) (σ₀ σ : State V) (h0 : Init s σ₀)
    (hrun : Relation.ReflTransGen (Step G (fun _ => 0)) σ₀ σ) (hdry : σ.R ⊆ σ.S) :
    s ∈ σ.S ∧ σ.d s = 0 ∧
      (∀ u v a, G u v a → u ∈ σ.S → v ∈ σ.S ∧ σ.d v ≤ σ.d u + a) ∧
      (∀ v ∈ σ.S, v ≠ s → ∃ u a, G u v a ∧ u ∈ σ.S ∧ σ.d v = σ.d u + a) := by
  obtain ⟨hreach, hopt⟩ := dijkstra_exhausted G hG s σ₀ σ h0 hrun hdry
  have hs : s ∈ σ.S := (hreach s).1 ⟨0, WalkCost.nil⟩
  refine ⟨hs, ?_, ?_, ?_⟩
  · exact le_antisymm ((hopt s hs).2 0 WalkCost.nil) (walkCost_nonneg hG (hopt s hs).1)
  · intro u v a huv hu
    have hw := (hopt u hu).1.snoc huv
    have hv := (hreach v).1 ⟨_, hw⟩
    exact ⟨hv, (hopt v hv).2 _ hw⟩
  · intro v hv hvs
    have hw := (hopt v hv).1
    have hle : ∀ c, WalkCost G s v c → σ.d v ≤ c := (hopt v hv).2
    generalize hc : σ.d v = c at hw hle
    cases hw with
    | nil => exact absurd rfl hvs
    | @snoc u _ c' a hw' huv =>
      have hu := (hreach u).1 ⟨_, hw'⟩
      have h1 := (hopt u hu).2 _ hw'
      have h2 := hle _ ((hopt u hu).1.snoc huv)
      exact ⟨u, a, huv, hu, by linarith⟩

/-- FD-29 in min form: at every settled node other than the source,
`T v = min { T u + a | u → v of time a, u settled }`. -/
theorem dijkstra_bellman_min [DecidableEq V] (G : V → V → ℝ → Prop)
    (hG : ∀ u v a, G u v a → 0 ≤ a) (s : V) (σ₀ σ : State V) (h0 : Init s σ₀)
    (hrun : Relation.ReflTransGen (Step G (fun _ => 0)) σ₀ σ) (hdry : σ.R ⊆ σ.S)
    (v : V) (hv : v ∈ σ.S) (hvs : v ≠ s) :
    IsLeast {t | ∃ u a, G u v a ∧ u ∈ σ.S ∧ t = σ.d u + a} (σ.d v) := by
  obtain ⟨-, -, hsub, hatt⟩ := dijkstra_bellman G hG s σ₀ σ h0 hrun hdry
  obtain ⟨u, a, huv, hu, he⟩ := hatt v hv hvs
  refine ⟨⟨u, a, huv, hu, he⟩, ?_⟩
  rintro t ⟨u', a', huv', hu', rfl⟩
  exact (hsub u' v a' huv' hu').2

/-- A graph whose links of length `ℓ` take time `ℓ / F` (Dijkstra on `ℓ_e / F`). -/
def speedLinks (L : V → V → ℝ → Prop) (F : ℝ) : V → V → ℝ → Prop :=
  fun u v a => ∃ ℓ, L u v ℓ ∧ a = ℓ / F

/-- The discrete eikonal equation. For Dijkstra on `ℓ / F` with positive lengths, at every
settled node other than the source the steepest backward difference quotient over settled
in-neighbours is exactly `1 / F`: `max_u (T v - T u) / ℓ(u, v) = 1 / F`, the graph form of
`|∇T| = 1/F` (one-sided, upwind). -/
theorem dijkstra_discrete_eikonal [DecidableEq V] (L : V → V → ℝ → Prop)
    (hL : ∀ u v ℓ, L u v ℓ → 0 < ℓ) {F : ℝ} (hF : 0 < F) (s : V) (σ₀ σ : State V)
    (h0 : Init s σ₀) (hrun : Relation.ReflTransGen (Step (speedLinks L F) (fun _ => 0)) σ₀ σ)
    (hdry : σ.R ⊆ σ.S) (v : V) (hv : v ∈ σ.S) (hvs : v ≠ s) :
    IsGreatest {q | ∃ u ℓ, L u v ℓ ∧ u ∈ σ.S ∧ q = (σ.d v - σ.d u) / ℓ} (1 / F) := by
  have hG : ∀ u v a, speedLinks L F u v a → 0 ≤ a := by
    rintro u v a ⟨ℓ, hℓ, rfl⟩
    exact div_nonneg (hL u v ℓ hℓ).le hF.le
  obtain ⟨-, -, hsub, hatt⟩ := dijkstra_bellman (speedLinks L F) hG s σ₀ σ h0 hrun hdry
  obtain ⟨u, a, ⟨ℓ, hℓ, rfl⟩, hu, he⟩ := hatt v hv hvs
  have hℓpos := hL u v ℓ hℓ
  refine ⟨⟨u, ℓ, hℓ, hu, ?_⟩, ?_⟩
  · rw [he]
    field_simp
    ring
  · rintro q ⟨u', ℓ', hℓ', hu', rfl⟩
    have hpos := hL u' v ℓ' hℓ'
    have := (hsub u' v (ℓ' / F) ⟨ℓ', hℓ', rfl⟩ hu').2
    rw [div_le_iff₀ hpos]
    have e : 1 / F * ℓ' = ℓ' / F := by ring
    rw [e]
    linarith

/-- A discrete comparison principle: on a finite graph with positive link times, a function
with `T s = 0`, `T v ≤ T u + a` on every link, and an attaining in-link at every other node
is the shortest-path arrival time (the cost of a walk, and at most every walk's cost). So
the Bellman equation determines the graph front uniquely, and any solver of it (Dijkstra,
or a label-correcting loop) yields the same front. -/
theorem bellman_unique [Finite V] (G : V → V → ℝ → Prop) (hpos : ∀ u v a, G u v a → 0 < a)
    (s : V) (T : V → ℝ) (h0 : T s = 0) (hsub : ∀ u v a, G u v a → T v ≤ T u + a)
    (hatt : ∀ v, v ≠ s → ∃ u a, G u v a ∧ T v = T u + a) (v : V) :
    WalkCost G s v (T v) ∧ ∀ c, WalkCost G s v c → T v ≤ c := by
  constructor
  · let r : V → V → Prop := fun a b => T a < T b
    have : IsTrans V r := ⟨fun _ _ _ h1 h2 => lt_trans h1 h2⟩
    have : Std.Irrefl r := ⟨fun _ => lt_irrefl _⟩
    have wf : WellFounded r := Finite.wellFounded_of_trans_of_irrefl r
    refine wf.induction (C := fun v => WalkCost G s v (T v)) v (fun v ih => ?_)
    by_cases hv : v = s
    · subst hv
      exact WalkCost.nil.cast h0.symm
    · obtain ⟨u, a, huv, he⟩ := hatt v hv
      have hlt : r u v := by
        show T u < T v
        linarith [hpos u v a huv]
      exact ((ih u hlt).snoc huv).cast he.symm
  · intro c hw
    induction hw with
    | nil => rw [h0]
    | @snoc u w c a _ huw ih => linarith [hsub u w a huw]

end Bellman

/-! ## The lattice with walls -/

section LatticeWalls

open FireEgress.Lattice

/-- Studio's lattice as a graph for Dijkstra: a link `u → v` exists when `v - u` is a link
of the walkable set `W` (`Lattice.Link`: no corner cutting), and takes time
`h · stepLen / F`. -/
def latLinks (W : Set (ℤ × ℤ)) (h F : ℝ) : ℤ × ℤ → ℤ × ℤ → ℝ → Prop :=
  fun u v a => Link W u (v - u) ∧ a = h * stepLen (v - u) / F

/-- Each lattice move has positive length (1 or `√2`). -/
lemma stepLen_pos (d : ℤ × ℤ) : 0 < stepLen d := by
  unfold stepLen
  split_ifs
  · norm_num
  · positivity

/-- A lattice move's length is the Euclidean length of its offset. -/
lemma stepLen_eq_sqrt (d : ℤ × ℤ) (hd : IsStep d) :
    stepLen d = √((d.1 : ℝ) ^ 2 + (d.2 : ℝ) ^ 2) := by
  obtain ⟨x, y⟩ := d
  obtain ⟨hx, hy, hne⟩ := hd
  simp only at hx hy
  rw [abs_le] at hx hy
  have hx' : x = -1 ∨ x = 0 ∨ x = 1 := by omega
  have hy' : y = -1 ∨ y = 0 ∨ y = 1 := by omega
  rcases hx' with rfl | rfl | rfl <;> rcases hy' with rfl | rfl | rfl <;>
    first
    | exact absurd rfl hne
    | (simp [stepLen] <;> norm_num)

/-- The lattice graph is a `speedLinks` graph with lengths `h · stepLen`. -/
theorem latLinks_eq_speedLinks (W : Set (ℤ × ℤ)) (h F : ℝ) :
    latLinks W h F =
      speedLinks (fun u v ℓ => Link W u (v - u) ∧ ℓ = h * stepLen (v - u)) F := by
  funext u v a
  apply propext
  constructor
  · rintro ⟨hl, rfl⟩
    exact ⟨_, ⟨hl, rfl⟩, rfl⟩
  · rintro ⟨ℓ, ⟨hl, rfl⟩, rfl⟩
    exact ⟨hl, rfl⟩

/-- If every lattice link lies in the free region, each lattice link of `latLinks` is a free
straight segment whose time is exactly its length / F. -/
theorem latLinks_free {W : Set (ℤ × ℤ)} {Ω : Set Plane} {h F : ℝ} (hh : 0 ≤ h)
    (hfree : ∀ u d, Link W u d → segment ℝ (latPos h u) (latPos h (u + d)) ⊆ Ω) :
    ∀ u v a, latLinks W h F u v a →
      segment ℝ (latPos h u) (latPos h v) ⊆ Ω ∧ ‖latPos h v - latPos h u‖ / F ≤ a := by
  rintro u v a ⟨hl, rfl⟩
  have e : u + (v - u) = v := by abel
  refine ⟨e ▸ hfree u (v - u) hl, le_of_eq ?_⟩
  rw [norm_latPos_sub hh, stepLen_eq_sqrt _ hl.1]

/-- With walls, if every lattice link lies in the free region, every lattice walk (so every
Dijkstra label) from the ignition node is at least the walled continuum time `d_Ω / F`,
which is at least the free-space time: the lattice front is never early. -/
theorem lattice_walls_never_early {W : Set (ℤ × ℤ)} {Ω : Set Plane} {h F : ℝ} (hh : 0 ≤ h)
    (hF : 0 < F)
    (hfree : ∀ u d, Link W u d → segment ℝ (latPos h u) (latPos h (u + d)) ⊆ Ω)
    {s v : ℤ × ℤ} {c : ℝ} (hw : AStar.WalkCost (latLinks W h F) s v c) :
    ENNReal.ofReal (pointArrival F (latPos h s) (latPos h v)) ≤
        geodArrival Ω F (latPos h s) (latPos h v) ∧
      geodArrival Ω F (latPos h s) (latPos h v) ≤ ENNReal.ofReal c :=
  let h' := graph_front_never_early (latPos h) Ω hF (latLinks W h F) (latLinks_free hh hfree) hw
  ⟨h'.1, h'.2.1⟩

/-- Extending a lattice walk by one link at its end. -/
lemma isWalk_snoc (W : Set (ℤ × ℤ)) (u : ℤ × ℤ) (p : List (ℤ × ℤ)) (d : ℤ × ℤ)
    (hp : IsWalk W u p) (hd : Link W (u + p.sum) d) : IsWalk W u (p ++ [d]) := by
  induction p generalizing u with
  | nil =>
    simp only [List.sum_nil, add_zero] at hd
    exact ⟨hd, hd.2.2.1⟩
  | cons e p ih =>
    refine ⟨hp.1, ih (u + e) hp.2 ?_⟩
    simpa [add_assoc] using hd

/-- Every walk in the lattice graph is a `Lattice.IsWalk` on `W`, with cost
`h · pathLen / F`. -/
theorem latLinks_walk {W : Set (ℤ × ℤ)} {h F : ℝ} {s v : ℤ × ℤ} {c : ℝ} (hs : s ∈ W)
    (hw : AStar.WalkCost (latLinks W h F) s v c) :
    ∃ p : List (ℤ × ℤ), IsWalk W s p ∧ s + p.sum = v ∧ c = h * pathLen p / F := by
  induction hw with
  | nil => exact ⟨[], hs, by simp, by simp [pathLen]⟩
  | @snoc u w c a _ hE ih =>
    obtain ⟨p, hp, hsum, hc⟩ := ih
    obtain ⟨hl, rfl⟩ := hE
    refine ⟨p ++ [w - u], isWalk_snoc W s p _ hp (hsum ▸ hl), ?_, ?_⟩
    · rw [List.sum_append, List.sum_singleton, ← add_assoc, hsum]; abel
    · rw [hc]
      simp only [pathLen, List.map_append, List.map_cons, List.map_nil, List.sum_append,
        List.sum_cons, List.sum_nil]
      ring

/-- Prepending a link to a walk. -/
lemma walkCost_cons {V : Type*} {G : V → V → ℝ → Prop} {u v w : V} {a c : ℝ} (huv : G u v a)
    (hw : AStar.WalkCost G v w c) : AStar.WalkCost G u w (a + c) := by
  induction hw with
  | nil => exact (AStar.WalkCost.nil.snoc huv).cast (by ring)
  | @snoc x y c b _ hxy ih => exact (ih.snoc hxy).cast (by ring)

/-- Every `Lattice.IsWalk` on `W` is a walk in the lattice graph, of cost
`h · pathLen / F`. -/
theorem isWalk_walkCost {W : Set (ℤ × ℤ)} {h F : ℝ} (u : ℤ × ℤ) (p : List (ℤ × ℤ))
    (hp : IsWalk W u p) : AStar.WalkCost (latLinks W h F) u (u + p.sum) (h * pathLen p / F) := by
  induction p generalizing u with
  | nil =>
    simp only [List.sum_nil, add_zero]
    exact AStar.WalkCost.nil.cast (by simp [pathLen])
  | cons d p ih =>
    obtain ⟨hl, hw⟩ := hp
    have e : u + d - u = d := add_sub_cancel_left u d
    have hl' : latLinks W h F u (u + d) (h * stepLen d / F) := ⟨by rw [e]; exact hl, by rw [e]⟩
    have := walkCost_cons hl' (ih (u + d) hw)
    rw [List.sum_cons, ← add_assoc]
    exact this.cast (by simp only [pathLen, List.map_cons, List.sum_cons]; ring)

/-- With walls (any walkable set `W`), every lattice walk from a walkable ignition node is at
least the open-lattice (octile) time, which is at least the free-space time: removing
cells only removes links (Lattice.lean's `walls_never_shorter`). -/
theorem lattice_walls_delay {W : Set (ℤ × ℤ)} {h F : ℝ} (hh : 0 ≤ h) (hF : 0 < F)
    {s v : ℤ × ℤ} {c : ℝ} (hs : s ∈ W) (hw : AStar.WalkCost (latLinks W h F) s v c) :
    pointArrival F (latPos h s) (latPos h v) ≤ latArrival h F s v ∧ latArrival h F s v ≤ c := by
  refine ⟨latArrival_never_early hh hF s v, ?_⟩
  obtain ⟨p, hp, hsum, rfl⟩ := latLinks_walk hs hw
  have hle := walls_never_shorter W s p hp
  have e : p.sum = v - s := by rw [← hsum]; abel
  rw [e] at hle
  unfold latArrival
  gcongr

/-- Every 8-connected move sequence is a walk on the open lattice. -/
lemma isWalk_univ (u : ℤ × ℤ) (p : List (ℤ × ℤ)) (hp : ∀ d ∈ p, IsStep d) :
    IsWalk univ u p := by
  induction p generalizing u with
  | nil => exact mem_univ u
  | cons d p ih =>
    exact ⟨⟨hp d (List.mem_cons_self ..), mem_univ _, mem_univ _,
      fun _ => ⟨mem_univ _, mem_univ _⟩⟩, ih (u + d) (fun e he => hp e (List.mem_cons_of_mem _ he))⟩

/-- On the open lattice Dijkstra computes exactly the octile front: at any point of any run,
the node it pops next carries `latArrival = h · oct / F`. -/
theorem open_lattice_dijkstra_exact {h F : ℝ} (hh : 0 < h) (hF : 0 < F) (s : ℤ × ℤ)
    (σ₀ σ : AStar.State (ℤ × ℤ)) (h0 : AStar.Init s σ₀)
    (hrun : Relation.ReflTransGen (AStar.Step (latLinks univ h F) (fun _ => 0)) σ₀ σ)
    (u : ℤ × ℤ) (hu : AStar.Selectable (fun _ => 0) σ u) : σ.d u = latArrival h F s u := by
  have hE : ∀ x y a, latLinks univ h F x y a → 0 ≤ a := by
    rintro x y a ⟨-, rfl⟩
    have := stepLen_pos (y - x)
    positivity
  obtain ⟨hreal, hopt⟩ := AStar.dijkstra_correct _ hE s σ₀ σ h0 hrun u hu
  apply le_antisymm
  · obtain ⟨⟨p, hs, hsum, hlen⟩, -⟩ := latArrival_isLeast hh hF s u
    have hw := isWalk_walkCost (h := h) (F := F) s p (isWalk_univ s p hs)
    rw [hsum] at hw
    rw [hlen]
    exact hopt _ hw
  · exact (lattice_walls_delay hh.le hF (mem_univ s) hreal).2

/-- The discrete eikonal equation on Studio's lattice (any walkable set `W`): when Dijkstra
on `h · stepLen / F` runs dry, at every reached node other than the ignition node the
steepest backward difference quotient `(T v - T u) / |uv|` over reached lattice neighbours
is exactly `1 / F`. -/
theorem lattice_discrete_eikonal {W : Set (ℤ × ℤ)} {h F : ℝ} (hh : 0 < h) (hF : 0 < F)
    (s : ℤ × ℤ) (σ₀ σ : AStar.State (ℤ × ℤ)) (h0 : AStar.Init s σ₀)
    (hrun : Relation.ReflTransGen (AStar.Step (latLinks W h F) (fun _ => 0)) σ₀ σ)
    (hdry : σ.R ⊆ σ.S) (v : ℤ × ℤ) (hv : v ∈ σ.S) (hvs : v ≠ s) :
    IsGreatest {q | ∃ u ℓ, (Link W u (v - u) ∧ ℓ = h * stepLen (v - u)) ∧ u ∈ σ.S ∧
      q = (σ.d v - σ.d u) / ℓ} (1 / F) := by
  rw [latLinks_eq_speedLinks] at hrun
  exact dijkstra_discrete_eikonal _ (fun u v ℓ hl => by rw [hl.2]; exact mul_pos hh (stepLen_pos _))
    hF s σ₀ σ h0 hrun hdry v hv hvs

end LatticeWalls

/-! ## 6. Speed: a slower medium never makes the front earlier -/

section Speed

open FireEgress.AStar

variable {V : Type*}

/-- A graph whose links of length `ℓ` take time `ℓ / F u v`, with a speed per link. -/
def edgeSpeedLinks (L : V → V → ℝ → Prop) (F : V → V → ℝ) : V → V → ℝ → Prop :=
  fun u v a => ∃ ℓ, L u v ℓ ∧ a = ℓ / F u v

/-- Faster links make every walk at least as quick: each walk under speeds `F` has a
counterpart under speeds `F' ≥ F` costing no more. -/
theorem walkCost_slower (L : V → V → ℝ → Prop) (hL : ∀ u v ℓ, L u v ℓ → 0 ≤ ℓ)
    (F F' : V → V → ℝ) (hF : ∀ u v, 0 < F u v) (hFF : ∀ u v, F u v ≤ F' u v) {s v : V} {c : ℝ}
    (hw : WalkCost (edgeSpeedLinks L F) s v c) :
    ∃ c', c' ≤ c ∧ WalkCost (edgeSpeedLinks L F') s v c' := by
  induction hw with
  | nil => exact ⟨0, le_rfl, WalkCost.nil⟩
  | @snoc u w c a _ hE ih =>
    obtain ⟨c', hc', hw'⟩ := ih
    obtain ⟨ℓ, hℓ, rfl⟩ := hE
    refine ⟨c' + ℓ / F' u w, ?_, hw'.snoc ⟨ℓ, hℓ, rfl⟩⟩
    have := div_le_div_of_nonneg_left (hL u w ℓ hℓ) (hF u w) (hFF u w)
    linarith

/-- A slower medium never makes the graph front arrive earlier: if every link is at least as
fast under `F'` as under `F`, the arrival time under `F'` is at most that under `F`. -/
theorem arrival_speed_mono (L : V → V → ℝ → Prop) (hL : ∀ u v ℓ, L u v ℓ → 0 ≤ ℓ)
    (F F' : V → V → ℝ) (hF : ∀ u v, 0 < F u v) (hFF : ∀ u v, F u v ≤ F' u v) (s v : V)
    {T T' : ℝ} (hT : IsLeast {c | WalkCost (edgeSpeedLinks L F) s v c} T)
    (hT' : IsLeast {c | WalkCost (edgeSpeedLinks L F') s v c} T') : T' ≤ T := by
  obtain ⟨c', hc', hw'⟩ := walkCost_slower L hL F F' hF hFF hT.1
  exact le_trans (hT'.2 hw') hc'

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The continuum travel time along a path `γ : [0, 1] → E` through a medium of speed
`F(x)`: `∫₀¹ ‖γ'(t)‖ / F(γ t) dt`, as a lower Lebesgue integral (so always defined). -/
noncomputable def travelTime (F : E → ℝ) (γ : ℝ → E) : ℝ≥0∞ :=
  ∫⁻ t in Icc (0 : ℝ) 1, ENNReal.ofReal (‖deriv γ t‖ / F (γ t))

/-- Along any one path, a faster medium never takes longer. -/
theorem travelTime_mono {F F' : E → ℝ} (hF : ∀ x, 0 < F x) (hFF : ∀ x, F x ≤ F' x)
    (γ : ℝ → E) : travelTime F' γ ≤ travelTime F γ :=
  MeasureTheory.lintegral_mono fun _ =>
    ENNReal.ofReal_le_ofReal (div_le_div_of_nonneg_left (norm_nonneg _) (hF _) (hFF _))

/-- The minimal travel time from the ignition set `Γ` to `x` over an admissible family of
paths `P` (`γ 0 ∈ Γ`, `γ 1 = x`). -/
noncomputable def minTravelTime (F : E → ℝ) (P : Set (ℝ → E)) (Γ : Set E) (x : E) : ℝ≥0∞ :=
  ⨅ γ ∈ {γ | γ ∈ P ∧ γ 0 ∈ Γ ∧ γ 1 = x}, travelTime F γ

/-- Variable speed `F(x) > 0`: the minimal travel time is antitone in the speed. A slower
medium never makes the continuum front arrive earlier. -/
theorem minTravelTime_speed_mono {F F' : E → ℝ} (hF : ∀ x, 0 < F x) (hFF : ∀ x, F x ≤ F' x)
    (P : Set (ℝ → E)) (Γ : Set E) (x : E) :
    minTravelTime F' P Γ x ≤ minTravelTime F P Γ x :=
  iInf₂_mono fun γ _ => travelTime_mono hF hFF γ

/-- More ignition points never make the front later. -/
theorem minTravelTime_ignition_mono (F : E → ℝ) (P : Set (ℝ → E)) {Γ Γ' : Set E} (h : Γ ⊆ Γ')
    (x : E) : minTravelTime F P Γ' x ≤ minTravelTime F P Γ x :=
  le_iInf₂ fun γ hγ => iInf₂_le γ ⟨hγ.1, h hγ.2.1, hγ.2.2⟩

/-- Consistency with section 1: at constant speed the straight segment's travel time is the
free-space arrival time. -/
theorem travelTime_segment (F : ℝ) (x0 x : E) :
    travelTime (fun _ => F) (fun t : ℝ => x0 + t • (x - x0)) =
      ENNReal.ofReal (pointArrival F x0 x) := by
  have hd : ∀ t : ℝ, deriv (fun t : ℝ => x0 + t • (x - x0)) t = x - x0 := by
    intro t
    have := ((hasDerivAt_id t).smul_const (x - x0)).const_add x0
    simpa using this.deriv
  unfold travelTime pointArrival
  simp only [hd]
  rw [MeasureTheory.setLIntegral_const, Real.volume_Icc]
  simp

omit [NormedSpace ℝ E] in
/-- Studio's radial model (`radialTimeline`) puts a node in band `⌊‖x - x₀‖ / ℓ̃⌋`: the
free-space arrival time at one median edge `ℓ̃` per step, rounded down, so within one step
below it. -/
theorem radial_band (ℓ : ℝ) (x0 x : E) :
    (⌊pointArrival ℓ x0 x⌋ : ℝ) ≤ pointArrival ℓ x0 x ∧
      pointArrival ℓ x0 x < ⌊pointArrival ℓ x0 x⌋ + 1 :=
  ⟨Int.floor_le _, Int.lt_floor_add_one _⟩

/-- The radial model is never later than the walled continuum front at the same speed: it
ignores walls, so it is the early (conservative-for-fire) bound, not the arrival time. -/
theorem radial_never_late (Ω : Set E) {ℓ : ℝ} (hℓ : 0 < ℓ) (x0 x : E) :
    ENNReal.ofReal (⌊pointArrival ℓ x0 x⌋ : ℝ) ≤ geodArrival Ω ℓ x0 x :=
  le_trans (ENNReal.ofReal_le_ofReal (Int.floor_le _)) (geodArrival_never_early Ω hℓ x0 x)

end Speed

end FireEgress.Eikonal

#print axioms FireEgress.Eikonal.hasFDerivAt_norm_sub
#print axioms FireEgress.Eikonal.pointArrival_hasFDerivAt
#print axioms FireEgress.Eikonal.pointArrival_eikonal
#print axioms FireEgress.Eikonal.pointArrival_gradient
#print axioms FireEgress.Eikonal.plane_eikonal
#print axioms FireEgress.Eikonal.pointArrival_not_differentiableAt_source
#print axioms FireEgress.Eikonal.pointArrival_lipschitz
#print axioms FireEgress.Eikonal.pointArrival_lipschitzWith
#print axioms FireEgress.Eikonal.norm_le_plLength
#print axioms FireEgress.Eikonal.pointArrival_isLeast_pl
#print axioms FireEgress.Eikonal.pointArrival_le_curveTime
#print axioms FireEgress.Eikonal.curveTime_segment
#print axioms FireEgress.Eikonal.pointArrival_isLeast_curve
#print axioms FireEgress.Eikonal.multiArrival_le
#print axioms FireEgress.Eikonal.multiArrival_lipschitz
#print axioms FireEgress.Eikonal.multiArrival_lipschitzWith
#print axioms FireEgress.Eikonal.hasFDerivAt_eq_of_touch
#print axioms FireEgress.Eikonal.multiArrival_eikonal
#print axioms FireEgress.Eikonal.multiArrival_hasFDerivAt_of_nearest
#print axioms FireEgress.Eikonal.multiArrival_not_differentiableAt_tie
#print axioms FireEgress.Eikonal.multiArrival_isLeast_pl
#print axioms FireEgress.Eikonal.norm_latPos_sub
#print axioms FireEgress.Eikonal.latArrival_isLeast
#print axioms FireEgress.Eikonal.latArrival_never_early
#print axioms FireEgress.Eikonal.latArrival_le_K
#print axioms FireEgress.Eikonal.latArrival_lt
#print axioms FireEgress.Eikonal.oct_12_5
#print axioms FireEgress.Eikonal.latPos_refine
#print axioms FireEgress.Eikonal.latArrival_no_refinement
#print axioms FireEgress.Eikonal.axisStep_vals
#print axioms FireEgress.Eikonal.l1Z_le_pathLen
#print axioms FireEgress.Eikonal.l1Z_attained
#print axioms FireEgress.Eikonal.rect_isShortest
#print axioms FireEgress.Eikonal.euclid_le_l1
#print axioms FireEgress.Eikonal.l1_le_sqrt2_mul_euclid
#print axioms FireEgress.Eikonal.l1_sqrt2_attained
#print axioms FireEgress.Eikonal.rectArrival_isLeast
#print axioms FireEgress.Eikonal.rectArrival_bounds
#print axioms FireEgress.Eikonal.rectArrival_diagonal
#print axioms FireEgress.Eikonal.PLIn.mono
#print axioms FireEgress.Eikonal.ofReal_norm_le_geodDist
#print axioms FireEgress.Eikonal.geodDist_le_plLength
#print axioms FireEgress.Eikonal.geodDist_anti
#print axioms FireEgress.Eikonal.geodDist_of_segment
#print axioms FireEgress.Eikonal.geodDist_univ
#print axioms FireEgress.Eikonal.geodArrival_never_early
#print axioms FireEgress.Eikonal.geodArrival_walls_delay
#print axioms FireEgress.Eikonal.geodArrival_open_floor
#print axioms FireEgress.Eikonal.plEnd_append
#print axioms FireEgress.Eikonal.plLength_append
#print axioms FireEgress.Eikonal.PLIn_append
#print axioms FireEgress.Eikonal.walkCost_polyline
#print axioms FireEgress.Eikonal.graph_front_never_early
#print axioms FireEgress.Eikonal.dijkstra_front_never_early
#print axioms FireEgress.Eikonal.walkCost_nonneg
#print axioms FireEgress.Eikonal.dijkstra_bellman
#print axioms FireEgress.Eikonal.dijkstra_bellman_min
#print axioms FireEgress.Eikonal.dijkstra_discrete_eikonal
#print axioms FireEgress.Eikonal.bellman_unique
#print axioms FireEgress.Eikonal.stepLen_pos
#print axioms FireEgress.Eikonal.stepLen_eq_sqrt
#print axioms FireEgress.Eikonal.latLinks_eq_speedLinks
#print axioms FireEgress.Eikonal.latLinks_free
#print axioms FireEgress.Eikonal.lattice_walls_never_early
#print axioms FireEgress.Eikonal.isWalk_snoc
#print axioms FireEgress.Eikonal.latLinks_walk
#print axioms FireEgress.Eikonal.walkCost_cons
#print axioms FireEgress.Eikonal.isWalk_walkCost
#print axioms FireEgress.Eikonal.lattice_walls_delay
#print axioms FireEgress.Eikonal.isWalk_univ
#print axioms FireEgress.Eikonal.open_lattice_dijkstra_exact
#print axioms FireEgress.Eikonal.lattice_discrete_eikonal
#print axioms FireEgress.Eikonal.walkCost_slower
#print axioms FireEgress.Eikonal.arrival_speed_mono
#print axioms FireEgress.Eikonal.travelTime_mono
#print axioms FireEgress.Eikonal.minTravelTime_speed_mono
#print axioms FireEgress.Eikonal.minTravelTime_ignition_mono
#print axioms FireEgress.Eikonal.travelTime_segment
#print axioms FireEgress.Eikonal.radial_band
#print axioms FireEgress.Eikonal.radial_never_late
