import Mathlib

/-!
# Federation: a second model in the first one's frame

Specifications for `src/core/federation.js` (a second IFC model, an MEP or structural one,
drawn in the architecture's frame; `wiki/research/sprinkler-ifc-mep-2026-11-07.md` section 6).
The viewer draws each model through its own coordination (`C`, That Open's coordination matrix
on web-ifc's frame); the second model is moved by `C_main ∘ R ∘ C_linked⁻¹`, with `R` the
alignment of the two IFC frames (the identity, or the two map conversions). Real arithmetic.

## Which theorem backs which part of the code

* `linkMatrix` (the composition, for any invertible maps): `link_draws` (a point the second model
  draws at `C_linked q` goes where the first draws `q`), `linkAligned_draws` (with an IFC-frame
  alignment `R`, to where the first draws `R q`), `turn_conj` (the alignment conjugated by the
  turn to y up acts on turned points as `R` on IFC points), `link_self` (one frame: the
  identity), `link_translate` (translations: `x ↦ x + (c_main - c_linked)`, That Open's
  `autoCoordinate`, base minus own).
* `mapMatrix` (IfcMapConversion in the plane, `E = E₀ + s (a x - b y)`,
  `N = N₀ + s (b x + a y)`, `a² + b² = 1`): `mapConv_dist` (distances scale by `s`: a
  conversion with `s = 1` moves a model rigidly), `mapInv_conv` and `mapConv_inv` (the inverse
  the alignment uses is exact for `s ≠ 0`), `relative_agrees` (the relative alignment puts a
  second model's point where the first model's conversion maps the same map coordinates).
* `planOverlap` (the share of the smaller plan the two have in common): `overlap_comm`,
  `overlap_nonneg`, `overlap_le_left`, `overlap_contained`, `share_mem` (in `[0, 1]`),
  `share_contained` (a plan inside the other gives `1`: a site-wide service model holding the
  building overlaps it fully).
-/

namespace Federation

section Link

variable {α β : Type*}

/-- The viewer's link: undo the second model's coordination, then apply the first's. -/
def link (Cm Cl : α ≃ β) : β ≃ β := Cl.symm.trans Cm

/-- A point the second model draws at `Cl q` is drawn where the first model draws `q`. -/
theorem link_draws (Cm Cl : α ≃ β) (q : α) : link Cm Cl (Cl q) = Cm q := by
  simp [link]

/-- With the IFC frames aligned by `R` (map conversions), the link with `R` in it. -/
def linkAligned (Cm Cl : α ≃ β) (R : α ≃ α) : β ≃ β := Cl.symm.trans (R.trans Cm)

/-- The point the second model calls `q` lands where the first draws `R q`. -/
theorem linkAligned_draws (Cm Cl : α ≃ β) (R : α ≃ α) (q : α) : linkAligned Cm Cl R (Cl q) = Cm (R q) := by
  simp [linkAligned]

/-- With `R` the identity, the aligned link is the plain one. -/
theorem linkAligned_refl (Cm Cl : α ≃ β) : linkAligned Cm Cl (Equiv.refl α) = link Cm Cl := by
  ext x
  simp [linkAligned, link]

/-- One frame: the link is the identity. -/
theorem link_self (C : α ≃ β) : link C C = Equiv.refl β := by
  ext x
  simp [link]

/-- The alignment `R` of IFC frames, conjugated by the turn `T` to the viewer's y up, acts on
turned points as `R` does on IFC points (`linkMatrix`'s `TURN · R · UNTURN`). -/
theorem turn_conj {γ : Type*} (T : α ≃ γ) (R : α ≃ α) (q : α) : (T.symm.trans (R.trans T)) (T q) = T (R q) := by
  simp

end Link

/-- Translations of `ℝ³` (or any additive group) by `c`. -/
def translate {V : Type*} [AddCommGroup V] (c : V) : V ≃ V where
  toFun x := x + c
  invFun x := x - c
  left_inv x := by simp
  right_inv x := by simp

/-- For translations the link adds `c_main - c_linked`: That Open's `autoCoordinate`, which moves a
later model by the first model's base minus its own. -/
theorem link_translate {V : Type*} [AddCommGroup V] (cm cl x : V) :
    link (translate cm) (translate cl) x = x + (cm - cl) := by
  simp only [link, Equiv.trans_apply]
  show (x - cl) + cm = x + (cm - cl)
  abel

section Map

/-- IfcMapConversion in the plane: `E = E₀ + s (a x - b y)`, `N = N₀ + s (b x + a y)`. -/
def mapConv (E0 N0 s a b : ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  (E0 + s * (a * p.1 - b * p.2), N0 + s * (b * p.1 + a * p.2))

/-- Its inverse for a unit axis `(a, b)` and `s ≠ 0`. -/
noncomputable def mapInv (E0 N0 s a b : ℝ) (P : ℝ × ℝ) : ℝ × ℝ :=
  ((a * (P.1 - E0) + b * (P.2 - N0)) / s, (-b * (P.1 - E0) + a * (P.2 - N0)) / s)

/-- Distances scale by `s` (squared form): a conversion with `s = 1` moves a model rigidly. -/
theorem mapConv_dist (E0 N0 s a b : ℝ) (h : a ^ 2 + b ^ 2 = 1) (p q : ℝ × ℝ) :
    ((mapConv E0 N0 s a b p).1 - (mapConv E0 N0 s a b q).1) ^ 2 +
        ((mapConv E0 N0 s a b p).2 - (mapConv E0 N0 s a b q).2) ^ 2 =
      s ^ 2 * ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2) := by
  simp only [mapConv]
  have e : (E0 + s * (a * p.1 - b * p.2) - (E0 + s * (a * q.1 - b * q.2))) ^ 2 +
      (N0 + s * (b * p.1 + a * p.2) - (N0 + s * (b * q.1 + a * q.2))) ^ 2 =
      s ^ 2 * (a ^ 2 + b ^ 2) * ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2) := by ring
  rw [e, h, mul_one]

/-- The inverse undoes the conversion. -/
theorem mapInv_conv (E0 N0 s a b : ℝ) (hs : s ≠ 0) (h : a ^ 2 + b ^ 2 = 1) (p : ℝ × ℝ) :
    mapInv E0 N0 s a b (mapConv E0 N0 s a b p) = p := by
  obtain ⟨x, y⟩ := p
  simp only [mapInv, mapConv, Prod.mk.injEq]
  constructor
  · field_simp
    linear_combination s * x * h
  · field_simp
    linear_combination s * y * h

/-- The conversion undoes the inverse. -/
theorem mapConv_inv (E0 N0 s a b : ℝ) (hs : s ≠ 0) (h : a ^ 2 + b ^ 2 = 1) (P : ℝ × ℝ) :
    mapConv E0 N0 s a b (mapInv E0 N0 s a b P) = P := by
  obtain ⟨X, Y⟩ := P
  simp only [mapInv, mapConv, Prod.mk.injEq]
  constructor
  · field_simp
    linear_combination (X - E0) * h
  · field_simp
    linear_combination (Y - N0) * h

/-- The relative alignment (`alignment`: the first conversion's inverse after the second's) puts
a point of the second model where the first model's conversion gives the same map coordinates. -/
theorem relative_agrees (E0 N0 s a b E0' N0' s' a' b' : ℝ) (hs : s ≠ 0) (h : a ^ 2 + b ^ 2 = 1)
    (q : ℝ × ℝ) :
    mapConv E0 N0 s a b (mapInv E0 N0 s a b (mapConv E0' N0' s' a' b' q)) = mapConv E0' N0' s' a' b' q :=
  mapConv_inv E0 N0 s a b hs h _

end Map

section Overlap

/-- The length two intervals `[a₀, a₁]` and `[b₀, b₁]` share (`planOverlap`, per axis). -/
noncomputable def overlap (a0 a1 b0 b1 : ℝ) : ℝ := max 0 (min a1 b1 - max a0 b0)

theorem overlap_comm (a0 a1 b0 b1 : ℝ) : overlap a0 a1 b0 b1 = overlap b0 b1 a0 a1 := by
  simp [overlap, min_comm, max_comm]

theorem overlap_nonneg (a0 a1 b0 b1 : ℝ) : 0 ≤ overlap a0 a1 b0 b1 := le_max_left _ _

/-- At most the first interval's length. -/
theorem overlap_le_left (a0 a1 b0 b1 : ℝ) (h : a0 ≤ a1) : overlap a0 a1 b0 b1 ≤ a1 - a0 := by
  unfold overlap
  apply max_le (by linarith)
  have h1 : min a1 b1 ≤ a1 := min_le_left _ _
  have h2 : a0 ≤ max a0 b0 := le_max_left _ _
  linarith

/-- An interval inside the other shares all of itself. -/
theorem overlap_contained (a0 a1 b0 b1 : ℝ) (h0 : a0 ≤ b0) (h1 : b1 ≤ a1) (hb : b0 ≤ b1) :
    overlap a0 a1 b0 b1 = b1 - b0 := by
  unfold overlap
  rw [min_eq_right h1, max_eq_right h0]
  exact max_eq_right (by linarith)

/-- The share of the smaller plan the two have in common, capped at 1 as the code caps it. -/
noncomputable def share (w d small : ℝ) : ℝ := min 1 (w * d / small)

/-- The share lies in `[0, 1]` for non-negative overlaps and a positive smaller area. -/
theorem share_mem (w d small : ℝ) (hw : 0 ≤ w) (hd : 0 ≤ d) (hs : 0 < small) :
    0 ≤ share w d small ∧ share w d small ≤ 1 := by
  refine ⟨le_min zero_le_one (div_nonneg (mul_nonneg hw hd) hs.le), min_le_left _ _⟩

/-- A plan inside the other (the overlap is the smaller plan itself) gives a share of 1. -/
theorem share_contained (w d : ℝ) (hp : 0 < w * d) : share w d (w * d) = 1 := by
  simp [share, div_self hp.ne']

end Overlap

end Federation
