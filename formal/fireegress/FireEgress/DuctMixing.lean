import Mathlib

/-!
# Running ventilation: the ducts' exchange (flamescapes.js `ducts`)

Each air terminal `t` of a ventilation system draws the share `f` of its cell's
gas, mass `f M t`, into the system's ducts, taken as one well-mixed volume, and
gets the same mass of the ducts' mixture back. A quantity per unit mass `x t` (a
species' mass fraction, or the temperature with the heat counted as `M T`)
becomes `(1 - f) x t + f x̄`, with `x̄` the mean of `x` weighted by the drawn
masses (the `f` cancels from the weights). Two properties:

* `ducts_conserve`: the system's total `∑ M x` is unchanged, so the ducts move
  soot, CO and heat between rooms without making or losing any;
* `ducts_after_mem`: no terminal cell leaves the range of values the system's
  terminals held before, so a mass fraction stays in `[0, 1]` and no cell is
  heated past the hottest terminal.

Not the paper's: Flamescapes and Fire-X model no ducts. Floating point is not
covered; the test in flamescapes.test.mjs checks the conservation to 1e-5.
-/

namespace FireEgress.DuctMixing

variable {ι : Type*}

/-- The ducts' mixture: the mean of `x` over the system's terminals `s`,
weighted by the cells' gas masses `M` (the drawn share `f` cancels). -/
noncomputable def mixture (s : Finset ι) (M x : ι → ℝ) : ℝ :=
  (∑ t ∈ s, M t * x t) / ∑ t ∈ s, M t

/-- A terminal cell's value after one exchange. -/
noncomputable def after (s : Finset ι) (f : ℝ) (M x : ι → ℝ) (t : ι) : ℝ :=
  (1 - f) * x t + f * mixture s M x

/-- The exchange conserves the system's total `∑ M x`: each species' mass, and the heat. -/
theorem ducts_conserve (s : Finset ι) (f : ℝ) (M x : ι → ℝ) (hM : ∑ t ∈ s, M t ≠ 0) :
    ∑ t ∈ s, M t * after s f M x t = ∑ t ∈ s, M t * x t := by
  have key : ∀ t ∈ s, M t * after s f M x t
      = (1 - f) * (M t * x t) + (f * mixture s M x) * M t := by
    intro t _
    unfold after
    ring
  rw [Finset.sum_congr rfl key, Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
  unfold mixture
  field_simp
  ring

/-- The mixture lies in the range of the values drawn (non-negative masses, some gas). -/
theorem mixture_mem (s : Finset ι) (M x : ι → ℝ) (a b : ℝ) (hM : ∀ t ∈ s, 0 ≤ M t)
    (hpos : 0 < ∑ t ∈ s, M t) (hx : ∀ t ∈ s, a ≤ x t ∧ x t ≤ b) :
    a ≤ mixture s M x ∧ mixture s M x ≤ b := by
  unfold mixture
  constructor
  · rw [le_div_iff₀ hpos]
    calc a * ∑ t ∈ s, M t = ∑ t ∈ s, M t * a := by
          rw [Finset.mul_sum]
          exact Finset.sum_congr rfl (fun t _ => by ring)
      _ ≤ ∑ t ∈ s, M t * x t :=
          Finset.sum_le_sum (fun t ht => mul_le_mul_of_nonneg_left (hx t ht).1 (hM t ht))
  · rw [div_le_iff₀ hpos]
    calc ∑ t ∈ s, M t * x t ≤ ∑ t ∈ s, M t * b :=
          Finset.sum_le_sum (fun t ht => mul_le_mul_of_nonneg_left (hx t ht).2 (hM t ht))
      _ = b * ∑ t ∈ s, M t := by
          rw [Finset.mul_sum]
          exact Finset.sum_congr rfl (fun t _ => by ring)

/-- After the exchange (`0 ≤ f ≤ 1`), every terminal cell keeps within the range
the system's terminals held before. -/
theorem ducts_after_mem (s : Finset ι) (f : ℝ) (M x : ι → ℝ) (a b : ℝ) (t : ι) (ht : t ∈ s)
    (hf0 : 0 ≤ f) (hf1 : f ≤ 1) (hM : ∀ u ∈ s, 0 ≤ M u) (hpos : 0 < ∑ u ∈ s, M u)
    (hx : ∀ u ∈ s, a ≤ x u ∧ x u ≤ b) :
    a ≤ after s f M x t ∧ after s f M x t ≤ b := by
  obtain ⟨h1, h2⟩ := mixture_mem s M x a b hM hpos hx
  obtain ⟨h3, h4⟩ := hx t ht
  have g : 0 ≤ 1 - f := by linarith
  unfold after
  constructor
  · nlinarith [mul_le_mul_of_nonneg_left h3 g, mul_le_mul_of_nonneg_left h1 hf0]
  · nlinarith [mul_le_mul_of_nonneg_left h4 g, mul_le_mul_of_nonneg_left h2 hf0]

end FireEgress.DuctMixing
