import Mathlib

/-!
# Physical dimensions, decidable

A dimension is a vector of exponents of mass, length, time and temperature.
Fire correlations raise quantities to fractional powers (1/2, 1/3, 2/5, 5/3,
5/6, 1/5, and 3.4 = 17/5 in the FED heat dose), so exponents are stored scaled
by 60 = lcm(2, 3, 4, 5, 6): every power in those equations then stays an
integer, and `decide` settles whether two sides of an equation agree.

`rpow a n d` is `a ^ (n / d)`. It is exact only when `exactPow a n d` holds
(every scaled exponent times `n` divisible by `d`); a homogeneity theorem
about a fractional power should state `exactPow` as well, so integer division
cannot hide a mismatch.
-/

namespace FireEgress

structure Dim where
  M : Int
  L : Int
  T : Int
  K : Int
  deriving DecidableEq, Repr

namespace Dim

/-- The scale of every exponent: 60 means a first power. -/
def unit : Int := 60

def one : Dim := ⟨0, 0, 0, 0⟩
def mass : Dim := ⟨60, 0, 0, 0⟩
def length : Dim := ⟨0, 60, 0, 0⟩
def time : Dim := ⟨0, 0, 60, 0⟩
def temp : Dim := ⟨0, 0, 0, 60⟩

instance : Mul Dim := ⟨fun a b => ⟨a.M + b.M, a.L + b.L, a.T + b.T, a.K + b.K⟩⟩
instance : Inv Dim := ⟨fun a => ⟨-a.M, -a.L, -a.T, -a.K⟩⟩
instance : Div Dim := ⟨fun a b => a * b⁻¹⟩

/-- `a ^ (n / d)`, exact when `exactPow a n d`. -/
def rpow (a : Dim) (n d : Int) : Dim := ⟨a.M * n / d, a.L * n / d, a.T * n / d, a.K * n / d⟩

def exactPow (a : Dim) (n d : Int) : Prop :=
  d ∣ a.M * n ∧ d ∣ a.L * n ∧ d ∣ a.T * n ∧ d ∣ a.K * n

instance (a : Dim) (n d : Int) : Decidable (exactPow a n d) := by
  unfold exactPow; infer_instance

-- Derived dimensions (SI).
def area : Dim := length * length
def volume : Dim := area * length
def velocity : Dim := length / time
def accel : Dim := velocity / time
def energy : Dim := mass * length * length / (time * time)
def power : Dim := energy / time
def pressure : Dim := mass / (length * time * time)
def density : Dim := mass / volume
def massFlow : Dim := mass / time
def specificHeat : Dim := energy / (mass * temp)
/-- Heat transfer coefficient, W m⁻² K⁻¹. -/
def htc : Dim := power / (area * temp)
/-- Thermal conductivity, W m⁻¹ K⁻¹. -/
def conductivity : Dim := power / (length * temp)

/-- Sanity: a watt is a joule per second, and a pascal times a cubic metre is a joule. -/
example : power = energy / time := rfl
example : pressure * volume = energy := by decide
example : exactPow (rpow power 1 3) 3 1 := by decide

end Dim
end FireEgress
