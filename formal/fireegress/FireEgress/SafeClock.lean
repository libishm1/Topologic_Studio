import Mathlib
import FireEgress.SafeEscape

/-!
# The clock lemma: safe on the step clock is safe in seconds (`rlForecast.js`)

Studio's RL card trains both learners, the shielded learner (`rlSafe.js`, proved in
`SafeEscape.lean`) and the shipped Q-learning (`rl.js`), against a fire forecast in seconds: the
closing time `c v` of every node, from a Flamescapes run (`forecast.js forecastFromFrames`).
The learners count moves, one link a step, so `rlForecast.js` puts `c` on their clock:

* **The step.** Every move takes at most `τ` seconds. The code takes `τ = (L / speed) (1 + 10⁻⁹)`
  with `L` the longest link walked in one move; on the "split" clock a link longer than `L` is
  walked as `⌈len / L⌉` moves through virtual nodes placed along it (`split_link_time`).
* **The conversion** (`forecastIgnition`). An evacuee leaving at `d` seconds is handed
  `ig v = (c v - d) / τ - 1` (`ig`), with `c` already multiplied by the forecast margin. `rl.js`
  allows a move at step `t` into `v` when `t < ig v`; `rlSafe.js` when `t + 1 ≤ opn v` with
  `opn v = min H (max 0 ⌈ig v⌉)` (`openUntil`; `opnOf` here). Both say the same:
  `d + (t + 1) τ < c v` (`move_safe_iff`, `opn_iff`).
* **The clock lemma** (`clock_safe`, `clock_safe_steps`). If each move takes at most `τ` seconds and
  a route is a safe escape on the step clock whose deadlines are made from `c`, `d` and `τ` as
  above, every node of the route is reached, in seconds, strictly before `c` closes it. `Walk` is
  the route of an escape (`walk_esc`, `esc_walk` against `SafeEscape.Esc`).
* **The forecast.** The guarantee is relative to `c`: it holds for any fire that closes no node
  sooner than `c` (`clock_safe_of_le`), in particular, with a margin `s ∈ (0, 1]` and
  `c = s c₀`, for the forecast `c₀` itself (`margin_le`).

## What is not covered

* Floating point: here `c`, `d` and `τ` are reals. The code's `10⁻⁹` slack on `τ` is there so that
  rounding in the conversion and in summing link times cannot turn a safe move into a late one;
  `rlForecast.test.mjs` checks the conversion and the seconds on random graphs, and every
  returned route is walked again in seconds (`forecast.js walkRoute`), a late arrival making
  the result uncertified.
* The start: the code refuses to train when `c start ≤ d`; the lemma is about the nodes entered.
* Anything about the fire beyond the forecast (`SafeEscape`'s "What is not covered").
-/

namespace FireEgress.SafeClock

open FireEgress.SafeEscape

variable {V : Type*}

/-- The ignition step `rlForecast.js forecastIgnition` hands both learners. -/
noncomputable def ig (c : V → ℝ) (d τ : ℝ) (v : V) : ℝ := (c v - d) / τ - 1

/-- `rlSafe.js openUntil` applied to `ig`: the last step at which `v` may be entered, `min H (max 0 ⌈ig v⌉)`. -/
noncomputable def opnOf (c : V → ℝ) (d τ : ℝ) (H : ℕ) (v : V) : ℕ := min H (⌈ig c d τ v⌉).toNat

/-- `rl.js`'s test `t < ig v` is arrival before closing: `d + (t + 1) τ < c v`. -/
theorem move_safe_iff (c : V → ℝ) {d τ : ℝ} (hτ : 0 < τ) (t : ℕ) (v : V) :
    (t : ℝ) < ig c d τ v ↔ d + ((t : ℝ) + 1) * τ < c v := by
  unfold ig
  constructor
  · intro h
    have h1 : (t : ℝ) + 1 < (c v - d) / τ := by linarith
    have h2 := (lt_div_iff₀ hτ).mp h1
    linarith
  · intro h
    have h1 : ((t : ℝ) + 1) * τ < c v - d := by linarith
    have h2 := (lt_div_iff₀ hτ).mpr h1
    linarith

/-- `rlSafe.js`'s test `t + 1 ≤ opn v`, within the step limit, is the same. -/
theorem opn_iff (c : V → ℝ) {d τ : ℝ} (hτ : 0 < τ) {H t : ℕ} (hH : t + 1 ≤ H) (v : V) :
    t + 1 ≤ opnOf c d τ H v ↔ d + ((t : ℝ) + 1) * τ < c v := by
  rw [← move_safe_iff c hτ t v]
  unfold opnOf
  constructor
  · intro h
    have h1 : t + 1 ≤ (⌈ig c d τ v⌉).toNat := le_trans h (min_le_right _ _)
    have h2 : t < (⌈ig c d τ v⌉).toNat := by omega
    have h3 : (t : ℤ) < ⌈ig c d τ v⌉ := Int.lt_toNat.mp h2
    exact_mod_cast Int.lt_ceil.mp h3
  · intro h
    have h3 : (t : ℤ) < ⌈ig c d τ v⌉ := Int.lt_ceil.mpr (by exact_mod_cast h)
    have h2 : t < (⌈ig c d τ v⌉).toNat := Int.lt_toNat.mpr h3
    exact le_min hH (by omega)

/-- The route of a safe escape on the step clock (`SafeEscape.Esc` with its moves kept): from `v`
at step `t`, `l` lists the nodes entered, each along an open link, arriving at
`t + 1 ≤ opn w` and `t + 1 ≤ H`; the walk ends at an exit. -/
inductive Walk (G : Net V) : ℕ → V → List V → Prop
  | here {t : ℕ} {v : V} : G.ex v = true → Walk G t v []
  | step {t : ℕ} {v w : V} {l : List V} : G.ex v = false → w ∈ G.nb v → t + 1 ≤ G.opn w →
      t + 1 ≤ G.H → Walk G (t + 1) w l → Walk G t v (w :: l)

/-- A route is an escape of its length. -/
theorem walk_esc {G : Net V} {t : ℕ} {v : V} {l : List V} (h : Walk G t v l) : Esc G t v l.length := by
  induction h with
  | here hx => exact Esc.here hx
  | step hv hw ho hH _ ih => exact Esc.step hv hw ho hH ih

/-- Every escape has a route. -/
theorem esc_walk {G : Net V} {t : ℕ} {v : V} {k : ℕ} (h : Esc G t v k) :
    ∃ l : List V, Walk G t v l ∧ l.length = k := by
  induction h with
  | here hx => exact ⟨[], Walk.here hx, rfl⟩
  | step hv hw ho hH _ ih =>
    obtain ⟨l, hl, hk⟩ := ih
    exact ⟨_ :: l, Walk.step hv hw ho hH hl, by simp [hk]⟩

/-- Along a route, the `i`-th node entered arrives at step `t + i + 1`, safely on the clock. -/
theorem walk_opn {G : Net V} {t : ℕ} {v : V} {l : List V} (h : Walk G t v l) :
    ∀ i (hi : i < l.length), t + i + 1 ≤ G.opn (l[i]'hi) ∧ t + i + 1 ≤ G.H := by
  induction h with
  | here _ => intro i hi; simp at hi
  | @step t v w l _ _ ho hH _ ih =>
    intro i hi
    cases i with
    | zero => simpa using ⟨ho, hH⟩
    | succ j =>
      have hj : j < l.length := by simpa using hi
      obtain ⟨h1, h2⟩ := ih j hj
      simp only [List.getElem_cons_succ]
      exact ⟨by omega, by omega⟩

/-- Moves of at most `τ` seconds: after `i` moves the clock has advanced at most `i τ`. -/
theorem arrival_le (a : ℕ → ℝ) {τ : ℝ} {n : ℕ} (hmove : ∀ i, i < n → a (i + 1) ≤ a i + τ) :
    ∀ i, i ≤ n → a i ≤ a 0 + i * τ := by
  intro i
  induction i with
  | zero => intro _; simp
  | succ j ih =>
    intro hj
    have h1 := ih (by omega)
    have h2 := hmove j (by omega)
    have h3 : a 0 + ((j + 1 : ℕ) : ℝ) * τ = (a 0 + (j : ℝ) * τ) + τ := by push_cast; ring
    rw [h3]
    linarith

/-- **The clock lemma, `rl.js`'s form.** The evacuee stands at the start at time `a 0 ≤ d + t τ`,
step `t`, and enters the `i`-th node of `l` at time `a (i + 1) ≤ a i + τ`. If every move passes
the learners' test, `t + i < ig (l i)`, every node is reached before it closes. -/
theorem clock_safe_steps (c : V → ℝ) {d τ : ℝ} (hτ : 0 < τ) {t : ℕ} (l : List V)
    (hsafe : ∀ i (hi : i < l.length), ((t + i : ℕ) : ℝ) < ig c d τ (l[i]'hi))
    (a : ℕ → ℝ) (h0 : a 0 ≤ d + t * τ) (hmove : ∀ i, i < l.length → a (i + 1) ≤ a i + τ) :
    ∀ i (hi : i < l.length), a (i + 1) < c (l[i]'hi) := by
  intro i hi
  have h1 := arrival_le a hmove (i + 1) (by omega)
  have h2 := (move_safe_iff c hτ (t + i) (l[i]'hi)).mp (hsafe i hi)
  have h3 : d + (t : ℝ) * τ + ((i + 1 : ℕ) : ℝ) * τ = d + (((t + i : ℕ) : ℝ) + 1) * τ := by push_cast; ring
  linarith

/-- **The clock lemma.** If each move takes at most `τ` seconds and a route is a safe escape on the
step clock whose deadlines are made from `c`, `d` and `τ` as the code makes them
(`G.opn = opnOf c d τ G.H`), then in seconds every node on the route is reached before `c`. -/
theorem clock_safe (c : V → ℝ) {d τ : ℝ} (hτ : 0 < τ) (G : Net V)
    (hG : ∀ v, G.opn v = opnOf c d τ G.H v) {t : ℕ} {v : V} {l : List V} (hw : Walk G t v l)
    (a : ℕ → ℝ) (h0 : a 0 ≤ d + t * τ) (hmove : ∀ i, i < l.length → a (i + 1) ≤ a i + τ) :
    ∀ i (hi : i < l.length), a (i + 1) < c (l[i]'hi) := by
  refine clock_safe_steps c hτ l (fun i hi => ?_) a h0 hmove
  obtain ⟨ho, hH⟩ := walk_opn hw i hi
  rw [hG] at ho
  have h := (opn_iff c hτ (t := t + i) (by omega) (l[i]'hi)).mp (by omega)
  exact (move_safe_iff c hτ (t + i) (l[i]'hi)).mpr (by exact_mod_cast h)

/-- The same from an escape: every safe escape on the step clock has a route reached in time. -/
theorem clock_safe_esc (c : V → ℝ) {d τ : ℝ} (hτ : 0 < τ) (G : Net V)
    (hG : ∀ v, G.opn v = opnOf c d τ G.H v) {t : ℕ} {v : V} {k : ℕ} (he : Esc G t v k) :
    ∃ l : List V, Walk G t v l ∧ l.length = k ∧
      ∀ (a : ℕ → ℝ), a 0 ≤ d + t * τ → (∀ i, i < l.length → a (i + 1) ≤ a i + τ) →
        ∀ i (hi : i < l.length), a (i + 1) < c (l[i]'hi) := by
  obtain ⟨l, hl, hk⟩ := esc_walk he
  exact ⟨l, hl, hk, fun a h0 hmove => clock_safe c hτ G hG hl a h0 hmove⟩

/-- Safe against any fire that closes no node sooner than `c`: a route reached before `c`
is reached before `c'` whenever `c ≤ c'`. -/
theorem clock_safe_of_le {c c' : V → ℝ} (hle : ∀ v, c v ≤ c' v) {l : List V} {a : ℕ → ℝ}
    (h : ∀ i (hi : i < l.length), a (i + 1) < c (l[i]'hi)) :
    ∀ i (hi : i < l.length), a (i + 1) < c' (l[i]'hi) :=
  fun i hi => lt_of_lt_of_le (h i hi) (hle _)

/-- A margin `s ∈ (0, 1]` only brings closing times forward: `s c₀ ≤ c₀` where `c₀ ≥ 0`. -/
theorem margin_le {s x : ℝ} (hs : s ≤ 1) (hx : 0 ≤ x) : s * x ≤ x := by nlinarith

/-- The split clock: a link of length `len`, walked at `speed`, takes no longer in seconds than
the `⌈len / L⌉` moves of `L / speed` seconds it is split into. -/
theorem split_link_time {len L speed : ℝ} (hL : 0 < L) (hs : 0 < speed) :
    len / speed ≤ (⌈len / L⌉₊ : ℝ) * (L / speed) := by
  have h1 : len / L ≤ (⌈len / L⌉₊ : ℝ) := Nat.le_ceil _
  have h2 : len ≤ (⌈len / L⌉₊ : ℝ) * L := by
    have := (div_le_iff₀ hL).mp h1
    linarith
  rw [div_le_iff₀ hs]
  calc len ≤ (⌈len / L⌉₊ : ℝ) * L := h2
    _ = (⌈len / L⌉₊ : ℝ) * (L / speed) * speed := by
      rw [mul_assoc, div_mul_cancel₀ L hs.ne']

end FireEgress.SafeClock
