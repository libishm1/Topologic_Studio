import Mathlib
import FireEgress.Forecast
import FireEgress.Deadline

/-!
# Routing-ready surrogates with error bounds (LG7)

Studio's surrogates (`topologicpy-web-frontend/src/core/surrogate.js`, gap LG7 of
`wiki/research/gap-analysis-2026-10-27/README.md`) replace a forecast's closing times `c`
and FED rates `r` by cheaper ones. The header of `surrogate.js` claims that the routers are
monotone in both, that the hard deadline is 1-Lipschitz in the closing times, that a
two-sided surrogate becomes conservative by closing every point `e` earlier, and that the
hop (flood) forecast is a conservative surrogate of heat diffusion. This file proves those
claims, reusing `Deadline` (safe walks, the deadline `L`, dose deadlines, `hotDose`),
`Forecast` (`runMax`, `closeStep`, `doseRate`) and `GraphNumerics` (the layered BFS `bfs`).

## What each result covers

* `withClosingMargin` (`surrogate.js`): `withMargin m c = max 0 (c - m)` on `EReal`
  (`Infinity - m = Infinity`, so the code's separate `Infinity` branch for `untenable` is the
  same formula). A larger margin never closes a node later (`withMargin_antitone`); the
  result is never later than the input (`withMargin_le_self`, for `c ≥ 0`, as every closing
  time of a forecast is); margins add, as the returned `margin` field records
  (`withMargin_withMargin`). For a surrogate within `e` of the true closing times,
  `withMargin e` is conservative (`withMargin_conservative`), its routes are safe under the
  true forecast (`withMargin_safe`), and its deadline lies in `[L - 2e, L]`
  (`withMargin_deadline`, the "at most 2e" of the header).
* The routers' monotonicity ("Surrogate.lean" in the header of `surrogate.js`): earlier
  closing times only remove safe walks (`safeWalk_mono`, `safeSet_mono`) and only bring
  deadlines forward (`deadline_mono`); with higher link doses as well, the good set of the
  dose limit shrinks, the dose deadline comes forward and the least dose rises
  (`goodSet_mono`, `doseDeadline_mono`, `leastDose_le_surrogate`); a higher FED rate gives a
  higher link dose (`hotDose_le_of_rate_le`). With the doses unchanged and non-decreasing in
  the departure time, the dose-limited deadline is 1-Lipschitz in the closing times too
  (`doseDeadline_le_add`, `doseDeadline_lipschitz`).
* The escape-deadline function `L` (`Deadline.deadline`, `escapeDeadlines`) is 1-Lipschitz
  in the closing times in the sup norm: `c' ≤ c + e` everywhere gives `L' ≤ L + e`
  (`deadline_le_add`, via `safeWalk_shift`), so closing times within `e` of each other give
  deadlines within `e` (`deadline_lipschitz`; with real closing times and finite deadlines,
  `|L' - L| ≤ e`, `deadline_lipschitz_real`); and a surrogate within `e` has `L = ⊥`
  (no route at all) exactly where the true forecast has (`deadline_eq_bot_of_close`).
* `compareForecasts`: when it reports no late node (no gap below `-1e-9`) and a largest
  earlier closing `M` (`maxEarlier`), the surrogate's deadlines are at most `1e-9` later and
  at most `M` earlier than the full forecast's (`compare_deadline`).
* `hopSurrogate`: the diffusion step of `temperatureSteps` (`fire.js`, `diffT`) and the
  flood forecast (`floodTimeline` as `GraphNumerics.bfs` with no burn check, `floodHeat`) on
  the same open-link lists, seed, ambient and fire values. Hypotheses, all from the code:
  the open-neighbour lists are symmetric (`csr` lists both ends of every edge and `blocked`
  is per edge), and `ambient ≤ fire` (20 and 120). Then a diffusion node is above ambient
  only once the flood has reached it (`diffT_hot_reach`), so at every step the flood is at
  least as hot as diffusion (`diffT_le_floodHeat`); hence for **every** threshold (the
  header's "above ambient" is not needed) the hop front closes every node no later
  (`hop_closeStep_le`, `hop_close_le`), its deadlines are no later (`hop_deadline_le`), its
  routes are safe under diffusion (`hop_safe`), and its stored FED rates are never lower
  (`hop_rate_ge`). The flood never cools (`floodHeat_monotone`), nor does diffusion for a
  non-negative `heatTransfer` (`diffT_monotone`).

## What is not covered

* Arithmetic is exact (`ℝ`, `EReal`). The code's `Float64Array` subtraction and its
  tolerances (`1e-9` in `compareForecasts`, `1e-6` on rates) are modelled as stated
  inequalities; `withClosingMargin(f, Infinity)` (allowed by the code's check) gives `NaN`
  for a closing time of `Infinity`, which the model, with a real margin, does not cover.
* `compareForecasts`' rate test reads the surrogate's frame at the full forecast's frame
  times; that both forecasts on the same graph and step count share their frames is read
  from `buildForecast`, not proved. `hop_rate_ge` compares the stored per-step heats.
* The diffusion is `temperatureSteps` with the code's defaults for the clip (`ambient`,
  `fire`), as `buildForecast` calls it; `heatTransfer` is any real for the comparison. The
  flood timeline is the layered BFS with `seen` flags; that `floodTimeline`'s arrays
  implement it is read from the code, as for `GraphNumerics.L7_bfs_layer_eq_earliest`.
* When a surrogate changes the FED rates as well as the closing times, the dose-limited
  deadline is shown monotone only; no bound in terms of the rate error is proved.
-/

namespace FireEgress.Surrogate

open FireEgress.GraphNumerics FireEgress.Deadline FireEgress.Forecast

/-! ## Extended-real shifts -/

/-- Leaving `e` seconds earlier arrives before `x` exactly when leaving on time arrives
before `x + e`, in `EReal` (`⊥ + e = ⊥`, `⊤ + e = ⊤`). -/
theorem coe_sub_lt_iff (t e : ℝ) (x : EReal) :
    ((t - e : ℝ) : EReal) < x ↔ (t : EReal) < x + (e : EReal) := by
  induction x using EReal.rec with
  | bot => simp
  | coe x =>
    rw [← EReal.coe_add, EReal.coe_lt_coe_iff, EReal.coe_lt_coe_iff]
    constructor <;> intro h <;> linarith
  | top =>
    rw [EReal.top_add_coe]
    exact ⟨fun _ => EReal.coe_lt_top _, fun _ => EReal.coe_lt_top _⟩

/-- `x - e ≤ y ↔ x ≤ y + e` in `EReal`, for a real `e`. -/
theorem sub_coe_le_iff (x y : EReal) (e : ℝ) : x - (e : EReal) ≤ y ↔ x ≤ y + (e : EReal) := by
  induction x using EReal.rec with
  | bot => simp
  | top =>
    rw [EReal.top_sub_coe]
    induction y using EReal.rec with
    | bot => simp
    | coe y => rw [← EReal.coe_add]; simp
    | top => simp
  | coe x =>
    induction y using EReal.rec with
    | bot =>
      rw [EReal.bot_add, le_bot_iff, le_bot_iff, ← EReal.coe_sub]
      exact ⟨fun h => absurd h (EReal.coe_ne_bot _), fun h => absurd h (EReal.coe_ne_bot _)⟩
    | coe y =>
      rw [← EReal.coe_sub, ← EReal.coe_add, EReal.coe_le_coe_iff, EReal.coe_le_coe_iff]
      constructor <;> intro h <;> linarith
    | top => rw [EReal.top_add_coe]; simp

/-! ## (1) The hard deadline is monotone and 1-Lipschitz in the closing times -/

section Hard

variable {V : Type*} (E : V → V → Prop) (τ : V → V → ℝ) (X : V → Bool)

/-- **Earlier closing only removes safe walks**: a walk that is safe when every node closes
at `c'` is safe when it closes at `c ≥ c'`. For Studio: a route a conservative surrogate
returns is safe under the full forecast. -/
theorem safeWalk_mono {c c' : V → EReal} (h : ∀ v, c' v ≤ c v) :
    ∀ (ws : List V) (v : V) (t : ℝ), SafeWalk E τ c' t v ws → SafeWalk E τ c t v ws
  | [], v, _, hs => lt_of_lt_of_le hs (h v)
  | w :: ws, v, _, ⟨h1, h2, h3⟩ => ⟨lt_of_lt_of_le h1 (h v), h2, safeWalk_mono h ws w _ h3⟩

theorem safeSet_mono {c c' : V → EReal} (h : ∀ v, c' v ≤ c v) (v : V) :
    safeSet E τ c' X v ⊆ safeSet E τ c X v :=
  fun t ⟨ws, hs, hx⟩ => ⟨ws, safeWalk_mono E τ h ws v t hs, hx⟩

/-- **Earlier closing never gives a later deadline.** -/
theorem deadline_mono {c c' : V → EReal} (h : ∀ v, c' v ≤ c v) (v : V) :
    deadline E τ c' X v ≤ deadline E τ c X v :=
  sSup_le_sSup (Set.image_mono (safeSet_mono E τ X h v))

/-- Shifting the clock: if every node closes under `c'` at most `e` seconds after it closes
under `c`, a walk safe from `t` under `c'` is safe from `t - e` under `c` (it reaches every
node `e` seconds earlier). -/
theorem safeWalk_shift {c c' : V → EReal} {e : ℝ} (h : ∀ v, c' v ≤ c v + (e : EReal)) :
    ∀ (ws : List V) (v : V) (t : ℝ), SafeWalk E τ c' t v ws → SafeWalk E τ c (t - e) v ws
  | [], v, t, hs => (coe_sub_lt_iff t e (c v)).2 (lt_of_lt_of_le hs (h v))
  | w :: ws, v, t, ⟨h1, h2, h3⟩ => by
    refine ⟨(coe_sub_lt_iff t e (c v)).2 (lt_of_lt_of_le h1 (h v)), h2, ?_⟩
    have := safeWalk_shift h ws w (t + τ v w) h3
    rwa [show t + τ v w - e = t - e + τ v w by ring] at this

/-- **One-sided Lipschitz bound**: closing times at most `e` later give deadlines at most
`e` later, `c' ≤ c + e ⇒ L' ≤ L + e` (in `EReal`, so it also holds where `L = ⊥` or
`⊤`). -/
theorem deadline_le_add {c c' : V → EReal} {e : ℝ} (h : ∀ v, c' v ≤ c v + (e : EReal))
    (v : V) : deadline E τ c' X v ≤ deadline E τ c X v + (e : EReal) := by
  apply le_of_coe_lt_imp
  intro t ht
  obtain ⟨ws, hs, hx⟩ := (mem_safeSet_iff E τ c' X v t).2 ht
  have hmem : t - e ∈ safeSet E τ c X v := ⟨ws, safeWalk_shift E τ h ws v t hs, hx⟩
  exact (coe_sub_lt_iff t e _).1 ((mem_safeSet_iff E τ c X v (t - e)).1 hmem)

/-- **The escape deadline is 1-Lipschitz in the closing times (sup norm)**: if every closing
time of `c'` is within `e` of `c`'s, every deadline of `c'` is within `e` of `c`'s. So a
surrogate certified within `e` of the true closing times gives deadlines within `e`. -/
theorem deadline_lipschitz {c c' : V → EReal} {e : ℝ} (h1 : ∀ v, c' v ≤ c v + (e : EReal))
    (h2 : ∀ v, c v ≤ c' v + (e : EReal)) (v : V) :
    deadline E τ c' X v ≤ deadline E τ c X v + (e : EReal) ∧
      deadline E τ c X v ≤ deadline E τ c' X v + (e : EReal) :=
  ⟨deadline_le_add E τ X h1 v, deadline_le_add E τ X h2 v⟩

/-- The Lipschitz bound with real closing times (a forecast's `closes` are finite) and
finite deadlines: `|L'(v) - L(v)| ≤ sup_w |c'(w) - c(w)|`. -/
theorem deadline_lipschitz_real {c c' : V → ℝ} {e : ℝ} (h : ∀ v, |c' v - c v| ≤ e) (v : V)
    {a b : ℝ} (ha : deadline E τ (fun w => (c w : EReal)) X v = a)
    (hb : deadline E τ (fun w => (c' w : EReal)) X v = b) : |b - a| ≤ e := by
  have h1 : ∀ w, ((c' w : ℝ) : EReal) ≤ (c w : EReal) + (e : EReal) := fun w => by
    rw [← EReal.coe_add, EReal.coe_le_coe_iff]; linarith [(abs_le.1 (h w)).2]
  have h2 : ∀ w, ((c w : ℝ) : EReal) ≤ (c' w : EReal) + (e : EReal) := fun w => by
    rw [← EReal.coe_add, EReal.coe_le_coe_iff]; linarith [(abs_le.1 (h w)).1]
  obtain ⟨k1, k2⟩ := deadline_lipschitz E τ X (c := fun w => (c w : EReal))
    (c' := fun w => (c' w : EReal)) h1 h2 v
  rw [ha, hb, ← EReal.coe_add, EReal.coe_le_coe_iff] at k1
  rw [ha, hb, ← EReal.coe_add, EReal.coe_le_coe_iff] at k2
  rw [abs_le]
  constructor <;> linarith

/-- A surrogate at most `e` later than the true forecast leaves a node with no route at all
(`L = ⊥`, `-Infinity`) without one. With the converse bound, the two forecasts agree on
which nodes have any route. -/
theorem deadline_eq_bot_of_close {c c' : V → EReal} {e : ℝ}
    (h : ∀ v, c' v ≤ c v + (e : EReal)) (v : V) (hb : deadline E τ c X v = ⊥) :
    deadline E τ c' X v = ⊥ := by
  have := deadline_le_add E τ X h v
  rw [hb, EReal.bot_add] at this
  exact le_bot_iff.1 this

/-! ### `withClosingMargin` -/

/-- `withClosingMargin`: every closing time moved `m` seconds earlier, not below `0`,
`max(0, c - m)`. With `c = ⊤` (`Infinity`) it stays `⊤`, as the code's `untenable`
branch keeps it. -/
noncomputable def withMargin (m : ℝ) (c : EReal) : EReal := max 0 (c - (m : EReal))

theorem withMargin_nonneg (m : ℝ) (c : EReal) : 0 ≤ withMargin m c := le_max_left _ _

theorem sub_le_withMargin (m : ℝ) (c : EReal) : c - (m : EReal) ≤ withMargin m c :=
  le_max_right _ _

/-- A node that never closes keeps never closing (the code's `u === Infinity` branch). -/
theorem withMargin_top (m : ℝ) : withMargin m ⊤ = ⊤ := by
  unfold withMargin
  rw [EReal.top_sub_coe]
  exact max_eq_right le_top

/-- On a finite closing time, `withMargin` is the real `max 0 (c - m)`. -/
theorem withMargin_coe (m c : ℝ) : withMargin m (c : EReal) = ((max 0 (c - m) : ℝ) : EReal) := by
  unfold withMargin
  rw [← EReal.coe_sub, EReal.coe_strictMono.monotone.map_max, EReal.coe_zero]

/-- **The margin never makes a node close later than the input** (for the closing times
of a forecast, which are `≥ 0`). -/
theorem withMargin_le_self {m : ℝ} (hm : 0 ≤ m) {c : EReal} (hc : 0 ≤ c) :
    withMargin m c ≤ c :=
  max_le hc ((sub_coe_le_iff c c m).2 (le_add_of_nonneg_right (EReal.coe_nonneg.2 hm)))

/-- **A larger margin never makes a node close later.** -/
theorem withMargin_antitone (c : EReal) : Antitone fun m : ℝ => withMargin m c :=
  fun _ _ hmm' => max_le_max le_rfl (EReal.sub_le_sub le_rfl (EReal.coe_le_coe_iff.2 hmm'))

/-- Margins add: applying `m` and then `m'` is applying `m + m'`, which is the `margin`
field `withClosingMargin` returns (`(forecast.margin || 0) + margin`). -/
theorem withMargin_withMargin {m m' : ℝ} (hm' : 0 ≤ m') (c : EReal) :
    withMargin m' (withMargin m c) = withMargin (m + m') c := by
  induction c using EReal.rec with
  | bot =>
    have h0 : withMargin m ⊥ = ((0 : ℝ) : EReal) := by
      unfold withMargin; rw [EReal.bot_sub, EReal.coe_zero]; exact max_eq_left bot_le
    rw [h0, withMargin_coe]
    unfold withMargin
    rw [EReal.bot_sub, max_eq_left (bot_le : (⊥ : EReal) ≤ 0), max_eq_left (by linarith),
      EReal.coe_zero]
  | coe c =>
    rw [withMargin_coe, withMargin_coe, withMargin_coe, EReal.coe_eq_coe_iff]
    rcases le_total 0 (c - m) with h | h
    · rw [max_eq_right h]; congr 1; ring
    · rw [max_eq_left h, max_eq_left (by linarith), max_eq_left (by linarith)]
  | top => rw [withMargin_top, withMargin_top, withMargin_top]

/-- **A two-sided surrogate made conservative**: if `c' ≤ c + e` everywhere and the true
closing times are `≥ 0`, closing every point `e` earlier gives closing times `≤ c`. -/
theorem withMargin_conservative {c c' : V → EReal} {e : ℝ} (hc : ∀ v, 0 ≤ c v)
    (h1 : ∀ v, c' v ≤ c v + (e : EReal)) (v : V) : withMargin e (c' v) ≤ c v :=
  max_le (hc v) ((sub_coe_le_iff _ _ e).2 (h1 v))

/-- ... and it closes no point more than `2e` early. -/
theorem le_withMargin_add {c c' : V → EReal} {e : ℝ} (h2 : ∀ v, c v ≤ c' v + (e : EReal))
    (v : V) : c v ≤ withMargin e (c' v) + ((2 * e : ℝ) : EReal) := by
  have hle : c' v ≤ (c' v - (e : EReal)) + (e : EReal) := (sub_coe_le_iff _ _ e).1 le_rfl
  calc c v ≤ c' v + (e : EReal) := h2 v
    _ ≤ (c' v - (e : EReal)) + (e : EReal) + (e : EReal) := add_le_add_left hle _
    _ = (c' v - (e : EReal)) + ((2 * e : ℝ) : EReal) := by
      rw [add_assoc, ← EReal.coe_add, show e + e = 2 * e by ring]
    _ ≤ withMargin e (c' v) + ((2 * e : ℝ) : EReal) :=
      add_le_add_left (sub_le_withMargin e (c' v)) _

/-- Routes found on the margin-shifted surrogate are safe under the true forecast. -/
theorem withMargin_safe {c c' : V → EReal} {e : ℝ} (hc : ∀ v, 0 ≤ c v)
    (h1 : ∀ v, c' v ≤ c v + (e : EReal)) (ws : List V) (v : V) (t : ℝ)
    (hs : SafeWalk E τ (fun w => withMargin e (c' w)) t v ws) : SafeWalk E τ c t v ws :=
  safeWalk_mono E τ (withMargin_conservative hc h1) ws v t hs

/-- **The hard deadline moves by at most `2e`** (the header of `surrogate.js`): for a
surrogate whose closing times are within `e` of the true ones (`c ≥ 0`), the deadline of
`withClosingMargin(surrogate, e)` lies in `[L - 2e, L]`. -/
theorem withMargin_deadline {c c' : V → EReal} {e : ℝ} (hc : ∀ v, 0 ≤ c v)
    (h1 : ∀ v, c' v ≤ c v + (e : EReal)) (h2 : ∀ v, c v ≤ c' v + (e : EReal)) (v : V) :
    deadline E τ (fun w => withMargin e (c' w)) X v ≤ deadline E τ c X v ∧
      deadline E τ c X v ≤
        deadline E τ (fun w => withMargin e (c' w)) X v + ((2 * e : ℝ) : EReal) :=
  ⟨deadline_mono E τ X (withMargin_conservative hc h1) v,
    deadline_le_add E τ X (le_withMargin_add h2) v⟩

/-! ### `compareForecasts` -/

/-- **What `compareForecasts` certifies about deadlines.** With real closing times `c`
(full) and `c'` (surrogate): if no node is late (`c v - c' v < -ε` never holds; the code
uses `ε = 1e-9`) and every node closes at most `M` earlier (`maxEarlier`), the surrogate's
deadlines are at most `ε` later and at most `M` earlier than the full forecast's. -/
theorem compare_deadline {c c' : V → ℝ} {ε M : ℝ} (hlate : ∀ v, ¬ (c v - c' v < -ε))
    (hM : ∀ v, c v - c' v ≤ M) (v : V) :
    deadline E τ (fun w => (c' w : EReal)) X v ≤
        deadline E τ (fun w => (c w : EReal)) X v + (ε : EReal) ∧
      deadline E τ (fun w => (c w : EReal)) X v ≤
        deadline E τ (fun w => (c' w : EReal)) X v + (M : EReal) := by
  constructor
  · apply deadline_le_add E τ X
    intro w
    rw [← EReal.coe_add, EReal.coe_le_coe_iff]
    have := hlate w
    push Not at this
    linarith
  · apply deadline_le_add E τ X
    intro w
    rw [← EReal.coe_add, EReal.coe_le_coe_iff]
    linarith [hM w]

end Hard

/-! ## (2) Doses: a conservative surrogate never lowers a dose or delays a dose deadline -/

section Dose

variable {V : Type*} (E : V → V → Prop) (τ : V → V → ℝ) (X : V → Bool)

/-- Higher link doses give a higher walk dose. -/
theorem walkDose_le {d d' : V → V → ℝ → ℝ} (hd : ∀ v w, E v w → ∀ t, d v w t ≤ d' v w t) :
    ∀ (ws : List V) (v : V) (t : ℝ), IsWalk E v ws →
      walkDose τ d t v ws ≤ walkDose τ d' t v ws
  | [], _, _, _ => le_rfl
  | w :: ws, v, _, hw => add_le_add (hd v w hw.1 _) (walkDose_le hd ws w _ hw.2)

/-- Earlier closing and higher doses shrink the good set of a dose limit. -/
theorem goodSet_mono {c c' : V → EReal} {d d' : V → V → ℝ → ℝ} (hc : ∀ v, c' v ≤ c v)
    (hd : ∀ v w, E v w → ∀ t, d v w t ≤ d' v w t) (θ : EReal) (v : V) :
    goodSet E τ c' X d' θ v ⊆ goodSet E τ c X d θ v := by
  rintro t ⟨ws, hs, hx, hθ⟩
  refine ⟨ws, safeWalk_mono E τ hc ws v t hs, hx, le_trans ?_ hθ⟩
  exact EReal.coe_le_coe_iff.2
    (walkDose_le E τ hd ws v t ((safeWalk_iff_margin E τ c' ws v t).1 hs).1)

/-- **A conservative surrogate never delays the dose-limited deadline.** -/
theorem doseDeadline_mono {c c' : V → EReal} {d d' : V → V → ℝ → ℝ} (hc : ∀ v, c' v ≤ c v)
    (hd : ∀ v w, E v w → ∀ t, d v w t ≤ d' v w t) (θ : EReal) (v : V) :
    doseDeadline E τ c' X d' θ v ≤ doseDeadline E τ c X d θ v :=
  sSup_le_sSup (Set.image_mono (goodSet_mono E τ X hc hd θ v))

/-- **A conservative surrogate never lowers the least dose** (the last point of
`doseFront`'s front). -/
theorem leastDose_le_surrogate {c c' : V → EReal} {d d' : V → V → ℝ → ℝ}
    (hc : ∀ v, c' v ≤ c v) (hd : ∀ v w, E v w → ∀ t, d v w t ≤ d' v w t) (v : V) (t : ℝ) :
    leastDose E τ c X d v t ≤ leastDose E τ c' X d' v t := by
  apply le_sInf
  rintro _ ⟨ws, hs, hx, rfl⟩
  have hw := ((safeWalk_iff_margin E τ c' ws v t).1 hs).1
  have h1 : leastDose E τ c X d v t ≤ ((walkDose τ d t v ws : ℝ) : EReal) :=
    sInf_le ⟨ws, safeWalk_mono E τ hc ws v t hs, hx, rfl⟩
  exact h1.trans (EReal.coe_le_coe_iff.2 (walkDose_le E τ hd ws v t hw))

/-- **The dose-limited deadline is 1-Lipschitz in the closing times as well**, when the link
doses never fall with the departure time (heat that never falls, `Deadline.hotDose_mono`) and
are the same under both forecasts: `c' ≤ c + e` with `e ≥ 0` gives `T*' ≤ T* + e`. The walk
that is good from `t` under `c'` is safe from `t - e` under `c` and accrues no more dose
there. -/
theorem doseDeadline_le_add {c c' : V → EReal} {e : ℝ} {d : V → V → ℝ → ℝ}
    (hd : ∀ v w, E v w → Monotone (d v w)) (he : 0 ≤ e) (h : ∀ v, c' v ≤ c v + (e : EReal))
    (θ : EReal) (v : V) :
    doseDeadline E τ c' X d θ v ≤ doseDeadline E τ c X d θ v + (e : EReal) := by
  apply sSup_le
  rintro _ ⟨t, ⟨ws, hs, hx, hθ⟩, rfl⟩
  have hw := ((safeWalk_iff_margin E τ c' ws v t).1 hs).1
  have hgood : t - e ∈ goodSet E τ c X d θ v :=
    ⟨ws, safeWalk_shift E τ h ws v t hs, hx,
      (EReal.coe_le_coe_iff.2 (walkDose_mono E τ d hd ws v (t - e) t hw (by linarith))).trans hθ⟩
  have h1 := le_doseDeadline E τ c X d hgood
  rw [EReal.coe_sub] at h1
  exact (sub_coe_le_iff _ _ e).1 h1

/-- The two-sided form: closing times within `e` give dose-limited deadlines within `e`. -/
theorem doseDeadline_lipschitz {c c' : V → EReal} {e : ℝ} {d : V → V → ℝ → ℝ}
    (hd : ∀ v w, E v w → Monotone (d v w)) (he : 0 ≤ e) (h1 : ∀ v, c' v ≤ c v + (e : EReal))
    (h2 : ∀ v, c v ≤ c' v + (e : EReal)) (θ : EReal) (v : V) :
    doseDeadline E τ c' X d θ v ≤ doseDeadline E τ c X d θ v + (e : EReal) ∧
      doseDeadline E τ c X d θ v ≤ doseDeadline E τ c' X d θ v + (e : EReal) :=
  ⟨doseDeadline_le_add E τ X hd he h1 θ v, doseDeadline_le_add E τ X hd he h2 θ v⟩

/-- A higher FED rate gives a higher link dose (`forecast.linkDose` as `hotDose`). -/
theorem hotDose_le_of_rate_le {r r' : V → ℝ → ℝ} (hr : ∀ v, Monotone (r v))
    (hr' : ∀ v, Monotone (r' v)) (hle : ∀ v s, r v s ≤ r' v s) (v w : V) {a : ℝ}
    (ha : 0 ≤ a) (t : ℝ) : hotDose r v w a t ≤ hotDose r' v w a t :=
  intervalIntegral.integral_mono_on (by linarith) (hotRate_mono r hr v w).intervalIntegrable
    (hotRate_mono r' hr' v w).intervalIntegrable (fun s _ => max_le_max (hle v s) (hle w s))

end Dose

/-! ## (3) The hop front is a conservative surrogate of heat diffusion -/

section Hop

variable {V : Type*} [DecidableEq V] (nb : V → List V) (deg : V → ℕ) (h amb fire : ℝ)
  (seed : V)

/-- The open-link relation of `csr` with `blocked` links skipped: `u → w` when `w` is listed
among `u`'s open neighbours (`neighbours[k]` for `k` in `u`'s row, `blocked[edgeIds[k]]`
false). `nb u` lists them with multiplicity, as the code's loop visits them. -/
def openLink (u w : V) : Prop := w ∈ nb u

/-- The flood has no burn check: `GraphNumerics.bfs` with every ignition time `⊤`. -/
def noBurn : V → WithTop ℝ := fun _ => ⊤

/-- The mean of node `i`'s open neighbours' values, `sum / count` (`temperatureSteps`). -/
noncomputable def nbMean (T : V → ℝ) (i : V) : ℝ := ((nb i).map T).sum / (nb i).length

/-- One node's update in `temperatureSteps` before the seed reset and the clip: with
`delta = mean - T i`, `T i + delta · min(1, heatTransfer / degree)` when the node has a
link (`degree ≠ 0`), an open one (`count ≠ 0`) and `delta > 0`; otherwise unchanged.
`deg i` is the full degree (blocked links count, as in the code). -/
noncomputable def diffUpd (T : V → ℝ) (i : V) : ℝ :=
  if deg i ≠ 0 ∧ (nb i).length ≠ 0 ∧ T i < nbMean nb T i then
    T i + (nbMean nb T i - T i) * min 1 (h / deg i)
  else T i

/-- One step of `temperatureSteps`: every node updated, the seed reset to `fire`, then every
node clipped to `[ambient, fire]`. -/
noncomputable def diffStep (T : V → ℝ) (i : V) : ℝ :=
  min fire (max amb (if i = seed then fire else diffUpd nb deg h T i))

/-- The diffusion forecast's frame `k`: frame 0 is ambient with the seed at `fire`. -/
noncomputable def diffT : ℕ → V → ℝ
  | 0 => fun i => if i = seed then fire else amb
  | k + 1 => diffStep nb deg h amb fire seed (diffT k)

/-- The discrete maximum principle of the clipped step: every frame lies in
`[ambient, fire]`. -/
theorem diffT_bounds (hle : amb ≤ fire) :
    ∀ k i, amb ≤ diffT nb deg h amb fire seed k i ∧ diffT nb deg h amb fire seed k i ≤ fire
  | 0, i => by
    show amb ≤ (if i = seed then fire else amb) ∧ (if i = seed then fire else amb) ≤ fire
    split_ifs
    · exact ⟨hle, le_rfl⟩
    · exact ⟨le_rfl, hle⟩
  | k + 1, i => ⟨le_min hle (le_max_left _ _), min_le_left _ _⟩

omit [DecidableEq V] in
theorem nbMean_le {T : V → ℝ} {i : V} {a : ℝ} (hne : (nb i).length ≠ 0)
    (hT : ∀ j ∈ nb i, T j ≤ a) : nbMean nb T i ≤ a := by
  unfold nbMean
  have hpos : (0 : ℝ) < (nb i).length := by exact_mod_cast Nat.pos_of_ne_zero hne
  rw [div_le_iff₀ hpos]
  have hs := List.sum_le_length_nsmul ((nb i).map T) a (by
    intro x hx
    obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hx
    exact hT j hj)
  rw [List.length_map, nsmul_eq_mul] at hs
  linarith

omit [DecidableEq V] in
/-- An update never goes past the neighbour mean (`min(1, ·) ≤ 1`), whatever the sign of
`heatTransfer`. -/
theorem diffUpd_le (T : V → ℝ) (i : V) :
    diffUpd nb deg h T i ≤ T i ∨
      (diffUpd nb deg h T i ≤ nbMean nb T i ∧ (nb i).length ≠ 0) := by
  unfold diffUpd
  split_ifs with hc
  · right
    refine ⟨?_, hc.2.1⟩
    have h1 : (nbMean nb T i - T i) * min 1 (h / deg i) ≤ nbMean nb T i - T i :=
      mul_le_of_le_one_right (by linarith [hc.2.2]) (min_le_left _ _)
    linarith
  · left
    exact le_rfl

/-- **Diffusion is no faster than the hop front.** A node is above ambient at frame `k` only
if some walk of at most `k` open links reaches it from the seed: heat moves at most one link
per step. Needs only that the open-neighbour lists are symmetric (`csr` lists both ends of
every edge, and `blocked` is per edge). -/
theorem diffT_hot_reach (hsymm : ∀ i j, j ∈ nb i → i ∈ nb j) :
    ∀ k i, amb < diffT nb deg h amb fire seed k i →
      ∃ j ≤ k, SafeReach (openLink nb) noBurn seed j i := by
  intro k
  induction k with
  | zero =>
    intro i hi
    refine ⟨0, le_rfl, (safeReach_zero (openLink nb) noBurn seed i).2 ?_⟩
    by_contra hne
    have e : diffT nb deg h amb fire seed 0 i = amb := by
      show (if i = seed then fire else amb) = amb
      simp [hne]
    rw [e] at hi
    exact lt_irrefl _ hi
  | succ k ih =>
    intro i hi
    by_cases hs : i = seed
    · exact ⟨0, Nat.zero_le _, (safeReach_zero (openLink nb) noBurn seed i).2 hs⟩
    have e : diffT nb deg h amb fire seed (k + 1) i =
        min fire (max amb (diffUpd nb deg h (diffT nb deg h amb fire seed k) i)) := by
      show diffStep nb deg h amb fire seed (diffT nb deg h amb fire seed k) i = _
      unfold diffStep
      simp [hs]
    rw [e] at hi
    have hu : amb < diffUpd nb deg h (diffT nb deg h amb fire seed k) i := by
      rcases lt_max_iff.1 (lt_of_lt_of_le hi (min_le_right _ _)) with h' | h'
      · exact absurd h' (lt_irrefl _)
      · exact h'
    have key : amb < diffT nb deg h amb fire seed k i ∨
        ∃ j ∈ nb i, amb < diffT nb deg h amb fire seed k j := by
      by_contra hcon
      push Not at hcon
      obtain ⟨h1, h2⟩ := hcon
      rcases diffUpd_le nb deg h (diffT nb deg h amb fire seed k) i with h3 | ⟨h3, hne⟩
      · linarith
      · have := nbMean_le nb hne h2
        linarith
    rcases key with h1 | ⟨j, hj, h1⟩
    · obtain ⟨j, hjk, hr⟩ := ih i h1
      exact ⟨j, by omega, hr⟩
    · obtain ⟨j', hj'k, hr⟩ := ih j h1
      exact ⟨j' + 1, by omega, (safeReach_succ (openLink nb) noBurn seed j' i).2
        ⟨j, hr, hsymm i j hj, WithTop.coe_lt_top _⟩⟩

/-- The flood forecast's heat at frame `k`: `fire` once the flood has reached the node (it is
in `seen` after `k` layers of `floodTimeline`, that is `ignite[v] ≤ k`), `ambient` before. -/
noncomputable def floodHeat (k : ℕ) (v : V) : ℝ :=
  open Classical in
  if v ∈ (bfs (openLink nb) noBurn seed k).2 then fire else amb

omit [DecidableEq V] in
/-- The flood has reached `v` by layer `k` exactly when some walk of at most `k` open links
does (`GraphNumerics.bfs_spec` with no burn check). -/
theorem mem_flood_iff (k : ℕ) (v : V) :
    v ∈ (bfs (openLink nb) noBurn seed k).2 ↔ ∃ j ≤ k, SafeReach (openLink nb) noBurn seed j v := by
  rw [(bfs_spec (openLink nb) noBurn seed k).2]
  exact Iff.rfl

omit [DecidableEq V] in
/-- The flood never cools: a burning node stays burning. -/
theorem floodHeat_monotone (hle : amb ≤ fire) (v : V) :
    Monotone fun k => floodHeat nb amb fire seed k v := by
  intro k k' hkk'
  simp only [floodHeat]
  split_ifs with h1 h2 h2
  · exact le_rfl
  · obtain ⟨j, hj, hr⟩ := (mem_flood_iff nb seed k v).1 h1
    exact absurd ((mem_flood_iff nb seed k' v).2 ⟨j, hj.trans hkk', hr⟩) h2
  · exact hle
  · exact le_rfl

/-- **At every frame the hop front is at least as hot as diffusion** (same graph, walls,
seed and clock). -/
theorem diffT_le_floodHeat (hle : amb ≤ fire) (hsymm : ∀ i j, j ∈ nb i → i ∈ nb j)
    (k : ℕ) (v : V) : diffT nb deg h amb fire seed k v ≤ floodHeat nb amb fire seed k v := by
  simp only [floodHeat]
  split_ifs with hv
  · exact (diffT_bounds nb deg h amb fire seed hle k v).2
  · by_contra hlt
    push Not at hlt
    obtain ⟨j, hj, hr⟩ := diffT_hot_reach nb deg h amb fire seed hsymm k v hlt
    exact hv ((mem_flood_iff nb seed k v).2 ⟨j, hj, hr⟩)

/-- Diffusion only heats for `heatTransfer ≥ 0` (the code's 1.2): every frame is at least the
one before, so the forecast's running maximum changes nothing for it. -/
theorem diffT_monotone (hle : amb ≤ fire) (hh : 0 ≤ h) (i : V) :
    Monotone fun k => diffT nb deg h amb fire seed k i := by
  refine monotone_nat_of_le_succ fun k => ?_
  have hb := diffT_bounds nb deg h amb fire seed hle k i
  show diffT nb deg h amb fire seed k i ≤
    diffStep nb deg h amb fire seed (diffT nb deg h amb fire seed k) i
  unfold diffStep
  refine le_min hb.2 (le_trans ?_ (le_max_right _ _))
  split_ifs with hs
  · exact hb.2
  · unfold diffUpd
    split_ifs with hc
    · have : 0 ≤ (nbMean nb (diffT nb deg h amb fire seed k) i - diffT nb deg h amb fire seed k i) *
          min 1 (h / deg i) :=
        mul_nonneg (by linarith [hc.2.2]) (le_min zero_le_one (div_nonneg hh (Nat.cast_nonneg _)))
      linarith
    · exact le_rfl

/-- The running maximum is monotone in the sequence. -/
theorem runMax_le_runMax {T T' : ℕ → ℝ} (hT : ∀ k, T k ≤ T' k) :
    ∀ k, runMax T k ≤ runMax T' k
  | 0 => hT 0
  | k + 1 => max_le_max (runMax_le_runMax hT k) (hT (k + 1))

/-- A closing step is at most any untenable step before the horizon. -/
theorem closeStep_le_of {P : ℕ → ℝ} {θ : ℝ} {K0 k : ℕ} (hk : k < K0) (hθ : θ ≤ P k) :
    closeStep P θ K0 ≤ k := by
  unfold closeStep
  split_ifs with hex
  · exact Nat.find_min' hex ⟨hk, hθ⟩
  · exact absurd ⟨k, hk, hθ⟩ hex

/-- Before the horizon, a node is untenable at its closing step. -/
theorem closeStep_spec {P : ℕ → ℝ} {θ : ℝ} {K0 : ℕ} (h : closeStep P θ K0 < K0) :
    θ ≤ P (closeStep P θ K0) := by
  unfold closeStep at h ⊢
  split_ifs at h ⊢ with hex
  · exact (Nat.find_spec hex).2
  · exact absurd h (lt_irrefl _)

/-- A hotter stored heat closes a node no later, for every threshold. -/
theorem closeStep_anti {P P' : ℕ → ℝ} (hP : ∀ k, P k ≤ P' k) (θ : ℝ) (K0 : ℕ) :
    closeStep P' θ K0 ≤ closeStep P θ K0 := by
  rcases lt_or_ge (closeStep P θ K0) K0 with hlt | hge
  · exact closeStep_le_of hlt ((closeStep_spec hlt).trans (hP _))
  · exact (closeStep_le P' θ K0).trans hge

/-- **The hop front closes every node no later than diffusion**, for every threshold `θ`
(`buildForecast`'s closing step of the stored heat `peak`, the running maximum). -/
theorem hop_closeStep_le (hle : amb ≤ fire) (hsymm : ∀ i j, j ∈ nb i → i ∈ nb j) (θ : ℝ)
    (K0 : ℕ) (v : V) :
    closeStep (runMax fun k => floodHeat nb amb fire seed k v) θ K0 ≤
      closeStep (runMax fun k => diffT nb deg h amb fire seed k v) θ K0 :=
  closeStep_anti (runMax_le_runMax fun k => diffT_le_floodHeat nb deg h amb fire seed hle hsymm k v)
    θ K0

/-- A forecast's closing time in seconds, `closes[v] = closeStep · dt` (`min(u, H)`), as an
extended real for `Deadline`. -/
noncomputable def closeTime (P : ℕ → ℝ) (θ : ℝ) (K0 : ℕ) (dt : ℝ) : EReal :=
  ((closeStep P θ K0 : ℝ) * dt : ℝ)

/-- The hop front's `closes` are never later than diffusion's (`dt ≥ 0`). -/
theorem hop_close_le (hle : amb ≤ fire) (hsymm : ∀ i j, j ∈ nb i → i ∈ nb j) (θ : ℝ)
    (K0 : ℕ) {dt : ℝ} (hdt : 0 ≤ dt) (v : V) :
    closeTime (runMax fun k => floodHeat nb amb fire seed k v) θ K0 dt ≤
      closeTime (runMax fun k => diffT nb deg h amb fire seed k v) θ K0 dt :=
  EReal.coe_le_coe_iff.2 (mul_le_mul_of_nonneg_right
    (Nat.cast_le.2 (hop_closeStep_le nb deg h amb fire seed hle hsymm θ K0 v)) hdt)

/-- **The hop surrogate's escape deadlines are never later than diffusion's.** -/
theorem hop_deadline_le (hle : amb ≤ fire) (hsymm : ∀ i j, j ∈ nb i → i ∈ nb j) (θ : ℝ)
    (K0 : ℕ) {dt : ℝ} (hdt : 0 ≤ dt) (E : V → V → Prop) (τ : V → V → ℝ) (X : V → Bool)
    (v : V) :
    deadline E τ (fun w => closeTime (runMax fun k => floodHeat nb amb fire seed k w) θ K0 dt)
        X v ≤
      deadline E τ (fun w => closeTime (runMax fun k => diffT nb deg h amb fire seed k w) θ K0 dt)
        X v :=
  deadline_mono E τ X (hop_close_le nb deg h amb fire seed hle hsymm θ K0 hdt) v

/-- **Routes found on the hop surrogate are safe under diffusion.** -/
theorem hop_safe (hle : amb ≤ fire) (hsymm : ∀ i j, j ∈ nb i → i ∈ nb j) (θ : ℝ) (K0 : ℕ)
    {dt : ℝ} (hdt : 0 ≤ dt) (E : V → V → Prop) (τ : V → V → ℝ) (ws : List V) (v : V) (t : ℝ)
    (hs : SafeWalk E τ (fun w => closeTime (runMax fun k => floodHeat nb amb fire seed k w) θ K0
      dt) t v ws) :
    SafeWalk E τ (fun w => closeTime (runMax fun k => diffT nb deg h amb fire seed k w) θ K0 dt)
      t v ws :=
  safeWalk_mono E τ (hop_close_le nb deg h amb fire seed hle hsymm θ K0 hdt) ws v t hs

/-- **The hop front's FED rates are never below diffusion's**: the rate of the stored heat
(`convectiveDoseRate(peak)`) at every step and node. -/
theorem hop_rate_ge (hle : amb ≤ fire) (hsymm : ∀ i j, j ∈ nb i → i ∈ nb j) (k : ℕ)
    (v : V) :
    doseRate (runMax (fun k => diffT nb deg h amb fire seed k v) k) ≤
      doseRate (runMax (fun k => floodHeat nb amb fire seed k v) k) :=
  doseRate_monotone
    (runMax_le_runMax (fun k => diffT_le_floodHeat nb deg h amb fire seed hle hsymm k v) k)

omit [DecidableEq V] in
/-- For the flood the stored running maximum is its own heat. -/
theorem floodHeat_runMax (hle : amb ≤ fire) (v : V) (k : ℕ) :
    runMax (fun k => floodHeat nb amb fire seed k v) k = floodHeat nb amb fire seed k v :=
  runMax_of_monotone (floodHeat_monotone nb amb fire seed hle v) k

end Hop

end FireEgress.Surrogate

/-! ## Axioms -/

#print axioms FireEgress.Surrogate.coe_sub_lt_iff
#print axioms FireEgress.Surrogate.sub_coe_le_iff
#print axioms FireEgress.Surrogate.safeWalk_mono
#print axioms FireEgress.Surrogate.safeSet_mono
#print axioms FireEgress.Surrogate.deadline_mono
#print axioms FireEgress.Surrogate.safeWalk_shift
#print axioms FireEgress.Surrogate.deadline_le_add
#print axioms FireEgress.Surrogate.deadline_lipschitz
#print axioms FireEgress.Surrogate.deadline_lipschitz_real
#print axioms FireEgress.Surrogate.deadline_eq_bot_of_close
#print axioms FireEgress.Surrogate.withMargin_nonneg
#print axioms FireEgress.Surrogate.sub_le_withMargin
#print axioms FireEgress.Surrogate.withMargin_top
#print axioms FireEgress.Surrogate.withMargin_coe
#print axioms FireEgress.Surrogate.withMargin_le_self
#print axioms FireEgress.Surrogate.withMargin_antitone
#print axioms FireEgress.Surrogate.withMargin_withMargin
#print axioms FireEgress.Surrogate.withMargin_conservative
#print axioms FireEgress.Surrogate.le_withMargin_add
#print axioms FireEgress.Surrogate.withMargin_safe
#print axioms FireEgress.Surrogate.withMargin_deadline
#print axioms FireEgress.Surrogate.compare_deadline
#print axioms FireEgress.Surrogate.walkDose_le
#print axioms FireEgress.Surrogate.goodSet_mono
#print axioms FireEgress.Surrogate.doseDeadline_mono
#print axioms FireEgress.Surrogate.leastDose_le_surrogate
#print axioms FireEgress.Surrogate.doseDeadline_le_add
#print axioms FireEgress.Surrogate.doseDeadline_lipschitz
#print axioms FireEgress.Surrogate.hotDose_le_of_rate_le
#print axioms FireEgress.Surrogate.diffT_bounds
#print axioms FireEgress.Surrogate.nbMean_le
#print axioms FireEgress.Surrogate.diffUpd_le
#print axioms FireEgress.Surrogate.diffT_hot_reach
#print axioms FireEgress.Surrogate.mem_flood_iff
#print axioms FireEgress.Surrogate.floodHeat_monotone
#print axioms FireEgress.Surrogate.diffT_le_floodHeat
#print axioms FireEgress.Surrogate.diffT_monotone
#print axioms FireEgress.Surrogate.runMax_le_runMax
#print axioms FireEgress.Surrogate.closeStep_le_of
#print axioms FireEgress.Surrogate.closeStep_spec
#print axioms FireEgress.Surrogate.closeStep_anti
#print axioms FireEgress.Surrogate.hop_closeStep_le
#print axioms FireEgress.Surrogate.hop_close_le
#print axioms FireEgress.Surrogate.hop_deadline_le
#print axioms FireEgress.Surrogate.hop_safe
#print axioms FireEgress.Surrogate.hop_rate_ge
#print axioms FireEgress.Surrogate.floodHeat_runMax
