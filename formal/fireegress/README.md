# Lean 4 proofs for Studio's fire-egress models

Machine-checked results behind Studio's fire-egress models and the model corrections in
[wiki/research/fire-egress-methods-2026-10-27.md](../../wiki/research/fire-egress-methods-2026-10-27.md)
(the literature review). Lean 4.34.1 with Mathlib (the same pin as the lab's
`earth3dp` project). Checked on 2026-10-27: all 11 theorems depend only on
Lean's standard axioms (`propext`, `Classical.choice`, `Quot.sound`) and no
`sorry`; see `axioms-2026-10-27.txt`.

| File | Theorem | What it says | Where it shows in Studio |
| --- | --- | --- | --- |
| Temperature.lean | `classic_overshoots` | Classic's step takes a dead end at 20 C beside a 120 C neighbour to 140 C. | Finding W1: why the coefficient is capped. |
| | `step_between` | With a coefficient in [0, 1] the update lies between the old value and the neighbour mean. | The capped step cannot overshoot. |
| | `capped_coefficient_unit` | `min 1 (k / deg)` lies in [0, 1]. | The cap always applies cleanly. |
| | `cap_inactive_from_degree_two` | For degree 2 or more the cap equals Classic's `1.2 / deg`. | Results stay comparable with Classic. |
| | `step_bounded` | One-step discrete maximum principle over any neighbour set. | Temperatures stay within 20-120 C without clipping. |
| Shaping.lean | `shaped_eq` | Shaped return = return + gamma^T Phi(s_T) - Phi(s_0). | The identity behind shaping. |
| | `shaping_preserves_order` | With Phi = 0 at terminals, shaping cannot reorder two trajectories from one start. | Finding W5: why terminals now have zero potential. |
| | `nonzero_terminal_flips_order` | With a non-zero terminal potential, the order of two outcomes flips. | The bug W5 fixed was real. |
| Heuristic.lean | `consistent` | The straight-line heuristic is consistent when weights are at least edge lengths. | A* closes nodes safely. |
| | `admissible` | Consistency plus h(goal) = 0 gives h(start) <= cost of any path. | A* returns the true shortest route. |
| | `hazard_weight_ge_length` | `len * (1 + alpha * h) >= len` for alpha, h >= 0. | Hazard reweighting keeps A* exact. |

## Second batch: every fire and egress equation (2026-10-27)

A multi-agent review extracted 90 equations from the literature review,
checked each against a source the agent actually read (71 confirmed, 11
corrected, 8 unverifiable; the tables, with units, sources and checks, are in
[wiki/research/fire-equations-verified-2026-10-27.json](../../wiki/research/fire-equations-verified-2026-10-27.json)),
and formalised them. `lake build` passes with no warnings; all 175 axiom
checks list only the standard axioms ([axioms-batch1-2026-10-27.txt](axioms-batch1-2026-10-27.txt)).

| File | What it covers |
| --- | --- |
| Dim.lean | A decidable SI dimension type (exponents of M, L, T, K scaled by 60 so fractional powers stay exact). |
| FireDynamics.lean | t-squared fires, Heskestad and McCaffrey plumes, Alpert and CFAST ceiling jets, MQH, Thomas and Babrauskas flashover, the two-zone model's pressure, volume and temperature equations, vent flow: units of every equation, monotonicity, inverses, and the branch jumps of the piecewise correlations. |
| Tenability.lean | FED for heat (convective, radiative) and asphyxiants, ASET/RSET, the hydraulic egress model, Weidmann: units, monotonicity, additivity, thresholds. |
| GraphNumerics.lean | Studio's own numerics: the explicit-Euler diffusion matrix (L1), the discrete maximum principle for all steps (L2), FIFO composition (L10a), waiting never helps (L10b), suffix optimality (L11), the earliest-escape BFS at specification level (L7). |

What the proofs found, beyond confirming:

- Two claims in our own review were false, and Lean proves the
  counterexamples: L2's boundary-condition clause (the correct bound must
  include 0: `L2_dmp_dirichlet`), and that every prefix of an earliest-arrival
  walk is earliest-arrival under plain FIFO (`R3_prefix_not_ea_under_fifo`).
- SFPE walking speed with the low-density switch at 0.54 persons/m2 is not
  monotone: speed jumps up there (`E2_not_antitone`); a switch at 75/133
  restores it. The stated upper density 3.8 exceeds the jam density 1/0.266,
  where speed is negative: cap density at 1/a.
- The MQH flashover constants 740 and 6.85 are dimensionally consistent but
  differ by about 4% in Q_fo (`FD04_740_vs_685`).
- The FED heat dose is a units trap the dimension type cannot see: kelvin
  instead of deg C reaches the 0.3 threshold within 4 s instead of 564 min
  (`T3_kelvin_trap`); minutes vs seconds is invisible too.
- Hidden assumptions stated: monotone (fixed) ignition times for the BFS;
  T_u > T_l for the plume-filling bound; a fixed room volume for the zone
  pressure equation.

Not formalised in this batch (each with a reason in the agents' reports):
full Dijkstra/A* correctness, the time-expanded minimum-dose DP, the eikonal
PDE, probabilistic pre-movement models, and some worked numbers.

## Third batch: the route and geometry claims (2026-10-27)

What a user sees in a route, each with a JavaScript property test in
`topologicpy-web-frontend/src/core/*.property.test.mjs` that checks the same
statement against the real functions, and an adversarial review of every
statement against the code and the app's wording. All 12 modules build with no
warnings; 294 axiom results, standard axioms only ([axioms-all-2026-10-27.txt](axioms-all-2026-10-27.txt)).

| File | Proved | Scope and limits |
| --- | --- | --- |
| AStar.lean | Reduced costs are non-negative under a consistent heuristic; walk costs telescope; Dijkstra's label-setting invariant; every A* run is a Dijkstra run on reduced costs; `astar_correct`, `astar_straight_line_correct` (L9). | Least cost on the walkable graph between the snapped nodes, for the browser engine. Not the heap's lazy deletion, not floating point, not TGraph (checked in tests). Needs every weight >= the link's length; fire.js now never produces a NaN weight. |
| Straighten.lean | A subsequence polyline with the same ends is never longer; shortcut clearance bound. | "0.25 m off walls" is not guaranteed: at s = 0.5 a run whose ends are 0.25 m clear keeps about 0.24 m between samples, and near doorways only as much as its ends. App wording corrected. |
| Segments.lean | `crossTest_iff`: the orientation test is true exactly when the closed segments meet and no endpoint lies on the other; soundness and completeness for proper crossings. | Touching, T-junctions and collinear overlap return false; the three inside samples catch most, not all. Exact arithmetic; the code now compares signs instead of multiplying (no underflow). |
| Lattice.lean | Octile distance is attained and is a lower bound for any 8-connected path; Euclidean <= octile <= sqrt(4 - 2 sqrt 2) Euclidean (8.2%); no corner cutting. | The code now blocks a diagonal when both of its axis detours are blocked, matching `diagonal_detour` (fired on 2 diagonals across 16 models; connectivity unchanged). |
| Frames.lean | (x, y, z) -> (x, z, -y) is a proper rotation; rigid motions preserve polyline lengths; That Open's matrix can never be a reflection. | For a model rotated in plan, That Open rounds the rotation to about 1e-3, so lengths are exact only for unrotated models. Docstrings corrected. |

## Fourth batch: routing while the fire changes (2026-10-27)

| File | What is proved | What it means for Studio |
|---|---|---|
| Yen.lean | Yen's algorithm returns k shortest loopless paths: every list it returns, at any length it reaches, has no loopless path left out that is cheaper than its last one (`yen_kshortest`). | The "next cheapest" alternative routes are exact. The different-space mode is a bounded heuristic and is described as one: it may stop before finding every route. |
| TimeDependent.lean | Time-dependent Dijkstra (Dreyfus label setting) gives the exact earliest arrival when every link is FIFO (leaving later never arrives earlier) and travel never takes negative time; a link closed by the fire is an arrival of top (`td_dijkstra_correct`, `td_dijkstra_EA_withTop`). Waiting never helps; replanning from the evacuee's current node and time keeps the plan's arrival (`replan_consistent`). FIFO is necessary (`fifo_necessary_every_run`). | The specification the forecast router must meet. |
| MinDose.lean | Backward induction on the time-expanded graph gives the least fractional effective dose of any schedule reaching an exit by the horizon (`dp_exact`); a schedule below a dose limit exists iff the value is below it (`safe_iff`); if every move dose is non-decreasing in the departure step and waiting doses are non-negative, some optimal schedule never waits (`no_holdover`), discharged for convective heat with a non-decreasing temperature (`no_holdover_convective`). | The specification the least-dose router must meet. |

## Building

Lake caches are gigabytes, so build outside OneDrive. The working copy is at
`C:\Users\lmurugesan\lean\fireegress`; to rebuild elsewhere:

```
# copy this folder somewhere outside OneDrive, then:
lake exe cache get      # prebuilt Mathlib
lake build              # ~2 minutes after the cache
lake env lean AxiomCheck.lean
```

Not yet proved (from the review's list): the earliest-escape BFS is exact under
a growing fire (L7), A* optimality as a whole (L9), time-dependent Dijkstra
(L10c), and dimensional checks of the zone-model and FED equations (L16).
