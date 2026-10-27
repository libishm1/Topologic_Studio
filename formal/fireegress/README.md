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
