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
| | `admissible` | Consistency plus h(goal) = 0 gives h(start) <= cost of any path. | With `consistent`, gives `AStar.astar_correct`: A* returns a least-cost route. |
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
| Lattice.lean | Octile distance is attained and is a lower bound for any 8-connected path; Euclidean <= octile <= sqrt(4 - 2 sqrt 2) Euclidean (8.24%); no corner cutting. | The code now blocks a diagonal when both of its axis detours are blocked, in the spirit of `diagonal_detour` (the Lean model has no wall-blocked links) (fired on 2 diagonals across 16 models; connectivity unchanged). |
| Frames.lean | (x, y, z) -> (x, z, -y) is a proper rotation; rigid motions preserve polyline lengths; That Open's matrix can never be a reflection. | For a model rotated in plan, That Open rounds the rotation to about 1e-3, so lengths are exact only for unrotated models. Docstrings corrected. |

## Fourth batch: routing while the fire changes (2026-10-27)

| File | What is proved | What it means for Studio |
|---|---|---|
| Yen.lean | Yen's algorithm returns k shortest loopless paths: every list it returns, at any length it reaches, has no loopless path left out that is cheaper than its last one (`yen_kshortest`). | The "next cheapest" alternative routes are exact. The different-space mode is a bounded heuristic and is described as one: it may stop before finding every route. |
| TimeDependent.lean | Time-dependent Dijkstra (Dreyfus label setting) gives the exact earliest arrival when every link is FIFO (leaving later never arrives earlier) and travel never takes negative time; a link closed by the fire is an arrival of top (`td_dijkstra_correct`, `td_dijkstra_EA_withTop`). Waiting never helps; replanning from the evacuee's current node and time keeps the plan's arrival (`replan_consistent`). FIFO is necessary (`fifo_necessary_every_run`). | The specification the forecast router must meet. |
| MinDose.lean | Backward induction on the time-expanded graph gives the least fractional effective dose of any schedule reaching an exit by the horizon (`dp_exact`); a schedule below a dose limit exists iff the value is below it (`safe_iff`); if every move dose is non-decreasing in the departure step and waiting doses are non-negative, some optimal schedule never waits (`no_holdover`), discharged for convective heat with a non-decreasing temperature (`no_holdover_convective`). | The specification the least-dose router must meet. |

## Fifth batch: learning, the fire front, rounding, and the forecast routers (2026-10-27/28)

| File | What is proved | What it means for Studio |
|---|---|---|
| QLearning.lean | On a deterministic problem the Bellman operator is a sup-norm contraction (`bellman_contracting`); value iteration converges at rate gamma^n (`value_iteration_error`); asynchronous Q-learning with steps in [alpha_min, 1], alpha_min > 0, converges on a successor-closed set of pairs (for example those reachable from the start) when every pair of it is updated infinitely often (`qlearning_tendsto`, `qlearning_tendsto_weighted`), and on a finite horizon for any gamma <= 1 (`qlearning_tendsto_dag`); n sweeps equal backward induction (`dag_value_iteration_exact`); shaping with zero terminal potential gives Q*_shaped = Q* - Phi (`shaped_qstar_eq`). Counterexamples: under the shaped Q*, non-zero potential at the step limit makes burning optimal on the last step (`horizon_potential_prefers_fire`). | Found that rl.js gave the step limit a non-zero potential and applied the step-limit rule without a clock; both fixed. The panel now shows the policy's return beside the exact optimum. Says nothing about a finite run. |
| Eikonal.lean | T(x) = norm(x - x0)/F solves the eikonal equation away from the source and is the least travel time over all curves (`pointArrival_eikonal`, `pointArrival_isLeast_curve`); with several sources it is not differentiable where fronts meet (`multiArrival_not_differentiableAt_tie`); the open 8-connected lattice front lies between 1 and sqrt(4 - 2 sqrt 2) times it (`latArrival_le_K`), the bound is approached (above 1.0823 at offset (12, 5)) and does not shrink under refinement (`latArrival_no_refinement`); with walls the graph front is never early provided its links lie in free space (`graph_front_never_early`); Dijkstra's labels satisfy the discrete eikonal equation and, with positive link lengths, are its only solution (`dijkstra_discrete_eikonal`, `bellman_unique`). | The 8.2% bound holds on open floor only; with walls only "never early" is proved. The distance-band fire is an early bound (`radial_never_late`); the spread-along-links fire is a hop count, not this front. |
| FloatError.lean | Standard rounding model; the orientation determinant's error bound and a sign certificate the code can evaluate (`flDet_sign_certain_code`); the browser's segment test equals the exact test whenever all four certificates hold (`segmentsCross_float_exact_code`), and can disagree only for a point within about 4.4e-16 times the local distance of a line (`misclassified_near`); summation error of a route (`route_sum_rel_err`); closed-set A* with a heuristic consistent up to delta is within k delta of optimal, and the naive bound delta fails (`astar_delta_near_optimal`, counterexample); float32 coordinates move a link length by at most 1.2e-5 m (`studio_float32_length_err`). | At building scale the rounding bounds are far below the lattice's own error (a judgement from the bounds, not a theorem); float32 storage is the largest effect. |
| Forecast.lean | The forecast's running maximum never falls and coarsening is conservative (`runMax_monotone`, `coarse_conservative`); the FED rate and link doses never fall in time (`doseRate_monotone`, `linkDose_monotone`); the router's closing links are FIFO and its first exit is the earliest safe arrival (`earliestArrival_exact`, `earliestArrival_none`); dominance is preserved and waiting never helps (`dominance_preserved`, `waiting_never_helps`, `dp_no_holdover`); the dose search's front is complete, sound, starts at the earliest arrival and ends at the least dose (`front_complete`, `front_sound`, `front_head_earliest`, `front_last_leastDose`); a route planned on the fire at departure can walk into it (`snapshot_plan_unsafe`). | The two routers in the Simulate step meet TimeDependent's and MinDose's conditions. Found a tie bug (`tie_counterexample`) and a negative-time edge case, both fixed. |
| Deadline.lean | Safe departures from a node are downward closed and open, and leaving at t is safe iff t < L (`safeSet_downward`, `mem_safeSet_iff`); L satisfies the (max, min) Bellman equation and is its least supersolution (`deadline_bellman`, `deadline_le_of_super`); the max-first label setting computes L exactly and the forward check agrees with it (`escapeDeadlines_correct`, `departureSafe_correct`); the least dose never falls with a later departure, so the dose-limited deadline is well defined, at most L, and equal to L with no limit (`leastDose_mono`, `doseDeadline_le_deadline`, `doseDeadline_top`); bisection brackets it (`doseDeadline_bisection`); the grid map is MinDose's recursion without holdover, non-decreasing in time, and never later than the exact deadline (`gridV_eq_val`, `gridV_mono`, `studio_map_le_exact`); a one-link example where the limit binds (`example_dose_limit_binds`). | The recommended method (C1). The proofs found that the map could round a link down by up to 1e-9 of a grid step (`epsilon_rounds_down`); the property tests found that leaving at exactly L can pass in doubles; both fixed. The map's gap below the exact deadline is not bounded. |

Full build on 2026-10-28: 20 modules, 863 theorems, 849 of them with an axiom print (838 in the modules and 11 in AxiomCheck.lean), every one on Lean's standard axioms or none (`axioms-all-2026-10-28.txt`, since regenerated with the sixth batch), no `sorry`, no warnings.

## Sixth batch: surrogates, the two-zone model, cores and exits (2026-10-28)

Three modules written against `surrogate.js` (gap LG7), `zone.js` with its specification
([wiki/research/zone-model-spec-2026-10-28.md](../../wiki/research/zone-model-spec-2026-10-28.md), properties P1 to P5),
`cores.js` (`walkingLine`) and `ifcSemantics.js` (`exitsPerPart`, `exitDoors`). Each module's header names the
function each theorem covers and lists what it leaves out.

| File | What is proved | What is not covered |
|---|---|---|
| Surrogate.lean (47 results) | `withClosingMargin` never closes a node later than its input, a larger margin never closes it later, and margins add as its `margin` field records (`withMargin_le_self`, `withMargin_antitone`, `withMargin_withMargin`). Earlier closing times and higher doses only remove safe routes, bring the hard and dose-limited deadlines forward and raise the least dose (`safeWalk_mono`, `deadline_mono`, `doseDeadline_mono`, `leastDose_le_surrogate`, `hotDose_le_of_rate_le`). The escape deadline L is 1-Lipschitz in the closing times in the sup norm (`deadline_le_add`, `deadline_lipschitz`, `deadline_lipschitz_real`), and so is the dose-limited deadline when the doses are unchanged and never fall with the departure time (`doseDeadline_lipschitz`), so a surrogate within e of the true closing times, moved e earlier, is conservative and its deadlines lie within 2e (`withMargin_deadline`); what `compareForecasts` reports bounds the deadlines both ways (`compare_deadline`). On the same open links the hop front is at least as hot as diffusion at every step (`diffT_hot_reach`, `diffT_le_floodHeat`), so for every threshold it closes no node later and its deadlines, routes and rates are conservative (`hop_closeStep_le`, `hop_deadline_le`, `hop_safe`, `hop_rate_ge`). | Arithmetic is exact; the code's tolerances (1e-9 on closing times, 1e-6 on rates) enter as stated inequalities. The hop comparison assumes symmetric open-neighbour lists and an ambient value at most the fire value, both true of the code; the frames of the two forecasts are compared step by step, not through `compareForecasts`' frame lookup. When a surrogate changes the FED rates as well, the dose-limited deadline is shown monotone only; no bound in the rate error is proved. |
| Zone.lean (44 results) | P1: the heat release never falls and the lining conductance never rises, across the penetration time included, so the MQH rise and the flashover latch never fall (`heatRelease_monotoneOn`, `liningConductance_antitoneOn`, `rawRise_monotoneOn`, `fireRoomRise_monotone`). P2: an Euler step never raises the interface and keeps it between the floor and the ceiling, for any inflow, outflow and density (`layer_antitone`). P3: door outflow rises with the upstream temperature and falls with the upstream interface, the fed room's rise is the specification's R3 and rises with its inflow and incoming rise, and by induction along the breadth-first order every room heats and fills monotonically (`doorOutflow_mono`, `fedRoomRise_eq_spec`, `fedRoomRise_mono`, `zone_rooms_monotone`). P4: node temperatures and FED rates never fall and untenability is an up-set in time, so the forecast hypotheses of Forecast.lean and Deadline.lean hold (`nodeTemperature_monotone`, `untenable_upset`, `nodeRate_monotone`, `node_closed_iff`). P5: the units of S2, S3 and R3 close, and F1, S1 and F2 are FD-01, FD-08, FD-17 and FD-18 (`P5_units`). | Arithmetic is real; that every double operation stays monotone is checked by `zone.test.mjs`, not proved. The simulation is modelled by its recurrences along a ranking in which feeding rooms come first; that `roomOrder` gives one is read from the code. Frame sampling beyond the running maximum, and P6 (convergence of the Euler step), are not covered. The dimension type does not see the kW and kJ prefixes. |
| Cores.lean (38 results) | Any polyline is at least the square root of P squared plus V squared long, P its plan length and V its height variation, so at least as long as the straight rise over its plan length; monotone heights are not needed (`len3_ge`, `len3_ge_ends`). The flights of `walkingLine` have plan length (hb - ha) / r * g and rise hb - ha, so the line is at least (hb - ha) times the square root of g squared plus r squared, over r, long: between 1.88 and 1.89 times the rise for the default 175 mm risers and 280 mm goings (`linePts_planLen`, `walkingLine_length_ge`, `default_ratio`); the mirrors, the axis swap and dropped repeated points keep the bound (`walkingLine_out_length_ge`). Heights along the returned line never decrease for ha at most hb (`walkingLine_heights`). Every part of the graph with at least max(50, 0.02 n) nodes that holds a fallback door ends with an exit that every node of the part can reach (`exitsPerPart_serves`, `large_part_reaches_exit`), and `exitsPerPart` adds a door only in such a part with no flagged exit (`mem_exitsPerPart`). | The 1e-6 de-duplication of points that are close but not equal is not covered by the length bound; in exact arithmetic it only drops repeated points unless a piece of a flight is shorter than 1e-6 m, which is not proved impossible. The numbers of flights and pieces, the lanes and the start are free parameters. `componentLabels` is assumed to return a component labelling. |

What the proofs found. None of the four files is wrong in what these modules state. Two
hypotheses were stronger than needed: the header of `surrogate.js` restricts the hop
surrogate to thresholds above ambient, but it is conservative for every threshold
(`hop_closeStep_le` has no threshold hypothesis), and P2 of the zone specification asks for
non-decreasing inputs, but the clamp alone keeps the interface from rising
(`layer_antitone`). Diffusion's only-heating property, which `forecast.js` states in its
header, needs a non-negative `heatTransfer` (`diffT_monotone`); the hop comparison does not.

Full build on 2026-10-28 after this batch: 23 modules, 991 theorems and lemmas (counted as
`theorem` and `lemma` declarations), 978 of them with an axiom print (967 in the modules and
11 in AxiomCheck.lean), every one on Lean's standard axioms or none
(`axioms-all-2026-10-28.txt`, regenerated from the build log), no `sorry`, no warnings.

## Building

Lake caches are gigabytes, so build outside OneDrive. The working copy is at
`C:\Users\lmurugesan\lean\fireegress`; to rebuild elsewhere:

```
# copy this folder somewhere outside OneDrive, then:
lake exe cache get      # prebuilt Mathlib
lake build              # ~2 minutes after the cache
lake env lean AxiomCheck.lean
```

Still outside the proofs: heaps, route reconstruction and termination of the search loops, and floating point beyond FloatError.lean. The earliest-escape search (L7), A* (L9), time-dependent Dijkstra (L10c) and the zone-model units (L16) that this paragraph once listed are now proved: `L7_bfs_layer_eq_earliest`, `astar_correct`, `td_dijkstra_correct`, `FD24_zone_units` and the `units_T*` theorems.