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

## Seventh batch: Fire-X, Flamescapes and the browser port (2026-11-05)

Three modules formalise the two papers behind Studio's Flamescapes fire model and the numerical
schemes of its JavaScript port: Fire-X (Wrede et al. 2025, https://doi.org/10.1145/3763338),
Flamescapes (Wrede, Wagner et al. 2026, https://doi.org/10.1145/3842518), and `flamescapes.js`,
`voxelize.js` and `flamescapesRun.js` as they stood on 2026-11-05. Every equation of the
transcriptions in [wiki/sources/papers/](../../wiki/sources/papers/) is a Lean definition stated
at a point; units are checked with `Dim` extended by an exponent of amount of substance
(`FireX.DN`), since both papers mix per-mole and per-kilogram quantities. Each module's header
maps every theorem to an equation number or a JS function and lists what it leaves out; the
port's theorems carry the name of the function they cover. The axiom prints of these three
modules are in `AxiomCheck.lean`, one line per theorem.

FireX.lean (78 results), Eqs. X1 to X32 with the boxes of Fig. 3, the legend of Fig. 17 and
the inline definitions I1 to I5:

| Equations | What is proved | What is not covered |
|---|---|---|
| X1 to X3, momentum, energy, species | The units of every term. Buoyancy lifts gas hotter than ambient, never faster than `\|g\|`, and under the ideal gas law is the Boussinesq deficit `(ρ_amb - ρ)/ρ_amb` (`X1_buoy_up`, `X1_buoy_lt_g`, `X1_boussinesq`); the radiative term cools hot gas; Fig. 3's boxes are X1 to X3 with X6 and X11 substituted. | The PDEs are stated by their right-hand sides; existence, uniqueness and the GPU schemes are not formalised. |
| X4 to X8, combustion | Eq. 4's bracket is the O₂ demand of `C_c H_h O_o`, so Eq. 4 releases 417 kJ per mole of O₂, 13.03 MJ per kg, Huggett's principle within 0.53% (`X4_per_kg_O2`); the six Table 1 heats are Eq. 4 per mole of fuel; Arrhenius rates are positive and rise with temperature; the reaction keeps `Σ Y_i` exactly when `Σ q_i M_i = 0`, which the six Table 1 reactions satisfy for any atomic masses; Eq. 8 with `Y_i / M_i` is the ideal gas law and falls with temperature. | The activation energies, which the paper does not give. |
| X9 to X11, X26 to X32, evaporation | Eq. 9 is the per-step evaporated density of Sec. 5.4 (`X9_eq_rho_e`); the evaporated mass is the available energy over the energy per kilogram; mixing (Eqs. 30 to 32) conserves enthalpy, each species' mass and momentum; `ρ'` lies in `(0, 1]`. | Where the vapour enters `Y_i` (the paper does not say). |
| X12 to X25, SPH and particle-grid transfer | The kernels lie in `[0, 1]` and fall with distance; the near-density is at most the density; the viscous pair terms cancel (momentum); Eqs. 23 and 24 are weighted means, between their inputs; Eq. 21 returns a uniform mass unchanged. | Neighbour search and the SPH integrator; vectors are `Fin 3 → ℝ`. |

Flamescapes.lean (60 results), Eqs. F1 to F19, the supplement's S1 and its two algorithms, and
the inline definitions:

| Equations | What is proved | What is not covered |
|---|---|---|
| F1 to F5, I1 to I3, voxels and interior fuel | Volume fractions are non-negative and sum to at most one when no ray spends longer in material than across the voxel; moisture is at most the biomass fraction; draws with mean `M_b / N_b` have expected total `M_b` (`F5_expected_total`); a destroyed module stays destroyed. | The code's random draws; the fuel graph. |
| F6, F7, the gas flow | Eq. 7 follows from the ideal gas law over the mixture once `ρ_amb` is a pressure, and density falls with temperature; the buoyancy term read as intended is the port's `\|g\| (ρ_amb/ρ_g - 1)` and equals Fire-X's `-g (1 - T_amb/T)` times `ρ_amb/ρ_g = T/T_amb` (`F6_vs_X1`), agreeing to first order with difference `\|g\| (T - T_amb)²/(T T_amb)`, and four times Fire-X's at 1200 K over 300 K. | Compressibility; the pressure solve. |
| F8 to F15, combustion and the solid | Eqs. 9, 10 and 12 are Fire-X's Eqs. 6, 5 and 7, and Table 3's gas values are Fire-X's methane; Arrhenius rates are positive and rise with temperature; Eq. 13's specific heat (and conductivity) lies between the constituents', and its density between `Σφ` times the least and largest; the units of every term. | The conversion from solid volume fraction to gas mass fraction in Eqs. 14 and 15, which the paper does not state; the unprinted Arrhenius forms of `v_p,s` and `v_c,s`. |
| F16 to F19, I4, radiation and the interface | Eq. 16's emission has the sign of `T_s - T_amb`; absorption probabilities lie in `[0, 1)`, grow with path length and compose along a ray, `1 - e^{-a(l₁+l₂)} = 1 - (1 - P₁)(1 - P₂)`; the interface carries heat out of a solid hotter than gas and ambient; the initial state is at rest. | Wavelength dependence (none in the paper). |
| Supplement algorithms, S1, S-I1, S-I2 | Algorithm 1 conserves energy in expectation, and what passes `n` segments is `E exp(-Σ κ Y_j ℓ_j)`; Algorithm 2's line 7, divided by the heat capacity, deposits exactly the embers' excess energy; the flame-front maximum can only raise the gas reaction rate; the building rules' compatibility is symmetric. | The wave function collapse solver; ember and flame-front motion. |

FlamescapesPort.lean (132 results), the browser schemes as of commit 30c019a, which fixed the six
defects these proofs found; the version before it is kept, with its counterexample, wherever a
defect is documented:

| Functions | What is proved | What is not covered |
|---|---|---|
| `survives`, `arrhenius` | `exp(-k dt)` is the exact solution of `dm/dt = -k m` and the only one; the remaining mass lies in `[0, m]`; two steps compose; the paper's Euler update `m (1 - k dt)` is its first-order expansion, below it by at most `m (k dt)²/2`, and goes negative once `k dt > 1`, where the exponential never does. | Float32 storage of the fields. |
| `exchange` | Energy is conserved; neither temperature overshoots the other; the moved heat has the sign of `Ta - Tb`; it is the closed form of the linear pair `Ca dTa/dt = G (Tb - Ta)`, `Cb dTb/dt = G (Ta - Tb)`; as `dt → 0` it agrees with Eq. 19's explicit flux `G (Ta - Tb) dt` to first order, within `\|Ta - Tb\| G² dt²/(2μ)`, while that flux over a whole step can reverse the two temperatures. | The order in which `conduct` visits pairs. |
| `changeGas`, `stoichiometry`, `addProducts`, `chemistry`, `fireClasses.js` | Species masses change by their deltas and nitrogen is kept, under the guards `chemistry` applies; with water closing the balance exactly, burning `dm` adds `dm` to the gas from a solid and nothing from vapour (less the ash, for class D) for every fuel, and all six fire classes (A, B, C, K and L from their carbon fractions, and D) are accepted; the heat released is `HEAT_PER_O2 = 417e3/0.032` J per kg of O₂ for every class without its own `o2`, within 1% of Huggett's 13.1 MJ kg⁻¹; the liquid item conserves `C T + L · vapour`, never cools below boiling, and gives off no more than its pool's burning rate over the step (`chemistry_liquid_capped`), a cap that lies between 0 and the infinite pool's `m''∞ A dt` (`poolCap_bounds`; the cap since 2026-11-05); the ash's heat capacity keeps the solid's from falling while the bulk refills the surface layer; mixing fresh fuel into the layer conserves its enthalpy and leaves the temperature between `T_amb` and the old one, and `fresh` refills the layer exactly. | That every call site meets the guards is read from the code; the ash's mass is not a field. |
| `march`, `radiation`, `particleToGrid` | Gas, solid and outside energy add up to the energy emitted, with nothing dropped, since what a ray carries when its loop stops goes into the cell it ends in (`march_balance`, `marchLoop_vs_old`); each part is non-negative; the deterministic attenuation is the expected value of the paper's Monte Carlo absorption (`march_expected`); an emitter never cools past the midpoint with ambient; ember deposits conserve energy. | The ray geometry (Amanatides-Woo traversal, Lambert sampling). |
| `gasDensity`, `buoyancy`, `diffuse` | The equation of state is the ideal gas law and falls with temperature; buoyancy is upward exactly for gas lighter than ambient and is Eq. 6 read as intended; explicit diffusion stays within its neighbours' bounds, conserves heat below its cap, and, with a face coefficient from the mean of the two cells' `ρD`, conserves each species' mass in a pair exchange below its cap (`diffuse_species_pair`). | Advection, projection, vorticity confinement. |
| `visibility`, `coDoseRate` (`flamescapesRun.js`), `volumeFractions`, `interiorFuel` (`voxelize.js`) | Visibility `S = C/(σ_m ρ Y_soot)`, capped at 100 m, never rises with the soot density and falls strictly below the cap; the CO dose rate `ppm/(35000 × 60)` per second integrates to 1 exactly at 35 000 ppm·min (`coDoseRate_reaches_one`) and is ISO 13571's FED (`Tenability.fed`); renormalised fractions sum to at most one; the interior fuel's draws, cut at zero and rescaled by `M / total`, sum to exactly `M` (`interiorFuel_rescaled_sum`). | The conversion of CO from mass fraction to ppm; the reading of the solver at head height. |

What the proofs found. **Paper typos and inconsistencies confirmed by a proof.** Fire-X, against
its transcriber's notes: note 1 (Eq. 2's radiative term has units kg K m⁻² s⁻¹, not K s⁻¹, and
needs a factor of area per volume over ρ); 3 (all six Table 1 heats are Eq. 4 per mole of
fuel); 4 (for the three fuels with `q_f = -2`, Eq. 6 releases 208.5 kJ per mole of O₂ that Eq. 7
removes, half of Eq. 4's 417); 5 (Eq. 8 with its comma is not a density, and makes a heavier gas
lighter); 6 (Eq. 9's right-hand side is a density, and any liquid above ambient evaporates below
its boiling point); 7 (Eq. 10's rate is in K m³ s⁻¹, and equal and opposite rates conserve
energy only for equal volumetric heat capacities); 9 (Eqs. 27 and 28 agree only if a droplet
fills the cell); 11 (Eq. 15 is a specific energy, and as a scalar cannot conserve momentum); 12;
13 (Eq. 19 gives a mass; as printed, Eqs. 19 and 21 agree with each other); 14; 15 (the rescaled
fractions sum to `ρ' < 1`); 17 (Table 1's 63 and 114 fall below the ideal-gas bound `5R/(2M)`
if read in J kg⁻¹ K⁻¹); 18 (the units of `A` vary with `a + b`); and 19 (configurable product
coefficients break `Σ q_i M_i = 0`). Notes 8 and 10 are substantiated beyond their claim: Eq. 11
driven by `ρ_l` creates vapour at `T_l = T_amb`, where Eq. 9 removes no liquid, and Eq. 13's
cubic kernel is negative beyond `h_s`. Flamescapes: notes 3 (Eq. 17's sign makes the radiation
term heat hot sooty gas), 5 (with Table 3's signs, oxidation is a heat sink and pyrolysis a
source), 6 (`ṁ_p,s` is in kg mol⁻¹ s⁻¹), 7, 8 (Eq. 7 needs a pressure where `ρ_amb` stands; as
printed it gives air at 300 K a density of 1.36e-5 kg m⁻³), 9 (with `g = -9.81` the printed
term pushes hot gas down), 10 (`κ_s` needs m⁻¹), 12 (`h` needs W m⁻² K⁻¹), and supplement notes
3 (Algorithm 2's line 5 is an energy) and 4 (line 7 must divide by `ρ_s C_p,s` and the cell
volume). **Refuted:** none. The notational and prose notes (Fire-X 2, 16, 20 to 24; Flamescapes
1, 2, 4, 11, 13 to 18 and the remaining supplement notes) are not mathematical claims and are not
decided. **Found beyond the notes:** Fire-X's particle-grid round trip is not
mass-conservative although Sec. 5.1 calls it so (masses 1 and 0 come back as 2/3 and 2/3,
`X21_not_conservative`); Flamescapes' leaf-thickness correction can push a volume fraction above
one; its Eq. 13 makes `ρ_s C_p,s` 8% lower than the summed heat capacities for half wood and
half moisture; and under one convention for `g` the two papers' printed buoyancy terms point in
opposite directions. **Papers against the port.** The port corrects five of these: Eq. 7 with
`p_amb`, Eq. 6 with the intended sign, Algorithm 2's line 7 divided by the heat capacity,
Eq. 13's heat capacity summed over the constituents, and Eq. 1's fractions renormalised to at
most one. It replaces the paper's explicit updates by the exact exponential and the exact
relaxation (first-order agreement proved) and its Monte Carlo absorption by its expected value.
Its buoyancy `|g| (ρ₀/ρ - 1)` exceeds Fire-X's by the factor `ρ₀/ρ`, and `HEAT_PER_O2`, Fire-X's
constant, lies 0.525% below Huggett's. **The port's six defects, fixed in commit 30c019a.** Each was
found by a proof and is kept as a lemma about the previous version: `march` dropped what was
left after `rayReach` cells whatever its size (`marchOld_reach_drops_all`), and now leaves it in
the cell the ray ends in, so gas, solid and outside account for every joule (`march_balance`);
`diffuse` gave each cell its own coefficient, so species mass changed across a temperature jump
(`diffuse_species_pair_old`), and a face now takes the mean of its cells' `ρD`, which conserves
each species while the cap of 1/7 does not bind on either side (`diffuse_species_pair`; where
it binds, `diffuse_species_capped`); the liquid branch set the surface to its boiling point even
when the layer ran out first (`chemistry_liquid_evaporation_energy_old`), and now takes the
latent heat of what evaporated, conserving `C T + L dv` (`chemistry_liquid_energy`); a metal's
oxide vanished, and now adds `ash · dOx · 900` J K⁻¹ to the cell (`chemistry_ash_capacity`);
yields heavier than the fuel and its oxygen created mass through the clamp on water
(`addProducts_clamp_counterexample`), and are now refused, so the mass balance holds for every
fuel (`addProducts_mass_solid`, `fireClasses_accepted`); and cutting the contents' draws at zero
put more than `M` in the building in expectation (`interiorFuel_cut_raises_mean`), and the
masses are now rescaled to exactly `M` (`interiorFuel_rescaled_sum`). **Still needing a
hypothesis:** the ash keeps the solid's heat capacity from falling only while the bulk refills
the surface layer, or where `fuelHeatCapacity ≤ ash · 900`; magnesium is given wood's
2000 J kg⁻¹ K⁻¹ and its oxide holds 1492 J K⁻¹ per kg burnt, so once the remaining metal lies
within the layer the capacity falls, from 200 to 174.6 J K⁻¹ when 0.05 kg of the last 0.1 kg
burns (`chemistry_ash_capacity_counterexample`). Class D sets its own `o2` and so releases about
37.6 MJ per kg of O₂, not `HEAT_PER_O2`, by design (`classD_heat_per_O2`). The tolerance of 1e-9
in `stoichiometry` admits water down to `-1e-9` kg per kg; the six classes close without it.

DuctMixing.lean (3 results, 2026-11-05), the running ventilation's ducts (`ducts` in
flamescapes.js; not the paper's, which models no ducts): one exchange between a system's air
terminals and its well-mixed ducts conserves each species' mass and the heat over the system
(`ducts_conserve`); the ducts' mixture lies within the range of the values drawn (`mixture_mem`);
and no terminal cell leaves that range (`ducts_after_mem`), so mass fractions stay in `[0, 1]`
and no cell is heated past the hottest terminal. Not covered: floating point (the unit test
checks the conservation to 1e-5) and the duct network's geometry, which the model does not read.

Full build on 2026-11-05 after this batch, the port's fixes, the liquid pool's cap and the ducts: 27
modules, 1264 theorems and lemmas, 1251 of them with an axiom print (967 in the modules of the
first six batches, 284 in AxiomCheck.lean: the original 11, the 270 of this batch and the ducts' 3), every one on Lean's standard axioms or none
(`axioms-all-2026-11-05.txt`: the build log's prints followed by AxiomCheck.lean's output), no
`sorry`, no warnings.

## Eighth batch: the shielded learned escape (2026-11-05)

SafeEscape.lean (33 results) proves the claims of `rlSafe.js`, the learned escape that replaces the
tabular Q-learning of `rl.js` in the study [wiki/research/rl-egress-study-2026-11-05.md](../../wiki/research/rl-egress-study-2026-11-05.md).
The model is the code's: one open link per step, a move at step `t` into `w` safe when it arrives
at `t + 1 ≤ opn w` (`openUntil`), exits terminal, a step limit `H`; `Esc G t v k` is an escape of
exactly `k` moves from `v` at step `t`. Property tests in `rlSafe.test.mjs` check the same
statements against the code.

| Part | What is proved | What is not covered |
|---|---|---|
| Value iteration (`escapeValues`) | Backward induction on the time-expanded graph is the earliest safe escape: a lower bound on every escape, each finite value attained, `⊤` exactly when none is left (`bi_spec`, `vi_le`, `vi_esc`, `vi_top_iff`, `vi_eq_earliest`); leaving later never escapes sooner (`esc_mono`, `val_mono_time`). | The breadth-first `earliestSafeEscape` as a program (it is `GraphNumerics.L7_bfs_layer_eq_earliest` for one target). |
| Deadline map (`hopDeadlines`) | Any solution of the max-min recursion `D x = opn x`, `D v = max (-1) (min (opn v) (max_w D w - 1))` satisfies `t ≤ D v` iff `v` can be entered at `t` with an escape left (`deadline_iff`), so the recursion has one solution, the deadline. | The bucket label-setting that computes it; the tests check the code's output against the recursion. |
| Shield (moves with `t + 1 ≤ D w`) | An allowed move is safe on arrival and keeps an escape; a move is masked only if no escape goes through it; some move is allowed until an exit; every walk inside the shield from a start with an escape reaches an exit by `H`, entering every node before it closes, however its moves are chosen (`shield_iff`, `shield_safe`, `shield_keeps`, `shield_complete`, `shield_nonempty`, `shield_reaches_exit`). | A fire that is not known in advance: the shield is as safe as the forecast. |
| Learner (`learnSafeEscape`) | A consistent potential (the fire-free or the deadline-restricted hop distance) is admissible (`consistent_admissible`); the learning real-time search update and `StepTable`'s sharing of a bound with every later step keep the table admissible (`lrta_admissible`, `raise_admissible`); a safe walk to an exit no longer than an admissible start value is an earliest escape (`walk_esc`, `certificate_optimal`); a trial that changes no value passes that test (`trial_no_change`); tables that only rise below a bound change finitely often (`bounded_raises`); initialising with a potential is shaping the costs (`potential_init_eq_shaping`, after Wiewiora 2003). | The number of trials on a given building (measured in the study, not bounded usefully here); the tie-breaking by fire-free metres. |
| Counterexamples | No policy on the node alone (the state of Jabi et al. 2019) escapes earliest from one start and at all from another (`node_policy_fails`); masking only nodes untenable on arrival can trap the agent while an escape exists, and the deadline shield masks the trapping move (`arrival_mask_traps`, `G3_no_escape_from_p`); with `rl.js`'s reward (+10, -0.1 per median link, gamma 0.99) an escape of 122 moves returns less than staying inside on short links until the step limit (`legacy_reward_prefers_staying`), as the study measured on one corpus model. | The unfolding lemmas `bi_exit`, `bi_zero`, `bi_succ` are counted among the 33. |

What the proofs found. Formalising corrected one claim of the first draft of `rlSafe.js`:
pruning the value table at a node's deadline does not leave `NEVER` at every pruned state (a
node past its closing step can still have an escape in the table); it leaves every value that is
read unchanged, which is what the code needs and what its header now says. A shield that masks
only nodes untenable on arrival, one of the candidates the study started from, is not enough
(`arrival_mask_traps`), and the shipped reward's optimum need not escape
(`legacy_reward_prefers_staying`).

Full build on 2026-11-05 after this batch: 28 modules, 1297 theorems and lemmas, 1284 of them with
an axiom print (967 in the modules of the first six batches, 317 in AxiomCheck.lean: the original
11, the seventh batch's 270, the ducts' 3 and this batch's 33), every one on Lean's standard axioms
or none (`axioms-all-2026-11-05.txt`, regenerated from this build's log and AxiomCheck.lean's
output, with every earlier line kept in order), no `sorry`, no warnings.

## Ninth batch: sprinklers and the performance check (2026-11-07)

Sprinklers.lean (28 theorems) backs the sprinkler research (layout, the link's step and its exact
form, activation time, Evans). Performance.lean (14) backs `performanceCheck.js`: the fitted
log-normal, the share that starts in time, soundness on the conservative deadline map, and the
ratio criterion as a margin of `f - 1` times RSET. Two modules added the same day back its second
round, [wiki/research/rset-components-2026-11-07.md](../../wiki/research/rset-components-2026-11-07.md):

| Module | What is proved | What is not covered |
|---|---|---|
| DoorQueue.lean (16) | Lindley's recursion on time bins, as `doorQueues` runs it: the queue is never negative, a bin passes at most `c Δ` and never a negative number, departures and arrivals balance (`conservation`), more arrivals never shorten the queue (`queue_mono`); what arrived by bin `b` has left by bin `b + ⌈Q(b+1)/(cΔ)⌉` (`clears`, `exit_clears`) and that bound never falls as `b` grows (`exit_mono`). The wait `W` enters the check as `required(τ + W) + W`, which is exactly ASET - RSET ≥ m and ASET ≥ f RSET with RSET = base + τ + W (`requiredQ_iff`, `requiredQ_zero`, `requiredQ_mono`); queues never raise the share that starts in time, and the safe starts are a down-set (`queued_share_le`, `safe_starts_downward`). | Floating point; the walk tree and the door order (the unit tests check them); the dose taken while queueing, which the check does not reckon. |
| PreMovement.lean (10) | The Weibull's quantile inverts its distribution function, which is monotone and in `[0, 1]` (`weibull_cdf_quantile`, `weibull_mono`, `weibull_nonneg`, `weibull_le_one`); with a share `q` that never responds, the share that starts in time is at most `1 - q` and at most the responders', is monotone, and a pass at the responders' percentile `s` gives at least `(1 - q) s` (`responding_le`, `responding_mono`, `responding_pass`); more detectors never detect later (`detection_more_detectors`, `detection_le`); the report's worst margin is at least 0 exactly when every case's is (`report_pass_iff`). | The fits themselves (Lovreglio et al. 2019's tables); where the detectors sit, which is geometry the unit tests check. |

What the proofs found. `weibull_nonneg` needs a scale of at least 0, which the clusters have; the
statement without it is false (a negative scale and shape 1 give a negative value).

Full build on 2026-11-07 after this batch: 32 modules, 1365 theorems and lemmas (counted as above),
no `sorry`, no warnings; the 26 new theorems' axiom prints are in `AxiomCheck.lean`, which now prints
343 results, every one on Lean's standard axioms (`axioms-all-2026-11-07.txt`).

## Tenth batch: suppression (2026-11-07)

Suppression.lean (58 results) backs the sprinkler build in `src/core/suppression.js` and the
`spray` and Evans steps of `flamescapes.js` and its WGSL port, items 8 to 15 of
[wiki/research/sprinkler-physics-2026-11-07.md](../../wiki/research/sprinkler-physics-2026-11-07.md)
section 9; it reuses Sprinklers.lean's decay and FlamescapesPort.lean's `changeGas`.

| Part | What is proved | What is not covered |
|---|---|---|
| Spray (item 8) | Down a column the water emitted is the water evaporated plus what is left, never negative and never growing downward; per head and over all heads, with heads sharing a column merged (`spray_column`, `spray_head`, `spray_budget`, `spray_budget_merged`). | How much evaporates in a cell (a parameter, not a theorem). |
| Evaporation (item 9) | The step conserves energy with the vapour carrying its heat, `C T = (C + c_v dv) T' + dv (e - c_v T_B)`; `0 ≤ dv ≤ m_w`; `T_B ≤ T' ≤ T` (`evaporate_energy`, `evaporate_bounds`, `evaporate_between`, `evaporate_step`); at the energy limit the cell sits at boiling (`evaporate_limit_pinned`); Fire-X's Eq. 30 is the case `e = 0` (`evaporate_vs_X30`), not `T' = T_B` as the report said. | Floating point. |
| Steam (items 10, 11) | Adding vapour through `changeGas` keeps every other species' mass, keeps the fractions valid, and never raises the oxygen's mole fraction (`vapour_changeGas`, `vapour_dilutes_O2`); at fixed temperature it scales the gas burn by the dilution factor, since the port's exponents sum to zero (`dilution_scales_burn`, `dilution_heat`). | Temperature changes, which item 12 treats. |
| Cooling (item 12) | Cooling lowers the first-order burn rate up to 15,154 K for the port's kinetics (`cooling_first_order_port`), though the cooled cell holds more gas and its oxygen cap rises (`cooling_raises_cap`, counterexample 1000 to 600 K: 5/3); in fast chemistry, or whenever the oxygen cap binds, cooling burns more (`cooling_fast_chemistry`, `cooling_oxygen_limited`). | The full scheme's sign, which these show depends on the regime. |
| Evans (item 14) | The cap never rises (`evansCap_antitone`); its sum is at most `r (tau + dt)` (`evansCap_sum`); cells under one envelope (`evans_cells`). | Evans's own data. |
| Wet fuel and extinction (items 13, 15) | Wet fuel sits at or below boiling and its oxidation is bounded there (`wet_skin_pinned`, `wet_skin_heat`); a coarse-grid extinction limit, if adopted, is never undone by water (`extinct_monotone`); steam at a hot dry surface raises pyrolysis up to four times (`pyrolysis_dilution_counterexample`, `pyrolysis_rises_with_steam`). | The app does not wet the fuel yet. |

Full build on 2026-11-07 after this batch: 33 modules, 1423 theorems and lemmas, no `sorry`, no
warnings; the 58 axiom prints are in `AxiomCheck.lean` (401 prints in all), every one on Lean's
standard axioms or none (`axioms-all-2026-11-07.txt`).

## Building

Lake caches are gigabytes, so build outside OneDrive. Keep the working copy
outside any synced folder (ours lives under the user's home, in `lean/fireegress`); to rebuild elsewhere:

```
# copy this folder somewhere outside OneDrive, then:
lake exe cache get      # prebuilt Mathlib
lake build              # ~2 minutes after the cache
lake env lean AxiomCheck.lean
```

Still outside the proofs: heaps, route reconstruction and termination of the search loops, and floating point beyond FloatError.lean. The earliest-escape search (L7), A* (L9), time-dependent Dijkstra (L10c) and the zone-model units (L16) that this paragraph once listed are now proved: `L7_bfs_layer_eq_earliest`, `astar_correct`, `td_dijkstra_correct`, `FD24_zone_units` and the `units_T*` theorems.