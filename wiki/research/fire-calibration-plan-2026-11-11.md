# Fire calibration: the transport stages and how they inform the app (2026-11-11)

Status: Stage 1a run (correlations, no external data); the rest planned. The user (2026-11-11): "lets take steps to get validation and the calibration to actually inform this". This note continues [flamescapes-calibration-plan-2026-11-08.md](flamescapes-calibration-plan-2026-11-08.md) (adopted; its principles P1 to P8, its acceptance tiers in section 8 and its decisions in section 11 stand). Nothing here changes the solver: every finding that would change a parameter or an algorithm is a proposal for the user (section 6), because any such change changes every run.

The rule that holds throughout: the Flamescapes solver is **not calibrated** until a stage passes against held-out data in at least three independent datasets (plan 2026-11-08, section 8). Single correlations, as in Stage 1a, can at most show that a quantity lies **within the calibrated band** or is **indicative** on that one comparison; they cannot make it calibrated.

## 1. What the app claims, and what each claim needs

| What the app shows or decides | Quantity | Stage that informs it |
| --- | --- | --- |
| Heat colouring, "untenable above", the heat dose | gas temperature at head height | 1a (plume, layer), 1b (layer data), 3 |
| Smoke view, visibility at 1.8 m, the visibility criterion | soot density, visibility | 1c |
| Sprinkler and detector activation | ceiling-jet temperature and speed | 1a (Alpert), sprinkler validation (done, 2026-11-08) |
| Smoke spreading between rooms, the egress forecast | door flows, layer depth | 1b (Steckler), 1c (filling) |
| Heat release on the run card | the solver's own burning | Stages 2 and 3 (2026-11-08 plan) |

## 2. Stages, acceptance and sources

Statistics as in the FDS Validation Guide (NIST SP 1018-3, release 6.11.1, section 4.4): the bias factor δ and the model's relative standard deviation σ̃M from the log ratios of model to measurement (or to a correlation), with the experimental σ̃E of its Table 4.1 (temperatures 0.07, velocity 0.08). The tiers of the 2026-11-08 plan (section 8): within the calibrated band at 0.8 ≤ δ ≤ 1.25 and σ̃M ≤ 0.35; indicative at 0.67 ≤ δ ≤ 1.5 and σ̃M ≤ 0.5; otherwise not calibrated. FDS's own results (Table 16.1 of the same guide, as read for the 2026-11-08 plan) are the bar beside them: layer temperature σ̃M 0.10, δ 1.07; ceiling jet 0.13, 1.08; plume 0.42, 1.00.

Resolution is reported for every run as the plume resolution index D*/δx, D* = (Q / (ρ c_p T √g))^(2/5) (FDS User's Guide, mesh resolution). The app's fires sit at 1.3 to 3.6 on its cells, at or below the lowest of FDS's validation studies (1 to 118) and NUREG-1934's 5 to 10 for gross smoke movement (2026-11-08 plan, section 4).

**Stage 1a: transport against classical correlations** (no data needed; this note, section 3). A prescribed heat release, ramped and held (`createBurner`), of the app's design-fire kind (the class A item's sooty fuel, no radiant fraction: plan principle P2, the model as used), on the app's cells 0.20, 0.27 and 0.34 m, CPU, 120 s, read as time averages from 60 s:

- the plume: centreline temperature rise and upward speed against McCaffrey (1979), his Table 1 and his velocity fits, in the intermittent and plume regions (the flame region is the burner's prescribed column);
- a room's upper layer: the solver tests' 5 x 4 x 2.7 m stone room with a 0.9 x 2.0 m door, three thermocouple trees reduced by Janssens and Tran (1992), against McCaffrey, Quintiere and Harkleroad (1981), with the walls' conductance h_k = √(kρc/t) for walls thicker than the heat has reached (the solver's stone, k 2.5, ρ 2700, c 800);
- the ceiling jet: a burner under an unconfined ceiling 3 m above it, the top open cell's rise along a radius against Alpert (1972), beyond r/H = 0.18.

**Stage 1b: transport against measured tests** (data in NIST's `firemodels/exp`, decision 4 of the 2026-11-08 plan): Steckler, Quintiere and Rinkinen (1982) doorway velocity and temperature profiles; NBS multi-room (Peacock, Davis and Babrauskas 1991); NIST/NRC layer, ceiling jet and gases; UL FSRI 2018 burner tests. Acceptance: the tiers, three datasets for "calibrated". Harness: `tools/calibration/transport-checks.mjs` extended with the tests' geometry, as `tools/sprinkler-validation.mjs` reads its tests.

**Stage 1c: smoke filling and visibility.** Filling: Zukoski's (1978) closed-room layer descent and NFPA 92's empirical filling equation for large spaces (its text to be read before it is cited by number), on an atrium-like box (the 2026-11-11 atrium fix makes this the app's case too); visibility: soot density read through Mulholland and Croarkin's (2000) specific extinction coefficient (already the app's), against a measured optical density in a NIST/NRC or UL FSRI test.

**Stage 0 again, with an energy budget** (2026-11-08 plan, section 4.1, "what follows"): heat stored in the gas, into walls, out of the openings and radiated, per run and per cell, so that a layer that is too hot can be traced to its cause (section 4).

**Stages 2 and 3** (one item burning; spread between items and room fires): as the 2026-11-08 plan.

## 3. Stage 1a results (2026-11-11)

`node tools/calibration/transport-checks.mjs --cases plume,room,ceiling --cells 0.20,0.27,0.34 --horizon 120 --json benchmark/results/2026-11-11-fire-calibration.json` (CPU; 2.5k to 32k cells; 3 to 42 s a run).

With the app's parameters (vorticity confinement 10, walls' convection 10 W/m²K, steps of 0.1 s):

| Check | 0.20 m | 0.27 m | 0.34 m | Verdict |
| --- | --- | --- | --- | --- |
| Plume, McCaffrey, 500 kW (D*/δx 3.6, 2.7, 2.1) | the centreline flows down in 61 % of samples; δ 0.01 | 43 %; no valid plume points | 26 %; δ 0.24, σ̃M 0.07 | fails: the plume does not stay on its axis |
| Room layer rise, MQH, 150 kW at 60/90/120 s | 2.29 to 2.39 x MQH (δ 2.34) | 2.86 to 3.11 x (δ 3.01) | 3.23 to 3.66 x (δ 3.49) | fails: too hot |
| Room layer rise, MQH, 400 kW | 2.35 to 2.63 x (δ 2.46) | 2.54 to 2.96 x (δ 2.74) | 2.46 to 2.91 x (δ 2.67) | fails: too hot |
| Ceiling jet, Alpert, 500 kW under 3 m, r/H 0.33 to 1.17 | diverged at 32 s | δ 0.85, σ̃M 0.26 | δ 1.07, σ̃M 0.50 | within the calibrated band at 0.27 m, indicative at 0.34 m, on this one correlation |

The layer's rise at 400 kW and 120 s, +237, +257 and +248 K on the three cells, agrees within 8 % between cells (FDS's layer scatter is 10 %): under a prescribed heat release in a closed room the layer is grid-converged to that tolerance at 400 kW, though not at 150 kW (+120, +150, +169 K). The interface sits 0.6 to 0.8 m over the floor in every run.

Diagnostic only (not the app's values), vorticity confinement 0:

| Check | 0.20 m | 0.27 m | 0.34 m |
| --- | --- | --- | --- |
| Plume holds its axis | always | always | always |
| Intermittent region, temperature / speed | δ 0.95 / 0.83 | δ 1.25 / 1.14 | δ 1.00 / 1.00 (all within the calibrated band) |
| Plume region, temperature / speed | δ 2.79 / 1.54 | δ 3.20 / 1.95 | δ 2.73 / 1.75 (too hot, too fast: too little entrainment) |
| Ceiling jet | δ 3.72 (no divergence) | δ 4.39 | δ 3.51 |

## 4. Findings

F1. **An impulsive start diverges.** A burner stepped from nothing to 500 kW in one 0.1 s step diverged within 6 to 8 s (speeds past 10^4 m/s at 0.34 m), stable with steps of 0.05 s or at 150 kW. The app's design fires grow as t² and never step, so its runs are not affected; the harness ramps over 30 s.

F2. **A burner whose cells cannot radiate diverges near 400 to 500 kW** even ramped (soot-free methane with its radiant part on rays, the sprinkler validation's kind): speeds to NaN once the ramp reaches its plateau, a cell near 6,400 K, at 0.20 and 0.27 m. Heat put into fixed cells with no radiative loss heats gas whose density falls as it heats, so each joule raises it further. With the app's own kind (sooty fuel, the hot cells radiating) the same runs were stable, except under a ceiling on 0.20 m cells, which diverged at 32 s with vorticity confinement 10 and not with 0. The sprinkler validation's runs (methane, 11 to 43 MJ over 90 to 205 s) were not watched for this; their activation readings come early (at 50 to 230 kW) and are probably unaffected, but they should be rerun with the health monitor.

F3. **Vorticity confinement is the solver's entrainment, and at 10 it loses the plume.** In an open box the plume wanders off its axis (the centreline flows down in a quarter to most of the samples) and its time-averaged centreline is far below McCaffrey's; with confinement off it holds its axis, matches McCaffrey's intermittent region within the calibrated band on all three cells, and runs 2.7 to 3.2 times hot higher up, with ceiling jets 3.5 to 4.4 times hot. So the app's value happens to give Alpert-like ceiling jets at 0.27 and 0.34 m (which the sprinkler validation also found) while it destabilises the free plume; no single value tested fits both.

F4. **The closed room's hot layer runs 2.3 to 3.7 times MQH's rise** on every cell and at both fires, a bias far outside FDS's 10 % on layer temperature. Candidates, to be told apart by the energy budget (Stage 0 again): the walls take too little heat (the solver's 10 W/m²K surface coefficient against MQH's wall conductance √(kρc/t), about 0.2 kW/m²K at a minute into stone), the radiation from a sooty layer to the walls, and the door's outflow on cells of 0.2 to 0.34 m (the 0.9 m door is 3 to 5 cells wide).

F5. **A harness lesson:** on coarse cells the 2.7 m ceiling snaps down to an open height of 2.50 m (0.27 m) and 2.52 m (0.34 m); thermocouple trees must read open cells only, or the wall's temperature enters the layer reduction. The harness now does; the 2026-11-08 grid study's trees (to 0.1 m under the nominal ceiling) should be checked for the same.

## 5. How the results inform the app

- Nothing in the app changes from Stage 1a: no quantity is calibrated, and the run card and Methods keep "not calibrated". When a quantity reaches a tier on three datasets (Stage 1b), its number is shown with its band and the Methods page gives δ, σ̃M and the range of cells and fires it holds for (2026-11-08 plan, section 9).
- What Stage 1a can already say, as caveats in the Methods page (proposed text, for the user): the solver's hot layer in a closed room runs hot against MQH on two of three cells; the ceiling jet reads within FDS-like bands on 0.27 and 0.34 m cells; the open plume is not resolved at the app's cells.
- A run's health (largest speed, non-finite values) is cheap to check each frame: proposed as a guard in the app's runs, reporting a diverged run instead of drawing it (section 6, proposal A).

## 6. Proposals for the user (none applied)

A. **A run-health guard in the app** (no physics change): each frame, the largest gas speed and a finite check; a run past 50 m/s or with a non-finite value stops and says so instead of drawing it. Cost: one pass over the faces a frame. Effect: none on healthy runs; a diverged run can no longer be shown as a fire.

B. **Stability:** cap the step by a CFL condition (substeps where the largest speed times the step exceeds about one cell), and scale the confinement force with the cell (ε h, as Fedkiw, Stam and Jensen 2001 define it) so that it does not grow relative to the flow on finer cells. Expected effect: no divergence at 0.20 m; runs that never reach the cap unchanged in kind but not bit-identical once ε is scaled (RUN_CACHE_VERSION bump).

C. **Calibrate the confinement as the entrainment parameter** (Stage 1a.2): sweep it (for example 2, 4 and 6, scaled by the cell as in B) on McCaffrey's plume and Alpert's ceiling jet together, on the three cells, and take the value that holds the plume's axis and fits both best; then judge it on Stage 1b's measured tests, which it was not fitted to. This is the plan's principle P3 (few parameters, fitted on training data, judged on held-out data).

D. **The hot layer:** run the energy budget first; only if the walls are short of heat, propose a wall surface model (the wall cell's surface temperature with one-dimensional conduction into it, as FDS does) in place of a single coefficient, since raising the coefficient would undo the sprinkler validation's fit of the ceiling jet. Judge it on MQH and then Steckler's and the NIST/NRC layers.

E. **Rerun the sprinkler validation with the health monitor** (F2), to confirm its activation times came from healthy runs.

Each of B, C and D changes every fire the app runs; none is applied until the user decides.

## 7. Sources

DOIs checked on Crossref on 2026-11-11 (Heskestad, Alpert, MQH, Zukoski, Mulholland and Croarkin, Fedkiw et al.) or on 2026-11-08 (the rest, in the earlier plan).

- Alpert, R. L. (1972). Calculation of response time of ceiling-mounted fire detectors. Fire Technology 8(3), 181-195. doi:10.1007/BF02590543
- Fedkiw, R., Stam, J. and Jensen, H. W. (2001). Visual simulation of smoke. Proceedings of SIGGRAPH 2001, 15-22. doi:10.1145/383259.383260 (vorticity confinement, its force scaled by the cell)
- FDS User's Guide (NIST SP 1019) and Validation Guide (NIST SP 1018-3), release 6.11.1 (cited by release, as in the 2026-11-08 plan).
- Heskestad, G. (1983). Luminous heights of turbulent diffusion flames. Fire Safety Journal 5(2), 103-108. doi:10.1016/0379-7112(83)90002-4
- Janssens, M. and Tran, H. C. (1992). Data reduction of room tests for zone model validation. Journal of Fire Sciences 10(6), 528-555. doi:10.1177/073490419201000604
- McCaffrey, B. J. (1979). Purely buoyant diffusion flames: some experimental results. NBSIR 79-1910. doi:10.6028/NBS.IR.79-1910
- McCaffrey, B. J., Quintiere, J. G. and Harkleroad, M. F. (1981). Estimating room temperatures and the likelihood of flashover using fire test data correlations. Fire Technology 17(2), 98-119. doi:10.1007/BF02479583
- Mulholland, G. W. and Croarkin, C. (2000). Specific extinction coefficient of flame generated smoke. Fire and Materials 24(5), 227-230. doi:10.1002/1099-1018(200009/10)24:5<227::AID-FAM742>3.0.CO;2-9
- NFPA 92, Standard for Smoke Control Systems (cited by designation; its filling equation to be read in the current edition).
- NUREG-1934 (2012), section 2.3.5 (as in the 2026-11-08 plan).
- Peacock, R. D., Davis, S. and Babrauskas, V. (1991). Data for room fire model comparisons. Journal of Research of NIST 96(4), 411. doi:10.6028/jres.096.022
- Steckler, K. D., Quintiere, J. G. and Rinkinen, W. J. (1982). Flow induced by fire in a compartment. NBSIR 82-2520. doi:10.6028/NBS.IR.82-2520
- Zukoski, E. E. (1978). Development of a stratified ceiling layer in the early stages of a closed-room fire. Fire and Materials 2(2), 54-62. doi:10.1002/fam.810020203
