# Fire calibration: the transport stages and how they inform the app (2026-11-11)

Status: Stage 1a run before and after; the user decided (2026-11-11) to apply proposals A to E of section 6, and sections 8 to 10 give what they changed and showed: the run-health guard and the cut steps are in the app, the confinement strength is fitted to 6, the energy budget found the walls not short of heat but the transport making heat (F6, proposal F), and a first slice of Stage 1b (three Steckler tests) does not pass. The user (2026-11-11): "lets take steps to get validation and the calibration to actually inform this". This note continues [flamescapes-calibration-plan-2026-11-08.md](flamescapes-calibration-plan-2026-11-08.md) (adopted; its principles P1 to P8, its acceptance tiers in section 8 and its decisions in section 11 stand). Every finding that would change a parameter or an algorithm is a proposal for the user, because any such change changes every run; F is open.

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

## 6. Proposals for the user (A to E applied 2026-11-11; section 8)

A. **A run-health guard in the app** (no physics change): each frame, the largest gas speed and a finite check; a run past 50 m/s or with a non-finite value stops and says so instead of drawing it. Cost: one pass over the faces a frame. Effect: none on healthy runs; a diverged run can no longer be shown as a fire.

B. **Stability:** cap the step by a CFL condition (substeps where the largest speed times the step exceeds about one cell), and scale the confinement force with the cell (ε h, as Fedkiw, Stam and Jensen 2001 define it) so that it does not grow relative to the flow on finer cells. Expected effect: no divergence at 0.20 m; runs that never reach the cap unchanged in kind but not bit-identical once ε is scaled (RUN_CACHE_VERSION bump).

C. **Calibrate the confinement as the entrainment parameter** (Stage 1a.2): sweep it (for example 2, 4 and 6, scaled by the cell as in B) on McCaffrey's plume and Alpert's ceiling jet together, on the three cells, and take the value that holds the plume's axis and fits both best; then judge it on Stage 1b's measured tests, which it was not fitted to. This is the plan's principle P3 (few parameters, fitted on training data, judged on held-out data).

D. **The hot layer:** run the energy budget first; only if the walls are short of heat, propose a wall surface model (the wall cell's surface temperature with one-dimensional conduction into it, as FDS does) in place of a single coefficient, since raising the coefficient would undo the sprinkler validation's fit of the ceiling jet. Judge it on MQH and then Steckler's and the NIST/NRC layers.

E. **Rerun the sprinkler validation with the health monitor** (F2), to confirm its activation times came from healthy runs.

Each of B, C and D changes every fire the app runs; none is applied until the user decides.

## 7. Sources (sections 1 to 6)

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

## 8. Proposals A to E, as applied (2026-11-11)

The user decided to apply A to E. What each became:

- **A, the run-health guard.** Each frame, the CPU and the GPU read four things: the largest gas temperature and speed, the gas's heat above ambient (ρ V c_p (T − T_a), c_p 1005), and whether any value is not finite. The GPU reads them in the frame's existing reduction and finds a non-finite value by its exponent bits.
  - A run stops at the first frame with a non-finite value, gas above 3,000 K, gas faster than 50 m/s, or gas holding more than 1.5 times the heat put in, plus 1 MJ. The heat put in is the fire's release plus what a pilot added (`flamescapes.js` `HEALTH`, `healthReadings`, `healthVerdict`).
  - The result keeps the frames before the failure and says why, when and where (`result.health`). A healthy run reports its largest speed and the most steps it took in a frame.
  - The limits were set before any run and have not been changed to make runs pass.
- **B, the cut steps.** Each frame's 0.1 s is cut into equal steps, at most 8 (`substepsFor`), decided the same way on the CPU and the GPU:
  - the fastest gas of the frame before moves at most one cell a step (CFL 1);
  - the confinement's explicit force stays stable (ε · step ≤ 0.5).

  Both paths already scaled the confinement force by the cell (ε h, Fedkiw, Stam and Jensen 2001). The instability was ε · step = 1 at the app's ε 10, so B added the step limit instead.
- **C, the confinement strength: 6** (was 10). The fit followed a rule fixed before the runs (`tools/calibration/fit-confinement.mjs`):
  - candidates 0, 1, 2, 3, 4, 6 and 10 on 0.27 and 0.34 m cells, and the best two also on 0.20 m;
  - a candidate is out if it diverges, or if the open plume's centreline flows down in more than 5 % of samples;
  - the score is the mean of (ln δ)² + (σ̃M − σ̃E)² over four checks: McCaffrey's intermittent temperature, plume temperature and plume velocity, and Alpert's ceiling-jet temperature.

  | Strength | 0 | 1 | 2 | 3 | 4 | 6 | 10 |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | Score on two cells | 0.94 | 1.02 | 0.96 | 0.78 | 0.50 | 0.26 | 0.69, out |
  | Score on three cells | | | | | 0.55 | 0.30 | |

  Strength 10 is out because the plume left its axis in 12 % of samples at 0.27 m. The fitted 6 was then judged on Steckler's tests (section 9.3), which it was not fitted to.
- **D, the energy budget first.** The walls are not short of heat on any cell (section 9.2), so no wall surface model was built. The budget found a different problem instead: F6.
- **E, the sprinkler validation rerun with the guard**, on the old solver and on the new (section 9.4).

`RUN_CACHE_VERSION` went to 10, so no run kept from the old solver is replayed. The tests are in `src/core/flamescapesHealth.test.mjs`: the constants, the cut steps, the verdict, a failing run and a healthy one, and a pilot's heat counted as heat put in.

## 9. Results after A to E

### 9.1 Stage 1a, before and after

`node tools/calibration/transport-checks.mjs --cells 0.20,0.27,0.34`. Before: ε 10, one step a frame. After: ε 6, cut steps. Each cell is δ, with σ̃M in brackets where it was recorded:

| Check | 0.20 m before / after | 0.27 m before / after | 0.34 m before / after |
| --- | --- | --- | --- |
| Plume off its axis (share of samples) | 61 % / 0 % | 43 % / 3 % | 26 % / 2 % |
| Intermittent region, temperature | 0.11 / 0.86 (0.26) | 0.03 / 1.01 (0.13) | 0.26 / 0.71 (0.13) |
| Plume region, temperature | 0.01 / 1.43 (0.09) | none / 1.15 (0.07) | 0.24 / 0.87 (0.07) |
| Plume region, velocity | 0.01 / 0.86 (0.08) | 0.01 / 0.72 (0.08) | 0.22 / 0.69 (0.08) |
| Room layer over MQH, 150 kW | 2.34 / 2.49 | 3.01 / 3.52 | 3.49 / 3.90 |
| Room layer over MQH, 400 kW | 2.46 / 2.48 | 2.74 / 2.91 | 2.67 / 2.88 |
| Ceiling jet, Alpert | diverged / 3.11 (0.07) | 0.85 (0.26) / 2.76 (0.17) | 1.07 (0.50) / 1.89 (0.45) |

What improved:
- The plume now holds its axis.
- Its temperature is within the calibrated band at 0.27 and 0.34 m, on this one correlation.
- The 0.20 m ceiling run no longer diverges.

What it cost:
- The ceiling jet, within the band at 0.27 m before, now runs 1.9 to 3.1 times Alpert's rise.
- The room's layer is no closer to MQH: 2.3 to 3.9 times hot before and after (2.4 to 3.9 with ε 10 and the cut steps).

By its pre-registered score, the fit traded the ceiling jet for the plume. Section 9.2 suggests why no confinement value can fit both.

### 9.2 Where the heat goes (D) and F6

`tools/calibration/energy-budget.mjs`, the Stage 1a room from 60 to 120 s. Each share is of the fire's heat over that window:

| Cell, fire | Walls | Door | Gas | Rest | MQH-implied walls |
| --- | --- | --- | --- | --- | --- |
| 0.20 m, 150 kW | 0.51 | 0.47 | −0.03 | 0.05 | 0.77 |
| 0.20 m, 400 kW | 0.60 | 0.37 | 0.00 | 0.04 | 0.82 |
| 0.27 m, 150 kW | 0.80 | 0.44 | 0.03 | −0.27 | 0.84 |
| 0.27 m, 400 kW | 0.91 | 0.28 | 0.00 | −0.18 | 0.88 |
| 0.34 m, 150 kW | 1.17 | 0.52 | 0.00 | −0.69 | 0.83 |
| 0.34 m, 400 kW | 1.05 | 0.26 | 0.00 | −0.31 | 0.88 |

The rule, fixed before the run, calls the walls short when they take less than half the share MQH implies. They are not short on any cell. But on 0.27 and 0.34 m cells the walls and the door together take 1.2 to 1.7 times the fire's heat: heat is made inside the room.

F6. **The transport does not conserve heat.**

Two scratch checks show it:
- **A sealed box with a burner.** A sealed stone box, with the wall exchange and radiation switched off, and a 150 kW burner. After 60 s, the gas holds 1.8 to 4.9 times the heat the burner released (1.9 to 6.9 times with confinement 0), and its mass has fallen to 36 to 76 % of the start.
- **The same box with no burner.** The lowest metre starts at ambient + 300 K. As it mixes into the box, its heat grows by 21 to 31 % in 60 s.

The likely mechanism fits both checks:
- The solver carries temperature with a flow that preserves volume (divergence-free).
- It then sets density from temperature by the gas law at constant pressure.
- Interpolating temperature takes an arithmetic mean. Gases mixing at constant pressure take a harmonic mean, by volume.
- So every mixing step loses mass and gains heat ρ c_p (T − T_a).
- Heat put into a cell that is already hot and thin is counted again when that gas mixes.

This is the likely common cause of three biases: the hot layer (F4), the hot ceiling jets and the hot far plume. Confinement has been trading these against each other. The numbers are in `benchmark/results/2026-11-11-fire-calibration-after.json` (`closure`).

### 9.3 Stage 1b, a first slice: Steckler's doorway tests

`node tools/calibration/steckler-checks.mjs` runs three tests of Steckler, Quintiere and Rinkinen (1982):
- the tests: 14, 410 and 610, each a 62.9 kW methane burner in a room with a 0.74 × 1.83 m door;
- the burner positions: the centre (14), a back corner (410), the back wall's middle (610);
- the data: measurements from NIST's `firemodels/exp`, test conditions from NIST's FDS validation inputs;
- the comparison: the model averaged from 200 to 300 s, against each test's steady state.

The model differs from the tests in two ways. Its walls are the solver's stone, a cell thick, where the tests' were fibre board. And it runs 300 s, not 30 min.

Each cell is δ (σ̃M) over the three tests:

| Quantity | 0.20 m | 0.27 m | 0.34 m |
| --- | --- | --- | --- |
| Upper layer rise (Janssens and Tran, room tree) | 0.93 (0.07) within the band | 0.97 (0.94) not calibrated | 1.61 (0.48) not calibrated |
| Interface height | 0.40 not calibrated | 1.48 indicative | 0.53 not calibrated |
| Largest outflow speed, door centreline | 0.55 not calibrated | 0.68 indicative | 0.59 not calibrated |
| Neutral plane height | 0.89 within the band | 0.85 within the band | 0.88 within the band |

Stage 1b does not pass:
- Only the neutral plane is within the band on every cell.
- The layer's rise is within the band on 0.20 m cells only.
- The doorway's peak outflow reads a third to a half low on every cell. The 0.74 m door is two to four cells wide, and the peak sits within a cell of the soffit.
- The interface is misplaced on every cell.

A single dataset could at most make a quantity indicative.

The other two solver settings did no better:
- **The old solver** (ε 10, one step a frame): test 14 diverges on every cell, and on 0.27 m cells the layer fails to form.
- **ε 10 with the cut steps:** it runs, and the layer's rise is within the band on 0.20 and 0.27 m (0.89, 0.88), but the outflow and the interface are no better.

### 9.4 The sprinkler validation with the guard (E)

`node tools/sprinkler-validation.mjs --convection 10` (46 runs); the old solver adds `--solver vorticity=10,cfl=0,vorticityStep=0`. Each cell is the first head's estimated activation time over the measured time, as a median per data set:

| Data set | 2026-11-08 (no guard) | Old solver, guard | New solver, guard |
| --- | --- | --- | --- |
| Vettori flat, tabulated ramp | 1.26 | 1.26 | 1.04 |
| Vettori flat, nominal ramp | 1.78 | 1.77 | 1.56 |
| Vettori sloped, tabulated | 0.99 | 0.91 | 0.90 |
| Vettori sloped, nominal | 1.31 | 1.31 | 1.15 |
| Bittern 2004 | 0.99 | 1.00 | 0.87 |
| UL/NFPRF series I | 1.29 | 1.29 | 1.08 |
| Runs stopped by the guard | (not watched) | 31 of 46 | 29 of 46 |

F7. **The validation's own burner makes gas above 3,000 K, on the old solver as on the new.**
- **Why they stop.** The stopped runs stop for gas above 3,000 K in the prescribed methane burner's own column; two old-solver runs stop for speed instead.
- **When.** From 70 to 90 s in Vettori's flat-ceiling tests, and later in the others.
- **Why that is unphysical.** Methane's adiabatic flame temperature is about 2,230 K.
- **The cause.** It is F2's mechanism, now seen at the validation's heat releases: the burner cells are soot-free and cannot radiate, and the heat goes into gas that thins as it heats.
- **What survives.** Every first head opened before its run stopped, so the first-head medians stand.
- **What is lost.** Later heads past the stop. On the old solver 14 of 18, 18 of 21 and 14 of 16 heads opened, where 16, 21 and 16 opened unguarded.
- **The new solver's direction.** It opens heads 10 to 20 % earlier, as its hotter ceiling jet (9.1) predicts. An earlier activation is the unconservative direction.

The app's design fires use the sooty, radiating kind and were not stopped (section 9.5).

### 9.5 Cost and the app's runs

- **CPU.** Cutting the steps costs about twice the solver time:
  - the test room at 0.20 m went from 3.7 to 1.6 times real time;
  - Vienna (27,744 cells of 0.40 m) went from 3.1 to 1.4.
- **GPU** (Intel UHD 770, headless Chrome). The step itself costs no more (2.45 ms a step at 27k cells, 12.5 ms at 187k), but each 0.1 s now takes two to five steps.
  - On the same grids: the room went from 37 to 16 times real time, and Vienna from 30 to 16.
  - On the grid the app's worker picks, Vienna ran at 1.8 times real time over 120 s, on 195,858 cells of 0.37 m. Before, it ran at 2.6 times on 279,585 cells of 0.34 m.
  - The worker's measured rate fell about 8 %, which dropped it one grid level: its choice of level is sensitive to timing noise.
  - Over 600 s on 279,585 cells of 0.34 m, Vienna ran at 1.2 times real time and the NIBS Clinic at 1.0 times. Both runs were complete and not stopped.
  - Vienna still runs on the web, faster than real time.
- **Parity.** The GPU still matches the CPU:
  - over 200 steps, its gaps are the size of a one-ulp CPU twin's;
  - one step from the same state releases the same heat;
  - the full model agrees on the node readings, the detectors' first alarm and the untenable nodes.
- **Long runs.** The app's class A design fire ran 600 s on the CPU at 0.20 and 0.35 m cells. It grew throughout and was not stopped (up to 5 steps a frame at 0.20 m).

### 9.6 The site's examples, old solver against new

The question was "does the fire change a lot?". The setup:
- **The two solvers.** The old is f7d631d: ε 10, one step a frame, no guard. The new is ε 6, cut steps, with the guard.
- **The runs.** Each runs on the same grid with the same ignition, for 300 s: the CPU budget's grid on both backends, a class A design fire, 780 MJ/m², and `tools/backend-bench.mjs`'s fire origin.
- **How a node counts.** It is untenable at 60 °C or at visibility under 10 m; in every run here, visibility decided it.
- **Where the numbers are.** The per-run table is in `benchmark/results/2026-11-11-fire-calibration-after.json` (`oldAgainstNew`).

Changes from old to new:

| Example (backend) | Heat released | Peak heat release | Hottest gas | Nodes untenable | First untenable | Solver time |
| --- | --- | --- | --- | --- | --- | --- |
| Duplex (CPU / GPU) | −1 % / −1 % | 0 % / 0 % | 532 → 507 / 542 → 498 °C | 12 → 12 / 12 → 12 | 35 → 35 / 35 → 40 s | 1.9 × / 1.7 × |
| NIBS Office (CPU / GPU) | −7 % / −9 % | −33 % / −31 % | 705 → 793 / 651 → 802 °C | 451 → 189 / 469 → 162 | 75 → 60 / 65 → 65 s | 2.0 × / 2.1 × |
| NIBS Clinic (CPU / GPU) | 0 % / 0 % | 0 % / 0 % | 553 → 782 / 562 → 811 °C | 85 → 56 / 85 → 48 | 90 → 65 / 90 → 65 s | 1.9 × / 2.0 × |
| Vienna (GPU, 27,744 cells) | −30 % | −65 % | 1,863 → 740 °C | 755 → 660 | 25 → 20 s | 1.8 × |

The heat release rises through all 300 s in every run, so none of them reaches its peak and falls. The guard stopped none of the new runs. Smoke first falls below 10 m visibility 5 s after the first untenable node, on both solvers.

Where only the design fire burns (the Duplex and the Clinic), the heat put in is the same, because it is prescribed:
- the gas near the fire runs up to 250 K hotter on the new solver;
- smoke reaches fewer nodes, but sooner (the Clinic's first untenable node at 65 s instead of 90 s).

Where the room's contents catch (the Office and Vienna), the new solver spreads less fire to them:
- the peak heat release is a third lower in the Office, and two thirds lower in Vienna, where the old run's hottest gas reached 1,863 °C;
- smoke reaches fewer nodes.

The old solver's extra spread came from the regime in which its plume leaves the axis (F3). These buildings have no measurements, so neither result can be called right. The new solver's plume is the one that matches McCaffrey.

### 9.7 The free pool, one step against the cut steps

The suppression systems test burns a pool of heptane, held hot by a pilot, in a room of the two-room test scene. With ε 10 on both runs:
- **Released heat.** One step a frame released 13.1 MJ in 150 s; the cut steps released 19.3 MJ.
- **The phase near 150 kW.** It lasted 7 frames (35 s) with one step, and 23 frames (115 s) with the cut steps.

The two drops have different causes:
- **One step a frame.** The heat release falls at 35 to 40 s while the air at the pool's node still holds 16 to 17 % oxygen, so the drop is not oxygen. Over the run, the pilot had to put 7.2 MJ into the pool to hold it hot; with the cut steps it put in only 3.2 MJ. The flame was not heating the pool, consistent with a plume pushed off its axis at ε · step = 1.
- **The cut steps.** The flame stays on the pool, and the heat release falls only when the room's air is used up: by 100 to 120 s the oxygen at the pool's node has fallen to 2 to 9 %. That drop is oxygen use, not fewer numerical losses; heat is not conserved on either solver (F6).

The pool has no measurement. On the transport data:
- **Steckler.** The cut steps are closer. The old solver diverges on Steckler's centre-burner test on every cell, and the stepped runs give the layer's rise within the band on 0.20 m cells.
- **MQH.** Neither is close: the old solver's layer is 2.3 to 3.5 times MQH's rise and the stepped runs' 2.4 to 3.9.

## 10. What still fails, and what is open

What still fails:
- **The ceiling jet** runs 1.9 to 3.1 times Alpert's rise (9.1).
- **The room's hot layer** runs 2.3 to 3.9 times MQH's rise (9.1).
- **Stage 1b**: only the neutral plane is within the band on every cell (9.3).
- **The validation's methane burner** passes 3,000 K in most sprinkler runs (F7).

None of these is calibrated.

`npm test` still fails one solver test, the foam test's "the free pool still burns at the end". Its last four frames sum to 197 kW against a threshold of 200. The sums on the other settings were:

| Setting | Last four frames | The late phase |
| --- | --- | --- |
| ε 10, one step | 322 kW | not limited by oxygen (9.7) |
| ε 10, cut steps | 257 kW | limited by oxygen |
| ε 6, cut steps | 197 kW | limited by oxygen |

The pool still burns at about 50 kW a frame, while the foamed pool is out. The main session (2026-11-11) set the threshold to 100 kW (25 kW a frame, a sixth of the pool's free burn of about 150 kW), with the reason in the test, so the test asks that the pool burn on rather than that it burn at the uncut step's rate; the check that the foamed pool is under a tenth of the free one is unchanged. Put to the user with the branch: they may restore 200 and take the change of ε instead.

Proposals for the user (none applied):

F. **Conserve heat in the transport (F6).** Carry density, or the gas's volume per unit mass (1/T at constant pressure), instead of temperature, or carry mass and enthalpy with a conservative scheme. Or add the gas's expansion to the projection (a low-Mach divergence, ∇·u = q̇/(ρ c_p T)), as FDS does. This changes every run.
  - Judge it first on the sealed box, which should hold exactly the heat it is given.
  - Then judge it on Stage 1a and Steckler.
  - Then refit the confinement (C), since the fitted 6 was fitted on a solver that makes heat.

G. **The validation's burner (F7).** Spread the burner's convective heat over a flame volume that grows with its heat release, as the design fire's sooty kind effectively does. Or give the methane burner a soot yield, so that its cells radiate. Either way, rerun the sprinkler validation. The activation readings are not in doubt; only the runs' later parts are.

H. **The worker's GPU grid.** It now sizes its grid by its measured step rate, with the cut steps uncounted, and a rate 8 % lower dropped Vienna one grid level. Size it by simulated seconds per wall second instead: then a run's grid would not depend on how many steps a frame takes.

## 11. Sources (sections 8 to 10)

- Fedkiw, Stam and Jensen (2001), McCaffrey (1979), MQH (1981), Alpert (1972), Janssens and Tran (1992), Steckler, Quintiere and Rinkinen (1982): as in section 7.
- NIST, `firemodels/exp`, Steckler_Compartment: the measured profiles and layer reductions of tests 14, 410 and 610 (a US Government work). NIST, `firemodels/fds`, Validation/Steckler_Compartment/FDS_Input_Files: the tests' geometry, burner and positions.
- FDS Validation Guide (NIST SP 1018-3), release 6.11.1, Table 4.1, for σ̃E of temperature (0.07) and velocity (0.08). The 0.05 used for heights in `steckler-checks.mjs` is an assumption, to be checked against the guide's layer depth.
