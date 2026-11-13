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

## 12. Held-out validation datasets (2026-11-12)

Status: sections 12.1 to 12.6 were written on 2026-11-11, before the suite's baseline started at 21:55, and are its pre-registration. They were not committed before the runs, so their content at 21:55 cannot be verified from git. Short smoke runs (30 to 120 s, or single tests) were used only to check the plumbing and the run times. The rule in 12.6 is not changed after seeing results; a changed rule is a new, dated pre-registration. Section 12.7 gives the runs, the two harness faults found and fixed, and the old transport's partial comparison; 12.8 gives the results on the final solver.

The suite is `tools/calibration/validation-suite.mjs`. Its data are made by `tools/calibration/validation-data.mjs`, and its shared code is in `tools/calibration/lib/` (`fds.mjs`, the FDS input reader and scene translator; `stats.mjs`, the statistics and the layer reduction). Rerun it after each change to the solver.

### 12.1 The datasets, and why they are independent

All four are measured full-scale tests from NIST's experimental database (`github.com/firemodels/exp`, commit 704e0d9), with each test's conditions from NIST's FDS validation inputs (`github.com/firemodels/fds`, commit 193faff). They are four separate test series: different years, test programmes, facilities and, for three of them, teams.

| Dataset | Series | Tests used | Fire | Enclosure | Compared |
| --- | --- | --- | --- | --- | --- |
| `steckler` | NBS, 1980 to 1981; Steckler, Quintiere and Rinkinen (1982) | 55 (all the FDS guide uses) | methane, 31.6 to 158 kW, steady, at 9 positions, on the floor or a 0.3 m block | 2.8 x 2.8 x 2.13 m room; a door 0.24 to 0.99 m wide, or a window with its sill at 0.45 to 1.37 m | the room tree's layer temperature and depth; the door centreline's peak outflow and neutral plane |
| `nbs` | NBS, 1985; Peacock, Davis and Babrauskas (1991) | 3 (100A, 100O, 100Z) | natural gas, 110 kW for 900 s | a burn room off a 12.2 m corridor (and a target room in 100Z); the corridor's end door open or shut | the layer temperature and depth at four trees |
| `nistnrc` | NIST, 2003; Hamins et al. (2006) for the US NRC | 10 (1, 2, 3, 7, 8, 9, 13, 14, 15, 18) | heptane spray, 0.4 to 2.3 MW | 21.7 x 7.04 x 3.82 m; the 2 x 2 m door open or shut, no fan | the layer at six trees; seven ceiling thermocouples; O2 and CO2 |
| `wtc` | NIST, 2004; Hamins et al. (2005) for the WTC investigation | 6 | heptane spray, 2.0 to 3.2 MW | 7.0 x 3.6 x 3.82 m; a low opening at one end, a high one at the other | the layer at two trees; two ceiling thermocouples; O2 and CO2 |

The weakest link in their independence: `nistnrc` and `wtc` were run a year apart in the same NIST laboratory under the same lead (Hamins), with the same kinds of instruments. They are different test series, so they count as two datasets here. Adding the ATF corridors (ATF's laboratory, 2008; in `firemodels/exp`) and UL FSRI's 2018 compartments (UL; MIT licence) would remove that weakness; neither is wired in yet.

Left out, and why:
- NIST/NRC tests 4, 5, 10 and 16, which ran a fan (the harness has no fan), and test 17 (toluene: its soot yield, 0.195 ± 0.052, leaves its heat release from the fuel flow uncertain, and no calorimetry was taken with the door shut).
- WTC on 0.20 m cells, and NIST/NRC on 0.20 m cells: about 60 min a test at 600 s on this PC (measured on 60 s runs), past the suite's budget. `--all-cells` runs them.
- VTT, FM/SNL, UL/NFPRF, Bittern and LLNL: other laboratories' data, which NIST's notice does not speak for (decision 4 of the 2026-11-08 plan).

**Held out.** None of these tests or measurements has been used to fit any solver parameter. The one fitted parameter, the confinement strength (6), was fitted on McCaffrey's plume and Alpert's ceiling jet (section 8, C). The walls' convection coefficient (10) came from the sprinkler validation (Vettori, UL/NFPRF and Bittern), none of which is here. Steckler's tests 14, 410 and 610 were compared in development (section 9.3). They set no solver parameter, but their 0.27 m corner tree prompted the connected-gas instrument rule (claim protocol, section 1). The rule stands: nothing in this suite is used to set a parameter. If one ever is, the dataset it was fitted on leaves the held-out set, and the claim needs three others.

### 12.2 What is measured and compared

Each quantity is compared as the FDS Validation Guide compares it (its dataplot inputs, `Utilities/Python/FDS_validation_dataplot_inputs.csv`, give the devices and the windows), as a rise above ambient:

| Quantity | Measured as | Metric | σ̃E (Table 4.1, unless stated) |
| --- | --- | --- | --- |
| Layer temperature | each thermocouple tree, reduced by Janssens and Tran (1992) | its largest rise in the window (Steckler: the steady value) | 0.07 |
| Layer depth | the same reduction: the ceiling less the interface | its largest in the window (Steckler: steady) | 0.05 |
| Ceiling jet | the thermocouples nearest the ceiling (NIST/NRC: each tree's top; WTC: one west and one east) | the largest rise | 0.07 |
| O2, CO2 | the gas analysers, as volume fractions | the largest drop of O2, rise of CO2 | 0.08 |
| Door velocity | Steckler's door centreline probes | the peak outflow (steady) | 0.08 |
| Neutral plane | the same profile | the flow's reversal height (steady) | 0.05, this project's assumption (the layer depth's); Table 4.1 gives none |

- **One reduction for both sides.** The measured trees are reduced by the same code as the model's (`lib/stats.mjs janssensTran`) at each measured time. This mirrors the guide's method. As a check, the measured interfaces match the guide's own two-layer files for every Steckler test. Its upper temperatures sometimes differ by a few per cent. For NIST/NRC and WTC the guide averages its trees; here each tree is its own pair, which is stricter.
- **Windows.** Steckler: the model's mean from 200 to 300 s, against the tests' steady state after 30 min. NBS: 0 to 900 s for the temperature, 300 to 900 s for the depth (the guide's). NIST/NRC: the guide's windows cut at 600 s. WTC: the guide's windows cut at 600 s, or at the burner's end plus 30 s.
- **The model's samples.** The model is read each second and averaged over each measured interval (10 s for NIST/NRC and WTC; 10 to 15 s for NBS), at the measured times, so both sides take their largest values from the same times.
- **Pairs.** Each device in each test is one pair. Pairs with a non-positive member are left out and counted.

### 12.3 How the model mirrors each test

- **Geometry.** The FDS input's meshes, obstructions less their holes, and open vents (`lib/fds.mjs`). Everything inside the domain but outside a mesh is wall; an open vent on a mesh's side opens to the domain's edge, which is the solver's open boundary. The scene is voxelised by the app's own `voxelize.js`, so a wall thinner than a cell seals as it would in the app. Obstructions of no thickness (FDS's thin plates: WTC's trusses and column flanges, NIST/NRC's fire pan) are left out.
- **The fire.** The app's design-fire burner (`flamescapesRun.js itemBurner`): the open cells over the burner's area, a layer at a time up to the ceiling. Heskestad's flame height picks the layers each step. The fuel is the class A item's sooty fuel and the radiant fraction is 0, with the hot sooty cells radiating (principle P2).
- **Heat release.** The test's own:
  - Steckler and NBS: the FDS input's HRRPUA times the burner's area, with its ramp. Steckler's rise takes 10 s, as `steckler-checks.mjs`.
  - NIST/NRC and WTC: the FDS input's fire ramp, scaled to the calorimetry's mean over the ramp's hold.
  - NIST/NRC's door-shut tests had no calorimetry, so the FDS input's fuel flow at 684 kg/m³ and 44.6 MJ/kg instead. On the open-door tests this flow gives 1 to 12 % more than the calorimetry.
  - NIST/NRC's burner is a 1 m circle at the pan's centre (the guide's note); WTC's is its 1 x 2 m pan.
- **Products.** The burner's CO2 yield is set to the test fuel's CO2 per unit heat (heptane 3.006 kg/kg over 44.6 MJ/kg). Here CO2 is a passive tracer of the heat, so this converts the fuel without touching the flow. O2 is consumed by Huggett's rule for every fuel.
- **Walls.** Each test's own surfaces from its FDS input, at most three per test (the solver's inert slots):
  - Steckler: 13 mm ceramic fibre board throughout.
  - NBS: the corridor's 13 mm calcium silicate on 13 mm gypsum; the burn room's 50 mm ceramic fibre and 113 mm fire brick floor.
  - NIST/NRC: 25.4 mm marinite, and a gypsum on pine floor.
  - WTC: 25.4 mm marinite and a 25.4 mm gypsum floor.

  The solver lumps a wall into whole cells, 0.2 to 0.34 m thick. So each surface's density is set to its areal mass over the cell (ρ = Σ ρ_i L_i / h) and its specific heat to Σ ρ_i c_i L_i / Σ ρ_i L_i. A wall one cell thick then holds the board's heat capacity per unit area, not ten to twenty times it. Its conductivity and emissivity are the exposed layer's, at the first point of any ramp. This is input fidelity, set from the documented materials by one rule for every test before any comparison. It is not tuning, and it changes no solver parameter: the solver takes it as a run option (`inert`). `--walls stone` runs the solver's stone instead, as a diagnostic.
- **The rest of the test conditions.** The solver's ambient is the test's (FDS TMPA). The cells are the app's: 0.20, 0.27 and 0.34 m, CPU.
- **The run itself.** Each second the run is checked as the app checks it (`healthReadings`, `healthVerdict`), and its steps are cut as the app cuts them (`substepsFor`). A run that fails stops, and its readings are set aside.
- **Instruments.** Trees, points and the door's column read only open cells in the fire's connected gas (the solver's regions after the first second). A tree's ceiling is the top of its connected open cells. A point whose cell is solid, or outside that gas, moves to the nearest connected open cell within four cells, and the distance is recorded.
- **One solver for the whole run.** The suite copies the solver's modules at its start, so every run uses the same code. It records the git state, the copies' digests and the solver options.

### 12.4 Known mismatches

- **Walls.** A wall is a lumped cell with no conduction depth: thermally thin. The tests' boards (13 to 113 mm) are thin for the first minutes and then lose heat through their backs. FDS conducts in one dimension through each layer.
- **Ceilings and openings snap to the grid.** The app's voxeliser seals any cell a slab enters by more than 8 %. So a ceiling snaps down by up to a cell: Steckler's 2.13 m room is 1.89 m tall on 0.27 m cells and 2.04 m on 0.34 m cells. Openings snap to whole cells: Steckler's 0.74 m door is 0.54 m wide on 0.27 m cells and 0.68 m on 0.34 m cells. Each run records its trees' ceilings and the door's modelled width.
- **The burner's fuel.** It is the class A item's sooty fuel, not methane or heptane, and it has no prescribed radiant fraction. A methane flame radiates 15 to 20 % of its heat and a heptane spray 44 %. Here that heat is lost only through the solver's soot radiation.
- **Duration.** NIST/NRC and WTC are cut at 600 s; their tests ran 5 to 50 min. Steckler's steady state is the model's 200 to 300 s mean against the tests' 30 min.
- **Leakage and spray.** NIST/NRC's door-shut rooms are sealed here; the measured leakage, 0.06 to 0.08 m², is below a cell. The spray burners are pool-like burners over the pan.
- **Not matched surfaces.** NBS: the corridor's floor (gypsum board in FDS) and the target room's walls and floor take the corridor walls' material.

### 12.5 Running it

`node tools/calibration/validation-data.mjs` fetches the raw files at the fixed commits into `tools/calibration/data/cache/` (git-ignored). It checks their SHA-256 digests against those recorded, and writes the reduced numbers to `tools/calibration/data/<dataset>.json`; `--offline` reduces from the cache. The suite reads only the reduced files.

`node tools/calibration/validation-suite.mjs [--datasets steckler,nbs,nistnrc,wtc] [--cells 0.20,0.27,0.34] [--solver vorticity=6,cfl=1] [--jobs N] [--tests REGEX] [--walls stone] [--all-cells] [--json out.json]`
- `--solver` overrides DEFAULTS as the other calibration tools do. For example, `--solver conservative=0,expansion=0,vorticity=6` reproduces the transport before proposal F.
- `--summary out.json` recomputes an earlier run's statistics with the current code, both ways (12.6), and prints them; `--write` saves them back.
- `--jobs` defaults to 4. A NIST/NRC run on 0.27 m cells holds about 200 MB and takes about 37 min on this PC; the whole suite took 167 min at 4 jobs.

**Licence.** Both repositories carry NIST's notice: NIST-developed works, not subject to copyright in the United States; to be acknowledged as NIST's, with changes marked. All four datasets are NIST or NBS tests, so the reduced numbers are committed with that notice and a note of the change, as in each file's `notice`. The raw files stay in the git-ignored cache.

### 12.6 Acceptance rule (pre-registered)

Statistics as in section 2 (FDS Validation Guide, section 4.4), with σ̃E from the table in 12.2. The bands are this project's (2026-11-08 plan, section 8), on the FDS Validation Guide's statistics: the guide defines the statistics, not the bands. The rule, per quantity and per cell size:

1. **A dataset is in band** when its own pairs (its tests and devices on that cell) give a tier within the calibrated band (0.8 ≤ δ ≤ 1.25 and σ̃M ≤ 0.35), and none of its runs on that cell stopped. One stopped run puts the whole dataset out of band for every quantity it measures on that cell, so a stopped run cannot help a verdict by leaving the rest.
2. **Calibrated on that cell** when at least three datasets are in band, and all datasets' pairs pooled also lie within the calibrated band.
3. **Indicative on that cell** when the pooled pairs lie within the indicative band (0.67 ≤ δ ≤ 1.5 and σ̃M ≤ 0.5).
4. **Otherwise not calibrated.**

What the rule allows, before any result:
- The layer's temperature and depth are measured in all four datasets, so they can be calibrated.
- The ceiling jet, O2 and CO2 are measured only in NIST/NRC and WTC, the two datasets from one laboratory. They can be indicative at most until a third dataset measures them.
- The door velocity and neutral plane come from Steckler alone, so they too are indicative at most.
- On 0.20 m cells only Steckler and NBS run, so nothing can be calibrated there. This costs the app little: the app's calibrated cells are 0.27 to 0.34 m (decision 2 of the 2026-11-08 plan).
- A quantity is calibrated only on the cells where the rule holds. The app may claim it only for runs on those cells, within the range of fires (D*/δx) and rooms the datasets cover.

The suite also reports the claim rule of [fire-calibration-claim-protocol-2026-11-12.md](fire-calibration-claim-protocol-2026-11-12.md), section 4, which was also written before any held-out result: in band in at least three datasets and outside the indicative band in none; indicative in at least two. A dataset with a stopped run counts as outside the indicative band. Where the two rules differ, the stricter one is the verdict.

**Added 2026-11-12, 00:28** (the claim protocol's addendum, commit 0afce90). A check on geometry alone found that the grid seals some of Steckler's doors:
- the 0.24 to 0.25 m doors (tests 10, 114 and 514) on every cell size;
- the 0.36 m doors (tests 11, 144 and 544) on 0.34 m cells.

The check floods the fire's open cells and asks whether they reach the domain's open boundary (`tools/calibration/lib/represented.mjs`). It was prompted by held-out output: test 10's missing door velocity in the old-transport comparison run. By 00:28 nine final-solver Steckler runs had finished, steckler/10@0.2 among them, though none had been read. In those 12 of the 165 Steckler runs the room has no opening, so they do not represent their tests. Every verdict is given both ways (12.8), and none moved.

The rule: a run whose fire's gas cannot reach the open boundary, though its test had an opening, is left out and reported as "not represented on this grid". Tests sealed by design (NIST/NRC 1, 2, 7, 8 and 13, and NBS 100O: door shut, leakage below a cell) are unaffected. Every statistic and verdict is reported both ways: as pre-registered, with every test, and with this rule. The rule's verdict is the one the app states, with the pre-registered one beside it. For the app the same fact is a limit: an opening narrower than about a cell closes on the grid.

### 12.7 The runs

**Which runs count.** The verdict comes from one run on the solver as it stands after proposal F and the confinement's refit:
- DEFAULTS vorticity 3, conservative 1, expansion 1, no overrides;
- `flamescapes.js` SHA-256 352694a972e2543b, the file as committed in 7e2e9f5 (heat-conserving transport, confinement refitted);
- 4 jobs, as this PC had about 2 GB of memory free; a NIST/NRC run on 0.27 m cells holds about 200 MB;
- started 2026-11-11 21:55, output `benchmark/results/2026-11-12-validation-baseline.json`.

Two earlier starts, with 16 jobs on an intermediate solver (vorticity 6), were stopped before any run finished and wrote nothing.

**The old transport, for comparison** (`--solver conservative=0,expansion=0,vorticity=6`, 2 jobs alongside the final run). It was stopped after 67 of its 206 runs by the main session's choice, to give the final run the CPU. Its log is kept, reconstructed into `benchmark/results/2026-11-12-validation-old-transport-partial.json`; the suite writes its full JSON only at the end.

- **What ran.** The queue runs the costliest first, so the 67 are:
  - every NIST/NRC run on 0.27 and 0.34 m;
  - every WTC and NBS run on 0.27 and 0.34 m;
  - NBS on 0.20 m;
  - 26 Steckler runs on 0.20 m.
- **What stopped.** 19 runs, all NIST/NRC: every one of its tests on 0.27 and 0.34 m, except test 15 on 0.34 m, which ran to 600 s.
  - The guard's reason, every time: the gas held more than 1.5 times the heat put in, plus 1 MJ (energy).
  - The times, 0.27 m: test 13 at 7 s; 02, 09 and 03 at 8 s; 08, 14 and 18 at 9 s; 15 at 10 s; 07 at 11 s; 01 at 18 s.
  - The times, 0.34 m: 13 at 7 s; 02 at 9 s; 08 and 14 at 10 s; 09 and 03 at 11 s; 18 at 16 s; 07 at 19 s; 01 at 25 s.
- **Why.** This is finding F6 at the scale of a 21.7 m compartment: the old transport makes heat as the burner's gas mixes, and early in a slow ramp the heat put in is still small.
- **The final solver** ran the same tests without a stop.

**A second harness fault, and its fix (2026-11-12, recorded in the claim protocol before its rerun).**
- **What it was.** The run records showed WTC's west tree on 0.27 m cells with a ceiling of 2.97 m, against 3.82 m. A small instrument bracket (1.0 to 1.2 x 0.5 to 0.6 x 3.2 to 3.3 m) was sealed into two whole cells in the tree's column. The tree read only the cells below it, so its Janssens and Tran reduction lost the top 0.8 m of the layer.
- **The fix.** A tree now reads every open, fire-connected cell of its column below the test's nominal ceiling, skipping solid ones, and takes its ceiling at the top of the highest (`lib/fds.mjs treeCells`). A check on geometry alone, over all 417 tree columns of every test on every grid, finds that this changes only those six: WTC tests 1 to 6, west tree, 0.27 m.
- **The rerun.** The six were rerun with the fix (`benchmark/results/2026-11-12-validation-wtc-tree-fix.json`). Ten unaffected runs rerun alongside repeated bit for bit: WTC on 0.34 m, NBS 100A and Steckler 14 on 0.27 and 0.34 m. `benchmark/results/2026-11-12-validation-baseline-wtc-fixed.json` is the as-made run with the six replaced.
- **What changed.** WTC on 0.27 m, as made then fixed (δ, with σ̃M in brackets): the layer temperature 1.03 (0.10) then 1.10 (0.07); the layer depth 0.95 (0.14) then 1.09 (0.05). No verdict changed.
- **The literal rerun.** The full rerun of all 206 runs on the fixed harness that protocol section 5 asks for started on 2026-11-12 at 01:09, as `benchmark/results/2026-11-12-validation-final.json`. Given the bit-for-bit repeat, it is expected to match the merged file.

### 12.8 Results on the final solver

The fixed harness, with the sealed-door rule (12.6). Each cell gives δ, σ̃M in brackets, the number of pairs, and the dataset's tier. "In band" means within the calibrated band.

| Quantity | Dataset | 0.20 m | 0.27 m | 0.34 m |
| --- | --- | --- | --- | --- |
| Layer temperature rise | Steckler | 1.30 (0.17), 52: indicative | 1.09 (0.08), 52: in band | 1.12 (0.11), 49: in band |
| | NBS multi-room | 0.64 (0.68), 12: not calibrated | 0.96 (0.61), 12: not calibrated | 0.78 (0.77), 12: not calibrated |
| | NIST/NRC | not run | 1.04 (0.20), 60: in band | 1.14 (0.20), 60: in band |
| | WTC | not run | 1.10 (0.07), 12: in band | 1.18 (0.07), 12: in band |
| Layer depth | Steckler | 0.97 (0.16), 52: in band | 1.16 (0.09), 52: in band | 1.28 (0.12), 49: indicative |
| | NBS multi-room | 1.14 (0.29), 12: in band | 1.11 (0.20), 12: in band | 1.29 (0.24), 12: indicative |
| | NIST/NRC | not run | 0.94 (0.11), 60: in band | 0.95 (0.11), 60: in band |
| | WTC | not run | 1.09 (0.05), 12: in band | 1.10 (0.05), 12: in band |
| Ceiling jet rise | NIST/NRC | not run | 1.15 (0.26), 69: in band | 1.31 (0.19), 69: indicative |
| | WTC | not run | 1.32 (0.09), 12: indicative | 1.30 (0.10), 12: indicative |
| O2 drop | NIST/NRC | not run | 1.31 (0.09), 15: indicative | 1.44 (0.13), 15: indicative |
| | WTC | not run | 1.77 (0.16), 6: not calibrated | 2.19 (0.17), 6: not calibrated |
| CO2 rise | NIST/NRC | not run | 1.03 (0.08), 10: in band | 1.11 (0.08), 10: in band |
| | WTC | not run | 1.59 (0.34), 6: not calibrated | 2.00 (0.33), 6: not calibrated |
| Door peak outflow | Steckler | 0.69 (0.14), 52: indicative | 0.65 (0.11), 52: not calibrated | 0.58 (0.40), 48: not calibrated |
| Neutral plane height | Steckler | 0.96 (0.05), 52: in band | 0.98 (0.10), 52: in band | 0.94 (0.11), 47: in band |

The verdicts. Each cell gives the pooled δ (σ̃M), the datasets in band, the verdict of plan 12.6 and of the claim protocol's section 4, and in bold the stricter, which is the verdict:

| Quantity | 0.20 m | 0.27 m | 0.34 m |
| --- | --- | --- | --- |
| Layer temperature rise | 1.21 (0.48); 0 of 2; indicative / not calibrated: **not calibrated** | 1.06 (0.24); 3 of 4; calibrated / indicative: **indicative** | 1.12 (0.33); 3 of 4; calibrated / indicative: **indicative** |
| Layer depth | 1.00 (0.19); 2 of 2; indicative / indicative: **indicative** | 1.05 (0.15); 4 of 4; calibrated / calibrated: **calibrated** | 1.11 (0.18); 2 of 4; indicative / indicative: **indicative** |
| Ceiling jet rise | not run | 1.18 (0.25); 1 of 2; indicative / indicative: **indicative** | 1.31 (0.18); 0 of 2; indicative / indicative: **indicative** |
| O2 drop | not run | 1.44 (0.18); 0 of 2; indicative / not calibrated: **not calibrated** | 1.65 (0.23); 0 of 2; not calibrated / not calibrated: **not calibrated** |
| CO2 rise | not run | 1.23 (0.27); 1 of 2; indicative / not calibrated: **not calibrated** | 1.43 (0.33); 1 of 2; indicative / not calibrated: **not calibrated** |
| Door peak outflow | 0.69 (0.14); 0 of 1; indicative / not calibrated: **not calibrated** | 0.65 (0.11); 0 of 1; not calibrated / not calibrated: **not calibrated** | 0.58 (0.40); 0 of 1; not calibrated / not calibrated: **not calibrated** |
| Neutral plane height | 0.96 (0.05); 1 of 1; indicative / not calibrated: **not calibrated** | 0.98 (0.10); 1 of 1; indicative / not calibrated: **not calibrated** | 0.94 (0.11); 1 of 1; indicative / not calibrated: **not calibrated** |

**As pre-registered, with every test.** Every verdict is the same. Only Steckler's numbers differ:
- its layer temperature: 1.29 (0.16), 1.09 (0.09) and 1.14 (0.12) on the three cells;
- its layer depth: 0.98 (0.17), 1.13 (0.23) and 1.27 (0.12);
- the pooled layer depth on 0.27 m: 1.04 (0.19).

The 12 runs left out under the rule:
- tests 10, 114 and 514 on 0.20, 0.27 and 0.34 m;
- tests 11, 144 and 544 on 0.34 m.

**How independent the four are.** The four series come from two institutions, NBS and NIST (NBS became NIST in 1988), and NIST/NRC and WTC share a laboratory and lead. Counted as one, with Steckler set aside, the depth would be indicative: the merged NIST/NRC and WTC pairs give δ 0.96, σ̃M 0.12 on 0.27 m cells (72 pairs), beside NBS's 1.11 (0.20), two datasets.

**What the one calibrated verdict covers.** The hot layer's depth, by Janssens and Tran's reduction of a thermocouple tree, on 0.27 m cells only. The range it holds over:
- **Fires.** Prescribed heat releases of 32 kW to 3.2 MW: D*/δx 0.9 to 5.6 on those cells (Steckler 0.9 to 1.7, NBS 1.5, NIST/NRC 2.5 to 5.0, WTC 4.7 to 5.6).
- **Enclosures.**
  - single rooms of 17 m³ (2.8 x 2.8 x 2.13 m) to 584 m³ (21.7 x 7.04 x 3.82 m);
  - a burn room off a 12.2 m corridor;
  - ceilings of 2.1 to 3.8 m.
- **Openings.**
  - one door 0.36 to 1.0 m wide, or a window with its sill up to 1.37 m;
  - a door to a corridor;
  - a 2 x 2 m door;
  - two vents at opposite ends;
  - rooms closed.
- **Walls.** Lightweight insulating boards (ceramic fibre, calcium silicate on gypsum, marinite), with gypsum or fire-brick floors.
- **Times.** Up to 600 to 900 s, or Steckler's steady state.

Outside it, the depth stays at most indicative:
- other cells: on 0.20 and 0.34 m it is indicative;
- narrower openings: doors under about a cell close on the grid;
- masonry or concrete walls;
- longer fires.

**What the rest says.**
- **The layer's temperature rise** is within the band in three datasets on 0.27 and 0.34 m. NBS's multi-room tests are outside even the indicative band, with large scatter (σ̃M 0.6 to 0.8). By tree, as a share of the measured rise:
- the burn room's tree: 0.8 to 1.3;
- the corridor tree a metre from its door: 0.6 to 1.8;
- the trees 5 and 11 m down the corridor and in the target room: 0.1 to 0.9.

The model's layer cools too fast along the 12 m corridor. So the rise is indicative.
- **The ceiling jet** is indicative on both cells, running 15 to 30 % hot.
- **O2 and CO2** fail on WTC's 2 to 3 MW fires, where the model's oxygen drop and CO2 rise run 1.6 to 2.2 times the measured. NIST/NRC is within or near the band for both.
- **Steckler's door outflow** reads a third low, as section 9.3 found. The neutral plane is in band on every cell, but it has one dataset.

**Costs** (CPU, 4 jobs, i7-14700; solver seconds a run, median):
- NIST/NRC: 37 min on 0.27 m, 11 min on 0.34 m;
- WTC: 13 min on 0.27 m, 4 min on 0.34 m;
- NBS: 16, 4 and 1.4 min on the three cells;
- Steckler: 49, 15 and 5 s.

The whole suite took 167 min at 4 jobs. The 30 min a cell size first aimed at needs about 16 jobs and more memory than this PC had free, and even then the longest single run, NIST/NRC on 0.27 m, takes about 40 min.

**The geometry as modelled** (each run records it):
- **Steckler.** The ceiling is 2.00 m on 0.20 m cells and 2.04 m on 0.34 m cells. On 0.27 m cells it snaps down to 1.89 m in most of the room. Beside two walls, where the tree stands, a row of cells stays open to 2.16 m, joined to the room from below.
- **Steckler's doors.** They are modelled at 0.54 to 1.11 times their width (the sealed ones aside). The 0.46 m window is one cell tall on 0.34 m cells, so it has no neutral plane there.
- **The others.** NIST/NRC and WTC ceilings are 3.74 to 3.78 m. Point devices moved at most 0.26 m to an open, connected cell. Every run took its walls' materials, and every burner released its table's heat to within 0.1 %.

**Superseded.** The results above are on the solver of 7e2e9f5. On 2026-11-12 an independent review found two solver faults that change the physics: the transport's fixed x, y, z sweep order (fixed by alternating the order by step) and the porous cells' projection. Under the claim protocol's section 5, every comparison is rerun on the fixed solver and the confinement is refitted. The full literal rerun on 7e2e9f5 was stopped after 4 of 206 runs, because that solver was about to change; its log is kept in `benchmark/results/2026-11-12-validation-final-partial.json`.

### 12.9 Extension for the next solver (pre-registered 2026-11-12, before any run on it)

The user now requires every quantity, smoke visibility included, to have its fair chance of calibration; the label still follows the evidence. What follows is fixed before any run on the next solver.

**1. A fifth dataset: ATF corridors.**
- **What it is.** The ATF Fire Research Laboratory's 2008 tests, in `firemodels/exp` under NIST's notice: the data of a US federal laboratory other than NIST.
- **The tests.** Six: 48, 97, 242, 250 and 485 kW, and a pulsed fire up to 485 kW, all natural gas, modelled as methane.
- **The enclosure.** Two 17 m corridors (1.75 m wide, 2.4 m tall) joined by a stairwell. The lower corridor's far door is shut; the upper corridor's end is open.
- **Compared,** as the guide's dataplot inputs choose:
  - the layer at five trees, each from its own floor: A and B in the lower corridor, C at the foot of the stairs, F and G in the upper corridor;
  - seven ceiling thermocouples: A-10, B-10, C-10, D-10, E-08, F-10 and G-10.
- **Runs.** 600 s, on 0.27 and 0.34 m cells. On 0.20 m cells the scene has about 78,000 cells, so it is not run there by default.
- **Walls.** The FDS input's 12.7 mm gypsum board, with its concrete ground floor. The guide's description has cement board on plywood for the floor; the pine stairs take the gypsum; the stairs and rails snap to whole cells.
- **Independence.** A different laboratory and team. It is the third dataset for the ceiling jet, after NIST/NRC and WTC, and the fifth for the layer.

**2. Smoke: NIST/NRC's obscuration.**
- **What is measured.** The laser path under the ceiling's far corner: FDS's TRANSMISSION device, 0.8 m long at 3.62 m.
- **The metric.** As the guide compares it: the largest obscuration per metre, 100 − T (%/m), with σ̃E 0.12 (the guide's scatter-plot input for smoke obscuration).
- **The model.** The soot density in the path's connected open cells gives K = 8700 m²/kg × ρ Y_soot (Mulholland and Croarkin 2000; `flamescapesRun.js SOOT_EXTINCTION`), as the app's visibility 3/K uses it. The obscuration is 100 (1 − e^−K). Visibility is 3/K, so its bias is the inverse of K's.
- **Only one dataset.** No other NIST or public dataset fits the app's cells:
  - FAA's cargo compartments are 1.4 m tall with a 5 kW tray, four cells tall on 0.34 m cells;
  - FM/FPRF's data centre is not NIST's;
  - NIST's smoke alarm tests give activation times only.

  So smoke cannot reach indicative under the protocol, which needs two datasets. It is reported and labelled not calibrated.

**3. The burner's fuel is the test's.** This is input fidelity, one rule for every test, found by a check that uses no measured data.
- **The fault.** The burner had the class A item's heat of combustion (18 MJ/kg) with its CO2 yield matched per joule. At the analysers, the model's CO2 rise over its O2 drop is 0.51 to 0.55, where the burner's inputs give 0.64 (heptane, 7/11 = 0.636).
- **Why.** The class A fuel adds 2.5 times heptane's mass per joule, and the mass balance closes with water. The extra product moles dilute the gas, so the O2 drop reads about 14 % high.
- **The fix.** The burner takes the test fuel's heat of combustion and yields:
  - methane: 50.0 MJ/kg, CO2 2.744;
  - heptane: 44.6 MJ/kg, CO2 3.006, CO 0.008;
  - soot as each series documents it: Steckler 0 and ATF 0 (their FDS inputs), NBS 0.005 (its FDS input), NIST/NRC 0.0149 (measured in the tests, as the guide quotes it), WTC 0.015 (its FDS input).
- **Its consequence.** The solver's flame radiates only through its soot, so the methane burners, with little or no soot, would lose no heat by radiation. A methane flame radiates 15 to 20 % of its heat. Amendment 8 gives the burners their radiant fraction for this reason.
- **The former rule** is kept as a diagnostic: `--fuel app`. It runs the app's design-fire burner: class A fuel, CO2 matched per joule, radiant fraction 0.

**4. NBS's geometry and materials, checked without measured data.**
- On 0.20 m cells the scene is exact: the burn room's door 0.80 m wide up to its 1.60 m soffit, ceilings 2.20 and 2.40 m, the end door 1.60 m², the plan areas.
- On 0.27 m cells the corridor's ceiling snaps to 2.16 m and the door to 0.54 m. On 0.34 m cells the ceilings are 2.04 and 2.38 m.
- The walls take the FDS input's surfaces.

No input fault explains NBS's scatter: its far corridor trees run cold on every cell (12.8), including the one where the scene is exact. Nothing changed.

**5. Unchanged:**
- the verdict rule (12.6 and the claim protocol's section 4; the stricter governs);
- the sealed-door rule and the tree rule (12.7);
- the windows and horizons;
- the cells: 0.20 m runs only Steckler and NBS;
- every other input.

**6. Records.** Each results file records SHA-256 digests of the harness (`lib/*.mjs`, `validation-suite.mjs`, `data/*.json`) in `git.harnessDigests`, and `--summary --write` refuses to rewrite a file made with a different harness. The run's output is logged to a file beside its results.

**7. The run.** After the main session's "solver final":

`node tools/calibration/validation-suite.mjs --cells 0.20,0.27,0.34 --jobs 6 --json benchmark/results/2026-11-12-validation-final.json > benchmark/results/2026-11-12-validation-final.log 2>&1`

It covers all five datasets, 218 runs: the 206 before and ATF's 12. Six jobs hold about 1.3 GB. It should take about 2.5 h.

**8. Amendment, 2026-11-12, before any run on the next solver: the burner's radiant fraction and yields from each FDS input.** With each test's own fuel, the methane burners carry little or no soot, so the solver's soot radiation would no longer take their radiant heat. Their layers would then run hot for a reason that is not the transport.

The rule, the same for every test:
- **Radiant fraction.** Each burner radiates the FDS input's RADIATIVE_FRACTION along its rays (`flamescapes.js createBurner` and `burnerStep`, 16 rays), its convective rest into the flame cells. Where the input leaves the fraction out, FDS's own default for its fuel applies: the FDS User's Guide's table of default radiative fractions gives METHANE 0.20 and N-HEPTANE 0.40, and 0.35 only for species it does not list.
- **Soot and CO.** As the input gives them. CO2 follows by the carbon balance: the fuel's carbon, less the soot's and the CO's, as CO2.

The values:

| Dataset | Tests | Radiant fraction | Soot | CO | CO2 (kg/kg) |
| --- | --- | --- | --- | --- | --- |
| Steckler | all 55 | 0.20 (FDS's methane default) | 0 | 0 | 2.743 |
| NBS | 100A | 0.20 (FDS's methane default) | 0.005 | 0 | 2.725 |
| NBS | 100O, 100Z (acetylene added) | 0.30 (input) | 0.010 | 0 | 2.707 |
| NIST/NRC | all 10 | 0.44 (input) | 0.0149 (the input's lumped products; the tests' measured value) | 0.008 | 3.007 |
| WTC | 1, 4, 5, 6 | 0.44 (input) | 0.015 | 0.008 | 3.007 |
| WTC | 2, 3 | 0.39 (input) | 0.114 | 0.042 | 2.591 |
| ATF | all 6 | 0.20 (FDS's methane default) | 0 | 0 | 2.743 |

Heats of combustion: methane 50.0 MJ/kg, heptane 44.6 MJ/kg.

Known mismatches:
- **Double counting for sooty fuels.** For WTC 2 and 3, the solver's soot radiation from the hot burner gas comes on top of the prescribed fraction, so those fires may radiate more than their stated share. FDS instead holds the flame's emission to the stated fraction.
- **NBS 100A.** The guide's text gives 0.20 for natural gas, which FDS's default also gives.
- **The burner's rays** are absorbed by soot on the way and by the first solid they meet, as the solver's own radiation is.

**What the rule allows, before any result:**
- Layer temperature and depth: five datasets, so they can be calibrated.
- Ceiling jet: three (NIST/NRC, WTC and ATF), so it can be calibrated; the first two share a laboratory.
- O2 and CO2: two (NIST/NRC and WTC), so indicative at most.
- Smoke obscuration, and with it visibility: one (NIST/NRC), so not calibrated whatever its result.
- Door velocity and neutral plane: one (Steckler), so not calibrated.

### 12.10 Results on the next solver (2026-11-12)

The hot layer's depth is calibrated on 0.27 m cells, by both rules and both ways. Every other quantity is indicative or not calibrated on every cell. The numbers come from two result files only, each recomputed from its runs to the third decimal.

**The two runs.**
- **The first run.** The suite as pre-registered in 12.9, with amendment 8: 218 runs, 6 jobs.
  - The solver was that of 0c62bd9, vorticity 3 after the refit. The run records its git head as the handoff commit 04c5d54, which carries the same `flamescapes.js` (SHA-256 80787713). The working tree held uncommitted changes to the GPU kernels and some interface modules, not to the CPU solver.
  - It started at 02:46 and took 5,510 s. Output: `benchmark/results/2026-11-12-validation-final.json` and `.log`.
- **Two runs stopped.** Both ATF runs on 0.27 m cells were stopped by the health guard's temperature limit: the 485 kW test at 56 s (3,105 K) and the pulsed test at 545 s (3,033 K).
- **The fault and its fix.** The main session traced the stops to the prescribed burner heating cells that had no oxygen left. Commit 6ee8c93 fixes it: a burner's fuel burns only with oxygen, and the rest reaches outward up to Heskestad's flame height or waits, pending. The claim protocol's addendum (7292e28, its time corrected in 51674ec) was committed before the rerun's verdict. It records the fault, the fix and that both runs are reported.
- **How the fault was found.** The search was prompted by held-out data: the two stopped runs are runs of the held-out tests. The fix rests on a check that uses no measured data. In every run of both files the burner released exactly its table's heat up to the run's end.
- **The second run.** The same 218 runs on 6ee8c93 (git head 6ee8c93, a clean solver tree), 6 jobs, 7,433 s. No run stopped. Output: `benchmark/results/2026-11-12-validation-o2.json` and `.log`.
- **The harness.** Both runs record the same harness digests as the present one.

**The verdicts on 6ee8c93, with the sealed-door rule.** Each row gives the pooled δ (σ̃M) and pairs, the series in band, the verdict of plan 12.6 and of the claim protocol's section 4, the stricter of the two, and the stricter on 0c62bd9.

| Quantity | Cell | Pooled δ (σ̃M), pairs | Series in band | Plan 12.6 | Protocol 4 | Verdict | On 0c62bd9 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Layer temperature rise | 0.20 m | 1.21 (0.45), 64 | 0 of 2 | indicative | not calibrated | **not calibrated** | same |
| Layer temperature rise | 0.27 m | 1.10 (0.61), 166 | 3 of 5 | not calibrated | indicative | **not calibrated** | same |
| Layer temperature rise | 0.34 m | 1.03 (0.39), 163 | 3 of 5 | indicative | indicative | **indicative** | same |
| Layer depth | 0.20 m | 1.01 (0.19), 64 | 2 of 2 | indicative | indicative | **indicative** | same |
| Layer depth | 0.27 m | 1.08 (0.15), 166 | 5 of 5 | calibrated | calibrated | **calibrated** | indicative |
| Layer depth | 0.34 m | 1.16 (0.18), 163 | 2 of 5 | indicative | indicative | **indicative** | same |
| Ceiling jet rise | 0.27 m | 1.23 (0.75), 123 | 2 of 3 | not calibrated | indicative | **not calibrated** | same |
| Ceiling jet rise | 0.34 m | 1.07 (0.49), 123 | 2 of 3 | indicative | indicative | **indicative** | same |
| O2 drop | 0.27 m | 1.34 (0.21), 21 | 1 of 2 | indicative | not calibrated | **not calibrated** | same |
| O2 drop | 0.34 m | 1.55 (0.29), 21 | 0 of 2 | not calibrated | not calibrated | **not calibrated** | same |
| CO2 rise | 0.27 m | 1.30 (0.27), 16 | 1 of 2 | indicative | not calibrated | **not calibrated** | same |
| CO2 rise | 0.34 m | 1.51 (0.34), 16 | 1 of 2 | not calibrated | not calibrated | **not calibrated** | same |
| Smoke obscuration | 0.27 m | 1.76 (0.37), 10 | 0 of 1 | not calibrated | not calibrated | **not calibrated** | same |
| Smoke obscuration | 0.34 m | 1.77 (0.37), 10 | 0 of 1 | not calibrated | not calibrated | **not calibrated** | same |
| Door peak outflow | 0.20 m | 0.69 (0.08), 52 | 0 of 1 | indicative | not calibrated | **not calibrated** | same |
| Door peak outflow | 0.27 m | 0.63 (0.10), 52 | 0 of 1 | not calibrated | not calibrated | **not calibrated** | same |
| Door peak outflow | 0.34 m | 0.56 (0.35), 48 | 0 of 1 | not calibrated | not calibrated | **not calibrated** | same |
| Neutral plane height | 0.20 m | 0.96 (0.07), 52 | 1 of 1 | indicative | not calibrated | **not calibrated** | same |
| Neutral plane height | 0.27 m | 0.97 (0.09), 52 | 1 of 1 | indicative | not calibrated | **not calibrated** | same |
| Neutral plane height | 0.34 m | 0.91 (0.15), 47 | 1 of 1 | indicative | not calibrated | **not calibrated** | same |

**By series, with the sealed-door rule.** Each cell gives δ, σ̃M in brackets, pairs and tier on 6ee8c93, with 0c62bd9's in square brackets where they print differently. NIST/NRC and WTC were not run on 0.20 m, nor ATF.

| Quantity | Series | 0.20 m | 0.27 m | 0.34 m |
| --- | --- | --- | --- | --- |
| Layer temperature rise | Steckler | 1.29 (0.18), 52: indicative | 1.08 (0.09), 52: in band | 1.09 (0.11), 49: in band |
| | NBS multi-room | 0.70 (0.65), 12: not calibrated | 1.08 (0.70), 12: not calibrated | 0.87 (0.77), 12: not calibrated |
| | NIST/NRC | not run | 0.92 (0.14), 60: in band | 0.99 (0.14), 60: in band |
| | WTC | not run | 1.02 (0.07), 12: in band [1.03 (0.07)] | 1.10 (0.07), 12: in band |
| | ATF corridors | not run | 1.66 (1.32), 30: not calibrated [1.63 (1.38), 20, 2 stopped] | 0.99 (0.66), 30: not calibrated [0.96 (0.65)] |
| Layer depth | Steckler | 0.99 (0.16), 52: in band | 1.15 (0.11), 52: in band | 1.28 (0.12), 49: indicative |
| | NBS multi-room | 1.13 (0.28), 12: in band | 1.10 (0.21), 12: in band | 1.26 (0.25), 12: indicative |
| | NIST/NRC | not run | 0.95 (0.11), 60: in band | 0.97 (0.12), 60: in band |
| | WTC | not run | 1.12 (0.05), 12: in band | 1.12 (0.05), 12: in band |
| | ATF corridors | not run | 1.20 (0.12), 30: in band [1.21 (0.12), 20, 2 stopped] | 1.29 (0.14), 30: indicative [1.28 (0.14)] |
| Ceiling jet rise | NIST/NRC | not run | 1.03 (0.20), 69: in band | 1.10 (0.15), 69: in band |
| | WTC | not run | 1.19 (0.07), 12: in band | 1.19 (0.07), 12: in band |
| | ATF corridors | not run | 1.61 (1.24), 42: not calibrated [1.55 (1.29), 28, 2 stopped] | 0.93 (0.73), 42: not calibrated [0.89 (0.71)] |
| O2 drop | NIST/NRC | not run | 1.17 (0.08), 15: in band | 1.26 (0.09), 15: indicative |
| | WTC | not run | 1.78 (0.16), 6: not calibrated [1.77 (0.18)] | 2.27 (0.17), 6: not calibrated [2.23 (0.19)] |
| CO2 rise | NIST/NRC | not run | 1.07 (0.08), 10: in band [1.06 (0.08)] | 1.14 (0.08), 10: in band |
| | WTC | not run | 1.71 (0.31), 6: not calibrated [1.71 (0.33)] | 2.17 (0.28), 6: not calibrated [2.17 (0.30)] |
| Smoke obscuration | NIST/NRC | not run | 1.76 (0.37), 10: not calibrated | 1.77 (0.37), 10: not calibrated |
| Door peak outflow | Steckler | 0.69 (0.08), 52: indicative | 0.63 (0.10), 52: not calibrated | 0.56 (0.35), 48: not calibrated |
| Neutral plane height | Steckler | 0.96 (0.07), 52: in band | 0.97 (0.09), 52: in band | 0.91 (0.15), 47: in band |

**As pre-registered, with every test.** Every verdict is the same as with the rule, in both runs. Only Steckler's rows differ, the same in both runs:
- its layer temperature: 1.31 (0.18), 1.11 (0.14) and 1.15 (0.18) over 55 tests on the three cells;
- its layer depth: 1.00 (0.16), 1.15 (0.11) and 1.28 (0.12).

With them the pooled values move:

| Quantity | Cell | 6ee8c93 | 0c62bd9 |
| --- | --- | --- | --- |
| Layer temperature rise | 0.20 m | 1.23 (0.45), 67 | 1.23 (0.45), 67 |
| Layer temperature rise | 0.27 m | 1.11 (0.61), 169 | 1.09 (0.55), 159 |
| Layer temperature rise | 0.34 m | 1.05 (0.40), 169 | 1.04 (0.40), 169 |
| Layer depth | 0.20 m | 1.02 (0.19), 67 | 1.02 (0.19), 67 |
| Layer depth | 0.27 m | 1.08 (0.15), 169 | 1.07 (0.15), 159 |
| Layer depth | 0.34 m | 1.16 (0.18), 169 | 1.16 (0.18), 169 |

The layer depth on 0.27 m is calibrated on 6ee8c93 as pre-registered too: 5 of 5 series in band.

**What changed between the runs, and why.**
- **Bit-identical.** Steckler (165 runs), NBS (9) and NIST/NRC (19 of 20) repeat exactly: their burners never ran short of oxygen.
- **Changed a little.** NIST/NRC test 13 on 0.27 m cells (2.3 MW, door shut) moved by up to 5.8 % per pair. Nine of WTC's 12 runs moved by up to 5 %.
- **Changed most.** ATF: 7 of its 12 runs, by up to 16 % on 0.27 m and 29 % on 0.34 m, and its two stopped runs now finish.
- **The one verdict that moved.** The layer depth on 0.27 m went from indicative to calibrated.
  - On 0c62bd9, ATF's two stops put ATF outside the band for every quantity it measures on 0.27 m (12.6, rule 1). Plan 12.6 still gave calibrated (4 of 5 in band, pooled 1.07 (0.15)), but the protocol counted the stopped series as outside the indicative band and gave indicative.
  - On 6ee8c93 all five series are in band.
  - ATF's own depth barely moved (1.21 to 1.20): the verdict moved because the runs finished, not because their numbers changed.

**How independent the five are.** Counting NIST/NRC and WTC as one series, and setting Steckler aside as it shares NBS with the multi-room tests, the depth on 0.27 m still has three groups in band:
- NBS 1.10 (0.21);
- NIST/NRC and WTC merged, 0.98 (0.12), 72 pairs;
- ATF 1.20 (0.12), a laboratory other than NIST's.

They pool to 1.05 (0.16) over 114 pairs. ATF removes the weakness that 12.8 found.

**What the calibrated verdict covers.** The hot layer's depth, by Janssens and Tran's reduction of a thermocouple tree, on 0.27 m cells:
- **Fires.** Prescribed heat releases of 32 kW to 3.2 MW, D*/δx 0.9 to 5.6: ATF 1.1 to 2.7, the others as 12.8.
- **Enclosures.** Single rooms of 17 to 584 m³, a burn room off a 12.2 m corridor, and two 17 m corridors joined by a stairwell; ceilings 2.1 to 3.8 m.
- **Burners.** Each burner takes its test fuel and its FDS input's radiant fraction (12.9, amendments 3 and 8).
- **Openings.** Doors at least about a cell wide.
- **Walls.** Lightweight boards with gypsum, concrete or fire-brick floors.
- **Times.** Up to 600 to 900 s, or Steckler's steady state.

On 0.20 and 0.34 m cells the depth is indicative.

**What the rest says.**
- **The layer temperature on 0.27 m.** Pooled it reaches only σ̃M 0.61, so plan 12.6 says not calibrated, which is the verdict. Three series are in band; NBS (1.08 (0.70)) and ATF (1.66 (1.32)) scatter widely.
  - ATF's scatter has a shape, tree by tree as a share of the measured rise on 0.27 m cells:
    - the lower corridor's tree A, 2 m from the burner: 1.9 to 2.6;
    - tree B: 1.3 to 1.8;
    - tree C at the foot of the stairs: 0.7 to 1.2;
    - the upper corridor's tree F: 0.1 to 1.3;
    - tree G, near its open end: 0.03 to 0.5.
  - The model keeps the heat downstairs and lets too little reach the upper corridor. On these cells the corridors' ceilings snap to 2.16 m (2.4 m nominal), and the ceiling thermocouples move up to 0.36 m to a connected cell.
  - On 0.34 m cells the temperature is indicative, with the same two series outside.
- **The ceiling jet.** Not calibrated on 0.27 m (pooled σ̃M 0.75, from ATF) and indicative on 0.34 m. NIST/NRC and WTC are in band on 0.27 m.
- **O2 and CO2.** Not calibrated. NIST/NRC is in band or indicative, but WTC runs 1.7 to 2.3 times the measured, which the test fuel's yields (12.9, amendment 3) did not change.
- **Smoke obscuration.** It reads 1.76 times the measured with σ̃M 0.37 on both cells, over 10 tests, so visibility (3/K) would read short. It has one dataset, so it is not calibrated whatever its value.
- **The door.** Steckler's peak outflow stays a third low. The neutral plane is in band on every cell, but has one dataset.

**Sprinkler activation, reported apart.** The walls' convection coefficient was set on these tests, so they are not fully held out (claim protocol, section 4). `benchmark/results/2026-11-12-sprinkler-o2-stats.txt` compares the first head to open with the first measured opening on 6ee8c93, with σ̃E taken as 0, so σ̃M can only overstate. All are within the calibrated band:

| Series | Tests | δ | σ̃M |
| --- | --- | --- | --- |
| Vettori, flat ceiling (1998) | 9 | 1.165 | 0.118 |
| Vettori, sloped set (2003) | 8 | 0.939 | 0.228 |
| Bittern (2004) | 8 | 1.01 | 0.051 |
| UL/NFPRF series I | 4 | 0.977 | 0.046 |
| Pooled | 29 | 1.033 | 0.163 |

**A record fault in the harness, found while writing this section.** For ATF's five steady tests, each run's `burner.expectedMJ` in both files is low by 15 to 19 times. The harness's `burnerIntegral` integrates a heat release table only up to its last row, while the burner holds the last value, and those ramps end at 60 s (the pulsed test's runs to the end). Checked against the table held at its end, every ATF run released exactly its heat. No comparison uses the field. It is to be fixed in the harness before the next run; the two files are left as they are.

**Costs** (6 jobs; median solver time a run on 6ee8c93):
- NIST/NRC: 38 min on 0.27 m, 8 min on 0.34 m;
- WTC: 15 and 5 min;
- ATF: 9 and 4 min;
- NBS: 19, 5 and 1.5 min on the three cells;
- Steckler: 39, 16 and 5 s.

The whole suite took 124 min.
## 13. Validating the rest (pre-registered 2026-11-12, before any run below)

At the user's request to validate every fire quantity. Written after the results of 12.10 and the data inventory ([fire-validation-data-inventory-2026-11-12.md](fire-validation-data-inventory-2026-11-12.md)), before any comparison below is run. Nothing here may change after a result below is seen; anything added later is dated, with its commit.

### 13.1 What the public data allow, before any result

Three or more independent laboratories (NBS and NIST count as one) measure:
- the layer's temperature and depth;
- the ceiling jet;
- door and vent velocity;
- gas temperature at head height.

They do not for O2, CO2, CO, the neutral plane, smoke obscuration, held-out sprinkler activation or detectors. Wall heat flux is measured, but the solver has no gauge for it. PRISME is not counted: its reuse needs the NEA's approval.

Section 12.6 counts test series, not laboratories (NIST/NRC and WTC are two series from one laboratory). That rule stays the rule that decides, so the verdicts stay comparable with 12.8 and 12.10. Beside each verdict the suite reports how many laboratories its in-band series come from. Where all come from one laboratory, the app says "from one laboratory's tests" with the verdict.

### 13.2 New comparisons

Each uses the test's own FDS input for its fire, radiant fraction, yields and walls (12.9, amendment 8). Each series that lacks an FDS input is named with how its scene is built.

1. **Gas temperature at head height:** a new quantity, `headTemperature`, with σ̃E 0.07 (the guide's gas temperature).
   - Devices: every thermocouple between 1.5 and 2.0 m above the floor in the five wired series' trees (Steckler 1.54 to 1.77 m, NBS 1.57 and 1.88 m, NIST/NRC 1.75 m, WTC 1.53 and 1.83 m, ATF 1.76 m), outside the burner's plume.
   - Reduction: the rise above ambient at each device, the peak over the test, as the guide compares point temperatures.
   - It needs a new reducer only, on the stored runs.
2. **0.20 m cells for NIST/NRC, WTC and ATF**, skipped before for time: about 4 h at 6 jobs.
3. **UL FSRI 2018 compartments** (UL, MIT licence), a new independent laboratory.
   - Tests: all burner tests whose door state and burner position the scene can hold.
   - Quantities: layer temperature and depth (four trees, Janssens and Tran), ceiling jet (top thermocouples, the plume quadrant left out), O2 (two heights), doorway velocity (seven probes, peak outflow and peak inflow), neutral plane (from the doorway profile), head-height temperature (1.5 and 1.8 m).
   - The scene is a box built from the test report's dimensions and wall materials, because no FDS input exists.
4. **WTC opening probes:** velocity at the inflow and outflow openings (8 probes each). WTC has no neutral plane: each opening carries flow one way.
5. **NIST/NRC door probes:** door velocity, neutral plane and door mass flow.
6. **UL/NIST vents** (NIST at UL's facility): layer, ceiling jet, door and vent velocity, and neutral plane, in the first interval only, before any vent changes.
7. **NIST DelCo trainers:** layer, ceiling jet, O2, CO2, door velocity and neutral plane.
8. **NIST FSE 2008:** layer, O2, CO2 and soot mass fraction.

Left out:
- NIST smoke alarms: neither the fire nor the soot yield was measured, so they cannot test transport;
- FM/SNL and LLNL: a fan is needed, and their terms are not stated;
- FAA cargo: too small for the cells;
- PRISME;
- any series whose terms are not stated.

### 13.3 Statistics

As 12.6, with one addition. The FDS Validation Guide and NUREG-1824 Supplement 1 weight the pairs over 10 bins of the measured value. The suite reports σ̃M and δ both unweighted (as 12.6) and weighted, and the stricter verdict decides. Bands, the stop rule, the sealed-door rule and the two verdict columns are as before.

### 13.4 Physics changes planned before the reruns

Each change is made only if its check, which uses no measured data, passes. Each is to be on the CPU and the GPU with parity, before the final rerun.

1. **A one-dimensional wall:** conduction normal to each wall face in sub-cell layers, for walls thinner than a cell or with a thin lining (the tests' gypsum boards are lumped into whole cells today).
   - Check: a slab heated on one face against the exact series solution, and a semi-infinite solid against erfc, both to 2 %.
2. **Soot deposition on surfaces:** thermophoretic and gravitational, as FDS models them.
   - Check: the deposition velocity on a cold wall in a hot gas against the analytic thermophoretic velocity, to 5 %.
   - The mass-specific extinction coefficient stays 8.7 m²/g (Mulholland and Croarkin) unless a test measured its own.
3. **Openings narrower than a cell:** each opening's area kept as a face fraction in the projection.
   - Check: the exchange flow through a door between a hot and a cold sealed box against the orifice formula (discharge coefficient 0.68), and the neutral plane against the hydrostatic prediction, both to 10 %.

A change that fails its check is not made, and that is reported.

### 13.5 Order and reporting

1. **13.A, on the current solver (d3c5751):** items 1 and 2 of 13.2 first, then 3 to 8 as their reducers are built. Each verdict is reported as it comes, both ways.
2. **13.B, on the solver with the changes of 13.4:** every comparison of section 12 and 13 again. Both 13.A and 13.B are reported; the app states 13.B's verdict, with 13.A beside it.

The search for faults stays as section 5 of the claim protocol says: a fault found while comparing is fixed only if a check that uses no measured data shows it, and every comparison is rerun after. A quantity that still falls short is stated as it is.

### 13.6 Added while carrying out 13.A (2026-11-12, 12:15, before any comparison they govern)

These are dated additions. None changes a criterion of 13.1 to 13.5.

1. **The burner's energy record.** `burnerIntegral` now holds a table's last value past its last row, as the burner does. ATF's tables end at 60 s, so its `expectedMJ` was the energy to 60 s only (12.10). A 90 s check on ATF at 0.34 m gives released = expected in all six tests. The physics is unchanged.
2. **The harness is snapshotted.** At the start of a run, the suite copies itself and `lib/*.mjs` beside the solver's copy, and the workers run from the copies. Before, each worker loaded the live harness, so an edit made during a run could reach that run's later tests. The harness digests are taken at the start.
3. **Head height: which devices.** The rule of 13.2 item 1 is "every thermocouple between 1.5 and 2.0 m above the floor". Its list in brackets is short of that rule; the rule governs. The rule adds:
   - Steckler 1.88 and 2.00 m;
   - NBS trees 3, 4, 5 and 8 at 1.52, 1.83 and 1.93 m;
   - ATF's upper-corridor trees F and G at 1.79 m over their floor.
   That makes 414 device readings over all tests.
4. **Head height: the plume.** A device is in the burner's plume when r < D/2 + 0.25 (z − z_burner). Here r is its horizontal distance from the burner's centre, D = √(4A/π), and z its height. The 0.25 is about twice Heskestad's spread of the temperature half-width, 0.12 (z − z0) at Tc/Ta near 1, so the cone holds the plume's edge. A device with a floor between it and the burner is not in the plume. The rule uses geometry only. It leaves out one device: NIST/NRC test 15, tree 5 at 1.75 m (r 0.70 m, reach 0.94 m). ATF tree G stands 0.02 m from the burner's axis but one floor up, so it stays.
5. **Head height: the model.** The model reads the tree's open cell nearest the device's height over the column's model floor. Its value at a measured time is the mean of its 1 s readings within half the measured interval, as for the point devices of section 12. The runs made before this keep no such series, so the head-height verdict comes from runs that keep them (`series.head` in each run). `tools/calibration/head-temperature.mjs` reduces those runs.
6. **The head-height reducer's check (no model).** The reducer is fed each device's measured series as if it were the model, each second taking the reading nearest it. It returns δ = 1 and σ̃M = 0 in every series (`head-temperature.mjs --check`).
7. **Two features of the measured data,** found by the same check (pre-ignition readings against the FDS input's ambient). Both are kept as measured, as the FDS Validation Guide keeps them:
   - In WTC tests 1 and 2, the fire was lit about 20 s before the file's time zero. The heat release is 173 to 310 kW at −10 s and 1.0 to 1.4 MW at 0 s, and head-height readings at t ≤ 0 are 25 to 72 K over the ambient. The FDS inputs ramp from t = 0.
   - In NBS, the burn room's tree read 21 to 51 K over the FDS input's ambient before ignition, at 1.57 and 1.88 m. Its layer temperature reads the same.
8. **The weighted statistics (13.3)** follow the FDS plotting code (`fdsplotlib.py`, its bin-weighted branch). Ten bins split the measured values from 0 to their largest, and each pair is weighted by n over its bin's count. σ̃E is capped at u/√2. A port in Python and the suite agree to three decimals on four random sets. The guide also drops pairs outside its plots' ranges; the suite has no plot ranges and keeps every positive pair. "The stricter verdict decides" is read as the strictest of four: 12.6 and the protocol's section 4, each unweighted and weighted.
9. **The 0.20 m run** (13.2 item 2) also runs Steckler and NBS on 0.20 m cells, so that head height has all five series on that cell. Its output is `benchmark/results/2026-11-12-validation-13a-020.{json,log}`.
10. **13.A's solver is pinned to git (13:30).** 13.B's changes are to be merged into this worktree, so 13.A runs started after 13:30 take the solver from git, `--solver-rev d3c5751`: every `src/core/*.js` at that commit. That includes the voxeliser the sealed-door rule uses.
    - The two runs started before then, `-13a-020` and `-13a-034`, took the working tree. Their snapshots match d3c5751 in all 13 modules the suite reaches (`benchmark/results/2026-11-12-validation-13a-solver-pin.txt`). Seven of the 13 differ from git's blobs only in line endings: the working tree is CRLF, the blobs LF.
    - Results files now record `solverRev` and each module's digest with and without that normalisation.
11. **Wall layers stay lumped in 13.A.** The suite passes each wall material to the solver as lumped values only:
    - density: the layers' areal mass over one cell, ρ = Σρ_i L_i / h;
    - specific heat: the layers' areal heat capacity over their areal mass;
    - conductivity and emissivity: the exposed layer's.
    For 13.B, `--wall-layers` also passes the layers themselves (thickness, ρ, c, k, exposed layer first, as `lib/fds.mjs` returns them). 13.A runs without it.

### 13.7 Added while building items 3 to 8 (2026-11-12, 13:00, before any of their comparisons)

These are dated additions, made before any model comparison of the quantities they govern.

1. **One quantity for door and vent velocity.** `doorVelocity` now holds Steckler's door (as before) and the opening probes of WTC, NIST/NRC, UL FSRI and UL/NIST. Each probe's value is its largest speed through the opening over the window, either way: the guide's "maxabs" for the WTC probes. σ̃E is 0.08, the guide's gas velocity.
2. **WTC opening probes (item 4).** These are the probes the guide compares, each over its own row's window: inflow 1, 2, 3, 6, 7 and 8, and outflow 1 to 8 (1 to 10 in tests 2 to 6). That gives 93 pairs a cell. In test 1 the guide leaves out inflow probes 4 and 5. A check (`probe-checks.mjs`) shows why: their thermocouples read the hot layer's 230 to 280 °C while the probes read inflow. They stay out.
3. **NIST/NRC door (item 5).** This uses the five open-door tests (3, 9, 14, 15 and 18) and their 14 recorded probes (`BP Door-12` is not in the file). Three quantities are compared:
   - each probe's velocity;
   - the neutral plane of each of the three probe columns, on the mean profile over 300 s to the run's end (the layer depth's window);
   - the door's largest inflow and outflow (kg/s, from file 1). Its σ̃E is 0.08, the velocity's; the guide gives none for a mass flow, so this is this project's assumption.
   Check: the file's door mass flows, divided by the probes' volume flows, give densities of 1.16 to 1.21 kg/m³ in (19 to 32 °C) and 0.80 to 0.91 kg/m³ out (113 to 169 °C), in all five tests.
4. **The door-probe rule.** A single column of door probes is compared only when it carries about as much mass out as in. The mass comes from the probes and their paired thermocouples (ρ = 353/T), each probe over its share of the door. Out over in, over the compared window, must lie within 1/3 to 3. Otherwise that test's door velocities and neutral plane are left out. The rule leaves out:
   - UL FSRI: 10 of the 46 open-door tests, with out/in 3.0 to 18.7;
   - UL/NIST: tests 1 (19.9) and 3 (3.2).
   NIST/NRC's three columns are checked against the file's own door flows instead (item 3).
5. **UL FSRI 2018 (item 3).** No FDS input exists, so the scene is a box built from the data's README and Willi et al. (2023):
   - the room: 3.66 × 3.66 × 2.44 m;
   - the door: 0.9 × 2.0 m in the front wall, 2.44 to 3.34 m from the left wall;
   - walls and ceiling: 13 mm marinite on 16 mm type X gypsum, with properties from NIST's WTC input;
   - the floor: concrete, with ATF's input's properties (the 13 mm cement board is not matched);
   - the burner: a 0.6 m or 0.3 m square, its top 0.5 m up on a solid block (the steel frame, simplified), against the walls as the README draws it.
   The fire: natural gas taken as methane, with FDS's default radiant fraction (0.20) and no soot, at the set heat release from ignition after a 10 s rise.
   - **The run:** from ignition to the first of Burner Off, a door or fan event, the file's end and 900 s. This leaves 91 of the 92 burner tests. 500kW_Closed_Center_1 is left out: its door was opened and shut at 12 s.
   - **The ambient:** the trees' mean over the 60 s before ignition.
   - **The data:** measured channels are kept as 5 s means, and the model's 1 s readings are averaged over the same 5 s.
   - **Quantities:** the layer by Janssens and Tran at four trees, with the depth's window from 120 s. The ceiling jet: each top thermocouple in a quadrant the burner's footprint does not overlap (none for the centre burner). O2 at the eight analysers. Door velocity and the neutral plane in the open-door tests, the plane on the mean over the run's last 180 s. Head-height temperature at 1.54 and 1.84 m.
   - **Sealed by design:** the closed-door tests (no leakage measured).
   - **The door on the grid:** its modelled area is 1.20, 1.53 and 0.58 m² of 1.80 m² on 0.20, 0.27 and 0.34 m cells.
   - **Checks without a model:** O2 reads 0.2087 to 0.2098 before ignition. The reducers return the measured values exactly. The doorway's centreline column carries a median 2.05 times as much mass out as in, a three-dimensional inflow that one column cannot close.
6. **UL/NIST vents (item 6).** Each fire's first interval only, the run ending at its end (1215, 570, 450 and 525 s). Both ceiling vents are shut throughout: the FDS inputs' lids are present until their controls change. The heat release follows the FDS ramp. Each quantity is compared over the guide's window, the interval's last 60 s:
   - the layer at two trees;
   - the guide's two ceiling-jet thermocouples;
   - head height at 1.5 and 1.8 m;
   - the door's ten probes and their neutral plane, in tests 2 and 4 only (item 4).
   There is no vent velocity, since the vents stay shut in the first interval.
7. **DelCo (item 7) is not built.** Its tests light three propane pans at 0, 180 and 360 s. They open a door slit at 2 s, a roof cavity at 128 s and doors from 420 to 600 s. The harness holds one burner and fixed openings, and the solver cannot change openings in time; that would be a solver change, which this work may not make. The longest fixed interval, 2 to 128 s with one pan and a 0.1 m slit, is too short to form a layer.
8. **FSE 2008 (item 8)** is limited by its data and its doors.
   - The repository holds only the guide's reduced layer for it (`_HGL.csv`), not the trees, so the layer cannot be reduced by the suite's own code and is not compared.
   - Of its 25 inputs, ISONG3 has a 0.8 m door and ISOHept25 and 26 have 0.4 m doors. The rest have 0.1 to 0.2 m doors, which the grid seals (the sealed-door rule), and are not built.
   - The three are run to the end of the guide's first window (580, 580 and 600 s) and compared over it. The quantities: O2 and CO2 at the front and rear sample points (dry, as measured), their thermocouples as the guide's ceiling jet, and the soot mass fraction where measured (ISOHept25 front, ISOHept26 front). Soot's σ̃E is 0.12, the smoke's; the guide gives none, so this is this project's assumption.
   - The fire: ISONG3's burner is the input's fuel mass flux times its 1 m² area at 50.0 MJ/kg. Check: that gives 755 kW in the first step, and the measured heat release reads 755 kW. The heptane tests' spray is the nozzle's flow at 684 kg/m³ and 44.6 MJ/kg onto the 0.36 m² pan (1.0 and 2.0 MW at the windows).
   - Both heptane tests are under-ventilated: the measured O2 falls to 9.5 % and 0.7 %. This solver releases the prescribed heat inside the room.

9. **A harness fault on UL FSRI's back-wall burner (16:30, after the 0.34 m run).** Six runs stopped with "no open cell over the burner": `100kWSB_{Open,Closed}_Back_{1,2,3}` on 0.34 m cells. The 0.3 m burner stands against the back wall. On 0.34 m cells the cell that holds its centre is mostly wall, and the voxeliser fills it.
   - **The check (geometry only, no measured data):** every test of every dataset on all three cells. Only these six runs lack an open cell over the burner's centre.
   - **The fix:** `lib/fds.mjs burnerBase` takes the open column nearest the burner's centre on its footprint, grown by half a cell, when the centre's column is solid. Here that moves the base 0.283 m. The suite and the sealed-door rule both use it, and each run records `burner.baseMoved`.
   - **The rerun:** no other run takes the new path, so only these six are rerun (`-13a-new-034-burnerfix`).

### 13.8 What 13.4's checks found, and two fixes added (2026-11-12, 13:21, commit 37afd3f, before any 13.B run)

The physics work of 13.4 was done blind: in its own worktree, by an agent that opened no validation file or result.
- **Item 1, the one-dimensional wall: made.** A slab against the exact series solution 0.41 %, a semi-infinite solid against erfc 0.80 % (2 % allowed). CPU and GPU agree. It is used where a material lists its layers, as the tests' boards do; the app's stone lists none, so app runs do not change.
- **Item 2, soot deposition: made, on by default as in FDS.** Thermophoretic and gravitational, after Floyd, Overholt and Ezekoye (2014). The deposition velocity on a cold wall matches Talbot's within 0.05 % (5 % allowed), and gas plus deposited soot stays within 0.1 % in a sealed box. CPU and GPU agree.
- **Item 3, openings narrower than a cell: not made.** Its check failed, and with whole-cell doors too. The exchange flow through a door between a hot and a cold box reads 13 to 19 % above the orifice formula at 600 K (within 5 % at 400 K). The cause found: the projection divides the pressure gradient by the ambient density, not each face's own.

Two faults found by these checks, which use no measured data, are added as 13.4 items 4 and 5. Each is made only if its check passes.

4. **The projection with each face's own density** (the low-Mach form, ∇·(∇p/ρ)).
   - Check: the door exchange of item 3's check within 10 % of the orifice formula (discharge coefficient 0.68) at 400 K and 600 K, and the neutral plane within 10 % of the hydrostatic prediction.
   - Item 3 is tried again after it.
5. **A pressure for each sealed region**, as FDS's pressure zones: a sealed region's background pressure follows its mass and heat, so a region cooled by its walls loses pressure instead of drawing in gas. Today a sealed box cooled by its walls gains 31 % of its mass in 60 s.
   - Check: a sealed box keeps its mass within 1 % while its walls cool it.
   - Check: its pressure rise when heated matches the ideal gas law within 2 %.

13.B runs on the solver with every change that passed. 13.A stays on d3c5751's solver, snapshotted from git.

### 13.9 Item 4's outcome, and a sixth item (2026-11-12, before any 13.B run)

- **Item 4, the projection with each face's own density: not made.** It brought the 600 K door exchange within 8 % of the orifice formula. But it failed its check in three of four cases: a door two cells wide flowed 8 to 16 % short, and the neutral plane sat 10 to 13 % high in boxes that had already layered. The check is not changed after its result. A check for layered rooms would have to be written and committed before any new attempt. The prototype is kept on a side branch.
- **Item 6, the flame put out where it cannot burn** (added after 13.A's 0.34 m results).
  - In those results, closed FSRI tests at 250 and 500 kW, whose fires went out in the test for lack of oxygen, read 1.5 to 1.9 times hot.
  - Today a burner cell burns its fuel until the cell's oxygen is gone, but a real flame goes out long before that.
  - The change: the burner's fuel, and the gas reaction, burn in a cell only where the mixture there could reach the critical flame temperature. That is FDS's extinction model, with FDS's default value, read from FDS's source and Technical Reference Guide, not fitted.
  - Check, which uses no measured data: a burner in a sealed box stops releasing heat when the box's oxygen falls to the volume fraction the critical flame temperature implies for its fuel (the analytic value), within 0.5 % oxygen by volume. The unit tests and closures must still hold.

  This item was prompted by held-out results, as the burner fault of 2026-11-12 was. Its value is FDS's, and its check uses no measured data. Both runs, 13.A and 13.B, are reported.

### 13.10 Items 5 and 6 (2026-11-12, before any 13.B run)

- **Item 5, a pressure for each sealed region: made** (FDS's pressure zones). A sealed box cooled by its walls keeps 1.00000 of its gas mass over 60 s (1 % allowed; it had gained 29.7 %), and its pressure falls to 0.736 of ambient. A heated sealed box rises 35.438 kPa against the ideal gas's 35.431 kPa (0.02 %; 2 % allowed). CPU and GPU agree, and open scenes are bit-identical to before.
- **Item 6, flame extinction: its check failed.** FDS's extinction model 2 was built with FDS's default critical flame temperatures (methane 1507 °C), from FDS master 5f8f194.
  - The pre-registered check (a 20 kW methane burner in a sealed box whose walls cool it) stopped with the box at 10.79 % oxygen against an analytic 10.13 %, a gap of 0.66 % where 0.5 % was allowed.
  - Diagnosis: per cell the model goes out at its own threshold (10.45 % against 10.46 % in the last cell to burn). But the walls cool the box unevenly, so cooler cells stop first and keep their oxygen. The analytic value was evaluated for a mixture the box did not hold.
  - The failed check stands as the record and is not re-scored.
- **Item 6's second check, written now, before it is run.** Both parts must pass for item 6 to be made; the critical flame temperature stays FDS's default and is not fitted.
  - (a) The same burner in a sealed box whose walls exchange no heat (a uniform box), so the box holds the mixture the analytic value is computed for. The box's mean oxygen at the stop must be within 0.5 % by volume of the analytic value, the same tolerance as before.
  - (b) At the stop, every cell that held fuel and burned must lie within 0.1 % oxygen by volume of its own threshold, computed from its own temperature and composition.

  This second check was written after the first one failed. A reader should weigh item 6 with that in mind.

### 13.11 Item 7, leakage from sealed regions (2026-11-12, before any 13.B run)

Item 5 gives each sealed region its own pressure. Without a way out, a region heats at constant volume: its pressure climbs (35 kPa for 2 MJ in 22.5 m³) and its gas runs up to γ ≈ 1.4 times hotter than at constant pressure. Real rooms leak, through door gaps and the fabric, so their pressure relieves within seconds. FDS pairs its pressure zones with leak areas, and the NIST/NRC closed-door tests' FDS inputs give theirs. Item 5 is therefore used in 13.B only together with item 7.

7. **Leakage between a sealed region and the outside,** as FDS's leakage model: a flow C_d A_L √(2 |Δp| / ρ), with FDS's discharge coefficient, from each sealed region at its pressure to ambient.
   - In the validation, each test's leak area is read from its FDS input (`&ZONE LEAK_AREA`, or `&SURF LEAK_PATH` areas). A test whose input gives none has none.
   - In the app, a default leak area per region from published construction tightness (typical walls and doors, from ASHRAE or NFPA 92 tables, cited), with a setting to change it.
   - Checks, which use no measured data:
     - a sealed box heated at a constant rate with leak area A_L reaches the steady pressure the orifice relation gives for that heat release, within 5 %;
     - with A_L = 0 the box behaves as item 5 alone;
     - mass and energy, including what leaves through the leak, close within 1 %.
   - CPU and GPU parity.

### 13.12 Trying a failed item again (2026-11-12, at the user's request, before any further attempt)

The user asked that a physics item that fails its check be tried again, or a better solution found. The rule for every further attempt:
1. **A new attempt changes the method, never the tolerance.** A check may be rewritten only where it computes its reference wrongly for the case it runs, and the rewrite is disclosed as written after a failure.
2. **Its method and check are committed to this plan before it runs.** The physics agent drafts them blind; the coordinator commits them; then the agent runs.
3. **Every attempt is reported,** failed ones included, with its numbers and commit. An item is made only on an attempt that passes.
4. **No value is fitted to make a check pass.** Constants come from their cited sources (FDS's defaults, published coefficients), and none from a test that 12 or 13 holds out. For example, Steckler's own discharge coefficient may not be used for doors, because Steckler's tests are held out.
5. **13.B runs on the items made by the time it starts.** An item made later gets its own full rerun, reported beside 13.B.

Open for another attempt now: item 4 (door flow on the grid) together with item 3 (openings narrower than a cell), and item 6 (extinction), whose second check is written in 13.10.

### 13.13 Items 6 and 7, and the next attempts at 6 and at 4 with 3 (2026-11-12, before any run below)

- **Item 7, leakage: made.** FDS's leak relation (master 5f8f194), ṁ = C_d A (Δp/Δp_ref)^(n−½) √(2ρ|Δp|), with each test's leak from its FDS input. The app's default is average construction from Klote, Nelson, Deal and Levin, NISTIR 4770 (1992), Table 1: walls 0.21 × 10⁻³ of their area, floors 0.52 × 10⁻⁴.
  - Steady pressure 224.7 Pa against the orifice relation's 224.0 Pa (0.32 %; 5 % allowed).
  - Mass and energy close to 1.00000, with 52.6 % of the gas let out.
  - With A_L = 0 it reproduces item 5 exactly. CPU and GPU agree.
- **Item 6, second check: passed, but it could not fail, so it is not counted.** In the box whose walls exchange no heat, the gas rose past the critical flame temperature and the model was never exercised: the run is bit-identical with extinction off. Not made.
- **Item 6, attempt 3** (rule 1 of 13.12: the first check computed its reference for a mixture the box did not hold; disclosed as written after two failures).
  - Case: the first check's own: a 20 kW methane burner in a sealed box whose walls cool it, leakage off.
  - Reference: each cell's threshold, from its own temperature and composition, at the stop.
  - Passes if (a) the gas-mass-weighted mean oxygen of the box at the stop is within 0.5 % by volume of the gas-mass-weighted mean of the cells' thresholds; (b) every cell that burned is within 0.1 % of its own threshold; and (c) the run differs from the same run with extinction off, so that it is exercised.
- **Items 4 and 3, attempt 1: method.**
  - Each face's own density in the projection (u −= dt Δp / (ρ_f h), ρ_f the mean of its two cells; Poisson links weighted ρ₀ / ρ_f with the porous weights).
  - Each partly open face's open share (FAVOR) in the divergence, Poisson, transport and diffusion, found by the voxeliser's rays (a geometry unit test: a 0.5 m door on 0.2 m cells is two cells and a half).
  - No discharge coefficient in the solver.
- **Items 4 and 3, attempt 1: check.**
  - Six cases: two sealed boxes 3.0 × 2.4 × 3.0 m on 0.2 m cells; a door 2.0 m high and 0.4, 0.5 or 0.6 m wide; the hot box at 400 K or 600 K. Radiation, wall exchange and deposition off; no leakage; read every 0.05 s from 2 to 6 s.
  - The reference at each read is Emmons' vent integral on each box's own layered density profile (the slab form CFAST uses), measured at least 0.4 m from the dividing wall:
    - Δp(z) = Δp₀ − g ∫ (ρ_hot − ρ_cold) dz′, with Δp₀ from volume balance;
    - ṁ_out = C_d w ∫_{Δp>0} √(2 ρ_hot Δp) dz, with C_d = 0.68 from Prahl and Emmons (1975, Combustion and Flame 25:369), independent of Steckler;
    - the neutral plane is Δp's lowest zero.
  - Passes in all six cases if:
    - (a) the mean outflow is within 10 %;
    - (b) the mean neutral plane is within 10 %;
    - (c) the hot box's change of mass over 2 to 6 s equals the net flow through the door within 1 % of the mass exchanged.
  - Plus the closures, and parity once on the GPU.
  - Disclosed: this reference was drafted after item 4's first check found the boxes layered. It tests the door's flow law given the rooms' states, not how the rooms layer.
- **Items 4 and 3, attempt 2** (only if attempt 1 fails): a sub-grid vent for doors under three cells wide, coupled into the projection. Its full method and check are to be committed here before it runs.

### 13.14 13.A complete (2026-11-12)

Solver d3c5751 (pinned from git), walls lumped, no leakage, eight series on each cell size. Files: `benchmark/results/2026-11-12-validation-13a-020-all.json`, `-13a-027.json`, `-13a-034-all.json`. Each verdict is the strictest of four readings (12.6 and the protocol's section 4, each unweighted and weighted). The sealed-door rule changes none.

| Quantity | 0.20 m | 0.27 m | 0.34 m |
| --- | --- | --- | --- |
| Smoke layer depth | calibrated: 6 of 7 series, two laboratories, pooled 1.00 / 0.14 | calibrated: 7 of 7, three laboratories, 1.06 / 0.11 | calibrated: 4 of 7, two laboratories, 1.12 / 0.14 |
| Layer temperature | indicative: 2 of 7, one laboratory, 1.30 / 0.32 | not calibrated | not calibrated |
| Ceiling jet | indicative: 2 of 6 | not calibrated | not calibrated |
| Gas temperature at head height | indicative: 1 of 7 | not calibrated: 3 of 7, two laboratories | not calibrated: 3 of 7, one laboratory |
| Neutral plane | indicative: 2 of 3, two laboratories | indicative: 2 of 4, two laboratories | indicative: 1 of 4 |
| Door and vent velocity | not calibrated: 0 of 4 | not calibrated: 0 of 5 | not calibrated: 0 of 5 |
| CO2 | indicative, one laboratory | indicative, one laboratory | not calibrated |
| O2 | not calibrated | not calibrated | not calibrated |
| Smoke obscuration, soot | not calibrated | not calibrated | not calibrated |
| Door mass flow | not run | one series (NIST/NRC, 0.94 / 0.08) | one series |

What the series show:
- **The large NIST rooms read well, the small lined rooms read hot.** On 0.20 m the layer temperature is in band in NIST/NRC (0.93) and WTC (0.92), and reads 1.31 in Steckler, 1.38 in UL FSRI and 1.91 in UL/NIST; NBS reads 0.70 with a scatter of 0.65, and ATF scatters 0.56. This is a pattern, not a cause: phase 2 of the roadmap lists the candidates, and 13.B tests the first of them (the layered walls).
- **UL FSRI's door** is held as 0.58 of 1.80 m² on 0.34 m cells, 1.53 m² on 0.27 m and 1.20 m² on 0.20 m. Its verdicts follow that.
- **UL FSRI's O2 is not usable in the open-door tests:** the analysers 1.8 m below the ceiling read almost no drop, so the ratio of changes has no meaning there. It pulls the pooled O2 out of every band (3.92 on 0.20 m). To pre-register before 13.B's O2 is read: section 14.7.
- **UL FSRI's closed-door tests at 250 and 500 kW:** the measured fire went out, and the prescribed burner goes on. Extinction (item 6) bears on it.
- **FSE on 0.20 m:** ISONG3 stopped on the speed check at 76 s, so FSE is out of band there for all it measures.
- **DelCo is not built** (13.7, item 7).

### 13.15 A heat ledger in every run (2026-11-12, 22:40, before any 13.B run)

13.A showed the layer too hot in the small lined rooms and in band in the large NIST rooms (13.14). To see why, 13.B would have to be run again with more output, at 9 to 11 hours a time. So from now on the suite writes a heat ledger into each run's record, and 13.B itself gives the diagnosis.

About ten times a run, and at its end, `tools/calibration/validation-suite.mjs` records, in MJ: the heat the burner has released; the heat radiated; the heat held by the gas above ambient (the health guard's sum over the whole grid); the heat held by the inert solids (each cell's heat capacity by its mean rise, for lumped and layered walls alike); the heat carried out through the grid's sides; and the mass (kg) and heat leaked from sealed regions. A solver without one of these counts gives `null` for it.

The ledger is the model's own account. It is compared with no measurement, enters no statistic and decides no verdict. It reads the state after a step and changes none, and a failure inside it is caught, so it cannot stop a run.

Checked before use: Steckler tests 14 and 410 on 0.34 m, on 13.A's solver (d3c5751), gave pairs identical to those in `benchmark/results/2026-11-12-validation-13a-034.json`. In test 14 at 300 s, of 18.55 MJ released the gas held 1.24, the walls 5.76 and 10.67 had left through the grid's sides; 0.89 MJ (5 %) is not in these three.

What I will read from it after 13.B, named now: for each series, the walls' share of the heat released over the steady period, against the share that the McCaffrey, Quintiere and Harkleroad correlation implies for the same room (as `tools/calibration/energy-budget.mjs` computes it for the stone room). This is a diagnosis. Whatever it shows, a change to the solver follows 13.12: stated and checked blind before it runs, and fitted on no held-out series (14.1).

### 13.16 A departure from 13.11 in 13.B, and its correction (2026-11-13, 03:55, after 13.B's 0.34 m and 0.27 m files were read, before any corrected run)

13.11 says that in the validation a test whose FDS input gives no leak has none. 13.B's harness did not do that. With `--leakage` on, `tools/calibration/validation-suite.mjs` passed the solver a leak list only where the input states a leak. Where it states none, the option was left out, and the solver then gave each sealed region the app's default construction leak: 0.21e-3 of the area of its upright faces and 0.52e-4 of its flat ones, with a discharge coefficient of 1. The mistake is in the harness. The solver does what its default says.

It was found on 2026-11-13 at about 03:30, while the heat ledger (13.15) of the 0.34 m and 0.27 m files was read: runs with no leak in their record show mass leaked. The 0.20 m cell was running on the same harness at the time and has the same departure.

Runs affected (no leak stated, a sealed region on the grid, mass leaked):
- **0.34 m, 52 of 178:** the 45 UL FSRI closed-door runs (net 12.1 to 20.8 kg out, median 15.1), NBS 100O (22.4 kg), and Steckler 10, 11, 114, 144, 514 and 544 (8.9 to 10.0 kg). The grid seals these Steckler tests' openings, so the representation rule already leaves them out of every statistic.
- **0.27 m, 49 of 178:** the same 45 (13.8 to 22.9 kg, median 16.1), NBS 100O (22.1 kg), and Steckler 10, 114 and 514.
- **0.20 m:** listed by the same rule when its file lands.

The 11 runs on each cell whose input states a leak (NIST/NRC) ran as 13.11 says. The other runs have no sealed region and leaked nothing.

The correction, fixed now:
1. **With `--leakage`, a test whose input states no leak is passed an empty leak list,** which the solver takes as no leak. Nothing else in the harness changes. The solver stays at 550fce5 with 13.B's flags.
2. **The affected runs are run again on each cell and merged over the first files** (`--merge`). The merged files are 13.B as 13.11 pre-registered it, called **13.B sealed** below. The first files stay in the repository as they ran, called **13.B default leak**.
3. **Both readings are reported** for every quantity and series. Where their words differ, the stricter word stands, because the choice between them comes after results were seen. The app states that word.
4. **A control.** The rerun on 0.27 m includes Steckler 11, 144 and 544, which have no sealed region on that cell. Their pairs must be identical to the first file's. If they are not, the correction changed more than it says, and nothing is merged until that is understood.
5. **A sealed run that stops, or fails the health guard, is reported as stopped,** and 12.6 counts it as it counts any stopped run. A room with no leak heats at constant volume (13.11), so high pressures are expected in the closed rooms. That is the pre-registered reading and not a fault to fit away.

What this does not settle. The UL FSRI closed rooms leaked by an amount that was not measured, and NBS 100O's input states no leak. Neither reading is the room as tested: one has no leak, the other a typical construction's. No leak area is chosen for these tests after their results are seen. If a later item gives sealed-by-design tests a leak, its source is cited and it is pre-registered under 13.12 before it runs.

### 13.17 Items 4 and 3, attempt 2: a sub-grid vent (2026-11-13, before any run of it)

Drafted blind to every validation result, as 13.12 sets. Nothing below changes after a run of it; anything added later is dated, with its commit.

#### What failed before

Attempt 1 (13.13; prototype 97d5be9 on `flamescapes/door-attempt1`, CPU only) divided each face's pressure gradient by the face's own density and kept each partly open face's open share. Its check, as 13.13 fixed it:

| Door | Hot box | (a) outflow (10 % allowed) | (b) neutral plane (10 %) | (c) mass (1 %) |
|---|---|---|---|---|
| 0.4 m | 400 K | −17.8 % | −4.7 % | −4.11 % |
| 0.5 m | 400 K | −7.3 % | −8.6 % | +1.98 % |
| 0.6 m | 400 K | −5.7 % | −13.8 % | +1.88 % |
| 0.4 m | 600 K | −14.1 % | −8.2 % | −3.90 % |
| 0.5 m | 600 K | −6.8 % | −9.3 % | −1.82 % |
| 0.6 m | 600 K | −10.4 % | −10.3 % | −1.21 % |

It failed all three parts: (a) at 0.4 m, and at 0.6 m and 600 K; (b) at 0.6 m; (c) in every case. It was not made.

Why (c) failed. Inside the hot box the transport's fluxes through its inner faces cancel. So the box's change of mass is the flux through its door faces plus two terms that are not fluxes (and any mass the transport's floor adds, counted in `stats.massFloor`):
- The transport's advective form (`transportGas`). Each cell takes its flux difference and gives back its starting density times its faces' divergence, ρ⁰_c (∇·u)_c V dt. Over the box this sum vanishes only where the projected flow is free of divergence in every cell. The projection stops at a tolerance (at most 30 sweeps), and ρ⁰ is layered, so the sum is not zero.
- The zone's rescaling (`zoneTransport`). The two boxes were one sealed region, so one pressure zone. After the transport, all the zone's densities are scaled by one factor back to its mass. That moves mass between the boxes in proportion to their mass, not through the door.

The face tally saw only the fluxes. The prototype's commit records the same two causes.

#### The method

An opening the grid holds poorly becomes a vent between the gas on its two sides, as a zone model treats a door (CFAST's vent flow): a two-way flow from the hydrostatic pressure difference across it, moved from the cells on one side to the cells on the other. The projection is unchanged. Item 4 (each face's own density) stays not made.

1. Which openings are vents. Each door and window is a rectangle in a wall one cell thick: its plane, its sill and soffit heights, and its two side edges. The scene lists them: in the validation, FDS's `&HOLE` through a wall; in the app, each door's and window's own bounds. An opening is a vent when either holds:
   - it spans fewer than four cells across its narrower side;
   - the grid's open area through it differs from its true area by more than a tenth.

   Any other opening stays on the grid as now. A vent's wall cells stay solid, so no grid face and no diffusion crosses it; the vent is its only path.

   Disclosed: 13.13 named "doors under three cells wide". Four is written here, after attempt 1, so that the check's 0.6 m door (three cells) is a vent too. On the grid that door failed (b) and (c) in attempt 1, and item 4 found doors two cells wide 8 to 16 % short. The tenth covers a door that is wide but off the cell faces: UL FSRI's 1.80 m² door is held as 0.58 to 1.53 m² on the three cells (13.7, item 5).

2. A vent's two sides, a and b, are the open cells against the wall on either side. Along its height it is cut at the grid's row boundaries into segments, from its true sill z_s to its true soffit z_t; a row the opening covers in part gives a segment in part. For each row r and each side:
   - the face cells: the side's cells against the wall that the opening covers, each weighted by its share of the opening's width;
   - the band: the side's open cells, in the same region, whose centres lie between one cell and 1.0 m from the wall's face, and within 1.0 m of the opening's edges along the wall.

   ρ_a,r and ρ_b,r are each side's mean density over its band in that row; with no band cell in a row, the face cells' mean. The band leaves out the column against the wall, which holds the vent's own jets. 1.0 m is about one door's width, the reach over which a doorway's approach flow is not hydrostatic. It is a modelling constant fixed here, taken from no data.

3. The pressure difference across the vent at height z, side a less side b, each against the ambient's hydrostatic pressure at that height:

   Δp(z) = Δp_s − g ∫_{z_s}^{z} (ρ_a(z′) − ρ_b(z′)) dz′,

   with ρ_a and ρ_b constant over each row. So Δp is linear on each segment.

4. The neutral planes are the zeros of Δp(z). On each segment a zero, if any, is found exactly by linear interpolation between the segment's ends, as CFAST's vent routine finds it (`ventw` in `flowhorizontal.f90`). The segments are cut there into slabs, each of one sign. A vent may have no plane, one or two; none is assumed. On a coarse grid the plane is exact for the rows' mean profile. How far that profile is from the rooms' is what (a), (b) and (f) below measure.

5. Each slab [z_1, z_2], with |Δp| going from p_1 to p_2, carries

   ṁ = C_d w (z_2 − z_1) (2/3) √(2 ρ_d) (p_1 + √(p_1 p_2) + p_2) / (√p_1 + √p_2),

   the exact integral of C_d w √(2 ρ_d Δp(z)) over the slab for a linear Δp (CFAST's slab form). w is the opening's true width. ρ_d is the donor's band density in that row: side a's where Δp > 0, side b's where Δp < 0. The vent's flow from a to b, ṁ_ab, is the sum over the slabs with Δp > 0, its flow back, ṁ_ba, the sum over the rest, and its net flow F = ṁ_ab − ṁ_ba.

6. The discharge coefficient is C_d = 0.68, from Prahl and Emmons (1975), "Fire induced flow through an opening", Combustion and Flame 25, 369 to 385, doi:10.1016/0010-2180(75)90109-1 (checked on Crossref). It is the value 13.13 fixed for the check's reference, independent of Steckler's tests (13.12, rule 4). It is not fitted.

7. The pressure difference at the sill, Δp_s = P_a(z_s) − P_b(z_s), each side's pressure against the ambient's at that height:
   - A side in a pressure zone (item 5): P_z(z_s) = p_∞ (π_z − 1) − g ∫_{y_z}^{z_s} (ρ̄_z(y) − ρ_0) dy. Here p_∞ = 101 325 Pa; π_z is the zone's pressure as a share of ambient; y_z is the floor of the zone's lowest open row; ρ̄_z is the zone's mean density per row over all its cells; ρ_0 is ambient air's.
   - A side in an open region: P_o(z_s) = p̄_o − g ∫_{y_o}^{z_s} (ρ_o(y) − ρ_0) dy. Here p̄_o is the mean of the projection's pressure (`sim.p`, Pa, the ambient's hydrostatics taken out) over the band's cells in the row that holds the sill, at their centre height y_o, from the step before; ρ_o is the band's density per row.
   - A vent whose two sides are both open regions, or one region, stays on the grid in this attempt, and each such opening is reported. The regions are found with every vent closed; a vent whose sides are both open or the same is opened again on the grid, and this is repeated until none changes.

8. The zones' pressures at the step's end, found together. A zone's pressure follows its mass at its present heat by item 5's relation, π_z(M) = (M / (π_z⁰^κ G_z))^{1/(1−κ)} (as `zoneLeak` takes it; G_z the mass its cells would hold at ambient pressure at their temperatures, κ = R / (M̄ c_p)). Over the step each zone's mass changes by its vents' net flows and its leaks:

   M_z′ = M_z − dt [ Σ_v σ_vz F_v(Δp_s,v) + L_z(π_z′) ],

   with σ_vz = +1 where z is vent v's side a and −1 where it is side b, and L_z item 7's leak taken at the step's end, as `zoneLeak` takes it. An open side's pressure is held at its value from 7. The end pressures solve these equations together, by Newton's method on the zones' pressures:
   - The Jacobian is symmetric. A vent adds dt ∂F/∂Δp_s to its two zones' diagonal entries and subtracts it from their shared entry; each zone adds its own dM/dπ and its leaks' derivative to its diagonal. Where Δp has one sign over a whole vent, ∂F/∂Δp_s is taken at |Δp| no smaller than 0.01 Pa, in the derivative only.
   - Each Newton step's linear system is solved by conjugate gradients with a diagonal preconditioner, and the step is halved while the residual grows.
   - It starts from the last step's pressures and stops when no zone's pressure changes by more than 0.01 Pa, after at most 8 Newton steps. The largest change left is counted (`stats.ventResidual`).

   Mass is conserved whether or not the solve converges, because the exchange in 9 is computed from the final Δp_s for both sides at once.

9. The exchange, conservative by construction. With Δp_s final, each slab's flow over the step, m = ṁ dt, leaves the donor's face cells of its row and enters the receiver's, shared by their weights. The gas carries the donor cells' mass fractions, soot included, and their temperature. Each receiving cell takes it in at constant pressure, as a leak's inflow is taken in (`pressureZones`, `ventLeaks`):

   Y_k ← (m_c Y_k + m Y_k,d) / (m_c + m),  T ← (m_c c_p T + m c_p,d T_d) / ((m_c + m) c_p′),

   with c_p′ at the new mass fractions. A donor cell's state is not changed in place: at its pressure a cell's mass follows its temperature. Its loss is a sink in its expansion, ln(1 − m / (ρ_g V)) / dt, and the receiver's gain a source, ln(1 + m / (ρ_g V)) / dt, with ρ_g the donor cells' density, so that the volume is the moved gas's own; both are added where `expansionEnd` adds the vent leaks'. The projection then carries the gas away from the vent on one side and towards it on the other.

   Each zone's mass (its ledger, `Z.mass`) changes by exactly its vents' net transfers, so `zoneTransport` and `pressureZones` hold its cells to it after every step, whatever the bulk transport does inside it. What leaves one side enters the other: the mass, each species, and the sensible enthalpy m c_p,d T_d are conserved across the vent by construction. In an open region nothing holds the cells to a ledger: there the transfer acts as a vent leak's does, and the region's closure is the transport's.

10. The step's length. `substepsFor` takes each vent's largest orifice speed, √(2 |Δp| / ρ_d), with the grid's, so that a step moves at most one face cell's gas through a vent row. A transfer that would take more is capped there and counted (`stats.ventClamped`).

11. The coupling, and what is not counted twice:
    - The grid: an opening's cells are solid, so only the vent crosses it.
    - Item 5: a region reached only through vents is a pressure zone. When the run has vents, the joint solve in 8 replaces `zoneLeak`'s bisection zone by zone, and a zone's leaks are terms in it, so each is counted once. Without vents the solver runs `zoneLeak` as now.
    - Item 7, the construction's leaks: a zone's faces against the solid cells that close an opening are not counted as leaky wall over the opening's area. The opening is open, and NISTIR 4770's figures are for walls with their doors' cracks.
    - Item 7, leaks from an FDS input: a zone leak with no cell goes today to every zone (`leakTerms`). A vent can split one region of a test into several zones. Its area is then shared among them by their areas of faces against solids, so that the zones' areas add up to the input's.
    - Vent leaks (`ventXb`) and ducts are unchanged.
    - The expansion: each sealed region's mean expansion is taken off as now. The vent's sources are part of that mean, which matches the ledger.

12. On the CPU (`flamescapes.js`):
    - `createSolver`: the vent test of 1 on `scene.openings`; each vent's cells closed; the loop of 7; each vent's segments, face cells and weights, band cells and sides' regions; each zone's vents. `P.subgridVents`: 0 voxelises openings as now, bit-identical.
    - `pressureZones(sim, leakDt)` at the step's end (the stage `density2`): the band means; the joint solve of 8; the transfers of 9 and their sources; then each zone's π from its ledger, as now. Tallies: `stats.ventMass` and `stats.ventEnthalpy` each way, and each vent's own for the check.
    - `expansionEnd` adds the vent sources; `substepsFor` the vent speeds.

13. On the GPU (`flamescapesGpu.js`, `kernels.js`):
    - Buffers: the vents' segments, face cells and weights, band cells (offsets and lists), and each zone's vents.
    - `ventProfile`, one thread a vent row and side: the band's mean density, and for an open side the band's mean pressure in the sill's row.
    - `ventSolve`, in place of `zoneCommitLeak` when the run has vents: one workgroup of 256 threads, one thread a zone, holding the zones' pressures in workgroup memory, with a barrier between iterations. More than 256 zones take one dispatch an iteration. The CPU follows the same order of operations, so that the two agree.
    - `ventApply`, one thread a vent row: the slabs' transfers, written to a list for each giving and receiving cell. `ventGather`, one thread a cell, mixes in what the cell receives and writes its sources. No two threads write one cell.
    - The zones' sums stay float32 trees, the pressure held as π − 1, as now.

#### Why it cannot fail (c) as attempt 1 did, and where it still could

In the six cases each box is its own pressure zone, and the vent is its only path. The hot box's mass after each step is its ledger. The ledger changes only by the vent's net transfer, since the boxes have no source and no leak, and that transfer is the tally (c) compares with. The transport's give-back and the zone's rescaling now act inside one box and cannot move mass across the vent. So the two numbers in (c) differ by rounding only: float32 cells, about 10⁻⁶ of the box's mass, against 1 % of the mass exchanged (0.013 kg in attempt 1's smallest case).

Where (c) could still fail:
- if an opening's cells are left in part open, so that a grid path bypasses the vent's tally;
- if the hot box is not one zone, or has a source or a leak the tally does not hold;
- if the box's mass is read between the transfer and the zone's update (the tool reads after the step).

In an open region (the app's, and the open side of (e) below) nothing holds the cells to a ledger. There the closure is the transport's, as attempt 1's was. (e) takes (c) on its zone side, so it does not test this either.

Where (a) and (b) can fail: the band's rows lie between one cell and 1.0 m from the wall and within 1.0 m of the door, where the reference's lie at least 0.4 m from it across the whole box; and the joint solve's Δp_s may sit away from the reference's volume balance.

Disclosed: the vent's flow law is the reference's own law, with the same C_d. So in this check (a) and (b) test the profiles the vent reads and the pressure difference it finds, not Emmons' law. The law itself is tested only by the held-out door data, in the rerun.

#### What the check reads

A change of the tool, not of its criteria, written before the run. Attempt 1's tool reads the outflow from the transport's tally of the hot box's door faces, and the neutral plane from those faces' velocities. A vent has no faces. The tool reads instead the vent's own transfer from the hot side over 2 to 6 s, and at each read the lowest zero of the vent's Δp(z). Its reference, its reads (every 0.05 s from 2 to 6 s), the hot box's mass and its criteria stay as 13.13 fixed them.

#### What is fixed in advance, and from where

- C_d = 0.68 (Prahl and Emmons 1975).
- The vent test: four cells; a tenth of the area. Both come from attempt 1, items 3 and 4 and 13.7, none of which uses measured data.
- The band: from one cell to 1.0 m from the wall, 1.0 m beyond the opening's edges.
- The solve: 0.01 Pa; at most 8 Newton steps; the derivative's floor of 0.01 Pa.

Nothing is fitted, and no series that 12 or 13 holds out is opened (14.1; 13.12, rule 4).

#### Acceptance, copied unchanged from 13.13 ("Items 4 and 3, attempt 1: check")

- Six cases: two sealed boxes 3.0 × 2.4 × 3.0 m on 0.2 m cells; a door 2.0 m high and 0.4, 0.5 or 0.6 m wide; the hot box at 400 K or 600 K. Radiation, wall exchange and deposition off; no leakage; read every 0.05 s from 2 to 6 s.
- The reference at each read is Emmons' vent integral on each box's own layered density profile (the slab form CFAST uses), measured at least 0.4 m from the dividing wall:
  - Δp(z) = Δp₀ − g ∫ (ρ_hot − ρ_cold) dz′, with Δp₀ from volume balance;
  - ṁ_out = C_d w ∫_{Δp>0} √(2 ρ_hot Δp) dz, with C_d = 0.68 from Prahl and Emmons (1975, Combustion and Flame 25:369), independent of Steckler;
  - the neutral plane is Δp's lowest zero.
- Passes in all six cases if:
  - (a) the mean outflow is within 10 %;
  - (b) the mean neutral plane is within 10 %;
  - (c) the hot box's change of mass over 2 to 6 s equals the net flow through the door within 1 % of the mass exchanged.
- Plus the closures, and parity once on the GPU.

#### Added, because the method makes them necessary (fixed now, before any run)

- (d) Energy. In each of the six cases mass and energy close within 1 %, as item 7's did with its leak: each box's change of gas heat above ambient, Σ ρ V c_p (T − T_a), over 2 to 6 s equals the vent's net heat into it, Σ m c_p,d (T_d − T_a), within 1 % of the heat carried through the vent each way.
- (e) A zone against an open region. The hot box alone, its door 0.5 m wide and 2.0 m high opening onto open cells that reach the domain's side; the hot box at 400 K and at 600 K; 0.2 m cells. (a), (b) and (c) as above, the reference taking the outside at ambient density and Δp₀ from volume balance. This is the room-to-outside case of the held-out door tests, where the open side's pressure comes from the projection.
- (f) The coarser cells. The six cases on 0.27 m and 0.34 m cells, each box the whole cells nearest 3.0 × 2.4 × 3.0 m, the door at its true height of 2.0 m; (a) to (d) with the same tolerances. The reference then integrates over the door's true height, its top row in part. That extends 13.13's reference to a door that is not a whole number of rows, and is disclosed as such.
- (g) The joint solve. Three sealed boxes in a row, on 0.2 m and 0.34 m cells, joined by two doors 0.5 m wide, the first box at 600 K, for 6 s. Each box's change of mass equals its vents' net transfers within 1 % of the mass exchanged. At every step every zone's pressure agrees with its mass within 0.01 Pa, that is, the solve converged. In the one-vent cases of (a) to (e), Newton's Δp_s equals a bisection's within 0.01 Pa.
- (h) Off is off. With `subgridVents` 0 the unit suite and the GPU parity phases are bit-identical to the build before.
- (i) The unit suite passes unbent with the vent on by default. Its scenes' doors of under four cells become vents.
- Reported, not judged: attempt 1's uniform-room formulas, as its tool reports them; the vents and zones each of Duplex, Office and Clinic gets, and each opening kept on the grid by 7; `stats.ventResidual` and `stats.ventClamped` in every case.

#### GPU parity

- The item: `tools/flamescapes-gpu-check.mjs` gets a phase `vents`. It runs 200 deterministic steps of the 0.5 m, 600 K case and of (e)'s 600 K case, and compares |CPU − GPU| for each zone's pressure and mass, gas T, O2, ρ and each vent's flow each way against the one-ulp twin's, as items 5 and 7 reported. Then (a) to (e) on the GPU's path: each outflow and neutral plane within 0.95 to 1.05 of the CPU's (14.2's ratio, applied to this check), and (c) passing.
- The rerun's verdicts on the GPU follow 14.2 as written.

#### Speed (14.6)

Milliseconds a step and run time on Duplex, Office and Clinic, CPU and GPU (`tools/backend-bench.mjs`), with the vent on and off, against the build before. If on costs more than 10 % a step on the GPU, it goes behind a switch, and the app states the verdicts found without it when it is off, which means a second full run.

#### Default

Off (`subgridVents` 0) until (a) to (i) pass and the unit suite passes unbent with it on. Then on, unless 14.6 puts it behind a switch. Made after 13.B began, it gets its own full rerun, reported beside 13.B (13.12, rule 5).

#### What ends this line

If attempt 2 fails any of (a) to (i) in any case, items 4 and 3 are not made, and doors stay as the voxeliser makes them. No third attempt at door flow is drafted. Door and vent velocity, door mass flow and the neutral plane stay as 13.A and 13.B find them, and the app states those verdicts. A further attempt needs the user's word and a new method committed here first.

#### The coordinator's decisions on this draft (2026-11-13, 03:58, before any run of it)

The draft above is the physics agent's, written blind (13.12, rule 2). The coordinator has read 13.A and 13.B's first two files, so the decisions below add no constant and move no tolerance. Each takes the draft's own proposal or the stricter reading.

1. **The vent test and the coupling stand as drafted.** 13.13's one line named "a sub-grid vent for doors under three cells wide, coupled into the projection". The draft's "fewer than four cells", its rule of a tenth of the area, and its coupling by volume sources in place of a link in the pressure equation are changes of method made after attempt 1 failed, and are disclosed as such.
2. **The band's 1.0 m and its one cell from the wall are fixed modelling constants with no source.** They are not changed after any run, whatever the check finds. Room-wide means were set aside because the check's reference reads nearly the same cells, and 13.13 refuses a check that cannot fail.
3. **13.13's layered check governs items 4 and 3.** The uniform-room formulas of 13.4 and 13.8 are reported and not judged.
4. **(f)'s cases on 0.27 m and 0.34 m cells are acceptance cases,** as (a) to (e) are. The app runs on those cells.
5. **A leak that an FDS input gives to a region is shared among the zones a vent makes of that region,** by their areas of faces against solids, so that the areas add up to the input's.
6. **C_d = 0.68.** The citation and its DOI were checked on Crossref. The value is taken as fire texts report it from that paper; the paper itself was not read. If a reading of the paper gives another value, that is reported and this item's check is run again with it, whatever the first run found.
7. **How the validation reads a door that is a vent** (a vent has no faces), fixed now:
   - the neutral plane is the lowest zero of the vent's Δp(z), averaged over the window as the face reading is;
   - the mass flow each way is the vent's own transfer each way;
   - the velocity at a probe at height z is the ideal speed sign(Δp(z)) √(2 |Δp(z)| / ρ_d), with ρ_d the donor's band density in that row. C_d times that speed is reported beside it. An orifice's coefficient is mostly the jet's contraction, which argues for the ideal speed on the door's centreline, but a probe in the door's plane may read less. So both are judged, and the stricter word stands (12.6).
8. **Not tested by (a) to (i), and stated with any result:** no momentum crosses a vent, so a spill plume's direction is lost; an open side's reference pressure comes from the projection's pressure, which is not the physical pressure in hot gas; and a door between two open regions stays on the grid.
9. **Made, the item changes the app widely:** most rooms behind open doors become pressure zones. 14.6's speed rule decides whether it sits behind a switch.

### 13.18 Item 6, attempt 4: FDS's extinction model 1 (2026-11-13, before any run of it)

Drafted blind to every validation result, as 13.12 sets. Nothing below changes after a run of it; anything added later is dated, with its commit.

#### What failed before

- Attempt 1 (13.9 and 13.10; f83977c on `flamescapes/extinction-prototype`): FDS's model 2 with FDS's critical flame temperatures. The box stopped at 10.79 % O2 by volume against an analytic 10.13 %, a gap of 0.66 % where 0.5 % was allowed. It failed.
- Attempt 2 (13.10's second check; b92dfc4): it passed, but the run was bit-identical with extinction off, so it could not fail and is not counted (13.13).
- Attempt 3 (13.13; record 0048490 on `flamescapes/extinction-prototype`, GPU port d1f8b6c on `flamescapes/extinction-wip`): model 2 again, its reference each cell's own threshold. In the cooled sealed box it stopped at 584.8 s after 11.657 MJ:
  - (a) the box's oxygen 0.242 % by volume below its cells' mean threshold (0.5 % allowed);
  - (b) one cell burned in the last step that burned, 0.012 % from its own threshold (0.1 % allowed);
  - (c) without extinction the run stopped at 1268.8 s after 25.369 MJ.

  It passes, but it is not made: on by default, two unit tests fail, and they are not bent (d1f8b6c).
  1. "every fire class burns: liquid pools give off at most their burning rate" (`flamescapes.test.mjs`). Class B, a pool lit by the pilot, releases 691 J in 10 s. The test needs more than 100 kJ.
  2. "a burner's fuel burns only with oxygen: in a sealed room its heat stops at the room's oxygen, its gas stays below methane's flame temperature, and the unburnt heat waits" (`flamescapes.burner.test.mjs`), called test 2 below. The burner releases more than 0.8 of the heat its table asks. The test needs less than 0.8.

Why test 1 fails. The pool's vapour enters the gas cell over it. Model 2 counts the fuel the cell's oxygen cannot burn as a heat sink, so the rich cell fails the criterion, the vapour builds up, and the cell never burns. Its neighbours stay cold, and the solver's gas reaction is an Arrhenius rate at the cell's temperature (or a flame particle's), so they do not light either.

Why test 2 fails is not diagnosed. Since item 7, the room leaks by default (below).

A first draft of this attempt took model 2 on the stoichiometric pair. Before any run, the coordinator swapped the order: model 1 is attempt 4, and that variant is the named attempt 5 (What ends this line). The reasons come from FDS's own text, not from any validation result:
- FDS makes model 1 the default of its default mode, VLES, and writes that it "is meant to be used for simulations where the grid cells are relatively large and flames cannot be resolved", while model 2 "requires a relatively fine grid". The cells here are 0.2 to 0.4 m.
- Model 1 takes FDS's tabulated oxygen index directly. Model 2 at this solver's constant specific heats puts the limit elsewhere: for methane in nitrogen-diluted air at 20 °C, about 11.3 % by volume against FDS's 13.9 % (a hand calculation from attempt 3's formula).
- Model 1 has no term for the fuel in the cell, so excess vapour is not a heat sink in it. The vapour still dilutes the cell's oxygen (The method, 4).

#### What was read in FDS

firemodels/fds at 10c758f (master, 2026-11-12, 21:27 UTC), fetched raw from https://raw.githubusercontent.com/firemodels/fds/10c758f7439f6612b503b88e47b3b1b97fe70cda/ and the path. Its `Source/fire.f90` and `Source/data.f90` are byte for byte those at 5f8f194, which attempt 3 read. The blob URLs, for reading, are https://github.com/firemodels/fds/blob/10c758f7439f6612b503b88e47b3b1b97fe70cda/ and the path.

- `Source/fire.f90` (https://github.com/firemodels/fds/blob/10c758f7439f6612b503b88e47b3b1b97fe70cda/Source/fire.f90):
  - `EXTINCT_1`, lines 1387 to 1411: the criterion (below).
  - `COMBUSTION_MODEL`, lines 768 to 1040. Line 796 keeps the cell's composition at the start of the step (`ZZ_0`). The reaction is integrated first (lines 866 to 987). Then, at lines 996 to 1000, under `IF (SUPPRESSION .AND. .NOT.EXTINCT)`, model 1 is called with `ZZ_0` and `TMP_IN`. If the cell is put out, lines 1003 to 1012 return its species to `ZZ_0` and set its heat release to zero.
  - The cell loop, lines 254 to 270: `TMP_IN` is the cell's temperature, `TMP(I,J,K)` (line 264).
  - `CHECK_AUTO_IGNITION`, lines 1347 to 1379, called at lines 868 to 881, before the integration.
- `Source/func.f90`, `GET_MASS_FRACTION`, lines 1570 to 1579: the oxygen mass fraction is a sum over all the cell's lumped species, the fuel among them.
- `Source/read.f90`:
  - lines 4573 to 4593: with no `EXTINCTION_MODEL` given, VLES and SVLES take model 1, DNS and LES model 2; line 4601 turns the free-burn temperature into kelvin;
  - line 4769: a reaction with Arrhenius parameters (`A` or `E`) sets `SUPPRESSION = .FALSE.`, which turns both models off;
  - lines 4788 to 4789 and 4817 to 4818: each reaction's critical flame temperature and oxygen index, looked up by its fuel's name when not given, and `Y_O2_MIN = X_OI W_O2 / (X_OI W_O2 + (1 - X_OI) W_N2)`;
  - line 5041: the default auto-ignition temperature is −273.15 °C; lines 1768 and 1906: the ambient, 20 °C by default.
- `Source/cons.f90`: `SUPPRESSION = .TRUE.` (line 193); `FREE_BURN_TEMPERATURE = 600` °C (line 428); `SIM_MODE = 3`, which is VLES (lines 23 and 559).
- `Source/data.f90`: `LOOKUP_CRITICAL_FLAME_TEMPERATURE`, lines 2673 to 2701, and `LOOKUP_LOWER_OXYGEN_LIMIT`, lines 2704 to 2732.
- Technical Reference Guide, combustion chapter (https://github.com/firemodels/fds/blob/10c758f7439f6612b503b88e47b3b1b97fe70cda/Manuals/FDS_Technical_Reference_Guide/Combustion_Chapter.tex), lines 433 to 486: the critical flame temperature (lines 441 to 465; Y_OI defined at line 462) and "Extinction Based Mainly on Oxygen Concentration" (lines 467 to 483).
  - Line 470, its footnote: the model "is written in terms of the oxygen volume fraction ... However, within the numerical algorithm, all values are converted into mass fraction".
  - Line 483: the free-burn temperature "is needed for simulations in which the characteristic grid cell size ... is much larger than 1 cm".
- User's Guide (https://github.com/firemodels/fds/blob/10c758f7439f6612b503b88e47b3b1b97fe70cda/Manuals/FDS_User_Guide/FDS_User_Guide.tex): "Flame Extinction", lines 4976 to 5048, with its table of X_OI and T_OI (lines 5002 to 5033); "Piloted Ignition", lines 5053 to 5062; finite-rate combustion, line 5374; the table of simulation modes, lines 8908 to 8932, where the default mode is VLES (line 8915).
- Not read: the compiled PDF guides; FDS's verification inputs (`Extinction/extinction_1.fds`); Beyler's chapter of the SFPE Handbook, and the measurements the guide cites for its values (Morehart et al., Pitts, Bundy); `Source/wall.f90` beyond a search, whose hits on extinction concern solid surfaces (lines 1103 to 1112, 1340 to 1345, 2996 to 2998). Crossref's DOIs for the FDS guides, 10.6028/NIST.SP.1018 and 10.6028/NIST.SP.1019, resolve to the version 4 editions (2006 and 2004), so they are not cited for this text.

#### FDS's model 1, as FDS codes it

For a cell at the start of its reaction step, with temperature T (the cell's mean, `TMP(I,J,K)`) and oxygen mass fraction Y_O2 (over all the cell's gas, fuel included):

- Y_OI = X_OI W_O2 / (X_OI W_O2 + (1 − X_OI) W_N2): the oxygen mass fraction of an oxygen and nitrogen mixture at the oxygen index X_OI by volume (`read.f90`, line 4818);
- if T < T_fb, Y_lim = Y_OI (T_OI − T) / (T_OI − T_∞); otherwise Y_lim = 0 (`fire.f90`, lines 1403 to 1408);
- the cell is put out if Y_O2 < Y_lim (line 1409).

T_OI is the critical flame temperature, T_∞ the run's ambient (`TMPA`), T_fb the free-burn temperature, 600 °C by default. Both comparisons are strict. T_OI and Y_OI are reaction 1's, for every cell (lines 1395 to 1398 and 1408). Nothing else enters: not the fuel in the cell, its heat of combustion, or any specific heat. The fuel enters only as part of the gas over which Y_O2 is taken, so vapour lowers Y_O2.

A cell put out keeps its start-of-step composition and releases no heat that step (lines 1003 to 1012). Its fuel stays in the gas, is carried by the flow, and "can re-ignite if it mixes with sufficient oxygen elsewhere in the domain" (User's Guide, line 5056).

Values (`data.f90`):

| Fuel | X_OI | T_OI |
|---|---|---|
| Methane | 0.139 | 1507 °C |
| n-Heptane | 0.134 | 1497 °C |
| Any fuel not listed | 0.135 | 1427 °C |

The User's Guide's table (lines 5009 to 5030) gives the same three rows. It lists some fuels `data.f90` does not (isobutane, isooctane, propanol; decane where `data.f90` has n-decane); the code uses `data.f90`.

Burner cells. A burner in FDS is a surface that gives fuel gas to the cell beside it, and that cell's reaction is tested as any other. I found no exemption for it in `fire.f90`: not in the cell loop (lines 254 to 270), in the test for a chemically active cell (lines 365 to 461), or in `COMBUSTION_MODEL`. With the default auto-ignition temperature (0 K) a burner's fuel burns on contact with air wherever model 1 allows. An auto-ignition exclusion zone lifts only the auto-ignition test (lines 1360 to 1371), not model 1.

#### The method

1. The criterion: FDS's model 1 as above, in this solver's quantities.
   - T: the cell's gas temperature, `sim.T[c]`, when the test is made. Not the flame particle's temperature that the gas reaction's rate takes (`Teff`): FDS takes the cell's mean.
   - Y_O2: `sim.Y[c * NS + SPECIES.O2]`, over the cell's whole gas (fuel vapour, products, soot and water; nitrogen the rest).
   - T_∞: the run's ambient, `sim.P.ambient`.
   - T_fb: a new parameter, `freeBurnTemperature`, 600 °C, FDS's default.
   - T_OI: the fuel's `st.cft`, as attempt 3 has it.
   - Y_OI: a new `st.yOI` from `stoichiometry`, by FDS's relation with the solver's own molar masses (W_O2 = 0.032, W_N2 = 0.028 kg/mol), so that the limit at T_∞ in oxygen and nitrogen is X_OI in the solver's own gas. FDS's molar masses (31.9988 and 28.0134 g/mol) would put it 0.044 % lower, relative (hand calculation).
   - X_OI: a new `lowerOxygenLimit(fuel)`: the fuel's own `lowerOxygenLimit` (a volume fraction) if given, else FDS's for its name (a table `FDS_LOWER_OXYGEN_LIMIT` with `data.f90`'s rows), else 0.135. It mirrors attempt 3's `criticalFlameTemperature`.

2. Where it sits. Attempt 3's two guards stay where they are; only what they test changes.
   - The gas reaction (`chemistry`, its gas combustion, line 1654 at d1f8b6c): ahead of the Arrhenius rate, with the vapour's fuel, `sim.stoich[sim.fuelVapourKind?.[c] ?? 0]`. A cell put out reacts nothing this step. Its vapour and oxygen stay, the flow carries them, and they may burn in a later step or another cell, as in FDS.
   - The burner (`burnInCell`, lines 3385 to 3398 at d1f8b6c): after the cell's oxygen is found to burn some fuel and before the burn, with the burner's fuel, `b.stoich`, on the cell's state before this step's burn in it. A cell put out takes none of the burner's fuel. Its share goes on to the burner's other cells and then to the cells its flame may reach (`burnerStep`), and what finds no cell waits (`b.pending`), as in attempt 3.
   - Not tested, as in attempt 3: the solids' oxidation (Eq. 15), a surface reaction outside FDS's gas-phase model.

3. Departures from FDS, disclosed.
   - FDS integrates the reaction and then discards it if the cell is put out. Here the test comes first and nothing reacts. Both test the state at the start of the cell's reaction, so the outcome is the same.
   - FDS takes reaction 1's T_OI and X_OI for every cell. Here a cell takes its vapour's fuel and a burner its own, as in attempt 3. With one fuel in a run the two agree. The fire classes carry no FDS fuel name, so they take FDS's values for an unlisted fuel (0.135, 1427 °C); the burner's methane takes methane's.
   - FDS lets a burner's unburnt fuel into the gas. This solver's burner holds it back and finds it other cells (d3c5751, 2026-11-12). That predates item 6 and is not changed here.
   - FDS turns extinction off in any run with an Arrhenius reaction (`read.f90`, line 4769), and its guide recommends finite-rate reactions only for DNS (line 5374). This solver's gas reaction is an Arrhenius rate. 13.9 puts the criterion on it, and attempt 3 did; it stays there, with no FDS precedent for that pairing.

4. What a rich cell meets. Model 1 has no term for the fuel, so the excess vapour that locked out attempt 3's pool cell is no longer a heat sink. But the vapour dilutes the cell's oxygen mass fraction, as FDS's code takes it. At FDS's values for an unlisted fuel, a cell of air and vapour is put out above about 35 % vapour by mass at 20 °C, 40 % at 400 K and 49 % at 600 K (hand calculation, from Y_F > 1 − Y_lim / 0.232); at 600 °C and above, never while it has oxygen. FDS lights fuel on contact. This solver's gas reaction waits for its Arrhenius rate, so vapour can gather in a cold cell. Whether the class B pool lights before its cell passes that share is not known before the run; (f) below decides.

5. What stays from attempt 3 and what goes.
   - Stays: the switch `extinction` (0 is off); `FDS_CRITICAL_FLAME`, `criticalFlameTemperature` and `st.cft`; METHANE's name; the two guards and their places; the burner's handling of fuel a cell refuses; the GPU's `extinction`, `cft0` and `cft1`; the parity phase; the test file's sealed box and its attempt 3 test, with the reference changed as "What the check reads" sets.
   - Goes: model 2's body (φ, the burnt share, the other gas's heat capacity, the heat balance) and its arguments `mF`, `Tfuel` and `fuelInGas`, on both paths; the test file's `criticalOxygen` and `diluent`, and its unit check of model 2's closed form, which (d) and (e) replace. The 13.10 second check's test is not carried: 13.13 does not count it, and its record is b92dfc4.
   - New: `FDS_LOWER_OXYGEN_LIMIT`, `lowerOxygenLimit`, `st.yOI`, `DEFAULTS.freeBurnTemperature`.

6. On the CPU (`flamescapes.js`):

   ```js
   export function flameSustained(sim, c, st) {
     if (!(sim.P.extinction > 0)) return true;
     const T = sim.T[c];
     if (!(T < sim.P.freeBurnTemperature + KELVIN)) return true;
     return !(sim.Y[c * NS + SPECIES.O2] < (st.yOI * (st.cft - T)) / (st.cft - sim.P.ambient));
   }
   ```

   It is called as `flameSustained(sim, c, sim.stoich[sim.fuelVapourKind?.[c] ?? 0])` in the gas reaction and `flameSustained(sim, c, b.stoich)` in `burnInCell`. The strict comparisons are FDS's. `DEFAULTS.extinction` is 0 until the item is made (Default).

7. On the GPU (`flamescapesGpu/kernels.js`, `flamescapesGpu.js`):
   - `PARAMS` gains `yOI0`, `yOI1` and `freeBurn` (K) beside attempt 3's `extinction`, `cft0` and `cft1`. `flamescapesGpu.js` writes them from `sim.stoich[0].yOI`, `sim.stoich[1].yOI` and `P.freeBurnTemperature + KELVIN`.
   - The WGSL `flameSustained(c, cft, yOI)` makes the same three tests in f32.
   - `chemCell` calls it with `select(U.cft0, U.cft1, vk == 1u)` and `select(U.yOI0, U.yOI1, vk == 1u)`; the burner's `burnIn` with `U.cft1` and `U.yOI1`. The GPU runs only a design-fire item's burner (`flamescapesGpu.js`, line 243), whose fuel is the item's (`fuelOf(1u)`), as attempt 3's port has it.

#### Where it could fail

- (a): a cell that went out and then cooled has a higher threshold at the stop. Near 382 K, model 1's limit for methane in its own products rises by about 0.0095 % by volume for each kelvin of cooling (hand calculation). Attempt 3's model 2 had the same exposure and sat 0.242 % below.
- (b): the last cell to burn may take much of the burner's fuel in its last step. One step's 2 kJ (20 kW over 0.1 s) in one 0.25 m cell at 382 K takes about 1 % of its oxygen by volume (hand calculation), ten times (b)'s tolerance. Attempt 3 had the same exposure, and its last cell was 0.012 % from its threshold.
- (f): the pool's cell, as item 4 of the method sets out.
- Test 2: below.

#### What the check reads

The tool is attempt 3's test in `flamescapes.extinction.test.mjs` (d1f8b6c), its case coded as there: a box of 10 × 9 × 10 cells on 0.25 m cells, its walls the domain's outer cells, the burner spread over every open cell, the walls at their default convection, no radiation, steps of 0.1 s. The stop is the first step whose flame rate is below 1 % of 20 kW, within 3000 s. (b)'s cells are those whose oxygen the last burning step's burner stage took, as the tool finds them (it runs that stage on a copy of the gas).

One change of the tool, not of its criteria, written before the run: each cell's threshold is model 1's, Y_lim at the cell's temperature at the stop (0 at or above 600 °C), carried to volume as the tool already carries a mass fraction: the cell's oxygen by volume times Y_lim / Y_O2.

#### What is fixed in advance, and from where

- X_OI and T_OI by fuel: FDS's `data.f90` (above).
- T_fb = 600 °C: FDS's default (`cons.f90`, line 428).
- T_∞: the run's ambient, as FDS's `TMPA`.
- Y_OI from X_OI: FDS's relation (`read.f90`, line 4818; Technical Reference Guide, line 462), with the solver's molar masses.
- The tolerance of (d) and (e), 10⁻⁴ relative: set from float32 rounding (below), not from any run.

Nothing is fitted, and no series that 12 or 13 holds out is opened (14.1; 13.12, rule 4).

#### Acceptance, copied unchanged from 13.13 ("Item 6, attempt 3")

- Case: the first check's own: a 20 kW methane burner in a sealed box whose walls cool it, leakage off.
- Reference: each cell's threshold, from its own temperature and composition, at the stop.
- Passes if (a) the gas-mass-weighted mean oxygen of the box at the stop is within 0.5 % by volume of the gas-mass-weighted mean of the cells' thresholds; (b) every cell that burned is within 0.1 % of its own threshold; and (c) the run differs from the same run with extinction off, so that it is exercised.

The threshold is now model 1's (What the check reads). And, from 13.9: the unit tests and closures must still hold.

#### Added, because the method makes them necessary (fixed now, before any run)

- (d) The criterion against FDS's closed form. One open cell of the sealed box, its fields set directly, as FDS's own verification sets 100 sealed boxes of methane (User's Guide, lines 5035 to 5048; its input was not read).
  - Three fuels: methane, a fuel named N-HEPTANE, and an unnamed fuel.
  - Two compositions: air vitiated with CO2 and H2O, as attempt 3's unit check set it; and the same with fuel vapour at 1.5 and 3 times what its oxygen can burn.
  - Temperatures: T_∞, 400, 600 and 800 K, T_fb − 1 K, T_fb + 1 K, 1200 K and T_OI + 1 K.
  - The closed form is evaluated in double precision at the cell's stored temperature.
  - Passes if, wherever Y_lim > 0, the flame lives with Y_O2 = Y_lim (1 + 10⁻⁴) and goes out with Y_lim (1 − 10⁻⁴), for every fuel, composition and temperature; and above T_fb it lives with Y_O2 = 10⁻⁵.
  - The tolerance: the cell's fields are float32, about 6 × 10⁻⁸ relative, so 10⁻⁴ leaves rounding three orders of magnitude of room, and it is 0.0014 % by volume at methane's limit at 20 °C.
  - It fails if the coded form differs: a volume fraction in place of the mass fraction, the free-burn cut on the wrong side, the fuel counted, or another fuel's values.
- (e) The oxygen index at 20 °C. One cell of oxygen and nitrogen only, at T_∞ = 293.15 K. For each of the three fuels the flame lives at X_O2 = X_OI (1 + 10⁻⁴) by volume and goes out at X_OI (1 − 10⁻⁴), the volume fraction from the solver's molar masses. The tolerance is (d)'s. It fails if Y_OI is taken with FDS's molar masses (4.4 × 10⁻⁴ low, relative; hand calculation), or by volume, or at another ambient.
- (f) The class B pool. "every fire class burns: liquid pools give off at most their burning rate" (`flamescapes.test.mjs`) passes unbent with extinction on: every class releases more than 100 kJ in 10 s. The limit is the test's own.
- (g) Off is off. With `extinction` 0 the unit suite and the GPU parity phases are bit-identical to the build before item 6's code, the base the item is built on. Bit for bit, because with 0 the guard returns before it reads any field.
- (h) The unit suite passes unbent with extinction on by default, test 2 included as the user sets it up (below).
- Reported, not judged: the copied case on 0.20 m and 0.34 m cells; in (f)'s class B scene, the heat released by 10 s with extinction on and off, and the largest vapour share of a cell put out; test 2's released share four ways, with extinction on and off, with the default leaks and with none.

#### The burner's sealed-room test (test 2)

Attempt 4 changes the burner's criterion too, so whether test 2 still fails is known only when it runs. Model 1 does not act above 600 °C, and that room's gas may pass it; that is a guess, not a result. Test 2's setup is still the user's decision, taken before the run, on the argument written after attempt 3's failure.

For setting `leakage: []` in its setup:
- Its name says "in a sealed room its heat stops at the room's oxygen", and its comment derives the limit from that oxygen: 1.2 kg, about 16 MJ, against 30 MJ asked. In a room that leaks, the oxygen is not bounded by the room's own, and 0.8 has no derivation.
- It was written at d3c5751 (2026-11-12, 06:53), before items 5 (1113a47) and 7 (e856c3a), when a room with no opening had no leak. Since item 7 it leaks by default as average construction: in this 4.5 m³ room, about 2.5 × 10⁻³ m² (hand calculation from `leakageWall` and `leakageFloor`). In the solver's own terms a sealed region is `leakage: []` (item 7).
- At 1113a47, with pressure zones on and no leaks, the unit suite passed (522 tests), this test among them, with extinction off.

Against it:
- It is proposed after a failure, which is how a test gets bent.
- With extinction off the test passes with the default leaks, so it guards that configuration too, and the app's closed rooms use it.

The proposal:
- The test's setup states `leakage: []`. Its limits are unchanged. This is disclosed as written after a failure.
- A second test guards the default leaks with a bound derived for a room that leaks: the burner releases no more than the heat of the room's oxygen at the start plus that of the oxygen in the air its leaks drew in (Huggett), within 0.1 %, as the first test's oxygen bound allows. It needs a new tally of the air the leaks drew in (`stats.leakAirIn`).

If the user accepts, test 2 runs as proposed and the second test must pass too. If the user does not, test 2 runs as it stands and must pass unbent. If it then fails, item 6 is not made, and no attempt is shaped to pass a limit that has no derivation for a room that leaks (What ends this line).

#### GPU parity

- The item: the WGSL mirror of the method's item 7. The extinction phase of `tools/flamescapes-gpu-check.mjs` (d1f8b6c) runs again as it stands: the closed room with no leakage, a fast 20 kg design-fire item, 2500 steps of 0.1 s, extinction on and off, |CPU − GPU| for each field beside the one-ulp twin's. The class B scene of (f) is added as a second case, over its 10 s. Judged: the heat released at each read within 0.95 to 1.05 of the CPU's, on and off (14.2's ratio, applied to this check).
- (a) to (c) are CPU checks. The GPU runs only a design-fire item's burner, so the copied box, whose burner is the validation's kind, does not run on it.
- The rerun's verdicts on the GPU follow 14.2 as written.

#### Speed (14.6)

Milliseconds a step and run time on Duplex, Office and Clinic, CPU and GPU (`tools/backend-bench.mjs`), with extinction on and off, against the build before. Model 1 is a few operations a tested cell, fewer than model 2's loop over the species, whose port cost nothing measurable on the GPU by the record. If on costs more than 10 % a step on the GPU, it goes behind a switch, and the app states the verdicts found without it when it is off, which means a second full run.

#### Default

Off (`extinction` 0) until (a) to (h) pass, test 2 included as the user sets it up. Then on by default, as FDS's suppression is (`cons.f90`, line 193), unless 14.6 puts it behind a switch. Made after 13.B began, it gets its own full rerun, reported beside 13.B (13.12, rule 5).

#### What ends this line

- If attempt 4 fails any check, an attempt 5 may be drafted, and only then: FDS's model 2 criterion taken on the stoichiometric pair.
  - It is attempt 3's criterion, with the gas reaction testing the fuel its cell's oxygen can burn, m_F = min(Y_F m, Y_O2 m / s), in place of all its vapour, Y_F m (m the cell's gas mass, s the fuel's oxygen per kilogram, Huggett's). The burner is unchanged: `burnInCell` already passes the fuel the cell's oxygen can burn.
  - A rich cell's excess vapour is then not a diluent; its products, its nitrogen and its other gas still are.
  - It departs from FDS's model 2, which counts excess fuel as a diluent by design (Technical Reference Guide, line 493), and is disclosed as such. Its limit stays below FDS's oxygen index at the solver's specific heats (What failed before).
  - Its full method and check are to be committed here before it runs.
- Attempt 5's burner is attempt 3's, which failed test 2 with the default leaks, and test 2's room holds no fuel but the burner's. So without the user's change of test 2's setup, attempt 5 is not drafted.
- If attempt 5 fails too, item 6 ends. Extinction stays off. The closed-door UL FSRI tests' overprediction (13.14) is stated as it is. A further attempt needs the user's word.

#### The coordinator's decisions on this draft (2026-11-13, 04:23, before any run of it)

The draft above is the physics agent's, written blind (13.12, rule 2). The coordinator has read 13.A and 13.B's first files, so the decisions below add no constant and move no tolerance. Each takes the draft's own choice, or the reading that needs no one's word.

**Checked by the coordinator in FDS's sources at 10c758f** (fetched again, apart from the agent): `EXTINCT_1` as quoted (`fire.f90`, lines 1387 to 1411), with reaction 1's values and both comparisons strict; its call after the integration and the reset of a cell put out (lines 996 to 1012); `GET_MASS_FRACTION` as a sum over every tracked species (`func.f90`, lines 1570 to 1579); model 1 as VLES's default (`read.f90`, lines 4586 to 4588) and VLES as the default mode (`cons.f90`, lines 23 and 559); `SUPPRESSION` on by default and turned off by Arrhenius parameters (`cons.f90`, line 193; `read.f90`, line 4769); the free-burn temperature of 600 °C (`cons.f90`, line 428); Y_O2_MIN's relation (`read.f90`, line 4818); the three fuels' values (`data.f90`, lines 2690 to 2729); and the four sentences quoted from the two guides. Two of the hand calculations were repeated: methane's Y_OI with the solver's molar masses is 0.1558, and air with vapour falls below the unlisted fuel's limit at 20 °C above 35 % vapour by mass.

1. **The order stands: model 1 is attempt 4.** The draft found that the third reason for the swap holds only in part: FDS takes the oxygen mass fraction over the whole cell's gas, so vapour still dilutes it, and a cold, rich pool cell can still be put out. That is stated in the method (item 4) and decided by (f). The order is not changed for it, because it was set from FDS's own text before any run and the first two reasons stand.
2. **The scope is 13.9's:** the burner's fuel and the gas reaction. Model 1 on the burner alone, which is where FDS itself would apply it, is not taken; it would narrow 13.9 after three failures.
3. **Y_OI takes the solver's molar masses,** as drafted, and (e) is judged as written.
4. **The cell's own temperature is tested,** not a flame particle's, as drafted and as FDS does.
5. **The fire classes' data are not changed.** Class B carries no FDS fuel name, so it takes FDS's values for a fuel not listed (0.135, 1427 °C). Naming it n-heptane would change a fire class to help a check.
6. **The 13.10 second check's test is dropped,** as drafted. Its record is b92dfc4.
7. **The base** is `next/calibration-ui` at the commit the item's branch starts from, whose solver is 13.B's (550fce5). Attempt 3's extinction code is brought over from d1f8b6c; that commit's door-check changes are not. "The build before" in (g) and in 14.6 is that base.
8. **GPU parity is judged as drafted:** the extinction phase and the class B scene, heat released within 0.95 to 1.05 of the CPU's (14.2's ratio). (a) to (c) are CPU checks, because the GPU runs only a design-fire item's burner.
9. **Test 2 runs as it stands.** The user is away and has not answered the draft's proposal, so its second branch applies: the test runs with the default leaks and must pass unbent. Its released share is reported four ways, as the draft lists. If it fails, item 6 is not made, extinction stays off, and the draft's proposal (a stated `leakage: []`, and a second test with a bound derived for a room that leaks) goes to the user with that result beside it. An acceptance then would be dated and disclosed as given after a result.
10. **What ends this line stands as drafted,** attempt 5's condition included.

### 13.19 Item 8: radiation from the gas, CO2 and H2O (2026-11-13, before any run of it)

Drafted blind to every validation result, as 13.12 sets. Nothing below changes after a run of it; anything added later is dated, with its commit. Line numbers are those of commit 550fce5, which the working tree's `src/core` equals.

#### What the solver does now

The claim to check: hot gas emits and absorbs only through its soot; nothing casts rays below `emitAbove`; so a soot-free hot layer reaches the walls only by convection and by the burner's radiant fraction. The code bears it out, with three qualifications after the list.

- Gas emission (`flamescapes.js` 845 to 853). A gas cell emits only if T > `emitAbove` (500 K, line 218) and its soot mass fraction Y_s > 0. It emits 4 κ_s Y_s σ (T⁴ − T_a⁴) V dt, with κ_s = `sootAbsorption` = 500 (line 162, "the paper's residual absorption coefficient"). So the soot's absorption coefficient is 500 Y_s m⁻¹. The emission is capped at half the cell's heat above ambient (851). CO2 and H2O appear nowhere in `radiation` or `march`.
- Surface emission (829 to 844; one-dimensional walls in `wallEmit`, 2114 to 2150). A solid or porous cell above `emitAbove` emits ε σ (T_s⁴ − T_a⁴) over its faces open to gas (ε 0.9 for the inert materials, 103 to 107), capped as the gas is. A face below 500 K emits nothing (2119, 2135).
- Rays (855 to 890). Each emitting cell casts `rays` = 8 rays (219) of equal energy from its centre. Their directions come from the cell's own stream of the step (`cellStream`, salt 2, line 860). From a surface, a uniform direction plus the face's normal gives Lambert's law.
- Absorption (`march`, 919 to 973). A ray loses the fraction 1 − exp(−500 Y_s ℓ) in each gas cell it crosses (938 to 943), except the cell that cast it (936). The first solid or porous cell it meets takes all that is left (951 to 963), whatever its emissivity. A ray that leaves the grid returns (949), and what it carries is counted nowhere. After `rayReach` = 40 crossings (220), what is left goes into the cell where the ray stopped (966 to 972).
- Bookkeeping. Each cell emits from its temperature at the stage's start; the deposits are added after the last cell has emitted (891 to 900). `stats.radiated` counts the cells' emission (855).
- The burner (`createBurner` 3113 to 3227, `burnerStep` 3339 to 3381). Of the heat burnt, (1 − χ_r) goes into the flame cells' gas (`burnInCell`, 3323). The rest leaves the axis cell along 16 fixed directions, turned each step (3361 to 3376), absorbed by soot on the way and by the first solid. χ_r defaults to 0.2 (3134); the validation passes the FDS input's value (`validation-suite.mjs` 249 to 253). With no rays the radiant part leaves the domain (3099). It is tallied in `b.radiant` (3379), not in `stats.radiated`. The burner runs in the `pilot` stage, before `radiation` (3402 to 3406). Methane carries no soot (3075).
- Convection. An inert solid's face exchanges with its gas at `wallConvection` = 10 W/m²K (132, 1872); one-dimensional walls use the same coefficient (2170, 2288).
- The GPU (`flamescapesGpu/kernels.js`) mirrors this: the same gas rule (1341 to 1349), absorption by soot alone (1271 to 1277), a ray leaving the grid returns (1281). Deposits are fixed-point kelvin, 16 384 steps a kelvin (357, 1190 to 1193), each clamped to ±20 000 K. A scene with a validation burner (one with rays) is sent to the CPU solver (`flamescapesGpu.js` 978 to 981). The GPU runs only the design-fire burner, whose χ_r is 0.
- The heat ledger (13.15) records `radiated` from `stats.radiated` (`validation-suite.mjs` 420). It has no term for radiation that leaves the grid.

Qualifications to the claim:
1. The burner's radiant part is cast whatever the temperature. The floor does not apply to it.
2. A surface above 500 K radiates to other surfaces. In a soot-free room the walls rarely pass 500 K away from the flame, so the layer itself reaches the walls only by convection, as claimed.
3. In a validation burner whose fuel has soot, the flame cells above 500 K emit soot radiation of their own (845), on top of the χ_r already cast. The radiant fraction is then counted twice in those cells. Methane burners are not affected.

#### The method

Chosen: the weighted sum of gray gases (WSGG) of Bordbar, Węcel and Hyppänen (2014), as FDS 6.11.1 codes it for its `WSGG_MODEL` option: four gray gases and one transparent window, each carried along the solver's rays as its own share of the energy.

Considered and set aside:
- FDS's default gray gas: one coefficient per species from RadCal, the smaller of the Planck mean and the effective coefficient over a 0.1 m path (Technical Reference Guide, appendix "Gas Phase Absorption Coefficients"; `radi.f90` 3134). It is FDS's default. But one gray coefficient set at 0.1 m overstates the emissivity of a layer a metre or more deep, and it would put RadCal's narrow-band tables inside the solver. RadCal serves instead as the independent reference of check (b).
- Planck-mean coefficients of CO2 and H2O: right only for optically thin paths. At room scale the band centres are saturated, so a Planck mean overstates emission and absorption alike.
- Smith, Shen and Friedman (1982): fitted for 600 to 2400 K. A compartment's layer is mostly 320 to 700 K.
- Bordbar et al. (2014): its constants are in FDS's own source (public domain, read); its weights are polynomials in T used by FDS down to ambient; its coefficients do not depend on a path length. On the solver's rays each gray gas is carried exactly as the soot is now, with no path length to choose. FDS's User Guide calls the option experimental, with very limited verification and validation. Check (b) therefore compares it with RadCal before it is made.

1. The gas's composition, in each gas cell at the radiation stage's start.
   - Mole fractions as the equation of state takes them (`gasDensity`, 290 to 300): n_k = Y_k / M_k with `MOLAR` (line 85), nitrogen the rest at 0.028 kg/mol; X_k = n_k / Σ n. X_c = X_CO2, X_w = X_H2O.
   - The molar ratio M_r = X_w / (X_c + 4.44 × 10⁻¹⁵), the second term FDS's TWENTY_EPSILON_EB (`prec.f90` 40).
   - The absorbing gases' pressure in atmospheres: p_a = p (X_w + X_c) / P_STP, with P_STP = 101 325 Pa (`cons.f90` 370) and p = 101 325 Pa. The solver's equation of state takes every cell at P_AMBIENT (line 299), and the radiation does the same. FDS takes the zone's background pressure (`radi.f90` 3991 to 3992); the difference is the zone's π − 1.
   - If X_c and X_w are both below 4.44 × 10⁻¹⁵, the gray gases' coefficients are 0 (FDS, `KAPPA_WSGG`).

2. The gray gases' absorption coefficients, m⁻¹, for j = 1 to 4 (FDS 6.11.1 `radi.f90` 5010 to 5049):

   κ_g,j = k_j(M_r) p_a,

   with k_j in atm⁻¹ m⁻¹:
   - M_r < 0.01 (CO2 alone): k_j = KP1_j;
   - M_r > 4 (H2O alone): k_j = KP2_j;
   - otherwise: k_j = Σ_{n=0..4} d_jn M_r^n.

   The window, j = 5, has κ_g,5 = 0 (FDS's KP1_5 = KP2_5 = 0).

   | j | KP1_j (M_r < 0.01) | KP2_j (M_r > 4) |
   |---|---|---|
   | 1 | 3.388079E-2 | 7.703541E-2 |
   | 2 | 4.544269E-1 | 8.242941E-1 |
   | 3 | 4.680226 | 6.854761 |
   | 4 | 1.038439E2 | 6.593653E1 |

   | j | d_j0 | d_j1 | d_j2 | d_j3 | d_j4 |
   |---|---|---|---|---|---|
   | 1 | 0.0340429 | 0.0652305 | -0.0463685 | 0.0138684 | -0.0014450 |
   | 2 | 0.3509457 | 0.7465138 | -0.5293090 | 0.1594423 | -0.0166326 |
   | 3 | 4.5707400 | 2.1680670 | -1.4989010 | 0.4917165 | -0.0542999 |
   | 4 | 109.81690 | -50.923590 | 23.432360 | -5.1638920 | 0.4393889 |

3. The weights, with t = T / 1200 K, for j = 1 to 4 (`radi.f90` 5054 to 5105):
   - M_r < 0.01: a_j = Σ_{m=0..4} b1_jm t^m;
   - M_r > 4: a_j = Σ_{m=0..4} b2_jm t^m;
   - otherwise: a_j = Σ_{m=0..4} (Σ_{n=0..4} c_jmn M_r^n) t^m;
   - the window: a_5 = 1 − (a_1 + a_2 + a_3 + a_4).

   No temperature is clamped, as in FDS.

   | j | b1_j0 | b1_j1 | b1_j2 | b1_j3 | b1_j4 |
   |---|---|---|---|---|---|
   | 1 | 8.425766E-1 | -1.442229E+0 | 1.286974E+0 | -5.202712E-1 | 7.581559E-2 |
   | 2 | -3.023864E-2 | 5.264245E-1 | -6.209696E-1 | 2.704755E-1 | -4.090690E-2 |
   | 3 | 1.070243E-1 | -1.989596E-1 | 3.101602E-1 | -1.737230E-1 | 3.081180E-2 |
   | 4 | 3.108972E-2 | 1.981489E-1 | -2.543676E-1 | 1.061331E-1 | -1.498231E-2 |

   | j | b2_j0 | b2_j1 | b2_j2 | b2_j3 | b2_j4 |
   |---|---|---|---|---|---|
   | 1 | 7.129509E-1 | -1.378353E+0 | 1.555028E+0 | -6.636291E-1 | 9.773674E-2 |
   | 2 | 1.589917E-1 | 5.635578E-2 | 2.666874E-1 | -2.040335E-1 | 3.742408E-2 |
   | 3 | -1.196373E-1 | 1.349665E+0 | -1.544797E+0 | 6.397595E-1 | -9.153650E-2 |
   | 4 | 3.078250E-1 | -6.003555E-1 | 4.441261E-1 | -1.468813E-1 | 1.824702E-2 |

   | j | m | c_jm0 | c_jm1 | c_jm2 | c_jm3 | c_jm4 |
   |---|---|---|---|---|---|---|
   | 1 | 0 | 0.7412956 | -0.5244441 | 0.5822860 | -0.2096994 | 0.0242031 |
   | 1 | 1 | -0.9412652 | 0.2799577 | -0.7672319 | 0.3204027 | -0.0391017 |
   | 1 | 2 | 0.8531866 | 0.0823075 | 0.5289430 | -0.2468463 | 0.0310940 |
   | 1 | 3 | -0.3342806 | 0.1474987 | -0.4160689 | 0.1697627 | -0.0204066 |
   | 1 | 4 | 0.0431436 | -0.0688622 | 0.1109773 | -0.0420861 | 0.0049188 |
   | 2 | 0 | 0.1552073 | -0.4862117 | 0.3668088 | -0.1055508 | 0.0105857 |
   | 2 | 1 | 0.6755648 | 1.4092710 | -1.3834490 | 0.4575210 | -0.0501976 |
   | 2 | 2 | -1.1253940 | -0.5913199 | 0.9085441 | -0.3334201 | 0.0384236 |
   | 2 | 3 | 0.6040543 | -0.0553385 | -0.1733014 | 0.0791608 | -0.0098934 |
   | 2 | 4 | -0.1105453 | 0.0464663 | -0.0016129 | -0.0035398 | 0.0006121 |
   | 3 | 0 | 0.2550242 | 0.3805403 | -0.4249709 | 0.1429446 | -0.0157408 |
   | 3 | 1 | -0.6065428 | 0.3494024 | 0.1853509 | -0.1013694 | 0.0130244 |
   | 3 | 2 | 0.8123855 | -1.1020090 | 0.4046178 | -0.0811822 | 0.0062981 |
   | 3 | 3 | -0.4532290 | 0.6784475 | -0.3432603 | 0.0883088 | -0.0084152 |
   | 3 | 4 | 0.0869309 | -0.1306996 | 0.0741446 | -0.0202929 | 0.0020110 |
   | 4 | 0 | -0.0345199 | 0.2656726 | -0.1225365 | 0.0300151 | -0.0028205 |
   | 4 | 1 | 0.4112046 | -0.5728350 | 0.2924490 | -0.0798076 | 0.0079966 |
   | 4 | 2 | -0.5055995 | 0.4579559 | -0.2616436 | 0.0764841 | -0.0079084 |
   | 4 | 3 | 0.2317509 | -0.1656759 | 0.1052608 | -0.0321935 | 0.0033870 |
   | 4 | 4 | -0.0375491 | 0.0229520 | -0.0160047 | 0.0050463 | -0.0005364 |

   These are the 170 numbers of FDS 6.11.1's `MAKE_WSGG_ARRAYS` (`radi.f90` 3223 to 3270; tag FDS-6.11.1, commit ff928db), which cites Bordbar et al. (2014). They are identical at FDS master 5f8f194, the commit 13.11 used. They were parsed from that file's text for this draft, not retyped. c_jmn is FDS's `WSGG_C_ARRAY(j, m, n)`: m the power of t, n the power of M_r.

4. Soot. Each band's coefficient is κ_j = κ_g,j + 500 Y_s, and the window's is 500 Y_s. The soot is added to every band, as FDS adds its soot to every WSGG band (`radi.f90` 3996 to 3997). The soot's coefficient stays the solver's, 500 Y_s; this item changes nothing about soot. (FDS's WSGG option uses another soot coefficient, `KAPPA_SOOT`, 5108 to 5114; it is not taken.) With no CO2 and no H2O the five bands share one coefficient and the weights sum to 1, so emission and absorption reduce to the solver's present soot terms.

5. Emission. A gas cell emits in band j, over the step,

   E_j = 4 κ_j σ [a_j(T, M_r) T⁴ − a_j(T_a, M_r) T_a⁴] V dt,

   FDS's band source a_j κ_j 4σT⁴ (`radi.f90` 3999) less what the same gas would emit at ambient temperature. The solver writes all its emission net of the ambient's (T⁴ − T_a⁴), and the item keeps that form. Taking a_j at T_a with the cell's own M_r keeps every band at zero in a domain at ambient temperature, whatever its composition. FDS instead gives its open boundaries the ambient air's weights (`radi.f90` 3849 to 3858). With a uniform M_r the two agree.
   - A band whose E_j comes out negative emits nothing, and the omission is counted (`stats.radNegative`). Hand calculation: no band is negative for T_a < T ≤ 2600 K at any M_r from 0 to 10⁶; bands 2 and 4 turn negative above about 2650 K.
   - The cell loses Σ E_j, capped as now at half its heat above ambient (851), every band scaled alike; the cap is counted (`stats.radCapped`).
   - The floor. With the item on, a gas cell emits when T > T_a + 10 K (`gasEmitRise` = 10). This is a numerical cut fixed here, from no source, so that cells at ambient cast no rays. The emission it omits, Σ E_j of the cells below it, is computed without casting rays and counted (`stats.radFloor`). It applies to sooty gas too, which now emits nothing below 500 K. Surfaces keep `emitAbove` (500 K) unchanged.

6. Where a gas cell's rays start. Each of its `rays` (8) rays carries E_j / 8 in every band j. A ray starts at a point drawn uniformly in the cell (three more draws from the cell's stream, after the two for its direction). It is attenuated from there through its own cell like any other, and its own cell takes what it absorbs. Now a ray starts at the centre and skips its own cell (936).
   - Why. k_4 is 63 to 104 atm⁻¹ m⁻¹, so at X_c + X_w = 0.15 one cell of 0.2 to 0.4 m has κ_4 h = 2 to 4. Hand calculation for a semi-infinite isothermal gas: rays from cell centres that cross their own half cell give 0.64 of the exact wall flux at κh = 2, and rays that skip their own cell about 4.6 times it. A uniform start is the cell's exact volume emission; on a gray cube with κh from 0.05 to 2 it was unbiased within its noise (hand calculation).

7. Absorption along a ray. A ray carries five energies e_1 to e_5. In each gas cell it crosses for a length ℓ, each loses e_j (1 − exp(−κ_j ℓ)) with that cell's κ_j, and the cell takes the sum. A burner's flame cell (item 9 below) is transparent. The first solid or porous cell takes Σ e_j, as now.
   - A ray marches until it meets a solid or leaves the grid (at most n_x + n_y + n_z crossings); `rayReach` does not stop it. Now a ray that has crossed 40 cells drops what is left into the gas there (966 to 972). In a soot-free room the window band would then heat gas that cannot absorb it.
   - The march stops when Σ |e_j| falls below 10⁻⁹ of its value at the start.

8. Surfaces. A solid or porous cell's emission is as now: ε σ (T_s⁴ − T_a⁴) per open face, above 500 K. It is shared among the bands by

   w_j = [a_j(T_s, M_r) T_s⁴ − a_j(T_a, M_r) T_a⁴] / (T_s⁴ − T_a⁴),

   with M_r of the gas across the face the ray leaves by (a porous cell's own gas; beyond the grid's side, M_r = 0). FDS weights a wall's emission by a_j at the wall's temperature and the adjacent cell's M_r (`radi.f90` 4155 to 4163). The w_j sum to 1, so a surface emits what it does now. A negative w_j is set to 0, the rest are scaled to keep the sum, and it is counted. A surface still takes all of each ray that reaches it, whatever its emissivity.

9. The burner, and the radiant fraction counted once. FDS puts all of a reaction's heat into the gas. In the cells where χ_r q̇''' > 1 kW/m³ it scales the emission so that the flaming region's net emission is χ_r times its heat release (Technical Reference Guide, the corrected emission source term, its factor bounded by 0.1 and 100; `radi.f90` 4083). Here the burner casts χ_r of the heat burnt itself (3361 to 3376). So, with the item on:
   - a cell where a burner with χ_r > 0 burnt fuel this step (`burnInCell` returned more than 0) neither emits nor absorbs in that step's radiation stage. The flaming region's net radiation is then exactly the χ_r cast, which is what FDS's correction aims at. The emission those cells would have had is counted (`stats.flameSilenced`). This also ends the double count of qualification 3, for sooty burners;
   - the burner's rays carry its radiant part in five bands, shared by the heat-weighted mean of a_j(T_c, M_r,c) over the cells that burnt, each taken after its burning. FDS shares χ_r q̇''' among its bands by each flaming cell's a_j (`radi.f90` 4002). The rays start at the axis cell's centre as now;
   - a burner with χ_r = 0 (the design-fire item, the app's) is not affected. Its flame cells emit by their own gas and soot.

10. Energy. All that a cell or a burner emits ends in a cell, a wall's end, or beyond the grid's sides. New tallies, in J:
    - `stats.radiatedOut`: what rays carry out through the grid's sides, the burner's included;
    - `stats.radFloor`, `stats.flameSilenced`, `stats.radNegative` and `stats.radCapped`, as above.

    The ledger (13.15) gains `radiatedOut` and keeps `radiated`. Without it, the radiation that leaves the grid sits in the ledger's unexplained remainder. It is a diagnostic: it is compared with nothing and decides nothing, as 13.15 says.

11. On the CPU (`flamescapes.js`):
    - `DEFAULTS`: `gasRadiation` (0 off, 1 on) and `gasEmitRise` (10 K). With 0 every path is as now, bit-identical.
    - The 170 constants and the functions a_j(T, M_r) and k_j(M_r), written from `radi.f90`'s text.
    - `radiation`: when on, a first pass over the gas cells (X, M_r, p_a and κ_g,1..4 into work arrays); then the five-band emission with the floor, the cap and the flame cells; the random starts; a surface's emission shared by w_j; the tallies.
    - `marchBands`, beside `march`: five energies, the own cell absorbing, no reach limit, the outflow counted. `march` itself is unchanged and serves the item off.
    - `burnerStep`: stamps the cells that burnt with the step's number, and when on shares its radiant part by band and casts it through `marchBands`.

12. On the GPU (`flamescapesGpu.js`, `kernels.js`):
    - Parameters: the flag and the floor; the 170 constants as WGSL constants.
    - `radKappa`, one thread a cell, before `radEmit`: X, M_r, p_a and κ_g,1..4 into a vec4 buffer, M_r into an f32 buffer (20 bytes a cell).
    - `radiateCell` and `march`: as the CPU's, five energies (a vec4 and a scalar), the same draws in the same order from the same stream.
    - Deposits in joules, not kelvin, when on: 4096 steps a joule, so 0.24 mJ is the smallest step and 524 kJ the most a cell can take in a step. A single deposit above 4096 J is clamped and counted. `radApply` divides by each cell's heat capacity (the gas's ρ c_p V, `solidCap`, a wall's end as `wallEnd` takes it). Why: in kelvin, a ray's 0.4 J reaching a stone cell of 0.027 m³ (58 kJ/K) is 7 × 10⁻⁶ K, under half a step of 1/16 384 K, and would round to nothing every step (hand calculation). Off, the deposits stay in kelvin.
    - `radEmit`'s reduction carries the outflow beside the emission (its vec4's second slot), read back with `radiated`.
    - A validation burner stays on the CPU (`supported`, unchanged), so item 9 is CPU only.

#### Where it can fail

- (b) can fail on the model rather than the code. Bordbar et al. fitted their model for oxy-fired combustion (its title). FDS applies it to products diluted in nitrogen, and so does this item; whether the fit holds there was not verified for this draft, since the paper was not read. RadCal's water vapour data are Ludwig et al.'s (1973), and TN 1402 shows RadCal and Hottel's charts for water vapour 12 to 25 % apart over most of 400 to 1500 K, and up to about a third at the smallest pressure path lengths. So (b) asks for agreement within the spread of accepted spectral models, and no closer.
- (c) to (e) and (h) can fail on the transport: the noise of eight rays a cell, the start points, the layer's edge on coarse cells, a surface's band shares.
- In use, the gas floor and the unlimited reach change cost and accuracy beyond what the checks hold. Both are counted in every run.

#### What the checks read

A new tool, `tools/gas-radiation-check.mjs`. It builds each scene, sets the gas and the walls to their temperatures and compositions, and calls the radiation stage alone (exported for the tool) for 200 steps of 0.05 s with `seed` fixed, setting every temperature back after each step. Each step it reads, in J: the emission by band and by source (gas, surfaces, burner); what the gas absorbs; what each group of solid cells absorbs (floor, ceiling, each side wall, from the deposits before they are applied); and `radiatedOut`. All walls in (c) to (e) and (h) have ε = 1 and are lumped unless stated. No measured data enters any check.

#### What is fixed in advance, and from where

- The 170 constants, t = T / 1200 K, the bounds 0.01 and 4 on M_r, p_a in atmospheres of 101 325 Pa, and the zero below 4.44 × 10⁻¹⁵: FDS 6.11.1 `radi.f90`, citing Bordbar et al. (2014).
- The soot's coefficient: unchanged, 500 Y_s m⁻¹ (line 162).
- The emission net of the ambient's, with a_j at T_a and the cell's M_r; a surface's weights at its temperature and the adjacent gas's M_r; the burner's band shares: points 5, 8 and 9, after FDS's practice where FDS has one.
- The gas floor T_a + 10 K, 8 rays a gas cell (as now), no reach limit, uniform start points: numerical choices fixed here.
- The GPU's 4096 steps a joule.
- The references: RadCal at github.com/firemodels/radcal commit abe2a8f (its `rcal.f90` is FDS's RadCal), built from its source with any Fortran 2008 compiler and run from its own input file; and FDS's `RADTMP`, 1173.15 K (`read.f90` 10092 and 10118), as the flame's temperature in (b)'s absorptivities.

Nothing is fitted, and no series that 12 or 13 holds out is opened (14.1; 13.12, rule 4).

#### Acceptance (fixed now, before any run)

(a) Constants and functions. The solver's 170 constants equal those the tool parses from FDS 6.11.1's `radi.f90` (3231 to 3269), digit for digit. The solver's a_j and k_j equal the tool's own transcription of `A_WSGG` and `KAPPA_WSGG` within 10⁻¹² relative on the CPU and 10⁻⁵ on the GPU, at T from 293.15 K to 2493.15 K in 50 K steps and M_r ∈ {0, 0.005, 0.01, 0.5, 1, 1.25, 1.9, 3.99, 4, 4.01, 10}. Reason: this catches transcription; float32 carries about seven digits.

(b) Emissivity and absorptivity against RadCal. A homogeneous, isothermal column at 1 atm: X_c + X_w ∈ {0.03, 0.15}; M_r ∈ {1.25, 1.90}, the ratios in the solver's products of its class A fuel and of methane (from `stoichiometry`); O2 and N2 the rest, in air's ratio; no soot; length L ∈ {0.3, 1, 3} m.
   - Emissivity at T ∈ {400, 600, 900, 1200} K, 48 cases: the solver's Σ_j a_j(T, M_r)(1 − e^{−κ_j L}) against RadCal's π I / (σ T⁴), I the radiance leaving the column with `TWALL` at 0 K (or the lowest RadCal takes), over 50 to 10 000 cm⁻¹.
   - Absorptivity at T ∈ {400, 600} K for black radiation from 1173.15 K, 24 cases: the solver's Σ_j a_j(1173.15 K, M_r)(1 − e^{−κ_j L}) against 1 − [I(`TWALL` 1173.15 K) − I(`TWALL` 0 K)] / (σ · 1173.15⁴ / π) from RadCal.
   - Passes if every case is within 25 % of RadCal's value or 0.01, whichever is larger. Reason: 25 % is about the largest gap TN 1402 shows between RadCal and Hottel's charts for water vapour over most of 400 to 1500 K (its Figs. 3 and 4), that is, how closely two accepted spectral models agree there; where those figures differ by more (the smallest pressure path lengths), the absolute 0.01 applies. 0.01 of emissivity is 1 % of black-body exchange, below anything the validation resolves.

(c) Gray transport in a cube. Soot only (X_c = X_w = 0), uniform, at 1000 K, in a cube of gas 3.0 m a side inside black walls at ambient, on 0.2 m and 0.3 m cells, Y_s set so that κL = 0.1, 0.5, 1, 2 and 5. The power into the walls, the mean over 200 steps, against 4κσ(T⁴ − T_a⁴) V P(κL). P is the exact escape fraction of a uniform cube, which the tool computes by reciprocity: P(τ) = (6 / 4τ) ⟨1 − e^{−τℓ}⟩ over one face with cosine-weighted directions, ℓ the chord in the unit cube, by quadrature fine enough that halving its steps moves P by less than 0.05 %. Passes within 2 % in every case. Reason: a hand calculation of this method (uniform starts) on this cube is unbiased within its noise, and 200 steps of 8 rays from 1000 to 3375 cells hold the noise under 0.5 %.

(d) Band transport in a cube. As (c), with X_c = 0.05 and X_w = 0.10 (M_r 2), no soot, at 600 K and 1000 K, in cubes of 1.2 m and 3.0 m. The reference: Σ_j 4 κ_j σ [a_j(T) T⁴ − a_j(T_a) T_a⁴] V P(κ_j L). Within 2 % in every case. Reason as (c). Band 4's κh of 2 to 4 is where the start points matter.

(e) A soot-free layer heats a surface it does not touch. A box of 3.0 × 2.4 × 3.0 m (13.17's) on 0.2, 0.27 and 0.34 m cells, each box the whole cells nearest those sizes. Its upper 1.2 m (the whole rows nearest) holds gas at 600 K with X_c = 0.05 and X_w = 0.10; below it, air at ambient; black walls at ambient.
   - Read: the power into the floor and into the ceiling.
   - Reference: Σ_j σ [a_j(T) T⁴ − a_j(T_a) T_a⁴] ∫_A ∫ (1 − e^{−κ_j ℓ}) (cos θ / π) dΩ dA over the floor (or the ceiling), ℓ the chord through the layer as the grid holds it, by quadrature as in (c).
   - Passes if both are within 3 % in every case, on lumped walls and on one-dimensional walls (13.4, item 1: a 16 mm gypsum board), whose power is read from their ends' deposits. Reason: (c)'s 2 %, and 1 % for the layer's edge.
   - With the item off the floor takes nothing. Hand calculation on the exact box: the floor 9.5 kW (1.06 kW/m²), the ceiling 20.8 kW, against the ceiling's convection at 10 W/m²K, 27.6 kW.

(f) Energy. In (c) to (e), and in two burner runs: a 100 kW burner, 0.3 m square, at the centre of (e)'s floor, the box with a door 0.8 × 2.0 m to open cells beyond, 0.2 m cells, 120 s, the flow on; one with methane at χ_r = 0.2, one with the class A fuel (soot yield 0.015) at χ_r = 0.2. In every step, what was emitted (gas, surfaces, burner) equals what the gas, the solids and the walls' ends took plus `radiatedOut`, within 10⁻⁹ of the emission on the CPU (float64 sums) and within 1 % on the GPU (fixed point), and the GPU clamps no deposit. Reason: conservation holds by construction; 1 % is 13.4's closure, far above the GPU's rounding.

(g) The radiant fraction counted once. In both burner runs, in every step: the cells that burnt emit and absorb nothing, and the burner's rays carry χ_r of the heat burnt within 10⁻¹² relative. `stats.flameSilenced` is reported; for the sooty fuel it is the double count of qualification 3.

(h) An isothermal box stays isothermal. (d)'s 3.0 m cube with its gas (soot Y_s = 10⁻³ added) and its black walls all at 600 K. Over 200 steps the gas's net radiative gain is within 2 % of its emission. Reason: the bands' sources and the surfaces' weights cancel exactly in the continuous equations, and 2 % is (c)'s allowance for the discrete ones.

(i) Off is off. With `gasRadiation` 0 the unit suite and the GPU parity phases are bit-identical to the build before.

(j) The unit suite passes unbent with the item on.

(k) The ledger. In both burner runs, at each of the ledger's times, the unexplained remainder, released + pilot − (gas + walls + shell + leak + `radiatedOut`), is no larger in magnitude with the item on than with it off, plus 1 % of the heat released. Reason: the item moves heat between terms the ledger holds; if the remainder grows, heat has gone where the ledger cannot see.

Reported, not judged: the optically thin limit, the solver's Σ_j a_j k_j against RadCal's Planck mean at the same T and M_r; `radFloor`, `radNegative`, `radCapped` and the mean crossings a ray, in every case and in the bench runs; the ceiling's radiative and convective gains in (e).

#### GPU parity

- The item: `tools/flamescapes-gpu-check.mjs` gets a phase `gasRadiation`. (c), (d), (e) and (h) run on the GPU's radiation passes alone. Each passes as on the CPU, and each power is within 0.95 to 1.05 of the CPU's (14.2's ratio, applied to this check). From one state, a step's emitted, gas-absorbed, solid-absorbed and outgoing totals agree with the CPU's within 1 %. Then 200 deterministic steps, with the flow on, of (d)'s 3.0 m cube at 600 K and of the app's design-fire item in (e)'s box: |CPU − GPU| for gas T, ρ and O2 against the one-ulp twin's, as items 5 and 7 reported.
- The burner runs of (f), (g) and (k) are CPU only: the GPU hands a validation burner to the CPU.
- The rerun's verdicts on the GPU follow 14.2 as written.

#### Speed (14.6)

Milliseconds a step and run time on Duplex, Office and Clinic, CPU and GPU (`tools/backend-bench.mjs`), with the item on and off, against the build before. If on costs more than 10 % a step on the GPU, it goes behind a switch, and the app states the verdicts found without it when it is off, which means a second full run. The CPU's cost is reported beside it, with the number of emitting gas cells and the mean crossings a ray.

#### Default

Off (`gasRadiation` 0) until (a) to (k) pass and the unit suite passes unbent with it on. Then on, unless 14.6 puts it behind a switch. Made after 13.B began, it gets its own full rerun, reported beside 13.B (13.12, rule 5).

#### What ends this line

If the item fails any of (a) to (k) in any case, it is not made. The gas stays transparent but for its soot, and the app states the verdicts as 13.A and 13.B find them. A failure of (b) alone says that the fit, not the code, departs from RadCal. A second attempt with FDS's default gray gas (RadCal's coefficients at a path length, with FDS's correction in flame cells) would then need another independent reference, since RadCal could no longer serve as one. No second attempt is drafted. It needs the user's word and a new method committed here first.

#### Sources (read, or metadata only)

- FDS 6.11.1, tag FDS-6.11.1, commit ff928dbf68a2623ced3a007be75391c6dc6c9f90, fetched from GitHub raw. Read: `Source/radi.f90` (WSGG constants 3223 to 3270, `KAPPA_WSGG` 5010 to 5049, `A_WSGG` 5054 to 5105, `KAPPA_SOOT` 5108 to 5114, WSGG emission 3973 to 4004, surfaces 4155 to 4163, ambient 3849 to 3858, gray path length 3134, correction 4083); `Source/read.f90` (`RADI` defaults, 10070 to 10118); `Source/cons.f90` (370, 486 to 491); `Source/prec.f90` (40); the Technical Reference Guide's `Radiation_Chapter.tex` and `Appendices.tex` (gas phase absorption coefficients); the User Guide (`FDS_User_Guide.tex` 5879 to 5920); the Verification Guide's radiation cases. The WSGG constants and functions are identical at master 5f8f194.
- Bordbar, Węcel and Hyppänen (2014). A line by line based weighted sum of gray gases model for inhomogeneous CO2–H2O mixture in oxy-fired combustion. Combustion and Flame 161, 2435 to 2445. doi:10.1016/j.combustflame.2014.03.013. Checked on Crossref; metadata only. Its constants are taken from FDS's source, which cites it.
- Grosshandler (1993). RADCAL: a narrow-band model for radiation calculations in a combustion environment. NIST Technical Note 1402. doi:10.6028/NIST.TN.1402. Checked on Crossref; read (sections I to III, Table 1, Figs. 1 to 4).
- RadCal, github.com/firemodels/radcal, commit abe2a8f1e84ef0b3afda73d0c08d5c2e2964e401. Read: `Documentation/Using_RADCAL_Chapter.tex` and `Validation_Tests_Chapter.tex`. Not built or run.
- Considered, not used, each checked on Crossref, metadata only: Smith, Shen and Friedman (1982), Journal of Heat Transfer 104, 602 to 608, doi:10.1115/1.3245174; Leckner (1972), Combustion and Flame 19, 33 to 48, doi:10.1016/S0010-2180(72)80084-1; Barlow, Karpetis, Frank and Chen (2001), Combustion and Flame 127, 2102 to 2118, doi:10.1016/S0010-2180(01)00313-3.

#### The coordinator's decisions on this draft (2026-11-13, 04:42, before any run of it)

The draft above is the physics agent's, written blind (13.12, rule 2). The coordinator has read 13.A and 13.B's first files, and knows that the rooms read hot, so the decisions below add no constant to the physics and move no tolerance. Each takes the draft's own choice, or the reading that needs no one's word.

**Checked by the coordinator** (apart from the agent, from FDS's sources fetched again at tag FDS-6.11.1, commit ff928db): the 168 numbers of the draft's five tables against `MAKE_WSGG_ARRAYS`, with a parser of the coordinator's own (no mismatch; the two window coefficients are 0 there); `KAPPA_WSGG` and `A_WSGG` as the draft's points 2 and 3 write them, the bounds 0.01 and 4, 1200 K and the window's weight included; the molar ratio, the pressure of the absorbing gases, soot added to every band and the band's source (`radi.f90`, 3990 to 3999); a wall's weights at the adjacent cell's molar ratio (4155 to 4163); `TWENTY_EPSILON_EB` and `P_STP`. In the solver at 550fce5: the gas emits by soot alone above 500 K, a ray skips its own cell, is absorbed by soot alone, is lost uncounted at the grid's edge and is dropped after 40 crossings, and the burner casts its radiant part itself. Three hand calculations were repeated from the coordinator's own transcription and agree: the emissivities at 500 K (0.091, 0.284, 0.537), the thin-limit coefficients (13.4, 11.9, 7.8), and the slab's 2.78 kW/m² against 3.07 kW/m² of convection. Not checked by the coordinator: TN 1402's figures, from which the draft reads its 25 %.

1. **The name.** Item 8 here is the eighth physics item, after 13.11's item 7. The comparison that 13.2 and 13.7 call item 8 (FSE 2008) belongs to another list.
2. **The reference for (b) is RadCal, as drafted, and nothing replaces it.** This machine has no Fortran compiler. The coordinator tries one route that changes nothing outside a throwaway folder: gfortran from conda-forge, fetched with the micromamba already on the machine into a folder outside the repository, and RadCal at abe2a8f built from its source with it. If RadCal cannot be built that way, (b) waits for the user's word on a compiler. Until (b) has run, the item is not made, its default stays off, and no validation run is made with it on. No port of RadCal and no other reference is taken in its place.
3. **(b)'s tolerance stands as drafted** (25 %, or 0.01), fixed here before RadCal is built or run. It is the agent's reading of TN 1402's Figs. 3 and 4, made blind.
4. **The scope stands as drafted.** The flame cells' silence, the uniform starts with the own cell absorbing, and the unlimited reach belong to the item, on only when the item is on. `stats.flameSilenced` shows the size of the first in every run.
5. **The numerical choices stand:** the gas floor at T_a + 10 K, 8 rays a gas cell, 4096 steps a joule on the GPU. If 14.6's cost calls for fewer rays, a higher floor or a reach limit, that is a new method under 13.12, drafted and committed before it runs.
6. **The pressure and the ambient weights stand as drafted** (points 1 and 5): P_AMBIENT, as the solver's equation of state has it, and the weights at ambient temperature with the cell's own molar ratio. (h) tests the second.
7. **A declared limit, kept and measured.** Surfaces keep their 500 K floor, so a wall between ambient and 500 K takes the gas's radiation and returns none. The gas's net loss to such a wall is then overstated: with the gas at 600 K and the wall at 400 K, by the factor (600⁴ − 293.15⁴) / (600⁴ − 400⁴) = 1.18. That bias cools the layer, which is the direction the rooms' reading would welcome, so it is stated here before any run and it is not left unseen: the solver counts what surfaces between T_a + 10 K and 500 K would have emitted (`stats.surfaceFloor`, in J, computed without casting rays, when the item is on), and the check tool and every validation run report it beside `radiated`. It is compared with nothing and decides nothing. A surface floor at T_a + 10 K is a candidate for its own item, drafted blind, and the rerun's reading carries this limit in words until then.
8. **A surface takes all of each ray and emits at its ε** (0.9 for the inert materials): the solver's present rule, kept. The checks use ε = 1, as drafted.
9. **The GPU's draws.** The draft could not settle whether the GPU's directions come from the cell's stream or from a hash of its own. Whichever the code has at the base is kept; where the draft says "the same draws in the same order from the same stream", the GPU uses its own stream's draws in that order. Parity is judged on the totals and the powers as drafted, which need no shared draws.
10. **The base** is `next/calibration-ui` at the commit the item's branch starts from, whose solver is 13.B's (550fce5). "The build before" in (i), (k) and 14.6 is that base.
11. **The reruns.** Each item made after 13.B began gets its own full rerun on 13.B's base, reported beside 13.B, before any two are combined (13.12, rule 5), so each one's effect is read alone. Their order is set when the builds have reported their checks.
12. **What ends this line stands as drafted.**

### 13.20 Items 4 and 3, attempt 2: the result (2026-11-13, 04:59)

**Attempt 2 fails. Items 4 and 3 are not made, and doors stay as the voxeliser makes them** (13.17, "What ends this line"). Door and vent velocity, door mass flow and the neutral plane stay as 13.A and 13.B find them. No third attempt is drafted; a further attempt needs the user's word and a new method committed here first.

The build is on branch `flamescapes/door-attempt2` (2948f2d the vent on the CPU, off by default; bc1cf65 and 290da56 the check, `tools/calibration/door-vent.mjs`). It is not merged. Steps 1 to 12 of 13.17 were built; the GPU port (step 13), the parity phase and the speed bench were not, because the CPU failed first. The coordinator ran the committed check again at 290da56 and got the agent's figures to the last digit. Checks (h) and (i) are the agent's and were not repeated, since nothing is merged.

**(a) to (d), two sealed boxes on 0.2 m cells.** (d)'s figure is each box's heat error as a share of the heat the larger direction carried above ambient.

| Door | Hot box | (a) outflow, 10 % | (b) neutral plane, 10 % | (c) mass, 1 % | (d) heat, hot and cold box, 1 % |
|---|---|---|---|---|---|
| 0.4 m | 400 K | +3.8 % | −3.2 % | 0.00 % | +39.8, −29.8 %: fails |
| 0.5 m | 400 K | +3.8 % | −3.3 % | 0.00 % | +40.9, −32.1 %: fails |
| 0.6 m | 400 K | +3.9 % | −3.5 % | 0.00 % | +41.9, −34.2 %: fails |
| 0.4 m | 600 K | +15.2 %: fails | −7.0 % | 0.00 % | +52.0, −27.5 %: fails |
| 0.5 m | 600 K | +16.3 %: fails | −7.1 % | 0.00 % | +51.5, −30.7 %: fails |
| 0.6 m | 600 K | +17.6 %: fails | −7.0 % | 0.00 % | +50.6, −33.6 %: fails |

- **(a)** passes at 400 K and fails at 600 K in every case. **(b)** passes in all six. **(c)** passes in all six, exactly, as the method builds it.
- **(d)** fails in every case, under every reading of "each way" (against the larger direction, the smaller, or their sum).
- **(e)**, a zone against open cells, 0.5 m door: (a) +3.9 % at 400 K, +15.6 % at 600 K. Fails at 600 K.
- **(f)**, the six cases on 0.27 m and 0.34 m cells: (a) +2.5 to +3.2 % and −0.5 to +1.0 % at 400 K; +14.0 to +16.8 % and +11.7 to +13.8 % at 600 K. (b) within −2.7 to −6.2 %. (d) +37 to +52 % and −18 to −29 %. Fails on (a) at 600 K and on (d) everywhere.
- **(g)**, three boxes: each box's mass closes on its vents' transfers within 0.0002 % (passes); Newton agrees with bisection within 3.9e-7 Pa (passes). The zones' pressure against their mass, 0.01 Pa: up to 217 Pa on the first step, whose transfers were capped, and up to 1.8 Pa (0.2 m) and 2.6 Pa (0.34 m) on later steps. Fails as worded. The solve itself converged at every step (largest last change 0.0098 Pa); 13.17's wording is not bent to that reading.
- **(h)**, off is off: the agent reports identical state fingerprints against the base in the six grid runs, a unit room for 30 s and a leaky sealed box with a burner, 536 of 536 unit tests, and identical GPU parity output. **(i)**: 536 of 536 with the vent on by default, but the check could not fail: no unit scene lists an opening, so no vent is made in any of them.

**What the agent found about the cause** (its reading, reported and not acted on):
- Measured: Emmons' integral on the vent's own band rows, with the sill pressure from a volume balance, is within 2.1 % of the reference; at the vent's own solved sill pressure it is +3.8 to +17.6 %. So the band profiles hold and the joint solve's sill pressure is what fails (a). In the sealed pair the gas's total heat over 2 to 6 s rises 19.8 kJ (400 K) and 150.9 kJ (600 K) with the vent, against 3.0 and 7.8 kJ with the door on the grid, and both zones drift to about +1.7 kPa at 600 K.
- Inferred, not proven: the gas a vent moves enters a zone twice, once through the ledger's compression and once through the expansion sources (13.17, steps 9 and 11 ask for both), and the transport then takes the duplicate out at fixed temperature, which would explain the heat gained; and step 8 counts transferred gas at the zone's mean density, which would bias the sill pressure towards the hot side.

**Two things any further attempt must face,** found by this one and outside its checks:
- On the app's own models almost no door becomes a vent. Taking each IFC door's bounds as an opening: the Duplex gets 1 vent of 14 doors at 30 000 cells and none at 120 000; the Office none of 102; the Clinic none of 254. Most doors sit at the grid's edge or outside it, or their wall is more than one cell thick, so a row of face cells is solid. The method assumes a wall one cell thick.
- The app does not yet pass a door's or window's bounds to the solver (`scene.openings` comes only from the tools), and no unit scene has one.

**Where 13.17 was unclear, and what was taken:** (d)'s "each way" (every reading fails); (g)'s 0.01 Pa (read as worded, fails); the leak's derivative, floored at 0.01 Pa as the vent's is; a run's first step has no vent speed for the step's length, so its transfers are capped. None of these changes a verdict.

### 13.21 Item 9: the voxeliser snaps to the nearest plane (2026-11-13, before any run of it)

Drafted blind to every validation result, as 13.12 sets. Nothing below changes after a run of it; anything added later is dated, with its commit. Line numbers are those of commit 191b1d3, which the working tree's `voxelize.js` (unchanged since 696683a), `flamescapes.js`, `flamescapesRun.js` and `tools/calibration/lib/fds.mjs` equal. The numbers called "present rule" below come from the present voxeliser run on synthetic boxes for this draft. None is a fire result.

#### What the voxeliser does now

Volume fractions (`voxelize.js` 122 to 195).
- Rays run along the up axis only, S × S of them per column of cells. S is `raysFor`, max(2, ⌈h / 0.05 m⌉) (257 to 259): 4, 6 and 7 on 0.20, 0.27 and 0.34 m cells. Sub-ray s sits at (s + 0.5)/S + 0.0137/S of the cell (129).
- A mesh's crossings with a ray are sorted, merged within 10⁻⁶ m (170) and paired: inside between the 1st and 2nd, the 3rd and 4th, and so on. An odd last crossing is dropped (175).
- A cell's fraction is the length inside over S² h (183). It is exact along the up axis and sampled in steps of 1/S across it. A total above 1 is scaled to 1 (189 to 193).

The solid mask (`occupancy`, 221 to 254). With v the cell's total fraction:
- v ≥ `SOLID_FRACTION` = 0.5 (40): solid (232).
- `THIN_FRACTION` = 0.08 (205) ≤ v < 0.5: solid if, along some axis, v is at least each of its two neighbours (233 to 238).
- 0 < v < 0.08: solid if, along some axis, the cell and its larger neighbour hold 0.08 between them, that neighbour holds less than 0.08, and the cell holds more than it, the lower of an equal pair (239 to 248).
- A cell with v = 0 is never solid.

The harness (`fds.mjs`).
- `solidBoxes` (283 to 331): the domain is the meshes' box grown by one cell (286). Everything in it outside the meshes is solid. An OPEN vent on a mesh side cuts that solid out to the domain's edge; the floor stays closed. Each obstruction less its holes is a set of boxes (`subtract`, 249 to 260).
- `gridFor` (339 to 356) puts the grid's origin at the domain's lower corner (342), so the meshes' floor lies on a cell plane. It calls `volumeFractions` with `raysFor` and `occupancy` (344 to 345), "as in the app" (335 to 337).
- `treeCells` (377 to 394) and `deviceCell` (396 to 401, 429 to 449) read the mask. `deviceCell`'s comment already allows for "a ceiling snapped down past" a device.

Why every face goes into the room. The test at 235 to 237 uses ≥. A slab's partial face layer has equal neighbours along the slab's own plane. So each of its cells is a "peak" along both in-plane axes once v ≥ 0.08. The rule written for thin elements fires on the face of every slab and wall wider than a cell. Across the up axis, a face seals the cell it enters by 0.08 of a cell or more. Across a horizontal axis, one sub-ray in the material is enough in a wall at least a cell thick (1/S ≥ 0.08): the face then seals a cell it enters by 0.5137/S from one side or 0.4863/S from the other (12.8 %, 8.6 % and 7.3 % of a cell on the three sizes).

The consequences, derived from the code and checked on swept synthetic boxes (200 positions across a cell, each size):
- A face across the up axis (a ceiling, a soffit, a sill, a floor's top) lies on the grid at the plane on the room's side of the cell it enters, unless it enters by less than 0.08. Its error runs from −0.92 to +0.08 cells (negative into the room), mean −(0.92² − 0.08²)/2 = −0.42 cells. Measured: −0.4175, −0.4225 and −0.4225 on 0.20, 0.27 and 0.34 m.
- A face across a horizontal axis (a wall's face, a jamb) of a wall at least a cell thick: mean −0.38, −0.42 and −0.43 cells (measured −0.3775, −0.4175, −0.4325).
- An opening in such a wall loses on average 1 − 1/S cells of width: 0.75, 0.83 and 0.86 cells (measured the same for widths of 1, 2 and 2.74 cells). In a 0.10 m wall it loses 0.75, 0.5 and 0.57 cells, since a jamb's cell then holds only the wall's share. An opening 0.6 of a cell wide is closed at every position.
- A slab thinner than a cell is one or two cells. A 0.25 m slab on 0.27 m cells is two cells at 76 of 100 positions; a 0.20 m slab on 0.34 m cells at 43. A plate thinner than 0.08 of a cell seals nothing: a 2 cm plate on 0.27 and 0.34 m cells left every column open at all 100 positions. This is a hole in the present rule's watertightness (glazing, sheet metal).
- Pockets. In a room whose walls lie on cell planes and whose ceiling enters its top row by 0.3 of a cell, that row is solid except its four corner cells, on all three sizes. A corner cell has a fuller neighbour along both horizontal axes, so it is a peak along none.

The examples, checked by running the present voxeliser through `fds.mjs` `gridFor` on a copy of Steckler_010's geometry (mesh x 0 to 3.6, y −1.4 to 1.4, z 0 to 2.13 m; wall x 2.80 to 2.90 m; doors as its HOLE lines; z 0 is a cell plane):

| Present rule | 0.20 m | 0.27 m | 0.34 m |
|---|---|---|---|
| Room, 2.13 m tall | 2.00 | 1.89 | 2.04 |
| 6/6 door, 0.74 m wide | 0.40 | 0.54 | 0.68 |
| 6/6 door's top, 1.83 m | 1.80 | 1.89, the ceiling | 1.70 |
| 8/6 door, 1.00 m wide | 0.80 | 0.81 | 0.68 |
| 2/6 door, 0.25 m wide | closed | closed | closed |

The examples hold. Corrections and additions: the "0.24 m" door is 0.25 m in FDS's input (y −0.10 to 0.15 m) and is closed on all three sizes. The 1.0 m door that is 0.68 m on 0.34 m cells is Steckler's 8/6 door. On 0.27 m cells the 6/6 door has no lintel: its top is the snapped ceiling, because the lintel's cell holds 0.074 of wall, under 0.08.

The solver.
- A solid cell's heat capacity is its fraction times each material's ρ c V (`flamescapes.js` 357 to 371). A solid cell with no fraction has none.
- Its regions are face-connected. `topology` (2774 onward) links each open cell to its six neighbours (`DIRS`, 592 to 599). The sealed regions and pressure zones are the connected parts that reach no open side.
- `RUN_CACHE_VERSION` (`runCache.js` 54) keys cached runs on the physics.

#### The reference: how FDS snaps (FDS 6.11.1, commit ff928db)

`Source/read.f90`, `READ_OBST`:
- 10998 to 11005: "Determine the indices of the obstruction according to cell edges, not centers." Then `OB%I1 = NINT( GINV(XB1-XS,1,NM)*RDXI )`, and so for each of the six faces. On a mesh without stretching `GINV` returns its argument (`func.f90` 6505 to 6507). Each face goes to the nearest cell plane. Cell p, counted from 0, is in the obstruction when NINT(a) ≤ p < NINT(b), that is when a < p + ½ ≤ b: when its centre lies in it (NINT rounds a half away from zero).
- 11007 to 11016, with `THICKEN`: `IF (OB%I1==OB%I2) THEN`, `OB%I1 = INT(GINV(.5_EB*(XB1+XB2)-XS,1,NM)*RDXI)`, `OB%I2 = MIN(OB%I1+1,IBAR)`. An obstruction whose two faces round to one plane becomes the one cell that holds its midpoint. A midpoint on a plane goes to the cell above it.
- 11017 to 11025, without `THICKEN`: such an obstruction stays a sheet of no thickness on that plane; one thinner than a quarter cell that straddles a plane goes to the nearer plane.
- 11061 to 11067, "Throw out obstructions that are too small": rejected when two of its dimensions round to no cell.
- `THICKEN_OBSTRUCTIONS` defaults to false (1806), and each obstruction's `THICKEN` takes it (10733).

`READ_HOLE`: 11716 to 11721 snap a hole's faces with NINT too, and 11736 to 11762 cut the snapped hole from each obstruction. A hole whose snapped width is zero across the obstruction is "merely sliced" (11750 to 11756) and cuts nothing. In FDS a door narrower than a cell is one cell or closed, by whether a cell centre lies in it.

User Guide (`Manuals/FDS_User_Guide/FDS_User_Guide.tex`):
- 1258: "An obstruction that is relatively thin compared to the gas phase numerical grid spacing is approximated as an infinitely thin sheet."
- 1260: "To prevent FDS from allowing thin sheet obstructions, set THICKEN_OBSTRUCTIONS=T on the MISC line, or THICKEN=T on each OBST line for which the thin sheet assumption is not allowed."
- 1268: "FDS then 'snaps' the geometry to fill (or void) the volume of the nearest cell. Obstructions that are too small relative to the underlying numerical mesh are rejected. That is, if at least two of the three obstruction dimensions are less than half of the corresponding cell dimensions, the obstruction is ignored." The code's test is the rounding at 11063, which depends on where the obstruction lies as well as its size.
- 1321, on HOLE: "Any solid mesh cells within the volume [...] are removed."
- 2172: an obstruction may "snap to the local grid but still have the heat transfer solution reflect the actual thickness".

What carries over. This solver blocks whole cells only; it has no sheet on a cell face. So the reference is FDS's rule with `THICKEN`: every face to the nearest plane, and an element whose faces round to one plane in the cell that holds its midpoint. FDS's rejection of an obstruction with two dimensions of no cell carries over unchanged.

#### The method

Chosen: a rule on lines through the cells' sample points, cast through the meshes along all three axes as `volumeFractions` casts its rays. For boxes it is FDS's rule above, with one addition (step 4).

Considered and set aside:
- A rule on the total volume fraction alone. For one plane face, v ≥ 0.5 already is the nearest plane: a plane through a cube's centre halves it, so the side holding the centre holds at least half. But a thin wall's cells hold its thickness share τ, and its jamb's cell holds τ times the jamb's share. On Steckler's 0.10 m wall on 0.27 m cells (τ = 1/3 by the sub-rays), a cell 60 % covered by a jamb and a cell fully crossed by a 0.06 m wall both hold 0.2. No threshold on the total can put a thin wall's jamb at the nearest plane. A rule on each mesh's own fractions could, but it needs each element's thickness, which a sloped or tapered element does not have.
- The present rule with a strict test (> for ≥ at 235 to 237). It ends the plateau, but it loses a thin slab split equally between two cells and keeps the jamb's bias in thin walls.
- FDS's rule with no step 4. Every gap holding a cell centre would open a slot one cell high or wide. An IFC wall stopping 1 cm under its slab would open a slot along its whole length, on 0.27 m cells at 4 % of placements. The present rule seals such gaps, and the app runs on IFC models drawn for no grid.

Frames. In the grid's local axes (`voxelize.js` 6 to 9: the two horizontal axes and the up axis), u_d = (x_d − origin_d) / h is the position in cells along axis d. Cell p spans [p, p + 1) and its sample coordinate is s_p = p + ½ − η, with η = 10⁻³. A cell's sample point is its centre moved η h toward the origin along every axis.

1. Lines. For each axis d and each pair (q, r) of cell indices on the other two axes, the line L_d(q, r) runs along d through their sample coordinates. There are n_b n_c + n_a n_c + n_a n_b lines.
2. Crossings. Each mesh's crossings with L are found as `volumeFractions` finds a vertical ray's: the barycentric test in the plane of the other two axes (144 to 158, axes permuted), a triangle with |det| ≤ 10⁻¹² in that plane skipped (145). They are sorted, merged within 10⁻⁶ m (as 170) and paired, an odd last crossing dropped (as 175). Each pair is an interval [a, b] in u_d, clipped to [0, n_d]. Lines with an odd count are counted by axis.
3. Union. All meshes' intervals on L are sorted and merged where they overlap or touch (within 10⁻⁶ m). An element and another it runs into become one interval.
4. Gaps. Two consecutive intervals on L whose gap a₂ − b₁ is less than g = ½ cell are joined into [a₁, b₂], until no gap on L is shorter. A gap open to an end of L is never closed.
5. Snap. For each interval [a, b] after step 4, every cell p with a < s_p ≤ b is solid: its sample point is in material. If no sample lies in (a, b], the interval is thin (then b − a < 1), and the one cell p = ⌊(a + b)/2 + η⌋ is solid: the cell holding the midpoint, as `THICKEN` (11012), a midpoint on a plane going to the cell above it.
6. Thin elements' own lines. For each thin interval of step 5, on L_d(q, r) with midpoint m, cast a line along each of the two other axes d′ through u_d = m and the sample coordinate of L's index on the third axis. On each, do steps 2 and 3. Each gap on it shorter than ½ cell, and not open to its end, seals the cells p′ along d′ with b₁ < s_p′ ≤ a₂ in the layer ⌊m + η⌋ along d. Nothing else from these lines is used.
7. The mask is the union of the cells set in steps 5 and 6 on every line.
8. Material. A solid cell whose total fraction is 0 holds an element thinner than the sub-rays' spacing that fell between them. It takes a fraction ℓ / h of the material of the mesh with the longest length inside it on the line that sealed it, ℓ that length; with several lines, the largest ℓ. No other fraction changes. These cells are counted (`filled`).
9. It returns the mask and the counts: cells solid by their sample, by the thin rule, by step 4, by step 6, `filled`, odd lines by axis, and the time taken.

Thin elements. An element thinner than a cell is one layer: on each line that crosses it, the cell that holds its midpoint. Where it straddles a plane, that is the cell holding more of it; split equally, the cell above. A line that crosses it at a sample point seals that cell, which also holds the midpoint. A 52 mm ceiling is one cell in every column. An element thinner than a cell in one direction is never lost. One thinner than a cell in two directions (a post, a bar, a pipe) is kept where it holds a sample point in either of them, and lost where it holds none in both. That is FDS's rejection at 11063.

Corners and junctions.
- Two elements that overlap or touch share one interval on every line through the touch (step 3). Their cells meet there, since one plane snaps both.
- Two elements less than half a cell apart are joined, on lines through sample points (step 4) and on lines through a thin element's midplane (step 6). A wall stopping 1 cm under its slab, a door leaf with 3 mm gaps and a 10 mm undercut, a partition ending 5 cm from the wall it meets: all seal.
- A thin element is sealed on every line along the axis nearest its normal. On those lines the sealed cell holds the midpoint, which moves by at most one cell between neighbouring lines when the element's slope against that axis is at most 1. The axis nearest the normal always gives that. If neighbouring lines seal cells p and p′ with |p − p′| ≤ 1, no open cell on one side shares a face with an open cell on the other; that would need p′ < p − 1. Such a layer meets itself only along edges at its steps, and an edge is no path: the solver's flow, regions and pressure zones pass through faces only (`topology`, six neighbours; the face velocities).
- Not proven: where a curved thin shell's nearest axis changes, two line families seal it and must overlap. Check (b) tests round walls and domes, and junctions of oblique walls.
- Outside the item: the advection's interpolation can read gas across a wall one cell thick (`tools/wall-leak-check.mjs`, its "solver" check). That is no gap in the mask, and the item changes nothing about it.

Oblique and curved geometry. The lines need only triangles, as `volumeFractions`' rays do. A face of any slope lies at the nearest plane along every line that crosses it. The three lines through a sample point agree on whether it is inside an element, so an element at least a cell thick everywhere is exactly the set of cells whose sample points it holds, of any shape. That set's volume is unbiased over positions. A turned or tilted element is treated as a box is, with no further choice.

Openings. A door, window or hole is a gap between material intervals, and each side snaps like a face, to the nearest plane. An opening less than half a cell wide is closed (step 4 or 6). One at least a cell wide is open, since it holds a sample. One between half a cell and a cell is open when a sample point lies in it. A closed door (an IfcDoor leaf, when the run closes doors, `flamescapesRun.js` 136) is a thin element and seals its opening.

Steckler under the item, by hand from steps 1 to 5 on the grid `fds.mjs` builds (positions in cells):
- The room. On 0.27 m cells the ceiling's shell is [8.889, 9.889]. Sample 9.499 lies in it and 8.499 does not, so the ceiling is plane 9 and the room 8 cells, 2.16 m. On 0.20 m: [11.65, 12.65], 2.20 m. On 0.34 m: [7.265, 8.265], 2.04 m.
- The 6/6 door. On 0.27 m its jambs are at 4.815 and 7.556; samples 5.499, 6.499 and 7.499 lie between: 0.81 m. On 0.20 m, 6.15 to 9.85, four samples: 0.80 m. On 0.34 m, 4.029 to 6.206, two: 0.68 m. Its top is 1.80, 1.89 and 1.70 m on 0.20, 0.27 and 0.34 m, under a lintel one cell deep on each (on 0.27 m, the cell the present rule leaves open).
- The 2/6 door (0.25 m) is one cell on each size: 0.20, 0.27 and 0.34 m wide (the gaps 7.50 to 8.75, 5.815 to 6.741 and 4.824 to 5.559 each hold one sample). The 8/6 door is 1.00, 1.08 and 1.02 m.
Check (d) must reproduce these exactly.

Pockets. The item fills none. Under it a flat ceiling's row is open or solid along its whole length, corners included. A pocket remains only where the true geometry has a recess whose plan, in that row, holds one sample point and is at least half a cell wide both ways (a narrower one is closed by step 4): a vent or shaft one cell wide through a ceiling, the apex of a pitched or domed ceiling, a narrow coffer. 13.22 deals with what a tree reads there.

Mass. The item changes no fraction but those of step 8. A partial cell it leaves open keeps its fraction, and the solver gives an open cell's material no heat capacity (`flamescapes.js` 368 to 371), as it does now for any open cell. A wall about a cell thick that straddles two cells, 0.6 and 0.4, keeps only the larger part's mass; the present rule keeps both. The change in Σ `inertC` is reported in every scene (check (i)).

The constants.
- Faces to the nearest plane; the midpoint's cell for an element whose faces round to one plane, the cell above for a midpoint on a plane; rejection of an element with two dimensions of no cell: FDS 6.11.1 `read.f90` 11000 to 11016 and 11063, `READ_HOLE` 11716 to 11721.
- g = ½ cell, the gap closed in steps 4 and 6: a modelling constant fixed here from no data. Below it, closing a gap biases no opening at least half a cell wide (its mean modelled width stays its width, check (c)), and it seals modelling gaps between IFC elements. A larger g would close real openings narrower than it.
- η = 10⁻³ cell: a numerical constant fixed here. It keeps sample points off the faces of FDS geometry drawn on a 0.1 m grid (on 0.20 m cells the centres lie at 0.1, 0.3, ... m), and it puts a face that lies on a sample point on the side NINT puts it. It biases every face by η, 10⁻³ cell.
- 10⁻⁶ m and 10⁻¹²: the voxeliser's own (`voxelize.js` 145, 170).

Where it lives.
- `voxelize.js`: `snapNearest(grid, meshes, fraction)` returns `{ solid, stats }` and writes step 8's fractions. `occupancy(grid, fraction, { snap, meshes })` calls it when `snap` is `"nearest"` and meshes are given, and otherwise runs lines 221 to 254 unchanged.
- `flamescapesRun.js` `buildScene` (107 onward) takes `snap` and passes the meshes it already holds (158 to 159). The worker's run settings carry it.
- `fds.mjs` `gridFor(geo, h, Vx, { snap })` passes the boxes' meshes (344 to 345). `validation-suite.mjs` takes `--snap nearest`.
- `createSolver`'s fallback (`flamescapes.js` 351) has no meshes and keeps the present rule. Every scene built with the item on carries its mask. The tools that call `occupancy` themselves (heat-closure, transport-checks, energy-budget, steckler-checks, flamescapes-gpu-check, sprinkler-validation) keep the present rule unless passed `snap`.
- The GPU takes the CPU's mask and needs no change.

#### The switch

`snap: "seal" | "nearest"`, default `"seal"`, the present rule. With `"seal"` no new code runs, and the mask, the fractions and every output are bit-identical to the build before; the cost is one comparison. When the default changes, `RUN_CACHE_VERSION` rises, so no cached run made under the other rule is reused.

#### Checks (fixed now, before any run)

All but (e) and the step costs in (f) run in a new tool, `tools/snap-check.mjs`. No measured data enters any check.

(a) Off is off. With `"seal"`, the mask and fraction arrays are byte-equal to the build before on: the eight series' scenes (steckler, nbs, nistnrc, wtc, atf, fsri, ulnist, fse; each test's geometry as the validation suite builds it) on 0.20, 0.27 and 0.34 m cells; every scene the unit suite builds through `occupancy`; Duplex, Office, Clinic and Vienna at the standard grid (`STANDARD_CELLS` 30 000, `lib/hardware.js` 15) and whole on 0.27 m cells. The unit suite and the GPU parity phases are bit-identical. A failure means the off path was touched.

(b) Watertightness. The item on, a 6-connected flood through open cells.
1. Closed rooms: `wall-leak-check.mjs`'s 5 × 4 × 2.7 m room, its walls, floor and roof t thick, t ∈ {0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30 m} up to h, and t = h; turned about the vertical by {0, 5, 15, 30, 45, 60, 75, 85}°; tilted about a horizontal axis by {0, 10, 30, 45}°, so every face is oblique; 10 translations within a cell per case, from a fixed seed; on 0.20, 0.27 and 0.34 m. The flood starts in the room's centre cell. Passes if it reaches no cell whose sample point lies outside the room's outer faces, in every case.
2. Junctions: the room split by a partition of the same thickness meeting the outer wall in a T, square and at 30 and 60°. Each half is sealed from the other and from the outside.
3. Gaps: a wall ending g under the roof, g ∈ {0.005, 0.01, 0.02, 0.05 m, 0.49 h}; a partition ending g from the wall it meets; a door leaf 0.04 m thick in a 0.9 × 2.0 m opening, 3 mm from its jambs and head, 10 mm undercut. All sealed. Reported, not judged: g = 0.51 h, open only where a sample lies in the gap.
4. Curved: a round room of radius 2 m (96 sides), and the same under a hemispherical shell, walls and shell 0.01 m to h thick. Sealed.
5. The buildings: Duplex, Office and Clinic (`public/examples`, read by `tools/ifc-extract.mjs` `extract`), whole, on 0.20, 0.27 and 0.34 m, doors closed and doors open. The sealed regions (`topology(sim).sealed`) are counted with the item off and on. Passes if on ≥ off in every case, or if each pair of off-regions that the item joins is joined only through cells within one cell of the bounding box of an IfcDoor (doors open), an IfcWindow or an IfcOpeningElement of the model (the tool reads these from the IFC). Any other join fails and is listed with its cells and the elements beside them. A region the item adds is listed, not judged.
Every line in 1 to 4 has an even crossing count (their meshes are closed). The same families with the item off are reported beside. Reason: watertightness must never be lost, so one leak in one case fails the item. A failure means an element vanished or leaks, a junction was not joined, or a line's parity failed.

(c) Unbiased. 1000 positions across a cell, in steps of 0.001 cell, on 0.20, 0.27 and 0.34 m.
1. Faces: a slab 3 cells thick, its underside and its top; a wall 3 cells thick, both faces, along each horizontal axis. The error e is the modelled plane less the true face, in cells. By the rule e lies in (−½ + η, ½ + η], and its mean over this sweep is η + 1/2000 = 0.0015. Passes if every e lies in (−½ + η − 10⁻⁶, ½ + η + 10⁻⁶] and |mean e| ≤ 0.005. Reason: 0.005 is three times the rule's own bias. The present rule's −0.42 fails it, and so does any rounding at a sub-ray, which is off by at least 1/(2S) = 0.07.
2. Oblique faces: the slab tilted 10, 30 and 45°, the wall turned 10, 30 and 45°, 50 positions each. On every line crossing the face, e within the bounds of 1; |mean e| ≤ 0.01. Reason: over 10⁴ lines or more, the standard error of a mean of errors spread over a cell is about 0.003.
3. Openings: width w ∈ {0.3, 0.49, 0.5, 0.6, 0.75, 0.9, 1.0, 1.5, 2.0, 2.74} cells, in a wall 1.5 cells thick, a 0.10 m wall and a 0.02 m wall, swept in position. The modelled width W is the count of open cells across it at mid-height. By the rule, W's mean over a cell's positions is exactly w for w ≥ ½: each sample lies inside for a span w of positions. The sweep's 1000 positions move the mean by at most 2/1000. Passes if |mean W − w| ≤ 0.005 for w ≥ ½, W = 0 at every position for w < ½, |W − w| < 1 always, and W ≥ 1 for w ≥ 1. The present rule's loss (1 − 1/S in the thick wall; closed at 0.6) is reported beside.
4. Thin slabs: t ∈ {0.01 m, 0.05 m, 0.2, 0.5, 0.8, 0.99 cell}: exactly one solid cell on every vertical line at every position, the cell ⌊m + η⌋. t ∈ {1.0, 1.4, 2.6 cells}: the mean count of solid cells on a line within 0.005 of t.
A failure means the snapping is not to the nearest plane.

(d) The named geometries. For each test of the eight series, on 0.20, 0.27 and 0.34 m, item off and on, a table of true against modelled: each room's ceiling over its floor, at each tree's column and the room's centre; each opening's width, height, sill and soffit (every HOLE, every OPEN vent on a mesh side, and FSRI's door from its box scene). True values from the FDS input (`readGeometry`); modelled ones from the mask, along the lines through the opening's centre in its wall's cell layer. Pass rule, derived from steps 1 to 5:
- every face of an element at least a cell thick, and every side of an opening, within (½ + η) h of its true place;
- every opening dimension within h of its true size;
- every opening at least h wide both ways open; every opening narrower than h/2 in a direction closed; one between h/2 and h open exactly when a sample point lies in it, the table giving the sample coordinates;
- Steckler's values above, exactly; with the item off, 12.4's values (1.89 and 2.04 m; 0.54 and 0.68 m) and the first table here, exactly, which checks the measuring itself.
"No opening wider than half a cell closed" cannot hold under nearest-plane snapping. An opening narrower than a cell that holds no cell centre closes in FDS too (11716 to 11721, 11750). The table lists each such opening. A failure means the item does not snap as drafted.

(e) The unit suite. `npm test` with the default set to `"nearest"` for the run, a one-line change not committed. Every test passes unchanged. Expected to depend on exact geometry: `flamescapes.test.mjs` 133 to 147 (a slab thinner than a cell seals), 148 to 167 (a 52 mm ceiling is one cell in each column; under the item the split case takes the upper cell, still one), 179 onward (the suspended ceiling within h of 2.6 m), 169 to 178 (the contents' floor area 16 to 24 m²) and 324 (duct terminals on open cells); `suppressionSystems.test.mjs` 138 to 160 (a room's open cells inside its walls, detectors under its ceiling); `flamescapes.burner.test.mjs` (open cells for a burner); `lod1.test.mjs` 354 (converted geometry voxelises). By the derivations above each still holds. `flamescapes.wall.test.mjs` 167 builds its mask by hand and does not call the voxeliser. A test that fails is not edited, and its failure is reported. Only a test whose reference is the present rule's own geometry may be rewritten, as 13.12 rule 1 allows, disclosed as written after a failure. Any other failure is the item's.

(f) Speed. Building a scene's mask (`volumeFractions` and the mask) with `"nearest"` takes at most 1.25 times as long as with `"seal"`, median of 5 builds, on Duplex, Office, Clinic and Vienna at the standard grid and whole on 0.27 m cells, and on each series' scene on 0.20 m cells. Reason: the lines number about 3 a column plus 2 for each thin crossing, against S² = 16 to 49 sub-rays a column, near a fifth of the rays already cast; 1.25 allows for the sorting and the union. Reported, not judged: ms a step on the three buildings, CPU and GPU (`tools/backend-bench.mjs`), with the count of open cells off and on. More open cells cost more a step; that cost follows from the geometry. A failure means the line pass is too slow for the app's set-up.

(g) FDS's own rule. 2000 random scenes per cell size, each of one to six boxes (sides 0.05 to 3 cells, positions uniform) and zero to three holes through them. The reference is an independent function in the tool, in integer arithmetic on the boxes: NINT on each face; `THICKEN`'s midpoint cell where a dimension's faces round to one plane; rejection where two do; holes cut by their NINT ranges, none where a hole rounds to no cell. Passes if the item's mask equals the reference in every scene where no two boxes are less than half a cell apart along an axis and no face lies within 2η of a sample point; and, in the other scenes, equals the reference plus the gaps step 4 closes, exactly. Reason: this is FDS's rule, computed without ray casting. A failure means the item departs from FDS where it claims to follow it.

(h) Invariance. The same mask with the meshes in 5 orders. Translated by 1, 2 and 3 whole cells along each axis, the mask translates exactly, in scenes with no face within 10⁻⁴ cell of a sample point. A failure means an order or origin fault.

(i) Material. With the item on, every solid cell's total fraction is above 0 in every scene of (a) to (d). Plates 0.01, 0.02 and 0.04 m thick, axis-aligned: their cells' fraction within 2 % of t / h. Reported, not judged, per scene: `filled`, Σ `inertC` off and on, odd lines by axis.

#### What the item does not claim

- It is a geometry item. It changes which cells are solid, and the fractions of step 8, and nothing in the solver.
- It does not model flow through a door narrower than a cell. Two attempts at that failed, and that line is closed.
- Steckler's 0.24 m door (0.25 m in the input) is one cell on each size under the item: 0.20, 0.27 and 0.34 m wide, so its area is 0.80, 1.08 and 1.36 times true. Under any snapping rule a door narrower than a cell is one cell or closed. Under this one it is one cell when it is at least half a cell wide and a sample point lies in it, and closed otherwise. That is FDS's own NINT, and over positions it is unbiased for every opening at least half a cell wide. Always closing would lose doors FDS keeps; always opening would overstate every narrow door by up to a cell.
- It changes every room's volume and every opening's area. The validation is run again in full on 0.20, 0.27 and 0.34 m cells and reported beside the existing run. No series' data is consulted to choose any constant: g and η are fixed here from no data, and the rest is FDS's rule.
- It does not keep a wall's mass in a partial cell it opens (reported in (i)).
- It does not touch the advection's reading across walls one cell thick.
- It does not make FDS's sheets of no thickness; this solver has none.
- It fills no pocket.

#### Default

Off (`"seal"`) until (a) to (i) pass. Then `"nearest"`, with `RUN_CACHE_VERSION` raised and 12.4's second point rewritten with the new numbers, dated. Made after 13.B began, it gets its own full rerun on 0.20, 0.27 and 0.34 m, reported beside 13.B (13.12, rule 5).

#### What ends this line

If the item fails any of (a) to (i) in any case, it is not made. The present rule stays, and 12.4 stands.
- A failure of (b) on oblique or curved shells or at junctions says the union of three line families does not seal everywhere. A second attempt would add a closing pass on the mask, for example sealing an open cell that lies between two cells one element's thin rule sealed, and it needs its own draft.
- A failure of (b) on the buildings through gaps at least half a cell wide between elements meant to meet says the models' own gaps exceed g. A larger g closes real openings, and that needs the user's word.
- A failure of (c), (d), (g) or (h) is a fault of the code against this method. Corrected code, run again in full, is the same attempt, reported with both commits.
- A failure of (f) alone may be met by faster code whose mask is bit-identical to the slower code's on every scene of (a).
No second attempt is drafted. It needs the user's word and a new method committed here first.

#### Sources (read, or run for this draft)

- FDS 6.11.1, tag FDS-6.11.1, commit ff928dbf68a2623ced3a007be75391c6dc6c9f90, fetched from GitHub raw. Read: `Source/read.f90` (`READ_OBST` 10585 to 11569, of it 10860 to 11067; 1806, 10733; `READ_HOLE` 11574 to 11931, of it 11703 to 11800); `Source/func.f90` (`GINV` 6496 to 6524); `Manuals/FDS_User_Guide/FDS_User_Guide.tex` (666, 1255 to 1270, 1314 to 1333, 2172). `Validation/Steckler_Compartment/FDS_Input_Files/Steckler_010.fds`: its MESH, OBST, HOLE and VENT lines only, for the examples' coordinates.
- This repository at 191b1d3: `voxelize.js`; `fds.mjs`; `flamescapes.js` 336 to 372, 592 to 599, 2774 onward; `flamescapesRun.js` 107 to 160 and 320 to 345; `runCache.js` 54; `lib/hardware.js` 15; `tools/ifc-extract.mjs`; `tools/backend-bench.mjs` 65; `tools/wall-leak-check.mjs` 1 to 80; the unit tests named in (e); `validation-suite.mjs` 129 to 138 and 270 to 300.
- Run: the present voxeliser, imported read only, on synthetic boxes: the Steckler copy through `fds.mjs` `gridFor`; faces and openings swept over 200 positions; thin slabs over 100; the corner room. No validation run, no fire simulation, no result file.

### 13.22 The harness: a measurement tree drops a pocket cell (2026-11-13, before any run of it)

Drafted blind with 13.21. It changes only which cells a thermocouple tree is compared through in the validation harness. The solver and the app are not touched.

#### What the harness does now

`treeCells` (`fds.mjs` 377 to 394) takes every open cell of the column over the tree's plan point, from its floor up to the test's nominal ceiling, skipping solid cells: a bracket sealed into whole cells interrupts the tree but does not end it. `validation-suite.mjs` 283 builds each tree with it, and after the first second the suite keeps only cells of the fire's connected gas (278 to 280). A pocket is an open cell whose four horizontal neighbours are solid. It joins the room through the cell below it, so it passes both. Under the present rule pockets form at the corners of a snapped ceiling's row (13.21, What the voxeliser does now).

#### The rule

- A tree cell (i, j, k) is a pocket when none of (i ± 1, j, k) and (i, j, k ± 1) is open (`G.solid` 0). A neighbour outside the grid is not open.
- `treeCells` walks the column bottom up as now. At the first pocket it stops: that cell and every cell above it are dropped. The tree's profile ends at the last cell below the pocket.
- It returns `dropped` beside `cells`: each dropped cell's height and index, and which one was the pocket. A pocket with open cells of the room above it, not at the tree's top, is flagged `midTree`.
- Every reading and comparison made from the tree uses `cells` only. A measured thermocouple with no kept cell at its height is compared with nothing, and it is listed.
- Each run lists, per tree and cell size, the dropped cells and the height where the tree now ends, beside the tree's snapped ceiling (`column().top`), as 12.4 has each run record its trees' ceilings.
- It applies with 13.21 off and on. Point devices (`deviceCell`) are not changed.

#### Why this is a measurement rule

The solver is not changed, and neither is any cell's gas. Only the cells a tree is compared through change. A thermocouple at that height hangs in the room's gas. On the grid the room has no gas at that height: the snapped ceiling is solid there everywhere but in the pocket. The pocket's gas meets the room through one face, under four walls and a ceiling, so its temperature is a recess's. Comparing the measurement with it would compare a room's gas with a recess. With no cell on the grid that stands for the room at that height, the tree has no reading there.

#### Checks (fixed now, before any run)

(a) No pocket, no change. On every tree of the eight series, on 0.20, 0.27 and 0.34 m, with 13.21 off and on: `cells` equals the old function's list up to its first pocket, cell for cell. The first pocket is found independently by the check tool, counting open horizontal neighbours on `G.solid`. Where the old list holds no pocket, the two are identical. A failure means the rule changed a tree beyond its pocket.

(b) Constructed cases, on the three sizes, with 13.21 off:
- a box room 3 × 3 × 2.4 m, walls on cell planes, the ceiling entering its top row by 0.3 of a cell: a tree in a corner column ends below the corner cell, which it lists as dropped; a tree at the room's centre loses nothing;
- a flat ceiling with a recess one cell square and one cell deep: a tree under it drops the recess cell; under a recess two cells by one, nothing is dropped;
- a doorway one cell wide: a tree in it loses nothing;
- a bracket two cells tall in a room: a tree through it skips the bracket's cells and loses nothing.
Each exactly. A failure means the pocket test is wrong.

(c) Under 13.21. The room of (b) has no pocket with the item on: its ceiling row is open to the corners. Pockets can remain only where the true geometry has a recess holding one sample point in its row and at least half a cell wide both ways: a vent or shaft one cell wide through a ceiling (an OPEN vent cut through the harness's ceiling shell), the apex of a pitched or domed ceiling, a narrow coffer. The run lists every pocket on every tree, 13.21 off and on. Reported, not judged.

#### What ends this line

If (a) or (b) fails, the rule is not used, and trees read as they do now. The rule catches pockets one cell wide only: a recess two cells wide, such as a wider vent, is kept, because its cells have an open neighbour. That limit is stated here, not addressed.

#### The coordinator's decisions on 13.21 and 13.22 (2026-11-13, 08:47, before any run of either)

The two drafts above are the physics agent's, written blind (13.12, rule 2). It opened no result file and no plan section beyond 12.4, 13.12 and the method part of 13.19. **Both items were conceived after results were read, and that is declared here.** The coordinator asked for them after reading 13.B's sealed file on 0.27 m (13.16), where three Steckler rooms whose door the grid closes read a collapsed layer through a pocket cell at the top of the tree. The coordinator has read 13.A and 13.B, so the decisions below add no constant and move no tolerance. Each takes the draft's own choice, or the stricter reading.

**The user's word, and what attempt 3 at the doors is.** 13.20 closed the door line until the user's word and a new method. On 2026-11-13 at 06:11 the user wrote: "yes do a third run try something different this time so theres a higher sucess rate". Attempt 3 is item 9 as drafted above. It differs in kind from attempt 1 (13.13) and attempt 2 (13.17). Those changed the flow the solver gives a door that the grid holds wrongly, and both failed their checks. This one changes the door the grid holds and leaves the solver's flow alone. The draft was written before the user's word, so its sentence "that line is closed" was true when written and stays as written. What the item claims for doors is geometry only: an opening at least half a cell wide has its true width on average over positions, and one at least a cell wide is never a cell or more from it, where the present rule, in a wall at least a cell thick, loses 0.75 to 0.86 of a cell on average. It claims nothing for flow through a door narrower than a cell, which is one cell or closed. The coordinator adds no flow check, because one written now would not be blind. Door flow, door velocity and the neutral plane are judged only by the held-out series in the item's own rerun, under 12.6's bands as they stand.

**Checked by the coordinator,** apart from the agent: FDS 6.11.1's sources fetched again at ff928db; in `read.f90`, NINT on an obstruction's six faces (10998 to 11005), the thickening to the midpoint's cell (11010 to 11016), the rejection (11061 to 11067), NINT on a hole's faces (11716 to 11721), the sliced hole (11750 to 11756) and the default of `THICKEN_OBSTRUCTIONS` (1806, 10733); `GINV` in `func.f90` (6496 to 6507); the User Guide's lines 1258, 1260 and 1268. The present rule's Steckler table was reproduced by running the agent's script again on this branch. Not checked: the sweeps' means, the hand values under the new rule (check (d) reproduces them or fails) and the unit tests' line numbers.

1. **The method stands as drafted,** steps 1 to 9, with step 4 (gaps), step 6 (a thin element's own lines) and step 8 (a fraction for a solid cell that holds none).
2. **g = ½ cell and η = 10⁻³ cell stand,** fixed from no data. g departs from FDS, which opens a hole 0.3 of a cell wide where it straddles a cell centre. It is there for the gaps IFC models leave between elements meant to meet.
3. **No material is moved between cells.** A partial cell the item opens loses its heat capacity, as any open cell does now. (i) reports the change in every scene.
4. **An element thinner than a cell in two directions is lost where it holds no sample point,** as FDS rejects it.
5. **The switch and the names are the draft's.** A cached run's key carries `snap`, so a run made under one rule is never reused under the other.
6. **(f) is judged as drafted.** The cost of a step with more cells open is reported, not judged, and goes to the user: the user's rule for a feature is the same or better speed with it on.
7. **The default: stricter than the draft.** Passing (a) to (i) makes `"nearest"` available to the harness and to the app's settings. It does not change the app's default. The default changes only after all of these: the item's full rerun on three cells, reported beside 13.B; the app's verdict text rewritten from that rerun; the sprinkler validation and the tools that build masks themselves passing `snap`, so that they voxelise as the app does; and the user's agreement, given with the step cost before them. `RUN_CACHE_VERSION` rises with the default, not before.
8. **(b)5 gains a case.** The three buildings are also flooded at the standard grid, clipped as the app clips them (`clipMesh`), under the same pass rule. The agent named the risk in its notes: a line inside a wall whose end lies outside the clip box sees an odd count. The case comes from that note and from no result, and it can only fail the item. The tool reads IfcOpeningElement itself; `ifc-extract.mjs` is not changed for it.
9. **(e) is the draft's.** A test that fails is not edited. Only a test whose reference is the present rule's own geometry may be rewritten, disclosed as written after a failure.
10. **13.22 ends a tree at its first pocket,** mid-tree included and flagged. Point devices are not changed.
11. **13.22 reaches 13.B by a rerun, not by scoring again.** 13.B's files hold each run's reduced values and no tree profile. The runs with a pocket on a tree are found from the masks alone, by 13.22's check tool, before any run. They run again on 13.B's solver (550fce5) and flags, in each leak reading of 13.16 where the two differ, and are merged over 13.B's files as 13.16 merged its reruns. The control is 13.16's: on each cell the first three runs of the suite's order that hold no pocket run again and must give pairs identical to 13.B's. If the control fails, nothing is merged.
12. **The order, and what is reported (13.12, rule 5).** First 13.22 alone over 13.B. Then 13.21 with 13.22 on: the eight series, and the chair series beside them as 15.9 read it, on 0.20, 0.27 and 0.34 m, on the harness as 13.16 corrected it (no leak where none is stated). Each is reported beside 13.B, verdict by verdict, before anything is combined. 13.B's words stay the app's until the user has seen both.
13. **The base** is `next/calibration-ui` at the commit the item's branch starts from. The solver's physics is 13.B's (550fce5); the item touches the voxeliser, the scene's set-up and the harness only.
14. **What ends this line stands as drafted.** If attempt 3 fails its checks, 13.20's sentence returns: doors stay as the voxeliser makes them now, and a fourth attempt needs the user's word and a new method here first. If the item passes its checks and the doors' words do not improve in its rerun, that is reported as it is. Nothing is fitted and no band moves.

### 13.23 Item 8, check (b): the result (2026-11-13, 08:47)

**The permission.** Check (b) of 13.19 waited for the user's permission to run the RadCal build: the session's permission check had refused to run it, and that was not worked around. On 2026-11-13 at 06:11 the user wrote: "Yes you get the permission". RadCal was run at 08:36. Its source, the compiler and the built program stay outside the repository.

**The build against RadCal's own example.** RadCal was built from `firemodels/radcal` at abe2a8f with gfortran. Its shipped example input gives, on this build, a total emissivity of 0.1129, a received flux of 1017.0 W/m²/sr and a total transmissivity of 0.8871. The user's guide prints, for the same input under revision 51 of June 2014, 0.1215, 1008.6 and 0.8787. So the build is not verified to the digit against a published output of its own revision, and that is a limit of this check. The build's figures agree among themselves: the flux is the wall's radiance times the transmissivity plus the gas's emission, to 0.02 %. RadCal's source changed after 2014 (its tables in 2016, its soot model in 2020, further commits to 2025), and the example holds soot and a fuel gas, which the cases of (b) do not. The difference was not traced further.

**The result: (b) fails.** 72 cases as 13.19 fixed them, over 50 to 10 000 cm⁻¹ at 1 atm. The solver's values were written by the item's check tool at 05:28, before RadCal ran. 38 of the 72 lie outside 25 % or 0.01.

| Quantity, gas temperature | Cases | Outside | Solver over RadCal |
|---|---|---|---|
| Emissivity, 400 K | 12 | 12 | 1.26 to 1.50 |
| Emissivity, 600 K | 12 | 10 | 1.16 to 1.38 |
| Emissivity, 900 K | 12 | 0 | 0.96 to 1.21 |
| Emissivity, 1200 K | 12 | 0 | 0.90 to 1.11 |
| Absorptivity of 1173 K radiation, gas at 400 K | 12 | 10 | 0.98 to 1.64 |
| Absorptivity of 1173 K radiation, gas at 600 K | 12 | 6 | 0.94 to 1.48 |

Every case, with RadCal's radiance beside it, is in `benchmark/results/2026-11-13-gas-radiation-13.19b-radcal.json`. The coordinator's own transcription of FDS's functions, written apart from the agent's code, fails the same 38 cases. So the code carries FDS's weighted sum of gray gases as written, and the fit departs from RadCal at a layer's temperatures. This is the case 13.19 named: "A failure of (b) alone says that the fit, not the code, departs from RadCal."

Part of the absorptivity's failure is in the model's form. Its absorptivity for radiation from a source takes its weights at the source's temperature, whatever the gas's own temperature. At X_c + X_w = 0.15, M_r = 1.25 and 3 m it gives 0.359 for a gas at 400 K and at 600 K alike, where RadCal gives 0.219 and 0.242.

**Item 8 is not made.** By 13.19's "What ends this line", a failure of any check in any case ends the attempt. The flag stays off, the branch `flamescapes/gas-radiation` is kept and not merged, and no validation run is made with the item on. The gas stays transparent but for its soot. No tolerance moves: 25 % was fixed from how closely two accepted spectral models agree, and the fit is outside it by up to 64 %.

**What the failure says, and what it does not.** At 400 to 600 K, the temperatures of a room's hot layer, this fit radiates a sixth to a half more than RadCal. With the item on, a layer would have lost that much more heat than the reference allows. 13.B's burner rooms read hot, so the item would have moved them toward the band partly for a wrong reason. The check was fixed before any run to stop that, and it did. Not concluded: which of the two is nearer a measurement at 400 to 600 K. RadCal is this plan's reference, not a measurement, and the fit was made for combustion temperatures.

**A second attempt needs the user's word,** and a new method committed here first (13.19). None is drafted. Two candidates, for the user's decision: a weighted sum of gray gases whose fit reaches ambient temperature, judged by this same check against these same 72 values, with nothing moved; or FDS's default gray gas from RadCal's own coefficients, which 13.19 notes would need another independent reference. Until then the hot reading of the burner rooms and the cold reading of the chair rooms (15.9) stand as 13.B reports them.

### 13.24 Item 8: the build's other checks, and a finding in the solver as it stands (2026-11-13, 08:54)

13.23 ended the attempt on check (b). The build agent then stopped and handed back its report. This section records its other checks, so that every check of the attempt is reported (13.12, rule 3), and one finding that concerns the solver with the item off.

**The branch.** `flamescapes/gas-radiation` holds three commits: a269323 (the solver, `gasRadiation` 0 by default), 0e15df5 (the suite's `--gas-radiation` and the ledger's tallies, never run) and 29ef017 (`tools/gas-radiation-check.mjs`). It is kept, not merged and not pushed.

**The checks.** These are the agent's figures. The coordinator repeated only (b) apart from the agent (13.23), and the geometry of (h) below.

| Check | Result | What was run |
|---|---|---|
| (a) constants and functions | pass: 170 constants, worst difference 0 | in full; the GPU's tolerance not run |
| (b) emissivity and absorptivity against RadCal | **fail**, 38 of 72 (13.23) | in full |
| (c) gray cube | within ±2 % (0.9991 to 1.0023) | 20 steps, where 13.19 fixed 200 |
| (d) band cube | within ±2 % (0.9963 to 1.0026) | 20 steps, where 13.19 fixed 200 |
| (e) soot-free layer | within ±3 % (worst 1.0212, the floor on 0.27 m) | 20 steps, where 13.19 fixed 200 |
| (f) energy | pass: 6e-13 of emission a step against 1e-9 | the trials and both burner runs; no GPU run |
| (g) radiant fraction counted once | pass | both burner runs |
| (h) isothermal box | **fail**: the gas's net gain is -6.95 % of its emission on 0.2 m and -7.08 % on 0.3 m, against ±2 % | 20 steps |
| (i) off is off | identical fingerprints (0b8778bd, beaa010f); unit suite 536 of 536 on base and branch | the tool's own comparison with the base and the GPU phases not run |
| (j) unit suite with the item on | **not run** | |
| (k) ledger | **fail**: methane at 1 of 10 times, by 0.00006 MJ beyond the allowance; class A at 8 of 10, by up to 0.0157 MJ | both burner runs |

GPU parity and speed (14.6) were not started. So the attempt fails (b), (h) and (k). (c), (d) and (e) are not complete as fixed, and (j) and the GPU were not run. Had (b) passed, the item would still not have been made.

**What (h) found: a surface's rays start at the centre of its solid cell.** This is in the solver as it stands, with the item off. 13.19's point 8 kept the surfaces "as now", and (h) is the first check in this plan that closes a radiation balance between surfaces and the gas, so it is the first that could see it.

The coordinator read it in the code on `next/calibration-ui`: `radiation` casts every ray of a solid, porous or one-dimensional wall cell from the cell's centre, in a direction drawn by Lambert's law about the normal of one of its open faces, and `march` skips the emitting cell and leaves the ray in the first other solid or porous cell it enters. A ray that crosses a side face of the emitting cell enters the neighbouring wall cell and stays there. The share that leaves through the open face is the view factor from a point to a coaxial parallel square of side 1 at distance ½:

(4/π) (1/√2) atan(1/√2) = 0.5541.

4 000 000 directions drawn as the solver draws them give 0.5542. The agent's own sampling gave 55.4 %.

So a flat wall of solid cells sends into the room 55 % of ε σ (T_s⁴ − T_a⁴) A. The other 45 % passes sideways into the wall cells beside it and stays in the wall. No energy is lost and the books close (f). The exchange between a hot surface and the room is too small by that factor on a flat wall, by other factors at edges and corners, and by nothing for a cell that stands alone. It acts only where a surface emits, above 500 K in the solver as it stands (13.19, the code as read there).

In (h) the walls emitted 855 kW on 0.2 m, the solids took 688 kW and 278 kW left the grid. In a copy outside any branch the agent started each ray at a uniform point on the emitting face, and (h) read +0.048 % and -0.014 %. That copy is a diagnostic, not a method, and the coordinator did not run it.

Not checked: where the GPU path starts its rays (the two paths are held to the same rays by the parity checks, so the same is expected); the size of the effect in any validation run. 13.B's runs carry this geometry on every surface above 500 K, and no word of 13.B changes for it.

**What follows from it.** A candidate item: a surface's rays start on its emitting face. It was found during item 8's build, after 13.A and 13.B were read, and that is declared. It gets a blind draft and its own checks, committed here before any code (13.12). It belongs with the limit 13.19's decision 7 declared, the surfaces' 500 K floor: lowering that floor would let every warm wall emit, and with this geometry 45 % of that emission would stay in the wall. Any second attempt at the gas's radiation must also pass (h), which this geometry fails by itself. So the surfaces come before the gas.

**The ledger's remainder under (k).** With the item on, the unexplained remainder grows: class A at 120 s, 1.345 MJ on against 1.225 MJ off, of 12.0 MJ released. The agent computed it again with each cell's own heat capacity in place of the ledger's fixed 1005 J/(kg K), and the gap stays (1.330 against 1.210 MJ). The radiation stage's own books close to 3e-13. The cause was not found. The agent's guess is the flow's own lack of conservation under a different temperature field, and it is a guess. The remainder with the item off is 13.15's, not new.

**Departures the agent declared,** none judged: a band share that comes out negative is set to 0 and the rest rescaled, counted in a tally of its own (never triggered); the surface floor's count uses the gas's 10 K rise as its lower bound; the burner's rays pass through the silenced flame cells, and their deposits are added after all rays of the stage.

**Item 8 stays not made.** Nothing here changes 13.23.

### 13.25 Item 10: a surface's rays start on its emitting face (2026-11-13, before any run of it)

Drafted blind to every validation result, as 13.12 sets. Nothing below changes after a run of it; anything added later is dated, with its commit. Line numbers are those of commit 6038006, which the working tree's `src/core` equals. The figures called "the model" below come from a model of the solver's surface rays written for this draft: the same draws (the solver's own `pcgHash` and `cellStream`, imported read only), the same directions and faces, and the same march (Amanatides and Woo, with the tie rule of line 932), in boxes of cells with transparent gas. They are not the solver's output, and none is a fire result.

#### What the solver does now

The claim to check (13.24): every ray of an emitting solid, porous or one-dimensional wall cell starts at the cell's centre, in a direction drawn by Lambert's law about the normal of one of its open faces, and `march` leaves it in the first other solid or porous cell it enters, so a ray that crosses a side face of the emitting cell stays in the wall beside it. The code bears it out on both paths.

The CPU (`flamescapes.js`):
- Which cells emit, and how much (`radiation`, 799 to 901). A one-dimensional wall cell emits through `wallEmit` (826; 2114 to 2150): each face open to gas or to the grid's side at its own temperature (an end face at its end node, a side face at the cell's mean over its edge's area, `wallSide`, 2088), each above `emitAbove`; `faceE` holds each face's energy. Any other non-gas cell above `emitAbove` (829) emits ε σ (T_s⁴ − T_a⁴) over A times its open faces, a porous cell over max(area, A) (838), capped at half its heat above ambient (841), and its own T_s falls at once (843). `emitAbove` is 500 K (218).
- Its rays (855 to 889). `rays` = 8 (219), each carrying emitted / 8 (857). The open faces are those whose neighbour is gas or lies beyond the grid's side (`openFaces`, 904 to 914). The draws come from the cell's stream of the step, `cellStream(c, steps, 2, seed)` (860): two for a uniform direction on the sphere (865 to 870), a third for the face, drawn by its share of `faceE` for a one-dimensional wall (873 to 881) and uniformly among the open faces otherwise (882). The face's normal is added and the sum normalised (883 to 886), which gives Lambert's law about that normal. Then `march(sim, i + 0.5, j + 0.5, k + 0.5, dir, share, c, fromSolid, dep)` (888): the cell's centre.
- `march` (919 to 973). Its first cell is the one holding the start point (923), with tMax from absolute positions (928). It absorbs by soot in each gas cell crossed except the emitting one (936 to 944). A step beyond the grid returns, and what the ray carries is counted nowhere (949). The first non-gas cell takes all that is left (951 to 963): a one-dimensional wall entered along its normal on the end it crossed (954 to 960), any other cell into its mean T_s (962). After `rayReach` = 40 crossings (220, 931) what is left goes into the cell where the ray stopped (966 to 972). The `skipOwn` test (951) never acts: a straight ray cannot re-enter the convex cell it left.
- Deposits wait in `dep` and are added once every cell has emitted (891 to 900). What a ray puts into a one-dimensional wall's mean is spread over its nodes at the walls stage, the same rise in each node (2176 to 2177).

The GPU (`flamescapesGpu/kernels.js`). 13.24 did not read it. It has the same geometry. `radiateCell` (1304 onward) has the same three cases and floors (1210, 1234, 1326) and draws from `seedOf(c, 2u)` (1354), which gives the same integers as `cellStream` (898 to 902; `flamescapes.js` 543), in the same order. It calls `march(c, p, dir, share, fromSolid)` (1380) from the cell p with tMax = 0.5 / |d| on each axis (1260): the centre. A step beyond the grid returns uncounted (1281). Deposits are fixed point, 16 384 steps a kelvin (`DEP_SCALE`, 357; `deposit`, 1190 to 1193), each clamped at ±20 000 K. A validation burner is sent to the CPU (`flamescapesGpu.js` `supported`, 978).

So 13.24's geometry holds on both paths. What it does, from the model (8 rays, 200 steps, the solver's draws):
- A flat wall. From an interior cell the share that leaves through the open face is the view factor from the cube's centre to one face, (4/π)(1/√2) atan(1/√2) = 0.5541; 0.4459 enters the four cells beside it. In closed boxes of 8 × 8 × 8, 12 × 10 × 14, 6 × 16 × 10 and 40 × 8 × 8 cells, 0.443 to 0.447 of a whole hot wall's emission stays in that wall. What leaves is shared out wrongly as well: in the 12 × 10 × 14 box the walls beside the hot one take 0.111 and 0.074 each against view factors of 0.236 and 0.168, the opposite wall 0.184 against 0.194.
- Inside edges and corners. A wall cell behind the meeting of two or three walls' faces has no open face. It never emits, and it takes the rays its neighbours leak into it. In an isothermal box of 12 × 10 × 14 cells with black walls and transparent gas (check (c)'s enclosure), cells beside an edge lose 15.1 % of their emission net, cells beside a corner 28.5 %, plane face cells gain 2.9 %, and the cells with no open face gain 3.7 % of all the emission.
- Outside corners. A 2 × 2 × 2 block (three open faces a cell) puts 22 % of its emission into itself, and each wall of a cube box round it takes 0.130 against 1/6. A pillar one cell square from floor to ceiling (four open faces a cell) is 18.4 % short of balance in the isothermal box. A cell that stands alone loses nothing (13.24).
- A wall one cell thick. A lumped partition puts 0.447 of its emission into itself. In a one-dimensional partition, 0.444 of a heated face's emission enters the neighbouring cells' means, which the walls stage spreads through their thickness (2176 to 2177): that heat crosses the board within one step.
- The grid's side. A plate whose back is the grid's side puts 0.445 of its emission into itself.
- The reach. In the 40 × 8 × 8 box no ray from the hot end reaches the far wall within 40 crossings (its view factor is 0.0124), and 1.9 % of the hot end's emission is dropped into transparent gas. With rays started on the face, 2.0 % of all surface rays in a closed box of 25 × 13 × 20 cells (5.0 × 2.6 × 4.0 m on 0.2 m cells) need more than 40 crossings.

#### The burner's rays, and why they are outside this item

`createBurner` (3113 onward) sets 16 fixed directions on a golden spiral about the vertical (3182 to 3191) and its origin at the centre of the axis cell (3198), the lowest layer's cell nearest the burner cells' centroid, an open cell. `burnerStep` turns the directions about the vertical each step (3367) and calls `march(sim, b.origin[0], b.origin[1], b.origin[2], b.dir, radiant / rays, b.axis, false)` (3374), with no `dep`.

13.24's defect does not apply. The start cell is open, so each ray's first cell beyond it is the cell it truly enters, gas or a surface it faces. Another question does apply: the radiant part leaves a point half a cell above the burner's top, and 8 of the 16 directions point below the horizontal (y_r = 1 − 2(r + ½)/16), so half of χ_r goes into the burner's top and the floor round it. That concerns where a flame's radiation comes from, not how a surface emits, and it is not in this item. It is stated as a limit below. With this item on, the burner's rays are unchanged, and what they carry beyond the grid stays uncounted.

#### Porous cells, and why they do not change

A porous cell is gas and fuel in one cell. The flow passes through it, and its fuel exchanges heat with its own gas over max(area, A) (`conduct`, 1863 to 1868). Its fuel's exposed area is `contentsArea` = 2 h² a contents cell (235, 388), or the ignition item's (`itemArea`, 1 m², 236). It emits over max(area, A) (838), which may exceed the area of its open faces: a cell in a layer of contents has one open face and emits over 2 h². It has no single emitting face. Started on its open faces, all of that would leave the pack, twice what a black face at T_s could send. So the rays of a porous cell keep their present start, the cell's centre, and their present reach, and the cells beside it in the pack keep taking what crosses into them. Whether that matches a pack's own absorption (a medium of opaque elements, whose absorption coefficient is its elements' projected area per volume) is not checked; it is stated as a limit. A ray from a solid face that enters a porous cell is absorbed there whole, as now.

#### The method

Chosen: the rule of Monte Carlo ray tracing for a diffuse, isothermal surface element, a point drawn uniformly on the element and a direction drawn by the cosine law about its normal (Howell, Mengüç, Daun and Siegel 2020; Modest 2013). The solver already draws the direction by that law and the face by its share. The item moves only the start, from the cell's centre to a uniform point on the drawn face, and enters the march in the gas cell beyond that face.

Considered and set aside:
- A start at the face's centre. It removes the leak, but every ray of a face then leaves one point, so the face sends the view factor of a point, not of the face. From a face in an inside edge the error is of the order of the leak it removes.
- Scaling the energy by 1/0.5541. It is right only for interior cells of a flat wall. At edges, corners and pillars the leak differs (above).
- Moving the start point off the face by a small length and finding its cell with `floor`. On the GPU, positions are float32: at cell 500 one step is 3 × 10⁻⁵ of a cell, so any smaller nudge rounds back onto the face, and `floor` then finds the emitting cell. Naming the start cell by its index avoids this.

1. Scope. The item applies to the rays of a lumped solid cell (kind `SOLID`, not a one-dimensional wall) and of a one-dimensional wall cell. The rays of porous cells, of gas cells and of the burner are as now.
2. Direction and face, as now. Per ray, from the salt-2 stream in the present order: two draws for the uniform direction u, the third for the face f (by `faceE` for a one-dimensional wall, uniformly among the open faces otherwise); d = (u + n_f) / |u + n_f|. These are the draws of the item off, ray for ray.
3. A guard. If d · n_f ≤ 0, then d = n_f. This happens only when u equals −n_f to round-off: on a face whose normal is +y when the first draw is 0 (one ray in 2²⁴), on other faces when two draws take exact values (one in 2⁴⁸). The present path leaves a zero or tangent direction there (885).
4. The start point. Two draws a ray, u₁ and u₂, from a second stream of the cell in the step, `cellStream(c, steps, 3, seed)`, taken in ray order. Salt 3 is drawn by no path now (1, 2 and 5 are in use), and nothing else draws from this stream. Let a be the axis of n_f, and b₁ and b₂ the two other axes in the order of the grid's indices i, j, k. In the frame of the gas cell beyond the face, the start's fractions φ are: along a, 0 if n_f points up the axis and 1 if down it; along b₁, u₁; along b₂, u₂. The start is then uniform on the face. A second stream keeps every ray's direction and face as with the item off, so on and off differ in the start alone.
5. One-dimensional walls. An end face (across the wall's axis) is the whole cell face. A side face emits over its edge's area, thickness × h (`wallSide`), but the solver does not hold where in the cell the wall lies, so its start is uniform on the whole cell face.
6. The start cell. It is named by index: g = (i, j, k) + n_f. If g lies beyond the grid's side, the ray leaves the grid at once: its share is added to `radiatedOut` (point 9) and no march is made. Otherwise the march starts in g, which is gas, being an open face's neighbour.
7. The march's entry. `march` is entered with the start cell g and the fractions φ in place of a position. On each axis tMax = (1 − φ)/|d| if d > 0, φ/|d| if d < 0, and ∞ if d = 0; tDelta is as now. Along n_f's axis this is 1/|d|: the ray crosses g. The first segment lies in g, and g absorbs by soot over it like any gas cell (g is not `from`, which stays c). Everything after is as now: absorption in gas cells, the whole remainder into the first non-gas cell, a one-dimensional wall's end when it is entered along its normal. Because g is named by index, the first cell is the gas cell beyond the face on both paths, whatever the rounding of positions; because φ is local, float32 on the GPU is as exact as float64 on the CPU at any cell index.
8. The reach. These rays march until they enter a non-gas cell, leave the grid, or carry less than 10⁻⁹ of their energy (as now): at most n_x + n_y + n_z crossings, the most a straight line can make in the grid. `rayReach` does not stop them. With the leak gone, a surface's rays cross the room, and dropping one after 40 crossings into gas that cannot absorb it would heat that gas (above: 2.0 % of rays in a 5 m room on 0.2 m cells). The rays of gas and porous cells keep `rayReach`.
9. Tallies, in J, only with the item on: `stats.radiatedOut`, what the stage's rays carry beyond the grid's sides (every ray of the radiation stage; the burner's rays, cast in the `pilot` stage with no `dep`, stay uncounted); `stats.surfaceRays` and `stats.surfaceCrossings`, the number of rays of point 1 and of the cells they cross. 13.19 point 10 gives the first the same name, so if both items are made they share it.
10. On the CPU.
    - `DEFAULTS`: `surfaceRayStart` 0.
    - `radiation`: with it 1, for the cells of point 1, the salt-3 stream, points 3 to 8, and the tallies. `dep` gains `out`, a Float64Array of one element.
    - `march` takes an optional start {cell g, fractions φ, crossing limit}. Without it (the item off, gas and porous rays, the burner) lines 923 to 928 and 931 run as now. When `dep.out` exists, a step beyond the grid adds what is left to it before returning (949).
    - The stage is exported for the check tool (`radiationStage`); no run calls it.
11. On the GPU.
    - `PARAMS`: `surfaceRayStart` (u32).
    - `radiateCell`: with it 1 and the cell `SOLID` or a one-dimensional wall, the stream `seedOf(c, 3u)` seeded before the ray loop, the guard, u₁ and u₂, the start cell named by index, and the outflow when the start cell lies beyond the grid, tested with `inside` before any `atI`.
    - `march` gains the start fractions (vec3) and its crossing limit, and returns what it carried beyond the grid (0 otherwise). The present call passes (0.5, 0.5, 0.5) and `U.rayReach`, for which tMax is 0.5/|d| as at 1260.
    - `radEmit` reduces the outflow beside the emission (its vec4's second slot) into a second reduction slot, read back with `radiated` as the frame's `radiatedOut`.
    - Deposits stay in kelvin, as now. A validation burner stays on the CPU.
12. What stays exactly as now: which cells emit and when (the floors); how much (A per open face, max(area, A) for porous cells, ε, the cap); the emitting cell's own update; the deferred deposits; 8 rays a cell and their shares; every direction and face drawn; `march`'s absorption rule (soot in gas, the rest whole in the first non-gas cell, absorptivity 1); one-dimensional walls' ends; porous cells; gas cells; the burner; `stats.radiated`; the GPU's fixed point.

#### The switch

`surfaceRayStart`: 0 (default), the cell's centre, as now; 1, the emitting face. With 0 the new code costs one test of the flag for each surface cell that emits and one untaken test for each ray that leaves the grid, and every output is bit-identical to the build before (check (h)). When the default changes, `RUN_CACHE_VERSION` (`runCache.js` 54, now 13) rises.

#### Checks (fixed now, before any run)

All but (h) to (j) run in a new tool, `tools/surface-rays-check.mjs`. It builds each scene as a grid of cells by hand (as `flamescapes.wall.test.mjs` builds its mask), sets every gas cell to T_a with no soot, so the gas is transparent, and every surface to its temperature. It calls the radiation stage alone for the steps stated, dt 0.05 s, seed 1, adds 1 to `sim.steps` after each call, and sets every temperature back (gas, T_s, wall nodes, `last`). Each step it reads the deposits before anything else adds to them, from `sim.work`'s `radGas`, `radSolid`, `radWallLo` and `radWallHi`, turned into J by the capacity the stage divided by (ρ c_p V; max(`solidCapacity`, 10⁻⁶)), and the tallies. The material is stone's ρ and c with ε = 1 (passed as `inert`). A box is given by its inner size in cells, n_x × n_y × n_z, with walls two cells thick whose outer layer stays at T_a and does not emit; a wall's share is read from the cells of its two layers, an edge column counted with the first of x-low, x-high, y-low, y-high, z-low, z-high that holds it. Cells of 0.20 m and 0.34 m. No measured data enters any check.

Tolerances. Where the rays of a case carry equal shares, a share F read from N rays has standard deviation √(F(1 − F)/N), F being the reference, and the case passes within 4 of them. Where shares differ (cells with several open faces), the energy a group g receives has a standard deviation of at most √(s_max E_g), s_max being the largest share of any ray in the scene and E_g what the group should receive, and the case passes within 4 of that. Reason: these are the ray-count statistics of a method whose expectation is the reference exactly. A correct code exceeds 4 standard deviations in one case in 16 000, so in under 0.4 % of the 60 or so cases below together. The numbers of rays are fixed here, and the tolerances follow from them.

(a) A flat wall's emission reaches the room. Boxes of 8 × 8 × 8, 12 × 10 × 14, 6 × 16 × 10 and 40 × 8 × 8 cells, the x-low wall's inner layer at 1000 K, 200 steps.
1. The x-low wall (both layers, its edge columns included), the gas and the grid's sides take exactly 0. This is an equality, not a tolerance: no ray can deposit there.
2. Each other wall takes its view factor within 4 standard deviations: the opposite wall F₁₁(n_y, n_z; n_x), each y wall F₁₄(n_y, n_x; n_z), each z wall F₁₄(n_z, n_x; n_y). Here F₁₁(a, b; c) is the view factor between directly opposed parallel rectangles a × b at distance c; with X = a/c and Y = b/c,

   F₁₁ = (2/(πXY)) [½ ln((1 + X²)(1 + Y²)/(1 + X² + Y²)) + X √(1 + Y²) atan(X/√(1 + Y²)) + Y √(1 + X²) atan(Y/√(1 + X²)) − X atan X − Y atan Y];

   and F₁₄(w, h; l) is the view factor from a w × l rectangle to an h × l rectangle at 90° that shares its edge of length l; with W = w/l, H = h/l and S² = W² + H²,

   F₁₄ = (1/(πW)) [W atan(1/W) + H atan(1/H) − S atan(1/S) + ¼ ln( (1 + W²)(1 + H²)/(1 + S²) · (W²(1 + S²)/((1 + W²) S²))^(W²) · (H²(1 + S²)/((1 + H²) S²))^(H²) )].

   Both were transcribed for this draft and checked against an independent Monte Carlo quadrature of 4 × 10⁶ samples (uniform points, cosine directions, exact intersection of a ray with a plane) on twelve configurations; every one agreed within its standard error of 2.5 × 10⁻⁴, and in each box the five shares sum to 1 within 10⁻⁸. The references, ± 4 standard deviations:

   | Box | Rays | Opposite wall | Each y wall | Each z wall |
   |---|---|---|---|---|
   | 8 × 8 × 8 | 102 400 | 0.1998 ± 0.0050 | 0.2000 ± 0.0050 | 0.2000 ± 0.0050 |
   | 12 × 10 × 14 | 224 000 | 0.1938 ± 0.0033 | 0.2355 ± 0.0036 | 0.1676 ± 0.0032 |
   | 6 × 16 × 10 | 256 000 | 0.4237 ± 0.0039 | 0.1083 ± 0.0025 | 0.1798 ± 0.0030 |
   | 40 × 8 × 8 | 102 400 | 0.0124 ± 0.0014 | 0.2469 ± 0.0054 | 0.2469 ± 0.0054 |

   The 40 × 8 × 8 box tests point 8: its far wall lies beyond 40 crossings.
3. Reciprocity. In the 12 × 10 × 14 box with the floor (y-low) at 1000 K instead, A_x F̂(x-low → floor) from the first run and A_y F̂(floor → x-low) from this one agree within 4 standard deviations of their difference, 0.72 cell² on 32.97 cell² (2.2 %), and each lies within its own 4 standard deviations of A_x F₁₄(10, 12; 14) = A_y F₁₄(12, 10; 14) = 32.97 cell².

Reported beside, with the item off: the hot wall's own share (the model: 0.443 to 0.447) and each wall's. A failure of 1 means a ray starts in, or returns to, its own wall. A failure of 2 or 3 means the start point, the direction or the march's entry departs from the method.

(b) Outside corners. A 2 × 2 × 2 block at the centre of a 12 × 12 × 12 box, at 1000 K, all else at T_a, 1000 steps (64 000 rays). The block takes exactly 0 of its own emission (a convex block cannot see itself). Each of the six walls takes 1/6 within 0.0059 (4 standard deviations); 1/6 is exact by the symmetry of a centred cube in a cube. Reported beside, with the item off: the model's 0.222 into the block and 0.130 a wall.

(c) An isothermal enclosure with transparent gas. The 12 × 10 × 14 box with a 2 × 2 × 2 block at its centre (inner cells 5 to 6, 4 to 5, 6 to 7) and a pillar one cell square from floor to ceiling at inner cell (2, ·, 3); every solid cell not on the grid's outer layer at 1000 K, ε = 1; 200 steps. The groups, by geometry: a wall's face cells whose gas neighbour touches no other solid cell in the face's plane (group i, 582 cells), one (ii, edges, 248), two (iii, corners, 24); cells of the walls' inner layer with no open face (iv, 154); the block (v, 8); the pillar (vi, 10).
- Each group's net, what it receives less what it emits, lies within 4 √(s_max E_g), s_max being a pillar cell's share, 4 A σ (T⁴ − T_a⁴) dt / 8: that is 0.8 %, 1.3 %, 4.1 %, 4.1 % and 3.2 % of the emission of groups i, ii, iii, v and vi.
- Group iv receives exactly 0.
- Reference: 0 for every group, by reciprocity. Every face emits σ (T⁴ − T_a⁴) over its area and absorbs all that reaches it, and each face's view factors to a closed enclosure sum to 1.

The model gives, item on, nets of +0.04 %, −0.07 %, −0.5 %, +0.4 % and −0.2 %; item off, +2.9 %, −15.1 %, −28.5 %, +5.3 % and −18.4 %, with group iv taking 3.7 % of all the emission. The solver's own figures with the item off are reported beside, not judged.

(d) Walls one cell thick with gas on both sides. The 12 × 10 × 14 box divided at inner x = 6 by a partition across its whole section, 200 steps.
1. Lumped, at 1000 K, all else at T_a. The partition takes exactly 0. The half-box on its low side takes ½ of its emission within 0.0042 (4 standard deviations of 224 000 rays at ½: the face is drawn uniformly between two), and the high side the rest.
2. One-dimensional (`wallLayers` on, the board of `flamescapes.wall.test.mjs`, ε = 1): its high end node at 1000 K, its other nodes at T_a, its T_s and `last` at its nodes' mean, and every other surface at T_a, so its high face alone emits. The partition's cells, means and ends take exactly 0. The high side's half-box takes the whole emission and the low side's nothing, within 10⁻⁹ of the emission (no ray can cross the plane of the face it left).

Reported beside, with the item off: the share that enters the partition (the model: 0.447 lumped, 0.444 one-dimensional) and, for 2, what the walls stage then puts into the low end node in one step.

(e) The grid's side. A gas region of 10 × 10 × 10 cells closed on five sides by walls two cells thick at T_a; on its sixth (x-low) side a layer one cell thick on the grid's side, whose 10 × 10 cells facing the gas are at 1000 K and whose rim is at T_a, so each hot cell has two open faces, its back on the grid's side and its front on the gas. 200 steps (160 000 rays). The layer takes exactly 0. `radiatedOut` (the rays drawn on the back face, point 6) is ½ of the emission within 0.0050, and the walls take the other ½ within the same. Reported beside, with the item off: the share into the layer (the model: 0.445).

(f) Energy. In every step of (a) to (e), what the stage emitted (`stats.radiated`) equals the J deposited in gas, solids and walls' ends plus `radiatedOut`, within 10⁻⁹ of the emission. Reason: conservation holds by construction, and float64 sums of 10⁵ deposits err by about 10⁻¹³.

(g) The CPU and the GPU cast the same rays. On 0.20 m cells: (a)'s 8 × 8 × 8 and 12 × 10 × 14 boxes, (c)'s enclosure and (d)1. From one state, one radiation stage on each path: on the GPU its radiation passes alone (`radEmit` on a state copied from the CPU's, through a hook the build adds, as 13.19's parity needed), its deposits read from the fixed-point accumulators before `radApply` and turned into J with the CPU's capacities.
1. The emission agrees within 10⁻⁵ relative.
2. Each wall's and group's received energy, and `radiatedOut`, lie within 1 % of the CPU's.
3. The mean absolute difference of the cells' deposits, GPU against CPU, is at most 0.1 of the CPU's own difference between seeds 1 and 2.

Then (a), (b) and (c) run 200 steps on the GPU's passes and pass as on the CPU, and the GPU's books close within 1 %.

Reason. The two paths draw the same integers (`cellStream` and `seedOf`) and name the same start cell, and their fractions k/2²⁴ are exact in float32. They differ by the float32 rounding of the direction, which changes the cell a ray enters only where its path passes within about 10⁻⁵ of a cell of an edge, and by the fixed point: at 1000 K a ray's share into a 0.20 m stone cell is 13.3 steps, so rounding is at most ±4 % of a deposit, unbiased, and about 0.2 % of the smallest group's sum in one stage. Between seeds, a cell's difference is of the order of a share times the square root of its hits, about 45 times the rounding, so 0.1 separates the same rays from different ones with a margin of 4. The 1 % is 13.4's closure, applied here.

(h) Off is off. With `surfaceRayStart` 0: `tools/flamescapes-fingerprint.mjs` (300 steps, both cases) gives the base's fingerprints; the unit suite passes with the same count; `tools/flamescapes-gpu-check.mjs`'s phases give the base's state comparisons bit for bit (timings aside). With 1 the fingerprints differ, which is reported.

(i) The unit suite passes unbent with the default set to 1 for the run (a one-line change, not committed). A test that fails is not edited, and its failure is reported.

(j) Speed, reported, not judged. Ms a step and the radiation stage's ms (the fingerprint tool's profile), on and off; `tools/backend-bench.mjs` on its room, Duplex, Office, Clinic and Vienna, CPU and GPU, on and off; `stats.surfaceRays` and the mean crossings a surface ray. In the model the mean crossings a surface ray rise from 8.1 to 11.8 in the 12 × 10 × 14 box and from 11.6 to 18.0 in the 25 × 13 × 20 box, since a leaked ray stopped after one crossing.

#### Default

Off until (a) to (i) pass. Then on, with `RUN_CACHE_VERSION` raised. Made after 13.B began, it gets its own full rerun on 13.B's base, reported beside 13.B (13.12, rule 5).

#### What ends this line

If any check of (a) to (i) fails in any case, the attempt fails, and it is reported with its numbers and commit (13.12, rule 3). No tolerance moves and nothing is fitted; the item has no free constant. The rays stay at the cell's centre, 13.24's geometry is stated as a limit of every verdict, and item 11 (13.26) cannot be made. Checks (a) to (e) have references that the method meets exactly in expectation, so a failure there is a departure of the code from the method; whether corrected code counts as the same attempt is for the coordinator to settle before the run. No second attempt with another method is drafted. It would need the user's word and a new method committed here first.

#### Declared limits

- Surfaces still take all of each ray that reaches them and emit at their ε, with no reflection, as now. The exchange between two gray surfaces is then overstated: between parallel plates of ε 0.9 the solver's ε σ (T₁⁴ − T₂⁴) against the gray value σ (T₁⁴ − T₂⁴)/(2/ε − 1), 0.900 against 0.818 (0.700 against 0.538 at ε 0.7). 13.26 says what this does to a balance.
- Porous cells keep the centre start, and the cells beside them in a pack keep taking what a cell's rays carry into them.
- The burner's radiant part leaves a point half a cell above its top, half of it downward into the burner and the floor.
- A one-dimensional wall's side face starts its rays on the whole cell face, not on its edge's strip.
- An oblique or curved surface is a staircase of cell faces, each emitting about its own axis's normal. The item makes each face exact, not the staircase.
- Gas cells' rays still start at the centre and skip their own cell (that is item 8's point 6, 13.19).
- Surfaces still emit only above 500 K (item 11, 13.26, is separate).

#### Not verified

- The size of the change in any validation run or app scene.
- The present figures above are the model's, not the solver's. Checks (a) to (e) report the solver's own beside.
- That no ray on the GPU enters a different cell from the CPU's beyond the float32 rounding argued in (g).
- The sources' Monte Carlo chapters were not re-read for this draft; the rule is the field's standard one, quoted from them. FDS's treatment of a wall's emission was not read.

#### Sources (read, or run for this draft)

- This repository at 6038006: `flamescapes.js` 70, 94 to 107, 114 to 265 (parts of `DEFAULTS`), 340 to 410, 473, 520 to 600, 767 to 973, 1855 to 1880, 2060 to 2200, 3095 to 3240 and 3339 to 3456; `flamescapesGpu/kernels.js` 95 to 140, 890 to 905 and 1180 to 1415; `flamescapesGpu.js` 978 (and its `radiated` read-back, by search); `runCache.js` 54; `tools/flamescapes-fingerprint.mjs`; `tools/flamescapes-gpu-check.mjs` 124 to 180 and 281 to 320; `tools/backend-bench.mjs` 1 to 70; the unit tests that set `emitAbove` (by search).
- Howell, Mengüç, Daun and Siegel (2020). Thermal Radiation Heat Transfer, 7th edition. CRC Press. doi:10.1201/9780429327308. Checked on Crossref; metadata only.
- Modest (2013). Radiative Heat Transfer, 3rd edition. Academic Press (Elsevier). doi:10.1016/C2010-0-65874-3. Checked on Crossref, which gives its title, publisher and year but no author; metadata only.
- Run for this draft: the model of the present and the proposed ray geometry described at the head of this section, and the independent Monte Carlo quadrature of F₁₁ and F₁₄. No validation run, no fire simulation, no result file.

### 13.26 Item 11: surfaces emit from just above ambient (2026-11-13, before any run of it)

Drafted blind with 13.25, as 13.12 sets, and with its line numbers. Nothing below changes after a run of it; anything added later is dated, with its commit.

#### What the solver does now

The claim to check (13.19, decision 7; 13.24): a surface emits only above `emitAbove` = 500 K, and absorbs whatever reaches it, at any temperature. The code bears it out.
- The floors. A lumped or porous cell emits when T_s > `emitAbove` (829). A one-dimensional wall casts only if an end node or its mean exceeds `emitAbove` (2119), and each face only if its own temperature does (2135). The GPU has the same (1210, 1234, 1326).
- Absorption. The first non-gas cell takes all that a ray carries, whatever its temperature and ε (951 to 963; the GPU 1283 to 1293).
- Emission is written net of the ambient's, ε σ (T_s⁴ − T_a⁴) (839, 2137), so a surface at T_a emits nothing, and one below T_a nothing either (842).

So a wall between T_a and 500 K is black to every ray and returns none. It takes the gas's and the hot surfaces' radiation and gives nothing back, and two such walls exchange nothing however different their temperatures. The floor is a step in a surface's emission: at ε 0.9, 2.81 kW/m² just above 500 K and nothing just below. At 450 K the emission omitted is 1.72 kW/m².

Two facts of the present storage bear on a lower floor.
- Float32 temperatures. T and T_s are Float32Arrays (407 to 408; the GPU's `cf` likewise). Between 256 and 512 K a float32 step is 3.05 × 10⁻⁵ K, so a change below 1.53 × 10⁻⁵ K in one write is lost. A lumped stone cell (ρ c = 2.16 × 10⁶ J/(m³ K)) with one open face at ε 0.9 changes by less than that in a step of 0.1 s while its surface is within 12.1, 16.0 and 19.7 K of ambient on 0.20, 0.27 and 0.34 m cells (22.8, 29.8 and 36.3 K at 0.05 s). Its convection near ambient is lost in the same way now.
- The GPU's fixed point. At T_a + 10 K (ε 0.9, dt 0.1 s) a ray carries 0.027, 0.049 and 0.078 J on 0.20, 0.27 and 0.34 m cells. Into a stone cell that is 0.026, 0.019 and 0.015 of a step of 1/16 384 K, which rounds to 0. A lower floor on the GPU in kelvin would deliver nothing.

#### Item 11 needs item 10

It may not be made without item 10, and its switch does nothing while item 10's is 0.
- With centre starts, 45 % of a flat wall's emission stays in the wall (13.25). A lower floor would extend that to every warm wall.
- In one-dimensional walls the leak enters the neighbours' means, which the walls stage spreads through the board (2176 to 2177): 44 % of a warm face's emission would cross the board within a step. The validation's rooms are lined with one-dimensional boards (13.4, item 1), so every warm board would conduct by this path.
- With centre starts an isothermal enclosure does not balance at edges and corners (13.25, What the solver does now: −15 % and −29 %, and the cells with no open face gaining). Item 11's balance checks would then fail on item 10's geometry, not on item 11.

#### Emission and absorption: what balances

Surfaces emit at ε and absorb at 1, with no reflection. In an enclosure at one temperature T with transparent gas, a surface i of area A_i emits ε_i σ (T⁴ − T_a⁴) A_i and receives Σ_j ε_j σ (T⁴ − T_a⁴) A_j F_ji, which by reciprocity is σ (T⁴ − T_a⁴) A_i Σ_j ε_j F_ij. Its net gain is

σ (T⁴ − T_a⁴) A_i (Σ_j ε_j F_ij − ε_i).

- With one ε everywhere, Σ_j F_ij = 1 makes it 0. The balance closes for every surface, at any T, on item 10's geometry (13.25, check (c)). Nothing must change for that.
- With mixed ε it does not close. A surface whose ε is below that of the surfaces it sees gains heat at one temperature, which the second law forbids. The solver's materials are stone and glass at ε 0.9 and metal at 0.7 (103 to 107), and the validation passes its own. A metal surface among stone ones gains 0.2 σ (T⁴ − T_a⁴): 40, 86, 207 and 625 W/m² at 323, 350, 400 and 500 K.
- The 500 K floor makes this worse, not better. Below it a surface's emissivity is in effect 0 against an absorptivity of 1, so every cool surface keeps whatever reaches it. With item 11 a surface above T_a + 10 K has the imbalance 1 − ε in place of 1, and none when all share one ε.
- For the balance to close with mixed ε, a surface must absorb ε of each ray and reflect the rest diffusely (Kirchhoff's law for a gray, diffuse surface). That changes `march` and casts a new ray at each reflection. It is not in this item; it is declared below, and it is a candidate for its own item, drafted blind.

Check (d) holds both: 0 with one ε, and with a block at 0.7 among surfaces at 0.9, the gain the formula gives, which measures the declared limit.

#### The method

1. The floor. With `surfaceEmitFloor` 1 and `surfaceRayStart` 1, a lumped solid cell (kind `SOLID`, not a one-dimensional wall) emits when T_s > θ, and a one-dimensional wall when an end node or its mean exceeds θ (the test at 2119), each face when its own temperature does (2135), where θ, the surfaces' floor, is

   θ = min(`emitAbove`, T_a + `surfaceEmitRise`) if `emitAbove` < 10⁴ K, and θ = `emitAbove` otherwise,

   with `surfaceEmitRise` = 10 K. The item never raises the floor: an `emitAbove` set below T_a + 10 K stays. An `emitAbove` of 10⁴ K or more lies above any temperature the solver reaches; five unit-test files use 10⁹ K to remove radiation (`flamescapes.leak`, `flamescapes.soot`, `flamescapes.wall`, `flamescapes.zones` and `flamescapesConservation`), and the item keeps that meaning. Porous cells and gas keep `emitAbove`.
2. Why T_a + 10 K. It is a numerical cut fixed here from no data, the same rise 13.19 point 5 fixed for the gas (`gasEmitRise`), so that a surface the fire has not warmed casts no rays; with no floor, every surface that any ray had touched would cast 8 rays a step. Its size: at T_a + 10 K a surface of ε 0.9 would emit 54 W/m², 1.9 % of what it emits at 500 K and about half its convection at the same rise (10 W/m² K, so 100 W/m²). What it omits is counted (point 5).
3. Emission, as now: ε σ (T_s⁴ − T_a⁴) over A per open face (a wall's side over `wallSide`), net of the ambient's, with the cell's ε. The rays are cast as item 10 casts them.
4. The cap, as now: at most half the heat above ambient in a step (841; 2138 for a wall's node). Near ambient the emission, about 4 ε σ T_a³ (T_s − T_a) A dt, and the cap, ½ C (T_s − T_a), are both linear in the rise, so the cap binds only where C < 8 ε σ T_a³ A dt = 10.3 A dt J/K (ε 0.9). On 0.27 m cells at dt 0.1 s that is C < 0.075 J/K, a cell holding under 2 × 10⁻⁶ of a cell of stone. What it withholds is counted.
5. Tallies on the CPU, in J, with the item on:
   - `stats.surfaceFloor`: what lumped and one-dimensional surfaces with T_a < T ≤ θ would have emitted over their open faces, computed without casting. 13.19's decision 7 names the same quantity with θ = 500 K.
   - `stats.surfaceCapped`: what the cap withheld from surfaces.
   - `stats.radRoundoff`, two sums: the J by which the stage's float32 writes of the emitting cells' own temperatures (843; wall nodes in `wallEmit`), and of its deposits (891 to 900), differ from the float64 changes the stage meant, each (stored − meant) × capacity.
6. The GPU's deposits in joules, with the item on. Every radiation deposit (gas, solid, a wall's end) is accumulated in J at 4096 steps a joule, the representation 13.19 point 12 fixed for item 8 (if both are made, they share it). Each deposit is clamped at ±4096 J and the clamps are counted. `radApply` divides by the capacity the CPU divides by: ρ c_p V for gas, max(`solidCap`, 10⁻⁶) for a solid, the same for a wall's end through `wallEnd`. The smallest step is 0.24 mJ, so a ray at T_a + 10 K is 111 to 320 steps (0.20 to 0.34 m, dt 0.1 s); the most a cell can hold in a step is 2³¹/4096 J = 524 kJ. With the item off the deposits stay in kelvin.
7. GPU uniforms: `surfaceFloor` (f32), which the host sets to θ with the item on and to `emitAbove` with it off, read in `wallEmit` (1210, 1234) and in `radiateCell`'s branch for a `SOLID` cell (1326) in place of `U.emitAbove`; and `depJoules` (u32). The porous and gas branches read `U.emitAbove` as now.
8. What stays as now: gas and porous cells, the burner, absorption at 1, emission at ε, item 10's rays, and the cap.

#### The switch

`surfaceEmitFloor`: 0 (default), θ = `emitAbove`, as now; 1, θ as in point 1, with `surfaceEmitRise` 10 K. It acts only with `surfaceRayStart` 1. With 0 the stage compares with one floor computed once a stage, of the same value as now, and every output is bit-identical (check (a)). `RUN_CACHE_VERSION` rises when the default changes.

#### Cost

With the item on, every surface above T_a + 10 K casts 8 rays a step, where now only those above 500 K do, and with item 10 each crosses the room: in the model, about 12 crossings a ray in a 2.4 × 2.0 × 2.8 m box and 18 in a 5.0 × 2.6 × 4.0 m one, on 0.2 m cells. The 5.0 × 2.6 × 4.0 m room has 2170 face cells, so once all its walls are warm it casts about 17 000 surface rays and 3 × 10⁵ crossings a step. Surfaces the fire has not warmed by 10 K cast nothing, so the cost follows the warmed region, not the size of the building. The cost on is reported (check (h)), not judged. Off it is nothing: the switch changes one number the stage already compares with.

The user's rule asks that a feature be off by default, cost nothing when off, and still run Vienna on the web. Check (h) runs Vienna on the GPU in the browser with both items on and reports whether it reaches its horizon, and at what ms a step.

#### An option that holds the cost down

`surfaceWarmRays`, an integer from 1 to `rays`, default `rays` (8), at which it changes nothing. Set lower, a surface cell all of whose emitting faces are at or below `emitAbove` casts that many rays, each carrying its emission divided by that number, with the same draws in the same order (fewer of them). Each ray is drawn as before, so the expected deposit in every cell is unchanged, and the cell's whole emission is carried every step; only the noise grows, as 1/√(number of rays). Both paths decide the count by comparing a float32 temperature with 500 K, which is exact on both. Its check is (k). It is not proposed as a default; the coordinator decides before any run whether to take it.

#### Checks (fixed now, before any run)

In `tools/surface-rays-check.mjs`, set up as in 13.25, with `surfaceRayStart` 1 throughout unless stated. Tolerances as in 13.25: 4 standard deviations of the ray counts.

(a) Off is off. With `surfaceEmitFloor` 0, and with `surfaceEmitFloor` 1 and `surfaceRayStart` 0: fingerprints, unit suite and the GPU parity phases' state comparisons identical to the build with item 10 alone at the same `surfaceRayStart`.

(b) The floor. In a box of 4 × 4 × 4 cells with every surface at T_a, one floor cell at T_a + Δ, Δ ∈ {−5, 0, 5, 9.99, 10.01, 11, 50, 206.84, 206.86} K (the last two about 500 K), one step. With the item on, the stage emits ε σ (T_s⁴ − T_a⁴) A dt from that cell exactly when T_s > T_a + 10 K, and `stats.surfaceFloor` holds that same formula's value exactly when T_a < T_s ≤ T_a + 10 K, each within 10⁻¹² relative; with the item off, the cell emits only at 206.86 K above ambient. With `emitAbove` 10⁹ K nothing emits, on or off; with `emitAbove` 300 K and the item on, θ is 300 K. The same for a one-dimensional wall's end face, its end node at each temperature. Reason: one formula in float64; this checks which cells pass the floor.

(c) Exchange near ambient. 13.25 (a)'s 12 × 10 × 14 box on 0.20 and 0.34 m cells: the x-low face at 323.15 K, the x-high face at 313.15 K, every other surface at T_a + 5 K (below θ: they absorb and do not emit); 200 steps.
- The x-high face receives F₁₁(10, 14; 12) E_lo = 0.1938 E_lo, and the x-low face 0.1938 E_hi, each within 4 standard deviations (1.7 % of each). Each y wall receives 0.2355 (E_lo + E_hi) and each z wall 0.1676 (E_lo + E_hi), within 4 standard deviations of the sum of two binomials.
- The net exchange between the two faces is σ (323.15⁴ − 313.15⁴) A F₁₁, 14.16 W/m² of face, within 4 standard deviations (5.6 % of it).
- With the item off nothing is emitted, which is reported.
- On the GPU (joule deposits), every wall's received energy lies within 1 % of the CPU's. With kelvin deposits it is reported beside; by the figures above each share is 0.04 of a kelvin step, so it should read 0.

(d) An isothermal enclosure near ambient. 13.25 (c)'s enclosure at 323.15 K, 200 steps.
1. ε = 1: every group within 13.25 (c)'s bound, and group iv exactly 0.
2. ε = 0.9 everywhere: the same.
3. The block at ε 0.7, all else at 0.9: the block's net gain within 4 √(s_max E_g) of 0.2 σ (T⁴ − T_a⁴) × its open area × dt × steps, which is 0.286 of its emission, and the other groups' nets summing to minus the block's within 10⁻⁹ of the emission. This checks the code against the declared model, not the physics; the figure is the declared limit's size.

(e) Energy. In every step of (b) to (d): on the CPU as in 13.25 (f), within 10⁻⁹; on the GPU (joules), the emission equals the accumulators' J plus `radiatedOut` within 1 %, with no deposit clamped. Reason: as 13.25 (f) and (g); in joules a share in (c) and (d) is 130 to 600 steps (313.15 to 323.15 K, 0.20 to 0.34 m, dt 0.05 s), so rounding is under 0.4 % of a deposit and unbiased.

(f) GPU parity near ambient. (c)'s box and (d)1 on 0.20 m cells, one stage from one state: the emission within 10⁻⁵ relative; each wall's and group's received energy within 1 % of the CPU's; the mean per-cell difference at most 0.1 of the CPU's between two seeds. Reason as in 13.25 (g), with about 200 joule steps a share.

(g) The unit suite passes unbent with both items' defaults set to 1 for the run (not committed). By point 1, the five files that pass `emitAbove` 10⁹ K keep no radiation.

(h) Speed and the floor's size, reported, not judged: as in 13.25 (j), with both items on against item 10 alone; the number of emitting surface cells a step; `stats.surfaceFloor` and `stats.surfaceCapped`; and Vienna (`backend-bench` case `vienna`) on the GPU in the browser with both items on, whether it reaches its horizon, and its ms a step.

(i) Float32 rounding, reported, not judged. (c)'s box on 0.27 m cells, stone, ε 0.9, dt 0.1 s, the x-low face at T_a + 15, 30 and 60 K in turn and every other surface at T_a + 12 K, with the temperatures written and not set back, 200 steps: both sums of `stats.radRoundoff` against `stats.radiated`.

(k) The option, only if the coordinator takes it before the run: (c) and (d)1 again with `surfaceWarmRays` 1 and 1600 steps (the same number of rays) pass with the same tolerances, and the standard deviation of each wall's received power in a step is reported with 8 rays and with 1.

#### Default

Off until (a) to (g) pass and item 10 is made. Then on with item 10, with `RUN_CACHE_VERSION` raised. Made after 13.B began, it gets its own full rerun on 13.B's base, reported beside 13.B and beside item 10's own rerun, so that each item's effect is read alone (13.12, rule 5; 13.19, decision 11).

#### What ends this line

If any check of (a) to (g) fails in any case, the attempt fails and is reported (13.12, rule 3). No tolerance moves, and T_a + 10 K is not tuned. Surfaces keep the 500 K floor, and 13.19's decision 7 stands as a limit stated in words. A failure of (c), (d) or (e) on the GPU alone, with the CPU passing, is a fault of the joule path; a failure of (d)3 means the code does not do what the declared model says. No second attempt is drafted. It would need the user's word and a new method committed here first.

#### Declared limits

- No reflection, absorptivity 1. With mixed ε an isothermal enclosure does not balance: a surface's net gain is σ (T⁴ − T_a⁴)(Σ_j ε_j F_ij − ε_i) per unit area, 0.2 σ (T⁴ − T_a⁴) for metal among stone. The exchange between gray surfaces is overstated, as 13.25 states. Gray reflection is a candidate for its own item.
- Porous cells keep the 500 K floor. A pack below 500 K absorbs all that reaches it and re-radiates nothing: at 450 K and ε 0.9 it omits 1.7 kW/m² of fuel area, so its heating before it ignites is overstated by that.
- Surfaces between T_a and T_a + 10 K emit nothing; it is counted (`stats.surfaceFloor`).
- Float32 storage. Where a lumped cell's own change in a step falls below half a float32 step (stone within about 12 to 20 K of ambient at dt 0.1 s, above), the cell's temperature does not fall when it emits, though its rays carry the energy, and a deposit into it is lost. Energy then appears wherever a receiver resolves the deposit (a light wall node, the gas) and vanishes where one does not. It is counted (`stats.radRoundoff`), not corrected. The solver's convection near ambient has the same property now. A compensated sum, a per-cell remainder carried to the next write, would correct it; it is not in this item.
- The GPU's deposits are in joules only with the item on.
- The gas keeps `emitAbove` unless item 8 is made (13.19, point 5).

#### Not verified

- The size of the change in any run, and its real cost.
- Whether Vienna on the web stays usable with the items on (reported in (h)).
- Whether the float32 loss matters in the validation's rooms. Their linings are one-dimensional boards, whose end nodes are light and resolve small changes; lumped cells there are the thick ones.
- That 4096 J per deposit is never reached on the GPU. A ray from the hottest cells the GPU runs carries an estimated 2.3 kJ at most (a lone stone cell at 1500 K, six open faces, 0.34 m, dt 0.1 s); clamps are counted.

#### Sources

As in 13.25, and in this repository 13.25's lines with `flamescapes.js` 407 to 408 and 2114 to 2150 and `flamescapesGpu/kernels.js` 1283 to 1293. Kirchhoff's law for a gray, diffuse surface and the gray exchange between parallel plates are the standard results of Howell et al. (2020) and Modest (2013), quoted, not re-read.

#### The coordinator's decisions on 13.25 and 13.26 (2026-11-13, 09:49, before any run of either)

The two drafts above are the physics agent's, written blind (13.12, rule 2). It opened the plan's headings, 12.4, 12.6, 13.12, 13.19 and 13.24, and no result file. 13.19's decisions and 13.24 told it that the rooms read hot and gave item 8's check figures; that is declared, and nothing in the drafts rests on either. **Both items were conceived after results were read, and that is declared here** (13.24): item 10 came from item 8's check (h), and item 11 from the limit that 13.19's decision 7 stated. The coordinator has read 13.A and 13.B, and knows which way item 11 leans: with the floor lowered, warm walls under a hot layer radiate to the cooler surfaces they face, stay cooler, and take more heat from the gas. That cools the layer, which is the direction the rooms' reading would welcome. So the decisions below add no constant to the physics and move no tolerance, and both items are judged only by checks whose references are closed forms (view factors, symmetry, reciprocity, conservation, bit-identity). No measured datum enters any of them. Each decision takes the draft's own choice or the stricter reading, except decision 5, whose reason is arithmetic done before any run.

**Checked by the coordinator,** apart from the agent, in the solver at d028c0a (its solver files equal 550fce5's):
- The CPU casts a surface's rays from the cell's centre (`flamescapes.js` 888), and `march` leaves a ray whole in the first non-gas cell it enters (951 to 963).
- The GPU does the same, which 13.24 left unchecked: `march` sets tMax = 0.5/|d| on each axis (`kernels.js` 1260) and `radiateCell` calls it from the emitting cell (1380).
- The GPU's deposits are fixed point in kelvin, 16 384 steps a kelvin (357, 1190 to 1197). The CPU's wait in float64 arrays (809 to 812). T and T_s are float32 (407, 408).
- The GPU draws from salts 1, 2 and 5 only (1042, 1071, 1354) and the CPU's stage from salt 2 (860), so salt 3 is free.
- Five unit-test files pass `emitAbove: 1e9`.
- F₁₁ and F₁₄, evaluated by the coordinator from the formulas as written above, give every figure of (a)'s table, references and tolerances; the five shares of each box sum to 1 within 10⁻⁸; and 32.97 cell² both ways in (a)3. The cube's two values are the known 0.1998 and 0.2000.
- Hand calculations repeated: the centre's view factor, 0.5541 (13.24); 54 W/m² at T_a + 10 K and ε 0.9, a ray of 0.027 J on 0.20 m cells at dt 0.1 s, 0.026 of a kelvin step in a stone cell; the float32 step of 3.05 × 10⁻⁵ K; (b)'s 0.0059 and (d)1's 0.0042; 13.26 (c)'s 14.16 W/m², 1.7 % and 5.6 %.

Not checked: the agent's model figures (the shares of 0.443 to 0.447, the nets at edges and corners, the crossing counts), which no check depends on; the sources' chapters.

1. **Item 10's method stands as drafted,** points 1 to 12, read with decision 5. The start comes from salt 3; each ray's direction and face are the draws of the item off.
2. **Porous cells are unchanged in both items,** as drafted, and keep the 500 K floor under item 11. The limit is declared: a pack below 500 K takes all that reaches it and returns none. Bringing packs in is another item.
3. **The burner's rays are outside both items.** What the draft read is recorded as a finding in the solver as it stands: the validation burner's radiant part leaves one point half a cell above its top, in 16 fixed directions of which 8 point below the horizontal. Every run of 13.A and 13.B carries it, and no word of either changes for it. It is a candidate for its own item, not drafted now.
4. **The reach.** `rayReach` does not stop a surface's rays (point 8), as drafted. (a)'s 40 × 8 × 8 box tests it.
5. **The GPU's deposits are in joules whenever item 10 is on,** not with item 11 alone. The agent offered this choice (its note 5). The reason is arithmetic from the draft's own figure, done before any run. In (g)'s scenes on 0.20 m cells every ray of a flat wall carries the same 14.07 J into stone cells of one capacity. That is 13.34 kelvin steps, and each rounds to 13. The rounding is then a bias of -2.6 % on every deposit and not noise, so correct code could not meet (g)'s 1 %. In joules the same deposit is 57 636 steps. So 13.26's point 6 belongs to item 10: with `surfaceRayStart` 1 every radiation deposit on the GPU is accumulated in joules at 4096 steps a joule, clamped at ±4096 J a deposit, the clamps counted in number and in J, and `radApply` divides by the capacity the CPU divides by. With `surfaceRayStart` 0 the deposits stay in kelvin, bit for bit. 13.25's point 11 (its last line) and (g)'s reason are read with this, and (g)'s tolerances do not move. The same arithmetic on 13.26's GPU checks: the smallest deposit is (b)'s ray at T_a + 10.01 K on 0.20 m, 61.6 joule steps, which rounds to 62, 0.6 % high and inside (e)'s 1 %.
6. **The floor's rule stands** (13.26, point 1). T_a + 10 K is 13.19 point 5's numerical cut, fixed from no data, and it is not tuned. One addition, for parity: with item 11 on, θ is rounded to float32 on both paths, so that the CPU and the GPU compare a float32 temperature with the same number. With the item off the comparison is the present one.
7. **Float32 storage is counted, not corrected,** as drafted. `stats.radRoundoff` is reported by (i) and beside `radiated` in every validation run made with item 11 on. A compensated sum is another item.
8. **No reflection.** A surface takes all of each ray and emits at its ε, as now. Gray reflection is a candidate for its own item, and (d)3 measures the limit.
9. **`surfaceWarmRays` is taken,** built inert (default 8), with check (k). (k) is outside what ends the line. If (k) fails, the option is withdrawn (held at 8), that is reported, and item 11 stands or falls on (a) to (g). The reason for taking it is the user's rule on cost; it changes no expected deposit.
10. **A one-dimensional wall's side face starts on the whole cell face,** as drafted, a declared limit.
11. **The tallies' names are the drafts'.**
12. **The tolerances stand,** 4 standard deviations of the ray counts, with the seed fixed at 1. A case that falls outside fails like any other. It is not run again with another seed.
13. **A fault of the code against the method** (13.21's rule). Every check of 13.25 (a) to (h) and of 13.26 (a) to (f) has a reference that the method meets exactly, in expectation or bit for bit. A failure there is a fault of the code. Corrected code, run again in full, is the same attempt, reported with both commits. The method, the seed, the ray counts and the tolerances do not change, and a change to the method ends the attempt. For the unit suite ((i); 13.26 (g)) the rule is 13.21's decision 9: a test that fails is not edited, and only a test whose reference is the present ray geometry itself (the centre start, or the 500 K floor for a surface) may be rewritten, disclosed as written after a failure.
14. **The default: stricter than the drafts.** Passing its checks makes an item's switch available to the harness and to the solver's options. It does not change the app's default, which stays 0 for both. A default changes only after all of these: the item's full rerun on three cells, reported beside 13.B; the app's verdict text rewritten from the rerun that the default follows; and the user's agreement, given with the step cost (CPU, GPU, and Vienna in the browser) and `radRoundoff` before them. `RUN_CACHE_VERSION` rises with a default, not before. A cached run's key carries both switches, so a run made under one setting is never reused under another.
15. **The order, and the reruns** (13.12, rule 5; 13.19, decision 11). Item 10 is built and checked first, on its own commit. Item 11 is built only if item 10's attempt passes. Then two reruns: item 10 alone, and items 10 and 11 together (item 11 cannot run alone). Each is the eight series, with the chair series beside them as 15.9 read it, on 0.20, 0.27 and 0.34 m, on 13.B's solver and flags and the harness as 13.16 corrected it. Each is reported beside 13.B, verdict by verdict, before anything else is combined. Their place beside item 9's reruns is set when the builds have reported. 13.B's words stay the app's until the user has seen them.
16. **The base** is `next/calibration-ui` at the commit the items' branch (`flamescapes/surface-rays`) starts from. Its solver's physics is 13.B's (550fce5).
17. **The gas's radiation.** A second attempt at item 8 still needs the user's word (13.23). If one is made, its check (h) runs on item 10's geometry. Nothing of item 8 is in these items.
18. **What ends each line stands as drafted,** read with decision 13.

### 13.27 13.B complete on the three cells, beside 13.A (2026-11-13, 09:13)

13.25 and 13.26 are held for the two surface items in draft (13.24).

Solver 550fce5 (pinned from git), walls in layers, `--leakage`, eight series, 178 runs on each cell size. Files, per cell (`034`, `027`, `020`):
- 13.B default leak, as it first ran: `benchmark/results/2026-11-13-validation-13b-<cell>.json`;
- 13.B sealed, 13.16's correction merged over it: `benchmark/results/2026-11-13-validation-13b-<cell>-sealed.json` (the reruns alone in `-sealed-runs.json`);
- 13.A beside it: 13.14's files.

Each word below is the strictest of 13.B's readings: 12.6 and the protocol's section 4, each unweighted and weighted, every run and the 2026-11-12 rule; default leak and sealed (13.16); eight series and nine (15.8, 15.9). 13.A's word is in brackets where it differs. The figures are the default leak's, eight series, every run, unweighted, unless a line says otherwise.

| Quantity | 0.20 m | 0.27 m | 0.34 m |
| --- | --- | --- | --- |
| Smoke layer depth | calibrated: 6 of 7 series, two laboratories, pooled 1.00 / 0.14 | **indicative** by the sealed reading (6 of 7, 1.05 / 0.23); default leak calibrated, 7 of 7, three laboratories, 1.05 / 0.12 (13.A: calibrated) | calibrated: 5 of 7, two laboratories, 1.11 / 0.14 |
| Layer temperature | indicative: 2 of 7, one laboratory, 1.35 / 0.29 | not calibrated: 3 of 7, 1.40 / 0.39, weighted 1.37 / 0.65 | not calibrated: 3 of 7, 1.48 / 0.37, weighted 1.40 / 0.53 |
| Ceiling jet | indicative: 2 of 6, 1.37 / 0.33 | not calibrated: 2 of 6, 1.45 / 0.53 | not calibrated: 2 of 6, 1.42 / 0.48, weighted 1.21 / 0.55 |
| Gas temperature at head height | **not calibrated**: 2 of 7, 1.37 / 0.34, weighted 1.54 / 0.42 (13.A: indicative) | not calibrated: 3 of 7, 1.33 / 0.38, weighted 1.53 / 0.51 | not calibrated: 3 of 7, 1.56 / 0.38 |
| Neutral plane | **calibrated**: 3 of 4, two laboratories, 0.90 / 0.12 (13.A: indicative, 2 of 3) | indicative: 2 of 4, two laboratories, 0.89 / 0.17 | **not calibrated** with the ninth series (15.9); eight series indicative, 1 of 4, 0.77 / 0.27, weighted 0.67 / 0.32 (13.A: indicative) |
| Door and vent velocity | not calibrated: 0 of 5, 0.98 / 0.59 | not calibrated: 0 of 5, 1.20 / 0.71 | not calibrated: 0 of 5, 1.76 / 1.19 |
| CO2 | indicative, one laboratory: 1 of 3, 1.18 / 0.20 | indicative, one laboratory: 1 of 3, 1.32 / 0.28 | not calibrated: 1 of 3, 1.49 / 0.32 |
| O2 | not calibrated | not calibrated | not calibrated |
| Smoke obscuration, soot | not calibrated | not calibrated | not calibrated |
| Door mass flow | one series (NIST/NRC, 1.11 / 0.08) | one series (0.98 / 0.08) | one series (0.91 / 0.08) |

Four words differ from 13.A's: layer depth on 0.27 m and head height on 0.20 m fall, the neutral plane rises on 0.20 m and falls on 0.34 m. 13.5 stands: both are reported, and the app states 13.B's words with 13.A's beside them.

**What the series show.**

- **The walls in layers did not cool the hot rooms. Every room reads warmer.** 13.14 named the layered walls as the first candidate for the hot reading in the small lined rooms, and 13.B tests it. In every series on every cell the model's layer rise is higher in 13.B than in 13.A (median of the runs' ratios 1.01 to 1.17). The layer temperature's δ, 13.A → 13.B:

  | Series | 0.20 m | 0.27 m | 0.34 m |
  | --- | --- | --- | --- |
  | NIST/NRC | 0.93 → 1.00 | 0.92 → 0.99 | 0.99 → 1.08 |
  | WTC | 0.92 → 1.02 | 1.02 → 1.06 | 1.10 → 1.11 |
  | Steckler | 1.31 → 1.38 | 1.11 → 1.15 | 1.15 → 1.18 |
  | NBS | 0.70 → 0.81 | 1.08 → 1.16 | 0.87 → 1.03 |
  | ATF | 1.23 → 1.35 | 1.66 → 1.65 | 0.99 → 1.07 |
  | UL FSRI | 1.38 → 1.41 | 1.32 → 1.48 | 1.58 → 1.62 |
  | UL/NIST | 1.91 → 1.98 | 1.80 → 1.85 | 2.07 → 2.16 |

  The large NIST rooms, which read a little cold, moved to the middle of the band. The small lined rooms, which read hot, read hotter. The pooled δ rose from 1.30 to 1.35 on 0.20 m, 1.27 to 1.40 on 0.27 m and 1.43 to 1.48 on 0.34 m. A wall in layers has a surface that warms sooner than a lumped cell's mean, so it takes less heat. That is the more faithful wall, and the hot reading is not the lumped wall's doing.
- **Head height on 0.20 m falls to not calibrated** for the same reason: the weighted pooled δ crossed 1.5 (1.48 to 1.54). Steckler's own figure went from 1.48 / 0.22 to 1.54 / 0.21. NIST/NRC came into band there (0.80 to 0.87) and WTC moved to 0.99.
- **Layer depth holds.** Pooled δ 1.00, 1.05 and 1.11, with a scatter of 0.12 to 0.14 on the default leak. On 0.27 m the sealed reading is indicative with every run and calibrated with the 2026-11-12 rule: Steckler's tests 10, 114 and 514, whose 0.24 m door the grid closes, fill with smoke when the room truly has no leak, and Steckler's scatter with every run goes from 0.12 to 0.66 (1.15 / 0.11 over the 52 runs the rule keeps). 12.6's addendum gives the rule's word to the app. The stricter word is kept here, as 13.16's point 3 has it, because the choice came after the result. It is the one place in 13.B where every run and the rule disagree.
- **The neutral plane is calibrated on 0.20 m,** in all four readings, with every run and with the rule. In band: Steckler 0.94 / 0.07 (52 pairs), NIST/NRC 0.84 / 0.17 (15) and UL FSRI 0.87 / 0.11 (36), three rooms and doors from two laboratories; UL/NIST 0.74 / 0.05 (2) is indicative. Pooled 0.90 / 0.12 over 105 pairs, weighted 0.89 / 0.17. Stated with it:
  - 13.A's file for that cell holds no NIST/NRC door probes (13.14: 2 of 3 series, door mass flow not run). The word changed because a third series is read there, not because the solver changed: Steckler and UL FSRI are within 0.02 of their 13.A figures.
  - The plane's height is calibrated where the door's velocities are not (0 of 5 series). A height at which the flow turns says nothing of how fast it flows.
  - On 0.27 m the word is indicative (2 of 4; NIST/NRC 0.72 / 0.23), and on 0.34 m not calibrated (15.9). The plane is too low wherever it is out of band.
- **Door and vent velocity: no series in band on any cell.** On 0.20 m Steckler reads 0.72 / 0.08, NIST/NRC 1.10 / 0.39, UL FSRI 1.10 / 0.41, WTC 0.51 / 0.67 and UL/NIST 1.65 / 0.67. Door mass flow in NIST/NRC is in band on every cell, and one series carries no word.
- **CO2** is in band in NIST/NRC only (1.09 / 0.08 on 0.20 m); WTC reads 1.37 / 0.27, and FSE has a stopped run on 0.20 m. The three readings of 14.7 and 14.8 agree for it.
- **O2 is not calibrated on every reading of 14.7 and 14.8** (`tools/calibration/species-readings.mjs`, the strictest of its six), on every cell and both leak readings. All pairs pooled read 3.67 / 0.93, 4.01 / 0.99 and 10.25 / 1.42 on 0.20, 0.27 and 0.34 m, which is UL FSRI's analysers reading almost no drop (13.14). With the floor of 0.005: 1.52 / 0.27, 1.48 / 0.21 and 1.92 / 0.41 on the default leak, and 1.31 / 0.41, 1.22 / 0.35 and 1.66 / 0.56 sealed. With each series' own figure: 1.38 / 0.24, 1.43 / 0.21 and 1.57 / 0.31, and 1.08 / 0.34, 1.07 / 0.32 and 1.22 / 0.43 sealed. NIST/NRC alone is in band on 0.20 m (1.19 / 0.08) and 0.27 m (1.16 / 0.08). The sealed closed rooms read nearer in the mean and wider in scatter, and neither is the room as tested (13.16).
- **Smoke obscuration** (NIST/NRC alone) fell from 1.74 to 1.62, 1.76 to 1.63 and 1.77 to 1.64, with a scatter of 0.37 to 0.38: still not calibrated. Soot (FSE, two pairs) is unchanged.

**13.16's open line: the leak departure on 0.20 m.** Of 178 runs, 11 state a leak, 118 state none and leaked nothing, and 49 state none and leaked: NBS 100O (21.75 kg), the 45 UL FSRI closed-door runs (11.46 to 23.38 kg, median 14.87) and Steckler 10, 114 and 514 (9.15 to 10.26 kg). 52 runs were run again: the 49, and the control, Steckler 11, 144 and 544. The control's pairs are identical to the first file's. None of the 49 leaked, none stopped and none failed the health guard. **On 0.20 m no word differs between the two leak readings,** and on 0.34 m none does either. Layer depth on 0.27 m is the only word 13.16 moved.

**The ninth series on 0.20 m,** which 15.9 left open. With `fsri-chairs` merged over each leak reading, every word is the same as with eight but one. Head height reads indicative with nine on the default leak (pooled 1.32 / 0.37 over 1249 pairs, weighted 1.00 / 0.50) and not calibrated with nine sealed (1.33 / 0.38, weighted 1.01 / 0.50). As on the coarser cells, it rises only because a cold series is pooled with hot ones, and the series in band stay at two. Not calibrated stands. The neutral plane is calibrated with nine as with eight (4 of 5 in band, the chairs 0.85 / 0.08; pooled 0.89 / 0.11 over 118 pairs).

**Stops and health.** No run stopped on 0.27 m or 0.34 m, in 13.A or 13.B. On 0.20 m FSE's ISONG3 stopped on the speed check again, at 52 s and 55 m/s (13.A: 76 s, 50.6 m/s), so FSE is out of band there for all it measures. Substeps: median 3 and at most 7 on 0.20 m, 2 and 5 on the coarser cells. On 0.34 m six UL FSRI runs have the burner's base moved by the grid, the same six as in 13.A.

**Cost.** The median over the 178 matched runs of a run's seconds in 13.B over 13.A is 1.14 on 0.20 m, 1.04 on 0.27 m and 1.05 on 0.34 m. By series on 0.20 m: Steckler 0.87, NBS 1.10, NIST/NRC 1.18, WTC 1.02, ATF 1.58, UL FSRI 1.49, UL/NIST 0.70, FSE 0.85. The two campaigns ran with different numbers of jobs beside other work, so this is a rough figure and not a bench. The 13.B files took 5.9 h, 1.4 h and 0.5 h at 8 jobs.

**The heat ledger, read as 13.15 named it.** The walls' share of the heat released over the compared window, the solver's against the share the McCaffrey, Quintiere and Harkleroad correlation implies, for the three series the correlation fits (one room, one open door). Medians; Steckler over the runs the grid represents.

| Series | Cell | Walls' share, solver | MQH | Solver / MQH | Model rise / MQH | MQH rise / measured |
| --- | --- | --- | --- | --- | --- | --- |
| UL/NIST | 0.20 m | 0.37 | 0.75 | 0.52 | 1.80 | 1.08 |
| UL/NIST | 0.27 m | 0.35 | 0.72 | 0.52 | 1.67 | 1.08 |
| UL/NIST | 0.34 m | 0.38 | 0.78 | 0.51 | 1.95 | 1.08 |
| UL FSRI, door open | 0.20 m | 0.47 | 0.72 | 0.70 | 1.59 | 0.97 |
| UL FSRI, door open | 0.27 m | 0.42 | 0.66 | 0.65 | 1.54 | 0.97 |
| UL FSRI, door open | 0.34 m | 0.56 | 0.78 | 0.69 | 1.72 | 0.97 |
| Steckler | 0.20 m | 0.25 | 0.45 | 0.60 | 1.16 | 1.12 |
| Steckler | 0.27 m | 0.18 | 0.26 | 0.72 | 0.99 | 1.12 |
| Steckler | 0.34 m | 0.26 | 0.31 | 0.83 | 1.00 | 1.12 |

The correlation gives the measured rise within 3 to 12 % in the median in all three. In the two rooms that read hottest, the solver's walls take half to seven tenths of the share the correlation implies, and the model's rise is 1.5 to 2 times the correlation's. In Steckler's room the walls matter less (most of the heat leaves by the door), and the model's rise is near the correlation's on the coarser cells. So where the layer reads hot, the walls take too little heat, with the walls in layers as with lumped walls. The ledger's unexplained remainder at the end of a run is 2 to 13 % of the heat released in the median by series (on 0.20 m: UL/NIST 7 %, with one run at 26 %; UL FSRI with the door open 3 %; Steckler 7 %). That is small beside the differences in the two UL rooms, and of the size of Steckler's on 0.27 m. This is the model's own account against a correlation: a diagnosis, in no statistic, and it names no cause. Two things known since bear on the walls' gain and are not judged by it: the gas radiates only by its soot (item 8 is not made, 13.23), and a surface's rays start at its cell's centre (13.24).

**What is not concluded.**
- Nothing here is fitted, and no constant changes for it.
- Neither leak reading is the closed rooms as tested (13.16).
- The words hold on their cells, for the fires and rooms the series cover (12.6). The neutral plane's word on 0.20 m is not a word for 0.27 m or 0.34 m.
- Items made or attempted after 13.B began (6, 8, 9, and the surface items) are not in these figures. Each that is made gets its own full rerun, reported beside 13.B (13.12, rule 5).

The comparison tables behind this section are a reading of committed files and are not committed. `node tools/calibration/validation-suite.mjs --summary` on each file prints its statistics and words, `--merge` with the chairs' file gives the nine-series reading (15.9), and `tools/calibration/species-readings.mjs` gives the O2 and CO2 readings.

### 13.28 Item 6, attempt 4: the result (2026-11-13, 09:14)

**Attempt 4 fails. Item 6 is not made, and extinction stays off** (13.18, "Default" and "What ends this line"). Check (f) fails: with extinction on, the class B pool releases 838.5 J in 10 s, and the test asks for more than 100 kJ. Check (h) fails because (f) does. Every other check passes, test 2 as it stands among them. The closed rooms' oxygen and the closed-door UL FSRI tests stand as 13.B reports them (13.27).

The build is on branch `flamescapes/extinction-attempt4` at d05dda3, from fe93150. It is a record and is not merged. Its default is `extinction: 0`, with `freeBurnTemperature: 600`. It changes `flamescapes.js`, `flamescapesGpu.js` and `flamescapesGpu/kernels.js`, adds `flamescapes.extinction.test.mjs` and `tools/calibration/extinction-report.mjs`, and adds an `extinction` phase to `tools/flamescapes-gpu-check.mjs`.

The coordinator ran the failing case again on that commit (`node tools/calibration/extinction-report.mjs --parts pool`) and got the agent's figures: 0.8 kJ with extinction on, 168.4 kJ with it off, one cell put out in 97 of the 100 steps, its vapour up to 0.667 by mass. The other checks are the agent's and were not repeated, since nothing is merged.

| Check | Limit (13.18) | Measured | Verdict |
|---|---|---|---|
| (a) the box's oxygen at the stop against the cells' mean threshold | within 0.5 % by volume | 12.807 % against 13.039 %: −0.232 % | passes |
| (b) each cell that burned against its own threshold | within 0.1 % by volume | 1 cell, 0.028 % | passes |
| (c) differs from extinction off | must differ | on: stops at 465.5 s, 9.239 MJ; off: 1268.8 s, 25.369 MJ | passes |
| (d) the criterion against FDS's closed form | lives at Y_lim (1 + 10⁻⁴), out at Y_lim (1 − 10⁻⁴); lives above T_fb | 72 cases (3 fuels, 3 compositions, 8 temperatures), all as required | passes |
| (e) the oxygen index at 20 °C | lives at X_OI (1 + 10⁻⁴), out at X_OI (1 − 10⁻⁴) | 6 cases, all as required | passes |
| (f) the class B pool with extinction on | every class above 100 kJ in 10 s | class B: 838.5 J | **fails** |
| (g) off is off | bit for bit against the base | 16 of 16 state hashes equal over four scenes; the GPU check's agreement, design, zones and manyzones phases identical; unit suite 541 of 541 | passes |
| (h) the unit suite with extinction on by default | all pass, unbent | 539 of 541: (f), and a wall-clock test | **fails** |
| Test 2, as it stands (default leaks) | released below 0.8 of the heat asked | 0.3264 with extinction on, 0.4541 off | passes |
| GPU parity, heat released, GPU over CPU | 0.95 to 1.05 at each read | closed room 1.0000 to 1.0139 on, 0.9994 to 1.0000 off; class B pool 0.9999 to 1.0000 | passes |
| Speed (14.6) | on costs at most 10 % a step on the GPU | not measured cleanly | inconclusive |

- **(f) fails as the method said it could.** Model 1 has no term for the fuel, but it takes the oxygen mass fraction over the whole cell's gas, so vapour dilutes it (13.18, the method's item 4: a cell of air and vapour is put out above about 35 % vapour by mass at 20 °C). The pool's own cell holds up to 67 % vapour and is put out in 97 steps of 100. Attempt 3 failed the same test by another route, the vapour counted as a heat sink. Two criteria taken from FDS as FDS writes them put out this solver's pool cell.
- **(d) and (e) can fail.** The agent ran five mutants, each caught: Y_OI with FDS's molar masses, oxygen by volume, the vapour left out of the oxygen fraction, the free-burn cut moved by 2 K, and every fuel on the default values.
- **(h)'s second failure is not the solver's.** "200 heads and 1000 obstructions stay well under 100 ms" (`sprinklerObstructions.test.mjs`) compares wall-clock times. Its code is the same in every copy and it imports no solver code. It failed on the untouched base too (105.0 ms), passed on the build with extinction off (76.6 ms) and failed with it on (106.1 ms), while other jobs ran on the machine. (h) fails on (f) with or without it.
- **Test 2, four ways** (released over asked, 150 s): default leaks 0.3264 on and 0.4541 off; no leaks 0.5646 on and 0.5646 off. With no leaks the two are identical: the room's oxygen runs out only in cells above 600 °C, where model 1 does not act. So under model 1 test 2 passes as it stands, and its setup did not decide this attempt.
- **Speed.** The bench rounds ran beside two validation jobs and another agent's unit suite, at one member and 60 s, where no cell is put out. No median showed a cost above 4.1 % a step, and the noise was larger than the limit. It is not a result.

**Reported, not judged** (13.18):
- The copied box on other cells: on 0.20 m, (a) −0.220 % and (b) 0.015 %, stopping at 478.0 s against 1305.2 s; on 0.34 m, (a) −0.209 % and (b) 0.019 %, 468.4 s against 1282.3 s.
- The app's models over 120 s, one run each, as a direction only: extinction cuts the Duplex fire's heat from 11.0 to 8.5 MJ and leaves the Office's and the Clinic's unchanged.

**Where 13.18 was unclear, and what the agent took:** the GPU phase sets `extinction: 1` itself and reads back the GPU's heat released; the class B case runs with the CPU fallback disabled; (d) ran three compositions (no vapour, 1.5 and 3 times); the boxes on other cells keep their size to the nearest whole cell; the vapour share is read at the start of the chemistry stage; attempt 3's test "off, and in fresh air, burners burn as before" is kept. None changes a verdict.

**Not run:** the whole GPU check, the bench at its defaults or on Vienna, a speed measurement on a quiet machine, and the app's build.

**What follows** (13.18, "What ends this line"):
- An attempt 5 may be drafted: FDS's model 2 on the stoichiometric pair, in which a rich cell's excess vapour is not a diluent. That is the cause (f) names.
- It is not drafted now. Its burner is attempt 3's, which failed test 2 with the default leaks, so it waits on the user's decision on test 2's setup: a stated `leakage: []`, with a second test whose bound is derived for a room that leaks (13.18, "The burner's sealed-room test"). The four shares above go to the user with that proposal.
- If attempt 5 is drafted and fails, item 6 ends.
- Any further attempt needs a speed measurement on a quiet machine.

### 13.29 The app's text states 13.B (2026-11-13, 09:41)

13.27's words are now the app's: the run card, the Simulate panel, Methods, the guided run, the guide and the README. 13.5 stands (13.B's words, with 13.A's beside them), and so does 13.21's decision 12: these words stay the app's until the user has seen item 9's reruns. No result was read again for this section. What was decided in carrying the words over:

1. **One table, a word for each quantity on each cell** (`calibrationStatus.js`, `VALIDATION`). A "calibrated" carries its pooled bias, its scatter and how many series are in band. An "indicative" says "from one laboratory's tests" where every series in band is one laboratory's (13.1). 13.A's word is kept beside 13.B's where the two differ, and Methods and the guide name the four places.
2. **What a run draws is not calibrated on any tested cell.** The run card's first line speaks for the visibility and the gas temperature a run draws. The temperature's word is the weakest of three: head height, the layer and the ceiling jet. Head height is not calibrated on any cell, so the line never says "calibrated" of anything drawn. It names the calibrated quantities the run does not draw: the layer's depth on 0.20 and 0.34 m, and the neutral plane on 0.20 m.
3. **The neutral plane's "calibrated" on 0.20 m is stated with its limit:** it is the height where a door's flow turns, and the velocities through the same doors are not calibrated on that cell.
4. **Layer depth on 0.27 m reads "indicative",** the stricter word (13.16, 13.27). The text says it was in band in all seven series as first run and names the three sealed tests. The user may rule for the 2026-11-12 rule's word.
5. **The tested range is unchanged:** heat release to 3.2 MW and 900 s. One test of the 178 ran longer (UL/NIST's first, 1,215 s), and one test does not move a limit. The walls are named as the tests' own, in layers; a run's walls are stone, and it predicts its own heat release, which is not calibrated.
6. **The app's 0.27 m grid option stays as it is.** Its constant is renamed (`TESTED_GRID_CELL`) and its text now says "the middle of the three cell sizes the held-out tests ran on"; nothing says the layer was calibrated on it. The stored setting's key and value are unchanged, so a saved choice survives. Which cell that option builds is the user's decision: 13.B's best words are on 0.20 m, at about 2.5 times the cells of 0.27 m.
7. **Sprinklers read "cross-validated on four series from three laboratories"** (14.9) in the guide and the README, never "calibrated". The app's wall coefficient is unchanged.

The solver is untouched: `git diff 550fce5 HEAD` over `src/core/flamescapes*.js`, `flamescapesGpu/` and `voxelize.js` is empty, and the worker's change is the constant's name. Gates before the commit: the unit suite (537 of 537), the Pages build, and both browser suites on the built page (0 failed steps, desktop and phone).

### 13.30 Item 9 and 13.22: the checks, and 13.22's rerun fixed before it runs (2026-11-13, 14:36)

Item 9 and 13.22 were built as 13.21 and 13.22 fixed them, in one commit on branch `flamescapes/voxel-snap` (8b0ae91, from cc57f70). The build agent changed nothing after its checks ran: no method, check, tolerance or constant, and no fix. The coordinator ran (a) and (d) again from that commit and got the same verdicts and the same figures.

**Item 9 is not made** (13.21, What ends this line; decision 14). (b) fails under both readings of its items 2 to 4, which list no turns or tilts.
- As written:
  - the axis-aligned door case leaks once: on 0.27 m cells with t 0.27 m, one cell's sample lies in both the leaf's 3 mm side gap and its 10 mm undercut, no line through it has material on both sides, and the cell stays open;
  - step 6's lines give odd crossing counts in 12,033 cases, 709 of them in item 1's closed rooms. All are at turns or tilts of 45°, where a thin element's own line passes through a corner edge. No leak follows, but 13.21 counts a failed parity as a failure of (b);
  - 9 of the 24 building cases join sealed regions through cells more than a cell from any door, window or opening. On the Office with doors closed, rooms of 1,120 to 19,821 cells join the main region. One traced path passes beside a closed leaf 25 mm thick that has no opening element.
- With item 1's turns and tilts applied to items 2 to 4, the stricter reading, gaps leak. Out of 7,360 cases each:

  | Case | Leaks |
  | --- | --- |
  | Wall ending 5 mm under the roof | 4 |
  | Wall ending 5 cm under the roof | 219 |
  | Wall ending 0.49 h under the roof | 2,502 |
  | Partition ending 0.49 h from the wall | 1,037 |
  | Door leaf | 25 |

  Step 4 measures a gap along axis lines. An oblique gap under half a cell can be wider than that along every axis, so the method's claim that elements less than half a cell apart are joined does not hold on oblique geometry. 13.21 names this a failure of the method, not of the code.
- What held: closed rooms (0 of 7,360), junctions at 90°, 30° and 60° (0 of 22,080), the round room and the dome (0 of 1,840), and the four standard clipped building grids.

(c), (d), (g), (i) and (f) fail too:
- **(d)**, in two series. An opening's side is bounded by an element 0.10 m thick, placed in the cell that holds its midpoint, as FDS's THICKEN places it:
  - UL/NIST's ceiling vent: the high side is at 4.05 m against 4.20 on 0.27 m cells; on 0.34 m the low side is at 2.04 against 1.80 and the width is 2.04 against 2.40;
  - FSE's vent: the low side is at 2.67 and 2.76 m against 2.5 on 0.27 and 0.34 m.

  13.21 counts a failure of (d) as a fault of the code. The agent traced it to the pass rule, which was derived for thick elements. No correction was tried, because (b) ends the attempt whatever (d) is. Steckler's hand values pass exactly, the item on and off.
- **(g):** with holes, about 40 % of scenes differ from FDS's rule. FDS thickens an obstruction before it cuts the holes; the item snaps pieces already cut. Without holes, 1 or 2 exact-class scenes a cell differ.
- **(c):** parts 2 to 4 fail. The causes are a clipped slab's sliver at the grid's sides, the check's own sweep along the normal at 45°, and exact ties decided by float32 coordinates. Part 1 passes.
- **(i):** 12 of 81 plate cases fail: a plate that contains a sample plane reads fraction 1. With the item on, the walls' heat capacity falls 6 to 14 %.
- **(f):** 1.27 on the Office whole on 0.27 m, and 1.39 on a scene built in 4 ms, on a machine about 39 % loaded.

(a), (e) and (h) pass: off is off byte for byte (492 series scenes and 28 room and building scenes), the unit suite 536 of 536 at its base, and invariance.

So 13.20's sentence returns. Doors stay as the voxeliser makes them, and door and vent velocity, door mass flow and the neutral plane stay as 13.B finds them. The app's words do not change. A fourth attempt at the doors needs the user's word and a new method committed here first. The branch is kept and is not merged.

**13.22 passes its checks and is used.** (a): 3,120 tree lists, 48 of them holding a pocket. (b): all 21 constructed cases exact. With item 9 off, pockets exist only on 0.27 m, in 56 tests: all 55 of Steckler and NBS 100Z. 47 Steckler trees end at z 2.025 m and NBS 100Z's at 2.295 m; none is mid-tree. There is no pocket on 0.20 or 0.34 m, so 13.22 changes nothing there.

#### The coordinator's decisions on 13.22's rerun (2026-11-13, 14:36, before any run of it)

1. **What runs, and from where.** The harness at 8b0ae91, with `--snap` left at `seal` (by (a), that is the voxeliser at 550fce5 byte for byte) and `--pockets` at `drop`. The flags are `--solver-rev 550fce5 --wall-layers --leakage --cells 0.27`, with the 56 tests named by `--tests`. Nothing else changes.
2. **The leak readings (decision 11).** For 53 of the 56 tests the two readings of 13.16 are one run: they have no sealed region, so no leak applies either way. Their rerun in the corrected harness serves both readings. The other three, Steckler 10, 114 and 514, differ between the readings. The grid seals their door on 0.27 m, so the representation rule leaves them out of every statistic in both readings. That rule reads the mask, not a result. They are rerun in the sealed reading, which is the harness as it stands. In the default-leak reading they keep their first entries, because no statistic can change through them, and no harness flag is added to bring back the old leak. This is a narrower reading of decision 11, declared here.
3. **The control (decision 11).** These are the first three runs of the suite's order on 0.27 m that hold no pocket: NIST/NRC 13, 02 and 08, the costliest runs. None is on 13.16's list, so both readings are one run. Their pairs must be identical to those of 13.B's file on 0.27 m. If any pair differs, nothing is merged, and that is reported.
4. **The merge.** The reruns are merged over 13.B sealed and over 13.B default leak on 0.27 m (`--merge`), into two new files. 13.B's files stay as they ran. The tool's merge keeps an earlier run's quantity when a later run lacks it. So after the merge, each rerun's merged pairs must equal its own pairs. If a pair from the first files survives a quantity that 13.22 removed, it is listed and taken out before any statistic is read.
5. **What is reported (decision 12).** Each quantity's word on 0.27 m in both readings, beside 13.B's, before anything is combined. The app's words change only after the user has seen it (13.5; 13.21, decision 12). If the rerun is merged, 13.22's harness reaches `next/calibration-ui` as its own commit, the harness files only, without item 9's voxeliser.

### 13.31 13.22 over 13.B on 0.27 m: the result (2026-11-13, 15:21)

The rerun ran as 13.30 fixed it: the harness at 8b0ae91 with `--snap seal`, the solver at 550fce5, walls in layers, leakage, on 0.27 m cells.

- **Runs.** 56 runs, none stopped. Each dropped one pocket cell at the top of one tree: the room tree in each of the 55 Steckler tests, and tree 8 in NBS 100Z. None was mid-tree.
- **The control passed.** NIST/NRC 13, 02 and 08 each gave 29 pairs, identical to 13.B's in both files. So the runs were merged.
- **The merge check passed.** In the sealed file, each of the 56 reruns' merged pairs equal its own. In the default-leak file, each of the 53 reruns' merged pairs equal its own. Steckler 10, 114 and 514 keep their first entries in the default-leak file (13.30, decision 2).
- **Files** in `benchmark/results/`, prefix `2026-11-13-validation-13b-027-`:
  - `pockets-runs.json`: the 56 runs;
  - `pockets-runs-open-doors.json`: the 53 runs without the three Steckler tests;
  - `pockets-control.json`: the control;
  - `sealed-pockets.json`: 13.B sealed with 13.22;
  - `pockets.json`: 13.B default leak with 13.22.

13.22 changes three quantities on 0.27 m. The table gives the pooled bias / scatter and the series in band.

| Quantity | 13.B sealed | with 13.22 | 13.B default leak | with 13.22 |
| --- | --- | --- | --- | --- |
| Smoke layer depth | indicative (1.05 / 0.23; 6 of 7) | **calibrated (1.03 / 0.11; 7 of 7)** | calibrated (1.05 / 0.12; 7 of 7) | calibrated (1.03 / 0.12; 7 of 7) |
| Smoke layer temperature | not calibrated (1.39 / 0.41; 3 of 7) | not calibrated (1.41 / 0.40; 3 of 7) | not calibrated (1.40 / 0.39; 3 of 7) | not calibrated (1.40 / 0.39; 3 of 7) |
| Gas temperature at head height | not calibrated (1.32 / 0.38; 3 of 7) | not calibrated (1.35 / 0.355; 2 of 7) | not calibrated (1.33 / 0.38; 3 of 7) | not calibrated (1.36 / 0.35; 2 of 7) |

Every other quantity on 0.27 m is unchanged in both readings: the ceiling jet, O2, CO2, smoke obscuration, door velocity, the neutral plane, door mass flow and soot.

What moved:
- **Steckler's layer depth.** In the sealed reading it was 1.23 / 0.66, out of the band, and is now 0.98 / 0.105, in band. In the default-leak reading it moves from 1.14 / 0.12 to 0.97 / 0.10. The pocket at the top of the room tree had read as a layer reaching the floor.
- **Steckler's layer temperature.** It moves from 1.11 to 1.22 sealed, and from 1.15 to 1.22 with the default leak. It stays in band.
- **Steckler's head-height temperature.** It moves from 1.14 to 1.27 sealed, and from 1.15 to 1.27 with the default leak. That leaves the band, so head height is in band in 2 series of 7 instead of 3. Its word does not change.
- **NBS.** The series moves a little through 100Z: layer depth from 1.07 to 1.10, still in band.

The reading:
- **Layer depth on 0.27 m is calibrated in both readings,** on the same seven series from three laboratories. 13.A called it calibrated too. The stricter reading no longer differs from the 2026-11-12 rule's, so the question 13.16 left to the user, which word the app states, has the same answer either way. The app's words change only after the user has seen this (13.5; 13.21, decision 12). The change is prepared separately for the user: layer depth on 0.27 m reads calibrated, with the figures above, and the layer temperature's figures on 0.27 m are brought up to date. No other word changes.
- **On 0.20 and 0.34 m nothing changes.** No tree holds a pocket there (13.30).
- **13.21 with 13.22 on, decision 12's second step, does not run,** because item 9 is not made.

**13.22's harness on this branch.** The changes of 8b0ae91 to `tools/calibration/lib/fds.mjs`, `lib/head.mjs`, `lib/represented.mjs` and `validation-suite.mjs` apply cleanly here and come in with this section. Item 9's voxeliser does not, so `--snap nearest` stops at the start with a message. With them:
- the unit suite passes, 537 of 537;
- `--summary` on the merged sealed file gives the same words as the merge;
- `--pockets keep` reads the trees as 13.B read them.

13.22's check tool stays on `flamescapes/voxel-snap` with item 9's checks, at 8b0ae91.

### 13.32 Item 10: the build's checks (2026-11-14, 00:49)

Item 10 was built as 13.25 and the coordinator's decisions fixed it, on branch `flamescapes/surface-rays` from 70cf906, in three commits:
- 14997d1: the item, off by default;
- 3135fb8: the suite's `--surface-ray-start`, and the ledger's tallies;
- 124ad7e: the check tool.

There was one attempt and no fault of the code against the method, so no correcting commit exists. Item 11 was not started.

**Item 10 passes its checks (a) to (i).** `tools/surface-rays-check.mjs` judged 199 lines at 124ad7e, and none failed. The coordinator ran it again from the commit, and every check passed again.
- **(a) A flat wall's emission reaches the room.** The hot wall, the gas and the grid's sides take exactly 0 J. Every other wall takes its view factor within 4 standard deviations: in the 12 × 10 × 14 box the opposite wall takes 0.1947, against F₁₁ = 0.1938 ± 0.0033. Reciprocity gives 33.028 and 33.012, against 32.974. With the item off, the hot wall keeps 0.443 to 0.447 of its own emission.
- **(b) Outside corners.** The block takes exactly 0 J, and the walls take 0.1647 to 0.1685, against 0.1667 ± 0.0059. With the item off, the block takes 0.2235.
- **(c) Isothermal enclosure.** The nets are +0.04, −0.07, −0.52, +0.42 and −0.16 %, against bounds of 0.83 to 4.08 %. With the item off they are +2.85, −15.09, −28.52, +5.26 and −18.44 %.
- **(d) Walls one cell thick.** A lumped partition takes 0 J, and its low side 0.4998 against 0.5 ± 0.0042. A one-dimensional partition takes 0 J, and its high side the whole emission within 5.4 × 10⁻¹³.
- **(e) The grid's side.** `radiatedOut` is 0.4993 and the walls take 0.5007, each against 0.5 ± 0.005.
- **(f) Energy.** The worst step's imbalance is 8.9 × 10⁻¹², against 10⁻⁹.
- **(g) CPU against GPU.** Emission agrees within 8.3 × 10⁻⁸ and the groups within 3.2 × 10⁻⁶. (a), (b) and (c) pass on the GPU with the CPU's tolerances, and no deposit was clamped.
- **(h) Off is off.**
  - With `surfaceRayStart` 0 the fingerprints are the base's, d62b581d and 29793d1a; the coordinator ran them again and got the same.
  - The unit suite passes 537 of 537, run again by the coordinator on a quiet machine. On the loaded machine the agent saw one timing test, `sprinklerObstructions`, fail on the base and the branch alike.
  - `tools/flamescapes-gpu-check.mjs` agrees with the base bit for bit in every phase the base repeats. 47 fields differ, apart from timings, all in the GPU `runScene` results of the sprinklers, design and systems phases. The coordinator compared the files leaf by leaf: each of the 47 differs between two runs of the unchanged base as well. The branch adds no difference of its own.
  - With the item on, the fingerprints do not differ, which (h) asks to be reported. In the fingerprint tool's two rooms no cell passes 500 K within 300 steps, so nothing radiates and the item never acts. (h) gives no verdict on that line.
- **(i)** With the default set to 1 for the run, the unit suite passes 537 of 537.
- **(j) Speed, reported and not judged.**
  - The radiation stage alone, every wall at 1000 K on 0.20 m cells: 4.7 to 5.1 ms off and 5.5 to 5.6 ms on in the 12 × 10 × 14 box; 13.7 to 14.8 ms and 17.9 to 18.2 ms in the 25 × 13 × 20 box.
  - Each surface ray crosses 11.79 and 17.98 cells on average, as the model said: 11.8 and 18.0.
  - `tools/backend-bench.mjs`, one run a set on a shared machine, gives ms a step that do not separate the item's cost from noise.

**Departures, as the agent reported them.** None changes an output with the item off.
- With the item off, the code costs a little more than the plan's cost line names: one extra test a ray on the CPU, and a few uniform branches on the GPU.
- `radApply` binds Y, for the gas's capacity in joules.
- The clamp tallies are named `radClamps` and `radClampJ`.
- (f) is judged on steps with the item on.
- The run cache's key carries the switch only when a run sets it off 0. Existing keys and `RUN_CACHE_VERSION` (13) are unchanged.

**Not verified.** The item has not acted in any fire yet. The fingerprint rooms and the bench's design fires never bring a surface past 500 K, and no validation run was made.

**What follows** (decisions 14 and 15). The switch becomes available to the harness and the solver's options, and the app's default stays 0. Item 11 (13.26) may now be built. Then come the two reruns, item 10 alone and items 10 and 11 together, on 13.B's solver and flags, the eight series and the chair series, on three cells, each reported beside 13.B. A default changes only after those reruns, the app's text rewritten from them, and the user's agreement with the step cost and `radRoundoff` before them. The item reaches `next/surface-rays` as one commit, with the conflict in `validation-suite.mjs` resolved by keeping both sides: 13.21 and 13.22's flags, and `--surface-ray-start`. The branch `flamescapes/surface-rays` keeps the three commits.

## 14. The user's decisions on the roadmap (2026-11-12, 22:30, before any run they govern)

The user answered "yes to all" to the five proposals of [fire-calibration-roadmap-2026-11-12.md](fire-calibration-roadmap-2026-11-12.md), section 6, and asked for the most accurate calibration that keeps the app fast. The rules below are written before any run they govern. Sections 12.6, 13.3 and 13.12 stand.

### 14.1 Fitting

No constant is fitted on a series that 12 or 13 holds out. If a constant has to be fitted, it is fitted only on a tuning series: a series new to this work, named as a tuning series in this plan before any of its measured data is compared with the model. A tuning series never counts towards a verdict.

### 14.2 The verdicts on the GPU

- **Runs:** the first and the last test by name of each series, on each cell size, on the GPU with the CPU's grid (the bench's `gpu-same`), with the same scene, reducers and flags as the CPU run it is compared with.
- **Reading:** per quantity and cell, δ over the subset's pairs on the GPU against δ over the same pairs on the CPU.
- **Passes** when every quantity's ratio of the two is within 0.95 to 1.05 and no GPU run stops where its CPU run did not.
- **If it fails,** the app states the verdicts for CPU runs only, and says on a GPU run that they were found on the CPU.
- The CPU pairs are 13.B's (or the latest full rerun's). No verdict is drawn from the GPU pairs themselves.

### 14.3 Sprinkler activation, cross-validated

The walls' convection coefficient was set on the four sprinkler series, so none is held out (claim protocol, section 4). This gives a held-out reading without new data.

- **Runs:** `tools/sprinkler-validation.mjs` as it stands (46 runs, its cells), on the solver of the latest full rerun, with the coefficient at 5, 7, 10, 14 and 20 W/m²K.
- **Folds:** by laboratory (NIST: Vettori flat and sloped; Canterbury: Bittern; UL: UL/NFPRF series I), three folds. By series, four folds, reported beside; Vettori's two series share a room, so the laboratory folds decide.
- **In each fold:** the coefficient is the candidate with the lowest (ln δ)² + σ̃M² over the training folds' pooled first-head pairs (`tools/calibration/sprinkler-stats.mjs`, tabulated or measured fires only, σ̃E taken as 0). Then δ and σ̃M of the held-out fold at that coefficient.
- **"Cross-validated"** when every held-out laboratory is within the calibrated band and none of its runs stopped. The app then says "cross-validated on four series from three laboratories", never "calibrated".
- The coefficient chosen in each fold is reported. The app's value changes only if all three folds choose the same value and it is not 10.

### 14.4 Requests for terms

Drafts for the user to send: UL (the UL/NIJ houses), VTT (the large hall), LLNL (the enclosure), FM Global and Sandia (FM/SNL), the NEA (PRISME). A series is wired only after its terms allow it, and is named here as held-out or tuning before its data are compared.

### 14.5 The fire's own heat release

- **Design fires:** each class's curve against the items of that kind in NIST's Fire Calorimetry Database: a check of the curve's source, not of the solver. Reported apart.
- **Spread:** against UL FSRI 2018's furnished tests, if their heat release was measured. To confirm from the report, without reading results, before its own pre-registration (section 15).

### 14.6 Speed

Each physics item made, and each full rerun, reports its cost beside its accuracy: milliseconds a step and run time on Duplex, Office and Clinic, CPU and GPU (`tools/backend-bench.mjs`), against the build before it. The app's default is the configuration its verdicts were found on. An item that costs more than 10 % a step on the GPU goes behind a switch, and the app states the verdicts found without it when it is off, which means a second full run for that configuration.

### 14.7 A measured change too small to compare (written after 13.A's O2, before 13.B's is read)

13.A showed that UL FSRI's analysers read almost no oxygen drop in the open-door tests, and the ratio of two changes has no meaning when the measured one is within the instrument's uncertainty. Disclosed as prompted by a held-out result. The rule, for every species pair in every series: a pair whose measured change (the drop of O2, the rise of CO2) is below 0.005 by volume fraction is left out and counted as "below the instrument". The 0.005 is a round figure chosen here, not taken from an analyser's certificate; if a series' report states its analyser's uncertainty, twice that figure replaces it for that series, looked up before 13.B's species are read. Every statistic and verdict is reported both ways, as pre-registered and with this rule, and the stricter decides the app's text. So the rule can show what the other pairs say, and can never lift a verdict.

### 14.8 The analysers' stated uncertainty, and how 14.7 uses it (2026-11-12, 23:20, before 13.B's species are read)

Four series compare species: NIST/NRC, WTC, FSE and UL FSRI (O2 only). Each report states a figure. All were read in the reports; I checked the NIST/NRC table and the UL FSRI sentence myself.

| series | gas | stated (expanded, about 95 %) | covers | source |
|---|---|---|---|---|
| NIST/NRC | O2 | 0.01 volume fraction | calibration, drift, sampling, response | NIST SP 1013-1, Table 8-4 |
| NIST/NRC | CO2 | 0.0025 volume fraction | the same | the same |
| WTC | O2 | 5 % of the reading (Table 2-3, "typical"); 0.0007 at 0.15 (Table 6-1) | Table 6-1: the analyser alone (span gas, noise, drift) | NIST NCSTAR 1-5B, volume 1 |
| WTC | CO2 | 2 % of the reading; 0.0008 at 0.04 | the same | the same |
| FSE | O2, CO2 | 12 % of the reading | span gas, equipment, mixing in the sample line, random | NIST TN 1603, Table 2.6 |
| UL FSRI | O2 | 12 %, taken from the FSE study, not derived for its own analysers | as FSE | Willi and others 2023, section 2.3 |

14.7 did not say how a relative figure becomes a volume fraction, nor what happens when a report's figure is below the 0.005 floor, nor which of WTC's two O2 figures holds. Fixed here, before any species pair of 13.B is read:

1. Each species statistic is reported three ways: over all pairs (as pre-registered); with the 0.005 floor; and with the series' own figure. The strictest of the three decides the app's text. So this section, like 14.7, cannot lift a verdict.
2. With the series' own figure, a pair is left out when its measured change is below the larger of 0.005 and twice the stated uncertainty. The floor stays a floor: a report's smaller figure does not lower it. This narrows 14.7's "replaces".
3. A relative figure is taken at the pair's own reading, rebuilt from the measured change and the nominal ambient: the O2 reading is 0.2095 less the drop; the CO2 reading is 0.0004 plus the rise. So with 12 % an O2 drop is left out below 0.0405, and with 5 % below 0.0190. For CO2 a relative figure never exceeds the floor, so the floor holds.
4. WTC's O2 takes the 5 % of Table 2-3. Table 6-1 covers the analyser alone, without sampling, and the other three series' figures include sampling. With Table 6-1 the threshold would be the floor, which the second reading already gives.

The thresholds that follow: NIST/NRC O2 0.020, CO2 0.005; WTC O2 0.0190, CO2 0.005; FSE O2 0.0405, CO2 0.005; UL FSRI O2 0.0405.

Known when this was written, from 13.A: UL FSRI's open-door O2 drops are small, so the third reading will leave most of them out. That is what prompted 14.7. The number of pairs left out is reported with every statistic.

The three readings are computed from each run's stored pairs (a pair keeps the measured change), so 13.B needs no flag for them.

### 14.9 Sprinkler activation: the cross-validation's result (2026-11-13, 08:50)

Run as 14.3 set it, after 13.B in the night's chain: `tools/sprinkler-validation.mjs` as it stands, its 46 runs at each of 5, 7, 10, 14 and 20 W/m²K, 230 runs on 13.B's solver (550fce5). None stopped. The folds were scored by `tools/calibration/sprinkler-crossval.mjs`, committed before the runs (012d287). Files: `benchmark/results/2026-11-13-sprinkler-crossval-runs.json` (the runs) and `benchmark/results/2026-11-13-sprinkler-crossval.json` (the folds).

Leaving one laboratory out, which decides:

| Left out | Coefficient chosen on the others, W/m²K | Tests | δ | σ̃M | Reading |
|---|---|---|---|---|---|
| NIST (Vettori flat and sloped) | 7 | 17 | 1.03 | 0.21 | in the calibrated band |
| Canterbury (Bittern) | 5 | 8 | 1.00 | 0.06 | in the calibrated band |
| UL (UL/NFPRF series I) | 5 | 4 | 0.98 | 0.05 | in the calibrated band |

Leaving one series out, beside:

| Left out | Coefficient chosen on the others, W/m²K | Tests | δ | σ̃M | Reading |
|---|---|---|---|---|---|
| Vettori flat (1998) | 10 | 9 | 1.17 | 0.12 | in the calibrated band |
| Vettori sloped (2003) | 5 | 8 | 0.88 | 0.20 | in the calibrated band |
| Bittern (2004) | 5 | 8 | 1.00 | 0.06 | in the calibrated band |
| UL/NFPRF series I (1998) | 5 | 4 | 0.98 | 0.05 | in the calibrated band |

**The word: cross-validated on four series from three laboratories.** Every laboratory left out is within the calibrated band at a coefficient chosen without it, and no run stopped. It is never "calibrated": the coefficient was first set on these four series, and none is held out from the solver's making.

**The app's value stays 10 W/m²K.** The laboratory folds chose 7, 5 and 5, not one value, so 14.3's condition for a change is not met.

What this reading does and does not carry:
- The score a fold minimises is nearly flat over the candidates: 0.0024 to 0.0050 with NIST left out, 0.034 to 0.060 with Canterbury left out, 0.029 to 0.052 with UL left out. The first head's time depends weakly on the coefficient between 5 and 20 W/m²K, so the folds' choices are weakly determined, and the reading says more about that weak dependence than about any one value.
- It covers the first head's opening time in the tests with a tabulated or measured fire (29 pairs), with the experiments' own scatter taken as 0. UL's fold holds four tests.
- It was run under the present voxeliser. If item 9 (13.21) becomes the app's default, the runs are made again with it (13.21, decision 7).

## 15. The fire's own heat release (2026-11-12, 22:35, before any heat release record is read)

14.5 asked first what UL FSRI's 2018 tests measured. Read for that: the repository's `02_Info/README.md`, `channel_list.csv` and `furniture_mass.csv` (descriptions, channel names and masses; no test record).

### 15.1 What the series holds

- 120 tests in one room (3.7 x 3.7 x 2.4 m, one 0.9 x 2.0 m door, open or shut throughout). 92 are the gas-burner tests of 13.A. The rest burn one upholstered chair each: a "red accent" chair of about 20 kg or an "overstuffed" chair of about 49 kg, at the centre, the corner or the back wall.
- **Heat release:** by oxygen consumption under the laboratory's hood, "during most open door experiments with a fire size larger than 100 kW". Not in the closed-door tests.
- **Also measured and not yet compared:** total and radiant heat flux to the side wall at two heights (and to the wall behind the fire), three pressure taps, and velocity and temperature in the plume. These serve the roadmap's phase 4 (a wall heat-flux gauge; room pressure from a second laboratory).

### 15.2 What follows from it

- **Spread between items cannot be checked on this series:** each test burns one item. The app's spread from the first item to the room's contents stays not calibrated, and no public series found so far measures it with terms that allow use.
- **The design fire's curve can be checked** (15.3).
- **The chairs add a series with a real, sooty fuel** (15.4).
- **The burner's nominal size can be checked** (15.5).

### 15.3 The design fire against measured items

A check of the curve the app takes from the literature, not of the solver. No verdict word is attached; it is reported apart.

- **Measured items:** the open-door chair tests with a heat release record, and the upholstered chairs and sofas of NIST's Fire Calorimetry Database (its terms read first).
- **From each record:** the peak; the energy released; and a growth coefficient, fitted by least squares as α (t − t0)² with t0 free, from the first reading above 25 kW to the first at 90 % of the peak.
- **Against:** the app's class A curve (medium, 0.0117 kW/s², after EN 1991-1-2 for dwellings and offices) and the fast curve its own note gives for upholstered furniture alone (0.0469 kW/s²).
- **Reported:** the measured coefficients' median and range by item kind, and where the two curves fall in it. The app's default changes only on the user's word.

### 15.4 The chair tests as a series

- **Named here as held out:** `fsri-chairs`, the open-door chair tests with a heat release record. Same laboratory as `fsri`, so it adds a series and no laboratory.
- **The fire:** the test's measured heat release, prescribed, at the chair's place and footprint.
- **The fuel:** the app's class A fuel as it stands (`fireClasses.js`): its heat of combustion and yields are what the app would use. Nothing is fitted.
- **Compared:** what 13.A compares in `fsri` (layer depth and temperature, the ceiling jet away from the fire's quadrant, head height, the neutral plane, door velocity, O2 under 14.7), by the same reducers and rule.
- **The closed-door chair tests** have no heat release record and are not run.

### 15.5 The burner's nominal size

For the open-door burner tests above 100 kW with a heat release record: the mean measured heat release over the steady period against the nominal 250 or 500 kW. Reported as a ratio per test. If the median ratio is outside 0.95 to 1.05, `fsri` is also run with the measured heat release prescribed and reported beside; 13.A's and 13.B's verdicts stand on the nominal size as pre-registered, and the stricter of the two decides the app's text.

### 15.6 Results of 15.3 (the UL FSRI part) and 15.5 (2026-11-12, 23:20)

Tool: `tools/calibration/heat-release-checks.mjs`; every number is in `benchmark/results/2026-11-12-heat-release-checks.json`. No model run is involved. I recomputed one burner mean and one chair's energy and raw peak from the raw records by a separate route; they agree.

**The burner's nominal size (15.5).** Of 22 open-door burner tests of 250 or 500 kW, 19 have a heat release record. The steady period is the events file's, from "Ignition" to "Burner Off", with the 1 s readings as recorded. The median of measured over nominal is 1.003 (range 0.817 to 1.139), inside 0.95 to 1.05. So `fsri` is not run again with the measured heat release, and its verdicts stand on the nominal size, as 15.5 says. Three 250 kW tests read 0.82 to 0.84 (Back_3, Center_3, Corner_2) and one 500 kW test reads 1.14 (Center_3); 15.5 tests the median only, so these are recorded and nothing follows from them here.

**The design fire's curve against the chair tests (15.3).** Of 14 open-door furniture tests, 13 have a record. The readings are the suite's 5 s means. Growth coefficients, in kW/s²:

| item | tests | median | range | the app's medium (0.0117) | the app's fast (0.0469) |
|---|---|---|---|---|---|
| RedAccent chair (about 20 kg) | 6 | 0.0222 | 0.0132 to 0.0517 | below the range | inside, near the top |
| Overstuffed (about 49 kg; the series calls it a sofa) | 7 | 0.0830 | 0.0503 to 0.1763 | below the range | below the range |

Peaks: RedAccent median 1,376 kW (1,094 to 1,718); Overstuffed median 2,653 kW (1,814 to 3,192). Energy: RedAccent median 296 MJ; Overstuffed median 835 MJ, a lower bound, because every item was put out while still burning.

What bends the method, for the reader to weigh: several Overstuffed records have an early bump of 50 to 90 kW before the main rise, and the free start time skips it; in two Overstuffed tests the coefficient nearly halves between the 5 s means and the raw readings (0.083 against 0.137, 0.067 against 0.127), because the window ends on a plateau. The series records no mass loss, so the energy cannot be checked against mass.

**Beside it, not pre-registered** (computed from the table's own coefficients and start times after they were read): the fitted curves start 43 to 236 s after ignition, and the app's design fire grows from the moment of ignition. Counted from ignition, the fitted curves reach 1 MW at a median of about 290 s for the RedAccent chair and about 250 s for the Overstuffed item. The app's medium curve reaches 1 MW at 292 s and its fast curve at 146 s. So the medium class describes these items' time to 1 MW from ignition closely, and understates how steeply they rise once they have caught.

No verdict word attaches to any of this (15.3). The app's default design fire is unchanged and changes only on the user's word.

**NIST's Fire Calorimetry Database (15.3, the terms).** Its data are a NIST public data set (doi:10.18434/mds2-2314), not Standard Reference Data. NIST's licence statement for such data allows use, modified works and their distribution, asks that a modified work say that it was changed, when and how, and asks that NIST be acknowledged as the source. The database's user guide asks that each project's own report be cited, NIST TN 2077 for the method, and the database itself. The sentences were read through a fetch tool that returns fragments; they are to be read on the pages before they are quoted. The upholstered chair and sofa records can be downloaded as 1 s files and carry initial and final mass. They have not been opened. Reading them is the next step of 15.3.

### 15.7 Results of 15.3 (the NIST part) (2026-11-13, 04:06)

Tool: `tools/calibration/heat-release-checks.mjs --nist`; every number is in `benchmark/results/2026-11-13-heat-release-checks-nist.json`. No model run is involved. The method is the UL FSRI part's, through the same functions: 5 s means, the window from the first mean above 25 kW to the first at 90 % of the peak, and α (t − t0)² fitted with t0 free.

**The items.** 31 were chosen from the database's listing by specimen name and description, before any record was opened: every single upholstered seat or sofa, and every full-scale upholstered chair mock-up, burnt alone under the hood. The file lists the search words and what was left out, with the reason. One record (D2) was not served, so 30 were read. Stacks of three chairs and mock-ups with a fire barrier are kept apart, because a stack is several items and a barrier is there to slow the fire.

Growth coefficients, in kW/s²:

| kind | fitted | median | range | the app's medium (0.0117) | the app's fast (0.0469) |
|---|---|---|---|---|---|
| chair | 9 of 10 | 0.0922 | 0.0012 to 0.2432 | inside | inside |
| sofa | 7 of 7 | 0.0546 | 0.0139 to 0.1841 | below the range | inside |
| chair mock-up, no barrier | 3 of 3 | 0.0871 | 0.0732 to 0.1258 | below the range | below the range |
| three stacked chairs (apart) | 3 of 3 | 0.0016 | 0.0006 to 0.0128 | inside | above the range |
| chair mock-up, fire barrier (apart) | 6 of 6 | 0.0006 | 0.00003 to 1.04 | inside | inside |

One chair (Test048) and the cigarette-lit mock-up never passed 25 kW and have no fit. The chairs' range spans a factor of 200, so "inside" says little for them: the kind runs from stackable, conference and office chairs of 7 to 18 kg, two of which barely burnt, to a 27 kg armchair.

Peaks: chairs median 430 kW (17 to 1,693); sofas median 2,988 kW (1,811 to 5,108); mock-ups without a barrier median 2,669 kW (2,506 to 2,823). Energy: chairs median 46 MJ (1.4 to 384); sofas median 891 MJ (634 to 1,438); mock-ups without a barrier median 321 MJ.

**A check of the reduction.** I recomputed one sofa (Test029) from its raw record by a separate route: raw peak 1,983.8 kW at 373 s, 5 s peak 1,943.9 kW at 375 s, 832.1 MJ from ignition to the record's end. All three agree with the file. The test's own page gives a peak of 1,984 ± 102 kW and 811 ± 45 MJ to its Fire Out event.

**Beside it, not pre-registered** (chosen after the table was read, as 15.6 did for the UL FSRI chairs):
- *Time to 1 MW from ignition.* The sofas' fitted curves reach 1 MW at a median of 295 s (244 to 501); the mock-ups without a barrier at 130 s (127 to 141). The app's medium curve reaches it at 292 s and its fast curve at 146 s. So, as in 15.6, the medium class gives a sofa's time to 1 MW from ignition closely and understates how steeply it rises once it has caught. The mock-ups, lit by an 18 kW burner for 80 s, follow the fast curve.
- *Heat of combustion.* NIST weighed each item before and after, so the energy over the mass lost is an effective heat of combustion: chairs median 19.2 MJ/kg (12.8 to 37.0), sofas 18.2 MJ/kg (16.0 to 18.6), mock-ups without a barrier 25.1 MJ/kg (24.6 to 25.6). The app's class A fuel has 18.0 MJ/kg, inside the sofas' range and near both medians.

No verdict word attaches to any of this (15.3). The app's default design fire is unchanged and changes only on the user's word. Taken with 15.6, the evidence for that decision is: two laboratories; 20 sofas and armchairs that burnt freely (13 at UL FSRI, 7 sofas at NIST), with medium below the fitted coefficient of every one; medium close on their time to 1 MW from ignition; and NIST's ten lighter chairs, whose coefficients lie on both sides of medium.

**The terms, read on the pages.** The licence statement's sentences that 15.6 summarised are on NIST's page as quoted in the results file (`terms`): the data are not subject to copyright in the United States; modified works may be made and distributed; a modified work should say that it was changed, when and how; NIST is to be acknowledged as the source. I fetched the licence page and the database's page again on 2026-11-13 and found each quoted sentence in the page's text. The results file carries NIST's notice, the database's citation, NIST TN 2077 for the method, and each project's report. Only reduced numbers are in the repository; the records stay in a cache that git ignores.

### 15.8 `fsri-chairs`: its open choices, fixed before any run of it (2026-11-13, 04:06)

15.4 named the series and left its details open. The agent that prepared it (without running it) listed eight choices and a recommendation for each; they are in `tools/calibration/data/fsri-chairs.json` (`openChoices`). The coordinator has read 13.A and 13.B's first two files, so each decision below takes the agent's recommendation as written, or 15.4's own words. No alternative is run.

1. **The items' size and place:** as recommended. Chair 0.685 × 0.755 m, sofa 2.19 × 0.94 m (read off Willi et al. 2023, Fig. 4), a solid block over each footprint up to 0.445 m. The sofa's seat height is not dimensioned and is taken as the chair's.
2. **The fire's shape:** the suite's burner over the footprint's area from ignition, as the app does. The fire does not spread from the matchbook. A known mismatch, stated with any result.
3. **The prescribed heat release:** the record's 5 s means from ignition, negative means set to 0, a missing mean from its neighbours, no shift for the hood's delay.
4. **The radiant fraction:** 0 along rays, the app's design-fire burner, since 15.4 asks for what the app would use.
5. **The fuel:** the app's class A fuel, from `fireClasses.js` at the solver revision that runs.
6. **The run and the windows:** `fsri`'s, as 15.4 says ("by the same reducers and rule"). The chairs peak and decay, so the last 180 s are the decay. That is known, and the windows are not moved for it.
7. **O2:** under 14.7 and 14.8, with UL FSRI's stated 12 %.
8. **The cells:** 0.20, 0.27 and 0.34 m.
9. **The solver and flags** (not in the agent's list): 13.B's, revision 550fce5 with walls in layers and `--leakage`. The tests have no FDS input and an open door, so no leak is stated and no region is sealed.

How it is reported:
- **As a ninth series, beside 13.B.** 13.B's verdicts are read on its eight series, as pre-registered. They are read again with `fsri-chairs` added. Where the two words differ, the stricter stands, because the series' details were fixed after 13.B's first results were seen.
- **It is the only series whose fire is the app's own design-fire burner** (class A fuel, no test fuel, no radiant fraction along rays). So its result says more about the app as used than a burner series does, and less about any one sub-model.
- **A stopped run counts as 12.6 counts it.**
- Later items (13.17 onward) run it with the other series in their own reruns.

### 15.9 `fsri-chairs`: the result (2026-11-13, 05:35)

**The run.** 39 runs, the 13 tests on each of the three cell sizes, on 13.B's solver (550fce5) with walls in layers and `--leakage`, as 15.8 fixed. No run stopped, and none is "not represented on this grid". Every run released the heat prescribed to it, to the last digit stored (the Overstuffed chairs 524 to 887 MJ with peaks of 1.8 to 3.2 MW, the RedAccent chairs 269 to 327 MJ with peaks of 1.1 to 1.7 MW). D*/δx is 2.9 to 7.6. The file is `benchmark/results/2026-11-13-validation-fsri-chairs.json`.

**The series alone.** δ / σ̃M, unweighted, with the tier of 12.6. One series from one laboratory can carry no verdict word by itself.

| Quantity | n | 0.20 m | 0.27 m | 0.34 m |
| --- | --- | --- | --- | --- |
| Layer depth | 52 | 0.90 / 0.10, in band | 0.95 / 0.08, in band | 1.02 / 0.12, in band |
| Layer temperature | 52 | 0.72 / 0.23, indicative | 0.79 / 0.22, indicative | 0.78 / 0.24, indicative |
| Gas temperature at head height | 92 | 0.68 / 0.19, indicative | 0.73 / 0.19, indicative | 0.77 / 0.24, indicative |
| Ceiling jet | 23 | 0.76 / 0.12, indicative | 0.78 / 0.14, indicative | 0.69 / 0.17, indicative |
| Neutral plane | 13 | 0.85 / 0.08, in band | 0.85 / 0.08, in band | 0.64 / 0.08, not calibrated |
| Door velocity | 91 | 0.74 / 0.50, indicative | 0.90 / 0.55, not calibrated | 1.47 / 1.16, not calibrated |
| O2 (drop), all pairs | 104 | 5.73 / 1.39, not calibrated | 5.17 / 1.34, not calibrated | 29.73 / 1.82, not calibrated |

The 10-bin weighted reading gives the same tiers except two on 0.20 m, both down: head height (0.66 / 0.20, not calibrated) and door velocity (0.57 / 0.34, not calibrated).

O2 under 14.7 and 14.8 (`tools/calibration/species-readings.mjs`): with the floor of 0.005, 79 pairs stay and δ / σ̃M is 1.57 / 0.73, 1.48 / 0.72 and 3.01 / 0.74 on 0.20, 0.27 and 0.34 m. With UL FSRI's own figure, 63 pairs stay: 1.40 / 0.63, 1.32 / 0.61 and 1.89 / 0.43. Not calibrated on every reading. The series has no CO2, smoke or door mass flow.

**What it shows.** With the app's own design fire and its class A fuel, this room reads cold: the layer, the head height and the ceiling jet are 21 to 32 % below the measured rise on all three cells. The burner series in the same laboratory's room (`fsri`) reads hot in 13.B, 1.48 on 0.27 m and 1.63 on 0.34 m for the layer. The layer's depth is in band on every cell, as it is for the burner series. So the two kinds of fire in one room err in opposite directions in temperature and agree in depth. The cause is not examined here. The two differ in fuel, soot yield, radiant fraction, the fire's shape and its history (15.8, items 2 to 6), and the chairs' last window is the decay. No constant is changed for it.

**A consequence written before item 8 has run.** Item 8 (13.19) takes heat from the gas, so it should cool every room. That is the direction the burner series need and the opposite of what this series needs. Item 8's own rerun includes this series (15.8), and its result is to be read on both, not on the pooled figure alone.

**As a ninth series beside 13.B.** Read on 0.34 m and 0.27 m, on both leak readings of 13.16 (the same words came out on each, except layer depth on 0.27 m, which keeps 13.16's difference). The 0.20 m reading follows when 13.B's file for that cell is in. Eight series are 13.B as pre-registered. Nine add `fsri-chairs`. The stricter word stands (15.8).

| Quantity | Cell | Eight series | Nine series | Stands |
| --- | --- | --- | --- | --- |
| Layer temperature | 0.34 m | not calibrated (pooled 1.48 / 0.39; weighted 1.41 / 0.54) | indicative (1.42 / 0.41; weighted 1.10 / 0.44) | not calibrated |
| Layer temperature | 0.27 m | not calibrated (1.39 / 0.41; weighted 1.39 / 0.66) | indicative (1.34 / 0.42; weighted 1.06 / 0.42) | not calibrated |
| Gas temperature at head height | 0.27 m | not calibrated (1.32 / 0.38; weighted 1.52 / 0.52) | indicative (1.27 / 0.40; weighted 1.01 / 0.46) | not calibrated |
| Neutral plane | 0.27 m | indicative (0.89 / 0.17; 2 of 4 series in band) | calibrated (0.88 / 0.17; 3 of 5 in band) | indicative |
| Neutral plane | 0.34 m | indicative (0.77 / 0.27; weighted 0.67 / 0.32) | not calibrated (0.76 / 0.26; weighted 0.66 / 0.32) | **not calibrated** |

Every other word is the same with eight and with nine. The figures in the table are the sealed reading's.

- **Where nine reads better than eight, nothing is claimed.** The temperature words rise only because a cold series is pooled with hot ones: the weighted pooled δ falls from 1.41 to 1.10 on 0.34 m while the number of series in band stays at three (of seven, then of eight). Two errors of opposite sign are not a calibration. The stricter word stands, as 15.8 says, and this is the case that rule was written for.
- **The neutral plane on 0.27 m** would read calibrated with nine (`fsri`, `steckler` and `fsri-chairs` in band). The third series is the same laboratory's room and door as `fsri`. The word stays indicative.
- **The neutral plane on 0.34 m falls to not calibrated,** and that word stands. The weighted pooled δ was at the edge of the indicative range with eight series (0.67) and is below it with nine (0.66). The chairs' own neutral plane on that cell is 0.64. So 13.B's word for the neutral plane is indicative on 0.27 m and not calibrated on 0.34 m.

**What is not concluded.** Nothing about the default design fire (15.6, 15.7): that choice stays the user's. The series prescribes each test's measured heat release, so it tests the room's response to the app's burner and fuel, not the medium curve.

Merged files and the two-way comparison are not committed (they are a reading of committed files): `node tools/calibration/validation-suite.mjs --solver-rev 550fce5 --merge <13.B file>,<the chairs' runs on that cell>` reproduces them.
