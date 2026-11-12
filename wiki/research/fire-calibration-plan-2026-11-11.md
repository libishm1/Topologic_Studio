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
