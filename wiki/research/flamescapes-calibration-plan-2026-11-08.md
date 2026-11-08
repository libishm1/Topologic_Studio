# Calibrating the Flamescapes fire against measured data: plan (2026-11-08)

Status: plan; nothing has been run. The user asked "How do we calibrate the Flamescapes model against measured data?" and then "Write up the calibration plan". The plan builds on Open 11 in `wiki/HANDOFF.md` (grid dependence), on the sprinkler validation (`sprinkler-validation-data-2026-11-07.md`, `sprinkler-activation-fix-2026-11-08.md`) and on the harness that validation built (`tools/sprinkler-validation.mjs`). Each stage ends in a note and a decision for the user before the next begins.

## 1. What calibration means here

Studio's Flamescapes solver is a browser port of a solver written for wildland and urban fire in computer graphics (Wrede et al. 2026), run on cells of 0.2 to 0.45 m with 0.1 s steps. Calibration cannot make it resolve flames or flame spread. What it can give is, for each output the app uses, a measured bias and scatter against full-scale tests within a stated range of rooms, fires and cell sizes, together with fitted values for the few sub-grid parameters where the port already departs from the paper. A result inside that range could then be shown with its band; one outside it would keep the present warning. Until a stage has been checked against tests it held out, its outputs are verified against the code's own equations, not validated.

The outputs that matter are the ones Studio acts on:

| Output | Where Studio uses it | Measured in tests as |
| --- | --- | --- |
| Gas temperature at head height at the graph's nodes | heat colouring, "Impassable above", the tenability heat dose | thermocouple trees |
| Hot layer temperature and depth | reading the run; detector and sprinkler context | thermocouple trees, reduced by Janssens and Tran's method |
| Soot density and visibility at 1.8 m | smoke view, the tenability visibility criterion | optical density meters |
| CO, CO2 and O2 at head height | the ISO 13571 asphyxiant dose | gas analysers |
| Ceiling-jet temperature and velocity | sprinkler and smoke detector activation | thermocouples, velocity probes, activation times |
| Heat release rate | the run card, the sprinklers' estimate | oxygen consumption calorimetry |
| Times of second-item ignition, flashover and untenable conditions | not shown yet; derived from the above | event times in the test reports |

## 2. Where it stands

Validated so far: transport under a prescribed burner, for sprinkler activation only. Against NIST's tests (Vettori 1998 and 2003, UL/NFPRF, Bittern) every measured head opens within twice the measured time, and the first head at medians of 0.78 to 1.26 of it, against FDS's 0.73 to 1.13 (`sprinkler-activation-fix-2026-11-08.md`, section 6.1). Vettori's 1998 room remains about a quarter late (idea I4 there).

Not calibrated:

- The solver's own burning. The same Duplex sofa fire peaked at 0.09 MW on 0.20 m cells, 2.5 MW on 0.23 m, 6.1 MW on 0.27 m and 0.75 MW on the CPU's 0.34 m, and the 0.23 m run settled to a constant 460 kW tail (Open 11). On the GPU the grid is sized by the device's speed, so the cell size, and with it the fire, differs between computers.
- The spread between items. The same Duplex fire with its contents laid out three ways released 137 to 272 MJ with no sprinklers (`tools/sprinkler-seeds.mjs`), and pyrolysis gas that does not burn where it is made can gather under the ceiling and ignite at once.
- Layer temperatures, smoke and gases have never been compared with measurements.

Since 2026-11-08 a first item of class A or C is a prescribed t-squared design fire, so its heat release is no longer the solver's. The solver still burns the room's contents, pools, metals and battery packs, and every item when "As the solver burns it" is chosen.

## 3. Principles

P1. Calibrate sub-models in order: transport first, with the heat release prescribed from the test; then one item burning alone; then spread between items in rooms. Otherwise one sub-model's error tunes another's parameters. The Dalmarnock round-robin (Rein et al. 2009) showed how widely a priori predictions of a furnished room fire can scatter, so the stages must be kept apart.

P2. Calibrate the model as used: the app's cell sizes, its time step, and both backends (CPU and WebGPU), not a refined research configuration.

P3. Few parameters, each with a range from the literature or the paper, fitted on training tests and judged on tests held out (leave one test out where there are few).

P4. Ensembles. Every test is run over several contents layouts (seeds) and the distribution is compared, not one run.

P5. The statistics of ASTM E1355 as applied in the FDS Validation Guide (section 8), so that Studio's numbers can be read beside FDS's.

P6. Every changed parameter is a departure from the paper, documented where the value is set, in the Methods page and in this note's follow-ups, with `RUN_CACHE_VERSION` bumped.

P7. Data rights are checked dataset by dataset. NIST's reports and data are US Government works; other laboratories' data carry their own terms. Only digitised measurements go into `benchmark/validation/`, each with its notice, as for the sprinkler tests; no report text is committed.

## 4. Stage 0: grid dependence (before anything else)

A parameter fitted on one grid means nothing on another while the heat release moves by two orders of magnitude between 0.20 and 0.34 m. So the first stage finds the cause.

G1, the study. Three scenes: the 5 x 4 m test room of the unit tests with one item; the Duplex living room; a room and corridor pair. Cells of 0.15, 0.20, 0.25, 0.30, 0.35 and 0.40 m, on the CPU and the GPU, three contents seeds each, with the first item both as a design fire (transport alone) and burnt by the solver. Recorded: the heat release curve, the energy released by 120 s, the layer temperature at three trees, visibility at 1.8 m and the hottest node.

G2, the hypotheses, each to be confirmed or ruled out by G1:

- H1. Solid burning scales with cells, not with exposed fuel area. The exposed area is `contentsArea` per unit floor area of a contents cell and `itemArea` for the item, and the reacting layer is `skinDensity` per exposed area; any of these tied to the cell would move the energy with the grid.
- H2. Gas combustion uses the cell's mean temperature in Arrhenius rates (`gasA`, `gasE`): on coarse cells the vapour may not ignite until it has gathered and then flash.
- H3. The threshold for a solid counting as burning (`burningFlux` per exposed area) interacts with the cell size.
- H4. The 460 kW tail at 0.23 m: a pilot that never ends, or a steady burn of the contents.

G3, the decision. If, once the dominant cause is fixed, the heat release and the layer temperature agree within a tolerance across 0.20 to 0.40 m (proposed: 20 %, for the user to set), Studio is calibrated once. If not, it is calibrated per allowed cell size and the GPU chooses only among those sizes, which also removes the device-dependent grid.

Deliverables: `tools/grid-study.mjs`, `benchmark/results/<date>-grid-study.json` and a section in this note.

## 5. Stage 1: transport, with the heat release prescribed

With the measured heat release given to a burner (`createBurner`, as the sprinkler validation does), only the movement of heat, smoke and gases is tested.

Data, all digitised in NIST's FDS validation repository (`github.com/firemodels/exp`) and reduced in the FDS Validation Guide:

- Steckler's compartment (Steckler, Quintiere and Rinkinen 1982): doorway velocity and temperature profiles over a range of burner sizes and door widths; tests the door flows that carry smoke between rooms.
- NBS multi-room tests (Peacock, Davis and Babrauskas 1991): smoke and heat moving from a burn room along a corridor.
- NIST/NRC compartment tests: layer temperature and depth, ceiling jet, O2 and CO2 at sampling points.
- The sprinkler tests already used (Vettori, UL/NFPRF, Bittern), extended from activation to the ceiling-jet and layer temperatures they recorded.
- McCaffrey's plume (1979): centreline temperature and velocity, already used to colour the drawn flame.

Quantities: layer temperature and depth by Janssens and Tran's (1992) reduction, as the FDS guide reduces them; ceiling-jet temperatures; doorway velocities; O2 and CO2; visibility where measured.

Parameters: none at first, as a validation. Only a systematic bias with a physical cause would open one of `wallConvection` (10), `sootAbsorption` (500) or `vorticity` (10), within its literature range.

Deliverables: `tools/transport-validation.mjs` (the sprinkler harness extended), `benchmark/validation/transport-tests.json` with notices, and the bias and scatter of each quantity.

## 6. Stage 2: one item burning alone

Data: NIST's Fire Calorimetry Database (`www.nist.gov/el/fcd`), the heat release of single items under a furniture calorimeter by oxygen consumption: upholstered chairs, sofas, mattresses and similar items, chosen to match the fuels Studio's contents stand for.

Set-up: one item in the open on the app's grids. Quantities: peak heat release, time to peak, energy to the end of the test, and the curve over fixed windows.

Parameters, each within its range: the fuel's surface coefficient `convection` (200; the paper's range is 0 to 200), `skinDensity` (1.0 kg/m²), the pyrolysis and oxidation kinetics (`pyrolysisA`, `pyrolysisE`, `oxidationA`, `oxidationE`; the paper's Tables 3 and 4), `moisture` (0.1) and the exposed areas `contentsArea` and `itemArea`. At most three are fitted at a time, by sampling the ranges (a Latin hypercube) and minimising the summed squared log ratios of the three quantities over the training items; held-out items judge the fit. If runs are too slow to sample, Bayesian calibration with an emulator (Kennedy and O'Hagan 2001) is the fallback.

This stage serves the room's contents, the class B, D, K and L items and the "As the solver burns it" choice; a class A or C first item already follows its design fire.

## 7. Stage 3: spread between items, and room fires

Data:

- UL's living room experiments with legacy and modern furnishings (Kerber 2012): times to flashover and room temperatures.
- Dalmarnock Fire Test One (Rein et al. 2009): an instrumented, furnished flat, kept as a blind check once the parameters are fixed; its data access terms are to be checked.
- NBS and NIST multi-item room tests, where digitised data exist.

Quantities: time to second-item ignition, time to flashover (the upper layer at 600 °C), layer temperature in the room, and temperature, CO and visibility in the next rooms.

Parameters: the contents' ignition (each fuel's `ignitionTemperature`, `burningFlux`), the contents' fuel load and its layout per occupancy, and whether the design fire's drawn flame should pilot nearby fuel vapour as the solver's own flame-front particles do (today it does not; `wiki/HANDOFF.md`, the design-fire flame). Ensembles over layouts are essential here: the scatter between layouts is as large as many of the effects to be calibrated.

## 8. Statistics and acceptance

For each quantity and test, the ratio of the model's value to the measured one at a defined point (the peak, or the value at a stated time). Across tests, the bias factor and the model's relative standard deviation are computed from the log ratios as the FDS Validation Guide computes them under ASTM E1355, taking out the experiments' own stated uncertainty. Each is reported with the number of tests and the range of room size, fire size and cell size they cover. The sprinkler validation's medians of ratios (section 2) are kept beside them for continuity.

Acceptance is a decision for the user, not a match with FDS. A proposal: a quantity is shown with its band in the app when its bias lies within 0.67 to 1.5 and its relative scatter is at most 0.5; otherwise it keeps its warning.

## 9. Using the results in Studio

- The Methods page and the run card give, for each calibrated output, its bias, scatter and range instead of "not calibrated", and a run outside the range says so.
- Node temperatures and visibility carry their band; the tenability checks can use the band's unfavourable side.
- Parameter values carry their source where they are set; `RUN_CACHE_VERSION` is bumped; the small calibration cases become unit tests and the large ones stay as tools.
- If the bands enter the performance verdicts, the band arithmetic is stated in Lean beside `Performance.lean`.

## 10. Effort and order

Estimated in weeks of work with the user's review between stages, not ideal days: Stage 0 about one week; Stage 1 one to two weeks; Stage 2 two to three; Stage 3 three to six, depending on the data that can be obtained; bringing the results into the app about one week. Stage 0 needs no new data and decides how every later stage is set up, so it comes first. Not in scope: embers, ventilation (Open 12) and the sprinklers, which have their own validation.

## 11. Decisions for the user

1. The tolerance for grid independence in Stage 0 (proposed 20 %).
2. Whether the GPU may choose only among calibrated cell sizes, if Stage 0 does not reach that tolerance.
3. The acceptance thresholds of section 8.
4. Which datasets to use, once their terms are checked.
5. Whether the design fire's flame should pilot nearby fuel vapour, if Stage 3's data say so.

## 12. Sources

DOIs checked on 2026-11-08 at Crossref and resolved at doi.org.

- ASTM E1355, Standard Guide for Evaluating the Predictive Capability of Deterministic Fire Models (cited by designation).
- Janssens, M. and Tran, H. C. (1992). Data reduction of room tests for zone model validation. Journal of Fire Sciences 10(6), 528-555. doi:10.1177/073490419201000604
- Kennedy, M. C. and O'Hagan, A. (2001). Bayesian calibration of computer models. Journal of the Royal Statistical Society B 63(3), 425-464. doi:10.1111/1467-9868.00294
- Kerber, S. (2012). Analysis of changing residential fire dynamics and its implications on firefighter operational timeframes. Fire Technology 48, 865-891 (online 2011). doi:10.1007/s10694-011-0249-2
- McCaffrey, B. J. (1979). Purely buoyant diffusion flames: some experimental results. NBSIR 79-1910. doi:10.6028/NBS.IR.79-1910
- McGrattan, K. et al. Fire Dynamics Simulator Technical Reference Guide, Volume 3: Validation. NIST Special Publication 1018-3 (cited by report number; its experiments' data are in `github.com/firemodels/exp`).
- NIST Fire Calorimetry Database, `www.nist.gov/el/fcd`.
- Peacock, R. D., Davis, S. and Babrauskas, V. (1991). Data for room fire model comparisons. Journal of Research of NIST 96(4), 411. doi:10.6028/jres.096.022
- Rein, G. et al. (2009). Round-robin study of a priori modelling predictions of the Dalmarnock Fire Test One. Fire Safety Journal 44(4), 590-602. doi:10.1016/j.firesaf.2008.12.008
- Steckler, K. D., Quintiere, J. G. and Rinkinen, W. J. (1982). Flow induced by fire in a compartment. NBSIR 82-2520. doi:10.6028/NBS.IR.82-2520
- Wrede, Wagner, Pałubicki, Michels and Pirk (2026). Flamescapes: Simulating the Wildlandfire Urban Interface. ACM Transactions on Graphics 45(6) 246 (project page `helgewrede.github.io/flamescapes`; its DOI was not yet registered when checked).
