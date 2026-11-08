# Calibrating the Flamescapes fire against measured data: plan (2026-11-08)

Status: adopted on 2026-11-08 ("Go through with the calibration recommendations"; section 11 records the decisions); Stage 0 under way. The user asked "How do we calibrate the Flamescapes model against measured data?" and then "Write up the calibration plan", had the five open questions researched, and added that "the colouration of the fire is important to preserve" (P8). The plan builds on Open 11 in `wiki/HANDOFF.md` (grid dependence), on the sprinkler validation (`sprinkler-validation-data-2026-11-07.md`, `sprinkler-activation-fix-2026-11-08.md`) and on the harness that validation built (`tools/sprinkler-validation.mjs`). Each stage ends in a note and a decision for the user before the next begins.

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

### 2.1 How the Flamescapes authors set and checked their parameters

Read in the paper and its supplement (Wrede et al. 2026, sections 5.2, 6.2 and 7; supplement Tables 4 to 6). The authors did not calibrate in the sense of this plan:

- Parameters are set per scene, by hand, within the ranges of the paper's Table 5. The supplement gives one set for each comparison: the pool fire with a convection coefficient of 3.0 on cells of about 0.12 m and steps of 0.008 s; the burning tree with 200, a residual absorption of 3000, cells of about 0.015 m and steps of 0.004 s; the wind-driven structure with 200, 500, cells of about 0.10 m and steps of 0.005 s. Studio's port runs one set for every scene on cells of 0.2 to 0.35 m with steps of 0.1 s, so 2 to 20 times coarser and 12 to 25 times longer.
- Three comparisons with real tests (section 6.2), each qualitative or reported as comparable values: NISTIR 6546's pool fire (the radiant flux above and below NIST's acceptable separation thresholds at the expected distances); a dry Douglas-fir tree from NIST's Fire Calorimetry Database ("a similar trend and comparable values for the heat release rate"; stopped at 65 s before the room's furniture caught); and Test 13 of Hedayati et al.'s wind-driven building-to-building experiments (target-facade fluxes "similar to those reported", on a scaled-down source of the same mass loss rate). No error statistics are given.
- Their limitations (section 7): simplified kinetics and material parameters, "a bias towards graphics rather than engineering-grade physics", "intended for physically informed visual analysis rather than safety-critical or site-specific prediction"; "improved material calibration against empirical fire data" is named as future work. This plan is that work, for Studio's port.
- Flame-front particles are "an ignition proxy on the grid ... to accelerate ignition where particle temperature exceeds the local cell temperature" (section 5.1): in Flamescapes the flame front is the pilot (section 7 of this plan).
- The flames are rendered by a path tracer, their appearance "based on the planckian blackbody radiation and the chemiluminescence scheme from Wrede et al. [2025]" (section 5.2); top-down temperature maps use the inferno colour map. Studio draws flame-front particles coloured by temperature from dull red at 800 K to yellow-white at 1800 K (`ViewerManager.js glowColor`), a simplified blackbody ramp (P8).

## 3. Principles

P1. Calibrate sub-models in order: transport first, with the heat release prescribed from the test; then one item burning alone; then spread between items in rooms. Otherwise one sub-model's error tunes another's parameters. The Dalmarnock round-robin (Rein et al. 2009) showed how widely a priori predictions of a furnished room fire can scatter, so the stages must be kept apart.

P2. Calibrate the model as used: the app's cell sizes, its time step, and both backends (CPU and WebGPU), not a refined research configuration.

P3. Few parameters, each with a range from the literature or the paper, fitted on training tests and judged on tests held out (leave one test out where there are few).

P4. Ensembles. Every test is run over several contents layouts (seeds) and the distribution is compared, not one run.

P5. The statistics of ASTM E1355 as applied in the FDS Validation Guide (section 8), so that Studio's numbers can be read beside FDS's.

P6. Every changed parameter is a departure from the paper, documented where the value is set, in the Methods page and in this note's follow-ups, with `RUN_CACHE_VERSION` bumped.

P7. Data rights are checked dataset by dataset. NIST's reports and data are US Government works; other laboratories' data carry their own terms. Only digitised measurements go into `benchmark/validation/`, each with its notice, as for the sprinkler tests; no report text is committed.

P8. The fire's colouring is preserved (the user's requirement). The colour scales stay as they are: flame-front particles from dull red at 800 K to yellow-white at 1800 K, the design fire's flame by McCaffrey's temperatures, the graph's heat colours and the smoke. Calibration adds bands and labels in the legend and the run card; it never recolours the fire or the graph to show uncertainty, and a bias correction applies to the numbers reported, not to the colours drawn. Any parameter change is checked against the look of reference runs: the share of drawn flame points in each colour band of the glow scale, frame by frame, for the Duplex and Office fires, before and after, must stay within a stated margin (proposed: each band's share within 10 percentage points), and a change that dims or washes out the flames is reported to the user before it is adopted.

## 4. Stage 0: grid dependence (before anything else)

A parameter fitted on one grid means nothing on another while the heat release moves by two orders of magnitude between 0.20 and 0.34 m. So the first stage finds the cause.

What to expect. The plume resolution index D*/δx (D* = (Q / (ρ c_p T √g))^(2/5)) of Studio's fires is 1.6 to 6.3 (0.5 to 2 MW on 0.20 to 0.45 m cells), at or below the lowest of FDS's validation studies (1 to 118; FDS Validation Guide 6.11.1, Table 3.56) and of NUREG-1934's "5 to 10" for gross smoke movement. A heat release the solver predicts itself is not expected to converge there: in an ISO 9705 room with a predicted burning rate, FDS's heat release first rose and then fell as cells shrank, and came within 5 % of the grid-independent value only at D*/δx near 105, while temperatures under a prescribed heat release were grid-independent near 10 (Moinuddin and Thomas 2007). So Stage 0 asks two separate questions: does transport converge under a prescribed heat release, and why does the predicted heat release move as it does.

G1, the study. Two scenes first: the 5 x 4 m test room of the unit tests with one item, and the Duplex living room; a room and corridor pair later. Cells of the app's ladder (0.2 x 1.08^k m; section 11, decision 2): 0.20, 0.27 and 0.34 m first, 0.32 m after; on the CPU in Node, the GPU's agreement with the CPU being checked by `tools/flamescapes-gpu-check.mjs`. The first item both as a design fire (transport alone) and burnt by the solver; one contents seed in the first pass, three after. Recorded: the heat release curve, the energy released by 120 s, the gas temperature rise at three trees from 0.3 m to the ceiling (the layer by Janssens and Tran's reduction), visibility at 1.8 m, the hottest node, and the drawn flame points by colour band (P8).

G2, the hypotheses, each to be confirmed or ruled out by G1:

- H1. Solid burning scales with cells, not with exposed fuel area. The exposed area is `contentsArea` per unit floor area of a contents cell and `itemArea` for the item, and the reacting layer is `skinDensity` per exposed area; any of these tied to the cell would move the energy with the grid.
- H2. Gas combustion uses the cell's mean temperature in Arrhenius rates (`gasA`, `gasE`): on coarse cells the vapour may not ignite until it has gathered and then flash.
- H3. The threshold for a solid counting as burning (`burningFlux` per exposed area) interacts with the cell size.
- H4. The 460 kW tail at 0.23 m: a pilot that never ends, or a steady burn of the contents.

G3, the decision (section 11, decisions 1 and 2). Transport under a prescribed heat release is grid-independent enough when each quantity changes between the calibrated cell sizes by less than FDS's own model scatter for it (FDS Validation Guide 6.11.1, Table 16.1): about 10 % for the layer's temperature rise, 14 % for its depth, 13 % for the ceiling jet, 15 to 17 % for O2 and CO2. If it is, transport is calibrated once. A heat release the solver predicts is not required to converge; it is calibrated per cell size and labelled so. In either case each run's cell is rounded up to the next calibrated size of the ladder, which never adds cells and removes the device-dependent grid.

Deliverables: `tools/grid-study.mjs`, `benchmark/results/<date>-grid-study.json` and a section in this note.

## 5. Stage 1: transport, with the heat release prescribed

With the measured heat release given to a burner (`createBurner`, as the sprinkler validation does), only the movement of heat, smoke and gases is tested.

Data, all digitised in NIST's FDS validation repository (`github.com/firemodels/exp`) and reduced in the FDS Validation Guide:

- Steckler's compartment (Steckler, Quintiere and Rinkinen 1982): doorway velocity and temperature profiles over a range of burner sizes and door widths; tests the door flows that carry smoke between rooms.
- NBS multi-room tests (Peacock, Davis and Babrauskas 1991): smoke and heat moving from a burn room along a corridor.
- NIST/NRC compartment tests: layer temperature and depth, ceiling jet, O2 and CO2 at sampling points.
- The sprinkler tests already used (Vettori, UL/NFPRF, Bittern), extended from activation to the ceiling-jet and layer temperatures they recorded. Vettori's are NIST's; UL/NFPRF and Bittern wait on decision 4.
- McCaffrey's plume (1979): centreline temperature and velocity, already used to colour the drawn flame.
- UL FSRI's 2018 compartment tests (`github.com/ulfsri/fsri-compartments-2018`, MIT licence; doi:10.5281/zenodo.5703474): 120 tests in a 3.7 x 3.7 x 2.4 m room with gas burners, a chair or a sofa, with temperatures, O2, heat flux, doorway velocity and measured heat release; the burner tests serve here, the furniture tests in Stages 2 and 3.

Quantities: layer temperature and depth by Janssens and Tran's (1992) reduction, as the FDS guide reduces them; ceiling-jet temperatures; doorway velocities; O2 and CO2; visibility where measured.

Parameters: none at first, as a validation. Only a systematic bias with a physical cause would open one of `wallConvection` (10), `sootAbsorption` (500) or `vorticity` (10), within its literature range.

Deliverables: `tools/transport-validation.mjs` (the sprinkler harness extended), `benchmark/validation/transport-tests.json` with notices, and the bias and scatter of each quantity.

## 6. Stage 2: one item burning alone

Data: NIST's Fire Calorimetry Database (`www.nist.gov/el/fcd`; one CSV a test at 1 s; NIST's open-data terms ask that NIST be acknowledged and changes marked), the heat release of single items under a furniture calorimeter by oxygen consumption: upholstered chairs, loveseats, sofas, mattresses and similar items, chosen to match the fuels Studio's contents stand for; and the chair and sofa tests of UL FSRI's 2018 compartments.

The first check is the paper's own: its dry Douglas-fir tree from the same database (section 2.1), run with the port's parameters and the paper's, so that the port's departures are measured against the comparison the authors published.

Set-up: one item in the open on the app's grids. Quantities: peak heat release, time to peak, energy to the end of the test, and the curve over fixed windows.

Parameters, each within its range: the fuel's surface coefficient `convection` (200; the paper's range is 0 to 200), `skinDensity` (1.0 kg/m²), the pyrolysis and oxidation kinetics (`pyrolysisA`, `pyrolysisE`, `oxidationA`, `oxidationE`; the paper's Tables 3 and 4), `moisture` (0.1) and the exposed areas `contentsArea` and `itemArea`. At most three are fitted at a time, by sampling the ranges (a Latin hypercube) and minimising the summed squared log ratios of the three quantities over the training items; held-out items judge the fit. If runs are too slow to sample, Bayesian calibration with an emulator (Kennedy and O'Hagan 2001) is the fallback.

This stage serves the room's contents, the class B, D, K and L items and the "As the solver burns it" choice; a class A or C first item already follows its design fire.

## 7. Stage 3: spread between items, and room fires

Data:

- Babrauskas (1981), "Will the second item ignite?": the irradiance a burning item gives off with distance and the distance at which a second item ignites for three ignitability levels, against the first item's peak burning rate (his Fig. 13; Tables 2 and 3). His conclusions (p. 28): 40 kW/m² "was not recorded farther than 0.44 m away and 20 kW/m² was not found beyond 0.88 m"; common furnishings "normally require a minimum irradiance approaching 20 kW/m²", so items 1 m or more away "would stand little hazard" short of flashover.
- NIST's Fire Calorimetry Database: furnished-room tests with spread to flashover (a sofa ignited by a pillow, and similar).
- UL FSRI's 2018 compartments: chair and sofa fires in a room.
- Not available now: UL's living rooms with legacy and modern furnishings (Kerber 2012; FSRI's terms allow non-commercial use only and bar sharing adapted material, and no dataset is published) and Dalmarnock Fire Test One (Rein et al. 2009; the Edinburgh repository holds PDFs only, © the University). Either would be used only with written permission.

Quantities: time to second-item ignition, time to flashover (the upper layer at 600 °C), layer temperature in the room, and temperature, CO and visibility in the next rooms.

Parameters: the contents' ignition (each fuel's `ignitionTemperature`, `burningFlux`) and the contents' fuel load and its layout per occupancy. Ensembles over layouts are essential here: the scatter between layouts is as large as many of the effects to be calibrated.

The pilot (section 11, decision 5). The literature treats a second item's ignition as piloted: by flames and hot gases impinging on it close by, by radiation with a pilot present further away (Babrauskas 1981, sections 2 and 4; his second-item ignitability was measured with a pilot). FDS lets fuel vapour burn as soon as it mixes with oxygen unless an extinction model forbids it (`AUTO_IGNITION_TEMPERATURE` defaults to -273.15 °C; FDS User Guide, "Piloted Ignition"). In Flamescapes the flame-front particles are that pilot (section 2.1). Studio's design fire puts its heat and products into its cells directly and spawns no flame-front particles, so since the design fire became the default the contents catch from its heat alone and their vapour must ignite by its own temperature; the ignition temperatures they are given come from piloted tests, so ignition is late, and gathered vapour can flash. The design fire's flame cells (the layers Heskestad's height reaches, already found each step to draw the flame) are to act as a pilot: they spawn flame-front particles as a burning cell does, so vapour they reach ignites. Behind a switch, off by default until Stage 3 has compared it on and off against Babrauskas's distances, the database's sofa tests and FSRI's chair and sofa tests; then on by default if it fits better. Its cost is a few hundred particles a step, nothing without a design fire. Its drawn flame keeps its look (P8): the pilot's particles carry the same McCaffrey temperatures as the drawn flame.

## 8. Statistics and acceptance

For each quantity and test, the ratio of the model's value to the measured one at a defined point (the peak, or the value at a stated time). Across tests, the bias factor and the model's relative standard deviation are computed from the log ratios as the FDS Validation Guide computes them under ASTM E1355, taking out the experiments' own stated uncertainty. Each is reported with the number of tests and the range of room size, fire size and cell size they cover. The sprinkler validation's medians of ratios (section 2) are kept beside them for continuity.

The formulas are the FDS Validation Guide's (section 4.4), on rises above ambient: the summed variances from the scatter of ln(M/E) (its Eq. 4.19), the bias factor δ = exp(mean ln(M/E) + σ̃M²/2 - σ̃E²/2) (Eq. 4.21), σ̃M never below σ̃E, the experimental σ̃E of its Table 4.1 (temperatures 0.07, layer depth 0.05, gases 0.08, smoke 0.19, heat flux 0.11, velocity 0.08, sprinkler activation 0.06), and the band of Eq. 4.23, the true value about N(M/δ, (σ̃M M/δ)²).

For reference, FDS's own results (Validation Guide 6.11.1, Table 16.1, read in the release PDF): the layer temperature with natural ventilation σ̃M 0.10 and δ 1.07, the layer depth 0.14 and 1.09, the ceiling jet 0.13 and 1.08, the plume 0.42 and 1.00, O2 0.17 and 1.00, CO2 0.15 and 1.04, CO 0.51 and 1.01, smoke concentration 0.63 and 0.83, target heat flux 0.50 and 1.11, sprinkler activation 0.16 and 1.04, smoke detector activation 0.21 and 0.62 (0.32 and 1.24 by the temperature-rise method), and a heat release FDS predicts itself 0.28 and 1.01. NUREG-1824 Supplement 1 (2016) sets no pass mark; it uses these numbers as uncertainty bounds.

Acceptance (section 11, decision 3), three tiers, decided per quantity:

- Calibrated: 0.8 ≤ δ ≤ 1.25 and σ̃M ≤ 0.35, from at least three independent datasets. Shown with its band, its number corrected by δ.
- Indicative: 0.67 ≤ δ ≤ 1.5 and σ̃M ≤ 0.5. Shown with its band and labelled indicative.
- Otherwise not calibrated, with its present warning.

The calibrated tier allows about twice FDS's scatter on transport, a fair target for a coarse solver under a prescribed heat release; the indicative tier is about where FDS's weakest outputs lie (CO, smoke, heat flux). Tenability checks use the unfavourable side of the band. All of this is arithmetic on finished runs; it costs nothing while the solver runs.

## 9. Using the results in Studio

- The Methods page and the run card give, for each calibrated output, its bias, scatter and range instead of "not calibrated", and a run outside the range says so.
- Node temperatures and visibility carry their band; the tenability checks can use the band's unfavourable side.
- Parameter values carry their source where they are set; `RUN_CACHE_VERSION` is bumped; the small calibration cases become unit tests and the large ones stay as tools.
- If the bands enter the performance verdicts, the band arithmetic is stated in Lean beside `Performance.lean`.

## 10. Effort and order

Estimated in weeks of work with the user's review between stages, not ideal days: Stage 0 about one week; Stage 1 one to two weeks; Stage 2 two to three; Stage 3 three to six, depending on the data that can be obtained; bringing the results into the app about one week. Stage 0 needs no new data and decides how every later stage is set up, so it comes first. Not in scope: embers, ventilation (Open 12) and the sprinklers, which have their own validation.

## 11. Decisions (2026-11-08)

The five questions were researched (the literature, FDS's validation statistics, the datasets' terms and the deployment's grids) and the recommendations adopted at the user's word ("Go through with the calibration recommendations").

1. Grid tolerance. No guide sets a fixed percentage; NUREG-1934 (section 2.3.5) judges differences between grids "in light of what is expected of the model accuracy". So: transport under a prescribed heat release must change by less than FDS's model scatter for each quantity (section 4, G3); a heat release the solver predicts is not required to converge and is calibrated per cell size. The proposed 20 % is withdrawn.
2. Cell sizes. The app's cell already comes from a ladder of eight steps, 0.2 x 1.08^k m (0.20, 0.216, 0.233, 0.252, 0.272, 0.294, 0.317, 0.343 m), coarsened until the grid fits the budget (`voxelize.js gridFor`; 15,000 or 30,000 cells on the CPU, 60,000 to 400,000 on the GPU by its memory and measured speed; models past 0.35 m are clipped round the fire). On the development machine's GPU the examples ran on 0.27, 0.32 and 0.34 m, and the CPU runs 0.34 m. Calibrate 0.27, 0.32 and 0.34 m and round each run up to the next calibrated step: never more cells than today, so as fast or faster, the same answer on every device, and Vienna still runs on the web. A run between steps loses at most about a quarter of its cells.
3. Acceptance: the three tiers of section 8.
4. Data. Use now, with NIST's or the licence's notice and any change marked: the NIST-origin folders of `firemodels/exp` (Steckler, NBS multi-room, NIST/NRC, McCaffrey, Vettori), NIST's Fire Calorimetry Database and UL FSRI's 2018 compartments (MIT), and Babrauskas (1981). Commit only the reduced numbers each check uses; the tools fetch the raw files (up to about 44 MB) and check their digests. Ask first: VTT's large hall, Bittern (Canterbury) and UL/NFPRF, which reach `firemodels/exp` from other laboratories that NIST's notice does not speak for. `benchmark/validation/sprinkler-tests.json`, already published, holds UL/NFPRF and Bittern values credited to NIST's repositories; the values are measurements, so the risk is low, and a note to NIST's FDS team would settle it (open, for the user). Not available now: Kerber's living rooms and Dalmarnock (section 7).
5. The pilot: yes, behind a switch until Stage 3 (section 7).

The fire's colouring is preserved throughout (P8).

## 12. Sources

DOIs checked on 2026-11-08 at Crossref and resolved at doi.org.

- ASTM E1355, Standard Guide for Evaluating the Predictive Capability of Deterministic Fire Models (cited by designation).
- Babrauskas, V. (1981). Will the second item ignite? NBSIR 81-2271. doi:10.6028/nbs.ir.81-2271 (conclusions read on p. 28 of the scanned report).
- FDS User's Guide, NIST Special Publication 1019, release FDS-6.11.1 (10 July 2026), and its LaTeX source in `github.com/firemodels/fds` (the default `AUTO_IGNITION_TEMPERATURE`). FDS Validation Guide, NIST Special Publication 1018-3, same release: Tables 3.56, 4.1 and 16.1, section 4.4 (read in the release PDF, `github.com/firemodels/fds/releases`). The DOIs printed in the guides resolve to their version 4 editions, so the current ones are cited by release.
- Moinuddin, K. A. M. and Thomas, I. R. (2007). Proceedings of the 16th Australasian Fluid Mechanics Conference (grid study of an ISO 9705 room fire; `people.eng.unimelb.edu.au/imarusic/proceedings/16/Moinuddin.pdf`; no DOI found).
- NUREG-1934 (2012), Nuclear Power Plant Fire Modeling Analysis Guidelines, section 2.3.5; NUREG-1824 Supplement 1 (2016), Verification and Validation of Selected Fire Models for Nuclear Power Plant Applications, Table 6-1 (both read from Internet Archive copies, as nrc.gov refuses scripted downloads).
- UL FSRI (2021). 2018 Compartment Fire Experimental Data. `github.com/ulfsri/fsri-compartments-2018` (MIT licence there; the Zenodo record lists its licence as "other-open", so the GitHub copy and its MIT notice are the ones used), doi:10.5281/zenodo.5703474 (checked at DataCite).
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
