# Verification and validation of people leaving (crowd.js), 2026-11-11

The user: "lets take steps to get validation and the calibration to actually inform this". This note sets out the tests that apply to the people-leaving model (`topologicpy-web-frontend/src/core/crowd.js`), the first run of them, what the run found, and what to fix and measure next. Nothing in crowd.js was changed for it.

Harness: `tools/validation/crowd-validation.mjs` (`node tools/validation/crowd-validation.mjs [--quick]`, about 3 s). Results: `benchmark/results/2026-11-11-crowd-validation.json` (crowd.js sha256 `b8f22870...79c57b`, recorded so a rerun after a fix can be compared).

## Sources (checked 2026-11-11)

| Source | What it gives | Access |
| --- | --- | --- |
| RiMEA e.V., *Guideline for Microscopic Evacuation Analysis*, version 4.1.1 of 11.09.2025 (former 4.0.0 of 28.04.2022), https://rimea.de | Annex 1: Tests 1 to 16, geometry, population and expected result; Table 3 stair speeds after Fruin; Fig. 3 speeds by age after Weidmann | Open; CC BY-ND 4.0; German text authoritative, English parallel |
| Ronchi, Kuligowski, Reneke, Peacock and Nilsson (2013), *The Process of Verification and Validation of Building Fire Evacuation Models*, NIST TN 1822, doi:10.6028/NIST.TN.1822 (DOI checked) | Table 1: Verif.1.1 to Verif.5.2 (many after IMO MSC/Circ.1238); Table 2: experimental data for validation; Section 4: behavioural uncertainty (not read in detail here) | Open |
| ISO 20414:2020, *Fire safety engineering: verification and validation protocol for building fire evacuation models* | Per the ISO page: verification of basic, behavioural, fire-people and building-specific components; component and global validation; a reporting template (Annex) | Paywalled; not read. Its tests should be checked against the text before claiming conformity |
| RiMEA Test 16 percentiles, https://rimea.de/wp-content/uploads/2025/01/test_16_percentiles.xlsx; raw data Chraibi and Subaih, SingleFileComparator, Zenodo doi:10.5281/zenodo.8378592 | 10, 50 and 90 % of mean speed against line density, single file, 0.1 to 6.7 /m | Open (Zenodo: "Other (Open)"); kept, converted, in `tools/validation/rimea-test16-percentiles.json` with attribution |
| Seyfried et al. (2009), *New insights into pedestrian flow through bottlenecks*, Transportation Science 43(3):395-406, doi:10.1287/trsc.1090.0263; Kretz, Grünebohm and Schreckenberg (2006), *Experimental study of pedestrian flow through a bottleneck*, J. Stat. Mech. P10014, doi:10.1088/1742-5468/2006/10/P10014 (both checked) | Flow through a bottleneck against its width; RiMEA Fig. 15 draws them with a line of about 1.9 w persons/s | Papers; trajectories in the Jülich archive (below) |

## The model as tested

A floor-field walk on the walkable graph (0.5 m lattice, one person a node, Burstedde et al. 2001), stepping on time steps of `dt` = 0.1 s. Speeds: each person's own, uniform within SFPE's 1.19 m/s +- 25 %, slowed to SFPE's k(1 - a D) with D the persons on the node ahead and its neighbours over their cells' area; stairs at kStair/k of it. A door node passes at most SFPE's specific flow (1.3155 persons/s per metre of effective width) at the one `doorWidth` given. Exit choice "quickest" or "nearest"; start a uniform stagger; optional speed fluctuation and step randomness.

## Results of the first run

| Test | Criterion (source) | Measured | Status |
| --- | --- | --- | --- |
| RiMEA 1 (1.33 m/s, 2 m x 40 m) | 40 m in 26 to 34 s (RiMEA, verbatim) | 40.35 s at 0 deg, 34.37 s at 45 deg; moving 1.22 m/s, realised 0.99 | Fail |
| NIST Verif.2.1 (1 m/s) | 40 s for 40 m (NIST; 5 % used here); two rotations (NIST "User's actions") | 40.78 s at 0 deg; 45.04 s at 45 deg | Pass at 0 deg, fail at 45 deg |
| RiMEA 2 and 3 (stairs, 10 m along the slope) | Test 1's tolerance at the model's stair speed 0.918 m/s | 12.31 s for 10 m against 10.89 expected (+13 %) | Fail (marginal) |
| NIST Verif.2.2 (1 m/s, 100 m on the incline) | 100 s | 123.1 s | Fail |
| RiMEA 13 (stairs up against down) | Speeds down higher than up | 0.893 up, 0.891 down | Fail |
| RiMEA 6, NIST Verif.2.3 (corner) | 20 round the corner, no wall crossed | 100 of 100 out, 0 points outside | Pass |
| RiMEA 5, NIST Verif.1.1 (pre-movement 10 to 100 s) | Each starts at an appropriate time | 11.0 to 99.3 s, KS p 0.79 | Pass |
| RiMEA 7, NIST Verif.2.4 (speed distribution) | Speeds consistent with the distribution assigned | Assigned 0.89 to 1.49 m/s; moving 0.895 to 1.234 (median 1.152); KS p 0 | Fail |
| NIST Verif.5.2 (100 through a 1 m exit) | Flow at most 1.33 persons/s (IMO MSC/Circ.1238) | Mean 0.672, peak 10 s 0.80 (capacity 0.921) | Pass |
| NIST Verif.4.1 (an exit closing at 10 s) | Not used after it closes | 0 steps on to it begun after 10 s | Pass |
| NIST Verif.3.1, RiMEA 10 (exit allocation) | Deterministic models send everyone to the closest exit (NIST) | 115 of 115 with "nearest"; 79 of 91 in rooms where RiMEA assigns them | Pass (assignment by room not representable) |
| RiMEA 9 (1,000 persons, 4 doors then 2) | About twice the time (documentation only, RiMEA footnote 8) | 376 s, 750 s: 1.99 | Documented |
| RiMEA 11 (two exits 5 m apart) | Most prefer the closer exit, congestion there, some use the other | Quickest 50.2 % by exit 1; nearest 59.6 % | Pass (marginal; persons spread over the room, not packed from the left) |
| RiMEA 12a (goal a past the bottleneck) | Time rises with a to a plateau (documentation) | 494, 488, 485, 495, 498 s for a 0.5 to 10 m | Documented (flat) |
| RiMEA 12b (5 m long bottleneck) | Longer than the short one | Short 498 s; long corridor 164 s; long with door nodes 510 s | Fail |
| RiMEA 12d (width 0.8, 1.0, 1.2 m) | Flow rises with width; beside measured flows | 0.294 persons/s at every width (SFPE 0.66 to 1.18; measured about 1.5 to 2.3) | Fail |
| RiMEA 16 (single file, 200 m) | Within the experiments' 10 to 90 % band | 10 of 14 points inside; at 0.27 /m 0.833 m/s below 0.859 | Fail at the ends |
| RiMEA 4 (2D, 10 m corridor) | Documentation; beside SFPE's relation and Weidmann (1993) | 0.75 /m2: 0.93 m/s (Weidmann 1.19); 2.41: 0.56 (0.48); 4.15: 0.15 (0.14) | Documented |

Deviations from the protocols are recorded per test in the results file. The main one: crowd.js places people at random over every walkable node, so tests that ask for a start area (RiMEA 6, 9, 10, 11, 12) are run with people spread over the geometry; readings for room 1 in Test 12 count only those who started there.

## What the run found, with the code and a proposed fix

1. **The mover counts itself in the density ahead** (`densityAt`, crowd.js:476-481, read at 589-590 before `holder[v]` is cleared at 594). Alone, a person meets D = 1 / ((degree + 1) x 0.25 m2): 0.44 persons/m2 in the open, 0.67 by a wall, 1.33 in single file, so speeds are capped at 1.235, 1.151 and 0.903 m/s. This clips the speed distribution (RiMEA 7, KS p 0), slows everyone in single file to 0.90 m/s whatever their own speed (RiMEA 16: the default population and "1.0 m/s for all" give the same numbers), and slows walking by walls. Fix: leave the mover out of the count, and use SFPE's unimpeded speed below its 0.54 persons/m2 threshold (SFPE's own rule), k(1 - a D) above it.
2. **Steps wait for the next time step** (`ready[i] > t + 1e-9`, crowd.js:544; the segment starts at the step time, 592). Each link loses up to `dt`, and more on the diagonal than along an axis: 0.98 m/s against 0.89 m/s at 1 m/s set (NIST 2.1, 0 and 45 deg); 0.99 against 1.16 at 1.33 m/s; 13 % on stairs. Fix: carry each person's own clock (start the next segment at their arrival, or at the time the node ahead was freed if later), or a much smaller step at ten times the cost.
3. **Door flow is set by the crawl into the queue, not by the door** (crowd.js:589-591 against 571 and 593). In a queue the node before a door is surrounded, D reaches 3.4 persons/m2, the speed falls to its 0.15 m/s floor, and the 0.5 m step into it takes 3.33 s: 0.294 persons/s through an inner door of any width (RiMEA 12d; median step 3.33 s, gap between steps on to the door 3.4 s), 0.67 at an exit door (NIST 5.2), against the SFPE capacity of 0.66 to 1.18 and measured flows of about 1.9 w (Seyfried et al. 2009; Kretz et al. 2006). The door's width never binds, and a long narrow corridor (164 s) clears faster than a door (498 s), the opposite of RiMEA 12b. Fix: with 1. done, measure density over a fixed area (a 1 m radius, the mover left out) rather than a node's neighbourhood; let the door's capacity (`doorFree`) govern the flow through it; and test the queue's speed against the Jülich bottleneck trajectories. An option for the flow basis (SFPE's design 1.3 persons/s/m, or the measured 1.9) would show what the design value costs.
4. **One door width for all doors** (`doorWidth`, crowd.js:206, 337): the IFC's widths are not read, though the code check and performance check read them. Fix: per-door widths from the model, the given width as the fallback.
5. **Stairs the same speed up and down** (crowd.js:586-588; RiMEA 13). Fix: a slower ascent, for instance by Fruin's ratios in RiMEA Table 3 (internal stairs, under 30: 0.55 up, 0.76 down m/s); to read in the SFPE chapter which direction its kStair is for before choosing.
6. **One speed distribution** (uniform +- `spread`, crowd.js:281-283). RiMEA 7 and NIST 2.4 ask for an assigned population (RiMEA's standard population: Weidmann's speeds by age, ages 10 to 85 normal about 50); NIST 2.10 for impaired mobility (RiMEA Table 2: 0.46 to 0.76 m/s). Fix: named populations, and a share of people with impaired mobility.
7. **No start areas** (crowd.js:263-283). Fix: an `origins` option (nodes or regions, with counts), which the tests need and a user would too (a full meeting room, an empty store).
8. **Lattice limits**: at most 4 persons/m2 (RiMEA 4 asks up to 6) and 2 /m in single file. Say so in the panel's notes; a finer lattice for crowding studies is the other way.

Not represented, said plainly: counterflow (NIST 2.8: everyone heads out), groups (2.9), smoke slowing people (2.5; gap 4 of the desire-lines plan), incapacitation by dose (2.6; the performance check computes dose on routes, the crowd does not), lifts (2.7: excluded), social influence and affiliation (3.2, 3.3), pre-movement other than a uniform stagger (the performance check models it: log-normal, PD 7974-6), per-person exit assignment (RiMEA 10).

## Calibration, and what validation then means

Calibrating means choosing the free parameters (the density area, the speed floor, the flow basis, the stair ratio) against one set of experiments; validating means then predicting others not used to choose them. NIST TN 1822 Table 2's footnote: data used for development "should not already be used by the model as benchmark". So split the data first:

- Calibrate on: the Jülich bottleneck series (Seyfried et al. 2009; widths 0.8 to 1.2 m) and the single-file corridor (the Zenodo SingleFileComparator data behind RiMEA 16).
- Validate on: Kretz et al. (2006) and Hoogendoorn and Daamen (2005, *Pedestrian behavior at bottlenecks*, Transportation Science 39(2):147-159, doi:10.1287/trsc.1040.0102, checked) for bottlenecks; Zhang, Klingsch, Schadschneider and Seyfried (2011, *Transitions in pedestrian fundamental diagrams of straight corridors and T-junctions*, J. Stat. Mech. P06004, doi:10.1088/1742-5468/2011/06/P06004, checked) for 2D corridors; and the NIST Table 2 sets (Kuligowski and Peacock 2010 stairwell evacuations; Frantzich et al. 2007; the Station nightclub) for whole buildings.
- Report each comparison with a stated metric (NIST TN 1822 and ISO 20414 discuss them; the KS test on speed distributions, as SingleFileComparator does, is one) and the run-to-run spread (several seeds), never a single run.

Data still to confirm: the Jülich Pedestrian Dynamics Data Archive (https://ped.fz-juelich.de/da) terms of use were not found in the search; read them before committing any of its data.

## After the fixes (2026-11-11, same day)

Findings 1 to 5 were fixed in crowd.js (`benchmark/results/2026-11-11-crowd-validation-after.json`):

- 1: the walker is left out of the density, which is now the others within DENSITY_RADIUS (1 m) of the node ahead over the walkable area there; below SFPE's 0.54 persons/m2 a step is at the person's own speed.
- 2: each person keeps their own clock: a step begins when they reached their node, the node ahead was left (and, on a stair, the nodes within the stair gap), and the door opens, whichever is latest, never before a look that found no way on.
- 3: within DOOR_QUEUE_M (2 m) of a door or exit the density slowing a step is taken at most at SFPE's greatest-flow density (1.9 persons/m2), so the door's capacity governs.
- 4: `doorWidths`, each door node's own width (a Float32Array a node, 0 for the default `doorWidth`).
- 5: stairs down at SFPE's kStair, up at STAIR_UP = 0.75 of it (Fruin's ratios via RiMEA Table 3; SFPE's own directional values were not to hand).

| Test | Before | After |
| --- | --- | --- |
| RiMEA 1 (0 / 45 deg) | 40.35 / 34.37 s, fail | 30.08 / 30.08 s, pass |
| NIST 2.1 (0 / 45 deg) | 40.78 / 45.04 s | 40.00 / 40.00 s, pass |
| RiMEA 2 / 3 (stairs) | 12.31 s, fail | up 14.52 s (expected 14.52), down 10.89 (10.89), pass |
| NIST 2.2 | 123.1 s | 100.0 s, pass |
| RiMEA 13 | up 0.893 = down 0.891, fail | up 0.689 < down 0.918, pass |
| RiMEA 7 | 0.895 to 1.234 m/s, KS p 0 | 0.895 to 1.484, KS p 0.28, pass |
| NIST 5.2 | 0.672 persons/s | 0.923 (capacity 0.921), pass |
| RiMEA 12d | 0.294 at every width | 0.658, 0.921, 1.184 (the SFPE capacities), pass; 0.43 to 0.52 of the measured flows |
| RiMEA 12a | 485 to 498 s | 160 to 163 s (flat; documented) |
| RiMEA 9 | 376 / 750 s | 274 / 545 s (1.99; documented) |
| RiMEA 12b | long 164 s < short 498, fail | long corridor 128.8 s < short door 163.3, fail; long with door nodes at its ends 167.4 s |
| RiMEA 16 | 10 of 14 in band (too slow alone) | 8 of 14 (too fast above 1.25 /m: 0.75 m/s against p90 0.66) |
| RiMEA 4 (documented) | 2.4 /m2: 0.56 m/s | 2.4 /m2: 0.77 m/s (SFPE 0.49, Weidmann 0.47) |

Still wrong, and why:

- RiMEA 12b: a narrow corridor carries no capacity, only one person a node, and its two lattice lanes pass about 1.15 persons/s, more than SFPE's specific flow for 1 m (about 0.8). Fix: a specific-flow capacity at narrow passages as at doors, their width from the graph's clearance (skeleton.js clearanceOf).
- RiMEA 16 and RiMEA 4 at higher densities: too fast (single file above 1.25 /m; 2D 60 % above SFPE and Weidmann at 2.4 /m2). The density is measured round the free node a person steps into, which is thinner than the crowd round them (people step into gaps). Before the fixes the crawl hid this. Fix to calibrate: measure the density round the walker's own node, or over a larger radius, and in single file a headway rule (Seyfried et al. 2005); calibrate on the Zenodo single-file data and the Jülich corridors, validate on Zhang et al. (2011).
- The flows are SFPE design values, about half the laboratory flows; an option for the flow basis remains to add.

What changed for runs: every run's timings differ from before the fixes (all five change the walk). Within the new model a run with the randomness off is still exactly reproducible and the randomness still draws from its own stream (the unit test holds), but a run with it on is not the old one with it on: the speed wobble is now evaluated at each step's own start, and a floor-field "stay" moves the person's clock on by a time step. viewer/crowd.js was not changed: segments keep their format, only their start times now fall between time steps. Cost on the NIBS Clinic (33,654 nodes, 708 people): 418 ms for the first run (about 120 ms building the neighbour lists, kept on the graph), 297 ms for the next seed.

To wire in App.jsx: pass `doorWidths` (App's `slimeDoorWidths(graph, semantics)`) to the crowd's `simulateCrowd` options and add it to the crowd's key; and the People leaving hint should say stairs are slower up (0.69 m/s) than down (0.92).

## Next steps

1. Calibrate the density reading (round the walker or a larger radius; single-file headway) against the calibration data above, and add passage capacities; rerun the harness against the after file.
2. Add `origins` (7) and per-door widths (4), then run RiMEA 9 to 12 as specified.
3. Calibrate and validate as above; only then call any part calibrated.
4. Add the harness to the test suite as a check that the passing tests stay passing.
