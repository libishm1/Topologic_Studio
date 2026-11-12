# Fire calibration: what it would take to calibrate the rest (literature study, 2026-11-12)

This note covers the quantities the held-out validation left short of "calibrated": the smoke layer's temperature, the ceiling jet, O2 and CO2, door flow and the neutral plane, smoke visibility, and the 0.20 and 0.34 m cells. For each it gives:
- what the FDS Validation Guide reports;
- which public measured data could be added;
- what physics matters on coarse grids, and what the solver lacks;
- a plan in order of priority, with its effort.

It follows [fire-calibration-plan-2026-11-11.md](fire-calibration-plan-2026-11-11.md) (section 12) and [fire-calibration-claim-protocol-2026-11-12.md](fire-calibration-claim-protocol-2026-11-12.md). Writing it changed no code, ran no solver and moved no verdict. The systematic review in [prisma-fire-model-validation/](prisma-fire-model-validation/PROTOCOL.md) is the wider search; this note is the targeted one.

**Sources.** Every number here comes from one of three places:
- a source read for this note (section 9 says what was read);
- the repository's own results files;
- arithmetic shown in the text.

A source known only through another is cited through it. Every DOI in section 9 was checked on Crossref, or on DataCite for Zenodo records, on 2026-11-12.

## 1. Summary

| Quantity | Status now (plan 12.8) | FDS guide 6.11.1: δ, σ̃M, datasets | What blocks it | First steps | Effort |
| --- | --- | --- | --- | --- | --- |
| Smoke layer temperature | indicative on 0.27 and 0.34 m | natural ventilation 1.07, 0.10, 18 | Corridors: NBS scatters (σ̃M 0.6 to 0.8), and in the newer rerun so does ATF. Single rooms are in band. | An energy audit of the NBS corridor. A held-out single-room set and a held-out multi-room set. A 1-D wall model. | 1 to 2 weeks for the audit and two datasets; 2 to 3 weeks for the wall model |
| Ceiling jet | indicative; 2 datasets from one laboratory | 1.08, 0.13, 22 | No third laboratory. ATF's two stopped runs. The model reads gas temperature where the tests read a bead. | A thermocouple model in the harness. Find why the ATF burner runs over 3,000 K. Add the UL/NIST vent and DelCo ceiling thermocouples. | 1 to 2 weeks, plus runs |
| O2 and CO2 | not calibrated | O2 1.00, 0.17, 13; CO2 1.04, 0.15, 13 | WTC runs 1.6 to 2.2 times measured, outside the indicative band, so no new dataset alone can make these calibrated. Only two datasets. | Read the dry or wet basis from the test reports. Diagnose WTC's analyser point. Add DelCo, and the FSRI 2018 rooms for O2. | 1 to 2 weeks |
| Door flow; neutral plane | peak outflow not calibrated (δ 0.58 to 0.69); neutral plane in band in its one dataset | velocity (all kinds pooled) 1.07, 0.25, 13; no door or neutral-plane statistic | Doors are 2 to 4 cells wide. One dataset. | Add WTC's opening probes (already cached), the UL/NIST vents' first intervals and the FSRI 2018 door probes. Pre-register door mass flow as a quantity. Sub-cell door apertures. | 1 to 2 weeks for data; 1 to 2 weeks for apertures |
| Smoke visibility (soot) | not compared in the stated verdict | obscuration 1.24, 0.26, 2; mass concentration 0.83, 0.63, 1 | The soot yield is an input. No deposition model. One usable dataset. | Add the NIST smoke alarm optical density data and the Börger et al. extinction data. Claim "smoke transport for a given yield", not visibility. | 2 to 3 weeks; a deposition model 1 to 2 weeks more |
| 0.20 and 0.34 m cells | depth indicative on both; on 0.20 m only two datasets run | none for these cells; the guide runs these series at D*/δx 3 to 14 | 0.20 m: compute time. 0.34 m: the layer reads deep in the small rooms (δ 1.28 Steckler, 1.29 NBS; plan 12.8). | Run NIST/NRC, WTC and ATF on 0.20 m overnight. For 0.34 m, a check of numerical thickening of the layer's interface. | 1 day of work, about 8 to 16 h of compute; 0.34 m open |

The order of work across quantities is in section 7.

## 2. Where things stand

### 2.1 The stated verdict

Plan section 12.8 states the verdict, and `topologicpy-web-frontend/src/core/calibrationStatus.js` gives it to the app. It rests on four held-out series: Steckler, NBS multi-room, NIST/NRC and WTC.
- **Calibrated:** the layer's depth on 0.27 m cells (δ 1.05, σ̃M 0.15).
- **Indicative:** the layer's temperature on 0.27 and 0.34 m; the ceiling jet.
- **Not calibrated:** O2 and CO2, and the door's peak outflow.
- **Neutral plane:** in band, but measured by one dataset, so not calibrated.
- **Visibility:** not compared.

### 2.2 The newer rerun (not yet reviewed)

`benchmark/results/2026-11-12-validation-final.json` was written at 04:17 and is untracked. It ran the extension of plan 12.9:
- ATF corridors added;
- NIST/NRC smoke obscuration added;
- each test burner's own fuel and radiant fraction;
- 6 jobs, 5,510 s.

It has not been reviewed into the plan under the claim protocol. Its numbers are given here only where they bear on what is missing. On 0.27 m cells:

| Quantity | Dataset | δ (σ̃M), pairs | Note |
| --- | --- | --- | --- |
| Layer temperature | ATF | 1.63 (1.38), 20 | two runs stopped: 500 kW at 56 s (3,105 K) and the pulsed fire at 545 s (3,033 K), both on the temperature guard |
| Layer temperature | NBS | 1.08 (0.70), 12 | |
| Layer temperature | Steckler, NIST/NRC, WTC | 1.11 (0.14), 0.92 (0.14), 1.03 (0.07) | each in band |
| Ceiling jet | ATF | 1.55 (1.29), 28 | |
| Ceiling jet | NIST/NRC, WTC | 1.03 (0.20), 1.19 (0.07) | |
| O2 drop | NIST/NRC; WTC | 1.17 (0.08); 1.77 (0.18) | |
| CO2 rise | NIST/NRC; WTC | 1.06 (0.08); 1.71 (0.33) | |
| Smoke obscuration | NIST/NRC | 1.76 (0.37), 10 | the first smoke comparison |
| Door peak outflow; neutral plane | Steckler | 0.63 (0.10); 0.98 (0.09) | as before |

In that file the layer's depth on 0.27 m pools to δ 1.08, σ̃M 0.15, in band in four of five datasets. The claim protocol's stricter reading counts a dataset with a stopped run as outside the indicative band, which makes the depth "indicative" there. This note does not settle that; the review under the protocol does.

### 2.3 How coarse the grids are

The plume resolution index is D*/δx. D* = (Q / (ρ c_p T √g))^(2/5) is the characteristic fire diameter and δx the cell. Below:
- the guide's values are its own runs (Table 3.56);
- ours for the four held-out series are plan 12.8's on 0.27 m, scaled to the other cells;
- the rest are computed here from the guide's heat release ranges (Table 3.55), with ρ 1.204 kg/m³, c_p 1.005 kJ/(kg K), T 293.15 K and g 9.81 m/s².

| Series | Q (kW) | FDS guide's runs | 0.20 m | 0.27 m | 0.34 m |
| --- | --- | --- | --- | --- | --- |
| Steckler | 31.6 to 158 | 5 to 9 | 1.2 to 2.3 | 0.9 to 1.7 | 0.7 to 1.3 |
| NBS multi-room | 110 | 4 | 2.0 | 1.5 | 1.2 |
| NIST/NRC | 410 to 2,300 (as run) | 5 to 11 | not run | 2.5 to 5.0 | 2.0 to 3.9 |
| WTC | 1,970 to 3,240 | 9 to 11 | not run | 4.7 to 5.6 | 3.7 to 4.4 |
| ATF corridors | 48 to 485 | 3 to 7 | not run | 1.1 to 2.7 | 0.8 to 2.1 |
| UL/NIST vents | 430 to 2,188 | 7 to 12 | 3.4 to 6.6 | 2.5 to 4.9 | 2.0 to 3.9 |
| Bryant doorway | 34 to 511 | 5 to 14 | 1.2 to 3.7 | 0.9 to 2.7 | 0.7 to 2.2 |
| NIST smoke alarms | 100 to 350 | 3.8 to 6.3 | 1.9 to 3.1 | 1.4 to 2.3 | 1.1 to 1.9 |

The app runs these series three to six times coarser than the guide's own runs. NUREG-1934 says a ratio of 5 to 10 "usually produces favorable results at a moderate computational cost for problems where gross smoke movement is of interest". The app's calibrated range, D*/δx 0.9 to 5.6, lies mostly below that. Nothing read for this note shows FDS-like accuracy near D*/δx 1. The held-out comparisons are the only evidence for this solver there.

## 3. What the FDS Validation Guide reports

The edition read is NIST SP 1018-3, sixth edition, dated July 10, 2026, revision FDS-6.11.1-0-gff928db.

Its title page prints the DOI 10.6028/NIST.SP.1018. Crossref registers that DOI for the 2006 version 4 Technical Reference Guide, not for this volume, and 10.6028/NIST.SP.1018-3 is not registered. The same holds for the User's Guide: 10.6028/NIST.SP.1019 resolves to the 2004 version 4 user's guide. This note cites both guides by number, edition and revision.

### 3.1 The guide's statistics (Table 16.1, the rows that matter here)

| Quantity (guide's section) | Datasets | Points | σ̃E | σ̃M | δ |
| --- | --- | --- | --- | --- | --- |
| HGL temperature, natural ventilation (5.24) | 18 | 330 | 0.07 | 0.10 | 1.07 |
| HGL temperature, forced ventilation (5.24) | 6 | 158 | 0.07 | 0.16 | 1.16 |
| HGL temperature, no ventilation (5.24) | 3 | 32 | 0.07 | 0.09 | 1.09 |
| HGL depth (5.24) | 15 | 278 | 0.05 | 0.14 | 1.09 |
| Ceiling jet temperature (7.1.21) | 22 | 1,061 | 0.07 | 0.13 | 1.08 |
| Plume temperature (6.1.15) | 16 | 480 | 0.07 | 0.42 | 1.00 |
| Oxygen (9.1.10) | 13 | 283 | 0.08 | 0.17 | 1.00 |
| Carbon dioxide (9.1.10) | 13 | 278 | 0.08 | 0.15 | 1.04 |
| Smoke concentration (9.2.5) | 1 | 10 | 0.19 | 0.63 | 0.83 |
| Smoke obscuration (9.2.5) | 2 | 33 | 0.07 | 0.26 | 1.24 |
| Velocity (8.16) | 13 | 554 | 0.08 | 0.25 | 1.07 |
| Aerosol deposition velocity (9.3.1) | 3 | 162 | 0.98 | 0.98 | 0.20 |
| Aerosol deposition (9.3.2) | 2 | 88 | 0.12 | 0.28 | 1.38 |
| Sprinkler activation time (7.2.1) | 6 | 272 | 0.06 | 0.16 | 1.04 |

Experimental uncertainty (Table 4.1), combined σ̃E with its measurement and propagated parts:
- temperatures 0.07 (0.05 and 0.05);
- HGL depth 0.05 (0.05 and 0.00);
- gas concentrations 0.08 (0.02 and 0.08);
- smoke concentration 0.19 (0.14 and 0.13);
- velocity 0.08 (0.07 and 0.03).

### 3.2 The datasets behind each statistic (the guide's summary figures)

- **HGL temperature, natural ventilation:** ATF Corridors, DelCo Trainers, JH/FRA, LLNL Enclosure, NBS Multiroom, NIST Composite Beam, NIST E119 Compartment, NIST FSE 2008, NIST/NRC, NIST/NRC Corner Effects, NIST Vent Study, NRCC Smoke Tower, Steckler Compartment, UL/NIJ Houses, UL/NIST Vents, VTT, WTC, TUS Facade.
- **HGL depth:** ATF Corridors, DelCo Trainers, FM/SNL, JH FRA, VTT, WTC, NIST FSE 2008, NIST/NRC, NIST/NRC Corner Effects, NIST Vent Study, NBS Multiroom, PRISME, Steckler Compartment, UL/NIJ Houses, UL/NIST Vents.
- **Ceiling jet:** Arup Tunnel, ATF Corridors, DelCo Trainers, FAA Cargo Compartments, FM/SNL, NIST Composite Beam, NIST FSE 2008, NIST/NRC, NIST/NRC Corner Effects, NIST RSE 2007, NIST Vent Study, NRCC Smoke Tower, PRISME, SP AST, UL/NFPRF Series I, UL/NIJ Houses, UL/NIST Vents, Vettori Flat Ceiling, Vettori Sloped Ceiling, NIST Smoke Alarms, WTC, NIST RSE 1994.
- **O2:** DelCo Trainers, LLNL Enclosure, NIST FSE 2008, NIST/NRC, NIST Pool Fires, NIST RSE 1994 and 2007, NRCC Smoke Tower, PRISME, SWJTU Tunnels, WTC, UMD Line Burner, Smyth Slot Burner.
- **CO2:** the same list, with FAA Cargo Compartments in place of the UMD Line Burner.
- **Velocity:** Purdue Flames, Sandia Plumes, ATF Corridors, Bryant Doorway, Edinburgh Vegetation Drag, JH FRA, McCaffrey Plume, NIST Pool Fires, PRISME, Restivo, Steckler Compartment, Waterloo Methanol, WTC. It pools plumes, corridors and doors. The guide gives no door-only statistic and none for a neutral plane.
- **Smoke obscuration:** NIST/NRC and FAA Cargo Compartments.
- **Smoke concentration:** FM/FPRF Datacenter.

### 3.3 What the guide says that bears on this project

- **Doors.**
  - FDS ran Steckler's tests on 5 cm cells. The peak outward velocity, compared, "always occurred at the uppermost measurement location". "Relatively minor changes in the velocity boundary conditions at the edges and bottom of the door soffit can have a noticeable impact on these results".
  - Bryant's bi-directional probes read up to 20 % above his PIV in places, so only the PIV is compared.
  - The UL/NIJ houses' door and window velocities are "too noisy to make precise comparisons" and are left out of the summary.
  - In the JH/FRA series, FDS predicted lower velocities than measured at full scale.
- **Smoke.** In the NIST/NRC tests with the door closed, FDS predicts significantly less light transmission than measured. The guide's likely reason is that FDS "has not accounted for the agglomeration and deposition of the smoke on the walls".
- **Species.**
  - WTC had one analyser near the ceiling, "roughly 2 m from the fire".
  - NIST/NRC had two O2 analysers (upper and lower layer) and one CO2 analyser in the upper layer.
  - For LLNL the report does not say whether the gas was sampled wet or dry, and FDS's results are taken as dry.
- **NBS multi-room.**
  - The heat release was 110 kW, with an uncertainty of about 15 %.
  - Only the layer's temperature and depth are compared, "because there was less confidence in the other measurements".
  - The series had 45 tests at 100, 300 and 500 kW; the guide uses three at 100 kW.
- **NIST smoke alarms.** The heat release curves were set by fitting t² ramps in FDS to the fire room's highest thermocouple. So the fire room's temperatures are not independent of the input.
- **The smoke obscuration σ̃E.**
  - The published guide gives 0.07 (Figure 9.51 and Table 16.1).
  - The FDS repository's `Utilities/Python/FDS_validation_scatterplot_inputs.csv` on its master branch gives 12 % (read 2026-11-12). `tools/calibration/lib/stats.mjs` uses 0.12.
  - The harness cites release 6.11.1, so one value must be pre-registered and recorded. On NIST/NRC's smoke the choice moves δ and σ̃M by about 0.01 each (computed from the statistics' formulae). No verdict depends on it.

## 4. Physics on coarse grids, and what the solver lacks

### 4.1 Plume resolution

Section 2.3 has the numbers. The FDS guide states that "the calculations performed for the various validation studies described in this Guide use a wide range of values of the Plume Resolution Index", from 1 to 3 (LLNL) to 20 to 118 (Sandia plumes). It also asks that "both the physical and numerical parameters are comparable to the given application". The app's verdicts must therefore carry the D*/δx range they were found on, as plan 12.8 does.

### 4.2 Openings

- **The rule.** NUREG-1934 advises "at least three cells across any flow path, such as a door or a window, and fire dimension". It asks the same across any partially isolated volume of interest.
- **Our doors.** Steckler's doors (0.24 to 0.99 m) are 0.9 to 3.7 cells wide on 0.27 m cells.
- **Snapping.** The voxeliser snaps a door to whole cells, at 0.54 to 1.11 times its width (plan 12.8). The narrowest doors close outright.
- **The soffit.** The measured peak outflow sits just under the soffit, in the shear layer. The model's soffit is one cell, and the guide finds even 5 cm runs sensitive to the soffit's boundary conditions.
- **The neutral plane.** It follows the pressure balance across the door. It is in band on every cell (plan 12.8).
- **What the solver lacks:** a door that keeps its true area. Face apertures (open area fractions on cell faces) would let a 0.74 m door stay 0.74 m wide on any cell size. The solver has porous cells (`porousVelocity` 0.5) but no aperture on faces.
- **Burners.** The same three-cell rule applies to the fire. ATF's burner is 0.45 m square, under two cells on 0.27 m. That run is the one that stopped on the temperature guard (section 2.2).

### 4.3 Radiation

- **FDS.** FDS prescribes a radiative fraction only in cells where combustion occurs. Its User's Guide gives the reason: "the flame sheet is not well-resolved on a relatively coarse numerical grid", so the T⁴ source "cannot be reliably calculated". The hot layer elsewhere radiates by its own temperature. Its gray absorption coefficient takes in "water vapor, carbon dioxide, carbon monoxide, fuel" and soot. FDS's default radiative fractions:
  - methane 0.20, propane 0.29, propylene 0.37, n-heptane 0.40;
  - 0.35 for species it does not list.
- **The solver** (`flamescapes.js`):
  - it absorbs and emits only through soot (`sootAbsorption` 500, τ = κ Y_soot l), with no CO2 or H2O;
  - the validation's burners radiate a prescribed fraction along 16 rays (CPU only);
  - the app's design fires have fraction 0, and their sooty cells radiate.
- **What follows.**
  - A clean methane layer (Steckler, NBS 100A, ATF: soot 0) cannot radiate to walls and floor, nor absorb anything.
  - For sooty fuels the prescribed fraction and the soot's own emission add up (the double count of plan 12.9, amendment 8).
  - The flame cells of a clean fuel lose heat only through the prescribed rays. The 3,000 K stops of ATF's 500 kW and pulsed runs fit that: they are methane, soot 0, on a burner under two cells wide.
- **What is missing:** gas radiation from CO2 and H2O, and a bound on flame cell temperature at the fuel's adiabatic flame temperature. The FDS Technical Reference Guide's gray gas or weighted-sum-of-gray-gases treatment would have to be read and transcribed before use. It was not read for this note.

### 4.4 Soot and visibility

- **The mass-specific extinction coefficient.** The FDS User's Guide gives K = K_m ρY_s and visibility S = C/K. C is 8 for a light-emitting sign and 3 for a light-reflecting one, and FDS's default is 3. Its default K_m is 8,700 m²/kg, "a value suggested for flaming combustion of wood and plastics". Its footnote gives 8,700 ± 1,100 m²/kg at 633 nm for most flaming fuels, from Mulholland and Croarkin (2000). That paper was not itself read for this note.
- **Wavelength and smouldering.** Börger et al. (2024) say K_m is close to fuel-independent for post-flame smoke, and "much lower and more variable" for smouldering and pyrolysis smoke. Their MIREX meters (880 nm, infrared) read lower than red light in the same smoke. A measured obscuration must therefore be compared at its own wavelength.
- **Soot yield depends on scale.** Gottuk, Mealy and Floyd (2008) measured soot yields in 100 kW fires under a hood:
  - ethylene 2.0 ± 0.2 %, about half the bench-scale 4.3 %;
  - propylene 4.8 %, against 9.5 to 9.6 % reported;
  - propane 0.41 %.

  They found the bench-scale yields 2 to 5 times larger than their measured ones.
- **Deposition.** The same corridor study (14.6 m long, 3.7 m wide, ceilings 2.7, 3.7 and 5.5 m) found:
  - FDS version 4 over-predicted the optical density 2 to 5 times;
  - its temperatures and velocities agreed with the measured;
  - soot deposited on the ceiling above the plume took up to 37 to 39 % of the soot produced (37 % in the abstract, 39 % in the conclusions).
- **FDS's own deposition model is not calibrated either.** Its aerosol deposition velocity gives δ 0.20 with σ̃M 0.98 (Table 16.1).
- **The solver** has no deposition and no agglomeration. Its K_m is fixed at 8,700 m²/kg (`flamescapesRun.js SOOT_EXTINCTION`), and its soot yield is the fire class's.
- **What follows.**
  - The app's visibility can be no better than the yield it is given.
  - A held-out test can judge how smoke is carried, for a given yield.
  - It cannot judge the visibility a user sees for an arbitrary fire.
  - The rerun's first smoke comparison (NIST/NRC, δ 1.76) over-predicts, as FDS does (1.24) and as Gottuk et al. found (2 to 5 times).

### 4.5 Walls

- **FDS** conducts heat in one dimension through up to 20 layers per surface, independent of the gas grid (User's Guide, 8.3).
- **The solver** lumps a wall into whole cells with no depth. The harness keeps each board's areal heat capacity (plan 12.3), but the lumped cell's face warms with its whole mass.
- **The scale of the error.** It is set by the penetration time L²/α. From the FDS inputs' properties:

| Surface (FDS input) | k (W/(m K)) | ρ (kg/m³) | c (kJ/(kg K)) | L | L²/α |
| --- | --- | --- | --- | --- | --- |
| Steckler's insulating board | 0.1 | 200 | 1.0 | 13 mm | about 6 min |
| UL/NIST vents' gypsum | 0.26 | 650 | 1.09 | 16 mm | about 12 min |
| UL/NIST vents' concrete floor | 1.8 | 2,280 | 1.04 | 102 mm | about 3.8 h |

- **Boards.** They heat through within the tests' 10 to 15 min windows, so lumping them is a modest error.
- **Thick walls.** Concrete's heated skin after 600 s is about √(αt) ≈ 21 mm deep. A 0.27 m concrete cell is about 13 times that, so its face warms roughly 13 times more slowly than a real concrete face. It would take too much heat from the layer early on.
- **What follows.**
  - The app gives every wall the solver's stone. The held-out tests had insulating boards (`calibrationStatus.js` COVERED).
  - So masonry and concrete are outside the tested range.
  - A run with heavy walls may read cooler than reality early on. This is untested. It bears on the user's worry that the fire looks weak, and it is worth a check.
- **What is missing:** a 1-D conduction model with a few nodes per wall face, as FDS has.

### 4.6 Instruments

- **Thermocouples.** FDS compares a modelled thermocouple, a bead's energy balance with its radiation and convection: a 1 mm bead with emissivity 0.85 by default (User's Guide 22.10.8). The validation inputs read here (ATF, DelCo, NIST smoke alarms, UL/NIST vents, WTC) use `QUANTITY='THERMOCOUPLE'`. The harness compares gas temperature (`validation-suite.mjs`, `sim.T`). A bare bead near a flame reads below the gas, and a bead in cool gas near hot surfaces reads above it. That fits the ceiling jet's high bias near the fire. It is also a candidate in NBS's cold far corridor trees, where the measured rise exceeds the model's. Modelling the bead needs the incident radiation at the point, which the solver does not keep. A first version can use the surfaces' temperatures in view.
- **Dry and wet gas.** Gas analysers usually read dry gas, with the water removed. FDS offers `DRY=T` for this (User's Guide 22.10.25), though the NIST/NRC and WTC inputs do not use it. The harness reads the wet mole fraction (`validation-suite.mjs moleFraction`). For heptane (C7H16 + 11 O2 → 7 CO2 + 8 H2O), worked here: a wet O2 drop of 0.070 brings 0.051 of water. A dry analyser would then read an O2 drop of 0.0625 and a CO2 rise of 0.0469, against 0.0445 wet.
  - Reading wet against dry analysers raises the model's O2 δ by about 11 % and lowers its CO2 δ by about 5 %.
  - That is the size and sign of NIST/NRC's gap between O2 (1.17) and CO2 (1.06) in the rerun.
  - Which basis the tests used must be read from the test reports (Hamins et al.), not inferred from this fit.

### 4.7 WTC's analyser: species and heat part ways

From the rerun's pairs (0.27 m), with the WTC data as the harness reduced them:
- The model's largest O2 drop at the analyser is 0.11 to 0.15 in all six tests. The measured drop is 0.052 to 0.069 in tests 1 to 3 (2.0 to 2.4 MW) and 0.100 to 0.107 in tests 4 to 6 (3.0 to 3.2 MW), within the compared windows.
- The thermocouple about 0.28 m from the analyser (TTRE1) reads 1.08 to 1.10 times measured in tests 4 to 6, and 1.11 to 1.32 in tests 1 to 3.
- The measured CO2 rise over O2 drop is 0.65 to 0.73, near heptane's 7/11 = 0.64. The analysers agree with each other and with the fuel.
- So at that point the model carries more product per degree of rise than the test gas did. This hints that two errors offset:
  - too little dilution, which raises both species and temperature;
  - too much heat loss, which lowers temperature only. Candidates are the prescribed radiant fraction of 0.44 on top of the soot's own emission, and the lumped walls.

  The guide's Table 3.55 puts WTC's flames at 0.8 to 1.1 of the ceiling height, so the analyser sits in or near the flame's spread under the ceiling. There the model's flame zone, the layers picked by Heskestad's height, matters.
- This is a hypothesis to test, not a finding. Under the protocol, a fix needs a check that uses no validation data.
- It also warns that a layer temperature in band can hide offsetting errors.

### 4.8 What the solver and harness lack, in one list

- **Solver:**
  - gas radiation from CO2 and H2O;
  - a bound on flame temperature;
  - wall conduction depth;
  - face apertures for openings;
  - soot deposition and agglomeration;
  - a subgrid turbulence model (it relies on its limiter and a fitted confinement).
- **Harness** (`tools/calibration/lib/fds.mjs` reads MESH, MULT, OBST less HOLE, open VENT, SURF with HRRPUA and RAMP_Q, RAMP, MISC TMPA, DEVC, REAC and PROP):
  - no fans or HVAC;
  - no leakage paths (`LEAK_AREA`, `ZONE`);
  - no obstructions removed on a timer (`CTRL`), so no vents opened or closed during a run;
  - thin obstructions are dropped;
  - no thermocouple model;
  - no dry gas reading;
  - no optical density path device other than NIST/NRC's.

## 5. Public measured data that could be added

"Fit" is whether the harness could run the dataset on 0.27 to 0.34 m cells as it stands or with small additions. Licences are as read on 2026-11-12. The `firemodels/exp` notice (LICENSE.md) covers NIST-developed software and says nothing about data from other organisations. Plan 12.1 keeps to NIST, NBS and other US federal data on that ground. Test sizes and geometry are from the guide's chapter 3 unless stated.

| Dataset | Laboratory, year | Measures | Fires; enclosure | Terms | Fit at 0.27 to 0.34 m | Helps |
| --- | --- | --- | --- | --- | --- | --- |
| ATF corridors | ATF Fire Research Laboratory, 2008 | layer, ceiling TCs, bi-directional probes in the corridors (trees H to K) | natural gas, 48 to 485 kW, 0.45 m burner; two 17 m corridors and a stair | US federal (ATF) | in the suite since the rerun; two 0.27 m runs stopped (section 2.2) | layer, ceiling jet, corridor velocity |
| UL/NIST vents | NIST Fire Fighting Technology Group at UL, 2012 (Opert 2012) | 2 TC trees; door TCs and 10 door velocity probes (`Door_V1` to `Door_V10`); vent probes. The guide uses only the TCs. | natural gas, 430 to 2,188 kW; 6.1 x 4.3 x 2.4 m, one 0.9 x 2.1 m door, two 1.2 m ceiling vents | NIST | good for the first interval of each fire (intervals 1, 11, 14, 20: door only, vents shut); all 31 intervals need timed obstructions | door velocity, neutral plane, layer, ceiling jet |
| DelCo trainers | NIST Fire Fighting Technology Group (Willi, NIST TN 1953) | TC arrays, gas velocity, heat flux, gas sampling (O2, CO2) | three propane burners, 440 to 1,190 kW; one- and two-level training structures | NIST | fair: the FDS inputs model leakage (`ZONE`, `LEAK_AREA`), which the harness lacks; tests with open doors or vents only | O2, CO2, layer, ceiling jet; a third lab group for species |
| NIST smoke alarms | NIST (Bukowski et al., TN 1455-1) | TCs; smoke columns `SMA_1` to `SMF_4` in the data files (the FDS input places its OPTICAL DENSITY devices at the `_1` positions); alarm times | chair, couch or mattress, flaming; 100 to 350 kW t² ramps fitted in FDS; manufactured home, peaked ceiling to 2.4 m | NIST | fair: 0.6 m interior doors are about two cells; a sloped ceiling; the fire room's TCs are not independent | smoke transport (second dataset), remote ceiling jet |
| Bryant doorway | NIST (Bryant 2009) | PIV in the doorway's lower two-thirds; probes | natural gas, 34 to 511 kW; ISO 9705 room raised 0.3 m, doorway 0.8 x 1.95 m (FDS input) | NIST | fair: 3 cells across on 0.27 m; inflow only | door inflow velocity |
| WTC openings | NIST (in the suite's data) | 16 bi-directional probes, PI1 to PI8 at the inlets and PO1 to PO8 at the outlets; the guide compares them | as WTC | NIST | good: data and geometry already cached; one lab with NIST/NRC | opening velocity |
| FSRI compartments 2018 | UL FSRI (NIJ award 2017-DN-BX-O163) | 4 TC arrays of 8; O2 at 0.6 and 1.8 m below the ceiling; 7 door probes with TCs; heat flux; calorimetry for most open-door fires over 100 kW | furniture; 3.7 x 3.7 x 2.4 m, door 0.9 x 2.0 m, open or shut; 120 tests | MIT (GitHub); Zenodo lists "Open Access" | good geometry; the fire is real furniture, so the input is the hood's measured heat release, with its delay | door velocity, neutral plane, O2, layer; an independent laboratory |
| FSRI residential gas burner 2016 (the guide's UL/NIJ houses) | UL FSRI (Madrzykowski and Weinschenk 2019) | TC arrays; 5 probes per opening; pressure | natural gas burner 0.6 x 0.6 m; ranch house 13.9 x 7.7 m and two-storey house; leakage 0.08 and 0.18 m² at 10 Pa | MIT (GitHub) | fair: leakage; the guide finds the opening velocities too noisy | layer and ceiling jet in a house |
| NIST FSE 2008 | NIST (Lock et al., TN 1603) | O2, CO2, CO, soot, TCs | under-ventilated by design; ISO 9705 room, door 0.1 to 0.8 m wide | NIST | poor: narrow doors and under-ventilated burning, which the solver does not treat | none soon |
| NRCC smoke tower | NRC Canada, 2006 to 2007 | O2 and CO2 in the stair shaft; TCs | propane, 3 MW; a 10-storey tower | not stated; NIST's notice does not cover it | fair geometry; terms unknown | species, if permitted |
| LLNL, FM/SNL, PRISME, VTT large hall | LLNL 1986; FM/SNL 1985; OECD/NEA at IRSN; VTT | layer, species (LLNL, PRISME) | mostly forced ventilation (FM/SNL all; PRISME all); VTT 19 m hall | PRISME reports "not publicly available" (guide); others not stated | poor: the harness has no fans; VTT's hall is not flat-ceilinged | none soon |
| FAA cargo, FM/FPRF data centre | FAA; FM Global | obscuration; smoke concentration | 5 kW in 1.4 to 1.7 m compartments; data centre with 78 or 265 air changes an hour | not stated | poor: 4 to 6 cells tall; forced flow | none |
| Fleury heat flux | University of Canterbury, MA thesis 2010 | heat flux beside propane burners | 100 to 300 kW; open | not stated | fair, outdoors | the burners' radiant fraction (a check, not a held-out set) |
| Börger et al. 2024 (data: Zenodo) | Universities of Wuppertal and Duisburg-Essen, Forschungszentrum Jülich | extinction coefficients by LED photometry, vertical and diagonal; MIREX at 1.52, 2.3 and 3.3 m (880 nm) | 500 g n-heptane in a 0.335 m pan, burned in about 225 s (about 99 kW on average, computed from 44.6 MJ/kg); ceiling 3.37 m; ventilation started 6 min after ignition | CC BY 4.0 | fair: geometry must come from the record's input files; raw images are about 30 GB, but MIREX.zip and data_analysis.zip are under 2 MB | smoke transport, independent laboratory, spatially resolved |
| Gottuk et al. 2008 corridor | Hughes Associates for the Fire Protection Research Foundation | optical density, TCs, velocity | 100 kW gas fires; corridor 14.6 x 3.7 m | data not located | n/a | the deposition evidence (section 4.4) |

## 6. Plan per quantity

The protocol's section 5 binds every item. A solver change is made only when a check that uses no validation data shows the fault. After it, every held-out comparison is rerun and the confinement refitted, and the results before and after are both reported. A full suite run takes about 1.5 to 3 h at 4 to 6 jobs (plan 12.8; the rerun took 5,510 s at 6 jobs). The refit took about 20 minutes overnight. A new dataset is pre-registered (tests, devices, windows, σ̃E) before any of its results are read. A dataset used to diagnose a fault stops counting as held out for that quantity, and the plan says so.

Effort is human working time, with review, tests and the GPU port where the solver changes (the GPU kernels must follow the CPU). Compute time is separate.

### 6.1 Smoke layer temperature (indicative)

The verdict fails on corridors. Single rooms are in band: Steckler, NIST/NRC and WTC on 0.27 m, in both runs. Plan 12.9 (item 4) found the NBS scene exact on 0.20 m and still cold down the corridor, so geometry is not the cause.

1. **Audit the corridor's energy, using no validation data** (2 to 3 days). Run `tools/calibration/energy-budget.mjs` on an NBS-like corridor. Track, segment by segment, the heat carried, the heat into the walls and the heat mixed below the interface. On 0.27 m the corridor layer is three to four cells deep, so numerical entrainment across the interface over 12 m is a suspect. So are the lumped walls and the missing gas radiation (section 4.3).
2. **Thermocouple model in the harness** (3 to 5 days, with the ceiling jet's item 1). It is instrument fidelity, set from the guide's method, not a fit. It may move NBS's far trees.
3. **A 1-D wall model** (2 to 3 weeks including the GPU and the rerun). A few nodes per exposed face, as FDS has. It is justified by the penetration scaling of section 4.5 and checked against a slab's analytic heating, not against fire data. It also lets the app's stone and concrete walls behave as walls.
4. **New held-out data** (3 to 5 days each). Pre-register before reading results:
   - UL/NIST vents' first intervals: a single room, NIST, 2012;
   - FSRI 2018: a single room, UL;
   - DelCo, with open doors: multi-room, NIST FFTG.
5. **The claim.** "Calibrated in single rooms" would be a new claim class. It is honest only if pre-registered now and judged on fresh single-room data (step 4), not on the four series already seen.

Realistic outcome: calibrated for single rooms on 0.27 m if steps 4 and 5 hold, with corridors stated as a limit. Multi-room calibration needs step 1 to find a cause and step 3 or a transport change to fix it.

### 6.2 Ceiling jet (indicative)

The third laboratory and instrument fidelity are the gaps. The guide's own statistic (1.08, 0.13) rests on 22 datasets.

1. **ATF's stopped runs** (3 to 5 days). The 0.27 m 500 kW and pulsed runs stopped above 3,000 K on a burner under two cells wide (sections 2.2 and 4.2). Check, with no validation data, an open 500 kW methane burner on 0.27 m cells against McCaffrey and Heskestad. If the flame cells overheat there too, bound them at the fuel's adiabatic flame temperature (a sourced value), or spread the burner's heat over at least the guide's three cells. Then rerun everything.
2. **Thermocouple model** (shared with 6.1 item 2).
3. **A third and fourth dataset:** UL/NIST vents' ceiling thermocouples (`BFLCEIL`, `FLCEIL` and others) in their first intervals, and DelCo with open doors. NIST smoke alarm stations outside the fire room could add a fifth (the fire room's are not independent).
4. **The jet's thickness.** At 0.27 m a thermocouple 2.5 cm under the ceiling reads a cell 0.27 m deep. Keep the present rule (read the cell that holds the device) and report the bias by distance from the plume, r/H, as the guide's parameter table does. Do not fit a near-wall correction.

Realistic outcome: calibrated on 0.27 m if ATF runs clean and one of the new datasets is in band, since NIST/NRC (1.03) and WTC (1.19) already are in the rerun.

### 6.3 O2 and CO2 (not calibrated)

WTC sits outside the indicative band. Under the claim protocol ("outside the indicative band in none") that blocks "calibrated" whatever is added, so WTC must be understood first.

1. **Dry or wet** (1 day). Read the gas sampling method in the NIST/NRC and WTC test reports (Hamins et al.). If the analysers read dry gas, read the model dry (section 4.6). This is instrument fidelity, set from the reports.
2. **WTC's analyser point** (1 week). It is development work, so WTC's species leave the held-out set while it runs.
   - Map the model's product and temperature fields around the analyser.
   - Compare the heat budget of the burner's gas with and without the prescribed radiant fraction (section 4.7).
   - Make any fix only from a check with no WTC data, for example the burner's radiated share against its stated fraction on an open burner.
3. **New data** (4 to 5 days each):
   - DelCo's O2 and CO2 (NIST FFTG, propane, leakage in the inputs: open-door tests only);
   - the FSRI 2018 rooms' O2. O2 consumed per joule is nearly the same for all fuels (13.1 MJ per kg of O2, Huggett, as `flamescapes.js` cites), so a furniture fire with measured heat release tests O2 transport. CO2 needs the fuel's composition, which furniture does not give.
4. **Two O2 heights.** NIST/NRC's lower analyser reads 1.20 to 1.44 times measured on 0.27 m, against 1.02 to 1.21 for the upper one. Products reach the lower layer too readily, the same numerical entrainment as 6.1 item 1.

Realistic outcome: indicative after items 1 and 3. Calibrated only if WTC moves inside the indicative band for a reason found without its data, and three datasets are in band.

### 6.4 Door flow and the neutral plane

The peak outflow at the top probe is the hardest quantity on 2 to 4 cell doors. The guide's Steckler runs used 5 cm cells.

1. **More data, at once** (1 to 2 weeks for all three):
   - WTC's 16 opening probes: data and geometry already cached; the guide compares them;
   - UL/NIST vents' 10 door probes, first intervals;
   - the FSRI 2018 rooms' 7 door probes, open-door tests with calorimetry.

   With Steckler that makes three laboratories: NBS 1980, NIST FFTG 2012 and UL FSRI 2018. The neutral plane comes from the same profiles; WTC's openings are one-way, so they give velocity only.
2. **Door mass flow as a quantity** (2 to 3 days to pre-register and code). Integrate each measured centreline profile with the door thermocouples' density, as the model's flow is integrated. Report it beside the peak velocity, which stays as the guide's metric. An integral is less sensitive to a one-cell soffit. It must be pre-registered as a new quantity with its own σ̃E, propagated from the probes' and thermocouples' σ̃E (Table 4.1), before any results.
3. **Face apertures** (1 to 2 weeks with the GPU and rerun). A door keeps its true width on any cell (section 4.2). Check it on a sealed-box flow, Bernoulli across an orifice, with no fire data.

Realistic outcome: the neutral plane calibrated after item 1. Door mass flow possibly calibrated after items 2 and 3. The peak outflow likely stays indicative at best on these cells. The app should say which it means.

### 6.5 Smoke visibility (soot)

Visibility is what a user reads, and it depends first on the soot yield, an input. The honest target is the transport of smoke for a given yield.

1. **Fix σ̃E's source** (half a day): the published 0.07 or the repository's 12 % (section 3.3), pre-registered.
2. **NIST smoke alarms' optical density** (1 to 1.5 weeks).
   - Read TN 1455-1 for the meters' path, wavelength and units.
   - Pre-register two metrics: the optical density at each station; and its ratio to the fire room's meter, in which the yield and K_m cancel.
   - The yield is the FDS input's. In SDC02, 0.135 mol of soot per mol of C6H10O5 is about 0.010 kg/kg, computed.
3. **Börger et al.** (1 to 2 weeks).
   - Geometry and boundary conditions from the record's input files.
   - MIREX and LED extinction coefficients, compared at their own wavelengths.
   - Before the ventilation starts at 6 min.
   - Heptane's soot yield from a source to be read; none was read here.
4. **The claim.** With NIST/NRC, these make three datasets from three laboratories for smoke obscuration. The app would then say "smoke transport for a given yield is calibrated" (if it is), and keep the visibility itself labelled by its yield's source.
5. **Deposition** (1 to 2 weeks, off by default). A thermophoretic and turbulent wall sink is physics with its own literature. It must not be fitted on these held-out sets. Gottuk et al.'s 37 to 39 % at the plume's impingement shows the size of the effect, and FDS's own deposition statistics (δ 0.20, σ̃M 0.98) show it is hard. Add it only with a source and a check that uses no fire data. Expect it to lower the over-prediction seen on NIST/NRC.

Realistic outcome: indicative to calibrated for transport given a yield. Visibility as such stays as good as its yield, and the app should say so.

### 6.6 The 0.20 and 0.34 m cells

- **0.20 m** is a compute problem.
  - Run NIST/NRC (10 tests), WTC (6) and ATF (6) on 0.20 m: about an hour a test for NIST/NRC and WTC (plan 12.1).
  - Memory scales with the cells, (0.27/0.20)³ ≈ 2.5 times. A NIST/NRC run at about 200 MB on 0.27 m becomes about 500 MB, so 2 jobs fit in this PC's free memory overnight: 8 to 16 h.
  - The suite is CPU only, because the burner's rays are CPU only (`flamescapes.js createBurner`). A laptop GPU does not help here until the rays are ported.
  - Effort: about 1 day of work, once the PC is free.
- **0.34 m** has a physical bias.
  - The depth reads 1.28 (Steckler) and 1.29 (NBS), the small rooms at D*/δx 0.7 to 1.3, against 0.95 and 1.10 in the large ones (plan 12.8).
  - A deeper modelled layer means a lower interface. With fewer cells across the interface, numerical thickening is the likely cause, but this is untested.
  - Check it on a two-layer box with no fire: hold a sharp interface, step it, and measure the Janssens and Tran depth on each cell size.
  - If that confirms it, the fix is in transport, and it moves every verdict. Otherwise state 0.34 m as indicative, with the range of D*/δx where it holds.

## 7. Order of work

1. **Instrument and input fidelity, no new physics** (1 to 1.5 weeks; one full rerun):
   - dry or wet gas (6.3 item 1);
   - thermocouple model (6.1 item 2, 6.2 item 2);
   - σ̃E for smoke (6.5 item 1).
2. **The ATF burner and flame temperature** (6.2 item 1), because it stops runs. It needs a check with no validation data first.
3. **New held-out data** (2 to 3 weeks; pre-register all of it before reading any): WTC openings, UL/NIST vents' first intervals, FSRI 2018, DelCo, the NIST smoke alarms' optical density.
4. **Door mass flow and the 0.20 m runs** (6.4 item 2, 6.6).
5. **Physics, one at a time**, each with its own check and full rerun: face apertures, then 1-D walls, then gas radiation, then deposition (2 to 3 weeks each).

Items 1, 3 and 4 can lift the neutral plane and the ceiling jet, and perhaps the single-room layer temperature, without touching the solver. O2 and CO2 need WTC understood. Door peak outflow and visibility need physics or a change of claim.

## 8. Rules kept

- Nothing in the held-out sets sets a parameter. A dataset used to diagnose a fault leaves the held-out set for that quantity.
- New datasets, metrics and claim classes are pre-registered with their dates before their results are read.
- Licences:
  - NIST, NBS and US federal data with NIST's notice and a note of changes, as in plan 12.5;
  - MIT data (FSRI) with its copyright notice;
  - CC BY 4.0 data (Börger et al.) with attribution.

  Data whose terms are not stated (NRCC, VTT, FM, FAA, Fleury) are not added without permission.
- The app states a list of quantities with their status, cells and fire range, never "the fire is calibrated".

## 9. Sources

Read for this note (in full, or the sections named):

- McGrattan, K., Hostikka, S., Floyd, J., McDermott, R., Vanella, M., Mueller, E., Paul, C. *Fire Dynamics Simulator Technical Reference Guide, Volume 3: Validation.* NIST Special Publication 1018-3, sixth edition, July 10, 2026, revision FDS-6.11.1-0-gff928db. Read: chapter 3 (sections 3.3, 3.11, 3.19, 3.23, 3.30, 3.54, 3.58, 3.67, 3.72, 3.76, 3.95, 3.97, 3.109, 3.115 with Tables 3.55 and 3.56), Table 4.1, sections 5.24, 6.1, 7.1.21, 7.2.1, 8.1 to 8.16 (door, corridor and opening parts), 9.1 and 9.2, and Table 16.1. The DOI it prints, 10.6028/NIST.SP.1018, is registered on Crossref to the 2006 version 4 Technical Reference Guide, so none is given for this volume.
- McGrattan, K., et al. *Fire Dynamics Simulator User's Guide.* NIST Special Publication 1019, sixth edition, July 10, 2026, revision FDS-6.11.1-0-gff928db. Read: sections 8.3, 14.1 to 14.3 (Table 14.1), 22.10.5, 22.10.8 and 22.10.25. Its printed DOI, 10.6028/NIST.SP.1019, is registered to the 2004 version 4 user's guide.
- `firemodels/fds`, `Utilities/Python/FDS_validation_scatterplot_inputs.csv` (master, read 2026-11-12), and `Utilities/Python/FDS_validation_dataplot_inputs.csv` at the suite's pinned commit 193faff (`tools/calibration/data/cache`). The FDS inputs read: ATF_Corridors_240_kW, Bryant_160_kW, ISONG3, NIST_Smoke_Alarms_SDC02, Steckler_010, DelCo Test_02, UL_NIST_Vents_Test_1, WTC_01, NIST_NRC_01.
- `firemodels/exp`: the root listing and LICENSE.md; the UL_NIST_Vents and NIST_Smoke_Alarms listings; the column headers of `UL_NIST_Vents_Test_1.csv` and `NIST_Smoke_Alarms_SDC02.csv` (read 2026-11-12).
- U.S. Nuclear Regulatory Commission and Electric Power Research Institute. *Nuclear Power Plant Fire Modeling Analysis Guidelines (NPP FIRE MAG).* NUREG-1934 / EPRI 1023259, final report, November 2012. Read: section 2.3.5 (FDS, grid size) and appendix F.4.2. No DOI.
- Gottuk, D., Mealy, C., Floyd, J. (2008). Smoke transport and FDS validation. *Fire Safety Science* 9, 129–140. https://doi.org/10.3801/IAFSS.FSS.9-129 (Crossref: verified). Read in full.
- Börger, K., Belt, A., Schultze, T., Arnold, L. (2024). Remote sensing of the light-obscuring smoke properties in real-scale fires using a photometric measurement method. *Fire Technology* 60(1), 19–45 (online 2023-09-22). https://doi.org/10.1007/s10694-023-01470-z (Crossref: verified; CC BY 4.0). Read in full.
- Börger, K., Belt, A., Schultze, T., Arnold, L. (2022). *Remote sensing of the light-obscuring smoke properties in real-scale fires using a photometric measurement method* [data set]. Zenodo. https://doi.org/10.5281/zenodo.7016689 (DataCite: verified; CC BY 4.0). Read: the record's metadata and file list.
- Willi, J., Madrzykowski, D., McKinnon, M., Dow, N. (2021). *2018 Compartment Fire Experimental Data*, v2.0 [data set]. Zenodo. https://doi.org/10.5281/zenodo.6376302 (DataCite: verified; rights "Open Access"). GitHub `ulfsri/fsri-compartments-2018` (MIT licence). Read: the README and `02_Info/README.md`.
- McKinnon, M., Weinschenk, C., Madrzykowski, D., Stakes, K. (2021). *2016 Residential Gas Burner Experimental Data*, v1.0 [data set]. Zenodo. https://doi.org/10.5281/zenodo.5703080 (DataCite: verified). GitHub `ulfsri/fsri-residential-gasburner-2016` (MIT licence). Read: the README and `02_Info/README.md`.

Cited through the FDS Validation Guide or User's Guide (their metadata verified; their content not read for this note):

- Mulholland, G. W., Croarkin, C. (2000). Specific extinction coefficient of flame generated smoke. *Fire and Materials* 24(5), 227–230. https://doi.org/10.1002/1099-1018(200009/10)24:5<227::AID-FAM742>3.0.CO;2-9 (Crossref: verified). Its value is quoted as the FDS User's Guide gives it.
- Bryant, R. A. (2009). A comparison of gas velocity measurements in a full-scale enclosure fire. *Fire Safety Journal* 44(5), 793–800. https://doi.org/10.1016/j.firesaf.2009.03.010 (Crossref: verified).
- Bukowski, R. W., Peacock, R. D., Averill, J. D., Cleary, T. G., Bryner, N. P., Walton, W. D., Reneke, P. A., Kuligowski, E. D. (2008). *Performance of Home Smoke Alarms: Analysis of the Response of Several Available Technologies in Residential Fire Settings.* NIST Technical Note 1455-1. https://doi.org/10.6028/NIST.TN.1455-1r2008 (Crossref: verified, the 2008 revision the guide cites).
- Willi, J. (2017; the guide gives 2016). *Propane Gas Fire Experiments in Residential Scale Structures.* NIST Technical Note 1953. https://doi.org/10.6028/NIST.TN.1953 (Crossref: verified, issued 2017-02).
- Lock, A., Bundy, M., Johnsson, E., Hamins, A., Ko, G., Hwang, C., Fuss, P., Harris, R. (2008). *Experimental Study of the Effects of Fuel Type, Fuel Distribution, and Vent Size on Full-Scale Underventilated Compartment Fires in an ISO 9705 Room.* NIST Technical Note 1603. https://doi.org/10.6028/NIST.TN.1603 (Crossref: verified).
- Madrzykowski, D., Weinschenk, C. (2019). *Impact of Fixed Ventilation on Fire Damage Patterns in Full-Scale Structures.* UL Firefighter Safety Research Institute. https://doi.org/10.54206/102376/wzkw3444 (Crossref: verified).
- Opert, K. (2012). *Assessment of Natural Vertical Ventilation for Smoke and Hot Gas Layer Control in a Residential Scale Structure.* MA thesis, University of Maryland. No DOI.
- Sheppard, D., Klein, B. (2009). *Burn Tests in Two Story Structure with Hallways.* ATF Laboratories. No DOI.
- Fleury, R. (2010). *Evaluation of Thermal Radiation Models for Fire Spread Between Objects.* MA thesis, University of Canterbury. No DOI.

The repository's own records:
- `benchmark/results/2026-11-12-validation-final.json` (the 04:17 rerun, untracked, not yet reviewed);
- `tools/calibration/data/wtc.json` and `nistnrc.json` (the reduced measured data);
- `tools/calibration/lib/fds.mjs`, `lib/stats.mjs` and `validation-suite.mjs`;
- `topologicpy-web-frontend/src/core/flamescapes.js`, `flamescapesRun.js` and `calibrationStatus.js`;
- plan sections 12.1 to 12.9 and the claim protocol.

The steady-state test sources of the four held-out series (Steckler et al. 1982; Peacock et al. 1991; Hamins et al. 2005 and 2006) are cited in plan section 12.1, not again here.
