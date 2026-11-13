# Fire validation data: an inventory per quantity (2026-11-12)

Status: inventory only, step 1 of the user's request to validate every fire quantity. It comes before any pre-registration. No solver was run and nothing was compared; data files were read as data. It continues [fire-calibration-plan-2026-11-11.md](fire-calibration-plan-2026-11-11.md), section 12, whose five held-out series are Steckler, NBS multi-room, NIST/NRC, WTC and ATF corridors.

## 1. Sources and how to read this

**Where the data come from.**
- NIST's experimental database `firemodels/exp`, commit 704e0d9, the cache the suite already uses. File names were listed without downloading, then candidate headers were read.
- NIST's `firemodels/fds`, commit 193faff:
  - the FDS Validation Guide's own dataset list per quantity (`Utilities/Python/FDS_validation_dataplot_inputs.csv`);
  - its Experiment chapter, for each series' laboratory, geometry and fire;
  - one FDS input per candidate, to size its domain, fire and `&REAC`.
- UL FSRI's `ulfsri/fsri-compartments-2018` on GitHub (repository metadata and READMEs only).
- The OECD/NEA catalogue, through search results only. Its pages refused scripted fetches (HTTP 403).

**What "independent" means here.** A laboratory other than NBS/NIST: NBS became NIST in 1988, so the two count as one. "Series" is a test series. The claim protocol counts series; the user's rule asks for different series or laboratories. Both are given.

**What "usable" means.** A series is usable when:
- its data are public with reuse terms we can rely on;
- its geometry and fire can be represented on 0.20 to 0.34 m cells, at about six or more cells per room height and a prescribed, measured heat release;
- it needs no feature the solver or the harness lacks, or says which one it needs.

**Runtimes.** Each estimate is per test, for 600 s unless stated, CPU, one job. It is the mesh volume (plus 15 % for the walls) over h³, times a rate measured on 6ee8c93 in seconds per cell per simulated second:
- 3e-5 for fires up to about 250 kW (Steckler, NBS, ATF: 2.6e-5 to 4.5e-5);
- 9e-5 for 0.4 to 1.2 MW (NIST/NRC: 9.2e-5);
- 1.5e-4 for 2 MW and more in a small room (WTC: 1.5e-4).

Each rate is scaled by 1.35 on 0.20 m cells and by 0.75 on 0.34 m cells, from the measured ratios. Treat every estimate as within a factor of two.

## 2. Terms of use

- **`firemodels/exp` has one licence file,** NIST's software notice. It covers works by NIST employees, which are not subject to copyright in the United States. No other folder of the candidates below carries a licence or terms note, except two READMEs (Hamins gas burners, NIST/NRC corner effects) that describe data, not terms.
- **Data of other US federal agencies are US Government works:** ATF corridors and FAA cargo compartments.
- **Data of other laboratories in the repository carry no stated terms there.** These are LLNL, FM/SNL, UL/NFPRF, UL/NIJ, VTT, University of Canterbury, Virginia Tech/Lattimer, NRC Canada, FM/FPRF and RISE/SP. Decision 4 of the 2026-11-08 plan applies: ask the owner first.
  - LLNL's data come from a DOE report and FM/SNL's from NUREG/CR reports; both reports are public, but the repository grants nothing for the data.
- **UL FSRI's 2018 compartments** are under the MIT licence (GitHub; DOI 10.5281/zenodo.5703474, checked at DataCite for the 2026-11-08 plan). They are free to reuse with the notice.
- **PRISME is not usable.**
  - The repository holds DOOR (`PRISME/PRS_D1` to `D6`), SOURCE (`PRS_SI_D1` to `D6a`), LEAK (`PRISME_Test_LK_1` to `4`) and one PRISME-2 test (`PR2_FES_PA2a`), with HGL reductions, and the FDS guide has 21 PRISME inputs.
  - No terms note covers them. The FDS Validation Guide says the PRISME test reports are not public.
  - The OECD/NEA catalogue entries for the PRISME and PRISME-2 data packages (CSNI2006/03 and CSNI2042), as the search results show them, restrict distribution, subject it to prior approval through the NEA, and use a non-disclosure order form. The pages refused scripted fetches, so their wording was not read directly.
  - What is public is only that the files can be downloaded. Nothing grants their reuse, so PRISME is not usable without the NEA's approval.

## 3. Per quantity

### 3.1 Hot gas layer temperature and depth

| Series | Laboratory | Measured | Geometry, fire | FDS input | Terms | Harness work | Min per test (0.20 / 0.27 / 0.34 m) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Steckler, NBS multi-room, NIST/NRC, WTC (wired) | NBS/NIST | trees | as plan 12 | yes | NIST | none | as run |
| ATF corridors (wired) | ATF | 5 trees | as plan 12 | yes | US Gov | none | 29 to 39 / 9 / 4 |
| **UL FSRI 2018 compartments** (`ulfsri/fsri-compartments-2018`) | UL FSRI | 4 trees of 8 TCs, 25 mm and 0.3 m steps below the ceiling | 3.7 x 3.7 x 2.4 m, 0.9 x 2.0 m door open or shut; natural gas 100, 250, 500 kW at 4 positions; 92 burner tests (46 open) | no (a box; easy by hand) | MIT | new reducer (CSV with a units row), a hand-built scene | 5 / 1 / under 1 |
| UL/NIST vents (`UL_NIST_Vents`) | NIST (FFTG, at UL's facility) | 2 trees | 6.1 x 4.3 x 2.4 m, a door and two 1.2 m ceiling vents opened and shut in time; 0.5 to 2 MW; 4 tests, 31 quasi-steady intervals | yes (4) | NIST | vents changing in time (or the first interval only); 100 min tests | 38 / 12 / 4 for 600 s |
| DelCo trainers (`DelCo_Trainers`) | NIST (FFTG) | 5 and 6 trees | training buildings, 1 and 2 levels; 3 propane burners | yes (9) | NIST | multi-room reducer; burners as controls | 59 / 18 / 7 |
| NIST Composite Beam (`NIST_Composite_Beam`) | NIST | trees | 12.8 x 3 x 4 m bay; 4 MW natural gas | yes (4) | NIST | none new | 45 / 13 / 5 |
| NIST E119 compartment (`NIST_E119_Compartment`) | NIST | trees | 10.8 x 7 x 3.8 m; to 10 MW | yes (3) | NIST | none new; beyond the app's fires | 101 / 30 / 11 |
| NIST FSE 2008 (`NIST_FSE_2008`) | NIST | 2 trees | ISO 9705 room, under-ventilated, heptane spray and others | yes (25) | NIST | spray burner; burning outside the door | 16 / 5 / 2 |
| UL/NIJ houses (`UL_NIJ_Houses`) | UL FSRI | trees | ranch and colonial houses; 250 kW gas burner; doors opened in time | yes (6) | not stated (ask UL) | openings changing in time | 24 / 7 / 3 |
| LLNL enclosure (`LLNL_Enclosure`) | LLNL (DOE) | TC averages | 6 x 4 x 4.5 m; methane 50 to 400 kW; forced flow 100 to 500 g/s, door shut | yes (64) | not stated (DOE report) | a prescribed fan | 6 / 2 / under 1 |
| FM/SNL (`FM_SNL`) | FMRC for SNL/NRC | layer files | 18 x 12 x 6 m; 0.5 to 2 MW; forced ventilation | yes (19) | not stated (NUREG/CR) | a prescribed fan | 236 / 71 / 27 |
| VTT hall, NRCC tower, PRISME | VTT, NRC Canada, IRSN | | 19 to 30 m tall; or restricted | yes | ask; PRISME restricted | | over 200 even on 0.34 m (VTT, NRCC) |

**Three or more independent laboratories, usable now: yes.** NBS/NIST (several series), ATF, and UL FSRI 2018 under MIT. UL/NIJ (the same UL laboratory), LLNL and FM/SNL would add more if their terms are cleared and a fan is added to the harness.

### 3.2 Ceiling jet

The FDS guide's ceiling-jet set is near-ceiling thermocouples away from the plume. Wired: NIST/NRC, WTC (NIST) and ATF.

Candidates:
- UL FSRI 2018: each tree's top thermocouple, 25 mm below the ceiling. The plume's quadrant must be excluded, so the reducer must pick the quadrants away from the burner.
- NIST: UL/NIST vents (62 in the guide), DelCo (49), Composite Beam (26), FSE 2008 (49), smoke alarms (40, a furniture fire).
- UL/NIJ (UL FSRI, 13): terms to ask.
- FM/SNL (38): needs a fan.
- Already used to set wallConvection, so not held out: Vettori and UL/NFPRF.

**Three or more independent laboratories, usable now: yes** (NIST, ATF, UL FSRI 2018). Runtimes as in 3.1.

### 3.3 O2, CO2 and CO

| Series | Laboratory | Measured | Notes | Terms |
| --- | --- | --- | --- | --- |
| NIST/NRC, WTC (wired) | NIST | O2, CO2 (CO in NIST/NRC, 5 tests) | yields documented | NIST |
| UL FSRI 2018 | UL FSRI | O2 only, 0.6 and 1.8 m below the ceiling in each quadrant | natural gas; no CO2 or CO | MIT |
| DelCo trainers | NIST | O2, CO2, CO at 2 points | propane, yields in the FDS input | NIST |
| NIST FSE 2008 | NIST | O2, CO2, CO front and rear; also soot mass fraction | under-ventilated: the yields of CO and soot depend on combustion the burner does not model | NIST |
| NIST smoke alarms | NIST | gas stations A, B, C, F (O2, CO2, CO) | furniture fire; heat release and yields estimated | NIST |
| LLNL enclosure | LLNL | O2, CO2 | forced flow | not stated |
| FAA cargo | FAA | CO, CO2 | 5 kW in a 1.4 m tall compartment: 4 to 7 cells tall, D*/δx 0.3 to 0.6 | US Gov |

**Three or more independent laboratories: no.**
- **O2:** NIST and UL FSRI are usable now. A third needs LLNL, with its terms cleared and a fan in the harness; FAA's compartment is too small for the app's cells.
- **CO2:** only NIST is usable; no second laboratory without LLNL.
- **CO:** only NIST, and the burner prescribes the CO yield, so CO tests transport, not CO formation. Under-ventilated CO is not representable.

### 3.4 Door and vent velocity, and mass flow

| Series | Laboratory | Devices | Notes |
| --- | --- | --- | --- |
| Steckler (wired) | NBS | door or window centreline probes, 16 heights | wired |
| NIST/NRC (open-door tests) | NIST | door bi-directional probes (`BP Door-1` to `14`, file_2), door mass flow in, out and net (kg/s, file_1) | the data are in the cache; reducer and an opening device needed |
| WTC | NIST | 8 inflow probes in the low west opening, 8 outflow probes in the high east opening, with temperatures (`WTC/WTCP1T*_velo.csv`; the guide compares 94) | each opening carries flow one way: velocities yes, neutral plane no |
| Bryant doorway (`Bryant_Doorway`) | NIST | PIV across an ISO 9705 doorway, lower two-thirds only; 34 to 511 kW | FDS inputs (7); steady profiles; 2 / under 1 / under 1 min |
| UL/NIST vents | NIST | door and ceiling-vent velocity profiles, 10 heights each, with temperatures | vents changing in time |
| DelCo trainers | NIST | 3 arrays of 8 bi-directional probes | |
| UL FSRI 2018 | UL FSRI | 7 doorway bi-directional probes, 25 cm apart, centred; open-door tests | MIT |
| UL/NIJ houses | UL FSRI | 5 probes per opening at several openings | terms to ask |
| ATF corridors | ATF | 4 arrays of 4 probes in the corridors (`VEL_BDP_H`, `I`, `J`, `K`; the guide compares 46) | corridor flow, not a door |

**Three or more independent laboratories, usable now: yes** for velocity: NBS/NIST, UL FSRI 2018, and ATF's corridor flow.
- For velocity in doorways alone, there are two (NBS/NIST and UL FSRI).
- Door mass flow is measured only by NIST/NRC.

Harness work:
- generalise the door device to any opening and orientation;
- add a plane mass-flow device;
- write reducers for each format.

### 3.5 Neutral plane

The neutral plane comes from a vertical velocity profile through a single opening that carries flow both ways. Candidates:
- Steckler (wired);
- NIST/NRC's door probes;
- UL/NIST vents' door;
- DelCo's arrays;
- Bryant (its PIV stops at two-thirds of the doorway, so the reversal is captured only where it lies low);
- UL FSRI 2018's doorway probes;
- UL/NIJ (terms).

WTC's openings each carry flow one way, so they have no neutral plane. ATF's corridor arrays show a counter-flow interface, not a doorway's neutral plane.

**Three or more independent laboratories: no.** Two are usable now: NBS/NIST and UL FSRI. A third holds only if ATF's corridor interface is accepted as the same quantity; it is not, as the FDS guide treats it.

### 3.6 Smoke obscuration, soot mass concentration, extinction and visibility

| Series | Laboratory | Measured | Soot yield known? | Usable? |
| --- | --- | --- | --- | --- |
| NIST/NRC (wired) | NIST | laser transmission (%/m) at one path; also `Smoke Conc.` (soot mass concentration) | yes: heptane 0.0149, measured in the tests | yes |
| NIST smoke alarms | NIST | smoke meters at stations A to F, 2 heights (`SMA_1` to `SMF_4`); detector signals | no: furniture fire; the FDS input's soot follows from its assumed reaction, and its heat release is a ramp of an estimate | weak: neither the fire nor the soot is measured in the test |
| NIST FSE 2008 | NIST | soot mass fraction front and rear (`YsootF`, `YsootR`) | heptane 0.0149 for well-ventilated burning; under-ventilated burning differs | weak: under-ventilated |
| FAA cargo | FAA | light transmission beams, forward, mid and aft, high, mid and low (`LT_*`; FDS `PATH OBSCURATION`) | 0.126, from cone calorimetry (FDS input) | not at the app's cells: 5 kW in 1.4 m |
| FM/FPRF data centre | FM Global for FPRF | obscuration at subfloor, hot aisle and ceiling | in the FDS inputs (propylene and others) | not now: mechanical airflow and perforated tiles; terms to ask |

The UL FSRI 2018 data have no smoke measurement.

**Three or more independent laboratories: no; there is not even a second usable one.** Visibility can be compared now only with NIST/NRC. The other two series in the repository are NIST's own and weak. Others outside the repository (UL FSRI residential, Kerber's living rooms, Dalmarnock) are not public with reuse terms (2026-11-08 plan, section 7). Smoke and visibility therefore cannot be calibrated from public data; at most they can be reported.

### 3.7 Gas temperature at head height (1.5 to 2.0 m)

The wired series' trees have thermocouples in the band:
- Steckler: the room tree's 1.54, 1.66 and 1.77 m;
- NBS: 62 and 74 inches (1.57 and 1.88 m);
- NIST/NRC: each tree's 1.75 m, with 1.40 and 2.10 m either side;
- WTC: 1.53 and 1.83 m;
- ATF: 1.76 m in the lower corridor trees; the upper trees from their own floor.

UL FSRI 2018's trees have 1.5 and 1.8 m above the floor. Others: UL/NIST vents, DelCo and UL/NIJ.

**Three or more independent laboratories: yes** (NBS/NIST, ATF, UL FSRI 2018). Five series are already run, so the work is a reducer that reads point thermocouples (a rise at each, largest over the window, σ̃E 0.07); no new runs are needed for those five.

### 3.8 Wall and target heat flux

| Series | Laboratory | Measured |
| --- | --- | --- |
| NIST/NRC | NIST | wall gauges, and target (cable and gauge) fluxes (the guide compares 306 surface and 103 target) |
| WTC | NIST | surface and target gauges (36 and 44) |
| NIST FSE 2008, DelCo, E119 | NIST | floor, ceiling and wall gauges |
| ATF corridors | ATF | wall radiative and total, hall total (`Heat_Flux_*`) |
| UL FSRI 2018 | UL FSRI | side wall total and radiative at 2 heights; the plume's flux on the adjacent wall |
| Hamins gas burners, NIST structure separation | NIST | open fires: gauges around burners |
| Fleury (Canterbury), Lattimer corridor ceiling (Virginia Tech) | other universities | open or semi-open burners; terms to ask |

**Three or more independent laboratories: yes in the data** (NIST, ATF, UL FSRI 2018). **But the solver has no heat flux device.** Its radiation is traced from emitting cells and absorbed in solids, so a gauge needs a new tally: the radiant energy arriving at a wall face per unit area and time, plus the convective exchange. This is a solver change, the heaviest item here.

### 3.9 Sprinkler and detector activation

- **Sprinklers:** Vettori flat and sloped (NIST), Bittern (Canterbury) and UL/NFPRF (UL) are the only public activation series in the repository. All are already used by `tools/sprinkler-validation.mjs`, and the walls' convection coefficient was set on them, so none is held out (claim protocol, section 4).
- **Detectors:** NIST's smoke alarm tests are the one series: ionization, photoelectric and heat alarms in a manufactured home, furniture fires. The guide compares 142 activations.

**Three or more held-out series: no, for either.** No public sprinkler or detector series outside these is in the repository.

## 4. The three series skipped on 0.20 m cells

Estimated from their 0.27 m runs on 6ee8c93, by the ratio of cells times 1.35 for the finer steps:

| Series | Cells on 0.20 m | Median min per test | Longest | Tests | CPU min |
| --- | --- | --- | --- | --- | --- |
| NIST/NRC | 92,796 | 116 | 164 | 10 | 1,052 |
| WTC | 20,680 | 42 | 43 | 6 | 232 |
| ATF corridors | 77,517 | 29 | 39 | 6 | 158 |

- **Together:** about 1,440 CPU min, about 4 h of wall time at 6 jobs, its length set by NIST/NRC's longest test (about 2.7 h).
- **Earlier timings:** 60 s runs on the old solver measured 65 min for NIST/NRC test 13 and 61 min for WTC test 4 on these cells (plan section 12.1), so the WTC figure may be low.

## 5. Corrections to earlier text

Plan section 12.9 says NIST's smoke alarm tests "give activation times only". They also record smoke meters (`SMA` to `SMF`) and gas analysers. They remain weak for smoke because neither their fire nor their soot yield was measured in the test.

## 6. Sources

- NIST, `firemodels/exp` at 704e0d9: the folders named above (NIST's notice, `LICENSE.md`).
- NIST, `firemodels/fds` at 193faff: FDS Validation Guide 6.11.1 sources (`Manuals/FDS_Validation_Guide/Experiment_Chapter.tex`, `Utilities/Python/FDS_validation_dataplot_inputs.csv`, `FDS_validation_scatterplot_inputs.csv`), and `Validation/<series>/FDS_Input_Files`.
- UL FSRI (2021), 2018 Compartment Fire Experimental Data, `github.com/ulfsri/fsri-compartments-2018` (MIT licence; doi:10.5281/zenodo.5703474, as checked for the 2026-11-08 plan): the `01_Data` and `02_Info` READMEs.
- OECD/NEA, PRISME and PRISME-2 data packages, catalogue entries CSNI2006/03 and CSNI2042 (`oecd-nea.org/tools/abstract/detail/csni2042`; `oecd-nea.org/jointproj/prisme.html`), read through search results; the pages returned HTTP 403 to scripted fetches.
