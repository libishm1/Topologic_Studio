# Measured data for checking the sprinkler model: what was extracted, and which tests to run first (2026-11-07)

Status: data gathering only. No app code was changed and nothing was committed. This note follows section 8 ("Validation data") of `sprinkler-physics-2026-11-07.md` and reports what was pulled out of each data set listed there, in a form the Flamescapes sprinkler model (`topologicpy-web-frontend/src/core/suppression.js`: RTI, C, activation temperature, head position, ceiling height, heat release) can be set up from.

The numbers are in one JSON file, `sprinkler-validation.json`, kept in the session scratchpad (`.../scratchpad/validation-data/sprinkler-validation.json`, 0.75 MB) with the downloads it was built from and the scripts in `.../scratchpad/vd-scripts/` (`parse_fds.py`, `crossref.py`, `build_json.py`, `energy_check.py`, `link_replay.py`). It is not in the repository. If it is to be kept, it should be copied in deliberately.

How the sources were read. The NIST reports were downloaded from nvlpubs.nist.gov. NISTIR 6253 has no text layer and was read as page images. NISTIR 5254, 4833 and NBSIR 88-3696 are Internet Archive scans with an OCR layer; their tables were read as images and two blotted digits were checked at 300 dpi. NISTIR 7079, 6196-1 and 7506 have text layers. The FDS validation inputs and the Validation Guide were read from github.com/firemodels/fds at commit b7dc77f (6 Nov 2026), the digitised measurements from github.com/firemodels/exp at 704e0d9, and the CFAST Vettori inputs from firemodels/cfast master. The Bittern thesis could not be opened (HTTP 403 from both the Canterbury repository and CORE). Everything downloaded was treated as data.

---

## 1. Summary

| Data set | Tests extracted | What it gives the solver | Licence |
|---|---|---|---|
| Vettori flat ceiling, NISTIR 6253 (1998) | all 45: 27 smooth, 18 joisted | room, burner positions, two HRR ramps, RTI 55, C 0.25 (measured by UL), 68 °C, four head positions; activation of up to four heads per test, gas temperature at the first head at activation, ceiling-jet speed at heads 1 and 2, layer temperatures at 1.54 m and 1.89 m | NIST, public domain |
| Vettori sloped ceilings, NISTIR 7079 (2003) | all 72 (36 configurations, 2 replicates); flat smooth 1 to 12 and flat with beams 13 to 24 are the main subsets | as above for a second room with an open door, ceilings flat, 13° and 24°, smooth or two beams; gas temperature at every activated head from 1 s data | NIST, public domain |
| Bittern 2004 (via the FDS Validation Guide) | 21 (test 11 excluded by the Guide) | chair HRR curve per test (from mass loss), two non-flowing heads of four types (RTI 36 or 95, C 0.4, 68 or 93 °C), door open or closed, centre or corner fire; activation of both heads | numbers from NIST repositories; thesis unread, licence unknown |
| UL/NFPRF Series I, NISTIR 6196-1 (1998) | all 22 (I-11 has no sprinkler data) | 7.6 m ceiling, 49 flowing heads (RTI 148, C 0.7, 74 °C, K 11.4 gpm/psi^0.5 at 131 kPa, 0.34 mm/s), heptane spray burner Q = 10 MW (t/75 s)^2 to a 3.7 to 6.0 MW plateau; first activation, number of activations, times of the first four heads, activation sequence | NIST, public domain |
| Madrzykowski and Vettori 1992 (NISTIR 4833), Walton 1988 (NBSIR 88-3696), Evans 1993 (NISTIR 5254) | Evans Table 1 (8 furnishing packages) and Table 2 (14 crib runs) | heat release at activation, decay constant, spray density | NIST/NBS, public domain |
| Madrzykowski 2008, NISTIR 7506 | the one sprinklered room test and its unsprinklered twin | 3.45 x 3.63 x 2.44 m room, dry tree, one residential head at 34.1 L/min; activation at 10 s, peak HRR under 1.8 MW against more than 6 MW | NIST, public domain |

Three findings change how the data should be used.

1. The two heat release curves for the Vettori burner disagree by a factor of 2 to 2.5 at the activation times. NISTIR 6253 and 7079 describe nominal t-squared fires (alpha 0.0468, 0.0117 and 0.00293 kW/s², 1055 kW at 150, 300 and 600 s) and show calibration runs that follow them. The FDS and CFAST validation inputs use a table that reaches 1053 kW at 100 s on the fast curve, which the FDS Validation Guide calls the actual heat release rate. An energy check settles which to trust: at the first activation of the nine smooth detached tests, the gas in the top 0.9 m of the room already holds 2.5 to 3.0 MJ above ambient (from the measured thermocouple arrays), while the nominal curve has released only 1.0 to 2.7 MJ by then and the tabulated one 3.8 to 6.8 MJ. The nominal curve cannot be what burned, at least not with the reported time origin (`energy_check.py`; my calculation). Use the tabulated ramp, and keep the nominal one only as a lower bracket.
2. The Vettori 1998 heads have a measured C of 0.25 (m/s)^0.5. NISTIR 6253 Appendix A reproduces UL's plunge tests: all three samples operated at zero air speed in the 128 °C prolonged plunge, so UL assigned C = 0.25 under the ISO 6182-1 rule, and the dynamic plunge gave RTI 55.8 (frame arms across the flow) and 64.3 (25° offset). The FDS validation inputs leave C at its default of 0. NISTIR 7079 does not report C for its heads.
3. The Bittern C of 0.4 and the 2 cm link offset are modelling choices made in the original study by sensitivity analysis, not measurements (FDS Validation Guide). The thesis abstract says C moved the FDS times by up to 50 %. So Bittern is a weak test of C and a good test of RTI.

The tests I would run first on a 0.3 m grid are listed in Section 9.

---

## 2. The JSON file

One object per data set under `datasets`, each with `enclosure`, `fire`, `sprinkler` (or `sprinkler_types`), `tests`, `subsets` where useful, and `gaps`. Every leaf is `{value, source}`, sometimes with an extra field (`alt`, `tolerance_C`, `suspect`, `wetting_drop_within_4s_C`). Units are in the field names: `_m`, `_s`, `_C`, `_kW`, `_m05_s05` for (m s)^0.5, `_m05_per_s05` for (m/s)^0.5, `_mm_per_s` for L/(s m²). A source string names the document and table, figure or file; "derived" means computed by `build_json.py` from the named inputs (radial distances, heat release at activation, burner centres). A script check found no value without a source.

Each data set states its coordinate frame. Where positions come from an FDS input, the FDS frame is used. The Bittern FDS input is the Validation Guide's sketch rotated by 180°; the JSON uses the room frame (FDS minus the 0.1 m wall offset) and says so.

Top-level `sources` holds the citation, URL, what was read, the licence and the Crossref result for each source. `best_first_checks_for_0p3m_grid` repeats Section 9.

---

## 3. Vettori flat ceiling (NISTIR 6253, 1998)

What it is. Forty-five non-flowing tests at NIST Building 534 on how a joisted ceiling delays a quick-response residential pendent head. A methane burner followed fast, medium or slow t-squared growth and was placed detached, against a wall or in a corner. Smooth ceiling: 3 replicates of each of 9 configurations. Joisted ceiling: 2 replicates of each.

Set-up extracted.
- Room 9.2 x 5.6 m. Ceiling 2.44 m (8 ft); the report rounds it to 2.4 m, but its "1.54 m elevation" thermocouple is the one 0.9 m below the ceiling, and the FDS mesh top is 2.44 m.
- Door 0.91 x 2.1 m, closed in every test, with a 25 mm gap under it. A 0.9 x 2.7 m stair opening in one corner of the ceiling leads to a closed upper floor of the same plan. The FDS inputs only include that opening in the joisted runs (it lies outside the smooth runs' mesh); CFAST models it as a 1.56 m² ceiling vent to outside.
- A 0.10 x 0.18 m steel I-beam runs the length of the room at mid-width (y = 2.72 to 2.80 m, 0.18 m deep below the joists), in both ceiling types. It separates heads 1 and 2 (burner side) from heads 3 and 4.
- Joisted ceiling: 0.038 x 0.24 m joists at 0.41 m centres spanning the width, plywood deck at 2.64 m (report: effective height 2.7 m). Heads 37 mm below the joists, 0.27 m below the deck.
- Burner 0.7 x 1.0 x 0.31 m, methane at 98 % or better, top 2.1 m below the ceiling. Positions (FDS blocks): detached x 4.30 to 5.00, y 1.00 to 2.00; wall x 4.30 to 5.00, y 0 to 1.00 (short side on the wall); corner x 0 to 0.70, y 0 to 1.00. FDS burns 2930 kW/m² on a 0.6 x 0.6 m patch of the block.
- Heads at (2.4, 1.4), (6.8, 1.4), (6.8, 4.2), (2.4, 4.2) m, bulb centre 25 mm below the gypsum. RTI 55, C 0.25 (Section 1), 68 ± 2.4 °C, 3 mm bulb. Dry pipe with 25 ml of water, air at about 100 kPa, pressure switch timing to ±0.1 s.
- Radial distances from the burner centre: about 2.2 to 2.4 m for the nearest head (detached and wall); corner 2.24 m to head 1 and 4.23 m to head 4.
- Ambient temperature per test (16.6 to 28.5 °C) from the CFAST inputs.

Measured, per test.
- Activation time of every head that opened, from the digitised Appendix B (exp `Vettori_Flat_Ceiling_Act.csv`), checked against Table 1 for the first head. Smooth, first head: detached fast 41 to 46 s, medium 67 to 88 s, slow 100 to 117 s; wall 39 to 42, 68 to 71 and 116 to 129 s; corner 29 to 35, 47 to 54 and 84 to 99 s. Joisted: 67 to 233 s, 57 to 137 % later (abstract).
- Gas temperature at the first head at activation (Table 3): 93 to 143 °C smooth, 125 to 165 °C joisted (for test 44 the value is from the thermocouple 25 mm lower). For the other heads, interpolated from the 25 mm thermocouple in Appendix B, which is sampled about every 7 s.
- Ceiling-jet speed 25 mm below the ceiling at heads 1 and 2, interpolated to activation: roughly 0.8 to 2.7 m/s smooth (corner fires fastest) and 0.6 to 1.05 m/s joisted. Probe uncertainty ±0.1 m/s.
- Mean temperature of the four arrays at 1.54 m and 1.89 m elevation at activation (Tables 4 and 5).
- Heat release at activation, from both ramps (derived).

Burner shut-off after the first activation is not stated in NISTIR 6253; in NISTIR 7079 it took 2 to 4 s. Second and later heads may therefore have opened after shut-off.

A side check (`link_replay.py`, my calculation). I fed the measured gas temperature and speed at heads 1 and 2 of the 27 smooth tests through the link equation of `suppression.js` (gas held at the mean of each interval, exact step) with RTI 55, C 0.25 and 68 °C, with no solver involved. Where the record lasts long enough to tell (18 head records), predicted over measured activation has a median of 1.00 and a range of 0.81 to 1.18. In 18 other records the link had not reached 68 °C when the record ends, i.e. the replay is late, by more than 9 to 14 s in the fast detached tests. The 7 s sampling and the thermocouple's own lag both under-read a fast-rising gas, so this says the parameters are about right for medium and slow fires and cannot speak for fast ones.

---

## 4. Vettori sloped and beamed ceilings (NISTIR 7079, 2003)

What it is. Seventy-two non-flowing tests in a second room at NIST Building 205: flat, 13° and 24° ceilings, smooth or with two beams, fast or slow growth, burner detached, against the rear wall or in a corner, two replicates each. The test-to-configuration table, the per-head times and the ambient temperatures are in exp `Vettori_Sloped_Guide.csv`, which agrees with the report's Tables 6 to 10.

Set-up extracted.
- Room 7.3 x 5.5 m, flat ceiling 2.45 m (Fig. 5; the text says 2.4 m). Door 1.0 x 2.1 m in the front wall, open.
- Slope: the front edge is raised 1.25 m (13°, front wall 3.70 m) or 2.5 m (24°, 4.95 m). The heads keep their plan positions and follow the ceiling: heads 1 and 2 at z = 2.425, 3.375 or 4.325 m; heads 3 and 4 at 2.425, 2.695 or 2.975 m (FDS inputs; report Tables 1 and 2).
- Beams: two, 0.20 m wide and 0.25 m deep, front to rear at x = 2.3 to 2.5 and 4.8 to 5.0 m, making three 2.3 m bays.
- Burner as in 1998 (0.7 x 1.0 x 0.3 m, methane); FDS uses the same fast and slow tables. Positions: detached x 3.3 to 4.0, y 2.25 to 3.25; rear wall x 3.3 to 4.0, y 4.5 to 5.5; front-right corner x 6.6 to 7.3, y 0 to 1.0.
- Heads at (1.85, 1.25), (5.5, 1.25), (5.5, 4.25), (1.85, 4.25) m, bulb 25 mm below the ceiling. RTI 56, 68 ± 2.4 °C. C not reported.
- Gas was shut off at the first activation, which took 2 to 4 s.

Measured, per test: activation time of each head that opened; the 25 mm thermocouple at each activated head at the moment of activation (1 s data, time measured from ignition). In 119 of the 126 head records the thermocouple drops by more than 10 °C within 4 s of activation, as the pipe water hits it, which confirms the time base. The other seven are flagged `suspect` in the JSON (for example test 4 head 2 reads 44.5 °C at activation). Ceiling-jet speeds were measured at all four heads but are not in the exp files. Uncertainties (Table 4, k = 2): activation 16 %, temperature 23 %, speed 25 %, heat release 24 %.

First activation in the flat subsets:

| Burner | Smooth, fast | Smooth, slow | Two beams, fast | Two beams, slow |
|---|---|---|---|---|
| Detached | 60 to 71 s | 206 to 211 s | 76 to 79 s | 215 to 221 s |
| Rear wall | 58 to 59 s | 162 s | 72 to 79 s | 174 to 211 s |
| Corner | 35 to 36 s | 102 to 104 s | 35 s | 92 to 103 s |

---

## 5. Bittern 2004 chair fires

What it is. Twenty-two tests at the University of Canterbury: one upholstered polyurethane chair burned in a gypsum-lined room with two heads that were not charged with water; a pressure gauge upstream of each head showed activation. Test 11 is excluded by the FDS Validation Guide (failed mass-loss measurement). The thesis (Bittern 2004, "Analysis of FDS predicted sprinkler activation times with experiments") could not be opened, so everything here comes through the Validation Guide (section and two tables), the 21 FDS inputs and exp `bittern_acts.csv`.

Set-up extracted.
- Room 8.0 x 4.0 x 2.4 m, flat smooth ceiling, 10 mm gypsum on timber framing, concrete floor.
- Door 0.8 x 2.1 m: open in tests 1 to 10, closed in 12 to 22. For the closed tests the FDS input spreads an assumed leakage of 0.053 m² over the door.
- Chair top face 0.4 x 0.5 m, 0.65 m above the floor (so 1.75 m below the ceiling). Centre position (4.0, 2.0) m, or corner position (7.8, 0.25) m, in the room frame.
- Heads 1 and 2 at (6.0, 2.0) and (2.0, 2.0) m. That puts both 2.0 m from the centre chair, and 2.51 m (head 1) and 6.06 m (head 2) from the corner chair. FDS puts the links 2 cm below the ceiling.
- Head types: Res A and Res B, residential 3 mm bulb, RTI 36, 68 °C; SS68, standard 5 mm bulb, RTI 95, 68 °C; SS93, RTI 95, 93 °C. C 0.4 for all.
- Heat release: the FDS ramp times the per-test HRRPUA (249 to 993 kW/m²) on 0.2 m². It was estimated from measured chair mass loss times the heat of combustion (21.9 MJ/kg in tests 1 to 10, 20.4 MJ/kg in 12 to 22). The full curve is in the JSON for each test; peaks are 50 to 199 kW. Ambient 20.6 to 29.4 °C.

Measured: activation of both heads (178 to 330 s; test 20 head 2 did not activate). There are no gas temperatures, speeds or layer data in the public files.

Caution. The heat release at activation read off these curves is low in several closed-door corner tests: 18 kW (test 16), 41 kW (17), 44 kW (20) and 75 kW (22) at the near head. A 68 °C residential bulb opening 2.5 m from an 18 kW fire seems hard to explain, even with a corner plume and a closed room. The mass-loss heat release may lag or under-read early burning, or the time origins of the curve and the activation times may differ. I could not resolve this without the thesis.

---

## 6. UL/NFPRF Series I (NISTIR 6196-1, 1998)

What it is. Twenty-two heptane spray burner tests in the UL Large Scale Fire Test Facility (1997). They studied sprinklers, a roof vent and draft curtains together.

Set-up extracted.
- Ceiling 30.5 x 30.5 m at 7.6 m in a 37 x 37 m cell, 16 mm ceiling tiles (properties in the JSON).
- Draft curtains 1.8 m deep, enclosing about 450 m², in I-1 to I-16; none in I-17 to I-22.
- One 1.2 x 2.4 m vent, closed, opened manually at 0:40 or 1:30, or on a 74 °C or 100 °C link. Facility exhaust at least 11 m³/s.
- Water-filled 2 in branch lines with their centreline 0.22 m below the ceiling.
- Burner: nozzles on a 1 x 1 m pipe square 0.6 m above the floor (FDS: 0.5 m). Heat release followed Q = 10 MW (t/75 s)^2, about 1.78 kW/s², by hand until a set size or the first activation, then held. The plateau per test is in Table 1 (4.4 MW at 50 s in most; 3.7 to 6.0 MW; I-16 5.0 MW at 110 s with tau 150 s). The FDS form Q = Qp (t/tp)^2 then Qp is recorded. Convective fraction 0.65.
- Burner positions A to D, each at the centre of a square of four heads, 2.15 m from each.
- Heads: Central ELO-231 upright; RTI 148, C 0.7 (UL); 74 °C; thermal element 0.11 m below the ceiling (deflector 0.08 m; FDS uses 0.14 m). 49 heads on a 3.05 m square grid, flowing. K 11.4 gpm/psi^0.5 (about 164 L/min/bar^0.5) at 131 kPa (19 psi), which is 189 L/min per head and 0.34 mm/s. All 49 head positions are in the JSON (FDS frame).

Measured, per test:
- first activation and number activated (Table 1). First activation: 54 to 74 s, and 106 s in I-16. Number activated: 4 to 14.
- activation times of the first four heads (exp `UL_NFPRF_Act_Times.csv`).
- the activation sequence, from the running count in exp `UL_NFPRF_Actuated_Sprinklers.csv` (1 s rows).
- the thermocouple next to each of the four heads just before its wetting drop: 108 to 195 °C, with two readings flagged because a neighbour's spray had already cooled that thermocouple.

Activation was detected by that wetting drop, at a 2 s scan. The exp CSV's final count differs from Table 1 in I-9 (13 against 12), I-12 (13 against 14), I-15 (7 against 5) and I-17 (6 against 4); both are kept. Times of heads beyond the first four are only drawn on the report's plan figures (Figs. 4 to 11) and were not transcribed.

---

## 7. Heat release after activation: Madrzykowski and Vettori 1992, Walton 1988, Evans 1993

What they are.
- Madrzykowski and Vettori (NISTIR 4833) burned office furnishing packages under an open calorimeter. A commercial pendent head, with its deflector 2.35 m above the platform and no ceiling, was switched on by hand before the free-burn peak. It flowed 25 gpm (1.6 x 10^-3 m³/s), giving 0.07 mm/s on average over the fuel and 0.035 to 0.13 mm/s locally. Their bound is Q(t) <= Q_act exp(-0.0023 (t - t_act)), i.e. tau = 435 s.
- Walton (NBSIR 88-3696) burned fir cribs under the same kind of head, deflector 2.43 m up, no ceiling. The cribs were 0.61 x 0.61 m, 38 mm sticks, 6 per layer, 8 layers (0.30 m, about 240 kW free burn) or 16 layers (0.61 m, about 460 kW). Water was applied about 330 s after ignition, at densities measured in four 0.36 m pans before and after each test.
- Evans (NISTIR 5254) fitted decay constants to both and gave tau = 3.0 w^-1.85 s for furnishings (410 s at 0.07 mm/s).

Extracted:
- Evans Table 1. Eight furnishing packages at 0.07 mm/s: heat release at activation 280 to 1365 kW and decay constants 9.1 s (sofa) to 713 s (workstation I). The two-test rows (office I, workstation I, the 0.61 m crib) keep both heat release values. The workstation II value is blotted in the scan; NISTIR 4833 section 4.6 confirms 700 kW.
- Evans Table 2. Fourteen crib runs at 0.026 to 0.133 mm/s: heat release at activation 181 to 439 kW, and decay constants 44 to 543 s with Evans's bracketing range.
- Walton's own pan-averaged densities (Table 1, mm/min) are stored beside each crib row. They match Evans's mm/s values (for example 2.36 mm/min = 0.039 mm/s; that digit is smudged in the Evans scan).
- Activation times and the shielded secretarial desk (550 kW at 240 s), which Evans left out, are kept from NISTIR 4833.

One point for the code. `suppression.js` treats 0.026 mm/s as the density below which "Evans found no measurable effect". In the source, the runs Evans dropped for having no measurable effect were at 0.013 mm/s (0.30 m crib) and 0.014 and 0.039 mm/s (0.61 m crib); Evans prints the second as "0.14", but Walton's table shows 0.86 mm/min. So 0.026 mm/s is the lowest density Evans used, not a measured threshold. For deep fuel even 0.039 mm/s did not register. This is an observation, not a requested change.

---

## 8. Madrzykowski 2008 (NISTIR 7506): Christmas tree room

What it is. Two room fires, one with a single residential pendent head, each with a dry Fraser fir and the same furnishings. The heat release was measured at a hood over the open door.

Extracted.
- Room 3.45 x 3.63 m, ceiling 2.44 m; two layers of 12.7 mm gypsum; open door 0.90 x 2.03 m.
- Tree 2.40 m tall, 1.50 m wide, 14.64 kg, needle moisture 6 to 9 %, in the rear-left corner.
- Head 1.73 m from the right wall and 1.83 m from the rear wall, about 1.55 m from the tree's axis (my reading of Fig. 8). 68 °C; RTI and C not given. Flowing 34.1 L/min (the listed minimum at 58.1 kPa for 3.7 x 3.7 m, about 0.042 mm/s over that area).
- Measured:
  - activation at 10 s, with the thermocouple beside the head at about 500 °C just before;
  - heat release about 1 MW within 10 s (summary);
  - sprinklered peak under 1.8 MW (about 1.7 MW at 60 s, read from Fig. 19), no flashover;
  - unsprinklered peak above 6 MW at about 70 s, flashover at about 60 s.

The heat release kept rising for 50 s after activation, so the Evans cap, which assumes the fire does not grow after activation, is the wrong model for this fuel. That makes it a test of the spray stage and of the "does one head prevent flashover" question, not of Evans.

---

## 9. Best first checks for a 0.3 m grid

The criteria:
- the geometry must be resolvable at 0.29 to 0.4 m;
- the plume should be away from walls, since wall and corner plumes are what a coarse grid does worst;
- activation should come late enough that start-up transients do not dominate;
- there should be replicates;
- the gas temperature and speed should be measured at the head, so the reading itself can be checked and not only the time.

1. Vettori flat smooth, detached burner, medium and slow growth: tests 6, 7, 8, 11, 12, 13.
   - Ceiling 2.44 m is 8 cells; the burner is 2 by 3 cells; the heads are 2.2 m out (7 cells).
   - First activation comes at 67 to 117 s, with three replicates of each configuration.
   - Gas temperature and speed are measured at heads 1 and 2.
   - Run both heat release ramps; Section 1 says the tabulated one is the one to believe.
   - Model the door as closed with a 25 mm slot, and the stair as an opening into a closed volume of the room's plan.
   - Compare the "estimate" and "late" readings of `suppression.js` with the measured times and with the measured 93 to 118 °C and 1.1 to 1.8 m/s at the head.
2. Vettori sloped set, flat subsets: tests 1 to 4 (smooth, detached) and 13 to 16 (two beams, detached).
   - The same burner and heads in a second room with an open door.
   - The beams are one cell deep and 2.3 m apart, so they are resolvable; the 1998 joists (0.24 m deep at 0.41 m centres) are not, and I would not expect the coarse solver to reproduce that subset.
   - The 13° and 24° tests are the next step, for sloped IFC roofs.
3. Bittern, tests 3, 4, 8 and 10 (door open, centre chair) and 12 to 15 (door closed, centre chair).
   - A real chair heat release curve, and two heads at 2.0 m from it.
   - Tests 12 to 15 put an SS68 (RTI 95) and a Res B (RTI 36) head in the same fire at the same distance. The measured gap of 10 to 18 s checks the link ODE's RTI scaling with the gas history shared.
   - Treat C = 0.4 as a guess.
4. UL/NFPRF, tests I-17, I-18, I-20 and I-22 (no curtains, 4 to 6 activations).
   - First activation at 54 to 60 s under a 7.6 m ceiling is the regime Alpert's correlation was built for, so the "estimate" reading should do well here.
   - The number of activations tests the spray's cooling of the ceiling jet once the spray stage is on.
   - The curtained domain is about 20 x 22 x 9 m (about 63k cells at 0.4 m); a run without curtains needs more of the 30.5 m ceiling.
5. Evans Tables 1 and 2: a zero-dimensional check that the per-cell cap reproduces the decay constants it was fitted to, before any grid run.

Corner fires (Vettori and Bittern) and the Christmas tree room come later. The corner plume and the tree's vertical flame spread are what the coarse grid is least likely to get right.

---

## 10. Licences

- NIST and NBS reports (NISTIR 6253, 7079, 6196-1, 4833, 5254, 7506; NBSIR 88-3696) are US Government works, not subject to copyright in the United States (17 U.S.C. 105). NISTIR 6253 Appendix A reproduces a UL test report (UL project 96NK9527); only numbers were taken from it.
- github.com/firemodels/exp, /fds and /cfast all carry the same LICENSE.md: NIST-developed material is not subject to copyright in the United States; it may be used, copied, modified and distributed if the notice is kept and NIST acknowledged; it is provided as is. GitHub reports it as "Other" (NOASSERTION).
- The Bittern thesis (University of Canterbury) has an unknown licence; it could not be opened. Wade et al. 2007 (Springer) and Hopkin et al. 2018 (SAGE) are copyrighted and were not read. Only their metadata is used.
- No text beyond short factual values was copied from any source.

---

## 11. Gaps, and the numbers I am least sure of

- The origin of the Vettori tabulated ramp. The energy check shows the nominal curve is too small, but I do not know whether the reported time origin is late (the burner ramp starting before the logged ignition) or the burner over-delivered. The tabulated ramp's shape is therefore uncertain even though its size is plausible.
- The Bittern heat release at activation in the closed-door corner tests (Section 5). The thesis would settle it, but it returned 403. The Bittern C (0.4) and link offset (2 cm) are modelling choices.
- Two scanned digits in Evans Table 2. The 0.039 mm/s density is confirmed by Walton. The heat release at activation of the 0.041 mm/s run reads 232 kW in the image but 242 in the OCR layer; 232 is kept and flagged.
- NISTIR 7506 values read from figures: the head position (from Fig. 8) and the peak heat release values (from Fig. 19) are by eye, to perhaps ±0.05 m and ±100 kW.
- The ceiling-jet speeds at activation in the Vettori flat set are interpolated over about 7 s gaps in a fluctuating signal; read them to about ±0.3 m/s.
- The UL ambient temperature (18 °C) comes from the FDS inputs and was not found in the report.
- The UL first-four-heads gas readings are thermocouple values near the head, possibly cooled by a neighbour's spray. Two are flagged as below the 74 °C rating.
- Not extracted:
  - UL per-head times beyond the first four;
  - UL Series II (10 MW) and the plastic commodity tests;
  - the Vettori sloped ceiling-jet speeds (not in the exp files);
  - the furnishing and crib heat release curves (plotted only).

---

## 12. DOI checks

Each DOI was queried at api.crossref.org/works/<doi> on 2026-11-07.

| DOI | Resolved | Crossref title; year |
|---|---|---|
| 10.6028/NIST.IR.6253 | yes | Effect of an obstructed ceiling on activation time of a residential sprinkler; 1998 |
| 10.6028/NIST.IR.7079 | yes | Effect of beamed, sloped, and sloped beamed ceilings on the activation time of a residential sprinkler; 2003 |
| 10.1007/s10694-007-0009-5 | yes | Assessing the Sprinkler Activation Predictive Capability of the BRANZFIRE Fire Model; 2007 (Wade, Spearpoint, Bittern, Tsai) |
| 10.1177/0734904118772306 | yes | Using experimental sprinkler actuation times to assess the performance of Fire Dynamics Simulator; 2018 (Hopkin, Spearpoint, Bittern) |
| 10.6028/NIST.IR.6196-1 | yes | Sprinkler, smoke & heat vent, draft curtain interaction; 1998 |
| 10.6028/NIST.IR.4833 | yes | A sprinkler fire suppression algorithm for the GSA engineering fire assessment system; 1992 |
| 10.1177/104239159200400403 | yes | A Sprinkler Fire Suppression Algorithm; 1992 (J. Fire Prot. Eng.) |
| 10.6028/NIST.IR.5254 | yes | Sprinkler fire suppression algorithm for HAZARD; 1993 (author listed as NIST) |
| 10.6028/NBS.IR.88-3696 | yes | Suppression of wood crib fires with sprinkler sprays; 1988 |
| 10.6028/NIST.IR.7506 | yes | Impact of a residential sprinkler on the heat release rate of a Christmas tree fire; 2008 |
| 10.6028/NIST.SP.1018 | yes, but to the FDS version 4 record (2006) | the current Validation Guide was read from its LaTeX source instead |
| 10.6028/NIST.SP.1018-3, 10.6028/NIST.IR.6196-2 | no (404) | tried as guesses for the Validation Guide volume and the companion NISTIR; neither exists at Crossref |

The Bittern 2004 report has no DOI. Its repository handle is ir.canterbury.ac.nz/handle/10092/14748.

---

## Results (2026-11-07)

Status: the checks of section 9 were run on the CPU solver; nothing was committed. Code added: a prescribed burner in `topologicpy-web-frontend/src/core/flamescapes.js` (`createBurner`, `burnerEnergy`, `burnerStep`, `METHANE`), the burner's plume for the sprinklers' estimate and an opt-in `sensorGas` frame field in `src/core/flamescapesRun.js`, the harness `tools/sprinkler-validation.mjs`, and the tests `src/core/flamescapes.burner.test.mjs` and `src/core/suppression.validation.test.mjs`. Data copied into the repository, each file with the NIST credit and the firemodels licence notice: `benchmark/validation/sprinkler-tests.json` (the chosen tests, 152 KB) and `src/core/suppression.validation.json` (the replay records and Evans's tables, 28 KB). Results: `benchmark/results/2026-11-07-sprinkler-validation.json` (105 runs).

### R1. What was run

The burner. The solver's `pilot` stage heats the first item's surface and does not prescribe a heat release, so a burner was added. Each step it releases the exact integral of a linear heat release table over the step. The convective part goes into the gas of a column of cells over the burner area, with the products of the matching mass of fuel (oxygen at 417 kJ per mol O2, as the solver's chemistry), the fuel entering at ambient temperature, so the column's gas enthalpy above ambient rises by exactly the convective part. The radiant part goes along 16 rays, turned each step, to the first solid. The whole heat release counts in the frame heat release that the estimate reads. The burner products carry no soot, since the radiant fraction already takes the flame's radiation. A unit test checks that the gas gains the table's integral times the convective fraction within 1 % (it is within 1e-7), that the walls take the radiant part (within 0.7 %, the float32 rounding of small increments), and that each frame's heat release in a full run is the table's mean over the frame. With no burner the solver does one null check per step. On the 5 x 4 x 2.7 m test room (17,732 cells, 0.2 m) and the Duplex (27,216 cells, 0.37 m), the fields after 700 and 420 steps were bit-identical to the unchanged solver, and the median of the paired ratio of step times (changed over unchanged) was 1.002, 1.014 and 1.001 on the room and 0.988, 1.014 and 1.010 on the Duplex in three sessions, against a round-to-round spread of 0.73 to 1.60; the medians in the last session were 28.0 against 28.2 ms and 50.9 against 49.1 ms per step.

The scenes. Each test is a box built from the data: walls, floor and ceiling one cell thick, doors as the tests had them, the burner or chair as an inert block. The cell is 0.3 m adjusted so the ceiling lies on a cell face (0.305 m for the 1998 room, 0.306 m for the 2003 room, 0.300 m for Bittern) and 0.507 m for UL/NFPRF. What the grid could not resolve:
- the 25 mm gap under the 1998 room's closed door and Bittern's assumed 0.053 m² leakage: the closed rooms are sealed;
- the 0.18 m steel I-beam of the 1998 room and the 0.25 m beams of the 2003 room snap to whole cells, so the I-beam is 0.305 m deep and one of the 2003 beams fills two cells across;
- the flame: the burner's column is as tall as Heskestad's mean flame height at the heat release of the measured first activation, at least two cells (2 to 5 cells, 0.6 to 1.5 m, in the rooms; 10 or 11 cells, 5.1 to 5.6 m, in UL);
- UL's ceiling covers the FDS input's open-sided 20.4 x 21.6 m domain, without walls, as in FDS.
The 1998 stair opening leads into a closed upper floor of the room's plan, 2.44 m high (the report gives no height). Sloped and joisted ceilings and the wall and corner fires were not run.

The heads, at the measured positions: 1998 RTI 55, C 0.25, 68 °C; 2003 RTI 56, 68 °C and C taken as 0.25 (not reported; the 1998 heads' value); Bittern per type (RTI 36 or 95, C 0.4, 68 or 93 °C); UL RTI 148, C 0.7, 74 °C, flowing 189 L/min each in the solver's "spray" mode. The other runs use mode "activation", with no water. Both readings are recorded: the estimate (Alpert's jet for the frame's heat release over the grid's layer, for heads in sight of the plume) and the late reading (the top open cell under the head).

Fire curves. Vettori: the tabulated ramp of the FDS and CFAST inputs and the nominal t-squared curve, radiant fraction 0.2 (methane; not measured). Bittern: the per-test curve from mass loss, radiant fraction 0.46 (the FDS Validation Guide's modelling choice). UL: Q = Qp (t/tp)^2 to the plateau, radiant fraction 0.35 (measured convective fraction 0.65). Each curve holds its last value past its end.

Surface coefficient. Each run was made with the app's 200 W/(m² K) and with 25 W/(m² K), the natural convection value the solver's own comment names. These scenes hold no fuel, so the coefficient sets only the heat lost to walls, floor and ceiling.

Horizon. Twice the last measured activation in the test (UL at most 200 s); frames of 1 s, steps of 0.1 s. A head not open by then counts as not opened. In the 2003 set the gas was shut off at the first activation and the solver's burner kept burning, so second and later heads in that set are biased early in the runs.

### R2. The link on measured gas, without the solver

`suppression.validation.test.mjs` drives `linkStep` and `crossing` with the gas temperature and speed measured 25 mm below the ceiling at heads 1 and 2 of the 27 smooth 1998 tests, as `activationFrame` does for the late reading (the gas held at the mean of each interval's two readings, the speed floored at 0.1 m/s), with RTI 55, C 0.25 and 68 °C. It reproduces the replay of section 3. Of the 45 records with a measured activation, 18 open within the record, at 0.81 to 1.18 of the measured time (median 1.00). Another 18 run past the measured time without opening, by 0 to 31 % of it: mostly the fast fires and the wall and corner fires, where rows about 7 s apart and the thermocouple's lag under-read a fast-rising gas. The other 9 end before the measured time. The replay is never more than 19 % early. With C = 0, the FDS inputs' default, 25 records open, median 0.96, minimum 0.76; with RTI 64.3 (UL's 25° value), 10 open, median 1.04.

### R3. Evans's cap against Evans's data

The same test compares the cap's decay time at each test's spray density, 3.0 w^-1.85 s (`evansTau`, the time constant `evansFactor` applies), with the 23 decay constants of Evans's Tables 1 and 2. The cap decays more slowly than measured for 19. The exceptions: Workstation I (713 s against the cap's 411 s); the second run of the 0.61 m crib in Table 1 (560 s; the first was 375 s); and the 0.61 m cribs of Table 2 at 0.066 and 0.081 mm/s (478 and 326 s against 458 and 314 s), within Evans's own bracketing fits. The furnishing fit is the crib fit at a 0.61 m crib height, so those are runs it was fitted to. The 0.305 m cribs decayed 1.8 to 7.1 times faster than the cap. As a least reduction, the cap holds on these data except for Workstation I and one crib run.

### R4. The runs on the grid

Predicted over measured activation for each head that opened in the test, by the estimate. "Opened" counts heads open within twice the measured time; the median and range are over those; "early" counts predictions before the measured time; h is the surface coefficient in W/(m² K). The first-head column compares the first head to open in the run with the first measured.

| Data set | Fire curve | h | Heads measured | Opened | Median | Range | Early | First head per test: opened, median, range |
|---|---|---|---|---|---|---|---|---|
| Vettori 1998, smooth, detached (9 tests) | tabulated | 200 | 18 | 16 | 1.68 | 1.45 to 2.03 | 0 | 8 of 9, 1.68, 1.49 to 2.00 |
| | tabulated | 25 | 18 | 18 | 1.31 | 1.17 to 1.86 | 0 | 9 of 9, 1.29, 1.24 to 1.79 |
| | nominal | 200 | 18 | 5 | 1.91 | 1.86 to 2.06 | 0 | 3 of 9, 2.02, 1.89 to 2.06 |
| | nominal | 25 | 18 | 16 | 1.82 | 1.48 to 1.98 | 0 | 8 of 9, 1.89, 1.53 to 1.97 |
| Vettori 2003, flat smooth (tests 1 to 4) | tabulated | 200 | 12 | 12 | 1.09 | 0.93 to 1.17 | 2 | 4 of 4, 1.08, 0.94 to 1.10 |
| | tabulated | 25 | 12 | 12 | 0.86 | 0.81 to 0.96 | 12 | 4 of 4, 0.84, 0.82 to 0.96 |
| | nominal | 200 | 12 | 12 | 1.36 | 1.27 to 1.57 | 0 | 4 of 4, 1.33, 1.28 to 1.57 |
| | nominal | 25 | 12 | 12 | 1.12 | 1.06 to 1.37 | 0 | 4 of 4, 1.14, 1.06 to 1.37 |
| Vettori 2003, flat, two beams (tests 13 to 16) | tabulated | 200 | 9 | 0 | - | - | 0 | 0 of 4 |
| | tabulated | 25 | 9 | 8 | 1.28 | 1.16 to 1.47 | 0 | 4 of 4, 1.34, 1.22 to 1.45 |
| | nominal | 200 | 9 | 1 | 2.00 | - | 0 | 2 of 4, 1.98, 1.95 to 2.01 |
| | nominal | 25 | 9 | 3 | 1.71 | 1.55 to 1.71 | 0 | 2 of 4, 1.58, 1.46 to 1.70 |
| Bittern, centre chair (8 tests) | measured | 200 | 16 | 9 | 1.18 | 1.17 to 1.41 | 0 | 5 of 8, 1.18, 1.17 to 1.21 |
| | measured | 25 | 16 | 16 | 1.07 | 1.00 to 1.28 | 1 | 8 of 8, 1.06, 1.00 to 1.18 |
| UL/NFPRF I-17, I-18, I-20, I-22 | measured | 200 | 16 | 16 | 1.18 | 0.41 to 1.96 | 5 | 4 of 4, 1.09, 0.98 to 1.11 |
| | measured | 25 | 16 | 16 | 1.16 | 0.41 to 1.90 | 5 | 4 of 4, 1.08, 0.96 to 1.11 |

Over the 29 runs on the curves the data favour (tabulated or measured) at the app's coefficient, the estimate opened the first head within twice the measured time in 21, at 0.94 to 2.00 of the measured time (median 1.18), later than measured in 19 of the 21. At 25 W/(m² K) it opened in all 29, at 0.82 to 1.79 (median 1.11).

The late reading at the app's coefficient opened 1 of the 110 measured heads in all runs within twice the measured time (one beamed-ceiling head on the nominal curve, at 2.00). At 25 W/(m² K) it opened 14 of 18 in the 1998 room (median 1.80), all 12 of the 2003 smooth set (1.27), 8 of 9 under the beams (1.28; there the heads are out of the plume's sight and both readings are the cell), 2 of 16 in Bittern (1.85) and none in UL.

The share of measured heads between the two readings (estimate at or before the measured time, late reading at or after it, a late reading not open counting as after) was 0 % in the 1998 room at either coefficient, 10 % (200) and 57 % (25) in the 2003 set, 0 and 6 % in Bittern, and 31 % in UL. At the app's coefficient the bracket fails almost everywhere because the estimate itself is late.

The gas each reading saw at the measured activation time, against the gas measured at the head (medians over the heads with a measured value; app coefficient; tabulated or measured curve):

| Data set | Heads | Measured | Top cell (late reading) | Estimate | Measured speed | Cell speed | Estimate speed |
|---|---|---|---|---|---|---|---|
| Vettori 1998 | 18 | 114 °C | 25 °C | 66 °C | 1.53 m/s | 1.13 m/s | 1.13 m/s |
| Vettori 2003 | 21 | 137 °C | 29 °C | 81 °C | not in the files | 0.64 m/s | 0.85 m/s |
| UL/NFPRF | 16 | 151 °C | 21 °C | 145 °C | not measured | 2.36 m/s | 4.33 m/s |

UL's count of heads opened, with the spray running: I-17, 8 by 160 s against 4 by 80 s (6 in the exp CSV, the last two at 662 and 694 s); I-18, 4 against 4; I-20, 8 by 200 s against 4 by 196 s; I-22, 4 by 140 s against 4 by 68 s (6 by 265 s). The spray evaporated 0.3 to 3.7 kg of the 790 to 1,870 kg it emitted per run.

Bittern's paired heads. In tests 12 to 15 a standard-response SS68 head (RTI 95) and a residential Res B head (RTI 36) sat 2.0 m from the chair on either side. The measured gaps, SS68 after Res B, were 18, 10, 16 and 17 s. The estimate's gaps at 25 W/(m² K) were 37, 27, 40 and 38 s; at 200 only test 13 opened both, 59 s apart.

### R5. What agrees

- The link equation with UL's measured RTI and C reproduces Vettori's activations when it is given the measured gas (R2).
- In the 2003 room with a smooth ceiling the estimate opened the first head at 0.94 to 1.10 of the measured time on the tabulated ramp at the app's coefficient, and the gas it read at the measured time was close to the measured gas in tests 1 and 2 (139 and 124 °C against 130 and 125 °C).
- In UL/NFPRF the estimate opened the first head at 0.98 to 1.11 of the measured time at the app's coefficient (0.96 to 1.11 at 25), and its gas at the 16 measured heads had a median of 145 °C against 151 °C measured. This is the regime Alpert's correlation was built for, a large fire under a high, open ceiling.
- In Bittern at 25 W/(m² K) all 16 heads opened at 1.00 to 1.28 of the measured time.
- The nominal t-squared curves gave later activations than the tabulated ramps in every set, by 0.2 to 0.5 of the measured time, which agrees with section 1's energy check that the nominal curves under-state what burned.

### R6. What does not agree, and the likely reasons

At the app's settings the model opens heads late in rooms, and its grid reading almost never opens them. The checks below point to five causes.

1. Heat lost to the walls. At 200 W/(m² K) the walls, floor and ceiling take most of the convective heat. In a box of the 1998 room's size (no beam or stair) on the fast tabulated ramp, the gas held 1.26 MJ above ambient at 45 s of the 4.80 MJ the burner had put in as convection; at 25 W/(m² K) it held 4.30 MJ. Section 1's energy check found 2.5 to 3.0 MJ in the top 0.9 m of the real room at the first activation, so the real loss lies between the two. Lowering the coefficient to 25 moved the 1998 median from 1.68 to 1.31 and Bittern's from 1.18 to 1.07, and let the late reading open. The smoke transport note (section 2.3) predicted this from the code.
2. The coarse plume. In an open domain with no walls or ceiling, a steady 100 kW burner (80 kW convective, 0.81 m² of cells, two cells high) on 0.3 m cells at the app's settings carried a time mean (30 to 90 s) of 38 kW through a plane 0.6 m above the floor and 30 kW at 2.1 m, and the mean excess on its axis in the cell centred 2.25 m up was 4.6 K, where Heskestad's centreline correlation gives about 85 to 105 K (depending on where the source is taken). The hottest cell at 1.2 and 2.1 m moved by 1 to 4 m between snapshots 10 s apart. With the vorticity confinement off, the plume stood still at about 280 K above ambient at every height up to 3 m, and its enthalpy flux grew with height from 97 to 184 kW, more than the input. Neither behaves as a plume. The layer that the estimate adds Alpert's jet to is therefore colder than the real layer, and the cell under a head colder still.
3. The ceiling jet against the cell. The jet's hottest gas lies within a few centimetres of the ceiling and the jet is about 0.1 H thick (about 0.2 m here), while the top cell's centre is 0.15 m down and the cell averages 0.3 m. The top cell read a median of 25 °C at the measured activations in the 1998 room, where 114 °C was measured, and 21 °C against 151 °C in UL.
4. Alpert's correlation in a closed room. Driven by Alpert's jet alone over ambient air, at the true H and r and the test's curve, the same link opens the 1998 heads at a median of 2.07 times the measured time on the tabulated ramp (1.60 to 2.96), the 2003 smooth set at 1.37, the beamed set at 1.08 and UL at 1.26, and never opens Bittern's heads within 600 s (my calculation, `alpert` and `linkStep` of suppression.js). In these rooms the hot layer carries much of the heat that reaches the heads. The estimate adds the grid's layer, which brings the 1998 room from 2.07 to 1.68 and lets Bittern's heads open, but that layer is too cool for causes 1 and 2.
5. Grid geometry. Beams snap to whole cells, and a head with no clear line from the plume's ceiling cell is read on its own cell only; at the app's coefficient no head behind the 2003 beams opened. In UL the heads that are all 2.15 m from the burner sat 1.43 to 2.87 m from it on the grid, because `sprinklerSetup` measures r between cell centres; in I-17, I-18 and I-20 the head at 1.43 m opened first whatever the measured order.

The fire curve and C matter less at the app's settings, but they are not small. On the nominal curves the estimate was later by a further 0.2 to 0.5 of the measured time (R5). In the replay, C 0.25 against 0 moved the median by 4 % and the number of records that open from 25 to 18; Bittern's C 0.4 is a modelling choice, and on a slowly rising gas the C term holds the link back longer than on the measured gas. Bittern's doubled RTI gaps (R4) are the same effect: on a gas that rises too slowly, a slow link lags further.

UL's count. The runs opened as many heads as the tests or twice as many. The solver's spray evaporates only where the gas is above 100 °C, the cells under the heads were near ambient (a median of 21 °C at the measured activations), and it evaporated a few kilograms of hundreds, so it hardly cooled the jet. The real spray also cooled the jet by convection to the drops and by drag.

### R7. A finer grid

Thirteen runs were repeated at 0.2 m (0.203 to 0.204 m; UL I-17 at 0.38 m) with the app's coefficient and the tabulated or measured curve. The cells under the heads stayed as cold, 19 to 36 °C at the measured activation times in the rooms where 93 to 179 °C was measured, and the estimate came at the same time or later:
- 1998 tests 6, 8, 12 and 13, head 1: 1.61, 1.73, 2.00 and 1.99 at 0.3 m against 1.68, 1.83, 2.12 and 2.07 at 0.2 m; in test 7 no head opened within twice the measured time at 0.2 m (1.98 at 0.3 m);
- 2003 tests 1 and 2: 0.93 to 1.10 at 0.3 m against 0.99 to 1.17; beamed test 13 opened no head at either size;
- Bittern tests 3, 4, 10 and 13: all 8 heads opened at 0.3 m (1.17 to 1.41), 3 of 8 at 0.2 m (1.23 to 1.28);
- UL I-17 at 0.38 m: first head 1.26 against 0.98 at 0.507 m, as the heads' cell-centre distances changed.

The results file was then regenerated with the finer grid on every case (46 runs at 0.2 m and UL at 0.38 m, the app's coefficient, both curves; 138 runs in `benchmark/results/2026-11-07-sprinkler-validation.json`). The 0.3 m matrix came out identical to the runs above, and the finer grid on every case agrees with the thirteen: estimate medians 1.70 (1998, 12 of 18 heads open), 1.14 (2003, 12 of 21), 1.26 (Bittern, 3 of 16) and 1.15 (UL, first heads 1.22 to 1.38) on the tabulated or measured curves, none better than at 0.3 m.
A smaller cell does not move the result toward the measurements at the app's coefficient. A smaller gas cell next to a wall relaxes to it faster (its relaxation time is proportional to the cell size at a given coefficient), which may be part of the reason; I did not test it.

### R8. What the app should say about its activation times

On these data, at the app's settings, the activation time Studio reports (the estimate) is late in rooms and fair under a high, open ceiling:
- in the three room data sets the first head opened at 0.94 to 2.00 of the measured time where it opened, and in 8 of 25 room tests no head opened within twice the measured time;
- under UL's 7.6 m ceiling the first head opened at 0.98 to 1.11 of the measured time;
- the link model itself is sound: on the measured gas it opens within 0.81 to 1.18 of the measured time;
- the late reading should not be shown as the time a head opens: at the app's settings it opened 1 of 110 heads within twice the measured time;
- a head separated from the fire by a beam a cell deep or deeper is read on its own cell and, at the app's settings, did not open.

A wording the run card could carry: "Sprinkler activation is estimated from Alpert's ceiling jet over the simulated smoke layer. Against 29 full-scale tests (NIST data, checked 2026-11-07) it opened the first head later than measured in rooms, up to twice as late or not at all, and within about 10 % under a high open ceiling. Treat it as a late estimate; heads behind beams may not open in the run." A late activation delays the modelled suppression, so for tenability the bias is conservative; for anything that starts on sprinkler flow, such as an alarm or a door release, it is not.

Changes the evidence supports, not made here: a lower surface coefficient for inert walls and ceilings than for fuel (R6.1); r from the head's position to the fire origin rather than between cell centres (R6.5); spray cooling below boiling (UL's count); and a check of the plume against Heskestad's centreline before any tuning of activation (R6.2).

### R9. Run times, and the results I trust least

Grids: the 1998 room 37 x 21 x 25 (19,425 cells, 0.305 m, with the upper floor), the 2003 room 30 x 12 x 24 (8,640, 0.306 m), Bittern 33 x 12 x 20 (7,920, 0.300 m), UL 41 x 17 x 43 (29,971, 0.507 m); at 0.2 m, 51,272, 22,176 and 19,136 cells, and UL at 0.38 m 67,716. Alone on an i7-14700 (Node 24), the slowest run of each set took 66 s (1998 test 12, 255 s horizon, 26 ms per step), 57 s (2003 test 16, 445 s, 13 ms), 64 s (Bittern 15, 540 s, 12 ms) and 94 s (UL I-20, 200 s, 47 ms), with the same activation times as in the batch. The 92 runs at 0.3 and 0.5 m took 660 s with 12 at a time (31 to 122 s each under that load); the 13 finer runs took 279 s with 13 at a time (74 to 278 s each). The link replay and Evans tests take a few milliseconds and the burner tests about 2 s.

The results I trust least:
- the 1998 room's set-up: the sealed door, the assumed upper-floor height and the snapped I-beam; it is the room where the model is furthest off, and I cannot separate how much of that is the set-up;
- the 2003 C (0.25) is assumed, and the later 2003 heads opened in the tests after the gas was shut off while the solver's burner kept burning;
- the flame column height is my choice; it puts UL's heat over 5.1 to 5.6 m and the rooms' over 0.6 to 1.5 m, and the UL agreement may depend on it;
- Bittern's curves come from mass loss, its radiant fraction (0.46) and C (0.4) are the original study's modelling choices, and the chair's orientation is not given;
- UL's counts depend on the 200 s horizon and the spray model, and I-17's measured count differs between Table 1 and the exp CSV;
- the open-plume and energy diagnostics in R6 are single runs of my own, not part of the harness.
