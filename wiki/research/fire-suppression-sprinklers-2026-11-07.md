# Sprinklers and fire suppression: codes, placement on an IFC model, simulation, and what is proved (2026-11-07)

Status: research and plan (below), then built on 2026-11-07 at the user's request ("work through 1 to 4"); the section "As implemented" says what was built and where it departs from the plan. Committed to `next/browser-core`, not deployed: the user releases the sprinkler work as a whole. The detailed sources are four companion reports:

- `sprinkler-codes-2026-11-07.md`: when the codes require sprinklers, NFPA 13 placement, hydraulics, response classes, outdoor assembly points (59-row value table with sources);
- `sprinkler-physics-2026-11-07.md`: activation, the coarse-grid problem, suppression models, data per sprinkler, the spray, validation data, statements to prove;
- `sprinkler-ifc-mep-2026-11-07.md`: open IFC models with fire protection, the IFC schema, federation of an MEP model;
- `smoke-transport-2026-11-07.md`: smoke under ceilings, ducts, open and closed buildings, stack effect, checks for the solver.

Earlier notes: `fire-suppression-note-2026-11-05.md`, `sbc-firefighting-future-notes-2026-11-05.md`.

## As implemented (2026-11-07)

All four steps of section 10, behind a **Sprinklers** switch in the Simulate step (under the Flamescapes models), off by default. Off, nothing is laid out, no field is allocated beyond a 16-byte placeholder on the GPU, no step does any sprinkler work, and a run keeps its kept-run key and its results (the same peak heat release and untenable nodes). Speed with it off: the CPU solver before and after on one scene, best of four rounds, 154.9 against 156.5 ms a step at 187k cells and 37.7 against 37.7 at 45k (single bench runs vary by about 15 % with the machine's load); the GPU 3.06 and 16.1 ms a step at 27k and 187k cells before and after. With it on, Vienna's run took 38.5 ms a step against 40.2 without.

1. **Record, IFC reading, layout and rule report** (`src/core/sprinklers.js`, its 19 tests; `ifcSemantics.js` reads `sprinklers`, `IMPORT_SETTINGS` 5). IfcFireSuppressionTerminal (IFC4) or an IfcFlowTerminal typed by IfcFireSuppressionTerminalType (IFC2x3), named as a sprinkler and not a hose reel or hydrant, with the Revit sets' temperature rating and K-factor; the local MEP models read 292 (NIBS Office, 57 °C, K 81), 418 (Clinic), 1,354 (hospital, no properties). Without IFC heads, rooms from the IfcSpace meshes (bottom faces, minimum-area rectangle), the rectangle rule of section 3 at bay centres (the Lean theorems' case), with extra heads where a bay centre is not clear of the walls, both orientations tried, heads under 1.8 m apart merged, and coverage checked on a 0.25 m sample; shafts and spaces drawn over smaller rooms reported, not laid out. Without rooms, a lattice over the walkable floor. Every rule cites SBC 801-CR-2018 903.3.1.1 and IBC 2021 903.3.1.1 with the NFPA 13-2019 clause; obstructions "cannot tell" without a structural or MEP model. The requirement report applies 903.2.8 (Group R), 903.2.11.3 (16.764 m, IBC's 55 ft, against SBC's 17 m), high-rise and the A-3 and E triggers to the occupancy chosen for the performance check. Office 315 heads in 99 rooms, Clinic 651 in 269, Duplex 46 in 21, Vienna 173 in 111 spaces (45 ms).
2. **Activation, card and view** (`src/core/suppression.js`). Both readings of section 4.4, stepped exactly per frame on the gas held at the mean of the frame's two readings (`exactStep`, `exactStep_add`), the crossing time found inside the frame (`linkExact_activation`). The estimate takes Alpert's jet for the frame's heat release with the ignition cell as the plume axis (not the heat-weighted centroid, so both backends read it alike), for heads with a clear line to the plume's ceiling cell at the ceiling (walls and lintels part rooms); the layer is the mean of those heads' cells at r of at least 0.2 H. Sensor cells are read with the nodes (after the ceiling detectors); the GPU's gather adds their speed. The run card gives the first opening by both readings with the heat release at each; the view shows the heads, blue once open.
3. **Evans** as a per-cell cap in `chemistry` on both backends (`evans` in the WGSL): the step after a cell is covered reads its rate, then its oxidation, pyrolysis or pool evaporation is scaled to at most rate times exp(-(t - t_act)/tau). Density from every open head over the cell; below 0.026 mm/s no cap. The rate reads zero for a cell not yet reacting, which then never ignites (the first version left such a cell armed until the fire reached it and recorded that later rate; the room test caught it). Suppression starts at the end of the frame in which a head opened, so up to one frame late on both backends.
4. **Spray** as a `spray` stage after `chemistry` (one GPU thread per column; heads sharing a column are one entry): each cell evaporates 1 - exp(-k (T - T_boil) h) of the water left (k from section 7.1's 1.7 % per 0.35 m cell at 300 K for 1 mm drops), at most what cools it to boiling, with the energy identity of item 9, steam through `changeGas`, and the drag of section 7.2; the rest reaches the floor; the fuel is not wetted. **Manual suppression**: the extinguishing agents per fire class (`fireClasses.js`) are listed in the Simulate step as before; no extinguisher's effect is modelled, since no source read gives it in terms this solver can use.

Departures from the plan and what the build found:

- **No forced cull of flame-front particles in sprayed cells.** With it, a test fire's heat halved while the water absorbed a few hundredths of it: the cull put the flame out by fiat. The solver's own cull (where combustion stops) is kept.
- **Peaks are not what Evans bounds.** In the 5 x 4 m test room, Evans halved the fuel burnt after the head opened (12.6 against 23.8 kg), but pyrolysis gas that does not burn where it is made gathers under the ceiling (in the free fire too: 9.6 kg by 120 s) and can ignite at once, so a single frame can peak higher with suppression than without. The tests therefore check the fuel and the energy, and every covered cell's cap step by step, not one frame's heat release.
- **The spray's direct effect is small at sprinkler scale**, as section 7.1 predicted: on the Duplex 0.14 of 37.7 kg evaporated in 60 s, and in one test room the spray released the same heat as Evans alone within 0.3 %. In the GPU check's room, where a 200 kW burner brings the room near a flash as the head opens, the spray released more heat than the free fire on the CPU (58.9 against 55.6 MJ) and less on the GPU (45.1 against 49.6 MJ): there the solver's sensitivity dominates, and no single run should be read as the spray's effect.
- **Section 9 item 9 does not reduce to Fire-X's X30 at T' = T_B**, as the report says; Suppression.lean proves the exact relation (`evaporate_vs_X30`: X30 is the case e = 0, Fire-X paying the latent heat from the liquid) and `evaporate_limit_pinned` for T' = T_B.

**Checks.** `src/core/suppression.test.mjs` (the report's worked activation times to 0.1 s, Alpert's branches, Evans's table, the cap, the column sharing, a room fire in four modes, the cap in every step); `tools/flamescapes-gpu-check.mjs --phase sprinklers` (CPU against GPU: a head over a 200 kW burner opens at 32.2 s on the CPU and 31.4 to 31.9 s on the GPU by the estimate; 35.4 kg of water on both); `tools/fire-bench.mjs --sprinklers` (Duplex and Vienna with heads placed per room); the browser suite (layout and run card). **Lean:** Suppression.lean, 58 theorems on the standard axioms, with Sprinklers.lean's 28.

**Left after the first round:** federating a second MEP IFC (section 6 of the MEP report); obstructions from a structural model; the wet-link term; wetting the fuel with a calibrated wetted fraction; Lagrangian drops for the view; comparing activation with Vettori and Bittern and suppression with the NIST tests of section 8; a plume axis from the heat-weighted centroid; extinguishers. The second round below built five of these.

## Second round (2026-11-07): the five items left

The user asked for the five items the first round left, built and tested for the next release. All are behind switches or appear only when used, and cost nothing otherwise.

1. **A second model** (`src/core/federation.js`, `ViewerManager.loadLinkedIfc`; Model step, "Second model (MEP, structure)", folded away). An MEP or structural IFC of the same building is converted and cached as the first is and drawn in its frame by `C_main R C_linked^-1`, the models' coordination matrices with `R` the IFC-frame alignment: the identity, or the two files' IfcMapConversion when both have one. The checks of section 6.4 of the MEP report follow: units, site placement, TrueNorth (metadata only), map conversions, storeys matched by elevation (a storey outside the first model's range, a foundation level, is a note), and the share of the smaller plan the two have in common (both measured by the fragments models' whole boxes; a site-wide service model holding the building overlaps it fully). A file with no geometry (the NIBS Office MEP export carries properties only) says so. Measured: NIBS Office architecture with its HVAC model (292 sprinklers, all placed 5 cm under the architecture's ceilings, 4.7 s to add) and its structural model (356 beams, 96 % overlap, the foundation level noted); the Duplex with its MEP model (98 %); the buildingSMART Simple-Scene building with its HVAC model (placements) and its plumbing (through the two map conversions, checked by hand to 0.1 m against the drawn origin marker; `federation.test.mjs`). The second model's sprinklers feed the layout when the first has none; `ifcSemantics` reads each file's frame and storeys' declared elevations (`IMPORT_SETTINGS` 6).
2. **Beams and ducts as obstructions** (`sprinklers.js` `checkObstructions`, `obstructionFromMesh`, `layoutRooms` with `obstructions`; `ViewerManager.overheadMeshes`). Beams from either model, ducts and cable trays from a second model (IFC2x3 flow segments by name); pipes are left out of the beam rule. NFPA 13-2019 Table 10.2.7.1.2 (the beam rule, no interpolation), 2022 10.2.7.3.1.3 and .4 (three times, structural members only in light and ordinary hazard), 2019 9.5.5.3 (beneath obstructions wider than 1.2 m), from the codes report. Heads failing a rule move within their bay to the nearest point meeting every rule; wide obstructions get heads beneath. Output without obstructions is unchanged (a test compares it with the old output). 198 heads against 1000 obstructions lay out in about 40 ms. The Flamescapes settings have a switch, "Beams and ducts in the fire grid", off by default, that makes them solid (NIBS Office: 999 items, 82 ms to extract).
3. **Water wetting the fuel** (sprinkler mode "wetting"; `flamescapes.js` `spray`, WGSL `spray`). No Evans cap: each column's water, after evaporating on its way down, soaks into the fuel cell it ends on, a share `WETTED_FRACTION` of what reaches it, mixed in at 20 C so the surface keeps its energy; the solver's drying then holds the surface near 100 C. Fuel stacked below the top cell stays dry. Calibrated (`tools/wetting-calibrate.mjs`, `benchmark/results/2026-11-07-wetting-calibration.txt`) on the test room: the share is 0.52, where wetting burns as much covered fuel after the head opens as the cap (7.4 kg, against 21.3 kg with no water) at 0.068 mm/s. It then follows the water: 9.0 kg at 0.026 mm/s and 4.5 kg at 0.133 mm/s, where the cap gives 8.6 and 7.4 kg, because on this room the cap mostly freezes covered cells not yet burning. CPU and GPU agree (54.3 and 52.9 MJ released in the GPU check's 200 kW case, against 72.5 and 66.7 MJ with no water); in that case the cap and spray modes released as much as the free fire or more, the flash sensitivity noted above.
4. **Validation against the measured tests** (`tools/sprinkler-validation.mjs`, `benchmark/validation/sprinkler-tests.json` with NIST's notice, `benchmark/results/2026-11-07-sprinkler-validation.json`; `wiki/research/sprinkler-validation-data-2026-11-07.md` with its results, R1 to R9). A prescribed heat-release burner was added to the CPU solver for it (`createBurner`; nothing when absent, fields bit-identical without it). The link itself is right: driven by the gas measured at Vettori's heads it opens at 0.81 to 1.18 of the measured time, median 1.00 (`suppression.validation.test.mjs`). On the grid, at the app's settings, the estimate is late in rooms: median 1.68 times the measured time in Vettori's 1998 room (16 of 18 heads open within twice), 1.09 under his 2003 flat ceiling, 1.18 in Bittern's rooms with 9 of 16 heads opening; under UL/NFPRF's high open ceiling the first heads open at 0.98 to 1.11. Heads behind the 2003 beams did not open; the late reading opened 1 head of 110. The cause is mostly the solver's surface coefficient of 200 W/(m² K), which takes most of the jet's heat into the walls (at 25 W/(m² K) the 1998 room's median is 1.31 and Bittern's 1.07 with all heads opening), and a coarse plume; a 0.2 m grid does not help. Evans's tau is slower than 19 of the 23 measured decay constants of his own tables. The app now says this on the run card and the Methods page. Changing the surface coefficient would change every Flamescapes run, so it is left for the user to decide.
5. **Extinguishers** (`src/core/extinguishers.js`, Simulate step, "Extinguishers", off by default; `wiki/research/extinguishers-2026-11-07.md`). Read from the IFC by name (here and in a second model), else placed on the walk by greedy cover; the placement check measures the walk to the nearest unit per storey (22.86 m, the stricter of SBC 801-CR-2018 Table 906.3(1) and NFPA 10 Table 6.2.1.1, passed at 22.51 m on a 0.5 m graph), the count rule and where they are required (906.1). After a fire, one person near it sets off when they see it (30 s) or after detection, the alarm and the median pre-movement, walks to the nearest suitable unit, readies it (13.4 s, 7.2 s trained) and attacks from 1.8 m; the heat release then (from the Flamescapes run, else the two-zone settings' t-squared fire, said so) against what the rating puts out gives put out, not put out, or cannot tell. Duplex: 4 units placed in 215 ms and passing; seeing the fire, the attack starts at 58 s against 37 kW (put out); after the alarm, at 222 s against 547 kW (cannot tell, between 80 and 900 kW for a 2-A unit). Not yet: the unit's effect on the fire in the run (`extinguisherCap` is written and tested, not wired), mounting heights, the on-route rule.

Also this round: the spray drawn under each open head (droplets falling in a cone, moved by their shader; nothing while no head is open); the sprinkler panel says when the obstruction rules were checked; closed doors and windows are airtight in the solver, measured on the Duplex (`tools/envelope-check.mjs`: all closed, 0.0 g of 138 g of soot in the outside air after 60 s; doors open, 24 %), and the panel now says so, and that a grid clipped round the fire lets smoke out at its sides.

**Water by fire class** (`fireClasses.js` `waterResponse`; `wiki/research/water-by-fire-class-2026-11-07.md`, asked by the user: "will the sprinkler actually wet the flame, and how does the flame react?"). Sprinkler drops, about 1 mm, are too large to put a flame out in the gas (Rasbash 1986: a spray must take about 0.4 of the flame's heat per unit volume and push down with 0.7 of its upward force); sprinkler water works by cooling the burning surface below its fire point, where it reaches it. So an item of class A, or C once the circuit is dead (its plastic), is wetted and capped; heptane (class B: fire point below the water's temperature, lighter than water, so the water sinks and the pool keeps burning), cooking oil (K: water under hot oil turns to steam and throws it out; a NIST residential sprinkler put out one of two cooktop fires and spread oil both times), magnesium (D: water gives hydrogen and heat, 17.5 MJ and 0.11 kg of hydrogen per kg reacted) and a lithium-ion pack (L: the water does not reach the cells) are neither wetted nor capped, on both backends, and the Sprinklers section says why when one burns first; the room's class A contents round them are wetted and capped as usual. These items also no longer start with wood's 10 % water, which the research found: this changes runs of classes B, D, K and L with or without sprinklers (the kept runs' version is now 3). Not modelled: the oil's eruption and the metal's reaction with water (no study measured the oil thrown out; the hydrogen needs a fuel kind the solver does not have).

**Left after the second round:** the solver's surface coefficient (the main cause of late activation; the user's decision, since it changes every run); the wet-link term; Lagrangian drops; the plume axis from the heat-weighted centroid; an extinguisher's effect on the fire in the run (`extinguisherCap`); the oil's eruption and the metals' reaction with water; NIST suppression tests beyond Evans's tables.

## 1. The user's requirements (2026-11-07)

1. Place sprinklers on a model per the codes: SBC rules for design and placement, from an IFC model and from 2D drawings (a university building is the target; drawings alone if no IFC exists).
2. Read sprinklers and the fire system where the IFC already has them, possibly from a second MEP IFC.
3. Simulate the suppression, following the stoichiometric approach of Fire-X (Wrede et al. 2025, KAUST VCC, doi:10.1145/3763338), whose gas chemistry the Flamescapes port already uses.
4. Plan, gather the resources and verify in Lean before implementing.
5. **Performance:** suppression is off by default, behind switches in the interface, and costs nothing when off; with it on, the solver must stay as fast as before or faster, and it must handle the Vienna model on the web.

## 2. What the codes require (summary; values and sources in the codes report)

**The SBC documents (identified 2026-11-07).**

- **The current edition is SBC 801-CR-2024**, "The Saudi Fire Protection Code, Code Requirements" (English). The Saudi Building Code site publishes its table of contents (12 pages) openly. It places Section 903 "Automatic Sprinkler Systems" at page 150. Sections 915 (CO detection), 916 (gas detection) and 917 (mass notification), the last two added in IFC 2021, confirm an IFC 2021 base. Chapter 10 is the means of egress (Section 1028 exit discharge). Chapter 80 holds the referenced standards, at page 682. Appendix M covers retroactive sprinklers in high-rise buildings, and the code runs to about 764 pages. **The 2024 text itself was not obtained** (the English edition is sold).
- **The previous edition, SBC 801-CR-2018**, was read in a licensed copy for internal use. It is kept locally and is not in the repository; its text is never quoted here, only clause numbers and values. Its Section 903 confirms the IBC values in metric form, with the Civil Defense in place of the fire department:
  - **903.2.8:** sprinklers throughout any building with a Group R fire area;
  - **903.2.1.3:** A-3 over 1,115 m², with 300 or more occupants, or off the level of exit discharge; a Saudi exception exempts detached mosques of Type I or II construction;
  - **903.2.3:** Group E over 1,115 m²;
  - **903.2.11.1:** storeys without openings over 140 m²;
  - **903.2.11.1.2 and 903.2.11.1.3:** 23 m for openings on one side only, and for basements;
  - **903.2.11.3, "Buildings 17 m or more in height":** a storey with 30 or more occupants 17 m or more above the lowest Civil Defense vehicle access;
  - **903.3.1.1, 903.3.1.2, 903.3.1.3:** NFPA 13; NFPA 13R for Group R up to four storeys and 18 m; NFPA 13D;
  - **903.3.2:** quick-response or residential sprinklers in dwelling and sleeping units of Groups I-1 and R, and in NFPA 13 light-hazard occupancies;
  - **903.3.3:** sprinklers under kiosks and equipment wider than 1,200 mm.

**The user's rule (2026-11-07): it has to pass SBC 2018, using IBC 2021 where it is stricter, until the SBC 2024 text is obtained.** The egress code check now applies, for each limit, the stricter of SBC 801-CR-2018 and IBC 2021 (`core/codeCheck.js`, `EGRESS_B`), so a pass meets both:

- travel distance 60 m, or 90 m with sprinklers (SBC);
- 9 m² per occupant (SBC);
- single exit within 22.86 m (IBC), or 30 m with sprinklers (SBC);
- doors 813 mm (IBC);
- high-rise above 22.86 m (IBC).

The sprinkler rules follow the same rule: SBC 801-CR-2018 Section 903 where read, IBC/IFC 2021 where stricter or where SBC 2018 was not read.

| Where | Requirement (IBC/IFC 2021) | Clause |
|---|---|---|
| Apartments (R-2) | sprinklers throughout; NFPA 13R only up to 4 storeys and 18.3 m | 903.2.8, 903.3.1.2 |
| Offices (B) | no trigger by use or area; sprinklers throughout when a storey with 30 or more occupants is 16.8 m or more above fire service access | 903.2.11.3 |
| High-rise | an occupied floor more than 22.86 m above fire service access; sprinklers throughout (the Saudi enforcement schedule speaks of towers above 23 m) | 202, 403.3 |
| University | Group B; lecture halls may be A-3 (over 1,115 m², 300 or more occupants, or off the exit discharge level); higher-education laboratory suites sprinklered | 304.1, 903.2.1.3, 428 |
| Standards | NFPA 13, 13R, 13D (sprinklers), 14 (standpipes), 20 (pumps); probably the 2019 editions for SBC 2024 | 903.3.1, 905, 913 |

**Placement (NFPA 13, standard spray, confirmed by two or more sources):**

| Rule | Light hazard | Ordinary hazard |
|---|---|---|
| Area per head, A = S x L | 20.9 m² | 12.1 m² |
| Largest spacing S | 4.6 m | 4.6 m |
| Distance to a wall | at most S/2 (2.3 m); 2.7 m to one wall in a light hazard room up to 74 m²; at least 0.102 m | same, without the small-room rule |
| Between heads | at least 1.8 m | at least 1.8 m |
| Deflector below a smooth ceiling | 25 to 305 mm | 25 to 305 mm |
| Obstructions | three-times rule (capped at 610 mm), beam rule, heads beneath obstructions wider than 1.2 m, 457 mm clear below the deflector | same |

Offices, dwellings and classrooms are light hazard; kitchens, car parks and plant rooms ordinary group 1; storage ordinary group 1 or 2 by height. Design density 4.1 mm/min over 139 m² (light) to 8.1 mm/min (OH2); Q = K sqrt(P), K80 for the common head; quick response RTI at most 50 (m s)^1/2, standard at least 80; ordinary bulbs 57 to 77 °C.

## 3. Placing sprinklers on a model

**Two grids, kept apart.**

- **A placement grid per room**, on its ceiling: built from the room's footprint (`IfcSpace`, or the LOD1 room from drawings) and its ceiling height (the slab above, or the space's height). This is where the code rules apply. It does not reuse the walkable graph, which lies on the floor at 0.5 m spacing, nor the solver's grid.
- **The solver's grid**, which only needs each head's sensor cell (the top open cell of its column) and the cells under its footprint.

**Layout per room (rectangles first).** For a room of L x W and the hazard's spacing S and area A:

1. `nx = ceil(L / S)` bays along, heads at the bay centres;
2. `ny = max(ceil(W / S), ceil((L / nx) W / A))` bays across;
3. check the wall distance (at least 0.102 m, at most S/2 or the small-room 2.7 m), the 1.8 m minimum, and the deflector distance;
4. report each head's protection area and every rule as pass, fail or "cannot tell" with its clause, as the code check does.

Steps 1 and 2 are proved in Lean to meet the spacing, wall-distance, coverage and area rules (`spacing_le`, `wall_distance_le`, `covered`, `area_le`). Non-rectangular rooms: decompose into rectangles, or lay the rectangle grid over the bounding box, drop heads outside the room, and check that every point of the room is within a protection rectangle of a head (a coverage check on a fine sample, flagged where it fails). Beams and ducts (obstructions) come only from a structural or MEP model; without one the report says the obstruction rules were not checked.

**Hazard class per room** from the room's name or `Pset_SpaceCommon`/`OccupancyType`, with the user able to change it (as the occupancy choice in Likely ignition points).

**From 2D drawings** (the university building): `drawingToLod1.js` already drafts rooms from DXF. Ceiling heights come from the storey heights the user confirms in the planned review screen. The sprinkler layout then runs on the LOD1 rooms. The university's drawings stay local unless the university allows their use.

## 4. The sprinkler record and IFC

Per head: position, type (pendent, upright, sidewall), activation temperature, RTI, C, K-factor and pressure or flow, coverage, room. `Pset_FireSuppressionTerminalTypeSprinkler` gives activation temperature, response class, coverage, flow, pressure and a discharge coefficient, but **no RTI or C**; defaults by class (quick 50, standard 100; C 0.4 with a sensitivity switch) are stated on the card. Placed heads and IFC heads use the same record.

## 5. Simulation, in order (physics report sections 3 to 10)

1. **Activation only.** Each head's link follows `dT/dt = (sqrt(u)(Tg - T) - C (T - Tm)) / RTI`, stepped exactly over each solver step (no time-step bound). The gas is read two ways: the top open cell (a late bound) and CFAST's hybrid (the grid's upper-layer temperature plus Alpert's ceiling-jet excess from the solver's own heat release). The card reports "head N opens at t = ... s" by both readings, and the fire is not changed. **Why two readings:** a 0.29 to 0.4 m cell averages the whole ceiling jet (0.2 to 0.3 m thick), which delays activation, by about 20 to 25 % by the physics report's estimate (an estimate to check against the Vettori and Bittern tests).
2. **Evans's suppression** (NISTIR 5254, 1993) as a per-cell cap on the burning rate of covered fuel: `Q(t) = Q(t_act) exp(-(t - t_act)/tau)`, `tau = 3.0 w''^-1.85` s (w'' in mm/s; 410 s at 0.07 mm/s). Calibrated on full-scale sprinklered furnishings, conservative, the same as CFAST, and one float per cell on both backends.
3. **Fire-X-style evaporation** in an Eulerian spray zone under each open head: energy-limited evaporation cools the gas, steam dilutes the oxygen, and the smoke layer can fall (visibility at head height may drop when a head opens; the app must show this). Wetting the fuel through the existing moisture field waits until a wetted fraction is calibrated so that covered fuel is never suppressed faster than Evans allows.
4. Manual suppression (extinguishers by fire class) and, optionally, Lagrangian droplets for visuals.

**Validation:** activation against Vettori (NISTIR 6253, 7079) and Bittern (2004); suppressed heat release against Madrzykowski and Vettori (NISTIR 4833), Walton (NBSIR 88-3696) and Madrzykowski (NISTIR 7506). Fire-X's own sprinkler case is a comparison with FDS only.

**Performance rules (the user's requirement):**

- Off by default, under a "Sprinklers" switch in the Simulate step with sub-switches (show heads, activation only, suppression model). When off: no extra buffers, no kernel dispatches, no per-step work, and the run's key and kept-run store unchanged.
- On: activation costs one exponential per head per step (Vienna would have a few hundred heads, against millions of cell updates); the Evans cap one multiply per fuel cell; the spray zone well under 1 ms per step by the physics report's estimate.
- Measured, not assumed: `tools/flamescapes-gpu-check.mjs --phase timing` and `tools/fire-bench.mjs` before and after each step, on the Duplex and on Vienna in the browser, with the switch off (must match the current timings) and on.

## 6. Open IFC models with sprinklers, and a second MEP model

The full report is `sprinkler-ifc-mep-2026-11-07.md`. No openly licensed **tall** office or apartment building with sprinklers was found in IFC. Three Revit exports of 2 to 7 storeys have sprinklers and a matching architectural model:

| Model | Sprinklers | Fire system | Architectural model | Licence |
|---|---|---|---|---|
| NIBS Office Building, `NBU_OfficeBuilding_Eng-HVAC` (IFC2x3, 2 storeys) | 292 (counted again here) | Revit "Fire Protection Wet" on 695 pipe segments | byte-identical to the local `Office_A_20110811.ifc`; same frame | none stated; the simplest first test |
| NIBS Medical Clinic, `NBU_MedicalClinic_Eng-MEP` (IFC2x3, 2 storeys) | 418 | Revit "Fire Protection Wet" on 1,015 segments and 1,121 fittings | `NBU_MedicalClinic_Arch.ifc`; same frame | buildingSMART's copy says CC BY 4.0 |
| West Riverside Hospital (Auckland repository, IFC4 and IFC2x3, 7 storeys) | 1,354 on levels 1 to 5 | 12 "Fire Protection Wet" and 3 "Pre-Action" systems; a full port graph | metric against imperial, storeys up to 0.27 m apart, no IfcSpace | Autodesk courtesy for research with attribution; "publishable under cc-by-sa-3.0" |

**What real exports contain.** None of the files uses `PredefinedType` SPRINKLER, an `IfcDistributionSystem` FIREPROTECTION or `Pset_FireSuppressionTerminalTypeSprinkler`: every sprinkler type is `.NOTDEFINED.`. Fire systems are plain `IfcSystem` objects named "Fire Protection ...", or only a Revit "System Type" property. K-factor and temperature rating sit in Revit property sets in Revit's internal units (feet, kelvin). The reader must therefore:

- accept IFC2x3 `IfcFlowTerminal` typed by `IfcFireSuppressionTerminalType`;
- recognise sprinklers by type name and Revit system as well as by the enumeration;
- convert Revit units.

**Corrections to the local note.** `advanced_model.ifc` does have 6 sprinklers (the IFC2x3 pattern). TalTech DS3's pipes are heating, not fire protection.

**Federation.** The viewer loads one model at a time. A second MEP model needs both models in one frame: the shared site placement in these exports, or `IfcMapConversion`, which only the buildingSMART certification scenes carry. The hospital pair shows what goes wrong (units, storey drift, true north). The report's section 6.4 proposes a check before overlaying.

All downloads stay in the local test corpus, outside OneDrive; none may enter the repository.

## 7. Outdoor assembly points (a pending feature)

No code or fire service guidance found sets a distance for an assembly point. The only code numbers are for a "safe dispersal area", used where no public way can be reached: at least 0.46 m² per person and 15.24 m from the building (IBC 1028.5). The fire code requires the assembly point, hydrants and fire service access on the evacuation site plan (IFC 404/405). NIOSH's collapse zone of 1.5 times the building height is for firefighters. Institutional guidance: away from hydrants and fire lanes, and never between the building and the responders. "Upwind" appears in no building code (it is hazmat practice). **Proposal for the user's decision:** an assembly-point check that reports, for a chosen point, its distance from the facade against 15.24 m and 1.5 x height, its capacity at 0.46 m² per person against the occupant load, and whether it lies on a fire service access route; the app would not pick the point.

## 8. Smoke under ceilings, ducts, open and closed buildings

The full report is `smoke-transport-2026-11-07.md`. The solver already computes buoyant flow, so hot smoke rising and pooling under ceilings is computed, not assumed. The report found three gaps in the code that probably matter more for smoke transport than the cell size. **All three were confirmed here in `flamescapes.js`:**

- **G1, no thermal expansion.** The pressure projection drives the plain velocity divergence to zero (`project`, Line 33): heated gas takes up no extra volume. A sealed room never pressurises, and an open room pushes no extra volume out of its door. In the Yamana and Tanaka closed-room filling model, leaving out expansion makes the layer reach head height 1.19 to 1.71 times later for layers of 350 to 500 K: late, the unsafe direction. Numerical diffusion partly offsets it; only a run can give the net effect.
- **G2, strong heat loss to walls.** One gas-solid coefficient, 200 W/(m² K) (`DEFAULTS.convection`), serves fuel and walls alike. It was raised from the paper's range so that fuel ignites. For a ceiling layer the FDS correlations give about 5 to 10, so the hot layer probably runs cooler than the MQH correlation predicts.
- **G3, open domain sides.** The grid's sides are open to still air (pressure zero). Where a large model is clipped round the fire, a "closed building" is therefore not closed at the clip.

**Resolution.** At 0.29 to 0.4 m cells, D*/dx is about 1 at 100 kW and 2.4 to 3.3 at 1 MW, against 3 to 12 in most FDS validation runs. FDS's published error statistics are a floor for this solver, not an estimate of its error.

**Ducts.** The current option shows how far ducts could carry smoke, but not the worst concentration: it spreads the fire room's smoke over every air terminal, and it has no fan state, transit delay or duct heat loss.

**Open and closed buildings.** The stack effect, wind as a boundary pressure, and a backdraft warning (unburnt fuel above about 7 to 10 % when a door opens) are reliable enough to add. In Riyadh's summer the stack effect runs in reverse, about -0.77 Pa per metre from the neutral plane.

**Checks first, before any change to the physics** (report section 7). They need a prescribed-heat-release burner, which the solver does not have yet.

1. C1: a closed box, then NIST/NRC Test 1 (410 kW, door closed). An energy budget, and the layer against Yamana and Tanaka, ASET, Beyler and the measurements; this sizes G1 and G2.
2. C3: the Steckler compartment (neutral plane, outflow, mass in and out).
3. C4: the NBS multi-room tests, then the ATF corridors (smoke arrival along a corridor, the quantity that drives the routes).

The candidate improvements, ordered by value per cost:

- the burner and diagnostics;
- a separate wall coefficient;
- thermal expansion;
- zone pressure with door leakage;
- duct realism;
- the stack effect and wind;
- a shaft network;
- the backdraft warning;
- closing the clip faces.

Each is off by default and skips its code path, following the user's performance rule. Six Lean statements (L1 to L6) are specified; layer descent in the two-zone model is already proved (`Zone.lean`, `layer_antitone`).

## 9. Lean: proved before implementation

`formal/fireegress/FireEgress/Sprinklers.lean`, 28 theorems, standard axioms only:

- **Layout:** `spacing_le`, `wall_distance_le`, `covered` (every point within half a bay of a head), `area_le` (each head covers at most A).
- **The link:** the Euler step is a convex combination (`linkStep_eq`), stays between the link, gas and mount temperatures (`linkStep_bounded`, `linkAfter_le`), and is monotone in both (`linkStep_mono_gas`, `linkStep_mono_link`), so a gas history at least as hot never opens a head later (`linkAfter_mono`, `activates_no_later`). The exact step is a special case for any step length (`exactStep_eq_linkStep`, `exactStep_weights`) and two steps compose into one (`exactStep_add`).
- **Constant gas:** the closed form solves the equation and is its only solution (`linkExact_deriv`, `linkExact_unique`); the head opens exactly at `t = (RTI/sqrt u) ln((Tg - T0)/(Tg - Ta))` (`linkExact_activation`), is shut before (`below_before_activation`), never opens if the gas does not exceed the activation temperature (`never_opens`), opens earlier in hotter gas (`activation_antitone`) and in proportion to the RTI (`activation_rti`).
- **Evans:** the decay never rises and never exceeds the heat release at activation (`decay_antitone`, `decay_le`), its time constant falls with more water (`evansTau_antitone`), and the heat released after activation is bounded however long the run (`evans_energy`).

Still to prove before steps 3 and later (physics report section 9): the spray's water budget, the evaporation energy identity (the form of `chemistry_liquid_energy`), species validity after adding vapour, oxygen dilution scaling the burn, and two cases where the obvious theorem is false and must be recorded as counterexamples (cooling can raise a cell's stoichiometric cap; steam at a hot dry fuel surface raises its pyrolysis).

## 10. Order of work and the user's decisions

1. Sprinkler record, IFC reading and the per-room layout with its rule report (no simulation).
2. Activation on both readings, card only.
3. Evans cap.
4. Spray zone with evaporation; then manual suppression.

Each step behind its switch, with the timing check before and after, and its Lean items proved first. The user asked (2026-11-07) to release the whole feature once it is built, not step by step to the public.

**Decisions for the user:**

- decided (2026-11-07): pass SBC 801-CR-2018, with IBC 2021 where stricter, until the SBC 2024 texts are obtained;
- the university building: deferred by the user to a future session (2026-11-07); then, which drawings or IFC can be used, and whether they may be committed or must stay local;
- the assembly-point check as proposed in section 7;
- whether the first release shows placed heads only (step 1) before any simulation.
