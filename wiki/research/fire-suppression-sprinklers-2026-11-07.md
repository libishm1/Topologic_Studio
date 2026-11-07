# Sprinklers and fire suppression: codes, placement on an IFC model, simulation, and what is proved (2026-11-07)

Status: research and plan, for the user's review before any of it is built. Nothing below is implemented in the app except the Lean specifications. The detailed sources are four companion reports:

- `sprinkler-codes-2026-11-07.md`: when the codes require sprinklers, NFPA 13 placement, hydraulics, response classes, outdoor assembly points (59-row value table with sources);
- `sprinkler-physics-2026-11-07.md`: activation, the coarse-grid problem, suppression models, data per sprinkler, the spray, validation data, statements to prove;
- `sprinkler-ifc-mep-2026-11-07.md`: open IFC models with fire protection, the IFC schema, federation of an MEP model;
- `smoke-transport-2026-11-07.md`: smoke under ceilings, ducts, open and closed buildings, stack effect (pending, see section 8).

Earlier notes: `fire-suppression-note-2026-11-05.md`, `sbc-firefighting-future-notes-2026-11-05.md`.

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

The triggers below are therefore SBC 2018 values (where read) or IBC/IFC 2021 values. Both are **to be confirmed against SBC 801-CR-2024 and SBC 201-2024** before the app cites them as the current code.

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

Pending the smoke-transport report. The solver already computes buoyant flow, so hot smoke rising and pooling under ceilings is computed, not assumed; the two-zone room model gives layer height and temperature; the ventilation option moves smoke between air terminals. Not modelled: the stack effect in tall buildings and wind.

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

- whether to use the SBC 801-CR-2018 values (and IBC/IFC 2021 where 2018 was not read) as provisional until the SBC 801-CR-2024 and SBC 201-2024 texts are obtained (the English editions are sold);
- the university building: deferred by the user to a future session (2026-11-07); then, which drawings or IFC can be used, and whether they may be committed or must stay local;
- the assembly-point check as proposed in section 7;
- whether the first release shows placed heads only (step 1) before any simulation.
