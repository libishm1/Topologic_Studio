# Open-source fire evacuation and egress software: a method study and the gaps it shows in Topologic Studio Next (2026-10-28)

Research note. No file in the repository was changed. No third-party code was built, installed or run: source was read at pinned commits, and the two closed-source tools were studied only from their public documentation and papers. Evidence IDs in square brackets (for example [F6]) point to the ledger in section 6.

## 1. Summary

We read the source of six open or source-available tools (FDS+Evac in FDS 6.7.7, CFAST, JuPedSim with its former JPSfire module, Vadere, Menge and MomenTUM v2) and the public documentation of two closed tools (Pathfinder and buildingEXODUS), and compared them with Studio Next's single-evacuee router.

Two findings favour Studio. First, none of the six open tools imports IFC: within the searches we ran, their inputs are FDS input files, Shapely polygons or a new multi-level walkable surface (JuPedSim), JSON scenarios with an OpenStreetMap converter (Vadere), XML and navigation meshes (Menge, MomenTUM) [F13, J9, V9, M7, T6]. Only the two commercial tools document IFC import [PF6, BE1]. Second, no studied tool states or proves properties of its router; Studio's exactness and conservativeness results in Lean (gap LG3) have no counterpart here.

The five largest gaps in Studio, in rank order, are these. The full ranking with impact, effort, fit and a Lean statement for each is in section 7.

1. **G1 Stairs at level walking speed.** Every link, stairs included, is walked at one speed (default 1.2 m/s) over its 3D length [S1, S2]. The references slow walkers on stairs to between 0.45 and 0.88 of their level speed [F11, P4, PF1]. Stair travel time is therefore underestimated, by up to about half on the slowest settings.
2. **G2 No slowing in smoke.** FDS+Evac, JPSfire and buildingEXODUS reduce speed with the smoke extinction coefficient [F6, P1, BE1]. Studio's zone model has no soot, and its routers use a constant speed [S7].
3. **G3 Dose from convective heat only.** FDS+Evac, CFAST, JPSfire and Pathfinder compute the asphyxiant-gas FED (CO, HCN, CO2 hyperventilation, low O2) [F7, C2, P2, PF5], and CFAST also adds radiant heat [C3]. Studio's dose is the convective-heat term alone [S5]. Studio's tolerance-time constants (ISO 13571, 5e7 T^-3.4 min) also differ from CFAST's (4.1e8 T^-3.61 min): at 100 °C they give 7.9 and 24.7 min.
4. **G4 No door capacity or queues.** With a population, door flow (about 1.3 persons per second per metre of effective width) and queues set egress time [F8, PF1]. A single evacuee walking on an empty graph gives a lower bound, not an RSET.
5. **G5 One departure time instead of pre-movement distributions and occupant types.** FDS+Evac draws detection and pre-movement times from nine distribution types and gives occupants class-specific speeds and body sizes [F9, F10]. Studio has one "leave after" value and one speed [S6].

Lifts without waiting (G6), route choice that assumes perfect foresight (G7), the lack of a standard test harness (G8) and path clearance (G9) follow.

Overlap with the existing gap list is partial. LG2 (FED as a route objective) and LG8 (the two-zone model) are the foundations that G2 and G3 would extend, but neither covers smoke speed or gas toxicity. G1, G4 and G5 are new to the LG list, although the methods note (`fire-egress-methods-2026-10-27.md`, section 6 and the model map) recommended stair speeds, pre-movement distributions and flows without building them. G6 to G9 are new.

G1, G2, G3, G5 and G6 can each be built so that every link's arrival time stays FIFO and every dose rate stays non-decreasing in time. Under those conditions the existing Forecast and Deadline results should carry over once link time is generalised from length over speed to a per-link non-decreasing function. G4 in full (flows over time) does not fit a single-evacuee router.

Standard tests:

- **RiMEA 4.1.1 (16 tests).** Studio can run four today (1, 6 per evacuee, 10 and 14). Tests 2 and 3 run today only with one global speed. Tests 5 and 7 need G5, test 8 is partial, and seven need crowd dynamics (4, 9, 11, 12, 13, 15, 16).
- **NIST TN 1822 (17 tests).** Studio can run four today (2.1, 2.3, 3.1, 4.1) and 2.6 in part. Seven need one of the gaps above (1.1, 2.2, 2.4, 2.5, 2.7, 3.3, and the single-evacuee part of 2.10). Five need crowds (2.8, 2.9, 3.2, 5.1, 5.2).
- **ISO 20414:2020.** We did not access the text; its published scope lists components consistent with TN 1822's five core components [I1]. The ISO numbering should be checked against the standard before any claim of conformity.

Eleven openly licensed IFC models were added under `C:\Users\lmurugesan\ifc-samples\more2\` (section 10). They include lifts as IfcTransportElement, ramps, stairs drawn as one stepped solid, a building whose only vertical links are lifts, and the only file in the corpus with FireExit flags.

## 2. Method

The method follows the reverse-engineer-anything (REA) skill (https://github.com/morluto/rea, MIT; skill text at `.agents/skills/reverse-engineer-anything/SKILL.md` and `references/evidence-workflows.md`, read on 2026-10-28). Only the method was adopted; REA and its tools were not installed or run. In REA's terms:

- the request was first turned into a checklist of questions, each naming the evidence its answer needs (below);
- the result is written summary first;
- every conclusion cites evidence IDs, and the ledger separates what was observed in source from what documentation states and from what we infer;
- absence claims are labelled as bounded searches, because a missing match in partial evidence does not establish absence;
- the checklist is revisited at the end, and each question is marked answered, partial or unresolved.

Sources were obtained as follows:

- **Read-only copies outside OneDrive.** Shallow clones of Menge, MomenTUM, JuPedSim and Vadere, and raw single-file downloads of FDS 6.7.7 (`evac.f90`, `ieva.f90`, `func.f90`), CFAST (seven files) and JPSfire (three files and `Pedestrian.cpp` from JuPedSim v0.8.4), all under `C:\Users\lmurugesan\oss-study\`. Line numbers come from `grep -n` and `sed -n` on these copies.
- **Documentation and standards** were read with WebFetch: the Pathfinder 2026.1 technical reference and V&V pages, the FSEG buildingEXODUS capability page, NIST TN 1822 (public domain) and the RiMEA 4.1.1 guideline (CC BY-ND 4.0).
- **DOIs** were checked against Crossref on 2026-10-28. One candidate DOI resolved to an unrelated paper and was dropped.
- **Fetched content** was treated as data only.

Pinned versions:

| Target | Repository and version read | Licence (evidence) |
|---|---|---|
| FDS+Evac | github.com/firemodels/fds, tag FDS6.7.7 (ref 89e78fe7594d), Source/evac.f90 (16,723 lines), ieva.f90, func.f90 | NIST public-domain notice, LICENSE.md [F1] |
| CFAST | github.com/firemodels/cfast, commit 5051acb3460b | NIST public-domain notice, LICENSE.md [C1] |
| JuPedSim | github.com/PedestrianDynamics/jupedsim, commit 189241a2cea6 (2026-10-27); JPSfire at tag v0.8.4 (968689b635e2) | LGPL-3.0-or-later, LICENSE and SPDX headers [J1] |
| Vadere | gitlab.lrz.de/vadere/vadere, commit 4cf5531421a3 (2026-10-27) | LGPL-3.0, LICENSE [V1] |
| Menge | github.com/MengeCrowdSim/Menge, commit 00f97eaec765 (2019-12-30) | Apache-2.0, LICENSE.txt [M1] |
| MomenTUM v2 | github.com/tumcms/MomenTUM, commit 2338947dff94 (v2.0.2) | Custom licence: non-commercial research and education use only, "further rights are not granted"; source-available, not an OSI open-source licence [T1] |
| Pathfinder | Documentation only, thunderheadeng.com/docs/2026-1/pathfinder | Commercial; not open source |
| buildingEXODUS | Documentation and papers only, fseg.gre.ac.uk | Commercial; not open source |

### Question checklist and final status

| Q | Question | Evidence needed | Status |
|---|---|---|---|
| Q1 | Licence of each target | LICENSE file or licence page | Answered for all eight |
| Q2 | Movement model and its equations | Source file and line | Answered for FDS+Evac, JuPedSim, Vadere and Menge. Partial for MomenTUM (models listed, equations not read). Pathfinder: documented, with equations missing from the extracted text. buildingEXODUS: capability level only |
| Q3 | Routing and route choice | Source or documentation | Answered for all |
| Q4 | Stairs | Source or documentation | Answered for all. MomenTUM has none in a bounded search |
| Q5 | Doors: flow capacity and queueing | Source or documentation | Answered. Explicit door capacity only in Pathfinder SFPE mode and in FDS+Evac's door choice; the others make door flow emerge from the movement model |
| Q6 | Lifts | Source or documentation | Answered: FDS+Evac, Pathfinder and buildingEXODUS have lift models; the others have none (bounded) |
| Q7 | Pre-movement and heterogeneity | Source or documentation | Answered for FDS+Evac, Vadere and JuPedSim. Partial for Pathfinder and buildingEXODUS |
| Q8 | Fire and smoke coupled to movement (FED, visibility, speed in smoke) | Source or documentation | Answered for FDS+Evac, JPSfire, CFAST (fire side only), Vadere (none, bounded), Menge (none, bounded) and MomenTUM (none, bounded). Partial for Pathfinder (FED documented; its effect on movement not stated) and buildingEXODUS (capabilities listed, equations not public in what we read) |
| Q9 | Geometry import | Source or documentation | Answered |
| Q10 | Published verification and validation | Repository test folders, V&V manuals | Answered for FDS+Evac (by reference only), JuPedSim, Vadere and Pathfinder. Partial for buildingEXODUS. Unresolved for Menge and MomenTUM (none found) |
| Q11 | Performance | Published timings or time steps | Partial: time steps and design choices only; no tool was run, so no timings were measured |
| Q12 | What a browser-first single-evacuee router can adopt, and whether it can be stated for Lean 4 | Our code plus the above | Answered in section 7 |

## 3. The baseline: Studio Next as read for this study

- **One evacuee, one speed.** The router walks one evacuee over the lattice graph at one speed, `speed = 1.2` m/s by default in every router [S1]. Link length is the 3D Euclidean distance between nodes [S2], so a stair link costs its pitch-line length at level speed.
- **Stairs.** Stairs with modelled treads become walking-line nodes [S3]. Stair cores without treads get an assumed dog-leg walking line with 175 mm risers and 280 mm goings, so the path length is right but the speed is still the level speed [S3].
- **Lifts.** A lift is linked only when `useLifts` is set (off by default). A ride then costs its rise as if walked, with no waiting [S4].
- **Dose.** The dose is the convective-heat FED, at rate 1 / (5e7 T^-3.4 min), from the forecast's running-maximum heat [S5]. The two-zone model (LG8) supplies a gas temperature and a layer height, but no species, soot or radiation [S7].
- **Pre-movement.** Pre-movement is one scalar, "leave after", 0 s by default [S6]. The deadline map computes, for every node, the latest safe departure.
- **Speed.** Planning took at most 38 ms, and the deadline map at most 6.4 s, on graphs of up to 64,000 nodes (gap-analysis README).

## 4. Comparison table

Table 1 covers the open-source tools and Table 2 the source-available and closed ones. Pathfinder and buildingEXODUS entries rest on documentation, not source.

**Table 1. Studio Next and the open-source tools**

| Aspect | Studio Next | FDS+Evac (FDS 6.7.7) | JuPedSim 1.x (JPSfire in 0.8.4) | Vadere |
|---|---|---|---|---|
| Licence | Repository's own | NIST public domain | LGPL-3.0+ | LGPL-3.0 |
| Movement | Graph walk at constant speed; one evacuee | Social force, three-circle bodies, anisotropic, with contact forces [F3, F4, F5] | Collision-free speed model and others: GCFM, social force, AVM, warp driver [J2, J3] | Optimal Steps Model on an eikonal potential; also SFM, GNM, BHM, CA [V2, V3] |
| Routing | Exact time-dependent Dijkstra against a forecast; dose Pareto front; deadline map | Per-exit flow fields; door choice by walk plus queue time, filtered by known, visible and smoke-free doors [F8] | Shortest path on a triangulation; static floor fields; journeys with fixed, round-robin or least-targeted transitions [J4, J5] | Floor field from an eikonal solve, optionally density-dependent [V3] |
| Stairs | Walking line, level speed [S1, S3] | STRS objects; up and down speed factors by occupant class [F11] | 3D walkable surface with inclined connectors up to 50° [J6]; 0.8.4 had up and down stair speeds of 0.6 m/s [P4] | Stairs with treads; OSM steps land on treads [V5] |
| Doors and queues | None | Queue time in door choice; flow emerges from the forces [F8] | Emerges from the model | Emerges; queuing time-cost options [V3] |
| Lifts | Walked rise, no wait [S4] | CORR with ELEVATOR=T, travel time and fill wait [F12] | None (bounded) | None (bounded) |
| Pre-movement and heterogeneity | One departure, one speed [S6] | Detection plus pre-movement, nine distributions; occupant classes [F9, F10] | Per-agent v0 and parameters; distributions module | Speed N(1.34, 0.26) clipped to [0.5, 2.2] m/s [V6]; psychology layer with threat stimuli [V10] |
| Fire coupling | Forecast heat, convective FED, deadline map | Gas FED (CO, CO2, O2, HCN, NOx, irritants); incapacitation at FED 1; speed falls with extinction coefficient [F6, F7] | JPSfire: speed in smoke (three correlations); FED gas and heat plus FIC as output only; smoke-weighted route graph [P1, P2, P3] | None (bounded) [V8] |
| Geometry import | IFC in the browser | FDS input (OBST and EVAC meshes) [F13] | Shapely polygons; new WalkableSurface; no IFC (bounded) [J9] | JSON scenarios; OSM converter [V9] |
| V&V published | Property tests, Lean, 29-model cross-checks | IMO tests (FDS+Evac guide; not read here) | RiMEA tests 1 to 15 in 0.8.4; notebooks for RiMEA-style cases [P5] | RiMEA 1, 4 to 14 scenarios [V7] |
| Time step and performance | 38 ms plan, 6.4 s map at 64k nodes | dt 0.001 to 0.01 s [F14] | dt 0.01 s default [J7] | Event-driven OSM; OpenCL variant [V2] |

**Table 2. CFAST, the source-available frameworks and the closed tools**

| Aspect | CFAST | Menge | MomenTUM v2 | Pathfinder (documentation) | buildingEXODUS (documentation) |
|---|---|---|---|---|---|
| Licence | NIST public domain | Apache-2.0 | Non-commercial only | Commercial | Commercial |
| Movement | Not an egress model | ORCA, PedVO, Helbing, Johansson, Karamouzas, Zanlungo, GCF plugins [M3, M5] | Social force (Helbing, Zeng), Moussaïd heuristic, CA, LWR and others [T2] | SFPE flow mode or inverse steering [PF1, PF2] | Fine node and arc grid with behaviour rules [BE1] |
| Routing | Not applicable | A* over navigation-mesh portals of a minimum width [M2] | Dijkstra, perceived-cost Dijkstra, cognitive and unified routing [T3] | A* on a navigation mesh; door choice by "locally quickest", re-evaluated each second [PF3] | Nearest serviceable or most familiar exit [BE1] |
| Stairs | Not applicable | Navigation-mesh elevation [M4] | None found (bounded) [T5] | SFPE k by riser and going; up and down curves [PF1, PF2] | Stairs and escalators [BE1] |
| Doors and queues | Not applicable | Emerges | Angular queuing model [T4] | SFPE door delay timer; room density limit 3.55 per m² [PF1] | Waiting is supervised in the movement submodel [BE1] |
| Lifts | Not applicable | None | None (bounded) | Call groups, open and close delays, travel times [PF4] | Agent-based lift model with dispatch strategies [BE3] |
| Pre-movement and heterogeneity | Not applicable | Per-agent profiles | Generators and per-agent parameters | Profiles; pre-movement delays | Response time, age, gender, agility [BE1] |
| Fire coupling | Two-zone fire; target FED gas and heat, including radiation [C2, C3] | None (bounded) [M6] | None (bounded) [T5] | Gas FED (CO, CO2, O2; no HCN) from FDS [PF5] | Two FED models, irritants, heat and radiation; smoke with crawling; reads CFAST and SMARTFIRE output [BE1] |
| Geometry import | CEdit and text input | XML and navigation mesh [M7] | XML layout; CityGML through a separate FME workflow [T6] | IFC (since 2018.4), DXF, DWG, FDS and others [PF6] | DXF, IFC, SMARTFIRE, FDS geometry [BE1] |
| V&V published | Verification and Validation Guides (NIST TN 1889 series) [C9] | None found | None found | NIST TN 1822 tests, IMO, SFPE examples, fundamental diagrams [PF7] | Staged protocol and full-scale validations [BE1] |
| Time step and performance | Stiff ODE (DASSL) [C6] | Multithreaded A* caching [M2] | Not measured | Not measured | Not measured |

## 5. Findings by target

**FDS+Evac (FDS 6.7.7, the last tagged release that contains it).**

- *Status.* `Source/evac.f90` exists at tag FDS6.7.7 (released 2021-11-19) and is gone at FDS6.7.8 (2022-05-25) [F2]. Comments from December 2020 describe a split of the fire run from the evacuation run, which exchange data through a FED file [F13].
- *Movement.* Agents are three-circle bodies driven by a social force. The motive term is (m/τ)(v0 e − v), solved semi-implicitly [F4]. Agent-agent repulsion is A·exp(−(d − r)/B)/d, weighted by the anisotropy λ + ½(1 − λ)(1 + cos φ) [F3, F5].
- *Smoke.* The desired speed is multiplied by c(K) = 1 + (β/α)K, with Frantzich and Nilsson's α = 0.706 m/s and β = −0.057 m²/s. K is the mass extinction coefficient times the local soot density. The factor is floored at 0.1 by default [F6].
- *Dose.* The dose is Purser's gas FED (CO, NOx, HCN, irritants, a CO2 hyperventilation multiplier, low O2), integrated per agent [F7]. FED ≥ 1 incapacitates the agent [F7]. Gas temperature and radiant flux are stored, but they do not enter the dose: a bounded search found no use after line 7300.
- *Door choice.* Each door gets an estimated time α·d/v + (1 − α)·min(1.5π d²/(F_q W), N_q/(F_q W)), with F_q = 1.3 persons per second per metre and α = 0.5 [F8]. Doors are kept only if they are known or visible, and only if the FED accrued along the straight line to them is below a threshold [F8].
- *Occupants.* Detection and pre-movement times are drawn from nine distribution types, and agents move only after T_det + T_pre [F9]. Body classes have mean speeds of 1.25 (adult), 1.35 (male), 1.15 (female), 0.90 (child) and 0.80 (elderly) m/s [F10].
- *Stairs and lifts.* Stairs scale speed by up and down factors, for example 0.452 and 0.683 for the IMO male-under-30 class [F11]. Lifts are CORR objects with a travel time and a five-second wait to fill [F12].

**CFAST.** CFAST has no egress model, but it is the reference for LG8 and for the gas and visibility data that G2 and G3 need.

- *Targets.* For each target it accumulates a gas FED: CO and HCN terms, multiplied by exp(%CO2/5) [C2]. It also accumulates a heat FED: convective 4.1e8 T^-3.61 min, plus a radiant term once the incident flux reaches 2.5 kW/m² [C3]. It records the soot concentration at the target [C4].
- *Fire and layers.* The plume follows Heskestad [C5]. The ODE system is solved with DASSL [C6], vent flow is integrated in slabs [C7] and wall conduction is one-dimensional [C8].
- *Use for Studio.* CFAST's species and soot outputs are exactly the inputs that Studio's zone model lacks.

**JuPedSim and JPSfire.**

- *Current JuPedSim (LGPL).* The library is C++ with Python bindings. The collision-free speed model sets speed to min(max(s/T, 0), v0), where s is the spacing ahead and T the time gap, and steers by exponential repulsion [J2].
- *Routing.* Routing uses a triangulation or static floor fields. The floor fields are travel-time grids with 0.2 m cells and speed reduced near walls [J4, J5]. Route plans are journeys of stages: waypoint, exit and direct steering [J8].
- *Multi-level geometry.* A new `WalkableSurface` stacks floors joined by inclined connectors (stairs or ramps) with slopes up to 50° [J6]. It is the closest open format to what Studio's lattice already builds, and so is a plausible export target for an offline crowd cross-check.
- *JPSfire (removed after 0.8.x).* Module B slows agents with Frantzich and Nilsson (floor 0.3 m/s), with Jin 1978 (separate fits for irritant and non-irritant smoke) or with Fridolf 2018 [P1]. Module C writes gas FED, heat FED and HCl FIC as output only and does not incapacitate agents [P2]. Module A raises route-graph edge weights by 1 + (1 − risk tolerance) × smoke factor [P3].
- *Stairs in 0.8.4.* Default up and down stair speeds were 0.6 m/s, blended smoothly at the stair ends [P4].

**Vadere (LGPL).**

- *Movement.* The Optimal Steps Model chooses the next position on a circle whose step length is 0.4625 + 0.2345 v_free plus noise [V2]. The target potential solves the eikonal equation, with an optionally density-dependent travel-time cost [V3]. A Weidmann speed adjuster is available [V4].
- *Stairs.* Stairs carry treads with depths of 0.10 to 0.35 m, and a dedicated optimiser places steps on treads [V5].
- *Occupants.* Free speeds default to N(1.34, 0.26), clipped to [0.5, 2.2] m/s [V6].
- *Tests.* The repository ships RiMEA scenarios 1 and 4 to 14 [V7].
- *Fire.* A bounded search found no fire, smoke or FED code. The "airflow" model serves aerosol transmission [V8].

**Menge (Apache-2.0).**

- *Movement.* Agents use local avoidance plugins, ORCA among them [M3, M5]. The Helbing plugin uses a driving force (v_pref − v)·m/τ and exponential obstacle repulsion [M3].
- *Routing.* Global routing is A* over navigation-mesh portals no narrower than the agent's width [M2], a simple clearance rule that Studio lacks (G9). The navigation mesh carries elevation [M4].
- *Fire.* No fire coupling was found (bounded) [M6].

**MomenTUM v2 (non-commercial licence).**

- *Layers.* The framework separates strategic, tactical and operational layers [T2]. The tactical routing models include a perceived-cost Dijkstra that scales edge costs by a per-pedestrian Gaussian factor [T3], and there is an angular queuing model [T4].
- *Coverage.* A bounded search of 750 Java files found no stairs, lifts or fire [T5].
- *Licence.* Its licence forbids commercial use, so ideas can be cited but code must not be reused [T1].

**Pathfinder and buildingEXODUS** are covered in section 9.

## 6. Evidence ledger

Types: **SRC**, observed in source at the pinned version; **DOC**, stated in documentation or a paper; **INF**, inferred by us. Confidence is high, medium or low. Paths are relative to the pinned repositories unless they start with `studio:` (Studio Next's `topologicpy-web-frontend/src/`) or `wiki:`.

| ID | Claim | Source | Type | Conf. |
|---|---|---|---|---|
| S1 | Studio's routers default to one speed of 1.2 m/s on every link; the UI default is also 1.2 m/s | studio:core/forecast.js:144, 191, 294, 314; core/deadline.js:64; state/useStudio.js:60 | SRC | high |
| S2 | Link cost is 3D Euclidean length; there is no per-kind speed | studio:core/route.js:138-152 | SRC | high |
| S3 | Treads become walking-line nodes; tread-less stair cores get an assumed 175/280 mm dog-leg line | studio:core/stairs.js:1-22; core/cores.js:14-19 | SRC | high |
| S4 | Lift links cost their length like any other link, with no waiting; lifts are linked only when `useLifts`, which is off by default | studio:core/lattice.js:810-814, 433 | SRC | high |
| S5 | The dose is convective-heat FED, rate T^3.4/(5e7·60) per second | studio:core/forecast.js:31-33, 51-54 | SRC | high |
| S6 | Pre-movement is one value, `departAfter`, 0 s by default | studio:state/useStudio.js:63; App.jsx:1178 | SRC | high |
| S7 | The zone model has no species, soot or radiant heat | wiki:research/zone-model-spec-2026-10-28.md, section 8 (line 66) | DOC | high |
| F1 | The FDS licence is the NIST public-domain notice | fds@FDS6.7.7 LICENSE.md:1-5 | SRC | high |
| F2 | evac.f90 is present at FDS6.7.7 (released 2021-11-19) and absent at FDS6.7.8 (2022-05-25) | raw.githubusercontent.com HTTP 200 and 404; GitHub releases API | SRC | high |
| F3 | Social repulsion A·cosφ-factor·exp(−(d − r)/B)/d, with anisotropy λ + ½(1 − λ)(1 + cos φ) | evac.f90:9413-9421, 9466-9469 | SRC | high |
| F4 | Motive force (m/τ)(v0 e − v) with a semi-implicit velocity update | evac.f90:10134-10151 | SRC | high |
| F5 | Agents are three circles (torso and two shoulders) | evac.f90:9423-9431 | SRC | high |
| F6 | Speed factor 1 + (β/α)K, with α = 0.706 and β = −0.057 (Frantzich and Nilsson 2003) and K from soot density; floor 0.1·v0 by default; applied as V0_FAC | evac.f90:1543-1546, 2154-2156, 8226-8261, 8300 | SRC | high |
| F7 | Gas FED integrand: CO (activity-dependent), NOx, HCN, irritants, CO2 multiplier, O2; no heat term; an agent with FED ≥ 1 is marked dead | func.f90:4178-4269; evac.f90:8214-8222, 8125-8127 | SRC | high |
| F8 | Door choice cost α·d/v + (1 − α)·queue time, with F_q = 1.3 /m/s and α = 0.5; FED or extinction screen on doors; known and visible door sets | evac.f90:1569-1572, 16309-16324, 15907-15921 | SRC | high |
| F9 | Detection plus pre-movement times from nine distributions; movement gated by T > T_pre + T_det; smoke-triggered detection | evac.f90:14696-14720, 7677, 8154, 8263-8267 | SRC | high |
| F10 | Class mean speeds: Adult 1.25, Male 1.35, Female 1.15, Child 0.90, Elderly 0.80 m/s | evac.f90:1851-1950 | SRC | high |
| F11 | Stair speed factors by class, e.g. up 0.452 and down 0.683 for IMO male under 30; Z interpolated along the stair | evac.f90:1725-1727, 11458-11500 | SRC | high |
| F12 | Lift as CORR with ELEVATOR=T, TRAVEL_TIME and a 5 s wait to fill | evac.f90:3268-3277 | SRC | high |
| F13 | Fire and evacuation runs decoupled through a FED file (2020 change); geometry from FDS input; no "ifc" string in evac.f90 (bounded) | evac.f90:2-38; grep count 0 | SRC | high |
| F14 | Evacuation time step between 0.001 and 0.01 s | evac.f90:1535-1536 | SRC | high |
| C1 | CFAST licence is the NIST public-domain notice | cfast@5051acb LICENSE.md:1 | SRC | high |
| C2 | Target gas FED: (dt/60)·exp(%CO2/5)·(CO ppm/35000 + HCN ppm^2.36/1.2e6) | Source/CFAST/target.f90:131-138 | SRC | high |
| C3 | Target heat FED: convective 4.1e8 T^-3.61 min, plus radiant 4.2 q^-1.9 min when q ≥ 2.5 kW/m² | target.f90:139-146 | SRC | high |
| C4 | Soot concentration at the target is recorded (fed_obs) | target.f90:147 | SRC | high |
| C5 | Heskestad plume entrainment, constrained for stable stratification | fire.f90:359-371, 450-465 | SRC | high |
| C6 | ODE system solved by DASSL | solve.f90:16, 492 | SRC | high |
| C7 | Slab-integrated vent flow | flowhorizontal.f90:408-543 | SRC | high |
| C8 | One-dimensional wall conduction | conduction.f90:32-87 | SRC | high |
| C9 | CFAST 7 Technical Reference Guide is NIST TN 1889v1 | Crossref 10.6028/NIST.TN.1889v1 | DOC | high |
| J1 | JuPedSim licence LGPL-3.0-or-later | jupedsim@189241a LICENSE:1-5; SPDX headers in libsimulator/src | SRC | high |
| J2 | CSM optimal speed min(max(s/T, 0), v0); exponential neighbour and wall repulsion | libsimulator/src/operational_models/collision_free_speed_model/collision_free_speed_model.cpp:121-126, 150-167 | SRC | high |
| J3 | Operational models: CSM (v1 to v3), GCFM, SFM, AVM, warp driver, custom | libsimulator/src/operational_models/ (listing) | SRC | high |
| J4 | Way finding by shortest path through a triangulation; journeys with fixed, round-robin and least-targeted transitions | jupedsim.org/docs/concepts/routing.rst:152-210, 273-283 | DOC | high |
| J5 | Floor-field routing engine: travel-time grids, 0.2 m cells, speed reduced near walls | libsimulator/src/floorfield_routing_engine.hpp:17-43 | SRC | high |
| J6 | WalkableSurface: regions stacked into floors and joined by stairs or ramps; connectors inclined up to 50° | python_modules/jupedsim/jupedsim/simulation.py:96-99; libsimulator/src/geometry/walkable_surface.hpp:36-44; geometry/validation.cpp:24-27 | SRC | high |
| J7 | Default dt 0.01 s | python_modules/jupedsim/jupedsim/simulation.py:77 | SRC | high |
| J8 | Stage types: waypoint, exit, direct steering | simulation.py:134, 156, 178 | SRC | high |
| J9 | No IFC import; a search for the word "ifc" in .py, .cpp, .hpp, .md and .rs files outside third-party returned nothing (bounded) | grep over the clone | SRC | medium |
| P1 | JPSfire speed in smoke: Frantzich and Nilsson with floor 0.3 m/s; Jin 1978 irritant and non-irritant fits; Fridolf 2018 bounded to [0.2, 1] m/s | jupedsim@v0.8.4 JPSfire/B_walking_speed/WalkingSpeed.cpp:109-141 | SRC | high |
| P2 | JPSfire FED_In (CO, HCN, O2, CO2 multiplier), FED_heat with tolerance 2e31 T^-16.963 + 4e8 T^-3.7561 min, and HCl FIC are written out only | JPSfire/C_toxicity_analysis/ToxicityAnalysis.cpp:122-157, 189-195 | SRC | high |
| P3 | Smoke sensor sets edge weight 1 + (1 − RiskTolerance)·SmokeFactor on the cognitive-map graph | JPSfire/A_smoke_sensor/SmokeSensor.cpp:105-128 | SRC | high |
| P4 | Default up and down stair speeds 0.6 m/s, smoothly blended at the stair ends | pedestrian/Pedestrian.cpp@v0.8.4:75-76, 585-650 (downstairs branch from 596, upstairs from 626) | SRC | high |
| P5 | RiMEA tests 1 to 15 shipped in Utest/rimea_tests | GitHub tree of v0.8.4 | SRC | high |
| V1 | Vadere licence LGPL-3.0 | vadere@4cf5531 LICENSE | SRC | high |
| V2 | OSM step length 0.4625 + 0.2345·v_free + N(0, sd); OpenCL variants exist | VadereSimulator/.../models/osm/PedestrianOSM.java:95-98; VadereState/.../AttributesOSM.java:112, 118; models/osm/opencl/ | SRC | high |
| V3 | Eikonal floor field, with dynamic travel-time costs from density and queuing | .../models/potential/solver/calculators/EikonalSolver.java:12-41; models/potential/timeCostFunction/ | SRC | high |
| V4 | Weidmann speed 1.34(1 − exp(−1.913(1/ρ − 1/5.4))) | .../models/osm/SpeedAdjusterWeidmann.java:31-32 | SRC | high |
| V5 | Stairs with treads 0.10 to 0.35 m deep; stair step optimiser | VadereState/.../scenario/Stairs.java:21-52; .../osm/stairOptimization/StairStepOptimizer.java:26-41 | SRC | high |
| V6 | Speed distribution mean 1.34, SD 0.26, clipped to [0.5, 2.2] m/s | VadereState/.../AttributesAgent.java:18-24 | SRC | high |
| V7 | RiMEA scenarios 1 and 4 to 14 in the TestOSM project; 2, 3, 15 and 16 not in that folder (bounded) | Scenarios/ModelTests/TestOSM/scenarios/ (listing) | SRC | high |
| V8 | No smoke, FED or toxicity code in .java files (bounded); "airflow" is for aerosols | grep over the clone; models/airflow, models/infection | SRC | medium |
| V9 | Own JSON scenario format; OpenStreetMap converter | Tools/Converters/osm2vadere/ | SRC | high |
| V10 | Psychology layer with threat stimuli and cognition models | VadereState/.../perception/types/Threat.java; AttributesThreatCognitionModel.java | SRC | medium |
| M1 | Menge licence Apache-2.0 | menge@00f97ea LICENSE.txt | SRC | high |
| M2 | A* over navigation-mesh portals with a minimum width; A* data cached per thread | src/Menge/MengeCore/resources/PathPlanner.cpp:132-141, 243 | SRC | high |
| M3 | Helbing plugin: mass 80 kg, driving force (v_pref − v)m/τ, exponential obstacle repulsion | src/Plugins/AgtHelbing/HelbingAgent.cpp:58, 170, 196 | SRC | high |
| M4 | Navigation mesh elevation per node | src/Menge/MengeCore/resources/NavMesh.h:267-273 | SRC | high |
| M5 | Agent plugins GCF, Helbing, Johansson, Karamouzas, Zanlungo; ORCA in core; StressGas adjusts parameters under stress | src/Plugins (listing); src/Plugins/StressGas/AgentStressor.h:41-49 | SRC | high |
| M6 | No fire or smoke model; "fire" matches are event triggers (bounded) | grep over src | SRC | medium |
| M7 | No IFC import (bounded) | grep over src | SRC | medium |
| T1 | MomenTUM licence allows non-commercial research and education only; "further rights are not granted" | momentum@2338947 Licence.txt:1-20 | SRC | high |
| T2 | Strategic, tactical and operational model packages | momentum-model/src/main/java/tum/cms/sim/momentum/model/ (listing) | SRC | high |
| T3 | Perceived-cost Dijkstra with a per-pedestrian Gaussian factor | .../model/tactical/routing/dijkstraPerceivedCostModel/DijkstraPerceivedCostTactical.java:65-86 | SRC | high |
| T4 | Angular queuing model | .../model/tactical/queuing/angularQueuingModel/ | SRC | high |
| T5 | No stairs, lifts, escalators or fire in 750 Java files (bounded) | grep over the clone | SRC | medium |
| T6 | XML layout input; CityGML through an external FME workflow (tum-gis repository) | README.md:43; github.com/tum-gis/momenTUM-layout-from-citygml | DOC | medium |
| PF1 | SFPE mode: door delay timer, max room density 3.55 /m², density clamped to [1.9, 3.0] for door flow, stair k from the SFPE table by slope | https://www.thunderheadeng.com/docs/2026-1/pathfinder/appendices/technical-reference/sfpe | DOC | medium |
| PF2 | Steering mode: inverse steering over candidate directions, Fruin density, separate up and down stair curves, Reynolds-style acceleration | .../technical-reference/steering/ | DOC | medium |
| PF3 | A* on a triangulated navigation mesh with string pulling; "locally quickest" door choice using room travel time, queue time, global travel time and distance, re-evaluated every 1 s | .../technical-reference/pathfinding/ | DOC | medium |
| PF4 | Elevators: call groups, open and close delays, pickup and discharge travel times set by the user, loading density nominal_load/area | .../technical-reference/elevators/ | DOC | medium |
| PF5 | FED from CO, CO2 and O2 following FDS+Evac, sampled at 90% of occupant height; excludes HCN; effect on movement not stated | .../technical-reference/fed/ | DOC | medium |
| PF6 | IFC import added in Pathfinder 2018.4; model generation from BIM | https://support.thunderheadeng.com/release-notes/pathfinder/2018/2018-4 ; https://www.thunderheadeng.com/docs/2026-1/pathfinder/geometry/generating-from-cad | DOC | medium |
| PF7 | V&V: NIST Verif 1.1, 2.1 to 2.10, 3.1 to 3.3, 4.1, 5.1 and 5.2; IMO; SFPE examples; fundamental diagrams | https://www.thunderheadeng.com/docs/2026-1/pathfinder/appendices/verification-validation/ | DOC | medium |
| BE1 | buildingEXODUS capabilities: node and arc grid; two FED models; irritant gases; convective and radiative heat; low-visibility smoke with crawling; lifts, escalators, stairs; imports DXF, IFC, SMARTFIRE and FDS geometry; reads CFAST history files | https://fseg.gre.ac.uk/exodus/work.html | DOC | medium |
| BE2 | Occupant interaction with fire conditions in buildingEXODUS (title-level evidence; full text not read) | Gwynne et al. 2001, doi:10.1016/S0379-7112(00)00060-6 | DOC | low |
| BE3 | Agent-based lift model in buildingEXODUS: kinematics, dispatch, bank and car selection, waiting behaviour; 50-floor case | Kinsey, Galea and Lawrence 2011, doi:10.1002/fam.1086 (abstract via Crossref) | DOC | medium |
| BE4 | Irritant-gas capability and CFD coupling (listing only) | Galea et al., Interflam 2004, firedoc.nist.gov/article/V3gzXYQBWEcjUZEYORQn | DOC | low |
| R1 | RiMEA 4.1.1 has tests 1 to 16; Test 1 expects 26 to 34 s over 40 m at 1.33 m/s; Test 5 uses uniform pre-movement between 10 and 100 s; Tests 10, 11 and 14 cover exit allocation, exit choice and route choice over floors | rimea.de, rimea-4.1.1-d-e-1.pdf, pp. 3, 29, 30, 35, 36, 44 | DOC | high |
| N1 | TN 1822 Table 1 lists 17 verification tests in five core components | Ronchi et al. 2013, doi:10.6028/NIST.TN.1822, p. 15 | DOC | high |
| N2 | Verif 2.5 varies the speed (1.0 to 0.25 m/s) and K (10 to 0.5 /m); Verif 2.6 holds an occupant in a 10 × 10 × 3 m room and compares the time to FED = 1 with a hand or fire-model calculation; Verif 2.7 uses a lift between two rooms 3.5 m apart | same, pp. 23-25 | DOC | high |
| I1 | ISO 20414:2020 scope: components for V&V of building evacuation models, accuracy assessment, acceptance criteria, reporting template; full text not accessed | Catalogue entries (iteh.es, sis.se) found by web search | DOC | medium |
| X1 | Studio's stair travel time at 1.2 m/s is 13% to 50% shorter than the references' unimpeded stair speeds (0.6 to 1.05 m/s) | Derived from S1, P4, F11 and the SFPE constants in the methods note (section 6.1) | INF | medium |
| X2 | Tolerance times differ across the tools: at 100 °C, 7.9 min (ISO 13571, Studio), 12.3 min (JPSfire) and 24.7 min (CFAST); at 60 °C, 45, 98 and 156 min | Computed from S5, P2, C3 | INF | high |

## 7. Ranked gaps in Studio Next

The ranking weighs the effect on the numbers Studio reports today against the fit with a browser-first, single-evacuee, exact router. Effort is in weeks of work with human review, calibrated to recent work on this project, not ideal-engineer time.

### G1. Stair and slope walking speed (rank 1)

- **What.** Use a per-link speed from the link's kind and direction. Stairs would use the SFPE speed by riser and going, about 0.85k with k between 1.00 and 1.23 at low density, along the line of travel. Down and up could differ, as in FDS+Evac's factors [F11] and Pathfinder's curves [PF2]. Ramps would use a slope factor. Core stairs without treads use their assumed 175/280 mm geometry, k ≈ 1.08.
- **Impact.** Today stair time is 13% to 50% too short against the references [X1]. Every multi-storey earliest arrival, dose and deadline is optimistic by that share of its stair time. RiMEA 2 and 3 and TN 1822 Verif 2.2 cannot be passed with a distinct stair speed.
- **Effort.** One to two weeks: link times, settings, tests and generalising the Lean statements.
- **Fit.** Good. It changes only link times.
- **Lean.** For each directed link e, τ_e = ℓ_e / v_e with v_e > 0 constant. Then A_e(t) = t + τ_e is FIFO, and time-dependent Dijkstra, the Pareto front and the deadline recursion L(v) = max over (v,w) of min(u(v), L(w) − τ_vw) remain exact. Stating this means replacing len/speed by τ_e in the existing modules. Directed (asymmetric) times need the backward deadline pass to read τ of the forward direction.
- **LG coverage.** Not covered by LG1 to LG8. The methods note (section 6.1 and the model map) proposed it, and it was not built.

### G2. Smoke-reduced walking speed and visibility (rank 2)

- **What.** Make speed depend on the local extinction coefficient K. Candidate correlations are Frantzich and Nilsson's c(K) = max(c_min, 1 − 0.0807 K) [F6, P1] and Fridolf et al.'s data. Add a soot output to the zone model: soot yield, then the upper-layer soot mass, then K = σ_m ρ Y_s.
- **Impact.** At K = 3 /m the factor is 0.76, and at K = 7.5 /m it is 0.39 [F6]. Time in smoke, and so dose, can more than double. Exposure is understated today, and routes through smoke look better than they are.
- **Effort.** Three to five weeks: soot in zone.js, per-node K in the forecast, time-dependent link times and proofs.
- **Fit.** Good, as a single evacuee with time-dependent link costs.
- **Lean.** With K_v(t) kept as a running maximum, the factor c is non-increasing in K and τ_e(t) is non-decreasing in t. Then A_e(t) = t + τ_e(t) is strictly increasing (FIFO), and the dose of a link entered later is no smaller, because rates and durations are both non-decreasing. These are the hypotheses Forecast.lean and Deadline.lean already use, generalised from a constant τ.
- **LG coverage.** Partly. LG8 gives the layer and its untenability criterion, and LG1 gives the time-dependent router; speed reduction is not covered.

### G3. Asphyxiant-gas FED and radiant heat (rank 3)

- **What.** Add a gas FED rate of the Purser or ISO 13571 form: CO and HCN terms multiplied by a CO2 hyperventilation factor, plus a low-O2 term [F7, C2]. Optionally add a radiant-heat term for nodes near the fire or under a hot layer [C3]. The zone model would need CO and HCN yields and O2 depletion in each layer, as CFAST does.
- **Impact.** In many enclosure fires the gas FED reaches 1 before the heat FED does. A heat-only dose can rank a route through a cool but toxic layer as safe. The tools also disagree on the convective tolerance-time constants: ISO 13571 against CFAST is a factor of about 3 at 100 °C [X2]. The choice should be stated wherever doses are shown.
- **Effort.** Three to five weeks: species in the zone model, rates, a TN 1822 Verif 2.6 check against hand calculations, and proofs. The dose machinery is unchanged.
- **Fit.** Good. The dose stays a sum of non-negative rates per node and time.
- **Lean.** Each gas rate term is non-decreasing in CO, HCN and CO2 concentration and non-increasing in O2. With running-maximum concentrations and a running-minimum O2, every node's total rate is non-decreasing in time. Every existing dose result (dominance preserved, waiting never helps, front complete and sound, deadline bracketed) then holds unchanged. Units can be checked in Dim.lean.
- **LG coverage.** Extends LG2, which is convective heat only, and LG8, which has no species.

### G4. Door capacity, queueing and population (rank 4)

- **What.** Add queueing at doors and stairs for a population, at a specific flow of about 1.3 persons per second per metre of effective width on the level [F8, PF1]. There are three options of increasing scope. (a) A per-evacuee queue delay at each door, n_ahead / (F_s W_e), as in FDS+Evac's door choice [F8]. (b) A hydraulic SFPE calculation on the room graph. (c) Flows over time (quickest transshipment) on the room graph, as the methods note proposed.
- **Impact.** This is the largest effect on RSET for any building with more occupants than its doors pass in a few seconds. Today Studio reports a best case for one person.
- **Effort.** (a) Two to three weeks. (b) Three to four weeks. (c) Six to ten weeks, as the methods note estimated.
- **Fit.** (a) and (b) fit a browser tool as estimates; (c) is a separate solver. Agent-based crowds do not fit the browser-first, exact design.
- **Lean.** (a) A FIFO queue gives a delay function d(t) with d'(t) ≥ −1, so A(t) = t + d(t) stays FIFO; this is statable. (b) is algebra over the hydraulic equations. (c) is a large formalisation (flows over time) and is not recommended first.
- **LG coverage.** Not covered. The methods note lists "Many evacuees (absent)".

### G5. Pre-movement distributions and occupant heterogeneity (rank 5)

- **What.** Replace the single "leave after" value with a distribution F_pre (uniform, normal, lognormal and others, as in FDS+Evac [F9] and RiMEA Test 5 [R1]). Add occupant profiles: speed by class [F10, V6], and mobility constraints, for example no stairs for a wheelchair user, using the existing `blocked` edges.
- **Impact.** Pre-movement is usually the largest and most variable part of RSET (methods note, section 6.1). Studio's deadline map already gives the latest safe departure L(v) at every node. What is missing is reporting the probability P(safe from v) = F_pre(L(v)), and the same for each speed class.
- **Effort.** One to two weeks.
- **Fit.** Very good. It composes with the deadline map at no extra search cost.
- **Lean.** If the departure T is independent of the forecast, then P(T < L(v)) = F_pre(L(v)⁻). Because the computed map is never later than the exact deadline (Deadline.lean), the reported probability is a lower bound on the exact one. This is a short, statable corollary.
- **LG coverage.** Partly. The deadline map (C1, from LG1 and LG2) is the dual quantity, and the distribution is new.

### G6. Lift waiting and kinematics (rank 6)

- **What.** Give each lift link a call wait, door open and close times, travel time (rise over car speed plus acceleration terms) and a car capacity [F12, PF4, BE3]. This replaces "ride costs its rise as a walk" [S4].
- **Impact.** Only with `useLifts`. Today a lift ride costs seconds where real waits are tens of seconds, which makes evacuation-lift studies optimistic. TN 1822 Verif 2.7 is not passable today.
- **Effort.** One to two weeks.
- **Fit.** Good. A scheduled link for a single evacuee is straightforward.
- **Lean.** A lift link with a non-decreasing next-departure function D(t) ≥ t and ride time r gives A(t) = D(t) + r. This is non-decreasing (FIFO with waiting), so time-dependent Dijkstra stays exact. Statable.
- **LG coverage.** Not covered.

### G7. Route choice with limited knowledge (rank 7)

- **What.** Studio's earliest-arrival router knows the whole future forecast, as an oracle would. The studied tools restrict choice to known or visible exits [F8], journeys [J4] or familiar exits [BE1], or use perceived costs [T3, P3]. Option: restrict exits to "familiar" ones, or route on the hazard as perceived now (live rerouting already exists), and report the result beside the oracle as a bracket.
- **Impact.** Medium. The oracle is a lower bound on egress time and dose. Users may read it as a prediction.
- **Effort.** Two to three weeks.
- **Fit.** Good.
- **Lean.** For every policy, the arrival of its walk is at least the oracle's earliest safe arrival. This follows from the exactness of `earliestArrival` already proved.
- **LG coverage.** Partly. LG1 gives the oracle, and the bracket and familiarity are new.

### G8. A standard verification harness (rank 8)

- **What.** Synthetic-geometry tests for the RiMEA and TN 1822 cases in section 8, run in Node with the browser code, with ISO 20414-style reporting.
- **Impact.** It does not change results. It makes them comparable and reviewable, and it gives a regression suite for G1 to G7.
- **Effort.** Two to three weeks.
- **Fit.** Good.
- **Lean.** Each test's expected result (for example 40 m at 1.33 m/s arrives in 30.1 s, inside RiMEA's 26 to 34 s) is a closed-form statement about the router on a fixed graph. These are suitable as Lean examples or decidable checks.
- **LG coverage.** Related to LG5 (benchmarks), which covers IFC graph metrics, not behaviour tests.

### G9. Clearance from walls and corners (rank 9)

- **What.** Route only through openings at least one body width wide [M2], and keep paths a radius off walls [PF3, J5].
- **Impact.** Low for a single evacuee on a lattice that already never cuts corners.
- **LG coverage.** Already considered as candidate C6 in the scoop check, and scored low.

### Gaps already covered by LG1 to LG8

The studied tools offer nothing beyond Studio on four counts.

- **Exact time-dependent routing (LG1).** No studied tool has it. FDS+Evac and Pathfinder re-evaluate door choice periodically with heuristic cost estimates [F8, PF3].
- **Proofs (LG3).** No studied tool publishes any.
- **IFC to routing in the browser (LG4).** None of the open tools has it [F13, J9, V9, M7].
- **Surrogates with error bounds (LG7).** None was found.

The study does not change LG5 (graph benchmarks) or LG6 (RL baselines).

## 8. Standard tests: what Studio can run

"Today" means the current engine can compute the expected result on a synthetic graph built in a Node test. "After Gn" names the gap that must be closed first. "Crowd" means the expected result depends on interactions between occupants.

**RiMEA 4.1.1** [R1]

| Test | Title | Studio |
|---|---|---|
| 1 | Walking speed in a corridor (40 m, 26 to 34 s at 1.33 m/s) | Today |
| 2 | Walking speed up stairs (10 m along the slope) | Today only with one global speed; a distinct stair speed after G1 |
| 3 | Walking speed down stairs | As Test 2 |
| 4 | Fundamental diagram | Crowd |
| 5 | Pre-movement time (10 persons, uniform 10 to 100 s) | After G5 (independent evacuees, no crowd needed for the criterion) |
| 6 | Movement around a corner (20 persons) | Today, per evacuee (the criterion is not passing through walls) |
| 7 | Allocation of demographic parameters | After G5 |
| 8 | Parameter study | Partial (speed and fire parameters only) |
| 9 | Crowd leaving a large public room | Crowd |
| 10 | Allocation of escape routes (assigned exits) | Today (the routers take an exit set per start) |
| 11 | Choice of escape route (1,000 persons, congestion) | Crowd |
| 12 | Effect of bottlenecks | Crowd |
| 13 | Fundamental diagram on stairs | Crowd |
| 14 | Choice of route over floors | Today (reports "short" or "long" from travel time; G1 may change the answer) |
| 15 | Large crowd around a corner | Crowd |
| 16 | 1D fundamental diagram | Crowd |

**NIST TN 1822** [N1, N2]. We use TN 1822 as the readable basis; ISO 20414 should be checked against its own text [I1].

| Test | Sub-element | Studio |
|---|---|---|
| Verif 1.1 | Pre-evacuation time distributions | After G5 |
| Verif 2.1 | Speed in a corridor | Today |
| Verif 2.2 | Speed on stairs | After G1 |
| Verif 2.3 | Movement around a corner | Today |
| Verif 2.4 | Assigned demographics | After G5 |
| Verif 2.5 | Reduced visibility and walking speed | After G2 |
| Verif 2.6 | Occupant incapacitation (static FED) | Partial today (convective heat from the zone model); full after G3 |
| Verif 2.7 | Elevator usage | After G6 |
| Verif 2.8 | Horizontal counterflow | Crowd |
| Verif 2.9 | Group behaviours | Crowd |
| Verif 2.10 | People with movement disabilities | Single-evacuee part (reduced speed, stairs blocked) after G5; assistance needs crowd |
| Verif 3.1 | Exit route allocation | Today |
| Verif 3.2 | Social influence | Crowd |
| Verif 3.3 | Affiliation | After G7 (familiar exits) |
| Verif 4.1 | Dynamic availability of exits | Today (closing times in the forecast) |
| Verif 5.1 | Congestion | Crowd |
| Verif 5.2 | Maximum flow rates | Crowd, or after G4(b) for a hydraulic estimate |

## 9. Closed-source tools studied from documentation only

Neither tool was decompiled, disassembled or run. The findings rest on the vendors' public documentation and on peer-reviewed papers.

### Pathfinder (Thunderhead Engineering)

- **Modes.** There are two movement modes [PF1, PF2]. SFPE mode uses the SFPE handbook's density-speed relation in each room. Door flow is metered by a delay timer, rooms are capped at 3.55 persons per m², and occupants do not collide with each other. Steering mode uses inverse steering over candidate directions, density from neighbour spacing (Fruin) and acceleration limits; door flow emerges and can fall below the nominal limit.
- **Stairs.** Speed scales by k from the SFPE riser-and-going table, interpolated by slope [PF1].
- **Route choice.** A* on a navigation mesh, plus a "locally quickest" door cost that includes an estimated queue time, re-evaluated every second [PF3].
- **Lifts.** Call groups, open and close delays and user-set travel times [PF4].
- **FED.** FED follows FDS+Evac with CO, CO2 and O2, but not HCN, and the page does not say whether FED changes movement [PF5].
- **Input and V&V.** IFC import exists since 2018.4 [PF6]. The V&V pages report all 17 TN 1822 tests, IMO tests and SFPE examples [PF7].

### buildingEXODUS (FSEG, University of Greenwich)

- **Capabilities.** The public capability page lists a fine node-and-arc grid, local and global behaviour rules, two FED models, irritant gases, convective and radiative heat, low-visibility smoke with crawling, and lifts and escalators [BE1].
- **Inputs.** It imports DXF, IFC, SMARTFIRE and FDS geometry, and reads CFAST history files and SMARTFIRE output as hazards [BE1].
- **Lift model.** The model (Kinsey, Galea and Lawrence 2011) covers kinematics, dispatch strategies, bank and car selection, and waiting behaviour [BE3].
- **Smoke and toxicity equations.** The smoke-speed and toxicity equations are described in papers we could not read in full here [BE2, BE4]. Their exact form is therefore unresolved in this study.

### What a black-box differential comparison would need

**Prerequisites.**

- A licensed copy of each tool, run by a person entitled to run it, within its licence terms.
- Only inputs and outputs are compared; nothing is reverse engineered from binaries.
- Single-occupant scenarios, or very low densities, so that crowd effects do not mask the comparison; in Pathfinder SFPE mode the first occupant through a door waits for no delay [PF1].
- The same speed, stair geometry and hazard fields in every tool.

**Tests to run in both tools and in Studio.**

| Test | What it compares | Notes |
|---|---|---|
| TN 1822 Verif 2.1, 2.3, 3.1, 4.1 and RiMEA 1, 6, 10, 14 | Speeds, corners, exit allocation, route choice | Studio can run these today |
| TN 1822 Verif 2.2 and RiMEA 2, 3 | Stair speed | After G1, with matched riser and going |
| TN 1822 Verif 2.5 | Smoke speed: 1.0, 0.75, 0.5, 0.25 m/s against K = 10, 7.5, 3, 0.5 /m | After G2. Pathfinder's support for it is not documented [PF5]; buildingEXODUS documents smoke interaction [BE1] |
| TN 1822 Verif 2.6 | Static FED in a 10 × 10 × 3 m room | After G3. Use one imposed gas history in all three tools: a CFAST run of the same room, which buildingEXODUS reads natively [BE1], and the same concentrations as tables for Studio. Restrict Pathfinder to CO, CO2 and O2 [PF5] |
| TN 1822 Verif 2.7 | Lift use | After G6, with the same car kinematics |
| RiMEA 5 and TN 1822 Verif 1.1 | Pre-movement | After G5, by distribution moments |

**IFC models.** Both tools document IFC import [PF6, BE1], so the same files can go to all three tools:

- `more2\bsc-duplex` (small, treads modelled, 4 exterior doors);
- `more2\kit-smiley-west` (5 storeys, 30 stairs, 70 exterior doors);
- `more2\zenodo-taltech-ds3` (Revit, stairs as single stepped solids, one lift);
- `more2\bsc-esplanades` (ramps, a lift and 285 spaces);
- `more2\osb-ifcfiles-hhs-office` (lifts are the only vertical link);
- the existing `AC20-FZK-Haus`.

Each tool's own conversion of the IFC (rooms, doors, stairs) should be saved, because a disagreement may come from geometry rather than movement.

**Outputs to compare.**

- For each start and exit: exit time, path length and the chosen exit.
- Stair and lift time on their own.
- FED at the exit, and the time to FED = 1 in the static test.
- For the IFC models: the number of rooms, doors and stairs each tool recognised, and the exit set.
- Metrics: relative error in exit time and path length, agreement on exit identity, and for stochastic inputs the convergence measures of TN 1822 section 4.1.

Studio's exactness results give a useful asymmetry: on the same graph and speeds, Studio's earliest arrival is a lower bound for any tool that walks the same network, so a tool reporting an earlier exit time is walking a different geometry or speed.

**Open alternatives for the same harness.**

- **JuPedSim (LGPL).** Its WalkableSurface [J6] accepts multi-level geometry, and Studio's lattice could be exported to it for crowd cross-checks (G4).
- **FDS+Evac 6.7.7 (public domain).** It is the documented basis of Pathfinder's FED [PF5]. Both could be run later, when a run is authorised, without licence cost.

## 10. IFC files obtained for testing

Eleven files, about 180 MB, each in its own subfolder of `C:\Users\lmurugesan\ifc-samples\more2\`. URLs, licence evidence, SHA-256 and counts are in `C:\Users\lmurugesan\ifc-samples\more2\SOURCES.md`. None matches the SHA-256 of a file already in the corpus. Counts are plain text counts of entity names; storeys are IfcBuildingStorey entities.

| Subfolder | Licence | Schema, tool | Storeys | IfcStair / Flight | IfcRamp | IfcTransportElement | IfcSpace | IfcDoor | Notable for egress testing |
|---|---|---|---|---|---|---|---|---|---|
| bsc-esplanades | CC BY 4.0 | IFC2X3, ArchiCAD 22 | 7 | 4 / 20 | 6 | 1 | 285 | 211 | Ramps, a lift element, a "LIFT" space; landings exported as flat flights |
| osb-testfiles-hitos | GPL-3.0 (repository licence) | IFC2X3, ArchiCAD 11 | 7 | 14 / 0 | 3 | 3 | 243 | 198 | Large; sloped ramp solids; lifts named "Heis" |
| zenodo-taltech-ds3 | CC BY 4.0 | IFC4, Revit 2023 | 5 | 29 / 3 | 0 | 1 | 115 | 119 | 14 stairs are single solids with a stepped profile; 16 exterior doors |
| zenodo-klassiqua-office-1970 | CC BY 4.0 | IFC4X3_ADD2, Bonsai 0.8.5 | 4 | 0 / 6 | 0 | 0 | 48 | 49 | Flights with no IfcStair parent; IfcOpenShell-authored |
| kit-smiley-west | KIT: "for unrestricted use" | IFC4, ArchiCAD 20 | 5 | 30 / 0 | 0 | 0 | 140 | 170 | 70 exterior doors (many exits) |
| zenodo-balconies-a5 | CC BY 4.0 | IFC2X3, ArchiCAD 17 | 7 | 16 / 0 | 0 | 0 | 686 | 818 | 109 apartments; large door count |
| bsc-duplex | CC BY 4.0 | IFC2X3, Revit 2011 | 4 | 2 / 2 | 0 | 0 | 21 | 14 | NIBS Duplex; treads modelled; 4 exterior doors |
| osb-ifcfiles-hhs-office | CC BY-ND 4.0 | IFC2X3, Revit 2010 | 4 | 0 / 0 | 0 | 0 | 33 | 133 | No stairs; only lifts link floors; every door flagged IsExternal, which tests the exit fallback |
| osb-testfiles-4351 | GPL-3.0 (repository) | IFC2X3, Revit 2011 | 5 | 4 / 3 | 0 | 0 | 6 | 17 | Spiral and quarter-turn stairs; lift only as a proxy plus lift doors |
| osb-testfiles-adt-fzk-haus | GPL-3.0 (repository); KIT model | IFC2X_FINAL, Architectural Desktop | 2 | 1 / 0 | 0 | 0 | 7 | 5 | Only file with Pset_DoorCommon FireExit TRUE; pre-2x3 schema tests graceful failure |
| osb-testfiles-allplan-smiley-west | GPL-3.0 (repository); KIT model | IFC2X_FINAL, Allplan 2003 | 5 | 15 / 0 | 0 | 0 | 70 | 85 | Only Allplan model; six stairs share one GlobalId |

Notes on the licences:

- **CC BY-ND (HHS Office).** It allows local testing but not sharing modified copies.
- **GPL files.** For the four opensourceBIM/TestFiles models the GPL grant is the repository's, and the original authors are not named there.
- **Skipped for unclear licence.** IfcOpenShell/files (no licence file), the Schependomlaan subcontractor models (academic-only wording that conflicts with the repository notice), Autodesk's Revit basic sample project in opensourceBIM/TestFiles, and Zenodo record 7686442 (no licence). These are listed in SOURCES.md.
- **No smooth stair solid.** No file found has stairs drawn as a smooth sloped solid with no steps. TalTech's stepped single solids are the closest case.

## 11. Unknowns, bounded searches and limits

- **No runs.** No tool was run, so no performance figure for any third-party tool is measured here. Time steps are reported instead.
- **Bounded negative claims.** The following absence claims come from text searches of the pinned copies only, and were labelled as such: no IFC in the open tools; no fire coupling in Vadere, Menge and MomenTUM; no lifts in JuPedSim, Vadere, Menge and MomenTUM; no stairs in MomenTUM.
- **Pathfinder.** Pathfinder's equations for base speed, density and door passage did not survive text extraction from its pages. Whether its FED changes occupant movement is not stated [PF5].
- **buildingEXODUS.** The smoke-speed, FED and FIC equations are unresolved here [BE2, BE4]. Reading Gwynne et al. (2001) in full would settle them.
- **ISO 20414.** The text was not accessed [I1].
- **FDS+Evac validation.** The FDS+Evac Technical Reference and User's Guide (VTT) was not read; its IMO verification results are cited only by reference.
- **Effort figures.** These are estimates for this project, not measurements.

## 12. References

DOIs were checked against Crossref on 2026-10-28; items without a DOI are cited by URL.

- Chraibi, M., Seyfried, A., Schadschneider, A. (2010). Generalized centrifugal-force model for pedestrian dynamics. Physical Review E 82, 046111. https://doi.org/10.1103/PhysRevE.82.046111
- Curtis, S., Best, A., Manocha, D. (2016). Menge: A modular framework for simulating crowd movement. Collective Dynamics 1. https://doi.org/10.17815/CD.2016.1
- Fridolf, K., Nilsson, D., Frantzich, H. (2013). The impact of smoke on walking speed. Fire and Materials. https://doi.org/10.1002/fam.2217
- Fridolf, K., et al. (2019). The representation of evacuation movement in smoke-filled underground transportation systems. Tunnelling and Underground Space Technology. https://doi.org/10.1016/j.tust.2019.04.016
- Gwynne, S., Galea, E.R., Owen, M., Lawrence, P.J., Filippidis, L. (1999). A review of the methodologies used in the computer simulation of evacuation from the built environment. Building and Environment 34(6). https://doi.org/10.1016/S0360-1323(98)00057-2
- Gwynne, S., Galea, E.R., Lawrence, P.J., Filippidis, L. (2001). Modelling occupant interaction with fire conditions using the buildingEXODUS evacuation model. Fire Safety Journal 36. https://doi.org/10.1016/S0379-7112(00)00060-6
- Gwynne, S.M.V., Rosenbaum, E.R. (2016). Engineering data. SFPE Handbook of Fire Protection Engineering, 5th ed., ch. 64. https://doi.org/10.1007/978-1-4939-2565-0_64
- Helbing, D., Molnár, P. (1995). Social force model for pedestrian dynamics. Physical Review E 51, 4282. https://doi.org/10.1103/PhysRevE.51.4282
- Kinsey, M.J., Galea, E.R., Lawrence, P.J. (2011). Investigating evacuation lift dispatch strategies using computer modelling. Fire and Materials. https://doi.org/10.1002/fam.1086
- Kleinmeier, B., Zönnchen, B., Gödel, M., Köster, G. (2019). Vadere: An open-source simulation framework to promote interdisciplinary understanding. Collective Dynamics 4. https://doi.org/10.17815/CD.2019.21
- Korhonen, T., Hostikka, S., Heliövaara, S., Ehtamo, H. (2010). FDS+Evac: An agent based fire evacuation model. In Pedestrian and Evacuation Dynamics 2008 (Crossref lists 2009). https://doi.org/10.1007/978-3-642-04504-2_8
- Kuligowski, E.D., Peacock, R.D., Hoskins, B.L. (2010). A review of building evacuation models, 2nd edition. NIST TN 1680. https://doi.org/10.6028/NIST.TN.1680
- Owen, M., Galea, E.R., Lawrence, P.J. (1996). The EXODUS evacuation model applied to building evacuation scenarios. Journal of Fire Protection Engineering 8(2). https://doi.org/10.1177/104239159600800202
- Peacock, R.D., et al. (2015). CFAST, Consolidated Model of Fire Growth and Smoke Transport (Version 7), Volume 1: Technical Reference Guide. NIST TN 1889v1. https://doi.org/10.6028/NIST.TN.1889v1
- Purser, D.A., McAllister, J.L. (2016). Assessment of hazards to occupants from smoke, toxic gases, and heat. SFPE Handbook, 5th ed., ch. 63. https://doi.org/10.1007/978-1-4939-2565-0_63
- Ronchi, E., Kuligowski, E.D., Reneke, P.A., Peacock, R.D., Nilsson, D. (2013). The process of verification and validation of building fire evacuation models. NIST TN 1822. https://doi.org/10.6028/NIST.TN.1822
- Ronchi, E., Nilsson, D. (2013). Fire evacuation in high-rise buildings: a review of human behaviour and modelling research. Fire Science Reviews 2. https://doi.org/10.1186/2193-0414-2-7
- Seitz, M.J., Köster, G. (2012). Natural discretization of pedestrian movement in continuous space. Physical Review E 86, 046108. https://doi.org/10.1103/PhysRevE.86.046108
- Tordeux, A., Chraibi, M., Seyfried, A. (2016). Collision-free speed model for pedestrian dynamics. In Traffic and Granular Flow '15. https://doi.org/10.1007/978-3-319-33482-0_29
- RiMEA e.V. (2025). RiMEA Guideline for Microscopic Evacuation Analysis, version 4.1.1 (CC BY-ND 4.0). https://rimea.de/wp-content/uploads/2025/09/rimea-4.1.1-d-e-1.pdf
- ISO 20414:2020. Fire safety engineering: Verification and validation protocol for building fire evacuation models. Catalogue entry only (full text not accessed).
- Thunderhead Engineering. Pathfinder 2026.1 Technical Reference and Verification and Validation pages. https://www.thunderheadeng.com/docs/2026-1/pathfinder/appendices/technical-reference/ and https://www.thunderheadeng.com/docs/2026-1/pathfinder/appendices/verification-validation/
- FSEG, University of Greenwich. buildingEXODUS capabilities. https://fseg.gre.ac.uk/exodus/work.html
- REA, reverse-engineer-anything (MIT). https://github.com/morluto/rea
