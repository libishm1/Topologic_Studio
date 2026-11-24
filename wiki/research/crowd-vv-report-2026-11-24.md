# People leaving: verification report (2026-11-24)

What the people-leaving model (`topologicpy-web-frontend/src/core/crowd.js`) passes, fails and cannot represent of three published test suites. The plan behind the harness is in `crowd-validation-plan-2026-11-11.md`.

**Status in one line.** A partly verified movement model. It is not validated against any real evacuation, and it has no behavioural model.

## The run

- Harness: `node tools/validation/crowd-validation.mjs --out 2026-11-24-crowd-validation.json --date 2026-11-24`.
- Result: 22 pass, 2 fail, 3 documentation. The 16 passes and 2 failures of 2026-11-11 are unchanged; six tests were added the same day, run as written with the start areas, assigned exits and populations the model now has (`groups` in `simulateCrowd`), and all six pass. File: `benchmark/results/2026-11-24-crowd-validation.json`, which records the hash of `crowd.js`.
- Code: main after pull request 19 (the gathering places and the count of people who reach no exit changed since 2026-11-11; the walk did not).
- Later the same day, on the branch `feature/crowd-premovement-smoke`: 25 pass, 2 fail, 3 documentation. Three tests were added with two new options, both off unless asked for: a pre-movement time for each person (`preMovement`, `alarm`) and a walking speed in smoke (`smoke`, `smokeSpeed`). The 22 passes and 2 failures above are unchanged, and runs without the new options are the same to the last digit. File: `benchmark/results/2026-11-24-crowd-validation-premovement-smoke.json`.
- With both sets of changes together (main after pull requests 22 and 23): 25 pass, 4 fail, 4 documentation by default, and 28 pass, 1 fail with `passageFlow`. The four default failures are RiMEA 12b and 16 and the two cases added with the flow capacity (NIST Verif.5.1 and the SFPE flows), which pass with it on.
- The harness now runs with the unit tests (`src/core/crowdValidation.test.mjs`): a new failure, or the loss of a pass, fails the suite. Since the update below it runs twice there, by default and with `passageFlow`.

Each test builds its geometry as a lattice of nodes 0.5 m apart and runs the model as the app does. Nothing is tuned to pass.

**What the new option is not.** `groups` is for analyses and these tests. The app's panel does not offer it: there, people still stand over all the floor and choose their own exit. A person given an exit keeps it until a fire cuts their walk to it; then they choose as everyone else does, and the run counts them (`released`).

**The words used below.**
- *Pass (run)*: run by the harness on 2026-11-24 and inside the test's criterion.
- *Pass with a deviation*: run, but not as the test is written; the deviation is named.
- *Fails*: run and outside the criterion, or the model cannot do what the test specifies.
- *Documented*: the test asks for a result to be reported, not a criterion.
- *Not run*: the model could attempt it; the harness does not yet.
- *Not represented*: the model has nothing of the kind.

## Update, later the same day: a flow capacity for corridors and stairs

`simulateCrowd` has a new option, `passageFlow`. It is off by default, in the function and in the app (the panel's "Corridors and stairs limit the flow"), because it changes the times wherever a corridor or a stair is narrow. With it off a run is exactly as before: the 27 earlier cases give the same output to the last digit.

What the option does:
- **Corridors.** Floor up to 3 m across passes at most the SFPE specific flow, 1.32 persons/s for each metre of effective width, the width less 0.2 m a side. The cross-section is found on the graph: the shorter run of floor nodes through a node, wall to wall. Its width is its nodes times the spacing, so it is known to about half a metre; a single file of nodes is taken as two nodes wide (1 m). `passageWidths` gives real widths. Corridors askew to the lattice are not found.
- **Stairs.** The graph now carries each stair node's real width (`graph.widths`: a tread's long side, or an assumed stair's flight). With the option a stair passes at most 1.01 persons/s for each metre of effective width (0.15 m a side). That value is not a new constant: it is what the code's stair speed gives, (1 - 0.266 D) k D at D = 1.9 with k = 1.08 m/s, and it equals the Handbook's tabulated maximum for 7 in risers and 11 in treads. Up a stair it is 0.75 of that, as the speed is. The stair is still one file of nodes, but a node holds one person abreast for each 0.6 m of width, and the stair fills to 1.9 persons/m², not beyond.
- **The queue.** Within 2 m of a corridor's mouth or of a stair with a width the density that slows a step is capped at 1.9 persons/m², as it already is before a door, so the passage's capacity and not the crawl into the queue sets the flow.

The harness has three new cases (RiMEA 12c, NIST Verif.5.1, and the SFPE flows of a door, a corridor and a stair) and a flag, `--passage-flow`, that runs every case with the option. Results: `benchmark/results/2026-11-24-crowd-capacity-default.json` and `2026-11-24-crowd-capacity-passage-flow.json`.

| | By default | With `passageFlow` |
| --- | --- | --- |
| Pass, fail, documentation | 22, 4, 4 | 25, 1, 4 |
| Fails | RiMEA 12b, RiMEA 16, NIST Verif.5.1, SFPE flows | RiMEA 16 |
| RiMEA 12b: room 1 through a short door, then a 5 m corridor, both 1 m wide | 163.3 s, 128.8 s | 163.3 s, 196.6 s |
| 1 m door, corridor, stair (SFPE 0.92, 0.79, 0.71 persons/s) | 0.921, 1.188, 0.542 | 0.921, 0.789, 0.707 |
| Stairs 1.2 m and 2.4 m wide, down (SFPE 0.913, 2.131) | 0.542, 0.542 | 0.913, 2.130 |
| NIST Verif.5.1: flow past the corridor's middle; on to the stair (SFPE 2.105; 1.725) | 0.983 (unsteady); 0.483 | 1.957 (steady); 1.707 |
| NIST Verif.5.1: last person out | 213.7 s | 79.9 s |
| RiMEA 16: points inside the band | 8 of 14 | 10 of 14 |

Every other case gives the same numbers either way, but for NIST Verif.3.1, where one person of 115 takes the other exit with the quickest choice.

What the table shows without the option: a stair drawn as one file of nodes passes 0.54 persons/s whatever its width, less than the SFPE value for any stair over about 0.85 m wide, and a narrow corridor passes half as much again as SFPE allows. So times through narrow corridors are optimistic by default and times down stairs are pessimistic.

A fault found afterwards on the example buildings, and its fix. With `passageFlow` people were left standing for good: 94 of 196 in the Vienna escape route model, 48 of 481 in the NIBS office, 11 of 325 in Smiley West, none without the option. The gap kept before a walker on a stair with a width counted everyone within the gap of the next node, the people behind the walker too, against a number that is smaller near a flight's ends than in its middle. Once a stair had filled while its foot was held up (a door or a landing that takes less than the stair brings), the foremost could not step on for the people behind, and they could not clear before the foremost moved. The gap is now kept ahead only: the people on the next node and on the stair nodes near it that are nearer the walker's exit than the walker. After the fix everyone leaves in all six example buildings that have an exit, with the option as without. The numbers in the table above move a little with it: the 1 m stair passes 0.705 persons/s (0.676 with a node a tread), and the last person of NIST Verif.5.1 is out at 80.7 s. The result files of this date were written before the fix.

Limits of the option, as measured:
- A 1 m stair whose nodes are a tread apart (0.33 m along the slope, as a modelled stair's are) passes 0.662 persons/s, 7 % under the SFPE value; at 0.5 m between nodes it passes 0.707. Wider stairs reach their capacity at either spacing.
- The SFPE flows are design values. Measured laboratory flows through bottlenecks are about twice as high (RiMEA 12d).
- People abreast on a stair node are drawn at the same point in the view.
- RiMEA 16 gains two points with the option only because a single file of nodes is capped as a 1 m corridor. The density reading is not changed by it.

## Update, later the same day: the density reading, calibrated and not adopted

The rules, the runs and the results are in `crowd-calibration-2026-11-24.md`. In short:
- Of 36 ways of reading the density, chosen in advance, the one that fits RiMEA 16's single-file data best counts the others within 2 m of the node ahead over the floor they could stand on. RiMEA 16 then has 14 of 14 points in the band (8 before).
- On the data held back it failed two of its three rules. RiMEA 4 at 2.27 persons/m²: 0.702 m/s, 26 % above the SFPE relation (the rule allowed 15 %; before, 56 % above at 2.44 persons/m²). And the queue before a corridor or a stair crawls again: a 1 m stair passes 0.534 persons/s with `passageFlow` (0.707 with the default reading) and a stair without a capacity 0.153.
- So the default reading is unchanged, and RiMEA 16 still fails by default. The fitted reading is in the code as `densityReading: "calibrated"`, for analysis only. It is calibrated on single-file data and not validated.

## The two failures, and their one cause

| Test | Result |
| --- | --- |
| RiMEA 12b | A bottleneck 5 m long and 1 m wide clears a room in 128.8 s; a short door of the same width takes 163.3 s. A corridor should not pass more than a door. |
| RiMEA 16 | Single-file speed against density: 8 of 14 points lie inside the 10 to 90 % band of the experiments. |

Both come from how density is read: round the free node ahead of a walker, so dense flow moves too fast. In RiMEA 4 the speed at 2.4 persons/m² is 0.77 m/s, where the SFPE relation gives 0.49. Only doors carry a flow limit; corridors and stairs have none.

## NIST Technical Note 1822 (Ronchi et al. 2013)

| Test | What it asks | Verdict |
| --- | --- | --- |
| Verif.1.1 | Pre-evacuation times follow the distribution assigned | Pass (run) as written, for each form the model takes: uniform, log-normal, log-normal cut off, normal cut off and Weibull, ten persons in the 8 m x 5 m room, 30 runs each, the alarm at 30 s. No one moves before the time drawn, all first moves lie in range, and they follow the distribution (KS p 0.85, 0.39, 0.54, 0.36, 0.85; 300 persons each). 2 of 1,500 first moves came later than drawn, by 0.37 s at most: the way was taken. The distributions' numbers are test inputs; none is built in |
| Verif.2.1 | Speed in a corridor | Pass (run), along the lattice and at 45 degrees. Sensitivity to the node spacing not shown |
| Verif.2.2 | Speed on a stair | Pass with a deviation: 10 m run scaled to the 100 m of the test |
| Verif.2.3 | Movement round a corner | Pass (run), from the start area |
| Verif.2.4 | Assigned demographics | Pass (run): a normal population (Weidmann's) and a uniform one (IMO's) are each followed (KS p 0.83 and 0.36, 400 persons) |
| Verif.2.5 | Reduced visibility against walking speed | Pass (run) as written for the one relation built in, FDS+Evac's use of Frantzich and Nilsson's fit: one occupant at 1.25 m/s along the 2 m x 100 m corridor at K = 1.0 /m walks at 1.1491 m/s, the relation's value (100.5 m in 87.46 s; 80.40 s clear). Over the suggested range (1.25 to 0.25 m/s, K 0.5 to 10 /m) 25 of 25 match the relation as the model applies it; in 4 of them, slow walkers in the thickest smoke, the model's least step speed of 0.15 m/s governs, above the relation's 0.1 of the person's speed. Off unless asked for. Irritancy, turning back and crawling are not represented |
| Verif.2.6 | Occupant incapacitation | Not represented in the walk; the doses are computed apart from it |
| Verif.2.7 | Lift use | Not represented: lifts are left out |
| Verif.2.8 | Horizontal counter-flows | Not represented: everyone heads for an exit |
| Verif.2.9 | Group behaviours | Not represented |
| Verif.2.10 | People with movement disabilities | Not represented: no individual speeds or body sizes |
| Verif.3.1 | Exit route allocation | Pass (run): 115 of 115 leave by the exit given to them. The nearest choice is deterministic too (115 of 115 to their closest exit) |
| Verif.3.2 | Social influence | Not represented |
| Verif.3.3 | Affiliation | Not represented |
| Verif.4.1 | Dynamic availability of exits | Pass (run) as written: one occupant at 1 m/s in a 10 m x 15 m room, Exit 1 closed at 1 s; Exit 1 is not used, from a start nearer Exit 1 too. Also with 240 people and the exit closing at 10 s |
| Verif.5.1 | Congestion | Fails by default; pass (run) with `passageFlow`: a queue before the room's exit (3.8 persons/m²), a steady 1.96 persons/s in the corridor, a queue at the stair's base (peak 2.3 persons/m²), the stair taking 1.71 persons/s. Deviations: the stair taken as 2 m wide; "congestion" read as a density above 1.9 persons/m² |
| Verif.5.2 | Maximum flow rates | Pass (run): 0.92 persons/s through a 1 m exit, under the 1.33 limit |

## RiMEA guideline, version 4.1.1

| Test | What it asks | Verdict |
| --- | --- | --- |
| 1 | Speed in a corridor | Pass (run) |
| 2 | Speed up a stair | Pass (run) |
| 3 | Speed down a stair | Pass (run) |
| 4 | Fundamental diagram | Documented: too fast above about 1.2 persons/m²; the lattice cannot reach 5 persons/m² |
| 5 | Reaction time | Pass (run): the stagger follows the uniform distribution set |
| 6 | Movement round a corner | Pass (run), from the 6 m x 2 m start area |
| 7 | Allocation of demographic parameters | Pass (run) for one normal population (mean 1.34 m/s, sd 0.26). RiMEA 4.1.1 gives its speeds by age as a curve only (Fig. 3, after Weidmann); no table is printed, so none is built in. Its one tabulated group, persons with impaired mobility (0.46 to 0.76 m/s, Table 2), is |
| 8 | Parameter analysis | Not run |
| 9 | Crowd leaving a large public room | Documented: two doors take 1.99 times as long as four |
| 10 | Allocation of escape routes | Pass (run): 23 persons in twelve rooms, each room given its exit; all leave by it. Two a room, one in room 3, as IMO's figure has them |
| 11 | Choice of escape route | Pass (run) as written, narrowly: 1,000 adults packed from the left at 4 persons/m², the most the lattice holds; 50.5 % leave by the closer exit 1, with a long queue there, the rest by exit 2. "Prefer" is met by half a per cent |
| 12a | Effect of a bottleneck's position | Documented: flat |
| 12b | Effect of a bottleneck's length | Fails by default; pass (run) with `passageFlow`: 196.6 s through the corridor against 163.3 s through the door |
| 12c | Influence of congestion (three rooms, two identical bottlenecks) | Documented: 0.921 and 0.917 persons/s through bottlenecks 1 and 2; at most 12 persons in room 2; no queue before bottleneck 2 (longest wait 0.7 s). Rooms taken as 10 m x 10 m |
| 12d | Flow against a bottleneck's width | Pass (run) on the stated aim: flow rises with width. At the SFPE design values, about half the measured flows |
| 13 | Stairs up against stairs down | Pass (run) on direction: 0.69 m/s up, 0.92 m/s down |
| 14 | Route choice between floors | Not run |
| 15 | Movement of a large crowd round a corner | Not run |
| 16 | Single-file fundamental diagram | Fails (run): 8 of 14 points in the band by default, 10 of 14 with `passageFlow` |

## IMO MSC.1/Circ.1533

No test of this suite is run under its own name. The verdicts follow from the runs above.

| Test | Verdict |
| --- | --- |
| 1 | Pass (run as NIST Verif.2.1) |
| 2, 3 | Not run at the suite's 10 m; the stair speeds pass under RiMEA 2 and 3 |
| 4, 5, 6 | Pass (run as NIST Verif.5.2, 1.1 and 2.3; 5 as RiMEA 5 and with the uniform distribution of Verif.1.1) |
| 7 | Pass (run): 400 males of 30 to 50 years, uniform 0.97 to 1.62 m/s |
| 8 | Not represented: no counter-flow |
| 9 | Pass (run as RiMEA 9) |
| 10 | Pass (run as RiMEA 10) |
| 11 | Run as NIST Verif.5.1, with the stair leading down as NIST has it, not up as IMO does |
| 12 | Not run under this suite |

## What is not claimed

- Validated, or calibrated, against people. No comparison with a recorded evacuation has been made.
- "Passes RiMEA" or "passes NIST TN 1822". It passes the tests listed, several with deviations.
- Conformity with ISO 20414, which was not read.
- An escape time from the animation. The performance check, not the animation, gives times, and it adds detection, alarm and pre-movement. In the app the animation still starts everyone within a short stagger.
- A validated speed in smoke. The relation is FDS+Evac's straight line through Frantzich and Nilsson's tunnel data: artificial smoke with acetic acid as an irritant, extinction coefficients of about 2 to 8 /m, means only. In thinner smoke it is an extrapolation, and it says nothing of non-irritant smoke.
- Dense crowds, or the time to clear a stair. In the app a stair is one file of nodes; its real width counts only with `passageFlow`, which is off by default.

## Next, in order

1. Done on 2026-11-24: start areas, exits assigned to people, and populations with their own speeds; NIST Verif.4.1 and RiMEA 11 run as written. The populations built in are IMO MSC.1/Circ.1533's twelve groups by age and sex (Table 3.4, uniform) with its passengers' composition (Table 3.1), RiMEA's persons with impaired mobility, and Weidmann's normal distribution. IMO's stair speeds by group (Table 3.5) are not used: stairs take the SFPE values for everyone.
2. Done on 2026-11-24 as an option, off by default: a flow limit for corridors and stairs, with the stair's real width (RiMEA 12b, NIST Verif.5.1). Before it becomes the default: check the stair widths the graph carries on real models, and decide what a corridor's width counted in nodes is worth.
3. Tried on 2026-11-24 and not adopted: the density reading, calibrated on the single-file data with other data held back (RiMEA 16, RiMEA 4). It fits the single file and fails the held-out rules. Next: a cap on the queue's density before every passage, and two-dimensional data to calibrate on (`crowd-calibration-2026-11-24.md`, section 7.4).
4. Done on 2026-11-24 as options, off by default: pre-movement times on each person and a speed in smoke (NIST Verif.1.1, 2.5). Left open: no pre-movement distribution is built in (the performance check's design values and measured clusters can be passed in, as their sources stand there); the app's panel does not offer pre-movement; the least step speed of 0.15 m/s should not lift a walker above their own speed in smoke; exits are chosen as if smoke slowed no one; Jin's curves and the recommendations of Fridolf et al. were not read and are not built in.

## Sources

- RiMEA e.V., Guideline for Microscopic Evacuation Analysis, version 4.1.1 of 11.09.2025, https://rimea.de (CC BY-ND 4.0).
- Ronchi E., Kuligowski E.D., Reneke P.A., Peacock R.D., Nilsson D. (2013). The Process of Verification and Validation of Building Fire Evacuation Models. NIST Technical Note 1822. doi:10.6028/NIST.TN.1822.
- Korhonen T., Hostikka S. (2009). Fire Dynamics Simulator with Evacuation: FDS+Evac, Technical Reference and User's Guide. VTT Working Papers 119. Eq. 10, p. 22: the walking speed in smoke, alpha 0.706 m/s, beta -0.057 m2/s, least speed 0.1 of the person's own (read; the same constants in `evac.f90` of FDS 6.7.5). Its source, Frantzich H., Nilsson D. (2003), Utrymning genom tät rök: beteende och förflyttning, Report 3126, Lund University, was not read.
- Ronchi E., Gwynne S.M.V., Purser D.A. (2011). The impact of default settings on evacuation model results: a study of visibility conditions vs occupant walking speeds. Advanced Research Workshop, Evacuation and Human Behaviour in Emergency Situations, Santander (read: how the Frantzich and Nilsson experiments were made, and that FDS+Evac uses their means).
- International Maritime Organization (2016). MSC.1/Circ.1533, Revised guidelines on evacuation analysis for new and existing passenger ships.
