# People leaving: verification report (2026-11-24)

What the people-leaving model (`topologicpy-web-frontend/src/core/crowd.js`) passes, fails and cannot represent of three published test suites. The plan behind the harness is in `crowd-validation-plan-2026-11-11.md`.

**Status in one line.** A partly verified movement model. It is not validated against any real evacuation, and it has no behavioural model.

## The run

- Harness: `node tools/validation/crowd-validation.mjs --out 2026-11-24-crowd-validation.json --date 2026-11-24`.
- Result: 22 pass, 2 fail, 3 documentation. The 16 passes and 2 failures of 2026-11-11 are unchanged; six tests were added the same day, run as written with the start areas, assigned exits and populations the model now has (`groups` in `simulateCrowd`), and all six pass. File: `benchmark/results/2026-11-24-crowd-validation.json`, which records the hash of `crowd.js`.
- Code: main after pull request 19 (the gathering places and the count of people who reach no exit changed since 2026-11-11; the walk did not).
- The harness now runs with the unit tests (`src/core/crowdValidation.test.mjs`): a third failure, or the loss of a pass, fails the suite.

Each test builds its geometry as a lattice of nodes 0.5 m apart and runs the model as the app does. Nothing is tuned to pass.

**What the new option is not.** `groups` is for analyses and these tests. The app's panel does not offer it: there, people still stand over all the floor and choose their own exit. A person given an exit keeps it until a fire cuts their walk to it; then they choose as everyone else does, and the run counts them (`released`).

**The words used below.**
- *Pass (run)*: run by the harness on 2026-11-24 and inside the test's criterion.
- *Pass with a deviation*: run, but not as the test is written; the deviation is named.
- *Fails*: run and outside the criterion, or the model cannot do what the test specifies.
- *Documented*: the test asks for a result to be reported, not a criterion.
- *Not run*: the model could attempt it; the harness does not yet.
- *Not represented*: the model has nothing of the kind.

## The two failures, and their one cause

| Test | Result |
| --- | --- |
| RiMEA 12b | A bottleneck 5 m long and 1 m wide clears a room in 128.8 s; a short door of the same width takes 163.3 s. A corridor should not pass more than a door. |
| RiMEA 16 | Single-file speed against density: 8 of 14 points lie inside the 10 to 90 % band of the experiments. |

Both come from how density is read: round the free node ahead of a walker, so dense flow moves too fast. In RiMEA 4 the speed at 2.4 persons/m² is 0.77 m/s, where the SFPE relation gives 0.49. Only doors carry a flow limit; corridors and stairs have none.

## NIST Technical Note 1822 (Ronchi et al. 2013)

| Test | What it asks | Verdict |
| --- | --- | --- |
| Verif.1.1 | Pre-evacuation times follow the distribution assigned | Pass with a deviation: a uniform stagger only; no other distribution can be assigned |
| Verif.2.1 | Speed in a corridor | Pass (run), along the lattice and at 45 degrees. Sensitivity to the node spacing not shown |
| Verif.2.2 | Speed on a stair | Pass with a deviation: 10 m run scaled to the 100 m of the test |
| Verif.2.3 | Movement round a corner | Pass (run), from the start area |
| Verif.2.4 | Assigned demographics | Pass (run): a normal population (Weidmann's) and a uniform one (IMO's) are each followed (KS p 0.83 and 0.36, 400 persons) |
| Verif.2.5 | Reduced visibility against walking speed | Not represented: smoke does not slow anyone |
| Verif.2.6 | Occupant incapacitation | Not represented in the walk; the doses are computed apart from it |
| Verif.2.7 | Lift use | Not represented: lifts are left out |
| Verif.2.8 | Horizontal counter-flows | Not represented: everyone heads for an exit |
| Verif.2.9 | Group behaviours | Not represented |
| Verif.2.10 | People with movement disabilities | Not represented: no individual speeds or body sizes |
| Verif.3.1 | Exit route allocation | Pass (run): 115 of 115 leave by the exit given to them. The nearest choice is deterministic too (115 of 115 to their closest exit) |
| Verif.3.2 | Social influence | Not represented |
| Verif.3.3 | Affiliation | Not represented |
| Verif.4.1 | Dynamic availability of exits | Pass (run) as written: one occupant at 1 m/s in a 10 m x 15 m room, Exit 1 closed at 1 s; Exit 1 is not used, from a start nearer Exit 1 too. Also with 240 people and the exit closing at 10 s |
| Verif.5.1 | Congestion | Not run |
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
| 12b | Effect of a bottleneck's length | Fails (run) |
| 12c | Congestion at a second bottleneck | Not run |
| 12d | Flow against a bottleneck's width | Pass (run) on the stated aim: flow rises with width. At the SFPE design values, about half the measured flows |
| 13 | Stairs up against stairs down | Pass (run) on direction: 0.69 m/s up, 0.92 m/s down |
| 14 | Route choice between floors | Not run |
| 15 | Movement of a large crowd round a corner | Not run |
| 16 | Single-file fundamental diagram | Fails (run) |

## IMO MSC.1/Circ.1533

No test of this suite is run under its own name. The verdicts follow from the runs above.

| Test | Verdict |
| --- | --- |
| 1 | Pass (run as NIST Verif.2.1) |
| 2, 3 | Not run at the suite's 10 m; the stair speeds pass under RiMEA 2 and 3 |
| 4, 5, 6 | Pass (run as NIST Verif.5.2, 1.1 and 2.3; 5 with its deviation) |
| 7 | Pass (run): 400 males of 30 to 50 years, uniform 0.97 to 1.62 m/s |
| 8 | Not represented: no counter-flow |
| 9 | Pass (run as RiMEA 9) |
| 10 | Pass (run as RiMEA 10) |
| 11 | Not run |
| 12 | Not run under this suite |

## What is not claimed

- Validated, or calibrated, against people. No comparison with a recorded evacuation has been made.
- "Passes RiMEA" or "passes NIST TN 1822". It passes the tests listed, several with deviations.
- Conformity with ISO 20414, which was not read.
- An escape time from the animation. The performance check, not the animation, gives times, and it adds detection, alarm and pre-movement.
- Dense crowds, or the time to clear a stair. In the app a stair is one file of nodes whatever its width.

## Next, in order

1. Done on 2026-11-24: start areas, exits assigned to people, and populations with their own speeds; NIST Verif.4.1 and RiMEA 11 run as written. The populations built in are IMO MSC.1/Circ.1533's twelve groups by age and sex (Table 3.4, uniform) with its passengers' composition (Table 3.1), RiMEA's persons with impaired mobility, and Weidmann's normal distribution. IMO's stair speeds by group (Table 3.5) are not used: stairs take the SFPE values for everyone.
2. A flow limit for corridors and stairs, with the stair's real width (RiMEA 12b, NIST Verif.5.1).
3. The density reading, calibrated on the single-file data with other data held back (RiMEA 16, RiMEA 4).
4. Pre-movement times on each person and a speed in smoke (NIST Verif.1.1, 2.5).

## Sources

- RiMEA e.V., Guideline for Microscopic Evacuation Analysis, version 4.1.1 of 11.09.2025, https://rimea.de (CC BY-ND 4.0).
- Ronchi E., Kuligowski E.D., Reneke P.A., Peacock R.D., Nilsson D. (2013). The Process of Verification and Validation of Building Fire Evacuation Models. NIST Technical Note 1822. doi:10.6028/NIST.TN.1822.
- International Maritime Organization (2016). MSC.1/Circ.1533, Revised guidelines on evacuation analysis for new and existing passenger ships.
