# Open items register

Reviewed each week. Last review: 2026-11-25 (the first). Next: 2026-12-02.

An item is closed only when its closing evidence exists (a merged PR, a plan section, a result file). "Parked" means a decision not to do it now, with the reason; parked items do not count as closed. The "Blocks" column names rows of `claims-traceability.md`.

## Closure rate

| Week ending | Open at start | Opened | Closed | Open at end | Closed of all handled |
| --- | --- | --- | --- | --- | --- |
| 2026-11-25 | n/a (first count) | 32 | 13 | 19 | 41 % (item 19 has a PR open, not merged) |

## Open

| # | Item | Area | Opened | Needs | Blocks |
| --- | --- | --- | --- | --- | --- |
| 1 | Heat and smoke arrive late far from the fire along corridors (NBS, ATF) | solver | 2026-11-24 | the user's go, then a pre-registered item | F8, F9 |
| 2 | CO yield is fixed; far too low when the fire is short of air | solver | 2026-11-25 | a proposal fixed before any run | F12 |
| 3 | No data set with smoke and gas measured at 1.5 to 2.0 m | data | 2026-11-24 | a literature search | F13 |
| 4 | Gas-radiation comparisons of smoke and gases on 0.34 m (plan 16.59) | calibration | 2026-11-24 | the storing run to finish | V2 |
| 5 | Gas radiation on 0.20 m: 87 hours against a 72 hour limit | calibration | 2026-11-25 | the user's word, or a faster step | V2 |
| 6 | Cause of the FSE ISOHept26 stop (a gas cell over 3000 K) | solver | 2026-11-24 | a run that stores the hottest cell | V2 |
| 7 | The suite's dead-channel rule drops a real near-zero O₂ reading | calibration | 2026-11-25 | a rule change proposed before a rerun | F12 |
| 8 | Sprinkler leave-one-laboratory-out not rerun on the current solver | calibration | 2026-11-24 | one rerun | F14 |
| 9 | GPU against CPU agreement last measured before 2026-11-12 | solver | 2026-11-12 | one comparison run | every F row, in a browser |
| 10 | RiMEA 16 fails: dense single-file flow too fast | people | 2026-11-24 | a queue-density cap and two-dimensional data | P1, P4 |
| 11 | No comparison with a real evacuation | people | 2026-11-24 | a data set and a protocol | P5 |
| 12 | Pre-movement times and groups are not in the app panel | people | 2026-11-24 | interface work | none |
| 13 | KIT Smiley West: stair flights are cross-linked | people | 2026-11-24 | a fix in the stair graph | P2 |
| 14 | Whether an elevated exit with no way down counts as final (Vienna) | people | 2026-11-24 | the user's decision | P2 |
| 15 | Guide footers say "version of 2026-11-12" | docs | 2026-11-24 | a text change | none |
| 16 | PR 10 (RL card) open since 2026-11-15 | repo | 2026-11-15 | the user's decision | none |
| 17 | `burnoutHeat` default; item 15, second attempt | solver | 2026-11-15 | the user's decision | F15 |
| 18 | A timing test fails only when the machine is loaded | tests | 2026-11-24 | a looser or isolated test | none |
| 19 | No stored result file for the people model at 28 of 29 | people | 2026-11-25 | **PR 26 to be merged** (the run is done: 28 pass, 1 fail; 25 pass, 4 fail with capacity off) | P1 |

## Closed this week

| Item | Closed | Evidence |
| --- | --- | --- |
| Guide inconsistencies from an outside review | 2026-11-24 | PR 15 |
| Stale text inside the app | 2026-11-24 | PR 16 |
| Dose-limit setting not applied to standing doses | 2026-11-24 | PR 17 |
| People drawn back inside the building after leaving | 2026-11-24 | PR 18 |
| Occupants who reach no exit not reported; upper exits | 2026-11-24 | PR 19 |
| People model had no verification harness in the tests | 2026-11-24 | PR 20 |
| A person keeps an exit the fire has cut; test deviations; populations | 2026-11-24 | PR 21 |
| No pre-movement time or smoke speed | 2026-11-24 | PR 22 |
| No corridor or stair capacity | 2026-11-24 | PR 23 |
| Stair deadlock with capacity on | 2026-11-24 | PR 24, browser session of 2026-11-24 |
| Clock ran past its total; smoke toggle shown without a fire | 2026-11-24 | PR 25 |
| Time to 60 °C and heat dose never compared | 2026-11-24 | plan 16.56 |
| The 0.20 m step; the four smoke decisions | 2026-11-25 | plan 16.55, 16.57, 16.58 |

## Review, each week

1. Close what has evidence: move its row to "Closed this week" and add a line to the closure table.
2. Add what was found in the week, with the claim it blocks.
3. For every item open more than four weeks: do it, park it with a reason, or drop it.
